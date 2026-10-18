//! `gate coverage`: the extraction-coverage gate.
//!
//! Unit: every non-test `fn` item in a tracked `src/**/*.rs` file (free functions, `impl` and
//! `trait` methods; nested fns and macro bodies are not items). Levels, counted separately:
//!
//! | level | holds when |
//! |---|---|
//! | extracted | `proof/ApeironWord.lean` has a `def` named for the fn under its extraction module |
//! | refined | a Lean file other than `ApeironWord.lean` names that `def` |
//!
//! A fn that is not extracted must be waived (`docs/contracts/extraction.toml` `[[waiver]]`, with a
//! closed class and a reason) or carried by its file's `[[debt]]` row. A debt is a ratchet: it
//! refuses growth and refuses a stale (too high) count, so the list only shrinks. The gate refuses
//! to report success on nothing: zero fns, zero Lean defs or zero extracted fns is a refusal.
//!
//! Limits (stated, not hidden): the Rust scanner is lexical, not a parser; a method matches on its
//! `impl` type and name, so two same-named methods of one type in one module share a verdict;
//! "refined" is a name reference in a Lean file, not a checked theorem about the def.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use crate::cli::{Verdict, EXIT_ACCEPTED, EXIT_ENVIRONMENT, EXIT_REFUSED};

mod scan;
use scan::{scan_items, Item};

pub(crate) const USAGE: &str =
    "usage: gate coverage (--check | --report | --baseline) [--root DIR]\n  \
  --check     refuse an unextracted, unwaived fn outside its file's debt row\n  \
  --report    per-directory table of fns, extracted, refined, waived, debt\n  \
  --baseline  print the [[debt]] rows for the current tree";

const CLASSES: [&str; 6] = [
    "ffi",
    "simd-twin",
    "gpu",
    "io",
    "unsafe-container",
    "test-support",
];

#[derive(Debug, Clone)]
pub(crate) struct Waiver {
    pub path: String,
    pub name: String,
    pub class: String,
    pub reason: String,
    pub twin: String,
}

#[derive(Debug, Clone)]
pub(crate) struct Debt {
    pub path: String,
    pub unextracted: usize,
    pub row: String,
}

/// One source file's items and the extraction module that mirrors it, if any.
pub(crate) struct SrcFile {
    pub path: String,
    pub module: Option<String>,
    pub items: Vec<Item>,
}

pub(crate) struct Inputs {
    pub files: Vec<SrcFile>,
    pub lean_defs: BTreeSet<String>,
    pub corpus_tokens: BTreeSet<String>,
    pub waivers: Vec<Waiver>,
    pub debts: Vec<Debt>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Refusal {
    pub rule: &'static str,
    pub path: String,
    pub detail: String,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Counts {
    pub fns: usize,
    pub extracted: usize,
    pub refined: usize,
    pub waived: usize,
    pub gap: usize,
}

/// Lean def names that match a fn: `M.f` (and `M.f.<loop>`) for a free fn; `M.T.…f` for a method.
fn def_for(f: &SrcFile, it: &Item, defs: &BTreeSet<String>) -> Option<String> {
    let m = f.module.as_ref()?;
    match &it.ty {
        None => {
            let exact = format!("{m}.{}", it.name);
            let pre = format!("{exact}.");
            if defs.contains(&exact) {
                return Some(exact);
            }
            defs.range(pre.clone()..)
                .next()
                .filter(|d| d.starts_with(&pre))
                .cloned()
        }
        Some(t) => {
            // A child module may `impl` its parent's type; the extraction names the method under
            // the module that defines the type, so try the module and each ancestor.
            let mut segs: Vec<&str> = m.split('.').collect();
            loop {
                let pre = format!("{}.{t}.", segs.join("."));
                let hit = defs
                    .range(pre.clone()..)
                    .take_while(|d| d.starts_with(&pre))
                    .find(|d| d.split('.').any(|s| s == it.name))
                    .cloned();
                if hit.is_some() || segs.len() <= 1 {
                    return hit;
                }
                segs.pop();
            }
        }
    }
}

fn refined(def: &str, tokens: &BTreeSet<String>) -> bool {
    tokens.contains(def) || tokens.contains(&format!("apeiron_word.{def}"))
}

fn waiver_for<'a>(path: &str, it: &Item, ws: &'a [Waiver]) -> Option<&'a Waiver> {
    ws.iter()
        .find(|w| w.path == path && (w.name == "*" || w.name == it.name))
}

/// The decision. Pure over `Inputs`, so every rule has a synthetic control.
///
/// Cost: time O(F log D) over F fns and D Lean defs (range lookups), space O(F).
pub(crate) fn verdict(inp: &Inputs) -> (Vec<Refusal>, Counts, BTreeMap<String, Counts>) {
    let mut out = Vec::new();
    let mut total = Counts::default();
    let mut per_file: BTreeMap<String, Counts> = BTreeMap::new();
    let mut gap_names: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut extracted_by_path: BTreeMap<(String, String), bool> = BTreeMap::new();

    for f in &inp.files {
        let c = per_file.entry(f.path.clone()).or_default();
        for it in &f.items {
            c.fns += 1;
            let def = def_for(f, it, &inp.lean_defs);
            extracted_by_path.insert((f.path.clone(), it.name.clone()), def.is_some());
            if let Some(d) = &def {
                c.extracted += 1;
                if refined(d, &inp.corpus_tokens) {
                    c.refined += 1;
                }
            } else if waiver_for(&f.path, it, &inp.waivers).is_some() {
                c.waived += 1;
            } else {
                c.gap += 1;
                gap_names
                    .entry(f.path.clone())
                    .or_default()
                    .push(it.name.clone());
            }
        }
        total.fns += c.fns;
        total.extracted += c.extracted;
        total.refined += c.refined;
        total.waived += c.waived;
        total.gap += c.gap;
    }

    if total.fns == 0 {
        out.push(r(
            "vacuous",
            "-",
            "no fn items scanned; refusing success on nothing",
        ));
    }
    if inp.lean_defs.is_empty() {
        out.push(r("vacuous", "-", "no Lean defs read from the extraction"));
    }
    if total.fns > 0 && !inp.lean_defs.is_empty() && total.extracted == 0 {
        out.push(r(
            "vacuous",
            "-",
            "0 fns matched an extracted def; the matcher or the module map is broken",
        ));
    }

    let debt: BTreeMap<&str, &Debt> = inp.debts.iter().map(|d| (d.path.as_str(), d)).collect();
    for d in &inp.debts {
        if d.row.is_empty() {
            out.push(r("debt-row", &d.path, "debt names no board row"));
        }
    }
    for (path, c) in &per_file {
        match (c.gap, debt.get(path.as_str())) {
            (0, None) => {}
            (g, None) => out.push(r(
                "uncovered",
                path,
                &format!(
                    "{g} fn(s) neither extracted nor waived, no debt row: {}",
                    gap_names[path].join(", ")
                ),
            )),
            (g, Some(d)) if g > d.unextracted => out.push(r(
                "debt-grew",
                path,
                &format!(
                    "{g} unextracted fns > debt {}: {}",
                    d.unextracted,
                    gap_names[path].join(", ")
                ),
            )),
            (g, Some(d)) if g < d.unextracted => out.push(r(
                "debt-stale",
                path,
                &format!(
                    "debt {} > {g} unextracted fns; lower it to {g}",
                    d.unextracted
                ),
            )),
            _ => {}
        }
    }
    for d in &inp.debts {
        if !per_file.contains_key(&d.path) {
            out.push(r(
                "debt-stale",
                &d.path,
                "debt for a path that is not a tracked src file",
            ));
        }
    }

    for w in &inp.waivers {
        let Some(f) = inp.files.iter().find(|f| f.path == w.path) else {
            out.push(r(
                "waiver-stale",
                &w.path,
                "waiver for a path that is not a tracked src file",
            ));
            continue;
        };
        if !CLASSES.contains(&w.class.as_str()) {
            out.push(r(
                "waiver-class",
                &w.path,
                &format!("class `{}` not in {:?}", w.class, CLASSES),
            ));
        }
        if w.reason.trim().len() < 12 {
            out.push(r(
                "waiver-reason",
                &w.path,
                &format!("`{}`: reason missing or under 12 characters", w.name),
            ));
        }
        if w.name == "*" {
            if !matches!(w.class.as_str(), "ffi" | "gpu" | "io" | "test-support") {
                out.push(r(
                    "waiver-scope",
                    &w.path,
                    "a whole-file waiver needs class ffi, gpu, io or test-support",
                ));
            }
        } else if !f.items.iter().any(|i| i.name == w.name) {
            out.push(r(
                "waiver-stale",
                &w.path,
                &format!("`{}` is not a fn of the file", w.name),
            ));
        } else if f
            .items
            .iter()
            .filter(|i| i.name == w.name)
            .all(|i| extracted_by_path[&(f.path.clone(), i.name.clone())])
        {
            out.push(r(
                "waiver-stale",
                &w.path,
                &format!("`{}` is extracted; delete the waiver", w.name),
            ));
        }
        if w.class == "simd-twin" {
            let twin_ok = inp
                .files
                .iter()
                .flat_map(|g| g.items.iter().map(move |i| (g, i)))
                .any(|(g, i)| i.name == w.twin && def_for(g, i, &inp.lean_defs).is_some());
            if !twin_ok {
                out.push(r(
                    "waiver-twin",
                    &w.path,
                    &format!("`{}`: twin `{}` is not an extracted fn", w.name, w.twin),
                ));
            }
        }
    }

    out.sort();
    (out, total, per_file)
}

fn r(rule: &'static str, path: &str, detail: &str) -> Refusal {
    Refusal {
        rule,
        path: path.to_string(),
        detail: detail.to_string(),
    }
}

// ---- impure boundary -------------------------------------------------------------------------

pub(crate) fn coverage_command(args: &[String]) -> ExitCode {
    let mut mode = None;
    let mut root = PathBuf::from(".");
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--check" | "--report" | "--baseline" => mode = Some(a.clone()),
            "--root" => match it.next() {
                Some(v) => root = PathBuf::from(v),
                None => return usage("--root needs a value"),
            },
            other => return usage(&format!("unknown argument `{other}`")),
        }
    }
    let Some(mode) = mode else {
        return usage("no mode given");
    };
    match load(&root) {
        Err(e) => {
            eprintln!("gate coverage: {e}");
            ExitCode::from(EXIT_ENVIRONMENT)
        }
        Ok(inp) => run(&mode, &inp),
    }
}

fn usage(m: &str) -> ExitCode {
    eprintln!("gate coverage: {m}\n{USAGE}");
    ExitCode::from(EXIT_ENVIRONMENT)
}

fn run(mode: &str, inp: &Inputs) -> ExitCode {
    let (refusals, total, per_file) = verdict(inp);
    match mode {
        "--baseline" => {
            for (p, c) in &per_file {
                if c.gap > 0 {
                    println!(
                        "[[debt]]\npath = \"{p}\"\nunextracted = {}\nrow = \"EXTRACT-COVERAGE\"\n",
                        c.gap
                    );
                }
            }
            ExitCode::from(EXIT_ACCEPTED)
        }
        "--report" => {
            print_report(&per_file, total);
            ExitCode::from(EXIT_ACCEPTED)
        }
        _ => {
            for x in &refusals {
                println!("refuse {} {} {}", x.rule, x.path, x.detail);
            }
            let mut v = Verdict::default();
            v.fields.push(Verdict::quoted(
                "verdict",
                if refusals.is_empty() {
                    "accepted"
                } else {
                    "refused"
                },
            ));
            v.fields.push(Verdict::number("fns", total.fns));
            v.fields.push(Verdict::number("extracted", total.extracted));
            v.fields.push(Verdict::number("refined", total.refined));
            v.fields.push(Verdict::number("waived", total.waived));
            v.fields.push(Verdict::number("unextracted", total.gap));
            v.fields.push(Verdict::number("refusals", refusals.len()));
            print!("{}", v.render());
            ExitCode::from(if refusals.is_empty() {
                EXIT_ACCEPTED
            } else {
                EXIT_REFUSED
            })
        }
    }
}

fn pct(n: usize, d: usize) -> String {
    if d == 0 {
        "-".into()
    } else {
        format!("{:.1}%", 100.0 * n as f64 / d as f64)
    }
}

fn print_report(per_file: &BTreeMap<String, Counts>, total: Counts) {
    let mut dirs: BTreeMap<String, Counts> = BTreeMap::new();
    for (p, c) in per_file {
        let parts: Vec<&str> = p.split('/').collect();
        let key = parts[..parts.len().saturating_sub(1).min(3)].join("/");
        let d = dirs.entry(key).or_default();
        d.fns += c.fns;
        d.extracted += c.extracted;
        d.refined += c.refined;
        d.waived += c.waived;
        d.gap += c.gap;
    }
    println!(
        "{:<28} {:>6} {:>10} {:>10} {:>7} {:>6}",
        "dir", "fns", "extracted", "refined", "waived", "gap"
    );
    for (k, c) in &dirs {
        println!(
            "{:<28} {:>6} {:>10} {:>10} {:>7} {:>6}",
            k,
            c.fns,
            pct(c.extracted, c.fns),
            pct(c.refined, c.fns),
            c.waived,
            c.gap
        );
    }
    println!(
        "{:<28} {:>6} {:>10} {:>10} {:>7} {:>6}",
        "TOTAL",
        total.fns,
        pct(total.extracted, total.fns),
        pct(total.refined, total.fns),
        total.waived,
        total.gap
    );
}

fn load(root: &Path) -> Result<Inputs, String> {
    let rd = |p: &str| {
        std::fs::read_to_string(root.join(p)).map_err(|e| format!("cannot read {p}: {e}"))
    };
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--", "src"])
        .output()
        .map_err(|e| format!("git ls-files: {e}"))?;
    if !out.status.success() {
        return Err("git ls-files failed".into());
    }
    let mut paths: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .split('\0')
        .filter(|p| p.ends_with(".rs"))
        .map(str::to_string)
        .collect();
    paths.sort();

    let mut modmap: BTreeMap<String, String> = BTreeMap::new();
    // Every extraction unit: a port file's `#[path]` modules are the real source files, and the
    // generated defs carry the port's namespace (`grad_solve_port.gguf.write_header`).
    for ex in [
        "lib.rs",
        "solve_port.rs",
        "learn_port.rs",
        "grad_solve_port.rs",
        "hashtrinity_port.rs",
        "map_table_port.rs",
        "fold_port.rs",
        "format_port.rs",
    ] {
        let prefix = ex.strip_suffix(".rs").filter(|_| ex != "lib.rs");
        // A port file that this revision does not carry contributes no modules; `lib.rs` is always
        // present and fails the read like every other required input.
        let Ok(text) = rd(&format!("proof/extract/{ex}")) else {
            if ex == "lib.rs" {
                return Err(format!("cannot read proof/extract/{ex}"));
            }
            continue;
        };
        for (p, m) in parse_path_mods(&text) {
            let m = match prefix {
                Some(pre) => format!("{pre}.{m}"),
                None => m,
            };
            modmap.insert(p, m);
        }
    }

    // Child modules are declared inside their parent (`mod x;` or `#[path = "rel"] mod x;`); the
    // extraction names them `<parent>.<child>`. Resolve them transitively from the roots.
    let mut test_files = BTreeSet::new();
    let mut queue: Vec<(String, String)> =
        modmap.iter().map(|(p, m)| (p.clone(), m.clone())).collect();
    while let Some((p, m)) = queue.pop() {
        let Ok(text) = rd(&p) else { continue };
        for (rel, name, is_test) in parse_child_mods(&text) {
            let dir = Path::new(&p).parent().unwrap_or(Path::new(""));
            let stem = Path::new(&p)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let base = if matches!(stem, "mod" | "lib" | "main") {
                dir.to_path_buf()
            } else {
                dir.join(stem)
            };
            let cands: Vec<PathBuf> = match rel {
                Some(r) => vec![dir.join(r)],
                None => vec![
                    base.join(format!("{name}.rs")),
                    base.join(&name).join("mod.rs"),
                ],
            };
            let Some(c) = cands.iter().find(|c| root.join(c).is_file()) else {
                continue;
            };
            let cp = normalise(c);
            if is_test {
                test_files.insert(cp);
                continue;
            }
            if !modmap.contains_key(&cp) {
                let cm = format!("{m}.{name}");
                modmap.insert(cp.clone(), cm.clone());
                queue.push((cp, cm));
            }
        }
    }

    let mut files = Vec::new();
    for p in paths {
        if test_files.contains(&p) {
            continue;
        }
        // Files that fail to scan are a refusal of the environment, not a silent zero.
        let items = scan_items(&rd(&p)?).map_err(|e| format!("{p}: {e}"))?;
        let module = modmap.get(&p).cloned();
        files.push(SrcFile {
            path: p,
            module,
            items,
        });
    }

    let lean = rd("proof/ApeironWord.lean")?;
    let lean_defs = lean_def_names(&lean);

    let mut corpus = String::new();
    let mut stack = vec![root.join("proof")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if p.is_dir() {
                if name != ".lake" && name != "target" {
                    stack.push(p);
                }
            } else if name.ends_with(".lean") && name != "ApeironWord.lean" {
                if let Ok(t) = std::fs::read_to_string(&p) {
                    corpus.push_str(&t);
                    corpus.push('\n');
                }
            }
        }
    }
    let corpus_tokens = tokens(&corpus);

    let (waivers, debts) = match rd("docs/contracts/extraction.toml") {
        Ok(t) => parse_rules(&t)?,
        Err(_) => (Vec::new(), Vec::new()),
    };
    Ok(Inputs {
        files,
        lean_defs,
        corpus_tokens,
        waivers,
        debts,
    })
}

fn tokens(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '\''))
        .filter(|t| !t.is_empty())
        .map(|t| t.trim_matches('.').to_string())
        .collect()
}

/// Every `def` name in a Lean module. The pretty-printer splits a long or instance name onto the
/// line after a bare `def`; that form names the same body and counts as extracted.
pub(crate) fn lean_def_names(text: &str) -> BTreeSet<String> {
    let mut defs = BTreeSet::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(n) = lean_def_name(line) {
            defs.insert(n);
            continue;
        }
        if is_bare_def(line) {
            if let Some(next) = lines.peek() {
                if !is_bare_def(next) {
                    if let Some(n) = lean_def_name(&format!("def {}", next.trim_start())) {
                        defs.insert(n);
                    }
                }
            }
        }
    }
    defs
}

/// True when the line is a `def` head with no name on it, after attributes and modifiers.
fn is_bare_def(line: &str) -> bool {
    let mut s = line.trim_start();
    while s.starts_with("@[") {
        match s.find(']') {
            Some(i) => s = s[i + 1..].trim_start(),
            None => return false,
        }
    }
    for w in ["noncomputable ", "partial ", "private ", "unsafe "] {
        s = s.strip_prefix(w).unwrap_or(s);
    }
    s.trim_end() == "def"
}

/// `def NAME` at the head of a line, after any `@[…]` attribute and `partial`/`private` words.
pub(crate) fn lean_def_name(line: &str) -> Option<String> {
    let mut s = line.trim_start();
    while s.starts_with("@[") {
        s = s[s.find(']')? + 1..].trim_start();
    }
    for w in ["noncomputable ", "partial ", "private ", "unsafe "] {
        s = s.strip_prefix(w).unwrap_or(s);
    }
    let s = s.strip_prefix("def ")?;
    let n: String = s
        .chars()
        .take_while(|c| !c.is_whitespace() && *c != ':' && *c != '(' && *c != '{' && *c != '[')
        .collect();
    (!n.is_empty()).then_some(n)
}

/// `#[path = "../../src/X.rs"]` (attributes may follow) then `pub mod NAME;`.
pub(crate) fn parse_path_mods(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut pending: Option<String> = None;
    for line in text.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("#[path = \"") {
            if let Some(end) = rest.find('"') {
                let p = &rest[..end];
                if let Some(i) = p.find("src/") {
                    pending = Some(p[i..].to_string());
                }
            }
        } else if let Some(p) = &pending {
            let m = l
                .strip_prefix("pub mod ")
                .or_else(|| l.strip_prefix("mod "))
                .and_then(|r| r.strip_suffix(';'));
            if let Some(m) = m {
                out.push((p.clone(), m.trim().to_string()));
                pending = None;
            } else if !l.starts_with("#[") && !l.is_empty() && !l.starts_with("//") {
                pending = None;
            }
        }
    }
    out
}

/// Collapse `.` and `..` components of a relative path.
fn normalise(p: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    for c in p.components() {
        match c.as_os_str().to_str().unwrap_or("") {
            "." | "" => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s.to_string()),
        }
    }
    parts.join("/")
}

/// Out-of-line `mod NAME;` declarations with their optional `#[path = "rel"]`.
pub(crate) fn parse_child_mods(text: &str) -> Vec<(Option<String>, String, bool)> {
    let mut out = Vec::new();
    let (mut path, mut test) = (None::<String>, false);
    for line in text.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("#[path = \"") {
            path = rest.find('"').map(|e| rest[..e].to_string());
        } else if l.starts_with("#[cfg(test") || l == "#[test]" {
            test = true;
        } else if l.starts_with("#[") || l.is_empty() || l.starts_with("//") {
        } else {
            let r = l
                .strip_prefix("pub(crate) mod ")
                .or_else(|| l.strip_prefix("pub(super) mod "))
                .or_else(|| l.strip_prefix("pub mod "))
                .or_else(|| l.strip_prefix("mod "));
            if let Some(n) = r.and_then(|r| r.strip_suffix(';')) {
                out.push((path.take(), n.trim().to_string(), test));
            }
            path = None;
            test = false;
        }
    }
    out
}

/// The TOML subset the file uses: `[[waiver]]` and `[[debt]]` tables of `key = "str"` / `key = N`.
pub(crate) fn parse_rules(text: &str) -> Result<(Vec<Waiver>, Vec<Debt>), String> {
    let (mut ws, mut ds) = (Vec::new(), Vec::new());
    let mut cur: Option<(String, BTreeMap<String, String>)> = None;
    let mut flush = |cur: &mut Option<(String, BTreeMap<String, String>)>| -> Result<(), String> {
        if let Some((kind, m)) = cur.take() {
            let g = |k: &str| m.get(k).cloned().unwrap_or_default();
            if kind == "waiver" {
                ws.push(Waiver {
                    path: g("path"),
                    name: g("fn"),
                    class: g("class"),
                    reason: g("reason"),
                    twin: g("twin"),
                });
            } else {
                let n = g("unextracted")
                    .parse()
                    .map_err(|_| format!("debt {}: unextracted is not a number", g("path")))?;
                ds.push(Debt {
                    path: g("path"),
                    unextracted: n,
                    row: g("row"),
                });
            }
        }
        Ok(())
    };
    for (i, line) in text.lines().enumerate() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if l == "[[waiver]]" || l == "[[debt]]" {
            flush(&mut cur)?;
            cur = Some((l[2..l.len() - 2].to_string(), BTreeMap::new()));
        } else if let Some((k, v)) = l.split_once('=') {
            let Some((_, m)) = cur.as_mut() else {
                return Err(format!("line {}: key outside a table", i + 1));
            };
            m.insert(k.trim().to_string(), v.trim().trim_matches('"').to_string());
        } else {
            return Err(format!("line {}: cannot parse `{l}`", i + 1));
        }
    }
    flush(&mut cur)?;
    Ok((ws, ds))
}

#[cfg(test)]
mod tests;
