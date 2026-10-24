//! `gate onepath`: one path through the crate, checked on the parsed source of every crate in the
//! repository (`src/`, `tests/`, `examples/`, `benches/`, `derive/`, `fold-abi/`, `fuzz/`, the
//! `proof/extract` inclusion list).
//!
//! | module | owns |
//! |---|---|
//! | `load.rs` | sources (working tree or git index), parse, module tree, fn table |
//! | `graph.rs` | name resolution and the call graph (precise and loose edges) |
//! | `rules.rs` | the rules; each reads structure, never a fn's name |
//! | `clones.rs` | normalised-token clone groups |
//! | `shader.rs` | the `.comp`/`.spv` inventory, its embed sites and the product text test |
//!
//! The contract is `docs/contracts/one-path.toml`: the declared forward bodies, one `[[kernel]]`
//! row per GPU shader with its role and faces, and one `[[debt]]` row per accepted finding naming an
//! open board ticket. A finding no row carries, a row no finding matches, and a consumer count that
//! moved are each a refusal: debt only shrinks, and a new caller of a debt body is a new refusal.
//! `--staged` also refuses what `HEAD` did not have: a new refusal, a debt row for a finding HEAD
//! did not carry, a grown consumer count and a new product kernel row. There is no mode that writes
//! the contract.

mod clones;
mod graph;
mod load;
mod rules;
mod shader;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use crate::cli::{Verdict, EXIT_ACCEPTED, EXIT_ENVIRONMENT, EXIT_REFUSED};
use crate::structure::{as_list, as_str, as_u64, field, need, parse_tables, Kind, Value};

pub(crate) const USAGE: &str =
    "usage: gate onepath (--check | --report) [--staged] [--root DIR]\n  \
  --check   refuse a finding the contract does not carry, a stale row, or a moved count\n  \
  --report  print every finding with its contract status; exit 0\n  \
  --staged  read sources and contract from the git index, and refuse only what HEAD did not have\n  \
  --root    repository root (default .)";

const CONTRACT: &str = "docs/contracts/one-path.toml";
/// Rules whose count is a consumer count: a row states it, so a new caller moves it.
const COUNTED: [&str; 4] = ["forward", "backward", "descent", "shader-product"];

pub(crate) struct Debt {
    rule: String,
    items: Vec<String>,
    consumers: Option<u64>,
    ticket: String,
    at: usize,
}

/// One GPU role: the shader that is the engine's only kernel for it.
pub(crate) struct Kernel {
    role: String,
    comp: String,
    faces: Vec<String>,
    product: bool,
    at: usize,
}

pub(crate) struct Contract {
    forward: Vec<String>,
    kernels: Vec<Kernel>,
    debts: Vec<Debt>,
}

pub(crate) fn parse_contract(src: &str) -> Result<Contract, String> {
    let tables = parse_tables(src, &[(false, "forward"), (true, "kernel")])?;
    let mut c = Contract {
        forward: Vec::new(),
        kernels: Vec::new(),
        debts: Vec::new(),
    };
    let mut saw_forward = false;
    for t in &tables {
        match t.kind {
            Kind::Top => {
                let s = need(t, &["schema"], "schema")?;
                if as_u64(s)? != 1 {
                    return Err(format!("{}:{}: schema must be 1", s.at.0, s.at.1));
                }
            }
            Kind::Named("forward") => {
                saw_forward = true;
                c.forward = as_list(need(t, &["bodies"], "bodies")?)?;
            }
            Kind::Named("kernel") => {
                const KEYS: [&str; 4] = ["role", "comp", "faces", "product"];
                let role = as_str(need(t, &KEYS, "role")?)?;
                let comp = as_str(need(t, &KEYS, "comp")?)?;
                let faces = as_list(need(t, &KEYS, "faces")?)?;
                let product = match need(t, &KEYS, "product")?.value {
                    Value::Bool(b) => b,
                    _ => {
                        return Err(format!(
                            "{}:{}: key `product` must be true or false",
                            t.at.0, t.at.1
                        ))
                    }
                };
                c.kernels.push(Kernel {
                    role,
                    comp,
                    faces,
                    product,
                    at: t.at.0,
                });
            }
            Kind::Debt => {
                const KEYS: [&str; 5] = ["rule", "item", "items", "consumers", "ticket"];
                let rule = as_str(need(t, &KEYS, "rule")?)?;
                let items = match (field(t, &KEYS, "item")?, field(t, &KEYS, "items")?) {
                    (Some(f), None) => vec![as_str(f)?],
                    (None, Some(f)) => {
                        let mut v = as_list(f)?;
                        v.sort();
                        v
                    }
                    _ => {
                        return Err(format!(
                            "{}:{}: a debt names exactly one of `item` or `items`",
                            t.at.0, t.at.1
                        ))
                    }
                };
                let consumers = field(t, &KEYS, "consumers")?.map(as_u64).transpose()?;
                let ticket = as_str(need(t, &KEYS, "ticket")?)?;
                c.debts.push(Debt {
                    rule,
                    items,
                    consumers,
                    ticket,
                    at: t.at.0,
                });
            }
            _ => return Err(format!("{}:{}: unknown table", t.at.0, t.at.1)),
        }
    }
    if !saw_forward {
        return Err("missing [forward] table".into());
    }
    Ok(c)
}

/// The state of a board ticket: `None` when the file is absent.
fn ticket_state(root: &Path, id: &str) -> Option<String> {
    let path = root.join("../indra/board/tickets").join(format!("{id}.md"));
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix("state:"))
        .map(|s| s.trim().to_string())
}

pub(crate) struct Outcome {
    pub refusals: Vec<String>,
    pub per_rule: BTreeMap<&'static str, (usize, usize)>,
    pub lines: Vec<String>,
    /// Every finding the source holds, carried or not, with the count it carries: the identity and
    /// the count the source ratchet compares.
    pub found: Vec<Found>,
}

/// One finding as the ratchet sees it: its rule, its items and its source-side count.
pub(crate) struct Found {
    pub rule: &'static str,
    pub items: Vec<String>,
    pub count: Option<usize>,
}

/// The kernel row that admits a shader finding, if any.
fn admit<'a>(
    f: &rules::Finding,
    comps: &BTreeMap<&str, &'a Kernel>,
    faces: &BTreeMap<&str, &'a Kernel>,
) -> Option<&'a Kernel> {
    let item = f.items.first()?;
    match f.rule {
        // A `.comp` or `.spv` is the engine's path only when a `[[kernel]]` row names it.
        "shader" => comps
            .get(item.as_str())
            .or_else(|| faces.get(item.as_str()))
            .copied(),
        // A product is the engine's path only on a row that says so.
        "shader-product" => comps.get(item.as_str()).copied().filter(|k| k.product),
        _ => None,
    }
}

/// Compares the findings with the contract.
///
/// Cost: time O(F log F + D log D + K) over F findings, D rows and K kernel rows; space O(F + D).
pub(crate) fn compare(
    root: &Path,
    a: &rules::Analysis,
    c: &Contract,
    inv: &shader::Inventory,
) -> Outcome {
    let mut refusals = Vec::new();
    let mut lines = Vec::new();
    let mut found = Vec::new();
    let mut per_rule: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let declared: BTreeSet<&str> = c.forward.iter().map(String::as_str).collect();
    let forwards: BTreeSet<&str> = a.forwards.iter().map(String::as_str).collect();
    for body in &declared {
        if !forwards.contains(body) {
            refusals.push(format!(
                "stale forward: `{body}` is not a minimal forward body in the source"
            ));
        }
    }
    // The GPU role map: a role is unique, and its comp and faces exist.
    let mut roles: BTreeSet<&str> = BTreeSet::new();
    let mut comps: BTreeMap<&str, &Kernel> = BTreeMap::new();
    let mut faces: BTreeMap<&str, &Kernel> = BTreeMap::new();
    for k in &c.kernels {
        if !roles.insert(k.role.as_str()) {
            refusals.push(format!(
                "{CONTRACT}:{}: duplicate kernel role `{}`",
                k.at, k.role
            ));
        }
        if !inv.comps.contains_key(&k.comp) {
            refusals.push(format!(
                "{CONTRACT}:{}: stale kernel row `{}`: no such .comp",
                k.at, k.comp
            ));
        }
        comps.insert(k.comp.as_str(), k);
        for f in &k.faces {
            if !inv.spvs.contains(f) {
                refusals.push(format!(
                    "{CONTRACT}:{}: stale face `{f}` of role `{}`: no such .spv",
                    k.at, k.role
                ));
            }
            faces.insert(f.as_str(), k);
        }
    }
    let mut rows: BTreeMap<(String, Vec<String>), &Debt> = BTreeMap::new();
    for d in &c.debts {
        if rows.insert((d.rule.clone(), d.items.clone()), d).is_some() {
            refusals.push(format!("{CONTRACT}:{}: duplicate debt row", d.at));
        }
        match ticket_state(root, &d.ticket).as_deref() {
            None => refusals.push(format!(
                "{CONTRACT}:{}: ticket {} has no file under ../indra/board/tickets/",
                d.at, d.ticket
            )),
            Some(s @ ("done" | "cancelled")) => refusals.push(format!(
                "{CONTRACT}:{}: ticket {} is {s}; its debt is unowned",
                d.at, d.ticket
            )),
            Some(_) => {}
        }
    }
    let mut matched = BTreeSet::new();
    for f in &a.findings {
        let e = per_rule.entry(f.rule).or_insert((0, 0));
        e.0 += 1;
        found.push(Found {
            rule: f.rule,
            items: f.items.clone(),
            count: f.count,
        });
        if f.rule == "forward" && declared.contains(f.items[0].as_str()) {
            e.1 += 1;
            lines.push(format!("declared\t{}\t{}\t{}", f.rule, f.at, f.items[0]));
            continue;
        }
        if let Some(k) = admit(f, &comps, &faces) {
            e.1 += 1;
            lines.push(format!(
                "kernel\t{}\t{}\t{}\t{}",
                f.rule,
                f.at,
                f.items.join(" "),
                k.role
            ));
            continue;
        }
        let key = (f.rule.to_string(), f.items.clone());
        let count = f.count.map(|n| n.to_string()).unwrap_or_default();
        let status = match rows.get(&key) {
            None => {
                refusals.push(format!(
                    "new {} at {}: {} [{}] {}",
                    f.rule,
                    f.at,
                    f.items.join(", "),
                    count,
                    f.note
                ));
                "new"
            }
            Some(d) => {
                matched.insert(key.clone());
                e.1 += 1;
                if COUNTED.contains(&f.rule) && d.consumers.map(|n| n as usize) != f.count {
                    refusals.push(format!(
                        "{CONTRACT}:{}: {} {} consumers moved: row {}, source {}",
                        d.at,
                        f.rule,
                        f.items.join(", "),
                        d.consumers
                            .map(|n| n.to_string())
                            .unwrap_or("absent".into()),
                        count
                    ));
                    "moved"
                } else {
                    "debt"
                }
            }
        };
        lines.push(format!(
            "{status}\t{}\t{}\t{}\t{}\t{}",
            f.rule,
            f.at,
            f.items.join(" "),
            count,
            f.note
        ));
    }
    for (key, d) in &rows {
        if !matched.contains(key) {
            refusals.push(format!(
                "{CONTRACT}:{}: stale debt {} {}: no such finding",
                d.at,
                d.rule,
                d.items.join(", ")
            ));
        }
    }
    Outcome {
        refusals,
        per_rule,
        lines,
        found,
    }
}

fn usage(m: &str) -> ExitCode {
    eprintln!("gate onepath: {m}\n{USAGE}");
    ExitCode::from(EXIT_ENVIRONMENT)
}

fn read_contract(root: &Path, staged: bool) -> Result<String, String> {
    if staged {
        git_contract(root, ":")
    } else {
        std::fs::read_to_string(root.join(CONTRACT)).map_err(|e| format!("{CONTRACT}: {e}"))
    }
}

/// The contract at a git object spec: `":"` for the index, `"HEAD:"` for the last commit.
fn git_contract(root: &Path, spec: &str) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["show", &format!("{spec}{CONTRACT}")])
        .output()
        .map_err(|e| format!("git show: {e}"))?;
    if !out.status.success() {
        return Err(format!("{CONTRACT} is not in {spec}"));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("{CONTRACT}: {e}"))
}

/// Which tree the shaders are read from.
#[derive(Clone, Copy)]
enum Where<'a> {
    Tree,
    Index,
    Rev(&'a str),
}

/// Loads, analyses and compares one tree; `Err` is an environment failure.
fn evaluate(
    root: &Path,
    sources: &load::Sources,
    contract_text: &str,
    at: Where,
) -> Result<(rules::Analysis, Outcome, Vec<String>), String> {
    let t0 = std::time::Instant::now();
    let contract = parse_contract(contract_text).map_err(|e| format!("{CONTRACT}: {e}"))?;
    let tree = load::load(sources);
    if tree.fns.is_empty() {
        return Err("no fn loaded: the gate refuses to pass on nothing".into());
    }
    let t_load = t0.elapsed().as_millis();
    let idx = graph::index(&tree);
    let edges = graph::edges(&tree, &idx);
    let t_graph = t0.elapsed().as_millis();
    let mut analysis = rules::analyse(&tree, &idx, &edges);
    let mut inv = match at {
        Where::Tree => shader::load(root, false),
        Where::Index => shader::load(root, true),
        Where::Rev(r) => shader::at_rev(root, r),
    }?;
    inv.embed = shader::embeds(&tree);
    analysis.findings.extend(shader::findings(&inv));
    let t_rules = t0.elapsed().as_millis();
    analysis.stats.push(("comp_files", inv.comps.len()));
    analysis.stats.push(("spv_faces", inv.spvs.len()));
    let outcome = compare(root, &analysis, &contract, &inv);
    analysis.stats.push(("ms_load", t_load as usize));
    analysis
        .stats
        .push(("ms_graph", (t_graph - t_load) as usize));
    analysis
        .stats
        .push(("ms_rules", (t_rules - t_graph) as usize));
    Ok((analysis, outcome, tree.errors))
}

/// A refusal string with every digit run collapsed, so the same refusal in two trees compares equal
/// **as an identity only**: this collapses the count too, so `norm` says "the same refusal", never
/// "the same count". Counts are compared by the source-count ratchet (e), which reads
/// [`Outcome::found`] on both trees; a row HEAD already refuses still refuses a grown count.
///
/// Cost: time O(L) over the L bytes, space O(L).
fn norm(r: &str) -> String {
    let mut out = String::with_capacity(r.len());
    let mut digits = false;
    for ch in r.chars() {
        if ch.is_ascii_digit() {
            if !digits {
                out.push('#');
                digits = true;
            }
        } else {
            digits = false;
            out.push(ch);
        }
    }
    out
}

/// The index against `HEAD`: a commit may remove refusals, never add one.
///
/// (a) a refusal the index has and `HEAD` does not; (b) a `[[debt]]` row whose finding `HEAD` did
/// not already hold; (c) a debt row whose contract `consumers` grew; (d) a new `product = true`
/// kernel row once the role map exists; (e) a finding whose source-side count grew — the count
/// `compare` reads from the source, which is the number a carried row's `consumers` must equal.
/// (e) is why a row `HEAD` already refuses still ratchets: `norm` collapses the count, so (a) cannot
/// see growth on an already-red row, and (c) reads only the contract. The rows of the commit that
/// introduces the map are the baseline for the 16 shader findings that were already red at `HEAD`,
/// so (d) admits them there and refuses a product row added afterwards.
///
/// Cost: one further full analysis of `HEAD`; time O(that) plus O((F + D) log(F + D)).
fn ratchet(root: &Path, contract_text: &str, index: &Outcome) -> Result<Vec<String>, String> {
    let index_contract = parse_contract(contract_text).map_err(|e| format!("{CONTRACT}: {e}"))?;
    let head_text = match git_contract(root, "HEAD:") {
        Ok(t) => t,
        // A repository whose first commit is this one has no HEAD contract to compare.
        Err(_) => "schema = 1\n\n[forward]\nbodies = []\n".to_string(),
    };
    let head_contract =
        parse_contract(&head_text).map_err(|e| format!("{CONTRACT} at HEAD: {e}"))?;
    let (_, head, _) = evaluate(
        root,
        &load::head_sources(root)?,
        &head_text,
        Where::Rev("HEAD"),
    )?;
    let mut out = Vec::new();
    let head_refusals: BTreeSet<String> = head.refusals.iter().map(|r| norm(r)).collect();
    for r in &index.refusals {
        if !head_refusals.contains(&norm(r)) {
            out.push(format!(
                "ratchet: the index adds a refusal HEAD had not: {r}"
            ));
        }
    }
    let head_found: BTreeSet<(String, Vec<String>)> = head
        .found
        .iter()
        .map(|f| (f.rule.to_string(), f.items.clone()))
        .collect();
    // (e) The source-side count, whichever tree the row is in. A row HEAD already refuses is still
    // ratcheted: growth on it is invisible to (a).
    let head_counts: BTreeMap<(String, Vec<String>), usize> = head
        .found
        .iter()
        .filter_map(|f| f.count.map(|c| ((f.rule.to_string(), f.items.clone()), c)))
        .collect();
    for f in &index.found {
        let Some(now) = f.count else { continue };
        let key = (f.rule.to_string(), f.items.clone());
        if let Some(&before) = head_counts.get(&key) {
            if now > before {
                out.push(format!(
                    "ratchet: {} {} source consumers grew {before} -> {now}",
                    f.rule,
                    f.items.join(" ")
                ));
            }
        }
    }
    let row = |c: &Contract, rule: &str, items: &[String]| -> Option<Option<u64>> {
        c.debts
            .iter()
            .find(|d| d.rule == rule && d.items == items)
            .map(|d| d.consumers)
    };
    for d in &index_contract.debts {
        match row(&head_contract, &d.rule, &d.items) {
            Some(head_count) => {
                if let (Some(a), Some(b)) = (d.consumers, head_count) {
                    if a > b {
                        out.push(format!(
                            "ratchet: {CONTRACT} debt row {} {} consumers grew {b} -> {a}",
                            d.rule,
                            d.items.join(" ")
                        ));
                    }
                }
            }
            None => {
                if !head_found.contains(&(d.rule.clone(), d.items.clone())) {
                    out.push(format!(
                        "ratchet: {CONTRACT} adds a debt row whose finding HEAD had not: {} {}",
                        d.rule,
                        d.items.join(" ")
                    ));
                }
            }
        }
    }
    if !head_contract.kernels.is_empty() {
        for k in &index_contract.kernels {
            let known = head_contract
                .kernels
                .iter()
                .any(|h| h.comp == k.comp && h.product);
            if k.product && !known {
                out.push(format!(
                    "ratchet: {CONTRACT} adds a product kernel row `{}` ({})",
                    k.role, k.comp
                ));
            }
        }
    }
    Ok(out)
}

/// Loads, analyses, compares, and on `--staged` holds the index against `HEAD`.
///
/// `--staged` refuses only what `HEAD` did not have: the refusals `HEAD` already carries are the
/// burn-down queue, printed as `prior_refusals` and not a blocker. A commit that only removes
/// refusals passes; a commit that adds one fails.
pub(crate) fn run(
    root: &Path,
    staged: bool,
) -> Result<(rules::Analysis, Outcome, Vec<String>), String> {
    let sources = if staged {
        load::staged_sources(root)?
    } else {
        load::standing_sources(root)?
    };
    let text = read_contract(root, staged)?;
    let at = if staged { Where::Index } else { Where::Tree };
    let (mut analysis, mut outcome, errors) = evaluate(root, &sources, &text, at)?;
    if staged {
        let added = ratchet(root, &text, &outcome)?;
        analysis
            .stats
            .push(("prior_refusals", outcome.refusals.len()));
        outcome.refusals = added;
    }
    Ok((analysis, outcome, errors))
}

pub(crate) fn onepath_command(args: &[String]) -> ExitCode {
    let mut mode = None;
    let mut root = PathBuf::from(".");
    let mut staged = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--check" | "--report" => mode = Some(a.clone()),
            "--root" => match it.next() {
                Some(v) => root = PathBuf::from(v),
                None => return usage("--root needs a value"),
            },
            "--staged" => staged = true,
            other => return usage(&format!("unknown argument `{other}`")),
        }
    }
    let Some(mode) = mode else {
        return usage("no mode given");
    };
    let root = match root.canonicalize() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gate onepath: root {}: {e}", root.display());
            return ExitCode::from(EXIT_ENVIRONMENT);
        }
    };
    let (analysis, mut outcome, errors) = match run(&root, staged) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gate onepath: {e}");
            return ExitCode::from(EXIT_ENVIRONMENT);
        }
    };
    // A file that does not parse is code no rule read.
    for e in &errors {
        outcome.refusals.push(format!("unparsed: {e}"));
    }
    if mode == "--report" {
        for l in &outcome.lines {
            println!("{l}");
        }
    }
    let mut v = Verdict::default();
    v.fields.push(Verdict::quoted("gate", "onepath"));
    v.fields.push(Verdict::quoted("mode", &mode));
    v.fields.push(Verdict::number("staged", staged));
    for (k, n) in &analysis.stats {
        v.fields.push(Verdict::number(k, n));
    }
    v.fields
        .push(Verdict::number("forward_bodies", analysis.forwards.len()));
    for (rule, (all, debt)) in &outcome.per_rule {
        v.fields.push(Verdict::quoted(
            &format!("rule.{rule}"),
            &format!("{all} findings, {debt} carried"),
        ));
    }
    v.fields
        .push(Verdict::number("refusals", outcome.refusals.len()));
    let _ = analysis.keys.len();
    if mode == "--check" {
        for r in &outcome.refusals {
            eprintln!("gate onepath: {r}");
        }
    }
    print!("{}", v.render());
    if mode == "--check" && !outcome.refusals.is_empty() {
        ExitCode::from(EXIT_REFUSED)
    } else {
        ExitCode::from(EXIT_ACCEPTED)
    }
}
