//! `gate structure --check`: the repository-layout gate. The seven rules and the tree they
//! describe are in `docs/contracts/tree.toml`; the decision is `gate::structure_verdict` in
//! `core.rs`. This module parses that file (the TOML subset it uses and nothing more), reads the
//! tracked list and counts lines, and prints the refusals and the verdict block.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use gate::{
    is_source, lean_verdict, structure_verdict, DebtRule, DirRule, LeanCounts, LeanModule,
    LeanRules, Rules,
};

use crate::cli::{Verdict, EXIT_ACCEPTED, EXIT_ENVIRONMENT, EXIT_REFUSED};

pub(crate) const USAGE: &str =
    "usage: gate structure --check [--root DIR] [--rules PATH] [--files PATH]\n  \
  --root DIR    repository root (default .)\n  \
  --rules PATH  layout rules (default <root>/docs/contracts/tree.toml)\n  \
  --files PATH  newline list of tracked paths (default: git -C <root> ls-files -z)";

struct Options {
    root: PathBuf,
    rules: Option<PathBuf>,
    files: Option<PathBuf>,
}

/// `gate structure --check …`. Exit 0 accepted, 1 refused, 2 environment or rules parse error.
pub(crate) fn structure_command(args: &[String]) -> ExitCode {
    let opts = match parse_args(args) {
        Ok(opts) => opts,
        Err(message) => {
            eprintln!("gate structure: {message}\n{USAGE}");
            return ExitCode::from(EXIT_ENVIRONMENT);
        }
    };
    match run(&opts) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("gate structure: {message}");
            ExitCode::from(EXIT_ENVIRONMENT)
        }
    }
}

/// Requires `--check` and one value per flag; any other argument is an error.
///
/// Cost: time O(A) over the A arguments, space O(1).
fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut check = false;
    let mut opts = Options {
        root: PathBuf::from("."),
        rules: None,
        files: None,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "--check" => check = true,
            "--root" => opts.root = PathBuf::from(value("--root")?),
            "--rules" => opts.rules = Some(PathBuf::from(value("--rules")?)),
            "--files" => opts.files = Some(PathBuf::from(value("--files")?)),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    if !check {
        return Err("no mode given; the only mode is --check".into());
    }
    Ok(opts)
}

fn run(opts: &Options) -> Result<ExitCode, String> {
    let rules_path = opts
        .rules
        .clone()
        .unwrap_or_else(|| opts.root.join("docs/contracts/tree.toml"));
    let text = std::fs::read_to_string(&rules_path)
        .map_err(|e| format!("cannot read rules {}: {e}", rules_path.display()))?;
    let rules = parse_rules(&text).map_err(|e| format!("{}:{e}", rules_path.display()))?;

    let paths = match &opts.files {
        Some(list) => read_list(list)?,
        None => git_list(&opts.root)?,
    };
    if paths.is_empty() {
        return Err("empty file list; refusing to report success on nothing".into());
    }

    let files = count_lines(&opts.root, &rules, &paths)?;
    let lean_modules = read_lean(&opts.root, &rules, &paths)?;
    let (lean_refusals, lean_counts) = lean_verdict(&rules, &paths, &lean_modules);
    let mut refusals = structure_verdict(&rules, &files);
    refusals.extend(lean_refusals);
    refusals.sort();
    for r in &refusals {
        println!("refuse {} {} {}", r.rule, r.path, r.detail);
    }

    let mut v = Verdict::default();
    let verdict = if refusals.is_empty() {
        "accepted"
    } else {
        "refused"
    };
    v.fields.push(Verdict::quoted("verdict", verdict));
    v.fields.push(Verdict::number("files", files.len()));
    v.fields.push(Verdict::number("refusals", refusals.len()));
    v.fields.push(Verdict::number("debt", rules.debts.len()));
    v.fields
        .push(Verdict::number("frozen_python", rules.frozen.len()));
    let LeanCounts {
        modules,
        linked,
        support,
    } = lean_counts;
    v.fields.push(Verdict::number("lean_modules", modules));
    v.fields.push(Verdict::number("linked", linked));
    v.fields.push(Verdict::number("support", support));
    v.fields.push(Verdict::number(
        "orphan_debt",
        rules.lean.as_ref().map_or(0, |l| l.orphan_debt.len()),
    ));
    v.fields
        .push(Verdict::quoted("rules", &rules_path.display().to_string()));
    print!("{}", v.render());

    Ok(ExitCode::from(if refusals.is_empty() {
        EXIT_ACCEPTED
    } else {
        EXIT_REFUSED
    }))
}

/// The newline list at `path`; empty lines are skipped.
///
/// Cost: time O(B) over the B bytes of the list, space O(B) for the owned paths.
fn read_list(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read file list {}: {e}", path.display()))?;
    Ok(text
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

/// `git -C <root> ls-files -z`, NUL-separated. A failed git is an error, never an empty list.
///
/// Cost: time O(B) over the B bytes of git's output, space O(B).
fn git_list(root: &Path) -> Result<Vec<String>, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .map_err(|e| format!("cannot run git -C {}: {e}", root.display()))?;
    if !out.status.success() {
        return Err(format!(
            "git -C {} ls-files -z failed ({}): {}",
            root.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let text = String::from_utf8(out.stdout)
        .map_err(|e| format!("git ls-files output is not UTF-8: {e}"))?;
    Ok(text
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect())
}

/// Each tracked path with its newline count when it is a source file (the `wc -l` count), else 0.
/// A source file that cannot be read is an environment error, not a zero count.
///
/// Cost: time O(S) over the S source-file bytes (one read and one byte scan each), space O(S)
/// for one file's bytes at a time plus O(F) for the result.
fn count_lines(
    root: &Path,
    rules: &Rules,
    paths: &[String],
) -> Result<Vec<(String, usize)>, String> {
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let lines = if is_source(rules, path) {
            let bytes =
                std::fs::read(root.join(path)).map_err(|e| format!("cannot read {path}: {e}"))?;
            bytes.iter().filter(|&&b| b == b'\n').count()
        } else {
            0
        };
        out.push((path.clone(), lines));
    }
    Ok(out)
}

// ---- the rules file: a TOML subset -------------------------------------------------------------------

#[derive(Debug)]
enum Value {
    Int(i64),
    Str(String),
    Bool(bool),
    List(Vec<String>),
}

#[derive(Debug)]
struct Field {
    key: String,
    value: Value,
    at: (usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Top,
    Dir,
    Debt,
    Python,
    Lean,
}

#[derive(Debug)]
struct Table {
    kind: Kind,
    at: (usize, usize),
    fields: Vec<Field>,
}

/// A position in the rules text, tracked as 1-based line and column in characters.
struct Cursor<'a> {
    src: &'a str,
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str) -> Self {
        Cursor {
            src,
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    /// `line:col: expected <what>, found <the rest of the line>`.
    fn fail<T>(&self, what: &str) -> Result<T, String> {
        let found = match self.peek() {
            None => "end of file".to_string(),
            Some('\n') => "end of line".to_string(),
            Some(_) => format!("`{}`", excerpt(&self.src[self.pos..])),
        };
        Err(format!(
            "{}:{}: expected {what}, found {found}",
            self.line, self.col
        ))
    }

    fn skip_spaces(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\r')) {
            self.bump();
        }
    }

    fn skip_comment(&mut self) {
        if self.peek() == Some('#') {
            while let Some(c) = self.peek() {
                if c == '\n' {
                    break;
                }
                self.bump();
            }
        }
    }

    /// Spaces, an optional comment, then the end of the line.
    fn end_of_line(&mut self) -> Result<(), String> {
        self.skip_spaces();
        self.skip_comment();
        match self.peek() {
            None => Ok(()),
            Some('\n') => {
                self.bump();
                Ok(())
            }
            Some(_) => self.fail("end of line"),
        }
    }

    /// Whitespace, newlines and comment lines: the separators between statements and array items.
    fn skip_blank(&mut self) {
        loop {
            self.skip_spaces();
            match self.peek() {
                Some('\n') => {
                    self.bump();
                }
                Some('#') => self.skip_comment(),
                _ => break,
            }
        }
    }

    fn bare_key(&mut self) -> String {
        let mut key = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                key.push(c);
                self.bump();
            } else {
                break;
            }
        }
        key
    }

    fn expect(&mut self, token: &str) -> Result<(), String> {
        if self.src[self.pos..].starts_with(token) {
            for _ in token.chars() {
                self.bump();
            }
            Ok(())
        } else {
            self.fail(&format!("`{token}`"))
        }
    }
}

/// The first line of `text`, cut to 60 characters, for a diagnostic.
fn excerpt(text: &str) -> String {
    text.lines().next().unwrap_or("").chars().take(60).collect()
}

/// Parses the file into tables. Every error carries `line:col`.
///
/// Cost: time O(N) over the N bytes of the text (one pass; each character is read once), space
/// O(N) for the owned keys and strings.
fn parse_tables(src: &str) -> Result<Vec<Table>, String> {
    let mut c = Cursor::new(src);
    let mut tables = vec![Table {
        kind: Kind::Top,
        at: (1, 1),
        fields: Vec::new(),
    }];
    let mut python_seen = false;
    loop {
        c.skip_blank();
        if c.peek().is_none() {
            break;
        }
        if c.peek() == Some('[') {
            let at = (c.line, c.col);
            let start = c.pos;
            c.bump();
            let double = c.peek() == Some('[');
            if double {
                c.bump();
            }
            let name = c.bare_key();
            let kind = match (double, name.as_str()) {
                (true, "dir") => Kind::Dir,
                (true, "debt") => Kind::Debt,
                (false, "python") => Kind::Python,
                (false, "lean") => Kind::Lean,
                _ => {
                    return Err(format!(
                        "{}:{}: unknown table header `{}`",
                        at.0,
                        at.1,
                        excerpt(&src[start..])
                    ))
                }
            };
            if kind == Kind::Python {
                if python_seen {
                    return Err(format!("{}:{}: duplicate [python] table", at.0, at.1));
                }
                python_seen = true;
            }
            c.expect(if double { "]]" } else { "]" })?;
            c.end_of_line()?;
            tables.push(Table {
                kind,
                at,
                fields: Vec::new(),
            });
            continue;
        }
        let at = (c.line, c.col);
        let key = c.bare_key();
        if key.is_empty() {
            return c.fail("a table header or a key");
        }
        c.skip_spaces();
        if c.peek() != Some('=') {
            return c.fail("`=`");
        }
        c.bump();
        c.skip_spaces();
        let value = parse_value(&mut c)?;
        c.end_of_line()?;
        if let Some(table) = tables.last_mut() {
            table.fields.push(Field { key, value, at });
        }
    }
    Ok(tables)
}

fn parse_value(c: &mut Cursor) -> Result<Value, String> {
    match c.peek() {
        Some('"') => Ok(Value::Str(parse_string(c)?)),
        Some('[') => {
            c.bump();
            let mut items = Vec::new();
            loop {
                c.skip_blank();
                match c.peek() {
                    Some(']') => {
                        c.bump();
                        break;
                    }
                    Some('"') => {}
                    _ => return c.fail("a string or `]`"),
                }
                items.push(parse_string(c)?);
                c.skip_blank();
                match c.peek() {
                    Some(',') => {
                        c.bump();
                    }
                    Some(']') => {
                        c.bump();
                        break;
                    }
                    _ => return c.fail("`,` or `]`"),
                }
            }
            Ok(Value::List(items))
        }
        Some('t' | 'f') => {
            let mut word = String::new();
            while let Some(ch) = c.peek() {
                if !ch.is_ascii_alphabetic() {
                    break;
                }
                word.push(ch);
                c.bump();
            }
            match word.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => c.fail("`true` or `false`"),
            }
        }
        Some(ch) if ch == '-' || ch.is_ascii_digit() => {
            let mut text = String::new();
            if ch == '-' {
                text.push('-');
                c.bump();
            }
            while let Some(d) = c.peek() {
                if !d.is_ascii_digit() {
                    break;
                }
                text.push(d);
                c.bump();
            }
            match text.parse::<i64>() {
                Ok(n) => Ok(Value::Int(n)),
                Err(_) => c.fail("an integer"),
            }
        }
        _ => c.fail("a value"),
    }
}

/// A basic string: `\\` and `\"` are the only escapes; a line break inside it is an error.
///
/// Cost: time O(L) over the L characters of the string, space O(L).
fn parse_string(c: &mut Cursor) -> Result<String, String> {
    if c.peek() != Some('"') {
        return c.fail("a string");
    }
    let (line, col) = (c.line, c.col);
    c.bump();
    let mut out = String::new();
    loop {
        match c.bump() {
            None | Some('\n') => {
                return Err(format!("{line}:{col}: unterminated string"));
            }
            Some('"') => return Ok(out),
            Some('\\') => match c.bump() {
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                _ => return Err(format!("{line}:{col}: unsupported escape in string")),
            },
            Some(ch) => out.push(ch),
        }
    }
}

/// Rejects an unknown key, a repeated key, and a key of the wrong type; returns the field by name.
fn field<'a>(t: &'a Table, allowed: &[&str], key: &str) -> Result<Option<&'a Field>, String> {
    let mut found = None;
    for f in &t.fields {
        if !allowed.contains(&f.key.as_str()) {
            return Err(format!(
                "{}:{}: unknown key `{}` in this table",
                f.at.0, f.at.1, f.key
            ));
        }
        if f.key == key {
            if found.is_some() {
                return Err(format!("{}:{}: duplicate key `{key}`", f.at.0, f.at.1));
            }
            found = Some(f);
        }
    }
    Ok(found)
}

fn need<'a>(t: &'a Table, allowed: &[&str], key: &str) -> Result<&'a Field, String> {
    field(t, allowed, key)?.ok_or_else(|| format!("{}:{}: missing key `{key}`", t.at.0, t.at.1))
}

fn mismatch(f: &Field, what: &str) -> String {
    format!("{}:{}: key `{}` must be {what}", f.at.0, f.at.1, f.key)
}

fn as_str(f: &Field) -> Result<String, String> {
    match &f.value {
        Value::Str(s) => Ok(s.clone()),
        _ => Err(mismatch(f, "a string")),
    }
}

fn as_list(f: &Field) -> Result<Vec<String>, String> {
    match &f.value {
        Value::List(items) => Ok(items.clone()),
        _ => Err(mismatch(f, "an array of strings")),
    }
}

fn as_u64(f: &Field) -> Result<u64, String> {
    match f.value {
        Value::Int(n) if n >= 0 => Ok(n as u64),
        _ => Err(mismatch(f, "a non-negative integer")),
    }
}

/// Parses and validates the rules file: the top-level keys, every `[[dir]]`, `[[debt]]` and the
/// `[python]` table. Every `[[dir]]` path is unique. The error is `line:col: message`.
///
/// Cost: time O(N + D log D) for N bytes of text and D `[[dir]]` entries (the uniqueness check is a
/// B-tree insert per entry); space O(N).
pub(crate) fn parse_rules(src: &str) -> Result<Rules, String> {
    let tables = parse_tables(src)?;
    let mut rules = Rules::default();
    let mut seen_dirs = BTreeSet::new();
    for t in &tables {
        match t.kind {
            Kind::Top => {
                const TOP: &[&str] = &[
                    "schema",
                    "max_lines",
                    "max_dir_files",
                    "source",
                    "generated",
                ];
                let schema = need(t, TOP, "schema")?;
                if !matches!(schema.value, Value::Int(1)) {
                    return Err(mismatch(schema, "the integer 1 (the only schema)"));
                }
                rules.max_lines = as_u64(need(t, TOP, "max_lines")?)?;
                rules.max_dir_files = as_u64(need(t, TOP, "max_dir_files")?)?;
                rules.source = as_list(need(t, TOP, "source")?)?;
                if let Some(f) = field(t, TOP, "generated")? {
                    rules.generated = as_list(f)?;
                }
            }
            Kind::Dir => {
                const DIR: &[&str] = &["path", "holds", "files", "recursive"];
                let path = as_str(need(t, DIR, "path")?)?;
                if !seen_dirs.insert(path.clone()) {
                    return Err(format!(
                        "{}:{}: duplicate [[dir]] path `{path}`",
                        t.at.0, t.at.1
                    ));
                }
                let files = as_list(need(t, DIR, "files")?)?;
                let recursive = match field(t, DIR, "recursive")? {
                    None => false,
                    Some(f) => match f.value {
                        Value::Bool(b) => b,
                        _ => return Err(mismatch(f, "true or false")),
                    },
                };
                if let Some(f) = field(t, DIR, "holds")? {
                    as_str(f)?;
                }
                rules.dirs.push(DirRule {
                    path,
                    recursive,
                    files,
                });
            }
            Kind::Debt => {
                const DEBT: &[&str] = &["rule", "path", "row"];
                let rule = as_str(need(t, DEBT, "rule")?)?;
                let path = as_str(need(t, DEBT, "path")?)?;
                as_str(need(t, DEBT, "row")?)?;
                rules.debts.push(DebtRule { rule, path });
            }
            Kind::Python => {
                const PY: &[&str] = &["frozen"];
                if let Some(f) = field(t, PY, "frozen")? {
                    rules.frozen = as_list(f)?;
                }
            }
            Kind::Lean => {
                const LEAN: &[&str] = &["root", "audit_modules", "orphan_debt"];
                if rules.lean.is_some() {
                    return Err(format!("{}:{}: duplicate [lean] table", t.at.0, t.at.1));
                }
                rules.lean = Some(LeanRules {
                    root: as_str(need(t, LEAN, "root")?)?,
                    audit_modules: as_list(need(t, LEAN, "audit_modules")?)?,
                    orphan_debt: as_list(need(t, LEAN, "orphan_debt")?)?,
                });
            }
        }
    }
    rules.dirs.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(rules)
}

/// Each tracked `.lean` file under `[lean].root` with its import names; empty when there is no
/// `[lean]` table. A file that cannot be read is an environment error.
///
/// Cost: time O(L) over the L bytes of the population's files (one read each), space O(L) for one
/// file at a time plus O(I) for the I import names kept.
fn read_lean(root: &Path, rules: &Rules, paths: &[String]) -> Result<Vec<LeanModule>, String> {
    let Some(lean) = &rules.lean else {
        return Ok(Vec::new());
    };
    let prefix = format!("{}/", lean.root);
    // The package's `lakefile.lean` is build configuration, not a module of the library.
    let lakefile = format!("{prefix}lakefile.lean");
    let mut out = Vec::new();
    for path in paths
        .iter()
        .filter(|p| p.starts_with(&prefix) && p.ends_with(".lean") && **p != lakefile)
    {
        let text = std::fs::read_to_string(root.join(path))
            .map_err(|e| format!("cannot read {path}: {e}"))?;
        out.push(LeanModule {
            path: path.clone(),
            imports: lean_imports(&text),
        });
    }
    Ok(out)
}

/// The import names of a Lean file: the `import` lines before the first line that is neither blank,
/// a comment, `module`, nor an import. `public`, `meta` and `private` precede `import`; a `/- -/`
/// block comment may span lines. A trailing `--` comment is dropped.
///
/// Cost: time O(H) over the H bytes up to the end of the header, space O(I) for the I names.
/// Each line is scanned once; the header ends at the first non-import line.
fn lean_imports(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_block = false;
    for raw in text.lines() {
        let line = raw.trim();
        if in_block {
            in_block = !line.contains("-/");
            continue;
        }
        if line.is_empty() || line.starts_with("--") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("/-") {
            in_block = !rest.contains("-/");
            continue;
        }
        let code = line.split("--").next().unwrap_or("").trim();
        if code == "module" {
            continue;
        }
        let words: Vec<&str> = code
            .split_whitespace()
            .skip_while(|w| matches!(*w, "public" | "meta" | "private"))
            .collect();
        match words.as_slice() {
            ["import", name, ..] => out.push((*name).to_string()),
            _ => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gate::StructureRefusal;

    /// A complete, valid rules text: one root file, one recursive directory, one source limit.
    const RULES: &str = r#"
schema = 1
max_lines = 3
max_dir_files = 2
source = ["rs", "py"]
generated = ["gen/out.rs"]

[[dir]]
path = "."
files = ["Cargo.toml"]

[[dir]]
path = "src"
recursive = true
files = ["*.rs", "*.py"]

[[dir]]
path = "tools"
holds = "tools, with a list"
files = [
  "*.sh",
  "*.py",
]

[python]
frozen = ["tools/old.py"]
"#;

    fn rules_with(extra: &str) -> Rules {
        let mut text = RULES.to_string();
        text.push_str(extra);
        parse_rules(&text).expect("fixture parses")
    }

    fn files(list: &[(&str, usize)]) -> Vec<(String, usize)> {
        list.iter().map(|(p, n)| (p.to_string(), *n)).collect()
    }

    fn rules_of(refusals: &[StructureRefusal]) -> Vec<(&'static str, String)> {
        refusals.iter().map(|r| (r.rule, r.path.clone())).collect()
    }

    /// A file list that satisfies every rule of `RULES`.
    fn clean() -> Vec<(&'static str, usize)> {
        vec![
            ("Cargo.toml", 1),
            ("src/lib.rs", 2),
            ("src/a/b.rs", 3),
            ("tools/run.sh", 0),
            ("tools/old.py", 2),
        ]
    }

    #[test]
    fn control_accept() {
        let rules = rules_with("");
        let refusals = structure_verdict(&rules, &files(&clean()));
        assert!(refusals.is_empty(), "{refusals:?}");
    }

    #[test]
    fn control_unadmitted() {
        let rules = rules_with("");
        let mut list = clean();
        list.push(("examples/new_probe.rs", 1));
        let refusals = structure_verdict(&rules, &files(&list));
        assert_eq!(
            rules_of(&refusals),
            vec![("unadmitted", "examples/new_probe.rs".to_string())]
        );
        assert!(refusals[0].detail.contains("no [[dir]] admits"));
    }

    #[test]
    fn control_empty_dir_entry() {
        let rules = rules_with("");
        let list: Vec<(&str, usize)> = clean()
            .into_iter()
            .filter(|(p, _)| !p.starts_with("tools/"))
            .collect();
        let refusals = structure_verdict(&rules, &files(&list));
        assert_eq!(
            rules_of(&refusals),
            vec![
                ("empty-dir-entry", "tools".to_string()),
                ("python-stale", "tools/old.py".to_string()),
            ]
        );
    }

    #[test]
    fn control_max_lines() {
        let rules = rules_with("");
        let mut list = clean();
        list.push(("src/big.rs", 4));
        let refusals = structure_verdict(&rules, &files(&list));
        assert_eq!(
            rules_of(&refusals),
            vec![("max_lines", "src/big.rs".to_string())]
        );
    }

    #[test]
    fn control_max_dir_files() {
        let rules = rules_with("");
        let mut list = clean();
        list.push(("src/c.rs", 1));
        list.push(("src/d.rs", 1));
        let refusals = structure_verdict(&rules, &files(&list));
        assert_eq!(
            rules_of(&refusals),
            vec![("max_dir_files", "src".to_string())]
        );
    }

    #[test]
    fn control_stale_debt() {
        let rules = rules_with(
            "\n[[debt]]\nrule = \"max_lines\"\npath = \"src/lib.rs\"\nrow = \"STRUCT-SPLIT\"\n",
        );
        let refusals = structure_verdict(&rules, &files(&clean()));
        assert_eq!(
            rules_of(&refusals),
            vec![("stale-debt", "src/lib.rs".to_string())]
        );
    }

    #[test]
    fn control_python_unfrozen() {
        let rules = rules_with("");
        let mut list = clean();
        list.push(("tools/new.py", 1));
        let refusals = structure_verdict(&rules, &files(&list));
        assert_eq!(
            rules_of(&refusals),
            vec![("python-unfrozen", "tools/new.py".to_string())]
        );
    }

    #[test]
    fn control_python_stale() {
        let rules = rules_with("");
        let list: Vec<(&str, usize)> = clean()
            .into_iter()
            .filter(|(p, _)| *p != "tools/old.py")
            .collect();
        let refusals = structure_verdict(&rules, &files(&list));
        assert_eq!(
            rules_of(&refusals),
            vec![("python-stale", "tools/old.py".to_string())]
        );
    }

    #[test]
    fn control_parse_error() {
        let bad = RULES.replace("max_lines = 3", "max_lines = 3.5");
        let err = parse_rules(&bad).expect_err("a float is outside the subset");
        assert!(err.starts_with("3:"), "{err}");
        assert!(err.contains("expected end of line"), "{err}");
        assert!(parse_rules(
            "schema = 1\nmax_lines = 1\nmax_dir_files = 1\nsource = []\n[other]\n"
        )
        .is_err());
    }

    /// A `[lean]` table over `proof/corpus` with the given debt list, appended to `RULES`.
    fn lean_rules(debt: &[&str]) -> Rules {
        let list: Vec<String> = debt.iter().map(|d| format!("\"{d}\"")).collect();
        rules_with(&format!(
            "\n[lean]\nroot = \"proof/corpus\"\naudit_modules = [\"AxiomAudit\"]\norphan_debt = [{}]\n",
            list.join(", ")
        ))
    }

    /// Runs the Lean rule over `(path, imports)` pairs, every pair tracked; returns `(rule, path)`.
    fn lean_run(debt: &[&str], mods: &[(&str, &[&str])]) -> Vec<(&'static str, String)> {
        let tracked: Vec<String> = mods.iter().map(|(p, _)| p.to_string()).collect();
        let modules: Vec<LeanModule> = mods
            .iter()
            .map(|(p, imps)| LeanModule {
                path: p.to_string(),
                imports: imps.iter().map(|s| s.to_string()).collect(),
            })
            .collect();
        let (refusals, _) = lean_verdict(&lean_rules(debt), &tracked, &modules);
        rules_of(&refusals)
    }

    const BRIDGE: &str = "proof/corpus/Word/Bridge.lean";
    const UTIL: &str = "proof/corpus/Lib/Util.lean";
    const STRAY: &str = "proof/corpus/Stray/New.lean";

    #[test]
    fn control_lean_accept() {
        // Bridge is linked by ApeironWord; Util is support, imported by the linked Bridge.
        let mods: &[(&str, &[&str])] = &[(BRIDGE, &["ApeironWord.Basic", "Lib.Util"]), (UTIL, &[])];
        assert!(lean_run(&[], mods).is_empty());
    }

    #[test]
    fn control_lean_support_accepted() {
        // Negative control: without the link from Bridge, Util is an orphan and is refused.
        let mods: &[(&str, &[&str])] = &[(BRIDGE, &["Lib.Other"]), (UTIL, &[])];
        assert_eq!(
            lean_run(&[], mods),
            vec![("orphan", BRIDGE.to_string()), ("orphan", UTIL.to_string())]
        );
        let linked: &[(&str, &[&str])] = &[(BRIDGE, &["ApeironWord", "Lib.Util"]), (UTIL, &[])];
        assert!(lean_run(&[], linked).is_empty());
    }

    #[test]
    fn control_lean_orphan_refused() {
        let mods: &[(&str, &[&str])] = &[(STRAY, &["Mathlib.Tactic"])];
        assert_eq!(lean_run(&[], mods), vec![("orphan", STRAY.to_string())]);
    }

    #[test]
    fn control_lean_debt_accepted() {
        let mods: &[(&str, &[&str])] = &[(STRAY, &["Mathlib.Tactic"])];
        assert!(lean_run(&[STRAY], mods).is_empty());
    }

    #[test]
    fn control_lean_stale_debt() {
        // Debt on a module that is now linked is refused; the list only shrinks.
        let mods: &[(&str, &[&str])] = &[(BRIDGE, &["ApeironWord"])];
        assert_eq!(
            lean_run(&[BRIDGE], mods),
            vec![("orphan-stale", BRIDGE.to_string())]
        );
    }

    #[test]
    fn control_lean_audit_only_refused() {
        // AxiomAudit imports the extraction and Dead, yet it links and supports nothing: Dead is an
        // orphan. The aggregator itself is exempt.
        let dead = "proof/corpus/Lib/Dead.lean";
        let mods: &[(&str, &[&str])] = &[
            (
                "proof/corpus/AxiomAudit.lean",
                &["ApeironWord.Basic", "Lib.Dead"],
            ),
            (dead, &[]),
        ];
        assert_eq!(lean_run(&[], mods), vec![("orphan", dead.to_string())]);
    }

    #[test]
    fn control_lean_imports_header() {
        let src = "module\n\n/-! doc\n  import Fake\n-/\npublic import ApeironWord.Basic -- note\n\
                   meta import Lib.Util\nimport Lib.Two\n\ndef x := 1\nimport Late\n";
        assert_eq!(
            lean_imports(src),
            vec!["ApeironWord.Basic", "Lib.Util", "Lib.Two"]
        );
    }
}
