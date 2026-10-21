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
//!
//! The contract is `docs/contracts/one-path.toml`: the declared forward bodies, and one `[[debt]]`
//! row per accepted finding naming an open board ticket. A finding no row carries, a row no finding
//! matches, and a consumer count that moved are each a refusal: debt only shrinks, and a new caller
//! of a debt body is a new refusal. There is no mode that writes the contract.

mod clones;
mod graph;
mod load;
mod rules;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use crate::cli::{Verdict, EXIT_ACCEPTED, EXIT_ENVIRONMENT, EXIT_REFUSED};
use crate::structure::{as_list, as_str, as_u64, field, need, parse_tables, Kind};

pub(crate) const USAGE: &str =
    "usage: gate onepath (--check | --report) [--staged] [--root DIR]\n  \
  --check   refuse a finding the contract does not carry, a stale row, or a moved count\n  \
  --report  print every finding with its contract status; exit 0\n  \
  --staged  read sources and contract from the git index\n  \
  --root    repository root (default .)";

const CONTRACT: &str = "docs/contracts/one-path.toml";
/// Rules whose count is a consumer count: a row states it, so a new caller moves it.
const COUNTED: [&str; 3] = ["forward", "backward", "descent"];

pub(crate) struct Debt {
    rule: String,
    items: Vec<String>,
    consumers: Option<u64>,
    ticket: String,
    at: usize,
}

pub(crate) struct Contract {
    forward: Vec<String>,
    debts: Vec<Debt>,
}

pub(crate) fn parse_contract(src: &str) -> Result<Contract, String> {
    let tables = parse_tables(src, &[(false, "forward")])?;
    let mut c = Contract {
        forward: Vec::new(),
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
}

/// Compares the findings with the contract.
///
/// Cost: time O(F log F + D log D) over F findings and D rows; space O(F + D).
pub(crate) fn compare(root: &Path, a: &rules::Analysis, c: &Contract) -> Outcome {
    let mut refusals = Vec::new();
    let mut lines = Vec::new();
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
        if f.rule == "forward" && declared.contains(f.items[0].as_str()) {
            lines.push(format!("declared\t{}\t{}\t{}", f.rule, f.at, f.items[0]));
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
    }
}

fn usage(m: &str) -> ExitCode {
    eprintln!("gate onepath: {m}\n{USAGE}");
    ExitCode::from(EXIT_ENVIRONMENT)
}

fn read_contract(root: &Path, staged: bool) -> Result<String, String> {
    if staged {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["show", &format!(":{CONTRACT}")])
            .output()
            .map_err(|e| format!("git show: {e}"))?;
        if !out.status.success() {
            return Err(format!("{CONTRACT} is not in the index"));
        }
        String::from_utf8(out.stdout).map_err(|e| format!("{CONTRACT}: {e}"))
    } else {
        std::fs::read_to_string(root.join(CONTRACT)).map_err(|e| format!("{CONTRACT}: {e}"))
    }
}

/// Loads, analyses and compares; `Err` is an environment failure.
pub(crate) fn run(
    root: &Path,
    staged: bool,
) -> Result<(rules::Analysis, Outcome, Vec<String>), String> {
    let t0 = std::time::Instant::now();
    let sources = if staged {
        load::staged_sources(root)?
    } else {
        load::standing_sources(root)?
    };
    let contract =
        parse_contract(&read_contract(root, staged)?).map_err(|e| format!("{CONTRACT}: {e}"))?;
    let tree = load::load(&sources);
    if tree.fns.is_empty() {
        return Err("no fn loaded: the gate refuses to pass on nothing".into());
    }
    let t_load = t0.elapsed().as_millis();
    let idx = graph::index(&tree);
    let edges = graph::edges(&tree, &idx);
    let t_graph = t0.elapsed().as_millis();
    let mut analysis = rules::analyse(&tree, &idx, &edges);
    let t_rules = t0.elapsed().as_millis();
    let outcome = compare(root, &analysis, &contract);
    analysis.stats.push(("ms_load", t_load as usize));
    analysis
        .stats
        .push(("ms_graph", (t_graph - t_load) as usize));
    analysis
        .stats
        .push(("ms_rules", (t_rules - t_graph) as usize));
    Ok((analysis, outcome, tree.errors))
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
