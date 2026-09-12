//! File walking, hashing, revision metadata, summary computation and output.

use crate::model::*;
use crate::targets::TargetConfig;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

/// Directory names never descended into.
pub const DEFAULT_SKIP_DIRS: &[&str] = &[
    "target",
    ".git",
    "node_modules",
    ".cargo",
    "vendor",
    ".lake",
    "build",
    "dist",
    ".claude",
    "worktrees",
];

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

/// SHA-256 over the tool's own source tree (manifest + `src` + `tests`), sorted
/// by path. Reproducible: identical sources produce an identical digest.
pub fn tool_source_sha256() -> Option<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    if !root.is_dir() {
        return None;
    }
    let mut files: Vec<PathBuf> = Vec::new();
    files.push(root.join("Cargo.toml"));
    for sub in ["src", "tests"] {
        let dir = root.join(sub);
        if dir.is_dir() {
            for entry in WalkDir::new(&dir).follow_links(false) {
                let Ok(e) = entry else { continue };
                if e.file_type().is_file() {
                    let p = e.path();
                    if p.extension().and_then(|x| x.to_str()) == Some("rs")
                        || p.file_name().and_then(|x| x.to_str()) == Some("Cargo.toml")
                    {
                        files.push(p.to_path_buf());
                    }
                }
            }
        }
    }
    files.sort();
    files.dedup();
    let mut h = Sha256::new();
    for p in files {
        let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy();
        h.update(rel.as_bytes());
        h.update([0u8]);
        if let Ok(bytes) = fs::read(&p) {
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        }
    }
    Some(hex::encode(h.finalize()))
}

/// SHA-256 of the running executable, so a report names the exact binary.
pub fn tool_binary_sha256() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let bytes = fs::read(exe).ok()?;
    Some(sha256_hex(&bytes))
}

/// Returns `(relative_path, absolute_path)` for every `.rs` file under `root`.
pub fn walk_rust_files(
    root: &Path,
    extra_skip: &[String],
) -> Result<Vec<(String, PathBuf)>, String> {
    let mut out = Vec::new();
    let mut skip: HashSet<&str> = DEFAULT_SKIP_DIRS.iter().copied().collect();
    for s in extra_skip {
        skip.insert(s.as_str());
    }
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            if e.file_type().is_dir() {
                !skip.contains(name.as_ref())
                    && !(name.starts_with('.') && name.len() > 1 && name != ".")
            } else {
                true
            }
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let rel = p
            .strip_prefix(root)
            .map(|r| r.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| p.to_string_lossy().to_string());
        out.push((rel, p.to_path_buf()));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

pub fn git_revision(root: &Path) -> RevisionMeta {
    let run = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let head = run(&["rev-parse", "HEAD"]);
    let branch = run(&["rev-parse", "--abbrev-ref", "HEAD"]);
    let describe = run(&["describe", "--always", "--dirty", "--tags"]);
    let dirty = run(&["status", "--porcelain"]).map(|s| {
        if s.is_empty() {
            0
        } else {
            s.lines().count() as u64
        }
    });
    RevisionMeta {
        git_head: head,
        git_branch: branch,
        git_describe: describe,
        git_status_porcelain_lines: dirty,
        working_tree_note: "hashes and line numbers are of the working tree that was walked, \
which may include uncommitted edits; HEAD is recorded for reference only"
            .to_string(),
    }
}

/// Formats the current time as `YYYY-MM-DDThh:mm:ssZ` without a date crate.
pub fn now_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

#[derive(Serialize)]
pub struct InventoryDoc {
    pub target_crates: Vec<String>,
    pub types: Vec<String>,
    pub methods: Vec<String>,
    pub fns: Vec<String>,
    pub macros: Vec<String>,
    pub fields: BTreeMap<String, BTreeMap<String, String>>,
    pub variants: BTreeMap<String, BTreeSet<String>>,
    pub tuple_structs: Vec<String>,
    pub traits: Vec<String>,
    pub trait_methods: BTreeMap<String, BTreeSet<String>>,
}

pub fn inventory_doc(cfg: &TargetConfig) -> InventoryDoc {
    InventoryDoc {
        target_crates: cfg.crates.clone(),
        types: cfg.types.iter().cloned().collect(),
        methods: cfg.methods.iter().cloned().collect(),
        fns: cfg.fns.iter().cloned().collect(),
        macros: cfg.macros.iter().cloned().collect(),
        fields: cfg
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        variants: cfg
            .variants
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        tuple_structs: cfg.tuple_structs.iter().cloned().collect(),
        traits: cfg.traits.iter().cloned().collect(),
        trait_methods: cfg
            .trait_methods
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compute_summary(
    root: &str,
    files: &[FileMeta],
    calls: &[CallRecord],
    imports: &[ImportRecord],
    macros: &[MacroRecord],
    _method_calls_total: u64,
    other_method_total: u64,
    other_method_by_name: &BTreeMap<String, u64>,
    unqualified_macro_total: u64,
    unqualified_macro_by_name: &BTreeMap<String, u64>,
) -> Summary {
    let mut s = Summary {
        schema_version: SCHEMA_VERSION,
        root: root.to_string(),
        ..Summary::default()
    };
    s.files_total = files.len() as u64;
    s.files_parsed = files.iter().filter(|f| f.parsed).count() as u64;
    s.files_parse_error = files.iter().filter(|f| !f.parsed).count() as u64;
    s.files_target_crate = files.iter().filter(|f| f.is_target_crate_file).count() as u64;
    s.records_total = calls.len() as u64;
    s.calls_total = calls.iter().filter(|c| c.is_call).count() as u64;
    s.direct_path_calls = calls
        .iter()
        .filter(|c| {
            c.category == crate::census::CAT_DIRECT_CALL
                || c.category == crate::census::CAT_ASSOC_CALL
        })
        .count() as u64;
    s.method_calls_attributed = calls
        .iter()
        .filter(|c| c.category == crate::census::CAT_METHOD_ATTRIBUTED)
        .count() as u64;
    s.unattributed_method_candidates = calls
        .iter()
        .filter(|c| c.category == crate::census::CAT_METHOD_CANDIDATE)
        .count() as u64;
    for c in calls
        .iter()
        .filter(|c| c.category == crate::census::CAT_METHOD_CANDIDATE)
    {
        if let Some(m) = &c.method {
            *s.unattributed_method_candidates_by_name
                .entry(m.clone())
                .or_insert(0) += 1;
        }
    }
    s.ambiguous_methods = calls
        .iter()
        .filter(|c| c.category == crate::census::CAT_METHOD_AMBIGUOUS)
        .count() as u64;
    s.heuristic_method_calls = calls
        .iter()
        .filter(|c| c.category == crate::census::CAT_METHOD_TRAIT_HEURISTIC)
        .count() as u64;
    s.dynamic_dispatch_methods = calls
        .iter()
        .filter(|c| c.category == crate::census::CAT_METHOD_DYNAMIC)
        .count() as u64;
    s.path_refs = calls.iter().filter(|c| c.record_kind == "path_ref").count() as u64;
    s.attributed_calls = calls
        .iter()
        .filter(|c| c.is_call && (c.confidence == "high" || c.confidence == "medium"))
        .count() as u64;
    s.attributed_high = calls
        .iter()
        .filter(|c| c.is_call && c.confidence == "high")
        .count() as u64;
    s.attributed_medium = calls
        .iter()
        .filter(|c| c.is_call && c.confidence == "medium")
        .count() as u64;

    s.target_imports = imports
        .iter()
        .filter(|i| i.is_target && !i.is_reexport)
        .count() as u64;
    s.target_reexports = imports
        .iter()
        .filter(|i| i.is_target && i.is_reexport)
        .count() as u64;

    s.macro_sites = macros
        .iter()
        .filter(|m| m.kind == "macro_invocation")
        .count() as u64;
    s.derive_sites = macros.iter().filter(|m| m.kind == "derive").count() as u64;
    s.attribute_macro_sites = macros
        .iter()
        .filter(|m| m.kind == "attribute_macro")
        .count() as u64;
    s.macro_rules_defs = macros
        .iter()
        .filter(|m| m.kind == "macro_rules_def")
        .count() as u64;
    s.unqualified_macro_invocations = unqualified_macro_total;
    s.unqualified_macro_by_name = unqualified_macro_by_name.clone();

    s.other_method_calls_total = other_method_total;
    s.other_method_calls_by_name = other_method_by_name.clone();

    let mut callees = BTreeSet::new();
    let mut callers = BTreeSet::new();
    let mut symbols = BTreeSet::new();
    for c in calls {
        if c.is_call && (c.confidence == "high" || c.confidence == "medium") {
            if let Some(sym) = &c.target_symbol {
                callees.insert(sym.clone());
                symbols.insert(sym.clone());
            } else if let Some(p) = &c.callee_path {
                callees.insert(p.clone());
            }
            callers.insert(format!("{}:{}", c.file, c.caller.qualified));
        }
    }
    s.distinct_callees = callees.len() as u64;
    s.distinct_callers = callers.len() as u64;
    s.distinct_target_symbols = symbols.len() as u64;

    for c in calls {
        *s.by_category.entry(c.category.clone()).or_insert(0) += 1;
        if c.is_call {
            *s.by_source_location
                .entry(c.source_location.clone())
                .or_insert(0) += 1;
            *s.by_operation_kind
                .entry(c.operation_kind.clone())
                .or_insert(0) += 1;
            let crate_name = c
                .crate_name
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string());
            let cs = s.by_crate.entry(crate_name).or_default();
            cs.records += 1;
            if c.confidence == "high" || c.confidence == "medium" {
                cs.attributed += 1;
            } else {
                cs.unattributed += 1;
            }
            let sub = c
                .module_path
                .split("::")
                .next()
                .filter(|x| !x.is_empty())
                .unwrap_or("<root>")
                .to_string();
            *s.by_subsystem.entry(sub).or_insert(0) += 1;
            if let Some(t) = &c.target_symbol {
                if c.confidence == "high" || c.confidence == "medium" {
                    *s.by_target_symbol.entry(t.clone()).or_insert(0) += 1;
                }
            }
            if let Some(t) = &c.target_type {
                if c.confidence == "high" || c.confidence == "medium" {
                    *s.by_target_type.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    for f in files {
        let cs = s
            .by_crate
            .entry(
                f.crate_name
                    .clone()
                    .unwrap_or_else(|| "<unknown>".to_string()),
            )
            .or_default();
        cs.files += 1;
    }
    for i in imports {
        if i.is_target {
            let cs = s
                .by_crate
                .entry(
                    i.crate_name
                        .clone()
                        .unwrap_or_else(|| "<unknown>".to_string()),
                )
                .or_default();
            cs.imports += 1;
            if i.is_reexport {
                cs.reexports += 1;
            }
        }
    }
    for m in macros {
        let cs = s
            .by_crate
            .entry(
                m.crate_name
                    .clone()
                    .unwrap_or_else(|| "<unknown>".to_string()),
            )
            .or_default();
        cs.macros += 1;
    }

    s.attribution_note = format!(
        "{} of {} recorded call expressions are SYNTACTICALLY ATTRIBUTED (high/medium). \
This is a count of source syntax, NOT a count of executed calls, NOT a mathematically \
assured lower bound, and NOT a completeness proof. {} method calls are candidates \
(unknown/ambiguous receiver), {} are labelled heuristic attributions via an imported \
target trait (receiver type unknown), {} are dynamic dispatch, and {} further method \
calls had a known non-target receiver or were otherwise unclassified and are only \
aggregated by name. Macro bodies are not expanded.",
        s.attributed_calls,
        s.calls_total,
        s.unattributed_method_candidates + s.ambiguous_methods,
        s.heuristic_method_calls,
        s.dynamic_dispatch_methods,
        s.other_method_calls_total
    );
    s.limitations = crate::LIMITATIONS.iter().map(|s| s.to_string()).collect();
    s
}

fn tsv_escape(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

pub struct WriteInputs<'a> {
    pub output: &'a Path,
    pub meta: &'a Meta,
    pub files: &'a [FileMeta],
    pub calls: &'a [CallRecord],
    pub imports: &'a [ImportRecord],
    pub macros: &'a [MacroRecord],
    pub summary: &'a Summary,
    pub inventory: &'a InventoryDoc,
}

pub fn write_all(inp: &WriteInputs) -> Result<(), String> {
    fs::create_dir_all(inp.output).map_err(|e| format!("create output dir: {e}"))?;
    write_json(&inp.output.join("meta.json"), inp.meta)?;
    write_json(&inp.output.join("files.json"), inp.files)?;
    write_json(&inp.output.join("inventory.json"), inp.inventory)?;
    write_json(&inp.output.join("records.json"), inp.calls)?;
    write_json(&inp.output.join("imports.json"), inp.imports)?;
    write_json(&inp.output.join("macros.json"), inp.macros)?;
    write_json(&inp.output.join("summary.json"), inp.summary)?;
    write_summary_md(&inp.output.join("summary.md"), inp.summary)?;

    let attributed: Vec<&CallRecord> = inp
        .calls
        .iter()
        .filter(|c| c.is_call && (c.confidence == "high" || c.confidence == "medium"))
        .collect();
    write_calls_tsv(&inp.output.join("calls.tsv"), &attributed)?;

    let unattributed: Vec<&CallRecord> = inp
        .calls
        .iter()
        .filter(|c| {
            c.record_kind == "method"
                && matches!(
                    c.confidence.as_str(),
                    "unattributed" | "ambiguous" | "dynamic" | "heuristic"
                )
        })
        .collect();
    write_unattributed_tsv(&inp.output.join("unattributed_methods.tsv"), &unattributed)?;

    write_imports_tsv(&inp.output.join("imports.tsv"), inp.imports)?;
    write_macros_tsv(&inp.output.join("macros.tsv"), inp.macros)?;
    Ok(())
}

fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<(), String> {
    let f = fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut w = BufWriter::new(f);
    serde_json::to_writer_pretty(&mut w, value)
        .map_err(|e| format!("serialize {}: {e}", path.display()))?;
    w.write_all(b"\n")
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(())
}

fn write_calls_tsv(path: &Path, calls: &[&CallRecord]) -> Result<(), String> {
    let mut w = new_tsv(path)?;
    writeln!(
        w,
        "id\tfile\tline\tcolumn\tend_line\tend_column\tcategory\toperation_kind\tconfidence\tsource_location\treachability\tcrate\tmodule\tcaller_kind\tcaller\timpl_type\tis_test\tcallee_text\tcallee_path\tmethod\treceiver_text\treceiver_type\ttarget_crate\ttarget_type\ttarget_symbol\tattribution_basis\tcfg"
    )
    .map_err(err)?;
    for c in calls {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            c.id,
            tsv_escape(&c.file),
            c.span.line,
            c.span.column,
            c.span.end_line,
            c.span.end_column,
            c.category,
            c.operation_kind,
            c.confidence,
            c.source_location,
            c.reachability,
            c.crate_name.as_deref().unwrap_or(""),
            c.module_path,
            c.caller.kind,
            c.caller.qualified,
            c.caller.impl_type.as_deref().unwrap_or(""),
            c.is_test_context,
            tsv_escape(&c.callee_text),
            c.callee_path.as_deref().unwrap_or(""),
            c.method.as_deref().unwrap_or(""),
            tsv_escape(c.receiver_text.as_deref().unwrap_or("")),
            tsv_escape(c.receiver_type.as_deref().unwrap_or("")),
            c.target_crate.as_deref().unwrap_or(""),
            c.target_type.as_deref().unwrap_or(""),
            c.target_symbol.as_deref().unwrap_or(""),
            tsv_escape(&c.attribution_basis),
            tsv_escape(&c.cfg.join(";")),
        )
        .map_err(err)?;
    }
    Ok(())
}

fn write_unattributed_tsv(path: &Path, calls: &[&CallRecord]) -> Result<(), String> {
    let mut w = new_tsv(path)?;
    writeln!(
        w,
        "id\tfile\tline\tcolumn\tcategory\tconfidence\tsource_location\tcrate\tmodule\tcaller\tmethod\treceiver_text\treceiver_type\tcandidate\tdynamic\tattribution_basis\tcfg"
    )
    .map_err(err)?;
    for c in calls {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            c.id,
            tsv_escape(&c.file),
            c.span.line,
            c.span.column,
            c.category,
            c.confidence,
            c.source_location,
            c.crate_name.as_deref().unwrap_or(""),
            c.module_path,
            c.caller.qualified,
            c.method.as_deref().unwrap_or(""),
            tsv_escape(c.receiver_text.as_deref().unwrap_or("")),
            tsv_escape(c.receiver_type.as_deref().unwrap_or("")),
            c.candidate,
            c.dynamic_dispatch,
            tsv_escape(&c.attribution_basis),
            tsv_escape(&c.cfg.join(";")),
        )
        .map_err(err)?;
    }
    Ok(())
}

fn write_imports_tsv(path: &Path, imports: &[ImportRecord]) -> Result<(), String> {
    let mut w = new_tsv(path)?;
    writeln!(
        w,
        "id\tfile\tline\tcolumn\tsource_location\tcrate\tmodule\ttext\tpath\troot\talias\tis_glob\tis_reexport\tvisibility\tis_target\tconfidence\tattribution_basis"
    )
    .map_err(err)?;
    for i in imports {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            i.id,
            tsv_escape(&i.file),
            i.line,
            i.column,
            i.source_location,
            i.crate_name.as_deref().unwrap_or(""),
            i.module_path,
            tsv_escape(&i.text),
            i.path,
            i.root,
            i.alias.as_deref().unwrap_or(""),
            i.is_glob,
            i.is_reexport,
            i.visibility,
            i.is_target,
            i.confidence,
            tsv_escape(&i.attribution_basis),
        )
        .map_err(err)?;
    }
    Ok(())
}

fn write_macros_tsv(path: &Path, macros: &[MacroRecord]) -> Result<(), String> {
    let mut w = new_tsv(path)?;
    writeln!(
        w,
        "id\tfile\tline\tcolumn\tkind\tsource_location\tcrate\tmodule\tcaller\tname\tpath\tis_target\tconfidence\tunexpanded\tattribution_basis\tcfg"
    )
    .map_err(err)?;
    for m in macros {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            m.id,
            tsv_escape(&m.file),
            m.line,
            m.column,
            m.kind,
            m.source_location,
            m.crate_name.as_deref().unwrap_or(""),
            m.module_path,
            m.caller.qualified,
            m.name,
            m.path.as_deref().unwrap_or(""),
            m.is_target,
            m.confidence,
            m.unexpanded,
            tsv_escape(&m.attribution_basis),
            tsv_escape(&m.cfg.join(";")),
        )
        .map_err(err)?;
    }
    Ok(())
}

fn new_tsv(path: &Path) -> Result<BufWriter<fs::File>, String> {
    let f = fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    Ok(BufWriter::new(f))
}

fn err(e: std::io::Error) -> String {
    format!("write tsv: {e}")
}

fn write_summary_md(path: &Path, s: &Summary) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("# Call census summary (syntactic attribution)\n\n");
    out.push_str(&format!("- schema_version: {}\n", s.schema_version));
    out.push_str(&format!("- root: `{}`\n", s.root));
    out.push_str(&format!(
        "- files: {} total, {} parsed, {} parse errors, {} in target crates\n",
        s.files_total, s.files_parsed, s.files_parse_error, s.files_target_crate
    ));
    out.push_str(&format!(
        "- recorded call expressions: {} (syntactically attributed: {}, high {}, medium {})\n",
        s.calls_total, s.attributed_calls, s.attributed_high, s.attributed_medium
    ));
    out.push_str(&format!(
        "- direct/associated path calls: {}, attributed method calls: {}\n",
        s.direct_path_calls, s.method_calls_attributed
    ));
    out.push_str(&format!(
        "- candidates: {}, ambiguous: {}, heuristic (imported target trait): {}, dynamic: {}, other (aggregated): {}\n",
        s.unattributed_method_candidates, s.ambiguous_methods, s.heuristic_method_calls,
        s.dynamic_dispatch_methods, s.other_method_calls_total
    ));
    out.push_str(&format!(
        "- target imports: {}, target re-exports: {}\n",
        s.target_imports, s.target_reexports
    ));
    out.push_str(&format!(
        "- macro sites: {}, derives: {}, attribute macros: {}, macro_rules defs: {}, unexpanded other macro invocations: {}\n",
        s.macro_sites, s.derive_sites, s.attribute_macro_sites, s.macro_rules_defs,
        s.unqualified_macro_invocations
    ));
    out.push_str(&format!(
        "- distinct callees: {}, distinct callers: {}, distinct target symbols: {}\n",
        s.distinct_callees, s.distinct_callers, s.distinct_target_symbols
    ));
    out.push_str("\n## What the attributed count means\n\n");
    out.push_str(&format!("{}\n", s.attribution_note));
    out.push_str("\n## By category\n\n");
    for (k, v) in &s.by_category {
        out.push_str(&format!("- {k}: {v}\n"));
    }
    out.push_str("\n## By operation kind\n\n");
    for (k, v) in &s.by_operation_kind {
        out.push_str(&format!("- {k}: {v}\n"));
    }
    out.push_str("\n## By source location (not reachability)\n\n");
    for (k, v) in &s.by_source_location {
        out.push_str(&format!("- {k}: {v}\n"));
    }
    out.push_str("\n## By crate\n\n");
    for (k, v) in &s.by_crate {
        out.push_str(&format!(
            "- {k}: files {}, records {}, attributed {}, unattributed {}, imports {}, reexports {}, macros {}\n",
            v.files, v.records, v.attributed, v.unattributed, v.imports, v.reexports, v.macros
        ));
    }
    out.push_str("\n## By subsystem (first module segment)\n\n");
    for (k, v) in &s.by_subsystem {
        out.push_str(&format!("- {k}: {v}\n"));
    }
    out.push_str("\n## By target symbol (attributed calls)\n\n");
    for (k, v) in &s.by_target_symbol {
        out.push_str(&format!("- {k}: {v}\n"));
    }
    out.push_str("\n## Candidate method calls by name (top 50)\n\n");
    let mut cand: Vec<(&String, &u64)> = s.unattributed_method_candidates_by_name.iter().collect();
    cand.sort_by(|a, b| b.1.cmp(a.1));
    for (k, n) in cand.into_iter().take(50) {
        out.push_str(&format!("- {k}: {n}\n"));
    }
    out.push_str("\n## Limitations\n\n");
    for l in &s.limitations {
        out.push_str(&format!("- {l}\n"));
    }
    fs::write(path, out).map_err(|e| format!("write summary.md: {e}"))?;
    Ok(())
}
