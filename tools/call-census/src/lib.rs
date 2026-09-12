//! `call-census` — a reproducible, syntax-level call-site census for Rust
//! consumers of a target crate (default: `hashtrinity`).
//!
//! The tool is deliberately honest about its limits: it is an AST census, not a
//! compiler. It performs **syntactic attribution**, never resolution. See
//! [`LIMITATIONS`] and the `attribution_basis`/`confidence` fields on every
//! record.

pub mod census;
pub mod model;
pub mod output;
pub mod targets;

use census::CensusOutput;
use model::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use targets::TargetConfig;

pub const TOOL_NAME: &str = "call-census";
pub const TOOL_VERSION: &str = "0.2.0";
pub const SYN_VERSION: &str = "2.0.119";
pub const PROC_MACRO2_VERSION: &str = "1.0.107";

/// The semantic limits every report carries.
pub const LIMITATIONS: &[&str] = &[
    "This is a syntax (syn) census. It performs no name resolution, no type inference, no generic instantiation, no trait selection and no macro expansion.",
    "'Attributed' means a syntactic match, not compiler resolution. The attributed count is NOT a mathematically assured lower bound, NOT a count of executed calls, and NOT a completeness proof.",
    "Receiver types are inferred only from explicit annotations, function parameters, `impl` self types, struct field types (local or target inventory), narrow constructor calls, simple assignments, and local type aliases whose right-hand side attributes to the target. Everything else is a candidate or aggregated.",
    "A `let` (or `for`) binding whose initializer type is unknown deliberately shadows any older binding of the same name; the stale type is never retained.",
    "Method names are never assumed to belong to a target type. `.get` on an unknown receiver is recorded as a candidate; `.get` on a receiver with a known non-target type is only aggregated.",
    "A field of a known non-target type is treated as non-target; the container's type is never used as the field's type.",
    "Candidate method calls are only emitted in files that import a target crate or whose source text names one. This is a recall filter, never an attribution.",
    "Macro bodies, `macro_rules!` bodies and derive output are NOT expanded. Calls they generate are absent; every macro site is flagged `unexpanded`. This limitation is separate from, and not addressed by, the syntax census.",
    "`cfg` predicates are recorded verbatim from the item and every enclosing module/impl, but are not evaluated. Items behind `cfg(test)`, `cfg(feature=...)` or platform cfgs are included and labelled, and may not be part of a given build.",
    "`source_location` is where the text lives (`src`, `src_bin`, `tests`, `benches`, `examples`, `build`, `other`). `reachability` is always `not_determined`: the tool does not build a module graph and never asserts that a file is compiled or reachable.",
    "Files belonging to a configured target crate are inventoried (types, inherent/trait methods, variants, fields, derives, re-exports) but are not censused as consumers. Test-only items (`#[cfg(test)]`, `#[test]`) are excluded from the inventory.",
    "Import scope is tracked per `mod` with a module-item pre-scan; block-level `use` is registered at module scope, and `mod foo;` file linkage is left to the directory walk, so cross-file re-export chains are not followed.",
    "Enum-variant and tuple-struct constructors are recognised from the target inventory, never from capitalisation. A target path call with no inventory match is `unclassified`.",
    "Glob imports of a target crate yield medium confidence, never high. A non-target glob or non-target explicit import with the same name blocks a target attribution.",
    "Only call expressions and method calls are recorded; a function-pointer reference such as `let f = Table::get;` is not a call and is not recorded.",
    "Comments and string literals cannot appear because the source is parsed to an AST; this is by construction, not a filter.",
    "Line and column numbers come from proc-macro2 `span-locations`; they are accurate outside a procedural-macro context.",
];

#[derive(Debug, Clone)]
pub struct Config {
    pub root: PathBuf,
    pub output: PathBuf,
    pub target_crates: Vec<String>,
    pub target_types: Vec<String>,
    pub target_methods: Vec<String>,
    pub target_src: Vec<PathBuf>,
    pub extra_skip: Vec<String>,
}

impl Config {
    pub fn new(root: PathBuf, output: PathBuf) -> Self {
        Config {
            root,
            output,
            target_crates: Vec::new(),
            target_types: Vec::new(),
            target_methods: Vec::new(),
            target_src: Vec::new(),
            extra_skip: Vec::new(),
        }
    }
}

#[derive(Default)]
struct Totals {
    calls: Vec<CallRecord>,
    imports: Vec<ImportRecord>,
    macros: Vec<MacroRecord>,
    method_calls_total: u64,
    other_method_total: u64,
    other_method_by_name: BTreeMap<String, u64>,
    macro_invocations_total: u64,
    unqualified_macro_total: u64,
    unqualified_macro_by_name: BTreeMap<String, u64>,
}

impl Totals {
    fn merge(&mut self, o: CensusOutput) {
        self.calls.extend(o.calls);
        self.imports.extend(o.imports);
        self.macros.extend(o.macros);
        self.method_calls_total += o.method_calls_total;
        self.other_method_total += o.other_method_total;
        for (k, v) in o.other_method_by_name {
            *self.other_method_by_name.entry(k).or_insert(0) += v;
        }
        self.macro_invocations_total += o.macro_invocations_total;
        self.unqualified_macro_total += o.unqualified_macro_total;
        for (k, v) in o.unqualified_macro_by_name {
            *self.unqualified_macro_by_name.entry(k).or_insert(0) += v;
        }
    }
}

/// Where the source text lives. This is NOT a reachability or build-target
/// claim.
fn detect_source_location(rel_to_crate: &str) -> &'static str {
    let p = rel_to_crate.replace('\\', "/");
    if p.starts_with("tests/") || p == "tests.rs" {
        "tests"
    } else if p.starts_with("benches/") {
        "benches"
    } else if p.starts_with("examples/") {
        "examples"
    } else if p == "build.rs" {
        "build"
    } else if p.starts_with("src/bin/") || p == "src/main.rs" || p == "main.rs" {
        "src_bin"
    } else if p.starts_with("src/") || p == "src/lib.rs" || p == "lib.rs" {
        "src"
    } else {
        "other"
    }
}

fn module_hint(rel_to_crate: &str, source_location: &str) -> Vec<String> {
    let p = rel_to_crate.replace('\\', "/");
    let core = match source_location {
        "src" => p.strip_prefix("src/").unwrap_or(&p),
        "src_bin" => p.strip_prefix("src/bin/").unwrap_or(&p),
        "tests" => p.strip_prefix("tests/").unwrap_or(&p),
        "benches" => p.strip_prefix("benches/").unwrap_or(&p),
        "examples" => p.strip_prefix("examples/").unwrap_or(&p),
        _ => &p,
    };
    let core = core.strip_suffix(".rs").unwrap_or(core);
    let core = core.strip_suffix("/mod").unwrap_or(core);
    let mut parts: Vec<String> = core
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if matches!(parts.last().map(|s| s.as_str()), Some("lib") | Some("main")) {
        parts.pop();
    }
    parts
}

/// Walks `root`, builds the target inventory, censuses consumer files and writes
/// the report to `config.output`.
pub fn run(config: &Config) -> Result<Summary, String> {
    let root = config
        .root
        .canonicalize()
        .map_err(|e| format!("root {}: {e}", config.root.display()))?;

    let mut targets = TargetConfig::from_names(
        config.target_crates.clone(),
        config.target_types.clone(),
        config.target_methods.clone(),
    );
    targets::add_target_src_crate_names(&mut targets, &config.target_src);

    let root_files = output::walk_rust_files(&root, &config.extra_skip)?;

    let mut crate_cache: HashMap<PathBuf, Option<(PathBuf, String)>> = HashMap::new();
    let mut file_crate: HashMap<PathBuf, Option<String>> = HashMap::new();
    let mut file_crate_root: HashMap<PathBuf, PathBuf> = HashMap::new();
    for (_, abs) in &root_files {
        let dir = abs.parent().unwrap_or(&root).to_path_buf();
        let found = crate_cache
            .entry(dir.clone())
            .or_insert_with(|| targets::crate_name_for_dir(&dir))
            .clone();
        if let Some((manifest, name)) = found {
            file_crate.insert(abs.clone(), Some(name.clone()));
            file_crate_root.insert(
                abs.clone(),
                manifest.parent().unwrap_or(&root).to_path_buf(),
            );
        } else {
            file_crate.insert(abs.clone(), None);
            file_crate_root.insert(abs.clone(), root.clone());
        }
    }

    // ── inventory pass ─────────────────────────────────────────────────────
    let mut inventory_seen: HashSet<PathBuf> = HashSet::new();
    for (_, abs) in &root_files {
        let is_target = file_crate
            .get(abs)
            .and_then(|c| c.as_ref())
            .map(|c| targets.is_target_crate(c))
            .unwrap_or(false);
        if !is_target {
            continue;
        }
        if let Ok(src) = std::fs::read_to_string(abs) {
            if let Ok(ast) = syn::parse_file(&src) {
                targets::collect_inventory_file(&mut targets, &ast);
            }
        }
        inventory_seen.insert(abs.clone());
    }
    for ts in &config.target_src {
        if !ts.exists() {
            return Err(format!("--target-src {} does not exist", ts.display()));
        }
        let files = if ts.is_dir() {
            let scan_root = if ts.join("src").is_dir() {
                ts.join("src")
            } else {
                ts.clone()
            };
            output::walk_rust_files(&scan_root, &config.extra_skip)?
        } else {
            vec![(ts.to_string_lossy().to_string(), ts.clone())]
        };
        for (_, abs) in files {
            let canon = abs.canonicalize().unwrap_or_else(|_| abs.clone());
            if inventory_seen.contains(&canon) {
                continue;
            }
            if let Ok(src) = std::fs::read_to_string(&abs) {
                if let Ok(ast) = syn::parse_file(&src) {
                    targets::collect_inventory_file(&mut targets, &ast);
                }
            }
            inventory_seen.insert(canon);
        }
    }
    targets.prune_primitives();

    // ── census pass ────────────────────────────────────────────────────────
    let mut totals = Totals::default();
    let mut files_meta: Vec<FileMeta> = Vec::new();

    for (rel, abs) in &root_files {
        let bytes = std::fs::read(abs).map_err(|e| format!("read {}: {e}", abs.display()))?;
        let sha = output::sha256_hex(&bytes);
        let crate_name = file_crate.get(abs).cloned().flatten();
        let is_target_crate_file = crate_name
            .as_deref()
            .map(|c| targets.is_target_crate(c))
            .unwrap_or(false);
        let crate_root = file_crate_root
            .get(abs)
            .cloned()
            .unwrap_or_else(|| root.clone());
        let rel_to_crate = abs
            .strip_prefix(&crate_root)
            .map(|r| r.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| rel.clone());
        let source_location = detect_source_location(&rel_to_crate);
        let lines = bytes.iter().filter(|b| **b == b'\n').count() as u64;

        let src = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => {
                files_meta.push(FileMeta {
                    path: rel.clone(),
                    crate_name,
                    source_location: source_location.to_string(),
                    sha256: sha,
                    bytes: 0,
                    lines,
                    parsed: false,
                    parse_error: Some("source is not valid UTF-8".to_string()),
                    is_target_crate_file,
                    reachability: "not_determined".to_string(),
                });
                continue;
            }
        };

        if is_target_crate_file {
            files_meta.push(FileMeta {
                path: rel.clone(),
                crate_name,
                source_location: source_location.to_string(),
                sha256: sha,
                bytes: src.len() as u64,
                lines,
                parsed: true,
                parse_error: None,
                is_target_crate_file,
                reachability: "not_determined".to_string(),
            });
            continue;
        }

        match syn::parse_file(&src) {
            Ok(ast) => {
                let module = module_hint(&rel_to_crate, source_location);
                let out = census::census_file(
                    &targets,
                    rel,
                    &src,
                    source_location,
                    crate_name.as_deref(),
                    module,
                    &ast,
                );
                totals.merge(out);
                files_meta.push(FileMeta {
                    path: rel.clone(),
                    crate_name,
                    source_location: source_location.to_string(),
                    sha256: sha,
                    bytes: src.len() as u64,
                    lines,
                    parsed: true,
                    parse_error: None,
                    is_target_crate_file,
                    reachability: "not_determined".to_string(),
                });
            }
            Err(e) => {
                files_meta.push(FileMeta {
                    path: rel.clone(),
                    crate_name,
                    source_location: source_location.to_string(),
                    sha256: sha,
                    bytes: src.len() as u64,
                    lines,
                    parsed: false,
                    parse_error: Some(e.to_string()),
                    is_target_crate_file,
                    reachability: "not_determined".to_string(),
                });
            }
        }
    }

    totals.calls.sort_by(|a, b| {
        (&a.file, a.span.line, a.span.column).cmp(&(&b.file, b.span.line, b.span.column))
    });
    totals
        .imports
        .sort_by(|a, b| (&a.file, a.line, a.column).cmp(&(&b.file, b.line, b.column)));
    totals
        .macros
        .sort_by(|a, b| (&a.file, a.line, a.column).cmp(&(&b.file, b.line, b.column)));

    let summary = output::compute_summary(
        &root.to_string_lossy(),
        &files_meta,
        &totals.calls,
        &totals.imports,
        &totals.macros,
        totals.method_calls_total,
        totals.other_method_total,
        &totals.other_method_by_name,
        totals.unqualified_macro_total,
        &totals.unqualified_macro_by_name,
    );

    let meta = Meta {
        schema_version: SCHEMA_VERSION,
        tool: TOOL_NAME.to_string(),
        tool_version: TOOL_VERSION.to_string(),
        generated_at_utc: output::now_utc(),
        root: root.to_string_lossy().to_string(),
        revision: output::git_revision(&root),
        target_crates: targets.crates.clone(),
        target_types: targets.types.iter().cloned().collect(),
        target_src_roots: config
            .target_src
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
        syn_version: SYN_VERSION.to_string(),
        proc_macro2_version: PROC_MACRO2_VERSION.to_string(),
        tool_source_sha256: output::tool_source_sha256(),
        tool_binary_sha256: output::tool_binary_sha256(),
        compatibility_note: format!(
            "schema_version {SCHEMA_VERSION}; tool {TOOL_VERSION}. Reports produced by \
call-census < 0.2.0 use different field names and different confidence semantics \
(`resolved` rather than `attributed`). They MUST be regenerated; do not merge or \
compare counts across schema versions."
        ),
        limitations: LIMITATIONS.iter().map(|s| s.to_string()).collect(),
    };

    let inventory = output::inventory_doc(&targets);
    let inputs = output::WriteInputs {
        output: &config.output,
        meta: &meta,
        files: &files_meta,
        calls: &totals.calls,
        imports: &totals.imports,
        macros: &totals.macros,
        summary: &summary,
        inventory: &inventory,
    };
    output::write_all(&inputs)?;
    Ok(summary)
}
