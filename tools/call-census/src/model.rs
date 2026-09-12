//! Serializable record types for the call census.
//!
//! These types are the contract between the parser/visitor and the writers.
//! Every field that carries an inference is paired with an `attribution_basis`
//! or `basis` string: the tool is required to say *why* it believes something,
//! never only *what* it believes.
//!
//! Wording discipline: the tool performs **syntactic attribution**, never
//! resolution. Counts are not a mathematically assured lower bound, and a
//! record never asserts that a file is compiled or reachable.

use serde::Serialize;
use std::collections::BTreeMap;

/// Bumped when the report schema or the meaning of a field changes. Reports
/// with a different schema must not be merged with this one.
pub const SCHEMA_VERSION: u32 = 2;

/// A 1-based line/column position, plus the end position, for a source span.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct SpanRec {
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

impl SpanRec {
    pub fn is_zero(&self) -> bool {
        self.line == 0 && self.column == 0
    }
}

/// The enclosing named item a call was found in. Closures are folded into the
/// nearest named function.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CallerCtx {
    pub kind: String,
    pub name: String,
    pub qualified: String,
    pub impl_type: Option<String>,
    pub module_path: String,
    pub is_test: bool,
    /// `cfg` predicates from the item and every enclosing module/impl, verbatim.
    /// Not evaluated.
    pub cfg: Vec<String>,
    pub span: Option<SpanRec>,
}

/// A single piece of evidence supporting an attribution.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Evidence {
    pub kind: String,
    pub text: String,
    pub file: String,
    pub line: u32,
    pub column: u32,
}

/// One recorded call expression, method call, or target path reference.
#[derive(Debug, Clone, Serialize)]
pub struct CallRecord {
    /// Stable across runs for the same source:
    /// `call:<file>:<line>:<col>-<end_line>:<end_col>`.
    pub id: String,
    /// `call` (path call), `method` (method call), `path_ref`.
    pub record_kind: String,
    /// Fine-grained category, see `census.rs` `CAT_*` constants.
    pub category: String,
    /// For call records: `free_fn`, `associated_fn`, `constructor`,
    /// `unclassified`; for method records: `method`; for `path_ref`: `path_ref`.
    /// Constructors are recognised from the target inventory (enum variants,
    /// tuple structs), never from capitalisation.
    pub operation_kind: String,
    /// `high`, `medium`, `ambiguous`, `unattributed`, `dynamic`.
    pub confidence: String,
    pub is_call: bool,
    pub file: String,
    /// Source location classification (`src`, `src_bin`, `tests`, `benches`,
    /// `examples`, `build`, `other`). This is where the text lives, NOT evidence
    /// that it is compiled or reachable.
    pub source_location: String,
    /// Always `not_determined`: the tool does not build a module graph, so it
    /// never claims a file is compiled.
    pub reachability: String,
    pub crate_name: Option<String>,
    pub module_path: String,
    pub caller: CallerCtx,
    /// Exact source slice of the callee expression (from the token span).
    pub callee_text: String,
    /// Dotted path for a path call, e.g. `hashtrinity::Table::insert_with`.
    pub callee_path: Option<String>,
    /// Method name for a method call.
    pub method: Option<String>,
    /// Exact source slice of the receiver expression.
    pub receiver_text: Option<String>,
    /// Syntactically inferred receiver type text, if any.
    pub receiver_type: Option<String>,
    pub target_crate: Option<String>,
    pub target_symbol: Option<String>,
    pub target_type: Option<String>,
    /// Human-readable reason the record has the confidence it does. This is a
    /// syntactic basis, not a proof.
    pub attribution_basis: String,
    pub evidence: Vec<Evidence>,
    pub cfg: Vec<String>,
    pub is_test_context: bool,
    /// True when the record describes a macro/derive site whose body was NOT
    /// expanded; the calls it generates are not in this census.
    pub macro_unexpanded: bool,
    /// True when the receiver is a trait object / generic bound we cannot
    /// statically resolve to a concrete impl.
    pub dynamic_dispatch: bool,
    /// True when the record is a *candidate* for manual resolution rather than a
    /// syntactically attributed target call.
    pub candidate: bool,
    pub span: SpanRec,
}

/// One `use` declaration, import, glob, or re-export.
#[derive(Debug, Clone, Serialize)]
pub struct ImportRecord {
    pub id: String,
    pub file: String,
    pub source_location: String,
    pub crate_name: Option<String>,
    pub module_path: String,
    pub line: u32,
    pub column: u32,
    /// Rendered `use` tree text.
    pub text: String,
    /// Full dotted imported path, e.g. `hashtrinity::codec::syntax::AsTerm`.
    pub path: String,
    /// Root segment as written (`hashtrinity`, `crate`, `ht`, ...).
    pub root: String,
    pub alias: Option<String>,
    pub is_glob: bool,
    pub is_reexport: bool,
    pub visibility: String,
    /// Whether the root resolves to one of the configured target crates.
    pub is_target: bool,
    /// `high` when the root is a target crate name; `medium` when only a
    /// re-export chain could connect it; `unattributed` otherwise.
    pub confidence: String,
    pub attribution_basis: String,
}

/// One macro invocation, derive use, attribute macro, or `macro_rules!`
/// definition. Macro bodies are not expanded; this record exists so agents can
/// see what could hide calls.
#[derive(Debug, Clone, Serialize)]
pub struct MacroRecord {
    pub id: String,
    pub file: String,
    pub source_location: String,
    pub crate_name: Option<String>,
    pub module_path: String,
    pub caller: CallerCtx,
    pub line: u32,
    pub column: u32,
    /// `macro_invocation`, `derive`, `attribute_macro`, `macro_rules_def`.
    pub kind: String,
    pub name: String,
    pub path: Option<String>,
    pub is_target: bool,
    pub confidence: String,
    pub attribution_basis: String,
    pub unexpanded: bool,
    pub cfg: Vec<String>,
}

/// Per-file metadata and digest.
#[derive(Debug, Clone, Serialize)]
pub struct FileMeta {
    pub path: String,
    pub crate_name: Option<String>,
    /// Where the file lives (`src`, `tests`, `benches`, ...).
    pub source_location: String,
    pub sha256: String,
    pub bytes: u64,
    pub lines: u64,
    pub parsed: bool,
    pub parse_error: Option<String>,
    pub is_target_crate_file: bool,
    /// Always `not_determined`; kept explicit so no report implies compilation.
    pub reachability: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RevisionMeta {
    pub git_head: Option<String>,
    pub git_branch: Option<String>,
    pub git_describe: Option<String>,
    pub git_status_porcelain_lines: Option<u64>,
    pub working_tree_note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Meta {
    pub schema_version: u32,
    pub tool: String,
    pub tool_version: String,
    pub generated_at_utc: String,
    pub root: String,
    pub revision: RevisionMeta,
    pub target_crates: Vec<String>,
    pub target_types: Vec<String>,
    pub target_src_roots: Vec<String>,
    pub syn_version: String,
    pub proc_macro2_version: String,
    /// SHA-256 over the tool's own source tree (Cargo.toml + src/*.rs + tests).
    /// Reproducible: same sources, same digest. `None` if sources are absent.
    pub tool_source_sha256: Option<String>,
    /// SHA-256 of the running executable, so a report can name the exact binary.
    pub tool_binary_sha256: Option<String>,
    pub compatibility_note: String,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CrateSummary {
    pub files: u64,
    pub records: u64,
    pub attributed: u64,
    pub unattributed: u64,
    pub imports: u64,
    pub reexports: u64,
    pub macros: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Summary {
    pub schema_version: u32,
    pub root: String,
    pub files_total: u64,
    pub files_parsed: u64,
    pub files_parse_error: u64,
    pub files_target_crate: u64,
    pub records_total: u64,
    pub calls_total: u64,
    /// Call expressions with a syntactic attribution (high/medium). This is a
    /// count of syntax, not of executed calls and not a proof of completeness.
    pub attributed_calls: u64,
    pub attributed_high: u64,
    pub attributed_medium: u64,
    pub direct_path_calls: u64,
    pub method_calls_attributed: u64,
    pub unattributed_method_candidates: u64,
    /// Candidate method calls grouped by method name.
    pub unattributed_method_candidates_by_name: BTreeMap<String, u64>,
    pub ambiguous_methods: u64,
    /// Labelled heuristic attributions: method name matches an imported target
    /// trait, receiver type unknown. Counted separately from typed attribution.
    pub heuristic_method_calls: u64,
    pub dynamic_dispatch_methods: u64,
    /// Method calls whose receiver was a known non-target type or otherwise
    /// unclassified. Aggregated, not individually recorded.
    pub other_method_calls_total: u64,
    pub other_method_calls_by_name: BTreeMap<String, u64>,
    pub path_refs: u64,
    pub target_imports: u64,
    pub target_reexports: u64,
    pub macro_sites: u64,
    pub derive_sites: u64,
    pub attribute_macro_sites: u64,
    pub macro_rules_defs: u64,
    pub unqualified_macro_invocations: u64,
    pub unqualified_macro_by_name: BTreeMap<String, u64>,
    pub distinct_callees: u64,
    pub distinct_callers: u64,
    pub distinct_target_symbols: u64,
    pub by_crate: BTreeMap<String, CrateSummary>,
    pub by_source_location: BTreeMap<String, u64>,
    pub by_category: BTreeMap<String, u64>,
    pub by_operation_kind: BTreeMap<String, u64>,
    pub by_subsystem: BTreeMap<String, u64>,
    pub by_target_symbol: BTreeMap<String, u64>,
    pub by_target_type: BTreeMap<String, u64>,
    pub attribution_note: String,
    pub limitations: Vec<String>,
}
