//! The AST visitor that produces the census.
//!
//! ## What this is, precisely
//!
//! `syn` parses tokens into syntax. It does not run name resolution, does not
//! instantiate generics, does not expand macros and does not know which trait a
//! method call dispatches to. Therefore:
//!
//! * A call is **syntactically attributed** only when the callee path is rooted
//!   at a configured target crate, or the receiver's type is written down in a
//!   form this file can read (an annotation, a field type, an `impl` self type,
//!   a constructor binding, an attributed local type alias). This is not a proof of
//!   completeness and not a mathematically assured lower bound.
//! * A method call on a receiver whose type is unknown is a **candidate**, never
//!   attributed to a target type just because the name matches. `.get` on an
//!   untyped receiver is recorded as unattributed.
//! * Macro bodies and derive output are **not expanded**; their generated calls
//!   are not in the census, and every macro site says so.
//! * Comments and string literals never appear: parsing to AST discards them.
//! * A file's `source_location` says where the text lives; it never asserts that
//!   the file is compiled or reachable. `reachability` is always
//!   `not_determined`.

use crate::model::*;
use crate::targets::TargetConfig;
use quote::ToTokens;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use syn::{Attribute, Expr, File, ImplItem, TraitItem, UseTree};

mod emit;
mod resolve;

// Categories.
pub const CAT_DIRECT_CALL: &str = "target_direct_call";
pub const CAT_ASSOC_CALL: &str = "target_assoc_fn_call";
pub const CAT_METHOD_ATTRIBUTED: &str = "method_attributed";
pub const CAT_METHOD_AMBIGUOUS: &str = "method_ambiguous";
pub const CAT_METHOD_CANDIDATE: &str = "method_candidate";
pub const CAT_METHOD_DYNAMIC: &str = "method_dynamic_dispatch";
/// A labelled heuristic: the method name is provided by a target trait that is
/// imported in this module, but the receiver type is unknown. It is counted
/// separately from typed attributions and still carries `candidate: true`.
pub const CAT_METHOD_TRAIT_HEURISTIC: &str = "method_trait_heuristic";

const CONSTRUCTOR_METHODS: &[&str] = &[
    "new",
    "default",
    "with_capacity",
    "coordinate",
    "coordinate_with_capacity",
    "from_iter",
    "from",
    "zeroed",
    "open",
];

/// Everything one file contributes.
#[derive(Default)]
pub struct CensusOutput {
    pub calls: Vec<CallRecord>,
    pub imports: Vec<ImportRecord>,
    pub macros: Vec<MacroRecord>,
    pub method_calls_total: u64,
    pub other_method_total: u64,
    pub other_method_by_name: BTreeMap<String, u64>,
    pub macro_invocations_total: u64,
    pub unqualified_macro_total: u64,
    pub unqualified_macro_by_name: BTreeMap<String, u64>,
}

#[derive(Debug, Clone)]
struct ImportTarget {
    full_path: Vec<String>,
    root: String,
    is_target: bool,
}

#[derive(Debug, Clone)]
struct TypeBinding {
    ty: String,
    confidence: String,
    basis: String,
    file: String,
    line: u32,
    column: u32,
}

#[derive(Debug, Clone, Default)]
struct Scope {
    path: Vec<String>,
    imports: HashMap<String, ImportTarget>,
    globs: Vec<ImportTarget>,
    local_types: BTreeSet<String>,
    local_fns: BTreeSet<String>,
    local_aliases: HashMap<String, String>,
    struct_fields: HashMap<String, BTreeMap<String, String>>,
    bindings: HashMap<String, TypeBinding>,
}

#[derive(Debug, Clone)]
enum NameScope {
    Local,
    Import(ImportTarget),
    None,
}

#[derive(Debug, Clone)]
enum TypeVerdict {
    Target {
        ty: String,
        target_type: String,
        target_crate: String,
        confidence: String,
        basis: String,
    },
    Ambiguous {
        name: String,
        basis: String,
    },
    CandidateName {
        name: String,
    },
    /// A concrete type name that is not a target type and not imported from a
    /// target crate. Calls on it are confidently *not* target calls.
    NonTarget,
    Local,
    Unknown,
}

#[derive(Debug, Clone)]
struct ReceiverInfo {
    text: String,
    verdict: TypeVerdict,
    ty_text: Option<String>,
    evidence: Vec<Evidence>,
    dynamic: bool,
}

#[derive(Debug, Clone)]
struct PathVerdict {
    target_crate: String,
    symbol: String,
    target_type: Option<String>,
    leaf: String,
    #[allow(dead_code)]
    owner_is_type: bool,
    confidence: String,
    basis: String,
}

pub struct Census<'a> {
    cfg: &'a TargetConfig,
    file: &'a str,
    src: &'a str,
    source_location: &'a str,
    crate_name: Option<&'a str>,
    module_scopes: Vec<Scope>,
    /// Imports introduced by `use` inside a block/function body. These shadow
    /// module imports for the duration of the block only.
    block_scopes: Vec<Scope>,
    type_scopes: Vec<HashMap<String, TypeBinding>>,
    impl_types: Vec<Option<String>>,
    callers: Vec<CallerCtx>,
    cfg_stack: Vec<String>,
    cfg_test_depth: usize,
    out: CensusOutput,
}

/// Parses and censuses one already-parsed file.
pub fn census_file<'a>(
    cfg: &'a TargetConfig,
    relpath: &'a str,
    src: &'a str,
    source_location: &'a str,
    crate_name: Option<&'a str>,
    initial_module: Vec<String>,
    ast: &'a File,
) -> CensusOutput {
    let mut c = Census {
        cfg,
        file: relpath,
        src,
        source_location,
        crate_name,
        module_scopes: Vec::new(),
        block_scopes: Vec::new(),
        type_scopes: Vec::new(),
        impl_types: Vec::new(),
        callers: Vec::new(),
        cfg_stack: Vec::new(),
        cfg_test_depth: 0,
        out: CensusOutput::default(),
    };
    c.module_scopes.push(Scope {
        path: initial_module,
        ..Scope::default()
    });
    c.prescan_items(&ast.items);
    syn::visit::visit_file(&mut c, ast);
    c.out
}

// ── free helpers ───────────────────────────────────────────────────────────

fn path_to_string_from_expr(expr: &Expr) -> String {
    match expr {
        Expr::Path(p) => path_to_string(&p.path),
        _ => expr.to_token_stream().to_string(),
    }
}

fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

/// Strips references, pointers, lifetimes and `mut`/`const` qualifiers, then
/// unwraps smart pointers, returning the base type path text and whether a
/// wrapper was present. Works on `proc-macro2` token strings such as
/// `& mut Table < K , V >` and on `Box < Table >`.
fn normalize_type(ty: &str) -> (String, bool) {
    let mut s = ty.trim().to_string();
    let mut wrapped = false;
    // Spaced qualifier forms.
    loop {
        let t = s.trim_start();
        if t.is_empty() {
            return (String::new(), wrapped);
        }
        if let Some(r) = t.strip_prefix('&') {
            s = r.to_string();
            wrapped = true;
            continue;
        }
        if let Some(r) = t.strip_prefix('*') {
            s = r.to_string();
            wrapped = true;
            continue;
        }
        if t.starts_with('\'') {
            let end = t.find(char::is_whitespace).unwrap_or(t.len());
            s = t[end..].to_string();
            continue;
        }
        if let Some(r) = t.strip_prefix("mut") {
            if r.is_empty() || r.starts_with(|c: char| c.is_whitespace() || c == '<' || c == '(') {
                s = r.to_string();
                continue;
            }
        }
        if let Some(r) = t.strip_prefix("const") {
            if r.is_empty() || r.starts_with(char::is_whitespace) {
                s = r.to_string();
                continue;
            }
        }
        break;
    }
    // Compact form.
    let mut cur: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    for _ in 0..4 {
        if let Some(r) = cur.strip_prefix("&mut") {
            cur = r.to_string();
            wrapped = true;
            continue;
        }
        if let Some(r) = cur.strip_prefix('&') {
            cur = r.to_string();
            wrapped = true;
            continue;
        }
        if let Some(r) = cur.strip_prefix('*') {
            cur = r.to_string();
            wrapped = true;
            continue;
        }
        if let Some(r) = cur.strip_prefix("mut") {
            if r.is_empty()
                || r.starts_with(|c: char| c == '<' || c == '(' || c.is_ascii_uppercase())
            {
                cur = r.to_string();
                continue;
            }
        }
        let mut inner: Option<String> = None;
        for w in ["Box", "Rc", "Arc", "Cell", "RefCell", "Cow", "Pin"] {
            if let Some(rest) = cur.strip_prefix(w) {
                if let Some(rest) = rest.strip_prefix('<') {
                    if rest.ends_with('>') {
                        inner = Some(rest[..rest.len() - 1].to_string());
                        break;
                    }
                }
            }
        }
        match inner {
            Some(i) => {
                cur = i;
                wrapped = true;
            }
            None => break,
        }
    }
    let base = cur.split('<').next().unwrap_or("").to_string();
    (base, wrapped)
}

/// The last path segment of a type text, e.g. `foo::Bar<T>` -> `Bar`.
fn type_base_name(ty: &str) -> Option<String> {
    let (base, _) = normalize_type(ty);
    base.split("::")
        .filter(|s| !s.is_empty())
        .last()
        .map(|s| s.to_string())
}

fn type_text_is_target(cfg: &TargetConfig, scope: &Scope, ty: &str) -> bool {
    let (base, _) = normalize_type(ty);
    let segs: Vec<&str> = base.split("::").filter(|s| !s.is_empty()).collect();
    if segs.is_empty() {
        return false;
    }
    if cfg.is_target_crate(segs[0]) {
        return true;
    }
    if let Some(it) = scope.imports.get(segs[0]) {
        if it.is_target {
            return true;
        }
    }
    if segs.len() == 1 && cfg.is_target_type(segs[0]) {
        return scope.globs.iter().any(|g| g.is_target)
            || scope.imports.values().any(|i| i.is_target);
    }
    false
}

fn flatten_use_tree(
    prefix: &[String],
    tree: &UseTree,
    out: &mut Vec<(Vec<String>, Option<String>, bool)>,
) {
    match tree {
        UseTree::Path(p) => {
            let mut np = prefix.to_vec();
            np.push(p.ident.to_string());
            flatten_use_tree(&np, &p.tree, out);
        }
        UseTree::Name(n) => {
            let mut np = prefix.to_vec();
            np.push(n.ident.to_string());
            out.push((np, None, false));
        }
        UseTree::Rename(r) => {
            let mut np = prefix.to_vec();
            np.push(r.ident.to_string());
            out.push((np, Some(r.rename.to_string()), false));
        }
        UseTree::Glob(_) => {
            out.push((prefix.to_vec(), None, true));
        }
        UseTree::Group(g) => {
            for t in &g.items {
                flatten_use_tree(prefix, t, out);
            }
        }
    }
}

fn register_use(cfg: &TargetConfig, scope: &mut Scope, tree: &UseTree) {
    let mut entries = Vec::new();
    flatten_use_tree(&[], tree, &mut entries);
    for (path, alias, is_glob) in entries {
        let root = path.first().cloned().unwrap_or_default();
        let is_target = cfg.is_target_crate(&root);
        let it = ImportTarget {
            full_path: path.clone(),
            root,
            is_target,
        };
        if is_glob {
            scope.globs.push(it);
        } else {
            let key = alias
                .clone()
                .unwrap_or_else(|| path.last().cloned().unwrap_or_default());
            scope.imports.insert(key, it);
        }
    }
}

fn attrs_have_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("test"))
}

fn attrs_have_cfg_test(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && a.meta.to_token_stream().to_string().contains("test"))
}

fn cfg_strings(attrs: &[Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("cfg") || a.path().is_ident("cfg_attr"))
        .map(|a| a.meta.to_token_stream().to_string())
        .collect()
}

fn impl_item_attrs(item: &ImplItem) -> Vec<Attribute> {
    match item {
        ImplItem::Const(i) => i.attrs.clone(),
        ImplItem::Fn(i) => i.attrs.clone(),
        ImplItem::Type(i) => i.attrs.clone(),
        ImplItem::Macro(i) => i.attrs.clone(),
        ImplItem::Verbatim(_) => Vec::new(),
        _ => Vec::new(),
    }
}

fn trait_item_attrs(item: &TraitItem) -> Vec<Attribute> {
    match item {
        TraitItem::Const(i) => i.attrs.clone(),
        TraitItem::Fn(i) => i.attrs.clone(),
        TraitItem::Type(i) => i.attrs.clone(),
        TraitItem::Macro(i) => i.attrs.clone(),
        TraitItem::Verbatim(_) => Vec::new(),
        _ => Vec::new(),
    }
}
