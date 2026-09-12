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
use crate::targets::{impl_base_name, TargetConfig};
use quote::ToTokens;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{
    Attribute, Expr, ExprAssign, ExprCall, ExprClosure, ExprForLoop, ExprMethodCall, File, FnArg,
    ImplItem, ImplItemFn, Item, ItemFn, ItemImpl, ItemMacro, ItemMod, ItemUse, Local, Pat,
    Signature, TraitItem, TraitItemFn, UseTree, Visibility,
};

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

impl<'a> Census<'a> {
    // ── span / text helpers ────────────────────────────────────────────────

    fn text_of(&self, span: proc_macro2::Span) -> String {
        let r = span.byte_range();
        if r.start <= r.end && r.end <= self.src.len() {
            self.src[r].trim().to_string()
        } else {
            String::new()
        }
    }

    fn span_rec(span: proc_macro2::Span) -> SpanRec {
        let s = span.start();
        let e = span.end();
        SpanRec {
            line: s.line as u32,
            column: s.column as u32,
            end_line: e.line as u32,
            end_column: e.column as u32,
        }
    }

    fn id(&self, prefix: &str, span: proc_macro2::Span) -> String {
        let s = span.start();
        let e = span.end();
        format!(
            "{}:{}:{}:{}-{}:{}",
            prefix, self.file, s.line, s.column, e.line, e.column
        )
    }

    /// Identifier for a record whose owner span may be shared by several
    /// entries (grouped imports, comma-separated derives).
    fn id_with(&self, prefix: &str, span: proc_macro2::Span, discriminator: &str) -> String {
        format!("{}:{}", self.id(prefix, span), discriminator)
    }

    fn module_path_string(&self) -> String {
        let parts: Vec<String> = self
            .module_scopes
            .iter()
            .flat_map(|s| s.path.iter().cloned())
            .collect();
        if parts.is_empty() {
            "crate".to_string()
        } else {
            parts.join("::")
        }
    }

    fn current_scope_path(&self) -> Vec<String> {
        self.module_scopes
            .last()
            .map(|s| s.path.clone())
            .unwrap_or_default()
    }

    fn combined_cfg(&self, extra: &[String]) -> Vec<String> {
        let mut v = self.cfg_stack.clone();
        v.extend(extra.iter().cloned());
        v
    }

    fn current_caller(&self) -> CallerCtx {
        self.callers.last().cloned().unwrap_or_else(|| CallerCtx {
            kind: "module".to_string(),
            name: "<module>".to_string(),
            qualified: self.module_path_string(),
            impl_type: None,
            module_path: self.module_path_string(),
            is_test: self.cfg_test_depth > 0 || self.source_location == "tests",
            cfg: self.cfg_stack.clone(),
            span: None,
        })
    }

    // ── scope / import helpers ─────────────────────────────────────────────

    fn prescan_items(&mut self, items: &[Item]) {
        let cfg = self.cfg;
        let scope = self.module_scopes.last_mut().expect("file scope");
        for it in items {
            match it {
                Item::Use(u) => register_use(cfg, scope, &u.tree),
                Item::Struct(s) => {
                    scope.local_types.insert(s.ident.to_string());
                    let mut fields = BTreeMap::new();
                    if let syn::Fields::Named(named) = &s.fields {
                        for f in &named.named {
                            if let Some(id) = &f.ident {
                                fields.insert(id.to_string(), f.ty.to_token_stream().to_string());
                            }
                        }
                    }
                    if !fields.is_empty() {
                        scope
                            .struct_fields
                            .entry(s.ident.to_string())
                            .or_insert(fields);
                    }
                }
                Item::Enum(e) => {
                    scope.local_types.insert(e.ident.to_string());
                }
                Item::Union(u) => {
                    scope.local_types.insert(u.ident.to_string());
                }
                Item::Trait(t) => {
                    scope.local_types.insert(t.ident.to_string());
                }
                Item::Fn(f) => {
                    scope.local_fns.insert(f.sig.ident.to_string());
                }
                Item::Type(t) => {
                    let name = t.ident.to_string();
                    scope.local_types.insert(name.clone());
                    let rhs = t.ty.to_token_stream().to_string();
                    if type_text_is_target(cfg, scope, &rhs) {
                        scope.local_aliases.insert(name, rhs);
                    }
                }
                Item::Mod(m) => {
                    scope.local_types.insert(m.ident.to_string());
                }
                Item::Const(c) => {
                    scope.bindings.insert(
                        c.ident.to_string(),
                        TypeBinding {
                            ty: c.ty.to_token_stream().to_string(),
                            confidence: "high".to_string(),
                            basis: "const type annotation".to_string(),
                            file: self.file.to_string(),
                            line: 0,
                            column: 0,
                        },
                    );
                }
                Item::Static(s) => {
                    scope.bindings.insert(
                        s.ident.to_string(),
                        TypeBinding {
                            ty: s.ty.to_token_stream().to_string(),
                            confidence: "high".to_string(),
                            basis: "static type annotation".to_string(),
                            file: self.file.to_string(),
                            line: 0,
                            column: 0,
                        },
                    );
                }
                _ => {}
            }
        }
    }

    fn lookup_binding(&self, name: &str) -> Option<TypeBinding> {
        for scope in self.type_scopes.iter().rev() {
            if let Some(b) = scope.get(name) {
                return Some(b.clone());
            }
        }
        for scope in self.module_scopes.iter().rev() {
            if let Some(b) = scope.bindings.get(name) {
                return Some(b.clone());
            }
        }
        None
    }

    fn lookup_import(&self, name: &str) -> Option<ImportTarget> {
        for scope in self.block_scopes.iter().rev() {
            if let Some(it) = scope.imports.get(name) {
                return Some(it.clone());
            }
        }
        for scope in self.module_scopes.iter().rev() {
            if let Some(it) = scope.imports.get(name) {
                return Some(it.clone());
            }
        }
        None
    }

    /// Scope-aware lookup: the innermost scope that binds the name decides,
    /// whether it binds a local item or an import.
    fn name_in_scope(&self, name: &str) -> NameScope {
        for scope in self.block_scopes.iter().rev() {
            if let Some(it) = scope.imports.get(name) {
                return NameScope::Import(it.clone());
            }
        }
        for scope in self.module_scopes.iter().rev() {
            if scope.local_types.contains(name) {
                return NameScope::Local;
            }
            if let Some(it) = scope.imports.get(name) {
                return NameScope::Import(it.clone());
            }
        }
        NameScope::None
    }

    fn is_local_type(&self, name: &str) -> bool {
        self.module_scopes
            .iter()
            .rev()
            .any(|s| s.local_types.contains(name))
    }

    fn is_local_fn(&self, name: &str) -> bool {
        self.module_scopes
            .iter()
            .rev()
            .any(|s| s.local_fns.contains(name))
    }

    fn local_alias(&self, name: &str) -> Option<String> {
        for scope in self.module_scopes.iter().rev() {
            if let Some(t) = scope.local_aliases.get(name) {
                return Some(t.clone());
            }
        }
        None
    }

    fn target_glob_crate(&self) -> Option<String> {
        for scope in self.block_scopes.iter().rev() {
            for g in &scope.globs {
                if g.is_target {
                    return Some(g.root.clone());
                }
            }
        }
        for scope in self.module_scopes.iter().rev() {
            for g in &scope.globs {
                if g.is_target {
                    return Some(g.root.clone());
                }
            }
        }
        None
    }

    fn has_non_target_glob(&self) -> bool {
        self.block_scopes
            .iter()
            .any(|s| s.globs.iter().any(|g| !g.is_target))
            || self
                .module_scopes
                .iter()
                .any(|s| s.globs.iter().any(|g| !g.is_target))
    }

    fn target_import_for(&self, name: &str) -> Option<ImportTarget> {
        self.lookup_import(name).filter(|it| it.is_target)
    }

    /// Whether this file has any reason to be considered a consumer at all: a
    /// target import/glob is in scope, or the source text names a target crate.
    /// This only gates *candidate* emission; it never turns a candidate into an
    /// attributed call.
    fn has_target_signal(&self) -> bool {
        let scope_has = |scopes: &[Scope]| {
            scopes.iter().any(|s| {
                s.imports.values().any(|i| i.is_target) || s.globs.iter().any(|g| g.is_target)
            })
        };
        if scope_has(&self.block_scopes) || scope_has(&self.module_scopes) {
            return true;
        }
        self.cfg
            .crates
            .iter()
            .any(|c| self.src.contains(c.as_str()))
    }

    /// Target traits imported in scope, innermost scope first. Used only for the
    /// clearly labelled `method_trait_heuristic` category.
    fn imported_target_traits(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut push = |it: &ImportTarget| {
            if !it.is_target {
                return;
            }
            if let Some(last) = it.full_path.last() {
                if self.cfg.is_target_trait(last) && !out.iter().any(|(t, _)| t == last) {
                    out.push((last.clone(), it.root.clone()));
                }
            }
        };
        for scope in self.block_scopes.iter().rev() {
            for it in scope.imports.values() {
                push(it);
            }
        }
        for scope in self.module_scopes.iter().rev() {
            for it in scope.imports.values() {
                push(it);
            }
        }
        out
    }

    /// The target trait, if any, that provides `method` and is imported here.
    fn heuristic_trait_for(&self, method: &str) -> Option<(String, String)> {
        self.imported_target_traits()
            .into_iter()
            .find(|(t, _)| self.cfg.trait_provides(t, method))
    }

    // ── type classification ────────────────────────────────────────────────

    fn classify_type(&self, ty: &str) -> TypeVerdict {
        self.classify_type_depth(ty, 0)
    }

    fn classify_type_depth(&self, ty: &str, depth: usize) -> TypeVerdict {
        if depth > 4 {
            return TypeVerdict::Ambiguous {
                name: ty.to_string(),
                basis: "type alias chain too deep to follow".to_string(),
            };
        }
        let (base, _wrapped) = normalize_type(ty);
        if base.is_empty() {
            return TypeVerdict::Unknown;
        }
        let segs: Vec<&str> = base.split("::").filter(|s| !s.is_empty()).collect();
        if segs.is_empty() {
            return TypeVerdict::Unknown;
        }
        let first = segs[0];

        // A fully qualified path rooted at a target crate.
        if self.cfg.is_target_crate(first) {
            let t = segs.last().unwrap().to_string();
            return TypeVerdict::Target {
                ty: base.clone(),
                target_type: t,
                target_crate: first.to_string(),
                confidence: "high".to_string(),
                basis: format!("qualified path root `{first}` is a configured target crate"),
            };
        }

        // A multi-segment path whose first segment is local is a local path.
        if segs.len() > 1 {
            if self.is_local_type(first) || self.is_local_fn(first) {
                return TypeVerdict::NonTarget;
            }
            return TypeVerdict::NonTarget;
        }

        let name = first;

        // A local type alias is followed only if its right-hand side attributes
        // to the target; otherwise it stays ambiguous.
        if let Some(rhs) = self.local_alias(name) {
            return match self.classify_type_depth(&rhs, depth + 1) {
                TypeVerdict::Target {
                    target_type,
                    target_crate,
                    confidence,
                    ..
                } => TypeVerdict::Target {
                    ty: name.to_string(),
                    target_type,
                    target_crate,
                    confidence,
                    basis: format!("local type alias `{name}` = `{rhs}`"),
                },
                _ => TypeVerdict::Ambiguous {
                    name: name.to_string(),
                    basis: format!(
                        "local type alias `{name}` = `{rhs}`; alias target not a target type"
                    ),
                },
            };
        }

        match self.name_in_scope(name) {
            NameScope::Local => TypeVerdict::Local,
            NameScope::Import(it) => {
                if it.is_target {
                    let t = it
                        .full_path
                        .last()
                        .cloned()
                        .unwrap_or_else(|| name.to_string());
                    TypeVerdict::Target {
                        ty: name.to_string(),
                        target_type: t,
                        target_crate: it.root.clone(),
                        confidence: "high".to_string(),
                        basis: format!("`{name}` is imported from target crate `{}`", it.root),
                    }
                } else {
                    TypeVerdict::NonTarget
                }
            }
            NameScope::None => {
                if self.cfg.is_target_type(name) {
                    if self.has_non_target_glob() {
                        TypeVerdict::Ambiguous {
                            name: name.to_string(),
                            basis: format!(
                                "`{name}` matches a target type but a non-target glob import is in scope"
                            ),
                        }
                    } else if let Some(c) = self.target_glob_crate() {
                        TypeVerdict::Target {
                            ty: name.to_string(),
                            target_type: name.to_string(),
                            basis: format!(
                                "name matches a target type and a `{c}` glob import is in scope"
                            ),
                            target_crate: c,
                            confidence: "medium".to_string(),
                        }
                    } else {
                        TypeVerdict::CandidateName {
                            name: name.to_string(),
                        }
                    }
                } else {
                    TypeVerdict::NonTarget
                }
            }
        }
    }

    fn classify_path(&self, path: &syn::Path) -> Option<PathVerdict> {
        let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if segs.is_empty() {
            return None;
        }
        let leaf = segs.last().unwrap().clone();
        let first = segs[0].as_str();

        if self.cfg.is_target_crate(first) {
            let symbol = if segs.len() > 1 {
                segs[1..].join("::")
            } else {
                first.to_string()
            };
            let target_type = if segs.len() > 1 && self.cfg.is_target_type(&segs[1]) {
                Some(segs[1].clone())
            } else {
                None
            };
            return Some(PathVerdict {
                target_crate: first.to_string(),
                symbol,
                owner_is_type: target_type.is_some(),
                target_type,
                leaf,
                confidence: "high".to_string(),
                basis: format!("path root `{first}` is a configured target crate"),
            });
        }

        if let Some(it) = self.target_import_for(first) {
            let mut full = it.full_path.clone();
            full.extend(segs[1..].iter().cloned());
            let symbol = if full.len() > 1 {
                full[1..].join("::")
            } else {
                first.to_string()
            };
            let target_type = if full.len() > 1 && self.cfg.is_target_type(&full[1]) {
                Some(full[1].clone())
            } else {
                None
            };
            return Some(PathVerdict {
                target_crate: it.root.clone(),
                symbol,
                owner_is_type: target_type.is_some(),
                target_type,
                leaf,
                confidence: "high".to_string(),
                basis: format!(
                    "`{first}` is an import alias for target path `{}`",
                    it.full_path.join("::")
                ),
            });
        }

        // `Table::insert_with(...)` where `Table` is a target type in scope.
        // A non-target glob (e.g. `use super::*`) must block attribution: the
        // name may come from that glob, as with a local `phasefold::Table`.
        if segs.len() >= 2 && self.cfg.is_target_type(first) {
            let (c, confidence) = match self.name_in_scope(first) {
                NameScope::Local => return None,
                NameScope::Import(it) if !it.is_target => return None,
                NameScope::Import(it) => (it.root.clone(), "high"),
                NameScope::None => {
                    if self.has_non_target_glob() {
                        return None;
                    }
                    match self.target_glob_crate() {
                        Some(c) => (c, "medium"),
                        None => return None,
                    }
                }
            };
            return Some(PathVerdict {
                target_crate: c,
                symbol: segs[1..].join("::"),
                target_type: Some(first.to_string()),
                leaf,
                owner_is_type: true,
                confidence: confidence.to_string(),
                basis: format!("`{first}` is a target type in scope; associated function call"),
            });
        }

        if segs.len() == 1 {
            let name = first;
            match self.name_in_scope(name) {
                NameScope::Local => return None,
                NameScope::Import(it) if !it.is_target => return None,
                _ => {}
            }
            if let Some(it) = self.target_import_for(name) {
                let symbol = it.full_path[1..].join("::");
                return Some(PathVerdict {
                    target_crate: it.root.clone(),
                    symbol,
                    target_type: None,
                    leaf: leaf.clone(),
                    owner_is_type: false,
                    confidence: "high".to_string(),
                    basis: format!("`{name}` is imported from target crate `{}`", it.root),
                });
            }
            if (self.cfg.is_target_fn(name) || self.cfg.is_target_type(name))
                && !self.has_non_target_glob()
            {
                if let Some(c) = self.target_glob_crate() {
                    let is_type = self.cfg.is_target_type(name);
                    return Some(PathVerdict {
                        target_crate: c,
                        symbol: name.to_string(),
                        target_type: if is_type {
                            Some(name.to_string())
                        } else {
                            None
                        },
                        leaf: leaf.clone(),
                        owner_is_type: is_type,
                        confidence: "medium".to_string(),
                        basis: "name matches a target symbol and a target glob import is in scope"
                            .to_string(),
                    });
                }
            }
        }
        None
    }

    fn operation_kind_for(&self, v: &PathVerdict) -> String {
        if let Some(t) = &v.target_type {
            if self.cfg.is_variant(t, &v.leaf) || (self.cfg.is_tuple_struct(t) && v.leaf == *t) {
                return "constructor".to_string();
            }
            return "associated_fn".to_string();
        }
        if self.cfg.is_tuple_struct(&v.leaf) {
            return "constructor".to_string();
        }
        if self.cfg.is_target_fn(&v.leaf) {
            return "free_fn".to_string();
        }
        "unclassified".to_string()
    }

    // ── receiver inference ─────────────────────────────────────────────────

    fn analyze_receiver(&self, expr: &Expr) -> ReceiverInfo {
        let text = self.text_of(expr.span());
        match expr {
            Expr::Path(p) if p.path.segments.len() == 1 => {
                let name = p.path.segments[0].ident.to_string();
                if name == "self" {
                    if let Some(Some(t)) = self.impl_types.last() {
                        let verdict = self.classify_type(t);
                        return ReceiverInfo {
                            text,
                            verdict,
                            ty_text: Some(t.clone()),
                            evidence: vec![Evidence {
                                kind: "impl_self".to_string(),
                                text: format!("self in `impl {t}`"),
                                file: self.file.to_string(),
                                line: 0,
                                column: 0,
                            }],
                            dynamic: false,
                        };
                    }
                    return ReceiverInfo {
                        text,
                        verdict: TypeVerdict::Unknown,
                        ty_text: None,
                        evidence: vec![],
                        dynamic: false,
                    };
                }
                if let Some(b) = self.lookup_binding(&name) {
                    // A binding introduced by an uninferable initializer
                    // deliberately shadows any older binding of the same name.
                    if b.confidence == "unattributed" || b.ty.is_empty() {
                        return ReceiverInfo {
                            text,
                            verdict: TypeVerdict::Unknown,
                            ty_text: None,
                            evidence: vec![Evidence {
                                kind: "shadowed_binding".to_string(),
                                text: format!("{name}: <unknown> ({})", b.basis),
                                file: b.file.clone(),
                                line: b.line,
                                column: b.column,
                            }],
                            dynamic: false,
                        };
                    }
                    let verdict = self.classify_type(&b.ty);
                    let dynamic = b.ty.contains("dyn ");
                    return ReceiverInfo {
                        text,
                        verdict,
                        ty_text: Some(b.ty.clone()),
                        evidence: vec![Evidence {
                            kind: b.basis.clone(),
                            text: format!("{name}: {}", b.ty),
                            file: b.file.clone(),
                            line: b.line,
                            column: b.column,
                        }],
                        dynamic,
                    };
                }
                ReceiverInfo {
                    text,
                    verdict: TypeVerdict::Unknown,
                    ty_text: None,
                    evidence: vec![],
                    dynamic: false,
                }
            }
            Expr::Reference(r) => self.analyze_receiver(&r.expr),
            Expr::Paren(p) => self.analyze_receiver(&p.expr),
            Expr::Group(g) => self.analyze_receiver(&g.expr),
            Expr::Unary(u) if matches!(u.op, syn::UnOp::Deref(_)) => self.analyze_receiver(&u.expr),
            Expr::Field(f) => {
                let base = self.analyze_receiver(&f.base);
                let field_name = match &f.member {
                    syn::Member::Named(id) => id.to_string(),
                    syn::Member::Unnamed(_) => {
                        return ReceiverInfo {
                            text,
                            verdict: TypeVerdict::Unknown,
                            ty_text: None,
                            evidence: base.evidence,
                            dynamic: false,
                        }
                    }
                };
                let base_type_name = base
                    .ty_text
                    .as_deref()
                    .and_then(type_base_name)
                    .map(|s| s.to_string());
                let field_ty = base_type_name.as_deref().and_then(|t| {
                    self.module_scopes
                        .iter()
                        .rev()
                        .find_map(|s| {
                            s.struct_fields
                                .get(t)
                                .and_then(|m| m.get(&field_name))
                                .cloned()
                        })
                        .or_else(|| self.cfg.field_type(t, &field_name).map(|s| s.to_string()))
                });
                if let Some(ft) = field_ty {
                    let verdict = self.classify_type(&ft);
                    let mut ev = base.evidence;
                    ev.push(Evidence {
                        kind: "field_type".to_string(),
                        text: format!("{}.{field_name}: {ft}", base.text),
                        file: self.file.to_string(),
                        line: 0,
                        column: 0,
                    });
                    ReceiverInfo {
                        text,
                        verdict,
                        ty_text: Some(ft),
                        evidence: ev,
                        dynamic: false,
                    }
                } else if matches!(base.verdict, TypeVerdict::Local | TypeVerdict::NonTarget) {
                    // The field belongs to a known non-target type and its type
                    // is unknown: it is not a target receiver.
                    ReceiverInfo {
                        text,
                        verdict: TypeVerdict::NonTarget,
                        ty_text: None,
                        evidence: base.evidence,
                        dynamic: false,
                    }
                } else {
                    let dynamic = base
                        .ty_text
                        .as_deref()
                        .map(|t| t.contains("dyn "))
                        .unwrap_or(false);
                    // Do NOT fall back to the base type: the field's type is
                    // unknown, and labelling the field with the container type
                    // is exactly how a Vec field gets read as a Table.
                    ReceiverInfo {
                        text,
                        verdict: TypeVerdict::Unknown,
                        ty_text: None,
                        evidence: base.evidence,
                        dynamic,
                    }
                }
            }
            _ => ReceiverInfo {
                text,
                verdict: TypeVerdict::Unknown,
                ty_text: None,
                evidence: vec![],
                dynamic: false,
            },
        }
    }

    fn infer_expr_type(&self, expr: &Expr) -> Option<TypeBinding> {
        match expr {
            Expr::Path(p) if p.path.segments.len() == 1 => {
                let name = p.path.segments[0].ident.to_string();
                self.lookup_binding(&name)
            }
            Expr::Reference(r) => self.infer_expr_type(&r.expr),
            Expr::Paren(p) => self.infer_expr_type(&p.expr),
            Expr::Group(g) => self.infer_expr_type(&g.expr),
            Expr::Struct(s) => Some(TypeBinding {
                ty: s.path.to_token_stream().to_string(),
                confidence: "medium".to_string(),
                basis: "struct literal binding".to_string(),
                file: self.file.to_string(),
                line: 0,
                column: 0,
            }),
            Expr::Call(c) => {
                if let Expr::Path(p) = &*c.func {
                    let segs: Vec<String> = p
                        .path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect();
                    if segs.len() >= 2 {
                        let method = segs.last().unwrap().as_str();
                        let owner = &segs[segs.len() - 2];
                        // Only a known constructor returns the owner type. This
                        // is deliberately narrow: `Arc::new(devices)` must NOT
                        // inherit `devices`'s type.
                        if CONSTRUCTOR_METHODS.contains(&method)
                            && (self.cfg.is_target_type(owner)
                                || self.cfg.is_target_crate(&segs[0]))
                        {
                            return Some(TypeBinding {
                                ty: segs[..segs.len() - 1].join("::"),
                                confidence: "medium".to_string(),
                                basis: format!("constructor call `{}`", segs.join("::")),
                                file: self.file.to_string(),
                                line: 0,
                                column: 0,
                            });
                        }
                    }
                }
                None
            }
            Expr::Field(_) => {
                let info = self.analyze_receiver(expr);
                info.ty_text.map(|t| TypeBinding {
                    ty: t,
                    confidence: "medium".to_string(),
                    basis: "field read binding".to_string(),
                    file: self.file.to_string(),
                    line: 0,
                    column: 0,
                })
            }
            _ => None,
        }
    }

    // ── record emission ────────────────────────────────────────────────────

    fn record_method_call(&mut self, node: &ExprMethodCall) {
        self.out.method_calls_total += 1;
        let method = node.method.to_string();
        let recv = self.analyze_receiver(&node.receiver);
        let recv_text = recv.text.clone();
        let callee_text = format!("{}.{}", recv_text, method);
        let caller = self.current_caller();
        let module_path = self.module_path_string();
        let span = Self::span_rec(node.span());

        let base =
            |category: &str, confidence: &str, resolution: String, candidate: bool| CallRecord {
                id: self.id("call", node.span()),
                record_kind: "method".to_string(),
                category: category.to_string(),
                operation_kind: "method".to_string(),
                confidence: confidence.to_string(),
                is_call: true,
                source_location: self.source_location.to_string(),
                reachability: "not_determined".to_string(),
                file: self.file.to_string(),
                crate_name: self.crate_name.map(|s| s.to_string()),
                module_path: module_path.clone(),
                caller: caller.clone(),
                callee_text: callee_text.clone(),
                callee_path: None,
                method: Some(method.clone()),
                receiver_text: Some(recv_text.clone()),
                receiver_type: recv.ty_text.clone(),
                target_crate: None,
                target_symbol: None,
                target_type: None,
                attribution_basis: resolution,
                evidence: recv.evidence.clone(),
                cfg: caller.cfg.clone(),
                is_test_context: caller.is_test,
                macro_unexpanded: false,
                dynamic_dispatch: false,
                candidate,
                span: span.clone(),
            };

        match &recv.verdict {
            TypeVerdict::Target {
                ty,
                target_type,
                target_crate,
                confidence,
                basis,
            } => {
                let mut rec = base(
                    CAT_METHOD_ATTRIBUTED,
                    confidence,
                    format!("receiver type `{ty}`: {basis}"),
                    false,
                );
                rec.target_crate = Some(target_crate.clone());
                rec.target_type = Some(target_type.clone());
                rec.target_symbol = Some(format!("{}::{}", target_type, method));
                self.out.calls.push(rec);
            }
            _ if recv.dynamic => {
                if !self.has_target_signal() {
                    self.note_other_method(&method);
                } else {
                    let mut rec = base(
                        CAT_METHOD_DYNAMIC,
                        "dynamic",
                        format!(
                            "receiver type `{}` is a trait object or generic bound; dispatch cannot be attributed syntactically",
                            recv.ty_text.clone().unwrap_or_default()
                        ),
                        true,
                    );
                    rec.dynamic_dispatch = true;
                    self.out.calls.push(rec);
                }
            }
            TypeVerdict::Ambiguous { name, basis } => {
                let rec = base(
                    CAT_METHOD_AMBIGUOUS,
                    "ambiguous",
                    format!("receiver type `{name}`: {basis}"),
                    true,
                );
                self.out.calls.push(rec);
            }
            TypeVerdict::CandidateName { name } => {
                let rec = base(
                    CAT_METHOD_CANDIDATE,
                    "unattributed",
                    format!(
                        "receiver type `{name}` matches a configured target type name but is not imported in this module; not attributed"
                    ),
                    true,
                );
                self.out.calls.push(rec);
            }
            TypeVerdict::Local | TypeVerdict::NonTarget => {
                self.note_other_method(&method);
            }
            TypeVerdict::Unknown => {
                if let Some((trait_name, crate_name)) = self.heuristic_trait_for(&method) {
                    let mut rec = base(
                        CAT_METHOD_TRAIT_HEURISTIC,
                        "heuristic",
                        format!(
                            "`{trait_name}` is imported from target crate `{crate_name}` and provides `{method}`; receiver type unknown, so this is a labelled heuristic attribution, not a typed one"
                        ),
                        true,
                    );
                    rec.target_crate = Some(crate_name);
                    rec.target_type = Some(trait_name.clone());
                    rec.target_symbol = Some(format!("{trait_name}::{method}"));
                    self.out.calls.push(rec);
                } else if self.cfg.is_target_method(&method) && self.has_target_signal() {
                    let rec = base(
                        CAT_METHOD_CANDIDATE,
                        "unattributed",
                        format!(
                            "receiver type unknown; `{method}` is a target method name, so this is a candidate, not an attributed call"
                        ),
                        true,
                    );
                    self.out.calls.push(rec);
                } else {
                    self.note_other_method(&method);
                }
            }
        }
    }

    fn note_other_method(&mut self, method: &str) {
        self.out.other_method_total += 1;
        *self
            .out
            .other_method_by_name
            .entry(method.to_string())
            .or_insert(0) += 1;
    }

    fn record_direct_call(&mut self, node: &ExprCall, v: PathVerdict) {
        let caller = self.current_caller();
        let operation_kind = self.operation_kind_for(&v);
        let is_assoc = operation_kind == "associated_fn" || operation_kind == "constructor";
        let category = if is_assoc {
            CAT_ASSOC_CALL
        } else {
            CAT_DIRECT_CALL
        };
        let rec = CallRecord {
            id: self.id("call", node.span()),
            record_kind: "call".to_string(),
            category: category.to_string(),
            operation_kind,
            confidence: v.confidence.clone(),
            is_call: true,
            source_location: self.source_location.to_string(),
            reachability: "not_determined".to_string(),
            file: self.file.to_string(),
            crate_name: self.crate_name.map(|s| s.to_string()),
            module_path: self.module_path_string(),
            caller: caller.clone(),
            callee_text: self.text_of(node.func.span()),
            callee_path: Some(path_to_string_from_expr(&node.func)),
            method: None,
            receiver_text: None,
            receiver_type: None,
            target_crate: Some(v.target_crate.clone()),
            target_symbol: Some(v.symbol.clone()),
            target_type: v.target_type.clone(),
            attribution_basis: v.basis.clone(),
            evidence: vec![],
            cfg: caller.cfg.clone(),
            is_test_context: caller.is_test,
            macro_unexpanded: false,
            dynamic_dispatch: false,
            candidate: false,
            span: Self::span_rec(node.span()),
        };
        self.out.calls.push(rec);
    }

    fn record_import(
        &mut self,
        u: &ItemUse,
        path: Vec<String>,
        alias: Option<String>,
        is_glob: bool,
    ) {
        let root = path.first().cloned().unwrap_or_default();
        let is_target = self.cfg.is_target_crate(&root);
        let is_reexport = matches!(u.vis, Visibility::Public(_));
        let last = path.last().cloned().unwrap_or_default();
        if !is_target && !(is_reexport && self.cfg.is_target_type(&last)) {
            return;
        }
        let visibility = match &u.vis {
            Visibility::Public(_) => "pub",
            Visibility::Restricted(_) => "pub(crate/self/super/in)",
            Visibility::Inherited => "private",
        }
        .to_string();
        let confidence = if is_target { "high" } else { "unattributed" };
        let resolution = if is_target {
            format!("root `{root}` is a configured target crate")
        } else {
            format!(
                "`pub use` of `{}`; the chain back to a target crate is not followed",
                path.join("::")
            )
        };
        let s = u.span().start();
        self.out.imports.push(ImportRecord {
            id: self.id_with("import", u.span(), &path.join("::")),
            file: self.file.to_string(),
            source_location: self.source_location.to_string(),
            crate_name: self.crate_name.map(|s| s.to_string()),
            module_path: self.module_path_string(),
            line: s.line as u32,
            column: s.column as u32,
            text: self.text_of(u.span()),
            path: path.join("::"),
            root,
            alias,
            is_glob,
            is_reexport,
            visibility,
            is_target,
            confidence: confidence.to_string(),
            attribution_basis: resolution,
        });
    }

    fn record_derive(&mut self, d: &syn::Path, span: proc_macro2::Span) {
        let segs: Vec<String> = d.segments.iter().map(|s| s.ident.to_string()).collect();
        if segs.is_empty() {
            return;
        }
        let name = segs.last().unwrap().clone();
        let first = segs[0].clone();
        let via_crate = self.cfg.is_target_crate(&first);
        let via_name = self.cfg.is_target_macro(&name);
        let via_import = self.target_import_for(&name).is_some();
        let is_target = via_crate || via_name || via_import;
        if !is_target {
            return;
        }
        let resolution = if via_crate {
            format!("derive path root `{first}` is a target crate")
        } else if via_name {
            format!("derive name `{name}` is in the target macro inventory")
        } else {
            format!("derive name `{name}` is imported from a target crate")
        };
        let caller = self.current_caller();
        let s = span.start();
        self.out.macros.push(MacroRecord {
            id: self.id_with("derive", span, &name),
            file: self.file.to_string(),
            source_location: self.source_location.to_string(),
            crate_name: self.crate_name.map(|s| s.to_string()),
            module_path: self.module_path_string(),
            caller: caller.clone(),
            line: s.line as u32,
            column: s.column as u32,
            kind: "derive".to_string(),
            name,
            path: Some(segs.join("::")),
            is_target: true,
            confidence: "high".to_string(),
            attribution_basis: resolution,
            unexpanded: true,
            cfg: caller.cfg.clone(),
        });
    }

    fn record_macro_invocation(&mut self, mac: &syn::Macro) {
        self.out.macro_invocations_total += 1;
        let segs: Vec<String> = mac
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        if segs.is_empty() {
            return;
        }
        let name = segs.last().unwrap().clone();
        let first = segs[0].clone();
        let via_crate = self.cfg.is_target_crate(&first);
        let via_name = self.cfg.is_target_macro(&name);
        let via_import = self.target_import_for(&name).is_some();
        let is_target = via_crate || via_name || via_import;
        if !is_target {
            self.out.unqualified_macro_total += 1;
            *self.out.unqualified_macro_by_name.entry(name).or_insert(0) += 1;
            return;
        }
        let resolution = if via_crate {
            format!("macro path root `{first}` is a target crate")
        } else if via_name {
            format!("macro name `{name}` is in the target macro inventory")
        } else {
            format!("macro name `{name}` is imported from a target crate")
        };
        let caller = self.current_caller();
        let s = mac.span().start();
        self.out.macros.push(MacroRecord {
            id: self.id("macro", mac.span()),
            file: self.file.to_string(),
            source_location: self.source_location.to_string(),
            crate_name: self.crate_name.map(|s| s.to_string()),
            module_path: self.module_path_string(),
            caller: caller.clone(),
            line: s.line as u32,
            column: s.column as u32,
            kind: "macro_invocation".to_string(),
            name,
            path: Some(segs.join("::")),
            is_target: true,
            confidence: if via_crate || via_import {
                "high"
            } else {
                "medium"
            }
            .to_string(),
            attribution_basis: resolution,
            unexpanded: true,
            cfg: caller.cfg.clone(),
        });
    }

    fn record_macro_rules_def(&mut self, node: &ItemMacro) {
        let Some(ident) = &node.ident else { return };
        let caller = self.current_caller();
        let s = node.span().start();
        self.out.macros.push(MacroRecord {
            id: self.id_with("macro_rules", node.span(), &ident.to_string()),
            file: self.file.to_string(),
            source_location: self.source_location.to_string(),
            crate_name: self.crate_name.map(|s| s.to_string()),
            module_path: self.module_path_string(),
            caller,
            line: s.line as u32,
            column: s.column as u32,
            kind: "macro_rules_def".to_string(),
            name: ident.to_string(),
            path: None,
            is_target: false,
            confidence: "n/a".to_string(),
            attribution_basis:
                "local macro definition; body is not expanded and calls inside it are not censused"
                    .to_string(),
            unexpanded: true,
            cfg: Vec::new(),
        });
    }

    fn note_attrs(&mut self, attrs: &[Attribute], owner_span: proc_macro2::Span) {
        for attr in attrs {
            let p = attr.path();
            if p.is_ident("cfg") || p.is_ident("cfg_attr") {
                continue;
            }
            if p.is_ident("derive") {
                let mut derives: Vec<syn::Path> = Vec::new();
                let _ = attr.parse_nested_meta(|meta| {
                    derives.push(meta.path.clone());
                    Ok(())
                });
                for d in derives {
                    self.record_derive(&d, attr.span());
                }
                continue;
            }
            let segs: Vec<String> = p.segments.iter().map(|s| s.ident.to_string()).collect();
            if segs.is_empty() {
                continue;
            }
            let name = segs.last().unwrap().clone();
            let first = segs[0].clone();
            let via_crate = self.cfg.is_target_crate(&first);
            let via_name = self.cfg.is_target_macro(&name);
            let via_import = self.target_import_for(&name).is_some();
            if via_crate || via_name || via_import {
                let caller = self.current_caller();
                let s = attr.span().start();
                self.out.macros.push(MacroRecord {
                    id: self.id("attr_macro", attr.span()),
                    file: self.file.to_string(),
                    source_location: self.source_location.to_string(),
                    crate_name: self.crate_name.map(|s| s.to_string()),
                    module_path: self.module_path_string(),
                    caller,
                    line: s.line as u32,
                    column: s.column as u32,
                    kind: "attribute_macro".to_string(),
                    name,
                    path: Some(segs.join("::")),
                    is_target: true,
                    confidence: "high".to_string(),
                    attribution_basis:
                        "attribute path resolves to a target crate or target macro inventory"
                            .to_string(),
                    unexpanded: true,
                    cfg: Vec::new(),
                });
            }
            let _ = owner_span;
        }
    }

    fn push_fn(
        &mut self,
        sig: &Signature,
        attrs: &[Attribute],
        kind: &str,
        span: proc_macro2::Span,
    ) {
        let impl_type = self.impl_types.last().cloned().flatten();
        let name = sig.ident.to_string();
        let qualified = match &impl_type {
            Some(t) => format!("{t}::{name}"),
            None => name.clone(),
        };
        let cfg = self.combined_cfg(&cfg_strings(attrs));
        let is_test = attrs_have_test(attrs)
            || self.cfg_test_depth > 0
            || self.source_location == "tests"
            || cfg.iter().any(|c| c.contains("test"));
        let caller = CallerCtx {
            kind: if impl_type.is_some() {
                "method".to_string()
            } else {
                kind.to_string()
            },
            name,
            qualified,
            impl_type,
            module_path: self.module_path_string(),
            is_test,
            cfg,
            span: Some(Self::span_rec(span)),
        };
        self.callers.push(caller);
        self.type_scopes.push(HashMap::new());
        for input in &sig.inputs {
            match input {
                FnArg::Receiver(r) => {
                    let b = if let Some(Some(t)) = self.impl_types.last() {
                        TypeBinding {
                            ty: t.clone(),
                            confidence: "high".to_string(),
                            basis: "method receiver self".to_string(),
                            file: self.file.to_string(),
                            line: 0,
                            column: 0,
                        }
                    } else {
                        TypeBinding {
                            ty: r.ty.to_token_stream().to_string(),
                            confidence: "high".to_string(),
                            basis: "receiver type annotation".to_string(),
                            file: self.file.to_string(),
                            line: 0,
                            column: 0,
                        }
                    };
                    self.type_scopes
                        .last_mut()
                        .unwrap()
                        .insert("self".to_string(), b);
                }
                FnArg::Typed(pt) => {
                    if let Pat::Ident(pi) = &*pt.pat {
                        let b = TypeBinding {
                            ty: pt.ty.to_token_stream().to_string(),
                            confidence: "high".to_string(),
                            basis: "parameter type annotation".to_string(),
                            file: self.file.to_string(),
                            line: 0,
                            column: 0,
                        };
                        self.type_scopes
                            .last_mut()
                            .unwrap()
                            .insert(pi.ident.to_string(), b);
                    }
                }
            }
        }
    }

    fn pop_fn(&mut self) {
        self.type_scopes.pop();
        self.callers.pop();
    }

    /// Binds a pattern name to a type. `ty` may be empty, in which case the
    /// binding deliberately shadows any older binding of the same name with an
    /// unknown type.
    fn bind_local(&mut self, pat: &Pat, ty: String, basis: &str, confidence: &str) {
        let name = match pat {
            Pat::Ident(pi) if pi.subpat.is_none() => pi.ident.to_string(),
            _ => return,
        };
        let b = TypeBinding {
            ty,
            confidence: confidence.to_string(),
            basis: basis.to_string(),
            file: self.file.to_string(),
            line: 0,
            column: 0,
        };
        self.type_scopes.last_mut().map(|s| s.insert(name, b));
    }

    fn invalidate_pat(&mut self, pat: &Pat) {
        self.bind_local(
            pat,
            String::new(),
            "shadowing binding with unknown type",
            "unattributed",
        );
    }
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

// ── the visitor ────────────────────────────────────────────────────────────

impl<'a> Visit<'a> for Census<'a> {
    fn visit_item(&mut self, node: &'a Item) {
        let attrs: Vec<Attribute> = match node {
            Item::Const(i) => i.attrs.clone(),
            Item::Enum(i) => i.attrs.clone(),
            Item::ExternCrate(i) => i.attrs.clone(),
            Item::Fn(i) => i.attrs.clone(),
            Item::ForeignMod(i) => i.attrs.clone(),
            Item::Impl(i) => i.attrs.clone(),
            Item::Macro(i) => i.attrs.clone(),
            Item::Mod(i) => i.attrs.clone(),
            Item::Static(i) => i.attrs.clone(),
            Item::Struct(i) => i.attrs.clone(),
            Item::Trait(i) => i.attrs.clone(),
            Item::TraitAlias(i) => i.attrs.clone(),
            Item::Type(i) => i.attrs.clone(),
            Item::Union(i) => i.attrs.clone(),
            Item::Use(i) => i.attrs.clone(),
            _ => Vec::new(),
        };
        self.note_attrs(&attrs, node.span());
        syn::visit::visit_item(self, node);
    }

    fn visit_impl_item(&mut self, node: &'a ImplItem) {
        let attrs = impl_item_attrs(node);
        self.note_attrs(&attrs, node.span());
        syn::visit::visit_impl_item(self, node);
    }

    fn visit_trait_item(&mut self, node: &'a TraitItem) {
        let attrs = trait_item_attrs(node);
        self.note_attrs(&attrs, node.span());
        syn::visit::visit_trait_item(self, node);
    }

    fn visit_item_mod(&mut self, node: &'a ItemMod) {
        let is_cfg_test = attrs_have_cfg_test(&node.attrs);
        if let Some((_, items)) = &node.content {
            let mut scope = Scope {
                path: self.current_scope_path(),
                ..Scope::default()
            };
            scope.path.push(node.ident.to_string());
            self.module_scopes.push(scope);
            self.prescan_items(items);
            let pushed = self.combined_cfg(&cfg_strings(&node.attrs));
            let added = pushed.len().saturating_sub(self.cfg_stack.len());
            let base = self.cfg_stack.len();
            self.cfg_stack = pushed;
            if is_cfg_test {
                self.cfg_test_depth += 1;
            }
            syn::visit::visit_item_mod(self, node);
            if is_cfg_test {
                self.cfg_test_depth -= 1;
            }
            self.cfg_stack.truncate(base);
            let _ = added;
            self.module_scopes.pop();
        } else {
            syn::visit::visit_item_mod(self, node);
        }
    }

    fn visit_item_use(&mut self, node: &'a ItemUse) {
        {
            let cfg = self.cfg;
            if let Some(scope) = self.block_scopes.last_mut() {
                register_use(cfg, scope, &node.tree);
            } else {
                let scope = self.module_scopes.last_mut().expect("scope");
                register_use(cfg, scope, &node.tree);
            }
        }
        let mut entries = Vec::new();
        flatten_use_tree(&[], &node.tree, &mut entries);
        for (path, alias, is_glob) in entries {
            self.record_import(node, path, alias, is_glob);
        }
    }

    fn visit_item_impl(&mut self, node: &'a ItemImpl) {
        let name = impl_base_name(&node.self_ty);
        self.impl_types.push(name);
        let pushed = self.combined_cfg(&cfg_strings(&node.attrs));
        let base = self.cfg_stack.len();
        self.cfg_stack = pushed;
        syn::visit::visit_item_impl(self, node);
        self.cfg_stack.truncate(base);
        self.impl_types.pop();
    }

    fn visit_item_fn(&mut self, node: &'a ItemFn) {
        self.push_fn(&node.sig, &node.attrs, "function", node.sig.ident.span());
        syn::visit::visit_item_fn(self, node);
        self.pop_fn();
    }

    fn visit_impl_item_fn(&mut self, node: &'a ImplItemFn) {
        self.push_fn(&node.sig, &node.attrs, "method", node.sig.ident.span());
        syn::visit::visit_impl_item_fn(self, node);
        self.pop_fn();
    }

    fn visit_trait_item_fn(&mut self, node: &'a TraitItemFn) {
        self.push_fn(&node.sig, &node.attrs, "method", node.sig.ident.span());
        syn::visit::visit_trait_item_fn(self, node);
        self.pop_fn();
    }

    fn visit_item_macro(&mut self, node: &'a ItemMacro) {
        if node.mac.path.is_ident("macro_rules") {
            self.record_macro_rules_def(node);
            return;
        }
        syn::visit::visit_item_macro(self, node);
    }

    fn visit_macro(&mut self, node: &'a syn::Macro) {
        if !node.path.is_ident("macro_rules") {
            self.record_macro_invocation(node);
        }
    }

    fn visit_block(&mut self, node: &'a syn::Block) {
        self.type_scopes.push(HashMap::new());
        self.block_scopes.push(Scope::default());
        syn::visit::visit_block(self, node);
        self.block_scopes.pop();
        self.type_scopes.pop();
    }

    fn visit_expr_closure(&mut self, node: &'a ExprClosure) {
        self.type_scopes.push(HashMap::new());
        for pat in &node.inputs {
            if let Pat::Ident(pi) = pat {
                let b = TypeBinding {
                    ty: String::new(),
                    confidence: "unattributed".to_string(),
                    basis: "closure parameter".to_string(),
                    file: self.file.to_string(),
                    line: 0,
                    column: 0,
                };
                self.type_scopes
                    .last_mut()
                    .unwrap()
                    .insert(pi.ident.to_string(), b);
            }
        }
        syn::visit::visit_expr_closure(self, node);
        self.type_scopes.pop();
    }

    fn visit_local(&mut self, node: &'a Local) {
        syn::visit::visit_local(self, node);
        if let Pat::Type(pt) = &node.pat {
            let t = pt.ty.to_token_stream().to_string();
            self.bind_local(&pt.pat, t, "local type annotation", "high");
            return;
        }
        match node
            .init
            .as_ref()
            .and_then(|i| self.infer_expr_type(&i.expr))
        {
            Some(b) => self.bind_local(&node.pat, b.ty, &b.basis, &b.confidence),
            // Even when the initializer's type is unknown, the `let` shadows any
            // older binding of the same name. Leaving the old type in place is
            // how `let devices = Arc::new(devices)` kept the stale `Table` type.
            None => self.invalidate_pat(&node.pat),
        }
    }

    fn visit_expr_for_loop(&mut self, node: &'a ExprForLoop) {
        self.type_scopes.push(HashMap::new());
        self.invalidate_pat(&node.pat);
        syn::visit::visit_expr_for_loop(self, node);
        self.type_scopes.pop();
    }

    fn visit_expr_assign(&mut self, node: &'a ExprAssign) {
        syn::visit::visit_expr_assign(self, node);
        if let Expr::Path(p) = &*node.left {
            if p.path.segments.len() == 1 {
                let name = p.path.segments[0].ident.to_string();
                let binding = match self.infer_expr_type(&node.right) {
                    Some(b) => TypeBinding {
                        ty: b.ty,
                        confidence: b.confidence,
                        basis: format!("assignment binding ({})", b.basis),
                        file: self.file.to_string(),
                        line: 0,
                        column: 0,
                    },
                    None => TypeBinding {
                        ty: String::new(),
                        confidence: "unattributed".to_string(),
                        basis: "assignment with unknown right-hand side".to_string(),
                        file: self.file.to_string(),
                        line: 0,
                        column: 0,
                    },
                };
                self.type_scopes.last_mut().map(|s| s.insert(name, binding));
            }
        }
    }

    fn visit_expr_call(&mut self, node: &'a ExprCall) {
        if let Expr::Path(p) = &*node.func {
            if let Some(v) = self.classify_path(&p.path) {
                self.record_direct_call(node, v);
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'a ExprMethodCall) {
        self.record_method_call(node);
        syn::visit::visit_expr_method_call(self, node);
    }
}
