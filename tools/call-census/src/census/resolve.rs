//! Span/text, scope/import, type and receiver resolution for the census.
//!
//! Split from `census.rs`; the logic is unchanged. Methods called from the
//! emission half or from `census_file` are `pub(super)`.

use super::*;
use quote::ToTokens;
use std::collections::BTreeMap;
use syn::spanned::Spanned;
use syn::{Expr, Item};

impl<'a> Census<'a> {
    // ── span / text helpers ────────────────────────────────────────────────

    pub(super) fn text_of(&self, span: proc_macro2::Span) -> String {
        let r = span.byte_range();
        if r.start <= r.end && r.end <= self.src.len() {
            self.src[r].trim().to_string()
        } else {
            String::new()
        }
    }

    pub(super) fn span_rec(span: proc_macro2::Span) -> SpanRec {
        let s = span.start();
        let e = span.end();
        SpanRec {
            line: s.line as u32,
            column: s.column as u32,
            end_line: e.line as u32,
            end_column: e.column as u32,
        }
    }

    pub(super) fn id(&self, prefix: &str, span: proc_macro2::Span) -> String {
        let s = span.start();
        let e = span.end();
        format!(
            "{}:{}:{}:{}-{}:{}",
            prefix, self.file, s.line, s.column, e.line, e.column
        )
    }

    /// Identifier for a record whose owner span may be shared by several
    /// entries (grouped imports, comma-separated derives).
    pub(super) fn id_with(
        &self,
        prefix: &str,
        span: proc_macro2::Span,
        discriminator: &str,
    ) -> String {
        format!("{}:{}", self.id(prefix, span), discriminator)
    }

    pub(super) fn module_path_string(&self) -> String {
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

    pub(super) fn current_scope_path(&self) -> Vec<String> {
        self.module_scopes
            .last()
            .map(|s| s.path.clone())
            .unwrap_or_default()
    }

    pub(super) fn combined_cfg(&self, extra: &[String]) -> Vec<String> {
        let mut v = self.cfg_stack.clone();
        v.extend(extra.iter().cloned());
        v
    }

    pub(super) fn current_caller(&self) -> CallerCtx {
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

    pub(super) fn prescan_items(&mut self, items: &[Item]) {
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

    pub(super) fn target_import_for(&self, name: &str) -> Option<ImportTarget> {
        self.lookup_import(name).filter(|it| it.is_target)
    }

    /// Whether this file has any reason to be considered a consumer at all: a
    /// target import/glob is in scope, or the source text names a target crate.
    /// This only gates *candidate* emission; it never turns a candidate into an
    /// attributed call.
    pub(super) fn has_target_signal(&self) -> bool {
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
    pub(super) fn heuristic_trait_for(&self, method: &str) -> Option<(String, String)> {
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

    pub(super) fn classify_path(&self, path: &syn::Path) -> Option<PathVerdict> {
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

    pub(super) fn operation_kind_for(&self, v: &PathVerdict) -> String {
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

    pub(super) fn analyze_receiver(&self, expr: &Expr) -> ReceiverInfo {
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

    pub(super) fn infer_expr_type(&self, expr: &Expr) -> Option<TypeBinding> {
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
}
