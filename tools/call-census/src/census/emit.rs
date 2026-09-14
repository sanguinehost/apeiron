//! Record emission and the `syn` visitor for the census.
//!
//! Split from `census.rs`; the logic is unchanged. Emission methods are
//! private to this module; the resolution helpers they call live in
//! `census::resolve`.

use super::*;
use crate::targets::impl_base_name;
use quote::ToTokens;
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{
    Attribute, Expr, ExprAssign, ExprCall, ExprClosure, ExprForLoop, ExprMethodCall, FnArg,
    ImplItem, ImplItemFn, Item, ItemFn, ItemImpl, ItemMacro, ItemMod, ItemUse, Local, Pat,
    Signature, TraitItem, TraitItemFn, Visibility,
};

impl<'a> Census<'a> {
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
