//! Name resolution and the call graph.
//!
//! A path resolves the way rustc resolves it, minus visibility: `crate`, `self`, `super`, `Self`,
//! then the module's own children and items, its `use` aliases, its glob imports, and finally the
//! crate names. A `pub use` is followed wherever it is reached, to depth 8. A method call resolves
//! through the receiver's type when the type is known (parameters, `let` annotations, constructor
//! and method return types, fields, `?`, `unwrap`, indexing, struct literals) and otherwise to every
//! crate method of that name, as a loose edge only.
//!
//! Two edge sets: `loose` holds every candidate; `precise` holds a call's candidates only when there
//! are at most [`PRECISE_FANOUT`] of them. Reachability for "is this fn used" runs on `loose` (an
//! over-approximation, so a used fn is never reported dead); class propagation for "does this fn
//! compute a forward" runs on `precise` (an untyped `.forward()` call does not make every caller a
//! forward).

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use proc_macro2::TokenStream;
use syn::punctuated::Punctuated;
use syn::visit::Visit;

use super::load::{ty_of, CrateKind, Func, Tree, Ty};

pub(crate) const PRECISE_FANOUT: usize = 3;
const DEPTH: usize = 8;

#[derive(Default)]
pub(crate) struct Index {
    /// Free fns by module path plus name; several when cfg alternates share the name.
    pub free: BTreeMap<Vec<String>, Vec<usize>>,
    /// Methods by (self type simple name, method name).
    pub methods: BTreeMap<(String, String), Vec<usize>>,
    /// Trait methods (impls and defaults) by (trait name, method name).
    pub trait_methods: BTreeMap<(String, String), Vec<usize>>,
    /// Every method by name: the candidates of an untyped receiver.
    pub by_method: BTreeMap<String, Vec<usize>>,
    /// Type definitions by declared simple name.
    pub types: BTreeMap<String, Vec<Vec<String>>>,
}

/// Cost: time O(F log F + T log T) over F fns and T types; space O(F + T).
pub(crate) fn index(tree: &Tree) -> Index {
    let mut idx = Index::default();
    for (i, f) in tree.fns.iter().enumerate() {
        match &f.self_ty {
            None => {
                let mut path = f.module.clone();
                path.push(f.name.clone());
                idx.free.entry(path).or_default().push(i);
            }
            Some(t) => {
                idx.methods
                    .entry((t.clone(), f.name.clone()))
                    .or_default()
                    .push(i);
                idx.by_method.entry(f.name.clone()).or_default().push(i);
                if let Some(tr) = &f.trait_name {
                    idx.trait_methods
                        .entry((tr.clone(), f.name.clone()))
                        .or_default()
                        .push(i);
                }
            }
        }
    }
    for path in tree.types.keys() {
        if let Some(name) = path.last() {
            idx.types
                .entry(name.clone())
                .or_default()
                .push(path.clone());
        }
    }
    idx
}

#[derive(Clone, Debug)]
pub(crate) enum Res {
    Module(Vec<String>),
    Fns(Vec<usize>),
    /// A type or trait, by the simple name its methods are indexed under.
    Type(String),
    External(Vec<String>),
    Value,
}

type Key = (Vec<String>, String);

pub(crate) struct Resolver<'a> {
    pub tree: &'a Tree,
    pub idx: &'a Index,
    /// `lookup` answers by `(module, name)`. An answer computed while a glob cycle or the depth
    /// guard cut the search is not stored, so a cached answer is the uncut answer.
    memo: RefCell<HashMap<Key, Option<Res>>>,
    active: RefCell<HashSet<Key>>,
    cut: Cell<bool>,
}

impl<'a> Resolver<'a> {
    pub fn new(tree: &'a Tree, idx: &'a Index) -> Self {
        Resolver {
            tree,
            idx,
            memo: RefCell::new(HashMap::new()),
            active: RefCell::new(HashSet::new()),
            cut: Cell::new(false),
        }
    }

    /// What `name` means inside `module`, memoised.
    ///
    /// Cost: each `(module, name)` is computed once (amortised O(G) over the G globs of the module
    /// chain); a repeat is one hash lookup. Without the memo, a missing name walks every glob chain
    /// to depth 8 on every occurrence.
    fn lookup(&self, module: &[String], name: &str, depth: usize) -> Option<Res> {
        if depth > DEPTH {
            self.cut.set(true);
            return None;
        }
        let key: Key = (module.to_vec(), name.to_string());
        if let Some(r) = self.memo.borrow().get(&key) {
            return r.clone();
        }
        if !self.active.borrow_mut().insert(key.clone()) {
            self.cut.set(true);
            return None;
        }
        let outer = self.cut.replace(false);
        let r = self.lookup_uncached(module, name, depth);
        self.active.borrow_mut().remove(&key);
        if !self.cut.get() {
            self.memo.borrow_mut().insert(key, r.clone());
        }
        self.cut.set(outer || self.cut.get());
        r
    }

    fn lookup_uncached(&self, module: &[String], name: &str, depth: usize) -> Option<Res> {
        let m = self.tree.modules.get(module)?;
        let mut path = module.to_vec();
        path.push(name.to_string());
        if m.children.contains(name) {
            return Some(Res::Module(path));
        }
        if let Some(ids) = self.idx.free.get(&path) {
            return Some(Res::Fns(ids.clone()));
        }
        if let Some(td) = self.tree.types.get(&path) {
            return Some(Res::Type(td.name.clone()));
        }
        if let Some(segs) = m.uses.get(name) {
            // `use foo::foo;` names an extern crate's item, not this alias again.
            if segs.first().map(String::as_str) != Some(name) || segs.len() > 1 {
                if let Some(r) = self.resolve(module, segs, depth + 1) {
                    return Some(r);
                }
            }
        }
        for g in &m.globs {
            match self.resolve(module, g, depth + 1) {
                Some(Res::Module(gm)) if gm != module => {
                    if let Some(r) = self.lookup(&gm, name, depth + 1) {
                        return Some(r);
                    }
                }
                // `use Enum::*`: a variant, not a fn.
                Some(Res::Type(_)) => {}
                _ => {}
            }
        }
        None
    }

    /// Resolves a path written inside `module`.
    pub fn resolve(&self, module: &[String], segs: &[String], depth: usize) -> Option<Res> {
        if depth > DEPTH {
            self.cut.set(true);
            return None;
        }
        let (first, rest) = segs.split_first()?;
        let mut cur = match first.as_str() {
            "crate" => Res::Module(vec![module.first()?.clone()]),
            "self" => Res::Module(module.to_vec()),
            "super" => Res::Module(module[..module.len().saturating_sub(1).max(1)].to_vec()),
            name => match self.lookup(module, name, depth) {
                Some(r) => r,
                None => {
                    let root = vec![name.to_string()];
                    match self.tree.modules.get(&root) {
                        Some(m) if self.tree.krates[m.krate].kind == CrateKind::Lib => {
                            Res::Module(root)
                        }
                        _ => Res::External(root),
                    }
                }
            },
        };
        for s in rest {
            cur = self.member(cur, s, depth)?;
        }
        Some(cur)
    }

    pub fn member(&self, cur: Res, seg: &str, depth: usize) -> Option<Res> {
        match cur {
            Res::Module(m) => {
                if seg == "super" {
                    return Some(Res::Module(m[..m.len().saturating_sub(1).max(1)].to_vec()));
                }
                if seg == "self" {
                    return Some(Res::Module(m));
                }
                self.lookup(&m, seg, depth + 1)
            }
            Res::Type(t) => {
                let ids = self.type_methods(&t, seg);
                if ids.is_empty() {
                    Some(Res::Value)
                } else {
                    Some(Res::Fns(ids))
                }
            }
            Res::External(mut v) => {
                v.push(seg.to_string());
                Some(Res::External(v))
            }
            Res::Fns(_) | Res::Value => None,
        }
    }

    /// The methods named `name` callable on type `t`: inherent and trait impls on `t`, the
    /// defaults of the traits `t` implements, and, when `t` is a trait, every impl of it.
    pub fn type_methods(&self, t: &str, name: &str) -> Vec<usize> {
        let mut out: BTreeSet<usize> = BTreeSet::new();
        let key = (t.to_string(), name.to_string());
        if let Some(ids) = self.idx.methods.get(&key) {
            out.extend(ids);
        }
        if let Some(ids) = self.idx.trait_methods.get(&key) {
            out.extend(ids);
        }
        if let Some(traits) = self.tree.impls.get(t) {
            for tr in traits {
                if let Some(ids) = self.idx.trait_methods.get(&(tr.clone(), name.to_string())) {
                    out.extend(ids);
                }
            }
        }
        out.into_iter().collect()
    }

    /// The declared type of field `member` of `t`, following one alias.
    fn field_ty(&self, t: &Ty, member: &str) -> Option<Ty> {
        for hop in 0..2 {
            let name = if hop == 0 {
                t.name.clone()
            } else {
                self.alias(&t.name)?
            };
            for path in self.idx.types.get(&name).into_iter().flatten() {
                if let Some(ft) = self.tree.types.get(path).and_then(|d| d.fields.get(member)) {
                    return Some(ft.clone());
                }
            }
        }
        None
    }

    fn alias(&self, name: &str) -> Option<String> {
        self.idx
            .types
            .get(name)?
            .iter()
            .filter_map(|p| self.tree.types.get(p))
            .find(|d| d.name != name)
            .map(|d| d.name.clone())
    }
}

pub(crate) fn segs(path: &syn::Path) -> Vec<String> {
    path.segments.iter().map(|s| s.ident.to_string()).collect()
}

/// The declared return type of a fn, with `Self` replaced by its impl type.
pub(crate) fn ret_ty(f: &Func) -> Option<Ty> {
    match &f.sig.output {
        syn::ReturnType::Default => None,
        syn::ReturnType::Type(_, t) => ty_of(t).map(|t| subst_self(t, f.self_ty.as_deref())),
    }
}

fn subst_self(mut t: Ty, self_ty: Option<&str>) -> Ty {
    if t.name == "Self" {
        if let Some(s) = self_ty {
            t.name = s.to_string();
        }
    }
    t.args = t.args.into_iter().map(|a| subst_self(a, self_ty)).collect();
    t
}

/// `Box<T>`, `Rc<T>`, `Arc<T>` and references read as `T` for method lookup.
fn deref(t: Ty) -> Ty {
    match t.name.as_str() {
        "Box" | "Rc" | "Arc" | "Cow" | "RefCell" | "Mutex" | "RwLock" | "MutexGuard"
            if !t.args.is_empty() =>
        {
            deref(t.args[0].clone())
        }
        _ => t,
    }
}

fn inner(t: Ty) -> Option<Ty> {
    match t.name.as_str() {
        "Option" | "Result" | "Vec" | "VecDeque" | "[]" | "Box" | "Rc" | "Arc" | "Cow"
        | "LockResult" => t.args.into_iter().next(),
        _ => None,
    }
}

/// Every `a::b::c` path in `tokens`, at any group depth, except a `$name` metavariable and the
/// fragment specifier after `$name:`.
///
/// Cost: time and space O(T) over the T tokens.
pub(crate) fn token_paths(tokens: &TokenStream) -> Vec<syn::Path> {
    use proc_macro2::{Spacing, TokenTree};
    let mut out = Vec::new();
    let mut cur: Vec<proc_macro2::Ident> = Vec::new();
    // `skip`: the previous token was `$` or a lone `:` (a metavariable or its specifier).
    let (mut skip, mut colons) = (false, 0);
    let flush = |cur: &mut Vec<proc_macro2::Ident>, out: &mut Vec<syn::Path>| {
        if !cur.is_empty() {
            let mut p = syn::Path::from(cur[0].clone());
            for i in &cur[1..] {
                p.segments.push(syn::PathSegment::from(i.clone()));
            }
            out.push(p);
            cur.clear();
        }
    };
    for tt in tokens.clone() {
        match tt {
            TokenTree::Ident(i) => {
                if skip {
                    skip = false;
                } else if colons == 2 && !cur.is_empty() {
                    cur.push(i);
                } else {
                    flush(&mut cur, &mut out);
                    cur.push(i);
                }
                colons = 0;
            }
            TokenTree::Punct(p) if p.as_char() == ':' => {
                if p.spacing() == Spacing::Joint || colons == 1 {
                    colons += 1;
                } else {
                    flush(&mut cur, &mut out);
                    skip = true;
                    colons = 0;
                }
            }
            TokenTree::Punct(p) => {
                flush(&mut cur, &mut out);
                skip = p.as_char() == '$';
                colons = 0;
            }
            TokenTree::Group(g) => {
                flush(&mut cur, &mut out);
                out.extend(token_paths(&g.stream()));
                skip = false;
                colons = 0;
            }
            TokenTree::Literal(_) => {
                flush(&mut cur, &mut out);
                skip = false;
                colons = 0;
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Parses a macro's tokens as comma-separated expressions, else as statements.
pub(crate) fn macro_exprs(tokens: &TokenStream) -> Vec<syn::Expr> {
    use syn::parse::Parser;
    if let Ok(list) =
        Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated.parse2(tokens.clone())
    {
        return list.into_iter().collect();
    }
    if let Ok(stmts) = syn::Block::parse_within.parse2(tokens.clone()) {
        return vec![syn::Expr::Block(syn::ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: syn::Block {
                brace_token: Default::default(),
                stmts,
            },
        })];
    }
    // `vec![x; n]`, `matches!(e, pat)`: the expressions before the separator.
    let mut out = Vec::new();
    let mut part = TokenStream::new();
    for tt in tokens.clone() {
        let sep = matches!(&tt, proc_macro2::TokenTree::Punct(p) if p.as_char() == ';' || p.as_char() == ',');
        if sep {
            if let Ok(e) = syn::parse2::<syn::Expr>(std::mem::take(&mut part)) {
                out.push(e);
            }
        } else {
            part.extend([tt]);
        }
    }
    if let Ok(e) = syn::parse2::<syn::Expr>(part) {
        out.push(e);
    }
    out
}

/// The typing environment and resolver of one fn body.
pub(crate) struct Scope<'a> {
    pub r: &'a Resolver<'a>,
    pub f: &'a Func,
    pub env: BTreeMap<String, Ty>,
}

impl<'a> Scope<'a> {
    pub fn new(r: &'a Resolver<'a>, f: &'a Func) -> Self {
        let mut env = BTreeMap::new();
        for input in &f.sig.inputs {
            match input {
                syn::FnArg::Receiver(_) => {
                    if let Some(t) = &f.self_ty {
                        env.insert("self".to_string(), Ty::named(t));
                    }
                }
                syn::FnArg::Typed(pt) => {
                    if let (syn::Pat::Ident(pi), Some(t)) = (&*pt.pat, ty_of(&pt.ty)) {
                        env.insert(pi.ident.to_string(), subst_self(t, f.self_ty.as_deref()));
                    }
                }
            }
        }
        Scope { r, f, env }
    }

    pub fn path_res(&self, path: &syn::Path) -> Option<Res> {
        let s = segs(path);
        if s.len() == 1 && self.env.contains_key(&s[0]) {
            return None;
        }
        if s.first().map(String::as_str) == Some("Self") {
            let mut cur = Res::Type(self.f.self_ty.clone()?);
            for seg in &s[1..] {
                cur = self.r.member(cur, seg, 0)?;
            }
            return Some(cur);
        }
        self.r.resolve(&self.f.module, &s, 0)
    }

    /// The method candidates of `recv.name(..)`, and whether the receiver's type selected them.
    /// An untyped receiver, or one typed by a type parameter, yields every crate method of that
    /// name, which is a reachability candidate set, never a precise edge.
    pub fn method(&self, recv: &syn::Expr, name: &str) -> (Vec<usize>, bool) {
        if let Some(t) = self.type_of(recv) {
            let t = deref(t);
            if self.f.generics.contains(&t.name) {
                return (
                    self.r.idx.by_method.get(name).cloned().unwrap_or_default(),
                    false,
                );
            }
            let ids = self.r.type_methods(&t.name, name);
            if !ids.is_empty() || !self.r.idx.types.contains_key(&t.name) {
                return (ids, true);
            }
        }
        (
            self.r.idx.by_method.get(name).cloned().unwrap_or_default(),
            false,
        )
    }

    pub fn bind_pat(&mut self, pat: &syn::Pat, init: Option<&syn::Expr>) {
        match pat {
            syn::Pat::Type(pt) => {
                if let (syn::Pat::Ident(pi), Some(t)) = (&*pt.pat, ty_of(&pt.ty)) {
                    self.env.insert(
                        pi.ident.to_string(),
                        subst_self(t, self.f.self_ty.as_deref()),
                    );
                }
            }
            syn::Pat::Ident(pi) => {
                let t = init.and_then(|e| self.type_of(e));
                match t {
                    Some(t) => {
                        self.env.insert(pi.ident.to_string(), t);
                    }
                    None => {
                        // A local of unknown type still shadows an item of the same name.
                        self.env.insert(pi.ident.to_string(), Ty::named("_"));
                    }
                }
            }
            _ => {}
        }
    }

    /// The type of an expression where it follows from declarations; `None` otherwise.
    pub fn type_of(&self, e: &syn::Expr) -> Option<Ty> {
        match e {
            syn::Expr::Path(p) => {
                let s = segs(&p.path);
                if s.len() == 1 {
                    if let Some(t) = self.env.get(&s[0]) {
                        return (t.name != "_").then(|| t.clone());
                    }
                }
                None
            }
            syn::Expr::Field(fe) => {
                let base = deref(self.type_of(&fe.base)?);
                let member = match &fe.member {
                    syn::Member::Named(i) => i.to_string(),
                    syn::Member::Unnamed(i) => i.index.to_string(),
                };
                if base.name == "()" {
                    return member
                        .parse::<usize>()
                        .ok()
                        .and_then(|k| base.args.get(k).cloned());
                }
                self.r.field_ty(&base, &member)
            }
            syn::Expr::MethodCall(m) => {
                let name = m.method.to_string();
                match name.as_str() {
                    "unwrap" | "expect" | "unwrap_or" | "unwrap_or_default" | "unwrap_or_else"
                    | "unwrap_unchecked" => inner(self.type_of(&m.receiver)?),
                    "clone" | "to_owned" | "as_ref" | "as_mut" | "borrow" | "borrow_mut"
                    | "as_slice" | "as_mut_slice" | "deref" => self.type_of(&m.receiver),
                    "lock" | "read" | "write" => self.type_of(&m.receiver).and_then(inner_lock),
                    _ => {
                        let (ids, _) = self.method(&m.receiver, &name);
                        self.common_ret(&ids)
                    }
                }
            }
            syn::Expr::Call(c) => match &*c.func {
                syn::Expr::Path(p) => match self.path_res(&p.path)? {
                    Res::Fns(ids) => self.common_ret(&ids),
                    Res::Type(t) => Some(Ty::named(&t)),
                    _ => None,
                },
                _ => None,
            },
            syn::Expr::Struct(s) => {
                let last = s.path.segments.last()?.ident.to_string();
                if last == "Self" {
                    return self.f.self_ty.as_deref().map(Ty::named);
                }
                Some(Ty::named(&last))
            }
            syn::Expr::Try(t) => inner(self.type_of(&t.expr)?),
            syn::Expr::Reference(r) => self.type_of(&r.expr),
            syn::Expr::Paren(p) => self.type_of(&p.expr),
            syn::Expr::Group(g) => self.type_of(&g.expr),
            syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Deref(_)) => {
                self.type_of(&u.expr).map(deref)
            }
            syn::Expr::Index(i) => {
                if matches!(&*i.index, syn::Expr::Range(_)) {
                    return self.type_of(&i.expr);
                }
                inner(deref(self.type_of(&i.expr)?))
            }
            syn::Expr::Cast(c) => ty_of(&c.ty),
            _ => None,
        }
    }

    /// The return type the candidates share; `None` when they disagree.
    fn common_ret(&self, ids: &[usize]) -> Option<Ty> {
        let mut out: Option<Ty> = None;
        for &i in ids {
            let t = ret_ty(&self.r.tree.fns[i])?;
            match &out {
                None => out = Some(t),
                Some(o) if *o == t => {}
                Some(_) => return None,
            }
        }
        out
    }
}

fn inner_lock(t: Ty) -> Option<Ty> {
    match t.name.as_str() {
        "Mutex" | "RwLock" | "RefCell" => Some(t),
        _ => None,
    }
}

/// The edges of one fn.
#[derive(Default, Clone)]
pub(crate) struct Edges {
    pub precise: BTreeSet<usize>,
    pub loose: BTreeSet<usize>,
    /// Paths into other crates (`std::env::var`), joined by `::`.
    pub external: BTreeSet<String>,
}

struct Walk<'a> {
    scope: Scope<'a>,
    out: Edges,
}

impl Walk<'_> {
    fn add(&mut self, ids: &[usize]) {
        self.out.loose.extend(ids);
        if ids.len() <= PRECISE_FANOUT {
            self.out.precise.extend(ids);
        }
    }
}

impl<'ast> Visit<'ast> for Walk<'_> {
    fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
        // `R::f` on a type parameter dispatches to every impl's `f`: a candidate set, as for a
        // method call on a receiver typed by a parameter.
        // `<R as Trait>::f` is the same dispatch.
        let s = segs(&p.path);
        let generic = match (&p.qself, s.as_slice()) {
            (Some(q), [.., name]) => ty_of(&q.ty)
                .filter(|t| self.scope.f.generics.contains(&t.name))
                .map(|_| name),
            (None, [ty, name]) if self.scope.f.generics.contains(ty) => Some(name),
            _ => None,
        };
        if let Some(name) = generic {
            if let Some(ids) = self.scope.r.idx.by_method.get(name) {
                self.out.loose.extend(ids);
            }
            return;
        }
        match self.scope.path_res(&p.path) {
            Some(Res::Fns(ids)) => self.add(&ids),
            Some(Res::External(v)) => {
                self.out.external.insert(v.join("::"));
            }
            _ => {}
        }
        syn::visit::visit_expr_path(self, p);
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let (ids, typed) = self.scope.method(&m.receiver, &m.method.to_string());
        if typed {
            self.add(&ids);
        } else {
            self.out.loose.extend(ids);
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_local(&mut self, l: &'ast syn::Local) {
        if let Some(init) = &l.init {
            self.visit_expr(&init.expr);
            if let Some((_, e)) = &init.diverge {
                self.visit_expr(e);
            }
        }
        self.scope
            .bind_pat(&l.pat, l.init.as_ref().map(|i| &*i.expr));
        // After binding: the pattern's const references and its ascribed type's lengths.
        self.visit_pat(&l.pat);
    }

    fn visit_expr_closure(&mut self, c: &'ast syn::ExprClosure) {
        for p in &c.inputs {
            self.scope.bind_pat(p, None);
        }
        syn::visit::visit_expr_closure(self, c);
    }

    fn visit_expr_for_loop(&mut self, f: &'ast syn::ExprForLoop) {
        self.visit_expr(&f.expr);
        if let syn::Pat::Ident(pi) = &*f.pat {
            let t = self
                .scope
                .type_of(&f.expr)
                .and_then(|t| inner(deref(t)))
                .unwrap_or_else(|| Ty::named("_"));
            self.scope.env.insert(pi.ident.to_string(), t);
        } else {
            self.scope.bind_pat(&f.pat, None);
        }
        self.visit_block(&f.body);
    }

    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        // A body-local `macro_rules!` is expanded only inside this body: every path its rules
        // name is a candidate reference.
        if m.path.is_ident("macro_rules") {
            for path in token_paths(&m.tokens) {
                if let Some(Res::Fns(ids)) = self.scope.path_res(&path) {
                    self.out.loose.extend(ids);
                }
            }
            return;
        }
        for e in macro_exprs(&m.tokens) {
            self.visit_expr(&e);
        }
        if let Some(Res::Fns(ids)) = self.scope.path_res(&m.path) {
            self.add(&ids);
        }
    }

    // A type path names no fn and reaches `visit_path`, not `visit_expr_path`; the default walk
    // still reaches the const expressions a type evaluates (`[u8; N]`, `Foo<{ N }>`).
    fn visit_pat(&mut self, p: &'ast syn::Pat) {
        if let syn::Pat::Ident(pi) = p {
            // An ident pattern naming a `const` in scope is that const, not a binding.
            if pi.by_ref.is_none() && pi.mutability.is_none() && pi.subpat.is_none() {
                let path = syn::Path::from(pi.ident.clone());
                if let Some(Res::Fns(ids)) = self.scope.path_res(&path) {
                    let consts: Vec<usize> =
                        ids.into_iter().filter(|&i| self.scope.r.tree.fns[i].init).collect();
                    if !consts.is_empty() {
                        self.add(&consts);
                        return;
                    }
                }
            }
            self.scope
                .env
                .entry(pi.ident.to_string())
                .or_insert_with(|| Ty::named("_"));
        }
        syn::visit::visit_pat(self, p);
    }
}

/// The edges of every fn.
///
/// Cost: time O(N·D) over the N AST nodes of all bodies and the resolution depth D ≤ 8 per path;
/// space O(E) for the E edges.
pub(crate) fn edges(tree: &Tree, idx: &Index) -> Vec<Edges> {
    let r = Resolver::new(tree, idx);
    tree.fns
        .iter()
        .map(|f| {
            let mut w = Walk {
                scope: Scope::new(&r, f),
                out: Edges::default(),
            };
            // The signature's types evaluate const expressions too (`fn f(x: [u8; N])`); a
            // `const`/`static` node carries its declared type as its return type.
            for arg in &f.sig.inputs {
                if let syn::FnArg::Typed(pt) = arg {
                    w.visit_type(&pt.ty);
                }
            }
            if let syn::ReturnType::Type(_, t) = &f.sig.output {
                w.visit_type(t);
            }
            w.visit_block(&f.body);
            w.out
        })
        .collect()
}

/// The fns reachable from `from` over `adj` (including `from`).
///
/// Cost: time O(V + E), space O(V).
pub(crate) fn reach(adj: &[Vec<usize>], from: &[usize]) -> Vec<bool> {
    let mut seen = vec![false; adj.len()];
    let mut stack: Vec<usize> = from.to_vec();
    while let Some(v) = stack.pop() {
        if seen[v] {
            continue;
        }
        seen[v] = true;
        stack.extend(adj[v].iter().copied().filter(|w| !seen[*w]));
    }
    seen
}
