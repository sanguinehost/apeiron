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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

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

    /// The methods named `name` that a call on a value of the concrete type `t` can run: `t`'s
    /// inherent impls, `t`'s own impls of the traits it implements, and those traits' defaults.
    ///
    /// Narrower than [`Resolver::type_methods`], which adds every impl of an implemented trait. A
    /// concrete type selects one impl, so the deferred dispatch is settled here; the receiver type
    /// of a method call, which may be an unsized or dynamically dispatched type, still uses
    /// `type_methods`.
    ///
    /// Cost: time O(I log I) over the I candidates; space O(I).
    pub fn own_methods(&self, t: &str, name: &str) -> Vec<usize> {
        let mut out: BTreeSet<usize> = BTreeSet::new();
        if let Some(ids) = self.idx.methods.get(&(t.to_string(), name.to_string())) {
            out.extend(ids);
        }
        for tr in self.tree.impls.get(t).into_iter().flatten() {
            let ids = self.idx.trait_methods.get(&(tr.clone(), name.to_string()));
            for &i in ids.into_iter().flatten() {
                // `t`'s impl, or the trait's default the impl inherits.
                match self.tree.fns[i].self_ty.as_deref() {
                    Some(s) if s == t || s == tr.as_str() => {
                        out.insert(i);
                    }
                    _ => {}
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

/// How a `recv.name(..)` call resolves.
pub(crate) enum Disp {
    /// The receiver's type selected the candidates.
    Typed(Vec<usize>),
    /// The receiver is typed by a type parameter `P`: the candidates are fixed by the caller's
    /// argument, not here.
    Param(String),
    /// No receiver type: every crate method of the name is a candidate.
    Loose(Vec<usize>),
}

/// The typing environment and resolver of one fn body.
pub(crate) struct Scope<'a> {
    pub r: &'a Resolver<'a>,
    pub f: &'a Func,
    pub env: BTreeMap<String, Ty>,
    /// Per type name, the `(declared field type, argument type)` pairs of the struct literals this
    /// body builds: the substitution that fixes a type parameter of the type's `impl`
    /// (`AttnHeadCtx { head: &h }` fixes `K`). Several literals of one type keep every pair, and a
    /// parameter the pairs disagree on is unknown, never narrowed.
    pub insts: BTreeMap<String, Vec<(Ty, Ty)>>,
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
        Scope {
            r,
            f,
            env,
            insts: BTreeMap::new(),
        }
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

    /// How `recv.name(..)` resolves. A known receiver type selects its own impl, not every impl of
    /// the trait it implements ([`Resolver::own_methods`]): `max.run(..)` on a `RowExpTask` runs
    /// `RowExpTask`'s `run`, not every `ChunkTask` impl. A type with no impl of its own falls back to
    /// the wider [`Resolver::type_methods`] — an inherited or blanket impl this index files under
    /// the trait — and only a type outside the crate falls to every method of the name.
    pub fn method(&self, recv: &syn::Expr, name: &str) -> Disp {
        if let Some(t) = self.type_of(recv) {
            let t = deref(t);
            if self.f.generics.contains(&t.name) {
                return Disp::Param(t.name);
            }
            let own = self.r.own_methods(&t.name, name);
            if !own.is_empty() {
                return Disp::Typed(own);
            }
            let ids = self.r.type_methods(&t.name, name);
            if !ids.is_empty() || !self.r.idx.types.contains_key(&t.name) {
                return Disp::Typed(ids);
            }
        }
        Disp::Loose(self.r.idx.by_method.get(name).cloned().unwrap_or_default())
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
            // `let Some(lin) = expr else` / `if let Ok(v) = ..`: the payload's type is the
            // scrutinee's own argument, so the binding is as known as the scrutinee is.
            syn::Pat::TupleStruct(ts) => {
                let t = init.and_then(|e| self.type_of(e)).map(deref);
                let variant = ts.path.segments.last().map(|s| s.ident.to_string());
                if let (Some(t), Some(variant)) = (t, variant) {
                    let payload = match (t.name.as_str(), variant.as_str()) {
                        ("Option", _) => t.args.first().cloned(),
                        ("Result", "Ok") => t.args.first().cloned(),
                        ("Result", "Err") => t.args.get(1).cloned(),
                        _ => None,
                    };
                    if let (Some(ty), Some(sub)) = (payload, ts.elems.first()) {
                        self.bind_as(sub, &ty);
                    }
                }
            }
            _ => {}
        }
    }

    /// Binds `pat` to the type `ty` where `pat` names a local; an ascription wins and a reference
    /// pattern is transparent. Cost: `O(1)` per binding, `O(d)` over a nested pattern of depth `d`.
    fn bind_as(&mut self, pat: &syn::Pat, ty: &Ty) {
        match pat {
            syn::Pat::Ident(pi) if pi.subpat.is_none() => {
                self.env.insert(pi.ident.to_string(), ty.clone());
            }
            syn::Pat::Type(pt) => {
                if let (syn::Pat::Ident(pi), Some(t)) = (&*pt.pat, ty_of(&pt.ty)) {
                    self.env.insert(pi.ident.to_string(), t);
                }
            }
            syn::Pat::Reference(r) => self.bind_as(&r.pat, ty),
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
                    _ => match self.method(&m.receiver, &name) {
                        Disp::Typed(ids) | Disp::Loose(ids) => self.common_ret(&ids),
                        // The candidates are the caller's argument's; nothing here fixes them.
                        Disp::Param(_) => None,
                    },
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
    /// Deferred dispatch on a type parameter, `P -> { name }`: `p.name(..)` and `P::name` inside
    /// the body, settled at the caller that fixes `P` (`resolve`). Empty when nothing is deferred.
    pub generic: BTreeMap<String, BTreeSet<String>>,
}

/// One call site: the callee candidates, and the type each argument has in the caller's scope.
struct Site {
    callees: Vec<usize>,
    args: Vec<Option<Ty>>,
}

/// One fn body, before the deferred dispatch is settled.
struct Body {
    edges: Edges,
    sites: Vec<Site>,
    /// Fns this body names where no argument and no receiver can fix their dispatch: a fn value, a
    /// macro path, a const pattern. A fn with a deferred map among them settles to every impl.
    other: BTreeSet<usize>,
    /// The method names this body calls on a receiver of unknown type, or whose instantiation it
    /// cannot fix: each is invoked with an unknown `Self`, so the name's own deferral closure
    /// widens *this* body, not the callee's edges, which a typed call site still reads.
    loose_calls: BTreeSet<String>,
    /// The struct literals' `(declared field type, argument type)` pairs, per type name
    /// ([`Scope::insts`]): what fixes a type parameter of a type's `impl` at its construction.
    insts: BTreeMap<String, Vec<(Ty, Ty)>>,
}

struct Walk<'a> {
    scope: Scope<'a>,
    out: Edges,
    sites: Vec<Site>,
    other: BTreeSet<usize>,
    loose_calls: BTreeSet<String>,
}

/// Adds a resolved call's candidates: `loose` always, `precise` when the fan-out is small.
fn add(edges: &mut Edges, ids: &[usize]) {
    edges.loose.extend(ids.iter().copied());
    if ids.len() <= PRECISE_FANOUT {
        edges.precise.extend(ids.iter().copied());
    }
}

impl Walk<'_> {
    fn add(&mut self, ids: &[usize]) {
        add(&mut self.out, ids);
    }

    /// A call: the candidates, plus the argument types this scope fixes.
    fn call(&mut self, ids: Vec<usize>, args: &Punctuated<syn::Expr, syn::Token![,]>) {
        let types = args.iter().map(|a| self.scope.type_of(a)).collect();
        self.add(&ids);
        self.sites.push(Site {
            callees: ids,
            args: types,
        });
    }

    /// `P::f` and `<P as Trait>::f` on a type parameter: the dispatch is deferred to the caller.
    fn generic_path(&self, p: &syn::ExprPath) -> Option<(String, String)> {
        let s = segs(&p.path);
        match (&p.qself, s.as_slice()) {
            (Some(q), [.., name]) => ty_of(&q.ty)
                .filter(|t| self.scope.f.generics.contains(&t.name))
                .map(|t| (t.name, name.clone())),
            (None, [ty, name]) if self.scope.f.generics.contains(ty) => {
                Some((ty.clone(), name.clone()))
            }
            _ => None,
        }
    }

    fn defer(&mut self, param: String, name: String) {
        self.out.generic.entry(param).or_default().insert(name);
    }
}

impl<'ast> Visit<'ast> for Walk<'_> {
    fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
        // `R::f` on a type parameter dispatches to the impl of `R` the caller passes.
        if let Some((param, name)) = self.generic_path(p) {
            self.defer(param, name);
            return;
        }
        match self.scope.path_res(&p.path) {
            Some(Res::Fns(ids)) => {
                self.add(&ids);
                self.other.extend(ids);
            }
            Some(Res::External(v)) => {
                self.out.external.insert(v.join("::"));
            }
            _ => {}
        }
        syn::visit::visit_expr_path(self, p);
    }

    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*c.func {
            if let Some((param, name)) = self.generic_path(p) {
                self.defer(param, name);
                for a in &c.args {
                    self.visit_expr(a);
                }
                return;
            }
            if let Some(Res::Fns(ids)) = self.scope.path_res(&p.path) {
                self.call(ids, &c.args);
                for a in &c.args {
                    self.visit_expr(a);
                }
                return;
            }
        }
        syn::visit::visit_expr_call(self, c);
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let name = m.method.to_string();
        match self.scope.method(&m.receiver, &name) {
            Disp::Typed(ids) => self.call(ids, &m.args),
            Disp::Param(param) => self.defer(param, name),
            Disp::Loose(ids) => {
                self.out.loose.extend(ids.iter().copied());
                self.loose_calls.insert(name);
            }
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

    /// A struct literal: its fields' declared types against the types the arguments have here, for
    /// the type parameters its `impl` may be generic in. Cost: `O(k)` per literal over its `k`
    /// fields; the store keeps at most one pair per field, so `O(k)` space per literal.
    fn visit_expr_struct(&mut self, s: &'ast syn::ExprStruct) {
        if let Some(seg) = s.path.segments.last() {
            let last = seg.ident.to_string();
            let name = match (last.as_str(), &self.scope.f.self_ty) {
                ("Self", Some(t)) => t.clone(),
                _ => last,
            };
            let declared = Ty::named(&name);
            let mut pairs: Vec<(Ty, Ty)> = Vec::new();
            for fv in &s.fields {
                let member = match &fv.member {
                    syn::Member::Named(i) => i.to_string(),
                    syn::Member::Unnamed(i) => i.index.to_string(),
                };
                if let (Some(d), Some(a)) = (
                    self.scope.r.field_ty(&declared, &member),
                    self.scope.type_of(&fv.expr),
                ) {
                    pairs.push((d, a));
                }
            }
            if !pairs.is_empty() {
                let e = self.scope.insts.entry(name).or_default();
                for p in pairs {
                    if !e.contains(&p) {
                        e.push(p);
                    }
                }
            }
        }
        syn::visit::visit_expr_struct(self, s);
    }

    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        // A body-local `macro_rules!` is expanded only inside this body: every path its rules
        // name is a candidate reference.
        if m.path.is_ident("macro_rules") {
            for path in token_paths(&m.tokens) {
                if let Some(Res::Fns(ids)) = self.scope.path_res(&path) {
                    self.out.loose.extend(ids.iter().copied());
                    self.other.extend(ids);
                }
            }
            return;
        }
        for e in macro_exprs(&m.tokens) {
            self.visit_expr(&e);
        }
        if let Some(Res::Fns(ids)) = self.scope.path_res(&m.path) {
            self.add(&ids);
            // The argument types are the expansion's, which this gate does not see.
            self.other.extend(ids);
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
                    let consts: Vec<usize> = ids
                        .into_iter()
                        .filter(|&i| self.scope.r.tree.fns[i].init)
                        .collect();
                    if !consts.is_empty() {
                        self.add(&consts);
                        self.other.extend(consts);
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

/// The edges of every fn, with every deferred type-parameter dispatch settled at the call site
/// that fixes the parameter.
///
/// Cost: time O(N·D + S·A + G·(V + E)) over the N AST nodes of all bodies, the resolution depth
/// D ≤ 8 per path, the S call sites, the A argument types per site and the G deferred
/// `(type parameter, method)` pairs; space O(E + S·A) for the E edges and the argument types kept
/// per site.
pub(crate) fn edges(tree: &Tree, idx: &Index) -> Vec<Edges> {
    let r = Resolver::new(tree, idx);
    let mut bodies: Vec<Body> = tree
        .fns
        .iter()
        .map(|f| {
            let mut w = Walk {
                scope: Scope::new(&r, f),
                out: Edges::default(),
                sites: Vec::new(),
                other: BTreeSet::new(),
                loose_calls: BTreeSet::new(),
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
            let insts = std::mem::take(&mut w.scope.insts);
            Body {
                edges: w.out,
                sites: w.sites,
                other: w.other,
                loose_calls: w.loose_calls,
                insts,
            }
        })
        .collect();
    resolve(&mut bodies, tree, idx);
    bodies.into_iter().map(|b| b.edges).collect()
}

/// The argument positions of `f` whose declared type names the type parameter `p` (`P`, `&P`,
/// `Box<P>`, `Option<&P>`, …), counted over the non-receiver inputs.
fn param_positions(f: &Func, p: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 0;
    for arg in &f.sig.inputs {
        if let syn::FnArg::Typed(pt) = arg {
            if ty_of(&pt.ty).is_some_and(|t| names_param(&t, p)) {
                out.push(i);
            }
            i += 1;
        }
    }
    out
}

fn names_param(t: &Ty, p: &str) -> bool {
    t.name == p || t.args.iter().any(|a| names_param(a, p))
}

/// Settles every deferred `(P, name)` at the call sites that fix `P`.
///
/// A deferred pair on fn `g` is read at each site `c` calling `g`, and settled there. `P` is a
/// parameter of `g`'s own signature when `g` declares it (`fn run_all<K: Job>(k: &K)`), and then the
/// argument filling that position fixes it; it is a parameter of `g`'s enclosing `impl` when only
/// the `impl` declares it (`impl<K: Job> Ctx<K> { fn go(&self) }`), and then the struct literal the
/// caller built the type with fixes it ([`Scope::insts`]).
///
/// A concrete type `T` adds `T`'s own methods named `name` ([`Resolver::own_methods`]) to `c`, as a
/// call site of its own, so a method resolved that way has its own deferred pairs settled in turn; a
/// type variable `Q` of `c` defers `(Q, name)` to `c`'s own callers; an unknown type, `_`, or a
/// parameter neither the argument nor the instantiation fixes, adds every crate method of the name
/// to `c` as a loose edge — the pre-resolution candidate set, so an unknown never narrows.
///
/// A fn whose deferred map nothing fixes — a fn value, a macro path, an untyped receiver's
/// candidate set, no call site at all — settles to every crate method of each name in its own
/// edges, and each fn added that way is settled in turn, so the graph never loses an impl.
///
/// Cost: time O(S·A + G·(V + E)), space O(V + S·A). Derivation: a fn re-enters the worklist only
/// when a deferred pair is inserted into its map, so at most G pushes; each push re-scans the fn's
/// own reference sites, whose total is E; the degradation queue admits a fn at most once.
fn resolve(bodies: &mut [Body], tree: &Tree, idx: &Index) {
    let n = bodies.len();
    // The sites that call a fn, as `(caller, site index)`.
    let mut callers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (c, b) in bodies.iter().enumerate() {
        for (s, site) in b.sites.iter().enumerate() {
            for &callee in &site.callees {
                callers[callee].push((c, s));
            }
        }
    }
    let queued: Vec<bool> = bodies.iter().map(|b| !b.edges.generic.is_empty()).collect();
    let queue: VecDeque<usize> = (0..n).filter(|&i| queued[i]).collect();
    let r = Resolver::new(tree, idx);
    let mut st = Settle {
        bodies,
        tree,
        idx,
        r: &r,
        callers,
        queued,
        queue,
        made: BTreeSet::new(),
    };
    st.settle_sites();
    st.degrade();
}

/// The worklist of [`resolve`]: the deferred pairs still to settle, the sites that name each fn, and
/// the degradation pass. Owns its state, so a settle step borrows one field at a time.
struct Settle<'a> {
    bodies: &'a mut [Body],
    tree: &'a Tree,
    idx: &'a Index,
    r: &'a Resolver<'a>,
    callers: Vec<Vec<(usize, usize)>>,
    queued: Vec<bool>,
    queue: VecDeque<usize>,
    /// The `(caller, type, name)` triples that already have a synthetic site.
    made: BTreeSet<(usize, String, String)>,
}

impl Settle<'_> {
    /// (1) Settle a deferred pair at every site that calls its fn. Cost: as [`resolve`].
    ///
    /// `P` is fixed one of two ways. A parameter of the callee's own signature
    /// (`fn run_all<K: Job>(k: &K)`) is fixed by the argument filling that position. A parameter of
    /// its enclosing `impl` (`impl<K: Job> Ctx<K> { fn go(&self) }`) is fixed by the struct literal
    /// the caller built the type with ([`Scope::insts`]), which is why the pool's
    /// `chunks_mut_global(out, hd, &AttnHeadCtx { head })` settles `AttnHeadCtx`'s `K`.
    fn settle_sites(&mut self) {
        let tree = self.tree;
        while let Some(callee) = self.queue.pop_front() {
            self.queued[callee] = false;
            let deferred = self.bodies[callee].edges.generic.clone();
            if deferred.is_empty() {
                continue;
            }
            let f = &tree.fns[callee];
            for (c, s) in self.callers[callee].clone() {
                for (param, names) in &deferred {
                    let positions = param_positions(f, param);
                    // `P` of the enclosing `impl`, not of the callee's own signature.
                    let self_ty = f
                        .self_ty
                        .as_deref()
                        .filter(|_| !f.sig.generics.type_params().any(|g| g.ident == *param));
                    for name in names {
                        let mut fixed: Vec<Ty> = Vec::new();
                        let mut unknown = positions.is_empty() && self_ty.is_none();
                        if !positions.is_empty() {
                            for &j in &positions {
                                match self.bodies[c].sites[s].args.get(j).cloned().flatten() {
                                    Some(t) if t.name != "_" => fixed.push(t),
                                    _ => unknown = true,
                                }
                            }
                        } else if let Some(st) = self_ty {
                            match instantiation(&self.bodies[c].insts, st, param) {
                                Inst::Bound(t) => fixed.push(t),
                                // No field of the type carries the parameter: this body does not
                                // build that struct, so it invokes no instantiation of it.
                                Inst::Absent => {}
                                Inst::Unknown => unknown = true,
                            }
                        }
                        if unknown {
                            loose_name(&mut self.bodies[c], self.idx, name);
                        }
                        for t in fixed {
                            self.one(c, name, &t);
                        }
                    }
                }
            }
        }
    }

    /// Whether the impl fn `i`'s structs are *not* what caller `c` builds under that simple name: a
    /// type parameter of the impl that no field of `c`'s literals of the name carries
    /// ([`Inst::Absent`]). Cost: O(d) per deferred pair over its type depth, O(1) when none.
    fn other_struct(&self, c: usize, i: usize) -> bool {
        let f = &self.tree.fns[i];
        let Some(st) = f.self_ty.as_deref() else {
            return false;
        };
        self.bodies[i]
            .edges
            .generic
            .keys()
            .any(|p| matches!(instantiation(&self.bodies[c].insts, st, p), Inst::Absent))
    }

    /// One resolved type `t` for method `name` at caller `c`. A type variable of `c` defers the
    /// pair to `c`'s own callers; a concrete type contributes its own impls, plus one synthetic site
    /// so those impls' own deferred pairs settle in turn.
    ///
    /// Cost: O(log V) lookup in each map plus O(I log I) over the `I` impls the type has.
    fn one(&mut self, c: usize, name: &str, t: &Ty) {
        let tree = self.tree;
        if tree.fns[c].generics.contains(&t.name) {
            let fresh = self.bodies[c]
                .edges
                .generic
                .entry(t.name.clone())
                .or_default()
                .insert(name.to_string());
            if fresh && !self.queued[c] {
                self.queued[c] = true;
                self.queue.push_back(c);
            }
            return;
        }
        // The determined impl, as a reachability edge. Not `precise`: a precise edge moves the
        // forward rule's minimal-forward set, which is a contract re-baseline
        // (`docs/contracts/one-path.toml`).
        // One simple name can name two structs in two modules (`AttnHeadCtx` in
        // `step::softmax` and `chart::ffn`). The body's literals of that name decide which impls
        // this call can run: an impl whose type parameter no literal field carries is another
        // struct, and its deferred dispatch would otherwise widen this site to every impl.
        let ids: Vec<usize> = self
            .r
            .own_methods(&t.name, name)
            .into_iter()
            .filter(|&i| !self.other_struct(c, i))
            .collect();
        if ids.is_empty() {
            return;
        }
        self.bodies[c].edges.loose.extend(ids.iter().copied());
        if !self.made.insert((c, t.name.clone(), name.to_string())) {
            return;
        }
        let site = self.bodies[c].sites.len();
        self.bodies[c].sites.push(Site {
            callees: ids.clone(),
            args: Vec::new(),
        });
        for callee in ids {
            self.callers[callee].push((c, site));
            if !self.queued[callee] && !self.bodies[callee].edges.generic.is_empty() {
                self.queued[callee] = true;
                self.queue.push_back(callee);
            }
        }
    }

    /// (2) A deferred map nothing fixes settles to every impl of each name. Cost: O(V + E).
    fn degrade(&mut self) {
        let n = self.bodies.len();
        // (2a) An invocation the gate cannot type widens *the caller*, not the callee: the caller's
        // own edges already hold every impl of the invoked name, and the names those impls defer in
        // turn are added here, so a typed call site of the same callee still reads its own edges.
        let defer: Vec<BTreeSet<String>> = self
            .bodies
            .iter()
            .map(|b| b.edges.generic.values().flatten().cloned().collect())
            .collect();
        let mut memo: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for c in 0..n {
            let calls: Vec<String> = self.bodies[c].loose_calls.iter().cloned().collect();
            for name in calls {
                let closure = name_closure(self.idx, &defer, &mut memo, &name);
                if let Some(ids) = self.idx.by_method.get(&name) {
                    let ids = ids.clone();
                    self.bodies[c].edges.loose.extend(ids);
                }
                for m in closure {
                    if m == name {
                        continue;
                    }
                    if let Some(ids) = self.idx.by_method.get(&m) {
                        let ids = ids.clone();
                        self.bodies[c].edges.loose.extend(ids);
                    }
                }
            }
        }
        // (2b) A fn named where no call site can fix it at all — a fn value, a macro path, a const
        // pattern — settles to every impl of each name it defers, and each fn added that way is
        // settled in turn, so the graph never loses an impl.
        let mut loose_refs: BTreeSet<usize> = BTreeSet::new();
        for b in self.bodies.iter() {
            loose_refs.extend(b.other.iter().copied());
        }
        let mut degraded = vec![false; n];
        let mut work: VecDeque<usize> = (0..n)
            .filter(|&i| {
                !self.bodies[i].edges.generic.is_empty()
                    && (loose_refs.contains(&i) || self.callers[i].is_empty())
            })
            .collect();
        while let Some(i) = work.pop_front() {
            if degraded[i] {
                continue;
            }
            degraded[i] = true;
            let names: Vec<String> = self.bodies[i]
                .edges
                .generic
                .values()
                .flatten()
                .cloned()
                .collect();
            for name in names {
                if let Some(ids) = self.idx.by_method.get(&name) {
                    let ids = ids.clone();
                    self.bodies[i].edges.loose.extend(ids.iter().copied());
                    for id in ids {
                        if loose_refs.insert(id)
                            && !degraded[id]
                            && !self.bodies[id].edges.generic.is_empty()
                        {
                            work.push_back(id);
                        }
                    }
                }
            }
        }
    }
}

/// Every method name an untyped invocation of `name` can defer in turn: `name`'s impls, the names
/// they defer, and so on, memoised per name.
///
/// Cost: O(N log N) per name computed, over the N `(fn, name)` pairs the closure reaches; the memo
/// makes a repeated name one lookup, so the whole pass is O(M·N log N) for the M loose names.
fn name_closure(
    idx: &Index,
    defer: &[BTreeSet<String>],
    memo: &mut BTreeMap<String, BTreeSet<String>>,
    name: &str,
) -> BTreeSet<String> {
    if let Some(hit) = memo.get(name) {
        return hit.clone();
    }
    let mut out: BTreeSet<String> = BTreeSet::new();
    let mut stack = vec![name.to_string()];
    while let Some(n) = stack.pop() {
        if !out.insert(n.clone()) {
            continue;
        }
        for &id in idx.by_method.get(&n).into_iter().flatten() {
            for m in &defer[id] {
                if !out.contains(m) {
                    stack.push(m.clone());
                }
            }
        }
    }
    memo.insert(name.to_string(), out.clone());
    out
}

/// What the struct literals of one body say about a type parameter of a type.
enum Inst {
    /// The argument a field of that parameter's type was built with.
    Bound(Ty),
    /// The body builds the type, and no field of it carries the parameter.
    Absent,
    /// Nothing in the body fixes it, or two of its literals disagree.
    Unknown,
}

/// The type the struct literal pairs give the type parameter `p` of the type `t`.
///
/// Cost: time O(k·d) over the k pairs and their type depth d; space O(d) for the bound type.
fn instantiation(insts: &BTreeMap<String, Vec<(Ty, Ty)>>, t: &str, p: &str) -> Inst {
    let Some(pairs) = insts.get(t) else {
        return Inst::Unknown;
    };
    let mut out: Option<Ty> = None;
    for (declared, actual) in pairs {
        if let Some(b) = bind_param(declared, actual, p) {
            match &out {
                None => out = Some(b),
                Some(o) if *o == b => {}
                Some(_) => return Inst::Unknown,
            }
        }
    }
    match out {
        Some(t) => Inst::Bound(t),
        // Every field the body's literals of `t` name is known, and none of them carries `p`: this
        // body does not build that type parameter's struct. A Rust literal names every field (a
        // `..base` rest needs a value of the same struct, which is a literal or a field of one), so
        // `p` bound through a field cannot be absent here. Two types of one simple name in
        // different modules are the case this separates: `step::softmax::AttnHeadCtx` against
        // `chart::ffn::AttnHeadCtx<E, K>`.
        None => Inst::Absent,
    }
}

/// The type `actual` gives the parameter `p` in `declared`, matching the two type shapes.
///
/// Cost: time O(d) over the type depth d of `declared`; space O(d).
fn bind_param(declared: &Ty, actual: &Ty, p: &str) -> Option<Ty> {
    if declared.name == p {
        return Some(actual.clone());
    }
    if declared.args.len() != actual.args.len() {
        return None;
    }
    let mut out: Option<Ty> = None;
    for (d, a) in declared.args.iter().zip(actual.args.iter()) {
        if let Some(b) = bind_param(d, a, p) {
            match &out {
                None => out = Some(b),
                Some(o) if *o == b => {}
                Some(_) => return None,
            }
        }
    }
    out
}

/// Adds every crate method named `name` to `b` as a loose edge, and records the reference as one
/// that fixes no dispatch.
fn loose_name(b: &mut Body, idx: &Index, name: &str) {
    if let Some(ids) = idx.by_method.get(name) {
        b.edges.loose.extend(ids.iter().copied());
    }
    b.loose_calls.insert(name.to_string());
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
