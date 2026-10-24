//! The rules. Each reads structure — the AST, the resolved call graph — never a fn's name.
//!
//! | rule | finding |
//! |---|---|
//! | `product` | a multiply-accumulate loop in non-test `src/` outside `src/engine/kern/` (§0 K1) |
//! | `kernel-adapter` | a `src/learn/` fn calling into `src/engine/kern/` other than `limb.rs` (§0 K2) |
//! | `forward` | a minimal forward body that is not one of the declared engine bodies |
//! | `descent` | a sqrt-normalised parameter step (Adam, RMSProp) in non-test code |
//! | `backward` | a backward-tree fn entered from non-test code outside the tree |
//! | `switch` | an environment read, a cfg-alternate body or a `cfg!` branch in non-test `src/` |
//! | `orphan` | a non-test `src/` fn no root reaches |
//! | `dead-file` | a `src/` file no `mod` declaration reaches |
//! | `clone`, `clone-test` | two bodies with the same normalised tokens (`clones.rs`) |
//!
//! A *forward* is a fn whose precise reach computes a product (P), an exponential (A: softmax,
//! attention scores) and a square root (N: a norm); it is *minimal* when no fn it calls directly is
//! itself a forward. A model forward split across helpers is minimal at the fn that assembles it, so
//! splitting the body moves the finding, never removes it.

use std::collections::{BTreeMap, BTreeSet};

use quote::ToTokens;
use syn::spanned::Spanned;
use syn::visit::Visit;

use super::graph::{macro_exprs, reach, segs, Edges, Res, Resolver, Scope};
use super::load::{CrateKind, Func, Tree};

pub(crate) struct Finding {
    pub rule: &'static str,
    /// The identity a debt row names: one key, or the sorted member keys of a clone group.
    pub items: Vec<String>,
    pub at: String,
    /// The transitive consumer count (`forward`, `backward`) or the group size (`clone*`).
    pub count: Option<usize>,
    pub note: String,
}

const KERN: &str = "src/engine/kern/";
const LIMB: &str = "src/engine/kern/limb.rs";

/// Product (P) seeds beyond a detected multiply-accumulate: the kernel tree and the GPU linear
/// dispatch, whose products run in SPIR-V and so have no CPU loop to detect.
const P_FILES: [&str; 3] = [KERN, "src/gpu/nn/fwd/lin.rs", "src/gpu/nn/coopmat_gemm.rs"];
/// Exponential (A) seeds beyond a detected `exp`: the fixed-point exponential and the attention
/// trees, CPU and GPU.
const A_FILES: [&str; 6] = [
    "src/numeric/fixed_exp.rs",
    "src/engine/attn.rs",
    "src/engine/attn/",
    "src/engine/step/attn_block.rs",
    "src/engine/step/softmax.rs",
    "src/gpu/nn/attn/",
];
/// Square-root (N) seeds beyond a detected `sqrt`: the fixed-point root and the norm bodies.
const N_FILES: [&str; 4] = [
    "src/numeric/fixed_rsqrt.rs",
    "src/engine/ops/norm.rs",
    "src/gpu/nn/norm.rs",
    "src/gpu/nn/fwd/norm.rs",
];
/// The backward trees: a fn here computes a gradient.
const BACKWARD: [&str; 4] = [
    "src/learn/grad/",
    "src/gpu/nn/bwd/",
    "src/gpu/nn/attn/bwd.rs",
    "src/gpu/nn/norm_bwd.rs",
];

fn under(file: &str, set: &[&str]) -> bool {
    set.iter().any(|p| {
        if p.ends_with('/') {
            file.starts_with(p)
        } else {
            file == *p
        }
    })
}

fn is_src(f: &Func) -> bool {
    f.file.starts_with("src/") && !f.test
}

fn line_of<T: Spanned>(node: &T) -> usize {
    node.span().start().line
}

// ---- multiply-accumulate detection ----------------------------------------------------------------

/// The data idents of a body: loop and closure bindings over non-range iterators, and `let`
/// bindings whose initialiser reads data (an index expression or another data ident).
struct Taint {
    data: BTreeSet<String>,
    lets: Vec<(String, syn::Expr)>,
}

fn pat_idents(p: &syn::Pat, out: &mut Vec<String>) {
    match p {
        syn::Pat::Ident(pi) => out.push(pi.ident.to_string()),
        syn::Pat::Tuple(t) => t.elems.iter().for_each(|e| pat_idents(e, out)),
        syn::Pat::TupleStruct(t) => t.elems.iter().for_each(|e| pat_idents(e, out)),
        syn::Pat::Reference(r) => pat_idents(&r.pat, out),
        syn::Pat::Type(t) => pat_idents(&t.pat, out),
        syn::Pat::Paren(p) => pat_idents(&p.pat, out),
        syn::Pat::Slice(s) => s.elems.iter().for_each(|e| pat_idents(e, out)),
        syn::Pat::Struct(s) => s.fields.iter().for_each(|f| pat_idents(&f.pat, out)),
        _ => {}
    }
}

/// The root of a method-call chain: `(0..n).map(..).sum()` roots at the range.
fn chain_root(e: &syn::Expr) -> &syn::Expr {
    match e {
        syn::Expr::MethodCall(m) => chain_root(&m.receiver),
        syn::Expr::Paren(p) => chain_root(&p.expr),
        syn::Expr::Reference(r) => chain_root(&r.expr),
        other => other,
    }
}

fn is_range_chain(e: &syn::Expr) -> bool {
    matches!(chain_root(e), syn::Expr::Range(_))
}

fn ends_enumerate(e: &syn::Expr) -> bool {
    match e {
        syn::Expr::MethodCall(m) => {
            m.method == "enumerate" || (m.method != "zip" && ends_enumerate(&m.receiver))
        }
        _ => false,
    }
}

/// Binds the pattern of a loop or closure over iterator `iter`: a range binds an index, the first
/// element of an `enumerate` pair is an index, everything else is data.
fn bind_iter(t: &mut Taint, pat: &syn::Pat, iter: Option<&syn::Expr>) {
    if iter.map(is_range_chain).unwrap_or(false) {
        return;
    }
    let skip_first = iter.map(ends_enumerate).unwrap_or(false);
    match pat {
        syn::Pat::Tuple(tp) if skip_first => {
            for e in tp.elems.iter().skip(1) {
                let mut v = Vec::new();
                pat_idents(e, &mut v);
                t.data.extend(v);
            }
        }
        _ => {
            let mut v = Vec::new();
            pat_idents(pat, &mut v);
            t.data.extend(v);
        }
    }
}

impl<'ast> Visit<'ast> for Taint {
    fn visit_expr_for_loop(&mut self, f: &'ast syn::ExprForLoop) {
        bind_iter(self, &f.pat, Some(&f.expr));
        syn::visit::visit_expr_for_loop(self, f);
    }
    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        for a in &m.args {
            if let syn::Expr::Closure(c) = a {
                for p in &c.inputs {
                    bind_iter(self, p, Some(&m.receiver));
                }
            }
        }
        syn::visit::visit_expr_method_call(self, m);
    }
    fn visit_local(&mut self, l: &'ast syn::Local) {
        if let Some(init) = &l.init {
            let mut v = Vec::new();
            pat_idents(&l.pat, &mut v);
            for name in v {
                self.lets.push((name, (*init.expr).clone()));
            }
        }
        syn::visit::visit_local(self, l);
    }
    fn visit_expr_match(&mut self, m: &'ast syn::ExprMatch) {
        // `match e { Ok(v) => v, .. }` binds the arm idents from `e`.
        for arm in &m.arms {
            let mut v = Vec::new();
            pat_idents(&arm.pat, &mut v);
            for name in v {
                self.lets.push((name, (*m.expr).clone()));
            }
        }
        syn::visit::visit_expr_match(self, m);
    }
    fn visit_expr_let(&mut self, l: &'ast syn::ExprLet) {
        let mut v = Vec::new();
        pat_idents(&l.pat, &mut v);
        for name in v {
            self.lets.push((name, (*l.expr).clone()));
        }
        syn::visit::visit_expr_let(self, l);
    }
}

/// Whether `e` reads data: an index expression or a data ident, not a length.
fn reads_data(e: &syn::Expr, data: &BTreeSet<String>) -> bool {
    struct R<'a> {
        data: &'a BTreeSet<String>,
        hit: bool,
    }
    impl<'ast> Visit<'ast> for R<'_> {
        fn visit_expr_index(&mut self, _: &'ast syn::ExprIndex) {
            self.hit = true;
        }
        fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
            if p.path.segments.len() == 1
                && self.data.contains(&p.path.segments[0].ident.to_string())
            {
                self.hit = true;
            }
        }
        fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
            if matches!(
                m.method.to_string().as_str(),
                "len" | "count" | "capacity" | "is_empty"
            ) {
                return;
            }
            syn::visit::visit_expr_method_call(self, m);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut r = R { data, hit: false };
    r.visit_expr(e);
    r.hit
}

fn taint(f: &Func, params_are_data: bool) -> BTreeSet<String> {
    let mut t = Taint {
        data: BTreeSet::new(),
        lets: Vec::new(),
    };
    if params_are_data {
        for input in &f.sig.inputs {
            if let syn::FnArg::Typed(pt) = input {
                let mut v = Vec::new();
                pat_idents(&pt.pat, &mut v);
                t.data.extend(v);
            }
        }
    }
    t.visit_block(&f.body);
    loop {
        let mut changed = false;
        for (name, init) in &t.lets {
            if !t.data.contains(name) && reads_data(init, &t.data) {
                t.data.insert(name.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    t.data
}

const MUL_METHODS: [&str; 7] = [
    "checked_mul",
    "wrapping_mul",
    "saturating_mul",
    "overflowing_mul",
    "widening_mul",
    "mul",
    "fmul",
];
const ADD_METHODS: [&str; 6] = [
    "checked_add",
    "wrapping_add",
    "saturating_add",
    "checked_sub",
    "wrapping_sub",
    "saturating_sub",
];

fn toks(e: &syn::Expr) -> String {
    e.to_token_stream().to_string()
}

/// Finds a product of two distinct data operands in `e`, skipping index positions. A square of one
/// operand (`x * x`) is a norm term, which the kernel does not carry.
struct Product<'a, 'b> {
    data: &'a BTreeSet<String>,
    scope: Option<&'a Scope<'b>>,
    mults: &'a BTreeSet<usize>,
    hit: Option<usize>,
}

impl Product<'_, '_> {
    fn pair(&mut self, a: &syn::Expr, b: &syn::Expr, at: usize) {
        if self.hit.is_none()
            && reads_data(a, self.data)
            && reads_data(b, self.data)
            && toks(a) != toks(b)
        {
            self.hit = Some(at);
        }
    }
}

impl<'ast> Visit<'ast> for Product<'_, '_> {
    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        self.visit_expr(&i.expr);
    }
    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        if matches!(b.op, syn::BinOp::Mul(_) | syn::BinOp::MulAssign(_)) {
            self.pair(&b.left, &b.right, line_of(b));
        }
        syn::visit::visit_expr_binary(self, b);
    }
    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let name = m.method.to_string();
        if MUL_METHODS.contains(&name.as_str()) || name == "mul_add" {
            if let Some(a) = m.args.first() {
                self.pair(&m.receiver, a, line_of(m));
            }
        }
        syn::visit::visit_expr_method_call(self, m);
    }
    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let (Some(scope), syn::Expr::Path(p), true) = (self.scope, &*c.func, c.args.len() >= 2) {
            if let Some(Res::Fns(ids)) = scope.path_res(&p.path) {
                if ids.iter().any(|i| self.mults.contains(i)) {
                    self.pair(&c.args[0], &c.args[1], line_of(c));
                }
            }
        }
        syn::visit::visit_expr_call(self, c);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        for e in macro_exprs(&m.tokens) {
            self.visit_expr(&e);
        }
    }
}

fn product_in(
    e: &syn::Expr,
    data: &BTreeSet<String>,
    scope: Option<&Scope>,
    mults: &BTreeSet<usize>,
) -> Option<usize> {
    let mut p = Product {
        data,
        scope,
        mults,
        hit: None,
    };
    p.visit_expr(e);
    p.hit
}

/// The accumulation sites of a body: `acc += p`, `acc = acc + p`, `acc = acc.checked_add(p)` inside
/// a loop or closure, and `.sum()`/`.fold()` over a closure, where `p` holds a data product.
struct Mac<'a, 'b> {
    data: &'a BTreeSet<String>,
    scope: &'a Scope<'b>,
    mults: &'a BTreeSet<usize>,
    depth: usize,
    hit: Option<usize>,
}

impl Mac<'_, '_> {
    fn product(&self, e: &syn::Expr) -> Option<usize> {
        product_in(e, self.data, Some(self.scope), self.mults)
    }
    fn found(&mut self, at: Option<usize>) {
        if self.hit.is_none() {
            self.hit = at;
        }
    }
}

fn closure_bodies(e: &syn::Expr, out: &mut Vec<syn::Expr>) {
    match e {
        syn::Expr::MethodCall(m) => {
            for a in &m.args {
                if let syn::Expr::Closure(c) = a {
                    out.push((*c.body).clone());
                }
            }
            closure_bodies(&m.receiver, out);
        }
        syn::Expr::Paren(p) => closure_bodies(&p.expr, out),
        _ => {}
    }
}

impl<'ast> Visit<'ast> for Mac<'_, '_> {
    fn visit_expr_for_loop(&mut self, f: &'ast syn::ExprForLoop) {
        self.visit_expr(&f.expr);
        self.depth += 1;
        self.visit_block(&f.body);
        self.depth -= 1;
    }
    fn visit_expr_while(&mut self, w: &'ast syn::ExprWhile) {
        self.depth += 1;
        syn::visit::visit_expr_while(self, w);
        self.depth -= 1;
    }
    fn visit_expr_loop(&mut self, l: &'ast syn::ExprLoop) {
        self.depth += 1;
        syn::visit::visit_expr_loop(self, l);
        self.depth -= 1;
    }
    fn visit_expr_closure(&mut self, c: &'ast syn::ExprClosure) {
        self.depth += 1;
        syn::visit::visit_expr_closure(self, c);
        self.depth -= 1;
    }
    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        if self.depth > 0 && matches!(b.op, syn::BinOp::AddAssign(_) | syn::BinOp::SubAssign(_)) {
            let at = self.product(&b.right);
            self.found(at);
        }
        syn::visit::visit_expr_binary(self, b);
    }
    fn visit_expr_assign(&mut self, a: &'ast syn::ExprAssign) {
        if self.depth > 0 {
            let lhs = toks(&a.left);
            let mut rhs: &syn::Expr = &a.right;
            while let syn::Expr::Paren(p) = rhs {
                rhs = &p.expr;
            }
            match rhs {
                syn::Expr::Binary(b) if matches!(b.op, syn::BinOp::Add(_) | syn::BinOp::Sub(_)) => {
                    if toks(&b.left) == lhs {
                        let at = self.product(&b.right);
                        self.found(at);
                    } else if toks(&b.right) == lhs {
                        let at = self.product(&b.left);
                        self.found(at);
                    }
                }
                syn::Expr::MethodCall(m) => {
                    let name = m.method.to_string();
                    if ADD_METHODS.contains(&name.as_str()) && toks(&m.receiver) == lhs {
                        if let Some(arg) = m.args.first() {
                            let at = self.product(arg);
                            self.found(at);
                        }
                    }
                    if name == "mul_add" && m.args.len() == 2 && toks(&m.args[1]) == lhs {
                        let at = self.product(&syn::Expr::MethodCall(syn::ExprMethodCall {
                            args: m.args.iter().take(1).cloned().collect(),
                            ..m.clone()
                        }));
                        self.found(at);
                    }
                }
                _ => {}
            }
        }
        syn::visit::visit_expr_assign(self, a);
    }
    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let name = m.method.to_string();
        if name == "sum" {
            let mut bodies = Vec::new();
            closure_bodies(&m.receiver, &mut bodies);
            for b in &bodies {
                let at = self.product(b);
                self.found(at);
            }
        } else if name == "fold" && m.args.len() == 2 {
            if let syn::Expr::Closure(c) = &m.args[1] {
                let mut adds = AddTerms(Vec::new());
                adds.visit_expr(&c.body);
                for t in &adds.0 {
                    let at = self.product(t);
                    self.found(at);
                }
            }
        }
        syn::visit::visit_expr_method_call(self, m);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        for e in macro_exprs(&m.tokens) {
            self.visit_expr(&e);
        }
    }
}

/// The operands of every `+`/`-` and `checked_add`-family call.
struct AddTerms(Vec<syn::Expr>);
impl<'ast> Visit<'ast> for AddTerms {
    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        if matches!(b.op, syn::BinOp::Add(_) | syn::BinOp::Sub(_)) {
            self.0.push((*b.left).clone());
            self.0.push((*b.right).clone());
        }
        syn::visit::visit_expr_binary(self, b);
    }
    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        if ADD_METHODS.contains(&m.method.to_string().as_str()) {
            self.0.extend(m.args.iter().cloned());
            self.0.push((*m.receiver).clone());
        }
        syn::visit::visit_expr_method_call(self, m);
    }
}

/// Fns with two or more parameters whose loop-free body multiplies two of them: scalar multipliers
/// (`mul_q32(a, b)`), so that `acc += mul_q32(x[i], w[i])` is a product like `x[i] * w[i]`.
fn multipliers(tree: &Tree) -> BTreeSet<usize> {
    struct Loops(bool);
    impl<'ast> Visit<'ast> for Loops {
        fn visit_expr_for_loop(&mut self, _: &'ast syn::ExprForLoop) {
            self.0 = true;
        }
        fn visit_expr_while(&mut self, _: &'ast syn::ExprWhile) {
            self.0 = true;
        }
        fn visit_expr_loop(&mut self, _: &'ast syn::ExprLoop) {
            self.0 = true;
        }
    }
    let empty = BTreeSet::new();
    let mut out = BTreeSet::new();
    for (i, f) in tree.fns.iter().enumerate() {
        let params = f
            .sig
            .inputs
            .iter()
            .filter(|a| matches!(a, syn::FnArg::Typed(_)))
            .count()
            + usize::from(f.sig.receiver().is_some());
        if params < 2 {
            continue;
        }
        let mut l = Loops(false);
        l.visit_block(&f.body);
        if l.0 {
            continue;
        }
        let mut data = taint(f, true);
        if f.sig.receiver().is_some() {
            data.insert("self".to_string());
        }
        let block = syn::Expr::Block(syn::ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: f.body.clone(),
        });
        if product_in(&block, &data, None, &empty).is_some() {
            out.insert(i);
        }
    }
    out
}

// ---- the descent step ----------------------------------------------------------------------------

/// A sqrt-normalised step: a subtraction whose subtrahend divides by a square root of a running
/// statistic, through `let` and `match` bindings (`p - lr·m/(√v+ε)`; the SPIR-V builder's
/// `fsub(p, fdiv(num, fadd(ext1(SQRT, v), ε)))`). The statistic is a name written back to state
/// (`self.v[i] = v`, `store_f(v, gid, v)`); a root of a fresh scalar (a Jacobi angle) is not one.
struct Descent {
    lets: BTreeMap<String, Vec<syn::Expr>>,
    state: BTreeSet<String>,
    hit: Option<usize>,
}

fn last_seg(e: &syn::Expr) -> Option<String> {
    match e {
        syn::Expr::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

fn is_sub_name(n: &str) -> bool {
    matches!(
        n,
        "checked_sub" | "wrapping_sub" | "saturating_sub" | "fsub" | "sub" | "sub_q32"
    )
}

fn is_div_name(n: &str) -> bool {
    n == "div" || n == "fdiv" || n.starts_with("checked_div") || n.starts_with("div_")
}

impl Descent {
    /// Searches `e` (expanding bound idents) for a node `pred` accepts.
    fn search(&self, e: &syn::Expr, pred: &dyn Fn(&Descent, &syn::Expr) -> bool) -> bool {
        struct S<'a> {
            d: &'a Descent,
            pred: &'a dyn Fn(&Descent, &syn::Expr) -> bool,
            seen: BTreeSet<String>,
            hit: bool,
        }
        impl<'ast> Visit<'ast> for S<'_> {
            fn visit_expr(&mut self, e: &'ast syn::Expr) {
                if self.hit {
                    return;
                }
                if (self.pred)(self.d, e) {
                    self.hit = true;
                    return;
                }
                if let syn::Expr::Path(p) = e {
                    if p.path.segments.len() == 1 {
                        let name = p.path.segments[0].ident.to_string();
                        if self.seen.insert(name.clone()) {
                            if let Some(inits) = self.d.lets.get(&name) {
                                for i in inits.clone() {
                                    self.visit_expr(&i);
                                }
                            }
                        }
                    }
                }
                syn::visit::visit_expr(self, e);
            }
            fn visit_macro(&mut self, m: &'ast syn::Macro) {
                for e in macro_exprs(&m.tokens) {
                    self.visit_expr(&e);
                }
            }
        }
        let mut s = S {
            d: self,
            pred,
            seen: BTreeSet::new(),
            hit: false,
        };
        s.visit_expr(e);
        s.hit
    }

    /// A square root whose operand reaches a state name.
    fn is_sqrt(d: &Descent, e: &syn::Expr) -> bool {
        let has = |s: &str| s.to_ascii_lowercase().contains("sqrt");
        let names_sqrt = |a: &syn::Expr| match a {
            syn::Expr::Path(p) => p.path.segments.iter().any(|s| has(&s.ident.to_string())),
            _ => false,
        };
        let operands: Vec<&syn::Expr> = match e {
            syn::Expr::MethodCall(m) if has(&m.method.to_string()) => vec![&*m.receiver],
            syn::Expr::MethodCall(m) if m.args.iter().any(names_sqrt) => {
                m.args.iter().filter(|a| !names_sqrt(a)).collect()
            }
            syn::Expr::Call(c) if names_sqrt(&c.func) => c.args.iter().collect(),
            _ => Vec::new(),
        };
        operands.iter().any(|o| d.search(o, &Descent::is_state))
    }

    fn is_state(d: &Descent, e: &syn::Expr) -> bool {
        matches!(e, syn::Expr::Path(p) if p.path.segments.len() == 1
            && d.state.contains(&p.path.segments[0].ident.to_string()))
    }

    fn is_sqrt_div(d: &Descent, e: &syn::Expr) -> bool {
        let den: Option<&syn::Expr> = match e {
            syn::Expr::Binary(b)
                if matches!(b.op, syn::BinOp::Div(_) | syn::BinOp::DivAssign(_)) =>
            {
                Some(&b.right)
            }
            syn::Expr::MethodCall(m) if is_div_name(&m.method.to_string()) => m.args.last(),
            syn::Expr::Call(c) if last_seg(&c.func).map(|n| is_div_name(&n)).unwrap_or(false) => {
                c.args.last()
            }
            _ => None,
        };
        den.map(|x| d.search(x, &Descent::is_sqrt)).unwrap_or(false)
    }

    fn subtrahend(e: &syn::Expr) -> Option<&syn::Expr> {
        match e {
            syn::Expr::Binary(b)
                if matches!(b.op, syn::BinOp::Sub(_) | syn::BinOp::SubAssign(_)) =>
            {
                Some(&b.right)
            }
            syn::Expr::MethodCall(m) if is_sub_name(&m.method.to_string()) => m.args.last(),
            syn::Expr::Call(c) if last_seg(&c.func).map(|n| is_sub_name(&n)).unwrap_or(false) => {
                c.args.last()
            }
            _ => None,
        }
    }
}

impl<'ast> Visit<'ast> for Descent {
    fn visit_expr(&mut self, e: &'ast syn::Expr) {
        if self.hit.is_none() {
            if let Some(s) = Descent::subtrahend(e) {
                if self.search(s, &Descent::is_sqrt_div) {
                    self.hit = Some(line_of(e));
                }
            }
        }
        syn::visit::visit_expr(self, e);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        for e in macro_exprs(&m.tokens) {
            self.visit_expr(&e);
        }
    }
}

fn descent(f: &Func) -> Option<usize> {
    struct Lets(BTreeMap<String, Vec<syn::Expr>>);
    impl<'ast> Visit<'ast> for Lets {
        fn visit_local(&mut self, l: &'ast syn::Local) {
            if let Some(init) = &l.init {
                let mut v = Vec::new();
                pat_idents(&l.pat, &mut v);
                for n in v {
                    self.0.entry(n).or_default().push((*init.expr).clone());
                }
            }
            syn::visit::visit_local(self, l);
        }
        fn visit_expr_match(&mut self, m: &'ast syn::ExprMatch) {
            for arm in &m.arms {
                let mut v = Vec::new();
                pat_idents(&arm.pat, &mut v);
                for n in v {
                    self.0.entry(n).or_default().push((*m.expr).clone());
                }
            }
            syn::visit::visit_expr_match(self, m);
        }
    }
    /// Names written whole into an indexed or field place, or as the value of a `store*` call.
    struct State(BTreeSet<String>);
    impl State {
        fn name(&mut self, e: &syn::Expr) {
            if let syn::Expr::Path(p) = e {
                if p.path.segments.len() == 1 {
                    self.0.insert(p.path.segments[0].ident.to_string());
                }
            }
        }
    }
    impl<'ast> Visit<'ast> for State {
        fn visit_expr_assign(&mut self, a: &'ast syn::ExprAssign) {
            if matches!(&*a.left, syn::Expr::Index(_) | syn::Expr::Field(_)) {
                self.name(&a.right);
            }
            syn::visit::visit_expr_assign(self, a);
        }
        fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
            if m.method.to_string().starts_with("store") {
                if let Some(v) = m.args.last() {
                    self.name(v);
                }
            }
            syn::visit::visit_expr_method_call(self, m);
        }
    }
    let mut lets = Lets(BTreeMap::new());
    lets.visit_block(&f.body);
    let mut state = State(BTreeSet::new());
    state.visit_block(&f.body);
    let mut d = Descent {
        lets: lets.0,
        state: state.0,
        hit: None,
    };
    d.visit_block(&f.body);
    d.hit
}

// ---- class seeds and switches --------------------------------------------------------------------

#[derive(Default)]
struct Seeds {
    exp: bool,
    sqrt: bool,
    cfg_macro: Option<usize>,
}

impl Seeds {
    fn name(&mut self, p: &syn::Path) {
        if let Some(s) = p.segments.last() {
            match s.ident.to_string().as_str() {
                "exp" | "exp2" | "exp_m1" => self.exp = true,
                "sqrt" | "rsqrt" => self.sqrt = true,
                _ => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for Seeds {
    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        match m.method.to_string().as_str() {
            "exp" | "exp2" | "exp_m1" => self.exp = true,
            "sqrt" | "rsqrt" => self.sqrt = true,
            _ => {}
        }
        syn::visit::visit_expr_method_call(self, m);
    }
    // A one-segment path is a local unless it is called: `let exp = e - 127` is not `f32::exp`.
    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*c.func {
            if p.path.segments.len() == 1 {
                self.name(&p.path);
            }
        }
        syn::visit::visit_expr_call(self, c);
    }
    fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
        if p.path.segments.len() > 1 {
            self.name(&p.path);
        }
        syn::visit::visit_expr_path(self, p);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        if m.path.is_ident("cfg") && self.cfg_macro.is_none() {
            self.cfg_macro = Some(line_of(&m.path));
        }
        for e in macro_exprs(&m.tokens) {
            self.visit_expr(&e);
        }
    }
}

// ---- strongly connected components ---------------------------------------------------------------

/// Tarjan's algorithm, iterative. Returns each vertex's component id; ids are in reverse
/// topological order (a component's id is below every component that reaches it).
///
/// Cost: time O(V + E), space O(V).
fn scc(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0; n];
    let mut on = vec![false; n];
    let mut comp = vec![usize::MAX; n];
    let mut stack = Vec::new();
    let mut next = 0;
    let mut ncomp = 0;
    for s in 0..n {
        if index[s] != usize::MAX {
            continue;
        }
        let mut work: Vec<(usize, usize)> = vec![(s, 0)];
        index[s] = next;
        low[s] = next;
        next += 1;
        stack.push(s);
        on[s] = true;
        while let Some(&mut (v, ref mut k)) = work.last_mut() {
            if *k < adj[v].len() {
                let w = adj[v][*k];
                *k += 1;
                if index[w] == usize::MAX {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on[w] = true;
                    work.push((w, 0));
                } else if on[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                work.pop();
                if let Some(&(u, _)) = work.last() {
                    low[u] = low[u].min(low[v]);
                }
                if low[v] == index[v] {
                    while let Some(w) = stack.pop() {
                        on[w] = false;
                        comp[w] = ncomp;
                        if w == v {
                            break;
                        }
                    }
                    ncomp += 1;
                }
            }
        }
    }
    comp
}

/// Propagates class bits up the call graph: a fn's classes are its seeds or'd with every callee's.
///
/// Cost: time O(V + E) after the SCC pass, space O(V).
fn propagate(adj: &[Vec<usize>], seeds: &[u8]) -> Vec<u8> {
    let comp = scc(adj);
    let ncomp = comp.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ncomp];
    for (v, &c) in comp.iter().enumerate() {
        members[c].push(v);
    }
    let mut bits = vec![0u8; ncomp];
    // Component ids are reverse-topological: callees complete before callers.
    for c in 0..ncomp {
        let mut b = 0u8;
        for &v in &members[c] {
            b |= seeds[v];
            for &w in &adj[v] {
                if comp[w] != c {
                    b |= bits[comp[w]];
                }
            }
        }
        bits[c] = b;
    }
    comp.iter().map(|&c| bits[c]).collect()
}

// ---- the analysis --------------------------------------------------------------------------------

pub(crate) struct Analysis {
    pub findings: Vec<Finding>,
    pub keys: Vec<String>,
    /// Every minimal forward body, declared or not.
    pub forwards: Vec<String>,
    pub stats: Vec<(&'static str, usize)>,
}

/// Unique keys: `file:Type::name`, with `#2`, `#3` on repeats in source order.
fn keys(tree: &Tree) -> Vec<String> {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    tree.fns
        .iter()
        .map(|f| {
            let k = f.key();
            let n = seen.entry(k.clone()).or_insert(0);
            *n += 1;
            if *n == 1 {
                k
            } else {
                format!("{k}#{n}")
            }
        })
        .collect()
}

fn reverse(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut rev = vec![Vec::new(); adj.len()];
    for (v, outs) in adj.iter().enumerate() {
        for &w in outs {
            rev[w].push(v);
        }
    }
    rev
}

/// Runs every rule.
///
/// Cost: time O(N + V + E + C) over N AST nodes, V fns, E edges and C clone-candidate pairs; the
/// consumer counts are one reverse traversal per finding, O(K·(V + E)) for K counted findings.
pub(crate) fn analyse(tree: &Tree, idx: &super::graph::Index, edges: &[Edges]) -> Analysis {
    let r = Resolver::new(tree, idx);
    let keys = keys(tree);
    let n = tree.fns.len();
    let precise: Vec<Vec<usize>> = edges
        .iter()
        .map(|e| e.precise.iter().copied().collect())
        .collect();
    let loose: Vec<Vec<usize>> = edges
        .iter()
        .map(|e| e.loose.iter().copied().collect())
        .collect();
    let rev_loose = reverse(&loose);
    let consumers = |v: usize| reach(&rev_loose, &[v]).iter().filter(|b| **b).count() - 1;
    let at = |i: usize| format!("{}:{}", tree.fns[i].file, tree.fns[i].line);
    let mut findings = Vec::new();

    let clock = std::time::Instant::now();
    let mut marks: Vec<(&'static str, usize)> = Vec::new();
    let mut mark = |name: &'static str| marks.push((name, clock.elapsed().as_millis() as usize));
    let mults = multipliers(tree);
    mark("ms_multipliers");
    let mut mac: Vec<Option<usize>> = vec![None; n];
    let mut seeds = vec![0u8; n];
    for (i, f) in tree.fns.iter().enumerate() {
        let scope = Scope::new(&r, f);
        let data = taint(f, false);
        let mut m = Mac {
            data: &data,
            scope: &scope,
            mults: &mults,
            depth: 0,
            hit: None,
        };
        m.visit_block(&f.body);
        mac[i] = m.hit;
        let mut s = Seeds::default();
        s.visit_block(&f.body);
        if mac[i].is_some() || under(&f.file, &P_FILES) {
            seeds[i] |= 1;
        }
        if s.exp || under(&f.file, &A_FILES) {
            seeds[i] |= 2;
        }
        if s.sqrt || under(&f.file, &N_FILES) {
            seeds[i] |= 4;
        }
        if is_src(f) {
            if let Some(l) = s.cfg_macro {
                findings.push(Finding {
                    rule: "switch",
                    items: vec![keys[i].clone()],
                    at: format!("{}:{l}", f.file),
                    count: None,
                    note: "`cfg!` selects a body at compile time".into(),
                });
            }
        }
    }

    mark("ms_mac");
    // product (K1)
    for (i, f) in tree.fns.iter().enumerate() {
        if let (true, false, Some(l)) = (is_src(f), f.file.starts_with(KERN), mac[i]) {
            findings.push(Finding {
                rule: "product",
                items: vec![keys[i].clone()],
                at: format!("{}:{l}", f.file),
                count: None,
                note: "multiply-accumulate loop outside src/engine/kern/".into(),
            });
        }
    }

    // kernel adapter (K2)
    for (i, f) in tree.fns.iter().enumerate() {
        if !(is_src(f) && f.file.starts_with("src/learn/")) {
            continue;
        }
        let bad: Vec<&str> = edges[i]
            .precise
            .iter()
            .map(|&w| tree.fns[w].file.as_str())
            .filter(|file| file.starts_with(KERN) && *file != LIMB)
            .collect();
        if let Some(file) = bad.first() {
            findings.push(Finding {
                rule: "kernel-adapter",
                items: vec![keys[i].clone()],
                at: at(i),
                count: None,
                note: format!("calls {file}, not {LIMB}"),
            });
        }
    }

    // forward
    let bits = propagate(&precise, &seeds);
    let is_fwd = |v: usize| bits[v] == 7;
    let mut forwards = Vec::new();
    for (i, callees) in precise.iter().enumerate() {
        if is_src(&tree.fns[i]) && is_fwd(i) && !callees.iter().any(|&w| w != i && is_fwd(w)) {
            forwards.push(i);
        }
    }

    mark("ms_forward");
    // descent and backward
    let in_bwd = |v: usize| under(&tree.fns[v].file, &BACKWARD);
    for (i, f) in tree.fns.iter().enumerate() {
        if f.test {
            continue;
        }
        if let Some(l) = descent(f) {
            findings.push(Finding {
                rule: "descent",
                items: vec![keys[i].clone()],
                at: format!("{}:{l}", f.file),
                count: Some(consumers(i)),
                note: "sqrt-normalised parameter step".into(),
            });
        }
    }
    let rev_precise = reverse(&precise);
    for i in 0..n {
        if !in_bwd(i) || tree.fns[i].test {
            continue;
        }
        // Precise callers only: a loose candidate set names every same-named method.
        let entered = rev_precise[i]
            .iter()
            .any(|&c| !in_bwd(c) && !tree.fns[c].test);
        if entered {
            findings.push(Finding {
                rule: "backward",
                items: vec![keys[i].clone()],
                at: at(i),
                count: Some(consumers(i)),
                note: "gradient body entered from outside the backward trees".into(),
            });
        }
    }

    mark("ms_descent_backward");
    // switch: environment reads and cfg alternates
    let mut by_key: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, f) in tree.fns.iter().enumerate() {
        if !is_src(f) {
            continue;
        }
        if let Some(var) = edges[i].external.iter().find(|e| {
            matches!(
                e.as_str(),
                "std::env::var" | "std::env::var_os" | "std::env::vars"
            )
        }) {
            findings.push(Finding {
                rule: "switch",
                items: vec![keys[i].clone()],
                at: at(i),
                count: None,
                note: format!("reads {var}"),
            });
        }
        by_key.entry(f.key()).or_default().push(i);
    }
    for (k, ids) in &by_key {
        let cfgs: BTreeSet<&str> = ids.iter().map(|&i| tree.fns[i].cfg.as_str()).collect();
        if ids.len() > 1 && cfgs.len() > 1 {
            findings.push(Finding {
                rule: "switch",
                items: vec![k.clone()],
                at: at(ids[0]),
                count: Some(ids.len()),
                note: format!(
                    "cfg alternates: {}",
                    cfgs.into_iter().collect::<Vec<_>>().join(" | ")
                ),
            });
        }
    }

    // orphan
    let mut roots = Vec::new();
    let mut prod_roots = Vec::new();
    for (i, f) in tree.fns.iter().enumerate() {
        let kind = tree.krates[f.krate].kind;
        let external_trait = f.trait_name.as_ref().is_some_and(|t| {
            !idx.types.get(t).is_some_and(|ps| {
                ps.iter()
                    .any(|p| tree.types.get(p).is_some_and(|d| d.is_trait))
            })
        });
        let lib_api = kind == CrateKind::Lib && !f.file.starts_with("src/");
        let prod = f.name == "main"
            || f.entry_attr
            || external_trait
            || (tree.extracted.contains(&f.file) && tree.extract_names.contains(&f.name))
            || lib_api
            || matches!(kind, CrateKind::Example | CrateKind::Fuzz);
        if prod {
            prod_roots.push(i);
            roots.push(i);
        } else if f.test {
            roots.push(i);
        }
    }
    let live = reach(&loose, &roots);
    let prod_live = reach(&loose, &prod_roots);
    // One orphan finding per file, its items the unreached fns: a debt row carries the file's
    // exact set, so a new orphan in the file is a changed set and a refusal.
    let mut test_only = 0;
    let mut orphans: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, f) in tree.fns.iter().enumerate() {
        if !is_src(f) {
            continue;
        }
        if !live[i] {
            orphans.entry(f.file.as_str()).or_default().push(i);
        } else if !prod_live[i] {
            test_only += 1;
        }
    }
    for ids in orphans.values() {
        let mut items: Vec<String> = ids.iter().map(|&i| keys[i].clone()).collect();
        items.sort();
        findings.push(Finding {
            rule: "orphan",
            note: format!("{} fns no root reaches", items.len()),
            items,
            at: at(ids[0]),
            count: None,
        });
    }
    for file in &tree.dead_files {
        findings.push(Finding {
            rule: "dead-file",
            items: vec![file.clone()],
            at: format!("{file}:1"),
            count: None,
            note: "no `mod` declaration reaches this file".into(),
        });
    }

    mark("ms_orphan");
    // clones
    for (group, kind) in super::clones::groups(tree) {
        let mut items: Vec<String> = group.iter().map(|&i| keys[i].clone()).collect();
        items.sort();
        findings.push(Finding {
            rule: kind,
            at: at(group[0]),
            count: Some(items.len()),
            items,
            note: "same normalised token sequence".into(),
        });
    }

    mark("ms_clones");
    let fwd_keys: Vec<String> = forwards.iter().map(|&i| keys[i].clone()).collect();
    for &i in &forwards {
        findings.push(Finding {
            rule: "forward",
            items: vec![keys[i].clone()],
            at: at(i),
            count: Some(consumers(i)),
            note: "minimal forward (product, exponential and root in its precise reach)".into(),
        });
    }

    mark("ms_forward_consumers");
    let mut stats = vec![
        ("fns", n),
        ("edges_precise", precise.iter().map(Vec::len).sum()),
        ("edges_loose", loose.iter().map(Vec::len).sum()),
        ("mac_fns", mac.iter().filter(|m| m.is_some()).count()),
        ("multipliers", mults.len()),
        ("test_only_src_fns", test_only),
    ];
    let _ = segs;
    stats.extend(marks);
    Analysis {
        findings,
        keys,
        forwards: fwd_keys,
        stats,
    }
}
