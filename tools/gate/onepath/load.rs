//! The loader: every crate of the repository, every module reached by a `mod` declaration, every
//! fn and type, parsed by `syn`.
//!
//! Crates: the library (`src/lib.rs`), the workspace members (`derive/src/lib.rs`,
//! `fold-abi/src/lib.rs`), and one crate per Cargo target (`tests/*.rs`, `tests/*/main.rs`,
//! `examples/*.rs`, `examples/*/main.rs`, `benches/*.rs`, `fuzz/fuzz_targets/*.rs`). A module file
//! follows Rust's rule: `mod c;` in `a/b.rs` is `a/b/c.rs` or `a/b/c/mod.rs`, in `a/mod.rs`,
//! `lib.rs` or `main.rs` it is `a/c.rs` or `a/c/mod.rs`; `#[path]` is relative to the declaring
//! file's directory. A source file under `src/` that no `mod` declaration reaches is reported dead.
//!
//! Test code: a test crate, an item under `#[cfg(test)]` (or a cfg naming `test` without `not`),
//! and a `#[test]` fn.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Command, Stdio};

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;

/// The `.rs` sources the gate reads, by repository-relative path.
pub(crate) type Sources = BTreeMap<String, String>;

const SOURCE_DIRS: [&str; 8] = [
    "src",
    "tests",
    "examples",
    "benches",
    "derive/src",
    "fold-abi/src",
    "fuzz/fuzz_targets",
    "proof/extract",
];

/// Reads every `.rs` file under [`SOURCE_DIRS`] from the working tree.
///
/// Cost: time O(B) over the B bytes read, space O(B).
pub(crate) fn standing_sources(root: &Path) -> Result<Sources, String> {
    let mut out = Sources::new();
    for dir in SOURCE_DIRS {
        walk(root, dir, &mut out)?;
    }
    Ok(out)
}

fn walk(root: &Path, rel: &str, out: &mut Sources) -> Result<(), String> {
    let path = root.join(rel);
    let Ok(entries) = std::fs::read_dir(&path) else {
        return Ok(());
    };
    for entry in entries {
        let entry = entry.map_err(|e| format!("{rel}: {e}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let child = format!("{rel}/{name}");
        let kind = entry.file_type().map_err(|e| format!("{child}: {e}"))?;
        if kind.is_dir() {
            if name != "target" && !name.starts_with('.') {
                walk(root, &child, out)?;
            }
        } else if name.ends_with(".rs") {
            let text =
                std::fs::read_to_string(entry.path()).map_err(|e| format!("{child}: {e}"))?;
            out.insert(child, text);
        }
    }
    Ok(())
}

/// Reads every `.rs` file under [`SOURCE_DIRS`] from the git index (`--staged`): the tree a
/// commit would record, not the working copy.
///
/// Cost: one `git ls-files` and one `git cat-file --batch`; time O(B) over the B bytes.
pub(crate) fn staged_sources(root: &Path) -> Result<Sources, String> {
    git_sources(root, None)
}

/// Reads every `.rs` file under [`SOURCE_DIRS`] at `HEAD`: the tree the last commit recorded, for
/// the `--staged` ratchet.
///
/// Cost: one `git ls-tree` and one `git cat-file --batch`; time O(B) over the B bytes.
pub(crate) fn head_sources(root: &Path) -> Result<Sources, String> {
    git_sources(root, Some("HEAD"))
}

fn git_sources(root: &Path, rev: Option<&str>) -> Result<Sources, String> {
    let args: Vec<&str> = match rev {
        None => vec!["ls-files", "-z", "--cached", "--"],
        Some(r) => vec!["ls-tree", "-r", "-z", "--name-only", r, "--"],
    };
    let ls = Command::new("git")
        .current_dir(root)
        .args(&args)
        .args(SOURCE_DIRS)
        .output()
        .map_err(|e| format!("git {}: {e}", args[0]))?;
    if !ls.status.success() {
        return Err(format!("git {} exited {}", args[0], ls.status));
    }
    let paths: Vec<String> = ls
        .stdout
        .split(|b| *b == 0)
        .filter(|p| p.ends_with(b".rs"))
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .collect();
    let mut child = Command::new("git")
        .current_dir(root)
        .args(["cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("git cat-file: {e}"))?;
    let mut request = String::new();
    for p in &paths {
        match rev {
            Some(r) => request.push_str(&format!("{r}:{p}\n")),
            None => request.push_str(&format!(":{p}\n")),
        }
    }
    let mut stdin = child.stdin.take().ok_or("git cat-file: no stdin")?;
    let writer = std::thread::spawn(move || {
        use std::io::Write;
        stdin.write_all(request.as_bytes())
    });
    let output = child
        .wait_with_output()
        .map_err(|e| format!("git cat-file: {e}"))?;
    writer
        .join()
        .map_err(|_| "git cat-file: writer panicked".to_string())?
        .map_err(|e| format!("git cat-file: {e}"))?;
    let bytes = output.stdout;
    let mut out = Sources::new();
    let mut at = 0;
    for p in paths {
        let nl = bytes[at..]
            .iter()
            .position(|b| *b == b'\n')
            .ok_or("git cat-file: truncated header")?;
        let header = String::from_utf8_lossy(&bytes[at..at + nl]).into_owned();
        at += nl + 1;
        let size: usize = header
            .rsplit(' ')
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("git cat-file: `{header}` for {p}"))?;
        let body = bytes
            .get(at..at + size)
            .ok_or("git cat-file: truncated body")?;
        out.insert(p, String::from_utf8_lossy(body).into_owned());
        at += size + 1;
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CrateKind {
    Lib,
    Test,
    Example,
    Bench,
    Fuzz,
}

pub(crate) struct Krate {
    pub kind: CrateKind,
}

/// A simplified type: the last path segment and its type arguments. References, parentheses and
/// raw pointers are transparent; a slice or array is `[]` with its element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Ty {
    pub name: String,
    pub args: Vec<Ty>,
}

impl Ty {
    pub fn named(name: &str) -> Ty {
        Ty {
            name: name.to_string(),
            args: Vec::new(),
        }
    }
}

/// Cost: time O(T) over the T nodes of the type.
pub(crate) fn ty_of(t: &syn::Type) -> Option<Ty> {
    match t {
        syn::Type::Path(p) => {
            let seg = p.path.segments.last()?;
            let mut args = Vec::new();
            if let syn::PathArguments::AngleBracketed(a) = &seg.arguments {
                for g in &a.args {
                    if let syn::GenericArgument::Type(t) = g {
                        args.push(ty_of(t).unwrap_or_else(|| Ty::named("_")));
                    }
                }
            }
            Some(Ty {
                name: seg.ident.to_string(),
                args,
            })
        }
        syn::Type::Reference(r) => ty_of(&r.elem),
        syn::Type::Paren(p) => ty_of(&p.elem),
        syn::Type::Group(g) => ty_of(&g.elem),
        syn::Type::Ptr(p) => ty_of(&p.elem),
        syn::Type::Slice(s) => Some(Ty {
            name: "[]".into(),
            args: vec![ty_of(&s.elem).unwrap_or_else(|| Ty::named("_"))],
        }),
        syn::Type::Array(a) => Some(Ty {
            name: "[]".into(),
            args: vec![ty_of(&a.elem).unwrap_or_else(|| Ty::named("_"))],
        }),
        syn::Type::Tuple(t) => Some(Ty {
            name: "()".into(),
            args: t
                .elems
                .iter()
                .map(|e| ty_of(e).unwrap_or_else(|| Ty::named("_")))
                .collect(),
        }),
        _ => None,
    }
}

/// One fn item: free, inherent method, trait-impl method or trait default method.
pub(crate) struct Func {
    pub krate: usize,
    pub file: String,
    pub line: usize,
    /// The enclosing module, as path segments from the crate root.
    pub module: Vec<String>,
    /// The `impl` self type's simple name, or the trait's name for a trait default method.
    pub self_ty: Option<String>,
    /// `impl Trait for X` or the trait of a default method.
    pub trait_name: Option<String>,
    pub name: String,
    pub test: bool,
    /// The item's `cfg` attributes, token text, outside test cfgs; empty when unconditional.
    pub cfg: String,
    /// Attributes that make the fn an entry by itself: `no_mangle`, `proc_macro*`, `export_name`.
    pub entry_attr: bool,
    pub sig: syn::Signature,
    pub body: syn::Block,
    /// The type parameters in scope: the enclosing `impl`/trait's and the fn's own.
    pub generics: Vec<String>,
    /// A `const`/`static` initializer, not a fn.
    pub init: bool,
}

/// The type parameter names of `outer` followed by those of `sig`.
fn type_params(outer: &syn::Generics, sig: &syn::Signature) -> Vec<String> {
    outer
        .type_params()
        .chain(sig.generics.type_params())
        .map(|p| p.ident.to_string())
        .collect()
}

impl Func {
    /// The finding key: `file:Type::name` or `file:name`.
    pub fn key(&self) -> String {
        match &self.self_ty {
            Some(t) => format!("{}:{}::{}", self.file, t, self.name),
            None => format!("{}:{}", self.file, self.name),
        }
    }
}

/// A struct, enum, union or trait.
pub(crate) struct TypeDef {
    pub name: String,
    pub fields: BTreeMap<String, Ty>,
    pub is_trait: bool,
}

pub(crate) struct Module {
    pub krate: usize,
    /// `use` aliases: the name bound in this module and the path it names, unresolved.
    pub uses: BTreeMap<String, Vec<String>>,
    pub globs: Vec<Vec<String>>,
    pub children: BTreeSet<String>,
}

#[derive(Default)]
pub(crate) struct Tree {
    pub krates: Vec<Krate>,
    pub modules: BTreeMap<Vec<String>, Module>,
    pub fns: Vec<Func>,
    pub types: BTreeMap<Vec<String>, TypeDef>,
    /// `impl Trait for Type` pairs, by the type's simple name.
    pub impls: BTreeMap<String, BTreeSet<String>>,
    /// Every file a crate reached, and the files under `src/` no crate reached.
    pub reached: BTreeSet<String>,
    pub dead_files: Vec<String>,
    /// Files that `proof/extract/lib.rs` includes by `#[path]`: the extracted contract.
    pub extracted: BTreeSet<String>,
    /// Every identifier the extraction's own items (its wrappers and `pub use` mirrors) name: an
    /// extracted fn is a root only when the extraction names it, not because its file is included.
    pub extract_names: BTreeSet<String>,
    pub errors: Vec<String>,
}

/// The library crates: root file and crate name. `apeiron` resolves to the first.
const LIBS: [(&str, &str); 3] = [
    ("src/lib.rs", "apeiron"),
    ("derive/src/lib.rs", "apeiron_derive"),
    ("fold-abi/src/lib.rs", "apeiron_fold_abi"),
];

/// Builds the tree from the sources.
///
/// Cost: time O(B) to parse B bytes plus O(I) over the I items; space O(B) for the retained ASTs.
pub(crate) fn load(src: &Sources) -> Tree {
    let mut tree = Tree::default();
    for (file, name) in LIBS {
        if src.contains_key(file) {
            tree.add_crate(src, file, name, CrateKind::Lib);
        }
    }
    for file in src.keys() {
        let parts: Vec<&str> = file.split('/').collect();
        let kind = match parts.as_slice() {
            ["tests", f] if f.ends_with(".rs") => Some(CrateKind::Test),
            ["tests", _, "main.rs"] => Some(CrateKind::Test),
            ["examples", f] if f.ends_with(".rs") => Some(CrateKind::Example),
            ["examples", _, "main.rs"] => Some(CrateKind::Example),
            ["benches", f] if f.ends_with(".rs") => Some(CrateKind::Bench),
            ["fuzz", "fuzz_targets", f] if f.ends_with(".rs") => Some(CrateKind::Fuzz),
            _ => None,
        };
        if let Some(kind) = kind {
            let stem = file.trim_end_matches(".rs").replace('/', "#");
            tree.add_crate(src, file, &stem, kind);
        }
    }
    if let Some(text) = src.get("proof/extract/lib.rs") {
        if let Ok(f) = syn::parse_file(text) {
            for item in &f.items {
                match item {
                    syn::Item::Mod(m) => match path_attr(&m.attrs) {
                        Some(p) => {
                            tree.extracted
                                .insert(normalise(&format!("proof/extract/{p}")));
                        }
                        // An inline mirror module (`pub mod map { pub use super::table; }`).
                        None => idents(&m.to_token_stream(), &mut tree.extract_names),
                    },
                    other => idents(&other.to_token_stream(), &mut tree.extract_names),
                }
            }
        }
        // A `#[path]` module inside `proof/extract/` is the extraction's own port: its items
        // name what it extracts, as the root file's do.
        for file in &tree.extracted {
            if !file.starts_with("proof/extract/") {
                continue;
            }
            if let Some(f) = src.get(file).and_then(|t| syn::parse_file(t).ok()) {
                idents(&f.to_token_stream(), &mut tree.extract_names);
            }
        }
    }
    for file in src.keys() {
        if file.starts_with("src/") && !tree.reached.contains(file) {
            tree.dead_files.push(file.clone());
        }
    }
    tree
}

/// Every identifier in `ts`, at any depth.
///
/// Cost: time O(T) over the T tokens.
fn idents(ts: &TokenStream, out: &mut BTreeSet<String>) {
    for t in ts.clone() {
        match t {
            TokenTree::Ident(i) => {
                out.insert(i.to_string());
            }
            TokenTree::Group(g) => idents(&g.stream(), out),
            _ => {}
        }
    }
}

/// Collapses `a/b/../c` to `a/c`.
fn normalise(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "." | "" => {}
            ".." => {
                out.pop();
            }
            p => out.push(p),
        }
    }
    out.join("/")
}

fn path_attr(attrs: &[syn::Attribute]) -> Option<String> {
    for a in attrs {
        if a.path().is_ident("path") {
            if let syn::Meta::NameValue(nv) = &a.meta {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
                {
                    return Some(s.value());
                }
            }
        }
    }
    None
}

/// Whether a token stream names the ident `test` and no `not`: `cfg(test)`, `cfg(any(test, ..))`.
fn names_test(tokens: &TokenStream) -> bool {
    fn scan(tokens: &TokenStream, test: &mut bool, not: &mut bool) {
        for t in tokens.clone() {
            match t {
                TokenTree::Ident(i) if i == "test" => *test = true,
                TokenTree::Ident(i) if i == "not" => *not = true,
                TokenTree::Group(g) => scan(&g.stream(), test, not),
                _ => {}
            }
        }
    }
    let (mut test, mut not) = (false, false);
    scan(tokens, &mut test, &mut not);
    test && !not
}

/// (is test code, the non-test cfg text, is an entry by attribute).
fn attrs_of(attrs: &[syn::Attribute]) -> (bool, String, bool) {
    let mut test = false;
    let mut cfg = String::new();
    let mut entry = false;
    for a in attrs {
        let p = a.path();
        if p.is_ident("test") || p.is_ident("bench") {
            test = true;
        } else if p.is_ident("cfg") {
            if let syn::Meta::List(l) = &a.meta {
                if names_test(&l.tokens) {
                    test = true;
                } else {
                    cfg.push_str(&l.tokens.to_string());
                }
            }
        } else if p.is_ident("no_mangle")
            || p.is_ident("export_name")
            || p.is_ident("proc_macro")
            || p.is_ident("proc_macro_derive")
            || p.is_ident("proc_macro_attribute")
        {
            entry = true;
        } else if p.is_ident("unsafe") {
            if let syn::Meta::List(l) = &a.meta {
                let t = l.tokens.to_string();
                if t.contains("no_mangle") || t.contains("export_name") {
                    entry = true;
                }
            }
        }
    }
    (test, cfg, entry)
}

struct Ctx<'a> {
    src: &'a Sources,
    krate: usize,
    file: String,
    /// The directory a child `mod c;` resolves in.
    child_dir: String,
    /// The declaring file's directory, for `#[path]`.
    file_dir: String,
    module: Vec<String>,
    test: bool,
}

impl Tree {
    fn add_crate(&mut self, src: &Sources, file: &str, name: &str, kind: CrateKind) {
        let krate = self.krates.len();
        self.krates.push(Krate { kind });
        let dir = parent(file);
        let ctx = Ctx {
            src,
            krate,
            file: file.to_string(),
            child_dir: dir.clone(),
            file_dir: dir,
            module: vec![name.to_string()],
            test: kind != CrateKind::Lib,
        };
        self.load_file(&ctx);
    }

    fn load_file(&mut self, ctx: &Ctx) {
        let Some(text) = ctx.src.get(&ctx.file) else {
            self.errors
                .push(format!("{}: module file not found", ctx.file));
            return;
        };
        self.reached.insert(ctx.file.clone());
        match syn::parse_file(text) {
            Ok(f) => self.load_items(ctx, &f.items),
            Err(e) => {
                let at = e.span().start();
                self.errors
                    .push(format!("{}:{}: parse error: {e}", ctx.file, at.line));
            }
        }
    }

    fn load_items(&mut self, ctx: &Ctx, items: &[syn::Item]) {
        self.modules.entry(ctx.module.clone()).or_insert(Module {
            krate: ctx.krate,
            uses: BTreeMap::new(),
            globs: Vec::new(),
            children: BTreeSet::new(),
        });
        for item in items {
            self.load_item(ctx, item);
        }
    }

    fn module_mut(&mut self, path: &[String]) -> &mut Module {
        self.modules
            .get_mut(path)
            .expect("the module is inserted before its items")
    }

    /// Pushes the const expressions of type definition `ident` (array lengths, const generic
    /// arguments, enum discriminants) as a root node. rustc's `dead_code` lint, under
    /// `clippy -D warnings`, already refuses an unused private type; a type is not a node of the
    /// call graph, so the consts its layout names are reached from here.
    fn push_type_consts(
        &mut self,
        ctx: &Ctx,
        attrs: &[syn::Attribute],
        ident: &syn::Ident,
        item: &syn::Item,
    ) {
        let mut c = TypeConsts(Vec::new());
        syn::visit::Visit::visit_item(&mut c, item);
        if c.0.is_empty() {
            return;
        }
        let (test, cfg, _) = attrs_of(attrs);
        let mut sig: syn::Signature = syn::parse_quote!(fn layout());
        sig.ident = ident.clone();
        self.fns.push(Func {
            krate: ctx.krate,
            file: ctx.file.clone(),
            line: ident.span().start().line,
            module: ctx.module.clone(),
            self_ty: None,
            trait_name: None,
            name: format!("{ident}#layout"),
            test: ctx.test || test,
            cfg,
            entry_attr: true,
            generics: Vec::new(),
            init: true,
            sig,
            body: syn::Block {
                brace_token: Default::default(),
                stmts: c
                    .0
                    .into_iter()
                    .map(|e| syn::Stmt::Expr(e, Some(Default::default())))
                    .collect(),
            },
        });
    }

    /// Pushes the initializer of `const`/`static` `ident: ty` as a free node named `ident`
    /// returning `ty`; an anonymous `const _` is a root.
    fn push_init(
        &mut self,
        ctx: &Ctx,
        attrs: &[syn::Attribute],
        ident: &syn::Ident,
        ty: &syn::Type,
        expr: &syn::Expr,
    ) {
        let (test, cfg, _) = attrs_of(attrs);
        let mut sig: syn::Signature = syn::parse_quote!(fn init() -> #ty);
        sig.ident = ident.clone();
        self.fns.push(Func {
            krate: ctx.krate,
            file: ctx.file.clone(),
            line: ident.span().start().line,
            module: ctx.module.clone(),
            self_ty: None,
            trait_name: None,
            name: ident.to_string(),
            test: ctx.test || test,
            cfg,
            // `const _` is evaluated by rustc whether or not anything names it.
            entry_attr: ident == "_",
            generics: Vec::new(),
            init: true,
            sig,
            body: syn::Block {
                brace_token: Default::default(),
                stmts: vec![syn::Stmt::Expr(expr.clone(), None)],
            },
        });
    }

    fn load_item(&mut self, ctx: &Ctx, item: &syn::Item) {
        match item {
            syn::Item::Fn(f) => {
                let (test, cfg, entry) = attrs_of(&f.attrs);
                self.fns.push(Func {
                    krate: ctx.krate,
                    file: ctx.file.clone(),
                    line: f.sig.ident.span().start().line,
                    module: ctx.module.clone(),
                    self_ty: None,
                    trait_name: None,
                    name: f.sig.ident.to_string(),
                    test: ctx.test || test,
                    cfg,
                    entry_attr: entry,
                    sig: f.sig.clone(),
                    body: (*f.block).clone(),
                    generics: type_params(&syn::Generics::default(), &f.sig),
                    init: false,
                });
            }
            syn::Item::Impl(i) => {
                let (test, cfg, _) = attrs_of(&i.attrs);
                let Some(self_ty) = ty_of(&i.self_ty) else {
                    return;
                };
                let trait_name = i
                    .trait_
                    .as_ref()
                    .and_then(|(_, p, _)| p.segments.last())
                    .map(|s| s.ident.to_string());
                if let Some(t) = &trait_name {
                    self.impls
                        .entry(self_ty.name.clone())
                        .or_default()
                        .insert(t.clone());
                }
                for it in &i.items {
                    if let syn::ImplItem::Fn(m) = it {
                        let (mtest, mcfg, entry) = attrs_of(&m.attrs);
                        self.fns.push(Func {
                            krate: ctx.krate,
                            file: ctx.file.clone(),
                            line: m.sig.ident.span().start().line,
                            module: ctx.module.clone(),
                            self_ty: Some(self_ty.name.clone()),
                            trait_name: trait_name.clone(),
                            name: m.sig.ident.to_string(),
                            test: ctx.test || test || mtest,
                            cfg: format!("{cfg}{mcfg}"),
                            entry_attr: entry,
                            sig: m.sig.clone(),
                            body: m.block.clone(),
                            generics: type_params(&i.generics, &m.sig),
                            init: false,
                        });
                    }
                }
            }
            syn::Item::Trait(t) => {
                let (test, cfg, _) = attrs_of(&t.attrs);
                let name = t.ident.to_string();
                let mut path = ctx.module.clone();
                path.push(name.clone());
                self.types.insert(
                    path,
                    TypeDef {
                        name: name.clone(),
                        fields: BTreeMap::new(),
                        is_trait: true,
                    },
                );
                for it in &t.items {
                    if let syn::TraitItem::Fn(m) = it {
                        if let Some(body) = &m.default {
                            let (mtest, mcfg, _) = attrs_of(&m.attrs);
                            self.fns.push(Func {
                                krate: ctx.krate,
                                file: ctx.file.clone(),
                                line: m.sig.ident.span().start().line,
                                module: ctx.module.clone(),
                                self_ty: Some(name.clone()),
                                trait_name: Some(name.clone()),
                                name: m.sig.ident.to_string(),
                                test: ctx.test || test || mtest,
                                cfg: format!("{cfg}{mcfg}"),
                                entry_attr: false,
                                sig: m.sig.clone(),
                                body: body.clone(),
                                generics: type_params(&t.generics, &m.sig),
                                init: false,
                            });
                        }
                    }
                }
            }
            syn::Item::Struct(s) => {
                self.push_type_consts(ctx, &s.attrs, &s.ident, item);
                let mut fields = BTreeMap::new();
                for (k, f) in s.fields.iter().enumerate() {
                    let key = f
                        .ident
                        .as_ref()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| k.to_string());
                    if let Some(t) = ty_of(&f.ty) {
                        fields.insert(key, t);
                    }
                }
                self.add_type(ctx, &s.ident, fields);
            }
            syn::Item::Enum(e) => {
                self.push_type_consts(ctx, &e.attrs, &e.ident, item);
                self.add_type(ctx, &e.ident, BTreeMap::new())
            }
            syn::Item::Union(u) => {
                self.push_type_consts(ctx, &u.attrs, &u.ident, item);
                self.add_type(ctx, &u.ident, BTreeMap::new())
            }
            syn::Item::Type(t) => {
                self.push_type_consts(ctx, &t.attrs, &t.ident, item);
                // `type A = B<..>;` resolves `A` to `B`'s simple name for method lookup.
                if let Some(target) = ty_of(&t.ty) {
                    let mut path = ctx.module.clone();
                    path.push(t.ident.to_string());
                    self.types.insert(
                        path,
                        TypeDef {
                            name: target.name,
                            fields: BTreeMap::new(),
                            is_trait: false,
                        },
                    );
                }
            }
            // A `const` or `static` is a node: its initializer runs on the build path, so a
            // reference to the item reaches the fns the initializer calls.
            syn::Item::Const(c) => self.push_init(ctx, &c.attrs, &c.ident, &c.ty, &c.expr),
            syn::Item::Static(s) => self.push_init(ctx, &s.attrs, &s.ident, &s.ty, &s.expr),
            syn::Item::Use(u) => {
                let (test, _, _) = attrs_of(&u.attrs);
                if !test || ctx.test {
                    let mut prefix = Vec::new();
                    self.add_use(&ctx.module, &u.tree, &mut prefix);
                }
            }
            syn::Item::Mod(m) => self.load_mod(ctx, m),
            // `include!("f.rs")` at item position: f's items join this module, the path relative
            // to the including file's directory.
            syn::Item::Macro(m) if m.mac.path.is_ident("include") => {
                let Ok(lit) = m.mac.parse_body::<syn::LitStr>() else {
                    return;
                };
                let inner = Ctx {
                    src: ctx.src,
                    krate: ctx.krate,
                    file: normalise(&format!("{}/{}", parent(&ctx.file), lit.value())),
                    child_dir: ctx.child_dir.clone(),
                    file_dir: ctx.file_dir.clone(),
                    module: ctx.module.clone(),
                    test: ctx.test,
                };
                self.load_file(&inner);
            }
            _ => {}
        }
    }

    fn add_type(&mut self, ctx: &Ctx, ident: &syn::Ident, fields: BTreeMap<String, Ty>) {
        let mut path = ctx.module.clone();
        path.push(ident.to_string());
        self.types.insert(
            path,
            TypeDef {
                name: ident.to_string(),
                fields,
                is_trait: false,
            },
        );
    }

    fn add_use(&mut self, module: &[String], tree: &syn::UseTree, prefix: &mut Vec<String>) {
        match tree {
            syn::UseTree::Path(p) => {
                prefix.push(p.ident.to_string());
                self.add_use(module, &p.tree, prefix);
                prefix.pop();
            }
            syn::UseTree::Name(n) => {
                let name = n.ident.to_string();
                let mut full = prefix.clone();
                if name == "self" {
                    let alias = prefix.last().cloned().unwrap_or_default();
                    self.module_mut(module).uses.insert(alias, full);
                } else {
                    full.push(name.clone());
                    self.module_mut(module).uses.insert(name, full);
                }
            }
            syn::UseTree::Rename(r) => {
                let mut full = prefix.clone();
                if r.ident != "self" {
                    full.push(r.ident.to_string());
                }
                self.module_mut(module)
                    .uses
                    .insert(r.rename.to_string(), full);
            }
            syn::UseTree::Glob(_) => {
                let full = prefix.clone();
                self.module_mut(module).globs.push(full);
            }
            syn::UseTree::Group(g) => {
                for t in &g.items {
                    self.add_use(module, t, prefix);
                }
            }
        }
    }

    fn load_mod(&mut self, ctx: &Ctx, m: &syn::ItemMod) {
        let (test, _, _) = attrs_of(&m.attrs);
        let name = m.ident.to_string();
        self.module_mut(&ctx.module).children.insert(name.clone());
        let mut module = ctx.module.clone();
        module.push(name.clone());
        let test = ctx.test || test;
        if let Some((_, items)) = &m.content {
            let child_dir = match path_attr(&m.attrs) {
                Some(p) => normalise(&format!("{}/{p}", ctx.child_dir)),
                None => format!("{}/{name}", ctx.child_dir),
            };
            let inner = Ctx {
                src: ctx.src,
                krate: ctx.krate,
                file: ctx.file.clone(),
                child_dir: child_dir.clone(),
                file_dir: child_dir,
                module,
                test,
            };
            self.load_items(&inner, items);
            return;
        }
        let candidates: Vec<String> = match path_attr(&m.attrs) {
            Some(p) => vec![
                normalise(&format!("{}/{p}", ctx.file_dir)),
                normalise(&format!("{}/{p}", ctx.child_dir)),
            ],
            None => vec![
                format!("{}/{name}.rs", ctx.child_dir),
                format!("{}/{name}/mod.rs", ctx.child_dir),
            ],
        };
        let Some(file) = candidates.into_iter().find(|c| ctx.src.contains_key(c)) else {
            self.errors.push(format!(
                "{}:{}: `mod {name};` names no file",
                ctx.file,
                m.ident.span().start().line
            ));
            return;
        };
        let file_dir = parent(&file);
        let child_dir = if file.ends_with("/mod.rs") || path_attr(&m.attrs).is_some() {
            if file.ends_with("/mod.rs") {
                file_dir.clone()
            } else {
                format!("{}/{}", file_dir, stem(&file))
            }
        } else {
            format!("{}/{name}", ctx.child_dir)
        };
        let inner = Ctx {
            src: ctx.src,
            krate: ctx.krate,
            file,
            child_dir,
            file_dir,
            module,
            test,
        };
        self.load_file(&inner);
    }
}

fn parent(file: &str) -> String {
    match file.rfind('/') {
        Some(i) => file[..i].to_string(),
        None => String::new(),
    }
}

fn stem(file: &str) -> String {
    let base = file.rsplit('/').next().unwrap_or(file);
    base.trim_end_matches(".rs").to_string()
}

/// The const expressions a type definition evaluates: array lengths, const generic arguments
/// and enum discriminants.
struct TypeConsts(Vec<syn::Expr>);

impl<'ast> syn::visit::Visit<'ast> for TypeConsts {
    fn visit_type_array(&mut self, a: &'ast syn::TypeArray) {
        self.0.push(a.len.clone());
        syn::visit::visit_type_array(self, a);
    }
    fn visit_generic_argument(&mut self, g: &'ast syn::GenericArgument) {
        if let syn::GenericArgument::Const(e) = g {
            self.0.push(e.clone());
        }
        syn::visit::visit_generic_argument(self, g);
    }
    fn visit_variant(&mut self, v: &'ast syn::Variant) {
        if let Some((_, e)) = &v.discriminant {
            self.0.push(e.clone());
        }
        syn::visit::visit_variant(self, v);
    }
}
