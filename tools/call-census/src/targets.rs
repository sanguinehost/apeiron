//! Target configuration and the target-crate symbol inventory.
//!
//! "Target" means the crate whose *consumers* we are censusing — for this task
//! `hashtrinity` and its sibling crates, but the tool is parameterised so the
//! same census can be pointed at `apeiron`. Nothing here uses `rustc` name
//! resolution: the inventory is a syntax-level list of items, and every
//! consumer-side attribution that depends on it says so.
//!
//! Test-only items (`#[cfg(test)]` modules, `#[test]` functions) are excluded
//! from the inventory: a struct that exists only under `cfg(test)` must not make
//! a consumer method call a candidate.

use quote::ToTokens;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// The set of crates, type names, method names, free-function names, macro
/// names, field types and constructor names that a call must match to be
/// considered relevant.
#[derive(Debug, Clone)]
pub struct TargetConfig {
    pub crates: Vec<String>,
    pub types: BTreeSet<String>,
    pub methods: BTreeSet<String>,
    pub fns: BTreeSet<String>,
    pub macros: BTreeSet<String>,
    /// type name -> (field name -> field type text)
    pub fields: HashMap<String, BTreeMap<String, String>>,
    /// enum type name -> variant names
    pub variants: HashMap<String, BTreeSet<String>>,
    /// tuple-struct type names (the type is also a constructor)
    pub tuple_structs: BTreeSet<String>,
    /// trait names defined by the target
    pub traits: BTreeSet<String>,
    /// trait name -> its method names (required and provided)
    pub trait_methods: HashMap<String, BTreeSet<String>>,
}

impl TargetConfig {
    pub fn is_target_crate(&self, name: &str) -> bool {
        self.crates.iter().any(|c| c == name)
    }

    pub fn is_target_type(&self, name: &str) -> bool {
        self.types.contains(name)
    }

    pub fn is_target_method(&self, name: &str) -> bool {
        self.methods.contains(name)
    }

    pub fn is_target_fn(&self, name: &str) -> bool {
        self.fns.contains(name)
    }

    pub fn is_target_macro(&self, name: &str) -> bool {
        self.macros.contains(name)
    }

    pub fn target_crate_for_root(&self, root: &str) -> Option<String> {
        self.crates.iter().find(|c| c.as_str() == root).cloned()
    }

    pub fn field_type(&self, type_name: &str, field: &str) -> Option<&str> {
        self.fields
            .get(type_name)
            .and_then(|m| m.get(field))
            .map(|s| s.as_str())
    }

    /// Whether `variant` is a known variant of enum `type_name`.
    pub fn is_variant(&self, type_name: &str, variant: &str) -> bool {
        self.variants
            .get(type_name)
            .map(|s| s.contains(variant))
            .unwrap_or(false)
    }

    pub fn is_tuple_struct(&self, name: &str) -> bool {
        self.tuple_structs.contains(name)
    }

    pub fn is_target_trait(&self, name: &str) -> bool {
        self.traits.contains(name)
    }

    pub fn trait_provides(&self, trait_name: &str, method: &str) -> bool {
        self.trait_methods
            .get(trait_name)
            .map(|s| s.contains(method))
            .unwrap_or(false)
    }

    /// Removes primitive type names that leaked in through `impl Trait for str`
    /// style impls. They are never target types and matching them turns every
    /// `&str` receiver into a candidate.
    pub fn prune_primitives(&mut self) {
        self.types.retain(|t| !is_primitive_type(t));
        self.tuple_structs.retain(|t| !is_primitive_type(t));
    }
}

pub fn is_primitive_type(name: &str) -> bool {
    matches!(
        name,
        "bool"
            | "char"
            | "str"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "f16"
            | "f32"
            | "f64"
            | "f128"
            | "()"
    )
}

/// Crates that are targeted when the caller passes nothing.
pub const DEFAULT_TARGET_CRATES: &[&str] = &[
    "hashtrinity",
    "hashtrinity_derive",
    "hashtrinity_crypto",
    "hashtrinity_codegen",
];

/// Fallback type names, used only when no `--target-src` inventory is supplied.
pub const DEFAULT_TARGET_TYPES: &[&str] = &[
    "Table",
    "CoordMap",
    "FreqMap",
    "Dense",
    "Group",
    "Stored",
    "Regen",
    "Face",
    "Species",
    "CoordSeed",
    "Locate",
    "Absorb",
    "Entry",
    "OccupiedEntry",
    "VacantEntry",
    "Iter",
    "IntoIter",
    "FoldExt",
    "Monoid",
    "KeyStore",
    "Coord",
    "CoordInv",
    "Validate",
    "ValidationError",
    "ValidationErrors",
    "ToSchema",
    "IntoParams",
    "Error",
    "Context",
    "SyntaxTerm",
    "Atom",
    "Surface",
    "AsTerm",
    "SyntaxError",
    "Seed",
    "WireErr",
    "Collapse",
    "Term",
];

/// Fallback method names. `--target-src` is strongly preferred.
pub const DEFAULT_TARGET_METHODS: &[&str] = &[
    "get",
    "get_exact",
    "get_all",
    "get_each",
    "get_reduce",
    "get_ref",
    "insert_with",
    "put",
    "add",
    "mark",
    "has",
    "count",
    "count_par",
    "push",
    "neighbors",
    "remove",
    "encircle",
    "encircle_floor",
    "fuse",
    "coord_seed",
    "longest_probe",
    "coordinate",
    "coordinate_with_capacity",
    "with_capacity",
    "from_term",
    "to_term",
    "collapse_encode",
    "collapse_decode",
    "collapse_open",
];

pub const DEFAULT_TARGET_MACROS: &[&str] = &[
    "Collapse",
    "Term",
    "Error",
    "Validate",
    "IntoParams",
    "ToSchema",
    "Coord",
    "Packed",
];

impl TargetConfig {
    pub fn from_names(
        crates: Vec<String>,
        extra_types: Vec<String>,
        extra_methods: Vec<String>,
    ) -> Self {
        let crates = if crates.is_empty() {
            DEFAULT_TARGET_CRATES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            crates
        };
        let mut types: BTreeSet<String> =
            DEFAULT_TARGET_TYPES.iter().map(|s| s.to_string()).collect();
        types.extend(extra_types);
        let mut methods: BTreeSet<String> = DEFAULT_TARGET_METHODS
            .iter()
            .map(|s| s.to_string())
            .collect();
        methods.extend(extra_methods);
        let macros: BTreeSet<String> = DEFAULT_TARGET_MACROS
            .iter()
            .map(|s| s.to_string())
            .collect();
        TargetConfig {
            crates,
            types,
            methods,
            fns: BTreeSet::new(),
            macros,
            fields: HashMap::new(),
            variants: HashMap::new(),
            tuple_structs: BTreeSet::new(),
            traits: BTreeSet::new(),
            trait_methods: HashMap::new(),
        }
    }
}

/// Reads `[package] name = "..."` from a Cargo.toml without a TOML dependency.
pub fn package_name_from_manifest(text: &str) -> Option<String> {
    let mut in_package = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(rest) = line.strip_prefix("name") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let val = rest.trim();
                let val = val.trim_matches('"').trim_matches('\'');
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
        }
    }
    None
}

/// Finds the nearest Cargo.toml walking up from `dir` and returns its package
/// name. Results are cached per directory by the caller.
pub fn crate_name_for_dir(dir: &Path) -> Option<(PathBuf, String)> {
    let mut cur = Some(dir);
    while let Some(d) = cur {
        let manifest = d.join("Cargo.toml");
        if manifest.is_file() {
            if let Ok(text) = std::fs::read_to_string(&manifest) {
                if let Some(name) = package_name_from_manifest(&text) {
                    return Some((manifest, name));
                }
            }
        }
        cur = d.parent();
    }
    None
}

/// True when an attribute set marks an item as test-only.
pub fn attrs_are_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        let p = a.path();
        p.is_ident("test")
            || (p.is_ident("cfg") && a.meta.to_token_stream().to_string().contains("test"))
    })
}

fn item_attrs(item: &syn::Item) -> Vec<syn::Attribute> {
    match item {
        syn::Item::Const(i) => i.attrs.clone(),
        syn::Item::Enum(i) => i.attrs.clone(),
        syn::Item::ExternCrate(i) => i.attrs.clone(),
        syn::Item::Fn(i) => i.attrs.clone(),
        syn::Item::ForeignMod(i) => i.attrs.clone(),
        syn::Item::Impl(i) => i.attrs.clone(),
        syn::Item::Macro(i) => i.attrs.clone(),
        syn::Item::Mod(i) => i.attrs.clone(),
        syn::Item::Static(i) => i.attrs.clone(),
        syn::Item::Struct(i) => i.attrs.clone(),
        syn::Item::Trait(i) => i.attrs.clone(),
        syn::Item::TraitAlias(i) => i.attrs.clone(),
        syn::Item::Type(i) => i.attrs.clone(),
        syn::Item::Union(i) => i.attrs.clone(),
        syn::Item::Use(i) => i.attrs.clone(),
        _ => Vec::new(),
    }
}

/// Adds one parsed target-crate source file to the inventory.
pub fn collect_inventory_file(cfg: &mut TargetConfig, file: &syn::File) {
    for item in &file.items {
        collect_type_names(cfg, item);
    }
    for item in &file.items {
        collect_inventory_item(cfg, item);
    }
}

fn collect_type_names(cfg: &mut TargetConfig, item: &syn::Item) {
    if attrs_are_test(&item_attrs(item)) {
        return;
    }
    match item {
        syn::Item::Struct(s) => {
            cfg.types.insert(s.ident.to_string());
        }
        syn::Item::Enum(e) => {
            cfg.types.insert(e.ident.to_string());
        }
        syn::Item::Union(u) => {
            cfg.types.insert(u.ident.to_string());
        }
        syn::Item::Trait(t) => {
            cfg.types.insert(t.ident.to_string());
        }
        syn::Item::TraitAlias(t) => {
            cfg.types.insert(t.ident.to_string());
        }
        syn::Item::Type(t) => {
            cfg.types.insert(t.ident.to_string());
        }
        syn::Item::Impl(imp) => {
            // Only an inherent impl proves the self type is local (the orphan
            // rule forbids inherent impls on foreign types).
            if imp.trait_.is_none() {
                if let Some(name) = impl_base_name(&imp.self_ty) {
                    cfg.types.insert(name);
                }
            }
        }
        syn::Item::Mod(m) => {
            if let Some((_, items)) = &m.content {
                for it in items {
                    collect_type_names(cfg, it);
                }
            }
        }
        _ => {}
    }
}

fn collect_inventory_item(cfg: &mut TargetConfig, item: &syn::Item) {
    if attrs_are_test(&item_attrs(item)) {
        return;
    }
    match item {
        syn::Item::Struct(s) => {
            let name = s.ident.to_string();
            cfg.types.insert(name.clone());
            match &s.fields {
                syn::Fields::Named(named) => {
                    let mut fields = BTreeMap::new();
                    for f in &named.named {
                        if let Some(id) = &f.ident {
                            fields.insert(id.to_string(), f.ty.to_token_stream().to_string());
                        }
                    }
                    if !fields.is_empty() {
                        cfg.fields.entry(name).or_insert(fields);
                    }
                }
                syn::Fields::Unnamed(_) => {
                    cfg.tuple_structs.insert(name);
                }
                syn::Fields::Unit => {}
            }
        }
        syn::Item::Enum(e) => {
            let name = e.ident.to_string();
            cfg.types.insert(name.clone());
            let mut variants = BTreeSet::new();
            for v in &e.variants {
                variants.insert(v.ident.to_string());
            }
            if !variants.is_empty() {
                cfg.variants.entry(name).or_insert(variants);
            }
        }
        syn::Item::Union(u) => {
            cfg.types.insert(u.ident.to_string());
        }
        syn::Item::Trait(t) => {
            let name = t.ident.to_string();
            cfg.types.insert(name.clone());
            cfg.traits.insert(name.clone());
            // Trait methods include *provided* methods such as `FoldExt::index_by`.
            let mut methods = BTreeSet::new();
            for it in &t.items {
                if let syn::TraitItem::Fn(m) = it {
                    let mname = m.sig.ident.to_string();
                    cfg.methods.insert(mname.clone());
                    methods.insert(mname);
                }
            }
            if !methods.is_empty() {
                cfg.trait_methods.entry(name).or_insert(methods);
            }
        }
        syn::Item::TraitAlias(t) => {
            cfg.types.insert(t.ident.to_string());
        }
        syn::Item::Type(t) => {
            cfg.types.insert(t.ident.to_string());
        }
        syn::Item::Fn(f) => {
            cfg.fns.insert(f.sig.ident.to_string());
        }
        syn::Item::Impl(imp) => {
            let type_name = impl_base_name(&imp.self_ty);
            if let Some(type_name) = type_name {
                // Add the self type only when it is a nominal local type or an
                // inherent impl; never because of `impl Trait for Vec`.
                if imp.trait_.is_none() || cfg.types.contains(&type_name) {
                    cfg.types.insert(type_name.clone());
                }
                let is_target_trait = match &imp.trait_ {
                    None => true,
                    Some((_, path, _)) => {
                        let first_is_target = path
                            .segments
                            .first()
                            .map(|s| cfg.is_target_crate(&s.ident.to_string()))
                            .unwrap_or(false);
                        let last = path
                            .segments
                            .last()
                            .map(|s| s.ident.to_string())
                            .unwrap_or_default();
                        first_is_target || !is_std_trait_name(&last)
                    }
                };
                if is_target_trait {
                    for it in &imp.items {
                        if let syn::ImplItem::Fn(m) = it {
                            cfg.methods.insert(m.sig.ident.to_string());
                        }
                    }
                }
            }
        }
        syn::Item::Macro(m) => {
            // Only exported macros are part of the public API.
            let exported = m.attrs.iter().any(|a| a.path().is_ident("macro_export"));
            if exported {
                if let Some(ident) = &m.ident {
                    cfg.macros.insert(ident.to_string());
                }
            }
        }
        syn::Item::Mod(m) => {
            if let Some((_, items)) = &m.content {
                for it in items {
                    collect_inventory_item(cfg, it);
                }
            }
        }
        _ => {}
    }
    for attr in item_attrs(item) {
        let p = attr.path();
        if p.is_ident("proc_macro_derive") {
            let mut first = true;
            let _ = attr.parse_nested_meta(|meta| {
                if first {
                    if let Some(seg) = meta.path.segments.last() {
                        cfg.macros.insert(seg.ident.to_string());
                    }
                    first = false;
                }
                Ok(())
            });
        } else if p.is_ident("proc_macro") || p.is_ident("proc_macro_attribute") {
            if let syn::Item::Fn(f) = item {
                cfg.macros.insert(f.sig.ident.to_string());
            }
        } else if p.is_ident("derive") {
            let _ = attr.parse_nested_meta(|meta| {
                if let Some(root) = meta.path.segments.first() {
                    if cfg.is_target_crate(&root.ident.to_string()) {
                        if let Some(seg) = meta.path.segments.last() {
                            cfg.macros.insert(seg.ident.to_string());
                        }
                    }
                }
                Ok(())
            });
        }
    }
}

/// Trait names whose methods are language/std-provided plumbing rather than part
/// of a crate's own API.
fn is_std_trait_name(name: &str) -> bool {
    matches!(
        name,
        "Clone"
            | "Copy"
            | "Debug"
            | "Default"
            | "Drop"
            | "Display"
            | "Iterator"
            | "IntoIterator"
            | "DoubleEndedIterator"
            | "ExactSizeIterator"
            | "FusedIterator"
            | "Extend"
            | "FromIterator"
            | "Sum"
            | "Product"
            | "From"
            | "Into"
            | "TryFrom"
            | "TryInto"
            | "AsRef"
            | "AsMut"
            | "Borrow"
            | "BorrowMut"
            | "Deref"
            | "DerefMut"
            | "Hash"
            | "Hasher"
            | "PartialEq"
            | "Eq"
            | "PartialOrd"
            | "Ord"
            | "Index"
            | "IndexMut"
            | "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Rem"
            | "Neg"
            | "Not"
            | "BitAnd"
            | "BitOr"
            | "BitXor"
            | "Shl"
            | "Shr"
            | "Fn"
            | "FnMut"
            | "FnOnce"
            | "Error"
            | "Write"
            | "Read"
            | "Seek"
            | "Send"
            | "Sync"
            | "Sized"
            | "Unpin"
            | "Serialize"
            | "Deserialize"
            | "SerializeOwned"
            | "DeserializeOwned"
            | "Visitor"
    )
}

/// Extracts the base type identifier from an impl self type, e.g.
/// `Table<K, V, Stored<K>>` -> `Table`, `&Foo` -> `Foo`.
pub fn impl_base_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        syn::Type::Reference(r) => impl_base_name(&r.elem),
        syn::Type::Paren(p) => impl_base_name(&p.elem),
        syn::Type::Group(g) => impl_base_name(&g.elem),
        _ => None,
    }
}

/// Scans a `--target-src` root for its crate name.
pub fn add_target_src_crate_names(cfg: &mut TargetConfig, roots: &[PathBuf]) {
    for root in roots {
        let dir = if root.is_dir() {
            root.clone()
        } else {
            root.parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| root.clone())
        };
        if let Some((_, name)) = crate_name_for_dir(&dir) {
            if !cfg.crates.iter().any(|c| c == &name) {
                cfg.crates.push(name);
            }
        }
    }
}
