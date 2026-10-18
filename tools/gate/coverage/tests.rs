use super::*;

fn item(name: &str, ty: Option<&str>) -> Item {
    Item {
        name: name.into(),
        ty: ty.map(String::from),
        line: 1,
    }
}

fn file(path: &str, module: Option<&str>, items: Vec<Item>) -> SrcFile {
    SrcFile {
        path: path.into(),
        module: module.map(String::from),
        items,
    }
}

fn base() -> Inputs {
    Inputs {
        files: vec![file(
            "src/a.rs",
            Some("a"),
            vec![item("f", None), item("g", None), item("mul", Some("Cell"))],
        )],
        lean_defs: ["a.f", "a.g.body", "a.g", "a.Cell.mul"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        corpus_tokens: ["apeiron_word.a.f"].iter().map(|s| s.to_string()).collect(),
        waivers: vec![],
        debts: vec![],
    }
}

fn rules(i: &Inputs) -> Vec<&'static str> {
    verdict(i).0.iter().map(|x| x.rule).collect()
}

#[test]
fn covered_tree_is_accepted_and_counted() {
    let (ref_, t, _) = verdict(&base());
    assert!(ref_.is_empty(), "{ref_:?}");
    assert_eq!((t.fns, t.extracted, t.refined, t.gap), (3, 3, 1, 0));
}

#[test]
fn control_new_unextracted_fn_is_refused() {
    let mut i = base();
    i.files[0].items.push(item("h", None));
    assert_eq!(rules(&i), ["uncovered"]);
}

#[test]
fn control_new_file_with_no_module_is_refused() {
    let mut i = base();
    i.files.push(file("src/b.rs", None, vec![item("x", None)]));
    assert_eq!(rules(&i), ["uncovered"]);
}

#[test]
fn control_debt_ratchet_refuses_growth_and_stale() {
    let mut i = base();
    i.files[0].items.push(item("h", None));
    i.debts.push(Debt {
        path: "src/a.rs".into(),
        unextracted: 1,
        row: "R".into(),
    });
    assert!(rules(&i).is_empty());
    i.files[0].items.push(item("k", None));
    assert_eq!(rules(&i), ["debt-grew"]);
    i.files[0].items.truncate(3);
    assert_eq!(rules(&i), ["debt-stale"]);
}

#[test]
fn control_empty_lean_or_empty_tree_is_vacuous() {
    let mut i = base();
    i.lean_defs.clear();
    assert!(rules(&i).contains(&"vacuous"));
    let mut j = base();
    j.files.clear();
    assert!(rules(&j).contains(&"vacuous"));
    // names that match nothing: the matcher is broken, not the tree covered
    let mut k = base();
    k.files[0].module = Some("zzz".into());
    k.debts.push(Debt {
        path: "src/a.rs".into(),
        unextracted: 3,
        row: "R".into(),
    });
    assert!(rules(&k).contains(&"vacuous"));
}

fn waiver(name: &str, class: &str, reason: &str) -> Waiver {
    Waiver {
        path: "src/a.rs".into(),
        name: name.into(),
        class: class.into(),
        reason: reason.into(),
        twin: String::new(),
    }
}

#[test]
fn control_waiver_rules() {
    let mut i = base();
    i.files[0].items.push(item("h", None));
    i.waivers.push(waiver(
        "h",
        "ffi",
        "calls the Vulkan loader across the C ABI",
    ));
    assert!(rules(&i).is_empty());
    assert_eq!(verdict(&i).1.waived, 1);
    // stale: the waived fn is extracted
    let mut s = base();
    s.waivers.push(waiver(
        "f",
        "ffi",
        "calls the Vulkan loader across the C ABI",
    ));
    assert_eq!(rules(&s), ["waiver-stale"]);
    // no reason, unknown class, whole-file simd-twin
    let mut b = i_with_h();
    b.waivers.push(waiver("h", "tired", "x"));
    let r = rules(&b);
    assert!(r.contains(&"waiver-class") && r.contains(&"waiver-reason"));
    let mut w = i_with_h();
    w.waivers
        .push(waiver("*", "simd-twin", "avx2 rows of the scalar path"));
    assert!(rules(&w).contains(&"waiver-scope"));
    // simd-twin names an extracted twin, or is refused
    let mut t = i_with_h();
    let mut tw = waiver("h", "simd-twin", "avx2 rows of the scalar path");
    tw.twin = "nope".into();
    t.waivers.push(tw.clone());
    assert!(rules(&t).contains(&"waiver-twin"));
    tw.twin = "f".into();
    t.waivers = vec![tw];
    assert!(rules(&t).is_empty());
}

fn i_with_h() -> Inputs {
    let mut i = base();
    i.files[0].items.push(item("h", None));
    i
}

#[test]
fn scanner_skips_tests_nested_fns_strings_and_names_impl_types() {
    let src = r#"
pub fn a() { fn inner() {} let _ = "}{ fn fake() {}"; let _c = '{'; }
// fn commented() {}
impl<T: Copy> Foo<T> { pub fn m(&self) -> u8 { 0 } fn n() {} }
impl core::ops::Mul for Bar { fn mul(self) {} }
trait Tr { fn d(&self) {} fn decl(&self); }
#[cfg(test)]
mod tests { fn t() {} }
#[test]
fn solo() {}
mod inner { pub fn deep() {} }
"#;
    let it = scan_items(src).unwrap();
    let got: Vec<(String, Option<String>)> = it.into_iter().map(|i| (i.name, i.ty)).collect();
    let want = [
        ("a", None),
        ("m", Some("Foo")),
        ("n", Some("Foo")),
        ("mul", Some("Bar")),
        ("d", Some("Tr")),
        ("decl", Some("Tr")),
        ("deep", None),
    ];
    let want: Vec<(String, Option<String>)> = want
        .iter()
        .map(|(a, b)| (a.to_string(), b.map(String::from)))
        .collect();
    assert_eq!(got, want);
}

#[test]
fn control_unbalanced_source_is_an_error_not_a_count() {
    assert!(scan_items("fn a() {").is_err());
    assert!(scan_items("fn a() {} }").is_err());
}

#[test]
fn parsers() {
    assert_eq!(
        lean_def_name("@[reducible] def cell.Cell.mul (x : A)"),
        Some("cell.Cell.mul".into())
    );
    assert_eq!(
        lean_def_name("  partial def step.f : U8"),
        Some("step.f".into())
    );
    assert_eq!(lean_def_name("theorem x"), None);
    let defs = lean_def_names(
        "def a.f\n\ndef\n  coord.address.LocatorSigma.Insts.f.from_sigma\n\ndef b.g : U8\n",
    );
    assert!(defs.contains("a.f"));
    assert!(defs.contains("coord.address.LocatorSigma.Insts.f.from_sigma"));
    assert!(defs.contains("b.g"));
    assert_eq!(defs.len(), 3);
    let m = parse_path_mods("#[path = \"../../src/cell.rs\"]\npub mod cell;\n#[path = \"../../src/map/table.rs\"]\n#[allow(x)]\npub mod table;\n");
    assert_eq!(
        m,
        [
            ("src/cell.rs".into(), "cell".into()),
            ("src/map/table.rs".into(), "table".into())
        ]
    );
    let (w, d) = parse_rules("[[waiver]]\npath = \"src/a.rs\"\nfn = \"x\"\nclass = \"ffi\"\nreason = \"long enough reason\"\n[[debt]]\npath = \"src/a.rs\"\nunextracted = 2\nrow = \"R\"\n").unwrap();
    assert_eq!((w.len(), d.len(), d[0].unextracted), (1, 1, 2));
    assert!(parse_rules("path = \"x\"\n").is_err());
}

#[test]
fn child_modules_resolve_with_paths_and_skip_tests() {
    let m = parse_child_mods("#[path = \"table/keyrep.rs\"]\npub mod keyrep;\nmod probe;\n#[cfg(test)]\nmod tests;\nmod inline { }\n");
    assert_eq!(
        m,
        [
            (Some("table/keyrep.rs".into()), "keyrep".into(), false),
            (None, "probe".into(), false),
            (None, "tests".into(), true)
        ]
    );
    assert_eq!(normalise(Path::new("src/map/../a/./b.rs")), "src/a/b.rs");
}
