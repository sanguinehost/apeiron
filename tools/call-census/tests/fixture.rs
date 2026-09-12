//! Fixture test for the census. Exercises comments, string literals, import
//! aliases, grouped and glob imports, re-exports, two calls on one line, a
//! non-target `.get`, a dynamic-dispatch `.get`, a local type alias, Arc
//! shadowing, a local type named `Table`, `&mut Table`, enum/tuple-struct
//! constructors, trait-provided builders, cfg propagation, a derive macro and a
//! `macro_rules!` body that is not expanded.

use call_census::{run, Config};
use serde_json::Value;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn load(path: &std::path::Path) -> Vec<Value> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn load_obj(path: &std::path::Path) -> Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn calls_in<'a>(records: &'a [Value], caller: &str) -> Vec<&'a Value> {
    records
        .iter()
        .filter(|r| r["caller"]["name"] == caller)
        .collect()
}

fn attributed(v: &Value) -> bool {
    matches!(v["confidence"].as_str(), Some("high") | Some("medium"))
}

#[test]
fn census_fixture_is_lexically_precise() {
    let root = fixture("consumer");
    let target = fixture("hashtrinity");
    let out = std::env::temp_dir().join(format!("call-census-fixture-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);

    let mut cfg = Config::new(root, out.clone());
    cfg.target_src = vec![target];
    let summary = run(&cfg).expect("census run");

    assert!(summary.files_total >= 1, "should find the consumer file");
    assert_eq!(summary.files_parse_error, 0, "fixture must parse cleanly");

    let records = load(&out.join("records.json"));
    let imports = load(&out.join("imports.json"));
    let macros = load(&out.join("macros.json"));
    let inventory = load_obj(&out.join("inventory.json"));
    let meta = load_obj(&out.join("meta.json"));

    // ── schema / version / fingerprint ────────────────────────────────────
    assert_eq!(meta["schema_version"], 2);
    assert_eq!(meta["tool_version"], "0.2.0");
    assert!(
        meta["tool_source_sha256"]
            .as_str()
            .map(|s| s.len())
            .unwrap_or(0)
            == 64,
        "tool source fingerprint must be present"
    );

    // ── stable identifiers must be unique across the census ───────────────
    {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for r in records.iter().chain(imports.iter()).chain(macros.iter()) {
            let id = r["id"].as_str().unwrap().to_string();
            assert!(seen.insert(id.clone()), "duplicate id {id}");
        }
    }

    // ── comments and strings are not calls ────────────────────────────────
    for r in &records {
        let line = r["span"]["line"].as_u64().unwrap();
        assert!(
            line > 5,
            "comment line {line} produced a record: {}",
            r["callee_text"]
        );
    }

    // ── direct qualified call ─────────────────────────────────────────────
    assert!(
        records
            .iter()
            .any(|r| r["callee_path"] == "hashtrinity::crypto::blake3"
                || r["target_symbol"] == "crypto::blake3"),
        "expected the qualified blake3 call"
    );

    // ── attributed method call on an annotated Table ──────────────────────
    let direct_gets: Vec<&Value> = calls_in(&records, "direct")
        .into_iter()
        .filter(|r| r["method"] == "get")
        .collect();
    assert_eq!(direct_gets.len(), 1, "direct should have exactly one .get");
    assert_eq!(direct_gets[0]["category"], "method_attributed");
    assert_eq!(direct_gets[0]["confidence"], "high");
    assert_eq!(direct_gets[0]["target_type"], "Table");

    // ── alias import ──────────────────────────────────────────────────────
    let alias_gets = calls_in(&records, "alias");
    let alias_get = alias_gets
        .iter()
        .find(|r| r["method"] == "get")
        .expect("alias .get");
    assert_eq!(alias_get["category"], "method_attributed");
    assert_eq!(alias_get["target_type"], "Table");
    assert!(
        alias_get["attribution_basis"]
            .as_str()
            .unwrap()
            .contains("Ht"),
        "basis should name the alias"
    );

    // ── two calls on one line, distinct stable ids ────────────────────────
    let two = calls_in(&records, "two_on_one_line");
    assert_eq!(two.len(), 2, "two .get calls on one line");
    assert_ne!(two[0]["id"], two[1]["id"], "ids must be distinct");
    assert_eq!(two[0]["span"]["line"], two[1]["span"]["line"]);

    // ── `&mut Table` must still attribute ─────────────────────────────────
    let mut_ref = calls_in(&records, "mut_ref_target");
    assert!(
        mut_ref
            .iter()
            .any(|r| r["method"] == "get" && attributed(r)),
        "&mut Table receiver must attribute"
    );

    // ── a known non-target .get is not a candidate ────────────────────────
    assert!(
        calls_in(&records, "non_target").is_empty(),
        "HashMap::get must not be attributed or recorded as a candidate"
    );
    assert!(
        summary
            .other_method_calls_by_name
            .get("get")
            .copied()
            .unwrap_or(0)
            >= 1,
        "non-target .get must still be aggregated"
    );

    // ── unknown receiver: candidate, not attribution ──────────────────────
    let unresolved = calls_in(&records, "unresolved");
    let cand = unresolved
        .iter()
        .find(|r| r["method"] == "get")
        .expect("candidate .get");
    assert_eq!(cand["category"], "method_candidate");
    assert_eq!(cand["confidence"], "unattributed");
    assert_eq!(cand["candidate"], true);
    assert!(
        cand["target_type"].is_null(),
        "candidate must not be attributed"
    );
    assert!(
        unresolved.iter().all(|r| r["method"] != "weird_thing"),
        "non-target method name should only be aggregated"
    );

    // ── dynamic dispatch ──────────────────────────────────────────────────
    let dyn_rec = calls_in(&records, "dynamic");
    let d = dyn_rec
        .iter()
        .find(|r| r["method"] == "get")
        .expect("dyn .get");
    assert_eq!(d["category"], "method_dynamic_dispatch");
    assert_eq!(d["dynamic_dispatch"], true);

    // ── local alias to a target type is attributed, labelled as an alias ──
    let alias_local = calls_in(&records, "alias_local");
    let a = alias_local
        .iter()
        .find(|r| r["method"] == "get")
        .expect("alias_local .get");
    assert_eq!(a["category"], "method_attributed");
    assert_eq!(a["target_type"], "Table");
    assert!(
        a["attribution_basis"]
            .as_str()
            .unwrap()
            .contains("local type alias"),
        "alias attribution must say it is an alias"
    );

    // ── Arc shadowing must invalidate the stale Table binding ─────────────
    assert!(
        calls_in(&records, "arc_shadow").is_empty(),
        "let devices = Arc::new(devices) must not keep the stale Table type"
    );

    // ── a local type named Table must not be read as the target Table ─────
    assert!(
        calls_in(&records, "local_table").is_empty(),
        "local Table::len must not be attributed to hashtrinity::Table"
    );

    // ── enum / tuple-struct constructors are separated from functions ─────
    let ctor_calls: Vec<&Value> = records
        .iter()
        .filter(|r| r["caller"]["name"] == "constructors")
        .collect();
    let bu = ctor_calls
        .iter()
        .find(|r| r["callee_path"] == "SyntaxTerm::Bu")
        .expect("SyntaxTerm::Bu call");
    assert_eq!(bu["operation_kind"], "constructor");
    let wrapper = ctor_calls
        .iter()
        .find(|r| r["callee_path"] == "hashtrinity::Wrapper")
        .expect("Wrapper call");
    assert_eq!(wrapper["operation_kind"], "constructor");
    let assoc = ctor_calls
        .iter()
        .find(|r| r["callee_path"] == "Table::new")
        .expect("Table::new call");
    assert_eq!(assoc["operation_kind"], "associated_fn");

    // ── FoldExt provided methods are inventoried and heuristically labelled ─
    let methods: Vec<String> = inventory["methods"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    for m in ["index_by", "set_by", "count_by"] {
        assert!(methods.contains(&m.to_string()), "inventory missing {m}");
    }
    // Test-only target items must not enter the inventory.
    let types: Vec<String> = inventory["types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(
        !types.contains(&"Engine".to_string()),
        "cfg(test) struct Engine leaked into the inventory"
    );
    assert!(
        !methods.contains(&"test_only_method".to_string()),
        "cfg(test) impl method leaked into the inventory"
    );
    let builder_recs = calls_in(&records, "builders");
    let idx = builder_recs
        .iter()
        .find(|r| r["method"] == "index_by")
        .expect("index_by call");
    assert_eq!(idx["category"], "method_trait_heuristic");
    assert_eq!(idx["confidence"], "heuristic");
    assert_eq!(idx["target_symbol"], "FoldExt::index_by");
    assert!(
        summary.heuristic_method_calls >= 2,
        "FoldExt builders must be counted as labelled heuristics"
    );

    // ── cfg propagation ───────────────────────────────────────────────────
    let gated = calls_in(&records, "cfg_gated");
    assert!(!gated.is_empty(), "cfg-gated fn body should be censused");
    assert!(
        gated[0]["cfg"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c.as_str().unwrap().contains("feature")),
        "caller cfg must include the feature predicate"
    );
    assert_eq!(gated[0]["is_test_context"], false);

    // module-level cfg
    let modgated = calls_in(&records, "from_gated_module");
    assert!(!modgated.is_empty(), "gated module body should be censused");
    assert!(
        modgated[0]["cfg"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c.as_str().unwrap().contains("modgated")),
        "module cfg must propagate to calls in the module"
    );

    // impl-level cfg
    let implgated = calls_in(&records, "from_gated_impl");
    let ig = implgated
        .iter()
        .find(|r| r["method"] == "get")
        .expect("gated impl .get");
    assert!(
        ig["cfg"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c.as_str().unwrap().contains("implgated")),
        "impl cfg must propagate to calls in the impl"
    );

    // ── cfg(test) context ─────────────────────────────────────────────────
    let unit = calls_in(&records, "unit");
    assert!(!unit.is_empty(), "unit test body should be censused");
    assert_eq!(unit[0]["is_test_context"], true);
    assert_eq!(unit[0]["source_location"], "src");
    assert_eq!(unit[0]["reachability"], "not_determined");

    // ── imports: alias, glob, re-export, grouped ──────────────────────────
    assert!(
        imports
            .iter()
            .any(|i| i["alias"] == "Ht" && i["path"] == "hashtrinity::Table"),
        "alias import missing"
    );
    assert!(
        imports
            .iter()
            .any(|i| i["is_glob"] == true && i["root"] == "hashtrinity"),
        "glob import missing"
    );
    assert!(
        imports
            .iter()
            .any(|i| i["is_reexport"] == true && i["path"] == "hashtrinity::CoordMap"),
        "re-export missing"
    );
    assert!(
        imports
            .iter()
            .any(|i| i["path"] == "hashtrinity::codec::syntax::AsTerm"),
        "grouped import missing"
    );

    // ── macros: derive and macro_rules are recorded unexpanded ────────────
    assert!(
        macros
            .iter()
            .any(|m| m["kind"] == "derive" && m["name"] == "Term" && m["unexpanded"] == true),
        "target derive missing"
    );
    assert!(
        macros
            .iter()
            .any(|m| m["kind"] == "macro_rules_def" && m["name"] == "my_macro"),
        "macro_rules definition missing"
    );
    assert!(
        summary
            .unqualified_macro_by_name
            .get("my_macro")
            .copied()
            .unwrap_or(0)
            >= 1,
        "macro invocation should be counted as unexpanded"
    );

    // ── summary sanity ────────────────────────────────────────────────────
    assert!(
        summary.attributed_calls >= 4,
        "expected several attributed calls"
    );
    assert!(summary.target_imports >= 3, "expected target imports");
    assert!(summary.target_reexports >= 1, "expected a target re-export");

    let _ = std::fs::remove_dir_all(&out);
}
