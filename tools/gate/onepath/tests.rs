//! Fixture controls: each rule refuses a seeded defect and accepts its rooted form.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

const EMPTY: &str = "schema = 1\n\n[forward]\nbodies = []\n";

/// A throwaway repository `<tmp>/repo` beside `<tmp>/indra/board/tickets/`.
fn fixture(files: &[(&str, &str)], contract: &str) -> Vec<String> {
    let base = std::env::temp_dir().join(format!(
        "onepath-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    ));
    let root: PathBuf = base.join("repo");
    let tickets = base.join("indra/board/tickets");
    std::fs::create_dir_all(&tickets).unwrap();
    std::fs::write(
        tickets.join("T-OPEN.md"),
        "```yaml ticket\nstate: open\n```\n",
    )
    .unwrap();
    std::fs::write(
        tickets.join("T-DONE.md"),
        "```yaml ticket\nstate: done\n```\n",
    )
    .unwrap();
    for (path, text) in files {
        let p = root.join(path);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }
    std::fs::create_dir_all(root.join("docs/contracts")).unwrap();
    std::fs::write(root.join("docs/contracts/one-path.toml"), contract).unwrap();
    let out = super::run(&root, false).map(|(_, o, errors)| {
        let mut r = o.refusals;
        r.extend(errors.into_iter().map(|e| format!("unparsed: {e}")));
        r
    });
    std::fs::remove_dir_all(&base).unwrap();
    out.unwrap()
}

fn has(refusals: &[String], needle: &str) -> bool {
    refusals.iter().any(|r| r.contains(needle))
}

const KERN_LIB: &str = "pub mod engine;\n";
const ENGINE: &str = "pub mod kern;\n";
const KERN: &str = "pub fn dot(a: &[i32], b: &[i32]) -> i32 {\n    let mut s = 0;\n    for i in 0..a.len() {\n        s += a[i] * b[i];\n    }\n    s\n}\n";

#[test]
fn product_in_kernel_is_accepted() {
    let r = fixture(
        &[
            ("src/lib.rs", KERN_LIB),
            ("src/engine/mod.rs", ENGINE),
            ("src/engine/kern/mod.rs", KERN),
            (
                "tests/t.rs",
                "#[test]\nfn t() {\n    apeiron::engine::kern::dot(&[1], &[2]);\n}\n",
            ),
        ],
        EMPTY,
    );
    assert!(r.is_empty(), "{r:?}");
}

#[test]
fn product_outside_kernel_is_refused() {
    let r = fixture(
        &[
            ("src/lib.rs", "pub mod learn;\n"),
            ("src/learn/mod.rs", KERN),
            (
                "tests/t.rs",
                "#[test]\nfn t() {\n    apeiron::learn::dot(&[1], &[2]);\n}\n",
            ),
        ],
        EMPTY,
    );
    assert!(has(&r, "new product"), "{r:?}");
}

#[test]
fn renamed_iterator_product_is_refused() {
    let src = "pub fn score(xs: &[f32], ws: &[f32]) -> f32 {\n    xs.iter().zip(ws).map(|(x, w)| x * w).sum()\n}\n";
    let r = fixture(
        &[
            ("src/lib.rs", src),
            (
                "tests/t.rs",
                "#[test]\nfn t() {\n    apeiron::score(&[1.0], &[2.0]);\n}\n",
            ),
        ],
        EMPTY,
    );
    assert!(has(&r, "new product"), "{r:?}");
}

#[test]
fn scalar_multiplier_product_is_refused() {
    let src = "fn mulq(a: i64, b: i64) -> i64 {\n    (a * b) >> 32\n}\npub fn acc(x: &[i64], w: &[i64]) -> i64 {\n    let mut s = 0i64;\n    for (a, b) in x.iter().zip(w) {\n        s = s.wrapping_add(mulq(*a, *b));\n    }\n    s\n}\n";
    let r = fixture(
        &[
            ("src/lib.rs", src),
            (
                "tests/t.rs",
                "#[test]\nfn t() {\n    apeiron::acc(&[1], &[2]);\n}\n",
            ),
        ],
        EMPTY,
    );
    assert!(has(&r, "new product"), "{r:?}");
}

#[test]
fn descent_step_is_refused() {
    let src = "pub fn step(p: &mut [f32], g: &[f32], m: &[f32], v: &mut [f32], lr: f32) {\n    for i in 0..p.len() {\n        let vi = 0.9 * v[i] + 0.1 * g[i] * g[i];\n        v[i] = vi;\n        let denom = vi.sqrt() + 1e-8;\n        let upd = lr * m[i] / denom;\n        p[i] = p[i] - upd;\n    }\n}\n";
    let r = fixture(
        &[
            ("src/lib.rs", src),
            ("tests/t.rs", "#[test]\nfn t() {\n    apeiron::step(&mut [1.0], &[1.0], &[1.0], &mut [1.0], 0.1);\n}\n"),
        ],
        EMPTY,
    );
    assert!(has(&r, "new descent"), "{r:?}");
}

#[test]
fn rotation_root_is_not_descent() {
    // A Jacobi rotation divides by the root of a fresh scalar, not of a stored statistic.
    let src = "pub fn rot(a: &mut [f32], p: usize, q: usize) {\n    let th = (a[q] - a[p]) / (2.0 * a[p + q]);\n    let t = 1.0 / (th.abs() + (1.0 + th * th).sqrt());\n    a[p] = a[p] - t * a[p + q];\n}\n";
    let r = fixture(
        &[
            ("src/lib.rs", src),
            (
                "tests/t.rs",
                "#[test]\nfn t() {\n    apeiron::rot(&mut [1.0; 4], 0, 1);\n}\n",
            ),
        ],
        EMPTY,
    );
    assert!(!has(&r, "descent"), "{r:?}");
}

#[test]
fn env_switch_is_refused() {
    let src = "pub fn pick() -> bool {\n    std::env::var(\"X\").is_ok()\n}\n";
    let r = fixture(
        &[
            ("src/lib.rs", src),
            ("tests/t.rs", "#[test]\nfn t() {\n    apeiron::pick();\n}\n"),
        ],
        EMPTY,
    );
    assert!(has(&r, "new switch"), "{r:?}");
}

#[test]
fn orphan_is_refused() {
    let r = fixture(
        &[("src/lib.rs", "pub fn lonely() -> u8 {\n    1\n}\n")],
        EMPTY,
    );
    assert!(has(&r, "new orphan"), "{r:?}");
}

#[test]
fn extraction_roots_only_what_it_names() {
    // A `#[path]` include roots the fns the extraction's wrappers name, not the whole file.
    let ext =
        "#[path = \"../../src/g.rs\"]\npub mod g;\npub fn word_kept() -> u8 {\n    g::kept()\n}\n";
    let lib = "pub mod g;\n";
    let named = "pub fn kept() -> u8 {\n    1\n}\n";
    let r = fixture(
        &[
            ("src/lib.rs", lib),
            ("src/g.rs", named),
            ("proof/extract/lib.rs", ext),
        ],
        EMPTY,
    );
    assert!(r.is_empty(), "{r:?}");
    let unnamed = "pub fn kept() -> u8 {\n    1\n}\npub fn unnamed() -> u8 {\n    2\n}\n";
    let r = fixture(
        &[
            ("src/lib.rs", lib),
            ("src/g.rs", unnamed),
            ("proof/extract/lib.rs", ext),
        ],
        EMPTY,
    );
    assert!(
        has(&r, "new orphan") && has(&r, "src/g.rs:unnamed"),
        "{r:?}"
    );
    // A port file the extraction includes from its own tree names its fns too.
    let ext = format!("{ext}#[path = \"port.rs\"]\npub mod port;\n");
    let port = "pub fn word_unnamed() -> u8 {\n    crate::g::unnamed()\n}\n";
    let r = fixture(
        &[
            ("src/lib.rs", lib),
            ("src/g.rs", unnamed),
            ("proof/extract/lib.rs", &ext),
            ("proof/extract/port.rs", port),
        ],
        EMPTY,
    );
    assert!(r.is_empty(), "{r:?}");
}

#[test]
fn generic_dispatch_reaches_every_impl() {
    // `R::make(..)` and `<R as Make>::make(..)` on a type parameter reach each impl's `make`;
    // each form alone roots both impls, and with no generic caller both are orphans.
    let make = "pub trait Make {\n    fn make(v: i64) -> Self;\n}\nimpl Make for i64 {\n    fn make(v: i64) -> Self {\n        v\n    }\n}\nimpl Make for i8 {\n    fn make(v: i64) -> Self {\n        v as i8\n    }\n}\n";
    let t = "#[test]\nfn t() {\n    let _: i64 = apeiron::a(1);\n}\n";
    for call in ["R::make(v)", "<R as Make>::make(v)"] {
        let lib = format!("{make}pub fn a<R: Make>(v: i64) -> R {{\n    {call}\n}}\n");
        let r = fixture(&[("src/lib.rs", &lib), ("tests/t.rs", t)], EMPTY);
        assert!(r.is_empty(), "{call}: {r:?}");
    }
    let t = "#[test]\nfn t() {\n    let _: i64 = 1;\n}\n";
    let r = fixture(&[("src/lib.rs", make), ("tests/t.rs", t)], EMPTY);
    assert!(
        r.iter().any(|x| x.contains("new orphan") && x.contains("make")),
        "{r:?}"
    );
}

#[test]
fn const_initializer_reaches_its_callees() {
    // `const T: u8 = build();` is a node: a use of `T` reaches `build`; an unused `T` and its
    // `build` are orphans.
    let lib = "const T: u8 = build();\nconst fn build() -> u8 {\n    1\n}\npub fn used() -> u8 {\n    T\n}\n";
    let t = "#[test]\nfn t() {\n    apeiron::used();\n}\n";
    let r = fixture(&[("src/lib.rs", lib), ("tests/t.rs", t)], EMPTY);
    assert!(r.is_empty(), "{r:?}");
    let t = "#[test]\nfn t() {\n    let _ = 1;\n}\n";
    let lib = lib.replace("    T\n", "    2\n");
    let r = fixture(&[("src/lib.rs", &lib), ("tests/t.rs", t)], EMPTY);
    assert!(
        has(&r, "src/lib.rs:T") && has(&r, "src/lib.rs:build"),
        "{r:?}"
    );
}

#[test]
fn const_reached_from_types_patterns_and_macro_rules() {
    // A const is used when a type evaluates it (a struct field, a const's own type, a `let`
    // ascription, a signature), when a pattern names it, or when a body-local `macro_rules!`
    // names it. Each form alone; the control is the same const with no use.
    let t = "#[test]\nfn t() {\n    apeiron::used(0);\n}\n";
    let uses = [
        "pub struct S {\n    pub a: [u8; N],\n}\npub fn used(_: u8) {}\n",
        "pub const M: [u8; N] = [0; 4];\npub fn used(_: u8) -> u8 {\n    M[0]\n}\n",
        "pub fn used(_: u8) {\n    let _a: [u8; N] = [0; 4];\n}\n",
        "pub fn used(x: [u8; N]) -> u8 {\n    x[0]\n}\n",
        "pub fn used(x: usize) -> bool {\n    match x {\n        N => true,\n        _ => false,\n    }\n}\n",
        "pub fn used(_: u8) -> usize {\n    macro_rules! m {\n        ($e:expr) => {\n            $e + N\n        };\n    }\n    m!(1)\n}\n",
    ];
    for u in uses {
        let lib = format!("const N: usize = 4;\n{u}");
        let t = if u.contains("x: [u8") {
            "#[test]\nfn t() {\n    apeiron::used([0; 4]);\n}\n"
        } else {
            t
        };
        let r = fixture(&[("src/lib.rs", &lib), ("tests/t.rs", t)], EMPTY);
        assert!(!has(&r, "src/lib.rs:N"), "{u}\n{r:?}");
    }
    let lib = "const N: usize = 4;\npub fn used(_: u8) {}\n";
    let r = fixture(&[("src/lib.rs", lib), ("tests/t.rs", t)], EMPTY);
    assert!(has(&r, "src/lib.rs:N"), "{r:?}");
}

#[test]
fn const_tables_clone_only_on_equal_values() {
    let table = |name: &str, base: u32| {
        let vals: Vec<String> = (base..base + 24).map(|v| v.to_string()).collect();
        format!("pub const {name}: [u32; 24] = [{}];\n", vals.join(", "))
    };
    let r = fixture(&[("src/lib.rs", &(table("A", 0) + &table("B", 100)))], EMPTY);
    assert!(!has(&r, "new clone"), "{r:?}");
    let r = fixture(&[("src/lib.rs", &(table("A", 0) + &table("B", 0)))], EMPTY);
    assert!(has(&r, "new clone"), "{r:?}");
}

#[test]
fn included_file_is_reached() {
    // `include!` joins the file's items to the including module: neither dead nor orphaned.
    let files = [
        ("src/lib.rs", "include!(\"part.rs\");\n"),
        ("src/part.rs", "pub fn used() -> u8 {\n    1\n}\n"),
        ("tests/t.rs", "#[test]\nfn t() {\n    apeiron::used();\n}\n"),
    ];
    let r = fixture(&files, EMPTY);
    assert!(r.is_empty(), "{r:?}");
    let r = fixture(
        &[
            files[0],
            ("src/part.rs", "pub fn used() -> u8 {\n    1\n}\n"),
            ("src/gone.rs", "pub fn g() {}\n"),
            files[2],
        ],
        EMPTY,
    );
    assert!(has(&r, "new dead-file"), "{r:?}");
}

#[test]
fn orphan_row_carries_the_file_set() {
    let row = "schema = 1\n\n[forward]\nbodies = []\n\n[[debt]]\nrule = \"orphan\"\nitems = [\"src/lib.rs:lonely\"]\nticket = \"T-OPEN\"\n";
    let one = fixture(
        &[("src/lib.rs", "pub fn lonely() -> u8 {\n    1\n}\n")],
        row,
    );
    assert!(one.is_empty(), "{one:?}");
    // A second orphan in the same file changes the set: a new finding and a stale row.
    let two = fixture(
        &[(
            "src/lib.rs",
            "pub fn lonely() -> u8 {\n    1\n}\npub fn other() -> u8 {\n    2\n}\n",
        )],
        row,
    );
    assert!(
        has(&two, "new orphan") && has(&two, "stale debt"),
        "{two:?}"
    );
}

const FWD_LIB: &str = "pub mod engine;\npub fn forward(x: &[i32], w: &[i32]) -> f32 {\n    let d = engine::kern::dot(x, w) as f32;\n    d.exp() / d.sqrt()\n}\n";

fn fwd_files(extra_lib: &str) -> Vec<(&'static str, String)> {
    vec![
        ("src/lib.rs", format!("{FWD_LIB}{extra_lib}")),
        ("src/engine/mod.rs", ENGINE.to_string()),
        ("src/engine/kern/mod.rs", KERN.to_string()),
        (
            "tests/t.rs",
            "#[test]\nfn t() {\n    apeiron::forward(&[1], &[2]);\n    apeiron::second(&[1], &[2]);\n}\n"
                .to_string(),
        ),
    ]
}

fn run_fwd(extra_lib: &str, contract: &str) -> Vec<String> {
    let files = fwd_files(extra_lib);
    let refs: Vec<(&str, &str)> = files.iter().map(|(p, t)| (*p, t.as_str())).collect();
    fixture(&refs, contract)
}

const DECLARED: &str = "schema = 1\n\n[forward]\nbodies = [\"src/lib.rs:forward\"]\n";
const SECOND: &str = "pub fn second(x: &[i32], w: &[i32]) -> f32 {\n    let d = engine::kern::dot(w, x) as f32;\n    d.sqrt() * d.exp()\n}\n";

#[test]
fn declared_forward_is_accepted_and_a_second_is_refused() {
    let one = run_fwd(
        "pub fn second(x: &[i32], w: &[i32]) -> f32 {\n    forward(x, w)\n}\n",
        DECLARED,
    );
    assert!(one.is_empty(), "{one:?}");
    let two = run_fwd(SECOND, DECLARED);
    assert!(has(&two, "new forward"), "{two:?}");
    let undeclared = run_fwd(SECOND, EMPTY);
    assert!(
        has(&undeclared, "new forward at src/lib.rs"),
        "{undeclared:?}"
    );
}

#[test]
fn split_forward_moves_the_finding() {
    // The second forward hidden behind helpers is still a forward at the fn that joins them.
    let split = "fn a(x: &[i32], w: &[i32]) -> f32 {\n    engine::kern::dot(x, w) as f32\n}\nfn b(d: f32) -> f32 {\n    d.exp()\n}\nfn c(d: f32) -> f32 {\n    d.sqrt()\n}\npub fn second(x: &[i32], w: &[i32]) -> f32 {\n    let d = a(x, w);\n    b(d) + c(d)\n}\n";
    let r = run_fwd(split, DECLARED);
    assert!(has(&r, "new forward") && has(&r, "second"), "{r:?}");
}

#[test]
fn debt_rows_ratchet() {
    let row = |consumers: usize, ticket: &str| {
        format!(
            "{EMPTY}\n[[debt]]\nrule = \"forward\"\nitem = \"src/lib.rs:second\"\nconsumers = {consumers}\nticket = \"{ticket}\"\n"
        )
    };
    let declared_second = |c: &str| c.replace("bodies = []", "bodies = [\"src/lib.rs:forward\"]");
    // One consumer (the test fn): the row carries it.
    let ok = run_fwd(SECOND, &declared_second(&row(1, "T-OPEN")));
    assert!(ok.is_empty(), "{ok:?}");
    // A moved count, a done ticket, a missing ticket.
    assert!(has(
        &run_fwd(SECOND, &declared_second(&row(2, "T-OPEN"))),
        "consumers moved"
    ));
    assert!(has(
        &run_fwd(SECOND, &declared_second(&row(1, "T-DONE"))),
        "is done"
    ));
    assert!(has(
        &run_fwd(SECOND, &declared_second(&row(1, "T-NONE"))),
        "has no file"
    ));
    // A row whose finding is gone is stale.
    let gone = run_fwd(
        "pub fn second(x: &[i32], w: &[i32]) -> f32 {\n    forward(x, w)\n}\n",
        &declared_second(&row(1, "T-OPEN")),
    );
    assert!(has(&gone, "stale debt"), "{gone:?}");
}

#[test]
fn clone_is_refused() {
    let body = "    let mut out = Vec::new();\n    for (i, v) in xs.iter().enumerate() {\n        if *v > 3 {\n            out.push(i + 1);\n        } else {\n            out.push(i * 2);\n        }\n    }\n    out.sort();\n    out.dedup();\n    out\n";
    let src = format!(
        "pub fn one(xs: &[u8]) -> Vec<usize> {{\n{body}}}\npub fn two(ys: &[u8]) -> Vec<usize> {{\n{}}}\n",
        body.replace("xs", "ys").replace("out", "acc")
    );
    let r = fixture(
        &[
            ("src/lib.rs", &src),
            (
                "tests/t.rs",
                "#[test]\nfn t() {\n    apeiron::one(&[1]);\n    apeiron::two(&[1]);\n}\n",
            ),
        ],
        EMPTY,
    );
    assert!(has(&r, "new clone"), "{r:?}");
}

#[test]
fn unparsed_file_is_refused() {
    let r = fixture(
        &[
            ("src/lib.rs", "pub fn broken( {\n"),
            ("tests/t.rs", "#[test]\nfn t() {}\n"),
        ],
        EMPTY,
    );
    assert!(has(&r, "unparsed"), "{r:?}");
}
