//! Fixture controls: each rule refuses a seeded defect and accepts its rooted form.

use std::path::{Path, PathBuf};
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
        r.iter()
            .any(|x| x.contains("new orphan") && x.contains("make")),
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
    let r = fixture(
        &[("src/lib.rs", &(table("A", 0) + &table("B", 100)))],
        EMPTY,
    );
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

// ---- the GPU half: `.comp`/`.spv`, the `[[kernel]]` role map and the ratchet ---------------------

const LIB: &str = "pub fn t() {}\n";
const T: &str = "#[test]\nfn t() {\n    apeiron::t();\n}\n";
/// A reduction over `k`: a product.
const MAC: &str =
    "void main() {\n    int s = 0;\n    for (int k = 0; k < 4; ++k) {\n        s += a[k] * b[k];\n    }\n}\n";
const COOP: &str = "void main() {\n    acc = coopMatMulAdd(a, b, acc);\n}\n";
const FMA: &str = "void main() {\n    float acc = 0.0;\n    for (uint i = 0u; i < n; ++i) {\n        acc = fma(a[i], w[i], acc);\n    }\n}\n";
/// A multiply-accumulate into the loop variable is a state update, not a reduction.
const UPDATE: &str =
    "void main() {\n    for (int i = 0; i < 4; ++i) {\n        m[i] += d * s[i];\n    }\n}\n";

fn row(role: &str, comp: &str, product: bool) -> String {
    let face = comp.replace(".comp", ".spv");
    format!(
        "[[kernel]]\nrole = \"{role}\"\ncomp = \"{comp}\"\nfaces = [\"{face}\"]\nproduct = {product}\n"
    )
}

fn files(comp: &'static str, text: &str) -> Vec<(&'static str, String)> {
    vec![
        ("src/lib.rs", LIB.to_string()),
        ("tests/t.rs", T.to_string()),
        ("src/gpu/spirv/a.spv", String::new()),
        (comp, text.to_string()),
    ]
}

fn shader_run(comp: &'static str, text: &str, contract: &str) -> Vec<String> {
    let f = files(comp, text);
    let refs: Vec<(&str, &str)> = f.iter().map(|(p, t)| (*p, t.as_str())).collect();
    fixture(&refs, contract)
}

#[test]
fn shader_without_a_kernel_row_is_refused() {
    let r = shader_run("src/gpu/spirv/a.comp", UPDATE, EMPTY);
    assert!(has(&r, "new shader at src/gpu/spirv/a.comp"), "{r:?}");
    assert!(has(&r, "new shader at src/gpu/spirv/a.spv"), "{r:?}");
    // The state update is not a reduction, so this comp is not a product.
    assert!(!has(&r, "new shader-product"), "{r:?}");
}

#[test]
fn undeclared_face_is_refused() {
    let two = vec![
        ("src/lib.rs", LIB),
        ("tests/t.rs", T),
        ("src/gpu/spirv/a.spv", ""),
        ("src/gpu/spirv/a.comp", UPDATE),
        ("src/gpu/spirv/b.spv", ""),
    ];
    let contract = format!("{EMPTY}\n{}", row("x", "src/gpu/spirv/a.comp", false));
    let r = fixture(&two, &contract);
    assert!(has(&r, "new shader at src/gpu/spirv/b.spv"), "{r:?}");
    assert!(!has(&r, "new shader at src/gpu/spirv/a"), "{r:?}");
}

#[test]
fn declared_shader_and_face_are_accepted() {
    let r = shader_run(
        "src/gpu/spirv/a.comp",
        COOP,
        &format!("{EMPTY}\n{}", row("gemm", "src/gpu/spirv/a.comp", true)),
    );
    assert!(r.is_empty(), "{r:?}");
}

#[test]
fn coopmat_in_a_non_product_row_is_refused() {
    let r = shader_run(
        "src/gpu/spirv/a.comp",
        COOP,
        &format!("{EMPTY}\n{}", row("gemm", "src/gpu/spirv/a.comp", false)),
    );
    assert!(
        has(&r, "new shader-product at src/gpu/spirv/a.comp"),
        "{r:?}"
    );
    assert!(has(&r, "coopMatMulAdd"), "{r:?}");
}

#[test]
fn fma_loop_in_a_non_product_row_is_refused() {
    let r = shader_run(
        "src/gpu/spirv/a.comp",
        FMA,
        &format!("{EMPTY}\n{}", row("lin", "src/gpu/spirv/a.comp", false)),
    );
    assert!(
        has(&r, "new shader-product at src/gpu/spirv/a.comp"),
        "{r:?}"
    );
    assert!(has(&r, "fma in a for body"), "{r:?}");
}

#[test]
fn a_product_debt_row_carries_the_shader_product_finding() {
    // The debt row is the only way a product shader the engine does not hold is accepted.
    let two = vec![
        ("src/lib.rs", LIB),
        ("tests/t.rs", T),
        ("src/gpu/spirv/a.spv", ""),
        ("src/gpu/spirv/a.comp", MAC),
    ];
    let contract = format!(
        "{EMPTY}\n{}\n[[debt]]\nrule = \"shader-product\"\nitem = \"src/gpu/spirv/a.comp\"\nconsumers = 0\nticket = \"T-OPEN\"\n",
        row("lin", "src/gpu/spirv/a.comp", false)
    );
    let r = fixture(&two, &contract);
    assert!(r.is_empty(), "{r:?}");
    // The same row with a count the source does not have is refused.
    let moved = contract.replace("consumers = 0", "consumers = 3");
    let r = fixture(&two, &moved);
    assert!(has(&r, "consumers moved"), "{r:?}");
}

#[test]
fn duplicate_role_and_stale_rows_are_refused() {
    let two = vec![
        ("src/lib.rs", LIB),
        ("tests/t.rs", T),
        ("src/gpu/spirv/a.spv", ""),
        ("src/gpu/spirv/b.spv", ""),
        ("src/gpu/spirv/a.comp", UPDATE),
        ("src/gpu/spirv/b.comp", UPDATE),
    ];
    let dup = format!(
        "{EMPTY}\n{}{}",
        row("gemm", "src/gpu/spirv/a.comp", false),
        row("gemm", "src/gpu/spirv/b.comp", false)
    );
    let r = fixture(&two, &dup);
    assert!(has(&r, "duplicate kernel role `gemm`"), "{r:?}");
    let stale = format!("{EMPTY}\n{}", row("gemm", "src/gpu/spirv/gone.comp", false));
    let r = fixture(&two, &stale);
    assert!(
        has(&r, "stale kernel row `src/gpu/spirv/gone.comp`"),
        "{r:?}"
    );
    let face = format!("{EMPTY}\n{}", row("gemm", "src/gpu/spirv/a.comp", false))
        .replace("a.spv", "gone.spv");
    let r = fixture(&two, &face);
    assert!(has(&r, "stale face `src/gpu/spirv/gone.spv`"), "{r:?}");
}

// The ratchet reads `HEAD`, so its fixtures are git repositories.
fn git_repo(files: &[(&str, &str)], contract: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "onepath-git-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    ));
    let root = base.join("repo");
    std::fs::create_dir_all(base.join("indra/board/tickets")).unwrap();
    std::fs::write(
        base.join("indra/board/tickets/T-OPEN.md"),
        "```yaml ticket\nstate: open\n```\n",
    )
    .unwrap();
    write_files(&root, files);
    write_contract(&root, contract);
    git(&root, &["init", "-q"]);
    git(&root, &["add", "-A"]);
    git(
        &root,
        &[
            "-c",
            "user.email=a@b",
            "-c",
            "user.name=a",
            "commit",
            "-q",
            "-m",
            "base",
        ],
    );
    root
}

fn write_files(root: &Path, files: &[(&str, &str)]) {
    for (path, text) in files {
        let p = root.join(path);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }
}

fn write_contract(root: &Path, contract: &str) {
    std::fs::create_dir_all(root.join("docs/contracts")).unwrap();
    std::fs::write(root.join("docs/contracts/one-path.toml"), contract).unwrap();
}

fn git(root: &Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {out:?}");
}

/// Stages `files` and runs `--staged`, returning the refusals it adds to `HEAD`'s.
fn staged(root: &Path, files: &[(&str, &str)]) -> Vec<String> {
    write_files(root, files);
    git(root, &["add", "-A"]);
    let out = super::run(root, true).map(|(_, o, errors)| {
        let mut r = o.refusals;
        r.extend(errors.into_iter().map(|e| format!("unparsed: {e}")));
        r
    });
    out.unwrap()
}

const BASE: [(&str, &str); 4] = [
    ("src/lib.rs", LIB),
    ("tests/t.rs", T),
    ("src/gpu/spirv/a.comp", UPDATE),
    ("src/gpu/spirv/a.spv", ""),
];

#[test]
fn ratchet_refuses_a_new_refusal() {
    let root = git_repo(
        &BASE,
        &format!("{EMPTY}\n{}", row("x", "src/gpu/spirv/a.comp", false)),
    );
    let r = staged(&root, &[("src/gpu/spirv/twin.comp", COOP)]);
    assert!(
        has(&r, "ratchet: the index adds a refusal HEAD had not")
            && has(&r, "new shader at src/gpu/spirv/twin.comp"),
        "{r:?}"
    );
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
}

#[test]
fn ratchet_refuses_a_staged_debt_row_for_a_finding_head_had_not() {
    let mut files = BASE.to_vec();
    files[2] = ("src/gpu/spirv/a.comp", MAC);
    let base = format!(
        "{EMPTY}\n{}\n[[debt]]\nrule = \"shader-product\"\nitem = \"src/gpu/spirv/a.comp\"\nconsumers = 0\nticket = \"T-OPEN\"\n",
        row("x", "src/gpu/spirv/a.comp", false)
    );
    let root = git_repo(&files, &base);
    // A second product shader declared beside the first, carried by a row of its own.
    let twin = (
        "src/gpu/spirv/twin.comp",
        "void main() {\n    s += a[k] * b[k];\n}\n",
    );
    write_contract(
        &root,
        &format!(
            "{base}{}\n[[debt]]\nrule = \"shader-product\"\nitem = \"src/gpu/spirv/twin.comp\"\nconsumers = 0\nticket = \"T-OPEN\"\n",
            row("y", "src/gpu/spirv/twin.comp", false)
        ),
    );
    let r = staged(&root, &[twin, ("src/gpu/spirv/twin.spv", "")]);
    assert!(
        has(&r, "adds a debt row whose finding HEAD had not"),
        "{r:?}"
    );
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
}

#[test]
fn ratchet_refuses_a_grown_consumer_count() {
    let mut files = BASE.to_vec();
    files[2] = ("src/gpu/spirv/a.comp", MAC);
    let base = format!(
        "{EMPTY}\n{}\n[[debt]]\nrule = \"shader-product\"\nitem = \"src/gpu/spirv/a.comp\"\nconsumers = 0\nticket = \"T-OPEN\"\n",
        row("x", "src/gpu/spirv/a.comp", false)
    );
    let root = git_repo(&files, &base);
    // A second embed site of the face, with the row re-baselined to 1 in the same commit: the
    // finding is carried, so only the ratchet sees the growth.
    write_contract(&root, &base.replace("consumers = 0", "consumers = 1"));
    let embed = "#[test]\nfn t() {\n    apeiron::t();\n}\n\npub static A_SPV: &[u8] = include_bytes!(\"../src/gpu/spirv/a.spv\");\n";
    let r = staged(&root, &[("tests/t.rs", embed)]);
    assert!(has(&r, "consumers grew 0 -> 1"), "{r:?}");
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
}

#[test]
fn ratchet_refuses_a_new_product_kernel_row() {
    let root = git_repo(
        &BASE,
        &format!("{EMPTY}\n{}", row("x", "src/gpu/spirv/a.comp", false)),
    );
    // A second product role beside the declared one: the engine holds one kernel per role.
    write_contract(
        &root,
        &format!(
            "{EMPTY}\n{}{}",
            row("x", "src/gpu/spirv/a.comp", false),
            row("twin", "src/gpu/spirv/twin.comp", true)
        ),
    );
    let r = staged(
        &root,
        &[
            ("src/gpu/spirv/twin.comp", COOP),
            ("src/gpu/spirv/twin.spv", ""),
        ],
    );
    assert!(has(&r, "adds a product kernel row `twin`"), "{r:?}");
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
}

#[test]
fn ratchet_accepts_deleting_a_debt_shader_and_its_row() {
    let base = format!(
        "{EMPTY}\n{}\n[[debt]]\nrule = \"shader-product\"\nitem = \"src/gpu/spirv/a.comp\"\nconsumers = 0\nticket = \"T-OPEN\"\n",
        row("x", "src/gpu/spirv/a.comp", false)
    );
    let mut files = BASE.to_vec();
    files[2] = ("src/gpu/spirv/a.comp", MAC);
    let root = git_repo(&files, &base);
    // The burn-down: remove the shader, its face and its row in one commit.
    std::fs::remove_file(root.join("src/gpu/spirv/a.comp")).unwrap();
    std::fs::remove_file(root.join("src/gpu/spirv/a.spv")).unwrap();
    write_contract(&root, EMPTY);
    git(&root, &["add", "-A"]);
    let out = super::run(&root, true).map(|(_, o, _)| o.refusals);
    assert_eq!(out.unwrap(), Vec::<String>::new());
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
}

#[test]
fn product_text_test() {
    assert!(super::shader::product_line(MAC).is_some());
    assert!(super::shader::product_line(COOP).is_some());
    assert!(super::shader::product_line(FMA).is_some());
    // A state update, index arithmetic, a comment, and bit packing are not products.
    assert!(super::shader::product_line(UPDATE).is_none());
    assert!(super::shader::product_line(
        "for (int i = 0; i < n; ++i) {\n    a[i * 4u + 1] += d;\n}\n"
    )
    .is_none());
    assert!(
        super::shader::product_line("// acc += a * b in block order\nvoid main() {}\n").is_none()
    );
    assert!(super::shader::product_line("for (uint z = 0u; z < 2u; ++z) {\n    v += (int64_t(o[(2u * z + 1u) * P]) << 32) | w;\n}\n").is_none());
}

// ---- (e) the source-count ratchet ----------------------------------------------------------------
//
// The hole this closes: `norm` collapses the count, so a row `HEAD` already refuses with "consumers
// moved: row 12, source 38" passes (a) unchanged when the source moves to 45, and (c) reads only the
// contract's `consumers`, not the source's. (e) compares the source count on both trees.

const STEP: &str = "pub fn step(x: f32) -> f32 {\n    x + 1.0\n}\n";

/// A `backward` body with `callers` source-side consumers: `callers - 1` reachable fns outside the
/// backward tree and the one test fn that reaches them.
fn backward_files(callers: usize) -> Vec<(&'static str, String)> {
    let mut app = String::new();
    let mut calls = String::new();
    for i in 0..callers - 1 {
        app.push_str(&format!(
            "pub fn u{i}() -> f32 {{\n    crate::learn::grad::thing::step(1.0)\n}}\n"
        ));
        calls.push_str(&format!("    apeiron::app::u{i}();\n"));
    }
    vec![
        ("src/lib.rs", "pub mod app;\npub mod learn;\n".to_string()),
        ("src/learn/mod.rs", "pub mod grad;\n".to_string()),
        ("src/learn/grad/mod.rs", "pub mod thing;\n".to_string()),
        ("src/learn/grad/thing.rs", STEP.to_string()),
        ("src/app.rs", app),
        ("tests/t.rs", format!("#[test]\nfn t() {{\n{calls}}}\n")),
    ]
}

/// `HEAD` carries 38 source consumers; the row states 12, so `HEAD` already refuses the row as moved.
fn moved_row() -> String {
    format!(
        "{EMPTY}\n[[debt]]\nrule = \"backward\"\nitem = \"src/learn/grad/thing.rs:step\"\nconsumers = 12\nticket = \"T-OPEN\"\n"
    )
}

fn refs<'a>(files: &'a [(&'static str, String)]) -> Vec<(&'static str, &'a str)> {
    files.iter().map(|(p, t)| (*p, t.as_str())).collect()
}

fn staged_backward(head_callers: usize, index_callers: usize) -> Vec<String> {
    let head = backward_files(head_callers);
    let root = git_repo(&refs(&head), &moved_row());
    let now = backward_files(index_callers);
    let r = staged(&root, &refs(&now));
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    r
}

#[test]
fn ratchet_refuses_a_grown_source_count_on_an_already_red_row() {
    // HEAD 38, index 45: the row is already refused at both, so only (e) can see it.
    let r = staged_backward(38, 45);
    assert!(
        has(
            &r,
            "ratchet: backward src/learn/grad/thing.rs:step source consumers grew 38 -> 45"
        ),
        "{r:?}"
    );
}

#[test]
fn ratchet_accepts_an_unchanged_or_shrunk_source_count() {
    // Equal: the row is refused at both trees and the commit does not grow it.
    let same = staged_backward(38, 38);
    assert!(same.is_empty(), "{same:?}");
    // Lower: a burn-down in the same commit is allowed.
    let down = staged_backward(38, 30);
    assert!(down.is_empty(), "{down:?}");
}

#[test]
fn ratchet_accepts_shrunk_orphan_debt_row() {
    let files = [
        ("src/lib.rs", "pub mod a;\n".to_string()),
        ("src/a.rs", "pub fn o1() {}\npub fn o2() {}\n".to_string()),
        ("tests/t.rs", "#[test]\nfn t() {}\n".to_string()),
    ];
    let head_contract = format!(
        "{EMPTY}\n[[debt]]\nrule = \"orphan\"\nitems = [\"src/a.rs:o1\", \"src/a.rs:o2\"]\nticket = \"T-OPEN\"\n"
    );
    let root = git_repo(&refs(&files), &head_contract);
    // In index, o2 becomes reached/used by test or code, so only o1 remains orphan:
    let now_files = [
        ("src/lib.rs", "pub mod a;\n".to_string()),
        ("src/a.rs", "pub fn o1() {}\npub fn o2() {}\n".to_string()),
        ("tests/t.rs", "#[test]\nfn t() {\n    apeiron::a::o2();\n}\n".to_string()),
    ];
    let now_contract = format!(
        "{EMPTY}\n[[debt]]\nrule = \"orphan\"\nitems = [\"src/a.rs:o1\"]\nticket = \"T-OPEN\"\n"
    );
    write_contract(&root, &now_contract);
    let r = staged(&root, &refs(&now_files));
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    assert!(r.is_empty(), "expected clean ratchet pass on shrunk orphan debt row, got {r:?}");
}

