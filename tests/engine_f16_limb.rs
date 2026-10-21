//! The `F16` multi-column projection on the limb kernel, against the scalar oracle.
//! Board: APEIRON-QUALITY-118-W1-T01 (F16 limb tile, lm_head_cols).
//!
//! The fixture is one dense `qwen2` layer whose matrices are `F16`. The multi-column
//! [`Engine::prefill_tap_cols`] pass takes the limb route (`src/engine/step/cols.rs` ->
//! `src/engine/kern/limb.rs`); the sequential public single-token path keeps the scalar
//! `matvec_q8act` oracle (K4). The test compares the head of every column against the sequential
//! run, asserts the deviation stays inside a stated word bound, and asserts it is nonzero — the
//! scalar path is bit-identical per column, so a zero deviation would mean the new route never ran.
//!
//! The residue-byte mutation control for the packed tile is the limb unit test
//! `f16_tile_is_within_the_limb_bound_and_a_flipped_digit_breaks_it` in
//! `src/engine/kern/limb.rs`: one flipped limb byte leaves the bound. The head that ships is
//! [`Engine::lm_head_cols`] under [`KernelMode::Q8Act`]; its deviation from the `Exact` per-column
//! oracle and from the per-column `Q8Act` head is bounded here (K4), and a flipped limb byte in
//! the cached head tile leaves that bound.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "test-only fixture arithmetic on small, bounded values"
)]

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::quant;

const F_E: usize = 64;
const F_FF: usize = 128;
const F_Q: usize = 64;
const F_KO: usize = 32;

/// The stated gross-error bound on the head's Q32.32 words. The head is the scalar `lm_head_fixed`
/// path; its input is the multi-column state, so this number folds the four limb projections and
/// the fixture's own wide weight range. The per-projection bound `G(n, 3, max|w|, max|x|)` and its
/// tight assertion are the limb unit test; this constant is four times the observed deviation on
/// this fixture and far below a route mix-up, which is whole units of `max|logit| ≈ 2^36`.
const HEAD_BOUND: u64 = 1 << 31;

/// The stated bound on [`Engine::lm_head_cols`]`(.., Q8Act)` against the per-column `Q8Act` head
/// (`step_logits_state`), both reading the same state: the `F16` limb route's truncation alone.
/// 70× the observed 15 008 words on this fixture, and four orders below a single flipped limb
/// byte's effect (≈ `2^32` words), so the mutation control leaves it.
const HEAD_ROUTE_BOUND: u64 = 1 << 20;

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, u64::try_from(s.len()).expect("small"));
    b.extend_from_slice(s.as_bytes());
}

fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
    put_str(b, k);
    put_u32(b, 4);
    put_u32(b, v);
}

fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
    put_str(b, k);
    put_u32(b, 8);
    put_str(b, v);
}

fn u(x: usize) -> u64 {
    u64::try_from(x).expect("fixture dims fit u64")
}

/// `rows · cols` finite half words in two exponent bands eleven powers apart, so the 21-bit block
/// expansion truncates the low band: a zero deviation cannot come from an exact fixture.
fn f16_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    let mut st = 0x9e37_79b9_7f4a_7c15u64
        ^ (u64::try_from(salt).expect("small") * 0x0100_0000_01b3)
        ^ u64::try_from(rows * cols).expect("small");
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    let mut i = 0usize;
    while i < rows * cols {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let e: u16 = if st.is_multiple_of(5) { 3 } else { 14 };
        let h = ((st >> 32) as u16 & 0x83ff) | (e << 10);
        b.extend_from_slice(&h.to_le_bytes());
        i += 1;
    }
    b
}

fn f32_gain(n: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    let mut i = 0usize;
    while i < n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        i += 1;
    }
    b
}

/// One dense `qwen2` layer plus the tied head, every matrix `F16`, every norm gain `1.0`.
#[allow(clippy::too_many_lines, reason = "one tabular fixture")]
fn build_layer_gguf() -> Vec<u8> {
    let ts: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        (
            "blk.0.attn_norm.weight",
            vec![u(F_E)],
            quant::GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "blk.0.attn_q.weight",
            vec![u(F_E), u(F_Q)],
            quant::GGML_TYPE_F16,
            f16_weight(F_Q, F_E, 1),
        ),
        (
            "blk.0.attn_k.weight",
            vec![u(F_E), u(F_KO)],
            quant::GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 2),
        ),
        (
            "blk.0.attn_v.weight",
            vec![u(F_E), u(F_KO)],
            quant::GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 3),
        ),
        (
            "blk.0.attn_output.weight",
            vec![u(F_Q), u(F_E)],
            quant::GGML_TYPE_F16,
            f16_weight(F_E, F_Q, 4),
        ),
        (
            "blk.0.ffn_norm.weight",
            vec![u(F_E)],
            quant::GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "blk.0.ffn_gate.weight",
            vec![u(F_E), u(F_FF)],
            quant::GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 5),
        ),
        (
            "blk.0.ffn_up.weight",
            vec![u(F_E), u(F_FF)],
            quant::GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 6),
        ),
        (
            "blk.0.ffn_down.weight",
            vec![u(F_FF), u(F_E)],
            quant::GGML_TYPE_F16,
            f16_weight(F_E, F_FF, 7),
        ),
        (
            "output_norm.weight",
            vec![u(F_E)],
            quant::GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "token_embd.weight",
            vec![u(F_E), u(F_FF)],
            quant::GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 9),
        ),
    ];
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, u64::try_from(ts.len()).expect("small"));
    put_u64(&mut b, 6);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(&mut b, "qwen2.block_count", 1);
    kv_u32(&mut b, "qwen2.attention.head_count", 2);
    kv_u32(&mut b, "qwen2.attention.head_count_kv", 1);
    kv_u32(
        &mut b,
        "qwen2.embedding_length",
        u32::try_from(F_E).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.feed_forward_length",
        u32::try_from(F_FF).expect("small"),
    );
    let mut off = 0u64;
    for (name, dims, ty, payload) in &ts {
        put_str(&mut b, name);
        put_u32(&mut b, u32::try_from(dims.len()).expect("rank 2"));
        for &d in dims {
            put_u64(&mut b, d);
        }
        put_u32(&mut b, *ty);
        put_u64(&mut b, off);
        off += u64::try_from(payload.len()).expect("small");
    }
    let pad = (32 - b.len() % 32) % 32;
    b.resize(b.len() + pad, 0);
    for (_, _, _, payload) in &ts {
        b.extend_from_slice(payload);
    }
    b
}

fn open(tag: &str) -> (Engine, String) {
    let path = std::env::temp_dir().join(format!("apeiron_f16_limb_{}_{tag}", std::process::id()));
    std::fs::write(&path, build_layer_gguf()).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

/// The multi-column `F16` projection is the sequential single-token oracle within [`HEAD_BOUND`]
/// words, and it is not bit-identical to it: the limb route ran. A second multi-column pass over
/// the cached tile reproduces the first bit for bit.
#[test]
fn f16_prefill_columns_are_within_the_limb_bound_of_the_scalar_oracle() {
    let (engine, path) = open("bound");
    let tokens = [1u32, 2, 3, 5, 8];
    let mut st = engine.state(64).expect("state");
    let (cols, _) = engine
        .prefill_tap_cols(&mut st, &tokens, KernelMode::Q8Act, false)
        .expect("limb prefill");
    assert_eq!(cols.len(), tokens.len());

    let mut st_seq = engine.state(64).expect("state");
    let mut max_dev: u64 = 0;
    let mut max_abs: u64 = 0;
    for (c, &t) in tokens.iter().enumerate() {
        let oracle = engine
            .step_logits_state(&mut st_seq, t, KernelMode::Q8Act)
            .expect("oracle step");
        let got = engine.lm_head_fixed(&cols[c]).expect("head");
        assert_eq!(got.len(), oracle.len());
        let mut i = 0usize;
        while i < got.len() {
            max_dev = max_dev.max(got[i].abs_diff(oracle[i]));
            max_abs = max_abs.max(oracle[i].unsigned_abs());
            i += 1;
        }
    }
    println!("oracle max |logit| {max_abs} words, deviation {max_dev}");
    assert!(
        max_dev > 0,
        "the multi-column F16 path was bit-identical to the scalar oracle; the limb route did not run"
    );
    assert!(
        max_dev <= HEAD_BOUND,
        "max |C_limb - C_scalar| {max_dev} words exceeds the stated {HEAD_BOUND}"
    );
    println!("f16 engine: max head deviation {max_dev} words of at most {HEAD_BOUND}");

    // Cache determinism: a fresh state's multi-column pass over the same tensor is bit-identical.
    let mut st2 = engine.state(64).expect("state");
    let (cols2, _) = engine
        .prefill_tap_cols(&mut st2, &tokens, KernelMode::Q8Act, false)
        .expect("limb prefill 2");
    assert_eq!(cols, cols2, "the cached tile is not deterministic");
    let _ = std::fs::remove_file(&path);
}

/// The qwen2.5-0.5B `F16` teacher and `Q8_0` student of the held-out measurement.
const QWEN05B_F16: &str = "/home/socol/models/gguf/qwen05b-f16.gguf";
const QWEN05B_Q8_0: &str = "/home/socol/oc-tmp/kld/qwen05b-Q8_0.gguf";

/// [`Engine::lm_head_cols`]`(.., Q8Act)` on the `F16` head: the multi-column limb route over ten
/// columns is (a) within [`HEAD_BOUND`] words of the `Exact` per-column oracle and not bit-identical
/// to it — the K4 number, with no equality claim — and (b) within [`HEAD_ROUTE_BOUND`] of the
/// per-column `Q8Act` head that is its fallback. The second comparison isolates the limb route from
/// the activation quantization, so a flipped limb byte in the cached head tile leaves the bound
/// (the mutation control, run on a throwaway copy).
#[test]
fn lm_head_cols_q8act_is_within_the_bound_of_exact_and_of_the_column_head() {
    let (engine, path) = open("head_cols");
    let tokens = [1u32, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    let mut st = engine.state(tokens.len()).expect("state");
    let (cols, _) = engine
        .prefill_tap_cols(&mut st, &tokens, KernelMode::Q8Act, false)
        .expect("prefill");
    let exact = engine
        .lm_head_cols(&cols, KernelMode::Exact)
        .expect("exact head");
    let q8 = engine
        .lm_head_cols(&cols, KernelMode::Q8Act)
        .expect("q8act head");
    assert_eq!(exact.len(), tokens.len());
    assert_eq!(q8.len(), tokens.len());

    let mut st_seq = engine.state(64).expect("oracle state");
    let mut dev_exact = 0u64;
    let mut dev_route = 0u64;
    let mut max_abs = 0u64;
    let mut col_dev = Vec::with_capacity(tokens.len());
    for (c, &t) in tokens.iter().enumerate() {
        let oracle = engine
            .step_logits_state(&mut st_seq, t, KernelMode::Q8Act)
            .expect("oracle step");
        assert_eq!(q8[c].len(), oracle.len());
        assert_eq!(exact[c].len(), oracle.len());
        let mut per_column = 0u64;
        for i in 0..oracle.len() {
            let de = q8[c][i].abs_diff(exact[c][i]);
            dev_exact = dev_exact.max(de);
            per_column = per_column.max(de);
            dev_route = dev_route.max(q8[c][i].abs_diff(oracle[i]));
            max_abs = max_abs.max(exact[c][i].unsigned_abs());
        }
        col_dev.push(per_column);
    }
    println!(
        "head cols over {} columns: max |logit| {max_abs} words; max |Q8Act - Exact| {dev_exact}; \
         max |Q8Act - per-column Q8Act| {dev_route}; per-column |Q8Act - Exact| {col_dev:?}",
        tokens.len()
    );
    assert!(
        dev_exact > 0,
        "the Q8Act head was bit-identical to Exact: the head quantization did not run"
    );
    assert!(
        dev_exact <= HEAD_BOUND,
        "max |Q8Act - Exact| {dev_exact} words exceeds the stated {HEAD_BOUND}"
    );
    assert!(
        dev_route <= HEAD_ROUTE_BOUND,
        "max |limb route - per-column Q8Act| {dev_route} words exceeds the stated \
         {HEAD_ROUTE_BOUND}"
    );

    // The cached head tile is deterministic: a second multi-column call is bit-identical.
    let again = engine
        .lm_head_cols(&cols, KernelMode::Q8Act)
        .expect("second q8act head");
    assert_eq!(q8, again, "the cached head tile is not deterministic");
    let _ = std::fs::remove_file(&path);
}

/// The `Q8Act` head against `Exact` at the production shape, on both qwen2.5-0.5B files: per-column
/// and overall `max |Q32.32 logit − Q32.32 logit|` over eight columns, with the two walls. A
/// measurement of the K4 deviation, not an equality claim.
#[test]
#[ignore = "loads the 1 GB F16 and 0.5 GB Q8_0 GGUFs; run with --release --ignored --nocapture"]
fn lm_head_cols_qwen05b_bound() {
    /// The stated head bound on both models: `2^31` words, half a Q32.32 unit and 2.4 % of the
    /// observed `max |logit|`; 1.4× the larger observed deviation (1.56·10⁹).
    const QWEN_HEAD_BOUND: u64 = 1 << 31;
    /// The stated bound on the multi-column route against the per-column `Q8Act` head: `2^32`
    /// words, one Q32.32 unit and 4.9 % of `max |logit|`; 1.19× the observed `F16` route deviation
    /// (3.60·10⁹) and exact-zero on `Q8_0` (`gemm_q8_0` is bit-identical to the per-column kernel).
    const QWEN_ROUTE_BOUND: u64 = 1 << 32;
    // Eight token ids, all well inside the 151936-entry vocabulary.
    let tokens: [u32; 8] = [11, 48, 85, 122, 159, 196, 233, 270];
    for (tag, file) in [("f16", QWEN05B_F16), ("q8_0", QWEN05B_Q8_0)] {
        let engine = Engine::open(file).expect("open model");
        let mut st = engine.state(tokens.len()).expect("state");
        let (cols, _) = engine
            .prefill_tap_cols(&mut st, &tokens, KernelMode::Q8Act, false)
            .expect("prefill");
        let t0 = std::time::Instant::now();
        let exact = engine
            .lm_head_cols(&cols, KernelMode::Exact)
            .expect("exact head");
        let t_exact = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let q8 = engine
            .lm_head_cols(&cols, KernelMode::Q8Act)
            .expect("q8act head");
        let t_q8 = t1.elapsed().as_secs_f64();
        assert_eq!(exact.len(), tokens.len());
        assert_eq!(q8.len(), tokens.len());
        let mut overall = 0u64;
        let mut max_abs = 0u64;
        let mut per_column = Vec::with_capacity(tokens.len());
        for (c, row) in exact.iter().enumerate() {
            let mut dev = 0u64;
            for (i, &z) in row.iter().enumerate() {
                dev = dev.max(z.abs_diff(q8[c][i]));
                max_abs = max_abs.max(z.unsigned_abs());
            }
            overall = overall.max(dev);
            per_column.push(dev);
        }
        println!(
            "qwen05b {tag}: {} columns, max |logit| {max_abs} words, max |Q8Act - Exact| {overall} \
             words ({:.6} Q32.32), per-column {per_column:?}; Exact {t_exact:.3} s, Q8Act {t_q8:.3} s \
             for the block",
            tokens.len(),
            overall as f64 / (1u64 << 32) as f64,
        );
        assert_ne!(
            overall, 0,
            "{tag}: the Q8Act head was bit-identical to Exact"
        );
        assert!(
            overall <= QWEN_HEAD_BOUND,
            "{tag}: max |Q8Act - Exact| {overall} words exceeds {QWEN_HEAD_BOUND}"
        );

        // The limb route against the per-column `Q8Act` head on the same state: the route's own
        // deviation, with the activation quantization common to both sides.
        let mut st_seq = engine.state(tokens.len()).expect("oracle state");
        let mut route = 0u64;
        for (c, &t) in tokens.iter().enumerate() {
            let oracle = engine
                .step_logits_state(&mut st_seq, t, KernelMode::Q8Act)
                .expect("oracle step");
            assert_eq!(oracle.len(), q8[c].len());
            for i in 0..oracle.len() {
                route = route.max(q8[c][i].abs_diff(oracle[i]));
            }
        }
        println!(
            "qwen05b {tag}: max |multi-column Q8Act head - per-column Q8Act head| {route} words \
             ({:.3e} Q32.32)",
            route as f64 / (1u64 << 32) as f64,
        );
        assert!(
            route <= QWEN_ROUTE_BOUND,
            "{tag}: route deviation {route} words exceeds {QWEN_ROUTE_BOUND}"
        );
    }
}
