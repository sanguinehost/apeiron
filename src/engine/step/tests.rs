#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    reason = "test-only fixture/reference arithmetic on bounded small values; not extracted"
)]

use super::*;
use crate::format::quant::GGML_TYPE_F16;

/// Embedding width, rotation block and folded width are all this one value.
const N: usize = 8;
const OUT: usize = 4;
const X: [i64; N] = [
    1_400_000_000,
    -900_000_000,
    2_000_000_000,
    -300_000_000,
    700_000_000,
    -1_700_000_000,
    1_100_000_000,
    -500_000_000,
];

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}
fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
    put_str(b, k);
    put_u32(b, 4);
    put_u32(b, v);
}
fn kv_bool(b: &mut Vec<u8>, k: &str) {
    put_str(b, k);
    put_u32(b, 7);
    b.push(1);
}
fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
    put_str(b, k);
    put_u32(b, 8);
    put_str(b, v);
}
fn kv_i32_arr(b: &mut Vec<u8>, k: &str, v: &[i32]) {
    put_str(b, k);
    put_u32(b, 9);
    put_u32(b, 5);
    put_u64(b, v.len() as u64);
    for x in v {
        put_u32(b, *x as u32);
    }
}
fn kv_str_arr(b: &mut Vec<u8>, k: &str, v: &[&str]) {
    put_str(b, k);
    put_u32(b, 9);
    put_u32(b, 8);
    put_u64(b, v.len() as u64);
    for s in v {
        put_str(b, s);
    }
}

/// A one-layer `qwen2` fixture with one `[N, OUT]` `F16` `token_embd.weight`. `version == 0`
/// writes no fold contract; `signs = Some((widths, values))` is `explicit` mode.
fn build_gguf(
    w: &[u8],
    version: u32,
    tied: bool,
    names: &[&str],
    signs: Option<(&[i32], &[i32])>,
) -> Vec<u8> {
    let mut n_kv = 6u64;
    if version != 0 {
        n_kv += 6 + u64::from(tied) + 2 * u64::from(signs.is_some());
    }
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, 1);
    put_u64(&mut b, n_kv);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(&mut b, "qwen2.block_count", 1);
    kv_u32(&mut b, "qwen2.attention.head_count", N as u32);
    kv_u32(&mut b, "qwen2.attention.head_count_kv", N as u32);
    kv_u32(&mut b, "qwen2.embedding_length", N as u32);
    kv_u32(&mut b, "qwen2.feed_forward_length", N as u32);
    if version != 0 {
        kv_u32(&mut b, "prism.hadamard.version", version);
        kv_u32(&mut b, "prism.hadamard.block_size", N as u32);
        kv_str(
            &mut b,
            "prism.hadamard.transform",
            "normalized-sylvester-walsh-hadamard",
        );
        kv_str(&mut b, "prism.hadamard.axis", "input-last-dimension");
        kv_str(
            &mut b,
            "prism.hadamard.sign_mode",
            if signs.is_some() {
                "explicit"
            } else {
                "identity"
            },
        );
        kv_str_arr(&mut b, "prism.hadamard.weight_names", names);
        if tied {
            kv_bool(&mut b, "prism.hadamard.tied_output");
        }
        if let Some((widths, values)) = signs {
            kv_i32_arr(&mut b, "prism.hadamard.sign_widths", widths);
            kv_i32_arr(&mut b, "prism.hadamard.sign_values", values);
        }
    }
    put_str(&mut b, "token_embd.weight");
    put_u32(&mut b, 2);
    put_u64(&mut b, N as u64);
    put_u64(&mut b, OUT as u64);
    put_u32(&mut b, GGML_TYPE_F16);
    put_u64(&mut b, 0);
    let pad = (32 - b.len() % 32) % 32;
    b.resize(b.len() + pad, 0);
    b.extend_from_slice(w);
    b
}

fn open(tag: &str, bytes: &[u8]) -> (Engine, String) {
    let path = std::env::temp_dir().join(format!(
        "apeiron_hadamard_fold_{}_{tag}",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

/// Round-to-nearest-even `f16` bits for a finite normal `f32` in `(2^-14, 65504)`.
fn f16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits << 1 == 0 {
        return 0;
    }
    let sign = ((bits >> 16) & 0x8000) as u16;
    let e = i32::try_from((bits >> 23) & 0xff).expect("8 bits") - 127 + 15;
    let mut m = i32::try_from((bits >> 13) & 0x3ff).expect("10 bits");
    let rem = bits & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        m += 1;
    }
    let carry = m >> 10;
    m &= 0x3ff;
    sign | (((e + carry) as u16) << 10) | (m as u16)
}

/// The `[-4, 4]` integer weight of the fixture.
fn w_int(o: usize, j: usize) -> i32 {
    i32::try_from((o * 7 + j * 3 + 1) % 9).expect("small") - 4
}

fn f16_bytes(v: &[i32]) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(v.len() * 2);
    for x in v {
        b.extend_from_slice(&f16_round(*x as f32).to_le_bytes());
    }
    b
}

fn weight_bytes() -> Vec<u8> {
    let mut v: Vec<i32> = Vec::with_capacity(OUT * N);
    for o in 0..OUT {
        for j in 0..N {
            v.push(w_int(o, j));
        }
    }
    f16_bytes(&v)
}

/// The f16 payload and exact values of `W' = (W ⊙ s)·H/√n`.
fn folded_weight(s: [i8; N]) -> (Vec<u8>, [[f64; N]; OUT]) {
    let sqrt_n = (N as f64).sqrt();
    let mut wfold = [[0f64; N]; OUT];
    let mut b: Vec<u8> = Vec::with_capacity(OUT * N * 2);
    for o in 0..OUT {
        for i in 0..N {
            let mut acc = 0f64;
            for j in 0..N {
                let h = if (i & j).count_ones() % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                acc += f64::from(w_int(o, j)) * f64::from(s[j]) * h;
            }
            wfold[o][i] = acc / sqrt_n;
            b.extend_from_slice(&f16_round(wfold[o][i] as f32).to_le_bytes());
        }
    }
    (b, wfold)
}

/// Folding identity: `W'·T(x) = W·x` for the signed normalized rotation `T`, up to the `f16`
/// requantization of `W'`; a tensor the contract does not name keeps the raw activation.
#[test]
fn hadamard_fold_projection_matches_the_plain_weight() {
    let s: [i8; N] = [1, -1, 1, 1, -1, -1, 1, -1];
    let signs: [i32; N] = [1, -1, 1, 1, -1, -1, 1, -1];
    let name = "token_embd.weight";
    let plain_w = weight_bytes();
    let (folded_w, wfold) = folded_weight(s);
    let width = [i32::try_from(N).expect("small")];
    let (plain, p0) = open("plain", &build_gguf(&plain_w, 0, false, &[], None));
    let (folded, p1) = open(
        "folded",
        &build_gguf(&folded_w, 1, false, &[name], Some((&width, &signs))),
    );
    let y_plain = plain.step_proj(name, OUT, N, &X).expect("plain projection");
    let y_fold = folded
        .step_proj(name, OUT, N, &X)
        .expect("folded projection");
    let xr = folded_activation(folded.hadamard(), name, &X)
        .expect("rotation")
        .expect("the weight is folded");
    let _ = std::fs::remove_file(&p0);
    let _ = std::fs::remove_file(&p1);
    // Tolerance (Q32.32 units), dominated by the f16 requantization of `W'`: per output
    // `Σ_i (ulp/2)·|x'_i|`; the slack covers the single-rounding `x'` and matvec terms.
    let mut tol = 1024f64;
    let mut diff = 0f64;
    for o in 0..OUT {
        for i in 0..N {
            let half_ulp = 2f64.powf(wfold[o][i].abs().log2().floor() - 11.0);
            tol = tol.max(half_ulp * (xr[i].abs() as f64));
        }
        diff = diff.max((y_fold[o] - y_plain[o]).abs() as f64);
    }
    assert_eq!(y_plain.len(), OUT);
    assert!(
        diff <= tol,
        "diff {diff} exceeds the requantization bound {tol}"
    );
    // An unlisted tensor gets no rotation.
    let meta = HadamardMeta {
        version: 1,
        block_size: N as u32,
        tied_output: false,
        weight_names: vec![name.to_string()],
        signs: None,
    };
    assert_eq!(
        folded_activation(Some(&meta), "output.weight", &X),
        Ok(None)
    );
}

/// An `explicit` contract with no sign vector of the activation width refuses typed.
#[test]
fn hadamard_fold_missing_sign_width_refuses() {
    let w = weight_bytes();
    let name = "token_embd.weight";
    let values: [i32; 16] = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1];
    let (e, path) = open(
        "missing",
        &build_gguf(&w, 1, false, &[name], Some((&[16], &values))),
    );
    let got = e.step_proj(name, OUT, N, &X);
    let _ = std::fs::remove_file(&path);
    match got {
        Err(StepError::MissingHadamardSigns { tensor, width }) => {
            assert_eq!(tensor, name);
            assert_eq!(width, N);
        }
        other => panic!("expected a missing-sign-width refusal, got {other:?}"),
    }
}

/// A version-2 tied-output contract is refused at step time, not executed in the wrong basis.
#[test]
fn hadamard_fold_version2_tied_output_refuses_at_step() {
    let w = weight_bytes();
    let (mut e, path) = open("v2", &build_gguf(&w, 2, true, &["token_embd.weight"], None));
    assert_eq!(e.hadamard().expect("contract").version, 2);
    let got = e.step(0);
    let _ = std::fs::remove_file(&path);
    assert_eq!(got, Err(StepError::TiedOutputUnsupported));
}

// ---- APEIRON-TPUT-T1: resident payloads and row-parallel projections --------------------

/// Fixture geometry: one layer, `n_embd 8`, `n_ff 128`, 2 query heads over 1 KV head of width
/// 4, vocabulary 128 (the tied embedding). The 128-row head and FFN give the row-chunk budget
/// more than one chunk on a multi-core host, so the bit-identity test exercises the chunked
/// branch rather than the single-chunk fast path.
const F_E: usize = 8;
const F_FF: usize = 128;
const F_Q: usize = 8;
const F_KO: usize = 4;

/// `usize` to `u64` for the fixture's dims, refusing only the unreachable overflow.
fn u(x: usize) -> u64 {
    u64::try_from(x).expect("fixture dims fit u64")
}

/// A deterministic `F16` payload of `rows · cols` weights in `{-2, -1, 0, 1, 2}·2^-5`, so the
/// composed residual stays inside `fixed_rsqrt`'s window; `salt` separates tensors.
fn f16_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    const SCALE: f32 = 1.0 / 32.0;
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    let mut r = 0usize;
    while r < rows {
        let mut c = 0usize;
        while c < cols {
            let code = i32::try_from((r * 7 + c * 3 + salt * 11) % 5).expect("small") - 2;
            b.extend_from_slice(&f16_round(code as f32 * SCALE).to_le_bytes());
            c += 1;
        }
        r += 1;
    }
    b
}

/// An all-`1.0` `F32` vector of `n` gains.
fn f32_gain(n: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    let mut i = 0usize;
    while i < n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        i += 1;
    }
    b
}

/// One full dense `qwen2` layer plus the head, so `step_logits` composes every row 0–8.
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
            GGML_TYPE_F16,
            f16_weight(F_Q, F_E, 1),
        ),
        (
            "blk.0.attn_k.weight",
            vec![u(F_E), u(F_KO)],
            GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 2),
        ),
        (
            "blk.0.attn_v.weight",
            vec![u(F_E), u(F_KO)],
            GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 3),
        ),
        (
            "blk.0.attn_output.weight",
            vec![u(F_E), u(F_E)],
            GGML_TYPE_F16,
            f16_weight(F_E, F_E, 4),
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
            GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 5),
        ),
        (
            "blk.0.ffn_up.weight",
            vec![u(F_E), u(F_FF)],
            GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 6),
        ),
        (
            "blk.0.ffn_down.weight",
            vec![u(F_FF), u(F_E)],
            GGML_TYPE_F16,
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
            GGML_TYPE_F16,
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

/// Write a fixture to a per-process temp file and open it; returns the engine and its path.
fn open_bytes(tag: &str, bytes: &[u8]) -> (Engine, String) {
    let path = std::env::temp_dir().join(format!("apeiron_tput_t1_{}_{tag}", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

/// `Engine::payload` serves exactly the bytes `Engine::read_range` returns and refuses the same
/// past-EOF range.
#[test]
fn payload_borrows_the_same_bytes_as_read_range() {
    let (e, path) = open("payload", &build_gguf(&weight_bytes(), 0, false, &[], None));
    let t = e.token_embd();
    let n = t.n_bytes.expect("geometry");
    let owned = e.read_range(t.byte_range.start, n).expect("read_range");
    let borrowed = e.payload(t.byte_range.start, n).expect("payload");
    assert_eq!(borrowed, owned.as_slice());
    assert!(
        matches!(
            e.payload(e.file_len, 1),
            Err(EngineError::RowOutOfBounds { .. })
        ),
        "payload past EOF must refuse like read_range"
    );
    assert!(matches!(
        e.read_range(e.file_len, 1),
        Err(EngineError::RowOutOfBounds { .. })
    ));
    let _ = std::fs::remove_file(&path);
}

/// The resident chunked path and the pre-change sequential path produce identical Q32.32
/// logits over a token sequence (the epic's bit-identity invariant).
#[test]
fn step_logits_is_bit_identical_to_the_sequential_reference() {
    let (engine, path) = open_bytes("bitidentity", &build_layer_gguf());
    let mut par = engine.decode_state(8).expect("kv parallel");
    let mut seq = engine.decode_state(8).expect("kv sequential");
    let tokens = [0u32, 1, 2, 5, 9];
    let mut i = 0usize;
    while i < tokens.len() {
        let a = engine
            .step_logits(&mut par, tokens[i])
            .expect("resident logits");
        let b = engine
            .step_logits_sequential(&mut seq, tokens[i])
            .expect("sequential logits");
        assert_eq!(a, b, "logits diverge at position {i}");
        assert_eq!(a.len(), F_FF, "logit vector is not the vocabulary size");
        i += 1;
    }
    // The 128-row head splits into two 64-row chunks on any multi-core host, so the fixture
    // exercises the scoped-thread branch rather than the single-chunk fast path.
    if std::thread::available_parallelism().map_or(1, std::num::NonZero::get) >= 2 {
        assert!(
            crate::engine::par::row_chunk_budget(F_FF) >= 2,
            "the fixture must exercise the chunked path"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// `Q8Act` keeps the step's logits `Ok` and its greedy token equal to `Exact` on the fixture;
/// the observed max `|Δlogit|` is printed. `Exact` itself is covered by the bit-identity test.
#[test]
fn q8act_step_logits_match_the_greedy_token_of_exact() {
    let (engine, path) = open_bytes("q8act", &build_layer_gguf());
    let mut a = engine.decode_state(8).expect("kv exact");
    let mut b = engine.decode_state(8).expect("kv q8");
    let exact = engine.step_logits(&mut a, 3).expect("exact");
    let q8 = engine
        .step_logits_with_mode(&mut b, 3, KernelMode::Q8Act)
        .expect("q8act");
    let _ = std::fs::remove_file(&path);
    assert_eq!(q8.len(), F_FF);
    let max_delta = exact.iter().zip(&q8).map(|(p, q)| (p - q).abs()).max();
    eprintln!("Q8Act max |delta logit| = {max_delta:?}");
    assert_eq!(
        argmax_fixed(&exact).expect("exact argmax"),
        argmax_fixed(&q8).expect("q8 argmax"),
    );
}

/// The `Q8Act` pool head dispatch equals the serial per-head loop over a token sequence.
#[test]
fn q8act_parallel_heads_match_the_serial_head_loop() {
    let (engine, path) = open_bytes("q8heads", &build_layer_gguf());
    let mut par = engine.decode_state(8).unwrap();
    let mut seq = engine.decode_state(8).unwrap();
    for t in [0u32, 1, 2, 5, 9, 3, 7] {
        let pooled = engine.step_logits_with_mode(&mut par, t, KernelMode::Q8Act);
        let serial =
            engine.compose_logits(&mut seq, None, t, Sched::Resident, KernelMode::Q8Act, false);
        assert_eq!(pooled, serial, "head schedules diverge at token {t}");
    }
    let _ = std::fs::remove_file(&path);
}

/// The `Q8Act` step's logits are bit-identical whether the activations are quantized by the
/// pooled chunked schedule or by the single call: the schedule is a partition of the same blocks,
/// so a chunk that drops or alters an element diverges here.
#[test]
fn q8act_step_logits_match_the_unpooled_quantization() {
    let (engine, path) = open_bytes("q8pool", &build_layer_gguf());
    let mut pooled_kv = engine.decode_state(8).expect("kv pooled");
    let mut single_kv = engine.decode_state(8).expect("kv single");
    let tokens = [0u32, 1, 2, 5, 9];
    let mut i = 0usize;
    while i < tokens.len() {
        crate::engine::chart::ffn::set_pool_quant(true);
        let pooled = engine
            .step_logits_with_mode(&mut pooled_kv, tokens[i], KernelMode::Q8Act)
            .expect("pooled logits");
        crate::engine::chart::ffn::set_pool_quant(false);
        let single = engine
            .step_logits_with_mode(&mut single_kv, tokens[i], KernelMode::Q8Act)
            .expect("single-call logits");
        crate::engine::chart::ffn::set_pool_quant(true);
        assert_eq!(
            pooled, single,
            "quantization schedules diverge at token {i}"
        );
        i += 1;
    }
    let _ = std::fs::remove_file(&path);
}

/// The resident `Q8Act` step — which resolves the per-layer weights through the directory index
/// table and reuses the FFN scratch across layers — is bit-identical to the name-based
/// `Q8Act` composition, which reads each tensor by name and projects it with one unchunked
/// [`q8act_projection`](super::q8act_projection) per stage.
///
/// This is the equality oracle for the index resolution: a table entry that points at a different
/// tensor, or a reused buffer that carries a stale row into a result, diverges here, while the
/// pooled schedule has no counterpart in the reference to cancel it.
#[test]
fn q8act_resident_matches_the_name_resolved_sequential_reference() {
    let (engine, path) = open_bytes("q8seqref", &build_layer_gguf());
    let mut resident = engine.decode_state(8).expect("kv resident");
    let mut sequential = engine.decode_state(8).expect("kv sequential");
    for t in [0u32, 1, 2, 5, 9, 3, 7] {
        let a = engine
            .step_logits_with_mode(&mut resident, t, KernelMode::Q8Act)
            .expect("resident q8act logits");
        let b = engine
            .compose_logits(
                &mut sequential,
                None,
                t,
                Sched::Sequential,
                KernelMode::Q8Act,
                false,
            )
            .expect("sequential q8act logits");
        assert_eq!(
            a, b,
            "resident and name-resolved Q8Act diverge at token {t}"
        );
    }
    let _ = std::fs::remove_file(&path);
}
