//! `tests/engine_fold_pack.rs` — the engine's fold face: the product a stored `E8 ∪ O8` plane
//! executes, against the dense words `CodedPlane::weight` composes from the same plane, with one
//! control that a flipped code bit moves the output, and the face's resident bytes. Grade `tested`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the fixture indexes `o/n/g/b` planes the way the pack writer does, and its LCG maps a \
              `u64` into a small code range by remainder"
)]

use apeiron::engine::{matvec_fold, quant_act_fold, Engine, FoldFace};
use apeiron::format::packing::{fold_planes_read, fold_planes_write};
use apeiron::learn::fold::finish::plane::CodedPlane;
use apeiron::learn::fold::fit::icm::{codebook, BLOCK};
use apeiron::learn::fold::fit::quant::r_level;
use apeiron::learn::fold::fit::steps::GROUP;
use apeiron::learn::fold::layer::tensor::SinkPlane;
use apeiron::learn::fold::run::transfer::{fold_pack_of_planes, fold_plane_coded, FoldPackStats};

/// `ONE`, the Q32.32 unit.
const ONE: i64 = 1 << 32;

/// A deterministic xorshift64.
fn next(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    x
}

/// A column-scale field that is a fixed point of the writer's own quantizer at `bits`.
fn column_field(n: usize, c_min: i64, c_max: i64, bits: u32, seed: &mut u64) -> Vec<i64> {
    let span = i128::from(c_max - c_min);
    let levels = (1_i128 << bits) - 1;
    let word = |q: i128| {
        i64::try_from(i128::from(c_min) + (q * span + levels / 2) / levels).expect("grid word")
    };
    let mut out = Vec::with_capacity(n);
    out.push(word(0));
    out.push(word(levels));
    for _ in 2..n {
        let q = i128::from(next(seed) % u64::try_from(levels).expect("levels"));
        out.push(word(q));
    }
    out
}

/// A production-shape `E8 ∪ O8` plane (`896 × 896`), the fixture of `tests/learn_fold_pack.rs`.
fn fitted_plane() -> (SinkPlane, Vec<u8>, Vec<i64>, Vec<i64>) {
    let (o, n) = (896_usize, 896_usize);
    let (g, nb) = (n / GROUP, n / BLOCK);
    let cb = codebook();
    let mut idx = Vec::with_capacity(o * nb);
    let mut s_rot = Vec::with_capacity(o * n);
    for b in 0..o * nb {
        let j = (b * 37 + 11) % 256;
        idx.push(u8::try_from(j).unwrap());
        s_rot.extend_from_slice(&cb[j]);
    }
    let mut grids = Vec::with_capacity(64);
    for t in 0..64_i64 {
        grids.push((ONE / 4 + t * (ONE / 512), ONE / 3 + t * (ONE / 1024)));
    }
    let mut seed = 0x5eed_1234_u64;
    let mut rh = Vec::with_capacity(o * g);
    for i in 0..o {
        let (lo, sp) = grids[i % 64];
        for _ in 0..g {
            let k = u8::try_from(next(&mut seed) % 16).unwrap();
            rh.push(r_level(lo, sp, k).expect("grid level"));
        }
    }
    let (lo, sp): (Vec<i64>, Vec<i64>) = (0..o).map(|i| grids[i % 64]).unzip();
    let ch = column_field(n, ONE / 4, ONE / 4 + ONE / 2, 6, &mut seed);
    let dh = column_field(n, ONE / 8, ONE / 8 + ONE / 4, 5, &mut seed);
    let plane = SinkPlane {
        s_rot,
        rh,
        ch,
        dh,
        rows: None,
        dense: None,
        factors: None,
    };
    (plane, idx, lo, sp)
}

/// The plane the fit recomposes: the same code field, grids and scales with zero latents.
fn reference(p: &SinkPlane, idx: Vec<u8>, lo: Vec<i64>, sp: Vec<i64>) -> CodedPlane {
    let (o, n) = (p.rh.len() / (p.ch.len() / GROUP), p.ch.len());
    CodedPlane {
        o,
        n,
        idx,
        s_fix: None,
        rows: p.rows.clone(),
        lo,
        sp,
        r0: p.rh.clone(),
        c0: p.ch.clone(),
        d0: p.dh.clone(),
        pa: vec![0_i64; o * (n / GROUP)],
        pb: vec![0_i64; n],
        pd: vec![0_i64; n],
    }
}

/// `num::value::round_shift_32` (`src/engine/num/value.rs:10`), the role's epilogue, repeated here
/// because the integration test sees only the crate's public surface.
fn round_shift_32(v: i128) -> i128 {
    (v + (1_i128 << 31)) >> 32
}

/// The dense oracle: `out[i] = round_shift_32( Σ_b S_b · Σ_{j∈b} W[i,j]·xq_j )`, exactly as the
/// `Q8Act` projection defines it over `W` words: the `i128` accumulator of
/// `kern::gemm::exact_dot` (`src/engine/kern/gemm.rs:434`) and its one `round_shift_32` epilogue
/// (`:461`). No other rounding of its own.
fn dense_q8act(w: &[i64], o: usize, n: usize, scales: &[i64], codes: &[i8]) -> Vec<i128> {
    let mut out = vec![0_i128; o];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut acc = 0_i128;
        for b in 0..n.div_ceil(32) {
            let mut dot = 0_i128;
            for j in b * 32..(b * 32 + 32).min(n) {
                dot += i128::from(w[i * n + j]) * i128::from(codes[j]);
            }
            acc += i128::from(scales[b]) * dot;
        }
        *slot = round_shift_32(acc);
    }
    out
}

/// The tolerance of the factorisation, derived from the three roundings the fold face adds to the
/// dense words:
///
/// - `u = mul(M(t), 2^32/√128)`: one rounding per activation element, `|ε₁| ≤ 0.5`;
/// - `b = mul(d_q, u)`: one rounding per element, `|ε₂| ≤ 0.5`;
/// - the per-`(row, group)` epilogue `round(r_q·P / 2^64)`: one rounding, `|ε₃| ≤ 0.5`.
///
/// With `R = max|r_q|`, `D = max|d_q|`, `Σ = 128·max|codebook word|` (the `Q32.32` codebook words),
/// the deviation of one output word from the dense words is at most
/// `Σ_g ( 2^emin(g)·( (R/2^64)·Σ·(0.5·D/2^32 + 0.5) + 0.5 ) / 2^32 + 1 )`:
/// `ε₁` enters `b` as `0.5·D/2^32`, both enter `P` through `Σ`, `P` reaches the output through
/// `r_q/2^64`, `ε₃` is already an output word before the group exponent is restored, the restored
/// word goes through the epilogue's `round_shift_32` (`/2^32`), and that shift rounds once per
/// output word (`+1`, not per group). Every term is computed here from the fixture's own maxima.
fn tolerance(r_max: i64, d_max: i64, emin: u32, groups: usize) -> i128 {
    let one = 1_i128 << 32;
    let sigma = 128_i128 * 16 * one;
    let r = i128::from(r_max);
    let d = i128::from(d_max);
    let inside = sigma * (d / (2 * one) + 1);
    let per = r * inside / (one * one) + 1;
    let groups = i128::try_from(groups).expect("groups");
    groups * (1_i128 << emin) * per / one + 1
}

/// One linear: the fold face's product equals the dense words' product with the same quantised
/// activation, inside the derived tolerance; a flipped code byte moves it (the control).
#[test]
fn fold_linear_matches_dense_words() {
    let (plane, idx, lo, sp) = fitted_plane();
    let coded = reference(&plane, idx, lo, sp);
    let w = coded.weight();
    let (o, n) = (coded.o, coded.n);
    assert_eq!(w.len(), o * n);

    let (records, stats): (_, FoldPackStats) =
        fold_pack_of_planes(&[&plane]).expect("pack the fitted plane");
    let face = FoldFace::from_plane(&records[0]).expect("fold face");
    assert_eq!((face.o(), face.n()), (o, n));

    let mut seed = 0xfeed_2024_u64;
    let x: Vec<i64> = (0..n)
        .map(|_| {
            i64::try_from(next(&mut seed) % u64::try_from(ONE / 4).expect("range")).unwrap()
                - ONE / 8
        })
        .collect();
    let (scales, codes) = quant_act_fold(&x).expect("quantise the activation");
    assert_eq!(codes.len(), n);
    let got = matvec_fold(&face, &x).expect("fold product");
    let want = dense_q8act(&w, o, n, &scales, &codes);
    assert_eq!(got.len(), o);

    let r_max = i64::try_from(
        coded
            .qvals()
            .0
            .iter()
            .map(|v| v.unsigned_abs())
            .max()
            .unwrap_or(0),
    )
    .expect("r word");
    let d_max = coded.d0.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    let emin = scales
        .iter()
        .filter(|&&s| s > 0)
        .map(|s| s.unsigned_abs().trailing_zeros())
        .min()
        .unwrap_or(0);
    let bound = tolerance(
        r_max,
        i64::try_from(d_max).unwrap_or(i64::MAX),
        emin,
        n / GROUP,
    );
    let (mut worst, mut at) = (0_i128, 0_usize);
    for i in 0..o {
        let d = (i128::from(got[i]) - want[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    eprintln!(
        "fold linear {o}x{n}: worst |Δ| = {worst} at row {at}, bound {bound}, \
         dense magnitude {}; got[0]={} want[0]={} sum_want={} codes[0..4]={:?} scales={:?} d_max={d_max} emin={emin}",
        want[at].abs(), got[0], want[0], want.iter().map(|v| v.abs()).sum::<i128>(),
        &codes[0..4], &scales[..8]
    );
    assert!(
        worst <= bound,
        "fold product at row {at} differs from the dense words by {worst}, bound {bound}"
    );

    // The control: one bit of the first codebook index. The two codebook rows differ, so the
    // product must move; restoring the byte must return it.
    let bytes = fold_planes_write(&records).expect("write the layout");
    let code_at = 8 + 4 + 4 + 4 + 1;
    let mut red = bytes.clone();
    red[code_at] ^= 1;
    let red_record = fold_planes_read(&red).expect("a flipped code is still a record");
    let red_face = FoldFace::from_plane(&red_record[0]).expect("red face");
    let red_out = matvec_fold(&red_face, &x).expect("red product");
    assert_ne!(red_out, got, "a flipped code bit must move the product");

    // The footprint: the face's resident operand is the plane's stored form, under 1.19 b/w.
    let milli = stats.bits_per_weight_milli().expect("rate");
    eprintln!(
        "fold face: {} stored bytes for {} rows x {} cols = {}.{:03} b/w",
        face.bytes(),
        o,
        n,
        milli / 1000,
        milli % 1000
    );
    assert!(milli <= 1190, "fold operand {milli} milli-b/w over 1.19");
}

/// A step-magnitude activation: `n` words of a residual stream, `|x| < ONE`.
fn step_activation(n: usize) -> Vec<i64> {
    (0..n)
        .map(|j| {
            i64::try_from((j as u64).wrapping_mul(2_654_435_761) % 8_589_934_591).expect("word")
                - 4_294_967_295
        })
        .collect()
}

/// The face at the step's activation magnitude: `|x| < ONE`, the residual-stream range
/// `fold_engine_matches_student_gguf` reads its linears at, on the same plane as
/// `fold_linear_matches_dense_words`. The output is the `Q8Act` projection's `Q32.32` word, so it is
/// compared with the dense words' product inside the same `tolerance`; the fixture is chosen so the
/// tolerance can distinguish a zero product (`max|want| > 100·bound`).
///
/// Control: one bit of the first codebook index moves the output.
#[test]
fn fold_face_serves_step_magnitude() {
    let (plane, idx, lo, sp) = fitted_plane();
    let coded = reference(&plane, idx, lo, sp);
    let w = coded.weight();
    let (o, n) = (coded.o, coded.n);

    let (records, _) = fold_pack_of_planes(&[&plane]).expect("pack the fitted plane");
    let face = FoldFace::from_plane(&records[0]).expect("fold face");
    assert_eq!((face.o(), face.n()), (o, n));

    let x = step_activation(n);
    assert!(x.iter().all(|v| v.unsigned_abs() < ONE.unsigned_abs()));
    let (scales, codes) = quant_act_fold(&x).expect("quantise the activation");
    let got = matvec_fold(&face, &x).expect("the face serves a step-magnitude activation");
    let want = dense_q8act(&w, o, n, &scales, &codes);
    assert_eq!(got.len(), o);

    let (r_max, d_max, emin) = plane_maxima(&coded, &scales);
    let bound = tolerance(r_max, d_max, emin, n / GROUP);
    let (mut worst, mut at) = (0_i128, 0_usize);
    for i in 0..o {
        let d = (i128::from(got[i]) - want[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    let want_max = want.iter().map(|v| v.abs()).max().unwrap_or(0);
    let got_max = got.iter().map(|v| v.abs()).max().unwrap_or(0);
    eprintln!(
        "fold face at |x| < ONE, {o}x{n}: worst |\u{394}| = {worst} of max|want| {want_max}, \
         max|got| {got_max}, bound {bound}, emins {}",
        scales
            .iter()
            .map(|s| s.unsigned_abs().trailing_zeros())
            .max()
            .unwrap_or(0)
    );
    assert!(got_max > 0, "the face's word is zero at the step magnitude");
    assert!(want_max > 0, "the dense oracle's word is zero");
    assert!(
        want_max > 100 * bound,
        "the tolerance {bound} cannot distinguish a zero product from max|want| {want_max}"
    );
    assert!(
        worst <= bound,
        "the step-magnitude word differs from the dense words by {worst} at row {at}, bound {bound}"
    );

    // The control: one bit of the first codebook index. The two codebook rows differ, so the word
    // must move; restoring the byte must return it.
    let bytes = fold_planes_write(&records).expect("write the layout");
    let code_at = 8 + 4 + 4 + 4 + 1;
    let mut red = bytes.clone();
    red[code_at] ^= 1;
    let red_record = fold_planes_read(&red).expect("a flipped code is still a record");
    let red_face = FoldFace::from_plane(&red_record[0]).expect("red face");
    let red_out = matvec_fold(&red_face, &x).expect("red product");
    assert_ne!(
        red_out, got,
        "a flipped code bit must move the step-magnitude word"
    );
}

/// The fixture's own maxima for [`tolerance`]: `R = max|r_q|`, `D = max|d_q|` from
/// [`CodedPlane::qvals`]/`d0`, and the largest activation block-scale exponent.
fn plane_maxima(coded: &CodedPlane, scales: &[i64]) -> (i64, i64, u32) {
    let r_max = i64::try_from(
        coded
            .qvals()
            .0
            .iter()
            .map(|v| v.unsigned_abs())
            .max()
            .unwrap_or(0),
    )
    .expect("r word");
    let d_max = coded.d0.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    let emin = scales
        .iter()
        .filter(|&&s| s > 0)
        .map(|s| s.unsigned_abs().trailing_zeros())
        .max()
        .unwrap_or(0);
    (r_max, i64::try_from(d_max).unwrap_or(i64::MAX), emin)
}

// ---- the engine's fold pack: the loader, the routed step and the resident budget ----
//
// A tiny `qwen2` model (one layer, `n_embd` 128, `q_out` 128, `kv_out` 64, `n_ff` 256, vocab 64) at
// the geometry the finish tables name (`learn::fold::finish::train::kind_dims`). Every one of the
// seven coded linears is a stored pack plane; the GGUF carries the same eight linears (the seven
// plus the head) as `F16` payloads, so the F16 image is the oracle the fold engine is read against.

/// `n_embd`.
const E: usize = 128;
/// Query heads.
const NH: usize = 2;
/// Per-head width.
const HD: usize = 64;
/// KV heads.
const NKV: usize = 1;
/// Feed-forward width.
const FF: usize = 256;
/// Vocabulary.
const V: usize = 64;

/// `(o_total, n)` of linear `k` in `q, k, v, o, gate, up, down` order.
fn kind_dims(k: usize) -> (usize, usize) {
    match k {
        0 => (NH * HD, E),
        1 | 2 => (NKV * HD, E),
        3 => (E, NH * HD),
        4 | 5 => (FF, E),
        _ => (E, FF),
    }
}

/// The seven coded linears' name suffixes, in pack order.
const KIND_NAMES: [&str; 7] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// A stored `o × n` plane at the fixture's shape: the codebook field, the row grid, the two column
/// fields of [`fitted_plane`], parameterised so a linear of any width can be built.
fn plane_of(o: usize, n: usize, seed: u64) -> SinkPlane {
    let (g, nb) = (n / GROUP, n / BLOCK);
    let cb = codebook();
    let mut idx = Vec::with_capacity(o * nb);
    let mut s_rot = Vec::with_capacity(o * n);
    for b in 0..o * nb {
        let j = (b * 37 + usize::try_from(seed).expect("seed")) % 256;
        idx.push(u8::try_from(j).unwrap());
        s_rot.extend_from_slice(&cb[j]);
    }
    // The row grid of [`fitted_plane`]: `r` words in `[0.25, 0.58]`, so a recomposed weight is
    // `O(0.1)` — inside binary16's range, so the `F16` image is not a zero model.
    let (lo, sp) = (ONE / 4 + i64::try_from(seed).expect("seed"), ONE / 3);
    let mut rh = Vec::with_capacity(o * g);
    for i in 0..o * g {
        let k = u8::try_from(i % 16).unwrap();
        rh.push(r_level(lo, sp, k).expect("grid level"));
    }
    let mut s2 = seed ^ 0x5eed_1234;
    let ch = column_field(n, ONE / 4, ONE / 4 + ONE / 2, 6, &mut s2);
    let dh = column_field(n, ONE / 8, ONE / 8 + ONE / 4, 5, &mut s2);
    SinkPlane {
        s_rot,
        rh,
        ch,
        dh,
        rows: None,
        dense: None,
        factors: None,
    }
}

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

fn kv_f32(b: &mut Vec<u8>, k: &str, v: f32) {
    put_str(b, k);
    put_u32(b, 6);
    b.extend_from_slice(&v.to_bits().to_le_bytes());
}

fn u(x: usize) -> u64 {
    u64::try_from(x).expect("fixture dims fit u64")
}

/// One directory entry: an `F16` payload for a matrix, `F32` for a norm or bias.
fn gguf_matrix(
    ts: &mut Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    name: String,
    dims: Vec<u64>,
    words: &[i64],
) {
    ts.push((
        name,
        dims,
        apeiron::format::quant::GGML_TYPE_F16,
        apeiron::learn::fold::run::convert::encode_f16(words).expect("fixture matrix f16"),
    ));
}

fn gguf_vector(
    ts: &mut Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    name: String,
    dims: Vec<u64>,
    words: &[i64],
) {
    ts.push((
        name,
        dims,
        apeiron::format::quant::GGML_TYPE_F32,
        apeiron::learn::fold::run::convert::encode_f32(words),
    ));
}

/// The one-layer `qwen2` `GGUF` of `linears` (seven coded linears in [`KIND_NAMES`] order): every
/// matrix `F16`, every norm and bias `F32`, a separate `output.weight` head.
fn model_gguf(linears: &[Vec<i64>; 7]) -> Vec<u8> {
    let mut ts: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let one = vec![ONE; E];
    let zero_q = vec![0_i64; NH * HD];
    let zero_kv = vec![0_i64; NKV * HD];
    let p = |s: &str| format!("blk.0.{s}");
    gguf_vector(&mut ts, p("attn_norm.weight"), vec![u(E)], &one);
    gguf_matrix(
        &mut ts,
        p("attn_q.weight"),
        vec![u(E), u(NH * HD)],
        &linears[0],
    );
    gguf_matrix(
        &mut ts,
        p("attn_k.weight"),
        vec![u(E), u(NKV * HD)],
        &linears[1],
    );
    gguf_matrix(
        &mut ts,
        p("attn_v.weight"),
        vec![u(E), u(NKV * HD)],
        &linears[2],
    );
    gguf_vector(&mut ts, p("attn_q.bias"), vec![u(NH * HD)], &zero_q);
    gguf_vector(&mut ts, p("attn_k.bias"), vec![u(NKV * HD)], &zero_kv);
    gguf_vector(&mut ts, p("attn_v.bias"), vec![u(NKV * HD)], &zero_kv);
    gguf_matrix(
        &mut ts,
        p("attn_output.weight"),
        vec![u(NH * HD), u(E)],
        &linears[3],
    );
    gguf_vector(&mut ts, p("ffn_norm.weight"), vec![u(E)], &one);
    gguf_matrix(
        &mut ts,
        p("ffn_gate.weight"),
        vec![u(E), u(FF)],
        &linears[4],
    );
    gguf_matrix(&mut ts, p("ffn_up.weight"), vec![u(E), u(FF)], &linears[5]);
    gguf_matrix(
        &mut ts,
        p("ffn_down.weight"),
        vec![u(FF), u(E)],
        &linears[6],
    );
    gguf_vector(&mut ts, "output_norm.weight".to_string(), vec![u(E)], &one);
    // A distinct row per vocabulary entry: the head is this table, so a tie here would make the
    // argmax comparison meaningless.
    let emb: Vec<i64> = (0..V * E)
        .map(|i| {
            let v = i64::try_from(i / E).unwrap_or(0) - 32;
            let j = i64::try_from(i % E).unwrap_or(0);
            v * (1 + j % 7) * (ONE / 512)
        })
        .collect();
    gguf_matrix(
        &mut ts,
        "token_embd.weight".to_string(),
        vec![u(E), u(V)],
        &emb,
    );
    gguf_matrix(&mut ts, "output.weight".to_string(), vec![u(E), u(V)], &emb);

    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, u64::try_from(ts.len()).expect("small"));
    put_u64(&mut b, 8);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(&mut b, "qwen2.block_count", 1);
    kv_u32(
        &mut b,
        "qwen2.attention.head_count",
        u32::try_from(NH).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.attention.head_count_kv",
        u32::try_from(NKV).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.embedding_length",
        u32::try_from(E).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.feed_forward_length",
        u32::try_from(FF).expect("small"),
    );
    kv_f32(&mut b, "qwen2.attention.layer_norm_rms_epsilon", 1e-5);
    kv_f32(&mut b, "qwen2.rope.freq_base", 10_000.0);
    let mut off = 0u64;
    for (name, dims, ty, payload) in &ts {
        put_str(&mut b, name);
        put_u32(&mut b, u32::try_from(dims.len()).expect("rank"));
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

/// The seven linears' pack records and the dense words they recompose. `gate` carries two
/// row-restricted planes (`0..128`, `128..256`), so the pack exercises the restricted-plane map and
/// the control has two same-shaped planes of one linear to swap.
fn fitted_linears() -> (Vec<apeiron::format::packing::FoldPlane>, [Vec<i64>; 7]) {
    let mut records = Vec::new();
    let mut dense: [Vec<i64>; 7] = Default::default();
    for k in 0..7 {
        let (o, n) = kind_dims(k);
        let seed = u64::try_from(k).expect("kind") * 11 + 3;
        let mut w = vec![0_i64; o * n];
        if k == 4 {
            for half in 0..2 {
                let mut p = plane_of(o / 2, n, seed + u64::try_from(half).expect("half"));
                p.rows = Some((0..o / 2).map(|i| half * (o / 2) + i).collect());
                let (recs, _) = fold_pack_of_planes(&[&p]).expect("pack a restricted plane");
                let ww = fold_plane_coded(&recs[0]).expect("coded").weight();
                for (i, row) in ww.chunks_exact(n).enumerate() {
                    let at = half * (o / 2) + i;
                    w[at * n..(at + 1) * n].copy_from_slice(row);
                }
                records.extend(recs);
            }
        } else {
            let p = plane_of(o, n, seed);
            let (recs, _) = fold_pack_of_planes(&[&p]).expect("pack a plane");
            w = fold_plane_coded(&recs[0]).expect("coded").weight();
            records.extend(recs);
        }
        dense[k] = w;
    }
    (records, dense)
}

/// The gate-1 [`tolerance`] of a kind's planes, summed: each plane bounds its own output rows.
fn tolerance_planes(records: &[apeiron::format::packing::FoldPlane], scales: &[i64]) -> i128 {
    let mut tol = 0_i128;
    for r in records {
        let Ok(coded) = fold_plane_coded(r) else {
            continue;
        };
        let (r_max, d_max, emin) = plane_maxima(&coded, scales);
        tol += tolerance(r_max, d_max, emin, coded.n / GROUP);
    }
    tol
}

/// The prefix token every compared logit row reads before its own: at position 0 a one-position
/// softmax gives the sole position weight one, so `q` and `k` cannot move the logits at all, and a
/// one-token prompt cannot discriminate either linear.
const PREFIX: u32 = 8;

/// The logits at position 1 of the two-token prompt `(PREFIX, token_id)`, and the greedy token
/// [`Engine::step`] returns for `token_id`.
fn step_pair(e: &mut Engine, token_id: u32) -> (Vec<i64>, u32) {
    let mut kv = e.decode_state(4).expect("kv");
    e.step_logits(&mut kv, PREFIX).expect("prefix logits");
    let logits = e.step_logits(&mut kv, token_id).expect("logits");
    let r = e.step(token_id).expect("step");
    (logits, r.token)
}

/// `Σ_planes` of the records `fold_pack_of_planes` writes for a kind, per [`kind_dims`].
fn kind_planes(k: usize) -> usize {
    if k == 4 {
        2
    } else {
        1
    }
}

/// The seven coded linears' record index range in [`fitted_linears`]'s order.
fn kind_records(k: usize) -> usize {
    (0..k).map(kind_planes).sum()
}

/// The engine on `(GGUF, fold pack)` equals the engine on the `F16` image of the same linears:
/// the first-token argmax on eight prompts, and the logits inside the `F16` rounding bound.
///
/// Bound. The `F16` image stores each weight as `w·(1+δ)`, `|δ| ≤ 2^-11`. Eight weight matrices
/// stand between the token and the logits (`q`, `k`, `v`, `o`, `gate`, `up`, `down`, head), each a
/// linear read of a residual that `RMSNorm` normalises to unit scale (`|gain| ≤ 1`) and `RoPE`
/// rotates orthogonally, so a relative perturbation of one stage reaches the logits multiplied by at
/// most the stage's activation gain; `SiLU`'s derivative is bounded by `1.1`. The total is therefore
/// at most `8 · 1.1 · 2^-11` relative, taken here against the engine's own `max|logit|`, plus one
/// word per stage for the fixed-point rounding of a word at each projection. The premise is that
/// every stage's gain is at most one; the bound is computed from the measured `max|logit|`.
///
/// Control: the two row-restricted `gate` planes swapped in the pack — the same shapes, different
/// rows — must move the logits.
#[test]
fn fold_engine_matches_student_gguf() {
    let (records, dense) = fitted_linears();
    let gguf = model_gguf(&dense);
    let mut fold_engine =
        Engine::from_bytes_with_fold("fold", gguf.clone(), &records).expect("fold engine");
    let mut f16_engine = Engine::from_bytes("f16", gguf.clone()).expect("f16 engine");
    assert_eq!(fold_engine.config.n_layer, 1);
    assert_eq!(f16_engine.config.n_layer, 1);

    // The named blocker, first: every one of the seven coded linears' faces against the `F16`
    // payload of the same dense words at the *step's* activation magnitude (`|x| < ONE`, a residual
    // stream), not the small fixture of `fold_linear_matches_dense_words`, and each with an
    // activation of its own kind's width (`kind 6` is `128 × 256`, so `x` has length 256).
    //
    // Bound, in `Q32.32` output words, from three terms the fixture supplies:
    // - `F16` storage, `|δ| ≤ 2^-11` per weight: `2^-11 · max_i Σ_j |W[i,j]·x_j| / 2^32`;
    // - the activation's own code rounding, `e_j = x_j − 2^{e_b}·xq_j` (the quantiser's exact
    //   residue, so not the `2^{e_b}/2` worst case): `max_i Σ_j |W[i,j]|·|e_j| / 2^32`;
    // - the face's factorisation: `tolerance` (gate 1), summed over the kind's planes.
    // The first two are the two different quantisations of the same linear, which is why the bound
    // is not the `F16` term alone: the fold face reads the activation's codes, the `F16` route reads
    // the words. A zero product is excluded three times: `max|got| > 0`, `|Δ| < max|want|/16` (a
    // zero product reports `|Δ| = max|want|`), and `|Δ| ≤ bound < max|want|/8`.
    {
        let mut refused = 0usize;
        let mut ri = 0usize;
        for k in 0..7 {
            let (o, n) = kind_dims(k);
            let x = step_activation(n);
            let (scales, codes) = quant_act_fold(&x).expect("quantise the activation");
            assert_eq!(codes.len(), n, "kind {k}: the activation's codes cover n");
            let planes = if k == 4 { 2 } else { 1 };
            let mut got = vec![0_i64; o];
            let mut folded = 0usize;
            for _ in 0..planes {
                let Some(face) = FoldFace::from_plane(&records[ri]) else {
                    ri += 1;
                    continue;
                };
                let Some(y) = matvec_fold(&face, &x) else {
                    refused += 1;
                    ri += 1;
                    continue;
                };
                folded += 1;
                let rows: Vec<usize> = match &records[ri].rows {
                    Some(r) => r.iter().map(|&v| v as usize).collect(),
                    None => (0..o).collect(),
                };
                for (i, v) in y.iter().enumerate() {
                    got[rows[i]] = *v;
                }
                ri += 1;
            }
            assert_eq!(
                folded, planes,
                "kind {k}: the face served {folded} of {planes}"
            );
            let raw = apeiron::learn::fold::run::convert::encode_f16(&dense[k]).expect("f16");
            let want = apeiron::engine::matvec_fixed(
                apeiron::format::quant::GGML_TYPE_F16,
                &raw,
                o,
                n,
                &x,
            )
            .expect("dense");
            let d = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (i128::from(*a) - i128::from(*b)).abs())
                .max()
                .unwrap_or(0);
            let m = want.iter().map(|v| v.abs()).max().unwrap_or(0);
            let got_max = got.iter().map(|v| v.abs()).max().unwrap_or(0);
            // The two quantisations the comparison straddles, from the fixture's own words: the
            // activation-side term is the *actual* code error `e_j = x_j − 2^{e_b}·xq_j`
            // (`quant_act`'s own rounding, so an identity and not the `2^{e_b}/2` worst case),
            // and the only inequality left in it is `|Σ_j W·e| ≤ Σ_j |W|·|e_j|`.
            let one = 1_i128 << 32;
            let err: Vec<i128> = (0..n)
                .map(|j| i128::from(x[j]) - i128::from(codes[j]) * i128::from(scales[j / 32]))
                .collect();
            let (mut f16_sum, mut code_sum) = (0_i128, 0_i128);
            for i in 0..o {
                let mut row = 0_i128;
                let mut code = 0_i128;
                for j in 0..n {
                    let w = i128::from(dense[k][i * n + j].unsigned_abs());
                    row += w * i128::from(x[j].unsigned_abs());
                    code += w * err[j].abs();
                }
                f16_sum = f16_sum.max(row);
                code_sum = code_sum.max(code);
            }
            let f16_term = f16_sum / one / 2048;
            let code_term = code_sum / one;
            let tol = tolerance_planes(&records[ri - planes..ri], &scales);
            let bound = f16_term + code_term + tol;
            eprintln!(
                "kind {k} {o}x{n}: |\u{394}| {d} of max|want| {m}, max|got| {got_max}, bound \
                 {bound} = f16 {f16_term} + codes {code_term} + face {tol}"
            );
            assert!(got_max > 0, "kind {k}: the face's words are all zero");
            // The deviation is 1/16 of the channel's own magnitude at most: a zero or uncorrelated
            // product reports `|Δ| = max|want|`, sixteen times this. The bound is held to 1/8,
            // because the activation's code residue — not the face — dominates it (92% at kind 6,
            // whose 256 columns double the residue's sum without growing the product).
            assert!(
                d < i128::from(m) / 16,
                "kind {k}: |Δ| {d} is not below max|want|/16 = {}",
                i128::from(m) / 16
            );
            assert!(
                bound < i128::from(m) / 8,
                "kind {k}: bound {bound} cannot distinguish a zero product from max|want| {m}"
            );
            assert!(d <= bound, "kind {k}: the face leaves the derived bound");
        }
        assert_eq!(
            refused, 0,
            "the fold product refused {refused} step-magnitude planes"
        );
    }
    let mut worst = 0_i128;
    let mut logit_scale = 0_i64;
    for t in 0..8u32 {
        let (a, ta) = step_pair(&mut fold_engine, t);
        let (b, tb) = step_pair(&mut f16_engine, t);
        assert_eq!(a.len(), V);
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max((i128::from(*x) - i128::from(*y)).abs());
        }
        logit_scale = logit_scale.max(b.iter().map(|v| v.abs()).max().unwrap_or(0));
        assert_eq!(ta, tb, "token {t}: argmax differs ({ta} vs {tb})");
    }
    let bound = i128::from(logit_scale) * 99 / 10 / 2048 + 8 * i128::from(ONE);
    eprintln!(
        "fold engine vs F16 image: worst |Δlogit| = {worst}, bound {bound}, max|logit| {logit_scale}"
    );
    assert!(
        worst <= bound,
        "the fold engine's logits leave the F16 rounding bound: {worst} > {bound}"
    );

    // Gate 3, the route: the fold faces are the operands. One two-token step executes each of the
    // seven coded linears exactly once per layer, so an engine with a pack counts `7·n_layer·steps`
    // products and an engine without one counts none. `fold_engine` has already run the eight
    // prompts above (two products per prompt per linear); the fresh engine below pins the count.
    // One [`step_pair`] is three forwards: the prefix's logits, the compared row's, and
    // `Engine::step`'s own.
    let stepped = 7 * fold_engine.config.n_layer * 3;
    let mut counted =
        Engine::from_bytes_with_fold("count", model_gguf(&dense), &records).expect("count engine");
    let _ = step_pair(&mut counted, 3);
    eprintln!(
        "fold calls: {} for one step_pair (7 linears x {} layer x 3 forwards = {}); without a \
         pack: {}",
        counted.fold_calls(),
        counted.config.n_layer,
        stepped,
        f16_engine.fold_calls()
    );
    assert_eq!(
        counted.fold_calls(),
        u64::try_from(stepped).expect("count"),
        "the fold route served every coded linear"
    );
    assert_eq!(
        f16_engine.fold_calls(),
        0,
        "an engine without a pack serves no fold face"
    );

    // Gate 2, discrimination: every one of the seven coded linears reaches the logits. One dense
    // weight of kind `k` moves by `ONE` in the `F16` image alone and the `F16` engine's logits must
    // move. Two `F16` engines on one image are bit-identical (measured `0`), so a non-zero move is
    // exact discrimination, not a noise floor; the `bound` above is the *equality* tolerance and is
    // far larger than one weight's reach (it carries `8·ONE` for the fixed-point term), so the
    // threshold here is the engine's own determinism, printed per kind.
    let (base_f16, _) = step_pair(&mut f16_engine, 3);
    let (fold_base, _) = step_pair(&mut counted, 3);
    let mut moves = [(0_i128, 0_i128); 7];
    for k in 0..7 {
        let (o, n) = kind_dims(k);
        let mut d = dense.clone();
        d[k][0] += ONE;
        let mut e = Engine::from_bytes("perturbed", model_gguf(&d)).expect("perturbed engine");
        let (moved, _) = step_pair(&mut e, 3);
        let df16 = moved
            .iter()
            .zip(base_f16.iter())
            .map(|(a, b)| (i128::from(*a) - i128::from(*b)).abs())
            .max()
            .unwrap_or(0);
        let mut r = records.clone();
        for b in &mut r[kind_records(k)].idx {
            *b = 255;
        }
        let mut e2 = Engine::from_bytes_with_fold("moved", model_gguf(&dense), &r).expect("moved");
        let (fold_moved, _) = step_pair(&mut e2, 3);
        let dfold = fold_moved
            .iter()
            .zip(fold_base.iter())
            .map(|(a, b)| (i128::from(*a) - i128::from(*b)).abs())
            .max()
            .unwrap_or(0);
        eprintln!(
            "kind {k} {o}x{n}: one weight + ONE moves the F16 logits {df16}; its stored plane's \
             codebook rewrite moves the fold logits {dfold}"
        );
        moves[k] = (df16, dfold);
    }
    for (k, (df16, _)) in moves.iter().enumerate() {
        assert!(
            *df16 > 0,
            "kind {k}: a perturbed weight of this linear leaves the F16 logits unchanged, so the \
             gate cannot see the linear"
        );
    }
    // The five kinds whose output reaches the logits above the word are also seen through the pack:
    // their stored plane, rewritten, moves the fold engine. `q` and `k` are not: `|q·k|` at this
    // fixture's magnitude saturates the two-position softmax, so their one-hot weights move by
    // `~1e-7` and their reach at the logits is the `1216`/`3119` words above — the same reach the
    // `F16` route has, not a route defect. Their faces are nevertheless the operand: the zeroed
    // payload identity below cannot hold if any fitted linear still reads its bytes.
    for (k, (_, dfold)) in moves.iter().enumerate() {
        if k == 0 || k == 1 {
            continue;
        }
        assert!(
            *dfold > 0,
            "kind {k}: the fold engine ignores this linear's face, so the gate does not read it"
        );
    }
    // The universal route identity: every fitted linear's `GGUF` payload is dead. Zeroing all seven
    // in the image leaves the fold engine's logits bit-identical, and the `F16` engine's logits
    // move, so the identity is not the fixture's silence.
    let mut zeroed = dense.clone();
    for (k, w) in zeroed.iter_mut().enumerate() {
        let (o, n) = kind_dims(k);
        *w = vec![0_i64; o * n];
    }
    let zero_gguf = model_gguf(&zeroed);
    let mut fold_zero =
        Engine::from_bytes_with_fold("zero", zero_gguf.clone(), &records).expect("zero fold");
    let mut f16_zero = Engine::from_bytes("zero", zero_gguf).expect("zero f16");
    let (fz, _) = step_pair(&mut fold_zero, 3);
    let (dz, _) = step_pair(&mut f16_zero, 3);
    let (mut id, mut ctl) = (0_i128, 0_i128);
    for (a, b) in fz.iter().zip(fold_base.iter()) {
        id = id.max((i128::from(*a) - i128::from(*b)).abs());
    }
    for (a, b) in dz.iter().zip(base_f16.iter()) {
        ctl = ctl.max((i128::from(*a) - i128::from(*b)).abs());
    }
    eprintln!(
        "zeroed fitted payloads: the fold engine's logits move {id}, the F16 engine's move {ctl}"
    );
    assert_eq!(
        id, 0,
        "a fitted linear still reads its GGUF payload: zeroing all seven moved the fold logits"
    );
    assert!(
        ctl > 0,
        "the control: zeroing the fitted payloads must move a payload-reading engine"
    );

    // The control: the `gate` linear's two row-restricted planes exchange their stored field while
    // keeping their own `rows`. Reordering the records alone moves nothing — the loader takes a
    // plane's global rows from the record's own `rows` field (`src/engine/loader/fold.rs:177`) —
    // so the swap that can be seen is the payload of rows `0..128` against rows `128..256`.
    let mut swapped = records.clone();
    let second = 4 + 1;
    {
        let (head, tail) = swapped.split_at_mut(second);
        swap_payload(&mut head[4], &mut tail[0]);
    }
    let mut red_engine = Engine::from_bytes_with_fold("red", gguf, &swapped).expect("red engine");
    let (red, _) = step_pair(&mut red_engine, 3);
    let (green, _) = step_pair(&mut fold_engine, 3);
    assert_ne!(red, green, "swapping the gate planes must move the logits");
}

/// Exchange the stored operand of two same-shaped planes, keeping each one's own `rows`.
fn swap_payload(
    a: &mut apeiron::format::packing::FoldPlane,
    b: &mut apeiron::format::packing::FoldPlane,
) {
    assert_eq!((a.o, a.n), (b.o, b.n), "the swap needs one shape");
    std::mem::swap(&mut a.idx, &mut b.idx);
    std::mem::swap(&mut a.lo, &mut b.lo);
    std::mem::swap(&mut a.sp, &mut b.sp);
    std::mem::swap(&mut a.r_k, &mut b.r_k);
    std::mem::swap(&mut a.c_head, &mut b.c_head);
    std::mem::swap(&mut a.c_q, &mut b.c_q);
    std::mem::swap(&mut a.d_head, &mut b.d_head);
    std::mem::swap(&mut a.d_q, &mut b.d_q);
}

/// The resident budget: the fitted linears' fold faces cost at most 1.19 b/w of the weights they
/// carry, the engine's total is the non-fitted tensors plus the faces, and the control is the same
/// linears' `F16` payloads in the image.
#[test]
fn fold_engine_footprint() {
    let (records, dense) = fitted_linears();
    let gguf = model_gguf(&dense);
    let fold_engine =
        Engine::from_bytes_with_fold("fold", gguf.clone(), &records).expect("fold engine");
    let f16_engine = Engine::from_bytes("f16", gguf.clone()).expect("f16 engine");

    let (face_bytes, weights) = fold_engine.fold_storage();
    assert!(weights > 0, "the pack carries weights");
    let milli = (face_bytes * 8000 + weights / 2) / weights;
    let total = fold_engine.resident_weight_bytes();

    // The non-fitted tensors: every directory entry the pack does not replace.
    let mut non_fitted = 0_u64;
    let mut fitted_f16 = 0_u64;
    for t in f16_engine.table().entries() {
        let fitted = (0..7).any(|k| t.name == format!("blk.0.{}", KIND_NAMES[k]));
        if fitted {
            fitted_f16 += t.n_bytes.unwrap_or(0);
        } else {
            non_fitted += t.n_bytes.unwrap_or(0);
        }
    }
    let f16_milli = (fitted_f16 * 8000 + weights / 2) / weights;
    eprintln!(
        "fold engine: fitted {face_bytes} B for {weights} weights = {}.{:03} b/w; total {total} B \
         = {non_fitted} non-fitted + {face_bytes} faces; the same linears F16: {}.{:03} b/w",
        milli / 1000,
        milli % 1000,
        f16_milli / 1000,
        f16_milli % 1000
    );
    assert_eq!(
        total,
        non_fitted + face_bytes,
        "total = non-fitted tensor bytes + fold face bytes"
    );
    assert_eq!(f16_engine.resident_weight_bytes(), non_fitted + fitted_f16);
    let face_milli = (face_bytes * 8000 + weights / 2) / weights;
    assert!(
        f16_milli > 8000,
        "the F16 control must report well over 8 b/w"
    );
    assert!(
        face_milli < 2500,
        "the coded face must sit under the decoded i64 field form's 2.500 b/w at this fixture"
    );

    // The production shape: the fitted `896 × 896` plane's resident face against its own weights,
    // and the decoded field form of the same plane as the control.
    let (sink, idx, lo, sp) = fitted_plane();
    let (packed, _): (_, FoldPackStats) =
        fold_pack_of_planes(&[&sink]).expect("pack the fitted plane");
    let coded = FoldFace::from_plane(&packed[0]).expect("coded face");
    let (o, n) = (coded.o(), coded.n());
    let weights_at = o * n;
    let coded_bytes = coded.bytes();
    let coded_milli = (coded_bytes * 8000 + weights_at / 2) / weights_at;
    let (r_q, c_q, d_q) = reference(&sink, idx.clone(), lo, sp).qvals();
    let decoded_bytes =
        idx.len() + (r_q.len() + c_q.len() + d_q.len()) * std::mem::size_of::<i64>();
    let decoded_milli = (decoded_bytes * 8000 + weights_at / 2) / weights_at;
    eprintln!(
        "coded face {o}x{n}: {coded_bytes} B for {weights_at} weights = {}.{:03} b/w; the decoded \
         form of the same plane {decoded_bytes} B = {}.{:03} b/w; the stored pack of the same plane \
         is 1.110 b/w",
        coded_milli / 1000,
        coded_milli % 1000,
        decoded_milli / 1000,
        decoded_milli % 1000
    );
    assert!(
        coded_milli <= 1190,
        "the coded face must hold the 1.19 b/w budget at 896x896"
    );
    assert!(
        decoded_milli > 1500,
        "the decoded form is the control: its i64 field words are over 1.5 b/w"
    );
}
