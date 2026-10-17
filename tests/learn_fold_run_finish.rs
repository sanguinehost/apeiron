//! `tests/learn_fold_run_finish.rs` — the fold fit wired to the finishing P-step: every fitted
//! `TensorFit` sink plane rebuilds into the finisher's `CodedPlane`s and reassembles to `fit.wp`
//! bit-for-bit (quantized and unquantized fronts), a synthetic `ModelResult` drives one
//! `finish_model` step that strictly lowers the batch mean KL, and the plane / loss digests are
//! pinned. Grade `tested`; nothing here is `proved`.
//!
//! Tolerance: zero words. [`CodedPlane::weight_unquantized`] uses the latent bases `(r0, c0, d0)`
//! and [`CodedPlane::weight`] uses their canonicalised forms; `from_pack` stores the pack scales as
//! their quantizer fixed points, so both forwards coincide and both equal `SinkPlane`'s stored
//! reconstruction bit-for-bit.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps its bits to f64 and to codebook indices; `k`/`o`/`n` are the \
              fit's own kind and shape names; the exact comparison is the round-trip contract (a \
              quantizer fixed point has no tolerance); the fixture enumerates the seven kinds in \
              one body."
)]

use apeiron::learn::fold::finish::train::{assemble, assemble_unquantized, kl_mean, Frozen};
use apeiron::learn::fold::fit::icm::{codebook, BLOCK};
use apeiron::learn::fold::fit::quant::{quant_c_u8, quant_r_hier};
use apeiron::learn::fold::fit::steps::GROUP;
use apeiron::learn::fold::layer::driver::ModelResult;
use apeiron::learn::fold::layer::tensor::{recon, SinkPlane, TensorBits, TensorFit, TensorKind};
use apeiron::learn::fold::run::finish::{finish_model, planes_from_fits, tunables_from_weights};
use apeiron::learn::grad::model::{forward, Config, LayerWeights, Weights};
use apeiron::learn::solve::q32::{to_grid, GRID_BOUND};

/// Trunk layers of the tiny configuration.
const L: usize = 2;
/// Residual width (a whole number of 128-blocks, so every plane column count is a multiple of
/// [`GROUP`]).
const E: usize = 128;
/// Query heads; `n_head · head_dim = n_embd`.
const NH: usize = 2;
/// KV heads (GQA).
const NKV: usize = 1;
/// Per-head width.
const HD: usize = 64;
/// Feed-forward width.
const FF: usize = 256;
/// Vocabulary.
const V: usize = 64;
/// Tokens in the fixed batch.
const T: usize = 16;
/// `round(1e-5 · 2^32)`, the `RMSNorm` epsilon.
const EPS_Q32: i64 = 42_950;
/// `RoPE` theta.
const THETA: f64 = 10_000.0;
/// Coded linears per layer.
const NKINDS: usize = 7;

/// `round(v·2^32)` for a test fixture value.
fn q32(v: f64) -> i64 {
    (v * 4_294_967_296.0).round() as i64
}

/// The f64 fixture as Q32.32 words, the quantiser's input at the test's own data.
fn words(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("fixture on the grid"))
        .collect()
}

/// The tiny Qwen2 geometry.
fn cfg() -> Config {
    Config {
        n_layer: L,
        n_head: NH,
        n_head_kv: NKV,
        n_embd: E,
        n_ff: FF,
        vocab: V,
        head_dim: HD,
        eps: EPS_Q32,
        rope_theta: THETA,
    }
}

/// `(o_total, n)` of linear kind `k`, in `q, k, v, o, gate, up, down` order.
fn dims(k: usize) -> (usize, usize) {
    match k {
        0 => (NH * HD, E),
        1 | 2 => (NKV * HD, E),
        3 => (E, NH * HD),
        4 | 5 => (FF, E),
        _ => (E, FF),
    }
}

/// The [`TensorKind`] of `k`, in `q, k, v, o, gate, up, down` order.
fn kind_of(k: usize) -> TensorKind {
    match k {
        0 => TensorKind::Q,
        1 => TensorKind::K,
        2 => TensorKind::V,
        3 => TensorKind::O,
        4 => TensorKind::Gate,
        5 => TensorKind::Up,
        _ => TensorKind::Down,
    }
}

/// Deterministic 64-bit linear congruential generator (fixed seed, no `rand` crate).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform in `[-0.5, 0.5)`.
    fn sym(&mut self) -> f64 {
        self.unit() - 0.5
    }
}

/// The exact code field from a list of codebook indices, `o × n` row-major, as Q32.32 words.
fn code_field(idx: &[u8], o: usize, n: usize) -> Vec<i64> {
    let cb = codebook();
    let nb = n / BLOCK;
    let mut s = vec![0_i64; o * n];
    for oi in 0..o {
        for m in 0..nb {
            let j = idx[oi * nb + m] as usize;
            for k in 0..BLOCK {
                s[oi * n + m * BLOCK + k] = cb[j][k];
            }
        }
    }
    s
}

/// One stored pack at the quantizers' fixed points: `s_rot` a codebook field, `rh` the hierarchical
/// row scale, `ch`/`dh` the 6-/5-bit column scales.
struct Pack {
    s: Vec<i64>,
    rh: Vec<i64>,
    ch: Vec<i64>,
    dh: Vec<i64>,
}

fn make_pack(rng: &mut Lcg, o: usize, n: usize) -> Pack {
    let idx: Vec<u8> = (0..o * (n / BLOCK))
        .map(|_| (rng.next_u64() % 256) as u8)
        .collect();
    let s = code_field(&idx, o, n);
    let g = n / GROUP;
    let r_raw: Vec<f64> = (0..o * g).map(|_| 0.5 + 0.5 * rng.unit()).collect();
    let rh = quant_r_hier(&words(&r_raw), o, g).expect("row scales");
    let c_raw: Vec<f64> = (0..n).map(|_| 0.1 + 0.1 * rng.unit()).collect();
    let ch = quant_c_u8(&words(&c_raw), 6).expect("column scales").0;
    let d_raw: Vec<f64> = (0..n).map(|_| 0.5 + 0.5 * rng.unit()).collect();
    let dh = quant_c_u8(&words(&d_raw), 5).expect("hadamard scales").0;
    Pack {
        s,
        rh: rh.rq,
        ch,
        dh,
    }
}

/// The [`SinkPlane`] of a pack: a full plane with no rows, dense addend or factors.
fn sink(p: &Pack) -> SinkPlane {
    SinkPlane {
        s_rot: p.s.clone(),
        rh: p.rh.clone(),
        ch: p.ch.clone(),
        dh: p.dh.clone(),
        rows: None,
        dense: None,
        factors: None,
    }
}

/// The seven [`TensorFit`]s of one layer, in `q, k, v, o, gate, up, down` order: a pack plane per
/// kind, the `q_mlp_trans` rows plane on `gate`/`up`, the dense corrector on `q`/`down`, the
/// `q`/`k`/`v` bias delta and the exact stored weight `recon(planes)`.
fn layer_fits(rng: &mut Lcg) -> Vec<TensorFit> {
    let mut out: Vec<TensorFit> = Vec::with_capacity(NKINDS);
    for k in 0..NKINDS {
        let (o, n) = dims(k);
        let mut planes: Vec<SinkPlane> = vec![sink(&make_pack(rng, o, n))];
        let mut top_rows: Option<Vec<usize>> = None;
        if k == 4 || k == 5 {
            let top: Vec<usize> = vec![1, 5, 9, 13];
            let mut p2 = sink(&make_pack(rng, top.len(), n));
            p2.rows = Some(top.clone());
            planes.push(p2);
            top_rows = Some(top);
        }
        if k == 0 || k == 6 {
            let dense: Vec<i64> = (0..o * n).map(|_| q32(0.02 * rng.sym())).collect();
            planes.push(SinkPlane {
                s_rot: Vec::new(),
                rh: Vec::new(),
                ch: Vec::new(),
                dh: Vec::new(),
                rows: None,
                dense: Some(dense),
                factors: None,
            });
        }
        let wp = recon(&planes, o, n).expect("stored reconstruction");
        let db = if k < 3 {
            Some((0..o).map(|_| q32(0.05 * rng.sym())).collect())
        } else {
            None
        };
        out.push(TensorFit {
            kind: kind_of(k),
            wp,
            planes,
            bits: TensorBits::default(),
            nb: None,
            top_rows,
            db,
        });
    }
    out
}

/// One layer's `LayerWeights` at the geometry's widths (only the shapes and biases are read by the
/// tunable build).
fn layer_weights(rng: &mut Lcg) -> LayerWeights {
    let jitter =
        |rng: &mut Lcg, n: usize| -> Vec<i64> { (0..n).map(|_| q32(0.01 * rng.sym())).collect() };
    LayerWeights {
        attn_norm: (0..E).map(|_| q32(0.8 + 0.2 * rng.unit())).collect(),
        wq: jitter(rng, NH * HD * E),
        bq: Some(jitter(rng, NH * HD)),
        wk: jitter(rng, NKV * HD * E),
        bk: Some(jitter(rng, NKV * HD)),
        wv: jitter(rng, NKV * HD * E),
        bv: Some(jitter(rng, NKV * HD)),
        wo: jitter(rng, E * NH * HD),
        ffn_norm: (0..E).map(|_| q32(0.8 + 0.2 * rng.unit())).collect(),
        w_gate: jitter(rng, FF * E),
        w_up: jitter(rng, FF * E),
        w_down: jitter(rng, E * FF),
    }
}

/// A synthetic [`ModelResult`]: the student layers, the final norm gain and the per-layer fits.
fn model_result(rng: &mut Lcg) -> ModelResult {
    let layers: Vec<LayerWeights> = (0..L).map(|_| layer_weights(rng)).collect();
    let fnorm: Vec<i64> = (0..E).map(|_| q32(0.8 + 0.2 * rng.unit())).collect();
    let fits: Vec<Vec<TensorFit>> = (0..L).map(|_| layer_fits(rng)).collect();
    ModelResult {
        layers,
        fnorm,
        bits: TensorBits::default(),
        fits,
    }
}

/// A deterministic nonzero nudge of every projection, embedding and head element; the teacher's
/// weights. `amp` is in Q32.32 units.
fn perturb(w: &mut Weights, amp: i64) {
    let mut s = 0x9E37_79B9_7F4A_7C15_u64;
    let span = u64::try_from(2 * amp).expect("span");
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % span).cast_signed() - amp
    };
    for lw in &mut w.layers {
        for buf in [&mut lw.wq, &mut lw.wk, &mut lw.wv, &mut lw.wo] {
            for x in buf.iter_mut() {
                *x += next();
            }
        }
        for buf in [&mut lw.w_gate, &mut lw.w_up, &mut lw.w_down] {
            for x in buf.iter_mut() {
                *x += next();
            }
        }
    }
    for x in &mut w.token_embd {
        *x += next();
    }
    for x in &mut w.lm_head {
        *x += next();
    }
}

/// FNV-1a over a word's eight little-endian bytes, the recorded-witness hash.
fn fnv_word(d: &mut u64, v: i64) {
    for b in v.to_le_bytes() {
        *d ^= u64::from(b);
        *d = d.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// FNV-1a over a buffer, length-prefixed so a truncation moves the digest.
fn fnv_buf(d: &mut u64, xs: &[i64]) {
    fnv_word(d, i64::try_from(xs.len()).unwrap_or(i64::MAX));
    for &v in xs {
        fnv_word(d, v);
    }
}

const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a over every rebuilt plane's shape, code field, row list, grid, bases and stored weight.
fn planes_digest(planes: &apeiron::learn::fold::finish::train::Planes) -> u64 {
    let mut d = FNV0;
    for row in &planes.layers {
        for lin in row {
            fnv_word(&mut d, i64::try_from(lin.planes.len()).expect("planes"));
            fnv_word(&mut d, i64::try_from(lin.o_total).expect("rows"));
            fnv_word(&mut d, i64::try_from(lin.n).expect("cols"));
            for p in &lin.planes {
                fnv_word(&mut d, i64::try_from(p.o).expect("o"));
                fnv_word(&mut d, i64::try_from(p.n).expect("n"));
                fnv_buf(
                    &mut d,
                    &p.idx.iter().map(|&b| i64::from(b)).collect::<Vec<_>>(),
                );
                fnv_word(&mut d, i64::from(p.s_fix.is_some()));
                if let Some(s) = &p.s_fix {
                    fnv_buf(&mut d, s);
                }
                if let Some(rows) = &p.rows {
                    fnv_buf(
                        &mut d,
                        &rows
                            .iter()
                            .map(|&r| i64::try_from(r).expect("row"))
                            .collect::<Vec<_>>(),
                    );
                }
                for v in [&p.lo, &p.sp, &p.r0, &p.c0, &p.d0, &p.pa, &p.pb, &p.pd] {
                    fnv_buf(&mut d, v);
                }
                fnv_buf(&mut d, &p.weight());
            }
            if let Some(e) = &lin.extra {
                fnv_buf(&mut d, e);
            }
        }
    }
    d
}

/// (b): one `finish_model` step lowers the batch mean KL on the tiny model.
#[test]
fn one_finish_step_lowers_kl() {
    let cfg = cfg();
    let mut rng = Lcg::new(0xB2);
    let res = model_result(&mut rng);
    let planes = planes_from_fits(&cfg, &res.fits).expect("planes");
    let tun = tunables_from_weights(&res.layers, &res.fnorm, &res.fits).expect("tunables");
    let emb: Vec<i64> = (0..V * E).map(|_| q32(0.1 * rng.sym())).collect();
    let head: Vec<i64> = (0..V * E).map(|_| q32(0.1 * rng.sym())).collect();
    let teacher_frozen = Frozen::new(&emb, &head);
    let mut teacher = assemble(&cfg, &planes, &tun, &teacher_frozen).expect("student");
    perturb(&mut teacher, 1 << 26);
    let frozen = Frozen::new(&teacher.token_embd, &teacher.lm_head);
    let tokens: Vec<u32> = (0..T).map(|_| (rng.next_u64() % V as u64) as u32).collect();
    let (tlogits, _) = forward(&teacher, &cfg, &tokens).expect("teacher logits");

    let before = {
        let w = assemble(&cfg, &planes, &tun, &frozen).expect("student");
        let (logits, _) = forward(&w, &cfg, &tokens).expect("student logits");
        kl_mean(&cfg, &logits, &tlogits).expect("kl").0
    };
    assert!(before > 0, "the teacher and student must differ");

    let fin =
        finish_model(&cfg, &res, &teacher, std::slice::from_ref(&tokens), 1, 0).expect("finish");
    assert_eq!(fin.loss.len(), 1, "one step, one loss word");
    assert_eq!(
        fin.loss[0], before,
        "the step reports the pre-update KL of the same student"
    );
    let (logits, _) = forward(&fin.weights, &cfg, &tokens).expect("finished logits");
    let after = kl_mean(&cfg, &logits, &tlogits).expect("kl after").0;
    assert!(
        after < before,
        "one step must lower KL: before {before}, after {after}"
    );

    let mut d = FNV0;
    fnv_buf(&mut d, &fin.loss);
    println!(
        "loss={before}->{after} loss_digest={d:#018x} planes_digest={:#018x}",
        planes_digest(&planes)
    );
    assert_eq!(d, LOSS_DIGEST, "loss trace digest");
}

/// (a): every fit's stored weight is reproduced exactly by the rebuilt planes.
#[test]
fn planes_from_fits_reproduces_each_fit_weight_exactly() {
    let cfg = cfg();
    let mut rng = Lcg::new(0xA1);
    let res = model_result(&mut rng);
    let planes = planes_from_fits(&cfg, &res.fits).expect("planes");
    let tun = tunables_from_weights(&res.layers, &res.fnorm, &res.fits).expect("tunables");
    let emb: Vec<i64> = (0..V * E).map(|_| q32(0.1 * rng.sym())).collect();
    let head: Vec<i64> = (0..V * E).map(|_| q32(0.1 * rng.sym())).collect();
    let frozen = Frozen::new(&emb, &head);
    let wq = assemble(&cfg, &planes, &tun, &frozen).expect("quantized assemble");
    let wu = assemble_unquantized(&cfg, &planes, &tun, &frozen).expect("unquantized assemble");

    for (l, fs) in res.fits.iter().enumerate() {
        let (q, u) = (&wq.layers[l], &wu.layers[l]);
        let quantized = [&q.wq, &q.wk, &q.wv, &q.wo, &q.w_gate, &q.w_up, &q.w_down];
        let unquantized = [&u.wq, &u.wk, &u.wv, &u.wo, &u.w_gate, &u.w_up, &u.w_down];
        for (k, fit) in fs.iter().enumerate() {
            assert_eq!(
                quantized[k],
                &fit.wp,
                "layer {l} kind {}: assemble != fit.wp",
                kind_of(k) as u8
            );
            assert_eq!(
                unquantized[k],
                &fit.wp,
                "layer {l} kind {}: assemble_unquantized != fit.wp",
                kind_of(k) as u8
            );
        }
    }

    let d = planes_digest(&planes);
    println!("planes_digest={d:#018x}");
    assert_eq!(d, PLANES_DIGEST, "rebuilt plane digest");
}

/// The pinned plane and loss digests (FNV-1a over the little-endian words, length-prefixed).
const PLANES_DIGEST: u64 = 0xfc3f_0275_32c2_7602;
const LOSS_DIGEST: u64 = 0x9aa4_b6eb_ff6e_2abf;
