//! `tests/learn_fold_driver.rs` — the production `fold_gqa` layer driver: the seven tensor shapes
//! and the `q`/`k`/`v` bias update, the student-path identity `hs == parts(&lq).out`, the bit sum and
//! the hand sign count, the `g_row` helper, the slab-size dependence of the sink weights, the
//! two-layer model loop and its final norm fit, and the refusals. Grade `tested`.
//!
//! The fixture is the brief's tiny model: `hidden = 128`, `n_head = 2`, `head_dim = 64`,
//! `n_kv_head = 1`, `inter = 128` (so every stage input width is `128`), `eps = 1e-6`,
//! `theta = 1e4`, `t = 8`, `n_seq = 3`, `slab = 2` (an uneven last slab), `qmlp_rows = 16`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps integers to `f64` by division, the flat row-major index \
              arithmetic is the reference payload format, and the equality checks are exact float \
              comparisons by construction (the driver's own expressions re-evaluated); the \
              `q/k/v/o/i/j` names are the reference's vocabulary"
)]

use std::time::Instant;

use apeiron::learn::fold::layer::driver::{
    g_row_from_matrix, g_row_from_vec, quantise_layer, quantise_model, DriverCfg, DriverError,
    LayerWeights,
};
use apeiron::learn::fold::layer::stage::sink_token_weights;
use apeiron::learn::fold::layer::tensor::TensorKind;
use apeiron::learn::fold::stats::parts::{parts, LayerCfg, RopeTables};

const HID: usize = 128;
const N_HEAD: usize = 2;
const HEAD_DIM: usize = 64;
const N_KV: usize = 1;
const INTER: usize = 128;
const T: usize = 8;
const N_SEQ: usize = 3;
const SLAB: usize = 2;
const QMLP: usize = 16;
const EPS: f64 = 1e-6;
const THETA: f64 = 1e4;

/// Deterministic xorshift64 LCG.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[lo, hi)` from the top 53 bits.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        lo + (hi - lo) * u
    }
}

/// A standard-normal sample (sum of twelve uniforms minus six).
fn gauss(rng: &mut Lcg) -> f64 {
    (0..12).map(|_| rng.uniform(0.0, 1.0)).sum::<f64>() - 6.0
}

fn rand_mat(o: usize, n: usize, scale: f64, rng: &mut Lcg) -> Vec<f64> {
    (0..o * n).map(|_| scale * gauss(rng)).collect()
}

fn rand_bias(n: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..n).map(|_| 0.05 * gauss(rng)).collect()
}

fn rand_ln(n: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..n).map(|_| 1.0 + 0.02 * gauss(rng)).collect()
}

fn layer_cfg() -> LayerCfg {
    LayerCfg {
        n_head: N_HEAD,
        n_kv_head: N_KV,
        head_dim: HEAD_DIM,
        hidden: HID,
        inter: INTER,
        eps: EPS,
        theta: THETA,
    }
}

/// The production driver configuration at the test geometry, with `slab`/`qmlp_rows` overridable.
fn driver_cfg(slab: usize) -> DriverCfg {
    let mut cfg = DriverCfg::production(layer_cfg(), T, N_SEQ);
    cfg.slab = slab;
    cfg.qmlp_rows = QMLP;
    cfg
}

fn make_layer(cfg: &LayerCfg, rng: &mut Lcg) -> LayerWeights {
    let qo = cfg.n_head * cfg.head_dim;
    let kvo = cfg.n_kv_head * cfg.head_dim;
    let s = 1.0 / (cfg.hidden as f64).sqrt();
    LayerWeights {
        ln1: rand_ln(cfg.hidden, rng),
        ln2: rand_ln(cfg.hidden, rng),
        q: rand_mat(qo, cfg.hidden, s, rng),
        qb: rand_bias(qo, rng),
        k: rand_mat(kvo, cfg.hidden, s, rng),
        kb: rand_bias(kvo, rng),
        v: rand_mat(kvo, cfg.hidden, s, rng),
        vb: rand_bias(kvo, rng),
        o: rand_mat(cfg.hidden, qo, s, rng),
        gate: rand_mat(cfg.inter, cfg.hidden, s, rng),
        up: rand_mat(cfg.inter, cfg.hidden, s, rng),
        down: rand_mat(cfg.hidden, cfg.inter, s, rng),
    }
}

fn rand_hidden(rows: usize, hid: usize, scale: f64, rng: &mut Lcg) -> Vec<f64> {
    (0..rows * hid).map(|_| scale * gauss(rng)).collect()
}

/// The fixture: driver config, `RoPE` tables, the teacher layer, the teacher/student hidden states
/// and the `down` `g_row` source.
fn fixture(
    seed: u64,
    slab: usize,
) -> (
    DriverCfg,
    RopeTables,
    LayerWeights,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let cfg = driver_cfg(slab);
    let rope = RopeTables::new(&cfg.layer, T).expect("rope tables");
    let mut rng = Lcg::new(seed);
    let lt = make_layer(&cfg.layer, &mut rng);
    let ht = rand_hidden(N_SEQ * T, HID, 1.0, &mut rng);
    let hs = ht.clone();
    let ln_next = rand_ln(HID, &mut rng);
    (cfg, rope, lt, ht, hs, ln_next)
}

/// The error of a refused driver call (the `Ok` side need not be `PartialEq`).
fn unwrap_err<T: core::fmt::Debug>(r: Result<T, DriverError>) -> DriverError {
    match r {
        Ok(v) => panic!("expected an error, got {v:?}"),
        Err(e) => e,
    }
}

fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "rel_err length");
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..a.len() {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += b[i] * b[i];
    }
    (num / den.max(1e-300)).sqrt()
}

fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

/// (a) one layer: seven shape-correct finite tensors, the fit order, the bias update, the untouched
/// norms.
#[test]
fn a_layer_shapes_bias_and_norms() {
    let t0 = Instant::now();
    let (cfg, rope, lt, ht, hs_in, ln_next) = fixture(7, SLAB);
    let mut hs = hs_in.clone();
    let res = quantise_layer(&cfg, &rope, &lt, &ln_next, &ht, &mut hs).expect("layer");

    assert_eq!(res.fits.len(), 7);
    let order = [
        TensorKind::Q,
        TensorKind::K,
        TensorKind::V,
        TensorKind::O,
        TensorKind::Gate,
        TensorKind::Up,
        TensorKind::Down,
    ];
    for (i, k) in order.iter().enumerate() {
        assert_eq!(res.fits[i].kind, *k, "fit order at {i}");
    }

    let hid = HID;
    let qo = N_HEAD * HEAD_DIM;
    let kvo = N_KV * HEAD_DIM;
    let lq = &res.lq;
    assert_eq!(lq.ln1.len(), hid);
    assert_eq!(lq.ln2.len(), hid);
    assert_eq!(lq.q.len(), qo * hid);
    assert_eq!(lq.qb.len(), qo);
    assert_eq!(lq.k.len(), kvo * hid);
    assert_eq!(lq.kb.len(), kvo);
    assert_eq!(lq.v.len(), kvo * hid);
    assert_eq!(lq.vb.len(), kvo);
    assert_eq!(lq.o.len(), hid * qo);
    assert_eq!(lq.gate.len(), INTER * hid);
    assert_eq!(lq.up.len(), INTER * hid);
    assert_eq!(lq.down.len(), hid * INTER);
    for s in [
        &lq.ln1, &lq.ln2, &lq.q, &lq.qb, &lq.k, &lq.kb, &lq.v, &lq.vb, &lq.o, &lq.gate, &lq.up,
        &lq.down,
    ] {
        assert!(all_finite(s), "a fitted tensor is not finite");
    }

    // `ln1`/`ln2` are not fitted: unchanged.
    assert_eq!(lq.ln1, lt.ln1);
    assert_eq!(lq.ln2, lt.ln2);

    // `Lq.q = Wp` and `Lq.qb = Lt.qb + db`.
    assert_eq!(lq.q, res.fits[0].wp);
    let db = res.fits[0].db.as_ref().expect("q bias");
    let mut changed = false;
    for i in 0..qo {
        assert!((lq.qb[i] - (lt.qb[i] + db[i])).abs() <= 1e-12);
        if db[i].abs() > 1e-9 {
            changed = true;
        }
    }
    assert!(changed, "the q bias did not move");
    assert_eq!(lq.k, res.fits[1].wp);
    assert_eq!(lq.v, res.fits[2].wp);
    assert_eq!(lq.o, res.fits[3].wp);
    assert_eq!(lq.gate, res.fits[4].wp);
    assert_eq!(lq.up, res.fits[5].wp);
    assert_eq!(lq.down, res.fits[6].wp);

    eprintln!("(a) ok in {:?}", t0.elapsed());
}

/// (b) THE KEY TEST: the student path is exactly the forward on the fitted `Lq` (per sequence), and
/// `ht_next` is the teacher forward.
#[test]
fn b_student_path_equals_forward() {
    let t0 = Instant::now();
    let (cfg, rope, lt, ht, hs_in, ln_next) = fixture(11, SLAB);
    let mut hs = hs_in.clone();
    let res = quantise_layer(&cfg, &rope, &lt, &ln_next, &ht, &mut hs).expect("layer");

    let hid = HID;
    let seq_len = T * hid;
    for s in 0..N_SEQ {
        let row = s * seq_len;
        let want = parts(
            &cfg.layer,
            &rope,
            &res.lq.as_f64(),
            &hs_in[row..row + seq_len],
            T,
        )
        .expect("forward")
        .out;
        let got = &hs[row..row + seq_len];
        let e = rel_err(got, &want);
        assert!(e <= 1e-9, "student seq {s}: rel {e:e}");

        let want_t = parts(&cfg.layer, &rope, &lt.as_f64(), &ht[row..row + seq_len], T)
            .expect("teacher forward")
            .out;
        let got_t = &res.ht_next[row..row + seq_len];
        let et = rel_err(got_t, &want_t);
        assert!(et <= 1e-9, "teacher seq {s}: rel {et:e}");
    }

    // The layer's own output error against the teacher.
    eprintln!(
        "(b) ok in {:?}; ||hs - ht_next||/||ht_next|| = {:.3e}",
        t0.elapsed(),
        rel_err(&hs, &res.ht_next)
    );
}

/// (c) the bit budget is the sum of the fits, and the sign count is the hand count.
#[test]
fn c_bits_and_sign_count() {
    let t0 = Instant::now();
    let (cfg, rope, lt, ht, hs_in, ln_next) = fixture(13, SLAB);
    let mut hs = hs_in.clone();
    let res = quantise_layer(&cfg, &rope, &lt, &ln_next, &ht, &mut hs).expect("layer");

    let mut nsign = 0_u64;
    let mut nr = 0_u64;
    let mut nc = 0_u64;
    let mut nd = 0_u64;
    let mut ncorr = 0_u64;
    for f in &res.fits {
        nsign += f.bits.nsign;
        nr += f.bits.nr;
        nc += f.bits.nc;
        nd += f.bits.nd;
        ncorr += f.bits.ncorr;
    }
    assert_eq!(res.bits.nsign, nsign);
    assert_eq!(res.bits.nr, nr);
    assert_eq!(res.bits.nc, nc);
    assert_eq!(res.bits.nd, nd);
    assert_eq!(res.bits.ncorr, ncorr);

    let hid = HID as u64;
    let qo = (N_HEAD * HEAD_DIM) as u64;
    let kvo = (N_KV * HEAD_DIM) as u64;
    let inter = INTER as u64;
    let k_rows = QMLP as u64;
    let want = [
        qo * hid,
        2 * kvo * hid,
        2 * kvo * hid,
        2 * hid * qo,
        inter * hid + k_rows * hid,
        inter * hid + k_rows * hid,
        hid * inter,
    ];
    for (i, w) in want.iter().enumerate() {
        assert_eq!(res.fits[i].bits.nsign, *w, "nsign at {i}");
    }
    eprintln!(
        "(c) ok in {:?}; bits total {} (sign {} r {} c {} d {} corr {})",
        t0.elapsed(),
        res.bits.nsign + res.bits.nr + res.bits.nc + res.bits.nd + res.bits.ncorr,
        res.bits.nsign,
        res.bits.nr,
        res.bits.nc,
        res.bits.nd,
        res.bits.ncorr
    );
}

/// (d) the `g_row` helper: mean `1`, and the all-zero `1e-12` clamp.
#[test]
fn d_g_row_helper() {
    let v = [1.0_f64, 2.0, 3.0, 4.0];
    let g = g_row_from_vec(&v).expect("g_row");
    let mean = g.iter().sum::<f64>() / g.len() as f64;
    assert!((mean - 1.0).abs() <= 1e-12, "mean {mean}");
    for i in 0..v.len() {
        assert!((g[i] - v[i] * v[i] / 7.5).abs() <= 1e-12);
    }
    // all-zero input: the clamp gives an all-zero vector, never NaN.
    let z = g_row_from_vec(&[0.0, 0.0, 0.0, 0.0]).expect("g_row zero");
    assert!(
        z.iter().all(|x| *x == 0.0),
        "all-zero g_row must be all zeros"
    );

    // The matrix form: column sums of squares, normalised.
    let m = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gm = g_row_from_matrix(&m, 2, 3).expect("g_row matrix");
    let mean_m = gm.iter().sum::<f64>() / gm.len() as f64;
    assert!((mean_m - 1.0).abs() <= 1e-12);
    let col: Vec<f64> = (0..3).map(|c| m[c] * m[c] + m[3 + c] * m[3 + c]).collect();
    let denom = col.iter().sum::<f64>() / 3.0;
    for c in 0..3 {
        assert!((gm[c] - col[c] / denom).abs() <= 1e-12);
    }
    assert_eq!(
        g_row_from_matrix(&m, 2, 4),
        Err(DriverError::Shape),
        "a wrong matrix length must refuse"
    );
}

/// (e) the slab size matters: `slab = 1` and `slab = n_seq` differ once a sequence's norms trip the
/// sink gate.
#[test]
fn e_slab_size_matters() {
    let t0 = Instant::now();
    let (cfg1, _, _, _, _, _) = fixture(17, 1);
    let (cfg3, rope, lt, mut ht, hs_in, ln_next) = fixture(17, N_SEQ);
    // Scale one sequence's teacher and student input so its token norms are ~30x the others: the
    // per-slab median then keeps its weights at 1, while the whole-set median pulls them down.
    for k in (2 * T * HID)..(3 * T * HID) {
        ht[k] *= 30.0;
    }
    let hs_in = hs_in
        .iter()
        .enumerate()
        .map(|(k, v)| if k >= 2 * T * HID { v * 30.0 } else { *v })
        .collect::<Vec<f64>>();

    let mut hs1 = hs_in.clone();
    let r1 = quantise_layer(&cfg1, &rope, &lt, &ln_next, &ht, &mut hs1).expect("slab 1");
    let mut hs3 = hs_in.clone();
    let r3 = quantise_layer(&cfg3, &rope, &lt, &ln_next, &ht, &mut hs3).expect("slab n_seq");

    // The sink-weighted token count per slab, on the same fixture (teacher `mid`/`out` are all the
    // weights read): computed directly with the installed helper. The same taps are concatenated
    // for the whole-slab call below.
    let mut sink_counts = Vec::new();
    let mut mid_all = Vec::with_capacity(N_SEQ * T * HID);
    let mut out_all = Vec::with_capacity(N_SEQ * T * HID);
    for s0 in 0..N_SEQ {
        let row = s0 * T * HID;
        let hts = &ht[row..row + T * HID];
        let hss = &hs_in[row..row + T * HID];
        let p = parts(&cfg3.layer, &rope, &lt.as_f64(), hts, T).expect("teacher");
        let w = sink_token_weights(hts, &p.mid, &p.out, hss, None, 1, T, HID, cfg3.sink_thresh)
            .expect("weights");
        sink_counts.push(w.iter().filter(|x| **x < 1.0).count());
        mid_all.extend_from_slice(&p.mid);
        out_all.extend_from_slice(&p.out);
    }
    let total1: usize = sink_counts.iter().sum();

    // The same weights on the whole slab in one call (`seqs = N_SEQ`): the median is taken over
    // every sequence at once, so the scaled sequence's positions fall below the threshold.
    let w_all = sink_token_weights(
        &ht,
        &mid_all,
        &out_all,
        &hs_in,
        None,
        N_SEQ,
        T,
        HID,
        cfg3.sink_thresh,
    )
    .expect("weights");
    let total_all = w_all.iter().filter(|x| **x < 1.0).count();

    let diff = r1
        .fits
        .iter()
        .zip(&r3.fits)
        .map(|(a, b)| {
            a.wp.iter()
                .zip(&b.wp)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);
    eprintln!(
        "(e) ok in {:?}; sink-weighted tokens per-seq slab {sink_counts:?} (total {total1}); \
         whole-slab (seqs={N_SEQ}) {total_all}; max |wp(slab=1) - wp(slab=n_seq)| = {diff:.3e}",
        t0.elapsed()
    );
    assert!(
        total_all > total1,
        "the whole-slab median left no more sink-filtered tokens than per-sequence: \
         per-seq {total1}, whole-slab {total_all}; the slab size has no effect on this fixture"
    );
    assert!(
        diff > 1e-12,
        "slab=1 and slab=n_seq gave the same weights; the slab size does not matter on this fixture"
    );
}

/// (f) the two-layer model loop: the copy rule, the final norm fit and the `ln_next` source.
#[test]
fn f_model_loop_and_final_norm() {
    let t0 = Instant::now();
    let cfg = driver_cfg(SLAB);
    let rope = RopeTables::new(&cfg.layer, T).expect("rope");
    let mut rng = Lcg::new(23);
    let layers = vec![
        make_layer(&cfg.layer, &mut rng),
        make_layer(&cfg.layer, &mut rng),
    ];
    // A distinct final-norm gain pattern (alternating `0.5`/`2.0`), so its `g_row` differs from
    // layer 1's `ln1` (`~1`).
    let fnorm: Vec<f64> = (0..HID)
        .map(|i| if i % 2 == 0 { 0.5 } else { 2.0 })
        .collect();
    let h0 = rand_hidden(N_SEQ * T, HID, 1.0, &mut rng);
    let vocab = 7;
    let embed = rand_hidden(vocab, HID, 0.5, &mut rng);

    // n_layers = 1: layer 1 is copied unchanged, layer 0 uses layer 1's `ln1` for `down`.
    let m1 = quantise_model(&cfg, &rope, &layers, &fnorm, &embed, vocab, &h0, 1, 0.15)
        .expect("model n1");
    assert_eq!(m1.layers.len(), 2);
    assert_eq!(m1.layers[1], layers[1], "layer >= n_layers must be copied");
    assert!(m1.fits[1].is_empty());
    assert_eq!(m1.fits[0].len(), 7);
    assert_eq!(m1.fnorm.len(), HID);
    assert!(all_finite(&m1.fnorm), "the final norm is not finite");

    // The `ln_next` source is read: `g_row` of layer 1's `ln1` differs from `g_row` of `fnorm`.
    let g_l1 = g_row_from_vec(&layers[1].ln1).expect("g ln1");
    let g_fn = g_row_from_vec(&fnorm).expect("g fnorm");
    let gdiff = g_l1
        .iter()
        .zip(&g_fn)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(gdiff > 1e-6, "the two g_row sources are indistinguishable");

    // n_layers = 2: layer 1 now enters, and its `ln_next` is the final norm weight. Layer 0's
    // `ln_next` is layer 1's `ln1` in both runs, so its `down` fit is the same.
    let m2 = quantise_model(&cfg, &rope, &layers, &fnorm, &embed, vocab, &h0, 2, 0.15)
        .expect("model n2");
    assert_eq!(
        m1.layers[0].down, m2.layers[0].down,
        "layer 0 ln_next moved"
    );
    assert_eq!(m2.fits[1].len(), 7);

    // The source is read: a direct layer-0 call with `fnorm` gives a different `down` fit, so the
    // driver's choice of `layers[1].ln1` (matched above) is observable.
    let mut hs_w = h0.clone();
    let wrong =
        quantise_layer(&cfg, &rope, &layers[0], &fnorm, &h0, &mut hs_w).expect("direct fnorm");
    let ddiff = m1.layers[0]
        .down
        .iter()
        .zip(&wrong.lq.down)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        ddiff > 1e-12,
        "the down fit is unchanged by the ln_next source; the fixture cannot see it"
    );
    assert_eq!(m2.fnorm.len(), HID);
    assert!(all_finite(&m2.fnorm));
    eprintln!(
        "(f) ok in {:?}; fnorm[0..3] = {:?}",
        t0.elapsed(),
        &m2.fnorm[0..3]
    );
}

/// (g) the refusals.
#[test]
fn g_refusals() {
    let (cfg, rope, lt, ht, hs_in, ln_next) = fixture(29, SLAB);

    // slab 0.
    let mut bad = cfg;
    bad.slab = 0;
    let mut hs = hs_in.clone();
    assert_eq!(
        unwrap_err(quantise_layer(&bad, &rope, &lt, &ln_next, &ht, &mut hs)),
        DriverError::Geometry
    );

    // t < 2.
    let mut bad = cfg;
    bad.t = 1;
    assert_eq!(
        unwrap_err(quantise_layer(&bad, &rope, &lt, &ln_next, &ht, &mut hs)),
        DriverError::TinySeq
    );

    // wrong `hs` length.
    let mut short = vec![0.0_f64; hs_in.len() - 1];
    assert_eq!(
        unwrap_err(quantise_layer(&cfg, &rope, &lt, &ln_next, &ht, &mut short)),
        DriverError::Shape
    );

    // non-finite weight.
    let mut lt_nan = lt.clone();
    lt_nan.q[3] = f64::NAN;
    let mut hs = hs_in.clone();
    assert_eq!(
        unwrap_err(quantise_layer(&cfg, &rope, &lt_nan, &ln_next, &ht, &mut hs)),
        DriverError::NonFinite
    );

    // `n_seq * t` mismatch: the hidden state is one row short of the geometry.
    let mut cfg_bad = cfg;
    cfg_bad.n_seq = N_SEQ + 1;
    let mut hs = hs_in.clone();
    assert_eq!(
        unwrap_err(quantise_layer(&cfg_bad, &rope, &lt, &ln_next, &ht, &mut hs)),
        DriverError::Shape
    );

    // a stage width that is not a multiple of 128.
    let mut narrow = driver_cfg(SLAB);
    narrow.layer.hidden = 64;
    narrow.layer.n_head = 1;
    narrow.layer.head_dim = 64;
    let rope64 = RopeTables::new(&narrow.layer, T).expect("rope");
    let mut hs = vec![0.0_f64; N_SEQ * T * 64];
    assert_eq!(
        unwrap_err(quantise_layer(
            &narrow, &rope64, &lt, &ln_next, &ht, &mut hs
        )),
        DriverError::Shape
    );
}
