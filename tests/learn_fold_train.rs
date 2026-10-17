//! `tests/learn_fold_train.rs` — the finishing P-step: the KL falls on a fixed batch over 20 Adam
//! steps, one plane latent's and one norm latent's gradient match a central difference of the
//! unquantized-forward loss, the frozen codes are bit-identical before and after training, the
//! step-counted learning-rate schedule matches its hand-computed factors, the geometry refusals
//! fire, and the tiny-configuration step cost and flop count are reported. Grade `tested`; nothing
//! here is `proved`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps its bits to f64 and to codebook indices; the short indices \
              are the reference arithmetic's vocabulary; the exact comparisons are the port's \
              bit-for-bit contract (the frozen codes) and the hand-computed schedule literals; the \
              refusal test enumerates the geometry arms in one body."
)]

use apeiron::learn::fold::finish::plane::CodedPlane;
use apeiron::learn::fold::finish::train::{
    assemble, assemble_unquantized, kl_mean, Adam, Finisher, Frozen, LayerTunables, Linear, Planes,
    Schedule, TrainError, Tunables, N_KINDS,
};
use apeiron::learn::fold::fit::icm::{codebook, BLOCK};
use apeiron::learn::fold::fit::quant::{quant_c_u8, quant_r_hier};
use apeiron::learn::fold::fit::steps::GROUP;
use apeiron::learn::grad::model::{forward, Config, Weights};
use apeiron::learn::solve::q32::from_grid;

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
/// The central-difference step of the latent gradient check.
const H: f64 = 1e-3;
/// `H` on the Q32.32 grid, the step a latent word is nudged by.
const H_Q32: i64 = 4_294_967;
/// The nonzero norm latent the chain-rule check runs at (`exp(0) = 1` hides the chain factor).
const D_NORM: f64 = 0.2;

/// `round(v·2^32)` for a test fixture value.
fn q32(v: f64) -> i64 {
    (v * 4_294_967_296.0).round() as i64
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

/// The exact code field from a list of codebook indices, `o × n` row-major.
fn code_field(idx: &[u8], o: usize, n: usize) -> Vec<f64> {
    let cb = codebook();
    let nb = n / BLOCK;
    let mut s = vec![0.0_f64; o * n];
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

/// One coded plane of shape `o × n` whose `r` grid, `c` and `d` are their quantizers' fixed points,
/// so at zero latent offsets `weight() == weight_unquantized()`.
fn make_plane(rng: &mut Lcg, o: usize, n: usize) -> CodedPlane {
    let idx: Vec<u8> = (0..o * (n / BLOCK))
        .map(|_| (rng.next_u64() % 256) as u8)
        .collect();
    let s = code_field(&idx, o, n);
    let g = n / GROUP;
    let r_raw: Vec<f64> = (0..o * g).map(|_| 0.5 + 0.5 * rng.unit()).collect();
    let rh = quant_r_hier(&r_raw, o, g).unwrap();
    let c_raw: Vec<f64> = (0..n).map(|_| 0.1 + 0.1 * rng.unit()).collect();
    let c0 = quant_c_u8(&c_raw, 6).unwrap().0;
    let d_raw: Vec<f64> = (0..n).map(|_| 0.5 + 0.5 * rng.unit()).collect();
    let d0 = quant_c_u8(&d_raw, 5).unwrap().0;
    let plane =
        CodedPlane::from_pack(&s, &rh.r_lo, &rh.r_sp, &rh.rq, &c0, &d0, None, o, n).unwrap();
    assert!(plane.s_fix.is_none(), "an exact field keeps idx");
    plane
}

/// A deterministic nonzero nudge of every projection, embedding and head element; the teacher's
/// weights. `amp` is in Q32.32 units.
fn perturb(w: &mut Weights, amp: i64) {
    let mut s = 0x9E37_79B9_7F4A_7C15_u64;
    let span = u64::try_from(2 * amp).unwrap();
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

/// The fixture: a two-layer coded-plane student, its zero-bit tunables, the frozen embedding and
/// head, a fixed 16-token batch and the fixed teacher logits of a nudged copy of the student.
struct Fixture {
    fin: Finisher,
    tokens: Vec<u32>,
    teacher: Vec<i64>,
}

fn fixture(seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let mut layers: Vec<Vec<Linear>> = Vec::with_capacity(L);
    for _ in 0..L {
        let mut row: Vec<Linear> = Vec::with_capacity(N_KINDS);
        for k in 0..N_KINDS {
            let (o, n) = dims(k);
            let mut lin = Linear::new(o, n);
            lin.planes.push(make_plane(&mut rng, o, n));
            if k == 0 || k == 6 {
                // The accepted dense corrector term the Python carries on `q` and `down`.
                lin.extra = Some((0..o * n).map(|_| 0.02 * rng.sym()).collect());
            }
            row.push(lin);
        }
        layers.push(row);
    }
    let mut tun_layers: Vec<LayerTunables> = Vec::with_capacity(L);
    for _ in 0..L {
        let ln1: Vec<f64> = (0..E).map(|_| 0.8 + 0.2 * rng.unit()).collect();
        let ln2: Vec<f64> = (0..E).map(|_| 0.8 + 0.2 * rng.unit()).collect();
        let qb: Vec<f64> = (0..NH * HD).map(|_| 0.05 * rng.sym()).collect();
        let kb: Vec<f64> = (0..NKV * HD).map(|_| 0.05 * rng.sym()).collect();
        let vb: Vec<f64> = (0..NKV * HD).map(|_| 0.05 * rng.sym()).collect();
        let mut lt = LayerTunables::new(&ln1, &ln2, &qb, &kb, &vb).unwrap();
        // The chain factor `exp(d)` is `1` at the zero initialisation and would hide a missing
        // factor, so the fixture starts the norm latents nonzero; the production init is zero.
        for j in 0..E {
            lt.ln1_d[j] = q32(D_NORM);
            lt.ln2_d[j] = -q32(D_NORM);
        }
        tun_layers.push(lt);
    }
    let fn_base: Vec<f64> = (0..E).map(|_| 0.8 + 0.2 * rng.unit()).collect();
    let mut tun = Tunables::new(tun_layers, &fn_base).unwrap();
    for j in 0..E {
        tun.fn_d[j] = q32(0.3);
    }
    let token_embd: Vec<f64> = (0..V * E).map(|_| 0.1 * rng.sym()).collect();
    let lm_head: Vec<f64> = (0..V * E).map(|_| 0.1 * rng.sym()).collect();
    let frozen = Frozen::new(&token_embd, &lm_head).unwrap();
    let tokens: Vec<u32> = (0..T).map(|_| (rng.next_u64() % V as u64) as u32).collect();

    let cfg = cfg();
    let planes = Planes::new(layers);
    let w = assemble(&cfg, &planes, &tun, &frozen).unwrap();
    let mut tw = w.clone();
    perturb(&mut tw, 1 << 26);
    let (teacher, _) = forward(&tw, &cfg, &tokens).unwrap();

    let fin = Finisher::new(cfg, planes, tun, frozen, Schedule::new(10, 100)).unwrap();
    Fixture {
        fin,
        tokens,
        teacher,
    }
}

/// The batch mean KL over the differentiable (unquantized-weight) forward: the f64 loss the latent
/// gradient is the exact gradient of.
fn loss_unquantized(fin: &Finisher, tokens: &[u32], teacher: &[i64]) -> f64 {
    let w = assemble_unquantized(&fin.cfg, &fin.planes, &fin.tun, &fin.frozen).unwrap();
    let (logits, _) = forward(&w, &fin.cfg, tokens).unwrap();
    kl_mean(&fin.cfg, &logits, teacher).unwrap().0
}

/// Every plane's code index, in `[layer][kind][plane]` order.
fn codes(fin: &Finisher) -> Vec<Vec<Vec<u8>>> {
    fin.planes
        .layers
        .iter()
        .map(|row| row.iter().map(|lin| lin.planes[0].idx.clone()).collect())
        .collect()
}

/// (a) The KL on the fixed batch falls over 20 steps: the assembled latents and tunables are on
/// the path the whole-model forward reads, and the mapped gradient has the descent sign.
#[test]
fn training_lowers_the_kl_on_a_fixed_batch() {
    let mut f = fixture(1);
    let kl0 = f.fin.loss(&f.tokens, &f.teacher).unwrap();
    let mut last = kl0;
    for _ in 0..20 {
        last = f.fin.step(&f.tokens, &f.teacher).unwrap();
    }
    let kl20 = f.fin.loss(&f.tokens, &f.teacher).unwrap();
    eprintln!("kl0 {kl0:.6} kl20 {kl20:.6} last-step {last:.6}");
    assert!(
        kl20 < kl0,
        "the KL did not fall: step 0 {kl0:.6}, step 20 {kl20:.6}"
    );
}

/// (b) Central differences of the unquantized-forward loss against the analytic gradient of one
/// plane latent (at the quantizers' fixed point, so the straight-through Jacobian is exact) and of
/// one norm latent (at `exp(d) != 1`, so the chain factor is exercised).
#[test]
fn latent_gradient_matches_finite_difference() {
    let f = fixture(2);
    let g = f.fin.gradients(&f.tokens, &f.teacher).unwrap();

    // The plane latent: `pa[0]` of `attn_q` in layer 0.
    let an_pa = from_grid(g.planes[0][0][0].0[0]);
    let mut plus = f.fin.clone();
    plus.planes.layers[0][0].planes[0].pa[0] += H_Q32;
    let lp = loss_unquantized(&plus, &f.tokens, &f.teacher);
    let mut minus = f.fin.clone();
    minus.planes.layers[0][0].planes[0].pa[0] -= H_Q32;
    let lm = loss_unquantized(&minus, &f.tokens, &f.teacher);
    let fd_pa = (lp - lm) / (2.0 * H);
    eprintln!("pa: analytic {an_pa:e} fd {fd_pa:e}");
    assert!(
        (fd_pa - an_pa).abs() <= 1e-8 + 2e-3 * an_pa.abs(),
        "pa[0]: analytic {an_pa:e} vs fd {fd_pa:e}"
    );

    // The norm latent: `ln1_d[0]` of layer 0, at `exp(D_NORM) != 1`.
    let an_ln1 = from_grid(g.norms[0].ln1[0]);
    let mut plus = f.fin.clone();
    plus.tun.layers[0].ln1_d[0] += H_Q32;
    let lp = loss_unquantized(&plus, &f.tokens, &f.teacher);
    let mut minus = f.fin.clone();
    minus.tun.layers[0].ln1_d[0] -= H_Q32;
    let lm = loss_unquantized(&minus, &f.tokens, &f.teacher);
    let fd_ln1 = (lp - lm) / (2.0 * H);
    eprintln!("ln1_d: analytic {an_ln1:e} fd {fd_ln1:e}");
    assert!(
        (fd_ln1 - an_ln1).abs() <= 1e-8 + 2e-3 * an_ln1.abs(),
        "ln1_d[0]: analytic {an_ln1:e} vs fd {fd_ln1:e}"
    );
    // The check is not vacuous: `exp(D_NORM) = 1` would make the two agree for the wrong reason.
    assert!(
        (D_NORM.exp() - 1.0).abs() > 0.2,
        "the chain factor is trivial"
    );
}

/// (c) The codes are bit-identical before and after 20 steps, and the latents actually moved (so
/// the identity is not the identity of a step that did nothing).
#[test]
fn codes_are_unchanged_by_training() {
    let mut f = fixture(3);
    let before = codes(&f.fin);
    let ln1_before = f.fin.tun.layers[0].ln1_d[0];
    for _ in 0..20 {
        f.fin.step(&f.tokens, &f.teacher).unwrap();
    }
    let after = codes(&f.fin);
    assert_eq!(before, after, "the frozen codes changed under training");
    assert_ne!(
        ln1_before, f.fin.tun.layers[0].ln1_d[0],
        "the norm latent did not move, so the code identity is vacuous"
    );
}

/// (d) The step-counted schedule against the hand-computed factors at steps `0, 5, 10, 50, 99` of
/// `steps = 100`, `warmup = 10`:
/// `f(s) = min(1, s/10) · (0.1 + 0.45·(1 + cos(π s/100)))`.
#[test]
fn schedule_matches_hand_computed_factors() {
    let s = Schedule::new(10, 100);
    let want = [
        (0_u64, 0.0_f64),
        (5, 0.497_229_876_634),
        (10, 0.977_975_432_333),
        (50, 0.55),
        (99, 0.100_222_047_835),
    ];
    // Derived bound: the Q32.32 grid quantum `2^-32` times the seven operations of
    // `wu·(0.1 + 0.45·(1 + cos))`, with the trig kernel's `2^-20` relative error folded in.
    let bound = 1e-9_f64;
    for (step, w) in want {
        let got = s.factor(step);
        assert!(
            (got - w).abs() < bound,
            "step {step}: factor {got:.12} vs hand-computed {w:.12}"
        );
    }
    // Past the warm-up the cosine alone drives the factor down and up to `steps`.
    assert!(s.factor(11) > s.factor(50));
    assert!(s.factor(50) > s.factor(99));
}

/// The pre-migration `f64` Adam step, the oracle the Q32.32 [`Adam::apply`] is asserted against.
fn ref_adam(lr: f64, p: &mut [f64], g: &[f64], m: &mut [f64], v: &mut [f64], t: u64) {
    let b1 = 0.9_f64;
    let b2 = 0.999_f64;
    let bc1 = 1.0 - b1.powi(i32::try_from(t).unwrap_or(i32::MAX));
    let bc2 = 1.0 - b2.powi(i32::try_from(t).unwrap_or(i32::MAX));
    for i in 0..p.len() {
        let gi = g[i];
        m[i] = b1 * m[i] + (1.0 - b1) * gi;
        v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
        p[i] -= lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + 1e-8);
    }
}

/// The migrated Adam agrees with the pre-migration `f64` body over ten steps. Bound: five Q32.32
/// roundings per element per step (`≤ 5·2^-33` of the update) plus the `1.3e-2` relative
/// representation error of `eps`, whose share of a step is `eps/(sqrt(v)+eps)`; ten steps put the
/// parameter bound at `1e-8`, and the fixture's gradients and learning rate keep the parameters near
/// `1e-3`.
#[test]
fn adam_agrees_with_the_f64_oracle() {
    let mut rng = Lcg::new(53);
    let n = 64_usize;
    let mut p_q = vec![0_i64; n];
    let mut p_f = vec![0_f64; n];
    let mut adam = Adam::zeros(n);
    let mut m_f = vec![0_f64; n];
    let mut v_f = vec![0_f64; n];
    let lr = 1e-3_f64;
    let mut worst = 0.0_f64;
    for step in 0..10 {
        let g_f: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let g_q: Vec<i64> = g_f.iter().map(|&x| q32(x)).collect();
        adam.apply(lr, &mut p_q, &g_q).unwrap();
        ref_adam(lr, &mut p_f, &g_f, &mut m_f, &mut v_f, step + 1);
    }
    for i in 0..n {
        let d = (from_grid(p_q[i]) - p_f[i]).abs();
        worst = worst.max(d);
        assert!(
            d <= 1e-8,
            "parameter {i}: {} vs {} (|d| {d:e})",
            from_grid(p_q[i]),
            p_f[i]
        );
    }
    eprintln!("adam: max |d| after 10 steps {worst:e} (bound 1e-8)");
    assert!(worst > 0.0, "the check is vacuous if nothing moved");
}

/// The geometry refusals: a plane whose shape disagrees with the model, a logits/teacher mismatch,
/// and an Adam width mismatch.
#[test]
fn refuses_mismatched_shapes() {
    let f = fixture(4);
    let mut bad = f.fin.planes.clone();
    bad.layers[0][0].n = 2 * E;
    let err = Finisher::new(
        f.fin.cfg.clone(),
        bad,
        f.fin.tun.clone(),
        f.fin.frozen.clone(),
        Schedule::new(10, 100),
    );
    assert!(matches!(err, Err(TrainError::Shape(_))), "got {err:?}");

    let short = f.teacher[..V].to_vec();
    let err = f.fin.gradients(&f.tokens, &short);
    assert!(matches!(err, Err(TrainError::Shape(_))), "got {err:?}");

    let mut adam = Adam::zeros(3);
    let err = adam.apply(1e-3, &mut [0_i64; 3], &[0_i64; 2]);
    assert!(matches!(err, Err(TrainError::Shape(_))), "got {err:?}");

    // A loader value outside the Q32.32 grid is refused, not silently rounded.
    let err = Frozen::new(&[3.0e9_f64; 4], &[0.0_f64; 4]);
    assert!(matches!(err, Err(TrainError::Range(_))), "got {err:?}");
    let err = LayerTunables::new(&[f64::NAN], &[1.0], &[1.0], &[1.0], &[1.0]);
    assert!(matches!(err, Err(TrainError::NotFinite(_))), "got {err:?}");
}

/// The tiny-configuration cost: the measured wall seconds per step and the analytic flop count, for
/// the lead's extrapolation to the production configuration (`CHUNK` 512 × batch 8).
#[test]
fn step_cost_and_flop_count() {
    let mut f = fixture(5);
    // Warm the allocator/caches with five steps, then time twenty.
    for _ in 0..5 {
        f.fin.step(&f.tokens, &f.teacher).unwrap();
    }
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        f.fin.step(&f.tokens, &f.teacher).unwrap();
    }
    let wall = t0.elapsed().as_secs_f64() / 20.0;

    // The trainable latents: per plane `o·n/128 + 2n`, per layer `2·n_embd + q_out + 2·kv_out`,
    // plus the final-norm latent.
    let mut n_train = 0_usize;
    for l in 0..L {
        for k in 0..N_KINDS {
            let (o, n) = dims(k);
            n_train += o * (n / GROUP) + 2 * n;
            let _ = l;
        }
    }
    let q_out = NH * HD;
    let kv_out = NKV * HD;
    n_train += L * (2 * E + q_out + 2 * kv_out) + E;
    let mut n_latents = 0_usize;
    for row in &f.fin.planes.layers {
        for lin in row {
            for p in &lin.planes {
                n_latents += p.pa.len() + p.pb.len() + p.pd.len();
            }
        }
    }
    for lt in &f.fin.tun.layers {
        n_latents +=
            lt.ln1_d.len() + lt.ln2_d.len() + lt.qb_d.len() + lt.kb_d.len() + lt.vb_d.len();
    }
    n_latents += f.fin.tun.fn_d.len();
    assert_eq!(
        n_train, n_latents,
        "the trainable latent count disagrees with the geometry"
    );

    // Per step: the student forward (2 flops per matmul weight, the attention and the head), the
    // backward (twice the forward's matmul flops), and the Adam update (~10 ops per latent).
    let mut n_w = 0_usize;
    for l in 0..L {
        for k in 0..N_KINDS {
            let (o, n) = dims(k);
            n_w += o * n;
            let _ = l;
        }
    }
    let matmul = 2 * n_w * T + 2 * V * E * T;
    let attn = 2 * NH * T * T * 2 * HD;
    let flops = 3 * (matmul + attn) + 10 * n_latents;
    eprintln!(
        "tiny config: {wall:.6} s/step (debug build), {flops} flops/step, {n_train} trainable latents"
    );
    assert!(wall > 0.0, "the wall clock did not advance");
}
