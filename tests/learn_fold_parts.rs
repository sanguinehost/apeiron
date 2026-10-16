//! Integration tests for the `f64` CPU forward of one `Qwen2` decoder layer
//! (`src/learn/fold/stats/parts.rs`).
//!
//! Grade: `tested`. The positive assertions cover the causality invariant (a later token never
//! changes an earlier output row, bit-for-bit), a hand-computed `RMSNorm` row, the `RoPE`
//! position-0 identity, a full-head attention compared against a naive triple-loop reference,
//! grouped-query attention against the duplicated-head expansion, and the `SwiGLU` residual
//! `out = mid + m·downᵀ` recomputed in the test. The refusal test reaches every [`PartsError`]
//! variant.
//!
//! The fixture is a tiny layer (`n_head 4`, `n_kv_head 2`, `head_dim 8`, `hidden 32`, `inter 48`,
//! `t 6`) with deterministic LCG weights. `n_head == 4` is the smallest even split, so the
//! duplication test in `gqa_equals_duplicated_kv_heads` has `rep = 2`.

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "the test fixture names the head geometry `nh`/`nkv`/`hd` and the widths `hid`/`inter` \
              as the module does, constructs f64 weights from a u64 LCG state (a deliberate lossy \
              cast), and asserts the bitwise-exact equalities the causality and GQA invariants are \
              stated as"
)]

use apeiron::learn::fold::stats::parts::{
    attn_a, parts, rms, LayerCfg, LayerF64, PartsError, RopeTables,
};

/// Query head count.
const NH: usize = 4;
/// Key/value head count.
const NKV: usize = 2;
/// Per-head width.
const HD: usize = 8;
/// Residual width.
const HID: usize = NH * HD;
/// Feed-forward width.
const INTER: usize = 48;
/// Sequence length.
const T: usize = 6;
/// Rope position count.
const TMAX: usize = 8;
/// `RMSNorm` epsilon.
const EPS: f64 = 1.0e-6;
/// `RoPE` base.
const THETA: f64 = 10_000.0;

/// The tiny geometry every test uses.
fn cfg() -> LayerCfg {
    LayerCfg {
        n_head: NH,
        n_kv_head: NKV,
        head_dim: HD,
        hidden: HID,
        inter: INTER,
        eps: EPS,
        theta: THETA,
    }
}

/// Deterministic LCG producing `f64` values in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    /// A generator at `seed`.
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next value in `[-1, 1)`: the top 53 bits of the state scaled by `2^-53`.
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = self.0 >> 11;
        (bits as f64) * (1.0 / 9_007_199_254_740_992.0) * 2.0 - 1.0
    }

    /// `n` consecutive values.
    fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_f64()).collect()
    }
}

/// The twelve weight tensors of one layer, owned so [`Weights::layer`] can borrow them.
struct Weights {
    ln1: Vec<f64>,
    ln2: Vec<f64>,
    q: Vec<f64>,
    qb: Vec<f64>,
    k: Vec<f64>,
    kb: Vec<f64>,
    v: Vec<f64>,
    vb: Vec<f64>,
    o: Vec<f64>,
    gate: Vec<f64>,
    up: Vec<f64>,
    down: Vec<f64>,
}

impl Weights {
    /// Deterministic weights for the geometry `cfg`.
    fn new(rng: &mut Lcg, cfg: &LayerCfg) -> Self {
        let hid = cfg.hidden;
        let qw = cfg.n_head * cfg.head_dim;
        let kw = cfg.n_kv_head * cfg.head_dim;
        let inter = cfg.inter;
        Self {
            ln1: rng.vec(hid),
            ln2: rng.vec(hid),
            q: rng.vec(qw * hid),
            qb: rng.vec(qw),
            k: rng.vec(kw * hid),
            kb: rng.vec(kw),
            v: rng.vec(kw * hid),
            vb: rng.vec(kw),
            o: rng.vec(hid * qw),
            gate: rng.vec(inter * hid),
            up: rng.vec(inter * hid),
            down: rng.vec(hid * inter),
        }
    }

    /// A borrowed [`LayerF64`] view of these weights.
    fn layer(&self) -> LayerF64<'_> {
        LayerF64 {
            ln1: &self.ln1,
            ln2: &self.ln2,
            q: &self.q,
            qb: &self.qb,
            k: &self.k,
            kb: &self.kb,
            v: &self.v,
            vb: &self.vb,
            o: &self.o,
            gate: &self.gate,
            up: &self.up,
            down: &self.down,
        }
    }
}

/// Element-wise closeness of two equal-length `f64` buffers.
fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length {} != {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} != {y} (tol {tol})");
    }
}

/// `true` iff the first `rows` rows (width `width`) of `a` and `b` are bitwise equal.
fn prefix_eq(a: &[f64], b: &[f64], rows: usize, width: usize) -> bool {
    a[..rows * width] == b[..rows * width]
}

/// A naive triple-loop reference for [`attn_a`]: rotate `q`/`k`, expand the kv heads, then score,
/// mask and normalise one `(head, query)` row at a time.
fn naive_attention(
    cfg: &LayerCfg,
    rope: &RopeTables,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    t: usize,
) -> Vec<f64> {
    let nh = cfg.n_head;
    let nkv = cfg.n_kv_head;
    let hd = cfg.head_dim;
    let half = hd / 2;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut qr = vec![0.0f64; q.len()];
    let mut kr = vec![0.0f64; k.len()];
    for p in 0..t {
        for h in 0..nh {
            let base = p * nh * hd + h * hd;
            for d in 0..hd {
                let r = if d < half {
                    -q[base + d + half]
                } else {
                    q[base + d - half]
                };
                qr[base + d] = q[base + d] * rope.cos[p * hd + d] + r * rope.sin[p * hd + d];
            }
        }
        for h in 0..nkv {
            let base = p * nkv * hd + h * hd;
            for d in 0..hd {
                let r = if d < half {
                    -k[base + d + half]
                } else {
                    k[base + d - half]
                };
                kr[base + d] = k[base + d] * rope.cos[p * hd + d] + r * rope.sin[p * hd + d];
            }
        }
    }
    let rep = nh / nkv;
    let mut out = vec![0.0f64; q.len()];
    for h in 0..nh {
        let g = h / rep;
        for i in 0..t {
            let mut scores = vec![f64::NEG_INFINITY; t];
            for j in 0..=i {
                let mut dot = 0.0f64;
                for d in 0..hd {
                    dot += qr[i * nh * hd + h * hd + d] * kr[j * nkv * hd + g * hd + d];
                }
                scores[j] = scale * dot;
            }
            let mut row_max = f64::NEG_INFINITY;
            for j in 0..=i {
                if scores[j] > row_max {
                    row_max = scores[j];
                }
            }
            let mut denom = 0.0f64;
            let mut acc = vec![0.0f64; hd];
            for j in 0..=i {
                let e = (scores[j] - row_max).exp();
                denom += e;
                for d in 0..hd {
                    acc[d] += e * v[j * nkv * hd + g * hd + d];
                }
            }
            for d in 0..hd {
                out[i * nh * hd + h * hd + d] = acc[d] / denom;
            }
        }
    }
    out
}

#[test]
fn parts_shapes_are_stated() {
    let cfg = cfg();
    let rope = RopeTables::new(&cfg, TMAX).unwrap();
    let mut rng = Lcg::new(1);
    let w = Weights::new(&mut rng, &cfg);
    let h = rng.vec(T * HID);
    let p = parts(&cfg, &rope, &w.layer(), &h, T).unwrap();
    assert_eq!(
        (p.xa.len(), p.q.len(), p.k.len(), p.v.len()),
        (T * HID, T * HID, T * 16, T * 16)
    );
    assert_eq!(
        (p.a.len(), p.mid.len(), p.xf.len()),
        (T * HID, T * HID, T * HID)
    );
    assert_eq!(
        (p.g.len(), p.u.len(), p.m.len()),
        (T * INTER, T * INTER, T * INTER)
    );
    assert_eq!(p.out.len(), T * HID);
}

#[test]
fn changing_a_later_token_leaves_earlier_rows_bitwise_unchanged() {
    let cfg = cfg();
    let rope = RopeTables::new(&cfg, TMAX).unwrap();
    let mut rng = Lcg::new(2);
    let w = Weights::new(&mut rng, &cfg);
    let h = rng.vec(T * HID);
    let p0 = parts(&cfg, &rope, &w.layer(), &h, T).unwrap();

    // Perturb only the last token.
    let mut h2 = h.clone();
    for c in 0..HID {
        h2[(T - 1) * HID + c] += 1.0;
    }
    let p1 = parts(&cfg, &rope, &w.layer(), &h2, T).unwrap();

    // Every part's rows `0..T-1` are unchanged. Rows <= i < T-1 depend only on tokens <= i.
    for i in 0..T - 1 {
        assert!(p0.xa[i * HID..(i + 1) * HID] == p1.xa[i * HID..(i + 1) * HID]);
        assert!(p0.q[i * HID..(i + 1) * HID] == p1.q[i * HID..(i + 1) * HID]);
        assert!(p0.k[i * 16..(i + 1) * 16] == p1.k[i * 16..(i + 1) * 16]);
        assert!(p0.v[i * 16..(i + 1) * 16] == p1.v[i * 16..(i + 1) * 16]);
        assert!(p0.a[i * HID..(i + 1) * HID] == p1.a[i * HID..(i + 1) * HID]);
        assert!(p0.mid[i * HID..(i + 1) * HID] == p1.mid[i * HID..(i + 1) * HID]);
        assert!(p0.xf[i * HID..(i + 1) * HID] == p1.xf[i * HID..(i + 1) * HID]);
        assert!(p0.g[i * INTER..(i + 1) * INTER] == p1.g[i * INTER..(i + 1) * INTER]);
        assert!(p0.u[i * INTER..(i + 1) * INTER] == p1.u[i * INTER..(i + 1) * INTER]);
        assert!(p0.m[i * INTER..(i + 1) * INTER] == p1.m[i * INTER..(i + 1) * INTER]);
        assert!(p0.out[i * HID..(i + 1) * HID] == p1.out[i * HID..(i + 1) * HID]);
    }
    // Non-vacuous: the last output row did change.
    assert!(p0.out[(T - 1) * HID..] != p1.out[(T - 1) * HID..]);
    // The multi-token prefix helper agrees with the manual loop above.
    assert!(prefix_eq(&p0.out, &p1.out, T - 1, HID));
}

#[test]
fn rms_matches_a_hand_computed_row() {
    // x = [1, -1, 1, -1]: mean square 1, so rsqrt(mean + 0) = 1 and out = w∘x exactly.
    let x = [1.0, -1.0, 1.0, -1.0];
    let w = [1.0, 2.0, 3.0, 4.0];
    let out = rms(&x, &w, 0.0, 1, 4).unwrap();
    assert_eq!(out, vec![1.0, -2.0, 3.0, -4.0]);
    // A constant row of ±2 over a width-4 vector: mean square 4, rsqrt = 0.5, out = w∘x/2.
    let x2 = [2.0, -2.0, 2.0, -2.0];
    let out2 = rms(&x2, &w, 0.0, 1, 4).unwrap();
    assert_eq!(out2, vec![1.0, -2.0, 3.0, -4.0]);
}

#[test]
fn rope_at_position_zero_is_the_identity() {
    let cfg = cfg();
    let rope = RopeTables::new(&cfg, TMAX).unwrap();
    // emb[0, :] = 0, so cos = 1 and sin = 0: x·1 + rot(x)·0 = x.
    for j in 0..HD {
        assert_eq!(rope.cos[j], 1.0);
        assert_eq!(rope.sin[j], 0.0);
    }
    // Position 1, column 0: inv[0] = theta^0 = 1, emb = 1.
    assert_eq!(rope.cos[HD], 1.0f64.cos());
    assert_eq!(rope.sin[HD], 1.0f64.sin());
    // Position 1, column 1: inv[1] = theta^(-2/8) = theta^(-1/4).
    let emb = 1.0 / THETA.powf(0.25);
    assert_eq!(rope.cos[HD + 1], emb.cos());
    assert_eq!(rope.sin[HD + 1], emb.sin());
    // The concatenated half repeats the frequency: column head_dim/2 matches column 0.
    assert_eq!(rope.cos[HD + HD / 2], 1.0f64.cos());
}

#[test]
fn full_head_attention_equals_naive_triple_loop() {
    let cfg = LayerCfg {
        n_kv_head: NH,
        ..cfg()
    };
    let rope = RopeTables::new(&cfg, TMAX).unwrap();
    let mut rng = Lcg::new(5);
    let w = Weights::new(&mut rng, &cfg);
    let h = rng.vec(T * HID);
    let p = parts(&cfg, &rope, &w.layer(), &h, T).unwrap();

    // `q`, `k`, `v` are the pre-rope projections; the reference rotates and attends independently.
    let reference = naive_attention(&cfg, &rope, &p.q, &p.k, &p.v, T);
    close(&p.a, &reference, 1e-12);
    // And the standalone entry point agrees with the one embedded in `parts`.
    let direct = attn_a(&cfg, &rope, &p.q, &p.k, &p.v, T).unwrap();
    assert_eq!(p.a, direct);
}

#[test]
fn gqa_equals_duplicated_kv_heads() {
    let cfg = cfg();
    let rope = RopeTables::new(&cfg, TMAX).unwrap();
    let mut rng = Lcg::new(9);
    let w = Weights::new(&mut rng, &cfg);
    let h = rng.vec(T * HID);
    let p = parts(&cfg, &rope, &w.layer(), &h, T).unwrap();

    // Expand the two kv heads to four, duplicating head `hh` from kv head `hh / rep`.
    let mut k4 = vec![0.0f64; T * HID];
    let mut v4 = vec![0.0f64; T * HID];
    for row in 0..T {
        for hh in 0..NH {
            let src = row * (NKV * HD) + (hh / (NH / NKV)) * HD;
            let dst = row * HID + hh * HD;
            k4[dst..dst + HD].copy_from_slice(&p.k[src..src + HD]);
            v4[dst..dst + HD].copy_from_slice(&p.v[src..src + HD]);
        }
    }
    let cfg_full = LayerCfg {
        n_kv_head: NH,
        ..cfg
    };
    let a_gqa = attn_a(&cfg, &rope, &p.q, &p.k, &p.v, T).unwrap();
    let a_dup = attn_a(&cfg_full, &rope, &p.q, &k4, &v4, T).unwrap();
    assert_eq!(a_gqa, a_dup);
    // `parts` used exactly this GQA path.
    assert_eq!(p.a, a_gqa);
}

#[test]
fn out_equals_mid_plus_m_down() {
    let cfg = cfg();
    let rope = RopeTables::new(&cfg, TMAX).unwrap();
    let mut rng = Lcg::new(13);
    let w = Weights::new(&mut rng, &cfg);
    let h = rng.vec(T * HID);
    let p = parts(&cfg, &rope, &w.layer(), &h, T).unwrap();

    let mut expected = p.mid.clone();
    for r in 0..T {
        for c in 0..HID {
            let mut acc = 0.0f64;
            for k in 0..INTER {
                acc += w.down[c * INTER + k] * p.m[r * INTER + k];
            }
            expected[r * HID + c] += acc;
        }
    }
    close(&expected, &p.out, 1e-12);
}

#[test]
fn refusals_are_typed() {
    let cfg = cfg();
    // hidden != n_head * head_dim.
    let bad_hidden = LayerCfg {
        hidden: HID + 1,
        ..cfg
    };
    assert_eq!(
        RopeTables::new(&bad_hidden, TMAX).unwrap_err(),
        PartsError::Shape
    );
    // n_head % n_kv_head != 0.
    let bad_split = LayerCfg {
        n_kv_head: 3,
        ..cfg
    };
    assert_eq!(
        RopeTables::new(&bad_split, TMAX).unwrap_err(),
        PartsError::Shape
    );
    // n_kv_head == 0.
    let no_kv = LayerCfg {
        n_kv_head: 0,
        ..cfg
    };
    assert_eq!(
        RopeTables::new(&no_kv, TMAX).unwrap_err(),
        PartsError::Shape
    );
    // Non-finite / non-positive eps and theta.
    let nan_eps = LayerCfg {
        eps: f64::NAN,
        ..cfg
    };
    assert_eq!(
        RopeTables::new(&nan_eps, TMAX).unwrap_err(),
        PartsError::NonFinite
    );
    let inf_theta = LayerCfg {
        theta: f64::INFINITY,
        ..cfg
    };
    assert_eq!(
        RopeTables::new(&inf_theta, TMAX).unwrap_err(),
        PartsError::NonFinite
    );
    let zero_theta = LayerCfg { theta: 0.0, ..cfg };
    assert_eq!(
        RopeTables::new(&zero_theta, TMAX).unwrap_err(),
        PartsError::NonFinite
    );

    // rms refusals.
    assert_eq!(
        rms(&[0.0; 3], &[1.0; 4], EPS, 1, 4).unwrap_err(),
        PartsError::Shape
    );
    assert_eq!(
        rms(&[0.0; 4], &[1.0; 3], EPS, 1, 4).unwrap_err(),
        PartsError::Shape
    );
    assert_eq!(
        rms(&[0.0; 4], &[1.0; 4], f64::NAN, 1, 4).unwrap_err(),
        PartsError::NonFinite
    );

    // t > t_max, and a layer whose head_dim disagrees with the tables.
    let short = RopeTables::new(&cfg, T).unwrap();
    let mut rng = Lcg::new(17);
    let w = Weights::new(&mut rng, &cfg);
    let h_long = rng.vec((T + 1) * HID);
    assert_eq!(
        parts(&cfg, &short, &w.layer(), &h_long, T + 1).unwrap_err(),
        PartsError::RopeTooShort
    );
    let other = LayerCfg {
        head_dim: HD + 2,
        hidden: NH * (HD + 2),
        ..cfg
    };
    let rope_other = RopeTables::new(&other, TMAX).unwrap();
    assert_eq!(
        attn_a(&cfg, &rope_other, &w.q, &w.k, &w.v, T).unwrap_err(),
        PartsError::Shape
    );

    // A truncated weight tensor is refused.
    let broken = LayerF64 {
        q: &w.q[1..],
        ..w.layer()
    };
    let rope = RopeTables::new(&cfg, TMAX).unwrap();
    let h = rng.vec(T * HID);
    assert_eq!(
        parts(&cfg, &rope, &broken, &h, T).unwrap_err(),
        PartsError::Shape
    );
}
