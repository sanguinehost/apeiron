//! `tests/learn_fold_icm.rs` — the `E8 ∪ O8` block-ICM refinement of the one-bit fold transfer:
//! `J` non-increasing per sweep and never above its initial value for both `e8_icm_step` and
//! `e8_joint_icm_2plane`, exact recovery of a target representable by one or two planes, the chosen
//! block code against a brute-force `J` surface, the `r_step2` normal equations, `row_chunk`
//! invariance, the `icm_ridge` metric, and the ragged-shape refusals. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps integers to `f64` by division and the flat row-major index \
              arithmetic is the reference payload format; the representability and argmin checks \
              are exact float comparisons by construction, and the `o/n/m/a_/i1/i2` names are the \
              reference's vocabulary"
)]

use apeiron::learn::code::e8::codeword;
use apeiron::learn::fold::icm::{
    e8_icm_step, e8_joint_icm_2plane, j_rows, nearest_index, plane_w, r_step2, ridged_metric,
    IcmError, Plane, PlaneRef,
};

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

const BLOCK: usize = 8;
const GROUP: usize = 128;
const CODE_LEN: usize = 256;

/// The 256 codewords in real values: `codeword(j)[k] / 2^32`, the same table the module builds.
fn codebook() -> Vec<[f64; BLOCK]> {
    let one = (1_i64 << 32) as f64;
    (0..CODE_LEN)
        .map(|j| {
            let row = codeword(j).expect("row in range");
            let mut out = [0.0_f64; BLOCK];
            for k in 0..BLOCK {
                out[k] = row[k] as f64 / one;
            }
            out
        })
        .collect()
}

fn pref(p: &Plane) -> PlaneRef<'_> {
    PlaneRef {
        s: &p.s,
        idx: &p.idx,
        r: &p.r,
        c: &p.c,
        d: &p.d,
    }
}

/// A positive-definite Toeplitz metric `Hl[i,j] = rho^|i-j|` (the autocorrelation of an AR(1) plus
/// the `j = i` term), a stand-in for the calibration Gram.
fn toeplitz(n: usize, rho: f64) -> Vec<f64> {
    let mut h = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let d = i.abs_diff(j);
            h[i * n + j] = rho.powi(i32::try_from(d).expect("d < 2^31"));
        }
    }
    h
}

/// A plane with random codebook rows, random group scales in `[0.5, 1.5)` and unit `c`, `d`.
fn rand_plane(o: usize, n: usize, rng: &mut Lcg) -> Plane {
    let nb = n / BLOCK;
    let g = n / GROUP;
    let cb = codebook();
    let mut s = vec![0.0_f64; o * n];
    let mut idx = vec![0_u16; o * nb];
    for row in 0..o {
        for m in 0..nb {
            let j = (rng.next_u64() % 256) as usize;
            idx[row * nb + m] = j as u16;
            for k in 0..BLOCK {
                s[row * n + m * BLOCK + k] = cb[j][k];
            }
        }
    }
    Plane {
        s,
        idx,
        r: (0..o * g).map(|_| rng.uniform(0.5, 1.5)).collect(),
        c: vec![1.0; n],
        d: vec![1.0; n],
    }
}

fn rand_mat(rows: usize, cols: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..rows * cols).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// `J` of one plane against `ws` in metric `hl`.
fn plane_j(ws: &[f64], p: &Plane, hl: &[f64], o: usize, n: usize) -> f64 {
    let w = plane_w(pref(p), o, n).expect("plane shape");
    j_rows(ws, &w, hl, o, n, None).expect("j shape")
}

/// `J` of a two-plane sum against `ws` in metric `hl`.
fn pair_j(ws: &[f64], p1: &Plane, p2: &Plane, hl: &[f64], o: usize, n: usize) -> f64 {
    let mut w = plane_w(pref(p1), o, n).expect("plane shape");
    let w2 = plane_w(pref(p2), o, n).expect("plane shape");
    for k in 0..o * n {
        w[k] += w2[k];
    }
    j_rows(ws, &w, hl, o, n, None).expect("j shape")
}

/// Independent `[rows, inner] × [inner, cols]` product for the normal-equation check.
fn matmul_ref(a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for k in 0..inner {
                acc += a[r * inner + k] * b[k * cols + c];
            }
            out[r * cols + c] = acc;
        }
    }
    out
}

#[test]
fn learn_fold_icm_step_j_is_non_increasing_per_sweep() {
    let n = 256;
    let hl = toeplitz(n, 0.4);
    for &o in &[4_usize, 8, 16] {
        let mut rng = Lcg::new(0x1ACE_5EED ^ (o as u64));
        let p0 = rand_plane(o, n, &mut rng);
        let ws = rand_mat(o, n, &mut rng);
        let j_init = plane_j(&ws, &p0, &hl, o, n);
        let mut prev = j_init;
        for sweeps in 1..=3 {
            let p = e8_icm_step(&ws, &hl, pref(&p0), o, n, sweeps, 0.0).expect("step");
            let j = plane_j(&ws, &p, &hl, o, n);
            assert!(
                j <= prev + 1e-9 * (1.0 + j.abs()),
                "single plane o={o} sweeps={sweeps}: J rose {prev} -> {j}"
            );
            prev = j;
        }
        assert!(
            prev <= j_init + 1e-9 * (1.0 + j_init.abs()),
            "single plane o={o}: final J {prev} > initial {j_init}"
        );
        assert!(
            prev < j_init,
            "single plane o={o}: no move at all ({j_init})"
        );
    }
}

#[test]
fn learn_fold_icm_joint_j_is_non_increasing_per_sweep() {
    let n = 256;
    let hl = toeplitz(n, 0.4);
    for &o in &[4_usize, 8, 16] {
        let mut rng = Lcg::new(0x2B1C_0DE5 ^ (o as u64));
        let p1 = rand_plane(o, n, &mut rng);
        let p2 = rand_plane(o, n, &mut rng);
        let ws = rand_mat(o, n, &mut rng);
        let j_init = pair_j(&ws, &p1, &p2, &hl, o, n);
        let mut prev = j_init;
        for sweeps in 1..=3 {
            let (q1, q2, st) = e8_joint_icm_2plane(
                &ws,
                &hl,
                pref(&p1),
                pref(&p2),
                o,
                n,
                sweeps,
                0.0,
                None,
                256,
                None,
            )
            .expect("joint");
            let j = pair_j(&ws, &q1, &q2, &hl, o, n);
            assert!(
                j <= prev + 1e-9 * (1.0 + j.abs()),
                "joint o={o} sweeps={sweeps}: J rose {prev} -> {j}"
            );
            assert!(
                st.j_after <= st.j_before + 1e-9 * (1.0 + st.j_before.abs()),
                "joint o={o}: stats J {} -> {}",
                st.j_before,
                st.j_after
            );
            assert!(
                (st.j_after - j).abs() <= 1e-9 * (1.0 + j.abs()),
                "joint o={o}: stats J {} != measured {j}",
                st.j_after
            );
            prev = j;
        }
        assert!(prev < j_init, "joint o={o}: no move at all ({j_init})");
    }
}

#[test]
fn learn_fold_icm_g_row_weights_the_reported_j() {
    let (o, n) = (5_usize, 256);
    let hl = toeplitz(n, 0.4);
    let mut rng = Lcg::new(99);
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let g_row: Vec<f64> = (0..o).map(|_| rng.uniform(0.25, 2.0)).collect();
    let unweighted = pair_j(&ws, &p1, &p2, &hl, o, n);
    let (_q1, _q2, st) = e8_joint_icm_2plane(
        &ws,
        &hl,
        pref(&p1),
        pref(&p2),
        o,
        n,
        0,
        0.0,
        Some(&g_row),
        256,
        None,
    )
    .expect("joint");
    // With sweeps = 0 nothing moves, so j_before = j_after = the weighted sum of the row norms.
    let mut w = plane_w(pref(&p1), o, n).expect("w1");
    let w2 = plane_w(pref(&p2), o, n).expect("w2");
    for k in 0..o * n {
        w[k] += w2[k];
    }
    assert_eq!(st.j_after, st.j_before);
    assert_eq!(st.moves, 0);
    // The weighted sum is the unweighted one only when the weights are all 1; check the ratio is in
    // the weight range, and the exact value against a local weighted accumulation.
    let mut expect = 0.0;
    for row in 0..o {
        let mut j = 0.0;
        for i in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += (ws[row * n + k] - w[row * n + k]) * hl[k * n + i];
            }
            j += acc * (ws[row * n + i] - w[row * n + i]);
        }
        expect += g_row[row] * j;
    }
    assert!((st.j_before - expect).abs() <= 1e-9 * (1.0 + expect.abs()));
    assert!(st.j_before > 0.0 && unweighted > 0.0);
}

#[test]
fn learn_fold_icm_step_recovers_a_representable_plane() {
    let (o, n) = (6_usize, 256);
    let hl = toeplitz(n, 0.4);
    let mut rng = Lcg::new(7);
    let truth = rand_plane(o, n, &mut rng);
    let ws = plane_w(pref(&truth), o, n).expect("plane_w");
    // Perturb the last block of row 0: on sweep 0 the order is reversed, so it is repaired first and
    // every later block then sees the exact state.
    let nb = n / BLOCK;
    let m = nb - 1;
    let cb = codebook();
    let mut init = truth.clone();
    let jp = (truth.idx[m] as usize + 1) % CODE_LEN;
    for k in 0..BLOCK {
        init.s[m * BLOCK + k] = cb[jp][k];
    }
    init.idx[m] = jp as u16;
    let j_init = plane_j(&ws, &init, &hl, o, n);
    assert!(
        j_init > 1e-6,
        "the perturbation must be visible, got {j_init}"
    );
    let out = e8_icm_step(&ws, &hl, pref(&init), o, n, 2, 0.0).expect("step");
    let j_out = plane_j(&ws, &out, &hl, o, n);
    assert!(
        j_out < 1e-6 * (1.0 + j_init),
        "J {j_init} -> {j_out}, expected ~0"
    );
    for row in 0..o {
        for blk in 0..nb {
            assert_eq!(
                out.idx[row * nb + blk],
                truth.idx[row * nb + blk],
                "row {row} block {blk} not recovered"
            );
        }
    }
}

#[test]
fn learn_fold_icm_joint_recovers_a_two_plane_representation() {
    let (o, n) = (6_usize, 256);
    let hl = toeplitz(n, 0.4);
    let mut rng = Lcg::new(13);
    let truth1 = rand_plane(o, n, &mut rng);
    let truth2 = rand_plane(o, n, &mut rng);
    let mut ws = plane_w(pref(&truth1), o, n).expect("w1");
    let w2 = plane_w(pref(&truth2), o, n).expect("w2");
    for k in 0..o * n {
        ws[k] += w2[k];
    }
    let nb = n / BLOCK;
    let m = nb - 1;
    let cb = codebook();
    let mut init1 = truth1.clone();
    let jp = (truth1.idx[m] as usize + 5) % CODE_LEN;
    for k in 0..BLOCK {
        init1.s[m * BLOCK + k] = cb[jp][k];
    }
    init1.idx[m] = jp as u16;
    let j_init = pair_j(&ws, &init1, &truth2, &hl, o, n);
    assert!(
        j_init > 1e-6,
        "the perturbation must be visible, got {j_init}"
    );
    let (out1, out2, st) = e8_joint_icm_2plane(
        &ws,
        &hl,
        pref(&init1),
        pref(&truth2),
        o,
        n,
        2,
        0.0,
        None,
        256,
        None,
    )
    .expect("joint");
    let j_out = pair_j(&ws, &out1, &out2, &hl, o, n);
    assert!(
        j_out < 1e-6 * (1.0 + j_init),
        "J {j_init} -> {j_out}, expected ~0"
    );
    assert!(st.j_after < 1e-6 * (1.0 + st.j_before));
    for row in 0..o {
        for blk in 0..nb {
            assert_eq!(out1.idx[row * nb + blk], truth1.idx[row * nb + blk]);
            assert_eq!(out2.idx[row * nb + blk], truth2.idx[row * nb + blk]);
        }
    }
}

#[test]
fn learn_fold_icm_step_block_argmin_is_exact() {
    let (o, n) = (4_usize, 256);
    let hl = toeplitz(n, 0.35);
    let mut rng = Lcg::new(11);
    let init = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let nb = n / BLOCK;
    let m = nb - 1; // first block of sweep 0 (order is reversed)
    let cb = codebook();
    let out = e8_icm_step(&ws, &hl, pref(&init), o, n, 1, 0.0).expect("step");
    // Brute force the `J` surface of row 0 block m over all 256 codewords, others held at `init`.
    let mut best = usize::MAX;
    let mut best_j = f64::INFINITY;
    for j in 0..CODE_LEN {
        let mut cand = init.clone();
        for k in 0..BLOCK {
            cand.s[m * BLOCK + k] = cb[j][k];
        }
        let jj = plane_j(&ws, &cand, &hl, o, n);
        if jj < best_j {
            best_j = jj;
            best = j;
        }
    }
    assert_eq!(
        out.idx[m] as usize, best,
        "row 0 block {m}: chose {} not the brute-force argmin {best}",
        out.idx[m]
    );
    let j_out = plane_j(&ws, &out, &hl, o, n);
    assert!(
        j_out <= best_j + 1e-9 * (1.0 + best_j.abs()),
        "J {j_out} above the brute-force block minimum {best_j}"
    );
}

#[test]
fn learn_fold_icm_joint_block_argmin_is_exact() {
    // One row keeps the 256 x 256 pair brute force affordable.
    let (o, n) = (1_usize, 128);
    let hl = toeplitz(n, 0.3);
    let mut rng = Lcg::new(23);
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let m = 5_usize;
    let (out1, out2, st) = e8_joint_icm_2plane(
        &ws,
        &hl,
        pref(&p1),
        pref(&p2),
        o,
        n,
        1,
        0.0,
        None,
        256,
        Some(&[m]),
    )
    .expect("joint");
    let cb = codebook();
    let mut best = (0_usize, 0_usize);
    let mut best_j = f64::INFINITY;
    for i1 in 0..CODE_LEN {
        let mut c1 = p1.clone();
        for k in 0..BLOCK {
            c1.s[m * BLOCK + k] = cb[i1][k];
        }
        let w1 = plane_w(pref(&c1), o, n).expect("w1");
        for i2 in 0..CODE_LEN {
            let mut c2 = p2.clone();
            for k in 0..BLOCK {
                c2.s[m * BLOCK + k] = cb[i2][k];
            }
            let w2 = plane_w(pref(&c2), o, n).expect("w2");
            let mut w = w1.clone();
            for k in 0..o * n {
                w[k] += w2[k];
            }
            let jj = j_rows(&ws, &w, &hl, o, n, None).expect("j");
            if jj < best_j {
                best_j = jj;
                best = (i1, i2);
            }
        }
    }
    assert_eq!(
        (out1.idx[m] as usize, out2.idx[m] as usize),
        best,
        "block {m}: chose {:?} not the brute-force pair argmin {best:?}",
        (out1.idx[m], out2.idx[m])
    );
    assert!(
        (st.j_after - best_j).abs() <= 1e-9 * (1.0 + best_j.abs()),
        "joint J {} != brute-force block minimum {best_j}",
        st.j_after
    );
}

#[test]
fn learn_fold_icm_r_step2_satisfies_normal_equations() {
    let (o, n) = (5_usize, 256);
    let hl = toeplitz(n, 0.45);
    let mut rng = Lcg::new(31);
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let (r1, r2) = r_step2(&ws, &hl, pref(&p1), pref(&p2), o, n).expect("r_step2");
    let g = n / GROUP;
    assert_eq!(r1.len(), o * g);
    assert_eq!(r2.len(), o * g);
    // P[o][a_][j]: a_ < G plane 1 block a_, else plane 2 block a_ - G.
    let mut pmat = vec![0.0_f64; o * 2 * g * GROUP];
    for row in 0..o {
        for k in 0..2 {
            let (s, c) = if k == 0 {
                (&p1.s, &p1.c)
            } else {
                (&p2.s, &p2.c)
            };
            for gg in 0..g {
                for j in 0..GROUP {
                    pmat[((row * 2 * g) + k * g + gg) * GROUP + j] =
                        s[row * n + gg * GROUP + j] * c[gg * GROUP + j];
                }
            }
        }
    }
    let wh = matmul_ref(&ws, &hl, o, n, n);
    let mut worst = 0.0_f64;
    for row in 0..o {
        let mut x = vec![0.0_f64; 2 * g];
        for k in 0..g {
            x[k] = r1[row * g + k];
            x[g + k] = r2[row * g + k];
        }
        for a_ in 0..2 * g {
            let ga = a_ % g;
            let mut t = 0.0;
            for j in 0..GROUP {
                t += pmat[(row * 2 * g + a_) * GROUP + j] * wh[row * n + ga * GROUP + j];
            }
            let mut mx = 0.0;
            for bcol in 0..2 * g {
                let gb = bcol % g;
                let mut m = 0.0;
                for tt in 0..GROUP {
                    for j in 0..GROUP {
                        m += pmat[(row * 2 * g + a_) * GROUP + tt]
                            * hl[(ga * GROUP + tt) * n + gb * GROUP + j]
                            * pmat[(row * 2 * g + bcol) * GROUP + j];
                    }
                }
                mx += m * x[bcol];
            }
            let resid = (mx - t).abs();
            if resid > worst {
                worst = resid;
            }
        }
    }
    assert!(
        worst <= 1e-9,
        "r_step2 normal-equation residual {worst} > 1e-9"
    );
}

#[test]
fn learn_fold_icm_icm_ridge_metric_is_non_increasing() {
    let (o, n) = (4_usize, 256);
    let hl = toeplitz(n, 0.4);
    let hr = ridged_metric(&hl, n, 0.2).expect("ridged");
    for i in 0..n {
        assert!(
            hr[i * n + i] > hl[i * n + i],
            "the ridge must raise the diagonal"
        );
    }
    let mut rng = Lcg::new(41);
    let p0 = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let j_init = plane_j(&ws, &p0, &hr, o, n);
    let mut prev = j_init;
    for sweeps in 1..=2 {
        let p = e8_icm_step(&ws, &hl, pref(&p0), o, n, sweeps, 0.2).expect("step");
        let j = plane_j(&ws, &p, &hr, o, n);
        assert!(
            j <= prev + 1e-9 * (1.0 + j.abs()),
            "ridged sweeps={sweeps}: J rose {prev} -> {j}"
        );
        prev = j;
    }
    let (q1, q2, st) = e8_joint_icm_2plane(
        &ws,
        &hl,
        pref(&p0),
        pref(&p0),
        o,
        n,
        1,
        0.2,
        None,
        256,
        None,
    )
    .expect("joint");
    let j = pair_j(&ws, &q1, &q2, &hr, o, n);
    assert!(j <= st.j_before + 1e-9 * (1.0 + st.j_before.abs()));
    assert!((st.j_after - j).abs() <= 1e-9 * (1.0 + j.abs()));
}

#[test]
fn learn_fold_icm_row_chunk_does_not_change_the_result() {
    let (o, n) = (8_usize, 256);
    let hl = toeplitz(n, 0.4);
    let mut rng = Lcg::new(53);
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let (a1, a2, ast) =
        e8_joint_icm_2plane(&ws, &hl, pref(&p1), pref(&p2), o, n, 2, 0.0, None, 1, None)
            .expect("chunk 1");
    let (b1, b2, bst) = e8_joint_icm_2plane(
        &ws,
        &hl,
        pref(&p1),
        pref(&p2),
        o,
        n,
        2,
        0.0,
        None,
        256,
        None,
    )
    .expect("chunk 256");
    assert_eq!(a1.s, b1.s);
    assert_eq!(a1.idx, b1.idx);
    assert_eq!(a2.s, b2.s);
    assert_eq!(a2.idx, b2.idx);
    assert_eq!(ast, bst);
}

#[test]
fn learn_fold_icm_nearest_index_matches_the_codebook_rule() {
    let cb = codebook();
    for j in 0..CODE_LEN {
        assert_eq!(nearest_index(&cb[j]), Ok(j), "row {j}");
    }
}

#[test]
fn learn_fold_icm_refuses_ragged_shapes() {
    let (o, n) = (2_usize, 256);
    let hl = toeplitz(n, 0.4);
    let p = rand_plane(o, n, &mut Lcg::new(3));
    let ws = rand_mat(o, n, &mut Lcg::new(5));
    let good = plane_w(pref(&p), o, n).expect("plane_w");
    assert_eq!(good.len(), o * n);

    // A short `ws`, `hl`, or any plane slice is a shape refusal.
    assert_eq!(
        e8_icm_step(&ws[1..], &hl, pref(&p), o, n, 1, 0.0).err(),
        Some(IcmError::Shape)
    );
    assert_eq!(
        e8_icm_step(&ws, &hl[1..], pref(&p), o, n, 1, 0.0).err(),
        Some(IcmError::Shape)
    );
    let short_s = PlaneRef {
        s: &p.s[1..],
        ..pref(&p)
    };
    assert_eq!(
        e8_icm_step(&ws, &hl, short_s, o, n, 1, 0.0).err(),
        Some(IcmError::Shape)
    );
    let short_idx = PlaneRef {
        idx: &p.idx[1..],
        ..pref(&p)
    };
    assert_eq!(
        e8_joint_icm_2plane(&ws, &hl, short_idx, pref(&p), o, n, 1, 0.0, None, 256, None,).err(),
        Some(IcmError::Shape)
    );
    let short_r = PlaneRef {
        r: &p.r[1..],
        ..pref(&p)
    };
    assert_eq!(
        e8_icm_step(&ws, &hl, short_r, o, n, 1, 0.0).err(),
        Some(IcmError::Shape)
    );
    let short_c = PlaneRef {
        c: &p.c[1..],
        ..pref(&p)
    };
    assert_eq!(
        r_step2(&ws, &hl, short_c, pref(&p), o, n).err(),
        Some(IcmError::Shape)
    );
    let short_d = PlaneRef {
        d: &p.d[1..],
        ..pref(&p)
    };
    assert_eq!(
        e8_icm_step(&ws, &hl, short_d, o, n, 1, 0.0).err(),
        Some(IcmError::Shape)
    );
    assert_eq!(
        r_step2(&ws, &hl[1..], pref(&p), pref(&p), o, n).err(),
        Some(IcmError::Shape)
    );
    assert_eq!(
        j_rows(&ws, &good, &hl, o, n, Some(&[1.0])).err(),
        Some(IcmError::Shape)
    );

    // `n` must be a positive multiple of BLOCK and GROUP.
    let bad_n = 64_usize;
    let q = Plane {
        s: vec![0.0; o * bad_n],
        idx: vec![0; o * (bad_n / BLOCK)],
        r: Vec::new(),
        c: vec![1.0; bad_n],
        d: vec![1.0; bad_n],
    };
    let bad_ws = vec![0.0; o * bad_n];
    let bad_hl = vec![0.0; bad_n * bad_n];
    assert_eq!(
        e8_icm_step(&bad_ws, &bad_hl, pref(&q), o, bad_n, 1, 0.0).err(),
        Some(IcmError::Width)
    );
    assert_eq!(plane_w(pref(&q), o, bad_n).err(), Some(IcmError::Width));

    // Zero chunk and out-of-range block subset.
    assert_eq!(
        e8_joint_icm_2plane(&ws, &hl, pref(&p), pref(&p), o, n, 1, 0.0, None, 0, None).err(),
        Some(IcmError::ZeroChunk)
    );
    let nb = n / BLOCK;
    assert_eq!(
        e8_joint_icm_2plane(
            &ws,
            &hl,
            pref(&p),
            pref(&p),
            o,
            n,
            1,
            0.0,
            None,
            256,
            Some(&[nb])
        )
        .err(),
        Some(IcmError::BlockOutOfRange)
    );
}
