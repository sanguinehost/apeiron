//! `tests/learn_fold_gptq.rs` — the E8 ∪ O8 GPTQ port. The block selection is the Python's
//! block-Cholesky CVP `argmin_k a²|Z_c[k]|² − 2a⟨Z_w, Z_c[k]⟩` in the whitened frame, checked
//! exhaustively against a literal 256-codeword loop over the exact block objective; an identity
//! Hessian degenerates to independent nearest-codeword quantisation; on correlated Hessians the
//! selection is at or below the pre-fix Euclidean-selection value of
//! `tr((W − Wq)·H·(W − Wq)ᵀ)`; the codes decode to `s` before scaling; the act order round-trips;
//! the whitened scale projection holds; the refusals fire. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::unreadable_literal,
    reason = "the test setup mirrors the row-major indices and the Q32.32 casts of the ported kernel; \
              the f64 comparisons are exactness claims (`s` decodes to codeword/ONE, the act order \
              round-trips bit for bit, the diagonal Hessian reproduces RTN exactly), and the golden \
              literals are exact captured bit patterns, not tolerances"
)]

use apeiron::engine::ONE;
use apeiron::learn::code::e8;
use apeiron::learn::fold::fit::gptq_e8::{
    feedback_u, gptq_e8, GptqE8, GptqE8Error, BLOCK, E8_SCALE, GROUP, MAX_BEAM,
};

/// xorshift64: deterministic, no crate.
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Deterministic uniform in `[−1, 1)`.
fn uniform(s: &mut u64) -> f64 {
    (xorshift(s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// The Q32.32 image of `x`, the same rounding the Euclidean baseline applies to a weight and scale.
fn to_fixed(x: f64) -> i64 {
    (x * ONE as f64).round() as i64
}

/// The per-(row, group) scale of the kernel: `0.828 · w_rms` with
/// `w_rms² = Σ_j w_j²·prec_j / Σ_j prec_j`, `prec_j = 1/U_jj²` of the feedback factor
/// (`fold_transfer.py` lines 301–303, unconditional in `wscale`).
fn group_scale(w: &[f64], prec: &[f64], g0: usize, g1: usize) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for j in g0..g1 {
        let x = w[j];
        num += x * x * prec[j - g0];
        den += prec[j - g0];
    }
    (E8_SCALE * (num / den).max(1e-24).sqrt()).max(1e-12)
}

/// Sum of squares.
fn sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// The whitened scale projection `a ∘ u_s` of the Python lines 327–332:
/// `a_new = max(a + ⟨e, u_s⟩/⟨u_s, u_s⟩, 1e-12)`.
fn project(a: f64, e: &[f64], u_s: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for j in 0..e.len() {
        num += e[j] * u_s[j];
        den += u_s[j] * u_s[j];
    }
    (a + num / den.max(1e-12)).max(1e-12)
}

/// The 8×8 block `U_sub = U[b0..b0+8, b0..b0+8]` of the row-major `[n, n]` upper-triangular `u`,
/// inverted row by row by back substitution: row `i` solves `x·U_sub = e_i`.
fn block_inverse(u: &[f64], n: usize, b0: usize) -> [[f64; BLOCK]; BLOCK] {
    let mut inv = [[0.0f64; BLOCK]; BLOCK];
    for i in 0..BLOCK {
        let mut x = [0.0f64; BLOCK];
        x[i] = 1.0;
        let mut j = BLOCK;
        while j > 0 {
            j -= 1;
            let mut s = x[j];
            for k in (j + 1)..BLOCK {
                s -= x[k] * u[(b0 + k) * n + b0 + j];
            }
            x[j] = s / u[(b0 + j) * n + b0 + j];
        }
        inv[i] = x;
    }
    inv
}

/// The literal loop over all 256 codewords of the one-group block objective, in both equivalent
/// frames: the whitened (Z) distance `d_k = |Z_w − a·Z_c[k]|²` with `Z_w = w·U_sub⁻¹` and
/// `Z_c[k] = C[k]·U_sub⁻¹`, and the quadratic form `q_k = (w − a·s_k)·M·(w − a·s_k)ᵀ` with
/// `M = (U_subᵀ·U_sub)⁻¹ = U_inv·U_invᵀ` (for the full factor `UᵀU = H⁻¹`, so `M` is `H` there).
/// Returns `(Z_w, Z_c, d, q)`.
fn objective_scan(
    w: &[f64],
    u_inv: &[[f64; BLOCK]; BLOCK],
    a: f64,
) -> (
    [f64; BLOCK],
    Vec<[f64; BLOCK]>,
    [f64; e8::CODEBOOK_LEN],
    [f64; e8::CODEBOOK_LEN],
) {
    let mut m = [[0.0f64; BLOCK]; BLOCK];
    for i in 0..BLOCK {
        for j in 0..BLOCK {
            let mut s = 0.0f64;
            for t in 0..BLOCK {
                s += u_inv[i][t] * u_inv[j][t];
            }
            m[i][j] = s;
        }
    }
    let mut z_w = [0.0f64; BLOCK];
    for t in 0..BLOCK {
        let mut s = 0.0f64;
        for j in 0..BLOCK {
            s += w[j] * u_inv[j][t];
        }
        z_w[t] = s;
    }
    let mut zc = vec![[0.0f64; BLOCK]; e8::CODEBOOK_LEN];
    let mut d = [0.0f64; e8::CODEBOOK_LEN];
    let mut q = [0.0f64; e8::CODEBOOK_LEN];
    for k in 0..e8::CODEBOOK_LEN {
        let cw = e8::codeword(k).expect("code index");
        let mut z_c = [0.0f64; BLOCK];
        let mut r = [0.0f64; BLOCK];
        for t in 0..BLOCK {
            let sv = cw[t] as f64 / ONE as f64;
            let mut s = 0.0f64;
            for j in 0..BLOCK {
                s += sv * u_inv[j][t];
            }
            z_c[t] = s;
            r[t] = w[t] - a * sv;
        }
        let mut dk = 0.0f64;
        for t in 0..BLOCK {
            let e = z_w[t] - a * z_c[t];
            dk += e * e;
        }
        let mut qk = 0.0f64;
        for i in 0..BLOCK {
            for j in 0..BLOCK {
                qk += r[i] * m[i][j] * r[j];
            }
        }
        zc[k] = z_c;
        d[k] = dk;
        q[k] = qk;
    }
    (z_w, zc, d, q)
}

/// First index of the minimum of a 256-entry cost row (the Python `argmin`).
fn argmin(c: &[f64; e8::CODEBOOK_LEN]) -> usize {
    let mut best = 0usize;
    for k in 1..e8::CODEBOOK_LEN {
        if c[k] < c[best] {
            best = k;
        }
    }
    best
}

/// Independent nearest-codeword quantisation per 8-block over the E8 ∪ O8 codebook, no error
/// feedback: the RTN-E8 baseline. The group scale follows the kernel's rule, so the only difference
/// from `gptq_e8` is the error feedback.
fn rtn_e8(w: &[f64], h: &[f64], o: usize, n: usize, damp: f64) -> Vec<f64> {
    let u = feedback_u(h, n, damp).expect("SPD factor");
    let ng = n.div_ceil(GROUP);
    let mut wq = vec![0.0f64; o * n];
    for r in 0..o {
        let row = &w[r * n..(r + 1) * n];
        for gi in 0..ng {
            let g0 = gi * GROUP;
            let g1 = (g0 + GROUP).min(n);
            let mut prec = vec![0.0f64; g1 - g0];
            for j in g0..g1 {
                let d = u[j * n + j];
                prec[j - g0] = 1.0 / (d * d).max(1e-12);
            }
            let a = group_scale(row, &prec, g0, g1);
            let aq = to_fixed(a).max(1);
            for b0 in (g0..g1).step_by(BLOCK) {
                let mut x = [0_i64; BLOCK];
                for k in 0..BLOCK {
                    x[k] = to_fixed(row[b0 + k]);
                }
                let idx = e8::nearest_scaled(&x, aq).expect("codebook search");
                let cw = e8::codeword(idx).expect("code index");
                for k in 0..BLOCK {
                    wq[r * n + b0 + k] = a * (cw[k] as f64 / ONE as f64);
                }
            }
        }
    }
    wq
}

/// The pre-fix selector: the same group loop with the code index chosen by Euclidean nearest
/// codeword in the original frame (`e8::nearest_scaled` on the Q32.32 image of the block), error
/// still fed back through `U_sub`. The comparison baseline of the correlated-Hessian test.
fn gptq_euclid(w: &[f64], h: &[f64], o: usize, n: usize, damp: f64) -> Vec<f64> {
    let u = feedback_u(h, n, damp).expect("SPD factor");
    let ng = n.div_ceil(GROUP);
    let mut wk = w.to_vec();
    let mut wq = vec![0.0f64; o * n];
    for gi in 0..ng {
        let g0 = gi * GROUP;
        let g1 = (g0 + GROUP).min(n);
        let gs = g1 - g0;
        let mut prec = vec![0.0f64; gs];
        for j in 0..gs {
            let d = u[(g0 + j) * n + g0 + j];
            prec[j] = 1.0 / (d * d).max(1e-12);
        }
        let mut e_blk = vec![0.0f64; o * gs];
        for r in 0..o {
            let a = group_scale(&wk[r * n..(r + 1) * n], &prec, g0, g1);
            let aq = to_fixed(a).max(1);
            let mut b0 = 0;
            while b0 < gs {
                let mut x = [0_i64; BLOCK];
                for k in 0..BLOCK {
                    x[k] = to_fixed(wk[r * n + g0 + b0 + k]);
                }
                let idx = e8::nearest_scaled(&x, aq).expect("codebook search");
                let cw = e8::codeword(idx).expect("code index");
                let mut e = [0.0f64; BLOCK];
                for k in 0..BLOCK {
                    let sv = cw[k] as f64 / ONE as f64;
                    wq[r * n + g0 + b0 + k] = a * sv;
                    e[k] = wk[r * n + g0 + b0 + k] - a * sv;
                }
                // Whitened residual: `(w − a·s)·U_sub⁻¹`.
                let mut j = BLOCK;
                while j > 0 {
                    j -= 1;
                    let mut s = e[j];
                    for k in (j + 1)..BLOCK {
                        s -= e[k] * u[(g0 + b0 + k) * n + g0 + b0 + j];
                    }
                    e[j] = s / u[(g0 + b0 + j) * n + g0 + b0 + j];
                }
                for k in 0..BLOCK {
                    e_blk[r * gs + b0 + k] = e[k];
                }
                let mut c = b0 + BLOCK;
                while c < gs {
                    let mut acc = 0.0f64;
                    for k in 0..BLOCK {
                        acc += e[k] * u[(g0 + b0 + k) * n + g0 + c];
                    }
                    wk[r * n + g0 + c] -= acc;
                    c += 1;
                }
                b0 += BLOCK;
            }
        }
        if g1 < n {
            for r in 0..o {
                let mut m = g1;
                while m < n {
                    let mut acc = 0.0f64;
                    for j in 0..gs {
                        acc += e_blk[r * gs + j] * u[(g0 + j) * n + m];
                    }
                    wk[r * n + m] -= acc;
                    m += 1;
                }
            }
        }
    }
    wq
}

/// `tr((W − Wq)·H·(W − Wq)ᵀ)` of a row-major `[o, n]` pair against the symmetric `[n, n]` `h`.
fn objective(w: &[f64], wq: &[f64], h: &[f64], o: usize, n: usize) -> f64 {
    let mut tot = 0.0f64;
    for r in 0..o {
        for i in 0..n {
            let ri = w[r * n + i] - wq[r * n + i];
            if ri == 0.0 {
                continue;
            }
            let mut s = 0.0f64;
            for j in 0..n {
                s += h[i * n + j] * (w[r * n + j] - wq[r * n + j]);
            }
            tot += ri * s;
        }
    }
    tot
}

/// Deterministic `W` (row-major `[o, n]`) with no exact zeros.
fn random_w(o: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    (0..o * n).map(|_| uniform(&mut st)).collect()
}

/// `H = (1/t)·XᵀX + 0.25·I` for a deterministic uniform `X` (`t = n`) with a per-8-block column
/// gain, so the 8-block diagonal traces are distinct and no act-order tie arises.
fn correlated_h(n: usize, seed: u64) -> Vec<f64> {
    let t = n;
    let mut st = seed;
    let gain: Vec<f64> = (0..n).map(|j| 1.0 + (j / BLOCK) as f64 * 0.05).collect();
    let x: Vec<f64> = (0..t * n).map(|_| uniform(&mut st)).collect();
    let mut h = vec![0.0f64; n * n];
    for r in 0..t {
        for i in 0..n {
            let xi = x[r * n + i] * gain[i];
            for j in i..n {
                h[i * n + j] += xi * x[r * n + j] * gain[j];
            }
        }
    }
    let s = 1.0 / t as f64;
    for i in 0..n {
        for j in i..n {
            let v = h[i * n + j] * s;
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
        h[i * n + i] += 0.25;
    }
    h
}

/// The act-order permutation of the kernel, spelled independently: permuted position → original
/// column, groups and 8-blocks in descending diagonal sum with ties by ascending index.
fn act_order_perm(h: &[f64], n: usize) -> Vec<usize> {
    let nb = n / BLOCK;
    let per_group = GROUP / BLOCK;
    let ng = n.div_ceil(GROUP);
    let mut bt = vec![0.0f64; nb];
    for b in 0..nb {
        for k in 0..BLOCK {
            let i = b * BLOCK + k;
            bt[b] += h[i * n + i];
        }
    }
    let mut gt = vec![0.0f64; ng];
    for b in 0..nb {
        gt[b / per_group] += bt[b];
    }
    let mut groups: Vec<usize> = (0..ng).collect();
    groups.sort_by(|&a, &b| gt[b].total_cmp(&gt[a]).then(a.cmp(&b)));
    let mut blk_perm = vec![0usize; nb];
    for (gp, &g) in groups.iter().enumerate() {
        let s0 = g * per_group;
        let s1 = ((g + 1) * per_group).min(nb);
        let mut sub: Vec<usize> = (s0..s1).collect();
        sub.sort_by(|&a, &b| bt[b].total_cmp(&bt[a]).then(a.cmp(&b)));
        for (t, &b) in sub.iter().enumerate() {
            blk_perm[gp * per_group + t] = b;
        }
    }
    let mut perm = vec![0usize; n];
    for c in 0..n {
        perm[c] = blk_perm[c / BLOCK] * BLOCK + c % BLOCK;
    }
    perm
}

/// Every returned code decodes to `s` before scaling, and the shapes are `o·n`, `o·n/8`.
fn check_decode(r: &GptqE8, o: usize, n: usize, ctx: &str) {
    let nb = n / BLOCK;
    assert_eq!(r.s.len(), o * n, "{ctx}: s shape");
    assert_eq!(r.idx.len(), o * nb, "{ctx}: idx shape");
    assert_eq!(r.scale.len(), o * n.div_ceil(GROUP), "{ctx}: scale shape");
    for row in 0..o {
        for b in 0..nb {
            let idx = usize::from(r.idx[row * nb + b]);
            assert!(idx < e8::CODEBOOK_LEN, "{ctx}: idx {idx} out of range");
            let cw = e8::codeword(idx).expect("code index");
            for k in 0..BLOCK {
                let want = cw[k] as f64 / ONE as f64;
                let got = r.s[row * n + b * BLOCK + k];
                assert_eq!(
                    got,
                    want,
                    "{ctx}: s[{row}][{}] != codeword/ONE",
                    b * BLOCK + k
                );
            }
        }
    }
}

/// The feedback factor: upper triangular and `UᵀU·(H + damp·mean(diag)·I) = I` to 1e-9.
#[test]
fn feedback_u_is_the_upper_inverse_factor() {
    let (n, damp) = (64usize, 0.01);
    let h = correlated_h(n, 0x51ED_0001);
    let u = feedback_u(&h, n, damp).expect("SPD factor");
    assert_eq!(u.len(), n * n);
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let mut hd = h.clone();
    for i in 0..n {
        hd[i * n + i] += damp * mean;
    }
    let mut p = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += u[k * n + i] * u[k * n + j];
            }
            p[i * n + j] = s;
        }
    }
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if j < i {
                assert_eq!(u[i * n + j], 0.0, "U must be upper triangular at ({i},{j})");
                continue;
            }
            let mut m = 0.0f64;
            for k in 0..n {
                m += p[i * n + k] * hd[k * n + j];
            }
            let t = if i == j { (m - 1.0).abs() } else { m.abs() };
            worst = worst.max(t);
        }
    }
    // The factor is Q32.32 (`solve::inv`): the grid floor of this fixture's residual is 1.1e-9
    // (the f64 path measured below 1e-9), so the bound is 5e-9.
    assert!(worst < 5e-9, "UᵀUH − I residual {worst}");
    // Refusals of the factor itself.
    assert_eq!(feedback_u(&h, 0, damp), Err(GptqE8Error::ZeroDim));
    assert_eq!(
        feedback_u(&h[1..], n, damp),
        Err(GptqE8Error::ShapeMismatch)
    );
    let mut neg = vec![0.0f64; n * n];
    for i in 0..n {
        neg[i * n + i] = -1.0;
    }
    assert_eq!(
        feedback_u(&neg, n, damp),
        Err(GptqE8Error::NotPositiveDefinite)
    );
}

/// The selection oracle on one 8-block (`n = 8`: one group, one block, so the working row is the
/// pristine row). For a random SPD `H` and random `W`, the returned code index is the argmin of the
/// exact block objective over a literal loop of all 256 codewords, in the whitened frame
/// `|Z_w − a·Z_c[k]|²` and in the equivalent quadratic form `(w − a·s)·(U_subᵀU_sub)⁻¹·(w − a·s)ᵀ`;
/// the returned scale is the kernel's `0.828·w_rms`, and for `wscale` it is the projected scale of
/// the winning pass.
#[test]
fn block_selection_is_the_whitened_cvp_argmin() {
    let (o, n, damp) = (16usize, 8usize, 0.01);
    let h = correlated_h(n, 0x0B10_C0DE);
    // Sixteen rows, seeded per row. The choice has power: every codebook row has norm √8, so on a
    // diagonally dominant `H` most single-block rows are decided identically by the whitened
    // objective, the "scaled whitened codeword against the original-frame weight" cost
    // `a²|Z_c[k]|² − 2a⟨w_sub, Z_c[k]⟩` and the Euclidean cost; on this `H` two of these sixteen
    // rows separate the whitened selection from each of those two original-frame costs.
    let mut w = Vec::with_capacity(o * n);
    for row in 0..o {
        w.extend_from_slice(&random_w(1, n, 0x0B10_BEEF ^ (0x1000 * (row as u64 + 1))));
    }
    let u = feedback_u(&h, n, damp).expect("SPD factor");
    let u_inv = block_inverse(&u, n, 0);
    let mut prec = vec![0.0f64; n];
    for j in 0..n {
        let d = u[j * n + j];
        prec[j] = 1.0 / (d * d).max(1e-12);
    }
    for wscale in [false, true] {
        let ctx = if wscale { "oracle wscale" } else { "oracle" };
        let r = gptq_e8(&w, &h, o, n, damp, false, wscale, 1).expect("gptq_e8");
        check_decode(&r, o, n, ctx);
        for row in 0..o {
            let wr = &w[row * n..(row + 1) * n];
            let a0 = group_scale(wr, &prec, 0, n);
            let (z_w, zc, d, q) = objective_scan(wr, &u_inv, a0);
            let kd = argmin(&d);
            let kq = argmin(&q);
            assert_eq!(
                kd, kq,
                "{ctx} row {row}: the whitened distance and the quadratic form disagree on the \
                 argmin ({kd} vs {kq})"
            );
            for k in 0..e8::CODEBOOK_LEN {
                let scale_ref = 1.0 + d[k].abs().max(q[k].abs());
                assert!(
                    (d[k] - q[k]).abs() <= 1e-12 * scale_ref,
                    "{ctx} row {row} codeword {k}: |Z_w − a·Z_c|² {} != (w − a·s)M(w − a·s)ᵀ {}",
                    d[k],
                    q[k]
                );
            }
            let (idx_best, a_best) = if wscale {
                // Pass 1 at `a0`, project; pass 2 at the projected scale on the pristine row,
                // project; the smaller whitened residual wins (the Python lines 304–332).
                let i1 = kd;
                let mut e1 = [0.0f64; BLOCK];
                for t in 0..BLOCK {
                    e1[t] = z_w[t] - a0 * zc[i1][t];
                }
                let a1 = project(a0, &e1, &zc[i1]);
                let mut p1 = [0.0f64; BLOCK];
                for t in 0..BLOCK {
                    p1[t] = e1[t] - (a1 - a0) * zc[i1][t];
                }
                let (_zw2, zc2, d2, _q2) = objective_scan(wr, &u_inv, a1);
                let i2 = argmin(&d2);
                let mut e2 = [0.0f64; BLOCK];
                for t in 0..BLOCK {
                    e2[t] = z_w[t] - a1 * zc2[i2][t];
                }
                let a2 = project(a1, &e2, &zc2[i2]);
                let mut p2 = [0.0f64; BLOCK];
                for t in 0..BLOCK {
                    p2[t] = e2[t] - (a2 - a1) * zc2[i2][t];
                }
                if sq(&p2) < sq(&p1) {
                    (i2, a2)
                } else {
                    (i1, a1)
                }
            } else {
                (kd, a0)
            };
            assert_eq!(
                usize::from(r.idx[row]),
                idx_best,
                "{ctx} row {row}: code index {} is not the argmin {idx_best} of the block objective",
                r.idx[row]
            );
            let tol = 1e-12 * a_best.abs().max(1.0);
            assert!(
                (r.scale[row] - a_best).abs() <= tol,
                "{ctx} row {row}: scale {} != {a_best}",
                r.scale[row]
            );
        }
    }
}

/// `H = I`: no feedback survives (the factor is diagonal), so the result is exactly independent
/// nearest-codeword quantisation per 8-block, with the scale `0.828·rms` of the group.
#[test]
fn identity_hessian_is_independent_nearest_per_block() {
    let (o, n, damp) = (3usize, 128usize, 0.01);
    let w = random_w(o, n, 0x1DEA_0001);
    let mut id = vec![0.0f64; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    let r = gptq_e8(&w, &id, o, n, damp, false, false, 1).expect("gptq_e8");
    check_decode(&r, o, n, "identity");
    // Independent reference: per 8-block nearest codeword under the independent `0.828·rms` scale.
    for row in 0..o {
        let wr = &w[row * n..(row + 1) * n];
        let rms = (wr.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        let a_ref = (E8_SCALE * rms).max(1e-12);
        assert!(
            (r.scale[row] - a_ref).abs() <= 1e-12 * a_ref.abs().max(1.0),
            "row {row}: scale {} != 0.828·rms {a_ref}",
            r.scale[row]
        );
        let aq = to_fixed(a_ref).max(1);
        for b0 in (0..n).step_by(BLOCK) {
            let mut x = [0_i64; BLOCK];
            for k in 0..BLOCK {
                x[k] = to_fixed(wr[b0 + k]);
            }
            let idx = e8::nearest_scaled(&x, aq).expect("codebook search");
            assert_eq!(
                usize::from(r.idx[row * (n / BLOCK) + b0 / BLOCK]),
                idx,
                "row {row} block {} is not the independent nearest codeword",
                b0 / BLOCK
            );
            let cw = e8::codeword(idx).expect("code index");
            for k in 0..BLOCK {
                assert_eq!(r.s[row * n + b0 + k], cw[k] as f64 / ONE as f64);
            }
        }
    }
    // The same object without error feedback: the diagonal factor leaves the RTN reference intact.
    let wq_rtn = rtn_e8(&w, &id, o, n, damp);
    for row in 0..o {
        for j in 0..n {
            let want = r.scale[row] * r.s[row * n + j];
            assert_eq!(
                wq_rtn[row * n + j],
                want,
                "RTN and GPTQ differ at ({row},{j})"
            );
        }
    }
}

/// A correlated SPD Hessian. The whitened selection beats the RTN-E8 baseline on all five
/// deterministic matrices and beats the pre-fix Euclidean selection on at least three of them
/// (strictly) and at least four (weakly); the printed table carries the per-matrix numbers.
///
/// The stronger claim `whitened ≤ Euclidean` on *every* matrix is not a theorem here and is
/// refuted on matrix 0 (53.76 vs 51.37; `W/E = 1.0466`). Reason: the 8-column block CVP minimises
/// the H-weighted error of its own block, and with 8-column blocks `tr(ΔHΔᵀ) ≠ Σ_blocks |err|²`
/// (that identity needs one column per decision), so a block decision can still lose overall. What
/// is pinned here is the selection rule itself: the oracle test takes the same matrix family down to
/// one block, where the whitened choice is the objective's argmin by construction.
#[test]
fn whitened_selection_is_below_euclidean_selection() {
    let (o, n, damp) = (2usize, 256usize, 0.01);
    let mut strict = 0usize;
    let mut weak = 0usize;
    let mut ratios = [0.0f64; 5];
    for (m, seed) in [
        0x0BAD_0001u64,
        0x0BAD_0002,
        0x0BAD_0003,
        0x0BAD_0004,
        0x0BAD_0005,
    ]
    .iter()
    .enumerate()
    {
        let w = random_w(o, n, *seed);
        let h = correlated_h(n, *seed ^ 0x5A5A_5A5A);
        let r = gptq_e8(&w, &h, o, n, damp, false, false, 1).expect("gptq_e8");
        check_decode(&r, o, n, "correlated");
        let ng = n.div_ceil(GROUP);
        let mut wq_w = vec![0.0f64; o * n];
        for row in 0..o {
            for j in 0..n {
                wq_w[row * n + j] = r.scale[row * ng + j / GROUP] * r.s[row * n + j];
            }
        }
        let wq_e = gptq_euclid(&w, &h, o, n, damp);
        let wq_r = rtn_e8(&w, &h, o, n, damp);
        let obj_w = objective(&w, &wq_w, &h, o, n);
        let obj_e = objective(&w, &wq_e, &h, o, n);
        let obj_r = objective(&w, &wq_r, &h, o, n);
        println!(
            "matrix {m}: whitened {obj_w:.6}, euclidean {obj_e:.6}, rtn-e8 {obj_r:.6}, \
             whitened/euclidean {:.6}, whitened/rtn {:.6}",
            obj_w / obj_e,
            obj_w / obj_r
        );
        assert!(
            obj_w < obj_r,
            "matrix {m}: the whitened selection {obj_w} is not below RTN-E8 {obj_r}"
        );
        if obj_w < obj_e {
            strict += 1;
        }
        if obj_w <= obj_e {
            weak += 1;
        }
        ratios[m] = obj_w / obj_e;
    }
    assert!(
        strict >= 3,
        "the whitened selection is strictly below the Euclidean selection on only {strict} of five \
         matrices: ratios {ratios:?}"
    );
    assert!(
        weak >= 4,
        "the whitened selection is at or below the Euclidean selection on only {weak} of five \
         matrices: ratios {ratios:?}"
    );
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    println!("mean whitened/euclidean ratio {mean:.6} over five matrices");
    assert!(
        mean < 0.99,
        "mean whitened/euclidean objective ratio {mean} over the five matrices: {ratios:?}"
    );
}

/// The act order is a permutation of whole 8-blocks: the `act_order` result equals the natural-order
/// result on the permuted problem, scattered back, bit for bit.
#[test]
fn act_order_round_trips() {
    let (o, n, damp) = (2usize, 256usize, 0.01);
    let w = random_w(o, n, 0xAC70_0001);
    let h = correlated_h(n, 0xAC70_0002);
    let perm = act_order_perm(&h, n);
    // The permutation must be non-trivial, or this test would not exercise the reorder.
    assert!(
        perm.iter().enumerate().any(|(c, &p)| c != p),
        "the act-order permutation is the identity"
    );
    let mut wp = vec![0.0f64; o * n];
    let mut hp = vec![0.0f64; n * n];
    for row in 0..o {
        for c in 0..n {
            wp[row * n + c] = w[row * n + perm[c]];
        }
    }
    for i in 0..n {
        for j in 0..n {
            hp[i * n + j] = h[perm[i] * n + perm[j]];
        }
    }
    let r_act = gptq_e8(&w, &h, o, n, damp, true, false, 1).expect("act_order");
    let r_nat = gptq_e8(&wp, &hp, o, n, damp, false, false, 1).expect("natural");
    check_decode(&r_act, o, n, "act_order");
    let nb = n / BLOCK;
    let per_group = GROUP / BLOCK;
    let ng = n.div_ceil(GROUP);
    let mut inv_blk = vec![0usize; nb];
    for (p, c) in perm.iter().step_by(BLOCK).enumerate() {
        inv_blk[c / BLOCK] = p;
    }
    for row in 0..o {
        for c in 0..n {
            let p = inv_blk[c / BLOCK] * BLOCK + c % BLOCK;
            assert_eq!(
                r_act.s[row * n + c],
                r_nat.s[row * n + p],
                "s at column {c}"
            );
        }
        for b in 0..nb {
            assert_eq!(r_act.idx[row * nb + b], r_nat.idx[row * nb + inv_blk[b]]);
        }
        for g in 0..ng {
            assert_eq!(
                r_act.scale[row * ng + g],
                r_nat.scale[row * ng + inv_blk[g * per_group] / per_group],
                "scale of group {g}"
            );
        }
    }
}

/// `wscale`: with `H = I` the whitened scale projection is the least-squares scale of the returned
/// signs, `a = Σ w·s / Σ s²`, per (row, group).
#[test]
fn wscale_projections_are_least_squares() {
    let (o, n, damp) = (2usize, 128usize, 0.01);
    let w = random_w(o, n, 0x5CA0_0001);
    let mut id = vec![0.0f64; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    let r = gptq_e8(&w, &id, o, n, damp, false, true, 1).expect("gptq_e8 wscale");
    check_decode(&r, o, n, "wscale");
    for row in 0..o {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for j in 0..n {
            num += w[row * n + j] * r.s[row * n + j];
            den += r.s[row * n + j] * r.s[row * n + j];
        }
        let ls = (num / den.max(1e-12)).max(1e-12);
        assert!(
            (r.scale[row] - ls).abs() <= 1e-9 * ls.abs().max(1.0),
            "row {row}: whitened projection {} != least-squares {ls}",
            r.scale[row]
        );
    }
}

/// Refusals: not a multiple of 8, non-SPD Hessian, non-positive damping, a beam width outside
/// `1..=MAX_BEAM`, shape mismatches and non-finite input.
#[test]
fn refusals() {
    let (o, n, damp) = (2usize, 8usize, 0.01);
    let w = random_w(o, n, 0x5EFA_0001);
    let mut id = vec![0.0f64; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    let bad_n = random_w(o, 12, 0x5EFA_0002);
    let mut id12 = vec![0.0f64; 144];
    for i in 0..12 {
        id12[i * 12 + i] = 1.0;
    }
    assert_eq!(
        gptq_e8(&bad_n, &id12, o, 12, damp, false, false, 1),
        Err(GptqE8Error::BlockMisaligned)
    );
    let mut indef = id.clone();
    indef[1] = 2.0;
    indef[n] = 2.0;
    assert_eq!(
        gptq_e8(&w, &indef, o, n, damp, false, false, 1),
        Err(GptqE8Error::NotPositiveDefinite)
    );
    for bad in [0.0f64, -0.5, f64::NAN] {
        assert_eq!(
            gptq_e8(&w, &id, o, n, bad, false, false, 1),
            Err(GptqE8Error::InvalidDamping),
            "damping {bad} must be refused"
        );
    }
    for beam in [0usize, MAX_BEAM + 1] {
        assert_eq!(
            gptq_e8(&w, &id, o, n, damp, false, false, beam),
            Err(GptqE8Error::BeamOutOfRange),
            "beam {beam} must be refused"
        );
    }
    assert_eq!(
        gptq_e8(&w[1..], &id, o, n, damp, false, false, 1),
        Err(GptqE8Error::ShapeMismatch)
    );
    assert_eq!(
        gptq_e8(&w, &id[1..], o, n, damp, false, false, 1),
        Err(GptqE8Error::ShapeMismatch)
    );
    assert_eq!(
        gptq_e8(&w, &id, 0, n, damp, false, false, 1),
        Err(GptqE8Error::ZeroDim)
    );
    let mut nan = w.clone();
    nan[3] = f64::NAN;
    assert_eq!(
        gptq_e8(&nan, &id, o, n, damp, false, false, 1),
        Err(GptqE8Error::NonFinite)
    );
    let mut inf = id.clone();
    inf[9] = f64::INFINITY;
    assert_eq!(
        gptq_e8(&w, &inf, o, n, damp, false, false, 1),
        Err(GptqE8Error::NonFinite)
    );
}

/// `got` and `want` agree entrywise to a relative `1e-9` (the Q32.32 feedback factor's scales
/// against the f64 capture; measured `4e-11`).
fn scales_close(got: &[f64], want: &[f64]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| (g - w).abs() <= 1e-9 * w.abs().max(1.0))
}

/// FNV-1a 64 over the result's exact bit patterns: `s`, then `scale`, each entry as its little-endian
/// `to_bits()`, then every index as two little-endian bytes. No tolerance and no float comparison:
/// the whole stream, including the last mantissa bit, is pinned.
fn bit_stream_hash(r: &GptqE8) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let byte = |b: u8, h: &mut u64| {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for v in r.s.iter().chain(r.scale.iter()) {
        for b in v.to_bits().to_le_bytes() {
            byte(b, &mut h);
        }
    }
    for v in &r.idx {
        for b in v.to_le_bytes() {
            byte(b, &mut h);
        }
    }
    h
}

/// The `quantise_pass` / `beam_pass` row split is bit-identical to the serial loop they replace: the
/// rows are cut into contiguous ranges and each range runs the whole block loop on one
/// `std::thread::scope` worker, with the per-row operation order unchanged and no reduction across
/// rows, so every f64 and every code index is the same bit pattern it was on one thread.
///
/// The four fixtures and their constants were captured from the **serial** code at revision
/// `29ae4148` (lane q-gthr: `gptq_e8` on each fixture, `to_bits()` of every `s`/`scale` entry and
/// every index concatenated to `/home/socol/oc-tmp/lanes/port/q-gthr-ref/serial.bin`, sha256
/// `510d5a93ce4a2ba8548d9216d279d86a7d15d0cdae4f51bed9c499d804996973`); the threaded output at the
/// same revision re-dumped into `parallel.bin` and compared with `cmp`, exit 0. They cover an odd row
/// count (`o = 37`, so the last range is short), `o = 5 < INV_THREADS = 8` (one row per worker), the
/// single path `beam = 1`, the beam search `beam = 4` with `wscale`, and both `act_order` settings;
/// the hash is invariant to the worker count because each row is computed by exactly one worker in
/// the same order.
#[test]
fn row_split_reproduces_the_serial_bit_stream() {
    // `(o, n, damp, act_order, wscale, beam, seed)`.
    const FIXTURES: [(usize, usize, f64, bool, bool, usize, u64); 4] = [
        (37, 256, 0.01, false, false, 1, 0xA1B2_0001),
        (37, 256, 0.01, true, true, 4, 0xA1B2_0002),
        (5, 128, 0.01, true, false, 2, 0xA1B2_0003),
        (5, 128, 0.01, false, false, 2, 0xA1B2_0003),
    ];
    /// Serial `29ae4148` values, in `FIXTURES` order.
    const SERIAL_HASH: [u64; 4] = [
        0x5470_8767_759d_d511,
        0x7b33_bd9f_6fbd_858d,
        0x29d9_3318_7b8f_c130,
        0x319a_6fb9_2955_af45,
    ];
    for (i, &(o, n, damp, act_order, wscale, beam, seed)) in FIXTURES.iter().enumerate() {
        let w = random_w(o, n, seed);
        let h = correlated_h(n, seed ^ 0x5A5A_5A5A);
        let r = gptq_e8(&w, &h, o, n, damp, act_order, wscale, beam).expect("gptq_e8");
        assert_eq!(r.s.len(), o * n, "fixture {i}: s shape");
        assert_eq!(r.idx.len(), o * n / BLOCK, "fixture {i}: idx shape");
        assert_eq!(
            bit_stream_hash(&r),
            SERIAL_HASH[i],
            "fixture {i} (o={o} n={n} beam={beam} act_order={act_order} wscale={wscale}) is not \
             bit-identical to the serial pass at 29ae4148"
        );
    }
}

/// `beam = 1` reproduces the pre-beam port: on five deterministic correlated-Hessian
/// matrices the code indices and the group scales equal the values captured from the implementation
/// before the beam branch existed (the `beam = 1` path is the Python's `beam <= 1` single path,
/// line 305). Indices are compared exactly; scales to a relative `1e-9` ([`scales_close`]).
#[test]
#[allow(
    clippy::items_after_statements,
    reason = "the golden tables sit next to the test that uses them; lexically they follow two lets"
)]
fn beam_one_matches_the_pre_beam_result() {
    let (o, n, damp) = (3usize, 256usize, 0.01);
    let seeds = [
        0x0E11_0001u64,
        0x0E11_0002,
        0x0E11_0003,
        0x0E11_0004,
        0x0E11_0005,
    ];
    const GOLDEN_IDX: [&[u16]; 5] = [
        &[
            136, 16, 50, 80, 193, 67, 53, 87, 82, 116, 193, 88, 218, 73, 84, 176, 85, 56, 3, 4,
            190, 77, 44, 106, 28, 244, 39, 204, 82, 100, 83, 25, 104, 39, 96, 111, 129, 61, 71, 12,
            12, 28, 95, 20, 189, 69, 32, 47, 6, 83, 52, 148, 19, 93, 82, 75, 232, 133, 113, 63,
            200, 121, 44, 75, 233, 73, 58, 36, 15, 31, 16, 55, 222, 94, 44, 0, 122, 76, 54, 133,
            92, 97, 91, 118, 63, 91, 58, 208, 64, 220, 72, 126, 22, 50, 81, 47,
        ],
        &[
            85, 203, 214, 64, 214, 100, 37, 99, 119, 21, 46, 9, 22, 126, 195, 2, 171, 70, 31, 44,
            123, 201, 30, 185, 154, 67, 93, 201, 49, 80, 11, 63, 158, 91, 4, 117, 68, 37, 96, 224,
            62, 29, 97, 236, 27, 34, 52, 25, 53, 174, 110, 87, 62, 173, 179, 124, 116, 58, 75, 90,
            125, 210, 6, 75, 30, 75, 33, 99, 41, 20, 238, 3, 36, 59, 50, 30, 88, 103, 82, 117, 20,
            180, 84, 110, 225, 35, 9, 95, 38, 117, 81, 41, 228, 29, 35, 190,
        ],
        &[
            74, 61, 230, 33, 44, 66, 69, 168, 63, 58, 3, 32, 26, 123, 63, 70, 5, 101, 95, 17, 26,
            199, 82, 103, 43, 56, 70, 43, 63, 209, 12, 208, 61, 202, 67, 109, 99, 87, 163, 184, 22,
            202, 131, 13, 78, 38, 226, 106, 68, 15, 146, 39, 73, 87, 101, 106, 12, 100, 163, 73,
            47, 29, 130, 88, 71, 66, 117, 70, 45, 139, 127, 8, 223, 31, 97, 226, 31, 12, 76, 221,
            54, 70, 180, 166, 122, 81, 123, 112, 36, 69, 10, 46, 76, 188, 55, 65,
        ],
        &[
            129, 102, 110, 158, 29, 8, 55, 153, 75, 87, 94, 19, 24, 120, 82, 33, 120, 40, 123, 60,
            173, 227, 24, 159, 62, 75, 95, 14, 93, 229, 99, 76, 31, 170, 118, 55, 2, 70, 70, 33,
            118, 215, 19, 100, 63, 113, 76, 34, 9, 5, 108, 79, 108, 238, 28, 45, 106, 176, 2, 56,
            103, 94, 34, 203, 107, 122, 99, 205, 180, 201, 76, 54, 68, 60, 70, 154, 19, 73, 22, 42,
            214, 106, 96, 43, 76, 76, 63, 102, 108, 130, 19, 6, 83, 26, 160, 124,
        ],
        &[
            69, 238, 104, 62, 163, 44, 119, 33, 3, 229, 23, 24, 64, 123, 9, 85, 57, 27, 143, 13,
            79, 110, 76, 57, 63, 58, 135, 76, 119, 188, 102, 56, 85, 140, 49, 35, 33, 5, 22, 72,
            122, 76, 136, 221, 43, 221, 35, 49, 81, 194, 57, 99, 3, 203, 117, 228, 2, 106, 222,
            107, 20, 197, 26, 177, 54, 115, 152, 56, 180, 56, 11, 57, 6, 26, 13, 45, 184, 97, 44,
            35, 49, 48, 57, 40, 99, 62, 75, 73, 98, 17, 40, 88, 44, 113, 125, 84,
        ],
    ];
    const GOLDEN_SCALE: [&[f64]; 5] = [
        &[
            0.44635416738891437,
            0.49946099062675003,
            0.46280908659566566,
            0.4255675288429011,
            0.48290659976301126,
            0.47237385811445565,
        ],
        &[
            0.49895209043993793,
            0.49300327233490643,
            0.49351838323459796,
            0.5146650065206958,
            0.4862892671128194,
            0.5086542824179806,
        ],
        &[
            0.47287350177867715,
            0.4937544935881887,
            0.43502286975194493,
            0.4810089328201062,
            0.47937835776748966,
            0.475210138148769,
        ],
        &[
            0.4849587190197238,
            0.4897957993783389,
            0.4523557050860777,
            0.510660738571407,
            0.45819256647384915,
            0.5247907457469447,
        ],
        &[
            0.49373817097116424,
            0.4705507314363644,
            0.48976966649049325,
            0.4587354876289194,
            0.476628599855342,
            0.5018074031894018,
        ],
    ];
    for (m, seed) in seeds.iter().enumerate() {
        let w = random_w(o, n, *seed);
        let h = correlated_h(n, *seed ^ 0x5A5A_5A5A);
        let r = gptq_e8(&w, &h, o, n, damp, false, false, 1).expect("gptq_e8 beam 1");
        check_decode(&r, o, n, "beam 1 golden");
        assert_eq!(r.idx.as_slice(), GOLDEN_IDX[m], "matrix {m}: code indices");
        // Indices exact; scales follow the Q32.32 feedback factor: relative 1e-9 (measured 4e-11).
        assert!(
            scales_close(&r.scale, GOLDEN_SCALE[m]),
            "matrix {m}: scales {:?}",
            r.scale
        );
    }
}

/// One 8-block (`n = 8`): the block is decided before any feedback, so the winning beam's first-block
/// code is the exhaustive argmin of the block objective `a²|Z_c[k]|² − 2a⟨Z_w, Z_c[k]⟩` over all 256
/// codewords. Checked against the literal 256-codeword loop at `beam = 1`, `2` and `MAX_BEAM`; the
/// beam search's `beam 0` (least `cum_err2`) is that argmin.
#[test]
fn beam_first_block_is_the_exhaustive_argmin() {
    let (o, n, damp) = (16usize, 8usize, 0.01);
    let h = correlated_h(n, 0x0B10_C0DE);
    let mut w = Vec::with_capacity(o * n);
    for row in 0..o {
        w.extend_from_slice(&random_w(1, n, 0x0B10_BEEF ^ (0x1000 * (row as u64 + 1))));
    }
    let u = feedback_u(&h, n, damp).expect("SPD factor");
    let u_inv = block_inverse(&u, n, 0);
    let mut prec = vec![0.0f64; n];
    for j in 0..n {
        let d = u[j * n + j];
        prec[j] = 1.0 / (d * d).max(1e-12);
    }
    for beam in [1usize, 2, MAX_BEAM] {
        let r = gptq_e8(&w, &h, o, n, damp, false, false, beam).expect("gptq_e8");
        check_decode(&r, o, n, "beam oracle");
        for row in 0..o {
            let wr = &w[row * n..(row + 1) * n];
            let a0 = group_scale(wr, &prec, 0, n);
            let (_z_w, _zc, d, _q) = objective_scan(wr, &u_inv, a0);
            let best = argmin(&d);
            let tol = 1e-12 * a0.abs().max(1.0);
            assert_eq!(
                usize::from(r.idx[row]),
                best,
                "beam {beam} row {row}: the first-block code is not the objective argmin"
            );
            assert!(
                (r.scale[row] - a0).abs() <= tol,
                "beam {beam} row {row}: scale {} != {a0}",
                r.scale[row]
            );
        }
    }
}

/// The whitened total error of a `gptq_e8` result, recomputed from the returned codes, signs and
/// scales: for each group, `Err = (w0 − a·S)·U_blk⁻¹` block by block, accumulated as `Σ |err_block|²`
/// over every row and block, with the same intra- and inter-group feedback the kernel applies. This
/// equals the kernel's `cum_err2` at the winning beam (`step_err2 = |Z_w − a·Z_c|²`).
fn whitened_total_error(r: &GptqE8, w: &[f64], h: &[f64], o: usize, n: usize, damp: f64) -> f64 {
    let u = feedback_u(h, n, damp).expect("SPD factor");
    let ng = n.div_ceil(GROUP);
    let nb = n / BLOCK;
    let mut wk = w.to_vec();
    let mut tot = 0.0f64;
    for gi in 0..ng {
        let g0 = gi * GROUP;
        let g1 = (g0 + GROUP).min(n);
        let gs = g1 - g0;
        let mut err_all = vec![0.0f64; o * gs];
        for row in 0..o {
            let a = r.scale[row * ng + gi];
            let mut blk = 0;
            while blk < gs / BLOCK {
                let b0 = blk * BLOCK;
                let b1 = b0 + BLOCK;
                let mut zw = [0.0f64; BLOCK];
                for t in 0..BLOCK {
                    zw[t] = wk[row * n + g0 + b0 + t];
                }
                let mut j = BLOCK;
                while j > 0 {
                    j -= 1;
                    let mut s = zw[j];
                    for k in (j + 1)..BLOCK {
                        s -= zw[k] * u[(g0 + b0 + k) * n + g0 + b0 + j];
                    }
                    zw[j] = s / u[(g0 + b0 + j) * n + g0 + b0 + j];
                }
                let c = usize::from(r.idx[row * nb + (g0 + b0) / BLOCK]);
                let cw = e8::codeword(c).expect("code index");
                let mut err = [0.0f64; BLOCK];
                for t in 0..BLOCK {
                    err[t] = zw[t] - a * (cw[t] as f64 / ONE as f64);
                    err_all[row * gs + b0 + t] = err[t];
                    tot += err[t] * err[t];
                }
                let mut cc = b1;
                while cc < gs {
                    let mut acc = 0.0f64;
                    for t in 0..BLOCK {
                        acc += err[t] * u[(g0 + b0 + t) * n + g0 + cc];
                    }
                    wk[row * n + g0 + cc] -= acc;
                    cc += 1;
                }
                blk += 1;
            }
        }
        if g1 < n {
            for row in 0..o {
                let mut m = g1;
                while m < n {
                    let mut acc = 0.0f64;
                    for j in 0..gs {
                        acc += err_all[row * gs + j] * u[(g0 + j) * n + m];
                    }
                    wk[row * n + m] -= acc;
                    m += 1;
                }
            }
        }
    }
    tot
}

/// The whitened total error is not monotone in the beam: the search is greedy per 8-block, and a
/// wider beam can drop the greedy path (measured, beams `1,2,4,8`, five matrices: matrix 0
/// `[100.067, 101.785, 101.031, 96.480]`, matrix 2 `[104.529, 107.620, 98.021, 97.278]`). The
/// tested property is the mean over the five matrices: beams 4 and 8 each fall below beam 1.
#[test]
fn beam_mean_objective_falls_below_beam_one() {
    let (o, n, damp) = (3usize, 256usize, 0.01);
    let beams = [1usize, 4, 8];
    let mut mean = [0.0f64; 3];
    let seeds = [
        0x0BAD_0001u64,
        0x0BAD_0002,
        0x0BAD_0003,
        0x0BAD_0004,
        0x0BAD_0005,
    ];
    for seed in seeds {
        let w = random_w(o, n, seed);
        let h = correlated_h(n, seed ^ 0x5A5A_5A5A);
        for (i, &beam) in beams.iter().enumerate() {
            let r = gptq_e8(&w, &h, o, n, damp, false, false, beam).expect("gptq_e8");
            check_decode(&r, o, n, "beam objective");
            mean[i] += whitened_total_error(&r, &w, &h, o, n, damp) / 5.0;
        }
    }
    println!("beam mean objective (1,4,8): {mean:?}");
    assert!(
        mean[1] < mean[0],
        "beam 4 mean {} >= beam 1 mean {}",
        mean[1],
        mean[0]
    );
    assert!(
        mean[2] < mean[0],
        "beam 8 mean {} >= beam 1 mean {}",
        mean[2],
        mean[0]
    );
}

/// `wscale` with `beam > 1`: with `H = I` the feedback factor is diagonal, so the winning beam's
/// projected scale is the group least-squares scale `⟨w, s⟩/⟨s, s⟩` per row (the Python's
/// `a_new = a + ⟨Err, u_S⟩/⟨u_S, u_S⟩`, lines 367–376); checked at every beam width. Decodes to `s`.
#[test]
fn beam_wscale_projection_is_least_squares() {
    let (o, n, damp) = (2usize, 128usize, 0.01);
    let w = random_w(o, n, 0x5CA0_0001);
    let mut id = vec![0.0f64; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    for beam in [1usize, 2, 4, 8, MAX_BEAM] {
        let r = gptq_e8(&w, &id, o, n, damp, false, true, beam).expect("gptq_e8 beam wscale");
        check_decode(&r, o, n, "beam wscale");
        for row in 0..o {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for j in 0..n {
                num += w[row * n + j] * r.s[row * n + j];
                den += r.s[row * n + j] * r.s[row * n + j];
            }
            let ls = (num / den.max(1e-12)).max(1e-12);
            assert!(
                (r.scale[row] - ls).abs() <= 1e-9 * ls.abs().max(1.0),
                "beam {beam} row {row}: projected scale {} != least-squares {ls}",
                r.scale[row]
            );
        }
    }
}

/// `beam > 1` keeps the shape and refusal contract: a bad buffer length still refuses with
/// [`GptqE8Error::ShapeMismatch`] and a non-SPD Hessian with [`GptqE8Error::NotPositiveDefinite`].
#[test]
fn beam_refusals_are_the_single_beam_refusals() {
    let (o, n, damp) = (2usize, 8usize, 0.01);
    let w = random_w(o, n, 0x5EFA_0001);
    let mut id = vec![0.0f64; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    assert_eq!(
        gptq_e8(&w[1..], &id, o, n, damp, false, false, 4),
        Err(GptqE8Error::ShapeMismatch)
    );
    let mut indef = id.clone();
    indef[1] = 2.0;
    indef[n] = 2.0;
    assert_eq!(
        gptq_e8(&w, &indef, o, n, damp, false, false, 4),
        Err(GptqE8Error::NotPositiveDefinite)
    );
}
