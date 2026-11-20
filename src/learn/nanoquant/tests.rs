use super::step::x_step;
use super::*;

/// The public error keeps the kernel's reason: a fit on a wide shape must report `Shape` (a
/// width past `MAX_INNER`) apart from `Range` (a scale placement), or the next run cannot tell a
/// missing bound from a data-dependent refusal.
#[test]
fn limb_refusals_keep_their_reason() {
    assert_eq!(
        NqError::from(LimbError::Shape),
        NqError::Kernel(LimbRefusal::Shape)
    );
    assert_eq!(
        NqError::from(LimbError::Range),
        NqError::Kernel(LimbRefusal::Range)
    );
    assert_eq!(
        NqError::from(LimbError::NoKernel),
        NqError::Kernel(LimbRefusal::NoKernel)
    );
}

/// The ADMM's loss on an exactly representable sign factor, against the reference's own loss on
/// the same input: `W = A·Bᵀ/√r` with `A`, `B` `±1` is *not* in the ADMM's model class. That class
/// is `(A_z·diag(1/c))·B_z` with `A_z`, `B_z` of `rank1_approx` shape `sign ⊙ outer(u, v)`
/// (`admm_nq.py:55-59`), whose magnitudes are separable per rank; a random sign matrix's are not.
/// Measured on the reference at `o=64, n=48, r=24`, 200 outer iters, `linear`, `reg=3e-2`:
/// `Loss 1.606 → 0.180`, served rel 0.440. `o=64, n=64`: 0.221/0.476. Ours on the same class:
/// `admm_loss 0.211`, so the test's band is the reference's, not a recovery claim.
#[test]
fn admm_loss_on_a_sign_factor_matches_the_reference_band() {
    let (o, n, r) = (64usize, 48usize, 24usize);
    let mut s = 12_345u64;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        if s >> 63 == 1 {
            1.0f64
        } else {
            -1.0
        }
    };
    let a: Vec<f64> = (0..o * r).map(|_| next()).collect();
    let b: Vec<f64> = (0..n * r).map(|_| next()).collect();
    let inv = (r as f64).sqrt().recip();
    let mut w = vec![0.0f64; o * n];
    for i in 0..o {
        for j in 0..n {
            w[i * n + j] = (0..r).map(|k| a[i * r + k] * b[j * r + k]).sum::<f64>() * inv;
        }
    }
    let wq = grid(&w).expect("grid");
    let cfg = NqConfig {
        outer_iters: 200,
        ..NqConfig::reference(r)
    };
    let f = factorize(&wq, o, n, &vec![1.0; n], &vec![1.0; o], &cfg).expect("factorize");
    assert!(
        f.admm_loss < 0.30,
        "admm_loss {} vs the reference's 0.180–0.221 band",
        f.admm_loss
    );
    let g = f.weight().expect("weight");
    let e: Vec<f64> = (0..o * n).map(|c| from_grid(g[c]) - w[c]).collect();
    let rel = (fro2(&e, o, n).expect("e") / fro2(&w, o, n).expect("w")).sqrt();
    assert!(rel < 0.50, "served rel {rel} vs the reference's 0.440");
}

/// The ADMM's B-step fits `X_B = A_z·diag(1/c_k)` (`admm_nq.py:172-177`), so the product it
/// minimises is `W_n = A_z·diag(1/c_k)·B_z`; the reference's `scale_factor = 1/A_z.norm(dim=0)`
/// (`:235`) puts that `1/c_k` into `A_final`, and `linear.py:250-253` serves the two-scale form
/// with no `scale_mid`. Publishing the norm through `s_mid` instead reconstructs `A_z·B_z`, the
/// fitted product times `diag(c_k)`.
///
/// `A_z` has the `rank1_approx` shape `sign ⊙ outer(u, v)` (`admm_nq.py:55-59`), so `s_post·U`
/// reproduces `A_final` exactly and `m = 1` serves the B-step product to rounding. The rank
/// columns' norms vary with `|v_k|`, which is what the old `s_mid` reweighted.
#[test]
fn export_serves_the_b_step_product_not_its_column_rescaling() {
    let (o, n, r) = (32usize, 24usize, 6usize);
    let mut vraw = [0.1f64, 0.2, 0.4, 0.7, 1.0, 1.4];
    let vnorm = vraw.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut vraw {
        *x /= vnorm;
    }
    let u: Vec<f64> = (0..o).map(|i| 1.5 + (i as f64 * 0.7).sin()).collect();
    let sign = |c: usize| if (c * 7 + 3) % 5 < 2 { -1.0 } else { 1.0 };
    let a: Vec<f64> = (0..o * r)
        .map(|c| sign(c) * u[c / r] * vraw[c % r])
        .collect();
    let b: Vec<f64> = (0..n * r).map(|c| sign(c + 11)).collect();
    let un = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    let c: Vec<f64> = (0..r).map(|k| vraw[k] * un).collect();
    let mut w = vec![0.0f64; o * n];
    for i in 0..o {
        for j in 0..n {
            w[i * n + j] = (0..r).map(|k| a[i * r + k] / c[k] * b[j * r + k]).sum();
        }
    }
    let w = grid(&w).expect("grid");
    let (no, ni) = (vec![1.0f64; o], vec![1.0f64; n]);
    let f = export(&[&a, &b, &a, &b], &ni, &no, o, n, r, 0.0).expect("export");
    let rel = |mid: &[f64]| {
        let hat = f.weight_with_mid(mid).expect("weight");
        let e: f64 = hat
            .iter()
            .zip(&w)
            .map(|(&p, &q)| (from_grid(p) - from_grid(q)).powi(2))
            .sum();
        let d: f64 = w.iter().map(|&q| from_grid(q).powi(2)).sum();
        (e / d).sqrt()
    };
    let cur = rel(&c);
    let one = rel(&f.s_mid);
    assert!(one < 1e-6, "s_mid = 1 serves the B-step product: rel {one}");
    assert!(
        cur > 0.1,
        "s_mid = column norm rescales every rank: rel {cur}"
    );
    assert!(
        f.s_mid.iter().all(|&x| x == 1.0),
        "the served form is two-scale"
    );
}

/// `fro2` keeps a matrix whose row norm leaves the kernel's word range: `export`'s `1/no`
/// row scale on 7B `ffn_down` makes one row of `a` 1.9e3 wide over 3424 columns, whose Q32.32
/// word is 5.6× `i64::MAX`; the exact `2^{-k}` rescale brings it back and the `f64` value is the
/// plain sum of squares.
#[test]
fn fro2_carries_a_row_past_the_word_range() {
    let (rows, cols) = (3usize, 3424usize);
    let m: Vec<f64> = (0..rows * cols)
        .map(|c| {
            let j = c % cols;
            1900.0 * (1.0 + (j % 7) as f64 * 1e-3)
        })
        .collect();
    let want: f64 = m.iter().map(|x| x * x).sum();
    let got = fro2(&m, rows, cols).expect("scaled norm");
    assert!(
        (got - want).abs() <= 1e-6 * want,
        "scaled {got} vs f64 {want}"
    );
}

/// `H·spd_inverse(H) = I` within `1e-5` (`f64` oracle) on a well-conditioned `H = AAᵀ + I` of
/// order 300, and no worse than the unrecursed base `UᵀU` on the same `H`.
#[test]
fn spd_inverse_inverts() {
    let n = 300usize;
    let mut rng = Rng(7);
    let scale = 1.0 / (n as f64).sqrt();
    let a = grid(&(0..n * n).map(|_| rng.normal() * scale).collect::<Vec<_>>()).unwrap();
    let aat = gemm_q32(&a, n, &a, n, n, LIMBS).unwrap();
    let h = grid_map(n * n, &|c| {
        from_grid(aat[c]) + if c / n == c % n { 1.0 } else { 0.0 }
    })
    .unwrap();
    // The oracle residual in `f64` (test-only scalar product, not on the build path).
    let hf: Vec<f64> = h.iter().map(|&w| from_grid(w)).collect();
    let mut res = Vec::new();
    for base in [n, INV_BASE] {
        let x = spd_inverse_at(&h, n, base, LIMBS).unwrap();
        let mut worst = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let d: f64 = (0..n)
                    .map(|t| hf[i * n + t] * from_grid(x[t * n + j]))
                    .sum();
                worst = worst.max((d - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        println!("INV residual base {base} n {n} max |H·X − I| = {worst:e}");
        res.push(worst);
    }
    assert!(
        res[1] < 1e-5 && res[1] <= res[0],
        "max |H·X − I|: recursion {:e}, none {:e}",
        res[1],
        res[0]
    );
}

/// The recursion's base case is a direct factor of the same block ([`base_inverse`]); the
/// assembled result must agree with a direct factor of the whole `h`. A transposed block read or
/// a wrong scale in the assembly satisfies the identity residual and still returns the wrong
/// inverse, so the residual is not the check.
#[test]
fn the_base_factor_agrees_with_the_recursion() {
    let n = 2 * INV_BASE + 8;
    let mut rng = Rng(11);
    let scale = 1.0 / (n as f64).sqrt();
    let a = grid(&(0..n * n).map(|_| rng.normal() * scale).collect::<Vec<_>>()).unwrap();
    let aat = gemm_q32(&a, n, &a, n, n, LIMBS).unwrap();
    let h = grid_map(n * n, &|c| {
        from_grid(aat[c]) + if c / n == c % n { 1.0 } else { 0.0 }
    })
    .unwrap();
    let r = spd_inverse(&h, n).expect("recursion");
    let d = base_inverse(&h, n, LIMBS).expect("direct");
    let worst = r
        .iter()
        .zip(&d)
        .map(|(&x, &y)| (from_grid(x) - from_grid(y)).abs())
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-5, "max |recursion − direct| = {worst:e}");
}

/// A rank `2¹⁶` quieter than the loudest keeps its digits in the `X`-update's Gram. `X = Z/d` is
/// invariant under scaling any rank of `Z`, so the step must return the same `Fᵀ` for the quiet
/// plane as for the scaled one; a step that normalises *after* the Gram floors the quiet rank's
/// diagonal at `EPS`, builds a `10¹²` matrix and refuses (`Solve(Overflow)`, measured on `l06
/// down` at `r = 3040`).
#[test]
fn x_step_keeps_a_quiet_rank_across_a_2_pow_16_spread() {
    let (r, n, k) = (6usize, 64usize, 5usize);
    let quiet = 1.0 / 65_536.0;
    let zx: Vec<f64> = (0..n)
        .flat_map(|j| {
            (0..r).map(move |i| {
                let s = if i + 1 == r { quiet } else { 1.0 };
                s * ((i + 1) as f64 * (j + 1) as f64).sin()
            })
        })
        .collect();
    let y: Vec<f64> = (0..k * n).map(|c| ((c % 9) as f64 - 4.0) / 8.0).collect();
    let wp = prepare(&grid(&y).expect("y on the grid"), k, n, LIMBS).expect("prepared");
    let z: Vec<f64> = (0..k * r).map(|c| ((c % 5) as f64 - 2.0) / 4.0).collect();
    let du: Vec<f64> = (0..k * r).map(|c| ((c % 3) as f64 - 1.0) / 8.0).collect();
    let out = x_step(&mut None, &zx, n, r, &wp, k, &z, &du, 0.82, 0.3).expect("no refusal");
    assert_eq!(out.len(), k * r);
    let loud: Vec<f64> = zx
        .iter()
        .enumerate()
        .map(|(c, &x)| if c % r + 1 == r { x * 65_536.0 } else { x })
        .collect();
    let same = x_step(&mut None, &loud, n, r, &wp, k, &z, &du, 0.82, 0.3).expect("no refusal");
    assert_eq!(out, same, "the rank's scale changed the normalised step");
}

/// [`Rng::normals`] equals sequential [`Rng::normal`] calls in every bit, in draw order and in the
/// transposed slot order `factorize` draws `Bᵀ` with, past the pool's serial cut, and leaves the
/// state where the sequential calls leave it.
#[test]
fn normals_match_sequential() {
    let (r, n) = (96usize, 300usize);
    let mut seq = Rng(5);
    let want: Vec<f64> = (0..2 * r * n).map(|_| seq.normal()).collect();
    let mut par = Rng(5);
    let a = par.normals(r * n, |c| c);
    let b = par.normals(r * n, |c| (c % r) * n + c / r);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a), bits(&want[..r * n]), "draw order");
    assert_eq!(
        bits(&b),
        bits(&transpose(&want[r * n..], r, n)),
        "transposed slots"
    );
    assert_eq!(par.0, seq.0, "state after");
}

/// The `[r0, r1) × [c0, c1)` block of a row-major matrix with `cols` columns.
fn block(m: &[i64], cols: usize, r0: usize, r1: usize, c0: usize, c1: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity((r1 - r0) * (c1 - c0));
    for i in r0..r1 {
        out.extend_from_slice(&m[i * cols + c0..i * cols + c1]);
    }
    out
}

/// The serial body of [`spd_inverse_at`] as committed at `7570d839`: the reference of the
/// pooled copies, symmetrisation and assembly.
fn spd_inverse_ref(h: &[i64], n: usize, base: usize) -> Result<Vec<i64>, NqError> {
    if n <= base {
        let u = inv_factor_upper(h, n)?;
        return Ok(syrk_q32(&transpose(&u, n, n), n, n, LIMBS)?);
    }
    let (p, q) = (n / 2, n - n / 2);
    let ai = spd_inverse_ref(&block(h, n, 0, p, 0, p), p, base)?;
    let bt = block(&transpose(h, n, n), n, p, n, 0, p);
    let x = gemm_q32(&ai, p, &bt, q, p, LIMBS)?;
    let btx = gemm_q32(&bt, q, &transpose(&x, p, q), q, p, LIMBS)?;
    let c = block(h, n, p, n, p, n);
    let mut s = vec![0i64; q * q];
    for i in 0..q {
        for j in 0..q {
            let m = (btx[i * q + j] >> 1) + (btx[j * q + i] >> 1);
            s[i * q + j] = c[i * q + j].checked_sub(m).ok_or(NqError::Range)?;
        }
    }
    let si = spd_inverse_ref(&s, q, base)?;
    let y = gemm_q32(&x, p, &si, q, q, LIMBS)?;
    let yx = gemm_q32(&y, p, &x, p, q, LIMBS)?;
    let mut out = vec![0i64; n * n];
    for i in 0..p {
        for j in 0..p {
            let m = (yx[i * p + j] >> 1) + (yx[j * p + i] >> 1);
            out[i * n + j] = ai[i * p + j].checked_add(m).ok_or(NqError::Range)?;
        }
        for j in 0..q {
            let v = 0i64.checked_sub(y[i * q + j]).ok_or(NqError::Range)?;
            out[i * n + p + j] = v;
            out[(p + j) * n + i] = v;
        }
    }
    for i in 0..q {
        out[(p + i) * n + p..(p + i) * n + n].copy_from_slice(&si[i * q..i * q + q]);
    }
    Ok(out)
}

/// [`spd_inverse_at`] equals the serial reference in every word at the production order 768,
/// an odd order and an asymmetric `h` (the reference reads `Bᵀ` by transposing `h`, the pooled
/// body reads it out of `h`), at bases 32 and 48; prints the wall of each.
#[test]
fn spd_inverse_matches_serial() {
    for &(n, base, skew) in &[
        (768usize, INV_BASE, false),
        (301, INV_BASE, false),
        (768, 48, false),
        (200, INV_BASE, true),
    ] {
        let mut rng = Rng(13 + n as u64);
        let scale = 1.0 / (n as f64).sqrt();
        let a = grid(&(0..n * n).map(|_| rng.normal() * scale).collect::<Vec<_>>()).unwrap();
        let aat = gemm_q32(&a, n, &a, n, n, LIMBS).unwrap();
        let h = grid_map(n * n, &|c| {
            let (i, j) = (c / n, c % n);
            let tilt = if skew && i < j { 1e-6 } else { 0.0 };
            from_grid(aat[c]) + if i == j { 1.0 } else { tilt }
        })
        .unwrap();
        let t = std::time::Instant::now();
        let want = spd_inverse_ref(&h, n, base).unwrap();
        let ref_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let got = spd_inverse_at(&h, n, base, LIMBS).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let diff = want.iter().zip(&got).filter(|(x, y)| x != y).count();
        println!(
            "INV serial n {n} base {base} skew {skew}: {diff} of {} words differ; ref {ref_ms:.2} ms, pooled {ms:.2} ms",
            n * n
        );
        assert_eq!(
            diff, 0,
            "spd_inverse_at differs from the serial body at n {n} base {base}"
        );
    }
}

/// [`spd_inverse`] under [`crate::engine::chart::pool::inline`] (every dispatch run on the calling
/// thread, the device inverse's host leaf) equals the pooled result word for word at the leaf order
/// 96 and at 301, and the guard restores pooled dispatch when dropped.
#[test]
fn spd_inverse_inline_matches_pooled() {
    for n in [96usize, 301] {
        let mut rng = Rng(29 + n as u64);
        let scale = 1.0 / (n as f64).sqrt();
        let a = grid(&(0..n * n).map(|_| rng.normal() * scale).collect::<Vec<_>>()).unwrap();
        let aat = gemm_q32(&a, n, &a, n, n, LIMBS).unwrap();
        let h = grid_map(n * n, &|c| {
            from_grid(aat[c]) + if c / n == c % n { 1.0 } else { 0.0 }
        })
        .unwrap();
        let pooled = spd_inverse(&h, n).unwrap();
        let inline = {
            let _inline = crate::engine::chart::pool::inline();
            spd_inverse(&h, n).unwrap()
        };
        let diff = pooled.iter().zip(&inline).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "inline spd_inverse differs at n {n}: {diff} words");
        assert_eq!(
            spd_inverse(&h, n).unwrap(),
            pooled,
            "pooled after the guard at n {n}"
        );
    }
}

/// Split of [`spd_inverse`] at order 768 into its kernel products and its bases (microbench;
/// prints, asserts nothing): every level's four products at its shapes and the 32 bases of
/// order 24, each timed over 5 runs.
#[test]
#[ignore = "microbench"]
fn spd_inverse_split_wall() {
    let n = 768usize;
    let mut rng = Rng(11);
    let scale = 1.0 / (n as f64).sqrt();
    let a = grid(&(0..n * n).map(|_| rng.normal() * scale).collect::<Vec<_>>()).unwrap();
    let aat = gemm_q32(&a, n, &a, n, n, LIMBS).unwrap();
    let h = grid_map(n * n, &|c| {
        from_grid(aat[c]) + if c / n == c % n { 1.0 } else { 0.0 }
    })
    .unwrap();
    let time = |f: &mut dyn FnMut()| {
        let mut ms = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            f();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.iter().sum::<f64>() / ms.len() as f64
    };
    let total = time(&mut || {
        std::hint::black_box(spd_inverse(&h, n).unwrap());
    });
    let mut prod = 0.0;
    let mut m = n / 2;
    let mut nodes = 1usize;
    while m >= INV_BASE / 2 && 2 * m > INV_BASE {
        let x = block(&h, n, 0, m, 0, m);
        let one = time(&mut || {
            std::hint::black_box(gemm_q32(&x, m, &x, m, m, LIMBS).unwrap());
        });
        println!("INV split level half-order {m} nodes {nodes} product {one:.3} ms x4");
        prod += 4.0 * nodes as f64 * one;
        m /= 2;
        nodes *= 2;
    }
    let leaf = 2 * m;
    let hb = block(&h, n, 0, leaf, 0, leaf);
    let base = time(&mut || {
        let u = inv_factor_upper(&hb, leaf).unwrap();
        std::hint::black_box(syrk_q32(&transpose(&u, leaf, leaf), leaf, leaf, LIMBS).unwrap());
    });
    println!(
        "INV split n {n} total {total:.2} ms products {prod:.2} ms bases {nodes} x {base:.3} = \
         {:.2} ms rest {:.2} ms",
        nodes as f64 * base,
        total - prod - nodes as f64 * base
    );
}

/// Wall of [`spd_inverse_at`] at the production order `r = 768` per base order (microbench;
/// prints, asserts nothing). `APEIRON_NQ_INV_BASES` is a comma list, default `16,32,48,96,192`.
#[test]
#[ignore = "microbench"]
fn spd_inverse_base_wall() {
    let n = 768usize;
    let mut rng = Rng(11);
    let scale = 1.0 / (n as f64).sqrt();
    let a = grid(&(0..n * n).map(|_| rng.normal() * scale).collect::<Vec<_>>()).unwrap();
    let aat = gemm_q32(&a, n, &a, n, n, LIMBS).unwrap();
    let h = grid_map(n * n, &|c| {
        from_grid(aat[c]) + if c / n == c % n { 1.0 } else { 0.0 }
    })
    .unwrap();
    let bases: Vec<usize> = std::env::var("APEIRON_NQ_INV_BASES")
        .unwrap_or_else(|_| String::from("16,32,48,96,192"))
        .split(',')
        .map(|s| s.parse().expect("base"))
        .collect();
    for &base in &bases {
        let mut ms = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let x = spd_inverse_at(&h, n, base, LIMBS).unwrap();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(x);
        }
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let (lo, hi) = ms
            .iter()
            .fold((f64::MAX, 0.0f64), |(l, h), &m| (l.min(m), h.max(m)));
        println!("INV base {base} n {n} mean {mean:.2} ms min {lo:.2} max {hi:.2} runs 5");
    }
}

/// Phase split of one [`x_update`] at the two side shapes of `down` (1024 × 3072), which
/// [`factorize`] transposes to `o = 3072, n = 1024` and runs at rank 864. Each timed block is the
/// call `x_update` makes, in the same order; the trailing `x_update` row is the whole function, so
/// the phase sum and the measured total can be compared in one table. The printed `threads` is
/// the pool's participant count, which is not `APEIRON_THREADS`: the pool caps at the pinnable
/// performance cores and adds the efficiency cores unless `APEIRON_ECORES=0`.
#[test]
#[ignore = "measurement: release run, three timed passes per side"]
fn x_update_phases() {
    use std::time::Instant;
    let r = 864usize;
    let names = [
        "zt grid_map",
        "syrk_q32",
        "d/hn/stab",
        "hw grid_map",
        "spd_inverse",
        "wp.gemm",
        "rt grid_map",
        "gemm_q32",
        "par_fill",
    ];
    for &(k, n) in &[(3072usize, 1024usize), (1024, 3072)] {
        let mut rng = Rng(0x5eed_0f00_1234_5678);
        let wf = rng.normals(k * n, |c| c);
        let wg = grid(&wf).expect("w on the grid");
        let wp = prepare(&wg, k, n, LIMBS).expect("prepared");
        let zx = rng.normals(n * r, |c| c);
        let z = rng.normals(k * r, |c| c);
        let du = rng.normals(k * r, |c| c);
        let (rho, reg) = (1.0f64, 1e-6f64);
        let mut best = [f64::MAX; 9];
        let mut total = f64::MAX;
        for _ in 0..3 {
            let mut ms = [0.0f64; 9];
            let t = Instant::now();
            let zt = grid_map(r * n, &|c| zx[(c % n) * r + c / n]).expect("zt");
            ms[0] = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let g = syrk_q32(&zt, r, n, LIMBS).expect("gram");
            ms[1] = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let d: Vec<f64> = (0..r)
                .map(|i| from_grid(g[i * r + i]).sqrt().max(EPS))
                .collect();
            let hn = |i: usize, j: usize| from_grid(g[i * r + j]) / (d[i] * d[j]);
            let mean_diag = (0..r).map(|i| hn(i, i)).sum::<f64>() / r as f64;
            let stab = (rho * mean_diag.abs() + reg).max(EPS);
            ms[2] = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let hw = grid_map(r * r, &|c| {
                let (i, j) = (c / r, c % r);
                if i == j {
                    hn(i, i) + stab
                } else {
                    hn(i, j).midpoint(hn(j, i))
                }
            })
            .expect("hw");
            ms[3] = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let hi = spd_inverse(&hw, r).expect("inverse");
            ms[4] = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let wz = wp.gemm(&zt, r).expect("wz");
            ms[5] = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let rt = grid_map(k * r, &|c| {
                from_grid(wz[c]) / d[c % r] + rho * (z[c] - du[c])
            })
            .expect("rt");
            ms[6] = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let f = gemm_q32(&rt, k, &hi, r, r, LIMBS).expect("f");
            ms[7] = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let mut out = vec![0.0f64; f.len()];
            par_fill(&mut out, &|c, o| *o = from_grid(f[c]));
            ms[8] = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&out);
            for (b, &m) in best.iter_mut().zip(&ms) {
                *b = b.min(m);
            }
            let t = Instant::now();
            let whole = x_update(&zx, n, r, &wp, k, &z, &du, rho, reg).expect("x_update");
            std::hint::black_box(&whole);
            total = total.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let sum: f64 = best.iter().sum();
        println!(
            "XU k {k} n {n} r {r} threads {} best of 3, ms",
            width_global()
        );
        for (name, &m) in names.iter().zip(&best) {
            println!("  {name:<14} {m:>9.2} {:>6.1}%", 100.0 * m / sum);
        }
        println!(
            "  {:<14} {sum:>9.2} {:>6.1}% of the whole",
            "phase sum",
            100.0 * sum / total
        );
        println!("  {:<14} {total:>9.2}", "x_update");
    }
}
