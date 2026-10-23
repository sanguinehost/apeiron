use super::*;

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
        let x = spd_inverse_at(&h, n, base).unwrap();
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
        let got = spd_inverse_at(&h, n, base).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let diff = want.iter().zip(&got).filter(|(x, y)| x != y).count();
        println!("INV serial n {n} base {base} skew {skew}: {diff} of {} words differ; ref {ref_ms:.2} ms, pooled {ms:.2} ms", n * n);
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
            let x = spd_inverse_at(&h, n, base).unwrap();
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
