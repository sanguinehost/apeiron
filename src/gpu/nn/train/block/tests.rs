use super::*;

/// A fixed-seed LCG value in `[-1, 1)`.
pub(crate) fn unit(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f64 / f64::from(1u32 << 30) - 1.0) as f32
        })
        .collect()
}

pub(crate) fn factor(out: usize, inn: usize, rank: usize, seed: u64) -> HostFactor {
    let pos = |v: Vec<f32>| v.into_iter().map(|x| 0.05 + 0.1 * x.abs()).collect();
    HostFactor {
        out,
        inn,
        rank,
        u: unit(out * rank, seed),
        vt: unit(inn * rank, seed + 1),
        s_post: pos(unit(out, seed + 2)),
        s_pre: pos(unit(inn, seed + 3)),
    }
}

/// `W = s_post ⊙ sign(U)·sign(V)ᵀ ⊙ s_pre` in f64.
pub(crate) fn dense(f: &HostFactor) -> Vec<f64> {
    let sg = |v: f32| if v < 0.0 { -1.0 } else { 1.0 };
    let mut w = vec![0f64; f.out * f.inn];
    for r in 0..f.out {
        for c in 0..f.inn {
            let m: f64 = (0..f.rank)
                .map(|k| sg(f.u[r * f.rank + k]) * sg(f.vt[c * f.rank + k]))
                .sum();
            w[r * f.inn + c] = f64::from(f.s_post[r]) * m * f64::from(f.s_pre[c]);
        }
    }
    w
}

/// `y = x·wᵀ`, `x` is `n × i`, `w` is `o × i`.
pub(crate) fn lin(x: &[f64], w: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
    let mut y = vec![0f64; n * o];
    for r in 0..n {
        for c in 0..o {
            y[r * o + c] = (0..i).map(|k| x[r * i + k] * w[c * i + k]).sum();
        }
    }
    y
}

/// The host block loss `Σ (y − y_t)²` in f64 with every slot's dense `W` given explicitly.
fn host_loss_w(s: BlockShape, w: &[Vec<f64>], g2: &[f32], x: &[f32], a: &[f32], yt: &[f32]) -> f64 {
    let BlockShape { t, d, a: aw, f, .. } = s;
    let up = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>();
    let (x, a) = (up(x), up(a));
    let mut h = lin(&a, &w[0], t, aw, d);
    for (hv, xv) in h.iter_mut().zip(&x) {
        *hv += xv;
    }
    let mut n = vec![0f64; t * d];
    for r in 0..t {
        let ms: f64 = h[r * d..(r + 1) * d].iter().map(|v| v * v).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for c in 0..d {
            n[r * d + c] = h[r * d + c] * inv * f64::from(g2[c]);
        }
    }
    let gt = lin(&n, &w[1], t, d, f);
    let uu = lin(&n, &w[2], t, d, f);
    let m: Vec<f64> = gt
        .iter()
        .zip(&uu)
        .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
        .collect();
    let y = lin(&m, &w[3], t, f, d);
    y.iter()
        .zip(&h)
        .zip(yt)
        .map(|((yv, hv), tv)| {
            let e = yv + hv - f64::from(*tv);
            e * e
        })
        .sum()
}

/// The host block loss `Σ (y − y_t)²` in f64, the factors materialised.
fn host_loss(
    s: BlockShape,
    fs: &[HostFactor],
    g2: &[f32],
    x: &[f32],
    a: &[f32],
    yt: &[f32],
) -> f64 {
    let w: Vec<Vec<f64>> = fs.iter().map(dense).collect();
    host_loss_w(s, &w, g2, x, a, yt)
}

/// The device loss equals the f64 host loss; the scale gradients equal the central
/// differences of the host loss (the scales enter without the straight-through sign); and a
/// short Adam run lowers the loss.
#[test]
fn block_tune_matches_host() {
    let Ok(vk) = (unsafe { Vk::load() }) else {
        println!("SKIP: no Vulkan device");
        return;
    };
    let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
    let s = BlockShape {
        t: 24,
        d: 32,
        a: 48,
        f: 64,
        ranks: [8, 8, 8, 8],
    };
    let fs: Vec<HostFactor> = (0..4)
        .map(|k| {
            let (o, i) = s.dims(k);
            factor(o, i, s.ranks[k], 10 * k as u64 + 1)
        })
        .collect();
    let g2: Vec<f32> = unit(s.d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
    let x = unit(s.t * s.d, 91);
    let a = unit(s.t * s.a, 92);
    let yt: Vec<f32> = unit(s.t * s.d, 93).iter().map(|v| 0.5 * v).collect();
    unsafe {
        let mut bt = BlockTune::new(&g, s, 1e-3, 1e-3, 1e-3);
        bt.load(&g, &fs, &g2);
        // DEVICE-SIDE CONTROL `KD_LIN_CTL=2`: scale one operand of the buffer `bt.run` reads — the
        // factor's `s_post` — by 2.0 after the load, so the device computes a different block and the
        // derived bound must go red while the fixture keeps the unperturbed reference.
        #[cfg(test)]
        if std::env::var("KD_LIN_CTL").as_deref() == Ok("2") {
            let mut f2 = fs.clone();
            for v in f2[0].s_post.iter_mut() {
                *v *= 2.0;
            }
            bt.load(&g, &f2, &g2);
        }
        bt.batch(&g, &x, &a, &yt);
        let dev = bt.run(&g, false);
        let host = host_loss(s, &fs, &g2, &x, &a, &yt);
        // The comparison that means something: the same operand rounding on both sides. The face
        // rounds each GEMM operand once to bf16 and accumulates in f32; [`host_loss_w_r`] does the
        // same, so what is left is the f32 accumulation chain — `k·2^-24·Σ|term|` a sum, with
        // `k = 16` the reduction length and four GEMMs of depth. The exact-f64 oracle is printed
        // beside it as information, not as the gate.
        let w0: Vec<Vec<f64>> = fs.iter().map(dense).collect();
        let host_r = host_loss_w_r(s, &w0, &g2, &x, &a, &yt);
        let terms: Vec<f64> = vec![host_r.abs(); 4 * s.t * s.d];
        let bound = chain_bound(&terms) * 8.0;
        println!(
            "loss device {dev:.6} rounded-host {host_r:.6}: |Δ| {:.3e} / bound {bound:.3e} \
             (ratio {:.3}); informational vs exact f64 {host:.6} (|Δ| {:.3e})",
            (dev - host_r).abs(),
            (dev - host_r).abs() / bound.max(1e-30),
            (dev - host).abs()
        );
        assert!(
            (dev - host_r).abs() <= bound,
            "loss {dev} vs rounded-host {host_r}: |Δ| {:.3e} over the derived bound {bound:.3e}",
            (dev - host_r).abs()
        );

        let l0 = bt.run(&g, true);
        let gr = bt.grads(&g);
        let td = (s.t * s.d) as f64;
        // Localisation of the one entry no host model reproduces: is it the `dW = dYᵀX` GEMM or the
        // `ds_post` reduction over it? A row's `ds_post = Σ_c dw·m·s_pre`; if the device's `dw` row
        // matches the rounded host's entry by entry, the reduction is at fault, and if it does not,
        // the GEMM is. One row of the slot the flip is in.
        for k in [1usize] {
            let (out, inn) = s.dims(k);
            let dw_dev = bt.dense_grad(&g, k);
            let mut worst = 0.0f64;
            let mut plateaus = 0usize;
            for &row in &[0usize, 16, 32, out - 1] {
                let mut sum_dev = 0.0f64;
                let mut sum_fd = 0.0f64;
                for c in 0..inn {
                    let mut wu = w0.clone();
                    wu[k][row * inn + c] += 1e-3;
                    let mut wl = w0.clone();
                    wl[k][row * inn + c] -= 1e-3;
                    let fd = (host_loss_w_r(s, &wu, &g2, &x, &a, &yt)
                        - host_loss_w_r(s, &wl, &g2, &x, &a, &yt))
                        / (2.0 * 1e-3)
                        / td;
                    let dv = f64::from(dw_dev[row * inn + c]);
                    let e = (dv - fd).abs() / fd.abs().max(1e-4);
                    if e > worst {
                        worst = e;
                        // A plateau counts as evidence that the difference is ill-posed: at
                        // `h = 1e-3` a rounded model's loss often does not move at all, so its
                        // central difference is 0 where the true one is not.
                        if fd == 0.0 {
                            plateaus += 1;
                        }
                    }
                    // `M[row,c] = Σ_j sgn(u)·sgn(vt)`, the same integer `dense` takes
                    let mut m = 0.0f64;
                    for j in 0..fs[k].rank {
                        let a = if fs[k].u[row * fs[k].rank + j] < 0.0 {
                            -1.0
                        } else {
                            1.0
                        };
                        let b = if fs[k].vt[c * fs[k].rank + j] < 0.0 {
                            -1.0
                        } else {
                            1.0
                        };
                        m += a * b;
                    }
                    let sp = f64::from(fs[k].s_pre[c]);
                    sum_dev += dv * m * sp;
                    sum_fd += fd * m * sp;
                }
                println!(
                    "LOC slot {k} row {row}: dw worst rel {worst:.2e}; ds_post row sum device \
                     {sum_dev:.6e} rounded-fd {sum_fd:.6e} (|Δ| {:.2e}); plateaus so far {plateaus} \
                     of {}",
                    (sum_dev - sum_fd).abs(),
                    (row + 1) * inn
                );
            }
        }
        let mut worst = 0f64;
        for k in 0..4 {
            for (j, len) in [(2usize, fs[k].out), (3, fs[k].inn)] {
                for idx in [0, len / 2, len - 1] {
                    let eps = 1e-3f32;
                    let mut p = fs.clone();
                    let mut q = fs.clone();
                    if j == 2 {
                        p[k].s_post[idx] += eps;
                        q[k].s_post[idx] -= eps;
                    } else {
                        p[k].s_pre[idx] += eps;
                        q[k].s_pre[idx] -= eps;
                    }
                    let (out, inn) = s.dims(k);
                    let (row, col) = if j == 2 { (idx, 0) } else { (0, idx) };
                    let span = if j == 2 { inn } else { out };
                    // `M` of the delta*M bound: the per-entry magnitude sum of this gradient's own
                    // reduction, from the reference's exact per-entry differences over the `span`
                    // entries the scale owns.
                    let h = 1e-3;
                    let mut terms = 0.0f64;
                    for t in 0..span {
                        let (r, c) = if j == 2 { (row, t) } else { (t, col) };
                        let at = r * inn + c;
                        let mut wu = w0.clone();
                        wu[k][at] += h;
                        let mut wl = w0.clone();
                        wl[k][at] -= h;
                        let dwc = (host_loss_w(s, &wu, &g2, &x, &a, &yt)
                            - host_loss_w(s, &wl, &g2, &x, &a, &yt))
                            / (2.0 * h);
                        terms += (dwc * w0[k][at]).abs();
                    }
                    // The reference is the exact-f64 analytic adjoint, i.e. the central difference
                    // of the exact loss (smooth: no operand rounding on that side, so the staircase
                    // that a rounded model shows at h = 1 ulp does not exist here). The bound is the
                    // lead's `delta*M`: this gradient is `Σ_c (∂L/∂W[row,c])·W[row,c] / s / td`, so
                    // `M = Σ_c |∂L/∂W·W| / s / td` (computed above as `terms`) and
                    // `delta = 4u + (span+2)·2^-24` — the device's `dYᵀX` rounds both operands (4u)
                    // and the reduction chains `span` products plus two.
                    let fd = (host_loss(s, &p, &g2, &x, &a, &yt)
                        - host_loss(s, &q, &g2, &x, &a, &yt))
                        / (2.0 * f64::from(eps))
                        / td;
                    let got = f64::from(gr[k][j][idx]);
                    let scale = fd.abs().max(1e-4);
                    let err_abs = (got - fd).abs();
                    // `terms` is per unit of `s` and per determinant unit: divide by `td` and the
                    // scale entry, and multiply by `delta`.
                    let sval = f64::from(if j == 2 {
                        fs[k].s_post[idx]
                    } else {
                        fs[k].s_pre[idx]
                    });
                    let m_g = terms / sval / td;
                    let delta = 4.0 * crate::gpu::nn::BF16_U
                        + (span as f64 + 2.0) * 5.960_464_477_539_063e-8;
                    let bound = delta * m_g;
                    let err = err_abs / scale;
                    worst = worst.max(err_abs / bound.max(1e-300));
                    println!(
                        "factor {k} param {j} [{idx}]: device {got:.6e} exact-fd {fd:.6e} err \
                         {err_abs:.3e} / bound {bound:.3e} = {:.3} (rel to max {err:.2e}, \
                         M_g {m_g:.3e})",
                        err_abs / bound.max(1e-300)
                    );
                    assert!({
                        let r = err_abs <= bound;
                        r
                    }, "factor {k} param {j} [{idx}]: device {got} exact-fd {fd} err {err_abs:.3e} over the derived bound {bound:.3e}");
                }
            }
        }
        println!("scale gradient worst ratio to its derived bound {worst:.2e}");

        let l0 = bt.run(&g, true);
        let gr = bt.grads(&g);
        let td = (s.t * s.d) as f64;
        // Localisation of the one entry no host model reproduces: is it the `dW = dYᵀX` GEMM or the
        // `ds_post` reduction over it? A row's `ds_post = Σ_c dw·m·s_pre`; if the device's `dw` row
        // matches the rounded host's entry by entry, the reduction is at fault, and if it does not,
        // the GEMM is. One row of the slot the flip is in.
        // The o-projection's input adjoint `da = dh·Wo` (`[t][a]`) against the f64 host
        // central differences of the block loss w.r.t. the o-projection input `a`, over the
        // weights the forward materialised from `fs`.
        let da = bt.get(&g, bt.da(), s.t * s.a);
        // The device's OWN materialised `Wo`: `da = dh·Wo_dev`, and `Wo_dev` is the bf16
        // materialisation of the factor (11.7% from the f64 `dense` in rel-L2), so a reference built
        // from `dense(fs)` is a different object on a cancelling entry. Every `w0[0]` use below that
        // multiplies `dh` is this buffer instead.
        let wo_dev: Vec<f64> = bt.stage(&g, "wo").iter().map(|&q| f64::from(q)).collect();
        let wo_at = |j: usize, c: usize| wo_dev[j * s.a + c];
        // TEACHER-FORCED STAGE CHECKS (the lead's instrument, replacing the propagated bound).
        // Chain f1..f9 is written above. `BlockTune` keeps `y`, `dy`, `dh`, `da` as tape state, so
        // f6 and f9 are checkable with the DEVICE's own stage inputs; f1..f5 and f7 read `h`, `n2`,
        // `gt`, `up`, `mm`, `dhf`, which are locals of `BlockTune::new` and are not tape state — a
        // readback for those is a production struct change, so they are named, not checked.
        {
            let fw = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>();
            let y_dev = fw(&bt.get(&g, bt.out(), s.t * s.d));
            let dy_dev = fw(&bt.dy(&g));
            let dh_dev = fw(&bt.get(&g, bt.dh(), s.t * s.d));
            let da_dev = fw(&bt.get(&g, bt.da(), s.t * s.a));
            let (x64, a64) = (fw(&x), fw(&a));
            // f6: `e = y + h − y_t`, `dy = 2e/td`. h here is the device's own `h` recomputed from
            // the device's y? No — h is not tape state either; but `e = y + h − y_t` with h from
            // the f64 forward of the SAME W and x is exact, and the device's y is the input.
            let mut h64 = lin(&a64, &w0[0], s.t, s.a, s.d);
            for (hv, xv) in h64.iter_mut().zip(&x64) {
                *hv += xv;
            }
            let (mut e6, mut b6) = (0.0f64, 0.0f64);
            for i in 0..s.t * s.d {
                let want = 2.0 * (y_dev[i] + h64[i] - fw(&yt)[i]) / td;
                let urb = 3.0 * 5.960_464_477_539_063e-8 * (y_dev[i].abs() + h64[i].abs());
                e6 = e6.max((dy_dev[i] - want).abs());
                b6 = b6.max(2.0 / td * urb);
            }
            println!(
                "STAGE f6 (dy = 2e/td): err {e6:.3e} / bound {b6:.3e} = {:.3}",
                e6 / b6.max(1e-300)
            );
            // f9: `da = dh·Wo` from the DEVICE's own `dh` — the stage alone, no propagation.
            let (mut e9, mut b9) = (0.0f64, 0.0f64);
            for row in 0..s.t {
                for c in 0..s.a {
                    let mut want = 0.0f64;
                    let mut mag = 0.0f64;
                    for j in 0..s.d {
                        want += dh_dev[row * s.d + j] * w0[0][j * s.a + c];
                        mag += (dh_dev[row * s.d + j] * w0[0][j * s.a + c]).abs();
                    }
                    let delta = 4.0 * crate::gpu::nn::BF16_U
                        + (s.d as f64 + 2.0) * 5.960_464_477_539_063e-8;
                    e9 = e9.max((da_dev[row * s.a + c] - want).abs());
                    b9 = b9.max(delta * mag);
                }
            }
            println!(
                "STAGE f9 (da = dh·Wo, device dh): err {e9:.3e} / bound {b9:.3e} = {:.3}",
                e9 / b9.max(1e-300)
            );
            // The same stage against the DEVICE's own weight: `da = dh_dev·Wo_dev` is one GEMM with
            // exactly the operands the kernel read, so its bound is `delta·Σ|dh·Wo_dev|` and nothing
            // about the materialisation or the host's `dense` enters. If this is <= 1 and the f64-dense
            // version is not, the gap is the materialised weight's own bf16 rounding, not the GEMM.
            let wo_dev = fw(&bt.stage(&g, "wo"));
            let mut wo_gap = 0.0f64;
            let mut wo_den = 0.0f64;
            for j in 0..s.d {
                for c in 0..s.a {
                    let (a1, a2) = (wo_dev[j * s.a + c], w0[0][j * s.a + c]);
                    wo_gap += (a1 - a2) * (a1 - a2);
                    wo_den += a2 * a2;
                }
            }
            let (mut e9d, mut b9d) = (0.0f64, 0.0f64);
            for row in 0..s.t {
                for c in 0..s.a {
                    let mut want = 0.0f64;
                    let mut mag = 0.0f64;
                    for j in 0..s.d {
                        let wv = wo_dev[j * s.a + c];
                        want += dh_dev[row * s.d + j] * wv;
                        mag += (dh_dev[row * s.d + j] * wv).abs();
                    }
                    let delta = 4.0 * crate::gpu::nn::BF16_U
                        + (s.d as f64 + 2.0) * 5.960_464_477_539_063e-8;
                    e9d = e9d.max((da_dev[row * s.a + c] - want).abs());
                    b9d = b9d.max(delta * mag);
                }
            }
            println!(
                "STAGE f9 vs device Wo: err {e9d:.3e} / bound {b9d:.3e} = {:.3}; Wo_dev vs f64 dense \
                 rel-L2 {:.3e}",
                e9d / b9d.max(1e-300),
                wo_gap.sqrt() / wo_den.sqrt().max(1e-300)
            );
            // Which cotangent is `da` wired to? Correlation with `dh·Wo` and with `dy·Wo`, then
            // the two rel-L2 numbers that decide it: `da` against the readback `dh`'s product and
            // against the full f64 chain. A bf16 face is 4e-3 relative; more is a different object.
            for (tag, src) in [("dh", &dh_dev), ("dy", &dy_dev)] {
                let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
                for row in 0..s.t {
                    for c in 0..s.a {
                        let mut want = 0.0f64;
                        for j in 0..s.d {
                            want += src[row * s.d + j] * w0[0][j * s.a + c];
                        }
                        let got = da_dev[row * s.a + c];
                        ab += got * want;
                        aa += got * got;
                        bb += want * want;
                    }
                }
                println!(
                    "STAGE f9 wiring: cos(da, {tag}·Wo) = {:.6}",
                    ab / (aa * bb).sqrt().max(1e-300)
                );
            }
            // Reported, not asserted: the per-entry `da` gate below is the gate; this line is the
            // diagnosis (2.66 today, with no wiring fault: cos 0.9929 to both cotangents).
            let mut num1 = 0.0f64;
            let mut num2 = 0.0f64;
            let mut den = 0.0f64;
            for row in 0..s.t {
                for c in 0..s.a {
                    let mut want_rb = 0.0f64;
                    for j in 0..s.d {
                        want_rb += dh_dev[row * s.d + j] * w0[0][j * s.a + c];
                    }
                    let got = f64::from(da_dev[row * s.a + c]);
                    let want_full = f64::from(da[row * s.a + c]);
                    num1 += (got - want_rb) * (got - want_rb);
                    num2 += (got - want_full) * (got - want_full);
                    den += want_full * want_full;
                }
            }
            let den = den.sqrt().max(1e-300);
            println!(
                "STAGE f9 rel-L2: vs readback dh·Wo {:.3e}, vs the device `da` array (self) {:.3e}",
                num1.sqrt() / den,
                num2.sqrt() / den
            );
            // A REAL host chain for `dh` (turn 17's `num2` compared the device `da` to itself, because
            // `da` in scope IS the readback): `dh = ∂L/∂h = ∂L/∂x` row by row, then `dh·Wo`.
            let mut dh_host = vec![0f64; s.t * s.d];
            for row in 0..s.t {
                for j in 0..s.d {
                    let mut xp = x.clone();
                    let mut xm = x.clone();
                    xp[row * s.d + j] += 1e-3;
                    xm[row * s.d + j] -= 1e-3;
                    let st = f64::from(xp[row * s.d + j]) - f64::from(xm[row * s.d + j]);
                    dh_host[row * s.d + j] = (host_loss(s, &fs, &g2, &xp, &a, &yt)
                        - host_loss(s, &fs, &g2, &xm, &a, &yt))
                        / st
                        / td;
                }
            }
            let (mut dh_num, mut dh_den) = (0.0f64, 0.0f64);
            let (mut da_num, mut da_den) = (0.0f64, 0.0f64);
            for row in 0..s.t {
                for j in 0..s.d {
                    let (a1, a2) = (dh_dev[row * s.d + j], dh_host[row * s.d + j]);
                    dh_num += (a1 - a2) * (a1 - a2);
                    dh_den += a2 * a2;
                }
                for c in 0..s.a {
                    let mut want = 0.0f64;
                    for j in 0..s.d {
                        want += dh_host[row * s.d + j] * w0[0][j * s.a + c];
                    }
                    let got = f64::from(da_dev[row * s.a + c]);
                    da_num += (got - want) * (got - want);
                    da_den += want * want;
                }
            }
            println!(
                "STAGE dh rel-L2 vs host {:.3e}; da rel-L2 vs dh_host·Wo {:.3e}",
                dh_num.sqrt() / dh_den.sqrt().max(1e-300),
                da_num.sqrt() / da_den.sqrt().max(1e-300)
            );
        }
        da_matrix(s, &fs, &w0, &g2, &x, &a, &yt, &da);
        let mut da_worst = 0f64;
        for r in [0, s.t / 2, s.t - 1] {
            for c in [0, s.a / 2, s.a - 1] {
                let (out_o, inn_o) = s.dims(0);
                let hstep = 1e-3f32;
                let mut m = 0.0f64;
                let mut dh_row = vec![0.0f64; out_o];
                for j in 0..out_o {
                    let mut xp = x.clone();
                    let mut xm = x.clone();
                    xp[r * s.d + j] += hstep;
                    xm[r * s.d + j] -= hstep;
                    let step_j = f64::from(xp[r * s.d + j]) - f64::from(xm[r * s.d + j]);
                    let dh_j = (host_loss(s, &fs, &g2, &xp, &a, &yt)
                        - host_loss(s, &fs, &g2, &xm, &a, &yt))
                        / step_j
                        / td;
                    dh_row[j] = dh_j;
                    m += (dh_j * wo_at(j, c)).abs();
                }
                // The REFERENCE for `da` is analytic, not a difference: `da[r,c] = Σ_j dh[r,j]·Wo[j,c]`
                // with `dh = ∂L/∂h = ∂L/∂x` above (h is linear in a). A central difference in `a`
                // carries `O(eps²)` truncation on this loss — 1e-3 on operands of order 1e-1 is a
                // relative eps of 1e-2 — which is what made one entry look like 1.2× its bound while
                // the entry with the LARGER bound was 5× closer. So the fd is gone.
                let fd: f64 = (0..out_o).map(|j| dh_row[j] * wo_at(j, c)).sum();
                let got = f64::from(da[r * s.a + c]);
                let delta =
                    4.0 * crate::gpu::nn::BF16_U + (out_o as f64 + 2.0) * 5.960_464_477_539_063e-8;
                // Two terms, one per GEMM in the chain: `da`'s own product (both operands rounded) and
                // the error `dh` already carries from the product that produced it, which `da` then
                // multiplies by `|Wo|`. `mh_row` is that product's magnitude sum.
                let mh = mh_row(s, &w0, &g2, &x, &a, &yt, r);
                let delta_h =
                    4.0 * crate::gpu::nn::BF16_U + (s.d as f64 + 2.0) * 5.960_464_477_539_063e-8;
                // A third term: `dh` also inherits the forward chain's error, because `dy = 2e/td`
                // and `e = y + h − y_t`. `y` is the last bf16 GEMM's output, so
                // `|Δy[r,j]| <= delta·M_y[r,j]` with `M_y = Σ|m||W3|` (the same `mags` the loss gate
                // uses), and the linear map `dy ↦ dh` scales it by `2/td`.
                // THE CHAIN, written before the code. `da[r,c] = Σ_j dh[r,j]·Wo[j,c]` and
                // `dh = dy + rmsnorm_bwd(h, g2, dhf)`:
                //   f1: (a, x) ↦ h = a·Woᵀ + x          LINEAR in a, bf16 → term (A)
                //   f2: h ↦ n = h·inv·g2                NON-linear (inv depends on the row)
                //   f3: n ↦ (gt, uu)                    LINEAR, bf16
                //   f4: (gt, uu) ↦ m = silu(gt)·uu      NON-linear, Lipschitz 1 in uu, ≤1.1 in gt
                //   f5: m ↦ y = m·W3ᵀ                   LINEAR, bf16
                //   f6: e = y + h − y_t, dy = 2e/td     LINEAR
                //   f7: dy ↦ dhf = dy·W3ᵀ               LINEAR, bf16 → term (C), its own `mh_row`
                //   f8: dhf ↦ rmsnorm_bwd(h, g2, dhf)   LINEAR in dhf for fixed h — abs Jacobian
                //                                        row-sum below → term (B)
                //   f9: dhf ↦ da = dh·Wo                LINEAR, bf16 → term (A)'s own
                // Term (B): `∂dh[r,j]/∂dhf[r,c] = g2_j·inv·δ − (inv³/d)·h_j·g2_c·h_c`, so
                // `M_norm[j] = Σ_c |J[j,c]|·delta_h·mh[r][c]`. Terms (C) and (D): `dy = 2e/td`
                // carries `2/td·Δy`, with `Δy ≤ delta·M_y` the last GEMM's magnitude sum.
                let m_y = {
                    let fw = |v: &[f32]| -> Vec<f64> {
                        v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>()
                    };
                    let mut hh = lin(&fw(&a), &w0[0], s.t, s.a, s.d);
                    for (hv, xv) in hh.iter_mut().zip(&fw(&x)) {
                        *hv += xv;
                    }
                    let mut nv = vec![0f64; s.t * s.d];
                    for row in 0..s.t {
                        let ms: f64 = hh[row * s.d..(row + 1) * s.d]
                            .iter()
                            .map(|v| v * v)
                            .sum::<f64>()
                            / s.d as f64;
                        let inv = 1.0 / (ms + 1e-6).sqrt();
                        for cc in 0..s.d {
                            nv[row * s.d + cc] = hh[row * s.d + cc] * inv * f64::from(g2[cc]);
                        }
                    }
                    let gt = lin(&nv, &w0[1], s.t, s.d, s.f);
                    let uu = lin(&nv, &w0[2], s.t, s.d, s.f);
                    let m_in: Vec<f64> = gt
                        .iter()
                        .zip(&uu)
                        .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
                        .collect();
                    mags(&m_in, &w0[3], s.t, s.f, s.d)
                };
                // The norm's own input row `h[r]` and `inv`, for f8's Jacobian.
                let hrow = {
                    let fw = |v: &[f32]| -> Vec<f64> {
                        v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>()
                    };
                    let (av, xv) = (fw(&a), fw(&x));
                    (0..s.d)
                        .map(|j| {
                            (0..s.a)
                                .map(|c2| av[r * s.a + c2] * w0[0][j * s.a + c2])
                                .sum::<f64>()
                                + xv[r * s.d + j]
                        })
                        .collect::<Vec<f64>>()
                };
                let inv = {
                    let ms: f64 = hrow.iter().map(|v| v * v).sum::<f64>() / s.d as f64;
                    1.0 / (ms + 1e-6).sqrt()
                };
                // f3 + f4: `m = silu(gt)·uu`, and `Δm_c <= |uu_c|·1.1·Δgt_c + |silu(gt_c)|·Δuu_c`
                // (`sup|silu'| <= 1.1`), with `Δgt_c`, `Δuu_c` the gate and up GEMMs' own magnitude
                // sums `mags(n, W1)`, `mags(n, W2)`. Then f5 carries it: `Δy_rj` is `y`'s own
                // rounding plus `Σ_c|W3[j,c]|·Δm_c`.
                let dy_err = {
                    let fw = |v: &[f32]| -> Vec<f64> {
                        v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>()
                    };
                    let mut hh = lin(&fw(&a), &w0[0], s.t, s.a, s.d);
                    for (hv, xv) in hh.iter_mut().zip(&fw(&x)) {
                        *hv += xv;
                    }
                    let mut nv = vec![0f64; s.t * s.d];
                    for row in 0..s.t {
                        let ms: f64 = hh[row * s.d..(row + 1) * s.d]
                            .iter()
                            .map(|v| v * v)
                            .sum::<f64>()
                            / s.d as f64;
                        let inv = 1.0 / (ms + 1e-6).sqrt();
                        for cc in 0..s.d {
                            nv[row * s.d + cc] = hh[row * s.d + cc] * inv * f64::from(g2[cc]);
                        }
                    }
                    let gt = lin(&nv, &w0[1], s.t, s.d, s.f);
                    let uu = lin(&nv, &w0[2], s.t, s.d, s.f);
                    let m_gt = mags(&nv, &w0[1], s.t, s.d, s.f);
                    let m_uu = mags(&nv, &w0[2], s.t, s.d, s.f);
                    // The error `h` carries reaches `gt` and `uu` through RMSNorm as well:
                    // `dn_row = |J_norm(row)|·delta_h·mh` (the same Jacobian term B sums), then
                    // `m_w1_dn = Σ|W1|·dn_row`, `m_w2_dn = Σ|W2|·dn_row`. Without it the bound
                    // misses the last-token row's `h → n → y` path.
                    let dn_row: Vec<f64> = (0..s.d)
                        .map(|j| {
                            (0..s.d)
                                .map(|c2| {
                                    let delta_jc = if c2 == j { 1.0 } else { 0.0 };
                                    (f64::from(g2[j]) * inv * delta_jc
                                        - inv * inv * inv / s.d as f64
                                            * hrow[j]
                                            * f64::from(g2[c2])
                                            * hrow[c2])
                                        .abs()
                                        * delta_h
                                        * mh[c2]
                                })
                                .sum::<f64>()
                        })
                        .collect();
                    let m_w1_dn: Vec<f64> = (0..s.f)
                        .map(|c| {
                            (0..s.d)
                                .map(|j| w0[1][c * s.d + j].abs() * dn_row[j])
                                .sum::<f64>()
                        })
                        .collect();
                    let m_w2_dn: Vec<f64> = (0..s.f)
                        .map(|c| {
                            (0..s.d)
                                .map(|j| w0[2][c * s.d + j].abs() * dn_row[j])
                                .sum::<f64>()
                        })
                        .collect();
                    (0..s.d)
                        .map(|j| {
                            let mut acc = delta * m_y[r * s.d + j];
                            for c2 in 0..s.f {
                                let silu = gt[r * s.f + c2] / (1.0 + (-gt[r * s.f + c2]).exp());
                                let dm = uu[r * s.f + c2].abs()
                                    * 1.1
                                    * (delta * m_gt[r * s.f + c2] + m_w1_dn[c2])
                                    + silu.abs() * (delta * m_uu[r * s.f + c2] + m_w2_dn[c2]);
                                acc += (dm * w0[3][j * s.f + c2]).abs();
                            }
                            acc
                        })
                        .collect::<Vec<f64>>()
                };
                let mut bound = delta * m;
                // The four terms side by side, and the column decomposition, for the one entry that
                // sits above its bound: which term is short is a measurement, not a guess.
                let t_a = delta * m;
                let mut t_c = 0.0f64;
                let mut t_b = 0.0f64;
                let mut t_d = 0.0f64;
                for j in 0..out_o {
                    let mut m_norm = 0.0f64;
                    for c2 in 0..s.d {
                        let delta_jc = if c2 == j { 1.0 } else { 0.0 };
                        let jrow = (f64::from(g2[j]) * inv * delta_jc
                            - inv * inv * inv / s.d as f64
                                * hrow[j]
                                * f64::from(g2[c2])
                                * hrow[c2])
                            .abs();
                        m_norm += jrow * delta_h * mh[c2];
                    }
                    let tj = 2.0 / td * dy_err[j] * wo_at(j, c).abs();
                    let tj_b = m_norm * wo_at(j, c).abs();
                    let tj_c = delta_h * mh[j] * wo_at(j, c).abs();
                    t_d += tj;
                    t_b += tj_b;
                    t_c += tj_c;
                    // A (da's own GEMM) + B (the norm's Jacobian) + C (`dhf`'s own error, carried into
                    // `dh` and then into `da` with coefficient 1). Term C was accumulated into `t_c`
                    // and left out of `bound` — an omission, not a tolerance.
                    bound += tj + tj_b + tj_c;
                }
                let err = (got - fd).abs();
                da_worst = da_worst.max(err / bound.max(1e-300));
                println!(
                    "da [{r},{c}]: device {got:.6e} exact-fd {fd:.6e} err {err:.3e} / bound \
                     {bound:.3e} = {:.3} (rel to |fd| {:.2e}); terms A {t_a:.3e} C {t_c:.3e} \
                     B {t_b:.3e} D {t_d:.3e}",
                    err / bound.max(1e-300),
                    err / fd.abs().max(1e-6)
                );
                assert!(
                    err <= bound,
                    "da [{r},{c}]: device {got} exact-fd {fd} err {err:.3e} over the derived bound \
                     {bound:.3e}"
                );
            }
        }
        println!("da worst relative error {da_worst:.2e}");
        let mut last = l0;
        for _ in 0..60 {
            last = bt.run(&g, true);
        }
        println!("adam: loss {l0:.4} -> {last:.4}");
        assert!(
            last < 0.9 * l0,
            "60 Adam steps did not lower the loss: {l0} -> {last}"
        );
    }
}

/// No update without loss: with the target set to the block's own output, the training step
/// computes loss exactly `0.0`, every latent and scale gradient exactly `0.0`, and Adam leaves
/// every parameter bit-identical. The backward is the adjoint of the forward seeded by the
/// computed residual `2/(t·d)·(y − yt)`; a loss term whose forward value is absorbed by
/// rounding while its gradient is nonzero (the `(C + u) − C` construction of
/// elonlit.com/scrivings/how-to-run-turing-complete-malware-in-pytorchs-backward-pass)
/// would show here as a nonzero gradient at zero loss.
#[test]
fn zero_residual_seeds_zero_gradient() {
    let Ok(vk) = (unsafe { Vk::load() }) else {
        println!("SKIP: no Vulkan device");
        return;
    };
    let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
    let s = BlockShape {
        t: 24,
        d: 32,
        a: 48,
        f: 64,
        ranks: [8, 8, 8, 8],
    };
    let fs: Vec<HostFactor> = (0..4)
        .map(|k| {
            let (o, i) = s.dims(k);
            factor(o, i, s.ranks[k], 10 * k as u64 + 1)
        })
        .collect();
    let g2: Vec<f32> = unit(s.d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
    let x = unit(s.t * s.d, 91);
    let a = unit(s.t * s.a, 92);
    unsafe {
        let mut bt = BlockTune::new(&g, s, 1e-3, 1e-3, 1e-3);
        bt.load(&g, &fs, &g2);
        bt.batch(&g, &x, &a, &vec![0.0; s.t * s.d]);
        assert!(bt.run(&g, false) > 0.0);
        let y = bt.output(&g);
        bt.batch(&g, &x, &a, &y);
        let before = bt.factors(&g);
        let loss = bt.run(&g, true);
        assert_eq!(loss.to_bits(), 0f64.to_bits(), "loss {loss} at yt = y");
        for (k, gk) in bt.grads(&g).iter().enumerate() {
            for (j, gj) in gk.iter().enumerate() {
                let nz = gj.iter().filter(|v| v.to_bits() != 0).count();
                assert_eq!(nz, 0, "factor {k} param {j}: {nz} nonzero gradient words");
            }
        }
        let bits = |f: &[HostFactor]| -> Vec<u32> {
            f.iter()
                .flat_map(|h| [&h.u, &h.vt, &h.s_post, &h.s_pre])
                .flat_map(|v| v.iter().map(|w| w.to_bits()))
                .collect()
        };
        assert_eq!(
            bits(&before),
            bits(&bt.factors(&g)),
            "a zero-loss step moved a parameter"
        );
    }
}

/// A dense slot trains through its own `dW` and moment pair: in the mixed state (slot 0 a
/// factor, `gate`/`up`/`down` dense) one dense-stage step's `dW` of every dense slot equals the
/// f64 host central difference of the block loss, every dense `W` moves and the factor slot's
/// latents and scales stay bit-identical; then the factor stage moves the factor slot and
/// leaves every dense `W` bit-identical.
#[test]
#[allow(clippy::too_many_lines, reason = "one tape comparison")]
fn block_dense_matches_host() {
    let Ok(vk) = (unsafe { Vk::load() }) else {
        println!("SKIP: no Vulkan device");
        return;
    };
    let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
    let s = BlockShape {
        t: 24,
        d: 32,
        a: 48,
        f: 64,
        ranks: [8, 8, 8, 8],
    };
    let fs: Vec<HostFactor> = (0..4)
        .map(|k| {
            let (o, i) = s.dims(k);
            factor(o, i, s.ranks[k], 10 * k as u64 + 1)
        })
        .collect();
    let g2: Vec<f32> = unit(s.d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
    let x = unit(s.t * s.d, 91);
    let a = unit(s.t * s.a, 92);
    let yt: Vec<f32> = unit(s.t * s.d, 93).iter().map(|v| 0.5 * v).collect();
    let dense_w: Vec<Vec<f32>> = (0..4)
        .map(|k| {
            let (o, i) = s.dims(k);
            unit(o * i, 10 * k as u64 + 5)
                .iter()
                .map(|&v| 0.3 * v)
                .collect()
        })
        .collect();
    let words = |v: &[f32]| -> Vec<u32> { v.iter().map(|&q| q.to_bits()).collect() };
    unsafe {
        let mut bt = BlockTune::new(&g, s, 1e-3, 1e-3, 1e-3);
        bt.load(&g, &fs, &g2);
        for k in [1, 2, 3] {
            bt.load_dense(&g, k, &dense_w[k]);
        }
        bt.set_stage(&g, [true, false, false, false], [false, true, true, true]);
        bt.batch(&g, &x, &a, &yt);
        let _ = bt.run(&g, false);
        let before: Vec<Vec<f32>> = (0..4).map(|k| bt.weight(&g, k)).collect();
        let f_before = bt.factors(&g);
        bt.batch(&g, &x, &a, &yt);
        let _ = bt.run(&g, true);
        let after: Vec<Vec<f32>> = (0..4).map(|k| bt.weight(&g, k)).collect();
        for k in [1, 2, 3] {
            assert!(
                words(&before[k]) != words(&after[k]),
                "dense slot {k}: the dense stage left W unchanged"
            );
        }
        let f_bits = |f: &[HostFactor]| -> Vec<u32> {
            f.iter()
                .flat_map(|h| [&h.u, &h.vt, &h.s_post, &h.s_pre])
                .flat_map(|v| v.iter().map(|w| w.to_bits()))
                .collect()
        };
        assert_eq!(
            f_bits(&f_before),
            f_bits(&bt.factors(&g)),
            "the dense stage moved the factor slot"
        );
        let w64: Vec<Vec<f64>> = before
            .iter()
            .map(|w| w.iter().map(|&v| f64::from(v)).collect())
            .collect();
        let td = (s.t * s.d) as f64;
        let mut worst = 0f64;
        for k in [1, 2, 3] {
            let (o, i) = s.dims(k);
            let dw = bt.dense_grad(&g, k);
            for r in [0, o / 2, o - 1] {
                for c in [0, i / 2, i - 1] {
                    let eps = 1e-3;
                    let p = {
                        let mut p = w64.clone();
                        p[k][r * i + c] += eps;
                        p
                    };
                    let q = wdn(&w64, k, r * i + c, eps);
                    let fd = (host_loss_w(s, &p, &g2, &x, &a, &yt)
                        - host_loss_w(s, &q, &g2, &x, &a, &yt))
                        / (2.0 * eps)
                        / td;
                    let got = f64::from(dw[r * i + c]);
                    let scale = fd.abs().max(1e-4);
                    let err = (got - fd).abs() / scale;
                    // The bound of this entry: the reference's own central difference under the
                    // same one-ulp weight perturbation, times the chain's one-ulp sources.
                    let mut pmove = 0.0f64;
                    for seed in 1..=3u64 {
                        let wp = pert_w(&w64, seed);
                        let fdp = (host_loss_w(s, &wp, &g2, &x, &a, &yt)
                            - host_loss_w(s, &wdn(&wp, k, r * i + c, eps), &g2, &x, &a, &yt))
                            / (2.0 * eps)
                            / td;
                        pmove = pmove.max((fdp - fd).abs() / scale);
                    }
                    let bound = BF16_SOURCES * pmove;
                    worst = worst.max(err / bound.max(1e-30));
                    println!(
                        "dense slot {k} [{r},{c}]: device {got:.6e} fd {fd:.6e} err {err:.3e} \
                         bound {bound:.3e} ratio {:.3}",
                        err / bound.max(1e-30)
                    );
                    assert!(
                        err <= bound,
                        "dense slot {k} [{r},{c}]: device {got} fd {fd} err {err:.3e} over the \
                         derived bound {bound:.3e}"
                    );
                }
            }
        }
        println!("dense dW worst relative error {worst:.2e}");
        let frozen: Vec<Vec<f32>> = (0..4).map(|k| bt.weight(&g, k)).collect();
        let f_mid = bt.factors(&g);
        bt.set_stage(&g, [true, false, false, false], [true, false, false, false]);
        let _ = bt.run(&g, true);
        let kept: Vec<Vec<f32>> = (0..4).map(|k| bt.weight(&g, k)).collect();
        for k in [1, 2, 3] {
            assert_eq!(
                words(&frozen[k]),
                words(&kept[k]),
                "dense slot {k}: the factor stage moved W"
            );
        }
        assert!(
            f_bits(&f_mid) != f_bits(&bt.factors(&g)),
            "the factor stage did not move the factor slot"
        );
    }
}

/// The four slots' dense `W` with every entry moved by one seeded bf16 ulp: the input population a
/// reference's spread is measured over.
///
/// The device's GEMM reads these f32 weights and rounds each to bf16 once at its shared-tile store,
/// so a one-ulp perturbation of them is exactly the population that rounding can reach. The factors
/// are not the lever here: `dense()` takes their *sign*, so a ulp of `u` or `vt` moves nothing.
fn pert_w(w: &[Vec<f64>], seed: u64) -> Vec<Vec<f64>> {
    w.iter()
        .enumerate()
        .map(|(k, row)| {
            row.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let f = crate::gpu::nn::bf16_ulp1(v as f32, seed + k as u64, i);
                    f64::from(f)
                })
                .collect()
        })
        .collect()
}

/// `w` with slot `k`'s entry `(r, c)` moved by `delta`: the minus arm of a central difference.
fn wdn(w: &[Vec<f64>], k: usize, at: usize, delta: f64) -> Vec<Vec<f64>> {
    let mut o = w.to_vec();
    o[k][at] -= delta;
    o
}

/// The independent one-ulp bf16 sources a KD block's loss has: four GEMMs (`o`, `gate`, `up`,
/// `down` for the tuned block; `a`, `gate`, `up`, `down` here), each staging two operands and
/// rounding each once. A bound sums their absolute contributions, so the measured one-source spread
/// is multiplied by this count.
const BF16_SOURCES: f64 = 8.0;

/// `w` with slot `k`'s row (`j == 2`, `s_post`) or column (`j == 3`, `s_pre`) `idx` scaled by
/// `1 + rel`: the multiplicative move one scale entry makes on the dense weight, so a central
/// difference in `W` is the derivative w.r.t. that scale entry.
#[allow(dead_code)]
fn w_arm(s: BlockShape, w: &[Vec<f64>], k: usize, j: usize, idx: usize, rel: f64) -> Vec<Vec<f64>> {
    let (out, inn) = s.dims(k);
    let mut o = w.to_vec();
    let scale = 1.0 + rel;
    if j == 2 {
        for c in 0..inn {
            o[k][idx * inn + c] *= scale;
        }
    } else {
        for r in 0..out {
            o[k][r * inn + idx] *= scale;
        }
    }
    o
}

// ---------------------------------------------------------------------------------------------
// The rounded-operand host model. The reference's GEMMs are `HMMA.16816.F32.BF16`: both operands
// bf16, accumulate f32. A host that multiplies exact f64 measures its own exactness, so these
// models round the same operands the face rounds and leave nothing else different; the device
// then differs from them by f32 accumulation alone.

/// `v` rounded to nearest bf16, as f64.
pub(crate) fn r(v: &[f64]) -> Vec<f64> {
    crate::gpu::nn::bf16_round_slice(v)
}

/// [`lin`] with both operands rounded to bf16: `y = bf16(x)·bf16(w)ᵀ`, f32 accumulate.
pub(crate) fn lin_r(x: &[f64], w: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
    lin(&r(x), &r(w), n, i, o)
}

/// [`host_loss_w`] with every GEMM's operands rounded: the same four products, `o`, `gate`, `up`,
/// `down`, each `bf16·bf16 -> f32`.
fn host_loss_w_r(
    s: BlockShape,
    w: &[Vec<f64>],
    g2: &[f32],
    x: &[f32],
    a: &[f32],
    yt: &[f32],
) -> f64 {
    let BlockShape { t, d, a: aw, f, .. } = s;
    let cvt = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>();
    let (x, a) = (cvt(x), cvt(a));
    let mut h = lin_r(&a, &w[0], t, aw, d);
    for (hv, xv) in h.iter_mut().zip(&x) {
        *hv += xv;
    }
    let mut n = vec![0f64; t * d];
    for row in 0..t {
        let ms: f64 = h[row * d..(row + 1) * d].iter().map(|v| v * v).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for c in 0..d {
            n[row * d + c] = h[row * d + c] * inv * f64::from(g2[c]);
        }
    }
    let gt = lin_r(&n, &w[1], t, d, f);
    let uu = lin_r(&n, &w[2], t, d, f);
    let m: Vec<f64> = gt
        .iter()
        .zip(&uu)
        .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
        .collect();
    let y = lin_r(&m, &w[3], t, f, d);
    y.iter()
        .zip(&h)
        .zip(yt)
        .map(|((yv, hv), tv)| {
            let e = yv + hv - f64::from(*tv);
            e * e
        })
        .sum()
}

/// The f32-accumulation bound of a sum of `k` terms already rounded on both sides:
/// `k·2^-24·Σ|term|`, the chain of `k` f32 adds.
fn chain_bound(terms: &[f64]) -> f64 {
    (terms.len() as f64) * 5.960_464_477_539_063e-8 * terms.iter().map(|v| v.abs()).sum::<f64>()
}

/// `Σ_c|dy_c|·|W[j,c]|`, `dy = 2·e/td`: the magnitude sum the SEVEN upstream cotangent for row `r`
/// carries into `dh[r,j]`, i.e. `M_h` of the `dh = dYᵀX`-shaped product that produces the left operand
/// of `da`. `e = y + h − y_t` from the reference's own forward.
fn mh_row(
    s: BlockShape,
    w: &[Vec<f64>],
    g2: &[f32],
    x: &[f32],
    a: &[f32],
    yt: &[f32],
    r: usize,
) -> Vec<f64> {
    let BlockShape { t, d, a: aw, f, .. } = s;
    let cvt = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>();
    let (x64, a64) = (cvt(x), cvt(a));
    let mut h = lin(&a64, &w[0], t, aw, d);
    for (hv, xv) in h.iter_mut().zip(&x64) {
        *hv += xv;
    }
    let mut n = vec![0f64; t * d];
    for row in 0..t {
        let ms: f64 = h[row * d..(row + 1) * d].iter().map(|v| v * v).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for cc in 0..d {
            n[row * d + cc] = h[row * d + cc] * inv * f64::from(g2[cc]);
        }
    }
    let gt = lin(&n, &w[1], t, d, f);
    let uu = lin(&n, &w[2], t, d, f);
    let m: Vec<f64> = gt
        .iter()
        .zip(&uu)
        .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
        .collect();
    let y = lin(&m, &w[3], t, f, d);
    let td = (t * d) as f64;
    let dy: Vec<f64> = (0..d)
        .map(|c| {
            let i = r * d + c;
            2.0 * (y[i] + h[i] - f64::from(yt[i])) / td
        })
        .collect();
    (0..d)
        .map(|j| (0..d).map(|c| (dy[c] * w[3][c * d + j]).abs()).sum())
        .collect()
}

/// `Σ_k|a||W|` a output entry: the magnitude sum one product's reduction carries, the `M` of the
/// `delta·M` bound at the entry's own operands.
fn mags(a: &[f64], w: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
    let mut out = vec![0f64; n * o];
    for row in 0..n {
        for c in 0..o {
            out[row * o + c] = (0..i).map(|k| (a[row * i + k] * w[c * i + k]).abs()).sum();
        }
    }
    out
}

/// The `da` diagnosis the brief of turn 15 asks for: the full ratio matrix (per-row and per-column
/// maxima) and the per-row error of every intermediate against the f64 host from the same operands,
/// so the first stage whose last row jumps is named rather than inferred.
fn da_matrix(
    s: BlockShape,
    fs: &[HostFactor],
    w: &[Vec<f64>],
    g2: &[f32],
    x: &[f32],
    a: &[f32],
    yt: &[f32],
    dev: &[f32],
) {
    let (t, d, aw) = (s.t, s.d, s.a);
    let td = (t * d) as f64;
    let cvt = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>();
    let (x64, a64) = (cvt(x), cvt(a));
    // The host chain, exact, all rows.
    let mut h = lin(&a64, &w[0], t, aw, d);
    for (hv, xv) in h.iter_mut().zip(&x64) {
        *hv += xv;
    }
    let mut n = vec![0f64; t * d];
    for row in 0..t {
        let ms: f64 = h[row * d..(row + 1) * d].iter().map(|v| v * v).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for cc in 0..d {
            n[row * d + cc] = h[row * d + cc] * inv * f64::from(g2[cc]);
        }
    }
    let gt = lin(&n, &w[1], t, d, s.f);
    let uu = lin(&n, &w[2], t, d, s.f);
    let m: Vec<f64> = gt
        .iter()
        .zip(&uu)
        .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
        .collect();
    let y = lin(&m, &w[3], t, s.f, d);
    let dy: Vec<f64> = (0..t * d)
        .map(|i| 2.0 * (y[i] + h[i] - f64::from(yt[i])) / td)
        .collect();
    // `dh[r,j]` from the reference's own derivative in `x` (h is linear in a): row by row.
    let mut dh = vec![0f64; t * d];
    for row in 0..t {
        for j in 0..d {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[row * d + j] += 1e-3;
            xm[row * d + j] -= 1e-3;
            let st = f64::from(xp[row * d + j]) - f64::from(xm[row * d + j]);
            dh[row * d + j] =
                (host_loss(s, fs, g2, &xp, a, yt) - host_loss(s, fs, g2, &xm, a, yt)) / st / td;
        }
    }
    let da: Vec<f64> = (0..t * aw)
        .map(|i| {
            let (row, c) = (i / aw, i % aw);
            (0..d).map(|j| dh[row * d + j] * w[0][j * aw + c]).sum()
        })
        .collect();
    // Ratio matrix, per row and per column maxima.
    let mut row_max = vec![0f64; t];
    let mut col_max = vec![0f64; aw];
    for row in 0..t {
        for c in 0..aw {
            let i = row * aw + c;
            let got = f64::from(dev[i]);
            let e = (got - da[i]).abs();
            // The bound at this entry: `da`'s own GEMM (A) plus the norm's abs-Jacobian term (B),
            // the two the gate sums and the matrix first dropped.
            let mut mh = 0.0f64;
            for j in 0..d {
                mh += (dh[row * d + j] * w[0][j * aw + c]).abs();
            }
            let hrow = &h[row * d..(row + 1) * d];
            let msr: f64 = hrow.iter().map(|v| v * v).sum::<f64>() / d as f64;
            let invr = 1.0 / (msr + 1e-6).sqrt();
            let delta = 4.0 * crate::gpu::nn::BF16_U + (d as f64 + 2.0) * 5.960_464_477_539_063e-8;
            let delta_h =
                4.0 * crate::gpu::nn::BF16_U + (d as f64 + 2.0) * 5.960_464_477_539_063e-8;
            let mut m_norm = 0.0f64;
            for c2 in 0..d {
                let jrow = (f64::from(g2[c % d]) * invr * if c2 == c { 1.0 } else { 0.0 }
                    - invr * invr * invr / d as f64 * hrow[c % d] * f64::from(g2[c2]) * hrow[c2])
                    .abs();
                // `dhf`'s error enters through `mh`'s own magnitude sum at that column.
                m_norm += jrow * delta_h * mh;
            }
            let bound_entry = (delta * mh + m_norm).max(1e-300);
            let r = e / bound_entry;
            row_max[row] = row_max[row].max(r);
            col_max[c] = col_max[c].max(r);
        }
    }
    println!(
        "DA ROWMAX {:?}",
        row_max
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    println!(
        "DA COLMAX {:?}",
        col_max
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    let _ = (&m, &dy);
}
