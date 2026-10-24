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
        bt.batch(&g, &x, &a, &yt);
        let dev = bt.run(&g, false);
        let host = host_loss(s, &fs, &g2, &x, &a, &yt);
        println!("loss device {dev:.6} host {host:.6}");
        assert!(
            (dev - host).abs() <= 1e-4 * host.abs().max(1.0),
            "loss {dev} vs {host}"
        );

        let l0 = bt.run(&g, true);
        let gr = bt.grads(&g);
        let td = (s.t * s.d) as f64;
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
                    let fd = (host_loss(s, &p, &g2, &x, &a, &yt)
                        - host_loss(s, &q, &g2, &x, &a, &yt))
                        / (2.0 * f64::from(eps))
                        / td;
                    let got = f64::from(gr[k][j][idx]);
                    let err = (got - fd).abs() / fd.abs().max(1e-4);
                    worst = worst.max(err);
                    assert!(
                        err < 0.02,
                        "factor {k} param {j} [{idx}]: device {got} fd {fd}"
                    );
                }
            }
        }
        println!("scale gradient worst relative error {worst:.2e}");
        // The o-projection's input adjoint `da = dh·Wo` (`[t][a]`) against the f64 host
        // central differences of the block loss w.r.t. the o-projection input `a`, over the
        // weights the forward materialised from `fs`.
        let da = bt.get(&g, bt.da(), s.t * s.a);
        let mut da_worst = 0f64;
        for r in [0, s.t / 2, s.t - 1] {
            for c in [0, s.a / 2, s.a - 1] {
                let eps = 1e-3f32;
                let mut p = a.clone();
                let mut q = a.clone();
                p[r * s.a + c] += eps;
                q[r * s.a + c] -= eps;
                // The applied step, exact in f64: the f32 perturbed values differ from
                // `±eps` by their own rounding.
                let step = f64::from(p[r * s.a + c]) - f64::from(q[r * s.a + c]);
                let fd = (host_loss(s, &fs, &g2, &x, &p, &yt)
                    - host_loss(s, &fs, &g2, &x, &q, &yt))
                    / step
                    / td;
                let got = f64::from(da[r * s.a + c]);
                let err = (got - fd).abs() / fd.abs().max(1e-6);
                da_worst = da_worst.max(err);
                assert!(
                    err < 1e-4,
                    "da [{r},{c}]: device {got} fd {fd} relative {err:.2e}"
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
                    let mut p = w64.clone();
                    p[k][r * i + c] += eps;
                    let mut q = w64.clone();
                    q[k][r * i + c] -= eps;
                    let fd = (host_loss_w(s, &p, &g2, &x, &a, &yt)
                        - host_loss_w(s, &q, &g2, &x, &a, &yt))
                        / (2.0 * eps)
                        / td;
                    let got = f64::from(dw[r * i + c]);
                    let err = (got - fd).abs() / fd.abs().max(1e-4);
                    worst = worst.max(err);
                    assert!(err < 0.02, "dense slot {k} [{r},{c}]: device {got} fd {fd}");
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
