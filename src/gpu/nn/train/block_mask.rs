//! The block fit's two contract checks that are not a host-tape comparison, beside the committed
//! `block_map` SPIR-V: the **importance-weighted loss** the reference minimises
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `src/nanoquant/core/compress_block.py:17-19`,
//! `((pred.float()-tgt.float()).square()*importance).sum()`, divided by the batch size at `:88`) and
//! the **per-slot train mask** its block loop sets, one linear at a time
//! (`src/nanoquant/core/compress_model.py:83-101`: `tune_nonfact` over the still-dense linears,
//! ADMM, `tune_fact` over the factor just made).
//!
//! Evidence grade: `tested`. Not covered: the driver and the SPIR-V compiler.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the block's shapes index u32 dispatches; the host reference is f64"
)]

#[cfg(test)]
mod tests {
    use crate::gpu::decode::GpuCtx;
    use crate::gpu::nn::train::block::tests::{dense, factor, lin, unit};
    use crate::gpu::nn::train::block::{BlockShape, BlockTune, HostFactor, BLOCK_MAP_SPV};
    use crate::gpu::nn::AdamHyper;
    use crate::gpu::nn::GLSLC_ARGS;
    use crate::gpu::vulkan::Vk;
    use std::path::Path;

    /// The committed map is the one `glslc` compiles from the `.comp` beside it.
    #[test]
    fn block_map_spirv_is_the_glsl() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv/train");
        let out = std::env::temp_dir().join(format!("block_map_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(dir.join("block_map.comp"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let fresh = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        assert!(
            fresh == BLOCK_MAP_SPV,
            "block_map.spv is stale: re-run glslc"
        );
    }

    /// One block case: the shape, the four factors, the norm gain, the step's columns and a dense
    /// `W` for every slot.
    struct Case {
        s: BlockShape,
        fs: Vec<HostFactor>,
        dense_w: Vec<Vec<f32>>,
        g2: Vec<f32>,
        x: Vec<f32>,
        a: Vec<f32>,
        yt: Vec<f32>,
    }

    fn case() -> Case {
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
        let dense_w = (0..4)
            .map(|k| {
                let (o, i) = s.dims(k);
                unit(o * i, 10 * k as u64 + 5)
                    .iter()
                    .map(|&v| 0.3 * v)
                    .collect()
            })
            .collect();
        Case {
            s,
            fs,
            dense_w,
            g2: unit(s.d, 90).iter().map(|v| 1.0 + 0.2 * v).collect(),
            x: unit(s.t * s.d, 91),
            a: unit(s.t * s.a, 92),
            yt: unit(s.t * s.d, 93).iter().map(|v| 0.5 * v).collect(),
        }
    }

    /// The block output's residual `e = y − y_t` in f64, from the materialised weights
    /// `w = [o, gate, up, down]` (the forward of `block.rs`'s header).
    fn residual(c: &Case, w: &[Vec<f64>]) -> Vec<f64> {
        let BlockShape { t, d, a: aw, f, .. } = c.s;
        let g2 = &c.g2;
        let up = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>();
        let (x, a) = (up(&c.x), up(&c.a));
        let mut h = lin(&a, &w[0], t, aw, d);
        for (hv, xv) in h.iter_mut().zip(&x) {
            *hv += xv;
        }
        let mut n = vec![0f64; t * d];
        for r in 0..t {
            let ms: f64 = h[r * d..(r + 1) * d].iter().map(|v| v * v).sum::<f64>() / d as f64;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for col in 0..d {
                n[r * d + col] = h[r * d + col] * inv * f64::from(g2[col]);
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
            .zip(&c.yt)
            .map(|((yv, hv), tv)| yv + hv - f64::from(*tv))
            .collect()
    }

    /// A non-uniform importance enters both the loss and its gradient: the device loss equals
    /// `Σ_c D[c]·(Σ_r e[r,c]²)` in f64 and the device `dy` equals `scale·D[c]·e[r,c]`, at the
    /// reference's `scale = 2` (batch size 1).
    #[test]
    fn block_importance_loss_matches_host() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let c = case();
        let imp: Vec<f32> = unit(c.s.d, 97)
            .iter()
            .map(|v| 0.1 + 0.3 * v.abs())
            .collect();
        let w: Vec<Vec<f64>> = c.fs.iter().map(dense).collect();
        let e = residual(&c, &w);
        unsafe {
            let mut bt = BlockTune::new(&g, c.s, 1e-3, 1e-3, 1e-3);
            bt.load(&g, &c.fs, &c.g2);
            bt.load_importance(&g, &imp);
            bt.set_loss_scale(&g, 2.0);
            // DEVICE-SIDE CONTROL `KD_LIN_CTL=3`: scale the buffer `bt.run` reads (the factor's
            // `s_post`) by 2.0 after the load; `c.fs` stays unperturbed, so the derived `dy` bound
            // must go red.
            #[cfg(test)]
            if std::env::var("KD_LIN_CTL").as_deref() == Ok("3") {
                let mut fs2 = c.fs.clone();
                for v in fs2[0].s_post.iter_mut() {
                    *v *= 2.0;
                }
                bt.load(&g, &fs2, &c.g2);
            }
            bt.batch(&g, &c.x, &c.a, &c.yt);
            let dev = bt.run(&g, false);
            let host: f64 = (0..e.len())
                .map(|i| f64::from(imp[i % c.s.d]) * e[i] * e[i])
                .sum();
            println!("weighted loss device {dev:.6} host {host:.6}");
            assert!(
                (dev - host).abs() <= 1e-4 * host.abs().max(1.0),
                "loss {dev} vs {host}"
            );
            let _ = bt.run(&g, true);
            let dy = bt.dy(&g);
            // `dy = scale·imp ⊙ e`, and `e = down(m) + h − y_t` comes from four bf16 GEMMs. The
            // reference is the exact-f64 `e` (no difference); the bound is `delta·M` with
            // `delta = 4·(4u) + (ΣK + 2)·2^-24` — four products, each rounding both operands (4u),
            // then the f32 chains of their reductions — and `M` the per-entry magnitude sum the four
            // contractions carry, `Σ|a||W|` each ([`mags`]).
            let (t, d, aw, f) = (c.s.t, c.s.d, c.s.a, c.s.f);
            let cvt = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>();
            let (x64, a64) = (cvt(&c.x), cvt(&c.a));
            let m_o = mags(&a64, &w[0], t, aw, d);
            let n = {
                let mut hh = lin(&a64, &w[0], t, aw, d);
                for (hv, xv) in hh.iter_mut().zip(&x64) {
                    *hv += xv;
                }
                let mut nv = vec![0f64; t * d];
                for row in 0..t {
                    let ms: f64 = hh[row * d..(row + 1) * d]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        / d as f64;
                    let inv = 1.0 / (ms + 1e-6).sqrt();
                    for cc in 0..d {
                        nv[row * d + cc] = hh[row * d + cc] * inv * f64::from(c.g2[cc]);
                    }
                }
                nv
            };
            let m_g = mags(&n, &w[1], t, d, f);
            let m_u = mags(&n, &w[2], t, d, f);
            let m_in = {
                let gt = lin(&n, &w[1], t, d, f);
                let uu = lin(&n, &w[2], t, d, f);
                gt.iter()
                    .zip(&uu)
                    .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
                    .collect::<Vec<f64>>()
            };
            let m_d = mags(&m_in, &w[3], t, f, d);
            let delta = 4.0 * (4.0 * crate::gpu::nn::BF16_U)
                + ((aw + d + d + f) as f64 + 2.0) * 5.960_464_477_539_063e-8;
            let (mut num, mut den, mut bound) = (0f64, 1e-30f64, 0f64);
            for (i, &got) in dy.iter().enumerate() {
                let want = 2.0 * f64::from(imp[i % c.s.d]) * e[i];
                let m = m_o[i] + m_g[i % (t * f)] + m_u[i % (t * f)] + m_d[i];
                num = num.max((f64::from(got) - want).abs());
                bound = bound.max(2.0 * f64::from(imp[i % c.s.d]) * delta * m);
                den = den.max(want.abs());
            }
            println!(
                "weighted dy err {num:.3e} / bound {bound:.3e} = {:.3}",
                num / bound.max(1e-300)
            );
            assert!(
                num <= bound,
                "dy error {num:.3e} over the derived bound {bound:.3e}"
            );
        }
    }

    /// Training one slot moves that slot's parameters and moments and leaves the other three slots'
    /// parameters and moments bit-identical, for a dense slot and for a factor slot.
    #[test]
    fn block_train_mask_isolates_slot() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let c = case();
        unsafe {
            let mut bt = BlockTune::new(&g, c.s, 1e-3, 1e-3, 1e-3);
            bt.load(&g, &c.fs, &c.g2);
            for k in 0..4 {
                bt.load_dense(&g, k, &c.dense_w[k]);
            }
            bt.batch(&g, &c.x, &c.a, &c.yt);
            // Slot 1 dense (slot 0 a factor), then slot 0 a factor with the rest dense.
            for (k, is_factor) in [(1usize, false), (0usize, true)] {
                let mut stage = [true; 4];
                stage[k] = is_factor;
                let mut mask = [false; 4];
                mask[k] = true;
                bt.set_stage(&g, stage, mask);
                let before: Vec<Vec<u32>> = (0..4).map(|j| bt.slot_words(&g, j)).collect();
                for _ in 0..3 {
                    let _ = bt.run(&g, true);
                }
                let after: Vec<Vec<u32>> = (0..4).map(|j| bt.slot_words(&g, j)).collect();
                for j in 0..4 {
                    if j == k {
                        assert!(
                            before[j] != after[j],
                            "slot {k} (factor {is_factor}) did not move"
                        );
                    } else {
                        assert_eq!(
                            before[j], after[j],
                            "slot {j} moved while only slot {k} trained"
                        );
                    }
                }
                println!("mask: slot {k} (factor {is_factor}) moved, the other three hold");
            }
            // The optimiser stage the driver sets: the reference's constants and its schedule
            // through the same tape.
            bt.set_hyper(&g, AdamHyper::OPTIMI);
            bt.set_schedule(crate::gpu::nn::cosine_lr(1.0, 1e-4, 256, 3));
            bt.set_stage(&g, [true; 4], [true; 4]);
            let before = bt.slot_words(&g, 0);
            let loss = bt.run(&g, true);
            assert!(loss.is_finite(), "scheduled step loss {loss}");
            assert!(
                before != bt.slot_words(&g, 0),
                "the scheduled step moved nothing"
            );
            println!("optimi + cosine schedule: one step, loss {loss:.6}");
        }
    }
    /// `Σ_k|a||W|` a output entry: the magnitude sum the reduction of `y = a·Wᵀ` carries, the `M`
    /// of the `delta·M` bound at the entry's own operands.
    fn mags(a: &[f64], w: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
        let mut out = vec![0f64; n * o];
        for row in 0..n {
            for c in 0..o {
                out[row * o + c] = (0..i).map(|k| (a[row * i + k] * w[c * i + k]).abs()).sum();
            }
        }
        out
    }
}
