use super::*;
use crate::gpu::nn::train::attn_bwd::tests::{ref_bwd_ungated, ref_fwd_ungated};

/// A fixed-seed LCG value in `[-1, 1)` (Knuth's MMIX constants).
fn unit(n: usize, seed: u64) -> Vec<f32> {
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

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn factor(out: usize, inn: usize, rank: usize, seed: u64) -> HostFactor {
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

fn sg(v: f64) -> f64 {
    if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// `M = sign(U)·sign(V)ᵀ` in f64, `out × inn`.
fn signs(f: &HostFactor) -> Vec<f64> {
    let (o, i, r) = (f.out, f.inn, f.rank);
    let mut m = vec![0f64; o * i];
    for row in 0..o {
        for c in 0..i {
            m[row * i + c] = (0..r)
                .map(|j| sg(f64::from(f.u[row * r + j])) * sg(f64::from(f.vt[c * r + j])))
                .sum();
        }
    }
    m
}

/// `W = s_post ⊙ M ⊙ s_pre` in f64, `out × inn`.
fn dense(f: &HostFactor) -> Vec<f64> {
    let m = signs(f);
    (0..f.out * f.inn)
        .map(|x| f64::from(f.s_post[x / f.inn]) * m[x] * f64::from(f.s_pre[x % f.inn]))
        .collect()
}

/// `y = x·wᵀ`: `x` is `n × i`, `w` is `o × i`.
fn lin(x: &[f64], w: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
    let mut y = vec![0f64; n * o];
    for row in 0..n {
        for c in 0..o {
            y[row * o + c] = (0..i).map(|k| x[row * i + k] * w[c * i + k]).sum();
        }
    }
    y
}

/// `dw = dyᵀ·x`: `dy` is `n × o`, `x` is `n × i`, the result `o × i`.
fn dwt(dy: &[f64], x: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
    let mut dw = vec![0f64; o * i];
    for r in 0..o {
        for c in 0..i {
            dw[r * i + c] = (0..n).map(|k| dy[k * o + r] * x[k * i + c]).sum();
        }
    }
    dw
}

/// `n1 = rmsnorm(x, g)` over `d` at the device forward's epsilon.
fn rmsnorm(x: &[f64], g: &[f64], t: usize, d: usize) -> Vec<f64> {
    let mut n = vec![0f64; t * d];
    for row in 0..t {
        let ms: f64 = x[row * d..(row + 1) * d].iter().map(|v| v * v).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for c in 0..d {
            n[row * d + c] = x[row * d + c] * inv * g[c];
        }
    }
    n
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The f64 host case of one front: the shape, the per-head norm gains, the output cotangent
/// `R` and `n1` (which no factor moves).
struct Host {
    sh: FrontShape,
    gn: Vec<f64>,
    r: Vec<f64>,
    n1: Vec<f64>,
}

impl Host {
    /// The projections' outputs `(pq, pk, pv)` from the factors' dense `W`.
    fn proj(&self, fs: &[HostFactor]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (t, d) = (self.sh.t(), self.sh.d);
        let (qw, kw) = (self.sh.qw(), self.sh.kw());
        (
            lin(&self.n1, &dense(&fs[0]), t, d, qw),
            lin(&self.n1, &dense(&fs[1]), t, d, kw),
            lin(&self.n1, &dense(&fs[2]), t, d, kw),
        )
    }

    /// The forward alone, `ao`.
    fn ao(&self, fs: &[HostFactor]) -> Vec<f64> {
        let (pq, pk, pv) = self.proj(fs);
        ref_fwd_ungated(self.sh.attn(), &pq, &pk, &self.gn, &pv)
    }

    /// `L = Σ ao ⊙ R`.
    fn loss(&self, fs: &[HostFactor]) -> f64 {
        dot(&self.ao(fs), &self.r)
    }

    /// The three slots' dense `dW = dyᵀ·n1` in `[q, k, v]` order.
    fn dw(&self, fs: &[HostFactor]) -> [Vec<f64>; 3] {
        let (t, d) = (self.sh.t(), self.sh.d);
        let (qw, kw) = (self.sh.qw(), self.sh.kw());
        let (pq, pk, pv) = self.proj(fs);
        let (_, dqf, dkf, dv) = ref_bwd_ungated(self.sh.attn(), &pq, &pk, &self.gn, &pv, &self.r);
        [
            dwt(&dqf, &self.n1, t, d, qw),
            dwt(&dkf, &self.n1, t, d, kw),
            dwt(&dv, &self.n1, t, d, kw),
        ]
    }
}

/// The factor adjoint of `f` given its dense `dW`: `[u, vt, s_post, s_pre]`, the twin of
/// `slot::factor_bwd`'s five dispatches (`src/gpu/spirv/train/block_map.comp` modes 1, 2, 3).
fn factor_grads(f: &HostFactor, dw: &[f64]) -> [Vec<f64>; 4] {
    let (o, i, r) = (f.out, f.inn, f.rank);
    let m = signs(f);
    let mut dm = vec![0f64; o * i];
    let mut ds_post = vec![0f64; o];
    let mut ds_pre = vec![0f64; i];
    for row in 0..o {
        for c in 0..i {
            let x = row * i + c;
            let sp = f64::from(f.s_post[row]);
            let sq = f64::from(f.s_pre[c]);
            dm[x] = dw[x] * sp * sq;
            ds_post[row] += dw[x] * m[x] * sq;
            ds_pre[c] += dw[x] * m[x] * sp;
        }
    }
    let mut du = vec![0f64; o * r];
    let mut dvt = vec![0f64; i * r];
    for row in 0..o {
        for c in 0..i {
            for j in 0..r {
                du[row * r + j] += dm[row * i + c] * sg(f64::from(f.vt[c * r + j]));
                dvt[c * r + j] += dm[row * i + c] * sg(f64::from(f.u[row * r + j]));
            }
        }
    }
    [du, dvt, ds_post, ds_pre]
}

/// `max|got − want| / max|want|`.
fn rel32(got: &[f32], want: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 1e-30f64;
    for (&p, &q) in got.iter().zip(want) {
        num = num.max((f64::from(p) - q).abs());
        den = den.max(q.abs());
    }
    num / den
}

/// The attention front against the f64 host reference at a small shape: (a) the reference's
/// scale gradients against central differences of its own loss, (b) the device factor
/// gradients of `q`, `k`, `v` against the reference, (c) a short Adam run lowers
/// `L = Σ ao ⊙ R`.
#[test]
#[allow(clippy::too_many_lines, reason = "one tape comparison")]
fn front_tune_matches_host() {
    let Ok(vk) = (unsafe { Vk::load() }) else {
        println!("SKIP: no Vulkan device");
        return;
    };
    let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
    let sh = FrontShape {
        chunks: 1,
        n_ctx: 16,
        d: 32,
        nh: 4,
        nkv: 2,
        hd: 128,
        rot: 16,
        theta: 1e4,
        eps: 1e-6,
        ranks: [4; 3],
    };
    let fs: Vec<HostFactor> = (0..3)
        .map(|k| {
            let (o, i) = sh.dims(k);
            factor(o, i, sh.ranks[k], 10 * k as u64 + 1)
        })
        .collect();
    let (t, d) = (sh.t(), sh.d);
    let x = unit(t * d, 91);
    let g1: Vec<f32> = unit(d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
    let gn: Vec<f32> = unit(2 * sh.hd, 92).iter().map(|v| 1.0 + 0.2 * v).collect();
    let r: Vec<f32> = unit(t * sh.qw(), 93).iter().map(|v| 0.5 * v).collect();
    let host = Host {
        sh,
        gn: to64(&gn),
        r: to64(&r),
        n1: rmsnorm(&to64(&x), &to64(&g1), t, d),
    };

    // (a) the reference's scale gradients against central differences of its own loss.
    let analytic: Vec<[Vec<f64>; 4]> = {
        let dw = host.dw(&fs);
        (0..3).map(|k| factor_grads(&fs[k], &dw[k])).collect()
    };
    let mut worst = 0f64;
    for k in 0..3 {
        for j in [2usize, 3] {
            let len = if j == 2 { fs[k].out } else { fs[k].inn };
            for idx in [0, len / 2, len - 1] {
                let eps = 1e-4f32;
                let (mut p, mut q) = (fs.clone(), fs.clone());
                if j == 2 {
                    p[k].s_post[idx] += eps;
                    q[k].s_post[idx] -= eps;
                } else {
                    p[k].s_pre[idx] += eps;
                    q[k].s_pre[idx] -= eps;
                }
                // The perturbed entries are f32, so the exact step is their difference, not
                // `2·eps`: rounding `s + eps` to f32 is a 6e-5 relative error at these scales
                // and it would dominate the difference.
                let (hi, lo) = if j == 2 {
                    (p[k].s_post[idx], q[k].s_post[idx])
                } else {
                    (p[k].s_pre[idx], q[k].s_pre[idx])
                };
                let fd = (host.loss(&p) - host.loss(&q)) / (f64::from(hi) - f64::from(lo));
                let got = analytic[k][j][idx];
                let err = (got - fd).abs() / fd.abs().max(1e-6);
                worst = worst.max(err);
            }
        }
    }
    println!("fd worst relative error {worst:.2e}");
    assert!(worst < 1e-6, "fd worst relative error {worst:.2e}");

    unsafe {
        let mut ft = FrontTune::new(&g, sh, 1e-3, 2e-3);
        ft.load(&g, &fs, &g1, &gn);
        // DEVICE-SIDE CONTROL `KD_LIN_CTL=1` (turn 22): perturb the operand `ft.forward` and
        // `ft.backward` actually read. This test stages the FACTOR path (`set_stage(factor = all
        // true)` below), so the dense slot is not consumed — the control scales the factor's `u`
        // buffer, which is the buffer the materialise dispatch reads. The fixture is untouched,
        // so every reference stays at the unperturbed value and the stage bound must go red.
        #[cfg(test)]
        if std::env::var("KD_LIN_CTL").as_deref() == Ok("1") {
            let mut f2 = fs.clone();
            // The factor path consumes `sign(u)·sign(vt)` and the two SCALE vectors; `u`'s
            // magnitude never reaches a kernel. The operand is `s_post`, and the control scales it
            // by 2.0 — 1.5% moves the ratio only 0.104 → 0.129 (turn 22), a 2× operand cleanly
            // doubles the signal while leaving the default path untouched.
            for v in f2[0].s_post.iter_mut() {
                *v *= 2.0;
            }
            ft.load(&g, &f2, &g1, &gn);
        }
        ft.batch(&g, &x, &r);
        ft.forward(&g);
        println!("ao vs host {:.3e}", rel32(&ft.output(&g), &host.ao(&fs)));

        // (b) the device factor gradients of the three slots against the reference.
        ft.backward(&g, &r);
        let gr = ft.grads(&g);
        let dw_host = host.dw(&fs);
        // Step 1 of the lead's brief: where do the two sides stop agreeing? Per slot, the
        // device's dense `dW` against the reference's, then each factor gradient's cosine.
        for k in 0..3 {
            let gd = ft.dense_grad(&g, k);
            let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
            for i in 0..gd.len() {
                let (x, y) = (f64::from(gd[i]), dw_host[k][i]);
                ab += x * y;
                aa += x * x;
                bb += y * y;
            }
            println!(
                "COS dW {} : cos {:.6} (|dev| {:.3e} |host| {:.3e})",
                ["q", "k", "v"][k],
                ab / (aa * bb).sqrt().max(1e-300),
                aa.sqrt(),
                bb.sqrt()
            );
            for j in 0..4 {
                let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
                for i in 0..analytic[k][j].len() {
                    let (x, y) = (f64::from(gr[k][j][i]), analytic[k][j][i]);
                    ab += x * y;
                    aa += x * x;
                    bb += y * y;
                }
                println!(
                    "COS {} j {j}: cos {:.6} (|dev| {:.3e} |host| {:.3e})",
                    ["q", "k", "v"][k],
                    ab / (aa * bb).sqrt().max(1e-300),
                    aa.sqrt(),
                    bb.sqrt()
                );
            }
        }
        for k in 0..3 {
            let (out, inn) = sh.dims(k);
            let rank = fs[k].rank;
            let m = signs(&fs[k]);
            // `dm = dW ⊙ s_post·s_preᵀ`, which is what every factor gradient sums: `du` over
            // `inn` signed terms, `ds_post` over `inn`, `ds_pre` over `out`. `dW` itself is a
            // bf16 GEMM's output, so |dm err| <= (4u + t·2^-24)·Σ_t|dy·n1|, and the reduction that
            // follows is exact in ±1 up to the f32 chain. κ is each sum's cancellation, taken
            // from the reference's own `dm` at the entry that deviates most.
            let mut worst = 0f64;
            for j in 0..4 {
                let (mut num, mut den, mut at) = (0.0f64, 1e-30f64, 0usize);
                let len = analytic[k][j].len();
                for i in 0..len {
                    let e = (f64::from(gr[k][j][i]) - analytic[k][j][i]).abs();
                    if e > num {
                        num = e;
                        at = i;
                    }
                    den = den.max(analytic[k][j][i].abs());
                }
                let mut mag = 0.0f64;
                let mut span = inn;
                match j {
                    0 => {
                        let row = at / rank;
                        for c in 0..inn {
                            mag += (dw_host[k][row * inn + c]
                                * f64::from(fs[k].s_post[row])
                                * f64::from(fs[k].s_pre[c]))
                            .abs();
                        }
                    }
                    1 => {
                        let c = at / rank;
                        for row in 0..out {
                            mag += (dw_host[k][row * inn + c]
                                * f64::from(fs[k].s_post[row])
                                * f64::from(fs[k].s_pre[c]))
                            .abs();
                        }
                        span = out;
                    }
                    2 => {
                        for c in 0..inn {
                            mag += (dw_host[k][at * inn + c]
                                * m[at * inn + c]
                                * f64::from(fs[k].s_pre[c]))
                            .abs();
                        }
                    }
                    _ => {
                        for row in 0..out {
                            mag += (dw_host[k][row * inn + at]
                                * m[row * inn + at]
                                * f64::from(fs[k].s_post[row]))
                            .abs();
                        }
                        span = out;
                    }
                }
                // The lead's form: `delta·M`, with `M` the per-entry magnitude sum this
                // entry's reduction carries (already accumulated above) and
                // `delta = 4u + (span+2)·2^-24` — both GEMM operands rounded (`2u` each) plus the
                // f32 chain of `span` products and the two of the reduction. The reference is
                // the exact f64 analytic gradient, not a difference.
                let delta =
                    4.0 * crate::gpu::nn::BF16_U + (span as f64 + 2.0) * 5.960_464_477_539_063e-8;
                let bound = delta * mag / den;
                let w = num / den;
                // Correlation, so an unrelated pair of vectors cannot read as a tolerance miss.
                let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
                for i in 0..len {
                    let (x, y) = (f64::from(gr[k][j][i]), analytic[k][j][i]);
                    ab += x * y;
                    aa += x * x;
                    bb += y * y;
                }
                println!(
                    "qkv grad {} j {j}: {w:.2e} / bound {bound:.2e} = {:.3} (at {at}, M {mag:.3e}, \
                     cos {:.6})",
                    ["q", "k", "v"][k],
                    w / bound.max(1e-300),
                    ab / (aa * bb).sqrt().max(1e-300)
                );
                worst = worst.max(w / bound.max(1e-300));
            }
            assert!(
                worst <= 1.0,
                "slot {k}: worst ratio to its derived bound {worst:.3}"
            );
        }

        // (c) a short Adam run on the same loss.
        ft.forward(&g);
        let l0 = dot(&to64(&ft.output(&g)), &host.r);
        for _ in 0..20 {
            ft.train(&g);
        }
        ft.forward(&g);
        let l1 = dot(&to64(&ft.output(&g)), &host.r);
        println!("adam: loss {l0:.6} -> {l1:.6}");
        assert!(
            l1 < l0,
            "20 Adam steps did not lower the loss: {l0} -> {l1}"
        );
    }
}

/// Dense `q`, `k`, `v`: the device `ao` and the three `dW` equal the f64 host reference of the
/// same dense weights (the twin of `front_tune_matches_host`), and a train mask over one slot
/// moves that slot only.
#[test]
fn front_dense_matches_host() {
    let Ok(vk) = (unsafe { Vk::load() }) else {
        println!("SKIP: no Vulkan device");
        return;
    };
    let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
    let sh = FrontShape {
        chunks: 1,
        n_ctx: 16,
        d: 32,
        nh: 4,
        nkv: 2,
        hd: 128,
        rot: 16,
        theta: 1e4,
        eps: 1e-6,
        ranks: [4; 3],
    };
    let (t, d) = (sh.t(), sh.d);
    let (qw, kw) = (sh.qw(), sh.kw());
    let x = unit(t * d, 91);
    let g1: Vec<f32> = unit(d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
    let gn: Vec<f32> = unit(2 * sh.hd, 92).iter().map(|v| 1.0 + 0.2 * v).collect();
    let r: Vec<f32> = unit(t * qw, 93).iter().map(|v| 0.5 * v).collect();
    let ws: Vec<Vec<f32>> = [(qw, 4u64), (kw, 5), (kw, 6)]
        .iter()
        .map(|&(o, sd)| unit(o * d, sd).iter().map(|&v| 0.3 * v).collect())
        .collect();
    let host = Host {
        sh,
        gn: to64(&gn),
        r: to64(&r),
        n1: rmsnorm(&to64(&x), &to64(&g1), t, d),
    };
    let w64: Vec<Vec<f64>> = ws.iter().map(|w| to64(w)).collect();
    let pq = lin(&host.n1, &w64[0], t, d, qw);
    let pk = lin(&host.n1, &w64[1], t, d, kw);
    let pv = lin(&host.n1, &w64[2], t, d, kw);
    let ao = ref_fwd_ungated(sh.attn(), &pq, &pk, &host.gn, &pv);
    let (_, dqf, dkf, dv) = ref_bwd_ungated(sh.attn(), &pq, &pk, &host.gn, &pv, &host.r);
    let want = [
        dwt(&dqf, &host.n1, t, d, qw),
        dwt(&dkf, &host.n1, t, d, kw),
        dwt(&dv, &host.n1, t, d, kw),
    ];
    unsafe {
        let mut ft = FrontTune::new(&g, sh, 1e-3, 1e-3);
        // The norm gains come from `load`, whose factors a dense stage never reads.
        let fs: Vec<HostFactor> = (0..3)
            .map(|k| {
                let (o, i) = sh.dims(k);
                factor(o, i, sh.ranks[k], 10 * k as u64 + 1)
            })
            .collect();
        ft.load(&g, &fs, &g1, &gn);
        for k in 0..3 {
            ft.load_dense(&g, k, &ws[k]);
        }
        ft.set_stage(&g, [false; 3], [true; 3]);
        for k in 0..3 {
            let bits = |v: &[f32]| v.iter().map(|&x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!(bits(&ft.dense_w(&g, k)), bits(&ws[k]), "dense W {k} upload");
        }
        ft.batch(&g, &x, &r);
        ft.forward(&g);
        println!("dense ao vs host {:.3e}", rel32(&ft.output(&g), &ao));
        ft.backward(&g, &r);
        let mut worst = 0f64;
        for k in 0..3 {
            let (out, inn) = sh.dims(k);
            let got = ft.dense_grad(&g, k);
            // `dW = sum_t dy*n1`, `t` products an entry, and the face rounds both staged
            // operands to bf16 once: |dW err| <= kappa*(4u + t*2^-24)*|dW| with kappa the dot's
            // cancellation, taken from the reference's own f64 `dy` and `n1` at the entry that
            // deviates most. The metric is |err| / max|want|.
            let dy = [&dqf, &dkf, &dv][k];
            let (mut num, mut den, mut at) = (0.0f64, 1e-30f64, 0usize);
            for i in 0..out * inn {
                let e = (f64::from(got[i]) - want[k][i]).abs();
                if e > num {
                    num = e;
                    at = i;
                }
                den = den.max(want[k][i].abs());
            }
            let (row, col) = (at / inn, at % inn);
            let mut mag = 0.0f64;
            for q in 0..t {
                mag += (dy[q * out + row] * host.n1[q * d + col]).abs();
            }
            let kappa = mag / want[k][at].abs().max(1e-300);
            let u = crate::gpu::nn::BF16_U;
            let bound =
                (kappa * (4.0 * u + t as f64 * 5.960_464_477_539_063e-8) * want[k][at].abs()) / den;
            let w = num / den;
            println!(
                "dense dW {w:.2e} / bound {bound:.2e} = {:.3} ({} at {at}, kappa {kappa:.1})",
                w / bound.max(1e-300),
                ["q", "k", "v"][k]
            );
            worst = worst.max(w / bound.max(1e-300));
        }
        assert!(
            worst <= 1.0,
            "dense dW worst ratio to its derived bound {worst:.3}"
        );
        // The mask: training slot 1 leaves slots 0 and 2 bit-identical.
        ft.set_stage(&g, [false; 3], [false, true, false]);
        let before: Vec<Vec<u32>> = (0..3).map(|k| ft.slot_words(&g, k)).collect();
        for _ in 0..3 {
            ft.train(&g);
        }
        let after: Vec<Vec<u32>> = (0..3).map(|k| ft.slot_words(&g, k)).collect();
        assert!(before[1] != after[1], "slot 1 did not move");
        assert_eq!(before[0], after[0], "slot 0 moved");
        assert_eq!(before[2], after[2], "slot 2 moved");
        println!("front mask: slot 1 moved, slots 0 and 2 hold");
        // The optimiser stage: the reference's constants and its schedule through the tape.
        ft.set_hyper(&g, AdamHyper::OPTIMI);
        ft.set_schedule(crate::gpu::nn::cosine_lr(1.0, 1e-4, 256, 2));
        ft.set_stage(&g, [true; 3], [true; 3]);
        let before = ft.slot_words(&g, 0);
        ft.train(&g);
        assert!(
            before != ft.slot_words(&g, 0),
            "the scheduled step moved nothing"
        );
        println!("front optimi + cosine schedule: one step moved slot 0");
    }
}
