use super::*;
use crate::gpu::decode::{make_step_named, prof_arm, prof_disarm, prof_take, rec_step};
use crate::gpu::nn::train::attn_bwd::tests::ref_forward_w;
use crate::gpu::nn::train::attn_bwd::AttnBwdShape;
use crate::gpu::vulkan::*;

/// bf16 unit roundoff, `2^-8`: q, k, v and P are each rounded once.
const U: f64 = 1.0 / 256.0;

/// Fixed-seed LCG (Knuth's MMIX constants); `gauss` is the 12-uniform sum, unit variance.
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// The KD operand amplitudes: every element `sigma` times a unit Gaussian. `q` and `k` come out
/// of `attn_prep`, which RMS-normalises each head row and multiplies the learned gain, so their
/// element RMS is the gain's; `v` is the v-projection of the same rows. Measured on one real KD
/// layer by `kd_operand_magnitudes` below.
struct Amps {
    q: f64,
    k: f64,
    v: f64,
}

/// The measured KD amplitudes: `kd_operand_magnitudes` reads `blk.0` of the KD student
/// (`~/nq-ref/q3-0p6b-f16.gguf`) and reports |q| element RMS 2.273 (gain RMS
/// 2.273, max 11.81), |k| 9.784 (max 96.5), |v| 0.151 (row sigma max 0.221, so max|v| ~0.77
/// over the layer's 4.2e6 elements). The scores this implies (S ~ 22 sigma, so l ~ 1-3) are the
/// amplitudes the fixture below runs at.
const KD_AMPS: Amps = Amps {
    q: 2.273,
    k: 9.784,
    v: 0.151,
};

struct Case {
    label: &'static str,
    t: usize,
    nh: usize,
    nkv: usize,
    gated: bool,
    amps: &'static Amps,
}

/// A unit-element-RMS band (`|q|`, `|k|`, `|v|` each 1). At the measured KD band the scores run
/// `S ~ 22 sigma`, so a row's softmax is a single-key spike and the online rescale of O is
/// numerically inert (the control below measures that); this band keeps the multi-key sum and
/// the rescale live, and the face must hold its bound there too.
const UNIT_AMPS: Amps = Amps {
    q: 1.0,
    k: 1.0,
    v: 1.0,
};

static KD_CASE: Case = Case {
    label: "kd T2048 nh16 nkv8 hd128 ungated",
    t: 2048,
    nh: 16,
    nkv: 8,
    gated: false,
    amps: &KD_AMPS,
};

/// `src/gpu/spirv/train/attn_map.spv`: the attention prep the two train tapes record ahead of
/// this face.
static ATTN_MAP_SPV: &[u8] = include_bytes!("../../../spirv/train/attn_map.spv");

/// The forward arm of a `train::attn_bwd::AttnBwdShape` tape: `attn_map`'s prep, then this face
/// writing `out` and the row statistics `klse`.
/// The face is hd 128 and one causal sequence; a shape outside that gets the prep alone and no
/// attention output (`out` is left untouched), which is what a backward fixture needs — the
/// backward reads `q`, `gate`, `k`, `v` and `dy`, never the forward's output.
pub(crate) fn fwd_steps(
    g: &GpuCtx,
    s: AttnBwdShape,
    bufs: [Handle; 8],
    klse: Handle,
    gated: bool,
) -> Vec<Step> {
    let (nh, nkv, hd) = (s.nh, s.nkv, s.hd);
    let t = s.t();
    let flag = if gated { 0u32 } else { 4u32 };
    let meta = [
        flag,
        s.n_ctx as u32,
        nh as u32,
        nkv as u32,
        hd as u32,
        s.rot as u32,
        s.theta.to_bits(),
        s.eps.to_bits(),
        (t * (nh + nkv)) as u32,
    ];
    let mut all = vec![unsafe { g.up_u32(&meta) }];
    all.extend_from_slice(&bufs);
    let (gx, gy) = fold(t * (nh + nkv));
    let mut steps = vec![unsafe { make_step_named(g, ATTN_MAP_SPV, &all, gx, gy, "attn_prep") }];
    if hd as u32 == ATTN_KD_HD && s.chunks == 1 {
        steps.push(unsafe {
            attn_kd_dispatch(
                g,
                bufs[3],
                bufs[5],
                bufs[6],
                bufs[4],
                bufs[7],
                klse,
                nh as u32,
                nkv as u32,
                t as u32,
                1.0 / (hd as f32).sqrt(),
                gated,
            )
        });
    }
    steps
}

/// [`ref_forward`] at the engine kernel's KD face precision, from prep outputs already in f64:
/// `q`, `k`, `v` and each softmax weight rounded once to bf16. The device's operand roundings
/// are accounted for, so what remains is its f32 accumulation and an entry where f32 and f64
/// prep outputs round to different bf16 neighbours (one ulp, `u = 2^-8`).
pub(crate) fn ref_forward_bf(
    x: &crate::gpu::nn::train::attn_bwd::tests::RefIn,
    p: &crate::gpu::nn::train::attn_bwd::tests::Prep,
) -> Vec<f64> {
    let mut pb = p.clone();
    pb.q = pb.q.iter().copied().map(round_bf16_64).collect();
    pb.k = pb.k.iter().copied().map(round_bf16_64).collect();
    let mut xb = x.clone();
    xb.v = xb.v.iter().copied().map(round_bf16_64).collect();
    ref_forward_w(&xb, &pb, true)
}

/// `x` rounded to bf16 in f64: the top 8 mantissa bits, round-to-nearest-even — the device's
/// `F2FP.BF16.F32.PACK_AB`.
pub(crate) fn round_bf16_64(x: f64) -> f64 {
    let y = x as f32;
    let b = y.to_bits();
    if y.is_nan() || y.is_infinite() || (b & 0x7fff_ffff) < 0x0080_0000 {
        return f64::from(y);
    }
    let keep = b & 0xffff_0000;
    let rest = b & 0x0000_ffff;
    let up = rest > 0x0000_8000 || (rest == 0x0000_8000 && (keep & 0x0001_0000) != 0);
    f64::from(f32::from_bits(if up { keep + 0x0001_0000 } else { keep }))
}

fn gen(rng: &mut Lcg, n: usize, s: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.gauss() * s) as f32).collect()
}

/// `x` rounded to bf16: the top 8 mantissa bits, round-to-nearest-even, the device's own
/// `F2FP.BF16.F32.PACK_AB` rounding.
fn round_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() || x.is_infinite() || (b & 0x7fff_ffff) < 0x0080_0000 {
        return x;
    }
    // the discarded low 16 bits, with the implicit bit's parity as the tie-breaker
    let keep = b & 0xffff_0000;
    let rest = b & 0x0000_ffff;
    let half = 0x0000_8000u32;
    let up = rest > half || (rest == half && (keep & 0x0001_0000) != 0);
    f32::from_bits(if up { keep + 0x0001_0000 } else { keep })
}

/// The f64 oracle's words: the output `[T][NH*HD]`, the logsumexp `[NH][T]`, the per-row score
/// scale `S_i = max_j scale·Σ_e |q_ie||k_je|`, and the largest row denominator `l`.
struct Ref {
    out: Vec<f64>,
    lse: Vec<f64>,
    s: Vec<f64>,
    l: f64,
}

/// The f64 oracle: causal softmax attention over the whole sequence, one (head, row) at a time,
/// `O(T)` memory per row. Run twice by the gates — once on the f32 operands (the port's total
/// error against exact arithmetic, bounded by the operand roundings) and once on the operands
/// rounded to bf16 (the kernel's own arithmetic, bounded by P's one rounding).
fn oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: Option<&[f32]>,
    c: &Case,
    hd: usize,
    scale: f64,
) -> Ref {
    let (t, nh, nkv) = (c.t, c.nh, c.nkv);
    let qd: Vec<f64> = q.iter().map(|&x| f64::from(x)).collect();
    let kd: Vec<f64> = k.iter().map(|&x| f64::from(x)).collect();
    let vd: Vec<f64> = v.iter().map(|&x| f64::from(x)).collect();
    let mut out = vec![0.0f64; t * nh * hd];
    let mut lse = vec![0.0f64; nh * t];
    let mut smax = vec![0.0f64; nh * t];
    let mut lmax = 1.0f64;
    let mut p = vec![0.0f64; t];
    let mut acc = vec![0.0f64; hd];
    for h in 0..nh {
        let kvh = h / (nh / nkv);
        for i in 0..t {
            let qb = (i * nh + h) * hd;
            let mut m = f64::NEG_INFINITY;
            let mut sabs = 0.0f64;
            for j in 0..=i {
                let kb = (j * nkv + kvh) * hd;
                let mut s = 0.0f64;
                let mut a = 0.0f64;
                for e in 0..hd {
                    s += qd[qb + e] * kd[kb + e];
                    a += qd[qb + e].abs() * kd[kb + e].abs();
                }
                let s = s * scale;
                p[j] = s;
                if s > m {
                    m = s;
                }
                if a > sabs {
                    sabs = a;
                }
            }
            smax[h * t + i] = sabs * scale;
            let mut l = 0.0f64;
            for x in p.iter_mut().take(i + 1) {
                let e = (*x - m).exp();
                *x = e;
                l += e;
            }
            for a in acc.iter_mut() {
                *a = 0.0;
            }
            for j in 0..=i {
                let pj = p[j];
                let vb = (j * nkv + kvh) * hd;
                for d in 0..hd {
                    acc[d] += pj * vd[vb + d];
                }
            }
            for d in 0..hd {
                let mut y = acc[d] / l;
                if let Some(g) = gate {
                    let gv = f64::from(g[qb + d]);
                    y *= 1.0 / (1.0 + (-gv).exp());
                }
                out[qb + d] = y;
            }
            lse[h * t + i] = m + l.ln();
            if l > lmax {
                lmax = l;
            }
        }
    }
    Ref {
        out,
        lse,
        s: smax,
        l: lmax,
    }
}

/// One dispatch, its device time in microseconds, and its wall time.
fn gpu_identity() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "device name/driver unavailable (nvidia-smi absent)".to_string(),
    }
}

unsafe fn run_once(vk: &Vk, step: &Step) -> (f64, f64) {
    assert!(prof_arm(vk, 1), "the device reports no usable timestamps");
    let cb = vk.command_buffer();
    vk.begin(cb);
    rec_step(vk, cb, step);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    let t0 = std::time::Instant::now();
    vk.submit_wait(cb);
    let wall = t0.elapsed().as_secs_f64() * 1e6;
    let tl = prof_take(vk);
    assert_eq!(tl.len(), 1, "one stamped dispatch");
    (tl[0].1, wall)
}

const WARM: usize = 3;
const REPS: usize = 7;

/// 3 warmup runs then 7 repeats; mean, min and max of the device span.
unsafe fn campaign(vk: &Vk, step: &Step) -> (f64, f64, f64) {
    for _ in 0..WARM {
        let _ = run_once(vk, step);
    }
    let mut us = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        us.push(run_once(vk, step).0);
    }
    let mean = us.iter().sum::<f64>() / REPS as f64;
    let min = us.iter().copied().fold(f64::MAX, f64::min);
    let max = us.iter().copied().fold(0.0, f64::max);
    (mean, min, max)
}

/// The committed `attn_coopmat_kd.spv` is what `glslc` makes of the committed code under `-DKDF=1`.
#[test]
fn attn_coopmat_kd_spirv_is_the_glsl() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("attn_coopmat_kd_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS_KD)
        .arg(dir.join("attn_coopmat.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    let committed = std::fs::read(dir.join("attn_coopmat_kd.spv")).expect("committed spv");
    assert!(
        fresh == committed,
        "src/gpu/spirv/attn_coopmat_kd.spv is stale: recompile with glslc {GLSLC_ARGS_KD:?}"
    );
}

/// The inference face stays byte-identical: the KD block is one `#ifdef` arm of the same file.
#[test]
fn attn_coopmat_spirv_is_the_glsl() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("attn_coopmat_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(crate::gpu::nn::GLSLC_ARGS)
        .arg(dir.join("attn_coopmat.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    let committed = std::fs::read(dir.join("attn_coopmat.spv")).expect("committed spv");
    assert!(
        fresh == committed,
        "src/gpu/spirv/attn_coopmat.spv is stale: the KD face changed the inference face"
    );
}

/// Read a device buffer back as f32.
unsafe fn read_f32(vk: &Vk, buf: Handle, n: usize) -> Vec<f32> {
    let (rb, mem) = vk
        .buffer(
            n as u64 * 4,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer");
    let p = vk.map(mem).cast::<f32>();
    let cb = vk.command_buffer();
    vk.begin(cb);
    vk.barrier(
        cb,
        STAGE_COMPUTE,
        STAGE_TRANSFER,
        ACCESS_SHADER_WRITE,
        ACCESS_TRANSFER_READ,
    );
    (vk.copy_buffer)(
        cb,
        buf,
        rb,
        1,
        &BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: n as u64 * 4,
        },
    );
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    std::slice::from_raw_parts(p, n).to_vec()
}

/// The error gate, derived: q, k, v and P are each rounded once to bf16 (`u = 2^-8`). With
/// `S_i = max_j scale·Σ_e |q_ie||k_je|`, the row bound is `(4·u·S_i + 2·u)·max|v|`; the
/// logsumexp inherits `|Δlse| ≤ 4·u·S_i + 2·u`. Gated twice: against the f64 oracle on the f32
/// operands (the operand roundings, gate 1) and on the bf16-rounded ones (the kernel's own
/// arithmetic, gate 2). Shapes: T 2048 at the measured KD band and at a unit band, T 8, T 128
/// gated, and T 200 — a length that is not a multiple of the 64-row tile, so the row and key
/// clamps are exercised (a KD tape is T 2048 = 32·64; the fronts admit any n_ctx <= 2048).
#[test]
fn attn_kd_face_matches_f64() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: no Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {} max_shared {}", gpu_identity(), vk.max_shared);
    let shared = attn_kd_shared(ATTN_KD_KEYS, ATTN_KD_ROWS);
    assert!(
        vk.max_shared >= shared,
        "the face needs {shared} B of shared, the device reports {}",
        vk.max_shared
    );
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let cases = [
        &Case {
            label: "kd T8 nh2 nkv1 hd128 ungated",
            t: 8,
            nh: 2,
            nkv: 1,
            gated: false,
            amps: &KD_AMPS,
        },
        &KD_CASE,
        &Case {
            label: "kd T2048 nh16 nkv8 hd128 ungated, unit band",
            t: 2048,
            nh: 16,
            nkv: 8,
            gated: false,
            amps: &UNIT_AMPS,
        },
        &Case {
            label: "kd T128 nh16 nkv8 hd128 gated",
            t: 128,
            nh: 16,
            nkv: 8,
            gated: true,
            amps: &KD_AMPS,
        },
        &Case {
            label: "kd T200 nh16 nkv8 hd128 ungated",
            t: 200,
            nh: 16,
            nkv: 8,
            gated: false,
            amps: &KD_AMPS,
        },
    ];
    for c in cases {
        run_case(&vk, &g, c);
    }
}

/// The f64 adjoint of the causal softmax attention at the KD step's cotangent magnitude, with
/// the magnitude sums each gradient's bound is counted in.
///
/// `dS_ij = p_ij (dP_ij − delta_i) scale` is a *centred* difference: `delta_i = Σ_d dO_ie O_ie`
/// is exactly `Σ_j p_ij dP_ij`, the `p`-weighted mean of the row's `dP` (the identity printed
/// below as `delta gap`), so `dP − delta` cancels most of `dP` at a softmax spike and the
/// absolute error of `dP` — `2u Σ_d |v_e dA_e|` — is what the gradient carries. A magnitude sum
/// of `|dS|` alone would therefore not bound the error; the carriers below are the terms that
/// carry each rounding:
///   `Mq_i = Σ_j (p(|dP| + Σ_d|v dA| + |delta|) + |dS|) |k_j|`,
///   `Mk_j = Σ_i (p(|dP| + Σ_d|v dA| + |delta|) + |dS|) |q_i|`,
///   `Mv_j = Σ_i |p dA_i|`.
#[allow(clippy::type_complexity)]
fn bwd_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dy: &[f32],
    out: &[f64],
    lse: &[f64],
    c: &Case,
    hd: usize,
    scale: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (t, nh, nkv) = (c.t, c.nh, c.nkv);
    let qd: Vec<f64> = q.iter().map(|&x| f64::from(x)).collect();
    let kd: Vec<f64> = k.iter().map(|&x| f64::from(x)).collect();
    let vd: Vec<f64> = v.iter().map(|&x| f64::from(x)).collect();
    let dd: Vec<f64> = dy.iter().map(|&x| f64::from(x)).collect();
    let mut dq = vec![0.0f64; t * nh * hd];
    let mut dk = vec![0.0f64; t * nkv * hd];
    let mut dv = vec![0.0f64; t * nkv * hd];
    let mut mq = vec![0.0f64; t * nh * hd];
    let mut mk = vec![0.0f64; t * nkv * hd];
    let mut mv = vec![0.0f64; t * nkv * hd];
    let mut gapmax = 0.0f64;
    for h in 0..nh {
        let kvh = h / (nh / nkv);
        for i in 0..t {
            let qb = (i * nh + h) * hd;
            let mut delta = 0.0f64;
            for e in 0..hd {
                delta += dd[qb + e] * out[qb + e];
            }
            let mut psum = 0.0f64;
            for j in 0..=i {
                let kb = (j * nkv + kvh) * hd;
                let mut s = 0.0f64;
                let mut dp = 0.0f64;
                let mut mdp = 0.0f64;
                for e in 0..hd {
                    s += qd[qb + e] * kd[kb + e];
                    dp += vd[kb + e] * dd[qb + e];
                    mdp += (vd[kb + e] * dd[qb + e]).abs();
                }
                let p = (s * scale - lse[h * t + i]).exp();
                psum += p * dp;
                let ds = p * (dp - delta) * scale;
                // in the gradient's own units: `dS` already carries `scale`, `p·dP` does not
                let car = scale * p * mdp + ds.abs();
                for e in 0..hd {
                    dq[qb + e] += ds * kd[kb + e];
                    mq[qb + e] += car * kd[kb + e].abs();
                    dk[kb + e] += ds * qd[qb + e];
                    mk[kb + e] += car * qd[qb + e].abs();
                    dv[kb + e] += p * dd[qb + e];
                    mv[kb + e] += (p * dd[qb + e]).abs();
                }
            }
            let gap = (psum - delta).abs() / (delta.abs() + 1.0e-30);
            if gap > gapmax {
                gapmax = gap;
            }
        }
        println!("  delta = Σ_j p dP: max relative gap {gapmax:.2e}");
    }
    (dq, dk, dv, mq, mk, mv)
}

/// The gradient's max error, max bound and their ratio over the entries the oracle covers.
/// `err = max|gpu − ref|`; `bound = max over entries of γ_k·M`, `γ_k = k·δ/(1 − k·δ)` with
/// `δ = 4u` (P, dA and dS each rounded once to bf16, plus the far operand `k` or `q`), evaluated
/// at both reduction depths: `k = HD = 128` inside a pair product and `k = T` along a row.
fn err_gate(label: &str, gpu: &[f64], rf: &[f64], m: &[f64], k_depth: usize, t: usize) -> f64 {
    let u = U;
    // The softmax's own Jacobian: S = q.k reduces over HD with scale = 1/sqrt(HD), so
    // delta(S*scale) <= HD*u*scale = sqrt(HD)*u, and dP/dS = P carries it, giving
    // delta_P = sqrt(HD)*u*P. Added to the four single-rounding terms (P, dA, dS, far operand)
    // and the f32 row summation.
    let d = (4.0 + (k_depth as f64).sqrt()) * u + (k_depth + t + 2) as f64 * (-24.0f64).exp2();
    // The bound is the operand-rounding term `delta * M`, `delta = 4u + (HD + T + 2)*2^-24`:
    // P, dA and dS each rounded once to bf16, plus the far operand's rounding, plus the f32
    // summation of a row. The old gamma_k form is vacuous at bf16 (gamma_128 = 2.0) and is
    // deleted.
    let mut err = 0.0f64;
    let mut bound = 0.0f64;
    let (mut ai, mut ei, mut oi) = (0usize, 0.0f64, 0.0f64);
    for i in 0..gpu.len() {
        let e = (gpu[i] - rf[i]).abs();
        if e > ei {
            ei = e;
            ai = i;
        }
        let b = d * m[i];
        if b > 0.0 && e / b > oi {
            oi = e / b;
        }
        err = err.max((gpu[i] - rf[i]).abs());
        bound = bound.max(d * m[i]);
    }
    println!(
            "  {label}: err {err:.3e} bound {bound:.3e} ratio {:.3e} worst-entry {oi:.3e} (k {k_depth} and {t}) argmax {ai} gpu {:.6e} ref {:.6e} m {:.3e}",
            err / bound,
            gpu[ai],
            rf[ai],
            m[ai]
        );
    err / bound
}

/// The KD backward face at the KD step's cotangent magnitude, and the mutation control.
fn bwd_case(vk: &Vk, g: &GpuCtx, c: &Case) {
    let hd = ATTN_KD_HD as usize;
    let (t, nh, nkv) = (c.t, c.nh, c.nkv);
    let scale = 1.0 / (hd as f64).sqrt();
    let mut rng = Lcg(0x2545_f491_4f6c_dd1d);
    let q = gen(&mut rng, t * nh * hd, c.amps.q);
    let kk = gen(&mut rng, t * nkv * hd, c.amps.k);
    let vv = gen(&mut rng, t * nkv * hd, c.amps.v);
    let gate = gen(&mut rng, t * nh * hd, 1.0);
    // the KD step's cotangent: max |dA| = 1e-4, the measured 5.5e-5..1.3e-4 band
    let mut dy = gen(&mut rng, t * nh * hd, 1.0);
    let amax = dy.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    for x in dy.iter_mut() {
        *x *= 1.0e-4 / amax;
    }
    let fwd = oracle(&q, &kk, &vv, None, c, hd, scale);
    let (dq, dk, dv, mq, mk, mv) = bwd_oracle(&q, &kk, &vv, &dy, &fwd.out, &fwd.lse, c, hd, scale);
    // The device's own operand precision: q, k, v and dA are each rounded once to bf16 at the
    // shared store, so the f64 adjoint run on those rounded operands is the arithmetic the
    // face performs; a bound on it counts accumulation and the bf16 roundings of P and dS.
    let rb = |x: &[f32]| -> Vec<f32> { x.iter().copied().map(round_bf16).collect() };
    let (rq_o, rk_o, rv_o) = (rb(&q), rb(&kk), rb(&vv));
    let rdy = rb(&dy);
    let rfwd = oracle(&rq_o, &rk_o, &rv_o, None, c, hd, scale);
    let (rdq, rdk, rdv, rmq, rmk, rmv) = bwd_oracle(
        &rq_o, &rk_o, &rv_o, &rdy, &rfwd.out, &rfwd.lse, c, hd, scale,
    );
    unsafe {
        let bufs = [
            g.up_f32(&q),
            g.up_f32(&kk),
            g.up_f32(&vv),
            g.up_f32(&dy),
            g.up_f32(&fwd.out.iter().map(|&x| x as f32).collect::<Vec<f32>>()),
            g.up_f32(&gate),
            g.up_f32(&fwd.lse.iter().map(|&x| x as f32).collect::<Vec<f32>>()),
        ];
        let dqb = g.dev_zeroed(t * nh * hd * 4);
        let dkb = g.dev_zeroed(t * nkv * hd * 4);
        let dvb = g.dev_zeroed(t * nkv * hd * 4);
        let dgb = g.dev(4);
        let tiles = nh * t.div_ceil(ATTN_KDB_QUERIES as usize);
        let lk = g.dev_zeroed(tiles * 4);
        let cn = g.dev_zeroed(tiles * 4);
        let qtmp = g.dev_zeroed(attn_kdb_qtmp_len(nh as u32, t as u32) * 4);
        let dispatch = |spv: &[u8]| {
            let step = make_step_named(
                g,
                spv,
                &[
                    bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], dqb, dkb, dvb,
                    dgb, lk, cn, qtmp,
                ],
                nkv as u32,
                t.div_ceil(ATTN_KDB_KEYS as usize) as u32,
                "attn_kd_bwd",
            );
            let cb = vk.command_buffer();
            vk.begin(cb);
            rec_step(vk, cb, &step);
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            vk.submit_wait(cb);
            (
                read_f32(vk, dqb, t * nh * hd),
                read_f32(vk, dkb, t * nkv * hd),
                read_f32(vk, dvb, t * nkv * hd),
            )
        };
        println!("KDB t {t} nh {nh} nkv {nkv} hd {hd} max|dA| {:.1e}", 1.0e-4);
        let (gq, gk, gv) = dispatch(&attn_kd_bwd_bytes(
            nh as u32,
            nkv as u32,
            t as u32,
            scale as f32,
            false,
        ));
        let to64 = |x: &[f32]| -> Vec<f64> { x.iter().map(|&y| f64::from(y)).collect() };
        let rq = err_gate("dq", &to64(&gq), &dq, &mq, hd, t);
        let rk = err_gate("dk", &to64(&gk), &dk, &mk, hd, t);
        let rv = err_gate("dv", &to64(&gv), &dv, &mv, hd, t);
        println!("  vs the bf16-ROUNDED-operand oracle (the face's own arithmetic):");
        let r2q = err_gate("dq", &to64(&gq), &rdq, &rmq, hd, t);
        let r2k = err_gate("dk", &to64(&gk), &rdk, &rmk, hd, t);
        let r2v = err_gate("dv", &to64(&gv), &rdv, &rmv, hd, t);
        for (name, r) in [("dq", rq), ("dk", rk), ("dv", rv)] {
            assert!(
                r.is_finite() && r <= 1.0,
                "{name}: error/bound ratio {r:.3e} exceeds 1 (the bound is delta*M)"
            );
        }
        let mag = |x: &[f32]| x.iter().fold(0.0f32, |a, y| a.max(y.abs()));
        println!(
            "  mags gpu dq {:.3e} dk {:.3e} dv {:.3e} | ref dq {:.3e} dk {:.3e} dv {:.3e}",
            mag(&gq),
            mag(&gk),
            mag(&gv),
            dq.iter().fold(0.0f64, |a, y| a.max(y.abs())),
            dk.iter().fold(0.0f64, |a, y| a.max(y.abs())),
            dv.iter().fold(0.0f64, |a, y| a.max(y.abs())),
        );
        // the control: `- delta` dropped from `dS = P (dP - delta) SCALE`, in a throwaway copy
        let src = std::fs::read_to_string(format!(
            "{}/src/gpu/spirv/attn_coopmat.comp",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("the face's source");
        let mutant = src.replace("- sD[ii]) * SCALE", ") * SCALE");
        assert_ne!(mutant, src, "the mutation did not change the source");
        let dir = std::env::temp_dir().join("kdb-mut");
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("attn_coopmat.comp"), &mutant).expect("mutant source");
        let out = dir.join("mut.spv");
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS_KDB)
            .args(["-o"])
            .arg(&out)
            .arg(dir.join("attn_coopmat.comp"))
            .status()
            .expect("glslc");
        assert!(st.success(), "the mutant did not compile: {st}");
        let mspv = std::fs::read(&out).expect("mutant spv");
        assert_ne!(mspv, ATTN_COOPMAT_KDB_SPV, "the mutant is the face");
        let mw: Vec<u32> = mspv
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let mw = attn_coop_specialize(
            &mw,
            &[
                (0, nh as u32),
                (1, nkv as u32),
                (2, t as u32),
                (3, (scale as f32).to_bits()),
                (4, 0),
            ],
        );
        let (hq, _, _) = dispatch(&crate::gpu::nn::bytes(mw));
        let err = |a: &[f32], b: &[f64]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| (f64::from(x) - y).abs())
                .fold(0.0f64, f64::max)
        };
        let me = err(&hq, &dq);
        let ce = err(&gq, &dq);
        let mb = (0..mq.len()).map(|i| mq[i]).fold(0.0f64, f64::max);
        let d = (4.0 + (hd as f64).sqrt()) * U + (hd + t + 2) as f64 * (-24.0f64).exp2();
        println!(
            "MUTATION no -delta: dq err {me:.3e} of bound {:.3e} (committed {ce:.3e}, {:.0}x)",
            d * mb,
            me / ce
        );
        // `dS` without its centring term is the load-bearing failure this face must not have.
        // The bound above now carries `p|delta|`, so the mutant's *error* is bounded too; the
        // control is the separation from the committed face, not the bound.
        assert!(
            me > 10.0 * ce,
            "the dropped `- delta` term is not separable: {me:.3e} against the committed \
                 face's {ce:.3e}"
        );
    }
}

/// The one-tile fixture: one key block, one query block, so every product is a single tile.
static KD_CASE_256: Case = Case {
    label: "kd bwd T32 nh2 nkv1 hd128 ungated, one tile",
    t: 32,
    nh: 2,
    nkv: 1,
    gated: false,
    amps: &KD_AMPS,
};

/// The KD backward face against the f64 adjoint at the KD step's cotangent magnitude, at the
/// one-tile shape. The bound is `delta*M` and the gate is `ratio <= 1` per gradient.
#[test]
fn attn_kd_bwd_face_matches_f64() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: no Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {} max_shared {}", gpu_identity(), vk.max_shared);
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    bwd_case(&vk, &g, &KD_CASE_256);
    // The unit band: the same face where the softmax is not a single-key spike, so P's own
    // sensitivity to the operand rounding does not amplify. A defect survives both; an
    // amplification of the spike does not.
    bwd_case(&vk, &g, &KDB_UNIT);
    // T 512: the causal walk crosses 8 key blocks and 8 query blocks, so `qtmp`'s
    // lock/counter is exercised across many contributors.
    bwd_case(&vk, &g, &KDB_T512);
}

static KDB_T512: Case = Case {
    label: "kd bwd T512 nh2 nkv1 hd128 ungated",
    t: 512,
    nh: 2,
    nkv: 1,
    gated: false,
    amps: &KD_AMPS,
};

/// The KDB face at the production shape on device timestamps: ms per call, against the
/// 15.01 ms the three-dispatch path takes (dq 8.87 + dkv 5.03 + prep 1.11) and fmhaB's 2.73.
#[test]
#[ignore = "timing campaign: run it alone on a quiet host"]
fn kdb_profile() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: no Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {}", gpu_identity());
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let hd = ATTN_KD_HD as usize;
    let (t, nh, nkv) = (2048usize, 16usize, 8usize);
    let scale = 1.0 / (hd as f64).sqrt();
    let mut rng = Lcg(0x2545_f491_4f6c_dd1d);
    let q = gen(&mut rng, t * nh * hd, KD_AMPS.q);
    let kk = gen(&mut rng, t * nkv * hd, KD_AMPS.k);
    let vv = gen(&mut rng, t * nkv * hd, KD_AMPS.v);
    let dy = gen(&mut rng, t * nh * hd, 1.0);
    let ob = gen(&mut rng, t * nh * hd, 1.0);
    let lb = gen(&mut rng, nh * t, 1.0);
    let gate = gen(&mut rng, nh * hd, 1.0);
    unsafe {
        let bufs = [
            g.up_f32(&q),
            g.up_f32(&kk),
            g.up_f32(&vv),
            g.up_f32(&dy),
            g.up_f32(&ob),
            g.up_f32(&gate),
            g.up_f32(&lb),
        ];
        let dqb = g.dev_zeroed(t * nh * hd * 4);
        let dkb = g.dev_zeroed(t * nkv * hd * 4);
        let dvb = g.dev_zeroed(t * nkv * hd * 4);
        let dgb = g.dev(hd * 4);
        let tiles = nh * t.div_ceil(ATTN_KDB_QUERIES as usize);
        let lk = g.dev_zeroed(tiles * 4);
        let cn = g.dev_zeroed(tiles * 4);
        let qtmp = g.dev_zeroed(attn_kdb_qtmp_len(nh as u32, t as u32) * 4);
        let step = attn_kd_bwd_dispatch(
            &g,
            bufs[0],
            bufs[1],
            bufs[2],
            bufs[3],
            bufs[4],
            bufs[5],
            bufs[6],
            dqb,
            dkb,
            dvb,
            dgb,
            lk,
            cn,
            qtmp,
            nh as u32,
            nkv as u32,
            t as u32,
            scale as f32,
            false,
        );
        let (mean, min, max) = campaign(&vk, &step);
        println!(
            "KDB PROFILE t {t} nh {nh} nkv {nkv} hd {hd}: {:.3} ms mean, {:.3}..{:.3} min..max",
            mean / 1000.0,
            min / 1000.0,
            max / 1000.0
        );
        println!(
            "  vs 15.01 ms (dq 8.87 + dkv 5.03 + prep 1.11): {:.2}x | vs fmhaB 2.73 ms: {:.2}x",
            15.01 / (mean / 1000.0),
            (mean / 1000.0) / 2.73
        );
    }
}

static KDB_UNIT: Case = Case {
    label: "kd bwd T32 nh2 nkv1 hd128 unit band",
    t: 32,
    nh: 2,
    nkv: 1,
    gated: false,
    amps: &UNIT_AMPS,
};

#[allow(clippy::too_many_lines)]
fn run_case(vk: &Vk, g: &GpuCtx, c: &Case) {
    let hd = ATTN_KD_HD as usize;
    let (t, nh, nkv) = (c.t, c.nh, c.nkv);
    let scale = 1.0 / (hd as f64).sqrt();
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let q = gen(&mut rng, t * nh * hd, c.amps.q);
    let k = gen(&mut rng, t * nkv * hd, c.amps.k);
    let v = gen(&mut rng, t * nkv * hd, c.amps.v);
    let gate = gen(&mut rng, t * nh * hd, 1.0);
    let gated = c.gated.then_some(&gate[..]);
    // gate 1: exact arithmetic on the f32 operands, the port's total error
    let exact = oracle(&q, &k, &v, gated, c, hd, scale);
    // gate 2: the same oracle on the operands rounded to bf16, the kernel's own arithmetic
    let bq: Vec<f32> = q.iter().copied().map(round_bf16).collect();
    let bk: Vec<f32> = k.iter().copied().map(round_bf16).collect();
    let bv: Vec<f32> = v.iter().copied().map(round_bf16).collect();
    let bf = oracle(&bq, &bk, &bv, gated, c, hd, scale);

    unsafe {
        let qb = g.up_f32(&q);
        let kb = g.up_f32(&k);
        let vb = g.up_f32(&v);
        let gb = g.up_f32(&gate);
        let ob = g.dev_zeroed(t * nh * hd * 4);
        let lb = g.dev_zeroed(nh * t * 4);
        let step = attn_kd_dispatch(
            g,
            qb,
            kb,
            vb,
            gb,
            ob,
            lb,
            nh as u32,
            nkv as u32,
            t as u32,
            scale as f32,
            c.gated,
        );
        let cb = vk.command_buffer();
        vk.begin(cb);
        rec_step(vk, cb, &step);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        let gpu = read_f32(vk, ob, t * nh * hd);
        let glse = read_f32(vk, lb, nh * t);
        assert!(
            gpu.iter().all(|x| x.is_finite()),
            "{}: non-finite output",
            c.label
        );

        let mut vmax = 0.0f64;
        for x in &v {
            vmax = vmax.max(f64::from(*x).abs());
        }
        // Gate 1, derived: q, k, v and P are each rounded once (u = 2^-8). With
        // S_i = max_j scale·Σ|q_ie||k_je|, |o - o_exact| <= (4u·S_i + 2u)·max|v| per row, and
        // |lse - lse_exact| <= 4u·S_i + 2u. At the KD band S_i ~ 267 makes this bound 3.0, past
        // the 2·max|v| = 1.44 two convex combinations of v can differ by: it catches structural
        // breakage only, and gate 2 below is the one with teeth there.
        let smax = exact.s.iter().fold(0.0f64, |a, b| a.max(*b));
        let mut e1 = 0.0f64;
        let mut l1 = 0.0f64;
        for h in 0..nh {
            for i in 0..t {
                let row = (i * nh + h) * hd;
                let mut e = 0.0f64;
                for d in 0..hd {
                    e = e.max((f64::from(gpu[row + d]) - exact.out[row + d]).abs());
                }
                e1 = e1.max(e);
                l1 = l1.max((f64::from(glse[h * t + i]) - exact.lse[h * t + i]).abs());
            }
        }
        let bound1 = (4.0 * U * smax + 2.0 * U) * vmax;
        let lbound1 = 4.0 * U * smax + 2.0 * U;
        println!(
            "{}: gate1 vs f64 on f32 operands: max row err {e1:.3e} / bound {bound1:.3e} \
                 ({:.3} of it), lse {l1:.3e} / {lbound1:.3e}, S {smax:.0}, lmax {:.2}",
            c.label,
            e1 / bound1,
            exact.l
        );
        assert!(
            e1 <= bound1,
            "{}: row error {e1:.3e} exceeds the derived operand bound {bound1:.3e}",
            c.label
        );
        assert!(
            l1 <= lbound1,
            "{}: logsumexp error {l1:.3e} exceeds {lbound1:.3e}",
            c.label
        );

        // Gate 2, derived from the operand roundings being accounted for by the bf16 oracle:
        // the only rounding left on the device is P's (one bf16 round, |Δp|_1 <= u) and the
        // f32 MMA accumulation. |o - o_bf16| <= (u + 2^-20)·max|v| per row; this is the gate
        // that sees a softmax bug at the KD band, where gate 1 is vacuous.
        let bound2 = (U + 9.6e-7) * vmax;
        let mut e2 = 0.0f64;
        let mut ratio2 = 0.0f64;
        for h in 0..nh {
            for i in 0..t {
                let row = (i * nh + h) * hd;
                let mut e = 0.0f64;
                for d in 0..hd {
                    e = e.max((f64::from(gpu[row + d]) - bf.out[row + d]).abs());
                }
                e2 = e2.max(e);
                ratio2 = ratio2.max(e / bound2);
            }
        }
        println!(
            "{}: gate2 vs f64 on bf16 operands: max row err {e2:.3e} / bound {bound2:.3e} \
                 ({ratio2:.3} of it), lse {:.3e}, max|v| {vmax:.3}",
            c.label,
            bf.lse
                .iter()
                .zip(&glse)
                .fold(0.0f64, |a, (x, y)| a.max((x - f64::from(*y)).abs())),
        );
        assert!(
            ratio2 <= 1.0,
            "{}: the kernel's own arithmetic is {ratio2:.3} of the bf16-operand bound \
                 ({e2:.3e} vs {bound2:.3e})",
            c.label
        );
    }
}

/// The face beside `attn_map` mode 1 at the KD production shape, device microseconds: 3 warmup
/// and 7 repeats each, mean [min, max], and the ratio to the reference's 964 us per call
/// (`~/nq-ref/kd/llvm/sass/fmhaF.sass`, grid [32,16,1]).
/// Compile one KD-face variant from the committed `.comp` under extra `-D` flags, into
/// `$TMPDIR`.
fn compile_variant(tag: &str, extra: &[&str]) -> Vec<u8> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("attn_kd_{tag}_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS_KD)
        .args(extra)
        .arg(dir.join("attn_coopmat.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH");
    assert!(st.success(), "glslc failed for the {tag} variant: {st}");
    let spv = std::fs::read(&out).expect("variant spv");
    let _ = std::fs::remove_file(&out);
    spv
}

/// A grid of `wg` workgroups folded into `(x <= 65535, y)`, the prep's launch shape.
fn fold(wg: usize) -> (u32, u32) {
    let gx = wg.clamp(1, 65_535);
    (gx as u32, wg.div_ceil(gx).max(1) as u32)
}

/// The fixture of one case: `(q, k, v, gate)`.
fn fixture(c: &Case) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = ATTN_KD_HD as usize;
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    (
        gen(&mut rng, c.t * c.nh * hd, c.amps.q),
        gen(&mut rng, c.t * c.nkv * hd, c.amps.k),
        gen(&mut rng, c.t * c.nkv * hd, c.amps.v),
        gen(&mut rng, c.t * c.nh * hd, 1.0),
    )
}

/// The committed tiling beside the NSG = 2 variant at the KD production shape, device
/// microseconds: 3 warmup and 7 repeats each, mean [min, max], and the ratio to the reference's
/// 964 us per call (`~/nq-ref/kd/llvm/sass/fmhaF.sass`, grid [32,16,1]). The
/// `attn_map` mode 1 arm this row was measured against is deleted; its receipt is the turn-1
/// 7000.1 us [6970.0, 7050.3].
#[test]
#[ignore = "timed campaign: opens the device and holds the GPU lock"]
fn attn_kd_face_speed() {
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let c = &KD_CASE;
    let hd = ATTN_KD_HD as usize;
    let (t, nh, nkv) = (c.t, c.nh, c.nkv);
    let scale = 1.0 / (hd as f64).sqrt();
    let (q, k, v, _) = fixture(c);
    println!("device: {} max_shared {}", gpu_identity(), vk.max_shared);
    unsafe {
        let qb = g.up_f32(&q);
        let kb = g.up_f32(&k);
        let vb = g.up_f32(&v);
        let gb = g.dev(4);
        let ob = g.dev_zeroed(t * nh * hd * 4);
        let lb = g.dev_zeroed(nh * t * 4);
        let ob2 = g.dev_zeroed(t * nh * hd * 4);
        let kd = attn_kd_dispatch(
            &g,
            qb,
            kb,
            vb,
            gb,
            ob,
            lb,
            nh as u32,
            nkv as u32,
            t as u32,
            scale as f32,
            false,
        );
        let nsg2 = compile_variant("nsg2", &["-DNSG=2"]);
        let kd2 = make_step_named(
            &g,
            &nsg2,
            &[qb, kb, vb, gb, ob2, lb],
            nh as u32,
            (t as u32).div_ceil(32),
            "attn_kd_nsg2",
        );
        for (name, step) in [("attn_kd NSG4", &kd), ("attn_kd NSG2", &kd2)] {
            let (mean, min, max) = campaign(&vk, step);
            println!(
                "{name}: {mean:.3} us mean [{min:.3}, {max:.3}] ({WARM} warmup, {REPS} reps), \
                     {:.3}x fmhaF 964 us",
                mean / 964.0
            );
            for s in vk.stats(step.pipe) {
                if s.name.to_lowercase().contains("register")
                    || s.name.to_lowercase().contains("shared")
                    || s.name.to_lowercase().contains("occup")
                {
                    println!("  {name} {} = {}", s.name, s.value);
                }
            }
        }
    }
    prof_disarm();
}

/// **The KD operand amplitudes, measured on one real KD layer.** The error gate's fixture is
/// built at these numbers. `q` and `k` leave `attn_prep` RMS-normalised per head with the learned
/// gain (`src/gpu/spirv/train/attn_map.comp` mode 0: `x *= inversesqrt(tot/HD + eps) * gains`),
/// so their element RMS is the gain's RMS; `v` is the v-projection of the same normalised rows,
/// so each output element's standard deviation is `sqrt(sum_e (W[row,e]·n1_e)^2)` at unit input
/// RMS, and its maximum over the layer's 4.2e6 elements is ~5.1 sigma.
#[test]
#[ignore = "measurement: reads the KD student's layer 0"]
fn kd_operand_magnitudes() {
    let model = "~/nq-ref/q3-0p6b-f16.gguf";
    let g = crate::format::gguf::Gguf::open(model).expect("open the KD student");
    let vd = g
        .header
        .tensor("blk.0.attn_v.weight")
        .expect("v desc")
        .dims
        .clone();
    let (nkvh, d) = (vd[1] as usize, vd[0] as usize);
    let hd = ATTN_KD_HD as usize;
    let qg = g.dequant("blk.0.attn_q_norm.weight").expect("q gain");
    let kg = g.dequant("blk.0.attn_k_norm.weight").expect("k gain");
    let n1 = g.dequant("blk.0.attn_norm.weight").expect("input norm");
    let wv = g.dequant("blk.0.attn_v.weight").expect("v projection");
    let rms =
        |v: &[f32]| (v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    let max = |v: &[f32]| v.iter().fold(0.0f64, |a, x| a.max(f64::from(*x).abs()));
    let (qr, qm) = (rms(&qg), max(&qg));
    let (kr, km) = (rms(&kg), max(&kg));
    // the v projection's rows against the input norm: element sigma per output dim
    let mut s2 = 0.0f64;
    let mut smax = 0.0f64;
    for r in 0..nkvh {
        let row = &wv[r * d..(r + 1) * d];
        let s: f64 = (0..d)
            .map(|e| (f64::from(row[e]) * f64::from(n1[e])).powi(2))
            .sum::<f64>();
        s2 += s;
        smax = smax.max(s.sqrt());
    }
    let vr = (s2 / nkvh as f64).sqrt();
    println!(
        "kd operands blk.0: |q| rms {qr:.3} max {qm:.3} (gain {hd} dims), |k| rms {kr:.3} max \
             {km:.3}, |v| rms {vr:.3} row sigma max {smax:.3} -> max|v| ~{:.2} at 4.2e6 samples; \
             d {d} nkv*hd {nkvh}",
        5.1 * vr
    );
}

/// One mutation of the committed `.comp`, compiled in `$TMPDIR` from a throwaway copy — never the
/// worktree file — with the face's spec constants patched as [`attn_coopmat_kd_module`] patches
/// them. `needle` must occur exactly once in the source; a moved needle fails the control.
fn mutant_bytes(
    nh: u32,
    nkv: u32,
    t: u32,
    scale: f32,
    gated: bool,
    needle: &str,
    repl: &str,
) -> Vec<u8> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let src = std::fs::read_to_string(dir.join("attn_coopmat.comp")).expect("comp");
    assert_eq!(
        src.matches(needle).count(),
        1,
        "the control's needle must occur once: {needle:?}"
    );
    let mut_src = src.replacen(needle, repl, 1);
    let pid = std::process::id();
    let mut_path = std::env::temp_dir().join(format!("attn_kd_mut_{pid}.comp"));
    std::fs::write(&mut_path, &mut_src).expect("write the mutated copy");
    let mut_spv = std::env::temp_dir().join(format!("attn_kd_mut_{pid}.spv"));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS_KD)
        .arg(&mut_path)
        .arg("-o")
        .arg(&mut_spv)
        .status()
        .expect("glslc must be on PATH");
    assert!(st.success(), "glslc failed on the mutated copy: {st}");
    let raw = std::fs::read(&mut_spv).expect("mutated spv");
    let _ = std::fs::remove_file(&mut_path);
    let _ = std::fs::remove_file(&mut_spv);
    let words: Vec<u32> = raw
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect();
    bytes(attn_coop_specialize(
        &words,
        &[
            (0, nh),
            (1, nkv),
            (2, t),
            (3, scale.to_bits()),
            (4, u32::from(gated)),
        ],
    ))
}

/// **The negative control, at both operand bands.** Two mutations of the committed `.comp`, each
/// a real semantic deletion, run beside the unmutated face and judged by gate 2 — the kernel's own
/// arithmetic against the f64 oracle on bf16-rounded operands, `bound = (u + 2^-20)·max|v|`.
///
/// - the O rescale: `acc = o[nb] * fal` becomes `acc = o[nb]`, dropping the previous slabs'
///   `exp2(m_prev - m_new)`;
/// - the causal guard: `len = lrow + 1` becomes `len = T`, letting every row attend to the keys
///   after it.
///
/// Gate 1 — the operand-rounding bound `(4u·S_i + 2u)·max|v|` — is reported beside them: at the
/// KD band `S_i ~ 267` makes it 2.75, larger than `2·max|v| = 1.44` (the largest difference two
/// convex combinations of `v` can have), so it is vacuous there and only gate 2 sees a softmax
/// bug: the same two mutants read 0.56 and 0.33 of gate 1. The turn-1 control used gate 1 alone
/// and read 0.56 — not red — which is why gate 2 exists.
#[test]
#[ignore = "negative control: opens the device and runs the f64 oracle"]
fn attn_kd_negative_controls() {
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let hd = ATTN_KD_HD as usize;
    let scale = 1.0 / (hd as f64).sqrt();
    let bands = [("unit band", &UNIT_AMPS), ("measured KD band", &KD_AMPS)];
    for (label, amps) in bands {
        let c = &Case {
            label: "kd T512 nh16 nkv8 hd128 ungated",
            t: 512,
            nh: 16,
            nkv: 8,
            gated: false,
            amps,
        };
        let (t, nh, nkv) = (c.t, c.nh, c.nkv);
        let (q, k, v, _) = fixture(c);
        let bq: Vec<f32> = q.iter().copied().map(round_bf16).collect();
        let bk: Vec<f32> = k.iter().copied().map(round_bf16).collect();
        let bv: Vec<f32> = v.iter().copied().map(round_bf16).collect();
        let exact = oracle(&q, &k, &v, None, c, hd, scale);
        let bf = oracle(&bq, &bk, &bv, None, c, hd, scale);
        let mut vmax = 0.0f64;
        for x in &v {
            vmax = vmax.max(f64::from(*x).abs());
        }
        let bound1 = (4.0 * U * exact.s.iter().fold(0.0f64, |a, b| a.max(*b)) + 2.0 * U) * vmax;
        let bound2 = (U + 9.6e-7) * vmax;
        unsafe {
            let qb = g.up_f32(&q);
            let kb = g.up_f32(&k);
            let vb = g.up_f32(&v);
            let gb = g.dev(4);
            let lb = g.dev_zeroed(nh * t * 4);
            let run = |spv: &[u8], name: &'static str| -> (f64, f64) {
                let ob = g.dev_zeroed(t * nh * hd * 4);
                let step = make_step_named(
                    &g,
                    spv,
                    &[qb, kb, vb, gb, ob, lb],
                    nh as u32,
                    (t as u32).div_ceil(ATTN_KD_ROWS),
                    name,
                );
                let cb = vk.command_buffer();
                vk.begin(cb);
                rec_step(&vk, cb, &step);
                check((vk.end_cb)(cb), "vkEndCommandBuffer");
                vk.submit_wait(cb);
                let gpu = read_f32(&vk, ob, t * nh * hd);
                let mut e1 = 0.0f64;
                let mut e2 = 0.0f64;
                for i in 0..t * nh * hd {
                    e1 = e1.max((f64::from(gpu[i]) - exact.out[i]).abs());
                    e2 = e2.max((f64::from(gpu[i]) - bf.out[i]).abs());
                }
                (e1, e2)
            };
            let (nh32, nkv32, t32) = (nh as u32, nkv as u32, t as u32);
            let good = run(
                &attn_coopmat_kd_bytes(nh32, nkv32, t32, scale as f32, false),
                "attn_kd",
            );
            let rescale = run(
                &mutant_bytes(
                    nh32,
                    nkv32,
                    t32,
                    scale as f32,
                    false,
                    "FRAG_KD_F acc = o[nb] * fal;",
                    "FRAG_KD_F acc = o[nb];",
                ),
                "attn_kd_norescale",
            );
            let causal = run(
                &mutant_bytes(
                    nh32,
                    nkv32,
                    t32,
                    scale as f32,
                    false,
                    "const uint len = lrow + 1u;",
                    "const uint len = T;",
                ),
                "attn_kd_nocausal",
            );
            println!(
                "{label}: committed {:.3e} of gate2 ({:.3e}) and {:.3} of gate1 ({:.3e}); \
                     rescale dropped {:.3e} ({:.0}x gate2, {:.2}x gate1); causal dropped {:.3e} \
                     ({:.0}x gate2, {:.2}x gate1)",
                good.1,
                bound2,
                good.0 / bound1,
                bound1,
                rescale.1,
                rescale.1 / bound2,
                rescale.0 / bound1,
                causal.1,
                causal.1 / bound2,
                causal.0 / bound1
            );
            assert!(
                good.1 <= bound2,
                "{label}: the committed face is red at gate2"
            );
            assert!(
                rescale.1 > bound2,
                "{label}: the dropped O rescale is green at gate2 ({:.3e} <= {bound2:.3e})",
                rescale.1
            );
            assert!(
                causal.1 > bound2,
                "{label}: the dropped causal guard is green at gate2 ({:.3e} <= {bound2:.3e})",
                causal.1
            );
        }
    }
}

/// Instrument: the driver's own compiled form of the committed face, alone, in a fresh pipeline
/// cache ([`Vk::isolated_cache_blob`]), written to `$TMPDIR/attn_kd_face.blob` for
/// `cuobjdump -sass` / `nvdisasm`. The static work in the face's key-slab loop, from the SPIR-V:
/// 32 `OpCooperativeMatrixMulAddKHR` for S (4 key blocks x 8 dim blocks) and 32 for P V (8 dim
/// blocks x 4 key blocks) per 64-key slab, from bf16 x bf16 into an f32 accumulator (the
/// reference's `HMMA.1688.F32.BF16`, 64 of them in its main loop).
#[test]
#[ignore = "instrument: dumps the pipeline-cache blob"]
fn attn_kd_face_sass() {
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _m) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let c = &KD_CASE;
    let hd = ATTN_KD_HD as usize;
    let scale = 1.0 / (hd as f64).sqrt();
    let (q, k, v, _) = fixture(c);
    unsafe {
        let bufs = [
            g.up_f32(&q),
            g.up_f32(&k),
            g.up_f32(&v),
            g.dev(4),
            g.dev_zeroed(c.t * c.nh * hd * 4),
            g.dev_zeroed(c.nh * c.t * 4),
        ];
        let spv = attn_coopmat_kd_bytes(c.nh as u32, c.nkv as u32, c.t as u32, scale as f32, false);
        let (layout, _dset) = vk.ssbo_set(&bufs);
        let (_pipe, blob) = vk.isolated_cache_blob(&spv, layout);
        let p = std::env::temp_dir().join("attn_kd_face.blob");
        std::fs::write(&p, &blob).expect("write the blob");
        let elf = blob.windows(4).position(|w| w == b"\x7fELF");
        let nv = blob.windows(8).position(|w| w == b"NVDANVVM");
        println!(
            "kd face blob: {} B, ELF magic at {elf:?}, NVDANVVM at {nv:?}, written to {}",
            blob.len(),
            p.display()
        );
    }
}
