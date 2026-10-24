//! The device backward of the softmax attention mixer, gated (Qwen3.5's output gate) or ungated
//! (Qwen3's, no gate): the adjoint of `src/gpu/spirv/train/attn_map.comp` as f32 SPIR-V, in
//! FlashAttention-2 form.
//!
//! The flag is [`AttnBwd::new`]'s `gated`: it sets bit 2 of every `mode` word the tape records, and
//! `mode & 3` stays the stage, so a gated tape is byte-identical to the one this wrapper recorded
//! before the flag existed. Ungated `qf` and `dqf` are `[T][nh·hd]` (head `h` at `h·hd`), `dA_i =
//! dy_i`, and the gate buffer is neither read nor written.
//!
//! Three dispatches, one fixed summation order per output, no float atomics:
//!
//! 1. `attn_bwd_dq` (`mode 0`), one workgroup per `(chunk, head, query i)`: recompute the causal
//!    scores and the row softmax, store `lse_i = max + ln Σ` ([`AttnBwdShape`] fixing `T = chunks ·
//!    n_ctx` independent sequences), write `dA_i = dy_i ⊙ σ(gate_i)` (`dy_i` ungated), `D_i = dA_i·A_i`,
//!    the gate half of `dqf` (`dgate_i`, gated only) and then `dq_rot_i = scale·Σ_{j≤i} dS_ij·k_j`
//!    with `dS_ij = P_ij·(dP_ij − D_i)`.
//! 2. `attn_bwd_dkv` (`mode 1`), one workgroup per `(chunk, kv head g, key j)`:
//!    `dk_rot_j = scale·Σ_{h∈g} Σ_{i≥j} dS_ij·q_i` and `dv_j = Σ_{h∈g} Σ_{i≥j} P_ij·dA_i`, the
//!    query heads of `g` ascending then `i` ascending.
//! 3. `attn_bwd_prep` (`mode 2`), one workgroup per `(row, slot)`: the rope transpose then the
//!    RMS-normalisation backward with its gain, into the q half of `dqf` or into `dkf`. The norm
//!    gains are frozen (the distillation tunes scales only), so no gain gradient is produced.
//!
//! # Trust boundary
//!
//! Not covered: the GPU driver, the SPIR-V compiler, the f32 transcendental precision of the device
//! and the f32 probability recomputation `exp(s_ij − lse_i)`. The evidence is `tested` against the
//! f64 forward and adjoint in this file (`attn_bwd_ref_matches_fd`,
//! `attn_bwd_device_matches_ref` and their ungated twins); the geometry constants of a layer are
//! the caller's [`AttnBwdShape`].

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "widths, rows and workgroup counts index u32 invocations; f64 references map to f32"
)]

use super::block::{download, host_buffer, upload, HostBuffer};
use crate::gpu::decode::{make_step_named, rec_step, GpuCtx, Step};
use crate::gpu::vulkan::*;

/// `attn_map_bwd.comp` compiled by `glslc` with [`crate::gpu::nn::GLSLC_ARGS`].
pub static ATTN_MAP_BWD_SPV: &[u8] = include_bytes!("../../spirv/train/attn_map_bwd.spv");

const MAX_GROUPS_X: usize = 65_535;

/// The mixer geometry of one layer over `chunks` independent sequences of `n_ctx` columns each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnBwdShape {
    /// Columns per chunk; `n_ctx ≤ 1024` (the shader holds 1024 scores).
    pub n_ctx: usize,
    /// Independent sequences in the tape; `T = chunks · n_ctx`.
    pub chunks: usize,
    /// Query heads; divisible by `nkv`.
    pub nh: usize,
    /// Key/value heads.
    pub nkv: usize,
    /// Head width; `≤ 256`.
    pub hd: usize,
    /// Rotated dims, even; the rest of the head passes the rope through.
    pub rot: usize,
    /// RoPE base.
    pub theta: f32,
    /// The model's RMS-norm epsilon.
    pub eps: f32,
}

impl AttnBwdShape {
    /// `T`, the tape's column count.
    #[must_use]
    pub fn t(&self) -> usize {
        self.n_ctx * self.chunks
    }
}

/// The forward's activations the backward reads, in the forward's buffer order.
///
/// `q`, `gate`, `k`, `v` are the `attn_map` prep outputs; `qf`, `kf`, `gains` are that prep's
/// inputs, which the adjoint of the norm and the rope needs; `dy` is `d out`.
pub struct AttnBwdIn<'a> {
    /// `[T][nh·2·hd]` gated (the pre-norm query rows, `q` at `h·2·hd` and gate at `h·2·hd + hd`),
    /// `[T][nh·hd]` ungated.
    pub qf: &'a [f32],
    /// `[T][nkv·hd]`, the pre-norm key rows.
    pub kf: &'a [f32],
    /// `[2·hd]`, the q norm gains then the k norm gains.
    pub gains: &'a [f32],
    /// `[T][nh·hd]`, the rotated queries.
    pub q: &'a [f32],
    /// `[T][nh·hd]`, the mixer gates; empty when the wrapper is ungated.
    pub gate: &'a [f32],
    /// `[T][nkv·hd]`, the rotated keys.
    pub k: &'a [f32],
    /// `[T][nkv·hd]`, the values.
    pub v: &'a [f32],
    /// `[T][nh·hd]`, `d out`.
    pub dy: &'a [f32],
}

/// The backward's outputs. `lse` and `dsum` are the FlashAttention-2 row statistics.
pub struct AttnBwdGrads {
    /// `[T·nh]`, `max_i + ln Σ_j exp(s_ij − max_i)`.
    pub lse: Vec<f32>,
    /// `[T·nh]`, `D_i = dA_i·A_i`.
    pub dsum: Vec<f32>,
    /// `[T][nh·2·hd]` gated: the q half through the rope and the norm, the gate half `dgate`;
    /// `[T][nh·hd]` ungated: the q half only.
    pub dqf: Vec<f32>,
    /// `[T][nkv·hd]`, through the rope and the norm.
    pub dkf: Vec<f32>,
    /// `[T][nkv·hd]`.
    pub dv: Vec<f32>,
}

/// A grid of `wg` workgroups folded into `(x ≤ 65535, y)`.
fn fold(wg: usize) -> (u32, u32) {
    let gx = wg.clamp(1, MAX_GROUPS_X);
    (gx as u32, wg.div_ceil(gx).max(1) as u32)
}

/// The backward over `chunks · n_ctx` columns: every activation and gradient device-resident, one
/// recorded tape per call.
pub struct AttnBwd {
    s: AttnBwdShape,
    t: usize,
    /// `false` sets bit 2 of every `mode` word: no gate half, `dA = dy`.
    gated: bool,
    // Inputs, in [`AttnBwdIn`] order.
    qf: Handle,
    kf: Handle,
    gains: Handle,
    q: Handle,
    gate: Handle,
    k: Handle,
    v: Handle,
    dy: Handle,
    // Row statistics and the recompute scratch.
    lse: Handle,
    dsum: Handle,
    #[allow(
        dead_code,
        reason = "the handle keeps the buffer alive for the recorded tape"
    )]
    da: Handle,
    #[allow(
        dead_code,
        reason = "the handle keeps the buffer alive for the recorded tape"
    )]
    dq_rot: Handle,
    #[allow(
        dead_code,
        reason = "the handle keeps the buffer alive for the recorded tape"
    )]
    dk_rot: Handle,
    // Outputs.
    dqf: Handle,
    dkf: Handle,
    dv: Handle,
    stage: HostBuffer,
    rb: HostBuffer,
    cb: CommandBuffer,
    xfer: CommandBuffer,
}

impl AttnBwd {
    /// Allocate every buffer and record the three-dispatch tape.
    ///
    /// `gated` selects the attention map's gate: `qf` and `dqf` are `[T][nh·2·hd]` and `dA = dy ⊙
    /// σ(gate)` when true, `[T][nh·hd]` and `dA = dy` when false.
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value.
    ///
    /// # Complexity
    ///
    /// `O(T·nh·hd)` device words allocated; no arithmetic on the host.
    pub unsafe fn new(g: &GpuCtx, s: AttnBwdShape, gated: bool) -> Self {
        assert!(
            s.hd <= 256 && s.nh % s.nkv == 0 && s.n_ctx <= 1024,
            "attn_map_bwd geometry"
        );
        assert!(s.rot == 0 || s.rot % 2 == 0, "rope pairs");
        let (nh, nkv, hd) = (s.nh, s.nkv, s.hd);
        let qw = if gated { 2 } else { 1 };
        let t = s.t();
        let dev = |n: usize| g.dev(n * 4);
        let (qf, kf, gains) = (dev(t * nh * qw * hd), dev(t * nkv * hd), dev(2 * hd));
        let (q, gate, k, v) = (
            dev(t * nh * hd),
            dev(t * nh * hd),
            dev(t * nkv * hd),
            dev(t * nkv * hd),
        );
        let dy = dev(t * nh * hd);
        let (lse, dsum) = (dev(t * nh), dev(t * nh));
        let da = dev(t * nh * hd);
        let (dq_rot, dk_rot) = (dev(t * nh * hd), dev(t * nkv * hd));
        let (dqf, dkf, dv) = (dev(t * nh * qw * hd), dev(t * nkv * hd), dev(t * nkv * hd));
        let bufs = [
            qf, kf, gains, q, gate, k, v, dy, lse, dsum, da, dq_rot, dk_rot, dqf, dkf, dv,
        ];
        let flag = if gated { 0u32 } else { 4u32 };
        let meta = |mode: u32, wg: usize| {
            [
                mode | flag,
                s.n_ctx as u32,
                nh as u32,
                nkv as u32,
                hd as u32,
                s.rot as u32,
                s.theta.to_bits(),
                s.eps.to_bits(),
                wg as u32,
            ]
        };
        let mut steps: Vec<Step> = Vec::with_capacity(3);
        for (mode, wg, name) in [
            (0u32, s.chunks * nh * s.n_ctx, "attn_bwd_dq"),
            (1, s.chunks * nkv * s.n_ctx, "attn_bwd_dkv"),
            (2, t * (nh + nkv), "attn_bwd_prep"),
        ] {
            let mut all = vec![g.up_u32(&meta(mode, wg))];
            all.extend_from_slice(&bufs);
            let (gx, gy) = fold(wg);
            steps.push(make_step_named(g, ATTN_MAP_BWD_SPV, &all, gx, gy, name));
        }
        let stage = host_buffer(g, t * nh * 2 * hd, BUFFER_USAGE_TRANSFER_SRC);
        let rb = host_buffer(g, t * nh * 2 * hd, BUFFER_USAGE_TRANSFER_DST);
        let xfer = g.vk.reusable_command_buffer();
        let cb = g.vk.command_buffer();
        g.vk.begin(cb);
        for st in &steps {
            rec_step(g.vk, cb, st);
        }
        check((g.vk.end_cb)(cb), "vkEndCommandBuffer");
        Self {
            s,
            t,
            gated,
            qf,
            kf,
            gains,
            q,
            gate,
            k,
            v,
            dy,
            lse,
            dsum,
            da,
            dq_rot,
            dk_rot,
            dqf,
            dkf,
            dv,
            stage,
            rb,
            cb,
            xfer,
        }
    }

    /// The tape submission alone, transfers excluded.
    ///
    /// # Safety
    ///
    /// No other submission of this value's command buffers is pending.
    pub unsafe fn submit(&self, g: &GpuCtx) {
        g.vk.submit_wait(self.cb);
    }

    /// The backward at `inp`: upload, one tape submission, download the gradients.
    ///
    /// # Safety
    ///
    /// No other submission of this value's command buffers is pending.
    ///
    /// # Complexity
    ///
    /// One tape submission of `O(T·nh·n_ctx·hd)` multiply-adds (two query-key passes), and
    /// `O(T·nh·hd + T·nkv·hd)` words moved out.
    pub unsafe fn run(&self, g: &GpuCtx, inp: &AttnBwdIn<'_>) -> AttnBwdGrads {
        let AttnBwdShape { nh, nkv, hd, .. } = self.s;
        let qw = if self.gated { 2 } else { 1 };
        let t = self.t;
        assert_eq!(inp.qf.len(), t * nh * qw * hd, "qf");
        assert_eq!(inp.kf.len(), t * nkv * hd, "kf");
        assert_eq!(inp.gains.len(), 2 * hd, "gains");
        assert_eq!(inp.q.len(), t * nh * hd, "q");
        if self.gated {
            assert_eq!(inp.gate.len(), t * nh * hd, "gate");
        }
        assert_eq!(inp.k.len(), t * nkv * hd, "k");
        assert_eq!(inp.v.len(), t * nkv * hd, "v");
        assert_eq!(inp.dy.len(), t * nh * hd, "dy");
        let mut pairs: Vec<(Handle, &[f32])> = vec![
            (self.qf, inp.qf),
            (self.kf, inp.kf),
            (self.gains, inp.gains),
            (self.q, inp.q),
            (self.k, inp.k),
            (self.v, inp.v),
            (self.dy, inp.dy),
        ];
        if self.gated {
            pairs.push((self.gate, inp.gate));
        }
        for (h, d) in pairs {
            upload(g, self.xfer, &self.stage, h, d);
        }
        self.submit(g);
        AttnBwdGrads {
            lse: download(g, self.xfer, &self.rb, self.lse, t * nh),
            dsum: download(g, self.xfer, &self.rb, self.dsum, t * nh),
            dqf: download(g, self.xfer, &self.rb, self.dqf, t * nh * qw * hd),
            dkf: download(g, self.xfer, &self.rb, self.dkf, t * nkv * hd),
            dv: download(g, self.xfer, &self.rb, self.dv, t * nkv * hd),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::nn::train::mixer::ATTN_MAP_SPV;
    use crate::gpu::nn::GLSLC_ARGS;
    use std::path::Path;
    use std::time::Instant;

    fn fresh(name: &str) -> Vec<u8> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv/train");
        let out = std::env::temp_dir().join(format!("{name}_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(dir.join(format!("{name}.comp")))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let b = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        b
    }

    #[test]
    fn attn_map_bwd_spirv_is_the_glsl() {
        assert!(
            fresh("attn_map_bwd") == ATTN_MAP_BWD_SPV,
            "attn_map_bwd.spv is stale"
        );
    }

    // ---- the f64 reference: the forward of `attn_map.comp` and its adjoint ---------------------

    const FD_H: f64 = 1e-6;

    fn sigm(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// One mixer's inputs in f64: the prep's inputs, the values and the output gradient.
    #[derive(Clone)]
    struct RefIn {
        /// The attention map's gate: `qf` and `dqf` are `[T][nh·2·hd]` when true, `[T][nh·hd]`
        /// when false.
        gated: bool,
        n_ctx: usize,
        chunks: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        rot: usize,
        theta: f64,
        eps: f64,
        qf: Vec<f64>,
        kf: Vec<f64>,
        gains: Vec<f64>,
        v: Vec<f64>,
        dy: Vec<f64>,
    }

    impl RefIn {
        fn t(&self) -> usize {
            self.n_ctx * self.chunks
        }

        /// The `qf`/`dqf` word of the q half of `(row, h)` at `e`.
        fn qf_q(&self, row: usize, h: usize, e: usize) -> usize {
            if self.gated {
                row * self.nh * 2 * self.hd + h * 2 * self.hd + e
            } else {
                row * self.nh * self.hd + h * self.hd + e
            }
        }

        /// The `qf` word of the gate half of `(row, h)` at `e` (gated layout only).
        fn qf_g(&self, row: usize, h: usize, e: usize) -> usize {
            self.qf_q(row, h, e) + self.hd
        }
    }

    /// The forward's prep outputs; `v` is an input and stays in [`RefIn`].
    struct Prep {
        q: Vec<f64>,
        gate: Vec<f64>,
        k: Vec<f64>,
    }

    /// The per-row half of the prep that both the forward and its adjoint apply: the split, the RMS
    /// norm with its gain and the rope. Returns `r` and the rotated row `z`.
    fn prep_row(x: &RefIn, row: usize, slot: usize) -> (f64, Vec<f64>) {
        let (nh, nkv, hd, rot) = (x.nh, x.nkv, x.hd, x.rot);
        let half = rot / 2;
        let is_q = slot < nh;
        let goff = if is_q { 0 } else { hd };
        let mut src = vec![0.0; hd];
        for (e, s) in src.iter_mut().enumerate() {
            *s = if is_q {
                x.qf[x.qf_q(row, slot, e)]
            } else {
                x.kf[(row * nkv + slot - nh) * hd + e]
            };
        }
        let mut m = 0.0;
        for s in &src {
            m += s * s;
        }
        let r = (m / hd as f64 + x.eps).powf(-0.5);
        let y: Vec<f64> = (0..hd).map(|e| src[e] * r * x.gains[goff + e]).collect();
        let mut z = y.clone();
        let pos = (row % x.n_ctx) as f64;
        for lid in 0..rot {
            let jj = lid % half;
            let ang = pos * x.theta.powf(-2.0 * jj as f64 / rot as f64);
            let (cs, sn) = (ang.cos(), ang.sin());
            z[lid] = if lid < half {
                y[lid] * cs - y[lid + half] * sn
            } else {
                y[lid] * cs + y[lid - half] * sn
            };
        }
        (r, z)
    }

    /// `attn_map` mode 0 over every row and slot.
    fn ref_prep(x: &RefIn) -> Prep {
        let (nh, nkv, hd) = (x.nh, x.nkv, x.hd);
        let t = x.t();
        let mut q = vec![0.0; t * nh * hd];
        let mut gate = vec![0.0; t * nh * hd];
        let mut k = vec![0.0; t * nkv * hd];
        for row in 0..t {
            for slot in 0..nh + nkv {
                let (_, z) = prep_row(x, row, slot);
                if slot < nh {
                    for e in 0..hd {
                        q[(row * nh + slot) * hd + e] = z[e];
                    }
                    if x.gated {
                        for e in 0..hd {
                            gate[(row * nh + slot) * hd + e] = x.qf[x.qf_g(row, slot, e)];
                        }
                    }
                } else {
                    for e in 0..hd {
                        k[(row * nkv + slot - nh) * hd + e] = z[e];
                    }
                }
            }
        }
        Prep { q, gate, k }
    }

    /// The causal row softmax of one `(chunk, head, query i)` in place: `s_j = p_j·tot`; returns
    /// `(tot, max)`.
    fn softmax_row(
        p: &Prep,
        x: &RefIn,
        base: usize,
        h: usize,
        g: usize,
        i: usize,
        s: &mut [f64],
    ) -> (f64, f64) {
        let (nh, nkv, hd) = (x.nh, x.nkv, x.hd);
        let scale = 1.0 / (hd as f64).sqrt();
        let row = base + i;
        let qi = &p.q[(row * nh + h) * hd..][..hd];
        for (j, sj) in s.iter_mut().enumerate().take(i + 1) {
            *sj = dot(qi, &p.k[((base + j) * nkv + g) * hd..][..hd]) * scale;
        }
        let mx = s[..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut tot = 0.0;
        for sj in s[..=i].iter_mut() {
            *sj = (*sj - mx).exp();
            tot += *sj;
        }
        (tot, mx)
    }

    /// `attn_map` mode 1: the causal softmax mix of the values, times the gate when gated.
    fn ref_forward(x: &RefIn, p: &Prep) -> Vec<f64> {
        let (nh, nkv, hd, nc) = (x.nh, x.nkv, x.hd, x.n_ctx);
        let rep = nh / nkv;
        let t = x.t();
        let mut out = vec![0.0; t * nh * hd];
        let mut s = vec![0.0; nc];
        for chunk in 0..x.chunks {
            let base = chunk * nc;
            for h in 0..nh {
                let g = h / rep;
                for i in 0..nc {
                    let row = base + i;
                    let tot = softmax_row(p, x, base, h, g, i, &mut s).0;
                    for e in 0..hd {
                        let mut acc = 0.0;
                        for j in 0..=i {
                            acc += s[j] / tot * x.v[((base + j) * nkv + g) * hd + e];
                        }
                        let gt = if x.gated {
                            sigm(p.gate[(row * nh + h) * hd + e])
                        } else {
                            1.0
                        };
                        out[(row * nh + h) * hd + e] = acc * gt;
                    }
                }
            }
        }
        out
    }

    /// The adjoint: the FlashAttention-2 reverse of `ref_forward`, then the prep adjoint.
    struct RefGrads {
        lse: Vec<f64>,
        dsum: Vec<f64>,
        dqf: Vec<f64>,
        dkf: Vec<f64>,
        dv: Vec<f64>,
    }

    fn ref_backward(x: &RefIn, p: &Prep) -> RefGrads {
        let (nh, nkv, hd, nc, rot) = (x.nh, x.nkv, x.hd, x.n_ctx, x.rot);
        let half = rot / 2;
        let scale = 1.0 / (hd as f64).sqrt();
        let rep = nh / nkv;
        let t = x.t();
        let mut dqr = vec![0.0; t * nh * hd];
        let mut dkr = vec![0.0; t * nkv * hd];
        let mut dv = vec![0.0; t * nkv * hd];
        let qw = if x.gated { 2 } else { 1 };
        let mut dqf = vec![0.0; t * nh * qw * hd];
        let mut lse = vec![0.0; t * nh];
        let mut dsum = vec![0.0; t * nh];
        let mut s = vec![0.0; nc];
        for chunk in 0..x.chunks {
            let base = chunk * nc;
            for h in 0..nh {
                let g = h / rep;
                for i in 0..nc {
                    let row = base + i;
                    let (tot, mx) = softmax_row(p, x, base, h, g, i, &mut s);
                    let mut da = vec![0.0; hd];
                    let mut a = vec![0.0; hd];
                    for e in 0..hd {
                        let o = (row * nh + h) * hd + e;
                        let gt = if x.gated { sigm(p.gate[o]) } else { 1.0 };
                        da[e] = x.dy[o] * gt;
                        for j in 0..=i {
                            a[e] += s[j] / tot * x.v[((base + j) * nkv + g) * hd + e];
                        }
                        if x.gated {
                            dqf[x.qf_g(row, h, e)] = x.dy[o] * a[e] * gt * (1.0 - gt);
                        }
                    }
                    let dval = dot(&da, &a);
                    lse[row * nh + h] = mx + tot.ln();
                    dsum[row * nh + h] = dval;
                    for j in 0..=i {
                        let pj = s[j] / tot;
                        let vj = &x.v[((base + j) * nkv + g) * hd..][..hd];
                        let ds = pj * (dot(&da, vj) - dval);
                        for e in 0..hd {
                            dqr[(row * nh + h) * hd + e] +=
                                ds * p.k[((base + j) * nkv + g) * hd + e];
                            dkr[((base + j) * nkv + g) * hd + e] +=
                                ds * p.q[(row * nh + h) * hd + e];
                            dv[((base + j) * nkv + g) * hd + e] += pj * da[e];
                        }
                    }
                }
            }
        }
        for v in dqr.iter_mut() {
            *v *= scale;
        }
        for v in dkr.iter_mut() {
            *v *= scale;
        }
        let mut dkf = vec![0.0; t * nkv * hd];
        for row in 0..t {
            for slot in 0..nh + nkv {
                let is_q = slot < nh;
                let goff = if is_q { 0 } else { hd };
                let (r, _) = prep_row(x, row, slot);
                let src: Vec<f64> = (0..hd)
                    .map(|e| {
                        if is_q {
                            x.qf[x.qf_q(row, slot, e)]
                        } else {
                            x.kf[(row * nkv + slot - nh) * hd + e]
                        }
                    })
                    .collect();
                let xh: Vec<f64> = src.iter().map(|s| s * r).collect();
                let d = if is_q {
                    dqr[(row * nh + slot) * hd..][..hd].to_vec()
                } else {
                    dkr[(row * nkv + slot - nh) * hd..][..hd].to_vec()
                };
                // The rope transpose: the pre-rotation gradient of dims < rot.
                let mut dyt = d.clone();
                let pos = (row % x.n_ctx) as f64;
                for lid in 0..rot {
                    let jj = lid % half;
                    let ang = pos * x.theta.powf(-2.0 * jj as f64 / rot as f64);
                    let (cs, sn) = (ang.cos(), ang.sin());
                    dyt[lid] = if lid < half {
                        d[lid] * cs + d[lid + half] * sn
                    } else {
                        d[lid] * cs - d[lid - half] * sn
                    };
                }
                let dxhat: Vec<f64> = (0..hd).map(|e| x.gains[goff + e] * dyt[e]).collect();
                let sdot: f64 = (0..hd).map(|e| dxhat[e] * xh[e]).sum();
                for e in 0..hd {
                    let gr = r * (dxhat[e] - xh[e] * sdot / hd as f64);
                    if is_q {
                        dqf[x.qf_q(row, slot, e)] = gr;
                    } else {
                        dkf[(row * nkv + slot - nh) * hd + e] = gr;
                    }
                }
            }
        }
        RefGrads {
            lse,
            dsum,
            dqf,
            dkf,
            dv,
        }
    }

    /// A deterministic uniform generator: no dependency, bounded values, smooth at every knee.
    struct Rng(u64);

    impl Rng {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
        }

        fn r(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }

        /// A normal sample by the sum of twelve uniforms, bounded at `±6σ`.
        fn n(&mut self) -> f64 {
            let mut s = 0.0;
            for _ in 0..12 {
                s += self.unit();
            }
            s - 6.0
        }
    }

    fn to32(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    fn to64(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x)).collect()
    }

    /// `N(0,1)` rows and output gradient, gains in `(0.5, 1.5)`, every sample already an f32.
    fn make_case(s: AttnBwdShape, seed: u64, gated: bool) -> RefIn {
        let (nh, nkv, hd) = (s.nh, s.nkv, s.hd);
        let qw = if gated { 2 } else { 1 };
        let t = s.t();
        let mut r = Rng(seed);
        let fill =
            |n: usize, r: &mut Rng| -> Vec<f64> { (0..n).map(|_| r.n() as f32 as f64).collect() };
        let qf = fill(t * nh * qw * hd, &mut r);
        let kf = fill(t * nkv * hd, &mut r);
        let v = fill(t * nkv * hd, &mut r);
        let dy = fill(t * nh * hd, &mut r);
        let gains = (0..2 * hd).map(|_| r.r(0.5, 1.5) as f32 as f64).collect();
        RefIn {
            gated,
            n_ctx: s.n_ctx,
            chunks: s.chunks,
            nh,
            nkv,
            hd,
            rot: s.rot,
            theta: f64::from(s.theta),
            eps: f64::from(s.eps),
            qf,
            kf,
            gains,
            v,
            dy,
        }
    }

    /// `max|got − want| / max|want|`.
    fn rel(got: &[f32], want: &[f64]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 1e-30f64;
        for (&p, &q) in got.iter().zip(want) {
            num = num.max((f64::from(p) - q).abs());
            den = den.max(q.abs());
        }
        num / den
    }

    /// The `f` half (0 = q, 1 = gate) of a `[T][nh·2·hd]` array as a contiguous `[T·nh·hd]` copy.
    fn half(a: &[f64], t: usize, nh: usize, hd: usize, f: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(t * nh * hd);
        for row in 0..t {
            for h in 0..nh {
                let b = row * nh * 2 * hd + h * 2 * hd + f * hd;
                out.extend_from_slice(&a[b..b + hd]);
            }
        }
        out
    }

    /// The `qf` index of the `f` half (0 = q, 1 = gate) sample `p` of a `[T·nh·hd]` half.
    fn qf_index(nh: usize, hd: usize, f: usize, p: usize) -> usize {
        let (row, rest) = (p / (nh * hd), p % (nh * hd));
        row * nh * 2 * hd + (rest / hd) * 2 * hd + f * hd + rest % hd
    }

    /// Central finite differences on the f64 forward against the f64 backward, sampled coordinates
    /// of `qf` (both halves), `kf` and `v`.
    #[test]
    fn attn_bwd_ref_matches_fd() {
        let s = AttnBwdShape {
            n_ctx: 16,
            chunks: 2,
            nh: 4,
            nkv: 2,
            hd: 32,
            rot: 8,
            theta: 1e4,
            eps: 1e-6,
        };
        let x = make_case(s, 0x5eed_1234, true);
        let gr = ref_backward(&x, &ref_prep(&x));
        // `w` is the output cotangent `dy`: `L = Σ out ⊙ dy`, so the adjoint is the gradient of
        // exactly this scalar, and `dy` must be the same vector on both sides.
        let wts: Vec<f64> = x.dy.clone();
        let loss = |y: &RefIn| -> f64 {
            let p = ref_prep(y);
            ref_forward(y, &p)
                .iter()
                .zip(&wts)
                .map(|(a, b)| a * b)
                .sum()
        };
        let fd = |name: &str,
                  analytic: &[f64],
                  n: usize,
                  pert: &dyn Fn(&RefIn, usize, f64) -> RefIn|
         -> f64 {
            let (mut num, mut den, mut k) = (0.0f64, 1e-30f64, 0usize);
            let (mut wi, mut wd, mut wa) = (0usize, 0.0f64, 0.0f64);
            let step = (n / 16).max(1);
            let mut i = 0;
            while i < n {
                let d = (loss(&pert(&x, i, FD_H)) - loss(&pert(&x, i, -FD_H))) / (2.0 * FD_H);
                let e = (d - analytic[i]).abs();
                if e > num {
                    num = e;
                    wi = i;
                    wd = d;
                    wa = analytic[i];
                }
                den = den.max(d.abs());
                k += 1;
                i += step;
            }
            let r = num / den;
            println!(
                "FD {name}: rel {r:.3e} over {k} of {n} coordinates; worst {wi}: fd {wd:.6e} vs {wa:.6e}"
            );
            r
        };
        let mut rels = Vec::new();
        rels.push(fd(
            "qf q",
            &half(&gr.dqf, x.t(), x.nh, x.hd, 0),
            x.qf.len() / 2,
            &|y, p, d| {
                let mut z = y.clone();
                z.qf[qf_index(x.nh, x.hd, 0, p)] += d;
                z
            },
        ));
        rels.push(fd(
            "qf gate",
            &half(&gr.dqf, x.t(), x.nh, x.hd, 1),
            x.qf.len() / 2,
            &|y, p, d| {
                let mut z = y.clone();
                z.qf[qf_index(x.nh, x.hd, 1, p)] += d;
                z
            },
        ));
        rels.push(fd("kf", &gr.dkf, x.kf.len(), &|y, i, d| {
            let mut z = y.clone();
            z.kf[i] += d;
            z
        }));
        rels.push(fd("v", &gr.dv, x.v.len(), &|y, i, d| {
            let mut z = y.clone();
            z.v[i] += d;
            z
        }));
        for (n, r) in ["qf q", "qf gate", "kf", "v"].iter().zip(&rels) {
            assert!(r <= &1e-6, "FD {n}: rel {r:.3e}");
        }
    }

    /// The `f` half of a `[T][nh·2·hd]` f32 array as a contiguous `[T·nh·hd]` copy.
    fn half32(a: &[f32], t: usize, nh: usize, hd: usize, f: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(t * nh * hd);
        for row in 0..t {
            for h in 0..nh {
                let b = row * nh * 2 * hd + h * 2 * hd + f * hd;
                out.extend_from_slice(&a[b..b + hd]);
            }
        }
        out
    }

    /// The backward's eight inputs as f32 words, in [`AttnBwdIn`] order.
    fn device_inputs(x: &RefIn, p: &Prep) -> [Vec<f32>; 8] {
        [
            to32(&x.qf),
            to32(&x.kf),
            to32(&x.gains),
            to32(&p.q),
            to32(&p.gate),
            to32(&p.k),
            to32(&x.v),
            to32(&x.dy),
        ]
    }

    /// The eight uploaded words as a borrowed [`AttnBwdIn`].
    fn as_in(d: &[Vec<f32>; 8]) -> AttnBwdIn<'_> {
        AttnBwdIn {
            qf: &d[0],
            kf: &d[1],
            gains: &d[2],
            q: &d[3],
            gate: &d[4],
            k: &d[5],
            v: &d[6],
            dy: &d[7],
        }
    }

    /// The device against the f64 reference at the model geometry: `nh 8, nkv 2, hd 256, rot 64`,
    /// `theta 1e7`, `n_ctx 512`, four chunks (`T = 2048`).
    #[test]
    fn attn_bwd_device_matches_ref() {
        let s = AttnBwdShape {
            n_ctx: 512,
            chunks: 4,
            nh: 8,
            nkv: 2,
            hd: 256,
            rot: 64,
            theta: 1e7,
            eps: 1e-6,
        };
        let x = make_case(s, 0x1234_5678, true);
        let p = ref_prep(&x);
        let d = device_inputs(&x, &p);
        let inp = as_in(&d);
        let vk = unsafe { Vk::load() }.expect("vulkan");
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let bwd = unsafe { AttnBwd::new(&g, s, true) };
        let t0 = Instant::now();
        let got = unsafe { bwd.run(&g, &inp) };
        let dev_s = t0.elapsed().as_secs_f64();
        // The reference reads the same f32 words the device reads, so the row compares the
        // backward's arithmetic and not the prep's rounding.
        let p32 = Prep {
            q: to64(&inp.q),
            gate: to64(&inp.gate),
            k: to64(&inp.k),
        };
        let want = ref_backward(&x, &p32);
        let (t, nh, hd) = (s.t(), s.nh, s.hd);
        let rows = [
            ("lse", rel(&got.lse, &want.lse)),
            ("dsum", rel(&got.dsum, &want.dsum)),
            (
                "dqf q",
                rel(
                    &half32(&got.dqf, t, nh, hd, 0),
                    &half(&want.dqf, t, nh, hd, 0),
                ),
            ),
            (
                "dqf gate",
                rel(
                    &half32(&got.dqf, t, nh, hd, 1),
                    &half(&want.dqf, t, nh, hd, 1),
                ),
            ),
            ("dkf", rel(&got.dkf, &want.dkf)),
            ("dv", rel(&got.dv, &want.dv)),
        ];
        let line: Vec<String> = rows.iter().map(|(n, r)| format!("{n} {r:.3e}")).collect();
        println!("AB T={t} dev {dev_s:.4} s: {}", line.join(" "));
        for (n, r) in rows {
            assert!(
                r <= 1e-3 && r.is_finite(),
                "{n}: rel {r:.3e} > 1e-3 at T={t}"
            );
        }
    }

    /// The device backward wall against the `attn_map` forward wall at `T = 2048`, paired: both are
    /// tape submissions on the same buffers, transfers excluded, `n = 5`.
    #[test]
    #[ignore = "needs a Vulkan device; reports the paired walls"]
    fn attn_bwd_timing() {
        let s = AttnBwdShape {
            n_ctx: 512,
            chunks: 4,
            nh: 8,
            nkv: 2,
            hd: 256,
            rot: 64,
            theta: 1e7,
            eps: 1e-6,
        };
        let x = make_case(s, 0xabcd_0f0f, true);
        let p = ref_prep(&x);
        let d = device_inputs(&x, &p);
        let inp = as_in(&d);
        let vk = unsafe { Vk::load() }.expect("vulkan");
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let bwd = unsafe { AttnBwd::new(&g, s, true) };
        let _ = unsafe { bwd.run(&g, &inp) };
        let (t, nh, nkv, hd) = (s.t(), s.nh, s.nkv, s.hd);
        let out = unsafe { g.dev(t * nh * hd * 4) };
        let bufs = [
            bwd.qf, bwd.kf, bwd.gains, bwd.q, bwd.gate, bwd.k, bwd.v, out,
        ];
        let mut fsteps: Vec<Step> = Vec::with_capacity(2);
        for (mode, wg, name) in [
            (0u32, t * (nh + nkv), "attn_prep"),
            (1, s.chunks * nh * s.n_ctx, "attn_attend"),
        ] {
            let meta = [
                mode,
                s.n_ctx as u32,
                nh as u32,
                nkv as u32,
                hd as u32,
                s.rot as u32,
                s.theta.to_bits(),
                s.eps.to_bits(),
                wg as u32,
            ];
            let mut all = vec![unsafe { g.up_u32(&meta) }];
            all.extend_from_slice(&bufs);
            let (gx, gy) = fold(wg);
            fsteps.push(unsafe { make_step_named(&g, ATTN_MAP_SPV, &all, gx, gy, name) });
        }
        let fcb = unsafe { g.vk.command_buffer() };
        unsafe {
            g.vk.begin(fcb);
            for st in &fsteps {
                rec_step(g.vk, fcb, st);
            }
            check((g.vk.end_cb)(fcb), "vkEndCommandBuffer");
        }
        // Warm both, then pair: no transfer is timed on either arm.
        unsafe { g.vk.submit_wait(fcb) };
        unsafe { bwd.submit(&g) };
        let (mut fw, mut bw) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            let t0 = Instant::now();
            unsafe { g.vk.submit_wait(fcb) };
            fw.push(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            unsafe { bwd.submit(&g) };
            bw.push(t1.elapsed().as_secs_f64());
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let spread = |v: &[f64]| {
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            for &p in v {
                lo = lo.min(p);
                hi = hi.max(p);
            }
            (lo, hi)
        };
        let (fl, fh) = spread(&fw);
        let (bl, bh) = spread(&bw);
        println!(
            "TB T={} fwd {:.4} ms [{fl:.4}, {fh:.4}] bwd {:.4} ms [{bl:.4}, {bh:.4}] x{:.2} (n=5)",
            s.t(),
            mean(&fw) * 1e3,
            mean(&bw) * 1e3,
            mean(&bw) / mean(&fw)
        );
        assert!(
            bw.iter().all(|&p| p.is_finite()) && mean(&bw) > 0.0,
            "bwd wall"
        );
    }

    // ---- the ungated map (Qwen3: no output gate) and the gated dump ----------------------------

    /// The forward's device words: the output and the prep's `q`, `gate` and `k`.
    struct Fwd {
        out: Vec<f32>,
        q: Vec<f32>,
        gate: Vec<f32>,
        k: Vec<f32>,
    }

    impl Fwd {
        /// The same words as the f64 reference reads, so a comparison isolates one stage's
        /// arithmetic and not the prep's rounding.
        fn prep(&self) -> Prep {
            Prep {
                q: to64(&self.q),
                gate: to64(&self.gate),
                k: to64(&self.k),
            }
        }
    }

    /// `attn_map` mode 0 (prep) then mode 1 (attend) with `x.gated` in the mode word, transfers
    /// included.
    fn device_forward(g: &GpuCtx, s: AttnBwdShape, x: &RefIn) -> Fwd {
        let (nh, nkv, hd) = (s.nh, s.nkv, s.hd);
        let qw = if x.gated { 2 } else { 1 };
        let t = s.t();
        let dev = |n: usize| unsafe { g.dev(n * 4) };
        let (qf, kf, gains) = (dev(t * nh * qw * hd), dev(t * nkv * hd), dev(2 * hd));
        let (q, gate, k, v) = (
            dev(t * nh * hd),
            dev(t * nh * hd),
            dev(t * nkv * hd),
            dev(t * nkv * hd),
        );
        let out = dev(t * nh * hd);
        let cap = t * nh * 2 * hd;
        let stage = unsafe { host_buffer(g, cap, BUFFER_USAGE_TRANSFER_SRC) };
        let rb = unsafe { host_buffer(g, cap, BUFFER_USAGE_TRANSFER_DST) };
        let xfer = unsafe { g.vk.reusable_command_buffer() };
        unsafe {
            upload(g, xfer, &stage, qf, &to32(&x.qf));
            upload(g, xfer, &stage, kf, &to32(&x.kf));
            upload(g, xfer, &stage, gains, &to32(&x.gains));
            upload(g, xfer, &stage, v, &to32(&x.v));
        }
        let bufs = [qf, kf, gains, q, gate, k, v, out];
        let flag = if x.gated { 0u32 } else { 4u32 };
        let mut steps = Vec::new();
        for (mode, wg, name) in [
            (0u32, t * (nh + nkv), "attn_prep"),
            (1, s.chunks * nh * s.n_ctx, "attn_attend"),
        ] {
            let meta = [
                mode | flag,
                s.n_ctx as u32,
                nh as u32,
                nkv as u32,
                hd as u32,
                s.rot as u32,
                s.theta.to_bits(),
                s.eps.to_bits(),
                wg as u32,
            ];
            let mut all = vec![unsafe { g.up_u32(&meta) }];
            all.extend_from_slice(&bufs);
            let (gx, gy) = fold(wg);
            steps.push(unsafe { make_step_named(g, ATTN_MAP_SPV, &all, gx, gy, name) });
        }
        let cb = unsafe { g.vk.command_buffer() };
        unsafe {
            g.vk.begin(cb);
            for st in &steps {
                rec_step(g.vk, cb, st);
            }
            check((g.vk.end_cb)(cb), "vkEndCommandBuffer");
            g.vk.submit_wait(cb);
        }
        Fwd {
            out: unsafe { download(g, xfer, &rb, out, t * nh * hd) },
            q: unsafe { download(g, xfer, &rb, q, t * nh * hd) },
            gate: unsafe { download(g, xfer, &rb, gate, t * nh * hd) },
            k: unsafe { download(g, xfer, &rb, k, t * nkv * hd) },
        }
    }

    /// The ungated reference backward against central finite differences: `qf` is `[T][nh·hd]`, so
    /// sample `p` is word `p` and the gradient has no gate half.
    #[test]
    fn attn_bwd_ref_matches_fd_ungated() {
        let s = AttnBwdShape {
            n_ctx: 16,
            chunks: 2,
            nh: 4,
            nkv: 2,
            hd: 32,
            rot: 8,
            theta: 1e4,
            eps: 1e-6,
        };
        let x = make_case(s, 0x5eed_1234, false);
        let gr = ref_backward(&x, &ref_prep(&x));
        let wts: Vec<f64> = x.dy.clone();
        let loss = |y: &RefIn| -> f64 {
            let p = ref_prep(y);
            ref_forward(y, &p)
                .iter()
                .zip(&wts)
                .map(|(a, b)| a * b)
                .sum()
        };
        let fd = |name: &str,
                  analytic: &[f64],
                  n: usize,
                  pert: &dyn Fn(&RefIn, usize, f64) -> RefIn|
         -> f64 {
            let (mut num, mut den, mut k) = (0.0f64, 1e-30f64, 0usize);
            let (mut wi, mut wd, mut wa) = (0usize, 0.0f64, 0.0f64);
            let step = (n / 16).max(1);
            let mut i = 0;
            while i < n {
                let d = (loss(&pert(&x, i, FD_H)) - loss(&pert(&x, i, -FD_H))) / (2.0 * FD_H);
                let e = (d - analytic[i]).abs();
                if e > num {
                    num = e;
                    wi = i;
                    wd = d;
                    wa = analytic[i];
                }
                den = den.max(d.abs());
                k += 1;
                i += step;
            }
            let r = num / den;
            println!(
                "FD {name}: rel {r:.3e} over {k} of {n} coordinates; worst {wi}: fd {wd:.6e} vs {wa:.6e}"
            );
            r
        };
        let rels = [
            fd("qf", &gr.dqf, x.qf.len(), &|y, i, d| {
                let mut z = y.clone();
                z.qf[i] += d;
                z
            }),
            fd("kf", &gr.dkf, x.kf.len(), &|y, i, d| {
                let mut z = y.clone();
                z.kf[i] += d;
                z
            }),
            fd("v", &gr.dv, x.v.len(), &|y, i, d| {
                let mut z = y.clone();
                z.v[i] += d;
                z
            }),
        ];
        for (n, r) in ["qf", "kf", "v"].iter().zip(&rels) {
            assert!(r <= &1e-6, "FD {n}: rel {r:.3e}");
        }
    }

    /// The ungated device forward and backward against the f64 reference at the Qwen3-0.6B
    /// geometry: `nh 16, nkv 8, hd 128`, full rotary, `theta 1e6`, `n_ctx 512`, four chunks
    /// (`T = 2048`).
    #[test]
    fn attn_bwd_device_matches_ref_ungated() {
        let s = AttnBwdShape {
            n_ctx: 512,
            chunks: 4,
            nh: 16,
            nkv: 8,
            hd: 128,
            rot: 128,
            theta: 1e6,
            eps: 1e-6,
        };
        let x = make_case(s, 0x5150_2a2a, false);
        let vk = unsafe { Vk::load() }.expect("vulkan");
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let t0 = Instant::now();
        let f = device_forward(&g, s, &x);
        let dev_s = t0.elapsed().as_secs_f64();
        let p32 = f.prep();
        let want_out = ref_forward(&x, &p32);
        let d = device_inputs(&x, &p32);
        let inp = as_in(&d);
        let bwd = unsafe { AttnBwd::new(&g, s, false) };
        let got = unsafe { bwd.run(&g, &inp) };
        let want = ref_backward(&x, &p32);
        let t = s.t();
        let rows = [
            ("out", rel(&f.out, &want_out)),
            ("dqf", rel(&got.dqf, &want.dqf)),
            ("dkf", rel(&got.dkf, &want.dkf)),
            ("dv", rel(&got.dv, &want.dv)),
        ];
        let line: Vec<String> = rows.iter().map(|(n, r)| format!("{n} {r:.3e}")).collect();
        println!("AU T={t} dev {dev_s:.4} s: {}", line.join(" "));
        for (n, r) in rows {
            assert!(
                r <= 1e-3 && r.is_finite(),
                "{n}: rel {r:.3e} > 1e-3 at T={t}"
            );
        }
    }

    /// Gated bit identity: the forward output and `dqf`, `dkf`, `dv` at the Qwen3.5 geometry, to the
    /// file named by `APEIRON_ATTN_DUMP`, so a build with the base `.spv` files `cmp`s identical.
    #[test]
    #[ignore = "writes a gradient dump named by APEIRON_ATTN_DUMP for a cross-build cmp"]
    fn attn_bwd_gated_dump() {
        let Ok(path) = std::env::var("APEIRON_ATTN_DUMP") else {
            return;
        };
        let s = AttnBwdShape {
            n_ctx: 512,
            chunks: 4,
            nh: 8,
            nkv: 2,
            hd: 256,
            rot: 64,
            theta: 1e7,
            eps: 1e-6,
        };
        let x = make_case(s, 0x1234_5678, true);
        let vk = unsafe { Vk::load() }.expect("vulkan");
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let f = device_forward(&g, s, &x);
        let d = device_inputs(&x, &f.prep());
        let inp = as_in(&d);
        let bwd = unsafe { AttnBwd::new(&g, s, true) };
        let got = unsafe { bwd.run(&g, &inp) };
        let mut b = Vec::new();
        for v in f.out.iter().chain(&got.dqf).chain(&got.dkf).chain(&got.dv) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &b).expect("write the dump");
        println!("DUMP {path} {} bytes", b.len());
    }
}
