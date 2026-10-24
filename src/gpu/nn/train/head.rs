//! The KD loss head at the full Qwen3 vocabulary, forward and backward, on the device: the top of
//! the 28-layer KD chain (`SamsungLabs/NanoQuant` @ `a9e0a43`,
//! `src/nanoquant/core/compress_model.py:160-181` and `:262-265`).
//!
//! With `d = hidden_size`, `v = vocab_size`, `T` rows and the final norm gain `g_final` frozen,
//! the step is
//!
//! ```text
//! hf_s = rmsnorm(h_s, g_final)        hf_t = rmsnorm(h_t, g_final)
//! z_s  = hf_s·W_embᵀ                  z_t  = hf_t·W_embᵀ            [T, v]
//! loss[t] = −Σ_v p_t[t,v]·log p_s[t,v]                              p = softmax(z), T = 1
//! dz[t,v] = (p_s[t,v] − p_t[t,v]) / T
//! dhf  = dz·W_emb                                                   [T, d]
//! dh   = rmsnorm_bwd(h_s, g_final, dhf)                             [T, d]
//! ```
//!
//! `loss` is the forward cross-entropy `H(p_t, p_s)`, not the KL: it carries the teacher entropy
//! (the reference's epoch-1 mean is 2.8497). The mask is all ones for Qwen3-0.6B-Base, which has
//! no pad id, so `Σ mask = T`; the reference's `+1e-8` in the denominator is dropped, because 2048
//! added to an f32 sum of order 1 is below the ulp. `W_emb` and `g_final` are frozen: the head
//! produces `dh` and nothing else flows back.
//!
//! The vocabulary is tiled: `rc` rows at a time, so the two `rc × v` logit tiles fit the weight
//! budget and only one chunk's logits exist on the device. `dz` is written in place over `z_s`.
//! Everything inside a step is device-to-device; the one host read is [`KdHead::loss`].
//!
//! Logits and softmaxes are f32; the reference computes them in bf16 (HF Qwen3 under
//! `model.to(bfloat16)`), so f32 is the more exact side of that comparison.
//!
//! Consumer: the KD driver that feeds layer 27's block adjoint (lane L4).

use crate::gpu::decode::{rec_step, GpuCtx, Step};
use crate::gpu::nn::fwd::{ce_row_dispatch, linear_fwd_dispatch, rmsnorm_fwd_dispatch};
use crate::gpu::nn::{linear_bwd_input_dispatch, rmsnorm_bwd_dispatch};
use crate::gpu::vulkan::*;

use super::block::stream_download;

/// The training contract's final-norm epsilon (`config.json` `rms_norm_eps`), as
/// [`rmsnorm_fwd_dispatch`] bakes it.
const EPS: f32 = 1e-6;

/// The device budget for the two `rc × v` f32 logit tiles.
const LOGIT_TILE_BUDGET: u64 = 640 << 20;

/// The default rows per chunk: `2·512·151936·4` bytes is 622 MB, under [`LOGIT_TILE_BUDGET`].
const DEFAULT_RC: u32 = 512;

/// One row chunk's dispatches; `start` is its first row in the `[T, d]` buffers.
struct Chunk {
    start: u32,
    rows: u32,
    fwd_s: Step,
    fwd_t: Step,
    ce: Step,
    bwd: Step,
}

/// The KD loss head: the frozen output embedding and final-norm gain, the per-chunk logit
/// buffers, and the recorded tape that turns `h_s`/`h_t` into `dh`.
/// Hold only what is read after the tape is recorded: the geometry, the two output handles the
/// caller consumes, the surviving cotangent tile, and the recorded command buffer. Every other
/// buffer of the step is allocated inside [`KdHead::new`] and kept alive by the device.
pub struct KdHead {
    t: u32,
    d: u32,
    v: u32,
    rc: u32,
    z_s: Handle,
    dh: Handle,
    loss: Handle,
    cb: CommandBuffer,
    /// The tape's own handles and dispatches: kept, not dropped, so the tape can be re-recorded
    /// with the profiler's timestamps ([`KdHead::rerecord`]) in the order [`KdHead::new`] recorded
    /// it. The buffers are held alive by the device; these handles are their tape-side addresses.
    tape: HeadTape,
}

/// What the head's tape reads: the two whole-`T` norm passes, the norm adjoint, and per chunk its
/// buffers and four dispatches. Only the recording touches these; the consumer reads `dh` and
/// `loss`.
struct HeadTape {
    hf_s: Handle,
    hf_t: Handle,
    hf_s_c: Handle,
    hf_t_c: Handle,
    dhf_c: Handle,
    dhf: Handle,
    loss_c: Handle,
    norm_s: Step,
    norm_t: Step,
    norm_bwd: Step,
    chunks: Vec<Chunk>,
}

impl KdHead {
    /// Allocate the head's work buffers and build its tape over `t` rows, model width `d` and
    /// vocabulary `v`, with `rc` rows per logit chunk (`0` selects [`DEFAULT_RC`], clamped to `t`).
    ///
    /// `w_emb` is the device buffer of `W_emb` `[v, d]` f32, `g_final` of `[d]` f32, `h_s` and
    /// `h_t` of `[t, d]` f32 (student and teacher residual at the final norm input). All four are
    /// caller-owned and must outlive the value.
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value; every handle holds its declared
    /// shape in f32; no command buffer is pending.
    ///
    /// # Complexity
    ///
    /// Five pipelines per chunk (two forward products, the loss head, the input adjoint) plus the
    /// two whole-`T` norms and the norm adjoint; `O(t·(d + 1) + rc·(d + v))` device words held
    /// (the `v·d` embedding and `d`-word gain are the caller's). The tape is recorded once:
    /// `3` whole-`T` dispatches, `4·⌈t/rc⌉` chunk dispatches, and `t·d + ⌈t/rc⌉·(2·rc·d + rc)`
    /// words copied device-to-device.
    ///
    /// # Panics
    ///
    /// If a dimension is zero, or the two `rc × v` logit tiles exceed [`LOGIT_TILE_BUDGET`].
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "one head, one tape; the dimensions index u32 dispatch shapes"
    )]
    pub unsafe fn new(
        g: &GpuCtx,
        t: u32,
        d: u32,
        v: u32,
        rc: u32,
        w_emb: Handle,
        g_final: Handle,
        h_s: Handle,
        h_t: Handle,
    ) -> Self {
        assert!(
            t > 0 && d > 0 && v > 0,
            "kd_head: dimensions must be non-zero"
        );
        let rc = if rc == 0 { DEFAULT_RC } else { rc };
        assert!(rc > 0, "kd_head: rows per chunk must be non-zero");
        let rc = rc.min(t);
        assert!(
            2 * u64::from(rc) * u64::from(v) * 4 <= LOGIT_TILE_BUDGET,
            "kd_head: two {rc}×{v} f32 logit tiles exceed the {LOGIT_TILE_BUDGET}-byte budget"
        );
        let dev = |n: u32| g.dev(n as usize * 4);
        let (hf_s, hf_t) = (dev(t * d), dev(t * d));
        let (hf_s_c, hf_t_c) = (dev(rc * d), dev(rc * d));
        let (z_s, z_t) = (dev(rc * v), dev(rc * v));
        let (dhf_c, dhf) = (dev(rc * d), dev(t * d));
        let (dh, dgr) = (dev(t * d), dev(t * d));
        let (loss_c, loss) = (dev(rc), dev(t));
        let inv_t = 1.0f32 / t as f32;
        let norm_s = rmsnorm_fwd_dispatch(g, h_s, g_final, hf_s, t, d);
        let norm_t = rmsnorm_fwd_dispatch(g, h_t, g_final, hf_t, t, d);
        let norm_bwd = rmsnorm_bwd_dispatch(g, h_s, g_final, dhf, dh, dgr, t, d, EPS);
        let mut chunks = Vec::with_capacity(t.div_ceil(rc) as usize);
        let mut start = 0u32;
        while start < t {
            let rows = rc.min(t - start);
            chunks.push(Chunk {
                start,
                rows,
                fwd_s: linear_fwd_dispatch(g, hf_s_c, w_emb, z_s, rows, d, v),
                fwd_t: linear_fwd_dispatch(g, hf_t_c, w_emb, z_t, rows, d, v),
                ce: ce_row_dispatch(g, z_s, z_t, loss_c, z_s, rows, v, inv_t),
                bwd: linear_bwd_input_dispatch(g, z_s, w_emb, dhf_c, rows, d, v),
            });
            start += rows;
        }
        // ── the tape: the norms, then per chunk (row copy down, two products, the loss head, the
        // input adjoint, the loss rows and the cotangent rows copied out), then the norm adjoint ──
        let cb = g.vk.reusable_command_buffer();
        let mut chunks = chunks;
        chunks.shrink_to_fit();
        let s = Self {
            t,
            d,
            v,
            rc,
            z_s,
            dh,
            loss,
            cb,
            // `hf_s`, `hf_t`, `hf_s_c`, `hf_t_c`, `dhf_c` and `loss_c` are reachable only from the
            // recorded tape; the tape struct holds the handles that keep their buffers' addresses.
            tape: HeadTape {
                hf_s,
                hf_t,
                hf_s_c,
                hf_t_c,
                dhf_c,
                dhf,
                loss_c,
                norm_s,
                norm_t,
                norm_bwd,
                chunks,
            },
        };
        s.record(g);
        s
    }

    /// Record the head's tape into [`KdHead::cb`]: the two whole-`T` norm passes, then per chunk
    /// (row copy down, the two logit products, the loss head, the input adjoint, the loss rows and
    /// the cotangent rows copied out), then the norm adjoint. The buffer is reset first.
    ///
    /// The tape is recorded from [`KdHead::tape`], so it can be re-recorded unchanged at any time
    /// ([`KdHead::rerecord`]) — the order above is the one the reference's step runs, and step 0's
    /// golden receipt pins it.
    ///
    /// # Safety
    ///
    /// No tape of this value is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; `3 + 4·⌈t/rc⌉` dispatches and the chunk copies recorded.
    unsafe fn record(&self, g: &GpuCtx) {
        let (vk, cb, d) = (g.vk, self.cb, self.d);
        let tape = &self.tape;
        let words = |rows: u32, w: u32| u64::from(rows) * u64::from(w) * 4;
        vk.reset_cb(cb);
        vk.begin(cb);
        rec_step(vk, cb, &tape.norm_s);
        rec_step(vk, cb, &tape.norm_t);
        for c in &tape.chunks {
            let at = u64::from(c.start);
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_TRANSFER,
                ACCESS_SHADER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            let down = BufferCopy {
                src_offset: at * u64::from(d) * 4,
                dst_offset: 0,
                size: words(c.rows, d),
            };
            (vk.copy_buffer)(cb, tape.hf_s, tape.hf_s_c, 1, &down);
            (vk.copy_buffer)(cb, tape.hf_t, tape.hf_t_c, 1, &down);
            vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_COMPUTE,
                ACCESS_TRANSFER_WRITE,
                ACCESS_SHADER_READ,
            );
            rec_step(vk, cb, &c.fwd_s);
            rec_step(vk, cb, &c.fwd_t);
            rec_step(vk, cb, &c.ce);
            rec_step(vk, cb, &c.bwd);
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_TRANSFER,
                ACCESS_SHADER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            let up = BufferCopy {
                src_offset: 0,
                dst_offset: at * u64::from(d) * 4,
                size: words(c.rows, d),
            };
            (vk.copy_buffer)(cb, tape.dhf_c, tape.dhf, 1, &up);
            let lrow = BufferCopy {
                src_offset: 0,
                dst_offset: at * 4,
                size: u64::from(c.rows) * 4,
            };
            (vk.copy_buffer)(cb, tape.loss_c, self.loss, 1, &lrow);
        }
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_COMPUTE,
            ACCESS_TRANSFER_WRITE,
            ACCESS_SHADER_READ,
        );
        rec_step(vk, cb, &tape.norm_bwd);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
    }

    /// Re-record the tape with the profiler's timestamps: armed ([`crate::gpu::decode::prof_arm`]),
    /// `rec_step` writes a timestamp pair around each dispatch, so one [`KdHead::run`] yields one
    /// span per dispatch. The head's only re-record hook; the tape's dispatch order is unchanged.
    ///
    /// # Safety
    ///
    /// No tape of this value is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; the tape carries `3 + 4·⌈t/rc⌉` dispatches.
    pub unsafe fn rerecord(&self, g: &GpuCtx) {
        self.record(g);
    }

    /// Submit the head's tape: `h_s`, `h_t` → `dh` and the `loss` rows. `W_emb` and `g_final` are
    /// frozen, so the tape is idempotent and replays on every call.
    ///
    /// # Safety
    ///
    /// No buffer of this head or of its four caller-owned inputs is in use elsewhere on the
    /// device queue.
    ///
    /// # Complexity
    ///
    /// Per chunk: two `rc×d·d×v` forward products, the loss head's `6·rc·v` word events, the
    /// `rc×v·v×d` input adjoint and `2·rc·d + rc` copied words; plus `2·t·d + 6·t·d` word events
    /// for the two norms and the norm adjoint.
    pub unsafe fn run(&self, g: &GpuCtx) {
        g.vk.submit_wait(self.cb);
    }

    /// The `[T, d]` cotangent at the final norm input: the handle layer 27's block adjoint reads.
    #[must_use]
    pub fn dh(&self) -> Handle {
        self.dh
    }

    /// The `[T]` per-row cross-entropy buffer.
    #[must_use]
    pub fn loss_rows(&self) -> Handle {
        self.loss
    }

    /// The surviving `rc × v` cotangent tile: after [`KdHead::run`], the last chunk's `dz`, at
    /// tile row `r` for buffer row `chunk_start + r`. Earlier chunks' `dz` are overwritten in
    /// place, so a caller checking an earlier row builds the head with `rc = t`.
    #[must_use]
    pub fn dz_tile(&self) -> Handle {
        self.z_s
    }

    /// `Σ_t loss[t] / T` in f64, the step's scalar loss.
    ///
    /// # Safety
    ///
    /// [`KdHead::run`] has completed.
    ///
    /// # Complexity
    ///
    /// `O(t)` f64 additions and `t` f32 words copied to the host — the head's only host read.
    #[must_use]
    pub unsafe fn loss(&self, g: &GpuCtx) -> f64 {
        let rows = stream_download(g, self.loss, self.t as usize);
        rows.iter().map(|&x| f64::from(x)).sum::<f64>() / f64::from(self.t)
    }

    /// The head's geometry: `(t, d, v, rc)`.
    #[must_use]
    pub fn shape(&self) -> (u32, u32, u32, u32) {
        (self.t, self.d, self.v, self.rc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's mean is over `T` rows with an all-ones mask; `+1e-8` is below the f32 ulp
    /// at `T = 2048` (`compress_model.py:262-265`).
    const EPS64: f64 = 1e-6;

    /// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[0, 1)`.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Uniform in `[-scale, scale)`.
        fn sym(&mut self, scale: f64) -> f64 {
            (self.unit() * 2.0 - 1.0) * scale
        }
    }

    fn wide(x: &[f32]) -> Vec<f64> {
        x.iter().map(|&y| f64::from(y)).collect()
    }

    /// `hf = rmsnorm(x, g)` in f64: `r = (mean(x²) + eps)^(-1/2)`, `hf = x·r·g`.
    fn rms_fwd64(x: &[f64], gain: &[f64], t: usize, d: usize) -> Vec<f64> {
        let mut out = vec![0.0; t * d];
        for r in 0..t {
            let row = &x[r * d..(r + 1) * d];
            let ss: f64 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / d as f64 + EPS64).sqrt();
            for j in 0..d {
                out[r * d + j] = row[j] * inv * gain[j];
            }
        }
        out
    }

    /// `dx = rmsnorm_bwd(x, g, dy)` in f64: `a_j = g_j·dy_j`, `S = Σ a_j·x_j`,
    /// `dx_i = r·a_i − x_i·r³·S/d`.
    fn rms_bwd64(x: &[f64], gain: &[f64], dy: &[f64], t: usize, d: usize) -> Vec<f64> {
        let mut out = vec![0.0; t * d];
        for r in 0..t {
            let row = &x[r * d..(r + 1) * d];
            let ss: f64 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / d as f64 + EPS64).sqrt();
            let mut s = 0.0;
            for j in 0..d {
                s += gain[j] * dy[r * d + j] * row[j];
            }
            for j in 0..d {
                let a = gain[j] * dy[r * d + j];
                out[r * d + j] = inv * a - row[j] * inv * inv * inv * s / d as f64;
            }
        }
        out
    }

    /// `out[t,o] = Σ_i a[t,i]·w[o,i]` (`w` is the `[v, d]` embedding).
    fn mm64(a: &[f64], w: &[f64], t: usize, d: usize, o: usize) -> Vec<f64> {
        let mut out = vec![0.0; t * o];
        for r in 0..t {
            for j in 0..o {
                let wr = &w[j * d..(j + 1) * d];
                let ar = &a[r * d..(r + 1) * d];
                out[r * o + j] = ar.iter().zip(wr).map(|(x, y)| x * y).sum();
            }
        }
        out
    }

    /// `out[t,i] = Σ_o dz[t,o]·w[o,i]`.
    fn mm_bwd64(dz: &[f64], w: &[f64], t: usize, d: usize, o: usize) -> Vec<f64> {
        let mut out = vec![0.0; t * d];
        for r in 0..t {
            for oi in 0..o {
                let g = dz[r * o + oi];
                let wr = &w[oi * d..(oi + 1) * d];
                let orow = &mut out[r * d..(r + 1) * d];
                for i in 0..d {
                    orow[i] += g * wr[i];
                }
            }
        }
        out
    }

    /// Per-row forward cross-entropy `−Σ_v p_t·log p_s` and the cotangent `(p_s − p_t)/T` in f64.
    fn ce64(zs: &[f64], zt: &[f64], t: usize, v: usize) -> (Vec<f64>, Vec<f64>) {
        let inv_t = 1.0 / t as f64;
        let mut ce = vec![0.0; t];
        let mut dz = vec![0.0; t * v];
        for r in 0..t {
            let (zr_s, zr_t) = (&zs[r * v..(r + 1) * v], &zt[r * v..(r + 1) * v]);
            let ms = zr_s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mt = zr_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let es: Vec<f64> = zr_s.iter().map(|z| (z - ms).exp()).collect();
            let et: Vec<f64> = zr_t.iter().map(|z| (z - mt).exp()).collect();
            let ss: f64 = es.iter().sum();
            let st: f64 = et.iter().sum();
            let ln_s = ss.ln();
            let mut acc = 0.0;
            for i in 0..v {
                let p_s = es[i] / ss;
                let p_t = et[i] / st;
                let lp_s = (zr_s[i] - ms) - ln_s;
                acc -= p_t * lp_s;
                dz[r * v + i] = (p_s - p_t) * inv_t;
            }
            ce[r] = acc;
        }
        (ce, dz)
    }

    fn max_abs_rel(a: &[f64], b: &[f64]) -> (f64, f64, usize) {
        let (mut ma, mut mr, mut at) = (0.0f64, 0.0f64, 0usize);
        let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs();
            if d > ma {
                ma = d;
                at = i;
            }
            if b[i].abs() > 0.01 * scale {
                mr = mr.max(d / b[i].abs());
            }
        }
        (ma, mr, at)
    }

    #[test]
    fn kd_head_full_vocab_wall() {
        let vk = match unsafe { Vk::load() } {
            Ok(vk) => vk,
            Err(e) => {
                println!("SKIP: no Vulkan device present: {e}");
                return;
            }
        };
        // Qwen3-0.6B-Base: hidden 1024, vocabulary 151936, T = 2048, rc = 512 (two 512×151936 f32
        // logit tiles, 622 MB).
        let (t, d, v, rc) = (2048usize, 1024usize, 151_936usize, 512u32);
        let mut rng = Lcg(0x1519_3600_0000_0001);
        let h_s: Vec<f32> = (0..t * d).map(|_| rng.sym(1.732) as f32).collect();
        let h_t: Vec<f32> = (0..t * d).map(|_| rng.sym(1.732) as f32).collect();
        let w_emb: Vec<f32> = (0..v * d).map(|_| rng.sym(0.031) as f32).collect();
        let g_final: Vec<f32> = (0..d).map(|_| (0.5 + rng.unit()) as f32).collect();
        let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
        let t0 = std::time::Instant::now();
        let (w_b, gf_b) = unsafe { (g.up_f32(&w_emb), g.up_f32(&g_final)) };
        let (hs_b, ht_b) = unsafe { (g.up_f32(&h_s), g.up_f32(&h_t)) };
        let head =
            unsafe { KdHead::new(&g, t as u32, d as u32, v as u32, rc, w_b, gf_b, hs_b, ht_b) };
        println!(
            "full-vocab setup (upload + tape): {:.2} s",
            t0.elapsed().as_secs_f64()
        );
        let mut walls = Vec::new();
        for i in 0..3 {
            let s = std::time::Instant::now();
            unsafe { head.run(&g) };
            let w = s.elapsed().as_secs_f64();
            walls.push(w);
            let loss = unsafe { head.loss(&g) };
            println!("step {i}: {w:.4} s, loss {loss:.6}");
            assert!(loss.is_finite(), "step {i}: loss {loss} is not finite");
        }
        let mut sorted = walls.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite walls"));
        println!(
            "full-vocab step wall: median {:.4} s over {} runs (min {:.4}, max {:.4}); d={d} \
             v={v} T={t} rc={rc}",
            sorted[sorted.len() / 2],
            walls.len(),
            sorted[0],
            sorted[sorted.len() - 1]
        );
        let dh = unsafe { stream_download(&g, head.dh(), t * d) };
        assert!(
            dh.iter().all(|x| x.is_finite()),
            "dh has a non-finite entry"
        );
        println!(
            "dh: {} words, max |dh| {:.3e}",
            dh.len(),
            dh.iter().fold(0.0f32, |m, x| m.max(x.abs()))
        );
        assert!(head.shape() == (t as u32, d as u32, v as u32, rc), "shape");
        assert_eq!(head.loss_rows(), head.loss_rows(), "loss handle stable");
    }

    /// The reference's rows {0, 1, 1023, 2047} of KD step 0, written by the lead from
    /// `compress_model.py`'s own logits: `ce_row` alone on them must reproduce `dz` and the per-row
    /// cross-entropy. Absent files skip the test with a reason, they do not pass it.
    #[test]
    fn kd_head_golden_rows() {
        let dir = std::path::Path::new("/home/socol/oc-tmp/nq-ref/kd/trace");
        let read = |name: &str| -> Option<Vec<f32>> {
            let p = dir.join(name);
            let bytes = std::fs::read(&p).ok()?;
            Some(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        };
        let (rows, v) = (4usize, 151_936usize);
        let (zs, zt, dz_ref) = match (
            read("z_s.rows.f32"),
            read("z_t.rows.f32"),
            read("dz.rows.f32"),
        ) {
            (Some(a), Some(b), Some(c))
                if [a.len(), b.len(), c.len()].iter().all(|&n| n == rows * v) =>
            {
                (a, b, c)
            }
            _ => {
                println!(
                    "SKIP: {} does not hold z_s/z_t/dz rows of [{rows}, {v}] f32",
                    dir.display()
                );
                return;
            }
        };
        let vk = match unsafe { Vk::load() } {
            Ok(vk) => vk,
            Err(e) => {
                println!("SKIP: no Vulkan device present: {e}");
                return;
            }
        };
        let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
        let (zs_b, zt_b) = unsafe { (g.up_f32(&zs), g.up_f32(&zt)) };
        let loss_b = unsafe { g.dev(rows * 4) };
        let inv_t = 1.0f32 / 2048.0;
        let step =
            unsafe { ce_row_dispatch(&g, zs_b, zt_b, loss_b, zs_b, rows as u32, v as u32, inv_t) };
        unsafe {
            let cb = vk.command_buffer();
            vk.begin(cb);
            rec_step(&vk, cb, &step);
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            vk.submit_wait(cb);
        }
        let dz_dev = unsafe { stream_download(&g, zs_b, rows * v) };
        let loss_dev = unsafe { stream_download(&g, loss_b, rows) };
        let pinned = [10.142_798_68f64, 4.956_850_45, 0.313_139_08, 3.649_900_13];
        let mut max_dz = 0.0f32;
        let mut at = 0usize;
        for i in 0..rows * v {
            let d = (dz_dev[i] - dz_ref[i]).abs();
            if d > max_dz {
                max_dz = d;
                at = i;
            }
        }
        println!(
            "golden dz: max |Δ| {max_dz:.3e} at {at} (row {}, v={}; reference {:.3e}, device \
             {:.3e}); |dz|max {:.3e}",
            at / v,
            at % v,
            dz_ref[at],
            dz_dev[at],
            dz_ref.iter().fold(0.0f32, |m, x| m.max(x.abs()))
        );
        assert!(
            max_dz <= 1e-6,
            "golden dz: max |Δ| {max_dz:.3e} exceeds 1e-6"
        );
        for r in 0..rows {
            let dev = f64::from(loss_dev[r]);
            let rel = (dev - pinned[r]).abs() / pinned[r].abs();
            println!(
                "golden CE row {r}: lead f64 {:.8} device {dev:.8} (rel {rel:.2e})",
                pinned[r]
            );
            assert!(
                rel <= 1e-4,
                "golden CE row {r}: device {dev:.8} vs {:.8} is {rel:.2e} relative",
                pinned[r]
            );
        }
    }

    #[test]
    fn kd_head_matches_f64() {
        let vk = match unsafe { Vk::load() } {
            Ok(vk) => vk,
            Err(e) => {
                println!("SKIP: no Vulkan device present: {e}");
                return;
            }
        };
        let (t, d, v) = (48usize, 64usize, 5003usize);
        let mut rng = Lcg(0x2025_1025_0bad_c0de);
        let h_s: Vec<f32> = (0..t * d).map(|_| rng.sym(1.732) as f32).collect();
        let h_t: Vec<f32> = (0..t * d).map(|_| rng.sym(1.732) as f32).collect();
        let w_emb: Vec<f32> = (0..v * d).map(|_| rng.sym(0.125) as f32).collect();
        let g_final: Vec<f32> = (0..d).map(|_| (0.5 + rng.unit()) as f32).collect();
        let (hs64, ht64, w64, gf64) = (wide(&h_s), wide(&h_t), wide(&w_emb), wide(&g_final));
        // ── the f64 reference of the same composition ──
        let hf_s64 = rms_fwd64(&hs64, &gf64, t, d);
        let hf_t64 = rms_fwd64(&ht64, &gf64, t, d);
        let zs64 = mm64(&hf_s64, &w64, t, d, v);
        let zt64 = mm64(&hf_t64, &w64, t, d, v);
        let (ce_ref, dz64) = ce64(&zs64, &zt64, t, v);
        let loss64: f64 = ce_ref.iter().sum::<f64>() / t as f64;
        let dhf64 = mm_bwd64(&dz64, &w64, t, d, v);
        let dh64 = rms_bwd64(&hs64, &gf64, &dhf64, t, d);
        let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
        let (w_b, gf_b) = unsafe { (g.up_f32(&w_emb), g.up_f32(&g_final)) };
        let (hs_b, ht_b) = unsafe { (g.up_f32(&h_s), g.up_f32(&h_t)) };
        // three chunks of 16 rows, then one chunk of all 48 for the full `dz` (in-place on `z_s`)
        let head16 =
            unsafe { KdHead::new(&g, t as u32, d as u32, v as u32, 16, w_b, gf_b, hs_b, ht_b) };
        let head48 =
            unsafe { KdHead::new(&g, t as u32, d as u32, v as u32, 48, w_b, gf_b, hs_b, ht_b) };
        unsafe {
            head16.run(&g);
            head48.run(&g);
        }
        let loss16 = unsafe { head16.loss(&g) };
        let loss48 = unsafe { head48.loss(&g) };
        let dh16 = wide(&unsafe { stream_download(&g, head16.dh(), t * d) });
        let dh48 = wide(&unsafe { stream_download(&g, head48.dh(), t * d) });
        let dz48 = wide(&unsafe { stream_download(&g, head48.dz_tile(), t * v) });
        println!(
            "loss f64 {loss64:.9} device rc=16 {loss16:.9} (rel {:.2e}) rc=48 {loss48:.9} (rel \
             {:.2e}); rc16 vs rc48 rel {:.2e}",
            (loss16 - loss64).abs() / loss64.abs(),
            (loss48 - loss64).abs() / loss64.abs(),
            (loss16 - loss48).abs() / loss48.abs(),
        );
        assert!(
            (loss16 - loss64).abs() <= 1e-5 * loss64.abs(),
            "rc=16 loss {loss16} vs f64 {loss64}: over 1e-5 relative"
        );
        assert!(
            (loss48 - loss64).abs() <= 1e-5 * loss64.abs(),
            "rc=48 loss {loss48} vs f64 {loss64}: over 1e-5 relative"
        );
        // ── dh against f64 central differences of the f64 loss in h_s: the check that ties the
        // device cotangent to the loss it differentiates, so a wrong `1/T` or a missing norm
        // backward reddens here ──
        let loss_at = |idx: usize, delta: f64| -> f64 {
            let mut x = hs64.clone();
            x[idx] += delta;
            let hf = rms_fwd64(&x, &gf64, t, d);
            let z = mm64(&hf, &w64, t, d, v);
            let (ce, _) = ce64(&z, &zt64, t, v);
            ce.iter().sum::<f64>() / t as f64
        };
        let step = 1e-3;
        let scale = dh64.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let mut idxs = vec![0usize, t * d - 1];
        for _ in 0..6 {
            idxs.push((rng.unit() * (t * d) as f64) as usize % (t * d));
        }
        for idx in idxs {
            let fd = (loss_at(idx, step) - loss_at(idx, -step)) / (2.0 * step);
            let an = dh64[idx];
            let dev = dh16[idx];
            let floor = 1e-3 * scale;
            let rel = (fd - an).abs() / an.abs().max(floor);
            let rel_dev = (fd - dev).abs() / an.abs().max(floor);
            println!(
                "fd h_s[{idx}]: central {fd:.9e} analysed f64 {an:.9e} device {dev:.9e} (rel \
                 {rel:.2e}, {rel_dev:.2e})"
            );
            assert!(
                rel <= 1e-3,
                "fd h_s[{idx}]: central {fd:.6e} vs analysed {an:.6e} is {rel:.2e} relative"
            );
            assert!(
                rel_dev <= 1e-3,
                "fd h_s[{idx}]: central {fd:.6e} vs device {dev:.6e} is {rel_dev:.2e} relative"
            );
        }
        // ── dz on the chunk-boundary rows of the 16-row tiling, plus the last row ──
        for &r in &[0usize, 15, 16, 47] {
            let a = &dz48[r * v..(r + 1) * v];
            let b = &dz64[r * v..(r + 1) * v];
            let (ma, _, at) = max_abs_rel(a, b);
            println!(
                "dz row {r}: max |Δ| {ma:.3e} (f64 {:.3e} device {:.3e} at v={at})",
                b[at], a[at]
            );
            assert!(ma <= 1e-6, "dz row {r}: max |Δ| {ma:.3e} exceeds 1e-6");
        }
        // ── dh, max over all entries ──
        for (label, dh) in [("rc=16", &dh16), ("rc=48", &dh48)] {
            let (ma, mr, at) = max_abs_rel(dh, &dh64);
            let scale = dh64.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            println!(
                "{label} dh: max |Δ| {ma:.3e} = {:.2e} of max|dh| {scale:.3e}; worst per-entry \
                 relative (|dh| ≥ 1% of max) {mr:.2e}; at {at} f64 {:.6e} device {:.6e}",
                ma / scale,
                dh64[at],
                dh[at]
            );
            assert!(
                ma <= 1e-4 * scale,
                "{label} dh: max |Δ| {ma:.3e} exceeds 1e-4 of max|dh| {scale:.3e}"
            );
            assert!(
                mr <= 1e-4,
                "{label} dh: worst per-entry relative error {mr:.2e} exceeds 1e-4"
            );
        }
    }
}
