//! One Qwen3 decoder layer's training step as a chained tape on the device: the attention front
//! ([`FrontTune`]) whose output feeds the residual block's o-projection ([`BlockTune`]), and whose
//! own input adjoint takes the block's residual adjoint `dh` as its `dres`.
//!
//! # The composition
//!
//! With `x` the layer input, `a = front(x)` and `y = block(x, a)`:
//!
//! - forward: `front.forward` writes `ao`; one device copy puts it in the block's `a`, one puts the
//!   front's `x` in the block's `x`; `block.run(eval)` writes `y`.
//! - backward from the caller's output cotangent `dy = dL/dy` in [`ChainLayer::dy`]:
//!   `block.run(train)` writes `dh` (`dL/dx` down the residual path) and `da = dh·Wo`; one device
//!   copy puts `dh` in the front's `dres` and one puts `da` in its `dao`; `front.backward_cols`
//!   then writes `dx = dL/dx`, which is the previous layer's `dy`.
//!
//! Every copy is device-to-device in a submission of its own, with a compute↔transfer barrier at
//! each end: no host round trip between layers.
//!
//! # Why: the quantisation-aware distillation's adjoint
//!
//! NanoQuant trains only the per-linear scale vectors and freezes every other parameter
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `src/nanoquant/core/compress_model.py:217-230`:
//! `requires_grad = True` for the parameters whose name contains `scale`, nothing else), and runs
//! that fit block by block over the quantised stream (`compress_model.py:83-101`). A block's scale
//! gradient needs the cotangent of the block's *input*, which needs the tokens of the blocks after
//! it in the loss: `dL/ds` of layer `k` is a function of `dL/dx_k`, and `dL/dx_k` is what this tape
//! produces. That is the piece `front.rs` did not carry (`dx` was never computed) and the piece a
//! splice of per-block losses cannot supply.
//!
//! # Evidence grade
//!
//! `tested` by `chain_dx_matches_fd` in this file: the layer input's adjoint of two chained layers
//! against f64 central differences of the chained loss, and layer 0's `q` and `down` scale
//! gradients against central differences of the same loss.
//!
//! # Trust boundary
//!
//! Not covered: the GPU driver and the SPIR-V compiler; the caller's `dy` (it is this layer's
//! output cotangent, and the loss that produces it at the end of a real stack is the distillation
//! objective's, not this tape's); and the Adam steps the block's training tape runs after the
//! adjoints are written.

use crate::gpu::decode::GpuCtx;
use crate::gpu::vulkan::*;

use super::block::{BlockShape, BlockTune, HostFactor};
use super::front::{FrontShape, FrontTune};

/// Copy the whole of `src` into `dst` on the device — `words` f32 in one submission, with the
/// compute→transfer and transfer→compute barriers the neighbouring dispatches need.
///
/// # Safety
///
/// `src` holds `words` f32 with `TRANSFER_SRC` and `dst` the same with `TRANSFER_DST`; no
/// submission of `cb` is pending.
///
/// # Complexity
///
/// `O(1)` host work; one submission copying `O(words)` bytes.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a word count times four indexes a u64 byte size"
)]
pub(super) unsafe fn copy_words(
    g: &GpuCtx,
    cb: CommandBuffer,
    src: Handle,
    dst: Handle,
    words: usize,
) {
    let vk = g.vk;
    vk.reset_cb(cb);
    vk.begin(cb);
    vk.barrier(
        cb,
        STAGE_COMPUTE,
        STAGE_TRANSFER,
        ACCESS_SHADER_WRITE,
        ACCESS_TRANSFER_READ,
    );
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: (words * 4) as u64,
    };
    (vk.copy_buffer)(cb, src, dst, 1, &region);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_COMPUTE,
        ACCESS_TRANSFER_WRITE,
        ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
}

/// One decoder layer as the chained pair: the attention front and the residual block, the block in
/// its external-cotangent mode.
pub struct ChainLayer {
    shape: FrontShape,
    block_shape: BlockShape,
    front: FrontTune,
    block: BlockTune,
    /// The layer's copies, one submission each.
    xfer: CommandBuffer,
    /// A zeroed `t × d` buffer: the block's `yt` under an external cotangent, where the loss rows
    /// carry no meaning.
    zero: Handle,
}

impl ChainLayer {
    /// Allocate one layer: a [`FrontTune`] and a [`BlockTune`] over the same columns, the block in
    /// its external-cotangent mode.
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value.
    ///
    /// # Complexity
    ///
    /// `O(1)` pipelines and `O(t·(d + a + f) + Σ out·inn·rank)` device words, the front's and the
    /// block's together.
    #[must_use]
    pub unsafe fn new(
        g: &GpuCtx,
        shape: FrontShape,
        block_shape: BlockShape,
        lr: f64,
        lr_scale: f64,
        lr_dense: f64,
    ) -> Self {
        assert_eq!(
            (shape.t(), shape.d, shape.qw()),
            (block_shape.t, block_shape.d, block_shape.a),
            "chain: the front's columns are the block's"
        );
        let mut block = BlockTune::new(g, block_shape, lr, lr_scale, lr_dense);
        block.set_external_cotangent(g, true);
        let zero = g.dev_zeroed(block_shape.t * block_shape.d * 4);
        let s = Self {
            shape,
            block_shape,
            front: FrontTune::new(g, shape, lr, lr_scale),
            block,
            xfer: g.vk.reusable_command_buffer(),
            zero,
        };
        copy_words(
            g,
            s.xfer,
            s.zero,
            s.block.cols().2,
            s.block_shape.t * s.block_shape.d,
        );
        s
    }

    /// Load the front's three factors (`[q, k, v]`) and its two norm gains, the block's four
    /// (`[o, gate, up, down]`) and its FFN gain.
    ///
    /// # Safety
    ///
    /// The factors match the shapes' dimensions and ranks and the gains their widths; no tape is
    /// pending.
    ///
    /// # Complexity
    ///
    /// `O(Σ (out + inn)·rank + d)` words copied.
    pub unsafe fn load(
        &mut self,
        g: &GpuCtx,
        ff: &[HostFactor],
        g1: &[f32],
        gn: &[f32],
        bf: &[HostFactor],
        g2: &[f32],
    ) {
        self.front.load(g, ff, g1, gn);
        self.block.load(g, bf, g2);
    }

    /// The layer input's buffer, `t × d`: the front's `x`, which the block's `x` is copied from.
    #[must_use]
    pub fn x(&self) -> Handle {
        self.front.cols().0
    }

    /// The layer output's buffer, `t × d`: the block's `y`.
    #[must_use]
    pub fn y(&self) -> Handle {
        self.block.out()
    }

    /// The output cotangent's buffer, `t × d` (`dy`): the block's `dy`, which the caller fills by a
    /// device copy before [`ChainLayer::backward`].
    #[must_use]
    pub fn dy(&self) -> Handle {
        self.block.dy_col()
    }

    /// The input cotangent's buffer, `t × d` (`dx`): the front's `dx`, valid after a backward.
    #[must_use]
    pub fn dx(&self) -> Handle {
        self.front.dx()
    }

    /// The attention front, for its factor gradients after a backward.
    #[must_use]
    pub fn front(&self) -> &FrontTune {
        &self.front
    }

    /// The residual block, for its factor gradients after a backward.
    #[must_use]
    pub fn block(&self) -> &BlockTune {
        &self.block
    }

    /// The forward: the front's pass, the two device copies onto the block's input columns, then
    /// the block's evaluation pass.
    ///
    /// # Safety
    ///
    /// [`ChainLayer::load`] has run and [`ChainLayer::x`] holds the layer's input; no tape of this
    /// value is pending.
    ///
    /// # Complexity
    ///
    /// The front's `O(t·n_ctx·nh·hd + t·d·(nh·hd + nkv·hd) + Σ out·inn·rank)` and the block's
    /// `O(t·(a·d + 3·d·f) + Σ out·inn·rank)` multiply-adds, plus `O(t·(d + nh·hd))` bytes copied.
    pub unsafe fn forward(&mut self, g: &GpuCtx) {
        let (t, d, qw) = (self.shape.t(), self.shape.d, self.shape.qw());
        self.front.forward(g);
        copy_words(g, self.xfer, self.front.ao(), self.block.cols().1, t * qw);
        copy_words(
            g,
            self.xfer,
            self.front.cols().0,
            self.block.cols().0,
            t * d,
        );
        let _ = self.block.run(g, false);
    }

    /// The backward from the cotangent [`ChainLayer::dy`] holds: the block's training pass — its
    /// `dh`, its `da`, its factor adjoints and its Adam steps — then the two device copies, then
    /// the front's backward, which writes [`ChainLayer::dx`].
    ///
    /// # Safety
    ///
    /// [`ChainLayer::forward`] has run on the same input and [`ChainLayer::dy`] holds the output
    /// cotangent; no tape of this value is pending.
    ///
    /// # Complexity
    ///
    /// The block's `O(2·t·(a·d + 3·d·f) + Σ out·inn·rank)` and the front's
    /// `O(t·n_ctx·nh·hd + t·d·(nh·hd + nkv·hd) + Σ out·inn·rank)` multiply-adds, plus
    /// `O(t·(d + nh·hd))` bytes copied.
    pub unsafe fn backward(&mut self, g: &GpuCtx) {
        let (t, d, qw) = (self.shape.t(), self.shape.d, self.shape.qw());
        let _ = self.block.run(g, true);
        copy_words(g, self.xfer, self.block.dh(), self.front.dres(), t * d);
        copy_words(g, self.xfer, self.block.da(), self.front.dao(), t * qw);
        self.front.backward_cols(g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::nn::train::attn_bwd::tests::ref_fwd_ungated;
    use crate::gpu::nn::train::block::tests::{dense, factor, lin, unit};
    use crate::gpu::nn::train::block::{stream_buffer, stream_download, stream_upload};

    fn to64(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x)).collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// `max|got − want| / max|want|`: the whole-vector relative error the twin tests use.
    fn rel_vec(got: &[f32], want: &[f64]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 1e-30f64;
        for (&p, &q) in got.iter().zip(want) {
            num = num.max((f64::from(p) - q).abs());
            den = den.max(q.abs());
        }
        num / den
    }

    /// One entry's relative error.
    fn rel_one(got: f64, want: f64) -> f64 {
        (got - want).abs() / want.abs().max(1e-6)
    }

    /// `n = rmsnorm(x, g)` over `d` at the device forward's epsilon.
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

    /// The central difference of `f` in entry `idx` of the f64 vector `at`, at the step `eps`.
    ///
    /// # Complexity
    ///
    /// Two evaluations of `f`.
    fn fd_at(f: &dyn Fn(&[f64]) -> f64, at: &[f64], idx: usize, eps: f64) -> f64 {
        let mut p = at.to_vec();
        let mut q = at.to_vec();
        p[idx] += eps;
        q[idx] -= eps;
        (f(&p) - f(&q)) / (p[idx] - q[idx])
    }

    /// One layer's parameters in f64: the front's factors, the block's factors and the gains.
    #[derive(Clone)]
    struct HostLayer {
        ff: Vec<HostFactor>,
        bf: Vec<HostFactor>,
        g1: Vec<f64>,
        gn: Vec<f64>,
        g2: Vec<f64>,
    }

    /// The two-layer f64 reference: `y0 = layer0(x0)`, `y1 = layer1(y0)`, `L = Σ y1 ⊙ R`.
    #[derive(Clone)]
    struct Host {
        sh: FrontShape,
        bs: BlockShape,
        l: [HostLayer; 2],
        r: Vec<f64>,
    }

    impl Host {
        /// The chained loss at the layer input `x0` (`t × d`), in f64.
        fn loss(&self, x0: &[f64]) -> f64 {
            dot(&self.layer(1, &self.layer(0, x0)), &self.r)
        }

        /// Layer `k`'s forward in f64, the twin of [`ChainLayer::forward`].
        fn layer(&self, k: usize, x: &[f64]) -> Vec<f64> {
            let ao = self.front(k, x);
            self.block(k, x, &ao)
        }

        /// The attention front's forward in f64: `n1 = rmsnorm(x, g1)`, the three projections and
        /// the ungated attention map, giving `ao`.
        ///
        /// # Complexity
        ///
        /// `O(t·d·(nh·hd + nkv·hd) + t·n_ctx·nh·hd)` multiply-adds, and `O(Σ out·inn·rank)` to
        /// materialise the three factors' dense weights.
        fn front(&self, k: usize, x: &[f64]) -> Vec<f64> {
            let (t, d) = (self.sh.t(), self.sh.d);
            let (qw, kw) = (self.sh.qw(), self.sh.kw());
            let hl = &self.l[k];
            let fw: Vec<Vec<f64>> = hl.ff.iter().map(dense).collect();
            let n1 = rmsnorm(x, &hl.g1, t, d);
            let pq = lin(&n1, &fw[0], t, d, qw);
            let pk = lin(&n1, &fw[1], t, d, kw);
            let pv = lin(&n1, &fw[2], t, d, kw);
            ref_fwd_ungated(self.sh.attn(), &pq, &pk, &hl.gn, &pv)
        }

        /// The residual block's forward in f64: `h = x + ao·Woᵀ`, `n = rmsnorm(h, g2)`,
        /// `y = h + (silu(n·Wgᵀ) ⊙ n·Wuᵀ)·Wdᵀ`.
        ///
        /// # Complexity
        ///
        /// `O(t·(a·d + 3·d·f))` multiply-adds, and `O(Σ out·inn·rank)` to materialise the four
        /// factors' dense weights.
        fn block(&self, k: usize, x: &[f64], ao: &[f64]) -> Vec<f64> {
            let (t, d) = (self.sh.t(), self.sh.d);
            let (aw, f) = (self.bs.a, self.bs.f);
            let hl = &self.l[k];
            let bw: Vec<Vec<f64>> = hl.bf.iter().map(dense).collect();
            let mut h = lin(ao, &bw[0], t, aw, d);
            for (hv, xv) in h.iter_mut().zip(x) {
                *hv += xv;
            }
            let n = rmsnorm(&h, &hl.g2, t, d);
            let gt = lin(&n, &bw[1], t, d, f);
            let up = lin(&n, &bw[2], t, d, f);
            let m: Vec<f64> = gt
                .iter()
                .zip(&up)
                .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
                .collect();
            let mut y = lin(&m, &bw[3], t, f, d);
            for (yv, hv) in y.iter_mut().zip(&h) {
                *yv += hv;
            }
            y
        }
    }

    /// Add `delta` to one scale entry of layer `k`'s front factor `fk`, `j = 2` (`s_post`) or `3`
    /// (`s_pre`), and return the value written. The entry is f32, so the caller takes the applied
    /// step from the two written values, never from `2·delta`.
    fn bump_front(h: &mut Host, k: usize, fk: usize, j: usize, idx: usize, delta: f32) -> f32 {
        let v = if j == 2 {
            &mut h.l[k].ff[fk].s_post
        } else {
            &mut h.l[k].ff[fk].s_pre
        };
        v[idx] += delta;
        v[idx]
    }

    /// [`bump_front`] over the block's factors.
    fn bump_block(h: &mut Host, k: usize, fk: usize, j: usize, idx: usize, delta: f32) -> f32 {
        let v = if j == 2 {
            &mut h.l[k].bf[fk].s_post
        } else {
            &mut h.l[k].bf[fk].s_pre
        };
        v[idx] += delta;
        v[idx]
    }

    /// Two chained layers at the brief's shape. Checks, each against central differences of the
    /// f64 chained loss `L = Σ y1 ⊙ R`:
    ///
    /// 1. layer 0's input adjoint `dx0` at eight indices, within `1e-3` relative;
    /// 2. layer 0's `ds_post`/`ds_pre` for slot `q` and slot `down` — the parameters the KD trains —
    ///    over every entry of the four vectors, within `1e-3` in the whole-vector metric and `1e-2`
    ///    per entry;
    /// 3. the block adjoints the chain hands the front: layer 1's `dh` and `da`.
    ///
    /// Every compared pair is printed. The per-entry tolerance is wider than the whole-vector one
    /// because a single entry of a scale gradient can be a cancelling sum: at
    /// `q s_pre[63]` the host's own analytic value from the device's f32 `dao` is already `1.7e-3`
    /// from its exact central difference, so `1e-3` per entry is below the f32 floor there.
    #[test]
    #[allow(clippy::too_many_lines, reason = "one chained tape comparison")]
    fn chain_dx_matches_fd() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let sh = FrontShape {
            chunks: 1,
            n_ctx: 32,
            d: 64,
            nh: 4,
            nkv: 2,
            hd: 16,
            rot: 16,
            theta: 1e4,
            eps: 1e-6,
            ranks: [8, 6, 6],
        };
        let bs = BlockShape {
            t: sh.t(),
            d: sh.d,
            a: sh.qw(),
            f: 128,
            ranks: [8, 8, 8, 8],
        };
        let (t, d, qw) = (sh.t(), sh.d, sh.qw());
        let gain = |n: usize, seed: u64| -> Vec<f64> {
            to64(
                &unit(n, seed)
                    .iter()
                    .map(|v| 1.0 + 0.2 * v)
                    .collect::<Vec<f32>>(),
            )
        };
        let host = Host {
            sh,
            bs,
            l: [0, 1].map(|k| HostLayer {
                ff: (0..3)
                    .map(|j| {
                        let (o, i) = sh.dims(j);
                        factor(o, i, sh.ranks[j], 100 + 10 * k as u64 + j as u64)
                    })
                    .collect(),
                bf: (0..4)
                    .map(|j| {
                        let (o, i) = bs.dims(j);
                        factor(o, i, bs.ranks[j], 200 + 10 * k as u64 + j as u64)
                    })
                    .collect(),
                g1: gain(d, 300 + k as u64),
                gn: gain(2 * sh.hd, 310 + k as u64),
                g2: gain(d, 320 + k as u64),
            }),
            r: gain(t * d, 400).iter().map(|&v| 0.25 * v).collect(),
        };
        let x0: Vec<f32> = unit(t * d, 900);
        let r32: Vec<f32> = host.r.iter().map(|&v| v as f32).collect();
        let x64 = to64(&x0);

        unsafe {
            let mut c0 = ChainLayer::new(&g, sh, bs, 1e-3, 1e-3, 1e-3);
            let mut c1 = ChainLayer::new(&g, sh, bs, 1e-3, 1e-3, 1e-3);
            for (c, k) in [&mut c0, &mut c1].into_iter().zip([0usize, 1]) {
                let hl = &host.l[k];
                let w32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
                c.load(&g, &hl.ff, &w32(&hl.g1), &w32(&hl.gn), &hl.bf, &w32(&hl.g2));
            }
            // Forward: `x0` into layer 0, layer 0's `y` into layer 1's `x`.
            let xf = g.vk.reusable_command_buffer();
            let xs = stream_buffer(&g, t * d);
            stream_upload(&g, xs, &x0);
            copy_words(&g, xf, xs, c0.x(), t * d);
            c0.forward(&g);
            copy_words(&g, xf, c0.y(), c1.x(), t * d);
            c1.forward(&g);
            let y0h = host.layer(0, &x64);
            let y1h = host.layer(1, &y0h);
            println!(
                "y0 vs host {:.3e}, y1 vs host {:.3e}",
                rel_vec(&stream_download(&g, c0.y(), t * d), &y0h),
                rel_vec(&stream_download(&g, c1.y(), t * d), &y1h)
            );

            // Backward from `dy1 = R`: layer 1, then its `dx` into layer 0's `dy`.
            let rs = stream_buffer(&g, t * d);
            stream_upload(&g, rs, &r32);
            copy_words(&g, xf, rs, c1.dy(), t * d);
            c1.backward(&g);
            copy_words(&g, xf, c1.dx(), c0.dy(), t * d);
            c0.backward(&g);

            // (1) layer 0's input adjoint.
            let dx0 = stream_download(&g, c0.dx(), t * d);
            let dx_f = |x: &[f64]| host.loss(x);
            let mut idxs = vec![0usize, t * d / 2, t * d - 1];
            let mut s = 12_345u64;
            for _ in 0..5 {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                idxs.push(((s >> 33) as usize) % (t * d));
            }
            let mut worst = 0f64;
            for idx in idxs {
                let fd = fd_at(&dx_f, &x64, idx, 1e-3);
                let dv = f64::from(dx0[idx]);
                let e = rel_one(dv, fd);
                worst = worst.max(e);
                println!("dx0 [{idx}]: device {dv:.8e} fd {fd:.8e} rel {e:.2e}");
            }
            println!("dx0 worst relative error {worst:.2e}");
            assert!(worst < 1e-3, "dx0 worst relative error {worst:.2e}");

            // (3) the block's two adjoints at layer 1, against FD of `Σ block(x, a) ⊙ R`.
            let dh1 = stream_download(&g, c1.block().dh(), t * d);
            let da1 = stream_download(&g, c1.block().da(), t * qw);
            let ao1 = host.front(1, &y0h);
            let dh_f = |x: &[f64]| dot(&host.block(1, x, &ao1), &host.r);
            let da_f = |a: &[f64]| dot(&host.block(1, &y0h, a), &host.r);
            let mut worst_dh = 0f64;
            for idx in [0, t * d / 2, t * d - 1] {
                let fd = fd_at(&dh_f, &y0h, idx, 1e-3);
                let dv = f64::from(dh1[idx]);
                worst_dh = worst_dh.max(rel_one(dv, fd));
                println!("dh1 [{idx}]: device {dv:.8e} fd {fd:.8e}");
            }
            let mut worst_da = 0f64;
            for idx in [0, t * qw / 2, t * qw - 1] {
                let fd = fd_at(&da_f, &ao1, idx, 1e-3);
                let dv = f64::from(da1[idx]);
                worst_da = worst_da.max(rel_one(dv, fd));
                println!("da1 [{idx}]: device {dv:.8e} fd {fd:.8e}");
            }
            println!("dh1 worst relative error {worst_dh:.2e}, da1 {worst_da:.2e}");
            assert!(worst_dh < 1e-3, "dh1 worst relative error {worst_dh:.2e}");
            assert!(worst_da < 1e-3, "da1 worst relative error {worst_da:.2e}");

            // (2) layer 0's `q` and `down` scales, the parameters the KD trains.
            let fg = c0.front().grads(&g);
            let bg = c0.block().grads(&g);
            let mut worst = 0f64;
            let mut worst_one = 0f64;
            for (name, got, fk, j, len, front) in [
                ("q s_post", fg[0][2].clone(), 0usize, 2usize, qw, true),
                ("q s_pre", fg[0][3].clone(), 0, 3, d, true),
                ("down s_post", bg[3][2].clone(), 3, 2, d, false),
                ("down s_pre", bg[3][3].clone(), 3, 3, bs.f, false),
            ] {
                let mut want = vec![0f64; len];
                for idx in 0..len {
                    let eps = 1e-4f32;
                    let mut p = host.clone();
                    let mut q = host.clone();
                    let (hi, lo) = if front {
                        (
                            bump_front(&mut p, 0, fk, j, idx, eps),
                            bump_front(&mut q, 0, fk, j, idx, -eps),
                        )
                    } else {
                        (
                            bump_block(&mut p, 0, fk, j, idx, eps),
                            bump_block(&mut q, 0, fk, j, idx, -eps),
                        )
                    };
                    let fd = (p.loss(&x64) - q.loss(&x64)) / (f64::from(hi) - f64::from(lo));
                    want[idx] = fd;
                    if [0, len / 2, len - 1].contains(&idx) {
                        let dv = f64::from(got[idx]);
                        let e = rel_one(dv, fd);
                        worst_one = worst_one.max(e);
                        println!("{name} [{idx}]: device {dv:.8e} fd {fd:.8e} rel {e:.2e}");
                    }
                }
                let w = rel_vec(&got, &want);
                worst = worst.max(w);
                println!("{name}: whole-vector relative error {w:.2e}");
            }
            println!(
                "scale gradient: worst whole-vector {worst:.2e}, worst sampled entry {worst_one:.2e}"
            );
            assert!(
                worst < 1e-3,
                "scale gradient worst whole-vector relative error {worst:.2e}"
            );
            assert!(
                worst_one < 1e-2,
                "scale gradient worst sampled entry {worst_one:.2e}"
            );
        }
    }
}
