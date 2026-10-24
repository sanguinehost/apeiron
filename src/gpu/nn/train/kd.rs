//! The knowledge-distillation step of NanoQuant's global scale fit, on the device: one sample of
//! 2048 tokens through all 28 layers, the cross-entropy head of [`KdHead`](super::head::KdHead), the
//! chained input adjoints of [`ChainLayer`](super::chain::ChainLayer), and one Adam step over the
//! 392 per-linear scale vectors (`SamsungLabs/NanoQuant` @ `a9e0a43`,
//! `src/nanoquant/core/compress_model.py:125-297` `compress_model_recon`).
//!
//! # What the reference trains
//!
//! `compress_model.py:213-230` sets `requires_grad = True` on the parameters whose name contains
//! `scale` — `scale_pre` (the input side) and `scale_post` (the output side) of every
//! `NanoQuantLinear`, 14 per layer over 28 layers — and nothing else: the binary `U`/`V` signs, the
//! norm gains, the embedding and the tied head are frozen. The optimizer is `optimi.AdamW(lr=1e-5)`
//! with betas `(0.9, 0.99)` and `eps = 1e-6` (`:238`), its weight decay is a measured no-op
//! (`decouple_wd=True` multiplies by `1 − 1e-7`, which is the identity on a bf16 parameter:
//! `optimi/adamw.py:67`, `optimi/adam.py:372-375`), so this driver applies **no weight decay**, and
//! the rate is stepped by `CosineAnnealingLR(T_max = 1024, eta_min = 0)` once per sample (`:240-241`).
//! One sample is 2048 tokens at batch 1 with an all-ones mask (`:253-265`).
//!
//! # The step
//!
//! ```text
//! h0   = E[ids]                              (frozen embedding, held on the device)
//! h_l  = layer_l(h_{l-1})            l = 0..27 (FrontTune forward + BlockTune forward)
//! dh28 = KdHead(h28_s, h28_t).dh              cross-entropy gradient, 1/T included
//! dh_l = layer_l.backward(dh_{l+1})  l = 27..0
//! s    = AdamW(grad s, step)                 392 scale vectors, latent rate 0
//! ```
//!
//! # Where the memory goes, and why the layers are swapped
//!
//! One [`ChainLayer`] holds the tape's parameter and activation buffers for one layer at `T = 2048`;
//! its size is fixed by the geometry, not by the layer index, and every Qwen3-0.6B layer has the
//! same geometry and the same four ranks (`dump/ranks.json`: `q` 768, `k`/`v` 576, `o` 768,
//! `gate`/`up`/`down` 864), so one trainer serves every layer. [`KdTrain::memory_line`] reports the
//! measured per-trainer bytes beside the free device memory, and [`KdTrain::new`] takes the number
//! of trainers that fit.
//!
//! Every layer's parameter image (`O(Σ (out + inn)·(rank + 1))` words, 70.9 MiB per layer for
//! Qwen3-0.6B, 1.94 GiB for 28) is device-resident in [`KdTrain`]'s store. A trainer takes the layer
//! it is scheduled on by one submission of device copies and moment fills
//! ([`ChainLayer::record_swap`], the device form of [`ChainLayer::load`]); the host uploads only
//! what moved — the whole image after [`KdTrain::set_pars`] or [`KdTrain::pars_mut`], the 22 528
//! scale words after an Adam step. The host upload this replaces was 62.78 ms per layer, 3.52 s of a
//! 5.368 s step (`~/nq-ref/kd/receipts/kd-layer-wall.log`).
//!
//! # Where the Adam step runs
//!
//! The device tapes' own Adam steps are built at `lr = lr_scale = 0`, so they are exact no-ops on
//! every parameter, and the scale update is this module's host arithmetic over the downloaded
//! `s_post`/`s_pre` gradients: 22 528 values per layer, four vectors per linear. That is the only
//! per-step host arithmetic, and it keeps the two moment vectors of all 392 scales in one place, so
//! a resumed round restores them exactly. `AdamHyper::OPTIMI` is the reference's constant set; the
//! documented `ε` deviation of the GPU kernel does not apply here because the update is not run on
//! the device.
//!
//! # Evidence grade and trust boundary
//!
//! `tested` by `kd_step0_golden` in `tests/learn_kd.rs`. Not covered: the GPU driver and the SPIR-V
//! compiler; the bf16 parameter precision of the reference (our parameters and moments are f32,
//! its optimiser state is bf16 — `optimi/adam.py:99-101`); and the teacher, whose `h28` is the F16
//! engine's stream where the reference's is a bf16 `transformers` forward.

use crate::gpu::decode::GpuCtx;
use crate::gpu::vulkan::*;

use super::block::{
    host_buffer, stream_buffer, stream_download, stream_upload, BlockShape, HostBuffer, HostFactor,
};
use super::chain::{copy_words, ChainLayer};
use super::front::FrontShape;
use super::head::KdHead;

/// The reference's seven linears per layer in its own order
/// (`src/nanoquant/utils/utils.py:77-88`, `tests/learn_block_tune.rs` `SLOT_KEYS`).
pub const KEYS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.k_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// The three linears of the attention front, in [`FrontTune`](super::front::FrontTune)'s `[q, k, v]`
/// order.
pub const FRONT_KEYS: [&str; 3] = [KEYS[0], KEYS[3], KEYS[1]];

/// The four linears of the residual block, in
/// [`BlockTune`](super::block::BlockTune)'s `[o, gate, up, down]` order.
pub const BLOCK_KEYS: [&str; 4] = [KEYS[2], KEYS[4], KEYS[5], KEYS[6]];

/// One layer's frozen and trainable parameters as the tapes want them: the front's three factors
/// and its two norm gains, the block's four factors and its FFN gain.
#[derive(Clone, Debug)]
pub struct LayerPars {
    /// `[q, k, v]`.
    pub ff: [HostFactor; 3],
    /// The attention norm's gain, `d` words.
    pub g1: Vec<f32>,
    /// The per-head `q` then `k` gains, `2·hd` words.
    pub gn: Vec<f32>,
    /// `[o, gate, up, down]`.
    pub bf: [HostFactor; 4],
    /// The FFN norm's gain, `d` words.
    pub g2: Vec<f32>,
}

/// One `.param` file's header and payload: a dense `W` (`out × inn`) or a factor's two latents and
/// two scale vectors (`tests/learn_block_tune.rs:686-742` `save_param` / `load_param`).
///
/// # Complexity
///
/// Time `O(out·inn + (out + inn)·rank)` file bytes read. Space the same in f32.
fn read_param(path: &str) -> Result<(usize, usize, Option<HostFactor>, Vec<f32>), String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let word = |i: usize| -> Result<usize, String> {
        b.get(8 * i..8 * i + 8)
            .ok_or_else(|| format!("{path}: short header"))?
            .try_into()
            .map(|w| u64::from_le_bytes(w) as usize)
            .map_err(|_| format!("{path}: word {i}"))
    };
    let (out, inn, rank, kind) = (word(0)?, word(1)?, word(2)?, word(3)?);
    let vals = |at: usize, n: usize| -> Result<Vec<f32>, String> {
        let s = b
            .get(32 + 4 * at..32 + 4 * (at + n))
            .ok_or_else(|| format!("{path}: payload short at {at}+{n}"))?;
        Ok(s.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("f32")))
            .collect())
    };
    match kind {
        0 => Ok((out, inn, None, vals(0, out * inn)?)),
        1 => {
            let (u, vt) = (vals(0, out * rank)?, vals(out * rank, inn * rank)?);
            let at = (out + inn) * rank;
            Ok((
                out,
                inn,
                Some(HostFactor {
                    out,
                    inn,
                    rank,
                    u,
                    vt,
                    s_post: vals(at, out)?,
                    s_pre: vals(at + out, inn)?,
                }),
                Vec::new(),
            ))
        }
        k => Err(format!("{path}: kind {k}")),
    }
}

impl LayerPars {
    /// The device image [`ChainLayer::record_swap`] reads, as its pieces in order, each marked
    /// `true` when it is a scale vector (the only part the KD step trains): the front's
    /// `[q, k, v]` as `[u, vt, s_post, s_pre]`, `g1`, `gn`, then the block's `[o, gate, up, down]`
    /// the same way and `g2`.
    ///
    /// # Complexity
    ///
    /// `O(1)`: 31 slices.
    fn image(&self) -> Vec<(&[f32], bool)> {
        fn fac(f: &HostFactor) -> [(&[f32], bool); 4] {
            [
                (f.u.as_slice(), false),
                (f.vt.as_slice(), false),
                (f.s_post.as_slice(), true),
                (f.s_pre.as_slice(), true),
            ]
        }
        let mut v = Vec::with_capacity(31);
        self.ff.iter().for_each(|f| v.extend(fac(f)));
        v.push((self.g1.as_slice(), false));
        v.push((self.gn.as_slice(), false));
        self.bf.iter().for_each(|f| v.extend(fac(f)));
        v.push((self.g2.as_slice(), false));
        v
    }

    /// Layer `l`'s seven linears out of `dir` (`<dir>/{key}.param`), the gains from `gains` (the
    /// front's `[q | k]` per-head gains and the two norm gains, `mixer::LayerHost`).
    ///
    /// # Errors
    ///
    /// A missing or malformed `.param` file, or a dense slot where the block loop wrote a factor.
    ///
    /// # Complexity
    ///
    /// `O(Σ out·inn + Σ (out + inn)·rank)` file bytes; 70 MiB at Qwen3-0.6B's geometry.
    pub fn load(dir: &str, g1: &[f32], gn: &[f32], g2: &[f32]) -> Result<Self, String> {
        let factor = |key: &str| -> Result<HostFactor, String> {
            match read_param(&format!("{dir}/{key}.param"))? {
                (_, _, Some(h), _) => Ok(h),
                (o, i, None, _) => Err(format!("{dir}/{key}.param: dense {o}×{i}, not a factor")),
            }
        };
        Ok(Self {
            ff: [
                factor(FRONT_KEYS[0])?,
                factor(FRONT_KEYS[1])?,
                factor(FRONT_KEYS[2])?,
            ],
            g1: g1.to_vec(),
            gn: gn.to_vec(),
            bf: [
                factor(BLOCK_KEYS[0])?,
                factor(BLOCK_KEYS[1])?,
                factor(BLOCK_KEYS[2])?,
                factor(BLOCK_KEYS[3])?,
            ],
            g2: g2.to_vec(),
        })
    }

    /// Write the seven linears back into `dir` as `.param` files, `kind` 1, `U`/`V` unchanged and the
    /// two scale vectors as this value holds them. The `nq_student` splice and `eval_arm.py` read
    /// this format unchanged.
    ///
    /// # Errors
    ///
    /// The directory cannot be created or a file cannot be written.
    ///
    /// # Complexity
    ///
    /// `O(Σ (out + inn)·rank)` bytes written, 70 MiB at Qwen3-0.6B's geometry.
    pub fn save(&self, dir: &str) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
        let put = |key: &str, h: &HostFactor| -> Result<(), String> {
            let mut b: Vec<u8> = [h.out, h.inn, h.rank, 1]
                .iter()
                .flat_map(|x| (*x as u64).to_le_bytes())
                .collect();
            for v in [&h.u, &h.vt, &h.s_post, &h.s_pre] {
                b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            }
            std::fs::write(format!("{dir}/{key}.param"), b).map_err(|e| format!("{dir}/{key}: {e}"))
        };
        for k in 0..3 {
            put(FRONT_KEYS[k], &self.ff[k])?;
        }
        for k in 0..4 {
            put(BLOCK_KEYS[k], &self.bf[k])?;
        }
        Ok(())
    }

    /// Every scale vector of the layer in a fixed order: `(slot, pre?)` → words. Slot majors the
    /// reference's seven linears, then `scale_pre` before `scale_post`.
    ///
    /// # Complexity
    ///
    /// `O(Σ (out + inn))` = 22 528 words at Qwen3-0.6B's geometry.
    pub fn scales(&self) -> Vec<Vec<f32>> {
        let mut v = Vec::with_capacity(14);
        for k in 0..3 {
            v.push(self.ff[k].s_pre.clone());
            v.push(self.ff[k].s_post.clone());
        }
        for k in 0..4 {
            v.push(self.bf[k].s_pre.clone());
            v.push(self.bf[k].s_post.clone());
        }
        v
    }

    /// Replace the layer's scale vectors from [`LayerPars::scales`]' order.
    ///
    /// # Complexity
    ///
    /// `O(Σ (out + inn))` words copied.
    pub fn set_scales(&mut self, s: &[Vec<f32>]) {
        assert_eq!(s.len(), 14, "kd: fourteen scale vectors per layer");
        for k in 0..3 {
            self.ff[k].s_pre.copy_from_slice(&s[2 * k]);
            self.ff[k].s_post.copy_from_slice(&s[2 * k + 1]);
        }
        for k in 0..4 {
            self.bf[k].s_pre.copy_from_slice(&s[6 + 2 * k]);
            self.bf[k].s_post.copy_from_slice(&s[6 + 2 * k + 1]);
        }
    }
}

/// The Adam state of every scale vector: the first and second moment in [`LayerPars::scales`]'
/// order, so a resumed round restores them beside the values.
#[derive(Clone, Debug, Default)]
pub struct ScaleAdam {
    /// `[layer][vector]`, each `m`.
    pub m: Vec<Vec<Vec<f32>>>,
    /// `[layer][vector]`, each `v`.
    pub v: Vec<Vec<Vec<f32>>>,
}

impl ScaleAdam {
    /// Allocate zero moments for `layers` layers of `scales`-shaped vectors.
    ///
    /// # Complexity
    ///
    /// `O(layers · Σ (out + inn))` words, 2.5 MiB per moment at Qwen3-0.6B's geometry.
    #[must_use]
    pub fn zero(layers: usize, scales: &[Vec<f32>]) -> Self {
        let z = |n: usize| vec![0f32; n];
        let per = |s: &[Vec<f32>]| s.iter().map(|x| z(x.len())).collect::<Vec<_>>();
        Self {
            m: (0..layers).map(|_| per(scales)).collect(),
            v: (0..layers).map(|_| per(scales)).collect(),
        }
    }
}

/// The one-sample distillation step over all 28 layers.
pub struct KdTrain {
    /// `resident` trainers; layer `l` runs on `l % resident`.
    layers: Vec<ChainLayer>,
    /// Which layer's parameters each trainer currently holds, if any.
    held: Vec<Option<usize>>,
    /// Every layer's [`LayerPars::image`] on the device, `28 × 70.9 MiB` at Qwen3-0.6B's geometry:
    /// a trainer takes a layer by one device copy instead of a host upload.
    store: Vec<Handle>,
    /// Which part of each layer's `store` image is behind `pars`.
    stale: Vec<Stale>,
    /// The host stage a stale image goes up through, one image long.
    stage: HostBuffer,
    /// Every layer's parameters on the host: `28 × 70 MiB` at Qwen3-0.6B's geometry.
    pars: Vec<LayerPars>,
    shapes: KdShapes,
    head: KdHead,
    /// `T × d` device buffers: `h[l]` is layer `l`'s input residual, `l = 0..=L` (`h[L]` the final
    /// residual). `L + 1` buffers of 8 MiB.
    h: Vec<Handle>,
    /// The student's `h0` for every sample, `n_samples · T × d`, device-resident.
    h0: Handle,
    /// The teacher's `h28` for the current sample, `T × d`.
    ht: Handle,
    /// The device buffers a step's copies need, one submission each.
    xfer: crate::gpu::vulkan::CommandBuffer,
    /// `true` when the backward also downloads each layer's input cotangent: the check path only.
    capture: bool,
}

/// The part of a layer's device image that differs from its host parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stale {
    /// The whole image: the parameters were set or lent out mutably.
    All,
    /// The scale vectors: an Adam step moved them.
    Scales,
    /// Nothing.
    Fresh,
}

/// One backward's result: every layer's scale gradients and, when [`KdTrain::capture_dh`] is on,
/// every layer's input cotangent.
pub struct Bwd {
    /// `[layer][vector]`, `[s_post, s_pre]` in [`LayerPars::scales`]' order.
    pub scales: Vec<Vec<[Vec<f32>; 2]>>,
    /// `[layer]`, `dL/dx_l` as `T × d`, empty unless the capture is on.
    pub dh: Vec<Vec<f32>>,
}

/// The geometry one step runs at: the layer count, the sample's token count and the model's widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdShapes {
    /// Layers (`28`).
    pub layers: usize,
    /// Tokens per sample (`2048`).
    pub t: usize,
    /// Residual width (`1024`).
    pub d: usize,
    /// Vocabulary (`151936`).
    pub v: usize,
}

impl KdTrain {
    /// The per-trainer device bytes and the free device memory, as the choice of `resident` reads
    /// them. `bytes` is `g.alloc`'s delta across one [`ChainLayer`]; the caller measures it, this
    /// only formats the line.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn memory_line(bytes: u64, free_mib: u64, layers: usize, d: usize, v: usize) -> String {
        let pool = layers as u64 * d as u64 * 2048 * 4 * 2;
        let train = layers as u64 * bytes;
        format!(
            "KD MEM one trainer {:.1} MiB; {} trainers {:.2} GiB; free {free_mib} MiB; teacher \
             stream 8 MiB/step; h0 pool {:.2} GiB; parameter pool (host) {:.2} GiB; embedding f32 \
             {:.2} GiB",
            bytes as f64 / (1 << 20) as f64,
            layers,
            train as f64 / (1u64 << 30) as f64,
            pool as f64 / (1u64 << 30) as f64,
            layers as f64 * 70.0 / 1024.0,
            v as f64 * d as f64 * 4.0 / (1u64 << 30) as f64,
        )
    }

    /// Allocate the step: `resident` layer trainers, the residual buffers, the teacher's cotangent
    /// target and the cross-entropy head, all device-resident.
    ///
    /// `pars` is the student's 28 layers, `h0` its `layers · T × d` embedding output for every
    /// sample in `order`, `w_emb`/`g_final` the frozen tied head and final-norm gain.
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value; `w_emb` is `v × d` f32, `g_final` `d`
    /// f32, `h0` `order.len() · T × d` f32; every `.param` factor matches the shapes' ranks.
    ///
    /// # Complexity
    ///
    /// `resident` trainers at `O(t·(d + a + f) + Σ out·inn)` words each, `L + 1` residual buffers,
    /// one teacher buffer and the head's `O(rc·(d + v))` tiles.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "one step; the dimensions index u32 dispatch shapes"
    )]
    #[must_use]
    pub unsafe fn new(
        g: &GpuCtx,
        shapes: KdShapes,
        fs: FrontShape,
        bs: BlockShape,
        pars: Vec<LayerPars>,
        w_emb: Handle,
        g_final: Handle,
        h0: Handle,
        rc: u32,
        resident: usize,
    ) -> Self {
        assert_eq!(
            pars.len(),
            shapes.layers,
            "kd: one parameter image per layer"
        );
        assert!(resident >= 1, "kd: at least one layer trainer");
        let t = shapes.t;
        let d = shapes.d;
        let mut layers = Vec::with_capacity(resident);
        for _ in 0..resident {
            layers.push(ChainLayer::new(g, fs, bs, 0.0, 0.0, 0.0));
        }
        let mut h = Vec::with_capacity(shapes.layers + 1);
        for _ in 0..=shapes.layers {
            h.push(stream_buffer(g, t * d));
        }
        let ht = stream_buffer(g, t * d);
        let words: usize = pars[0].image().iter().map(|(p, _)| p.len()).sum();
        let store = (0..shapes.layers)
            .map(|_| stream_buffer(g, words))
            .collect();
        let stage = host_buffer(g, words, BUFFER_USAGE_TRANSFER_SRC);
        let head = KdHead::new(
            g,
            t as u32,
            d as u32,
            shapes.v as u32,
            rc,
            w_emb,
            g_final,
            h[shapes.layers],
            ht,
        );
        Self {
            held: (0..resident).map(|_| None).collect(),
            store,
            stale: vec![Stale::All; shapes.layers],
            stage,
            layers,
            pars,
            shapes,
            head,
            h,
            h0,
            ht,
            xfer: g.vk.reusable_command_buffer(),
            capture: false,
        }
    }

    /// Copy `words` f32 from `src` at word `src_w` to `dst` at word `dst_w` on the device.
    ///
    /// # Safety
    ///
    /// Both buffers hold their ranges; no submission of `cb` is pending.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; one submission copying `O(words)` bytes.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a word offset times four indexes a u64 byte offset"
    )]
    unsafe fn copy_at(
        &self,
        g: &GpuCtx,
        src: Handle,
        src_w: usize,
        dst: Handle,
        dst_w: usize,
        words: usize,
    ) {
        let vk = g.vk;
        let cb = self.xfer;
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
            src_offset: (src_w * 4) as u64,
            dst_offset: (dst_w * 4) as u64,
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

    /// Record that the host parameters moved by `s`: every trainer must swap its layer in again,
    /// and every device image is behind by at least `s`.
    ///
    /// # Complexity
    ///
    /// `O(L)`.
    fn mark(&mut self, s: Stale) {
        self.held.iter_mut().for_each(|h| *h = None);
        for st in &mut self.stale {
            if *st != Stale::All {
                *st = s;
            }
        }
    }

    /// Bring layer `l`'s device image up to its host parameters: the whole image when
    /// [`Stale::All`], its 14 scale vectors when [`Stale::Scales`], through one submission.
    ///
    /// # Safety
    ///
    /// No submission is pending.
    ///
    /// # Complexity
    ///
    /// `O(words)` host words staged and uploaded: 18.6 M at `All`, 22 528 at `Scales`.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "word offsets index a u64 byte offset"
    )]
    unsafe fn refresh(&mut self, g: &GpuCtx, l: usize) {
        let all = match self.stale[l] {
            Stale::Fresh => return,
            Stale::All => true,
            Stale::Scales => false,
        };
        let (mut at, mut put) = (0usize, 0usize);
        let mut regions = Vec::with_capacity(31);
        for (p, scale) in self.pars[l].image() {
            if all || scale {
                std::ptr::copy_nonoverlapping(p.as_ptr(), self.stage.ptr.add(put), p.len());
                regions.push(BufferCopy {
                    src_offset: (put * 4) as u64,
                    dst_offset: (at * 4) as u64,
                    size: (p.len() * 4) as u64,
                });
                put += p.len();
            }
            at += p.len();
        }
        let vk = g.vk;
        let cb = self.xfer;
        vk.reset_cb(cb);
        vk.begin(cb);
        let n = regions.len() as u32;
        (vk.copy_buffer)(cb, self.stage.buf, self.store[l], n, regions.as_ptr());
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        self.stale[l] = Stale::Fresh;
    }

    /// Put layer `l` into the trainer it runs on, if that trainer holds another layer: the device
    /// form of [`ChainLayer::load`], its image copied from `store` and its moments zeroed in one
    /// submission.
    ///
    /// # Safety
    ///
    /// No tape is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work and one submission moving the image's 18.6 M words on the device, plus
    /// [`KdTrain::refresh`]'s upload when the image is stale.
    unsafe fn use_layer(&mut self, g: &GpuCtx, l: usize) {
        let slot = l % self.layers.len();
        if self.held[slot] == Some(l) {
            return;
        }
        self.refresh(g, l);
        let vk = g.vk;
        let cb = self.xfer;
        vk.reset_cb(cb);
        vk.begin(cb);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_TRANSFER,
            ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
            ACCESS_TRANSFER_WRITE,
        );
        let words = self.layers[slot].record_swap(vk, cb, self.store[l]);
        assert_eq!(words, self.stage.n, "kd use_layer: image layout");
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_COMPUTE,
            ACCESS_TRANSFER_WRITE,
            ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
        );
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        self.held[slot] = Some(l);
    }

    /// Start a step on sample `s` of `self.order`: the student's `h0` slice onto `h[0]`, the
    /// teacher's `h28` for that sample onto the head's target.
    ///
    /// # Safety
    ///
    /// `s` is in `self.order`, `h_t` is the `T × d` teacher residual at the final norm input for
    /// that sample, and no submission is pending.
    ///
    /// # Complexity
    ///
    /// `O(t·d)` words copied on the device and `O(t·d)` uploaded.
    pub unsafe fn begin(&mut self, g: &GpuCtx, s: usize, h_t: &[f32]) {
        let n = self.shapes.t * self.shapes.d;
        assert_eq!(h_t.len(), n, "kd: teacher residual is T × d");
        self.copy_at(g, self.h0, s * n, self.h[0], 0, n);
        stream_upload(g, self.ht, h_t);
    }

    /// The forward through all `L` layers: `h[l]` in, `h[l + 1]` out, each layer's trainer holding
    /// its own parameters.
    ///
    /// # Safety
    ///
    /// [`KdTrain::begin`] has run; no tape is pending.
    ///
    /// # Complexity
    ///
    /// `Σ_l` the front's `O(t·n_ctx·nh·hd + t·d·(nh·hd + nkv·hd) + Σ out·inn·rank)` and the block's
    /// `O(t·(a·d + 3·d·f) + Σ out·inn·rank)` multiply-adds, plus `L` parameter loads when
    /// `resident < L` and `2·L·t·d` words copied.
    pub unsafe fn forward(&mut self, g: &GpuCtx) {
        let (t, d) = (self.shapes.t, self.shapes.d);
        for l in 0..self.shapes.layers {
            self.use_layer(g, l);
            let slot = l % self.layers.len();
            let ch = &mut self.layers[slot];
            Self::put_residual(g, self.xfer, self.h[l], ch.x(), t * d);
            ch.forward(g);
            Self::put_residual(g, self.xfer, ch.y(), self.h[l + 1], t * d);
        }
    }

    /// One device copy on `cb`, as [`ChainLayer`] makes for its own hand-offs.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; one submission.
    unsafe fn put_residual(
        g: &GpuCtx,
        cb: crate::gpu::vulkan::CommandBuffer,
        src: Handle,
        dst: Handle,
        words: usize,
    ) {
        copy_words(g, cb, src, dst, words);
    }

    /// The head: `dh28 = KdHead(h28_s, h28_t).dh` and the step's scalar loss.
    ///
    /// # Safety
    ///
    /// [`KdTrain::forward`] has run; [`KdTrain::begin`] put the teacher's residual on the head's
    /// target.
    ///
    /// # Complexity
    ///
    /// `O(t·d·v)` multiply-adds over the chunked vocabulary, and the `O(t)` loss readback — the
    /// step's only host read besides the teacher upload.
    pub unsafe fn head(&mut self, g: &GpuCtx) -> f64 {
        self.head.run(g);
        self.head.loss(g)
    }

    /// The backward through all `L` layers from the head's cotangent, leaving `dh[l]` in
    /// `self.h[l]`'s slot: layer `l`'s input cotangent `dL/dx_l` for `l = 0..L`.
    ///
    /// Returns the scale gradients per layer in [`LayerPars::scales`]' order: `[layer][vector]`,
    /// `[s_post, s_pre]`.
    ///
    /// # Safety
    ///
    /// [`KdTrain::head`] has run; no tape is pending.
    ///
    /// # Complexity
    ///
    /// `Σ_l` three times the block's and the front's forward multiply-adds (each layer's forward is
    /// re-run from its saved input before its backward), `L` parameter loads and
    /// `O(Σ (out + inn))` scale-gradient words downloaded per layer.
    pub unsafe fn backward(&mut self, g: &GpuCtx) -> Bwd {
        let (t, d, nl) = (self.shapes.t, self.shapes.d, self.shapes.layers);
        let mut out = Vec::with_capacity(nl);
        let mut dh = Vec::with_capacity(nl);
        for _ in 0..nl {
            out.push(Vec::new());
            dh.push(Vec::new());
        }
        for l in (0..nl).rev() {
            self.use_layer(g, l);
            let slot = l % self.layers.len();
            // The cotangent `dy` this layer's block reads: the head's for the last layer, else the
            // next layer's `dx`, which a single trainer leaves in its own buffer.
            let dy_in = if l + 1 == nl {
                self.head.dh()
            } else {
                self.layers[(l + 1) % self.layers.len()].dx()
            };
            let dy = self.layers[slot].dy();
            Self::put_residual(g, self.xfer, dy_in, dy, t * d);
            // The trainer holds one layer's buffers, so the front's `q`, `k`, `v` and `lse` are the
            // last layer the forward ran, not this one: the layer's own forward is re-run from its
            // saved input before its backward, which is the reference's gradient checkpointing
            // (`compress_model.py:217-220`).
            Self::put_residual(g, self.xfer, self.h[l], self.layers[slot].x(), t * d);
            let ch = &mut self.layers[slot];
            ch.forward(g);
            ch.backward(g);
            let fg = ch.front().scale_grads(g);
            let bg = ch.block().scale_grads(g);
            for k in 0..3 {
                out[l].push(fg[k].clone());
            }
            for k in 0..4 {
                out[l].push(bg[k].clone());
            }
            if self.capture {
                dh[l] = stream_download(g, ch.dx(), t * d);
            }
        }
        Bwd { scales: out, dh }
    }

    /// Download each layer's input cotangent during the backward (the check path's switch; off, a
    /// step reads back nothing but the loss).
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; `O(L·T·d)` words downloaded per step when on.
    pub fn capture_dh(&mut self, on: bool) {
        self.capture = on;
    }

    /// The residual at layer `l`'s input as the forward left it (`l = 0` is `h0`, `l = L` the final
    /// residual).
    ///
    /// # Safety
    ///
    /// [`KdTrain::forward`] has run.
    ///
    /// # Complexity
    ///
    /// `O(t·d)` words downloaded.
    pub unsafe fn residual(&self, g: &GpuCtx, l: usize) -> Vec<f32> {
        stream_download(g, self.h[l], self.shapes.t * self.shapes.d)
    }

    /// The head's cotangent at the final residual, `dL/dh_28` as `T × d`.
    ///
    /// # Safety
    ///
    /// [`KdTrain::head`] has run.
    ///
    /// # Complexity
    ///
    /// `O(t·d)` words downloaded.
    pub unsafe fn head_dh(&self, g: &GpuCtx) -> Vec<f32> {
        stream_download(g, self.head.dh(), self.shapes.t * self.shapes.d)
    }

    /// The teacher's `h28` buffer (the head's target).
    #[must_use]
    pub fn teacher(&self) -> Handle {
        self.ht
    }

    /// Set `self.h[l]` from a host slice: the `h_l`/`dh_l` inputs a check compares against.
    ///
    /// # Safety
    ///
    /// `x` is `T × d`; no submission is pending.
    ///
    /// # Complexity
    ///
    /// `O(t·d)` words uploaded.
    pub unsafe fn set_residual(&self, g: &GpuCtx, l: usize, x: &[f32]) {
        stream_upload(g, self.h[l], x);
    }

    /// Set the student's 28 layer images.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    pub fn set_pars(&mut self, pars: Vec<LayerPars>) {
        assert_eq!(pars.len(), self.shapes.layers, "kd: one image per layer");
        self.pars = pars;
        self.mark(Stale::All);
    }

    /// The student's layer images, the read side of [`KdTrain::set_pars`].
    #[must_use]
    pub fn pars(&self) -> &[LayerPars] {
        &self.pars
    }

    /// The student's layer images mutably, for the returned Adam step's result.
    pub fn pars_mut(&mut self) -> &mut [LayerPars] {
        self.mark(Stale::All);
        &mut self.pars
    }

    /// One AdamW step over every scale vector at base rate `lr`, the recurrent form torch's
    /// `AdamW` takes with no weight decay (`torch/optim/adamw.py:161-176`):
    ///
    /// ```text
    /// m ← β₁m + (1−β₁)g   v ← β₂v + (1−β₂)g²
    /// m̂ = m/(1−β₁ᵗ)       v̂ = v/(1−β₂ᵗ)
    /// s ← s − lr·m̂/(√v̂ + ε)
    /// ```
    ///
    /// `t` is the 1-based step and `(β₁, β₂, ε) = (0.9, 0.99, 1e-6)` are the reference's
    /// (`compress_model.py:238`; `optimi.AdamW`'s defaults, `optimi/adamw.py:30-44`). No weight
    /// decay: `decouple_wd=True` multiplies by `1 − 1e-7`, the identity on a bf16 parameter
    /// (`optimi/adamw.py:67`, `optimi/adam.py:372-375`).
    ///
    /// # Complexity
    ///
    /// `O(L · Σ (out + inn))` f64 operations and f32 stores — 630 784 values per step — plus one
    /// `Vec` per scale vector.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the parameter is f32 and the update is computed in f64 once"
    )]
    pub fn adam(
        &mut self,
        adam: &mut ScaleAdam,
        scale_grads: &[Vec<[Vec<f32>; 2]>],
        t: usize,
        lr: f64,
    ) {
        const B1: f64 = 0.9;
        const B2: f64 = 0.99;
        const EPS: f64 = 1e-6;
        assert!(t >= 1, "kd adam: the step ordinal is 1-based");
        let (b1t, b2t) = (B1.powi(t as i32), B2.powi(t as i32));
        for l in 0..self.shapes.layers {
            assert_eq!(scale_grads[l].len(), 7, "kd adam: seven linears per layer");
            let mut s = self.pars[l].scales();
            for j in 0..s.len() {
                // `scales()` holds `s_pre` then `s_post`; the tape's grads hold `[post, pre]`.
                let g = &scale_grads[l][j >> 1][1 - (j & 1)];
                assert_eq!(g.len(), s[j].len(), "kd adam: gradient width");
                let (m, v) = (&mut adam.m[l][j], &mut adam.v[l][j]);
                for i in 0..s[j].len() {
                    let gi = f64::from(g[i]);
                    m[i] = (B1 * f64::from(m[i]) + (1.0 - B1) * gi) as f32;
                    v[i] = (B2 * f64::from(v[i]) + (1.0 - B2) * gi * gi) as f32;
                    let mh = f64::from(m[i]) / (1.0 - b1t);
                    let vh = f64::from(v[i]) / (1.0 - b2t);
                    s[j][i] = (f64::from(s[j][i]) - lr * mh / (vh.sqrt() + EPS)) as f32;
                }
            }
            self.pars[l].set_scales(&s);
        }
        self.mark(Stale::Scales);
    }

    /// Write every layer's state into `dir` as `l<k>/{key}.param`, the `loop_words` format the
    /// `nq_student` splice and `eval_arm.py` read, with `U`/`V` unchanged and the new scales.
    ///
    /// # Errors
    ///
    /// A layer's directory cannot be written.
    ///
    /// # Complexity
    ///
    /// `O(L · Σ (out + inn)·rank)` bytes written, 1.9 GiB at Qwen3-0.6B's geometry.
    pub fn export(&self, dir: &str) -> Result<(), String> {
        for (l, p) in self.pars.iter().enumerate() {
            p.save(&format!("{dir}/l{l}"))?;
        }
        std::fs::write(format!("{dir}/done"), b"1\n").map_err(|e| format!("{dir}/done: {e}"))
    }

    /// The resume state: the step counter, every scale vector, both moments. One f32 file, `state`,
    /// so a 1024-step run splits into rounds without a re-encode.
    ///
    /// # Errors
    ///
    /// The file cannot be written.
    ///
    /// # Complexity
    ///
    /// `O(1 + L · Σ (out + inn))` f32 written, 7.6 MiB for three of them at Qwen3-0.6B.
    pub fn save_state(
        &self,
        dir: &str,
        adam: &ScaleAdam,
        step: usize,
        order: &[usize],
    ) -> Result<(), String> {
        let mut b = Vec::new();
        for v in [step, order.len()]
            .into_iter()
            .chain(order.iter().copied())
            .chain(
                self.pars
                    .iter()
                    .flat_map(|p| p.scales().into_iter().map(|s| s.len())),
            )
        {
            b.extend((v as u64).to_le_bytes());
        }
        for l in 0..self.pars.len() {
            for s in self.pars[l].scales() {
                b.extend(s.iter().flat_map(|x| x.to_le_bytes()));
            }
        }
        for l in 0..self.pars.len() {
            for v in [&adam.m[l], &adam.v[l]] {
                for s in v {
                    b.extend(s.iter().flat_map(|x| x.to_le_bytes()));
                }
            }
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
        std::fs::write(format!("{dir}/state"), b).map_err(|e| format!("{dir}/state: {e}"))
    }

    /// Read what [`KdTrain::save_state`] wrote. Returns `(step, order, adam)`; the `order` is
    /// checked against the live draw list by the caller.
    ///
    /// # Errors
    ///
    /// The file is absent or malformed.
    ///
    /// # Complexity
    ///
    /// `O(1 + L · Σ (out + inn))` f32 read.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the state's counters are small and its values are f32"
    )]
    pub fn load_state(&mut self, dir: &str) -> Result<(usize, Vec<usize>, ScaleAdam), String> {
        let b = std::fs::read(format!("{dir}/state")).map_err(|e| format!("{dir}/state: {e}"))?;
        let word = |i: usize| -> Result<usize, String> {
            b.get(8 * i..8 * i + 8)
                .ok_or_else(|| format!("{dir}/state: short"))?
                .try_into()
                .map(|w| u64::from_le_bytes(w) as usize)
                .map_err(|_| format!("{dir}/state: word {i}"))
        };
        let step = word(0)?;
        let n = word(1)?;
        let mut order = Vec::with_capacity(n);
        let mut at = 2;
        for _ in 0..n {
            order.push(word(at)?);
            at += 1;
        }
        let lens: Vec<Vec<usize>> = (0..self.pars.len())
            .map(|l| {
                let v = (0..14).map(|j| word(at + j)).collect::<Result<Vec<_>, _>>();
                at += 14;
                v
            })
            .collect::<Result<Vec<_>, _>>()?;
        let f32s = |from: usize, n: usize| -> Result<Vec<f32>, String> {
            b.get(from..from + 4 * n)
                .ok_or_else(|| format!("{dir}/state: short payload"))
                .map(|s| {
                    s.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().expect("f32")))
                        .collect()
                })
        };
        let mut from = 8 * at;
        for l in 0..self.pars.len() {
            let mut s = Vec::with_capacity(14);
            for &n in &lens[l] {
                s.push(f32s(from, n)?);
                from += 4 * n;
            }
            self.pars[l].set_scales(&s);
        }
        let mut adam = ScaleAdam::zero(self.pars.len(), &self.pars[0].scales());
        for l in 0..self.pars.len() {
            for v in 0..2 {
                for j in 0..14 {
                    let n = lens[l][j];
                    let got = f32s(from, n)?;
                    from += 4 * n;
                    let dst = if v == 0 { &mut adam.m } else { &mut adam.v };
                    dst[l][j] = got;
                }
            }
        }
        self.mark(Stale::Scales);
        Ok((step, order, adam))
    }
}

/// Build the front and block shapes of one Qwen3 layer.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn shapes_of(
    t: usize,
    d: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    rot: usize,
    theta: f32,
    eps: f32,
    a: usize,
    f: usize,
    ranks: [usize; 7],
) -> (FrontShape, BlockShape) {
    let attn = FrontShape {
        chunks: 1,
        n_ctx: t,
        d,
        nh,
        nkv,
        hd,
        rot,
        theta,
        eps,
        ranks: [ranks[0], ranks[3], ranks[1]],
    };
    let block = BlockShape {
        t,
        d,
        a,
        f,
        ranks: [ranks[2], ranks[4], ranks[5], ranks[6]],
    };
    (attn, block)
}

/// Read the raw f32 payload of `path` (the trace's little-endian dumps).
///
/// # Errors
///
/// The file cannot be read, or its length is not a whole number of f32.
///
/// # Complexity
///
/// `O(bytes)` read.
pub fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() % 4 != 0 {
        return Err(format!("{path}: {} bytes is not whole f32", b.len()));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("f32")))
        .collect())
}

/// The relative L2 error `‖got − want‖ / ‖want‖` over two equal-length slices in f64.
///
/// # Complexity
///
/// `O(n)`.
///
/// # Panics
///
/// If the lengths differ.
#[must_use]
pub fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "rel_l2 lengths");
    let (mut num, mut den) = (0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        let (a, b) = (f64::from(a), f64::from(b));
        num += (a - b) * (a - b);
        den += b * b;
    }
    (num / den.max(1e-30)).sqrt()
}

/// The cosine between two equal-length slices in f64.
///
/// # Complexity
///
/// `O(n)`.
///
/// # Panics
///
/// If the lengths differ.
#[must_use]
pub fn cosine(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "cosine lengths");
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in got.iter().zip(want) {
        let (x, y) = (f64::from(x), f64::from(y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt().max(1e-30)
}

/// The free device memory in MiB as `nvidia-smi` reports it, or `0` when it cannot be read.
///
/// # Complexity
///
/// `O(1)` (one process spawn).
#[must_use]
pub fn free_mib() -> u64 {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output();
    out.ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().and_then(|l| l.trim().parse().ok()))
        .unwrap_or(0)
}
