//! `GpuDecode`: the whole-model greedy decode context, ported from fold
//! `bedrock/src/serve/gpu.rs:945-1073` (the type) and `GpuDecode::new` `3195-3500` (the model-wide
//! tail), `step` `3503-3553`, `step_logits`/`last_argmax` `3602-3620`.
//!
//! One command buffer holds `embed -> n_layer x layer -> final norm (quantized + f32) -> LM head ->
//! two-stage argmax -> DMA of [logits | hidden | argmax] into one HOST_CACHED window`; `step`
//! writes `(pos, tok)` into the mapped params buffer and replays it.
//!
//! Restrictions of fold's generality: dense softmax attention only (no recurrent layers, MoE,
//! QK-norm, attention gate, knot or compact pool), the fused default path only (`DEC_FOLD`,
//! `DEC_FUSENORM`, `DEC_ACTQFUSE`, `DEC_COMBFUSE`, `DEC_ROPEFUSE` at 1), one `Carrier` for every
//! projection, the embedding and the head on the `dm` scale face (no two-level or superblock
//! face), no `DEC_PROF` timestamp pool, no `seed_kv` prefill bridge, no delta-net reset.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's bedrock/src/serve/gpu.rs; lint profile is fold's, not apeiron's"
)]

use super::layer::{embed_step, final_norm_step, rec_nodes, record_layer};
use super::state::{LayerW, Scratch, Stream};
use super::upload::{layer_cfg, upload_embed, upload_final_norm, upload_lm_head};
use super::*;
use crate::engine::loader::EngineConfig;
use crate::format::gguf::Gguf;

/// A device-resident decode context producing host logits, the device-side greedy token and the
/// final normed hidden per step.
pub struct GpuDecode {
    vk: Vk,
    cb: CommandBuffer,
    logits_map: *mut f32,
    params_map: *mut u32,
    hidden_map: *mut f32,
    argmax_map: *const u32,
    vocab: usize,
    hidden: usize,
    hout: Vec<f64>,
    /// KV capacity in positions.
    pub tmax: u32,
    /// Dispatches recorded into the replayed command buffer.
    pub n_disp: u32,
}

// SAFETY: `GpuDecode` owns its Vulkan context and its mapped host-visible buffers exclusively; the
// recorded command buffer is only submitted through `&mut self`. Not `Sync`, as in fold.
unsafe impl Send for GpuDecode {}

impl GpuDecode {
    /// Build the device context for `gguf` (config `ec`), KV capacity `tmax`, every projection on
    /// `carrier`. `Err` if Vulkan is unavailable or the model is outside the ported subset.
    pub fn new(
        gguf: &Gguf,
        ec: &EngineConfig,
        tmax: u32,
        carrier: Carrier,
    ) -> Result<GpuDecode, String> {
        unsafe {
            let vk = Vk::load()?;
            let c = layer_cfg(gguf, ec, tmax, carrier)?;
            let (hidden, vocab) = (c.hidden, ec.vocab_size as u32);
            let ((n_disp, logits_map, params_map), cb) = {
                let (g, _stage) = GpuCtx::new(&vk)?;
                let sc = Scratch::new(&g, &c)?;
                let embed = upload_embed(&g, gguf, &c, ec.vocab_size)?;
                let lm = upload_lm_head(&g, gguf, &c, ec.vocab_size, embed)?;
                let final_norm = upload_final_norm(&g, gguf)?;
                // The activation chart `final_rms` writes and the LM head reads (`gpu.rs:2269-2275`):
                // `hidden` is the head's k.
                let xq = g.dev(hidden as usize);
                let xs = g.dev((hidden / 32) as usize * 4);
                let sxb = g.dev((hidden / 32) as usize * 4);
                let (logits, _) = vk
                    .buffer(
                        vocab as u64 * 4,
                        BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC,
                        MEMORY_DEVICE_LOCAL,
                    )
                    .ok_or("logits alloc failed")?;
                let (hid_buf, _) = vk
                    .buffer(
                        hidden as u64 * 4,
                        BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC,
                        MEMORY_DEVICE_LOCAL,
                    )
                    .ok_or("hidden alloc failed")?;

                let embed_node = embed_step(&g, &c, vocab, embed.0, embed.1, &sc);
                let mut st = Stream::new(&sc, &c);
                let mut layers: Vec<Vec<Node>> = Vec::with_capacity(ec.n_layer);
                for li in 0..ec.n_layer {
                    let lw = LayerW::load(&g, gguf, li, &c)?;
                    layers.push(record_layer(&g, &c, &lw, &sc, li, &mut st));
                }
                let dpw_lm = c.head_carrier().pack() / 4;
                let final_rms = make_step_named(
                    &g,
                    &rmsnorm_add_q_seg_bytes(hidden, c.eps, dpw_lm, st.prev_ffn_segs),
                    &[st.cur, final_norm, sc.d, xq, sxb, xs],
                    1,
                    1,
                    "rmsnorm_add_q_seg",
                );
                let final_rms_f32 = final_norm_step(&g, &c, &st, &sc, final_norm, hid_buf);
                let lm_head = Node::Gemv(make_gemv(
                    &g,
                    c.head_carrier(),
                    lm.0,
                    lm.1,
                    xq,
                    (xs, sxb),
                    logits,
                    sc.partials,
                    sc.ones,
                    vocab,
                    hidden,
                    false,
                    ScaleFace::Dm32,
                ));
                let amaxp = g.dev(2 * AMAX_P as usize * 4);
                // 8 bytes: `argmax_finish` writes the winner at BYTE OFFSET 4 (`params[1]`'s slot).
                let amax_out = g.dev(8);
                let amax1 = make_step_named(
                    &g,
                    &argmax_partial_bytes(vocab, AMAX_P, 256),
                    &[logits, amaxp],
                    1,
                    1,
                    "argmax_partial",
                );
                let amax2 = make_step_named(
                    &g,
                    &argmax_finish_bytes(AMAX_P),
                    &[amaxp, amax_out],
                    1,
                    1,
                    "argmax_finish",
                );

                // `[vocab logits | hidden | argmax]` in one HOST_CACHED window.
                let rbytes = (ec.vocab_size + hidden as usize + 1) as u64 * 4;
                let cached = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT | MEMORY_HOST_CACHED;
                let plain = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
                let (rb, rb_mem) = match vk.buffer(rbytes, BUFFER_USAGE_TRANSFER_DST, cached) {
                    Some(x) => x,
                    None => vk
                        .buffer(rbytes, BUFFER_USAGE_TRANSFER_DST, plain)
                        .ok_or("readback alloc failed")?,
                };
                let logits_map = vk.map(rb_mem).cast::<f32>();

                let cb = vk.command_buffer();
                vk.begin(cb);
                let mut n_disp = rec_step(&vk, cb, &embed_node);
                for l in &layers {
                    n_disp += rec_nodes(&vk, cb, l);
                }
                n_disp += rec_step(&vk, cb, &final_rms);
                n_disp += rec_step(&vk, cb, &final_rms_f32);
                n_disp += rec_nodes(&vk, cb, std::slice::from_ref(&lm_head));
                n_disp += rec_step(&vk, cb, &amax1);
                n_disp += rec_step(&vk, cb, &amax2);
                vk.barrier(
                    cb,
                    STAGE_COMPUTE,
                    STAGE_TRANSFER,
                    ACCESS_SHADER_WRITE,
                    ACCESS_TRANSFER_READ,
                );
                let v4 = ec.vocab_size as u64 * 4;
                let copy = |src: Handle, so: u64, dst_off: u64, size: u64| {
                    let r = BufferCopy {
                        src_offset: so,
                        dst_offset: dst_off,
                        size,
                    };
                    (vk.copy_buffer)(cb, src, rb, 1, &r);
                };
                copy(logits, 0, 0, v4);
                copy(hid_buf, 0, v4, hidden as u64 * 4);
                copy(amax_out, 4, v4 + hidden as u64 * 4, 4);
                vk.barrier(
                    cb,
                    STAGE_TRANSFER,
                    STAGE_HOST,
                    ACCESS_TRANSFER_WRITE,
                    ACCESS_HOST_READ,
                );
                check((vk.end_cb)(cb), "vkEndCommandBuffer");
                ((n_disp, logits_map, sc.params_map), cb)
            };
            let hidden_map = logits_map.add(ec.vocab_size);
            let argmax_map = logits_map
                .add(ec.vocab_size + hidden as usize)
                .cast::<u32>();
            Ok(GpuDecode {
                vk,
                cb,
                logits_map,
                params_map,
                hidden_map,
                argmax_map,
                vocab: ec.vocab_size,
                hidden: hidden as usize,
                hout: vec![0f64; hidden as usize],
                tmax,
                n_disp,
            })
        }
    }

    /// One decode step: write `(pos, tok)`, replay the recorded chain, return the full logit row
    /// (the mapped HOST_CACHED window; valid until the next step).
    pub fn step(&mut self, pos: usize, tok: usize) -> &[f32] {
        assert!(
            (pos as u32) < self.tmax,
            "gpu decode: pos {pos} exceeds tmax {}",
            self.tmax
        );
        unsafe {
            self.params_map.write_volatile(pos as u32);
            self.params_map.add(1).write_volatile(tok as u32);
            self.vk.submit_wait(self.cb);
            for (o, i) in self.hout.iter_mut().zip(0..self.hidden) {
                *o = *self.hidden_map.add(i) as f64;
            }
            std::slice::from_raw_parts(self.logits_map, self.vocab)
        }
    }

    /// The full logit row for the step just run.
    pub fn step_logits(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.logits_map, self.vocab) }
    }

    /// The greedy token of the step just run, computed on the device (lowest-index tie-break).
    pub fn last_argmax(&self) -> usize {
        unsafe { self.argmax_map.read() as usize }
    }

    /// The final normed hidden of the step just run.
    pub fn last_hidden(&self) -> &[f64] {
        &self.hout
    }
}
