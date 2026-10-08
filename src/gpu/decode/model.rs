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
//! face), no `seed_kv` prefill bridge, no delta-net reset.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's bedrock/src/serve/gpu.rs; lint profile is fold's, not apeiron's"
)]

use super::layer::{embed_step, final_norm_step, rec_nodes, record_layer};
use super::prefill::PrefillStack;
use super::state::{LayerW, Scratch, Stream};
use super::upload::{layer_cfg, upload_embed, upload_final_norm, upload_lm_head};
use super::*;
use crate::engine::loader::EngineConfig;
use crate::format::gguf::Gguf;
use crate::gpu::vulkan::QueryPool;

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
    prefill: Option<PrefillStack>,
    /// Greedy token of the last [`GpuDecode::prefill`] until the next [`GpuDecode::step`].
    pf_argmax: Option<usize>,
    prof_pool: Option<QueryPool>,
    prof_labels: Vec<String>,
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
        Self::new_with_prefill(gguf, ec, tmax, carrier, 0)
    }

    /// [`GpuDecode::new`] plus a whole-prompt prefill plan for prompts of up to `maxb` positions
    /// (`maxb == 0`: no prefill plan).
    pub fn new_with_prefill(
        gguf: &Gguf,
        ec: &EngineConfig,
        tmax: u32,
        carrier: Carrier,
        maxb: usize,
    ) -> Result<GpuDecode, String> {
        unsafe {
            let vk = Vk::load()?;
            let c = layer_cfg(gguf, ec, tmax, carrier)?;
            let (hidden, vocab) = (c.hidden, ec.vocab_size as u32);
            let ((((n_disp, logits_map, params_map), prefill), (prof_pool, prof_labels)), cb) = {
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
                let mut weights: Vec<LayerW> = Vec::with_capacity(ec.n_layer);
                let carrier_layer = std::env::var("DEC_CARRIER_LAYER")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok());
                let orig_sel_mask = std::env::var("DEC_SEL_MASK").ok();
                for li in 0..ec.n_layer {
                    if let Some(k) = carrier_layer {
                        if li == k {
                            std::env::set_var("DEC_SEL_MASK", "0");
                        } else {
                            std::env::set_var("DEC_SEL_MASK", "8");
                        }
                    }
                    let lw = LayerW::load(&g, gguf, li, &c)?;
                    layers.push(record_layer(&g, &c, &lw, &sc, li, &mut st));
                    weights.push(lw);
                }
                if carrier_layer.is_some() {
                    match orig_sel_mask {
                        Some(ref m) => std::env::set_var("DEC_SEL_MASK", m),
                        None => std::env::remove_var("DEC_SEL_MASK"),
                    }
                }
                let prefill = if maxb > 0 {
                    Some(PrefillStack::build(
                        &g, &c, vocab, &weights, embed, lm, final_norm, maxb,
                    )?)
                } else {
                    None
                };
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
                // `DEC_AMAXP` = the argmax partial count `P`. Default 2048 (P52/P55): `P` lanes
                // scan the logits in `P / 256` workgroups and the ONE-workgroup tree finish
                // reduces the `P` partials, so the stage-2 serial walk over `P` no longer grows
                // with `P`. `DEC_AMAXP=256` (`AMAX_P`) records the two pre-P55 Steps.
                let amax_p = envu("DEC_AMAXP", 2048);
                assert!(
                    amax_p >= 256 && amax_p % 256 == 0,
                    "DEC_AMAXP={amax_p} must be a multiple of 256 at least 256"
                );
                let amaxp = g.dev(2 * amax_p as usize * 4);
                // 8 bytes: `argmax_finish` writes the winner at BYTE OFFSET 4 (`params[1]`'s slot).
                let amax_out = g.dev(8);
                let amax1 = make_step_named(
                    &g,
                    &argmax_partial_bytes(vocab, amax_p, 256),
                    &[logits, amaxp],
                    amax_p / 256,
                    1,
                    "argmax_partial",
                );
                let amax2 = if amax_p == AMAX_P {
                    make_step_named(
                        &g,
                        &argmax_finish_bytes(amax_p),
                        &[amaxp, amax_out],
                        1,
                        1,
                        "argmax_finish",
                    )
                } else {
                    make_step_named(
                        &g,
                        &argmax_finish_tree_bytes(amax_p),
                        &[amaxp, amax_out],
                        1,
                        1,
                        "argmax_finish",
                    )
                };

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

                let dec_prof = envu("DEC_PROF", 0);
                let cb = vk.command_buffer();
                vk.begin(cb);
                let (prof_pool, prof_labels, n_disp) = if dec_prof >= 2 {
                    // Lower-overhead in-chain schemes (DECODE-ATTRIB T5/T6). `DEC_PROF` selects the
                    // stamp set and the `vkCmdWriteTimestamp2` stage:
                    //   2 sparse / 3 full  at `STAGE_ALL_COMMANDS`;
                    //   4 sparse / 5 full  at `STAGE_BOTTOM_OF_PIPE`;
                    //   6 sparse / 7 full  at `VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT`;
                    //   8 dn-and-successor / 9 single-layer  at `VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT`.
                    let sparse = matches!(dec_prof, 2 | 4 | 6);
                    let stage = match dec_prof {
                        4 | 5 => STAGE_BOTTOM_OF_PIPE,
                        6 | 7 | 8 | 9 => STAGE2_COMPUTE,
                        _ => STAGE_ALL_COMMANDS,
                    };
                    let prof_layer = envu("DEC_PROF_LAYER", 0) as usize;
                    // A recorded node is a `Step` (embed, final norms, argmax) or a `Node`
                    // (layer nodes, LM head).
                    enum Item<'a> {
                        S(&'a Step),
                        N(&'a Node),
                    }
                    let mut flat: Vec<(Item, String)> = Vec::new();
                    let mut stamp: Vec<bool> = Vec::new();
                    flat.push((Item::S(&embed_node), "- embed".to_string()));
                    stamp.push(true);
                    for li in 0..ec.n_layer {
                        let ln = &layers[li];
                        let base = flat.len();
                        let roles = if ln.len() == 5 {
                            vec!["qkv", "rope/flash", "o", "gu", "dn"]
                        } else {
                            let mut r = vec!["qkv"];
                            for n in &ln[1..ln.len() - 3] {
                                let name = match n {
                                    Node::Step(s) => s.name,
                                    Node::Gemv(_) => "rope/flash",
                                };
                                r.push(name);
                            }
                            r.push("o");
                            r.push("gu");
                            r.push("dn");
                            r
                        };
                        for (i, (n, r)) in ln.iter().zip(&roles).enumerate() {
                            let lab = match dec_prof {
                                8 => {
                                    if i == 0 {
                                        "pre"
                                    } else {
                                        r
                                    }
                                }
                                9 => {
                                    if li == prof_layer && i == ln.len() - 1 {
                                        "dn"
                                    } else if li == prof_layer + 1 && i == 0 {
                                        "rest"
                                    } else {
                                        r
                                    }
                                }
                                _ => {
                                    if sparse && i == 0 {
                                        "pre"
                                    } else {
                                        r
                                    }
                                }
                            };
                            let full_lab = if dec_prof == 9 && li == prof_layer + 1 && i == 0 {
                                "- rest".to_string()
                            } else {
                                format!("{li} {lab}")
                            };
                            flat.push((Item::N(n), full_lab));
                        }
                        let dn = base + ln.len() - 1;
                        for i in base..flat.len() {
                            let s = match dec_prof {
                                8 => (li > 0 && i == base) || i == dn,
                                9 => {
                                    (li == prof_layer && i == dn)
                                        || (prof_layer + 1 < ec.n_layer
                                            && li == prof_layer + 1
                                            && i == base)
                                }
                                _ => !sparse || i == base || i == dn,
                            };
                            stamp.push(s);
                        }
                    }
                    flat.push((
                        Item::S(&final_rms),
                        if dec_prof == 8 {
                            "- tail".to_string()
                        } else {
                            "- final_rms".to_string()
                        },
                    ));
                    stamp.push(true);
                    flat.push((Item::S(&final_rms_f32), "- final_rms_f32".to_string()));
                    stamp.push(dec_prof != 8);
                    flat.push((Item::N(&lm_head), "- lm_head".to_string()));
                    stamp.push(dec_prof != 8);
                    flat.push((Item::S(&amax1), "- amax1".to_string()));
                    stamp.push(dec_prof != 8);
                    flat.push((Item::S(&amax2), "- amax2".to_string()));
                    stamp.push(dec_prof != 8);
                    let labels: Vec<String> = flat
                        .iter()
                        .zip(&stamp)
                        .filter(|(_, s)| **s)
                        .map(|((_, l), _)| l.clone())
                        .collect();
                    let npool = labels.len() as u32 + 1;
                    let pool = vk.timestamp_pool(npool);
                    vk.cmd_reset_timestamps(cb, &pool);
                    let mut q = 0u32;
                    let mut n_disp = 0u32;
                    for ((item, _), s) in flat.iter().zip(&stamp) {
                        if *s {
                            vk.cmd_write_timestamp(cb, &pool, stage, q);
                            q += 1;
                        }
                        n_disp += match item {
                            Item::S(st) => rec_step(&vk, cb, st),
                            Item::N(Node::Step(st)) => rec_step(&vk, cb, st),
                            Item::N(Node::Gemv(gm)) => rec_gemv(&vk, cb, gm),
                        };
                    }
                    vk.cmd_write_timestamp(cb, &pool, stage, q);
                    q += 1;
                    assert_eq!(q, npool);
                    (Some(pool), labels, n_disp)
                } else if dec_prof == 1 {
                    let mut labels = Vec::new();
                    labels.push("- embed".to_string());
                    for li in 0..ec.n_layer {
                        let ln = &layers[li];
                        if ln.len() == 5 {
                            labels.push(format!("{li} qkv"));
                            labels.push(format!("{li} rope/flash"));
                            labels.push(format!("{li} o"));
                            labels.push(format!("{li} gu"));
                            labels.push(format!("{li} dn"));
                        } else {
                            labels.push(format!("{li} qkv"));
                            for n in &ln[1..ln.len() - 3] {
                                let name = match n {
                                    Node::Step(s) => s.name,
                                    Node::Gemv(_) => "rope/flash",
                                };
                                labels.push(format!("{li} {name}"));
                            }
                            labels.push(format!("{li} o"));
                            labels.push(format!("{li} gu"));
                            labels.push(format!("{li} dn"));
                        }
                    }
                    labels.push("- final_rms".to_string());
                    labels.push("- final_rms_f32".to_string());
                    labels.push("- lm_head".to_string());
                    labels.push("- amax1".to_string());
                    labels.push("- amax2".to_string());

                    let pool = vk.timestamp_pool(labels.len() as u32 + 1);
                    vk.cmd_reset_timestamps(cb, &pool);
                    let mut q = 0u32;
                    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, q);
                    q += 1;
                    let mut n_disp = rec_step(&vk, cb, &embed_node);
                    for l in &layers {
                        for n in l {
                            vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, q);
                            q += 1;
                            n_disp += match n {
                                Node::Step(s) => rec_step(&vk, cb, s),
                                Node::Gemv(gm) => rec_gemv(&vk, cb, gm),
                            };
                        }
                    }
                    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, q);
                    q += 1;
                    n_disp += rec_step(&vk, cb, &final_rms);
                    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, q);
                    q += 1;
                    n_disp += rec_step(&vk, cb, &final_rms_f32);
                    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, q);
                    q += 1;
                    n_disp += rec_nodes(&vk, cb, std::slice::from_ref(&lm_head));
                    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, q);
                    q += 1;
                    n_disp += rec_step(&vk, cb, &amax1);
                    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, q);
                    q += 1;
                    n_disp += rec_step(&vk, cb, &amax2);
                    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, q);
                    assert_eq!(q, labels.len() as u32);
                    (Some(pool), labels, n_disp)
                } else {
                    let mut n_disp = rec_step(&vk, cb, &embed_node);
                    for l in &layers {
                        n_disp += rec_nodes(&vk, cb, l);
                    }
                    n_disp += rec_step(&vk, cb, &final_rms);
                    n_disp += rec_step(&vk, cb, &final_rms_f32);
                    n_disp += rec_nodes(&vk, cb, std::slice::from_ref(&lm_head));
                    n_disp += rec_step(&vk, cb, &amax1);
                    n_disp += rec_step(&vk, cb, &amax2);
                    (None, Vec::new(), n_disp)
                };
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
                (
                    (
                        ((n_disp, logits_map, sc.params_map), prefill),
                        (prof_pool, prof_labels),
                    ),
                    cb,
                )
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
                prefill,
                pf_argmax: None,
                prof_pool,
                prof_labels,
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
        self.pf_argmax = None;
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
        match self.pf_argmax {
            Some(t) => t,
            None => unsafe { self.argmax_map.read() as usize },
        }
    }

    /// Per-node GPU-active microseconds of one prefill of `toks` (see [`PrefillStack::profile`]).
    pub fn prefill_profile(&mut self, toks: &[usize]) -> Result<Vec<(String, f64)>, String> {
        let pf = self
            .prefill
            .as_mut()
            .ok_or("GpuDecode: built without a prefill plan")?;
        pf.run(&self.vk, toks)?;
        Ok(pf.profile(&self.vk, toks.len()))
    }

    /// Prefill `toks` at positions `0..n` in one device submit (requires a plan from
    /// [`GpuDecode::new_with_prefill`]). The decode KV cache then holds `0..n`, so
    /// `step(n, token)` continues. Returns the last position's logit row; [`GpuDecode::last_argmax`]
    /// and [`GpuDecode::last_hidden`] then describe the last prompt position.
    pub fn prefill(&mut self, toks: &[usize]) -> Result<&[f32], String> {
        let pf = self
            .prefill
            .as_mut()
            .ok_or("GpuDecode: built without a prefill plan")?;
        pf.run(&self.vk, toks)?;
        self.pf_argmax = Some(pf.argmax());
        for (o, &v) in self.hout.iter_mut().zip(pf.hidden()) {
            *o = v as f64;
        }
        Ok(pf.logits())
    }

    /// The final normed hidden of the step just run.
    pub fn last_hidden(&self) -> &[f64] {
        &self.hout
    }

    /// Per-node GPU-active microseconds of the decode step just run (requires `DEC_PROF=1`).
    pub fn step_profile(&self) -> Option<Vec<(String, f64)>> {
        let pool = self.prof_pool.as_ref()?;
        let ts = unsafe { self.vk.read_timestamps(pool) };
        let mut out = Vec::with_capacity(self.prof_labels.len());
        for (i, label) in self.prof_labels.iter().enumerate() {
            let ticks = (ts[i + 1] as i64).wrapping_sub(ts[i] as i64);
            let us = ticks as f64 * self.vk.timestamp_period as f64 / 1000.0;
            out.push((label.clone(), us));
        }
        Some(out)
    }
}
