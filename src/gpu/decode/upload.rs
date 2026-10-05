//! Model-wide uploads for [`super::model::GpuDecode`]: the embedding chart, the output norm and the
//! LM head, plus the dense-attention restriction check.
//!
//! Replaces fold's `Model` loader slice (`bedrock/src/serve/model`) and the head of `GpuDecode::new`
//! (`gpu.rs:1265-1346`): `embed` is one chart over the full vocabulary, packed once; the head is the
//! same chart when the checkpoint has no `output.weight` (tied), else its own chart. Both are
//! packed at the one `Carrier` the layers use, through [`pack_proj_f32`].
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's bedrock/src/serve/gpu.rs; lint profile is fold's, not apeiron's"
)]

use super::pack::pack_proj_f32;
use super::state::{dequant_rows, LayerCfg};
use super::*;
use crate::engine::loader::{EngineConfig, MixerKind};
use crate::format::gguf::Gguf;

/// The layer configuration of `ec`, or the first restriction of fold's generality it violates.
///
/// Restrictions (each a refusal, not a silent reduction): softmax mixers only, no attention output
/// gate, no QK-norm, no MoE experts, `hidden`, `qn` and `dff` multiples of 32, `head_dim` a
/// multiple of 4.
pub fn layer_cfg(
    gguf: &Gguf,
    ec: &EngineConfig,
    tmax: u32,
    carrier: Carrier,
) -> Result<LayerCfg, String> {
    if ec.mixers.iter().any(|m| *m != MixerKind::Softmax) {
        return Err("GpuDecode: affine mixers are not ported".into());
    }
    if ec.attn_gate {
        return Err("GpuDecode: attention output gate is not ported".into());
    }
    for t in [
        "attn_q_norm.weight",
        "ffn_gate_exps.weight",
        "ffn_gate_inp.weight",
    ] {
        if gguf.header.tensor(&format!("blk.0.{t}")).is_some() {
            return Err(format!(
                "GpuDecode: blk.0.{t} (QK-norm / MoE) is not ported"
            ));
        }
    }
    let c = LayerCfg {
        hidden: ec.n_embd as u32,
        dff: ec.n_ff as u32,
        nh: ec.n_head as u32,
        nkv: ec.n_head_kv as u32,
        hd: ec.head_dim as u32,
        tmax,
        eps: ec.rms_eps as f32,
        theta: ec.rope_theta as f32,
        rope_dims: ec.rot_dims as u32,
        carrier,
    };
    if c.hidden % 32 != 0 || c.qn() % 32 != 0 || c.dff % 32 != 0 || c.hd % 4 != 0 {
        return Err(format!(
            "GpuDecode: chart carriers need 32-multiples: {c:?}"
        ));
    }
    Ok(c)
}

/// The embedding chart `(words, dm)` over all `vocab` rows; row `params[1]` is the gathered token.
///
/// # Safety
/// Vulkan buffer creation and upload.
pub unsafe fn upload_embed(
    g: &GpuCtx,
    gguf: &Gguf,
    c: &LayerCfg,
    vocab: usize,
) -> Result<(Handle, Handle), String> {
    upload_rows(g, gguf, "token_embd.weight", c, vocab)
}

/// The LM head chart: its own `output.weight` when present, else the embedding chart (tied).
///
/// # Safety
/// Vulkan buffer creation and upload.
pub unsafe fn upload_lm_head(
    g: &GpuCtx,
    gguf: &Gguf,
    c: &LayerCfg,
    vocab: usize,
    embed: (Handle, Handle),
) -> Result<(Handle, Handle), String> {
    if gguf.header.tensor("output.weight").is_some() {
        upload_rows(g, gguf, "output.weight", c, vocab)
    } else {
        Ok(embed)
    }
}

unsafe fn upload_rows(
    g: &GpuCtx,
    gguf: &Gguf,
    name: &str,
    c: &LayerCfg,
    vocab: usize,
) -> Result<(Handle, Handle), String> {
    let rows = dequant_rows(gguf, name, 0, vocab)?;
    let p = pack_proj_f32(c.head_carrier(), &[&rows], c.hidden as usize, false);
    Ok((g.up_u32(&p.words), g.up_u32(&p.dm)))
}

/// The output RMSNorm gain as f32.
///
/// # Safety
/// Vulkan buffer creation and upload.
pub unsafe fn upload_final_norm(g: &GpuCtx, gguf: &Gguf) -> Result<Handle, String> {
    Ok(g.up_f32(&dequant_rows(gguf, "output_norm.weight", 0, 1)?))
}
