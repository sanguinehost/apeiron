//! The pack writer at the end of a `NanoQuant` campaign: one block-loop checkpoint directory per
//! layer (`<root>/l{l}/{q,k,v,o,gate,up,down}.param`, [`LayerPars::save`](super::kd::LayerPars))
//! becomes the `APNQPK01` pack `src/format/nq.rs` serves.
//!
//! Each linear is the reference's finalise form `Ŵ = diag(s_post)·sign(U)·sign(V)ᵀ·diag(s_pre)`
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `src/nanoquant/modules/linear.py:264-283`). The writer
//! stores the signs and the two scale vectors; `s_mid` is one because the block loop's
//! [`HostFactor`] carries no middle scale, and a pack with no salient columns and no residual
//! planes writes version 2. The rate is the header's: `NqPack::bits_per_weight`.
//!
//! This is the campaign's export step, not a second fit: it reads the checkpoints the block loop
//! and the model KD wrote and packs them. `tests/learn_block_tune.rs` calls it after block 27 and
//! `tests/learn_kd.rs` after the KD epochs.

use crate::engine::loader::Engine;
use crate::format::nq::{write, NqPack, PackedLinear};
use crate::learn::grad::model::Config;

use super::block::HostFactor;
use super::kd::{LayerPars, BLOCK_KEYS, FRONT_KEYS};
use super::mixer::{LayerHost, Mixer};

/// `f32` to bfloat16 bits, round to nearest even. The pack's scale words decode through
/// `bf16_to_fixed` (`src/engine/kern/nq.rs`), not a binary16 decode.
fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

/// `±1` signs `rows × cols` packed row-major, bit 1 for `+1`.
fn pack_signs(signs: impl Fn(usize, usize) -> f32, rows: usize, cols: usize) -> Vec<u32> {
    let words = cols.div_ceil(32);
    let mut out = vec![0u32; rows * words];
    for r in 0..rows {
        for c in 0..cols {
            if signs(r, c) > 0.0 {
                out[r * words + c / 32] |= 1 << (c % 32);
            }
        }
    }
    out
}

/// One factor as a pack section: `U` (`out × rank` row-major), `V` (`rank × inn`, the checkpoint's
/// `vt` transposed), the two scale vectors and a one `s_mid`. `O(out·rank + rank·inn)` words.
fn packed(layer: usize, kind: usize, f: &HostFactor) -> PackedLinear {
    let (out, inn, rank) = (f.out, f.inn, f.rank);
    assert_eq!(f.u.len(), out * rank, "layer {layer} kind {kind}: U");
    assert_eq!(f.vt.len(), inn * rank, "layer {layer} kind {kind}: V");
    assert_eq!(f.s_pre.len(), inn, "layer {layer} kind {kind}: s_pre");
    assert_eq!(f.s_post.len(), out, "layer {layer} kind {kind}: s_post");
    PackedLinear {
        layer,
        kind,
        out,
        inn,
        rank,
        u: pack_signs(|r, c| f.u[r * rank + c], out, rank).into(),
        v: pack_signs(|k, c| f.vt[c * rank + k], rank, inn).into(),
        s_pre: f
            .s_pre
            .iter()
            .map(|&x| bf16(x))
            .collect::<Vec<u16>>()
            .into(),
        s_mid: vec![bf16(1.0); rank].into(),
        s_post: f
            .s_post
            .iter()
            .map(|&x| bf16(x))
            .collect::<Vec<u16>>()
            .into(),
        salient_idx: Vec::new().into(),
        salient: Vec::new().into(),
        res_layers: 0,
        shared: false,
        res: Vec::new().into(),
        s_res: Vec::new().into(),
    }
}

/// Pack every layer of `root`'s checkpoints into `out` and return the parsed pack.
///
/// The seven sections of layer `l` come from [`LayerPars::load`]`(root/l{l}, attn_norm, gains,
/// ffn_norm)` in the reference's slot order: `q, k, v` then `o, gate, up, down`, which is the
/// pack's [`NQ_KIND_NAMES`](crate::format::nq::NQ_KIND_NAMES) order. A dense slot is a refusal:
/// the pack serves factors, and the campaign is complete only when every linear is one.
///
/// # Errors
///
/// A missing or malformed checkpoint, a softmax-only assumption the model does not meet, a pack
/// order refusal, or a write failure.
///
/// # Complexity
///
/// `O(n_layer · Σ(out·rank + rank·inn))` words read and packed; the file is that many bytes plus
/// the header. One layer's factors are resident at a time.
pub fn write_pack(engine: &Engine, cfg: &Config, root: &str, out: &str) -> Result<NqPack, String> {
    write_pack_layers(engine, cfg, root, out, 0..cfg.n_layer)
}

/// [`write_pack`] over a layer range. The header still covers every layer of the model, so the
/// dense skeleton serves the layers outside the range; a single layer is the preflight form
/// (fit layer 0 at the campaign's T, score it before the full wall), not a product pack.
///
/// # Errors
///
/// [`write_pack`]'s: a missing checkpoint, a layer that is not a softmax block, or a
/// [`NqPack::new`] refusal.
pub fn write_pack_layers(
    engine: &Engine,
    cfg: &Config,
    root: &str,
    out: &str,
    layers: std::ops::Range<usize>,
) -> Result<NqPack, String> {
    assert!(
        layers.start <= layers.end && layers.end <= cfg.n_layer,
        "layer range"
    );
    let mut linears = Vec::with_capacity((layers.end - layers.start) * 7);
    for l in layers {
        let host =
            LayerHost::from_engine(engine, cfg, l).map_err(|e| format!("layer {l}: {e:?}"))?;
        let gains = match &host.mixer {
            Mixer::Softmax { gains, .. } => gains.clone(),
            Mixer::Affine { .. } => {
                return Err(format!(
                    "layer {l}: not a softmax layer: no NanoQuant block fit"
                ))
            }
        };
        let pars = LayerPars::load(
            &format!("{root}/l{l}"),
            &host.attn_norm,
            &gains,
            &host.ffn_norm,
        )?;
        for (k, f) in pars.ff.iter().enumerate() {
            linears.push(packed(l, k, f));
        }
        for (k, f) in pars.bf.iter().enumerate() {
            linears.push(packed(l, FRONT_KEYS.len() + k, f));
        }
    }
    let pack = NqPack::new(cfg.n_layer, linears).map_err(|e| format!("pack: {e:?}"))?;
    let bytes = write(&pack).map_err(|e| format!("pack write: {e:?}"))?;
    std::fs::write(out, bytes).map_err(|e| format!("{out}: {e}"))?;
    Ok(pack)
}

/// The checkpoint directory names the block loop writes, in the pack's kind order: `q, k, v`
/// from [`FRONT_KEYS`], then `o, gate, up, down` from [`BLOCK_KEYS`].
#[must_use]
pub fn kind_names() -> [&'static str; 7] {
    [
        FRONT_KEYS[0],
        FRONT_KEYS[1],
        FRONT_KEYS[2],
        BLOCK_KEYS[0],
        BLOCK_KEYS[1],
        BLOCK_KEYS[2],
        BLOCK_KEYS[3],
    ]
}
