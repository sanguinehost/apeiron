//! The minimal model and KV state one layer's dispatch needs.
//!
//! Replaces the slice of fold `bedrock/src/serve/model/{types.rs,state.rs}` and the buffer
//! allocation block of `GpuDecode::new` (`gpu.rs:2078-2290`) that [`super::layer::record_layer`]
//! reads: [`LayerCfg`] (the `Config` fields), [`LayerW`] (fold's per-layer `LayerGpu`), [`Scratch`]
//! (the shared activation buffers) and [`Stream`] (the residual ping-pong and the pending split-K
//! count). Weights come from apeiron's GGUF loader and `pack_proj_f32`; fold's `Model`/`Proj`
//! loader is not ported.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's bedrock/src/serve/gpu.rs; lint profile is fold's, not apeiron's"
)]

use super::pack::{pack_proj_f32, pack_q4_k_raw, Proj};
use super::*;
use crate::engine::dequant_row_fixed;
use crate::format::gguf::Gguf;
use crate::format::quant::{block_geom, GGML_TYPE_F32, GGML_TYPE_Q4_K};

/// The `Config` fields a dense attention layer reads.
#[derive(Clone, Copy, Debug)]
pub struct LayerCfg {
    pub hidden: u32,
    pub dff: u32,
    pub nh: u32,
    pub nkv: u32,
    pub hd: u32,
    /// KV capacity in positions.
    pub tmax: u32,
    pub eps: f32,
    pub theta: f32,
    pub rope_dims: u32,
    /// The weight carrier mode. `Carrier::Q4`: each projection whose GGUF tensors are all Q4_K is
    /// kept as Q4_K (lossless `pack_q4_k_raw`), every other projection and the embedding/head go to
    /// Q8. Any other carrier packs every projection at that carrier.
    pub carrier: Carrier,
}

impl LayerCfg {
    pub fn qn(&self) -> u32 {
        self.nh * self.hd
    }
    pub fn kn(&self) -> u32 {
        self.nkv * self.hd
    }
    pub fn qkv_m(&self) -> u32 {
        self.qn() + 2 * self.kn()
    }
    /// The carrier of the embedding chart and the LM head (never Q4: the head is Q6_K/Q8_0 in the
    /// checkpoints served here, and a Q4 chart of it would be lossy).
    pub fn head_carrier(&self) -> Carrier {
        if self.carrier == Carrier::Q4 {
            Carrier::Q8
        } else {
            self.carrier
        }
    }
    pub fn scale(&self) -> f32 {
        1.0f32 / (self.hd as f32).sqrt()
    }
    /// fold `gpu.rs:2078-2116`: the flash split count for `tmax`.
    pub fn nsplit(&self) -> u32 {
        let nblocks = self.tmax.div_ceil(64);
        let psh_cap = envu("DEC_FLASH_SHARED_CAP", 48 * 1024);
        let need = (self.tmax * 4).div_ceil(psh_cap);
        envu("DEC_FLASHSPLIT", (self.tmax / 64).clamp(8, 32).max(need))
            .min(nblocks.saturating_sub(1))
            .max(1)
    }
    pub fn segs_o(&self) -> u32 {
        segs(self.hidden, self.qn() / 32)
    }
    pub fn segs_d(&self) -> u32 {
        segs(self.hidden, self.dff / 32)
    }
    pub fn s_qkv(&self) -> u32 {
        segs(self.qkv_m(), self.hidden / 32)
    }
}

/// One layer's resident buffers (fold's `LayerGpu`, dense subset). All projections are one `Carrier` chart
/// on the per-32 `dm` scale face.
pub struct LayerW {
    pub ln1: Handle,
    pub ln2: Handle,
    pub qkv_w: Handle,
    pub qkv_dm: Handle,
    /// Per-projection carriers (qkv, o, gate/up, down) the chart above was packed at.
    pub qkv_c: Carrier,
    pub o_c: Carrier,
    pub gu_c: Carrier,
    pub dn_c: Carrier,
    pub qkv_b: Option<Handle>,
    pub o_w: Handle,
    pub o_dm: Handle,
    pub gu_w: Handle,
    pub gu_dm: Handle,
    pub dn_w: Handle,
    pub dn_dm: Handle,
    /// int8 K cache `[kn · tmax]`.
    pub kcq: Handle,
    /// K scale plane `[nkv · tmax]` f32.
    pub sk: Handle,
    /// f16 V cache `[kn · tmax]` halves.
    pub vc: Handle,
}

/// Rows `r0..r0+n` of tensor `name` as f32, through apeiron's engine dequantizer
/// (`dequant_row_fixed`, Q32.32 → f32) for every quant family it reads; `F32` tensors are read
/// as stored. `gguf.dequant` has no `Q5_0` reader, which this model's layer 0 uses.
pub fn dequant_rows(gguf: &Gguf, name: &str, r0: usize, n: usize) -> Result<Vec<f32>, String> {
    let t = gguf
        .header
        .tensor(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if t.ggml_type == GGML_TYPE_F32 {
        let v = gguf.dequant(name).map_err(|e| format!("{name}: {e:?}"))?;
        let k = t.dims.first().copied().unwrap_or(1) as usize;
        return Ok(v[r0 * k..(r0 + n) * k].to_vec());
    }
    let k = t.dims.first().copied().unwrap_or(0) as usize;
    let (qk, bb) = block_geom(t.ggml_type).ok_or_else(|| format!("{name}: unknown type"))?;
    let row_bytes = k.div_ceil(qk) * bb;
    let raw = gguf
        .tensor_bytes_prefix(t, ((r0 + n) * row_bytes) as u64)
        .map_err(|e| format!("{name}: {e:?}"))?;
    let mut out = Vec::with_capacity(n * k);
    let mut row = Vec::with_capacity(k);
    for r in r0..r0 + n {
        row.clear();
        dequant_row_fixed(
            t.ggml_type,
            &raw[r * row_bytes..(r + 1) * row_bytes],
            k,
            &mut row,
        )
        .map_err(|e| format!("{name}: {e:?}"))?;
        out.extend(row.iter().map(|&v| (v as f64 / 4294967296.0) as f32));
    }
    Ok(out)
}

fn dq(gguf: &Gguf, name: &str) -> Result<Vec<f32>, String> {
    let t = gguf
        .header
        .tensor(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let rows: u64 = t.dims.iter().skip(1).product::<u64>().max(1);
    dequant_rows(gguf, name, 0, rows as usize)
}

/// Pack the row-concatenation of `names` (each `[m_i, k]`) as one chart. Carrier `Q4` mode with every
/// tensor Q4_K: the raw GGUF bytes concatenate row-wise and go through `pack_q4_k_raw` (lossless,
/// the bytes llama.cpp reads). Otherwise the f32 dequantization is packed at Q8 (`Q4` mode) or the
/// mode's own carrier. Returns the chart and the carrier it is at.
fn pack_group(
    gguf: &Gguf,
    names: &[String],
    k: usize,
    c: &LayerCfg,
) -> Result<(Proj, Carrier), String> {
    if c.carrier == Carrier::Q4 {
        let all_q4k = names.iter().all(|nm| {
            gguf.header
                .tensor(nm)
                .is_some_and(|t| t.ggml_type == GGML_TYPE_Q4_K && k % 256 == 0)
        });
        if all_q4k {
            let mut raw: Vec<u8> = Vec::new();
            let mut m = 0usize;
            for nm in names {
                let t = gguf.header.tensor(nm).unwrap();
                raw.extend_from_slice(&gguf.tensor_bytes(t).map_err(|e| format!("{nm}: {e:?}"))?);
                m += t.dims.get(1).copied().unwrap_or(1) as usize;
            }
            let p = pack_q4_k_raw(&raw, m, k, false).map_err(|e| format!("{names:?}: {e:?}"))?;
            return Ok((p, Carrier::Q4));
        }
    }
    let carrier = c.head_carrier();
    let rows: Vec<Vec<f32>> = names
        .iter()
        .map(|nm| dq(gguf, nm))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    Ok((pack_proj_f32(carrier, &refs, k, false), carrier))
}

impl LayerW {
    /// Upload layer `li`: dequantize each tensor with apeiron's loader and repack into the Q4
    /// chart (`q`,`k`,`v` rows are concatenated into one `qkv` chart, `gate`,`up` into one `gu`
    /// chart, as fold's loader does).
    ///
    /// # Safety
    /// Vulkan buffer creation and upload.
    pub unsafe fn load(g: &GpuCtx, gguf: &Gguf, li: usize, c: &LayerCfg) -> Result<LayerW, String> {
        let n = |s: &str| format!("blk.{li}.{s}");
        let (h, dff) = (c.hidden as usize, c.dff as usize);
        let (qkv, qkv_c) = pack_group(
            gguf,
            &[n("attn_q.weight"), n("attn_k.weight"), n("attn_v.weight")],
            h,
            c,
        )?;
        let (o, o_c) = pack_group(gguf, &[n("attn_output.weight")], c.qn() as usize, c)?;
        let (gu, gu_c) = pack_group(gguf, &[n("ffn_gate.weight"), n("ffn_up.weight")], h, c)?;
        let (dn, dn_c) = pack_group(gguf, &[n("ffn_down.weight")], dff, c)?;
        let bias = if gguf.header.tensor(&n("attn_q.bias")).is_some() {
            let mut b = dq(gguf, &n("attn_q.bias"))?;
            b.extend(dq(gguf, &n("attn_k.bias"))?);
            b.extend(dq(gguf, &n("attn_v.bias"))?);
            Some(g.up_f32(&b))
        } else {
            None
        };
        let (kn, nkv, tmax) = (c.kn() as usize, c.nkv as usize, c.tmax as usize);
        Ok(LayerW {
            ln1: g.up_f32(&dq(gguf, &n("attn_norm.weight"))?),
            ln2: g.up_f32(&dq(gguf, &n("ffn_norm.weight"))?),
            qkv_w: g.up_u32(&qkv.words),
            qkv_dm: g.up_u32(&qkv.dm),
            qkv_c,
            o_c,
            gu_c,
            dn_c,
            qkv_b: bias,
            o_w: g.up_u32(&o.words),
            o_dm: g.up_u32(&o.dm),
            gu_w: g.up_u32(&gu.words),
            gu_dm: g.up_u32(&gu.dm),
            dn_w: g.up_u32(&dn.words),
            dn_dm: g.up_u32(&dn.dm),
            kcq: g.dev_maybe_zeroed(kn * tmax),
            sk: g.dev_maybe_zeroed(nkv * tmax * 4),
            vc: g.dev_maybe_zeroed(kn * tmax * 2),
        })
    }
}

/// The activation buffers every layer shares (`gpu.rs:2078-2262`).
pub struct Scratch {
    pub x: Handle,
    pub h: Handle,
    pub qkv: Handle,
    pub qkvp: Handle,
    pub partials: Handle,
    pub ones: Handle,
    pub o: Handle,
    pub gu: Handle,
    pub d: Handle,
    pub part_o: Handle,
    pub part_m: Handle,
    pub part_l: Handle,
    /// `params[0]` = position, `params[1]` = token id; host-visible.
    pub params: Handle,
    pub params_map: *mut u32,
}

impl Scratch {
    /// # Safety
    /// Vulkan buffer creation and mapping.
    pub unsafe fn new(g: &GpuCtx, c: &LayerCfg) -> Result<Scratch, String> {
        let (hidden, dff, qn, qkv_m) = (c.hidden, c.dff, c.qn(), c.qkv_m());
        let nsplit = c.nsplit();
        let max_ms = [
            (qkv_m, segs(qkv_m, hidden)),
            (hidden, segs(hidden, qn)),
            (2 * dff, segs(2 * dff, hidden)),
            (hidden, segs(hidden, dff)),
        ]
        .iter()
        .map(|&(m, s)| (m * s) as usize)
        .max()
        .unwrap();
        let (params, params_mem) =
            g.vk.buffer(
                8,
                BUFFER_USAGE_STORAGE,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .ok_or_else(|| "params buffer alloc failed".to_string())?;
        let params_map = g.vk.map(params_mem).cast::<u32>();
        Ok(Scratch {
            x: g.dev(hidden as usize * 4),
            h: g.dev(hidden as usize * 4),
            qkv: g.dev(qkv_m as usize * 4),
            qkvp: g.dev(qkv_m as usize * c.s_qkv() as usize * 4),
            partials: g.dev(max_ms * 4),
            ones: g.up_f32(&vec![1f32; 64]),
            o: g.dev(c.segs_o() as usize * hidden as usize * 4),
            gu: g.dev(2 * dff as usize * 4),
            d: g.dev(c.segs_d() as usize * hidden as usize * 4),
            part_o: g.dev((c.nh * nsplit * c.hd) as usize * 4),
            part_m: g.dev((c.nh * nsplit) as usize * 4),
            part_l: g.dev((c.nh * nsplit) as usize * 4),
            params,
            params_map,
        })
    }
}

/// The residual stream across layers (fold's `cur`/`alt`/`prev_ffn_segs`, `gpu.rs:2303-2316`):
/// `cur` holds the live stream, `alt` the ping-pong target, and `prev_ffn_segs` the number of
/// split-K rows the previous layer's down projection left in `d` for the next residual fold.
#[derive(Clone, Copy)]
pub struct Stream {
    pub cur: Handle,
    pub alt: Handle,
    pub prev_ffn_segs: u32,
}

impl Stream {
    pub fn new(sc: &Scratch, c: &LayerCfg) -> Stream {
        Stream {
            cur: sc.x,
            alt: sc.h,
            prev_ffn_segs: c.segs_d(),
        }
    }
}
