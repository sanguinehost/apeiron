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

use super::pack::{pack_proj_f32, pack_q4_k_raw, pack_q5_0_raw, pack_q6_k_raw, Proj};
use super::*;
use crate::engine::dequant_row_fixed;
use crate::format::gguf::{Gguf, GgufHeader};
use crate::format::quant::{
    block_geom, GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K,
};

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
    /// kept as Q4_K (lossless `pack_q4_k_raw`), a projection whose tensors are all Q5_0 (Q6_K) and
    /// whose chart the cooperative-matrix prefill GEMM serves is kept as Q5 (Q6) (lossless
    /// `pack_q5_0_raw`, `pack_q6_k_raw`),
    /// every other projection and the embedding/head go to Q8. Any other carrier packs every
    /// projection at that carrier.
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

/// Whether every named tensor is stored as [`GGML_TYPE_Q5_0`] — the whole-group type test the
/// all-Q5_0 branch of [`pack_group`] consults. Pure over the header (no file bytes are read), so a
/// synthetic-directory unit test exercises it without a checkpoint.
fn all_q5_0(h: &GgufHeader, names: &[String]) -> bool {
    names
        .iter()
        .all(|nm| h.tensor(nm).is_some_and(|t| t.ggml_type == GGML_TYPE_Q5_0))
}

/// Whether every named tensor is stored as [`GGML_TYPE_Q6_K`] — the whole-group type test the
/// all-Q6_K branch of [`pack_group`] consults. Pure over the header (no file bytes are read), so a
/// synthetic-directory unit test exercises it without a checkpoint.
fn all_q6_k(h: &GgufHeader, names: &[String]) -> bool {
    names
        .iter()
        .all(|nm| h.tensor(nm).is_some_and(|t| t.ggml_type == GGML_TYPE_Q6_K))
}

/// The raw GGUF bytes of `names` concatenated row-wise, and the total row count. Every name must be
/// present (the caller's type test has already looked each one up). Time and space O(Σ bytes), one
/// copy of each tensor's bytes.
fn raw_rows(gguf: &Gguf, names: &[String]) -> Result<(Vec<u8>, usize), String> {
    let mut raw: Vec<u8> = Vec::new();
    let mut m = 0usize;
    for nm in names {
        let t = gguf
            .header
            .tensor(nm)
            .ok_or_else(|| format!("missing tensor {nm}"))?;
        raw.extend_from_slice(&gguf.tensor_bytes(t).map_err(|e| format!("{nm}: {e:?}"))?);
        m += t.dims.get(1).copied().unwrap_or(1) as usize;
    }
    Ok((raw, m))
}

/// Pack the row-concatenation of `names` (each `[m_i, k]`) as one chart. Carrier `Q4` mode with every
/// tensor Q4_K: the raw GGUF bytes concatenate row-wise and go through `pack_q4_k_raw` (lossless,
/// the bytes llama.cpp reads). With every tensor Q5_0 (Q6_K) and `plane_b_ok`: the same row
/// concatenation goes through `pack_q5_0_raw` (`pack_q6_k_raw`) into the `Carrier::Q5` (`Carrier::Q6`)
/// two-plane chart. Otherwise the f32 dequantization
/// is packed at Q8 (`Q4` mode) or the mode's own carrier. Returns the chart and the carrier it is at.
fn pack_group(
    gguf: &Gguf,
    names: &[String],
    k: usize,
    c: &LayerCfg,
    plane_b_ok: bool,
) -> Result<(Proj, Carrier), String> {
    if c.carrier == Carrier::Q4 {
        let all_q4k = names.iter().all(|nm| {
            gguf.header
                .tensor(nm)
                .is_some_and(|t| t.ggml_type == GGML_TYPE_Q4_K && k % 256 == 0)
        });
        if all_q4k {
            let (raw, m) = raw_rows(gguf, names)?;
            let p = pack_q4_k_raw(&raw, m, k, false).map_err(|e| format!("{names:?}: {e:?}"))?;
            return Ok((p, Carrier::Q4));
        }
        if plane_b_ok && k % 32 == 0 && all_q5_0(&gguf.header, names) {
            let (raw, m) = raw_rows(gguf, names)?;
            let p = pack_q5_0_raw(&raw, m, k).map_err(|e| format!("{names:?}: {e:?}"))?;
            return Ok((p, Carrier::Q5));
        }
        if plane_b_ok && k % 256 == 0 && all_q6_k(&gguf.header, names) {
            let (raw, m) = raw_rows(gguf, names)?;
            let p = pack_q6_k_raw(&raw, m, k).map_err(|e| format!("{names:?}: {e:?}"))?;
            return Ok((p, Carrier::Q6));
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
    /// Upload layer `li`: dequantize each tensor with apeiron's loader and repack into the mode's
    /// chart (`q`,`k`,`v` rows are concatenated into one `qkv` chart, `gate`,`up` into one `gu`
    /// chart, as fold's loader does). Each group's carrier is [`pack_group`]'s: Q4_K in `Q4` mode,
    /// Q5_0 or Q6_K in `Q4` mode on a coopmat device, else Q8.
    ///
    /// # Safety
    /// Vulkan buffer creation and upload.
    pub unsafe fn load(g: &GpuCtx, gguf: &Gguf, li: usize, c: &LayerCfg) -> Result<LayerW, String> {
        let n = |s: &str| format!("blk.{li}.{s}");
        let (h, dff) = (c.hidden as usize, c.dff as usize);
        // A two-plane group (`Carrier::Q5`, `Carrier::Q6`) is served only where the cooperative-matrix
        // prefill GEMM reads its plane B; the non-coopmat prefill fallback
        // (`gemv_dp4a_batch_rt_module`) panics on both, so on a device without coopmat the group stays
        // Q8 (`coopmat_ok` in `prefill.rs` reads the same flag). `DEC_SEL=0` forces every group off the
        // two-plane selection, so the loader takes exactly the pre-selection Q4_K/Q8 path.
        // `DEC_SEL_MASK` (default 15) selects per group for attribution: bit 0 qkv, 1 o, 2 gu, 3 dn;
        // a cleared bit keeps that group on the pre-selection path.
        let plane_b_ok = g.vk.has_coopmat && envu("DEC_SEL", 1) != 0;
        let sel_mask = envu("DEC_SEL_MASK", 15);
        let sel = |bit: u32| plane_b_ok && sel_mask & (1 << bit) != 0;
        let (qkv, qkv_c) = pack_group(
            gguf,
            &[n("attn_q.weight"), n("attn_k.weight"), n("attn_v.weight")],
            h,
            c,
            sel(0),
        )?;
        let (o, o_c) = pack_group(gguf, &[n("attn_output.weight")], c.qn() as usize, c, sel(1))?;
        let (gu, gu_c) = pack_group(
            gguf,
            &[n("ffn_gate.weight"), n("ffn_up.weight")],
            h,
            c,
            sel(2),
        )?;
        let (dn, dn_c) = pack_group(gguf, &[n("ffn_down.weight")], dff, c, sel(3))?;
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

#[cfg(test)]
mod tests {
    use super::{all_q5_0, all_q6_k};
    use crate::engine::testdata::model_path;
    use crate::format::gguf::{Gguf, GgufHeader, TensorDesc};
    use crate::format::quant::{GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K};

    const QWEN05B_PATH: &str = "models/gguf/qwen05b-Q4_K_M.gguf";

    fn desc(name: &str, ggml_type: u32) -> TensorDesc {
        TensorDesc {
            name: name.as_bytes().to_vec(),
            dims: vec![896, 896],
            ggml_type,
            offset: 0,
        }
    }

    fn header(types: &[(&str, u32)]) -> GgufHeader {
        GgufHeader {
            version: 3,
            kv: Vec::new(),
            tensors: types.iter().map(|&(n, t)| desc(n, t)).collect(),
            alignment: 32,
            header_end: 0,
            data0: 0,
        }
    }

    /// The whole-group type test over a synthetic directory: `all_q5_0` holds only when every named
    /// tensor is present and Q5_0.
    #[test]
    fn all_q5_0_needs_every_tensor() {
        let both = header(&[("a", GGML_TYPE_Q5_0), ("b", GGML_TYPE_Q5_0)]);
        assert!(all_q5_0(&both, &["a".to_string(), "b".to_string()]));
        let mixed = header(&[("a", GGML_TYPE_Q5_0), ("b", GGML_TYPE_Q4_K)]);
        assert!(!all_q5_0(&mixed, &["a".to_string(), "b".to_string()]));
        let one = header(&[("a", GGML_TYPE_Q5_0)]);
        assert!(!all_q5_0(&one, &["a".to_string(), "missing".to_string()]));
    }

    /// The whole-group type test over a synthetic directory: `all_q6_k` holds only when every named
    /// tensor is present and Q6_K.
    #[test]
    fn all_q6_k_needs_every_tensor() {
        let both = header(&[("a", GGML_TYPE_Q6_K), ("b", GGML_TYPE_Q6_K)]);
        assert!(all_q6_k(&both, &["a".to_string(), "b".to_string()]));
        let mixed = header(&[("a", GGML_TYPE_Q6_K), ("b", GGML_TYPE_Q4_K)]);
        assert!(!all_q6_k(&mixed, &["a".to_string(), "b".to_string()]));
        let one = header(&[("a", GGML_TYPE_Q6_K)]);
        assert!(!all_q6_k(&one, &["a".to_string(), "missing".to_string()]));
    }

    /// The selection predicate the loader reads, on qwen05b-Q4_K_M blk.0: the all-Q5_0 groups
    /// (`attn_output`, `ffn_gate`+`ffn_up`) select Q5; `ffn_down` (Q6_K) does not, and stays Q8.
    #[test]
    fn q5_0_selection_on_qwen05b_blk0() {
        let Some(path) = model_path(QWEN05B_PATH) else {
            eprintln!("skip q5_0_selection_on_qwen05b_blk0: {QWEN05B_PATH} absent");
            return;
        };
        let gguf = Gguf::open(&path).expect("open qwen05b");
        let h = &gguf.header;
        let names = |s: &[&str]| -> Vec<String> { s.iter().map(|x| (*x).to_string()).collect() };
        assert!(
            all_q5_0(h, &names(&["blk.0.attn_output.weight"])),
            "blk.0.attn_output.weight must select Q5"
        );
        assert!(
            all_q5_0(h, &names(&["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"])),
            "blk.0 ffn_gate+ffn_up must select Q5"
        );
        assert!(
            !all_q5_0(h, &names(&["blk.0.ffn_down.weight"])),
            "blk.0.ffn_down.weight (Q6_K) must not select Q5"
        );
    }

    /// The selection predicate the loader reads, on qwen05b-Q4_K_M: ffn_down groups that are
    /// Q6_K are selected by `all_q6_k`, and those that are Q4_K are not.
    #[test]
    fn q6_k_selection_on_qwen05b() {
        let Some(path) = model_path(QWEN05B_PATH) else {
            eprintln!("skip q6_k_selection_on_qwen05b: {QWEN05B_PATH} absent");
            return;
        };
        let gguf = Gguf::open(&path).expect("open qwen05b");
        let h = &gguf.header;
        let names = |s: &[&str]| -> Vec<String> { s.iter().map(|x| (*x).to_string()).collect() };
        let mut q6_count = 0usize;
        let mut q4_count = 0usize;
        for i in 0..24 {
            let name = format!("blk.{i}.ffn_down.weight");
            if let Some(t) = h.tensor(&name) {
                let nms = names(&[&name]);
                if t.ggml_type == GGML_TYPE_Q6_K {
                    assert!(
                        all_q6_k(h, &nms),
                        "blk.{i}.ffn_down.weight (Q6_K) must select Q6"
                    );
                    q6_count += 1;
                } else if t.ggml_type == GGML_TYPE_Q4_K {
                    assert!(
                        !all_q6_k(h, &nms),
                        "blk.{i}.ffn_down.weight (Q4_K) must not select Q6"
                    );
                    q4_count += 1;
                }
            }
        }
        assert!(q6_count > 0, "qwen05b must have Q6_K ffn_down groups");
        assert!(q4_count > 0, "qwen05b must have Q4_K ffn_down groups");
    }

    /// The upload byte inventory `LayerW::load` (`state.rs:247`) and `upload_embed`/`upload_lm_head`
    /// (`upload.rs:74,87`) produce for qwen05b-Q4_K_M, recomputed on the CPU through the same
    /// `pack_group` / `dequant_rows` / `pack_proj_f32` calls the loader makes — no Vulkan context, no
    /// GPU (`layer_cfg` and `Engine::open` are CPU-only).
    ///
    /// `plane_b_ok` is the value `LayerW::load` computes as `has_coopmat && DEC_SEL != 0`
    /// (`state.rs:255`): `true` is a coopmat device at `DEC_SEL=1`; `false` is `DEC_SEL=0` (or a
    /// non-coopmat device). Byte count per projection is `4 * (words.len() + dm.len())` — exactly
    /// what `g.up_u32` uploads (`state.rs:290-302`).
    ///
    /// Run in release: `cargo test --release --features gpu --lib -- --ignored
    /// qwen05b_upload_bytes_match_inventory --test-threads=1 --nocapture`.
    #[test]
    #[ignore = "reads the 397 MB qwen05b checkpoint; run in release with --ignored"]
    fn qwen05b_upload_bytes_match_inventory() {
        use super::{dequant_rows, pack_group};
        use crate::engine::loader::Engine;
        use crate::gpu::decode::pack::pack_proj_f32;
        use crate::gpu::decode::upload::layer_cfg;
        use crate::gpu::simt::Carrier;

        let Some(path) = model_path(QWEN05B_PATH) else {
            eprintln!("skip qwen05b_upload_bytes_match_inventory: {QWEN05B_PATH} absent");
            return;
        };
        let gguf = Gguf::open(&path).expect("open qwen05b");
        let eng = Engine::open(&path).expect("engine open qwen05b");
        let ec = &eng.config;
        let c = layer_cfg(&gguf, ec, 256, Carrier::Q4).expect("layer_cfg qwen05b");
        assert_eq!(c.carrier, Carrier::Q4, "the loader runs Carrier::Q4 here");

        const CARRIERS: [Carrier; 4] = [Carrier::Q4, Carrier::Q5, Carrier::Q6, Carrier::Q8];
        let cidx = |car: Carrier| match car {
            Carrier::Q4 => 0,
            Carrier::Q5 => 1,
            Carrier::Q6 => 2,
            _ => 3,
        };
        // `LayerW::load`'s four groups: (label, tensor names, k = packer `k` argument).
        let groups = |li: usize| -> [(&'static str, Vec<String>, usize); 4] {
            let n = |s: &str| format!("blk.{li}.{s}");
            [
                (
                    "qkv",
                    vec![n("attn_q.weight"), n("attn_k.weight"), n("attn_v.weight")],
                    c.hidden as usize,
                ),
                ("o", vec![n("attn_output.weight")], c.qn() as usize),
                (
                    "gu",
                    vec![n("ffn_gate.weight"), n("ffn_up.weight")],
                    c.hidden as usize,
                ),
                ("dn", vec![n("ffn_down.weight")], c.dff as usize),
            ]
        };

        // Head: qwen05b has no `output.weight`, so `upload_lm_head` returns the tied embedding
        // `upload_embed` built from `upload_rows(gguf, "token_embd.weight", 0, vocab)` — dequantize
        // then `pack_proj_f32(c.head_carrier(), …, hidden, false)` (`upload.rs:101-111`).
        let tied = gguf.header.tensor("output.weight").is_none();
        let head_name = if tied {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        let head_desc = gguf.header.tensor(head_name).expect("head tensor").clone();
        let head_rows = dequant_rows(&gguf, head_name, 0, ec.vocab_size).expect("head dequant");
        let head_ref: [&[f32]; 1] = [head_rows.as_slice()];
        let head_proj = pack_proj_f32(c.head_carrier(), &head_ref, c.hidden as usize, false);
        let head_bytes = 4 * (head_proj.words.len() + head_proj.dm.len());
        eprintln!(
            "head: tied={tied} tensor={head_name} ggml_type={} stored={} B carrier={:?} \
words={} dm={} -> {head_bytes} B",
            head_desc.ggml_type,
            gguf.tensor_bytes(&head_desc).expect("head bytes").len(),
            c.head_carrier(),
            head_proj.words.len(),
            head_proj.dm.len(),
        );
        assert!(
            tied,
            "qwen05b must tie the head to token_embd.weight (no output.weight)"
        );
        assert_eq!(head_bytes, 153_151_488, "head upload bytes");

        // The GGUF's own stored bytes for the tensors `pack_group` reads, over the 24 layers, and for
        // the embedding tensor.
        let mut stored_layers = 0usize;
        for li in 0..ec.n_layer {
            for (_, names, _) in groups(li) {
                for name in &names {
                    let td = gguf.header.tensor(name).expect("layer tensor").clone();
                    stored_layers += gguf.tensor_bytes(&td).expect("layer bytes").len();
                }
            }
        }
        let emb = gguf
            .header
            .tensor("token_embd.weight")
            .expect("token_embd")
            .clone();
        let stored_embd = gguf.tensor_bytes(&emb).expect("embd bytes").len();
        eprintln!("stored: layers={stored_layers} B token_embd={stored_embd} B");
        assert_eq!(stored_layers, 246_930_432, "GGUF stored layer-matrix bytes");
        assert_eq!(stored_embd, 144_643_072, "GGUF token_embd.weight bytes");

        let mut arms: Vec<(bool, usize, [usize; 4])> = Vec::new();
        for plane_b_ok in [true, false] {
            let mut per_group = [0usize; 4];
            let mut per_carrier = [[0usize; 4]; 4];
            let mut count = [[0usize; 4]; 4];
            for li in 0..ec.n_layer {
                for (gi, (_, names, k)) in groups(li).iter().enumerate() {
                    let (p, car) = pack_group(&gguf, names, *k, &c, plane_b_ok)
                        .unwrap_or_else(|e| panic!("pack_group blk.{li} group {gi}: {e}"));
                    let bytes = 4 * (p.words.len() + p.dm.len());
                    per_group[gi] += bytes;
                    per_carrier[gi][cidx(car)] += bytes;
                    count[gi][cidx(car)] += 1;
                }
            }
            let total: usize = per_group.iter().sum();
            eprintln!(
                "plane_b_ok={plane_b_ok} (DEC_SEL={}): layers total {total} B",
                u8::from(plane_b_ok)
            );
            for (gi, label) in ["qkv", "o", "gu", "dn"].iter().enumerate() {
                for (ci, car) in CARRIERS.iter().enumerate() {
                    if count[gi][ci] > 0 {
                        eprintln!(
                            "  {label} {car:?}: {} groups, {} B",
                            count[gi][ci], per_carrier[gi][ci]
                        );
                    }
                }
                eprintln!("  {label} total: {} B", per_group[gi]);
            }
            arms.push((plane_b_ok, total, per_group));
        }
        for (plane_b_ok, total, per_group) in &arms {
            let want = if *plane_b_ok {
                273_014_784
            } else {
                376_406_016
            };
            assert_eq!(
                *total, want,
                "plane_b_ok={plane_b_ok} layer upload bytes (per-group {per_group:?})"
            );
        }
    }
}
