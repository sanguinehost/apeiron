// Row: `inf-resident-prefill-GEB633UfORCK7XSmzJifHd` (INF-RESIDENT-PREFILL W1): fold `GpuPrefillStack` (gpu.rs:4448-5991).
//! `PrefillStack`: whole-prompt prefill on the device, ported from fold
//! `bedrock/src/serve/gpu.rs:4448-5991` (`GpuPrefillStack`) onto [`super::model::GpuDecode`]'s
//! resident weights and KV cache, with the batched-projection tile rule of `3886-4130`
//! (`prefill_tile`).
//!
//! One command buffer per call: batched embed gather -> per layer {RMSNorm+quantize, qkv GEMM,
//! RoPE+KV append, causal attention (`attn_prefill_tiled`), KV re-layout into the decode cache,
//! quantize, o GEMM, RMSNorm+residual+quantize, gate/up GEMM, SwiGLU, quantize, down GEMM} ->
//! final norm -> last-row gather -> one-tile LM head -> two-stage argmax -> DMA of
//! `[logits row | hidden row | argmax]`. The residual stream, activations and KV never leave the
//! device; the decode cache is left holding positions `0..n` for [`super::model::GpuDecode::step`].
//!
//! Restrictions (each a refusal in [`PrefillStack::build`], not a silent reduction): dense softmax
//! attention, no QK-norm, full rotary (`rope_dims == hd`), `hd` a power of two `<= 256`, prompt
//! starts at position 0 (no chunk continuation), `n <= maxb`, `n <= tmax`; per-projection carriers
//! as `GpuDecode` packs them (each consumer gets its own activation-chart word order); only the last
//! position's logits are produced (fold's `LogitRows::Last`); V reaches the decode cache as
//! `f16(int8 code * scale)` (fold's `seed_kv`), where token-by-token decode stores f16(V) directly.
//! Not ported: the superblock face, `PhysicalStorageBuffer` GEMM, `LogitRows::All`, the host KV
//! mirrors and `CpuState` fold, `profile`, KV seed from host.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's bedrock/src/serve/gpu.rs; lint profile is fold's, not apeiron's"
)]

use super::state::{LayerCfg, LayerW};
use super::*;
use crate::simt::gemv_dp4a_batch_rt_bytes;
use crate::simt_nn::{
    attn_prefill_coopmat_bytes, coopmat_gemm_bytes, coopmat_gemm_chart_bytes, coopmat_gemm_groups,
    coopmat_gemm_swiglu_bytes, ATTN_COOP_ROWS, TILE_BM, TILE_BN, TILE_KB,
};

/// Largest batch tile any node picks; activation scratch is padded to a multiple of it so every
/// node's `ceil(n/bt)*bt` rows stay inside zeroed memory (fold `PREFILL_BT_MAX`). It is the
/// cooperative-matrix kernel's token tile [`TILE_BN`] (64), not the DP4A tiles' 32: the coopmat
/// kernel carries no token guard, so `bp = ceil(maxb/PREFILL_BT_MAX)·PREFILL_BT_MAX` must be a
/// multiple of 64 and every scratch buffer it writes (`xq`, `xsc`, `xsm`, `qkv_buf`, `proj_h`,
/// `gu_buf`, `att_buf`, `hout`) is sized in `bp`. For maxb 512 (pp512) and the match
/// test's maxb 64 that adds no rows; for other maxb it adds up to `32·kmax/4` u32 words per
/// activation chart and up to `32·m` f32 per output.
pub const PREFILL_BT_MAX: usize = TILE_BN as usize;
/// Workgroup shared-memory budget used to refuse an attention tile (fold `WG_SHARED_MAX`).
const WG_SHARED_MAX: u32 = 49_152;

/// fold `gpu.rs:4026-4051` (`prefill_tile`): the 2-D (batch tile, row tile) for one GEMM shape.
/// `rt*bt <= 64`; `bt = 32` when the chart exceeds 32 MB (no longer L2-resident), else 8; `rt` as
/// large as that allows with at least two row slabs and at most 50 % tail waste.
pub fn prefill_tile(m: u32, k: u32, carrier: Carrier, local: u32) -> (u32, u32) {
    let chart_bytes = (k / carrier.pack()) as u64 * m as u64 * 4 + (k / 32) as u64 * m as u64 * 4;
    let bt = match std::env::var("DEC_PREFILL_BT") {
        Ok(v) => v.parse().unwrap_or(8u32).max(1).min(PREFILL_BT_MAX as u32),
        Err(_) if chart_bytes > 32 << 20 => 32,
        Err(_) => 8,
    };
    if let Some(rt) = std::env::var("DEC_PREFILL_RT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        return (bt, rt.max(1));
    }
    let mut rt = (64 / bt).max(1);
    while rt > 1 {
        let slabs = m.div_ceil(local * rt);
        let waste = slabs * local * rt - m;
        if slabs >= 2 && waste * 2 <= m {
            break;
        }
        rt /= 2;
    }
    (bt, rt)
}

/// Whether the cooperative-matrix chart GEMM can serve an `m × k` projection at `cr` on this
/// device. The kernel's static preconditions are `coopmat_gemm_module`'s: `m % 16 == 0` (rows past
/// `m` in the last 128-row tile are clamped on load and never stored, so `m % 128` may be nonzero),
/// `k % (32·TILE_KB) == 0` (K32 is one chart block, two blocks per shared stage), and a
/// single-plane chart carrier the device's signed-int8 MMA serves (Q8/Q4/Q2 — never Q3). When the
/// predicate is false the DP4A batch GEMM stays the floor. `vk.has_coopmat` is false when the
/// device lacks the extension or `APEIRON_COOPMAT=0`.
fn coopmat_ok(vk: &Vk, m: u32, k: u32, cr: Carrier) -> bool {
    vk.has_coopmat
        && m % 16 == 0
        && m > 0
        && k % (32 * TILE_KB) == 0
        && matches!(cr, Carrier::Q8 | Carrier::Q4 | Carrier::Q2)
}

/// The attention query tile (`rows`, `rts`, `ct`, `kp`, `fuse`, `mode`); `DEC_ATTN_TILE=0|off` pins
/// the untiled kernel, `rows:rts:ct[:fuse[:mode[:kp]]]` pins a tile (fold `attn_tile`,
/// `gpu.rs:4466-4483`; `mode` is [`attn_prefill_tiled_module_m`]'s bitmask, `0` = the three-walk
/// bit-exact kernel). `kp` chunks the K slab over the head dim: `hd=64` defaults to the whole-head
/// tile (`kp=1`), `hd>=128` to the 16-row, `ct=4`, `kp=4` split tile (its `hd·hd·4` slab is 64 KiB
/// at hd=128, over the 48 KiB budget, and the split's peak is `(hd/kp)·hd·4` = 16 KiB).
fn attn_tile(hd: u32, default_mode: u32) -> Option<(u32, u32, u32, u32, bool, u32)> {
    let d = if hd >= 128 {
        (
            ATTN_TILE_ROWS_HD128,
            ATTN_TILE_RTS,
            ATTN_TILE_CT_HD128,
            ATTN_TILE_KP_HD128,
            ATTN_TILE_FUSE,
            default_mode,
        )
    } else {
        (
            ATTN_TILE_ROWS,
            ATTN_TILE_RTS,
            ATTN_TILE_CT,
            1,
            ATTN_TILE_FUSE,
            default_mode,
        )
    };
    match std::env::var("DEC_ATTN_TILE") {
        Err(_) => Some(d),
        Ok(v) if v == "0" || v == "off" => None,
        Ok(v) => {
            let p: Vec<u32> = v.split(':').filter_map(|s| s.parse().ok()).collect();
            Some((
                *p.first().unwrap_or(&d.0),
                *p.get(1).unwrap_or(&d.1),
                *p.get(2).unwrap_or(&d.2),
                *p.get(5).unwrap_or(&d.3),
                p.get(3).map_or(d.4, |&f| f != 0),
                *p.get(4).unwrap_or(&d.5),
            ))
        }
    }
}

/// The workgroup y-extent of a planned dispatch.
#[derive(Clone, Copy)]
enum Gy {
    /// one workgroup column per position
    Batch,
    /// one column per batch tile of `bt` positions
    Tiles(u32),
    /// exactly one column (the last-row tail)
    One,
}

struct PNode {
    name: String,
    pipe: Handle,
    layout: Handle,
    dset: Handle,
    gx: u32,
    gy: Gy,
}

/// Pipelines are compiled once per kernel key and shared by every layer's node (the descriptor
/// layouts are identically defined, hence compatible).
struct Planner<'a, 'b> {
    g: &'a GpuCtx<'b>,
    cache: Vec<(String, Handle, Handle)>,
}

impl Planner<'_, '_> {
    unsafe fn node(
        &mut self,
        key: String,
        code: impl FnOnce() -> Vec<u8>,
        bufs: &[Handle],
        gx: u32,
        gy: Gy,
    ) -> PNode {
        let (layout1, dset) = self.g.vk.ssbo_set(bufs);
        let (pipe, layout) = match self.cache.iter().find(|e| e.0 == key) {
            Some(e) => (e.1, e.2),
            None => {
                let p = self.g.vk.pipeline(&code(), layout1);
                self.cache.push((key.clone(), p, layout1));
                (p, layout1)
            }
        };
        PNode {
            name: key,
            pipe,
            layout,
            dset,
            gx,
            gy,
        }
    }
}

/// The whole-prompt prefill context (device handles only; owned by `GpuDecode`).
pub struct PrefillStack {
    cb: CommandBuffer,
    plan: Vec<PNode>,
    tail: Vec<PNode>,
    params_map: *mut u32,
    toks_map: *mut u32,
    rb_map: *mut f32,
    rb: Handle,
    hout: Handle,
    logits1: Handle,
    amax_out: Handle,
    pub maxb: usize,
    tmax: u32,
    hidden: usize,
    vocab: usize,
    /// Dispatches in a full-chunk command buffer.
    pub n_disp: usize,
}

impl PrefillStack {
    /// Build the plan for chunks of up to `maxb` positions over `layers`' resident weights.
    ///
    /// # Safety
    /// Vulkan buffer, pipeline and descriptor creation over caller-owned handles.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn build(
        g: &GpuCtx,
        c: &LayerCfg,
        vocab: u32,
        layers: &[LayerW],
        embed: (Handle, Handle),
        lm: (Handle, Handle),
        final_norm: Handle,
        maxb: usize,
    ) -> Result<PrefillStack, String> {
        let vk = g.vk;
        let (hidden, dff, nh, nkv, hd, tmax) = (c.hidden, c.dff, c.nh, c.nkv, c.hd, c.tmax);
        let (qn, qkvm) = (c.qn() as usize, c.qkv_m());
        if !hd.is_power_of_two() || hd > 256 || nh % nkv != 0 {
            return Err(format!(
                "PrefillStack: head shape nh={nh} nkv={nkv} hd={hd} outside this cut"
            ));
        }
        if c.rope_dims != hd {
            return Err("PrefillStack: partial rotary is not ported".into());
        }
        if maxb == 0 || maxb > tmax as usize {
            return Err(format!(
                "PrefillStack: maxb {maxb} must be in 1..=tmax {tmax}"
            ));
        }
        let bias = layers[0].qkv_b.is_some();
        if layers.iter().any(|l| l.qkv_b.is_some() != bias) {
            return Err("PrefillStack: bias not uniform across layers".into());
        }
        let lw = local_wg();
        let hc = c.head_carrier();
        let bp = maxb.div_ceil(PREFILL_BT_MAX) * PREFILL_BT_MAX;
        let (hidden_u, dff_u, nkv_u, hd_u, tmax_u) = (
            hidden as usize,
            dff as usize,
            nkv as usize,
            hd as usize,
            tmax as usize,
        );
        let kmax = hidden_u.max(qn).max(dff_u);

        // ── scratch, zero-initialised, padded to `bp` rows ─────────────────────────────────────
        let z = |elems: usize| g.dev_zeroed(elems * 4);
        let x_dev = z(bp * hidden_u);
        let hbuf = z(bp * hidden_u);
        let proj_h = z(bp * hidden_u);
        let qkv_buf = z(bp * qkvm as usize);
        let att_buf = z(bp * qn);
        let gu_buf = z(bp * 2 * dff_u);
        // power-of-two chart capacities: the last-row gather closes its source index with one `and`
        let qcap = (bp * kmax / 4).next_power_of_two();
        let bcap = (bp * kmax / 32).next_power_of_two();
        let (xq, xsc, xsm) = (z(qcap), z(bcap), z(bcap));
        let dummy = g.up_f32(&[0f32]);
        let (kc_p, ks_p, vc_p, vs_p) = (
            z(nkv_u * hd_u * tmax_u),
            z(tmax_u * nkv_u),
            z(nkv_u * tmax_u * hd_u),
            z(tmax_u * nkv_u),
        );
        let (kout, ksout, vout, vsout) = (
            z(maxb * nkv_u * hd_u),
            z(maxb * nkv_u),
            z(maxb * nkv_u * hd_u),
            z(maxb * nkv_u),
        );
        let (params, params_mem) = vk
            .buffer(
                16,
                BUFFER_USAGE_STORAGE,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .ok_or("prefill params alloc failed")?;
        let params_map = vk.map(params_mem).cast::<u32>();
        let (toks, toks_mem) = vk
            .buffer(
                (maxb * 4) as u64,
                BUFFER_USAGE_STORAGE,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .ok_or("prefill toks alloc failed")?;
        let toks_map = vk.map(toks_mem).cast::<u32>();
        let src_dev = |bytes: usize| -> Result<Handle, String> {
            Ok(vk
                .buffer(
                    bytes.max(4) as u64,
                    BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST,
                    MEMORY_DEVICE_LOCAL,
                )
                .ok_or("prefill device buffer alloc failed")?
                .0)
        };
        let hout = src_dev(bp * hidden_u * 4)?;

        // ── the LM-head tail's scratch ─────────────────────────────────────────────────────────
        // The head emits the FINAL position's logits only (fold's `LogitRows::Last`): `gather`
        // copies the last chunk row into slot 0, `argmax_partial` walks row 0 alone, and the
        // readback copies `logits1` row 0. `prefill_tile(vocab, hidden, hc, lw)` is a 32-slot token
        // tile on this chart (169 MB > 32 MB) and runs the `rt·bt` outer product for 31 slots whose
        // activations are zero and whose logits are never read: the head's token tile is 1.
        // `rt_lm = 2` keeps `ceil(vocab/(lw·2))` workgroups (594 at vocab 151936, lw 128) and two
        // independent weight-word streams per lane for latency hiding; the measured lm-node time
        // is 181 us against 383 us for the 32-token tile at the same GPU clock (see the receipt).
        let (bt_lm, rt_lm) = (1u32, 2u32);
        let qrow = hidden_u / 4;
        let brow = hidden_u / 32;
        let xq1 = z(bt_lm as usize * qrow);
        let xsc1 = z(bt_lm as usize * brow);
        let xsm1 = z(bt_lm as usize * brow);
        let logits1 = src_dev(bt_lm as usize * vocab as usize * 4)?;
        let amaxp = g.dev(2 * AMAX_P as usize * 4);
        let amax_out = src_dev(8)?;

        // ── the plan ───────────────────────────────────────────────────────────────────────────
        let mut pl = Planner {
            g,
            cache: Vec::new(),
        };
        let dpw = |cr: Carrier| cr.pack() / 4;
        let scale = c.scale();
        // The one-walk (online) form reduces the slab max with `subgroupMax` over 32 lanes, so it
        // is selected by default only where a 32-wide subgroup is established. `Vk` carries no
        // subgroup-size query; the cooperative-matrix int8 shape (NVIDIA Turing+, warp 32) stands
        // in for it, and `DEC_ATTN_TILE=…:<mode>` pins a mode explicitly.
        let default_mode = if vk.has_coopmat {
            ATTN_TILE_MODE
        } else {
            ATTN_TILE_MODE & !ATTN_MODE_ONLINE
        };
        let tile = attn_tile(hd, default_mode).filter(|&(rows, rts, ct, kp, _, mode)| {
            let (tsz, shared) = attn_tile_shape(hd, rows, rts, ct, kp);
            let ok = rows as usize <= PREFILL_BT_MAX
                && rows % rts == 0
                && hd % ct == 0
                && kp > 0
                && hd % kp == 0
                && rows.is_power_of_two()
                && tsz % hd == 0
                && rows % (tsz / hd) == 0
                && (tsz / hd) > 0
                && (hd / kp) % (tsz / hd) == 0
                && tsz <= 1024
                // the online form adds `rows` f32 of per-row rescale factors
                && shared + if mode & ATTN_MODE_ONLINE != 0 { rows * 4 } else { 0 } <= WG_SHARED_MAX
                && (mode & ATTN_MODE_ONLINE == 0 || hd / ct == 32);
            if !ok {
                eprintln!("PrefillStack: attn tile {rows}:{rts}:{ct}:kp{kp} outside this cut at hd={hd}; untiled kernel");
            }
            ok
        });
        // The cooperative-matrix attention (fp16 operands, f32 accumulate, `attn_coopmat.comp`) is
        // the default where the device exposes the tensor-core shape and the cut holds; `DEC_ATTN_COOP=0`
        // pins the SIMT tile (`DEC_ATTN_TILE` set also pins it).
        let attn_coop = vk.has_coopmat
            && hd == 64
            && qkvm % 4 == 0
            && tmax % 4 == 0
            && nh % nkv == 0
            // the kernel writes the o-projection's activation chart itself, in natural word order
            && layers.iter().all(|l| dpw(l.o_c) == 1)
            && std::env::var("DEC_ATTN_COOP").map_or(true, |v| v != "0")
            && std::env::var("DEC_ATTN_TILE").is_err();
        let attn_gy = if attn_coop {
            Gy::Tiles(ATTN_COOP_ROWS)
        } else {
            match tile {
                Some((rows, _, _, _, _, _)) => Gy::Tiles(rows),
                None => Gy::Batch,
            }
        };
        let attn_key = if attn_coop {
            format!("attn_coopmat_{ATTN_COOP_ROWS}")
        } else {
            match tile {
                Some((r, s, t, k, f, m)) => format!("attn_tiled_{r}_{s}_{t}_kp{k}_{f}_m{m}"),
                None => "attn_untiled".to_string(),
            }
        };
        let mut plan: Vec<PNode> = Vec::with_capacity(layers.len() * 12 + 4);
        // One projection GEMM node: the cooperative-matrix chart kernel when [`coopmat_ok`] holds
        // (the device exposes the tensor-core shape, the carrier is single-plane, the shape meets
        // the kernel's static preconditions), else the DP4A batch GEMM (the floor). Both kernels
        // bind `[w, xq, out, dm, xsm, xsc]` (the coopmat GLSL's W/X/O/D/XS/SX); only the code and
        // the dispatch differ. `Gy::Tiles(TILE_BN)` is the kernel's `ceil(tokens/64)` y-grid
        // recomputed from `n` at dispatch; the x-grid is `coopmat_gemm_groups`'s `ceil(m/TILE_BM)`,
        // independent of the token count.
        macro_rules! gemm_node {
            ($m:expr, $k:expr, $cr:expr, $bufs:expr) => {{
                let (m, k, cr) = ($m, $k, $cr);
                if coopmat_ok(vk, m, k, cr) {
                    plan.push(pl.node(
                        format!("gemm_{m}_{k}_{cr:?}_cm"),
                        move || coopmat_gemm_bytes(m, k, cr),
                        $bufs,
                        coopmat_gemm_groups(m, TILE_BN).0,
                        Gy::Tiles(TILE_BN),
                    ));
                } else {
                    let (bt, rt) = prefill_tile(m, k, cr, lw);
                    plan.push(pl.node(
                        format!("gemm_{m}_{k}_{cr:?}_{bt}_{rt}"),
                        move || gemv_dp4a_batch_rt_bytes(m, k, lw, cr, bt, rt),
                        $bufs,
                        m.div_ceil(lw * rt),
                        Gy::Tiles(bt),
                    ));
                }
            }};
        }
        // Chart of the fused gate/up GEMM inside `gu_buf`: words at 0, then the per-32-block scales,
        // then the sums (floats). `gu_buf` holds `bp * 2 * dff` floats; the chart takes `bp * dff * 9/32`.
        let sc_off = (bp * dff_u / 4) as u32;
        let sm_off = sc_off + (bp * dff_u / 32) as u32;
        let swiglu_fuse = vk.has_coopmat
            && dff % 64 == 0
            && std::env::var("DEC_SWIGLU_FUSE").map_or(true, |v| v != "0");
        plan.push(pl.node(
            format!("embed_batch_{:?}", hc),
            || embed_chart_batch_bytes(hidden, vocab, lw, hc),
            &[embed.0, embed.1, toks, x_dev],
            hidden.div_ceil(lw),
            Gy::Batch,
        ));
        for (li, l) in layers.iter().enumerate() {
            // norm1 (+ the previous layer's down output as residual from layer 1)
            let dq = dpw(l.qkv_c);
            if li == 0 {
                plan.push(pl.node(
                    format!("rms0_{dq}"),
                    || rmsnorm_batch_q_noh_bytes(hidden, c.eps, dq, false),
                    &[x_dev, l.ln1, x_dev, xq, xsc, xsm, hbuf],
                    1,
                    Gy::Batch,
                ));
            } else {
                plan.push(pl.node(
                    format!("rms_{dq}"),
                    || rmsnorm_batch_q_noh_bytes(hidden, c.eps, dq, true),
                    &[x_dev, l.ln1, proj_h, xq, xsc, xsm, hbuf],
                    1,
                    Gy::Batch,
                ));
            }
            gemm_node!(
                qkvm,
                hidden,
                l.qkv_c,
                &[l.qkv_w, xq, qkv_buf, l.qkv_dm, xsm, xsc]
            );
            plan.push(pl.node(
                format!("rope_{bias}"),
                || qkv_rope_batch_bytes(nh, nkv, hd, qkvm, tmax, c.theta, c.eps, bias, false),
                &[
                    qkv_buf,
                    l.qkv_b.unwrap_or(dummy),
                    dummy,
                    dummy,
                    kc_p,
                    ks_p,
                    vc_p,
                    vs_p,
                    params,
                    kout,
                    ksout,
                    vout,
                    vsout,
                    l.kcq,
                    l.sk,
                    l.vc,
                ],
                nh + 2 * nkv,
                Gy::Batch,
            ));
            // the coopmat kernel's epilogue is the chart: xq, xsc (scales), xsm (sums)
            let attn_bufs: Vec<_> = if attn_coop {
                vec![qkv_buf, kc_p, ks_p, vc_p, vs_p, xq, params, xsc, xsm]
            } else {
                vec![qkv_buf, kc_p, ks_p, vc_p, vs_p, att_buf, params]
            };
            plan.push(pl.node(
                attn_key.clone(),
                || match tile {
                    _ if attn_coop => attn_prefill_coopmat_bytes(nh, nkv, qkvm, tmax, scale),
                    Some((rows, rts, ct, kp, fuse, mode)) => attn_prefill_tiled_bytes_m(
                        nh, nkv, hd, qkvm, tmax, scale, rows, rts, ct, kp, fuse, mode,
                    ),
                    None => attn_prefill_bytes(nh, nkv, hd, qkvm, tmax, scale),
                },
                &attn_bufs,
                nh,
                attn_gy,
            ));
            if !attn_coop {
                let dqo = dpw(l.o_c);
                plan.push(pl.node(
                    format!("quant_{qn}_{dqo}"),
                    || quantize_batch_bytes(qn as u32, dqo),
                    &[att_buf, xq, xsc, xsm],
                    1,
                    Gy::Batch,
                ));
            }
            gemm_node!(
                hidden,
                qn as u32,
                l.o_c,
                &[l.o_w, xq, proj_h, l.o_dm, xsm, xsc]
            );
            // norm2: residual (+= o) then the gate/up activation chart
            let dqg = dpw(l.gu_c);
            plan.push(pl.node(
                format!("rms_{dqg}"),
                || rmsnorm_batch_q_noh_bytes(hidden, c.eps, dqg, true),
                &[x_dev, l.ln2, proj_h, xq, xsc, xsm, hbuf],
                1,
                Gy::Batch,
            ));
            let dqd = dpw(l.dn_c);
            // The gate/up GEMM's epilogue writes the down projection's chart itself (into `gu_buf`,
            // which then holds no gate/up row); the down GEMM reads it at the chart offsets. Both
            // GEMMs must be the coopmat kernel, and the tile pairs 64 gate rows with 64 up rows.
            let swiglu_fused = swiglu_fuse
                && coopmat_ok(vk, 2 * dff, hidden, l.gu_c)
                && coopmat_ok(vk, hidden, dff, l.dn_c)
                && (2 * dff) % TILE_BM == 0
                && matches!(dqd, 1 | 2 | 4);
            if swiglu_fused {
                plan.push(pl.node(
                    format!("gemm_{}_{hidden}_{:?}_cm_swiglu{dqd}", 2 * dff, l.gu_c),
                    || coopmat_gemm_swiglu_bytes(2 * dff, hidden, l.gu_c, dqd, sc_off, sm_off),
                    &[l.gu_w, xq, gu_buf, l.gu_dm, xsm, xsc],
                    coopmat_gemm_groups(2 * dff, TILE_BN).0,
                    Gy::Tiles(TILE_BN),
                ));
                plan.push(pl.node(
                    format!("gemm_{hidden}_{dff}_{:?}_cm_chart", l.dn_c),
                    || coopmat_gemm_chart_bytes(hidden, dff, l.dn_c, sc_off, sm_off),
                    &[l.dn_w, gu_buf, proj_h, l.dn_dm, gu_buf, gu_buf],
                    coopmat_gemm_groups(hidden, TILE_BN).0,
                    Gy::Tiles(TILE_BN),
                ));
                continue;
            }
            gemm_node!(
                2 * dff,
                hidden,
                l.gu_c,
                &[l.gu_w, xq, gu_buf, l.gu_dm, xsm, xsc]
            );
            plan.push(pl.node(
                format!("swiglu_quant_{dff}_{dqd}"),
                || swiglu_quant_batch_bytes(dff, dqd),
                &[gu_buf, xq, xsc, xsm],
                1,
                Gy::Batch,
            ));
            gemm_node!(
                hidden,
                dff,
                l.dn_c,
                &[l.dn_w, xq, proj_h, l.dn_dm, xsm, xsc]
            );
        }
        // final norm: folds the last down output, emits the head's activation chart and f32 hidden
        let dh = dpw(hc);
        plan.push(pl.node(
            format!("rms_{dh}"),
            || rmsnorm_batch_q_bytes(hidden, c.eps, dh, true),
            &[x_dev, final_norm, proj_h, xq, xsc, xsm, hout],
            1,
            Gy::Batch,
        ));
        // tail: gather the last row, the same lm GEMM over one tile, argmax
        let lm_gx = vocab.div_ceil(lw * rt_lm);
        let tail = vec![
            pl.node(
                "gather".to_string(),
                || {
                    last_row_gather_bytes(
                        qrow as u32,
                        brow as u32,
                        (qcap - 1) as u32,
                        (bcap - 1) as u32,
                    )
                },
                &[xq, xsc, xsm, xq1, xsc1, xsm1, params],
                1,
                Gy::One,
            ),
            pl.node(
                format!("gemm_{vocab}_{hidden}_{:?}_{bt_lm}_{rt_lm}", hc),
                || gemv_dp4a_batch_rt_bytes(vocab, hidden, lw, hc, bt_lm, rt_lm),
                &[lm.0, xq1, logits1, lm.1, xsm1, xsc1],
                lm_gx,
                Gy::One,
            ),
            pl.node(
                "argmax_partial".to_string(),
                || argmax_partial_bytes(vocab, AMAX_P, 256),
                &[logits1, amaxp],
                1,
                Gy::One,
            ),
            pl.node(
                "argmax_finish".to_string(),
                || argmax_finish_bytes(AMAX_P),
                &[amaxp, amax_out],
                1,
                Gy::One,
            ),
        ];

        // `[vocab logits row | hidden row | argmax]` in one HOST_CACHED window
        let rbytes = (vocab as usize + hidden_u + 1) as u64 * 4;
        let cached = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT | MEMORY_HOST_CACHED;
        let plain = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
        let (rb, rb_mem) = match vk.buffer(rbytes, BUFFER_USAGE_TRANSFER_DST, cached) {
            Some(x) => x,
            None => vk
                .buffer(rbytes, BUFFER_USAGE_TRANSFER_DST, plain)
                .ok_or("prefill readback alloc failed")?,
        };
        let rb_map = vk.map(rb_mem).cast::<f32>();
        let n_disp = plan.len() + tail.len();
        Ok(PrefillStack {
            cb: vk.reusable_command_buffer(),
            plan,
            tail,
            params_map,
            toks_map,
            rb_map,
            rb,
            hout,
            logits1,
            amax_out,
            maxb,
            tmax,
            hidden: hidden_u,
            vocab: vocab as usize,
            n_disp,
        })
    }

    /// Prefill `toks` at positions `0..n`: fills the decode KV cache for those positions and the
    /// last position's `[logits | hidden | argmax]` window.
    ///
    /// Cost: one submit; GEMM work `n*(sum of projection params) + n^2*hd*nh/2` (attention) MACs.
    pub fn run(&mut self, vk: &Vk, toks: &[usize]) -> Result<(), String> {
        let n = toks.len();
        if n == 0 || n > self.maxb || n > self.tmax as usize {
            return Err(format!(
                "PrefillStack: n={n} outside 1..=min(maxb {}, tmax {})",
                self.maxb, self.tmax
            ));
        }
        unsafe {
            for (i, &t) in toks.iter().enumerate() {
                self.toks_map.add(i).write_volatile(t as u32);
            }
            self.params_map.write_volatile(0);
            self.params_map.add(1).write_volatile(n as u32);
            let cb = self.cb;
            vk.reset_cb(cb);
            vk.begin(cb);
            for p in self.plan.iter().chain(self.tail.iter()) {
                let gy = match p.gy {
                    Gy::Batch => n as u32,
                    Gy::Tiles(bt) => (n as u32).div_ceil(bt),
                    Gy::One => 1,
                };
                (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, p.pipe);
                (vk.bind_sets)(
                    cb,
                    PIPELINE_BIND_POINT_COMPUTE,
                    p.layout,
                    0,
                    1,
                    &p.dset,
                    0,
                    std::ptr::null(),
                );
                (vk.dispatch)(cb, p.gx, gy, 1);
                step_barrier(vk, cb);
            }
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_TRANSFER,
                ACCESS_SHADER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            let v4 = self.vocab as u64 * 4;
            let h4 = self.hidden as u64 * 4;
            let copy = |src: Handle, so: u64, dst_off: u64, size: u64| {
                let r = BufferCopy {
                    src_offset: so,
                    dst_offset: dst_off,
                    size,
                };
                (vk.copy_buffer)(cb, src, self.rb, 1, &r);
            };
            copy(self.logits1, 0, 0, v4);
            copy(self.hout, (n as u64 - 1) * h4, v4, h4);
            copy(self.amax_out, 4, v4 + h4, 4);
            vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_HOST,
                ACCESS_TRANSFER_WRITE,
                ACCESS_HOST_READ,
            );
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            vk.submit_wait(cb);
        }
        Ok(())
    }

    /// GPU-active microseconds per node name for the chunk of `n` tokens last written by
    /// [`PrefillStack::run`] (timestamps between consecutive dispatches of one command buffer; the
    /// inter-node barrier is included in the node it precedes). Names are summed over layers.
    pub fn profile(&mut self, vk: &Vk, n: usize) -> Vec<(String, f64)> {
        unsafe {
            let nodes = self.plan.len() + self.tail.len();
            let pool = vk.timestamp_pool(nodes as u32 + 1);
            let cb = self.cb;
            vk.reset_cb(cb);
            vk.begin(cb);
            vk.cmd_reset_timestamps(cb, &pool);
            for (i, p) in self.plan.iter().chain(self.tail.iter()).enumerate() {
                vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, i as u32);
                let gy = match p.gy {
                    Gy::Batch => n as u32,
                    Gy::Tiles(bt) => (n as u32).div_ceil(bt),
                    Gy::One => 1,
                };
                (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, p.pipe);
                (vk.bind_sets)(
                    cb,
                    PIPELINE_BIND_POINT_COMPUTE,
                    p.layout,
                    0,
                    1,
                    &p.dset,
                    0,
                    std::ptr::null(),
                );
                (vk.dispatch)(cb, p.gx, gy, 1);
                step_barrier(vk, cb);
            }
            vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, nodes as u32);
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            vk.submit_wait(cb);
            let ts = vk.read_timestamps(&pool);
            let mut acc: Vec<(String, f64)> = Vec::new();
            for (i, p) in self.plan.iter().chain(self.tail.iter()).enumerate() {
                let us = ts[i + 1].wrapping_sub(ts[i]) as f64 * vk.timestamp_period as f64 / 1000.0;
                match acc.iter_mut().find(|e| e.0 == p.name) {
                    Some(e) => e.1 += us,
                    None => acc.push((p.name.clone(), us)),
                }
            }
            acc
        }
    }

    /// The last position's logit row of the call just run.
    pub fn logits(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.rb_map, self.vocab) }
    }
    /// The last position's final-normed hidden of the call just run.
    pub fn hidden(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.rb_map.add(self.vocab), self.hidden) }
    }
    /// The device greedy token of the call just run.
    pub fn argmax(&self) -> usize {
        unsafe {
            self.rb_map
                .add(self.vocab + self.hidden)
                .cast::<u32>()
                .read() as usize
        }
    }
}
