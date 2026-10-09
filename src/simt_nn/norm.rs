//! simt_nn::norm — the norm/GEMV fusion family: the standalone RMSNorm, the rmsnorm-co-reduced
//! GEMV (`gemv_norm`, `gemv_actq`, `gemv_dp4a_norm`, carrier the dial), the quantized SiLU·mul
//! and vector quantizer, and the chart embed gather. Cut verbatim from `simt_nn.rs`.

use super::*;

// ── [DEC-PDL] the `o → gu` device-scope completion counter (DECODE-ATTRIB T8) ──────────────────
//
// Vulkan has no programmatic dependent launch. The decode path's equivalent is: the producer `o`
// dispatch's workgroups each release-increment one shared counter word after their last store; the
// consumer `gu` dispatch's lane 0 spin-waits (CAPPED) on an acquire load of that word before it
// reads the activation `o` produced. The role is a per-node property set by `record_layer`; a node
// with no role emits byte-identical SPIR-V to the pre-PDL path.
//
// Hang safety (desktop GPU): every spin is bounded by `PDL_SPIN_CAP`. A workgroup that trips the
// cap increments `ovf[0]` and lets the kernel retire (following `simt_nn/gridsync.rs`). A tripped
// cap is reported as INVALID, never as a cheap edge.
std::thread_local! {
    static PDL_ROLE: std::cell::Cell<u8> = std::cell::Cell::new(0);
    static PDL_TARGET: std::cell::Cell<u32> = std::cell::Cell::new(0);
    static PDL_CNT: std::cell::Cell<u64> = std::cell::Cell::new(0);
    static PDL_OVF: std::cell::Cell<u64> = std::cell::Cell::new(0);
}

/// Role constants for a PDL node (`record_layer`). 0 = none.
pub const PDL_NONE: u8 = 0;
pub const PDL_PRODUCER: u8 = 1;
pub const PDL_CONSUMER: u8 = 2;
pub const PDL_RESET: u8 = 3;

/// Set the PDL role, buffers and target for the NEXT kernel a dispatcher emits. The dispatcher
/// appends `cnt` (and, for the consumer, `ovf`) to its binding list; the emitter reads the role.
/// Every `make_*` clears it after the build, so a node outside the `o → gu` edge emits the
/// unchanged kernel.
pub fn pdl_begin(role: u8, cnt: u64, ovf: u64, target: u32) {
    PDL_ROLE.with(|r| r.set(role));
    PDL_TARGET.with(|t| t.set(target));
    PDL_CNT.with(|c| c.set(cnt));
    PDL_OVF.with(|o| o.set(ovf));
}

/// Clear the role (used by a dispatcher after its `emit`, and by a caller with no PDL node).
pub fn pdl_set_role(role: u8, target: u32) {
    PDL_ROLE.with(|r| r.set(role));
    PDL_TARGET.with(|t| t.set(target));
}

/// The role the next dispatcher should honour (`record_layer`'s active marker).
pub fn pdl_role() -> u8 {
    PDL_ROLE.with(|r| r.get())
}

/// The counter buffer handle for the active PDL node.
pub fn pdl_cnt() -> u64 {
    PDL_CNT.with(|c| c.get())
}

/// The overflow buffer handle for the active PDL node.
pub fn pdl_ovf() -> u64 {
    PDL_OVF.with(|o| o.get())
}

fn pdl_target() -> u32 {
    PDL_TARGET.with(|t| t.get())
}

// The counter is the next free SSBO binding of its node: `o` (actq, FlashCombine aux at 4/5) → 6;
// `dn` (actq, no aux) → 4; `gu` (fused norm with residual at 5/6) → 7, overflow → 8. Those indices
// are pinned in the emitters below and matched by the host buffer order in `record_layer`.
//
// SPIR-V atomics under the default GLSL450 memory model. `gridsync.rs` emits the same ops on this
// device; the device requests `vulkan_memory_model_device_scope` (`gpu/vulkan.rs`).
const PDL_OP_ATOMIC_LOAD: u32 = 227;
const PDL_OP_ATOMIC_STORE: u32 = 228;
const PDL_OP_ATOMIC_IADD: u32 = 234;
const PDL_SCOPE_DEVICE: u32 = 1;
/// `AcquireRelease | UniformMemory` — the RMW's semantics.
const PDL_SEM_ACQREL: u32 = 0x48;
/// `Acquire | UniformMemory`. `OpAtomicLoad` may not carry Release or AcquireRelease.
const PDL_SEM_ACQ: u32 = 0x42;
/// `Release | UniformMemory`. `OpAtomicStore` may not carry Acquire or AcquireRelease.
const PDL_SEM_REL: u32 = 0x44;
/// Re-reads before a spinning workgroup gives up; a healthy edge resolves in a handful of reads.
const PDL_SPIN_CAP: u32 = 1 << 16;

/// Pointer to `buf[idx]` as a `uint` StorageBuffer pointer.
fn pdl_word_ptr(kb: &mut Kb, buf: u32, idx: u32) -> u32 {
    let p = kb.ptr_sb_u;
    let z = kb.cu(0);
    kb.e(OP_ACCESS_CHAIN, p, &[buf, z, idx])
}

/// Producer: lane 0 of every workgroup release-increments `cnt[0]` after the workgroup's stores.
fn pdl_producer(kb: &mut Kb, cnt: u32, tid: u32) {
    // No workgroup barrier: the release on the RMW orders lane 0's OWN prior store to `out`, and
    // every lane stores the same row value, so lane 0's store carries the workgroup's row.
    let z = kb.cu(0);
    let is0 = kb.e(OP_IEQ, kb.bool_t, &[tid, z]);
    kb.iff(is0, |kb| {
        let word = pdl_word_ptr(kb, cnt, z);
        let sc = kb.cu(PDL_SCOPE_DEVICE);
        // AcquireRelease, not Release: the increments form an RMW release sequence, and an acquire
        // that reads the final count must synchronize with every workgroup's store (the same
        // `SEM_DEV_ACQREL` the `gridsync.rs` grid barrier uses for its arrival count).
        let sem = kb.cu(PDL_SEM_ACQREL);
        let one = kb.cu(1);
        let _ = kb.e(PDL_OP_ATOMIC_IADD, kb.uint, &[word, sc, sem, one]);
    });
}

/// Consumer: lane 0 spin-waits (CAPPED) on an acquire load of `cnt[0]` until it reaches `target`,
/// then a workgroup barrier releases the rest. A give-up increments `ovf[0]` and is INVALID.
fn pdl_consumer(kb: &mut Kb, cnt: u32, ovf: u32, target: u32, tid: u32) {
    let z = kb.cu(0);
    let is0 = kb.e(OP_IEQ, kb.bool_t, &[tid, z]);
    kb.iff(is0, |kb| {
        let word = pdl_word_ptr(kb, cnt, z);
        let sc = kb.cu(PDL_SCOPE_DEVICE);
        let sl = kb.cu(PDL_SEM_ACQ);
        let tgt = kb.cu(target);
        let cap = kb.cu(PDL_SPIN_CAP);
        let ctr = kb.fvar_u();
        kb.store_var(ctr, z);
        let (hdr, chk, bod, cont, merge) = (kb.id(), kb.id(), kb.id(), kb.id(), kb.id());
        kb.ins(OP_BRANCH, &[hdr]);
        kb.ins(OP_LABEL, &[hdr]);
        kb.ins(OP_LOOP_MERGE, &[merge, cont, 0]);
        kb.ins(OP_BRANCH, &[chk]);
        kb.ins(OP_LABEL, &[chk]);
        let i = kb.load_var_u(ctr);
        let got = kb.e(PDL_OP_ATOMIC_LOAD, kb.uint, &[word, sc, sl]);
        let ready = kb.e(OP_UGE, kb.bool_t, &[got, tgt]);
        let over = kb.e(OP_UGE, kb.bool_t, &[i, cap]);
        let stop = kb.e(OP_LOGICAL_OR, kb.bool_t, &[ready, over]);
        let go = kb.e(OP_LOGICAL_NOT, kb.bool_t, &[stop]);
        kb.ins(OP_BRANCH_CONDITIONAL, &[go, bod, merge]);
        kb.ins(OP_LABEL, &[bod]);
        kb.ins(OP_BRANCH, &[cont]);
        kb.ins(OP_LABEL, &[cont]);
        let i2 = kb.load_var_u(ctr);
        let one = kb.cu(1);
        let inext = kb.iadd(i2, one);
        kb.store_var(ctr, inext);
        kb.ins(OP_BRANCH, &[hdr]);
        kb.ins(OP_LABEL, &[merge]);
        // record a give-up: a capped spin is an INVALID edge, not a cheap one
        let i3 = kb.load_var_u(ctr);
        let tripped = kb.e(OP_UGE, kb.bool_t, &[i3, cap]);
        kb.iff(tripped, |kb| {
            let ow = pdl_word_ptr(kb, ovf, z);
            let sc2 = kb.cu(PDL_SCOPE_DEVICE);
            let sa = kb.cu(PDL_SEM_ACQREL);
            let one2 = kb.cu(1);
            let _ = kb.e(PDL_OP_ATOMIC_IADD, kb.uint, &[ow, sc2, sa, one2]);
        });
    });
    kb.wg_barrier();
}

/// Reset: lane 0 release-stores 0 into `cnt[0]`. Ordered after the consumer by the pipeline barrier
/// the decode path keeps between `gu` and `dn`, so the next token's producer starts from zero.
fn pdl_reset(kb: &mut Kb, cnt: u32, tid: u32) {
    let z = kb.cu(0);
    let is0 = kb.e(OP_IEQ, kb.bool_t, &[tid, z]);
    kb.iff(is0, |kb| {
        let word = pdl_word_ptr(kb, cnt, z);
        let sc = kb.cu(PDL_SCOPE_DEVICE);
        let sr = kb.cu(PDL_SEM_REL);
        kb.ins(PDL_OP_ATOMIC_STORE, &[word, sc, sr, z]);
    });
}

/// **Parametric RMSNorm — ONE dispatch of ONE workgroup**, the norm node for every decode site.
/// LocalSize 256, dispatched (1,1,1): each thread folds a strided partial sum of squares, a
/// shared-memory tree reduce closes it, every thread computes `1/√(Σ/n+eps)` and scales its
/// strided slice. Three flags select the site:
/// - `residual = Some(res_segs)` first folds the split-K stage-2 residual `Σ_{s<res_segs}
///   res[s·n+i]` into x in place (`SplitReduce.stage2_is_the_same_fold`; `res_segs = 1` is a plain
///   residual vector) — binding 2 = res.
/// - `quantize_out = true` scales into shared `nrm[]` and emits the DP4A activation chart in place
///   (`block_scale`+`actquant_pack` folded in, the f32 never lands in global) → bindings xq,
///   scales, xs; `false` scales `out[i] = x[i]·inv·g[i]` to a global f32 buffer (the f16-carrier
///   path). Bindings, in order: 0 = x, 1 = g, [2 = res if residual], then {out} | {xq, scales, xs}.
pub fn rmsnorm_module(
    n: u32,
    eps: f32,
    dpw: u32,
    residual: Option<u32>,
    quantize_out: bool,
) -> Vec<u32> {
    const WG: u32 = 256;
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let g = k.ssbo_f(1);
    let mut b = 2;
    let res = residual.map(|_| {
        let r = k.ssbo_f(b);
        b += 1;
        r
    });
    let (xq, scales, xs, out) = if quantize_out {
        (k.ssbo_u(b), k.ssbo_f(b + 1), k.ssbo_f(b + 2), 0)
    } else {
        (0, 0, 0, k.ssbo_f(b))
    };
    let (shv, shp) = k.shared_f32(WG);
    let (nrm, nrmp, sint) = if quantize_out {
        let (nrm, nrmp) = k.shared_f32(n);
        (nrm, nrmp, k.sint_ty())
    } else {
        (0, 0, 0)
    };
    let tid = k.gid_comp(0); // one workgroup: gid.x == local id
    let cwg = k.cu(WG);
    let cn = k.cu(n);
    // residual add (the folded stage-2 reduce), in place, same-lane
    if let (Some(res), Some(res_segs)) = (res, residual) {
        k.loop_step(tid, cn, cwg, |k, i| {
            let xv = k.load_f(x, i);
            let rv = if res_segs <= 1 {
                k.load_f(res, i)
            } else {
                let acc = k.fvar_f();
                let z = k.cf(0.0);
                k.store_var(acc, z);
                let c0 = k.cu(0);
                let cseg = k.cu(res_segs);
                k.loop_u(c0, cseg, |k, s| {
                    let cnn = k.cu(n);
                    let base = k.imul(s, cnn);
                    let idx = k.iadd(base, i);
                    let p = k.load_f(res, idx);
                    let a0 = k.load_var_f(acc);
                    let a1 = k.fadd(a0, p);
                    k.store_var(acc, a1);
                });
                k.load_var_f(acc)
            };
            let s = k.fadd(xv, rv);
            k.store_f(x, i, s);
        });
    }
    // strided partial sum of squares → shared tree reduce → inv rms → scale
    if quantize_out {
        k.rmsnorm_wg(x, g, shv, shp, n, eps, tid, WG, NormOut::Shared(nrm, nrmp));
        k.wg_barrier();
        k.emit_quantize_from_shared(nrm, nrmp, xq, scales, xs, n, dpw, sint, tid, WG, None);
    } else {
        k.rmsnorm_wg(x, g, shv, shp, n, eps, tid, WG, NormOut::Global(out));
    }
    k.finish(WG)
}

/// **The RMSNorm co-reduction folded into the DP4A GEMV — a lone reduction never runs alone**
/// (sanguine `FusionHidesLatency.fusion_hides_the_coreduction`). A standalone `rmsnorm_fused_q`
/// is a degree-2 reduction (`Σxᵢ²`, absmax) dispatched at pipeline depth 1 — `lone_reduction_is_
/// latency_bound` makes it `n·L`, MEASURED at 1 GB/s (159µs, ~11% of the decode step). Folded into
/// the consuming gemv — which runs transport-bound at depth `p ≥ L`, the `latency_hidden` regime —
/// the norm's sumsq/absmax reductions issue into the gemv's already-full weight-transport pipe and
/// the entire dependent stall `n·(L−1)` disappears. The megakernel's value was never fewer barriers;
/// it is that **x-in-shared lets the norm ride the gemv's reads**.
///
/// Each workgroup (LocalSize 256, `m.div_ceil(256)` × `segments` groups) recomputes the whole
/// norm+quant over shared x — redundant across WGs but sub-µs and barrier-free relative to the
/// weight transport (this is the co-reduction, NOT a separate latency-bound dispatch). The reduction
/// is byte-identical to `rmsnorm_module(n,eps,dpw,·,true)` at WG=256 (same strided sumsq + 256-tree)
/// and the quantize byte-identical to `emit_quantize_from_shared`, so the fused logits match the
/// `[rmsnorm_fused_q → gemv_dp4a_dyn]` pair bit-for-bit.
///
/// Bindings — `residual = None`: 0 = W (u32 codes, k-packed), 1 = x (f32 residual stream),
/// 2 = γ (ln gain, f32), 3 = OUT (f32), 4 = DM (u32 f16-packed d,m per block·row). `residual =
/// Some(res_segs)`: + 5 = res (f32 branch, split-K `res_segs` partials at `res[s·n+i]`), 6 = x_out
/// (f32, the updated residual stream `x + Σres`). The residual add persists to x_out **≠ x_in**
/// (every WG writes the identical sum — idempotent, no cross-WG read-after-write on x); the caller
/// ping-pongs x_out ↔ x_in. XQ/XS/SX are internal shared (the old separate bindings dissolve). The
/// DP4A inner loop, DM f16 unpack, and `d'·float(Σqx)·sx + m·Σx` epilogue mirror
/// `simt::gemv_dp4a_module_dyn` (dyn_sx form) exactly.
///
/// **Carrier is the dial — one kernel, every precision.** Chart carriers (Q8/Q4/Q2) fold in the
/// DP4A chart (`k % 32 == 0`, binding 4 = DM). Direct carriers (F16/F32) skip the quantize and
/// [`Kb::direct_fold`] the f32 `nrm` against the weight (F16 = `UnpackHalf2x16`, lane `r % 2`) —
/// no DM binding, so `residual`'s res/x_out shift to bindings 4/5. `m % pack == 0`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_norm_module(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: crate::simt::Carrier,
    eps: f32,
    residual: Option<u32>,
) -> Vec<u32> {
    gemv_norm_module_sc(m, k, local, segments, carrier, eps, residual, false)
}

/// [`gemv_norm_module`] with the **scale face** chosen — `two_level` reads `sub`/`sup`
/// (0.5 bits/weight) in place of `dm` (1.0). See [`crate::simt_nn::ChartScale`],
/// `results/where-the-27b-bits-go.md` and `results/scale-ladder.md`, which measures the
/// reconstruction RMS as IDENTICAL to six decimals — so this is half the metadata for nothing.
///
/// **Every binding is unchanged.** `sup` shares `dm`'s buffer at a compile-time element offset
/// (`ChartScale::two_level`), so binding 4 simply holds a different, smaller layout and the
/// residual pair stays at 5,6. The host's only change is to upload
/// [`crate::simt_nn::two_level_chart_buffer`]'s concatenation instead of `Proj::dm`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_norm_module_sc(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: crate::simt::Carrier,
    eps: f32,
    residual: Option<u32>,
    two_level: bool,
) -> Vec<u32> {
    let chart = carrier.is_chart();
    assert!(
        !two_level || chart,
        "the two-level scale face is a CHART face — carrier {} has no (d, lo) to split",
        carrier.name()
    );
    assert!(!chart || k % 32 == 0, "chart carriers need k % 32 == 0");
    assert!(m % carrier.pack() == 0, "m must divide the carrier pack");
    let _ = local; // WG forced to 256 to byte-match rmsnorm_fused's reduction tree
    const WG: u32 = 256;
    let n = k; // the norm/reduction dimension (`k` the builder is captured in closures)
    let dpw = carrier.pack() / 4; // chart dots per word: Q8→1 Q4→2 Q2→4
    let blocks = n / 32;

    let mut kb = Kb::new();
    if chart {
        kb.dp4a = true;
    }
    let uint = kb.uint;
    // bindings: 0=W (u32 codes / f16-packed, or f32 for F32), 1=x, 2=γ, 3=OUT. Chart adds 4=DM;
    // residual adds res + x_out after (index shifts by whether DM is present).
    let w = if carrier == crate::simt::Carrier::F32 {
        kb.ssbo_f(0)
    } else {
        kb.ssbo_u(0)
    };
    let xin = kb.ssbo_f(1);
    let gain = kb.ssbo_f(2);
    let out = kb.ssbo_f(3);
    let dm = if chart { kb.ssbo_u(4) } else { 0 }; // f16-packed (d, m), col-major idx b·M + r
                                                   // NOTE: the two-level face adds NO binding — its per-row super-pairs live at a compile-time
                                                   // offset inside this same buffer. See `ChartScale`.
    let rbase = if chart { 5 } else { 4 };
    let (res, xout) = match residual {
        Some(_) => (kb.ssbo_f(rbase), kb.ssbo_f(rbase + 1)),
        None => (0, 0),
    };
    // shared: normalized f32 + the reduce tree (both carriers); the DP4A chart (xq/sx/xs) only for
    // chart carriers — never lands in global.
    let (nrm, nrmp) = kb.shared_f32(n);
    let (red, redp) = kb.shared_f32(WG);
    let (sxs, sxsp, xss, xssp, xqs, xqsp) = if chart {
        let (a, ap) = kb.shared_f32(n / 32);
        let (b, bp) = kb.shared_f32(n / 32);
        let (c, cp) = kb.shared_u32(n / 4);
        (a, ap, b, bp, c, cp)
    } else {
        (0, 0, 0, 0, 0, 0)
    };

    let tid = kb.local_index(); // 0..256 within the WG (the cooperative-reduction lane)
    let r = kb.gid_comp(0); //     the output row (global)
    let cwg = kb.cu(WG);
    let cn = kb.cu(n);
    let normsrc = if residual.is_some() { xout } else { xin };

    // ── [DEC-PDL] consumer: wait for the `o` edge's completion counter BEFORE reading the
    //    activation `o` produced (the residual `res` below). Bindings 7 = counter, 8 = overflow
    //    are appended by `make_fused_gemv` on the `gu` node only; every other norm node has
    //    `pdl_role() == PDL_NONE` and emits the unchanged kernel. ──
    if pdl_role() == PDL_CONSUMER {
        let cnt = kb.ssbo_u(7);
        let ovf = kb.ssbo_u(8);
        // Prefetching the weight rows before the wait was measured a net loss (T8: recovery
        // ~35% -> 0-16%) and is not emitted.
        pdl_consumer(&mut kb, cnt, ovf, pdl_target(), tid);
    }

    // ── residual add → x_out (idempotent across WGs; x_out ≠ x_in), carrier-independent ──
    if let Some(res_segs) = residual {
        kb.loop_step(tid, cn, cwg, |k, i| {
            let xv = k.load_f(xin, i);
            let rv = if res_segs <= 1 {
                k.load_f(res, i)
            } else {
                let acc = k.fvar_f();
                let z = k.cf(0.0);
                k.store_var(acc, z);
                let c0 = k.cu(0);
                let cseg = k.cu(res_segs);
                k.loop_u(c0, cseg, |k, s| {
                    let cnn = k.cu(n);
                    let base = k.imul(s, cnn);
                    let idx = k.iadd(base, i);
                    let p = k.load_f(res, idx);
                    let a0 = k.load_var_f(acc);
                    let a1 = k.fadd(a0, p);
                    k.store_var(acc, a1);
                });
                k.load_var_f(acc)
            };
            let s = k.fadd(xv, rv);
            k.store_f(xout, i, s);
        });
    }

    // ── fused RMSNorm: normalize x into shared `nrm`. Chart also quantizes it to the DP4A chart
    //    (xq/sx/xs); direct (F16/F32) folds against the f32 `nrm` straight. ──
    if chart {
        kb.rmsnorm_quant_wg(
            normsrc, gain, red, redp, nrm, nrmp, xqs, xqsp, sxs, sxsp, xss, xssp, n, eps, dpw, tid,
            WG,
        );
    } else {
        kb.rmsnorm_wg(
            normsrc,
            gain,
            red,
            redp,
            n,
            eps,
            tid,
            WG,
            NormOut::Shared(nrm, nrmp),
        );
        kb.wg_barrier();
    }

    // ── the gemv fold: chart windows over 32-blocks (DP4A), direct over elements (FMA) ──
    let cm = kb.cu(m);
    let in_range = kb.ult(r, cm);
    let span = if chart { blocks } else { n };
    let (start, bound, out_idx) = if segments > 1 {
        let seg = kb.gid_comp(1);
        let seg_len = span.div_ceil(segments);
        let cseglen = kb.cu(seg_len);
        let w0 = kb.imul(seg, cseglen);
        let wend_raw = kb.iadd(w0, cseglen);
        let cspan = kb.cu(span);
        let over = kb.ult(cspan, wend_raw);
        let wend = kb.e(OP_SELECT, uint, &[over, cspan, wend_raw]);
        let oi1 = kb.imul(seg, cm);
        let oi2 = kb.iadd(oi1, r);
        (w0, wend, oi2)
    } else {
        let cspan = kb.cu(span);
        let c0 = kb.cu(0);
        (c0, cspan, r)
    };
    kb.iff(in_range, |kb| {
        let outv = if chart {
            let sc = if two_level {
                crate::simt_nn::ChartScale::two_level(dm, m, blocks)
            } else {
                crate::simt_nn::ChartScale::dm(dm)
            };
            kb.dp4a_chart_fold(
                w,
                sc,
                r,
                cm,
                start,
                bound,
                xqs,
                xqsp,
                sxs,
                sxsp,
                xss,
                xssp,
                carrier,
                true,
                None,
                carrier.plane_b_off(m, k),
            )
        } else {
            kb.direct_fold(w, r, start, bound, nrm, nrmp, m, carrier)
        };
        kb.store_f(out, out_idx, outv);
    });
    kb.finish(WG)
}

/// Which activation the fused [`gemv_actq_module`] produces before quantizing it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActQ {
    /// `x[i] = src[i]` — the attention output feeding the o-projection (was `quantize_vec`).
    Copy,
    /// `x[i] = silu(src[i]) · src[i + off]` — the SwiGLU gate feeding down (was `silu_mul_q`).
    SiluMul(u32),
    /// `x[i]` = the split-K flash-attention merge over `nsplit` partials for head `i / hd`, dim
    /// `i % hd` (was `flash_combine`). Bindings 1/4/5 = `part_o` / `part_m` / `part_l`.
    /// `(hd, nsplit)`.
    FlashCombine(u32, u32),
}

/// **`[DEC-FUSE]` — the activation producer + its int8 quantize folded INTO the consuming GEMV.**
/// `quantize_vec → gemv(o)` and `silu_mul_q → gemv(down)` were each a dispatch PAIR; this is the
/// same arithmetic as one. The motive is not arithmetic and not bytes — it is that a compute→compute
/// `vkCmdPipelineBarrier` on this box drains the pipeline for a MEASURED 3.71 µs, so at batch 1 the
/// token is barriers, not MACs (`DEC_BAR=9`: the same 268 dispatches run in 398 µs without them and
/// 1393 µs with them). Two dispatches per layer here is 48 barriers per token.
///
/// It is deliberately the SAME shape `gemv_norm_module` already proves: every workgroup redundantly
/// materializes the whole `k`-vector into shared and quantizes it there, then folds only its own
/// row-slab × split-K window. The redundancy is an L2 re-read of ≤ 39 KB per workgroup; the barrier
/// it removes is a full drain of 84 SMs. `dot_plane_sum`/`ternary_dot_two_mask` are untouched — the
/// int8 chart and the DP4A fold are byte-for-byte the ones `quantize_vec_module` and
/// `silu_mul_q_module` emitted, so the result is bit-identical to the pair.
///
/// Bindings: 0 = W, 1 = src (f32; `SiluMul` reads its `u` half at `off` in the SAME buffer, as
/// `silu_mul_q` was already bound), 2 = OUT, 3 = DM (chart only). `segments > 1` writes stage-1
/// partials at `out[seg·m + r]` for the consumer to fold — the `fold` shape, no stage 2.
pub fn gemv_actq_module(
    m: u32,
    k: u32,
    segments: u32,
    carrier: crate::simt::Carrier,
    act: ActQ,
) -> Vec<u32> {
    gemv_actq_module_sc(m, k, segments, carrier, act, false)
}

/// [`gemv_actq_module`] with the **scale face** chosen. Bindings are UNCHANGED — `sup` shares
/// `dm`'s buffer at a compile-time offset, so a [`ActQ::FlashCombine`] pair stays at 4,5. See
/// [`gemv_norm_module_sc`] and [`crate::simt_nn::ChartScale`].
pub fn gemv_actq_module_sc(
    m: u32,
    k: u32,
    segments: u32,
    carrier: crate::simt::Carrier,
    act: ActQ,
    two_level: bool,
) -> Vec<u32> {
    let chart = carrier.is_chart();
    assert!(
        chart,
        "gemv_actq is the chart (DP4A) path — the f16 carrier has no activation chart"
    );
    assert!(k % 32 == 0, "chart carriers need k % 32 == 0");
    const WG: u32 = 256;
    let n = k;
    let dpw = carrier.pack() / 4;
    let blocks = n / 32;

    let mut kb = Kb::new();
    kb.dp4a = true;
    let uint = kb.uint;
    let w = kb.ssbo_u(0);
    let src = kb.ssbo_f(1);
    let out = kb.ssbo_f(2);
    let dm = kb.ssbo_u(3);
    // NOTE: the two-level face adds NO binding — see `ChartScale`.
    let (pm, pl) = match act {
        ActQ::FlashCombine(..) => (kb.ssbo_f(4), kb.ssbo_f(5)),
        _ => (0, 0),
    };
    // ── the staging is sized to the SPLIT-K WINDOW, not to all of k ────────────────────────────
    //
    // The work below is already windowed to `[32·start, 32·bound)` — this function's own comment
    // says the fusion costs O(k/segments) per workgroup. The ALLOCATION was not: it reserved all of
    // `k` at **5.25 bytes per element** (4 for `nrm`, 1 for `xqs`, 0.25 for the two scale planes),
    // so `ffn_down` at k=17408 asked for **91 392 B against this device's 49 152 B**
    // `maxComputeSharedMemorySize` — and the driver accepts it silently
    // (`results/gemv-shared-is-over-the-limit.md`).
    //
    // MEASURED cost of that overrun, 27B in-chain, same mapping and same bytes, fused vs unfused:
    // `ffn_down` 536 → 763 GB/s, **1389 µs** (`results/shape-derived-gemv.md`). Unfusing collects it
    // but pays 80 dispatches; sizing the window collects it for nothing.
    //
    // At `segments = 1` this is `blocks` and `n` exactly as before, so every unsegmented kernel is
    // byte-identical rather than merely equivalent.
    // `DEC_GEMVWIN=0` restores the old all-of-`k` allocation. It exists ONLY so the overrun can be
    // A/B'd against itself under the median protocol of `results/best-of-3-is-not-a-measurement.md`
    // — the +5.4 % first recorded for this change was a single best-of-3 draw on a ±8.7 % estimator.
    // It is not a tuning dial: the windowed form is bit-identical, and the wide form is over the
    // device limit at any k > ~9362.
    let win_blocks = if std::env::var_os("DEC_GEMVWIN").is_some_and(|v| v == "0") {
        blocks
    } else {
        blocks.div_ceil(segments)
    };
    let win_n = win_blocks * 32;
    let (nrm, nrmp) = kb.shared_f32(win_n);
    let (sxs, sxsp) = kb.shared_f32(win_blocks);
    let (xss, xssp) = kb.shared_f32(win_blocks);
    let (xqs, xqsp) = kb.shared_u32(win_n / 4);

    let tid = kb.local_index();
    let r = kb.gid_comp(0);
    let cwg = kb.cu(WG);

    // ── this workgroup's split-K window, in 32-blocks. Computed FIRST: it masks the activation it
    //    has to produce, so the fusion costs O(k/segments) per workgroup, not O(k). ──
    let cm = kb.cu(m);
    let (start, bound, out_idx) = if segments > 1 {
        let seg = kb.gid_comp(1);
        let seg_len = blocks.div_ceil(segments);
        let cseglen = kb.cu(seg_len);
        let w0 = kb.imul(seg, cseglen);
        let wend_raw = kb.iadd(w0, cseglen);
        let cspan = kb.cu(blocks);
        let over = kb.ult(cspan, wend_raw);
        let wend = kb.e(OP_SELECT, uint, &[over, cspan, wend_raw]);
        let oi1 = kb.imul(seg, cm);
        let oi2 = kb.iadd(oi1, r);
        (w0, wend, oi2)
    } else {
        let cspan = kb.cu(blocks);
        let c0 = kb.cu(0);
        (c0, cspan, r)
    };

    // ── the activation over `[32·start, 32·bound)`, into shared f32 ──
    let c32 = kb.cu(32);
    let e0 = kb.imul(start, c32);
    let e1 = kb.imul(bound, c32);
    let etid = kb.iadd(e0, tid);
    kb.loop_step(etid, e1, cwg, |kb, i| {
        let v = match act {
            ActQ::Copy => kb.load_f(src, i),
            ActQ::SiluMul(off) => {
                let gv = kb.load_f(src, i);
                let coff = kb.cu(off);
                let ui = kb.iadd(i, coff);
                let uv = kb.load_f(src, ui);
                let f = kb.float;
                let ng = kb.e(OP_FNEG, f, &[gv]);
                let eg = kb.ext1(GLSL_EXP, ng);
                let one = kb.cf(1.0);
                let den = kb.fadd(one, eg);
                let sil = kb.fdiv(gv, den);
                kb.fmul(sil, uv)
            }
            // The split-K flash merge, instruction-for-instruction `flash_combine_module`: the same
            // two loops (global max over splits, then Σ w·o / Σ w·l with w = exp(m_sp − M)), the same
            // f32 association, the same `po[(h·nsplit+sp)·hd + d]` indexing. Only the dispatch is gone.
            ActQ::FlashCombine(hd, nsplit) => {
                let chd = kb.cu(hd);
                let h = kb.udiv(i, chd);
                let d = kb.umod(i, chd);
                let cns = kb.cu(nsplit);
                let hbase = kb.imul(h, cns);
                let gm = kb.fvar_f();
                let neg = kb.cf(-1.0e30);
                kb.store_var(gm, neg);
                let c0 = kb.cu(0);
                kb.loop_u(c0, cns, |k, sp| {
                    let hs = k.iadd(hbase, sp);
                    let mv = k.load_f(pm, hs);
                    let g0 = k.load_var_f(gm);
                    let g1 = k.ext2(GLSL_FMAX, g0, mv);
                    k.store_var(gm, g1);
                });
                let gmax = kb.load_var_f(gm);
                let onum = kb.fvar_f();
                let lden = kb.fvar_f();
                let z = kb.cf(0.0);
                kb.store_var(onum, z);
                kb.store_var(lden, z);
                kb.loop_u(c0, cns, |k, sp| {
                    let hs = k.iadd(hbase, sp);
                    let mv = k.load_f(pm, hs);
                    let dd = k.fsub(mv, gmax);
                    let wgt = k.ext1(GLSL_EXP, dd);
                    let lv = k.load_f(pl, hs);
                    let wl = k.fmul(wgt, lv);
                    let ld0 = k.load_var_f(lden);
                    let ld1 = k.fadd(ld0, wl);
                    k.store_var(lden, ld1);
                    let ob = k.imul(hs, chd);
                    let oi = k.iadd(ob, d);
                    let ov = k.load_f(src, oi);
                    let wo = k.fmul(wgt, ov);
                    let on0 = k.load_var_f(onum);
                    let on1 = k.fadd(on0, wo);
                    k.store_var(onum, on1);
                });
                let on = kb.load_var_f(onum);
                let ld = kb.load_var_f(lden);
                kb.fdiv(on, ld)
            }
        };
        // `i` is absolute over `[32·start, 32·bound)`; `nrm` now spans only the window
        let irel = if segments > 1 {
            let u = kb.uint;
            kb.e(OP_ISUB, u, &[i, e0])
        } else {
            i
        };
        kb.shared_store(nrm, nrmp, irel, v);
    });
    kb.wg_barrier();
    kb.emit_quantize_to_shared_win(
        nrm,
        nrmp,
        xqs,
        xqsp,
        sxs,
        sxsp,
        xss,
        xssp,
        dpw,
        tid,
        WG,
        start,
        bound,
        (segments > 1).then_some(start),
    );
    kb.wg_barrier();

    // ── the gemv fold over that same window ──
    let in_range = kb.ult(r, cm);
    kb.iff(in_range, |kb| {
        let sc = if two_level {
            crate::simt_nn::ChartScale::two_level(dm, m, blocks)
        } else {
            crate::simt_nn::ChartScale::dm(dm)
        };
        let outv = kb.dp4a_chart_fold(
            w,
            sc,
            r,
            cm,
            start,
            bound,
            xqs,
            xqsp,
            sxs,
            sxsp,
            xss,
            xssp,
            carrier,
            true,
            (segments > 1).then_some(start),
            carrier.plane_b_off(m, k),
        );
        kb.store_f(out, out_idx, outv);
    });

    // ── [DEC-PDL] the edge's two ends. Producer `o` (role=1) release-increments the counter after
    //    its last store (binding 6, appended by `make_actq_gemv`); resetter `dn` (role=3)
    //    release-stores 0 (binding 4). Any other actq node emits the unchanged kernel. ──
    match pdl_role() {
        PDL_PRODUCER => {
            let cnt = kb.ssbo_u(6);
            pdl_producer(&mut kb, cnt, tid);
        }
        PDL_RESET => {
            let cnt = kb.ssbo_u(4);
            pdl_reset(&mut kb, cnt, tid);
        }
        _ => {}
    }
    kb.finish(WG)
}

/// [`gemv_actq_module`] as bytes.
pub fn gemv_actq_bytes(
    m: u32,
    k: u32,
    segments: u32,
    carrier: crate::simt::Carrier,
    act: ActQ,
) -> Vec<u8> {
    gemv_actq_bytes_sc(m, k, segments, carrier, act, false)
}

/// [`gemv_actq_bytes`] with the **scale face** chosen — see [`gemv_actq_module_sc`]. Bindings are
/// unchanged; only binding 3's layout differs.
pub fn gemv_actq_bytes_sc(
    m: u32,
    k: u32,
    segments: u32,
    carrier: crate::simt::Carrier,
    act: ActQ,
    two_level: bool,
) -> Vec<u8> {
    bytes(gemv_actq_module_sc(m, k, segments, carrier, act, two_level))
}

/// Back-compat alias — the fused-norm gemv is carrier-generic now (F16 direct or chart DP4A);
/// the historical name forwards to [`gemv_norm_module`].
#[allow(clippy::too_many_arguments)]
pub fn gemv_dp4a_norm_module(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: crate::simt::Carrier,
    eps: f32,
    residual: Option<u32>,
) -> Vec<u32> {
    gemv_norm_module(m, k, local, segments, carrier, eps, residual)
}

/// **SiLU gate fused with the activation quantize — one dispatch, emits the DP4A chart.** The
/// down-proj GEMV read `silu → f32 → block_scale → pack → int8` (3 dispatches, an f32 round-trip);
/// here one workgroup computes `g/(1+e^{−g})·u` into shared and quantizes it in place, so the
/// f32 never lands in global. `u` at binding 1 offset `u_off` (the fused gate/up buffer). Bindings:
/// 0 = g, 1 = u, 2 = XQ, 3 = scales, 4 = XS. Dispatch (1,1,1). n = dff.
pub fn silu_mul_q_module(n: u32, u_off: u32, dpw: u32) -> Vec<u32> {
    const WG: u32 = 256;
    let mut k = Kb::new();
    let g = k.ssbo_f(0);
    let u = k.ssbo_f(1);
    let xq = k.ssbo_u(2);
    let scales = k.ssbo_f(3);
    let xs = k.ssbo_f(4);
    let (nrm, nrmp) = k.shared_f32(n);
    let sint = k.sint_ty();
    let tid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let cn = k.cu(n);
    let cuo = k.cu(u_off);
    k.loop_step(tid, cn, cwg, |k, i| {
        let gv = k.load_f(g, i);
        let ui = k.iadd(i, cuo);
        let uv = k.load_f(u, ui);
        let f = k.float;
        let ng = k.e(OP_FNEG, f, &[gv]);
        let eg = k.ext1(GLSL_EXP, ng);
        let one = k.cf(1.0);
        let den = k.fadd(one, eg);
        let sil = k.fdiv(gv, den);
        let ov = k.fmul(sil, uv);
        k.shared_store(nrm, nrmp, i, ov);
    });
    k.wg_barrier();
    k.emit_quantize_from_shared(nrm, nrmp, xq, scales, xs, n, dpw, sint, tid, WG, None);
    k.finish(WG)
}

/// **Fused vector quantize** — load a vector into shared, then per-32-block absmax→scale + pack
/// to the chart, in ONE workgroup/dispatch (replaces the block_scale + actquant_pack PAIR for the
/// attention output; the producer-fused quant `rmsnorm_fused_q`/`silu_mul_q` shape). Bindings:
/// 0 = in (f32), 1 = xq (packed codes), 2 = scales (sx per block), 3 = xs (block sums). Dispatch 1.
pub fn quantize_vec_module(n: u32, dpw: u32) -> Vec<u32> {
    const WG: u32 = 256;
    let mut k = Kb::new();
    let inp = k.ssbo_f(0);
    let xq = k.ssbo_u(1);
    let scales = k.ssbo_f(2);
    let xs = k.ssbo_f(3);
    let (nrm, nrmp) = k.shared_f32(n);
    let sint = k.sint_ty();
    let tid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let cn = k.cu(n);
    k.loop_step(tid, cn, cwg, |k, i| {
        let v = k.load_f(inp, i);
        k.shared_store(nrm, nrmp, i, v);
    });
    k.wg_barrier();
    k.emit_quantize_from_shared(nrm, nrmp, xq, scales, xs, n, dpw, sint, tid, WG, None);
    k.finish(WG)
}

/// **Embed gather from the chart (q4/q2)** — the tied weight stored ONCE in the DP4A k-packed
/// chart, gathered by row: `x[i] = d·q + lo`, `q = (W[(i/pack)·vocab + tok] >> (i%pack)·width)
/// & mask`, `(d, lo)` the f16-packed word at `DM[(i/32)·vocab + tok]`. Bindings: 0 = W (u32),
/// 1 = DM (u32 f16 pairs), 2 = params (`params[1]` = token), 3 = x out. Dispatch ≥ hidden.
pub fn embed_chart_module(
    hidden: u32,
    vocab: u32,
    local: u32,
    carrier: crate::simt::Carrier,
) -> Vec<u32> {
    let pack = carrier.pack();
    let width = 32 / pack;
    let mask = carrier.qmask();
    let mut k = Kb::new();
    let w = k.ssbo_u(0);
    let dm = k.ssbo_u(1);
    let params = k.ssbo_u(2);
    let x = k.ssbo_f(3);
    let i = k.gid_comp(0);
    let ch = k.cu(hidden);
    let guard = k.ult(i, ch);
    k.iff(guard, |k| {
        let c1 = k.cu(1);
        let tok = k.load_u(params, c1);
        let cv = k.cu(vocab);
        let cp = k.cu(pack);
        let iw = k.udiv(i, cp);
        let widx0 = k.imul(iw, cv);
        let widx = k.iadd(widx0, tok);
        let word = k.load_u(w, widx);
        let lane = k.umod(i, cp);
        let cwd = k.cu(width);
        let sh = k.imul(lane, cwd);
        let u = k.uint;
        let shifted = k.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[word, sh]);
        let cm = k.cu(mask);
        let q = k.e(OP_BITWISE_AND, u, &[shifted, cm]);
        let qf = k.utof(q);
        let c32 = k.cu(32);
        let b = k.udiv(i, c32);
        // col-major DM: idx = b·vocab + tok, one u32 (d16 | lo16<<16); val = d·q + lo
        let dmi0 = k.imul(b, cv);
        let dmi = k.iadd(dmi0, tok);
        let word2 = k.load_u(dm, dmi);
        let unp = {
            let (v2, g) = (k.vec2f, k.glsl);
            k.e(OP_EXT_INST, v2, &[g, GLSL_UNPACK_HALF2X16, word2])
        };
        let f = k.float;
        let d = k.e(OP_COMPOSITE_EXTRACT, f, &[unp, 0]);
        let lo = k.e(OP_COMPOSITE_EXTRACT, f, &[unp, 1]);
        let dq = k.fmul(d, qf);
        let val = k.fadd(dq, lo);
        k.store_f(x, i, val);
    });
    k.finish(local)
}
