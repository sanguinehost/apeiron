//! simt_nn::pool — the delta fill: copy the MISSED blocks of the cold KV into the compact pool
//! `flash_partial` reads under `compact`.
//!
//! `flash_partial`'s compact addressing already assumes this buffer exists — "with a scattered
//! upload writing kept block `j` into staging slot `j`, key `v` lives at `j·blk + off`, so the
//! resident K/V is O(budget)". Nothing wrote that upload, which is why `DEC_KNOT_COMPACT` is
//! gated off with "would read uninitialised staging".
//!
//! The naive filler copies every kept block every step. It does not have to: a stored key is not
//! a function of the step that reads it, so a block already in the pool needs no second fetch
//! however far the selection moved (Sanguine `ColdTierDecode.delta_serve_is_exact`). The caller
//! decides WHICH blocks are missing — `bedrock::serve::model::kvpool::KvPool::plan`, whose
//! capacity bound `P ≥ |kept|` is `victims_suffice_iff_pool_covers_shortlist` — and this kernel
//! moves exactly those. Measured on the prototype: 89.93% of a step's selection is carried over,
//! so the fill is a tenth of the naive upload (1168.6 → 60.0 MiB/step, 19.5×).
//!
//! The three planes are copied in their OWN layouts, which differ and are not negotiable:
//! K is word-i-major `kcq[(kvh·(hd/4) + w)·tmax + pos]` (the DP4A plane `fused_rope_kv` packs),
//! V is row-major `vc[(kvh·tmax + pos)·hd + i]`, and the K absmax scale is `sk[kvh·tmax + pos]`.
//! The pool mirrors each with `tmax → pblocks·blk`, so the only change is the stride: a fill is a
//! pure copy and cannot alter a value.

use super::*;

/// **The delta fill.** One WG per KV-HEAD, looping over that head's fills.
///
/// It was one WG per (fill, kv-head) first, which dispatches `maxfill x nkv` workgroups every
/// layer whether or not there is anything to move. DEC_PROF measured that: 1237.6 µs/token, 43.3%
/// of all GPU time, 51.6 µs per dispatch against theta_scan's 4 — for a steady state that fills
/// ONE block, because carryover is high and only the unsealed block must be refetched. The guard
/// made each idle workgroup cheap but not free, and 64 of them per layer times 24 layers is not
/// free at all. Looping inside the workgroup makes the cost track the fills instead of the cap.
///
/// `fills[(kvh·maxfill + i)·2]` is the SOURCE block in the cold store, `+1` the DESTINATION slot.
///
/// Bindings: 0=kcq(src K u32) 1=vc(src V f16) 2=sk(src scale f32) 3=fills(u32 pairs)
/// 4=pk(pool K u32) 5=pv(pool V f16) 6=psk(pool scale f32) 7=nfill(u32, per kv-head).
/// **THE DISPATCH THIS KERNEL REQUIRES — derived, not transcribed.**
///
/// `pool_fill_module` decodes its workgroup id as `kvh = wg / NPART`, so it needs exactly
/// `nkv·NPART` workgroups. It was reworked from one-WG-per-(fill, kv-head) to NPART-per-kv-head
/// (see the latency note inside it) and the DOCSTRING was not updated, so the caller kept
/// dispatching `max_fills·nkv`. On Qwen3.5-0.8B (`nkv = 2`, needing 32) that is 8 to 16 workgroups,
/// every one of them landing on `kvh = 0` — so **kv-head 1's pool was never written at all**, and
/// half the q-heads attended over uninitialised device memory. Compact scored min corr 0.11919
/// against 0.99484 at every pool size, because the dominant error term does not depend on the pool
/// size at all.
///
/// A caller that computes this itself is a second copy of `NPART`; there is one, here.
pub const POOL_FILL_NPART: u32 = 16;

/// Workgroups `pool_fill_module` must be dispatched with.
pub fn pool_fill_groups(nkv: u32) -> u32 {
    nkv * POOL_FILL_NPART
}

/// Dispatch **[`pool_fill_groups`]`(nkv)`**, i.e. `nkv·NPART` workgroups — NOT `max_fills·nkv`,
/// which is what this line said until 2026-08-18 and what the caller obeyed.
pub fn pool_fill_module(
    nkv: u32,
    hd: u32,
    tmax: u32,
    blk: u32,
    pblocks: u32,
    maxfill: u32,
) -> Vec<u32> {
    const WG: u32 = 64;
    // WORKGROUPS PER KV-HEAD. The copy is LATENCY-bound, not bandwidth-bound: the source is host
    // memory across PCIe (~1 µs a round trip), and one workgroup per kv-head is 128 threads total
    // serialising ~32 dependent loads each — 32 µs measured per dispatch to move 25 KB, i.e. 0.76
    // GB/s against a 12.9 GB/s link. Splitting the element range across NPART workgroups puts
    // NPART x more loads in flight, which is the only thing that hides a fixed latency.
    const NPART: u32 = 16;
    debug_assert_eq!(
        NPART, POOL_FILL_NPART,
        "the dispatch helper and the kernel must agree"
    );
    let hd4 = hd / 4;
    let hd2 = hd / 2; // f16 per position, counted in u32 WORDS
    let pb = pblocks * blk;
    let mut k = Kb::new();
    let kcq = k.ssbo_u(0);
    // V is copied, not read: binding it as u32 moves two halves per instruction and drops the
    // f16 -> f32 -> f16 round-trip the typed path emits for a pure copy. `hashtrinity`'s own
    // assembly note is the rule — the packed form IS the operand form, nothing is unpacked.
    let vc = k.ssbo_u(1);
    let sk = k.ssbo_f(2);
    let fills = k.ssbo_u(3);
    let pk = k.ssbo_u(4);
    let pv = k.ssbo_u(5);
    let psk = k.ssbo_f(6);
    let nfillb = k.ssbo_u(7);
    let gid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let cnp = k.cu(NPART);
    let wg = k.udiv(gid, cwg);
    let lane = k.umod(gid, cwg);
    let kvh = k.udiv(wg, cnp);
    let part = k.umod(wg, cnp);
    // this workgroup's slot in the partitioned grid-stride: lane + part·WG, striding NPART·WG
    let base = k.imul(part, cwg);
    let tid = k.iadd(base, lane);
    let cstride = k.cu(NPART * WG);
    let cnkv = k.cu(nkv);
    let c0 = k.cu(0);
    let inrange = k.ult(kvh, cnkv);
    k.iff(inrange, |mut k| {
        let nfill = k.load_u(nfillb, kvh);
        k.loop_u(c0, nfill, |mut k, f| {
            let c1 = k.cu(1);
            let c2 = k.cu(2);
            let cmf = k.cu(maxfill);
            let fbase = k.imul(kvh, cmf);
            let fe = k.iadd(fbase, f);
            let fi = k.imul(fe, c2);
            let sblk = k.load_u(fills, fi);
            let fi1 = k.iadd(fi, c1);
            let dslot = k.load_u(fills, fi1);
            let cblk = k.cu(blk);
            let spos0 = k.imul(sblk, cblk);
            let dpos0 = k.imul(dslot, cblk);
            let chd4 = k.cu(hd4);
            let chd2 = k.cu(hd2);
            let ctm = k.cu(tmax);
            let cpb = k.cu(pb);

            // K — hd/4 runs of `blk` CONTIGUOUS words (row stride tmax / pb). The old form
            // flattened (w, off) into one index and paid a UDiv and a UMod per element to get
            // them back; both operands are powers of two and the driver still emitted the
            // divides. Nesting the loops means the address is two adds.
            let kbase = k.imul(kvh, chd4);
            k.loop_u(c0, chd4, |mut k, w| {
                let row = k.iadd(kbase, w);
                let srow = k.imul(row, ctm);
                let drow = k.imul(row, cpb);
                let sb = k.iadd(srow, spos0);
                let db = k.iadd(drow, dpos0);
                k.loop_step(tid, cblk, cstride, |k, off| {
                    let si = k.iadd(sb, off);
                    let di = k.iadd(db, off);
                    let v = k.load_u(kcq, si);
                    k.store_u(pk, di, v);
                });
                k
            });

            // V — ONE contiguous run. `vc[(kvh·tmax + pos)·hd + i]` over `blk` consecutive
            // positions is `blk·hd` consecutive halves, so in words it is a flat memcpy: one add
            // for the source, one for the destination, no division and no conversion. This is
            // llama.cpp's `contig_copy` shape (vulkan-shaders/contig_copy.comp: grid-stride,
            // `data_d[idx] = data_a[idx]`, unrolled) rather than a strided element loop.
            let vsb0 = k.imul(kvh, ctm);
            let vs1 = k.iadd(vsb0, spos0);
            let vsw = k.imul(vs1, chd2);
            let vdb0 = k.imul(kvh, cpb);
            let vd1 = k.iadd(vdb0, dpos0);
            let vdw = k.imul(vd1, chd2);
            let nvw = k.cu(blk * hd2);
            k.loop_step(tid, nvw, cstride, |k, w| {
                let si = k.iadd(vsw, w);
                let di = k.iadd(vdw, w);
                let v = k.load_u(vc, si);
                k.store_u(pv, di, v);
            });

            // the per-token K absmax — `blk` contiguous floats, and the score is wrong without it
            let ssb = k.iadd(vsb0, spos0);
            let sdb = k.iadd(vdb0, dpos0);
            k.loop_step(tid, cblk, cstride, |k, off| {
                let si = k.iadd(ssb, off);
                let di = k.iadd(sdb, off);
                let v = k.load_f(sk, si);
                k.store_f(psk, di, v);
            });
            k
        });
        k
    });
    k.finish(WG)
}

/// `pool_fill_module` as bytes.
pub fn pool_fill_bytes(
    nkv: u32,
    hd: u32,
    tmax: u32,
    blk: u32,
    pblocks: u32,
    maxfill: u32,
) -> Vec<u8> {
    bytes(pool_fill_module(nkv, hd, tmax, blk, pblocks, maxfill))
}
