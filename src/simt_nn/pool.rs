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
    k.iff(inrange, |k| {
        let nfill = k.load_u(nfillb, kvh);
        k.loop_u(c0, nfill, |k, f| {
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
            k.loop_u(c0, chd4, |k, w| {
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
        });
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

/// **The device-side allocator** — residency for the delta gather, with the map never leaving the
/// card.
///
/// The kept list is produced by `theta_select` on device. Planning on the HOST would mean a
/// readback per layer per decode step, which is the sync cost coldkv measured and warned about, so
/// this decides residency where the list already is. `bedrock::serve::model::kvpool::KvPool` is the
/// same algorithm in Rust and is the readable reference for it (and is unit-tested against the
/// theorem); this is its device twin.
///
/// **One workgroup per kv-head**, so each owns its pool, slot map and fill list and no two
/// invocations touch the same word — which is what makes it correct without atomics, since the
/// builder has none.
///
/// It ran at LocalSize 1 first, and that was measured, not assumed: it cost 740.6 -> 360.9 tok/s
/// with host KV, because ONE lane walked `group x |kept|` entries per layer, 24 layers deep. Pass 1
/// is embarrassingly parallel (a probe and a stamp per entry), so it runs across the workgroup;
/// only the allocation is order-dependent, and with high carryover there are one or two misses to
/// place. So the serial region is a scan for the few entries pass 1 left unassigned, not the whole
/// list.
///
/// Equality is not in the instruction set here, and none is needed. Valid slots are `< P` and
/// absent is `0xFFFFFFFF`, so `P-1 < s` is exactly the miss test; stamps are `≤ step`, so
/// `lu < step` is exactly "not wanted by this step" — the one slot class eviction may never take
/// (`ColdTierDecode.victims_suffice_iff_pool_covers_shortlist`, whose `P ≥ |kept|` makes such a
/// slot always exist).
///
/// Bindings: 0=kept 1=nkept 2=slot_of(u32, in/out) 3=block_at(in/out) 4=last_used(in/out)
/// 5=slot(out) 6=fills(out, pairs) 7=nfill(out, per kv-head) 8=params(`params[0]` = step).
/// Dispatch `(nkv, 1, 1)`.
pub fn pool_alloc_module(
    nh: u32,
    nkv: u32,
    nblocks: u32,
    maxblk: u32,
    pslots: u32,
    maxfill: u32,
    blk: u32,
) -> Vec<u32> {
    const EMPTY: u32 = 0xFFFF_FFFF;
    let group = nh / nkv.max(1);
    let mut k = Kb::new();
    let kept = k.ssbo_u(0);
    let nkept = k.ssbo_u(1);
    let slot_of = k.ssbo_u(2);
    let block_at = k.ssbo_u(3);
    let last_used = k.ssbo_u(4);
    let slot = k.ssbo_u(5);
    let fills = k.ssbo_u(6);
    let nfill = k.ssbo_u(7);
    let params = k.ssbo_u(8);
    const WG: u32 = 64;
    let gid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let kvh = k.udiv(gid, cwg);
    let lane = k.umod(gid, cwg);
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let c2 = k.cu(2);
    let cempty = k.cu(EMPTY);
    let cbig = k.cu(EMPTY);
    let cbigm1 = k.cu(EMPTY - 1);
    let cp = k.cu(pslots);
    let cpm1 = k.cu(pslots - 1);
    let cmb = k.cu(maxblk);
    let cnb = k.cu(nblocks);
    let cgroup = k.cu(group);
    let cmf = k.cu(maxfill);
    let cnkv = k.cu(nkv);
    // `params[0]` is the POSITION, which is monotone per token — the stamp this needs. +1
    // because `last_used` starts at 0 and `lu < step` is the not-wanted-this-step test, so a
    // step of 0 would find no evictable slot on the very first token.
    let pos0 = k.load_u(params, c0);
    let step = k.iadd(pos0, c1);
    // THE ACTIVE BLOCK IS NOT CACHEABLE. `delta_serve_is_exact` licenses reusing bytes because a
    // stored key is not a function of the step that reads it — true of a SEALED block, false of
    // the partial one holding the current token, which `fused_rope_kv` rewrites every step. Cached
    // once it would be served stale for the rest of the generation. This is the same seam the CPU
    // tier draws with `cold_seal_range` (spill AT SEAL); here the active block is simply
    // invalidated each step, so it re-fills. One block of traffic per step.
    let cblk = k.cu(blk);
    let active = k.udiv(pos0, cblk);
    let live = k.ult(kvh, cnkv);
    k.iff(live, |k| {
        let sob = k.imul(kvh, cmb); // this head's slot_of base
        let pb = k.imul(kvh, cp); // this head's pool base
        let fb = k.imul(kvh, cmf); // this head's fill-list base
        let nf = k.fvar_u();
        k.store_var(nf, c0);
        let hbase = k.imul(kvh, cgroup);
        // PASS 1 — stamp the HITS, and only then allocate. Single-pass is wrong and the device
        // check caught it at 402 slot mismatches: a miss early in the list would evict a slot a
        // LATER entry of the same step still wants, because that entry had not been stamped yet.
        // Stamping first is what makes `P >= |kept|` sufficient
        // (`ColdTierDecode.victims_suffice_iff_pool_covers_shortlist`) — after this loop every
        // slot backing this step's selection carries `step`, and nothing else does.
        k.loop_u(c0, cgroup, |k, hr| {
            let h = k.iadd(hbase, hr);
            let n = k.load_u(nkept, h);
            let krow = k.imul(h, cnb);
            k.loop_step(lane, n, cwg, |k, j| {
                let ki = k.iadd(krow, j);
                let b = k.load_u(kept, ki);
                let sidx = k.iadd(sob, b);
                // invalidate the unsealed block so pass 2 re-fetches it
                let sealed = k.ult(b, active);
                let inv = k.e(OP_SELECT, k.uint, &[sealed, c0, c1]);
                let unsealed = k.ult(c0, inv);
                k.iff(unsealed, |k| {
                    // ── CLEAR BOTH SIDES. `slot_of` and `block_at` are a BIJECTION ─────────────
                    // `slot_of[b] = s ⟺ block_at[s] = b` is the invariant `KvPool::plan` keeps for
                    // free, because there a block only ever leaves a slot by EVICTION and the
                    // victim's `block_at` is overwritten by the new tenant in the same breath. The
                    // active block is different: it is re-homed EVERY step, and clearing only
                    // `slot_of` left `block_at[old_slot]` still naming it — a trail of stale
                    // back-pointers all claiming the same block.
                    //
                    // The debris detonates in pass 2, not here. Its eviction clears
                    // `slot_of[block_at[victim]]` unconditionally; when the victim is one of these
                    // stale slots, `block_at` names a block that now legitimately lives elsewhere
                    // — possibly one mapped EARLIER IN THIS SAME STEP — and its fresh mapping is
                    // wiped. The `again` dedupe then re-reads EMPTY, re-allocates the same block,
                    // evicts another slot, and cascades until fills overrun `maxfill` (silently
                    // dropped: "a dropped fill is a wrong READ") or the no-evictable-slot fallback
                    // proceeds with `best` still at its initialiser 0, evicting a slot this step is
                    // using.
                    //
                    // MEASURED: compact was bit-identical to the non-compact reference for 67
                    // positions and diverged in one deterministic step at pos 68 (P=3) / pos 129
                    // (P=5) — both predicted exactly by a simulation of this algorithm, and both
                    // the measured worst position. The cascade needs ≥2 kept blocks and a trail to
                    // land in, which is why position 0 was always clean and why it read as
                    // "degrades as context grows": one wrong attention step perturbs the delta-net
                    // carry and the error persists.
                    //
                    // Idempotent across the group's duplicate entries and race-free: every racing
                    // lane writes the same EMPTY.
                    let s0pre = k.load_u(slot_of, sidx);
                    let had = k.ult(s0pre, cp);
                    k.iff(had, |k| {
                        let bidx = k.iadd(pb, s0pre);
                        k.store_u(block_at, bidx, cempty);
                    });
                    k.store_u(slot_of, sidx, cempty);
                });
                let s0 = k.load_u(slot_of, sidx);
                let hit = k.ult(s0, cp);
                k.iff(hit, |k| {
                    let lidx = k.iadd(pb, s0);
                    k.store_u(last_used, lidx, step);
                });
                // EMPTY here marks "still owed a slot" for pass 2 to find.
                k.store_u(slot, ki, s0);
            });
        });
        // Every hit is stamped before any allocation runs — the invariant `P >= |kept|` rests on
        // (`victims_suffice_iff_pool_covers_shortlist`). The barrier is what makes that true
        // ACROSS lanes now that pass 1 is parallel.
        k.wg_barrier();
        // PASS 2 — place the misses, on one lane: allocation is order-dependent (each victim
        // choice changes the next), and with carryover this high there are one or two of them.
        let solo = k.ult(lane, c1);
        k.iff(solo, |k| {
            k.loop_u(c0, cgroup, |k, hr| {
                let h = k.iadd(hbase, hr);
                let n = k.load_u(nkept, h);
                let krow = k.imul(h, cnb);
                k.loop_u(c0, n, |k, j| {
                    let ki = k.iadd(krow, j);
                    let cur = k.load_u(slot, ki);
                    let miss = k.ult(cpm1, cur);
                    k.iff(miss, |k| {
                        let b = k.load_u(kept, ki);
                        let sidx = k.iadd(sob, b);
                        // A block can appear in two heads of the same group. The first miss fills it,
                        // so re-read before allocating or the same block takes two slots and is
                        // fetched twice.
                        let again = k.load_u(slot_of, sidx);
                        let still = k.ult(cpm1, again);
                        let sv = k.fvar_u();
                        k.store_var(sv, again);
                        k.iff(still, |k| {
                            // the least-recently-selected slot NOT stamped by this step
                            let best = k.fvar_u();
                            k.store_var(best, c0);
                            let bestv = k.fvar_u();
                            k.store_var(bestv, cbig);
                            k.loop_u(c0, cp, |k, t| {
                                let lidx = k.iadd(pb, t);
                                let lu = k.load_u(last_used, lidx);
                                let free_now = k.ult(lu, step);
                                k.iff(free_now, |k| {
                                    let bv = k.load_var_u(bestv);
                                    let better = k.ult(lu, bv);
                                    k.iff(better, |k| {
                                        k.store_var(bestv, lu);
                                        k.store_var(best, t);
                                    });
                                });
                            });
                            // NO EVICTABLE SLOT — every slot is already wanted by this step, i.e.
                            // P < |kept|. `victims_suffice_iff_pool_covers_shortlist` is an IFF, so
                            // this is not a tuning miss but the bound being violated, and the read
                            // that follows is WRONG. `KvPool::plan` asserts; a shader cannot, so it
                            // records instead: `nfill[nkv + kvh]` is this head's overflow count, and a
                            // nonzero one invalidates every number from the run.
                            let bv2 = k.load_var_u(bestv);
                            let none_free = k.ult(cbigm1, bv2);
                            k.iff(none_free, |k| {
                                let oi = k.iadd(cnkv, kvh);
                                let prev = k.load_u(nfill, oi);
                                let nx = k.iadd(prev, c1);
                                k.store_u(nfill, oi, nx);
                            });
                            let bs = k.load_var_u(best);
                            let bidx = k.iadd(pb, bs);
                            // evict what it held, so no block maps to a slot it no longer owns
                            let oldb = k.load_u(block_at, bidx);
                            let hasold = k.ult(oldb, cmb);
                            k.iff(hasold, |k| {
                                let oidx = k.iadd(sob, oldb);
                                k.store_u(slot_of, oidx, cempty);
                            });
                            k.store_u(block_at, bidx, b);
                            k.store_u(slot_of, sidx, bs);
                            // append the fill, bounded: an overrun corrupts a neighbour's list and a
                            // dropped fill is a wrong READ, so the cap is checked, not assumed.
                            let n0 = k.load_var_u(nf);
                            let room = k.ult(n0, cmf);
                            k.iff(room, |k| {
                                let e = k.iadd(fb, n0);
                                let e2 = k.imul(e, c2);
                                k.store_u(fills, e2, b);
                                let e2b = k.iadd(e2, c1);
                                k.store_u(fills, e2b, bs);
                                let n1 = k.iadd(n0, c1);
                                k.store_var(nf, n1);
                            });
                            k.store_var(sv, bs);
                        });
                        let sfin = k.load_var_u(sv);
                        let fidx = k.iadd(pb, sfin);
                        k.store_u(last_used, fidx, step);
                        k.store_u(slot, ki, sfin);
                    });
                });
            });
            let nfv = k.load_var_u(nf);
            k.store_u(nfill, kvh, nfv);
        });
    });
    k.finish(WG)
}

/// `pool_alloc_module` as bytes.
pub fn pool_alloc_bytes(
    nh: u32,
    nkv: u32,
    nblocks: u32,
    maxblk: u32,
    pslots: u32,
    maxfill: u32,
    blk: u32,
) -> Vec<u8> {
    bytes(pool_alloc_module(
        nh, nkv, nblocks, maxblk, pslots, maxfill, blk,
    ))
}
