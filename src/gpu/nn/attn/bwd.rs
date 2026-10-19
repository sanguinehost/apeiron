//! simt_nn::attn::bwd — the causal GQA attention backward as f32 SPIR-V, the twin of
//! `learn::grad::attn::attn_bwd_kern`.
//!
//! Five deterministic dispatches, no float atomics, every output sums in a fixed association:
//! `ds`, `dq` and `dk`/`dv` sum per-thread in ascending order; `rowsum` sums each 16-key tile in
//! ascending `j`, then the tile partials in ascending tile order.
//!
//! 1. `rowsum`: `r[h,i] = Σ_{j≤i} P[h,i,j]·Σ_e dO[i,h,e]·V[j,h/rep,e]`, one 16×16 `(i,j)` tile
//!    per workgroup of 256 threads; the tile's `dout` and `v` rows are staged in shared memory,
//!    the 16 per-row tile partials go to `rowsum_parts`, and a second dispatch sums those
//!    partials in ascending tile order into `rowsum`.
//! 2. `ds`: `dS[h,i,j] = P[h,i,j]·(dP_ij − r[h,i])` with `dP_ij = Σ_e dO[i,h,e]·V[j,h/rep,e]`,
//!    one 16×16 output tile per workgroup of 256 threads; the tile's `dout` and `v` rows are
//!    staged in shared memory, `j > i` writes 0.
//! 3. `dq`: `dq[i,h,d] = scale·Σ_{j≤i} dS[h,i,j]·K[j,h/rep,d]`, one thread per `(i,h,d)`.
//! 4. `dk`/`dv`: `dk[j,g,d] = scale·Σ_{h∈g} Σ_{i≥j} dS[h,i,j]·Q[i,h,d]`,
//!    `dv[j,g,d] = Σ_{h∈g} Σ_{i≥j} P[h,i,j]·dO[i,h,d]`, one workgroup of 256 threads per
//!    `(j,g,d-tile)`: 32 lanes per `d` (`d = dtile·32 + lane`) and eight lane groups that split
//!    the `i` walk, reduced through a fixed `8×32` shared array per output.
//!
//! Dispatch 2 materialises the score gradient once and is tiled: workgroup `(h, ti, tj)` stages
//! the 16 `dout` rows of `i ∈ [16·ti, 16·ti+16)` and the 16 `v` rows of
//! `j ∈ [16·tj, 16·tj+16)` into two `16·hd`-word shared arrays, one barrier, then thread
//! `tid = 16·ty + tx` computes the dot for `(i,j) = (16·ti+ty, 16·tj+tx)`. Every staged `v` word
//! is read by 16 threads and every staged `dout` word likewise, so the dispatch reads each operand
//! word `⌈t/16⌉` times instead of once per `(i,j)` pair. Before the tile, `dq` and `dk`/`dv` each
//! recomputed `dP_ij` in thread; the materialised form pays the staged traffic in dispatch 2 and
//! one `ds` word per visit in flight 3 and 4. The arithmetic order is unchanged: `dP_ij` sums `e`
//! ascending with fused multiply-add from zero over the staged words (bit-identical operands), `dS`
//! rounds once, `dq` sums `j` ascending and `dk`/`dv` sum `h` then `i` ascending, so the f32 values
//! are bit-identical to the untiled form.
//!
//! The shapes `t`, `nh`, `nkv`, `hd` are module constants, the reduction bounds are compile-time
//! integers, and the `rep = nh/nkv` query heads of a kv head are folded into one thread's
//! accumulator in ascending head order.
//!
//! Bindings follow the order of the twin's tensors; the layouts are `q`/`dout` token-major
//! `[(i·nh+h)·hd+d]`, `k`/`v` `[(j·nkv+g)·hd+d]`, `P`/`dS` `[(h·t+i)·t+j]`, `rowsum[h·t+i]` and
//! the tile partials `rowsum_parts[(h·t+i)·⌈t/16⌉+tj]`. The five dispatches run in one command
//! buffer with a compute→compute barrier after each, so `rowsum` is visible to `ds`, and `dS` is
//! visible to `dq` and `dk`/`dv`.

use super::*;

/// **Attention-backward row sum (f32)** — dispatch 1 of the five, the twin of `attn_bwd_kern`'s
/// `rowsum(dP ⊙ P)` materialised as a 16×16 `(i,j)` tile plus a partials reduce. Workgroup
/// `(h, ti, tj)` of `nh·⌈t/16⌉²` owns the tile `i ∈ [16·ti, 16·ti+16)`,
/// `j ∈ [16·tj, 16·tj+16)`; thread `tid = 16·ty + tx` owns `(i,j) = (16·ti+ty, 16·tj+tx)`.
///
/// Staging (the `ds` pattern): every thread strides the `TILE·hd` positions, position `s` standing
/// for `(row, e) = (s/hd, s%hd)`; it writes `sdout[s] = dout[((ibase+row)·nh+h)·hd + e]` and
/// `sv[s] = v[((jbase+row)·nkv+h/rep)·hd + e]`, with a row past `t` left zero. One
/// [`Kb::wg_barrier`] makes both tiles visible. A thread with `i < t`, `j < t` and `j ≤ i` then
/// computes `dP_ij = Σ_e sdout[ty·hd+e]·sv[tx·hd+e]` with `e` ascending and fused multiply-add —
/// the same f32 words in the same order as the global-load body — and its per-key term
/// `P[h,i,j]·dP_ij`; every other thread contributes `0`. The 256 terms go to a 16×16 shared array;
/// after a second barrier thread `tid < 16` sums row `tid` in ascending `tx` — the 16 keys of
/// this tile — and writes `rowsum_parts[(h·t+16·ti+tid)·⌈t/16⌉ + tj]`.
///
/// The `j` association is a per-tile ascending 16-term sum followed by an ascending partial sum
/// over tiles (in [`attn_bwd_rowsum_reduce_module`]); the strided-lane 256-term tree is retired.
/// The `e` order and the `P·dP` form are unchanged, so `dP_ij` is bit-identical to the `ds` pass's
/// and to the former body; only the `j` sum re-associates.
///
/// Bindings: 0 = v (f32, `t·nkv·hd`), 1 = p (f32, `nh·t·t`), 2 = dout (f32, `t·nh·hd`),
/// 3 = rowsum_parts (f32, `nh·t·⌈t/16⌉`). Grid `nh·⌈t/16⌉²`, LocalSize 256.
///
/// Complexity: workgroup `(h,ti,tj)` stages `min(16, t−16·ti)·hd` words of `dout` and
/// `min(16, t−16·tj)·hd` of `v` (each valid row once), reads one `p` word per causal thread and
/// writes at most 16 partial words; summed over tiles the operand reads are `⌈t/16⌉·t·hd` each,
/// the `p` reads are the `t(t+1)/2` causal pairs per head and the partial writes are `t·⌈t/16⌉`
/// per head. The dispatch therefore moves `2·nh·⌈t/16⌉·t·hd + nh·t(t+1)/2 + nh·t·⌈t/16⌉` words
/// (plus workgroup-memory shared stores), against the P94 form's
/// `nh·(t(t+1)·hd + t(t+1)/2)`: at `hd = 32, t = 512, nh = 4` that is 4,850,176 against
/// 34,145,280 words, 7.0× less.
pub fn attn_bwd_rowsum_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    const TILE: u32 = 16;
    assert!(t > 0 && hd > 0, "attn_bwd_rowsum: shape must be non-zero");
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_bwd_rowsum: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let v = k.ssbo_f(0);
    let p = k.ssbo_f(1);
    let dout = k.ssbo_f(2);
    let parts = k.ssbo_f(3);
    let tiles = t.div_ceil(TILE);
    // Two 16-row operand tiles (`16·hd` f32 words each) and the 16×16 term tile.
    let (sdout, sdoutp) = k.shared_f32(TILE * hd);
    let (sv, svp) = k.shared_f32(TILE * hd);
    let (sred, sredp) = k.shared_f32(TILE * TILE);
    let gid = k.gid_comp(0);
    let tid = k.local_index();
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let c16 = k.cu(TILE);
    let cwg = k.cu(WG);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let ctiles = k.cu(tiles);
    let cstage = k.cu(TILE * hd);
    let zero = k.cf_bits(0x0000_0000);
    let bt = k.bool_t;
    // One workgroup per (h, i-tile, j-tile); h is the outer axis of the exact grid.
    let wg = k.udiv(gid, cwg);
    let tt = k.imul(ctiles, ctiles);
    let h = k.udiv(wg, tt);
    let rem = k.umod(wg, tt);
    let ti = k.udiv(rem, ctiles);
    let tj = k.umod(rem, ctiles);
    let kvh = k.udiv(h, crep);
    let ibase = k.imul(ti, c16);
    let jbase = k.imul(tj, c16);
    let ty = k.udiv(tid, c16);
    let tx = k.umod(tid, c16);
    let i = k.iadd(ibase, ty);
    let j = k.iadd(jbase, tx);
    // Stage both operand tiles: position `s` is `(row, e) = (s / hd, s % hd)`; a row past `t`
    // keeps the zero already stored, so a partial tile stages only its valid rows.
    k.loop_step(tid, cstage, cwg, |k, s| {
        let row = k.udiv(s, chd);
        let e = k.umod(s, chd);
        let gi = k.iadd(ibase, row);
        let i_ok = k.ult(gi, ct);
        k.shared_store(sdout, sdoutp, s, zero);
        k.iff(i_ok, |mut k| {
            let rb = k.imul(gi, cnh);
            let r = k.iadd(rb, h);
            let base = k.imul(r, chd);
            let idx = k.iadd(base, e);
            let val = k.load_f(dout, idx);
            k.shared_store(sdout, sdoutp, s, val);
            k
        });
        let gj = k.iadd(jbase, row);
        let j_ok = k.ult(gj, ct);
        k.shared_store(sv, svp, s, zero);
        k.iff(j_ok, |mut k| {
            let rb = k.imul(gj, ckv);
            let r = k.iadd(rb, kvh);
            let base = k.imul(r, chd);
            let idx = k.iadd(base, e);
            let val = k.load_f(v, idx);
            k.shared_store(sv, svp, s, val);
            k
        });
    });
    k.wg_barrier();
    // Per-key term of this thread: P[h,i,j]·dP_ij for a causal (i,j) pair, 0 otherwise.
    let term = k.fvar_f();
    k.store_var(term, zero);
    let i_ok = k.ult(i, ct);
    let j_ok = k.ult(j, ct);
    let act = k.e(OP_LOGICAL_AND, bt, &[i_ok, j_ok]);
    k.iff(act, |mut k| {
        let cj = k.iadd(i, c1);
        let causal = k.ult(j, cj);
        k.iff(causal, |mut k| {
            let dp = k.fvar_f();
            k.store_var(dp, zero);
            let drow = k.imul(ty, chd);
            let vrow = k.imul(tx, chd);
            k.loop_u(c0, chd, |mut k, e| {
                let da = k.iadd(drow, e);
                let dov = k.shared_load(sdout, sdoutp, da);
                let va = k.iadd(vrow, e);
                let vv = k.shared_load(sv, svp, va);
                let a0 = k.load_var_f(dp);
                let a1 = k.fma(dov, vv, a0);
                k.store_var(dp, a1);
                k
            });
            // P[h,i,j] at (h·t+i)·t+j.
            let ht = k.imul(h, ct);
            let hi = k.iadd(ht, i);
            let pb = k.imul(hi, ct);
            let pidx = k.iadd(pb, j);
            let pij = k.load_f(p, pidx);
            let dpv = k.load_var_f(dp);
            let s = k.fmul(pij, dpv);
            k.store_var(term, s);
            k
        });
        k
    });
    let s = k.load_var_f(term);
    k.shared_store(sred, sredp, tid, s);
    k.wg_barrier();
    // Thread `tid < 16` sums row `tid` of the term tile in ascending `j` (ascending `tx`) and
    // writes this tile's partial; the reduce dispatch sums the partials in ascending `tj`.
    let first = k.ult(tid, c16);
    k.iff(first, |mut k| {
        let gi = k.iadd(ibase, tid);
        let ok = k.ult(gi, ct);
        k.iff(ok, |mut k| {
            let acc = k.fvar_f();
            k.store_var(acc, zero);
            k.loop_u(c0, c16, |mut k, tx2| {
                let rowbase = k.imul(tid, c16);
                let idx = k.iadd(rowbase, tx2);
                let a = k.shared_load(sred, sredp, idx);
                let b = k.load_var_f(acc);
                let sum = k.fadd(b, a);
                k.store_var(acc, sum);
                k
            });
            let ht = k.imul(h, ct);
            let hi = k.iadd(ht, gi);
            let base = k.imul(hi, ctiles);
            let pidx = k.iadd(base, tj);
            let a = k.load_var_f(acc);
            k.store_f(parts, pidx, a);
            k
        });
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_rowsum_module`] as bytes.
pub fn attn_bwd_rowsum_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_rowsum_module(t, nh, nkv, hd))
}

/// **Attention-backward row-sum reduce (f32)** — dispatch 2 of the five: one thread per `(h,i)`
/// sums the `⌈t/16⌉` tile partials of [`attn_bwd_rowsum_module`] in ascending tile order and
/// writes `rowsum[h·t+i]`. With one `j` tile the sum is the single partial, bit-identical to the
/// tiled dispatch's own row sum. Grid `⌈nh·t/256⌉`, LocalSize 256.
///
/// Bindings: 0 = rowsum_parts (f32, `nh·t·⌈t/16⌉`), 1 = rowsum (f32, `t·nh`).
///
/// Complexity: one thread reads `⌈t/16⌉` words and writes one, so the dispatch moves
/// `2·nh·t·⌈t/16⌉` words — `O(t)` reads per thread.
pub fn attn_bwd_rowsum_reduce_module(t: u32, nh: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(
        t > 0 && nh > 0,
        "attn_bwd_rowsum_reduce: shape must be non-zero"
    );
    let tiles = t.div_ceil(16);
    let mut k = Kb::new();
    let parts = k.ssbo_f(0);
    let rowsum = k.ssbo_f(1);
    let gid = k.gid_comp(0);
    let c0 = k.cu(0);
    let ctiles = k.cu(tiles);
    let total = k.cu(t * nh);
    let zero = k.cf_bits(0x0000_0000);
    let inb = k.ult(gid, total);
    k.iff(inb, |mut k| {
        let acc = k.fvar_f();
        k.store_var(acc, zero);
        let base = k.imul(gid, ctiles);
        k.loop_u(c0, ctiles, |mut k, tj| {
            let idx = k.iadd(base, tj);
            let a = k.load_f(parts, idx);
            let b = k.load_var_f(acc);
            let sum = k.fadd(b, a);
            k.store_var(acc, sum);
            k
        });
        let s = k.load_var_f(acc);
        k.store_f(rowsum, gid, s);
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_rowsum_reduce_module`] as bytes.
pub fn attn_bwd_rowsum_reduce_bytes(t: u32, nh: u32) -> Vec<u8> {
    bytes(attn_bwd_rowsum_reduce_module(t, nh))
}

/// **Attention-backward score gradient (f32)** — dispatch 2, the materialisation of
/// `dS[h,i,j] = P[h,i,j]·(dP_ij − rowsum[h·t+i])` as one 16×16 output tile per workgroup.
/// Workgroup `(h, ti, tj)` of `nh·⌈t/16⌉²` owns the tile `i ∈ [16·ti, 16·ti+16)`,
/// `j ∈ [16·tj, 16·tj+16)`; thread `tid = 16·ty + tx` computes `(i,j) = (16·ti+ty, 16·tj+tx)`.
///
/// Staging: every thread strides the `TILE·hd` positions (`TILE = 16`), position `s` standing for
/// `(row, e) = (s/hd, s%hd)`; it writes `sdout[s] = dout[((ibase+row)·nh+h)·hd + e]` and
/// `sv[s] = v[((jbase+row)·nkv+h/rep)·hd + e]`. A row past `t` (a partial last tile) is written
/// `0`: the shared arrays start zeroed per position and only a valid row is loaded. One
/// [`Kb::wg_barrier`] after the staging loop makes both tiles visible.
///
/// Each thread with `i < t` and `j < t` then computes `dP_ij = Σ_e sdout[ty·hd+e]·sv[tx·hd+e]`
/// with `e` ascending and fused multiply-add — the stage holds the same f32 words the global loads
/// held, so the sum is bit-identical — rounds `dS` once and writes it; `j > i` writes `0` without
/// reading `p` or `rowsum`. Grid `nh·⌈t/16⌉²`, LocalSize 256.
///
/// Bindings: 0 = v (f32, `t·nkv·hd`), 1 = p (f32, `nh·t·t`), 2 = dout (f32, `t·nh·hd`),
/// 3 = rowsum (f32, `t·nh`), 4 = ds (f32, `nh·t·t`).
///
/// Complexity: workgroup `(h,ti,tj)` stages `min(16, t−16·ti)·hd` words of `dout` and
/// `min(16, t−16·tj)·hd` of `v` (each valid row once), reads one `p` and one `rowsum` word per
/// causal thread and writes one `ds` word per valid thread. Summed over tiles the operand reads
/// are `⌈t/16⌉·t·hd` each and the causal threads are all `t(t+1)/2` pairs, so the dispatch moves
/// `2·nh·⌈t/16⌉·t·hd + nh·t(t+1) + nh·t²` words (`O(hd)` per thread). The untiled form moved
/// `nh·t²·(2·hd+2) + nh·t²`; at `hd = 32, t = 64, nh = 4` that is 1,097,728 words against
/// 88,560, 12.4× less.
pub fn attn_bwd_ds_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    const TILE: u32 = 16;
    assert!(t > 0 && hd > 0, "attn_bwd_ds: shape must be non-zero");
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_bwd_ds: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let v = k.ssbo_f(0);
    let p = k.ssbo_f(1);
    let dout = k.ssbo_f(2);
    let rowsum = k.ssbo_f(3);
    let ds_buf = k.ssbo_f(4);
    let tiles = t.div_ceil(TILE);
    // Two 16-row operand tiles, `16·hd` f32 words (64·hd bytes) each: workgroup memory only,
    // every staged word read by the 16 threads of the opposite tile axis.
    let (sdout, sdoutp) = k.shared_f32(TILE * hd);
    let (sv, svp) = k.shared_f32(TILE * hd);
    let gid = k.gid_comp(0);
    let tid = k.local_index();
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let c16 = k.cu(TILE);
    let cwg = k.cu(WG);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let ctiles = k.cu(tiles);
    let ctt = k.cu(tiles * tiles);
    let cstage = k.cu(TILE * hd);
    let zero = k.cf_bits(0x0000_0000);
    let bt = k.bool_t;
    // One workgroup per (h, i-tile, j-tile); h is the outer axis of the exact grid.
    let wg = k.udiv(gid, cwg);
    let h = k.udiv(wg, ctt);
    let rem = k.umod(wg, ctt);
    let ti = k.udiv(rem, ctiles);
    let tj = k.umod(rem, ctiles);
    let kvh = k.udiv(h, crep);
    let ibase = k.imul(ti, c16);
    let jbase = k.imul(tj, c16);
    let ty = k.udiv(tid, c16);
    let tx = k.umod(tid, c16);
    let i = k.iadd(ibase, ty);
    let j = k.iadd(jbase, tx);
    // Stage both operand tiles: position `s` is `(row, e) = (s / hd, s % hd)`; a row past `t`
    // keeps the zero already stored, so a partial tile stages only its valid rows.
    k.loop_step(tid, cstage, cwg, |k, s| {
        let row = k.udiv(s, chd);
        let e = k.umod(s, chd);
        let gi = k.iadd(ibase, row);
        let i_ok = k.ult(gi, ct);
        k.shared_store(sdout, sdoutp, s, zero);
        k.iff(i_ok, |mut k| {
            let rb = k.imul(gi, cnh);
            let r = k.iadd(rb, h);
            let base = k.imul(r, chd);
            let idx = k.iadd(base, e);
            let val = k.load_f(dout, idx);
            k.shared_store(sdout, sdoutp, s, val);
            k
        });
        let gj = k.iadd(jbase, row);
        let j_ok = k.ult(gj, ct);
        k.shared_store(sv, svp, s, zero);
        k.iff(j_ok, |mut k| {
            let rb = k.imul(gj, ckv);
            let r = k.iadd(rb, kvh);
            let base = k.imul(r, chd);
            let idx = k.iadd(base, e);
            let val = k.load_f(v, idx);
            k.shared_store(sv, svp, s, val);
            k
        });
    });
    k.wg_barrier();
    // (h·t+i)·t+j: `P` and the output share this index; `rowsum` is `h·t+i`.
    let ht = k.imul(h, ct);
    let hi = k.iadd(ht, i);
    let pb = k.imul(hi, ct);
    let didx = k.iadd(pb, j);
    let out = k.fvar_f();
    k.store_var(out, zero);
    let i_ok = k.ult(i, ct);
    let j_ok = k.ult(j, ct);
    let act = k.e(OP_LOGICAL_AND, bt, &[i_ok, j_ok]);
    k.iff(act, |mut k| {
        // dP ascending e over the head width from the staged rows; one rounding for dS.
        let cj = k.iadd(i, c1);
        let causal = k.ult(j, cj);
        k.iff(causal, |mut k| {
            let dp = k.fvar_f();
            k.store_var(dp, zero);
            let drow = k.imul(ty, chd);
            let vrow = k.imul(tx, chd);
            k.loop_u(c0, chd, |mut k, e| {
                let da = k.iadd(drow, e);
                let dov = k.shared_load(sdout, sdoutp, da);
                let va = k.iadd(vrow, e);
                let vv = k.shared_load(sv, svp, va);
                let a0 = k.load_var_f(dp);
                let a1 = k.fma(dov, vv, a0);
                k.store_var(dp, a1);
                k
            });
            let pij = k.load_f(p, didx);
            let rs = k.load_f(rowsum, hi);
            let dpv = k.load_var_f(dp);
            let diff = k.fsub(dpv, rs);
            let s = k.fmul(pij, diff);
            k.store_var(out, s);
            k
        });
        let s = k.load_var_f(out);
        k.store_f(ds_buf, didx, s);
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_ds_module`] as bytes.
pub fn attn_bwd_ds_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_ds_module(t, nh, nkv, hd))
}

/// **Attention-backward dQ (f32)** — dispatch 3, the twin of `attn_bwd_kern`'s `dQ = scale·dS·K`.
/// Thread `(i·nh+h)·hd+d` (of `t·nh·hd`) walks `j` in `0..=i` ascending and accumulates
/// `ds[h,i,j]·K[j,h/rep,d]` with fused multiply-add; it writes `scale·acc` with
/// `scale = 1/sqrt(hd)` held as an f32 module constant. The `dP_ij` recomputation is gone: the
/// value is read from dispatch 2, with the same one-rounding `dS`. Grid `⌈t·nh·hd/256⌉`,
/// LocalSize 256.
///
/// Bindings: 0 = k (f32, `t·nkv·hd`), 1 = ds (f32, `nh·t·t`), 2 = dq (f32, `t·nh·hd`).
///
/// Complexity: one thread reads `i+1` `ds` words and `i+1` `k` words and writes one `dq` word,
/// so the dispatch moves `t·nh·hd·(t + 2)` words — `O(t)` per thread — against the pre-change
/// `2·t²·nh·hd² + t·nh·hd` words of the in-thread `dP` form and the reused-read floor
/// `t·nkv·hd + nh·t² + t·nh·hd`.
pub fn attn_bwd_dq_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(t > 0 && hd > 0, "attn_bwd_dq: shape must be non-zero");
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_bwd_dq: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let kb = k.ssbo_f(0);
    let ds_buf = k.ssbo_f(1);
    let dq = k.ssbo_f(2);
    let gid = k.gid_comp(0);
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let total = k.cu(t * nh * hd);
    let zero = k.cf_bits(0x0000_0000);
    let scale = k.cf(1.0 / (hd as f32).sqrt());
    let inb = k.ult(gid, total);
    k.iff(inb, |mut k| {
        let d = k.umod(gid, chd);
        let r = k.udiv(gid, chd);
        let h = k.umod(r, cnh);
        let i = k.udiv(r, cnh);
        let kvh = k.udiv(h, crep);
        let ht = k.imul(h, ct);
        let rsi = k.iadd(ht, i);
        let pb = k.imul(rsi, ct);
        let acc = k.fvar_f();
        k.store_var(acc, zero);
        let nrow = k.iadd(i, c1);
        k.loop_u(c0, nrow, |mut k, j| {
            let didx = k.iadd(pb, j);
            let dsij = k.load_f(ds_buf, didx);
            let jn = k.imul(j, ckv);
            let jk = k.iadd(jn, kvh);
            let kbse = k.imul(jk, chd);
            let ka = k.iadd(kbse, d);
            let kk = k.load_f(kb, ka);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(dsij, kk, a0);
            k.store_var(acc, a1);
            k
        });
        let accv = k.load_var_f(acc);
        let s = k.fmul(scale, accv);
        k.store_f(dq, gid, s);
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_dq_module`] as bytes.
pub fn attn_bwd_dq_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_dq_module(t, nh, nkv, hd))
}

/// **Attention-backward dK/dV (f32)** — dispatch 4, the twin of `attn_bwd_kern`'s `dK = scale·dSᵀ·Q`
/// and `dV = Pᵀ·dO`. One workgroup per `(j,g,d-tile)` of `t·nkv·⌈hd/32⌉`: thread `(g_lane, lane)`
/// owns one output column `d = dtile·32 + lane` (last tile partial, `d < hd`) and one of eight
/// reduction groups `g_lane = tid / 32`. Group `g_lane` walks the `rep` query heads
/// `h ∈ [g·rep, (g+1)·rep)` ascending and, inside each head, the rows
/// `i = j+g_lane, j+g_lane+8, … < t` ascending; it reads `ds[h,i,j]`, accumulates `ds·Q[i,h,d]`
/// and `P[h,i,j]·dO[i,h,d]` in two registers. The eight per-`d` partials go to two `8·32`-word
/// shared arrays (dk and dv, indexed `g_lane·32 + lane`), and after one barrier the lanes with
/// `d < hd` sum them in ascending group order, multiply dk's sum by `scale` once and write
/// `dk`/`dv` at `(j·nkv+g)·hd+d`. Grid `t·nkv·⌈hd/32⌉`, LocalSize 256.
///
/// The product sequence per output is the same products in the same `h`-then-`i` order, cut into
/// eight ascending sub-sequences and recombined by seven adds; the parity pins are bounds, and the
/// measured maxima stay inside them.
///
/// Bindings: 0 = q (f32, `t·nh·hd`), 1 = p (f32, `nh·t·t`), 2 = ds (f32, `nh·t·t`),
/// 3 = dout (f32, `t·nh·hd`), 4 = dk (f32, `t·nkv·hd`), 5 = dv (f32, `t·nkv·hd`).
///
/// Complexity: a workgroup reads `32·4·rep·(t−j)` global words in a full tile (`4·rem·rep·(t−j)`
/// with `rem = hd − dtile·32` active lanes), so the `⌈hd/32⌉` tiles of one `(j,g)` read the same
/// `4·hd·rep·(t−j)` and the dispatch reads `2·hd·nh·t·(t+1)` words and writes `2·t·nkv·hd`;
/// `O(rep·(t−j))` per lane, unchanged from the one-thread-per-output form, which read the same
/// words. The grid drops from `⌈t·nkv·hd/256⌉` = 16 workgroups at (64,2,32) to
/// `t·nkv·⌈hd/32⌉` = 128 (1 tile) and runs `t·nkv·2` = 256 at `hd = 64`; the serial walk per lane
/// goes from `rep·(t−j)` to `rep·⌈(t−j)/8⌉`; the reduce adds one barrier plus `7·32` shared adds
/// per full tile.
pub fn attn_bwd_dkdv_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    const GROUPS: u32 = 8;
    const DLANES: u32 = 32;
    assert!(t > 0 && hd > 0, "attn_bwd_dkdv: shape must be non-zero");
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_bwd_dkdv: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let q = k.ssbo_f(0);
    let p = k.ssbo_f(1);
    let ds_buf = k.ssbo_f(2);
    let dout = k.ssbo_f(3);
    let dk = k.ssbo_f(4);
    let dv = k.ssbo_f(5);
    // One 32-column d-tile per workgroup; the partial arrays are per tile, not per head width.
    let tiles = hd.div_ceil(DLANES);
    let (red_dk, dkp) = k.shared_f32(GROUPS * DLANES);
    let (red_dv, dvp) = k.shared_f32(GROUPS * DLANES);
    let gid = k.gid_comp(0);
    let tid = k.local_index();
    let c0 = k.cu(0);
    let c8 = k.cu(GROUPS);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let clanes = k.cu(DLANES);
    let cwg = k.cu(WG);
    let ctiles = k.cu(tiles);
    let zero = k.cf_bits(0x0000_0000);
    let scale = k.cf(1.0 / (hd as f32).sqrt());
    // One workgroup per (j,g,d-tile): the grid is exactly `t·nkv·⌈hd/32⌉`, so every workgroup
    // writes, and the last d-tile carries `hd mod 32` columns. A one-tile shape emits no tile
    // division at all, so `hd ≤ 32` keeps the grid and the lane map `d = lane` of the installed
    // single-tile body.
    let wg = k.udiv(gid, cwg);
    // Lane layout: `d` is the output column, `grp` the reduction group that owns a slice of `i`.
    let lane = k.umod(tid, clanes);
    let grp = k.udiv(tid, clanes);
    let (row_g, d) = if tiles == 1 {
        (wg, lane)
    } else {
        let dtile = k.umod(wg, ctiles);
        let row_g = k.udiv(wg, ctiles);
        let dstart = k.imul(dtile, clanes);
        (row_g, k.iadd(dstart, lane))
    };
    let kvh = k.umod(row_g, ckv);
    let j = k.udiv(row_g, ckv);
    let dk_acc = k.fvar_f();
    let dv_acc = k.fvar_f();
    k.store_var(dk_acc, zero);
    k.store_var(dv_acc, zero);
    let act = k.ult(d, chd);
    k.iff(act, |mut k| {
        let h0 = k.imul(kvh, crep);
        let h1 = k.iadd(h0, crep);
        k.loop_u(h0, h1, |mut k, h| {
            let ht = k.imul(h, ct);
            let istart = k.iadd(j, grp);
            k.loop_step(istart, ct, c8, |k, i| {
                // q/dout row of query i, head h: (i·nh+h)·hd; ds/P at (h·t+i)·t+j.
                let ih = k.imul(i, cnh);
                let rib = k.iadd(ih, h);
                let qb = k.imul(rib, chd);
                let qa = k.iadd(qb, d);
                let hi = k.iadd(ht, i);
                let pbase = k.imul(hi, ct);
                let pj = k.iadd(pbase, j);
                let dsij = k.load_f(ds_buf, pj);
                let qd = k.load_f(q, qa);
                let a0 = k.load_var_f(dk_acc);
                let a1 = k.fma(dsij, qd, a0);
                k.store_var(dk_acc, a1);
                let pij = k.load_f(p, pj);
                let dod = k.load_f(dout, qa);
                let b0 = k.load_var_f(dv_acc);
                let b1 = k.fma(pij, dod, b0);
                k.store_var(dv_acc, b1);
            });
            k
        });
        // Partial of this group at column `lane`: red[grp·32 + lane].
        let goff = k.imul(grp, clanes);
        let pidx = k.iadd(goff, lane);
        let dkv = k.load_var_f(dk_acc);
        k.shared_store(red_dk, dkp, pidx, dkv);
        let dvv = k.load_var_f(dv_acc);
        k.shared_store(red_dv, dvp, pidx, dvv);
        k
    });
    k.wg_barrier();
    // Group 0's 32 threads reduce and write (as before); a partial last tile writes `hd mod 32`
    // columns, so the column guard is separate from the lane-group guard.
    let red = k.ult(tid, clanes);
    k.iff(red, |mut k| {
        let col = k.ult(d, chd);
        k.iff(col, |mut k| {
            let dk_sum = k.fvar_f();
            let dv_sum = k.fvar_f();
            k.store_var(dk_sum, zero);
            k.store_var(dv_sum, zero);
            k.loop_u(c0, c8, |mut k, g2| {
                let goff = k.imul(g2, clanes);
                let gi = k.iadd(goff, lane);
                let a = k.shared_load(red_dk, dkp, gi);
                let a0 = k.load_var_f(dk_sum);
                let a1 = k.fadd(a0, a);
                k.store_var(dk_sum, a1);
                let b = k.shared_load(red_dv, dvp, gi);
                let b0 = k.load_var_f(dv_sum);
                let b1 = k.fadd(b0, b);
                k.store_var(dv_sum, b1);
                k
            });
            // One rounding after the head sum, as before; the output word is (j·nkv+g)·hd+d.
            let ak = k.load_var_f(dk_sum);
            let wgo = k.imul(row_g, chd);
            let oi = k.iadd(wgo, d);
            let akv = k.fmul(scale, ak);
            k.store_f(dk, oi, akv);
            let bv = k.load_var_f(dv_sum);
            k.store_f(dv, oi, bv);
            k
        });
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_dkdv_module`] as bytes.
pub fn attn_bwd_dkdv_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_dkdv_module(t, nh, nkv, hd))
}

/// Host wrapper: build and record the tiled backward row sum. `v` is `t·nkv·hd` f32, `p`
/// `nh·t·t`, `dout` `t·nh·hd`, `rowsum_parts` `nh·t·⌈t/16⌉`. Grid `(nh·⌈t/16⌉², 1, 1)` — one
/// 256-thread workgroup per 16×16 `(i,j)` tile; LocalSize 256. The partials are reduced by
/// [`attn_bwd_rowsum_reduce_dispatch`].
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn attn_bwd_rowsum_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    v: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    dout: crate::gpu::vulkan::Handle,
    rowsum_parts: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_rowsum_bytes(t, nh, nkv, hd),
        &[v, p, dout, rowsum_parts],
        nh * t.div_ceil(16) * t.div_ceil(16),
        1,
        "attn_bwd_rowsum_tiled",
    )
}

/// Host wrapper: reduce the tiled row sum's partials. `rowsum_parts` is `nh·t·⌈t/16⌉` f32,
/// `rowsum` `t·nh`. Grid `(⌈nh·t/256⌉, 1, 1)` — one thread per `(h,i)`; LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn attn_bwd_rowsum_reduce_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    rowsum_parts: crate::gpu::vulkan::Handle,
    rowsum: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_rowsum_reduce_bytes(t, nh),
        &[rowsum_parts, rowsum],
        (t * nh).div_ceil(256),
        1,
        "attn_bwd_rowsum_reduce",
    )
}

/// Host wrapper: build and record the backward score-gradient pass. `ds` holds `nh·t·t` f32.
/// Grid `(nh·⌈t/16⌉², 1, 1)` — one 256-thread workgroup per 16×16 output tile.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_bwd_ds_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    v: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    dout: crate::gpu::vulkan::Handle,
    rowsum: crate::gpu::vulkan::Handle,
    ds: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_ds_bytes(t, nh, nkv, hd),
        &[v, p, dout, rowsum, ds],
        nh * t.div_ceil(16) * t.div_ceil(16),
        1,
        "attn_bwd_ds",
    )
}

/// Host wrapper: build and record the backward dQ pass. `ds` holds `nh·t·t` f32.
/// Grid `(⌈t·nh·hd/256⌉, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn attn_bwd_dq_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    k: crate::gpu::vulkan::Handle,
    ds: crate::gpu::vulkan::Handle,
    dq: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_dq_bytes(t, nh, nkv, hd),
        &[k, ds, dq],
        (t * nh * hd).div_ceil(256),
        1,
        "attn_bwd_dq",
    )
}

/// Host wrapper: build and record the backward dK/dV pass. `ds` holds `nh·t·t` f32.
/// Grid `(t·nkv·⌈hd/32⌉, 1, 1)` — one workgroup of 256 threads per `(j,g,d-tile)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_bwd_dkdv_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    q: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    ds: crate::gpu::vulkan::Handle,
    dout: crate::gpu::vulkan::Handle,
    dk: crate::gpu::vulkan::Handle,
    dv: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_dkdv_bytes(t, nh, nkv, hd),
        &[q, p, ds, dout, dk, dv],
        t * nkv * hd.div_ceil(32),
        1,
        "attn_bwd_dkdv",
    )
}
