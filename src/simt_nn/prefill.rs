//! simt_nn::prefill — the prefill attention kernels: the flat `attn_prefill` pass and the tiled
//! form with its tile-shape constants. Cut verbatim from `simt_nn.rs`.

use super::*;

/// **Batched causal attention over the chunk — one workgroup per (head, position).**
///
/// Reproduces `attn_head_into`'s reduction ORDER rather than re-deriving the read: the score is an
/// f32 query against the f32-valued int8 key codes accumulated dim-ascending (the CPU's `sc32[s] +=
/// qv·kb` inner loop), the softmax takes a global max first and a global sum second (not an online
/// rescale, which would regroup the exponentials), and the value fold walks keys ASCENDING with the
/// per-key weight `p = (e/Σe)·vs[s]` applied before the multiply — term for term what the CPU does.
/// Three passes over K is the price of that order; K is a few hundred KB per layer and stays in L2.
///
/// **Dense only.** The top-A / mass cuts (`topk`, `READ_DROP`) and the union/wcap probes are host
/// state this kernel does not read; the driver REFUSES the offload when any of them is armed rather
/// than silently running a different read.
///
/// Bindings: 0 = qkv (f32, `qkvm` per position, roped), 1 = kc, 2 = ks, 3 = vc, 4 = vs (the caches
/// [`qkv_rope_batch_module`] wrote), 5 = att (f32, `nh·hd` per position), 6 = params
/// (`params[0]` = `pos0`). Dispatch `(nh, B)` at LocalSize `hd`.
/// The largest KV window [`attn_prefill_module`] will hold a shared-memory score cache for. 8192
/// f32 = 32 KB, which still leaves room under a 48 KB workgroup allocation; past it the kernel
/// recomputes.
pub const SCORE_CACHE_MAX: u32 = 8192;

pub fn attn_prefill_module(
    nh: u32,
    nkv: u32,
    hd: u32,
    qkvm: u32,
    tmax: u32,
    scale: f32,
) -> Vec<u32> {
    assert!(
        hd.is_power_of_two(),
        "the tree reduce needs a power-of-two workgroup"
    );
    let group = nh / nkv.max(1);
    let qn = nh * hd;
    let mut k = Kb::new();
    let qkv = k.ssbo_f(0);
    let kc = k.ssbo_f(1);
    let ks = k.ssbo_f(2);
    let vc = k.ssbo_f(3);
    let vs = k.ssbo_f(4);
    let att = k.ssbo_f(5);
    let params = k.ssbo_u(6);
    let (qs, qsp) = k.shared_f32(hd); // this head's query, read hd× per key
    let (red, redp) = k.shared_f32(hd);
    let (psh, pshp) = k.shared_f32(hd); // one tile of softmax weights
                                        // The SCORE CACHE. The three passes below need every key's score three times, and the original
                                        // kernel paid for it three times: `score` is an `hd`-long dependent FADD chain plus an `hd`-long
                                        // walk of K, so the workgroup did 4·len·hd MACs where the read itself only needs 2·len·hd
                                        // (one score pass, one value fold). Caching pass 1's scores in shared memory removes passes 2
                                        // and 3's recompute WITHOUT touching the reduction order: the cached f32 is bit-for-bit the
                                        // value the recompute produced, the global max is still taken first and the partition function
                                        // second, and the value fold still walks keys ascending. It is the same arithmetic, read once.
                                        //
                                        // `tmax` floats is the whole KV window (16 KB at tmax = 4096). Past `SCORE_CACHE_MAX` the
                                        // allocation would cost more occupancy than the recompute costs time, so the kernel falls back
                                        // to recomputing — the cache is an optimization with an explicit domain, not an assumption.
    let cached = tmax <= SCORE_CACHE_MAX;
    let (sc_sh, sc_shp) = k.shared_f32(if cached { tmax } else { 1 });
    let chd = k.cu(hd);
    let gid = k.gid_comp(0);
    let i = k.gid_comp(1);
    let h = k.udiv(gid, chd); // WG-uniform: LocalSize == hd
    let lane = k.umod(gid, chd);
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let pos0 = k.load_u(params, c0);
    let pos = k.iadd(pos0, i);
    let len = k.iadd(pos, c1);
    let cg = k.cu(group);
    let kvh = k.udiv(h, cg);
    let cnkv = k.cu(nkv);
    let ct = k.cu(tmax);
    let cqkvm = k.cu(qkvm);
    let qrow = k.imul(i, cqkvm);
    let hb = k.imul(h, chd);
    let qbase = k.iadd(qrow, hb);
    let kdim0 = k.imul(kvh, chd); // K row base: (kvh·hd + d)·tmax
    let vrow0 = k.imul(kvh, ct); // V row base: (kvh·tmax + s)·hd
                                 // the query into shared — every key rereads all hd of it
    let qi = k.iadd(qbase, lane);
    let qv = k.load_f(qkv, qi);
    k.shared_store(qs, qsp, lane, qv);
    k.wg_barrier();
    // the score of key `s`: Σ_d q[d]·kc[(kvh·hd+d)·tmax + s], dim-ascending, ×ks[s]·scale
    let score = move |k: &mut Kb, s: u32| -> u32 {
        let acc = k.fvar_f();
        let z = k.cf(0.0);
        k.store_var(acc, z);
        let chdb = k.cu(hd);
        let c0b = k.cu(0);
        k.loop_u(c0b, chdb, |k, d| {
            let qd = k.shared_load(qs, qsp, d);
            let kr0 = k.iadd(kdim0, d);
            let kr1 = k.imul(kr0, ct);
            let kidx = k.iadd(kr1, s);
            let kv = k.load_f(kc, kidx);
            let pr = k.fmul(qd, kv);
            let a0 = k.load_var_f(acc);
            let a1 = k.fadd(a0, pr);
            k.store_var(acc, a1);
        });
        let raw = k.load_var_f(acc);
        let srow = k.imul(s, cnkv);
        let sidx = k.iadd(srow, kvh);
        let skv = k.load_f(ks, sidx);
        let csc = k.cf(scale);
        let t0 = k.fmul(raw, skv);
        k.fmul(t0, csc)
    };
    // ── pass 1: the global max
    let lmax = k.fvar_f();
    let neg = k.cf(-1.0e30);
    k.store_var(lmax, neg);
    k.loop_step(lane, len, chd, |k, s| {
        let sc = score(k, s);
        if cached {
            k.shared_store(sc_sh, sc_shp, s, sc);
        }
        let m0 = k.load_var_f(lmax);
        let m1 = k.ext2(GLSL_FMAX, m0, sc);
        k.store_var(lmax, m1);
    });
    let lm = k.load_var_f(lmax);
    k.shared_store(red, redp, lane, lm);
    k.wg_barrier();
    k.tree_reduce(red, redp, hd, lane, true);
    let m = k.shared_load(red, redp, c0);
    k.wg_barrier();
    // ── pass 2: the partition function
    let lsum = k.fvar_f();
    let z2 = k.cf(0.0);
    k.store_var(lsum, z2);
    k.loop_step(lane, len, chd, |k, s| {
        let sc = if cached {
            k.shared_load(sc_sh, sc_shp, s)
        } else {
            score(k, s)
        };
        let dd = k.fsub(sc, m);
        let e = k.ext1(GLSL_EXP, dd);
        let s0 = k.load_var_f(lsum);
        let s1 = k.fadd(s0, e);
        k.store_var(lsum, s1);
    });
    let lsv = k.load_var_f(lsum);
    k.shared_store(red, redp, lane, lsv);
    k.wg_barrier();
    k.tree_reduce(red, redp, hd, lane, false);
    let tot = k.shared_load(red, redp, c0);
    k.wg_barrier();
    // ── pass 3: the value fold, tiles of `hd` keys, lane = output dim, keys ASCENDING
    let oacc = k.fvar_f();
    let z3 = k.cf(0.0);
    k.store_var(oacc, z3);
    k.loop_step(c0, len, chd, |k, t0| {
        let s = k.iadd(t0, lane);
        let inr = k.ult(s, len);
        let zf = k.cf(0.0);
        // an over-hanging lane scores the CLAMPED key and is zeroed — branchless, and `pos` is
        // always a valid slot (the position this workgroup is computing).
        let f = k.float;
        let u = k.uint;
        let ssafe = k.e(OP_SELECT, u, &[inr, s, pos]);
        // `ssafe` is either an in-range key or `pos` (this workgroup's own position, always < len),
        // so the cached slot was written by pass 1 either way — the clamp keeps the cache read
        // in-region for the same reason it kept the K read in-region.
        let sc = if cached {
            k.shared_load(sc_sh, sc_shp, ssafe)
        } else {
            score(k, ssafe)
        };
        let dd = k.fsub(sc, m);
        let e0 = k.ext1(GLSL_EXP, dd);
        let en = k.fdiv(e0, tot);
        let srow = k.imul(ssafe, cnkv);
        let sidx = k.iadd(srow, kvh);
        let vsv = k.load_f(vs, sidx);
        let p0 = k.fmul(en, vsv);
        let p = k.e(OP_SELECT, f, &[inr, p0, zf]);
        k.wg_barrier();
        k.shared_store(psh, pshp, lane, p);
        k.wg_barrier();
        let chdb = k.cu(hd);
        let c0b = k.cu(0);
        k.loop_u(c0b, chdb, |k, j| {
            let pj = k.shared_load(psh, pshp, j);
            let sj0 = k.iadd(t0, j);
            let jin = k.ult(sj0, len);
            // an over-hanging tail slot addresses `pos` (always written) and is zeroed: its V is
            // UNINITIALIZED device memory, and `0 · Inf = NaN` would poison the whole output —
            // `flash_partial_module` records the same trap from the decode side.
            let sj = k.e(OP_SELECT, u, &[jin, sj0, pos]);
            let vb0 = k.iadd(vrow0, sj);
            let vb1 = k.imul(vb0, chd);
            let vidx = k.iadd(vb1, lane);
            let vv0 = k.load_f(vc, vidx);
            let vv = k.e(OP_SELECT, f, &[jin, vv0, zf]);
            let pr = k.fmul(pj, vv);
            let a0 = k.load_var_f(oacc);
            let a1 = k.fadd(a0, pr);
            k.store_var(oacc, a1);
        });
    });
    let of = k.load_var_f(oacc);
    let cqn = k.cu(qn);
    let arow = k.imul(i, cqn);
    let ab = k.iadd(arow, hb);
    let aidx = k.iadd(ab, lane);
    k.store_f(att, aidx, of);
    k.finish(hd)
}

/// The default query-tile of [`attn_prefill_tiled_module`]: `rows` query POSITIONS per workgroup,
/// `rts` rows × `ct` keys held per thread in registers during the score GEMM.
///
/// `rows` must divide [`crate::simt_nn::ATTN_TILE_ROWS`]'s consumer tile (`PREFILL_BT_MAX` = 32 in
/// servefold) so the `ceil(n/rows)·rows` rows a tiled dispatch touches stay inside the padded
/// activation scratch the caller zeroed.
pub const ATTN_TILE_ROWS: u32 = 32;
/// Rows per thread in the score GEMM's register tile. `rts·ct` = 8 accumulators is the measured
/// knee: 4 (`rts` 1) costs 4 948 µs against 8's 3 291, and 16 (`rts` 4 `ct` 4, or `rts` 2 `ct` 8)
/// costs 3 464 / 4 757 — the register tile stops paying before it stops fitting.
pub const ATTN_TILE_RTS: u32 = 4;
/// Keys per thread in the score GEMM's register tile.
pub const ATTN_TILE_CT: u32 = 2;
/// Whether the default tile FUSES the partition function with the value fold — two walks of K
/// instead of three, at the cost of factoring `1/Σe` out of the fold (see
/// [`attn_prefill_tiled_module`]'s `fuse`). OFF: the default read is bit-identical to
/// [`attn_prefill_module`], and the dial is what prices the alternative.
pub const ATTN_TILE_FUSE: bool = false;
/// The hd ≥ 128 query tile. At hd=128 the K slab `hd·hd·4` = 64 KiB alone exceeds the planner's
/// 48 KiB workgroup budget, so the slab is walked in [`ATTN_TILE_KP_HD128`] dim chunks and the
/// query tile drops to 16 positions. Only the PEAK footprint changes: every `(dim, key)` pair is
/// staged exactly once across the chunks, so the FMAs per staged byte (`rows/4`, the score GEMM's
/// `rows·kt·hd` MACs over `hd·kt·4` staged bytes) and the denominator's width-`hd` reduction tree
/// are the hd=64 tile's. `kt` stays `hd` because the tree needs slot `s &&& (hd−1)`.
pub const ATTN_TILE_ROWS_HD128: u32 = 16;
/// Key-dim chunks the hd ≥ 128 K slab is walked in; `dchunk = hd/kp = 32` at hd=128, so the peak
/// K-slab footprint is `32·128·4` = 16 KiB.
pub const ATTN_TILE_KP_HD128: u32 = 4;
/// The hd ≥ 128 tile's keys per thread in the score GEMM; `hd/ct == 32` keeps the online slab max
/// a single 32-lane `subgroupMax`.
pub const ATTN_TILE_CT_HD128: u32 = 4;

/// **Query-tiled causal attention — the same read, with K read once for `rows` queries.**
///
/// [`attn_prefill_module`] dispatches one workgroup per (head, POSITION), so every workgroup walks
/// the whole causal K prefix on its own: at `nh` = 14, `nkv` = 2, B = 640 a single KV head's K is
/// pulled out of L2 **4 480 times** (7 query heads × 640 positions). The score cache a83508e added
/// removed the *recompute* of that walk, not the *traffic*, and the traffic is what the stage is
/// bound by: `nh·B·L̄·hd·4` = 734 MB per layer, 17.6 GB per chunk, ≈ 8.8 ms at L2 speed against a
/// measured 6.4 ms. That arithmetic is the whole thesis of this kernel.
///
/// So the workgroup here owns a TILE of `rows` consecutive query positions of one head, stages each
/// 64-key slab of K in shared memory ONCE, and lets all `rows` queries read it from there. K global
/// traffic per KV head drops from `group·B` passes to `3·group·B/rows` (three, because the exact
/// reduction below needs three walks) — 4 480 → 420 at `rows` = 32.
///
/// **THE ORDER IS THE ORDER `attn_head_into` AND [`attn_prefill_module`] USE — this is a traffic
/// restructuring, not a reformulation.** `AttentionSpinor.softmax_attention_is_online_spinor` proves
/// the online/flash form exact in ℝ and would let one walk of K do the whole read, but the online
/// rescale REASSOCIATES in f32, so it is not taken here. Instead:
///   * the score is `Σ_d q[d]·kc[(kvh·hd+d)·tmax+s]` accumulated **d-ascending** in one f32
///     accumulator — the register tile changes which thread owns the accumulator, never the chain;
///   * the max is taken over the whole prefix FIRST (max is exactly associative, so its tree is
///     free), the partition function over the whole prefix SECOND (no rescale);
///   * the 64 partial sums are the SAME 64 partials the untiled kernel forms — partial `p` is
///     `Σ_{s ≡ p (mod 64)} e_s` ascending — and they meet in the same log₂(64) pairing tree, so the
///     denominator is bit-for-bit the untiled one;
///   * the value fold keeps one accumulator per (row, output dim) and walks keys ASCENDING, with the
///     per-key weight `p = (e/Σe)·vs[s]` formed before the multiply.
/// A key past a row's causal length contributes `fmax(·,−1e30)`, `+0.0` and `0.0·0.0` respectively —
/// all three are exact identities on the accumulator, so masking a wider tile costs no bits.
///
/// **Every index is closed the mandala way** (`MandalaMemory.access` = `base + (key &&& (2^k−1))`,
/// `masked_load_never_oob`): a key `s` is folded into `[0, tmax)` by one `and` when `tmax` is a power
/// of two and by `UMin` when it is not, before it is scaled into K, V, `ks` or `vs`; the query row is
/// folded to `n−1`; the output store is the one site left guarded by a branch, because `att` has no
/// power-of-two extent and a wrong row there is a wrong ANSWER, not just a wrong read.
///
/// Bindings and semantics are [`attn_prefill_module`]'s. Dispatch `(nh, ceil(B/rows))` at
/// LocalSize `(hd/ct)·(rows/rts)`.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_tiled_module(
    nh: u32,
    nkv: u32,
    hd: u32,
    qkvm: u32,
    tmax: u32,
    scale: f32,
    rows: u32,
    rts: u32,
    ct: u32,
    fuse: bool,
) -> Vec<u32> {
    attn_prefill_tiled_module_m(nh, nkv, hd, qkvm, tmax, scale, rows, rts, ct, 1, fuse, 0)
}

/// Bit 0 of the `mode` of [`attn_prefill_tiled_module_m`]: launch the heaviest query tile first.
pub const ATTN_MODE_REVERSE: u32 = 1;
/// Bit 1 of the `mode`: ONE walk of K — online softmax (running per-row max `M`, rescale
/// `α = exp(M_old − M_new)` of the denominator partials and of the value accumulator once per
/// slab). Needs `hd / ct` = 32 key lanes = one subgroup per row slot (the slab max is a
/// `subgroupMax`). The result equals the three-walk form in ℝ; in f32 it reassociates.
pub const ATTN_MODE_ONLINE: u32 = 2;
/// The default `mode`: heaviest tile first, one walk of K. The online bit additionally needs a
/// 32-lane subgroup (see [`ATTN_MODE_ONLINE`]); the prefill planner clears it where that is not
/// established.
pub const ATTN_TILE_MODE: u32 = ATTN_MODE_REVERSE | ATTN_MODE_ONLINE;

/// `subgroupMax(v)` — OpGroupNonUniformFMax over the whole subgroup, every lane receives it.
fn sg_fmax(k: &mut Kb, v: u32) -> u32 {
    const OP_GROUP_NON_UNIFORM_FMAX: u32 = 358;
    k.sg_arith = true;
    let f = k.float;
    let sc = k.cu(3);
    k.e(OP_GROUP_NON_UNIFORM_FMAX, f, &[sc, 0, v])
}

/// [`attn_prefill_tiled_module`] with a `mode` bitmask of the scheduling and arithmetic variants
/// priced in the gpu-prefill note/`: [`ATTN_MODE_REVERSE`].
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_tiled_module_m(
    nh: u32,
    nkv: u32,
    hd: u32,
    qkvm: u32,
    tmax: u32,
    scale: f32,
    rows: u32,
    rts: u32,
    ct: u32,
    kp: u32,
    fuse: bool,
    mode: u32,
) -> Vec<u32> {
    assert!(
        hd.is_power_of_two(),
        "the reduction tree needs a power-of-two head width"
    );
    assert!(ct > 0 && hd % ct == 0, "ct must divide hd");
    assert!(rts > 0 && rows % rts == 0, "rts must divide rows");
    assert!(kp > 0 && hd % kp == 0, "kp must divide hd");
    let dchunk = hd / kp; // dims staged in one chunk of the K slab (kp == 1 is the whole head)
    let kt = hd; // key tile == hd, so lane p of the tile owns key t0+p and the stride-hd partials hold
    let kl = hd / ct; // key lanes in the score mapping
    let ss = rows / rts; // row slots in the score mapping
    let tsz = kl * ss; // workgroup size
    assert!(
        tsz % hd == 0,
        "the value-fold mapping needs LocalSize divisible by hd"
    );
    let fslots = tsz / hd;
    assert!(
        rows % fslots == 0,
        "the value fold needs rows divisible by LocalSize/hd"
    );
    let rtf = rows / fslots; // rows per thread in the value fold
    assert!(
        tsz <= 1024,
        "workgroup {tsz} over the Vulkan invocation limit"
    );
    let online = mode & ATTN_MODE_ONLINE != 0;
    assert!(
        !online || kl == 32,
        "the online form reduces the slab max over one 32-lane subgroup per row slot"
    );
    // the online form normalizes in the epilogue exactly as the fused two-walk form does
    let fuse = fuse || online;
    let group = nh / nkv.max(1);
    let qn = nh * hd;

    let mut k = Kb::new();
    let qkvb = k.ssbo_f(0);
    let kcb = k.ssbo_f(1);
    let ksb = k.ssbo_f(2);
    let vcb = k.ssbo_f(3);
    let vsb = k.ssbo_f(4);
    let attb = k.ssbo_f(5);
    let parb = k.ssbo_u(6);
    // Q for the whole tile (read hd× per key), one K slab, and the per-row score/weight row —
    // which doubles as the 64-partial reduction array, because it is exactly `rows × hd` wide and is
    // dead at every point a reduction runs.
    // MEASURED NEGATIVE, recorded because it was the obvious next move and it does NOT work on this
    // driver: staging the K slab as f32x4 — one `LDS.128` per four keys instead of four `LDS.32`,
    // the score GEMM's only non-broadcast shared read — regressed attn from 3 427 to 3 650 µs at
    // rows 32 / rts 2 / ct 4 and from 4 591 to 5 787 at 8/1/4, with 32/4/4 the one shape that moved
    // the right way (3 464 → 3 339). Composing the vector costs four scalar global loads plus an
    // `OpCompositeConstruct` in the stage, and the driver evidently already coalesces the four
    // consecutive scalar shared reads it was meant to replace. The slab stays scalar.
    let (qs, qsp) = k.shared_f32(rows * hd);
    let (kst, kstp) = k.shared_f32(dchunk * kt);
    let (sst, sstp) = k.shared_f32(rows * kt);
    // per-row slab rescale factors α, published by the score mapping for the fold mapping
    let (als, alsp) = if online { k.shared_f32(rows) } else { (0, 0) };

    let uid = k.uint;
    let fid = k.float;
    let glsl = k.glsl;
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let c_hd = k.cu(hd);
    let c_kt = k.cu(kt);
    let c_t = k.cu(tsz);
    let c_kl = k.cu(kl);
    let c_nkv = k.cu(nkv);
    let c_tmax = k.cu(tmax);
    let c_qkvm = k.cu(qkvm);
    let c_qn = k.cu(qn);
    let c_scale = k.cf(scale);
    let zf = k.cf(0.0);
    let neg = k.cf(-1.0e30);
    let c_tm1 = k.cu(tmax - 1);
    let tpow = tmax.is_power_of_two();

    // the mandala closure on a KV slot: `key &&& (tmax−1)` when the window is a petal (2^k cells),
    // `UMin(key, tmax−1)` when the caller sized it otherwise. Either way the load's image IS the
    // window, so no key — including a `pos0` a host buffer got wrong — has an out-of-region form.
    let clampk = move |k: &mut Kb, s: u32| -> u32 {
        if tpow {
            k.e(OP_BITWISE_AND, uid, &[s, c_tm1])
        } else {
            k.e(OP_EXT_INST, uid, &[glsl, GLSL_UMIN, s, c_tm1])
        }
    };
    let umin = move |k: &mut Kb, a: u32, b: u32| -> u32 {
        k.e(OP_EXT_INST, uid, &[glsl, GLSL_UMIN, a, b])
    };

    let gid0 = k.gid_comp(0);
    let ti = k.gid_comp(1);
    let tid = k.umod(gid0, c_t);
    let h = k.udiv(gid0, c_t);
    let cg = k.cu(group);
    let kvh = k.udiv(h, cg);
    let pos0 = k.load_u(parb, c0);
    let nn = k.load_u(parb, c1);
    let nm1 = k.e(OP_ISUB, uid, &[nn, c1]);
    let c_rows = k.cu(rows);
    // a query tile's work grows with its index (the causal prefix), so launching the LAST tile
    // first lets the heavy workgroups claim SMs before the light ones fill the machine
    let ti = if mode & ATTN_MODE_REVERSE != 0 {
        let cr1 = k.cu(rows - 1);
        let nr = k.iadd(nn, cr1);
        let nt = k.udiv(nr, c_rows);
        let nt1 = k.e(OP_ISUB, uid, &[nt, c1]);
        k.e(OP_ISUB, uid, &[nt1, ti])
    } else {
        ti
    };
    let i0 = k.imul(ti, c_rows);
    let hb = k.imul(h, c_hd);

    // two thread mappings, both over the same `tsz` threads and separated by barriers:
    //   FOLD  (dl, fs): dl = output dim, fs = row slot — the Q stage, every reduction tree, the
    //                   value fold and the store;
    //   SCORE (kli, sli): kli = key lane (owns keys kli·ct .. +ct of a slab), sli = row slot.
    let dl = k.umod(tid, c_hd);
    let fs = k.udiv(tid, c_hd);
    let kli = k.umod(tid, c_kl);
    let sli = k.udiv(tid, c_kl);
    let c_rtf = k.cu(rtf);
    let c_rts = k.cu(rts);
    let c_ct = k.cu(ct);
    let fbase = k.imul(fs, c_rtf);
    let sbase = k.imul(sli, c_rts);
    let kcol = k.imul(kli, c_ct);

    // ── the tile's query rows into shared (fold mapping) ────────────────────────────────────────
    // A row past `n` is folded onto `n−1`: it then recomputes the last real row's answer, which is
    // finite and in-region, instead of scoring uninitialized KV. Its store is dropped below.
    for rt in 0..rtf {
        let crt = k.cu(rt);
        let lr = k.iadd(fbase, crt);
        let ir = k.iadd(i0, lr);
        let ic = umin(&mut k, ir, nm1);
        let qrow = k.imul(ic, c_qkvm);
        let qi0 = k.iadd(qrow, hb);
        let qi = k.iadd(qi0, dl);
        let qv = k.load_f(qkvb, qi);
        let s0 = k.imul(lr, c_hd);
        let si = k.iadd(s0, dl);
        k.shared_store(qs, qsp, si, qv);
    }
    k.wg_barrier();

    // the widest causal prefix in this tile — the slab loop's bound, WG-uniform
    let c_rm1 = k.cu(rows - 1);
    let ilast = k.iadd(i0, c_rm1);
    let imax = umin(&mut k, ilast, nm1);
    let mpos = k.iadd(pos0, imax);
    let maxlen = k.iadd(mpos, c1);

    // the score mapping's rows and their own causal lengths
    let mut lrow = Vec::with_capacity(rts as usize);
    let mut qbase = Vec::with_capacity(rts as usize);
    let mut lenv = Vec::with_capacity(rts as usize);
    for j in 0..rts {
        let cj = k.cu(j);
        let lr = k.iadd(sbase, cj);
        let ir = k.iadd(i0, lr);
        let ic = umin(&mut k, ir, nm1);
        let p = k.iadd(pos0, ic);
        let ln = k.iadd(p, c1);
        let qb = k.imul(lr, c_hd);
        lrow.push(lr);
        qbase.push(qb);
        lenv.push(ln);
    }

    // ── the K slab stage ────────────────────────────────────────────────────────────────────────
    // `kc` is dim-major — `kc[(kvh·hd+d)·tmax+s]` — which is exactly the layout this wants: for a
    // fixed `d` the slab's keys are CONTIGUOUS, so the stage is one coalesced run per row, and the
    // shared copy `kst[d·kt+j]` is read by consecutive key lanes at consecutive banks. The
    // "transposed relative to llama.cpp" note in the ranked gap is true and does not cost anything
    // here; it is what makes the stage coalesced without a transpose.
    let j0 = k.umod(tid, c_kt);
    let d0 = k.udiv(tid, c_kt);
    let dstep = tsz / kt;
    assert!(
        dstep > 0 && dchunk % dstep == 0,
        "the K stage needs LocalSize/hd to divide the dim chunk"
    );
    let c_dstep = k.cu(dstep);
    let c_dchunk = k.cu(dchunk);
    let c_hdt = k.cu(hd * tmax);
    let kvbase = k.imul(kvh, c_hdt);

    // ── the K slab stage (score mapping) ────────────────────────────────────────────────────────
    // Stage chunk `p` — `dchunk` dims of every `kt` key — at `kst[dj·kt + j0]`. `d0`/`dstep` split
    // the `dchunk` dims across the workgroup; the global dim is `p·dchunk + dj`. `kp == 1` is the
    // whole-head stage (`dj` IS the global dim), verbatim the pre-split kernel.
    let load_chunk = move |k: &mut Kb, t0: u32, p: u32| {
        let s = k.iadd(t0, j0);
        let sm = clampk(k, s);
        let b0 = k.iadd(kvbase, sm);
        let pbase = k.cu(p * dchunk);
        k.loop_step(d0, c_dchunk, c_dstep, |k, dj| {
            let d = if kp == 1 { dj } else { k.iadd(pbase, dj) };
            let dt = k.imul(d, c_tmax);
            let gi = k.iadd(b0, dt);
            let v = k.load_f(kcb, gi);
            let s0 = k.imul(dj, c_kt);
            let si = k.iadd(s0, j0);
            k.shared_store(kst, kstp, si, v);
        });
    };

    // ── the score GEMM (score mapping) ──────────────────────────────────────────────────────────
    // Per `d` the thread loads `rts` Q values and `ct` K values and issues `rts·ct` FMAs — the
    // register tile that keeps this off the shared-memory roof (a scalar dot needs two shared reads
    // per MAC; this needs (rts+ct)/(rts·ct) = 0.75 at 2×4). The `d` loop is the SAME ascending chain
    // `attn_prefill_module::score` walks, one accumulator per (row, key).
    //
    // The `kp`-way dim split folds every chunk of K into the same `rts·ct` registers, so the peak
    // slab is `dchunk·kt·4` instead of `hd·kt·4`. The staged bytes, the FMA count and the
    // `d`-ascending accumulator chain are the whole-head kernel's — only the shared footprint
    // falls. The CALLER stages chunk 0 (`barrier, load_chunk(·,0), barrier`) before this call; the
    // remaining chunks are staged here, so `kp == 1` emits the one-chunk form verbatim.
    let scores = |k: &mut Kb, t0: u32| -> Vec<u32> {
        let accs: Vec<u32> = (0..rts * ct)
            .map(|_| {
                let v = k.fvar_f();
                k.store_var(v, zf);
                v
            })
            .collect();
        let score_chunk = |k: &mut Kb, p: u32| {
            let pbase = k.cu(p * dchunk);
            let cend = k.cu((p + 1) * dchunk);
            k.loop_u(pbase, cend, |k, d| {
                let dl = if kp == 1 {
                    d
                } else {
                    k.e(OP_ISUB, uid, &[d, pbase])
                };
                let dk = k.imul(dl, c_kt);
                let kb = k.iadd(dk, kcol);
                let kvv: Vec<u32> = (0..ct)
                    .map(|c| {
                        let cc = k.cu(c);
                        let idx = k.iadd(kb, cc);
                        k.shared_load(kst, kstp, idx)
                    })
                    .collect();
                for j in 0..rts as usize {
                    let qb = k.iadd(qbase[j], d);
                    let qv = k.shared_load(qs, qsp, qb);
                    for c in 0..ct as usize {
                        let pr = k.fmul(qv, kvv[c]);
                        let a0 = k.load_var_f(accs[j * ct as usize + c]);
                        let a1 = k.fadd(a0, pr);
                        k.store_var(accs[j * ct as usize + c], a1);
                    }
                }
            });
        };
        score_chunk(k, 0);
        for p in 1..kp {
            k.wg_barrier();
            load_chunk(k, t0, p);
            k.wg_barrier();
            score_chunk(k, p);
        }
        // ×ks[s]·scale, in `attn_prefill_module`'s multiply order
        let ksv: Vec<u32> = (0..ct)
            .map(|c| {
                let cc = k.cu(c);
                let s0 = k.iadd(t0, kcol);
                let s = k.iadd(s0, cc);
                let sm = clampk(k, s);
                let r = k.imul(sm, c_nkv);
                let idx = k.iadd(r, kvh);
                k.load_f(ksb, idx)
            })
            .collect();
        let mut out = Vec::with_capacity((rts * ct) as usize);
        for j in 0..rts as usize {
            for c in 0..ct as usize {
                let raw = k.load_var_f(accs[j * ct as usize + c]);
                let t1 = k.fmul(raw, ksv[c]);
                out.push(k.fmul(t1, c_scale));
            }
        }
        out
    };
    // the key this thread's (j, c) slot holds inside the slab starting at `t0`
    let keyof = move |k: &mut Kb, t0: u32, c: u32| -> u32 {
        let cc = k.cu(c);
        let s0 = k.iadd(t0, kcol);
        k.iadd(s0, cc)
    };
    // `rows` independent log₂(hd) trees at once, over `sst` viewed as [row][hd] — the same pairing
    // `Kb::tree_reduce` uses, so the denominator matches the untiled kernel bit for bit.
    let reduce_rows = move |k: &mut Kb, is_max: bool| {
        let mut stride = hd / 2;
        while stride >= 1 {
            let cs = k.cu(stride);
            let lt = k.ult(dl, cs);
            k.iff(lt, |k| {
                for rt in 0..rtf {
                    let crt = k.cu(rt);
                    let lr = k.iadd(fbase, crt);
                    let base = k.imul(lr, c_kt);
                    let ia = k.iadd(base, dl);
                    let a = k.shared_load(sst, sstp, ia);
                    let ib0 = k.iadd(ia, cs);
                    let b = k.shared_load(sst, sstp, ib0);
                    let r = if is_max {
                        k.ext2(GLSL_FMAX, a, b)
                    } else {
                        k.fadd(a, b)
                    };
                    k.shared_store(sst, sstp, ia, r);
                }
            });
            k.wg_barrier();
            stride /= 2;
        }
    };

    // ── pass 1: the global max (the online form has no such walk) ────────────────────────────────
    let mrow: Vec<u32> = if online {
        Vec::new()
    } else {
        let mx: Vec<u32> = (0..rts * ct)
            .map(|_| {
                let v = k.fvar_f();
                k.store_var(v, neg);
                v
            })
            .collect();
        k.loop_step(c0, maxlen, c_kt, |k, t0| {
            k.wg_barrier();
            load_chunk(k, t0, 0);
            k.wg_barrier();
            let sc = scores(k, t0);
            for j in 0..rts as usize {
                for c in 0..ct as usize {
                    let s = keyof(k, t0, c as u32);
                    let inr = k.ult(s, lenv[j]);
                    let v = k.e(OP_SELECT, fid, &[inr, sc[j * ct as usize + c], neg]);
                    let m0 = k.load_var_f(mx[j * ct as usize + c]);
                    let m1 = k.ext2(GLSL_FMAX, m0, v);
                    k.store_var(mx[j * ct as usize + c], m1);
                }
            }
        });
        k.wg_barrier();
        for j in 0..rts as usize {
            for c in 0..ct as usize {
                let base = k.imul(lrow[j], c_kt);
                let cc = k.cu(c as u32);
                let p0 = k.iadd(base, kcol);
                let p = k.iadd(p0, cc);
                let v = k.load_var_f(mx[j * ct as usize + c]);
                k.shared_store(sst, sstp, p, v);
            }
        }
        k.wg_barrier();
        reduce_rows(&mut k, true);
        let mrow: Vec<u32> = (0..rts as usize)
            .map(|j| {
                let base = k.imul(lrow[j], c_kt);
                k.shared_load(sst, sstp, base)
            })
            .collect();
        k.wg_barrier();
        mrow
    };

    // ── pass 2: the partition function (and, when `fuse`, the value fold with it) ───────────────
    let sums: Vec<u32> = (0..rts * ct)
        .map(|_| {
            let v = k.fvar_f();
            k.store_var(v, zf);
            v
        })
        .collect();
    let c_hdtv = k.cu(tmax);
    let vrow0 = k.imul(kvh, c_hdtv);
    let oacc: Vec<u32> = (0..rtf)
        .map(|_| {
            let v = k.fvar_f();
            k.store_var(v, zf);
            v
        })
        .collect();
    // the value fold over one staged slab: one V element per (key, output dim) serves all `rtf`
    // rows the thread owns, and the per-key weight comes out of `sst` where the key's owner put it
    let fold_slab = |k: &mut Kb, t0: u32| {
        k.loop_u(c0, c_kt, |k, jj| {
            let s = k.iadd(t0, jj);
            let jin = k.ult(s, maxlen);
            let sm = clampk(k, s);
            let vb0 = k.iadd(vrow0, sm);
            let vb1 = k.imul(vb0, c_hd);
            let vidx = k.iadd(vb1, dl);
            let vv0 = k.load_f(vcb, vidx);
            // a slot past the written window is zeroed: `0·NaN` would poison the row, and
            // `0.0·0.0` leaves the accumulator exactly where it was
            let vv = k.e(OP_SELECT, fid, &[jin, vv0, zf]);
            for rt in 0..rtf {
                let crt = k.cu(rt);
                let lr = k.iadd(fbase, crt);
                let base = k.imul(lr, c_kt);
                let pi = k.iadd(base, jj);
                let pj = k.shared_load(sst, sstp, pi);
                let pr = k.fmul(pj, vv);
                let a0 = k.load_var_f(oacc[rt as usize]);
                let a1 = k.fadd(a0, pr);
                k.store_var(oacc[rt as usize], a1);
            }
        });
    };
    if online {
        let mrun: Vec<u32> = (0..rts)
            .map(|_| {
                let v = k.fvar_f();
                k.store_var(v, neg);
                v
            })
            .collect();
        let lane0 = k.ult(kli, c1);
        k.loop_step(c0, maxlen, c_kt, |k, t0| {
            k.wg_barrier();
            load_chunk(k, t0, 0);
            k.wg_barrier();
            let sc = scores(k, t0);
            let cts = ct as usize;
            let inr: Vec<u32> = (0..rts as usize * cts)
                .map(|jc| {
                    let s = keyof(k, t0, (jc % cts) as u32);
                    k.ult(s, lenv[jc / cts])
                })
                .collect();
            // the slab max of each row: this thread's `ct` masked keys, then the subgroup's 32 lanes
            let mut alpha = Vec::with_capacity(rts as usize);
            let mut mnew = Vec::with_capacity(rts as usize);
            for j in 0..rts as usize {
                let mut mj = k.e(OP_SELECT, fid, &[inr[j * cts], sc[j * cts], neg]);
                for c in 1..cts {
                    let v = k.e(OP_SELECT, fid, &[inr[j * cts + c], sc[j * cts + c], neg]);
                    mj = k.ext2(GLSL_FMAX, mj, v);
                }
                let sm = sg_fmax(k, mj);
                let mo = k.load_var_f(mrun[j]);
                let mn = k.ext2(GLSL_FMAX, mo, sm);
                let dm = k.fsub(mo, mn);
                alpha.push(k.ext1(GLSL_EXP, dm));
                k.store_var(mrun[j], mn);
                mnew.push(mn);
            }
            for j in 0..rts as usize {
                for c in 0..cts {
                    let s = keyof(k, t0, c as u32);
                    let dd = k.fsub(sc[j * cts + c], mnew[j]);
                    let e = k.ext1(GLSL_EXP, dd);
                    let ee = k.e(OP_SELECT, fid, &[inr[j * cts + c], e, zf]);
                    let s0 = k.load_var_f(sums[j * cts + c]);
                    let s1a = k.fmul(s0, alpha[j]);
                    let s1 = k.fadd(s1a, ee);
                    k.store_var(sums[j * cts + c], s1);
                    let sm = clampk(k, s);
                    let r = k.imul(sm, c_nkv);
                    let vi = k.iadd(r, kvh);
                    let vsv = k.load_f(vsb, vi);
                    let p0 = k.fmul(e, vsv);
                    let p = k.e(OP_SELECT, fid, &[inr[j * cts + c], p0, zf]);
                    let base = k.imul(lrow[j], c_kt);
                    let cc = k.cu(c as u32);
                    let q0 = k.iadd(base, kcol);
                    let q = k.iadd(q0, cc);
                    k.shared_store(sst, sstp, q, p);
                }
                // one lane per subgroup publishes the row's α for the fold mapping
                k.iff(lane0, |k| {
                    k.shared_store(als, alsp, lrow[j], alpha[j]);
                });
            }
            k.wg_barrier();
            for rt in 0..rtf {
                let crt = k.cu(rt);
                let lr = k.iadd(fbase, crt);
                let a = k.shared_load(als, alsp, lr);
                let o0 = k.load_var_f(oacc[rt as usize]);
                let o1 = k.fmul(o0, a);
                k.store_var(oacc[rt as usize], o1);
            }
            fold_slab(k, t0);
        });
    } else {
        k.loop_step(c0, maxlen, c_kt, |k, t0| {
            k.wg_barrier();
            load_chunk(k, t0, 0);
            k.wg_barrier();
            let sc = scores(k, t0);
            for j in 0..rts as usize {
                for c in 0..ct as usize {
                    let s = keyof(k, t0, c as u32);
                    let inr = k.ult(s, lenv[j]);
                    let dd = k.fsub(sc[j * ct as usize + c], mrow[j]);
                    let e = k.ext1(GLSL_EXP, dd);
                    // `+0.0` is the exact identity, so a masked key leaves the partial untouched
                    let ee = k.e(OP_SELECT, fid, &[inr, e, zf]);
                    let s0 = k.load_var_f(sums[j * ct as usize + c]);
                    let s1 = k.fadd(s0, ee);
                    k.store_var(sums[j * ct as usize + c], s1);
                    if fuse {
                        // the UNNORMALIZED weight `e·vs[s]` — the fold's `1/Σe` is factored out to the
                        // epilogue, which is the whole (and only) arithmetic difference `fuse` makes
                        let sm = clampk(k, s);
                        let r = k.imul(sm, c_nkv);
                        let vi = k.iadd(r, kvh);
                        let vsv = k.load_f(vsb, vi);
                        let p0 = k.fmul(e, vsv);
                        let p = k.e(OP_SELECT, fid, &[inr, p0, zf]);
                        let base = k.imul(lrow[j], c_kt);
                        let cc = k.cu(c as u32);
                        let q0 = k.iadd(base, kcol);
                        let q = k.iadd(q0, cc);
                        k.shared_store(sst, sstp, q, p);
                    }
                }
            }
            if fuse {
                k.wg_barrier();
                fold_slab(k, t0);
            }
        });
    }
    k.wg_barrier();
    for j in 0..rts as usize {
        for c in 0..ct as usize {
            let base = k.imul(lrow[j], c_kt);
            let cc = k.cu(c as u32);
            let p0 = k.iadd(base, kcol);
            let p = k.iadd(p0, cc);
            let v = k.load_var_f(sums[j * ct as usize + c]);
            k.shared_store(sst, sstp, p, v);
        }
    }
    k.wg_barrier();
    reduce_rows(&mut k, false);
    let trow: Vec<u32> = (0..rts as usize)
        .map(|j| {
            let base = k.imul(lrow[j], c_kt);
            k.shared_load(sst, sstp, base)
        })
        .collect();
    // the fold mapping's view of the same denominators, for the fused epilogue
    let tfold: Vec<u32> = if fuse {
        (0..rtf)
            .map(|rt| {
                let crt = k.cu(rt);
                let lr = k.iadd(fbase, crt);
                let base = k.imul(lr, c_kt);
                k.shared_load(sst, sstp, base)
            })
            .collect()
    } else {
        vec![]
    };
    k.wg_barrier();

    // ── pass 3: the value fold, keys ASCENDING (skipped when `fuse` already did it) ─────────────
    if !fuse {
        k.loop_step(c0, maxlen, c_kt, |k, t0| {
            k.wg_barrier();
            load_chunk(k, t0, 0);
            k.wg_barrier();
            let sc = scores(k, t0);
            // the slab's softmax weights, formed by the thread that owns the key and published once
            for j in 0..rts as usize {
                for c in 0..ct as usize {
                    let s = keyof(k, t0, c as u32);
                    let inr = k.ult(s, lenv[j]);
                    let sm = clampk(k, s);
                    let dd = k.fsub(sc[j * ct as usize + c], mrow[j]);
                    let e0 = k.ext1(GLSL_EXP, dd);
                    let en = k.fdiv(e0, trow[j]);
                    let r = k.imul(sm, c_nkv);
                    let vi = k.iadd(r, kvh);
                    let vsv = k.load_f(vsb, vi);
                    let p0 = k.fmul(en, vsv);
                    let p = k.e(OP_SELECT, fid, &[inr, p0, zf]);
                    let base = k.imul(lrow[j], c_kt);
                    let cc = k.cu(c as u32);
                    let q0 = k.iadd(base, kcol);
                    let q = k.iadd(q0, cc);
                    k.shared_store(sst, sstp, q, p);
                }
            }
            k.wg_barrier();
            fold_slab(k, t0);
        });
    }

    // ── the store — the one branch left, and the reason it is a branch ──────────────────────────
    // `att` is `bp·qn` f32 with `bp` NOT a power of two, so there is no mask that folds a padded row
    // back into a HARMLESS cell: the mandala closure would land it on some other position's output.
    // The overhang is `ceil(n/rows)·rows − n` rows of a dispatch the host sized, so the guard is a
    // WG-uniform-ish predicate on data the host wrote, and dropping the store is the correct
    // closure — `no_key_escapes` for a region whose petals are ANSWERS, not bytes.
    for rt in 0..rtf {
        let crt = k.cu(rt);
        let lr = k.iadd(fbase, crt);
        let ir = k.iadd(i0, lr);
        let live = k.ult(ir, nn);
        let raw = k.load_var_f(oacc[rt as usize]);
        let of = if fuse {
            k.fdiv(raw, tfold[rt as usize])
        } else {
            raw
        };
        k.iff(live, |k| {
            let arow = k.imul(ir, c_qn);
            let ab = k.iadd(arow, hb);
            let aidx = k.iadd(ab, dl);
            k.store_f(attb, aidx, of);
        });
    }
    k.finish(tsz)
}

/// [`attn_prefill_tiled_module_m`] as bytes.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_tiled_bytes_m(
    nh: u32,
    nkv: u32,
    hd: u32,
    qkvm: u32,
    tmax: u32,
    scale: f32,
    rows: u32,
    rts: u32,
    ct: u32,
    kp: u32,
    fuse: bool,
    mode: u32,
) -> Vec<u8> {
    bytes(attn_prefill_tiled_module_m(
        nh, nkv, hd, qkvm, tmax, scale, rows, rts, ct, kp, fuse, mode,
    ))
}

/// [`attn_prefill_tiled_module`] as bytes.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_tiled_bytes(
    nh: u32,
    nkv: u32,
    hd: u32,
    qkvm: u32,
    tmax: u32,
    scale: f32,
    rows: u32,
    rts: u32,
    ct: u32,
    fuse: bool,
) -> Vec<u8> {
    bytes(attn_prefill_tiled_module(
        nh, nkv, hd, qkvm, tmax, scale, rows, rts, ct, fuse,
    ))
}

/// The LocalSize [`attn_prefill_tiled_module`] compiles to for a given tile — the driver needs it to
/// size its dispatch, and the peak shared-memory footprint `(2·rows + hd/kp)·hd·4` to refuse a tile
/// the device cannot hold. `kp` is the K-slab dim-chunk count (1 = the whole head staged at once);
/// the query tile and the score row are `rows·hd` f32 each, the slab is `(hd/kp)·hd` f32.
pub fn attn_tile_shape(hd: u32, rows: u32, rts: u32, ct: u32, kp: u32) -> (u32, u32) {
    let tsz = (hd / ct) * (rows / rts);
    let shared = (2 * rows * hd + (hd / kp) * hd) * 4;
    (tsz, shared)
}
/// [`attn_prefill_module`] as bytes.
pub fn attn_prefill_bytes(nh: u32, nkv: u32, hd: u32, qkvm: u32, tmax: u32, scale: f32) -> Vec<u8> {
    bytes(attn_prefill_module(nh, nkv, hd, qkvm, tmax, scale))
}

/// Query rows per workgroup of the cooperative-matrix attention kernel (`BR = 16·NSG` in
/// `src/spirv/attn_coopmat.comp`, committed at `NSG = 4`): the `Gy::Tiles` unit.
pub const ATTN_COOP_ROWS: u32 = 64;

/// The compiled cooperative-matrix attention kernel (`glslc` [`GLSLC_ARGS`] over
/// `src/spirv/attn_coopmat.comp`), specialization defaults unpatched.
pub static ATTN_COOPMAT_SPV: &[u8] = include_bytes!("../spirv/attn_coopmat.spv");

/// Rewrite the default of every 32-bit `OpSpecConstant` whose `SpecId` is a key of `vals` (the
/// value is the constant's raw word; an `f32` goes in as `to_bits`). Panics when a `SpecId` is
/// absent: an unpatched constant is a different kernel. Same walk as `coopmat_gemm`'s private
/// `specialize`, kept local because that file is not this task's.
///
/// Cost: two linear passes over the module, O(words) time, O(1) extra space beyond the copy.
fn attn_coop_specialize(module: &[u32], vals: &[(u32, u32)]) -> Vec<u32> {
    const OP_DECORATE: u32 = 71;
    const OP_SPEC_CONSTANT: u32 = 50;
    const DEC_SPEC_ID: u32 = 1;
    let mut out = module.to_vec();
    let mut ids: Vec<(u32, u32)> = Vec::new();
    let mut i = 5;
    while i < out.len() {
        let (len, op) = ((out[i] >> 16) as usize, out[i] & 0xffff);
        assert!(
            len > 0 && i + len <= out.len(),
            "malformed SPIR-V at word {i}"
        );
        if op == OP_DECORATE && len == 4 && out[i + 2] == DEC_SPEC_ID {
            if let Some(&(_, v)) = vals.iter().find(|(s, _)| *s == out[i + 3]) {
                ids.push((out[i + 1], v));
            }
        }
        i += len;
    }
    assert_eq!(ids.len(), vals.len(), "a SpecId is missing from the module");
    let mut patched = 0;
    i = 5;
    while i < out.len() {
        let (len, op) = ((out[i] >> 16) as usize, out[i] & 0xffff);
        if op == OP_SPEC_CONSTANT && len == 4 {
            if let Some(&(_, v)) = ids.iter().find(|(id, _)| *id == out[i + 2]) {
                out[i + 3] = v;
                patched += 1;
            }
        }
        i += len;
    }
    assert_eq!(
        patched,
        vals.len(),
        "a SpecId names no 32-bit OpSpecConstant"
    );
    out
}

/// **Cooperative-matrix causal prefill attention** (fp16 operands, f32 accumulate): the drop-in
/// for [`attn_prefill_tiled_module_m`] at `hd == 64` (same seven bindings, same caches, dispatch
/// `(nh, ceil(n / ATTN_COOP_ROWS))`). `qkvm` and `tmax` must be multiples of 4.
pub fn attn_prefill_coopmat_module(
    nh: u32,
    nkv: u32,
    qkvm: u32,
    tmax: u32,
    scale: f32,
    dpw: u32,
) -> Vec<u32> {
    assert!(nkv > 0 && nh % nkv == 0, "GQA needs nkv | nh");
    assert!(
        qkvm % 4 == 0 && tmax % 4 == 0 && tmax >= 4,
        "qkvm and tmax must be multiples of 4"
    );
    assert!(dpw == 1 || dpw == 2, "dpw must be 1 or 2");
    assert!(ATTN_COOPMAT_SPV.len() % 4 == 0, "SPIR-V is a word stream");
    let words: Vec<u32> = ATTN_COOPMAT_SPV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0], crate::simt::SPV_MAGIC, "not a SPIR-V module");
    attn_coop_specialize(
        &words,
        &[
            (0, nh),
            (1, nkv),
            (2, qkvm),
            (3, tmax),
            (4, scale.to_bits()),
            (5, dpw),
        ],
    )
}

/// [`attn_prefill_coopmat_module`] as bytes.
pub fn attn_prefill_coopmat_bytes(
    nh: u32,
    nkv: u32,
    qkvm: u32,
    tmax: u32,
    scale: f32,
    dpw: u32,
) -> Vec<u8> {
    bytes(attn_prefill_coopmat_module(nh, nkv, qkvm, tmax, scale, dpw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attn_coopmat_specialization_dpw() {
        let m1 = attn_prefill_coopmat_module(14, 2, 1152, 2048, 0.125, 1);
        assert!(!m1.is_empty());
        let m2 = attn_prefill_coopmat_module(14, 2, 1152, 2048, 0.125, 2);
        assert!(!m2.is_empty());
        assert_ne!(m1, m2);
    }

    #[test]
    fn attn_coopmat_dpw2_word_layout_matches_quantize_batch_module() {
        let dpw = 2u32;
        for w in 0..16u32 {
            let coop_indices: Vec<u32> =
                (0..4u32).map(|b| (w >> 1) * 8 + (w & 1) + b * 2).collect();
            let cc = w / dpw;
            let ll = w % dpw;
            let base0 = cc * 4 * dpw;
            let base = base0 + ll;
            let quant_indices: Vec<u32> = (0..4u32).map(|b| base + b * dpw).collect();
            assert_eq!(coop_indices, quant_indices, "mismatch at word {w}");
        }
    }
}
