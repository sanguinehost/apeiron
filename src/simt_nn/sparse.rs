//! simt_nn::sparse — the knot (sparse-attention) kernels: the theta scan / e-score / theta
//! select block chooser and the flash partial/combine split-softmax pair.
//! Cut verbatim from `simt_nn.rs`.

use super::*;

/// **The SELECTOR, scan stage — the degree-1 half of attention** (sanguine `SpinorFold.one_half_
/// decides`: the block box is the boundary coordinate that decides relevance without touching the
/// keys). One WG per (q-head, slice) folds `ub_b = scale·(⟨q⁺,max_b⟩ + ⟨q⁻,min_b⟩) ≥ ⟪q,k⟫`
/// (`SparseSelection.dot_ball_bound`; `q⁺=max(q,0)`, `q⁻=min(q,0)` split the sign-select into two
/// straight dots) over its slice of the ⌈len/blk⌉ blocks, caches every `ub` and tree-maxes a
/// per-slice partial. The dots ride the int8 SDot planes `fused_rope_kv` packs (4 dims/inst, 4×
/// less traffic, 16-deep chain vs the f16 form's 64 — the same `fold_in_the_seed_chart` atom as
/// the score). Fanned at `nh·nsplit` WGs — one-WG-per-head was the `lone_reduction_is_latency_
/// bound` starvation (969µs at 64k). Bindings: 0=q 1=boxq(u32 planes) 2=sb 3=params
/// 4=pmax(nh·nsplit) 5=ub(nh·nblocks). Dispatch (nh·nsplit,1).
pub fn theta_scan_module(
    nh: u32,
    nkv: u32,
    hd: u32,
    tmax: u32,
    scale: f32,
    blk: u32,
    nsplit: u32,
) -> Vec<u32> {
    const WG: u32 = 64;
    let group = nh / nkv.max(1);
    let nblocks = tmax.div_ceil(blk);
    let hd4 = hd / 4;
    let mut k = Kb::new();
    k.dp4a = true;
    let q = k.ssbo_f(0);
    let bxq = k.ssbo_u(1); // int8 CENTROID plane: bxq[(kvh·(hd/4)+w)·nblocks+b]
    let sbb = k.ssbo_f(2); // per-block centroid scale sc[kvh·nblocks+b]
    let params = k.ssbo_u(3);
    let pmax = k.ssbo_f(4);
    let ubo = k.ssbo_f(5); // ub[h·nblocks + b] — theta_select compacts it, flash never re-reads it
    let rho = k.ssbo_f(6); // per-block covering radius ρ[kvh·nblocks+b]
    let (qq, qqp) = k.shared_u32(hd4); // packed int8 q (single plane) for ⟪q,c⟫
    let (red, redp) = k.shared_f32(WG);
    let gid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let hs = k.udiv(gid, cwg); // h·nsplit + slice
    let lane = k.umod(gid, cwg);
    let cns = k.cu(nsplit);
    let h = k.udiv(hs, cns);
    let sp = k.umod(hs, cns);
    let ctot = k.cu(nh * nsplit);
    let guard = k.ult(hs, ctot);
    k.iff(guard, |k| {
        let c0 = k.cu(0);
        let c1 = k.cu(1);
        let pos = k.load_u(params, c0);
        let len = k.iadd(pos, c1);
        let cg = k.cu(group);
        let kvh = k.udiv(h, cg);
        let chd4 = k.cu(hd4);
        let cnb = k.cu(nblocks);
        let chd = k.cu(hd);
        let qbase = k.imul(h, chd);
        // int8-quantize q into ONE shared plane (scale sq), and compute ‖q‖ (L2) for the ball term.
        k.absmax_reduce_global(q, qbase, red, redp, hd, WG, lane);
        let qamax = k.shared_load(red, redp, c0);
        let inv127 = k.cf(1.0 / 127.0);
        let sq = k.fmul(qamax, inv127);
        k.loop_step(lane, chd4, cwg, |k, w| {
            let c4 = k.cu(4);
            let w4 = k.imul(w, c4);
            let base4 = k.iadd(qbase, w4);
            let word = k.pack_int8_global(q, base4, sq);
            k.shared_store_u(qq, qqp, w, word);
        });
        // ‖q‖ = sqrt(Σ q²): each lane sums its strided dims, tree-reduce
        let sq2 = k.fvar_f();
        let zf0 = k.cf(0.0);
        k.store_var(sq2, zf0);
        k.loop_step(lane, chd, cwg, |k, i| {
            let qi = k.iadd(qbase, i);
            let v = k.load_f(q, qi);
            let vv = k.fmul(v, v);
            let a0 = k.load_var_f(sq2);
            let a1 = k.fadd(a0, vv);
            k.store_var(sq2, a1);
        });
        let sq2v = k.load_var_f(sq2);
        k.wg_barrier();
        k.shared_store(red, redp, lane, sq2v);
        k.wg_barrier();
        k.tree_reduce(red, redp, WG, lane, false);
        let qn2 = k.shared_load(red, redp, c0);
        let qnorm = k.ext1(GLSL_SQRT, qn2); // ‖q‖ (L2) — the ball bound's radius coefficient
        k.wg_barrier();
        // this slice's block range [b0,b1) of the nba = ⌈len/blk⌉ active blocks
        let cblk = k.cu(blk);
        let blkm1 = k.cu(blk - 1);
        let lp = k.iadd(len, blkm1);
        let nba = k.udiv(lp, cblk);
        let nsm1 = k.cu(nsplit - 1);
        let bnum = k.iadd(nba, nsm1);
        let bslice = k.udiv(bnum, cns);
        let b0 = k.imul(sp, bslice);
        let b0e = k.iadd(b0, bslice);
        let bover = k.ult(nba, b0e);
        let b1 = k.e(OP_SELECT, k.uint, &[bover, nba, b0e]);
        let lub = k.fvar_f();
        let negu = k.cf(-1.0e30);
        k.store_var(lub, negu);
        let bstart = k.iadd(b0, lane);
        let sint = k.sint_ty();
        k.loop_step(bstart, b1, cwg, |k, b| {
            // BALL bound (`dot_ball_bound` + mirror): scale·⟪q,k⟫ ∈ [t1 − w, t1 + w] where
            // t1 = scale·sq·sc·⟨q_packed, c_packed⟩ (one dp4a, not the box's 4 SDOTs) and
            // w = scale·‖q‖·ρ. ub = t1 + w is the keep test; max_lb (t1 − w folded) is the exact
            // anchor ≤ m_true. Tighter than the axis box by ~√hd — the L2 radius, not the L1 corner.
            let iacc = k.fvar_s();
            let sz = k.sconst0();
            k.store_var(iacc, sz);
            let crow = k.imul(kvh, chd4); // centroid plane row base (kvh·(hd/4)+w)·nblocks
            let c0w = k.cu(0);
            k.loop_u(c0w, chd4, |k, w| {
                let qw = k.shared_load_u(qq, qqp, w);
                let cr0 = k.iadd(crow, w);
                let cr1 = k.imul(cr0, cnb);
                let cidx = k.iadd(cr1, b);
                let cw = k.load_u(bxq, cidx);
                let d0 = k.e(OP_SDOT, sint, &[qw, cw, PACKED_VEC_4X8]);
                let a0 = k.load_var_s(iacc);
                let a1 = k.e(OP_IADD, sint, &[a0, d0]);
                k.store_var(iacc, a1);
            });
            let f = k.float;
            let sbrow = k.imul(kvh, cnb);
            let sbi = k.iadd(sbrow, b);
            let scv = k.load_f(sbb, sbi); // centroid scale sc
            let csc = k.cf(scale);
            let t0 = k.fmul(csc, sq);
            let tsc = k.fmul(t0, scv); // scale·sq·sc
            let isum = k.load_var_s(iacc);
            let fdot = k.e(OP_CONVERT_S_TO_F, f, &[isum]);
            let t1 = k.fmul(tsc, fdot); // scale·⟪q,c⟫ — the block's centroid score
                                        // The ball's UPPER and LOWER bounds (`dot_ball_bound` / `dot_ball_lower_bound`):
                                        // scale·⟪q,k⟫ ∈ [t1 − w, t1 + w], w = scale·‖q‖·ρ. theta_select does NOT rank/cut by
                                        // the bound distribution (that is `heavy_cut_is_score_threshold` fed BOUNDS, whose
                                        // argmax ≠ the score argmax — `ub_rank_is_not_score_rank`, the measured needle miss).
                                        // Instead it runs `score_anchored_gather_keeps_argmax`: keep block b iff ub_b > θ with
                                        // θ = mlb − L SCORE-anchored (mlb = the max LOWER bound ≤ the true max score). So the
                                        // scan emits ub_b (the keep test) and anchors on the max lb_b (the score floor mlb).
            let rhov = k.load_f(rho, sbi); // ρ shares the (kvh·nblocks+b) layout with sc
            let w = k.fmul(csc, qnorm); // scale·‖q‖
            let w2 = k.fmul(w, rhov); // scale·‖q‖·ρ = the covering-radius slack
            let ub = k.fadd(t1, w2); // ⟪q,k⟫ ≤ ub (sound upper bound — the keep test)
            let ubrow = k.imul(h, cnb);
            let ubi = k.iadd(ubrow, b);
            k.store_f(ubo, ubi, ub);
            // Anchor mlb on the CENTROID score t1 = ⟪q,c⟫ = MEAN member score (inner product is
            // linear ⇒ ⟪q, mean(k)⟫ = mean⟪q,k⟫). mean ≤ max, so max_b t1_b ≤ the global max true
            // score — a VALID lower bound (`score_anchored_gather_keeps_argmax` holds for any such
            // mlb) yet far TIGHTER than the ball lower bound ⟪q,c⟫−‖q‖ρ (which floored θ so low the
            // cut kept every block). Tighter mlb ⇒ higher θ ⇒ real sparsity, needle still kept.
            let m0 = k.load_var_f(lub);
            let m1 = k.ext2(GLSL_FMAX, m0, t1);
            k.store_var(lub, m1);
        });
        let lubv = k.load_var_f(lub);
        k.shared_store(red, redp, lane, lubv);
        k.wg_barrier();
        k.tree_reduce(red, redp, WG, lane, true);
        let mlb = k.shared_load(red, redp, c0);
        let isz = k.e(OP_IEQ, k.bool_t, &[lane, c0]);
        k.iff(isz, |k| {
            k.store_f(pmax, hs, mlb); // pmax now carries the per-slice max lower bound
        });
    });
    k.finish(WG)
}

/// **DEBUG: exact per-block MAX score** — the bound-vs-truth instrument. One WG per head; lane owns
/// blocks `b = lane, lane+64, …`; `escore[h·nblocks+b]` = max over the block's valid keys of the
/// EXACT score `scale·sq·sk[s]·⟨q,K[s]⟩` (the same `dp4a_score` the flash folds). Placed next to the
/// selector's `ub` it shows whether the bound RANKS by score (`ub_rank_is_not_score_rank` on metal).
/// Bindings: 0=q 1=kcq(int8) 2=params 3=sk 4=escore(f32 nh·nblocks). Dispatch (nh,1,1) at WG.
pub fn escore_module(nh: u32, nkv: u32, hd: u32, tmax: u32, scale: f32, blk: u32) -> Vec<u32> {
    const WG: u32 = 64;
    let group = nh / nkv.max(1);
    let nblocks = tmax.div_ceil(blk);
    let hd4 = hd / 4;
    let mut k = Kb::new();
    k.dp4a = true;
    let q = k.ssbo_f(0);
    let kcq = k.ssbo_u(1);
    let params = k.ssbo_u(2);
    let sk = k.ssbo_f(3);
    let esc = k.ssbo_f(4);
    let (qq, qqp) = k.shared_u32(hd4);
    let (red, redp) = k.shared_f32(WG);
    let gid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let h = k.udiv(gid, cwg);
    let lane = k.umod(gid, cwg);
    let cnh = k.cu(nh);
    let guard = k.ult(h, cnh);
    k.iff(guard, |k| {
        let c0 = k.cu(0);
        let c1 = k.cu(1);
        let pos = k.load_u(params, c0);
        let len = k.iadd(pos, c1);
        let cg = k.cu(group);
        let kvh = k.udiv(h, cg);
        let chd = k.cu(hd);
        let chd4 = k.cu(hd4);
        let ct = k.cu(tmax);
        let cnb = k.cu(nblocks);
        let cblk = k.cu(blk);
        let qbase = k.imul(h, chd);
        k.absmax_reduce_global(q, qbase, red, redp, hd, WG, lane);
        let qamax = k.shared_load(red, redp, c0);
        let inv127 = k.cf(1.0 / 127.0);
        let sq = k.fmul(qamax, inv127);
        k.loop_step(lane, chd4, cwg, |k, w| {
            let c4 = k.cu(4);
            let w4 = k.imul(w, c4);
            let base4 = k.iadd(qbase, w4);
            let word = k.pack_int8_global(q, base4, sq);
            k.shared_store_u(qq, qqp, w, word);
        });
        k.wg_barrier();
        let kbase_w = k.imul(kvh, chd4);
        let skrow = k.imul(kvh, ct);
        let blkm1 = k.cu(blk - 1);
        let lp = k.iadd(len, blkm1);
        let nba = k.udiv(lp, cblk);
        k.loop_step(lane, nba, cwg, |k, b| {
            let bmax = k.fvar_f();
            let neg = k.cf(-1.0e30);
            k.store_var(bmax, neg);
            let s0 = k.imul(b, cblk);
            let s0e = k.iadd(s0, cblk);
            let over = k.ult(len, s0e);
            let s1 = k.e(OP_SELECT, k.uint, &[over, len, s0e]);
            k.loop_u(s0, s1, |k, s| {
                let fsum = k.dp4a_score(kcq, qq, qqp, kbase_w, s, ct, hd4);
                let skidx = k.iadd(skrow, s);
                let skv = k.load_f(sk, skidx);
                let csc = k.cf(scale);
                let t0 = k.fmul(csc, sq);
                let t1 = k.fmul(t0, skv);
                let sc = k.fmul(t1, fsum);
                let m0 = k.load_var_f(bmax);
                let m1 = k.ext2(GLSL_FMAX, m0, sc);
                k.store_var(bmax, m1);
            });
            let bm = k.load_var_f(bmax);
            let erow = k.imul(h, cnb);
            let ei = k.iadd(erow, b);
            k.store_f(esc, ei, bm);
        });
    });
    k.finish(WG)
}

/// **The SELECTOR, select stage — the three faces {sink, local, content} and the DERIVED cut.**
/// One WG per q-head. The keep set is the forced trinity (`SparseAttentionBudget.
/// sparse_attention_route_derived`, `Polymorphic.three_constructor_exhaustiveness`):
///   • **sink (NB)** — the first `n_sink` blocks (the softmax-mass drain, position 0), kept always;
///   • **local (B, the monad)** — the last `n_local` blocks (the residual/self/recent home,
///     `home(c)=c`, on neither side of the bound), kept always — never the bound's to judge;
///   • **content (B/U)** — the tail between them, judged by the DERIVED heavy cut.
/// The content budget is NOT a dialed band. It is the participation-ratio heavy cut
/// (`heavy_count_le_PR`, `budget_ceil_ppl_valid`): keep block `i` iff its softmax prob `p_i ≥
/// ∑ p_j²`, the heavy set whose count is `≤ PR = 1/∑p²` — read off the geometry, self-adjusting to
/// concentration (peaked head → few kept, diffuse → many; uniform → keep all, no false sparsity).
/// Realized as a closed-form SCORE threshold (`heavy_cut_is_score_threshold`): with `e_i =
/// exp(ub_i − m)`, `Se = Σe`, `Se2 = Σe²` over content, `p_i ≥ ∑p² ⟺ ub_i ≥ m + ln(Se2/Se)` — one
/// pass, no band, no top-k search, exactly the proven cut. `m` is any shift (θ is shift-invariant);
/// we reuse the scan's whole-cone max. `cut=false` keeps every content block ⇒ the list-fold is
/// byte-identical to dense (the equivalence gate). MEASURED (superseded band, 2026-07-08): pure
/// ub-ranking evicted the current-token block (`ub_rank_is_not_score_rank`) because the bound is a
/// content instrument reaching into the monad face; the local exemption is that face made explicit.
/// Bindings: 0=pmax 1=ub 2=params 3=kept(u32 nh·nblocks) 4=nkept(u32 nh). Dispatch (nh,1).
pub fn theta_select_module(
    nh: u32,
    tmax: u32,
    blk: u32,
    nsplit: u32,
    n_sink: u32,
    n_local: u32,
    lslack: f32,
    cut: bool,
    budget: u32,
    iters: u32,
) -> Vec<u32> {
    const WG: u32 = 64;
    // `budget` is the compact pool's slot count `P`. Sink and local ride UNCONDITIONALLY, so no
    // threshold can bring the selection under a budget smaller than their floor — asserted here
    // rather than discovered as a bisection that converges to a set the pool cannot hold.
    assert!(
        budget == 0 || budget >= n_sink + n_local,
        "theta_select budget {budget} is below the unconditional sink+local floor {} — no \
threshold can satisfy it",
        n_sink + n_local
    );
    let nblocks = tmax.div_ceil(blk);
    let chunk = nblocks.div_ceil(WG);
    let _ = nh; // dispatch is exactly (nh,1,1); h = wg index, no guard (barriers stay uniform)
    let mut k = Kb::new();
    let pmax = k.ssbo_f(0);
    let ubo = k.ssbo_f(1);
    let params = k.ssbo_u(2);
    let kept = k.ssbo_u(3);
    let nkept = k.ssbo_u(4);
    let (cnt, cntp) = k.shared_u32(WG);
    let (red, redp) = k.shared_f32(WG); // Se / Se2 reduction for the participation cut
    let gid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let h = k.udiv(gid, cwg);
    let lane = k.umod(gid, cwg);
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let _ = pmax; // the anchor is now the EXACT max read off ub (escore), not theta_scan's pmax
    let _ = nsplit;
    // nba = ⌈len/blk⌉; my contiguous chunk [lo,hi)
    let pos = k.load_u(params, c0);
    let len = k.iadd(pos, c1);
    let cblk = k.cu(blk);
    let blkm1 = k.cu(blk - 1);
    let lp = k.iadd(len, blkm1);
    let nba = k.udiv(lp, cblk);
    let cchunk = k.cu(chunk);
    let lo = k.imul(lane, cchunk);
    let hie = k.iadd(lo, cchunk);
    let hover = k.ult(nba, hie);
    let hi = k.e(OP_SELECT, k.uint, &[hover, nba, hie]);
    let cnb = k.cu(nblocks);
    let ubrow = k.imul(h, cnb);
    let csink = k.cu(n_sink);
    let clocal = k.cu(n_local);
    // **m = the EXACT global max score**, read off `ub` (which `escore` fills with the per-block
    // MAX key score, `quantized_glance_complete`): each lane maxes its chunk of `ub`, a tree-reduce
    // closes the whole-cone max. Unlike the centroid/ball BOUND (a q-independent block summary that
    // dilutes a lone needle and inflates uniformly — `ub_rank_is_not_score_rank`, the measured miss
    // AND the keep-all), the per-block max IS the block's true relevance, so its argmax is the score
    // argmax and the anchor is TIGHT.
    let mv = k.fvar_f();
    let neg = k.cf(-1.0e30);
    k.store_var(mv, neg);
    k.loop_u(lo, hi, |k, b| {
        let ubi = k.iadd(ubrow, b);
        let ub = k.load_f(ubo, ubi);
        let m0 = k.load_var_f(mv);
        let m1 = k.ext2(GLSL_FMAX, m0, ub);
        k.store_var(mv, m1);
    });
    let mmine = k.load_var_f(mv);
    k.shared_store(red, redp, lane, mmine);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, lane, true); // max reduction over lanes → whole-cone max
    let m = k.shared_load(red, redp, c0);
    k.wg_barrier();
    // **The cut is the participation ratio — over the EXACT scores (escore), where it is proven.**
    // `SparseAttentionBudget.heavy_cut_is_score_threshold` / `argmax_is_heavy` keep the heavy set
    // `p_b ≥ Σp²` = `ub_b ≥ m + ln(Se2/Se)` — but that guarantee is about the argmax of the FED
    // distribution. The bug was feeding it per-block BOUNDS, whose argmax ≠ the score argmax
    // (`ub_rank_is_not_score_rank`) — a lone needle's block bound is dominated by its neighbours, so
    // it was cut, and a looser bound cut it earlier (the box/ball inversion; the fixed-margin form
    // was non-adaptive and failed across depth). Fed `escore` (the block's TRUE max key score) the
    // argmax IS the needle block, so it is heavy ⇒ kept, ADAPTIVELY (peaked → few, diffuse → many,
    // `heavy_count_le_PR`). `m` is the exact whole-cone max (the exp shift; θ is shift-invariant).
    // `lslack` (DEC_LSLACK) is the additive keep-more margin. Sink/local ride unconditionally.
    let se = k.fvar_f();
    let se2 = k.fvar_f();
    let zf0 = k.cf(0.0);
    k.store_var(se, zf0);
    k.store_var(se2, zf0);
    k.loop_u(lo, hi, |k, b| {
        let sink = k.ult(b, csink);
        let bl = k.iadd(b, clocal);
        let local = k.e(OP_UGE, k.bool_t, &[bl, nba]);
        let s_or_l = k.e(OP_LOGICAL_OR, k.bool_t, &[sink, local]);
        let ubi = k.iadd(ubrow, b);
        let ub = k.load_f(ubo, ubi);
        let d = k.fsub(ub, m);
        let e = k.ext1(GLSL_EXP, d);
        let ce = k.e(OP_SELECT, k.float, &[s_or_l, zf0, e]); // content-only (sink/local → 0)
        let s0 = k.load_var_f(se);
        let s1 = k.fadd(s0, ce);
        k.store_var(se, s1);
        let ee = k.fmul(ce, ce);
        let q0 = k.load_var_f(se2);
        let q1 = k.fadd(q0, ee);
        k.store_var(se2, q1);
    });
    let sev = k.load_var_f(se);
    k.wg_barrier();
    k.shared_store(red, redp, lane, sev);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, lane, false);
    let se_tot = k.shared_load(red, redp, c0);
    k.wg_barrier();
    let se2v = k.load_var_f(se2);
    k.shared_store(red, redp, lane, se2v);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, lane, false);
    let se2_tot = k.shared_load(red, redp, c0);
    let tiny = k.cf(1.0e-30);
    let sed = k.ext2(GLSL_FMAX, se_tot, tiny);
    let ratio = k.fdiv(se2_tot, sed);
    let lnr = k.ext1(GLSL_LOG, ratio);
    let th0 = k.fadd(m, lnr); // m + ln(Se2/Se) — the participation heavy cut over escore
    let clsl = k.cf(lslack); // additive keep-more margin, 0 = pure derived cut
    let th = k.fsub(th0, clsl);
    let _ = len;
    // ── THE BUDGET: `|kept| ≤ P`, made a HARD bound rather than a hope ──────────────────────────
    // `ColdTierDecode.victims_suffice_iff_pool_covers_shortlist` is an IFF: a pool of `P` slots can
    // serve the selection **exactly when** `sel.card ≤ P`. The participation cut above has no such
    // bound — `heavy_count_le_PR` bounds the heavy set by the participation ratio, and PR is data
    // dependent, running to the whole content set on a flat distribution. So a compact pool smaller
    // than `nblocks` was unsound, and it was MEASURED unsound: gpugate 5/6 at the full pool, 0/6 at
    // 12 and at 8. That refusal is what pins the pool to the full block count, which at 1M is the
    // VRAM line that does not fit.
    //
    // The fix is a bisection on the threshold, not a different rule. `count(θ)` is monotone
    // non-increasing in θ; `count(th)` may exceed the budget and `count(m)` is exactly the
    // unconditional sink+local floor. Bisect on `[th, m]` keeping the invariant **`hi` always
    // satisfies the bound**, and return `hi`. Two consequences worth stating:
    //
    //   * the result is a SUBSET of the unbounded heavy set (θ' ≥ th), so the budget never ADDS a
    //     block the participation rule rejected — it only drops the lowest-scoring ones it kept;
    //   * it is a no-op by construction when the cut is already under budget (`hi` starts at `th`),
    //     so this cannot move a run that did not need it.
    //
    // Every lane runs every iteration and the loop count is a CONSTANT, because the barriers inside
    // `count_at` must stay workgroup-uniform. The cost is `iters` extra passes over the block list;
    // it is a dial (`DEC_KNOT_BUDGET_ITERS`) and it is measured rather than assumed.
    let th = if cut && budget > 0 {
        let count_at = move |k: &mut Kb, thv: u32| -> u32 {
            let cv = k.fvar_f();
            let zc = k.cf(0.0);
            k.store_var(cv, zc);
            k.loop_u(lo, hi, |k, b| {
                let sink = k.ult(b, csink);
                let bl = k.iadd(b, clocal);
                let local = k.e(OP_UGE, k.bool_t, &[bl, nba]);
                let s_or_l = k.e(OP_LOGICAL_OR, k.bool_t, &[sink, local]);
                let ubi = k.iadd(ubrow, b);
                let ub = k.load_f(ubo, ubi);
                let heavy = k.e(OP_FORD_GREATER_THAN, k.bool_t, &[ub, thv]);
                let kp = k.e(OP_LOGICAL_OR, k.bool_t, &[s_or_l, heavy]);
                let one = k.cf(1.0);
                let zz = k.cf(0.0);
                // counted in f32 and exact: a count is at most `nblocks`, far under 2^24, and it
                // rides the f32 `red` scratch the reductions above already own.
                let add = k.e(OP_SELECT, k.float, &[kp, one, zz]);
                let a0 = k.load_var_f(cv);
                let a1 = k.fadd(a0, add);
                k.store_var(cv, a1);
            });
            let mine = k.load_var_f(cv);
            k.wg_barrier();
            k.shared_store(red, redp, lane, mine);
            k.wg_barrier();
            k.tree_reduce(red, redp, WG, lane, false);
            let tot = k.shared_load(red, redp, c0);
            k.wg_barrier();
            tot
        };
        let bud = k.cf(budget as f32);
        let c_th = count_at(&mut k, th);
        let over = k.e(OP_FORD_GREATER_THAN, k.bool_t, &[c_th, bud]);
        // `hi` is the threshold known to satisfy the bound. Under budget already ⇒ hi = lo = th and
        // every iteration below re-tests `th` and leaves it, which is the no-op.
        let lov = k.fvar_f();
        let hiv = k.fvar_f();
        k.store_var(lov, th);
        let h0 = k.e(OP_SELECT, k.float, &[over, m, th]);
        k.store_var(hiv, h0);
        for _ in 0..iters.max(1) {
            let l0 = k.load_var_f(lov);
            let h0 = k.load_var_f(hiv);
            let sum = k.fadd(l0, h0);
            let half = k.cf(0.5);
            let mid = k.fmul(sum, half);
            let cm = count_at(&mut k, mid);
            let still = k.e(OP_FORD_GREATER_THAN, k.bool_t, &[cm, bud]);
            let l1 = k.e(OP_SELECT, k.float, &[still, mid, l0]);
            let h1 = k.e(OP_SELECT, k.float, &[still, h0, mid]);
            k.store_var(lov, l1);
            k.store_var(hiv, h1);
        }
        k.load_var_f(hiv)
    } else {
        th
    };
    // keep = sink ∨ local ∨ (content ∧ ub ≥ θ). By absorption (content = ¬(sink∨local)) this is
    // exactly (sink∨local) ∨ (ub ≥ θ). cut=false ⇒ keep every block ⇒ dense-equivalent.
    let keepp = move |k: &mut Kb, b: u32| -> u32 {
        let sink = k.ult(b, csink);
        let bl = k.iadd(b, clocal);
        let local = k.e(OP_UGE, k.bool_t, &[bl, nba]);
        let s_or_l = k.e(OP_LOGICAL_OR, k.bool_t, &[sink, local]);
        let heavy = if cut {
            let ubi = k.iadd(ubrow, b);
            let ub = k.load_f(ubo, ubi);
            k.e(OP_FORD_GREATER_THAN, k.bool_t, &[ub, th])
        } else {
            k.ult(c0, c1) // true: keep all content ⇒ dense
        };
        k.e(OP_LOGICAL_OR, k.bool_t, &[s_or_l, heavy])
    };
    // count my chunk's kept blocks
    let cntv = k.fvar_u();
    let z = k.cu(0);
    k.store_var(cntv, z);
    k.loop_u(lo, hi, |k, b| {
        let kp = keepp(k, b);
        k.iff(kp, |k| {
            let c0v = k.load_var_u(cntv);
            let one = k.cu(1);
            let c1v = k.iadd(c0v, one);
            k.store_var(cntv, c1v);
        });
    });
    let myc = k.load_var_u(cntv);
    k.shared_store_u(cnt, cntp, lane, myc);
    k.wg_barrier();
    // inclusive Hillis-Steele scan over cnt[0..WG] (barrier splits read from write — no hazard)
    let mut stride = 1;
    while stride < WG {
        let cs = k.cu(stride);
        let ge = k.e(OP_UGE, k.bool_t, &[lane, cs]);
        let lm = k.e(OP_ISUB, k.uint, &[lane, cs]);
        let idx = k.e(OP_SELECT, k.uint, &[ge, lm, lane]);
        let tv = k.shared_load_u(cnt, cntp, idx);
        let z2 = k.cu(0);
        let t = k.e(OP_SELECT, k.uint, &[ge, tv, z2]);
        k.wg_barrier();
        let cur = k.shared_load_u(cnt, cntp, lane);
        let nv = k.iadd(cur, t);
        k.shared_store_u(cnt, cntp, lane, nv);
        k.wg_barrier();
        stride *= 2;
    }
    // exclusive offset = inclusive − mine; write my chunk's kept blocks in order
    let inc = k.shared_load_u(cnt, cntp, lane);
    let off0 = k.e(OP_ISUB, k.uint, &[inc, myc]);
    let wv = k.fvar_u();
    k.store_var(wv, off0);
    k.loop_u(lo, hi, |k, b| {
        let kp = keepp(k, b);
        k.iff(kp, |k| {
            let w0 = k.load_var_u(wv);
            let ki = k.iadd(ubrow, w0); // kept row base = ub row base = h·nblocks
            k.store_u(kept, ki, b);
            let one = k.cu(1);
            let w1 = k.iadd(w0, one);
            k.store_var(wv, w1);
        });
    });
    let cwm1 = k.cu(WG - 1);
    let isl = k.e(OP_IEQ, k.bool_t, &[lane, cwm1]);
    k.iff(isl, |k| {
        let tot = k.shared_load_u(cnt, cntp, cwm1);
        k.store_u(nkept, h, tot);
    });
    k.finish(WG)
}

/// **Flash-attention partial (split-K) — the FOLD over the kept-block list, the degree-2 half.**
/// One WG per (head, slice) folds its slice of the kept list into an un-normalized partial
/// (o, m, l); `flash_combine` merges the `nsplit` slices in the log-sum-exp chart (`osFuse_is_
/// add`), exact by ℝ associativity (`osFuse_grouping_free` — any slicing of the list denotes the
/// same drag). ONE drive for dense and sparse: the fold walks list positions `j`, and
/// `blockAt(j)` is `kept[h·nblocks+j]` under the selector (`knot`) or the identity `j` (dense =
/// the degenerate selector that keeps every cone block). Slices split the LIST, not positions —
/// kept blocks distribute evenly across WGs however concentrated the relevance, and a dead block
/// is never touched (no per-block test in any pass; the old `ub>θ` re-check is gone with the θ/ub
/// bindings). Bindings: 0=q 1=K(int8) 2=V(f16) 3=partial_o 4=partial_m 5=partial_l 6=params 7=sk
/// [knot: 8=kept 9=nkept]. Dispatch `(nh·nsplit, 1, 1)` at LocalSize WG. `blk > 0` always.
#[allow(clippy::too_many_arguments)]
pub fn flash_partial_module(
    nh: u32,
    nkv: u32,
    hd: u32,
    tmax: u32,
    scale: f32,
    nsplit: u32,
    blk: u32,
    knot: bool,
    compact: bool,
    v8: bool,
    nb_list: u32,
) -> Vec<u32> {
    const WG: u32 = 64;
    assert!(
        knot || !compact,
        "compact addressing is meaningless without the kept list"
    );
    // The compact POOL's V plane is a byte-for-byte copy of the KV's (`pool_fill` moves it as
    // opaque u32 words), so an int8 V there needs a pool V-SCALE plane beside `psk` — which does
    // not exist. Refused by name rather than served against a scale that is not there: this is the
    // class of defect `[AC4-LOADER]` names, where the wrong plane still produces tokens.
    assert!(
        !(v8 && compact),
        "int8 V has no compact-pool form yet (pool_fill would need a psv plane beside psk)"
    );
    let group = nh / nkv.max(1);
    let nblocks = tmax.div_ceil(blk);
    // ── THE KEPT LIST'S ROW STRIDE IS NOT THIS KERNEL'S BLOCK COUNT ────────────────────────────
    // Under COMPACT the caller passes `tmax = ppos` (the pool's position count), because every
    // PLANE this kernel fetches is the pool's and the row pitch must be the pool's. But `kept` and
    // `slot` are not planes — they are the SELECTOR's output, written by `theta_select` and
    // `pool_alloc` at `h·nb_knot + j` with `nb_knot` the real context's block count. Deriving their
    // stride from `ppos` instead made head `h` read row `h·pblk`, so every head but 0 addressed
    // another head's list the moment `pblk ≠ nb_knot`.
    //
    // That is what "gpugate 5/6 at the full pool, 0/6 at 12 and at 8" was: at the full pool the two
    // strides COINCIDE and the bug is invisible. It was recorded as the participation cut being
    // unbudgeted — a real defect, and not this one. A wrong diagnosis that reproduces is the
    // expensive kind.
    //
    // `nb_list == 0` means "same as this kernel's own count", which is exactly the non-compact case
    // and keeps that path byte-identical.
    let nb_list = if nb_list == 0 { nblocks } else { nb_list };
    let slice_max = nblocks.div_ceil(nsplit) * blk; // worst-case keys a slice owns (kept ≤ all)
    let mut k = Kb::new();
    let q = k.ssbo_f(0);
    let kcq = k.ssbo_u(1); // int8 K cache (word-i-major, 4 codes/word) — the DP4A scoring lever
                           // `v8`: int8 V, position-major 4 codes/word, Vq[(kvh·tmax + pos)·(hd/4) + w], with its scale
                           // on binding 11. Otherwise f16 at hd per position.
    let vc = if v8 { k.ssbo_u(2) } else { k.ssbo_f16(2) };
    let po = k.ssbo_f(3);
    let pm = k.ssbo_f(4);
    let pl = k.ssbo_f(5);
    let params = k.ssbo_u(6);
    let sk = k.ssbo_f(7); // per-token K absmax scale sk[kvh·tmax + pos]
    let kept = knot.then(|| k.ssbo_u(8)); // the selector's compacted block list kept[h·nblocks+j]
    let nkeptb = knot.then(|| k.ssbo_u(9)); // list length nkept[h]
                                            // COMPACT: the pool slot holding kept[h·nblocks+j], same shape as `kept`. Written by the
                                            // caller's `KvPool::plan` (bedrock::serve::model::kvpool), which is what makes the slot a
                                            // free variable at all — without a pool the slot IS the list position and this is the
                                            // identity. Preloaded into shared beside the block ids below, for the reason the kept
                                            // preload gives: a per-key GLOBAL load measured 2x the dense per-key cost.
    let slotb = compact.then(|| k.ssbo_u(10));
    // COMPUTED, for the reason `fused_rope_kv_fold_module`'s twin gives: bindings are positional
    // and 8-10 are optional, so a fixed number mis-addresses every buffer with the knot off.
    let vsc_bind = if compact {
        11
    } else if knot {
        10
    } else {
        8
    };
    let vsc = v8.then(|| k.ssbo_f(vsc_bind)); // per-token V absmax scale vsc[kvh·tmax + pos]
    let (qq, qqp) = k.shared_u32(hd / 4); // the packed int8 query (DP4A vec1), built once per WG
    let kseg = knot.then(|| k.shared_u32(slice_max / blk + 2)); // the slice's kept range, preloaded
    let sseg = compact.then(|| k.shared_u32(slice_max / blk + 2)); // ...and its pool slots
    let (psh, pshp) = k.shared_f32(slice_max);
    let (red, redp) = k.shared_f32(WG);
    let gid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let hs = k.udiv(gid, cwg); // h·nsplit + split
    let lane = k.umod(gid, cwg);
    let cns = k.cu(nsplit);
    let h = k.udiv(hs, cns);
    let split = k.umod(hs, cns);
    let ctot = k.cu(nh * nsplit);
    let guard = k.ult(hs, ctot);
    k.iff(guard, |k| {
        let c0 = k.cu(0);
        let c1 = k.cu(1);
        let pos = k.load_u(params, c0);
        let len = k.iadd(pos, c1);
        let cg = k.cu(group);
        let kvh = k.udiv(h, cg);
        let chd = k.cu(hd);
        let ct = k.cu(tmax);
        let qbase = k.imul(h, chd);
        let vbase0 = k.imul(kvh, ct); // V row base: (kvh·Tmax + s)·hd + i (K base is word-i-major, below)
                                      // int8-quantize the query into shared packed codes qq + scale sq — the DP4A vec1, built
                                      // once per WG (the gemv's activation chart, reused). Score = scale·sq·sk[s]·⟨qq,Kq⟩.
        k.absmax_reduce_global(q, qbase, red, redp, hd, WG, lane);
        let qamax = k.shared_load(red, redp, c0);
        let inv127 = k.cf(1.0 / 127.0);
        let sq = k.fmul(qamax, inv127);
        let chd4 = k.cu(hd / 4);
        let kbase_w = k.imul(kvh, chd4); // K word-row base: (kvh·(hd/4)+w)·tmax + s
        k.loop_step(lane, chd4, cwg, |k, w| {
            let c4 = k.cu(4);
            let w4 = k.imul(w, c4);
            let base4 = k.iadd(qbase, w4);
            let word = k.pack_int8_global(q, base4, sq);
            k.shared_store_u(qq, qqp, w, word);
        });
        k.wg_barrier();
        // the kept VIRTUAL KEY STREAM: nk blocks × blk keys, sliced at KEY granularity. (Block-
        // granular slicing idled WGs whenever nk < nsplit — measured 3.5× vs the position-sliced
        // legacy at 2k.) Virtual key v ∈ [0, nk·blk): list index j = v/blk, block b = kept[j] (or
        // j dense), real key s = b·blk + v%blk, valid iff s < len (only the final cone block is
        // partial). psh slot = v − v0 — contiguous; invalid tail slots are never written nor read.
        let cblk = k.cu(blk);
        let cnb = k.cu(nblocks);
        let blkm1 = k.cu(blk - 1);
        let lp = k.iadd(len, blkm1);
        let nba = k.udiv(lp, cblk);
        // dense: the virtual stream IS the position stream — vtot = len, s = v, no validity test
        // (nba·blk would re-add a per-key guard the legacy drive never paid). knot: nk·blk.
        // NOTE (measured 2026-07-08): vtot_knot=nk·blk vs vtot_dense=len gives DIFFERENT slice
        // boundaries whenever ⌈nk·blk/nsplit⌉ ≠ ⌈len/nsplit⌉ → different f32 fold grouping →
        // ulp-level output drift, which the int8 requant chart amplifies chaotically across
        // layers (KNOTCMP keep-all rel dev ~1e0, argmax 15/16 — DEC_FLASHSPLIT=1 → 0.000e0
        // bit-identical). Benign (same class as the q4 carrier deviation), deterministic;
        // token-exactness is the MARGIN statement (SparseSelection.sparse_emits_same_token),
        // not bit-exactness. The pf=4096 0.000e0 of 2026-07-08 was a vlen coincidence (33=33).
        let vtot = if knot {
            let nk = k.load_u(nkeptb.unwrap(), h);
            let raw = k.imul(nk, cblk);
            // ── THE CONE BLOCK IS PARTIAL, AND COUNTING IT FULL IS WHAT MOVED THE BOUNDARY ──────
            // `nk·blk` counts the final block as `blk` keys when it holds only `len − (nba−1)·blk`.
            // With every block kept that makes vtot = nba·blk ≠ len, so `vlen = ⌈vtot/nsplit⌉`
            // differs from dense's and the split-K partials group different keys — the ~1e0 keep-all
            // deviation this file recorded on 2026-07-08 and called benign. It is not benign if the
            // cold read has to MATCH dense: subtracting the pad makes keep-all give vtot = len
            // EXACTLY, the boundaries coincide, and keep-all is bit-identical to dense at every
            // nsplit rather than only at nsplit=1.
            // `kept` is ascending, so the cone block is the last entry; block ids are `< nba`, so
            // `last ≥ nba−1` IS `last == nba−1` and needs no equality op.
            let one = k.cu(1);
            let nkm1 = k.e(OP_ISUB, k.uint, &[nk, one]);
            let cnbl0 = k.cu(nb_list);
            let krow0 = k.imul(h, cnbl0);
            let lastix = k.iadd(krow0, nkm1);
            let lastb = k.load_u(kept.unwrap(), lastix);
            let nbam1 = k.e(OP_ISUB, k.uint, &[nba, one]);
            let has_cone = k.e(OP_UGE, k.bool_t, &[lastb, nbam1]);
            let full = k.imul(nba, cblk);
            let pad = k.e(OP_ISUB, k.uint, &[full, len]);
            let zeru = k.cu(0);
            let padsel = k.e(OP_SELECT, k.uint, &[has_cone, pad, zeru]);
            k.e(OP_ISUB, k.uint, &[raw, padsel])
        } else {
            let _ = nba;
            len
        };
        let nsm1 = k.cu(nsplit - 1);
        let vnum = k.iadd(vtot, nsm1);
        let vlen = k.udiv(vnum, cns);
        let v0 = k.imul(split, vlen);
        let v0e = k.iadd(v0, vlen);
        let vover = k.ult(vtot, v0e);
        let v1 = k.e(OP_SELECT, k.uint, &[vover, vtot, v0e]);
        let cnbl = k.cu(nb_list);
        let krow = k.imul(h, cnbl); // the SELECTOR's stride, not this kernel's — see `nb_list`
                                    // EMPTY slices (v0 ≥ vtot: nsplit finer than the fold — a sparse cut folds only nkept blocks,
                                    // so high nsplit leaves trailing slices with no keys) trap the driver if they run the drive.
                                    // MEASURED (2026-07-08, qwen q4 sparse cut nk=3): a SINGLE empty slice → DEVICE_LOST (fs=96
                                    // no-empty clean, fs=97 one-empty crashes) — the degenerate v0≥vtot execution traps below the
                                    // SPIR-V (passes are provably empty-loop no-ops on paper). osFuse_grouping_free makes an empty
                                    // slice a clean no-op: write the neutral partial (m=−1e30 ⇒ combine weights it 0, l=0, o=0)
                                    // and SKIP preload+passes. v0 is WG-uniform ⇒ the guarded wg_barriers stay uniform (a whole
                                    // WG enters `haswork` or a whole WG takes the neutral branch).
        let haswork = k.ult(v0, vtot);
        let mt = k.e(OP_UGE, k.bool_t, &[v0, vtot]);
        k.iff(mt, |k| {
            k.loop_step(lane, chd, cwg, |k, i| {
                let ob = k.imul(hs, chd);
                let oidx = k.iadd(ob, i);
                let z = k.cf(0.0);
                k.store_f(po, oidx, z);
            });
            let is0e = k.e(OP_IEQ, k.bool_t, &[lane, c0]);
            k.iff(is0e, |k| {
                let ng = k.cf(-1.0e30);
                let z = k.cf(0.0);
                k.store_f(pm, hs, ng);
                k.store_f(pl, hs, z);
            });
        });
        k.iff(haswork, |k| {
            // knot: preload the slice's kept range [v0/blk, ⌈v1/blk⌉] into shared — a per-key GLOBAL
            // kept load put a memory dependency in front of every score's address chain (measured 2×
            // the dense per-key cost); from shared it is off the critical path.
            let klo = k.udiv(v0, cblk);
            if knot {
                let v1p = k.iadd(v1, blkm1);
                let khi = k.udiv(v1p, cblk);
                let nkw = k.e(OP_ISUB, k.uint, &[khi, klo]);
                k.loop_step(lane, nkw, cwg, |k, w| {
                    let jw = k.iadd(klo, w);
                    let ki = k.iadd(krow, jw);
                    let b = k.load_u(kept.unwrap(), ki);
                    let (ksv, ksp) = kseg.unwrap();
                    k.shared_store_u(ksv, ksp, w, b);
                    if compact {
                        let sl = k.load_u(slotb.unwrap(), ki);
                        let (ssv, ssp) = sseg.unwrap();
                        k.shared_store_u(ssv, ssp, w, sl);
                    }
                });
                k.wg_barrier();
            }
            // virtual key v → (clamped key s, psh slot, valid). Dense: s = v (identity — the legacy
            // drive, always valid). Knot: via the preloaded shared kept segment; an overhanging key
            // (final partial block / synthetic fill) is CLAMPED to `pos` (a safe address) and its
            // score poisoned to −1e30 in pass 1 — exp gives exact 0, so passes stay BRANCHLESS
            // (a guard `iff` around the score loop measured ~4× per-key: it wrecks the schedule).
            let key_of = move |k: &mut Kb, v: u32| -> (u32, u32, Option<u32>) {
                let soff = k.e(OP_ISUB, k.uint, &[v, v0]);
                if knot {
                    let j = k.udiv(v, cblk);
                    let jrel = k.e(OP_ISUB, k.uint, &[j, klo]);
                    let (ks, ksp) = kseg.unwrap();
                    let b = k.shared_load_u(ks, ksp, jrel);
                    let off = k.umod(v, cblk);
                    let bb = k.imul(b, cblk);
                    let s = k.iadd(bb, off);
                    let valid = k.ult(s, len);
                    // ── COMPACT: the KV address is the LIST slot, not the block id ───────────────
                    // This is what lets the device hold `nkept·blk` positions instead of `tmax`.
                    // With a scattered upload writing kept block `j` into staging slot `j`, key
                    // `v` lives at `j·blk + off`, so the resident K/V is O(budget) and flat in
                    // context — the 104 GiB of KV an 8B needs at 1M becomes ~113 MB of hot window.
                    // `s` is still the REAL position and is still what decides validity (`s < len`)
                    // and what the caller's bookkeeping means; only the fetch address moves.
                    // compact=false ⇒ `addr = s`, byte-identical to the pre-compact kernel (D3).
                    //
                    // The slot is NOT the list position `j`. It was, while the only filler
                    // re-uploaded every kept block every step and could therefore choose to write
                    // block j at j. A delta filler cannot: a block that survives from the previous
                    // step keeps the slot it already occupies, wherever the selection has since
                    // moved it in the list. So the caller supplies the mapping and the fetch reads
                    // it — `ColdTierDecode.delta_serve_is_exact` is what says the bytes there are
                    // still the right bytes.
                    let addr = if compact {
                        let (ssv, ssp) = sseg.unwrap();
                        let sl = k.shared_load_u(ssv, ssp, jrel);
                        let jb = k.imul(sl, cblk);
                        k.iadd(jb, off)
                    } else {
                        s
                    };
                    // The invalid-key clamp must land INSIDE the buffer being addressed. Under
                    // compact that buffer is the pool, whose position count is far below `pos`,
                    // so clamping to `pos` would read out of bounds. Slot 0 is always in range and
                    // the score is poisoned to -1e30 either way.
                    let safe = if compact { k.cu(0) } else { pos };
                    let sc = k.e(OP_SELECT, k.uint, &[valid, addr, safe]);
                    (sc, soff, Some(valid))
                } else {
                    (v, soff, None)
                }
            };
            // pass 1: score → psh[v−v0], local max (valid keys only)
            let lmax = k.fvar_f();
            let neg = k.cf(-1.0e30);
            k.store_var(lmax, neg);
            let v0l = k.iadd(v0, lane);
            k.loop_step(v0l, v1, cwg, |k, v| {
                let (s, soff, valid) = key_of(k, v);
                // the score dot in the DP4A chart: ⟨qq, Kq[·,s]⟩ integer, ×scale·sq·sk[s] (4×
                // throughput + K one byte not two). `fold_in_the_seed_chart` on the SCORE.
                let fsum = k.dp4a_score(kcq, qq, qqp, kbase_w, s, ct, hd / 4);
                let skrow = k.imul(kvh, ct);
                let skidx = k.iadd(skrow, s);
                let skv = k.load_f(sk, skidx);
                let csc = k.cf(scale);
                let t0 = k.fmul(csc, sq);
                let t1 = k.fmul(t0, skv);
                let sc0 = k.fmul(t1, fsum);
                let sc = if let Some(vd) = valid {
                    let neg9 = k.cf(-1.0e30);
                    k.e(OP_SELECT, k.float, &[vd, sc0, neg9])
                } else {
                    sc0
                };
                k.shared_store(psh, pshp, soff, sc);
                let m0 = k.load_var_f(lmax);
                let m1 = k.ext2(GLSL_FMAX, m0, sc);
                k.store_var(lmax, m1);
            });
            let lm = k.load_var_f(lmax);
            k.shared_store(red, redp, lane, lm);
            k.wg_barrier();
            k.tree_reduce(red, redp, WG, lane, true);
            let m = k.shared_load(red, redp, c0);
            // pass 2: exp(score − m) into psh, local sum — the same virtual drive
            let lsum = k.fvar_f();
            let z2 = k.cf(0.0);
            k.store_var(lsum, z2);
            k.loop_step(v0l, v1, cwg, |k, v| {
                let (_s, soff, valid) = key_of(k, v);
                let sv = k.shared_load(psh, pshp, soff);
                let dd = k.fsub(sv, m);
                let e0 = k.ext1(GLSL_EXP, dd);
                // FORCE poison → 0, not via underflow: an ALL-poison slice has m = −1e30 = the poison
                // value, so exp(psh − m) = exp(0) = 1 (the underflow assumption fails). The SELECT is
                // branchless and unconditional-correct.
                let e = if let Some(vd) = valid {
                    let zf = k.cf(0.0);
                    k.e(OP_SELECT, k.float, &[vd, e0, zf])
                } else {
                    e0
                };
                // ── int8 V: the per-POSITION scale rides in `psh`, not in the pass-3 inner loop ──
                // `Σ_s p_s·(v̂_{s,i}·λ_s) = Σ_s (p_s·λ_s)·v̂_{s,i}` — the scale depends on the key,
                // not the dim, so folding it here costs ONE multiply per key and folding it in
                // pass 3 would cost one per (key, dim) plus a global load per element. Pass 3 then
                // reads raw codes and multiplies by nothing.
                //
                // Reassociation, and it is not free: the products are formed as `(p·λ)·v̂` rather
                // than `p·(v̂·λ)`. The ADDITION order into each accumulator is untouched — which is
                // the invariant `tier.rs` keeps and the one that matters — but the roundings differ,
                // so this is not bit-identical to the f16 path and is not claimed to be.
                let stored = match vsc {
                    // `e0`, not `e`: an invalid key's scale may be uninitialised, and `0·NaN` is
                    // NaN. The SELECT below zeroes the product outright, which a multiply cannot.
                    Some(vsc) => {
                        let si = k.iadd(vbase0, _s);
                        let lam = k.load_f(vsc, si);
                        let p0 = k.fmul(e0, lam);
                        if let Some(vd) = valid {
                            let zf = k.cf(0.0);
                            k.e(OP_SELECT, k.float, &[vd, p0, zf])
                        } else {
                            p0
                        }
                    }
                    None => e,
                };
                k.shared_store(psh, pshp, soff, stored);
                // the DENOMINATOR is the unscaled Σ exp — `λ` belongs to V, not to the softmax
                let s0v = k.load_var_f(lsum);
                let s1v = k.fadd(s0v, e);
                k.store_var(lsum, s1v);
            });
            let ls = k.load_var_f(lsum);
            k.shared_store(red, redp, lane, ls);
            k.wg_barrier();
            k.tree_reduce(red, redp, WG, lane, false);
            let d = k.shared_load(red, redp, c0);
            // pass 3: o_i = Σ_s psh·V[s,i] (UN-normalized). Dim per lane, serial over the slice's keys.
            k.loop_step(lane, chd, cwg, |k, i| {
                let acc = k.fvar_f();
                let z = k.cf(0.0);
                k.store_var(acc, z);
                k.loop_u(v0, v1, |k, v| {
                    let (s, soff, valid) = key_of(k, v);
                    let p = k.shared_load(psh, pshp, soff); // 0 for poisoned slots (pass 2)
                    let vi1 = k.iadd(vbase0, s);
                    // int8: Vq[(kvh·tmax + s)·(hd/4) + i/4], byte i%4. The scale is already in `p`.
                    let vv0 = if vsc.is_some() {
                        let c4v = k.cu(4);
                        let chd4v = k.cu(hd / 4);
                        let vr = k.imul(vi1, chd4v);
                        let w = k.udiv(i, c4v);
                        let widx = k.iadd(vr, w);
                        let word = k.load_u(vc, widx);
                        let sub = k.umod(i, c4v);
                        k.int8_lane_f(word, sub)
                    } else {
                        let vi2 = k.imul(vi1, chd);
                        let vidx = k.iadd(vi2, i);
                        k.load_f16(vc, vidx)
                    };
                    // poison V is UNINITIALIZED (may be Inf/NaN); p=0 but 0·Inf=NaN, so zero the V too.
                    let vv = if let Some(vd) = valid {
                        let zf = k.cf(0.0);
                        k.e(OP_SELECT, k.float, &[vd, vv0, zf])
                    } else {
                        vv0
                    };
                    let pr = k.fmul(p, vv);
                    let a0 = k.load_var_f(acc);
                    let a1 = k.fadd(a0, pr);
                    k.store_var(acc, a1);
                });
                let af = k.load_var_f(acc);
                let ob = k.imul(hs, chd);
                let oidx = k.iadd(ob, i);
                k.store_f(po, oidx, af);
            });
            // lane 0 writes this slice's m and l
            let is0 = k.e(OP_IEQ, k.bool_t, &[lane, c0]);
            k.iff(is0, |k| {
                k.store_f(pm, hs, m);
                k.store_f(pl, hs, d);
            });
        }); // end iff(haswork) — empty slices skipped the drive above
    });
    k.finish(WG)
}

/// **Flash-attention split-K — COMBINE the per-slice partials.** One WG per head merges the
/// `nsplit` partials in the log-sum-exp chart (`osFuse_is_add`): `gm = max_split m`, then
/// `out_i = Σ_split e^{m_split−gm}·o_i^{split} / Σ_split e^{m_split−gm}·l_split`. Bindings:
/// 0=partial_o 1=partial_m 2=partial_l 3=out. Dispatch `(nh, 1, 1)` at LocalSize `hd`.
pub fn flash_combine_module(nh: u32, hd: u32, nsplit: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let po = k.ssbo_f(0);
    let pm = k.ssbo_f(1);
    let pl = k.ssbo_f(2);
    let out = k.ssbo_f(3);
    let chd = k.cu(hd);
    let gid = k.gid_comp(0);
    let h = k.udiv(gid, chd);
    let i = k.umod(gid, chd);
    let cnh = k.cu(nh);
    let guard = k.ult(h, cnh);
    k.iff(guard, |k| {
        let cns = k.cu(nsplit);
        let hbase = k.imul(h, cns); // h·nsplit
                                    // global max over splits
        let gm = k.fvar_f();
        let neg = k.cf(-1.0e30);
        k.store_var(gm, neg);
        let c0 = k.cu(0);
        let c1 = k.cu(1);
        k.loop_u(c0, cns, |k, sp| {
            let hs = k.iadd(hbase, sp);
            let mv = k.load_f(pm, hs);
            let g0 = k.load_var_f(gm);
            let g1 = k.ext2(GLSL_FMAX, g0, mv);
            k.store_var(gm, g1);
        });
        let g = k.load_var_f(gm);
        // O = Σ w·o_i ; L = Σ w·l ; w = exp(m_split − gm)
        let onum = k.fvar_f();
        let lden = k.fvar_f();
        let z = k.cf(0.0);
        k.store_var(onum, z);
        k.store_var(lden, z);
        k.loop_u(c0, cns, |k, sp| {
            let hs = k.iadd(hbase, sp);
            let mv = k.load_f(pm, hs);
            let dd = k.fsub(mv, g);
            let w = k.ext1(GLSL_EXP, dd);
            let lv = k.load_f(pl, hs);
            let wl = k.fmul(w, lv);
            let ld0 = k.load_var_f(lden);
            let ld1 = k.fadd(ld0, wl);
            k.store_var(lden, ld1);
            let ob = k.imul(hs, chd);
            let oi = k.iadd(ob, i);
            let ov = k.load_f(po, oi);
            let wo = k.fmul(w, ov);
            let on0 = k.load_var_f(onum);
            let on1 = k.fadd(on0, wo);
            k.store_var(onum, on1);
        });
        let on = k.load_var_f(onum);
        let ld = k.load_var_f(lden);
        let res = k.fdiv(on, ld);
        let ob = k.imul(h, chd);
        let oidx = k.iadd(ob, i);
        k.store_f(out, oidx, res);
        let _ = c1;
    });
    k.finish(hd)
}
