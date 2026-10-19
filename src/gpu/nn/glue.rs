//! simt_nn::glue — the small per-token glue kernels and the byte-emitting face: the embed
//! gather, the fused RoPE + KV-append fold, SiLU·mul, the residual add, QK-norm, the argmax
//! split/finish and the commit-advance, followed by the `_bytes` wrappers the runtime loads.
//! Cut verbatim from `simt_nn.rs`.

use super::*;

impl Kb {
    /// `max(|v|, 1e-12)` tree-reduced into `red[0]` (Workgroup memory), the absmax of a value
    /// already held in the register `v`. `absmax_reduce_global` walks a strided slice of an SSBO;
    /// here `n == wg == hd`, so each lane folds exactly its own element and the two are the same
    /// reduce — this one re-reads no storage buffer, which is the point.
    fn absmax_reduce_reg(&mut self, v: u32, red: u32, redp: u32, wg: u32, tid: u32) {
        let f = self.float;
        let g = self.glsl;
        let av = self.e(OP_EXT_INST, f, &[g, GLSL_FABS, v]);
        let tiny = self.cf_bits(0x2b8c_bccc);
        let am = self.ext2(GLSL_FMAX, av, tiny);
        self.shared_store(red, redp, tid, am);
        self.wg_barrier();
        self.tree_reduce(red, redp, wg, tid, true);
    }

    /// Pack 4 f32 values `buf[base..base+4]` from a WORKGROUP array into one int8 word (4 codes)
    /// with `scale` — `pack_int8_global`'s twin, reading workgroup memory instead of an SSBO.
    fn pack_int8_shared(&mut self, buf: u32, bufp: u32, base: u32, scale: u32) -> u32 {
        let word = self.fvar_u();
        let z = self.cu(0);
        self.store_var(word, z);
        let c0 = self.cu(0);
        let four = self.cu(4);
        self.loop_u(c0, four, |mut k, b| {
            let idx = k.iadd(base, b);
            let v = k.shared_load(buf, bufp, idx);
            let byte = k.int8_code(v, scale);
            let u = k.uint;
            let c8 = k.cu(8);
            let sh = k.imul(b, c8);
            let shifted = k.e(OP_SHIFT_LEFT_LOGICAL, u, &[byte, sh]);
            let w0 = k.load_var_u(word);
            let w1 = k.e(OP_BITWISE_OR, u, &[w0, shifted]);
            k.store_var(word, w1);
            k
        });
        self.load_var_u(word)
    }
}

// ═══════════════ the kernels ═══════════════

/// **Embed gather (f16 store)** — `x[i] = E[i·(vocab/2) + tok/2].lane(tok%2)`; E is the SAME
/// f16 col-major pack the lm_head GEMV reads (the tied weight stored once, two access patterns).
/// Bindings: 0 = E (u32 words), 1 = params (`params[1]` = token id), 2 = x out. Dispatch ≥ hidden.
pub fn embed_f16_module(hidden: u32, vocab: u32, local: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let e = k.ssbo_u(0);
    let params = k.ssbo_u(1);
    let x = k.ssbo_f(2);
    let i = k.gid_comp(0);
    let ch = k.cu(hidden);
    let guard = k.ult(i, ch);
    k.iff(guard, |mut k| {
        let c1 = k.cu(1);
        let tok = k.load_u(params, c1);
        let cv2 = k.cu(vocab / 2);
        let c2 = k.cu(2);
        let t2 = k.udiv(tok, c2);
        let widx1 = k.imul(i, cv2);
        let widx = k.iadd(widx1, t2);
        let word = k.load_u(e, widx);
        let c2b = k.cu(2);
        let lane = k.umod(tok, c2b);
        let unp = {
            let (v2, g) = (k.vec2f, k.glsl);
            k.e(OP_EXT_INST, v2, &[g, GLSL_UNPACK_HALF2X16, word])
        };
        let f = k.float;
        let val = k.e(OP_VECTOR_EXTRACT_DYNAMIC, f, &[unp, lane]);
        k.store_f(x, i, val);
        k
    });
    k.finish(local)
}

/// **Fused RoPE + KV-append — one workgroup per head, the intra-WG hand-off.** Rotates q/k in place
/// (`rope`) then the k-head workgroups append the roped k and the (unroped) v to the f16 caches — the
/// rope writes are read by the SAME workgroup, so a cheap workgroup barrier replaces the pipeline
/// drain (no cross-workgroup hand-off). When `blk>0`, each k-head also folds its dim into the
/// per-block key box (max_i,min_i, f16 truth) AND re-packs the current block's box into the int8
/// selector chart (two SDot planes, word-i-major by block, + per-block scale) — old blocks are
/// immutable (their keys never change), so only the block being built is re-packed per append.
/// The int8 pack widens the box's f16 epsilon slightly (ub stays a bound up to quant ε; the
/// keep_band + quality gate absorb it — same seam as the K cache's own int8 chart).
/// Dispatch `(nh+nkv, 1, 1)` at LocalSize `hd`. Bindings: 0 = qkv (rw), 1 = K cache, 2 = V cache,
/// 3 = params (`params[0]`=pos), 4 = sk, blk>0: 5 = box f16, 6 = boxq (u32 planes), 7 = sb (f32).
pub fn fused_rope_kv_module(
    nh: u32,
    nkv: u32,
    hd: u32,
    tmax: u32,
    theta: f32,
    blk: u32,
    rope_dims: u32,
    v8: bool,
) -> Vec<u32> {
    fused_rope_kv_fold_module(nh, nkv, hd, tmax, theta, blk, 1, false, rope_dims, v8)
}

/// **`[DEC-FUSE]` — RoPE + KV-append with the QKV projection's SPLIT-K FOLD and its BIAS pulled into
/// the prologue.** Three dispatches collapse to one: the qkv GEMV's stage-2 reduce, the bias add, and
/// this kernel. Each costs a full `vkCmdPipelineBarrier` between compute dispatches, MEASURED on this
/// box at **3.71 µs of pure pipeline drain** (268 barriers = 995 µs of a 1533 µs token; `DEC_BAR=9`
/// removes them and the same 268 dispatches run in 398 µs). Dispatch count is therefore the decode
/// currency, and this is the cheapest 2 of them per layer to spend.
///
/// **Why it is safe without a global barrier — the mandala mask** (`MandalaMemory.access_in_region`).
/// The workgroup for head-slot `s` folds exactly the qkv region it will then rotate and append:
/// `base + lane` for its q-or-k head, plus `qn + kn + kvh·hd + lane` for its v head. Every index is
/// `base + (lane masked into the head)` — the ring closes inside the workgroup, so no WG reads a slot
/// another WG folded, and `wg_barrier` (cheap, intra-SM) replaces the drain. The v region is covered
/// exactly once, by the k-head workgroups, so the `nh+nkv` slots tile all `qn+2·kn` outputs.
///
/// `segs` = the qkv GEMV's split-K segment count (stage-1 partials at `src[s·m + i]`, the identical
/// layout and the identical sequential summation order the dropped stage-2 gemv used, so the result
/// is bit-identical — see the residual fold in `gemv_norm_module`). `bias` adds binding 6.
/// Extra bindings (only when folding/biasing, and only valid at `blk == 0`):
/// 5 = qkv stage-1 partials (f32, `segs·(qn+2·kn)`), 6 = qkv bias (f32, `qn+2·kn`).
///
/// `v8` — **the V plane is int8 with a per-(kv-head, position) scale, exactly as K already is.**
///
/// It was the only f16 plane in the KV, and at long context that is not a rounding decision, it is
/// the footprint: `vc = kn·tmax·2` against `kcq = kn·tmax`, so **V is 66 % of the KV**, and on
/// Qwen3.8-27B at 1M positions that is 51.81 GB against 34.90 (`knot_preflight`). The format is not
/// new — `CpuState` has stored `vc: Vec<i8>` + `vs: Vec<f32>` since it was written, the disk tier's
/// record is `block·(2·hd + 8)` for exactly this reason, and `GpuPrefillStack` carries a `vs` plane.
/// The DECODE KV was the one place that widened it, and `seed_kv` had to dequantize-and-rewiden
/// every element to feed it.
///
/// Binding 9 is the V scale plane when `v8` (5-8 are the knot's, 5-6 the fold/bias prologue's, and
/// those two are already mutually exclusive by assert).
#[allow(clippy::too_many_arguments)]
pub fn fused_rope_kv_fold_module(
    nh: u32,
    nkv: u32,
    hd: u32,
    tmax: u32,
    theta: f32,
    blk: u32,
    segs: u32,
    bias: bool,
    rope_dims: u32,
    v8: bool,
) -> Vec<u32> {
    // **PARTIAL rotary.** Qwen3.5 sets `partial_rotary_factor: 0.25` — 64 of a 256-wide head are
    // rotated and the tail is left ALONE (`rope_partial`, hybrid.rs:62). This kernel rotated the
    // whole head, which is not a wrong angle, it is a rotation applied to 192 dimensions that
    // carry no positional structure. `rope_dims == hd` is the full-rotation case every dense model
    // takes, and is byte-identical to what this emitted before the parameter existed.
    assert!(
        rope_dims <= hd && rope_dims % 2 == 0,
        "rope_dims {rope_dims} must be even and at most hd {hd}"
    );
    let d2 = rope_dims / 2;
    let qn = nh * hd;
    let kn = nkv * hd;
    let qkvm = qn + 2 * kn;
    let prologue = segs > 1 || bias;
    assert!(
        !(prologue && blk > 0),
        "the qkv fold/bias prologue reuses bindings 5-6 (the knot's)"
    );
    let nblocks = if blk > 0 { tmax.div_ceil(blk) } else { 0 };
    let mut k = Kb::new();
    let qkv = k.ssbo_f(0);
    let kcq = k.ssbo_u(1); // int8 K cache, word-i-major: Kq[(kvh·(hd/4)+w)·tmax + pos], 4 codes/word
                           // V cache. `v8`: int8, position-major 4 codes/word —
                           // Vq[(kvh·tmax + pos)·(hd/4) + w]. Otherwise f16 at hd per position.
    let vc = if v8 { k.ssbo_u(2) } else { k.ssbo_f16(2) };
    let params = k.ssbo_u(3);
    let sk = k.ssbo_f(4); // per-token K absmax scale sk[kvh·tmax + pos]
                          // SparseSelection KNOT = (centroid c, covering radius ρ) per block — the ball bound
                          // `⟪q,k⟫ ≤ ⟪q,c⟫ + ‖q‖ρ` (`dot_ball_bound`), tighter than the axis box by ~√hd (L2 vs L1).
    let bx = (blk > 0).then(|| k.ssbo_f16(5)); // centroid c: bx[(kvh·hd+i)·nblocks+b] (dim-major-by-block)
    let bxq = (blk > 0).then(|| k.ssbo_u(6)); // int8 centroid plane: bxq[(kvh·(hd/4)+w)·nblocks+b]
    let sbb = (blk > 0).then(|| k.ssbo_f(7)); // per-block centroid scale sc[kvh·nblocks+b]
    let rho = (blk > 0).then(|| k.ssbo_f(8)); // per-block covering radius ρ[kvh·nblocks+b]
    let qkvp = prologue.then(|| k.ssbo_f(5)); // qkv stage-1 split-K partials, src[s·qkvm + i]
    let qkvb = bias.then(|| k.ssbo_f(6)); // qkv bias
                                          // **The binding index is COMPUTED, because the slots below it are optional.** Descriptors are
                                          // bound positionally, so a fixed number here would silently mis-address every buffer whenever
                                          // the knot or the prologue was off — the class of defect that emits tokens rather than failing.
                                          // Order: 0-4 always · 5-8 the knot · 5 the fold prologue · 6 its bias · then this.
    let vsc_bind = if blk > 0 {
        9
    } else if bias {
        7
    } else if prologue {
        6
    } else {
        5
    };
    let vsc = v8.then(|| k.ssbo_f(vsc_bind)); // per-token V absmax scale vsc[kvh·tmax + pos]
    let (red, redp) = k.shared_f32(hd); // the per-token absmax tree-reduce scratch

    // The roped head values, read back by the int8 pack. Every leader→lane hand-off in this kernel
    // travels through WORKGROUP memory, never the qkv storage buffer: under the GLSL450 memory
    // model a barrier whose semantics carry only WorkgroupMemory does not order one invocation's
    // SSBO store against another's load — the landed decode RoPE/KV race. `qkv_rope_batch_module`
    // is this hand-off's fixed twin; the pattern is copied from there, not invented.
    let (sval, svalp) = k.shared_f32(hd);
    let chd = k.cu(hd);
    let gid = k.gid_comp(0);
    let slot = k.udiv(gid, chd); // one WG per head-slot (q heads 0..nh, k heads nh..nh+nkv)
    let lane = k.umod(gid, chd);
    let ntot = k.cu(nh + nkv);
    let guard = k.ult(slot, ntot);
    k.iff(guard, |mut k| {
        let c0 = k.cu(0);
        let pos = k.load_u(params, c0);
        let cnh = k.cu(nh);
        let is_k = k.e(OP_UGE, k.bool_t, &[slot, cnh]);
        // base offset of this head in qkv: q head → slot·hd; k head → qn + (slot−nh)·hd
        let qbase = k.imul(slot, chd);
        let kvh = k.e(OP_ISUB, k.uint, &[slot, cnh]);
        let cqn = k.cu(qn);
        let kvbase0 = k.imul(kvh, chd);
        let kbase = k.iadd(cqn, kvbase0);
        let f = k.uint;
        let base = k.e(OP_SELECT, f, &[is_k, kbase, qbase]);
        // The lane's head value lives in the register `vf` from here to the single qkv store; the
        // (unroped) V value in `vv`. Each cross-lane read below takes the register or a workgroup
        // array — none re-reads this invocation's SSBO word.
        let vf = k.fvar_f();
        let vv = k.fvar_f();
        let ckn = k.cu(kn);
        let vbase = k.iadd(cqn, ckn); // V region: qn + kn + kvh·hd + lane

        // ── the folded/biased prologue: this WG materializes its OWN qkv region, nothing else's ──
        if let Some(src) = qkvp {
            let iq = k.iadd(base, lane);
            k.fold_bias_store(src, qkvb, qkv, iq, qkvm, segs);
            // this lane's OWN folded word, read back before any barrier: a same-invocation
            // read-after-write is ordered by program order alone, so it carries into `vf`.
            let xq = k.load_f(qkv, iq);
            k.store_var(vf, xq);
            k.iff(is_k, |mut k| {
                let v1 = k.iadd(vbase, kvbase0);
                let vi = k.iadd(v1, lane);
                k.fold_bias_store(src, qkvb, qkv, vi, qkvm, segs);
                let xv = k.load_f(qkv, vi);
                k.store_var(vv, xv);
                k
            });
            k.wg_barrier();
        } else {
            let i0 = k.iadd(base, lane);
            let xq = k.load_f(qkv, i0);
            k.store_var(vf, xq);
            k.iff(is_k, |mut k| {
                let v1 = k.iadd(vbase, kvbase0);
                let vsrc = k.iadd(v1, lane);
                let xv = k.load_f(qkv, vsrc);
                k.store_var(vv, xv);
                k
            });
        }
        // ── RoPE, pair lanes; partners exchange through WORKGROUP memory (`red`), never the qkv
        //    storage buffer. Only `lane < rope_dims` rotates; the tail passes through untouched.
        let cd2 = k.cu(d2);
        let crot = k.cu(rope_dims);
        let is_rot = k.ult(lane, crot);
        let vcur = k.load_var_f(vf);
        k.shared_store(red, redp, lane, vcur);
        k.wg_barrier();
        k.iff(is_rot, |mut k| {
            let jl = k.umod(lane, cd2);
            let is_hi = k.e(OP_UGE, k.bool_t, &[lane, cd2]);
            let jf = k.utof(jl);
            // `-2·j/rope_dims·ln(theta)`, dividing by the ROTATED width — the reference's
            // `rope_partial` uses `dims`, not the head width, so a partial rope is a full rope over
            // a shorter head and not a truncated one over a long head. Dividing by `hd` here would
            // give every rotated dim the wrong frequency while still looking like a rope.
            let two_over_hd = k.cf(-2.0 / rope_dims as f32 * theta.ln());
            let expo = k.fmul(jf, two_over_hd);
            let invf = k.ext1(GLSL_EXP, expo);
            let posf = k.utof(pos);
            let ang0 = k.fmul(posf, invf);
            // RANGE-REDUCE to [−π,π] before the trig. GPU builtin sin/cos lose ALL range past
            // ~2^12 rad: MEASURED (2026-07-08) dense output collapses to token 0 at EXACTLY pos
            // 4096 — the highest-freq RoPE dim has invf=1 ⇒ ang=pos, so ang overruns the range
            // and every position ≥ 4096 was garbage (the entire >4k context was broken). Cody–
            // Waite two-part 2π so n·2π stays exact in f32 at pos ~64k (single-const loses ~0.5
            // ulp·n ≈ 0.04 rad by 64k). n = round(ang/2π); ang = (ang − n·C1) − n·C2.
            let inv2pi = k.cf_bits(0x3e22_f983);
            let nr0 = k.fmul(ang0, inv2pi);
            let n = k.ext1(GLSL_ROUND, nr0);
            let c1 = k.cf_bits(0x40c9_1000); // 2π high part, exact in f32 (13 bits)
            let c2 = k.cf_bits(0xb795_777a); // 2π low part
            let nc1 = k.fmul(n, c1);
            let a1 = k.fsub(ang0, nc1);
            let nc2 = k.fmul(n, c2);
            let ang = k.fsub(a1, nc2);
            let c = k.ext1(GLSL_COS, ang);
            let sn = k.ext1(GLSL_SIN, ang);
            let j2 = k.iadd(jl, cd2);
            let x0 = k.shared_load(red, redp, jl);
            let x1 = k.shared_load(red, redp, j2);
            let a = k.fmul(x0, c);
            let b = k.fmul(x1, sn);
            let r_lo = k.fsub(a, b);
            let cc = k.fmul(x1, c);
            let dd = k.fmul(x0, sn);
            let r_hi = k.fadd(cc, dd);
            let fl = k.float;
            let r = k.e(OP_SELECT, fl, &[is_hi, r_hi, r_lo]);
            k.store_var(vf, r);
            k
        });
        k.wg_barrier(); // every lane has read `red`

        // the roped value returns to qkv ONCE (the next dispatch reads q from it); the k-head below
        // takes the register `vf` and workgroup memory, not this storage word.
        let vfin = k.load_var_f(vf);
        let roped_i = k.iadd(base, lane);
        k.store_f(qkv, roped_i, vfin);
        // k-head workgroups append the roped k + the (unroped) v to the caches
        k.iff(is_k, |mut k| {
            let ct = k.cu(tmax);
            let kval = vfin; // roped k at this lane (for the key box below)
                             // V[(kvh·tmax + pos)·hd + lane], carried unroped in the register `vv`
            let v1 = k.imul(kvh, ct);
            let v2 = k.iadd(v1, pos);
            if let Some(vsc) = vsc {
                // ── int8 V, the same two steps K takes, on the register source ──────────────────
                // Placed FIRST in the k-head block, and closed with a barrier, because `red` is
                // then handed to the centroid fold and to the K absmax in turn; every one of those
                // hand-offs in this kernel is already barrier-separated for the same WAR reason
                // (all lanes read `red[0]`, then all lanes overwrite `red[lane]`).
                k.absmax_reduce_reg(vv, red, redp, hd, lane);
                let c0v = k.cu(0);
                let vamax = k.shared_load(red, redp, c0v);
                let inv127v = k.cf_bits(0x3c01_0204);
                let vs = k.fmul(vamax, inv127v); // amax ≥ 1e-12 (floored) ⇒ nonzero scale
                let isz = k.e(OP_IEQ, k.bool_t, &[lane, c0v]);
                k.iff(isz, |mut k| {
                    let vsi = k.iadd(v1, pos);
                    k.store_f(vsc, vsi, vs);
                    k
                });
                // V is POSITION-major, so lanes 4w..4w+3 are contiguous and land in one word:
                // Vq[(kvh·tmax + pos)·(hd/4) + w]. The four codes are packed from the
                // workgroup-memory copy of `vv`, not from the storage buffer. (K is word-i-major
                // and packs the same four source lanes into a word at a strided destination —
                // different address, same pack.)
                let chd4v = k.cu(hd / 4);
                let inpack = k.ult(lane, chd4v);
                k.shared_store(sval, svalp, lane, vv);
                k.wg_barrier();
                k.iff(inpack, |mut k| {
                    let c4 = k.cu(4);
                    let l4 = k.imul(lane, c4);
                    let word = k.pack_int8_shared(sval, svalp, l4, vs);
                    let vr = k.imul(v2, chd4v);
                    let vidx = k.iadd(vr, lane);
                    k.store_u(vc, vidx, word);
                    k
                });
                k.wg_barrier(); // `red` and `sval` are the centroid fold's / K pack's next
            } else {
                let vval = k.load_var_f(vv);
                let v3 = k.imul(v2, chd);
                let vidx = k.iadd(v3, lane);
                k.store_f16(vc, vidx, vval);
            }
            // SparseSelection KNOT, incremental (`radius_update_sound`): fold this dim into the
            // block's running-mean centroid c and covering radius ρ, O(1)/token, from the f16 roped
            // key directly (no int8 round-trip). bx[(kvh·hd+lane)·nblocks+block] = c_lane; ρ is a
            // scalar (lane 0). First token seeds c=k, ρ=0. Each next: c' = c + (k−c)/count; ρ' =
            // max(ρ + ‖c−c'‖, ‖k−c'‖) — the triangle rule keeps ρ ≥ max‖k−c'‖ between rebuilds
            // (`radius_update_sound`). ‖k−c‖ is one dim-reduce; ‖c−c'‖ = ‖k−c‖/count and ‖k−c'‖ =
            // ‖k−c‖·r/count fall out algebraically, so only ONE reduction per append.
            if let (Some(bx), Some(rho)) = (bx, rho) {
                let cblk = k.cu(blk);
                let block = k.udiv(pos, cblk);
                let cnb = k.cu(nblocks);
                let c1 = k.cu(1);
                let bt0 = k.imul(kvh, chd);
                let bt1 = k.iadd(bt0, lane);
                let bt2 = k.imul(bt1, cnb);
                let ci = k.iadd(bt2, block); // centroid slot for (kvh, dim=lane, block)
                let rrow = k.imul(kvh, cnb);
                let rhi = k.iadd(rrow, block); // radius slot for (kvh, block)
                let r = k.umod(pos, cblk);
                let is_first = k.e(OP_IEQ, k.bool_t, &[r, c0]);
                k.iff(is_first, |mut k| {
                    k.store_f16(bx, ci, kval); // c = k
                    let is0 = k.e(OP_IEQ, k.bool_t, &[lane, c0]);
                    k.iff(is0, |mut k| {
                        let z = k.cf_bits(0x0000_0000);
                        k.store_f(rho, rhi, z); // ρ = 0
                        k
                    });
                    k
                });
                let notf = k.ult(c0, r); // 0 < pos%blk (uniform: pos is params[0]) — not the first token
                k.iff(notf, |mut k| {
                    let cold = k.load_f16(bx, ci);
                    let diff = k.fsub(kval, cold); // k − c_old, this dim
                    let d2v = k.fmul(diff, diff);
                    k.shared_store(red, redp, lane, d2v);
                    k.wg_barrier();
                    k.tree_reduce(red, redp, hd, lane, false); // Σ_dim (k−c)²
                    let sumsq = k.shared_load(red, redp, c0);
                    let dist = k.ext1(GLSL_SQRT, sumsq); // ‖k − c_old‖ (same for all lanes)
                    let cntv = k.iadd(r, c1); // count' = r+1
                    let cntf = k.utof(cntv);
                    let dc = k.fdiv(diff, cntf);
                    let cnew = k.fadd(cold, dc); // c' = c + (k−c)/count'
                    k.store_f16(bx, ci, cnew);
                    let is0 = k.e(OP_IEQ, k.bool_t, &[lane, c0]);
                    k.iff(is0, |mut k| {
                        let rold = k.load_f(rho, rhi);
                        // ρ' = max(ρ_old, ‖k − c'‖) — running max distance to the CURRENT centroid.
                        let rf = k.utof(r);
                        let dnew0 = k.fmul(dist, rf);
                        let dnew = k.fdiv(dnew0, cntf); // ‖k − c'‖ = ‖k−c‖·r/count'
                        let rnew = k.ext2(GLSL_FMAX, rold, dnew);
                        k.store_f(rho, rhi, rnew);
                        k
                    });
                    k
                });
                k.wg_barrier(); // centroid/ρ writes done before the int8 pack (below) re-reads bx
            }
            // int8 K chart: the whole k-head WG reduces the roped-k absmax → per-token scale sk,
            // then lanes lane<hd/4 pack 4 codes/word into the DP4A K cache (word-i-major). The
            // roped k is the register `kval` (workgroup memory carries it to the pack), not a fresh
            // read of the storage word.
            k.absmax_reduce_reg(kval, red, redp, hd, lane);
            let c0b = k.cu(0);
            let amax = k.shared_load(red, redp, c0b);
            let inv127 = k.cf_bits(0x3c01_0204);
            let sc = k.fmul(amax, inv127); // amax ≥ 1e-12 (floored) ⇒ nonzero scale
            let isz = k.e(OP_IEQ, k.bool_t, &[lane, c0b]);
            k.iff(isz, |mut k| {
                let skrow = k.imul(kvh, ct);
                let skidx = k.iadd(skrow, pos);
                k.store_f(sk, skidx, sc);
                k
            });
            let chd4 = k.cu(hd / 4);
            let inpack = k.ult(lane, chd4);
            k.shared_store(sval, svalp, lane, kval);
            k.wg_barrier();
            k.iff(inpack, |mut k| {
                let c4 = k.cu(4);
                let l4 = k.imul(lane, c4);
                let word = k.pack_int8_shared(sval, svalp, l4, sc);
                let wrow0 = k.imul(kvh, chd4);
                let wrow = k.iadd(wrow0, lane);
                let w1 = k.imul(wrow, ct);
                let kidx = k.iadd(w1, pos);
                k.store_u(kcq, kidx, word);
                k
            });
            // int8 selector chart: re-pack THIS block's f16 CENTROID → one SDot plane + per-block
            // scale (old blocks are immutable). Same-WG as the centroid fold; the barrier above
            // ordered the cross-lane centroid writes. theta_scan reads this plane for ⟪q,c⟫.
            if let (Some(bx), Some(bxq), Some(sbb)) = (bx, bxq, sbb) {
                let cblk = k.cu(blk);
                let block = k.udiv(pos, cblk);
                let cnb = k.cu(nblocks);
                k.wg_barrier();
                // absmax over the block's hd centroid values: lane i folds its own c_i
                let bt0 = k.imul(kvh, chd);
                let bt1 = k.iadd(bt0, lane);
                let bt2 = k.imul(bt1, cnb);
                let ci = k.iadd(bt2, block);
                let cval = k.load_f16(bx, ci);
                let f = k.float;
                let g = k.glsl;
                let ac = k.e(OP_EXT_INST, f, &[g, GLSL_FABS, cval]);
                let tiny = k.cf_bits(0x2b8c_bccc);
                let am = k.ext2(GLSL_FMAX, ac, tiny);
                k.shared_store(red, redp, lane, am);
                k.wg_barrier();
                k.tree_reduce(red, redp, hd, lane, true);
                let c0r = k.cu(0);
                let bam = k.shared_load(red, redp, c0r);
                let inv127b = k.cf_bits(0x3c01_0204);
                let sb = k.fmul(bam, inv127b);
                let isz0 = k.e(OP_IEQ, k.bool_t, &[lane, c0r]);
                k.iff(isz0, |mut k| {
                    let sbi0 = k.imul(kvh, cnb);
                    let sbi = k.iadd(sbi0, block);
                    k.store_f(sbb, sbi, sb);
                    k
                });
                // lanes w < hd/4 pack word w of the centroid plane: bxq[(kvh·(hd/4)+w)·nblocks+block]
                let inw = k.ult(lane, chd4);
                k.iff(inw, |mut k| {
                    let word = k.fvar_u();
                    let zw = k.cu(0);
                    k.store_var(word, zw);
                    let c0j = k.cu(0);
                    let four = k.cu(4);
                    k.loop_u(c0j, four, |mut k, j| {
                        let c4b = k.cu(4);
                        let l4b = k.imul(lane, c4b);
                        let i = k.iadd(l4b, j);
                        let e0 = k.imul(kvh, chd);
                        let e1 = k.iadd(e0, i);
                        let e2 = k.imul(e1, cnb);
                        let eip = k.iadd(e2, block);
                        let v = k.load_f16(bx, eip);
                        let byte = k.int8_code(v, sb);
                        let u = k.uint;
                        let c8 = k.cu(8);
                        let sh = k.imul(j, c8);
                        let shifted = k.e(OP_SHIFT_LEFT_LOGICAL, u, &[byte, sh]);
                        let w0 = k.load_var_u(word);
                        let w1 = k.e(OP_BITWISE_OR, u, &[w0, shifted]);
                        k.store_var(word, w1);
                        k
                    });
                    let wv = k.load_var_u(word);
                    let p0 = k.imul(kvh, chd4);
                    let p3 = k.iadd(p0, lane);
                    let p4 = k.imul(p3, cnb);
                    let p5 = k.iadd(p4, block);
                    k.store_u(bxq, p5, wv);
                    k
                });
            }
            k
        });
        k
    });
    k.finish(hd)
}

/// **SiLU gate** — `out[i] = g/(1+e^{−g}) · u[i]` (the NB gate, `SeedForward.swigluB`).
/// Bindings: 0 = g, 1 = u, 2 = out. Dispatch ≥ n.
pub fn silu_mul_module(n: u32, local: u32, u_off: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let g = k.ssbo_f(0);
    let u = k.ssbo_f(1);
    let out = k.ssbo_f(2);
    let i = k.gid_comp(0);
    let cn = k.cu(n);
    let guard = k.ult(i, cn);
    k.iff(guard, |mut k| {
        let gv = k.load_f(g, i);
        let cuo = k.cu(u_off);
        let ui = k.iadd(i, cuo);
        let uv = k.load_f(u, ui);
        let f = k.float;
        let ng = k.e(OP_FNEG, f, &[gv]);
        let eg = k.ext1(GLSL_EXP, ng);
        let one = k.cf_bits(0x3f80_0000);
        let den = k.fadd(one, eg);
        let sil = k.fdiv(gv, den);
        let o = k.fmul(sil, uv);
        k.store_f(out, i, o);
        k
    });
    k.finish(local)
}

/// **Residual / bias add, in place** — `x[i] += y[i]` (the monad pass-through). Bindings:
/// 0 = x (rw), 1 = y. Dispatch ≥ n.
pub fn add_inplace_module(n: u32, local: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let y = k.ssbo_f(1);
    let i = k.gid_comp(0);
    let cn = k.cu(n);
    let guard = k.ult(i, cn);
    k.iff(guard, |mut k| {
        let xv = k.load_f(x, i);
        let yv = k.load_f(y, i);
        let s = k.fadd(xv, yv);
        k.store_f(x, i, s);
        k
    });
    k.finish(local)
}

/// **Per-head QK-norm** (Qwen3) — RMSNorm over each head's `hd` slice, one invocation per head
/// (`SeedForward.qkNormB`, plain-`w` convention). Bindings: 0 = x (rw), 1 = g. Dispatch ≥ heads.
pub fn qk_norm_module(heads: u32, hd: u32, eps: f32, off: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let g = k.ssbo_f(1);
    let h = k.gid_comp(0);
    let ch = k.cu(heads);
    let guard = k.ult(h, ch);
    k.iff(guard, |mut k| {
        let chd = k.cu(hd);
        let base0 = k.imul(h, chd);
        let coff = k.cu(off);
        let base = k.iadd(base0, coff);
        let acc = k.fvar_f();
        let zero = k.cf_bits(0x0000_0000);
        k.store_var(acc, zero);
        let c0 = k.cu(0);
        let chd2 = k.cu(hd);
        k.loop_u(c0, chd2, |mut k, i| {
            let idx = k.iadd(base, i);
            let v = k.load_f(x, idx);
            let sq = k.fmul(v, v);
            let a0 = k.load_var_f(acc);
            let a1 = k.fadd(a0, sq);
            k.store_var(acc, a1);
            k
        });
        let ss = k.load_var_f(acc);
        let invn = k.cf(1.0 / hd as f32);
        let mean = k.fmul(ss, invn);
        let ce = k.cf(eps);
        let me = k.fadd(mean, ce);
        let inv = k.ext1(GLSL_INVERSE_SQRT, me);
        let c0b = k.cu(0);
        let chd3 = k.cu(hd);
        k.loop_u(c0b, chd3, |mut k, i| {
            let idx = k.iadd(base, i);
            let v = k.load_f(x, idx);
            let gv = k.load_f(g, i);
            let t = k.fmul(v, inv);
            let o = k.fmul(t, gv);
            k.store_f(x, idx, o);
            k
        });
        k
    });
    k.finish(1)
}

/// **Argmax stage 1 — strided partial argmax.** Invocation `p < P` scans `x[p], x[p+P], …`,
/// writing its best value to `partials[p]` and its index (as f32 — exact below 2²⁴) to
/// `partials[P+p]`. Bindings: 0 = x, 1 = partials. Dispatch ≥ P.
pub fn argmax_partial_module(n: u32, p_count: u32, local: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let partials = k.ssbo_f(1);
    let p = k.gid_comp(0);
    let cp = k.cu(p_count);
    let guard = k.ult(p, cp);
    k.iff(guard, |mut k| {
        let best = k.fvar_f();
        let bi = k.fvar_f();
        let neg = k.cf_bits(0xf149_f2ca);
        k.store_var(best, neg);
        let z = k.cf_bits(0x0000_0000);
        k.store_var(bi, z);
        let iv = k.fvar_u();
        k.store_var(iv, p);
        let (hdr, chk, bod, cont, merge) = (k.id(), k.id(), k.id(), k.id(), k.id());
        k.ins(OP_BRANCH, &[hdr]);
        k.ins(OP_LABEL, &[hdr]);
        k.ins(OP_LOOP_MERGE, &[merge, cont, 0]);
        k.ins(OP_BRANCH, &[chk]);
        k.ins(OP_LABEL, &[chk]);
        let i = k.load_var_u(iv);
        let cn = k.cu(n);
        let cond = k.ult(i, cn);
        k.ins(OP_BRANCH_CONDITIONAL, &[cond, bod, merge]);
        k.ins(OP_LABEL, &[bod]);
        let v = k.load_f(x, i);
        let b0 = k.load_var_f(best);
        let bt = k.bool_t;
        let gt = k.e(OP_FORD_GREATER_THAN, bt, &[v, b0]);
        k.iff(gt, |mut k| {
            k.store_var(best, v);
            let inf = k.utof(i);
            k.store_var(bi, inf);
            k
        });
        k.ins(OP_BRANCH, &[cont]);
        k.ins(OP_LABEL, &[cont]);
        let i2 = k.load_var_u(iv);
        let step = k.cu(p_count);
        let inx = k.iadd(i2, step);
        k.store_var(iv, inx);
        k.ins(OP_BRANCH, &[hdr]);
        k.ins(OP_LABEL, &[merge]);
        let bf = k.load_var_f(best);
        k.store_f(partials, p, bf);
        let bif = k.load_var_f(bi);
        let cpp = k.cu(p_count);
        let pi = k.iadd(p, cpp);
        k.store_f(partials, pi, bif);
        k
    });
    k.finish(local)
}

/// **Argmax stage 2 — the finish.** ONE invocation reduces the P (value, index) pairs and
/// writes the winning index into `params[1]` — the sampled token, never leaving the device.
/// Bindings: 0 = partials, 1 = params (rw). Dispatch (1,1,1).
pub fn argmax_finish_module(p_count: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let partials = k.ssbo_f(0);
    let params = k.ssbo_u(1);
    let i = k.gid_comp(0);
    let c0 = k.cu(0);
    let b = k.bool_t;
    let is0 = k.e(OP_IEQ, b, &[i, c0]);
    k.iff(is0, |mut k| {
        let best = k.fvar_f();
        let bi = k.fvar_f();
        let neg = k.cf_bits(0xf149_f2ca);
        k.store_var(best, neg);
        let z = k.cf_bits(0x0000_0000);
        k.store_var(bi, z);
        let c0u = k.cu(0);
        let cp = k.cu(p_count);
        k.loop_u(c0u, cp, |mut k, j| {
            let v = k.load_f(partials, j);
            let b0 = k.load_var_f(best);
            let bt = k.bool_t;
            let cpp = k.cu(p_count);
            let ji = k.iadd(j, cpp);
            let idxf = k.load_f(partials, ji);
            // **Ties resolve to the LOWEST GLOBAL INDEX, not the lowest partial.** Stage 1 strides
            // (`x[p], x[p+P], …`), so partial `p` holds the lowest index of its own stride, but
            // across partials the index order is NOT the partial order — `p=3, j=1` is global index
            // `259` while `p=5, j=0` is `5`. A plain `v > best` would return 259 on an exact tie
            // where a host `iter().position(max)` returns 5, i.e. a DIFFERENT TOKEN for the same
            // logits. The `==` arm is what makes this device argmax bit-for-bit the host's.
            let gt = k.e(OP_FORD_GREATER_THAN, bt, &[v, b0]);
            let eq = k.e(OP_FORD_EQUAL, bt, &[v, b0]);
            let b1 = k.load_var_f(bi);
            let lower = k.e(OP_FORD_LESS_THAN, bt, &[idxf, b1]);
            let eqlow = k.e(OP_LOGICAL_AND, bt, &[eq, lower]);
            let take = k.e(OP_LOGICAL_OR, bt, &[gt, eqlow]);
            k.iff(take, |mut k| {
                k.store_var(best, v);
                k.store_var(bi, idxf);
                k
            });
            k
        });
        let bif = k.load_var_f(bi);
        let u = k.uint;
        let tok = k.e(OP_CONVERT_F_TO_U, u, &[bif]);
        let psu = k.ptr_sb_u;
        let c0b = k.cu(0);
        let c1 = k.cu(1);
        let ptr = k.e(OP_ACCESS_CHAIN, psu, &[params, c0b, c1]);
        k.ins(OP_STORE, &[ptr, tok]);
        k
    });
    k.finish(1)
}

/// **Argmax stage 2 — the workgroup tree finish.** ONE workgroup of `WG = 256` invocations reduces
/// the `P` (value, index) partial pairs of [`argmax_partial_module`] and writes the winning index
/// into `params[1]` (byte offset 4, as u32) — exactly where [`argmax_finish_module`] writes it.
///
/// Lane `t` first scans the strided slice `partials[t], partials[t+WG], … < P` with the SAME take
/// rule (greater value wins; equal value → LOWER GLOBAL INDEX wins), then stores its per-lane
/// winner into workgroup memory and the `log₂(WG)` tree combines with the same rule. The rule is a
/// total order on (value desc, index asc) and the partial indices are disjoint across lanes'
/// slices, so the tree's winner is the global maximiser independent of the grouping — bit-identical
/// to the serial finish.
///
/// Where the serial finish is one invocation walking `P` pairs serially, this spends `WG` lanes on
/// the same `P` pairs: the finish stops being the `O(P)` bottleneck of the pair. Bindings:
/// 0 = partials, 1 = params (rw). Dispatch (1,1,1) at LocalSize `WG`.
pub fn argmax_finish_tree_module(p_count: u32) -> Vec<u32> {
    const WG: u32 = 256;
    let mut k = Kb::new();
    let partials = k.ssbo_f(0);
    let params = k.ssbo_u(1);
    let i = k.gid_comp(0);
    let bt = k.bool_t;
    // Per-lane serial scan of the strided slice, the shipped stage-2 take rule.
    let best = k.fvar_f();
    let bi = k.fvar_f();
    let neg = k.cf_bits(0xf149_f2ca);
    k.store_var(best, neg);
    let z = k.cf_bits(0x0000_0000);
    k.store_var(bi, z);
    let cwg = k.cu(WG);
    let cp = k.cu(p_count);
    k.loop_step(i, cp, cwg, |k, j| {
        let v = k.load_f(partials, j);
        let b0 = k.load_var_f(best);
        let cpp = k.cu(p_count);
        let ji = k.iadd(j, cpp);
        let idxf = k.load_f(partials, ji);
        let gt = k.e(OP_FORD_GREATER_THAN, bt, &[v, b0]);
        let eq = k.e(OP_FORD_EQUAL, bt, &[v, b0]);
        let b1 = k.load_var_f(bi);
        let lower = k.e(OP_FORD_LESS_THAN, bt, &[idxf, b1]);
        let eqlow = k.e(OP_LOGICAL_AND, bt, &[eq, lower]);
        let take = k.e(OP_LOGICAL_OR, bt, &[gt, eqlow]);
        k.iff(take, |mut k| {
            k.store_var(best, v);
            k.store_var(bi, idxf);
            k
        });
    });
    // Workgroup tree: (value, index) pairs in workgroup memory, combined with the same rule.
    let (sval, svalp) = k.shared_f32(WG);
    let (sidx, sidxp) = k.shared_f32(WG);
    let bf = k.load_var_f(best);
    let bif = k.load_var_f(bi);
    k.shared_store(sval, svalp, i, bf);
    k.shared_store(sidx, sidxp, i, bif);
    k.wg_barrier();
    let mut stride = WG / 2;
    while stride >= 1 {
        let cs = k.cu(stride);
        let lt = k.ult(i, cs);
        k.iff(lt, |mut k| {
            let a = k.shared_load(sval, svalp, i);
            let ti = k.iadd(i, cs);
            let b = k.shared_load(sval, svalp, ti);
            let ai = k.shared_load(sidx, sidxp, i);
            let bvi = k.shared_load(sidx, sidxp, ti);
            // take `b` iff its value is greater, or equal with a lower global index.
            let gt = k.e(OP_FORD_GREATER_THAN, bt, &[b, a]);
            let eq = k.e(OP_FORD_EQUAL, bt, &[b, a]);
            let lower = k.e(OP_FORD_LESS_THAN, bt, &[bvi, ai]);
            let eqlow = k.e(OP_LOGICAL_AND, bt, &[eq, lower]);
            let take = k.e(OP_LOGICAL_OR, bt, &[gt, eqlow]);
            k.iff(take, |mut k| {
                k.shared_store(sval, svalp, i, b);
                k.shared_store(sidx, sidxp, i, bvi);
                k
            });
            k
        });
        k.wg_barrier();
        stride /= 2;
    }
    // Lane 0 writes the winning index into `params[1]` — the sampled token, never leaving the device.
    let c0 = k.cu(0);
    let is0 = k.e(OP_IEQ, bt, &[i, c0]);
    k.iff(is0, |mut k| {
        let win = k.shared_load(sidx, sidxp, c0);
        let u = k.uint;
        let tok = k.e(OP_CONVERT_F_TO_U, u, &[win]);
        let psu = k.ptr_sb_u;
        let c0b = k.cu(0);
        let c1 = k.cu(1);
        let ptr = k.e(OP_ACCESS_CHAIN, psu, &[params, c0b, c1]);
        k.ins(OP_STORE, &[ptr, tok]);
        k
    });
    k.finish(WG)
}
const OP_CONVERT_F_TO_U: u32 = 109;

/// **The GPU-resident decode step: commit + advance.** The one on-device state transition that
/// closes the spinor loop without a CPU round-trip. Copies the just-sampled token `params[1]` (the
/// argmax_finish output) into `tokens_out[slot]` for host readback, and increments `params[0]` (the
/// position) so the NEXT unrolled token-copy's embed/rope/attention see pos+1 — everything flows on
/// the device. `slot` is baked per recorded copy (each copy gets its own pipeline). One invocation,
/// dispatch (1,1,1). Bindings: 0 = params (rw u32: [0]=pos, [1]=token), 1 = tokens_out (u32[]).
pub fn commit_advance_module(slot: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let params = k.ssbo_u(0);
    let tokens = k.ssbo_u(1);
    let i = k.gid_comp(0);
    let c0 = k.cu(0);
    let b = k.bool_t;
    let is0 = k.e(OP_IEQ, b, &[i, c0]);
    k.iff(is0, |mut k| {
        // tokens_out[slot] = params[1] (the sampled token)
        let one = k.cu(1);
        let tok = k.load_u(params, one);
        let cslot = k.cu(slot);
        k.store_u(tokens, cslot, tok);
        // params[0] = pos + 1
        let zero = k.cu(0);
        let pos = k.load_u(params, zero);
        let cone = k.cu(1);
        let pos1 = k.iadd(pos, cone);
        k.store_u(params, zero, pos1);
        k
    });
    k.finish(1)
}

// bytes wrappers
/// `embed_f16_module` as bytes.
pub fn embed_f16_bytes(hidden: u32, vocab: u32, local: u32) -> Vec<u8> {
    bytes(embed_f16_module(hidden, vocab, local))
}
/// `embed_chart_module` as bytes.
pub fn embed_chart_bytes(
    hidden: u32,
    vocab: u32,
    local: u32,
    carrier: crate::gpu::simt::Carrier,
) -> Vec<u8> {
    bytes(embed_chart_module(hidden, vocab, local, carrier))
}
/// `rmsnorm_fused_q_module` as bytes.
pub fn rmsnorm_fused_q_bytes(n: u32, eps: f32, dpw: u32) -> Vec<u8> {
    bytes(rmsnorm_module(n, eps, dpw, None, true))
}
/// `rmsnorm_module` (residual-add + quantize-out) as bytes.
pub fn rmsnorm_add_q_seg_bytes(n: u32, eps: f32, dpw: u32, res_segs: u32) -> Vec<u8> {
    bytes(rmsnorm_module(n, eps, dpw, Some(res_segs), true))
}
/// `rmsnorm_module` (residual-add, f32 out — the F16 carrier's final-norm twin of
/// `rmsnorm_add_q_seg_bytes`). Bindings: 0 = x, 1 = g, 2 = res, 3 = out.
pub fn rmsnorm_add_seg_bytes(n: u32, eps: f32, res_segs: u32) -> Vec<u8> {
    bytes(rmsnorm_module(n, eps, 0, Some(res_segs), false))
}
/// `quantize_vec_module` as bytes.
pub fn quantize_vec_bytes(n: u32, dpw: u32) -> Vec<u8> {
    bytes(quantize_vec_module(n, dpw))
}
/// `gemv_norm_module` (fused RMSNorm → gemv, carrier the dial) as bytes.
pub fn gemv_norm_bytes(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: crate::gpu::simt::Carrier,
    eps: f32,
    residual: Option<u32>,
) -> Vec<u8> {
    gemv_norm_bytes_sc(m, k, local, segments, carrier, eps, residual, false)
}

/// [`gemv_norm_bytes`] with the **scale face** chosen — see [`gemv_norm_module_sc`]. Bindings are
/// unchanged; only binding 4's layout differs.
#[allow(clippy::too_many_arguments)]
pub fn gemv_norm_bytes_sc(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: crate::gpu::simt::Carrier,
    eps: f32,
    residual: Option<u32>,
    two_level: bool,
) -> Vec<u8> {
    bytes(gemv_norm_module_sc(
        m, k, local, segments, carrier, eps, residual, two_level,
    ))
}
/// Back-compat alias for [`gemv_norm_bytes`] (historical DP4A-only name).
#[allow(clippy::too_many_arguments)]
pub fn gemv_dp4a_norm_bytes(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: crate::gpu::simt::Carrier,
    eps: f32,
    residual: Option<u32>,
) -> Vec<u8> {
    gemv_norm_bytes(m, k, local, segments, carrier, eps, residual)
}
/// `silu_mul_q_module` as bytes.
pub fn silu_mul_q_bytes(n: u32, u_off: u32, dpw: u32) -> Vec<u8> {
    bytes(silu_mul_q_module(n, u_off, dpw))
}
/// `rmsnorm_module` (f16-carrier f32 out, no residual) as bytes.
pub fn rmsnorm_fused_bytes(n: u32, eps: f32) -> Vec<u8> {
    bytes(rmsnorm_module(n, eps, 0, None, false))
}
/// `flash_partial_module` as bytes.
pub fn flash_partial_bytes(
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
) -> Vec<u8> {
    bytes(flash_partial_module(
        nh, nkv, hd, tmax, scale, nsplit, blk, knot, compact, v8, nb_list,
    ))
}
/// [`flash_partial_fold_module`] as bytes — `flash_partial` with the `fused_rope_kv_fold` prologue
/// folded in (the `DEC_ROPEFLASH` path).
#[allow(clippy::too_many_arguments)]
pub fn flash_partial_fold_bytes(
    nh: u32,
    nkv: u32,
    hd: u32,
    tmax: u32,
    scale: f32,
    nsplit: u32,
    blk: u32,
    theta: f32,
    segs: u32,
    bias: bool,
    rope_dims: u32,
) -> Vec<u8> {
    bytes(flash_partial_fold_module(
        nh, nkv, hd, tmax, scale, nsplit, blk, theta, segs, bias, rope_dims,
    ))
}
/// `theta_scan_module` as bytes.
pub fn theta_scan_bytes(
    nh: u32,
    nkv: u32,
    hd: u32,
    tmax: u32,
    scale: f32,
    blk: u32,
    nsplit: u32,
) -> Vec<u8> {
    bytes(theta_scan_module(nh, nkv, hd, tmax, scale, blk, nsplit))
}
/// `escore_module` as bytes (debug: exact per-block max score).
pub fn escore_bytes(nh: u32, nkv: u32, hd: u32, tmax: u32, scale: f32, blk: u32) -> Vec<u8> {
    bytes(escore_module(nh, nkv, hd, tmax, scale, blk))
}
/// `theta_select_module` as bytes.
pub fn theta_select_bytes(
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
) -> Vec<u8> {
    bytes(theta_select_module(
        nh, tmax, blk, nsplit, n_sink, n_local, lslack, cut, budget, iters,
    ))
}
/// `flash_combine_module` as bytes.
pub fn flash_combine_bytes(nh: u32, hd: u32, nsplit: u32) -> Vec<u8> {
    bytes(flash_combine_module(nh, hd, nsplit))
}
/// `fused_rope_kv_module` as bytes.
#[allow(clippy::too_many_arguments)]
pub fn fused_rope_kv_bytes(
    nh: u32,
    nkv: u32,
    hd: u32,
    tmax: u32,
    theta: f32,
    blk: u32,
    rope_dims: u32,
    v8: bool,
) -> Vec<u8> {
    bytes(fused_rope_kv_module(
        nh, nkv, hd, tmax, theta, blk, rope_dims, v8,
    ))
}
/// `fused_rope_kv_fold_module` as bytes — RoPE + KV-append with the qkv split-K fold and bias.
#[allow(clippy::too_many_arguments)]
pub fn fused_rope_kv_fold_bytes(
    nh: u32,
    nkv: u32,
    hd: u32,
    tmax: u32,
    theta: f32,
    blk: u32,
    segs: u32,
    bias: bool,
    rope_dims: u32,
    v8: bool,
) -> Vec<u8> {
    bytes(fused_rope_kv_fold_module(
        nh, nkv, hd, tmax, theta, blk, segs, bias, rope_dims, v8,
    ))
}
/// `argmax_partial_module` as bytes.
pub fn argmax_partial_bytes(n: u32, p: u32, local: u32) -> Vec<u8> {
    bytes(argmax_partial_module(n, p, local))
}
/// `argmax_finish_module` as bytes.
pub fn argmax_finish_bytes(p: u32) -> Vec<u8> {
    bytes(argmax_finish_module(p))
}
/// `argmax_finish_tree_module` as bytes.
pub fn argmax_finish_tree_bytes(p: u32) -> Vec<u8> {
    bytes(argmax_finish_tree_module(p))
}
/// `commit_advance_module` as bytes.
pub fn commit_advance_bytes(slot: u32) -> Vec<u8> {
    bytes(commit_advance_module(slot))
}
/// `silu_mul_module` as bytes.
pub fn silu_mul_bytes(n: u32, local: u32, u_off: u32) -> Vec<u8> {
    bytes(silu_mul_module(n, local, u_off))
}
/// `add_inplace_module` as bytes.
pub fn add_inplace_bytes(n: u32, local: u32) -> Vec<u8> {
    bytes(add_inplace_module(n, local))
}
/// `qk_norm_module` as bytes.
pub fn qk_norm_bytes(heads: u32, hd: u32, eps: f32, off: u32) -> Vec<u8> {
    bytes(qk_norm_module(heads, hd, eps, off))
}

/// **De-interleave a GATED q projection** — `[(q_h, gate_h) × nh | k | v]` → `[q | k | v]` + `gate`.
///
/// Qwen3.5/3.8 set `attn_output_gate`, and then `q_proj` emits TWO `hd`-wide vectors per head: the
/// query and a gate that multiplies the attention output before the o-projection
/// (`attn_step`, hybrid.rs:78). So the q stride is `2·hd`, the k/v split lands at `nh·2·hd`, and
/// every kernel downstream — `fused_rope_kv`, `qk_norm_module`, `flash_partial_module` — computes
/// its base as `head·hd` off a `qn = nh·hd`.
///
/// Threading a stride through all three was the alternative. This is one dispatch per attention
/// layer (16 of 64 blocks at 27B) that hands them the layout they already agree on, and leaves the
/// three most performance-critical kernels in the decode path untouched. The gate lands contiguous,
/// which is what [`attn_gate_apply_module`] then wants.
///
/// Bindings: 0 = `qkv_in` (f32, `nh·2·hd + 2·nkv·hd`), 1 = `qkv_out` (f32, `nh·hd + 2·nkv·hd`),
/// 2 = `gate` (f32, `nh·hd`). Dispatch ≥ `nh·hd + 2·nkv·hd`.
pub fn attn_qgate_split_module(nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const LOCAL: u32 = 256;
    let qn = nh * hd;
    let kvn = 2 * nkv * hd;
    let mut k = Kb::new();
    let src = k.ssbo_f(0);
    let dst = k.ssbo_f(1);
    let gate = k.ssbo_f(2);
    let i = k.gid_comp(0);
    let ctot = k.cu(qn + kvn);
    let guard = k.ult(i, ctot);
    k.iff(guard, |mut k| {
        let cqn = k.cu(qn);
        let chd = k.cu(hd);
        let is_q = k.ult(i, cqn);
        k.iff(is_q, |mut k| {
            // q element `i` is head `i/hd`, lane `i%hd`, living at `head·2·hd + lane`
            let h = k.udiv(i, chd);
            let lane = k.umod(i, chd);
            let c2 = k.cu(2);
            let h2 = k.imul(h, c2);
            let qb = k.imul(h2, chd);
            let qi = k.iadd(qb, lane);
            let qv = k.load_f(src, qi);
            k.store_f(dst, i, qv);
            let gi0 = k.iadd(qb, chd);
            let gi = k.iadd(gi0, lane);
            let gv = k.load_f(src, gi);
            k.store_f(gate, i, gv);
            k
        });
        // k and v sit AFTER the interleaved q/gate block, i.e. at `nh·2·hd` in the source and at
        // `qn` in the destination — the whole point of the split.
        let is_kv = k.e(OP_UGE, k.bool_t, &[i, cqn]);
        k.iff(is_kv, |mut k| {
            let off = k.e(OP_ISUB, k.uint, &[i, cqn]);
            let csrc0 = k.cu(2 * qn);
            let si = k.iadd(csrc0, off);
            let v = k.load_f(src, si);
            k.store_f(dst, i, v);
            k
        });
        k
    });
    k.finish(LOCAL)
}

/// **The attention output gate** — `att[i] *= sigmoid(gate[i])`, in place.
///
/// Applied BEFORE the o-projection, which is where the reference applies it. After it would be a
/// different model: `o_proj` is linear, so gating its output gates a MIXTURE of heads rather than
/// each head's own readout.
///
/// Bindings: 0 = `att` (f32, read-write), 1 = `gate` (f32). Dispatch ≥ `n`.
pub fn attn_gate_apply_module(n: u32) -> Vec<u32> {
    const LOCAL: u32 = 256;
    let mut k = Kb::new();
    let att = k.ssbo_f(0);
    let gate = k.ssbo_f(1);
    let i = k.gid_comp(0);
    let cn = k.cu(n);
    let guard = k.ult(i, cn);
    k.iff(guard, |mut k| {
        let f = k.float;
        let gv = k.load_f(gate, i);
        let ng = k.e(OP_FNEG, f, &[gv]);
        let eg = k.ext1(GLSL_EXP, ng);
        let one = k.cf_bits(0x3f80_0000);
        let den = k.fadd(one, eg);
        let sig = k.fdiv(one, den);
        let av = k.load_f(att, i);
        let o = k.fmul(av, sig);
        k.store_f(att, i, o);
        k
    });
    k.finish(LOCAL)
}

/// [`attn_qgate_split_module`] as bytes.
pub fn attn_qgate_split_bytes(nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_qgate_split_module(nh, nkv, hd))
}

/// [`attn_gate_apply_module`] as bytes.
pub fn attn_gate_apply_bytes(n: u32) -> Vec<u8> {
    bytes(attn_gate_apply_module(n))
}

/// **`dst[i] = src[i]`** — a plain vector copy. Bindings: 0 = dst, 1 = src. Dispatch ≥ `n`.
///
/// Exists for the debug readback in `GpuDecode` (`DEC_DEBUG_ATT`): the only host-visible buffer in
/// that chain is the final hidden, so seeing any INTERMEDIATE means copying it there. A dedicated
/// kernel rather than reusing `add_inplace` because that one accumulates, and accumulating into a
/// buffer the final norm just wrote reads back the sum of two things instead of the one under test.
pub fn copy_vec_module(n: u32, src_off: u32) -> Vec<u32> {
    const LOCAL: u32 = 256;
    let mut k = Kb::new();
    let dst = k.ssbo_f(0);
    let src = k.ssbo_f(1);
    let i = k.gid_comp(0);
    let cn = k.cu(n);
    let guard = k.ult(i, cn);
    k.iff(guard, |mut k| {
        let coff = k.cu(src_off);
        let si = k.iadd(i, coff);
        let v = k.load_f(src, si);
        k.store_f(dst, i, v);
        k
    });
    k.finish(LOCAL)
}

/// [`copy_vec_module`] as bytes.
pub fn copy_vec_bytes(n: u32, src_off: u32) -> Vec<u8> {
    bytes(copy_vec_module(n, src_off))
}
