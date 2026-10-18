//! simt::superblock — the row-tiled and SUPERBLOCK DP4A kernels: the `rt` row-tile GEMM, the
//! `sb` family (accsat / LDS env gates, the packed-block form, the full `_ex` builder) and the
//! host-side q2 superblock chart transpose they read. Cut verbatim from `simt.rs`.

use super::*;

/// **The 2-D register-tiled integer-dot GEMM — `rt` rows × `bt` columns per invocation.**
///
/// [`gemv_dp4a_batch_module`] tiles only the batch axis, so its inner loop is ONE global load per
/// `OpSUDot`: for each of the `bt` columns it loads that column's activation word and immediately
/// consumes it. The weight word is reused `bt` ways, but the activation word is reused ZERO ways,
/// and a load issues on the LSU at a small fraction of the ALU rate — so the dot never gets near
/// the instruction's own ceiling no matter how large `bt` grows (measured: bt 8→16 buys ~0%).
///
/// The fix is the second tile axis. Each invocation owns `rt` output rows as well as `bt` columns,
/// so ONE activation load feeds `rt` dots and ONE weight load feeds `bt` dots:
///
/// ```text
///   loads per 32-block  = rt (scales) + cpb·rt (weights) + 8·bt (activations)
///   OpSUDot per 32-block = 8·rt·bt
///   rt=1,bt=8  →  69 loads : 64 dots  = 0.93   (the old kernel)
///   rt=4,bt=8  →  84 loads : 256 dots = 3.05
///   rt=8,bt=8  → 104 loads : 512 dots = 4.92
/// ```
///
/// This is llama.cpp MMQ's shape argument arriving at a different instruction: its 128×128×256
/// tile at 255 registers and occupancy 1 (measured on this box) buys arithmetic intensity with
/// registers, and takes the occupancy loss. It is NOT its async-copy pipeline — MMQ has no
/// `cp.async` and no double buffering at all (measured: `LDGSTS` count 0), so neither does this.
///
/// **Three further levers, all in the emitter because the shape is known at module-build time:**
/// 1. **The `cpb` inner loop is FULLY UNROLLED.** `cpb = 32/codes_per_word` is a compile-time
///    constant (Q8→8, Q4→4, Q2→2), so the loop, its merge block and its counter all vanish and the
///    driver can hoist every load of the block to the top and schedule against one latency.
/// 2. **Strength-reduced addressing.** Per 32-block the kernel forms three bases and every access
///    is `base + literal`, which folds into the load's immediate-offset field instead of an `IADD`.
/// 3. **The activation key is WARP-UNIFORM by construction.** The batch index comes from
///    `WorkgroupId.y`, not `GlobalInvocationId.y`. Both are uniform across the workgroup when
///    `local_size.y = 1`, but only the former says so syntactically, which is what lets the driver
///    put the activation address in the uniform datapath instead of one address per lane.
///
/// **The mandala closure (`MandalaMemory.access_in_region`).** The only unconstrained keys this
/// kernel receives are the two invocation-id components; every other index is a loop counter with a
/// compile-time bound. The row key is closed into the weight's row ring by `UMin(row, m−1)` before
/// it addresses W or DM — the non-power-of-two form of `key &&& (2ᵏ−1)`, one instruction, no branch
/// and no check to elide (`masked_load_never_oob`: the loads cannot form an out-of-region address).
/// It also REPLACES the old kernel's `if r < m` guard around the whole body: the ring closure is
/// the fast path and the safety at once, and only the final store keeps a predicate (a clamped row
/// must not write). The batch key is left bounded by dispatch construction, not by mask — closing
/// it needs the activation and logit buffers padded to a power of two, which at the 0.5B lm head is
/// 152k × 1024 × 4 B = 622 MB of VRAM against 443 MB for the whole model. Stated, not hidden.
///
/// Bindings and chart layout are byte-identical to [`gemv_dp4a_batch_module`]; only the dispatch
/// changes: `groups = (ceil(M/(local·rt)), B/bt)`. `rt = 1` is the same kernel as before.
pub fn gemv_dp4a_batch_rt_module(
    m: u32,
    k: u32,
    local: u32,
    carrier: Carrier,
    bt: u32,
    rt: u32,
) -> Vec<u32> {
    assert!(k % 32 == 0, "chart carriers need k % 32 == 0");
    assert!(
        carrier.is_chart(),
        "DP4A folds a chart carrier (Q8/Q4/Q2 are one plane; Q3/Q5 are two and are refused below)"
    );
    // A two-plane chart (Q3/Q5) carries its high bit in a plane appended to `w`, one word per
    // 32-block per row; this kernel walks ONE code plane and would read plane B as the next row's
    // plane A, returning a plausible wrong number. Refuse by name, and point at the path that does
    // decode both: the cooperative-matrix chart GEMM is a two-plane carrier's prefill GEMM.
    assert!(
        carrier.plane_b_off(m, k) == 0,
        "the row-tiled DP4A batch GEMM reads one code plane: {carrier:?} is a two-plane chart \
(Q3/Q5) whose plane B it cannot decode. The cooperative-matrix chart GEMM is the prefill path \
for a two-plane carrier; this kernel refuses it rather than folding plane A alone."
    );
    assert!(bt >= 1 && rt >= 1, "both tiles must be >= 1");
    let codes_per_word = carrier.pack(); //   Q8 → 4, Q4 → 8, Q2 → 16
    let dots_per_word = codes_per_word / 4; // Q8 → 1, Q4 → 2, Q2 → 4
    let cpb = 32 / codes_per_word; //          weight words per 32-block
    let wbits = 32 / codes_per_word; //        bits per code
    let byte_mask = 0x0101_0101u32 * ((1u32 << wbits) - 1);
    let split = dots_per_word > 1;
    let blocks = k / 32;
    let xstride = k / 4; // u32 activation words per batch row (4 int8 per word)

    let mut idc = 0u32;
    macro_rules! id {
        () => {{
            idc += 1;
            idc
        }};
    }
    let (void, fnty, uint, sint, float, bool_t) = (id!(), id!(), id!(), id!(), id!(), id!());
    let (uvec3, vec2f, ptr_in_uvec3, lid, wid) = (id!(), id!(), id!(), id!(), id!());
    let (rta_u, buf_u, ptr_sb_buf_u, rta_f, buf_f, ptr_sb_buf_f) =
        (id!(), id!(), id!(), id!(), id!(), id!());
    let (v_w, v_xq, v_out, v_dm, v_xs, v_sx) = (id!(), id!(), id!(), id!(), id!(), id!());
    let (ptr_sb_u, ptr_sb_f, ptr_fn_u, ptr_fn_s, ptr_fn_f, glsl, main) =
        (id!(), id!(), id!(), id!(), id!(), id!(), id!());

    let mut const_words: Vec<u32> = Vec::new();
    let mut cvals: Vec<(u32, u32)> = Vec::new();
    let mut uconst = |v: u32| -> u32 {
        if let Some(&(_, cid)) = cvals.iter().find(|&&(val, _)| val == v) {
            return cid;
        }
        idc += 1;
        let cid = idc;
        cvals.push((v, cid));
        const_words.extend([(4 << 16) | OP_CONSTANT, uint, cid, v]);
        cid
    };
    let c_u0 = uconst(0);
    let c_u1 = uconst(1);
    let c_m = uconst(m);
    let c_m1 = uconst(m - 1); // the ring's last cell — the UMin modulus
    let c_blocks = uconst(blocks);
    let c_mask = uconst(byte_mask);
    let c_xstride = uconst(xstride);
    let c_bt = uconst(bt);
    let c_slab = uconst(local * rt); // rows owned by one workgroup
    let c_cpbm = uconst(cpb * m); //   weight words per 32-block, all rows
    let c_eight = uconst(8); //        activation words per 32-block (cpb·dots_per_word ≡ 8)
                             // literals folded into load immediate offsets: u·local (row), c·m (weight), c·dots+l (act)
    let c_ustep: Vec<u32> = (0..rt).map(|u| uconst(u * local)).collect();
    let c_cm: Vec<u32> = (0..cpb).map(|c| uconst(c * m)).collect();
    let c_xo: Vec<Vec<u32>> = (0..cpb)
        .map(|c| {
            (0..dots_per_word)
                .map(|l| uconst(c * dots_per_word + l))
                .collect()
        })
        .collect();
    let c_j: Vec<u32> = (0..bt).map(&mut uconst).collect();
    let shifts: Vec<u32> = (1..dots_per_word).map(|l| uconst(wbits * l)).collect();
    let f32_0 = id!();
    const_words.extend([(4 << 16) | OP_CONSTANT, float, f32_0, 0f32.to_bits()]);
    let sint_0 = id!();
    const_words.extend([(4 << 16) | OP_CONSTANT, sint, sint_0, 0]);

    let (b_entry, b_ohdr, b_ochk, b_obody, b_ocont, b_omerge) =
        (id!(), id!(), id!(), id!(), id!(), id!());
    let cvar = id!();
    let nt = (rt * bt) as usize;
    let acc: Vec<u32> = (0..nt).map(|_| id!()).collect(); // f32, live across the whole k walk
    let accq: Vec<u32> = (0..nt).map(|_| id!()).collect(); // i32, live within one 32-block

    let mut body: Vec<u32> = Vec::new();
    macro_rules! e {
        ($($x:expr),* $(,)?) => { body.extend([$($x),*]) };
    }
    // ── prologue: close the row key into the ring, form the per-lane bases ───────────────────────
    e!((2 << 16) | OP_LABEL, b_entry);
    e!((4 << 16) | OP_VARIABLE, ptr_fn_u, cvar, SC_FUNCTION);
    for a in &acc {
        e!((4 << 16) | OP_VARIABLE, ptr_fn_f, *a, SC_FUNCTION);
    }
    for a in &accq {
        e!((4 << 16) | OP_VARIABLE, ptr_fn_s, *a, SC_FUNCTION);
    }
    let (lidv, widv, lane, wgx, tile) = (id!(), id!(), id!(), id!(), id!());
    e!((4 << 16) | OP_LOAD, uvec3, lidv, lid);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, lane, lidv, 0);
    e!((4 << 16) | OP_LOAD, uvec3, widv, wid);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, wgx, widv, 0);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, tile, widv, 1);
    let (slab, row0) = (id!(), id!());
    e!((5 << 16) | OP_IMUL, uint, slab, wgx, c_slab);
    e!((5 << 16) | OP_IADD, uint, row0, slab, lane);
    let (mut row, mut rowc, mut live) = (Vec::new(), Vec::new(), Vec::new());
    for u in 0..rt as usize {
        let ru = if u == 0 {
            row0
        } else {
            let t = id!();
            e!((5 << 16) | OP_IADD, uint, t, row0, c_ustep[u]);
            t
        };
        // THE MANDALA CLOSURE: no key reaches outside the weight's row ring.
        let (rc, lv) = (id!(), id!());
        e!((7 << 16) | OP_EXT_INST, uint, rc, glsl, GLSL_UMIN, ru, c_m1);
        e!((5 << 16) | OP_ULESS_THAN, bool_t, lv, ru, c_m);
        row.push(ru);
        rowc.push(rc);
        live.push(lv);
    }
    let i0 = id!();
    e!((5 << 16) | OP_IMUL, uint, i0, tile, c_bt);
    let (mut ij, mut xbase, mut sbase) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..bt as usize {
        let iv = if j == 0 {
            i0
        } else {
            let t = id!();
            e!((5 << 16) | OP_IADD, uint, t, i0, c_j[j]);
            t
        };
        let (xb, sb) = (id!(), id!());
        e!((5 << 16) | OP_IMUL, uint, xb, iv, c_xstride);
        e!((5 << 16) | OP_IMUL, uint, sb, iv, c_blocks);
        ij.push(iv);
        xbase.push(xb);
        sbase.push(sb);
    }
    for a in &acc {
        e!((3 << 16) | OP_STORE, *a, f32_0);
    }
    e!((3 << 16) | OP_STORE, cvar, c_u0);
    e!((2 << 16) | OP_BRANCH, b_ohdr);
    // ── the walk over 32-blocks ─────────────────────────────────────────────────────────────────
    e!((2 << 16) | OP_LABEL, b_ohdr);
    e!((4 << 16) | OP_LOOP_MERGE, b_omerge, b_ocont, 0);
    e!((2 << 16) | OP_BRANCH, b_ochk);
    e!((2 << 16) | OP_LABEL, b_ochk);
    let (ochk_c, ocond) = (id!(), id!());
    e!((4 << 16) | OP_LOAD, uint, ochk_c, cvar);
    e!((5 << 16) | OP_ULESS_THAN, bool_t, ocond, ochk_c, c_blocks);
    e!((4 << 16) | OP_BRANCH_CONDITIONAL, ocond, b_obody, b_omerge);
    e!((2 << 16) | OP_LABEL, b_obody);
    let bb = id!();
    e!((4 << 16) | OP_LOAD, uint, bb, cvar);
    // three bases per block; every access below is one of these plus a literal
    let (bm, wbase, xbb) = (id!(), id!(), id!());
    e!((5 << 16) | OP_IMUL, uint, bm, bb, c_m);
    e!((5 << 16) | OP_IMUL, uint, wbase, bb, c_cpbm);
    e!((5 << 16) | OP_IMUL, uint, xbb, bb, c_eight);
    // the block's (d, lo) for each of the rt rows — one f16 word, shared by every column
    let (mut dv, mut mv, mut wr) = (Vec::new(), Vec::new(), Vec::new());
    for u in 0..rt as usize {
        let (di, pd, dmw, dmunp, d, lo) = (id!(), id!(), id!(), id!(), id!(), id!());
        e!((5 << 16) | OP_IADD, uint, di, bm, rowc[u]);
        e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, pd, v_dm, c_u0, di);
        e!((4 << 16) | OP_LOAD, uint, dmw, pd);
        e!((6 << 16) | OP_EXT_INST, vec2f, dmunp, glsl, 62, dmw); // UnpackHalf2x16
        e!((5 << 16) | OP_COMPOSITE_EXTRACT, float, d, dmunp, 0);
        e!((5 << 16) | OP_COMPOSITE_EXTRACT, float, lo, dmunp, 1);
        dv.push(d);
        mv.push(lo);
        let t = id!();
        e!((5 << 16) | OP_IADD, uint, t, wbase, rowc[u]);
        wr.push(t);
    }
    let mut xr = Vec::new();
    for j in 0..bt as usize {
        let t = id!();
        e!((5 << 16) | OP_IADD, uint, t, xbase[j], xbb);
        xr.push(t);
    }
    for a in &accq {
        e!((3 << 16) | OP_STORE, *a, sint_0);
    }
    // ── the FULLY UNROLLED weight-word loop: cpb words, each feeding rt·bt·dots_per_word dots ────
    for c in 0..cpb as usize {
        let mut wword = Vec::new();
        for u in 0..rt as usize {
            let (wi, pw, ww) = (id!(), id!(), id!());
            if c == 0 {
                e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, pw, v_w, c_u0, wr[u]);
            } else {
                e!((5 << 16) | OP_IADD, uint, wi, wr[u], c_cm[c]);
                e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, pw, v_w, c_u0, wi);
            }
            e!((4 << 16) | OP_LOAD, uint, ww, pw);
            wword.push(ww);
        }
        for l in 0..dots_per_word as usize {
            // the bt activation words — warp-uniform (their key came from WorkgroupId)
            let mut xw = Vec::new();
            for j in 0..bt as usize {
                let (xi, px, xv) = (id!(), id!(), id!());
                if c == 0 && l == 0 {
                    e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, px, v_xq, c_u0, xr[j]);
                } else {
                    e!((5 << 16) | OP_IADD, uint, xi, xr[j], c_xo[c][l]);
                    e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, px, v_xq, c_u0, xi);
                }
                e!((4 << 16) | OP_LOAD, uint, xv, px);
                xw.push(xv);
            }
            // phase l of each row's weight word: the codes at bit offset wbits·l, one per byte
            let mut src = Vec::new();
            for u in 0..rt as usize {
                src.push(if !split {
                    wword[u]
                } else if l == 0 {
                    let t = id!();
                    e!((5 << 16) | OP_BITWISE_AND, uint, t, wword[u], c_mask);
                    t
                } else {
                    let (sh, t) = (id!(), id!());
                    e!(
                        (5 << 16) | OP_SHIFT_RIGHT_LOGICAL,
                        uint,
                        sh,
                        wword[u],
                        shifts[l - 1]
                    );
                    e!((5 << 16) | OP_BITWISE_AND, uint, t, sh, c_mask);
                    t
                });
            }
            // the rt·bt outer product of this k-quad — the whole point of the two tiles
            for u in 0..rt as usize {
                for j in 0..bt as usize {
                    let (dot, a0, a1) = (id!(), id!(), id!());
                    let n = u * bt as usize + j;
                    e!(
                        (6 << 16) | OP_SUDOT,
                        sint,
                        dot,
                        xw[j],
                        src[u],
                        PACKED_VEC_4X8
                    );
                    e!((4 << 16) | OP_LOAD, sint, a0, accq[n]);
                    e!((5 << 16) | OP_IADD, sint, a1, a0, dot);
                    e!((3 << 16) | OP_STORE, accq[n], a1);
                }
            }
        }
    }
    // ── block epilogue: acc += (d·sx)·Σqx + lo·Σx, two FFMAs per output ─────────────────────────
    let (mut xsv, mut sxv) = (Vec::new(), Vec::new());
    for j in 0..bt as usize {
        let (idx, pxs, xv, psx, sv) = (id!(), id!(), id!(), id!(), id!());
        e!((5 << 16) | OP_IADD, uint, idx, sbase[j], bb);
        e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_f, pxs, v_xs, c_u0, idx);
        e!((4 << 16) | OP_LOAD, float, xv, pxs);
        e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_f, psx, v_sx, c_u0, idx);
        e!((4 << 16) | OP_LOAD, float, sv, psx);
        xsv.push(xv);
        sxv.push(sv);
    }
    for u in 0..rt as usize {
        for j in 0..bt as usize {
            let n = u * bt as usize + j;
            // The association is FIXED, not chosen: `((d·qf)·sx + lo·Σx)` is exactly the order the
            // decode gemv folds in, and the gate compares the two bit for bit. Reassociating to
            // `(d·sx)·qf` is the same three multiplies and two adds but a different rounding — it
            // costs nothing and it broke max-ulp against the decode path on every shape (measured:
            // ulp 0 → 5.5e3…2.2e6, |Δ| 0 → 4.8e-6). Bit-exactness is the cheaper of the two.
            let (qs, qf, dq, dqs, mxs, sum2, ab0, ab1) =
                (id!(), id!(), id!(), id!(), id!(), id!(), id!(), id!());
            e!((4 << 16) | OP_LOAD, sint, qs, accq[n]);
            e!((4 << 16) | OP_CONVERT_S_TO_F, float, qf, qs);
            e!((5 << 16) | OP_FMUL, float, dq, dv[u], qf);
            e!((5 << 16) | OP_FMUL, float, dqs, dq, sxv[j]);
            e!((5 << 16) | OP_FMUL, float, mxs, mv[u], xsv[j]);
            e!((5 << 16) | OP_FADD, float, sum2, dqs, mxs);
            e!((4 << 16) | OP_LOAD, float, ab0, acc[n]);
            e!((5 << 16) | OP_FADD, float, ab1, ab0, sum2);
            e!((3 << 16) | OP_STORE, acc[n], ab1);
        }
    }
    e!((2 << 16) | OP_BRANCH, b_ocont);
    e!((2 << 16) | OP_LABEL, b_ocont);
    let (ctr2, ctrn) = (id!(), id!());
    e!((4 << 16) | OP_LOAD, uint, ctr2, cvar);
    e!((5 << 16) | OP_IADD, uint, ctrn, ctr2, c_u1);
    e!((3 << 16) | OP_STORE, cvar, ctrn);
    e!((2 << 16) | OP_BRANCH, b_ohdr);
    // ── the store, the ONE place the clamped row still needs a predicate ─────────────────────────
    e!((2 << 16) | OP_LABEL, b_omerge);
    for u in 0..rt as usize {
        let (b_then, b_end) = (id!(), id!());
        e!((3 << 16) | OP_SELECTION_MERGE, b_end, 0);
        e!((4 << 16) | OP_BRANCH_CONDITIONAL, live[u], b_then, b_end);
        e!((2 << 16) | OP_LABEL, b_then);
        for j in 0..bt as usize {
            let (o1, ob, accf, pout) = (id!(), id!(), id!(), id!());
            e!((5 << 16) | OP_IMUL, uint, o1, ij[j], c_m);
            e!((5 << 16) | OP_IADD, uint, ob, o1, row[u]);
            e!((4 << 16) | OP_LOAD, float, accf, acc[u * bt as usize + j]);
            e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_f, pout, v_out, c_u0, ob);
            e!((3 << 16) | OP_STORE, pout, accf);
        }
        e!((2 << 16) | OP_BRANCH, b_end);
        e!((2 << 16) | OP_LABEL, b_end);
    }
    e!((1 << 16) | OP_RETURN);

    let bound = idc + 1;
    let mut w: Vec<u32> = Vec::new();
    w.extend([SPV_MAGIC, SPV_VERSION_1_3, 0, bound, 0]);
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_DOT_PRODUCT]);
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_DOT_INPUT_4X8_PACKED]);
    let ext = lit_string("SPV_KHR_integer_dot_product");
    w.push(((1 + ext.len() as u32) << 16) | OP_EXTENSION);
    w.extend(&ext);
    let g = lit_string("GLSL.std.450");
    w.push(((2 + g.len() as u32) << 16) | OP_EXT_INST_IMPORT);
    w.push(glsl);
    w.extend(&g);
    w.extend([(3 << 16) | OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450]);
    let mn = lit_string("main");
    w.push(((4 + mn.len() as u32 + 1) << 16) | OP_ENTRY_POINT);
    w.extend([MODEL_GL_COMPUTE, main]);
    w.extend(&mn);
    w.extend([lid, wid]);
    w.extend([
        (6 << 16) | OP_EXECUTION_MODE,
        main,
        MODE_LOCAL_SIZE,
        local,
        1,
        1,
    ]);
    w.extend([
        (4 << 16) | OP_DECORATE,
        lid,
        DEC_BUILT_IN,
        BUILTIN_LOCAL_INVOCATION_ID,
    ]);
    w.extend([
        (4 << 16) | OP_DECORATE,
        wid,
        DEC_BUILT_IN,
        BUILTIN_WORKGROUP_ID,
    ]);
    w.extend([(4 << 16) | OP_DECORATE, rta_u, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, buf_u, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, buf_u, 0, DEC_OFFSET, 0]);
    w.extend([(4 << 16) | OP_DECORATE, rta_f, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, buf_f, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, buf_f, 0, DEC_OFFSET, 0]);
    for (var, binding) in [
        (v_w, 0u32),
        (v_xq, 1),
        (v_out, 2),
        (v_dm, 3),
        (v_xs, 4),
        (v_sx, 5),
    ] {
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_BINDING, binding]);
    }
    w.extend([(2 << 16) | OP_TYPE_VOID, void]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, fnty, void]);
    w.extend([(4 << 16) | OP_TYPE_INT, uint, 32, 0]);
    w.extend([(4 << 16) | OP_TYPE_INT, sint, 32, 1]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, float, 32]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, bool_t]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, uvec3, uint, 3]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, vec2f, float, 2]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_in_uvec3, SC_INPUT, uvec3]);
    w.extend([(4 << 16) | OP_VARIABLE, ptr_in_uvec3, lid, SC_INPUT]);
    w.extend([(4 << 16) | OP_VARIABLE, ptr_in_uvec3, wid, SC_INPUT]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, rta_u, uint]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, buf_u, rta_u]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_buf_u,
        SC_STORAGE_BUFFER,
        buf_u,
    ]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, rta_f, float]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, buf_f, rta_f]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_buf_f,
        SC_STORAGE_BUFFER,
        buf_f,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_u,
        v_w,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_u,
        v_xq,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_f,
        v_out,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_u,
        v_dm,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_f,
        v_xs,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_f,
        v_sx,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_u,
        SC_STORAGE_BUFFER,
        uint,
    ]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_f,
        SC_STORAGE_BUFFER,
        float,
    ]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_fn_u, SC_FUNCTION, uint]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_fn_s, SC_FUNCTION, sint]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_fn_f, SC_FUNCTION, float]);
    w.extend(&const_words);
    w.extend([(5 << 16) | OP_FUNCTION, void, main, 0, fnty]);
    w.extend(&body);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The 2-D register-tiled integer-dot GEMM as bytes.
pub fn gemv_dp4a_batch_rt_bytes(
    m: u32,
    k: u32,
    local: u32,
    carrier: Carrier,
    bt: u32,
    rt: u32,
) -> Vec<u8> {
    gemv_dp4a_batch_rt_module(m, k, local, carrier, bt, rt)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod batch_rt_plane_tests {
    use super::gemv_dp4a_batch_rt_module;
    use super::Carrier;

    /// **A two-plane chart is REFUSED, not silently folded on plane A alone.**
    ///
    /// `gemv_dp4a_batch_rt_module` walks one code plane; `Carrier::Q5` (and `Carrier::Q3`) carry the
    /// high bit in a second plane appended to `w`, so plane B would be read as the next row's plane A
    /// and the kernel would emit a plausible wrong number. The cooperative-matrix chart GEMM is their
    /// prefill path; this emitter must refuse and name them. CPU-only: it only builds SPIR-V words.
    #[test]
    #[should_panic(expected = "two-plane chart")]
    fn q5_batch_rt_module_is_refused() {
        let _ = gemv_dp4a_batch_rt_module(64, 128, 128, Carrier::Q5, 8, 4);
    }

    /// The control: a single-plane chart still emits a module, so the refusal above is specific to
    /// the two-plane carriers rather than a blanket refusal of chart carriers.
    #[test]
    fn q4_batch_rt_module_still_emits() {
        let b = gemv_dp4a_batch_rt_module(64, 128, 128, Carrier::Q4, 8, 4);
        assert!(b.len() > 1024, "Q4 emitted nothing usable");
        assert_eq!(b[0], 0x0723_0203, "Q4 must begin with the SPIR-V magic");
    }
}

/// **The SUPERBLOCK integer-dot GEMM — the float epilogue collapsed 8×.**
///
/// [`gemv_dp4a_batch_rt_module`] carries one affine `(d, lo)` f16 pair per **32** weights, so it
/// leaves the integer ring every 8 `OpSUDot`s: convert, three `FMul`, two `FAdd` — 6 float ops per
/// 8 dots, which is what capped that design near 68 TOPS against the measured 133 TOPS DP4A
/// ceiling. This kernel carries ONE f16 scale per **256** weights and the symmetric chart
/// `w = s·(code − 1)`, so the whole superblock — 16 weight words, **64 `OpSUDot`s** — folds into a
/// single i32 accumulator and pays **one** epilogue:
///
/// ```text
///   Σ w·x  =  s · sx · ( Σ code·q  −  Σ q )          per 256 weights
///             ^^^^^^^   ^^^^^^^^^^   ^^^^
///             2 FMul    the i32 accumulator   a PRECOMPUTED i32 (one OpISub, not a float term)
/// ```
///
/// Ops per output per superblock: 1 `OpISub`, 1 `ConvertSToF`, 2 `FMul`, 1 `FAdd` — 4 float ops per
/// 64 dots against 6 per 8. **0.0625 float-ops/dot against 0.75, a 12× collapse**; the scale/offset
/// loads collapse 8× with it (`rt + 2·bt` per 256 k-elements instead of per 32).
///
/// This is the design law of `song`'s `torusdot` (`emit_faces.slang:1036-1059`) at a wider block:
/// an INTEGER accumulator is order-free, so partials combine bit-exactly, and the float scale is
/// applied once on the way out. `song`'s own per-block variant `gridblock` names the cost this
/// kernel is reducing — *"cross-block order-freedom is traded (the float `addsd` reassociates) for
/// outlier precision… the ONLY floats are the per-block rescale `mulsd` and the ONE cross-block
/// `addsd` per block"* — and the superblock carrier pays that toll 8× less often.
///
/// **The chart is `servefold`'s `Proj::rowq2s`/`Proj::dsb`, transposed for coalescing and NOT
/// otherwise changed** (`q2sb_chart`, llama.cpp's `block_tq2_0` shape). Byte `i` of a 128-weight
/// unit holds weight `unit·128 + 32p + i` at bit offset `2p`, so one u32 of the row carries four
/// consecutive weights per plane `p` — which means **the activation operand is in NATURAL order**,
/// 4 int8 per u32, with none of the deinterleave the per-32 chart needs. Weight word `W` of a
/// superblock (`W = 8·unit + q`) pairs with activation word `32·unit + 8·p + q`.
///
/// Bindings (6, all `std430` runtime arrays):
/// ```text
///   0  w    u32   codes,      w[(sb·16 + W)·m + r]   — row-major over ROWS: lane r reads lane r+1's
///                                                       neighbour word, one coalesced 128-B request
///   1  xq   u32   int8 acts,  xq[i·(k/4) + sb·64 + 32·unit + 8·p + q]   (natural order)
///   2  out  f32   out[i·m + r]
///   3  dsb  u32   f16 scale in the LOW half, dsb[sb·m + r]  (UnpackHalf2x16 → .x = `widen_f16`)
///   4  xs   f32   activation scale per (column, superblock): xs[i·nsb + sb]
///   5  ys   u32   Σq  per (column, superblock) as i32 bits: ys[i·nsb + sb]  (OpBitcast, no cost)
/// ```
/// `dispatch = (ceil(m/(local·rt)), b/bt)`, exactly as the per-32 kernel.
///
/// **The odd tail is emitted, not branched.** `k % 256 ≠ 0` is admissible (`q2sb_fits` only needs
/// `k % 128 == 0`; qwen2.5-0.5B's `k = 896` is 7 units ⇒ 3 full superblocks + one 128-wide one), so
/// the module emits `full = nunits/2` iterations of a two-unit body plus, when `nunits` is odd, a
/// straight-line one-unit body — the same split `row_dot_q2sb_avx2` makes, and for the same reason:
/// no per-superblock branch inside the loop.
///
/// **The mandala closure** (`MandalaMemory.access_in_region`, `masked_load_never_oob`). The row key
/// is closed into the weight's row ring by `UMin(row, m−1)` before it addresses `w` or `dsb` — the
/// non-power-of-two form of `key &&& (2ᵏ−1)`, one instruction, no branch and no check to elide; only
/// the final store keeps a predicate, because a clamped row must not write. Every other index is a
/// loop counter with a compile-time bound or a literal. The batch key is left bounded by dispatch
/// construction rather than by mask, for the same reason `gemv_dp4a_batch_rt_module` left it:
/// closing it needs the activation and logit buffers padded to a power of two, 622 MB at the 0.5B
/// lm head against 443 MB for the whole model. Stated, not hidden.
///
/// **Why the multiply-free ternary form (`TernaryDP4A.ternary_dot_two_mask`) is not used more
/// directly.** The trinity dot IS `Σ_{w=+1} x − Σ_{w=−1} x`, and that is exactly what this kernel
/// computes: `q2_affine_dequant_exact` says the symmetric chart `s·(code−1)` at `code = trit+1` is
/// the ternary weight with zero residual, and `Σ code·q − Σ q` is the two-mask difference with the
/// `code = 1` (monad) lanes cancelling. Carrying the two masks as separate 1-bit planes instead
/// would need an 8-bit `{0,1}` operand per weight for `OpSUDot` to consume — 4× the bytes of the
/// 2-bit code — so the packed code plus one precomputed `Σq` is the two-mask form at its floor.
///
/// **MEASURED** (`modelfold --example prefill_sb`, RTX 5080, qwen2.5-0.5B shapes, GPU-active device
/// timestamps, min of 7; box carrying another agent's CPU job at load ~7, GPU otherwise idle).
/// Both kernels swept over the whole admissible tile grid, so neither is handicapped:
/// ```text
///   whole-stack projection tok/s   b=1    b=8     b=64     b=256    b=640
///     superblock, STAGED (today)  1603  11637    43328    60263    70526
///     superblock, unstaged        1418   9806    34938    49242    61485
///     per-32 (e882143 incumbent)  1058   7483    25377    33240    42434
///   best shape TOPS at b=640, staged: 77.1 (lm_head), 72.8 gateup, 63.1 down, 56.7 o, 54.8 qkv,
///     against the unstaged 74.0 / 66.6 / 48.8 / 49.9 / 39.9 and the per-32 kernel's 38.0 / 51.5.
///   projection chart VRAM over the 24-layer stack: 126.0 MB against 294.4 MB (57.2% less).
/// ```
/// **The issue-slot budget, now COUNTED IN THE SASS instead of estimated.** The hand budget above
/// this line used to read "3232 slots for 8192 MACs = 2.54 MACs/slot against the per-32 kernel's
/// 1.84". Both numbers were optimistic by ~1.7×. The driver's own SM120 stream
/// (`modelfold --example simt_sass stack` → `scripts/nvuc_sass.py`, hot loop delimited by its
/// backward `BRA`) counts, per thread per superblock:
///
/// ```text
///   this kernel, bt8×rt4×uu2   2048 IDP.4A.S8.U8  in  5583 slots  =  1.47 MACs/slot
///   this kernel, bt4×rt8×uu1   1024 IDP.4A.S8.U8  in  2578 slots  =  1.59   (the wired gateup tile)
///   this kernel, bt4×rt8×uu2   2048 IDP.4A.S8.U8  in  5362 slots  =  1.53   (the wired down tile)
///   gemv_dp4a_batch_rt bt8×rt4  256 IDP.4A.S8.U8  in  1063 slots  =  0.96
/// ```
///
/// so the ratio is **1.53–1.66×**, and the measured 1.44–1.45× standalone speedup at b = 640 sits
/// just under it. The float collapse is real and is the whole of the win: 96 float slots
/// (32 `I2FP.F32.S32` + 32 `FMUL.FTZ` + 32 `FFMA.FTZ`) per 8192 MACs here against 160
/// (+64 `FADD.FTZ`) per 1024 there — **0.0117 against 0.156 float-ops per MAC, a 13.3× collapse**,
/// which is the claimed 12× measured at the metal. `IDP.4A.S8.U8` is what the ternary dot compiles
/// to — no float path.
///
/// **RETRACTED, by opcode census of the same window: "no separate accumulate slot, so the `OpIAdd`
/// really is folded into the addend operand".** It is not folded. The exact mix of the wired
/// `gateup` bt4×rt8×uu1 loop (2714 slots) is `IDP.4A.S8.U8` ×1024 **with `RZ` as the addend**,
/// `IADD3` ×524, `LOP3.LUT` ×256, `ISETP.GE.U32.AND` ×208, `LDG.E` ×208, `SHF.R.U32.HI` ×192,
/// `LEA` ×183. The driver keeps the dots independent for ILP and combines them in a three-input add
/// tree (`IADD3 Racc, Racc, Rdot1, Rdot2`, two dots per add) — so **524 of the 707 `IADD3+LEA` are
/// the ACCUMULATE, not address arithmetic**, and the real address term is 183 `LEA` + ~50 `IMAD`.
/// `OpSUDotAccSat` IS worth reaching for on a device that accelerates it, and this one does not —
/// see [`sb_accsat`] for the measurement that closes it.
///
/// **What eats the other 62%, from the same disassembly** (bt4×rt8×uu1, 2714-slot loop):
/// 524 `IADD3` = 19.3% the DOT ACCUMULATE (above); 256 `LOP3.LUT` + 192 `SHF.R.U32.HI` = 16.5%
/// the 2-bit code extract (`& 0x03030303` and the `>> 2p` plane shift); 183 `LEA` = 6.7% address
/// arithmetic; 208 `LDG.E` = 7.7%; and 208 `ISETP.GE.U32.AND` = 7.7% —
/// **exactly one descriptor-range compare per global load**, which
/// the driver inserts unconditionally (`pEnabledFeatures` is null and the device reports
/// `defaultRobustnessStorageBuffers = DISABLED`, so this is NVIDIA guarding the descriptor, not
/// `robustBufferAccess`). That guard is `MandalaMemory`'s point inverted: the law says the bound and
/// the address are ONE instruction (`base + (key &&& (2^k−1))`, `masked_load_never_oob`), and here
/// the driver is paying a second one because a `logical` `OpAccessChain` into an `OpTypeRuntimeArray`
/// gives it a dynamic element index it cannot fold into an immediate. The 128 of those 208 loads that
/// are ACTIVATION words are also the reuse ceiling — each feeds `rt` dots and nothing more, because
/// this kernel allocated **zero** workgroup shared memory at bt4×rt8 (driver `Shared Memory Size` 0).
///
/// **AND THAT ONE IS NOW TAKEN — [`sb_lds`], the round's win.** The activation key carries no lane
/// term, so all `local` threads were re-fetching the same words; staged once per superblock into a
/// power-of-two shared ring the read becomes `LDS`, and the driver VECTORIZES it (`LDS.128`, four
/// words per instruction) because the shared offsets are consecutive compile-time literals where
/// the global ones each carried their own `!Pn` predicate and could not be merged. At the wired
/// bt4×rt4×uu2 tile: **3125 → 2551 slots, 1.31 → 1.61 MACs/slot**, `ISETP` 333 → 80, `IADD3+LEA`
/// 835 → 567, `LDG` 332 → 78, registers 168 → 128 with no spill. `song`'s
/// `emit_parallel.slang:995-1014` is the shape (the order-free integer partial, combined once).
///
/// **THE GUARD WAS REMOVED, AND IT WAS NOT THE WALL — [`gemv_dp4a_sb_pb_module`] settles both
/// halves of the paragraph above.** Rebuilding the same kernel on `PhysicalStorageBuffer` pointers
/// (no descriptor ⇒ no range ⇒ nothing to compare) takes the inner-loop `ISETP` from 210 to **8** at
/// `bt4×rt8×uu1` and from 401 to **7** at `bt4×rt8×uu2`, exactly as predicted. The other half of the
/// prediction — that the `IADD3`/`LEA` chain would collapse with it — is **FALSE**: 26.1 % → 26.7 %
/// at gateup, 28.7 % → 31.4 % at qkv. The address arithmetic was never the descriptor's; it is this
/// chart's own index algebra (`sb·16m + rowc + wq·m`, and `xbase + xsb + lit`), and it survives the
/// pointer change untouched. Net at the metal: standalone 61 156 → 67 323 tok/s at b = 640, wired
/// **−10 to −14 %** — the driver caps the physical face at 128 registers where the logical face got
/// 166, so every `rt = 8` shape loses loads-in-flight. `servefold::gpu::sb_pb` carries the table.
///
/// **Against llama.cpp, honestly — re-disassembled, and one of our own "confirmed facts" was
/// wrong.** `cuobjdump -sass` on the sm_120a build's `mmq-instance-*.cu.o`, densest kernel of each
/// (re-run 2026-08-16 against the same objects; all three counts below reproduce exactly):
///
/// ```text
///   mul_mat_q<Q4_0>   256 IMMA.16832.S8.S8 in  7 520 instructions = 139.4 MACs/issue-slot (per warp)
///   mul_mat_q<Q2_K>   576 IMMA             in 12 336              = 191.3               (per warp)
///   gemv_dp4a_sb      3072 IDP.4A.S8.U8    in  9 255              =  42.5               (per warp)
/// ```
///
/// **Convention, named: MACs per WARP per issue-slot** — matches `results/moe-ir-reference.md`
/// §"Correction to the corpus". `IMMA.16832` does `16·8·32 = 4096` int8 MACs per warp for one static
/// instruction; `IDP.4A`/`OpSUDot` does 4 int8 MACs per THREAD, i.e. `4·32 = 128` per warp once every
/// lane's copy of the instruction is counted — the earlier "1.33 (whole kernel)" figure here divided
/// per-thread MACs by instruction count, which is 32× too small for a warp-to-warp comparison. On
/// this one convention the density gap is **3.3× (Q4_0) and 4.5× (Q2_K)** — not the previously
/// recorded ~105–144×, which mixed the two conventions. It is still not closeable with a dot
/// instruction. Confirmed unchanged: 0 `IDP.4A` and **0 `LDGSTS`** in every MMQ kernel — no
/// `cp.async`, no double buffering.
///
/// CORRECTED: the claim that "llama.cpp has no MMQ kernel for a 2-bit type at all" is FALSE. The
/// build tree carries `mmq-instance-q2_k`, `-iq2_s`, `-iq2_xs`, `-iq2_xxs`, and `mul_mat_q<Q2_K>`
/// disassembles to 4120 IMMA over its 64 kernels. What is genuinely absent is the TERNARY type:
/// there is no `mmq-instance-tq2_0` or `-tq1_0`. So the bpw advantage this chart has over the
/// Q4_K_M bar (2.125 against 5.5) is real, but it is not "they cannot do 2-bit" — it is "they do
/// not do 1.58-bit-class ternary", which is a narrower and truer statement.
///
/// `uu` = 128-weight units unrolled per inner-loop iteration (1 or 2). `uu = 2` unrolls the whole
/// superblock and removes the inner loop entirely; `uu = 1` keeps a 2-trip loop and a smaller body.
/// Neither wins everywhere — the driver reports 128 registers and a 148 KB binary at `bt8×rt4×uu2`
/// against 106 KB at `uu1` and 20 KB for the per-32 kernel at the same tile, with **zero spill on
/// every tile measured** — so both are emitted and the bench picks per shape.
///
/// **`segs` = SPLIT-K, and it is what makes this kernel usable at DECODE.** At `b = 1` a 896-row
/// projection is 7 workgroups at `local = 128` and cannot fill 84 SMs; the served decode gemv
/// (`gemv_dp4a_dyn`, `servefold::segs`) splits `k` instead, and without that this kernel is 4.3×
/// slower than the incumbent on the `down` shape (measured, `prefill_sb`). With `segs > 1`:
/// `WorkgroupId.y` is the SEGMENT, not the batch column, so `bt` must be 1; segment `g` walks
/// superblocks `[g·per, min((g+1)·per, full))` with `per = ceil(full/segs)`; the odd tail unit is
/// predicated onto the LAST segment; and the write lands at `out[g·m + r]`, i.e. the `segs·m`
/// partials the existing stage-2 `gemv(F32)` reduction already consumes. The activation base is
/// pinned to column 0 rather than to `WorkgroupId.y`, which is the one line that differs.
pub fn gemv_dp4a_sb_module(
    m: u32,
    k: u32,
    local: u32,
    bt: u32,
    rt: u32,
    uu: u32,
    segs: u32,
    carrier: Carrier,
) -> Vec<u32> {
    gemv_dp4a_sb_module_ex(m, k, local, bt, rt, uu, segs, false, carrier)
}

/// **`SB_ACCSAT` — fold the dot's accumulate into the dot instruction. BUILT, DISASSEMBLED, and
/// DEFAULT OFF because the hardware does not have the instruction.**
///
/// The first half of the finding stands and corrects this file's own earlier claim. The wired
/// `gemv_dp4a_sb.gateup` inner loop is 2714 slots, and its exact census is
/// `IDP.4A.S8.U8 Rd, Ra, Rb, **RZ**` ×1024 beside `IADD3` ×524 — the driver does NOT fold
/// `OpIAdd`-into-`accq` into the dot's addend; it builds a three-input add tree next to it
/// (`IADD3 Racc, Racc, Rdot1, Rdot2`, two dots per add). So **524 of the 707 `IADD3+LEA` this file
/// used to call "address arithmetic" are the ACCUMULATE**, 19.3 % of every issue slot.
///
/// The fix does not exist at the metal. `OpSUDotAccSat` is legal SPIR-V and spirv-val accepts it,
/// but SM120 has no accumulating IDP4A: the driver emits `IDP.4A.S8.U8 Rd, Ra, Rb, RZ` **and then
/// `VIADD.S32.ISAT`** — one saturating add per dot, 1024 of them, where the plain form got 512
/// three-input adds. Measured on the same kernel, same tile: 2714 → 3140 slots with the staging
/// win already in it (the accumulate term went 524 → 1024 + 42). Vulkan reports the reason ahead of
/// time and it was not read: `integerDotProductAccumulatingSaturating4x8BitPackedMixedSignedness
/// Accelerated` is FALSE on this device while the non-accumulating one is TRUE. Kept behind the
/// dial rather than deleted — it is one line, and a device that accelerates it wins 512 slots.
pub fn sb_accsat() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SB_ACCSAT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
            != 0
    })
}

/// **`SB_LDS` — stage the activation tile in workgroup shared memory.** Default ON.
///
/// The activation word's key is `(WorkgroupId.y·bt + j)·xstride + …`: it carries NO lane term, so
/// all `local` threads of the workgroup load the SAME `32·bt` words per unit out of global, each
/// paying `LEA` + `ISETP.GE.U32` (the descriptor range compare) + `LDG.E` — three issue slots per
/// load, 384 of the 2714-slot loop at `bt = 4` and 768 of 2812 at `bt = 8`. Staged once per
/// superblock into shared, the read becomes `LDS [Rn+imm]`: no range to compare (shared has no
/// descriptor) and no address chain (every offset is a compile-time literal, because the shared
/// re-index `j·64 + t·32 + p·8 + q` drops the workgroup term entirely).
///
/// **The mandala closure is exact here, not "bounded by dispatch".** The shared extent is the
/// power-of-two `cap = 64·bt` words, so `MandalaMemory.access` applies verbatim:
/// `base + (key &&& (cap−1))`, `no_key_escapes`, and the read index is a compile-time literal
/// `< cap` by the emitter's own arithmetic (`j ≤ bt−1`, `t ≤ 1`, `p ≤ 3`, `q ≤ 7` ⇒ `≤ 64bt−1`).
/// The one runtime key — the staging thread's `lane + it·local` — is closed twice: `UMin` to the
/// last slot (so an over-wide workgroup re-stores the LAST slot's own correct value rather than
/// escaping) and the `&&& (cap−1)` mask on the store index.
///
/// Off when `segs > 1`: split-K is the DECODE path at `bt = 1`, where there is no tile to share and
/// the two barriers per superblock would be pure loss.
pub fn sb_lds() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SB_LDS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1)
            != 0
    })
}

/// **The same kernel with the descriptor removed — `PhysicalStorageBuffer` pointers.**
///
/// `pb = true` swaps the six `StorageBuffer` variables for ONE binding holding their six `u64` GPU
/// addresses (`VK_KHR_buffer_device_address`, `Gpu::bind_addrs`). The kernel `OpConvertUToPtr`s each
/// address once in the entry block and every access becomes an `OpAccessChain` on a raw pointer,
/// loaded with the `Aligned 4` memory operand.
///
/// **Why, and it is the law and not a micro-optimization.** A descriptor carries a base AND a range,
/// and this driver spends one `ISETP.GE.U32.AND` per global load comparing the index against that
/// range — 192 of them in the `bt4×rt8×uu1` hot loop, 7.4 % of issue slots, with
/// `robustBufferAccess` DISABLED. `MandalaMemory` says the bound and the address are ONE instruction:
/// `access r key = base + (key &&& (2^k − 1))`, and `masked_load_never_oob` holds because the mask
/// closed the ring, not because anything compared afterwards. A logical `OpAccessChain` into an
/// `OpTypeRuntimeArray` makes the driver pay the second instruction anyway. With a raw address there
/// is no range to compare, so the mask is the only bound — the law, at the metal.
///
/// **WHERE THE MASK ACTUALLY IS, stated site by site, because "no descriptor" means the kernel owns
/// the closure:**
/// - **row (`v_w`, `v_dm`, `v_out`)** — `rowc = UMin(row, m−1)` (`GLSL_UMIN`), the mandala closure
///   proper: the key is `WorkgroupId.x · local · rt + lane + u · local`, and no value of it reaches
///   outside the row ring. The `v_out` store additionally predicates on `row < m`, so a clamped row
///   writes nothing.
/// - **superblock (`sb`)** — the loop bound is the COMPILE-TIME constant `full = k/256`; no host
///   value enters, so `sb·16m + rowc + wq·m` is closed by construction over the chart's own extent.
/// - **batch column (`WorkgroupId.y`)** — **BOUNDED BY DISPATCH, and this is the one site left
///   open.** `xbase = (tile·bt + j)·xstride` indexes the activation scratch, whose length the HOST
///   picked (`PREFILL_BT_MAX`-padded rows). Closing it needs the scratch capacity as a mask operand,
///   which the caller already computes as a power of two (`qcap`/`bcap` in `servefold::gpu` are
///   `next_power_of_two`) — but the kernel does not take it yet, so today the guarantee is the
///   caller's `groups.y = ceil(n/bt)` and `n ≤ maxb`, not a mask. Named, not hidden.
pub fn gemv_dp4a_sb_pb_module(
    m: u32,
    k: u32,
    local: u32,
    bt: u32,
    rt: u32,
    uu: u32,
    segs: u32,
    carrier: Carrier,
) -> Vec<u32> {
    gemv_dp4a_sb_module_ex(m, k, local, bt, rt, uu, segs, true, carrier)
}

/// The superblock integer-dot GEMM in its `PhysicalStorageBuffer` form, as bytes.
/// See [`gemv_dp4a_sb_pb_module`].
pub fn gemv_dp4a_sb_pb_bytes(
    m: u32,
    k: u32,
    local: u32,
    bt: u32,
    rt: u32,
    uu: u32,
    segs: u32,
    carrier: Carrier,
) -> Vec<u8> {
    gemv_dp4a_sb_pb_module(m, k, local, bt, rt, uu, segs, carrier)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// The body of [`gemv_dp4a_sb_module`] / [`gemv_dp4a_sb_pb_module`] — one emitter, two faces. Only
/// the six buffers' ADDRESSING differs (descriptor variables vs converted device addresses); every
/// index, every dot and every rounding decision is the same code.
pub fn gemv_dp4a_sb_module_ex(
    m: u32,
    k: u32,
    local: u32,
    bt: u32,
    rt: u32,
    uu: u32,
    segs: u32,
    pb: bool,
    carrier: Carrier,
) -> Vec<u32> {
    assert!(
        k % 128 == 0,
        "the superblock chart needs k % 128 == 0 (q2sb_fits)"
    );
    assert!(bt >= 1 && rt >= 1, "both tiles must be >= 1");
    assert!(uu == 1 || uu == 2, "uu is the unit unroll: 1 or 2");
    assert!(segs >= 1, "segs is the split-K count");
    assert!(
        segs == 1 || bt == 1,
        "split-K takes WorkgroupId.y, so it cannot also carry a batch tile"
    );
    assert!(m >= 1);
    // ── the CODE WIDTH's geometry, derived rather than hardcoded ──────────────────────────────
    //
    // This emitter was Q2-only: `0x03030303`, `>> 2p`, four planes, eight words per unit, sixteen
    // per superblock. Every one of those is `Carrier::pack()` in disguise, and at Q2 two distinct
    // quantities coincide at 8 — words-per-unit-per-row (`128/pack`) and activation-words-per-plane
    // (`32/planes`) — which is why the literal read as one number.
    //
    //   width  = 32/pack     bits per code            Q2 2   Q4 4   Q8 8
    //   planes = pack/4      codes per byte lane      Q2 4   Q4 2   Q8 1
    //   wpu    = 128/pack    weight words per unit    Q2 8   Q4 16  Q8 32
    //
    // `planes·width = 8` is the invariant: one byte lane holds exactly `planes` codes, and the
    // DP4A consumes four lanes per dot at every width. At `Carrier::Q2` every constant below
    // reproduces its previous literal EXACTLY, which is what the SEQHASH gate checks.
    //
    // ⚠ **Expressible is not usable.** The packer's superblock face is `q2sb_chart`, which asserts
    // `code <= 2` and lays `s·(code − 1)` with ONE stream and no `lo`. Its licence is
    // `native_ternary_zero_residual`: a 256-wide scale costs a NATIVELY TERNARY tensor nothing
    // because the scale is per-output-row and constant over any partition. That argument does not
    // extend to Q4/Q8, and no `rowq4s`/`rowq8s` chart exists to feed them. Widening the emitter is
    // therefore necessary but not sufficient, and the missing half is a FORMAT question (what a
    // symmetric 4-bit 256-wide block costs on a non-ternary tensor) rather than a wiring one.
    assert!(
        matches!(carrier, Carrier::Q2 | Carrier::Q4 | Carrier::Q8),
        "the superblock face needs a byte-divisible chart width; {carrier:?} has none \
(Q3 is two planes and would need its high-bit plane threaded through the phase loop too)"
    );
    let pack = carrier.pack();
    let width = 32 / pack; //  bits per code
    let planes = pack / 4; //  codes per byte lane
    let wpu = (128 / pack) as usize; // weight words per 128-weight unit, per row
    const OP_ISUB: u32 = 130;
    const OP_IEQUAL: u32 = 170;
    let nunits = k / 128; //  128-weight plane units
    let full = nunits / 2; // superblocks holding the full 2 units
    let odd = nunits % 2; // a trailing single-unit superblock
    let nsb = k.div_ceil(256); // = full + odd, the scale stream's stride
    let xstride = k / 4; //     u32 activation words per column (4 int8 each)

    let mut idc = 0u32;
    macro_rules! id {
        () => {{
            idc += 1;
            idc
        }};
    }
    let (void, fnty, uint, sint, float, bool_t) = (id!(), id!(), id!(), id!(), id!(), id!());
    let (uvec3, vec2f, ptr_in_uvec3, lid, wid) = (id!(), id!(), id!(), id!(), id!());
    let (rta_u, buf_u, ptr_sb_buf_u, rta_f, buf_f, ptr_sb_buf_f) =
        (id!(), id!(), id!(), id!(), id!(), id!());
    let (v_w, v_xq, v_out, v_dm, v_xs, v_ys) = (id!(), id!(), id!(), id!(), id!(), id!());
    let (ptr_sb_u, ptr_sb_f, ptr_fn_u, ptr_fn_s, ptr_fn_f, glsl, main) =
        (id!(), id!(), id!(), id!(), id!(), id!(), id!());
    // the PhysicalStorageBuffer face: a u64 address table at binding 0, and raw pointers into the
    // SAME block types the descriptor face uses (so the layout the driver sees is identical)
    let (ulong, rta_ul, buf_ul, ptr_sb_buf_ul, ptr_sb_ul, v_addr) =
        (id!(), id!(), id!(), id!(), id!(), id!());
    let (ptr_pb_buf_u, ptr_pb_buf_f, ptr_pb_u, ptr_pb_f) = (id!(), id!(), id!(), id!());
    // the pointer types the BODY uses — one face or the other, chosen once
    let (pty_u, pty_f) = if pb {
        (ptr_pb_u, ptr_pb_f)
    } else {
        (ptr_sb_u, ptr_sb_f)
    };
    // the LDS staging face: one shared u32 array of `stage_cap` words + its element pointer type
    let (arr_stage, ptr_arr_stage, v_stage, ptr_wg_u) = (id!(), id!(), id!(), id!());
    // `SB_LDS`, but never under split-K (bt = 1 there: nothing to share, two barriers to pay)
    let lds = sb_lds() && segs == 1;
    let accsat = sb_accsat();
    // the shared ring, a power of two so the mandala mask is one `and` (`MandalaMemory`)
    let stage_cap = (64 * bt).next_power_of_two();

    let mut const_words: Vec<u32> = Vec::new();
    let mut cvals: Vec<(u32, u32)> = Vec::new();
    let mut uconst = |v: u32| -> u32 {
        if let Some(&(_, cid)) = cvals.iter().find(|&&(val, _)| val == v) {
            return cid;
        }
        idc += 1;
        let cid = idc;
        cvals.push((v, cid));
        const_words.extend([(4 << 16) | OP_CONSTANT, uint, cid, v]);
        cid
    };
    let c_u0 = uconst(0);
    let c_u1 = uconst(1);
    // the address table's six slots, in binding order: w, xq, out, dm, xs, ys
    let c_slot: Vec<u32> = (0..6).map(&mut uconst).collect();
    let c_m = uconst(m);
    let c_m1 = uconst(m - 1); // the ring's last cell — the UMin modulus
    let c_full = uconst(full);
    // the code mask in every byte lane: 0x03030303 (Q2), 0x0F0F0F0F (Q4), 0xFFFFFFFF (Q8)
    let lane = (1u32 << width) - 1;
    let c_mask = uconst(lane * 0x0101_0101);
    let c_xstride = uconst(xstride);
    let c_nsb = uconst(nsb);
    let c_bt = uconst(bt);
    let c_slab = uconst(local * rt); // rows owned by one workgroup
    let c_16m = uconst(2 * wpu as u32 * m); // weight words per superblock, all rows (Q2: 16m)
    let c_64 = uconst(64); //          activation words per superblock
    let c_8m = uconst(wpu as u32 * m); //   weight words per unit, all rows (Q2: 8m)
    let c_32 = uconst(32); //          activation words per unit
    let c_ustep: Vec<u32> = (0..rt).map(|u| uconst(u * local)).collect();
    // literals folded into the load's immediate offset: W·m (weight), 32·unit + 8·p + q (activation)
    let c_wo: Vec<u32> = (0..2 * wpu as u32).map(|wq| uconst(wq * m)).collect();
    // `32/planes == wpu` (both are `4·width`), so the activation's per-plane stride is `wpu` too
    let c_xo: Vec<Vec<u32>> = (0..2 * wpu as u32)
        .map(|wq| {
            (0..planes)
                .map(|p| uconst((wq / wpu as u32) * 32 + p * wpu as u32 + (wq % wpu as u32)))
                .collect()
        })
        .collect();
    let c_j: Vec<u32> = (0..bt).map(&mut uconst).collect();
    let c_sh: Vec<u32> = (1..planes).map(|p| uconst(width * p)).collect();
    let c_two = uconst(2); // the unit-loop trip count when uu = 1 (units is 2 or 1, so iters is 2 or 1)
                           // the LDS staging constants: the ring mask, the barrier's scope/semantics, and the two shifts
                           // that split a staging thread's flat key into (column, word-in-superblock)
    let c_ring = uconst(stage_cap - 1); // the mandala mask — `key &&& (2^k − 1)`
    let c_cap = uconst(stage_cap); //     the ring's extent, the `OpTypeArray` length
    let c_scope_wg = uconst(SCOPE_WORKGROUP);
    let c_sem_wg = uconst(SEM_WG_ACQREL);
    // [units−1] → the per-column shared stride's shift/mask and the ring's last slot
    let c_sshift = [uconst(5), uconst(6)];
    let c_soffm = [uconst(31), uconst(63)];
    let c_snlast = [uconst(32 * bt - 1), uconst(64 * bt - 1)];
    // `it·local`, the staging sweep's stride — at most `64·bt/local + 1` iterations
    let c_itl: Vec<u32> = (0..(64 * bt).div_ceil(local) + 1)
        .map(|it| uconst(it * local))
        .collect();
    // every shared word index is a compile-time literal `< 64·bt` (the emitter's own arithmetic is
    // the closure); emitted only on the staged face so the constant pool stays as it was otherwise
    let c_word: Vec<u32> = if lds {
        (0..64 * bt).map(&mut uconst).collect()
    } else {
        Vec::new()
    };
    let per = full.div_ceil(segs); // superblocks per split-K segment
    let c_per = uconst(per);
    let c_last = uconst(segs - 1);
    let f32_0 = id!();
    const_words.extend([(4 << 16) | OP_CONSTANT, float, f32_0, 0f32.to_bits()]);
    let sint_0 = id!();
    const_words.extend([(4 << 16) | OP_CONSTANT, sint, sint_0, 0]);

    let cvar = id!(); // the superblock counter
    let uvar = id!(); // the unit counter (only when uu = 1)
    let nt = (rt * bt) as usize;
    let acc: Vec<u32> = (0..nt).map(|_| id!()).collect(); //  f32, live across the whole k walk
    let accq: Vec<u32> = (0..nt).map(|_| id!()).collect(); // i32, live within ONE superblock

    let mut body: Vec<u32> = Vec::new();
    macro_rules! e {
        ($($x:expr),* $(,)?) => { body.extend([$($x),*]) };
    }
    let b_entry = id!();
    e!((2 << 16) | OP_LABEL, b_entry);
    e!((4 << 16) | OP_VARIABLE, ptr_fn_u, cvar, SC_FUNCTION);
    e!((4 << 16) | OP_VARIABLE, ptr_fn_u, uvar, SC_FUNCTION);
    for a in &acc {
        e!((4 << 16) | OP_VARIABLE, ptr_fn_f, *a, SC_FUNCTION);
    }
    for a in &accq {
        e!((4 << 16) | OP_VARIABLE, ptr_fn_s, *a, SC_FUNCTION);
    }
    // ── the six bases ───────────────────────────────────────────────────────────────────────────
    // Descriptor face: the six `OpVariable`s themselves. PhysicalStorageBuffer face: six
    // `OpConvertUToPtr` of the addresses read ONCE from the table — after this the body is the same
    // emitter either way, because an `OpAccessChain` does not care which base it started from.
    let (b_w, b_xq, b_out, b_dm, b_xs, b_ys) = if pb {
        let mut bases = Vec::with_capacity(6);
        for (slot, ptr_buf) in [
            (0usize, ptr_pb_buf_u), // w
            (1, ptr_pb_buf_u),      // xq
            (2, ptr_pb_buf_f),      // out
            (3, ptr_pb_buf_u),      // dm
            (4, ptr_pb_buf_f),      // xs
            (5, ptr_pb_buf_u),      // ys
        ] {
            let (pa, av, bp) = (id!(), id!(), id!());
            e!(
                (6 << 16) | OP_ACCESS_CHAIN,
                ptr_sb_ul,
                pa,
                v_addr,
                c_u0,
                c_slot[slot]
            );
            e!((4 << 16) | OP_LOAD, ulong, av, pa);
            e!((4 << 16) | OP_CONVERT_U_TO_PTR, ptr_buf, bp, av);
            bases.push(bp);
        }
        (bases[0], bases[1], bases[2], bases[3], bases[4], bases[5])
    } else {
        (v_w, v_xq, v_out, v_dm, v_xs, v_ys)
    };
    // A load/store through one of those bases. The `Aligned 4` memory operand is REQUIRED on the
    // physical face (a raw pointer carries no layout the driver can infer) and must be ABSENT on the
    // logical one. Function-local `acc`/`accq` loads are NOT routed through these.
    macro_rules! gload {
        ($ty:expr, $res:expr, $p:expr) => {
            if pb {
                e!((6 << 16) | OP_LOAD, $ty, $res, $p, MEM_ALIGNED, 4)
            } else {
                e!((4 << 16) | OP_LOAD, $ty, $res, $p)
            }
        };
    }
    macro_rules! gstore {
        ($p:expr, $v:expr) => {
            if pb {
                e!((5 << 16) | OP_STORE, $p, $v, MEM_ALIGNED, 4)
            } else {
                e!((3 << 16) | OP_STORE, $p, $v)
            }
        };
    }
    let (lidv, widv, lane, wgx, tile) = (id!(), id!(), id!(), id!(), id!());
    e!((4 << 16) | OP_LOAD, uvec3, lidv, lid);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, lane, lidv, 0);
    e!((4 << 16) | OP_LOAD, uvec3, widv, wid);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, wgx, widv, 0);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, tile, widv, 1);
    let (slab, row0) = (id!(), id!());
    e!((5 << 16) | OP_IMUL, uint, slab, wgx, c_slab);
    e!((5 << 16) | OP_IADD, uint, row0, slab, lane);
    let (mut row, mut rowc, mut live) = (Vec::new(), Vec::new(), Vec::new());
    for u in 0..rt as usize {
        let ru = if u == 0 {
            row0
        } else {
            let t = id!();
            e!((5 << 16) | OP_IADD, uint, t, row0, c_ustep[u]);
            t
        };
        // THE MANDALA CLOSURE: no key reaches outside the weight's row ring.
        let (rc, lv) = (id!(), id!());
        e!((7 << 16) | OP_EXT_INST, uint, rc, glsl, GLSL_UMIN, ru, c_m1);
        e!((5 << 16) | OP_ULESS_THAN, bool_t, lv, ru, c_m);
        row.push(ru);
        rowc.push(rc);
        live.push(lv);
    }
    let i0 = id!();
    e!((5 << 16) | OP_IMUL, uint, i0, tile, c_bt);
    let (mut ij, mut xbase, mut sbase) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..bt as usize {
        let iv = if j == 0 {
            i0
        } else {
            let t = id!();
            e!((5 << 16) | OP_IADD, uint, t, i0, c_j[j]);
            t
        };
        // With split-K every segment folds the SAME activation column, so the base is 0, not `iv`
        // (`iv` still addresses the output — the partial slab this segment owns).
        let (xb, sb) = if segs > 1 {
            (c_u0, c_u0)
        } else {
            let (xb, sb) = (id!(), id!());
            e!((5 << 16) | OP_IMUL, uint, xb, iv, c_xstride);
            e!((5 << 16) | OP_IMUL, uint, sb, iv, c_nsb);
            (xb, sb)
        };
        ij.push(iv);
        xbase.push(xb);
        sbase.push(sb);
    }
    // the segment's superblock range [lo, hi) — `[0, full)` and free of any arithmetic at segs = 1
    let (sk_lo, sk_hi, sk_last) = if segs > 1 {
        let (lo, hi0, hi, last) = (id!(), id!(), id!(), id!());
        e!((5 << 16) | OP_IMUL, uint, lo, tile, c_per);
        e!((5 << 16) | OP_IADD, uint, hi0, lo, c_per);
        e!(
            (7 << 16) | OP_EXT_INST,
            uint,
            hi,
            glsl,
            GLSL_UMIN,
            hi0,
            c_full
        );
        e!((5 << 16) | OP_IEQUAL, bool_t, last, tile, c_last);
        (lo, hi, last)
    } else {
        (c_u0, c_full, c_u0)
    };
    for a in &acc {
        e!((3 << 16) | OP_STORE, *a, f32_0);
    }

    // ── the two phases: the `full` two-unit superblocks in a loop, then the odd one-unit tail ─────
    // Phase 0 exists only when `full > 0`; phase 1 only when `nunits` is odd. Both run the SAME
    // superblock body, so the tail is straight-line code, not a branch inside the hot loop.
    let nphase = (if full > 0 { 1 } else { 0 }) + odd as usize;
    for phase in 0..nphase {
        let looped = phase == 0 && full > 0;
        let units = if looped { 2u32 } else { 1 };
        let sb;
        let (b_ohdr, b_ochk, b_obody, b_ocont, b_omerge) = (id!(), id!(), id!(), id!(), id!());
        // the odd tail belongs to exactly ONE segment — the last — or it would be counted `segs` times
        let (b_tthen, b_tend) = (id!(), id!());
        if !looped && segs > 1 {
            e!((3 << 16) | OP_SELECTION_MERGE, b_tend, 0);
            e!((4 << 16) | OP_BRANCH_CONDITIONAL, sk_last, b_tthen, b_tend);
            e!((2 << 16) | OP_LABEL, b_tthen);
        }
        if looped {
            e!((3 << 16) | OP_STORE, cvar, sk_lo);
            e!((2 << 16) | OP_BRANCH, b_ohdr);
            e!((2 << 16) | OP_LABEL, b_ohdr);
            e!((4 << 16) | OP_LOOP_MERGE, b_omerge, b_ocont, 0);
            e!((2 << 16) | OP_BRANCH, b_ochk);
            e!((2 << 16) | OP_LABEL, b_ochk);
            let (ochk_c, ocond) = (id!(), id!());
            e!((4 << 16) | OP_LOAD, uint, ochk_c, cvar);
            e!((5 << 16) | OP_ULESS_THAN, bool_t, ocond, ochk_c, sk_hi);
            e!((4 << 16) | OP_BRANCH_CONDITIONAL, ocond, b_obody, b_omerge);
            e!((2 << 16) | OP_LABEL, b_obody);
            let t = id!();
            e!((4 << 16) | OP_LOAD, uint, t, cvar);
            sb = t;
        } else {
            sb = c_full; // the tail superblock's index is a compile-time constant
        }
        // three bases per superblock; every access below is one of these plus a literal
        let (wsb, xsb, dsb_i) = (id!(), id!(), id!());
        e!((5 << 16) | OP_IMUL, uint, wsb, sb, c_16m);
        e!((5 << 16) | OP_IMUL, uint, xsb, sb, c_64);
        e!((5 << 16) | OP_IMUL, uint, dsb_i, sb, c_m);
        // the superblock's ONE f16 scale per row, and the row's weight-word base
        let (mut dv, mut wr) = (Vec::new(), Vec::new());
        for u in 0..rt as usize {
            let (di, pd, dmw, dmunp, d, t) = (id!(), id!(), id!(), id!(), id!(), id!());
            e!((5 << 16) | OP_IADD, uint, di, dsb_i, rowc[u]);
            e!((6 << 16) | OP_ACCESS_CHAIN, pty_u, pd, b_dm, c_u0, di);
            gload!(uint, dmw, pd);
            e!(
                (6 << 16) | OP_EXT_INST,
                vec2f,
                dmunp,
                glsl,
                GLSL_UNPACK_HALF2X16,
                dmw
            );
            e!((5 << 16) | OP_COMPOSITE_EXTRACT, float, d, dmunp, 0);
            e!((5 << 16) | OP_IADD, uint, t, wsb, rowc[u]);
            dv.push(d);
            wr.push(t);
        }
        // the column's activation base, its scale, and its PRECOMPUTED integer Σq
        let (mut xr, mut xsv, mut ysv) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..bt as usize {
            let (t, si, pxs, xv, pys, ysu, ysi) = (id!(), id!(), id!(), id!(), id!(), id!(), id!());
            e!((5 << 16) | OP_IADD, uint, t, xbase[j], xsb);
            e!((5 << 16) | OP_IADD, uint, si, sbase[j], sb);
            e!((6 << 16) | OP_ACCESS_CHAIN, pty_f, pxs, b_xs, c_u0, si);
            gload!(float, xv, pxs);
            e!((6 << 16) | OP_ACCESS_CHAIN, pty_u, pys, b_ys, c_u0, si);
            gload!(uint, ysu, pys);
            e!((4 << 16) | OP_BITCAST, sint, ysi, ysu);
            xr.push(t);
            xsv.push(xv);
            ysv.push(ysi);
        }
        // ── LDS STAGING: the workgroup-uniform activation tile, once per superblock ──────────────
        // Every thread of the workgroup was loading the SAME `32·units·bt` words out of global, each
        // costing LEA + ISETP (the descriptor range compare) + LDG. Staged here, the read below is
        // `LDS [Rn+imm]` at a compile-time offset. Two barriers per superblock, against 128–256
        // three-slot loads per unit.
        if lds {
            let sstride = units * 32; // words per column in THIS phase's superblock
            let ui = (units - 1) as usize;
            // the ring must be free of the previous superblock's readers before it is overwritten
            e!(
                (4 << 16) | OP_CONTROL_BARRIER,
                c_scope_wg,
                c_scope_wg,
                c_sem_wg
            );
            for it in 0..(sstride * bt).div_ceil(local) as usize {
                let w0 = if it == 0 {
                    lane
                } else {
                    let t = id!();
                    e!((5 << 16) | OP_IADD, uint, t, lane, c_itl[it]);
                    t
                };
                // THE MANDALA CLOSURE on the staging key, twice: `UMin` to the ring's last slot (an
                // over-wide workgroup re-stores that slot's own correct value — same key, same
                // value, no race) and the `&&& (cap−1)` mask on the index the store forms.
                let (w, j, off, col, g0, g1, g2, px, xv, si, ps) = (
                    id!(),
                    id!(),
                    id!(),
                    id!(),
                    id!(),
                    id!(),
                    id!(),
                    id!(),
                    id!(),
                    id!(),
                    id!(),
                );
                e!(
                    (7 << 16) | OP_EXT_INST,
                    uint,
                    w,
                    glsl,
                    GLSL_UMIN,
                    w0,
                    c_snlast[ui]
                );
                e!((5 << 16) | OP_SHIFT_RIGHT_LOGICAL, uint, j, w, c_sshift[ui]);
                e!((5 << 16) | OP_BITWISE_AND, uint, off, w, c_soffm[ui]);
                e!((5 << 16) | OP_IADD, uint, col, i0, j);
                e!((5 << 16) | OP_IMUL, uint, g0, col, c_xstride);
                e!((5 << 16) | OP_IADD, uint, g1, g0, xsb);
                e!((5 << 16) | OP_IADD, uint, g2, g1, off);
                e!((6 << 16) | OP_ACCESS_CHAIN, pty_u, px, b_xq, c_u0, g2);
                gload!(uint, xv, px);
                e!((5 << 16) | OP_BITWISE_AND, uint, si, w, c_ring);
                e!((5 << 16) | OP_ACCESS_CHAIN, ptr_wg_u, ps, v_stage, si);
                e!((3 << 16) | OP_STORE, ps, xv);
            }
            e!(
                (4 << 16) | OP_CONTROL_BARRIER,
                c_scope_wg,
                c_scope_wg,
                c_sem_wg
            );
        }
        for a in &accq {
            e!((3 << 16) | OP_STORE, *a, sint_0);
        }
        // ── the unit walk: `uu` units unrolled, the rest looped ──────────────────────────────────
        let unroll = uu.min(units);
        let iters = units / unroll;
        let (b_uhdr, b_uchk, b_ubody, b_ucont, b_umerge) = (id!(), id!(), id!(), id!(), id!());
        // the SHARED read base per column when the unit index is a runtime value (`uu = 1`):
        // `(j·64 + unit·32) &&& (cap−1)`, one `and` per column per superblock. With `uu = 2` every
        // shared index is a literal and this stays empty.
        let mut su_base: Vec<u32> = Vec::new();
        let (wu_base, xu_base) = {
            let mut wb = Vec::new();
            let mut xb = Vec::new();
            if iters > 1 {
                e!((3 << 16) | OP_STORE, uvar, c_u0);
                e!((2 << 16) | OP_BRANCH, b_uhdr);
                e!((2 << 16) | OP_LABEL, b_uhdr);
                e!((4 << 16) | OP_LOOP_MERGE, b_umerge, b_ucont, 0);
                e!((2 << 16) | OP_BRANCH, b_uchk);
                e!((2 << 16) | OP_LABEL, b_uchk);
                debug_assert_eq!(iters, 2);
                let (uc, ucond) = (id!(), id!());
                e!((4 << 16) | OP_LOAD, uint, uc, uvar);
                e!((5 << 16) | OP_ULESS_THAN, bool_t, ucond, uc, c_two);
                e!((4 << 16) | OP_BRANCH_CONDITIONAL, ucond, b_ubody, b_umerge);
                e!((2 << 16) | OP_LABEL, b_ubody);
                let (tv, wo, xo) = (id!(), id!(), id!());
                e!((4 << 16) | OP_LOAD, uint, tv, uvar);
                e!((5 << 16) | OP_IMUL, uint, wo, tv, c_8m);
                e!((5 << 16) | OP_IMUL, uint, xo, tv, c_32);
                for u in 0..rt as usize {
                    let t = id!();
                    e!((5 << 16) | OP_IADD, uint, t, wr[u], wo);
                    wb.push(t);
                }
                for j in 0..bt as usize {
                    let t = id!();
                    e!((5 << 16) | OP_IADD, uint, t, xr[j], xo);
                    xb.push(t);
                    if lds {
                        let (s0, s1) = (id!(), id!());
                        e!((5 << 16) | OP_IADD, uint, s0, c_word[j * 64], xo);
                        e!((5 << 16) | OP_BITWISE_AND, uint, s1, s0, c_ring);
                        su_base.push(s1);
                    }
                }
            } else {
                wb = wr.clone();
                xb = xr.clone();
            }
            (wb, xb)
        };
        // `unroll` units × `wpu` words × `planes` planes; every index is a base plus a literal
        for t in 0..unroll as usize {
            for q in 0..wpu {
                let wq = t * wpu + q; // the weight word's slot in the superblock
                let mut wword = Vec::new();
                for u in 0..rt as usize {
                    let (wi, pw, ww) = (id!(), id!(), id!());
                    if wq == 0 {
                        e!(
                            (6 << 16) | OP_ACCESS_CHAIN,
                            pty_u,
                            pw,
                            b_w,
                            c_u0,
                            wu_base[u]
                        );
                    } else {
                        e!((5 << 16) | OP_IADD, uint, wi, wu_base[u], c_wo[wq]);
                        e!((6 << 16) | OP_ACCESS_CHAIN, pty_u, pw, b_w, c_u0, wi);
                    }
                    gload!(uint, ww, pw);
                    wword.push(ww);
                }
                for p in 0..planes as usize {
                    // the bt activation words — warp-uniform (their key came from WorkgroupId)
                    let mut xw = Vec::new();
                    for j in 0..bt as usize {
                        let (xi, px, xv) = (id!(), id!(), id!());
                        if lds {
                            // the shared re-index: `j·(units·32) + t·32 + p·8 + q`, and with `uu = 2`
                            // every term is a literal, so the driver forms `LDS [Rn+imm]` outright.
                            let lit = t * 32 + p * wpu + q;
                            if su_base.is_empty() {
                                let idx = j * (units as usize * 32) + lit;
                                e!(
                                    (5 << 16) | OP_ACCESS_CHAIN,
                                    ptr_wg_u,
                                    px,
                                    v_stage,
                                    c_word[idx]
                                );
                            } else if lit == 0 {
                                e!(
                                    (5 << 16) | OP_ACCESS_CHAIN,
                                    ptr_wg_u,
                                    px,
                                    v_stage,
                                    su_base[j]
                                );
                            } else {
                                e!((5 << 16) | OP_IADD, uint, xi, su_base[j], c_word[lit]);
                                e!((5 << 16) | OP_ACCESS_CHAIN, ptr_wg_u, px, v_stage, xi);
                            }
                            e!((4 << 16) | OP_LOAD, uint, xv, px);
                        } else if c_xo[wq][p] == c_u0 {
                            e!(
                                (6 << 16) | OP_ACCESS_CHAIN,
                                pty_u,
                                px,
                                b_xq,
                                c_u0,
                                xu_base[j]
                            );
                            gload!(uint, xv, px);
                        } else {
                            e!((5 << 16) | OP_IADD, uint, xi, xu_base[j], c_xo[wq][p]);
                            e!((6 << 16) | OP_ACCESS_CHAIN, pty_u, px, b_xq, c_u0, xi);
                            gload!(uint, xv, px);
                        }
                        xw.push(xv);
                    }
                    // plane p of each row's word: the codes at bit offset 2p, one per byte lane
                    let mut src = Vec::new();
                    for u in 0..rt as usize {
                        src.push(if p == 0 {
                            let t2 = id!();
                            e!((5 << 16) | OP_BITWISE_AND, uint, t2, wword[u], c_mask);
                            t2
                        } else {
                            let (sh, t2) = (id!(), id!());
                            e!(
                                (5 << 16) | OP_SHIFT_RIGHT_LOGICAL,
                                uint,
                                sh,
                                wword[u],
                                c_sh[p - 1]
                            );
                            e!((5 << 16) | OP_BITWISE_AND, uint, t2, sh, c_mask);
                            t2
                        });
                    }
                    // the rt·bt outer product of this k-quad — 64 of these per superblock
                    for u in 0..rt as usize {
                        for j in 0..bt as usize {
                            let (dot, a0, a1) = (id!(), id!(), id!());
                            let n = u * bt as usize + j;
                            e!((4 << 16) | OP_LOAD, sint, a0, accq[n]);
                            if accsat {
                                // the addend IS an operand of the dot — `IDP.4A Rd, Ra, Rb, Racc`
                                // instead of `IDP.4A Rd, Ra, Rb, RZ` plus an `IADD3` beside it
                                e!(
                                    (7 << 16) | OP_SUDOT_ACC_SAT,
                                    sint,
                                    a1,
                                    xw[j],
                                    src[u],
                                    a0,
                                    PACKED_VEC_4X8
                                );
                            } else {
                                e!(
                                    (6 << 16) | OP_SUDOT,
                                    sint,
                                    dot,
                                    xw[j],
                                    src[u],
                                    PACKED_VEC_4X8
                                );
                                e!((5 << 16) | OP_IADD, sint, a1, a0, dot);
                            }
                            e!((3 << 16) | OP_STORE, accq[n], a1);
                        }
                    }
                }
            }
        }
        if iters > 1 {
            e!((2 << 16) | OP_BRANCH, b_ucont);
            e!((2 << 16) | OP_LABEL, b_ucont);
            let (u2, un) = (id!(), id!());
            e!((4 << 16) | OP_LOAD, uint, u2, uvar);
            e!((5 << 16) | OP_IADD, uint, un, u2, c_u1);
            e!((3 << 16) | OP_STORE, uvar, un);
            e!((2 << 16) | OP_BRANCH, b_uhdr);
            e!((2 << 16) | OP_LABEL, b_umerge);
        }
        // ── THE EPILOGUE: one per 64 dots. `acc += ((Σcode·q) − Σq) · (d · sx)` ──────────────────
        // The offset leaves the integer ring by an OpISub, not a float term — that is the whole
        // format. The association `f·(d·sx)` is the one `row_dot_q2sb_avx2` uses
        // (`fmadd(cvt(t), set1(d*s), vacc)`); reassociating changes the rounding for nothing.
        for u in 0..rt as usize {
            for j in 0..bt as usize {
                let n = u * bt as usize + j;
                let (qs, qsub, qf, ds, prod, ab0, ab1) =
                    (id!(), id!(), id!(), id!(), id!(), id!(), id!());
                e!((4 << 16) | OP_LOAD, sint, qs, accq[n]);
                e!((5 << 16) | OP_ISUB, sint, qsub, qs, ysv[j]);
                e!((4 << 16) | OP_CONVERT_S_TO_F, float, qf, qsub);
                e!((5 << 16) | OP_FMUL, float, ds, dv[u], xsv[j]);
                e!((5 << 16) | OP_FMUL, float, prod, qf, ds);
                e!((4 << 16) | OP_LOAD, float, ab0, acc[n]);
                e!((5 << 16) | OP_FADD, float, ab1, ab0, prod);
                e!((3 << 16) | OP_STORE, acc[n], ab1);
            }
        }
        if looped {
            e!((2 << 16) | OP_BRANCH, b_ocont);
            e!((2 << 16) | OP_LABEL, b_ocont);
            let (ctr2, ctrn) = (id!(), id!());
            e!((4 << 16) | OP_LOAD, uint, ctr2, cvar);
            e!((5 << 16) | OP_IADD, uint, ctrn, ctr2, c_u1);
            e!((3 << 16) | OP_STORE, cvar, ctrn);
            e!((2 << 16) | OP_BRANCH, b_ohdr);
            e!((2 << 16) | OP_LABEL, b_omerge);
        }
        if !looped && segs > 1 {
            e!((2 << 16) | OP_BRANCH, b_tend);
            e!((2 << 16) | OP_LABEL, b_tend);
        }
    }
    // ── the store, the ONE place the clamped row still needs a predicate ─────────────────────────
    for u in 0..rt as usize {
        let (b_then, b_end) = (id!(), id!());
        e!((3 << 16) | OP_SELECTION_MERGE, b_end, 0);
        e!((4 << 16) | OP_BRANCH_CONDITIONAL, live[u], b_then, b_end);
        e!((2 << 16) | OP_LABEL, b_then);
        for j in 0..bt as usize {
            let (o1, ob, accf, pout) = (id!(), id!(), id!(), id!());
            e!((5 << 16) | OP_IMUL, uint, o1, ij[j], c_m);
            e!((5 << 16) | OP_IADD, uint, ob, o1, row[u]);
            e!((4 << 16) | OP_LOAD, float, accf, acc[u * bt as usize + j]);
            e!((6 << 16) | OP_ACCESS_CHAIN, pty_f, pout, b_out, c_u0, ob);
            gstore!(pout, accf);
        }
        e!((2 << 16) | OP_BRANCH, b_end);
        e!((2 << 16) | OP_LABEL, b_end);
    }
    e!((1 << 16) | OP_RETURN);

    let bound = idc + 1;
    let mut w: Vec<u32> = Vec::new();
    w.extend([SPV_MAGIC, SPV_VERSION_1_3, 0, bound, 0]);
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_DOT_PRODUCT]);
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_DOT_INPUT_4X8_PACKED]);
    if pb {
        // Int64 is not a convenience here: `OpConvertUToPtr` under `PhysicalStorageBuffer64` takes a
        // 64-bit integer operand, so the device's `shaderInt64` gates the whole face.
        w.extend([(2 << 16) | OP_CAPABILITY, CAP_INT64]);
        w.extend([
            (2 << 16) | OP_CAPABILITY,
            CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES,
        ]);
    }
    let ext = lit_string("SPV_KHR_integer_dot_product");
    w.push(((1 + ext.len() as u32) << 16) | OP_EXTENSION);
    w.extend(&ext);
    if pb {
        let e2 = lit_string("SPV_KHR_physical_storage_buffer");
        w.push(((1 + e2.len() as u32) << 16) | OP_EXTENSION);
        w.extend(&e2);
    }
    let g = lit_string("GLSL.std.450");
    w.push(((2 + g.len() as u32) << 16) | OP_EXT_INST_IMPORT);
    w.push(glsl);
    w.extend(&g);
    // The addressing model IS the change, at module level: `Logical` means every pointer descends
    // from a descriptor (base AND range); `PhysicalStorageBuffer64` admits a pointer made from a
    // number, which leaves the driver nothing to range-compare against.
    let addr_model = if pb {
        ADDR_PHYSICAL_STORAGE_BUFFER64
    } else {
        ADDR_LOGICAL
    };
    w.extend([(3 << 16) | OP_MEMORY_MODEL, addr_model, MEM_GLSL450]);
    let mn = lit_string("main");
    w.push(((4 + mn.len() as u32 + 1) << 16) | OP_ENTRY_POINT);
    w.extend([MODEL_GL_COMPUTE, main]);
    w.extend(&mn);
    w.extend([lid, wid]);
    w.extend([
        (6 << 16) | OP_EXECUTION_MODE,
        main,
        MODE_LOCAL_SIZE,
        local,
        1,
        1,
    ]);
    w.extend([
        (4 << 16) | OP_DECORATE,
        lid,
        DEC_BUILT_IN,
        BUILTIN_LOCAL_INVOCATION_ID,
    ]);
    w.extend([
        (4 << 16) | OP_DECORATE,
        wid,
        DEC_BUILT_IN,
        BUILTIN_WORKGROUP_ID,
    ]);
    w.extend([(4 << 16) | OP_DECORATE, rta_u, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, buf_u, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, buf_u, 0, DEC_OFFSET, 0]);
    w.extend([(4 << 16) | OP_DECORATE, rta_f, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, buf_f, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, buf_f, 0, DEC_OFFSET, 0]);
    if pb {
        // ONE descriptor for the whole kernel: the address table. The six operands have none.
        w.extend([(4 << 16) | OP_DECORATE, rta_ul, DEC_ARRAY_STRIDE, 8]);
        w.extend([(3 << 16) | OP_DECORATE, buf_ul, DEC_BLOCK]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, buf_ul, 0, DEC_OFFSET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, v_addr, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, v_addr, DEC_BINDING, 0]);
    } else {
        for (var, binding) in [
            (v_w, 0u32),
            (v_xq, 1),
            (v_out, 2),
            (v_dm, 3),
            (v_xs, 4),
            (v_ys, 5),
        ] {
            w.extend([(4 << 16) | OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
            w.extend([(4 << 16) | OP_DECORATE, var, DEC_BINDING, binding]);
        }
    }
    w.extend([(2 << 16) | OP_TYPE_VOID, void]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, fnty, void]);
    w.extend([(4 << 16) | OP_TYPE_INT, uint, 32, 0]);
    w.extend([(4 << 16) | OP_TYPE_INT, sint, 32, 1]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, float, 32]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, bool_t]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, uvec3, uint, 3]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, vec2f, float, 2]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_in_uvec3, SC_INPUT, uvec3]);
    w.extend([(4 << 16) | OP_VARIABLE, ptr_in_uvec3, lid, SC_INPUT]);
    w.extend([(4 << 16) | OP_VARIABLE, ptr_in_uvec3, wid, SC_INPUT]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, rta_u, uint]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, buf_u, rta_u]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, rta_f, float]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, buf_f, rta_f]);
    if pb {
        // the u64 address table (a real descriptor), then RAW pointers into the SAME block types —
        // identical layout decorations, so what the driver sees per element does not change; only
        // where the base came from does.
        w.extend([(4 << 16) | OP_TYPE_INT, ulong, 64, 0]);
        w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, rta_ul, ulong]);
        w.extend([(3 << 16) | OP_TYPE_STRUCT, buf_ul, rta_ul]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_sb_buf_ul,
            SC_STORAGE_BUFFER,
            buf_ul,
        ]);
        w.extend([
            (4 << 16) | OP_VARIABLE,
            ptr_sb_buf_ul,
            v_addr,
            SC_STORAGE_BUFFER,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_sb_ul,
            SC_STORAGE_BUFFER,
            ulong,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_pb_buf_u,
            SC_PHYSICAL_STORAGE_BUFFER,
            buf_u,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_pb_buf_f,
            SC_PHYSICAL_STORAGE_BUFFER,
            buf_f,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_pb_u,
            SC_PHYSICAL_STORAGE_BUFFER,
            uint,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_pb_f,
            SC_PHYSICAL_STORAGE_BUFFER,
            float,
        ]);
    } else {
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_sb_buf_u,
            SC_STORAGE_BUFFER,
            buf_u,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_sb_buf_f,
            SC_STORAGE_BUFFER,
            buf_f,
        ]);
        w.extend([
            (4 << 16) | OP_VARIABLE,
            ptr_sb_buf_u,
            v_w,
            SC_STORAGE_BUFFER,
        ]);
        w.extend([
            (4 << 16) | OP_VARIABLE,
            ptr_sb_buf_u,
            v_xq,
            SC_STORAGE_BUFFER,
        ]);
        w.extend([
            (4 << 16) | OP_VARIABLE,
            ptr_sb_buf_f,
            v_out,
            SC_STORAGE_BUFFER,
        ]);
        w.extend([
            (4 << 16) | OP_VARIABLE,
            ptr_sb_buf_u,
            v_dm,
            SC_STORAGE_BUFFER,
        ]);
        w.extend([
            (4 << 16) | OP_VARIABLE,
            ptr_sb_buf_f,
            v_xs,
            SC_STORAGE_BUFFER,
        ]);
        w.extend([
            (4 << 16) | OP_VARIABLE,
            ptr_sb_buf_u,
            v_ys,
            SC_STORAGE_BUFFER,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_sb_u,
            SC_STORAGE_BUFFER,
            uint,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_sb_f,
            SC_STORAGE_BUFFER,
            float,
        ]);
    }
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_fn_u, SC_FUNCTION, uint]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_fn_s, SC_FUNCTION, sint]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_fn_f, SC_FUNCTION, float]);
    w.extend(&const_words);
    if lds {
        // the shared ring — a FIXED power-of-two extent, which is what makes `MandalaMemory.access`
        // apply verbatim to every index that reaches it. Declared after the constants because
        // `OpTypeArray`'s length operand is one of them.
        w.extend([(4 << 16) | OP_TYPE_ARRAY, arr_stage, uint, c_cap]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            ptr_arr_stage,
            SC_WORKGROUP,
            arr_stage,
        ]);
        w.extend([
            (4 << 16) | OP_VARIABLE,
            ptr_arr_stage,
            v_stage,
            SC_WORKGROUP,
        ]);
        w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_wg_u, SC_WORKGROUP, uint]);
    }
    w.extend([(5 << 16) | OP_FUNCTION, void, main, 0, fnty]);
    w.extend(&body);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The superblock integer-dot GEMM as bytes. See [`gemv_dp4a_sb_module`].
pub fn gemv_dp4a_sb_bytes(
    m: u32,
    k: u32,
    local: u32,
    bt: u32,
    rt: u32,
    uu: u32,
    segs: u32,
    carrier: Carrier,
) -> Vec<u8> {
    gemv_dp4a_sb_module(m, k, local, bt, rt, uu, segs, carrier)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// **The superblock chart, transposed for the GPU.** Takes `servefold`'s row-major `Proj::rowq2s`
/// (bytes) and `Proj::dsb` (f16) and lays them out the way [`gemv_dp4a_sb_module`] reads them:
/// `w[(W)·m + r]` over the row's `k/16` u32 words, and `dsb[sb·m + r]` in the low half of a u32.
/// **No value changes** — this is a permutation of storage so that lane `r` and lane `r+1` read
/// adjacent words (one coalesced request per warp instead of 32 scattered ones); the dot the kernel
/// computes is the same chart's dot, element for element.
pub fn q2sb_gpu_chart(rowq2s: &[u8], dsb: &[u16], m: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(rowq2s.len(), m * k / 4);
    let words = k / 16; // u32 weight words per row
    let nsb = k.div_ceil(256);
    assert_eq!(dsb.len(), m * nsb);
    let mut w = vec![0u32; m * words];
    for r in 0..m {
        let base = r * (k / 4);
        for wq in 0..words {
            let b = &rowq2s[base + wq * 4..base + wq * 4 + 4];
            w[wq * m + r] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
    }
    let mut d = vec![0u32; m * nsb];
    for r in 0..m {
        for sb in 0..nsb {
            d[sb * m + r] = dsb[r * nsb + sb] as u32;
        }
    }
    (w, d)
}

/// **The two-level scale face, transposed for the GPU** — `qsub`/`qsup` laid out the way a DP4A
/// gemv reads them, at HALF the bytes of `dm`.
///
/// `dm` is one u32 per `(block, row)`: two f16, the block's `(d, lo)`. That is **1.0 bits/weight**
/// of scale metadata, and it is the reason the device face is 5.0 bpw at Q4 where the CPU row chart
/// serves 4.5 — the CPU already reads `qsub`/`qsup` through `decode_scales_row`, and the device
/// never learned to (`results/qwen38-27b-gpu.md`: 12.90 GB uploaded against an 11.47 GB CPU face).
///
/// The two-level face is one BYTE PAIR per `(block, row)` under one f16 pair per ROW. So:
///
/// * `sub[b·(m/2) + r/2]` packs TWO rows into one u32 — row `r` in the low half, `r+1` in the high.
///   Lane `r` and lane `r+1` then read the SAME word, which is half the loads of `dm` and still one
///   coalesced request per warp. Rows are padded to even; a model with an odd row count pads by one.
/// * `sup[r]` is the row's `(dsup, losup)` f16 pair, read ONCE before the block loop rather than per
///   block — which is why this face is also *cheaper* per block than unpacking an f16 pair, not
///   merely smaller.
///
/// No value changes. `d = d8·dsup`, `lo = l8·losup` is exactly what `decode_scales_row` computes on
/// the CPU, from the same bytes.
pub fn q4_sub8_gpu_chart(
    qsub: &[u8],
    qsup: &[u16],
    m: usize,
    blocks: usize,
) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(
        qsub.len(),
        m * blocks * 2,
        "qsub is one (d8, l8) pair per (row, block)"
    );
    assert_eq!(
        qsup.len(),
        m * 2,
        "qsup is one (dsup, losup) f16 pair per row"
    );
    let mrow = m.div_ceil(2);
    let mut sub = vec![0u32; blocks * mrow];
    for r in 0..m {
        for b in 0..blocks {
            let d8 = u32::from(qsub[r * blocks * 2 + b * 2]);
            let l8 = u32::from(qsub[r * blocks * 2 + b * 2 + 1]);
            let half = d8 | (l8 << 8);
            let w = &mut sub[b * mrow + r / 2];
            *w |= half << (16 * (r & 1));
        }
    }
    let sup: Vec<u32> = (0..m)
        .map(|r| u32::from(qsup[r * 2]) | (u32::from(qsup[r * 2 + 1]) << 16))
        .collect();
    (sub, sup)
}

#[cfg(test)]
mod two_level_chart_tests {
    use super::q4_sub8_gpu_chart;

    /// **The GPU two-level chart is a PERMUTATION of the CPU one, not a re-quantisation.**
    ///
    /// The whole claim of `q4_sub8_gpu_chart` is that it moves bytes and changes no value — so the
    /// test decodes the packed form the way a kernel must (`sub[b·mrow + r/2]` shifted by the row's
    /// parity, `sup[r]` as two f16) and requires the SAME `(d8, l8, dsup, losup)` the CPU reads out
    /// of `qsub`/`qsup`. A layout that loses a byte pair here would still produce plausible logits,
    /// which is the failure mode this whole face already has a history of.
    ///
    /// An ODD row count is included deliberately: rows pack two-to-a-word, so `m = 5` exercises the
    /// half-empty final word that an even-only test would never reach.
    #[test]
    fn two_level_gpu_chart_is_a_permutation_of_the_row_chart() {
        for &(m, blocks) in &[(4usize, 3usize), (5, 2), (8, 7), (1, 1)] {
            let mut qsub = vec![0u8; m * blocks * 2];
            for (i, v) in qsub.iter_mut().enumerate() {
                // d8 is UNSIGNED 0..255 and l8 is SIGNED — a packing that sign-extends the wrong
                // lane is exactly what this catches, so the fixture spans both ranges.
                *v = (i * 37 % 256) as u8;
            }
            let qsup: Vec<u16> = (0..m * 2).map(|i| (0x3800 + i * 11) as u16).collect();

            let (sub, sup) = q4_sub8_gpu_chart(&qsub, &qsup, m, blocks);
            let mrow = m.div_ceil(2);
            assert_eq!(sub.len(), blocks * mrow, "one u32 per (block, row PAIR)");
            assert_eq!(sup.len(), m, "one u32 per row");

            for r in 0..m {
                // the row's super-pair, as the kernel would unpack it
                assert_eq!(sup[r] & 0xffff, u32::from(qsup[r * 2]), "dsup row {r}");
                assert_eq!(sup[r] >> 16, u32::from(qsup[r * 2 + 1]), "losup row {r}");
                for b in 0..blocks {
                    let word = sub[b * mrow + r / 2];
                    let half = (word >> (16 * (r & 1))) & 0xffff;
                    assert_eq!(
                        half & 0xff,
                        u32::from(qsub[r * blocks * 2 + b * 2]),
                        "d8 at row {r} block {b} of {m}x{blocks}"
                    );
                    assert_eq!(
                        half >> 8,
                        u32::from(qsub[r * blocks * 2 + b * 2 + 1]),
                        "l8 at row {r} block {b} of {m}x{blocks}"
                    );
                }
            }
        }
    }

    /// The face is `dm`/2 **plus one word per row**, and the second term is not noise at every
    /// shape — asserted as the exact formula rather than a threshold, because a threshold is a
    /// guess about which shapes matter.
    ///
    /// `dm` is `blocks·m` words. This is `blocks·⌈m/2⌉ + m`, so the ratio is `⌈m/2⌉/m + 1/blocks`
    /// — **two** correction terms, and this test caught both by refusing the simpler claims I made
    /// first. `1/blocks` is the per-ROW super-pair, which amortises over the row's blocks. `⌈m/2⌉/m`
    /// is the row-pairing pad, which is 0.5 only for EVEN `m` and 0.571 at `m = 7`.
    ///
    /// At Qwen3.8-27B (`k = 5120` ⇒ 160 blocks, `m` in the thousands and even) the ratio is 0.506
    /// and the 0.5 bits/weight is real. The degenerate shapes are here because the formula is the
    /// claim; "about half" was the claim twice, and it was wrong at 16 blocks and again at `m = 7`.
    #[test]
    fn two_level_gpu_chart_is_dm_over_two_plus_one_word_per_row() {
        for &(m, blocks) in &[(64usize, 160usize), (64, 16), (5120, 160), (7, 3)] {
            let qsub = vec![0u8; m * blocks * 2];
            let qsup = vec![0u16; m * 2];
            let (sub, sup) = q4_sub8_gpu_chart(&qsub, &qsup, m, blocks);
            let dm_words = m * blocks; // `dm` is one u32 per (block, row)
            let ours = sub.len() + sup.len();
            assert_eq!(ours, blocks * m.div_ceil(2) + m, "at {m}x{blocks}");
            let ratio = ours as f64 / dm_words as f64;
            let want = m.div_ceil(2) as f64 / m as f64 + 1.0 / blocks as f64;
            assert!(
                (ratio - want).abs() < 0.02,
                "at {m}x{blocks} the face is {ratio:.4} of `dm`, not the ceil(m/2)/m + 1/blocks = \
{want:.4} its layout implies"
            );
        }
        // and at a SERVED shape the saving must actually be the 0.5 bits/weight it is built for
        let (m, blocks) = (5120usize, 160usize);
        let (sub, sup) =
            q4_sub8_gpu_chart(&vec![0u8; m * blocks * 2], &vec![0u16; m * 2], m, blocks);
        let saved_bits =
            32.0 * (m * blocks - (sub.len() + sup.len())) as f64 / (m * blocks * 32) as f64;
        assert!(
            saved_bits > 0.49,
            "saves {saved_bits:.4} bits/weight at a served shape, not the ~0.5 claimed"
        );
    }
}

/// **The two-plane 3-bit GPU chart** — `Carrier::Q3`'s `words` buffer, plane A then plane B.
///
/// `codes` is row-major `[m][k]`, values `0..=7`. The output is one buffer:
///
/// * `[0, (k/16)·m)` — **plane A**, the low 2 bits, laid out EXACTLY as `Carrier::Q2` lays out its
///   codes (`A[(kk/16)·m + r] |= (q & 3) << ((kk % 16)·2)`), so the DP4A phase loop, the block
///   stride and the whole activation chart are the ones already in production.
/// * `[(k/16)·m, …)` — **plane B**, the high bit, one u32 per (32-block, row) at `B[b·m + r]`.
///
/// **Plane B is bit-PERMUTED, and that is the point.** A DP4A phase `(cj, l)` extracts the codes at
/// `{16·cj + 4·i + l : i = 0..3}` into bytes `0..3` — a stride of 4. The nibble-to-bytes expand a
/// kernel can afford (`x · 0x00204081 & 0x01010101`) needs those bits CONTIGUOUS; at stride 4 the
/// multiply's partial products collide (`4j + 4j' ≡ 0 mod 8` for every `j + j'` even) and the
/// result is silently wrong for half the lanes. So the permutation happens once, here, offline:
/// the bit for code `j = 16·cj + 4·i + l` is stored at position `16·cj + 4·l + i`, and the kernel
/// reads `(B >> (16·cj + 4·l)) & 0xF`.
///
/// Gated by `q3_planes_round_trip`: a layout that loses a bit here still produces finite logits.
pub fn q3_planes_gpu_chart(codes: &[u8], m: usize, k: usize) -> Vec<u32> {
    assert_eq!(codes.len(), m * k, "codes is row-major [m][k]");
    assert!(k % 32 == 0, "the 3-bit chart is per-32-block");
    let blocks = k / 32;
    let a_len = (k / 16) * m;
    let mut out = vec![0u32; a_len + blocks * m];
    for r in 0..m {
        for kk in 0..k {
            let q = u32::from(codes[r * k + kk]);
            debug_assert!(q < 8, "3-bit code {q} out of range at ({r}, {kk})");
            // plane A — Q2's expression, term for term
            out[(kk / 16) * m + r] |= (q & 3) << ((kk % 16) as u32 * 2);
            // plane B — the permuted high bit
            if q & 4 != 0 {
                let (b, j) = (kk / 32, kk % 32);
                let (cj, rem) = (j / 16, j % 16);
                let (i, l) = (rem / 4, rem % 4);
                out[a_len + b * m + r] |= 1u32 << (cj * 16 + l * 4 + i);
            }
        }
    }
    out
}

#[cfg(test)]
mod sb_carrier_tests {
    use super::gemv_dp4a_sb_bytes;
    use super::Carrier;

    /// `planes · width = 8` is the invariant the emitter's geometry rests on: one byte lane holds
    /// exactly `planes` codes at `width` bits, and the DP4A eats four lanes per dot at every width.
    #[test]
    fn the_code_geometry_closes_at_one_byte_per_lane() {
        for c in [Carrier::Q2, Carrier::Q4, Carrier::Q8] {
            let pack = c.pack();
            let (width, planes, wpu) = (32 / pack, pack / 4, 128 / pack);
            assert_eq!(
                width * planes,
                8,
                "{c:?}: a byte lane must hold whole codes"
            );
            // the two quantities that COINCIDE at Q2 and so read as one literal in the old emitter
            assert_eq!(wpu, 4 * width, "{c:?}: weight words per 128-weight unit");
            assert_eq!(32 / planes, wpu, "{c:?}: activation words per plane");
        }
    }

    /// The emitter is carrier-generic: Q4 and Q8 EMIT. They are not reachable through
    /// [`crate::gpu::nn::spec::GemvSpec`] — the packer has no `rowq4s`/`rowq8s` chart to feed them
    /// and `legalize` refuses accordingly — so this is the only place the widened path is exercised,
    /// and it is here so "expressible" is checked rather than asserted.
    #[test]
    fn q4_and_q8_superblock_modules_emit_and_are_distinct() {
        let (m, k, local, bt, rt, uu, segs) = (512u32, 1024u32, 128u32, 4u32, 4u32, 1u32, 1u32);
        let q2 = gemv_dp4a_sb_bytes(m, k, local, bt, rt, uu, segs, Carrier::Q2);
        let q4 = gemv_dp4a_sb_bytes(m, k, local, bt, rt, uu, segs, Carrier::Q4);
        let q8 = gemv_dp4a_sb_bytes(m, k, local, bt, rt, uu, segs, Carrier::Q8);
        for (n, b) in [("q2", &q2), ("q4", &q4), ("q8", &q8)] {
            assert!(b.len() > 1024, "{n}: emitted nothing usable");
            assert_eq!(&b[..4], &0x0723_0203u32.to_le_bytes(), "{n}: SPIR-V magic");
        }
        // a wider code means fewer, fatter words per unit -- so the unrolled loop SHRINKS
        assert!(
            q2 != q4 && q4 != q8,
            "the width must change the emitted code"
        );
    }
}

#[cfg(test)]
mod q3_plane_tests {
    use super::q3_planes_gpu_chart;

    /// **Decode the chart the way the KERNEL must, and require the original codes back.**
    ///
    /// This walks plane A's phase extraction (`(A >> 2l) & 0x03030303`, byte `i` = code `4i + l`)
    /// and plane B's contiguous-nibble read (`(B >> (16cj + 4l)) & 0xF`, bit `i` = byte `i`)
    /// rather than inverting the packing expression — inverting the packer would agree with a
    /// packer that is wrong in the same direction, which is the failure mode a chart has.
    ///
    /// An ODD `m` is included: rows are the minor axis of the col-major layout, so a stride bug
    /// shows up at the last row or not at all.
    #[test]
    fn q3_planes_round_trip() {
        for &(m, k) in &[(4usize, 32usize), (3, 64), (8, 96), (1, 32)] {
            let codes: Vec<u8> = (0..m * k).map(|i| ((i * 7 + i / 5) % 8) as u8).collect();
            let w = q3_planes_gpu_chart(&codes, m, k);
            let a_len = (k / 16) * m;
            assert_eq!(w.len(), a_len + (k / 32) * m);
            for r in 0..m {
                for b in 0..k / 32 {
                    let bw = w[a_len + b * m + r];
                    for cj in 0..2 {
                        let aw = w[(b * 2 + cj) * m + r];
                        for l in 0..4 {
                            let lo = (aw >> (2 * l)) & 0x0303_0303;
                            let hi = (bw >> (cj * 16 + l * 4)) & 0xF;
                            for i in 0..4 {
                                let code = ((lo >> (8 * i)) & 3) | (((hi >> i) & 1) << 2);
                                let kk = b * 32 + cj * 16 + 4 * i + l;
                                assert_eq!(
                                    code as u8,
                                    codes[r * k + kk],
                                    "({m}x{k}) row {r} block {b} cj {cj} phase {l} lane {i} -> k {kk}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The size formula IS the 3.0 bits/weight claim, so it is asserted rather than commented.
    #[test]
    fn q3_chart_is_three_bits_per_weight() {
        for &(m, k) in &[(4usize, 32usize), (7, 256), (128, 5120)] {
            let w = q3_planes_gpu_chart(&vec![0u8; m * k], m, k);
            assert_eq!(
                w.len() * 32,
                m * k * 3,
                "({m}x{k}) must be exactly 3 bits/weight"
            );
        }
    }
}
