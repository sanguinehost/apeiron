//! simt::dp4a — the integer-dot (DP4A) chart GEMV: the k-packed `OpSUDot` decode kernel with
//! its dynamic-activation-scale form, the batch-tiled prefill GEMM, and the roofline probe that
//! measures the instruction's own ceiling. Cut verbatim from `simt.rs`.

use super::*;

/// **The integer-dot GEMV — the chart fold in its native form** (`SealChartFold`, headline).
/// The float chart path (Q8/Q4 in [`gemv_module`]) dequantizes every code to fp32 and does a
/// float FMA per element — the SASS named that inner loop `SHR·AND·I2FP·FFMA`, the float tax.
/// The theorem says the fold *is* integer in the (code, scale) chart:
/// `Σ (d·q + m)·x = d·(Σ q·x) + m·(Σ x)`, with `Σ q·x` an int32-bounded integer sum. This
/// module computes that inner sum with [`OP_SUDOT`] (DP4A, hardware-accelerated on the 5080):
/// one instruction folds 4 codes × 4 int8 activations into one i32. One float multiply-add
/// per *block* (32 codes), not per element.
///
/// **The layout is k-packed** (not the row-packed [`gemv_module`] chart): weight word
/// `W[c·M + r]` holds the 4 uint8 codes at `k = 4c … 4c+3` for row `r` — so consecutive rows
/// read consecutive words (coalesced, `colmajor_unit_stride`) *and* each thread owns a whole
/// word (this is the "word-per-thread" lever — 8 chunks per 32-block, not 32 elements, killing
/// the per-element address IMAD). Activations `XQ[c]` = 4 int8 at the same k, broadcast. The
/// scale `d` is premultiplied by the activation scale on the host (`d' = d·sx`), so the chart
/// stays `d'·(Σ q·xq) + m·(Σ x)` with the activation quantization folded into `d'` — no extra
/// binding. Bindings: 0 = W (u32 codes, k-packed), 1 = XQ (u32 int8 activations), 2 = y,
/// 3 = DM (interleaved d', m per block·row), 4 = XS (`Σ_block x`, true fp32).
///
/// **The carrier is the dial** (`MetalNative` §7b): Q8 packs 4 uint8 codes per word (one DP4A),
/// Q4 packs 8 nibbles per word (half the transport — the energy floor is bytes moved, not
/// arithmetic; measured, the SASS's transport ops dominate once the float tax is gone). Q4 feeds
/// DP4A by splitting the word into even/odd nibbles (`w & 0x0F0F0F0F` and `(w>>4) & 0x0F0F0F0F`),
/// two `OpSUDot` per word against deinterleaved activation words — the split is 3 ops per 8
/// codes, amortized, versus the float path's ~4 ops *per* code. `k % 32 == 0` required.
/// Split-K as [`gemv_module`].
pub fn gemv_dp4a_module(m: u32, k: u32, local: u32, segments: u32, carrier: Carrier) -> Vec<u32> {
    gemv_dp4a_module_dyn(m, k, local, segments, carrier, false)
}

/// [`gemv_dp4a_module`] with an optional DYNAMIC activation scale (binding 5 = `sx`, one float
/// PER 32-BLOCK): the decode loop int8-quantizes activations per token on the device with a
/// per-block scale (a single global scale is destroyed by activation outliers — measured
/// 7.9 rel dev on qwen0.5b logits), so `d` stays RAW in DM and the block epilogue multiplies
/// `d·(Σ q·xq)·sx[b]` at runtime. In this form DM is also **f16-packed** — one u32 word
/// `f16(d) | f16(lo)<<16` per (block, row), halving the block-param traffic (at q2 the f32
/// params equalled the codes themselves — 2 GB/token on an 8B). `dyn_sx = false` is the
/// bench's premultiplied f32 form, byte-identical.
pub fn gemv_dp4a_module_dyn(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: Carrier,
    dyn_sx: bool,
) -> Vec<u32> {
    gemv_dp4a_module_moe(m, k, local, segments, carrier, dyn_sx, None)
}

/// **The MoE DP4A gemv — the served decode kernel, with the expert chosen per slot by a buffer.**
///
/// This is the one that matters: the decode chain runs the DP4A charts, not [`gemv_module`]. A
/// command buffer is recorded once and replayed every token, so the routed expert cannot be a
/// descriptor offset — it is read inside the shader from `SEL[gid.y]`.
///
/// It reuses split-K's 2D grid rather than inventing a second convention: `gid.y` is the SLOT
/// (0..n_used) instead of a k-window, and the output already lands at `y[s·M + r]` exactly as
/// split-K partials do. The only new work per invocation is one indirect load and two `IADD`s.
///
/// Each expert's chart is a CONTIGUOUS run — `words_per_expert` u32 codes and `dm_per_expert` u32
/// scale words — which is the same layout `load_model_moe` keeps on the host, and the unit an
/// expert cache moves as a single transfer (`results/moe-route-locality.md`).
///
/// **`weighted` folds the router's mixture weight into the EPILOGUE** (vLLM's
/// `MUL_ROUTED_WEIGHT`, `fused_moe.py::fused_moe_kernel`): binding 7 = SWT (f32, `n_used`, exactly
/// what `moe_topk` writes), and the store becomes `OUT[s·M + r] = SWT[s]·acc`. One FMUL per output
/// element, after the f32 accumulation — where the down projection already stands. It replaces the
/// separate `moe_combine` dispatch that used to do the weighted sum, priced at ~1.3 µs marginal
/// (`results/dispatch-cost.md`) × 48 layers ≈ 62 µs/token (`results/moe-reference-set.md` §1).
/// **Slots stay SEPARATE in OUT** (stride `M` per slot, no cross-slot accumulation, no atomics):
/// the consumer seg-reduce (`rmsnorm_add_q_seg` with `segs = n_used`) sums them, so the fusion adds
/// a multiply and removes a dispatch without moving where the sum happens.
///
/// The weight applies ONCE per routed FFN, on the DOWN projection — the gate/up gemv is built with
/// `weighted = false`. That is why the flag is per-construction and not a runtime uniform.
///
/// **`x_slot` strides the ACTIVATION by slot as well** — `XQ[s·(k/4) + …]`, `XS`/`SX`
/// `[s·(k/32) + …]`, which is exactly the layout [`crate::gpu::nn::silu_mul_q_slot_module`] writes.
/// The two routed gemvs differ here and it is not cosmetic: the gate/up projection reads ONE
/// activation (the layer's normed hidden) for every slot, so it passes `false`; the down projection
/// reads the SwiGLU output of *its own* expert, so it must pass `true`. With `false` on the down
/// gemv every slot folds slot 0's activation against its own expert's chart — a wrong number, not a
/// wrong shape, and the composition gate is what names it. Same argument as the `dyn_sx` DM base
/// above (`moe.is_some()` there), one binding over.
///
/// Bindings: 0 = W, 1 = XQ, 2 = OUT, 3 = DM, 4 = XS, 5 = SX (when `dyn_sx`), **6 = SEL**,
/// **7 = SWT (when `weighted`)**.
/// `m` is the per-EXPERT row count. Dispatch `groups = (ceil(m/local), n_used)`.
pub fn gemv_dp4a_moe_bytes(
    m: u32,
    k: u32,
    local: u32,
    carrier: Carrier,
    words_per_expert: u32,
    dm_per_expert: u32,
    weighted: bool,
    x_slot: bool,
) -> Vec<u8> {
    gemv_dp4a_module_moe(
        m,
        k,
        local,
        1,
        carrier,
        true,
        Some((words_per_expert, dm_per_expert, weighted, x_slot)),
    )
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect()
}

fn gemv_dp4a_module_moe(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: Carrier,
    dyn_sx: bool,
    moe: Option<(u32, u32, bool, bool)>,
) -> Vec<u32> {
    gemv_dp4a_module_moe_sc(m, k, local, segments, carrier, dyn_sx, moe, false)
}

#[allow(clippy::too_many_arguments)]
fn gemv_dp4a_module_moe_sc(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: Carrier,
    dyn_sx: bool,
    moe: Option<(u32, u32, bool, bool)>,
    two_level: bool,
) -> Vec<u32> {
    assert!(k % 32 == 0, "chart carriers need k % 32 == 0");
    // **The two-level scale face has no expert stride.** An expert's chart is a contiguous run of
    // `dm_per_expert` scale words; the two-level layout is `[sub][sup]` per PROJECTION, so a routed
    // read would index `sup` as though it were the next expert's `sub`. That is a plausible-logits
    // failure, not a crash, so it is refused by name rather than left to the reader.
    assert!(
        !two_level || moe.is_none(),
        "the two-level scale face is not wired for the ROUTED gemv — an expert's chart is a \
contiguous run and `[sub][sup]` per projection has no per-expert stride"
    );
    // `gid.y` is either the k-window or the expert slot; the output index `s·M + r` means one
    // thing at a time.
    assert!(
        moe.is_none() || segments == 1,
        "split-K and MoE both claim gid.y"
    );
    // The routed weight is indexed by the SLOT, so it only exists on the MoE grid — the tuple
    // carries it rather than a free parameter so a dense module cannot ask for it. The epilogue
    // that applies it is shared by both `dyn_sx` branches, so no `dyn_sx` boundary is needed here.
    let weighted = moe.is_some_and(|(_, _, wt, _)| wt);
    // ...and the per-slot ACTIVATION base rides the same tuple, for the same reason.
    let x_slot = moe.is_some_and(|(_, _, _, xs)| xs);
    let grid_y = segments > 1 || moe.is_some();
    assert!(carrier.is_chart(), "DP4A is a chart carrier (Q8/Q4/Q2)");
    let codes_per_word = carrier.pack(); //  Q8 → 4, Q4 → 8, Q2 → 16
    let dots_per_word = codes_per_word / 4; // Q8 → 1, Q4 → 2, Q2 → 4 (codes per byte lane)
    let cpb = 32 / codes_per_word; //         weight words per 32-block: Q8 → 8, Q4 → 4, Q2 → 2
    let wbits = 32 / codes_per_word; //       bits per code: Q8 → 8, Q4 → 4, Q2 → 2
    let byte_mask = 0x0101_0101u32 * ((1u32 << wbits) - 1); // per-byte code mask: FF / 0F / 03
    let split = dots_per_word > 1; //         Q4/Q2 mask-split the word; Q8 uses it whole
    let blocks = k / 32; // outer range (one 32-code block per iteration)

    const VOID: u32 = 1;
    const FNTY: u32 = 2;
    const UINT: u32 = 3;
    const SINT: u32 = 4; //     signed i32: the DP4A result + code-fold accumulator
    const FLOAT: u32 = 5;
    const BOOL: u32 = 6;
    const UVEC3: u32 = 7;
    const PTR_IN_UVEC3: u32 = 8;
    const GID: u32 = 9;
    const RTA_U: u32 = 10;
    const BUF_U: u32 = 11;
    const PTR_SB_BUF_U: u32 = 12;
    const RTA_F: u32 = 13;
    const BUF_F: u32 = 14;
    const PTR_SB_BUF_F: u32 = 15;
    const W: u32 = 16;
    const XQ: u32 = 17;
    const OUT: u32 = 18;
    const DM: u32 = 19;
    const XS: u32 = 20;
    const PTR_SB_U: u32 = 21;
    const PTR_SB_F: u32 = 22;
    const PTR_FN_U: u32 = 23;
    const PTR_FN_S: u32 = 24;
    const PTR_FN_F: u32 = 25;
    const C_U0: u32 = 26;
    const C_U1: u32 = 27;
    const C_M: u32 = 28;
    const C_BLOCKS: u32 = 29;
    const C_CPB: u32 = 30; //   chunks per 32-block (Q8 → 8, Q4 → 4)
    const C_F0: u32 = 31;
    const C_S0: u32 = 32; //    sint 0 (the integer accumulator's zero)
    const C_SEG: u32 = 33; //   split-K window length in blocks
    const C_DOTS: u32 = 34; //  dots per word (Q4 → 2; activation stride within a chunk)
    const C_NIBMASK: u32 = 35; // per-byte code mask (0F0F… Q4 / 0303… Q2); phase shifts at 181..183
    const MAIN: u32 = 40;
    const B_ENTRY: u32 = 41;
    const CVAR: u32 = 42; //    outer counter (block b)
    const JVAR: u32 = 43; //    inner counter (chunk cj in 0..8)
    const ACCQVAR: u32 = 44; // sint code-fold accumulator (per block)
    const ACCVAR: u32 = 45; //  float output accumulator
    const GIDV: u32 = 46;
    const R: u32 = 47;
    const S: u32 = 48;
    const W0: u32 = 49;
    const WEND_RAW: u32 = 50;
    const OVER: u32 = 51;
    const WEND: u32 = 52;
    const OI1: u32 = 53;
    const OI2: u32 = 54;
    const IN_RANGE: u32 = 55;
    const B_THEN: u32 = 56;
    const B_ENDIF: u32 = 57;
    const B_OHDR: u32 = 58;
    const B_OCHK: u32 = 59;
    const B_OBODY: u32 = 60;
    const B_OCONT: u32 = 61;
    const B_OMERGE: u32 = 62;
    const OCHK_C: u32 = 63;
    const OCOND: u32 = 64;
    const BB: u32 = 65;
    // block prelude (fused d',m + XS)
    const DM1: u32 = 66;
    const DM2: u32 = 67;
    const DMIDX: u32 = 68;
    const DMIDX1: u32 = 69;
    const PD: u32 = 70;
    const DV: u32 = 71;
    const PM: u32 = 72;
    const MV: u32 = 73;
    const PXS: u32 = 74;
    const XSV: u32 = 75;
    const B_IHDR: u32 = 76;
    const B_ICHK: u32 = 77;
    const B_IBODY: u32 = 78;
    const B_ICONT: u32 = 79;
    const B_IMERGE: u32 = 80;
    const JJ: u32 = 81;
    const ICOND: u32 = 82;
    const CBASE: u32 = 83; //   b·cpb
    const CC: u32 = 84; //      chunk index c = b·cpb + cj
    const WIDX1: u32 = 85; //   c·M
    const WIDX: u32 = 86; //    c·M + r
    const PW: u32 = 87;
    const WWORD: u32 = 88; //   the packed codes (4 bytes Q8, 8 nibbles Q4)
    const CDOTS: u32 = 89; //   dpw·c — the phase activation base index (split carriers)
                           // The per-phase DP4A temporaries live at ids `200 + l·8 + {0..7}` (allocated in the inner
                           // loop); the shift constants `wbits·l` at `181..183`. Q8 uses one phase (the word whole).
    const J2: u32 = 115;
    const JN: u32 = 116;
    // block epilogue: acc += d'·float(Σqx) + m·XS
    const ACCQF_S: u32 = 96;
    const ACCQF: u32 = 97;
    const DQ: u32 = 98;
    const MXS: u32 = 99;
    const SUM2: u32 = 100;
    const AB0: u32 = 101;
    const AB1: u32 = 102;
    const CTR2: u32 = 103;
    const CTRN: u32 = 104;
    const ACCF: u32 = 105;
    const POUT: u32 = 106;
    const SXBUF: u32 = 36; //  dyn_sx: the activation-scale SSBO (binding 5)
    const PSX: u32 = 37;
    const SXV: u32 = 38;
    const DQS: u32 = 39; //    dyn_sx: (d·Σqx)·sx
    const GLSL: u32 = 117; //  dyn_sx: GLSL.std.450 (UnpackHalf2x16 for the f16 DM pair)
    const VEC2F: u32 = 118;
    const DMW: u32 = 119; //   the packed (d, lo) word
    const DMUNP: u32 = 120; // its vec2 unpack
                            // MoE-only ids: chosen at 240..247, clear of the phase temporaries at 200 + l·8 (<= 231)
                            // and the phase shift constants at 181..183.
    const SEL: u32 = 240; //     binding 6: the selected expert per slot
    const PSEL: u32 = 241;
    const EIDX: u32 = 242; //    SEL[gid.y]
    const WBASE: u32 = 243; //   expert · words_per_expert
    const DMBASE: u32 = 244; //  expert · dm_per_expert
    const C_WPE: u32 = 245;
    const C_DMPE: u32 = 246;
    const WIDX2: u32 = 247; //   chart index before the expert base
    const DMIDX2: u32 = 248;
    // Weighted-epilogue ids, 249..252 — deliberately above the MoE block and still under BOUND.
    const SWT: u32 = 249; //     binding 7: the per-slot mixture weight (MUL_ROUTED_WEIGHT)
    const PSWT: u32 = 250;
    const SWTV: u32 = 251; //    SWT[gid.y]
    const ACCW: u32 = 252; //    SWT[gid.y] · acc — what the weighted store writes
                           // Per-slot ACTIVATION base, 253..258 — still under BOUND, so `x_slot` costs no bound bump.
    const C_XPE: u32 = 253; //   activation words per slot = k/4 (int8, 4 per word)
    const C_XBPE: u32 = 254; //  activation 32-blocks per slot = k/32
    const XBASE: u32 = 255; //   gid.y · C_XPE
    const XBBASE: u32 = 256; //  gid.y · C_XBPE
    const BBX: u32 = 257; //     block index + XBBASE (XS/SX)
    const XQIB: u32 = 258; //    activation word index + XBASE (XQ)
    const BOUND: u32 = 260; // phase temporaries live at 200 + l·8 (l ≤ 3) → ≤ 231; shifts 181..183

    // ── TWO-LEVEL scale face, ids 260..286 ────────────────────────────────────────────────────
    // Seven constants and twenty temporaries, all ABOVE the dense module's bound of 260, so the
    // bound is bumped only when the face is asked for and every existing module stays
    // byte-identical (which `SEQHASH` in `gpu_decode_bench` is the gate for).
    const C_MROW: u32 = 260; //   ⌈m/2⌉ — the `sub` row-pair stride
    const C_SUPOFF: u32 = 261; // blocks·⌈m/2⌉ — where `sup` starts in the SAME buffer
    const C_U4: u32 = 262;
    const C_U8: u32 = 263;
    const C_FF: u32 = 264;
    const C_80: u32 = 265;
    const C_NEG80: u32 = 266; // 0xffffff80: +this == −0x80 in wrapping uint, so no OpISub needed
    const SUPI: u32 = 267; //     C_SUPOFF + R
    const PSUP: u32 = 268;
    const SUPW: u32 = 269;
    const SUPUNP: u32 = 270;
    const DSUP: u32 = 271;
    const LOSUP: u32 = 272;
    const SB1: u32 = 273; //      BB · C_MROW
    const RH: u32 = 274; //       R >> 1
    const SB2: u32 = 275;
    const PSUB: u32 = 276;
    const SUBW: u32 = 277;
    const PAR: u32 = 278; //      R & 1
    const SHF: u32 = 279; //      PAR << 4  (= PAR·16)
    const HALF: u32 = 280; //     the 16 bits this row owns
    const D8: u32 = 281;
    const HI8: u32 = 282;
    const L8: u32 = 283;
    const XR: u32 = 284;
    const SB8: u32 = 285;
    const SI8: u32 = 286;
    const D8F: u32 = 287;
    const L8F: u32 = 288;
    const TL_BOUND: u32 = 289;

    // ── Carrier::Q3's TWO-PLANE decode, ids 289..332 ───────────────────────────────────────────
    // Plane A rides Q2's phase loop untouched; plane B contributes one high bit per code. Its
    // block word is loop-invariant across `cj` and `l`, so it is fetched in the block prelude.
    const C_PBOFF: u32 = 289; // plane B's element offset inside `w`
    const C_U16: u32 = 290;
    const C_0F: u32 = 291;
    const C_SPREAD: u32 = 292; // 0x00204081 — scatters nibble bit i to byte i
    const C_ONE8: u32 = 293; // 0x01010101
                             // the per-phase bit offset 4·l, for l = 1..3 (l = 0 needs no add). The 180+l constants above
                             // hold `wbits·l`, which is a DIFFERENT ladder — reusing them here read 2/4/6, not 4/8/12.
    const C_Q3SH: u32 = 294; // 294,295,296 = 4, 8, 12
    const C_TWO: u32 = 297;
    const BIDX1: u32 = 298;
    const BIDX2: u32 = 299;
    const BIDX3: u32 = 300; // SSA: three ids, not one reassigned three times
    const PBP: u32 = 301;
    const BWORD: u32 = 302; // the block's 32 permuted high bits
                            // phase temporaries at 320 + l·8 (l <= 3) -> <= 351, mirroring the 200 + l·8 convention above
    const Q3_BOUND: u32 = 352;

    // Conditional bound, unlike the MoE ids (240..252) which already fit under 260: the two-level
    // temporaries do not, and bumping unconditionally would change one header word in every
    // module this emitter produces.
    let q3 = carrier == Carrier::Q3;
    assert!(
        !q3 || two_level,
        "Carrier::Q3 is only wired against the two-level scale face — its 3.5 bpw figure in \
results/scale-ladder.md is 3.0 of codes plus 0.5 of scale, and the single-level face would make \
it 4.0 and pointless"
    );
    let bound = if q3 {
        Q3_BOUND
    } else if two_level {
        TL_BOUND
    } else {
        BOUND
    };
    let mut w = vec![SPV_MAGIC, SPV_VERSION_1_3, 0, bound, 0];
    // capabilities → extensions → imports → memory model (the spec's logical-layout order)
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_DOT_PRODUCT]);
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_DOT_INPUT_4X8_PACKED]);
    let ext = lit_string("SPV_KHR_integer_dot_product");
    w.push(((1 + ext.len() as u32) << 16) | OP_EXTENSION);
    w.extend(&ext);
    // the two-level face unpacks its per-row super-pair with UnpackHalf2x16 too, so it needs the
    // import even on a `dyn_sx = false` module
    if dyn_sx || two_level {
        let g = lit_string("GLSL.std.450");
        w.push(((2 + g.len() as u32) << 16) | OP_EXT_INST_IMPORT);
        w.push(GLSL);
        w.extend(&g);
    }
    w.extend([(3 << 16) | OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450]);
    let main = lit_string("main");
    w.push(((3 + main.len() as u32 + 1) << 16) | OP_ENTRY_POINT);
    w.extend([MODEL_GL_COMPUTE, MAIN]);
    w.extend(&main);
    w.push(GID);
    w.extend([
        (6 << 16) | OP_EXECUTION_MODE,
        MAIN,
        MODE_LOCAL_SIZE,
        local,
        1,
        1,
    ]);
    // decorations
    w.extend([
        (4 << 16) | OP_DECORATE,
        GID,
        DEC_BUILT_IN,
        BUILTIN_GLOBAL_INVOCATION_ID,
    ]);
    w.extend([(4 << 16) | OP_DECORATE, RTA_U, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, BUF_U, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, BUF_U, 0, DEC_OFFSET, 0]);
    w.extend([(4 << 16) | OP_DECORATE, RTA_F, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, BUF_F, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, BUF_F, 0, DEC_OFFSET, 0]);
    let mut binds: Vec<(u32, u32)> = vec![(W, 0u32), (XQ, 1), (OUT, 2), (DM, 3), (XS, 4)];
    if dyn_sx {
        binds.push((SXBUF, 5));
    }
    if moe.is_some() {
        binds.push((SEL, 6));
    }
    if weighted {
        binds.push((SWT, 7));
    }
    for (var, binding) in binds {
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_BINDING, binding]);
    }
    // types
    w.extend([(2 << 16) | OP_TYPE_VOID, VOID]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, FNTY, VOID]);
    w.extend([(4 << 16) | OP_TYPE_INT, UINT, 32, 0]);
    w.extend([(4 << 16) | OP_TYPE_INT, SINT, 32, 1]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, FLOAT, 32]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, BOOL]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, UVEC3, UINT, 3]);
    if dyn_sx {
        w.extend([(4 << 16) | OP_TYPE_VECTOR, VEC2F, FLOAT, 2]);
    }
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_IN_UVEC3, SC_INPUT, UVEC3]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_IN_UVEC3, GID, SC_INPUT]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, RTA_U, UINT]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, BUF_U, RTA_U]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_BUF_U,
        SC_STORAGE_BUFFER,
        BUF_U,
    ]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, RTA_F, FLOAT]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, BUF_F, RTA_F]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_BUF_F,
        SC_STORAGE_BUFFER,
        BUF_F,
    ]);
    // variables: W, XQ (+ f16-packed DM when dyn) are uint buffers; the rest float
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_U, W, SC_STORAGE_BUFFER]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_U, XQ, SC_STORAGE_BUFFER]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        PTR_SB_BUF_F,
        OUT,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        if dyn_sx { PTR_SB_BUF_U } else { PTR_SB_BUF_F },
        DM,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, XS, SC_STORAGE_BUFFER]);
    if dyn_sx {
        w.extend([
            (4 << 16) | OP_VARIABLE,
            PTR_SB_BUF_F,
            SXBUF,
            SC_STORAGE_BUFFER,
        ]);
    }
    if moe.is_some() {
        w.extend([
            (4 << 16) | OP_VARIABLE,
            PTR_SB_BUF_U,
            SEL,
            SC_STORAGE_BUFFER,
        ]);
    }
    if weighted {
        w.extend([
            (4 << 16) | OP_VARIABLE,
            PTR_SB_BUF_F,
            SWT,
            SC_STORAGE_BUFFER,
        ]);
    }
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_U,
        SC_STORAGE_BUFFER,
        UINT,
    ]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_F,
        SC_STORAGE_BUFFER,
        FLOAT,
    ]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_U, SC_FUNCTION, UINT]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_S, SC_FUNCTION, SINT]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_F, SC_FUNCTION, FLOAT]);
    // constants
    let mut consts: Vec<(u32, u32)> = vec![
        (C_U0, 0),
        (C_U1, 1),
        (C_M, m),
        (C_BLOCKS, blocks),
        (C_CPB, cpb),
    ];
    if split {
        consts.push((C_DOTS, dots_per_word)); //      activation stride = dots per word
        consts.push((C_NIBMASK, byte_mask)); //       per-byte code mask (0F0F… / 0303…)
        for l in 1..dots_per_word {
            consts.push((180 + l, wbits * l)); //     shift amount wbits·l for phase l (ids 181..183)
        }
    }
    if segments > 1 {
        consts.push((C_SEG, blocks.div_ceil(segments)));
    }
    if let Some((wpe, dmpe, _, _)) = moe {
        consts.extend([(C_WPE, wpe), (C_DMPE, dmpe)]);
    }
    if x_slot {
        consts.extend([(C_XPE, k / 4), (C_XBPE, k / 32)]);
    }
    if q3 {
        consts.extend([
            (C_PBOFF, carrier.plane_b_off(m, k)),
            (C_U16, 16),
            (C_0F, 0xF),
            (C_SPREAD, 0x0020_4081),
            (C_ONE8, 0x0101_0101),
            (C_TWO, 2),
            (C_Q3SH, 4),
            (C_Q3SH + 1, 8),
            (C_Q3SH + 2, 12),
        ]);
    }
    if two_level {
        let mrow = m.div_ceil(2);
        consts.extend([
            (C_MROW, mrow),
            (C_SUPOFF, blocks * mrow),
            (C_U4, 4),
            (C_U8, 8),
            (C_FF, 0xff),
            (C_80, 0x80),
            (C_NEG80, 0xffff_ff80),
        ]);
    }
    for (id, v) in consts {
        w.extend([(4 << 16) | OP_CONSTANT, UINT, id, v]);
    }
    w.extend([(4 << 16) | OP_CONSTANT, FLOAT, C_F0, 0f32.to_bits()]);
    w.extend([(4 << 16) | OP_CONSTANT, SINT, C_S0, 0]);
    // main
    w.extend([(5 << 16) | OP_FUNCTION, VOID, MAIN, 0, FNTY]);
    w.extend([(2 << 16) | OP_LABEL, B_ENTRY]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_U, CVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_U, JVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_S, ACCQVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_F, ACCVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_LOAD, UVEC3, GIDV, GID]);
    w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, UINT, R, GIDV, 0]);
    if grid_y {
        w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, UINT, S, GIDV, 1]);
        if segments > 1 {
            w.extend([(5 << 16) | OP_IMUL, UINT, W0, S, C_SEG]);
            w.extend([(5 << 16) | OP_IADD, UINT, WEND_RAW, W0, C_SEG]);
            w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, OVER, C_BLOCKS, WEND_RAW]);
            w.extend([(6 << 16) | OP_SELECT, UINT, WEND, OVER, C_BLOCKS, WEND_RAW]);
        }
        if moe.is_some() {
            // expert = SEL[slot]; chart and scales start at a whole-expert stride
            w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_U, PSEL, SEL, C_U0, S]);
            w.extend([(4 << 16) | OP_LOAD, UINT, EIDX, PSEL]);
            w.extend([(5 << 16) | OP_IMUL, UINT, WBASE, EIDX, C_WPE]);
            w.extend([(5 << 16) | OP_IMUL, UINT, DMBASE, EIDX, C_DMPE]);
        }
        if x_slot {
            // the activation base is the SLOT's, not the expert's: `silu_mul_q_slot` wrote this
            // slot's SwiGLU output at `slot·(k/4)` words and its scales at `slot·(k/32)`.
            w.extend([(5 << 16) | OP_IMUL, UINT, XBASE, S, C_XPE]);
            w.extend([(5 << 16) | OP_IMUL, UINT, XBBASE, S, C_XBPE]);
        }
        w.extend([(5 << 16) | OP_IMUL, UINT, OI1, S, C_M]);
        w.extend([(5 << 16) | OP_IADD, UINT, OI2, OI1, R]);
    }
    w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, IN_RANGE, R, C_M]);
    w.extend([(3 << 16) | OP_SELECTION_MERGE, B_ENDIF, 0]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, IN_RANGE, B_THEN, B_ENDIF]);
    w.extend([(2 << 16) | OP_LABEL, B_THEN]);
    w.extend([
        (3 << 16) | OP_STORE,
        CVAR,
        if segments > 1 { W0 } else { C_U0 },
    ]);
    w.extend([(3 << 16) | OP_STORE, ACCVAR, C_F0]);
    if two_level {
        // ── ROW-INVARIANT half of the two-level decode, hoisted out of the block loop ──
        // `sup[R]`, its unpack, and the row-parity shift depend only on R. Emitting them here
        // (B_THEN dominates the loop, so the definitions dominate their uses) takes a dependent
        // GLOBAL LOAD plus eight instructions out of a loop that runs `blocks` times per row —
        // 160 times at the 27B's k = 5120.
        //
        // The first version left them inside, on the argument that decode is bandwidth-bound and
        // the load hits one cache line. Then the 27B went 13.00 -> 11.60 GB (-10.8% of the bytes)
        // and 46.3 -> 46.0 tok/s (FLAT), which is a measurement that argument cannot survive.
        w.extend([(5 << 16) | OP_IADD, UINT, SUPI, C_SUPOFF, R]);
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_U, PSUP, DM, C_U0, SUPI]);
        w.extend([(4 << 16) | OP_LOAD, UINT, SUPW, PSUP]);
        w.extend([(6 << 16) | OP_EXT_INST, VEC2F, SUPUNP, GLSL, 62, SUPW]);
        w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, FLOAT, DSUP, SUPUNP, 0]);
        w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, FLOAT, LOSUP, SUPUNP, 1]);
        w.extend([(5 << 16) | OP_SHIFT_RIGHT_LOGICAL, UINT, RH, R, C_U1]);
        w.extend([(5 << 16) | OP_BITWISE_AND, UINT, PAR, R, C_U1]);
        w.extend([(5 << 16) | OP_SHIFT_LEFT_LOGICAL, UINT, SHF, PAR, C_U4]);
    }
    w.extend([(2 << 16) | OP_BRANCH, B_OHDR]);
    // outer loop over 32-blocks
    w.extend([(2 << 16) | OP_LABEL, B_OHDR]);
    w.extend([(4 << 16) | OP_LOOP_MERGE, B_OMERGE, B_OCONT, 0]);
    w.extend([(2 << 16) | OP_BRANCH, B_OCHK]);
    w.extend([(2 << 16) | OP_LABEL, B_OCHK]);
    w.extend([(4 << 16) | OP_LOAD, UINT, OCHK_C, CVAR]);
    w.extend([
        (5 << 16) | OP_ULESS_THAN,
        BOOL,
        OCOND,
        OCHK_C,
        if segments > 1 { WEND } else { C_BLOCKS },
    ]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, OCOND, B_OBODY, B_OMERGE]);
    w.extend([(2 << 16) | OP_LABEL, B_OBODY]);
    w.extend([(4 << 16) | OP_LOAD, UINT, BB, CVAR]);
    // block prelude: fused (d', m) — dyn: ONE f16-packed word; bench: two f32 slots — and Σx
    w.extend([(5 << 16) | OP_IMUL, UINT, DM1, BB, C_M]);
    w.extend([(5 << 16) | OP_IADD, UINT, DM2, DM1, R]);
    if two_level {
        // **THE TWO-LEVEL FACE** — `d = d8·dsup` with d8 UNSIGNED 0..255, `lo = i8(l8)·losup` with
        // l8 SIGNED, reproducing `fold_scales_two_level` exactly. Half the scale metadata of the
        // `dyn_sx` branch below for values `results/scale-ladder.md` measures as identical to six
        // decimals, which at the 27B is 13.00 -> 11.3 GB (`results/where-the-27b-bits-go.md`).
        //
        // `sup` shares DM's binding at a compile-time offset, so nothing after binding 3 moves.
        // The super-pair is re-read per block rather than hoisted out of the loop: decode is
        // bandwidth- and dispatch-bound (`results/decode-surface-audit.md`), the load hits one
        // cache line every iteration, and hoisting would mean restructuring the loop header for a
        // saving no measurement on this path can see.
        // Only the BB-dependent half is here; `sup`, its unpack and the row-parity shift were
        // hoisted above the loop. sub[BB·mrow + R/2], then the 16-bit half this row owns.
        w.extend([(5 << 16) | OP_IMUL, UINT, SB1, BB, C_MROW]);
        w.extend([(5 << 16) | OP_IADD, UINT, SB2, SB1, RH]);
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_U, PSUB, DM, C_U0, SB2]);
        w.extend([(4 << 16) | OP_LOAD, UINT, SUBW, PSUB]);
        w.extend([(5 << 16) | OP_SHIFT_RIGHT_LOGICAL, UINT, HALF, SUBW, SHF]);
        w.extend([(5 << 16) | OP_BITWISE_AND, UINT, D8, HALF, C_FF]);
        w.extend([(5 << 16) | OP_SHIFT_RIGHT_LOGICAL, UINT, HI8, HALF, C_U8]);
        w.extend([(5 << 16) | OP_BITWISE_AND, UINT, L8, HI8, C_FF]);
        // sign-extend the byte with `(x ^ 0x80) + (−0x80)`: wraps in uint to exactly i8's value
        // over 0..255. This emitter has no OpShiftRightArithmetic and no OpISub in scope.
        w.extend([(5 << 16) | OP_BITWISE_XOR, UINT, XR, L8, C_80]);
        w.extend([(5 << 16) | OP_IADD, UINT, SB8, XR, C_NEG80]);
        w.extend([(4 << 16) | OP_BITCAST, SINT, SI8, SB8]);
        w.extend([(4 << 16) | OP_CONVERT_U_TO_F, FLOAT, D8F, D8]);
        w.extend([(4 << 16) | OP_CONVERT_S_TO_F, FLOAT, L8F, SI8]);
        w.extend([(5 << 16) | OP_FMUL, FLOAT, DV, D8F, DSUP]);
        w.extend([(5 << 16) | OP_FMUL, FLOAT, MV, L8F, LOSUP]);
    } else if dyn_sx {
        // f16-packed DM: one u32 word (d16 | lo16<<16) per (block,row), col-major idx = b·M + R (DM2)
        // MoE offsets that by a whole expert's worth of words. This branch is the one the SERVED
        // decode takes (`dyn_sx = true`), so missing it here meant every slot read expert 0's
        // scales while reading its own codes — off by a factor, not by a shape, and the only thing
        // that caught it was grading against the CPU fold.
        if moe.is_some() {
            w.extend([(5 << 16) | OP_IADD, UINT, DMIDX2, DM2, DMBASE]);
        }
        w.extend([
            (6 << 16) | OP_ACCESS_CHAIN,
            PTR_SB_U,
            PD,
            DM,
            C_U0,
            if moe.is_some() { DMIDX2 } else { DM2 },
        ]);
        w.extend([(4 << 16) | OP_LOAD, UINT, DMW, PD]);
        w.extend([(6 << 16) | OP_EXT_INST, VEC2F, DMUNP, GLSL, 62, DMW]); // UnpackHalf2x16 → (d, lo)
        w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, FLOAT, DV, DMUNP, 0]);
        w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, FLOAT, MV, DMUNP, 1]);
    } else {
        if moe.is_some() {
            w.extend([(5 << 16) | OP_IADD, UINT, DMIDX2, DM2, DM2]);
            w.extend([(5 << 16) | OP_IADD, UINT, DMIDX, DMIDX2, DMBASE]);
        } else {
            w.extend([(5 << 16) | OP_IADD, UINT, DMIDX, DM2, DM2]);
        }
        w.extend([(5 << 16) | OP_IADD, UINT, DMIDX1, DMIDX, C_U1]);
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PD, DM, C_U0, DMIDX]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, DV, PD]);
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PM, DM, C_U0, DMIDX1]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, MV, PM]);
    }
    // BB stays raw for the DM/weight indices (those are per-EXPERT); only the activation faces
    // take the slot base.
    let bbx = if x_slot {
        w.extend([(5 << 16) | OP_IADD, UINT, BBX, BB, XBBASE]);
        BBX
    } else {
        BB
    };
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PXS, XS, C_U0, bbx]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, XSV, PXS]);
    if dyn_sx {
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PSX, SXBUF, C_U0, bbx]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, SXV, PSX]);
    }
    if q3 {
        // plane B: ONE u32 per (32-block, row), at `C_PBOFF + BB·m + R`
        w.extend([(5 << 16) | OP_IMUL, UINT, BIDX1, BB, C_M]);
        w.extend([(5 << 16) | OP_IADD, UINT, BIDX2, BIDX1, R]);
        w.extend([(5 << 16) | OP_IADD, UINT, BIDX3, BIDX2, C_PBOFF]);
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_U, PBP, W, C_U0, BIDX3]);
        w.extend([(4 << 16) | OP_LOAD, UINT, BWORD, PBP]);
    }
    w.extend([(3 << 16) | OP_STORE, ACCQVAR, C_S0]);
    w.extend([(3 << 16) | OP_STORE, JVAR, C_U0]);
    w.extend([(2 << 16) | OP_BRANCH, B_IHDR]);
    // inner loop: cpb chunks; each a weight word → 1 (Q8) or 2 (Q4) DP4A (c = b·cpb + cj)
    w.extend([(2 << 16) | OP_LABEL, B_IHDR]);
    w.extend([(4 << 16) | OP_LOOP_MERGE, B_IMERGE, B_ICONT, 0]);
    w.extend([(2 << 16) | OP_BRANCH, B_ICHK]);
    w.extend([(2 << 16) | OP_LABEL, B_ICHK]);
    w.extend([(4 << 16) | OP_LOAD, UINT, JJ, JVAR]);
    w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, ICOND, JJ, C_CPB]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, ICOND, B_IBODY, B_IMERGE]);
    w.extend([(2 << 16) | OP_LABEL, B_IBODY]);
    w.extend([(5 << 16) | OP_IMUL, UINT, CBASE, BB, C_CPB]);
    w.extend([(5 << 16) | OP_IADD, UINT, CC, CBASE, JJ]);
    // weight word W[c·M + r] (coalesced across r) — one load, 4 (Q8) or 8 (Q4) codes
    w.extend([(5 << 16) | OP_IMUL, UINT, WIDX1, CC, C_M]);
    if moe.is_some() {
        w.extend([(5 << 16) | OP_IADD, UINT, WIDX2, WIDX1, R]);
        w.extend([(5 << 16) | OP_IADD, UINT, WIDX, WIDX2, WBASE]);
    } else {
        w.extend([(5 << 16) | OP_IADD, UINT, WIDX, WIDX1, R]);
    }
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_U, PW, W, C_U0, WIDX]);
    w.extend([(4 << 16) | OP_LOAD, UINT, WWORD, PW]);
    // The N-way split: `dots_per_word` DP4A per weight word. Phase l extracts one code per byte
    // via `(WWORD >> wbits·l) & byte_mask` — Q8 (dpw=1) dots the word whole; Q4 (dpw=2) is even/odd
    // nibbles; Q2 (dpw=4) is the four 2-bit phases (the ternary/trit floor). Each phase dots against
    // the deinterleaved activation word XQ[dpw·c + l]; same accumulator, 8 code·MACs per 32-block.
    // Per-phase temporaries live at ids 200 + l·8 + {0:src,1:shifted,2:xqidx,3:pxw,4:xw,5:dot,6,7}.
    let mut xq_cur = CC;
    if split {
        w.extend([(5 << 16) | OP_IMUL, UINT, CDOTS, CC, C_DOTS]); // CDOTS = dpw·c
        xq_cur = CDOTS;
    }
    if x_slot {
        // one add per block-chunk, before the phase chain — every phase index derives from this
        w.extend([(5 << 16) | OP_IADD, UINT, XQIB, xq_cur, XBASE]);
        xq_cur = XQIB;
    }
    for l in 0..dots_per_word {
        let p = 200 + l * 8;
        let xqi = if l == 0 {
            xq_cur
        } else {
            w.extend([(5 << 16) | OP_IADD, UINT, p + 2, xq_cur, C_U1]); // XQ index += 1
            xq_cur = p + 2;
            p + 2
        };
        // SRC: the codes this phase dots (4, one per byte). Q8: the word whole.
        let src = if !split {
            WWORD
        } else if l == 0 {
            w.extend([(5 << 16) | OP_BITWISE_AND, UINT, p, WWORD, C_NIBMASK]);
            p
        } else {
            w.extend([
                (5 << 16) | OP_SHIFT_RIGHT_LOGICAL,
                UINT,
                p + 1,
                WWORD,
                180 + l,
            ]);
            w.extend([(5 << 16) | OP_BITWISE_AND, UINT, p, p + 1, C_NIBMASK]);
            p
        };
        // Q3: four CONTIGUOUS high bits for phase (JJ, l), expanded one per byte at bit 2.
        // `q3_planes_gpu_chart` permuted plane B so these bits are adjacent; at the natural
        // stride of 4 the multiply's partial products collide and half the lanes come out wrong.
        let src = if q3 {
            let q = 320 + l * 8;
            w.extend([(5 << 16) | OP_IMUL, UINT, q, JJ, C_U16]); // JJ·16
            let sh = if l == 0 {
                q // 4·l is zero, so no add
            } else {
                w.extend([(5 << 16) | OP_IADD, UINT, q + 1, q, C_Q3SH + l - 1]);
                q + 1
            };
            w.extend([(5 << 16) | OP_SHIFT_RIGHT_LOGICAL, UINT, q + 2, BWORD, sh]);
            w.extend([(5 << 16) | OP_BITWISE_AND, UINT, q + 3, q + 2, C_0F]);
            w.extend([(5 << 16) | OP_IMUL, UINT, q + 4, q + 3, C_SPREAD]);
            w.extend([(5 << 16) | OP_BITWISE_AND, UINT, q + 5, q + 4, C_ONE8]);
            w.extend([(5 << 16) | OP_SHIFT_LEFT_LOGICAL, UINT, q + 6, q + 5, C_TWO]);
            w.extend([(5 << 16) | OP_BITWISE_OR, UINT, q + 7, src, q + 6]);
            q + 7
        } else {
            src
        };
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_U, p + 3, XQ, C_U0, xqi]);
        w.extend([(4 << 16) | OP_LOAD, UINT, p + 4, p + 3]);
        w.extend([
            (6 << 16) | OP_SUDOT,
            SINT,
            p + 5,
            p + 4,
            src,
            PACKED_VEC_4X8,
        ]);
        w.extend([(4 << 16) | OP_LOAD, SINT, p + 6, ACCQVAR]);
        w.extend([(5 << 16) | OP_IADD, SINT, p + 7, p + 6, p + 5]);
        w.extend([(3 << 16) | OP_STORE, ACCQVAR, p + 7]);
    }
    w.extend([(2 << 16) | OP_BRANCH, B_ICONT]);
    w.extend([(2 << 16) | OP_LABEL, B_ICONT]);
    w.extend([(4 << 16) | OP_LOAD, UINT, J2, JVAR]);
    w.extend([(5 << 16) | OP_IADD, UINT, JN, J2, C_U1]);
    w.extend([(3 << 16) | OP_STORE, JVAR, JN]);
    w.extend([(2 << 16) | OP_BRANCH, B_IHDR]);
    // block epilogue: acc += d'·float(Σqx) + m·XS_b
    w.extend([(2 << 16) | OP_LABEL, B_IMERGE]);
    w.extend([(4 << 16) | OP_LOAD, SINT, ACCQF_S, ACCQVAR]);
    w.extend([(4 << 16) | OP_CONVERT_S_TO_F, FLOAT, ACCQF, ACCQF_S]);
    w.extend([(5 << 16) | OP_FMUL, FLOAT, DQ, DV, ACCQF]);
    let dq_final = if dyn_sx {
        w.extend([(5 << 16) | OP_FMUL, FLOAT, DQS, DQ, SXV]);
        DQS
    } else {
        DQ
    };
    w.extend([(5 << 16) | OP_FMUL, FLOAT, MXS, MV, XSV]);
    w.extend([(5 << 16) | OP_FADD, FLOAT, SUM2, dq_final, MXS]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, AB0, ACCVAR]);
    w.extend([(5 << 16) | OP_FADD, FLOAT, AB1, AB0, SUM2]);
    w.extend([(3 << 16) | OP_STORE, ACCVAR, AB1]);
    w.extend([(2 << 16) | OP_BRANCH, B_OCONT]);
    w.extend([(2 << 16) | OP_LABEL, B_OCONT]);
    w.extend([(4 << 16) | OP_LOAD, UINT, CTR2, CVAR]);
    w.extend([(5 << 16) | OP_IADD, UINT, CTRN, CTR2, C_U1]);
    w.extend([(3 << 16) | OP_STORE, CVAR, CTRN]);
    w.extend([(2 << 16) | OP_BRANCH, B_OHDR]);
    w.extend([(2 << 16) | OP_LABEL, B_OMERGE]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, ACCF, ACCVAR]);
    // MUL_ROUTED_WEIGHT: one FMUL per output element, once, after the whole k-fold has landed in
    // f32 — never inside the block loop (that would scale `m·XS` per block as well as `d·Σqx`, the
    // same result but `blocks` times the arithmetic). The slot stays its own row of OUT.
    let stored = if weighted {
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PSWT, SWT, C_U0, S]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, SWTV, PSWT]);
        w.extend([(5 << 16) | OP_FMUL, FLOAT, ACCW, SWTV, ACCF]);
        ACCW
    } else {
        ACCF
    };
    w.extend([
        (6 << 16) | OP_ACCESS_CHAIN,
        PTR_SB_F,
        POUT,
        OUT,
        C_U0,
        if grid_y { OI2 } else { R },
    ]);
    w.extend([(3 << 16) | OP_STORE, POUT, stored]);
    w.extend([(2 << 16) | OP_BRANCH, B_ENDIF]);
    w.extend([(2 << 16) | OP_LABEL, B_ENDIF]);
    w.extend([(1 << 16) | OP_RETURN]);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The integer-dot GEMV module as bytes.
pub fn gemv_dp4a_bytes(m: u32, k: u32, local: u32, segments: u32, carrier: Carrier) -> Vec<u8> {
    gemv_dp4a_module(m, k, local, segments, carrier)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// The dynamic-activation-scale integer-dot GEMV as bytes (binding 5 = `sx[0]`).
pub fn gemv_dp4a_dyn_bytes(m: u32, k: u32, local: u32, segments: u32, carrier: Carrier) -> Vec<u8> {
    gemv_dp4a_dyn_bytes_sc(m, k, local, segments, carrier, false)
}

/// [`gemv_dp4a_dyn_bytes`] with the **scale face** chosen. `two_level` reads `qsub`/`qsup` off
/// binding 3 in place of `dm` — half the scale metadata for values `results/scale-ladder.md`
/// measures as identical to six decimals. Bindings are UNCHANGED: `sup` shares `dm`'s buffer at a
/// compile-time offset (`crate::gpu::nn::ChartScale::two_level`, and
/// `crate::gpu::nn::two_level_chart_buffer` builds the layout both sides agree on).
pub fn gemv_dp4a_dyn_bytes_sc(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: Carrier,
    two_level: bool,
) -> Vec<u8> {
    gemv_dp4a_module_moe_sc(m, k, local, segments, carrier, true, None, two_level)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// **The batch-tiled integer-dot GEMM — PREFILL's kernel.** `OUT[i·M + r] = Σ_k W[r,k]·X[i,k]` for a
/// tile of `bt` activation columns `i` at once, on the SAME chart layout the decode gemv walks
/// ([`gemv_dp4a_module_dyn`] with `dyn_sx`), so a model packed for decode needs no re-pack.
///
/// **Why this module has to exist.** Prefill is compute-bound; the two modules that could already
/// carry a batch both fail it for a stated reason, measured in `examples/prefill_kernels`:
/// [`gemv_batch_module`] is f32 and re-reads the whole weight per batch element (zero reuse ⇒ DRAM
/// bound: 0.45 TF/s at the 0.5B lm head, 1.1k tok/s whole-stack), and [`coopmat_gemm_module`] has the
/// intensity (10.7 TF/s) but its operands are f16 — it cannot consume the ternary chart, and staging
/// the weights as f16 is 8× the VRAM the DP4A carrier costs. This kernel keeps `OpSUDot` (portable
/// `integerDotProduct4x8BitPacked`, not a vendor path) and gets the intensity from REUSE: one
/// `W[c·M + r]` load feeds `bt·dots_per_word` dots instead of one, so weight traffic falls by `bt`
/// while the arithmetic stays the same. `bt = 1` is byte-equivalent to the decode gemv.
///
/// **Layout.** Bindings as the dyn gemv, with every activation face batch-strided:
/// 0 = W (u32 codes, k-packed `W[c·M + r]`, SHARED — the whole point), 1 = XQ (u32 int8 activations,
/// row `i` at `i·k/4`), 2 = OUT (f32, `i·M + r`), 3 = DM (u32 f16-packed `(d, lo)` at `b·M + r`,
/// SHARED), 4 = XS (f32 `Σ_block x`, `i·blocks + b`), 5 = SX (f32 per-block activation scale,
/// `i·blocks + b`).
///
/// **Dispatch** `groups = (ceil(M/local), B/bt)`; `gl_GlobalInvocationId.x` = row, `.y` = tile, so
/// batch element `i = tile·bt + j`. **The caller zero-pads B to a multiple of `bt`** (as
/// [`coopmat_gemm_module`] pads to 16) — the kernel carries no batch guard, and a zero activation row
/// folds to zero, so the padded OUT rows are discardable. No split-K: at prefill the batch already
/// fills the machine, and split-K would need a `bt`-wide partial buffer.
pub fn gemv_dp4a_batch_module(m: u32, k: u32, local: u32, carrier: Carrier, bt: u32) -> Vec<u32> {
    assert!(k % 32 == 0, "chart carriers need k % 32 == 0");
    assert!(carrier.is_chart(), "DP4A is a chart carrier (Q8/Q4/Q2)");
    assert!(bt >= 1, "batch tile must be >= 1");
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
    let void = id!();
    let fnty = id!();
    let uint = id!();
    let sint = id!();
    let float = id!();
    let bool_t = id!();
    let uvec3 = id!();
    let vec2f = id!();
    let ptr_in_uvec3 = id!();
    let gid = id!();
    let rta_u = id!();
    let buf_u = id!();
    let ptr_sb_buf_u = id!();
    let rta_f = id!();
    let buf_f = id!();
    let ptr_sb_buf_f = id!();
    let v_w = id!();
    let v_xq = id!();
    let v_out = id!();
    let v_dm = id!();
    let v_xs = id!();
    let v_sx = id!();
    let ptr_sb_u = id!();
    let ptr_sb_f = id!();
    let ptr_fn_u = id!();
    let ptr_fn_s = id!();
    let ptr_fn_f = id!();
    let glsl = id!();
    let main = id!();

    // constants, deduped by value
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
    let c_blocks = uconst(blocks);
    let c_cpb = uconst(cpb);
    let c_dots = uconst(dots_per_word);
    let c_mask = uconst(byte_mask);
    let c_xstride = uconst(xstride);
    let c_bt = uconst(bt);
    let shifts: Vec<u32> = (1..dots_per_word).map(|l| uconst(wbits * l)).collect();
    let c_jf: Vec<u32> = (0..bt).map(|j| uconst(j)).collect();
    let f32_0 = id!();
    const_words.extend([(4 << 16) | OP_CONSTANT, float, f32_0, 0f32.to_bits()]);
    let sint_0 = id!();
    const_words.extend([(4 << 16) | OP_CONSTANT, sint, sint_0, 0]);

    // body ids
    let b_entry = id!();
    let b_then = id!();
    let b_endif = id!();
    let b_ohdr = id!();
    let b_ochk = id!();
    let b_obody = id!();
    let b_ocont = id!();
    let b_omerge = id!();
    let b_ihdr = id!();
    let b_ichk = id!();
    let b_ibody = id!();
    let b_icont = id!();
    let b_imerge = id!();
    let cvar = id!();
    let jvar = id!();
    let acc: Vec<u32> = (0..bt).map(|_| id!()).collect();
    let accq: Vec<u32> = (0..bt).map(|_| id!()).collect();

    let mut w: Vec<u32> = Vec::new();
    let mut body: Vec<u32> = Vec::new();
    // every instruction after the constant block allocates its ids lazily, so `bound` is only known
    // once the body is emitted — build the body first, then the header.
    macro_rules! e {
        ($($x:expr),* $(,)?) => { body.extend([$($x),*]) };
    }

    e!((2 << 16) | OP_LABEL, b_entry);
    e!((4 << 16) | OP_VARIABLE, ptr_fn_u, cvar, SC_FUNCTION);
    e!((4 << 16) | OP_VARIABLE, ptr_fn_u, jvar, SC_FUNCTION);
    for a in &acc {
        e!((4 << 16) | OP_VARIABLE, ptr_fn_f, *a, SC_FUNCTION);
    }
    for a in &accq {
        e!((4 << 16) | OP_VARIABLE, ptr_fn_s, *a, SC_FUNCTION);
    }
    let gidv = id!();
    let r = id!();
    let tile = id!();
    e!((4 << 16) | OP_LOAD, uvec3, gidv, gid);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, r, gidv, 0);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, tile, gidv, 1);
    let in_range = id!();
    e!((5 << 16) | OP_ULESS_THAN, bool_t, in_range, r, c_m);
    e!((3 << 16) | OP_SELECTION_MERGE, b_endif, 0);
    e!((4 << 16) | OP_BRANCH_CONDITIONAL, in_range, b_then, b_endif);
    e!((2 << 16) | OP_LABEL, b_then);
    // per-lane bases: i = tile·bt + j
    let i0 = id!();
    e!((5 << 16) | OP_IMUL, uint, i0, tile, c_bt);
    let (mut xbase, mut sbase, mut obase) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..bt as usize {
        let ij = if j == 0 {
            i0
        } else {
            let t = id!();
            e!((5 << 16) | OP_IADD, uint, t, i0, c_jf[j]);
            t
        };
        let xb = id!();
        e!((5 << 16) | OP_IMUL, uint, xb, ij, c_xstride);
        xbase.push(xb);
        let sb = id!();
        e!((5 << 16) | OP_IMUL, uint, sb, ij, c_blocks);
        sbase.push(sb);
        let ob1 = id!();
        let ob = id!();
        e!((5 << 16) | OP_IMUL, uint, ob1, ij, c_m);
        e!((5 << 16) | OP_IADD, uint, ob, ob1, r);
        obase.push(ob);
        e!((3 << 16) | OP_STORE, acc[j], f32_0);
    }
    e!((3 << 16) | OP_STORE, cvar, c_u0);
    e!((2 << 16) | OP_BRANCH, b_ohdr);
    // ── outer loop over 32-blocks ──
    e!((2 << 16) | OP_LABEL, b_ohdr);
    e!((4 << 16) | OP_LOOP_MERGE, b_omerge, b_ocont, 0);
    e!((2 << 16) | OP_BRANCH, b_ochk);
    e!((2 << 16) | OP_LABEL, b_ochk);
    let ochk_c = id!();
    let ocond = id!();
    e!((4 << 16) | OP_LOAD, uint, ochk_c, cvar);
    e!((5 << 16) | OP_ULESS_THAN, bool_t, ocond, ochk_c, c_blocks);
    e!((4 << 16) | OP_BRANCH_CONDITIONAL, ocond, b_obody, b_omerge);
    e!((2 << 16) | OP_LABEL, b_obody);
    let bb = id!();
    e!((4 << 16) | OP_LOAD, uint, bb, cvar);
    // the block's (d, lo) — ONE f16-packed word, shared by every column in the tile
    let dm1 = id!();
    let dm2 = id!();
    let pd = id!();
    let dmw = id!();
    let dmunp = id!();
    let dv = id!();
    let mv = id!();
    e!((5 << 16) | OP_IMUL, uint, dm1, bb, c_m);
    e!((5 << 16) | OP_IADD, uint, dm2, dm1, r);
    e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, pd, v_dm, c_u0, dm2);
    e!((4 << 16) | OP_LOAD, uint, dmw, pd);
    e!((6 << 16) | OP_EXT_INST, vec2f, dmunp, glsl, 62, dmw); // UnpackHalf2x16
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, float, dv, dmunp, 0);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, float, mv, dmunp, 1);
    for a in &accq {
        e!((3 << 16) | OP_STORE, *a, sint_0);
    }
    e!((3 << 16) | OP_STORE, jvar, c_u0);
    e!((2 << 16) | OP_BRANCH, b_ihdr);
    // ── inner loop: cpb weight words per block; each word feeds bt·dots_per_word DP4As ──
    e!((2 << 16) | OP_LABEL, b_ihdr);
    e!((4 << 16) | OP_LOOP_MERGE, b_imerge, b_icont, 0);
    e!((2 << 16) | OP_BRANCH, b_ichk);
    e!((2 << 16) | OP_LABEL, b_ichk);
    let jj = id!();
    let icond = id!();
    e!((4 << 16) | OP_LOAD, uint, jj, jvar);
    e!((5 << 16) | OP_ULESS_THAN, bool_t, icond, jj, c_cpb);
    e!((4 << 16) | OP_BRANCH_CONDITIONAL, icond, b_ibody, b_imerge);
    e!((2 << 16) | OP_LABEL, b_ibody);
    let cbase = id!();
    let cc = id!();
    e!((5 << 16) | OP_IMUL, uint, cbase, bb, c_cpb);
    e!((5 << 16) | OP_IADD, uint, cc, cbase, jj);
    let widx1 = id!();
    let widx = id!();
    let pw = id!();
    let wword = id!();
    e!((5 << 16) | OP_IMUL, uint, widx1, cc, c_m);
    e!((5 << 16) | OP_IADD, uint, widx, widx1, r);
    e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, pw, v_w, c_u0, widx);
    e!((4 << 16) | OP_LOAD, uint, wword, pw);
    let cdots = if split {
        let t = id!();
        e!((5 << 16) | OP_IMUL, uint, t, cc, c_dots);
        t
    } else {
        cc
    };
    // phase l: the codes at bit offset wbits·l, one per byte. THE WEIGHT WORD IS LOADED ONCE and
    // every phase/column below reuses it — that reuse is the whole kernel.
    let mut xq_cur = cdots;
    for l in 0..dots_per_word {
        let xoff = if l == 0 {
            xq_cur
        } else {
            let t = id!();
            e!((5 << 16) | OP_IADD, uint, t, xq_cur, c_u1);
            xq_cur = t;
            t
        };
        let src = if !split {
            wword
        } else if l == 0 {
            let t = id!();
            e!((5 << 16) | OP_BITWISE_AND, uint, t, wword, c_mask);
            t
        } else {
            let sh = id!();
            let t = id!();
            e!(
                (5 << 16) | OP_SHIFT_RIGHT_LOGICAL,
                uint,
                sh,
                wword,
                shifts[(l - 1) as usize]
            );
            e!((5 << 16) | OP_BITWISE_AND, uint, t, sh, c_mask);
            t
        };
        for j in 0..bt as usize {
            let xi = id!();
            let px = id!();
            let xw = id!();
            let dot = id!();
            let a0 = id!();
            let a1 = id!();
            e!((5 << 16) | OP_IADD, uint, xi, xbase[j], xoff);
            e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, px, v_xq, c_u0, xi);
            e!((4 << 16) | OP_LOAD, uint, xw, px);
            e!((6 << 16) | OP_SUDOT, sint, dot, xw, src, PACKED_VEC_4X8);
            e!((4 << 16) | OP_LOAD, sint, a0, accq[j]);
            e!((5 << 16) | OP_IADD, sint, a1, a0, dot);
            e!((3 << 16) | OP_STORE, accq[j], a1);
        }
    }
    e!((2 << 16) | OP_BRANCH, b_icont);
    e!((2 << 16) | OP_LABEL, b_icont);
    let j2 = id!();
    let jn = id!();
    e!((4 << 16) | OP_LOAD, uint, j2, jvar);
    e!((5 << 16) | OP_IADD, uint, jn, j2, c_u1);
    e!((3 << 16) | OP_STORE, jvar, jn);
    e!((2 << 16) | OP_BRANCH, b_ihdr);
    // ── block epilogue, per column: acc_j += d·float(Σqx_j)·sx[i_j,b] + lo·XS[i_j,b] ──
    e!((2 << 16) | OP_LABEL, b_imerge);
    for j in 0..bt as usize {
        let (pxs, xsv, psx, sxv) = (id!(), id!(), id!(), id!());
        let idx = id!();
        e!((5 << 16) | OP_IADD, uint, idx, sbase[j], bb);
        e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_f, pxs, v_xs, c_u0, idx);
        e!((4 << 16) | OP_LOAD, float, xsv, pxs);
        e!((6 << 16) | OP_ACCESS_CHAIN, ptr_sb_f, psx, v_sx, c_u0, idx);
        e!((4 << 16) | OP_LOAD, float, sxv, psx);
        let (qs, qf, dq, dqs, mxs, sum2, ab0, ab1) =
            (id!(), id!(), id!(), id!(), id!(), id!(), id!(), id!());
        e!((4 << 16) | OP_LOAD, sint, qs, accq[j]);
        e!((4 << 16) | OP_CONVERT_S_TO_F, float, qf, qs);
        e!((5 << 16) | OP_FMUL, float, dq, dv, qf);
        e!((5 << 16) | OP_FMUL, float, dqs, dq, sxv);
        e!((5 << 16) | OP_FMUL, float, mxs, mv, xsv);
        e!((5 << 16) | OP_FADD, float, sum2, dqs, mxs);
        e!((4 << 16) | OP_LOAD, float, ab0, acc[j]);
        e!((5 << 16) | OP_FADD, float, ab1, ab0, sum2);
        e!((3 << 16) | OP_STORE, acc[j], ab1);
    }
    e!((2 << 16) | OP_BRANCH, b_ocont);
    e!((2 << 16) | OP_LABEL, b_ocont);
    let ctr2 = id!();
    let ctrn = id!();
    e!((4 << 16) | OP_LOAD, uint, ctr2, cvar);
    e!((5 << 16) | OP_IADD, uint, ctrn, ctr2, c_u1);
    e!((3 << 16) | OP_STORE, cvar, ctrn);
    e!((2 << 16) | OP_BRANCH, b_ohdr);
    e!((2 << 16) | OP_LABEL, b_omerge);
    for j in 0..bt as usize {
        let accf = id!();
        let pout = id!();
        e!((4 << 16) | OP_LOAD, float, accf, acc[j]);
        e!(
            (6 << 16) | OP_ACCESS_CHAIN,
            ptr_sb_f,
            pout,
            v_out,
            c_u0,
            obase[j]
        );
        e!((3 << 16) | OP_STORE, pout, accf);
    }
    e!((2 << 16) | OP_BRANCH, b_endif);
    e!((2 << 16) | OP_LABEL, b_endif);
    e!((1 << 16) | OP_RETURN);

    let bound = idc + 1;
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
    w.push(((3 + mn.len() as u32 + 1) << 16) | OP_ENTRY_POINT);
    w.extend([MODEL_GL_COMPUTE, main]);
    w.extend(&mn);
    w.push(gid);
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
        gid,
        DEC_BUILT_IN,
        BUILTIN_GLOBAL_INVOCATION_ID,
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
    w.extend([(4 << 16) | OP_VARIABLE, ptr_in_uvec3, gid, SC_INPUT]);
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

/// The batch-tiled integer-dot GEMM as bytes.
pub fn gemv_dp4a_batch_bytes(m: u32, k: u32, local: u32, carrier: Carrier, bt: u32) -> Vec<u8> {
    gemv_dp4a_batch_module(m, k, local, carrier, bt)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// **The DP4A roofline probe — measure the instruction's own ceiling, do not infer it.**
///
/// The published arithmetic for a 5080 (10752 lanes × 2.6 GHz × 4 MACs × 2 ops = 223.6 TOPS)
/// assumes every shader lane retires one `IDP4A` per clock. That assumption is exactly what a
/// consumer part is most likely to break (on Ampere/Ada only HALF the "CUDA cores" carry the INT
/// datapath, which would halve it), so the number is measured here, not quoted: a kernel with NO
/// memory in its loop that does nothing but `OpSUDot`.
///
/// Anti-elision, and it took two tries. The `b` operands come from the buffer (unknown at compile
/// time) and the 2 `a` operands advance every iteration, so no dot is loop-invariant and none can be
/// hoisted. The subtler hole is CSE: the first cut reused 8 `b`s across all `unroll` slots as
/// `b[u % 8]`, so at `unroll = 64` each accumulator saw the SAME expression eight times, the driver
/// folded them, and the loop time did not move between `unroll` 16 and 64 — reporting 861 TOPS,
/// 385% of a ceiling that is supposed to be 223.6. A microbenchmark that beats the speed of light
/// is measuring the compiler, not the machine. So there is now **one distinct `b` per unroll slot**
/// (`nb = unroll` operands loaded from the buffer) and every dot in the body is a distinct
/// expression. The 8 accumulators give the dependent chain 8-way ILP. Body cost is `unroll` dots +
/// 2 advances + ~3 loop ops, so the measured rate is a LOWER bound on the ceiling by
/// `unroll/(unroll+5)`.
///
/// Bindings: 0 = params (`[iters, a0, a1, b0..b_{unroll−1}]`), 1 = out (one u32 per invocation, the
/// xor of the accumulators — the sink that keeps every dot live). Dispatch `(threads/local, 1)`.
pub fn dp4a_roofline_module(local: u32, unroll: u32) -> Vec<u32> {
    assert!(
        unroll % 8 == 0,
        "unroll is a multiple of the 8 accumulators"
    );
    let mut idc = 0u32;
    macro_rules! id {
        () => {{
            idc += 1;
            idc
        }};
    }
    let (void, fnty, uint, sint, bool_t, uvec3) = (id!(), id!(), id!(), id!(), id!(), id!());
    let (ptr_in_uvec3, gid) = (id!(), id!());
    let (rta_u, buf_u, ptr_sb_buf_u, v_in, v_out, ptr_sb_u, ptr_fn_u, ptr_fn_s, main) = (
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
    // params[0] = iters, params[1..3] = the two advancing `a`s, params[3..3+unroll] = the `b`s
    let c_u: Vec<u32> = (0..unroll + 4).map(&mut uconst).collect();
    let sint_0 = id!();
    const_words.extend([(4 << 16) | OP_CONSTANT, sint, sint_0, 0]);

    let (b_entry, b_hdr, b_chk, b_body, b_cont, b_merge) =
        (id!(), id!(), id!(), id!(), id!(), id!());
    let tvar = id!();
    let avar: Vec<u32> = (0..2).map(|_| id!()).collect();
    let acc: Vec<u32> = (0..8).map(|_| id!()).collect();

    let mut body: Vec<u32> = Vec::new();
    macro_rules! e {
        ($($x:expr),* $(,)?) => { body.extend([$($x),*]) };
    }
    e!((2 << 16) | OP_LABEL, b_entry);
    e!((4 << 16) | OP_VARIABLE, ptr_fn_u, tvar, SC_FUNCTION);
    for v in avar.iter().chain(acc.iter()) {
        e!(
            (4 << 16) | OP_VARIABLE,
            if acc.contains(v) { ptr_fn_s } else { ptr_fn_u },
            *v,
            SC_FUNCTION
        );
    }
    let (gidv, lane) = (id!(), id!());
    e!((4 << 16) | OP_LOAD, uvec3, gidv, gid);
    e!((5 << 16) | OP_COMPOSITE_EXTRACT, uint, lane, gidv, 0);
    let ld = |body: &mut Vec<u32>, idc: &mut u32, i: u32| -> u32 {
        *idc += 1;
        let p = *idc;
        *idc += 1;
        let v = *idc;
        body.extend([(6 << 16) | OP_ACCESS_CHAIN, ptr_sb_u, p, v_in, c_u[0], i]);
        body.extend([(4 << 16) | OP_LOAD, uint, v, p]);
        v
    };
    let n = ld(&mut body, &mut idc, c_u[0]);
    // one distinct b per unroll slot — the CSE hole the first cut fell into
    let bv: Vec<u32> = (0..unroll as usize)
        .map(|i| ld(&mut body, &mut idc, c_u[3 + i]))
        .collect();
    for (i, av) in avar.iter().enumerate() {
        let raw = ld(&mut body, &mut idc, c_u[1 + i]);
        let s = id!();
        e!((5 << 16) | OP_IADD, uint, s, raw, lane);
        e!((3 << 16) | OP_STORE, *av, s);
    }
    for a in &acc {
        e!((3 << 16) | OP_STORE, *a, sint_0);
    }
    e!((3 << 16) | OP_STORE, tvar, c_u[0]);
    e!((2 << 16) | OP_BRANCH, b_hdr);
    e!((2 << 16) | OP_LABEL, b_hdr);
    e!((4 << 16) | OP_LOOP_MERGE, b_merge, b_cont, 0);
    e!((2 << 16) | OP_BRANCH, b_chk);
    e!((2 << 16) | OP_LABEL, b_chk);
    let (tv, cond) = (id!(), id!());
    e!((4 << 16) | OP_LOAD, uint, tv, tvar);
    e!((5 << 16) | OP_ULESS_THAN, bool_t, cond, tv, n);
    e!((4 << 16) | OP_BRANCH_CONDITIONAL, cond, b_body, b_merge);
    e!((2 << 16) | OP_LABEL, b_body);
    let acur: Vec<u32> = avar
        .iter()
        .map(|v| {
            let t = id!();
            e!((4 << 16) | OP_LOAD, uint, t, *v);
            t
        })
        .collect();
    for u in 0..unroll as usize {
        let (dot, a0, a1) = (id!(), id!(), id!());
        e!(
            (6 << 16) | OP_SUDOT,
            sint,
            dot,
            acur[u % 2],
            bv[u],
            PACKED_VEC_4X8
        );
        e!((4 << 16) | OP_LOAD, sint, a0, acc[u % 8]);
        e!((5 << 16) | OP_IADD, sint, a1, a0, dot);
        e!((3 << 16) | OP_STORE, acc[u % 8], a1);
    }
    for (i, v) in avar.iter().enumerate() {
        let t = id!();
        e!((5 << 16) | OP_IADD, uint, t, acur[i], c_u[1 + 2 * i]);
        e!((3 << 16) | OP_STORE, *v, t);
    }
    e!((2 << 16) | OP_BRANCH, b_cont);
    e!((2 << 16) | OP_LABEL, b_cont);
    let (t2, tn) = (id!(), id!());
    e!((4 << 16) | OP_LOAD, uint, t2, tvar);
    e!((5 << 16) | OP_IADD, uint, tn, t2, c_u[1]);
    e!((3 << 16) | OP_STORE, tvar, tn);
    e!((2 << 16) | OP_BRANCH, b_hdr);
    e!((2 << 16) | OP_LABEL, b_merge);
    let mut x = {
        let t = id!();
        e!((4 << 16) | OP_LOAD, sint, t, acc[0]);
        t
    };
    for a in &acc[1..] {
        let (t, n2) = (id!(), id!());
        e!((4 << 16) | OP_LOAD, sint, t, *a);
        e!((5 << 16) | OP_BITWISE_XOR, sint, n2, x, t);
        x = n2;
    }
    let (xu, pout) = (id!(), id!());
    e!((4 << 16) | OP_BITCAST, uint, xu, x);
    e!(
        (6 << 16) | OP_ACCESS_CHAIN,
        ptr_sb_u,
        pout,
        v_out,
        c_u[0],
        lane
    );
    e!((3 << 16) | OP_STORE, pout, xu);
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
    w.extend([(3 << 16) | OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450]);
    let mn = lit_string("main");
    w.push(((3 + mn.len() as u32 + 1) << 16) | OP_ENTRY_POINT);
    w.extend([MODEL_GL_COMPUTE, main]);
    w.extend(&mn);
    w.push(gid);
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
        gid,
        DEC_BUILT_IN,
        BUILTIN_GLOBAL_INVOCATION_ID,
    ]);
    w.extend([(4 << 16) | OP_DECORATE, rta_u, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, buf_u, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, buf_u, 0, DEC_OFFSET, 0]);
    for (var, binding) in [(v_in, 0u32), (v_out, 1)] {
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_BINDING, binding]);
    }
    w.extend([(2 << 16) | OP_TYPE_VOID, void]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, fnty, void]);
    w.extend([(4 << 16) | OP_TYPE_INT, uint, 32, 0]);
    w.extend([(4 << 16) | OP_TYPE_INT, sint, 32, 1]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, bool_t]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, uvec3, uint, 3]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_in_uvec3, SC_INPUT, uvec3]);
    w.extend([(4 << 16) | OP_VARIABLE, ptr_in_uvec3, gid, SC_INPUT]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, rta_u, uint]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, buf_u, rta_u]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_buf_u,
        SC_STORAGE_BUFFER,
        buf_u,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_u,
        v_in,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_u,
        v_out,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_u,
        SC_STORAGE_BUFFER,
        uint,
    ]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_fn_u, SC_FUNCTION, uint]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_fn_s, SC_FUNCTION, sint]);
    w.extend(&const_words);
    w.extend([(5 << 16) | OP_FUNCTION, void, main, 0, fnty]);
    w.extend(&body);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The DP4A roofline probe as bytes.
pub fn dp4a_roofline_bytes(local: u32, unroll: u32) -> Vec<u8> {
    dp4a_roofline_module(local, unroll)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}
