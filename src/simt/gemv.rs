//! simt::gemv — the float-carrier GEMM family: the carrier-dial GEMV fold, the two training
//! matvecs (batched GEMV, the dW transpose-contract) and the cooperative-matrix (tensor-core)
//! tiled GEMM with its f16 operand cast. Cut verbatim from `simt.rs`.

use super::*;

/// **The GEMV kernel module — one builder, the carrier is the dial.**
/// `y[r] = Σ_k W[k·M + r] · x[k]` (W column-major), one invocation per row, fp32 accumulate.
///
/// The licenses are theorems (sanguine): `SplitReduce.splitk_exact` (the windowed fold at
/// the emitted bounds) · `stage2_is_the_same_fold` · `colmajor_unit_stride` /
/// `colmajor_reads_once` (at M/pack for packed carriers — one u32 word carries `pack`
/// consecutive rows, shared through L1) · `SealChartFold.fold_in_the_seed_chart` (chart
/// carriers: d·(Σ q·x) + m·(Σ x) per block, the FMA at the block boundary only) ·
/// `MemoryLevelParallelism.latency_hidden` (split-K restores in-flight depth) ·
/// `FusionCost.reduce_fusion_serializes` (the REDUCE stays fanned).
///
/// Bindings at set 0 — direct: 0 = W, 1 = x, 2 = y. Chart: 0 = Wq (u32 words),
/// 1 = x, 2 = y, 3 = DM (INTERLEAVED d,m pairs per (block, row) — one cache line per
/// block-param read, the `GroupProbe.fused_payload_from_one_cell` move: split arrays force
/// a second line), 4 = XS (`Σ_block x`, data). Chart carriers require `k % 32 == 0`,
/// packed carriers `m % pack == 0`.
///
/// Split-K (`segments > 1`): 2D grid, windows in elements (direct) or blocks (chart, so
/// every block lives in one window); partials at `y[s·M + r]`; stage 2 is THIS module at
/// `(m, segments, local, 1, F32)` over the partials with a ones-vector.
pub fn gemv_module(m: u32, k: u32, local: u32, segments: u32, carrier: Carrier) -> Vec<u32> {
    gemv_module_impl(m, k, local, segments, carrier, None)
}

/// **The MoE variant: the SAME fold, with the weight base chosen per dispatch-row by a buffer.**
///
/// A decode command buffer is recorded once and replayed every token, so a data-dependent expert
/// choice cannot be a descriptor offset — it has to be resolved inside the shader. This reuses the
/// split-K machinery wholesale rather than adding a second grid convention: `gid.y` is the SLOT
/// (0..n_used) instead of a k-window, the output already lands at `y[s·M + r]` exactly as partials
/// do, and the only new work is two `IADD`s and one indirect load.
///
/// `expert = SEL[gid.y]`, then the chart is read at `expert·words_per_expert + …` and its scales at
/// `expert·dm_per_expert + …`. Each expert's chart is therefore a CONTIGUOUS block — the same
/// layout `load_model_moe` keeps on the host, and the same one an expert cache moves as a single
/// transfer (`results/moe-route-locality.md`: 1.46 MB at 96% of bulk PCIe bandwidth).
///
/// Bindings extend the chart set by one: 5 = SEL (u32 runtime array, `n_used` entries).
/// `m` is the per-EXPERT row count, not the concatenated total. Dispatch
/// `groups = (ceil(m/local), n_used)`.
pub fn gemv_moe_module(
    m: u32,
    k: u32,
    local: u32,
    carrier: Carrier,
    words_per_expert: u32,
    dm_per_expert: u32,
) -> Vec<u32> {
    gemv_module_impl(
        m,
        k,
        local,
        1,
        carrier,
        Some((words_per_expert, dm_per_expert)),
    )
}

/// The MoE module as bytes.
pub fn gemv_moe_bytes(
    m: u32,
    k: u32,
    local: u32,
    carrier: Carrier,
    words_per_expert: u32,
    dm_per_expert: u32,
) -> Vec<u8> {
    gemv_moe_module(m, k, local, carrier, words_per_expert, dm_per_expert)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

fn gemv_module_impl(
    m: u32,
    k: u32,
    local: u32,
    segments: u32,
    carrier: Carrier,
    moe: Option<(u32, u32)>,
) -> Vec<u32> {
    let pack = carrier.pack();
    assert!(
        moe.is_none() || carrier.is_chart(),
        "the MoE gemv indexes a CHART's words and scales; a direct carrier has no per-expert stride"
    );
    // `gid.y` is either the k-window (split-K) or the expert slot (MoE) — never both, because the
    // output index `s·M + r` means one thing at a time.
    assert!(
        moe.is_none() || segments == 1,
        "split-K and MoE both claim gid.y"
    );
    // the 2D-grid prologue is shared: split-K needs the window, MoE needs the slot
    let grid_y = segments > 1 || moe.is_some();
    let chart = carrier.is_chart();
    assert!(m % pack == 0, "m must be divisible by the carrier pack");
    assert!(!chart || k % 32 == 0, "chart carriers need k % 32 == 0");
    let blocks = k / 32;
    // the outer counter's range: elements (direct) or blocks (chart)
    let span = if chart { blocks } else { k };

    // ── ids, hand-laid; `module_walks` + spirv-val gate the layout ──
    const VOID: u32 = 1;
    const FNTY: u32 = 2;
    const UINT: u32 = 3;
    const FLOAT: u32 = 4;
    const BOOL: u32 = 5;
    const UVEC3: u32 = 6;
    const PTR_IN_UVEC3: u32 = 7;
    const GID: u32 = 8;
    const RTA_F: u32 = 9;
    const BUF_F: u32 = 10;
    const PTR_SB_BUF_F: u32 = 11;
    const RTA_U: u32 = 12;
    const BUF_U: u32 = 13;
    const PTR_SB_BUF_U: u32 = 14;
    const W: u32 = 15;
    const X: u32 = 16;
    const OUT: u32 = 17;
    const DM: u32 = 18;
    const XS: u32 = 19;
    const PTR_SB_F: u32 = 20;
    const PTR_SB_U: u32 = 21;
    const PTR_FN_U: u32 = 22;
    const PTR_FN_F: u32 = 23;
    const VEC2F: u32 = 24; //   F16 unpack target
    const GLSL: u32 = 25; //    F16: OpExtInstImport "GLSL.std.450"
    const C_U0: u32 = 26;
    const C_U1: u32 = 27;
    const C_M: u32 = 28;
    const C_WORDS: u32 = 29; // M / pack — the packed row stride
    const C_SPAN: u32 = 30; //  the outer range: K (direct) or K/32 (chart)
    const C_32: u32 = 31;
    const C_LOGP: u32 = 32; //  log2 pack (wordcol shift)
    const C_LMASK: u32 = 33; // pack − 1 (lane mask)
    const C_LOGW: u32 = 34; // log2(32 / pack) (lane → bit-shift multiplier)
    const C_QMASK: u32 = 35;
    const C_F0: u32 = 36;
    const C_SEG: u32 = 37; //   window length (elements or blocks)
    const MAIN: u32 = 38;
    const B_ENTRY: u32 = 39;
    const CVAR: u32 = 40; //    the outer counter (k or b)
    const JVAR: u32 = 41; //    chart inner counter
    const ACCVAR: u32 = 42;
    const ACCQVAR: u32 = 43; // chart code fold
    const GIDV: u32 = 44;
    const R: u32 = 45;
    const WORDCOL: u32 = 46;
    const LANE: u32 = 47;
    const SHAMT: u32 = 48;
    const IN_RANGE: u32 = 49;
    const S: u32 = 50;
    const W0: u32 = 51;
    const WEND_RAW: u32 = 52;
    const OVER: u32 = 53;
    const WEND: u32 = 54;
    const OI1: u32 = 55;
    const OI2: u32 = 56;
    const B_THEN: u32 = 57;
    const B_ENDIF: u32 = 58;
    const B_OHDR: u32 = 59;
    const B_OCHK: u32 = 60;
    const B_OBODY: u32 = 61;
    const B_OCONT: u32 = 62;
    const B_OMERGE: u32 = 63;
    const OCHK_C: u32 = 64;
    const OCOND: u32 = 65;
    const BB: u32 = 66; //      outer counter in the body
                        // chart block prelude (DM fused + XS)
    const DM1: u32 = 67;
    const DM2: u32 = 68;
    const DMIDX: u32 = 69;
    const DMIDX1: u32 = 70;
    const PD: u32 = 71;
    const DV: u32 = 72;
    const PM: u32 = 73;
    const MV: u32 = 74;
    const PXS: u32 = 75;
    const XSV: u32 = 76;
    const B_IHDR: u32 = 77;
    const B_ICHK: u32 = 78;
    const B_IBODY: u32 = 79;
    const B_ICONT: u32 = 80;
    const B_IMERGE: u32 = 81;
    const JJ: u32 = 82;
    const ICOND: u32 = 83;
    const KB: u32 = 84;
    const KKI: u32 = 85; //     chart: element index b·32 + j
                         // the element decode fragment (one site per module)
    const WIDX1: u32 = 86;
    const WIDX: u32 = 87;
    const PW: u32 = 88;
    const WRAW: u32 = 89; //    f32 value (F32) or u32 word (packed)
    const UNPACKED: u32 = 90; // F16: vec2
    const SHIFTED: u32 = 91; // Q: word >> shamt
    const Q: u32 = 92; //       Q: masked code
    const WVAL: u32 = 93; //    the decoded f32 weight
    const PX: u32 = 94;
    const XV: u32 = 95;
    const PROD: u32 = 96;
    const A0: u32 = 97;
    const A1: u32 = 98;
    const CTR2: u32 = 99;
    const CTRN: u32 = 100;
    // chart inner counter inc + block epilogue
    const J2: u32 = 101;
    const JN: u32 = 102;
    const ACCQF: u32 = 103;
    const DQ: u32 = 104;
    const MXS: u32 = 105;
    const SUM2: u32 = 106;
    const AB0: u32 = 107;
    const AB1: u32 = 108;
    const ACCF: u32 = 109;
    const POUT: u32 = 110;
    // MoE-only ids (unused, and un-emitted, when `moe` is None)
    const SEL: u32 = 111; //     binding 5: the selected expert per slot
    const PSEL: u32 = 112;
    const EIDX: u32 = 113; //    SEL[gid.y]
    const WBASE: u32 = 114; //   expert · words_per_expert
    const DMBASE: u32 = 115; //  expert · dm_per_expert
    const C_WPE: u32 = 116;
    const C_DMPE: u32 = 117;
    const WIDX2: u32 = 118; //   chart index before the expert base is added
    const DMIDX2: u32 = 119;
    const BOUND: u32 = 120;
    // The id bound is a HEADER word, so widening it unconditionally would change the bytes of
    // every dense module ever emitted. The MoE ids only exist when the MoE arm does, and the
    // dense modules stay byte-identical — which is the whole point of sharing this builder.
    const BOUND_DENSE: u32 = 111;

    let mut w = vec![
        SPV_MAGIC,
        SPV_VERSION_1_3,
        0,
        if moe.is_some() { BOUND } else { BOUND_DENSE },
        0,
    ];
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
    if carrier == Carrier::F16 {
        let glsl = lit_string("GLSL.std.450");
        w.push(((2 + glsl.len() as u32) << 16) | OP_EXT_INST_IMPORT);
        w.push(GLSL);
        w.extend(&glsl);
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
    w.extend([(4 << 16) | OP_DECORATE, RTA_F, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, BUF_F, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, BUF_F, 0, DEC_OFFSET, 0]);
    if pack > 1 {
        w.extend([(4 << 16) | OP_DECORATE, RTA_U, DEC_ARRAY_STRIDE, 4]);
        w.extend([(3 << 16) | OP_DECORATE, BUF_U, DEC_BLOCK]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, BUF_U, 0, DEC_OFFSET, 0]);
    }
    let bindings: &[(u32, u32)] = if moe.is_some() {
        &[(W, 0), (X, 1), (OUT, 2), (DM, 3), (XS, 4), (SEL, 5)]
    } else if chart {
        &[(W, 0), (X, 1), (OUT, 2), (DM, 3), (XS, 4)]
    } else {
        &[(W, 0), (X, 1), (OUT, 2)]
    };
    for &(var, binding) in bindings {
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_BINDING, binding]);
    }
    // types / constants / variables
    w.extend([(2 << 16) | OP_TYPE_VOID, VOID]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, FNTY, VOID]);
    w.extend([(4 << 16) | OP_TYPE_INT, UINT, 32, 0]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, FLOAT, 32]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, BOOL]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, UVEC3, UINT, 3]);
    if carrier == Carrier::F16 {
        w.extend([(4 << 16) | OP_TYPE_VECTOR, VEC2F, FLOAT, 2]);
    }
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_IN_UVEC3, SC_INPUT, UVEC3]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_IN_UVEC3, GID, SC_INPUT]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, RTA_F, FLOAT]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, BUF_F, RTA_F]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_BUF_F,
        SC_STORAGE_BUFFER,
        BUF_F,
    ]);
    if pack > 1 {
        w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, RTA_U, UINT]);
        w.extend([(3 << 16) | OP_TYPE_STRUCT, BUF_U, RTA_U]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            PTR_SB_BUF_U,
            SC_STORAGE_BUFFER,
            BUF_U,
        ]);
        w.extend([
            (4 << 16) | OP_TYPE_POINTER,
            PTR_SB_U,
            SC_STORAGE_BUFFER,
            UINT,
        ]);
    }
    let w_buf_ptr = if pack > 1 { PTR_SB_BUF_U } else { PTR_SB_BUF_F };
    w.extend([(4 << 16) | OP_VARIABLE, w_buf_ptr, W, SC_STORAGE_BUFFER]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, X, SC_STORAGE_BUFFER]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        PTR_SB_BUF_F,
        OUT,
        SC_STORAGE_BUFFER,
    ]);
    if chart {
        w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, DM, SC_STORAGE_BUFFER]);
        w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, XS, SC_STORAGE_BUFFER]);
    }
    if moe.is_some() {
        w.extend([
            (4 << 16) | OP_VARIABLE,
            PTR_SB_BUF_U,
            SEL,
            SC_STORAGE_BUFFER,
        ]);
    }
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_F,
        SC_STORAGE_BUFFER,
        FLOAT,
    ]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_U, SC_FUNCTION, UINT]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_F, SC_FUNCTION, FLOAT]);
    let mut consts: Vec<(u32, u32)> = vec![(C_U0, 0), (C_U1, 1), (C_M, m), (C_SPAN, span)];
    if pack > 1 {
        consts.extend([
            (C_WORDS, m / pack),
            (C_LOGP, pack.trailing_zeros()),
            (C_LMASK, pack - 1),
            (C_LOGW, (32 / pack).trailing_zeros()),
        ]);
    }
    if chart {
        consts.extend([(C_32, 32u32), (C_QMASK, carrier.qmask())]);
    }
    if segments > 1 {
        consts.push((C_SEG, span.div_ceil(segments)));
    }
    if let Some((wpe, dmpe)) = moe {
        consts.extend([(C_WPE, wpe), (C_DMPE, dmpe)]);
    }
    for (id, v) in consts {
        w.extend([(4 << 16) | OP_CONSTANT, UINT, id, v]);
    }
    w.extend([(4 << 16) | OP_CONSTANT, FLOAT, C_F0, 0f32.to_bits()]);
    // main / entry
    w.extend([(5 << 16) | OP_FUNCTION, VOID, MAIN, 0, FNTY]);
    w.extend([(2 << 16) | OP_LABEL, B_ENTRY]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_U, CVAR, SC_FUNCTION]);
    if chart {
        w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_U, JVAR, SC_FUNCTION]);
        w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_F, ACCQVAR, SC_FUNCTION]);
    }
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_F, ACCVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_LOAD, UVEC3, GIDV, GID]);
    w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, UINT, R, GIDV, 0]);
    if pack > 1 {
        // the pack algebra: r = pack·(r >> log p) + (r & (pack−1)); bit shift = lane·(32/pack)
        w.extend([(5 << 16) | OP_SHIFT_RIGHT_LOGICAL, UINT, WORDCOL, R, C_LOGP]);
        w.extend([(5 << 16) | OP_BITWISE_AND, UINT, LANE, R, C_LMASK]);
        w.extend([(5 << 16) | OP_SHIFT_LEFT_LOGICAL, UINT, SHAMT, LANE, C_LOGW]);
    }
    if grid_y {
        w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, UINT, S, GIDV, 1]);
        if segments > 1 {
            w.extend([(5 << 16) | OP_IMUL, UINT, W0, S, C_SEG]);
            w.extend([(5 << 16) | OP_IADD, UINT, WEND_RAW, W0, C_SEG]);
            w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, OVER, C_SPAN, WEND_RAW]);
            w.extend([(6 << 16) | OP_SELECT, UINT, WEND, OVER, C_SPAN, WEND_RAW]);
        }
        if moe.is_some() {
            // expert = SEL[slot]; its chart and scales start at a whole-expert stride
            w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_U, PSEL, SEL, C_U0, S]);
            w.extend([(4 << 16) | OP_LOAD, UINT, EIDX, PSEL]);
            w.extend([(5 << 16) | OP_IMUL, UINT, WBASE, EIDX, C_WPE]);
            w.extend([(5 << 16) | OP_IMUL, UINT, DMBASE, EIDX, C_DMPE]);
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
    w.extend([(2 << 16) | OP_BRANCH, B_OHDR]);
    // outer loop (over elements for direct, blocks for chart)
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
        if segments > 1 { WEND } else { C_SPAN },
    ]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, OCOND, B_OBODY, B_OMERGE]);
    w.extend([(2 << 16) | OP_LABEL, B_OBODY]);
    w.extend([(4 << 16) | OP_LOAD, UINT, BB, CVAR]);

    // the element decode fragment: weight at element index `kid` → f32 in WVAL
    let decode = |w: &mut Vec<u32>, kid: u32| {
        if pack > 1 {
            w.extend([(5 << 16) | OP_IMUL, UINT, WIDX1, kid, C_WORDS]);
            if moe.is_some() {
                w.extend([(5 << 16) | OP_IADD, UINT, WIDX2, WIDX1, WORDCOL]);
                w.extend([(5 << 16) | OP_IADD, UINT, WIDX, WIDX2, WBASE]);
            } else {
                w.extend([(5 << 16) | OP_IADD, UINT, WIDX, WIDX1, WORDCOL]);
            }
            w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_U, PW, W, C_U0, WIDX]);
            w.extend([(4 << 16) | OP_LOAD, UINT, WRAW, PW]);
        } else {
            w.extend([(5 << 16) | OP_IMUL, UINT, WIDX1, kid, C_M]);
            w.extend([(5 << 16) | OP_IADD, UINT, WIDX, WIDX1, R]);
            w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PW, W, C_U0, WIDX]);
            w.extend([(4 << 16) | OP_LOAD, FLOAT, WVAL, PW]);
        }
        match carrier {
            Carrier::F32 => {}
            Carrier::F16 => {
                // GLSL.std.450 UnpackHalf2x16 = 62 (extinst grammar); dynamic lane extract
                w.extend([
                    (6 << 16) | OP_EXT_INST,
                    VEC2F,
                    UNPACKED,
                    GLSL,
                    GLSL_UNPACK_HALF2X16,
                    WRAW,
                ]);
                w.extend([
                    (5 << 16) | OP_VECTOR_EXTRACT_DYNAMIC,
                    FLOAT,
                    WVAL,
                    UNPACKED,
                    LANE,
                ]);
            }
            // `Carrier::Q3` is a TWO-PLANE chart (`q3_planes_gpu_chart`): the high bit lives in a
            // second plane at a compile-time offset, and this single-word shift-and-mask decode
            // cannot see it. Reading Q3 here would silently serve the low two bits of every
            // weight — a quarter of the model's magnitude range, and finite logits. Refused.
            Carrier::Q3 => panic!(
                "gemv_module has no two-plane decode: Carrier::Q3 is served by the DP4A path \
(`gemv_dp4a_*`), which folds plane B per phase. Reading it here would use the low 2 bits only."
            ),
            // `Carrier::Q5` is the same shape of two-plane chart as `Carrier::Q3`: plane A is Q4's
            // nibble plane and plane B is the permuted fifth bit, so the single-word decode below
            // sees only the low four bits. Served by the DP4A path. Refused.
            Carrier::Q5 => panic!(
                "gemv_module has no two-plane decode: Carrier::Q5 is served by the DP4A path \
(`gemv_dp4a_*`), which folds plane B per phase. Reading it here would use the low 4 bits only."
            ),
            Carrier::Q8 | Carrier::Q4 | Carrier::Q2 => {
                w.extend([
                    (5 << 16) | OP_SHIFT_RIGHT_LOGICAL,
                    UINT,
                    SHIFTED,
                    WRAW,
                    SHAMT,
                ]);
                w.extend([(5 << 16) | OP_BITWISE_AND, UINT, Q, SHIFTED, C_QMASK]);
                w.extend([(4 << 16) | OP_CONVERT_U_TO_F, FLOAT, WVAL, Q]);
            }
        }
    };
    // multiply by x[kid] and fold into `acc_var`
    let fma = |w: &mut Vec<u32>, kid: u32, acc_var: u32| {
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PX, X, C_U0, kid]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, XV, PX]);
        w.extend([(5 << 16) | OP_FMUL, FLOAT, PROD, WVAL, XV]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, A0, acc_var]);
        w.extend([(5 << 16) | OP_FADD, FLOAT, A1, A0, PROD]);
        w.extend([(3 << 16) | OP_STORE, acc_var, A1]);
    };

    if chart {
        // block prelude: the FUSED (d, m) pair — adjacent slots, one line
        // (`GroupProbe.fused_payload_from_one_cell`) — and the block's Σx
        w.extend([(5 << 16) | OP_IMUL, UINT, DM1, BB, C_M]);
        w.extend([(5 << 16) | OP_IADD, UINT, DM2, DM1, R]);
        if moe.is_some() {
            w.extend([(5 << 16) | OP_IADD, UINT, DMIDX2, DM2, DM2]); // 2·(b·M + r)
            w.extend([(5 << 16) | OP_IADD, UINT, DMIDX, DMIDX2, DMBASE]);
        } else {
            w.extend([(5 << 16) | OP_IADD, UINT, DMIDX, DM2, DM2]); // 2·(b·M + r)
        }
        w.extend([(5 << 16) | OP_IADD, UINT, DMIDX1, DMIDX, C_U1]);
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PD, DM, C_U0, DMIDX]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, DV, PD]);
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PM, DM, C_U0, DMIDX1]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, MV, PM]);
        w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PXS, XS, C_U0, BB]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, XSV, PXS]);
        w.extend([(3 << 16) | OP_STORE, ACCQVAR, C_F0]);
        w.extend([(3 << 16) | OP_STORE, JVAR, C_U0]);
        w.extend([(2 << 16) | OP_BRANCH, B_IHDR]);
        // inner loop: the code fold
        w.extend([(2 << 16) | OP_LABEL, B_IHDR]);
        w.extend([(4 << 16) | OP_LOOP_MERGE, B_IMERGE, B_ICONT, 0]);
        w.extend([(2 << 16) | OP_BRANCH, B_ICHK]);
        w.extend([(2 << 16) | OP_LABEL, B_ICHK]);
        w.extend([(4 << 16) | OP_LOAD, UINT, JJ, JVAR]);
        w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, ICOND, JJ, C_32]);
        w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, ICOND, B_IBODY, B_IMERGE]);
        w.extend([(2 << 16) | OP_LABEL, B_IBODY]);
        w.extend([(5 << 16) | OP_IMUL, UINT, KB, BB, C_32]);
        w.extend([(5 << 16) | OP_IADD, UINT, KKI, KB, JJ]);
        decode(&mut w, KKI);
        fma(&mut w, KKI, ACCQVAR);
        w.extend([(2 << 16) | OP_BRANCH, B_ICONT]);
        w.extend([(2 << 16) | OP_LABEL, B_ICONT]);
        w.extend([(4 << 16) | OP_LOAD, UINT, J2, JVAR]);
        w.extend([(5 << 16) | OP_IADD, UINT, JN, J2, C_U1]);
        w.extend([(3 << 16) | OP_STORE, JVAR, JN]);
        w.extend([(2 << 16) | OP_BRANCH, B_IHDR]);
        // block epilogue: acc += d·(Σ q·x) + m·XS_b (`fold_in_the_seed_chart`)
        w.extend([(2 << 16) | OP_LABEL, B_IMERGE]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, ACCQF, ACCQVAR]);
        w.extend([(5 << 16) | OP_FMUL, FLOAT, DQ, DV, ACCQF]);
        w.extend([(5 << 16) | OP_FMUL, FLOAT, MXS, MV, XSV]);
        w.extend([(5 << 16) | OP_FADD, FLOAT, SUM2, DQ, MXS]);
        w.extend([(4 << 16) | OP_LOAD, FLOAT, AB0, ACCVAR]);
        w.extend([(5 << 16) | OP_FADD, FLOAT, AB1, AB0, SUM2]);
        w.extend([(3 << 16) | OP_STORE, ACCVAR, AB1]);
        w.extend([(2 << 16) | OP_BRANCH, B_OCONT]);
    } else {
        decode(&mut w, BB);
        fma(&mut w, BB, ACCVAR);
        w.extend([(2 << 16) | OP_BRANCH, B_OCONT]);
    }
    w.extend([(2 << 16) | OP_LABEL, B_OCONT]);
    w.extend([(4 << 16) | OP_LOAD, UINT, CTR2, CVAR]);
    w.extend([(5 << 16) | OP_IADD, UINT, CTRN, CTR2, C_U1]);
    w.extend([(3 << 16) | OP_STORE, CVAR, CTRN]);
    w.extend([(2 << 16) | OP_BRANCH, B_OHDR]);
    w.extend([(2 << 16) | OP_LABEL, B_OMERGE]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, ACCF, ACCVAR]);
    w.extend([
        (6 << 16) | OP_ACCESS_CHAIN,
        PTR_SB_F,
        POUT,
        OUT,
        C_U0,
        if grid_y { OI2 } else { R },
    ]);
    w.extend([(3 << 16) | OP_STORE, POUT, ACCF]);
    w.extend([(2 << 16) | OP_BRANCH, B_ENDIF]);
    w.extend([(2 << 16) | OP_LABEL, B_ENDIF]);
    w.extend([(1 << 16) | OP_RETURN]);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The GEMV module as bytes.
pub fn gemv_bytes(m: u32, k: u32, local: u32, segments: u32, carrier: Carrier) -> Vec<u8> {
    gemv_module(m, k, local, segments, carrier)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

// ── The training matvecs: the forward/backward GEMMs as one batched GEMV + one dW reduction. ────
// Training is three matmuls on the metal (nullseed `gemm_nt`/`gemm_nn`/`gemm_tn_add`). Two of them
// are the SAME shape as inference's [`gemv_module`] with a batch axis bolted on — the batched GEMV
// below — and the third is the transpose-contract dW, its own one-invocation-per-output fold. Both
// are f32-only (the chart/carrier dial is an inference concern; the trainer holds master weights in
// f32). Same hand-laid-SSA discipline, same spirv-val gate. New words beyond [`gemv_module`]: none —
// the batch axis is a second `OpCompositeExtract` of the GlobalInvocationId vector (component 1).

/// **Batched GEMV — the training forward/backward matvec, f32.**
/// `Y[b, r] = Σ_{kk} W[kk·M + r] · X[b·K + kk]` (W column-major, stride M, fp32 accumulate),
/// one invocation per `(b, r)`: `gl_GlobalInvocationId.x = r` (guarded `< M`),
/// `gl_GlobalInvocationId.y = b` (the batch, exact from `groups.y = BATCH` with local size
/// `(local, 1, 1)`). Dispatch `groups = (ceil(M/local), BATCH)`. Bindings at set 0:
/// 0 = W, 1 = X, 2 = Y (all f32 runtime arrays).
///
/// **The kernel does NOT transpose** — the caller lays W out column-major. This one module serves
/// both non-dW training GEMMs (nullseed):
/// * forward `gemm_nt(a[m,k], w[n,k], out[m,n])` (`out = a·wᵀ`): pass W = w transposed to
///   column-major `[k,n]` (element `(r=n_idx, kk)` at `kk·n + r`), X = a, Y = out,
///   BATCH = m, M = n, K = k.
/// * backward `gemm_nn(a[m,l], b[l,n], out[m,n])` (`out = a·b`): b is already row-major `[l,n]`,
///   which IS column-major with stride n (element `(r=n_idx, kk)` at `kk·n + r`), so pass W = b
///   directly, X = a, Y = out, BATCH = m, M = n, K = l.
///
/// Inner reduction is byte-identical to [`gemv_module`]'s F32 path (`W[kk·M+r]·x[kk]`, fp32
/// accumulate); the batch just offsets X by `b·K` and Y by `b·M`. SPIR-V ops as gemv_module
/// (spirv.core.grammar.json): OpTypeRuntimeArray 29, OpAccessChain 65, OpCompositeExtract 81,
/// OpFMul 133, OpFAdd 129, OpULessThan 176, OpLoopMerge 246, OpSelectionMerge 247, OpBranch 249,
/// OpBranchConditional 250; BuiltIn.GlobalInvocationId 28.
pub fn gemv_batch_module(m: u32, k: u32, local: u32) -> Vec<u32> {
    const VOID: u32 = 1;
    const FNTY: u32 = 2;
    const UINT: u32 = 3;
    const FLOAT: u32 = 4;
    const BOOL: u32 = 5;
    const UVEC3: u32 = 6;
    const PTR_IN_UVEC3: u32 = 7;
    const GID: u32 = 8;
    const RTA_F: u32 = 9;
    const BUF_F: u32 = 10;
    const PTR_SB_BUF_F: u32 = 11;
    const PTR_SB_F: u32 = 12;
    const PTR_FN_U: u32 = 13;
    const PTR_FN_F: u32 = 14;
    const W: u32 = 15;
    const X: u32 = 16;
    const Y: u32 = 17;
    const C_U0: u32 = 18;
    const C_U1: u32 = 19;
    const C_M: u32 = 20;
    const C_K: u32 = 21;
    const C_F0: u32 = 22;
    const MAIN: u32 = 23;
    const B_ENTRY: u32 = 24;
    const CVAR: u32 = 25; //   the inner counter kk
    const ACCVAR: u32 = 26; // the fp32 accumulator
    const GIDV: u32 = 27;
    const R: u32 = 28; //      GID.x — the output row
    const BID: u32 = 29; //    GID.y — the batch index
    const IN_RANGE: u32 = 30;
    const B_THEN: u32 = 31;
    const B_ENDIF: u32 = 32;
    const XBASE: u32 = 33; //  b·K
    const YBASE1: u32 = 34; // b·M
    const YBASE: u32 = 35; //  b·M + r
    const B_OHDR: u32 = 36;
    const B_OCHK: u32 = 37;
    const B_OBODY: u32 = 38;
    const B_OCONT: u32 = 39;
    const B_OMERGE: u32 = 40;
    const OCHK_C: u32 = 41;
    const OCOND: u32 = 42;
    const KK: u32 = 43;
    const WIDX1: u32 = 44;
    const WIDX: u32 = 45;
    const PW: u32 = 46;
    const WVAL: u32 = 47;
    const XIDX: u32 = 48;
    const PX: u32 = 49;
    const XV: u32 = 50;
    const PROD: u32 = 51;
    const A0: u32 = 52;
    const A1: u32 = 53;
    const CTR2: u32 = 54;
    const CTRN: u32 = 55;
    const ACCF: u32 = 56;
    const POUT: u32 = 57;
    const BOUND: u32 = 58;

    let mut w = vec![SPV_MAGIC, SPV_VERSION_1_3, 0, BOUND, 0];
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
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
    w.extend([(4 << 16) | OP_DECORATE, RTA_F, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, BUF_F, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, BUF_F, 0, DEC_OFFSET, 0]);
    for (var, binding) in [(W, 0u32), (X, 1), (Y, 2)] {
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_BINDING, binding]);
    }
    // types / variables / constants
    w.extend([(2 << 16) | OP_TYPE_VOID, VOID]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, FNTY, VOID]);
    w.extend([(4 << 16) | OP_TYPE_INT, UINT, 32, 0]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, FLOAT, 32]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, BOOL]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, UVEC3, UINT, 3]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_IN_UVEC3, SC_INPUT, UVEC3]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_IN_UVEC3, GID, SC_INPUT]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, RTA_F, FLOAT]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, BUF_F, RTA_F]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_BUF_F,
        SC_STORAGE_BUFFER,
        BUF_F,
    ]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, W, SC_STORAGE_BUFFER]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, X, SC_STORAGE_BUFFER]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, Y, SC_STORAGE_BUFFER]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_F,
        SC_STORAGE_BUFFER,
        FLOAT,
    ]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_U, SC_FUNCTION, UINT]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_F, SC_FUNCTION, FLOAT]);
    for (id, v) in [(C_U0, 0u32), (C_U1, 1), (C_M, m), (C_K, k)] {
        w.extend([(4 << 16) | OP_CONSTANT, UINT, id, v]);
    }
    w.extend([(4 << 16) | OP_CONSTANT, FLOAT, C_F0, 0f32.to_bits()]);
    // main
    w.extend([(5 << 16) | OP_FUNCTION, VOID, MAIN, 0, FNTY]);
    w.extend([(2 << 16) | OP_LABEL, B_ENTRY]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_U, CVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_F, ACCVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_LOAD, UVEC3, GIDV, GID]);
    w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, UINT, R, GIDV, 0]);
    w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, UINT, BID, GIDV, 1]);
    w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, IN_RANGE, R, C_M]);
    w.extend([(3 << 16) | OP_SELECTION_MERGE, B_ENDIF, 0]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, IN_RANGE, B_THEN, B_ENDIF]);
    w.extend([(2 << 16) | OP_LABEL, B_THEN]);
    w.extend([(5 << 16) | OP_IMUL, UINT, XBASE, BID, C_K]);
    w.extend([(5 << 16) | OP_IMUL, UINT, YBASE1, BID, C_M]);
    w.extend([(5 << 16) | OP_IADD, UINT, YBASE, YBASE1, R]);
    w.extend([(3 << 16) | OP_STORE, CVAR, C_U0]);
    w.extend([(3 << 16) | OP_STORE, ACCVAR, C_F0]);
    w.extend([(2 << 16) | OP_BRANCH, B_OHDR]);
    w.extend([(2 << 16) | OP_LABEL, B_OHDR]);
    w.extend([(4 << 16) | OP_LOOP_MERGE, B_OMERGE, B_OCONT, 0]);
    w.extend([(2 << 16) | OP_BRANCH, B_OCHK]);
    w.extend([(2 << 16) | OP_LABEL, B_OCHK]);
    w.extend([(4 << 16) | OP_LOAD, UINT, OCHK_C, CVAR]);
    w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, OCOND, OCHK_C, C_K]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, OCOND, B_OBODY, B_OMERGE]);
    w.extend([(2 << 16) | OP_LABEL, B_OBODY]);
    w.extend([(4 << 16) | OP_LOAD, UINT, KK, CVAR]);
    // W[kk·M + r]
    w.extend([(5 << 16) | OP_IMUL, UINT, WIDX1, KK, C_M]);
    w.extend([(5 << 16) | OP_IADD, UINT, WIDX, WIDX1, R]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PW, W, C_U0, WIDX]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, WVAL, PW]);
    // X[b·K + kk]
    w.extend([(5 << 16) | OP_IADD, UINT, XIDX, XBASE, KK]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PX, X, C_U0, XIDX]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, XV, PX]);
    w.extend([(5 << 16) | OP_FMUL, FLOAT, PROD, WVAL, XV]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, A0, ACCVAR]);
    w.extend([(5 << 16) | OP_FADD, FLOAT, A1, A0, PROD]);
    w.extend([(3 << 16) | OP_STORE, ACCVAR, A1]);
    w.extend([(2 << 16) | OP_BRANCH, B_OCONT]);
    w.extend([(2 << 16) | OP_LABEL, B_OCONT]);
    w.extend([(4 << 16) | OP_LOAD, UINT, CTR2, CVAR]);
    w.extend([(5 << 16) | OP_IADD, UINT, CTRN, CTR2, C_U1]);
    w.extend([(3 << 16) | OP_STORE, CVAR, CTRN]);
    w.extend([(2 << 16) | OP_BRANCH, B_OHDR]);
    w.extend([(2 << 16) | OP_LABEL, B_OMERGE]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, ACCF, ACCVAR]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, POUT, Y, C_U0, YBASE]);
    w.extend([(3 << 16) | OP_STORE, POUT, ACCF]);
    w.extend([(2 << 16) | OP_BRANCH, B_ENDIF]);
    w.extend([(2 << 16) | OP_LABEL, B_ENDIF]);
    w.extend([(1 << 16) | OP_RETURN]);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The batched GEMV module as bytes.
pub fn gemv_batch_bytes(m: u32, k: u32, local: u32) -> Vec<u8> {
    gemv_batch_module(m, k, local)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// **The dW reduction — the training weight-gradient contract, f32.**
/// `OUT[i·K + j] += Σ_{b} A[b·N + i] · B[b·K + j]` for `i ∈ 0..N`, `j ∈ 0..K`, contracting the
/// batch `b ∈ 0..M`. This is nullseed `gemm_tn_add(a[m,n], b[m,k], out[n,k])` — `out += aᵀ·b` —
/// exactly (A = a, B = b, OUT = out, M = m, N = n, K = k).
///
/// One invocation per output element `(i, j)`: `gl_GlobalInvocationId.x = j` (guarded `< K`),
/// `gl_GlobalInvocationId.y = i` (exact from `groups.y = N` with local size `(local, 1, 1)`).
/// Dispatch `groups = (ceil(K/local), N)`. The invocation folds `b` over `0..M` in fp32 then
/// **read-modify-writes** OUT (`+=`): since each `(i, j)` is owned by exactly one invocation the
/// accumulate is race-free WITHOUT atomics — OUT must be pre-initialized by the caller (the
/// gradient it adds onto). Bindings at set 0: 0 = A, 1 = B, 2 = OUT (all f32 runtime arrays).
///
/// SPIR-V ops as [`gemv_batch_module`] (spirv.core.grammar.json); the `+=` is an extra
/// OpLoad(OUT)/OpFAdd/OpStore(OUT) at the tail. spirv-val (`--target-env vulkan1.1`) gates it.
pub fn gemm_dw_module(m: u32, n: u32, k: u32, local: u32) -> Vec<u32> {
    const VOID: u32 = 1;
    const FNTY: u32 = 2;
    const UINT: u32 = 3;
    const FLOAT: u32 = 4;
    const BOOL: u32 = 5;
    const UVEC3: u32 = 6;
    const PTR_IN_UVEC3: u32 = 7;
    const GID: u32 = 8;
    const RTA_F: u32 = 9;
    const BUF_F: u32 = 10;
    const PTR_SB_BUF_F: u32 = 11;
    const PTR_SB_F: u32 = 12;
    const PTR_FN_U: u32 = 13;
    const PTR_FN_F: u32 = 14;
    const A: u32 = 15;
    const B: u32 = 16;
    const OUT: u32 = 17;
    const C_U0: u32 = 18;
    const C_U1: u32 = 19;
    const C_N: u32 = 20;
    const C_K: u32 = 21;
    const C_M: u32 = 22;
    const C_F0: u32 = 23;
    const MAIN: u32 = 24;
    const B_ENTRY: u32 = 25;
    const BVAR: u32 = 26; //   the contraction counter b
    const ACCVAR: u32 = 27; // the fp32 accumulator
    const GIDV: u32 = 28;
    const J: u32 = 29; //      GID.x — output column
    const I: u32 = 30; //      GID.y — output row
    const IN_RANGE: u32 = 31;
    const B_THEN: u32 = 32;
    const B_ENDIF: u32 = 33;
    const OUTIDX1: u32 = 34; // i·K
    const OUTIDX: u32 = 35; //  i·K + j
    const B_OHDR: u32 = 36;
    const B_OCHK: u32 = 37;
    const B_OBODY: u32 = 38;
    const B_OCONT: u32 = 39;
    const B_OMERGE: u32 = 40;
    const OCHK_C: u32 = 41;
    const OCOND: u32 = 42;
    const BB: u32 = 43; //     loaded contraction index
    const AIDX1: u32 = 44; //  b·N
    const AIDX: u32 = 45; //   b·N + i
    const PA: u32 = 46;
    const AV: u32 = 47;
    const BIDX1: u32 = 48; //  b·K
    const BIDX: u32 = 49; //   b·K + j
    const PB: u32 = 50;
    const BV: u32 = 51;
    const PROD: u32 = 52;
    const A0: u32 = 53;
    const A1: u32 = 54;
    const CTR2: u32 = 55;
    const CTRN: u32 = 56;
    const ACCF: u32 = 57;
    const POUT: u32 = 58;
    const OLD: u32 = 59;
    const NEW: u32 = 60;
    const BOUND: u32 = 61;

    let mut w = vec![SPV_MAGIC, SPV_VERSION_1_3, 0, BOUND, 0];
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
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
    w.extend([(4 << 16) | OP_DECORATE, RTA_F, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, BUF_F, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, BUF_F, 0, DEC_OFFSET, 0]);
    for (var, binding) in [(A, 0u32), (B, 1), (OUT, 2)] {
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_BINDING, binding]);
    }
    // types / variables / constants
    w.extend([(2 << 16) | OP_TYPE_VOID, VOID]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, FNTY, VOID]);
    w.extend([(4 << 16) | OP_TYPE_INT, UINT, 32, 0]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, FLOAT, 32]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, BOOL]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, UVEC3, UINT, 3]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_IN_UVEC3, SC_INPUT, UVEC3]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_IN_UVEC3, GID, SC_INPUT]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, RTA_F, FLOAT]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, BUF_F, RTA_F]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_BUF_F,
        SC_STORAGE_BUFFER,
        BUF_F,
    ]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, A, SC_STORAGE_BUFFER]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, B, SC_STORAGE_BUFFER]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        PTR_SB_BUF_F,
        OUT,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_F,
        SC_STORAGE_BUFFER,
        FLOAT,
    ]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_U, SC_FUNCTION, UINT]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_F, SC_FUNCTION, FLOAT]);
    for (id, v) in [(C_U0, 0u32), (C_U1, 1), (C_N, n), (C_K, k), (C_M, m)] {
        w.extend([(4 << 16) | OP_CONSTANT, UINT, id, v]);
    }
    w.extend([(4 << 16) | OP_CONSTANT, FLOAT, C_F0, 0f32.to_bits()]);
    // main
    w.extend([(5 << 16) | OP_FUNCTION, VOID, MAIN, 0, FNTY]);
    w.extend([(2 << 16) | OP_LABEL, B_ENTRY]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_U, BVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_F, ACCVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_LOAD, UVEC3, GIDV, GID]);
    w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, UINT, J, GIDV, 0]);
    w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, UINT, I, GIDV, 1]);
    w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, IN_RANGE, J, C_K]);
    w.extend([(3 << 16) | OP_SELECTION_MERGE, B_ENDIF, 0]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, IN_RANGE, B_THEN, B_ENDIF]);
    w.extend([(2 << 16) | OP_LABEL, B_THEN]);
    w.extend([(5 << 16) | OP_IMUL, UINT, OUTIDX1, I, C_K]);
    w.extend([(5 << 16) | OP_IADD, UINT, OUTIDX, OUTIDX1, J]);
    w.extend([(3 << 16) | OP_STORE, BVAR, C_U0]);
    w.extend([(3 << 16) | OP_STORE, ACCVAR, C_F0]);
    w.extend([(2 << 16) | OP_BRANCH, B_OHDR]);
    w.extend([(2 << 16) | OP_LABEL, B_OHDR]);
    w.extend([(4 << 16) | OP_LOOP_MERGE, B_OMERGE, B_OCONT, 0]);
    w.extend([(2 << 16) | OP_BRANCH, B_OCHK]);
    w.extend([(2 << 16) | OP_LABEL, B_OCHK]);
    w.extend([(4 << 16) | OP_LOAD, UINT, OCHK_C, BVAR]);
    w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, OCOND, OCHK_C, C_M]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, OCOND, B_OBODY, B_OMERGE]);
    w.extend([(2 << 16) | OP_LABEL, B_OBODY]);
    w.extend([(4 << 16) | OP_LOAD, UINT, BB, BVAR]);
    // A[b·N + i]
    w.extend([(5 << 16) | OP_IMUL, UINT, AIDX1, BB, C_N]);
    w.extend([(5 << 16) | OP_IADD, UINT, AIDX, AIDX1, I]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PA, A, C_U0, AIDX]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, AV, PA]);
    // B[b·K + j]
    w.extend([(5 << 16) | OP_IMUL, UINT, BIDX1, BB, C_K]);
    w.extend([(5 << 16) | OP_IADD, UINT, BIDX, BIDX1, J]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PB, B, C_U0, BIDX]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, BV, PB]);
    w.extend([(5 << 16) | OP_FMUL, FLOAT, PROD, AV, BV]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, A0, ACCVAR]);
    w.extend([(5 << 16) | OP_FADD, FLOAT, A1, A0, PROD]);
    w.extend([(3 << 16) | OP_STORE, ACCVAR, A1]);
    w.extend([(2 << 16) | OP_BRANCH, B_OCONT]);
    w.extend([(2 << 16) | OP_LABEL, B_OCONT]);
    w.extend([(4 << 16) | OP_LOAD, UINT, CTR2, BVAR]);
    w.extend([(5 << 16) | OP_IADD, UINT, CTRN, CTR2, C_U1]);
    w.extend([(3 << 16) | OP_STORE, BVAR, CTRN]);
    w.extend([(2 << 16) | OP_BRANCH, B_OHDR]);
    w.extend([(2 << 16) | OP_LABEL, B_OMERGE]);
    // OUT[i·K + j] += acc  (read-modify-write; one owner per element, no atomics)
    w.extend([(4 << 16) | OP_LOAD, FLOAT, ACCF, ACCVAR]);
    w.extend([
        (6 << 16) | OP_ACCESS_CHAIN,
        PTR_SB_F,
        POUT,
        OUT,
        C_U0,
        OUTIDX,
    ]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, OLD, POUT]);
    w.extend([(5 << 16) | OP_FADD, FLOAT, NEW, OLD, ACCF]);
    w.extend([(3 << 16) | OP_STORE, POUT, NEW]);
    w.extend([(2 << 16) | OP_BRANCH, B_ENDIF]);
    w.extend([(2 << 16) | OP_LABEL, B_ENDIF]);
    w.extend([(1 << 16) | OP_RETURN]);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The dW reduction module as bytes.
pub fn gemm_dw_bytes(m: u32, n: u32, k: u32, local: u32) -> Vec<u8> {
    gemm_dw_module(m, n, k, local)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

// ── The cooperative-matrix (tensor-core) tiled GEMM — MILESTONE 3. ───────────────────────────────
// The batched GEMV re-reads the weight per batch row (O(m·n·k) traffic, no reuse) — memory-bound,
// slower than the CPU. The fix is the 16×16×16 MMA (SPV_KHR_cooperative_matrix, confirmed on the
// RTX 5080: M16 N16 K16, A=f16 B=f16 C=f32 — `examples/vk_coopmat_probe.rs`): one instruction folds
// a whole 16×16×16 block on the tensor cores, restoring arithmetic intensity. Mixed precision: f16
// matmul operands, f32 accumulate (the AdamW master weights stay f32 on the host).
//
// ONE builder covers all three trainer GEMMs by mapping each to the canonical `C[M,N] (+)= A[M,K]·
// B[K,N]` with per-operand row/col-major flags (the caller stages each f32 operand into an f16
// buffer, no transpose — the layout flag reads the natural source order) + an accumulate flag:
//   * forward   gemm_nt (out=a·bᵀ):  A=a row-major, B=b COL-major (bᵀ), C store.   M=m N=n K=k
//   * backward  gemm_nn (out=a·b):   A=a row-major, B=b row-major,     C store.   M=m N=n K=l
//   * backward  gemm_tn_add(out+=aᵀ·b): A=a COL-major (aᵀ), B=b row-major, C load+store. M=n N=k K=m
// One subgroup (32 lanes) computes one 16×16 output tile; dispatch `groups = (Np/16, Mp/16)`. All
// dims MUST be multiples of 16 — the caller zero-pads (zero products are exact, extra rows/cols are
// discarded on readback), so the kernel needs no boundary guards.

const OP_COMPOSITE_CONSTRUCT: u32 = 80; //  splat a scalar into a cooperative matrix (zero-init)
const OP_PHI: u32 = 245; //                 loop-carried accumulator (structured, no Function spill)
const OP_TYPE_COOPMAT: u32 = 4456; //       OpTypeCooperativeMatrixKHR (comp, scope, rows, cols, use)
const OP_COOPMAT_LOAD: u32 = 4457; //       OpCooperativeMatrixLoadKHR (ty, res, ptr, layout, stride)
const OP_COOPMAT_STORE: u32 = 4458; //      OpCooperativeMatrixStoreKHR (ptr, obj, layout, stride)
const OP_COOPMAT_MULADD: u32 = 4459; //     OpCooperativeMatrixMulAddKHR (ty, res, a, b, c)
const CAP_FLOAT16: u32 = 9; //              gates OpTypeFloat 16 (device shaderFloat16)
const CAP_STORAGE_BUFFER_16BIT: u32 = 4433; // f16 SSBO element (SPV_KHR_16bit_storage)
const CAP_COOPERATIVE_MATRIX: u32 = 6022; // the MMA (SPV_KHR_cooperative_matrix)
const CAP_VULKAN_MEMORY_MODEL: u32 = 5345; // required alongside CooperativeMatrixKHR
const MEM_VULKAN: u32 = 3; //               MemoryModel.Vulkan (coopmat needs it, not GLSL450)
const COOPMAT_SCOPE_SUBGROUP: u32 = 3; //   the 16×16×16 tile is a subgroup-collective object
const COOP_USE_A: u32 = 0;
const COOP_USE_B: u32 = 1;
const COOP_USE_ACC: u32 = 2;
const COOP_LAYOUT_ROW: u32 = 0;
const COOP_LAYOUT_COL: u32 = 1;
const OP_TYPE_RUNTIME_ARRAY_: u32 = OP_TYPE_RUNTIME_ARRAY; // (alias for readability below)

/// **The cooperative-matrix tiled GEMM module** — `C[M,N] (+)= A[M,K]·B[K,N]`, f16 A/B, f32 C.
///
/// `mp`/`np`/`kp` are the padded dims (multiples of 16). `a_col`/`b_col` pick each operand's memory
/// layout (col-major = the transpose read for free — no host reorder); `accumulate` loads the
/// existing C tile as the MMA seed (`+=`, the dW read-modify-write) instead of a zero splat.
///
/// Layout ⇒ stride (leading dim, in elements): A row-major `Kp` / col-major `Mp`; B row-major `Np`
/// / col-major `Kp`; C always row-major `Np`. One subgroup owns one output tile; the K axis is a
/// structured loop over `Kp/16` MMAs with the accumulator carried by `OpPhi`. Gate:
/// `spirv-val --target-env vulkan1.3` (the coopmat capability needs 1.3) + `coopmat_gemm_check`.
pub fn coopmat_gemm_module(
    mp: u32,
    np: u32,
    kp: u32,
    a_col: bool,
    b_col: bool,
    accumulate: bool,
) -> Vec<u32> {
    assert!(
        mp % 16 == 0 && np % 16 == 0 && kp % 16 == 0,
        "coopmat dims must be padded to a multiple of 16"
    );
    let ktiles = kp / 16;
    let stride_a = if a_col { mp } else { kp };
    let stride_b = if b_col { kp } else { np };
    let stride_c = np;
    let layout_a = if a_col {
        COOP_LAYOUT_COL
    } else {
        COOP_LAYOUT_ROW
    };
    let layout_b = if b_col {
        COOP_LAYOUT_COL
    } else {
        COOP_LAYOUT_ROW
    };

    let mut idc = 0u32;
    macro_rules! id {
        () => {{
            idc += 1;
            idc
        }};
    }
    // structural ids
    let void = id!();
    let fnty = id!();
    let uint = id!();
    let f16 = id!();
    let f32t = id!();
    let bool_t = id!();
    let uvec3 = id!();
    let ptr_in_uvec3 = id!();
    let wgid = id!();
    let rta_f16 = id!();
    let buf_ab = id!();
    let ptr_sb_buf_ab = id!();
    let rta_f32 = id!();
    let buf_c = id!();
    let ptr_sb_buf_c = id!();
    let w_a = id!();
    let w_b = id!();
    let w_c = id!();
    let ptr_sb_f16 = id!();
    let ptr_sb_f32 = id!();
    let ty_a = id!();
    let ty_b = id!();
    let ty_c = id!();
    let main = id!();

    // uint constants (deduped by value) + f32 zero
    let mut const_words: Vec<u32> = Vec::new();
    let mut cvals: Vec<(u32, u32)> = Vec::new(); // (value, id)
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
    let c_16 = uconst(16);
    let c_scope = uconst(COOPMAT_SCOPE_SUBGROUP);
    let c_use_a = uconst(COOP_USE_A);
    let c_use_b = uconst(COOP_USE_B);
    let c_use_c = uconst(COOP_USE_ACC);
    let c_lay_a = uconst(layout_a);
    let c_lay_b = uconst(layout_b);
    let c_lay_row = uconst(COOP_LAYOUT_ROW);
    let c_stride_a = uconst(stride_a);
    let c_stride_b = uconst(stride_b);
    let c_stride_c = uconst(stride_c);
    let c_ktiles = uconst(ktiles);
    let c_1 = uconst(1);
    let f32_0 = id!();
    const_words.extend([(4 << 16) | OP_CONSTANT, f32t, f32_0, 0f32.to_bits()]);

    // body ids
    let b_entry = id!();
    let b_header = id!();
    let b_check = id!();
    let b_body = id!();
    let b_cont = id!();
    let b_merge = id!();
    let acc_phi = id!();
    let kk_phi = id!();
    let acc_init = id!();
    let acc_next = id!();
    let wg = id!();
    let colb = id!();
    let rowb = id!();
    let r0 = id!();
    let c0 = id!();
    let rc0 = id!();
    let base_c = id!();
    let p_cinit = id!();
    let kk16 = id!();
    let base_a = id!();
    let a_tmp = id!(); // col-major kk16*stride
    let base_b = id!();
    let b_tmp = id!();
    let p_a = id!();
    let p_b = id!();
    let mat_a = id!();
    let mat_b = id!();
    let p_out = id!();
    let cond = id!();
    let kk_next = id!();
    let bound = idc + 1;

    let mut w = vec![SPV_MAGIC, SPV_VERSION_1_3, 0, bound, 0];
    // capabilities → extensions → memory model → entry (logical layout order). The cooperative-matrix
    // capability REQUIRES the Vulkan memory model (spirv-val: "…must also be declared").
    for cap in [
        CAP_SHADER,
        CAP_FLOAT16,
        CAP_STORAGE_BUFFER_16BIT,
        CAP_VULKAN_MEMORY_MODEL,
        CAP_COOPERATIVE_MATRIX,
    ] {
        w.extend([(2 << 16) | OP_CAPABILITY, cap]);
    }
    for ext in [
        "SPV_KHR_16bit_storage",
        "SPV_KHR_vulkan_memory_model",
        "SPV_KHR_cooperative_matrix",
    ] {
        let e = lit_string(ext);
        w.push(((1 + e.len() as u32) << 16) | OP_EXTENSION);
        w.extend(&e);
    }
    w.extend([(3 << 16) | OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_VULKAN]);
    let mn = lit_string("main");
    w.push(((3 + mn.len() as u32 + 1) << 16) | OP_ENTRY_POINT);
    w.extend([MODEL_GL_COMPUTE, main]);
    w.extend(&mn);
    w.push(wgid);
    w.extend([
        (6 << 16) | OP_EXECUTION_MODE,
        main,
        MODE_LOCAL_SIZE,
        32,
        1,
        1,
    ]);
    // decorations
    w.extend([(4 << 16) | OP_DECORATE, wgid, DEC_BUILT_IN, 26]); // BuiltIn.WorkgroupId
    w.extend([(4 << 16) | OP_DECORATE, rta_f16, DEC_ARRAY_STRIDE, 2]);
    w.extend([(3 << 16) | OP_DECORATE, buf_ab, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, buf_ab, 0, DEC_OFFSET, 0]);
    w.extend([(4 << 16) | OP_DECORATE, rta_f32, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, buf_c, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, buf_c, 0, DEC_OFFSET, 0]);
    for (var, binding) in [(w_a, 0u32), (w_b, 1), (w_c, 2)] {
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, var, DEC_BINDING, binding]);
    }
    // scalar types
    w.extend([(2 << 16) | OP_TYPE_VOID, void]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, fnty, void]);
    w.extend([(4 << 16) | OP_TYPE_INT, uint, 32, 0]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, f16, 16]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, f32t, 32]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, bool_t]);
    w.extend([(4 << 16) | OP_TYPE_VECTOR, uvec3, uint, 3]);
    // constants (must precede the cooperative-matrix types that reference them)
    w.extend(&const_words);
    // cooperative-matrix types: A/B are f16, the accumulator is f32
    w.extend([
        (7 << 16) | OP_TYPE_COOPMAT,
        ty_a,
        f16,
        c_scope,
        c_16,
        c_16,
        c_use_a,
    ]);
    w.extend([
        (7 << 16) | OP_TYPE_COOPMAT,
        ty_b,
        f16,
        c_scope,
        c_16,
        c_16,
        c_use_b,
    ]);
    w.extend([
        (7 << 16) | OP_TYPE_COOPMAT,
        ty_c,
        f32t,
        c_scope,
        c_16,
        c_16,
        c_use_c,
    ]);
    // input builtin + SSBO types/vars
    w.extend([(4 << 16) | OP_TYPE_POINTER, ptr_in_uvec3, SC_INPUT, uvec3]);
    w.extend([(4 << 16) | OP_VARIABLE, ptr_in_uvec3, wgid, SC_INPUT]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY_, rta_f16, f16]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, buf_ab, rta_f16]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_buf_ab,
        SC_STORAGE_BUFFER,
        buf_ab,
    ]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY_, rta_f32, f32t]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, buf_c, rta_f32]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_buf_c,
        SC_STORAGE_BUFFER,
        buf_c,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_ab,
        w_a,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_ab,
        w_b,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ptr_sb_buf_c,
        w_c,
        SC_STORAGE_BUFFER,
    ]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_f16,
        SC_STORAGE_BUFFER,
        f16,
    ]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ptr_sb_f32,
        SC_STORAGE_BUFFER,
        f32t,
    ]);

    // ── main ──
    w.extend([(5 << 16) | OP_FUNCTION, void, main, 0, fnty]);
    w.extend([(2 << 16) | OP_LABEL, b_entry]);
    w.extend([(4 << 16) | OP_LOAD, uvec3, wg, wgid]);
    w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, uint, colb, wg, 0]);
    w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, uint, rowb, wg, 1]);
    w.extend([(5 << 16) | OP_IMUL, uint, r0, rowb, c_16]);
    w.extend([(5 << 16) | OP_IMUL, uint, c0, colb, c_16]);
    // C tile base (row-major, kk-independent): r0·Np + c0
    w.extend([(5 << 16) | OP_IMUL, uint, rc0, r0, c_stride_c]);
    w.extend([(5 << 16) | OP_IADD, uint, base_c, rc0, c0]);
    // accumulator seed: load the current C tile (+=) or splat zero
    if accumulate {
        w.extend([
            (6 << 16) | OP_ACCESS_CHAIN,
            ptr_sb_f32,
            p_cinit,
            w_c,
            c_u0,
            base_c,
        ]);
        w.extend([
            (6 << 16) | OP_COOPMAT_LOAD,
            ty_c,
            acc_init,
            p_cinit,
            c_lay_row,
            c_stride_c,
        ]);
    } else {
        w.extend([(4 << 16) | OP_COMPOSITE_CONSTRUCT, ty_c, acc_init, f32_0]);
    }
    w.extend([(2 << 16) | OP_BRANCH, b_header]);
    // loop header: the accumulator + k-tile counter are loop-carried (OpPhi)
    w.extend([(2 << 16) | OP_LABEL, b_header]);
    w.extend([
        (7 << 16) | OP_PHI,
        ty_c,
        acc_phi,
        acc_init,
        b_entry,
        acc_next,
        b_cont,
    ]);
    w.extend([
        (7 << 16) | OP_PHI,
        uint,
        kk_phi,
        c_u0,
        b_entry,
        kk_next,
        b_cont,
    ]);
    w.extend([(4 << 16) | OP_LOOP_MERGE, b_merge, b_cont, 0]);
    w.extend([(2 << 16) | OP_BRANCH, b_check]);
    w.extend([(2 << 16) | OP_LABEL, b_check]);
    w.extend([(5 << 16) | OP_ULESS_THAN, bool_t, cond, kk_phi, c_ktiles]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, cond, b_body, b_merge]);
    w.extend([(2 << 16) | OP_LABEL, b_body]);
    w.extend([(5 << 16) | OP_IMUL, uint, kk16, kk_phi, c_16]);
    // A tile base: row-major r0·Ka + kk16 ; col-major kk16·Ma + r0
    if a_col {
        w.extend([(5 << 16) | OP_IMUL, uint, a_tmp, kk16, c_stride_a]);
        w.extend([(5 << 16) | OP_IADD, uint, base_a, a_tmp, r0]);
    } else {
        w.extend([(5 << 16) | OP_IMUL, uint, a_tmp, r0, c_stride_a]);
        w.extend([(5 << 16) | OP_IADD, uint, base_a, a_tmp, kk16]);
    }
    // B tile base: row-major kk16·Nb + c0 ; col-major c0·Kb + kk16
    if b_col {
        w.extend([(5 << 16) | OP_IMUL, uint, b_tmp, c0, c_stride_b]);
        w.extend([(5 << 16) | OP_IADD, uint, base_b, b_tmp, kk16]);
    } else {
        w.extend([(5 << 16) | OP_IMUL, uint, b_tmp, kk16, c_stride_b]);
        w.extend([(5 << 16) | OP_IADD, uint, base_b, b_tmp, c0]);
    }
    w.extend([
        (6 << 16) | OP_ACCESS_CHAIN,
        ptr_sb_f16,
        p_a,
        w_a,
        c_u0,
        base_a,
    ]);
    w.extend([
        (6 << 16) | OP_COOPMAT_LOAD,
        ty_a,
        mat_a,
        p_a,
        c_lay_a,
        c_stride_a,
    ]);
    w.extend([
        (6 << 16) | OP_ACCESS_CHAIN,
        ptr_sb_f16,
        p_b,
        w_b,
        c_u0,
        base_b,
    ]);
    w.extend([
        (6 << 16) | OP_COOPMAT_LOAD,
        ty_b,
        mat_b,
        p_b,
        c_lay_b,
        c_stride_b,
    ]);
    w.extend([
        (6 << 16) | OP_COOPMAT_MULADD,
        ty_c,
        acc_next,
        mat_a,
        mat_b,
        acc_phi,
    ]);
    w.extend([(2 << 16) | OP_BRANCH, b_cont]);
    w.extend([(2 << 16) | OP_LABEL, b_cont]);
    w.extend([(5 << 16) | OP_IADD, uint, kk_next, kk_phi, c_1]);
    w.extend([(2 << 16) | OP_BRANCH, b_header]);
    // merge: store the accumulated tile (the header phi holds the final value here)
    w.extend([(2 << 16) | OP_LABEL, b_merge]);
    w.extend([
        (6 << 16) | OP_ACCESS_CHAIN,
        ptr_sb_f32,
        p_out,
        w_c,
        c_u0,
        base_c,
    ]);
    w.extend([
        (5 << 16) | OP_COOPMAT_STORE,
        p_out,
        acc_phi,
        c_lay_row,
        c_stride_c,
    ]);
    w.extend([(1 << 16) | OP_RETURN]);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The cooperative-matrix GEMM module as bytes.
pub fn coopmat_gemm_bytes(
    mp: u32,
    np: u32,
    kp: u32,
    a_col: bool,
    b_col: bool,
    accumulate: bool,
) -> Vec<u8> {
    coopmat_gemm_module(mp, np, kp, a_col, b_col, accumulate)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// f32 → IEEE-754 binary16 bits (round-to-nearest-even) — the operand cast at the coopmat buffer
/// boundary (the MMA takes f16 A/B). Handles subnormals, overflow→inf, and inf/nan.
pub fn f16_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let e = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | (if mant != 0 { 0x200 } else { 0 }); // inf / nan
    }
    let exp = e - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // too small → signed zero
        }
        let m = mant | 0x80_0000; // implicit 1
        let shift = (14 - exp) as u32;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = ((exp as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1; // carry rolls mantissa into exponent naturally
    }
    sign | half
}
