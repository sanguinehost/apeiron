// A faithful port of fold's `modelfold::simt` (`fold` is the lineage named by the programme
// direction). The emitter is byte-identical in structure and id layout to the lineage; rewriting
// it to apeiron's pedantic clippy profile would be a re-derivation, not a port. The lint profile is
// therefore inherited from fold for this module tree only, with the reason stated here.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's modelfold::simt SPIR-V emitters; lint profile is fold's, \
              not apeiron's"
)]

//! simt — the SIMT (GPU) substrate value: the 12-op trinity alphabet encoded to SPIR-V.
//!
//! The scalar substrates (x86-64 / ARM64 / RISC-V) are three *values* of one `Isa` structure —
//! the target is a data parameter, never a hand-split backend (song `song/lean/Song/run/Metal.lean`,
//! sanguine `proof/Runtime/MetalNative.lean`). This module is the **fourth value**: the same
//! twelve ops (song `song/lean/Song/kernel/ISA.lean`, `SimtIsa` — `encodeOp : Op → (tyId resId aId bId) →
//! List Nat`), lowered to SPIR-V words instead of instruction bytes. SSA ids replace registers;
//! everything else is the same shape: one op table, one module wrapper (the ELF-header analog),
//! one reference the emitted code must match ([`op_ref`], the `Prog.eval` analog).
//!
//! The compute alphabet is ONE trinity {throne, select, accum} — the same {B, NB, B/U} as the
//! table's merges — each face 3 core ops + 1 doorway. 3 × 4 = 12, closed.
//!
//! Every SPIR-V word below is cited to the local Khronos grammar
//! (`~/Workspace/SPIRV-Headers/include/spirv/unified1/spirv.core.grammar.json`, and
//! `extinst.glsl.std.450.grammar.json` for the throne extremum), never from memory. The modules
//! are SPIR-V 1.3 (`StorageBuffer` storage class, version 1.3 in the grammar) targeting
//! Vulkan 1.1+; gate with `spirv-val --target-env vulkan1.1` (`tests/simt.rs`) and run on the
//! metal with `examples/simt_check.rs`.
//!
//! Scope, honestly: the encoder, a one-invocation op check, and ONE kernel — the GEMV fold
//! ([`gemv_module`], licensed by sanguine `SplitReduce`/`FusionCost`/`MemoryLevelParallelism`,
//! sustained 895±5 GB/s fp32 on RTX 5080 vs cuBLAS's 902-911 at boost — parity within 1-2%,
//! both ≥93% of the 960 spec). The wave face (ballot/shuffle — sanguine `SimtSelect.movemask`,
//! proven), the four merges as atomics, and the nibble chart GEMV are named next rungs.
//!
//! **Builder families — the GPU primitive surface** (dispatched by [`crate::gpu::Gpu`], the
//! device-side parallel of the CPU `Table`). The public `_bytes` builders here are two families:
//! the *op alphabet* ([`spv_bytes`] — one of the 12 ops as a one-invocation kernel) and *GEMM*
//! ([`gemv_bytes`], [`gemv_dp4a_bytes`], [`gemv_dp4a_dyn_bytes`] — the quantized matmul, the
//! [`Carrier`] the dial, the crown jewel). The transformer glue (norm/rotor/gate/attention)
//! lives one module over in the transformer-glue module (fold `simt_nn`); both are `#[cfg]`-free (always built) — only the
//! runtime that dispatches them is behind the `gpu` feature.
// ---------------------------------------------------------------------------------------------
// The public surface, unchanged by the split of this file into `simt/`.
//
// Every path a caller could name before the split -- `simt::gemv_bytes`, `::gemv_dp4a_sb_bytes`,
// `::coopmat_gemm_module`, ... -- still resolves here. The alphabet, the pinned SPIR-V response
// set / id layout, the module wrapper, the `Carrier` dial and the CPU references stay in this
// file because every kernel family below reads them; `use super::*;` in each submodule
// reproduces exactly the name set the single file had, private items included.
// ---------------------------------------------------------------------------------------------

mod dp4a;
mod gemv;
mod superblock;

pub use dp4a::*;
pub use gemv::*;
pub use superblock::*;

/// The three compute faces — the trinity, same as the table's merges.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpFace {
    /// B — extremum / unit / compare: the throne face (the set/max merge on the metal).
    Throne,
    /// NB — the bit ops: the select face (the keep/bit merge on the metal).
    Select,
    /// B/U — arithmetic / place: the accum face (the count merge on the metal).
    Accum,
}

/// The 12 metal ops: each face's sub-trinity (3 core + 1 doorway). Mirrors song
/// `song/lean/Song/kernel/ISA.lean` `Op` exactly — same names, same split, same doorways.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// throne: max (the B merge).
    TMax,
    /// throne: min.
    TMin,
    /// throne: unit — the deliberate no-op (result = first operand, no instruction).
    TUnit,
    /// throne doorway: signed less-than predicate (1 / 0).
    TCmp,
    /// select: xor.
    SXor,
    /// select: and.
    SAnd,
    /// select: or.
    SOr,
    /// select doorway: complement (unary, on the first operand).
    SNot,
    /// accum: add (the count merge).
    AAdd,
    /// accum: mul.
    AMul,
    /// accum: signed division (truncated, like x86 `idiv` / RISC-V `DIV`).
    ADiv,
    /// accum doorway: shift left (place).
    AShift,
}

impl Op {
    /// All twelve, 4/4/4 across the faces — the closed compute alphabet.
    pub const ALL: [Op; 12] = [
        Op::TMax,
        Op::TMin,
        Op::TUnit,
        Op::TCmp,
        Op::SXor,
        Op::SAnd,
        Op::SOr,
        Op::SNot,
        Op::AAdd,
        Op::AMul,
        Op::ADiv,
        Op::AShift,
    ];

    /// Which face this op belongs to.
    pub const fn face(self) -> OpFace {
        match self {
            Op::TMax | Op::TMin | Op::TUnit | Op::TCmp => OpFace::Throne,
            Op::SXor | Op::SAnd | Op::SOr | Op::SNot => OpFace::Select,
            Op::AAdd | Op::AMul | Op::ADiv | Op::AShift => OpFace::Accum,
        }
    }

    /// The three doorways (compare / complement / place) — one per face.
    pub const fn is_doorway(self) -> bool {
        matches!(self, Op::TCmp | Op::SNot | Op::AShift)
    }

    /// A short stable name (test labels, artifact filenames).
    pub const fn name(self) -> &'static str {
        match self {
            Op::TMax => "tmax",
            Op::TMin => "tmin",
            Op::TUnit => "tunit",
            Op::TCmp => "tcmp",
            Op::SXor => "sxor",
            Op::SAnd => "sand",
            Op::SOr => "sor",
            Op::SNot => "snot",
            Op::AAdd => "aadd",
            Op::AMul => "amul",
            Op::ADiv => "adiv",
            Op::AShift => "ashift",
        }
    }
}

// ── The SPIR-V response set (spirv.core.grammar.json; version word = 1.3). ──────────────────────
// Instruction word 0 is (word_count << 16) | opcode; operands follow, one word each (strings are
// packed UTF-8, NUL-terminated, LE — "Physical Layout of a SPIR-V Module and Instruction").

// Core opcodes, cited from spirv.core.grammar.json (`"opname" : …, "opcode" : …`):
const OP_EXT_INST_IMPORT: u32 = 11; // OpExtInstImport (result, name)
const OP_EXT_INST: u32 = 12; //        OpExtInst (type, result, set, instruction, operands…)
const OP_MEMORY_MODEL: u32 = 14; //    OpMemoryModel (addressing, memory)
const OP_ENTRY_POINT: u32 = 15; //     OpEntryPoint (model, entry, name, interface…)
const OP_EXECUTION_MODE: u32 = 16; //  OpExecutionMode (entry, mode, literals…)
const OP_CAPABILITY: u32 = 17; //      OpCapability (capability)
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21; //        OpTypeInt (result, width, signedness)
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32; //    OpTypePointer (result, storage class, type)
const OP_TYPE_FUNCTION: u32 = 33;
const OP_CONSTANT: u32 = 43; //        OpConstant (type, result, value)
const OP_FUNCTION: u32 = 54; //        OpFunction (type, result, control, fn type)
const OP_FUNCTION_END: u32 = 56;
const OP_VARIABLE: u32 = 59; //        OpVariable (type, result, storage class)
const OP_LOAD: u32 = 61; //            OpLoad (type, result, pointer)
const OP_STORE: u32 = 62; //           OpStore (pointer, object)
const OP_ACCESS_CHAIN: u32 = 65; //    OpAccessChain (type, result, base, indexes…)
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_IADD: u32 = 128;
const OP_IMUL: u32 = 132;
const OP_SDIV: u32 = 135;
const OP_SELECT: u32 = 169; //         OpSelect (type, result, cond, then, else)
const OP_SLESS_THAN: u32 = 177; //     OpSLessThan (bool type, result, a, b)
const OP_SHIFT_LEFT_LOGICAL: u32 = 196; // (type, result, base, shift)
const OP_BITWISE_OR: u32 = 197;
const OP_BITWISE_XOR: u32 = 198;
const OP_BITWISE_AND: u32 = 199;
const OP_NOT: u32 = 200;
const OP_LABEL: u32 = 248;
const OP_RETURN: u32 = 253;

// Enums, cited from spirv.core.grammar.json operand_kinds:
const CAP_SHADER: u32 = 1; //          Capability.Shader
const MODEL_GL_COMPUTE: u32 = 5; //    ExecutionModel.GLCompute
const ADDR_LOGICAL: u32 = 0; //        AddressingModel.Logical
const MEM_GLSL450: u32 = 1; //         MemoryModel.GLSL450
const MODE_LOCAL_SIZE: u32 = 17; //    ExecutionMode.LocalSize
const SC_STORAGE_BUFFER: u32 = 12; //  StorageClass.StorageBuffer (version 1.3)
const DEC_BLOCK: u32 = 2; //           Decoration.Block
const DEC_BINDING: u32 = 33; //        Decoration.Binding
const DEC_DESCRIPTOR_SET: u32 = 34; // Decoration.DescriptorSet
const DEC_OFFSET: u32 = 35; //         Decoration.Offset
                            // extinst.glsl.std.450.grammar.json: the throne extremum is the GLSL.std.450 extended set —
                            // core SPIR-V has no integer max/min instruction (grep the grammar), same as x86 needing
                            // cmp+cmov for `maxRR`.
const GLSL_UNPACK_HALF2X16: u32 = 62; // UnpackHalf2x16 (v) — the F16 carrier's decode
const GLSL_SMIN: u32 = 39; //          SMin (x, y)
const GLSL_SMAX: u32 = 42; //          SMax (x, y)

/// SPIR-V magic (spirv.core.grammar.json `"magic_number"`).
pub const SPV_MAGIC: u32 = 0x0723_0203;
/// Version word 1.3 (0 | major | minor | 0) — the floor for `StorageBuffer` without extensions.
pub const SPV_VERSION_1_3: u32 = 0x0001_0300;

// ── The fixed id layout of [`spv_module`] — the wrapper is ours, so the auxiliary ids the
//    doorway ops need (bool type, 0/1 constants, the ext-inst set, one scratch) are pinned data,
//    exactly like the scalar ISAs pinning rd=reg0/rs=reg1 for `shlV`/`setlt` (Metal.lean). ──────
/// %1 — OpExtInstImport "GLSL.std.450" (the throne extremum's set).
pub const ID_GLSL: u32 = 1;
const ID_VOID: u32 = 2;
const ID_FN_TY: u32 = 3;
/// %4 — OpTypeInt 32 1: the one carrier type (the paper's abstract model is per-thread 32-bit
/// scalar; signedness tags the type, the opcode carries the semantics).
pub const ID_INT: u32 = 4;
/// %5 — OpTypeBool (the compare doorway's intermediate).
pub const ID_BOOL: u32 = 5;
const ID_PTR_INT: u32 = 6;
const ID_STRUCT: u32 = 7;
const ID_PTR_STRUCT: u32 = 8;
const ID_VAR: u32 = 9;
/// %10 — int constant 0 (compare doorway's "else"; struct index of `a`).
pub const ID_C0: u32 = 10;
/// %11 — int constant 1 (compare doorway's "then"; struct index of `b`).
pub const ID_C1: u32 = 11;
const ID_C2: u32 = 12;
const ID_MAIN: u32 = 13;
const ID_ENTRY: u32 = 14;
const ID_AC_A: u32 = 15;
/// %16 — the loaded first operand `a`.
pub const ID_A: u32 = 16;
const ID_AC_B: u32 = 17;
/// %18 — the loaded second operand `b`.
pub const ID_B: u32 = 18;
/// %19 — the doorway scratch (TCmp's OpSLessThan bool result); free for single-instruction ops.
pub const ID_SCRATCH: u32 = 19;
/// %20 — the op result every body writes (TUnit writes none; its result IS `a`).
pub const ID_RES: u32 = 20;
const ID_AC_OUT: u32 = 21;
const ID_BOUND: u32 = 22;

/// **The SimtIsa encoder** — song `ISA.lean`'s `SimtIsa.encodeOp : Op → (tyId resId aId bId) →
/// List Nat`, filled in. One op → its SPIR-V words, SSA form (`res = op a b` over type `ty`).
///
/// The three doorways lean on the wrapper's pinned ids, exactly as the scalar encoders pin
/// registers: `TCmp` is OpSLessThan (into [`ID_SCRATCH`], type [`ID_BOOL`]) then OpSelect
/// 1/0 ([`ID_C1`]/[`ID_C0`]) — the SPIR-V `setlt` sequence; `TMax`/`TMin` are OpExtInst
/// GLSL.std.450 SMax/SMin via [`ID_GLSL`]; `TUnit` is the deliberate no-op (empty, result = `a`),
/// same as `toIns`'s `[]`.
pub fn encode_op(op: Op, ty: u32, res: u32, a: u32, b: u32) -> Vec<u32> {
    let bin = |opc: u32| vec![(5 << 16) | opc, ty, res, a, b];
    match op {
        Op::AAdd => bin(OP_IADD),
        Op::AMul => bin(OP_IMUL),
        Op::ADiv => bin(OP_SDIV),
        Op::AShift => bin(OP_SHIFT_LEFT_LOGICAL),
        Op::SXor => bin(OP_BITWISE_XOR),
        Op::SAnd => bin(OP_BITWISE_AND),
        Op::SOr => bin(OP_BITWISE_OR),
        Op::SNot => vec![(4 << 16) | OP_NOT, ty, res, a],
        Op::TMax => vec![(7 << 16) | OP_EXT_INST, ty, res, ID_GLSL, GLSL_SMAX, a, b],
        Op::TMin => vec![(7 << 16) | OP_EXT_INST, ty, res, ID_GLSL, GLSL_SMIN, a, b],
        Op::TCmp => vec![
            (5 << 16) | OP_SLESS_THAN,
            ID_BOOL,
            ID_SCRATCH,
            a,
            b,
            (6 << 16) | OP_SELECT,
            ty,
            res,
            ID_SCRATCH,
            ID_C1,
            ID_C0,
        ],
        Op::TUnit => vec![],
    }
}

/// A NUL-terminated LE-packed literal string, padded to the word (the SPIR-V string encoding).
fn lit_string(s: &str) -> Vec<u32> {
    let mut bytes: Vec<u8> = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// **The module wrapper — the ELF-header analog** (`Metal.elfReducerM` for the SIMT face): wrap
/// one op body in a complete SPIR-V 1.3 compute module. One SSBO (set 0, binding 0) of three
/// ints `{a, b, out}` at offsets 0/4/8; entry `main`, LocalSize 1·1·1 — ONE invocation, the
/// scalar check (the wave face is the next rung, deliberately not here). Loads `a`,`b`, runs the
/// op ([`encode_op`] at the pinned ids), stores the result.
pub fn spv_module(op: Op) -> Vec<u32> {
    let mut w = vec![SPV_MAGIC, SPV_VERSION_1_3, 0, ID_BOUND, 0];
    // capabilities / imports / memory model / entry (the spec's logical-layout order)
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
    let glsl = lit_string("GLSL.std.450"); // 4 words
    w.push(((2 + glsl.len() as u32) << 16) | OP_EXT_INST_IMPORT);
    w.push(ID_GLSL);
    w.extend(&glsl);
    w.extend([(3 << 16) | OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450]);
    let main = lit_string("main"); // 2 words
    w.push(((3 + main.len() as u32) << 16) | OP_ENTRY_POINT);
    w.extend([MODEL_GL_COMPUTE, ID_MAIN]);
    w.extend(&main);
    w.extend([
        (6 << 16) | OP_EXECUTION_MODE,
        ID_MAIN,
        MODE_LOCAL_SIZE,
        1,
        1,
        1,
    ]);
    // decorations: the SSBO block {int,int,int} @ 0/4/8, set 0 binding 0
    w.extend([(3 << 16) | OP_DECORATE, ID_STRUCT, DEC_BLOCK]);
    for (member, off) in [(0u32, 0u32), (1, 4), (2, 8)] {
        w.extend([
            (5 << 16) | OP_MEMBER_DECORATE,
            ID_STRUCT,
            member,
            DEC_OFFSET,
            off,
        ]);
    }
    w.extend([(4 << 16) | OP_DECORATE, ID_VAR, DEC_DESCRIPTOR_SET, 0]);
    w.extend([(4 << 16) | OP_DECORATE, ID_VAR, DEC_BINDING, 0]);
    // types / constants / the variable
    w.extend([(2 << 16) | OP_TYPE_VOID, ID_VOID]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, ID_FN_TY, ID_VOID]);
    w.extend([(4 << 16) | OP_TYPE_INT, ID_INT, 32, 1]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, ID_BOOL]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ID_PTR_INT,
        SC_STORAGE_BUFFER,
        ID_INT,
    ]);
    w.extend([
        (5 << 16) | OP_TYPE_STRUCT,
        ID_STRUCT,
        ID_INT,
        ID_INT,
        ID_INT,
    ]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        ID_PTR_STRUCT,
        SC_STORAGE_BUFFER,
        ID_STRUCT,
    ]);
    w.extend([
        (4 << 16) | OP_VARIABLE,
        ID_PTR_STRUCT,
        ID_VAR,
        SC_STORAGE_BUFFER,
    ]);
    for (id, v) in [(ID_C0, 0u32), (ID_C1, 1), (ID_C2, 2)] {
        w.extend([(4 << 16) | OP_CONSTANT, ID_INT, id, v]);
    }
    // main: load a, load b, the op body, store out
    w.extend([(5 << 16) | OP_FUNCTION, ID_VOID, ID_MAIN, 0, ID_FN_TY]);
    w.extend([(2 << 16) | OP_LABEL, ID_ENTRY]);
    w.extend([
        (5 << 16) | OP_ACCESS_CHAIN,
        ID_PTR_INT,
        ID_AC_A,
        ID_VAR,
        ID_C0,
    ]);
    w.extend([(4 << 16) | OP_LOAD, ID_INT, ID_A, ID_AC_A]);
    w.extend([
        (5 << 16) | OP_ACCESS_CHAIN,
        ID_PTR_INT,
        ID_AC_B,
        ID_VAR,
        ID_C1,
    ]);
    w.extend([(4 << 16) | OP_LOAD, ID_INT, ID_B, ID_AC_B]);
    w.extend(encode_op(op, ID_INT, ID_RES, ID_A, ID_B));
    let result = if matches!(op, Op::TUnit) {
        ID_A
    } else {
        ID_RES
    };
    w.extend([
        (5 << 16) | OP_ACCESS_CHAIN,
        ID_PTR_INT,
        ID_AC_OUT,
        ID_VAR,
        ID_C2,
    ]);
    w.extend([(3 << 16) | OP_STORE, ID_AC_OUT, result]);
    w.extend([(1 << 16) | OP_RETURN]);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// The module as bytes (LE words) — what `spirv-val` reads and `vkCreateShaderModule` loads.
pub fn spv_bytes(op: Op) -> Vec<u8> {
    spv_module(op)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

// ── The first kernel on the substrate: the GEMV fold. ──────────────────────────────────────────
// Decode is bandwidth-bound (sanguine THE_LATENCY_IS_THE_PATH, measured 2026-07-05), and the
// bandwidth kernel is the matvec. The framework answer to GPU coalescing is LAYOUT AS DATA:
// store W column-major (free at seal time — the transpose is data, not code), give each
// invocation one output row, and the kernel is a bare fold over k — at step k, invocation r
// reads `W[k·M + r]`: consecutive invocations read consecutive addresses, perfectly coalesced.
// No shared memory, no atomics, no scheduler — the fold IS the kernel (`fold_is_one_kernel`,
// sanguine FusionCost.lean).
//
// New words this needs, all cited from spirv.core.grammar.json: OpTypeFloat 22, OpTypeVector 23,
// OpTypeRuntimeArray 29, OpCompositeExtract 81, OpFAdd 129, OpFMul 133, OpULessThan 176,
// OpLoopMerge 246, OpSelectionMerge 247, OpBranch 249, OpBranchConditional 250; Decoration
// ArrayStride = 6, BuiltIn = 11; BuiltIn.GlobalInvocationId = 28; StorageClass Input = 1,
// Function = 7.

const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_EXTENSION: u32 = 10; //     OpExtension (name) — SPV_KHR_integer_dot_product on 1.3
const OP_CONVERT_S_TO_F: u32 = 111; // OpConvertSToF (type, result, operand) — int acc → float
const OP_CONVERT_U_TO_F: u32 = 112; // OpConvertUToF (type, result, operand)
const OP_VECTOR_EXTRACT_DYNAMIC: u32 = 77; // (type, result, vector, index)
const OP_SHIFT_RIGHT_LOGICAL: u32 = 194; // (type, result, base, shift)
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_FADD: u32 = 129;
const OP_FMUL: u32 = 133;
const OP_ULESS_THAN: u32 = 176;
const OP_LOOP_MERGE: u32 = 246;
const OP_SELECTION_MERGE: u32 = 247;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILT_IN: u32 = 11;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const SC_INPUT: u32 = 1;
const SC_FUNCTION: u32 = 7;

// The integer-dot lever (SPV_KHR_integer_dot_product; grammar opcodes/enumerants above).
// OpSUDot(a, w, 4x8) = Σ of 4 products (a signed int8, w unsigned uint8) into one i32 —
// the hardware DP4A/IDP4A, `integerDotProduct4x8BitPackedMixedSignednessAccelerated = true`
// on the 5080. Collapses the q-code inner loop's per-element SHR·AND·I2FP·FFMA (the float tax
// the SASS named) into ONE accelerated instruction per 4 codes; licensed by
// `SealChartFold.fold_in_the_seed_chart` (the integer inner sum, |Σqx| int32-bounded).
const OP_SUDOT: u32 = 4452; //         OpSUDot (type, result, vec1-signed, vec2-unsigned, format)
const CAP_DOT_PRODUCT: u32 = 6019; //  Capability.DotProduct
const CAP_DOT_INPUT_4X8_PACKED: u32 = 6018; // Capability.DotProductInput4x8BitPacked
                                            // ── the PhysicalStorageBuffer face (VK_KHR_buffer_device_address) ────────────────────────────────
                                            // A descriptor is a base AND a range, and the driver spends one compare per load on the range. A
                                            // raw device address has no range, so the mandala mask inside the kernel becomes the only bound —
                                            // `MandalaMemory.masked_load_never_oob` with nothing bolted on after it.
const CAP_INT64: u32 = 11; //                          gates OpTypeInt 64 (device shaderInt64)
const CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES: u32 = 5347;
const ADDR_PHYSICAL_STORAGE_BUFFER64: u32 = 5348; //   AddressingModel.PhysicalStorageBuffer64
const SC_PHYSICAL_STORAGE_BUFFER: u32 = 5349; //       StorageClass.PhysicalStorageBuffer
const OP_CONVERT_U_TO_PTR: u32 = 120; //               OpConvertUToPtr (type, result, u64 address)
/// `MemoryAccess.Aligned` — mandatory on every load/store through a `PhysicalStorageBuffer`
/// pointer, because a raw address carries no layout the driver could infer alignment from.
const MEM_ALIGNED: u32 = 0x2;
const PACKED_VEC_4X8: u32 = 0; //      PackedVectorFormat.PackedVectorFormat4x8Bit
/// `OpSUDotAccSat` — the SAME dot with a third operand, the accumulator. Read out of the SASS, not
/// guessed: `gemv_dp4a_sb`'s hot loop emits `IDP.4A.S8.U8 Rd, Ra, Rb, **RZ**` and then a separate
/// `IADD3` tree, so **524 of that loop's 707 `IADD3+LEA` are the ACCUMULATE, not address
/// arithmetic** — 19.3 % of every issue slot spent adding a number the dot instruction can take as
/// its addend. `SPV_KHR_integer_dot_product` gates it under the same `DotProduct` capability.
///
/// Saturation is unreachable here and that is what makes the swap exact, not approximate: one dot
/// is `Σ codeᵢ·qᵢ` over 4 lanes with `code ∈ {0,1,2}` (`TernaryDP4A.trit_code_fits`) and
/// `|q| ≤ 127`, so `|dot| ≤ 1524`; `accq` is cleared every superblock and takes 64 of them, so
/// `|accq| ≤ 97 536 ≪ 2³¹`. The clamp therefore never fires and `OpSUDotAccSat ≡ OpSUDot + OpIAdd`
/// on every input this kernel can be given.
const OP_SUDOT_ACC_SAT: u32 = 4455; // OpSUDotAccSat (type, result, vec1-s, vec2-u, acc, format)
                                    // ── the LDS staging face: the activation tile is WORKGROUP-uniform, so it belongs in shared ──────
const OP_TYPE_ARRAY: u32 = 28; //      OpTypeArray (result, element type, length)
const OP_CONTROL_BARRIER: u32 = 224; // OpControlBarrier (exec scope, mem scope, semantics)
const SC_WORKGROUP: u32 = 4; //        StorageClass.Workgroup
const SCOPE_WORKGROUP: u32 = 2; //     Scope.Workgroup
const SEM_WG_ACQREL: u32 = 0x108; //   MemorySemantics.AcquireRelease | WorkgroupMemory

/// The weight carrier — **precision is one dial, not a kernel zoo** (sanguine
/// `MetalNative` §7b: the bit width is the applicative, ternary/4-bit/8-bit are the SAME
/// kernel at different α; song `Seed.quantGen`: one mechanism, dialed). Direct carriers
/// (F32, F16) load the weight; chart carriers (Q8, Q4) run the fold in the (code, scale)
/// chart (`SealChartFold.fold_in_the_seed_chart`) — nothing dequantizes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Carrier {
    /// 32-bit float, direct load.
    F32,
    /// 16-bit float, 2 per u32 word, decoded by GLSL.std.450 UnpackHalf2x16 (no 16-bit
    /// storage capability needed — the pack is the same shared-word algebra as the codes).
    F16,
    /// 8-bit unsigned codes, 4 per word, affine per 32-block: `w = d·q + m`.
    Q8,
    /// 4-bit unsigned codes, 8 per word, affine per 32-block.
    Q4,
    /// 2-bit unsigned codes, 16 per word, affine per 32-block — the ternary/trit floor
    /// (`{0,1,2}` codes, `w = scale·(q−1)` gives `{−1,0,+1}`; sanguine `TernaryDP4A`). 4× the
    /// weight density of Q8: the byte lever's terminal DP4A width (`ternary_transport_floor`).
    Q2,
    /// **3-bit unsigned codes as TWO PLANES**, affine per 32-block — the rung between Q2 and Q4
    /// that `results/scale-ladder.md` says we need: our 3-bit chart at 3.5 bpw scores 0.001839
    /// against llama.cpp IQ3_XXS's 0.001861 at 3.0625, a tie, where our Q2 at 2.5 bpw is 2.32x
    /// worse. Three bits do not divide 32, which is why `block_q3_K` splits into `qs` + `hmask`
    /// and why this does too:
    ///
    /// * **plane A** — the low 2 bits, 16 codes per u32, byte-identical in layout to [`Self::Q2`]
    ///   (`A[(kk/16)·m + r]`), so the DP4A phase loop and the whole activation chart are unchanged.
    /// * **plane B** — the high bit, 32 codes per u32 (`B[b·m + r]`), appended AFTER plane A in the
    ///   same `words` buffer at a compile-time offset, so no binding moves.
    ///
    /// `(k/16 + k/32)·m` words = **3.0 bits/weight**, plus the two-level scale face's 0.5 = 3.5.
    ///
    /// **Plane B is bit-PERMUTED at pack time**, which is the whole trick. A DP4A phase wants the
    /// high bits of codes `{4i + l}` — stride 4 — and the nibble-to-bytes expand
    /// (`x · 0x00204081 & 0x01010101`) only works on CONTIGUOUS bits; at stride 4 the partial
    /// products collide. So the packer stores the bit for code `j = 16·cj + 4·i + l` at position
    /// `16·cj + 4·l + i`, and the kernel reads four contiguous bits per phase. See
    /// [`q3_planes_gpu_chart`].
    Q3,
}

impl Carrier {
    /// Elements per u32 word (1 = not packed).
    pub const fn pack(self) -> u32 {
        match self {
            Carrier::F32 => 1,
            Carrier::F16 => 2,
            Carrier::Q8 => 4,
            Carrier::Q4 => 8,
            Carrier::Q2 => 16,
            // plane A's width — Q3 rides Q2's phase loop and adds the high-bit plane per phase
            Carrier::Q3 => 16,
        }
    }

    /// **Code words per ROW** — `k/pack` for a single-plane carrier, and plane A + plane B for
    /// [`Self::Q3`]. `pack()` alone cannot express a two-plane chart, and using it for byte
    /// accounting would undercount Q3 by a third; `chart_bytes_tl` and the packers go through here.
    pub const fn code_words_per_row(self, k: u32) -> u32 {
        match self {
            Carrier::Q3 => k / 16 + k / 32,
            _ => k / self.pack(),
        }
    }

    /// Element offset of plane B within the `words` buffer — `q3_planes_gpu_chart`'s layout, and
    /// the constant the kernel folds in. Zero for every single-plane carrier.
    pub const fn plane_b_off(self, m: u32, k: u32) -> u32 {
        match self {
            Carrier::Q3 => (k / 16) * m,
            _ => 0,
        }
    }
    /// Whether the fold runs in the (code, scale) chart (per-32-block d/m + XS).
    pub const fn is_chart(self) -> bool {
        matches!(self, Carrier::Q8 | Carrier::Q4 | Carrier::Q2 | Carrier::Q3)
    }
    /// The code mask for chart carriers.
    pub const fn qmask(self) -> u32 {
        match self {
            Carrier::Q8 => 255,
            Carrier::Q4 => 15,
            Carrier::Q2 => 3,
            Carrier::Q3 => 7,
            _ => 0,
        }
    }
    /// Stable label.
    pub const fn name(self) -> &'static str {
        match self {
            Carrier::F32 => "f32",
            Carrier::F16 => "f16",
            Carrier::Q8 => "q8",
            Carrier::Q4 => "q4",
            Carrier::Q2 => "q2",
            Carrier::Q3 => "q3",
        }
    }
    /// Weight bytes per element (params excluded).
    pub const fn bytes_per_elem_x4(self) -> u32 {
        // ×4 to stay integral: 16, 8, 4, 2, 1 quarter-bytes (F32…Q2)
        match self {
            Carrier::F32 => 16,
            Carrier::F16 => 8,
            Carrier::Q8 => 4,
            Carrier::Q4 => 2,
            Carrier::Q2 => 1,
            // 3 bits = 1.5 quarter-bytes; this table is integral, so Q3 is not expressible here
            // and callers must use `code_words_per_row`. Returning 2 (Q4's) would OVERstate.
            Carrier::Q3 => 2,
        }
    }
}
const OP_BITCAST: u32 = 124; //          OpBitcast (type, result, operand) — sint↔uint, no reinterpret cost
const BUILTIN_WORKGROUP_ID: u32 = 26; // BuiltIn.WorkgroupId — the WARP-UNIFORM key (see below)
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27; // BuiltIn.LocalInvocationId
const GLSL_UMIN: u32 = 38; //            GLSL.std.450 UMin — the mandala ring closure at non-2ⁿ modulus

/// **RMSNorm** — `out[i] = x[i] · rsqrt(mean(x²) + eps) · w[i]`, the transformer's per-token norm.
/// One invocation folds the whole n-vector (two passes: sum-of-squares, then the scale); tiny next
/// to the weight GEMVs at batch-1, correctness over speed. Bindings: 0 = x, 1 = w (gain), 2 = out,
/// all f32 runtime arrays; dispatch (1,1,1). Hand-emitted SPIR-V (no glslang, no CUDA) — the glue
/// kernels join the GEMV as our own Vulkan compute, vendor-neutral.
pub fn rmsnorm_module(n: u32, eps: f32) -> Vec<u32> {
    const GLSL_INVERSE_SQRT: u32 = 32; // GLSL.std.450 InverseSqrt (rsqrt) — no runtime divide
    const VOID: u32 = 1;
    const FNTY: u32 = 2;
    const UINT: u32 = 3;
    const FLOAT: u32 = 4;
    const BOOL: u32 = 5;
    const RTA_F: u32 = 9;
    const BUF_F: u32 = 10;
    const PTR_SB_BUF_F: u32 = 11;
    const X: u32 = 12;
    const W: u32 = 13;
    const OUT: u32 = 14;
    const PTR_SB_F: u32 = 15;
    const PTR_FN_F: u32 = 16;
    const PTR_FN_U: u32 = 17;
    const GLSL: u32 = 18;
    const C_N: u32 = 20;
    const C_U0: u32 = 21;
    const C_U1: u32 = 22;
    const C_INVN: u32 = 23;
    const C_EPS: u32 = 24;
    const C_F0: u32 = 25;
    const MAIN: u32 = 30;
    const ENTRY: u32 = 31;
    const ACCVAR: u32 = 32;
    const IVAR: u32 = 33;
    const L1HDR: u32 = 34;
    const L1CHK: u32 = 35;
    const L1BODY: u32 = 36;
    const L1CONT: u32 = 37;
    const L1MERGE: u32 = 38;
    const II: u32 = 39;
    const COND1: u32 = 40;
    const PX: u32 = 41;
    const XV: u32 = 42;
    const SQ: u32 = 43;
    const A0: u32 = 44;
    const A1: u32 = 45;
    const I2: u32 = 46;
    const IN: u32 = 47;
    const ACC: u32 = 48;
    const MEANSQ: u32 = 49;
    const MSEPS: u32 = 50;
    const INV: u32 = 51;
    const L2HDR: u32 = 52;
    const L2CHK: u32 = 53;
    const L2BODY: u32 = 54;
    const L2CONT: u32 = 55;
    const L2MERGE: u32 = 56;
    const JJ: u32 = 57;
    const COND2: u32 = 58;
    const PX2: u32 = 59;
    const XV2: u32 = 60;
    const PW: u32 = 61;
    const WV: u32 = 62;
    const TT: u32 = 63;
    const OO: u32 = 64;
    const PO: u32 = 65;
    const J2: u32 = 66;
    const JN: u32 = 67;
    const BOUND: u32 = 70;

    let mut w = vec![SPV_MAGIC, SPV_VERSION_1_3, 0, BOUND, 0];
    w.extend([(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
    let glsl = lit_string("GLSL.std.450");
    w.push(((2 + glsl.len() as u32) << 16) | OP_EXT_INST_IMPORT);
    w.push(GLSL);
    w.extend(&glsl);
    w.extend([(3 << 16) | OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450]);
    let main = lit_string("main");
    w.push(((3 + main.len() as u32) << 16) | OP_ENTRY_POINT);
    w.extend([MODEL_GL_COMPUTE, MAIN]);
    w.extend(&main);
    w.extend([
        (6 << 16) | OP_EXECUTION_MODE,
        MAIN,
        MODE_LOCAL_SIZE,
        1,
        1,
        1,
    ]);
    w.extend([(4 << 16) | OP_DECORATE, RTA_F, DEC_ARRAY_STRIDE, 4]);
    w.extend([(3 << 16) | OP_DECORATE, BUF_F, DEC_BLOCK]);
    w.extend([(5 << 16) | OP_MEMBER_DECORATE, BUF_F, 0, DEC_OFFSET, 0]);
    for (v, b) in [(X, 0u32), (W, 1), (OUT, 2)] {
        w.extend([(4 << 16) | OP_DECORATE, v, DEC_DESCRIPTOR_SET, 0]);
        w.extend([(4 << 16) | OP_DECORATE, v, DEC_BINDING, b]);
    }
    w.extend([(2 << 16) | OP_TYPE_VOID, VOID]);
    w.extend([(3 << 16) | OP_TYPE_FUNCTION, FNTY, VOID]);
    w.extend([(4 << 16) | OP_TYPE_INT, UINT, 32, 0]);
    w.extend([(3 << 16) | OP_TYPE_FLOAT, FLOAT, 32]);
    w.extend([(2 << 16) | OP_TYPE_BOOL, BOOL]);
    w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, RTA_F, FLOAT]);
    w.extend([(3 << 16) | OP_TYPE_STRUCT, BUF_F, RTA_F]);
    w.extend([
        (4 << 16) | OP_TYPE_POINTER,
        PTR_SB_BUF_F,
        SC_STORAGE_BUFFER,
        BUF_F,
    ]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, X, SC_STORAGE_BUFFER]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_SB_BUF_F, W, SC_STORAGE_BUFFER]);
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
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_F, SC_FUNCTION, FLOAT]);
    w.extend([(4 << 16) | OP_TYPE_POINTER, PTR_FN_U, SC_FUNCTION, UINT]);
    w.extend([(4 << 16) | OP_CONSTANT, UINT, C_N, n]);
    w.extend([(4 << 16) | OP_CONSTANT, UINT, C_U0, 0]);
    w.extend([(4 << 16) | OP_CONSTANT, UINT, C_U1, 1]);
    w.extend([
        (4 << 16) | OP_CONSTANT,
        FLOAT,
        C_INVN,
        (1.0f32 / n as f32).to_bits(),
    ]);
    w.extend([(4 << 16) | OP_CONSTANT, FLOAT, C_EPS, eps.to_bits()]);
    w.extend([(4 << 16) | OP_CONSTANT, FLOAT, C_F0, 0f32.to_bits()]);
    w.extend([(5 << 16) | OP_FUNCTION, VOID, MAIN, 0, FNTY]);
    w.extend([(2 << 16) | OP_LABEL, ENTRY]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_F, ACCVAR, SC_FUNCTION]);
    w.extend([(4 << 16) | OP_VARIABLE, PTR_FN_U, IVAR, SC_FUNCTION]);
    w.extend([(3 << 16) | OP_STORE, ACCVAR, C_F0]);
    w.extend([(3 << 16) | OP_STORE, IVAR, C_U0]);
    w.extend([(2 << 16) | OP_BRANCH, L1HDR]);
    w.extend([(2 << 16) | OP_LABEL, L1HDR]);
    w.extend([(4 << 16) | OP_LOOP_MERGE, L1MERGE, L1CONT, 0]);
    w.extend([(2 << 16) | OP_BRANCH, L1CHK]);
    w.extend([(2 << 16) | OP_LABEL, L1CHK]);
    w.extend([(4 << 16) | OP_LOAD, UINT, II, IVAR]);
    w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, COND1, II, C_N]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, COND1, L1BODY, L1MERGE]);
    w.extend([(2 << 16) | OP_LABEL, L1BODY]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PX, X, C_U0, II]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, XV, PX]);
    w.extend([(5 << 16) | OP_FMUL, FLOAT, SQ, XV, XV]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, A0, ACCVAR]);
    w.extend([(5 << 16) | OP_FADD, FLOAT, A1, A0, SQ]);
    w.extend([(3 << 16) | OP_STORE, ACCVAR, A1]);
    w.extend([(2 << 16) | OP_BRANCH, L1CONT]);
    w.extend([(2 << 16) | OP_LABEL, L1CONT]);
    w.extend([(4 << 16) | OP_LOAD, UINT, I2, IVAR]);
    w.extend([(5 << 16) | OP_IADD, UINT, IN, I2, C_U1]);
    w.extend([(3 << 16) | OP_STORE, IVAR, IN]);
    w.extend([(2 << 16) | OP_BRANCH, L1HDR]);
    w.extend([(2 << 16) | OP_LABEL, L1MERGE]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, ACC, ACCVAR]);
    w.extend([(5 << 16) | OP_FMUL, FLOAT, MEANSQ, ACC, C_INVN]);
    w.extend([(5 << 16) | OP_FADD, FLOAT, MSEPS, MEANSQ, C_EPS]);
    w.extend([
        (6 << 16) | OP_EXT_INST,
        FLOAT,
        INV,
        GLSL,
        GLSL_INVERSE_SQRT,
        MSEPS,
    ]);
    w.extend([(3 << 16) | OP_STORE, IVAR, C_U0]);
    w.extend([(2 << 16) | OP_BRANCH, L2HDR]);
    w.extend([(2 << 16) | OP_LABEL, L2HDR]);
    w.extend([(4 << 16) | OP_LOOP_MERGE, L2MERGE, L2CONT, 0]);
    w.extend([(2 << 16) | OP_BRANCH, L2CHK]);
    w.extend([(2 << 16) | OP_LABEL, L2CHK]);
    w.extend([(4 << 16) | OP_LOAD, UINT, JJ, IVAR]);
    w.extend([(5 << 16) | OP_ULESS_THAN, BOOL, COND2, JJ, C_N]);
    w.extend([(4 << 16) | OP_BRANCH_CONDITIONAL, COND2, L2BODY, L2MERGE]);
    w.extend([(2 << 16) | OP_LABEL, L2BODY]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PX2, X, C_U0, JJ]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, XV2, PX2]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PW, W, C_U0, JJ]);
    w.extend([(4 << 16) | OP_LOAD, FLOAT, WV, PW]);
    w.extend([(5 << 16) | OP_FMUL, FLOAT, TT, XV2, INV]);
    w.extend([(5 << 16) | OP_FMUL, FLOAT, OO, TT, WV]);
    w.extend([(6 << 16) | OP_ACCESS_CHAIN, PTR_SB_F, PO, OUT, C_U0, JJ]);
    w.extend([(3 << 16) | OP_STORE, PO, OO]);
    w.extend([(2 << 16) | OP_BRANCH, L2CONT]);
    w.extend([(2 << 16) | OP_LABEL, L2CONT]);
    w.extend([(4 << 16) | OP_LOAD, UINT, J2, IVAR]);
    w.extend([(5 << 16) | OP_IADD, UINT, JN, J2, C_U1]);
    w.extend([(3 << 16) | OP_STORE, IVAR, JN]);
    w.extend([(2 << 16) | OP_BRANCH, L2HDR]);
    w.extend([(2 << 16) | OP_LABEL, L2MERGE]);
    w.extend([(1 << 16) | OP_RETURN]);
    w.extend([(1 << 16) | OP_FUNCTION_END]);
    w
}

/// RMSNorm module as bytes.
pub fn rmsnorm_bytes(n: u32, eps: f32) -> Vec<u8> {
    rmsnorm_module(n, eps)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// The CPU reference for the GEMV fold: `y[r] = Σ_k W[k·M + r] · x[k]`, W column-major, same
/// accumulation order as the kernel (a plain fold over k, so fp32 results match bit-for-bit —
/// no reassociation on either side).
pub fn gemv_ref(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m)
        .map(|r| (0..k).fold(0f32, |acc, kk| acc + w[kk * m + r] * x[kk]))
        .collect()
}

/// **The reference the metal must match** (the `Prog.eval` / `whnfHeadTag` analog): each op's
/// meaning on two i32 words, as u32 bit patterns. Semantics follow the scalar encoders'
/// *instructions* (`Metal.Ins`: `xorRR` really XORs, `idiv` truncates toward zero, `setlt` is the
/// signed predicate), which on the select face is the wire form — not `Op.binOp`'s identity stub.
/// `AShift` masks the count to 0..31 (the test vectors stay in range; SPIR-V leaves ≥ width
/// undefined, x86 masks by 63 — the width dial, documented, not hidden).
pub fn op_ref(op: Op, a: u32, b: u32) -> u32 {
    let (ia, ib) = (a as i32, b as i32);
    match op {
        Op::TMax => ia.max(ib) as u32,
        Op::TMin => ia.min(ib) as u32,
        Op::TUnit => a,
        Op::TCmp => u32::from(ia < ib),
        Op::SXor => a ^ b,
        Op::SAnd => a & b,
        Op::SOr => a | b,
        Op::SNot => !a,
        Op::AAdd => a.wrapping_add(b),
        Op::AMul => a.wrapping_mul(b),
        Op::ADiv => ia.wrapping_div(ib) as u32,
        Op::AShift => a.wrapping_shl(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Twelve ops, 4/4/4 across the faces, three doorways — `ISA.lean isa_is_twelve`, mirrored.
    #[test]
    fn isa_is_twelve() {
        assert_eq!(Op::ALL.len(), 12);
        for f in [OpFace::Throne, OpFace::Select, OpFace::Accum] {
            assert_eq!(Op::ALL.iter().filter(|o| o.face() == f).count(), 4, "{f:?}");
        }
        assert_eq!(Op::ALL.iter().filter(|o| o.is_doorway()).count(), 3);
    }

    /// The `#guard` idiom: known words asserted byte-for-byte against the grammar.
    #[test]
    fn known_encodings() {
        // OpIAdd: word count 5, opcode 128; %20 = %16 + %18 over %4.
        assert_eq!(
            encode_op(Op::AAdd, ID_INT, ID_RES, ID_A, ID_B),
            vec![(5 << 16) | 128, 4, 20, 16, 18]
        );
        // The throne extremum is an OpExtInst (12) into GLSL.std.450: SMax = 42.
        assert_eq!(
            encode_op(Op::TMax, ID_INT, ID_RES, ID_A, ID_B),
            vec![(7 << 16) | 12, 4, 20, 1, 42, 16, 18]
        );
        // The compare doorway is the two-instruction setlt: OpSLessThan (177) + OpSelect (169).
        let tcmp = encode_op(Op::TCmp, ID_INT, ID_RES, ID_A, ID_B);
        assert_eq!(tcmp[0] & 0xFFFF, 177);
        assert_eq!(tcmp[5] & 0xFFFF, 169);
        // The deliberate no-op.
        assert!(encode_op(Op::TUnit, ID_INT, ID_RES, ID_A, ID_B).is_empty());
        // Every other op is non-empty on this substrate too (ISA.lean's `#guard` line).
        for op in Op::ALL {
            assert!(
                op == Op::TUnit || !encode_op(op, ID_INT, ID_RES, ID_A, ID_B).is_empty(),
                "{op:?}"
            );
        }
    }

    /// The module header is well-formed and word counts are self-consistent (each instruction's
    /// declared length walks exactly to the module's end — the two-pass assembler's invariant).
    #[test]
    fn module_walks() {
        for op in Op::ALL {
            let m = spv_module(op);
            assert_eq!(m[0], SPV_MAGIC);
            assert_eq!(m[1], SPV_VERSION_1_3);
            assert_eq!(m[3], ID_BOUND);
            let mut i = 5;
            while i < m.len() {
                let wc = (m[i] >> 16) as usize;
                assert!(wc >= 1, "{op:?}: zero word count at {i}");
                i += wc;
            }
            assert_eq!(i, m.len(), "{op:?}: instruction stream misaligned");
        }
    }

    /// Every carrier's module walks (word counts consistent) at plain and split-K shapes,
    /// and the reference computes the fold.
    #[test]
    fn gemv_module_walks() {
        for carrier in [Carrier::F32, Carrier::F16, Carrier::Q8, Carrier::Q4] {
            for (m, k, l, sg) in [(64u32, 64u32, 64u32, 1u32), (8192, 8192, 256, 8)] {
                let w = gemv_module(m, k, l, sg, carrier);
                assert_eq!(w[0], SPV_MAGIC);
                let mut i = 5;
                while i < w.len() {
                    let wc = (w[i] >> 16) as usize;
                    assert!(wc >= 1);
                    i += wc;
                }
                assert_eq!(i, w.len(), "gemv({m},{k},{l},{sg},{carrier:?}) misaligned");
            }
        }
        // y[r] = Σ_k W[k·M+r]·x[k]: W = [[1,2],[3,4]] column-major = [1,3,2,4], x = [10, 100]
        // → y = [1·10 + 2·100, 3·10 + 4·100] = [210, 430].
        assert_eq!(
            gemv_ref(&[1., 3., 2., 4.], &[10., 100.], 2, 2),
            vec![210., 430.]
        );
    }

    /// The integer-dot module walks (word counts consistent) at plain and split-K shapes;
    /// Q8 emits one OpSUDot per word, Q4 two (the nibble split).
    #[test]
    fn dp4a_module_walks() {
        for (carrier, dots) in [(Carrier::Q8, 1), (Carrier::Q4, 2), (Carrier::Q2, 4)] {
            for (m, k, l, sg) in [(64u32, 64u32, 64u32, 1u32), (8192, 8192, 256, 8)] {
                let w = gemv_dp4a_module(m, k, l, sg, carrier);
                assert_eq!(w[0], SPV_MAGIC);
                assert_eq!(
                    w.iter().filter(|&&x| x & 0xFFFF == OP_SUDOT).count(),
                    dots,
                    "{carrier:?}"
                );
                let mut i = 5;
                while i < w.len() {
                    let wc = (w[i] >> 16) as usize;
                    assert!(wc >= 1);
                    i += wc;
                }
                assert_eq!(i, w.len(), "dp4a({m},{k},{l},{sg},{carrier:?}) misaligned");
            }
        }
    }

    /// The reference at a few pinned points (signed semantics where the opcode is signed).
    #[test]
    fn reference_points() {
        assert_eq!(op_ref(Op::AAdd, 3, 4), 7);
        assert_eq!(op_ref(Op::ADiv, (-7i32) as u32, 2), (-3i32) as u32); // trunc toward zero
        assert_eq!(op_ref(Op::TCmp, (-1i32) as u32, 0), 1); // signed less-than
        assert_eq!(op_ref(Op::TMax, (-5i32) as u32, 3), 3);
        assert_eq!(op_ref(Op::SNot, 0, 0), u32::MAX);
        assert_eq!(op_ref(Op::AShift, 1, 5), 32);
    }
}
