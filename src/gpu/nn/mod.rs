// Fold `modelfold/src/simt_nn/` at `d497161f`, ported verbatim up to rustfmt.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    unused_variables,
    reason = "faithful port of fold's modelfold::simt_nn kernel builders; lint profile is fold's, \
              not apeiron's"
)]

//! simt_nn — the transformer forward's glue kernels on the SIMT substrate.
//!
//! [`crate::gpu::simt`] carries the alphabet (12 ops) and the one weight kernel (the GEMV fold, the
//! carrier the dial). A transformer decode step is that GEMV at every projection (QKV, O,
//! gate/up/down, lm_head, and attention's Q·Kᵀ / ·V — the same fold), plus a handful of small
//! glue maps: the norm (`x = B/U`), the rotor (RoPE), the gate (SiLU·up), the monad residual
//! (add), the boost (softmax), the KV append, and the embed gather. This module emits those as
//! hand-built SPIR-V 1.3 compute modules — same substrate, no glslang, no CUDA — each gated by
//! `spirv-val` (tests/simt_nn.rs) and by a metal check against a CPU reference
//! (`examples/simt_nn_check.rs`). The op semantics mirror sanguine
//! `proof/Runtime/SeedForward.lean` (`rmsNormB` / `ropeB` / `attnHead` / `swigluB` / `addB`),
//! the cell faces the forward is made of.
//!
//! Dynamic state (the token position / token id) lives in a tiny `params` SSBO — `params[0] =
//! pos`, `params[1] = token id` — so the whole token step records ONCE and replays per token
//! (the STREAM-spine collapse, `FusionCost.fold_is_one_kernel`: one submit per token, the loop
//! bounds read `pos` on the device). Every opcode below is cited from the local Khronos grammars
//! (`spirv.core.grammar.json`, `extinst.glsl.std.450.grammar.json`), never from memory.
//!
//! **Builder families — the GPU primitive surface** (dispatched by [`crate::gpu::Gpu`]).
//! GEMM+norm fusion: [`gemv_norm_bytes`] (rmsnorm co-reduced into the gemv, carrier the dial —
//! F16 direct or chart DP4A in one kernel).
//! Reduction: [`rmsnorm_fused_bytes`], [`argmax_partial_bytes`] / [`argmax_finish_bytes`].
//! Elementwise: [`silu_mul_bytes`], [`add_inplace_bytes`], [`qk_norm_bytes`]. Sparse-attention
//! (the knot): [`theta_scan_bytes`] / [`theta_select_bytes`], [`flash_partial_bytes`] /
//! [`flash_combine_bytes`]. Model glue (compositions/consumers of the above): [`embed_f16_bytes`]
//! / [`embed_chart_bytes`], [`fused_rope_kv_bytes`], [`commit_advance_bytes`]. All `#[cfg]`-free
//! (always built); only the runtime that dispatches them is behind the `gpu` feature.

// ---------------------------------------------------------------------------------------------
// The public surface, unchanged by the split of this file into `simt_nn/`.
//
// Every path a caller could name before the split -- `simt_nn::gemv_norm_bytes`,
// `::flash_partial_bytes`, `::attn_prefill_tiled_bytes`, ... -- still resolves here. The `Kb`
// builder and the opcode/enumerant table stay in this file because every kernel below is written
// against them; `use super::*;` in each submodule reproduces exactly the name set the single file
// had, private items included.
// ---------------------------------------------------------------------------------------------

mod attn;
mod batch;
mod coop;
mod coopmat_gemm;
mod deltanet;
mod glue;
mod gridsync;
mod moe;
mod norm;
mod spec;
#[cfg(test)]
mod tests;

pub use attn::*;
pub use batch::*;
pub use coop::*;
pub use coopmat_gemm::*;
pub use deltanet::*;
pub use glue::*;
pub use gridsync::*;
pub use moe::*;
pub use norm::*;
pub use spec::*;

use crate::gpu::simt::{SPV_MAGIC, SPV_VERSION_1_3};

// ── core opcodes (spirv.core.grammar.json) ──────────────────────────────────────────────────────
const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;
const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_CAPABILITY: u32 = 17;
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_TYPE_FUNCTION: u32 = 33;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_VECTOR_EXTRACT_DYNAMIC: u32 = 77;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_CONVERT_U_TO_F: u32 = 112;
const OP_IADD: u32 = 128;
const OP_FADD: u32 = 129;
const OP_ISUB: u32 = 130;
const OP_FSUB: u32 = 131;
const OP_IMUL: u32 = 132;
const OP_FMUL: u32 = 133;
const OP_UDIV: u32 = 134;
const OP_FDIV: u32 = 136;
const OP_UMOD: u32 = 137;
const OP_SELECT: u32 = 169;
const OP_ULESS_THAN: u32 = 176;
const OP_FORD_GREATER_THAN: u32 = 186;
const OP_FORD_EQUAL: u32 = 180;
const OP_FORD_LESS_THAN: u32 = 184;
const OP_LOGICAL_AND: u32 = 167;
const OP_SHIFT_LEFT_LOGICAL: u32 = 196;
const OP_BITWISE_XOR: u32 = 198;
const OP_BITWISE_AND: u32 = 199;
const OP_LOOP_MERGE: u32 = 246;
const OP_SELECTION_MERGE: u32 = 247;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_RETURN: u32 = 253;
const OP_FCONVERT: u32 = 115;
const CAP_STORAGE_BUFFER_16BIT_ACCESS: u32 = 4433;

// enums (spirv.core.grammar.json operand_kinds)
const CAP_SHADER: u32 = 1;
const MODEL_GL_COMPUTE: u32 = 5;
const ADDR_LOGICAL: u32 = 0;
const MEM_GLSL450: u32 = 1;
const MODE_LOCAL_SIZE: u32 = 17;
const SC_STORAGE_BUFFER: u32 = 12;
const SC_INPUT: u32 = 1;
const SC_FUNCTION: u32 = 7;
const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILT_IN: u32 = 11;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// GLSL.std.450 (extinst.glsl.std.450.grammar.json)
const GLSL_ROUND: u32 = 1; // GLSL.std.450 Round — for RoPE angle range reduction
const GLSL_SIN: u32 = 13;
const GLSL_COS: u32 = 14;
const GLSL_EXP: u32 = 27;
const GLSL_LOG: u32 = 28; // natural log (GLSL.std.450) — the heavy-cut score threshold
const GLSL_SQRT: u32 = 31;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_FMAX: u32 = 40;
// GLSL.std.450 enumerant 50 `Fma` (enumerant 46 is `FMix`; the two differ by 4 — `spirv-dis`
// named the mistaken 46 to catch it).
const GLSL_FMA: u32 = 50;
const GLSL_UNPACK_HALF2X16: u32 = 62;

/// A NUL-terminated LE-packed literal string (the SPIR-V string encoding).
fn lit_string(s: &str) -> Vec<u32> {
    let sb = s.as_bytes();
    let mut bytes: Vec<u8> = Vec::new();
    let mut j = 0;
    while j < sb.len() {
        bytes.push(sb[j]);
        j += 1;
    }
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let mut out: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let w = (bytes[i] as u32)
            | ((bytes[i + 1] as u32) << 8)
            | ((bytes[i + 2] as u32) << 16)
            | ((bytes[i + 3] as u32) << 24);
        out.push(w);
        i += 4;
    }
    out
}

/// The kernel builder: an id allocator over the module's fixed sections. The gemv encoder pins
/// its ids by hand (it mirrors song's ISA tables); a dozen glue kernels would breed id bugs that
/// way, so these allocate — the assembled module is gated by `module walks` + spirv-val + the
/// metal check, same as the hand-pinned ones.
pub(crate) struct Kb {
    deco: Vec<u32>,
    tycon: Vec<u32>, // types, constants, global variables
    fvars: Vec<u32>, // function-scope OpVariables (spliced at the entry block's head)
    body: Vec<u32>,
    next: u32,
    glsl: u32, // OpExtInstImport id (always emitted; unused imports are legal)
    // the fixed types
    void_t: u32,
    fnty: u32,
    uint: u32,
    float: u32,
    float16: u32,
    ptr_sb_buf_f16: u32,
    ptr_sb_f16: u32,
    f16_storage: bool,
    dp4a: bool,    // uses OpSDot → emit the integer-dot capability + extension
    sint0: u32,    // signed-i32 constant 0 (the DP4A accumulator seed), lazily made
    ptr_fn_s: u32, // function pointer to sint (the DP4A accumulator var), lazily made
    bool_t: u32,
    uvec3: u32,
    vec2f: u32,
    ptr_sb_f: u32,
    ptr_sb_u: u32,
    ptr_fn_f: u32,
    ptr_fn_u: u32,
    ptr_sb_buf_f: u32,
    ptr_sb_buf_u: u32,
    gid: u32,
    main_id: u32,
    ucache: Vec<(u32, u32)>,
    fcache: Vec<(u32, u32)>,
    lii: u32,       // LocalInvocationIndex builtin var, lazily made (0 = unused)
    wgid: u32,      // WorkgroupId builtin var, lazily made (0 = unused)
    sg_arith: bool, // uses OpGroupNonUniformFAdd -> emit the subgroup-arithmetic capabilities
}

impl Kb {
    pub(crate) fn new() -> Kb {
        let mut k = Kb {
            deco: vec![],
            tycon: vec![],
            fvars: vec![],
            body: vec![],
            next: 1,
            glsl: 0,
            void_t: 0,
            fnty: 0,
            uint: 0,
            float: 0,
            float16: 0,
            ptr_sb_buf_f16: 0,
            ptr_sb_f16: 0,
            f16_storage: false,
            dp4a: false,
            sint0: 0,
            ptr_fn_s: 0,
            bool_t: 0,
            uvec3: 0,
            vec2f: 0,
            ptr_sb_f: 0,
            ptr_sb_u: 0,
            ptr_fn_f: 0,
            ptr_fn_u: 0,
            ptr_sb_buf_f: 0,
            ptr_sb_buf_u: 0,
            gid: 0,
            main_id: 0,
            ucache: vec![],
            fcache: vec![],
            lii: 0,
            wgid: 0,
            sg_arith: false,
        };
        k.glsl = k.id();
        k.main_id = k.id();
        k.void_t = k.id();
        k.fnty = k.id();
        k.uint = k.id();
        k.float = k.id();
        k.bool_t = k.id();
        k.uvec3 = k.id();
        k.vec2f = k.id();
        let (void_t, fnty, uint, float, bool_t, uvec3, vec2f) = (
            k.void_t, k.fnty, k.uint, k.float, k.bool_t, k.uvec3, k.vec2f,
        );
        k.ty(OP_TYPE_VOID, &[void_t]);
        k.ty(OP_TYPE_FUNCTION, &[fnty, void_t]);
        k.ty(OP_TYPE_INT, &[uint, 32, 0]);
        k.ty(OP_TYPE_FLOAT, &[float, 32]);
        k.ty(OP_TYPE_BOOL, &[bool_t]);
        k.ty(OP_TYPE_VECTOR, &[uvec3, uint, 3]);
        k.ty(OP_TYPE_VECTOR, &[vec2f, float, 2]);
        let ptr_in_uvec3 = k.id();
        k.ty(OP_TYPE_POINTER, &[ptr_in_uvec3, SC_INPUT, uvec3]);
        k.gid = k.id();
        let gid = k.gid;
        k.ty(OP_VARIABLE, &[ptr_in_uvec3, gid, SC_INPUT]);
        k.deco(&[OP_DECORATE, gid, DEC_BUILT_IN, BUILTIN_GLOBAL_INVOCATION_ID]);
        k.ptr_sb_f = k.id();
        k.ptr_sb_u = k.id();
        k.ptr_fn_f = k.id();
        k.ptr_fn_u = k.id();
        let (psf, psu, pff, pfu) = (k.ptr_sb_f, k.ptr_sb_u, k.ptr_fn_f, k.ptr_fn_u);
        k.ty(OP_TYPE_POINTER, &[psf, SC_STORAGE_BUFFER, float]);
        k.ty(OP_TYPE_POINTER, &[psu, SC_STORAGE_BUFFER, uint]);
        k.ty(OP_TYPE_POINTER, &[pff, SC_FUNCTION, float]);
        k.ty(OP_TYPE_POINTER, &[pfu, SC_FUNCTION, uint]);
        // the two SSBO block shapes: { float[] } and { uint[] }
        let (rta_f, buf_f, pbf) = (k.id(), k.id(), k.id());
        k.ty(OP_TYPE_RUNTIME_ARRAY, &[rta_f, float]);
        k.ty(OP_TYPE_STRUCT, &[buf_f, rta_f]);
        k.ty(OP_TYPE_POINTER, &[pbf, SC_STORAGE_BUFFER, buf_f]);
        k.deco(&[OP_DECORATE, rta_f, DEC_ARRAY_STRIDE, 4]);
        k.deco(&[OP_DECORATE, buf_f, DEC_BLOCK]);
        k.deco(&[OP_MEMBER_DECORATE, buf_f, 0, DEC_OFFSET, 0]);
        let (rta_u, buf_u, pbu) = (k.id(), k.id(), k.id());
        k.ty(OP_TYPE_RUNTIME_ARRAY, &[rta_u, uint]);
        k.ty(OP_TYPE_STRUCT, &[buf_u, rta_u]);
        k.ty(OP_TYPE_POINTER, &[pbu, SC_STORAGE_BUFFER, buf_u]);
        k.deco(&[OP_DECORATE, rta_u, DEC_ARRAY_STRIDE, 4]);
        k.deco(&[OP_DECORATE, buf_u, DEC_BLOCK]);
        k.deco(&[OP_MEMBER_DECORATE, buf_u, 0, DEC_OFFSET, 0]);
        k.ptr_sb_buf_f = pbf;
        k.ptr_sb_buf_u = pbu;
        k
    }
    /// Lazily declare the f16 storage-buffer types (only kernels that use the KV cache get the
    /// `OpTypeFloat 16` + `StorageBuffer16BitAccess` — else spirv-val rejects the unused f16 type).
    pub(crate) fn ensure_f16(&mut self) {
        if self.f16_storage {
            return;
        }
        self.f16_storage = true;
        self.float16 = self.id();
        let f16 = self.float16;
        self.ty(OP_TYPE_FLOAT, &[f16, 16]);
        self.ptr_sb_f16 = self.id();
        let psf16 = self.ptr_sb_f16;
        self.ty(OP_TYPE_POINTER, &[psf16, SC_STORAGE_BUFFER, f16]);
        let (rta_f16, buf_f16, pbf16) = (self.id(), self.id(), self.id());
        self.ty(OP_TYPE_RUNTIME_ARRAY, &[rta_f16, f16]);
        self.ty(OP_TYPE_STRUCT, &[buf_f16, rta_f16]);
        self.ty(OP_TYPE_POINTER, &[pbf16, SC_STORAGE_BUFFER, buf_f16]);
        self.deco(&[OP_DECORATE, rta_f16, DEC_ARRAY_STRIDE, 2]);
        self.deco(&[OP_DECORATE, buf_f16, DEC_BLOCK]);
        self.deco(&[OP_MEMBER_DECORATE, buf_f16, 0, DEC_OFFSET, 0]);
        self.ptr_sb_buf_f16 = pbf16;
    }
    /// A binding to an f16 storage buffer (the KV cache) → `StorageBuffer16BitAccess`.
    pub(crate) fn ssbo_f16(&mut self, binding: u32) -> u32 {
        self.ensure_f16();
        let pbf16 = self.ptr_sb_buf_f16;
        let id = self.id();
        self.ty(OP_VARIABLE, &[pbf16, id, SC_STORAGE_BUFFER]);
        self.deco(&[OP_DECORATE, id, DEC_DESCRIPTOR_SET, 0]);
        self.deco(&[OP_DECORATE, id, DEC_BINDING, binding]);
        id
    }
    /// Load `buf[idx]` as f16, converted up to f32 for arithmetic.
    pub(crate) fn load_f16(&mut self, buf: u32, idx: u32) -> u32 {
        let (psf16, f16, f) = (self.ptr_sb_f16, self.float16, self.float);
        let c0 = self.cu(0);
        let p = self.e(OP_ACCESS_CHAIN, psf16, &[buf, c0, idx]);
        let h = self.e(OP_LOAD, f16, &[p]);
        self.e(OP_FCONVERT, f, &[h])
    }
    /// Store f32 `val` as f16 into `buf[idx]`.
    pub(crate) fn store_f16(&mut self, buf: u32, idx: u32, val: u32) {
        let (psf16, f16) = (self.ptr_sb_f16, self.float16);
        let c0 = self.cu(0);
        let p = self.e(OP_ACCESS_CHAIN, psf16, &[buf, c0, idx]);
        let h = self.e(OP_FCONVERT, f16, &[val]);
        self.ins(OP_STORE, &[p, h]);
    }

    fn id(&mut self) -> u32 {
        let i = self.next;
        self.next += 1;
        i
    }
    fn deco(&mut self, words: &[u32]) {
        // `words[0]` is the opcode — the count is the whole slice's length
        self.deco.push(((words.len() as u32) << 16) | words[0]);
        push_all(&mut self.deco, &words[1..]);
    }
    fn ty(&mut self, opcode: u32, operands: &[u32]) {
        self.tycon
            .push(((operands.len() as u32 + 1) << 16) | opcode);
        push_all(&mut self.tycon, operands);
    }
    fn ins(&mut self, opcode: u32, operands: &[u32]) {
        self.body.push(((operands.len() as u32 + 1) << 16) | opcode);
        push_all(&mut self.body, operands);
    }
    /// `res = op ty a b …` into the body; returns `res`.
    fn e(&mut self, opcode: u32, ty: u32, args: &[u32]) -> u32 {
        let res = self.id();
        let mut ops = vec![ty, res];
        push_all(&mut ops, args);
        self.ins(opcode, &ops);
        res
    }
    fn cu(&mut self, v: u32) -> u32 {
        let mut i = 0;
        let mut hit: Option<u32> = None;
        while i < self.ucache.len() {
            let (w, id) = self.ucache[i];
            if w == v {
                hit = Some(id);
            }
            i += 1;
        }
        match hit {
            Some(id) => id,
            None => {
                let id = self.id();
                let uint = self.uint;
                self.ty(OP_CONSTANT, &[uint, id, v]);
                self.ucache.push((v, id));
                id
            }
        }
    }
    fn cf(&mut self, v: f32) -> u32 {
        let bits = v.to_bits();
        self.cf_bits(bits)
    }
    /// [`Kb::cf`] with the IEEE-754 bit pattern already formed. A literal `f32` argument reaches
    /// `to_bits` as a constant in the Aeneas interpreter, which refuses a constant float there; the
    /// bit pattern keeps the emitted `OpConstant` word identical.
    fn cf_bits(&mut self, bits: u32) -> u32 {
        let mut i = 0;
        let mut hit: Option<u32> = None;
        while i < self.fcache.len() {
            let (w, id) = self.fcache[i];
            if w == bits {
                hit = Some(id);
            }
            i += 1;
        }
        match hit {
            Some(id) => id,
            None => {
                let id = self.id();
                let float = self.float;
                self.ty(OP_CONSTANT, &[float, id, bits]);
                self.fcache.push((bits, id));
                id
            }
        }
    }
    /// An SSBO of f32 at (set 0, `binding`); returns the variable id.
    fn ssbo_f(&mut self, binding: u32) -> u32 {
        let var = self.id();
        let pbf = self.ptr_sb_buf_f;
        self.ty(OP_VARIABLE, &[pbf, var, SC_STORAGE_BUFFER]);
        self.deco(&[OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        self.deco(&[OP_DECORATE, var, DEC_BINDING, binding]);
        var
    }
    /// An SSBO of u32 at (set 0, `binding`).
    fn ssbo_u(&mut self, binding: u32) -> u32 {
        let var = self.id();
        let pbu = self.ptr_sb_buf_u;
        self.ty(OP_VARIABLE, &[pbu, var, SC_STORAGE_BUFFER]);
        self.deco(&[OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0]);
        self.deco(&[OP_DECORATE, var, DEC_BINDING, binding]);
        var
    }
    pub(crate) fn fvar(&mut self, ptr_ty: u32) -> u32 {
        let v = self.id();
        self.fvars.push((4 << 16) | OP_VARIABLE);
        push_all(&mut self.fvars, &[ptr_ty, v, SC_FUNCTION]);
        v
    }
    pub(crate) fn fvar_f(&mut self) -> u32 {
        let t = self.ptr_fn_f;
        self.fvar(t)
    }
    pub(crate) fn fvar_u(&mut self) -> u32 {
        let t = self.ptr_fn_u;
        self.fvar(t)
    }
    pub(crate) fn load_var_f(&mut self, var: u32) -> u32 {
        let f = self.float;
        self.e(OP_LOAD, f, &[var])
    }
    pub(crate) fn load_var_u(&mut self, var: u32) -> u32 {
        let u = self.uint;
        self.e(OP_LOAD, u, &[var])
    }
    pub(crate) fn store_var(&mut self, var: u32, val: u32) {
        self.ins(OP_STORE, &[var, val]);
    }
    /// `buf[idx]` load, f32 SSBO.
    fn load_f(&mut self, buf: u32, idx: u32) -> u32 {
        let (psf, f) = (self.ptr_sb_f, self.float);
        let c0 = self.cu(0);
        let p = self.e(OP_ACCESS_CHAIN, psf, &[buf, c0, idx]);
        self.e(OP_LOAD, f, &[p])
    }
    fn store_f(&mut self, buf: u32, idx: u32, val: u32) {
        let psf = self.ptr_sb_f;
        let c0 = self.cu(0);
        let p = self.e(OP_ACCESS_CHAIN, psf, &[buf, c0, idx]);
        self.ins(OP_STORE, &[p, val]);
    }
    fn load_u(&mut self, buf: u32, idx: u32) -> u32 {
        let (psu, u) = (self.ptr_sb_u, self.uint);
        let c0 = self.cu(0);
        let p = self.e(OP_ACCESS_CHAIN, psu, &[buf, c0, idx]);
        self.e(OP_LOAD, u, &[p])
    }
    /// GlobalInvocationId.(x|y).
    fn gid_comp(&mut self, comp: u32) -> u32 {
        let (uv3, u, gid) = (self.uvec3, self.uint, self.gid);
        let v = self.e(OP_LOAD, uv3, &[gid]);
        self.e(OP_COMPOSITE_EXTRACT, u, &[v, comp])
    }
    // arithmetic sugar
    fn iadd(&mut self, a: u32, b: u32) -> u32 {
        let u = self.uint;
        self.e(OP_IADD, u, &[a, b])
    }
    fn imul(&mut self, a: u32, b: u32) -> u32 {
        let u = self.uint;
        self.e(OP_IMUL, u, &[a, b])
    }
    fn udiv(&mut self, a: u32, b: u32) -> u32 {
        let u = self.uint;
        self.e(OP_UDIV, u, &[a, b])
    }
    fn umod(&mut self, a: u32, b: u32) -> u32 {
        let u = self.uint;
        self.e(OP_UMOD, u, &[a, b])
    }
    /// **The mandala closure — `key &&& (2^k − 1)`, one instruction.** `MandalaMemory.access` is
    /// `base + (key &&& (2^kbits − 1))`, and `masked_load_never_oob` is the statement that a load
    /// whose address was formed this way has no out-of-bounds representing input: the ring has no
    /// edge to walk off, so the vulnerability class is EMPTY rather than caught. `mask` must be
    /// `2^k − 1` for a region of `2^k` elements — the caller sizes the buffer to the power of two so
    /// the closure costs exactly the `andRR` the ISA already emits (song `emitMaskHeap`).
    pub(crate) fn mandala(&mut self, idx: u32, mask: u32) -> u32 {
        let u = self.uint;
        let m = self.cu(mask);
        self.e(OP_BITWISE_AND, u, &[idx, m])
    }
    fn fadd(&mut self, a: u32, b: u32) -> u32 {
        let f = self.float;
        self.e(OP_FADD, f, &[a, b])
    }
    pub(crate) fn fsub(&mut self, a: u32, b: u32) -> u32 {
        let f = self.float;
        self.e(OP_FSUB, f, &[a, b])
    }
    fn fmul(&mut self, a: u32, b: u32) -> u32 {
        let f = self.float;
        self.e(OP_FMUL, f, &[a, b])
    }
    /// `fma(a, b, c) = a·b + c` with a single rounding (`GLSL.std.450 Fma`, enumerant 50). Emitted
    /// where the driver would otherwise contract the multiply into the add on one arm but not
    /// another; the explicit form pins the f32 value on both.
    pub(crate) fn fma(&mut self, a: u32, b: u32, c: u32) -> u32 {
        let (f, g) = (self.float, self.glsl);
        self.e(OP_EXT_INST, f, &[g, GLSL_FMA, a, b, c])
    }
    fn fdiv(&mut self, a: u32, b: u32) -> u32 {
        let f = self.float;
        self.e(OP_FDIV, f, &[a, b])
    }
    fn ult(&mut self, a: u32, b: u32) -> u32 {
        let b_t = self.bool_t;
        self.e(OP_ULESS_THAN, b_t, &[a, b])
    }
    pub(crate) fn utof(&mut self, a: u32) -> u32 {
        let f = self.float;
        self.e(OP_CONVERT_U_TO_F, f, &[a])
    }
    /// One-operand GLSL.std.450 call.
    fn ext1(&mut self, inst: u32, x: u32) -> u32 {
        let (f, g) = (self.float, self.glsl);
        self.e(OP_EXT_INST, f, &[g, inst, x])
    }
    pub(crate) fn ext2(&mut self, inst: u32, x: u32, y: u32) -> u32 {
        let (f, g) = (self.float, self.glsl);
        self.e(OP_EXT_INST, f, &[g, inst, x, y])
    }
    /// `min(a, b)` on **uint** — the ring closure for a fold whose last petal is partial. Unlike
    /// [`Kb::ext2`] the result type is the integer, so it can bound a loop counter.
    pub(crate) fn umin(&mut self, a: u32, b: u32) -> u32 {
        let (u, g) = (self.uint, self.glsl);
        self.e(OP_EXT_INST, u, &[g, GLSL_UMIN, a, b])
    }
    /// `if cond { then(self) }` — a selection construct. The closure is threaded **by value**:
    /// a closure typed `FnOnce(&mut Kb)` extracts as `FnOnce(Kb) -> Unit` while its generated body
    /// returns the mutated `Kb`, and Aeneas emits the call site expecting `Unit × Kb`; the three
    /// disagree. `FnOnce(Kb) -> Kb` has one type on all three sides. The receiver stays `&mut self`,
    /// so `core.mem.replace` (modelled as `(dst, src)`) moves the builder out and the closure's
    /// result is moved back in; no other builder helper changes shape.
    fn iff(&mut self, cond: u32, then: impl FnOnce(Kb) -> Kb) {
        let (then_l, end_l) = (self.id(), self.id());
        self.ins(OP_SELECTION_MERGE, &[end_l, 0]);
        self.ins(OP_BRANCH_CONDITIONAL, &[cond, then_l, end_l]);
        self.ins(OP_LABEL, &[then_l]);
        let k = core::mem::replace(self, Kb::new());
        *self = then(k);
        self.ins(OP_BRANCH, &[end_l]);
        self.ins(OP_LABEL, &[end_l]);
    }
    /// `for c in start..bound { body(self, c) }` — a structured loop; `c` is the loaded counter.
    /// Closure and receiver threaded as in [`Kb::iff`].
    pub(crate) fn loop_u(&mut self, start: u32, bound: u32, body: impl FnOnce(Kb, u32) -> Kb) {
        let ctr = self.fvar_u();
        self.store_var(ctr, start);
        let (hdr, chk, bod, cont, merge) = (self.id(), self.id(), self.id(), self.id(), self.id());
        self.ins(OP_BRANCH, &[hdr]);
        self.ins(OP_LABEL, &[hdr]);
        self.ins(OP_LOOP_MERGE, &[merge, cont, 0]);
        self.ins(OP_BRANCH, &[chk]);
        self.ins(OP_LABEL, &[chk]);
        let c = self.load_var_u(ctr);
        let cond = self.ult(c, bound);
        self.ins(OP_BRANCH_CONDITIONAL, &[cond, bod, merge]);
        self.ins(OP_LABEL, &[bod]);
        let k = core::mem::replace(self, Kb::new());
        *self = body(k, c);
        self.ins(OP_BRANCH, &[cont]);
        self.ins(OP_LABEL, &[cont]);
        let c2 = self.load_var_u(ctr);
        let one = self.cu(1);
        let cn = self.iadd(c2, one);
        self.store_var(ctr, cn);
        self.ins(OP_BRANCH, &[hdr]);
        self.ins(OP_LABEL, &[merge]);
    }
    /// **`dst[i] = Σ_{s<segs} src[s·n + i] (+ bias[i])`** — the split-K stage-2 reduce as an inline
    /// prologue instead of its own dispatch. The summation walks `s` ascending, which is exactly the
    /// order the `gemv_bytes(m, segs, …, F32)` stage-2 gemv folds in (`Σ_j partials[j·m+r]·ones[j]`,
    /// `ones ≡ 1`), so the f32 association is unchanged and the result is bit-identical.
    pub(crate) fn fold_bias_store(
        &mut self,
        src: u32,
        bias: Option<u32>,
        dst: u32,
        i: u32,
        n: u32,
        segs: u32,
    ) {
        let v = if segs <= 1 {
            self.load_f(src, i)
        } else {
            let acc = self.fvar_f();
            let z = self.cf_bits(0x0000_0000);
            self.store_var(acc, z);
            let c0 = self.cu(0);
            let cseg = self.cu(segs);
            self.loop_u(c0, cseg, |mut k, s| {
                let cn = k.cu(n);
                let b = k.imul(s, cn);
                let idx = k.iadd(b, i);
                let p = k.load_f(src, idx);
                let a0 = k.load_var_f(acc);
                let a1 = k.fadd(a0, p);
                k.store_var(acc, a1);
                k
            });
            self.load_var_f(acc)
        };
        let out = match bias {
            Some(b) => {
                let bv = self.load_f(b, i);
                self.fadd(v, bv)
            }
            None => v,
        };
        self.store_f(dst, i, out);
    }
    /// `for c in (start..bound).step_by(step) { body(self, c) }` — the workgroup-strided loop
    /// (each lane walks its own slice; `step` is the workgroup width). `step` is a constant id.
    fn loop_step(&mut self, start: u32, bound: u32, step: u32, body: impl FnOnce(&mut Kb, u32)) {
        let ctr = self.fvar_u();
        self.store_var(ctr, start);
        let (hdr, chk, bod, cont, merge) = (self.id(), self.id(), self.id(), self.id(), self.id());
        self.ins(OP_BRANCH, &[hdr]);
        self.ins(OP_LABEL, &[hdr]);
        self.ins(OP_LOOP_MERGE, &[merge, cont, 0]);
        self.ins(OP_BRANCH, &[chk]);
        self.ins(OP_LABEL, &[chk]);
        let c = self.load_var_u(ctr);
        let cond = self.ult(c, bound);
        self.ins(OP_BRANCH_CONDITIONAL, &[cond, bod, merge]);
        self.ins(OP_LABEL, &[bod]);
        body(self, c);
        self.ins(OP_BRANCH, &[cont]);
        self.ins(OP_LABEL, &[cont]);
        let c2 = self.load_var_u(ctr);
        let cn = self.iadd(c2, step);
        self.store_var(ctr, cn);
        self.ins(OP_BRANCH, &[hdr]);
        self.ins(OP_LABEL, &[merge]);
    }
    /// Assemble the module: header, capability, GLSL import, entry `main` at `LocalSize local·1·1`.
    fn finish(mut self, local: u32) -> Vec<u32> {
        let entry_l = self.id();
        let mut w = vec![SPV_MAGIC, SPV_VERSION_1_3, 0, self.next, 0];
        // SPIR-V logical layout: ALL OpCapability first, THEN ALL OpExtension.
        push_all(&mut w, &[(2 << 16) | OP_CAPABILITY, CAP_SHADER]);
        if self.f16_storage {
            push_all(
                &mut w,
                &[(2 << 16) | OP_CAPABILITY, CAP_STORAGE_BUFFER_16BIT_ACCESS],
            );
        }
        if self.sg_arith {
            push_all(&mut w, &[(2 << 16) | OP_CAPABILITY, CAP_GROUP_NON_UNIFORM]);
            push_all(
                &mut w,
                &[(2 << 16) | OP_CAPABILITY, CAP_GROUP_NON_UNIFORM_ARITHMETIC],
            );
        }
        if self.dp4a {
            push_all(&mut w, &[(2 << 16) | OP_CAPABILITY, CAP_DOT_PRODUCT]);
            push_all(
                &mut w,
                &[(2 << 16) | OP_CAPABILITY, CAP_DOT_INPUT_4X8_PACKED],
            );
        }
        // OpExtension follows ALL capabilities, precedes ext-inst imports (logical layout).
        if self.dp4a {
            let ext = lit_string("SPV_KHR_integer_dot_product");
            w.push(((1 + ext.len() as u32) << 16) | OP_EXTENSION);
            push_all(&mut w, &ext);
        }
        let g = lit_string("GLSL.std.450");
        w.push(((2 + g.len() as u32) << 16) | OP_EXT_INST_IMPORT);
        w.push(self.glsl);
        push_all(&mut w, &g);
        let mem_model = MEM_GLSL450;
        push_all(
            &mut w,
            &[(3 << 16) | OP_MEMORY_MODEL, ADDR_LOGICAL, mem_model],
        );
        // the entry-point interface: every Input/Output global the entry uses (SPIR-V ≤1.3).
        let mut iface = vec![self.gid];
        if self.lii != 0 {
            iface.push(self.lii);
        }
        if self.wgid != 0 {
            iface.push(self.wgid);
        }
        let main = lit_string("main");
        w.push(((3 + main.len() as u32 + iface.len() as u32) << 16) | OP_ENTRY_POINT);
        push_all(&mut w, &[MODEL_GL_COMPUTE, self.main_id]);
        push_all(&mut w, &main);
        push_all(&mut w, &iface);
        push_all(
            &mut w,
            &[
                (6 << 16) | OP_EXECUTION_MODE,
                self.main_id,
                MODE_LOCAL_SIZE,
                local,
                1,
                1,
            ],
        );
        push_all(&mut w, &self.deco);
        push_all(&mut w, &self.tycon);
        push_all(
            &mut w,
            &[
                (5 << 16) | OP_FUNCTION,
                self.void_t,
                self.main_id,
                0,
                self.fnty,
            ],
        );
        push_all(&mut w, &[(2 << 16) | OP_LABEL, entry_l]);
        push_all(&mut w, &self.fvars);
        push_all(&mut w, &self.body);
        push_all(&mut w, &[(1 << 16) | OP_RETURN]);
        push_all(&mut w, &[(1 << 16) | OP_FUNCTION_END]);
        w
    }
}

/// Append every word of `xs`. The Aeneas interpreter refuses `Vec::extend` here (it is an
/// unmodeled external, `alloc.vec.Vec.Insts.CoreIterTraitsCollectExtendSharedAT.extend`), so the
/// append is the explicit index loop: O(xs.len()) time, O(1) extra space.
fn push_all(dst: &mut Vec<u32>, xs: &[u32]) {
    let mut i = 0;
    while i < xs.len() {
        dst.push(xs[i]);
        i += 1;
    }
}

pub(crate) fn bytes(words: Vec<u32>) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let b = words[i].to_le_bytes();
        out.push(b[0]);
        out.push(b[1]);
        out.push(b[2]);
        out.push(b[3]);
        i += 1;
    }
    out
}
const OP_IEQ: u32 = 170;
const OP_FNEG: u32 = 127;

const GLSL_FABS: u32 = 4;

const OP_CONVERT_F_TO_S: u32 = 110;
const OP_BITCAST: u32 = 124;
const OP_BITWISE_OR: u32 = 197;
const OP_UGE: u32 = 174;
const GLSL_FMIN: u32 = 37;
const GLSL_FMAX2: u32 = 40;

impl Kb {
    /// The signed i32 type, created on first use.
    pub(crate) fn sint_ty(&mut self) -> u32 {
        let mut i = 0;
        let mut hit: Option<u32> = None;
        while i < self.ucache.len() {
            let (v, id) = self.ucache[i];
            if v == u32::MAX {
                hit = Some(id); // stashed under the sentinel key
            }
            i += 1;
        }
        match hit {
            Some(id) => id,
            None => {
                let id = self.id();
                self.ty(OP_TYPE_INT, &[id, 32, 1]);
                self.ucache.push((u32::MAX, id));
                id
            }
        }
    }
}

const OP_TYPE_ARRAY: u32 = 28;
const OP_CONTROL_BARRIER: u32 = 224;
const SC_WORKGROUP: u32 = 4;
const SCOPE_WORKGROUP: u32 = 2;
// MemorySemantics: AcquireRelease (0x8) | WorkgroupMemory (0x100)
const SEM_WG_ACQREL: u32 = 0x108;

/// Where `Kb::rmsnorm_wg` writes its normalized·gain output: a shared array (`buf, bufp`, the
/// fused-norm-into-gemv shape) or a global SSBO (the standalone `rmsnorm_module` shape).
#[derive(Clone, Copy)]
enum NormOut {
    Shared(u32, u32),
    Global(u32),
}

impl Kb {
    /// A Workgroup-scope shared f32 array of `len`; returns (var id, element ptr type id).
    pub(crate) fn shared_f32(&mut self, len: u32) -> (u32, u32) {
        let clen = self.cu(len);
        let (arr, ptr_arr, var, ptr_el) = (self.id(), self.id(), self.id(), self.id());
        let f = self.float;
        self.ty(OP_TYPE_ARRAY, &[arr, f, clen]);
        self.ty(OP_TYPE_POINTER, &[ptr_arr, SC_WORKGROUP, arr]);
        self.ty(OP_VARIABLE, &[ptr_arr, var, SC_WORKGROUP]);
        self.ty(OP_TYPE_POINTER, &[ptr_el, SC_WORKGROUP, f]);
        (var, ptr_el)
    }
    pub(crate) fn shared_load(&mut self, var: u32, ptr_el: u32, idx: u32) -> u32 {
        let f = self.float;
        let p = self.e(OP_ACCESS_CHAIN, ptr_el, &[var, idx]);
        self.e(OP_LOAD, f, &[p])
    }
    pub(crate) fn shared_store(&mut self, var: u32, ptr_el: u32, idx: u32, val: u32) {
        let p = self.e(OP_ACCESS_CHAIN, ptr_el, &[var, idx]);
        self.ins(OP_STORE, &[p, val]);
    }
    /// OpControlBarrier at Workgroup scope, acquire-release on workgroup memory.
    pub(crate) fn wg_barrier(&mut self) {
        let sc = self.cu(SCOPE_WORKGROUP);
        let sem = self.cu(SEM_WG_ACQREL);
        self.ins(OP_CONTROL_BARRIER, &[sc, sc, sem]);
    }
    /// `tid + t·wg` as an id (`t == 0` is `tid`).
    pub(crate) fn strided_idx(&mut self, tid: u32, t: u32, wg: u32) -> u32 {
        if t == 0 {
            return tid;
        }
        let off = self.cu(t * wg);
        self.iadd(tid, off)
    }
    /// Reduce `red[0..wg]` (a power-of-two shared f32 array) into `red[0]` by `+` (`is_max=false`)
    /// or `max` (`is_max=true`): the log₂(wg) tree, a barrier between levels. `tid` is the lane id.
    /// Grouping-free by ℝ associativity (`MLPrimitivesBridge.osFuse_grouping_free` at the softmax
    /// chart) — the parallel tree denotes the same sum/max as the serial fold.
    fn tree_reduce(&mut self, red: u32, redp: u32, wg: u32, tid: u32, is_max: bool) {
        // MEASURED (2026-07-08, q4 smollm, DEC_UNROLL=32, 3-run medians): a subgroup two-phase
        // form (OpGroupNonUniformFAdd per warp + balanced warp-0 combine, 8 barriers -> 3) timed
        // 944.6 us/token GPU-active vs 938.5 for this tree -- NO win: the decode kernels are
        // work/bandwidth-bound, so barrier stalls overlap memory latency at occupancy and sync
        // savings don't convert (the megakernel lesson again). The FAdd reorder also rode the
        // rmsnorm quantize-to-int8 threshold through 30 requant layers -> a q4 argmax edge-flip
        // (1/24 teacher-forced vs 0/24 here). Tree kept; Kb's subgroup machinery (sg_reduce /
        // sg_builtin) stays for genuinely sync-bound future sites.
        let mut stride = wg / 2;
        while stride >= 1 {
            let cs = self.cu(stride);
            let lt = self.ult(tid, cs);
            self.iff(lt, |mut k| {
                let a = k.shared_load(red, redp, tid);
                let ti = k.iadd(tid, cs);
                let b = k.shared_load(red, redp, ti);
                let r = if is_max {
                    k.ext2(GLSL_FMAX, a, b)
                } else {
                    k.fadd(a, b)
                };
                k.shared_store(red, redp, tid, r);
                k
            });
            self.wg_barrier();
            stride /= 2;
        }
    }
    /// The WG-cooperative RMSNorm wrapper: strided sumsq over `src[0..n]` → shared tree-reduce →
    /// `rsqrt(mean+eps)` → per-element `src[j]·inv·gain[j]`, written to `out`. `red/redp` is a
    /// caller-owned WG-sized shared f32 array (reused as reduction scratch, not the output). `tid`
    /// is whichever builtin fits the caller's dispatch shape (`LocalInvocationId` when the
    /// workgroup shares the grid with other rows/segments, `GlobalInvocationId.x` when the
    /// dispatch is a single workgroup) — the caller supplies it, this method does not choose.
    /// Does NOT emit a trailing `wg_barrier()`: call sites differ on whether one is needed before
    /// the next stage reads `out`. Literal extraction of the shape shared by `rmsnorm_module` and
    /// `gemv_dp4a_norm_module`'s norm section — same emission order, bit-identical results.
    ///
    /// Complexity: each lane issues `u = ⌈n/wg⌉` guarded strided x loads and `u` guarded gain
    /// loads, and `u` `Fma(x, x, acc)` plus `u` `fmul`/`fmul` on the scale pass — `O(⌈n/wg⌉)`
    /// global loads and ops per lane, counted in global loads and arithmetic ops. The reduction
    /// adds the `O(log₂ wg)` shared tree with one barrier per level. The bodies are unrolled
    /// (arm (a)) so a lane's loads issue before its dependent adds.
    fn rmsnorm_wg(
        &mut self,
        src: u32,
        gain: u32,
        red: u32,
        redp: u32,
        n: u32,
        eps: f32,
        tid: u32,
        wg: u32,
        out: NormOut,
    ) {
        let cn = self.cu(n);
        let u = n.div_ceil(wg);
        let acc = self.fvar_f();
        let zero = self.cf_bits(0x0000_0000);
        self.store_var(acc, zero);
        // Arm (a): every lane issues all of its `⌈n/wg⌉` x loads before the first dependent
        // square/add, keeping each in a function register for the scale loop (no second load).
        // The per-lane add order is the strided loop's ascending `i`, and the accumulation is the
        // explicit `Fma(x, x, acc)` the driver contracts the naive loop into. The driver does not
        // contract this unrolled body: a bare `fmul`+`fadd` is the naive serial sum, one ULP off
        // (measured: it moved `SEQHASH` to `27ee7624bee91cf8`), so the fma is emitted explicitly to
        // keep the HEAD value `414708724a03be6e`.
        let mut xv: Vec<u32> = Vec::new();
        for t in 0..u {
            xv.push(self.fvar_f());
            let ti = self.strided_idx(tid, t, wg);
            let inb = self.ult(ti, cn);
            let var = xv[t as usize];
            self.iff(inb, |mut k| {
                let v = k.load_f(src, ti);
                k.store_var(var, v);
                k
            });
        }
        for t in 0..u {
            let ti = self.strided_idx(tid, t, wg);
            let inb = self.ult(ti, cn);
            let var = xv[t as usize];
            self.iff(inb, |mut k| {
                let v = k.load_var_f(var);
                let a0 = k.load_var_f(acc);
                let a1 = k.fma(v, v, a0);
                k.store_var(acc, a1);
                k
            });
        }
        let af = self.load_var_f(acc);
        self.shared_store(red, redp, tid, af);
        self.wg_barrier();
        self.tree_reduce(red, redp, wg, tid, false);
        let c0 = self.cu(0);
        let tot = self.shared_load(red, redp, c0);
        let invn = self.cf(1.0 / n as f32);
        let mean = self.fmul(tot, invn);
        let ce = self.cf(eps);
        let me = self.fadd(mean, ce);
        let inv = self.ext1(GLSL_INVERSE_SQRT, me);
        let cn2 = self.cu(n);
        // gamma issued early — all `⌈n/wg⌉` gain loads before the first dependent multiply.
        let mut gv: Vec<u32> = Vec::new();
        for t in 0..u {
            gv.push(self.fvar_f());
            let ji = self.strided_idx(tid, t, wg);
            let inb = self.ult(ji, cn2);
            let var = gv[t as usize];
            self.iff(inb, |mut k| {
                let v = k.load_f(gain, ji);
                k.store_var(var, v);
                k
            });
        }
        for t in 0..u {
            let ji = self.strided_idx(tid, t, wg);
            let inb = self.ult(ji, cn2);
            let xvar = xv[t as usize];
            let gvar = gv[t as usize];
            self.iff(inb, |mut k| {
                let xj = k.load_var_f(xvar);
                let gj = k.load_var_f(gvar);
                let tt = k.fmul(xj, inv);
                let ov = k.fmul(tt, gj);
                match out {
                    NormOut::Shared(buf, bufp) => k.shared_store(buf, bufp, ji, ov),
                    NormOut::Global(buf) => k.store_f(buf, ji, ov),
                }
                k
            });
        }
    }
    /// `rmsnorm_wg` into a shared scratch array, then quantize that into the DP4A activation
    /// chart (`emit_quantize_to_shared`) — the fused-norm gemv's whole phase-1a in one call,
    /// barriers included (one after the norm write, one after the quantize, matching the
    /// pre-existing call-site convention exactly). The shape `gemv_dp4a_norm_module` and Stage
    /// 2's chart megakernel both need for "normalize x, quantize it, gemv folds read the chart."
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_quant_wg(
        &mut self,
        src: u32,
        gain: u32,
        red: u32,
        redp: u32,
        nrm: u32,
        nrmp: u32,
        xqs: u32,
        xqsp: u32,
        sxs: u32,
        sxsp: u32,
        xss: u32,
        xssp: u32,
        n: u32,
        eps: f32,
        dpw: u32,
        tid: u32,
        wg: u32,
    ) {
        self.rmsnorm_wg(
            src,
            gain,
            red,
            redp,
            n,
            eps,
            tid,
            wg,
            NormOut::Shared(nrm, nrmp),
        );
        self.wg_barrier();
        self.emit_quantize_to_shared(nrm, nrmp, xqs, xqsp, sxs, sxsp, xss, xssp, n, dpw, tid, wg);
        self.wg_barrier();
    }
    /// Quantize a shared f32 array `nrm[0..n]` into the DP4A activation chart, in place from
    /// shared: per-32-block absmax → `scales[b]` + block sum → `xs[b]` (a `wg_barrier` closes the
    /// scales), then per-word int8 pack deinterleaved for `dpw` → `xq[c]`. The `block_scale` +
    /// `actquant_pack` pair folded into a producer kernel, reading shared not global.
    #[allow(clippy::too_many_arguments)]
    fn emit_quantize_from_shared(
        &mut self,
        nrm: u32,
        nrmp: u32,
        xq: u32,
        scales: u32,
        xs: u32,
        n: u32,
        dpw: u32,
        sint: u32,
        tid: u32,
        wg: u32,
        // MoE: `(word_base, block_base)` value ids to offset the three output streams by, when
        // this workgroup owns one SLOT of a routed batch. `None` keeps the emitted words
        // byte-identical to the single-vector form, which every existing caller relies on.
        slot_base: Option<(u32, u32)>,
    ) {
        let cwg = self.cu(wg);
        // the per-block scale crosses threads (written by block owners, read by every word lane):
        // it travels through workgroup memory, which `wg_barrier` orders. Reaching it through the
        // `scales` SSBO is the cross-invocation storage race (GLSL450: a WorkgroupMemory barrier
        // does not make a storage-buffer write visible to another invocation).
        let (ssc, sscp) = self.shared_f32(n / 32);
        // per-block absmax → scale + block sum
        let cblk = self.cu(n / 32);
        self.loop_step(tid, cblk, cwg, |k, b| {
            let c32 = k.cu(32);
            let base = k.imul(b, c32);
            let mxv = k.fvar_f();
            let sumv = k.fvar_f();
            let tiny = k.cf_bits(0x2b8c_bccc);
            k.store_var(mxv, tiny);
            let z = k.cf_bits(0x0000_0000);
            k.store_var(sumv, z);
            let c0u = k.cu(0);
            let c32b = k.cu(32);
            k.loop_u(c0u, c32b, |mut k, j| {
                let idx = k.iadd(base, j);
                let v = k.shared_load(nrm, nrmp, idx);
                let f = k.float;
                let gl = k.glsl;
                let av = k.e(OP_EXT_INST, f, &[gl, GLSL_FABS, v]);
                let m0 = k.load_var_f(mxv);
                let m1 = k.ext2(GLSL_FMAX, m0, av);
                k.store_var(mxv, m1);
                let s0 = k.load_var_f(sumv);
                let s1 = k.fadd(s0, v);
                k.store_var(sumv, s1);
                k
            });
            let mx = k.load_var_f(mxv);
            let inv127 = k.cf_bits(0x3c01_0204);
            let sc = k.fmul(mx, inv127);
            let bo = match slot_base {
                Some((_, bb)) => k.iadd(bb, b),
                None => b,
            };
            k.store_f(scales, bo, sc);
            k.shared_store(ssc, sscp, b, sc);
            let sm = k.load_var_f(sumv);
            k.store_f(xs, bo, sm);
        });
        self.wg_barrier();
        // per-word int8 pack, deinterleaved for dpw
        let cwords = self.cu(n / 4);
        self.loop_step(tid, cwords, cwg, |k, c| {
            let cdpw = k.cu(dpw);
            let cc = k.udiv(c, cdpw);
            let ll = k.umod(c, cdpw);
            let c4d = k.cu(4 * dpw);
            let base0 = k.imul(cc, c4d);
            let base = k.iadd(base0, ll);
            let c32 = k.cu(32);
            let kb0 = k.udiv(base0, c32);
            let sx = k.shared_load(ssc, sscp, kb0);
            let word = k.fvar_u();
            let z = k.cu(0);
            k.store_var(word, z);
            let c0u = k.cu(0);
            let four = k.cu(4);
            k.loop_u(c0u, four, |mut k, b| {
                let bd = k.imul(b, cdpw);
                let idx = k.iadd(base, bd);
                let v = k.shared_load(nrm, nrmp, idx);
                let q = k.fdiv(v, sx);
                let half = k.cf_bits(0x3f00_0000);
                let nhalf = k.cf_bits(0xbf00_0000);
                let zf = k.cf_bits(0x0000_0000);
                let bt = k.bool_t;
                let pos = k.e(OP_FORD_GREATER_THAN, bt, &[q, zf]);
                let f = k.float;
                let adj = k.e(OP_SELECT, f, &[pos, half, nhalf]);
                let qa = k.fadd(q, adj);
                let hi = k.cf_bits(0x42fe_0000);
                let lo = k.cf_bits(0xc2fe_0000);
                let gl = k.glsl;
                let q1 = k.e(OP_EXT_INST, f, &[gl, GLSL_FMIN, qa, hi]);
                let q2 = k.e(OP_EXT_INST, f, &[gl, GLSL_FMAX2, q1, lo]);
                let si = k.e(OP_CONVERT_F_TO_S, sint, &[q2]);
                let u = k.uint;
                let ub = k.e(OP_BITCAST, u, &[si]);
                let cff = k.cu(0xFF);
                let byte = k.e(OP_BITWISE_AND, u, &[ub, cff]);
                let c8 = k.cu(8);
                let sh = k.imul(b, c8);
                let shifted = k.e(OP_SHIFT_LEFT_LOGICAL, u, &[byte, sh]);
                let w0 = k.load_var_u(word);
                let w1 = k.e(OP_BITWISE_OR, u, &[w0, shifted]);
                k.store_var(word, w1);
                k
            });
            let wv = k.load_var_u(word);
            let psu = k.ptr_sb_u;
            let c0b = k.cu(0);
            let co = match slot_base {
                Some((qb, _)) => k.iadd(qb, c),
                None => c,
            };
            let p = k.e(OP_ACCESS_CHAIN, psu, &[xq, c0b, co]);
            k.ins(OP_STORE, &[p, wv]);
        });
    }
    /// `emit_quantize_from_shared`'s twin that writes the DP4A chart to SHARED (`xqs`/`sxs`/`xss`)
    /// instead of global SSBOs — the co-reduction form for [`gemv_dp4a_norm_module`]: the quantized
    /// activation never lands in global, it rides the consuming gemv's own workgroup. The VALUES are
    /// byte-identical to the global twin (same 1e-12 absmax floor, 1/127 scale, round-half-away int8,
    /// per-`dpw` deinterleaved pack) so the fused logits match the un-fused path bit-for-bit.
    #[allow(clippy::too_many_arguments)]
    fn emit_quantize_to_shared(
        &mut self,
        nrm: u32,
        nrmp: u32,
        xqs: u32,
        xqsp: u32,
        sxs: u32,
        sxsp: u32,
        xss: u32,
        xssp: u32,
        n: u32,
        dpw: u32,
        tid: u32,
        wg: u32,
    ) {
        let (b0, b1) = (self.cu(0), self.cu(n / 32));
        self.emit_quantize_to_shared_win(
            nrm, nrmp, xqs, xqsp, sxs, sxsp, xss, xssp, dpw, tid, wg, b0, b1, None,
        );
    }
    /// [`Kb::emit_quantize_to_shared`] restricted to the 32-block window `[b0, b1)` — the mandala
    /// mask on the activation chart (`MandalaMemory.access_in_region`: the workgroup touches only the
    /// region its own split-K segment will fold). Every 32-block is self-contained (its own absmax,
    /// its own sum, its own 8 packed words), so windowing is exact, not an approximation: the chart
    /// bytes for a block are identical whether the whole vector or only that block was quantized.
    /// This is what makes the producer fusion in [`gemv_actq_module`] cost O(window) instead of
    /// O(k) per workgroup. `b0`/`b1` are ids, not constants.
    #[allow(clippy::too_many_arguments)]
    fn emit_quantize_to_shared_win(
        &mut self,
        nrm: u32,
        nrmp: u32,
        xqs: u32,
        xqsp: u32,
        sxs: u32,
        sxsp: u32,
        xss: u32,
        xssp: u32,
        dpw: u32,
        tid: u32,
        wg: u32,
        b0: u32,
        b1: u32,
        // Block offset the staged arrays are zero-based against — see `dp4a_chart_fold`'s. `None`
        // emits no subtraction, so the unwindowed callers keep byte-identical SPIR-V.
        base_blk: Option<u32>,
    ) {
        let cwg = self.cu(wg);
        // per-32-block absmax → scale + block sum, into shared
        let cblk = b1;
        let btid = self.iadd(b0, tid);
        self.loop_step(btid, cblk, cwg, |k, b| {
            let c32 = k.cu(32);
            // `b` stays absolute (it walks the split-K window); the staged arrays are
            // window-relative when `base_blk` is set, so each access re-bases independently.
            let brel = match base_blk {
                Some(bb) => {
                    let u = k.uint;
                    k.e(OP_ISUB, u, &[b, bb])
                }
                None => b,
            };
            let base = k.imul(brel, c32);
            let mxv = k.fvar_f();
            let sumv = k.fvar_f();
            let tiny = k.cf_bits(0x2b8c_bccc);
            k.store_var(mxv, tiny);
            let z = k.cf_bits(0x0000_0000);
            k.store_var(sumv, z);
            // Arm (b): the 32 shared loads issue back to back, then the max/add chain runs in the
            // same element order with the same single load per element as the naive loop.
            // Complexity: 32 shared loads + 32 fabs/32 fmax/32 fadd per block; `loop_step` covers
            // `⌈blocks/wg⌉` blocks per lane → `O(32·⌈blocks/wg⌉)` shared loads per lane.
            let mut vs: Vec<u32> = Vec::with_capacity(32);
            for j in 0..32u32 {
                let cj = k.cu(j);
                let idx = k.iadd(base, cj);
                vs.push(k.shared_load(nrm, nrmp, idx));
            }
            for j in 0..32u32 {
                let v = vs[j as usize];
                let f = k.float;
                let gl = k.glsl;
                let av = k.e(OP_EXT_INST, f, &[gl, GLSL_FABS, v]);
                let m0 = k.load_var_f(mxv);
                let m1 = k.ext2(GLSL_FMAX, m0, av);
                k.store_var(mxv, m1);
                let s0 = k.load_var_f(sumv);
                let s1 = k.fadd(s0, v);
                k.store_var(sumv, s1);
            }
            let mx = k.load_var_f(mxv);
            let inv127 = k.cf_bits(0x3c01_0204);
            let sc = k.fmul(mx, inv127);
            k.shared_store(sxs, sxsp, brel, sc);
            let sm = k.load_var_f(sumv);
            k.shared_store(xss, xssp, brel, sm);
        });
        self.wg_barrier();
        // per-word int8 pack, deinterleaved for dpw, into shared. Exactly 8 words per 32-block
        // (n/4 words over n/32 blocks), so the window `[b0, b1)` in blocks is `[8·b0, 8·b1)` in words.
        let c8w = self.cu(8);
        let cwords = self.imul(b1, c8w);
        let wstart = self.imul(b0, c8w);
        let wtid = self.iadd(wstart, tid);
        self.loop_step(wtid, cwords, cwg, |k, c| {
            let cdpw = k.cu(dpw);
            let cc = k.udiv(c, cdpw);
            let ll = k.umod(c, cdpw);
            let c4d = k.cu(4 * dpw);
            let base0 = k.imul(cc, c4d);
            let base = k.iadd(base0, ll);
            let c32 = k.cu(32);
            let kb = k.udiv(base0, c32);
            let kb = match base_blk {
                Some(bb) => {
                    let u = k.uint;
                    k.e(OP_ISUB, u, &[kb, bb])
                }
                None => kb,
            };
            let sx = k.shared_load(sxs, sxsp, kb);
            let word = k.fvar_u();
            let z = k.cu(0);
            k.store_var(word, z);
            let c0u = k.cu(0);
            let four = k.cu(4);
            k.loop_u(c0u, four, |mut k, b| {
                let bd = k.imul(b, cdpw);
                let idx = k.iadd(base, bd);
                // `base` comes from `base0 = (c/dpw)·4·dpw` and `c` is the ABSOLUTE word index, so
                // `idx` is absolute too — the earlier comment claiming otherwise was wrong and
                // SEQHASH caught it. `nrm` spans only the window, so re-base by `32·b0`.
                let idx = match base_blk {
                    Some(bb) => {
                        let c32e = k.cu(32);
                        let eoff = k.imul(bb, c32e);
                        let u = k.uint;
                        k.e(OP_ISUB, u, &[idx, eoff])
                    }
                    None => idx,
                };
                let v = k.shared_load(nrm, nrmp, idx);
                let byte = k.int8_code(v, sx);
                let u = k.uint;
                let c8 = k.cu(8);
                let sh = k.imul(b, c8);
                let shifted = k.e(OP_SHIFT_LEFT_LOGICAL, u, &[byte, sh]);
                let w0 = k.load_var_u(word);
                let w1 = k.e(OP_BITWISE_OR, u, &[w0, shifted]);
                k.store_var(word, w1);
                k
            });
            let wv = k.load_var_u(word);
            let c = match base_blk {
                Some(bb) => {
                    let c8b = k.cu(8);
                    let woff = k.imul(bb, c8b);
                    let u = k.uint;
                    k.e(OP_ISUB, u, &[c, woff])
                }
                None => c,
            };
            k.shared_store_u(xqs, xqsp, c, wv);
        });
    }
}

// ── the DP4A attention lever (int8 q·k via OpSDot, the gemv's integer-dot fold on the scores) ──
const OP_EXTENSION: u32 = 10;
const OP_SDOT: u32 = 4450; // OpSDot — signed·signed 4×8-bit dot → i32 (both q and k are signed
                           // absmax-int8; the gemv's OpSUDot=4452 is for unsigned WEIGHTS, this is its sibling)
const OP_SUDOT: u32 = 4452; // OpSUDot — signed activation · UNSIGNED weight 4×8-bit dot → i32 (the
                            // gemv chart fold; mirrors simt::gemv_dp4a_module_dyn's inner instruction)
const OP_CONVERT_S_TO_F: u32 = 111;
const CAP_DOT_PRODUCT: u32 = 6019;
const CAP_GROUP_NON_UNIFORM: u32 = 61;
const CAP_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
const CAP_DOT_INPUT_4X8_PACKED: u32 = 6018;
const PACKED_VEC_4X8: u32 = 0;

/// **Which scale face a chart gemv reads**, so the two-level chart is one value threaded through
/// `dp4a_chart_fold` rather than a bool and a loose offset at every call site.
///
/// The bytes this decides are the subject of `results/where-the-27b-bits-go.md`: the single-level
/// face spends **1.0 bits/weight** on scale metadata against llama.cpp IQ3_XXS's 0.1875, which is
/// where the 27B's whole quality deficit came from. Two-level halves it to 0.5 at **zero** change
/// to the codes — `q4_sub8_gpu_chart` is a permutation of `qsub`/`qsup`, not a re-quantisation,
/// and `results/scale-ladder.md` measures the reconstruction RMS as identical to six decimals.
///
/// **`sup` shares `dm`'s BINDING.** The per-row super-pairs are appended to the `sub` words in one
/// buffer and reached at a compile-time element offset, because `m` and `k` are both known when the
/// module is built. The alternative — a seventh SSBO — shifts every binding after it, and
/// `Gpu::pipeline` takes a binding COUNT rather than a set, so it would have forced a matching
/// re-order of the host's buffer list at every one of `make_gemv`'s call sites. One `Vec::extend`
/// at upload replaces all of that.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChartScale {
    /// Binding of the scale buffer — one f16 `(d, lo)` pair per (block, row) for the single-level
    /// face, else `sub` (a `(d8, l8)` byte pair per row, two rows per word) followed by `sup`.
    pub dm: u32,
    /// Element index of the per-row f16 `(dsup, losup)` array WITHIN `dm`, or **0 for the
    /// single-level face** — `sub` always starts at 0, so a real offset is never 0.
    pub sup_off: u32,
    /// `m.div_ceil(2)` — the `sub` row-pair stride. Ignored when [`Self::sup_off`] is 0.
    pub mrow: u32,
}

impl ChartScale {
    /// The single-level `dm` face — one f16 pair per (block, row).
    pub fn dm(dm: u32) -> Self {
        Self {
            dm,
            sup_off: 0,
            mrow: 0,
        }
    }

    /// The two-level face at `m` rows over `blocks` 32-blocks, laid out as
    /// `[sub; blocks·⌈m/2⌉][sup; m]` in the buffer at binding `dm` — exactly what
    /// [`two_level_chart_buffer`] builds.
    pub fn two_level(dm: u32, m: u32, blocks: u32) -> Self {
        let mrow = m.div_ceil(2);
        Self {
            dm,
            sup_off: blocks * mrow,
            mrow,
        }
    }
}

/// **The one buffer a two-level chart gemv binds** — `q4_sub8_gpu_chart`'s `sub` words followed by
/// its per-row `sup` pairs, which is the layout [`ChartScale::two_level`] computes offsets for.
///
/// Built here rather than at the call site so the host and the kernel cannot disagree about where
/// `sup` starts; that disagreement reads as plausible logits, not as a crash.
pub fn two_level_chart_buffer(sub: &[u32], sup: &[u32]) -> Vec<u32> {
    let mut v = Vec::with_capacity(sub.len() + sup.len());
    v.extend_from_slice(sub);
    v.extend_from_slice(sup);
    v
}

const OP_SHIFT_RIGHT_LOGICAL2: u32 = 194;

const OP_LOGICAL_OR: u32 = 166;
const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;

impl Kb {
    /// The `LocalInvocationIndex` builtin (the flattened lane id within a workgroup), lazily made.
    pub(crate) fn local_index(&mut self) -> u32 {
        if self.lii == 0 {
            let ptr_in_u = self.id();
            let u = self.uint;
            self.ty(OP_TYPE_POINTER, &[ptr_in_u, SC_INPUT, u]);
            self.lii = self.id();
            let v = self.lii;
            self.ty(OP_VARIABLE, &[ptr_in_u, v, SC_INPUT]);
            self.deco(&[OP_DECORATE, v, DEC_BUILT_IN, BUILTIN_LOCAL_INVOCATION_INDEX]);
        }
        let (u, v) = (self.uint, self.lii);
        self.e(OP_LOAD, u, &[v])
    }
}
const GLSL_UMIN: u32 = 38;

const OP_LOGICAL_NOT: u32 = 168;
