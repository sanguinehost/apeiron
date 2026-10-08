//! simt_nn::coopmat_gemm — **the tensor-core chart GEMM for prefill.**
//!
//! A drop-in for [`crate::simt::gemv_dp4a_batch_rt_module`]: the same six bindings, the same
//! k-packed chart, the same per-32-block f32 epilogue `acc += ((d·q)·sx + lo·xs)` in block order.
//! Only the per-block integer dot `q` changes engine: the `rt × bt` grid of `OpSUDot` becomes
//! signed-int8 `M16N16K32` cooperative-matrix multiplies (`SPV_KHR_cooperative_matrix`). K32 is one
//! chart block, so two blocks with unequal scales never share an integer accumulator, and `q` is the
//! exact i32 the DP4A kernel forms (`tests/gpu_coopmat_gemm.rs` checks every output cell).
//!
//! **Signedness.** The RTX 5080 lists `s8·s8→s32` and `u8·u8→u32` only, no mixed entry, while
//! `OpSUDot` is `signed(x)·unsigned(w)`. Q4/Q2 codes (≤ 15) are valid s8 as they are; Q8 codes are
//! recentred `s = q − 128` (byte `XOR 0x80`) and `Σ q·x = Σ s·x + 128·Σ x` is restored in the integer
//! ring before the f32 conversion.
//!
//! **Source.** The kernel is GLSL, `src/spirv/coopmat_gemm.comp`, compiled once by `glslc` into the
//! committed `src/spirv/coopmat_gemm.spv` (the command is [`GLSLC_ARGS`]; the test recompiles it and
//! requires byte identity). The per-shape constants `M`, `BLOCKS`, `PACK` are specialization
//! constants whose defaults this module rewrites in the binary, so the pipeline needs no
//! `VkSpecializationInfo` and no push constants. Tile shape is fixed at compile time: [`TILE_BM`]
//! rows × [`TILE_BN`] tokens per workgroup of 4 subgroups, [`TILE_KB`] blocks per shared stage.
//!
//! **SwiGLU fusion.** With `FUSE = dpw` (spec id 3) the gate/up projection's epilogue also forms
//! `silu(gate) * up` and writes the down projection's activation chart (words, scales, sums) into the
//! buffer bound at O, so `swiglu_quant_batch_module` and the `gu` round trip drop out; the down GEMM
//! reads that chart with `SMOFF`/`SCOFF` (spec ids 5, 4) as the sum/scale offsets.
//! See [`coopmat_gemm_swiglu_bytes`] and [`coopmat_gemm_chart_bytes`].
//!
//! **Dispatch** `(m / TILE_BM rounded up, tokens / TILE_BN rounded up)`. The caller pads the
//! activation chart (XQ, XS, SX) and OUT to a multiple of [`TILE_BN`] rows, as the DP4A path pads to
//! its `bt`; the kernel carries no token guard. `m % 16 == 0`; rows past `m` in the last row tile are
//! clamped on load and not stored; `k % (32·TILE_KB) == 0`.

use crate::simt::Carrier;

/// Rows per workgroup (the shader's `BM`).
pub const TILE_BM: u32 = 128;
/// Tokens per workgroup (the shader's `BN`); activation/OUT rows are padded to a multiple of it.
pub const TILE_BN: u32 = 64;
/// Chart blocks per shared-memory stage (the shader's `KB`): `k` must be a multiple of `32·TILE_KB`.
pub const TILE_KB: u32 = 2;

/// The `glslc` invocation that produced [`COOPMAT_GEMM_SPV`] from `src/spirv/coopmat_gemm.comp`
/// (append `<source> -o <output>`).
pub const GLSLC_ARGS: &[&str] = &[
    "--target-env=vulkan1.1",
    "--target-spv=spv1.3",
    "-O",
    "-fshader-stage=compute",
];

/// The compiled kernel, specialization defaults unpatched.
pub static COOPMAT_GEMM_SPV: &[u8] = include_bytes!("../spirv/coopmat_gemm.spv");

const OP_DECORATE: u32 = 71;
const OP_SPEC_CONSTANT: u32 = 50;
const DEC_SPEC_ID: u32 = 1;
const SPEC_M: u32 = 0;
const SPEC_BLOCKS: u32 = 1;
const SPEC_PACK: u32 = 2;
const SPEC_FUSE: u32 = 3;
const SPEC_SCOFF: u32 = 4;
const SPEC_SMOFF: u32 = 5;
const SPEC_KSPLIT: u32 = 6;
const SPEC_PLANEB: u32 = 7;
const SPEC_Q6: u32 = 8;

/// Rewrite the default value of every 32-bit `OpSpecConstant` whose `SpecId` is a key of `vals`.
/// Panics when a requested `SpecId` is absent: a constant left at its default is a different kernel.
///
/// Cost: two linear passes over the module, O(words) time, O(1) extra space beyond the copy.
fn specialize(module: &[u32], vals: &[(u32, u32)]) -> Vec<u32> {
    let mut out = module.to_vec();
    let mut ids: Vec<(u32, u32)> = Vec::new(); // (result id, value)
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

/// The coopmat chart GEMM specialized to `m × k` at `carrier` (Q8, Q4, Q2 or the two-plane Q5/Q6).
pub fn coopmat_gemm_module(m: u32, k: u32, carrier: Carrier) -> Vec<u32> {
    coopmat_gemm_module_from(COOPMAT_GEMM_SPV, m, k, carrier)
}

/// [`coopmat_gemm_module`] over another compile of `src/spirv/coopmat_gemm.comp` (a tile variant
/// built with `-DBM= -DBN= -DSGM= -DSGN= -DKB=`; `examples/gemm_bench.rs --variants` sweeps them).
/// The `k` check here is the committed `TILE_KB`'s; a variant with a larger `KB` needs its own.
pub fn coopmat_gemm_module_from(spv: &[u8], m: u32, k: u32, carrier: Carrier) -> Vec<u32> {
    coopmat_gemm_specialized(spv, m, k, carrier, 0, 0, 0, 1)
}

/// The kernel as [`coopmat_gemm_module`] with the chart's sum stream at float offset `smoff` and its
/// scale stream at `scoff` of the buffers bound at XS and SX: the reader of a chart written by
/// [`coopmat_gemm_swiglu_module`] into one buffer.
pub fn coopmat_gemm_chart_module(
    m: u32,
    k: u32,
    carrier: Carrier,
    scoff: u32,
    smoff: u32,
) -> Vec<u32> {
    coopmat_gemm_specialized(COOPMAT_GEMM_SPV, m, k, carrier, 0, scoff, smoff, 1)
}

/// [`coopmat_gemm_chart_module`] as bytes.
pub fn coopmat_gemm_chart_bytes(
    m: u32,
    k: u32,
    carrier: Carrier,
    scoff: u32,
    smoff: u32,
) -> Vec<u8> {
    to_bytes(&coopmat_gemm_chart_module(m, k, carrier, scoff, smoff))
}

/// The gate/up GEMM (`m = 2·dff` rows, gate then up) with the SwiGLU epilogue: instead of the `m`-row
/// output it writes the down projection's activation chart at `dpw_out` (the down carrier's
/// `pack / 4`) into the buffer bound at O (words at word 0, scales at float `scoff`, sums at float
/// `smoff`, token-major, `dff/4` words and `dff/32` blocks a token). Bit-identical to the unfused pair
/// GEMM + `swiglu_quant_batch_module`. Requires `m % 128 == 0` and `dpw_out` in {1, 2, 4}; dispatch
/// `(m / 128, ceil(tokens / TILE_BN))` ([`coopmat_gemm_groups`]).
pub fn coopmat_gemm_swiglu_module(
    m: u32,
    k: u32,
    carrier: Carrier,
    dpw_out: u32,
    scoff: u32,
    smoff: u32,
) -> Vec<u32> {
    assert!(
        m % TILE_BM == 0,
        "the fused epilogue pairs 64 gate rows with 64 up rows: m % {TILE_BM} == 0"
    );
    assert!(
        matches!(dpw_out, 1 | 2 | 4),
        "the down chart's dpw is 1, 2 or 4, not {dpw_out}"
    );
    coopmat_gemm_specialized(COOPMAT_GEMM_SPV, m, k, carrier, dpw_out, scoff, smoff, 1)
}

/// [`coopmat_gemm_swiglu_module`] as bytes.
pub fn coopmat_gemm_swiglu_bytes(
    m: u32,
    k: u32,
    carrier: Carrier,
    dpw_out: u32,
    scoff: u32,
    smoff: u32,
) -> Vec<u8> {
    to_bytes(&coopmat_gemm_swiglu_module(
        m, k, carrier, dpw_out, scoff, smoff,
    ))
}

fn coopmat_gemm_specialized(
    spv: &[u8],
    m: u32,
    k: u32,
    carrier: Carrier,
    fuse: u32,
    scoff: u32,
    smoff: u32,
    ksplit: u32,
) -> Vec<u32> {
    assert!(
        ksplit >= 1 && (fuse == 0 || ksplit == 1),
        "ksplit is at least 1 and the SwiGLU epilogue does not split"
    );
    assert!(
        k % (32 * TILE_KB) == 0 && (k / 32 / TILE_KB) >= ksplit,
        "split-K needs at least one {TILE_KB}-block stage per split: k = {k}, ksplit = {ksplit}"
    );
    assert!(
        matches!(
            carrier,
            Carrier::Q8 | Carrier::Q4 | Carrier::Q2 | Carrier::Q5 | Carrier::Q6
        ),
        "coopmat GEMM serves the chart carriers Q8/Q4/Q2 and the two-plane Q5/Q6, not {carrier:?}"
    );
    assert!(
        k % (32 * TILE_KB) == 0,
        "k must be a multiple of {} (32 x {TILE_KB} blocks per stage)",
        32 * TILE_KB
    );
    assert!(m % 16 == 0 && m > 0, "m must be a positive multiple of 16");
    assert!(spv.len() % 4 == 0, "SPIR-V is a word stream");
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0], crate::simt::SPV_MAGIC, "not a SPIR-V module");
    specialize(
        &words,
        &[
            (SPEC_M, m),
            (SPEC_BLOCKS, k / 32),
            (SPEC_PACK, carrier.pack()),
            (SPEC_FUSE, fuse),
            (SPEC_SCOFF, scoff),
            (SPEC_SMOFF, smoff),
            (SPEC_KSPLIT, ksplit),
            (SPEC_PLANEB, carrier.plane_b_off(m, k)),
            (SPEC_Q6, u32::from(carrier == Carrier::Q6)),
        ],
    )
}

/// The coopmat chart GEMM split along k into `ksplit` partial products: the grid is
/// `(coopmat_gemm_groups(m, tokens).0, .1, ksplit)`, split `z` sums the `TILE_KB`-block stages
/// `[z*U/ksplit, (z+1)*U/ksplit)`, `U = k/(32*TILE_KB)`, in block order, and stores its
/// `[ceil(tokens/TILE_BN)*TILE_BN][m]` partial tile at row `z * ceil(tokens/TILE_BN)*TILE_BN`
/// of the buffer bound at O, which holds `ksplit * rows * m` floats. [`splitk_reduce_module`] sums the
/// partials. `scoff`/`smoff` are the chart offsets of [`coopmat_gemm_chart_module`] (0 for a plain chart).
/// `ksplit == 1` is [`coopmat_gemm_chart_module`] bit for bit. The sum order differs from the unsplit
/// kernel's single serial f32 chain by reassociation of the `ksplit` partial chains.
pub fn coopmat_gemm_split_module(
    m: u32,
    k: u32,
    carrier: Carrier,
    scoff: u32,
    smoff: u32,
    ksplit: u32,
) -> Vec<u32> {
    coopmat_gemm_specialized(COOPMAT_GEMM_SPV, m, k, carrier, 0, scoff, smoff, ksplit)
}

/// [`coopmat_gemm_split_module`] as bytes.
pub fn coopmat_gemm_split_bytes(
    m: u32,
    k: u32,
    carrier: Carrier,
    scoff: u32,
    smoff: u32,
    ksplit: u32,
) -> Vec<u8> {
    to_bytes(&coopmat_gemm_split_module(
        m, k, carrier, scoff, smoff, ksplit,
    ))
}

/// The reduction kernel `src/spirv/splitk_reduce.comp` (same [`GLSLC_ARGS`]), specialization defaults unpatched.
pub static SPLITK_REDUCE_SPV: &[u8] = include_bytes!("../spirv/splitk_reduce.spv");

/// Workgroups x of the reduction: each of its `ceil(tokens/TILE_BN)` rows strides a `TILE_BN * m` float
/// tile over `REDUCE_GX * 256` lanes.
pub const REDUCE_GX: u32 = 28;

/// The reduction of `ksplit` partials `[ksplit][rows][m]` into `[rows][m]`, `(p0 + p1) + p2 ...` in order.
/// Dispatch `(REDUCE_GX, ceil(tokens/TILE_BN), 1)`. Requires `(TILE_BN * m) % 4 == 0`.
///
/// Cost: `ksplit * rows * m` reads and `rows * m` writes of f32, `(ksplit - 1) * rows * m` adds; one
/// vec4 a lane per iteration, so a workgroup row of `TILE_BN * m / 4` vec4 takes
/// `ceil(TILE_BN * m / 4 / (REDUCE_GX * 256))` iterations a lane.
pub fn splitk_reduce_module(m: u32, ksplit: u32) -> Vec<u32> {
    assert!(ksplit >= 2, "a reduction has at least two partials");
    assert!((TILE_BN * m) % 4 == 0, "the reduction moves whole vec4");
    assert!(SPLITK_REDUCE_SPV.len() % 4 == 0, "SPIR-V is a word stream");
    let words: Vec<u32> = SPLITK_REDUCE_SPV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0], crate::simt::SPV_MAGIC, "not a SPIR-V module");
    specialize(&words, &[(0, m), (1, ksplit)])
}

/// [`splitk_reduce_module`] as bytes.
pub fn splitk_reduce_bytes(m: u32, ksplit: u32) -> Vec<u8> {
    to_bytes(&splitk_reduce_module(m, ksplit))
}

fn to_bytes(module: &[u32]) -> Vec<u8> {
    module.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// [`coopmat_gemm_module`] as bytes.
pub fn coopmat_gemm_bytes(m: u32, k: u32, carrier: Carrier) -> Vec<u8> {
    to_bytes(&coopmat_gemm_module(m, k, carrier))
}

/// Workgroup grid `(x, y)` for `m` rows and `tokens` tokens.
pub fn coopmat_gemm_groups(m: u32, tokens: u32) -> (u32, u32) {
    (m.div_ceil(TILE_BM), tokens.div_ceil(TILE_BN))
}
