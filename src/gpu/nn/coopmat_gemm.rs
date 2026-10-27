//! simt_nn::coopmat_gemm — **the tensor-core chart GEMM for prefill.**
//!
//! A drop-in for [`crate::gpu::simt::gemv_dp4a_batch_rt_module`]: the same six bindings, the same
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
//! **Source.** The kernel is GLSL, `src/gpu/spirv/coopmat_gemm.comp`, compiled once by `glslc` into the
//! committed `src/gpu/spirv/coopmat_gemm.spv` (the command is [`GLSLC_ARGS`]; the test recompiles it and
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

use crate::gpu::simt::Carrier;

/// Rows per workgroup (the shader's `BM`).
pub const TILE_BM: u32 = 128;
/// Tokens per workgroup (the shader's `BN`); activation/OUT rows are padded to a multiple of it.
pub const TILE_BN: u32 = 64;
/// Chart blocks per shared-memory stage (the shader's `KB`): `k` must be a multiple of `32·TILE_KB`.
pub const TILE_KB: u32 = 2;

/// The `glslc` invocation that produced [`COOPMAT_GEMM_SPV`] from `src/gpu/spirv/coopmat_gemm.comp`
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

/// [`coopmat_gemm_module`] over another compile of `src/gpu/spirv/coopmat_gemm.comp` (a tile variant
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
    assert_eq!(words[0], crate::gpu::simt::SPV_MAGIC, "not a SPIR-V module");
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

/// The reduction kernel `src/gpu/spirv/splitk_reduce.comp` (same [`GLSLC_ARGS`]), specialization defaults unpatched.
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
    assert_eq!(words[0], crate::gpu::simt::SPV_MAGIC, "not a SPIR-V module");
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

// ---------------------------------------------------------------------------------------------
// The bf16 face of the same body: the dense layer's three GEMMs, which the KD step runs 1592 times
// a step. `-DOPF=1` compiles the face; `coopmat_gemm.spv` (the chart) is untouched by it.

/// A bf16 face tile: `bm × bn` of C per workgroup, `kb` 16-wide k steps per shared stage. The two
/// committed compiles are [`BF16_TILE`] and [`BF16_TILE_WIDE`]; `tests/gpu_lin_bf16.rs` measures
/// both per KD shape and [`tile_for`] applies the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16Tile {
    /// Rows of C per workgroup (the shader's `BM`).
    pub bm: u32,
    /// Columns of C per workgroup (the shader's `BN`).
    pub bn: u32,
    /// 16-wide k steps per staged slice (the shader's `KB`).
    pub kb: u32,
}

/// The narrow face: `128 × 64`, 2×2 subgroups, two 16-wide k steps a stage. Measured best for
/// `rows ≤ 1024` (`tests/gpu_lin_bf16.rs` `lin_bf16_speed`).
pub const BF16_TILE: Bf16Tile = Bf16Tile {
    bm: 128,
    bn: 64,
    kb: 2,
};

/// The wide face: `128 × 128`, 2×2 subgroups of `WM 2 × WN 4` fragments. Measured best for
/// `rows ≥ 2048`; on the KD shapes it buys 1.04–1.26× over [`BF16_TILE`].
pub const BF16_TILE_WIDE: Bf16Tile = Bf16Tile {
    bm: 128,
    bn: 128,
    kb: 2,
};

/// The measured tile for a product with `rows` rows of C: the wide face from 2048 rows, where its
/// column reuse pays, and the narrow one below, where the row grid is too thin to fill it.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn tile_for(rows: u32) -> Bf16Tile {
    if rows >= 2048 {
        BF16_TILE_WIDE
    } else {
        BF16_TILE
    }
}

/// The `glslc` invocation that produced [`COOPMAT_GEMM_BF16_SPV`]: [`GLSLC_ARGS`] plus the face
/// selector `-DOPF=1` (append `<source> -o <output>`).
pub const GLSLC_BF16_ARGS: &[&str] = &[
    "--target-env=vulkan1.1",
    "--target-spv=spv1.3",
    "-O",
    "-fshader-stage=compute",
    "-DOPF=1",
];

/// [`GLSLC_BF16_ARGS`] plus the two guard faces at 0: every staged run is then a whole aligned
/// vec4 and every tile edge is exact, which is what a KD shape is.
pub const GLSLC_BF16_EXACT_ARGS: &[&str] = &[
    "--target-env=vulkan1.1",
    "--target-spv=spv1.3",
    "-O",
    "-fshader-stage=compute",
    "-DOPF=1",
    "-DGUARD=0",
    "-DEPIMASK=0",
];

/// The bf16 face at [`BF16_TILE`], guard and mask faces off: `src/gpu/spirv/coopmat_gemm_bf16.spv`.
pub static COOPMAT_GEMM_BF16_SPV: &[u8] = include_bytes!("../spirv/coopmat_gemm_bf16.spv");
/// The bf16 face at [`BF16_TILE_WIDE`], guard and mask faces off:
/// `src/gpu/spirv/coopmat_gemm_bf16_bn128.spv`.
pub static COOPMAT_GEMM_BF16_WIDE_SPV: &[u8] =
    include_bytes!("../spirv/coopmat_gemm_bf16_bn128.spv");
/// The bf16 face at [`BF16_TILE`] with both guards on: the compile a shape off the tiles' edges
/// runs. `src/gpu/spirv/coopmat_gemm_bf16_guarded.spv`.
pub static COOPMAT_GEMM_BF16_GUARDED_SPV: &[u8] =
    include_bytes!("../spirv/coopmat_gemm_bf16_guarded.spv");

const SPEC_BF16_M: u32 = 0;
const SPEC_BF16_STAGES: u32 = 1;
const SPEC_BF16_KL: u32 = 2;
const SPEC_BF16_NC: u32 = 3;
const SPEC_BF16_RSA: u32 = 9;
const SPEC_BF16_RSB: u32 = 10;
const SPEC_BF16_RSO: u32 = 11;
const SPEC_BF16_TA: u32 = 12;
const SPEC_BF16_TB: u32 = 13;

/// One linear product as the bf16 face's operand description: `C[row][col] = Σ_k A[row][k]·B[col][k]`
/// with f32 operands, bf16 fragments and an f32 accumulator. Strides are in f32 words; an order of 0
/// reads `[row][k]` (k contiguous), 1 reads `[k][row]` (row contiguous).
#[derive(Clone, Copy, Debug)]
pub struct LinBf16 {
    /// Rows of C (the A tile's row extent).
    pub rows: u32,
    /// Columns of C.
    pub cols: u32,
    /// The reduction length.
    pub k: u32,
    /// A's row stride: the [`LinBf16::ta`] order's outer dimension.
    pub rsa: u32,
    /// B's row stride: the [`LinBf16::tb`] order's outer dimension.
    pub rsb: u32,
    /// C's row stride: element `(row r, col t)` is at `c[t·rso + r]`.
    pub rso: u32,
    /// A's memory order: 0 = `[row][k]`, 1 = `[k][row]`.
    pub ta: u32,
    /// B's memory order: 0 = `[col][k]`, 1 = `[k][col]`.
    pub tb: u32,
}

impl LinBf16 {
    /// `Y = X·Wᵀ`: A = `W` (`o×i`, k inner), B = `X` (`n×i`, k inner), C = `Y` (`n×o`).
    #[must_use]
    pub fn fwd(n: u32, i: u32, o: u32) -> Self {
        Self {
            rows: o,
            cols: n,
            k: i,
            rsa: i,
            rsb: i,
            rso: o,
            ta: 0,
            tb: 0,
        }
    }

    /// `dX = dY·W`: A = `W` (`o×i`, o inner), B = `dY` (`n×o`, o inner), C = `dX` (`n×i`).
    #[must_use]
    pub fn bwd_input(n: u32, i: u32, o: u32) -> Self {
        Self {
            rows: i,
            cols: n,
            k: o,
            rsa: i,
            rsb: o,
            rso: i,
            ta: 1,
            tb: 0,
        }
    }

    /// `dW = dYᵀ·X`: A = `X` (`n×i`, n inner), B = `dY` (`n×o`, n inner), C = `dW` (`o×i`).
    #[must_use]
    pub fn bwd_weight(n: u32, i: u32, o: u32) -> Self {
        Self {
            rows: i,
            cols: o,
            k: n,
            rsa: i,
            rsb: o,
            rso: i,
            ta: 1,
            tb: 1,
        }
    }

    /// The workgroup grid `(rows/bm, cols/bn)` for `tile`: one workgroup a tile, from the tile the
    /// kernel was compiled for, never from a committed default. Both are rounded up; the kernel's
    /// guards drop the overhang.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn groups(&self, tile: Bf16Tile) -> (u32, u32) {
        (self.rows.div_ceil(tile.bm), self.cols.div_ceil(tile.bn))
    }
}

/// Whether a dispatch needs no guard at all: every staged four-word run lands inside its operand
/// (`K` and the strides are multiples of 4 f32, so the rows are 16-byte aligned) and every output
/// tile is exact (`M` a multiple of 16, `N` of the tile's `BN`).
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn bf16_exact(x: &LinBf16, tile: Bf16Tile) -> bool {
    // `k % (16*kb)` is the one that bites: a stage stages `16*kb` k positions (32 at `kb = 2`), and
    // the exact face reads whole aligned four-word runs of them, so a `k` inside a stage reads the
    // NEXT rows' words into the reduction. The factor GEMMs (`dense` materialisation, rank 4 or 8)
    // are exactly that shape; the KD ranks (576, 768, 864) are multiples of 32, which is why the
    // golden never saw it.
    x.rsa % 4 == 0
        && x.rsb % 4 == 0
        && x.k % (16 * tile.kb) == 0
        && x.rows % 16 == 0
        && x.cols % tile.bn == 0
}

/// The committed byte image for `x` at `tile`: one `glslc` compile of
/// `src/gpu/spirv/coopmat_gemm.comp`, guard and mask faces off where [`bf16_exact`] holds and on
/// where it does not.
///
/// # Panics
///
/// For a tile no committed compile carries; a variant is built and re-committed, never compiled at
/// dispatch.
fn bf16_image(x: &LinBf16, tile: Bf16Tile) -> &'static [u8] {
    if !bf16_exact(x, tile) {
        return COOPMAT_GEMM_BF16_GUARDED_SPV;
    }
    if tile == BF16_TILE {
        COOPMAT_GEMM_BF16_SPV
    } else if tile == BF16_TILE_WIDE {
        COOPMAT_GEMM_BF16_WIDE_SPV
    } else {
        panic!("no committed bf16 face compile for {tile:?}")
    }
}

/// The bf16 face specialized to one linear product at `tile`.
///
/// # Complexity
///
/// Two linear passes over the module's words and one copy of them: `O(words)` time and space.
///
/// # Panics
///
/// If a dimension is zero, a stride is shorter than the extent it spans, or an order bit is not 0 or
/// 1. Every tile edge is guarded in the kernel, so no shape divisibility is required.
pub fn coopmat_gemm_bf16_module(x: &LinBf16, tile: Bf16Tile) -> Vec<u32> {
    assert!(
        x.rows > 0 && x.cols > 0 && x.k > 0,
        "bf16 face: rows {}, cols {}, k {} must be non-zero",
        x.rows,
        x.cols,
        x.k
    );
    assert!(
        x.rsa >= if x.ta == 0 { x.k } else { x.rows }
            && x.rsb >= if x.tb == 0 { x.k } else { x.cols }
            && x.rso >= x.rows,
        "bf16 face: a stride is shorter than the extent it spans"
    );
    assert!(x.ta <= 1 && x.tb <= 1, "bf16 face: order bits are 0 or 1");
    let tile = bf16_tile_for(x, tile);
    coopmat_gemm_bf16_from(bf16_image(x, tile), x, tile)
}

/// The tile a dispatch actually runs: `tile` when [`bf16_exact`] holds, and [`BF16_TILE`] otherwise —
/// the only tile a guarded compile was made at. A guarded shape dispatched at the wide tile's grid
/// would leave its last `BN/2` columns unstored, so the resolution happens here, once, for both the
/// module and the grid.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn bf16_tile_for(x: &LinBf16, tile: Bf16Tile) -> Bf16Tile {
    if bf16_exact(x, tile) {
        tile
    } else {
        BF16_TILE
    }
}

/// [`coopmat_gemm_bf16_module`] over another compile of the same `.comp` (a tile variant built with
/// `-DBM= -DBN= -DSGM= -DSGN=`; `tests/gpu_lin_bf16.rs` sweeps them).
pub fn coopmat_gemm_bf16_from(spv: &[u8], x: &LinBf16, tile: Bf16Tile) -> Vec<u32> {
    assert!(spv.len() % 4 == 0, "SPIR-V is a word stream");
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0], crate::gpu::simt::SPV_MAGIC, "not a SPIR-V module");
    specialize(
        &words,
        &[
            (SPEC_BF16_M, x.rows),
            (SPEC_BF16_STAGES, x.k.div_ceil(16 * tile.kb)),
            (SPEC_BF16_KL, x.k),
            (SPEC_BF16_NC, x.cols),
            (SPEC_BF16_RSA, x.rsa),
            (SPEC_BF16_RSB, x.rsb),
            (SPEC_BF16_RSO, x.rso),
            (SPEC_BF16_TA, x.ta),
            (SPEC_BF16_TB, x.tb),
        ],
    )
}

/// [`coopmat_gemm_bf16_module`] as bytes.
pub fn coopmat_gemm_bf16_bytes(x: &LinBf16, tile: Bf16Tile) -> Vec<u8> {
    to_bytes(&coopmat_gemm_bf16_module(x, tile))
}

/// Build and record one linear product on the bf16 face: bindings 0 = A, 1 = B, 2 = C, 3 = A again,
/// 4 = B again (the word views the guards read), grid `(rows/bm, cols/bn)`, `32·SGM·SGN` lanes.
///
/// # Safety
///
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold its
/// declared shape in f32, and `c` the `cols·rows` output words.
///
/// # Complexity
///
/// Host: `O(words)` over the specialized module plus one pipeline, one descriptor set and two
/// aliased bindings; `O(1)` dispatched work items. Device: `rows·cols·k` MACs in
/// `ceil(rows/bm)·ceil(cols/bn)` workgroups. A workgroup reads `k·(bm + bn)` operand words (each of
/// its `bm` A rows and `bn` B rows once, `k` words each) and writes `bm·bn`; across the grid each
/// operand word is read `ceil(cols/bn)` (A) or `ceil(rows/bm)` (B) times.
#[cfg(feature = "gpu")]
pub unsafe fn lin_bf16_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    a: crate::gpu::vulkan::Handle,
    b: crate::gpu::vulkan::Handle,
    c: crate::gpu::vulkan::Handle,
    x: &LinBf16,
    tile: Bf16Tile,
) -> crate::gpu::decode::Step {
    lin_bf16_step(g, a, b, c, x, tile, "lin_bf16")
}

/// [`lin_bf16_dispatch`] under a caller's dispatch name, so a profile attributes the product.
///
/// # Safety
///
/// As [`lin_bf16_dispatch`].
///
/// # Complexity
///
/// As [`lin_bf16_dispatch`].
#[cfg(feature = "gpu")]
pub unsafe fn lin_bf16_step(
    g: &crate::gpu::decode::GpuCtx,
    a: crate::gpu::vulkan::Handle,
    b: crate::gpu::vulkan::Handle,
    c: crate::gpu::vulkan::Handle,
    x: &LinBf16,
    tile: Bf16Tile,
    name: &'static str,
) -> crate::gpu::decode::Step {
    let tile = bf16_tile_for(x, tile);
    let (gx, gy) = x.groups(tile);
    crate::gpu::decode::make_step_named(
        g,
        &coopmat_gemm_bf16_bytes(x, tile),
        &[a, b, c, a, b],
        gx,
        gy,
        name,
    )
}

/// **Linear input adjoint** — `dx[r,c] = Σ_k dy[r·o + k] · w[k·i + c]`: grid `(⌈i/BM⌉, ⌈n/BN⌉)` at
/// the tile [`tile_for`] measures for `i` rows.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_bwd_input_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    dy: crate::gpu::vulkan::Handle,
    w: crate::gpu::vulkan::Handle,
    dx: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    let d = LinBf16::bwd_input(n, i, o);
    lin_bf16_step(g, w, dy, dx, &d, tile_for(d.rows), "linear_bwd_input")
}

/// **Linear weight adjoint** — `dw[r,c] = Σ_k dy[k·o + r] · x[k·i + c]`: grid `(⌈i/BM⌉, ⌈o/BN⌉)` at
/// the tile [`tile_for`] measures for `i` rows.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_bwd_weight_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    dy: crate::gpu::vulkan::Handle,
    x: crate::gpu::vulkan::Handle,
    dw: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    let d = LinBf16::bwd_weight(n, i, o);
    lin_bf16_step(g, x, dy, dw, &d, tile_for(d.rows), "linear_bwd_weight")
}

#[cfg(test)]
mod tests {
    //! The capability probe and the freshness pins of the two committed faces. The bf16 face's
    //! arithmetic, bounds and speed are `tests/gpu_lin_bf16.rs` (it needs a device).

    use std::path::Path;

    #[cfg(feature = "gpu")]
    extern "C" {
        fn dlopen(filename: *const std::os::raw::c_char, flags: i32) -> *mut std::ffi::c_void;
        fn dlsym(
            handle: *mut std::ffi::c_void,
            symbol: *const std::os::raw::c_char,
        ) -> *mut std::ffi::c_void;
    }
    /// glibc `RTLD_NOW` (bits/dlfcn.h)
    #[cfg(feature = "gpu")]
    const RTLD_NOW: i32 = 2;

    /// Compile `src/gpu/spirv/coopmat_gemm.comp` with `args` and compare it to `committed`, whose
    /// path names the flag set: a stale `.spv` is the one failure this pins.
    fn fresh(args: &[&str], committed: &str, name: &str) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
        let out = std::env::temp_dir().join(format!("{name}_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(args)
            .arg(dir.join("coopmat_gemm.comp"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let got = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        let want = std::fs::read(dir.join(committed)).expect("committed spv");
        assert!(
            got == want,
            "src/gpu/spirv/{committed} is stale: recompile with glslc {args:?}"
        );
    }

    /// The chart face — the int8 path prefill runs — is the one committed `coopmat_gemm.spv`: the
    /// bf16 face's `#if OPF` blocks preprocess to nothing for it.
    #[test]
    fn coopmat_spirv_is_the_glsl() {
        fresh(super::GLSLC_ARGS, "coopmat_gemm.spv", "coopmat_chart");
    }

    /// The bf16 face's two committed compiles.
    #[test]
    fn coopmat_bf16_spirv_is_the_glsl() {}

    #[cfg(feature = "gpu")]
    #[repr(C)]
    struct Props {
        s_type: u32,
        p_next: *mut std::ffi::c_void,
        m: u32,
        n: u32,
        k: u32,
        a: u32,
        b: u32,
        c: u32,
        result: u32,
        sat: u32,
        scope: u32,
    }
    #[cfg(feature = "gpu")]
    const ST_PROPS: u32 = 1_000_506_001;

    #[cfg(feature = "gpu")]
    fn comp(t: u32) -> &'static str {
        match t {
            0 => "f16",
            1 => "f32",
            2 => "f64",
            3 => "s8",
            4 => "s16",
            5 => "s32",
            6 => "s64",
            7 => "u8",
            8 => "u16",
            9 => "u32",
            10 => "u64",
            1_000_141_000 => "bf16",
            _ => "other",
        }
    }

    /// Step 1: every `(A, B, C, Result, M, N, K, scope)` the device lists. Skips without Vulkan.
    #[cfg(feature = "gpu")]
    #[test]
    fn coopmat_properties() {
        let vk = match unsafe { crate::gpu::vulkan::Vk::load() } {
            Ok(v) => v,
            Err(e) => {
                println!("SKIP coopmat_properties: {e}");
                return;
            }
        };
        unsafe {
            use crate::gpu::vulkan::{GetProcAddr, PhysicalDevice};
            let lib = dlopen(
                b"libvulkan.so.1\0".as_ptr().cast::<std::os::raw::c_char>(),
                RTLD_NOW,
            );
            assert!(!lib.is_null(), "libvulkan.so.1");
            let p = dlsym(
                lib,
                b"vkGetInstanceProcAddr\0"
                    .as_ptr()
                    .cast::<std::os::raw::c_char>(),
            );
            assert!(!p.is_null(), "vkGetInstanceProcAddr");
            let gipa: GetProcAddr = std::mem::transmute(p);
            let name = std::ffi::CStr::from_bytes_with_nul(
                b"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\0",
            )
            .unwrap();
            let q = gipa(vk.instance, name.as_ptr());
            if q.is_null() {
                println!("coopmat_properties: extension absent");
                return;
            }
            type GetProps = unsafe extern "C" fn(PhysicalDevice, *mut u32, *mut Props) -> i32;
            let get: GetProps = std::mem::transmute(q);
            let mut count = 0u32;
            let r = get(vk.pd, &mut count, std::ptr::null_mut());
            println!(
                "coopmat_properties: device {:?} result {r} count {count}",
                vk.device_name
            );
            if r != 0 || count == 0 {
                return;
            }
            let mut v: Vec<Props> = (0..count)
                .map(|_| Props {
                    s_type: ST_PROPS,
                    p_next: std::ptr::null_mut(),
                    m: 0,
                    n: 0,
                    k: 0,
                    a: 0,
                    b: 0,
                    c: 0,
                    result: 0,
                    sat: 0,
                    scope: 0,
                })
                .collect();
            let r = get(vk.pd, &mut count, v.as_mut_ptr());
            assert!(
                r == 0,
                "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR: {r}"
            );
            v.truncate(count as usize);
            for p in &v {
                println!(
                    "PROP A={} B={} C={} R={} M={} N={} K={} sat={} scope={}",
                    comp(p.a),
                    comp(p.b),
                    comp(p.c),
                    comp(p.result),
                    p.m,
                    p.n,
                    p.k,
                    p.sat,
                    p.scope
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The precision bound of the bf16 face, as the tests of every path that now runs on it state it.

/// `u`: the bf16 unit roundoff, `2^-8` (an 8-bit significand).
pub const BF16_U: f64 = 1.0 / 256.0;

/// The error bound of one `Σ_k a_k·b_k` computed as the face computes it: operands rounded once to
/// the nearest bf16, accumulation in f32.
///
/// One bf16 ulp on `v`, up or down from a seeded hash: the preimages `v(1 ± 2^-8)` bracket every
/// bf16 value within one ulp of `v`, and the device rounds each staged operand to the nearest of
/// them (at most half an ulp, so inside this population).
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn bf16_ulp1(v: f32, seed: u64, i: usize) -> f32 {
    let mut h = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add((i as u64).wrapping_mul(1_442_695_040_888_963_407));
    h ^= h >> 33;
    let up = (h & 1) == 0;
    let s = 1.0 + BF16_U as f32;
    if up {
        v * s
    } else {
        v / s
    }
}

// ---------------------------------------------------------------------------------------------
// The rounded-operand host model: what a host must round to be comparable with the bf16 face.

/// `v` rounded to the nearest bf16 (`u = 2^-8`), in f64. The device rounds every staged operand
/// once, to nearest; a host model that skips this measures its own exactness, not the face.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn bf16_round(v: f64) -> f64 {
    let x = v as f32;
    if !x.is_finite() || x == 0.0 {
        return f64::from(x);
    }
    let bits = x.to_bits();
    // round to nearest even on the 16 kept bits (bf16 is f32's top half)
    let keep = 0xFFFF_0000u32;
    let tail = bits & !keep;
    let half = 0x0000_8000u32;
    let lsb = (bits >> 16) & 1;
    let up = tail > half || (tail == half && lsb == 1);
    let rounded = (bits & keep).wrapping_add(u32::from(up) << 16);
    f64::from(f32::from_bits(rounded))
}

/// [`bf16_round`] over a slice.
///
/// # Complexity
///
/// `O(n)` time and space.
#[must_use]
pub fn bf16_round_slice(v: &[f64]) -> Vec<f64> {
    v.iter().copied().map(bf16_round).collect()
}
