//! The KD face of the engine attention kernel: `src/gpu/spirv/attn_coopmat_kd.spv`, compiled from
//! `src/gpu/spirv/attn_coopmat.comp` with `-DKDF=1` (one `.comp`, two faces, one GPU path).
//!
//! The face is NanoQuant's KD attention forward: hd 128, bf16 x bf16 -> f32 M16N16K16, f32 global
//! tensors in the KD layout (`q [T][NH*HD]`, `k`/`v` `[T][NKV*HD]`, `out [T][NH*HD]`), causal over
//! the whole sequence, `scale*log2e` folded with `exp2` as `iterative_softmax`, P rounded once to
//! bf16, O f32, `logsumexp [NH][T]` f32 for the backward face, and a gated epilogue under a spec
//! constant. The reference is `mem_eff_attention`'s bf16 causal forward, instance
//! `kernels/cutlassF.h:20` (`AttentionKernel<bfloat16_t, Sm80, true, 64, 128, 128, true, true>`,
//! torch 2.8.0+cu128): key-block loop `kernel_forward.h:700`, MM0 = Q.K^T :747-801 with the scale
//! folded :798-801, `iterative_softmax` :1182-1290 (running max, exp2 rescale :1257), MM1 = P.V
//! :1004-1140, logsumexp :1160-1178.
//!
//! Deviation from the reference, stated: the reference's q/k/v are bf16 in memory; here they are f32
//! and are rounded once at the shared store, so both MMAs see the same bf16 operands.

use super::prefill::attn_coop_specialize;
use crate::gpu::decode::{make_step_named, GpuCtx, Step};
use crate::gpu::nn::bytes;
use crate::gpu::vulkan::Handle;

/// Query rows per workgroup (`BR = 16*NSG` at the committed `NSG = 4`).
pub const ATTN_KD_ROWS: u32 = 64;
/// Keys per staged slab (`BK`).
pub const ATTN_KD_KEYS: u32 = 64;
/// Head width the face is compiled for (`HD`).
pub const ATTN_KD_HD: u32 = 128;
/// Threads per workgroup (`NT = 32*NSG`); `attn_coopmat_kd_module` asserts the compiled face's
/// `LocalSize` against it.
pub const ATTN_KD_THREADS: u32 = 128;

/// The face's static shared-memory footprint in bytes at [`ATTN_KD_KEYS`] staged keys and
/// [`ATTN_KD_ROWS`] rows, from `src/gpu/spirv/attn_coopmat.comp`'s padded strides: `sKV`
/// `BK*SKV4` bf16vec4 at `SKV4 = 2*(BK/4) + 2`, `sP` `BR*SP4` at `SP4 = BK/4 + 2`, `sQ` `BR*SQ4` at
/// `SQ4 = 6`, `sS` `BR*SSTR` f32 at `SSTR = BK + 4`, `sAl` `BR` f32. It must not exceed the
/// device's `maxComputeSharedMemorySize` (49152 B here), which the dispatch asserts.
///
/// Cost: O(1).
pub fn attn_kd_shared(keys: u32, rows: u32) -> u32 {
    let (skv4, sp4, sq4, sstr) = (2 * (keys / 4) + 2, keys / 4 + 2, 6, keys + 4);
    rows * sq4 * 8 + keys * skv4 * 8 + rows * sp4 * 8 + rows * sstr * 4 + rows * 4
}

/// The module's declared `LocalSize` `x`: `OpExecutionMode` (`16`) with `LocalSize` (`17`).
///
/// Cost: one linear pass over the `L` words of the module, O(L) time and O(1) space.
fn kd_local_size(module: &[u32]) -> u32 {
    const OP_EXECUTION_MODE: u32 = 16;
    const MODE_LOCAL_SIZE: u32 = 17;
    let mut i = 5;
    while i < module.len() {
        let (len, op) = ((module[i] >> 16) as usize, module[i] & 0xffff);
        if len == 0 || i + len > module.len() {
            return 0;
        }
        if op == OP_EXECUTION_MODE && module[i + 2] == MODE_LOCAL_SIZE {
            return module[i + 3];
        }
        i += len;
    }
    0
}

/// The compiled KD face (`glslc` [`crate::gpu::nn::GLSLC_ARGS`] plus `-DKDF=1` over
/// `src/gpu/spirv/attn_coopmat.comp`), specialization defaults unpatched.
pub static ATTN_COOPMAT_KD_SPV: &[u8] = include_bytes!("../../spirv/attn_coopmat_kd.spv");

/// The `glslc` invocation that produced [`ATTN_COOPMAT_KD_SPV`].
pub const GLSLC_ARGS_KD: &[&str] = &[
    "--target-env=vulkan1.1",
    "--target-spv=spv1.3",
    "-O",
    "-fshader-stage=compute",
    "-DKDF=1",
];

/// [`ATTN_COOPMAT_KD_SPV`] with the spec constants patched: `(NH, NKV, T, SCALE, GATED)` at ids
/// 0..4. Panics on `nkv == 0`, an `nkv` that does not divide `nh`, or a non-finite `scale`.
///
/// Cost: two linear passes over the module's word stream, O(words) time, O(words) space for the
/// patched copy.
pub fn attn_coopmat_kd_module(nh: u32, nkv: u32, t: u32, scale: f32, gated: bool) -> Vec<u32> {
    assert!(nh > 0 && nkv > 0 && nh % nkv == 0, "GQA needs nkv | nh");
    assert!(t > 0 && scale.is_finite(), "T and the scale must be real");
    assert!(
        ATTN_COOPMAT_KD_SPV.len() % 4 == 0,
        "SPIR-V is a word stream"
    );
    let words: Vec<u32> = ATTN_COOPMAT_KD_SPV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0], crate::gpu::simt::SPV_MAGIC, "not a SPIR-V module");
    // The face's tile shape has one source: the compiled workgroup must be `ATTN_KD_THREADS`, so a
    // `.comp` edit that moves it is a different kernel, not a wider one.
    assert_eq!(
        kd_local_size(&words),
        ATTN_KD_THREADS,
        "the KD face's LocalSize moved away from ATTN_KD_THREADS"
    );
    attn_coop_specialize(
        &words,
        &[
            (0, nh),
            (1, nkv),
            (2, t),
            (3, scale.to_bits()),
            (4, u32::from(gated)),
        ],
    )
}

/// [`attn_coopmat_kd_module`] as bytes.
///
/// Cost: `attn_coopmat_kd_module`'s O(words) passes plus one O(words) byte copy.
pub fn attn_coopmat_kd_bytes(nh: u32, nkv: u32, t: u32, scale: f32, gated: bool) -> Vec<u8> {
    bytes(attn_coopmat_kd_module(nh, nkv, t, scale, gated))
}

/// The KD attention forward as one recorded step, dispatch `(nh, ceil(t / ATTN_KD_ROWS))`.
///
/// Bindings, in order: 0 `q` `[T][NH*HD]` f32, 1 `k` `[T][NKV*HD]` f32, 2 `v` `[T][NKV*HD]` f32,
/// 3 `gate` `[T][NH*HD]` f32 (read only when `gated`; pass a one-word buffer when it is not),
/// 4 `out` `[T][NH*HD]` f32, 5 `lse` `[NH][T]` f32.
///
/// # Safety
///
/// The handles are live SSBOs of those lengths and outlive the submission.
///
/// # Complexity
///
/// One `vkCreateShaderModule`/`vkCreateComputePipelines` (the caller's cost, not per dispatch) and
/// `O(ceil(t/64))` workgroups; the device work is `O(nh·t²·hd / 16)` MMA lane-ops over the causal
/// half ([`attn_kd_shared`] bytes of shared, `O(hd)` registers of accumulators).
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_kd_dispatch(
    g: &GpuCtx,
    q: Handle,
    k: Handle,
    v: Handle,
    gate: Handle,
    out: Handle,
    lse: Handle,
    nh: u32,
    nkv: u32,
    t: u32,
    scale: f32,
    gated: bool,
) -> Step {
    // The face is compiled for a fixed shared footprint; a device under it is refused here rather
    // than on the first `vkCreateComputePipelines` error.
    let shared = attn_kd_shared(ATTN_KD_KEYS, ATTN_KD_ROWS);
    assert!(
        g.vk.max_shared >= shared,
        "the KD face needs {shared} B of shared, the device reports {}",
        g.vk.max_shared
    );
    let spv = attn_coopmat_kd_bytes(nh, nkv, t, scale, gated);
    let tiles = t.div_ceil(ATTN_KD_ROWS);
    make_step_named(g, &spv, &[q, k, v, gate, out, lse], nh, tiles, "attn_kd")
}

/// The KD backward face: `src/gpu/spirv/attn_coopmat.comp` compiled with `-DKDB=1`, the second face
/// of the one attention kernel.
///
/// One workgroup = one (kv head, [`ATTN_KDB_KEYS`]-key block), grid `(NKV, ceil(T/ATTN_KDB_KEYS))`,
/// walking every query block at or after its key block once per query head of its kv head.
/// `delta = rowsum(dy ⊙ out)` is folded into the walk (reference `computeDelta`,
/// `kernel_backward.h:2499`), `S = K Q^T`, `P = exp2(S·SCALE·log2e − lse·log2e)`, `dP = V dA^T`,
/// `dS = P (dP − delta)·SCALE`, `dK += dS^T Q`, `dV += P^T dA` — the first two accumulated in f32 in
/// their own output buffers by the one workgroup that owns their rows (the reference's
/// `kOutputInRF == false` gmem accumulators, `:745`) — and `dQ` accumulated in the reference's
/// `GradQTempStorage` lock/counter tile (`:159-185`, `:622`).
///
/// Tile [`ATTN_KDB_KEYS`] keys × [`ATTN_KDB_QUERIES`] queries, four subgroups, each owning the
/// 32×32 quadrant of `S`, `P`, `dP` and `dS` at `(keys 32·(sg>>1).., queries 32·(sg&1)..)`, and the
/// rows `[32·(sg>>1)..)` × the 64 head dims `[64·(sg&1)..)` of the three output gradients.
pub static ATTN_COOPMAT_KDB_SPV: &[u8] = include_bytes!("../../spirv/attn_coopmat_kd_bwd.spv");

/// The `glslc` invocation that produced [`ATTN_COOPMAT_KDB_SPV`].
pub const GLSLC_ARGS_KDB: &[&str] = &[
    "--target-env=vulkan1.1",
    "--target-spv=spv1.3",
    "-O",
    "-fshader-stage=compute",
    "-DKDB=1",
];

/// Keys per staged block of the backward face (`KBB`, the reference's `kBlockSizeJ`).
pub const ATTN_KDB_KEYS: u32 = 64;
/// Queries per staged block of the backward face (`QBB`, the reference's `kBlockSizeI`).
pub const ATTN_KDB_QUERIES: u32 = 64;
/// Threads per workgroup of the backward face (`NT = 32·4`: four subgroups, one 32×32 quadrant
/// each).
pub const ATTN_KDB_THREADS: u32 = 128;

/// The backward face's static shared footprint in bytes at [`ATTN_KDB_KEYS`] keys and
/// [`ATTN_KDB_QUERIES`] queries, from the strides in `src/gpu/spirv/attn_coopmat.comp`'s `KDB`
/// block: two 64×`HD` bf16 operand tiles at `SB = HD/4 + 2`, the 64×64 bf16 exchange tile at
/// `XB = QBB/4 + 2`, one 16×16 f32 scratch per subgroup, delta and lse.
///
/// Cost: O(1).
pub fn attn_kd_bwd_shared(keys: u32) -> u32 {
    let (sb, xb) = (ATTN_KD_HD / 4 + 2, ATTN_KDB_QUERIES / 4 + 2);
    2 * keys * sb * 8 + keys * xb * 8 + 4 * 16 * 16 * 4 + 2 * ATTN_KDB_QUERIES * 4 + 4
}

/// Words of the `qtmp` scratch: one [`ATTN_KDB_QUERIES`]-query × `HD` f32 tile per (query head,
/// query block). The face clears each tile as it writes it out, so the buffer is reusable across
/// submissions once zeroed.
///
/// Cost: O(1).
pub fn attn_kdb_qtmp_len(nh: u32, t: u32) -> usize {
    let tiles = (nh * t.div_ceil(ATTN_KDB_QUERIES)) as usize;
    tiles * (ATTN_KDB_QUERIES * ATTN_KD_HD) as usize
}

/// [`ATTN_COOPMAT_KDB_SPV`] with the spec constants patched: `(NH, NKV, T, SCALE, GATED)` at ids
/// 0..4, exactly as [`attn_coopmat_kd_module`] patches the forward face.
///
/// Cost: two linear passes over the module's word stream, O(words) time and space.
pub fn attn_kd_bwd_module(nh: u32, nkv: u32, t: u32, scale: f32, gated: bool) -> Vec<u32> {
    assert!(nh > 0 && nkv > 0 && nh % nkv == 0, "GQA needs nkv | nh");
    assert!(t > 0 && scale.is_finite(), "T and the scale must be real");
    let words: Vec<u32> = ATTN_COOPMAT_KDB_SPV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0], crate::gpu::simt::SPV_MAGIC, "not a SPIR-V module");
    assert_eq!(
        kd_local_size(&words),
        ATTN_KDB_THREADS,
        "the KDB face's LocalSize moved away from ATTN_KDB_THREADS"
    );
    attn_coop_specialize(
        &words,
        &[
            (0, nh),
            (1, nkv),
            (2, t),
            (3, scale.to_bits()),
            (4, u32::from(gated)),
        ],
    )
}

/// [`attn_kd_bwd_module`] as bytes.
///
/// Cost: `attn_kd_bwd_module`'s O(words) passes plus one O(words) byte copy.
pub fn attn_kd_bwd_bytes(nh: u32, nkv: u32, t: u32, scale: f32, gated: bool) -> Vec<u8> {
    bytes(attn_kd_bwd_module(nh, nkv, t, scale, gated))
}

/// The KD attention backward as one recorded step, dispatch `(nkv, ceil(t / ATTN_KDB_KEYS))`.
///
/// Bindings, in order: 0 `q`, 1 `k`, 2 `v` `[T][NKV*HD]`, 3 `dy` `[T][NH*HD]`, 4 `out`
/// `[T][NH*HD]` (the KDF face's stored output, source of `delta`), 5 `gate` `[T][NH*HD]` (read only
/// when `gated`), 6 `lse` `[NH][T]` (the KDF face's), 7 `dq` `[T][NH*HD]` f32 (written by the last
/// contributor of each tile, not accumulated), 8 `dk`, 9 `dv` `[T][NKV*HD]` f32 (each **added to**:
/// the face holds the reference's gmem accumulators there, so both must be zero before the first
/// submission and are complete only after the one dispatch), 10 `dgate` `[T][NH*HD]` (written only
/// when `gated`; pass a one-word buffer when it is not), 11 `qlock`, 12 `qcnt`
/// `[NH*ceil(T/ATTN_KDB_QUERIES)]` u32 zeroed, 13 `qtmp` [`attn_kdb_qtmp_len`] f32 zeroed.
///
/// # Safety
///
/// The handles are live SSBOs of those lengths and outlive the submission; `qlock`, `qcnt` and
/// `qtmp` are zero-filled, and `dq`/`dk`/`dv` are writable, `dk`/`dv` zeroed.
///
/// # Complexity
///
/// One pipeline creation (the caller's) and `O(nkv · ceil(t/ATTN_KDB_KEYS))` workgroups; the device
/// work is `O(nh·t²·hd/16)` MMA lane-ops over the causal half and [`attn_kd_bwd_shared`] bytes of
/// shared.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_kd_bwd_dispatch(
    g: &GpuCtx,
    q: Handle,
    k: Handle,
    v: Handle,
    dy: Handle,
    out: Handle,
    gate: Handle,
    lse: Handle,
    dq: Handle,
    dk: Handle,
    dv: Handle,
    dgate: Handle,
    qlock: Handle,
    qcnt: Handle,
    qtmp: Handle,
    nh: u32,
    nkv: u32,
    t: u32,
    scale: f32,
    gated: bool,
) -> Step {
    let shared = attn_kd_bwd_shared(ATTN_KDB_KEYS);
    assert!(
        g.vk.max_shared >= shared,
        "the KDB face needs {shared} B of shared, the device reports {}",
        g.vk.max_shared
    );
    let spv = attn_kd_bwd_bytes(nh, nkv, t, scale, gated);
    let blocks = t.div_ceil(ATTN_KDB_KEYS);
    make_step_named(
        g,
        &spv,
        &[
            q, k, v, dy, out, gate, lse, dq, dk, dv, dgate, qlock, qcnt, qtmp,
        ],
        nkv,
        blocks,
        "attn_kd_bwd",
    )
}

#[cfg(test)]
mod kdb_fresh {
    use super::*;

    /// The committed `attn_coopmat_kd_bwd.spv` is what `glslc` makes of the committed GLSL: one
    /// `.comp`, three faces, each pinned by its own bytes.
    #[test]
    fn kdb_spv_is_fresh() {
        let out = std::env::temp_dir().join("attn_coopmat_kd_bwd_fresh.spv");
        let st = std::process::Command::new("glslc")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args(GLSLC_ARGS_KDB)
            .arg("-o")
            .arg(&out)
            .arg("src/gpu/spirv/attn_coopmat.comp")
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let fresh = std::fs::read(&out).expect("glslc output");
        assert!(
            fresh == ATTN_COOPMAT_KDB_SPV,
            "src/gpu/spirv/attn_coopmat_kd_bwd.spv is stale: recompile with glslc {GLSLC_ARGS_KDB:?}"
        );
    }

    /// The KDB face's shared footprint is static and must fit the device's 49152 B.
    #[test]
    fn kdb_shared_fits() {
        let b = attn_kd_bwd_shared(ATTN_KDB_KEYS);
        assert_eq!(b, 48_644, "attn_kd_bwd_shared moved");
        assert!(b <= 49_152);
    }
}

#[cfg(test)]
#[allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    missing_docs,
    reason = "measurement test: a fixture generator, an f64 oracle and the timed campaign"
)]
pub mod tests;
