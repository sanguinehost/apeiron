//! simt_nn::bwd::coop — the cooperative-matrix (tensor-core) twins of the three dense-layer GEMMs:
//! `linear_fwd_coop` beside `fwd::lin::linear_fwd`, `linear_bwd_input_coop` and
//! `linear_bwd_weight_coop` beside [`super::lin`].
//!
//! **One emitted body, three operand descriptions**, as in [`super::lin`]: a workgroup owns a
//! `BM × BN` output tile, `BK`-wide reduction slices stage into two shared operand tiles, and
//! `SGM × SGN` subgroups each hold `WM × WN` `16×16` cooperative-matrix fragments for the whole
//! reduction. Every output is `C[m][n] = Σ_k A[m][k]·B[k][n]`, so only *where* `A` and `B` live
//! changes per product: the forward product reads `x[m][k]` and `w[n][k]` (k contiguous in both),
//! the input adjoint `dy[m][k]` and `w[k][n]` (n contiguous in `w`), the weight adjoint
//! `dy[k][m]` and `x[k][n]`. A `ColumnMajor` fragment load covers every operand whose *output*
//! index is its contiguous global one, so each staged read is one contiguous 16-byte `vec4` and
//! there is no transpose kernel.
//!
//! **Precision.** A/B are `bfloat16_t` on the shared-memory load and the accumulator is `float`:
//! the device lists `bf16·bf16 → f32` `M16N16K16` at subgroup scope ([`tests::coopmat_properties`]),
//! the reference's own dtype. The buffers in and out are f32 and the dispatch signatures are the
//! f32 kernels'; the conversion is the shared-tile store, the accumulation is f32. The measured
//! cost is one rounding of each operand to bf16 (8-bit significand) against the f32 body, and
//! [`tests::coop_error`] prints the relative Frobenius error against an f64 host product per shape.
//!
//! **Source.** GLSL `src/gpu/spirv/train/lin_coop.comp`, compiled once by `glslc` with
//! [`crate::gpu::nn::GLSLC_ARGS`] into the committed `lin_coop.spv`; [`tests::lin_coop_spirv_is_the_glsl`]
//! recompiles it and requires byte identity. The per-shape constants (`COLS`, `KD`, `RSA`, `RSB`,
//! `ACOL`, `BCOL`) are `OpSpecConstant`s this module rewrites in the binary, so the pipeline needs
//! no `VkSpecializationInfo` and no push constants — the `coopmat_gemm` mechanism.
//!
//! **Preconditions** (asserted here): `rows % BM == 0`, `cols % BN == 0`, `kdim % BK == 0`. The
//! five training shapes are all multiples of 1024, so every tile is full and no bound check exists
//! in the kernel.

/// Rows of the output tile a workgroup owns (the shader's `BM`).
pub const BM: u32 = 64;
/// Columns of the output tile (the shader's `BN`).
pub const BN: u32 = 64;
/// Reduction width staged per barrier pair (the shader's `BK`).
pub const BK: u32 = 32;
/// Workgroup width in lanes: `SGM·SGN = 4` subgroups (the shader's `NT`).
pub const NT: u32 = 128;

/// The compiled kernel, specialization defaults unpatched.
pub static LIN_COOP_SPV: &[u8] = include_bytes!("../../spirv/train/lin_coop.spv");

const OP_DECORATE: u32 = 71;
const OP_SPEC_CONSTANT: u32 = 50;
const DEC_SPEC_ID: u32 = 1;
const SPEC_COLS: u32 = 0;
const SPEC_KD: u32 = 1;
const SPEC_RSA: u32 = 2;
const SPEC_RSB: u32 = 3;
const SPEC_ACOL: u32 = 4;
const SPEC_BCOL: u32 = 5;

/// Rewrite the default value of every 32-bit `OpSpecConstant` whose `SpecId` is a key of `vals`.
/// Panics when a requested `SpecId` is absent: a constant left at its default is a different
/// kernel.
///
/// Cost: two linear passes over the module, `O(words)` time, `O(1)` space beyond the copy.
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

/// One operand's description: its row stride in f32 words and its order (`0` = `[row][k]`,
/// `1` = `[k][row]`, the fragment layout the shared tile is read with).
struct Weld {
    /// the stride in f32 words between consecutive rows of the operand's outer dimension
    rs: u32,
    /// `0` = `[row][k]`, `1` = `[k][row]`
    col: u32,
}

/// **The one emitted body**: `rows × cols` output, reduction length `kdim`, `BM × BN` tile per
/// workgroup, `BK`-wide slice. Bindings are the caller's: `a.buf`, `b.buf`, `out`, in that order.
///
/// Complexity: `(rows/BM)·(cols/BN)` workgroups; per workgroup per reduction step
/// `(BM + BN)·BK` halves staged (one coalesced `vec4` read per 4 halves) and
/// `BM·BN·BK` multiply-accumulates issued as `SGM·SGN·WM·WN·BK/16` cooperative-matrix multiplies;
/// shared words are `(BM + BN)·BK` halves, `O(1)` beyond the staged tiles.
fn lin_coop_module(a: &Weld, b: &Weld, rows: u32, cols: u32, kdim: u32) -> Vec<u32> {
    assert!(
        rows > 0 && cols > 0 && kdim > 0,
        "lin_coop: shape must be non-zero"
    );
    assert!(
        rows % BM == 0 && cols % BN == 0 && kdim % BK == 0,
        "lin_coop: rows % {BM}, cols % {BN}, kdim % {BK} must all be 0 (shape {rows}x{cols}x{kdim})"
    );
    let words: Vec<u32> = LIN_COOP_SPV
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    specialize(
        &words,
        &[
            (SPEC_COLS, cols),
            (SPEC_KD, kdim),
            (SPEC_RSA, a.rs),
            (SPEC_RSB, b.rs),
            (SPEC_ACOL, a.col),
            (SPEC_BCOL, b.col),
        ],
    )
}

/// **Linear forward (tensor cores)** — `fwd::lin::linear_fwd`'s twin: `x` is `n × i`, `w` is
/// `o × i`, `y` is `n × o`, `y[r,c] = Σ_k x[r·i + k]·w[c·i + k]`. Both operands are k-inner, so
/// both fragment loads are `RowMajor`.
///
/// Bindings: 0 = x (f32, n×i), 1 = w (f32, o×i), 2 = y (f32, n×o).
pub fn linear_fwd_coop_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    let a = Weld { rs: i, col: 0 };
    let b = Weld { rs: i, col: 1 };
    lin_coop_module(&a, &b, n, o, i)
}

/// [`linear_fwd_coop_module`] as bytes.
pub fn linear_fwd_coop_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    let m = linear_fwd_coop_module(n, i, o);
    m.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// **Linear input adjoint (tensor cores)** — `bwd::lin::linear_bwd_input`'s twin: `dy` is `n × o`,
/// `w` is `o × i`, `dx` is `n × i`, `dx[r,c] = Σ_k dy[r·o + k]·w[k·i + c]`. `dy` is k-inner
/// (`RowMajor`), `w` is n-inner (`ColumnMajor`).
///
/// Bindings: 0 = dy (f32, n×o), 1 = w (f32, o×i), 2 = dx (f32, n×i).
pub fn linear_bwd_input_coop_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    let a = Weld { rs: o, col: 0 };
    let b = Weld { rs: i, col: 0 };
    lin_coop_module(&a, &b, n, i, o)
}

/// [`linear_bwd_input_coop_module`] as bytes.
pub fn linear_bwd_input_coop_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    let m = linear_bwd_input_coop_module(n, i, o);
    m.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// **Linear weight adjoint (tensor cores)** — `bwd::lin::linear_bwd_weight`'s twin: `dy` is
/// `n × o`, `x` is `n × i`, `dw` is `o × i`, `dw[r,c] = Σ_k dy[k·o + r]·x[k·i + c]`. Both
/// operands are output-inner (`ColumnMajor`), the reduction running over `n`.
///
/// Bindings: 0 = dy (f32, n×o), 1 = x (f32, n×i), 2 = dw (f32, o×i).
pub fn linear_bwd_weight_coop_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    let a = Weld { rs: o, col: 1 };
    let b = Weld { rs: i, col: 0 };
    lin_coop_module(&a, &b, o, i, n)
}

/// [`linear_bwd_weight_coop_module`] as bytes.
pub fn linear_bwd_weight_coop_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    let m = linear_bwd_weight_coop_module(n, i, o);
    m.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Host wrapper: the tensor-core linear forward. `x` is the device buffer of `n·i` f32, `w` of
/// `o·i`, `y` of `n·o`. Grid `(n/BM · o/BN, 1, 1)`, LocalSize [`NT`].
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_fwd_coop_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    x: crate::gpu::vulkan::Handle,
    w: crate::gpu::vulkan::Handle,
    y: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &linear_fwd_coop_bytes(n, i, o),
        &[x, w, y],
        (n / BM) * (o / BN),
        1,
        "linear_fwd_coop",
    )
}

/// Host wrapper: the tensor-core linear input adjoint. `dy` is `n·o` f32, `w` of `o·i`, `dx` of
/// `n·i`. Grid `(n/BM · i/BN, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_bwd_input_coop_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    dy: crate::gpu::vulkan::Handle,
    w: crate::gpu::vulkan::Handle,
    dx: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &linear_bwd_input_coop_bytes(n, i, o),
        &[dy, w, dx],
        (n / BM) * (i / BN),
        1,
        "linear_bwd_input_coop",
    )
}

/// Host wrapper: the tensor-core linear weight adjoint. `dy` is `n·o` f32, `x` of `n·i`, `dw` of
/// `o·i`. Grid `(o/BM · i/BN, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_bwd_weight_coop_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    dy: crate::gpu::vulkan::Handle,
    x: crate::gpu::vulkan::Handle,
    dw: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &linear_bwd_weight_coop_bytes(n, i, o),
        &[dy, x, dw],
        (o / BM) * (i / BN),
        1,
        "linear_bwd_weight_coop",
    )
}

#[cfg(test)]
mod tests {
    use crate::gpu::nn::GLSLC_ARGS;
    use std::path::Path;

    extern "C" {
        fn dlopen(filename: *const std::os::raw::c_char, flags: i32) -> *mut std::ffi::c_void;
        fn dlsym(
            handle: *mut std::ffi::c_void,
            symbol: *const std::os::raw::c_char,
        ) -> *mut std::ffi::c_void;
    }
    /// glibc `RTLD_NOW` (bits/dlfcn.h)
    const RTLD_NOW: i32 = 2;

    /// The committed SPIR-V is what `glslc` makes of the committed GLSL.
    #[test]
    fn lin_coop_spirv_is_the_glsl() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv/train");
        let out = std::env::temp_dir().join(format!("lin_coop_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(dir.join("lin_coop.comp"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let fresh = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        assert!(fresh == super::LIN_COOP_SPV, "lin_coop.spv is stale");
    }

    /// `VkCooperativeMatrixPropertiesKHR` (KHR layout).
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
    const ST_PROPS: u32 = 1_000_506_001;

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

#[cfg(test)]
mod cases {
    use crate::gpu::decode::{rec_step, Step};
    use crate::gpu::vulkan::*;
    use std::time::Instant;

    /// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-1, 1)`.
    pub(super) struct Lcg(pub u64);

    impl Lcg {
        pub(super) fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            u * 2.0 - 1.0
        }
    }

    /// Copy `count` f32 from a device buffer.
    ///
    /// # Safety
    /// `src` must hold at least `count` f32; the readback buffer is allocated here.
    pub(super) unsafe fn read_f32(vk: &Vk, src: Handle, count: usize) -> Vec<f32> {
        let (rb, m) = vk
            .buffer(
                count as u64 * 4,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .expect("readback buffer");
        let p = vk.map(m).cast::<f32>();
        let cb = vk.command_buffer();
        vk.begin(cb);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_TRANSFER,
            ACCESS_SHADER_WRITE,
            ACCESS_TRANSFER_READ,
        );
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (count * 4) as u64,
        };
        (vk.copy_buffer)(cb, src, rb, 1, &region);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_HOST,
            ACCESS_TRANSFER_WRITE,
            ACCESS_HOST_READ,
        );
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        std::slice::from_raw_parts(p, count).to_vec()
    }

    /// Record `steps` between two GPU timestamps and time begin → `submit_wait`.
    /// Returns `(wall us, device us)`.
    ///
    /// # Safety
    /// Vulkan submission over caller-owned handles; the steps' buffers must outlive the call.
    pub(super) unsafe fn timed(vk: &Vk, steps: &[&Step], pool: &QueryPool) -> (f64, f64) {
        let cb = vk.command_buffer();
        let t0 = Instant::now();
        vk.begin(cb);
        vk.cmd_reset_timestamps(cb, pool);
        vk.cmd_write_timestamp(cb, pool, STAGE_ALL_COMMANDS, 0);
        for step in steps {
            rec_step(vk, cb, step);
        }
        vk.cmd_write_timestamp(cb, pool, STAGE_ALL_COMMANDS, 1);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        let wall = t0.elapsed().as_secs_f64() * 1e6;
        let ts = vk.read_timestamps(pool);
        let ticks = ts[1].wrapping_sub(ts[0]) as f64;
        (wall, ticks * f64::from(vk.timestamp_period) / 1000.0)
    }

    /// `(mean, min, max)` of the device interval over `reps` runs, in µs.
    ///
    /// # Safety
    /// As [`timed`].
    pub(super) unsafe fn series(
        vk: &Vk,
        steps: &[&Step],
        pool: &QueryPool,
        reps: usize,
    ) -> (f64, f64, f64) {
        for _ in 0..2 {
            timed(vk, steps, pool);
        }
        let mut dev = Vec::with_capacity(reps);
        for _ in 0..reps {
            let (_, d) = timed(vk, steps, pool);
            dev.push(d);
        }
        let mean = dev.iter().sum::<f64>() / dev.len() as f64;
        let min = dev.iter().copied().fold(f64::INFINITY, f64::min);
        let max = dev.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (mean, min, max)
    }
}

#[cfg(test)]
mod kernels {
    use super::cases::{read_f32, series, Lcg};
    use crate::gpu::decode::GpuCtx;
    use crate::gpu::vulkan::Vk;

    /// The five Qwen3-0.6B dense-layer `(i, o)` pairs at the training token count.
    const SHAPES: [(usize, usize); 5] = [
        (1024, 3072),
        (3072, 1024),
        (1024, 2048),
        (2048, 1024),
        (1024, 1024),
    ];
    /// Tokens of the f64 correctness check (`rows % 64`, `kdim % 32` hold at 64).
    const NC: usize = 64;
    /// Tokens of the timing campaign.
    const NT: usize = 2048;
    /// Reported repeats per cell.
    const REPS: usize = 5;

    /// `||got − ref||_F / ||ref||_F`.
    fn rel_err(got: &[f32], want: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (g, w) in got.iter().zip(want) {
            let d = f64::from(*g) - w;
            num += d * d;
            den += w * w;
        }
        (num / den).sqrt()
    }

    /// `y[m][c] = Σ_k x[m·i+k]·w[c·i+k]` (forward).
    fn ref_fwd(x: &[f32], w: &[f32], n: usize, i: usize, o: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; n * o];
        for m in 0..n {
            for c in 0..o {
                let mut s = 0.0f64;
                for k in 0..i {
                    s += f64::from(x[m * i + k]) * f64::from(w[c * i + k]);
                }
                y[m * o + c] = s;
            }
        }
        y
    }

    /// `dx[m][c] = Σ_k dy[m·o+k]·w[k·i+c]` (input adjoint).
    fn ref_dx(dy: &[f32], w: &[f32], n: usize, i: usize, o: usize) -> Vec<f64> {
        let mut dx = vec![0.0f64; n * i];
        for m in 0..n {
            for c in 0..i {
                let mut s = 0.0f64;
                for k in 0..o {
                    s += f64::from(dy[m * o + k]) * f64::from(w[k * i + c]);
                }
                dx[m * i + c] = s;
            }
        }
        dx
    }

    /// `dw[r][c] = Σ_k dy[k·o+r]·x[k·i+c]` (weight adjoint).
    fn ref_dw(dy: &[f32], x: &[f32], n: usize, i: usize, o: usize) -> Vec<f64> {
        let mut dw = vec![0.0f64; o * i];
        for r in 0..o {
            for c in 0..i {
                let mut s = 0.0f64;
                for k in 0..n {
                    s += f64::from(dy[k * o + r]) * f64::from(x[k * i + c]);
                }
                dw[r * i + c] = s;
            }
        }
        dw
    }

    /// Every shape and all three products against the f64 host product on seeded f32 inputs.
    #[test]
    fn coop_matches_f64() {
        let vk = match unsafe { Vk::load() } {
            Ok(v) => v,
            Err(e) => {
                println!("SKIP coop_matches_f64: {e}");
                return;
            }
        };
        let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx");
        let pool = unsafe { vk.timestamp_pool(2) };
        let mut rng = Lcg(0x5eed_0fc0_0f1e_2d3c);
        let mut worst = 0.0f64;
        for &(i, o) in &SHAPES {
            let n = NC;
            let gen = |len: usize, rng: &mut Lcg| -> Vec<f32> {
                (0..len).map(|_| (0.5 * rng.unit()) as f32).collect()
            };
            let x = gen(n * i, &mut rng);
            let w = gen(o * i, &mut rng);
            let dy = gen(n * o, &mut rng);
            unsafe {
                let xb = g.up_f32(&x);
                let wb = g.up_f32(&w);
                let dyb = g.up_f32(&dy);
                let yb = g.dev(n * o * 4);
                let dxb = g.dev(n * i * 4);
                let dwb = g.dev(o * i * 4);
                let n32 = n as u32;
                let i32 = i as u32;
                let o32 = o as u32;

                let s = super::linear_fwd_coop_dispatch(&g, xb, wb, yb, n32, i32, o32);
                super::cases::timed(&vk, &[&s], &pool);
                let e = rel_err(&read_f32(&vk, yb, n * o), &ref_fwd(&x, &w, n, i, o));
                println!("ERR fwd n={n} i={i} o={o} rel {e:.3e}");
                worst = worst.max(e);
                assert!(e < 4e-3, "fwd n={n} i={i} o={o}: rel {e:.3e} >= 4e-3");

                let s = super::linear_bwd_input_coop_dispatch(&g, dyb, wb, dxb, n32, i32, o32);
                super::cases::timed(&vk, &[&s], &pool);
                let e = rel_err(&read_f32(&vk, dxb, n * i), &ref_dx(&dy, &w, n, i, o));
                println!("ERR dx n={n} i={i} o={o} rel {e:.3e}");
                worst = worst.max(e);
                assert!(e < 4e-3, "dx n={n} i={i} o={o}: rel {e:.3e} >= 4e-3");

                let s = super::linear_bwd_weight_coop_dispatch(&g, dyb, xb, dwb, n32, i32, o32);
                super::cases::timed(&vk, &[&s], &pool);
                let e = rel_err(&read_f32(&vk, dwb, o * i), &ref_dw(&dy, &x, n, i, o));
                println!("ERR dw n={n} i={i} o={o} rel {e:.3e}");
                worst = worst.max(e);
                assert!(e < 4e-3, "dw n={n} i={i} o={o}: rel {e:.3e} >= 4e-3");
            }
        }
        println!("ERR worst rel {worst:.3e} (bound 4e-3, bf16 operands, f32 accumulator)");
    }

    /// Step 0 and the tensor-core kernels side by side at `n = T = 2048`: device time in ms from
    /// the GPU timestamp interval, mean of [`REPS`] runs, and `2·n·i·o` TFLOP/s per call.
    #[test]
    fn coop_timing() {
        let vk = match unsafe { Vk::load() } {
            Ok(v) => v,
            Err(e) => {
                println!("SKIP coop_timing: {e}");
                return;
            }
        };
        let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx");
        let pool = unsafe { vk.timestamp_pool(2) };
        let ident = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "nvidia-smi absent".into());
        println!(
            "TCOOP machine {ident} device {:?} n={NT} reps={REPS}",
            vk.device_name
        );
        let mut rng = Lcg(0x00c0_ffee_1234_5678);
        for &(i, o) in &SHAPES {
            let n = NT;
            let gen = |len: usize, rng: &mut Lcg| -> Vec<f32> {
                (0..len).map(|_| rng.unit() as f32).collect()
            };
            let x = gen(n * i, &mut rng);
            let w = gen(o * i, &mut rng);
            let dy = gen(n * o, &mut rng);
            let flops = 2.0 * (n * i * o) as f64;
            unsafe {
                let xb = g.up_f32(&x);
                let wb = g.up_f32(&w);
                let dyb = g.up_f32(&dy);
                let yb = g.dev(n * o * 4);
                let dxb = g.dev(n * i * 4);
                let dwb = g.dev(o * i * 4);
                let n32 = n as u32;
                let i32 = i as u32;
                let o32 = o as u32;
                let rows = [
                    (
                        "fwd",
                        crate::gpu::nn::fwd::linear_fwd_dispatch(&g, xb, wb, yb, n32, i32, o32),
                        super::linear_fwd_coop_dispatch(&g, xb, wb, yb, n32, i32, o32),
                    ),
                    (
                        "dx",
                        crate::gpu::nn::linear_bwd_input_dispatch(&g, dyb, wb, dxb, n32, i32, o32),
                        super::linear_bwd_input_coop_dispatch(&g, dyb, wb, dxb, n32, i32, o32),
                    ),
                    (
                        "dw",
                        crate::gpu::nn::linear_bwd_weight_dispatch(&g, dyb, xb, dwb, n32, i32, o32),
                        super::linear_bwd_weight_coop_dispatch(&g, dyb, xb, dwb, n32, i32, o32),
                    ),
                ];
                for (name, f32s, coops) in &rows {
                    let (fm, fmin, fmax) = series(&vk, &[f32s], &pool, REPS);
                    let (cm, cmin, cmax) = series(&vk, &[coops], &pool, REPS);
                    println!(
                        "TCOOP {name} i={i} o={o} n={n}: f32 {:.3} ms [{fmin:.3},{fmax:.3}] \
                         {:.1} TFLOP/s | coop {:.3} ms [{cmin:.3},{cmax:.3}] {:.1} TFLOP/s | \
                         ratio {:.2}x",
                        fm / 1000.0,
                        flops / (fm * 1e-6) / 1e12,
                        cm / 1000.0,
                        flops / (cm * 1e-6) / 1e12,
                        fm / cm
                    );
                }
            }
        }
    }
}
