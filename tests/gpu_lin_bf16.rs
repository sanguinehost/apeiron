//! The bf16 face of the engine GEMM (`src/gpu/spirv/coopmat_gemm.comp`, `-DOPF=1`) against an f64
//! host product at every KD caller shape, its speed beside the scalar bodies it replaced, and its
//! refusal controls.
//!
//! The face is one body with the chart GEMM that holds pp512 1.795x llama.cpp CUDA: same tile
//! macros, same register prefetch of the next stage, and the three linear products are three
//! operand descriptions, not three kernels. Operands are f32 in global memory and are narrowed to
//! bf16 once per staged element (`bf16·bf16 → f32` `M16N16K16` at subgroup scope, `u = 2^-8`); the
//! reference's tensors are bf16 in memory and its kernels convert nowhere, so this is a stated
//! deviation of the face, not of the arithmetic.
//!
//! Error bound (derived): bf16 rounding of each operand costs at most `2u` of `Σ|a||b|`, `u = 2^-8`,
//! and the f32 accumulation chain of `k` terms at most `k·2^-24`; with `k <= 151936` the second term
//! is `9.1e-3`... which is why the bound is stated per shape: the assertion is
//! `|C − C64| / Σ|a||b| ≤ 2.2·2^-8 + k·2^-24`, and the measured figure is reported beside it.
//!
//! Shapes: every `(n, i, o)` the KD step passes to a `linear_*_dispatch`, read from the call sites of
//! `src/gpu/nn/train/{mod,front,block,mixer,slot,head}.rs` at Qwen3-0.6B's geometry (`t = 2048`,
//! `d = 1024`, `nf = 3072`, `qw = 2048`, `kw = 1024`, `v = 151936`, ranks `768/576/576/768/864/864/864`,
//! head chunk `rc 256`), each as the three products `Y = X·Wᵀ`, `dX = dY·W`, `dW = dYᵀ·X`; then
//! shapes with `M`, `N` and `K` off every tile edge.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_lin_bf16 -- --nocapture --test-threads=1
//! ```
//!
//! Board row: `APEIRON-QUALITY-118-W1-T01`. Prediction `kd-kernel-port-1-1`.
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_kernel_timings/gpu_linear_fwd tests"
)]

use apeiron::gpu::decode::{make_step_named, prof_arm, prof_take, rec_step, GpuCtx, Step};
use apeiron::gpu::nn::{
    coopmat_gemm_bf16_bytes, coopmat_gemm_bf16_from, lin_bf16_dispatch, tile_for, Bf16Tile,
    LinBf16, BF16_TILE, BF16_TILE_WIDE, COOPMAT_GEMM_BF16_WIDE_SPV, GLSLC_BF16_ARGS,
    GLSLC_BF16_EXACT_ARGS,
};
use apeiron::gpu::vulkan::*;
use std::path::Path;

/// Qwen3-0.6B geometry: tokens a sample, residual width, feed-forward width, query and key/value
/// widths, vocabulary, and the head's row chunk.
const T: u32 = 2048;
const D: u32 = 1024;
const NF: u32 = 3072;
const QW: u32 = 2048;
const KW: u32 = 1024;
const V: u32 = 151936;
const RC: u32 = 256;
/// Warmup runs per cell, not reported.
const WARM: usize = 3;
/// Reported repeats per cell.
const REPS: usize = 7;

/// One caller shape: the label, the `(n, i, o)` a `linear_*_dispatch` is called with, and how many
/// times one KD layer issues it per product.
struct Cell {
    label: &'static str,
    n: u32,
    i: u32,
    o: u32,
    per_layer: u32,
}

/// The seven activation shapes of one Qwen3 layer, per `src/gpu/nn/train/{mod,front,block}.rs`.
const ACT: [Cell; 7] = [
    Cell {
        label: "q",
        n: T,
        i: D,
        o: QW,
        per_layer: 1,
    },
    Cell {
        label: "k",
        n: T,
        i: D,
        o: KW,
        per_layer: 1,
    },
    Cell {
        label: "v",
        n: T,
        i: D,
        o: KW,
        per_layer: 1,
    },
    Cell {
        label: "o",
        n: T,
        i: QW,
        o: D,
        per_layer: 1,
    },
    Cell {
        label: "gate",
        n: T,
        i: D,
        o: NF,
        per_layer: 1,
    },
    Cell {
        label: "up",
        n: T,
        i: D,
        o: NF,
        per_layer: 1,
    },
    Cell {
        label: "down",
        n: T,
        i: NF,
        o: D,
        per_layer: 1,
    },
];

/// The seven factor shapes (`materialise` / `factor_bwd`, `src/gpu/nn/train/slot.rs`): `n = out`,
/// `i = rank`, `o = inn`.
const MAT: [Cell; 7] = [
    Cell {
        label: "q mat",
        n: QW,
        i: 768,
        o: D,
        per_layer: 1,
    },
    Cell {
        label: "k mat",
        n: KW,
        i: 576,
        o: D,
        per_layer: 1,
    },
    Cell {
        label: "v mat",
        n: KW,
        i: 576,
        o: D,
        per_layer: 1,
    },
    Cell {
        label: "o mat",
        n: D,
        i: 768,
        o: QW,
        per_layer: 1,
    },
    Cell {
        label: "gate mat",
        n: NF,
        i: 864,
        o: D,
        per_layer: 1,
    },
    Cell {
        label: "up mat",
        n: NF,
        i: 864,
        o: D,
        per_layer: 1,
    },
    Cell {
        label: "down mat",
        n: D,
        i: 864,
        o: NF,
        per_layer: 1,
    },
];

/// Shapes with `M`, `N` and `K` off every tile edge (`128 × 64` and `128 × 128` tiles, 16- and
/// 4-wide runs): the guards' own cases.
const RAGGED: [Cell; 21] = [
    // The `da` adjoint's own dispatch: `linear_bwd_input(dy, w, dx, n = 24, i = 48, o = 32)` —
    // rows `i` 48, cols `n` 24, k `o` 32, `ta = 1`, `rsa = i`. No other cell has that triple, and one
    // entry of `block_tune`'s `da` sits above its bound there.
    Cell {
        label: "da 24x48x32",
        n: 24,
        i: 48,
        o: 32,
        per_layer: 0,
    },
    Cell {
        label: "da 24x48x64",
        n: 24,
        i: 48,
        o: 64,
        per_layer: 0,
    },
    Cell {
        label: "da 48x48x32",
        n: 48,
        i: 48,
        o: 32,
        per_layer: 0,
    },
    // The block-tune adjoint's own shapes (`da = dh·Wo` at rows 48, cols 24, k 32, and the sweep
    // around it): cols below `BN` 64 and rows not a multiple of 16 take the masked epilogue, and the
    // last-column tile of a narrow output is a case no other cell covered (`da[23,0]` at 2.28x).
    Cell {
        label: "tune 48x32x24",
        n: 48,
        i: 32,
        o: 24,
        per_layer: 0,
    },
    Cell {
        label: "tune 48x32x8",
        n: 48,
        i: 32,
        o: 8,
        per_layer: 0,
    },
    Cell {
        label: "tune 48x32x40",
        n: 48,
        i: 32,
        o: 40,
        per_layer: 0,
    },
    Cell {
        label: "tune 48x32x72",
        n: 48,
        i: 32,
        o: 72,
        per_layer: 0,
    },
    Cell {
        label: "tune 48x32x100",
        n: 48,
        i: 32,
        o: 100,
        per_layer: 0,
    },
    Cell {
        label: "tune 16x32x24",
        n: 16,
        i: 32,
        o: 24,
        per_layer: 0,
    },
    Cell {
        label: "tune 50x32x24",
        n: 50,
        i: 32,
        o: 24,
        per_layer: 0,
    },
    Cell {
        label: "tune 100x32x24",
        n: 100,
        i: 32,
        o: 24,
        per_layer: 0,
    },
    Cell {
        label: "tune 48x64x24",
        n: 48,
        i: 64,
        o: 24,
        per_layer: 0,
    },
    Cell {
        label: "tune 24x32x100",
        n: 24,
        i: 32,
        o: 100,
        per_layer: 0,
    },
    Cell {
        label: "ragged 200x864x100",
        n: 200,
        i: 864,
        o: 100,
        per_layer: 0,
    },
    Cell {
        label: "ragged 7x33x5",
        n: 7,
        i: 33,
        o: 5,
        per_layer: 0,
    },
    Cell {
        label: "ragged 13x5x7",
        n: 13,
        i: 5,
        o: 7,
        per_layer: 0,
    },
    Cell {
        label: "ragged 129x257x65",
        n: 129,
        i: 257,
        o: 65,
        per_layer: 0,
    },
    // the block-tune test's own geometry (t = 24, d = 32, f = 64, rank 8): K a multiple of 4 but
    // not of 32, N short of the tile, M inside one row tile
    Cell {
        label: "block 24x24x32",
        n: 24,
        i: 24,
        o: 32,
        per_layer: 0,
    },
    Cell {
        label: "block 24x32x64",
        n: 24,
        i: 32,
        o: 64,
        per_layer: 0,
    },
    Cell {
        label: "block 24x64x32",
        n: 24,
        i: 64,
        o: 32,
        per_layer: 0,
    },
    Cell {
        label: "block 24x8x32",
        n: 24,
        i: 8,
        o: 32,
        per_layer: 0,
    },
];

/// THE ESCAPE SHAPES: `k` inside one 32-position stage with `rsa == k` (the factor GEMM's rank
/// contraction), every other criterion of an unguarded dispatch satisfied (`rows % 16 == 0`,
/// `cols % BN == 0`, strides a multiple of 4). The exact face reads whole aligned four-word runs of
/// a 32-wide stage, so a `k` below 32 there pulls the next rows' words into the reduction; these
/// cells exist so no criterion can quietly drop that case again (`bf16_exact`'s K test, turn 8).
const ESCAPE: [Cell; 6] = [
    Cell {
        label: "escape k=4 rank",
        n: 64,
        i: 4,
        o: 32,
        per_layer: 0,
    },
    Cell {
        label: "escape k=8 rank",
        n: 64,
        i: 8,
        o: 32,
        per_layer: 0,
    },
    Cell {
        label: "escape k=16 rank",
        n: 64,
        i: 16,
        o: 32,
        per_layer: 0,
    },
    Cell {
        label: "escape k=24 rank",
        n: 64,
        i: 24,
        o: 32,
        per_layer: 0,
    },
    Cell {
        label: "escape k=31 rank",
        n: 64,
        i: 31,
        o: 32,
        per_layer: 0,
    },
    Cell {
        label: "escape k=28 wide rows",
        n: 64,
        i: 28,
        o: 48,
        per_layer: 0,
    },
];

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-0.5, 0.5)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        u - 0.5
    }
}

fn device_line() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "device name/driver unavailable (nvidia-smi absent)".to_string(),
    }
}

/// Compile `src/gpu/spirv/coopmat_gemm.comp` with `args` into a fresh byte vector.
fn build(args: &[&str], tag: &str) -> Vec<u8> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("coopmat_bf16_{tag}_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(args)
        .arg(dir.join("coopmat_gemm.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH");
    assert!(st.success(), "glslc failed for {tag}: {st}");
    let bytes = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    bytes
}

/// The face built with extra `-D`s beside [`GLSLC_BF16_ARGS`].
fn variant(defines: &[&str], tag: &str) -> Vec<u8> {
    let mut args: Vec<&str> = GLSLC_BF16_ARGS.to_vec();
    args.extend_from_slice(defines);
    build(&args, tag)
}

/// `a`'s value at `(row, k)` under the product's order.
fn aval(x: &LinBf16, a: &[f32], row: usize, k: usize) -> f64 {
    f64::from(if x.ta == 0 {
        a[row * x.rsa as usize + k]
    } else {
        a[k * x.rsa as usize + row]
    })
}

/// `b`'s value at `(col, k)` under the product's order.
fn bval(x: &LinBf16, b: &[f32], col: usize, k: usize) -> f64 {
    f64::from(if x.tb == 0 {
        b[col * x.rsb as usize + k]
    } else {
        b[k * x.rsb as usize + col]
    })
}

/// Words each of `a`, `b` and `c` must hold.
fn wc(x: &LinBf16) -> (usize, usize, usize) {
    let a = if x.ta == 0 {
        x.rows as usize * x.rsa as usize
    } else {
        x.k as usize * x.rsa as usize
    };
    let b = if x.tb == 0 {
        x.cols as usize * x.rsb as usize
    } else {
        x.k as usize * x.rsb as usize
    };
    (a, b, x.cols as usize * x.rso as usize)
}

/// Max of `|C − C64| / Σ_k|a||b|` over cells sampled from the four tile corners: 64 × 64 a corner,
/// 8 × 8 when the reduction is long enough that an f64 host product would be the slow part.
fn worst(x: &LinBf16, a: &[f32], b: &[f32], c: &[f32]) -> (f64, usize, usize) {
    let (rows, cols, k) = (x.rows as usize, x.cols as usize, x.k as usize);
    let s = if k > 8192 { 8 } else { 64 };
    let (rs, cs) = (rows.saturating_sub(s), cols.saturating_sub(s));
    let mut worst = 0.0f64;
    let mut at = (0, 0);
    for (r0, c0) in [(0, 0), (rs, 0), (0, cs), (rs, cs)] {
        for r in r0..(r0 + s).min(rows) {
            for t in c0..(c0 + s).min(cols) {
                let mut sum = 0.0f64;
                let mut mag = 0.0f64;
                for kk in 0..k {
                    let (av, bv) = (aval(x, a, r, kk), bval(x, b, t, kk));
                    sum += av * bv;
                    mag += av.abs() * bv.abs();
                }
                let e = (f64::from(c[t * x.rso as usize + r]) - sum).abs() / mag.max(1e-30);
                if e > worst {
                    worst = e;
                    at = (r, t);
                }
            }
        }
    }
    (worst, at.0, at.1)
}

/// Upload operands, run one step to completion, read `c` back.
fn run(
    g: &GpuCtx,
    vk: &Vk,
    spv: &[u8],
    x: &LinBf16,
    tile: Bf16Tile,
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let (_, _, wc) = wc(x);
    // C is filled with NaN: any cell the kernel does not store stays NaN, which no finite product
    // of these operands can be. A finite sentinel (7.0) cannot make that claim — one cell of a
    // 2048-term sum landed on it — and a stored cell is always finite.
    let sentinel = vec![f32::NAN; wc];
    let (buf_a, buf_b, buf_c) = unsafe { (g.up_f32(a), g.up_f32(b), g.up_f32(&sentinel)) };
    let step = face_step(g, Some(spv), x, tile, (buf_a, buf_b, buf_c));
    let (rb, mapped) = unsafe {
        vk.buffer(
            wc as u64 * 4,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer")
    };
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_step(vk, cb, &step);
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
            size: (wc * 4) as u64,
        };
        (vk.copy_buffer)(cb, buf_c, rb, 1, &region);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_HOST,
            ACCESS_TRANSFER_WRITE,
            ACCESS_HOST_READ,
        );
        assert_eq!((vk.end_cb)(cb), 0, "vkEndCommandBuffer");
        vk.submit_wait(cb);
        std::slice::from_raw_parts(vk.map(mapped).cast::<f32>(), wc).to_vec()
    }
}

/// The face's step for one product over the committed image (`None`) or a built variant.
fn face_step(
    g: &GpuCtx,
    module: Option<&[u8]>,
    x: &LinBf16,
    tile: Bf16Tile,
    bufs: (Handle, Handle, Handle),
) -> Step {
    unsafe {
        match module {
            None => lin_bf16_dispatch(g, bufs.0, bufs.1, bufs.2, x, tile),
            Some(b) => {
                let (gx, gy) = x.groups(tile);
                make_step_named(
                    g,
                    b,
                    &[bufs.0, bufs.1, bufs.2, bufs.0, bufs.1],
                    gx,
                    gy,
                    "lin_bf16",
                )
            }
        }
    }
}

/// One submitted step's device time, one timestamp pair.
fn stamp(vk: &Vk, step: &Step) -> f64 {
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        prof_arm(vk, 1);
        vk.begin(cb);
        rec_step(vk, cb, step);
        assert_eq!((vk.end_cb)(cb), 0, "vkEndCommandBuffer");
        vk.submit_wait(cb);
        let t = prof_take(vk);
        t.first().map_or(f64::NAN, |p| p.1)
    }
}

fn mean_span(v: &[f64]) -> (f64, f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

/// The three operand descriptions of one caller shape.
fn products(c: &Cell) -> [(&'static str, LinBf16); 3] {
    [
        ("fwd", LinBf16::fwd(c.n, c.i, c.o)),
        ("bwd_input", LinBf16::bwd_input(c.n, c.i, c.o)),
        ("bwd_weight", LinBf16::bwd_weight(c.n, c.i, c.o)),
    ]
}

/// The derived bound of one product: `2.2·2^-8` of `Σ|a||b|`, widened by the f32 chain's
/// `k·2^-24`.
fn bound(x: &LinBf16) -> f64 {
    2.2 / 256.0 + f64::from(x.k) / 16_777_216.0
}

#[test]
fn lin_bf16_matches_f64() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {}", device_line());
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    for c in ACT.iter().chain(&MAT).chain(&RAGGED).chain(&ESCAPE) {
        for (kind, x) in products(c) {
            let (wa, wb, _) = wc(&x);
            let a: Vec<f32> = (0..wa).map(|_| rng.unit() as f32).collect();
            let b: Vec<f32> = (0..wb).map(|_| rng.unit() as f32).collect();
            let spv = coopmat_gemm_bf16_bytes(&x, tile_for(x.rows));
            let out = run(&g, &vk, &spv, &x, tile_for(x.rows), &a, &b);
            let gaps: Vec<usize> = out
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_nan())
                .map(|(i, _)| i)
                .collect();
            if let Some(&g) = gaps.first() {
                let (r, t) = (g % x.rso as usize, g / x.rso as usize);
                let mut sum = 0.0f64;
                for kk in 0..x.k as usize {
                    sum += aval(&x, &a, r, kk) * bval(&x, &b, t, kk);
                }
                println!("   gap cell (row {r}, col {t}) computes to {sum:.6}");
            }
            assert!(
                gaps.is_empty(),
                "{} {kind}: {} output words never stored, at {:?} of {}",
                c.label,
                gaps.len(),
                &gaps[..gaps.len().min(8)],
                out.len()
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{} {kind}: non-finite C",
                c.label
            );
            let (w, r, t) = worst(&x, &a, &b, &out);
            let bd = bound(&x);
            println!(
                "{:>18} {kind:>10} M {} N {} K {} tile {}x{}: max |C-C64|/sum|a||b| = {w:.3e} at \
                 ({r},{t}), bound {bd:.3e}",
                c.label,
                x.rows,
                x.cols,
                x.k,
                tile_for(x.rows).bm,
                tile_for(x.rows).bn
            );
            assert!(
                w <= bd,
                "{} {kind}: {w:.3e} exceeds the derived bound {bd:.3e}",
                c.label
            );
        }
    }
    // the tile variants on one production shape: same arithmetic, different tile
    let x = LinBf16::fwd(T, D, QW);
    let (wa, wb, _) = wc(&x);
    let a: Vec<f32> = (0..wa).map(|_| rng.unit() as f32).collect();
    let b: Vec<f32> = (0..wb).map(|_| rng.unit() as f32).collect();
    for (tag, defines, tile) in [(
        "cutlass_256x128",
        &["-DBM=256", "-DBN=128", "-DSGM=4", "-DSGN=2"][..],
        Bf16Tile {
            bm: 256,
            bn: 128,
            kb: 2,
        },
    )] {
        let spv = variant(defines, tag);
        let words = coopmat_gemm_bf16_from(&spv, &x, tile);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let out = run(&g, &vk, &bytes, &x, tile, &a, &b);
        let (w, r, t) = worst(&x, &a, &b, &out);
        println!(
            "variant {tag} on q fwd: max |C-C64|/sum|a||b| = {w:.3e} at ({r},{t}), bound {:.3e}",
            bound(&x)
        );
        assert!(w <= bound(&x), "variant {tag}: {w:.3e} exceeds the bound");
    }
}

#[test]
fn lin_bf16_speed() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!(
        "device: {}; {} warmup + {REPS} repeats, device timestamps, mean [min,max] us",
        device_line(),
        WARM
    );
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let v256 = variant(&["-DBM=256", "-DBN=128", "-DSGM=4", "-DSGN=2"], "s256");
    let tiles = [
        ("128x64", None, BF16_TILE),
        ("128x128", None, BF16_TILE_WIDE),
        (
            "256x128",
            Some(&v256[..]),
            Bf16Tile {
                bm: 256,
                bn: 128,
                kb: 2,
            },
        ),
    ];
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    // per step: 3 bands of (7 activation + 7 factor) forward products, one of each adjoint per
    // layer, and the head's chunked products (16 forward, 8 input adjoint)
    let mut step_us = [0.0f64; 3];
    for (label, n, i, o, per_layer) in ACT
        .iter()
        .map(|c| (c.label, c.n, c.i, c.o, c.per_layer))
        .chain(MAT.iter().map(|c| (c.label, c.n, c.i, c.o, c.per_layer)))
        .chain([("head", RC, D, V, 0)])
    {
        for (kind, x) in [
            ("fwd", LinBf16::fwd(n, i, o)),
            ("bwd_input", LinBf16::bwd_input(n, i, o)),
            ("bwd_weight", LinBf16::bwd_weight(n, i, o)),
        ] {
            let (wa, wb, wcc) = wc(&x);
            let a: Vec<f32> = (0..wa).map(|_| rng.unit() as f32).collect();
            let b: Vec<f32> = (0..wb).map(|_| rng.unit() as f32).collect();
            let bufs = unsafe { (g.up_f32(&a), g.up_f32(&b), g.dev(wcc * 4)) };
            let flops = 2.0 * f64::from(x.rows) * f64::from(x.cols) * f64::from(x.k);
            let mut line = format!("{label:>10} {kind:>10} K {} M {} N {}", x.k, x.rows, x.cols);
            for (tag, module, tile) in &tiles {
                let module = match module {
                    None => None,
                    Some(m) => {
                        let words = coopmat_gemm_bf16_from(m, &x, *tile);
                        Some(
                            words
                                .iter()
                                .flat_map(|w| w.to_le_bytes())
                                .collect::<Vec<u8>>(),
                        )
                    }
                };
                let step = face_step(&g, module.as_deref(), &x, *tile, bufs);
                let mut v = Vec::new();
                for _ in 0..WARM {
                    stamp(&vk, &step);
                }
                for _ in 0..REPS {
                    v.push(stamp(&vk, &step));
                }
                let (m, lo, hi) = mean_span(&v);
                line.push_str(&format!(
                    " | {tag} {m:9.1} [{lo:7.1},{hi:7.1}] {:.1} TF",
                    flops / (m * 1e6)
                ));
                if *tile == tile_for(x.rows) {
                    // the chosen tile: the per-step total this product contributes
                    let calls = match label {
                        "head" if kind == "fwd" => 16.0,
                        "head" => 8.0,
                        _ => 28.0 * f64::from(per_layer) * if kind == "fwd" { 3.0 } else { 1.0 },
                    };
                    let slot = match kind {
                        "fwd" => 0,
                        "bwd_input" => 1,
                        _ => 2,
                    };
                    step_us[slot] += calls * m / 1e3;
                }
            }
            println!("{line}");
        }
    }
    println!(
        "per step on the chosen tiles: linear_fwd {:.1} ms, linear_bwd_input {:.1} ms, \
         linear_bwd_weight {:.1} ms, total {:.1} ms (P144 scalar bodies: 426.5 / 300.0 / 156.0 / 882.5)",
        step_us[0],
        step_us[1],
        step_us[2],
        step_us[0] + step_us[1] + step_us[2]
    );
}

/// A throwaway copy of the shader with `from` replaced by `to`, compiled to a fresh image.
fn mutate(from: &str, to: &str, tag: &str) -> Vec<u8> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv/coopmat_gemm.comp");
    let text = std::fs::read_to_string(&src).expect("shader");
    let mutated = text.replacen(from, to, 1);
    assert_ne!(mutated, text, "the control's site was not found: {tag}");
    let out = std::env::temp_dir().join(format!(
        "coopmat_bf16_ctl_{tag}_{}.comp",
        std::process::id()
    ));
    std::fs::write(&out, mutated).expect("control shader");
    let spv =
        std::env::temp_dir().join(format!("coopmat_bf16_ctl_{tag}_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_BF16_ARGS)
        .arg(&out)
        .arg("-o")
        .arg(&spv)
        .status()
        .expect("glslc");
    let _ = std::fs::remove_file(&out);
    assert!(st.success(), "control shader must compile: {tag}: {st}");
    let bytes = std::fs::read(&spv).expect("control spv");
    let _ = std::fs::remove_file(&spv);
    bytes
}

#[test]
fn lin_bf16_controls() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let mut rng = Lcg(0x0bad_c0de_1234_5678);

    // 1. B staged at the wrong k offset: the reduction is the wrong one
    let spv = mutate(
        "rb[it] = run_b(col0 + c, s * KS + kv * 4u, NC);",
        "rb[it] = run_b(col0 + c, s * KS + ((kv + 1u) % K4) * 4u, NC);",
        "koff",
    );
    let x = LinBf16::fwd(T, D, KW);
    let (wa, wb, _) = wc(&x);
    let a: Vec<f32> = (0..wa).map(|_| rng.unit() as f32).collect();
    let b: Vec<f32> = (0..wb).map(|_| rng.unit() as f32).collect();
    let words = coopmat_gemm_bf16_from(&spv, &x, tile_for(x.rows));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let out = run(&g, &vk, &bytes, &x, tile_for(x.rows), &a, &b);
    let (w, r, t) = worst(&x, &a, &b, &out);
    println!(
        "CONTROL 1 wrong B k offset: max |C-C64|/sum|a||b| = {w:.3e} at ({r},{t}), bound {:.3e}, \
         red = {}",
        bound(&x),
        w > bound(&x)
    );
    assert!(w > bound(&x), "control 1 must turn the error gate red");

    // 2. a ragged shape on the exact face — the selection `bf16_exact` makes, bypassed: the fetch
    //    then stages whole aligned vec4s of an operand whose runs are neither
    // the shape whose K and columns are both short of the tile: the guards must bite
    let c = RAGGED
        .iter()
        .find(|c| c.label == "ragged 7x33x5")
        .expect("the control's shape");
    let x = LinBf16::fwd(c.n, c.i, c.o);
    let (wa, wb, _) = wc(&x);
    let a: Vec<f32> = (0..wa).map(|_| rng.unit() as f32).collect();
    let b: Vec<f32> = (0..wb).map(|_| rng.unit() as f32).collect();
    let exact = build(GLSLC_BF16_EXACT_ARGS, "exact");
    let words = coopmat_gemm_bf16_from(&exact, &x, BF16_TILE);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let out = run(&g, &vk, &bytes, &x, BF16_TILE, &a, &b);
    let (w, r, t) = worst(&x, &a, &b, &out);
    println!(
        "CONTROL 2 ragged {}x{}x{} on the exact face: max |C-C64|/sum|a||b| = {w:.3e} at ({r},{t}), \
         bound {:.3e}, red = {}",
        c.n,
        c.i,
        c.o,
        bound(&x),
        w > bound(&x)
    );
    assert!(w > bound(&x), "control 2 must turn the guard test red");

    // 3. the masked epilogue removed from the guarded face: the same ragged shape then stores
    //    through full fragments
    let spv = mutate("    if (EPIMASK != 0) {", "    if (false) {", "mask");
    let words = coopmat_gemm_bf16_from(&spv, &x, BF16_TILE);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let out = run(&g, &vk, &bytes, &x, BF16_TILE, &a, &b);
    let (w, r, t) = worst(&x, &a, &b, &out);
    println!(
        "CONTROL 3 masked epilogue removed on ragged {}x{}x{}: max |C-C64|/sum|a||b| = {w:.3e} at \
         ({r},{t}), bound {:.3e}, red = {}",
        c.n,
        c.i,
        c.o,
        bound(&x),
        w > bound(&x)
    );
    assert!(w > bound(&x), "control 3 must turn the guard test red");
}

#[test]
#[ignore = "probe: the face's SPIR-V instruction mix; the driver's SASS container has no decoder here"]
fn lin_bf16_code_shape() {
    let bytes = coopmat_gemm_bf16_bytes(&LinBf16::fwd(T, D, QW), tile_for(QW));
    let spv = std::env::temp_dir().join("lin_bf16.spv");
    std::fs::write(&spv, &bytes).expect("spv");
    let dis = std::process::Command::new("spirv-dis")
        .arg(&spv)
        .output()
        .expect("spirv-dis");
    let text = String::from_utf8_lossy(&dis.stdout).to_string();
    for op in [
        "OpCooperativeMatrixMulAddKHR",
        "OpCooperativeMatrixLoadKHR",
        "OpCooperativeMatrixStoreKHR",
        "OpFConvert",
        "OpLoad",
        "OpStore",
    ] {
        println!("  {op:34} {}", text.matches(op).count());
    }
}

/// The two committed tiles on one small shape: same arithmetic, different tiling. Cheap enough to
/// run beside the shape sweep and the fastest way to see a tile-local fault.
#[test]
fn lin_bf16_both_tiles() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let mut rng = Lcg(0x5eed_1234_5678_9abc);
    let x = LinBf16::fwd(256, 64, 256);
    let (wa, wb, _) = wc(&x);
    let a: Vec<f32> = (0..wa).map(|_| rng.unit() as f32).collect();
    let b: Vec<f32> = (0..wb).map(|_| rng.unit() as f32).collect();
    for (tag, tile) in [("128x64", BF16_TILE), ("128x128", BF16_TILE_WIDE)] {
        let spv = coopmat_gemm_bf16_bytes(&x, tile);
        let out = run(&g, &vk, &spv, &x, tile, &a, &b);
        let (w, r, t) = worst(&x, &a, &b, &out);
        println!(
            "tile {tag} on 256x64 C: max |C-C64|/sum|a||b| = {w:.3e} at ({r},{t}), bound {:.3e}",
            bound(&x)
        );
        assert!(w <= bound(&x), "tile {tag}: {w:.3e} exceeds the bound");
    }
}

/// Probe: one product of one shape at one tile, from `LIN_SPV` (a compiled image path, default the
/// committed face) — the iteration loop for a shader change, 3 warmup + 7 repeats.
#[test]
#[ignore = "probe: one shape, one image, timed; `LIN_SPV` and `LIN_SHAPE` select them"]
fn lin_bf16_probe() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    let (g, _mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let path = std::env::var("LIN_SPV").unwrap_or_default();
    let image = if path.is_empty() {
        None
    } else {
        Some(std::fs::read(&path).expect("LIN_SPV"))
    };
    let which = std::env::var("LIN_SHAPE").unwrap_or_else(|_| "q fwd".to_string());
    let tile = BF16_TILE_WIDE;
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let x = LinBf16::fwd(T, D, QW);
    let (wa, wb, wcc) = wc(&x);
    let a: Vec<f32> = (0..wa).map(|_| rng.unit() as f32).collect();
    let b: Vec<f32> = (0..wb).map(|_| rng.unit() as f32).collect();
    let bufs = unsafe { (g.up_f32(&a), g.up_f32(&b), g.dev(wcc * 4)) };
    let words = coopmat_gemm_bf16_from(
        image.as_deref().unwrap_or(COOPMAT_GEMM_BF16_WIDE_SPV),
        &x,
        tile,
    );
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let step = face_step(&g, Some(&bytes), &x, tile, bufs);
    let mut v = Vec::new();
    for _ in 0..WARM {
        stamp(&vk, &step);
    }
    for _ in 0..REPS {
        v.push(stamp(&vk, &step));
    }
    let (m, lo, hi) = mean_span(&v);
    let flops = 2.0 * f64::from(x.rows) * f64::from(x.cols) * f64::from(x.k);
    println!(
        "{which} at {}x{}: {m:.1} [{lo:.1},{hi:.1}] us, {:.1} TF{}",
        tile.bm,
        tile.bn,
        flops / (m * 1e6),
        if path.is_empty() { "" } else { " (LIN_SPV)" }
    );
}
