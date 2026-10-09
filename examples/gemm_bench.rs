//! Prefill projection GEMM microbench: the DP4A batch GEMM the prefill path runs
//! (`gemv_dp4a_batch_rt_bytes` at `prefill_tile`'s tile) against the cooperative-matrix chart GEMM
//! (`simt_nn::coopmat_gemm`), on the Qwen2.5-0.5B layer-0 projection shapes (qkv, o, gate/up, down)
//! at 512 tokens, both chart carriers (Q8, Q4), real layer-0 weights packed with `pack_proj_f32`.
//!
//! Timing: GPU timestamps around `INNER` back-to-back dispatches (compute barrier between), per
//! dispatch = span / `INNER`; `--reps` rounds, arms and shapes interleaved with the arm order rotated
//! per round. Reports mean and min–max µs, GMAC/s and int8 TOPS (2 ops per MAC), the paired
//! per-round dp4a/arm and coopmat/arm ratios, and (without `--shape`) the pp512 implied by scaling
//! the GEMM node times of `docs/evidence/gpu-prefill-vs-llamacpp.md` by the measured ratios.
//!
//! Board row `inf-resident-prefill-GEB633UfORCK7XSmzJifHd`.
//!
//! **Instrument.** `--grid FILE` compiles one arm per line (`NAME [src=PATH] [-DK=V ...]`) with
//! `glslc` (`simt_nn::GLSLC_ARGS` plus the `-D`s) into `--spv-cache DIR`, keyed by the sha256 of the
//! source bytes and flags; the dispatch BM/BN come from `-DBM`/`-DBN` or the shader's own `#ifndef`
//! default. Two names compiling to the same SPIR-V are one arm (`NAME == OTHER`). Every arm writes
//! its own output buffer and is checked cell by cell against the dp4a arm (arm 0) in the same run:
//! `OK` or `REJECTED n_cells first_index`. Rejected arms stay in the table and are excluded from
//! every ratio. An arm whose name starts `ctl_` must be REJECTED (`CONTROL ACCEPTED` and exit 2
//! otherwise); the committed `coopmat` arm must be OK (exit 3 otherwise). One TSV provenance row per
//! arm and shape is appended to `--ledger FILE`; the ledger is never read back (a baseline is an arm
//! in the same invocation).
//!
//! `cargo run --release --locked --features gpu --example gemm_bench -- [--reps N] [--model PATH]
//! [--shape NAME] [--synth M K] [--stats] [--ir DIR] [--blob DIR] [--variants DIR] [--grid FILE]
//! [--spv-cache DIR] [--ledger FILE]` — `--shape` keeps one shape (the implied pp512 then needs all
//! four and is skipped), `--stats` prints the driver's pipeline statistics (registers, spills),
//! `--blob DIR` writes each kernel's one-pipeline cache blob (the SM120 SASS:
//! `fold/scripts/nvuc_sass.py`), `--variants DIR` adds one arm per `NAME.bm<BM>.bn<BN>.spv` compile
//! of the GLSL in DIR.
#![allow(
    clippy::pedantic,
    missing_docs,
    reason = "GPU microbench; the lint profile of the other gpu examples"
)]

use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::pack::pack_proj_f32;
use apeiron::gpu::decode::prefill::prefill_tile;
use apeiron::gpu::decode::state::dequant_rows;
use apeiron::gpu::decode::{local_wg, make_step_named, rec_step, step_barrier, GpuCtx, Step};
use apeiron::gpu::vulkan::*;
use apeiron::simt::{gemv_dp4a_batch_rt_bytes, Carrier};
use apeiron::simt_nn::{coopmat_gemm_module_from, COOPMAT_GEMM_SPV, GLSLC_ARGS, TILE_BM, TILE_BN};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const TOKENS: u32 = 512;
const INNER: u32 = 10;
const DEFAULT_SRC: &str = "src/spirv/coopmat_gemm.comp";
const DEFAULT_SPV_CACHE: &str = concat!(env!("HOME"), "/oc-tmp/bench/spv");
const DEFAULT_LEDGER: &str = concat!(env!("HOME"), "/oc-tmp/bench/ledger.tsv");

/// The receipt's pp512 profile (`docs/evidence/gpu-prefill-vs-llamacpp.md`, `PREFILL_PROFILE=1`,
/// q4): GPU-active µs per call summed over 24 layers, and the whole call.
const RECEIPT_TOTAL_US: f64 = 22156.0;
const RECEIPT_TOK_S: f64 = 26587.8;
/// (shape, carrier, µs): qkv, o, gate/up at Q8; down 12 layers at Q4 and 12 at Q8.
const RECEIPT_NODES: [(&str, Carrier, f64); 5] = [
    ("qkv", Carrier::Q8, 1646.0),
    ("o", Carrier::Q8, 1551.0),
    ("gate_up", Carrier::Q8, 6486.0),
    ("down", Carrier::Q4, 4017.0),
    ("down", Carrier::Q8, 4007.0),
];
/// llama.cpp Vulkan pp512 in the same receipt.
const LLAMA_VULKAN_TOK_S: f64 = 50682.8;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

/// One arm's oracle verdict against the dp4a reference.
enum Verdict {
    Ok,
    Rejected { cells: usize, first: usize },
    BuildFail(String),
}

/// One timed arm of one case: its own output buffer, dispatched module, verdict and rounds.
struct Arm {
    name: String,
    /// Provenance: sha256 of the compile the arm's SPIR-V comes from (dp4a: of the generated module).
    sha: String,
    /// The dispatched SPIR-V bytes (also the `--blob` / `--stats` object).
    code: Vec<u8>,
    /// The six bindings, binding 2 this arm's own output.
    bufs: [Handle; 6],
    out: Handle,
    st: Option<Step>,
    verdict: Verdict,
    us: Vec<f64>,
}

/// One compiled variant, before it is built per shape.
struct Kernel {
    name: String,
    spv: Vec<u8>,
    bm: u32,
    bn: u32,
    sha: String,
}

struct Case {
    name: &'static str,
    carrier: Carrier,
    m: u32,
    k: u32,
    arms: Vec<Arm>,
}

/// Mean, min, max.
fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

/// Median of `v` (mean of the two middle values on an even length).
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite ratios"));
    let m = s.len() / 2;
    if s.is_empty() {
        f64::NAN
    } else if s.len().is_multiple_of(2) {
        (s[m - 1] + s[m]) / 2.0
    } else {
        s[m]
    }
}

/// Paired per-round ratios `num[i] / den[i]`; mean, median, min, max.
fn ratio_stats(num: &[f64], den: &[f64]) -> (f64, f64, f64, f64) {
    let r: Vec<f64> = num.iter().zip(den).map(|(n, d)| n / d).collect();
    let (mean, lo, hi) = stats(&r);
    (mean, median(&r), lo, hi)
}

/// GPU-active µs per dispatch of `s`, over `INNER` back-to-back dispatches.
unsafe fn time(vk: &Vk, s: &Step) -> f64 {
    let pool = vk.timestamp_pool(2);
    let cb = vk.command_buffer();
    vk.begin(cb);
    vk.cmd_reset_timestamps(cb, &pool);
    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, 0);
    for _ in 0..INNER {
        (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, s.pipe);
        (vk.bind_sets)(
            cb,
            PIPELINE_BIND_POINT_COMPUTE,
            s.layout,
            0,
            1,
            &s.dset,
            0,
            std::ptr::null(),
        );
        (vk.dispatch)(cb, s.groups.0, s.groups.1, 1);
        step_barrier(vk, cb);
    }
    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, 1);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    let ts = vk.read_timestamps(&pool);
    ts[1].wrapping_sub(ts[0]) as f64 * f64::from(vk.timestamp_period) / 1000.0 / f64::from(INNER)
}

/// One dispatch of `s` and a full wait — the oracle's single-shot run.
unsafe fn run_once(vk: &Vk, s: &Step) {
    let cb = vk.command_buffer();
    vk.begin(cb);
    rec_step(vk, cb, s);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
}

/// Copy `n` f32 out of `src` (`tests/gpu_coopmat_gemm.rs:67`'s readback pattern).
unsafe fn readback(vk: &Vk, src: Handle, n: usize) -> Vec<f32> {
    let (rb, mem) = vk
        .buffer(
            n as u64 * 4,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer");
    let cb = vk.command_buffer();
    vk.begin(cb);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: n as u64 * 4,
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
    std::slice::from_raw_parts(vk.map(mem).cast::<f32>(), n).to_vec()
}

/// Count of bit-differing cells and the first differing index.
fn compare(reference: &[f32], other: &[f32]) -> (usize, usize) {
    let mut cells = 0usize;
    let mut first = 0usize;
    for (i, (x, y)) in reference.iter().zip(other).enumerate() {
        if x.to_bits() != y.to_bits() {
            if cells == 0 {
                first = i;
            }
            cells += 1;
        }
    }
    (cells, first)
}

fn verdict_str(v: &Verdict) -> String {
    match v {
        Verdict::Ok => "OK".to_string(),
        Verdict::Rejected { cells, first } => format!("REJECTED {cells} cells @{first}"),
        Verdict::BuildFail(m) => format!("BUILD-FAIL {m}"),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Civil date from days since 1970-01-01 (Howard Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// The `#ifndef NAME` / `#define NAME <n>` default the shader states.
fn shader_default(src: &str, name: &str) -> u32 {
    let ifndef = format!("#ifndef {name}");
    let define = format!("#define {name} ");
    let mut lines = src.lines().map(str::trim);
    while let Some(l) = lines.next() {
        if l == ifndef {
            let d = lines.next().unwrap_or("");
            return d
                .strip_prefix(&define)
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or_else(|| panic!("{name}: no default after `{ifndef}`"));
        }
    }
    panic!("{name}: no `{ifndef}` in the shader");
}

/// `-DBM=64` (or `-DBM 64`) as a number, if the grid line carries it.
fn flag_u32(flags: &[String], name: &str) -> Option<u32> {
    let joined = format!("-D{name}=");
    let spaced = format!("-D{name}");
    for (i, f) in flags.iter().enumerate() {
        if let Some(v) = f.strip_prefix(&joined) {
            return v.parse().ok();
        }
        if *f == spaced {
            return flags.get(i + 1).and_then(|v| v.parse().ok());
        }
    }
    None
}

struct GridArm {
    name: String,
    src: PathBuf,
    flags: Vec<String>,
}

/// `NAME [src=PATH] [-DK=V ...]`, `#` comments, blanks skipped.
fn read_grid(file: &str, default_src: &Path) -> Vec<GridArm> {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("--grid {file}: {e}"));
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let name = it.next().expect("token").to_string();
        let mut src = default_src.to_path_buf();
        let mut flags: Vec<String> = Vec::new();
        let mut want_define = false;
        for tok in it {
            if want_define {
                flags.push(tok.to_string());
                want_define = false;
                continue;
            }
            if let Some(s) = tok.strip_prefix("src=") {
                src = PathBuf::from(s);
            } else if tok == "-D" {
                flags.push("-D".to_string());
                want_define = true;
            } else if tok.starts_with("-D") {
                flags.push(tok.to_string());
            } else {
                panic!("--grid {file}:{}: unknown token {tok:?}", ln + 1);
            }
        }
        out.push(GridArm { name, src, flags });
    }
    out
}

/// Compile `src` with `GLSLC_ARGS` + `flags` into the sha256-keyed `cache`; returns the SPIR-V.
fn compile(src: &Path, flags: &[String], cache: &Path) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(src).map_err(|e| format!("read {}: {e}", src.display()))?;
    let mut key = Sha256::new();
    key.update(&bytes);
    for f in flags {
        key.update([0u8]);
        key.update(f.as_bytes());
    }
    let hex = key
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let out = cache.join(format!("{hex}.spv"));
    if !out.exists() {
        std::fs::create_dir_all(cache)
            .map_err(|e| format!("spv-cache {}: {e}", cache.display()))?;
        let tmp = cache.join(format!("{hex}.tmp"));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .args(flags)
            .arg(src)
            .arg("-o")
            .arg(&tmp)
            .output()
            .map_err(|e| format!("glslc spawn: {e}"))?;
        if !st.status.success() {
            let msg = String::from_utf8_lossy(&st.stderr);
            return Err(msg
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("glslc failed")
                .to_string());
        }
        std::fs::rename(&tmp, &out).map_err(|e| format!("rename: {e}"))?;
    }
    std::fs::read(&out).map_err(|e| format!("read {}: {e}", out.display()))
}

fn panic_msg(e: &(dyn std::any::Any + Send)) -> String {
    let s = if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        "panic".to_string()
    };
    s.lines().next().unwrap_or("panic").to_string()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut reps = 5usize;
    let mut model = concat!(env!("HOME"), "/models/gguf/qwen05b-Q4_K_M.gguf").to_string();
    let mut only: Option<String> = None;
    let mut stats_on = false;
    let mut blob_dir: Option<String> = None;
    let mut variants_dir: Option<String> = None;
    let mut grid_file: Option<String> = None;
    let mut spv_cache = PathBuf::from(DEFAULT_SPV_CACHE);
    let mut ledger = PathBuf::from(DEFAULT_LEDGER);
    let mut synth: Option<(usize, usize)> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reps" => reps = args.next().and_then(|v| v.parse().ok()).expect("--reps N"),
            "--model" => model = args.next().expect("--model PATH"),
            "--shape" => only = Some(args.next().expect("--shape NAME")),
            "--stats" => stats_on = true,
            "--synth" => {
                let mut dim = |w: &str| -> usize {
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| panic!("--synth M K: {w}"))
                };
                let (m, k) = (dim("M"), dim("K"));
                assert!(
                    m % 16 == 0 && k % 64 == 0,
                    "--synth needs M % 16 == 0 and K % 64 == 0"
                );
                synth = Some((m, k));
            }
            "--blob" => blob_dir = Some(args.next().expect("--blob DIR")),
            "--variants" => variants_dir = Some(args.next().expect("--variants DIR")),
            "--grid" => grid_file = Some(args.next().expect("--grid FILE")),
            "--spv-cache" => spv_cache = PathBuf::from(args.next().expect("--spv-cache DIR")),
            "--ledger" => ledger = PathBuf::from(args.next().expect("--ledger FILE")),
            _ => panic!("unknown argument {a}"),
        }
    }
    let default_src = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_SRC);

    // The committed kernel is always arm 1, so the compiled `same`/`selfcheck` lines dedupe to it.
    let mut kernels: Vec<Kernel> = vec![Kernel {
        name: "coopmat".to_string(),
        spv: COOPMAT_GEMM_SPV.to_vec(),
        bm: TILE_BM,
        bn: TILE_BN,
        sha: sha256_hex(COOPMAT_GEMM_SPV),
    }];
    let mut seen: HashMap<String, String> = HashMap::new();
    seen.insert(kernels[0].sha.clone(), "coopmat".to_string());

    if let Some(file) = &grid_file {
        for a in read_grid(file, &default_src) {
            match compile(&a.src, &a.flags, &spv_cache) {
                Ok(spv) => {
                    let sha = sha256_hex(&spv);
                    if let Some(other) = seen.get(&sha) {
                        println!("{} == {other} (spv {sha})", a.name);
                        continue;
                    }
                    let text = std::fs::read_to_string(&a.src).unwrap_or_default();
                    let bm =
                        flag_u32(&a.flags, "BM").unwrap_or_else(|| shader_default(&text, "BM"));
                    let bn =
                        flag_u32(&a.flags, "BN").unwrap_or_else(|| shader_default(&text, "BN"));
                    seen.insert(sha.clone(), a.name.clone());
                    kernels.push(Kernel {
                        name: a.name,
                        spv,
                        bm,
                        bn,
                        sha,
                    });
                }
                Err(e) => println!("BUILD-FAIL {}: {e}", a.name),
            }
        }
    }
    if let Some(dir) = &variants_dir {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .expect("variants dir")
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "spv"))
            .collect();
        files.sort();
        for f in files {
            let stem = f.file_stem().unwrap().to_string_lossy().to_string();
            let tile = |key: &str| -> u32 {
                stem.split('.')
                    .find_map(|p| p.strip_prefix(key))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| panic!("{stem}: no .{key}<n> in the name"))
            };
            let spv = std::fs::read(&f).expect("variant spv");
            let sha = sha256_hex(&spv);
            if let Some(other) = seen.get(&sha) {
                println!("{stem} == {other} (spv {sha})");
                continue;
            }
            let (bm, bn) = (tile("bm"), tile("bn"));
            seen.insert(sha.clone(), stem.clone());
            kernels.push(Kernel {
                name: stem,
                spv,
                bm,
                bn,
                sha,
            });
        }
    }

    let vk = unsafe { Vk::load() }.expect("Vulkan device");
    let gguf = if synth.is_none() {
        Some(Gguf::open(&model).expect("gguf open"))
    } else {
        None
    };
    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let lw = local_wg();
    let n = TOKENS as usize;
    let mut rng = Rng(0x6e66_b3a7_c0de_0005);
    println!(
        "device {} | model {model} | tokens {TOKENS} | reps {reps} x {INNER} dispatches | dp4a local {lw}",
        vk.device_name
    );
    println!(
        "arms: dp4a + coopmat + {} variants | spv-cache {} | ledger {} (driver version not exposed by Vk: column omitted)",
        kernels.len() - 1,
        spv_cache.display(),
        ledger.display()
    );

    let l0 = |s: &str| format!("blk.0.{s}.weight");
    let mut groups: Vec<(&'static str, Vec<String>)> = vec![
        ("qkv", vec![l0("attn_q"), l0("attn_k"), l0("attn_v")]),
        ("o", vec![l0("attn_output")]),
        ("gate_up", vec![l0("ffn_gate"), l0("ffn_up")]),
        ("down", vec![l0("ffn_down")]),
    ];
    if let Some((sm, sk)) = synth {
        // synthetic shape: one group of random weights at (m, k); the model is not read
        let label: &'static str = Box::leak(format!("synth{sm}x{sk}").into_boxed_str());
        groups = vec![(label, vec![String::new()])];
    }
    let mut cases: Vec<Case> = Vec::new();
    for (name, names) in &groups {
        if only.as_deref().is_some_and(|o| o != *name) {
            continue;
        }
        let (rows, k): (Vec<Vec<f32>>, usize) = match (&gguf, synth) {
            (Some(gguf), _) => {
                let rows = names
                    .iter()
                    .map(|nm| {
                        let t = gguf.header.tensor(nm).expect("tensor");
                        dequant_rows(gguf, nm, 0, t.dims[1] as usize).expect("dequant")
                    })
                    .collect();
                let k = gguf.header.tensor(&names[0]).expect("tensor").dims[0] as usize;
                (rows, k)
            }
            (None, Some((sm, sk))) => {
                let w = (0..sm * sk)
                    .map(|_| (rng.next() % 2001) as f32 * 1e-3 - 1.0)
                    .collect();
                (vec![w], sk)
            }
            (None, None) => unreachable!("synth is set when the model is not opened"),
        };
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let (ku, blocks) = (k, k / 32);
        let xq: Vec<u32> = (0..n * ku / 4).map(|_| rng.next()).collect();
        let sx: Vec<f32> = (0..n * blocks)
            .map(|_| 1e-3 + (rng.next() % 4096) as f32 * 4e-6)
            .collect();
        let xs: Vec<f32> = (0..n * blocks)
            .map(|_| (rng.next() % 2001) as f32 * 1e-3 - 1.0)
            .collect();
        let (bxq, bsx, bxs) = unsafe { (g.up_u32(&xq), g.up_f32(&sx), g.up_f32(&xs)) };
        for carrier in [Carrier::Q8, Carrier::Q4] {
            let p = pack_proj_f32(carrier, &refs, k, false);
            let (m, k) = (p.m as u32, p.k as u32);
            let (wb, dmb) = unsafe { (g.up_u32(&p.words), g.up_u32(&p.dm)) };
            let out_bytes = n * p.m * 4;
            let (bt, rt) = prefill_tile(m, k, carrier, lw);
            let mut arms: Vec<Arm> = Vec::new();

            // arm 0: the dp4a reference, its own output buffer.
            let out0 = unsafe { g.dev(out_bytes) };
            let bufs0 = [wb, bxq, out0, dmb, bxs, bsx];
            let dp_code = gemv_dp4a_batch_rt_bytes(m, k, lw, carrier, bt, rt);
            let dp = unsafe {
                make_step_named(
                    &g,
                    &dp_code,
                    &bufs0,
                    m.div_ceil(lw * rt),
                    TOKENS / bt,
                    "dp4a",
                )
            };
            arms.push(Arm {
                name: "dp4a".to_string(),
                sha: sha256_hex(&dp_code),
                code: dp_code,
                bufs: bufs0,
                out: out0,
                st: Some(dp),
                verdict: Verdict::Ok,
                us: Vec::new(),
            });

            for ker in &kernels {
                let out = unsafe { g.dev(out_bytes) };
                let bufs = [wb, bxq, out, dmb, bxs, bsx];
                let name = ker.name.clone();
                if !TOKENS.is_multiple_of(ker.bn) {
                    println!(
                        "BUILD-FAIL {name}: tokens {TOKENS} not a multiple of BN {}",
                        ker.bn
                    );
                    arms.push(Arm {
                        name,
                        sha: ker.sha.clone(),
                        code: Vec::new(),
                        bufs,
                        out,
                        st: None,
                        verdict: Verdict::BuildFail(format!(
                            "BN {} does not tile {TOKENS}",
                            ker.bn
                        )),
                        us: Vec::new(),
                    });
                    continue;
                }
                let hook = std::panic::take_hook();
                std::panic::set_hook(Box::new(|_| {}));
                let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let code: Vec<u8> = coopmat_gemm_module_from(&ker.spv, m, k, carrier)
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect();
                    let (gx, gy) = (m.div_ceil(ker.bm), TOKENS / ker.bn);
                    let st = unsafe { make_step_named(&g, &code, &bufs, gx, gy, "coopmat") };
                    (code, st)
                }));
                std::panic::set_hook(hook);
                match built {
                    Ok((code, st)) => arms.push(Arm {
                        name,
                        sha: ker.sha.clone(),
                        code,
                        bufs,
                        out,
                        st: Some(st),
                        verdict: Verdict::Ok,
                        us: Vec::new(),
                    }),
                    Err(e) => {
                        let msg = panic_msg(&*e);
                        println!("BUILD-FAIL {name}: {msg}");
                        arms.push(Arm {
                            name,
                            sha: ker.sha.clone(),
                            code: Vec::new(),
                            bufs,
                            out,
                            st: None,
                            verdict: Verdict::BuildFail(msg),
                            us: Vec::new(),
                        });
                    }
                }
            }

            // oracle: one dispatch per arm, every cell against the dp4a arm.
            let cells = n * p.m;
            unsafe { run_once(&vk, arms[0].st.as_ref().expect("dp4a step")) };
            let reference = unsafe { readback(&vk, arms[0].out, cells) };
            for arm in arms.iter_mut().skip(1) {
                let Some(s) = arm.st.as_ref() else {
                    continue;
                };
                unsafe { run_once(&vk, s) };
                let v = unsafe { readback(&vk, arm.out, cells) };
                arm.verdict = match compare(&reference, &v) {
                    (0, _) => Verdict::Ok,
                    (c, first) => Verdict::Rejected { cells: c, first },
                };
            }

            println!("shape {name:8} {carrier:?} m={m} k={k}: dp4a bt={bt} rt={rt}");
            for arm in &arms {
                if stats_on {
                    if let Some(s) = &arm.st {
                        let st = unsafe { vk.stats(s.pipe) };
                        let line: Vec<String> = st
                            .iter()
                            .map(|x| format!("{}={}", x.name, x.value))
                            .collect();
                        println!("  stats {}: {}", arm.name, line.join(", "));
                    }
                }
                if let Some(dir) = &blob_dir {
                    if !arm.code.is_empty() {
                        let (layout, _) = unsafe { vk.ssbo_set(&arm.bufs) };
                        let (_, blob) = unsafe { vk.isolated_cache_blob(&arm.code, layout) };
                        let f = format!("{dir}/{name}_{carrier:?}_{}.pcache", arm.name);
                        std::fs::write(&f, &blob).expect("write blob");
                    }
                }
            }
            cases.push(Case {
                name,
                carrier,
                m,
                k,
                arms,
            });
        }
    }

    // warm-up, then interleaved rounds with the timable arm order rotated every round
    for c in &cases {
        for a in &c.arms {
            if let Some(s) = &a.st {
                unsafe { time(&vk, s) };
            }
        }
    }
    for rep in 0..reps {
        for c in cases.iter_mut() {
            let order: Vec<usize> = c
                .arms
                .iter()
                .enumerate()
                .filter(|(_, a)| a.st.is_some())
                .map(|(i, _)| i)
                .collect();
            let n_arms = order.len();
            for i in 0..n_arms {
                let at = order[(i + rep) % n_arms];
                let t = unsafe { time(&vk, c.arms[at].st.as_ref().expect("timable")) };
                c.arms[at].us.push(t);
            }
        }
    }

    println!();
    println!(
        "| shape | carrier | m | k | arm | verdict | µs mean (min-max) | GMAC/s | TOPS | dp4a/arm mean (min-max) | coopmat/arm mean med (min-max) |"
    );
    println!("|---|---|---:|---:|---|---|---|---:|---:|---|---|");
    let mut ratio_of = Vec::new();
    for c in &cases {
        let macs = f64::from(c.m) * f64::from(c.k) * f64::from(TOKENS);
        let dp_ok = matches!(c.arms[0].verdict, Verdict::Ok);
        let cm_at = c.arms.iter().position(|a| a.name == "coopmat");
        let cm_ok = cm_at.is_some_and(|i| matches!(c.arms[i].verdict, Verdict::Ok));
        for (ai, a) in c.arms.iter().enumerate() {
            let verdict = verdict_str(&a.verdict);
            let Some(_) = a.st.as_ref() else {
                println!(
                    "| {} | {:?} | {} | {} | {} | {} | - | - | - | - | - |",
                    c.name, c.carrier, c.m, c.k, a.name, verdict
                );
                continue;
            };
            let (mean, lo, hi) = stats(&a.us);
            let gmac = macs / mean / 1e3;
            let dp_col = if ai == 0 || !matches!(a.verdict, Verdict::Ok) || !dp_ok {
                "-".to_string()
            } else {
                let (rm, _, rlo, rhi) = ratio_stats(&c.arms[0].us, &a.us);
                format!("{rm:.2} ({rlo:.2}-{rhi:.2})")
            };
            let cm_col = match cm_at {
                Some(ci) if ci != ai && cm_ok && matches!(a.verdict, Verdict::Ok) => {
                    let (rm, rmed, rlo, rhi) = ratio_stats(&c.arms[ci].us, &a.us);
                    format!("{rm:.2} {rmed:.2} ({rlo:.2}-{rhi:.2})")
                }
                _ => "-".to_string(),
            };
            println!(
                "| {} | {:?} | {} | {} | {} | {} | {mean:.1} ({lo:.1}-{hi:.1}) | {gmac:.0} | {:.1} | {dp_col} | {cm_col} |",
                c.name,
                c.carrier,
                c.m,
                c.k,
                a.name,
                verdict,
                2.0 * gmac / 1e3,
            );
            if a.name == "coopmat" && cm_ok && dp_ok {
                let (rm, _, _, _) = ratio_stats(&c.arms[0].us, &a.us);
                ratio_of.push((c.name, c.carrier, rm));
            }
        }
    }

    // noise floor: the committed arm's own round-to-round spread
    println!();
    println!(
        "noise floor (round-to-round spread of the committed coopmat arm over {reps} rounds):"
    );
    for c in &cases {
        if let Some(i) = c
            .arms
            .iter()
            .position(|a| a.name == "coopmat" && a.st.is_some())
        {
            let (mean, lo, hi) = stats(&c.arms[i].us);
            println!(
                "  {} {:?}: mean {mean:.1} µs, (max-min)/mean {:.1}%, max/min {:.2}",
                c.name,
                c.carrier,
                (hi - lo) / mean * 100.0,
                hi / lo,
            );
        }
    }

    // ledger: provenance only, one row per arm and shape
    let head = std::process::Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let utc = utc_now();
    let mut lf = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&ledger)
        .unwrap_or_else(|e| panic!("--ledger {}: {e}", ledger.display()));
    if lf.metadata().map(|m| m.len()).unwrap_or(1) == 0 {
        writeln!(
            lf,
            "utc\thead\tspv_sha256\tshape\tcarrier\tdevice\treps\tmean_us\tmin_us\tmax_us\tverdict"
        )
        .expect("ledger header");
    }
    for c in &cases {
        for a in &c.arms {
            let num = if a.st.is_some() {
                let (mean, lo, hi) = stats(&a.us);
                format!("{mean:.3}\t{lo:.3}\t{hi:.3}")
            } else {
                "-\t-\t-".to_string()
            };
            writeln!(
                lf,
                "{utc}\t{head}\t{}\t{}\t{:?}\t{}\t{reps}\t{num}\t{}",
                a.sha,
                c.name,
                c.carrier,
                vk.device_name,
                verdict_str(&a.verdict)
            )
            .expect("ledger row");
        }
    }

    // controls: a `ctl_` arm accepted, or a bad committed arm, is an instrument failure
    let mut control_accepted: Option<String> = None;
    let mut coopmat_bad = false;
    for c in &cases {
        for a in &c.arms {
            if a.name.starts_with("ctl_") && matches!(a.verdict, Verdict::Ok) {
                control_accepted = Some(a.name.clone());
            }
        }
        if !c
            .arms
            .iter()
            .any(|a| a.name == "coopmat" && matches!(a.verdict, Verdict::Ok))
        {
            coopmat_bad = true;
        }
    }
    if let Some(nm) = control_accepted {
        println!("CONTROL ACCEPTED {nm}");
        std::process::exit(2);
    }
    if coopmat_bad {
        println!("COOPMAT NOT OK");
        std::process::exit(3);
    }
    if only.is_some() || synth.is_some() {
        return;
    }
    // implied pp512: replace each receipt GEMM node by node · (coopmat/dp4a) of its shape
    let mut saved = 0.0;
    for (name, carrier, us) in RECEIPT_NODES {
        let r = ratio_of
            .iter()
            .find(|(n, c, _)| *n == name && *c == carrier)
            .expect("shape measured")
            .2;
        saved += us - us / r;
    }
    let total = RECEIPT_TOTAL_US - saved;
    let pp = RECEIPT_TOK_S * RECEIPT_TOTAL_US / total;
    println!();
    println!(
        "implied pp512 (derived, not measured end to end): GPU-active {RECEIPT_TOTAL_US:.0} µs -> {total:.0} µs; \
{RECEIPT_TOK_S:.0} tok/s -> {pp:.0} tok/s = {:.3}x llama.cpp Vulkan ({LLAMA_VULKAN_TOK_S:.0})",
        pp / LLAMA_VULKAN_TOK_S
    );
}
