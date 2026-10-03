//! The engine benchmark: a fixed protocol, interleaved against `llama-bench`, with per-phase
//! attribution, written as a JSON receipt and a Markdown summary.
//!
//! Run:
//! `cargo run --release --locked --example engine_profile -- --model <gguf> --ids <file>
//!  [--pp 200] [--tg 64] [--reps 5] [--spec <draft rows>] [--llama <llama-bench>]
//!  [--llama-threads 8,16,24,32] [--max-busy 0.05] [--label <name>] [--out the bench note]`
//!
//! # Protocol
//!
//! 1. Identity: CPU model, governor, online CPUs, kernel, `rustc`, git `HEAD` and dirty-path
//!    count, and the sha256 of the source tree, the model file and the prompt-id file.
//! 2. One unrecorded warm-up of every workload (page-in, lazy tables, pool start).
//! 3. `reps` rounds. Each round waits for the host to be quiet (busy fraction from `/proc/stat`
//!    over one second below `max-busy`, at most 120 s; the observed value is recorded either
//!    way), then runs, interleaved, `llama-bench` over each thread count and this engine's
//!    workloads, so drift hits both sides alike. Each side warms up immediately before it is
//!    measured: `llama-bench` runs its default warm-up per test, and this engine runs one
//!    unrecorded `pp` and `tg` before its measured `pp` and `tg`.
//!    - `pp`: [`Engine::prefill_state`] over the first `pp` prompt ids from an empty state.
//!    - `tg`: `tg` greedy [`Engine::next_token_state`] steps from position 0.
//!    - `spec` (with `--spec`): [`Engine::generate_speculative`] of `tg` tokens from the first
//!      prompt id, wall clock including its one-token prefill.
//!
//!    Each engine workload records wall time, process CPU time (`getrusage`, so the cores it
//!    actually kept busy) and the mean scaling frequency across CPUs just before it ran.
//! 4. One traced run of each workload ([`apeiron::engine::trace`]): per phase the time per token,
//!    the share, and achieved bandwidth and multiply-add rate from model-derived byte and
//!    multiply-add counts.
//!
//! The summary reports median, min and max per metric and the median of the per-round
//! engine/llama ratios against llama's best thread count in that round. A round whose quiet wait
//! timed out is flagged, not dropped.

use std::error::Error;
use std::fmt::Write as _;
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use apeiron::engine::loader::Engine;
use apeiron::engine::trace;
use apeiron::engine::KernelMode;
use sha2::{Digest, Sha256};

/// Parsed command line.
struct Args {
    model: String,
    ids: String,
    pp: usize,
    tg: usize,
    reps: usize,
    spec: Option<usize>,
    llama: Option<String>,
    llama_threads: String,
    max_busy: f64,
    label: String,
    out: String,
    /// A/B arms: a name and the environment the arm's child process runs under.
    arms: Vec<(String, Vec<(String, String)>)>,
    /// Run one round of the engine workloads and print one `CHILD` line (an arm's process).
    child: bool,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        model: "~/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf".into(),
        ids: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen35-prompt2.ids"
        )
        .into(),
        pp: 200,
        tg: 64,
        reps: 5,
        spec: None,
        llama: None,
        llama_threads: "8,16,24,32".into(),
        max_busy: 0.05,
        label: "run".into(),
        out: "the bench note".into(),
        arms: Vec::new(),
        child: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--ids" => a.ids = val()?,
            "--pp" => a.pp = val()?.parse()?,
            "--tg" => a.tg = val()?.parse()?,
            "--reps" => a.reps = val()?.parse()?,
            "--spec" => a.spec = Some(val()?.parse()?),
            "--llama" => a.llama = Some(val()?),
            "--llama-threads" => a.llama_threads = val()?,
            "--max-busy" => a.max_busy = val()?.parse()?,
            "--label" => a.label = val()?,
            "--out" => a.out = val()?,
            // `--arm name` or `--arm name:KEY=VAL,KEY=VAL`.
            "--arm" => {
                let spec = val()?;
                let (name, env) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
                let env = env
                    .split(',')
                    .filter(|kv| !kv.is_empty())
                    .map(|kv| {
                        kv.split_once('=')
                            .map(|(k, v)| (k.to_string(), v.to_string()))
                            .ok_or_else(|| format!("arm {name}: {kv} is not KEY=VAL"))
                    })
                    .collect::<Result<_, _>>()?;
                a.arms.push((name.to_string(), env));
            }
            "--child" => a.child = true,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    Ok(a)
}

// ---- host state ----------------------------------------------------------------------------

/// Total and idle jiffies across all CPUs from the first line of `/proc/stat`.
fn cpu_jiffies() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/stat").ok()?;
    let line = s.lines().next()?;
    let v: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let idle = v.get(3)? + v.get(4).copied().unwrap_or(0);
    Some((v.iter().sum(), idle))
}

/// The host's busy fraction over `window` while this process sleeps.
#[allow(
    clippy::cast_precision_loss,
    reason = "a jiffy ratio for display and gating"
)]
fn host_busy(window: Duration) -> f64 {
    let Some((t0, i0)) = cpu_jiffies() else {
        return f64::NAN;
    };
    std::thread::sleep(window);
    let Some((t1, i1)) = cpu_jiffies() else {
        return f64::NAN;
    };
    let dt = t1.saturating_sub(t0).max(1);
    1.0 - i1.saturating_sub(i0) as f64 / dt as f64
}

/// Waits up to 120 s for a one-second busy fraction below `max`; returns the last reading and
/// whether it passed.
fn wait_quiet(max: f64) -> (f64, bool) {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let b = host_busy(Duration::from_secs(1));
        if b < max {
            return (b, true);
        }
        if Instant::now() > deadline {
            return (b, false);
        }
    }
}

/// Mean `scaling_cur_freq` across CPUs, in MHz.
#[allow(clippy::cast_precision_loss, reason = "a mean frequency for display")]
fn mean_mhz() -> f64 {
    let mut sum = 0u64;
    let mut n = 0u64;
    for cpu in 0..256 {
        let p = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq");
        match std::fs::read_to_string(&p) {
            Ok(s) => {
                if let Ok(k) = s.trim().parse::<u64>() {
                    sum += k;
                    n += 1;
                }
            }
            Err(_) => break,
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        sum as f64 / n as f64 / 1000.0
    }
}

/// This process's user plus system CPU seconds.
fn cpu_seconds() -> f64 {
    // SAFETY: `getrusage` writes a zeroed `rusage` owned by this frame.
    let ru = unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &raw mut ru);
        ru
    };
    #[allow(clippy::cast_precision_loss, reason = "second counts for a ratio")]
    let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 * 1e-6;
    t(ru.ru_utime) + t(ru.ru_stime)
}

/// The machine's streaming read ceiling: `threads` threads each summing a disjoint slice of a
/// 1 GiB buffer (far larger than the 36 MB L3), best of three passes, in GB/s.
///
/// # Complexity
///
/// Three passes over 1 GiB plus one initializing write.
#[allow(clippy::cast_precision_loss, reason = "a bandwidth ratio")]
fn mem_read_gbs(threads: usize) -> f64 {
    const WORDS: usize = 1 << 27;
    let buf: Vec<u64> = (0..WORDS as u64).collect();
    let per = WORDS / threads;
    let mut best = 0.0f64;
    for _ in 0..3 {
        let t0 = Instant::now();
        let total: u64 = std::thread::scope(|sc| {
            let hs: Vec<_> = buf
                .chunks(per)
                .map(|c| sc.spawn(move || c.iter().fold(0u64, |a, &x| a.wrapping_add(x))))
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or(0))
                .fold(0u64, u64::wrapping_add)
        });
        std::hint::black_box(total);
        best = best.max((WORDS * 8) as f64 / t0.elapsed().as_secs_f64() / 1e9);
    }
    best
}

fn sh(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn sha256_file(path: &str) -> String {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return "unreadable".into();
    };
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match f.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => h.update(&buf[..n]),
        }
    }
    hex(&h.finalize())
}

/// sha256 over every `.rs` file under `root`, sorted by path, each prefixed by its path.
fn sha256_tree(root: &str) -> String {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
    }
    let mut files = Vec::new();
    walk(std::path::Path::new(root), &mut files);
    files.sort();
    let mut h = Sha256::new();
    for f in &files {
        h.update(f.to_string_lossy().as_bytes());
        h.update(std::fs::read(f).unwrap_or_default());
    }
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    b.iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

// ---- workloads -----------------------------------------------------------------------------

/// One timed engine workload.
#[derive(Clone, Copy, Default)]
struct Sample {
    secs: f64,
    cpu: f64,
    mhz: f64,
    tokens: usize,
}

impl Sample {
    #[allow(clippy::cast_precision_loss, reason = "a token rate")]
    fn tps(self) -> f64 {
        self.tokens as f64 / self.secs
    }
    fn cores(self) -> f64 {
        self.cpu / self.secs
    }
}

fn timed(
    tokens: usize,
    f: impl FnOnce() -> Result<(), Box<dyn Error>>,
) -> Result<Sample, Box<dyn Error>> {
    let mhz = mean_mhz();
    let c0 = cpu_seconds();
    let t0 = Instant::now();
    f()?;
    let secs = t0.elapsed().as_secs_f64();
    Ok(Sample {
        secs,
        cpu: cpu_seconds() - c0,
        mhz,
        tokens,
    })
}

fn run_pp(e: &Engine, ids: &[u32]) -> Result<Sample, Box<dyn Error>> {
    timed(ids.len(), || {
        let mut st = e.state(ids.len() + 1)?;
        e.prefill_state(&mut st, ids, KernelMode::Q8Act)?;
        Ok(())
    })
}

fn run_tg(e: &Engine, first: u32, n: usize) -> Result<Sample, Box<dyn Error>> {
    let mut st = e.state(n + 1)?;
    timed(n, || {
        let mut t = first;
        for _ in 0..n {
            t = e.next_token_state(&mut st, t, KernelMode::Q8Act)?;
        }
        Ok(())
    })
}

fn run_spec(
    e: &Engine,
    first: u32,
    n: usize,
    rows: usize,
) -> Result<(Sample, f64), Box<dyn Error>> {
    let mut acc = 0.0;
    let s = timed(n, || {
        let (_, stats) = e.generate_speculative(&[first], n, rows)?;
        #[allow(clippy::cast_precision_loss, reason = "an acceptance ratio")]
        {
            acc = stats.accepted as f64 / stats.rounds.max(1) as f64;
        }
        Ok(())
    })?;
    Ok((s, acc))
}

/// One `llama-bench` row: prompt tokens, generated tokens, threads, mean tokens/s.
#[derive(Clone, Copy)]
struct LlamaRow {
    n_prompt: usize,
    n_gen: usize,
    threads: usize,
    tps: f64,
}

/// The numeric value after `"key":` in a flat JSON object.
fn json_num(obj: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let at = obj.find(&pat)? + pat.len();
    let rest = obj[at..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn json_text(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let at = obj.find(&pat)? + pat.len();
    let rest = obj[at..].trim_start().strip_prefix('"')?;
    Some(rest[..rest.find('"')?].to_string())
}

/// Parses one JSON fragment. Time is O(B) byte scans over the fragment (four numeric keys and
/// one commit key); space is O(C) for the returned build-commit string of C bytes.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "validated integral llama-bench counts are bounded before conversion"
)]
fn parse_llama_row(obj: &str) -> Result<Option<(LlamaRow, String)>, std::io::Error> {
    use std::io::{Error as IoError, ErrorKind};

    let parsed = (
        json_num(obj, "n_prompt"),
        json_num(obj, "n_gen"),
        json_num(obj, "n_threads"),
        json_num(obj, "avg_ts"),
    );
    if parsed == (None, None, None, None) {
        return Ok(None);
    }
    let (Some(p), Some(g), Some(t), Some(ts)) = parsed else {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "malformed llama-bench row",
        ));
    };
    if !p.is_finite()
        || !g.is_finite()
        || !t.is_finite()
        || p < 0.0
        || g < 0.0
        || t <= 0.0
        || p.fract() != 0.0
        || g.fract() != 0.0
        || t.fract() != 0.0
        || p >= usize::MAX as f64
        || g >= usize::MAX as f64
        || t >= usize::MAX as f64
        || !ts.is_finite()
        || ts <= 0.0
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "invalid llama-bench row value",
        ));
    }
    let row = LlamaRow {
        n_prompt: p as usize,
        n_gen: g as usize,
        threads: t as usize,
        tps: ts,
    };
    let commit = json_text(obj, "build_commit")
        .filter(|c| !c.is_empty())
        .ok_or_else(|| {
            IoError::new(
                ErrorKind::InvalidData,
                "llama-bench row has no build_commit",
            )
        })?;
    Ok(Some((row, commit)))
}

/// Validates standalone prefill and generation rows. Time is O(B + R + T) expected, where B is
/// output bytes, R is parsed rows and T is requested thread counts: each output fragment is parsed
/// once and each row/thread membership check uses a hash set. Space is O(R + T) for rows and sets.
fn parse_llama_output(
    text: &str,
    pp: usize,
    tg: usize,
    threads: &str,
) -> Result<(Vec<LlamaRow>, String), std::io::Error> {
    use std::collections::HashSet;
    use std::io::{Error as IoError, ErrorKind};

    let requested: Vec<usize> = threads
        .split(',')
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| IoError::new(ErrorKind::InvalidInput, e))
        })
        .collect::<Result<_, _>>()?;
    let requested_set: HashSet<_> = requested.iter().copied().collect();
    if requested.is_empty() || requested.contains(&0) || requested_set.len() != requested.len() {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "llama thread counts must be nonzero and unique",
        ));
    }

    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let mut build_commit: Option<String> = None;
    for obj in text.split('}') {
        let Some((row, commit)) = parse_llama_row(obj)? else {
            continue;
        };
        if (row.n_prompt, row.n_gen) != (pp, 0) && (row.n_prompt, row.n_gen) != (0, tg) {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "unexpected llama-bench workload row",
            ));
        }
        if !requested_set.contains(&row.threads) {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "unexpected llama-bench thread count",
            ));
        }
        if !seen.insert((row.n_prompt, row.n_gen, row.threads)) {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "duplicate llama-bench row",
            ));
        }
        if build_commit
            .as_ref()
            .is_some_and(|previous| previous != &commit)
        {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "inconsistent llama-bench build_commit",
            ));
        }
        build_commit = Some(commit);
        rows.push(row);
    }

    let expected = requested.len() * 2;
    if rows.len() != expected {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "expected {expected} standalone llama-bench rows, found {}",
                rows.len()
            ),
        ));
    }
    for &thread_count in &requested {
        if !seen.contains(&(pp, 0, thread_count)) || !seen.contains(&(0, tg, thread_count)) {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                format!("missing llama-bench pp/tg row at {thread_count} threads"),
            ));
        }
    }
    Ok((
        rows,
        build_commit.expect("nonempty expected row set has a commit"),
    ))
}

/// Environment that hides every GPU backend from the comparator child. This engine is a CPU
/// implementation, so a `llama-bench` that keeps its CUDA or Vulkan device silently compares CPU
/// against GPU. `CUDA_VISIBLE_DEVICES=""` hides CUDA at the runtime and `GGML_VK_VISIBLE_DEVICES=""`
/// selects no Vulkan device (`ggml-vulkan.cpp` requires every listed index to be in range, so the
/// empty list exposes none); both are cheap and backend-independent on this machine.
const CPU_ONLY_ENV: [(&str, &str); 2] = [
    ("CUDA_VISIBLE_DEVICES", ""),
    ("GGML_VK_VISIBLE_DEVICES", ""),
];

/// Builds the CPU-only `llama-bench` child. Time and space are O(1): two environment assignments;
/// the argument vector and the affinity reset are applied by the caller.
fn llama_command(bin: &str) -> Command {
    let mut cmd = Command::new(bin);
    cmd.envs(CPU_ONLY_ENV);
    cmd
}

/// `llama-bench`'s own report of the device it used and the backends it was built with. An empty
/// `gpu_info` means no GPU backend exposed a device to the child: the rows are CPU. The receipt
/// keeps this report so a reader can see the comparator was CPU, not take it on trust.
#[derive(Clone, Default)]
struct ComparatorDevice {
    gpu_info: String,
    backends: String,
}

/// Reads the comparator's device report and rejects a loaded GPU. `gpu_info` nonempty means the
/// child kept a device, so the comparison would be CPU against GPU: fail rather than label it CPU.
///
/// Time is O(B) over the B output bytes; space is O(D) for the two D-byte report strings.
fn comparator_device(text: &str) -> Result<ComparatorDevice, std::io::Error> {
    use std::io::{Error as IoError, ErrorKind};

    let device = ComparatorDevice {
        gpu_info: json_text(text, "gpu_info").unwrap_or_default(),
        backends: json_text(text, "backends").unwrap_or_default(),
    };
    if !device.gpu_info.is_empty() {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "llama-bench exposed a GPU device ({:?}); refusing to label a non-CPU comparator CPU",
                device.gpu_info
            ),
        ));
    }
    Ok(device)
}

/// Runs llama-bench and rejects failed or incomplete results. Time is the child benchmark plus
/// O(B + R + T) expected validation for its B output bytes, R rows and T requested thread counts;
/// space is O(B + R + T) for captured output and parsed collections.
fn run_llama(
    bin: &str,
    model: &str,
    pp: usize,
    tg: usize,
    threads: &str,
) -> Result<(Vec<LlamaRow>, String, ComparatorDevice), Box<dyn Error>> {
    let mut cmd = llama_command(bin);
    cmd.args(["-m", model, "-p", &pp.to_string(), "-n", &tg.to_string()])
        .args(["-r", "1", "-t", threads, "-o", "json"]);
    // The engine's pool pins this thread to one core and a child inherits the mask: unpinned,
    // `llama-bench` ran every thread on CPU 0 (tg 22 instead of 45 tok/s). Reset it in the child.
    // SAFETY: `sched_setaffinity` is async-signal-safe; the set is a zeroed local with every CPU
    // below `CPU_SETSIZE` added.
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(|| {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for cpu in 0..libc::CPU_SETSIZE as usize {
                libc::CPU_SET(cpu, &mut set);
            }
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw const set)
                != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(format!(
            "llama-bench failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let device = comparator_device(&stdout)?;
    let (rows, commit) = parse_llama_output(&stdout, pp, tg, threads)?;
    Ok((rows, commit, device))
}

// ---- statistics ----------------------------------------------------------------------------

fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(f64::total_cmp);
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        f64::midpoint(s[m - 1], s[m])
    }
}

fn span(v: &[f64]) -> (f64, f64) {
    v.iter()
        .filter(|x| x.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        })
}

// ---- traced run ----------------------------------------------------------------------------

/// Phase table of one traced workload over `tokens` tokens, as Markdown and JSON.
#[allow(
    clippy::cast_precision_loss,
    reason = "nanosecond and byte ratios for display"
)]
fn phase_report(name: &str, tokens: usize, wall: f64) -> (String, String) {
    let stats = trace::snapshot();
    let tok = tokens.max(1) as f64;
    let enclosing: u64 = stats.iter().filter(|s| s.encloses).map(|s| s.ns).sum();
    let leaves: u64 = stats
        .iter()
        .filter(|s| !s.encloses && !s.overlay && !matches!(s.name, "head" | "restore"))
        .map(|s| s.ns)
        .sum();
    let outside: u64 = stats
        .iter()
        .filter(|s| matches!(s.name, "head" | "restore"))
        .map(|s| s.ns)
        .sum();
    let wall_ns = wall * 1e9;
    let mut md = format!(
        "\n#### {name}: {tokens} tokens, {:.2} ms/token wall\n\n\
         | phase | ms/token | share of wall | calls | GB/s | GMAC/s |\n|---|---:|---:|---:|---:|---:|\n",
        wall * 1e3 / tok
    );
    let mut js = format!(
        "{{\"workload\":{},\"tokens\":{tokens},\"wall_s\":{wall},\"phases\":[",
        json_str(name)
    );
    let mut first = true;
    let mut row = |label: &str, ns: u64, calls: u64, bytes: u64, macs: u64, md: &mut String| {
        if ns == 0 {
            return;
        }
        let secs = ns as f64 * 1e-9;
        let gbs = if bytes > 0 {
            format!("{:.1}", bytes as f64 / secs / 1e9)
        } else {
            "-".into()
        };
        let gms = if macs > 0 {
            format!("{:.1}", macs as f64 / secs / 1e9)
        } else {
            "-".into()
        };
        let _ = writeln!(
            md,
            "| {label} | {:.3} | {:.1}% | {calls} | {gbs} | {gms} |",
            ns as f64 / 1e6 / tok,
            100.0 * ns as f64 / wall_ns
        );
        if !first {
            js.push(',');
        }
        first = false;
        let _ = write!(
            js,
            "{{\"phase\":{},\"ns\":{ns},\"calls\":{calls},\"bytes\":{bytes},\"macs\":{macs}}}",
            json_str(label)
        );
    };
    for s in stats.iter().filter(|s| !s.encloses && !s.overlay) {
        row(s.name, s.ns, s.calls, s.bytes, s.macs, &mut md);
    }
    row(
        "untraced in pass",
        enclosing.saturating_sub(leaves),
        0,
        0,
        0,
        &mut md,
    );
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "wall ns"
    )]
    let rest = (wall_ns as u64).saturating_sub(enclosing + outside);
    row("outside passes", rest, 0, 0, 0, &mut md);
    for s in stats.iter().filter(|s| s.overlay) {
        let label = format!(
            "{} (overlay; tasks/dispatch {:.1})",
            s.name,
            s.cols as f64 / s.calls.max(1) as f64
        );
        row(&label, s.ns, s.calls, 0, 0, &mut md);
    }
    js.push_str("]}");
    (md, js)
}

// ---- A/B arms ------------------------------------------------------------------------------

/// One arm's round: the engine workloads once after a warm-up, printed as one line.
fn run_child(a: &Args, ids: &[u32]) -> Result<(), Box<dyn Error>> {
    let engine = Engine::open(&a.model)?;
    run_pp(&engine, ids)?;
    run_tg(&engine, ids[0], a.tg)?;
    let p = run_pp(&engine, ids)?;
    let t = run_tg(&engine, ids[0], a.tg)?;
    let (s, acc) = match a.spec {
        Some(rows) => {
            run_spec(&engine, ids[0], a.tg, rows)?;
            let (s, acc) = run_spec(&engine, ids[0], a.tg, rows)?;
            (s.tps(), acc)
        }
        None => (f64::NAN, f64::NAN),
    };
    println!(
        "CHILD {} {} {s} {acc} {} {}",
        p.tps(),
        t.tps(),
        p.mhz,
        t.mhz
    );
    Ok(())
}

/// Interleaved A/B: every round runs every arm as its own process (rotating the order by round,
/// so neither arm always runs first), each under its environment; the summary reports each arm's
/// medians and its per-round ratio to the first arm.
#[allow(clippy::too_many_lines, reason = "the A/B protocol in order")]
fn run_arms(a: &Args) -> Result<(), Box<dyn Error>> {
    let exe = std::env::current_exe()?;
    let n = a.arms.len();
    // metric index: 0 pp, 1 tg, 2 spec
    let mut vals: Vec<[Vec<f64>; 3]> = (0..n).map(|_| [vec![], vec![], vec![]]).collect();
    let mut ratios: Vec<[Vec<f64>; 3]> = (0..n).map(|_| [vec![], vec![], vec![]]).collect();
    let mut rounds = String::new();
    let mut quiet_all = true;
    for r in 0..a.reps {
        let (busy, quiet) = wait_quiet(a.max_busy);
        quiet_all &= quiet;
        let mut got: Vec<[f64; 3]> = vec![[f64::NAN; 3]; n];
        for i in 0..n {
            let arm = (i + r) % n;
            let (name, env) = &a.arms[arm];
            // `EXE=<path>` selects another engine_profile build for this arm (a binary A/B in one
            // interleaved run); it is not passed to the child's environment.
            let arm_exe = env
                .iter()
                .find(|(k, _)| k == "EXE")
                .map_or_else(|| exe.clone(), |(_, v)| std::path::PathBuf::from(v));
            let mut cmd = Command::new(&arm_exe);
            cmd.args(["--child", "--model", &a.model, "--ids", &a.ids])
                .args(["--pp", &a.pp.to_string(), "--tg", &a.tg.to_string()]);
            if let Some(rows) = a.spec {
                cmd.args(["--spec", &rows.to_string()]);
            }
            for (k, v) in env.iter().filter(|(k, _)| k != "EXE") {
                cmd.env(k, v);
            }
            let out = cmd.output()?;
            let text = String::from_utf8_lossy(&out.stdout);
            let line = text
                .lines()
                .find(|l| l.starts_with("CHILD "))
                .ok_or_else(|| {
                    format!(
                        "arm {name}: no result ({})",
                        String::from_utf8_lossy(&out.stderr)
                    )
                })?;
            let f: Vec<f64> = line
                .split_whitespace()
                .skip(1)
                .map(|x| x.parse().unwrap_or(f64::NAN))
                .collect();
            got[arm] = [f[0], f[1], f[2]];
            let _ = write!(
                rounds,
                "{}{{\"round\":{r},\"arm\":{},\"busy\":{busy},\"quiet\":{quiet},\"pp\":{},\"tg\":{},\
                 \"spec\":{},\"spec_accept\":{},\"mhz_pp\":{},\"mhz_tg\":{}}}",
                if rounds.is_empty() { "" } else { "," },
                json_str(name),
                f[0],
                f[1],
                if f[2].is_nan() { "null".into() } else { f[2].to_string() },
                if f[3].is_nan() { "null".into() } else { f[3].to_string() },
                f[4],
                f[5]
            );
        }
        for arm in 0..n {
            for m in 0..3 {
                vals[arm][m].push(got[arm][m]);
                ratios[arm][m].push(got[arm][m] / got[0][m]);
            }
        }
        eprintln!(
            "round {r}: busy {busy:.3} quiet {quiet}  {}",
            (0..n)
                .map(|i| format!(
                    "{} pp {:.1} tg {:.2} spec {:.2}",
                    a.arms[i].0, got[i][0], got[i][1], got[i][2]
                ))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let arms_js: Vec<String> = a
        .arms
        .iter()
        .map(|(name, env)| {
            let e: Vec<String> = env
                .iter()
                .map(|(k, v)| format!("{}:{}", json_str(k), json_str(v)))
                .collect();
            format!(
                "{{\"name\":{},\"env\":{{{}}}}}",
                json_str(name),
                e.join(",")
            )
        })
        .collect();
    let js = format!(
        "{{\"schema\":\"apeiron.bench.ab.v1\",\"unix_time\":{stamp},\"label\":{},\
         \"protocol\":{{\"pp\":{},\"tg\":{},\"reps\":{},\"spec_rows\":{},\"max_busy\":{}}},\
         \"identity\":{{\"git_head\":{},\"src_sha256\":{},\"model\":{},\"ids\":{}}},\
         \"arms\":[{}],\"rounds\":[{rounds}]}}",
        json_str(&a.label),
        a.pp,
        a.tg,
        a.reps,
        a.spec.map_or("null".into(), |x| x.to_string()),
        a.max_busy,
        json_str(&sh("git", &["rev-parse", "HEAD"])),
        json_str(&sha256_tree("src")),
        json_str(&a.model),
        json_str(&a.ids),
        arms_js.join(",")
    );
    let mut md = format!(
        "# A/B `{}`\n\n- pp {}, tg {}, {} rounds, arm order rotated per round; quiet gate busy < {}: {}\n\n\
         | arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |\n\
         |---|---|---:|---:|---:|---:|---:|---:|\n",
        a.label,
        a.pp,
        a.tg,
        a.reps,
        a.max_busy,
        if quiet_all { "every round passed" } else { "at least one round timed out (flagged)" }
    );
    for (i, (name, env)) in a.arms.iter().enumerate() {
        let e: Vec<String> = env.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let _ = writeln!(
            md,
            "| {name} | `{}` | {:.2} | {:.2} | {:.2} | {:.3} | {:.3} | {:.3} |",
            e.join(" "),
            median(&vals[i][0]),
            median(&vals[i][1]),
            median(&vals[i][2]),
            median(&ratios[i][0]),
            median(&ratios[i][1]),
            median(&ratios[i][2])
        );
    }
    md.push_str("\nRatios are the median of per-round ratios to the first arm.\n");
    std::fs::create_dir_all(&a.out)?;
    let base = format!("{}/{stamp}-{}", a.out, a.label);
    std::fs::write(format!("{base}.json"), js)?;
    std::fs::write(format!("{base}.md"), &md)?;
    println!("{md}");
    eprintln!("wrote {base}.json and {base}.md");
    Ok(())
}

// ---- main ----------------------------------------------------------------------------------

#[allow(clippy::too_many_lines, reason = "the protocol in order")]
fn run(a: &Args) -> Result<(), Box<dyn Error>> {
    let all: Vec<u32> = std::fs::read_to_string(&a.ids)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if all.is_empty() {
        return Err("empty prompt-id file".into());
    }
    let ids: Vec<u32> = all.iter().copied().cycle().take(a.pp).collect();
    let first = ids[0];
    if a.child {
        return run_child(a, &ids);
    }
    if !a.arms.is_empty() {
        return run_arms(a);
    }

    eprintln!("identity ...");
    let id = [
        (
            "cpu",
            sh(
                "sh",
                &["-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
            ),
        ),
        ("online_cpus", sh("nproc", &["--all"])),
        (
            "governor",
            sh(
                "cat",
                &["/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"],
            ),
        ),
        ("kernel", sh("uname", &["-r"])),
        ("rustc", sh("rustc", &["-V"])),
        ("git_head", sh("git", &["rev-parse", "HEAD"])),
        (
            "git_dirty_paths",
            sh("sh", &["-c", "git status --porcelain | wc -l"]),
        ),
        ("src_sha256", sha256_tree("src")),
        ("model", a.model.clone()),
        ("model_sha256", sha256_file(&a.model)),
        ("ids_sha256", sha256_file(&a.ids)),
        ("trace_default", format!("{}", trace::enabled())),
        ("mem_read_gbs_8t", format!("{:.1}", mem_read_gbs(8))),
        ("mem_read_gbs_16t", format!("{:.1}", mem_read_gbs(16))),
        ("mem_read_gbs_32t", format!("{:.1}", mem_read_gbs(32))),
    ];
    trace::set_enabled(false);

    let engine = Engine::open(&a.model)?;
    eprintln!("warm-up ...");
    run_pp(&engine, &ids)?;
    run_tg(&engine, first, a.tg)?;
    if let Some(rows) = a.spec {
        run_spec(&engine, first, a.tg, rows)?;
    }

    let mut rounds = String::new();
    let (mut pp, mut tg, mut sp, mut acc) = (vec![], vec![], vec![], vec![]);
    let (mut lpp, mut ltg) = (vec![], vec![]);
    let (mut pcores, mut tcores, mut pmhz, mut tmhz) = (vec![], vec![], vec![], vec![]);
    let (mut rpp, mut rtg) = (vec![], vec![]);
    let mut llama_commit = String::new();
    let mut llama_device = ComparatorDevice::default();
    let mut quiet_all = true;
    for r in 0..a.reps {
        let (busy, quiet) = wait_quiet(a.max_busy);
        quiet_all &= quiet;
        eprintln!("round {r}: busy {busy:.3} quiet {quiet}");
        let (lrows, commit, device) = match &a.llama {
            Some(bin) => run_llama(bin, &a.model, a.pp, a.tg, &a.llama_threads)?,
            None => (Vec::new(), String::new(), ComparatorDevice::default()),
        };
        if !commit.is_empty() {
            llama_commit = commit;
        }
        if llama_device.gpu_info.is_empty() && llama_device.backends.is_empty() {
            llama_device = device;
        }
        let best = |f: &dyn Fn(&LlamaRow) -> bool| {
            lrows
                .iter()
                .filter(|x| f(x))
                .map(|x| (x.tps, x.threads))
                .fold(
                    (f64::NAN, 0),
                    |b, x| {
                        if b.0.is_nan() || x.0 > b.0 {
                            x
                        } else {
                            b
                        }
                    },
                )
        };
        let lp = best(&|x| x.n_prompt > 0);
        let lt = best(&|x| x.n_gen > 0);
        // Unrecorded warm-up right before the measured workloads, matching `llama-bench`'s
        // default per-test warm-up run (the `--llama` wrapper does not pass `--no-warmup`).
        run_pp(&engine, &ids)?;
        run_tg(&engine, first, a.tg)?;
        let p = run_pp(&engine, &ids)?;
        let t = run_tg(&engine, first, a.tg)?;
        let s = match a.spec {
            Some(rows) => Some(run_spec(&engine, first, a.tg, rows)?),
            None => None,
        };
        pp.push(p.tps());
        pcores.push(p.cores());
        tcores.push(t.cores());
        pmhz.push(p.mhz);
        tmhz.push(t.mhz);
        tg.push(t.tps());
        lpp.push(lp.0);
        ltg.push(lt.0);
        rpp.push(p.tps() / lp.0);
        rtg.push(t.tps() / lt.0);
        if let Some((s, a2)) = s {
            sp.push(s.tps());
            acc.push(a2);
        }
        if !rounds.is_empty() {
            rounds.push(',');
        }
        let lj: Vec<String> = lrows
            .iter()
            .map(|x| {
                format!(
                    "{{\"n_prompt\":{},\"n_gen\":{},\"threads\":{},\"tps\":{}}}",
                    x.n_prompt, x.n_gen, x.threads, x.tps
                )
            })
            .collect();
        let sample = |x: Sample| {
            format!(
                "{{\"tps\":{},\"secs\":{},\"cpu_s\":{},\"cores\":{},\"mhz\":{}}}",
                x.tps(),
                x.secs,
                x.cpu,
                x.cores(),
                x.mhz
            )
        };
        let _ = write!(
            rounds,
            "{{\"round\":{r},\"busy\":{busy},\"quiet\":{quiet},\"pp\":{},\"tg\":{},\"spec\":{},\
             \"spec_accept\":{},\"llama\":[{}]}}",
            sample(p),
            sample(t),
            s.map_or("null".into(), |(x, _)| sample(x)),
            s.map_or("null".into(), |(_, x)| x.to_string()),
            lj.join(",")
        );
        eprintln!(
            "  pp {:.1} tok/s ({:.1} cores)  tg {:.2} tok/s ({:.1} cores)  llama pp {:.1}@{}t tg {:.2}@{}t",
            p.tps(),
            p.cores(),
            t.tps(),
            t.cores(),
            lp.0,
            lp.1,
            lt.0,
            lt.1
        );
    }

    eprintln!("traced run ...");
    trace::set_enabled(true);
    let mut traces_md = String::new();
    let mut traces_js = Vec::new();
    let mut traced = |name: &str,
                      tokens: usize,
                      f: &dyn Fn() -> Result<Sample, Box<dyn Error>>|
     -> Result<(), Box<dyn Error>> {
        trace::reset();
        let s = f()?;
        let (m, j) = phase_report(name, tokens, s.secs);
        traces_md.push_str(&m);
        traces_js.push(j);
        Ok(())
    };
    traced("pp", a.pp, &|| run_pp(&engine, &ids))?;
    traced("tg", a.tg, &|| run_tg(&engine, first, a.tg))?;
    if let Some(rows) = a.spec {
        traced("spec", a.tg, &|| {
            run_spec(&engine, first, a.tg, rows).map(|x| x.0)
        })?;
    }
    trace::set_enabled(false);

    // ---- write ----
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let mut js = String::from("{\"schema\":\"apeiron.bench.v1\",");
    let _ = write!(
        js,
        "\"unix_time\":{stamp},\"label\":{},",
        json_str(&a.label)
    );
    let _ = write!(
        js,
        "\"protocol\":{{\"pp\":{},\"tg\":{},\"reps\":{},\"spec_rows\":{},\"max_busy\":{},\"llama_threads\":{}}},",
        a.pp,
        a.tg,
        a.reps,
        a.spec.map_or("null".into(), |x| x.to_string()),
        a.max_busy,
        json_str(&a.llama_threads)
    );
    js.push_str("\"identity\":{");
    for (i, (k, v)) in id.iter().enumerate() {
        if i > 0 {
            js.push(',');
        }
        let _ = write!(js, "{}:{}", json_str(k), json_str(v));
    }
    let _ = write!(
        js,
        ",\"llama_commit\":{},\"llama_gpu_info\":{},\"llama_backends\":{}}},",
        json_str(&llama_commit),
        json_str(&llama_device.gpu_info),
        json_str(&llama_device.backends)
    );
    let _ = write!(
        js,
        "\"rounds\":[{rounds}],\"traces\":[{}]}}",
        traces_js.join(",")
    );

    let row = |name: &str, v: &[f64]| {
        let (lo, hi) = span(v);
        format!("| {name} | {:.2} | {lo:.2} | {hi:.2} |\n", median(v))
    };
    let mut md = format!(
        "# Engine benchmark `{}`\n\n\
         - model `{}` (sha256 `{}`)\n- prompt ids `{}` (sha256 `{}`), pp {}, tg {}, {} rounds\n\
         - git `{}` with {} dirty paths; src sha256 `{}`\n- {}; governor {}; {} CPUs; {}\n\
         - read ceiling (1 GiB sum, best of 3): {} / {} / {} GB/s at 8 / 16 / 32 threads\n\
         - llama.cpp `{}`, threads {}; comparator gpu_info `{}` backends `{}`\n\
         - quiet gate busy < {}: {}\n\n\
         | metric (tok/s) | median | min | max |\n|---|---:|---:|---:|\n",
        a.label,
        a.model,
        id[9].1,
        a.ids,
        id[10].1,
        a.pp,
        a.tg,
        a.reps,
        id[5].1,
        id[6].1,
        id[7].1,
        id[0].1.trim(),
        id[2].1,
        id[1].1,
        id[4].1,
        id[12].1,
        id[13].1,
        id[14].1,
        llama_commit,
        a.llama_threads,
        llama_device.gpu_info,
        llama_device.backends,
        a.max_busy,
        if quiet_all {
            "every round passed"
        } else {
            "at least one round timed out (flagged)"
        }
    );
    md.push_str(&row("apeiron pp", &pp));
    md.push_str(&row("llama.cpp pp (best threads)", &lpp));
    md.push_str(&row("ratio pp (per round)", &rpp));
    md.push_str(&row("apeiron tg", &tg));
    md.push_str(&row("llama.cpp tg (best threads)", &ltg));
    md.push_str(&row("ratio tg (per round)", &rtg));
    if !sp.is_empty() {
        md.push_str(&row("apeiron spec (e2e)", &sp));
        md.push_str(&row("spec acceptance", &acc));
    }
    md.push_str(&row("apeiron pp busy cores", &pcores));
    md.push_str(&row("apeiron tg busy cores", &tcores));
    md.push_str(&row("mean MHz before pp", &pmhz));
    md.push_str(&row("mean MHz before tg", &tmhz));
    md.push_str("\n### Traced phases\n");
    md.push_str(&traces_md);

    std::fs::create_dir_all(&a.out)?;
    let base = format!("{}/{stamp}-{}", a.out, a.label);
    std::fs::write(format!("{base}.json"), js)?;
    std::fs::write(format!("{base}.md"), &md)?;
    println!("{md}");
    eprintln!("wrote {base}.json and {base}.md");
    Ok(())
}

#[cfg(test)]
mod comparator_device_tests {
    use super::comparator_device;

    #[test]
    fn accepts_a_cpu_report_and_rejects_a_gpu_device() {
        let cpu = r#"{"gpu_info": "", "backends": "CPU", "n_prompt": 200}"#;
        let dev = comparator_device(cpu).unwrap();
        assert!(dev.gpu_info.is_empty());
        assert_eq!(dev.backends, "CPU");

        let gpu = r#"{"gpu_info": "NVIDIA GeForce RTX 5080", "backends": "CUDA"}"#;
        assert!(comparator_device(gpu).is_err());
    }
}

#[cfg(test)]
mod llama_command_tests {
    use super::llama_command;

    /// Every backend that can silently own the comparator must be present and empty: an unset
    /// variable lets the backend default to every device. The expected names and the empty value
    /// are hard-coded, not read from [`super::CPU_ONLY_ENV`], so a weakened constant fails here.
    const REQUIRED_HIDDEN: [&str; 2] = ["CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"];

    #[test]
    fn child_env_hides_gpu_backends() {
        let cmd = llama_command("llama-bench");
        let envs: Vec<(String, Option<String>)> = cmd
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect();
        for key in REQUIRED_HIDDEN {
            let got = envs
                .iter()
                .find(|(k, _)| k == key)
                .unwrap_or_else(|| panic!("{key} is not set on the comparator child"));
            assert_eq!(
                got.1.as_deref(),
                Some(""),
                "{key} must be explicitly empty to hide its backend"
            );
        }
    }
}

#[cfg(test)]
mod llama_output_tests {
    use super::parse_llama_output;

    const THREADS: &str = "8,16";
    const VALID: &str = r#"{"n_prompt":200,"n_gen":0,"n_threads":8,"avg_ts":250.0,"build_commit":"abc"}
{"n_prompt":0,"n_gen":64,"n_threads":8,"avg_ts":45.0,"build_commit":"abc"}
{"n_prompt":200,"n_gen":0,"n_threads":16,"avg_ts":400.0,"build_commit":"abc"}
{"n_prompt":0,"n_gen":64,"n_threads":16,"avg_ts":50.0,"build_commit":"abc"}"#;

    #[test]
    fn accepts_complete_standalone_rows_with_one_build_commit() {
        let (rows, commit) = parse_llama_output(VALID, 200, 64, THREADS).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(commit, "abc");
    }

    #[test]
    fn rejects_missing_duplicate_and_malformed_rows() {
        let missing = VALID.replace(
            "{\"n_prompt\":0,\"n_gen\":64,\"n_threads\":16,\"avg_ts\":50.0,\"build_commit\":\"abc\"}",
            "",
        );
        assert!(parse_llama_output(&missing, 200, 64, THREADS).is_err());
        let duplicate = format!(
            "{VALID}\n{{\"n_prompt\":200,\"n_gen\":0,\"n_threads\":8,\"avg_ts\":250.0,\"build_commit\":\"abc\"}}"
        );
        assert!(parse_llama_output(&duplicate, 200, 64, THREADS).is_err());
        let malformed = VALID.replace("\"avg_ts\":50.0", "\"avg_ts\":NaN");
        assert!(parse_llama_output(&malformed, 200, 64, THREADS).is_err());
        let fractional = VALID.replace("\"n_threads\":16", "\"n_threads\":16.5");
        assert!(parse_llama_output(&fractional, 200, 64, THREADS).is_err());
        let oversized = VALID.replace("\"n_threads\":16", "\"n_threads\":1e30");
        assert!(parse_llama_output(&oversized, 200, 64, THREADS).is_err());
    }

    #[test]
    fn rejects_inconsistent_build_commits() {
        let inconsistent = VALID.replace(
            "\"avg_ts\":50.0,\"build_commit\":\"abc\"",
            "\"avg_ts\":50.0,\"build_commit\":\"def\"",
        );
        assert!(parse_llama_output(&inconsistent, 200, 64, THREADS).is_err());
    }
}

fn main() -> ExitCode {
    match parse_args().and_then(|a| run(&a)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("engine_profile: {e}");
            ExitCode::FAILURE
        }
    }
}
