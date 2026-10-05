//! swalk — `select` over `walk`, the unified search.
//!
//! The invariant (UniversalEncoding on the search family): every one of find/fd/grep/rg is
//! `select(boundary : Term) over universalLift(walk)`. Here:
//!   * `walk`     — one iterative directory traversal (the tree as an α-stream; no per-source code).
//!   * `boundary` — a `tregex::Regex` (the pattern as `Polymorphic.Term`, the membership GET).
//!   * `select`   — apply the boundary at each walked element; matches fall out (the predicate floor).
//! The only thing that varies is the **applicative** (the kind), reducing the five tools to three:
//!   * `name`    — boundary over a node's filename  (fd ≡ findutils)   — kind 1, the WHERE.
//!   * `content` — boundary over a file's lines     (ripgrep ≡ grep)   — kind 2, the WHAT.
//!   (`awk` — kind 3, pattern + an action Term — is the reflexive fourth, a later rung.)
//!
//! Usage:
//!   swalk name    <pattern> [path]   [-c]
//!   swalk content <pattern> [path]   [-c]   [-a]
//! Flags: -c count only (like grep -c / rg -c) ; -a scan all files incl. binary (content).

use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use hashtrinity::tregex::Regex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Default, Clone)]
struct Stats {
    seen: u64,
    visited: u64,
    matches: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// bench — the three-walk measurement, absorbed from qwalk: full vs quantum vs
// NB-summary pruning. Usage: swalk bench <needle> [path]
// ─────────────────────────────────────────────────────────────────────────────
fn quantum_keep(s_n_1: f64, s_n_2: f64, is_dir: bool) -> (bool, f64) {
    let cost_base = if is_dir { 0.1 } else { 0.05 };
    let s_n = s_n_1 + cost_base * (1.0 + s_n_1);
    let h = s_n - 2.0 * s_n_1 + s_n_2;
    let a = 1.0 / ((h.abs() + 1e-8).sqrt());
    (a >= 0.1, s_n)
}

const BW: usize = 64;
type Bloom = [u64; BW];

#[inline]
fn tri_bit(w: &[u8]) -> usize {
    let mut h = 0xcbf29ce484222325u64;
    for &b in w {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    (h % (BW as u64 * 64)) as usize
}
fn bloom_add_name(b: &mut Bloom, name: &[u8]) {
    if name.len() < 3 {
        let i = tri_bit(name);
        b[i >> 6] |= 1u64 << (i & 63);
    } else {
        for w in name.windows(3) {
            let i = tri_bit(w);
            b[i >> 6] |= 1u64 << (i & 63);
        }
    }
}
fn bloom_or(a: &mut Bloom, c: &Bloom) {
    for i in 0..BW {
        a[i] |= c[i];
    }
}
/// the NB membership read: can a descendant name contain `needle`? (necessary condition).
fn bloom_admits(b: &Bloom, needle: &[u8]) -> bool {
    if needle.len() < 3 {
        return true; // too short to trigram-filter — descend (still lossless)
    }
    needle.windows(3).all(|w| {
        let i = tri_bit(w);
        b[i >> 6] & (1u64 << (i & 63)) != 0
    })
}

fn name_contains(path: &Path, needle: &[u8]) -> bool {
    match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n
            .as_bytes()
            .windows(needle.len().max(1))
            .any(|w| w == needle),
        None => false,
    }
}

// ── 1. the full walk (ground truth) ──────────────────────────────────────────────────────────────
fn walk_full(root: &Path, needle: &[u8], st: &mut Stats) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            st.seen += 1;
            st.visited += 1;
            let p = e.path();
            if name_contains(&p, needle) {
                st.matches += 1;
            }
            if ft.is_dir() {
                stack.push(p);
            }
        }
    }
}

// ── 2. Lucas's fixed-cost quantum walk (content-blind) ────────────────────────────────────────────
fn walk_quantum(root: &Path, needle: &[u8], st: &mut Stats) {
    let mut stack = vec![root.to_path_buf()];
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            st.seen += 1;
            let (keep, s_n) = quantum_keep(s1, s2, ft.is_dir());
            s2 = s1;
            s1 = s_n;
            if !keep {
                continue;
            }
            st.visited += 1;
            let p = e.path();
            if name_contains(&p, needle) {
                st.matches += 1;
            }
            if ft.is_dir() {
                stack.push(p);
            }
        }
    }
}

// ── 3a. build the NB-summary index (the seed, certify-once): each dir's Bloom = ∨ of children ──────
fn build_index(dir: &Path, index: &mut HashMap<PathBuf, Bloom>) -> Bloom {
    let mut b: Bloom = [0u64; BW];
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if let Some(name) = e.file_name().to_str() {
                bloom_add_name(&mut b, name.as_bytes());
            }
            if ft.is_dir() {
                let child = build_index(&e.path(), index);
                bloom_or(&mut b, &child); // the NB merge, folded up the edges
            }
        }
    }
    index.insert(dir.to_path_buf(), b);
    b
}

// ── 3b. the NB-summary walk: prune a subtree iff its Bloom does NOT admit the needle (lossless) ────
fn walk_nb(root: &Path, needle: &[u8], index: &HashMap<PathBuf, Bloom>, st: &mut Stats) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            st.seen += 1;
            st.visited += 1;
            let p = e.path();
            if name_contains(&p, needle) {
                st.matches += 1;
            }
            if ft.is_dir() {
                match index.get(&p) {
                    Some(b) if !bloom_admits(b, needle) => {} // prune: provably no match below
                    _ => stack.push(p),
                }
            }
        }
    }
}

fn pct(a: u64, b: u64) -> f64 {
    if b == 0 {
        100.0
    } else {
        100.0 * a as f64 / b as f64
    }
}
fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .map(str::to_string)
        })
        .and_then(|l| l.split_whitespace().nth(1).and_then(|n| n.parse().ok()))
        .unwrap_or(0)
}
fn bloom_fill(b: &Bloom) -> u32 {
    b.iter().map(|w| w.count_ones()).sum()
}

fn bench_main(needle: &str, root: &Path) {
    let needle_b = needle.as_bytes();
    let mut st = Stats::default();
    let t = Instant::now();
    walk_full(root, needle_b, &mut st);
    let full = t.elapsed();
    let (full_seen, full_matches) = (st.seen, st.matches);

    let mut st = Stats::default();
    let t = Instant::now();
    walk_quantum(root, needle_b, &mut st);
    let q = t.elapsed();
    let (q_seen, q_matches) = (st.seen, st.matches);

    let mut index: HashMap<PathBuf, Bloom> = HashMap::new();
    let t = Instant::now();
    let _ = build_index(root, &mut index);
    let idx = t.elapsed();
    let mut st = Stats::default();
    let t = Instant::now();
    walk_nb(root, needle_b, &index, &mut st);
    let nb = t.elapsed();
    let (nb_seen, nb_matches) = (st.seen, st.matches);

    println!("bench '{needle}' at {}", root.display());
    println!(
        "  full   : {full_seen:>6} seen {full_matches:>4} matches  {:.1}ms",
        ms(full)
    );
    println!(
        "  quantum: {q_seen:>6} seen {q_matches:>4} matches  {:.1}ms  prune {:+.0}%",
        ms(q),
        pct(q_seen, full_seen) - 100.0
    );
    println!("  nb     : {nb_seen:>6} seen {nb_matches:>4} matches  {:.1}ms (idx {:.1}ms)  prune {:+.0}%  recall {}%", ms(nb), ms(idx), pct(nb_seen, full_seen) - 100.0, pct(nb_matches, full_matches));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<String> = Vec::new();
    let (mut count_only, mut scan_all) = (false, false);
    for a in &args {
        match a.as_str() {
            "-c" | "--count" => count_only = true,
            "-a" | "--text" => scan_all = true,
            "-h" | "--help" => {
                eprintln!("usage: swalk <name|content> <pattern> [path] [-c] [-a]");
                return;
            }
            _ => positional.push(a.clone()),
        }
    }
    if positional.len() < 2 {
        eprintln!("usage: swalk <name|content> <pattern> [path] [-c] [-a]");
        std::process::exit(2);
    }
    // bench subcommand (absorbed qwalk): swalk bench <needle> [path]
    if positional.first().map(|s| s == "bench").unwrap_or(false) && positional.len() >= 2 {
        let needle = positional[1].clone();
        let root = positional
            .get(2)
            .cloned()
            .unwrap_or_else(|| ".".to_string());
        bench_main(&needle, Path::new(&root));
        return;
    }
    let kind = positional[0].clone();
    let pattern = positional[1].clone();
    let root = positional
        .get(2)
        .cloned()
        .unwrap_or_else(|| ".".to_string());

    let re = match Regex::new(&pattern) {
        Ok(re) => re,
        Err(e) => {
            eprintln!("swalk: bad pattern: {e}");
            std::process::exit(2);
        }
    };

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(1 << 16, stdout.lock());
    let mut total: u64 = 0;

    match kind.as_str() {
        // kind 1 — the node applicative: the boundary reads the filename.
        "name" => walk(Path::new(&root), &mut |path, is_file| {
            let _ = is_file;
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if re.is_match(name.as_bytes()) {
                    total += 1;
                    if !count_only {
                        let _ = writeln!(out, "{}", path.display());
                    }
                }
            }
        }),
        // kind 2 — the content applicative. THE STREAMING LAW (`Substrate/Collapse.streaming_exact`):
        // the scan is a FOLD, not a re-scan. The rest state is a per-file memoized result keyed by
        // (path, mtime, size); it is certified once and persisted, so a re-run re-folds ONLY the
        // changed files (the Δ) and reads each unchanged file straight from the fold — its bytes are
        // never touched again. ripgrep re-scans O(n) every run because it holds no such state; this
        // is O(Δ = changed) + an O(files) stat-walk. The Δ itself is still scanned in parallel.
        "content" => {
            let fpath = fold_path(&root, &pattern, count_only);
            let mut fold = load_fold(&fpath);
            let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
            let mut todo: Vec<(PathBuf, u64, u64)> = Vec::new();
            walk(Path::new(&root), &mut |path, is_file| {
                if !is_file {
                    return;
                }
                let (mtime, size) = stat_key(path);
                let p = path.to_path_buf();
                let unchanged = fold
                    .get(&p)
                    .is_some_and(|e| e.mtime == mtime && e.size == size);
                if !unchanged {
                    todo.push((p.clone(), mtime, size)); // the Δ: new or changed
                }
                seen.insert(p);
            });
            // a file that vanished leaves the fold (streaming_exact_removal — the remove is exact too).
            fold.retain(|p, _| seen.contains(p));
            // re-fold ONLY the Δ, in parallel across cores; each thread returns its fresh entries.
            let next = std::sync::atomic::AtomicUsize::new(0);
            let nthreads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let fresh: Vec<(PathBuf, FoldEntry)> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..nthreads)
                    .map(|_| {
                        s.spawn(|| {
                            let mut entries: Vec<(PathBuf, FoldEntry)> = Vec::new();
                            loop {
                                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let Some((path, mtime, size)) = todo.get(i) else {
                                    break;
                                };
                                let (mtime, size) = (*mtime, *size);
                                let Ok(data) = std::fs::read(path) else {
                                    continue;
                                };
                                let (count, blob) = if scan_all || !is_binary(&data) {
                                    scan_file(&re, &data, path, count_only)
                                } else {
                                    (0, Vec::new())
                                };
                                entries.push((
                                    path.clone(),
                                    FoldEntry {
                                        mtime,
                                        size,
                                        count,
                                        blob,
                                    },
                                ));
                            }
                            entries
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|h| h.join().ok())
                    .flatten()
                    .collect()
            });
            for (p, e) in fresh {
                fold.insert(p, e);
            }
            // the answer is the monoid fold over the whole rest state (cached + just-folded).
            for e in fold.values() {
                total += e.count;
                if !count_only {
                    let _ = out.write_all(&e.blob);
                }
            }
            save_fold(&fpath, &fold);
        }
        other => {
            eprintln!("swalk: unknown kind '{other}' (use name|content)");
            std::process::exit(2);
        }
    }

    if count_only {
        let _ = writeln!(out, "{total}");
    }
    let _ = out.flush();
}

/// Scan one file's bytes: the count of matching LINES + (output mode) the rendered `path:lineno:line`
/// block. ONE whole-buffer `find_iter` — the membership/class GET pipelines over the buffer — plus a
/// single forward cursor that maps matches → lines lazily. NOT a per-line `is_match` (the per-line
/// setup + the separate line-split were the ~14× overhead under tregex's raw scan speed). A match
/// holds no `\n` (line-oriented patterns), so the cursor's newline count is the exact line.
fn scan_file(re: &Regex, data: &[u8], path: &Path, count_only: bool) -> (u64, Vec<u8>) {
    let mut count = 0u64;
    let mut blob: Vec<u8> = Vec::new();
    let (mut cursor, mut lineno, mut line_start, mut last) = (0usize, 1u64, 0usize, 0u64);
    for m in re.find_iter(data) {
        while cursor < m.start {
            if data[cursor] == b'\n' {
                lineno += 1;
                line_start = cursor + 1;
            }
            cursor += 1;
        }
        if lineno != last {
            count += 1;
            last = lineno;
            if !count_only {
                let mut e = m.start;
                while e < data.len() && data[e] != b'\n' {
                    e += 1;
                }
                blob.extend_from_slice(path.to_string_lossy().as_bytes());
                let _ = write!(blob, ":{lineno}:");
                blob.extend_from_slice(&data[line_start..e]);
                blob.push(b'\n');
            }
        }
    }
    (count, blob)
}

// ── the streaming fold (Substrate/Collapse.streaming_exact): the per-file rest state ──────────────

/// One memoized per-file scan result — a contribution to the fold. `(mtime, size)` is the validity
/// key (cheap to check from a `stat`, no content read); `(count, blob)` is the monoid contribution
/// (matching-line count + the rendered output lines). The whole-corpus answer is the sum over these.
struct FoldEntry {
    mtime: u64,
    size: u64,
    count: u64,
    blob: Vec<u8>,
}

/// (mtime_ns, size) for a path — the change key. Missing metadata ⇒ (0,0), i.e. treated as changed.
fn stat_key(path: &Path) -> (u64, u64) {
    let Ok(m) = std::fs::metadata(path) else {
        return (0, 0);
    };
    let mtime = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos() as u64);
    (mtime, m.len())
}

/// The fold's persisted location — deterministic in (root, pattern, mode) under the cache dir, so a
/// re-run of the same query finds its rest state. A different pattern or root folds independently.
fn fold_path(root: &str, pattern: &str, count_only: bool) -> PathBuf {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a over root | pattern | mode
    for b in root
        .bytes()
        .chain([0u8])
        .chain(pattern.bytes())
        .chain([u8::from(count_only)])
    {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let dir = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("swalk"));
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("fold-{h:016x}.bin"))
}

/// Read the fold (the certified rest). A missing/short/corrupt file is an empty rest — the next scan
/// is the cold one that builds it. Record: `[u32 path_len | path | u64 mtime | u64 size | u64 count
/// | u32 blob_len | blob]`, repeated.
fn load_fold(path: &Path) -> std::collections::HashMap<PathBuf, FoldEntry> {
    let mut map = std::collections::HashMap::new();
    let Ok(d) = std::fs::read(path) else {
        return map;
    };
    let r32 = |d: &[u8], i: usize| {
        d.get(i..i + 4)
            .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
    };
    let r64 = |d: &[u8], i: usize| {
        d.get(i..i + 8)
            .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
    };
    let mut i = 0usize;
    loop {
        let Some(pl) = r32(&d, i).map(|v| v as usize) else {
            break;
        };
        i += 4;
        let Some(pb) = d.get(i..i + pl) else { break };
        i += pl;
        let (Some(mtime), Some(size), Some(count)) = (r64(&d, i), r64(&d, i + 8), r64(&d, i + 16))
        else {
            break;
        };
        i += 24;
        let Some(bl) = r32(&d, i).map(|v| v as usize) else {
            break;
        };
        i += 4;
        let Some(bb) = d.get(i..i + bl) else { break };
        i += bl;
        let p = PathBuf::from(std::ffi::OsStr::from_bytes(pb));
        map.insert(
            p,
            FoldEntry {
                mtime,
                size,
                count,
                blob: bb.to_vec(),
            },
        );
    }
    map
}

/// Persist the fold. Cost is O(fold size) — per-file metadata + the (sparse) matching-line blobs —
/// never O(corpus). A refinement is a per-Δ append log (cgrep's segment log); whole-write is simplest.
fn save_fold(path: &Path, fold: &std::collections::HashMap<PathBuf, FoldEntry>) {
    let mut out: Vec<u8> = Vec::new();
    for (p, e) in fold {
        let pb = p.as_os_str().as_bytes();
        out.extend_from_slice(&(pb.len() as u32).to_le_bytes());
        out.extend_from_slice(pb);
        out.extend_from_slice(&e.mtime.to_le_bytes());
        out.extend_from_slice(&e.size.to_le_bytes());
        out.extend_from_slice(&e.count.to_le_bytes());
        out.extend_from_slice(&(e.blob.len() as u32).to_le_bytes());
        out.extend_from_slice(&e.blob);
    }
    let _ = std::fs::write(path, out);
}

/// One iterative tree-walk (the universalLift over the directory). Calls `visit(path, is_file)` for
/// every entry. No recursion limit, no symlink-follow (matches find/rg default), errors skipped.
fn walk(root: &Path, visit: &mut dyn FnMut(&Path, bool)) {
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                visit(&path, false);
                stack.push(path);
            } else if ft.is_file() {
                visit(&path, true);
            }
        }
    }
}

/// Binary sniff (matches grep/rg default: a NUL byte in the first 8 KB ⇒ skip).
fn is_binary(data: &[u8]) -> bool {
    data.iter().take(8192).any(|&b| b == 0)
}
