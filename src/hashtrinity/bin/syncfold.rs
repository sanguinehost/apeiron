//! syncfold — the fastest file sync.
//!
//! File transfer is one operation: `clone (tree → the difference set) → move the
//! flaws → done`. cp is the bare fold, luminS the parallel fold, rsync the delta
//! fold — all the *same* operation (`sang adapt web` clusters them to one kind).
//! syncfold is that operation at its three faces: the parallel set-difference
//! walk (the Web, the B/U medium), the fast in-kernel copy on the flaws only
//! (the metal, the B face), and the (size,mtime) throne-check that skips the
//! unchanged bulk without reading it (the streaming law, the NB face).
//!
//! Usage:
//!   syncfold [OPTIONS] SRC DST
//!     -j, --threads N   workers (default: all cores)
//!     --delete          mirror — remove dst entries absent from src
//!     -n, --dry-run     report what would happen, change nothing
//!     -q, --quiet       no per-file error lines, no summary

// Sovereignty invariant, enforced by the compiler: not one line of `unsafe` in
// the whole crate. The platform cut (pread/pwrite, devices, symlinks) and the
// crypto stack are all safe `std`.
#![forbid(unsafe_code)]
// Clippy runs with `-D warnings` in CI. These few categories are intentional and
// stay allowed: crypto/format loops index several arrays in lockstep to mirror
// their reference specs (`needless_range_loop` would make them worse); a couple of
// low-level ops legitimately take many coordinate arguments; the TLS handshake
// carries a genuinely complex state type; and collapsing guard `if`s often reads
// worse than the nested form. Everything else is held to the strict bar.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::collapsible_if)]

// Same module tree as `lib.rs` (this crate is usable as a library too, e.g. by
// Galvanize's tfplugin6 client for HPACK/HTTP2/gRPC/cert-gen reuse) — these
// `use` aliases keep every module reference below resolving exactly as it did
// when they were local `mod` declarations. Only the ones this binary's own CLI
// dispatch references directly; the rest (crypto/grpc/hpack/http2/tls/etc.)
// live in the library for other crates, reached only transitively from here.
use hashtrinity::sync::{archive, block, http, image, s3, scan, sync};

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let mut delete = false;
    let mut dry_run = false;
    let mut quiet = false;
    let mut positional: Vec<String> = Vec::new();

    // Subcommands (S3 path uses our own TLS). `get` is a curl-lite for testing it.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("get") {
        return cmd_get(&argv[1..]);
    }
    if argv.first().map(String::as_str) == Some("s3") {
        return cmd_s3(&argv[1..]);
    }
    if argv.first().map(String::as_str) == Some("scan") {
        return cmd_scan(&argv[1..]);
    }
    if argv.first().map(String::as_str) == Some("dd") {
        return cmd_dd(&argv[1..]);
    }

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-j" | "--threads" => match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if n > 0 => threads = n,
                _ => {
                    eprintln!("syncfold: --threads needs a positive integer");
                    return ExitCode::from(2);
                }
            },
            "--delete" => delete = true,
            "-n" | "--dry-run" => dry_run = true,
            "-q" | "--quiet" => quiet = true,
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            s if s.starts_with('-') && s != "-" => {
                eprintln!("syncfold: unknown option {s}");
                return ExitCode::from(2);
            }
            s => positional.push(s.to_string()),
        }
    }

    if positional.len() != 2 {
        eprintln!(
            "syncfold: need exactly SRC and DST (got {})",
            positional.len()
        );
        print_help();
        return ExitCode::from(2);
    }
    let src = PathBuf::from(&positional[0]);
    let dst = PathBuf::from(&positional[1]);

    let opts = sync::Opts {
        threads,
        delete,
        dry_run,
        quiet,
    };
    match sync::run(&src, &dst, &opts) {
        Ok(st) => {
            if !quiet {
                let tag = if dry_run { "would " } else { "" };
                eprintln!(
                    "syncfold: {tag}copy {} files ({:.2} MiB), {}skip {}, {tag}mkdir {}, {tag}symlink {}, {tag}delete {}, errors {}",
                    st.files_copied,
                    st.bytes_copied as f64 / (1024.0 * 1024.0),
                    tag,
                    st.files_skipped,
                    st.dirs_made,
                    st.symlinks,
                    st.deleted,
                    st.errors,
                );
            }
            if st.errors > 0 {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("syncfold: {e}");
            ExitCode::from(1)
        }
    }
}

/// `syncfold get https://host[:port]/path` — fetch over our own pure-Rust TLS.
fn cmd_get(args: &[String]) -> ExitCode {
    let Some(url) = args.first() else {
        eprintln!("usage: syncfold get https://host[:port]/path");
        return ExitCode::from(2);
    };
    let rest = match url.strip_prefix("https://") {
        Some(r) => r,
        None => {
            eprintln!("syncfold get: only https:// URLs");
            return ExitCode::from(2);
        }
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(443)),
        None => (authority.to_string(), 443u16),
    };
    match http::request(&host, port, "GET", path, &[], &[], true) {
        Ok(resp) => {
            eprintln!("HTTP {} — {} body bytes", resp.status, resp.body.len());
            for (k, v) in &resp.headers {
                eprintln!("  {k}: {v}");
            }
            if !resp.body.is_empty() {
                print!("{}", String::from_utf8_lossy(&resp.body));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("syncfold get: {e}");
            ExitCode::from(1)
        }
    }
}

/// `syncfold s3 backup <dir> <bucket> <key> [--endpoint URL] [--region R]`
/// `syncfold s3 restore <bucket> <key> <dir> [--endpoint URL] [--region R]`
/// Creds from env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY. The whole transport
/// is our own pure-Rust TLS + SigV4 — no aws-sdk, no openssl.
fn cmd_s3(args: &[String]) -> ExitCode {
    let mut pos: Vec<String> = Vec::new();
    let mut endpoint: Option<String> = None;
    let mut region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into());
    let mut do_scan = false;
    let mut insecure = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--endpoint" => endpoint = it.next().cloned(),
            "--region" => {
                if let Some(r) = it.next() {
                    region = r.clone();
                }
            }
            "--scan" => do_scan = true,
            "--insecure" => insecure = true,
            s => pos.push(s.to_string()),
        }
    }
    let ak = std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default();
    let sk = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default();

    // endpoint → (host, port, path_style). With no endpoint: AWS vhost style.
    let (host, port, path_style, plain) = match &endpoint {
        Some(url) => {
            let (plain, rest) = if let Some(r) = url.strip_prefix("http://") {
                (true, r)
            } else {
                (false, url.strip_prefix("https://").unwrap_or(url))
            };
            let (h, p) = match rest.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), p.parse().unwrap_or(if plain { 80 } else { 443 })),
                None => (rest.to_string(), if plain { 80 } else { 443 }),
            };
            (h, p, true, plain)
        }
        None => (String::new(), 443u16, false, false),
    };

    let mk = |bucket: &str| s3::S3 {
        host: if path_style {
            host.clone()
        } else {
            format!("{bucket}.s3.{region}.amazonaws.com")
        },
        port,
        region: region.clone(),
        access_key: ak.clone(),
        secret: sk.clone(),
        path_style,
        verify: !insecure,
        plain,
    };

    match pos.first().map(String::as_str) {
        Some("backup") if pos.len() == 4 => {
            let (dir, bucket, key) = (&pos[1], &pos[2], &pos[3]);
            let n = archive::count_files(std::path::Path::new(dir));
            let part_mib = std::env::var("SYNCFOLD_PART_MIB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(8)
                .max(5);
            let part_size = part_mib * 1024 * 1024;
            eprintln!(
                "syncfold s3: streaming {n} files → s3://{bucket}/{key} (part {part_mib} MiB, multipart if >part)"
            );
            match mk(bucket).backup_stream(bucket, key, std::path::Path::new(dir), part_size) {
                Ok(s) => {
                    eprintln!("syncfold s3: backup complete HTTP {s} ✓");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("syncfold s3: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Some("sync") if pos.len() == 4 => {
            // raw per-object: each file → its own S3 object (NOT a bundled seed)
            let (dir, bucket, prefix) = (&pos[1], &pos[2], &pos[3]);
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
            let mut files = Vec::new();
            list_files(
                std::path::Path::new(dir),
                std::path::Path::new(dir),
                prefix,
                &mut files,
            );
            // --scan: friction scan first (secure sync) — only frictionless files are uploaded
            if do_scan {
                let sigs = scan::default_sigs();
                let r = scan::scan_named(&files, &sigs, threads, |t| std::fs::read(t).ok());
                if !r.is_clean() {
                    eprintln!(
                        "syncfold s3: scan found {} THREAT(S) — quarantined, NOT uploaded:",
                        r.threats.len()
                    );
                    for (name, hits) in &r.threats {
                        eprintln!("  ⛔ {name} — {}", hits.join(", "));
                    }
                    let mut bad = hashtrinity::Table::new();
                    for (n, _) in &r.threats {
                        bad.mark(n.clone());
                    }
                    files.retain(|(_, key)| !bad.has(key));
                } else {
                    eprintln!(
                        "syncfold s3: scan clean ✓ ({} files frictionless)",
                        r.scanned
                    );
                }
            }
            eprintln!(
                "syncfold s3: {} files → {} objects, {threads} connections",
                files.len(),
                files.len()
            );
            let (up, errs) = mk(bucket).sync_raw(bucket, &files, threads);
            eprintln!("syncfold s3: uploaded {up} objects, errors {errs}");
            if errs > 0 {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Some("scan") if pos.len() == 3 => {
            // the application acting on the round-trip: scan objects already in the bucket
            let (bucket, prefix) = (&pos[1], &pos[2]);
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
            let sigs = scan::default_sigs();
            match mk(bucket).scan_bucket(bucket, prefix, &sigs, threads) {
                Ok(r) => report_scan(&r),
                Err(e) => {
                    eprintln!("syncfold s3 scan: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Some("restore") if pos.len() == 4 => {
            let (bucket, key, dir) = (&pos[1], &pos[2], &pos[3]);
            let chunk = std::env::var("SYNCFOLD_PART_MIB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(8)
                * 1024
                * 1024;
            match mk(bucket).restore_stream(bucket, key, std::path::Path::new(dir), chunk) {
                Ok(total) => {
                    eprintln!("syncfold s3: streamed {total} bytes → {dir} ✓ (bounded RAM)");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("syncfold s3: {e}");
                    ExitCode::from(1)
                }
            }
        }
        _ => {
            eprintln!(
                "usage:\n  syncfold s3 backup <dir> <bucket> <key> [--endpoint URL] [--region R]\n  syncfold s3 restore <bucket> <key> <dir> [--endpoint URL] [--region R]"
            );
            ExitCode::from(2)
        }
    }
}

/// `syncfold scan <dir>` — the friction scan over a tree (the Web encircle).
fn cmd_scan(args: &[String]) -> ExitCode {
    let Some(dir) = args.first() else {
        eprintln!("usage: syncfold scan <dir>");
        return ExitCode::from(2);
    };
    let sigs = scan::default_sigs();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let r = scan::scan_tree(std::path::Path::new(dir), &sigs, threads);
    report_scan(&r)
}

/// `syncfold dd <src> <dst> [--sync] [--bs N] [-j N] [-n] [-q]`
///
/// Block duplication — `dd` at α = raw block stream. `src`/`dst` may be regular
/// files or block devices (on Linux, the same kind of object — a device is just
/// `/dev/sdX`). `--sync` writes only the blocks that differ from the destination
/// (the streaming-law per-block throne-check), the win plain `dd` lacks.
fn cmd_dd(args: &[String]) -> ExitCode {
    // Network sub-mode: `syncfold dd serve <dst> [--port N]` receives a stream.
    if args.first().map(String::as_str) == Some("serve") {
        return cmd_dd_serve(&args[1..]);
    }
    // Standalone verify: `syncfold dd verify <a> <b>` — compare two images
    // (each read through its inferred format) byte for byte.
    if args.first().map(String::as_str) == Some("verify") {
        return cmd_dd_verify(&args[1..]);
    }

    let mut pos: Vec<String> = Vec::new();
    let mut if_src: Option<String> = None; // dd-style if=
    let mut of_dst: Option<String> = None; // dd-style of=
    let mut chunk: usize = 1 << 20; // 1 MiB, like block-duplicator's default
                                    // Sequential by default: a single device is write-bandwidth-bound, so fanning
                                    // writes across cores only adds contention (measured: -j1 ties or beats -j16
                                    // on both incompressible and sparse copies). The win is the skip, not the
                                    // cores. `-j N` stays an opt-in for genuine multi-device / fast-NVMe setups.
    let mut threads = 1usize;
    let mut do_sync = false;
    let mut sparse = true; // skip all-zero blocks → sparse target (output-identical)
    let mut verify = false;
    let mut dry_run = false;
    let mut quiet = false;
    let mut src_fmt: Option<image::Format> = None;
    let mut dst_fmt: Option<image::Format> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sync" => do_sync = true,
            "--no-sparse" => sparse = false,
            "--verify" => verify = true,
            "-n" | "--dry-run" => dry_run = true,
            "-q" | "--quiet" => quiet = true,
            "--src-format" => match it.next().and_then(|v| image::Format::parse(v)) {
                Some(f) => src_fmt = Some(f),
                None => {
                    eprintln!("syncfold dd: --src-format must be raw|qcow2");
                    return ExitCode::from(2);
                }
            },
            "--dst-format" => match it.next().and_then(|v| image::Format::parse(v)) {
                Some(f) => dst_fmt = Some(f),
                None => {
                    eprintln!("syncfold dd: --dst-format must be raw|qcow2");
                    return ExitCode::from(2);
                }
            },
            "--bs" | "--chunk-size" => match it.next().and_then(|v| parse_size(v)) {
                Some(n) if n >= 512 => chunk = n,
                _ => {
                    eprintln!("syncfold dd: --bs needs a size >= 512 (e.g. 4M, 1048576)");
                    return ExitCode::from(2);
                }
            },
            "-j" | "--threads" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if n > 0 => threads = n,
                _ => {
                    eprintln!("syncfold dd: --threads needs a positive integer");
                    return ExitCode::from(2);
                }
            },
            // dd-style if=/of=/bs= for muscle memory
            s if s.starts_with("if=") => if_src = Some(s[3..].to_string()),
            s if s.starts_with("of=") => of_dst = Some(s[3..].to_string()),
            s if s.starts_with("bs=") => match parse_size(&s[3..]) {
                Some(n) if n >= 512 => chunk = n,
                _ => {
                    eprintln!("syncfold dd: bad bs=");
                    return ExitCode::from(2);
                }
            },
            s if s.starts_with('-') => {
                eprintln!("syncfold dd: unknown option {s}");
                return ExitCode::from(2);
            }
            s => pos.push(s.to_string()),
        }
    }

    let (src_s, dst_s) = match (if_src, of_dst, pos.len()) {
        (Some(s), Some(d), 0) => (s, d),
        (None, None, 2) => (pos[0].clone(), pos[1].clone()),
        _ => {
            eprintln!("usage: syncfold dd <src> <dst> [--sync] [--bs SIZE] [-j N] [-n] [-q]");
            eprintln!(
                "       syncfold dd <src> tcp://host:port      (push over network, holes skipped)"
            );
            eprintln!("       syncfold dd serve <dst> [--port N]     (receive a pushed stream)");
            eprintln!("       syncfold dd if=<src> of=<dst> [bs=SIZE] [--sync]");
            return ExitCode::from(2);
        }
    };

    // Resolve formats: explicit flag wins, else infer from the path extension.
    let src_format = src_fmt.unwrap_or_else(|| image::Format::infer(std::path::Path::new(&src_s)));
    let dst_format = dst_fmt.unwrap_or_else(|| image::Format::infer(std::path::Path::new(&dst_s)));

    // Network destination: push the source over TCP, skipping holes on the wire.
    if let Some(hostport) = dst_s.strip_prefix("tcp://") {
        let (host, port) = match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse().unwrap_or(0)),
            None => (hostport.to_string(), 0),
        };
        if port == 0 {
            eprintln!("syncfold dd: tcp:// destination needs host:port");
            return ExitCode::from(2);
        }
        return match block::push(
            &PathBuf::from(&src_s),
            src_format,
            &host,
            port,
            sparse,
            quiet,
        ) {
            Ok(st) => {
                if !quiet {
                    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
                    eprintln!(
                        "syncfold dd: pushed {:.2} MiB data, skipped {:.2} MiB holes → tcp://{host}:{port}",
                        mib(st.bytes_copied),
                        mib(st.bytes_zero)
                    );
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("syncfold dd push: {e}");
                ExitCode::from(1)
            }
        };
    }

    let (src, dst) = (PathBuf::from(&src_s), PathBuf::from(&dst_s));
    let opts = block::Opts {
        chunk,
        threads,
        sync: do_sync,
        sparse,
        src_format,
        dst_format,
        dry_run,
        quiet,
    };
    match block::run(&src, &dst, &opts) {
        Ok(st) => {
            if !quiet {
                let verb = if dry_run { "would write" } else { "wrote" };
                let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
                eprintln!(
                    "syncfold dd: {verb} {:.2} MiB ({} blocks), holes {:.2} MiB ({}), unchanged {:.2} MiB ({}), errors {}",
                    mib(st.bytes_copied),
                    st.blocks_copied,
                    mib(st.bytes_zero),
                    st.blocks_zero,
                    mib(st.bytes_skipped),
                    st.blocks_skipped,
                    st.errors,
                );
            }
            if st.errors > 0 {
                return ExitCode::from(1);
            }
            if verify && !dry_run {
                match block::verify(&src, src_format, &dst, dst_format, quiet) {
                    Ok(None) => {
                        if !quiet {
                            eprintln!("syncfold dd: verify OK — destination reproduces source ✓");
                        }
                    }
                    Ok(Some(off)) => {
                        eprintln!("syncfold dd: VERIFY FAILED — first mismatch at byte {off}");
                        return ExitCode::from(1);
                    }
                    Err(e) => {
                        eprintln!("syncfold dd: verify error: {e}");
                        return ExitCode::from(1);
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("syncfold dd: {e}");
            ExitCode::from(1)
        }
    }
}

/// `syncfold dd verify <a> <b>` — compare two images (formats inferred), byte exact.
fn cmd_dd_verify(args: &[String]) -> ExitCode {
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let quiet = args.iter().any(|a| a == "-q" || a == "--quiet");
    if pos.len() != 2 {
        eprintln!("usage: syncfold dd verify <a> <b>");
        return ExitCode::from(2);
    }
    let (a, b) = (PathBuf::from(pos[0]), PathBuf::from(pos[1]));
    let fa = image::Format::infer(&a);
    let fb = image::Format::infer(&b);
    match block::verify(&a, fa, &b, fb, quiet) {
        Ok(None) => {
            eprintln!("syncfold dd verify: IDENTICAL ✓");
            ExitCode::SUCCESS
        }
        Ok(Some(off)) => {
            eprintln!("syncfold dd verify: DIFFER at byte {off} ✗");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("syncfold dd verify: {e}");
            ExitCode::from(1)
        }
    }
}

/// `syncfold dd serve <dst> [--port N]` — receive one pushed stream into <dst>.
fn cmd_dd_serve(args: &[String]) -> ExitCode {
    let mut dst: Option<String> = None;
    let mut port: u16 = 9000;
    let mut quiet = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => match it.next().and_then(|v| v.parse::<u16>().ok()) {
                Some(p) => port = p,
                None => {
                    eprintln!("syncfold dd serve: --port needs a number");
                    return ExitCode::from(2);
                }
            },
            "-q" | "--quiet" => quiet = true,
            s if s.starts_with('-') => {
                eprintln!("syncfold dd serve: unknown option {s}");
                return ExitCode::from(2);
            }
            s => dst = Some(s.to_string()),
        }
    }
    let Some(dst) = dst else {
        eprintln!("usage: syncfold dd serve <dst> [--port N]");
        return ExitCode::from(2);
    };
    match block::serve(&PathBuf::from(&dst), port, quiet) {
        Ok(st) => {
            if !quiet {
                let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
                eprintln!(
                    "syncfold dd serve: received {:.2} MiB data ({} frames), {:.2} MiB holes left sparse → {dst} ✓",
                    mib(st.bytes_copied),
                    st.blocks_copied,
                    mib(st.bytes_zero)
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("syncfold dd serve: {e}");
            ExitCode::from(1)
        }
    }
}

/// Parse `4096`, `4K`, `1M`, `2G` (binary multipliers) into bytes.
fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('K') | Some('k') => (&s[..s.len() - 1], 1usize << 10),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1usize << 20),
        Some('G') | Some('g') => (&s[..s.len() - 1], 1usize << 30),
        _ => (s, 1usize),
    };
    num.parse::<usize>().ok().map(|n| n * mult)
}

fn report_scan(r: &scan::ScanReport) -> ExitCode {
    if r.is_clean() {
        eprintln!(
            "syncfold scan: {} scanned, ALL CLEAN ✓ (residual 0 — frictionless)",
            r.scanned
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "syncfold scan: {} scanned, {} clean, {} THREAT(S):",
            r.scanned,
            r.clean,
            r.threats.len()
        );
        for (name, hits) in &r.threats {
            eprintln!("  ⛔ {name} — {}", hits.join(", "));
        }
        ExitCode::from(1)
    }
}

/// Collect (local path, object key) for every file under `root`, key = prefix + relpath.
fn list_files(
    dir: &std::path::Path,
    root: &std::path::Path,
    prefix: &str,
    out: &mut Vec<(std::path::PathBuf, String)>,
) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for de in rd.flatten() {
        let Ok(ft) = de.file_type() else { continue };
        if ft.is_dir() && !ft.is_symlink() {
            list_files(&de.path(), root, prefix, out);
        } else if ft.is_file() {
            let rel = de
                .path()
                .strip_prefix(root)
                .unwrap_or(&de.path())
                .to_string_lossy()
                .replace('\\', "/");
            let key = if prefix.is_empty() {
                rel
            } else {
                format!("{}/{}", prefix.trim_end_matches('/'), rel)
            };
            out.push((de.path(), key));
        }
    }
}

fn print_help() {
    eprintln!(
        "syncfold — the fastest file sync (parallel set-difference fold, zero deps)\n\
         \n\
         USAGE\n\
         \x20 syncfold [OPTIONS] SRC DST\n\
         \n\
         OPTIONS\n\
         \x20 -j, --threads N   workers (default: all cores)\n\
         \x20 --delete          mirror — remove dst entries absent from src\n\
         \x20 -n, --dry-run     report what would happen, change nothing\n\
         \x20 -q, --quiet       suppress the summary and per-file errors\n\
         \x20 -h, --help        this map\n\
         \n\
         SUBCOMMANDS\n\
         \x20 dd  <src> <dst> [--sync] [--bs SIZE]   block duplication (files or /dev/sdX);\n\
         \x20                                        skips all-zero blocks (sparse) by default;\n\
         \x20                                        --sync writes only differing blocks,\n\
         \x20                                        --no-sparse forces every block,\n\
         \x20                                        --verify reads the result back,\n\
         \x20                                        -j N for multi-device (default: sequential)\n\
         \x20                                        .qcow2/.vhdx in/out converts (--src/--dst-format raw|qcow2|vhdx)\n\
         \x20 dd  <src> tcp://host:port             push a disk/image over TCP (holes skipped)\n\
         \x20 dd  serve <dst> [--port N]            receive a pushed stream into <dst>\n\
         \x20 s3 backup|restore|sync|scan ...        sovereign S3 over our own TLS\n\
         \x20 scan <dir>                             friction scan (the Web encircle)"
    );
}
