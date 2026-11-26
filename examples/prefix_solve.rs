//! Two turns through the solved-prefix table: the first pays the prefill, the second resolves the
//! prefix by probe plus the missed tail. Prints both walls, the hit length, and decode.
//!
//! `cargo run --release --locked --example prefix_solve -- --model <gguf> --tokenizer <json>
//!  [--pack <nqp>] [--prompt TEXT] [--len 2048] [--differ 205] [--decode 32] [--mb 512] [--phases]`
//!
//! `--len` cycles the prompt's tokens to that many positions; `--differ` replaces that many tail
//! tokens for the third run, so it shares `len - differ` positions with the first. `--mb` must
//! cover `bytes_per_position × (len + decode + 8)`; the line it prints states the charge, and a
//! refusal leaves the cache empty (`0 entries`) rather than failing the turn. `--phases` turns on
//! [`apeiron::engine::trace`] and prints each turn's per-phase split: the ns inside each leaf, its
//! share of that turn's wall, and its span count. The split resets per turn, so the cold row shows
//! the prefill and the shared row shows the tail.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the printed rates and the vocabulary cast are diagnostics; every count here is below \
              2^53 and the vocabulary is a `u32` in the GGUF metadata"
)]

use apeiron::engine::loader::Engine;
use apeiron::engine::prefix::{prefill_with_cache_mode, FnvPrefixHasher, PrefixCache};
use apeiron::engine::{argmax_fixed, KernelMode};
use apeiron::format::bpe::Bpe;
use apeiron::format::nq;
use std::error::Error;
use std::time::Instant;

struct Args {
    model: String,
    pack: Option<String>,
    tokenizer: String,
    prompt: String,
    len: usize,
    differ: usize,
    decode: usize,
    mb: usize,
    phases: bool,
    mode: KernelMode,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        model: String::new(),
        pack: None,
        tokenizer: String::new(),
        prompt: String::from("The quick brown fox jumps over the lazy dog. "),
        len: 2048,
        differ: 205,
        decode: 32,
        mb: 512,
        phases: false,
        mode: KernelMode::Q8Act,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--pack" => a.pack = Some(val()?),
            "--tokenizer" => a.tokenizer = val()?,
            "--prompt" => a.prompt = val()?,
            "--len" => a.len = val()?.parse()?,
            "--differ" => a.differ = val()?.parse()?,
            "--decode" => a.decode = val()?.parse()?,
            "--mb" => a.mb = val()?.parse()?,
            "--phases" => a.phases = true,
            "--kernel" => {
                a.mode = match val()?.as_str() {
                    "exact" => KernelMode::Exact,
                    "q8act" => KernelMode::Q8Act,
                    other => return Err(format!("--kernel is q8act|exact, got {other:?}").into()),
                }
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if a.model.is_empty() || a.tokenizer.is_empty() {
        return Err("--model and --tokenizer are required".into());
    }
    if a.len == 0 || a.differ >= a.len {
        return Err("--len must exceed --differ".into());
    }
    Ok(a)
}

/// One turn: probe, prefill, decode `n` tokens. Returns the prefill wall and the first token.
fn turn(
    engine: &Engine,
    cache: &mut PrefixCache,
    ids: &[u32],
    decode: usize,
    mode: KernelMode,
) -> Result<(f64, f64, usize, u32), Box<dyn Error>> {
    // The probe is part of the prefill the caller pays; it also reports the hit this turn uses.
    let t0 = Instant::now();
    let hit = cache
        .longest_prefix(&ids[..ids.len() - 1])
        .map_or(0, |(m, _)| m);
    let (kv, first) = prefill_with_cache_mode(engine, cache, ids, decode + 8, true, mode)
        .map_err(|e| format!("{e:?}"))?;
    let prefill = t0.elapsed().as_secs_f64();

    let mut st = engine.state_with_kv(kv);
    let t1 = Instant::now();
    let mut cur = first.token;
    for _ in 0..decode {
        let logits = engine
            .step_logits_with_mode(&mut st.kv, cur, mode)
            .map_err(|e| format!("{e:?}"))?;
        cur = argmax_fixed(&logits).map_err(|e| format!("{e:?}"))?;
    }
    let decode_s = t1.elapsed().as_secs_f64();
    Ok((prefill, decode_s, hit, first.token))
}

fn main() -> Result<(), Box<dyn Error>> {
    let a = parse_args()?;
    let bpe = Bpe::from_tokenizer_json(&a.tokenizer)?;
    let engine = match &a.pack {
        Some(path) => {
            let pack = nq::open(path).map_err(|e| e.to_string())?;
            eprintln!(
                "pack          {} linears, {:.6} b/w",
                pack.linears.len(),
                pack.bits_per_weight().unwrap_or(0.0)
            );
            Engine::open_with_nq(&a.model, pack)?
        }
        None => Engine::open(&a.model)?,
    };
    let seed = bpe.encode(&a.prompt)?;
    if seed.is_empty() {
        return Err("the prompt is empty".into());
    }
    let c = &engine.config;
    let values = 2 * c.n_head_kv * c.head_dim * c.n_layer;
    let probe = engine.decode_state(1).map_err(|e| format!("{e:?}"))?;
    let per_position = PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(&probe);
    eprintln!(
        "kv            {} layers, {} kv heads x {}, {} values/position, {} B/position charged",
        c.n_layer, c.n_head_kv, c.head_dim, values, per_position
    );
    let ids: Vec<u32> = (0..a.len).map(|i| seed[i % seed.len()]).collect();
    let vocab = engine.config.vocab_size as u32;
    let mut other = ids.clone();
    for id in &mut other[a.len - a.differ..] {
        *id = (*id + 1) % vocab;
    }

    // Page the mapped weights and the pack in before the cold turn, so the cold row is the
    // steady-state prefill and not the first touch of a 500 MB skeleton.
    let mut warm_state = engine.state(64).map_err(|e| format!("{e:?}"))?;
    engine
        .prefill_state(&mut warm_state, &ids[..ids.len().min(32)], a.mode)
        .map_err(|e| format!("{e:?}"))?;
    drop(warm_state);

    let mut cache = PrefixCache::new(a.mb.saturating_mul(1024 * 1024));
    println!(
        "len {} differ {} decode {} cache {} MiB, {} B needed",
        a.len,
        a.differ,
        a.decode,
        a.mb,
        per_position.saturating_mul(a.len + a.decode + 8)
    );
    for (name, prompt) in [("cold", &ids), ("warm", &ids), ("shared", &other)] {
        if a.phases {
            apeiron::engine::trace::set_enabled(true);
            apeiron::engine::trace::reset();
        }
        let (prefill, decode_s, hit, first) = turn(&engine, &mut cache, prompt, a.decode, a.mode)?;
        if a.phases {
            // The trace covers the probe, the prefill and the decode; the denominator is the
            // measured wall of both so the shares partition the turn rather than overflow it.
            let wall = (prefill + decode_s).max(f64::MIN_POSITIVE) * 1e9;
            for s in apeiron::engine::trace::snapshot() {
                if s.ns > 0 {
                    eprintln!(
                        "  {:12} {:10.3} ms {:6.1}%  calls {}",
                        s.name,
                        s.ns as f64 / 1e6,
                        100.0 * s.ns as f64 / wall,
                        s.calls
                    );
                }
            }
            let (alloc_ns, alloc_calls, insert_ns, insert_calls) =
                apeiron::engine::trace::outside_split();
            eprintln!(
                "  {:14} {:8.3} ms {:6.1}%  calls {}",
                "outside.alloc",
                alloc_ns as f64 / 1e6,
                100.0 * alloc_ns as f64 / wall,
                alloc_calls
            );
            eprintln!(
                "  {:14} {:8.3} ms {:6.1}%  calls {}",
                "outside.insert",
                insert_ns as f64 / 1e6,
                100.0 * insert_ns as f64 / wall,
                insert_calls
            );
            print_wait_histogram();
        }
        let rate = a.len as f64 / prefill.max(f64::MIN_POSITIVE);
        println!(
            "{name:6} hit {hit:5}  prefill {prefill:8.3} s ({rate:7.2} tok/s)  \
             decode {} in {decode_s:6.3} s ({:.2} tok/s)  first {first}",
            a.decode,
            a.decode as f64 / decode_s.max(f64::MIN_POSITIVE)
        );
    }
    println!(
        "cache {} entries, {} bytes stored",
        cache.entry_count(),
        cache.current_bytes()
    );
    Ok(())
}

/// The per-class `pool.wait` distribution: one line per class with at least one dispatch, each
/// percentile printed as its bucket's upper edge.
fn print_wait_histogram() {
    let h = apeiron::engine::trace::pool_wait_histogram();
    for (class, label) in apeiron::engine::trace::WAIT_CLASS_LABELS.iter().enumerate() {
        let total: u64 = h.buckets[class].iter().sum();
        if total == 0 {
            continue;
        }
        let mut cum = 0u64;
        let mut p50 = 0u64;
        let mut p95 = 0u64;
        for (i, &c) in h.buckets[class].iter().enumerate() {
            cum += c;
            let upper = 1u64 << (i + 1);
            if p50 == 0 && cum * 2 >= total {
                p50 = upper;
            }
            if p95 == 0 && cum * 20 >= total * 19 {
                p95 = upper;
            }
        }
        eprintln!(
            "  pool.wait n={label:5} {total:5} calls  p50 < {p50} ns  p95 < {p95} ns  max {} ns",
            h.max_ns[class]
        );
    }
}
