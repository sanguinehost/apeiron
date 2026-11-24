//! Two turns through the solved-prefix table: the first pays the prefill, the second resolves the
//! prefix by probe plus the missed tail. Prints both walls, the hit length, and decode.
//!
//! `cargo run --release --locked --example prefix_solve -- --model <gguf> --tokenizer <json>
//!  [--pack <nqp>] [--prompt TEXT] [--len 2048] [--differ 205] [--decode 32] [--mb 512]`
//!
//! `--len` cycles the prompt's tokens to that many positions; `--differ` replaces that many tail
//! tokens for the third run, so it shares `len - differ` positions with the first. `--mb` must
//! cover `bytes_per_position × (len + decode + 8)`; the line it prints states the charge, and a
//! refusal leaves the cache empty (`0 entries`) rather than failing the turn.

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
        let (prefill, decode_s, hit, first) = turn(&engine, &mut cache, prompt, a.decode, a.mode)?;
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
