//! Prompt-lookup drafter acceptance on a real greedy stream.
//!
//! `lookup_accept --model <gguf> [--pack <nqp>] [--first 42] [--n 128] [--ngram 3] [--draft 8]`
//!
//! Generates `n` greedy ids with [`Engine::next_token_state`], then replays an exact prompt-lookup
//! drafter over the causal stream: at each position it finds the latest earlier occurrence of the
//! trailing `ngram` ids and proposes the ids that followed it, up to `draft` tokens. The accepted
//! prefix is the longest match against the already-generated stream, so the counts are exactly what
//! a verified speculative decoder would emit. No model work is repeated: this is an acceptance
//! measurement, not a decoder.
use std::process::ExitCode;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::trace;
use apeiron::engine::KernelMode;
use apeiron::format::nq;

struct Args {
    model: String,
    pack: Option<String>,
    first: u32,
    n: usize,
    ngram: usize,
    draft: usize,
    trace: bool,
    lookup_only: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        pack: None,
        first: 42,
        n: 128,
        ngram: 3,
        draft: 8,
        trace: false,
        lookup_only: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--pack" => a.pack = Some(val()?),
            "--first" => a.first = val()?.parse().map_err(|e| format!("--first: {e}"))?,
            "--n" => a.n = val()?.parse().map_err(|e| format!("--n: {e}"))?,
            "--ngram" => a.ngram = val()?.parse().map_err(|e| format!("--ngram: {e}"))?,
            "--draft" => a.draft = val()?.parse().map_err(|e| format!("--draft: {e}"))?,
            "--trace" => a.trace = true,
            "--lookup-only" => a.lookup_only = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.model.is_empty() {
        return Err("--model is required".into());
    }
    if a.n == 0 || a.ngram == 0 || a.draft == 0 {
        return Err("--n, --ngram and --draft must be positive".into());
    }
    Ok(a)
}

fn open(a: &Args) -> Result<Engine, String> {
    match &a.pack {
        Some(p) => {
            let pack = nq::open(p).map_err(|e| format!("{p}: {e:?}"))?;
            Engine::open_with_nq(&a.model, pack).map_err(|e| format!("{}: {e:?}", a.model))
        }
        None => Engine::open(&a.model).map_err(|e| format!("{}: {e:?}", a.model)),
    }
}

/// The replay of an exact prompt-lookup drafter over a causal stream.
struct Sim {
    rounds: usize,
    drafts: usize,
    accepted: usize,
    hist: Vec<usize>,
    emitted: usize,
}

/// Replays the drafter over `stream` with the decoder's own rules: the proposal reads only the
/// causal prefix ending at `p`, and the accepted prefix is the longest match against the stream.
#[allow(
    clippy::many_single_char_names,
    reason = "p, q, t, a and n are the position, search cursor, token and occurrence the replay               names"
)]
fn simulate(stream: &[u32], ngram: usize, draft: usize) -> Sim {
    let mut p = 0usize;
    let mut rounds = 0usize;
    let mut drafts = 0usize;
    let mut accepted = 0usize;
    let mut hist = vec![0usize; draft + 1];
    while p + 1 < stream.len() {
        rounds += 1;
        let end = p + 1;
        let start = end.saturating_sub(ngram);
        if start == 0 && end < ngram {
            p += 1;
            hist[0] += 1;
            continue;
        }
        let needle = &stream[start..end];
        // The latest earlier occurrence whose match ends at or before `start`.
        let mut found: Option<usize> = None;
        let mut q = 0usize;
        while q + needle.len() <= start {
            if stream[q..q + needle.len()] == *needle {
                found = Some(q + needle.len());
            }
            q += 1;
        }
        let Some(mut at) = found else {
            p += 1;
            hist[0] += 1;
            continue;
        };
        // The drafter may only read the causal prefix: the context ends at `p`, so the earlier
        // match's continuation stops there too. A proposal that reached past `p` would see the
        // future the decoder does not have. The stream's own end caps the replay.
        let take = draft.min(p + 1 - at).min(stream.len() - 1 - p);
        let mut got = 0usize;
        while got < take && stream[at] == stream[p + 1 + got] {
            got += 1;
            at += 1;
        }
        drafts += take;
        accepted += got;
        hist[got] += 1;
        p += 1 + got;
    }
    Sim {
        rounds,
        drafts,
        accepted,
        hist,
        emitted: p,
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lookup_accept: {e}");
            ExitCode::FAILURE
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "display only: accepted counts and round counts to printed ratios"
)]
fn run() -> Result<(), String> {
    let a = parse_args()?;
    let e = open(&a)?;
    let mut st = e.state(a.n + 8).map_err(|e| format!("{e:?}"))?;
    let mut stream: Vec<u32> = Vec::with_capacity(a.n + 1);
    stream.push(a.first);
    let mut t = a.first;
    let g0 = Instant::now();
    // The greedy arm is the exactness reference and the wall's comparison; `--lookup-only` drops
    // it so a profiler sees the decode pass alone.
    if !a.lookup_only {
        for _ in 0..a.n {
            t = e
                .next_token_state(&mut st, t, KernelMode::Q8Act)
                .map_err(|e| format!("{e:?}"))?;
            stream.push(t);
        }
    }
    let greedy = g0.elapsed().as_secs_f64();
    let sim = simulate(&stream, a.ngram, a.draft);
    println!(
        "stream {} ids; rounds {}; mean {:.3} ids/round; drafts {}; accepted {} \
         ({:.1}% per draft); accepted/round {:.3}",
        stream.len() - 1,
        sim.rounds,
        sim.emitted as f64 / sim.rounds as f64,
        sim.drafts,
        sim.accepted,
        if sim.drafts == 0 {
            0.0
        } else {
            100.0 * sim.accepted as f64 / sim.drafts as f64
        },
        sim.accepted as f64 / sim.rounds as f64,
    );
    println!("accepted-run histogram (0..={}): {:?}", a.draft, sim.hist);
    println!("first ids: {:?}", &stream[..stream.len().min(80)]);
    println!(
        "greedy wall {greedy:.3} s = {:.3} ms/token",
        greedy * 1e3 / a.n as f64
    );
    decode(&e, &a, &stream)
}

/// The real decoder on the same stream: exactness against the greedy ids, the wall and the split.
#[allow(
    clippy::cast_precision_loss,
    reason = "display only: a measured nanosecond count and token count to a printed rate"
)]
fn decode(e: &Engine, a: &Args, stream: &[u32]) -> Result<(), String> {
    if a.trace {
        trace::reset();
        trace::set_enabled(true);
    }
    let t = Instant::now();
    let (ids, stats) = e
        .generate_lookup(std::slice::from_ref(&a.first), a.n, a.ngram, a.draft)
        .map_err(|e| format!("{e:?}"))?;
    let wall = t.elapsed().as_secs_f64();
    if a.trace {
        trace::set_enabled(false);
    }
    if !a.lookup_only && ids.as_slice() != &stream[1..] {
        return Err(format!(
            "lookup ids differ from greedy: {} ids, first mismatch at {:?}",
            ids.len(),
            ids.iter().zip(stream[1..].iter()).position(|(x, y)| x != y)
        ));
    }
    println!(
        "lookup wall {wall:.3} s = {:.3} ms/token; rounds {} accepted {} drafted {}; exact",
        wall * 1e3 / a.n as f64,
        stats.rounds,
        stats.accepted,
        stats.drafted,
    );
    if a.trace {
        print_split(wall, a.n);
    }
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "display only: a measured nanosecond count and token count to printed shares"
)]
fn print_split(wall: f64, n: usize) {
    let phases = trace::snapshot();
    let wall_ns = wall * 1e9;
    let pass_ns = phases.iter().find(|s| s.name == "pass").map_or(0, |s| s.ns);
    let inner: u64 = phases
        .iter()
        .filter(|s| !s.encloses && !s.overlay)
        .filter(|s| !matches!(s.name, "head" | "restore"))
        .map(|s| s.ns)
        .sum();
    let untraced = pass_ns.saturating_sub(inner);
    println!("phase split of the lookup decode, ms/token and share of wall:");
    for s in phases.iter().filter(|s| s.ns > 0) {
        println!(
            "  {:<16} {:8.3} ms/token {:6.1}% calls {}",
            s.name,
            s.ns as f64 / 1e6 / n as f64,
            100.0 * s.ns as f64 / wall_ns,
            s.calls
        );
    }
    println!(
        "  {:<16} {:8.3} ms/token {:6.1}%",
        "untraced",
        untraced as f64 / 1e6 / n as f64,
        100.0 * untraced as f64 / wall_ns
    );
}
