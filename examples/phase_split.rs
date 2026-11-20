//! The traced decode split, and the dominant-term gate.
//!
//! `phase_split --model <gguf> [--pack <nqp>] [--steps 16] [--target <phase>] [--json]`
//!
//! Runs decode steps under [`apeiron::engine::trace`], prints each phase's milliseconds per token
//! and share of wall, and names the largest term. With `--target`, exits 1 unless that phase owns
//! the largest share: a change to a phase holding share `s` cannot exceed `1/s`, so a target that
//! is not the dominant term is refused before it is built. `untraced` is a legal target: it names
//! the part of a pass no span covers, and choosing it means instrumenting the pass first.
//!
//! The wall denominator is the measured decode loop, so the shares include the frame around the
//! traced spans. `pass` and `mtp` enclose their leaves and are reported, not counted as terms;
//! `pool.wait` overlaps the others and is reported separately.
use std::process::ExitCode;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::trace;
use apeiron::engine::KernelMode;
use apeiron::format::nq;

struct Args {
    model: String,
    pack: Option<String>,
    steps: usize,
    target: Option<String>,
    json: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        pack: None,
        steps: 16,
        target: None,
        json: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--pack" => a.pack = Some(val()?),
            "--steps" => a.steps = val()?.parse().map_err(|e| format!("--steps: {e}"))?,
            "--target" => a.target = Some(val()?),
            "--json" => a.json = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.model.is_empty() {
        return Err("--model is required".into());
    }
    if a.steps == 0 {
        return Err("--steps must be positive".into());
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

/// One named term of the split: a leaf phase, the head, or the pass's untraced remainder.
#[derive(Clone)]
struct Term {
    name: String,
    ns: u64,
}

#[allow(
    clippy::cast_precision_loss,
    reason = "display rates and shares of measured nanosecond counters"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the split, its report and the gate are one measurement"
)]
fn run(a: &Args) -> Result<bool, String> {
    let e = open(a)?;
    let mut warm = e.state(8).map_err(|e| format!("{e:?}"))?;
    let _ = e
        .next_token_state(&mut warm, 1, KernelMode::Q8Act)
        .map_err(|e| format!("{e:?}"))?;
    let mut st = e.state(a.steps + 8).map_err(|e| format!("{e:?}"))?;
    trace::reset();
    trace::set_enabled(true);
    let t = Instant::now();
    let mut id = 1u32;
    for _ in 0..a.steps {
        id = e
            .next_token_state(&mut st, id, KernelMode::Q8Act)
            .map_err(|e| format!("{e:?}"))?;
    }
    let wall = t.elapsed().as_secs_f64();
    trace::set_enabled(false);
    let stats = trace::snapshot();
    let wall_ns = wall * 1e9;
    let tok = a.steps as f64;
    let ns_of = |name: &str| stats.iter().find(|s| s.name == name).map_or(0, |s| s.ns);
    let pass_ns = ns_of("pass");
    let inner: u64 = stats
        .iter()
        .filter(|s| !s.encloses && !s.overlay)
        .filter(|s| !matches!(s.name, "head" | "restore"))
        .map(|s| s.ns)
        .sum();
    let untraced = pass_ns.saturating_sub(inner);
    let mut terms: Vec<Term> = stats
        .iter()
        .filter(|s| !s.encloses && !s.overlay && s.ns > 0)
        .map(|s| Term {
            name: s.name.to_string(),
            ns: s.ns,
        })
        .collect();
    terms.push(Term {
        name: "untraced".into(),
        ns: untraced,
    });
    let top = terms.iter().max_by_key(|t| t.ns).cloned().unwrap_or(Term {
        name: "none".into(),
        ns: 0,
    });

    if a.json {
        println!("{{\"schema\":\"apeiron.phase_split.v1\",\"model\":\"{}\",\"steps\":{},\"wall_s\":{wall:.6},\"phases\":[", a.model, a.steps);
        let mut first = true;
        for s in stats.iter().filter(|s| s.ns > 0) {
            if !first {
                print!(",");
            }
            first = false;
            print!(
                "{{\"phase\":\"{}\",\"ns\":{},\"share\":{:.6},\"calls\":{},\"encloses\":{},\"overlay\":{}}}",
                s.name,
                s.ns,
                s.ns as f64 / wall_ns,
                s.calls,
                s.encloses,
                s.overlay
            );
        }
        print!(
            "],\"untraced_ns\":{untraced},\"dominant\":\"{}\",\"dominant_share\":{:.6},\
             \"dominant_ceiling\":{:.4}}}",
            top.name,
            top.ns as f64 / wall_ns,
            1.0 / (1.0 - top.ns as f64 / wall_ns).max(1e-9)
        );
        println!();
    } else {
        println!(
            "decode split: {} steps, {:.3} s wall, {:.2} ms/token",
            a.steps,
            wall,
            wall * 1e3 / tok
        );
        println!("| phase | ms/token | share | calls |");
        println!("|---|---:|---:|---:|");
        for s in &stats {
            if s.ns == 0 {
                continue;
            }
            let note = if s.encloses {
                " (encloses)"
            } else if s.overlay {
                " (overlay)"
            } else {
                ""
            };
            println!(
                "| {}{} | {:.3} | {:.1}% | {} |",
                s.name,
                note,
                s.ns as f64 / 1e6 / tok,
                100.0 * s.ns as f64 / wall_ns,
                s.calls
            );
        }
        println!(
            "| untraced | {:.3} | {:.1}% | |",
            untraced as f64 / 1e6 / tok,
            100.0 * untraced as f64 / wall_ns
        );
        println!(
            "dominant term: {} at {:.1}%, ceiling {:.2}x",
            top.name,
            100.0 * top.ns as f64 / wall_ns,
            1.0 / (1.0 - top.ns as f64 / wall_ns).max(1e-9)
        );
    }

    let Some(target) = &a.target else {
        return Ok(true);
    };
    let target_ns = if target == "untraced" {
        untraced
    } else {
        terms.iter().find(|t| &t.name == target).map_or(0, |t| t.ns)
    };
    let share = target_ns as f64 / wall_ns;
    let top_share = top.ns as f64 / wall_ns;
    if target_ns != top.ns {
        eprintln!(
            "REFUSED: target {target} holds {:.1}% (ceiling {:.2}x); the dominant term is {} at \
             {:.1}% (ceiling {:.2}x). Target the dominant term, or instrument it if it is the \
             untraced remainder.",
            100.0 * share,
            1.0 / (1.0 - share).max(1e-9),
            top.name,
            100.0 * top_share,
            1.0 / (1.0 - top_share).max(1e-9)
        );
        return Ok(false);
    }
    eprintln!(
        "TARGET {target}: share {:.1}%, ceiling {:.2}x",
        100.0 * share,
        1.0 / (1.0 - share).max(1e-9)
    );
    Ok(true)
}

fn main() -> ExitCode {
    match parse_args().and_then(|a| run(&a)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("phase_split: {e}");
            ExitCode::FAILURE
        }
    }
}
