//! Phase split of one decode token: the LM head alone, then the whole step with and without a pack.
//!
//! `head_time <model.gguf> [model.nqp]` prints three timed `lm_head_fixed` calls (the `Exact`
//! head), then three one-token steps under `Exact` and `Q8Act` with the pack installed. The `Exact`
//! step minus the head is the trunk (NQ projections, attention and norms); the `Q8Act` step is the
//! shipped path. With a pack, three `lm_head_cols(&[x], Q8Act)` calls time the shipped head on its
//! own reader — the line that says whether the head or the trunk owns the token.
use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::nq;
use std::time::Instant;

#[allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    reason = "measurement harness: `n`, `x`, `t`, `s` are the width, the activation, the clock and \
              the elapsed seconds it prints; the weight count is far below 2^53"
)]
fn main() -> Result<(), String> {
    let model = std::env::args().nth(1).ok_or("model")?;
    let pack = std::env::args().nth(2);
    let e = Engine::open(&model).map_err(|e| format!("{e:?}"))?;
    let n = e.config.n_embd;
    let x = vec![1i64 << 32; n];
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let logits = e.lm_head_fixed(&x).map_err(|e| format!("{e:?}"))?;
        let s = t.elapsed().as_secs_f64();
        best = best.min(s);
        eprintln!("head {:?} words {:.3} s", logits.len(), s);
    }
    let vocab = e.config.vocab_size;
    #[allow(clippy::cast_precision_loss, reason = "display rate")]
    let rate = (vocab * n) as f64 / best / 1e9;
    eprintln!("head best {best:.3} s, {rate:.2} G weight/s");
    let Some(pack) = pack else { return Ok(()) };
    let p = nq::open(&pack).map_err(|e| format!("{e:?}"))?;
    let e = Engine::open_with_nq(&model, p).map_err(|e| format!("{e:?}"))?;
    if std::env::var_os("APEIRON_HEAD_NO_COLS").is_none() {
        let xs = vec![x.clone()];
        let mut hbest = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            let out = e
                .lm_head_cols(&xs, KernelMode::Q8Act)
                .map_err(|e| format!("{e:?}"))?;
            let s = t.elapsed().as_secs_f64();
            hbest = hbest.min(s);
            eprintln!("head Q8Act words {} {:.3} s", out[0].len(), s);
        }
        let rate = (vocab * n) as f64 / hbest / 1e9;
        eprintln!("head Q8Act best {hbest:.3} s, {rate:.2} G weight/s");
    }
    for mode in [KernelMode::Exact, KernelMode::Q8Act] {
        let mut sbest = f64::INFINITY;
        for _ in 0..3 {
            let mut st = e.state(64).map_err(|e| format!("{e:?}"))?;
            let t = Instant::now();
            let _ = e
                .next_token_state(&mut st, 1, mode)
                .map_err(|e| format!("{e:?}"))?;
            sbest = sbest.min(t.elapsed().as_secs_f64());
        }
        eprintln!("step {mode:?} best {sbest:.3} s, {:.2} tok/s", 1.0 / sbest);
    }
    Ok(())
}
