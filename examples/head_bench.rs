//! The LM head in isolation: `head_bench <model.gguf> [--cols 9] [--reps 20]`.
//!
//! Times [`Engine::lm_head_cols`] with `KernelMode::Q8Act` on deterministic activation columns,
//! so a head carrier or head-kernel change is a wall number without the 196-projection pass around
//! it. The activation is built from a fixed xorshift stream in Q32.32, the same shape `head_cmp`
//! uses. Prints ms per call and per column.
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;

#[allow(
    clippy::cast_precision_loss,
    reason = "measurement harness: the column count is far below 2^53, and the ratio is display only"
)]
fn main() -> Result<(), String> {
    let mut a = std::env::args().skip(1);
    let path = a.next().ok_or("model")?;
    let mut cols = 9usize;
    let mut reps = 20usize;
    while let Some(flag) = a.next() {
        match flag.as_str() {
            "--cols" => {
                cols = a
                    .next()
                    .ok_or("--cols value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--reps" => {
                reps = a
                    .next()
                    .ok_or("--reps value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let e = Engine::open(&path).map_err(|e| format!("{e:?}"))?;
    let width = e.config.n_embd;
    let mut xs: Vec<Vec<i64>> = Vec::with_capacity(cols);
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..cols {
        let mut act = Vec::with_capacity(width);
        for _ in 0..width {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let word = i64::try_from((seed >> 32) & 0x7fff_ffff).map_err(|e| format!("{e}"))?;
            act.push((word - (1 << 30)) << 1);
        }
        xs.push(act);
    }
    // One unmeasured call installs the payload and the pool's first dispatch.
    let warm = e
        .lm_head_cols(&xs, KernelMode::Q8Act)
        .map_err(|e| format!("{e:?}"))?;
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let got = e
            .lm_head_cols(&xs, KernelMode::Q8Act)
            .map_err(|e| format!("{e:?}"))?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(got.len());
        best = best.min(ms);
    }
    println!(
        "{path}: cols {cols} reps {reps} best {best:.3} ms = {:.3} ms/column, logits {}",
        best / cols as f64,
        warm.len()
    );
    Ok(())
}
