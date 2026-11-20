//! Compare two LM-head images on one deterministic activation: `head_cmp <a.gguf> <b.gguf>`.
//!
//! Both heads run the `Exact` per-column projection, so the difference is the weight image alone
//! (a quantised `output.weight` against the tied `F16` one), not the kernel or the activation
//! quantisation. Prints the largest absolute logit, the largest deviation, the deviation relative
//! to that logit, and each image's argmax token.
use apeiron::engine::loader::Engine;

fn main() -> Result<(), String> {
    let path_a = std::env::args().nth(1).ok_or("model a")?;
    let path_b = std::env::args().nth(2).ok_or("model b")?;
    let ea = Engine::open(&path_a).map_err(|e| format!("{e:?}"))?;
    let eb = Engine::open(&path_b).map_err(|e| format!("{e:?}"))?;
    let width = ea.config.n_embd;
    let mut act = Vec::with_capacity(width);
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..width {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // Q32.32 with |value| < 2, both signs: 31 bits before the shift.
        let word = i64::try_from((seed >> 32) & 0x7fff_ffff).expect("31 bits");
        act.push((word - (1 << 30)) << 1);
    }
    let la = ea.lm_head_fixed(&act).map_err(|e| format!("{e:?}"))?;
    let lb = eb.lm_head_fixed(&act).map_err(|e| format!("{e:?}"))?;
    let mut max_dev = 0u64;
    let mut max_abs = 0u64;
    let mut top_a = (0usize, i64::MIN);
    let mut top_b = (0usize, i64::MIN);
    for col in 0..la.len() {
        max_dev = max_dev.max(la[col].abs_diff(lb[col]));
        max_abs = max_abs.max(la[col].unsigned_abs());
        if la[col] > top_a.1 {
            top_a = (col, la[col]);
        }
        if lb[col] > top_b.1 {
            top_b = (col, lb[col]);
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "display ratio")]
    let relative = max_dev as f64 / max_abs.max(1) as f64;
    println!(
        "max|logit| {max_abs} words, max|a-b| {max_dev} words, relative {relative:.3e}, \
         top {}/{}",
        top_a.0, top_b.0
    );
    Ok(())
}
