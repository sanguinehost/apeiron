//! End-to-end checks of the packed `NanoQuant` linears through the engine.
//!
//! The unit tests in `src/engine/kern/nq.rs` own the kernel algebra: the fused product against a
//! term-by-term reference, the wide-accumulator fallback, and the sign-flip mutation. This file
//! checks the two properties only the whole model can show:
//!
//! * multi-column prefill and one-token steps are the same executor, including every packed
//!   projection;
//! * reading the pack twice and running the same prompt twice gives the same logits bit for bit,
//!   so the served model does not depend on load order or on residual state between runs.
//!
//! Both are ignored by default: they open the converted 0.6B skeleton (337 MB) and pack (63 MB).
//! Run with
//! `cargo test --release --test engine_nq -- --ignored --nocapture`
//! and optionally `NQ_PACK_DIR=...`.

use std::path::PathBuf;

use apeiron::engine::{argmax_fixed, Engine, KernelMode};

fn pack_dir() -> PathBuf {
    std::env::var_os("NQ_PACK_DIR").map_or_else(
        || PathBuf::from("/home/lucas/models/arelath/nq06"),
        PathBuf::from,
    )
}

fn join(name: &str) -> String {
    pack_dir().join(name).to_string_lossy().into_owned()
}

fn cosine(a: &[i64], b: &[i64]) -> f64 {
    // Sums of squared logits (about 3.8e10 each) exceed `i64`, so the dot product is `i128`.
    let (mut dot, mut na, mut nb) = (0i128, 0i128, 0i128);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (i128::from(*x), i128::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    // The ratio is compared to 0.999999; `f64`'s 53-bit mantissa leaves 1e-16 relative error on
    // sums of this size, which is far below the assertion's gap.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the cosine is compared to 0.999999"
    )]
    let (dot, na, nb) = (dot as f64, na as f64, nb as f64);
    dot / (na.sqrt() * nb.sqrt())
}

/// Ordinary Qwen3 token ids; the exact text is irrelevant, the width and the repetition are not.
const PROMPT: &[u32] = &[9707, 11, 1879, 323, 358, 1079, 13, 576, 220, 17, 433, 303];

/// The multi-column prefill must be the same executor as one-token steps, including the packed
/// projections. A divergence here is a prefill-only path, not a pack error.
#[test]
#[ignore = "opens the 337 MiB skeleton and 63 MiB pack; run with --release --ignored"]
fn packed_prefill_columns_equal_single_steps() {
    let pack = apeiron::format::nq::open(&join("model.nqp")).expect("open pack");
    assert_eq!(pack.linears.len(), 196);
    let engine = Engine::open_with_nq(&join("skeleton.gguf"), pack).expect("open packed");

    let mut a = engine.state(PROMPT.len()).expect("prefill state");
    let columns = engine
        .prefill_states(&mut a, PROMPT, KernelMode::Exact)
        .expect("prefill");
    let mut b = engine.state(PROMPT.len()).expect("step state");
    for (i, &id) in PROMPT.iter().enumerate() {
        let step = engine
            .step_trunk_state(&mut b, id, KernelMode::Exact)
            .expect("step");
        assert_eq!(step, columns[i], "column {i}");
    }
}

/// Two loads of the same pack, and two prefills through one engine, must produce the same logits.
/// A load-order dependence would show as a nonzero `max|d|`; the check is exact, not a tolerance.
#[test]
#[ignore = "opens the 337 MiB skeleton and 63 MiB pack twice; run with --release --ignored"]
fn packed_logits_are_bit_stable_across_loads() {
    let first = Engine::open_with_nq(
        &join("skeleton.gguf"),
        apeiron::format::nq::open(&join("model.nqp")).expect("open pack"),
    )
    .expect("open packed");
    let second = Engine::open_with_nq(
        &join("skeleton.gguf"),
        apeiron::format::nq::open(&join("model.nqp")).expect("open pack"),
    )
    .expect("open packed");

    let mut worst = 0u64;
    for (p, engine) in [&first, &second].into_iter().enumerate() {
        let mut state = engine.state(PROMPT.len()).expect("state");
        let hidden = engine
            .prefill_states(&mut state, PROMPT, KernelMode::Exact)
            .expect("prefill");
        let logits = engine
            .lm_head_fixed(hidden.last().expect("column"))
            .expect("head");
        println!(
            "load {p}: token {}, max|logit| {}",
            argmax_fixed(&logits).expect("argmax"),
            logits.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
        );
        if p == 1 {
            let mut state = first.state(PROMPT.len()).expect("repeat state");
            let hidden = first
                .prefill_states(&mut state, PROMPT, KernelMode::Exact)
                .expect("repeat prefill");
            let again = first
                .lm_head_fixed(hidden.last().expect("column"))
                .expect("repeat head");
            worst = logits
                .iter()
                .zip(&again)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            assert!(cosine(&logits, &again) > 0.999_999);
        }
    }
    assert_eq!(worst, 0, "the same engine and prompt gave different logits");
}
