//! `Engine::open` and one decode step over an attacker GGUF file.
//!
//! The input is written to a per-process file and opened through the public loader: header
//! parse, config resolution, the `prism.hadamard.*` contract, the tensor-extent table and the
//! decode vectors. An opened engine then runs `step(0)`, so every tensor shape the loader
//! admits reaches the kernels that index it. Every refusal is a pass; a panic, an out-of-bounds
//! index, an allocation past `-rss_limit_mb` or a sanitizer report is a failure. The seed
//! `fuzz/seeds/gguf_open/qwen2-one-layer-f16` is `build_layer_gguf` from
//! `src/engine/step/prefill_tests.rs`, a model the loader accepts and steps.
#![no_main]

use apeiron::engine::Engine;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let path = std::env::temp_dir().join(format!("apeiron_fuzz_gguf_open_{}", std::process::id()));
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let Some(p) = path.to_str() else {
        return;
    };
    let Ok(mut engine) = Engine::open(p) else {
        return;
    };
    let _ = engine.step(0);
});
