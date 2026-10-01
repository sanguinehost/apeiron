# Unified quant projection kernel

Owner: Apeiron lead. Source base: `4b5eff5`. Status: dispatched (byte-identical unification).

## Intent

The executing transformer must not be reimplemented once per quantization format. The slow
`matvec_fixed` layer already keeps one kernel across every kind (`dequant_row_fixed` block codec +
`row_dot_fixed`), but the *fast* prefill layer forks into one dot per family in
`src/engine/chart/mod.rs`:

```
520  fn q8act_q8_0_row_fast(...)
604  fn q8act_q5_0_row_fast(...)
722  fn q8act_q4_k_row_fast(...)
803  fn q8act_q6_k_row_fast(...)
```

Each owns its own `#[target_feature(enable = "avx2")]` accumulator loop and its own portable twin.
Waves 5–6 tuned only the `Q8_0` arm; the dense Qwen3-8B-Q4_K_M prefill measured `ffn` at
~113 GMAC/s versus ~1000 GMAC/s for the Q8_0 hybrid. The two are the same mathematical object;
only the block decode differs. The goal is one accumulator kernel parameterized by a block decoder,
so a new quantization format costs a decoder, not a dot and not a prefill path.

Non-goals: no change to any output value, refusal, fold order or evidence grade. No speed claim is
made by this epic; a measured Q4_K improvement is a separate optimization claim gated by
`tools/dispatch/admit.py --launch`.

## Specification

| boundary | contract |
|---|---|
| Arithmetic | one accumulation kernel computes `acc += act_scale · (w_scale · dot − w_min · xsum)` per 32-lane group, in the existing per-group order; `w_min = 0` for the offset-free families |
| Decode | per-family block decoder yields groups; AVX2 lane primitives (`x86::dot32_ptr`, `q5_0_dot32_ptr`, `q4_k_block_dots`, `q4_k_window_dot_ptr`, `q6_k_half_dots`) are retained, not restated |
| Value identity | every output byte-identical to the current fast rows and to `matvec_fixed`; the existing family parity tests are the oracle |
| Fallback | non-finite weight scale, non-fused family, bound failure and non-multiple widths keep routing to `matvec_fixed`; the refusal/fallback contract is unchanged |
| Extraction | `chart` is outside the extracted fragment (`src/engine/dequant.rs` and `src/engine/matvec.rs` are the extracted roots); `dequant.rs` must not change |
| Complexity | `O(n_in)` time, `O(1)` extra space per row, unchanged |

## Plan

One writer per file. The task touches the three `chart` files, all disjoint from the GPU wave.

| task | owner / files | gate / control |
|---|---|---|
| APEIRON-QUANT-UNIFY-W1-T1 | worker `gemini-flash`; `src/engine/chart/mod.rs`, `src/engine/chart/x86.rs`, `src/engine/chart/portable.rs` | gate `cargo test --locked --lib engine::chart` byte-identical for every family; control mutates one codec scale/min in a throwaway copy and requires a parity failure, then restores green; `cargo test --locked --lib engine::kern engine::step` also green |

Arc Block:

```text
SCOPE:      chart/mod.rs, chart/x86.rs, chart/portable.rs; collapse the four q8act_*_row_fast
            kernels into one decoder-parameterized accumulator. Completion: family parity green.
REGISTER:   structural unification/refactor; no numerical, contract or performance claim.
CORPUS:     chart/ (outside extraction), dequant.rs, matvec.rs read; no canonical theorem changed.
REFERENCE:  /home/socol/Workspace/apeiron/src/engine/chart/mod.rs.
GATE:       cargo test --locked --lib engine::chart; observed red is the seeded codec mutation.
GRADE:      tested on byte identity; any speedup is unregistered and admission-gated.
```

## Open

- Q4_K prefill attribution: whether the dense 8B is falling back to `matvec_fixed` or taking an
  untuned fast arm is unmeasured. Separate bounded task; the collapse does not assume it.
- `engine_profile`'s `--llama` path passes no device pin, so a GPU-visible llama-bench silently
  compares Apeiron CPU against llama.cpp GPU. Harness fix, separate task.
