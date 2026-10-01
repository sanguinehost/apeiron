# Inference integration receipt

The integrated source is `39298a1c858a11397b7a9e3d638c5a0b796b4a47`. Checkpoint
`4cc0ede368f081516c4f71dc9f3b82c0cb8d3637` preserves pre-existing work; it certifies none of it.
The subsequent repairs check convolution accumulation and rounding overflow, validate complete
external benchmark rows, and extract the real half-word conversion through locked Cargo inputs.
The mixer tests move to `src/engine/mixer/tests.rs`. Protected policy paths and `src/lib.rs` retain
their checkpoint bytes.

## Gates

Logs resolve under `/home/socol/oc-tmp/astra-integration/`. Every invocation and its descendants
has a 540-second limit. Lean additionally uses the shared lane lock, one thread, 12 GiB memory
and no swap. Times below are observed wall seconds, not performance comparisons.

| command | exit | seconds | artifact |
|---|---:|---:|---|
| `cargo test --locked --lib engine::mixer::tests` | 0 | 3.60 | `conv-t2.log`: 6 passed, 1 ignored |
| `cargo test --locked --example engine_profile` | 0 | 1.79 | `benchmark-t3.log`: 3 passed |
| `cargo test --locked --lib f16_split_is_f16_to_fixed` | 0 | 2.95 | `pow2-t5.log`: exhaustive half-word test |
| `proof/regen.sh` | 0 | 124.92 | `regen.log`: actual extraction and 1,708 build jobs |
| `proof/regen.sh --check` | 0 | 101.10 | `regen-check.log`: fresh extraction |
| `cd proof/corpus && lake build` | 0 | 492.88 | `bridge.log`: 9,218 build jobs, existing audits |
| full Rust set below, then release profile build | 0 | 220.25 | `rust-barrier-final.log`: 526 passed, 12 ignored |
| real-source extraction mutation and restoration | 0 | 160.89 | `extraction-control.log` |

The full Rust invocation runs `cargo test --workspace --all-features --locked`,
`cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`,
`cargo fmt --all -- --check`, then `cargo build --release --locked --example engine_profile`.
A prior formatting failure is retained in `rust-barrier.log`; the final invocation follows its
repair. The intentionally cancelled overlapping extraction/Rust run is not acceptance evidence.

The extraction control changes the real fraction mask from `0x3ff` to `0x3fe` in a disposable
copy. Freshness rejects it with exit 1 and the generated `1022#u16` mismatch at 70.44 seconds;
restoring the source gives exit 0 and a successful Lean package build. Detailed red/restore logs
and source identities accompany `extraction-control-identity.json`. Earlier convolution and
benchmark-parser defect controls are retained in
`/home/socol/oc-tmp/astra-luna-baseline/receipt.md`.

## Identity and scope

- Charon `0.1.254`, `b104e24fea7d721b71e6c39fd70f26ff20bc0980`;
  Aeneas `505b6ca-dirty`; extraction Rust `nightly-2026-08-18`.
- Lean `4.31.0`; Mathlib `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`;
  canonical manifest `880aa57e47e782ed7a9d9ee1974cfc87360b446b` remains unchanged.
- Benchmark source digest `96c75f9773af6cb9d0f0ef2536cc7c0c8ccb86020759fe614c214a706300c8de`;
  binary hashes are in `baseline-binaries.sha256` beside the logs.
- Fresh measurements and hardware limits: [prefill attribution](perf-attribution-prefill.md).

Grade: Rust behavior is **tested**; the existing named Lean/audit build is **proved** within its
existing premises. Extraction alone does not prove the new `pow2` refinements; the subsequent
section records the checked half-conversion link. Activation scaling and finishing remain open.
These gates do not establish compiler/code-generation correctness, CPU hardware correctness,
universal numerical stability, low-end hardware throughput or rendered frames per second. The
programme remains open. The subsequent registry split discharges that file's size obligation;
other existing oversized files, including `step.rs` and `engine/tests.rs`, remain outside this slice.

## Exact splitter and rejected traversal

Source identity: `8c142228c42ff979ab904e71c97fbaf1f0a9915e`, with per-file SHA-256 values in
`/home/socol/oc-tmp/astra-integration/w1-source-identity.json`. Protected paths and corpus pins
still match the preservation checkpoint. All new owned source files are below 1,500 lines.

The module `Numeric.Pow2Refinement` proves `apeiron_word.word_f16_split_exact`. For every
`h : U16` with `f16Exp h ≠ 31`, it exhibits `m : I32` and `k : U32`, proves the actual extracted
wrapper returns `ok (some (m,k))`, and proves `m.val * 2^k.val = f16Value h`. The companion
`word_f16_split_nonfinite` proves `ok none` for exponent 31. Normal finite and nonfinite joint
witnesses are checked. All four declarations pass the existing closed axiom cone. Activation
scaling, finishing and the current activation-chart specification remain separate open refinements.

The audit registry preserves all 155 original imports and 1,797 audit commands in their original
order. Its two parts reconstruct the original bytes with the original boundary newline; the root
still reaches every check. The new root additionally audits the four splitter declarations. A fresh
Astra review accepts the Luna registry factoring; a fresh Luna review accepts the Astra finite
proof. Reviews do not replace the gates below.

| command/control | exit | seconds | log |
|---|---:|---:|---|
| `lake build AxiomAudit` after factoring | 0 | 13.02 | `audit-factoring.log` |
| forbidden-axiom baseline/red/restore | 0/1/0 | 6.33 total | `audit-control.log` |
| `lake build Numeric.Pow2Refinement AxiomAudit` | 0 | 10.01 | `pow2-proof-build.log` |
| actual Rust fraction-mask mutation, regeneration, proof rejection and restoration | 0 | 252.26 | `pow2-proof-control.log` |
| retained AVX2 boundary test baseline/coefficient defect/restore | 0/101/0 | 4.16 total | `retained-write-control.log` |
| full Rust set, release profile build and release long-prompt parity test | 0 | 254.27 | `w1-rust-barrier.log` |

The Rust set has 527 passing tests and 12 ignored; the separately requested release long-prompt
case adds one pass. That case checks 32 generated token IDs after the 200-token prompt. The prompt
is now `tests/fixtures/qwen35-prompt2.ids`, unchanged SHA-256
`894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`.

The proof control rebuilds the disposable extraction from real Rust with `0x3ff` changed to
`0x3fe`. The imported module printout contains `1022#u16`; the new proof fails at its body
correspondence with that exact changed mask. Restoration regenerates and rebuilds `1023#u16`,
then the proof passes. The control overrides only the extracted-module path and reuses unchanged
checked dependencies. A kernel-environment scan checks 419,092 imported declarations and finds
no reference to either changed declaration outside the rebuilt splitter and wrapper. The generated
files differ only at that mask. Logs, printed imported bodies and hashes retain those checks; no
stale original extraction can account for the intended red.

The affine traversal experiment has no supported speedup. Five quiet rotated rounds measure
column/row median paired ratios 1.000292 prefill and 0.996381 decode; four-core ratios are 1.002071
and 0.966693. [The receipt](bench-ledger.md#1790608521-astra-write-traversal) records the complete samples.
The candidate and its switch are removed. The executing loop matches its pre-experiment body;
only a complexity comment and stronger scalar boundary/refusal tests remain. The four-core row
measurement is 49.41 tokens/s on the i9, not a 60-FPS or low-end-machine result.

The final live `proof/regen.sh --check` (including the extraction package build), followed by
`cd proof/corpus && lake build`, passes with exit 0 and 9,221 bridge jobs. The whole invocation,
including its serialized lane wait, takes 269.31 seconds (`w1-proof-barrier.log`). This completes
the slice's Rust/extraction/bridge/control barrier on the named source identity; it does not close
the programme's remaining refinements, size inventory or performance targets.


## Final scheduling decision

Final source identity: `86f4a42c19014eaf76438223841cc3d093782ad6`; per-file hashes are in
`/home/socol/oc-tmp/astra-integration/final-source-identity.json`. The six protected tracked paths
and both proof packages' toolchain/manifest files retain their checkpoint bytes. Task worktrees
and branches are removed; unrelated untracked files and the pre-existing `one-allocation` branch
remain untouched.

The flat-buffer and owned-output SwiGLU variants pass scalar parity, overflow/geometry tests and
real-model preflight, but neither supplies sufficient measured benefit for adoption. The owned
variant groups columns into at most 64 tasks and reuses the existing scalar/AVX2 arithmetic with
no nested pool call. Independent Astra review accepts its internal owned-output interface and
task bound; this is a review verdict, not a new universal arithmetic theorem.

| command/control | exit | seconds | log |
|---|---:|---:|---|
| flat-buffer focused FFN tests, full Clippy and formatting | 0 | 4.01 | `swiglu-clippy-final.log` |
| flat-buffer operand-swap baseline/red/restore | 0/101/0 | 2.06 | `swiglu-control-final.log` |
| flat-buffer selected release model tests and profile build | 0 | 15.77 | `swiglu-model-preflight.log` |
| flat-buffer five-round comparison | 0 | 179.59 | `swiglu-ab.log` |
| owned-output focused FFN tests, full Clippy and formatting | 0 | 6.47 | `swiglu-owned-focused.log` |
| owned-output column-zero operand-swap baseline/red/restore | 0/101/0 | 2.32 | `swiglu-owned-control.log` |
| owned-output selected release model tests and profile build | 0 | 15.67 | `swiglu-owned-model-preflight.log` |
| owned-output five-round comparison | 0 | 180.88 | `swiglu-owned-ab.log` |

Each selected release model run passes four tests with three explicitly ignored. The owned-output
control swaps gate/up only in scheduled column zero and fails scalar parity at 64 columns of
width seven; restoration passes. Both controls mutate disposable copies only.

[Flat-buffer samples](bench-ledger.md#1790610276-astra-swiglu-batch) give median paired prefill ratio
0.986669. [Owned-column samples](bench-ledger.md#1790611132-astra-swiglu-owned) give 1.027322, but
two of five prefill rounds regress, four-core prefill is 1.001618, and speculation loses four of
five rounds in both settings. These samples do not justify the extra path; the decision is rejection,
not a claim that its true performance effect is zero. The latest baseline measures 68.86 greedy
tokens/s default and 49.97 on four i9 performance cores. It measures inference, not rendered FPS
or throughput on a separate low-end machine.

Cleanup restores `src/engine/chart/ffn.rs` and `src/engine/step/cols.rs` exactly to `608e726`,
removing every provisional helper consumer and the environment switch. A source-wide search
confirms neither remains. The accepted overflow fixes, retained mixer boundary tests, reproducible
prompt, audit factoring and exact half conversion proof remain. No performance candidate survives
this slice. Activation scaling, finishing, the projection epilogue's isolated measured cost, whole
executor refinement and the programme's throughput targets remain open.

The final Rust invocation runs the focused five-test FFN gate, the full workspace/all-feature test
set, full Clippy with warnings denied, formatting, release profile build and the explicit release
long-prompt test. It exits 0 in 221.24 seconds (`final-rust-barrier.log`): 527 full-suite passes,
12 ignored, plus the separately run long-prompt pass and five focused passes. The long-prompt
case again checks the expected 32 token IDs after the 200-token input.

The final `proof/regen.sh --check` freshly extracts the real Rust and builds the extraction package;
`cd proof/corpus && lake build` then completes 9,221 bridge jobs. The combined locked, memory-capped
invocation exits 0 in 78.85 seconds (`final-proof-barrier.log`). The four splitter headlines still
pass their registered closed-cone audits. All final source hashes match the receipt after both gates.
This closes the local integration slice at grades **tested**, **proved** for the named conversion
statements, and **measured** for the recorded workloads; the performance and refinement residuals
above prevent programme closure.
