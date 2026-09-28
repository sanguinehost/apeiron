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
existing premises. New extracted `pow2` roots do not yet have their own checked refinement links.
In particular, half conversion, activation scaling and finishing are not promoted to proved by
extraction alone. The half conversion link is the next bounded proof task. These gates do not
establish compiler/code-generation correctness, CPU hardware correctness, universal numerical
stability, low-end hardware throughput or rendered frames per second. The programme remains open.
Existing oversized files, including the 2,373-line audit registry, retain their size obligations;
the next task factors that registry without changing its audit population.
