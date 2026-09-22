# Plan — Parallel stable scatter

- **Owner:** Astra architecture; Sol contract/attribution; Flash implementation
- **Intent:** `docs/intent/parallel-stable-scatter.md`
- **Spec:** `docs/spec/parallel-stable-scatter.md`
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for serialized dispatch

## Arc block

```text
SCOPE:      stable scatter source/test work and matched spatial measurement; complete when focused gates/controls pass and every strict spatial pair is recorded without weakening G1
REGISTER:   state/test/measurement with an open refinement consequence; excludes layout, routing, kernels, grouped reduction, validator removal, proof implementation and W7 composition
CORPUS:     Apeiron/Table/Basic.lean `build_eq_of_key_streams_eq` (proved canonical authority); GrowableSchedule.partition_preserves_key_streams remains open
REFERENCE:  /home/socol/Workspace/apeiron/src/table/partition.rs, src/table/parallel.rs, tests/parallel.rs and examples/micro_spatial.rs read from the supplied checkout
GATE:       focused commands below; strict full-profile spatial baseline is red and no task may reduce pairs, incumbents, controls or the interval ceiling
GRADE:      lifecycle documents anchor; source targets tested; runtime targets measured; scatter refinement open
```

The documentation task is the supplied-checkout exception and owns four documentation files. Every
implementation task below owns at most three files and one gate. Use an isolated
`worktree-agent-APEIRON-PERF-SCATTER-<id>` at the recorded integration revision. Serialize every wave;
`target/`, `proof/ApeironWord.lean`, `proof/.lake/` and `proof/corpus/.lake/` remain shared lanes.

## Goal and completion condition

The parallel build reserves all storage before worker start, constructs byte-identical serial
`Parts` through chunk-private stable scatter and chunk-order concatenation, retains validation and
ordered callbacks, and records focused plus strict spatial results. G1 remains open unless the strict
full-profile command exits 0 for every applicable pair.

```sh
cargo test --workspace --all-features --locked && cargo clippy --workspace --all-features --all-targets --locked -- -D warnings && cargo fmt --all -- --check && cargo kani && proof/regen.sh --check
```

Run bounded `lake build` separately in `proof/` and `proof/corpus/` under the repository lane, memory,
swap, timeout and thread rules. They are integration barriers, not scatter-proof claims.

## Non-goals

- No extraction/proof edit, validator removal, unsafe storage, callback-order change, grouped
  reduction, routing/LPT/kernel/layout/API change or second representation.
- No strict threshold, incumbent, applicability, workload, interval, control, HashDOS or coverage
  change.

## Wave map

| Wave | Task | Files | Owner | Gate | Rejecting control |
|---|---|---|---|---|---|
| W0 | `APEIRON-PERF-SCATTER-W0-T1` freeze micro baseline | no edit; read `examples/micro_spatial.rs` | Flash | `cargo run --release --locked --example micro_spatial -- --controls` | Corrupt one expected build/probe digest in a throwaway copy; the gate rejects before timing, then restored gate passes. |
| W1 | `APEIRON-PERF-SCATTER-W1-T1` stable chunk scatter primitives | `src/table/partition.rs` | Flash | `cargo test --locked --lib table::partition::tests` | Reverse two adjacent chunk runs for one shard; exact serial equality or integrity test fails, then restored gate passes. |
| W2 | `APEIRON-PERF-SCATTER-W2-T1` reserve-all parallel integration | `src/table/parallel.rs`, `tests/parallel.rs` | Flash | `cargo test --locked --test parallel` | Select completion order or start scatter before final-shard reservation; non-commutative equivalence or zero-effect refusal fails, then restored gate passes. |
| W3 | `APEIRON-PERF-SCATTER-W3-T1` focused post-change spatial measurement | no edit; read `examples/micro_spatial.rs` | Flash | `cargo run --release --locked --example micro_spatial -- --controls` | Rerun W0's digest corruption against the byte-identical harness; reject, restore, pass. |
| W4 | `APEIRON-PERF-SCATTER-W4-T1` strict all-spatial-pair result | no edit; read existing `examples/reg1/` | Flash | strict full-profile command below | Every built-in workload/oracle/incumbent/selected-path/interval/single-failing-pair control must reject; any surviving control rejects the campaign. |

W0 freezes the harness and baseline before source changes. W1 lands only the allocation-free stable
scatter/assembly primitives and exact-output unit oracle. W2 starts from restored W1 and alone owns
the executor/refusal integration. W3 uses the byte-identical W0 harness. W4 starts only after W3.
No source, test or measurement wave overlaps another.

## Task blocks

### APEIRON-PERF-SCATTER-W0-T1 · Freeze the spatial micro baseline

- **Scope:** Run the existing controlled phase harness on the unchanged source and record raw
  partition, validation, reservation, schedule, kernel, assembly and end-to-end samples.
- **Files:** none; read `examples/micro_spatial.rs`
- **Depends on:** active source owner releases the frozen source identity
- **Grade:** measured baseline plus tested harness controls
- **Gate:**

```sh
cargo run --release --locked --example micro_spatial -- --controls
```

- **Rejecting control:** Corrupt one expected build or probe digest in a throwaway worktree. The same
  command must exit nonzero before timing; restore and rerun green.
- **Evidence to record:** source/harness/input SHA-256, host/compiler/governor/cache/thread conditions,
  raw samples, medians, phase shares and baseline/red/restored exits. Do not reuse the supplied
  46–55%/about-28% anchors as this baseline.

### APEIRON-PERF-SCATTER-W1-T1 · Stable chunk scatter primitives

- **Scope:** Add checked chunk planning, pre-sized private-buffer scatter and disjoint initialized
  destination assembly helpers. Concatenate shard runs by input-chunk number and expose exact
  `Parts`/loads to the executor. Keep `Parts::of`, `partition_chunk` and `is_intact` unchanged.
- **Files:** `src/table/partition.rs`
- **Depends on:** W0
- **Grade:** tested target; no extraction/proof claim
- **Gate:**

```sh
cargo test --locked --lib table::partition::tests
```

- **Rejecting control:** In a throwaway copy, reverse two nonempty adjacent chunk runs for one shard.
  The exact `buf`/`ends` serial oracle or strict-order integrity test must fail for that mutation;
  restore and rerun green. A test that compares only loads does not count.
- **Evidence to record:** source SHA-256; concrete `n + fixed*256p` scratch bound; empty, tail, skew,
  repeated-key and worker-count cases; test names/counts and baseline/red/restored exits.

### APEIRON-PERF-SCATTER-W2-T1 · Reserve-all parallel integration

- **Scope:** Make only the parallel build entry points execute W1's scatter and assembly in bounded
  sequential scopes. On the calling thread, complete checked planning and reserve/materialize all
  scratch, destination, result and table-shard storage before the first scope. Retain
  `Parts::is_intact`, unchanged loads/LPT bins, per-shard kernels, joins, query-order readback and
  serial fallback.
- **Files:** `src/table/parallel.rs`, `tests/parallel.rs`
- **Depends on:** W1
- **Grade:** tested target
- **Gate:**

```sh
cargo test --locked --test parallel
```

- **Rejecting control:** In a throwaway copy, assemble one shard by worker completion order; the
  existing non-commutative serial equivalence across worker counts must fail. Separately move the
  final-shard injected refusal after scatter worker start; the zero hold/clone/merge and no-worker
  observation must fail. Restore once and rerun the gate green.
- **Evidence to record:** source/test SHA-256, named worker counts, exact serial map and query-order
  results, F3 refusal counters, panic result, baseline/red/restored exits, and call-graph confirmation
  that worker bodies allocate nothing.

### APEIRON-PERF-SCATTER-W3-T1 · Focused post-change spatial measurement

- **Scope:** Rerun the unchanged W0 schedules and compare phase and end-to-end samples on the final
  source. Report validation separately; its retained cost is not attributed to scatter.
- **Files:** none; read `examples/micro_spatial.rs`
- **Depends on:** W2 and the full implementation barrier
- **Grade:** measured target
- **Gate:**

```sh
cargo run --release --locked --example micro_spatial -- --controls
```

- **Rejecting control:** Apply W0's digest mutation to a throwaway copy of the byte-identical harness;
  require red before timing, then restore and pass.
- **Evidence to record:** final source/harness/input/tool/machine identities, raw samples, medians and
  intervals/protocol for every phase and end-to-end before/after. This diagnostic cannot pass G1.

### APEIRON-PERF-SCATTER-W4-T1 · Strict all-spatial-pair result

- **Scope:** Run the complete unchanged registry and record every spatial pair, including
  concurrency/address-face outcomes and all named incumbents. Do not filter to improved or passing
  cells.
- **Files:** none; read existing `examples/reg1/`
- **Depends on:** W3
- **Grade:** measured target; acceptance only on command exit 0
- **Gate:**

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

- **Rejecting control:** Require every command-provided wrong-workload, oracle, omitted-incumbent,
  selected-implementation, interval-crossing and strict-single-failing-pair control to report red. A
  surviving control, missing pair or inconclusive interval rejects the run.
- **Evidence to record:** complete spatial pair table, intervals, full headline and exit, raw-log hash,
  machine/tool/source/harness/input identities. Any interval maximum above `1.0` keeps G1 open.

## Barrier checklist

- [ ] W1/W2 gates exit 0; every mutation is red and restored green.
- [ ] Exact `Parts` equality covers empty, skewed, repeated-key, chunk-tail and worker-count cases.
- [ ] Reserve-all refusal starts no scatter/build worker and returns no partial table or callback effect.
- [ ] `Parts::is_intact`, routing, LPT, worker budgets, kernels and query callback order are unchanged.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` and `cargo kani` pass.
- [ ] `proof/regen.sh --check` and bounded `lake build` in both proof packages pass on one identity;
  this freshness/regression evidence is not a scatter proof.
- [ ] Micro and strict outputs bind the same final source and unchanged harness identities.
- [ ] Every claim is labelled anchor, tested, measured, proved-at-canonical-scope or open.
- [ ] No acceptance bar, incumbent, pair, workload, interval, control or trust exclusion changed.

## Stop and rollback conditions

- Stop on serial `Parts` mismatch, same-key reorder, worker-count-dependent output, changed `ends` or
  LPT bins, callback/query reorder, allocation after worker start, partial refusal, worker-budget
  increase, validator bypass, unsafe write or surviving control.
- Stop and return to Astra/Sol if safe ownership of initialized disjoint destination slices requires
  raw pointers/`MaybeUninit`, or if the resource order cannot keep every allocation before worker
  start. Do not weaken initialization or refusal.
- After three materially different failed safe implementations, report the smallest ownership or
  reservation obstacle. Do not change layout, routing, kernels, validator or `<= 1.0`.
