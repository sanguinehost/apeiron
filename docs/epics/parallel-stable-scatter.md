# Parallel stable scatter

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

Merged epic record: `intent/parallel-stable-scatter.md`, `spec/parallel-stable-scatter.md`, `plan/parallel-stable-scatter.md` (one file per epic; see `docs/README.md`).

## Intent — Parallel stable scatter

- **Owner:** Astra architecture; Sol contract/attribution; Flash implementation
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for bounded implementation
- **Spec:** `docs/epics/parallel-stable-scatter.md`
- **Plan:** `docs/epics/parallel-stable-scatter.md`

### Problem

`Parts::of` serially counts, prefixes and scatters every input index before the existing scoped
per-shard build begins. The supplied diagnostic attributes about 28% of the spatial build to
partition plus `Parts::is_intact`, while the per-shard build/probe kernel accounts for 46–55% on its
respective runs. Those percentages are historical anchors, not a fresh measurement and not an
additive speedup forecast. Every strict spatial pair currently fails. The selected bounded change
parallelizes only stable index scatter and concatenation without changing routing, validation,
per-shard kernels, LPT assignment, grouped probes, readback, callbacks or refusal semantics.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Parallelize merge/application within one shard | It can reorder a logical key's non-commutative value stream and needs stronger merge laws than `build_eq_of_key_streams_eq`. |
| Concatenate chunk runs in worker-completion order | Completion order is nondeterministic and can change same-key value order. |
| Let workers grow private vectors | Allocation can fail after worker start and violates reserve-all transactional refusal. |
| Write one shared destination with raw pointers or `MaybeUninit` | It adds unsafe initialization, aliasing and drop obligations outside the selected safe boundary. |
| Remove or debug-gate `Parts::is_intact` | The validator remains the production tested guard until a source-linked constructor theorem replaces it. |
| Parallelize grouped readback or `sum_parallel` reduction in this slice | Query callback order, reduction order and overflow effects need a separate contract. |
| Relax the strict spatial threshold or reduce the matrix | It changes G1/REG-1 rather than improving the selected implementation. |

### Completion condition

The production parallel entry points use the stable scatter, focused tests and rejecting controls
pass, the phase split is remeasured, and the unchanged strict command records every spatial pair.
G1 closes only if that command exits 0 with every applicable pair at interval maximum `<= 1.0`.

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

### Invariants

1. For every shard, final `Parts::buf[ends[s]..ends[s+1]]` is exactly the serial `Parts::of` run:
   all and only routed global input indexes, strictly increasing.
2. Contiguous input chunks scatter into private pre-reserved buffers; final shard runs concatenate
   chunk runs in increasing input-chunk order, never worker-completion order.
3. Output `buf`, `ends`, loads and the resulting logical table are independent of worker count.
   For each normalized worker count, `Schedule::lpt` receives the unchanged loads and produces the
   same bins as before this optimization; different bin counts may assign shards differently.
4. All checked scratch, destination, schedule/result and shard-region reservation succeeds on the
   calling thread before the first scatter worker starts. Refusal starts no worker and returns no
   partial table.
5. Merge/application remains per shard and in input order. Grouped probe results and query callbacks
   remain in query order; the scatter invokes neither merge nor query callbacks.
6. `Parts::is_intact` remains the production tested validator. Routing, worker/crossover budgets,
   per-shard kernels, grouped probes, reservation errors and panic propagation retain their contracts.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| Parallel and serial `Parts` have byte-identical indexes and `ends` across empty, skewed, repeated-key and worker-count cases. | tested target | Focused unit tests in `src/table/partition.rs`. |
| Parallel build is logically identical to serial build for non-commutative merge across worker counts and preserves query callback order. | tested target | `cargo test --locked --test parallel`. |
| Injected final reservation failure starts no worker, invokes no hold/clone/merge/relocation and returns no table. | tested target | Existing F3 seam plus `tests/parallel/main.rs` refusal control. |
| Temporary scatter storage is bounded by `O(n + 256p)` words. | anchor, then inspected/tested | One private index slot per input plus fixed shard metadata per input chunk; no key/value copy. |
| Partition/validation and kernel phase costs are measured before and after. | measured target | `cargo run --release --locked --example micro_spatial -- --controls`. |
| Every spatial pair meets the unchanged strict interval ceiling. | measured target | Strict full-profile command; no aggregate or partial pass. |
| Scatter parity is attainable. | anchor/open | Unestablished; the bounded campaign makes no speedup promise. |

### Non-goals

- No change to table layout, routing, LPT policy, per-shard build/probe kernels, query-order readback,
  worker budget, crossover, `sum_parallel`, merge laws or public API.
- No removal of `Parts::is_intact`, unsafe writes, `MaybeUninit`, nested pool or extra timed-arm
  concurrency.
- No extraction or Lean proof claim for `Parts`, partition orchestration or `std::thread::scope` in
  this implementation slice.
- No threshold, incumbent, applicability, workload, interval, HashDOS, coverage or control change.

### Trust boundary

Rust tests establish selected finite executions, not the generic stable-partition theorem. The
scatter executor and `Parts::is_intact` remain unextracted/tested. Evidence excludes allocator and
global-allocator realization, scoped-thread memory/spawn/join/unwind/liveness, callbacks outside their
pure/total laws, rustc/LLVM, ABI, GPU/SPIR-V and the unsafe container. The sequential
`GrowableSchedule` refinement remains required; the Rust/canonical closed cone and Slang empty cone
remain separate.

### Open questions

- If the bounded scatter leaves any spatial pair red, Astra owns the next architecture decision; no
  threshold change, validator removal or grouped-reduction fallback is authorized here.
- A future proof task must decide whether to extract the allocation-free chunk scatter and assembly
  helpers. This implementation task does not change the current tested partition/parallel boundary.

## Specification — Parallel stable scatter

Kani retirement supersedes older model-checker requirements; see [the retirement contract](kani-retirement.md).

- **Owner:** Sol contract/attribution; Astra mechanism boundary
- **Intent:** `docs/epics/parallel-stable-scatter.md`
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for bounded implementation

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | Parallel construction of `Parts { buf, ends }` for `Table::{build_at_parallel,build_parallel}` |
| Domain | Every input slice `&[((Sigma,K),V)]`, admitted group width, stable equality-respecting `Home<K>`, and requested worker count normalized by the existing crossover/input/worker budget |
| Totality | Checked `Result` at the parallel-build boundary; scatter helpers execute only after successful checked planning and reservation |
| Preconditions | Checked chunk geometry and prefix sums; every scratch/destination/shard reservation succeeds; admitted pure/total routing law |
| Postcondition | Exactly the `Parts::of(home,input,_)` output, then the same loads and deterministic `Schedule::lpt` bins |

The serial `Parts::of` and sequential `Table::build_at` remain unchanged. The parallel entry points
select this constructor only above the existing crossover with at least two bounded tasks. Empty,
one-worker and below-crossover inputs retain the ordered serial path.

### Exact scatter contract

Let `n = input.len()` and let `p` be the normalized scatter task count. Split the input into `p`
contiguous, nonoverlapping chunks in increasing input order. The calling thread performs checked chunk
planning and counts each chunk's 256 routed loads. It derives each chunk's local shard prefixes,
global loads and final `ends` with checked addition.

Before the first `thread::scope`, the calling thread reserves and materializes:

- one initialized private `usize` index buffer of exact chunk length for every chunk;
- the initialized final `Parts::buf` of length `n` and fixed `ends` storage;
- fixed per-chunk shard metadata and all scoped handle/result containers;
- the unchanged LPT bins and every nonempty destination table shard region.

The selected parallel work is exact:

1. **Per-chunk stable scatter.** One scoped task per chunk revisits that chunk in increasing global
   input-index order and writes each index into its private pre-sized shard run. It allocates nothing.
2. **Stable concatenation.** After every scatter task joins, shard destination ranges are split into
   disjoint initialized mutable slices. Scoped tasks copy shard `s`'s local runs in increasing
   input-chunk number. They allocate nothing and never select completion order.

Counting, checked prefixes, reservation, `Parts::is_intact`, LPT construction, per-shard
build/application and query readback are not parallelized by this change. The two scatter scopes are
sequential; each has at most `p` tasks. There is no nested scope or increase above the registered
worker budget.

For chunk `c` and shard `s`, every index in its local run lies in chunk `c`, routes to `s`, and is
strictly increasing. Since chunk ranges are contiguous and ordered, every index in chunk `c` is less
than every index in chunk `c+1`. Concatenating local runs by `c` therefore yields exactly the serial
stable counting-sort run for `s`. Global prefix sums give the same `ends`; equal run lengths give the
same loads; deterministic `Schedule::lpt` therefore gives the same bins.

#### Temporary-memory bound

Beyond the final `Parts` output of `n + 257` `usize` words, scratch contains exactly one private index
slot per input plus a fixed number of 256/257-entry count, prefix and cursor arrays per chunk. Thus
scatter scratch is bounded by `O(n + 256p)` words, with no `K` or `V` clone and no per-element heap
allocation. The implementation must state its concrete fixed-array count at the definition and may
reduce this bound; it may not add another `O(np)` buffer. Shard regions are table output storage, not
scatter scratch, and retain their existing checked byte accounting.

### Concurrency, refusal and panic semantics

The order is:

```text
checked chunk/load plan -> reserve/materialize all scatter and table destinations
  -> scope scatter -> join all -> scope stable concatenation -> join all
  -> Parts::is_intact -> unchanged LPT per-shard build scope -> assemble table
```

`SizeOverflow` or `AllocationFailed` before the common reservation point starts no worker, performs no
hold/key/value/default clone, merge, relocation or publication, and returns no table. A malformed
scatter rejected by `Parts::is_intact` also returns no table and starts no per-shard build worker.
Workers perform no fallible allocation. Every started task is joined; the first panic is propagated
only after all sibling tasks join, and a panic cannot return a partial table.

Routing may execute concurrently only under the existing admitted `Home` law: total, pure, stable and
equality-respecting. The scatter invokes no merge or query callback. Later `Merge` invocations retain
each logical key's input order; `get_each*` invokes its callback once in original query order. No
callback whose order is part of the table contract is reordered. Behavior of effectful, allocating,
panicking or divergent routing callbacks remains outside the admitted domain.

`std::thread::scope`, exclusive-borrow realization, spawn/join/unwind and liveness remain the
unextracted executor boundary. Safe disjoint initialized slices are mandatory; raw concurrent writes,
`MaybeUninit` and unsafe slice construction are outside this contract.

### Determinism and validation

For all admitted `workers_a` and `workers_b`, successful scatter output has equal `buf` and `ends`.
Consequently each shard sees the same sequence, LPT receives the same load vector, the same bins are
constructed for the same normalized bin count, and the resulting table equals the serial ordered
build for arbitrary non-commutative merge satisfying the existing callback laws. Worker completion
order cannot affect any output.

`Parts::is_intact` remains the production fail-closed validator with its existing endpoint,
cardinality, strict local order, bounds and fresh-routing checks. It remains `tested`, not extracted.
This constructor contract does not replace, weaken, remove or debug-gate it.

### Representation relation (Rust to Lean)

| Layer | Symbol | Path and consequence |
|---|---|---|
| Serial reference | `Parts::of`, `partition_chunk` | `src/table/partition.rs`; exact output reference retained |
| Parallel scatter | allocation-free chunk scatter and stable assembly helpers | `src/table/partition.rs`; production-called by `src/table/parallel.rs` |
| Executor | `Table::{build_at_parallel,build_parallel}` | `src/table/parallel.rs`; `thread::scope` remains unextracted |
| Runtime guard | `Parts::is_intact` | `src/table/partition.rs`; tested only and deliberately absent from extraction |
| Physical schedule obligation | `GrowableSchedule.partition_preserves_key_streams` | implementation-specific theorem remains open |
| Canonical authority | `Apeiron.Table.build_eq_of_key_streams_eq` | pinned `Apeiron/Table/Basic.lean:167-174`; already proved at its canonical scope |

The canonical theorem says that equal per-key value subsequences give equal builds. It does not prove
this scatter, Rust threads, allocation or slice ownership. The required future link is: exact `Parts`
equality implies the full logical key `(Sigma,K)` receives the identical value subsequence, then
`GrowableSchedule.partition_preserves_key_streams` composes that fact with
`build_eq_of_key_streams_eq`.

This implementation does not change extraction. `Parts`, partition orchestration, reserve-all and
parallel execution remain in the tested/unextracted boundary. Existing `GrowableGeometry`,
`Directory`, `Identity`, `Scan`, `Search`, `Relocation`, `Representation`, `Resources`, `Rebuild`,
`Refinement`, `Witnesses` and `Batch` statements and grades do not change. A later extraction task
may root allocation-free sequential helpers only through wrappers that call the production bodies;
until then no scatter headline is `proved`.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `scatter_eq_serial` | Parallel `buf` and `ends` equal `Parts::of` for every admitted input and worker count. | tested target; proved remains open | Partition unit oracle; future `GrowableSchedule` theorem. |
| `scatter_stable` | Every shard run is strictly increasing in global input index. | tested target | Empty, skew, repeated-key and chunk-boundary unit tests. |
| `scatter_worker_independent` | Output does not depend on task count or completion order. | tested target | Serial equality across `1,2,3,7,16,64,300` workers where applicable. |
| `scatter_reserved` | Workers write only initialized pre-sized private or disjoint destination slices and allocate nothing. | tested/inspection target | Reservation-failure seam and source call-graph inspection. |
| `parallel_build_eq_serial` | Logical table and ordered query readback equal the serial path under non-commutative merge. | tested target | `tests/parallel/main.rs`. |
| `scatter_schedule_link` | Exact per-key subsequences compose with canonical ordered build. | proved target, future task | `GrowableSchedule.partition_preserves_key_streams` plus canonical theorem. |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Checked size/prefix overflow | `Err(SizeOverflow)`, no table, no worker | checked plan |
| Scratch, handle/result or shard reservation failure | `Err(AllocationFailed)`, no table, no worker/callback | reserve-all |
| Malformed assembled `Parts` | `Err(SizeOverflow)`, no table and no build worker | `!Parts::is_intact` |
| Scatter/build worker panic | join every started sibling, then resume unwind; no table | scoped panic boundary |
| Successful scatter | exact serial `Parts`, then unchanged build | contracts above |

### Evidence plan

1. **Partition unit tests.** Compare exact `buf` and `ends` with serial `Parts::of` for empty input,
   one chunk, uneven chunk tails, all-one-shard skew, all-shard spread, repeated logical keys and
   worker counts above input/shard count. Keep every existing malformed-output validator test.
2. **Rejecting partition controls.** In a throwaway copy, reverse two chunk runs for one shard and
   separately duplicate/omit one index. Exact serial equality or `Parts::is_intact` must fail; restore
   and rerun green.
3. **Parallel equivalence.** `tests/parallel/main.rs` retains deliberately non-commutative Horner merge,
   worker-count coverage, skew, `Sigma` qualification, crossover execution, query-order readback,
   panic propagation and final-shard injected reservation refusal. These are the rejecting controls:
   completion-order concatenation, same-key reversal, query-order reversal or worker-start-before-
   reserve must make a named test fail.
4. **Micro spatial split.** Run
   `cargo run --release --locked --example micro_spatial -- --controls` before and after on the same
   machine/protocol. Record raw samples and partition, validate, reserve, schedule, kernel, assembly
   medians. The supplied 46–55% kernel and about 28% partition-plus-validate figures are anchors until
   rerun and do not predict parity.
5. **Strict spatial pair gate.** Run
   `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`.
   Record every spatial pair, including concurrency/address-face rejection and every named incumbent.
   Missing, invalid, inconclusive or interval maximum above `1.0` stays red. No micro result, aggregate
   or serial improvement closes G1.
6. **Integration barrier.** After source work, run the complete Rust/extraction/both-Lean-package
   barrier and existing semantic controls on one source/tool identity. Unchanged extraction is still
   required to be fresh; it does not prove the scatter.

### Trust boundary

The contract and lifecycle artifacts are `anchor`. Focused Rust results may become `tested`; runtime
numbers may become `measured`. No scatter result is `proved` without real-source extraction and the
audited `GrowableSchedule` link. Allocator/global allocator, thread memory and executor realization,
spawn/join/unwind/liveness, callbacks outside admitted laws, rustc/LLVM, ABI, GPU/SPIR-V and unsafe
container realization remain excluded. These exclusions do not permit partial output, reordering or
unchecked writes.

### Open questions

- Future Sol work owns the exact source-linked `GrowableSchedule` theorem and joint Stored/Regen
  witnesses. It is not required to begin this tested performance experiment and remains mandatory for
  end-to-end acceptance.
- Any proposal to remove `Parts::is_intact`, cache routes, change chunk geometry or parallelize
  grouped reduction requires a separate contract and rejecting control.

## Plan — Parallel stable scatter

Kani retirement supersedes older model-checker requirements; see [the retirement contract](kani-retirement.md).

- **Owner:** Astra architecture; Sol contract/attribution; Flash implementation
- **Intent:** `docs/epics/parallel-stable-scatter.md`
- **Spec:** `docs/epics/parallel-stable-scatter.md`
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for serialized dispatch

### Arc block

```text
SCOPE:      stable scatter source/test work and matched spatial measurement; complete when focused gates/controls pass and every strict spatial pair is recorded without weakening G1
REGISTER:   state/test/measurement with an open refinement consequence; excludes layout, routing, kernels, grouped reduction, validator removal, proof implementation and W7 composition
CORPUS:     Apeiron/Table/Basic.lean `build_eq_of_key_streams_eq` (proved canonical authority); GrowableSchedule.partition_preserves_key_streams remains open
REFERENCE:  /home/socol/Workspace/apeiron/src/table/partition.rs, src/table/parallel.rs, tests/parallel/main.rs and examples/micro_spatial.rs read from the supplied checkout
GATE:       focused commands below; strict full-profile spatial baseline is red and no task may reduce pairs, incumbents, controls or the interval ceiling
GRADE:      lifecycle documents anchor; source targets tested; runtime targets measured; scatter refinement open
```

The documentation task is the supplied-checkout exception and owns four documentation files. Every
implementation task below owns at most three files and one gate. Use an isolated
`worktree-agent-APEIRON-PERF-SCATTER-<id>` at the recorded integration revision. Serialize every wave;
`target/`, `proof/ApeironWord.lean`, `proof/.lake/` and `proof/corpus/.lake/` remain shared lanes.

### Goal and completion condition

The parallel build reserves all storage before worker start, constructs byte-identical serial
`Parts` through chunk-private stable scatter and chunk-order concatenation, retains validation and
ordered callbacks, and records focused plus strict spatial results. G1 remains open unless the strict
full-profile command exits 0 for every applicable pair.

```sh
cargo test --workspace --all-features --locked && cargo clippy --workspace --all-features --all-targets --locked -- -D warnings && cargo fmt --all -- --check && proof/regen.sh --check
```

Run bounded `lake build` separately in `proof/` and `proof/corpus/` under the repository lane, memory,
swap, timeout and thread rules. They are integration barriers, not scatter-proof claims.

### Non-goals

- No extraction/proof edit, validator removal, unsafe storage, callback-order change, grouped
  reduction, routing/LPT/kernel/layout/API change or second representation.
- No strict threshold, incumbent, applicability, workload, interval, control, HashDOS or coverage
  change.

### Wave map

| Wave | Task | Files | Owner | Gate | Rejecting control |
|---|---|---|---|---|---|
| W0 | `APEIRON-PERF-SCATTER-W0-T1` freeze micro baseline | no edit; read `examples/micro_spatial.rs` | Flash | `cargo run --release --locked --example micro_spatial -- --controls` | Corrupt one expected build/probe digest in a throwaway copy; the gate rejects before timing, then restored gate passes. |
| W1 | `APEIRON-PERF-SCATTER-W1-T1` stable chunk scatter primitives | `src/table/partition.rs` | Flash | `cargo test --locked --lib table::partition::tests` | Reverse two adjacent chunk runs for one shard; exact serial equality or integrity test fails, then restored gate passes. |
| W2 | `APEIRON-PERF-SCATTER-W2-T1` reserve-all parallel integration | `src/table/parallel.rs`, `tests/parallel/main.rs` | Flash | `cargo test --locked --test parallel` | Select completion order or start scatter before final-shard reservation; non-commutative equivalence or zero-effect refusal fails, then restored gate passes. |
| W3 | `APEIRON-PERF-SCATTER-W3-T1` focused post-change spatial measurement | no edit; read `examples/micro_spatial.rs` | Flash | `cargo run --release --locked --example micro_spatial -- --controls` | Rerun W0's digest corruption against the byte-identical harness; reject, restore, pass. |
| W4 | `APEIRON-PERF-SCATTER-W4-T1` strict all-spatial-pair result | no edit; read existing `examples/reg1/` | Flash | strict full-profile command below | Every built-in workload/oracle/incumbent/selected-path/interval/single-failing-pair control must reject; any surviving control rejects the campaign. |

W0 freezes the harness and baseline before source changes. W1 lands only the allocation-free stable
scatter/assembly primitives and exact-output unit oracle. W2 starts from restored W1 and alone owns
the executor/refusal integration. W3 uses the byte-identical W0 harness. W4 starts only after W3.
No source, test or measurement wave overlaps another.

### Task blocks

#### APEIRON-PERF-SCATTER-W0-T1 · Freeze the spatial micro baseline

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

#### APEIRON-PERF-SCATTER-W1-T1 · Stable chunk scatter primitives

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

#### APEIRON-PERF-SCATTER-W2-T1 · Reserve-all parallel integration

- **Scope:** Make only the parallel build entry points execute W1's scatter and assembly in bounded
  sequential scopes. On the calling thread, complete checked planning and reserve/materialize all
  scratch, destination, result and table-shard storage before the first scope. Retain
  `Parts::is_intact`, unchanged loads/LPT bins, per-shard kernels, joins, query-order readback and
  serial fallback.
- **Files:** `src/table/parallel.rs`, `tests/parallel/main.rs`
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

#### APEIRON-PERF-SCATTER-W3-T1 · Focused post-change spatial measurement

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

#### APEIRON-PERF-SCATTER-W4-T1 · Strict all-spatial-pair result

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

### Barrier checklist

- [ ] W1/W2 gates exit 0; every mutation is red and restored green.
- [ ] Exact `Parts` equality covers empty, skewed, repeated-key, chunk-tail and worker-count cases.
- [ ] Reserve-all refusal starts no scatter/build worker and returns no partial table or callback effect.
- [ ] `Parts::is_intact`, routing, LPT, worker budgets, kernels and query callback order are unchanged.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` pass.
- [ ] `proof/regen.sh --check` and bounded `lake build` in both proof packages pass on one identity;
  this freshness/regression evidence is not a scatter proof.
- [ ] Micro and strict outputs bind the same final source and unchanged harness identities.
- [ ] Every claim is labelled anchor, tested, measured, proved-at-canonical-scope or open.
- [ ] No acceptance bar, incumbent, pair, workload, interval, control or trust exclusion changed.

### Stop and rollback conditions

- Stop on serial `Parts` mismatch, same-key reorder, worker-count-dependent output, changed `ends` or
  LPT bins, callback/query reorder, allocation after worker start, partial refusal, worker-budget
  increase, validator bypass, unsafe write or surviving control.
- Stop and return to Astra/Sol if safe ownership of initialized disjoint destination slices requires
  raw pointers/`MaybeUninit`, or if the resource order cannot keep every allocation before worker
  start. Do not weaken initialization or refusal.
- After three materially different failed safe implementations, report the smallest ownership or
  reservation obstacle. Do not change layout, routing, kernels, validator or `<= 1.0`.
