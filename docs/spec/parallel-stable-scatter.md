# Specification — Parallel stable scatter

- **Owner:** Sol contract/attribution; Astra mechanism boundary
- **Intent:** `docs/intent/parallel-stable-scatter.md`
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for bounded implementation

## Scope and admissible domain

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

## Exact scatter contract

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

### Temporary-memory bound

Beyond the final `Parts` output of `n + 257` `usize` words, scratch contains exactly one private index
slot per input plus a fixed number of 256/257-entry count, prefix and cursor arrays per chunk. Thus
scatter scratch is bounded by `O(n + 256p)` words, with no `K` or `V` clone and no per-element heap
allocation. The implementation must state its concrete fixed-array count at the definition and may
reduce this bound; it may not add another `O(np)` buffer. Shard regions are table output storage, not
scatter scratch, and retain their existing checked byte accounting.

## Concurrency, refusal and panic semantics

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

## Determinism and validation

For all admitted `workers_a` and `workers_b`, successful scatter output has equal `buf` and `ends`.
Consequently each shard sees the same sequence, LPT receives the same load vector, the same bins are
constructed for the same normalized bin count, and the resulting table equals the serial ordered
build for arbitrary non-commutative merge satisfying the existing callback laws. Worker completion
order cannot affect any output.

`Parts::is_intact` remains the production fail-closed validator with its existing endpoint,
cardinality, strict local order, bounds and fresh-routing checks. It remains `tested`, not extracted.
This constructor contract does not replace, weaken, remove or debug-gate it.

## Representation relation (Rust to Lean)

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

## Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `scatter_eq_serial` | Parallel `buf` and `ends` equal `Parts::of` for every admitted input and worker count. | tested target; proved remains open | Partition unit oracle; future `GrowableSchedule` theorem. |
| `scatter_stable` | Every shard run is strictly increasing in global input index. | tested target | Empty, skew, repeated-key and chunk-boundary unit tests. |
| `scatter_worker_independent` | Output does not depend on task count or completion order. | tested target | Serial equality across `1,2,3,7,16,64,300` workers where applicable. |
| `scatter_reserved` | Workers write only initialized pre-sized private or disjoint destination slices and allocate nothing. | tested/inspection target | Reservation-failure seam and source call-graph inspection. |
| `parallel_build_eq_serial` | Logical table and ordered query readback equal the serial path under non-commutative merge. | tested target | `tests/parallel.rs`. |
| `scatter_schedule_link` | Exact per-key subsequences compose with canonical ordered build. | proved target, future task | `GrowableSchedule.partition_preserves_key_streams` plus canonical theorem. |

## Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Checked size/prefix overflow | `Err(SizeOverflow)`, no table, no worker | checked plan |
| Scratch, handle/result or shard reservation failure | `Err(AllocationFailed)`, no table, no worker/callback | reserve-all |
| Malformed assembled `Parts` | `Err(SizeOverflow)`, no table and no build worker | `!Parts::is_intact` |
| Scatter/build worker panic | join every started sibling, then resume unwind; no table | scoped panic boundary |
| Successful scatter | exact serial `Parts`, then unchanged build | contracts above |

## Evidence plan

1. **Partition unit tests.** Compare exact `buf` and `ends` with serial `Parts::of` for empty input,
   one chunk, uneven chunk tails, all-one-shard skew, all-shard spread, repeated logical keys and
   worker counts above input/shard count. Keep every existing malformed-output validator test.
2. **Rejecting partition controls.** In a throwaway copy, reverse two chunk runs for one shard and
   separately duplicate/omit one index. Exact serial equality or `Parts::is_intact` must fail; restore
   and rerun green.
3. **Parallel equivalence.** `tests/parallel.rs` retains deliberately non-commutative Horner merge,
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
6. **Integration barrier.** After source work, run the complete Rust/Kani/extraction/both-Lean-package
   barrier and existing semantic controls on one source/tool identity. Unchanged extraction is still
   required to be fresh; it does not prove the scatter.

## Trust boundary

The contract and lifecycle artifacts are `anchor`. Focused Rust results may become `tested`; runtime
numbers may become `measured`. No scatter result is `proved` without real-source extraction and the
audited `GrowableSchedule` link. Allocator/global allocator, thread memory and executor realization,
spawn/join/unwind/liveness, callbacks outside admitted laws, rustc/LLVM, ABI, GPU/SPIR-V and unsafe
container realization remain excluded. These exclusions do not permit partial output, reordering or
unchecked writes.

## Open questions

- Future Sol work owns the exact source-linked `GrowableSchedule` theorem and joint Stored/Regen
  witnesses. It is not required to begin this tested performance experiment and remains mandatory for
  end-to-end acceptance.
- Any proposal to remove `Parts::is_intact`, cache routes, change chunk geometry or parallelize
  grouped reduction requires a separate contract and rejecting control.
