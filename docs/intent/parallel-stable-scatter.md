# Intent — Parallel stable scatter

- **Owner:** Astra architecture; Sol contract/attribution; Flash implementation
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for bounded implementation
- **Spec:** `docs/spec/parallel-stable-scatter.md`
- **Plan:** `docs/plan/parallel-stable-scatter.md`

## Problem

`Parts::of` serially counts, prefixes and scatters every input index before the existing scoped
per-shard build begins. The supplied diagnostic attributes about 28% of the spatial build to
partition plus `Parts::is_intact`, while the per-shard build/probe kernel accounts for 46–55% on its
respective runs. Those percentages are historical anchors, not a fresh measurement and not an
additive speedup forecast. Every strict spatial pair currently fails. The selected bounded change
parallelizes only stable index scatter and concatenation without changing routing, validation,
per-shard kernels, LPT assignment, grouped probes, readback, callbacks or refusal semantics.

## Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Parallelize merge/application within one shard | It can reorder a logical key's non-commutative value stream and needs stronger merge laws than `build_eq_of_key_streams_eq`. |
| Concatenate chunk runs in worker-completion order | Completion order is nondeterministic and can change same-key value order. |
| Let workers grow private vectors | Allocation can fail after worker start and violates reserve-all transactional refusal. |
| Write one shared destination with raw pointers or `MaybeUninit` | It adds unsafe initialization, aliasing and drop obligations outside the selected safe boundary. |
| Remove or debug-gate `Parts::is_intact` | The validator remains the production tested guard until a source-linked constructor theorem replaces it. |
| Parallelize grouped readback or `sum_parallel` reduction in this slice | Query callback order, reduction order and overflow effects need a separate contract. |
| Relax the strict spatial threshold or reduce the matrix | It changes G1/REG-1 rather than improving the selected implementation. |

## Completion condition

The production parallel entry points use the stable scatter, focused tests and rejecting controls
pass, the phase split is remeasured, and the unchanged strict command records every spatial pair.
G1 closes only if that command exits 0 with every applicable pair at interval maximum `<= 1.0`.

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

## Invariants

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

## Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| Parallel and serial `Parts` have byte-identical indexes and `ends` across empty, skewed, repeated-key and worker-count cases. | tested target | Focused unit tests in `src/table/partition.rs`. |
| Parallel build is logically identical to serial build for non-commutative merge across worker counts and preserves query callback order. | tested target | `cargo test --locked --test parallel`. |
| Injected final reservation failure starts no worker, invokes no hold/clone/merge/relocation and returns no table. | tested target | Existing F3 seam plus `tests/parallel.rs` refusal control. |
| Temporary scatter storage is bounded by `O(n + 256p)` words. | anchor, then inspected/tested | One private index slot per input plus fixed shard metadata per input chunk; no key/value copy. |
| Partition/validation and kernel phase costs are measured before and after. | measured target | `cargo run --release --locked --example micro_spatial -- --controls`. |
| Every spatial pair meets the unchanged strict interval ceiling. | measured target | Strict full-profile command; no aggregate or partial pass. |
| Scatter parity is attainable. | anchor/open | Unestablished; the bounded campaign makes no speedup promise. |

## Non-goals

- No change to table layout, routing, LPT policy, per-shard build/probe kernels, query-order readback,
  worker budget, crossover, `sum_parallel`, merge laws or public API.
- No removal of `Parts::is_intact`, unsafe writes, `MaybeUninit`, nested pool or extra timed-arm
  concurrency.
- No extraction or Lean proof claim for `Parts`, partition orchestration or `std::thread::scope` in
  this implementation slice.
- No threshold, incumbent, applicability, workload, interval, HashDOS, coverage or control change.

## Trust boundary

Rust tests establish selected finite executions, not the generic stable-partition theorem. The
scatter executor and `Parts::is_intact` remain unextracted/tested. Evidence excludes allocator and
global-allocator realization, scoped-thread memory/spawn/join/unwind/liveness, callbacks outside their
pure/total laws, rustc/LLVM, ABI, GPU/SPIR-V and the unsafe container. The sequential
`GrowableSchedule` refinement remains required; the Rust/canonical closed cone and Slang empty cone
remain separate.

## Open questions

- If the bounded scatter leaves any spatial pair red, Astra owns the next architecture decision; no
  threshold change, validator removal or grouped-reduction fallback is authorized here.
- A future proof task must decide whether to extract the allocation-free chunk scatter and assembly
  helpers. This implementation task does not change the current tested partition/parallel boundary.
