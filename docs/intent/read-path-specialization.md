# Intent — Growable table read-path specialization

- **Owner:** Astra architecture; Flash implementation; Sol bridge repair
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for bounded implementation
- **Spec:** `docs/spec/read-path-specialization.md`
- **Plan:** `docs/plan/read-path-specialization.md`

## Problem

The production `Table::get_at` uses the insertion classifier
`search_decision_at -> search_decision -> spill_locate` and then `apply_get`. A summary-negative spill
read therefore still searches for the first vacancy even though `SpillSummaryMissSound` already
certifies that the routed spill contains no matching key. Resident reads also construct and apply a
general insertion decision instead of classifying and indexing the resident slot once. The recorded
strict receipt remains red: `pass=421 fail=1029`; selected spill misses cost about 16–19 ns/op,
spill hits about 10 ns/op, and the named resident miss remains 2.645 ns/op. These are measurements of
the receipt's source identity, not forecasts for this specialization.

## Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Relax `selected/incumbent <= 1.0` | It changes G1/REG-1 instead of improving the selected implementation. |
| Dispatch on `RawStrataHome`, a benchmark key population, or operation label | It creates benchmark-specific behavior and does not preserve the generic table contract. |
| Add a second read container or cache | It creates a second representation and a new coherence/refinement boundary. |
| Return absent at the first spill vacancy on a summary-positive query | A later matching key may exist; this permits a false negative. |
| Treat a summary or fingerprint hit as presence | Both are filters. Collisions require full `(Sigma,K)` identity. |
| Cache summary width, change flat geometry, or add a hot directory in this slice | Each adds constructor and preservation invariants requiring a separate Astra/Sol decision. |
| Remove payload initialization or weaken transactional refusal | Neither is read-path work; both retain their existing resource and proof obligations. |

## Completion condition

The bounded mechanism is implemented on the production path, its Rust and extracted-read contracts
pass with rejecting controls, and the strict full-profile command records every affected pair under
the unchanged interval ceiling. Global G1 closes only when that same command exits 0 for every
applicable pair; a scoped read improvement does not convert unrelated red cells into passes.

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

## Invariants

1. `get_at(sigma,key)` is total on every admitted table/key input and returns the unique stored value
   for `(sigma,key)`, or the table default when absent; `get` is the default-`Sigma` specialization.
2. Full `(Sigma,K)` equality decides identity. Address, direct lane, summary and fingerprint are
   placement or exclusion filters only.
3. Absence is justified by exactly one sound certificate: direct empty under composite
   `DirectCover`; checked direct exclusion plus a routed-spill summary miss; or complete finite match
   coverage. A false negative is forbidden. A summary/fingerprint false positive may cost a bounded
   sweep and full-key comparisons.
4. A summary-positive spill miss retains complete match coverage, including a match after an earlier
   prefix vacancy. Read-only code performs no first-vacancy bookkeeping.
5. Reads clone, merge and allocate nothing, mutate no state, and introduce no `Result` refusal.
   `SizeOverflow`/`AllocationFailed` remain write/build outcomes; `ShardFull` remains compatibility
   only.
6. Insert search keeps its five `SearchDecision` cases, exact first fit, growth, and refusal/resource
   behavior. The specialization creates no second container or proof-only implementation.

## Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| Direct, spill, absent and unequal-`Sigma` reads retain exact values for both carriers. | tested target | `cargo test --locked --lib --test table` |
| Summary-negative reads perform no vacancy scan; summary-positive late hits remain visible. | tested target | Focused real-path trace tests and red/restore mutations in `src/table.rs`/`src/table/state.rs`. |
| The generated Stored/Regen read bodies denote canonical `Apeiron.Table.get`. | proved target | Bounded `lake build Growable.Scan Growable.Search AxiomAudit` after `proof/regen.sh --check`. |
| The focused micro schedules change in the intended direction. | measured target | `cargo run --release --locked --example micro_spill -- --controls`, same machine/input/sample protocol before and after. |
| Every affected REG-1 pair satisfies the existing per-pair decision. | measured target | Strict full-profile command; interval maximum must be `<= 1.0`. |
| Parity is attainable by this mechanism. | anchor/open | Unestablished; the implementation campaign tests the mechanism and makes no speedup promise. |

## Non-goals

- No threshold relaxation, historical envelope, aggregate pass, std-only pass, or reduced matrix.
- No benchmark-specific dispatch, second container, changed identity, or changed spill fallback.
- No summary-width cache, flat-geometry change, new control directory, payload-layout replacement,
  first-empty termination, SIMD/GPU/ISA specialization, or spatial-scatter implementation.
- No reopening or discharge of `W7-SearchCommitBoundary` from performance evidence.

## Trust boundary

Rust tests and measurements do not prove generic search correctness. The bridge remains conditional
on reachable `RepR`, carrier/address laws, composite `DirectCover`, spill-summary shape/recordedness
and the source-linked three-way absence certificate. Evidence excludes allocator/global-allocator
realization, callbacks outside their laws, executor/`std::thread::scope`, rustc/LLVM codegen, the
kernel ABI, GPU/SPIR-V and the unsafe container. The Rust/canonical closed cone and Slang empty cone
remain separate.

## Open questions

- If every affected pair remains red after the bounded campaign, Astra owns the decision whether to
  design a compact hot lookup directory/control lane in the same `Table`; no fallback design is
  authorized here.
- Aeneas translatability of the production-specialized read body is unestablished until the
  regeneration task runs; failure returns to Sol/Astra and does not authorize a model body.
