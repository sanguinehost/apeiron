# Growable table read-path specialization

Merged epic record: `intent/read-path-specialization.md`, `spec/read-path-specialization.md`, `plan/read-path-specialization.md` (one file per epic; see `docs/README.md`).

## Intent — Growable table read-path specialization

- **Owner:** Astra architecture; Flash implementation; Sol bridge repair
- **Source baseline:** historical dispatch revision `c450c222905c050f793a23332708c230c658f82c`; current evidence is digest-bound in [the retirement receipt](../evidence/kani-retirement.md)
- **Status:** source specialization and real-source extraction landed; universal read/search refinement and allocator realization remain open under [the next bounded plan](read-proof.md)
- **Spec:** `docs/epics/read-path-specialization.md`
- **Plan:** `docs/epics/read-path-specialization.md`

### Problem

At the recorded dispatch baseline, production `Table::get_at` used the insertion classifier
`search_decision_at -> search_decision -> spill_locate` and then `apply_get`. A summary-negative spill
read therefore still searched for the first vacancy even though `SpillSummaryMissSound` already
certified that the routed spill contained no matching key. Resident reads also constructed and applied a
general insertion decision instead of classifying and indexing the resident slot once. The recorded
strict receipt remains red: `pass=421 fail=1029`; selected spill misses cost about 16–19 ns/op,
spill hits about 10 ns/op, and the named resident miss remains 2.645 ns/op. These are measurements of
the receipt's source identity, not forecasts for this specialization.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Relax `selected/incumbent <= 1.0` | It changes G1/REG-1 instead of improving the selected implementation. |
| Dispatch on `RawStrataHome`, a benchmark key population, or operation label | It creates benchmark-specific behavior and does not preserve the generic table contract. |
| Add a second read container or cache | It creates a second representation and a new coherence/refinement boundary. |
| Return absent at the first spill vacancy on a summary-positive query | A later matching key may exist; this permits a false negative. |
| Treat a summary or fingerprint hit as presence | Both are filters. Collisions require full `(Sigma,K)` identity. |
| Cache summary width, change flat geometry, or add a hot directory in this slice | Each adds constructor and preservation invariants requiring a separate Astra/Sol decision. |
| Remove payload initialization or weaken transactional refusal | Neither is read-path work; both retain their existing resource and proof obligations. |

### Recorded completion condition

The bounded mechanism is implemented on the production path and covered by ordinary Rust tests and
real-source mutation controls. Fresh extraction and corpus builds are recorded in
[the retirement receipt](../evidence/kani-retirement.md); they do not prove the still-open universal
read/search contracts. The strict full-profile command must record every affected pair under the
unchanged interval ceiling. Global G1 closes only when that same command exits 0 for every applicable
pair; a scoped read improvement does not convert unrelated red cells into passes.

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

### Invariants

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

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| Direct, spill, absent and unequal-`Sigma` reads retain exact values for both carriers. | tested target | `cargo test --locked --lib --test table` |
| Summary-negative reads perform no vacancy scan; summary-positive late hits remain visible. | tested target | Focused real-path trace tests and red/restore mutations in `src/table.rs`/`src/table/state.rs`. |
| The generated Stored/Regen read bodies denote canonical `Apeiron.Table.get`. | proved target | Bounded `lake build Growable.Scan Growable.Search AxiomAudit` after `proof/regen.sh --check`. |
| The focused micro schedules change in the intended direction. | measured target | `cargo run --release --locked --example micro_spill -- --controls`, same machine/input/sample protocol before and after. |
| Every affected REG-1 pair satisfies the existing per-pair decision. | measured target | Strict full-profile command; interval maximum must be `<= 1.0`. |
| Parity is attainable by this mechanism. | anchor/open | Unestablished; the implementation campaign tests the mechanism and makes no speedup promise. |

### Non-goals

- No threshold relaxation, historical envelope, aggregate pass, std-only pass, or reduced matrix.
- No benchmark-specific dispatch, second container, changed identity, or changed spill fallback.
- No summary-width cache, flat-geometry change, new control directory, payload-layout replacement,
  first-empty termination, SIMD/GPU/ISA specialization, or spatial-scatter implementation.
- No reopening or discharge of `W7-SearchCommitBoundary` from performance evidence.

### Trust boundary

Rust tests and measurements do not prove generic search correctness. The bridge remains conditional
on reachable `RepR`, carrier/address laws, composite `DirectCover`, spill-summary shape/recordedness
and the source-linked three-way absence certificate. Evidence excludes allocator/global-allocator
realization, callbacks outside their laws, executor/`std::thread::scope`, rustc/LLVM codegen, the
kernel ABI, GPU/SPIR-V and the unsafe container. The Rust/canonical closed cone and Slang empty cone
remain separate.

### Open questions

- If every affected pair remains red after the bounded campaign, Astra owns the decision whether to
  design a compact hot lookup directory/control lane in the same `Table`; no fallback design is
  authorized here.
- Aeneas translatability of the production-specialized read body is unestablished until the
  regeneration task runs; failure returns to Sol/Astra and does not authorize a model body.

## Specification — Growable table read-path specialization

- **Owner:** Sol contract/attribution; Astra mechanism boundary
- **Intent:** `docs/epics/read-path-specialization.md`
- **Source baseline:** historical dispatch revision `c450c222905c050f793a23332708c230c658f82c`; current evidence is digest-bound in [the retirement receipt](../evidence/kani-retirement.md)
- **Status:** executing read shape extracted and regression-tested; `SearchDecisionCorrect`, universal read/search links and the allocator adapter remain open under [the next bounded plan](read-proof.md)

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | `Table::{get_at,get}` and their production-called resident/spill read helpers |
| Domain | Every reachable `Table<K,V,H,N,R>` satisfying `RepR`, for both `Stored<K>` and admitted exact-codec `Regen<C>` carriers; every `Sigma` and `K`; admissible `N`/geometry and callback/address laws are unchanged |
| Totality | Total borrowed read `&V`; no `Option` or `Result` in the public read API |
| Preconditions | Reachable representation, full-key equality laws, stable equality-respecting address/carrier laws, occupancy authority, composite `DirectCover`, and spill-summary shape/recordedness when summary exclusion is used |
| Postcondition | Return `Apeiron.Table.get h (sigma,key)`: the unique represented value, or the table default exactly when `(sigma,key)` is absent |

`get(key)` is exactly `get_at(Sigma::default(),key)`. A read has no resource-refusal branch.
`SizeOverflow` and `AllocationFailed` remain transactional outcomes of insertion/build preparation;
`ShardFull` remains a compatibility variant not returned by the growable implementation. The read
domain is not narrowed to direct coordinates, word keys, benchmark populations, noncolliding tags,
summary negatives, or a fixed occupancy.

### Representation relation (Rust to Lean)

The represented logical key is `(Sigma,K)`. `RepR t h` composes unique flat/spill representation,
decoded carrier observations, occupancy authority, address/directory consistency, length and
composite `DirectCover`. The specialization changes read control flow, not this denotation.

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `Table::{get_at,get}` and the shared read-only spill match helper | `src/table.rs`, `src/table/state.rs` |
| Monomorphic wrappers | Stored/Regen public-read wrappers that only call the production body | `proof/extract/lib.rs` |
| Extracted bodies | Stored/Regen wrappers, `table.Table.get_at`, resident flat indexing, summary query, fingerprint scan and read-only spill matching | `proof/ApeironWord.lean` (generated) |
| Physical relation | `RepR`, `DirectCover`, occupancy/summary fields | `proof/corpus/Growable/Representation.lean` |
| Search relation | `SearchDecisionCorrect` plus the read-only found/absent composition | `proof/corpus/Growable/Search.lean` |
| Filter relation | `SpillSummaryShape`, `SpillSummaryRecorded`, `SpillSummaryMissSound`, `SpillFingerprintExact`, `scan_exact` | `proof/corpus/Growable/Scan.lean`, with whole-state fields in `Growable/Representation.lean` |
| Canonical definition | `Apeiron.Table.HMap` and `Apeiron.Table.get` | pinned `Apeiron/Table/Basic.lean` |

The existing `word_{stored,regen}_apply_get` roots and insertion `SearchDecision` remain valid at
their exact scopes. The extraction task adds or retargets Stored/Regen roots for the real specialized
public read; a wrapper may specialize types and copy the borrowed result, but may not restate lookup.

### Read result and absence certificate

The read classifier has only three semantic outcomes: found in flat, found in routed spill, or
absent. It does not return insertion `Vacant`/`Exhausted`, compute first fit, reserve, grow, clone or
merge. The implementation may encode these outcomes through control flow rather than a public enum.

For query `q = (sigma,key)` and generated address `a`, a returned absence must carry one of this
disjoint three-way certificate:

1. **Direct certificate.** `a.direct_word` has an admitted resident index, the flat occupancy
   authority is empty there, and composite `DirectCover` proves `q` absent from the whole table.
2. **Summary-miss certificate.** The admitted direct candidate was absent or checked and unequal;
   `SpillSummaryShape` and `SpillSummaryRecorded` hold for `shard(a)`; and
   `SpillSummaryMissSound` derives that no live payload in that routed spill decodes to `q`.
3. **Coverage certificate.** Every possible live representative location after direct exclusion is
   checked for a full `(Sigma,K)` match. This is mandatory when the summary reports “maybe”, including
   a summary/fingerprint collision and a prefix vacancy followed by a late matching key.

No vacancy proves routed-spill absence. A summary miss alone proves only routed-spill absence and
therefore needs direct exclusion. A summary hit proves no presence. Equal fingerprints trigger the
full key comparison; unequal fingerprints exclude equality under `SpillFingerprintExact`.

`SearchDecisionCorrect` retains exact `Found`, first-fit `Vacant` and full-space `Exhausted` for
insertion. Its non-found field is repaired to have the logical form
`decision.class ≠ Found → DirectEmpty ∨ (DirectExcluded ∧ SummaryMissSound) ∨ CompleteMatchCoverage`.
`DirectEmpty` includes the admitted direct location, vacant occupancy authority and `DirectCover`;
`SummaryMissSound` includes the routed shard plus `SpillSummaryShape`, `SpillSummaryRecorded` and the
negative generated summary query; `CompleteMatchCoverage` includes every possible live location and
full `(Sigma,K)` inequality at every checked candidate. These constructors establish `Absent`; they
do not establish `VacantAt`. `vacantExact`, `FirstFit` and `exhaustedExact` stay separate insertion
fields. The read-only theorem projects only presence/absence and cannot manufacture a `Vacant`
insertion decision from an absent read.

### Exactness of the authorized changes

| Change | Preservation argument required |
|---|---|
| Inline resident classification, flat index and read apply | Compute the generated address and admitted flat index once; test `live`, `Sigma` and carrier/full-key identity exactly as before; return the same indexed value on a match, the same default under direct-empty `DirectCover`, and retain spill fallback for every occupied non-match or out-of-extent coordinate. |
| Remove vacancy work from reads | The read result needs no insertion destination. Summary-negative absence uses `SpillSummaryMissSound`; summary-positive reads scan matches only. Removing `match_empty`, first-vacancy state and first-fit ordering from this path changes no presence or absence predicate. |
| Retain complete summary-positive coverage | The finite window/sweep geometry still visits every possible spill representative, fingerprints filter candidates, and full `(Sigma,K)` comparison decides a hit. A late hit after any empty lane remains observable. |

These arguments preserve exactness only when their named invariants are established. A false
negative is forbidden. A false positive from summary/fingerprint collision may cost the bounded
`WINDOWS + groups` match sweep; it cannot change the returned value.

### Translation-compatible read shape

**Decision (`APEIRON-PERF-READ-CONTRACT-T2`, `anchor`): select the single-exit production body.**
Do not move `spill_read` behind an unextracted boundary and do not change the tool pin. The prepared-
shard and public-read roots both reach this real helper, so excluding it would leave the selected
performance-bearing matcher outside the claimed extraction. No retrieved newer Aeneas revision is
shown to accept the body, while changing the pin would change the extraction identity, patched-tool
surface and regression obligation without changing the read contract. The pinned Aeneas failure is
instead removed by replacing only the return from the nested hit loop with loop-carried state.

The exact authorized body shape is:

```rust
fn spill_read<K, H, R: KeyRep<K, H>, V>(
    r: &Region<R, V>,
    sigma: Sigma,
    address: &TableAddress,
    key: &K,
    n: usize,
) -> Option<usize> {
    let mut hit = None;
    if r.slots.len() != 0 && valid_n(n) {
        let groups = r.slots.len() / n;
        if groups >= 2 && groups * n == r.slots.len() && groups.is_power_of_two() {
            if r.summary_maybe(address.place) {
                let mask = groups - 1;
                let wanted = tag(address.place);
                let mut step = 0;
                while step < WINDOWS + groups && hit.is_none() {
                    #[cfg(test)]
                    read_trace::record_sweep_step();
                    let base = group(address.place, step, mask) * n;
                    let mut hits = scan::match_tag(&r.tags[base..base + n], wanted);
                    while hits != 0 && hit.is_none() {
                        let lane = least_set_lane(hits);
                        let slot = base + lane;
                        if matches::<K, H, R, V>(&r.slots[slot], sigma, key, address) {
                            hit = Some(slot);
                        }
                        hits &= hits - 1;
                    }
                    step += 1;
                }
            } else {
                #[cfg(test)]
                read_trace::record_summary_miss();
            }
        }
    }
    hit
}
```

`hit` starts at `None`; a summary miss leaves it unchanged without invoking `match_tag`; a summary
positive walks `group(address.place, step, mask)` in the existing order; `least_set_lane` preserves
lane order inside each group; and the first full `(Sigma,K)` match sets `hit` and clears both loop
conditions. Invalid or empty geometry remains `None`. The body has no vacancy state, `match_empty`,
allocation, clone or merge.

`get_prepared_shard`, `get_prepared_shard_at` and `get_at` do not require a single-exit rewrite for
this Aeneas limitation: none contains a return inside a loop. The throwaway audit does require
`get_prepared_shard` to spell its `Option::map` as an explicit `match`; that is an external-model
repair, not a loop-exit change. `get_prepared_shard_at`'s `unwrap_or` and `get_at`'s resident early
returns may remain. The other transitive read helpers
`summary_trailing_zeros`, `summary_maybe`, `group`, `least_set_lane`, `scan::match_tag`, `matches` and
`Flat::index` contain no nested-loop return. The insertion helpers `spill_locate` and
`spill_locate_held` already carry `found` through their nested loops and return after the loops;
`apply_get` is straight-line. This verdict does not authorize unrelated early-return cleanup.

The no-axiom candidate additionally replaces `Flat::index`'s `TryFrom::ok`/`?` chain with: compare
`u64::from(word >> 8)` against `self.lanes as u64` before conversion; on admission cast the proved-
fitting lane and the at-most-255 shard to `usize`; then explicitly match `checked_mul` and return
`base.checked_add(lane)`. This preserves out-of-range `None` on narrow `usize` targets and checked
index arithmetic; an unchecked or pre-guard truncating cast is not authorized.

**Throwaway tool result.** Under `/tmp/opencode/read-contract-t2`, Charon `0.1.254`
(`b104e24f…`, binary sha256 `747e165b…`) and Aeneas `505b6ca-dirty` (binary sha256 `33d0ce1a…`)
generated `table.spill_read`, both public-read roots and both prepared-shard roots from the candidate.
The accumulator-only candidate translated but its generated audit exposed five external axioms from
the pre-existing `Option::map` and `Flat::index` `TryFrom::ok`/`?` shapes. After the two explicit
straight-line rewrites above, the same four-root extraction exited 0 and the generated file contained
no `axiom`, `sorry` or `admit`. This is translation/audit evidence for the candidate call graph, not
`proof/regen.sh --check`, a Lean build or semantic refinement. The regeneration task must repeat the
full selector set and fail-closed audit on the production source.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `read_found_exact` | A found-flat or found-spill result is in bounds and contains the unique full logical key; the returned reference is its value. | proved | Stored/Regen read-refinement headlines in `Growable/Search.lean` |
| `read_absent_exact` | Absent follows one of the three certificates above and returns the represented default. | proved | Strengthened search/summary composition and `Apeiron.Table.get` |
| `read_summary_miss_no_vacancy` | A sound summary miss performs no tag-empty scan or first-vacancy search. | tested | Real-path trace test; insertion-locator mutation rejects |
| `read_positive_complete` | A summary-positive query checks complete match coverage and finds a late hit after an earlier vacancy. | proved/tested | Search headline plus late-hit Rust control |
| `read_identity` | Summary/tag/address equality never replaces full `(Sigma,K)` equality; unequal `Sigma` stays distinct. | proved/tested | Both-carrier collision witnesses and Rust tests |
| `read_effect_free` | Read changes no field and invokes no hold/clone/merge/allocation. | proved/tested | Generated read body/call graph and counted callback tests |
| `insert_search_unchanged` | Five insertion decision cases, first fit, growth and transactional refusal retain their statements. | proved regression target | Existing `Growable/Search`, `Refinement`, `Resources`, `Rebuild` headlines |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Present resident key | Borrowed resident value | exact `live` plus full logical-key match |
| Occupied resident non-match | Spill fallback | direct occupancy is not identity |
| Empty admitted resident slot | Borrowed default | composite `DirectCover` direct certificate |
| Routed spill summary miss after direct exclusion | Borrowed default | `SpillSummaryMissSound` |
| Summary/fingerprint false positive | Bounded complete match sweep; value if matched, otherwise default | full `(Sigma,K)` comparison and coverage certificate |
| No direct coordinate or coordinate outside extent | Routed spill read | unchanged fallback |

There is no read refusal, panic, clone, merge, allocation or state transition under the admitted
callback/representation domain. Found-key behavior does not change.

### Bridge consequence

Regeneration changes the generated definitions reached by the Stored/Regen public-read roots:
`word_stored_table_get_at`, `word_regen_table_get_at`, `table.Table.get_at`, the shared read-only spill
matcher, resident index/classification code, and any retargeted
`word_{stored,regen}_prepared_shard_get` body. `table.Table.search_decision`, insertion
`spill_locate`, `word_{stored,regen}_apply_insert`, relocation, resource publication and commit bodies
must remain semantically unchanged; generated declaration movement still requires freshness and
regression builds.

The required bridge repair after regeneration is:

1. `Growable/Scan.lean`: retain `summary_trailing_zeros_eq` and `scan_exact`; establish the
   summary shape, recordedness, miss-soundness and fingerprint equations used by the read body.
2. `Growable/Representation.lean`: carry those summary facts in `RepR`/`ReservedRepR` and preserve
   the existing `DirectCover`, uniqueness and observation projections; statement changes are allowed
   only where the already-approved summary fields require them.
3. `Growable/Search.lean`: repair `SearchDecisionCorrect` to the three-way absence certificate; add
   both-carrier public-read found/absent/refinement headlines against `Apeiron.Table.get`; preserve
   all insertion/apply headlines and `search_decision_simulates` at their exact premises.
4. `AxiomAudit.lean`: after Scan, Representation and Search are green, audit every new/revised
   headline under
   `{propext, Classical.choice, Quot.sound}`. `proof/regen.sh --check` and the bounded bridge build
   bind the same source/generated identity.

`Growable/Relocation.lean`, `Growable/Refinement.lean`, `Growable/Rebuild.lean` and
`Growable/Resources.lean` then receive regression builds only; repair generated-name dependencies if
regeneration moved them, without weakening a premise or grade. They are not predecessors of the read
theorem.

### Serialized implementation dispatch

1. **Flash source edit.** Own `src/table.rs` and `src/table/state.rs`: replace `spill_read` with the
   exact accumulator body above; replace `get_prepared_shard`'s `Option::map` with `Some`/`None`
   matching; and replace `Flat::index` with the guarded, checked explicit form above. Make no change
   to `get_at`, insertion locate or tests. Gate: `cargo test --locked --lib --test table`. Control in
   a throwaway copy: truncate the positive outer sweep after its first group; existing
   `read_helper_finds_a_late_match_after_a_window_vacancy` and
   `read_finds_a_late_spill_hit_after_a_window_vacancy` must fail, then the restored gate must pass.
2. **Flash regeneration.** Own `proof/extract/lib.rs`, `proof/regen.sh` and
   `proof/ApeironWord.lean`; selectors remain the two public reads and two prepared-shard reads. Run
   `proof/regen.sh` once to regenerate, then gate with `proof/regen.sh --check`. Inventory
   `table.spill_read`, both `word_{stored,regen}_table_get_at` and both
   `word_{stored,regen}_prepared_shard_get` declarations; the audit must contain no `axiom`, `sorry`
   or `admit`. Control: mutate the generated hit assignment in a throwaway copy; freshness must fail,
   then restore and pass. Freshness is not the semantic proof control.
3. **Sol bridge repair.** Serialize `Growable/Scan.lean`, then
   `Growable/Representation.lean`, then `Growable/Search.lean`, and edit `AxiomAudit.lean` last.
   Each task uses the bounded, lane-locked focused build named in the plan and its semantic
   red/restore control. Revalidate Relocation, Refinement, Rebuild and Resources after the audited
   read chain is green.

### Evidence plan

1. **Positive Rust tests.** `cargo test --locked --lib --test table` covers Stored/Regen resident hit,
   resident direct absence, occupied-direct spill fallback, spill hit, summary-negative spill miss,
   summary-positive miss, fingerprint collision, unequal `Sigma`, disabled summary and late hit after
   a prefix vacancy. Counted test hooks assert zero hold/clone/merge/allocation and zero vacancy scans
   for read-only summary misses.
2. **Rust rejecting controls.** In throwaway copies: route the read through insertion vacancy search;
   omit the mandatory positive sweep; treat a summary hit as presence; bypass full-key comparison;
   or map an occupied direct non-match to default. Each named test must fail for its semantic reason,
   then the restored gate passes.
3. **Extraction and proof.** Regenerate once, then require `proof/regen.sh --check`; run bounded,
   serialized `lake build Growable.Scan Growable.Representation Growable.Search AxiomAudit` in
   `proof/corpus/` under the repository memory/thread rules.
4. **Proof controls.** Mutate the generated summary-negative branch to skip direct exclusion, and
   mutate positive coverage to omit a late group. The applicable read/absence headline must fail;
   restore and rebuild green. Freshness rejection alone is not the semantic control.
5. **Focused micro measurement.** First extend `examples/micro_spill.rs` without changing the REG-1
   schedules: add resident hit/miss cells, raw sample output, independent readback digests and
   fail-closed controls. Run
   `cargo run --release --locked --example micro_spill -- --controls` on the unchanged source for the
   baseline and on the final source for the result. Record raw samples, median ns/op and the unchanged
   4096-operation schedules. This diagnostic does not pass REG-1.
6. **Strict pair gate.** Run
   `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`.
   Record every affected resident/spill present-hit and absent-miss pair. Each applicable pair passes
   only when its interval maximum is `<= 1.0`; every other pair and all controls remain load-bearing.
   The full command may remain red because another cell fails, and that keeps G1 open.

### Trust boundary

The read-loop correctness premise remains outside an unconditional claim until the real production
body is extracted and the new headlines close. Tests and timing do not prove generic absence.
Allocator, executor, callback panic/allocation outside admitted laws, rustc/LLVM, ABI, GPU/SPIR-V and
unsafe-container realization remain excluded. No exclusion changes the public domain or permits a
false negative.

### Open questions

- Aeneas may retain the generic-borrow context-join blocker for a combined public-read root. The
  extraction task must report the exact body and failure; a model/reimplemented wrapper is forbidden.
- Attainability of all affected intervals at `<= 1.0` is unestablished. A red bounded campaign returns
  to Astra; it does not change the threshold or authorize the deferred directory/layout designs.

## Plan — Growable table read-path specialization

Kani retirement supersedes older model-checker requirements; see [the retirement contract](kani-retirement.md).

- **Owner:** Astra architecture; Flash source/extraction; Sol bridge
- **Intent:** `docs/epics/read-path-specialization.md`
- **Spec:** `docs/epics/read-path-specialization.md`
- **Source baseline:** historical dispatch revision `c450c222905c050f793a23332708c230c658f82c`; current evidence is digest-bound in [the retirement receipt](../evidence/kani-retirement.md)
- **Status:** source/extraction campaign complete at its recorded tested/freshness scopes; remaining universal read/search proof and allocator work moves to [the next bounded plan](read-proof.md)

### Arc block

```text
SCOPE:      production read specialization, real-source extraction, bridge repair and matched read measurement; complete when every child gate/control is green and strict affected pairs are recorded without weakening G1
REGISTER:   state/refinement/test/measurement; excludes layout, cached summary state, second containers, spatial scatter and W7 composition
CORPUS:     Apeiron/Table/Basic.lean `get`, `get_insertWith_self`, `get_insertWith_ne` (proved); Growable/Search.lean `SearchDecisionCorrect` and Stored/Regen apply-get headlines at their current conditional scopes
REFERENCE:  /home/socol/Workspace/apeiron/src/table.rs and src/table/state.rs read from the supplied checkout; mechanism derived from docs/direction.md `Serial read specialization and stable parallel scatter`
GATE:       focused commands below; strict REG-1 baseline is red at pass=421 fail=1029, and no task may convert that observed red to acceptance by reducing coverage
GRADE:      lifecycle documents are anchor; implementation targets tested, bridge targets proved, runtime targets measured
```

This plan is the historical dispatch record for the completed source/extraction campaign. Every
implementation/proof task below owns at most three files. Use an isolated
`worktree-agent-APEIRON-PERF-READ-<id>` at the recorded base revision and serialize `target/`,
`proof/ApeironWord.lean`, `proof/.lake/` and `proof/corpus/.lake/`.

### Goal and completion condition

The production read performs no insertion-only vacancy work, exact found/default behavior is tested
and source-refined for both carriers, and before/after micro plus strict full-profile results are
recorded on one source/tool identity. These local gates do not close G1 unless the strict command
itself exits 0 for every applicable pair.

```sh
cargo test --workspace --all-features --locked && cargo clippy --workspace --all-features --all-targets --locked -- -D warnings && cargo fmt --all -- --check && proof/regen.sh --check
```

Run the two bounded Lean package builds separately under the required memory cap, timeout,
single-thread setting and `/tmp/opencode/apeiron-lane.lock`; they are barrier gates, not part of the
shell chain above.

### Non-goals

- No threshold, applicability, workload, incumbent, interval or control change.
- No benchmark-specific dispatch, second representation, cached summary width, flat geometry,
  first-empty rule, payload replacement, spatial scatter or W7 retry.

### Historical wave map

The table below is the completed source/extraction campaign and its then-planned downstream proof
work. Its W3/W4 proof rows do not describe current dispatch; the remaining Read requirement is
decomposed by [the next bounded plan](read-proof.md).

| Wave | Task | Files | Owner | Gate | Rejecting control |
|---|---|---|---|---|---|
| W0 | `APEIRON-PERF-READ-W0-T1` micro observability and frozen-source baseline | `examples/micro_spill.rs` | Flash | `cargo run --release --locked --example micro_spill -- --controls` | Corrupt one expected readback digest in a throwaway copy; the gate fails before timing, then restored gate passes. |
| W1 | `APEIRON-PERF-READ-W1-T1` production read specialization | `src/table.rs`, `src/table/state.rs`, `tests/table.rs` | Flash | `cargo test --locked --lib --test table` | In a throwaway copy, stop a summary-positive match scan at the first vacancy; the late-hit test fails, then restored gate passes. |
| W2 | `APEIRON-PERF-READ-W2-T1` real-read extraction | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | Flash | `proof/regen.sh --check` | In a throwaway copy, mutate one generated read exclusion branch; freshness fails, then restored gate passes. This is not the later semantic proof control. |
| W3 | `APEIRON-PERF-READ-W3-T1` summary/representation bridge repair | `proof/corpus/Growable/Scan.lean`, `proof/corpus/Growable/Representation.lean`, `proof/corpus/AxiomAudit.lean` | Sol | bounded `lake build Growable.Scan Growable.Representation AxiomAudit` in `proof/corpus/` | Remove recordedness from one live spill witness; the representation/summary headline fails, then restored build passes. |
| W4 | `APEIRON-PERF-READ-W4-T1` read/search bridge repair | `proof/corpus/Growable/Search.lean`, `proof/corpus/AxiomAudit.lean` | Sol | bounded `lake build Growable.Search AxiomAudit` in `proof/corpus/` | In a throwaway generated copy, let summary miss imply global absence without direct exclusion; the read theorem fails, then restored build passes. |
| W5 | `APEIRON-PERF-READ-W5-T1` focused post-change measurement | no edit; read `examples/micro_spill.rs` | Flash | `cargo run --release --locked --example micro_spill -- --controls` | The W0 digest/control set reruns unchanged; any accepted corruption rejects the result. |
| W6 | `APEIRON-PERF-READ-W6-T1` strict pair/full-matrix measurement | no source edit; existing `examples/reg1/` | Flash | strict full-profile command below | `--controls` must report every registered corruption red; any surviving control rejects the run. |

W1 starts only after W0 freezes the harness and records the unchanged-source baseline. W2 starts only
after W1 is green/restored. W3 starts only after W2 fixes the generated identity. W4 starts after W3
because both Sol tasks own `AxiomAudit.lean` and the corpus build lane. W5 starts from the
proof-repaired identity and uses the byte-identical W0 harness. W6 starts only after W5 records its
diagnostic result. No source or proof edit may overlap these waves.

### Task blocks

#### APEIRON-PERF-READ-W0-T1 · Freeze the focused read micro harness and baseline

- **Scope:** Preserve the existing spill absent-miss/present-hit 4096-operation schedules; add the
  corresponding resident hit/miss schedules, independent expected readback digests, raw sample
  output and fail-closed `--controls`; run the baseline before any timed source edit.
- **Files:** `examples/micro_spill.rs`
- **Depends on:** none; run after the current frozen-source owner releases `src/table.rs`
- **Grade:** measured baseline plus tested harness controls
- **Gate:**

```sh
cargo run --release --locked --example micro_spill -- --controls
```

- **Rejecting control:** In a throwaway copy, corrupt one expected readback digest. The same gate must
  exit nonzero before timing; restore and rerun green. The control validates harness rejection, not
  table correctness.
- **Evidence to record:** harness/source/input SHA-256, host/compiler/governor/cache/thread conditions,
  all raw samples, medians/ns per op, digest results and baseline/red/restored exits.

#### APEIRON-PERF-READ-W1-T1 · Production read specialization

- **Scope:** Inline resident address/index/classification/read application; route occupied non-matches
  to spill; add one read-only spill match path with summary-miss absence and complete
  summary-positive match coverage; leave insertion `SearchDecision`/first-fit/growth unchanged.
- **Files:** `src/table.rs`, `src/table/state.rs`, `tests/table.rs`
- **Depends on:** W0 and completion of the active frozen-source Sol owner
- **Grade:** tested target
- **Gate:**

```sh
cargo test --locked --lib --test table
```

- **Rejecting control:** In a throwaway worktree, terminate the summary-positive read match scan when
  the first empty tag is observed. The real-path late-hit test must fail on a wrong default; restore
  and rerun green. The positive suite also asserts a summary-negative read performs zero vacancy
  scans and zero hold/clone/merge/allocation, so restoring insertion-locator vacancy bookkeeping is a
  separate required test failure.
- **Evidence to record:** source SHA-256; named tests and counts; baseline/red/restored exits; exact
  call graph showing `get_at` uses the production read path and insertion still uses
  `search_decision`/`spill_locate`.

#### APEIRON-PERF-READ-W2-T1 · Extract the real specialized read

- **Scope:** Add/retarget monomorphic Stored/Regen wrappers and selectors for the production read,
  regenerate once, and retain every existing root. Wrappers specialize and call; they do not restate
  resident or spill matching.
- **Files:** `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`
- **Depends on:** W1
- **Grade:** extraction audit; not semantic proof
- **Gate:**

```sh
proof/regen.sh --check
```

- **Rejecting control:** In a throwaway copy of the generated artifact, change the summary-negative
  read branch or a full-key comparison. The freshness diff must fail; restore and rerun green. W4
  supplies the semantic control, because freshness alone proves no read theorem.
- **Evidence to record:** all source/generated SHA-256 values, actual Aeneas/Charon versions, generated
  body inventory for both carriers, audit output and baseline/red/restored exits. A context-join or
  external-axiom failure is a blocker, not permission to add a model root.

#### APEIRON-PERF-READ-W3-T1 · Repair summary and representation premises

- **Scope:** Source-link/audit spill-summary shape, recordedness, miss soundness and fingerprint
  exactness and carry them through `RepR`/`ReservedRepR`, preserving all existing projections.
- **Files:** `proof/corpus/Growable/Scan.lean`, `proof/corpus/Growable/Representation.lean`,
  `proof/corpus/AxiomAudit.lean`
- **Depends on:** W2
- **Grade:** proved target under the closed Rust/canonical cone
- **Gate:** bounded per repository policy:

```sh
lake build Growable.Scan Growable.Representation AxiomAudit
```

- **Rejecting control:** Remove a required recorded summary bit from one live spill witness in a
  throwaway proof copy. The named constructor/projection or miss-soundness dependency fails; restore
  and rebuild green.
- **Evidence to record:** source/generated/proof digests, Lean/Mathlib revisions, named headlines,
  closed axiom sets and baseline/red/restored exits.

#### APEIRON-PERF-READ-W4-T1 · Repair the read/search refinement

- **Scope:** Give `SearchDecisionCorrect` the direct/summary/coverage non-found sum; prove both
  carrier public-read found and absence branches against `Apeiron.Table.get`; preserve every
  insertion/apply headline and premise.
- **Files:** `proof/corpus/Growable/Search.lean`, `proof/corpus/AxiomAudit.lean`
- **Depends on:** W3
- **Grade:** proved target, conditional exactly on the named reachable representation/search laws
- **Gate:** bounded per repository policy:

```sh
lake build Growable.Search AxiomAudit
```

- **Rejecting control:** In a throwaway generated/proof copy, remove direct exclusion from the
  summary-miss branch or omit one summary-positive late group. The corresponding public-read theorem
  must fail semantically; restore and rebuild green. Compile errors from deleting a declaration do
  not count.
- **Evidence to record:** named theorem statements, joint Stored/Regen witnesses, canonical `get`
  attribution, axiom audit and baseline/red/restored exits.

#### APEIRON-PERF-READ-W5-T1 · Focused post-change measurement

- **Scope:** Measure the unchanged resident/spill hit/miss 4096-operation schedules with the
  byte-identical W0 harness and compare against its frozen-source baseline; use no result as an
  acceptance substitute.
- **Files:** none; read `examples/micro_spill.rs`
- **Depends on:** W4
- **Grade:** measured target
- **Gate:**

```sh
cargo run --release --locked --example micro_spill -- --controls
```

- **Rejecting control:** Rerun the W0 digest corruption against the byte-identical harness in a
  throwaway copy; it must still exit nonzero before timing, then the restored gate passes. W6's
  independent oracle, workload identity, selected-implementation and slowed/failing-pair controls
  remain mandatory before any acceptance reading.
- **Evidence to record:** host, compiler, source/harness/input digests, governor/cache/thread
  conditions, raw samples, median ns/op and interval/protocol before and after.

#### APEIRON-PERF-READ-W6-T1 · Strict affected-pair and full-matrix result

- **Scope:** Run the unchanged full registry and record every resident/spill present-hit and
  absent-miss pair plus the global headline. Do not filter to passing pairs.
- **Files:** none; read existing `examples/reg1/`
- **Depends on:** W5
- **Grade:** measured target; acceptance only if the command exits 0
- **Gate:**

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

- **Rejecting control:** The command's wrong workload/oracle, omitted incumbent, selected
  implementation, interval-crossing and strict-single-failing-cell controls must all report red. A
  surviving control or any missing/inconclusive pair rejects the campaign.
- **Evidence to record:** complete pair table, affected-cell intervals, full headline, exit status,
  raw log hash, machine/tool/source/harness/input identities. Any interval maximum above `1.0` stays
  red; no scoped improvement closes G1.

### Barrier checklist

- [ ] W1–W4 focused gates exit 0; each control is red and restored green.
- [ ] Both carriers have joint direct hit/absence, spill hit/absence, collision and unequal-`Sigma`
  witnesses; no separate incompatible witnesses.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` pass.
- [ ] `proof/regen.sh --check`, bounded `lake build` in `proof/`, and bounded full `lake build` in
  `proof/corpus/` pass on one source/tool identity.
- [ ] Existing relocation/resource/search controls are rerun; W7 remains frozen.
- [ ] Micro and strict full-profile outputs bind the same final source/harness identity.
- [ ] Every claim is labelled tested, proved, measured or open at its actual scope.
- [ ] No acceptance threshold, incumbent, applicability, workload, interval or exclusion changed.

### Stop and rollback conditions

- Stop on a false negative, late-hit loss, unequal-`Sigma` alias, found-value change, read callback or
  allocation, insertion-search change, surviving semantic mutation, generated axiom, or extraction
  wrapper that restates lookup.
- Stop and return to Sol/Astra if the real specialized read cannot be extracted under the pinned
  toolchain; retain the production correctness fix only if its Rust gate/control pass, but claim no
  proved read refinement.
- After three materially different failed implementation shapes, report the smallest unresolved
  source body and generated/proof goal. Do not change geometry, add cached state/another container,
  weaken complete coverage or relax `<= 1.0`.

## Flat spill read — port record

Sol answered the absence/Grow contract for the flat open-addressed spill region (shape A, retire the
summary; the spec below is the transcription). The measured basis is in `docs/plan.md` E6: the
prototype reads 1.34 ns/key against the canonical 8.14 in one process over nine interleaved rounds
with identical geometry, a 6.06x ratio reproduced on an uncontended box (E6.W2d, E6.W2e). Two
instrument facts bound that measurement: the `read-stats` feature costs 1.90 ns/read on the
canonical path, so the timing and attribution builds are separate, and the prototype's steps per
key depend on its start-index hash choice (1.0104 with a splitmix mix), not on the flat layout.

### Spec

- One flat open-addressed region per shard; capacity and the linear probe sequence are specified
  with the bound covering every slot exactly once; no deletions, or a separately specified
  tombstone protocol.
- Absence certificate: bounded probing. The registered replacement fact is
  `SpillProbeBoundAbsentSound` — a lookup returns absent only after reaching an empty slot or
  exhausting the bound; any published matching key is encountered before the terminating empty
  slot under the insertion-placement invariant; exhausting the bound with no match means no
  matching key is in the region; an in-progress publication cannot produce a false absence.
- Growth: `Grow` is a full probe cycle that found every slot occupied, not a `live == capacity`
  observation alone; `live` is the occupied count, updated exactly once per publication;
  `spill_locate` and `spill_locate_held` carry the same outcome contract.
- Publication: payload before the nonzero-tag marker; summary publication leaves the protocol; the
  tests that assert payload → summary bit → tag are rewritten to the marker order.
- Proof surface: `SpillSummaryMissSound`, its classifier bridge and its audit entry are retired
  with the replacement theorem named, and `AxiomAudit`'s headline set reflects the new contract.
- Extraction: `spill_read`, `spill_locate`, `spill_locate_held`, `Region::publish_live` and the
  summary helpers change shape, so `proof/ApeironWord.lean` regenerates and the refinements naming
  them re-link. The regeneration rides behind the unrelated pin-assert blocker; it is a sequencing
  dependency, not a reason to retain the summary.

### Plan

| task | change | gate |
|---|---|---|
| invariant spec | capacity, probe bound, tombstone policy and publication order stated at the declarations and in tests | the spec text resolves to the declarations it names |
| region port | `src/table/probe.rs`, `src/table/state.rs`, `src/table/scan.rs`, `src/table.rs`, `src/table/prepared.rs`: flat region, drop the tags bitmap, the summary and the group descent | `cargo test --workspace --all-features --locked` and the harness at parity |
| publication fixtures | rewrite the publication-order tests and the geometry fixtures to the marker protocol | the named tests |
| proof replacement | add `SpillProbeBoundAbsentSound`; retire `SpillSummaryMissSound`'s representation theorem, bridge and audit entries | `cd proof/corpus && lake build` |
| extraction re-link | regenerate and re-link the spill refinements | `proof/regen.sh --check` (behind the pin assert) |
| integrated benchmark | the production publication and concurrency rules, not the scratch prototype | the harness counters and `micro_spill` |
