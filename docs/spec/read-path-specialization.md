# Specification — Growable table read-path specialization

- **Owner:** Sol contract/attribution; Astra mechanism boundary
- **Intent:** `docs/intent/read-path-specialization.md`
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for bounded implementation

## Scope and admissible domain

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

## Representation relation (Rust to Lean)

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

## Read result and absence certificate

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

## Exactness of the authorized changes

| Change | Preservation argument required |
|---|---|
| Inline resident classification, flat index and read apply | Compute the generated address and admitted flat index once; test `live`, `Sigma` and carrier/full-key identity exactly as before; return the same indexed value on a match, the same default under direct-empty `DirectCover`, and retain spill fallback for every occupied non-match or out-of-extent coordinate. |
| Remove vacancy work from reads | The read result needs no insertion destination. Summary-negative absence uses `SpillSummaryMissSound`; summary-positive reads scan matches only. Removing `match_empty`, first-vacancy state and first-fit ordering from this path changes no presence or absence predicate. |
| Retain complete summary-positive coverage | The finite window/sweep geometry still visits every possible spill representative, fingerprints filter candidates, and full `(Sigma,K)` comparison decides a hit. A late hit after any empty lane remains observable. |

These arguments preserve exactness only when their named invariants are established. A false
negative is forbidden. A false positive from summary/fingerprint collision may cost the bounded
`WINDOWS + groups` match sweep; it cannot change the returned value.

## Translation-compatible read shape

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

## Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `read_found_exact` | A found-flat or found-spill result is in bounds and contains the unique full logical key; the returned reference is its value. | proved | Stored/Regen read-refinement headlines in `Growable/Search.lean` |
| `read_absent_exact` | Absent follows one of the three certificates above and returns the represented default. | proved | Strengthened search/summary composition and `Apeiron.Table.get` |
| `read_summary_miss_no_vacancy` | A sound summary miss performs no tag-empty scan or first-vacancy search. | tested | Real-path trace test; insertion-locator mutation rejects |
| `read_positive_complete` | A summary-positive query checks complete match coverage and finds a late hit after an earlier vacancy. | proved/tested | Search headline plus late-hit Rust control |
| `read_identity` | Summary/tag/address equality never replaces full `(Sigma,K)` equality; unequal `Sigma` stays distinct. | proved/tested | Both-carrier collision witnesses and Rust tests |
| `read_effect_free` | Read changes no field and invokes no hold/clone/merge/allocation. | proved/tested | Generated read body/call graph and counted callback tests |
| `insert_search_unchanged` | Five insertion decision cases, first fit, growth and transactional refusal retain their statements. | proved regression target | Existing `Growable/Search`, `Refinement`, `Resources`, `Rebuild` headlines |

## Failure semantics

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

## Bridge consequence

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

## Serialized implementation dispatch

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

## Evidence plan

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

## Trust boundary

The read-loop correctness premise remains outside an unconditional claim until the real production
body is extracted and the new headlines close. Tests and timing do not prove generic absence.
Allocator, executor, callback panic/allocation outside admitted laws, rustc/LLVM, ABI, GPU/SPIR-V and
unsafe-container realization remain excluded. No exclusion changes the public domain or permits a
false negative.

## Open questions

- Aeneas may retain the generic-borrow context-join blocker for a combined public-read root. The
  extraction task must report the exact body and failure; a model/reimplemented wrapper is forbidden.
- Attainability of all affected intervals at `<= 1.0` is unestablished. A red bounded campaign returns
  to Astra; it does not change the threshold or authorize the deferred directory/layout designs.
