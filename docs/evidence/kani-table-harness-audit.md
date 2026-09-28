# Kani harness audit: stale fixed-shard premises and CBMC cost

Status: superseded by [user-directed Kani retirement](../epics/kani-retirement.md).
Kani receipts and requirements below describe the earlier work only; do not run or retry them.
Retirement does not turn timeouts into passes or close missing Lean refinements.

Gate: `cargo kani` (`src/proofs.rs`, `#[cfg(kani)] mod harnesses`, 63 harnesses).
Machine: `i9-14900KF`, 32 threads, 62 GiB. Kani `0.67.0`, CBMC `6.8.0`.
Run: `nice -n 10 timeout <T> cargo kani -j 16 --output-format terse -Z unstable-options --harness-timeout <t>s`.

## Result

| cap | verified | failed | total |
|---|---|---|---|
| 60 s | 49 | 14 | 63 |
| 300 s | 51 | 12 | 63 |

Raising the cap does not close the gap: the 11 table/search harnesses do not finish at 90 s,
600 s or 900 s. The historical 62/62 receipt predates the growable rework
(`docs/flash-handoff.md`: historical completion records are not fresh acceptance).

## Two distinct faults

### 1 · Stale premise: the harnesses encode the retired fixed-shard `Table`

`table_zero_capacity_refuses_without_merging` is a **counterexample in 8.37 s**, not a timeout:

```text
** 1 of 2998 failed (306 unreachable)
Failed Checks: assertion failed: result == Err(StorageError::ShardFull)
 File: "src/proofs.rs", line 1268, in table_zero_capacity_refuses_without_merging
```

The refutation is the error **value**, not the refusal. `N = 0` is not a valid group width
(`valid_n`, `src/table/state.rs:317`), so `insert_with_at` returns `SizeOverflow` before addressing
or merging (`src/table.rs:1435-1437`); the harness expected `ShardFull`. Corrected to
`SizeOverflow`. The refusal obligation stands; the fix is the asserted variant. `ShardFull` is
compatibility-only (`src/table.rs:89`): the growable shard expands rather than filling, so
`ShardFull` is not the current full-shard outcome. Once the assertion is true CBMC must prove it,
and the harness's post-refusal `get` searches all `SHARDS = 256` regions, so the formerly-cheap
harness now times out — the geometry cost again. `table_refusal_leaves_the_table_unchanged`
asserted the same retired fixed-shard contract; it now checks a non-power-of-two group width
(`N = 3`) refused as `SizeOverflow` with the panicking merge proving nothing was published.

`the_word_home_route_is_the_low_byte` asserted `WordHome::home(&a) % 256 == (a as u64) & 0xff`.
`WordHome::home` is `coord::spread(word).place` (`src/table.rs:145`); `spread` became one widening
multiply on 2026-08-29 (`src/coord.rs:366-374`), so the route is not the key word's low byte.
Rewritten to `the_word_home_route_is_the_placement_word`, which passes in **0.29 s**.

The rewritten route harness was still not the kernel's selector. The kernel routes through
`state::shard(&address)` (`src/table/state.rs:325-329`): `direct_word` when a key has one
(`|word| <= 2^32`, so the raw low byte), otherwise `place`. `WordHome.home` is consumed only on the
`None` branch. It is replaced by `the_shard_selector_reads_the_direct_word_or_the_place`, which
consumes `state::shard` over a symbolic `TableAddress` and over concrete keys straddling `u32::MAX`,
and passes in **0.38 s**.

`search_decision_classifies_a_bounded_two_lane_spill` holds a second false assertion. The default is
`-1` and `WordMerge` is new-first wrapping addition, so a first write of `v` stores
`v.wrapping_sub(1)`; the harness expected `v`. Corrected at the two read sites.

### 2 · CBMC cost

The remaining harnesses time out. `N = 0` decides in 8 s because it returns before the growable
path; that shows the early-refusal path is cheap, not that growth is the sole cause. With `N >= 1`
the executing path runs `try_reserve` + region construction + relocate + **locator
reservation/publication** (`src/table.rs:1449-1459`) over `SHARDS = 256`
(`src/table/partition.rs:26`) shards and `lanes * 256` flat slots (`src/table/state.rs:478-500`).
Default-face writes execute the locator path, so geometry is not the only cost and `home`/`address`
are not the only fixture methods consumed. Three algebra harnesses
(`the_dyadics_are_a_commutative_ring`, `the_norm_factors_into_the_channels`,
`byte_cell_mul_is_associative`) also exceed 60 s but pass under 300 s.

Making `ConstZeroHome` key-independent (no `WordFaceHome` multiply on the symbolic key) did not
change the outcome.

## Decision (Astra)

**Reduced `cfg(kani)` geometry in the shared implementation, retaining the growth, reservation,
relocation and search obligations; replace the `MiniTable` model with real-`Table` harnesses at that
width.** Grade `anchor`: an architecture decision, not evidence the reduced configuration finishes
in bound. The bridge stays about production geometry; Kani is not accommodated by changing it.

| Quantity | Normal | `cfg(kani)` |
|---|---:|---:|
| `SHARD_BITS` | 8 | 2 |
| `SHARDS = 1 << SHARD_BITS` | 256 | 4 |
| `SHARD_MASK = SHARDS - 1` | 255 | 3 |
| Direct-word lane shift | 8 | 2 |
| Flat byte budget | 262144 | 4096 |
| Maximum direct-word lane bits `32 - SHARD_BITS` | 24 | 30 |

These hardcoded 256s move together: `SHARDS` (`src/table/partition.rs:26`); `word & 255` and
`lanes.checked_mul(256)` / `self.lanes * 256` (`src/table/state.rs:472,478,485,492`). Changing only
the masks and allocation lengths is invalid: `word >> 8` with a two-bit mask discards coordinate
bits. Keep unchanged: summary minimum width 256 bits, fingerprint/descent extraction from
`place >> 8`, the seven descent windows, locator byte labels/levels, key/value widths, codec and
group-width validation. Compile-time assertions bind normal `(8,256,255)` and Kani `(2,4,3)`, and
the flat-index identities must agree.

Unwinds: keep unwinding assertions, bound per harness. Loop-local floors: directory construction
`4+1`; two-group search `7+2+1`; four-group after growth `7+4+1`; materialization/relocation, max
reached length plus one; parabolic locator `>= 65`. For the first feasibility run keep the current
global bounds (257, and 1025 for the four-byte-key harness); tighter numbers are an implementation
deliverable. If they time out, return the dominating loop as a blocker; do not disable checks.

Staged ownership: 1 Sol review (this document) → 2 geometry (`partition.rs`, `state.rs`, `table.rs`)
→ 3 harnesses (`src/proofs.rs`, this document) → 4 extraction (`proof/ApeironWord.lean`,
`proof/regen.sh --check`) → 5 bridge only if regeneration moves it (Sol-owned). Extraction source
digests move; generated bytes must be regenerated and rechecked.

Acceptance:

```sh
nice -n 10 timeout 1800s cargo kani -j 16 --output-format terse -Z unstable-options --harness-timeout 300s
```

Residual Kani claim: bounded checking of the shared executing algorithms at four-shard geometry, the
recorded populations and callback instances, including the explicitly exercised resource and
relocation branches. It does not establish 256-shard equivalence, allocator realization,
concurrency, complete growable-table refinement or codegen. The Aeneas search refinement is
conditional on `SearchDecisionCorrect` (`proof/corpus/Growable/Search.lean`); it does not prove the
decision loop.

## Stage 1 verdicts (Astra, Sol)

Astra (architecture, `anchor`): reduced `cfg(kani)` geometry as above; the bridge stays about
production geometry.

Sol (contract/claim):
- `SizeOverflow` ratified for an invalid group width; the refusal obligation survives. `ShardFull`
  is compatibility-only, not the growable full-shard outcome.
- The route harness must consume `state::shard(&address)` (direct word when present, else place,
  masked) — applied.
- Reduced `cfg(kani)` geometry ratified, grade `anchor` until the gate passes.
- Extraction digests move and require regeneration; the bridge is not changed for Kani. The Kani
  result becomes `tested`/bounded at four shards, not `proved`; bridge theorem grades are unchanged,
  but current-source `proved` attribution is stale until regeneration, audit and the bridge build
  pass.
- `table_refusal_leaves_the_table_unchanged` was a remaining stale assertion — replaced (above).

## Stage 2 result (geometry)

Applied: `SHARD_BITS`/`SHARDS`/`SHARD_MASK` are `cfg(kani)` (`2`/`4`/`3`) and `8`/`256`/`255`
otherwise, with compile-time assertions binding both; `MAX_LANE_BITS = 32 - SHARD_BITS`;
`DEFAULT_BUDGET_BYTES = SHARDS * 1024`; the lane shift and mask in `state::shard`, `Flat::index`,
`Flat::byte_len_for`, `Flat::reserve`, `Flat::initialize` and `flat_index_for` now use `SHARDS`,
`SHARD_BITS`, `SHARD_MASK`. Production is unchanged: `cargo test --workspace --all-features`
rc 0, `cargo clippy … -D warnings` rc 0, `cargo fmt --all -- --check` rc 0. The `cfg(kani)` build
compiles and the assertions hold.

Effect: `cargo kani` at a 120 s cap is 53/63 (from 49/63 at 60 s). All 10 remaining are timeouts,
no counterexamples. The two simplest table harnesses still time out at 600 s.

## Blocker: the probe's data-dependent loops, not the geometry

The stuck loop is `table::probe::spill_read` (`src/table/probe.rs:238`), unwinding to the global
bound. Its outer loop is `while step < WINDOWS + groups` with `groups = r.slots.len() / n` and its
inner loop is `while hits != 0`, both bounded only by values CBMC cannot bound: `r.slots.len()` is a
symbolic `Vec` length and `hits` is a data-dependent bitmask. A sound `#[kani::unwind]` therefore
covers the full symbolic bound and the nested body is intractable.

Evidence: a throwaway `#[cfg(kani)]` early return in `spill_read` makes
`table_fresh_read_is_the_default_for_every_key` pass in **0.8 s**; reverting it restores the
timeout. `--unwind 8` fails the unwinding assertion on the same loop; the full bound times out.
Disabling `--no-assertion-reach-checks --no-memory-safety-checks --no-overflow-checks` does not
help, so the cost is the core formula, not the check instrumentation. `--slice-formula` is already
on by default. A `cfg(kani)` `kani::assume(r.slots.len() <= 16)` with a matching unwind of 24 was
also tried and still times out: the unrolled body, not the unroll count, dominates.

Consequence: geometry reduction is necessary but not sufficient. A green gate needs the probe
entered with either a stubbed boundary (Kani `#[kani::stub]` is blocked because `probe` is private
to `table`, so the stub must be a `#[cfg(kani)]` variant inside `table`/`probe`) or harnesses that
do not build the real `Table`. Astra's decision requires returning this as the cost blocker rather
than disabling checks or removing the path.

Next: a `cfg(kani)` bounded probe (the same algorithm with provably bounded loops) or an explicit
stub of the probe/reservation boundary, then Stage 3 harnesses, Stage 4 regeneration
(`proof/ApeironWord.lean`, `proof/regen.sh --check`), Stage 5 bridge only if regeneration moves
production definitions.

## Structural verdict (Astra)

The probe is extracted but not linked. `proof/ApeironWord.lean` contains `table.probe.spill_read`
(line 6471), `table.probe.spill_locate` (6263) and `table.Table.search_decision` (7662), so all
three are inside the extraction call graph; none has a bridge theorem establishing that the matcher
returns exactly the represented key's location or sound absence.
`Apeiron.GrowableSearch.{word_stored,word_regen}_search_decision_simulates`
(`proof/corpus/Growable/Search.lean:338,397`; conjunction 937) conclude about
`word_{stored,regen}_apply_get`, not about executing the classifier, and take
`SearchDecisionCorrect` as a premise. Canonical `Apeiron.Table.get` is `m k`
(`Apeiron/Table/Basic.lean:69`); `get_insertWith_self`/`get_insertWith_ne`/`get_build_acc`
specify abstract-map observations and not probing, fingerprints or summary exclusion. So
"unextracted" in `Growable/Search.lean:9-12` is stale, its statement that the decision premise is
unproved is correct, and the axiom-free extraction does not discharge the hypothesis. The upstream
claim that the probe read path is already proved unbounded is therefore qualified.

Decision: keep the real-probe Kani obligation. A stub, a separate `cfg(kani)` mirror, or retiring
the real-`Table` harnesses are each rejected on present evidence: a mirror proves the mirror unless
its correspondence is separately established, a stub is only a conditional caller check, and
recording lost coverage does not satisfy the gate. The timeout is a feasibility blocker, not a
reason to move sequential search outside the refinement boundary.

Immediate bounded task (Astra, after Sol ratifies the bounded domain and claims): establish
feasibility of directly checking the real probe helper on concretely sized, initialized `Region`
fixtures, without carrying whole allocation/growth/locator construction through every probe check,
in `src/table/probe.rs`, `src/proofs.rs`, `docs/evidence/kani-table-harness-audit.md`. Keep the
whole-`Table` obligations as blocked until separately covered; a passing helper subset is not the
full gate. Require reachable positive and negative cases and a real-source late-sweep mutation that
fails.

Universal refinement, independently required and named in `docs/epics/read-path-specialization.md`:
`Growable/Scan.lean` (source-linked summary/fingerprint exclusion facts),
`Growable/Representation.lean` (representation facts and their construction/preservation), and
`Growable/Search.lean` (prove the generated `spill_read` and both `word_{stored,regen}_table_get_at`
roots refine canonical `get`, and separately establish `SearchDecisionCorrect` for the generated
insertion classifier including `spill_locate`). These are missing goals, not existing theorem
names.

## Probe fixture harnesses (Sol-ratified, applied)

`src/table/probe.rs` now carries a `#[cfg(kani)] mod harnesses` that runs the **real** `spill_read`
body, the real `Region`, tags, summary and the real `Stored<i64>` full-key comparison, on a fixed
placement (so the probe order is deterministic) over a symbolic query key. Fixture: `n = 2`, four
slots, two groups; group 0 lane 0 a same-fingerprint live slot whose key differs and lane 1 vacant;
group 1 lane 0 the matching key and lane 1 a same-fingerprint live slot whose key differs;
placement `0` sends all prefix windows to group 0, so the match is only reached in the cyclic sweep.

Four harnesses, all green (Kani, 1.5-3.4 s each):
`spill_read_returns_the_late_matching_slot`,
`spill_read_reports_absence_without_a_full_key_match`,
`spill_read_returns_absence_on_a_summary_miss`,
`spill_read_sweeps_when_the_summary_is_unavailable`.

Negative control (throwaway copy, per AGENTS §3): the real source line
`while step < WINDOWS + groups && hit.is_none()` (`probe.rs:229`) mutated to
`while step < WINDOWS && hit.is_none()` makes `spill_read_returns_the_late_matching_slot` fail on
its own assertion in 2.5 s; baseline green, mutant red, working tree untouched, restored green.

Claim (grade `tested`/bounded), Sol-ratified: the real `spill_read`, instantiated at `Stored<i64>`,
`n = 2`, on the stated four-slot initialized regions, returns the unique late matching slot and
returns `None` for the stated absent queries, including summary-disabled execution, a summary miss,
and a fingerprint collision on every live lane. Non-claims: no whole-`Table` constructor, growth,
locator, reservation, relocation or resource result; no `search_decision`/`spill_locate`/`get_at`/
`apply_get` refinement; no `Regen` result; no four-shard/production equivalence; no universal
`SearchDecisionCorrect` or canonical `get` refinement; no `proved` grade. The `Table`
construction/preservation of the fixture's facts is a separate blocked obligation.

## Current gate

`cargo kani -j 16 --harness-timeout 120s`: **58/67 verified, 9 timeouts, no counterexamples**. The
9 are the whole-`Table` read/write/search harnesses; they remain enabled and blocked. Production is
green: `cargo test --workspace --all-features` rc 0, `cargo clippy -D warnings` rc 0,
`cargo fmt --all -- --check` rc 0.

## Insertion-locator fixture task

`spill_locate_classifies_a_fixed_initialized_region` directly calls the real `spill_locate` body
for `Stored<i64>`, `WordHome`, `Sigma::Parabolic`, `n = 2`, four slots and two groups. The regions
are created through `Region::reserve_empty(4)` and `initialize`; live slots are published only
through `publish_live`, which records the summary before the live tag. All keys and queries are
concrete, so no `kani::assume` is needed to inhabit the following branches:

- group 0 `[key 7, vacant]`, group 1 `[key 8, key 9]`, query `8`: `Found(2)`, after the earlier
  vacancy and in the cyclic coverage sweep;
- the same region, query `10`: `Vacant(1)`, after rejecting same-fingerprint full-key collisions;
- all four lanes live with keys unequal to `10`: `Grow`;
- the first region, key `8` queried under `Sigma::Elliptic`: `Vacant(1)`, because the stored
  `Sigma::Parabolic` payload is not a match.

Focused gate:

```sh
cargo kani --harness spill_locate_classifies_a_fixed_initialized_region \
  -Z unstable-options --harness-timeout 120s
```

Receipt: `src/table/probe.rs` SHA-256
`9cbf4ead2188957f2100833a8e84a4ba3505b25cd66b18a57834abd1a0a22a0f`; Kani `0.67.0`, CBMC
`6.8.0`. The focused baseline exited 0 with 803 checks, 23 unreachable and verification time
7.423595 s (`target/handoff-w2-t2/baseline-final.log`). In a disposable copy, changing the real
`spill_locate` outer bound from `step < WINDOWS + groups` to `step < WINDOWS` exited 1 with the
`Found(2)` assertion `an earlier vacancy must not hide the late full-key match` as the sole failed
check, 29 unreachable and verification time 5.7235403 s
(`target/handoff-w2-t2/mutant-final.log`). The unchanged restored source exited 0 with 803 checks,
23 unreachable and verification time 7.4622197 s
(`target/handoff-w2-t2/restored-final.log`). `cargo fmt --all -- --check` exited 0
(`target/handoff-w2-t2/fmt.log`). No timeout, compilation error, unwind failure or unreachable
assertion supplied the red result.

The claim is `tested`/bounded only for these exact initialized helper fixtures:
the real locator distinguishes a late full-key match, the first probe-order vacancy, exhaustion,
and unequal-`Sigma` identity. `reserve_empty`, `initialize` and `publish_live` make the fixture
representation-valid, but no result shows it reachable from `Table::new` followed by public
`insert_with`. `Region::reserve_empty(4)` executes the real fixture-helper allocation path, but the
harness establishes no whole-`Table` allocation/growth or allocator-realization/refusal result. It
does not execute table construction, locator publication, relocation, resource failure,
direct-region selection, `Table::search_decision`, `get_at`, `Regen`, production 256-shard
equivalence, canonical `get`, or universal `SearchDecisionCorrect`. The nine whole-`Table`
harnesses remain enabled and blocked.
