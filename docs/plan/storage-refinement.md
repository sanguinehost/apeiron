# Plan - Single Table Refinement

## Performance-preserving consolidation: lifecycle annex

Owner Astra; task `APEIRON-PERF-RESTORE-PLAN-W1-T1`; base
`c450c222905c050f793a23332708c230c658f82c`. Lead-authorized documentation exception in the supplied
checkout; no build lane, commit or production edit. Preserve existing dirty REG-1/spinor artifacts.
This annex governs current performance restoration and dispatch; earlier implementation sequences
below are historical checkpoints, not instructions to restart proofs of the current scan table.

```text
SCOPE: direction.md, flash-handoff.md, storage-refinement.md; performance-preserving consolidation
REGISTER: planning/measurement/refinement; no production implementation or speed claim
CORPUS: canonical Apeiron.Table/AnyRing; historical CountTable/Map at named revisions
REFERENCE: 71942db, 3a95bb5, 00a3629, dde9d58, f6db050; RFC 0001 A2/A7/G1
GATE: git diff --check; RFC renderer; performance/refinement requirement consistency search
GRADE: owner-approved requirement; historical numbers remain historical until rerun
```

### Intent

Fold/hashtrinity -> Apeiron consolidation must preserve performance as well as canonical meaning.
The proof subject is the high-performance executing implementation, not a slower proof-only
replacement or a reference alongside unproved production. Extraction convenience, sampled
SwissTable/hashbrown-style confidence and historical speed alone cannot select an accepted
primitive. Recover historical performance; do not resurrect deleted source as accepted production.
Fixed N, bounded callbacks and linear whole-source reading remain qualifications, not exemptions.
Non-goals: production changes in this task, new roadmap, identity/admission reduction, proof-time
claims presented as runtime, or a universal slowdown claim about fixed shards.

### Spec and evidence register

Owner definition of done (apeiron core): formally verified end to end (real-source extraction,
refinement to canonical `Apeiron.Table`/`AnyRing`, closed cone `{propext, Classical.choice,
Quot.sound}` with an empty cone for the Slang chain, mutation controls); competitive-or-SOTA speed;
not HashDOS-vulnerable; cryptographic strength not mandatory but achievable as a face. Full statement
in [direction](../direction.md#owner-definition-of-done-for-the-apeiron-core).

The [G1 matrix](../direction.md#required-r0-acceptance-g1--reg-1) is the contract: original spatial
parallel CountTable; serial operation/lookup cost; occupancy 1/2/4/8/16 at fixed `N = 16` with
direct-index-resident, offset, skewed and seeded-random coordinates; separate hit/miss/update/
insert/refusal; hashtrinity, hashbrown/foldhash and std baselines where historically applicable.
Keep identity (`10^8` colliding keys), round-trip sentinels, cross-process determinism, growth/remove,
ordered/commutative merge and Aeneas/Kani coverage. Missing operations and unmatched admission are
explicit open rows, not passes or silently narrowed populations. Timing requires full-key oracle
agreement first; default/new-first semantics, overflow, callbacks, effects and refusal remain explicit.

| Source / artifact | Registered role; grade and limits |
|---|---|
| `71942db` | Original spatial parallel CountTable versus packed hashtrinity: archival 3.16/10.27 ms. `--check history` reproduces `cargo bench --bench spatial_binning --offline` in an isolated `71942db` worktree with a pinned `4cc14471` Fold worktree and the 15 registered source digests; all 10 arms are oracle-checked before timing and 9 arms (18 cells at both sizes) are promoted to `measured` with the exact correctness-line digest. `hashtrinity_tuple` reproduces WRONG (off by 92 at 100k, 8236 at 1e6) and stays historical. i9-14900KF (32 threads) / rustc 1.97.0-nightly (17584a181 2026-04-13). No CountTable proof. |
| `3a95bb5` | Serial Map campaign, machine as above. `map_sweep` ns/key reproduced in an isolated `3a95bb5` worktree; it asserts no oracle, so those 16 rows stay historical. `probe_cost` instructions/lookup reproduced under `taskset -c 2 perf stat -e instructions:u -- target/release/examples/probe_cost`, three reps; presence oracle is the printed hit count `138800000` (every key found). Reproduced u64: apeiron 34.82, hashbrown+foldhash 35.14, hashbrown+fxhash 34.17, std 128.27; String: apeiron 93.35, hashbrown+foldhash 84.43, hashbrown+fxhash 92.34 (intervals in `matrix.json` `fresh.result`). The 7 probe cells are promoted; `fold_instr` has no independent oracle and stays historical. Not hashtrinity comparisons. |
| `00a3629`, `dde9d58`, `f6db050` | Supplied history: Map/Set/SafeTable deletion; CountTable plus 25 benchmark/example targets deletion; current fixed-shard scan Table installation, respectively. These revisions are retrieval inputs, not proof/performance acceptance. |
| `examples/reg1/{main,workload,correctness,performance,coverage}.rs`, `examples/reg1/coverage.json` | Existing uncommitted partial gate: identity/operation boundaries, correctness, determinism, full16 timing and coverage. Preserve and adapt, not replace with a one-workload optimization. |
| `examples/reg1/candidate.rs` | Benchmark-only cached-placement candidate; supplied 2.65–2.69x hashtrinity result is rejected. No production/refinement status. |
| Current REG-1 and verification receipts | Fresh strict full-profile receipt [`perf-remeasure-2026-09-19-post-spill.md`](../evidence/perf-remeasure-2026-09-19-post-spill.md), sha256 `946b5d40…`: HEADLINE `registered=2592 load_bearing=2268 applicable=1450 pass=387 fail=1063 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`, exit 1; spill absent-miss and growth-insert cells fixed (lifecycle annex below). Kani 448.58 s, 62/62 green, one unreachable, largest harness ~94.23 s; freshness red in 3.28 s on line-comment drift. Reported measurements, not rerun here; details and missing receipts in `docs/flash-handoff.md`. |

Historical numbers are hypotheses for a registered rerun, never copied absolute thresholds.
Acceptance threshold (owner decision): for every applicable cell, `selected/incumbent <= 1.0` against
**every** named incumbent, never behind and ideally ahead. Named incumbents:
`hashtrinity::{Table,CoordMap,FreqMap}` serial and packed-parallel (`count_par`/`get_each`),
`hashbrown+foldhash`, `hashbrown+fxhash`, `hashbrown+rayon`, `std` SipHash, `foldhash`/`fxhash`
par16, and — per [hash-comparison.md](../hash-comparison.md) — `xxHash` XXH3/XXH64, `wyhash`, `ahash`
and `fnv`. The five named hashers are load-bearing on the opaque-key exact map face in
`examples/reg1/acceptance.rs`; demotion to informational or omission rejects. Applicability is
registered per cell/incumbent pair before timing with a semantic or API reason; no family-wide
exemption; missing, duplicate, invalid or inconclusive pairs reject.
`--accept --profile smoke` is diagnostic only; full acceptance requires all registered pairs. This
supersedes and is stricter than the RFC elapsed-median floor `selected/hashtrinity <= 1.25`, and
replaces the former historical-envelope question: there is no separate envelope tolerance. Cells
where the incumbent has no corresponding path (for example refusal) are reported without the bound.
`docs/hash-comparison.md` carries the axes (crypto versus non-crypto, throughput
small/large, latency, collisions/avalanche, HashDOS resistance, determinism, `no_std`, licensing,
verification) and the per-domain "good enough" reading for OS, languages, provers, games, servers
and aerospace. Historical Apeiron values are direction evidence, not a second gate.

Current strict result: the fresh post-spill full-profile receipt
([`perf-remeasure-2026-09-19-post-spill.md`](../evidence/perf-remeasure-2026-09-19-post-spill.md),
sha256 `946b5d40…`) gives HEADLINE `registered=2592 load_bearing=2268 applicable=1450 pass=387
fail=1063 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`, exit 1.
Spill absent-miss cells fell ~15–21x and growth-insert ~145–152x; the worst cell moved to the spill
present-hit path (~30.9x). No speed recovery is claimed; G1/REG-1 remains OPEN. The current
performance dispatch is the re-anchored bounded order in
[programme-roadmap.md#performance-re-anchor-and-bounded-dispatch](programme-roadmap.md#performance-re-anchor-and-bounded-dispatch);
its W1-T1 (five hashers load-bearing) has landed.
Pre-register sample/repetition counts, timed region, thread/cache conditions and uncertainty rule;
report raw samples and relative intervals. Inconclusive or failed required cells keep G1 open, even
if an aggregate is fast. Use matched before/after machine, compiler/features, inputs and baseline;
record every unavoidable mismatch rather than pooling incompatible samples.

### Lineage: the performant structure against the algebra

The canonical denotation is `Apeiron.Table.build f d kvs = kvs.foldl (insertWith f) (fun _ => d)` with
`insertWith f k v m x = if x = k then f v (m x) else m x` (`proof/Apeiron/Table/Basic.lean:80,121`).
`get_build_acc` (`:134`) reads a key as the stream-order fold of `f` over that key's values.
`build_eq_of_key_streams_eq` (`:169`) licenses any regrouping that preserves each key's value
subsequence, for **every** merge; `build_gauge_invariance` (`:191`) licenses permutation only when
`f` is commutative and associative, and `build_gauge_invariance_mul` (`:224`) instantiates the
algebra's own `mul σ`.

The historical `CountTable` (`71942db:src/table.rs:91`, `3a95bb5:src/table.rs:103`) computes
`get a = |{i · addresses[i] = a}|`, which is exactly `build (fun new old => new + old) 0
(addresses.map (·, 1))` under the additive `i64` merge with default `0` (`src/merge.rs:110`). Its
shard-and-chunk schedule is `build_eq_of_key_streams_eq`: chunking is contiguous and regrouping is
by key, so each key's value subsequence is preserved and no commutativity is used. Region sizing,
the window descent and the tag line are placement, not semantics. The historical header's citation
is therefore correct, and the performant parallel structure is an instance of the canonical build.

The load-bearing premise is identity: the slot stores the *identity* word (`3a95bb5:src/table.rs:726`)
so slot equality equals key equality; `finish`/`spread` must be injective. That premise is asserted at
grade tested/bounded — `tests/the_counts_are_exact.rs` and the Kani lemmas cited at
`71942db:src/table.rs:36` — not `proved` in Lean.

Divergence: the current `Table` (`src/table.rs:6,74`) is a *bounded* 256xN fixed inline-shard
structure with `ShardFull`, total at most `256·N` keys (N=16 -> 4096). It neither holds the historical
1M-key workload nor shares the growable, load-sized structure. The general performant form
(`map`/`Map`) was deleted at `00a3629`; `CountTable` at `dde9d58`. No checked Lean refinement exists
for the counting/growable structure, for `spread`/`finish` injectivity, or for `locate` termination
(`region_slots_for`, `group_at` at `3a95bb5:src/table.rs:780,791` are exposed "for proof" but unproved
here); `proof/corpus/TableRefinement*.lean` refines the bounded fixed-shard Table only, and
`src/lib.rs:10` states that refinement pending.

Consequence: the equally-good solution is a growable, key-partitioned, cache-resident table generic
in `V` and merge `f`, whose `get` is proved equal to `get (build f d kvs)` under the injectivity
premise — the general `map` lineage generalized over `V`/`f`, not the bounded fixed-shard Table. It
is simultaneously the semantic refinement and the performance-bearing mechanism, so the choice does
not trade proof for speed. Selecting it replaces the current representation/ownership contract and
is a separately approved boundary decision.

### Selection: Astra boundary and Sol contract

Target: one growable, key-partitioned `Table` with load-sized home regions, tag-line scan and
window-descent probe, generic over `V` and `f`, superseding the bounded fixed-shard representation;
no second production mode. Every operation carries `Sigma`; `N` becomes the lanes examined by one
tag-group primitive, not keys per home, so no positive `N` imposes a `256·N` ceiling. The canonical
schedule licence `build_eq_of_key_streams_eq` is `proved` and sufficient; no canonical theorem or
attribution changes, and an implementation-specific scheduling lemma must supply its per-key
subsequence premise. The Rust representation-to-canonical refinement is open, grade `tested`/bounded.

Qualifications that must survive: cache residency is a workload-dependent target, not an invariant;
a fixed probe-group width does not establish constant time, so the current universal O(1)-at-fixed-`N`
claim does not transfer to the growable representation and must not be renamed onto it. Load-bearing
open obligations: extracted `finish`/`spread` identity-word injectivity, `locate` termination and
first-fit specification, allocation/ownership semantics, SIMD/SWAR scan equivalence, and thread/join
simulation are unproved; the real widened multiply `id = ((x as u128).wrapping_mul(PHI as u128)) as u64`
has no corpus link yet. `build_gauge_invariance` is unnecessary unless same-key values are reordered
or partial worker accumulators are recombined. The bridge `proof/corpus/lakefile.lean` pins canonical
`8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f`, not the dirty sibling base; no sibling theorem enters
without a separately gated pin update. Aeneas/Charon translation of allocation, ownership, SIMD and
threads is a feasibility barrier to probe before implementation expands. The acceptance threshold is
match-or-beat the applicable incumbent per cell (`selected/incumbent <= 1.0`, owner decision), not a
historical-envelope tolerance.

Measured mechanism (first-cut diagnostic, superseded by the spill/bitmap rework and the fresh
receipt): the fixed-16 loss was the conjunction of direct-index-resident coordinates, `absent-miss`
and high-occupancy `present-hit`, and the fixed 16-slot linear scan. Under clustered probing the
first-cut Table was faster than hashtrinity (0.10-0.53x); under seeded-random it was at parity
(1.0-1.46x); the direct-path `absent-miss` cell was the worst at 5.16-6.76x. The selected mechanism
resolves a direct-index coordinate in O(1) while retaining bounded probes on clustered input: the
historical `coord::Addr` identity/placement split plus window descent, not a linear scan. Current
strict run (post-spill receipt, sha256 `946b5d40…`): HEADLINE `registered=2592 load_bearing=2268
applicable=1450 pass=387 fail=1063 missing=0 concurrency_unmatched=2 address_face_unmatched=4
full_acceptance=false`, exit 1; spill absent-miss fell ~15–21x and growth-insert ~145–152x, and the
worst cell is now spill present-hit (~30.9x, ~24.7 ns/op). No speed recovery is claimed.

Extraction feasibility (spike, Charon `0.1.254` / Aeneas `505b6ca-dirty`): the `coord.spread`/`finish`
body translates to Lean with no `axiom`/`sorry` — the identity-word arithmetic is extractable, so a
proved descendant is reachable for that premise. Unsupported as written: `std::thread::scope` (HRTB
lifetime constraint in `scoped.rs`), `return` inside nested loops in `locate` (`71942db:src/table.rs`
709-738), and the `Held::Ref<'a>` GAT (`3a95bb5:src/map.rs:394`); `Vec` allocation/push translates.
Identity injectivity is not derivable from the extracted model (Aeneas/Std has no odd-multiply lemma):
it needs a hand-proved `ZMod 2^64`/`BitVec` odd-multiply injectivity theorem plus a refinement link
unfolding `coord.spread`. Consequence: extract the sequential per-worker bodies, hoist threading behind
an unextracted execution boundary, restructure `locate` for extraction, and avoid GATs in the interface.

Extraction readiness (spike, same tools): the rendered growable core is structurally extractable.
Extractable as-is: `Vec` region allocation/zero-init/indexing, the explicit-search-state `locate`
with no `return` inside nested loops, the SWAR tag scan, sequential `insert_into`/`build`, and the
real `coord::spread`/`finish`/`unspread`. Rewrite required: `usize::next_power_of_two` and
`{u64,u16,usize}::trailing_zeros` surface as `axiom`; replaced by bounded loops the restructured
`build` extracts with **zero axioms**. Boundary: the SSE2 intrinsics and raw-pointer ops
(`_mm_loadu_si128`, `_mm_cmpeq_epi8`, `_mm_set1_epi8`, `_mm_movemask_epi8`, `Slice::as_ptr`,
`const_ptr::cast`) surface as axioms; keep the axiom-free SWAR kernel as the extracted implementation
and carry SSE2 as an unextracted optimization with an equivalence obligation anchored at
`tests/both_tag_paths_agree.rs`. `std::thread::scope` and the `Held::Ref<'a>` GAT stay outside the
sequential core. Aeneas already lowers scalar `+ - *` to checked monadic operations, so no
unchecked-arithmetic rewrite is needed. Verdict: in a position to reimplement with proper Aeneas
integration.

Every performance claim carries source/harness/input SHA-256, resolved revisions/dependencies,
actual rustc/Charon/Aeneas/Lean/Mathlib identities, machine, command, metric and interval. Register
the actual selected implementation symbol/build profile, not just HEAD on a dirty tree. For each
claim run baseline/red/restored controls: wrong oracle/result digest; wrong workload identity
(including occupancy/threads); deliberately slowed executing path or selected-implementation
mismatch. Each must reject for the intended reason; corrupted sample metadata alone does not
establish sensitivity to runtime regression. Controls mutate throwaway copies, never production.

### Plan: serialized waves, not a new roadmap

Lead records integrated base and transfers file ownership before each dispatch; REG-1 paths remain
with their current owner until transfer. Use `worktree-agent-<task-id>` and exclusive R/E/B lanes
defined below. Routine harness/measurement tasks belong to Flash; Sol reviews proof/contract
sufficiency. A representation boundary exceeding three files requires Astra architecture
decomposition before implementation; routine measurement does not. Other AGENTS escalation
triggers, including governing-boundary changes, remain in force.

The existing Rust `reg1` executable remains the single aggregate gate: no Python/shell runner or
second gate executable. Its Rust history module invokes child `cargo`/git commands for historical
builds in isolated `/tmp/opencode` directories, never mutates the current or Fold working trees,
and removes every temporary worktree, including on failure. Source-file line limits remain binding.

| Wave / task | Owner; owned files (at most three per task) | Completion, one focused gate and control |
|---|---|---|
| `APEIRON-PERF-RESTORE-W1-T1` — first gate-first dispatch | Flash; edit `examples/reg1/main.rs`, create `examples/reg1/history.rs` and `examples/reg1/matrix.json` (exactly three files); isolated benchmark outputs | Restore/adapt the original spatial benchmark from `71942db` in temporary revision-pinned builds; register original serial targets from `3a95bb5` and every required matrix cell. No production optimization. Implement the history mode (currently MISSING) in the existing Rust gate: `cargo run --release --locked --example reg1 -- --check history --controls`; oracle-checked original spatial reproduction, exact source/command/digest receipts, matrix inventory and wrong-oracle/digest/workload controls. Exit 0 means framework valid; omitted matrix results remain MISSING and G1 false. |
| `APEIRON-PERF-RESTORE-W2-T1` — matrix module split | Flash; edit `examples/reg1/main.rs` and `examples/reg1/performance.rs`, create `examples/reg1/matrix.rs`; R | Move shared workload, measurement and matrix-cell machinery into `matrix.rs`; preserve every existing performance result and control while bringing `performance.rs` below 1,500 lines. Gate: the existing full performance command with controls. No matrix cell changes in this task. |
| `APEIRON-PERF-RESTORE-W2-T2` — current matrix adapter | Flash; `examples/reg1/matrix.rs`, `examples/reg1/performance.rs`, `examples/reg1/matrix.json`; R, after W2-T1 | Measure occupancy, coordinate and separated-operation strata; preserve correctness/coverage modules. Extend the existing full performance gate to enumerate registered cells and reject workload/digest mismatch. Every owned source file remains below 1,500 lines. Done as W2-T2 (160 strata cells measured in `strata.rs`) and W2-T3 (registration and enforcement, 3 new rejecting controls; `matrix.rs`, `performance.rs`, `matrix.json`). |
| `APEIRON-PERF-RESTORE-W2-T4` — threshold enforcement | Flash; `examples/reg1/matrix.rs`, `examples/reg1/performance.rs`, `examples/reg1/strata.rs` | Apply the owner threshold: each applicable cell passes only when `selected/incumbent <= 1.0`; report per-cell pass/fail and keep `reg1_pass=false` until every applicable cell passes. Gate: `--check performance --profile full --controls`. |
| `APEIRON-PERF-RESTORE-W3-T1` — historical/baseline campaign | Flash; `examples/reg1/history.rs`, `examples/reg1/matrix.json`, this plan, only after W2 releases `matrix.json`; isolated R | Complete historical serial and applicable hashbrown/foldhash/std adapters and matched before/after receipts. Extend W1's Rust history mode; gate `cargo run --release --locked --example reg1 -- --check history --controls`; run real slowed-path or implementation-mismatch control per claim. If dependency edits are needed, stop for a separate at-most-three-file manifest task. Full matrix acceptance mode is MISSING until implemented; measurement completion is not G1. Done: the gate exits 0 with 25 promoted cells (18 spatial, 7 serial probe), 20 historical, 2 missing, 26 boundary; 7 controls red at `child_exit=2`, two of them targeting promoted receipts; `history.rs` 1,498 lines, under the 1,500-line bound. `fold_instr` and the 16 `map_sweep` rows remain historical; the current `performance`/`correctness` boundary cells are unchanged. |
| Selection barrier | Lead selects from same-run matrix; Sol reviews domains, callbacks, admission and proposed canonical relation; no production edits | **Threshold (owner decision): match or beat every named applicable incumbent on every applicable cell, `selected/incumbent <= 1.0` (see the lifecycle annex for the incumbent list and per-pair applicability); no family-wide exemption, no historical-envelope tolerance.** Registered comparisons identify the mechanism (direct-index O(1) plus bounded clustered probes) and the selection is the growable key-partitioned structure above, with the Astra boundary and Sol contract recorded. Blocking G1: the 2 missing rows (identity 10^8, coverage census) and the current applicable cells above 1.0. No representation migration before this evidence. |
| Production recovery, then extraction/refinement | Flash implementation; Sol proof/contract review; separate serialized tasks each at most three named files, assigned after selection | Selected real build path, then `proof/regen.sh --check`, then `lake build` in the bridge as separate focused tasks. Real constructor witnesses, invariant preservation, successful/refusing branches and canonical `Apeiron.Table`/`AnyRing` denotation; fail-closed axiom audits and source/generated mutation controls. Unsupported Aeneas constructs require a bounded tool/refinement task, not a slower substitute. |
| Joint terminal barrier | Lead; no new implementation ownership | Same selected source identity passes the complete performance matrix, canonical extraction/refinement, S1–S8 and all AGENTS §6 integration gates and controls. No proof-only or speed-only closure. |

Existing partial commands (root) remain evidence inputs, not all-row acceptance:
`cargo run --release --locked --example reg1 -- --check correctness --profile smoke --controls`,
`cargo run --release --locked --example reg1 -- --check performance --profile full --controls`,
and `cargo run --release --locked --example reg1 -- --check coverage --controls`.
The all-row matrix acceptance command remains **MISSING**; the threshold is the owner's match-or-beat
rule above, not a historical-envelope tolerance.
No performance harness deletion without a replacement gate retaining its useful tested surface.

**Terminal acceptance:** the selected implementation is on the real build path; its actual Aeneas
extraction and canonical refinement close under the allowed axiom cone; the performance matrix is
green on that same source identity with matched before/after and rejecting controls. An unsupported
operation/admission row cannot disappear into a faster workload. Historical proofs stay attributed
to historical bodies. The current boundary document describes the executing fixed-shard layout;
changing its representation/ownership contract requires a separately approved boundary task.
Source refinement does not cover rustc/LLVM codegen, allocator/unsafe/SIMD/thread realization,
kernel ABI or GPU merely by association. Any selected implementation using them needs explicit
simulation obligations and external trust inventory; no desired simulation may be assumed.
All Lean gates use AGENTS §6's serialized, bounded wrapper; Kani/Lean latency is not runtime speed.

## Fixed-shard representation: current boundary and evidence

Owner Astra; base `d3423b2ec3dea6ddd0a8a668381d7749ecfdfff4`, initially clean main.
The lead assigns this supplied checkout, representation implementation, build/test acceptance,
per-body extraction verdicts and a building main commit/push. The unrelated `one-allocation`
branch is untouched. This section and `docs/kernel-boundary.md` supersede the growable contract
below and the obsolete storage statements in other documents; no epic closure is claimed.

**Intent/spec:** one executing `Table<K,V,H,N>` with 256 inline optional shards, `N` optional
full-key entries each, default `N = 16`. Checked `ShardFull` replaces allocator failure, not total
lookup. `Home<K>`/`Merge<V>` have concrete word instances; adapters call the same generic methods.
Build borrows a slice and clones entries in input order. Exact admission, representation relation,
callback premises, rejected alternatives and stack/translation trust cost are in the boundary.
`Home.lean` does not specify shard capacity, allocation or Rust simulation. Its collision facts do
not license discarding keys. Fixed capacity is an explicit resource restriction, not a reduced
statement of `HMap` or an alternative table outside the executing path.

Three serialized owned slices, IDs `APEIRON-TABLE-W8-T1` through `T3`, owner Astra:
`src/table.rs`, `src/merge.rs`, `tests/table.rs` (R, build/test);
`proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` (E, regeneration);
the two governing documents (no build lane). They share the authorized representation commit.

```text
SCOPE:      the three slices above; fixed inline storage, real-body extraction, bounded proof plan
REGISTER:   representation/extraction/test/planning; excludes completed Rust-to-HMap refinement
CORPUS:     Apeiron.Table "get_insertWith_self", "get_build_acc", "get_build_append"; native proved
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean
GATE:       cargo build/test --workspace --all-features --locked; proof/regen.sh --check (root)
            supplied red: capacity/try_reserve_exact/TryReserveError axioms after arrow removal
GRADE:      tested representation; clean extraction/elaboration, not proved table refinement
```

### Open obligations: stable Sigma/address interface and the executing path

The fixed-shard `Table<K, V, H, N>` and its word refinement are the current evidence. The following
obligations are open; each names a gap to the O(1)-at-fixed-N canonical object, not a completed
result.

1. **Stable Sigma/address interface (open).** A read or write takes `(sigma, face, admittedInput)`.
   The stored full key carries `sigma`. `Apeiron.Table.get` stays total: a missing key returns the
   default, never a new refusal. No new refusal is introduced under `ShardFull`; the only refusals
   remain the absent-key full-shard refusal and the `N = 0` write refusal. The present
   `Table<K, V, H, N>` signatures take no `Sigma`, so this interface is not implemented.
2. **Face-routing of `WordHome` (open).** `WordHome` routes by the word bit pattern only and selects
   no face. Routing each face's admitted input to its home shard — elliptic pack, parabolic carried
   complete key, hyperbolic observation domain — is not implemented.
3. **Real `Cell` multiplication merge (open).** `WordMerge` is wrapping addition. A merge that calls
   the real `Cell::mul` at the selected `sigma` and refines `Apeiron.Table.insertWith` is not wired
   into the executing table.
4. **Fixed-capacity cost model (open).** The cost boundary at fixed `N`, bounded callbacks and
   nonconstant key length is stated in the spec. No cost model or measurement binds the executing
   path.
5. **Hyperbolic observation domain (open).** The discard face's observation domain, and its refusal
   of identity outside that domain, are not defined in the executing path.

Each obligation is named open; none is closed by the existing word-table refinement.

### Observed gates and extraction verdict

| Command | Exit/result |
|---|---|
| `cargo build --workspace --all-features --locked` | 0 |
| `cargo test --workspace --all-features --locked` | 0; 94 tests and 2 doctests |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 |
| `cargo fmt --all -- --check` | 0 |
| `proof/regen.sh`; `proof/regen.sh --check` | 0 each; fresh, no `axiom`/`sorry`/`admit`, 1708-job build |
| `lake build` in `proof/corpus` | 1; obsolete `TableRefinement.lean:56:28` error below; WordRefinement builds |

Per-body command, root cwd, `b` each of `new get insert build`, existing writable `$out`:

```sh
target/aeneas/charon rustc --preset aeneas --dest-file "$out/$b.llbc" \
  --start-from "crate::word_table_$b" -- --edition 2021 --crate-type lib \
  --crate-name apeiron_word proof/extract/lib.rs
target/aeneas/aeneas -backend lean -dest "$out" -warnings-as-errors \
  -no-progress-bar "$out/$b.llbc"
```

Each Charon and Aeneas exit is 0; each generated file has the real method `def`, zero forbidden
declarations and passes `lake env lean <absolute-generated-file>` from `proof/` (exit 0).
Verbatim Aeneas output excerpts, ANSI color removed:

```text
[Info ] Generated: /tmp/opencode/apeiron-fixed-verdicts/New.lean
[Info ] Generated: /tmp/opencode/apeiron-fixed-verdicts/Get.lean
[Info ] Generated: /tmp/opencode/apeiron-fixed-verdicts/Insert.lean
[Info ] Generated: /tmp/opencode/apeiron-fixed-verdicts/Build.lean
```

Full local per-body logs: `/tmp/opencode/apeiron-fixed-verdicts/{new,get,insert,build}-{charon,aeneas,lean}.log`.
No remaining unsupported construct in these roots. Trials exposed nested const-repeat failure,
early return inside an outer optional-shard match, then loop-context joining failure. The final
source uses a single const-repeat per construction, an isolated array search loop, and no local
`table` identifier shadowing the generated namespace. `proof/regen.sh` strips backend-emitted
trailing blanks identically during regeneration and freshness checks; it does not rewrite Lean.

Control: `python3 /tmp/opencode/control-fixed-table.py` compiles throwaway `src/table.rs` copies
with `rustc --edition=2021 --test`, then runs them. Baseline 0, replace both fresh-write merges
with `let merged = value;` 101, restored 0. Verbatim failing assertion: `left: 3`, `right: 13` in
`full_shard_refuses_unchanged_but_existing_key_still_updates`. This is a Rust test control, not a
refinement control. The temporary runner is evidence only; the durable control task remains below.

### Source/tool identity

Corpus read/resolved: `50c94192399add3b9f8e64bd14e36d8cba270705`;
Mathlib manifest: `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`; Lean `4.31.0`.
Rust tests use `rustc 1.97.0-nightly (17584a181 2026-04-13)`; Charon extraction uses
`nightly-2026-08-18`. Charon reports `0.1.254 (b104e24fea7d721b71e6c39fd70f26ff20bc0980)`;
Aeneas reports `aeneas 505b6ca-dirty`. Existing const-repeat patches, not an unmodified release:

| Artifact | SHA-256 |
|---|---|
| `src/table.rs` | `f30e31e46c18680283d0333ff04a679d519e904cf0958e86763372a639b074e3` |
| `src/merge.rs` | `7d4ce7656b38da200ba9c5265eca972e7cc475f133ae03efb6a1fa24c728c498` |
| `tests/table.rs` | `177ff01a04bc731f2ff7ffe6a6e1845e156bfdd3a29de95dcbfd3ef5d7f1a207` |
| `proof/extract/lib.rs` | `a2ecf3b1a01ca4b77fc575ff21254c0f1f3fe1e7fdc83532b110c996ab051f4e` |
| `proof/regen.sh` | `1625b0147f458f709fd46f7504a0e41258d0660b79fd60fee00f7086e783dc2d` |
| `proof/ApeironWord.lean` | `2237cf4040ce7ecafa5f528afab1e9539cc0b922d8043f31adfc2e56a8fb43dd` |
| `target/aeneas/charon` | `747e165bc62bb963f8d741e8c193c721a4b94555bcb1f9f86c79090fd7a59d0e` |
| `target/aeneas/charon-driver` | `207b223af35cf9a05227b329819b73532e94f655ccdefd64e77b755a15fdf297` |
| `target/aeneas/aeneas` | `33d0ce1a1cc2f1d3935c0cd2ab34f283797da61c531ed6d4e09d5c97b0d0228a` |

Source `table.rs` is 246 lines; generated `ApeironWord.lean` 991. This is not a complete trusted
perimeter inventory. No performance measurement or new proved table theorem is claimed.

### Remaining serial waves

IDs `APEIRON-TABLE-W<number>-T1`. Lead records integrated base and creates
`worktree-agent-<id>` before dispatch. **R** = `target/`; **E** = R + generated extraction +
`proof/.lake/`; **B** = `proof/corpus/.lake/` + `proof/.lake/` + backend cache. Exclusive lanes;
controls use throwaway files and separate writable outputs. All proofs quantify over `N`, with
explicit pure/total, equality-respecting home, faithful equality/clone and merge premises.

| Wave | Owner; lane | Owned files (≤3) / result | One gate | Rejecting control |
|---|---|---|---|---|
| 9 | Sol; B | `proof/corpus/TableRefinement.lean`, `proof/corpus/AxiomAudit.lean`: replace obsolete error witnesses, define denotation/Rep, prove real constructor initialization and joint default/zero-capacity refusal witnesses | `lake build` in `proof/corpus` | Mutate generated constructor default; unchanged constructor/default witness must fail. No invented reservation axiom or deleted canonical contract. |
| 10 | Sol; B | `proof/corpus/TableReadRefinement.lean`, `proof/corpus/TableRefinement.lean`, `proof/corpus/AxiomAudit.lean`: terminating `find_slots` invariant at measure `N-i`, routing, get = canonical get | `lake build` in `proof/corpus` | Generated search always misses; present-key read theorem fails. |
| 11 | Sol; B | `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/TableReadRefinement.lean`, `proof/corpus/AxiomAudit.lean`: exact admission/refusal, successful insertWith, Rep preservation; N=0, collision/full and full-existing witnesses | `lake build` in `proof/corpus` | Bypass `f new default`, or refuse an existing key in a full shard; unchanged theorem/witness fails. |
| 12 | Sol; B | `proof/corpus/TableBuildRefinement.lean`, `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/AxiomAudit.lean`: exact per-shard distinct-key admission, ordered build refinement and canonical order/gauge corollaries | `lake build` in `proof/corpus` | Reverse same-key stream; replace/append control fails. Gauge corollary retains commutativity/associativity. |
| 13 | Flash; E+B | `proof/storage-controls.py`, this plan: durable fail-closed baseline/red/restored runner for the preceding source/generated controls; runner present (W13-T2/T4, W14-T3), twelve encoded controls with name-based locators | `python3 proof/storage-controls.py --jobs 3` | Runner must reject a surviving mutation and missing tools. |
| 14 | Flash/Sol; B/E | split the generic admission module (W14-T1); discharge the abstract callback contract (W14-T2/T5); persist the toolchain patches (W14-T4); name-based control runner (W14-T3) | `lake build` in `proof/corpus`; `proof/install-patches.sh --check`; `python3 proof/storage-controls.py --jobs 3` | Each slice's throwaway mutation red at a named declaration; one-byte patch mutation red. |
| 15 | Flash; B/E | run the epic barrier (W15-T1); real `cfg(kani)` harnesses in `src/proofs.rs` (W15-T2) | §6 barrier set | `Cell::add` → `sub` fails `word_cell_add_matches_native_wrapping`. |

### Generic-N refinement slices (waves 9-11 progress)

Waves 9-10 are integrated at `1403767e671c81737525702946a3eb1eae4f0b0b` (`TableRefinement.lean` carries
the constructor/read/refusal witnesses at that commit). Wave 11 is discharged serially in three
slices on top of it; the planned `TableWriteRefinement.lean`/`TableReadRefinement.lean` module split
was not used, and the obligations landed in `TableRefinement.lean`. Wave 12 (build) is not started.

Owner Sol; lane B; owned files `proof/corpus/TableRefinement.lean`,
`proof/corpus/AxiomAudit.lean`.

| Slice | Obligation | Headline results | Gate |
|---|---|---|---|
| `APEIRON-TABLE-W11-T1` | generic-`N` present-key write preservation | `insert_with_present_exact`, `stateRead_replacePresent`, `insert_with_present_preserves`, `insert_with_present_rep_preservation`, `insert_with_present_joint_witness` | `lake build`, cwd `proof/corpus`, exit 0, 3358 jobs |
| `APEIRON-TABLE-W11-T2` | generic-`N` executing read refinement | `find_slots_spec_generic`, `word_shard_index_spec`, `generatedRoute_exists`, `generatedLocator_routes`, `table_get_eq_stateRead_of_route`, `table_get_eq_stateRead`, `functionalRep_iff_readRep_of_routes`, `generatedFunctionalRep_iff_readRep`, `readRep_iff_Rep` | same |
| `APEIRON-TABLE-W11-T3` | generic-`N` fresh-key insertion and vacant-slot loop | `stateRead_insertVacant`, `stateRead_insertEmptyShard`, `insert_with_empty_shard_exact`, `insert_with_loop0_vacant_body_exact`, `insert_with_loop0_vacant_spec`, `insert_with_loop0_full_spec_generic`, `insert_with_vacant_exact`, `insert_with_vacant_rep_preservation`, `insert_with_vacant_readRep_preservation`, `insert_with_absent_joint_witness` | same |

`stateRead` is the total physical-state denotation at a locator; `FunctionalRep` is extensional
equality with it. A state-independent locator is unsound because the generated `find` searches
`t.shards`, so the read relation is state-indexed: `ReadRep t h` states the generated `get` result
directly, and `readRep_iff_Rep` shows it equals the existing extensional `Rep` at `N = 16` in both
directions. `generatedLocator t` is derived from the generated `shard_index`/`find` through
`generatedRoute_exists`, so `table_get_eq_stateRead` is total over every generated state and key.

Controls: mutate the generated present-key branch to store `value`; `insert_with_present_exact` fails
at `TableRefinement.lean:630`. Mutate generated `get`'s present branch to return `self.default`;
`table_get_eq_stateRead_of_route` fails at `TableRefinement.lean:963` with goal `t.default = value`.
Mutate the throwaway generated `insert_with_loop0.body` vacant branch to store `value` instead of
`wrapping_add value default`; `insert_with_loop0_vacant_body_exact` fails at
`TableRefinement.lean:1240:69`. Restore builds green (3358 jobs). Every new headline `#audit`s inside
`{propext, Classical.choice, Quot.sound}`; no forbidden token. Source, generated file and pins
unchanged: `src/table.rs` `83597ecf...`, `proof/ApeironWord.lean` `958b1ab5...`.
`TableRefinement.lean` `fcf322ccb1c743097105514bfef422c03ee9d58ee690512a9cf0d3760b1a9100` (587 lines),
`TableRefinementGeneric.lean` `5db832f56256a22efcd0455dcba73f6e6fdb03d4a953c37f9d4d19481f306720` (1319
lines), `TableRefinementBuild.lean` `7a8121d7ad7c1e64459f24e3dc09418837b8ccf6366a583bb21ba122daccbbbb`
(1377 lines), `TableRefinementAdmission.lean`
`1a6146c728c15660feab1594d56e07fa7b91636341d2c91b762874863422cd42` (1345 lines), `AxiomAudit.lean`
`a8e5d11a5fadef761611d222423a0f62abc234913b77a9180e8a5672040e96f9` (249 lines).

Module split (`APEIRON-TABLE-W11-T4`): the generic-`N` material moved from `TableRefinement.lean` to
`TableRefinementGeneric.lean` under the same namespace (audit names unchanged), with
`TableRefinementGeneric` added to `lakefile.lean` roots and imported by `AxiomAudit.lean`.
`lakefile.lean` `09c2fb38f143a2ee9eaee9485800bdd0e220cd6bc500512481b6b18d84f2a88b`. Gate `lake build`,
cwd `proof/corpus`, exit 0, 3359 jobs. Control: `#check Apeiron.TableRefinement.insert_with_present_exact`
fails `lean.unknownIdentifier` with only `import TableRefinement` (exit 1) and resolves with
`import TableRefinementGeneric` (exit 0).

Unconditional post-state routing (`APEIRON-TABLE-W11-T5`): `word_shard_index_state_independent`,
`find_slots_eq_of_matches`, `find_slots_insertVacant_ne`, `find_slots_insertVacant_self`,
`table_find_insertVacant_ne`, `table_find_replacePresent`, `routes_replacePresent` and
`routes_insertVacant` show the exact generated updates preserve generated routing;
`insert_with_present_readRep_preservation` and `insert_with_vacant_readRep_preservation_routed` then
derive `ReadRep` from `Routes t locate` alone, with no post-state routing premise. Gate `lake build`,
cwd `proof/corpus`, exit 0, 3359 jobs; cone `{propext, Classical.choice, Quot.sound}`. Control:
generated `insert_with_loop0.body` vacant branch `if b` → `if false`; red at
`TableRefinementGeneric.lean:1062:17`; restore green. No forbidden token.

Ordered build refinement, first slice (`APEIRON-TABLE-W12-T1`): `proof/corpus/TableRefinementBuild.lean`
defines `sliceEntries`, `orderedFold` (`orderedFold_eq_build` shows it is `Apeiron.Table.build`
definitionally) and the per-step physical predicate `BuildInsertPremises`;
`build_insert_premises_success` discharges one real `insert_with` and preserves `ReadRep`;
`table_build_loop_result_readRep` and `table_build_loop_readRep` give the `build_loop`
`loop.spec_decr_nat` result (Ok returns the ordered fold, Err is `ShardFull`); `word_table_build_readRep`
composes with the real empty constructor. `TableRefinementBuild` added to `lakefile.lean` roots and
imported by `AxiomAudit.lean`. Gate `lake build`, cwd `proof/corpus`, exit 0, 3360 jobs; cone
`{propext, Classical.choice, Quot.sound}`. Control: generated `build_loop.body` guard `i < i1` →
`i.val + 1 < i1.val` skips the last element; red at `TableRefinementBuild.lean:163`; restore green.

Physical invariant and unconditional build result (`APEIRON-TABLE-W12-T2`): `TableInv` states that
every populated shard has an occupied prefix and vacant suffix (`OccupiedPrefix`);
`emptyWordTable_tableInv` and `tableInv_insert_with` establish initialization and preservation by the
real generated `insert_with` in every branch (present, vacant, untouched, and full-shard refusal via
`insert_with_full_exact`). `buildInsertResult_of_readRep` and `allBuildInsertResults` expose the
Err-allowing per-step result, and `word_table_build_result_readRep` is unconditional:
`word_table_build default kvs = ok (Err ShardFull) ∨ ∃ final, = ok (Ok final) ∧ ReadRep final
(Apeiron.Table.build core.num.I64.wrapping_add default (sliceEntries kvs))`. Gate `lake build`, cwd
`proof/corpus`, exit 0, 3360 jobs; cone `{propext, Classical.choice, Quot.sound}`. Control: generated
`insert_with_loop0.body` writes at `i + 1` instead of the first vacant slot; `tableInv_insert_with`
fails to rebuild; restore green.

Per-shard capacity admission (`APEIRON-TABLE-W12-T3`): `PrefixShardCapacity` is an explicit
physical-state capacity contract (every occupied-prefix witness of a populated shard is shorter than
16); `allBuildInsertsAdmitted_of_prefixShardCapacity` derives `AllBuildInsertsAdmitted`, and
`word_table_build_readRep_of_prefixShardCapacity` gives the Ok-only canonical-fold refinement. Gate
`lake build`, cwd `proof/corpus`, exit 0, 3360 jobs; cone `{propext, Classical.choice, Quot.sound}`.
Control: generated present-key-as-absent mutation turns the build red (the first failure is the
pre-existing present-key theorem at `TableRefinement.lean:220`, so localization to a new headline was
not achieved); restore green.

Physical no-duplicate invariant (`APEIRON-TABLE-W12-T4`): `ShardKeysDistinct`/`TableInv2` require
pairwise-distinct keys in each populated shard's occupied prefix; `emptyWordTable_tableInv2` and
`tableInv2_insert_with` establish initialization and preservation across every generated branch;
`occupiedPrefix_distinct_key_count` counts them. `DistinctKeysPerShard` is the stream hypothesis;
`prefixShardCapacity_of_distinctKeysPerShard` and `word_table_build_readRep_of_distinctKeysPerShard`
close the capacity bridge conditional on `PrefixShardKeyCoverage`. Gate `lake build`, cwd
`proof/corpus`, exit 0, 3360 jobs; cone `{propext, Classical.choice, Quot.sound}`. Control: generated
present branch writes at the first vacant slot instead of the matching slot (duplicate key);
`tableInv2_insert_with` fails at `TableRefinementBuild.lean:1013`; restore green.

Stream-capacity admission (`APEIRON-TABLE-W12-T5`): `proof/corpus/TableRefinementAdmission.lean`
proves the Ok-only canonical-fold build from the stream hypothesis `DistinctKeysPerShard` alone.
`ReachablePrefixCoverage` is indexed by the carried state and prefix (not quantified over arbitrary
`ReadRep` representatives); `emptyWordTable_reachablePrefixCoverage` and `reachablePrefix_insert`
carry it through the real generated `insert_with`; `reachablePrefixCoverage_capacity` bounds each
shard load by the distinct routed stream keys; and
`word_table_build_readRep_of_distinctKeysPerShard_alone` composes with the real constructor. This
supersedes the `PrefixShardCapacity`/`PrefixShardKeyCoverage` chain (those declarations remain). Gate
`lake build`, cwd `proof/corpus`, exit 0, 3361 jobs; cone `{propext, Classical.choice, Quot.sound}`.
Control: generated present-as-vacant duplicate mutation turns the build red (first failure
`tableInv2_insert_with` at `TableRefinementBuild.lean:1013`, upstream of the new module); restore
green.

Exact admission characterization (`APEIRON-TABLE-W12-T6`/`T7`): `ReachablePrefixExact` strengthens the
loop coverage to a bidirectional locator↔routed-keys equivalence (`emptyWordTable_reachablePrefixExact`,
`reachablePrefixExact_insert`). `BuildAdmissible` is the tight per-step stream condition (the next key
is already present in its routed shard, or that shard holds fewer than 16 distinct prefix keys).
`word_table_build_readRep_of_buildAdmissible` gives the Ok branch; `buildAdmissible_of_distinctKeysPerShard`
shows the earlier strict condition is sufficient but stronger than necessary;
`word_table_build_success_implies_buildAdmissible`, `word_table_build_refuses_of_not_buildAdmissible`
and `word_table_build_refuses_iff_not_buildAdmissible` close
`word_table_build default kvs = ok (Err ShardFull) ↔ ¬ BuildAdmissible default kvs`. Gate `lake build`,
cwd `proof/corpus`, exit 0, 3361 jobs; cone `{propext, Classical.choice, Quot.sound}`. Control:
generated full-shard Err branch mutated to `Ok` with unchanged state; red at
`TableRefinementAdmission.lean:1018`; restore green.

Generic-`N` build and durable controls (`APEIRON-TABLE-W13-T1`/`T2`). W13-T1 adds
`proof/corpus/TableRefinementBuildGeneric.lean` (1090 lines): the ordered build refinement at general
`N` with the concrete word callbacks — `build_insert_premises_successN`, `buildInsertResult_of_readRepN`,
`table_build_loop_result_readRepN` (unconditional `Err ShardFull ∨ ∃ final, Ok final ∧ ReadRep …`),
`table_build_result_readRepN`, `table_build_readRepN`, `table_build_readRepN_of_distinctKeysPerShardN` —
plus an abstract-callback loop layer whose one-step contract `BuildInsertResultG`/`BuildInsertOkG` and
clone identities are explicit hypotheses (`table_build_loop_result_readRepG`, `table_build_loop_readRepG`).
Gate `lake build`, cwd `proof/corpus`, exit 0, 3362 jobs; cone `{propext, Classical.choice, Quot.sound}`.
Control: `BuildInsertResultN`'s `Ok` branch mutated to `ReadRep result.2 h`; red at
`TableRefinementBuildGeneric.lean:497:12` (`table_build_loop_result_readRepN`); restore green. W13-T2
adds `proof/storage-controls.py` (sha256 `16d6d4289b32ea8c872f3e81e6caf5f51c6f14c21c8547962c2907b0ba92b854`):
a stdlib-only runner with `--list`, `--only`, `--jobs`, `--self-test` and 7 encoded generated-body
controls (`w11-t1`, `w11-t2`, `w11-t3`, `w11-t5`, `w12-t1`, `w12-t6`, `s6-fresh-write`); each copies
`proof/corpus`, applies one exact-string mutation, asserts the first `lake build` error locator,
restores and re-asserts green. Gate: `python3 -m py_compile` exit 0, `--list` exit 0 (7 controls),
`--self-test` exit 0 (10/10), `--only w11-t5` red at `TableRefinementGeneric.lean:1062:17` then green,
`--jobs 3` 7/7. The repository tree and owned source digests are unchanged by the runner.

Exact admission at general `N` and the durable runner (`APEIRON-TABLE-W13-T3`/`T4`). W13-T3 adds
`proof/corpus/TableRefinementAdmissionGeneric.lean` (1500 lines, the file-size limit):
`ReachablePrefixCoverageN`, `ReachablePrefixExactN`, `BuildAdmissibleN`,
`emptyWordTableN_reachablePrefixExactN`, `reachablePrefixExact_insertN`,
`reachablePrefixExact_admissible_of_insert_successN`, `buildAdmissibleN_of_distinctKeysPerShardN`,
`table_build_readRepN_of_buildAdmissibleN`, `table_build_success_implies_buildAdmissibleN`,
`table_build_refusesN_of_not_buildAdmissibleN`, and `table_build_refusesN_iff_not_buildAdmissibleN`
(`wordBuildN default kvs = ok (Err ShardFull) ↔ ¬ BuildAdmissibleN default kvs`, `0 < N`). Gate
`lake build`, cwd `proof/corpus`, exit 0, 3363 jobs; cone `{propext, Classical.choice, Quot.sound}`.
Control: generated full-shard `Err` → `Ok ()`; the whole-package build fails first at
`TableRefinement.lean:517` (`insert_full_loop_spec`), so the new headline is falsified transitively,
not with a new-file red line. W13-T4 hardens `proof/storage-controls.py` (917 lines, sha256
`ffde2f60e96de3b432cece34f9b1c4eb906b78dbed6181d153aba6ae8c626213`): nine encoded generated-body
controls (`w11-t1`, `w11-t2`, `w11-t3`, `w11-t5`, `w12-t1`, `w12-t2`, `w12-t3`, `w12-t6`,
`s6-fresh-write`), each a whole-package `lake build` in a throwaway copy with a current recorded
locator (no `(stale)`), plus `--list`/`--only`/`--jobs`/`--self-test`; self-test 15/15 including
missing-tool rejection (exit 3). Gate: `py_compile` exit 0, `--list` exit 0, `--self-test` exit 0,
`--jobs 3` 9/9. Repository tree and source digests unchanged by the runner.

Contract characterization, module split, runner hardening, toolchain patches (`APEIRON-TABLE-W14-T1`…`T5`).
W14-T1 split the 1500-line `TableRefinementAdmissionGeneric.lean` into
`TableRefinementReachablePrefixExactN.lean` (1271 lines) and `TableRefinementAdmissionGeneric.lean`
(247 lines), byte-for-byte reconstruction verified; gate exit 0, 3364 jobs. W14-T2/W14-T5 discharged
the abstract callback contract: `cloneI64_identity`, `buildInsertResultG_word_iff`,
`buildInsertResultG_word`, `buildInsertOkG_word`, `table_build_loop_result_readRepG_word`,
`table_build_loop_readRepG_word`, `WordBuildOneStepContractN`, `wordBuildOneStepContractN_iff_concrete`,
`wordBuildOneStepContractN_sufficient`, plus the Aeneas `wpSpec_bind_iff` and the necessity
characterization `BuildLoopSpecG`, `BuildInsertSpecG`, `AllBuildInsertSpecsG`,
`buildLoopSpecG_iff_allBuildInsertSpecsG`, `buildLoopSpecG_true_iff_buildInsertResultG`,
`buildLoopSpecG_false_iff_buildInsertOkG`; `TableRefinementBuildGeneric.lean` 1468 lines. W14-T4
persisted the toolchain patches: `proof/patches/{charon-repeat-optional-trait.patch,
aeneas-array-repeat.patch,PINS}`, `proof/install-patches.sh`, a `proof/README.md` pin section; gate
`proof/install-patches.sh --check --aeneas-src /tmp/opencode/aeneas-src` exit 0, one-byte patch
mutation red. W14-T3 hardened `proof/storage-controls.py` (1149 lines, sha256
`06079130309505b8df59ad6e732e8e55969592fbb8c5e2480251e50588701914`): name-based locators, twelve
encoded controls, `w12-t5` a decided rejection, and two in-file red lines (`w14-admission-generic` in
`TableRefinementAdmissionGeneric.lean`, `w14-reachable-exact` in `TableRefinementReachablePrefixExactN.lean`);
`--jobs 3` 12/12, `--self-test` 34/34.

Epic barrier (`APEIRON-TABLE-W15-T1`/`T2`). All ten §6 commands exit 0: `git diff --check`,
`cargo test --workspace --all-features --locked` (96 passed, 0 failed), `cargo clippy … -D warnings`,
`cargo fmt --all -- --check`, `cargo kani`, `proof/regen.sh --check` (1708 jobs), `lake build` in
`proof/` (1708) and `proof/corpus/` (3364), `proof/install-patches.sh --check` (patches apply to the
pinned bases), `python3 proof/storage-controls.py --jobs 3` (12/12). The barrier mutated nothing:
`git status --porcelain` and `sha256(proof/ApeironWord.lean)` (`958b1ab5…`) identical before and after.
W15-T2 replaced the vacuous `cargo kani` gate: `src/proofs.rs` (958 lines, sha256
`3be901668aee40abd04a1a1f5d99b140663f02a76d659aa7ea3a672692b464c5`) now carries 57 bounded
`#[kani::proof]` harnesses (`Complete - 57 successfully verified harnesses, 0 failures, 57 total`);
the control mutating `Cell::add` to `sub` fails `word_cell_add_matches_native_wrapping`; `cargo test`,
`clippy`, `fmt` stay green.

Residual (S4 open): the `Table` Kani behavioral harnesses use concrete edge keys because a symbolic key
builds a 256-way mux that CBMC does not decide in reasonable time; `i64` multiplication
associativity/distributivity is verified at the `i8` width `src/scalar.rs` prescribes, not at full
`i64`. Both are bounded-model-checking limits, recorded, not correctness gaps. No rustc/LLVM codegen,
ABI or GPU claim is established.

**First gate (historical):** `lake build`, cwd `proof/corpus`; observed exit 1:

```text
error: TableRefinement.lean:56:28: Unknown constant `apeiron_word.table.StorageError.CapacityOverflow`
```

The next task is Sol's constructor/resource-premise proof, not another extraction redesign.
Required Sol verdict: are the fixed-resource domain and callback contracts sufficient, and do
actual constructor/refusal witnesses inhabit them jointly? Astra's decision does not supply it.
Terra has no open implementation assignment: all four real bodies extract. If a proof reveals an
executing-source defect, stop and assign Terra a separate ≤3-file source/caller repair, gate
`cargo test --workspace --all-features --locked`, before re-extraction in its own E lane. Do not
silently widen a proof task into source/tool redesign.

The lead's full Rust/Kani, both Lake builds, controls and trusted-perimeter gates remain required
for epic closure. The present authorization accepts a building representation and extraction
verdict, not closure of the red bridge. Other governing-doc corrections are a separately scoped
owner task; old growable and deleted-container claims below are historical only.

## Historical binding boundary replacement (superseded by fixed shards)

Owner Astra; base `da01485d24be8900824750431a70f19100bee200`, initially clean. The lead authorizes
source deletion and a red boundary commit in the supplied checkout. No branch/worktree is created;
the existing `one-allocation` branch is unrelated and is not modified. The two-container decision
below is superseded, not a prerequisite to this sequence. No new markdown or other governing-doc
edits are authorized. `docs/kernel-boundary.md` contains the replacement contract and trust cut.

**Intent:** the executing table is the thesis's proof subject. A proved reference beside an unproved
container does not satisfy it. Delete both old containers and their wrappers rather than preserving
that boundary under aliases. Compatibility breakage is accepted; no performance claim transfers.
**Spec:** one `Table<K,V>`, default-valued total lookup, new-first `insertWith`, finite-stream `build`,
and home-sharded finite overrides, as specified in the boundary document. No fixed-capacity-four
restriction, automatic `Option` lifting or commutativity assumption for ordinary builds survives.
The old intent/spec documents below describe retired source, not the replacement's contract.

The boundary change is three serial owned slices: `APEIRON-TABLE-W0-T1` owns these two documents;
`APEIRON-TABLE-W0-T2` deletes `src/map.rs`, `src/safe_table.rs`, `src/set.rs`;
`APEIRON-TABLE-W0-T3` owns `src/table.rs`, `src/lib.rs`, `src/merge.rs`. Astra owns all three for this
explicit lead-assigned deletion. Only the last slice takes lane R. They share one authorized
boundary commit, not a claim of implementation completion.

```text
SCOPE:      the three slices above; remove competing containers, expose only Table, freeze its API
REGISTER:   planning/API removal; excludes completed implementation, refinement and performance
CORPUS:     Apeiron.Table "get_build_acc", "build_eq_of_key_streams_eq",
            "build_gauge_invariance"; Adjacency and Home, native grade proved
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean
GATE:       root: cargo check --lib --locked baseline 0; cargo test --doc --locked after deletion
            101: E0599, Table::new missing at src/merge.rs:47; git diff --check 0
GRADE:      anchor; declaration only, no methods or executing container; no new proof/measurement
```

The merge doctest is rewired to the required API and deliberately fails without its implementation.
The library alone compiles, with unread-field warnings. Do not suppress those warnings or add a
compatibility implementation to manufacture green. The old integration tests, benchmarks, examples,
fuzz targets and storage Kani harnesses are retained as red migration inputs, not silently disabled.
`proof/extract/lib.rs` still names deleted source; generated Lean and its old bridge are stale.
A successful build of those cached definitions alone cannot establish freshness or new-table proof.

Observed boundary checks (root):

| Command | Outcome |
|---|---|
| `cargo test --doc --locked` | 101; one passes, the Table example fails with E0599 (`Table::new` missing). |
| `cargo test --workspace --all-features --locked` | 101; E0432/E0433 on removed `map`, `safe_table`, `set`, `CountTable` and scanner imports. |
| `proof/regen.sh --check` | 1; source digest stage rejects deleted `src/safe_table.rs`, before extraction. Charon `0.1.254` (`b104e24f`), Aeneas `505b6ca-dirty`, Rust extraction toolchain `nightly-2026-08-18`; script reports Lean pin `4.31.0`. |
| `git diff --check` | 0. Formatting evidence only; no storage proof or mutation-control closure. |
| `cargo fmt --all -- --check` | 0 after correcting the changed re-export formatting. |

### Identity

Corpus read: clean `apeiron-rebuild` at `50c94192399add3b9f8e64bd14e36d8cba270705`.
The bridge currently resolves `d9aefc93522e2109bc96158526de6dbd52477276`.

| Source | SHA-256 at boundary edit |
|---|---|
| `src/lib.rs` | `56c0ce5d9fcf14a7075e286cc7710461dc2b986aaf3cecea91844b6b5c15c7a7` |
| `src/table.rs` | `18c9ce400f8fb8f872e6c9cfe370d4f0415081589833d58f0668240c3feea72f` |
| `src/merge.rs` | `f1a056ad3d19c460ff49ec565ad1e22d06ac77ac08da3d24efdff48dc63d1874` |
| corpus `Table/Basic.lean` | `e7558bac1d255643f58100f5a1adf32926a423fe6b49e7c991f8d3a3b0eb26b9` |
| corpus `Table/Adjacency.lean` | `5593d39db83213694ef51b9cb4d73686b9e2b5e8b548a2840e61b5a94d382732` |
| corpus `Table/Home.lean` | `bab40ae04d1bd19cde8c97e09eb2453652cbddf253969ebad03ac582cf5d599b` |

No Lean/toolchain acceptance is inferred from those digests. Each execution task records its actual
Rust, Charon/Aeneas binaries (including patches), Lean, Mathlib and corpus identities. Tool pins
and the historical constructor patch are in `proof/README.md`; do not assume the patch solves
growable vectors, fallible reservation, function pointers or generic callback translation.

### Extraction feasibility stop

`APEIRON-TABLE-W4-T1`, Sol, base `dde9d58`, lanes E+B. Charon emits the real bodies for all four
monomorphic probes. Aeneas rejects each LLBC before translating a method because
`Table.home : fn(&K) -> u64` is an arrow-valued struct field. Verbatim:
`Arrow types are not supported yet`; `Could not translate type decl
'apeiron_word::table::Table because of previous error`. The backend therefore does not reach a
separate verdict on `Vec`, allocation, 256-shard initialization, `usize`, default values, iterator
loops or callback invocation. `StorageError` and both constructors extract. Wave 4 stops here;
waves 5-7 remain blocked on a generated `Table` representation.

```text
SCOPE:      proof extraction roots/generated file, TableRefinement/audits/pin, storage status docs
REGISTER:   extraction/refinement; excludes compiler, allocator and ungenerated Table transitions
CORPUS:     Apeiron.Table Basic/Adjacency/Home at 50c94192399add3b9f8e64bd14e36d8cba270705
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean
GATE:       proof/regen.sh --check; lake build in proof/corpus; initial stale source digest exit 1
GRADE:      proved canonical contracts/error constructors; Rust Table implementation link open
```

### Serial waves and lanes

IDs are `APEIRON-TABLE-W<n>-T1`; one owner per file. Before dispatch the lead records the integrated
base and creates `worktree-agent-<id>`. Each row is a serial wave; a rename is one owned artifact.
**R** = `target/`; **E** = R + `proof/ApeironWord.lean` + `proof/.lake/`;
**B** = `proof/corpus/.lake/` + `proof/.lake/` + Aeneas backend cache. Controls use throwaway copies
and separate writable build outputs. No simultaneous lane occupants. Future filenames denote
required artifacts, not already existing tests or proofs.

| Wave | Owned artifacts / obligation | Owner; lane | One focused gate | One rejecting control |
|---|---|---|---|---|
| 1 | `src/table.rs`, `src/merge.rs`: implement the declared constructor/get/insert_with/build, initialized growable shards and fallible resources; expand executable doctests for nonzero default, collisions and directional merges | Terra; R | `cargo test --doc --locked` (root) | First insertion bypasses `f v d` in a throwaway real body; nonidentity-default fixture must fail. |
| 2 | `tests/safe_table.rs` renamed to `tests/table.rs`: replace the two-container trace with single-table default/write/build tests, growth, exact error witnesses and replace/append order fixtures | Flash; R | `cargo test --locked --test table` (root) | Compare only home addresses instead of full keys in a throwaway real body; collision fixture must fail. |
| 3 | `proof/corpus/lakefile.lean`, `proof/corpus/lake-manifest.json`: pin the read gen2 revision, check Basic/Adjacency/Home on the bridge toolchain | Sol; B | `lake build Apeiron.Table.Basic Apeiron.Table.Adjacency Apeiron.Table.Home` (cwd `proof/corpus`; preparation: `lake update apeiron`) | Nonexistent corpus revision in a throwaway package must fail resolution, never fall back to sibling source. |
| 4 | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: replace `word_safe_table4_*` roots with actual Table constructor/get/insert_with/build roots, no capacity bound; include executing routing/growth and concrete merge instances | Sol; E | `proof/regen.sh --check` (root; first regenerate with `proof/regen.sh`) | Mutate the real default-read branch; unchanged generated output must fail freshness. |
| 5 | `proof/corpus/SafeTableRefinement.lean` renamed/reshaped to `TableRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: remove retired roots/audits; prove constructor, joint representation witnesses, terminating home-shard lookup and total default read | Sol; B | `lake build` (cwd `proof/corpus`) | Generated lookup returns the default for a present key; unchanged lookup theorem must fail. |
| 6 | `proof/corpus/TableRefinement.lean`, `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/AxiomAudit.lean`: writes/growth refine canonical insertWith, preserve Rep, characterize success/resource refusal with witnesses; audit imports the new module | Sol; B | `lake build` (cwd `proof/corpus`) | Generated first-write branch stores `v` instead of `f v d`; unchanged insertWith theorem must fail. |
| 7 | `proof/corpus/TableBuildRefinement.lean`, `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/AxiomAudit.lean`: actual build = canonical build; compose get_build_acc, per-key stream equality and conditional gauge invariance; exact Home attribution and order counterexamples | Sol; B | `lake build` (cwd `proof/corpus`) | Reverse same-key contributions in the generated build; unchanged replace/append build theorem or witness must fail. |

All gates are required with baseline/red/restored controls; the proof tasks target `proved`, tests
only `tested`. A type or empty module that elaborates is not a completed row. No `sorry`, admission,
opaque reservation axiom, assumed desired simulation, or reduced-capacity proof can close a row.
If proof size would exceed 1,500 lines, split the obligation into another serialized ≤3-file task
before editing beyond the declared paths. Sol must approve the resource and callback premises;
this architecture decision is not that proof-sufficiency verdict.

**First gate:** `cargo test --doc --locked`, cwd repository root. It currently exits 101 on the
missing `Table::new`. Implement the same path; do not ignore the doctest. The extraction row is an
explicit feasibility stop: after three materially different unsupported shapes, return the exact
unsupported primitive and next tool-model task. Never restore a separate reference to evade it.

### Caller migration and integration barrier

The seven rows do not repair the whole compatibility surface. After the API lands, Flash migrates
retained callers in serialized ≤3-file batches, without introducing old-name aliases:

- `tests/a_map_is_a_map.rs`, `tests/a_set_is_a_set.rs`, `tests/the_counts_are_exact.rs`: one initial
  batch, gate `cargo test --locked --test a_map_is_a_map --test a_set_is_a_set --test the_counts_are_exact`;
  control: change a new-first merge to old-first and require the unchanged directional fixture red.
  Preserve corresponding semantic/boundary cases; entry/iterator-only contracts are explicitly
  retired, not claimed for the new minimal API.
- `src/proofs.rs`: separate Kani batch, gate `cargo kani`; replace retired storage/scan claims with
  bounds/routing/collision harnesses for Table; a routing-mask mutation must fail. Preserve unrelated
  algebra harnesses. Bounded Kani is supplementary, not the storage refinement theorem.
- Remaining test/example/bench/fuzz callers need an exact path manifest and ≤3-file, single-command
  batches selected by their owners; no unbounded "repair all callers" dispatch. That manifest is
  **MISSING** and blocks whole-workspace closure, not the seven focused gates. Benchmarks must be
  remeasured on the new executing path before old numbers are reused.

Integration order: declaration/deletion → implementation and tests → corpus pin → actual extraction
→ constructor/read → write/resource → build/layout → caller migrations → full assurance barrier.
The lead, not an individual task, runs all AGENTS §6 Rust commands (including Kani), regeneration
freshness, both Lake builds and the controls on one source/tool identity. Existing S7/S8 requirements
remain: claim/witness/audit coverage, trusted checker ≤5,000 lines, owned source ≤1,500 lines and
≤12 immediate source files. Missing assurance gates remain missing; deletion alone does not close
them. Corrections to out-of-scope README, direction, handoff and old lifecycle documents need a
separate lead-owned documentation batch; their retired-container claims are not current acceptance.

**Residual:** no executing Table, no new extraction/refinement, no runtime measurement. SIMD, raw
slot lifecycle and workers are removed rather than left in a tested-only path. Source-level proof
still trusts translation/model fidelity, compiler/runtime/allocator realization and machine execution
as explicitly bounded in `docs/kernel-boundary.md`; it may not assume the table semantics themselves.

## Historical bounded-reference plan and evidence (superseded)

- **Owner:** OpenCode
- **Intent:** `docs/intent/storage-refinement.md`
- **Spec:** `docs/spec/storage-refinement.md`
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean before this epic
- **Status:** approved

## Goal and completion condition

Establish one safe finite storage semantics, prove its successful transitions against the canonical
Table definitions, then relate `Map` without promoting sampled agreement to proof.

```sh
sh -c 'cargo test --workspace --all-features --locked && proof/regen.sh --check && cd proof/corpus && lake build'
```

## Non-goals

- Whole-container extraction in one task.
- Proof by differential testing.
- Performance changes before semantic equivalence.

## Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | Fixed-capacity container and differential trace | `src/safe_table.rs`, `src/lib.rs`, `tests/safe_table.rs` | OpenCode | `cargo test --locked --test safe_table` | Full-check or removal mutation makes the trace fail. |
| W2 | Monomorphic extraction | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | OpenCode | `proof/regen.sh --check` | Generated transition mutation breaks freshness or proof. |
| W2 | Extraction evidence | `proof/README.md` | OpenCode | `git diff --check` | Not applicable: extraction freshness owns the executable control. |
| W3 | Canonical Table refinement | `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean` | OpenCode | `lake build` in `proof/corpus` | Transition mutation breaks unchanged theorem. |
| W4 | Optimized relation decision | `docs/kernel-boundary.md`, `docs/plan/storage-refinement.md` | Astra | `git diff --check` | Requirement review; no executable equivalence claim. |

## Task blocks

### W1-T1 - Safe finite reference

- **Scope:** Implement fixed-capacity lookup/replace/remove and compare against `Map` on
  the same bounded traces.
- **Files:** `src/safe_table.rs`, `src/lib.rs`, `tests/safe_table.rs`
- **Depends on:** None
- **Grade:** tested
- **Gate:** `cargo test --locked --test safe_table`
- **Negative control:** In a throwaway copy, treat a full-table replacement as refusal or leave a
  removed key readable; the unchanged test must fail and the original must pass.
- **Evidence to record:** Test population, full/refusal witnesses and source digest.

### W2-T1 - Extract the safe transition body

- **Scope:** Add concrete word/capacity wrappers that call the real safe implementation and regenerate
  the accepted Aeneas output.
- **Files:** `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`
- **Depends on:** W1-T1
- **Grade:** tested
- **Gate:** `proof/regen.sh --check`
- **Negative control:** A generated body mutation must make freshness or W3 fail.
- **Evidence to record:** Source/tool digests, selected wrappers and extraction result.

### W2-T2 - Extraction evidence

- **Scope:** Record the selected safe-table bodies, source digest, result and exclusions.
- **Files:** `proof/README.md`
- **Depends on:** W2-T1
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. W2-T1 owns extraction freshness.
- **Evidence to record:** Generated names, source digest and unsupported operations.

### W3-T1 - Canonical refinement

- **Scope:** Prove successful word-table transitions denote `Apeiron.Table` lookup and replacement,
  with explicit full and missing-key witnesses.
- **Files:** `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean`
- **Depends on:** W2-T1
- **Grade:** proved
- **Gate:** `lake build` in `proof/corpus`
- **Negative control:** Mutate replacement or removal in a throwaway generated file; unchanged
  refinement must fail and the original must pass.
- **Evidence to record:** Theorem names and complete axiom cones.

### W4-T1 - Independent relation decision

- **Scope:** Freeze the independent relation and trust cuts; neither decision nor reference proof authorizes removing `Map`.
- **Files:** `docs/kernel-boundary.md`, `docs/plan/storage-refinement.md`
- **Depends on:** W3-T1
- **Grade:** anchor (architecture), not implementation equivalence
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. No universal optimized-map claim closes in this task.
- **Evidence to record:** Selected representation, excluded layers and next red command.

## Barrier checklist

- [ ] Container tests and controls pass.
- [ ] Extraction is fresh and free of `axiom`, `sorry` and `admit`.
- [ ] Canonical refinement and fail-closed audits pass.
- [ ] Independent optimized relation decision and complete controls satisfy S6; no `Map` retirement authorized.
- [ ] Universal optimized representation/transition relation is checked before any optimized proof claim; remains open independently of reference closure.
- [ ] S7 integration and S8 ledger/perimeter enforcement pass; see `docs/flash-handoff.md`.
- [ ] Rust tests, clippy, formatting and Kani pass.
- [ ] Source and tool digests and trust boundaries are recorded.

## Stop conditions

- Stop extraction after three unsupported Aeneas representations and record the smallest missing
  safe construct; do not rewrite formulas in wrappers.
- Stop optimized-map promotion if the relation remains sampled; retain tested status and name the
  exact universal obligation.

## Historical W3 blocker record

`W1` and `W2` are closed: the safe fixed-capacity reference passes its focused tests and the
10,000-step differential comparison, and `word_safe_table4_get/insert/remove` are extracted from the
real `src/safe_table.rs` with a fresh, axiom-free generated file. `W3` is blocked before the first
refinement theorem.

Three materially distinct attempts were made at the refinement:

| Attempt | Change | Result |
|---|---|---|
| 1 | Iterator/`for`-loop reference | Charon refused outer-loop breaks and iterator closures. |
| 2 | Bounded recursive `find_from`/`empty_from` reference | Extracts, but the generated body is a `partial_fixpoint`; direct `simp`/`rw` cannot unfold the recursion and the scalar `Result` binds are not `[simp]`. |
| 3 | Aeneas `fun_induction` / `step` on the generated body | `fun_induction` reports no functional induction theorem; `step` accepts only `WP` specs, not pure `Eq` goals. |

The precise unresolved goal: a total Lean function for the safe search, or a canonical refinement
stated as a `WP.dspec` Hoare triple. The extracted search is a `partial_fixpoint`; the only supplied
reasoning principle is the auto-generated `.fixpoint_induct`, which targets `WP.dspec` goals and
requires admissibility over a `Result` CCPO. A pure structural characterization additionally needs a
denotation from the Aeneas array state to `Apeiron.Table.HMap`.

Next bounded task, unblocked now: give the Rust search a form Aeneas translates to a structurally or
well-founded recursive function (for example recursion on a slice/list tail rather than a numeric
index, with the index carried as an accumulator), so the extracted body has equation lemmas and
`fun_induction` support. If that fails, state the refinement through `Apeiron.WP`/`loop.spec_decr_nat`
with the array denotation above.

### W3 unblock attempt (2026-09-12)

Two further materially distinct shapes were tried.

| Attempt | Change | Result |
|---|---|---|
| 4 | Recursion on a slice tail via `split_first` | The generated body is still a `partial_fixpoint`, and `core.slice.Slice.split_first` enters as an unmodeled external `axiom`. Rejected. |
| 5 | Fixed-array `while index < N` search with an early `return` | Charon translates each search to a total `loop` combinator marked `@[rust_loop]`; no `partial_fixpoint`, no external axiom. The supplied `Aeneas.Std.loop.spec_decr_nat` measure principle now applies. |

`src/safe_table.rs::find_from` and `empty_from` are now the `while` loops of attempt 5. The
extraction is regenerated: `word_safe_table4_get/insert/remove` call the loop-based generated
bodies, `proof/regen.sh --check` exits 0, and the throwaway source mutation (`stored == key` to
`stored != key`) makes `--check` exit 1. `cargo test --locked --test safe_table` passes 4/4 and
`cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` exits 0.

The remaining obligation is the canonical refinement itself: a `loop.spec_decr_nat` invariant
relating the generated `Array (Option (K × V)) N` state to a denotation over that array, then
`Apeiron.Table.get`/`insertWith`/`replace` lemmas for `get`, `try_insert` and `remove`. That is a
loop-invariant proof, not an extraction blocker; the first four attempts failed because no total
generated search existed, and one does now.

### W3 first slice (2026-09-12)

`proof/corpus/SafeTableRefinement.lean` proves the generated search semantics with
`Aeneas.Std.loop.spec_decr_nat` at measure `4 - index` and invariant `NoMatchIn key l start index`:

- `find_from_loop_spec`: a returned `some j` is the first key match in `[start, 4)`, and `none` means
  no match there. `find_from_spec` lifts it to the `@[reducible]` `find_from`.
- The body is discharged through `Array.index_usize_spec` and a local `@[step]` `I64` equality lemma.
- Cone `{propext, Classical.choice, Quot.sound}`; `#audit`ed in `proof/corpus/AxiomAudit.lean`.
- Control: a throwaway copy with the generated `done (some index)` changed to `done none` fails the
  same proof. The real `proof/regen.sh --check` and the 3277-job bridge build stay green.

### W3 write links

`try_insert_spec` and `remove_spec` in `proof/corpus/SafeTableRefinement.lean` link the extracted
`try_insert`/`remove` to the canonical table writes, under
`WellFormed t := t.len.val = List.countP (fun e => e.isSome) t.slots.val ∧ KeysUnique t.slots.val`:

- `try_insert_spec`: `Ok old` gives `old = tableHMap self key` and
  `tableHMap t' = Apeiron.Table.insertWith (fun new _ => new) key (some value) (tableHMap self)`;
  `Err _` gives `t' = self`.
- `remove_spec`: `old = tableHMap self key` and
  `tableHMap t' = Apeiron.Table.insertWith (fun _ _ => none) key none (tableHMap self)`.

The invariant supplies the `self.len + 1`/`self.len - 1` scalar bounds (`Aeneas.Std.Usize.add_spec`
needs `len + 1 ≤ max`; `Usize.sub_spec` needs `1 ≤ len`) and, for `remove`, the key uniqueness
that makes clearing the first match equal to the canonical `none` write. Cones inside
`{propext, Classical.choice, Quot.sound}`; `#audit`ed. Control: a throwaway copy with `try_insert`'s
generated `Ok none` changed to `Ok (some value)` fails `try_insert_spec` at
`SafeTableRefinement.lean:786`; the unmutated bridge builds 3351 jobs.

Both specs now conclude `WellFormed t'`. Preservation is proved by the `countP` step
(`countP_set_some`/`countP_set_none`, `one_le_countP_of_get`) and the uniqueness step
(`keysUnique_set_some`/`keysUnique_set_none`). Still open: `WellFormed` at the initial state
(`SafeTable::new`) at this historical checkpoint. The constructor and exact outcomes are now closed;
neither result authorizes removing `Map`.

## Optimized representation decision and remaining sequence

**Historical (superseded by the fixed-shard representation above).** This section names the retired
`SafeTable`/`Map` source; it is kept as the record of that decision, not as current acceptance.

`APEIRON-STORAGE-W4-T1`, owner Astra; base `7d6100c450e5da1fa469ffe9adc16a65c7e720ba`.
User-authorized supplied-checkout documentation exception; initially clean. No branch/worktree created.
The frozen relation is in `docs/kernel-boundary.md` §4. `SafeTable` is the checked reference;
whole unsafe/SIMD/threaded extraction is rejected. Share safe executing scalar bodies where extraction
permits; an abstract layout proof is not a Rust heap proof. S4/S5 stand; S6-S8 and optimized refinement
remain separate open obligations. The earlier W1-W4 table is historical, not renewed acceptance.

```text
SCOPE:      docs/kernel-boundary.md, docs/plan/storage-refinement.md; boundary and bounded sequence
REGISTER:   planning; excludes Rust/Lean edits, optimized promotion and retirement
CORPUS:     Apeiron.Table.Basic: "get_insertWith_self", "get_insertWith_ne",
            "build_eq_of_key_streams_eq"; native grade proved
REFERENCE:  /home/socol/Workspace/apeiron/proof/corpus/.lake/packages/apeiron/proof/Apeiron/Table/Basic.lean
GATE:       git diff --check (root), baseline exit 0; optimized refinement gate MISSING
GRADE:      anchor; bridge baseline lake build exit 0, 3351 jobs; no new refinement or measurement
```

Baseline source SHA-256 (bridge build only; extraction freshness not rerun):

| Path | SHA-256 |
|---|---|
| `src/map.rs` | `4226acc5282ce7626eefb955dff55ce3023d6e5e80167fd536bec982d86c1961` |
| `src/table.rs` | `5f6a62cb47a791162c0173f41d62342f7a0981c650ceed004e8cf0c83f0b9b9d` |
| `src/safe_table.rs` | `c09c5419456d355b2856064479114cf1c575db5016d9787078c0645a663a0479` |
| `proof/extract/lib.rs` | `e5a86d5add7d83a4c9b72dba045dd25cf6e9aa8924442cdc88faa23ae7a79fc8` |
| `proof/ApeironWord.lean` | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` |
| `proof/corpus/SafeTableRefinement.lean` | `6d089d616581d958b8379d1c68b69fdca01d1fa481d63c9424f23528ed278685` |

Resolved corpus checkout: `d9aefc93522e2109bc96158526de6dbd52477276`; manifest Mathlib:
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. Full source/tool identity acceptance remains S7.

### Dispatch and lanes

Each row below is its own serial wave, task T1, with ID `APEIRON-STORAGE-W<number>-T1`.
Flash owns ordinary execution; Sol owns the explicitly marked proof-sufficiency tasks. Lead records
the integrated base and assigns `worktree-agent-<id>` before dispatch. This is a decomposition, not
permission to edit all listed paths now. Proposed files/commands marked **MISSING** require gate-first
implementation; exit 0 from a missing-test filter is never acceptance.

Lanes: **R** = `target/`; **E** = `proof/ApeironWord.lean` plus R and `proof/.lake/`;
**B** = `proof/corpus/.lake/` plus `proof/.lake/` and the Aeneas backend cache under R.
Any task running both extraction and bridge takes E+B exclusively. Controls use throwaway source and
generated copies and separate build outputs; immutable dependency caches may be reused read-only.
No sibling task shares an occupied lane. Every control records baseline/red/restored exit statuses,
source/generated digests, actual patched tool binaries and resolved Lean/Mathlib/corpus identities.

| Wave | Owned files (maximum three) / result | Owner; lane | Single gate (root unless stated) | Control |
|---|---|---|---|---|
| 5 | `proof/storage-controls.py`, this plan: durable S6 proof-control runner and record; **MISSING runner** | Flash; E+B | `python3 proof/storage-controls.py` | Independently mutate generated search hit, fresh write/old return, removal, constructor len, full payload/state, and force refusal on free/present keys. Each unchanged bridge build must fail semantically, not merely fail freshness; restored copies pass. Runner must reject a surviving mutation and missing tools. |
| 6 | `tests/safe_table.rs`, this plan: I64/capacity-four common-domain traces, compare all mappings/old values; separate fifth-key refusal; retain generic tests | Flash; R | `cargo test --locked --test safe_table` | In a throwaway `src/map.rs`, replace the displaced-value return with `None`; unchanged trace must fail. Safe full-refusal test must not call the unbounded Map insert as if it could refuse. |
| 7 | `proof/corpus/OptimizedTableRelation.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: formal logical projection/common-domain relation, joint witnesses and capacity mismatch counterexample; **MISSING module** | Sol; B | `lake build` (cwd `proof/corpus`) | Mutate the logical projection to drop a live entry; unchanged witnessed lookup/size correspondence fails. Grade is proved mathematics about layouts, only model/anchor for actual Map. No assumed Map transition theorem. |
| 8 | `src/table.rs`, `src/table/scan.rs`, `tests/both_tag_paths_agree.rs`: move the existing scanner unchanged to a shared module; preserve exports and target selection; **MISSING module** | Flash; R | `cargo test --locked --test both_tag_paths_agree` | Corrupt one returned lane bit in the executing scanner; unchanged scalar oracle fails. Fix fixtures to include actual EMPTY/TOMB, one-live and one-free lines. No algorithm or performance claim. |
| 9 | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: extract actual shared SWAR match/free bodies only | Flash; E | `proof/regen.sh --check` | Mutate a real SWAR mask constant; unchanged generated artifact must fail freshness. Unsupported calls/axioms stop the task, not trigger a proof-only scalar rewrite. |
| 10 | `proof/corpus/ScanRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: exact masks for all 16 lanes/all byte values; **MISSING module** | Sol; B | `lake build` (cwd `proof/corpus`) | Mutate generated gather or free polarity; unchanged exact-mask theorem fails; restored copy passes. This proves SWAR only, not SSE2 dispatch or Map. |
| 11 | `docs/spec/storage-refinement.md`, this plan: gate-first design for the actual scalar probe/transition core and heap adapter, constructor/preservation/refusal obligations | Astra owner, Sol read-only verdict; no build lane | `git diff --check` | Requirement review must reject a proposal that assumes the desired heap simulation, omits unwind, equates full with OOM, or selects a proof-only path. Semantic acceptance command **MISSING** until the memory semantics/tool route is selected; this row cannot close optimized refinement. |

The scanner sequence is a follow-up after S6 and the shared assurance barrier, not a prerequisite
invented for S4/S5 or a substitute for the heap bridge. Scalar probe geometry (`Geom`, `Descent`,
`group_at`) is the next candidate after scan, but contains `assert_unchecked` and finite counters:
do not allocate its implementation before the last design row specifies actual-source extraction,
overflow bounds and rejecting controls. Serial `CountTable` partition/count refinement follows that
as a distinct contract; live workers/SSE2/unsafe heap semantics need separately approved tool routes.

### Integration barriers and exact blockers

1. Run `lake build` in `proof/corpus` as the single next baseline gate; then implement the S6 runner,
   execute its controls, integrate the focused trace tests, and obtain Sol's relation/premise verdict.
2. Before optimized-source work, complete S7 on that identity: all four AGENTS §6 Rust commands,
   `proof/regen.sh --check`, `lake build` in both packages, and named controls, serialized. Missing
   tools block. The constructor needs the patched tools documented in `proof/README.md`; record
   `AENEAS_HOME`, do not substitute the unpatched release silently.
3. S8 is a separate gate-design dependency, not waived: contrary to the handoff's installed wording,
   `tools/check-claims.py` and `tools/check-assurance.py` are absent at this base. The latter command
   `python3 tools/check-assurance.py` exits 2 (file not found). Lead must authorize a separate
   assurance task and correction to the out-of-scope handoff/AGENTS wiring. Proposed separate tasks:
   - Claim gate: `tools/check-claims.py` plus the existing ledger selected by its owner; gate
     `python3 tools/check-claims.py`; delete a required joint witness/audit binding in a throwaway
     fixture and require rejection. Gate/ledger design is **MISSING**, not dispatch-ready proof work.
   - Perimeter gate: `tools/check-assurance.py` plus this plan; gate
     `python3 tools/check-assurance.py`; inject unlisted production source, a 1,501-line owned file,
     a thirteenth immediate source file and a >5,000-line trusted checker population, requiring
     rejection. Bind all production `src/`/`tests/`, inventory external trust separately; do not
     exclude optimized files to turn the size check green. Existing over-limit populations require
     a separately frozen structural plan, not an exemption here.
   - Gate wiring: `AGENTS.md`, `docs/flash-handoff.md`; gate `git diff --check`; requirement review
     checks that missing checker execution cannot be reported as S8 closed. No semantic grade.
   These are separate serial waves owned by Flash except boundary/claim decisions, which require
   Astra/Sol. Checker tasks take no Rust/Lean lane unless their approved design invokes those tools.
4. Integrate shared scanner, extraction, then mask proof in order. Re-run S7/S8 and controls on one
   identity. Stop before claiming the optimized relation: no real heap-to-layout simulation exists.

**Sol trigger:** the relation changes theorem premises/attribution. Required decision: are the
joint `MemoryValid`/`PlacementValid` witnesses and operation domains sufficient and non-vacuous,
and which actual Rust semantics could discharge the heap simulation without an extra axiom?
No Sol verdict is supplied by this architecture decision. Source observations needing scrutiny:
`Map::clear` drops before blanking tags; `rehome` temporarily removes arrays while counters remain;
generic callbacks may unwind. These are proof obligations, not a completed safety audit.

**Residual claim:** only the existing safe/kernel refinements are checked. No performance measurement,
whole-Map equivalence, generic trait correctness, CountTable overflow/scheduling theorem, compiled
execution guarantee or ring-0 guarantee is added. A stored-state kernel consumer also needs its real
read/compute/write body and exact failure contract selected before composition work is dispatchable.

## S6 baseline/red/restore controls

**Historical (superseded by the fixed-shard representation above).** These controls bind the retired
`SafeTable` source and its generated bodies; they are the record of that checkpoint.

Throwaway root `.mut-s6/` (btrfs reflink of `proof/`, `target/aeneas` symlink; same filesystem as the
repository). Baseline `cd .mut-s6/proof/corpus && lake build` exit 0, 3351 jobs. Toolchain: the single
patched `target/aeneas` symlink, no parallel toolchain directory.

| Obligation | Mutated file (throwaway) | Mutation | Red command (cwd) | Exit | Red line | Restore sha256 | Restore gate |
|---|---|---|---|---|---|---|---|
| Search `find_from_loop_spec` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.find_from_loop.body`: `if index < N` -> `if index > N` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:103:10` (`find_from_loop_spec`) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Write `try_insert_fresh_iff`, `try_insert_err_iff` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.try_insert`: fresh-branch `Ok none` -> `Ok (some value)` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:877:16` (`try_insert_spec`; write link the exactness headlines consume) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Initialization `safe_table_new_eq_emptyTable` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.new`: `len := 0#usize` -> `len := 1#usize` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:1067:2` (`safe_table_new_eq_emptyTable`) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Full handling `try_insert_fullTable_refusal` | `.mut-s6/proof/corpus/SafeTableRefinement.lean` | `fullTable`: slot 0 `some (k0, v0)` -> `none` (falsifies `NoEmptyIn`) | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:1222:67` (`fullTable_noEmpty`; `NoEmptyIn` witness the refusal theorem consumes) | `6d089d616581d958b8379d1c68b69fdca01d1fa481d63c9424f23528ed278685` | exit 0, 3351 jobs |

Restored repository gate: `cd proof/corpus && lake build` exit 0, 3351 jobs; `proof/regen.sh --check`
exit 0, fresh extraction, `ApeironWord` 1708 jobs.

## Revised production boundary and dispatch order

Owner: lead; recorded by `APEIRON-PERF-RESTORE-W4-T1`; base
`c450c222905c050f793a23332708c230c658f82c`. Documentation only; gate `git diff --check` and
`python3 tools/check-assurance.py --check`. This section records the Astra architecture decision that
replaces the earlier production decomposition and the lead-ratified resource/refusal contract
(`APEIRON-PERF-RESTORE-W4-T2`). It supersedes the
"Production recovery, then extraction/refinement" row and the fixed-shard "Remaining serial waves"
above as the dispatch sequence; those sections remain the historical record, not current
instructions.

### Acceptance threshold (owner decision)

The selected implementation must match or beat **every** named applicable incumbent on every
applicable cell, `selected/incumbent <= 1.0`, never behind and ideally ahead. The named incumbents
and the per-cell/incumbent applicability registration are in the
[lifecycle annex](#performance-preserving-consolidation-lifecycle-annex) above; there is no
family-wide exemption and missing, duplicate, invalid or inconclusive pairs reject. There is no
historical-envelope tolerance. This supersedes the RFC elapsed-median floor
`selected/hashtrinity <= 1.25`. The superseded 1.25 floor and historical-envelope language still
appears in `docs/direction.md` and `docs/flash-handoff.md`; it is reconciled in
`APEIRON-PERF-RESTORE-W5-T1`, not here. Within this plan the threshold above is the only acceptance
rule.

### Staging: serialized representation integration wave

The growable-representation change is a multi-task serialized integration wave, not a three-file
edit. Each task owns at most three files and keeps `cargo test`, `cargo clippy` and `cargo fmt`
green plus its focused gate; the wave closes only when extraction, refinement, Kani, the REG-1
harness and the shared gates pass on the integrated identity. An intermediate proof failure is not
an accepted wave result. Reason: changing the representation simultaneously changes admission
(`src/table.rs` positive-`N` `ShardFull`), the extracted array representation
(`proof/ApeironWord.lean`), the bounded-admission refinements, `src/proofs.rs` Kani predicates and
the REG-1 workloads/oracles.

### Revised dispatch order prefix

| Order | Task | Owned scope |
|---|---|---|
| 1 | `APEIRON-PERF-RESTORE-W4-T1` | this documentation task: revised production boundary |
| 2 | `APEIRON-PERF-RESTORE-W5-T1` | reconcile `docs/direction.md`, `docs/flash-handoff.md`, `proof/README.md` |
| 3 | `APEIRON-PERF-RESTORE-W6-T1` | root real `spread`/`finish`: `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` |
| 4 | canonical corpus task (separately assigned) | `Table/WordIdentity.lean`, registered in `Audit.lean` |
| 5 | `APEIRON-PERF-RESTORE-W7-T1` | bridge pin transport |
| 6 | `APEIRON-PERF-RESTORE-W8-T1` | `proof/corpus/TableRefinement/CoordIdentity.lean` plus `AxiomAudit` |
| 7 | interface expand / migrate / contract batches | separately scoped tasks, at most three files each |
| 8 | representation integration wave | the serialized wave above |

The resource/refusal contract is ratified in `docs/kernel-boundary.md` (option (b), checked
reservation/allocation failure, transactional); the identity theorem, bridge pin transport and
representation wave are unblocked. The canonical release/commit remains separately authorized; the
exact remaining proof/task inventory is no longer blocked on the resource decision.

## Rework dispatch order

Owner Astra; recorded by the rework contract; base
`c450c222905c050f793a23332708c230c658f82c`. Documentation only; gate `git diff --check` and
`python3 tools/check-assurance.py --check`. This order supersedes the "Revised dispatch order
prefix" above. The first growable `Table` at `src/table.rs` sha256
`c34ddf17ef2ed7a81fd4b3e16447479b55d737af8f5253b99dd397414d9b1dc1` is rejected; its measured red
and defect list are in the
[local boundary](../kernel-boundary.md#reworked-growable-representation-astra-decision-first-cut-rejected).
The bounded fixed-shard table is not restored and `examples/reg1/candidate.rs` is not promoted.

```text
SCOPE:      docs/kernel-boundary.md, docs/spec/storage-refinement.md, docs/plan/storage-refinement.md;
            adjudicate address, Sigma, search and refinement premises
REGISTER:   refinement/state/planning; no source, proof, benchmark or extraction change
CORPUS:     Apeiron.Table.{get_insertWith_self,get_insertWith_ne,get_build_acc,
            build_eq_of_key_streams_eq}, Basic.lean:87-174; native proved
REFERENCE:  src/{table.rs,table/scan.rs,coord.rs}; 3a95bb5:src/{table.rs,coord.rs}; baseline receipt
GATE:       git diff --check, repository root; no task-specific red for this documentation gate
GRADE:      specification/anchor; downstream tested/proved grades require their named gates
```

### Revised representation boundary: flat resident area and spills

The Astra verdict revises the selected structure from a single growable key-partitioned `Table` to
two coupled regions:

- A **table-level flat resident area**: shard-major `256 x L`, admission
  `direct_word = Some(w) ∧ lane < L`, checked sizing, a stated byte budget and a density test.
  Failures spill to the grouped regions rather than refuse.
- **Independently load-sized grouped spill regions** beside the flat area.

One representative per key is held under a composite `DirectCover`; extent growth must reclassify
newly eligible spills. Key-dropping uses a compile-time `KeyRep` policy (`Stored<K>` or `Regen`
with an exact codec law), not a sibling count table and not a lossy `FreqMap`. Restoration also
includes the key-partitioned load-sized shard build, deterministic schedule, grouped probe and
key-dropping count/dedup; the proof, extraction, Kani, registration and measurement work follows on
the same source identity. The boundary is a recorded plan, not a built, extracted or measured
result, and no speed recovery is claimed. REG-1/G1 stays OPEN.

### Sol contract adjudication (`APEIRON-PERF-REWORK-W5-T2`)

The Astra boundary is accepted with these proof premises:

1. One generic `Table` uses `TableAddress { place: u64, direct_word: Option<u32> }` and
   `Home::address(sigma,key)`. `WordHome` supplies `Some(x as u32)` exactly below `2^32`; all
   placement-only adapters supply `None`. The direct lane is `u64::from(x) >> 8`, not
   `place >> 8`, and full `(Sigma,K)` equality decides identity.
2. The logical key and payload key are `(Sigma,K)`; reads/writes receive `sigma`; build consumes
   `&[((Sigma,K),V)]`. Equal `K` at unequal `Sigma` remains two keys through every transition.
3. Direct-empty early absence is admitted only from reachable `DirectCover`, monotone occupancy and
   tag authority. A first window vacancy may stop the prefix, not the coverage sweep. Every absence
   without an earlier proved certificate requires finite all-group coverage. Window extraction masks
   each seven-bit value by `0x7f` before group reduction; `gmask` never extracts a window.
4. `GrowableGeometry`, `GrowableDirectory`, `GrowableScan`, `GrowableSearch`,
   `GrowableRefinement` and `GrowableSchedule` carry the theorem set in the spec and boundary.
   `GrowableSchedule` composes its per-key subsequence equation with
   `Apeiron.Table.build_eq_of_key_streams_eq` (`Basic.lean:167-174`).

This is a contract change. Relative to `Home::home(&K) -> u64`, refinement adds `Sigma`-indexed
address laws, an optional exact-word certificate and its routing equation, full-pair identity,
reachable search certificates and relocation preservation. These are anchors until the named
real-source bridge gates pass.

The strict 126/160 failures at that adjudication are against the serial `hashtrinity::Table`
incumbent at `4cc14471`; the current strict run is the fresh post-spill full-profile receipt
(lifecycle annex above; sha256 `946b5d40…`), which still reports `full_acceptance=false`. Serial
recovery remains open and no speed recovery is claimed. Partitioned/parallel build plus grouped probe for the spatial packed
family is distinct production work, not evidence supplied by serial recovery. REG-1/G1 remains OPEN.

### Gate-first tasks

`APEIRON-PERF-REWORK-W1-T1` and `W2-T1` are gate-first and must land before the storage rework;
`W3-T1` captures the unchanged first-cut baseline before any rework edit.

| Task | Owned files (at most three) | Completion, one focused gate and control |
|---|---|---|
| `APEIRON-PERF-REWORK-W1-T1` | `examples/reg1/strata.rs`, `examples/reg1/matrix.rs`, `examples/reg1/matrix.json` | Focused registration tests; re-register refusal/resource cells and archive the old fixed-shard receipts. Control: the old `ShardFull` digest rejects. |
| `APEIRON-PERF-REWORK-W2-T1` | `examples/reg1/performance.rs`, `examples/reg1/strata.rs`, `examples/reg1/main.rs` | Strict performance-acceptance mode: a per-cell failure makes the command exit nonzero. Control: a valid report with one failing cell exits nonzero. |
| `APEIRON-PERF-REWORK-W3-T1` | new `docs/evidence/` receipt | Capture the unchanged first-cut before-receipt under `docs/evidence/` before rework. |
| `APEIRON-PERF-REWORK-W4-T1` | `src/table.rs`, new `src/table/scan.rs`, `tests/table.rs` | Geometry, exact SWAR, bounded first-fit, non-optional slots. |
| `APEIRON-PERF-REWORK-W5-T1` | `src/table.rs`, `tests/table.rs` | Direct candidate/directory and transactional relocation. |
| `APEIRON-PERF-REWORK-W6-T1` | `src/table.rs`, `tests/table.rs` | Implement `TableAddress`, `Home::address`, full `(Sigma,K)` identity/build input and direct-word routing; test unequal-`Sigma`/equal-`K` separation and placement-only `None`. Consumer migrations are separate at-most-three-file tasks. |
| `APEIRON-PERF-REWORK-W7-T1` | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | Root the real address, search, relocation and sequential build bodies and regenerate. |
| `APEIRON-PERF-REWORK-W8-T1` | `proof/corpus` bridge theorems (serialized) | `GrowableGeometry`, `GrowableDirectory`, `GrowableScan`, including the seven-bit mask and direct routing. |
| `APEIRON-PERF-REWORK-W9-T1` | `proof/corpus` bridge theorems plus `AxiomAudit.lean` (serialized) | `GrowableSearch`, `GrowableRefinement`, audited under `{propext, Classical.choice, Quot.sound}`. |
| `APEIRON-PERF-REWORK-W10-T1` | sequential partition/worker/assembly source plus tests, separately decomposed | Restore partitioned/parallel build and grouped probe as spatial production work; preserve per-key subsequences. |
| `APEIRON-PERF-REWORK-W11-T1` | `proof/corpus` schedule bridge plus `AxiomAudit.lean` (serialized) | `GrowableSchedule` and composition with `build_eq_of_key_streams_eq`; thread realization remains outside the extracted claim. |
| `APEIRON-PERF-REWORK-W12-T1` | `src/proofs.rs`, `examples/reg1/coverage.rs`, `examples/reg1/coverage.json` | Kani harnesses for the reworked geometry and locate. |
| `APEIRON-PERF-REWORK-W13-T1` | `examples/reg1/workload.rs`, `examples/reg1/correctness.rs`, `examples/reg1/matrix.rs` | Workload and correctness re-registration. |
| `APEIRON-PERF-REWORK-W14-T1` | `examples/reg1/matrix.json` plus serial and spatial before/after receipts | Every applicable cell `selected/incumbent <= 1.0`; serial and spatial incumbent families remain distinct. |

### Refusal-cell re-registration

Do not merely mark the 8 rejected cells not-applicable. Archive the old fixed-shard refusal receipts;
register successful growable insertion and update traces with fresh oracle digests and explicit
operation names; add genuine zero-width, sizing-overflow and injected-reservation-failure rows with
unchanged-state and no-callback observations. The strict performance-acceptance mode from `W2-T1` is
the per-cell authority; the aggregate exit status is not.

## Extraction-shaped representation dispatch order (APEIRON-EXTRACT-REP)

Owner Astra; recorded by `APEIRON-EXTRACT-REP-W1-T1`; documentation only; gate `git diff --check`.
This order supersedes the "Rework dispatch order" and "Gate-first tasks" above as the dispatch
sequence; those remain the historical record. The representation boundary is in the
[local boundary](../kernel-boundary.md#extraction-shaped-representation-boundary-astra-apeiron-extract-rep);
the contract is in the
[spec](../spec/storage-refinement.md#extraction-shaped-representation-contract-astra-apeiron-extract-rep).

| Wave | Obligation |
|---|---|
| `W1` | Projection-free carrier on both faces: `Stored<K>{key}`, `Regen<C>{identity,codec}` with manual `Clone`, `FlatSlot<R,V>` and spill `(Sigma,R,V)`, no `R::Held`; preserve `Table<…,Stored<K>>` and `Table<…,Regen<I64WordCodec>>`. |
| `W2` | Digest fix: `proof/regen.sh` drops the deleted `src/table/resident.rs` from the hashed source list. |
| `W3` | Resource interface decision: keep planning, reservation and allocation unextracted; allocating public methods stay unrooted until a checked allocator primitive/tool model exists. Decision recorded by `APEIRON-EXTRACT-REP-W5-T1`. |
| `W4` | Prepared resource adapter in production: `Prepared { plan, storage, outcome } | Refused(StorageError)`; checked-plan, reserve-all, then apply/commit; pending insertion inside the transaction; refusal stutters. This remains unextracted. |
| `W5`–`W10` | Prepared-spill extraction checkpoint: `proof/ApeironWord.lean` sha256 `faa257a8720c71ee7a08497621ddecd40495cf3729ee10c56f18119f5778113a` contains the real spill-only roots. It does not contain composite `Table::locate`, `Site` or `DirectCover`; no composite-search refinement follows. |
| `W11` | Search boundary decision, `APEIRON-EXTRACT-REP-W11-T2`: keep the probe-order decision unextracted under `PinnedJoinBoundary`; root the real straight-line apply under `SearchDecisionCorrect`. Contract grade `anchor`; supplied context-join evidence `tested`. |
| `W12` | **Next bounded task:** factor the production search in `src/table.rs` into a decision returning `(Found|Vacant|Exhausted, Flat(i)|Spill(shard,i)|NoIndex)` and a straight-line read/insert apply; keep direct, seven-window and mandatory-sweep order unchanged. Add focused cases in `tests/table.rs`. Gate and a late-hit/sweep mutation control must be assigned in the source task. |
| `W13` | Root both carrier specializations of the real apply in `proof/extract/lib.rs`, add only those roots to `proof/regen.sh`, and regenerate `proof/ApeironWord.lean`. Gate `proof/regen.sh --check`; control mutates an apply branch in a throwaway generated copy. |
| `W14` | Prove `GrowableSearch` over the extracted apply under `SearchDecisionCorrect`, with `search_decision_simulates` targeting total-by-default `Apeiron.Table.get`, `get_insertWith_self` and `get_insertWith_ne`; register the headline and rejecting control under the closed cone. |
| `W15`–`W17` | Prove, in order, `GrowableRelocation`, `GrowableRefinement` and `GrowableWitnesses`, each carrying `SearchDecisionCorrect`, `RepR` and `PreparedResources` and each with its own rejecting control. |
| `W18` onward | `GrowableDirectory`, `GrowableSchedule` and `GrowableBatch` retain their existing routing, per-key-subsequence and grouped-assembly contracts; then controls, Kani, coverage, matched correctness and performance. An allocator primitive-model task is added only if allocating public methods are later required as roots. |

**W3 Sol contract.** Premise `PreparedResources` requires checked arithmetic, exact old/pending entry
classification, in-bounds destinations, sufficient initialized reserved storage and no further
fallible allocation in the core. The simulation obligation `resource_adapter_simulates` has two
branches: prepared core success preserves `RepR` and denotes canonical
`Apeiron.Table.{get,insertWith,build}`; refusal supplies no core input, leaves insert state unchanged
with no clone/merge/relocation, and returns no build table. Allocator/global-allocator realization is
outside the claim. Grade: contract `anchor`; supplied refusal observations and context-join spike
`tested`; no proof.

**W11 Sol contract.** Premise `SearchDecisionCorrect` requires `Found` iff the queried full
`(Sigma,K)` key is present at the returned in-bounds unique representative; `Vacant` only for an
absent key at the first-fit lane; and `Exhausted` iff absent with no vacancy. Direct-empty absence
requires `DirectCover`; a window vacancy may stop the prefix but never the mandatory all-group match
sweep; every other absence requires complete finite coverage. `GrowableSearch` refines the real apply
under this premise. `GrowableRelocation`, `GrowableRefinement` and `GrowableWitnesses` carry it;
`GrowableDirectory`, `GrowableSchedule` and `GrowableBatch` are unchanged. Rust/Kani can support only
`tested`/bounded evidence for the decision; the apply becomes `proved` only after real-source
extraction, canonical refinement, bounded bridge build, audit and rejecting control.

`GrowableSchedule` still proves the implementation-specific per-key value-subsequence equation and
composes it with canonical `Apeiron.Table.build_eq_of_key_streams_eq`
(`Table/Basic.lean:167-174`). `Song/kernel/Web.lean`
`preduce_schedule_free`/`preduce_append`, read at
`/home/socol/Workspace/song-ref/lean/Song/kernel/Web.lean` revision `ae688f66`, is the external Song
twin only; no Song declaration enters the Apeiron Rust/canonical cone.

**Prerequisites and open gates.**

- `W2` removed the deleted `src/table/resident.rs` digest path. The current generated artifact remains
  bound to the live source list and is spill-only.
- `proof/corpus/lakefile.lean` still lists the fixed-shard and joint consumers; a separate archival
  manifest is required before a full corpus build.
- Supplied spike evidence, grade `tested`: Aeneas `505b6ca-dirty` / Charon `0.1.254` fails
  `Could not match the contexts` at `Table::plan_rebuild` (`src/table.rs:678:16-680:17`). The generic
  borrowed-carrier/live-guard/computed-classification shape reproduces; scalar and tested
  straight-line/accumulated-`Option`/cast variants pass; `-loops-to-rec`, `-loops-no-rec` and
  `-strict-joins` do not remove it. Classification is pinned Aeneas limitation (b), not allocator
  case (c); further planner reshaping is not scheduled.
- Reachable `try_reserve_exact` is independently unmodeled. The planner/reserver, allocator/global
  allocator, executor/`thread::scope`, rustc/LLVM, ABI, GPU and `unsafe` container stay outside the
  proved claim. Slang's empty cone remains separate.
- The current generated artifact is spill-only. Composite search, the decision contract and its real
  apply root remain open. No proof or speed is claimed. REG-1/G1 remains **OPEN**.
