# Post-compaction continuation — programme dispatch (6)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md`, `docs/reviews/multi-repo-fleet-brief.md`. Source and gates are
the record. Live mirror: `/home/socol/oc-tmp/fleet-state.md`; escalation verdicts:
`/home/socol/oc-tmp/escalation/`. This file replaces `post-compaction-continuation-11.md`.

## 0 · Scope: Song and Slang are OFF the fleet

- Owner direction: **Song and Slang work stopped**. All five Song workers were stopped with receipts
  under `/home/socol/oc-tmp/song-*/` (closure, census-det, census-closure, conecorpus, k212). The
  Slang registry-extension worker was stopped with nothing written to the repo. Do not dispatch or
  commit Song or Slang work; the owner carries Song on a separate branch.
- One Slang commit landed before the stop: `c73f39e` (tools/slang completeness gate). Its open
  finding: with completeness active, **13 of 38 active stages refuse — 197 definitions unregistered,
  130 in no transitive closure**. Receipts: `/home/socol/oc-tmp/slang-complete/receipt.md`,
  `/home/socol/oc-tmp/slang-registry/receipt.md` (the latter also corrects a classification: the
  `Slang.Table.nat_lt_*` cases are hand-written theorems and belong on the registration side).
- Remaining lanes: **apeiron** (corpus reconciliation + gate machinery), **penglai**, **webfold**,
  **intent**. Routes: Space Bunny default (`openrouter/stealth/space-bunny-alpha`, free until
  2026-10-05), `deepseek-flash` on mandatory triggers, `gemini-flash` non-default. Sol:
  `/home/socol/oc-tmp/bin/sol.sh`; LmHead verdicts recorded (`lmhead-row-body-visit-answer.md`,
  `lmhead-row-visit-retirement-answer.md`).

## 1 · Traps and exact commands

- Lean lane: `flock -w 600 /tmp/opencode/apeiron-lane.lock` + `systemd-run --user --scope -q
  -p MemoryMax=12G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1` + `timeout 540`. Corpus workspace:
  `/home/socol/oc-tmp/corpus-t1-f32`; workers copy it and never edit it; the lead syncs deliverables
  in, and the repo is the landing target.
- **The `find` shim recurred**: a worker's olean-delete loop was a silent no-op (use globs and
  dry-run the delete). `/home/socol/.local/bin/rg` also misreported a filename once — verify with
  `/usr/bin/rg` or Python when a path looks odd.
- Regen: `AENEAS_HOME=/home/socol/Workspace/penglai/target/aeneas APEIRON_LAKE=0
  ./proof/regen.sh --check`.
- Webfold: the live tree is owned by a codex session — do not touch it. Use the push lane
  `/home/socol/oc-tmp/wf-ci3` (tip `95bd6a1a`) and lane copies with their own lock files.
- Intent: workers sometimes deliver inside detached worktrees (copy the files in before
  committing); `scripts/w*-controls.py` are shared add-only and get 3-way merged when branches
  diverge; the justfile hooks for new scripts are the lead's.
- Heavy background shells may hold a repo's git index: do not commit into the same repo until the
  notification lands.
- Song notes for the owner's branch: K2.1.2 stopped at `src/prover/gate/build.slang:719`
  (`song.unlink` absent from the pinned `songc` while `TOKEN_UNLINK=110` is in the compiler source;
  six call sites). The census "nondeterminism" is `depLoad` reading live dependency bytes plus a
  silent `fd < 0` skip, not hashing. The closure gap: a callee `given` is never a caller obligation
  — the minimal closure turns the tree red with 47 FAIL — and the uncommitted fixture
  `redteam/prover/refute/a200_lawless_call_site_precondition.slang` trips `suite-declared` until
  declared or removed.

## 2 · Landed since continuation-11

- apeiron: W6 rows batches 1–7 migrated (`7096bdd`, `2f3c415`, `2e77815`, `529726a`, `5111316`);
  repairs `30d9edc` (ApplyPending), `32aa596` (PublishPrepared), `50ebfe2` (RegenRelocateEntry),
  `25b4c6c` (RelocateEntry), `47a36ee` (LmHead retirement, Sol (b)); `c73f39e` (Slang completeness,
  see §0). Matvec's commit is pending a background controls run from this session.
- intent: `e71811b` plate, `5ed6dab` recall pricing, `1c0228f` oracle grading, `1689f5d` agents
  plate, `7f90252` curation, `8b82604` controls-recipe wiring, `f097194` turn seam, `250aa6d`
  chain, `e81f629` budget-admission spec (W36–W40 rows), `32d63dc` evidence states, `b530dbd` host
  snapshots, `8ad92aa` literal store + scope round-trip, `d7892b3` provenance wiring, `55472c0`
  budget reservations, plus the W25 anchor narrowing (control_catalog.json `3ef1969d`).
- penglai: `ae76e8d` schedule, `ba83ee9` adapter, `45bfe87` bank, `0cddf0f` ledger, `3d4ba33` W10
  economy, `1f71677` W8 spec (D1–D7), `53695bb` W7 journal. The journal codec split is uncommitted
  and green: `src/runtime/journal.rs` `e1b64242` (1,137 lines) + `src/runtime/journal/codec.rs`
  `0941b0b8` (549), 24 tests, invariance checks — commit owed.
- webfold (pushed lane): `f73694bd`, `71531196`, `bef5ee9e`, `95bd6a1a`; the control registry is
  211/211 validated on a frozen snapshot; the acquisition **harness repair is done in a copy**
  (`extract_acquisition.rs` `5916ad27`, `regen_acquisition.sh` `f519de59`, `lake-manifest.json`
  `dc246cdb`, `AcquisitionMath.lean` `4d5e8f50`; gate PASS at `95bd6a1a`, audit empty cones,
  controls C1–C9) — landing to the lane owed.
- song (handover history): `87e91489`, `5fb8f9a9`, `ca35170e`, `65ef3ba0`, `363d6e05`, `e59b684b`,
  `f5b66c0e`, `b36a133e`, `44c0d51f`.

## 3 · Apeiron corpus state (workspace `corpus-t1-f32`, artifact `ac563ff7`)

Green and synced: `Growable.Scan` (`e75e9018`), `Search` (`f20a9c05`), `Relocation` (`6a7cc951`),
`Identity` (`5eca14f7`), `Native.LocatorSpec`, `Refinement` (`ef1c2917`), `Representation`
(`16bda5dc`), `Numeric.KvReachable` (`b6e17415`, file-green), `RecipLink` (`e1344f5f`),
`AttnLoopOutput` (`c9f6c546`, window narrowed), `AttnHeadWindowsValue` (`9f582724`).

In flight: the `attn_domain_v2` carrier conjunct + KvReachable; Representation's two residual sites
(`:1293` `htagAt` destructuring, `:1357` `Vec.set` rewrite; ref4). Then: `AttnHeadRefines` final —
re-state premise 2 narrow and **prove** it from `plane_at_spec`, add the additive length export,
thread the carrier premise; premise 3 (the recip `some` arm) stays a documented hypothesis — then
KvReachable's module gate and the EngineAttnRefinement aggregate; and
`ReadLoop`/`ReadState`/`Classifier`/`Resources`/`LocateLoop` → AxiomAudit re-records (drop five
summary rows, add `SpillTagRecorded` + `reservedRepR_tagRecorded`) → full same-identity build.

Not adopted: the alternative Scan derivation `dfc23ff8` (2,298 lines, keeps the retired SWAR math).
Its finding — a missing `MatchResult` discrimination lemma for a let-bound `vis` scrutinee — is in
`/home/socol/oc-tmp/corpus-fix-scan3/receipt.md`.

## 4 · Apeiron gate machinery

E1.W6: 39 rows migrated; per-row controls 28/29 green. Matvec is finalizing (background); RowDotAdmission's
redesign is in flight (its `--controls` needs ~571 s against a 525 s inner budget; the redesign
shares the package and cache across legs without weakening the control). After both: commit the W6
exit record, then open **E1.W7** (delete `*_gate.py`, reject new ones). Open item: where the 29
row-level control results are recorded — `docs/plan.md` has no row-level register.

## 5 · Penglai

W1–W7, W9, W10 landed; W8 implementation in flight under the frozen D1–D7 spec; the codec-split
commit is owed. Island W2-T1/W3-T1 are blocked on `src/physics.rs` ownership (long-lived dirty
work) — a lead/owner call before dispatch. Native surface rows hold.

## 6 · Webfold

Pushed tip `95bd6a1a`; full registry 211/211 green on the snapshot (the full sweep measures 1,074.8
s — stage in ≥3 invocations at 540 s; the single 570 s shape is not viable). Landing owed once the
codex session releases the live tree: the harness repair + the live-script fixes (lock-before-budget,
`blocked_on_lock` receipt, record-and-continue, the unguarded `timeout=180`, the mtime-restore
no-op, splittable groups). E5 remains blocked: no `#[path]` harness over `src/html_extract.rs` is
viable, so its policy types are a checked stand-in with a drift check; Aeneas still cannot host the
containment transitions under the empty cone. D5 in flight. **The supply-chain GPL red is the
owner's call** (re-pin to an MPL-2.0 rev recommended, or amend `deny.toml`).

## 7 · Intent

W32 T1–T7 and W33 T1–T4 landed; the store is literal and round-tripping; host snapshots (W36) and
budget reservations (W37) landed with their decisive controls (four host legs; the two-supervisor
race). In flight: the W36 and W37 wave briefs + controls scripts. Residuals: `LiveDriver` still gets
the default empty plate; the extraction candidacy needs `#[pure_core]` + register rows (host `admit`
core; budget identity closure, `RateCard::charge`, the disposition machine, the settled fold);
`docs/TRUST.md` owes the proc and statvfs rows; W38–W40 next; `scripts/w25-controls.py` still
carries the wide anchor and mutates the live tree (narrow it or retire it in favour of
`intent controls --group w25`). The W25 catalog anchor was narrowed and committed (coverage back to
147/151, uncovered = the four `ctrl-w15-*` routes).

## 8 · Owner decisions outstanding

1. **Webfold supply-chain GPL** — re-pin vs policy (the last CI red).
2. **W06-O5** — the trusted-checker size bound value (Intent states none; Apeiron's 5,000 does not
   fit its 12,195 measured lines; a ratchet at the measured value is recommended).
3. **W36–W40 milestone** — recommended as a new **M10 "bounded spend and device admission"**, with
   the demonstration clause drafted in chat.

## 9 · Next actions, ordered

1. When the Matvec notification lands: verify its commit; then RowDotAdmission; then the W6 exit
   record and E1.W7.
2. Corpus: fold ref4 + the carrier task; dispatch HeadRefines final; then the ReadLoop chain; then
   the AxiomAudit re-records and the full build.
3. Penglai: commit the codec split; W8 on return; the island ownership call.
4. Webfold: land the harness repair to `wf-ci3` and push; D5 on return; the registry-registration
   landing when codex frees; the live-script fixes; the GPL call.
5. Intent: fold the W36/W37 briefs; wire their scripts into the justfile; the extraction-candidacy
   task; W38.
6. Refill on every return; keep `/home/socol/oc-tmp/fleet-state.md` current.
