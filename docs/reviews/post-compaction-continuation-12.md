# Post-compaction continuation — programme dispatch (6)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md`, `docs/reviews/multi-repo-fleet-brief.md`. Source and gates are
the record. Live mirror: `/home/socol/oc-tmp/fleet-state.md`; escalation verdicts:
`/home/socol/oc-tmp/escalation/`. This file replaces `post-compaction-continuation-11.md`.

## 0 · Scope: Song and Slang are OFF the fleet

- Owner direction: **Song and Slang work stopped**. All five Song workers were stopped with receipts
  under `/home/socol/oc-tmp/song-*/`. The Slang registry-extension worker was stopped with nothing
  written to the repo. Do not dispatch or commit Song or Slang work; the owner carries Song on a
  separate branch.
- One Slang commit landed before the stop: `c73f39e` (tools/slang completeness gate). Its open
  finding: with completeness active, **13 of 38 active stages refuse — 197 definitions unregistered,
  130 in no transitive closure**. Receipts: `/home/socol/oc-tmp/slang-complete/receipt.md`,
  `/home/socol/oc-tmp/slang-registry/receipt.md`.
- Remaining lanes: **apeiron** (corpus + gate machinery), **penglai**, **webfold**, **intent**.
  Routes: Space Bunny default (`openrouter/stealth/space-bunny-alpha`, free until 2026-10-05; OpenRouter schedules in UTC - treat 00:00 UTC Oct 5 (11:00 AEDT) as the dispatch boundary, after which gemini-flash is the sanctioned lane),
  `deepseek-flash` on mandatory triggers, `gemini-flash` non-default. Sol:
  `/home/socol/oc-tmp/bin/sol.sh`; LmHead verdicts recorded (`lmhead-row-body-visit-answer.md`,
  `lmhead-row-visit-retirement-answer.md`).

## 1 · Traps and exact commands

- Lean lane: `flock -w 600 /tmp/opencode/apeiron-lane.lock` + `systemd-run --user --scope -q
  -p MemoryMax=12G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1` + `timeout 540`. Corpus workspace:
  `/home/socol/oc-tmp/corpus-t1-f32`; workers copy it and never edit it; the lead syncs deliverables
  in, and the repo is the landing target.
- **`find` shim recurred** (an olean-delete loop silently no-opped; use globs and dry-run the
  delete). `/home/socol/.local/bin/rg` misreported a filename once — verify with `/usr/bin/rg` or
  Python when a path looks odd.
- Regen: `AENEAS_HOME=/home/socol/Workspace/penglai/target/aeneas APEIRON_LAKE=0
  ./proof/regen.sh --check`.
- Webfold: the live tree is owned by a codex session — do not touch it. Push lane
  `/home/socol/oc-tmp/wf-ci3`, tip `65524839`.
- Intent: workers sometimes deliver inside detached worktrees (copy the files in before
  committing); `scripts/w*-controls.py` are shared add-only; the w36/w37 justfile wiring is done.
- **Long foreground shells can be aborted by the harness** (a corpus full-build run was interrupted
  mid-session): keep heavy runs in smaller steps or background them, and re-verify digests.
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
  `25b4c6c` (RelocateEntry), `47a36ee` (LmHead retirement, Sol (b)), `95501ca` (Matvec);
  `c73f39e` (Slang completeness, off-fleet history); `b373121` (this prompt family).
- intent: `e71811b`, `5ed6dab`, `1c0228f`, `1689f5d`, `7f90252`, `8b82604`, `f097194`, `250aa6d`,
  `e81f629` (budget-admission spec + W36–W40 rows), `32d63dc`, `b530dbd` (host snapshots), `8ad92aa`
  (literal store + scope round-trip), `d7892b3` (provenance wiring), `55472c0` (budget
  reservations), `1d316bb` (W25 anchor), `63a9aed` (W36/W37 briefs + controls + `pub mod host` +
  justfile: 21 scripts reachable).
- penglai: `ae76e8d`, `ba83ee9`, `45bfe87`, `0cddf0f`, `3d4ba33` (W10 economy), `1f71677` (W8 spec
  D1–D7), `53695bb` (W7 journal). The journal codec split is **delivered uncommitted and green**:
  `src/runtime/journal.rs` `e1b64242` (1,137 lines) + `src/runtime/journal/codec.rs` `0941b0b8`
  (549); 24 tests; invariance verified — commit owed.
- webfold (pushed lane): `f73694bd`, `71531196`, `bef5ee9e`, `95bd6a1a`, `7b47d964` (harness repair:
  `extract_acquisition.rs` `5916ad27`, `regen_acquisition.sh` `f519de59`, `lake-manifest.json`
  `dc246cdb`, `AcquisitionMath.lean` `4d5e8f50`), `65524839` (frames scalar core: `frames.rs`
  `3a1a6646`, `extract_acquisition.rs` `85223b49`, `regen_extraction_state.sh` `da6e46e5`).
- song (handover history): `87e91489`, `5fb8f9a9`, `ca35170e`, `65ef3ba0`, `363d6e05`, `e59b684b`,
  `f5b66c0e`, `b36a133e`, `44c0d51f`.

## 3 · Apeiron corpus state (workspace `corpus-t1-f32`, artifact `ac563ff7`)

Green and synced: `Growable.Scan` (`e75e9018`), `Search` (`f20a9c05`), `Relocation` (`6a7cc951`),
`Identity` (`5eca14f7`), `Native.LocatorSpec`, `Refinement` (`93ffa832`, with
`committedProjection` retired), `Representation` (`c26eb817`, both residual sites repaired and the
four previously-unverified sites kernel-checked; two-target gate 3349 jobs green), `Numeric.KvReachable`
(`b6e17415`, file-green), `RecipLink` (`e1344f5f`), `AttnLoopOutput` (`c9f6c546`, window narrowed),
`AttnHeadWindowsValue` (`9f582724`).

Sync check owed: if the workspace's `Representation.lean`/`Refinement.lean` digests are not
`c26eb817`/`93ffa832`, copy both from `/home/socol/oc-tmp/corpus-fix-rep4/` into
`corpus-t1-f32/proof/corpus/Growable/` (a shell was interrupted mid-sync), then run the full
same-identity `lake build` and dispatch from its red list.

In flight: the `attn_domain_v2` carrier conjunct + KvReachable. Then: `AttnHeadRefines` final —
re-state premise 2 narrow and **prove** it from `plane_at_spec`, add the additive length export,
thread the carrier premise; premise 3 (the recip `some` arm) stays a documented hypothesis — then
KvReachable's module gate and the EngineAttnRefinement aggregate; and
`ReadLoop`/`ReadState`/`Classifier`/`Resources`/`LocateLoop` → AxiomAudit re-records (drop five
summary rows, add `SpillTagRecorded` + `reservedRepR_tagRecorded`) → full build.

Not adopted: the alternative Scan derivation `dfc23ff8` (2,298 lines); its `MatchResult`
discrimination-lemma finding is in `/home/socol/oc-tmp/corpus-fix-scan3/receipt.md`.

## 4 · Apeiron gate machinery

E1.W6: 39 rows migrated; Matvec green (`95501ca`). The last row is RowDotAdmission's redesign, in
flight (its `--controls` needs ~571 s against a 525 s inner budget; the redesign shares the package
and cache across legs without weakening the control). After it: the W6 exit record, then **E1.W7**
(delete `*_gate.py`, reject new ones). Open item: where the 29 row-level control results are
recorded — `docs/plan.md` has no row-level register.

## 5 · Penglai

W1–W7, W9, W10 landed; W8 implementation in flight under the frozen D1–D7 spec; the codec-split
commit is owed (files and digests in §2). Island W2-T1/W3-T1 are blocked on `src/physics.rs`
ownership (long-lived dirty work) — a lead/owner call before dispatch. Native surface rows hold.

## 6 · Webfold

Push lane tip `65524839`. This stretch: the acquisition harness boundary was repaired and pushed
(`7b47d964`; its gate had been red since `c2823471`; now PASS, audit 111 + 38 declarations with
empty cones), and D5's scalar seam was extracted and pushed (`65524839`;
`regen_extraction_state.sh --check` green, 4 admitted declarations, empty cone, two mutation
controls red by name). In flight: the D5 refinement slice (`AcquisitionProofs.lean`: `Admissible`
iff `d <= m and not own_parent and not repeated`, refusal mappings, totality, witnesses, plus the
generated body's own committed module so `AcquisitionMath.lean` stays byte-identical). E5 remains
blocked (no `#[path]` over `src/html_extract.rs`; Aeneas cannot host the containment transitions).
The registry landing and live-script fixes wait for the codex session to release the live tree.
**The supply-chain GPL red is the owner's call** (re-pin to an MPL-2.0 rev recommended, or amend
`deny.toml`).

## 7 · Intent

W32 T1–T7 and W33 T1–T4 landed; the store is literal and round-tripping; host snapshots (W36) and
budget reservations (W37) landed with their decisive controls, and both wave briefs plus the
controls scripts are committed and wired (`63a9aed`). In flight: nothing hard-blocked. Residuals:
`LiveDriver` still gets the default empty plate; the extraction candidacy needs `#[pure_core]` +
register rows (host `admit` core; budget identity closure, `RateCard::charge`, the disposition
machine, the settled fold); `docs/TRUST.md` owes the proc/cgroup and statvfs rows; `docs/waves/README.md`
is not indexed; W38–W40 next; `scripts/w25-controls.py` still carries the wide anchor and mutates
the live tree (narrow it or retire it in favour of `intent controls --group w25`).

## 8 · Owner decisions outstanding

1. **Webfold supply-chain GPL** — re-pin vs policy (the last CI red).
2. **W06-O5** — the trusted-checker size bound value (Intent states none; Apeiron's 5,000 does not
   fit its 12,195 measured lines; a ratchet at the measured value is recommended).
3. **W36–W40 milestone** — recommended as a new **M10 "bounded spend and device admission"**.

## 9 · Next actions, ordered

1. Corpus: verify the ref4 sync (digests above), run the full same-identity build, dispatch from its
   red list; fold the carrier task; then HeadRefines final.
2. Apeiron gates: RowDotAdmission on return → W6 exit record → E1.W7.
3. Penglai: commit the codec split; W8 on return; the island ownership call.
4. Webfold: fold the D5 refinement proof; then the README/check.sh/controls slice; the registry
   landing when codex frees; the live-script fixes.
5. Intent: the extraction-candidacy task; TRUST.md rows; W38.
6. Refill on every return; keep `/home/socol/oc-tmp/fleet-state.md` current.
