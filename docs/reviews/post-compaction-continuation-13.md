# Post-compaction continuation — programme dispatch (7)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md`. Source and gates are the record. Live mirror:
`/home/socol/oc-tmp/fleet-state.md`. This file replaces the content of continuation-12 (kept at
`docs/evidence/reviews/post-compaction-continuation-12.md` as an archived receipt).

## 0 · Scope, and what killed the last fleet

- **Streams**: this stream runs **apeiron** first, with **intent**, **webfold** and **penglai** as
  secondary lanes. **Song is handled in a different chat/stream**: never dispatch, commit or report
  Song work here. Fleet target: 32 agents total across the operation, split with Song; keep this
  stream at **≤16 active lanes**, refilled in batches of ≤4.
- **Failure mode measured (2026-10-02 evening): BTRFS ENOSPC from the copy load** (not tmpfs;
  `/tmp` was 19 MB used). A reboot cancelled ~50 sessions; a rebalance regained ~142 GB unallocated.
  **Rules now, in every brief**: scratch copies live under `/home/socol/oc-tmp` (never `/tmp`); one
  private `CARGO_TARGET_DIR` per task (under `/home/socol/oc-tmp`); copy the 38 GB intent tree with
  `rsync -a --exclude target/ --exclude proof/.lake/ --exclude .lake/`; prune a copy at its receipt
  (preserve small cited files); monitor btrfs chunk/metadata headroom, not just `df`; long gates run
  under `systemd-run --user --scope` so a client restart cannot orphan them. The purge agent writes
  `/home/socol/oc-tmp/disk-rules.md`.
- Any session started before the reboot is dead. The relaunched lanes below are the live fleet.

## 1 · Owner DoD (deadline Oct 5; unchanged)

1. **apeiron beats hashtrinity on every metric** and foldhash/hashbrown plus the C libs on the G1
   matrix. Frontier: spill port landed; insert (~119x) and resident cells (reproduced 1.68–2.63x)
   in work.
2. **intent replaces OpenCode**: interface + invoked agent per the design (ratatui; the two parity
   audits are the map). **Plugins are deferred by owner direction** (internal tool for now).
   **MCP and the API surface for cross-agent interaction/orchestration are the priority.**
3. **webfold**: PR at 100% on everything webfold and Firecrawl currently cannot parse (the
   54-fixture `check_review.py` set is in-repo; the `bench.rs` dataset is not), ARIA line;
   handover-ready for Clay.
4. **penglai**: island water slice at 120 fps with the `~/Downloads` reference look; viability
   coupled to (1). Measured 29.15 fps, CPU-bound; apeiron cost on that path is zero.

## 2 · Reconciliation state (what is true now)

- **Landed during/after the outage**: webfold D5 audit `3f3fa55c` (wf-ci3, pushed); intent `d5e83a1`
  (gates.py single_dispatch); apeiron `c95aa00` + `0be5c85` (resident-claim corrections). Earlier:
  W6 closed `daede3f`; corpus Resources `b014e0b5` (workspace); E6 port `dfbea06`; intent splits and
  waves (`e7c3c6c`, `d667410`, `8f30760`, `8b3c2fa`, `2fadfe9`, `1b404c1`); E2 evidence fold
  `0053850`.
- **Repos**: apeiron `0be5c85` (~90 dirty; the repo corpus is a third state versus the workspace —
  reconcile per file before landing); intent `d5e83a1` (~1 dirty); penglai `144c63d` (~117 dirty,
  physics island work); wf-ci3 `3f3fa55c`.
- **Apeiron corpus**: true red list (oleans deleted before trusting a build):
  `Numeric.AttnHeadRefines` + `Growable.ReadState` (19 diagnostics), with `ReadLoop`, `LocateLoop`,
  `Read`, `ReadComposition`, `Classifier` masked behind ReadState. Base `corpus-t1-f32` carries the
  carrier conjunct (`AttnHeadWindowsValue 7fca91b1`, `AttnSoftmaxPerturb 98a7e14a`,
  `AttnHeadWindowsScore 6bc11cca`, `KvReachable ba5114da`) and Resources `b014e0b5`.
- **E6**: port landed (`dfbea06`; measured ratio 1.6639 ≤ 2.2 in one interleaved release run,
  correctness counters zero); W3.T1 (placement invariant + `SpillProbeBoundAbsentSound` + RepR
  summary retirements) relaunched; insert spike relaunched; the resident fix implements the spike's
  §5 first task; W4-T2 comparator cells relaunched. G1's receipt of record is stale (its refresh is
  `APEIRON-REANCHOR-W5-T1`); current cut measured `registered=2592 load_bearing=2268
  applicable=1450 pass=550 fail=900 full_acceptance=false`.
- **E1**: W6 closed; `gate docs check` 38 rows / 0 mismatch; E1.W4's aggregate lives at
  `/home/socol/oc-tmp/e1w4-agg` and its default-budget defect (a 540 s row is unattemptable under
  the default 540 s invocation) is being fixed in `e1w4-fix2`; E1.W2 `admit` extraction partial at
  `e1w2-admit`.
- **Intent parity audits** (both landed): UI `/home/socol/oc-tmp/int-parity-ui/parity.md` — 0/15
  top-level rows meet the standard (8 covered / 2 partial / 26 absent of 36); **G1 (the shell never
  constructs `LiveDriver`) blocks all 11 cutover workflows**; tasks `PARITY-UI-G1-T1`..`R12` with
  files/gates/controls; doc corrections D1–D3. Agent `/home/socol/oc-tmp/int-parity-agent/parity.md`
  — 29/16/28 of 73; top gaps: headless sessions truncate (`P1-T3`), MCP is a catalog (`P0-T3`),
  `spawn` runs no loop (`P2-T3`), no `--format json` (`P0-T4`, queued behind `main.rs`), `read`
  raw / `edit` exact-only (`P0-T1`, `P0-T2`). New controls need registration in
  `crates/intent-cli/src/tooling/route/registration.rs` first.
- **Webfold**: D5 refinement proof landed (`341cbeae`, pushed); fresh-leg cold budget blocker
  (180 s; measured 195–216 s) needs a warm seed or an owner-approved revision; no weight control
  for `edge_*` yet. Coverage partial at `/home/socol/oc-tmp/wf-parse-audit2/coverage.md`
  (I1 = 54 Firecrawl fixtures via `tools/corpus_review/check_review.py --require-webfold-pass`;
  I3 `firecrawl_compat.rs`; I4's dataset is not in checkout).
- **Penglai**: `/home/socol/oc-tmp/penglai-water/findings.md` — 29.15 fps at 1280×720
  `core:reedbank`; GPU passes 0.588 ms; CPU water sim 18.4 ms + simulate 11.9 ms + present 3.9 ms;
  `resolveWaterAscii` absent (12 glyphs already in `assets/glyphs.ron`); slices recorded; GGX is a
  lead call — keep the measured Blinn default, opt-in only.

## 3 · Live fleet at the cut (refill on returns; ≤16)

- apeiron: AttnHeadRefines (`attn-head-final3`), ReadState (`corpus-fix-readstate3`), insert spike
  (`e6-insert-spike3`), W4-T2 cells (`e6w4-t2b`), resident fix (`e6-resident-fix`), E1.W4 fix
  (`e1w4-fix2`).
- intent: W38 impl (`int-w38impl3`), oracle split (`int-split-oracle2`), live driver
  (`int-parity-g1t1c`), headless sessions (`int-p1t3c`), MCP transport (`int-p0t3c`), spawn
  (`int-p2t3c`).
- webfold: coverage finish (`wf-parse-audit2b`). hygiene: purge (`disk-rules.md`).

## 4 · Traps and exact commands

- Lean lane: `flock -w 600 /tmp/opencode/apeiron-lane.lock` + `systemd-run --user --scope -q
  -p MemoryMax=12G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1` + `timeout 540`.
- Corpus workspace `/home/socol/oc-tmp/corpus-t1-f32`; workers copy it, never write it; the lead
  syncs digests. **Delete target oleans before trusting a build** — stale replay produced a false
  red list.
- Timed runs bind to a source identity: private `CARGO_TARGET_DIR` or `touch src/lib.rs`.
- **fmt instrument**: `cargo fmt -p <crate> -- --check` is canonical; a raw
  `rustfmt --edition 2024` uses style edition 2015 and reports false diffs.
- Restoring a mutation with `cp -p`/`cp -a` preserves mtime; `touch src/lib.rs` before re-running.
- `tools/gate`'s directory row covers the whole spec directory: editing anything under `tools/gate/`
  moves its identity — re-point the row.
- `find` is a broken shim; use `rg`/globs. `/home/socol/.local/bin/rg` misreported a filename once;
  use `/usr/bin/rg` when a path looks odd.
- Regen: `AENEAS_HOME=/home/socol/Workspace/penglai/target/aeneas APEIRON_LAKE=0
  ./proof/regen.sh --check`.
- the webfold live tree is codex-owned: never edit it; use `wf-ci3` and lane copies.

## 5 · Next actions

1. Fold the live lanes' returns as they arrive; refill to ≤16 in batches of ≤4; apeiron first.
2. Corpus: AttnHeadRefines green → KvReachable module → ReadState → the ReadLoop family
   re-derivation (5 files, ~4,500 lines; `spill_read_loop0` is absent from the artifact — the
   Resources receipt has the shape map) → AxiomAudit re-records → full build; then reconcile the
   repo corpus (third state) per file and land it.
3. E6/P1: resident fix + insert fix land; W3.T1/T2a/T2b; W4-T2 cells; then the W5 acceptance run
   with `--controls` and an honest full-failure report. "Every column" is not scheduled until the
   insert column and the resident band land.
4. Intent P2: G1-T1 then G1-T2/T3; P0-T3 MCP; P2-T3 spawn; P1-T3; P0-T1/P0-T2; P0-T4 JSON queued
   behind `main.rs`.
5. Webfold: finish coverage → ARIA task list → execute to 100% on the in-repo set → handover pack.
6. Penglai: ASCII resolve; 120 fps CPU cuts per the findings' plan.
7. Owner items: the webfold GPL call, W06-O5 bound, M10 milestone; Clay handover; the fresh-leg
   warm-seed decision.
