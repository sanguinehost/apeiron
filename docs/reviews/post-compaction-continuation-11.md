# Post-compaction continuation — programme dispatch (5)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md`, `docs/reviews/multi-repo-fleet-brief.md`. Source and gates are
the record. Live mirror: `/home/socol/oc-tmp/fleet-state.md`; escalation verdicts:
`/home/socol/oc-tmp/escalation/`. This file replaces `post-compaction-continuation-10.md`.

## 0 · Dispatch floor and lanes

Owner direction: 20-50 agents, refill on every return. The `intent` supervisor (`2c43f0c`,
`243355e`, `708e62d`, plus the ledger rows `77a50c0`) holds leases, a closed refusal set and the
`<6` urgent path; it is not yet systemd-run, so the orchestrator still refills. **Retired lanes:
`glm`, `glm-flash`, `xiaomi-pro` — the opencode agent definitions are deleted and every standing
mention removed (`apeiron` `75089bb`, `penglai` `0aead6d`, `song` `f6265e24`) because both balances
are empty.** Current lanes: Space Bunny default (free until 2026-10-05), DeepSeek Flash paid
escalation on a mandatory trigger or a twice-failed free-lane task, Gemini Flash non-default.

Two incidents to carry: (1) copying a **live tree file into the push lane while another worker is
mid-edit** shipped a dangling `pub mod promotion;` line (fixed `63ea51e0`) — before copying a
shared file (`lib.rs`, `Cargo.toml`), diff the live version against the lane's and check it compiles
in isolation; (2) the "concurrent writer" that startled two workers is the lead committing their
verified work — workers never commit, the lead does, and their working tree will change under them.

## 1 · Traps, lanes, exact commands

- Lean lane: `flock -w 600 /tmp/opencode/apeiron-lane.lock` + `systemd-run --user --scope -q
  -p MemoryMax=12G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 540`. Corpus builds run in
  `/home/socol/oc-tmp/corpus-t1-f32` (now carries the regenerated artifact) and the `corpus-fix-*`
  copies (older artifacts). Timed runs never share `CARGO_TARGET_DIR`. libtest filters on names.
  `rg` only. Retry `gate docs check` up to three times on the shared 30 s lock.
- Extraction regeneration, in-repo now:
  `cd /home/socol/Workspace/apeiron && AENEAS_HOME=/home/socol/Workspace/penglai/target/aeneas
  APEIRON_LAKE=0 ./proof/regen.sh --check` (~3-4 min; "extraction is fresh" is the green).
  The per-toolchain MIRI sysroot is symlinked at `target/aeneas-sysroot/nightly-2026-08-18`; a fresh
  host rebuilds it via Charon. `boundary-axioms.txt` lists the four admitted items and
  `audit-extraction.py` is fail-closed; `strip-modeled-axioms.py` must list every model axiom (a
  missing `I128.Insts.CoreConvertFromU64.from` cost one full check cycle).
- `gate`: `cargo run --quiet --manifest-path tools/gate/Cargo.toml -- docs check|run <row-id>
  [--manifest <file>]`; exit 0 accepted / 1 refusal / 2 environment / 124 budget. `EXTRACTION_DRIFT`
  from a `*_gate.py` means the source moved and the remedy is the gate's own `--prepare` (sequenced
  by the lead, never mid-wave).
- Clippy: `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` exits 0;
  the vendored `opsfold/Cargo.toml` duplicate-key message is non-fatal noise. fmt + clippy run
  before any commit that touches Rust.
- Song commit hook: subject ≤72 `<area>: `, body ONE paragraph, no dates, no "not yet", no ` -- `.
  Penglai's hook runs its whole check suite on commit (green required). Webfold pushes from
  `/home/socol/oc-tmp/wf-ci3` only.

## 2 · Landed since continuation-10

| repo | commits |
|---|---|
| apeiron | `dfbea06` port; `c3b0d63` scan.rs retirement §10; `301e065` engine-test clippy; `fa4fab5` extraction boundary+audit; `6143469` regenerated artifact; `94247e8` registry/manifest re-record; `6fc532b` doc corrections; `2f3c415` + `7096bdd` W6 batches 1-2; `0069dcc`/`7b6c7b7` index_check + map; `d0f39c9` continuation-10 |
| webfold (pushed lane tip `0ef248b4`) | `38425b9a` C1; `eab111d1` B3; `8da2af32` CI2; `86b03f10` metadata fix; `400aa82a` CI3; `cf46cf9b` D1; `542ccfdc` C3; `d99b5fdd` deny; `1f59e9e9` D2; `c2823471` D4+fmt; `eb459d15` C3 integration; `e0370b01` headless; `bd8c0161` E4; `558f31e0` C3c; `693eec37` E1; `a978b5e6` D3; `e9b7f9ee`+`63ea51e0` F1+fixup; `74a11d31` F4; `6ff986b8` E3; `0ef248b4` E2; plus the credential-control tool |
| intent | `a9c61a4`/`2c43f0c`/`243355e`/`708e62d` supervisor; `e07bc6d` W25 docs; `53f08fc` W32-T1; `bb8bd94` W32-T2; `a5e2d01` W32-T3; `662aa36` W33-T1; `77a50c0` ledger rows; `851cf25` oracle T1; `2bba1f4` AGENTS.md+VOICE.md |
| song | `9bd32b70`/`a2869f13`/`e971fbc8`/`aa20e7b8` A-row; `a4fb9d4f` wrap fix; `c6878680` A0.4 driver |
| penglai | `6edac67`/`30cd29a`/`34870ab`/`f6161a6`/`18d994c` native; `0aead6d` lane purge; `dbc8e96` T4b; `6b33132` W4-T1; `aaef953` W2-T2 + view tree |

## 3 · The extraction epoch (the point)

- **Accepted artifact: `ac563ff7c7398d05aeec3f8fe1b0c66bd4ec365cce621f4d34749949c8e1e1c9`** (996,624
  bytes; `proof/ApeironWord.lean`; regenerated from the ported source; `regen.sh --check` exit 0).
  Supersedes `c16a68f0` and `bd7c612a` (the latter kept at
  `/home/socol/oc-tmp/f32-narrow/proof/ApeironWord.baseline-bd7c612a.lean` for diffs).
- Boundary: exactly `attn.alloc_f32_storage`, `attn.KvState.attn_head_f32`,
  `core.f32.F32.is_finite`, `mixer.f32_carrier.q32_to_f32`; four Charon `--opaque` flags; the
  fail-closed audit reports 0 unlisted axioms; `scan.rs` symbols are gone (retired under §10,
  `retire_check` 4 records clean, `index_check` 67/67).
- The reciprocal weight path is proved in scratch `Numeric/RecipProbe.lean` (sha `7081a224...`):
  `recipFast_eq` (`⌊(N·R)/2^64⌋ + fixup = ⌊N/T⌋` for `R = ⌊2^64/T⌋ + 1`), the sharp bound
  `|w − e·2^32/total| ≤ 1/2`, `weight_at_abs_bound`, and a delegated-arm counterexample outside the
  nonnegative-numerator domain; closed cone, control red. The extraction-level link (`RecipLink`)
  is in flight — its output is what `AttnHeadRefines` needs for the Sol route-1 re-statement.
- Corpus green against `c16a68f0` (needs a re-check against `ac563ff7`, task in flight): sampler
  `6195f52e`; engine `4ebcff52`; softmax `11887417`; LoopScore `fe5b80a1`; ScoreStage `9b21a643`.
  `AttnLoopOutput` is at 25→8 red sites with one real change (`AttnAccInv`'s content clause becomes
  `acc.getD i = acc0.getD i + Σ…`); `AttnHeadWindowsValue` still needs its three referent-less
  theorems restated; both are with the attn-B worker.

## 4 · Corpus reconciliation wave (in flight)

Workspace `/home/socol/oc-tmp/corpus-t1-f32` now holds `ac563ff7`. Running: `Growable/Scan.lean`
re-derive; `Growable/Representation.lean` (sequenced after Scan); full-build triage (enumerates
every red with first errors and ownership); attn-region artifact diff (read-only; decides whether
the attn family needs a pass); `RecipLink.lean`. Owed after those land: `KvReachable` re-dispatch,
`AxiomAudit/Remaining.lean` + `Core.lean` cone registration (the live edits are +2/+49, build
blocked until Scan is green), the attn-A closure re-run, and the E1.W6 `--prepare` wave for the
table-bound gates that refuse `EXTRACTION_DRIFT` (ApplyPending, Capacity, plus any batch-3 refusal).

## 5 · Gate machinery (E1.W6)

Batches 1-2 committed (`7096bdd`, `2f3c415`); batch 3 in flight (Fwht and Gguf accepted; I64Codec,
I64Decode, KvAppend pending). `94247e8` re-recorded ten registry rows and three manifests against
disk after the retirement; `gate docs check` is clean (38 rows). Batch-3 rows commit when all five
return (the accepted/refused split is the batch-1 pattern: no drift on these modules so far).
W7 (delete `*_gate.py`) stays gated behind W6.

## 6 · Webfold / CI

Pushed lane tip `0ef248b4`; the Five-Pillar runs cleared the Nix disk wall; `--locked` is not broken
(the earlier alarm was a stale-tree comparison). `supply-chain` has **one red left: the pinned
apeiron rev carries GPL-3.0-or-later while `deny.toml` excludes strong copyleft — owner decision:
re-pin to an MPL-2.0 rev (recommended) or amend the policy.** Everything else there is disposed
(lopdf ignored with an expiry tied to the `superdocuments` pin, paste, sources via one exact URL,
wildcards pinned, transcode clarified). In flight: F3 selection, rendered wiring, E2 control
registration, plus the E4/D3/E3/E1/F1/F4/C3 landings already committed. Heavy lanes not started:
F2 plan proofs, E5 extraction refinement, D5 frames refinement, G1 cost bridge.

## 7 · Fleet queue at writing (~28 live)

apeiron: 5 W6 rows, Scan, Representation, triage, E3.W2 Regen draft adoption, attn-region diff,
RecipLink, attn-B finish. webfold: F3, rendered wiring, E2 control registration (C3 controls just
landed). intent: W2b-T2 grading, W32-T4 curation, T6 recall, T7 plate wiring, W33-T2 agents
browser. song: A0.6.EO body lowering, A0.10 constant-offset, K2.1.1 Slang driver library. penglai:
W5-W1 wiring, W6-T1 scheduler, W9-T1 ledger, W2-T3 bank display. Next refills when those return:
batch-3 commit + batch 4, W32-T5 (after T4), penglai W7/W8 (after mod.rs frees), the triage's
per-module tasks.

## 8 · Open decisions and residuals

- **GPL re-pin vs policy** (CI; the one owner decision outstanding).
- **W06-O5 trusted-checker bound**: no value registered; the design is ready
  (`check_budget` + a `TRUSTED_CHECKING` manifest + one constant + a seeded control); the measured
  population is 12,195 lines so the Apeiron 5,000 does not fit — the value is the owner's/lead's.
- `oracle_ring6_randomised.rs:182-190` carries a comparison that can never fire (call vs
  `call == value`); fix or retire it (found by W2b T1).
- Song `tools/song/controls.txt` is still another stream's; owed entries live in receipts (a115,
  a232, a240, a252, a253, the A5.10 fallback).
- E3.W1 LmHead stays blocked on its proof-search ceiling (cycle cap spent); a fresh session on a
  different model is the sanctioned route.
- Penglai `src/physics.rs` and friends carry another writer's uncommitted work; island tasks that
  touch them sequence behind it.
- The stray root files (`test_closure_pointer`, `libtest_closure_capture.rlib`, `.held-*`, `a.js`,
  `VP_VULKANINFO_*.json`) are doc-purge W3 material, not blockers.

## 9 · Next actions, ordered

1. Owner: the GPL call. Then the final CI push and watch.
2. Take the triage's red list; dispatch per-module reconciliation against `ac563ff7` (Scan →
   Representation → KvReachable; attn family per the diff); re-run the same-identity `lake build` +
   `AxiomAudit` when green.
3. Commit W6 batch 3 (five rows) when all return; open the `--prepare` wave for the drift-refused
   gates; then batch 4.
4. Land the in-repo extraction harness verdict as the landing unit (already committed: `fa4fab5` +
   `6143469`); the scratch `f32bound/landing` set is obsolete — do not re-apply it.
5. Fold each return: commit on green, re-run the gates, refill.
