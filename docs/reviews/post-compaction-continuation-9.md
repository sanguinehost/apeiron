# Post-compaction continuation — programme dispatch (3)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md` (E1–E6), `docs/reviews/multi-repo-fleet-brief.md`. Source and
gates are the record; a document is corrected to them. Live mirror: `/home/socol/oc-tmp/fleet-state.md`
(fifteen deltas). Escalation verdicts: `/home/socol/oc-tmp/escalation/`. This file replaces
`post-compaction-continuation-8.md`.

## 0 · The dispatch floor

Owner direction: **20–50 agents running at all times; six is the floor, not the target.** At every
return: verify the active set, integrate what returned, and refill before ending a turn. A turn that
ends below the floor with dispatchable work queued is a failed turn. The durable mechanism is under
construction: `intent/docs/SWARM.md` §8 (the fleet supervisor) plus the landed state module
(`crates/intent-runtime/src/fleet.rs`, commit `e7d245a`); the next tasks add leases, launch,
controls and the `fleet` CLI. Two server restarts killed worker shells mid-run; several sessions
were resumed with `sessionID` — when a return is missing, check its scratch before re-dispatching.

## 1 · Routes and discipline

- Workers: `openrouter/stealth/space-bunny-alpha` via `agent: general` (free until 2026-10-05).
  Sol: `/home/socol/oc-tmp/bin/sol.sh <prompt-file>`; answers under `/home/socol/oc-tmp/escalation/`.
  Confirm the prompt file exists before launching (two launches raced the write and logged
  "No such file" while still answering; re-run clean when provenance matters).
- Heavy Lean: lane lock `/tmp/opencode/apeiron-lane.lock` + `systemd-run` caps, 600 s ceiling.
  Corpus builds run in copies: a prepared workspace exists at `/home/socol/oc-tmp/corpus-t1-f32`
  (staged artifact + patched model + warm `.lake`); copy it per worker.
- **Timed runs never share `target/` with another worktree** (register row, apeiron `5cd5541`).
  Force a rebuild (`touch src/lib.rs`) and check the cached warnings' line numbers before quoting a
  figure; the harness header in `examples/spill_read_probe.rs` states the rule.
- **Same-crate tasks serialize.** Two webfold tasks (`src/mcp.rs`, `src/fetch.rs`) made each other's
  gates unrunnable mid-flight; cargo runs take `/tmp/webfold-repair-cargo.lock`, source edits need
  sequencing or private copies.
- **libtest filters on test names, not module paths.** A gate whose selected tests lack the token
  runs zero tests and reports green (`INTENT-SUP-W1-T1` caught this live); prefix test names.
- Tool traps: `find`/`grep` are broken shims (`rg` only); `cut`/`basename`/`tr` are PATH-unreliable
  here; `local path=x` in zsh clobbers `$PATH`; `read IFS=$'\t'` collapses tab runs (use
  `awk -F'\t'`); `echo ===` is zsh `=cmd` expansion (quote it).
- `AENEAS_HOME` in the environment points at `/home/socol/Workspace/aeneas` (`557f7a15`), whose
  Lean backend does **not** take `proof/patches/aeneas-lean-model.patch`; the model build must read
  `target/aeneas` (→ `/home/socol/Workspace/penglai/target/aeneas`, `11283835`). Unset the variable
  or pass the right home; three revisions are in play and `proof/patches/PINS` records `505b6ca`.
- Song commits: subject ≤72 with an `<area>: ` prefix, one contiguous ≤12-line block, no dates,
  no "not yet", no ` -- ` asides; write the message to a file and `git commit -F`.

## 2 · Landed this session (commits)

| repo | commit | what |
|---|---|---|
| apeiron | `715592c` | spill gate of record graded on the matched ratio spill/resident ≤ 2.2; the unported tree is RED (exit 101, ratio 6.5–6.9) |
| apeiron | `5cd5541` | failure-register row: shared `CARGO_TARGET_DIR` poisoned timed builds (a false 2.02 ns/key "pass") |
| apeiron | `a1d7853` | enforcement map and plan name `retire_check.py --check` (the bare command exits 2) |
| apeiron | `5323f96` | E6 port decomposed into ≤3-file serialized tasks W2-T1/T2/T3 + gate-of-record registration |
| apeiron | `3452500` | E5 corrected: the Ring 6 selection gate is landed (`df33ee9`/`14c6932`); W2b adds the oracle slice |
| apeiron | `c46ef98` | `retire_check` M3 resolves `argv[0]` on PATH, M4 resolves receipts against the evidence ledger |
| apeiron | `ecf5388` | `tools/gate` adapter: `RunRecord`/`RunRefusal`/`run_captured`, streaming sha256, 21 tests, 8/8 mutation controls |
| webfold | `0dbd987a` | ledger rows HARNESS-17..CLI-04; frontier now `.github/workflows/ci.yml` (31 stale inventory rows) |
| webfold | `ef0d2295` | CI fixes: audit action, `--override-filename`, `deny.toml` duplicate key, free-disk step |
| webfold | `424e222d` | origin tip: merge of T9/T10 + CI fix, **pushed** to `feat/tiered-algebraic-extraction` |
| song | `59ab3ea3` | `src/std` suite + io test (root cause: `io.slang` redefined `print`); `docs/ROADMAP.md` A5.8 corrected |
| song | `858ef133` | CAVEATS step-4 row removed, `ConeFold.lean` cite corrected to `Apeiron/Decider/Model.lean` (`plans/` is gitignored; its BOARD tick is local-only) |
| intent | `fc66a71` | W20/W27 closure records discharge the emitter residual |
| intent | `af3ad9a` | `docs/SWARM.md` §8: the authenticated OpenCode contract + `supervisor_opencode` gate |
| intent | `e7d245a` | `fleet.rs` state module + `fleet_state.rs` gate (11 tests; quarantine semantics per Sol) |
| intent | `52f8938` | drop the useless `format!` calls in the supervisor gate; `intent-cli` clippy clean again |

## 3 · In flight / resumed after the restarts

- **E6 port landing** (`/home/socol/oc-tmp/e6-land/`): `port.patch` (100 KB) + `backup/`; the live
  tree currently carries only `src/table/probe.rs` staged (37+/23−). Must keep commit `55421fa`'s
  top-bits fingerprint while porting; barrier: `cargo test --locked --lib --test table`,
  `cargo run --release --example spill_read_probe` judged on ratio ≤ 2.2, slow-spill reject/restore
  control, fmt/clippy. Lead commits after the gate.
- **CORPUS-FIX-SCAN / ATTN / SAMPLER** (`/home/socol/oc-tmp/corpus-fix-*`): Scan = fingerprint
  staleness in `Growable/Scan.lean`; Attn = `k`/`v` → `carrier`/`storage`; Sampler stopped at the
  class-D boundary (see §4). Each gates `lake build <target>` in its own copy.
- **ENGINE-T1** (`engine-t1`): per-dispatch-site pool-wait attribution, driver
  `examples/engine_profile.rs`, two-run repeatability control.
- **WEBFOLD-C1**: `src/fetch.rs` + `tests/fetch_transport_review.rs` +
  `tools/webfold/fetch_transport_controls.py`; decided redirect strategy is webfold's own per-hop
  loop + byte cap (C0 findings in `/home/socol/oc-tmp/wf-c0/contract.md`). B3 (coordinator routing)
  is implemented and held for commit until C1's build is clean.
- **ORACLE-T1 mutant rerun** (intent): oracle trio in place (`oracle.rs`, `oracle_cases.rs`,
  `oracle_ring6_randomised.rs`); the blocked mutation control can run now that `intent-runtime`
  compiles.
- **INT-W06R-1**: verify post-restart; resume if dead.
- **SONG-CAVEATS landed** (`858ef133`): the CAVEATS row is removed per the repo's §5, the BOARD E6
  tick lives only in the gitignored `plans/`, and `corpus.slang`'s `ConeFold.lean` cite is corrected
  to `Apeiron/Decider/Model.lean`; residual: `src/prover/ring/prover0.slang:34` cites a gen1
  `Ring.lean` path, and the cite check built for this task lives only in scratch.
- **CI watchers**: webfold runs on `424e222d` (`ci` 36945322595, `supply-chain` 36945322526); PR #21
  not yet checked for the same CI fixes.

## 4 · Open decisions

- **Class-D — answered (act on it).** Sol: make `with_carrier` transparent with the geometry checks
  before any opaque call, and move the boundary to the F32 **storage allocation helper** (an opaque
  helper returning `KvState` or `Result KvState` is wrong — it would hide the facts the geometry
  theorem needs); replace `crate::attn::_::with_carrier` in the opaque list with that allocation
  symbol and keep the other three F32 patterns. Then the Q32 refusal branches reduce through the
  extracted checks and the geometry fields stay visible; the seven theorems re-derive. If the whole
  constructor is kept opaque, withdraw the seven as non-claims in the boundary receipt/`AxiomAudit`
  at grade `anchor` (do not condition them — that only relocates the premise). The unrefined F32
  branch of `word_kv_append_ok` gets the same non-claim disposition until `q32_to_f32`/`is_finite`
  have transparent definitions or audited specifications. Answer file:
  `escalation/class-d-f32-boundary-answer.md`. The action is an extraction change (a small
  behaviour-preserving refactor if no allocation helper exists, the opaque-set edit, regenerate) —
  dispatch `F32-BOUNDARY-NARROW`; the regenerated artifact changes the landing sha and re-shapes
  the sampler/attn reconciliation.
- **Penglai headers**: the owner-committable patch is staged (`/home/socol/oc-tmp/penglai-headers/`,
  six tracked files, comment-only, control verified); `git am` refuses on the dirty tree — use route
  B (`update-index --cacheinfo`) and **drop `src/view/field.rs`** (in flight; its header rides with
  its owner).
- **Webfold ledger inventory**: 31 stale `source_files` rows, first failure
  `.github/workflows/ci.yml` (2952 → 3816 bytes); owner decision on refresh policy.
- **Song follow-ups**: six control rows (a113/a114/a116/a117/a118/a121) carry frozen census
  literals as signatures (load-bearing reds, stale signatures); the `access_budget` ratchet 1120 is
  10 under the measured 1130 (7 from the untracked `dep_closure.slang`); `docs/ROADMAP.md` line 183
  and lines 543–550 contradict each other; `ST[ST_INBODY] = 0` has no registered control.
- **Intent integration queue**: the W25 pair (`model_switch.rs`, `model_loop.rs`, `main.rs` +
  `w25_cli.rs` fixture) is green and uncommitted; the oracle trio awaits its mutant rerun; settle
  `cargo fmt` for `oracle.rs`/`oracle_cases.rs` at integration and leave `main.rs`/`w25_cli.rs`
  alone (already reflowed once by a foreign `cargo fmt -p intent-cli`).

## 5 · The formal chain (the point of the move)

- **Model**: the staged landing item 4 is `proof/patches/aeneas-lean-model.patch` with `F32`
  (`abbrev F32 := BitVec 32`, a definition, no cone growth) and the `I128` `From<u64>` def; the
  19560-byte version is at `/home/socol/oc-tmp/f32model/aeneas-lean-model.patch`, and
  `proof/model-patches.sh`'s header counts nine files, not eight. The patched model builds
  (1705 jobs); the corpus error count dropped 26 → 3 with it.
- **Landing unit** staged at `/home/socol/oc-tmp/f32bound/landing` (8 files + artifact
  `bd7c612a…`); preflight PASS with defect F-1: `git apply` carries no modes, so landing must
  `chmod 755 proof/audit-extraction.py` (or `git update-index --chmod=+x`) before the commit.
  Acceptance: `proof/regen.sh --check` fresh, `audit-extraction.py` exit 0 (4 admitted), then warm
  `cd proof/corpus && lake build`.
- **After the port lands**: regenerate (E6 W3-T3), then the subtraction S1..S6 from
  `/home/socol/oc-tmp/e6-subtract/` (S1 corpus → S2 regen → S3 gate source paths → S4 prose →
  S5 delete `src/` last → S6 register in `removals.json`), and `SpillProbeBoundAbsentSound` derives
  absence from the new placement invariant. 48 h acceptance: production spill reads meet the
  existing ≤1.0 interval criterion against the pinned predecessor on the registered workloads, with
  correctness and mutation/restoration controls.

## 6 · Next actions, ordered

1. Class-D answer → execute; unblock SAMPLER/ATTN and fix the W3 slice scope.
2. Port landing barrier → lead commit; corpus fixes → land with the artifact; landing unit (with
   F-1) → `regen.sh --check` + warm corpus build → commit; then W3-T3 regen.
3. C1 gate → commit B3+C1; oracle mutant → commit the oracle trio; commit the W25 pair.
4. CI: watch the `424e222d` runs; apply the same fixes to PR #21 if it carries the broken
   workflows; report run conclusions.
5. Resume INT-W06R-1 / SONG-CAVEATS if the restart killed them; song follow-ups; ENGINE-T2
   (makespan) after T1; ORACLE-T2 after T1; supervisor tasks 3–4 (leases/launch/controls).
6. Refill to 20–50 at every return; update `/home/socol/oc-tmp/fleet-state.md`; never end a turn
   below the floor while dispatchable work exists.
