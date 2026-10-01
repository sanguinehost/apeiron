# Post-compaction continuation — programme dispatch

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md` (E1–E6), `docs/reviews/multi-repo-fleet-brief.md`.
Source and gates are the record; a document is corrected to them. Live mirror:
`/home/socol/oc-tmp/fleet-state.md`. Escalation verdicts: `/home/socol/oc-tmp/escalation/`
(`e6-spill-contract-answer.md`, `e6-f32-extraction-answer.md`, `e6-audit-final-answer.md`).

## 0 · The failure this session and the rule it sets

The fleet drained to one stale session and the lead ended turns waiting instead of refilling; the
owner woke to no agents working. **Rule: at every return, check `/api/session/active` and refill to a
dispatch floor of at least six Space Bunny tasks before ending a turn.** A turn that ends below the
floor without a stated blocker is a failed turn. The 50-concurrent-agent target is exercised by
Intent's harness (`crates/intent-cli/tests/evaluation_worker_width.rs`, 50 workers green) — the
campaign itself must hold that discipline.

## 1 · Routes and discipline

- **Workers**: `openrouter/stealth/space-bunny-alpha` via `agent: general`, free until **2026-10-05**.
  `deepseek-flash` only on a mandatory trigger or explicit owner authorization.
- **Sol**: `/home/socol/oc-tmp/bin/sol.sh <prompt-file> [pro|<model>]`, answers recorded under
  `/home/socol/oc-tmp/escalation/`. Default `openai/gpt-6-luna-pro`. **The lead does not idle
  waiting on the owner — escalate, record, act, report.** Three verdicts this session are recorded
  there and transcribed into `docs/plan.md` E6 and the read-path epic.
- ≤3 simultaneous heavy Lean checks under `flock /tmp/opencode/apeiron-lane.lock`; every gate
  declares a numeric wall-clock budget; 600 s is a failure ceiling. A cold `proof/` `lake build`
  cannot finish in that budget — the barrier uses the warm `proof/corpus/.lake`.
- **`find` and `grep` are broken shims**; use `rg`, `rg --files`, `/usr/bin/find`. A claim of
  absence names its command.
- Song's commit hook: subject ≤72 chars with an `<area>: ` prefix, one paragraph ≤12 lines, bans
  "not yet" and ` -- ` asides; write messages to a file and `git commit -F`.
- `AENEAS_HOME=/home/socol/Workspace/aeneas` in the environment has **no executables** (exit 126);
  run regen with `env -u AENEAS_HOME` plus the tool home. The working Aeneas is
  `/home/socol/Workspace/penglai/target/aeneas` (sha `33d0ce1a…`); the rebuilt
  `/home/socol/oc-tmp/aeneas-pin-home` (sha `2a0d7827…`) regresses on `core::array::repeat`.
- Committing another owner's dirty file: build the staged blob as HEAD plus your own lines
  (`git hash-object -w` + `git update-index --cacheinfo`). Foreign hunks exist in
  `proof/regen.sh` (+17), `proof/extract/lib.rs` (+56), `proof/ApeironWord.lean`.

## 2 · Owner decisions in force

1. **Kalman reading: yes** — belongs in Song's provenance, formalise when suitable (task in flight).
2. **PATH shims: permanent architectural fix** (`c00b0b6`), not a patch.
3. **Sustained width is a requirement on Intent** (50 concurrent agents): harness landed and green
   (`607e77a`), with the lease half and four controls; the P1 completion check is live again — a
   development run had disabled it with `if false &&`, and with it disabled two controls passed
   silently.
4. **webfold's callers are untrusted**: the SSRF guard landed (`701de0ea`) with the server-side
   policy, loopback refused by default; the PR carries it.
5. **The hashtrinity gap is the highest-value item** — E6 in `docs/plan.md`. Measured: the resident
   read is at parity (1.06 vs 1.05); the gap is the spill read; the flat open-addressed prototype
   reads **1.34 ns/key against 8.14, 6.06×** (E6.W2d/W2e). Sol fixed the port contract (shape A:
   bounded probe as absence certificate, `Grow` = full probe cycle, summary retired,
   `SpillProbeBoundAbsentSound`), transcribed in `docs/epics/read-path-specialization.md`.
6. **f32 boundary (Sol)**: the binary32 chain hangs from the root `crate::word_attn_head`; four
   Charon `--opaque` patterns make the corpus regenerate with exactly four admitted boundary
   axioms — `attn.KvState.with_carrier`, `attn.KvState.attn_head_f32`,
   `mixer.f32_carrier.q32_to_f32`, `core.f32.F32.is_finite`. Claims are relative to the boundary;
   `append`'s F32-storage overflow guard is not proved; the corpus covers the production constructor
   `KvState::with_carrier` and records `KvState::new`/`carrier_from_env` as non-claims.

## 3 · Landed this session (by repo)

- **apeiron**: `470e3ab` §8 numeral rule (rule D 38 sites) · `6ed4ff4` unit vocabulary (D 38→15,
  self-test 26/19) · `55421fa` tag = top bits of the spread place (adversarial compares 107.25→1.36)
  · `3a69a4f` + `4642760` attn/f32_core loop-shape reshapes · `5a90e13` the redundant FunsAnalysis
  hunk dropped from the Aeneas patch (the internal assert, fixed) · `4d887f0` rebuilt tool recorded ·
  E6 measurement/decision commits `d05e22d`…`7c17903`.
- **webfold** (PR `feat/tiered-algebraic-extraction`, remote head `ee79d13f`): proxy egress guard and
  bounded bodies · flare session-lock fix (the branch's red test now green) · bundle materialize
  resumable · SSRF hop guard + untrusted-client policy · crawl contracts (max_depth/retries/metrics)
  · `Retried` emitted and counted · diff `modified` classification · file:// resolver · ledger
  re-reviewed through `APP-BOUND-01`, APP-CRAWL-01 closed, checker now at `APP-MCP-01`. Merged-tree
  lib **384/0**.
- **song**: `43bf652b` vacant-path merge obligation · `2479399e` intern hit-path byte re-check.
- **penglai**: `1128383` citation gate scopes symbols to the citation's own clause.
- **intent**: `607e77a` the 50-worker width harness + selection hardening.

## 4 · The extraction corpus: state and landing conditions

The pipeline regenerates in the scratch (`/home/socol/oc-tmp/f32bound`): `sh proof/regen-a3.sh` with
`AENEAS_HOME=/home/socol/Workspace/penglai/target/aeneas` exits **0** in 68.8 s; the generated
`ApeironWord.lean` is sha256 `bd7c612a…`, byte-identical over two runs, with exactly the four
boundary axioms. The scratch holds the complete diff set: the `word_kv_state_new` wrapper →
`with_carrier(.., Carrier::Q32)`, the model patch's `I128.Insts.CoreConvertFromU64.from`, the ALLOW
entry, `proof/boundary-axioms.txt`, `proof/audit-extraction.py` (eight controls), `regen.sh`'s
`audit()` + the four `--opaque` flags, the README boundary block, and `AxiomAudit/Core.lean`'s cone.
**Landing conditions**: (1) the exact axiom name form the cone expects (APEIRON-LAND-T1 is settling
it; the insert used `apeiron_word.*` and `Lean.collectAxioms` may prefix the module — fail-closed
either way, one `#print axioms` probe settles it); (2) the four `--opaque` flags land with the
corpus; (3) the tool identity — the translating Aeneas has no recoverable source on this box, the
rebuilt one regresses on `core::array::repeat`, so landing needs a rebuild fix or an explicitly
recorded tool identity. `proof/lake build` cold cannot fit the budget; use the warm
`proof/corpus/.lake`.

## 5 · In flight at compaction

APEIRON-LAND-T1 (name form + landing patch) · APEIRON-E6-W2F (flat region + read path, scratch) ·
WEBFOLD-LEDGER-T8 (from `APP-MCP-01`) · SONG-KALMAN-T2 · PENGLAI-HEADERS-T5.

## 6 · Next actions

1. Land the extraction integration (T1's patch, foreign hunks preserved), regenerate, run the warm
   corpus build, settle the cone names, commit. This is the longest-standing blocker.
2. Continue the E6 port in ≤3-file slices per the epic's plan rows: region layout, read path,
   publication fixtures, proof replacement (`SpillProbeBoundAbsentSound`), extraction re-link,
   integrated benchmark. Grade the harness at parity as the gate for each slice.
3. Refill the fleet at every return (the floor is six); keep the ledger batches moving (T8 then the
   next; after the findings rows, the `source_files` inventory has ~30 stale rows owned by other
   writers' uncommitted merges — that needs an owner decision).
4. Repair the rebuilt Aeneas's `core::array::repeat` regression (`Translate.ml:390`) so the tool
   identity becomes reproducible; until then the translating binary is recorded by hash only.
5. Song/Penglai/Intent returns as they land; each gate re-run before commit.
