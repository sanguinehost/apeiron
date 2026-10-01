# Post-compaction continuation — programme dispatch (2)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md` (E1–E6), `docs/reviews/multi-repo-fleet-brief.md`. Source and
gates are the record; a document is corrected to them. Live mirror: `/home/socol/oc-tmp/fleet-state.md`.
Escalation verdicts: `/home/socol/oc-tmp/escalation/` — `e6-spill-contract-answer.md`,
`e6-f32-extraction-answer.md`, `e6-audit-final-answer.md`; a night-review covering the direct-port
question and keep-alive enforcement (`review-night-and-keepalive.md`) is with the owner's review
point and its verdict is pending.

## 0 · The dispatch floor

The fleet drained overnight and the lead ended turns waiting; the owner woke to no agents working.
**At every return: check the active set and refill to a floor of at least six Space Bunny tasks
before ending a turn.** A turn that ends below the floor without a stated blocker is a failed turn.
The review point is being asked for an enforcement mechanism; implement its verdict on arrival.

## 1 · Routes and discipline

- Workers: `openrouter/stealth/space-bunny-alpha` via `agent: general` (free until **2026-10-05**).
  `deepseek-flash` only on a mandatory trigger or explicit owner authorization.
- Sol: `/home/socol/oc-tmp/bin/sol.sh <prompt-file> [pro|<model>]`; record answers under
  `/home/socol/oc-tmp/escalation/`. The lead does not idle waiting on the owner.
- ≤3 heavy Lean checks under `flock /tmp/opencode/apeiron-lane.lock`; every gate declares a numeric
  wall-clock budget; 600 s is a failure ceiling. A cold `proof/` `lake build` does not fit it — use
  the warm `proof/corpus/.lake`. For disposable corpus builds, the CORPUS-T1 recipe recreates a
  workspace in ~7.5 s (`cp -a proof` + `target` symlink; script `/tmp/opencode/corpus-t1-run2.sh`);
  restore `proof/corpus/Policy/Core.lean` from HEAD **in the copy** — it is mid-refactor by another
  writer and fails 33 ways in the live tree.
- **`find`/`grep` are broken shims**; use `rg`, `rg --files`, `/usr/bin/find`.
- Tool homes: the Aeneas that translates is `/home/socol/Workspace/penglai/target/aeneas`
  (sha256 `33d0ce1a…`, `505b6ca-dirty`) — it has no recoverable source on this box. The rebuilt
  Aeneas `/home/socol/oc-tmp/aeneas-pin-home` (`2a0d7827…`) regresses on `core::array::repeat`
  (`Translate.ml:390`). Charon is `747e165b…` in both. `AENEAS_HOME=/home/socol/Workspace/aeneas`
  in the environment has no executables; pass the tool home explicitly.
- Committing around foreign hunks: build the staged blob as HEAD plus your lines
  (`git hash-object -w` + `git update-index --cacheinfo`). Foreign hunks live in `proof/regen.sh`
  (+17, sysroot fix), `proof/extract/lib.rs` (+56), `proof/ApeironWord.lean`,
  `proof/corpus/Policy/Core.lean` (mid-refactor), and more.
- Song's commit hook: subject ≤72 chars with an `<area>: ` prefix, one contiguous block ≤12 lines,
  bans "not yet", ` -- ` asides, **and any date in prose** (a literal "1960" was refused; git
  carries dates).
- webfold ledger batches: ten rows per pass in file order, separate-row proof through the checker's
  own `check()` with other rows neutralised, a reserialisation guard (`json.dumps(indent=2)` on
  edited rows must be byte-identical), and the Markdown cell byte-identical to the JSON grade.

## 2 · Owner decisions in force

1. **Kalman: decided and landed** (`8b710943`) — the midpoint face only (`kalmanHalf`,
   `kalmanhalf_bound`/`kalmanhalf_sym`); a parameter gain and the innovation form are refused by
   the extraction chamber (measured); the gain-weighted corpus theorem stays open in Apeiron.
2. **PATH shims: permanent architectural fix** (`c00b0b6`).
3. **Intent width (50 concurrent agents): landed** (`607e77a`) with the lease half and four
   controls; the P1 completion check is live again.
4. **webfold callers are untrusted**: SSRF guard with server-side policy, loopback refused by
   default; the PR carries it.
5. **The hashtrinity gap is the highest-value item** (E6): measured; port contract shape A (bounded
   probe absence certificate, `Grow` = full probe cycle, summary retired, `SpillProbeBoundAbsentSound`
   replaces `SpillSummaryMissSound`); spec in `docs/epics/read-path-specialization.md`.
6. **f32 boundary (Sol)**: four admitted axioms — `attn.KvState.with_carrier`,
   `attn.KvState.attn_head_f32`, `mixer.f32_carrier.q32_to_f32`, `core.f32.F32.is_finite`; the
   corpus scope is the production constructor (`KvState::with_carrier`); `KvState::new` and
   `carrier_from_env` are recorded non-claims; `append`'s F32-storage overflow guard is not proved.
7. **NEW, pending the review verdict**: the owner's direct-port directive — read the reference
   source (hashtrinity) and generated code (LLVM/asm), reach llama.cpp/hashtrinity level directly,
   then push SOTA. Act on the verdict when it lands.

## 3 · The extraction landing unit — composition and blockers

The landing is **one atomic unit**; it is fully staged but not landed. Staged bytes at
`/home/socol/oc-tmp/f32bound/landing/<path>`; unified diff `/tmp/opencode/landing.patch` (344 lines);
`git apply --check` passes against the current dirty tree.

| # | path | change |
|---|---|---|
| 1 | `proof/regen.sh` | four `--opaque` flags (f32_core, f32_carrier, `crate::attn::_::attn_head_f32`, `crate::attn::_::with_carrier`) + `audit()` → `audit-extraction.py` |
| 2 | `proof/extract/lib.rs` | `word_kv_state_new` → `with_carrier(.., Carrier::Q32)` + coverage note |
| 3 | `proof/strip-modeled-axioms.py` | `I128.Insts.CoreConvertFromU64.from` in ALLOW |
| 4 | `proof/patches/aeneas-lean-model.patch` | the `I128` model def (plus the P0 `F32` addition when it lands) |
| 5 | `proof/boundary-axioms.txt` | new — the four items with reasons |
| 6 | `proof/audit-extraction.py` | new — sorry/admit, unlisted-axiom, absent-entry rules (8 controls) |
| 7 | `proof/README.md` | the f32 carrier boundary block |
| 8 | `proof/corpus/AxiomAudit/Core.lean` | the four names in `auditClosedAxioms` (correct as written — no module prefix) |
| 9 | `proof/ApeironWord.lean` | the regenerated artifact, sha256 `bd7c612a…` (24,668 lines) |

Add to the same landing: the SRCLIST additions — 16 missing `SOURCE_FILES` entries (children of
`mixer/`, `step/`, `chart/`, `kern/`, `attn/f32_core.rs`, `dequant_lowbit.rs`) — staged at
`/home/socol/oc-tmp/srclist-t1/regen.patch`, additive-only, with `src/engine/mod.rs` kept on a
documented allowlist reason.

**Blocker chain, in order:**

1. **Model `F32` (P0, in flight).** The staged artifact fails `lake build` with 26 errors: `F32`
   undefined (42 uses, 0 definitions in the model), `I128.Insts.CoreConvertFromU64.from` unknown in
   the built model, and 5 downstream `declaration uses sorry`. Fix: add `F32` to
   `proof/patches/aeneas-lean-model.patch` as a **definition** over a primitive carrier (no new
   axiom — the cone stays at the four boundary items), rebuild the model via
   `proof/model-patches.sh`, verify both defs in `target/aeneas-model/lean`, then build the corpus
   against the staged artifact.
2. **Corpus reconciliation.** The corpus builds **green today** (9227 jobs) against the committed
   artifact, once the unrelated `Policy/Core.lean` refactor is set aside. Against the new artifact
   it fails inside `ApeironWord.lean` (blocker 1), and once that clears, ten removed declarations
   must be reconciled in `proof/corpus/Numeric/`: **A — mechanical (3 files)**: `AttnStageWitness`,
   `AttnSoftmaxPerturb`, `AttnStage` (struct literals and `state.k/.v` reads → `carrier`/`storage`);
   **B — re-scope + re-derived spec (2)**: `AttnHeadWindowsScore`, `AttnHeadWindowsValue`;
   **C — re-derived proofs (4)**: `AttnLoopScore`, `AttnLoopOutput`, `EngineAttnRefinement`,
   `AttnHeadRefines` (subjects changed from standalone defs to `ControlFlow` loop bodies; the value
   loop gained a nesting level); **D — unprovable as stated (2)**: `EngineSamplerRefinement`,
   `KvReachable` — their five `BadGeometry` refusal theorems and `word_kv_state_new_geometry/inv`
   are statements about the axiom `with_carrier`; options are a contract on the axiom (new theorem)
   or a boundary change (not feasible while `with_carrier`'s body needs float). The f32 path is
   unproved today and the boundary is where that obligation now lives; `AxiomAudit` must record it.
3. **Tool identity.** The translating Aeneas has no recoverable source; the rebuilt one regresses.
   The landing artifact is bound to `33d0ce1a`'s translation and records the residual.

Landing verification: `AENEAS_HOME=/home/socol/Workspace/penglai/target/aeneas sh proof/regen.sh`
(fresh) then `--check` (expect "extraction is fresh"); `python3 proof/audit-extraction.py
proof/ApeironWord.lean proof/boundary-axioms.txt` (expect 0 unlisted, 4 admitted);
`python3 tools/check-claims.py`; warm `cd proof/corpus && lake build`.

## 4 · E6 — the port

Measured and decided: resident read at parity (1.0625 vs 1.0521 ns/key); the gap is the spill read
(~8.14 ns/key canonical); the flat prototype reads 1.34 (6.06×), below the resident anchor, all
reproduced; timing runs must be built WITHOUT `read-stats` (the feature costs 1.90 ns/read on the
canonical path). Contract: shape A, in the read-path epic.

Slice state (scratch `/home/socol/oc-tmp/e6w1`, uncommitted):
- Slice 1 done: `Region` is one flat `Vec<SpillSlot { tag, sigma, held, value }>` with a `live`
  count; `slots_for` keeps a vacancy (load ≤ ½ at the bound); `home(place)` keeps the retired
  summary's mixer so placements do not move; `spill_read`/`spill_locate`/`spill_locate_held` are
  bounded linear walks; adaptations in `table.rs`, `prepared.rs`, `locator.rs`, `parallel.rs`,
  `resources.rs`, `tests.rs` (+460/−833 over 8 files). Invariants are stated as comments.
- Open from slice 1: one fixture red (`tests::prepared_shard_reserves_before_hold_on_sizing_refusal`
  — rewrite against the flat capacity rule); counters and ns/key unrun; two coverage holes to
  replace (forced same-fingerprint collision with no false negative; completed-cycle absence at the
  `Table` level); dead code (`WINDOWS`, `group`, `least_set_lane`, `scan.rs`) MUST stay until the
  subtraction task lands with the extraction symbol update; `SpillProbeBoundAbsentSound` does not
  exist yet (Lean task).
- Slice 2 (in flight) finishes the fixture, adds the two fixtures, runs counters and the parity
  measurement. The committed gate of record (`examples/spill_read_probe.rs`) is in flight; use it
  once landed. Acceptance: ported canonical within ~2× of resident, ideally at parity.

## 5 · Fleet state at compaction

In flight: W2F2 (port slice 2), W4A (gate-of-record harness), MODEL-F32 (model unblock), T10
(ledger rows from `HARNESS-17`). Pending: the night-review verdict (direct-port mode + keep-alive
enforcement); the Penglai headers decision (staged `HEAD + header` blobs for the 6 tracked files,
the 8 untracked children ride with their owners).

Repo tips: apeiron `75e8686`; webfold local `6fde83a0` (remote PR `9bca1791`, remote ref
`feat/tiered-algebraic-extraction`); song `8b710943`; penglai `1128383`; intent `607e77a`.
The webfold ledger checker's first failure is `HARNESS-17 examples/mcp_probe.rs` (index 96); 31
`source_files` inventory entries are stale repo-wide and unreachable until the rows pass.

## 6 · Next actions

1. MODEL-F32 lands → corpus build of the staged artifact → reconcile A/B/C/D → land the unit plus
   the SRCLIST additions (foreign hunks preserved) → regenerate, audit, cone, warm corpus build →
   commit. This is the longest-standing blocker.
2. W2F2 lands → subtraction + extraction update → `SpillProbeBoundAbsentSound` → integrated
   benchmark; every slice gated through `spill_read_probe`.
3. Ledger from `HARNESS-17` (T10 then the next batch); push the webfold PR (merge lane,
   `cargo check` + lib) after each batch.
4. Night-review verdict → implement the direct-port directive and the keep-alive enforcement;
   record under `/home/socol/oc-tmp/escalation/`.
5. Refill at every return; floor six.
