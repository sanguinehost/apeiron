# Retire Kani from Apeiron

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

Merged epic record: `intent/kani-retirement.md`, `spec/kani-retirement.md`, `plan/kani-retirement.md` (one file per epic; see `docs/README.md`).

## Retire Kani from Apeiron

The user directs retirement of Kani after recurring multi-minute and timed-out runs. Kani is no
longer a development, acceptance, closure or dispatch requirement. This supersedes earlier
instructions to preserve/retry its harnesses. Do not run Kani or add new Kani work.

Keep production behavior, canonical definitions/pins, source extraction, Lean contracts, ordinary
Rust tests and semantic mutation controls. Retain useful concrete probe regressions as ordinary
unit tests; do not describe them as symbolic exhaustive verification. Historical Kani receipts
remain historical evidence, not current acceptance. No machine-global tool uninstall is needed.

Success: no Kani source/configuration path or required gate remains; coverage validation no longer
requires Kani, reports the remaining evidence honestly, and rejects forged evidence. The real
universal read, representation and allocator obligations remain open. No performance claim follows.

## Kani retirement contract

- Ordinary production geometry stays `SHARD_BITS=8`, `SHARDS=256`, `SHARD_MASK=255`.
- Remove Kani-only source/harness wiring and the alternate four-shard configuration.
- Preserve actual helper regression cases: late match past a vacancy, unequal full keys at equal
  fingerprints, summary miss, unequal Sigma, first vacancy and full-region growth classification.
- Ordinary tests use concrete inputs and report grade `tested`; they do not inherit the old
  symbolic population or prove canonical `get`.
- Coverage reports keep Aeneas, refinement and executed tests distinct. Remove Kani as a required
  dimension without silently converting historical Kani receipts into another grade. Current
  extraction/Lean claims retain their exact domains, open premises and source identity.
- The required barrier is Rust test/clippy/fmt, source extraction freshness, both bounded Lean
  builds, applicable audits and named semantic controls. No Kani run is a gate or retry action.
- Historical records may mention Kani; current commands and policy must make retirement explicit.

## Kani retirement execution

Lead: Astra. Base `262bb0a94eff2cc7d5655122a954f44d74de96b6`, with the accepted uncommitted continuation changes preserved.
User authorization supersedes the prior no-retirement instruction. No commit or push.

```text
SCOPE: docs/epics/kani-retirement.md, docs/epics/kani-retirement.md,
       docs/epics/kani-retirement.md; ratify user-directed tool retirement and bounded tasks.
REGISTER: planning/evidence; no proof premise or canonical definition change.
CORPUS: Growable/Search.lean SearchDecisionCorrect remains open; no theorem closes by retirement.
REFERENCE: /home/socol/Workspace/apeiron/AGENTS.md and actual src/tools/coverage registry read.
GATE: git diff --check; pre-edit exit0. Current Kani run cancelled, not graded successful.
GRADE: anchor policy; retained Rust regressions tested only.
```

The lead authorizes task-only worktrees/branches below from the recorded base, with accepted dirty
inputs copied only for each task's required baseline. Record digests before editing. One owner per
file. No Kani command is authorized. Parent integrates reviewed owned files and removes task trees.

| Task | Owner | Owned files | Gate/control |
|---|---|---|---|
| APEIRON-KRET-W1-T1 | Sol source | src/lib.rs, src/proofs.rs, src/table/probe.rs | Focused ordinary probe unit tests; real sweep truncation rejects late-hit assertion in disposable copy, restore passes |
| APEIRON-KRET-W1-T2 | Sol evidence | examples/reg1/coverage.rs, examples/reg1/coverage.json, tools/check-assurance.py | Existing coverage gate/controls after Kani dimension retirement; preserve non-Kani rejecting controls and exact source/refinement grades |
| APEIRON-KRET-W2-T1 | Sol extraction | src/table/partition.rs, src/coord/locator.rs, proof/ApeironWord.lean | Remove alternate cfg geometry/comment, regenerate/check under bounded lane; unchanged production geometry and full bridge |
| APEIRON-KRET-W0-T1 | Lead documentation exception | AGENTS.md, docs/direction.md, docs/kernel-boundary.md | Diff check; explicit retirement supersedes older plan references, unchanged mathematical obligations |
| APEIRON-KRET-W2-T2 | Lead documentation exception | docs/plan.md, docs/archive/handoff-openai.md, README.md | Diff check; current gate commands contain no Kani invocation |

Source worktree `worktree-agent-APEIRON-KRET-W1-T1`, branch `codex/kani-source-retirement`;
evidence worktree `worktree-agent-APEIRON-KRET-W1-T2`, branch `codex/kani-evidence-retirement`;
extraction worktree `worktree-agent-APEIRON-KRET-W2-T1`, branch `codex/kani-geometry-retirement`.

Source and evidence prepare in parallel; compilation lanes serialize and source deletion waits
for evidence include-path repair at integration. Geometry extraction starts after the library/probe source slice is integrated; coverage
code/JSON is not an extraction input and may prepare concurrently. Coverage validation follows
the final generated identity. All Lean gates retain flock, 8G/no-swap, threads1 and timeout900s. Mutation
copies live on disk, not bulk /tmp caches. Stop after3 materially different failed approaches.

Final checks: no Kani source macros/configuration or executable gate, focused tests/control,
coverage/control, Rust clippy/fmt and appropriate full test, freshness/two builds, source digests.
Historical Kani receipts remain with retirement status; missing universal proof is still missing.
Performance measurement from the prior continuation waits; this user-directed retirement is the
active task. Existing size/claim-checker failures remain separately recorded and cannot be waived.

Documentation follow-through: the lead owns a separate three-file trivial command correction
in `docs/epics/read-path-specialization.md`, `docs/epics/parallel-stable-scatter.md` and
`docs/epics/parallel-stable-scatter.md`. Remove the runnable Kani chains and stale checklist/barrier
requirement; `git diff --check` and fenced-command inspection close this documentation slice.
Historical evidence commands are retained only as receipts, not dispatch instructions.

The lead marks the three previous continuation/Kani receipt documents superseded: no old
command or no-retirement instruction overrides the user's direction or current AGENTS.md.

Two text-only follow-through tasks owned by the source Sol worker: `APEIRON-KRET-W1-T3` owns
`examples/reg1/{correctness,workload,history}.rs` in `worktree-agent-APEIRON-KRET-W1-T3` /
`codex/kani-cli-text`; `APEIRON-KRET-W1-T4` owns `examples/reg1/main.rs` and
`examples/reg1/matrix.json` in `worktree-agent-APEIRON-KRET-W1-T4` / `codex/kani-cli-notes`.
Remove stale Kani-required CLI/help descriptions; no executable behavior or matrix schema change.
Each closes on diff-check and inspected text-only diff; integration gets all-target compilation.

Final evidence slice (lead documentation exception): `proof/README.md`,
`docs/evidence/kani-retirement.md`, and this plan. Register: source identity and tested/proved
receipts, excluding universal closure. Reference: accepted source and task logs under
`target/kret-w1-t1/` and `target/kret-w2-t1/`. Gate: diff-check plus inspected command receipts;
no new proof grade. Update source hashes only from final files.

Integration decisions: source removal, extraction and coverage tasks accepted on inspected owned
diffs and gates. Task worktrees/branches are removed; only the supplied checkout and unrelated
`one-allocation` branch remain. No commit/push. The coverage formatter follow-through owns only
`examples/reg1/coverage.rs` in the supplied checkout. Its gate is `cargo fmt --all -- --check`.

Older plan clarification uses two sequential documentation-only supplied-checkout slices:
`docs/plan/{programme-roadmap,algebra-native-addressing}.md` plus
`docs/epics/algebra-native-addressing.md`, then `docs/{plan,spec}/storage-refinement.md`.
Owner: Sol source. Add explicit retirement precedence and remove runnable current Kani commands;
historical receipts remain. Gate: diff-check and command inspection. Arc: `target/kret-doc-review.md`.

Retirement evidence is recorded in `docs/evidence/kani-retirement.md`. Next bounded mathematical
work is the missing Scan summary-bit link and its concrete witness/control; universal read and
SearchDecisionCorrect are still open. No Kani retry is a next task.
