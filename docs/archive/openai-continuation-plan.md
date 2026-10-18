# OpenAI continuation

Status: superseded by [user-directed Kani retirement](../epics/kani-retirement.md).
Kani receipts and requirements below describe the earlier work only; do not run or retry them.
Retirement does not turn timeouts into passes or close missing Lean refinements.

Owner: lead Astra. Base: `262bb0a94eff2cc7d5655122a954f44d74de96b6`.
The supplied-checkout exception covers this plan, `docs/archive/openai-continuation-evidence.md`
and corrections to `docs/archive/handoff-openai.md` only. No commit or push is authorized.
Direction, kernel-boundary and execution plan retain authority and all acceptance gates.

## Intent and contract

Re-establish the handoff's source/tool identity before extending the universal read proof.
Preserve all nine enabled whole-Table Kani obligations; helper fixtures cannot replace them.
Preserve every extraction root, real production body, theorem premise and both axiom cones.
Do not use `SearchDecisionCorrect` as the assumed conclusion of a classifier proof.
Allocator conformance and whole-state insertion composition remain open.

Rejected: trusting old green receipts, accepting stale generated code, stubbing the probe,
retiring timed-out harnesses, concurrent cache ownership, or measuring during verification.

```text
SCOPE: docs/archive/openai-continuation-plan.md, docs/archive/openai-continuation-evidence.md,
       docs/archive/handoff-openai.md; ownership, fresh results, corrected dispatch.
REGISTER: planning/evidence; no runtime or mathematical acceptance from documentation.
CORPUS: Growable/Scan.lean scan_exact; Search.lean SearchDecisionCorrect;
        Representation.lean RepR read; universal read links absent.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Growable/
GATE: git diff --check; repository root; pre-edit exit 0, no documentary red invented.
GRADE: anchor decisions; command results retain their actual scope.
```

## Ownership and dependencies

Task prefix `APEIRON-HANDOFF-`. Read-only spikes own no tracked source files.
Their outputs live in `target/handoff-run/`; the lead registers decisions in the receipt.
Each spike has a 20-minute bound. A missing gate or semantic control blocks implementation.

| Task | Owner | Files/output | Gate/control | Dependency |
|---|---|---|---|---|
| SPIKE-extraction | Sol extraction reviewer | `target/handoff-run/extraction-verdict.md` | Diagnose actual freshness exit 1; propose bounded repair and semantic control | Initial freshness run |
| SPIKE-kani | Sol Kani reviewer | `target/handoff-run/kani-verdict.md` | Ratify concrete real-helper domain and rejecting source mutation; keep whole-Table blocked | Existing harness/source review; fresh capped run |
| SPIKE-reserve | Sol model reviewer | `target/handoff-run/reserve-verdict.md` | Pin may-fail WP semantics and inhabited success/refusal; no deterministic allocator claim | Actual backend/model review |
| W1-T1 | Lead, verification lane | Command logs in `target/handoff-run/` | Freshness, both package builds, capped Kani, Rust gates; record every exit | Frozen baseline source |
| W1-T2 | Lead, docs exception | Three documents in scope above | `git diff --check`, review against logs and unchanged obligations | Specialist verdicts and fresh results |
| B1 | Sol summary, W3-T1 below | `Growable/Scan.lean`, `AxiomAudit.lean` | Bounded focused build plus source-linked semantic and cone-rejection controls | Fresh extraction; exact theorem task |
| B2 | Sol, held | `Growable/Representation.lean`, `AxiomAudit.lean` | Bounded build and representation mutation | B1 accepted |
| B3 | Sol, held | Read composition module and audit, exact paths pending | Generated read semantic control; separate insertion classifier obligation | B1/B2 and acyclic module boundary |
| C1 | Held | Relocation/Rebuild/Resources | Existing premises retained | Kani boundary settled |
| D1 | Held | Model patch/stripper; exact scope pending verdict | May-fail contract and real refusal control | Sol model verdict; dedicated model/build lane |
| E1 | Lead measurement | Existing strict REG-1 executable; raw logs only | Full `--accept --controls`; no reduced matrix or acceptance-bar change | Exclusive quiet host; no source/harness edits |
| G1 | Sol read-only attribution review | `target/handoff-run/security-verdict.md` | Read staged security plan and exact pinned targets; no security implementation | Canonical identity; lead decomposition |
| F1 | Held | One existing epic's exact next task | Its named gate and control | Lifecycle/source identity and dependency check |
| H1 | Decomposed | Evidence correction now; source-size repair separately | Documentation check now; source split needs its own Rust/extraction barrier | Stable source/extraction identity |

## Authorized bounded execution

The lead authorizes the following task-only branches/worktrees. Changes remain uncommitted;
after review, copy only accepted owned files to the integration checkout, preserve raw logs,
then remove the task worktree and branch. No unrelated branch is removed.

| Task / owner | Worktree / branch | Owned files | Gate and control |
|---|---|---|---|
| `APEIRON-HANDOFF-W2-T1` / Sol extraction | `worktree-agent-APEIRON-HANDOFF-W2-T1`, `codex/handoff-extraction` | `proof/ApeironWord.lean`, `proof/corpus/Growable/Geometry.lean` | Completion: bounded full bridge `lake build`; prerequisites regeneration/freshness. Generated drift must reject freshness; generated shard-bits 8→7 must reject the unchanged direct-shard theorem; restored pass |
| `APEIRON-HANDOFF-W2-T2` / Sol Kani | `worktree-agent-APEIRON-HANDOFF-W2-T2`, `codex/handoff-kani-fixtures` | `src/map/table/probe.rs`, `docs/evidence/kani-table-harness-audit.md` | `cargo kani -Z unstable-options --harness spill_locate_classifies_a_fixed_initialized_region --harness-timeout 120s`; truncate real locator sweep in disposable copy, require late-hit assertion failure, restored pass |
| `APEIRON-HANDOFF-W3-T1` / Sol summary | `worktree-agent-APEIRON-HANDOFF-W3-T1`, `codex/handoff-summary-link` | `proof/corpus/Growable/Scan.lean`, `proof/corpus/AxiomAudit.lean` | Bounded `lake build Growable.Scan AxiomAudit`; generated summary-set OR-to-AND mutation must refute recordedness, restored pass |

`APEIRON-HANDOFF-W3-T2` / Sol attribution: lead authorizes task-only branch
`codex/handoff-proof-docs` and `worktree-agent-APEIRON-HANDOFF-W3-T2`. Own exactly
`proof/README.md`, `proof/corpus/Growable/Search.lean`, `proof/extract/lib.rs`. Correct stale
extraction/root/digest statements and comments only; no declarations, premises, roots or code
change. Gate: `git diff --check` plus inspected comment-only source diff and actual selector/root
lookup; grade `anchor` documentation. Semantic mutation control is inapplicable to comment-only
changes; extraction/bridge controls remain the implementation barrier. Prepare alongside summary;
no heavy command until the lead grants a lane. Parent checks final extraction freshness/builds.

All bases are the recorded HEAD. Heavy execution serializes extraction → Kani fixture →
geometry regression repair → summary.
Summary proof text may be prepared against the candidate generated artifact; execution and
acceptance require the accepted fresh generated artifact first. A build regression
opens a separate bounded proof-repair task before acceptance. Each worker records its Arc before
editing and receives an explicit lane grant. Kani changes are confined to `cfg(kani)` harness code.

Summary scope: `summary_index_zero`, `summary_maybe_empty`, `summary_set_empty`,
`summary_set_implies_maybe`. The substantive theorem must derive the generated query's `true`
result from a successful generated summary write, without assuming a desired classifier result
or adding an unratified index premise. Require a nonempty success witness and semantic rejection.
This supplies one local recordedness link; it is not whole-state preservation or canonical `get`.
The Rust bridge retains its closed cone; the handoff's B1 empty-cone wording does not transfer
the Slang-specific restriction to this bridge.

`Representation` imports `Refinement`, which imports `Search`. A public-read theorem using
`RepR` cannot be added to `Search` by importing `Representation`; that creates a cycle.
Choose an acyclic composition module or lower-level interface before dispatching B3.
No file move or altered theorem premise is authorized by this observation.

All Lean commands use `flock /tmp/opencode/apeiron-lane.lock`,
`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`,
`LEAN_NUM_THREADS=1` and `timeout 900s`. No unbounded reductions.
The lead leases `target/`, generated extraction and both Lake caches for baseline commands.
Specialists perform read-only work until a lane is explicitly released.
All mutations use disposable copies; no live-source mutation control.

## Barrier

Keep the full AGENTS Rust/Kani, regeneration, two-package build, audit, control, digest,
size/perimeter and REG-1 requirements. A green old-generated bridge is not current-Rust proof.
Any unresolved child or shared gate keeps the wave open. No benchmark result discharges the
universal read links, allocator simulation, constructor/preservation or insertion composition.
Trust exclusions remain allocator/executor realization, callbacks outside their laws,
rustc/LLVM, ABI, GPU/SPIR-V and the unsafe container.

## Closed local task

`APEIRON-HANDOFF-W2-T2`: focused Kani baseline/red/restored exits `0/1/0`; the red is the
late-match assertion, not a timeout. Both owned files are integrated uncommitted. Raw logs are
preserved in `target/handoff-w2-t2/`; its task worktree and branch are removed. This closes only
the helper fixture task, not A1 or the shared Kani barrier.

`APEIRON-HANDOFF-W2-T1`: regeneration/freshness and full bridge exit 0. Both generated drift
and semantic shard-bit controls reject; restored gates pass. Generated and Geometry files are
integrated uncommitted; logs are preserved in `target/handoff-run/extraction/`. Its worktree and
branch are removed. The summary task receives the accepted generated/Geometry digests as baseline.

`APEIRON-HANDOFF-W3-T1`: rejected after three failed approaches; focused build exits 1.
No Scan/AxiomAudit diff is integrated. The rejected patch and command diagnostic summaries are
preserved in `target/handoff-run/summary/`; the task worktree and branch are removed.
Semantic control/restored gate are unrun. Exact residuals
and the next scalar-lemma task are recorded in the continuation evidence; B2/B3 remain held.

`APEIRON-HANDOFF-W3-T2`: three attribution files integrated; comment-stripped source and line
counts unchanged, 43 selectors and 18 source hashes verified, diff-check exits 0. Worktree and
branch removed. No task-created branch/worktree remains; unrelated `one-allocation` is preserved.

## Final verification lanes

After all task source edits stop, the lead owns every verification lane. The two Lake builds
serialize under the shared Lean lock; they may overlap the read-only running Rust test in its
distinct `target/` lane. Rust test/clippy/fmt serialize. Regeneration waits for Rust completion.
Full capped Kani follows regeneration; measurement waits for all verification and a quiet host.
No worker edits or measurement overlaps these final verification commands.
