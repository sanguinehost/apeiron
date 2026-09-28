# Continuation evidence

Status: superseded by [user-directed Kani retirement](../epics/kani-retirement.md).
Kani receipts and requirements below describe the earlier work only; do not run or retry them.
Retirement does not turn timeouts into passes or close missing Lean refinements.

Base `262bb0a94eff2cc7d5655122a954f44d74de96b6`; initially clean.
`git diff dbc1bc8 HEAD --stat` contains only `docs/archive/handoff-openai.md` (254 inserted lines).
Raw command logs: `target/handoff-run/` in the supplied checkout. Results below are fresh
command observations, not acceptance of the handoff's entire wave.

| Initial gate | Exit | Scope |
|---|---:|---|
| `proof/regen.sh --check` | 1 | Generated code stale: derived shard constants and executable consumers differ, not only source annotations |
| `cd proof && lake build` | 0 | 1708 jobs, committed generated artifact only |
| `cd proof/corpus && lake build` | 0 | 3482 jobs, existing conditional headlines and audits on committed generated artifact |
| `nice -n 10 timeout 1800s cargo kani -j 16 --output-format terse -Z unstable-options --harness-timeout 300s` | 1 | 58/67 verified; the same nine CBMC timeouts, no reported counterexample |
| `python3 tools/check-assurance.py --check` | 1 | Nine breaches; full size/perimeter acceptance remains open |
| `python3 tools/check-claims.py --check` | 1 | 14 structural claim-resolution findings in governing documents; not a semantic refutation of every reported row |

Lean gates use the lock, 8G/no-swap scope, single-thread environment and 900-second timeout
in the [plan](openai-continuation-plan.md). The freshness failure prevents attributing the
two successful builds to current Rust. No new universal theorem is asserted.

## Identity

| Input | Observed identity |
|---|---|
| Clean resolved canonical package | `3d66ea3aa2fe1de5997b6b757247454a2f771358` |
| Mathlib manifest revision | `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f` |
| Initial generated Lean SHA-256 | `e1385a4d2927b7b47c4454e16a30ce365d505c97606ecf3429622dac26cd5be9` |
| Model patch SHA-256 | `210e56e1ed54bfd4b91e12a102e7dceb17c429b7f14903a176c27b1c0b23b191` |
| Charon | `0.1.254 (b104e24fea7d721b71e6c39fd70f26ff20bc0980)` |
| Charon binary SHA-256 | `747e165bc62bb963f8d741e8c193c721a4b94555bcb1f9f86c79090fd7a59d0e` |
| Charon driver SHA-256 | `207b223af35cf9a05227b329819b73532e94f655ccdefd64e77b755a15fdf297` |
| Aeneas | `505b6ca-dirty`; binary `33d0ce1a1cc2f1d3935c0cd2ab34f283797da61c531ed6d4e09d5c97b0d0228a` |
| Extraction Rust | `nightly-2026-08-18` |
| Ordinary rustc | `1.97.0-nightly (17584a181 2026-04-13)` |
| Kani | `0.67.0` |
| Host | Intel i9-14900KF; 32 logical CPUs |

The canonical sibling has unrelated dirty work; the resolved bridge dependency is clean at the
recorded pin. No sibling file is changed or used as a substitute for that package.
The extraction log records all 18 extraction-source digests. The installed Aeneas path resolves
to `/home/socol/Workspace/penglai/target/aeneas`; no installed-tool mutation is authorized.

## Decisions and remaining obligations

- Sol extraction verdict: regenerate the sole generated file from unchanged source and all 43
  roots, then replay freshness and both builds. Derived geometry constants now generate checked
  `Result` computations; do not rewrite production merely to imitate stale Lean.
- Sol Kani verdict: a real `spill_locate` fixture may check late `Found`, first-fit `Vacant` and
  `Grow`, with a truncated-sweep counterexample. It does not discharge any whole-Table harness,
  public-constructor reachability, `Regen`, universal `SearchDecisionCorrect` or canonical `get`.
- Sol allocator verdict: the current `Result` effect offers no allocator nondeterminism.
  `try_reserve`'s deterministic success below the size ceiling cannot model real may-failure.
  An oracle-indexed abstract contract is feasible but does not bind allocation execution;
  actual tool-model integration exceeds the handoff's two-file D1 and needs decomposition.
- Universal read refinement remains missing. `Scan` lacks summary/fingerprint exclusion;
  `RepR` lacks those construction/preservation facts; generated reads and the insertion
  classifier need distinct source-linked results. Importing `Representation` into `Search`
  creates a cycle through `Refinement`; composition needs an acyclic task boundary.
- Size gate: `src/proofs.rs` 1508 lines (reported twice by the checker),
  `examples/reg1/acceptance.rs` 1799, `Slang/Generated/Table.lean` 14414,
  `src/table.rs` 1678, `tools/slang/export.py` 2299, `tools/slang/spec_check.py` 1562;
  `Slang/Source` has 14 immediate source files and `tests` has 15. No cap is waived.

Whole-state insertion composition, allocator simulation, performance acceptance and the full
wave remain open. Existing trust exclusions and both axiom cones are unchanged.

## Fresh regeneration regression

The isolated regeneration produces SHA-256
`49079c1f1f3ce0bde20403a9a7bd6b1fe792bc46f85666b37be17bf1b9a2f208`.
`proof/regen.sh` and `proof/regen.sh --check` exit 0; the proof package builds 1708 jobs.
The fresh bridge exits 1 at `Growable/Geometry.lean:59`,
`extracted_shard_of_direct_word`: its old proof leaves the new checked `table.SHARD_MASK`
computation unreduced. The theorem statement and source geometry are unchanged.
The lead assigns that one proof-body repair to the same extraction owner before integration.

## Accepted insertion-locator fixture

Changed `src/table/probe.rs` only inside `cfg(kani)` and appended
[`../evidence/kani-table-harness-audit.md`](../evidence/kani-table-harness-audit.md). The focused command
`cargo kani -Z unstable-options --harness spill_locate_classifies_a_fixed_initialized_region
--harness-timeout 120s` exits 0: 803 checks, 23 unreachable, 7.423595 seconds.
The disposable real-source sweep truncation exits 1 at the named late-match assertion only;
restored source exits 0: 803 checks, 7.4622197 seconds. Formatting exits 0.
Logs remain at `target/handoff-w2-t2/`; source SHA-256 is
`9cbf4ead2188957f2100833a8e84a4ba3505b25cd66b18a57834abd1a0a22a0f`.

Grade: `tested`/bounded helper execution; no public-Table reachability or universal classifier
claim. The nine whole-Table harnesses remain enabled. The lead reviewed and copied both owned
files, preserved the logs, and removed the task worktree and branch without a commit or push.

## Security dispatch decision

Sol's read-only review in `target/handoff-run/security-verdict.md` checks the clean resolved
canonical package. Algebraic face/channel/idempotent declarations exist with their original
root, inverse and half premises; they supply no secrecy or federation statement. No SEC-R
implementation is dispatched: the staged plan waits for storage closure and does not yet name
ratified output declarations. Proposed transport decomposition is planning only, not a new
canonical definition or security result. The separate dirty sibling remains untouched.

## Accepted extraction repair

Changed `proof/ApeironWord.lean` by regeneration and only the proof body of
`Growable.Geometry.extracted_shard_of_direct_word`. Statement, premises, production geometry,
43 extraction roots, tool pins and axiom cones are unchanged. The proof now discharges the
checked constant shift/subtraction before proving the same low-byte result.

| Gate/control | Exit | Evidence |
|---|---:|---|
| `proof/regen.sh` | 0 | 1708 jobs; modeled-axiom audit passes |
| `proof/regen.sh --check` | 0 | 1708 jobs, fresh generated file |
| Disposable generated shard-count drift | 1 | Freshness rejects; restored check exits 0 |
| Bounded `lake build Growable.Geometry` | 0 | 3342 jobs |
| Bounded full bridge `lake build` | 0 | 3482 jobs |
| Disposable generated `SHARD_BITS` 8→7 | 1 | Geometry headline requires mask 255 but generated mask is 127; no type/resource failure |
| Restored full bridge `lake build` | 0 | 3482 jobs |

Grade: existing geometry headline `proved` at unchanged scope; extraction freshness alone is
not universal read refinement. Generated SHA-256:
`49079c1f1f3ce0bde20403a9a7bd6b1fe792bc46f85666b37be17bf1b9a2f208`; Geometry SHA-256:
`627ab73df9cd88120bf886f07b5f6cf11a71a84d1fb42702491e453bb3ea6cc6`.
Logs/Arc: `target/handoff-run/extraction/`. The lead reviewed the generated executable diff,
integrated both files, preserved receipts, and removed the task worktree/branch.

## Rejected summary proof attempt

`APEIRON-HANDOFF-W3-T1` stops after three materially different failed proof approaches.
Bounded `lake build Growable.Scan AxiomAudit` exits 1. Neither owned file is integrated.
The attempted patch is retained only under `target/handoff-run/summary/rejected.patch`.
Command/diagnostic summaries are alongside it; the rejected worktree and branch are removed.
Empty-summary facts alone do not close the task. The OR-to-AND control and restored green
gate are unrun, so no summary recordedness claim is registered as proved.

Smallest residuals in `Growable/Scan.lean`:

- Prove the one-bit mask `1 << (index % 64)` is nonzero and survives OR/AND, using bounded
  bit-vector lemmas rather than looping simplification.
- Normalize generated `Vec.len` lets and invert the successful mutable-index/update result
  to prove `summary_set_implies_maybe` without an added classifier or index premise.
- Check one nonempty same-input write/query witness; the index computation is still unproved.

Next bounded proof task: the scalar recorded-bit lemma in `Growable/Scan.lean` with its
`AxiomAudit.lean` entry, one focused build and a semantic mask-clearing control. This is a
prerequisite, not permission to advance representation or public-read composition.

## Accepted attribution corrections

Changed `proof/README.md`, `proof/corpus/Growable/Search.lean` and `proof/extract/lib.rs`.
The source edits are comments only, with identical comment-stripped content and line counts.
The README lists 43 actual selectors, all 18 current extraction-source digests, the generated
digest, historical fixed-shard scope and exact conditional rebuild fields. The source comments
now distinguish extracted probe bodies from unproved `SearchDecisionCorrect`.

`git diff --check` exits 0; the lead verifies all 18 listed hashes against the integrated files.
Grade: `anchor` documentation; no new proof claim. Semantic control is inapplicable to comment-only
changes; the final integration freshness/build checks still apply. Receipt:
`target/handoff-run/proof-docs/`. The task worktree/branch are removed.

## Final integration verification

Bound inputs are recorded in `target/handoff-run/final-source-digests.json`; final command
logs use the `final-` prefix in the same directory. No accepted source changes follow this freeze.

| Command | Exit | Observed result |
|---|---:|---|
| `cargo test --workspace --all-features --locked` | 0 | 258 passed, 1 ignored; 541.523 seconds including the 100000-key growth/readback test |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 | No rejected warnings |
| `cargo fmt --all -- --check` | 0 | Formatting passes |
| Bounded `cd proof && lake build` | 0 | 1708 jobs |
| Bounded `cd proof/corpus && lake build` | 0 | 3482 jobs, unchanged conditional scopes |
| Bounded `proof/regen.sh --check` | 0 | Fresh at generated SHA-256 `49079c1f…`; 1708 jobs |
| `python3 tools/check-assurance.py --check` | 1 | Nine existing breaches; local trusted checker inventory 4776 lines, generated extraction 10974 lines |
| `python3 tools/check-claims.py --check` | 1 | 14 unresolved claim references; includes missing `generated_tag_nonzero` and unresolved relative paths |

The Cargo log emits a duplicate-key diagnostic for excluded `opsfold/Cargo.toml` in the
pinned fold dependency while continuing the selected build. The dependency has no tracked
changes at `4cc14471ae24f3f204fb0b0e91736ef75c909358`; the recorded Cargo exits above are 0.

## User-directed stop

The final Kani rerun was cancelled at the user's request; no final 68-harness verdict is claimed.
The owned driver and solver processes were terminated and their absence checked. Cancellation
is recorded in `target/handoff-run/final-kani-cancelled.json`. Kani retirement supersedes the
old preservation requirement. The queued performance measurement was not run; tool retirement
is the active follow-up. Pre-retirement Rust and Lean gate results above remain identity-bound.
