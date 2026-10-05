# Read-proof execution evidence

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

Status: contracts, shift prerequisite, universal index bound, mask laws and generated summary
write/read composition are accepted. Rust, freshness and both integrated Lean builds pass;
seven existing size breaches and public-read, representation and classifier obligations remain open.
Base HEAD `262bb0a94eff2cc7d5655122a954f44d74de96b6` plus accepted dirty changes,
manifest `target/readlink/baseline.json`. Plan: [owned tasks and bounds](../epics/read-proof.md).

| Task | Owner | Status |
|---|---|---|
| Initial summary-index attempt | Sol scalar | Rejected after three approaches; retained below as historical evidence |
| Shift prerequisite and resumed index bound | Astra with Sol review | Accepted, audited and mutation-controlled |
| Summary mask laws | Astra with Sol review | Accepted, audited and scalar-control checked |
| Generated summary write/read | Astra with Sol review | Accepted, nonempty witness and generated control checked |
| Read composition contract | Sol read | Accepted planning artifact; diff-check exit0; task worktree removed |
| Allocator may-fail contract | Sol allocator | Accepted model/design; diff-check exit0; task worktree removed |

Kani is retired. Existing size and claim-checker failures remain open. Universal read,
SearchDecisionCorrect and allocator conformance are not asserted by dispatch or a document gate.

## Accepted contracts

- `docs/epics/read-proof.md`: proposed/uncompiled Stored/Regen signatures, precise
  missing-producer inventory, paired carrier witnesses, semantic control cases and acyclic downstream
  Read placement. Packaging fields remain unimplemented; no read theorem is established.
- `docs/epics/allocator-model.md`: same-input success/refusal relation requiring actual growth and
  valid non-zero-sized layout, no-shrink capacity condition, and separation from
  `ResourceAdapterSimulates`. The active model already has capacity state; its missing effect choice
  and capacity/layout invariants remain open. Existing oversized refusal tests do not distinguish
  allocator exhaustion from Vec capacity/layout rejection.

Owned-file review receipts: `target/readlink/w1-t2/review.json`,
`target/readlink/w1-t3/review.json`. No semantic gate applies to these prose-only artifacts.

Read-contract review supplement (lead supplied-checkout documentation exception, one owned file
`docs/epics/read-proof.md`): resolve the previously candidate vector normalization and
length lemmas against active `target/aeneas-model/lean/Aeneas/Std/Vec.lean` and pinned Lean core.
Gate: diff-check and exact declaration inspection. Grade: reference inventory only; no proof run.

## Scalar rejection and stop boundary

The requested `lake build Growable.Scan AxiomAudit` reports a failure in Scan, then the worker
terminates the aggregate; its final shell exit is not captured. The full gate is not accepted.
The recorded focused `lake build Growable.Scan` diagnostics (`run-03`, `run-08`) exit1. Structural unfolding, broad scalar simplification and
small AND/cast helpers with targeted normalization do not close the generated shift/branch goal.
The broad simplification reaches the normal recursion guard; no bound is raised. The task stops
at three materially different failed approaches within the 30-minute cap, not at a timeout.

The immediate remaining expression is `core.num.Usize.MAX >>> 6#i32`, a guarded Result whose
`6 < System.Platform.numBits` condition needs a small checked normalization using
`System.Platform.numBits_eq`. Fast-path mask arithmetic and fallback trailing-zero divisibility
remain after that. This is an incomplete proof, not a counterexample to the Rust property.

No positive gate passed, so the generated mutation control and restored-proof gate are unrun.
Temporary proof placeholders are removed. Parent byte comparison confirms both owned files equal
the accepted root: Scan SHA256 `3d6755d9747875157bf303524a9a99cb110f23ce5ee90f680bc47e76d1d2e5d5`,
AxiomAudit SHA256 `21188d9d56262b69d7cf35bb7bfc65174c3a8b1ec8b315c8fa87e2813f8ddedd`.
Raw diagnostics and verdict are preserved under `target/readlink/w1-t1/`.

The dependent mask and summary write/read tasks stay held. The shared implementation barrier is
not rerun: no Rust, generated or Lean source change is integrated, and the proof wave did not close.
Existing source/tool receipts retain their original scope; no new full-suite result is claimed.
Next bounded proof prerequisite: exact success/value normalization of the machine-width shift,
without changing the generated operation or assuming the desired summary index bound.

All task worktrees and branches are removed after receipt preservation; the unrelated
`one-allocation` branch is retained. Final documentation diff-check exits0; local links resolve.
No commit/push. Non-document baseline digests are unchanged. Final document digests are recorded
in `target/readlink/final-doc-digests.json`.

## Accepted Astra shift prerequisite

User-directed Astra escalation `APEIRON-READLINK-W2-T0` resolves the immediate normalization
obstruction without retrying the whole index theorem. Integrated files:
`proof/corpus/Growable/Scan.lean` and `proof/corpus/AxiomAudit.lean`.

`Apeiron.GrowableScan.summary_max_shift_six` proves, without caller premises, an actual `Usize`
witness for `core.num.Usize.MAX >>> 6#i32 = ok bound` and `bound.val = Usize.max / 64`.
It covers both admitted platform widths using `System.Platform.numBits_eq`.

- Bounded `lake build Growable.Scan AxiomAudit`: exit0, 3368 jobs; restored gate exit0.
- Audit: `{propext, Classical.choice, Quot.sound}`.
- Exact copied theorem statement mutation (shift6 to shift5, expected /64 retained): baseline0,
  mutant1, restored0. No live source mutation.
- Independent audited semantic control: exit0; shift5 succeeds with /32, differs from /64 on both
  widths, and refutes the mutated existential. The red is not merely a syntax/import failure.
- Sol final proof-sufficiency review: accept.

Receipts and candidate diff: `target/readlink/astra-shift/`. Scan SHA256
`5f18b3951ff7e6119bf2e90bd593bedf35ff0facd1886c51a4c4d0bd0b2d245b`;
AxiomAudit SHA256 `4277292bb1debce585d9448fd8fde10b28efb3815727a8d0f36f6c20d0f1aea4`.
The task worktree and branch are removed after receipt preservation. No commit/push.

Residual: `summary_index_word_lt_len` still needs the fast-path mask/cast/AND quotient bound and
fallback trailing-zero divisibility. The broader summary write/read, public read and insertion
classifier obligations remain open. This is a proved local scalar prerequisite, not whole-table
refinement or full-epic acceptance. Rust/generated source and installed model are unchanged.

Integration check: bounded full `lake build` in the supplied `proof/corpus/` exits0, 3482 jobs.
Log: `target/readlink/astra-shift/integration-bridge.log`. Final diff-check exits0.

## Accepted universal index bound

`APEIRON-READLINK-W3-T0` integrates `summary_index_word_lt_len` and its audit from the
Astra task after independent Sol acceptance. The theorem covers every successful generated index
on both admitted platform widths, with no extra caller premises.

- Bounded `lake build Growable.Scan AxiomAudit`: exit0, 3368 jobs.
- Concrete positive generated witness: length1/place256 returns index30; exit0.
- Disposable real generated fast mask `power - 1` to `power`: full gate exit1 after the
  mutated generated module compiles. Kernel-checked mutant witness returns index64 and proves
  the quotient bound false; clean witness exit0. Restored focused/audit gate exit0, 3368 jobs.
- Headline and witness axiom cones: `{propext, Classical.choice, Quot.sound}`.

Receipts: `target/readlink/index-resume/`, including `sol-review.md`, exact exits, generated
mutation, witness sources and digest manifests. Scan SHA256
`07a28cde5ec0559c52c67127e77f252a657307b755511870b88a037d302acb14`; audit SHA256
`353aca532ab599035b126d1a2c73fe3442ec825f0c3a682d46e290434c11e766`.
The generated source retains SHA256 `92c771510c259bc56c9e437e7d3f3108fa9e2796e6ba41ca2cd826f4127ff74e`.
This closes the local index task and opens the mask task; summary write/read and full read
refinement remain open. No Rust/model change or new full-suite result is claimed here.

## Accepted summary mask laws

`APEIRON-READLINK-W2-T1` integrates five audited lemmas in `Growable/Scan.lean`:
`u64_one_shift_nonzero`, `u64_or_mask_and_mask`, `summary_remainder_lt64`,
`summary_mask_nonzero`, and `summary_mask_zero_word_witness`. The generated remainder supplies
the shift bound; no new public caller premise is introduced.

- Focused/restored `lake build Growable.Scan AxiomAudit`: exit0, 3368 jobs.
- Scalar fixture OR-to-AND control: baseline0 / mutant1 / restored0.
- Independent checked zero-word counterexample: exit0; computed mutated result zero refutes
  the full nonzero law. This is scalar sensitivity only, not a generated summary update control.
- All five audits remain within `{propext, Classical.choice, Quot.sound}`; Sol accepts.

Receipts: `target/readlink/mask/`. Scan SHA256
`ecd9c9f3bbcf4d6a68175ccf66e9cf220a95eaeb7af15227d20fc6e30b31f23f`; audit SHA256
`acc076d6d941abaa66920c24b5ed5aca855db67327c267e244755630671e5a66`.
The generated source, Rust and installed model are unchanged. The next task composes actual
vector updates and reads, with a nonempty witness and a separate generated OR-to-AND control.

## Accepted generated summary write/read link

`APEIRON-READLINK-W3-T1` integrates `summary_set_implies_maybe`, `oneWordZeroRegion`,
`summary_set_maybe_one_word_witness` and their audits. The exact universal statement retains
only the successful generated `summary_set` equation as its premise. The proof uses the actual
`Vec.index_mut` back function, unchanged vector length and generated query; it adds no classifier,
nonempty or desired-result premise. The existing index proof is factored into a private reusable
WP specification without changing its public statement.

- Bounded positive/restored `lake build Growable.Scan AxiomAudit`: exit0, 3368 jobs.
- The in-source witness starts with one zero summary word and place0, proves the generated
  write succeeds, and proves the same generated query returns true.
- Disposable actual generated `summary_set` OR-to-AND mutation: baseline0 / mutant1 / restored0.
  The mutated generated module compiles; both the universal proof and concrete write witness reject.
- Independent mutant witness: exit0; the actual mutated write succeeds on the same nonempty input,
  its query returns false, and the universal claim is refuted.
- Audits remain within `{propext, Classical.choice, Quot.sound}`. Independent Sol verdict: accept.

Receipts: `target/readlink/summary/`, including `sol-review-final.md`. Scan SHA256
`b8f0d7a64dbbc24f42492cbf4c3c2e1b593d5875d9a2a2fbbee6863b923635fc`; audit SHA256
`23f51de84ca5b0cc44ba939962a44b211ee17d97efdb3cba9b0b460d79e2f155`.
Generated SHA256 remains `92c771510c259bc56c9e437e7d3f3108fa9e2796e6ba41ca2cd826f4127ff74e`.
These are local source-linked summary facts, not complete `get` refinement, `RepR` preservation,
`SearchDecisionCorrect` or allocator conformance. No Rust or installed model changes occur in this leg.

## Integrated gate receipt

All proof-task outputs are integrated on the recorded dirty source identity. Every command below
uses the 8GiB/no-swap/single-Lean-thread lane wrapper with a 900-second limit.

| Command | Cwd | Exit / observed result |
|---|---|---|
| `cargo test --workspace --all-features --locked` | repository root | 0; 263 passed, 1 ignored, 601.746s |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | repository root | 0 |
| `cargo fmt --all -- --check` | repository root | 0 |
| `proof/regen.sh --check` | repository root | 0; extraction fresh and 1708-job package build |
| `lake build` | `proof/` | 0; 1708 jobs |
| `lake build` | `proof/corpus/` | 0; 3482 jobs |
| `python3 tools/check-assurance.py --check` | repository root | 1; same seven pre-existing size breaches |
| `git diff --check` and four-document local-link check | repository root | 0; 15 links resolve |

Raw logs, commands, elapsed times, source digests and exit statuses:
`target/readlink/final-barrier/`. The test run emits an external fold `opsfold/Cargo.toml` duplicate-key
diagnostic before continuing; both Cargo test and Clippy exit0. No dependency file is edited.
The long test inserts and reads 100,000 entries in `tests/table.rs`; no performance claim derives
from the regression-suite duration. Kani is not run.

Fresh extraction records Charon `0.1.254` / `b104e24fea7d721b71e6c39fd70f26ff20bc0980`,
Aeneas `505b6ca-dirty`, extraction Rust `nightly-2026-08-18` and Lean `4.31.0`. Binary hashes and
18 source digests are in `freshness.log`; the generated digest remains unchanged. Canonical and
Mathlib identities remain recorded in `summary/digests.json`.

The size failures remain `examples/reg1/acceptance.rs`, `proof/corpus/Slang/Generated/Table.lean`,
`src/table.rs`, `tools/slang/export.py`, `tools/slang/spec_check.py`, and immediate source counts
in `proof/corpus/Slang/Source` and `tests`. Earlier claim-checker findings are not rerun or waived.
No full assurance or epic acceptance is claimed.

Changed in this continuation: `proof/corpus/Growable/Scan.lean`, `proof/corpus/AxiomAudit.lean`,
this receipt, `docs/epics/read-proof.md`, `docs/archive/handoff-openai.md`, and the separately scoped
reference refresh in `docs/epics/read-proof.md`. All task worktrees/branches are
removed; unrelated `one-allocation` is retained. No commit/push.

Next bounded obligation: derive summary recordedness through the actual `publish_live` path and
representation preservation, using the accepted summary write/read theorem. Complete spill-loop
coverage and downstream Stored/Regen public-read links still need separate bounded tasks;
`SearchDecisionCorrect` and allocator conformance remain separate obligations.
