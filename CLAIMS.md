# Claim ledger

This is the headline index, not new verification evidence. Recorded results bind to the cited
revision and evidence file; none is re-run by a documentation change. An absent headline row is an
open ledger obligation, not an implicit proof. `docs/flash-handoff.md` owns epic closure.

## Record contract

Each headline record MUST name: stable ID, exact declaration/artifact, domain, native evidence grade,
delivery status, source/extraction/corpus/tool identities, joint-precondition witness, audit path and
actual axiom cone, gate command with cwd, control and baseline/red/restore outcomes, and evidence
path. A required but missing value MUST read `MISSING`; it prevents closure. Keep one declaration
per proof record when expanding the grouped inventory below. Do not copy the theorem into the ledger.

The Markdown tables are reviewable data; no current command validates their completeness. Proposed
`python3 tools/check-claims.py` is **MISSING**, not an installed gate. Its first task MUST reject a
missing headline/audit/witness, stale identity, duplicate ID and unrecognized grade. Witness presence
is a structural check; Lean checks the witness and its joint hypotheses. A `#guard` is an execution
check, not an audited theorem. Require a removal control for every required record class.

## Authority and grades

Canonical navigation: `../sanguine-apeiron/proof/Apeiron/Audit.lean` (including
`AuditCarrier.lean`); general algebra: `Trichotomy/AnyRing.lean`. The build uses the Git revision in
`proof/corpus/lake-manifest.json`, not the mutable sibling. Current recorded pin:
`d9aefc93522e2109bc96158526de6dbd52477276`. Local audit: `proof/corpus/AxiomAudit.lean`.
Allowed cone: `{propext, Classical.choice, Quot.sound}`; record each actual subset at verification.
Corpus `@grade` vocabulary is `proved`, `anchored`, `next`, `open`; it is descriptive, not enforced.

| Song label | Local meaning; no automatic promotion |
|---|---|
| `L0` | Algebraic containment category; `proved` here only with the exact audited theorem and, for Rust, extraction/refinement. |
| `L1` | SMT-contract category; not synonymous with Lean `proved` or sampled `tested`. Record the foreign method and its boundary separately. |
| `anchor` | `model`/`anchor`; corpus `anchored`. Physical interpretation is not implementation refinement. |
| No corresponding level | `tested` and `measured` remain separate evidence; corpus `next`/`open` describe unfinished work, not weaker proofs. |

Song `docs/SDLC.md` §5 has five pillars and treats performance separately; its current
`docs/ROADMAP.md` §1 requires six, including measurements. This adoption uses five headings without
dropping an epic's required measurements. The Song owner must reconcile that source conflict.

## Headline inventory

Abbreviations: `B` = `lake build` in `proof/corpus/`; `E` = `proof/regen.sh --check` at root.
All local proof rows require both. `A` = `proof/corpus/AxiomAudit.lean`; `C` = allowed cone above.
Rows marked recorded retain prior evidence; fresh outcomes and a complete witness inventory are
MISSING until verified. Tool/source digests are recorded in `proof/README.md`.

| ID | Declaration/artifact and domain | Grade / delivery | Witness / audit / gate | Evidence / remaining work |
|---|---|---|---|---|
| WORD | `apeiron_word.word_mul_denote`, `word_norm_denote`, `word_norm_mul`, `word_mul_conj`; full `I64`, `ZMod (2^64)` | proved / recorded | No logical precondition; A, C; E+B | `proof/README.md`, `docs/plan/carrier-refinement.md`; dial mutation recorded. |
| CARRIER | `apeiron_word.word_point_density_denote`, `word_point_flux_denote`, `word_exchange_impulse_denote`, `word_accept_derived_impulse_spec` | proved / recorded | Full words; acceptance branch witnesses in A, C; E+B | `proof/README.md`; remaining audited headlines must be expanded into individual records. |
| LINE3 | `apeiron_word.word_diffusive_line3_step_denote`, `word_diffusive_line3_step_total_mass_zero_source`; successful three-cell steps | proved / recorded | `word_diffusive_line3_step_zero_success`, `_odd_refusal`, `_unrepresentable_refusal`; A, C; E+B | `proof/README.md`; always-`none` mutation recorded. No word-ring energy claim. |
| TABLE-GET | `apeiron_word.safe_table.get_spec_canonical`; capacity four, `I64` keys | proved / recorded | No `WellFormed` premise; A, C; E+B | `bbecaf6`, `docs/plan/storage-refinement.md`. |
| TABLE-INSERT | `apeiron_word.safe_table.try_insert_spec`; capacity four, `WellFormed self` | proved conditional / partial | Joint initial witness MISSING; A, C; E+B | `3748d30`, `76a6824`; success preserves denotation and invariant; `Err` only promises unchanged state. |
| TABLE-REMOVE | `apeiron_word.safe_table.remove_spec`; same domain | proved conditional / partial | Joint initial witness MISSING; A, C; E+B | `76a6824`, `docs/plan/storage-refinement.md`; preservation proved. |
| TABLE-INIT | Actual `SafeTable::new` establishes the generated initial state and `WellFormed` | no evidence / open | Theorem and extraction link MISSING; future E+B | `proof/README.md`: generic array-repeat extraction blocked. A hand-built generated state proves inhabitation only. |
| TABLE-BRANCH | Successful insertion iff present key or free slot; full refusal returns original key/value | no evidence / open | Theorem and inhabited full/present/free cases MISSING; future E+B | Required by `docs/spec/storage-refinement.md`; always-`Err` must fail, not merely preserve `self`. |
| MAP-TRACE | `tests/safe_table.rs`; 10,000 selected production/reference transitions | tested / recorded | `cargo test --locked --test safe_table` at root | `docs/plan/storage-refinement.md`; no universal optimized `Map` refinement. |
| BINNING | `benches/spatial_binning.rs` | measured / historical only | `cargo bench --locked --bench spatial_binning` at root | `README.md`; machine/current intervals MISSING for renewed claims. Storage epic makes no performance claim. |

`proved conditional` above means native grade `proved`, with the written conditional domain; it is
not a new grade or non-vacuous epic acceptance. The inventory is deliberately partial, not a claim
that all `#audit` lines already have complete records.

## Perimeter and non-vacuity barrier

Proposed `python3 tools/check-assurance.py` is **MISSING**. Before implementing it, freeze a manifest
of local trust: extraction selection/audit driver, unverified translation/model boundary, axiom and
claim checkers, and every helper able to admit a headline. Count each local file once using physical
lines including comments/blanks; total local trusted checking code ≤5,000. Count owned Rust/Lean/
gate source separately: ≤1,500 lines/file and ≤12 immediate source files/directory, excluding build
products, vendored dependencies and worktrees but listing those exclusions. Include all owned
`proof/` packages and source dependencies; no hand-picked bridge subset.

List Lean kernel, Lean/Mathlib, Charon/Aeneas and their models/pins outside the local count without
calling them verified by that count. Generated code and kernel-checked proofs are assurance surface,
not automatically trusted checking code. `rustc`/LLVM, unsafe `Map`, kernel ABI and GPU remain outside
the source-refinement claim. Changing the perimeter requires review, not an exclusion to meet budget.

The gate MUST reject absent manifest members, over-budget totals/files/directories, missing joint
witness coverage and omitted audits. Test each violation in a throwaway copy, baseline/red/restore;
infrastructure failure is not semantic rejection. The current `lake build` checks only imported
declarations; it cannot detect a required theorem that was never written. Missing assurance tooling
keeps the storage epic open even after its next Lean proof passes.
