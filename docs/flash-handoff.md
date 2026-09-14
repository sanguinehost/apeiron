# Apeiron Operator Handoff

This file owns Apeiron execution detail: order, gates and stop points for Apeiron work. The
programme thesis, ownership and outcomes are in the [programme direction](direction.md); local
representation and trust are in the [local boundary](kernel-boundary.md). Local authority: this
document owns the Apeiron execution sequence and its gates. It coordinates named cross-repo tasks
but does not dispatch an entire repository, and those local gates stay owned there.

## Current status

Extraction and the named word-kernel refinements are done at their stated scopes. `word_mul_denote`
and `word_norm_denote` quantify over every `Cell<I64>` input and dial under the `ZMod (2^64)`
denotation; partial carrier kernels retain their named success and refusal premises. The fixed-line
carrier refinement is closed at its stated scope. The real `Table` bodies are extracted, and its
constructor, key-zero write/read, two-write build and concrete full-shard refusal witnesses are
proved. Universal `Rep` preservation and admission completeness for arbitrary `N` remain open. The release identity is in
[proof/README.md](../proof/README.md); the run narrative is in
[autonomous-run-log.md](autonomous-run-log.md). A source-proof run log is historical evidence, not
fresh acceptance.

## Apeiron storage exit: all rows required

| ID | Required output | Gate (cwd) / remaining work |
|---|---|---|
| S1 | Executing `new`/`get`/`insert_with`/`build` on the real source path | `cargo test --locked --test table` (root); `proof/regen.sh --check` accepts the real bodies. |
| S2 | Constructor and witness from the actual constructor | `word_table_new_rep`; `word_table_new_joint_witness`; `word_table_joint_branch_witness`. |
| S3 | Canonical default read, new-first write and ordered merge | `emptyWordTable_read`, `word_table_get_new_first`, `word_table_get_replaced`, `word_table_build_get_acc`; bridge `lake build` exits 0. |
| S4 | Joint admission precondition and success/refusal invariant preservation for arbitrary `N` | Open. Universal `Rep` preservation and admission completeness are not proved; the current witnesses are concrete word instances at the default `N = 16`. |
| S5 | Actual extraction audit | `proof/regen.sh --check`; `proof/corpus/AxiomAudit.lean` `#audit`s every headline inside `{propext, Classical.choice, Quot.sound}`. |
| S6 | Baseline / red / restore control on the real source | Same-filesystem generated-body mutation `merge value default` -> `merge default value` fails `table_insert_empty_zero`; the restored copy builds. |
| S7 | Full Rust and proof integration on one source/tool identity | `cargo test --workspace --all-features --locked`; `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`; `cargo fmt --all -- --check`; `cargo kani`; `proof/regen.sh --check`; `lake build` in `proof/` and `proof/corpus/`. All are required for epic closure. |
| S8 | Audited cone, source/tool identity, bounded perimeter, size and performance | New headlines audited inside the standard cone. TCB <= 5000 physical lines, owned source files <= 1500 lines, <= 12 immediate source files per directory. Required performance is not waived; no table performance claim is made. |

Grades at actual scope: S2–S3 are proved at their named word-instance theorems; S1 is test
evidence; S5 is an extraction audit; S6 is a mutation control, not a proof. S4 is open. S7 and S8
remain open for epic closure. This recorded evidence is not fresh all-green acceptance. Full
coverage and perimeter enforcement are not existing gate scripts: the missing coverage/perimeter
gate is open and must be named when it is created, not pointed at by a path that does not exist. Do
not reintroduce deleted `check_claims`/`check_assurance` as if they were mandated tools.

## Other owners

- **Song A0–A6/B6** is owned by Song. The joint readiness profile is MISSING and is not satisfied by
  A0 alone; the existing milestones are not dropped.
- **Corpus F2/F3 and A5** are owned by the corpus repository. The canonical decider is now complete
  on its named fragments: `certificate_complete` (uncapped `CertOp`), `completeSat_eq_true_iff`
  (residue decision), `residue_certificate_complete` and `checkCertStream_sound` (framed replay). The
  Song executing reader refinement and the Slang source binding remain open.
- **Penglai material integration** is owned by Penglai.
- **Fold sunset** is owned by Fold.

Apeiron points to these owners; it does not dispatch whole other repositories. Current owner
roadmaps: [canonical corpus](../../sanguine-apeiron/theory/ROADMAP.md),
[Song](../../song/docs/ROADMAP.md), [Penglai](../../penglai/README.md),
[Whispering Tides](../../whispering-tides/ROADMAP.md).

## Local commands

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
cargo kani
proof/regen.sh --check
cd proof/corpus && lake build
```

`cargo kani` runs the bounded harnesses in `src/proofs.rs` (`cfg(kani)`); no feature is needed.
`proof/regen.sh` requires the pinned Aeneas/Charon under `target/aeneas/` or `AENEAS_HOME`. Tool pins
and the archive checksum are in [proof/README.md](../proof/README.md); record the versions a run
actually used rather than citing a pin from memory.

## Barrier order

1. Storage: S1–S6 first; S4 is the open universal `Rep` obligation.
2. Integration: S7 on one source/tool identity.
3. Assurance: S8, with the coverage/perimeter gate named when it is created.
4. An Apeiron barrier is a necessary component of joint readiness, not sufficient: the Song
   foundation, the joint primitive correspondence and the named profiles are also required before
   the next large implementation opens.

## Storage next wave

The next bounded storage wave defines a physical `Rep` invariant for arbitrary fixed-shard word
states and proves successful `insert_with`/`build` preservation plus exact admission completeness.
The old `SafeTableRefinement` and `word_safe_table4_*` rows are superseded. No storage performance
claim is made.

## Evidence labels

A claim is `proved` only when a Lean kernel check passes for a named theorem with the extraction from
the real source and a closed axiom cone; `tested` when a `cargo test` executes the build path and
its boundary cases; `measured` when a named command ran and its output was read. A model or anchor
is a mapping, not evidence of correctness.

## Historical evidence pointers

- [proof/README.md](../proof/README.md) — extraction, bridge, pins, axiom cone and controls.
- [autonomous-run-log.md](autonomous-run-log.md) — run narrative and decisions.
- The predecessor documentation at Git `1403767e671c81737525702946a3eb1eae4f0b0b`:
  [storage plan](plan/storage-refinement.md), [carrier plan](plan/carrier-refinement.md) and
  [corpus release pin plan](plan/corpus-release-pin.md).

These are pointed to, not copied. Their contents bind to the revisions they record.

## Stop points and evidence

- A documentation pass does not require the full software barrier. A source, extraction, package or
  pin change needs a separate task and its own local barrier.
- Each task owns at most three files and one gate with one negative control; the next wave opens
  only when every child and shared gate is green. A missing gate is named as an artifact to
  create, never replaced by a command that does not exist.
- Report per slice: changed files, the actual commands and exit codes, what is proved versus tested
  versus open, and the next bounded task. A successful build, a sampled test and a universal proof
  are different evidence.
- Preserve unrelated dirty work. Do not commit, push, post externally, or modify another active
  workstream unless the user requests it.
