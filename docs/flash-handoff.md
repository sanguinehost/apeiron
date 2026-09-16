# Apeiron Operator Handoff

This file owns Apeiron execution detail: order, gates and stop points for Apeiron work. The
programme thesis, ownership and outcomes are in the [programme direction](direction.md); local
representation and trust are in the [local boundary](kernel-boundary.md). Local authority: this
document owns the Apeiron execution sequence and its gates. It coordinates named cross-repo tasks
but does not dispatch an entire repository, and those local gates stay owned there.

## Canonical root and scope

The canonical object is `Apeiron.Table.{HMap,get,insertWith,build}` and
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` with the extracted `Cell<i64>` word
bodies `word_mul_denote` and `word_norm_denote`; the dotted roots are `Apeiron.Table.HMap`,
`Apeiron.Table.get`, `Apeiron.Table.insertWith` and `Apeiron.Table.build`. Every executing face must
refine the canonical `Apeiron.Table.insertWith` / `Apeiron.Table.build` object; anything not rooted
in it is not accepted.

The basis is O(1) at fixed N with bounded callbacks and nonconstant key length; reading a whole
source stays at least linear in its bytes at fixed N. The rooting gate is
`python3 tools/rooting/check.py --self-test`; `python3 tools/rooting/check.py --check` is red only
on the binned routes until their retirement.

## Current status

Extraction and the named word-kernel refinements are done at their stated scopes. `word_mul_denote`
and `word_norm_denote` quantify over every `Cell<I64>` input and dial under the `ZMod (2^64)`
denotation; partial carrier kernels retain their named success and refusal premises. The fixed-line
carrier refinement is closed at its stated scope. The real `Table` bodies are extracted, and the
generated table refinement covers arbitrary callbacks satisfying `CallbackLawsG`: the constructor
establishes `PhysicalInvG`; admitted `insert_with` and build preserve it; the concrete word instance preserves
`PhysicalInv` and canonical `insertWith` on success and returns the identical representative on
`ShardFull`, with exact refusal at `N = 0`; for positive `N`, `BuildAdmissibleN` exactly
characterizes build success versus `ShardFull` and `table_build_physicalInv_of_buildAdmissibleN`
gives the physical `build` postcondition; and `wordBuildN_zero_ok_iff_empty` characterizes the
zero-capacity build. The release identity is in
[proof/README.md](../proof/README.md); the run narrative is in
[autonomous-run-log.md](autonomous-run-log.md). A source-proof run log is historical evidence, not
fresh acceptance.

## Apeiron storage exit: all rows required

| ID | Required output | Gate (cwd) / remaining work |
|---|---|---|
| S1 | Executing `new`/`get`/`insert_with`/`build` on the real source path | `cargo test --locked --test table` (root); `proof/regen.sh --check` accepts the real bodies. |
| S2 | Constructor and witness from the actual constructor | `word_table_new_rep`; `word_table_new_joint_witness`; `word_table_joint_branch_witness`. |
| S3 | Canonical default read, new-first write and ordered merge | `emptyWordTable_read`, `word_table_get_new_first`, `word_table_get_replaced`, `word_table_build_get_acc`; bridge `lake build` exits 0. |
| S4 | Joint admission precondition and success/refusal invariant preservation for arbitrary `N` | Closed under explicit `CallbackLawsG`: generic admitted insertion and build preserve `PhysicalInvG`; `table_build_zero_ok_iff_empty` gives the exact zero-capacity split; `table_build_physicalInv_of_distinctRoutedKeysPerShardG_all` gives the all-`N` sufficient admission result. |
| S5 | Actual extraction audit | `proof/regen.sh --check`; `proof/corpus/AxiomAudit.lean` `#audit`s every headline inside `{propext, Classical.choice, Quot.sound}`. |
| S6 | Baseline / red / restore control on the real source | Same-filesystem generated-body mutation `merge value default` -> `merge default value` fails `table_insert_empty_zero`; the restored copy builds. |
| S7 | Full Rust and proof integration on one source/tool identity | `cargo test --workspace --all-features --locked`; `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`; `cargo fmt --all -- --check`; `cargo kani`; `proof/regen.sh --check`; `lake build` in `proof/` and `proof/corpus/`. All are required for epic closure. |
| S8 | Audited cone, source/tool identity, bounded perimeter, size and performance | New headlines audited inside the standard cone. TCB <= 5000 physical lines, owned source files <= 1500 lines, <= 12 immediate source files per directory. Required performance is not waived; no table performance claim is made. |

Grades at actual scope: S2–S4 are proved at their named theorems; S1 is test evidence; S5 is an
extraction audit; S6 is a mutation control, not a proof. S7 passed the full Rust/Kani/extraction and
Lean barrier on the recorded source/tool identity. S8 passed `tools/check-claims.py` and
`tools/check-assurance.py`; the latter's 6/6 rejecting controls enforce the 5,000/1,500/12 perimeter.
These storage results do not establish the separate Rust–Song correspondence.

## Other owners

- **Song A0–A6/B6** is owned by Song. Joint profiles exist, but readiness remains open until their
  selected rows have checked source-connected correspondence; A0 alone does not satisfy it.
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
python3 tools/rooting/check.py --self-test
```

`cargo kani` runs the bounded harnesses in `src/proofs.rs` (`cfg(kani)`); no feature is needed.
`proof/regen.sh` requires the pinned Aeneas/Charon under `target/aeneas/` or `AENEAS_HOME`. Tool pins
and the archive checksum are in [proof/README.md](../proof/README.md); record the versions a run
actually used rather than citing a pin from memory. `python3 tools/rooting/check.py --self-test`
exercises the canonical-rooting rejection paths; `--check` is red only on the binned routes until
their retirement.

## Barrier order

1. **Gate repair.** Restore the rejecting gates and run the canonical-rooting gate
   `python3 tools/rooting/check.py --self-test`. Gate repair and the canonical-root acceptance
   below precede any producer, generator, source-translation, proof-producing generation or
   derivation work.
2. **Canonical-root acceptance.** Every executing face is accepted only when it is rooted in
   `Apeiron.Table.HMap`, `Apeiron.Table.get`, `Apeiron.Table.insertWith` and `Apeiron.Table.build`,
   and in `Apeiron.Trichotomy.AnyRing.RingCell`, `ringMul` and `ringNormForm`, with the extracted
   `Cell<i64>` word bodies `word_mul_denote` and `word_norm_denote`. Anything not rooted in it is
   not accepted.
3. **Retire the rejected route.** The binned `proof/corpus/Slang/Derivation/` tower and the binned
   `Slang/Generated/Trace/` outputs are unrooted; the `Run`/`Derives` acceptance in the binned
   `tools/slang/trace_check.py` is unrooted. They are binned and re-rooted, not extended. The
   retired `DecodeTrace.Run`/`Derives` trace representation, the retired named proof DAG and the
   retired segment lemma are historical and are not the active architecture.
4. **Storage.** S1–S8 are closed at the explicit callback-law and no-performance-claim scope.
5. **Integration.** S7 on one source/tool identity.
6. **Correspondence.** Use the checked `Joint.WordBridge` carrier/map transport to close the Song
   source binding and R2/T3 composite before any larger vertical implementation.
7. **Apeiron barrier** is a necessary component of joint readiness, not sufficient: the Song
   foundation, the joint primitive correspondence and the named profiles are also required before
   the next large implementation opens.

## Outstanding obligations

Each item is open, not a completed result:

- **Stable `Sigma`/address interface.** The interface that carries `Sigma` in every executing
  signature and fixes `Coord` addressing per face is open; it must root in `Apeiron.Table.get` and
  `Apeiron.Table.insertWith`.
- **Hyperbolic observation domain.** The hyperbolic regime's admitted observation domain is open.
- **Fixed-shard capacity sufficient for the selected source.** The capacity `N` at which the
  selected source is admitted under `BuildAdmissibleN` is open.
- **Constructive canonical roots.** Canonical `Apeiron.Table.HMap`, `Apeiron.Table.get`,
  `Apeiron.Table.insertWith`, `Apeiron.Table.build`, `RingCell`, `ringMul` and `ringNormForm` are
  open as constructive roots for every executing face.
- **Source-fold re-root.** Re-rooting the fold onto canonical `Apeiron.Table.insertWith` and
  `Apeiron.Table.build` is open.
- **Audited cross-revision equivalence artifact.** The audited cross-revision equivalence between
  the fold lineage and canonical `ringMul` is missing.

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
