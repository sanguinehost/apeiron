# Slang Source Refinement Plan

This plan decomposes the checked route from exact selected Slang bytes to the canonical table
definitions through the re-rooted memo-store contract.
[`slang-source-semantics.md`](slang-source-semantics.md) is normative. Each stage is a
barrier: its gate and negative control pass before the next stage opens. Planned paths are outputs,
not claims that those files exist.

## Arc block

SCOPE: the slang-source-semantics note, the slang-source-refinement-plan note, and
`tools/slang/spec_check.py`; complete when all specification checker modes pass.

REGISTER: planning/specification; source arena, memo store, addressing, storage and the algebraic
link; no correspondence claim.

CORPUS: `../sanguine-apeiron/proof/Apeiron/Table/Basic.lean`, exact
`Apeiron.Table.{HMap,get,insertWith,build}` targets, and
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}`.

REFERENCE: `~/song/src/table/table.slang` and its selected import/grammar files
listed in the specification.

GATE: `python3 tools/slang/spec_check.py --check`; baseline before this task was a missing checker.

GRADE: specification consistency only; no proved or tested semantic claim.

## Contract decision

The prescribed `Slang/Derivation/` `Layer`/`Derives` route is rejected as unrooted. The
`Slang/Generated/Trace/` generated traces are rejected as unrooted. The `DecodeTrace.Run`
representation is rejected, the named proof DAG is rejected, and the segment-lemma method is
rejected. The active route is the re-rooted contract below: an immutable source arena, a memo store
refining `Apeiron.Table.HMap MemoKey MemoValue`, an ordered fold of checked transitions, and a
constructive algebraic link. The selected table body is decoded for source identity and is
not the active refinement object.

## Bounded source-witness route

Decision: extend the existing `tools/slang/export.py` with a chunked witness generator that emits
named bounded derivations over `proof/corpus/Slang/Source/Tokenizer/` and
`proof/corpus/Slang/Source/Parsing/`. Compose them with the lemmas at
`37b95c385ae0e58d020346a0e697053199af5940`, listed with their premises in the
[specification](slang-source-semantics.md#bounded-source-witness-composition).
This is a required implementation, not a claim that the exporter already emits these witnesses:
its `render_derivation` currently emits source definitions and compiled `#guard` checks.

The retired `Slang/Derivation/` named proof DAG / segment-lemma evaluator and the retired
`Slang/Generated/Trace/` outputs are binned, not revived by this route. Proof-term chunking uses
shared data/index equations and named bounded constructor derivations of the existing source
relations; it introduces no replacement evaluator or trace-acceptance relation. Kernel checking
and the empty axiom cone remain mandatory, at default recursion depth with bounded heartbeats,
time and memory. Source witnesses discharge byte binding only; they do not establish the memo-store
refinement or compiler/ABI correctness.

### Barrier-probe measurements

The supplied barrier probe reports flat-term infeasibility and motivates chunking:

| Measured item | Barrier-probe result |
|---|---|
| Largest elaborated flat `Utf8Scan` term | 33 constructors over 16 bytes |
| `tableSource` | 7,171 bytes / 1,457 tokens / 2,394 lexer steps |
| `stdSource` | 27,160 bytes / 3,736 tokens / 6,097 lexer steps |
| Smallest failing goal | `Utf8Scan (tableSource.length + 1) (tableSource.map UInt8.ofFin) tableChars` |

These are reported measurements, not proofs or a universal maximum. The source inputs are bound by
`tools/slang/export.py` and the source identities in the specification. Residual evidence: the
barrier-probe command, raw log, machine and tool identity are not supplied here; these figures are
not reproduced by this documentation gate. Chunked whole-source feasibility remains unmeasured.

### Retirement before generation

1. Retire the binned files, imports, Lake roots, registry entries and trace acceptance; require
   `python3 tools/rooting/retire_check.py (unpublished) --check` and `--self-test` before generating witnesses.
2. Retain producer rule R6 in `tools/rooting/retire_check.py (unpublished)` and
   `tools/rooting/retire_cases.json (unpublished)`: reject an exporter that can reintroduce the retired
   `Slang/Generated/Trace/` or retired `Slang/Derivation/` route, or emit `maxHeartbeats 0` or
   `maxRecDepth 1000000`. R6 is a textual rejection gate, not proof of arbitrary Python behavior.
3. Generate bounded table/std UTF-8 witnesses first, sharing data and index equations; keep
   identifiers and comments within one lexer chunk and satisfy every composition premise in the
   specification. Default recursion depth applies to data definitions as well as proofs; the
   exporter's current `maxRecDepth 10000` output is not acceptance for this route.
4. Compose lexer/parser witnesses and the closed binding, then run the unconditional source gate.
   Each implementation dispatch owns at most three files and one focused gate/control; generated
   output paths and the focused UTF-8 gate must be assigned before dispatch. No implementation
   ownership or additional evaluator is allocated by this documentation decision.

### Documentation boundary arc

SCOPE: the slang-source-refinement-plan note, the slang-source-semantics note,
the design notes, the design notes; explicitly assigned four-file the reviewer documentation
pass in the supplied checkout, base `8fe3eb3d51b6d9ef6bccb80774372f31e90cd54a`; no Lean build lane.

REGISTER: planning/specification; route and trust boundary only; no code, proof or measurement run.

CORPUS: composition lemmas at `37b95c3`; `Utf8Scan.continue`, `DistinctDerivation.append` and
`DecodeTokensDerivation.assemble`; source signatures, not fresh proof acceptance.

REFERENCE: `~/apeiron/proof/corpus/Slang/Source/Parsing/Validation.lean`,
`DistinctDerivation.append`; companion modules listed in the specification.

GATE: root `python3 tools/slang/spec_check.py --check`; baseline exit 0, no observed red;
control: the same checker in `--self-test` mode.

GRADE: specification consistency only; barrier-probe figures are reported, not rerun measurements.

## Re-rooted contract

- **Source arena.** An immutable arena owns the exact bytes and bounded parsed records; `SourceRef`
  is a file identity with a checked byte interval and a checked token interval plus in-bounds and
  correspondence proofs. No copied text, filename-only identity, unchecked offset or prefix parse.
- **State and result arenas.** Bounded handles with checked membership; immutable records;
  allocation returns a fresh handle or an explicit arena-capacity refusal.
- **Memo key.** The complete computation identity: source ref, operation, context/environment,
  precedence, every recursion, token, byte, stack and result limit, every input state handle,
  `Sigma` and the selected addressing face. A digest or coordinate alone is not key equality.
- **Memo value.** A bounded result handle plus resulting state handles, the consumed source interval,
  and checked evidence that the result denotes the specified transition; a value without evidence is
  not a memo hit.
- **Addressing.** Every executing signature carries `Sigma`; elliptic exact reversible packing on an
  explicit admitted domain; parabolic carried complete key (a coordinate never replaces key
  comparison); hyperbolic scramble/discard only on an explicit observation domain. One home shard; no
  fallback probing.
- **Storage.** 256 shards, capacity `N`, full-key comparison; no growth, fallback container, silent
  eviction or unrelated-key replacement. The canonical type is `Apeiron.Table.HMap MemoKey MemoValue`;
  reads, updates and builds denote the canonical `Apeiron.Table.get`, `Apeiron.Table.insertWith` and
  `Apeiron.Table.build` definitions.
- **Fold.** An ordered fold of checked transitions; successful execution denotes the same ordered
  update stream under `Apeiron.Table.build`; reordering, skipping or synthesizing a transition is
  forbidden.
- **Pointwise invariant and refusal.** `RepMemo(store, h)` with `N = 0` (constructor and total default
  reads valid; every insertion refuses; empty fold succeeds; any storage-requiring fold refuses
  without a partial table); `ShardFull` and the non-storage refusals are distinct, each with joint
  witnesses.
- **Algebraic link.** The constructive `Slang.Refinement.wordDenote`, `Slang.Refinement.cellDenote`,
  `Slang.Refinement.cellDenote_mul` and `Slang.Refinement.cellNormForm_denote` link to
  `Apeiron.Trichotomy.AnyRing.ringMul` and `Apeiron.Trichotomy.AnyRing.ringNormForm`;
  `word_mul_denote` and any `ZMod` declaration MUST NOT enter the empty-cone chain.

## Staged tasks

`S/` abbreviates `proof/corpus/Slang/`. Each task owns at most three files and one gate. A stage's
gate and its negative control pass before the next stage opens.

| Wave | Purpose | Owned files (maximum three) | Gate | Negative control | Opens / stop condition |
|---|---|---|---|---|---|
| W21 | Rejecting gate harness and exact fixture manifest | `tools/slang/spec_check.py`; `tools/slang/controls.json`; `tools/slang/source_manifest.json` | `python3 tools/slang/spec_check.py --self-test` | Change one selected byte while retaining the old manifest; the gate MUST reject source identity. | Requires this specification gate. Stop if a mutation can survive or the Song/corpus checkout identity cannot be resolved. |
| W22 | Decode exact bytes into bounded parsed records with complete body consumption | `proof/corpus/Slang/Syntax.lean`; `proof/corpus/Slang/Decode.lean`; `proof/corpus/Slang/DecodeTest.lean` | `cd proof/corpus && lake env lean Slang/DecodeTest.lean` | Delete the final statement of `tableInsertWith` or append a reachable unsupported token; the decode test MUST reject. | Requires W21. Stop on accepted prefixes, opaque statements, or an unsupported reachable construct. |
| W23 | Source arena owning exact bytes and checked `SourceRef` records | `S/Source/Arena.lean`; `S/Source/ArenaAudit.lean`; `S/Source/Ref.lean` | `cd proof/corpus && lake env lean Slang/Source/ArenaAudit.lean` | Copy record text instead of referencing its interval, or use an unchecked offset; the arena audit MUST fail. | Requires W22. Stop if any record can name bytes it does not own or parse a prefix. |
| W24 | State and result arenas with bounded, checked handles | `S/Arena/State.lean`; `S/Arena/Result.lean`; `S/Arena/HandleAudit.lean` | `cd proof/corpus && lake env lean Slang/Arena/HandleAudit.lean` | Accept an out-of-capacity allocation or reuse a live handle; the handle audit MUST fail. | Requires W23. Stop on any unchecked membership or mutable record. |
| W25 | Memo key as complete computation identity | `S/Memo/Key.lean`; `S/Memo/KeyAudit.lean`; `S/Memo/KeyEquality.lean` | `cd proof/corpus && lake env lean Slang/Memo/KeyAudit.lean` | Equate two keys differing in a limit, state handle or `Sigma`; the key-equality audit MUST fail. | Requires W24. Stop if a digest or coordinate alone decides key equality. |
| W26 | Memo value with checked transition evidence | `S/Memo/Value.lean`; `S/Memo/ValueAudit.lean`; `S/Memo/Hit.lean` | `cd proof/corpus && lake env lean Slang/Memo/ValueAudit.lean` | Drop the transition evidence from a stored value and still report a hit; the value audit MUST fail. | Requires W25. Stop on any value without evidence, or any hit without a consumed interval. |
| W27 | Addressing faces with `Sigma` in every executing signature | `S/Address/Faces.lean`; `S/Address/FacesAudit.lean`; `S/Address/Sigma.lean` | `cd proof/corpus && lake env lean Slang/Address/FacesAudit.lean` | Replace the parabolic full-key comparison with a coordinate comparison; the faces audit MUST fail. | Requires W26. Stop on any signature without `Sigma` or any fallback probe. |
| W28 | Fixed-shard memo store with full-key comparison | `S/Memo/Store.lean`; `S/Memo/StoreAudit.lean`; `S/Memo/Refinement.lean` | `cd proof/corpus && lake env lean Slang/Memo/StoreAudit.lean` | Allow an unrelated-key replacement or a silent eviction; the store audit MUST fail. | Requires W27. Stop on growth, a fallback container or a partial table. |
| W29 | Ordered fold of checked transitions and canonical `Apeiron.Table.build` | `S/Memo/Fold.lean`; `S/Memo/FoldAudit.lean`; `S/Memo/Joint.lean` | `cd proof/corpus && lake env lean Slang/Memo/FoldAudit.lean` | Reorder two transitions; the ordered-build denotation MUST fail. | Requires W28. Stop on any reordered, skipped or synthesized transition. |
| W30 | Pointwise invariant, `N = 0`/`ShardFull` refusals and the constructive algebraic link | `S/Memo/Invariant.lean`; `S/Refinement/Algebra.lean`; `S/Refinement/AlgebraAudit.lean` | `cd proof/corpus && lake env lean Slang/Refinement/AlgebraAudit.lean` | Make an `N = 0` insertion succeed, or merge `ShardFull` with a non-storage refusal; the invariant audit MUST fail. | Requires W29. Stop without joint success and refusal witnesses and an empty axiom cone. |

## Integration order and barriers

1. W21 makes false-green source drift rejectable.
2. W22 decodes the exact bytes; W23 owns them as bounded parsed records.
3. W24 establishes checked state/result handles; W25 and W26 define the memo key and value over them.
4. W27 fixes the addressing faces and `Sigma`; W28 stores them in the fixed-shard memo store.
5. W29 composes the ordered fold and links it to the canonical `build`.
6. W30 closes the pointwise invariant, the `N = 0`/`ShardFull` refusals and the constructive
   algebraic link.

Every stage records source/corpus identities and preserves failures from earlier stages. A failed
gate, surviving negative control, unresolved source/canonical reference, uninhabited joint premise,
or unapproved axiom stops integration. Every stage control is a baseline green, mutation red, restored green sequence on one source/tool identity; a control that survives, or a red that fails
for an unrelated reason, is a failed control, not a pass. Shared `proof/corpus/.lake/` use is
serialized.

## Forbidden dispatches

- **FORBIDDEN: prove all Slang.** Dispatch only the reachable selected fragment; unsupported reachable
  syntax rejects and unreachable syntax remains outside the theorem.
- **FORBIDDEN: implement a Rust evaluator.** Rust/Aeneas runtime linkage is outside the route and a
  second evaluator would not bind the Song source.
- Do not use ISA records as source semantics, accept a body prefix, copy the table algorithm into
  Lean, infer status from raw `-1`, merge occupied state into the extensional map, or recode memory or
  stack faults as table refusal.
- Do not present a retired route as active. The derivation tower, generated decoder traces,
  `DecodeTrace.Run`, the named proof DAG and the segment lemma are retired.

## Source-binding contract

The source headline remains open until the unconditional theorem is checked with an empty axiom cone.
`ClosedBinding selectedBundle value` carries the entry binding, the exact table-to-cell and
cell-to-std edges with bundle lookups, the `ImportedCoordDerivation` for the literal
`8796093022208`, and `value = Decode.appendImportedCoord entryDecoded coord`. The closed program has all nine selected constants. The old premise pair `ExpectedSelectedSyntax value.val.program = true`
and `ParsesClosure selectedBundle value.val.program` is contradictory and MUST NOT remain under any
name. The corrected helper remains only as `source_binding_of_closed_binding`; it is not the headline
or acceptance target. Replacing the unconditional target with the conditional helper MUST fail.
Acceptance requires the unconditional `Slang.SourceAudit.source_binding` theorem, its empty axiom
cone and full-byte freshness. The active refinement is the memo store; generated decoder traces are
not an acceptance condition.

## Retired route

The `Slang/Derivation/` `Layer`/`Derives` tower is retired. The `Slang/Generated/Trace/` generated
decoder traces are retired. The `DecodeTrace.Run` representation is retired, the named proof DAG is
retired, and the segment lemma is retired; none MUST be presented as the active architecture.
`RepSong`, `observeRead`, raw `memory[]` and `MERGE_MAP` are retired as the primary refinement.
Historical references to a retired route carry a retirement marker and are not acceptance.

## Implementation acceptance

The implementation acceptance is the `tools/slang/refine.py` exact-declaration probe on the
unconditional existential, red until the theorem exists. The probe binds
`Slang.SourceAudit.source_binding` to the exact unconditional type, rejects a same-named helper, a
theorem of `True`, a wrong type, an axiom or a `sorry` target, and fails before the declared
`TABLE_PHI` source mutation. A timeout or memory failure is inconclusive, not a passing control.

## Storage S1-S8 (Rust-table scope)

The Rust `Table` storage evidence S1-S8 remains at its Rust-table scope as recorded in
[`../plan.md`](../plan.md) and [`../epics/storage-refinement.md`](../epics/storage-refinement.md):
S1 executing `new`/`get`/`insert_with`/`build` on the real source path; S2 constructor and joint
witnesses from the actual constructor; S3 canonical default read, new-first write and ordered merge;
S4 joint admission and success/refusal preservation; S5 the extraction audit; S6 the
baseline/red/restore control; S7 full Rust and proof integration on one source/tool identity; S8 the
audited cone, perimeter, size and performance statement. These are Rust-table results; none
transfers to the memo store without its own refinement.

## Not covered

This plan does not prove correspondence, all-Slang semantics, compiler/emitter/ISA correctness,
native stack safety, the kernel ABI, rustc/LLVM, GPU/SPIR-V, or performance. The staged tasks produce
those claims only where a named stage says so; the present task establishes specification consistency
only.
