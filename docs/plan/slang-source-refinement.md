# Slang Source Refinement Plan

This plan decomposes the checked route from exact selected Slang bytes to the canonical table
definitions. [`../spec/slang-source-semantics.md`](../spec/slang-source-semantics.md) is normative.
Each stage is a barrier: its gate and negative control pass before the next stage opens. Planned paths
are outputs, not claims that those files exist.

## Arc block

SCOPE: `docs/spec/slang-source-semantics.md`, `docs/plan/slang-source-refinement.md`, and
`tools/slang/spec_check.py`; complete when all specification checker modes pass.

REGISTER: planning/specification; source binding, state semantics, table refinement, and resources;
no correspondence claim.

CORPUS: `../sanguine-apeiron/proof/Apeiron/Table/Basic.lean`, exact
`Apeiron.Table.{HMap,get,insertWith,build,get_insertWith_self,get_insertWith_ne}` targets.

REFERENCE: `/home/socol/Workspace/song/src/table/table.slang` and its selected import/grammar files
listed in the specification.

GATE: `python3 tools/slang/spec_check.py --check`; baseline before this task was a missing checker.

GRADE: specification consistency only; no proved or tested semantic claim.

Contract-correction decision: the existing conditional `Slang.SourceAudit.source_binding` has
contradictory entry/closed indices. The source headline remains open until the unconditional theorem
below is checked with an empty axiom cone. Compiled whole-source guards do not downgrade this
obligation to `tested`. `python3 tools/slang/spec_check.py --check` remains red while the conditional
artifact occupies that name.

## Staged tasks

The table below is the original route. The Astra root-cause freeze (`## Root-cause freeze`) revises
W21-W30: it reopens W21 and W22, replaces the W24 artifact with a proof-producing translation, and
records the decoder's inline-conditional precedence defect. Where the two disagree, the freeze governs.

| Wave | Purpose | Owned files (maximum three) | Gate | Negative control | Opens / stop condition |
|---|---|---|---|---|---|
| W21 | Rejecting gate harness and exact fixture manifest | `tools/slang/source_semantics_check.py`; `tools/slang/source_manifest.json`; `tools/slang/source_controls.json` | `python3 tools/slang/source_semantics_check.py --check harness` | Change one selected byte while retaining the old manifest; gate MUST reject source identity. | Requires this specification gate. Stop if a mutation can survive or the Song/corpus checkout identity cannot be resolved. |
| W22 | Decode exact bytes into the selected AST with complete body consumption | `proof/corpus/Slang/Syntax.lean`; `proof/corpus/Slang/Decode.lean`; `proof/corpus/Slang/DecodeTest.lean` | `cd proof/corpus && lake env lean Slang/DecodeTest.lean` | Delete the final statement of `tableInsertWith` or append a reachable unsupported token; decode test MUST reject. | Requires W21. Stop on accepted prefixes, opaque statements, or an unsupported reachable construct. |
| W23 | Define the 64-bit source machine, environments, control, traces, and raw memory | `proof/corpus/Slang/Word.lean`; `proof/corpus/Slang/Memory.lean`; `proof/corpus/Slang/Semantics.lean` | `cd proof/corpus && lake env lean Slang/Semantics.lean` | Replace eager bitwise `or` with short-circuit evaluation or make multiplication unbounded; semantic fixtures MUST fail. | Requires W22. Stop until signed arithmetic, lazy conditionals, faults, and state transitions have one rule each. |
| W24 | Bind decoded declarations, constants, imports, and reachable calls to exact source bytes | `proof/corpus/Slang/SourceBinding.lean`; `proof/corpus/Slang/TableSource.lean`; `tools/slang/render_source_binding.py` | `python3 tools/slang/source_semantics_check.py --check binding` | Change `TABLE_PHI`, a brace boundary, or a reachable callee while retaining the theorem input; binding gate MUST reject. | Requires W22-W23. Stop if a digest is the only dependency or any selected body is handwritten in Lean. |
| W25 | Prove constructor execution and `RepSong` establishment | `proof/corpus/Slang/Table/TableRep.lean`; `proof/corpus/Slang/Table/TableInit.lean`; `proof/corpus/Slang/Table/TableWitness.lean` | `cd proof/corpus && lake env lean Slang/Table/TableInit.lean` | Omit the value-cell store in decoded `tableInitGo`; constructor establishment MUST fail. | Requires W24. Stop without one joint valid constructor witness and exact outside-memory preservation. |
| W26 | Prove probe/read correctness and state-aware observation | `proof/corpus/Slang/Table/TableProbe.lean`; `proof/corpus/Slang/Table/TableRead.lean`; `proof/corpus/Slang/Table/ReadWitness.lean` | `cd proof/corpus && lake env lean Slang/Table/TableRead.lean` | Classify raw `-1` as absence without consulting `O`; the present-`-1` witness MUST fail. | Requires W25. Stop unless collision reachability, misses, raw return, and unchanged memory are all covered. |
| W27 | Prove replacement write preservation and exact write admission/refusal | `proof/corpus/Slang/TableWrite.lean`; `proof/corpus/Slang/TableAdmission.lean`; `proof/corpus/Slang/WriteWitness.lean` | `cd proof/corpus && lake env lean Slang/TableWrite.lean` | Make a full table reject a present key or write before an explicit refusal; preservation/admission MUST fail. | Requires W26. Stop without jointly inhabited successful and full-table refusing examples. |
| W28 | Prove termination and separate source-level resource projections | `proof/corpus/Slang/TableTermination.lean`; `proof/corpus/Slang/TableResources.lean`; `proof/corpus/Slang/ResourceWitness.lean` | `cd proof/corpus && lake env lean Slang/TableResources.lean` | Remove `step + 1` or count the spread mask while omitting its multiply; termination/resource gate MUST fail. | Requires W27. Stop if timeout is treated as a verdict or source depth is promoted to a native stack guarantee. |
| W29 | Link replacement execution to canonical `Apeiron.Table.insertWith` under the empty axiom cone | `proof/corpus/Slang/Joint/SongTable.lean`; `proof/corpus/Slang/Joint/SongTableControl.lean` | `cd proof/corpus && lake env lean Slang/Joint/SongTableControl.lean` | Change decoded `MERGE_MAP` from replacement to addition; `SongTableControl` or the audited headline MUST fail. | Requires W24-W28. Stop on any unapproved axiom, missing source dependency, lost raw result/state, or claim beyond replacement. |
| W30 | Specify the unmatched accumulator/default adaptation before proof dispatch | `docs/spec/joint-table-accumulator.md`; `proof/corpus/Slang/Joint/AccumulatorProfile.lean` | `cd proof/corpus && lake env lean Slang/Joint/AccumulatorProfile.lean` | Equate fresh Song ACC insertion with canonical addition at an arbitrary nonzero default; profile check MUST reject. | Requires W29. Stop until fresh/present behavior, absence/default adaptation, admission, state, and resources are explicit. |

## Integration order and barriers

1. W21 makes false-green source drift rejectable.
2. W22 and W23 define syntax and machine semantics separately; W24 composes them with exact bytes.
3. W25 establishes the only admissible initial table population. W26 and W27 consume that invariant.
4. W28 adds termination and resources without changing functional semantics.
5. W29 is the first correspondence-relevant replacement link and runs the complete corpus audit.
6. W30 specifies the accumulator/default mismatch; no accumulator theorem is dispatched before it.

Every stage records source/corpus identities and preserves failures from earlier stages. A failed gate,
surviving negative control, unresolved source/canonical reference, uninhabited joint premise, or
unapproved axiom stops integration. Shared `proof/corpus/.lake/` use is serialized.

## Forbidden dispatches

- **FORBIDDEN: prove all Slang.** Dispatch only the reachable selected fragment; unsupported reachable
  syntax rejects and unreachable syntax remains outside the theorem.
- **FORBIDDEN: implement a Rust evaluator.** Rust/Aeneas runtime linkage is outside the route and a
  second evaluator would not bind the Song source.
- Do not use ISA records as source semantics, accept a body prefix, copy the table algorithm into
  Lean, infer status from raw `-1`, merge occupied state into the extensional map, or recode memory or
  stack faults as table refusal.

## Not covered

This plan does not prove correspondence, all-Slang semantics, compiler/emitter/ISA correctness,
native stack safety, the kernel ABI, transactional multi-insert build, accumulator adaptation,
rustc/LLVM, GPU/SPIR-V, or performance. W21-W30 produce those claims only where a named stage says so;
the present task establishes specification consistency only.

## Root-cause freeze

Source: Astra escalation on the W24 source binding. W24 stays ungraded until its headline exists; the
whole-file kernel normalization is replaced by a proof-producing source translation plus
compositional denotational refinement. Bound every Lean and `tools/slang/refine.py` run with
`LEAN_NUM_THREADS=1`, a `timeout`, and a process-tree memory limit where available (for example a
cgroup: `systemd-run --user --scope -p MemoryMax=6G ...`). Do **not** use `ulimit -v`: it breaks
Lean's thread creation and has produced spurious `failed to create thread` failures.

### Why the original W24 cannot be proved as written

The kernel cannot evaluate the decoder on the real bytes: `ByteArray` is not kernel-computable
(ByteArray.mk is `noncomputable`, `"...".toUTF8` is stuck), and `String` operations exceed the
heartbeat limit at 7 KB. A pure-`List Char` lexer evaluates in 2.8 s / ~1 GB, but the parser path
(`scanSelected` -> the `mutual` `parseBlockFuel`/`parseStmtFuel` and
`parseExprFuel`/`parsePrimaryFuel`/`parseArgsFuel`/`parseBinaryFuel`) exceeds 18 GB and does not
finish. Proving the equality by evaluation is therefore both intractable and the wrong method.

### Decoder defects found during the freeze

| Location | Defect |
|---|---|
| `proof/corpus/Slang/Decode.lean:202-207` | Inline-conditional parsing ignores `minPrec`, so `0 - k if k < 0 else k` parses as `0 - (k if k < 0 else k)` rather than `(0 - k) if k < 0 else k`. This is the actual `tableSpread` first assignment. |
| `proof/corpus/Slang/DecodeTest.lean:44-46` | The fixture checks only the final multiply/mask expression, so it does not catch the precedence defect. |
| `proof/corpus/Slang/Decode.lean:300-319` | Token-pattern scanning rather than top-level declaration-boundary parsing; a removed selected name can skip a later conflicting occurrence. |
| `proof/corpus/Slang/Decode.lean:404-410` | `decodeClosure` receives only table and std bytes; it neither validates the intermediate cell.slang import nor the textual import edges. |
| Certificate.Valid | Five Boolean equalities set to `true` by the success branch; it does not express grammar or import correctness. |
| Machine.Context.ofProgram | Takes its constant environment independently; binding it to the decoded constants is still open. |
| `tools/slang/controls.json` | Wires the source stage to `tools/slang/source_semantics_check.py`, which is absent, and accepts the prefix `Slang.SourceAudit.*` rather than the exact `source_binding`; the probe checks existence, not theorem type or axiom cone. |

### Method

An untrusted producer supplies literal bytes, candidate syntax and local derivations. Kernel-checked
rules validate byte decoding, token boundaries, grammar productions, declaration boundaries and
import resolution. Soundness lemmas relate those derivations to the real Decode.decodeClosure, and
per-declaration proofs compose by named lemmas, so no application unfolds the whole mutual parser.
Candidate ASTs are generated, never handwritten table algorithms. Producer execution proves nothing;
only the checked proof terms do. This is not a grade downgrade: W24 remains `proved`.

Concrete whole-source decoding is not a proof obligation. As in the canonical corpus's `Decider`
parity gate, kernel reduction is reserved for a small `decide` core and the abstract
soundness lemmas; every concrete whole-source fact (the complete entry decodes, a duplicate is
rejected, the import edge resolves) is a build-failing `#guard`, which runs the compiled evaluator,
adds no declaration and so pollutes no axiom cone, and fails the build unless it returns `true`.
Kernel reduction of the parser does not terminate in budget and OOMs; `#guard` does not.

### Frozen interfaces

```lean
abbrev SourceByte := Fin 256
abbrev SourceBytes := List SourceByte
structure SourceFile where path : SourceBytes; bytes : SourceBytes
structure SourceBundle where entry : SourceBytes; files : List SourceFile

encodeBytes : SourceBytes -> ByteArray
ByteRep : SourceBytes -> ByteArray -> Prop                 -- extensional; generic view lemmas

ParsesClosure : SourceBundle -> Program -> Prop
DecodeDerivation : SourceBytes -> SourceBytes -> Decode.CertifiedDecoded -> Prop
decodeDerivation_sound :
  DecodeDerivation table std value ->
  Decode.decodeClosure (encodeBytes table) (encodeBytes std) = .ok value
```

`Decode.decodeClosure : ByteArray -> ByteArray -> Result CertifiedDecoded` keeps its type; only its
body is repaired. No tail-recursion rewrite is authorized merely to make evaluation cheaper. The
strengthened binding is

```lean
theorem source_binding :
  ∃ value : Decode.CertifiedDecoded,
    Decode.decodeClosure tableBytes stdBytes = .ok value ∧
    value.val.certificate.Valid ∧
    ExpectedSelectedSyntax value.val.program = true ∧
    ClosedBinding selectedBundle value
```

`ParsesClosure selectedBundle entryProgram` is entry-indexed: its ordered constant names are the
eight table constants, excluding `COORD_CEILING`. `ClosedBinding selectedBundle value` carries that
entry binding, exact table-to-cell and cell-to-std edges with bundle lookups, an
`ImportedCoordDerivation` for the literal `8796093022208`, and
`value = Decode.appendImportedCoord entryDecoded coord`. The closed program has all nine selected
constants. The old premise pair `ExpectedSelectedSyntax value.val.program = true` and
`ParsesClosure selectedBundle value.val.program` is contradictory and MUST NOT remain under any
name. The corrected helper remains only as `source_binding_of_closed_binding`; it is not the
headline or acceptance target.

`selectedBundle` includes table, cell and std under their imported paths. The fallback empty program
is removed; rejection must not select a replacement. A checked constant environment derived from `Program.constants` builds
`Machine.Context`. Trichotomy enters through `wordDenote : Slang.Word -> ZMod (2^64)` and
`cellDenote : Slang.Word × Slang.Word -> Apeiron.Trichotomy.AnyRing.RingCell (ZMod (2^64))`,
licensing reuse of `ringMul_comm`, `ringMul_assoc` and `ringNormForm_mul` only for matching
operations; replacement is not claimed to be ring multiplication.

### Revised barriers

`S/` abbreviates `proof/corpus/Slang/`; `G(stage)` is
`LEAN_NUM_THREADS=1 python3 tools/slang/refine.py --stage stage --controls --song-root ../song`, run
under the process-tree memory limit above (never `ulimit -v`).

| Barrier | Owner / owned files (max three) | Gate | Negative control |
|---|---|---|---|
| W21 repair | Flash: `tools/slang/refine.py`, `tools/slang/controls.json`, `proof/corpus/lakefile.lean` | `refine.py --self-test` | A namespace helper without the required theorem must not satisfy a proof target. |
| W22 precedence | Sol: `S/Decode.lean`, `S/DecodeTest.lean` | `G(decode)` | Reinstate the misplaced conditional precedence; the complete `tableSpread` assignment fixture fails. |
| W24 byte bridge | Sol: `S/Source/Bytes.lean`, `S/Source/BytesAudit.lean` | `lake env lean Slang/Source/BytesAudit.lean` | Alter one encoded byte; the representation proof fails. |
| W22 lexical fidelity | Sol: `S/Source/Lexing.lean`, `S/Source/LexingAudit.lean`, `S/Decode.lean` | `lake env lean Slang/Source/LexingAudit.lean` | Incorrect byte offset after a multibyte character fails; an unknown reachable token is not trivia. |
| W22 grammar fidelity | Sol: `S/Source/Grammar.lean`, `S/Source/GrammarAudit.lean`, `S/Decode.lean` | `lake env lean Slang/Source/GrammarAudit.lean` | A selected-body prefix or incorrect conditional grouping fails. |
| W22 closure fidelity | Sol: `S/Source/Closure.lean`, `S/Source/ClosureAudit.lean`, `S/Decode.lean` | `lake env lean Slang/Source/ClosureAudit.lean` | A conflicting duplicate declaration or missing import edge cannot produce a binding. |
| W24 decoder refinement | Sol: `S/Source/DecodeProof.lean`, `S/Source/DecodeProofAudit.lean`, `S/Decode.lean` | `lake env lean Slang/Source/DecodeProofAudit.lean` | Corrupt an intermediate parser result; the soundness application fails. |
| W24 generation | Flash: `tools/slang/export.py`, `S/Generated/Table.lean`, `S/Generated/TableDerivation.lean` | `export.py --check` plus a proof-producing check mode (MISSING) | Altered candidate syntax with unchanged source/derivation fails kernel checking. |
| W24 integration | Sol: `S/SourceAudit.lean`, `tools/slang/controls.json`, `proof/corpus/lakefile.lean` | `G(source)` | Replacing the unconditional target with the conditional helper MUST fail before the stale-source `TABLE_PHI` and regenerated-byte/stale-derivation controls run. |
| W23 arithmetic link | Sol: `S/Refinement/WordDenotation.lean`, `S/Refinement/WordDenotationAudit.lean` | `lake env lean Slang/Refinement/WordDenotationAudit.lean` | Replace source-word multiplication with addition; the denotation theorem fails. |
| W23 context binding | Sol: `S/Refinement/Context.lean`, `S/Refinement/ContextAudit.lean` | `lake env lean Slang/Refinement/ContextAudit.lean` | A constant environment disagreeing with decoded `TABLE_PHI` fails context binding. |
| Downstream wiring | Flash: `tools/slang/refine.py`, `tools/slang/controls.json` | `refine.py --self-test` | A mutation rejected only by stale identity must not count as a semantic-refinement control. |
| W25-W30 | as the original rows, with witness/audit files imported by their gate target | the original stage gates | the original controls, regenerated after the byte-level change |

### Integration order

W21 repair -> W22 precedence -> byte/lexical/grammar/closure barriers -> decoder refinement ->
generation -> W24 integration -> W23 arithmetic link and context binding -> downstream wiring ->
W25 -> W26 -> W27 -> W28 -> W29 -> W30.

### Barrier evidence

- Byte bridge (`S/Source/Bytes.lean`, `S/Source/BytesAudit.lean`): the frozen
  `SourceByte`/`SourceBytes`/`encodeBytes`/`ByteRep` interface with functionality, canonical-view,
  length and injectivity lemmas, and a literal positive/negative audit fixture. Gate
  `LEAN_NUM_THREADS=1 timeout 900 lake build Slang.Source.BytesAudit` and
  `LEAN_NUM_THREADS=1 timeout 900 lake env lean Slang/Source/BytesAudit.lean` exit 0; every headline
  is inside `{propext, Classical.choice, Quot.sound}`. Control: setting the changed fixture equal to
  the original makes `mutated_rejected` fail (exit 1); the unmodified audit rebuilds green.
- Lexical fidelity (`S/Source/Lexing.lean`, `S/Source/LexingAudit.lean`, `S/Decode.lean`): token
  spans are now byte offsets, computed from UTF-8 widths (`Decode.utf8ByteLength`), and `tokenize`
  returns the lexer's actual consumption instead of `bytes.size`. Gate
  `LEAN_NUM_THREADS=1 timeout 900 lake build Slang.Source.LexingAudit`,
  `LEAN_NUM_THREADS=1 timeout 900 lake env lean Slang/Source/LexingAudit.lean` and the full
  `lake build` exit 0; every headline is inside `{propext, Classical.choice, Quot.sound}`. The
  `decode` stage control still passes (baseline green, mutation red at step 1, restored green).
  Control: a char-count span endpoint makes `multibyte_derivation` and its dependents fail (exit 1);
  the unmodified audit rebuilds green. Next barrier: closure fidelity
  (`S/Source/Closure.lean`, `S/Source/ClosureAudit.lean`).
- Grammar fidelity (`S/Source/Grammar.lean`, `S/Source/GrammarAudit.lean`, `S/Decode.lean`):
  `GrammarDerivation`/`ParsesDecl`/`ParsesExpr` relate the real `Decode.parseFunction`/
  `parseExprFuel`/`decodeTokens` to source bytes, with declaration-span bounds and non-overlap, a
  body consumed through its matching close brace with zero leftover tokens, and the labeled
  conditional fixture grouping as `(0 - k) if k < 0 else k`. Gate
  `LEAN_NUM_THREADS=1 timeout 900 lake build Slang.Source.GrammarAudit`,
  `LEAN_NUM_THREADS=1 timeout 900 lake env lean Slang/Source/GrammarAudit.lean` and the full
  `lake build` (8888 jobs) exit 0; the `decode` stage control passes. Control: substituting the
  incorrectly grouped conditional makes `conditional_derivation` fail (exit 1); the unmodified audit
  rebuilds green. Next barrier: closure fidelity.
- Closure fidelity (`S/Source/Closure.lean`, `S/Source/ClosureAudit.lean`, `S/Decode.lean`):
  `SourceFile`/`SourceBundle`, the textual `importPaths` scanner, and `ParsesClosure` bind an entry
  to its decoded program, a `std` file supplying `COORD_CEILING`, every import edge, and exact
  selected names; `unresolved_import_not_parses` and `duplicate_or_skipped_selected_not_parses` are
  the rejection lemmas. `Decode.scanSelected` now errors on a repeated selected constant while
  keeping `erase` for function declarations, so call expressions still parse. Gate
  `LEAN_NUM_THREADS=1 timeout 900 lake build Slang.Source.ClosureAudit`,
  `LEAN_NUM_THREADS=1 timeout 900 lake env lean Slang/Source/ClosureAudit.lean`, the full
  `lake build` (8890 jobs) and the `decode` stage control exit 0; every headline is inside
  `{propext, Classical.choice, Quot.sound}`. Control: adding the imported file to the open bundle
  makes `missing_cell_rejected` fail (exit 1); the unmodified audit rebuilds green.
- W24 decoder refinement (`S/Source/DecodeProof.lean`, `S/Source/DecodeProofAudit.lean`,
  `S/Decode.lean`): the frozen `DecodeDerivation table std value` is inhabited from a
  `GrammarDerivation` plus an `ImportedCoordDerivation`, and `decodeDerivation_sound` proves it
  implies the real `Decode.decodeClosure (encodeBytes table) (encodeBytes std) = .ok value`;
  `ParsesClosure.decodeDerivation`/`decodeClosure_ok` connect it to the closure layer. `S/Decode.lean`
  changes the coordinate test to a structural `match` so the body reduces for the proof. Gate
  `lake build Slang.Source.DecodeProofAudit`, `lake env lean Slang/Source/DecodeProofAudit.lean`,
  the full `lake build` (8892 jobs) and the `decode` stage control exit 0; every headline is inside
  `{propext, Classical.choice, Quot.sound}`. Concrete complete/corrupt fixtures are `#guard`-gated.
  Control: setting the std fixture to `8796093022207` fails the guard (exit 1); unmodified green.
- W24 generation (`tools/slang/export.py`, `S/Generated/Table.lean`, `S/Generated/TableDerivation.lean`):
  the untrusted renderer now also emits `S/Generated/TableDerivation.lean`, which reads the exported bytes
  back to `SourceBytes` by the compiled evaluator (`toSource`), builds `entryBundle` from the real
  `table`/`cell`/`std` bytes, and gates the concrete facts with `#guard` (`importPaths tableSource`,
  `importPaths cellSource`, `decodeAccepts`, `selectedSourceExact`, the two bundle lookups). Gate
  `python3 tools/slang/export.py --check`, `lake build Slang.Generated.TableDerivation` and the full
  `lake build` (8893 jobs) exit 0. Control: asserting `importPaths tableSource == []` fails (exit 1);
  unmodified green.
- W24 integration (`S/SourceAudit.lean`, `tools/slang/controls.json`, `proof/corpus/lakefile.lean`):
  reopened. The current `source_binding` is conditional on an impossible pair: the closed program's
  nine-name `ExpectedSelectedSyntax` and the entry program's eight-name `ParsesClosure`. Its empty
  axiom cone proves only that vacuous implication. The replacement headline is the unconditional
  existential above, with `ClosedBinding` retaining both indices. `decodedSourceExact` remains a
  compiled guard and supplies no theorem witness. The source stage's command regenerates
  the byte and derivation modules from the (possibly mutated) Song source, then builds; its
  `TABLE_PHI` mutation is red at step 1 with output `decodedSourceExact`. Gate
  `python3 tools/slang/spec_check.py --check`, `python3 tools/slang/export.py --check`,
  `lake build Slang.SourceAudit`, the full `lake build` (8894 jobs) and
  `refine.py --stage source --controls --song-root ../song` previously exited 0 against the
  conditional target; those runs are not source-binding acceptance. Required control: a manifest or
  audit target naming `source_binding_of_closed_binding` instead of `source_binding` is red before
  the stage mutation control.
- W23 arithmetic link (`S/Refinement/WordDenotation.lean`, `S/Refinement/WordDenotationAudit.lean`):
  `wordDenote : Slang.Word → ZMod (2^64)` with `wordDenote_{ofNat,add,sub,mul}`, an honest
  natural-representative `wordDenote_bitAnd` and the spread-mask bound; `cellDenote` into
  `Apeiron.Trichotomy.AnyRing.RingCell`, with `cellDenote_mul`/`cellNormForm_denote` matching
  `ringMul`/`ringNormForm` and `cellMul_comm_denote`/`cellMul_assoc_denote`/
  `cellNormForm_mul_denote` reusing `AnyRing.ringMul_comm`, `ringMul_assoc`, `ringNormForm_mul`.
  Gate `lake build Slang.Refinement.WordDenotationAudit`, `lake env lean …`, the full `lake build`
  (8896 jobs) and `spec_check --check` exit 0; every headline is inside the closed cone. Control:
  making the multiplication theorem conclude addition fails (exit 1); unmodified green.
- W23 context binding (`S/Refinement/Context.lean`, `S/Refinement/ContextAudit.lean`): `denoteExpr`,
  `constantEnvironment` (from `Program.constants`), and `checkedContext` replace the independently
  supplied environment; `constantEnvironment_tablePhi` and `checkedContext_constants` bind the
  environment to the decoded `TABLE_PHI`, and `disagreeing_tablePhi_not_checkedContext`/
  `zeroTablePhiContext_not_checkedContext` reject a disagreeing environment. Gate
  `lake build Slang.Refinement.ContextAudit`, `lake env lean …` and the full `lake build`
  (8898 jobs) exit 0; every headline is inside the closed cone. Control: setting the fixture
  `TABLE_PHI` to 2654435768 fails `constantEnvironment_tablePhi` (exit 1); unmodified green.
- W25 constructor establishment (`S/Table/TableRep.lean`, `S/Table/TableInit.lean`, `S/Table/TableWitness.lean`): the
  six-clause `RepSong` over `Machine.Memory`; `tableSlotSet_execution`, `tableInitGo_execution` by
  strong induction on `cap.val - i.val`, `tableCapacityOk_execution` and `tableInit_execution`, all
  from the decoded shape hypotheses that the W24 `DecodeDerivation`/`ExpectedSelectedSyntax`/
  `ParsesClosure` discharge; `constructor_execution_of_certificate` constructs `ConstructorExecution`
  (call, initialized-empty layout, exact `storeAddresses = expectedInitStores`, outside-frame
  preservation, `(resources trace).stores = 2 * cap.val`) and yields
  `RepSong memory' base cap (ofNat 0) ∅ (fun _ => ofNat 0)`. Concrete executing witness at
  `base = 0`, `cap = 1`; `#guard` binds the exported source to the constructor shapes. Control:
  omitting the value-cell store from `TableSlotSetShape` fails construction (exit 1). Gate
  `lake env lean Slang/Table/TableInit.lean`, `lake env lean Slang/Table/TableWitness.lean`, `lake build`
  (8901 jobs) and `spec_check --check` exit 0; every headline in the closed cone.
- W28 store projection (`S/Machine.lean`): `account`/`isProbe` are public and
  `resources_{arithmeticOps,comparisons,bitwiseOps,loads,stores,calls,probeIterations,maxCallStack}`
  are public projections over the event stream, all `[propext]`, so the `2 * cap.val` store count is
  a proved conjunct rather than a deferred obligation. Control: dropping the store increment from
  `account` fails `resources_stores` (exit 1); `S/MachineAudit.lean` and the full build stay green.
  Remaining W28 work is termination and the other source-level operation bounds.
- Whole-source gate (`#guard`), following the canonical corpus: kernel reduction of
  `Decode.scanSelected`/`decodeTable` does not terminate in budget (24 B exceeded 60 s at 5 GB; 472 B
  OOM-killed `lean` at ~10.3 GB). Kernel `decide` covers a small core only; concrete whole-source
  facts are gated by `#guard`, which runs the compiled evaluator, adds no declaration and no axiom,
  and fails the build unless the result is `true`. The closure audit proves the abstract lemmas and
  small fixtures and gates `decodeAccepts`, `selectedSourceExact` and `importPaths` on the complete
  and duplicated entries. Control: negating `selectedSourceExact completeEntry` fails the build
  (exit 1). Residual: duplicate *function* declarations are still skipped by `scanSelected`
  (function names are erased to keep calls parseable).
- Extraction freshness: `proof/regen.sh --check` run under
  `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0` exits 0 — `extraction is fresh`,
  committed `proof/ApeironWord.lean` byte-identical to regeneration, `proof/` `lake build` 1708 jobs. The
  6 GB scope held; direct regeneration was never run unbounded.

After W22 changes, re-run the decode and machine stages with their controls. Before W25, re-run
decode, machine and source on one source/tool identity. W29 re-runs all stages through
replacement-link and the complete corpus axiom audit.

### Acceptance command

```sh
# cwd: /home/socol/Workspace/apeiron
# Run under a process-tree memory limit (cgroup), not `ulimit -v`.
LEAN_NUM_THREADS=1 timeout 1800 python3 tools/slang/refine.py \
  --stage source --controls --song-root ../song
```

Acceptance requires the unconditional `Slang.SourceAudit.source_binding` theorem, its empty axiom
cone, generated `DecodeTrace.Run` derivations for every recursive call and continuation, full-byte freshness, kernel
rejection of a corrupt derivation, and baseline/red/restored-green outcomes. A timeout or memory
failure is inconclusive, not a passing control. The architecture removes the known global
normalization dependency; tractability is not yet measured or proved.

## Active route and Wave 1

The freeze above governs the original W21-W30 table. `tools/slang/controls.json` is runtime
configuration; where it disagrees it is reconciled to this plan. The active route is: gate repair,
then byte/lexical/grammar/closure fidelity, decoder refinement, proof-producing generation, W24
integration, W23 arithmetic and context binding, downstream wiring, then W25-W30.

Wave 1 repairs the status and gate surface before the source binding proceeds. Each task owns at most
three files and one gate. Current status:

| Task | Owned files | Gate | Rejecting control |
|---|---|---|---|
| Storage and assurance wording | `README.md`, `proof/README.md`, `docs/kernel-boundary.md` | `python3 tools/check-claims.py` | restore a stale "universal `Rep` open" or "size gate absent" claim |
| Programme storage status | `docs/direction.md`, `docs/flash-handoff.md` | `python3 tools/check-claims.py` | promote the physical build gap to closed |
| Corpus identity binding | `tools/slang/spec_check.py`, `tools/joint/jointlib.py`, `tools/joint/jointspec.py`, `tools/joint/profile_check.py` | `python3 tools/slang/spec_check.py --check`; the five `tools/joint/*_check.py --check` | a wrong or unresolvable pin rejects; an unrelated sibling HEAD move does not |
| Exact proof-target binding | `tools/slang/refine.py`, `tools/slang/proof_target.py`, `tools/slang/controls.json`, `tools/slang/refine_selftest.py` | `python3 tools/slang/refine.py --self-test` | a same-named helper, a theorem of `True`, a wrong type, an axiom or a `sorry` target rejects |
| Joint evaluator status | `docs/spec/joint-evaluator.md`, `tools/joint/evaluator_profile_rows.json`, `tools/joint/evaluator_profile_check.py` | `python3 tools/joint/evaluator_profile_check.py --self-test` | delete the residue-terminal versus old-reader distinction |
| Active-plan/parser repair | `docs/plan/slang-source-refinement.md`, `tools/slang/spec_check.py` | `python3 tools/slang/spec_check.py --self-test` | an unresolved reference after a fenced block or inside a multi-backtick span rejects |

Fresh evidence required this wave (not acceptance): `python3 tools/slang/spec_check.py --check`,
`python3 tools/slang/spec_check.py --self-test`, `LEAN_NUM_THREADS=1 python3 tools/slang/refine.py
--self-test`, `LEAN_NUM_THREADS=1 python3 tools/slang/refine.py --stage decode --stage machine
--controls --song-root ../song`, `python3 tools/check-claims.py`, `python3 tools/check-assurance.py`,
and the five `tools/joint/*_check.py --self-test`. The source stage is blocked until
`S/SourceAudit.lean` defines the unconditional headline; the present conditional theorem and source
control do not close it.
