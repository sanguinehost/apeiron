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
| W25 | Prove constructor execution and `RepSong` establishment | `proof/corpus/Slang/TableRep.lean`; `proof/corpus/Slang/TableInit.lean`; `proof/corpus/Slang/TableWitness.lean` | `cd proof/corpus && lake env lean Slang/TableInit.lean` | Omit the value-cell store in decoded `tableInitGo`; constructor establishment MUST fail. | Requires W24. Stop without one joint valid constructor witness and exact outside-memory preservation. |
| W26 | Prove probe/read correctness and state-aware observation | `proof/corpus/Slang/TableProbe.lean`; `proof/corpus/Slang/TableRead.lean`; `proof/corpus/Slang/ReadWitness.lean` | `cd proof/corpus && lake env lean Slang/TableRead.lean` | Classify raw `-1` as absence without consulting `O`; the present-`-1` witness MUST fail. | Requires W25. Stop unless collision reachability, misses, raw return, and unchanged memory are all covered. |
| W27 | Prove replacement write preservation and exact write admission/refusal | `proof/corpus/Slang/TableWrite.lean`; `proof/corpus/Slang/TableAdmission.lean`; `proof/corpus/Slang/WriteWitness.lean` | `cd proof/corpus && lake env lean Slang/TableWrite.lean` | Make a full table reject a present key or write before an explicit refusal; preservation/admission MUST fail. | Requires W26. Stop without jointly inhabited successful and full-table refusing examples. |
| W28 | Prove termination and separate source-level resource projections | `proof/corpus/Slang/TableTermination.lean`; `proof/corpus/Slang/TableResources.lean`; `proof/corpus/Slang/ResourceWitness.lean` | `cd proof/corpus && lake env lean Slang/TableResources.lean` | Remove `step + 1` or count the spread mask while omitting its multiply; termination/resource gate MUST fail. | Requires W27. Stop if timeout is treated as a verdict or source depth is promoted to a native stack guarantee. |
| W29 | Link replacement execution to canonical `Apeiron.Table.insertWith` under the closed axiom cone | `proof/corpus/Joint/SongTable.lean`; `proof/corpus/AxiomAudit.lean`; `proof/corpus/Joint/SongTableControl.lean` | `cd proof/corpus && lake build` | Change decoded `MERGE_MAP` from replacement to addition; `SongTableControl` or the axiom-audited headline MUST fail. | Requires W24-W28. Stop on any unapproved axiom, missing source dependency, lost raw result/state, or claim beyond replacement. |
| W30 | Specify the unmatched accumulator/default adaptation before proof dispatch | `docs/spec/joint-table-accumulator.md`; `tools/joint/accumulator_profile.json`; `tools/joint/accumulator_profile_check.py` | `python3 tools/joint/accumulator_profile_check.py --check` | Equate fresh Song ACC insertion with canonical addition at an arbitrary nonzero default; profile checker MUST reject. | Requires W29. Stop until fresh/present behavior, absence/default adaptation, admission, state, and resources are explicit. |

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

Source: Astra escalation on the W24 source binding. W24 stays `proved`; the whole-file kernel
normalization is replaced by a proof-producing source translation plus compositional denotational
refinement. Every Lean and `refine.py` run is memory-capped (`ulimit -v 8000000`,
`LEAN_NUM_THREADS=1`).

### Why the original W24 cannot be proved as written

The kernel cannot evaluate the decoder on the real bytes: `ByteArray` is not kernel-computable
(`ByteArray.mk` is `noncomputable`, `"...".toUTF8` is stuck), and `String` operations exceed the
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
| `proof/corpus/Slang/Decode.lean:404-410` | `decodeClosure` receives only table and std bytes; it neither validates the intermediate `cell.slang` import nor the textual import edges. |
| `Certificate.Valid` | Five Boolean equalities set to `true` by the success branch; it does not express grammar or import correctness. |
| `Machine.Context.ofProgram` | Takes its constant environment independently; binding it to the decoded constants is still open. |
| `tools/slang/controls.json` | Wires the source stage to `tools/slang/source_semantics_check.py`, which is absent, and accepts the prefix `Slang.SourceAudit.*` rather than the exact `source_binding`; the probe checks existence, not theorem type or axiom cone. |

### Method

An untrusted producer supplies literal bytes, candidate syntax and local derivations. Kernel-checked
rules validate byte decoding, token boundaries, grammar productions, declaration boundaries and
import resolution. Soundness lemmas relate those derivations to the real `Decode.decodeClosure`, and
per-declaration proofs compose by named lemmas, so no application unfolds the whole mutual parser.
Candidate ASTs are generated, never handwritten table algorithms. Producer execution proves nothing;
only the checked proof terms do. This is not a grade downgrade: W24 remains `proved`.

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
    ParsesClosure selectedBundle value.val.program
```

`selectedBundle` includes table, cell and std. The fallback empty program is removed; rejection must
not select a replacement. A checked constant environment derived from `Program.constants` builds
`Machine.Context`. Trichotomy enters through `wordDenote : Slang.Word -> ZMod (2^64)` and
`cellDenote : Slang.Word × Slang.Word -> Apeiron.Trichotomy.AnyRing.RingCell (ZMod (2^64))`,
licensing reuse of `ringMul_comm`, `ringMul_assoc` and `ringNormForm_mul` only for matching
operations; replacement is not claimed to be ring multiplication.

### Revised barriers

`S/` abbreviates `proof/corpus/Slang/`; `G(stage)` is
`ulimit -v 8000000; LEAN_NUM_THREADS=1 python3 tools/slang/refine.py --stage stage --controls --song-root ../song`.

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
| W24 integration | Sol: `S/SourceAudit.lean`, `tools/slang/controls.json`, `proof/corpus/lakefile.lean` | `G(source)` | The stale-source `TABLE_PHI` control plus regenerated-byte/stale-derivation rejection. |
| W23 arithmetic link | Sol: `S/Refinement/WordDenotation.lean`, `S/Refinement/WordDenotationAudit.lean` | `lake env lean Slang/Refinement/WordDenotationAudit.lean` | Replace source-word multiplication with addition; the denotation theorem fails. |
| W23 context binding | Sol: `S/Refinement/Context.lean`, `S/Refinement/ContextAudit.lean` | `lake env lean Slang/Refinement/ContextAudit.lean` | A constant environment disagreeing with decoded `TABLE_PHI` fails context binding. |
| Downstream wiring | Flash: `tools/slang/refine.py`, `tools/slang/controls.json` | `refine.py --self-test` | A mutation rejected only by stale identity must not count as a semantic-refinement control. |
| W25-W30 | as the original rows, with witness/audit files imported by their gate target | the original stage gates | the original controls, regenerated after the byte-level change |

### Integration order

W21 repair -> W22 precedence -> byte/lexical/grammar/closure barriers -> decoder refinement ->
generation -> W24 integration -> W23 arithmetic link and context binding -> downstream wiring ->
W25 -> W26 -> W27 -> W28 -> W29 -> W30.

After W22 changes, re-run the decode and machine stages with their controls. Before W25, re-run
decode, machine and source on one source/tool identity. W29 re-runs all stages through
replacement-link and the complete corpus axiom audit.

### Acceptance command

```sh
# cwd: /home/socol/Workspace/apeiron
ulimit -v 8000000
LEAN_NUM_THREADS=1 python3 tools/slang/refine.py --stage source --controls --song-root ../song
```

Acceptance requires the strengthened theorem, the checked axiom cone, full-byte freshness, kernel
rejection of a corrupt derivation, and baseline/red/restored-green outcomes. A timeout or memory
failure is inconclusive, not a passing control. The architecture removes the known global
normalization dependency; tractability is not yet measured or proved.
