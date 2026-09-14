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
