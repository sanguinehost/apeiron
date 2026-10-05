# Assurance factoring evidence

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

All seven original size breaches pass the unchanged assurance checker. All eleven generated-table
shard gates and controls pass; extraction refresh and integrated proof gates are recorded below. Limits and owned populations
are unchanged. Base is `262bb0a94eff2cc7d5655122a954f44d74de96b6` plus accepted dirty source.
No commit, canonical/Song pin change or Kani run is part of this work.

## Changes and controls

| Original breach | Factoring and evidence |
|---|---|
| `examples/reg1/acceptance.rs`, 1799 lines | Report and registry-test child modules; root1340 lines at factoring,1356 after concurrent instrumentation. REG-1 tests14/14; real threshold mutation red101, restoration0. |
| `src/table.rs`, 1678 lines | Exact prepared-helper text in `src/table/prepared.rs`; root1478 lines. Extraction digest list includes the child. Real merge-order mutant red101, restoration0. |
| `tools/slang/export.py`, 2299 lines | Decoder/scanner/output-family modules; root969 lines. Decoder44 and scanner12 function/class ASTs preserved with explicit source-provider substitution. Complete freshness rejects real leaf/constructor changes. |
| `tools/slang/spec_check.py`, 1562 lines | Frozen constants in `spec_constants.py`; root1329 lines.41 constants and176 mutation controls preserved. Separate reviewed stale-revision correction changes no corpus pin or module digest;176+5 controls pass. |
| `tests/`, 15 immediate files | Three byte-identical moves to Cargo directory entry points; count12. Target discovery unchanged; individual expectation mutants red101/restored0. All-feature inventories16/24/13 remain. |
| `Slang/Source/`, 14 immediate files | Byte and lexing audits in `Source/Audit`; count12. Names/types unchanged, obsolete shims removed and Lake roots updated. Lexing recursion override removed; default guard passes. Bytes17 and lexing23 empty-axiom audits plus mutation/restoration and17 rejection controls pass. |
| `Slang/Generated/Table.lean`, 14414 lines | Eleven serial declaration-shard barriers pass; root16 lines, eleven shards438–1409 lines. |

Generated source is frozen by Song `aeaf3baac434b2e3c01154d275bea65fdadc69bf` and original
monolith SHA256 `369998b2d3e219d4ada7e0b86d0d56ef758dc94026a7c52b07647fcbbcfbc226`.
Each shard task compares the entire rendered monolith and all three source JSON objects with the
accepted snapshot before writing. Prior shard bytes remain unchanged. Mutations of a real leaf
byte and complete declaration order reject in disposable copies; restoration passes. Every stage
builds `Slang.Generated.Table`. No public declaration term or source byte is changed.

The registered Table import closure now scans all reached shards for forbidden declarations.
Missing/cyclic/escaped imports and unused forbidden declarations reject; shared ancestors pass.
All registered declaration names/types remain. Output-map and closure tests pass20/20; actual
tokenizer-witness audit and controls pass before and after sharding.

## Integrated acceptance

| Gate | Result |
|---|---|
| Full Rust tests | Exit0;263 passed,1 ignored;555.752s. |
| Clippy, all workspace/features/targets, warnings denied | Exit0;2.997s. |
| Cargo format check | Exit0;0.170s. |
| Claim checker and8 rejecting controls | Exit0. |
| Assurance6 rejecting controls | Exit0; final size check0: local trusted checking3279/5000 lines, all owned files≤1500, all immediate directories≤12. |
| Spec checker controls | Exit0;176 requirement mutations plus5 headline fixtures. |
| Export/audit Python controls | Exit0;20 tests. |
| Extraction freshness and extraction proof package | Exit0;1708 jobs; exact normalized declaration inventory preserved. |
| Final bridge package | Exit0;3493 jobs;135.127s. |
| Final tokenizer/parsing source witnesses and controls | Both exit0;12.839s/12.679s. Baseline green, mutation red, restored green and17 rejection controls each. |

Rust receipt `target/open-areas/final-rust/` binds77 recorded Rust source files. Concurrent work later
changed `examples/reg1/{acceptance,performance}.rs`; the earlier full Rust receipt does not cover
those later edits. They are preserved, and this task does not claim a current whole-programme barrier. Focused task
receipts are retained under `target/open-areas/APEIRON-*`; the lead scripts, manifests, exact
commands, elapsed times and control logs are local ignored evidence. Earlier declared900-second
limits remain historical; remaining invocations use the current540-second whole-process-tree cap.
The [full performance attempt](open-area-performance.md) failed its600-second ceiling; no current
ratio result follows from it.

## Proof and programme boundary

`GrowableScan.publish_live_implies_summary_maybe` is a new checked universal local fact: successful
generated live publication makes the same-place summary query true. Its generated-body control
omits the summary write, still publishes successfully, returns false on that query and breaks
the original proof/guard; restoration passes. This is distinct from source factoring.

Universal public read, reachable representation construction/preservation, insertion
`SearchDecisionCorrect`, actual allocator effects/refusal, whole W7 composition, G1 and
cross-repository correspondence remain open. Factoring proves no additional runtime behavior
and does not cover codegen, ABI, unsafe-container correctness or deployed cryptography.

## Final identity documentation

```text
SCOPE: this receipt, docs/archive/handoff-openai.md, proof/README.md; lead supplied-checkout documentation exception.
REGISTER: final identities and scoped acceptance, no new proof claim.
CORPUS: 589 generated chunks text-identical modulo order and Source lines; Sol verdict recorded.
REFERENCE: target/open-areas/extraction-order-verdict.md and APEIRON-SIZE-W20-T1/receipt.json.
GATE: extraction freshness0, proof build0; final claim/diff checks.
GRADE: accepted factoring and original proof scope; whole-programme obligations remain open.
```

Extraction refresh changes only declaration order and source locations:589 exact normalized
chunks, zero added or removed. Freshness and the1708-job extraction proof build pass. Generated
SHA256 is `5599fe7d22685de05fa8349b727f852ebf5cb70308a1bb67b95f21128d3e9d40`.
A later locator fix supersedes this generated identity; the current generated sha and its gates are in
[locator and attribution evidence](locator-reserve-quadratic-and-reg1-attribution.md).
The prepared-source child is included in `regen.sh`'s mandatory source-digest population.

Final proof verification holds one global lease with an outer540-second cap: bridge135.127s,
tokenizer12.839s, parsing12.679s, all exit0. Logs/results: `target/open-areas/final-proof/`.
Claim checker and diff check pass after documentation updates. Task worktrees/branches are removed
after integration or preserved candidate handback; `main` and unrelated `one-allocation` remain.
No commits or pushes occur. Next bounded tasks are the exact unintegrated allocator probe goals
and security witness gate/control, followed by the remaining read/classifier producer obligations;
performance needs bounded phase attribution before another full acceptance campaign.
