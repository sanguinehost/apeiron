# Documentation index

This is the index of record for `docs/`. Source and gates win over any document (AGENTS §1).

## Governing

| document | role |
|---|---|
| [`direction.md`](direction.md) | programme authority: decisions, evidence, open work |
| [`kernel-boundary.md`](kernel-boundary.md) | layer ownership and the trust boundary |
| [`flash-handoff.md`](flash-handoff.md) | execution order, gates, stop points, epic exits |
| [`roadmap.md`](roadmap.md) | staged programme sequence and the prior-art index |
| [`state-ledger.tsv`](state-ledger.tsv) | task ledger |
| [`extraction-requirements.md`](extraction-requirements.md), [`hash-comparison.md`](hash-comparison.md) | reference notes |

## Epics (`epics/`) — one file per epic

Each epic file carries its intent, specification and plan as consecutive sections. New epics start
from [`templates/epic.md`](templates/epic.md).

| epic | status (as recorded in the file) |
|---|---|
| [agent-throughput-segmentation](epics/agent-throughput-segmentation.md) | proposed operating plan |
| [ai-track-reanchor](epics/ai-track-reanchor.md) | draft |
| [algebra-native-addressing](epics/algebra-native-addressing.md) | approved for founding |
| [allocator-model](epics/allocator-model.md) | proposed architecture (`anchor`) |
| [assurance-factoring](epics/assurance-factoring.md) | implementation under factoring evidence |
| [build-cost-optimization](epics/build-cost-optimization.md) | plan |
| [carrier-refinement](epics/carrier-refinement.md) | complete |
| [combinator-links](epics/combinator-links.md) | in progress |
| [corpus-release-pin](epics/corpus-release-pin.md) | approved |
| [engine-throughput](epics/engine-throughput.md) | W2 tested (Q8Act 40/40 parity); T6B running |
| [decode-dominance](epics/decode-dominance.md) | draft; beat llama.cpp via batch-invariance, exact speculative greedy, zero skip, traffic statements |
| [frontier-consumer](epics/frontier-consumer.md) | draft; cross-track roadmap (product ladder R0–R5) |
| [gpu-spv](epics/gpu-spv.md) | draft |
| [inference-parity](epics/inference-parity.md) | draft for decision |
| [kani-retirement](epics/kani-retirement.md) | retired by user direction |
| [kernel-topology-optimizations](epics/kernel-topology-optimizations.md) | draft |
| [moe-router](epics/moe-router.md) | W1 complete (tested + proved); W2 at barrier |
| [model-algebra](epics/model-algebra.md) | draft; closing wave receipted |
| [neural-rebase](epics/neural-rebase.md) | draft |
| [open-areas](epics/open-areas.md) | plan |
| [parallel-stable-scatter](epics/parallel-stable-scatter.md) | approved for bounded implementation |
| [physical-computation-bounds](epics/physical-computation-bounds.md) | draft |
| [policy-engine](epics/policy-engine.md) | W0–W2 closed (core and analysis proved, extraction fresh, oracle tested); W3 refinement next |
| [read-path-specialization](epics/read-path-specialization.md) | extraction landed; universal read open |
| [read-proof](epics/read-proof.md) | proposed |
| [resource-security-continuation](epics/resource-security-continuation.md) | plan |
| [security-algebra-rebase](epics/security-algebra-rebase.md) | attribution review active |
| [spinor](epics/spinor.md) | partly proved at the canonical gate |
| [storage-refinement](epics/storage-refinement.md) | approved |
| [subquadratic-decode](epics/subquadratic-decode.md) | in progress |
| [qwen35-hybrid](epics/qwen35-hybrid.md) | exact path = llama.cpp ids; speed next |
| [one-executor](epics/one-executor.md) | S1–S3 landed: Qwen2.5, Bonsai, Qwen3.5 on one executor |
| [ternary-hadamard](epics/ternary-hadamard.md) | W1–W4 proved/tested; W5 measurement open |

The policy-engine epic's prior-art study is [`study/policy-engines.md`](study/policy-engines.md).

## Gate-read contracts (`contracts/`) — moved verbatim, content unchanged

These files are parsed by gates. They are not merged or reformatted:

| file | read by |
|---|---|
| `kepler-canonical-{intent,spec,plan}.md` | `tools/check-kepler.py` |
| `joint-{primitives,table,evaluator,carrier-exchange,table-accumulator}.md` | `tools/joint/*_check.py` |
| `slang-source-entry.md` | `tools/joint/source_entry_check.py` |
| `slang-source-semantics.md`, `slang-source-refinement-plan.md` | `tools/slang/spec_check.py`, `refine.py` |

## Evidence (`evidence/`)

Receipts, one per wave or finding. The roll-ups are:
- [`perf-ledger.md`](evidence/perf-ledger.md): every `perf-*` receipt;
- [`security-spikes.md`](evidence/security-spikes.md): the security spike stubs;
- [`anative.md`](evidence/anative.md): the algebra-native addressing receipts.

## Archive (`archive/`)

Historical handoffs and run logs, moved verbatim: the OpenAI handoff and continuation, and the
autonomous run log.

## Moved paths

The full old → new map (102 paths) is in `docs/moved.tsv`. Merged epic files name their source files
on their second line.
