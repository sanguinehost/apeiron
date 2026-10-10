# Documentation index

This is the index of record for `docs/`. Source and gates win over any document (AGENTS §1).

## Governing

| document | role |
|---|---|
| [`direction.md`](direction.md) | programme authority: decisions, evidence, open work |
| [`kernel-boundary.md`](kernel-boundary.md) | layer ownership and the trust boundary |
| [`plan.md`](plan.md) | execution order, gates, stop points, epic exits |
| [`roadmap.md`](roadmap.md) | staged programme sequence and the prior-art index |
| [`state-ledger.tsv`](state-ledger.tsv) | task ledger |
| [`hash-comparison.md`](hash-comparison.md) | reference note |
| [`../proof/README.md`](../proof/README.md), "Extraction requirements on the source" | what Aeneas needs from the source (absorbed from the retired `docs/extraction-requirements.md`) |

## Epics (`epics/`) — one file per epic

Each epic file carries its intent, specification and plan as consecutive sections. New epics start
from [`templates/epic.md`](templates/epic.md).

| epic | status (as recorded in the file) |
|---|---|
| [agent-throughput-segmentation](epics/agent-throughput-segmentation.md) | proposed operating plan |
| [ai-track-reanchor](epics/ai-track-reanchor.md) | draft |
| [algebra-native-addressing](epics/algebra-native-addressing.md) | approved for founding |
| [allocator-model](epics/allocator-model.md) | proposed architecture (`anchor`) |
| [apeiron-intent-loop](epics/apeiron-intent-loop.md) | transition 4 closed: extraction freshness open; `tools/dispatch/contracts.json` registers no target |
| [assurance-factoring](epics/assurance-factoring.md) | implementation under factoring evidence |
| [build-cost-optimization](epics/build-cost-optimization.md) | plan |
| [book](epics/book.md) | complete |
| [carrier-refinement](epics/carrier-refinement.md) | complete |
| [combinator-links](epics/combinator-links.md) | in progress |
| [corpus-release-pin](epics/corpus-release-pin.md) | approved |
| [engine-throughput](epics/engine-throughput.md) | W2 tested (Q8Act 40/40 parity); T6B running |
| [exact-scan-contract](epics/exact-scan-contract.md) | draft; chunked exact accumulation (DeltaNet state + online softmax), prefill ratio target |
| [decode-dominance](epics/decode-dominance.md) | draft; beat llama.cpp via batch-invariance, exact speculative greedy, zero skip, traffic statements |
| [structural-decode-search](epics/structural-decode-search.md) | prefill priority; direct norm candidate tested with source controls; five-round timing blocked by host quietness, integration open |
| [frontier-consumer](epics/frontier-consumer.md) | draft; cross-track roadmap (product ladder R0–R5) |
| [gate-driver](epics/gate-driver.md) | plan only: replace 33 `proof/resource/*_gate.py` with one `tools/gate` binary; no status line in the file |
| [gpu-spv](epics/gpu-spv.md) | draft |
| [icarus-sha256-rooting](epics/icarus-sha256-rooting.md) | active; ports the campaign's 199 bound SHA-256/256d rooting results into a new `Icarus` corpus library |
| [inference-parity](epics/inference-parity.md) | draft for decision |
| [kani-retirement](epics/kani-retirement.md) | retired by user direction |
| [kernel-topology-optimizations](epics/kernel-topology-optimizations.md) | draft |
| [moe-router](epics/moe-router.md) | W1 complete (tested + proved); W2 at barrier |
| [model-algebra](epics/model-algebra.md) | draft; closing wave receipted |
| [neural-rebase](epics/neural-rebase.md) | draft (stage 2 vendoring complete; [stage 3 learning track](epics/neural-rebase.md#stage-3--learning-track-intent-spec-plan) intent, spec, plan) |
| [open-areas](epics/open-areas.md) | plan |
| [parallel-stable-scatter](epics/parallel-stable-scatter.md) | approved for bounded implementation |
| [physical-computation-bounds](epics/physical-computation-bounds.md) | draft |
| [policy-engine](epics/policy-engine.md) | W0–W2 closed (core and analysis proved, extraction fresh, oracle tested); W3 refinement next |
| [read-path-specialization](epics/read-path-specialization.md) | extraction landed; universal read open |
| [read-proof](epics/read-proof.md) | proposed |
| [resource-security-continuation](epics/resource-security-continuation.md) | plan |
| [security-algebra-rebase](epics/security-algebra-rebase.md) | attribution review active |
| [spinor](epics/spinor.md) | partly proved at the canonical gate |
| [storage-refinement](epics/storage-refinement.md) | approved; coordinate-keyed counting-container contract recorded (`anchor`, `APEIRON-COUNT-W1-T1`) for the counting face, generic growable `Table` retained as the map face; implementation blocked until the contract names an executable gate and its rejecting control |
| [subquadratic-decode](epics/subquadratic-decode.md) | in progress |
| [qwen35-hybrid](epics/qwen35-hybrid.md) | exact path = llama.cpp ids; speed next |
| [one-executor](epics/one-executor.md) | S1–S3 landed: Qwen2.5, Bonsai, Qwen3.5 on one executor |
| [ternary-hadamard](epics/ternary-hadamard.md) | W1–W4 proved/tested; W5 measurement open; Part II (one-bit faces) W6–W7 open; Part III (one-bit attention on the hyperbolic face, `LEARN-HATTN`) open |
| [unified-quant-kernel](epics/unified-quant-kernel.md) | approved: quantisation level as one parameter across GPU, CPU and proof; CPU chart face dispatched (W2c) |

The policy-engine epic's prior-art study is [`study/policy-engines.md`](study/policy-engines.md).

## Counting face — non-claims

The coordinate-keyed counting face ([contract](epics/storage-refinement.md#coordinate-keyed-counting-container-contract),
[boundary](kernel-boundary.md#counting-face-representation-boundary-apeiron-reorient),
[supersession](direction.md#five-barrier-execution-sequence-apeiron-reorient)) is a recorded
architecture decision at grade `anchor` and carries no result:

- no source, extraction, proof, test or measurement result; G1/REG-1 stays **OPEN**,
  `report_complete=false`;
- the generic growable `Table` is retained unchanged as the map face, and its `TableRefinement` and
  `Growable/*` headlines discharge no counting-face obligation;
- **MISSING** and therefore blocking: the codec name with its sentinel and boundary values, the
  default value, the count carrier and the overflow policy, the real production consumers, the
  reachable joint witnesses, and the executable gate with its rejecting control;
- the coordinate/codec injectivity law has no canonical link; HashDOS resistance is required, not
  established;
- the supersession covers the one-`Table` restriction and the "a new container" non-goal **for the
  counting face only**; every other non-goal, obligation and acceptance criterion stands.

## Gate-read contracts (`contracts/`) — moved verbatim, content unchanged

These files are parsed by gates. They are not merged or reformatted:

| file | read by |
|---|---|
| `kepler-canonical-{intent,spec,plan}.md` | `tools/check-kepler.py` |
| `joint-{primitives,table,evaluator,carrier-exchange}.md` | `tools/joint/{profile,table_profile,evaluator_profile,carrier_profile}_check.py` |
| `joint-table-accumulator.md` | no checker reads it; its content is carried by `joint-primitives.md` R1 and `iota-source-entry.md` SE3, and its obligation that no accumulator correspondence exists stays open |
| `iota-source-entry.md` | `tools/joint/source_entry_check.py` |
| `iota-source-semantics.md`, `iota-source-refinement-plan.md` | `tools/iota/spec_check.py`, `refine.py` |

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

## Next inference epics (draft)

- [GPU prefill from verified integer primitives](epics/gpu-prefill.md): resident-buffer execution and the existing SPIR-V refinement boundary.
- [Performance theory: the cost model and the shape of fast](epics/resource-guided-evolution.md): a move ledger, one resource-vector cost model in Lean with predictions checked against measurement, and the earlier resource-guided evolution record (PERF-THEORY).
