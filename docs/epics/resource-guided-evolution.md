# Resource-guided evolution and cross-model transfer

## Intent

Owner: lead. Status: draft; no synthesis or inference implementation dispatch. Base: `04d206013f70f245dfdaf2619c0b1846d10b9922` measured
CPU gains and the separately checked Rust/Aeneas admission core. Lead documentation ownership and
three-file exception are recorded in gpu-prefill.md. The CPU stack is now merged on main
at 01de89d; historical measured source and current integration identities remain separate. Intent's active review remains owned by thread
01a0f458-538e-7470-ae35-6c7779b474cc; this plan changes no Intent source.

Hybrid-path optimisation also improves dense models without a separate dense optimisation campaign:
the retained five-round medians are 2.17x/2.45x on hybrid pp200/pp512 and 1.58x/1.77x on the listed dense
pp200 workloads. This is a transfer result worth explaining. The research question is which shared
primitive, packing, scheduling and memory costs explain it, and whether that explanation predicts
benefit on another model/shape before another implementation campaign.

Start with one shared primitive cost record and one certified sharing/scheduling decision. Typed
SKI/DAG search is a consumer of that record, not a general synthesizer in this epic. A candidate carries
its observation contract and resource derivation; proof eliminates invalid or dominated candidates
before hardware evaluation. Hardware measurements calibrate the remaining envelopes. The SKI basis
supplies a term representation; sharing, effects, schedule and resource usage remain explicit.

Include Fold's already executing GPU graphs in this same census, rather than treating GPU as a
future backend. Source d497161f32e0d366220f2065ec36e680bd9aed86 already supplies GemvSpec,
row/token-tiled projections, a resident dense prefill stack and hybrid decode recurrence. Review and
digests are in artifacts/gain-integration/fold-gpu-reuse.md; retained performance protocols are in
fold-gpu-receipts.md. The shared principle is reuse: weight loads across tokens, activation loads
across rows, scale/packing work across operators and resident state across dispatches. Prove the
resource change of the actual loop/graph before selecting an implementation candidate.

Fold's typed kernel configuration is an existing finite search space for resource-guided choices.
Use its carrier/layout/mapping/fusion/reduction constraints as the initial synthesis domain, then
map admissible choices into typed SKI/DAG terms with explicit effects and sharing. Algebraic
normalization can merge equivalent computations; hardware cost selects a realization under named
shape/register/storage premises. Do not silently normalize float association. Batch recurrence
may offer a distinct algebraic scheduling seam once its real state/rounding contract is registered.

Alternatives: end-to-end speed alone gives little transferable explanation; primitive operation
counts alone miss memory/dispatch/codegen; an untyped rewrite search can duplicate state updates or
silently change arithmetic. All three fail this epic's intended decision interface.

Completion: one actual production-source transformation has a reachable preservation/resource
certificate, passes admission with source/refusal controls, and its predicted resource change is
compared with matched hardware receipts on both a hybrid and a dense workload. A held-out shape
checks transfer prediction. No universal hardware speed or complete search claim is required.

## Specification

| object | required information |
|---|---|
| Primitive | Named actual Rust body/extraction, exact canonical denotation, admissible domain, outputs/state/refusals and audited refinement |
| Term/graph | Typed application plus explicit shared nodes, effects, storage layout and schedule; no implicit copying/elimination of stateful work |
| Resource semantics | Separate arithmetic, packing/copy bytes, allocations, uploads/readbacks, dispatches/drains, peak-live storage and dependency critical path; derive from actual CPU loops and Fold resident graphs |
| Platform envelope | Named CPU/GPU/tool identity, calibrated latency/bandwidth/dispatch intervals and working-set assumptions; model result separated from measured latency |
| Candidate certificate | Reachable joint witness, universal observation/refusal preservation and strict registered resource improvement, as required by proof/DispatchAdmission.lean |
| Selection | Reject invalid domains, failed preservation, wrong source bindings or incompatible measurements before comparing admissible candidates |
| Evolver boundary | Intent owns oracle/isolation/archive repairs; Apeiron supplies immutable source/proof/cost/receipt identities and explicit applicability/refusal, not a candidate-controlled reward marker |

Comparing two upper latency bounds does not prove one implementation faster. A conditional ratio
requires a candidate upper bound and baseline lower bound under the same platform premises, or a
matched measurement result. Correlation between removed work and speed is tested/measured; causal
attribution needs controlled ablation. Existing five-round receipts establish end-to-end gain, not
per-primitive attribution. Counterfactuals are registered and measured separately.

For algorithm-preserving transformations, outputs/refusals/state are invariant. For model or harness
evolution, the intended observation changes: use separately registered quality/coverage/completion
constraints and resource objectives. Do not force learned-model improvements into a false exact-output
equivalence theorem. A candidate cannot alter its own trusted evaluator or select omitted failures.

## Plan

Each row becomes a ≤3-file, one-gate task with an isolated worktree and Arc before edits. The user's Luna/Sol assignment applies: Luna owns bounded source/refinement/resource work,
Sol reviews Luna artifacts, and the lead integrates. Lead-written boundary changes receive
a different-model independent verdict. Inference implementation remains behind tools/dispatch/admit.py --launch; registry is empty.
Source/refinement/cost tasks establish its prerequisites. Gate budgets include all descendants and
lock waiting, at most 540 s, with standard Lean lane/memory/thread limits.

| order | task / owner | owned paths | focused gate / rejecting control |
|---|---|---|---|
| 1 | Explain transfer; Luna analysis, Sol independent review, `APEIRON-RESOURCE-W1-T1` | artifacts/gain-integration/resource-census.py, resource-census.json, resource-transfer.md | MISSING `timeout 60s python3 artifacts/gain-integration/resource-census.py --check --controls`; create this focused gate first; wrong shape/source binding and omitted packing must reject |
| 2 | Source-cost primitive; Luna bounded proof worker | Select real source root plus extraction/refinement pair, at most 3 | Gate-first Aeneas cost correspondence and reachable success/refusal witnesses; change loop work or refusal and require source-connected red |
| 3 | Sharing/schedule resource algebra; Luna bounded proof worker | At most3 new cost/graph proof files selected after contract review | MISSING named composition/dominance theorem with nonvacuous witness; duplicated effect or omitted packing cost must reject |
| 4 | One production admission contract; lead + Sol independent reviewer | Bound contract/proof/source identities, at most 3 | Actual launcher positive, attributable refusal and restored positive; no synthetic-only acceptance |
| 5 | Transfer/held-out validation; lead | At most3 attribution/measurement files | Matched quiet protocol with controlled ablation on hybrid/dense and held-out shape; no comparator setting or workload omission |
| 6 | Evolver consumption; Intent owner after handoff | No Intent paths allocated here | MISSING jointly approved immutable observation/receipt interface and adversarial candidate control; oracle repairs close in their existing thread |

First deliverable: an exact inventory of shared hot-path work and one tractable source-cost proof
target, not a general synthesizer. Expose the same resource interface to CPU paths and the existing Fold GPU graphs. Proof
checking, universal preservation under registered premises and deterministic rejection are the formal
claims; search completeness, evaluator security, IO authenticity, compiler/device behaviour and
hardware performance each need their own evidence. This epic does not silently widen the admission
core's conditional-observation theorem into an unforgeable hardware guarantee.

### First bounded deliverable

Create the three census artifacts above without changing inference implementation. Read retained
`04d2060` source through `git show` and bind it to `artifacts/track-consolidation/source-manifest.json`
and `archived-ratios.json`; an integrated candidate has a separate digest and is not measured merely
by inheriting that revision's ratios. The existing ratios are against llama.cpp `0324696b8` on
Intel Core i9-14900KF, with measured thread choices 8/16/24/32, not before/after ablation ratios.

GPU census source: /home/socol/Workspace/fold/modelfold/src/simt/superblock.rs,
modelfold/src/simt_nn/spec.rs, simt_nn/batch.rs, simt_nn/deltanet.rs and
bedrock/src/serve/gpu.rs. Record chart/carrier and tile shape, existing per-projection versus resident
schedule, owner/dev_len transitions, last-row versus all-logit mode, upload/readback bytes and queue
drains. A successful CPU fallback is not successful GPU coverage; require an executed-path witness.
Keep the existing dense graph distinct from the additional hybrid batched recurrence contract.
Fold's typed legalize domain is a useful first candidate-domain witness, not a runtime refinement.

Inventory shared production paths `src/engine/step/cols.rs`, `src/engine/kern/matmat.rs`,
`src/engine/kern/gemm.rs` and `src/engine/chart/pool.rs` at that identity. Record columns/rows/block
width, packing and copy bytes, allocation count, pool dispatches, arithmetic work and peak-live
buffers. Derive each count from the actual loop/branch; mark unavailable counters unknown. Pool
waiting is an overlapping measurement, not additive work or a speedup. Include exact fallback and
refusal branches. Test the census on hybrid Qwen3.5-0.8B and dense Qwen2.5-0.5B/Qwen2.5-1.5B Q8_0
pp200; pp512 is a distinct population.

Before further measurements, freeze one prediction for a previously unused shape: the direction
and range of packing/dispatch reduction, its applicability domain and a possible falsifier. This
is a resource prediction (`model`), not a latency theorem. If source or counters cannot support it,
return the smallest missing instrument rather than inventing attribution. Controlled ablation and
held-out timing are separate measurement tasks under an approved protocol.

Select one pure source body whose loop/resource accounting can be extracted in three files. Write
the exact constructor-reachable success/refusal witness and red source-cost gate before allocating
a proof worker. The SIMD/unsafe pool path is currently tested, not established by the generic
word theorem. A new resource instrumentation/extraction seam has separate ownership and controls.

### Exit obligations and consumers

| obligation | grade / current status | closing evidence |
|---|---|---|
| Shared transfer explanation | model + measured / open | source-bound census; retained raw receipt identities; one frozen held-out prediction |
| One source-connected resource derivation | proved / open | real extraction, reachable joint witness, success/refusal and loop-cost correspondence; gate MISSING |
| One exact sharing/schedule choice | proved / open | typed observations include state/effects/refusals; strict model improvement; gate MISSING |
| Actual admission | tested + proved / closed to dispatch | independent contract verdict, production launcher positive, attributable refusal, restoration |
| Transfer validation | measured / open | matched quiet receipts and separately registered ablation/held-out protocol; no dense 2x requirement |
| Shared barrier and trust | open | all AGENTS.md §6 Rust/Lean gates and controls on the same identity; source-size/TCB inventory |

The next epic can consume the first certificate in a bounded typed SKI/DAG rewrite selector.
Intent's Evolver consumes an immutable applicability/observation/resource receipt after its own
oracle/isolation/archive repairs and joint interface review; no Intent file or task is allocated
here. Changing a model or evaluation harness registers changed observations and quality/coverage
constraints separately. The current `ProofDispatch.Admission` requires equality of its fixed
`Output` on its joint reachable/admissible domain and strict `Nat` cost reduction; it cannot admit
changed outputs by relabelling a quality claim. A richer admission schema is a separate task.

Native agents may establish separately scoped source/refinement/resource prerequisites. Every
inference optimization implementation worker, including Luna, must execute its registered worker
argv through `python3 tools/dispatch/admit.py --launch <proposal.json>` in its task worktree.
A native route the launcher cannot execute remains blocked; direct native dispatch is no bypass.
