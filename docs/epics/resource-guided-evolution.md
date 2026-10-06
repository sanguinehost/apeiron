# Performance theory: the cost model and the shape of fast

Row: `perf-theory-formal-cost-HiI4vx6ezYPSHkfGWbq7YX` (PERF-THEORY). Absorbs the earlier resource-guided-evolution epic, retained verbatim after the `---` separator under "Evolution record".

## Intent

**Problem.** The gains so far were found by search. Each next target (a loss against CUDA, a loss at one head dimension, a map column) is rediscovered by trial. No artifact states, for a kernel, which resource binds, what the lower bound is for its denotation, and which denotation-preserving rewrite moves it. A refuted lever list exists (prefetch of `vals`, TAGPLANE, MAP-GLANCE, INS levers A and B, launch-path megakernel) and no theory says why each failed.

**Owner direction (2026-10-07).** Beat CUDA in every area (pp and tg, every model and size), and make the theory of what makes a kernel fast the primary deliverable, so the result transfers to any device, model or old code.

**Thesis (grade `anchor` until W3 passes).** A kernel is a denotation `f` plus a schedule. Its time satisfies `t(f) ≥ max_r d_r(f) / c_r` over resources `r` (bytes at each memory level, instructions by class, serial chain depth, launches, cache-line touches), with capacities `c_r` measured per machine. A *move* is a rewrite `f → g` with `⟦f⟧ = ⟦g⟧`. Its speedup is bounded by the demand ratio on the binding resource, `t(f)/t(g) ≤ max_r d_r(f)/d_r(g)` when `r` binds, and it is zero when `r` does not bind. The shape (topology) is the reuse structure of the dataflow: use count and fanout of each value, number of walks over each operand, cache lines touched per operation, instructions per element. A kernel is fast when the demand on its binding resource is at the lower bound the denotation permits.

**Evidence the thesis must explain (measured, `docs/plan.md` rows and receipts):**

| move | gain | binding resource it relieves |
|---|---|---|
| int8 cooperative-matrix GEMM over DP4A (`a93dc44`) | 2.0–2.8× pp512, every carrier and model run | instructions per MAC (matrix-unit width vs 4-way dot) |
| one-walk online softmax and heaviest-first tiles (`1f0d2ad`) | attention node 2214 → 1160 µs; 1.09–1.28× pp512 at head_dim 64; none at head_dim 128 (tile refused: shared memory 98304 B > 49152 B) | K/V walks (3 → 1) and shared-memory residency |
| CPU chart: raw-byte AVX2 cores | 8.4 → 35 → 78 tok/s decode | bytes and instructions per weight (2.0 instr/weight) |
| MAP-L3 hot/cold probe split; COUNT single `k·PHI` read path | measured, `docs/plan.md` | cache lines touched per probe |
| refuted: prefetch `vals[home]` | none outside spread (MAP-FOOTPRINT) | not binding: the term it targets is not the one that binds |

The map face is the control for the thesis: ap executes fewer instructions than hashbrown+Fx yet is 1.5–2.4× slower at equal slots and load, with 4.1× the LLC references. A model that counts instructions only predicts the opposite sign; the line-touch term must carry it.

**Rejected alternatives.**
- Per-kernel tuning: no transfer, and the refuted list shows it does not predict its own failures.
- Launch cost as the bottleneck: falsified by the lineage itself (`sanguine/proof/Closures/Engineering/FusionCost.lean` header: a decode step is ~925 µs active kernel work, ~16% CPU submit).
- Instruction counts as the model: contradicted by the map face above.
- A cost model stated in prose: no gate reads it (AGENTS.md §1a).

**Completion condition.**
1. `docs/evidence/perf-move-ledger.md`: every measured move, with receipt path, before/after, spread, binding resource, location (layer, kernel, shape) and licensing identity or `none`; refuted moves included.
2. `proof/corpus/Perf/`: one resource carrier and machine record, the time lower bound, the move preorder, and one theorem per move class in the ledger (fusion on a use-count-1 edge, tiling reuse, instruction width `dpw = 8/b`, walk count, line count, hot/cold split), under the closed axiom cone.
3. A predictions table written before the measurement is read, for held-out cases (head_dim 128 attention, the one `b`-parameterised prefill GEMM, a map layout change): predicted ratio versus measured, per case. A miss outside the measured spread refutes or refines the row, and the row says which.
4. The model names the binding resource and the next move for each red cell: CUDA decode (0.88×), qwen2.5-1.5b pp512 (0.65× Vulkan), map CoordMap columns.
5. Cost counted over the real Aeneas-extracted bodies (pattern of `proof/corpus/Growable/CostModel.lean`) and over the emitted SPIR-V module structure (pattern of `Dispatch.dpw` and `emittedAccessChains`), for one family beyond the map.

**Non-goals.** A Lean cost proof is not a hardware speed measurement (AGENTS.md §9.1). No device microarchitecture beyond named, measured capacity parameters. No kernel change in this epic except through its own rows. No claim of CUDA parity from the model.

**Trust boundary.** Capacities are measured parameters, not theorems. rustc/LLVM codegen, the Vulkan and CUDA drivers and compilers, clock and power behaviour, and cache replacement policy are outside every theorem here. The lineage files are `model` or `anchor` until ported and re-checked.

**Open questions.** Whether one carrier covers CPU and GPU levels or needs a per-device instance. Whether latency-bound chains need an explicit critical-path term (`FusionHidesLatency`) or fold into the max. Whether occupancy is a resource or a derived capacity.

## Specification

**Carrier.** `Res` is a vector of counted events, extending `Growable.Cost` (`proof/corpus/Growable/CostModel.lean:62`): bytes moved per memory level, instructions per class (FMA, DP4A, MMA, ALU), serial chain depth, launches, distinct 64-byte lines touched. Each field names its measure (AGENTS.md §8). `Machine` holds a capacity per field, as measured parameters.

**Bound.** `time_lb m r = max_i (r.field i / m.cap i)`, with chain depth times latency as a separate additive term.

**Move.** `Move f g` carries `⟦f⟧ = ⟦g⟧` and the demand relation `r_g ≤ r_f` on the relieved fields, with any increase on other fields stated. Theorem shape: `speedup_le_binding_ratio`.

**Lineage port targets (search recorded; each is ported, not re-derived):**

| target | file | content |
|---|---|---|
| fusion legal on a use-count-1 edge, latency `floor·#segments` | `../sanguine/proof/Closures/Engineering/FusionCost.lean` (sanguine HEAD `e830d3fb`) | `seg_denotation_invariant`, `fusion_reduces_latency`, `fanout_forces_cost` |
| lone reduction latency-bound, absorbed into a transport-bound gemv | `../sanguine/proof/Closures/Engineering/FusionHidesLatency.lean` | `lone_reduction_is_latency_bound`, `fusion_hides_the_coreduction`, `MemModel` |
| also read in W1: `PartitionBandwidth`, `MemoryLevelParallelism`, `OccupancySweep`, `MegakernelOccupancy`, `InstructionEnergy`, `TernaryDP4A`, `SynchronizationCost`, `RecomputeThesis`, `WarpSpecialization`, `ConcreteProbeCost` | same directory | each to be classified port, refute-by-measurement, or out of scope |
| in-tree anchors | `proof/corpus/Growable/CostModel.lean`; `proof/corpus/Spirv/Dp4a.lean` (`Dispatch.dpw`); `proof/corpus/Spirv/Carrier.lean` (b-indexed split, `f544f26`) | cost over extracted bodies; ops per instruction |

Lean sources of the lineage contain `Real` and Mathlib imports; the port decides per theorem whether the carrier is `Nat` (counted events) or `Real` (rates), and records it.

## Plan

| wave | task | files | gate | control |
|---|---|---|---|---|
| W0 | PERF-LEDGER: move ledger from the receipts and the plan rows | `docs/evidence/perf-move-ledger.md` | `python3 tools/check-claims.py` (every row names a resolving receipt) | a row with an unresolvable receipt path fails the checker |
| W1 | PERF-PORT: classify the lineage files; port `FusionCost`/`FusionHidesLatency` into `Perf/Roofline.lean` over the carrier | `proof/corpus/Perf/Roofline.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build` | drop the use-count-1 side condition: `fanout_forces_cost` analogue goes red |
| W2 | PERF-INSTANCES: one theorem per move class; one task each (instruction width from `Carrier.lean`; walk count; line count from `Growable/CostModel.lean`; hot/cold split) | `proof/corpus/Perf/*.lean` | `cd proof/corpus && lake build` | change the field the move claims to relieve: red |
| W3 | PERF-PREDICT: predictions table before measurement; ATTN-HD128, the `b`-parameterised prefill GEMM and a map layout change are the held-out cases | `docs/evidence/perf-predictions.md` | the same command that measures each case (paired, 5 rounds) | a prediction recorded after the measurement is rejected by commit order |
| W4 | PERF-EXTRACT: cost over extracted bodies and emitted SPIR-V module structure for one GPU family | `proof/corpus/Perf/`, `proof/extract/lib.rs` | `AENEAS_HOME=/home/socol/oc-tmp/aeneas-pinned proof/regen.sh --check` and `cd proof/corpus && lake build` | mutate the emitter's shift constant: the counted field changes and the theorem goes red |
| W5 | PERF-RED-CELLS: apply the model to CUDA decode, 1.5B pp512, map columns; one dispatch per named next move | per dispatched row | per row | per row |

Stop and rollback: three failed approaches on one theorem or five patch-and-rebuild cycles stops the lane. A refuted prediction is recorded and refines the model; it does not close the row.

---

## Evolution record (earlier epic: resource-guided evolution and cross-model transfer)

## Resource-guided evolution and cross-model transfer

### Intent

Owner: lead. Status: source-bound census checked and scalar source prerequisite proved; no synthesis or inference implementation dispatch. Base: `04d206013f70f245dfdaf2619c0b1846d10b9922` measured
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

Use Fold's GPU graphs as read-only source references in this census; Apeiron owns the new executing
GPU kernels and runtime. Source d497161f32e0d366220f2065ec36e680bd9aed86 already supplies GemvSpec,
row/token-tiled projections, a resident dense prefill stack and hybrid decode recurrence. Review and
digests are in artifacts/gain-integration/fold-gpu-reuse.md; retained performance protocols are in
fold-gpu-receipts.md. The shared principle is reuse: weight loads across tokens, activation loads
across rows, scale/packing work across operators and resident state across dispatches. Prove the
resource change of the actual loop/graph before selecting an implementation candidate.

Fold's typed kernel configuration is an existing finite search space for resource-guided choices.
Use its carrier/layout/mapping/fusion/reduction constraints as the initial synthesis domain, then
derive Apeiron-native admissible choices and map them into typed SKI/DAG terms with explicit effects and sharing. Algebraic
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

### Specification

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

### Plan

Each row becomes a ≤3-file, one-gate task with an isolated worktree and Arc before edits. The user's Luna/Sol assignment applies: Luna owns bounded source/refinement/resource work,
Sol reviews Luna artifacts, and the lead integrates. Lead-written boundary changes receive
a different-model independent verdict. Inference implementation remains behind tools/dispatch/admit.py --launch; registry is empty.
Source/refinement/cost tasks establish its prerequisites. Gate budgets include all descendants and
lock waiting, at most 540 s, with standard Lean lane/memory/thread limits.

| order | task / owner | owned paths | focused gate / rejecting control |
|---|---|---|---|
| 1 | Explain transfer; Luna analysis, Sol independent review, `APEIRON-RESOURCE-W1-T1` | artifacts/gain-integration/resource-census.py, resource-census.json, resource-transfer.md | `timeout 60s python3 artifacts/gain-integration/resource-census.py --check --controls`; checked model gate; wrong shape/source binding and omitted packing must reject |
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

#### First bounded deliverable

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

#### Exit obligations and consumers

| obligation | grade / current status | closing evidence |
|---|---|---|
| Shared transfer explanation | model + measured / open | source-bound census; retained raw receipt identities; one frozen held-out prediction |
| One source-connected resource derivation | proved / scalar prerequisite accepted; transformation certificate open | ActScaleResource loop/scale source refinement; gate.py --check --controls; projection/graph cost correspondence remains open |
| One exact sharing/schedule choice | model (algebra proved): `proof/resource/CostAlgebra.lean` composition + fieldwise dominance, `dominates_improves` (strict `Nat` reduction), nonvacuous `residentPlan` < `perProjectionPlan` witness; three controls red→restored; `python3 proof/resource/CostAlgebraGate.py --check --controls` exit 0 in 10.2 s. Source-derived candidate counts, the universal `Admission` premises and the production contract stay open | typed observations include state/effects/refusals; strict model improvement; gate present at model scope |
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

#### Executing prerequisite allocation

`APEIRON-RESOURCE-W1-T1`: Sol owns exactly resource-census.py, resource-census.json and
resource-transfer.md under artifacts/gain-integration in /tmp/worktree-agent-APEIRON-RESOURCE-W1-T1,
branch codex/resource-census-w1, base ad707f4. No shared build lane. Slice budget 480 s; local gate
budget 60 s, including controls/descendants. Baseline: resource-census.py absent, test -f exits 1.
Census binds actual CPU/Fold source identities, separates derived counts from unavailable counters,
and rejects wrong source/shape and omitted packing. Luna independently reviews the return.
The explicit owner selection for this allocation uses the user's authorised Sol lane.

`APEIRON-RESOURCE-W1-T2`: independent scalar source prerequisite alongside the census, base ad707f4,
/tmp/worktree-agent-APEIRON-RESOURCE-W1-T2, branch codex/actscale-source-w1. Lead owns
proof/resource/gate.py and ActScaleSource.lean; Sol owns ActScaleResource.lean. Their three-path
union extracts the actual src/engine/kern/pow2.rs::act_scale body through a forwarding #[path]
wrapper; no inference source is changed. Gate: fresh extraction, audited Lean source/loop theorem,
real doubling mutation and restoration via gate.py --check --controls, 540 s whole invocation,
flock lane, 12 GiB/no swap and LEAN_NUM_THREADS=1. Baseline: ActScaleResource.lean absent.
Luna independently reviews after the kernel/control return. The resource observation is loop
iterations/rank under named scalar premises, not cycles or whole GPU cost. All downstream
implementation registration waits for the census/contract/refinement barrier; this independent
source prerequisite changes neither canonical definitions nor carrier semantics.

#### Accepted prerequisite evidence

The census/checker is committed at fa497d0; its 60-second check exits 0 with wrong-source,
wrong-shape and omitted-packing red/restoration controls. Formulas are model evidence and the
checker is tested. Independent review is artifacts/gain-integration/resource-census-review.md.
The original worker preparation exceeded 539/480 s and remains a failed slice; it is not task or
wave closure. The frozen artifacts have separate bounded gate and independent acceptance receipts.

Shared activation packing across projections, row/token reuse and resident execution are named
source mechanisms. The hypothetical M3072/K2048/B384 rt1-to-rt4 scenario predicts 67.4157% fewer
logical loads at unchanged dot/store counts. Fold currently selects rt8 for that shape, so rt1 is
not an observed baseline or a current-policy speed improvement. Historical hybrid/dense receipts
remain whole-prefill results; neither this census nor the scalar theorem establishes their causal
attribution or a new GPU timing result.

The real act_scale source refinement is committed at 8b6db81 and independently accepted in
artifacts/gpu-prefill/actscale-review.md: least-cover return, full-u64 unreachable None, reachable
zero/positive witnesses and strictly decreasing rank with at most 58 doublings. Its actual-source
multiply-by-three mutation fails the proof and restored extraction passes. This meets the separately
scoped scalar prerequisite; strict resource improvement for a transformation, graph composition,
canonical primitive correspondence and the production admission contract remain open.

artifacts/gpu-prefill/prerequisite-acceptance.json retains the complete focused evidence. The next
bounded prerequisite selects the actual exact_dot extraction closure and its numeric/refusal domain.
GPU whole-prefill validation, calibrated hardware envelopes, the evolver interface and the common
Rust/Lean epic barrier retain their own exit obligations.

#### Native GPU ownership correction

The user selects new Apeiron-owned SPIR-V kernels and runtime, informed by llama.cpp and Fold.
Fold source-derived counts and historical receipts remain reference evidence, not the execution
backend or native performance baseline. docs/epics/gpu-prefill.md now fixes the actual CPU Q8
observation, signed-dot/two-limb native baseline and exact device-parity fixture ABI.
The user explicitly authorises only that initial baseline bootstrap alongside Aeneas work; later
inference optimisations remain behind registered source/refinement/resource admission. The empty
registry is not populated by that exception. Native source and emitted-body costs replace borrowed
counts before any native improvement certificate is registered.
