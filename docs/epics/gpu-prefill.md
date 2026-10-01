# GPU prefill through Fold's existing SPIR-V machinery

## Intent

Owner: lead. Status: adapter model checked and scalar source prerequisite proved; no implementation dispatch.
Main merge: `01de89d020954897f48e9863f806ac9df0bbf5af`, with the tested CPU runtime and fixtures.
Fold source: `/home/socol/Workspace/fold`, `d497161f32e0d366220f2065ec36e680bd9aed86`;
read-only source review and digests: artifacts/gain-integration/fold-gpu-reuse.md.
Documentation exception: lead owns this document, resource-guided-evolution.md and
artifacts/gain-integration/fold-plan-review.md in the supplied checkout; no build lane.

The hybrid CPU prefill work exceeds its 2x target on the retained pp200/pp512 receipts and also
improves dense models. Carry that shared algebraic structure into GPU execution using Fold's
existing performant implementation. Dense improvements are a transfer result, with no dense 2x gate.

Fold already supplies register-tiled integer-dot projections, fused batch glue, tiled attention,
hybrid decode and a resident dense prefill graph. Its retained RTX 5080 GEMV comparison records
f16 SPIR-V at 155.9 versus cuBLAS 154.1 microseconds for 8192 squared, and 137.8 versus 131.1 for
4096 by 14336: 1.2% and 5.1% time overhead at matched shapes/carrier. Q4 DP4A records 61.0/53.0
microseconds, 2.53x/2.47x versus that f16 baseline with a different carrier. These are historical
kernel receipts in /home/socol/Workspace/fold/hashtrinity/bench/results_gpu.633ea8b.md,
not new whole-prefill measurements. Source and protocol details are in fold-gpu-receipts.md.

The structural starting point is existing two-axis reuse: one weight load serves several tokens,
and one activation load serves several rows. The resident GpuPrefillStack already carries
intermediates through a whole dense layer stack with one submit/drain per chunk. Extend that
mechanism to Apeiron's packing, session state and hybrid recurrence after establishing their
correspondence. GpuPrefill's projection-only route remains a distinct baseline with per-projection
transfers; it is not the resident stack. Do not schedule construction of another GPU backend.

Alternatives: rebuilding the emitter/runtime repeats existing work; per-operator offload retains
unnecessary transfers; replacing the float scale epilogue with fixed point changes existing
arithmetic. Use existing capabilities where they fit, and isolate each additional proof profile.
Programme direction requires performance-preserving consolidation, not slower extraction substitutes.

Completion: one source-bound resident prefill integration contract and an admitted, measured
transformation on its registered domain, preserving outputs/state/refusals under the selected
numeric contract. Register the hybrid coverage and pp200/pp512 success criterion before execution.
Measure cold upload and retained residency separately; count transfers, launches, synchronization,
readback and continuation. Report GPU improvement against its registered GPU baseline.

## Specification

Every Fold path below resolves under /home/socol/Workspace/fold/; inspected APIs and digests are in
artifacts/gain-integration/fold-gpu-reuse.md. Apeiron owns the reusable primitives/refinements under
docs/direction.md. Selecting code for consolidation allocates a separate bounded migration task;
this document changes neither sibling source nor the programme boundary.

| interface | existing source and contract to retain |
|---|---|
| Kernel selection | modelfold/src/simt_nn/spec.rs GemvSpec; typed carrier, scale face, mapping, fusion and reduction, with legalize refusal |
| Projection | modelfold/src/simt/superblock.rs gemv_dp4a_batch_rt_module; rt rows by bt tokens; exact packed lanes and existing float scale/bias association |
| Binding ABI | W, XQ, OUT, DM, XS, SX; K divisible by 32, positive dimensions/tiles, checked sizes, padded token extent and guarded live-row stores |
| Batch operators | modelfold/src/simt_nn/batch.rs norm/quantize, SwiGLU, QKV/RoPE and last-row gather; actual emitted bodies and operation order |
| Dense resident graph | bedrock/src/serve/gpu.rs GpuPrefillStack; constructor domain, one drain/chunk, KV mirror/continuation, LogitRows All/Last and explicit refusal |
| Hybrid state | simt_nn/deltanet.rs and GpuDecode's existing recurrent buffers; convolution history, decay/beta, normalization, state reset and gated readout |
| Model/session adapter | Apeiron Q8 packing versus Fold per-32 chart; named conversion or incompatibility, stable identity, ownership and constructor-derived capacities |
| Resource record | Executing graph's arithmetic, chart/packing bytes, uploads/readbacks, allocations, dispatches/drains, peak-live storage and critical path |

GpuPrefillStack's constructor currently selects uniform dense fused QKV; it does not provide
hybrid batched DeltaNet prefill. Existing hybrid decode kernels supply the recurrence source,
while batched recurrence scheduling is the additional contract. Dense integration validates shared
machinery; hybrid prefill remains the objective.

Proof profiles follow actual components. Spirv.gpu_dp4a_word_refines_cpu_word links model integer
words to dp4a_gemv_spec; actual CPU/emitted-body links remain open. Existing DP4A shaders have
integer dots followed by float epilogues. Preserve their association and register separately the
integer exact observation and the full float contract; an argmax match is not full-output equality.
Batch norms, attention and DeltaNet use barriers/workgroup memory; cooperative matrices and
physical addresses have additional profiles. Extend each required profile without silently widening
the existing checker. docs/epics/gpu-spv.md retains checker ownership and its composition residuals.

Mapped result lifetime, exclusive submission and buffer reclamation need an explicit adapter
contract. Fold's Copy Buffer handles have no Drop; lifecycle/reclamation is not inferred from
word proofs. Driver/compiler lowering, device execution and measurement provenance remain external
trust. Source/refinement/resource work may establish prerequisites without an optimization worker.

## Plan

The user's Luna/Sol assignment applies: Luna owns bounded work, Sol reviews Luna artifacts,
and a different-model independent reviewer checks lead-written boundary changes. Lead integrates.
Each implementation task owns at most three files, one focused gate and an attributable control.
Every gate is bounded at 540 s including descendants/lock waiting; Lean uses the repository lock,
12 GiB/no swap and LEAN_NUM_THREADS=1. No new timing or runtime change is allocated here.

| order | task / owner | owned paths | gate and rejecting control |
|---|---|---|---|
| 1 | Source-bound adapter design; Luna, Sol review; APEIRON-GPU-PREFILL-W1-T2 | artifacts/gpu-prefill/adapter-contract.md, adapter-identities.json, adapter-gate.py | `timeout 120s python3 artifacts/gpu-prefill/adapter-gate.py --check --controls`; gate-first; wrong lane signedness, stride, session identity and padded extent must reject |
| 2 | Actual primitive refinement; Luna, Sol review | Real source extraction/refinement pair and control, at most 3 selected paths | MISSING selected CPU/emitted-body link; integer core and unchanged float epilogue carry distinct observations; source mutation must fail the named link |
| 3 | Existing graph cost correspondence; Luna, Sol review | Source-bound graph/resource pair and control, at most 3 | MISSING constructor-reachable success/refusal and cost gate; omitted packing, transfer or drain must reject; consume resource-guided-evolution.md |
| 4 | Production admission contract; lead plus independent reviewer | Bound contract/proof/source identities, at most 3 | Production-path launcher positive, attributable refusal and restoration; preservation and strict improvement in registered resource model |
| 5 | Adapter or hybrid schedule implementation; admitted worker | Existing primitives plus adapter/schedule paths selected by accepted contract, at most 3 | MISSING actual CPU/GPU observation/state/refusal and continuation gate; wrong binding, row store or recurrent reset must reject |
| 6 | Matched whole-prefill validation; lead | Registered runner/manifest/receipt, at most 3 | Five quiet paired rounds per workload, fixed cold/resident policy; omitted upload, sync or readback must fail harness control |

Reuse existing Fold runtime and schedule before designing new ones. Register the numeric, shape and
hybrid state domain before allocating implementation. Refresh performance on that exact identity;
isolated kernel timings cannot establish the end-to-end result. No canonical migration, pin change,
training or Intent source change is included.

### First executable prerequisite

Write the source-bound adapter contract comparing actual Apeiron Q8/float production paths with
Fold's six-binding per-32 ABI and resident stack. Establish stable model/session identity, checked
sizes, ownership, successful/refusing witnesses, KV continuation and recurrent-state reset. Select
one actual source-cost seam and its executable red gate; do not start a kernel writer from a citation.

Existing capability and syntax receipts remain useful: RTX 5080, Vulkan 1.4.351, integer-dot support;
the captured 676-word module passes spirv-val. Its dynamic float epilogue is not covered by an
integer-only execution theorem. Checker composition stays in gpu-spv.md; its contract/verification
claim labels need reconciliation before relying on those rows for dispatch. A complete new backend
or whole-SPIR-V proof is not the prerequisite for this read-only correspondence task.

### Exit obligations

| obligation | grade / current status | closing evidence |
|---|---|---|
| Existing GPU source and measured starting point | anchor + historical measured / retained | Fold revision, source digests, exact matched GEMV receipt and reusable resident/hybrid APIs |
| Adapter domain and production body links | proved + tested / open | Actual packing/numeric/constructor relations and emitted-body refinement; focused gate MISSING |
| Resident/hybrid state and resource correspondence | proved + tested / open | Reachable state, output/refusal/continuation, loop/graph cost and source mutation; gate MISSING |
| Implementation admission | proved + tested / closed to dispatch | Independent contract verdict; production positive, rejection and restoration |
| End-to-end hybrid pp200/pp512 | measured / open | Five quiet paired rounds including all required transfers/launches/drains; pre-registered success criterion |
| Shared barrier and trust | tested + proved / unrun | AGENTS.md Rust gates, real-source regeneration freshness, both Lean builds and controls; source-size/TCB inventory |

SCOPE: docs/epics/gpu-prefill.md, docs/epics/resource-guided-evolution.md,
       artifacts/gain-integration/fold-plan-review.md; lead documentation exception on main 01de89d.
REGISTER: planning; inference implementation, new proof acceptance and throughput excluded.
CORPUS: Spirv.gpu_dp4a_word_refines_cpu_word at its model domain; actual Fold sources read.
REFERENCE: /home/socol/Workspace/fold/bedrock/src/serve/gpu.rs and
           /home/socol/Workspace/fold/modelfold/src/simt/superblock.rs.
GATE: timeout 60s document/source check and throwaway missing-Intent control,
      recorded in artifacts/gain-integration/fold-plan-review.md; baseline lacks Fold reuse contract.
GRADE: anchor/planning; historical measurements retain workload/carrier/protocol identities.

Every inference optimization implementation worker, including native agents, must execute its
registered argv through python3 tools/dispatch/admit.py --launch <proposal.json> in its task worktree.
A route the launcher cannot execute is not admitted; direct native dispatch is no bypass.

### Executing prerequisite allocation

`APEIRON-GPU-PREFILL-W1-T2`: Luna owns exactly the three adapter artifacts above in
/tmp/worktree-agent-APEIRON-GPU-PREFILL-W1-T2, branch codex/gpu-adapter-w1, base ad707f4.
No shared build lane. Slice budget 480 s; local gate budget 120 s, including controls/descendants.
Baseline: adapter-gate.py absent, test -f exits 1. The gate binds actual source identities and checks
packing/shape/state contract witnesses; it must not present a handwritten oracle as runtime
refinement. Sol independently reviews the resulting contract; lead alone merges.

### Accepted prerequisite evidence

The main merge remains 01de89d. The source-bound adapter model is committed at 57db58f;
its exact 120-second check exits 0 with fourteen attributable red/restoration controls.
The initial checker admitted four defects; the repaired checker rejects them. Independent
acceptance is artifacts/gpu-prefill/adapter-final-review.md. Empty-input and allocation-overflow
refusal remain declared policies, not exercised witnesses. Production packing, float epilogue
equality, constructor/continuation integration and GPU execution remain open.

The independently reviewed scalar prerequisite is committed at 8b6db81:
proof/resource/ActScaleSource.lean extracts the real pow2.rs act_scale body;
ActScaleResource.scale_positive proves least covering power-of-two return for every positive u64,
and scale_never_none proves None unreachable on the full u64 domain. Loop rank proves at most
58 doublings. The focused gate exits 0 in 12.682 s under the 540-second lane/memory/thread bound;
changing the real doubling to multiplication by three makes the extracted proof red and restoration
green. This source-level result does not establish the full quantization/projection or GPU result.

artifacts/gpu-prefill/prerequisite-acceptance.json retains commands, source/tool identities,
full scalar stage logs, model controls and review identities. The next source-linked target is
src/engine/kern/gemm.rs::exact_dot with its actual QuantAct and rounding callees; its small extracted
call closure and admissible domain must be established before allocating a projection proof worker.
The census supplies modeled graph counts; actual graph/cost correspondence and production admission
remain open. No optimization implementation is admitted by these prerequisite results.

SCOPE: docs/epics/gpu-prefill.md, docs/epics/resource-guided-evolution.md and
       artifacts/gpu-prefill/prerequisite-acceptance.json; lead documentation exception on main.
REGISTER: acceptance records; GPU execution and whole-epic closure excluded.
CORPUS: ActScaleResource.scale_positive and scale_never_none; actual extracted source;
        closed-cone kernel receipt retained in prerequisite-acceptance.json.
REFERENCE: /home/socol/Workspace/apeiron/proof/resource/ActScaleResource.lean.
GATE: timeout 60s python3 -c 'import json,subprocess; from pathlib import Path;
      r=json.loads(Path("artifacts/gpu-prefill/prerequisite-acceptance.json").read_text());
      raise SystemExit(subprocess.call(r["record_gate"]["argv"]))';
      absent receipt baseline exits 1; missing-proof-evidence control rejects; restored record passes.
GRADE: tested document/evidence consistency; scalar proved, source census/adapter model;
       no new speed measurement or implementation admission.
