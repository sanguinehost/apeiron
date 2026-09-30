# GPU prefill from verified integer primitives

## Intent

Owner: lead. Status: draft for architecture review; no implementation dispatch. Base: measured CPU
prefill stack `04d206013f70f245dfdaf2619c0b1846d10b9922`; integration receipt lives in artifacts/gain-integration. Planning exception:
lead owns this document, resource-guided-evolution.md and one additive docs/README.md index entry.

The hybrid prefill work exceeds its 2x target on the retained pp200/pp512 receipts and also improves
dense models. Move the reusable computation to GPU execution while retaining that shared structure.
The host exposes an RTX 5080 with Vulkan 1.4.351, driver 615.71.09 and 16303 MiB; hardware availability is
an observed capability, not a throughput result. Existing docs/epics/gpu-spv.md owns the integer
fragment semantics/checker refinement. This epic consumes that boundary rather than replacing it.

First structural hypothesis: compile a prefill block into a resident-buffer dispatch graph so weight
packing, activation packing and intermediate lifetimes are shared across operators and tokens.
Measure the whole prompt, including dispatch, synchronisation and required data transfer. A fast
isolated dot product is insufficient to choose the graph. Dense models use the same primitive
interfaces and remain useful validation workloads without acquiring a 2x requirement.

Alternatives: a per-operator port loses the opportunity to share buffers/packing; adopting cooperative
matrices immediately introduces a different fragment/concurrency boundary before the existing one is
closed. Both remain later candidates if measured bottlenecks justify them.

Completion: one registered integer projection block in a resident-buffer dispatch graph; exact
CPU/GPU integer outputs and committed state, explicit refusing cases,
source-connected primitive links, rejecting controls and five quiet paired end-to-end receipts at
pp200/pp512. Report each model separately. Performance success is improvement on the registered GPU
baseline; no CPU-to-GPU ratio is assumed. Whole Rust/proof barriers remain required at closure.

## Specification

| interface | contract / authority |
|---|---|
| Quantised integer dot | Actual packed inputs/accumulator widths, overflow and scale/refusal semantics; shared dp4a_gemv_spec at the realised canonical ring face |
| Emitted words | src/spirv/check.rs check_fragment; concrete word stream bound by hash; Spirv.Check header is present, full soundness and concrete module denotation remain obligations |
| Buffer graph | Explicit shapes/strides, read/write regions, live ranges, device residency and constructor-derived capacities; no hidden host allocation or overlapping writes |
| Dispatch | Actual group counts, row guards and supported opcodes; reuse Spirv.Dp4a accessChain_in_bounds at its stated model scope |
| Carrier | Integer fixed-point v0; the current served float epilogue is a separately named migration/error-model task, not silently included in the integer theorem |
| Resource record | Packing/copy bytes, allocations, dispatches, workgroups, scratch/peak-live bytes and operation counts; supplied to resource-guided-evolution.md |

CPU and GPU bodies must refine the same named primitive spec. A model word theorem does not prove an
emitter, Vulkan driver, shader compiler or device. Modular-word reassociation does not license binary32
reassociation. Barriers, atomics, subgroups and cooperative matrices open separate fragment tasks.
Driver lowering, device execution, clocks and measurement provenance remain the platform boundary.

## Plan

One owner per path, private task targets; no inference optimization worker before mandatory admission.
The user's Luna/Sol assignment applies: Luna owns bounded source/refinement/resource work,
Sol supplies independent review, and the lead integrates.
Each implementation task is written after its preceding proof/cost prerequisite passes, owns at most
three files and has one focused gate plus a rejecting control. Gates below are task targets, not
historical approvals. Every invocation is bounded at 540 s including lock waiting and descendants;
Lean uses the repository lane lock, 12 GiB/no swap and LEAN_NUM_THREADS=1.

| order | bounded task / owner | owned paths | gate and negative control |
|---|---|---|---|
| 1 | Fragment prerequisite; lead, `APEIRON-GPU-PREFILL-W1-T1` | artifacts/gain-integration/gpu-fragment-prerequisite.md | `timeout 540s cargo test --test gpu_spv_check --locked`; existing malformed-word tests reject; source-mutation control is a separate required GPU-SPV task |
| 2 | Close checker components; Luna bounded proof worker | Select at most 3 files from proof/corpus/Spirv/CheckScan*.lean, CheckApplyShape*.lean, Check.lean after exact source review | Existing gpu-spv.md decomposition; lake build named component; change real shape/refusal in throwaway extraction and require attributable red |
| 3 | Bind one concrete integer primitive; Luna bounded source/refinement worker | Exact extraction/bridge paths selected by gate-first task | MISSING concrete emitted-module/CPU-body refinement gate; establish it before allocating a kernel writer |
| 4 | Register resource correspondence; lead + Sol independent reviewer | Exact source-cost bridge and registry payload, selected after step 3 | Production-path positive admission and rejecting/restoration controls; typed reachable witness and universal preservation/improvement |
| 5 | Resident block dispatch; admitted Luna kernel worker | GPU host/kernel paths selected from measured residency boundary; at most 3 | MISSING constructor-to-prefill CPU/GPU state/output/refusal gate; corrupt stride/row guard/binding must reject |
| 6 | Matched prefill receipt; lead | Benchmark runner and identity manifest; at most 3 | Five paired quiet rounds per registered workload with all outputs; transfer/launch omitted from timer must fail the harness control |

Do not allocate steps 3–6 from this draft until their source paths, red commands and independent
architecture decision exist. The first deliverable is a fresh capability/fragment receipt and exact
checker residual, not a new GPU runtime. Reuse the existing GPU epic's accepted premises; any boundary
change goes to independent review. No canonical migration, pin change or model training is included.

SCOPE: gpu-prefill.md, resource-guided-evolution.md, artifacts/gain-integration/sol-epic-review.md;
       Sol review in the supplied checkout, base e33977aa72534ad56775110be99cf002ca96fa37.
       Lead separately owns docs/README.md; no build lane acquired.
REGISTER: planning; source proof, GPU throughput and implementation admission excluded.
CORPUS: Spirv.checkFragment_header and Spirv.gpu_dp4a_word_refines_cpu_word read with their
        explicit residuals; no fresh Lean acceptance implied by citation.
REFERENCE: /home/socol/Workspace/apeiron/docs/epics/gpu-spv.md and proof/corpus/Spirv/Dp4a.lean.
GATE: timeout 60s document checks recorded in artifacts/gain-integration/sol-epic-review.md;
      baseline pending, missing-Intent control must reject; runtime/cost gates MISSING as above.
GRADE: anchor/planning; GPU capability observation separate, historical CPU receipts unchanged.

### First executable prerequisite

After the integration owner releases `target/`, run the existing focused checker test above,
record source/extraction/fixture sha256 and actual tools, and report the exact decoder residual.
The captured `tests/gpu_spv_check.rs` module has 676 words and a dynamic float epilogue; acceptance
is syntax coverage, not integer-execution refinement. The integer executing subset has 37 opcodes
and the served checker admits 42; no admitted float opcode acquires a value theorem.

The first gate-first design task owns only the prerequisite receipt and this epic, selects one
integer projection's source, shapes, output/refusal type and concrete module, and writes its missing
refinement gate and control. It does not start a kernel writer. Full checker composition remains
in `gpu-spv.md`; do not duplicate its proof ownership here. That document's contract table labels
targets `proved` while its verification table and actual source say open; lead reconciliation is
required before relying on those table rows for dispatch.

### Exit obligations

| obligation | grade / current status | closing evidence |
|---|---|---|
| One constructor-reachable integer block and concrete module | proved / open | CPU-body and decoded-module links to `Spirv.dp4a_gemv_spec`; exact gate MISSING |
| Resident graph observations | tested / open | constructor, shape/stride/live-region, output/state/refusal tests and binding mutation; gate MISSING |
| Implementation admission | proved + tested / closed to dispatch | registered actual-source/cost contract and production positive/refusal/restoration |
| End-to-end pp200/pp512 | measured / open | five quiet paired rounds including required uploads, launches, synchronization and readback; fixed residency policy |
| Shared local barrier | tested + proved / unrun | AGENTS.md §6 Rust gates, regeneration freshness, `lake build` in both proof packages, all named controls |
| Trust and source size | planning / open | source/tool digests, ≤1,500 lines per owned source, ≤12 immediate sources per directory, local checker ≤5,000 physical lines; external trust inventory |

Pre-register the GPU baseline before graph changes and report cold upload and retained-residency
results separately. A throughput improvement is a measured result of the named protocol. A new
carrier/quality observation is a separate contract and verdict; it cannot discharge exact integer
preservation. No whole-model GPU implementation or concurrency fragment is allocated by this epic.

Native agents may establish separately scoped source/refinement/resource prerequisites. Every
inference optimization implementation worker, including Luna, must execute its registered worker
argv through `python3 tools/dispatch/admit.py --launch <proposal.json>` in its task worktree.
A native route the launcher cannot execute remains blocked; direct native dispatch is no bypass.
