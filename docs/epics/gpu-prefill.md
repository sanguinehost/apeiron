# Native Apeiron SPIR-V prefill kernels

## Intent

Owner: Apeiron lead. Source base: 439d5e6. Status: initial baseline bootstrap authorised by user;
later inference optimisations retain AGENTS.md admission. Existing CPU gains remain merged.

Apeiron owns new GPU kernels, their emitted SPIR-V, runtime and formal refinements. llama.cpp and
Fold are read-only implementation references and comparators. Their existing kernels are not the
executing target or runtime dependency. This supersedes the earlier Fold-backend integration plan;
its source-bound adapter/census/scalar receipts remain evidence at their recorded scope.

Focus: hybrid prefill, first through the shared Q8 batched projection. The retained hybrid CPU
pp200/pp512 gains exceed the 2x goal; dense gains are additional transfer, with no dense 2x demand.
The GPU comparison is against matched llama.cpp GPU prefill, not CPU timing or a different carrier.
A fast primitive is a milestone, not a whole-model speed claim.

The first native kernel uses packed signed-by-signed integer dots, row/token register tiling,
per-block power-of-two scales, exact two-limb i128 accumulation and one Q32.32 rounding/narrowing.
It preserves sequential checked accumulation/refusal; no reassociation or floating epilogue is
introduced. This removes the need for Fold's unsigned recoding, f16 affine offset and XS correction.
That is a new design choice whose implementation is checked separately, not inherited proof.

Alternatives: adopting Fold's executing kernels conflicts with owner direction; float epilogues
change Apeiron's numerical observation; per-block CPU readback prevents a resident projection path.
Initial runtime is a minimal Apeiron-owned Vulkan correctness/timing harness. Serving integration
and resident hybrid state remain subsequent bounded tasks.

Completion: native kernel and actual CPU observation agree on registered positive/boundary/refusal
fixtures, source mutation rejects and restoration passes; real-source/refinement/cost links and
production admission then precede optimisation, followed by matched whole-prefill measurements.

## Specification

| boundary | baseline contract |
|---|---|
| CPU observation | src/engine/kern/gemm.rs::exact_dot with actual QuantAct::from_parts/accessors, f16_split and round_shift_32 |
| Shape | positive M/tokens/blocks; 32 lanes per block; checked buffer extents; row/token tails guarded before loads/stores |
| Inputs | signed i8 weights/activations; finite binary16 weight scales or explicit non-finite refusal; activation scale zero or 2^e, 0<=e<=58 |
| Packing | W[(block*8+quad)*M+row]; DM[block*M+row]=(signed mantissa, exponent); token-major X packed signed lanes and activation exponent E |
| Outputs | exact i64 Q32.32 value plus refusal flag; real CPU checked i128 fold/narrowing is the oracle |
| Accumulator | two u64 limbs form two's-complement i128; each sequential addition checks signed overflow; no float |
| Rounding | add 2^31 once and arithmetic shift32; output must fit i64; source rounding-add panic boundary excluded explicitly |
| Refusal | non-finite weight, i128 accumulation overflow or final i64 overflow; malformed geometry rejected by host before device access |
| State | isolated buffers per fixture, complete output/refusal readback, explicit submit/wait; no KV or recurrence claim |

The fixture producer calls the actual CPU body and actual f16 splitter. Native DM/packed-word
construction is tested through shader parity, not presented as already source-refined. Buffer and
shader ABI are checked by the independent fixture dimensions and exact observations. Compiler,
Vulkan driver/device and host FFI remain trusted/tested boundaries; spirv-val supplies syntax only.
The current Spirv model covers a different Q15/mixed-signed profile; the new OpSDot/int64 and
checked i128 profile needs explicit extension/refinement, without silently widening the checker.

## Plan

Lead owns this document and resource-guided-evolution.md in supplied checkout (documentation
exception). Native Luna/Sol assignment applies. One writer per path; all tasks <=3 files.
The user explicitly allows only the initial native baseline before optimisation registration.
No later tuning/optimisation worker is exempt from tools/dispatch/admit.py --launch.

| task | owner / files | gate / control |
|---|---|---|
| Ownership correction, APEIRON-GPU-PREFILL-W2-T0 | lead: this document, resource-guided-evolution.md, artifacts/gpu-prefill/native-boundary-review.md | 60s document/domain check; missing native ownership or admission boundary rejects; Sol review |
| Actual CPU fixtures, APEIRON-GPU-PREFILL-W2-T1 | Sol: src/engine/kern/gemm.rs test-only module hook, src/engine/kern/native_gpu_probe.rs | 480s focused cargo test --lib engine::kern::gemm::native_gpu_probe::native_q8_prefill_fixture -- --ignored --exact; producer writes agreed binary; missing output path rejects |
| Native baseline, APEIRON-GPU-PREFILL-W2-T2 | Luna: src/spirv/q8_tile.comp, tools/gpu/q8_driver.c, tools/gpu/q8_gate.py | 540s whole gate compile/validate/actual CPU fixtures/device parity; negate real dot update makes red; restoration green |
| Native source refinement | lead selects <=3 actual extraction/model/refinement paths after closure inspection | gate-first; real conversion and integer-dot source links, canonical RingCell attribution, axiom/mutation controls |
| Cost and production admission | separately scoped registry/cost tasks, independent verdict | joint reachable witness, universal observation/refusal preservation, strict registered improvement; production positive/red/restore |
| Serving and matched GPU prefill | admitted worker and lead measurement | native serving integration and exact continuation; five quiet paired rounds against matched llama.cpp GPU |

The binary fixture ABI is SPIQ8V01 (8 bytes), case count u32 little-endian. Each case begins
(M, tokens, blocks, case_id), four u32. Payload order: packed W u32[M*blocks*8], DM u32[M*blocks*2]
(mantissa two's-complement, exponent; exponent UINT32_MAX denotes non-finite), X u32[tokens*blocks*8],
E u32[tokens*blocks] (UINT32_MAX denotes zero scale), expected i64[tokens*M] and refusal u32[tokens*M].
Golden values are observed only for refusal0. Fixture output path is APEIRON_NATIVE_Q8_FIXTURE.

Budgets include descendants and any lane wait. CPU test lane is root allocated; shader/C compilation
uses a private temporary directory. Lean stays serialized, 12GiB/no swap/one thread/540s invocation.
Hardware: fresh host capability check reports RTX5080, driver615.71.09, Vulkan1.4.351; sandbox probes
need device access. Local tool paths/revisions and benchmark receipts accompany each executed gate.

## Evidence retained and exits

8b6db81 proves the actual scalar act_scale least-cover return and <=58 doublings, with real-source
mutation/restoration. 57db58f checks the earlier Fold adapter model with14controls; it does not
establish a native GPU adapter. fa497d0 is a checked source census, not hardware attribution.
artifacts/gpu-prefill/prerequisite-acceptance.json retains full logs/identities and failed preparation
budgets. These retain their grades and source identities; this ownership change does not upgrade them.

| exit | current status |
|---|---|
| Native ownership / first-baseline exception | user decision; independent Sol architecture review accepts baseline domain |
| Actual CPU/native device parity and mutation/restoration | tested: 14 cases, 14,678 exact outputs and 32 refusals; dot mutation 13,973 mismatches; restoration green |
| Actual native emitted-body / canonical / source-cost correspondence | open |
| Optimisation admission | closed to dispatch until accepted registration |
| Native serving, hybrid recurrence, state continuation | open |
| Matched llama.cpp GPU whole-prefill gain | unmeasured |
| Shared Rust/extraction/both Lean barriers and trust inventory | open |

SCOPE: this document, resource-guided-evolution.md, native-boundary-review.md; lead documentation exception.
REGISTER: native GPU ownership, first baseline bootstrap and validation scope; no speed claim.
CORPUS: Spirv.Dp4a's stated CPU/emitted-body exclusion and actual exact_dot/source helpers read.
REFERENCE: /home/socol/Workspace/apeiron/src/engine/kern/gemm.rs:311.
GATE: 60s native document/domain check and missing-ownership/control; old Fold runtime target is red.
GRADE: planning/anchor; user-authorised baseline implementation, later optimisation remains admitted.

Native baseline acceptance is in artifacts/gpu-prefill/native-baseline.md and native-baseline-acceptance.json.
The fixture domain is validated synthetic QuantAct::from_parts input, not established quant_act
constructor reachability. Production admission requires additional actual-constructor fixtures.
The baseline uses local8x8; each invocation computes RT2BT4; dispatch is ceil(M/16),ceil(tokens/32).
The whole epic remains open at the source/cost, serving, comparison and shared-proof exits above.

## Constructor and source prerequisites

The next fixture population is separate from SPIQ8V01's historical synthetic population. It calls
actual `chart::quant_act` on retained deterministic i64 inputs, checks `dequantizes` and exact
scales/codes before reconstruction, and observes actual `exact_dot`. Zero, scale transitions,
signed ties, high representable inputs, constructor refusals, non-finite weights, final narrowing
and row/token tails are required. Constructor refusal is a host witness, not a GPU dispatch with
an invented activation. This is tested reachability, not universal constructor refinement.

| task | owner / files | gate / control |
|---|---|---|
| APEIRON-GPU-PREFILL-W3-T1 | Luna; native_gpu_probe.rs, q8_gate.py, new tools/gpu/quantizer_gate.py; isolated codex/native-real-quantizer at3855580 | gate-first: timeout540 python3 tools/gpu/quantizer_gate.py; exact CPU/device parity, real quant_act tie-rounding mutation in a throwaway copy and restoration; GPU dot mutation retained |
| APEIRON-GPU-PREFILL-SPIKE-narrow-error | Sol; scratch only | 300s feasibility question: exact source error representation included with actual TryFrom body, without tool/model changes or unmodelled axioms |
| APEIRON-GPU-PREFILL-SPIKE-next-links | independent Sol review; scratch report only | 300s source/cost decomposition; no native source, cost or performance claim from a citation |

Main Rust target is one serialized lane owned by the constructor-fixture task. Source/proof/model
tasks require reviewed decomposition before implementation; optimization retains production admission.

SCOPE: native_gpu_probe.rs, q8_gate.py, quantizer_gate.py; constructor fixtures and controls.
REGISTER: tested actual-constructor reachability and exact CPU/device observations; no optimization.
CORPUS: actual chart::quant_act, QuantAct::dequantizes and gemm::exact_dot read; no constructor theorem.
REFERENCE: /home/socol/Workspace/apeiron/src/engine/chart/mod.rs.
GATE: MISSING initially; gate-first timeout540 python3 tools/gpu/quantizer_gate.py, from task worktree.
GRADE: tested target; universal source/cost link and GPU throughput remain open.
