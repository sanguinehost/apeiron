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
The original baseline domain is validated synthetic QuantAct::from_parts input. The separate
constructor-backed population below tests actual quant_act reachability; universal constructor
refinement and production admission remain open.
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

The narrowed representation task is APEIRON-GPU-PREFILL-W3-T2, Sol, in an isolated worktree:
`proof/native/NarrowSource.lean`, `proof/native/NarrowRefinement.lean`,
`proof/native/narrow_gate.py`. Dispatch requires the independent Luna verdict on the exact Charon
sysroot, `nightly-2026-08-18`; the shell-default Rust sysroot is a different source identity.
The real error is a single-field `TryFromIntError(IntErrorKind)` struct. A reviewed file-local
alias may represent its extracted tuple erasure; the shared Unit model and tool pins stay fixed.
The adapter rejects LLBC field/variant drift and preserves every generated conversion operation.
Universal inclusive-range success, lower/upper refusal, boundary witnesses and closed-cone audits
are required. The gate freshly extracts the actual std conversion, checks bridge/source identity,
checks Lean and independently mutates lower/upper generated guards, then restores both.

SCOPE: proof/native/NarrowSource.lean, NarrowRefinement.lean, narrow_gate.py; actual std narrowing.
REGISTER: extraction/type representation/range refinement; no inference implementation change.
CORPUS: actual std conversion and Aeneas IScalar.cast_inBounds_spec; no canonical ring claim.
REFERENCE: /home/socol/Workspace/apeiron/target/aeneas-model/lean/Aeneas/Std/Scalar/Casts.lean.
GATE: MISSING initially; gate-first python3 proof/native/narrow_gate.py --controls, whole budget540s.
GRADE: proved target for this scalar conversion only; shader and aggregate source-cost links open.

## Private iterator compatibility prerequisite

APEIRON-GPU-PREFILL-W3-T3 owns only `proof/native/step_compat.patch`,
`proof/native/StepRefinement.lean`, and `proof/native/step_gate.py`; Luna writes, Sol verifies.
Its isolated worktree starts from the current main revision. The private patch adds the missing
Step record fields and wires the existing scalar overflowing functions. The private Aeneas
binary derives from pinned505b6ca35217e7be5c96c3e2f8045edfbdf47291 plus the retained array-repeat
patch; its builtin registry is generated from the privately patched model. Installed tools,
shared model, pins and production Rust remain unchanged. A changed binary has its own digest.

Intent: translate actual unchanged `gemm::exact_dot` after the observed missing
`Step.forward_overflowing` diagnostic. Source loop normalization changes an inference body and
requires admission; private model compatibility is a separately scoped prerequisite.
Spec: add no scalar formula, axiom or opaque implementation. Validate exact pinned archives,
model patch set and real std source. Check StepUsize field wiring and overflow witnesses under
the closed cone. Wiring proofs establish model behavior; actual Rust Step semantics require
their own source link. A later extraction error remains a blocker, not acceptance.
Plan: one bounded private-build/extraction/wiring gate with unmodified-tool failure, generated
registry mutation failure and restoration. No hardware or source-cost improvement claim.

SCOPE: step_compat.patch, StepRefinement.lean, step_gate.py; private compatibility and controls.
REGISTER: tested tool/model compatibility and proved model wiring; excludes actual Step/source-cost refinement.
CORPUS: actual pinned core/src/iter/range.rs and Aeneas/Std/Core/Iter.lean read; no Rust Step refinement found.
REFERENCE: /home/socol/Workspace/apeiron/target/aeneas-model/lean/Aeneas/Std/Core/Iter.lean.
GATE: MISSING initially; gate-first timeout540 python3 proof/native/step_gate.py --actual-exact-dot --controls;
      fresh unchanged actual body, generated registry rejecting control, restoration, wiring audit;
      serialized Lean lane,12GiB/no swap/one thread, one whole540s budget including all builds.
GRADE: tested compatibility target and proved model-wiring target; full-body/cost/hardware links open.

Independent Sol architecture verdict: /tmp/opencode/native-step-compatibility-review.md.
Build feasibility within540s is unmeasured. Private COW caches stay on the repository filesystem;
no /tmp multi-gigabyte dependency copies or shared model-patches.sh invocation.

## Prerequisite acceptance

`artifacts/gpu-prefill/native-prerequisites.md` and `native-prerequisites-acceptance.json` retain
the fresh scalar narrowing and actual-constructor fixture/control results. e1ec6a5 contains the
scalar proof;942c01f contains the final constructor fixtures. Narrowing is proved only for the
direct selected std conversion. The actual quantizer population is tested on RTX5080:12cases,
14581successful outputs and5refusals. The original14case synthetic profile remains green.
All three Rust commands pass on the final fixture source identity. Whole source/cost, emitted
semantics, admission, serving, matched GPU prefill and shared proof exits remain open.

SCOPE: this document, native-prerequisites.md, native-prerequisites-acceptance.json; lead documentation exception.
REGISTER: scoped acceptance and retained failures; no performance or epic closure.
CORPUS: NarrowRefinement.in_range/below_min/above_max and actual quant_act/exact_dot fixture source read.
REFERENCE: /home/socol/Workspace/apeiron/proof/native/NarrowRefinement.lean.
GATE: timeout60 receipt/source/control check; corrupted lower mutation status rejects.
GRADE: proved scalar conversion; tested constructor/device population; remaining links open.

The first private compatibility run exits1: registry generation takes406.676s, then model
boundary proofs fail; actual-body/private-tool controls are not reached. Sol takes the same
three paths after Luna releases them. The interpreted generator fails on its missing native
initializer wrapper; that route is rejected. The compiled `lake exe extract` route uses a
private COW cache only after complete model/generator/dependency source identity checks, then
freshly regenerates the registry. An intermediate full gate exits1 in169.909s: model proofs and16
audits pass, private translation exits0, but generated `ExactDot.lean` declares the final signed
conversion as an axiom. The no-axiom policy rejects it before registry mutation/restoration.
Neither translation exit0 nor the model proofs close this task.

The next selection includes only the exact real signed `TryFrom` method and its error types.
Broad `core::convert::num::_` selection is rejected: it changes unrelated widening-conversion
call effects and fails full Lean checking. The local error alias preserves the six-variant
payload and every generated conversion operation. Check the emitted conversion against the
accepted `NarrowSource.lean`, prove its inclusive-range/refusal observation through the actual
loop-exhaustion branch, and audit the complete extracted body. Keep the Step registry control,
LLBC field/enum control, both narrowing guard controls and a compiling semantic mutation of
the private `Result.ok` error branch. The `Result.ok` proof is model-level until its actual
Rust body/model correspondence is established. Full loop denotation and source cost stay open.

Independent Luna boundary verdict: /tmp/opencode/native-full-narrow-link-review.md.
Sol retains the same three owned files; Luna reviews the frozen final artifact. A named
`target/native-step-model-cache` replaces the task-history cache path. The gate checks its
complete prepared source/configuration/dependency identity before private COW reuse, runs all
fresh generation/build/control stages within one540s budget, and rejects an absent or stale
cache. Root owns cache preparation in the serialized lane; cold preparation has its own
bounded outcome and cannot substitute for the gate. Installed tools/models, pins, Rust and
inference implementation remain unchanged. No gate or theorem premise is removed.

## Extracted terminal-branch acceptance

Main df57362 adds step_compat.patch, StepRefinement.lean and step_gate.py. The fresh main gate
`python3 proof/native/step_gate.py --actual-exact-dot --controls` exits0 in439.032s of540s.
Independent Luna accepts the frozen source identity. Private pinned compatibility translates the
unchanged actual Rust root; all102 generated declarations audit under the closed cone. The
terminal-branch theorems link actual checked rounding and real signed narrowing through the
Aeneas Result.ok model, with inclusive endpoint and adjacent refusal witnesses. Guard, error-shape,
model Err-to-panic and unique registry method-pair controls reject; all restorations pass.
The failed434.615s attempt stays rejected because its registry mutant translated successfully.
Full receipts, logs, source texts, tool/cache identities and verdicts are retained in
artifacts/gpu-prefill/native-prerequisites-acceptance.json.

Open: Rust Result.ok source refinement, actual constructor and accumulator-prefix reachability,
whole-loop denotation and cost, native emitted/resource equivalence, admitted optimization,
resident hybrid serving, matched200/512token GPU prefill and shared epic proof/size exits.
The next bounded proof task connects a costed trace to the actual generated inner loop; a
handwritten32B recurrence alone is model evidence. No runtime optimization is allocated from it.
Signed cooperative block tiles and guarded exponent factoring remain read-only candidates;
capability properties and fixture eligibility establish no throughput ratio.

## Finite loop-trace prerequisite

Intent: the actual extracted exact_dot loops call Aeneas.Std.loop, whose WP theorem proves a
postcondition without a counted trace. A recurrence over a substitute body cannot establish
source cost. Add a reusable finite trace which observes the actual body and prove its value
projection to that exact loop definition before using it for Rust lane counts.

Spec: for any state/result types, deterministic Result-valued body and state charge in Nat,
a finite trace has continuation edges observed from body, then success or failure. Count the
charge at every visited body state, including the terminal/failing state. Prove result projection
and uniqueness of result/count. Divergence has no finite trace and is excluded from this finite
certificate; do not claim universal termination. Joint witnesses cover zero-continuation success,
continuation success, immediate and later failure. An inductively proved countdown family with
charge1 on continuation states and0 on termination has n charged visits; instantiate n=32.
This is a generic Aeneas model theorem, not a Rust extraction, constructor, lane-denotation,
resource-improvement or hardware claim. Actual loop/source cost remains a following task.

Plan: APEIRON-GPU-PREFILL-W4-T1, Sol writer, Luna independent verifier, owns only
proof/native/LoopTrace.lean and proof/native/loop_trace_gate.py in its isolated task worktree.
Gate-first task: `python3 proof/native/loop_trace_gate.py --controls`, one540s budget including
lane wait and all descendants; Lean12GiB/no swap/one thread. Bind the actual pinned
Aeneas/Std/Primitives.lean, dependency/model sources, tool identities and owned artifacts;
validate source identity before private COW use, then freshly compile/audit all owned declarations
under the closed cone. Change the actual private loop's done branch from `ok x` to `fail .panic`:
mutant model compiles but named projection/success witness must fail; restore exact bytes,
recompile model and recheck proof. No shared model or inference changes. A malformed proof or
unrelated build failure is not that negative control. Absent gate starts red; runtime feasibility
within540s remains unmeasured until the full invocation exits. Root owns merge and cache lane.

SCOPE: LoopTrace.lean and loop_trace_gate.py; finite trace of actual Aeneas loop and projection.
REGISTER: generic model/count refinement infrastructure; no Rust or GPU cost certificate.
CORPUS: Aeneas/Std/Primitives.lean loop and Aeneas/Std/WP.lean loop.spec_decr_nat read; no existing trace/count theorem found.
REFERENCE: /home/socol/Workspace/apeiron/target/aeneas-model/lean/Aeneas/Std/Primitives.lean.
GATE: MISSING; gate-first python3 proof/native/loop_trace_gate.py --controls;540s whole invocation.
GRADE: proved target for generic finite loop model; Rust/source-resource and performance open.

Independent Luna accepts this generic finite scope in
/tmp/opencode/native-loop-trace-plan-review.md. Result is an ITree representation; preserve the
exact failure error and failing-state charge, rather than replacing it with Option. A private
lean_lib NativeLoopTrace rooted at LoopTrace supports the focused
`lake build NativeLoopTrace:leanArts` import-closure gate. It freshly rebuilds the actual
Primitives import after every mutation/restoration. The actual loop source SHA256 is
702946866c83fa0bb8b9d2a248c14164d38aa15e1eebd67caf4f39b0076d0ff8.
No generic finite witness proves termination, reachability or a cost for Rust exact_dot.

The first finite-trace gate exits0 in13.553s but remains unaccepted: independent Luna identifies
omitted AeneasMeta sources in the prepared9574-file manifest, including the directly imported
BvEnumToBitVec helper. Retain that receipt as provenance-incomplete. Root prepares a supplemental
manifest from the immutable pinned Aeneas git blobs, including AeneasMeta.lean and every
AeneasMeta/**/*.lean file; compare both named-cache and main model bytes to those blobs. A
corrupted prepared-source digest must reject; restoring the expected digest passes. Sol then
extends only loop_trace_gate.py to validate the supplemental source population, freshly rebuild
and bind the imported helper's artifact, and rerun the complete projection/semantic-control gate.
The finite Trace contract and theorem source stay unchanged. Luna reviews the corrected frozen
candidate. No performance/admission claim follows from either receipt.

SCOPE: target/native-step-model-cache/meta-source-identity.json preparation; root-owned cache provenance only.
REGISTER: source/tool dependency identity; no proof or cache receipt substitutes for fresh checking.
CORPUS: pinned505b6ca35217e7be5c96c3e2f8045edfbdf47291 backends/lean/AeneasMeta sources.
REFERENCE: /home/socol/Workspace/aeneas; pinned git blobs, not mutable HEAD.
GATE: timeout60 supplemental source-identity preparation/check; injected digest mismatch rejects/restores.
GRADE: tested prepared identity target; generic trace acceptance requires its fresh repaired gate.

Finite loop-trace prerequisite accepted at generic model scope on main624fd7d. The fresh main
`python3 proof/native/loop_trace_gate.py --controls` exits0 in15.138s of540s; independent Luna
accepts the corrected dependency provenance and unchanged finite contract. All40 declarations
audit; positive actual-primitive compile, compiling done-to-panic mutant/named projection red,
exact restoration and final compile/audit pass. Source population9592 includes the pinned18-source
Meta supplement and freshly rebuilt/bound imported elaborator. Earlier provenance-incomplete
and preparatory failures remain retained. This result supplies a finite counted-value projection,
not a charge-to-Rust correspondence or a native GPU cost certificate. Actual-lane and Result.ok
source-link feasibility reports identify the next scoped tasks; neither runs a proof gate.
Full receipts, source identities/texts, failures and reviews are retained in
artifacts/gpu-prefill/native-prerequisites-acceptance.json.
