# Native GPU prerequisite acceptance

Decision: accept scalar narrowing and actual-constructor fixture tasks at their stated grades.
Whole-native source/cost, optimization admission, serving and GPU speed comparison remain open.
Full logs, digests, tool identities and independent verdicts are retained in
`native-prerequisites-acceptance.json`; measured durations below are gate durations, not inference latency.

| result | grade | fresh gate exit / duration |
|---|---|---|
| Actual pinned std i64::try_from(i128), inclusive endpoints and signed overflow errors | proved, scalar scope | narrow_gate.py --controls:0 /20.400s |
| Actual quant_act fixtures:12cases,14581exact outputs,5refusals | tested on RTX5080 | timeout540 quantizer_gate.py:0 /289.544s including Rust lane wait |
| Original synthetic14case fixture, unchanged digest | tested on RTX5080 | timeout540 q8_gate.py:0 /7.399s |
| Workspace all-feature tests | tested | 0 /281.682s |
| Strict all-target/all-feature Clippy | tested | 0 /2.948s |
| Format check | tested | 0 /0.454s |

Narrowing controls corrupt the LLBC error-field binding and each generated strict range guard.
Both guard mutants compile, fail their named endpoint theorem, and restore to green. The
namespace-local error alias preserves the selected nightly's six IntErrorKind constructors;
shared Unit-valued helpers, tools and pins are unchanged. This is not the exact_dot caller proof.

The actual quantizer supplies every serialized activation, and actual exact_dot supplies every
GPU expected output/refusal. Reconstruction checks scales/codes/dequantization only. The new
fixture is949340bytes, SHA25649f14cc4d2a219f278353245b454ed31ef7da10af429c4ed122beadc57b3b958.
Coverage includes signed ties, scale transitions, zero/mixed/high scales, nonzero row/token tails,
nonfinite weights, both signed accumulator overflows, final narrowing and large cancellation.
i64MIN constructor success and i64MAX/mixed constructor refusal remain separate host witnesses.
The real quantizer tie mutation compiles and fails its named assertion; shader dot mutation
produces14577mismatches. Both restored producers and device outputs pass.

The exact_dot production extraction currently stops at Aeneas Step.forward_overflowing. A
reviewed private pinned-tool compatibility task follows; no inference body or admission registry
changes. CPU source cost must include early refusals; GPU sticky refusal continues later work.
The existing Word32/mixed-signed fragments do not prove this OpSDot/int64 shader. Native serving
is absent. The planned comparison uses the same Qwen3.5 Q8_0 GGUF and raw200/512token IDs, fresh
state, matched timing boundaries, explicit GPU offload and five quiet paired rounds. Current
llama-bench random-token timing is a shape proxy, not that exact-token campaign.

Excluded: universal quantizer/actual exact_dot refinement, emitted-module/source-cost equivalence,
code generation, GPU ABI/driver correctness, serving state and any GPU throughput ratio.
Failed preparation budgets/cache/link runs are retained as failures, not semantic controls.
