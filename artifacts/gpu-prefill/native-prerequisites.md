# Native GPU prerequisite acceptance

Decision: accept scalar narrowing, actual-constructor fixtures and extracted terminal-branch tasks at their stated grades.
Whole-native source/cost, optimization admission, serving and GPU speed comparison remain open.
Full logs, digests, tool identities and independent verdicts are retained in
`native-prerequisites-acceptance.json`; measured durations below are gate durations, not inference latency.

| result | grade | fresh gate exit / duration |
|---|---|---|
| Actual pinned std i64::try_from(i128), inclusive endpoints and signed overflow errors | proved, scalar scope | narrow_gate.py --controls:0 /20.400s |
| Real exact_dot exhausted tail: round, narrowing and model Result.ok | proved, terminal branch | step_gate.py --actual-exact-dot --controls:0 /439.032s |
| Actual quant_act fixtures:12cases,14581exact outputs,5refusals | tested on RTX5080 | timeout540 quantizer_gate.py:0 /289.544s including Rust lane wait |
| Original synthetic14case fixture, unchanged digest | tested on RTX5080 | timeout540 q8_gate.py:0 /7.399s |
| Workspace all-feature tests | tested | 0 /281.682s |
| Strict all-target/all-feature Clippy | tested | 0 /2.948s |
| Format check | tested | 0 /0.454s |

Narrowing controls corrupt the LLBC error-field binding and each generated strict range guard.
Both guard mutants compile, fail their named endpoint theorem, and restore to green. The
namespace-local error alias preserves the selected nightly's six IntErrorKind constructors;
shared Unit-valued helpers, tools and pins are unchanged. The terminal-branch proof below links this conversion to the extracted caller.

The actual quantizer supplies every serialized activation, and actual exact_dot supplies every
GPU expected output/refusal. Reconstruction checks scales/codes/dequantization only. The new
fixture is949340bytes, SHA25649f14cc4d2a219f278353245b454ed31ef7da10af429c4ed122beadc57b3b958.
Coverage includes signed ties, scale transitions, zero/mixed/high scales, nonzero row/token tails,
nonfinite weights, both signed accumulator overflows, final narrowing and large cancellation.
i64MIN constructor success and i64MAX/mixed constructor refusal remain separate host witnesses.
The real quantizer tie mutation compiles and fails its named assertion; shader dot mutation
produces14577mismatches. Both restored producers and device outputs pass.

Private pinned Step compatibility now translates the unchanged actual exact_dot body. All102
generated declarations audit in the closed cone. StepRefinement.exhausted_tail, tail_in_range,
tail_below_min and tail_above_max prove the terminal round/conversion/Option branch. The joint
witnesses cover terminal model states; they do not prove constructor or loop-prefix reachability.
Result.ok remains an Aeneas model, with its actual Rust source refinement open. Lower/upper guard
mutants compile and break the named endpoint proofs; model Err-to-panic mutation compiles and
breaks refusal; a registry method-pair deletion rejects with missing forward_overflowing.
Restoring each source, recompiling the complete generated body and rechecking the
terminal-branch proof passes. The prior
434.615s run is rejected because its registry mutation exited0; the corrected fresh main run
is439.032s. The named source-validated model cache accelerates fresh generation only. Production
Rust, shared tools/models, pins and admission registry remain unchanged.

CPU source cost must include early refusals; GPU sticky refusal continues later work.
The existing Word32/mixed-signed fragments do not prove this OpSDot/int64 shader. Native serving
is absent. The planned comparison uses the same Qwen3.5 Q8_0 GGUF and raw200/512token IDs, fresh
state, matched timing boundaries, explicit GPU offload and five quiet paired rounds. Current
llama-bench random-token timing is a shape proxy, not that exact-token campaign.

Read-only device inspection confirms packed signed4x8 OpSDot acceleration and signed cooperative
M16N8K32/M16N16K32 support. These matrix tiles and guarded power-of-two factoring are research
candidates. Their implementation, source/resource correspondence and timings remain open.

Excluded: universal quantizer/whole-loop exact_dot refinement, actual Rust Result.ok, prefix
reachability, emitted-module/source-cost equivalence,
code generation, GPU ABI/driver correctness, serving state and any GPU throughput ratio.
Failed preparation budgets/cache/link runs are retained as failures, not semantic controls.
