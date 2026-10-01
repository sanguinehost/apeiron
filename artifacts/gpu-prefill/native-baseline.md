# Native Q8 baseline acceptance

Decision: retain the Apeiron-owned baseline on main. Grade: **tested** on the pinned synthetic
`QuantAct::from_parts` fixture domain. Independent reviewers: Luna for the Sol-written CPU producer;
Sol for the Luna-written native shader, driver and gate. Their verdicts and complete GPU output are
embedded in `native-baseline-acceptance.json`.

| artifact | behavior |
|---|---|
| src/spirv/q8_tile.comp | signed packed OpSDot; local8x8; RT2BT4 per invocation; ordered checked i128; Q32 rounding and i64 refusal |
| tools/gpu/q8_driver.c | native Vulkan buffers/dispatch; extent/device-limit checks; shader-write to host-read barrier; no Fold runtime linkage |
| tools/gpu/q8_gate.py | actual CPU fixture parity; compiled shader subtraction mutation; restoration |
| src/engine/kern/native_gpu_probe.rs | real exact_dot/f16_split observations; checked synthetic constructor inputs; test-only hook in gemm.rs |

## Acceptance

`timeout 540 env PYTHONDONTWRITEBYTECODE=1 python3 tools/gpu/q8_gate.py --fixture /tmp/opencode/native-q8-fixture-main.bin`
from `/home/socol/Workspace/apeiron` exits0 in5.271s on RTX5080, driver615.71.09, Vulkan1.4.351.
Fourteen cases yield14,678 exact successful outputs and32 exact refusals. The real dot-update
mutation compiles and validates, yields13,973 mismatches, then restored source passes all cases.
This interval measures a correctness campaign, not kernel latency or throughput.

Regenerate the pinned binary using the ignored test
`engine::kern::gemm::native_gpu_probe::native_q8_prefill_fixture` with
`APEIRON_NATIVE_Q8_FIXTURE` naming its output. SHA256 is
`573a0e1c020dba300ac0dd58693fa5e31b8212e96881e9793c388e42892d5fa2` (946,420bytes).
The CPU missing-output control exits101; restoration exits0. It controls fixture IO;
the shader mutation supplies the numerical negative control.

Full `cargo test --workspace --all-features --locked` exits0 within540s (410 library tests pass,
33 ignored; seven doctests pass). Strict workspace/all-target clippy exits0 after two documentation
backtick fixes; format check exits0. Full test output was truncated by the tool; the observed exit
and result are retained, not presented as a complete transcript. New source/tool/fixture identities,
GPU commands and outputs, independent reviews and the clippy red/restoration output are in the JSON.

## Next bounded proof task

Register: extraction/model prerequisite; no inference implementation change.
Question: can the actual Rust `TryFrom<i128> for i64` body and its current error representation
translate with no unmodelled axiom, then support a closed-cone range/refusal theorem?
Scope: select at most three extraction/model/refinement files after the gate-first review;
keep source/proof/runtime numeric meanings unchanged. Gate: **MISSING**, define the focused actual
conversion extraction/build and lower/upper-bound mutation/restoration before dispatch. Budget540s,
Lean lane serialized, one thread,12GiB/no swap. Model/pin changes require independent review.

The selective source extraction succeeds in Charon but Aeneas fails constructing opaque
`core::num::error::TryFromIntError`. The current Rust body stores `IntErrorKind::NegOverflow` or
`PosOverflow`; the current model's unit error representation does not yet carry that source link.
Opaque extraction emits an unmodelled conversion axiom and is not proof. Exact diagnostics are
retained in the JSON; the read source is `core/src/convert/num.rs` in the pinned nightly sysroot.

After conversion/source closure: add actual `quant_act` constructor fixtures, check emitted-body
and canonical numeric correspondence, prove refusal preservation and the registered cost relation,
then accept a production launcher contract. Subsequent tiling/fusion/residency optimization uses
that contract. Serving integration and five quiet matched llama.cpp GPU whole-prefill rounds follow.

## Boundaries

The broad synthetic stress fixtures exercise arithmetic refusals beyond established production
constructor reachability. CPU rounding-add overflow is explicitly excluded. Arbitrary direct C
DM/E payloads are outside the validated fixture path. Compiler/codegen, Vulkan/FFI/driver/device,
source/emitted-body/cost proof, hardware gain, whole-model state continuation and shared Lean
barriers remain open at their own grades. The GPU epic is not closed by this baseline.
