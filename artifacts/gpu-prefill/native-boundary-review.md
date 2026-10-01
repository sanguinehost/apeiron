# Native Q8 GPU baseline boundary review

Decision: the initial Apeiron-owned projection baseline may proceed under the user's explicit one-time bootstrap authorization. Apeiron owns the shader, minimal Vulkan driver, fixtures, and subsequent source/refinement work. Fold and llama.cpp are read-only design and comparison sources. This decision covers no later tuning, serving integration, hybrid recurrence, or whole-model speed claim. Later inference optimization remains subject to a registered `tools/dispatch/admit.py --launch` contract; `tools/dispatch/contracts.json` is empty.

Numerical scope: `src/engine/kern/gemm.rs::exact_dot` is the CPU observation. Its 32 signed weight codes and 32 signed activation codes yield a signed block dot. `src/engine/kern/pow2.rs::f16_split` gives a signed mantissa and nonnegative exponent for finite binary16; `src/engine/value.rs::round_shift_32` adds `2^31` once, then shifts arithmetically by 32. The native two-`u64` accumulator must perform each block's `d.checked_mul(S)?.checked_mul(dot)?` and each subsequent `checked_add` in the same order and report signed overflow; modular wrap is not parity. The final round and `i64` narrowing must match exactly. Non-finite scales and malformed geometry have explicit refusal paths. The CPU's unchecked rounding addition can panic for `acc > i128::MAX - 2^31`; that population is excluded from baseline parity and must be rejected by the fixture domain before dispatch rather than assigned a fabricated output.

The `e ≤ 58` bound covers every positive-`u64` result of the actual `act_scale` body: the accepted `ActScaleResource.scale_positive` source theorem gives `s = 2^e` with `e ≤ 58`, because `ceil(u64::MAX / 127) < 2^58`. The loop guard of 62 is not a reachable output bound. `QuantAct::from_parts` can construct scales outside that image, so the fixture gate must validate `e ≤ 58` and the power-of-two shape independently, then refuse or mark unsupported manual scales rather than inferring the constructor invariant. Under that bound, the per-block term remains below signed `i128` magnitude using `|m| ≤ 2047`, `k ≤ 37`, 32 signed byte products, and the stated activation bound; sequential sum and final-round overflow remain checked separately. Negative zero weight scale is numerically zero after `f16_split` and remains finite.

`SPIQ8V01` names a little-endian byte contract: eight magic bytes, one `u32` case count, then four `u32` geometry/id fields per case; W, DM, X, E, expected values and refusal flags follow at the specified lengths. `UINT32_MAX` is reserved for non-finite DM and zero activation scale as specified. The producer and driver must check each byte-count multiplication and total payload length before reads, reject truncated/trailing payloads, and preserve signed lane bits in the 32-bit packed words. Golden values are authoritative only where refusal is zero. A mutation of the real shader dot update must make device parity fail and restoration must pass. `spirv-val` checks syntax, while actual device parity remains tested evidence; neither proves emitted SPIR-V denotation.

The current `Spirv.Dp4a` model uses mixed-signed Q15 words and explicitly excludes the executing CPU body, emitted-module correspondence, and float epilogue. It does not prove this signed-dot, two-limb Q32 baseline. Compiler, driver, device, host FFI, and buffer lifetime remain trusted/tested boundaries. Source extraction, canonical refinement, cost correspondence, production optimization admission, and matched llama.cpp GPU prefill are open gates.

Gate: a 60-second document/domain check must find native ownership, the exact CPU source, signed/two-limb/rounding/refusal constraints, `SPIQ8V01`, and the later admission boundary. Removing native ownership or later admission from a disposable in-memory copy must reject; the unchanged documents must pass.

Observed gate in `/home/socol/Workspace/apeiron`, exit `0`:

```sh
timeout 60s python3 - <<'PY'
from pathlib import Path
p=Path('docs/epics/gpu-prefill.md').read_text()
r=Path('docs/epics/resource-guided-evolution.md').read_text()
v=Path('artifacts/gpu-prefill/native-boundary-review.md').read_text()
checks={
'native ownership': 'Apeiron owns new GPU kernels, their emitted SPIR-V, runtime and formal refinements.' in p,
'reference only': 'llama.cpp and\nFold are read-only implementation references and comparators.' in p,
'actual CPU body': 'src/engine/kern/gemm.rs::exact_dot' in p,
'numerical contract': all(x in p for x in ('signed-by-signed integer dots','two-limb i128','one Q32.32 rounding','sequential checked accumulation/refusal')),
'fixture ABI': all(x in p for x in ('SPIQ8V01','UINT32_MAX','APEIRON_NATIVE_Q8_FIXTURE')),
'scoped bootstrap': 'only the initial native baseline before optimisation registration' in p,
'later admission': 'No later tuning/optimisation worker is exempt from tools/dispatch/admit.py --launch.' in p,
'resource ownership': 'Fold source-derived counts and historical receipts remain reference evidence, not the execution' in r,
'review finding': 'QuantAct::from_parts' in v and 'ActScaleResource.scale_positive' in v,
}
for key,ok in checks.items(): print(f'{key}: {"PASS" if ok else "FAIL"}')
assert all(checks.values())
for key,bad in [('missing native ownership',p.replace('Apeiron owns new GPU kernels, their emitted SPIR-V, runtime and formal refinements.','')),('missing later admission',p.replace('No later tuning/optimisation worker is exempt from tools/dispatch/admit.py --launch.',''))]:
    reject=not (('Apeiron owns new GPU kernels, their emitted SPIR-V, runtime and formal refinements.' in bad) if key=='missing native ownership' else ('No later tuning/optimisation worker is exempt from tools/dispatch/admit.py --launch.' in bad))
    print(f'{key} control: {"REJECTED" if reject else "MISSED"}')
    assert reject
print('NATIVE_DOCUMENT_DOMAIN_PASS')
PY
```

```text
native ownership: PASS
reference only: PASS
actual CPU body: PASS
numerical contract: PASS
fixture ABI: PASS
scoped bootstrap: PASS
later admission: PASS
resource ownership: PASS
review finding: PASS
missing native ownership control: REJECTED
missing later admission control: REJECTED
NATIVE_DOCUMENT_DOMAIN_PASS
```

This checks the written boundary and the control's sensitivity to two missing clauses. It does not execute a kernel or establish numerical parity.

Lead reran the 60-second document gate: nine exact domain/ownership markers, removal of native ownership and later optimisation admission each red, unchanged restoration green; exit0. Marker list: ['Apeiron owns new GPU kernels', 'read-only implementation references', 'exact_dot', 'signed-by-signed', 'two-limb i128', 'rounding-add panic boundary', 'SPIQ8V01', 'UINT32_MAX', 'later optimisation remains admitted'].
