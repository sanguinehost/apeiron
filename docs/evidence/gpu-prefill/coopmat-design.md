# Signed i8×i8→i32 cooperative-matrix Q8 tile: design analysis

Status: read-only design analysis, grade **anchor**. No kernel, driver, shader or benchmark is
written or run here. No throughput, admission or whole-prefill claim. Task
`APEIRON-GPU-PREFILL-W5-T4`; owned artifact `docs/evidence/gpu-prefill/coopmat-design.md`. Worktree
revision `4b5eff58eb0c072ad88ad2d937c60d757ff9e4a2`; repository files are cited at worktree-absolute
paths.

```text
SCOPE:      docs/evidence/gpu-prefill/coopmat-design.md only; design analysis of a signed
            SINT8×SINT8→SINT32 cooperative-matrix Q8 block tile. Completion: every citation resolves.
REGISTER:   read-only design/planning; no implementation, no numerical or contract change.
CORPUS:     src/spirv/q8_tile.comp, gemm.rs::exact_dot, docs/epics/gpu-prefill.md,
            docs/kernel-boundary.md, retained capability probe read; no canonical theorem changed.
REFERENCE:  /home/socol/Workspace/apeiron/src/spirv/q8_tile.comp;
            /tmp/opencode/native-q8-hardware-directions.md.
GATE:       structural citation-resolution check, <=540 s, no GPU run.
GRADE:      anchor; capability properties are not measured issue rates.
```

## 1 · Hardware basis

Retained read-only probe `/tmp/opencode/native-q8-hardware-directions.md` (not re-run here; no
`vulkaninfo` probe is treated as fresh evidence):

- Device: NVIDIA GeForce RTX 5080, driver `2580660800` (615.71.09), Vulkan 1.4.351
  (`/tmp/opencode/native-q8-hardware-directions.md:12`).
- Features: `shaderInt64=1 shaderIntegerDotProduct=1 cooperativeMatrix=1` (`…:13`).
- Extensions: `VK_KHR_cooperative_matrix rev=2` (`…:15`), `VK_NV_cooperative_matrix2 rev=1`
  (`…:16`).
- Signed shapes (`…:19`, `…:20`):

```text
COOP 7  M=16 N=16 K=32 A=3 B=3 C=5 R=5 sat=0 scope=3
COOP 9  M=16 N=8  K=32 A=3 B=3 C=5 R=5 sat=0 scope=3
```

`A=3`/`B=3` are SINT8 operands, `C=5`/`R=5` are the SINT32 accumulator/result, `scope=3` is
subgroup scope and `sat=0` is non-saturating. The enum values resolve in the installed header:
`VK_COMPONENT_TYPE_SINT8_KHR = 3` (`/usr/include/vulkan/vulkan_core.h:13499`),
`VK_COMPONENT_TYPE_SINT32_KHR = 5` (`…:13501`), `VK_SCOPE_SUBGROUP_KHR = 3` (`…:13536`); the
property struct is `VkCooperativeMatrixPropertiesKHR` (`…:13544`); the extension name is
`VK_KHR_cooperative_matrix` (`…:13493`) at spec version 2 (`…:13492`). The unsigned entries
`VK_COMPONENT_TYPE_UINT8_KHR = 7` (`…:13503`) and `VK_COMPONENT_TYPE_UINT32_KHR = 9`
(`…:13505`) are the u8/u8→u32 shapes only; the retained list reports no mixed-signedness
cooperative-matrix entry (`/tmp/opencode/native-q8-hardware-directions.md:23`). Both signed shapes
are SINT8×SINT8→SINT32, i.e. the Apeiron signed-code domain, so no recoding is required; this is the
same design choice the epic records for the packed native kernel, which removes Fold's unsigned
recoding (`docs/epics/gpu-prefill.md:18`).

Emitted SPIR-V op family. The extension is `SPV_KHR_cooperative_matrix`, the capability is
`CooperativeMatrixKHR`, the cooperative matrix type is
`OpTypeCooperativeMatrixKHR (comp, scope, rows, cols, use)` (`opcode 4456`) and the multiply-add is
`OpCooperativeMatrixMulAddKHR (ty, res, a, b, c)` (`opcode 4459`). These opcode/extension identities
are read from the independent Fold emitter
`/home/socol/Workspace/fold/modelfold/src/simt/gemv.rs:1046`, `…:1049`, `…:1052`, which also uses
subgroup scope (`…:1055`). The `comp` operand is the component scalar type: signed 8-bit for A/B and
signed 32-bit for C. This lane emitted no SPIR-V; the concrete component types, operand order and
layouts must be read from `spirv-dis` output at the implementation gate (§4), not assumed here.

Capability properties are not measurements. The probe reports availability flags and a property
table, not issue rates or latency (`/tmp/opencode/native-q8-hardware-directions.md:23`), and its
decision line states the cooperative tile is "a supported hardware candidate, not an implemented or
measured improvement" (`…:3`). No speed ratio follows from a shape existing.

## 2 · Tile mapping

- K32 is exactly one Apeiron Q8 block. The block loop is
  `src/spirv/q8_tile.comp:71`; one block reads eight packed weight quads `w[(b*8+q)*M+row]`
  (`…:98`) and eight activation quads `x[(token*blocks+b)*8+q]` (`…:102`), i.e. 32 signed i8 lanes
  per side. The full block is one `OpCooperativeMatrixMulAddKHR`; no cooperative accumulation along
  K is needed.
- M indexes output rows and N indexes prompt tokens. The row axis is
  `rowbase = gl_GlobalInvocationID.x*2` (`…:62`) with weight address `(b*M+row)*2` (`…:78`); the
  token axis is `tokenbase = gl_GlobalInvocationID.y*4` (`…:63`) with activation address
  `(token*blocks+b)*8+q` (`…:102`). Output is `out64[token*M+row]` and `refusal[token*M+row]`
  (`…:147`, `…:149`). For one block, A = weights indexed `[row][block-lane]`, B = activations
  indexed `[block-lane][token]`, C[i][j] = Σ_k A[i][k]·B[k][j].
- Shape choice: M16N8K32 yields one 16-row × 8-token C tile per subgroup; M16N16K32 doubles the
  token extent. Both were reported; no issue-rate difference is claimed. The runtime may select per
  shape.
- Tails (`M % 16`, `tokens % 8` or `% 16`): zero-pad the signed code lanes of padded operand rows
  and columns, and mask the C store. The current kernel already zero-pads — `wr[r] = row < p.M ? … : 0`
  (`…:98`) and `xt[t] = token < p.tokens ? … : 0` (`…:102`) — and skips out-of-range cells before
  load and before store (`…:117`). A zero signed code contributes zero to the i32 dot, so padding
  cannot perturb a live cell; the store is still masked to live cells.
- Dispatch: the baseline dispatches `ceil(M/16), ceil(tokens/32)` with local8×8 and RT2BT4 per
  invocation (`docs/epics/gpu-prefill.md:114`). A cooperative variant dispatches one subgroup per
  (row-tile × token-tile) and iterates `p.blocks` (`src/spirv/q8_tile.comp:71`); a 32-lane subgroup
  is the natural workgroup for `VK_SCOPE_SUBGROUP_KHR`.

## 3 · Correctness-preservation constraint

Per-row weight scales and per-token activation scales differ between blocks, so two blocks with
unequal scale **must not** share one unscaled i32 accumulator. Today each block's dot is scaled and
folded separately:

- coefficient `coefficient = int64_t(mantissa[r]) * int64_t(dots[c])`
  (`src/spirv/q8_tile.comp:122`);
- shift `shift = weightExp[r] + actExp[t]` (`…:121`);
- ordered two-limb addition `addChecked(acc[c], shiftedTerm(coefficient, shift))` (`…:123`), with
  sticky `REFUSE_I128` on signed overflow (`…:124`) and non-finite refusal (`…:118`);
- final Q32.32 round-to-nearest with the `+2^31` bias (`…:137`, `…:140`), an i64 fit check
  (`…:145`) and `REFUSE_I64` (`…:146`); the value is stored only when `flags[c] == 0` (`…:136`)
  and the refusal flag is always stored (`…:149`); the two-limb helper is `shiftedTerm` (`…:41`).

A cooperative tile therefore replaces **only** `dots[c]` for one block: read the i32 cell C[i][j],
then feed the unchanged per-block term into the unchanged sequential i128 accumulator. `acc[c]`,
its block order, the shift, the rounding/narrowing and every refusal flag stay outside the
cooperative operation. This is the wave contract: K32 aligns with one Q8 block, per-block scaling is
preserved, and unequal-scale blocks never share an unscaled accumulator
(`docs/epics/gpu-prefill.md:380`).

Per-block i32 bound. Each signed code is in [-128,127]; each product magnitude is at most 16384, so
a 32-lane block dot satisfies |dot| ≤ 32·16384 = 524288 < 2^31 = 2147483648, which fits the i32 C
tile started at zero (`/tmp/opencode/native-lane-instantiation-feasibility.md:13`; the same result
at `/tmp/opencode/native-q8-hardware-directions.md:31`). Because the exact sum fits, the
non-saturating property (`sat=0`) and the signed SINT8→SINT32 result are sufficient in range; the
value must equal the 32-lane signed dot the actual CPU `exact_dot` oracle produces, which is the
parity the epic already requires (`docs/epics/gpu-prefill.md:96`).

Open equivalence obligation. The hardware i32 cell must equal the sum of the eight packed
`dotPacked4x8EXT` results the current kernel computes (`src/spirv/q8_tile.comp:107`), and both must
equal `exact_dot`. That is a per-cell test against real-constructor fixtures, not a consequence of
the shape existing.

## 4 · Load layout / signedness risks

- Operand role and layout are declared, not inferred. Cooperative matrix loads/stores carry an
  explicit layout and stride; the Fold emitter names `OpTypeCooperativeMatrixKHR`,
  `OpCooperativeMatrixLoadKHR`, `OpCooperativeMatrixStoreKHR` and `OpCooperativeMatrixMulAddKHR` as
  a family (`/home/socol/Workspace/fold/modelfold/src/simt/gemv.rs:1046`, with the multiply-add at
  `…:1049`). A swapped A/B role, a wrong transpose, a wrong row-/column-major choice or a wrong
  stride silently changes the product. The current ABI stores weights and activations as packed
  `uint w[]`/`uint x[]` (`src/spirv/q8_tile.comp:9`, `…:11`) while cooperative loads consume a
  scalar-typed pointer; the pack/unpack boundary is a layout risk.
- Signedness is a type-level property. A/B must be the signed 8-bit component type and C the signed
  32-bit type; loading an i8 code as unsigned flips every negative code. The device exposes no
  mixed-signedness entry (`/tmp/opencode/native-q8-hardware-directions.md:23`), so both operands
  must be signed.
- Negative-fixture precedent, not an Apeiron result. The retained Fold note records an isolated
  `OpCooperativeMatrixMulAddKHR` M16N16K32 u8×u8→u32 kernel that passed `spirv-val` and matched a
  CPU f64 reference only with real 8-bit SSBOs (Route B, `check_coopmat`), while its packed-`uint32`
  pointer route validated but produced values "off by 2-3 orders of magnitude on real hardware"
  (`/home/socol/Workspace/fold/hashtrinity/GPU_PRIMITIVES.md:172`, `…:173`, `…:174`, `…:175`). That is
  unsigned Fold, not Apeiron's signed Q8 source or its exact epilogue; it transfers nothing and is
  not cited as an Apeiron result. It is useful only as a negative-fixture idea: a deliberately wrong
  pointer/layout or signedness variant must produce an attributable mismatch against the CPU oracle.
- Required at implementation: disassemble the actual emitted module and check (a)
  `OpTypeCooperativeMatrixKHR` component types are signed 8-bit for A/B and signed 32-bit for C,
  (b) `OpCooperativeMatrixMulAddKHR` operand order, (c) load/store layout and stride, (d) subgroup
  scope; then compare every live cell on device output to the CPU oracle. `spirv-val` supplies
  syntax only, not semantics (`docs/epics/gpu-prefill.md:50`).

## 5 · Admission and evidence plan

Ordered prerequisites; none is satisfied by this document.

1. Measured baseline, `APEIRON-GPU-PREFILL-W5-T1` (`docs/epics/gpu-prefill.md:392`): device-compute
   timestamps with `timestampPeriod`/valid-bit checks and host submit→wait reported separately. No
   cooperative comparison exists without this instrument.
2. Matched comparator, `APEIRON-GPU-PREFILL-W5-T2` (`…:393`): read-only llama.cpp CUDA Q8
   projection shapes derived from the real GGUF; no GPU benchmark.
3. Actual source-cost/refinement registration: executing-source semantics, refinement of the emitted
   body to the canonical object, and a cost model with a joint reachable witness. The epic records
   the actual native emitted-body/canonical/source-cost correspondence as open (`…:97`) and
   optimization admission as closed to dispatch until a registration is accepted (`…:98`).
4. `python3 tools/dispatch/admit.py --launch <proposal.json>` in the task worktree. `admit.py` is a
   compatibility entry whose admission and launch decisions execute only in the Rust core
   (`/home/socol/Workspace/apeiron/tools/dispatch/admit.py:2`); the rule requires the launcher before
   an inference-optimization implementation worker
   (`/home/socol/Workspace/apeiron/AGENTS.md:343`). The registry is empty — `"contracts": {}`
   (`/home/socol/Workspace/apeiron/tools/dispatch/contracts.json:3`). This worktree does not carry
   `tools/dispatch/`; it is untracked in the main checkout (`docs/epics/gpu-prefill.md:339`).
   The contract must carry a joint reachable witness, universal preservation of value and refusal,
   strict improvement in the registered model, and rejecting controls. A capability flag, a shape
   table or this design note does not satisfy it, and the registry stays empty
   (`docs/epics/gpu-prefill.md:419`).
5. Implementation, only after 1–4: at most three owned files (cooperative shader, private gate
   variant, driver mode). Its gate compiles and validates the emitted SPIR-V, compares every
   successful and refused cell to the actual CPU `exact_dot` on real-constructor and boundary
   fixtures, forces an attributable operand-sign/layout or dot-update mutation red, restores green,
   and runs within the 540 s whole-invocation ceiling
   (`/tmp/opencode/native-q8-hardware-directions.md:33`).
6. Then matched multi-round device timing with GPU timestamps and end-to-end transfer/launch timing;
   the source/cost model's logical operations are not hardware latency
   (`/tmp/opencode/native-q8-hardware-directions.md:33`).

GPU timing is exclusive to W5-T1 in this wave (`docs/epics/gpu-prefill.md:385`); this lane ran no
Vulkan, CUDA or llama.cpp command and produced no timing.

Residual, stated plainly: no implementation, no emitted SPIR-V, no throughput or latency claim, no
admission. Grade **anchor**. GPU/SPIR-V remains an explicit trust-boundary exclusion
(`docs/kernel-boundary.md:486`). The retained baseline is a correctness campaign, not a latency
measurement (`artifacts/gpu-prefill/native-baseline.md:18`, `…:19`); the gap between a capability
property and an issue rate remains open.
