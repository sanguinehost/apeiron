# Verifiable SPIR-V subset for the integer GPU lane

Merged epic record: `intent/gpu-spv.md`, `spec/gpu-spv.md`, `plan/gpu-spv.md` (one file per epic; see `docs/README.md`).

## Intent — Verifiable SPIR-V subset for the integer GPU lane

- **Epic:** `APEIRON-GPU-SPV`
- **Owner:** Apeiron lead
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (the `Nous`/
  `Growable`/`Numeric` lanes and this documentation pass; see `docs/archive/handoff-openai.md`)
- **Status:** draft (owner decision recorded below; spec and plan follow this intent)
- **Spec:** [`docs/epics/gpu-spv.md`](gpu-spv.md) (planned by W0-T1)
- **Plan:** [`docs/epics/gpu-spv.md`](gpu-spv.md) (planned by W0-T1)

### Problem

The GPU face has no refinement at all: the barrier-free integer DP4A fragment that the served decode
model runs, `gemv_dp4a_module_dyn` (`/home/socol/Workspace/fold/modelfold/src/simt/dp4a.rs`, sha256
`ffe4f6c3…`; consumer `/home/socol/Workspace/fold/bedrock/src/serve/gpu.rs`), is emitted as
`Vec<u32>` by pure safe Rust and consumed by the Vulkan driver, with nothing between the emitted bytes
and the hardware. `docs/kernel-boundary.md` records the row `GPU / SPIR-V | target | not covered`; the
realization row in `docs/direction.md` ("each realization refines the same supported semantics")
is empty for that face.

No mechanized SPIR-V value or memory semantics exists to reuse. The only retrieved formal SPIR-V
artifact is the Klimis et al. structured-control-flow model (POPL 2023), which is a
validation/fuzzing model of control flow, not a value semantics; `spirv-val` is a validator, not a
semantics. Apeiron therefore builds the fragment semantics; that construction is the epic's cost.

The served kernel is small enough to carry: 47 distinct SPIR-V opcodes, no barriers, no atomics, no
subgroups, no workgroup memory, no function calls, no `OpPhi`, no images, one `GLCompute` entry point,
`Logical` + `GLSL450`, and an integer core that is exactly the modular-word object Apeiron already
denotates (`ZMod (2^w)`, `word_mul_denote`/`word_norm_denote` in `proof/corpus/WordRefinement.lean`).
The binding constraint is the float scale epilogue, not the integer core.

### Owner decision in force

- Mechanized verification of a **subset** of SPIR-V is in scope for Apeiron/Nous and the integer
  numerics.
- The driver lowering and the GPU hardware are the named **platform boundary with a runtime-checked
  margin**, in the CompCert pattern: mechanized source and target semantics for the used fragment,
  the assembler/linker/hardware equivalents excluded.
- Song's separate full end-to-end language/OS/prover requirement is untouched by this epic; it is not
  the artifact this subset produces and is not weakened by it.

### Continuity with the canonical object and Nous (user direction)

1. **Canonical object, not a second foundation.** The v0 refinement target is a face-parameterized
   instance of `Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` via the word pattern
   (`word_mul_denote`, `word_norm_denote` in `proof/corpus/WordRefinement.lean`), not a standalone
   `ZMod (2^w)` semantics. A standalone fragment semantics would be a second foundation and is not
   accepted; the CPU face and the GPU face meet at the same canonical object.
2. **Word-level trichotomy caveat.** `Z/2^w` ramifies at `2`. `hyperbolic_decomposition`
   (`proof/corpus/Nous/Model/Decomposition.lean`) is stated with units `r` (`r ^ 2 = σ`, `r * s = 1`)
   and `2` (`2 * h = 1`) — true over `ℚ`, false over `Z/2^w`. Which structures transport (the ring
   equations, `ringMul_assoc`, the fold, the trace form) and which degenerate (idempotent splitting,
   norm-form nondegeneracy, the spinor double-cover metric) is fixed in the spec, not discovered later.
3. **First design question — layout versus representation.** Which face representation the emitted
   Q4/Q8 packing realizes: `OpSUDot`'s signed·unsigned asymmetry is the fingerprint, and the even/odd
   nibble split is the candidate channel structure. Two admissible outcomes: (a) prove the layout
   theorem for the face it actually realizes, or (b) record the refutation and name the emitter
   migration item. Outcome (a) is not asserted to hold.
4. **Fold continuity.** Block accumulation and host-side split-K are the same fold as `token_step_fold`
   (`proof/corpus/Nous/Model/State.lean`) and `memoryStep` (`proof/corpus/Nous/Model/Memory.lean`);
   split-K correctness is the `ringMul_assoc` fold-split lemma, not a second reduction argument.
5. **Nous consumers / naming authority.** The GPU refinement targets the Nous normal forms. The named
   authorities are the planned `Nous.Model.Instances` module (W4-T2 faces) and
   the planned `Nous.Model.Attention` module (`attention_channel_form`, W4-T3), neither present at the
   source revision (see `docs/epics/ai-track-reanchor.md`), alongside the present
   `proof/corpus/Nous/Model/Decomposition.lean` (`hyperbolic_decomposition`) and
   `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean` (`attention_is_spinor_scan`,
   `softmax_attention_is_online_spinor`, `so2_spinor_recurrence`). The names `spinor_scan_preserves_norm`
   and `online_softmax_exact` do not resolve at this revision; the present names are the two above.
   Norm preservation is claimed only on the nondegenerate domain (zero divisors at `Z/2^w`); a
   runtime margin carries the rest.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Full SPIR-V | Admitting the whole language imports concurrency (barriers, atomics, subgroups, cooperative matrices) and `AccSat` poison definedness **before** the base value/memory semantics is proved. The `_sb` superblock path (`OpControlBarrier`, `Workgroup`, `OpSUDotAccSat`), `../fold/modelfold/src/simt_nn/gridsync.rs` atomics (opcodes 227/228/234), `../fold/modelfold/src/simt_nn/coop.rs` subgroups and `../fold/modelfold/src/simt/gemv.rs` cooperative matrices (4456–4459) each exceed v0. Rejected by scope, not by tractability. |
| Testing only (`spirv-val` plus differential runs) | `spirv-val` is a validator, not a semantics; it accepts modules and rejects malformed ones without a value or memory denotation. A differential run bounds observed behaviour, not all accepted modules. Neither supplies the refinement the realization row requires. |
| Skip the GPU face | Leaves the `GPU / SPIR-V` boundary row as `not covered` and the realization row empty for the lane the served decode model runs. The subset is bounded and the integer core already has a denotation; skipping is a choice against the owner decision. |
| Prove the 9k-LOC Rust `Kb` emitter | The fold inventory itself calls extraction of `../fold/modelfold/src/simt_nn/mod.rs` "feasible but impractical". The epic instead proves a pure, extractable `checkFragment : List U32 → Bool` sound against the fragment predicate and runs it on the actual emitter output (`tested`). The emitter stays named as unproved. |
| State the v0 theorem over bit-exact `f32`/`f16` (`OpFMul`/`OpFAdd`/`UnpackHalf2x16`) | Apeiron has no IEEE-754 model; the Aeneas scalar model is integer-only (`target/aeneas-model/lean/Aeneas/Std/Scalar/`). The v0 executing face is integer fixed-point, matching the owner integer-numerics direction. A float artifact on the verified path is a named trigger, not a v0 premise. |

### Completion condition

The v0 fragment is green when every headline module builds under the closed axiom cone and the
concrete emitter check passes with its controls:

```sh
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 \
    env LEAN_NUM_THREADS=1 lake build Spirv.Fragment Spirv.Semantics Spirv.Decode \
    Spirv.Check Spirv.Dp4a AxiomAudit
cargo test --workspace --all-features --locked   # the checkFragment emitter test + controls
```

The shared spec `dp4a_gemv_spec` is face-parameterized and the GPU refinement instantiates it at the
face the emitted packing realizes; the layout-versus-representation question is settled by either a
proved layout theorem or a recorded refutation naming the emitter migration item.

### Invariants

1. The v0 executing face is integer-only: no `OpFMul`, `OpFAdd`, `OpConvertSToF`, `OpConvertUToF` or
   `UnpackHalf2x16` inside the denotation; the scale epilogue is integer fixed-point.
2. The CPU lane and the GPU lane refine **one shared Lean spec** (`dp4a_gemv_spec`), face-parameterized
   over the canonical object, not a standalone `ZMod (2^w)` semantics; the two refinements compose by
   transitivity of refinement, not by a new correspondence.
3. The SPIR-V spec revision is pinned: SPIR-V 1.3 plus `SPV_KHR_integer_dot_product` rev 3
   (2021-09-08); every opcode semantics quoted resolves to that pin.
4. Out-of-bounds `OpAccessChain` is undefined behaviour and must be proved unreachable, never
   handled; the row guard and the split-K `OpSelect` are load-bearing.
5. One in-progress owner per file; `proof/corpus/.lake` and `proof/corpus/AxiomAudit.lean` are one
   lane each (the audit file is lead-owned).
6. A grader for a v0 claim is one gate command plus one control; a green build is not a green proof
   until the control that can make it red has run.

### Verification hypotheses

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| `sdot_denote`: `OpSUDot` equals `∑ sext8(byte i a) · zext8(byte i b)` in `ZMod (2^32)` | proved | `lake build Spirv.Semantics AxiomAudit`; theorem `sdot_denote` |
| Decoder soundness: `decode ws = some m → WellFormed m ∧ IsDp4aFragment m` | proved | `lake build Spirv.Decode AxiomAudit`; theorem `decode_sound` |
| `checkFragment_sound`: the extractable checker accepts only fragment modules | open | only the header leg is proved (`Spirv.Check.checkFragment_header`, `Spirv/Check.lean`); residual components (framing/shape scan, scan state, loop composition) are listed in `Spirv/Check.lean:45-65`. `IsDp4aFragment` (`Spirv/Fragment.lean:233`) holds of every `Module` by `inst_opcode_mem_served` (`Spirv/Decode.lean:514`), so the checker's burden is decode success + `WellFormed` |
| The actual `gemv_dp4a_module_dyn` output passes `checkFragment`; a byte flip rejects | tested | Rust test over `Kb`/emitter output plus byte-flip control |
| `gpu_dp4a_refines_cpu_spec`: GPU `denote ∘ decode` equals `dp4a_gemv_spec` | open (target); anchor for the landed identity | the landed Lean `Spirv.gpu_dp4a_word_refines_cpu_word` (`Spirv/Dp4a.lean:252`, renamed from the contract name) proves only the model-level word identity `gpu_dp4a_word = cpu_dp4a_word` (`Refines := Eq`, hand-written words differing by `scaleConst` vs `2 ^ scaleShift`); the `denote ∘ decode` statement remains open and `Spirv/Dp4a.lean:33-46` records it as not proved |
| Every emitted `OpAccessChain` index is in bounds under dispatch parameters | proved | in-bounds obligations in `Spirv.Dp4a`; guard-removal control must fail |
| Axiom cone | proved | `proof/corpus/AxiomAudit.lean` `#audit`, cone `{propext, Classical.choice, Quot.sound}` |
| Epic cost | anchor | ≈ a dozen bounded tasks calibrated by reading `docs/epics/inference-parity.md` receipts; not a schedule, not measured |

### Non-goals

- Full SPIR-V, and any graphics stage, image, derivative or tessellation op.
- A standalone `ZMod (2^w)` fragment semantics as a second foundation; the target is the
  face-parameterized instance of the canonical object.
- Phase 2: barriers/`Workgroup` memory, atomics, subgroups, cooperative matrices, and the `AccSat`
  poison definedness obligation.
- IEEE-754 `f32`/`f16` epilogue semantics; the fold float epilogue is a migration item.
- Driver or hardware equivalence; the SPIR-V module semantics is the artifact.
- Proving the 9k-LOC Rust `Kb` emitter; the epic uses the extracted checker seam.
- Any change to the `fold` tree, its 13 worktrees or its dirty files; it is a migration source.

### Trust boundary

The claim stops at the SPIR-V module denotation. `rustc`/LLVM codegen, the SPIR-V toolchain, the
Vulkan driver, the GPU ISA and the hardware are the named platform boundary, with a runtime-checked
margin. The concrete emitted module is `tested`, not proved; `Kb` is unproved and named. No
bit-parity or hardware-equivalence claim is made. This epic changes no CPU-lane contract.

### Open questions

- Scale epilogue: confirm the integer fixed-point form (recommended by the owner direction) before
  W1 opens; if the fold f32/f16 epilogue must be modelled bit-exactly, the task stops and returns to
  the owner (Aeneas float-fork trigger).
- `checkFragment` Rust seam: a `#[path]` extraction of the Lean-authored checker versus a Rust body
  under Aeneas; the spec must name one body, not two.
- Sequencing: the implementation waves stay behind the inference-parity wave barrier unless the lead
  schedules otherwise.
- **Layout/face question.** Which face representation the emitted Q4/Q8 packing realizes; the verdict
  is a proved layout theorem or a recorded refutation plus the named emitter migration item. Owner:
  W1-T1, reviewed by the lead.
- **Opcode-set reconciliation.** `47` per the spike; the grouped transcription sums to `46`; W1-T1
  re-derives the exact set mechanically from the emitters and records the resolved count, rather than
  trusting either figure.
- **Naming authority.** `Nous.Model.Instances`, `Nous.Model.Attention` and their
  `attention_channel_form` are W4-T2/W4-T3 outputs not present at this revision; the epic re-derives
  the exact consumer names when they land and does not block v0 on them.

## Specification — Verifiable SPIR-V subset for the integer GPU lane

- **Epic:** `APEIRON-GPU-SPV`
- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/gpu-spv.md`](gpu-spv.md)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (see
  `docs/archive/handoff-openai.md`)
- **Status:** draft

```text
SCOPE:      mechanized denotational semantics of the barrier-free integer DP4A SPIR-V fragment, its
            decoder and extractable fragment checker, and the GPU-side refinement of the shared,
            face-parameterized integer GEMV spec, rooted in the canonical object
            Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}. Files: planned
            proof/corpus/Spirv/*.lean, the Rust checker test, AxiomAudit.lean registration. No Rust
            emitter change, no float denotation, no phase 2.
REGISTER:   canonical-object semantics + refinement + test; exclusions: concurrency, poison
            definedness, IEEE-754 float epilogue, driver/hardware equivalence, the unproved 9k-LOC
            emitter
CORPUS:     proof/corpus/WordRefinement.lean word_mul_denote/word_norm_denote (the canonical
            RingCell/ringMul/ringNormForm face carried by ZMod words);
            proof/corpus/Nous/Model/Decomposition.lean hyperbolic_decomposition (unit hypotheses);
            proof/corpus/Nous/Model/State.lean token_step_fold (fold split);
            proof/corpus/Nous/Model/Memory.lean memoryStep;
            proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean attention_is_spinor_scan,
            softmax_attention_is_online_spinor, so2_spinor_recurrence;
            proof/corpus/Numeric/Engine/Exp.lean fixed_exp_refines (refinement pattern);
            proof/corpus/AxiomAudit.lean closed cone {propext, Classical.choice, Quot.sound};
            primary source /tmp/opencode/toolchain-spike-spirv-subset.md (spike, no repository file)
REFERENCE:  /tmp/opencode/toolchain-spike-spirv-subset.md §1.1-1.3, §2, §4, §5.3, §6;
            /home/socol/Workspace/fold/modelfold/src/simt/dp4a.rs (sha256 ffe4f6c3…);
            /home/socol/Workspace/fold/modelfold/src/simt_nn/mod.rs (Kb, sha256 6073bfe1…);
            /home/socol/Workspace/fold/bedrock/src/serve/gpu.rs;
            docs/epics/ai-track-reanchor.md W4-T2/T3 (planned Nous.Model.Instances/Attention, not
            present at this revision)
GATE:       cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540
            env LEAN_NUM_THREADS=1 lake build Spirv.Fragment Spirv.Semantics Spirv.Decode
            Spirv.Check Spirv.Dp4a AxiomAudit;
            proof/regen.sh --check only if the checker body is extracted;
            cargo test --workspace --all-features --locked for the emitter check
GRADE:      proved for the face-parameterized sdot_denote and the split-K fold reassociation,
            decode_sound and the in-bounds obligations; target, not yet proved, for
            checkFragment_sound and gpu_dp4a_refines_cpu_spec (see the Contracts status note);
            tested for the concrete emitter run and its controls; anchor for the layout/face
            verdict and the cost until W1-T1 proves or refutes the layout
```

### Scope and admissible domain

This specification fixes the v0 barrier-free integer DP4A fragment, its denotation, the decoder and
checker that connect emitted bytes to that fragment, and the refinement of the shared GEMV spec. It
does not define the emitter, the driver or the hardware.

| Item | Value |
|---|---|
| Operation | `denote (decode ws)` for `ws : List U32`; fragment predicate `IsDp4aFragment`; checker `checkFragment` |
| Domain | SPIR-V 1.3 binaries (little-endian `u32` words) of the fragment below, produced by `Kb` for `gemv_dp4a_module_dyn` |
| Totality | `decode : List U32 → Option Module` (total, rejects malformed); `denote` total on `WellFormed` fragment modules |
| Preconditions | `decode ws = some m`, `WellFormed m`, `IsDp4aFragment m`, all `OpAccessChain` indices in bounds (proved per module) |
| Postcondition | `denote (decode ws) = dp4a_gemv_spec face M K segs c …` with the integer fold, the face-parameterized canonical object, and integer fixed-point scale |
| Face | **decided (W1-T1): hyperbolic `σ = +1`** in the regular-representation reading of the emitted Q4/Q8 packing (`sdot_split_ringMul`, `sdot_split_realized`); only channel 0 is computed — cross-channel accumulation is the named emitter migration item (`cross_channel_not_realized`) and `ringNormForm` self-pairing is not realized (SU asymmetry) |

#### Fragment (served module and v0 executing set)

Primary source: `/tmp/opencode/toolchain-spike-spirv-subset.md` §1.1–1.3. The served
`gemv_dp4a_module_dyn` module reads (spike's count) 47 distinct opcodes; the grouped inventory below
is that set, transcribed from the spike. The v0 **denotation** admits the integer executing subset:
the float epilogue syntax is inventoried but is not in the v0 denotation (see the epilogue premise).

The count is mechanically derived (the spike report was lost in the reboot; `W1-T1` and `W3-T19`
re-derived it from the emitters): whole body **47**, served path **42** (`Spirv.Fragment.servedOpcodes`),
v0 executing set **37** (`executingOpcodes`); off the served path are `OpConvertUToF`(112),
`OpBitcast`(124), `OpShiftLeftLogical`(196), `OpBitwiseOr`(197), `OpBitwiseXor`(198). The set
`IsDp4aFragment` admits is the served 42; the grouped inventory below is the earlier spike
transcription, kept for reference.

| Group | Opcodes (number) |
|---|---|
| Module/framing | `OpCapability`(17), `OpExtension`(10), `OpExtInstImport`(11), `OpMemoryModel`(14), `OpEntryPoint`(15), `OpExecutionMode`(16), `OpDecorate`(71), `OpMemberDecorate`(72) |
| Types/constants | `OpTypeVoid`(19), `OpTypeBool`(20), `OpTypeInt`(21), `OpTypeFloat`(22), `OpTypeVector`(23), `OpTypeRuntimeArray`(29), `OpTypeStruct`(30), `OpTypePointer`(32), `OpTypeFunction`(33), `OpConstant`(43) |
| Functions/objects | `OpFunction`(54), `OpFunctionEnd`(56), `OpVariable`(59), `OpLoad`(61), `OpStore`(62), `OpAccessChain`(65) |
| Value ops | `OpCompositeExtract`(81), `OpSelect`(169), `OpULessThan`(176), `OpIAdd`(128), `OpIMul`(132), `OpShiftLeftLogical`(196), `OpShiftRightLogical`(194), `OpBitwiseAnd`(199), `OpBitwiseOr`(197), `OpBitwiseXor`(198), `OpConvertSToF`(111), `OpConvertUToF`(112), `OpFAdd`(129), `OpFMul`(133) |
| Integer dot | `OpSUDot`(4452) with `PackedVectorFormat4x8Bit`(0) |
| Ext-inst | `OpExtInst`(12) `GLSL.std.450` `UnpackHalf2x16`(62) |
| Control | `OpLabel`(248), `OpBranch`(249), `OpBranchConditional`(250), `OpLoopMerge`(246), `OpSelectionMerge`(247), `OpReturn`(253) |

Capabilities: `Shader`(1), `DotProduct`(6019), `DotProductInput4x8BitPacked`(6018); extension
`SPV_KHR_integer_dot_product`; version word `SPV_VERSION_1_3`. Memory model: `Logical`(0) +
`GLSL450`(1). Storage classes: `Input`(1), `Function`(7), `StorageBuffer`(12). BuiltIn:
`GlobalInvocationId`(28) only. Execution model `GLCompute`(5), one entry point `main`, `LocalSize`
default 128 (`DEC_LOCAL`; tests use 64 and 256). Grid 2-D: `.x` = row, `.y` = segment/slot/tile.

**Excluded from v0** (each is a named rejection, not a silent omission): barriers
(`OpControlBarrier`), atomics (`OpAtomicLoad`227, `OpAtomicStore`228, `OpAtomicIAdd`234), subgroups
(`OpGroupNonUniformFAdd`350), cooperative matrices (`OpTypeCooperativeMatrixKHR`4456 and
4457–4459), `Workgroup` storage and cross-invocation shared writes, function calls, `OpPhi`, images
and derivatives, `OpSDiv`/`OpUMod`, and the float epilogue surface `OpTypeFloat`,
`OpExtInst`/`UnpackHalf2x16`, `OpConvertSToF`, `OpConvertUToF`, `OpFMul`, `OpFAdd`.

#### Epilogue premise: integer fixed-point only

The v0 executing face is integer fixed-point. The current fold `f32`/`f16` scale epilogue
(`OpConvertSToF`, `OpConvertUToF`, `OpFMul`, `OpFAdd`, `UnpackHalf2x16`) is a **migration item**, not
a v0 premise; the served module's epilogue is replaced by an integer fixed-point form before the
refinement is stated. Apeiron has no IEEE-754 model; the Aeneas scalar model is integer-only
(`target/aeneas-model/lean/Aeneas/Std/Scalar/`). If bit-exact float semantics become necessary, the
task **stops and returns to the owner**; that is the Aeneas float-fork trigger, and it is not
absorbed by this epic.

#### Denotation

From spike §4.1, the fragment type denotation reuses the corpus integer-word pattern
(`proof/corpus/WordRefinement.lean`):

```text
⟦Int w _⟧        := ZMod (2^w)
⟦Bool⟧           := Bool
⟦Vec n τ⟧        := Fin n → ⟦τ⟧
⟦Ptr sc τ⟧       := Addr sc
⟦RuntimeArray τ⟧ := Nat → ⟦τ⟧
⟦Struct fs⟧      := ∀ i, ⟦fs i⟧
```

The word carrier is not a standalone foundation. `⟦Int w⟧ := ZMod (2^w)` is the face at which the
extracted word kernels already agree with the canonical
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` (`word_mul_denote`, `word_norm_denote`,
`proof/corpus/WordRefinement.lean`). The fragment semantics is defined as a face instance of that
canonical object, so the CPU face and the GPU face meet at the same object; a standalone `ZMod (2^w)`
semantics would be a second foundation and is not accepted.

`eval : Module → Env → Store → Inst → Option Val` is defined for the v0 opcode set; integer
arithmetic is the ring operation in `ZMod (2^w)` (the spec's low-order-`N`-bits law for
`OpIAdd`/`OpIMul`; no `NoSignedWrap` decoration is emitted). `denote : Module → Inputs → Outputs`
maps `GlobalInvocationId = (r, y, 0)` to the invocation result and assembles the grid. The
structured control flow is emitted (`OpLoopMerge`/`OpSelectionMerge`); the Klimis et al. (POPL 2023)
structured-control-flow model is the **well-formedness premise** so `denote` is well-defined without
re-deriving SPIR-V's control-flow rules. `denote` is total on fragment modules once the loop nest is
shown terminating: the counter increments by 1 against a constant bound.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `sdot_denote` (face-parameterized) | on the face `f` the emitted packing realizes, `eval (.SUDot a b .packed4x8)` equals the trace/regular-representation form of the canonical `ringMul`/`ringNormForm` at `f`; otherwise a recorded refutation naming the realized reduction | **proved at `σ = +1` (W1-T1)** | `Spirv.Semantics.sdot_denote`; `sdot_split_ringMul`, `sdot_split_realized`; `byte_order` |
| byte order | `byte` reads component `i` from least-significant byte up (`PackedVectorFormat4x8Bit`) | proved | lemma in `Spirv.Semantics`; `sdot_denote` depends on it |
| D1 boundaries | instruction boundaries come from the high-16-bit `WordCount`; `0` or overrun rejects | proved | `Spirv.Decode` |
| D2 ids | result-id uniqueness, definition-before-use, with the label/`OpFunction` forward-reference exception | proved | `Spirv.Decode` |
| D3 types | used-op type agreement (`OpSUDot` result width ≥ 32, both operands 32-bit packed) | proved | `Spirv.Decode` |
| D4 storage | storage-class and decoration well-formedness | proved | `Spirv.Decode` |
| `decode_sound` | `decode ws = some m → WellFormed m ∧ IsDp4aFragment m`; completeness only for the shape `Kb` emits | proved | `Spirv.Decode.decode_sound` |
| `words_of_bytes_bytes` | `words_of_bytes (bytes ws) = ws` (little-endian round trip) | proved | `Spirv.Decode` |
| `checkFragment_sound` | `checkFragment ws = true → IsDp4aFragment (decode ws).get` | proved | target, not yet stated: current evidence is the header leg `Spirv.Check.checkFragment_header` and the scan lookups `Spirv.ty_get_spec`, `Spirv.defs_contains_spec` (`Spirv/CheckScan.lean`); conclusion to be restated as `∃ m, decode ws = some m` because `IsDp4aFragment` holds of every `Module` |
| `gpu_dp4a_refines_cpu_spec` | `denote (decode (gemv_dp4a_module_dyn face M K local segs c)) = dp4a_gemv_spec face M K segs c …` | proved | target, not yet stated: the proved word-model link is `Spirv.gpu_dp4a_word_refines_cpu_word` (`Refines (gpu_dp4a_word …) (cpu_dp4a_word …)`) |
| split-K fold reassociation | block accumulation and host-side split-K are one `ringMul` fold; the reassociation is `ringMul_assoc` through `token_step_fold` | proved | `Spirv.Dp4a`; `proof/corpus/Nous/Model/State.lean` `token_step_fold` |
| in-bounds | every emitted `OpAccessChain` index is in bounds under dispatch parameters | proved | `Spirv.Dp4a` obligations; guard-removal control |
| concrete run | the actual `gemv_dp4a_module_dyn` output passes `checkFragment` | tested | Rust test over emitter output |

Status note on the grade targets above (descriptive drift correction; no contract change). The Lean
theorem `Spirv.gpu_dp4a_word_refines_cpu_word` carries the weaker model-level word refinement
`Refines (gpu_dp4a_word …) (cpu_dp4a_word …)`, not `denote (decode ws) = dp4a_gemv_spec …`; the latter remains the
target. Likewise `checkFragment_sound` is a target: only the header leg
`Spirv.Check.checkFragment_header` is proved, with the residual components listed in
`proof/corpus/Spirv/Check.lean:45-65`.

The shared spec `dp4a_gemv_spec (face : Face) (M K segs : Nat) (c : Carrier) (scale : ScaleFace)
(W XQ : Array (ZMod (2^32))) …` is the same face-parameterized instance of the canonical
`RingCell`/`ringMul`/`ringNormForm` object that both lanes refine; the CPU-side link (Rust CPU body ⊑
`dp4a_gemv_spec face` via Aeneas, in the style of `proof/corpus/Numeric/Engine/Exp.lean`) and this
GPU-side link **compose by transitivity of refinement**, not by a second correspondence. Block
accumulation and host-side split-K are the same fold: `token_step_fold`
(`proof/corpus/Nous/Model/State.lean`) is `ringMul_assoc` over `ringMul`, and the memory step is
`memoryStep` (`proof/corpus/Nous/Model/Memory.lean`).

#### Word-level degeneracy caveat and face ledger

`Z/2^w` ramifies at `2`: `2` is not a unit, so `hyperbolic_decomposition`
(`proof/corpus/Nous/Model/Decomposition.lean`, hypotheses `r ^ 2 = σ`, `r * s = 1`, `2 * h = 1`) is
true over `ℚ` and false over `Z/2^w`. The spec states, before W1, which structures transport and
which degenerate at the word face; the ledger is binding on every downstream claim.

| Structure | At `Z/2^w` | Basis |
|---|---|---|
| Ring equations, `ringMul_assoc`, `ringMul_comm`, unit laws | transport | `CommRing (ZMod (2^w))`; `word_mul_denote` |
| `opSUDot` fold and its reassociation (`token_step_fold`) | transport | `ringMul_assoc` |
| Trace/regular-representation form of the face | **proved at `σ = +1`** (channel-0 sum; the `ringForm (-1)` reading also realized) | `Spirv.sdot_split_ringMul`, `sdot_split_realized` |
| Cross-channel accumulation | **not realized** (only channel 0 is computed) | `Spirv.cross_channel_not_realized`; named emitter migration item |
| `RingCell R ≅ R × R` channel splitting (`hyperbolic_decomposition`) | degenerates for even `2`; needs the `IsUnit` form | `2` not a unit at `Z/2^w` |
| Idempotent splitting (nontrivial idempotents) | degenerates | ramification at `2` |
| Norm-form nondegeneracy (no zero divisors) | degenerates | ramification at `2` |
| Spinor double-cover / rotor metric (`spinor_scan_preserves_norm` target) | degenerate domain only | `AttentionSpinor` is over `ℝ`; `so2_spinor_recurrence`; `Z/2^w` has zero divisors |

Norm preservation is claimed only on the nondegenerate domain; a runtime margin carries the rest and
is named, not silently assumed. The Nous normal forms are the refinement targets; where the naming
authority is not yet present (the planned `Nous.Model.Instances` module, W4-T2, and the planned
`Nous.Model.Attention` module's `attention_channel_form`, W4-T3, per
`docs/epics/ai-track-reanchor.md`), the present anchors are `hyperbolic_decomposition`,
`attention_is_spinor_scan`, `softmax_attention_is_online_spinor` and `so2_spinor_recurrence`.

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| malformed framing / bad `WordCount` | `decode` returns `none` | D1 |
| duplicate or undefined result-id | `decode` returns `none` | D2 |
| type disagreement on a used op | `decode` returns `none` | D3 |
| bad storage class/decoration | `decode` returns `none` | D4 |
| out-of-bounds `OpAccessChain` | undefined behaviour — **must be proved unreachable** | in-bounds obligations; not handled at runtime |
| division or `OpUMod` in the fragment | excluded by `IsDp4aFragment` | no division/`OpUMod` is emitted, so no SPIR-V division UB surface |

Out-of-bounds is undefined behaviour under `Logical` + `GLSL450`; the contract is a proof that
every index is in bounds, not a bounds check. The row guard `OpULessThan(r, M)` and the split-K
`OpSelect` load-bearing term are the proof's load-bearing terms.

### Evidence plan

1. **Proof.** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 env
   LEAN_NUM_THREADS=1 lake build Spirv.Fragment Spirv.Semantics Spirv.Decode Spirv.Check Spirv.Dp4a
   AxiomAudit` from `proof/corpus/`.
2. **Extraction.** `proof/regen.sh --check` **if and only if** the checker body is added to
   `proof/extract/lib.rs` and `proof/ApeironWord.lean` is regenerated; otherwise the fragment lives
   in `proof/corpus/` alone and the regeneration is not in this epic's gate.
3. **Axiom cone.** Every headline is registered in `proof/corpus/AxiomAudit.lean`; anything outside
   `{propext, Classical.choice, Quot.sound}` fails the build (lead-owned file).
4. **Positive tests.** `cargo test --workspace --all-features --locked` runs `checkFragment` on the
   actual `gemv_dp4a_module_dyn` output for the served parameter set and on boundary modules.
5. **Negative controls.**
   - byte flip in a throwaway copy of the emitted words → `checkFragment`/decoder rejects;
   - removal of the row guard in the emitted module → the in-bounds obligation is unsolved;
   - flip one fixed-point scale constant → the refinement/`#guard` comparison is red;
   - wrong root witness (`r = 1` for `σ = -1`) → the face/layout verdict is red, matching the
     `APEIRON-AI-W4-T2` control in `docs/epics/ai-track-reanchor.md`.
   Restore each and show green.
6. **Measurement.** None. No performance claim is in this epic.

### Trust boundary

The artifact is the SPIR-V module denotation. `rustc`/LLVM codegen, the SPIR-V toolchain, the Vulkan
driver, the GPU ISA and the hardware are the named platform boundary with a runtime-checked margin
(CompCert pattern). The concrete emitted-module run is `tested`, not proved; `Kb`
(`/home/socol/Workspace/fold/modelfold/src/simt_nn/mod.rs`, 9k LOC) is unproved and named. No
bit-parity or hardware-equivalence claim is made, and no CPU-lane contract changes.

### Open questions

- Confirm the integer fixed-point epilogue form before W1 opens (owner; Aeneas float-fork trigger
  otherwise).
- Decide the checker seam: Rust body under Aeneas, or a `#[path]` extraction of a pure Rust checker
  that mirrors `Spirv.Check`; the spec names exactly one body.
- LocalSize for the refined module: the default 128 versus the 64/256 test values; the in-bounds
  obligations must be parametric or the gate pins one value.
- Layout/face verdict: which face the emitted Q4/Q8 packing realizes (`OpSUDot` signed·unsigned
  asymmetry and the even/odd nibble split are the fingerprint); W1-T1 records a proved layout theorem
  or a refutation naming the emitter migration item, and the face ledger above binds either way.
- Opcode set: the mechanical re-derivation gives whole body 47 / served 42 / executing 37;
  `IsDp4aFragment` admits the served 42 (`Spirv.Fragment.servedOpcodes`; `W3-T19` corrected the
  earlier 44 transcription, which had included 196/197).

## Plan — Verifiable SPIR-V subset for the integer GPU lane

- **Epic:** `APEIRON-GPU-SPV`
- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/gpu-spv.md`](gpu-spv.md)
- **Spec:** [`docs/epics/gpu-spv.md`](gpu-spv.md)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (see
  `docs/archive/handoff-openai.md`)
- **Status:** draft

### Goal and completion condition

The v0 wave-level barrier: every task gate exit 0 with its control restored; the face-parameterized
`sdot_denote`, `decode_sound`, `checkFragment_sound` and `gpu_dp4a_refines_cpu_spec` kernel-check under
the closed axiom cone; the concrete `gemv_dp4a_module_dyn` output passes the checker with the byte-flip
control red before restore; the in-bounds obligations hold with the guard-removal control red; and the
layout/face verdict is recorded — a proved layout theorem or a refutation naming the emitter migration
item. Barrier commands:

```sh
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 \
    env LEAN_NUM_THREADS=1 lake build Spirv.Fragment Spirv.Semantics Spirv.Decode Spirv.Check Spirv.Dp4a AxiomAudit
proof/regen.sh --check
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
```

`lake build Spirv.*` is shorthand for the five explicit module names above. Phase 2 (barriers,
atomics, subgroups, cooperative matrices, `AccSat` poison) opens only after this barrier is green.

### Non-goals

- No full SPIR-V, no graphics stages, no images or derivatives.
- No standalone `ZMod (2^w)` fragment semantics as a second foundation; the face-parameterized
  canonical object is the target.
- No phase 2 concurrency/poison fragments before v0 is green.
- No IEEE-754 `f32`/`f16` denotation and no Aeneas float fork (trigger returns to the owner).
- No proof of the 9k-LOC Rust `Kb` emitter; the checker seam carries the module claim.
- No driver or hardware equivalence claim; no performance measurement.
- No task before its files are named, ≤3 files, one owner per file.

### Module layout

Planned, per spike §4.5, under `proof/corpus/Spirv/`:

```text
proof/corpus/Spirv/
  Fragment.lean    -- VType/Val/Inst/Module, IsDp4aFragment, well-formedness, re-derived opcode set
  Semantics.lean   -- eval, denote, face-parameterized sdot_denote, integer ring lemmas over ZMod
  Decode.lean      -- decode, words_of_bytes, D1-D4, decode_sound
  Check.lean       -- checkFragment_sound over the extracted Rust checker
  Dp4a.lean        -- dp4a_gemv_spec (face), gpu_dp4a_refines_cpu_spec, split-K fold split, in-bounds obligations
```

Two shared-file seams: `proof/corpus/lakefile.lean` roots are registered by the **lead** before
dispatch (a task whose module is unregistered reports its gate blocked, not green);
`proof/corpus/AxiomAudit.lean` is **lead-owned** and the lead registers each wave's headlines at the
barrier. A change to `proof/extract/lib.rs` regenerates `proof/ApeironWord.lean` and rebinds every
digest; that regeneration is a barrier lane (`proof/corpus/.lake` is one lane, serialized on
`flock /tmp/opencode/apeiron-lane.lock`).

### Wave map

Tasks in one wave touch disjoint files. Implementation waves stay behind the inference-parity wave
barrier unless the lead schedules otherwise. Owners are intended lanes; the lead assigns at dispatch.

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W0 | W0-T1 docs draft + W0-T2 continuity addendum (landed; digests `7e056f4b…`/`5393c34b…`/`c2c904af…`, intent/spec/plan) | `docs/epics/gpu-spv.md`, `docs/epics/gpu-spv.md`, `docs/epics/gpu-spv.md` | deepseek-flash | path-resolution check over every cited path | n/a — documents |
| W1 | canonical-face fragment types + denotation + face-parameterized `sdot_denote` + layout verdict + opcode re-derivation | `proof/corpus/Spirv/Fragment.lean`, `proof/corpus/Spirv/Semantics.lean` | xiaomi-pro (gemini lane down) | `lake build Spirv.Fragment Spirv.Semantics` | swap the byte-order rule → `sdot_denote` unsolved; wrong-σ witness (`r = 1` for `σ = -1`) → face/layout verdict red |
| W2 | decoder soundness | `proof/corpus/Spirv/Decode.lean` | gemini-flash | `lake build Spirv.Decode` | corrupt a `WordCount`/result-id in a fixture → `decode_sound` red |
| W2 | Rust checker body + emitter run | `src/gpu/spirv/mod.rs`, `src/gpu/spirv/check.rs`, `tests/gpu_spv_check.rs` | deepseek-flash | `cargo test --workspace --all-features --locked` | byte flip in the emitted words → test red |
| W3 | checker refinement | `proof/corpus/Spirv/Check.lean`, `proof/extract/lib.rs` | gemini-flash | `proof/regen.sh --check`; `lake build Spirv.Check` | mutate the checker body → `checkFragment_sound` red |
| W3 | GPU refinement + in-bounds | `proof/corpus/Spirv/Dp4a.lean` | gemini-flash | `lake build Spirv.Dp4a` | remove the row guard → in-bounds obligation unsolved; flip one fixed-point scale constant → refinement red |
| W4 | barrier + phase-2 admission | `proof/corpus/AxiomAudit.lean` (lead) | lead | barrier commands above | n/a — barrier |

### Task blocks

The first implementation tasks carry an Arc Block per `AGENTS.md` §3; the later tasks use the same
fields.

#### APEIRON-GPU-SPV-W1-T1 · fragment types, denotation, `sdot_denote`

```text
SCOPE:      define VType/Val/Inst/Module, eval/denote for the v0 opcode set, IsDp4aFragment and the
            integer ring lemmas as a face instance of Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,
            ringNormForm}; the face-parameterized sdot_denote; re-derive the exact opcode set
            mechanically from the emitters; settle the layout-vs-representation question with a proved
            layout theorem or a recorded refutation naming the emitter migration item.
            Completion: goal checked, verdict recorded, opcode set recorded.
REGISTER:   canonical-face semantics + layout verdict; exclusions: memory-model concurrency, float
            epilogue, phase 2, a standalone ZMod (2^w) foundation
CORPUS:     proof/corpus/WordRefinement.lean word_mul_denote/word_norm_denote (canonical face);
            proof/corpus/Nous/Model/Decomposition.lean hyperbolic_decomposition (unit hypotheses);
            proof/corpus/Nous/Model/State.lean token_step_fold (fold split); no existing SPIR-V value
            semantics in the pinned corpus or /tmp source
REFERENCE:  /tmp/opencode/toolchain-spike-spirv-subset.md §1.1-1.3, §2, §4.1-4.2; SPIR-V 1.3 +
            SPV_KHR_integer_dot_product rev 3 as quoted there
GATE:       cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540
            env LEAN_NUM_THREADS=1 lake build Spirv.Fragment Spirv.Semantics
GRADE:      proved for the face-parameterized sdot_denote and the byte-order lemma, or refuted with
            the realized reduction named; anchor before W1
```

- **Scope:** the fragment datatypes; `eval`/`denote` as a face instance of the canonical object; the
  face-parameterized `OpSUDot`/byte-order lemmas; the mechanical re-derivation of the exact opcode set
  from the emitters; and the layout-versus-representation verdict.
- **Files:** `proof/corpus/Spirv/Fragment.lean`, `proof/corpus/Spirv/Semantics.lean`
- **Depends on:** none
- **Grade:** proved for the semantics; proved-or-refuted for the layout verdict (`anchor` until run)
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 env LEAN_NUM_THREADS=1 lake build Spirv.Fragment Spirv.Semantics`
- **Negative control:** in a throwaway copy, swap the `PackedVectorFormat4x8Bit` byte order (read
  component `3-i`); `sdot_denote` must fail with an attributable goal. Swap in a wrong-σ witness
  (`r = 1` for `σ = -1`, matching `APEIRON-AI-W4-T2` in `docs/epics/ai-track-reanchor.md`); the
  face/layout verdict must be red. Restore both green.
- **Evidence to record:** module sha256, gate exit, control receipts, the re-derived opcode set and
  its count, the layout verdict and the face it names, elapsed.
- **Budget:** 30 min task; Lean command ceiling 540 s.
- **Open:** a refutation outcome names the emitter migration item and does not block the integer
  semantics result; the face ledger in `docs/epics/gpu-spv.md` binds either verdict.

- **`APEIRON-GPU-SPV-W1-T1` receipt** (xiaomi-pro, ~53 min of the 45-hard box — overrun; all stages
  complete): `Spirv/Fragment.lean` `bf2cc021…` (271 lines) and `Spirv/Semantics.lean` `47f12803…`
  (632 lines); gate `lake build Spirv.Fragment Spirv.Semantics` exit 0 (2973 jobs); zero
  `sorry`/`set_option`. Opcode set re-derived mechanically from the emitter: whole body 47, served
  path 44 (`servedOpcodes`, `#guard`), v0 executing 39 (`executingOpcodes`, `executingOpcodes_eq`) —
  **corrected to 42/37 by `W3-T19`** (the 44 transcription included 196/197, off the served path per
  the derivation log). **Layout verdict: proved** — the emitted Q4/Q8 packing realizes the **hyperbolic face `σ = +1`** in
  the regular-representation reading (`sdot_split_ringMul`: the two `OpSUDot` phase results sum per
  byte lane to `ringMul` channel 0; `sdot_split_realized` records the `ringForm (-1)` reading), root
  witness checked (`root_witness`); the SU asymmetry admits no self-pairing so `ringNormForm` is not
  realized; 2-adic caveat kept (`ringTrace_two_channel0`, `two_not_unit`). Cross-channel accumulation
  is the named emitter migration item. Controls: byte-order swap → `byte_order`/`sdot_denote` red;
  wrong-σ witness (`-1` with root `1`) → `#guard`/`root_witness`/`sdot_split_ringMul` red; both
  restored green. Lead registration: `AxiomAudit.lean` lines appended; the full audit build is
  deferred to the W3 lane's settle (transient cross-import) and reported on its next run. `W2-T1`
  (decoder) is in flight.
- **`APEIRON-GPU-SPV-W2-T1` receipt** (deepseek-flash, ~3 min of commands; inside): `Spirv/Decode.lean`
  `905f5710…` (558 lines) lands the pure decoder (`parseInsts`/`decode`, little-endian
  `words_of_bytes_bytes` round trip) with D1–D4 as checked predicates and
  `decode_sound : decode ws = some m → WellFormed m ∧ IsDp4aFragment m`; gate `lake build Spirv.Decode`
  exit 0 (2967 jobs), zero `set_option`/`sorry`. Controls: WordCount `6 → 1` red; duplicate result-id
  red; both restored. Lead registration: the decoder declarations audited (`AxiomAudit.lean`
  `65e949c7…`, 3900 jobs). Residual: the concrete emitter words are not decoded in Lean (hand-built
  served-shape fixtures; the Rust test is the completeness seam), `bound` is checked nonzero only, and
  D3 covers the named `OpSUDot` obligation; `W2-T2` (Rust checker + captured-words test) is in flight.
- **`APEIRON-GPU-SPV-W2-T2` receipt** (deepseek-flash, ~17 min of a 30/45 box — inside): the Rust
  checker lands (`src/gpu/spirv/mod.rs` `3e9e255f…`; `src/gpu/spirv/check.rs` `2fac3308…`, 652 lines;
  `tests/gpu_spv_check.rs` `befaf444…`, 778 lines): `check_fragment(&[u32]) -> Result<(), CheckError>`
  (pure, 16 typed refusals) over header/D1/D2/D3/D4 and the served set; the captured stream is
  generated from the emitter (`gemv_dp4a_module_dyn(4096, 14336, 256, 8, Q4, dyn_sx)`, 676 words,
  fold rev `d497161f`, emitter `ffe4f6c3…`) and exercises 42 distinct served opcodes. Gates: focused
  6/6; workspace tests 22 suites green; fmt green; clippy green on a clean-HEAD copy (the real tree was
  transiently blocked by the engine lane's untracked `src/engine/mod.rs`). Controls: packed-format byte
  flip and opcode-set flip red, restored. **Reconciliation items:** (1) the served set is 42, not the
  44 transcribed in `Spirv.Fragment` (196/197 are off-path per the derivation log); (2)
  `Spirv.Decode.Inst.uses` is over-strict on the forward references SPIR-V 1.3 permits
  (`OpExecutionMode`/`OpDecorate`/`OpMemberDecorate`). `W3-T19` (Lean) and `W2-T3` (Rust) align both.

#### APEIRON-GPU-SPV-W2-T1 · decoder soundness

```text
SCOPE:      decode : List U32 -> Option Module with words_of_bytes round trip, obligations D1-D4, and
            decode_sound (WellFormed and IsDp4aFragment); completeness only for the emitted shape.
REGISTER:   semantics/decoder; exclusions: emitter correctness (checker seam instead)
CORPUS:     no existing decoder; /tmp source §4.4 defines D1-D4
REFERENCE:  /tmp/opencode/toolchain-spike-spirv-subset.md §4.4; fold Kb header/WordCount at
            /home/socol/Workspace/fold/modelfold/src/simt_nn/mod.rs
GATE:       cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540
            env LEAN_NUM_THREADS=1 lake build Spirv.Decode
GRADE:      proved for decode_sound; tested later for the concrete emitter run
```

- **Scope:** the pure decoder and its soundness theorem.
- **Files:** `proof/corpus/Spirv/Decode.lean`
- **Depends on:** W1-T1
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 env LEAN_NUM_THREADS=1 lake build Spirv.Decode`
- **Negative control:** in a throwaway copy, corrupt one `WordCount`/result-id in a test module;
  `decode_sound` must be red; restore green.
- **Evidence to record:** module sha256, gate exit, control receipt, elapsed.
- **Budget:** 30 min task; Lean command ceiling 540 s.

#### APEIRON-GPU-SPV-W2-T2 · Rust checker body and emitter run

```text
SCOPE:      pure integer Rust check_fragment : &[u32] -> bool implementing IsDp4aFragment, with a
            test running it on the actual gemv_dp4a_module_dyn words. Completion: cargo test green
            with the byte-flip control red before restore.
REGISTER:   test; exclusions: proof of the checker (W3-T1), Kb correctness
CORPUS:     none; this body is the extraction root for Spirv.Check
REFERENCE:  /tmp/opencode/toolchain-spike-spirv-subset.md §4.4; emitter
            /home/socol/Workspace/fold/modelfold/src/simt/dp4a.rs (sha256 ffe4f6c3…),
            /home/socol/Workspace/fold/modelfold/src/simt_nn/mod.rs (sha256 6073bfe1…)
GATE:       cargo test --workspace --all-features --locked
GRADE:      tested; the module bytes carry their fold revision and sha256
```

- **Scope:** the real Rust checker body (the epic's executing path) and the concrete emitter run.
- **Files:** `src/gpu/spirv/mod.rs`, `src/gpu/spirv/check.rs`, `tests/gpu_spv_check.rs`
- **Depends on:** none (the checker is independent of the Lean fragment)
- **Grade:** tested
- **Gate:** `cargo test --workspace --all-features --locked`
- **Negative control:** in a throwaway copy, flip one byte of the emitted words; the test must reject
  with an attributable failure; restore green. A second control flips one checker opcode-set entry.
- **Evidence to record:** checker and test sha256, the captured module sha256 with its fold revision
  `d497161f32e0d366220f2065ec36e680bd9aed86`, gate exits, control receipts, elapsed.
- **Budget:** 20 min task.
- **Open decision:** the emitted words are captured from the fold emitter at the named revision; if
  the checker is instead extracted, `proof/extract/lib.rs` moves into this task and the file set
  changes before dispatch.

#### APEIRON-GPU-SPV-W3-T1 · checker refinement

- **Scope:** extract `check_fragment` into `proof/ApeironWord.lean` and prove
  `checkFragment_sound : checkFragment ws = true → IsDp4aFragment (decode ws).get`.
- **Files:** `proof/corpus/Spirv/Check.lean`, `proof/extract/lib.rs`
- **Depends on:** W2-T1, W2-T2
- **Grade:** proved
- **Gate:** `proof/regen.sh --check`; then `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 env LEAN_NUM_THREADS=1 lake build Spirv.Check`
- **Negative control:** mutate one checker predicate in a throwaway copy; `checkFragment_sound` must
  be red; restore green. Regeneration rebinds `proof/ApeironWord.lean` and all digests.
- **Evidence to record:** extraction digests, module sha256, gate exits, control receipt, elapsed.
- **Budget:** 30 min task; Lean command ceiling 540 s.

#### APEIRON-GPU-SPV-W3-T2 · GPU refinement and in-bounds obligations

- **Scope:** `dp4a_gemv_spec face`, `gpu_dp4a_refines_cpu_spec face`, the split-K fold reassociation
  (`ringMul_assoc` through `token_step_fold`), and the proof that every emitted `OpAccessChain` index
  is in bounds under dispatch parameters.
- **Files:** `proof/corpus/Spirv/Dp4a.lean`
- **Depends on:** W1-T1, W2-T1, W3-T1
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 env LEAN_NUM_THREADS=1 lake build Spirv.Dp4a`
- **Negative control:** remove the row guard `OpULessThan(r, M)` in a throwaway copy; the in-bounds
  obligation must be unsolved. Flip one fixed-point scale constant; the refinement/`#guard`
  comparison must be red. Restore both green.
- **Evidence to record:** module sha256, gate exits, both control receipts, the shared-spec identity,
  elapsed.
- **Budget:** 30 min task; Lean command ceiling 540 s.

### Barrier checklist

Run at the wave barrier, not before:

- [ ] Every task gate exit 0, with its negative control run and restored.
- [ ] No task touched a file owned by another task; lead-owned `proof/corpus/AxiomAudit.lean` and
      `proof/corpus/lakefile.lean` edits were made by the lead, not a worker.
- [ ] Headlines registered in `proof/corpus/AxiomAudit.lean`; cone closed.
- [ ] Layout/face verdict recorded: a proved layout theorem for the realized face, or a refutation
      naming the emitter migration item; the spec's face ledger matches the verdict.
- [ ] Exact opcode set re-derived mechanically from the emitters and recorded; `IsDp4aFragment` admits
      that set, not the spike's 47 or the transcription's 46.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` pass.
- [ ] `proof/regen.sh --check` reports fresh; `cd proof/corpus && lake build` passes.
- [ ] Source and tool digests recorded: fragment modules, checker, captured module, fold revision.
- [ ] Claims registered at their evidence grade; the emitter named unproved; citations not verified.
- [ ] Trust-boundary statement present; phase 2 not opened early.

### Sequencing and cost

- The shared integer spec `dp4a_gemv_spec` is fixed before W3-T2; the CPU-side refinement
  (`proof/corpus/Numeric/Engine/Exp.lean` pattern) and the GPU-side refinement compose by
  transitivity.
- The shared spec is the face-parameterized instance of the canonical
  `Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` object; `word_mul_denote`/
  `word_norm_denote` (`proof/corpus/WordRefinement.lean`) are the word pattern. No standalone
  `ZMod (2^w)` foundation.
- Fold continuity: block accumulation and host-side split-K are the same `ringMul` fold;
  `token_step_fold` (`proof/corpus/Nous/Model/State.lean`) is the split lemma and `memoryStep`
  (`proof/corpus/Nous/Model/Memory.lean`) the memory update.
- Nous consumers: the GPU refinement targets the Nous normal forms. The planned `Nous.Model.Instances`
  module (W4-T2 faces) and the planned `Nous.Model.Attention` module (`attention_channel_form`, W4-T3)
  are not present at this revision (see `docs/epics/ai-track-reanchor.md`); the present anchors are
  `proof/corpus/Nous/Model/Decomposition.lean` `hyperbolic_decomposition` and
  `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean` `attention_is_spinor_scan`,
  `softmax_attention_is_online_spinor`, `so2_spinor_recurrence`. Norm preservation holds only on the
  nondegenerate domain; a runtime margin carries the rest.
- Implementation waves stay behind the inference-parity barrier unless the lead schedules otherwise.
- Cost anchor: v0 is roughly a dozen bounded tasks (spike §5.1). This figure is an **anchor**,
  calibrated by reading `docs/epics/inference-parity.md` receipts, not a schedule and not measured.

### Phase 2 (not opened by this plan)

Open only after the v0 barrier: `../fold/modelfold/src/simt/superblock.rs` barriers and `Workgroup`
memory, `../fold/modelfold/src/simt_nn/gridsync.rs` atomics,
`../fold/modelfold/src/simt_nn/coop.rs` subgroups and `../fold/modelfold/src/simt/gemv.rs` cooperative
matrices, using the SDV semantics
(GPUVerify, TOPLAS 2015) or an ITree monadic interpreter with bisimulation/refinement (Vellvm, NFM
2025), with the `OpSUDotAccSat` overflow-poison definedness obligation named. The control-flow model
of Klimis et al. (POPL 2023) supplies only the v0 structured-CFG premise.

### Stop and rollback conditions

- Stop the epic and return to the owner if the integer fixed-point epilogue cannot be stated without
  IEEE-754 semantics (Aeneas float-fork trigger).
- Stop and report if a change needs more than three files, a second gate command, or a
  governing-document or canonical-corpus contract change.
- If three materially different approaches to one obligation fail, report the smallest unresolved
  goal with the red artifact and its command.
- Roll back is restore-only: no compensating edits to a contract; the mutation controls run in
  throwaway copies and the working tree is left on the named source revision.
