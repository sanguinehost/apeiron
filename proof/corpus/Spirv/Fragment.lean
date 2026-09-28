import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

/-!
# The barrier-free integer DP4A SPIR-V fragment: types, instructions, admission

Syntax of the v0 fragment the served decode kernel `gemv_dp4a_module_dyn` emits
(`/home/socol/Workspace/fold/modelfold/src/simt/dp4a.rs`, sha256 `ffe4f6c3…`, builder `Kb` in
`/home/socol/Workspace/fold/modelfold/src/simt_nn/mod.rs`, sha256 `6073bfe1…`), per
`docs/epics/gpu-spv.md`. The word carrier is the face at which the extracted word kernels already
agree with the canonical `Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}`
(`word_mul_denote`/`word_norm_denote`, `proof/corpus/WordRefinement.lean`); a standalone
`ZMod (2^w)` foundation is not accepted and is not built here.

## Re-derived opcode set (mechanical, from the emitters)

The count is re-derived from the emission sites of `gemv_dp4a_module_moe_sc`
(`simt/dp4a.rs` lines 132–901, the body every `gemv_dp4a_module*` entry point calls), resolving
each `OP_*` constant through its definition in `simt/mod.rs`, `simt_nn/mod.rs` and `simt/dp4a.rs`,
and restricting to the served path `moe = None`, `two_level = false`, unioned over the accepted
chart carriers (`Q8`, `Q4`, `Q2`, `Q3`), `dyn_sx ∈ {false, true}` and `segments ∈ {1, >1}`.
Derivation log: `/tmp/opencode/gpu-spv-w1t1-opcode-derive.log`.

| set | count | contents |
|---|---|---|
| whole function body (every branch, incl. `two_level`/`q3`/MoE) | 47 | the spike's `47` is this count |
| served path (`gemv_dp4a_module_dyn`) | 42 | `servedOpcodes` below |
| v0 executing subset (denotation domain) | 37 | `executingOpcodes = servedOpcodes \ floatEpilogueOpcodes` |

The 5 opcodes emitted only off the served path (`OpConvertUToF` 112, `OpBitcast` 124,
`OpShiftLeftLogical` 196, `OpBitwiseOr` 197, `OpBitwiseXor` 198 — `moe`/`two_level`/`q3` branches)
are not admitted. The float epilogue surface on the
served path (`OpExtInst` 12 / `UnpackHalf2x16`, `OpTypeFloat` 22, `OpConvertSToF` 111, `OpFAdd` 129,
`OpFMul` 133) is **inventoried syntax** — `IsDp4aFragment` admits it so the emitted module is in the
fragment — and is **not** in the v0 denotation (`executingOpcodes`); it is the named migration item
of `docs/epics/gpu-spv.md`'s epilogue premise. Neither the spike's `47` nor the transcription's `46`
is the admitted set: `47` counts branches the served path never takes, and `46` admits four of
them (`112`, `196`, `197`, `198`) while omitting `OpBitcast` (`124`).

## Well-formedness and domain

`decode : List U32 → Option Module` with obligations D1–D4 and `decode_sound` arrive in
`Spirv.Decode`; `checkFragment` in `Spirv.Check`; the structured-CFG premise (Klimis et al., POPL
2023) and the module-level invocation semantics in `Spirv.Dp4a`. This file fixes the datatypes and
the admission predicate only.
-/

namespace Spirv

/-- A SPIR-V result id (SSA name). -/
abbrev SpvId := Nat

/-- A 32-bit instruction word (`u32`, little-endian in the binary). -/
abbrev U32Word := Nat

/-- The 32-bit integer carrier: the `Int 32` face, `ZMod (2 ^ 32)`. -/
abbrev W32 := ZMod (2 ^ 32)

/-- Storage classes the fragment uses. -/
inductive StorageClass where
  | input
  | function
  | storageBuffer
  deriving DecidableEq, Repr

/-- A pointer value in the fragment's `Logical` addressing model. -/
abbrev Addr (_sc : StorageClass) := Nat

/-- `SPV_KHR_integer_dot_product` packed vector formats the fragment uses. -/
inductive PackedFormat where
  | packed4x8
  deriving DecidableEq, Repr

/-- The v0 fragment types, the spec's denotation table `⟦-⟧`. -/
inductive VType where
  /-- `OpTypeInt w signed`: `⟦Int w _⟧ = ZMod (2 ^ w)`. -/
  | int (w : Nat) (signed : Bool)
  /-- `OpTypeBool`: `⟦Bool⟧ = Bool`. -/
  | bool
  /-- `OpTypeVector`: `⟦Vec n τ⟧ = Fin n → ⟦τ⟧`. -/
  | vec (n : Nat) (τ : VType)
  /-- `OpTypePointer`: `⟦Ptr sc τ⟧ = Addr sc`. -/
  | ptr (sc : StorageClass) (τ : VType)
  /-- `OpTypeRuntimeArray`: `⟦RuntimeArray τ⟧ = Nat → ⟦τ⟧`. -/
  | runtimeArray (τ : VType)
  /-- `OpTypeStruct`: `⟦Struct fs⟧ = ∀ i, ⟦fs i⟧`. -/
  | str (n : Nat) (fs : Fin n → VType)

/-- The denotation of a fragment type: `⟦Int w _⟧ = ZMod (2 ^ w)`, `⟦Bool⟧ = Bool`,
`⟦Vec n τ⟧ = Fin n → ⟦τ⟧`, `⟦Ptr sc τ⟧ = Addr sc`, `⟦RuntimeArray τ⟧ = Nat → ⟦τ⟧`,
`⟦Struct fs⟧ = ∀ i, ⟦fs i⟧`. -/
inductive Val : VType → Type where
  | int {w : Nat} {s : Bool} (z : ZMod (2 ^ w)) : Val (.int w s)
  | bool (b : Bool) : Val .bool
  | vec {n : Nat} {τ : VType} (v : Fin n → Val τ) : Val (.vec n τ)
  | ptr {sc : StorageClass} {τ : VType} (a : Addr sc) : Val (.ptr sc τ)
  | arr {τ : VType} (f : Nat → Val τ) : Val (.runtimeArray τ)
  | str {n : Nat} {fs : Fin n → VType} (v : (i : Fin n) → Val (fs i)) : Val (.str n fs)

/-- The linear instruction set of the v0 fragment: one constructor per admitted opcode, carrying
the operands the denotation reads. Module-level framing (`OpCapability`, decorations, types,
constants) is instruction syntax in SPIR-V's linear form and is carried here as well. -/
inductive Inst where
  -- module framing
  | capability (cap : Nat)                                     -- 17 OpCapability
  | extension (name : String)                                  -- 10 OpExtension
  | extInstImport (r : SpvId) (name : String)                  -- 11 OpExtInstImport
  | extInst (ty r : SpvId) (set instr : Nat) (args : List SpvId) -- 12 OpExtInst (float surface)
  | memoryModel (addr mem : Nat)                               -- 14 OpMemoryModel
  | entryPoint (model : Nat) (entry : SpvId) (iface : List SpvId) -- 15 OpEntryPoint
  | executionMode (entry mode : Nat) (opts : List Nat)         -- 16 OpExecutionMode
  | decorate (target dec : Nat) (lits : List Nat)              -- 71 OpDecorate
  | memberDecorate (ty member dec : Nat) (lits : List Nat)     -- 72 OpMemberDecorate
  -- types and constants
  | typeVoid (r : SpvId)                                       -- 19 OpTypeVoid
  | typeBool (r : SpvId)                                       -- 20 OpTypeBool
  | typeInt (r : SpvId) (width signed : Nat)                   -- 21 OpTypeInt
  | typeFloat (r : SpvId) (width : Nat)                        -- 22 OpTypeFloat (float surface)
  | typeVector (r comp : SpvId) (count : Nat)                  -- 23 OpTypeVector
  | typeRuntimeArray (r elem : SpvId)                          -- 29 OpTypeRuntimeArray
  | typeStruct (r : SpvId) (members : List SpvId)              -- 30 OpTypeStruct
  | typePointer (r : SpvId) (sc : Nat) (pointee : SpvId)       -- 32 OpTypePointer
  | typeFunction (r ret : SpvId) (params : List SpvId)         -- 33 OpTypeFunction
  | constant (ty r : SpvId) (v : Nat)                          -- 43 OpConstant
  -- functions and objects
  | function (ty r : SpvId) (ctl : Nat) (sig : SpvId)          -- 54 OpFunction
  | functionEnd                                                -- 56 OpFunctionEnd
  | variable (ty r : SpvId) (sc : Nat) (init : Option SpvId)   -- 59 OpVariable
  | load (ty r ptr : SpvId)                                    -- 61 OpLoad
  | store (ptr val : SpvId)                                    -- 62 OpStore
  | accessChain (ty r base : SpvId) (idxs : List SpvId)        -- 65 OpAccessChain
  -- integer core
  | compositeExtract (ty r agg : SpvId) (idxs : List Nat)      -- 81 OpCompositeExtract
  | iadd (ty r a b : SpvId)                                    -- 128 OpIAdd
  | imul (ty r a b : SpvId)                                    -- 132 OpIMul
  | select (ty r c a b : SpvId)                                -- 169 OpSelect
  | ult (ty r a b : SpvId)                                     -- 176 OpULessThan
  | shiftRightLogical (ty r a s : SpvId)                       -- 194 OpShiftRightLogical
  | bitAnd (ty r a b : SpvId)                                  -- 199 OpBitwiseAnd
  | sudot (ty r a b : SpvId) (fmt : PackedFormat)              -- 4452 OpSUDot
  -- off-path syntax (two_level/MoE branches; not admitted below) and float surface
  | convertSToF (ty r a : SpvId)                               -- 111 OpConvertSToF (float surface)
  | fadd (ty r a b : SpvId)                                    -- 129 OpFAdd (float surface)
  | fmul (ty r a b : SpvId)                                    -- 133 OpFMul (float surface)
  -- control
  | loopMerge (merge cont ctl : Nat)                           -- 246 OpLoopMerge
  | selectionMerge (merge ctl : Nat)                           -- 247 OpSelectionMerge
  | label (l : SpvId)                                          -- 248 OpLabel
  | branch (l : SpvId)                                         -- 249 OpBranch
  | branchCond (c l1 l2 : SpvId)                               -- 250 OpBranchConditional
  | ret                                                        -- 253 OpReturn

/-- The SPIR-V opcode number of a fragment instruction. -/
def Inst.opcode : Inst → Nat
  | .capability _ => 17
  | .extension _ => 10
  | .extInstImport _ _ => 11
  | .extInst .. => 12
  | .memoryModel _ _ => 14
  | .entryPoint .. => 15
  | .executionMode .. => 16
  | .decorate .. => 71
  | .memberDecorate .. => 72
  | .typeVoid _ => 19
  | .typeBool _ => 20
  | .typeInt .. => 21
  | .typeFloat .. => 22
  | .typeVector .. => 23
  | .typeRuntimeArray .. => 29
  | .typeStruct .. => 30
  | .typePointer .. => 32
  | .typeFunction .. => 33
  | .constant .. => 43
  | .function .. => 54
  | .functionEnd => 56
  | .variable .. => 59
  | .load .. => 61
  | .store .. => 62
  | .accessChain .. => 65
  | .compositeExtract .. => 81
  | .iadd .. => 128
  | .imul .. => 132
  | .select .. => 169
  | .ult .. => 176
  | .shiftRightLogical .. => 194
  | .bitAnd .. => 199
  | .sudot .. => 4452
  | .convertSToF .. => 111
  | .fadd .. => 129
  | .fmul .. => 133
  | .loopMerge .. => 246
  | .selectionMerge .. => 247
  | .label _ => 248
  | .branch _ => 249
  | .branchCond .. => 250
  | .ret => 253

/-- A fragment module: the linear instruction stream of one `GLCompute` entry point. The typed
sections (D2 ids, D3 types, D4 storage) are `WellFormed`'s concern in `Spirv.Decode`. -/
structure Module where
  /-- The emitted instruction stream, in binary order. -/
  code : List Inst

/-- The opcode set the emitter's served path (`gemv_dp4a_module_dyn`) can emit: 42 distinct opcode
numbers, re-derived mechanically from `gemv_dp4a_module_moe_sc`'s emission sites. This is the set
`IsDp4aFragment` admits — not the spike's 47 (whole function body) and not the transcription's 46. -/
def servedOpcodes : List Nat :=
  [10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 29, 30, 32, 33, 43, 54, 56, 59, 61, 62, 65,
    71, 72, 81, 111, 128, 129, 132, 133, 169, 176, 194, 199, 246, 247, 248, 249, 250,
    253, 4452]

/-- The float epilogue surface, inventoried but not in the v0 denotation (the named migration item
of the spec's epilogue premise). `OpConvertUToF` (112) is named here for completeness even though the
served path does not emit it. -/
def floatEpilogueOpcodes : List Nat := [12, 22, 111, 112, 129, 133]

/-- The v0 executing set: the served set minus the float epilogue surface. -/
def executingOpcodes : List Nat :=
  [10, 11, 14, 15, 16, 17, 19, 20, 21, 23, 29, 30, 32, 33, 43, 54, 56, 59, 61, 62, 65, 71, 72,
    81, 128, 132, 169, 176, 194, 199, 246, 247, 248, 249, 250, 253, 4452]

-- The served set is the derived inventory: 42 numbers, no duplicates.
#guard servedOpcodes.length = 42
#guard (servedOpcodes.eraseDups.length = servedOpcodes.length)

-- The v0 executing set is the derived inventory minus the named float surface: 37 numbers.
#guard executingOpcodes.length = 37
#guard executingOpcodes = servedOpcodes.removeAll floatEpilogueOpcodes

/-- Admission: every instruction of the module carries an opcode in the mechanically derived served
set. The float epilogue is admitted as inventoried syntax and is excluded from the denotation by
`executingOpcodes`. -/
def IsDp4aFragment (m : Module) : Prop :=
  ∀ i ∈ m.code, i.opcode ∈ servedOpcodes

/-- Membership in the v0 executing set: the domain of `Spirv.Semantics.eval`. -/
abbrev IsExecuting (i : Inst) : Prop := i.opcode ∈ executingOpcodes

-- The served set admits the executing core; the float surface is rejected as a denotation domain.
#guard decide (IsExecuting (.iadd 0 0 0 0))
#guard decide (IsExecuting (.imul 0 0 0 0))
#guard decide (IsExecuting (.sudot 0 0 0 0 .packed4x8))
#guard !decide (IsExecuting (.fadd 0 0 0 0))
#guard !decide (IsExecuting (.fmul 0 0 0 0))
#guard !decide (IsExecuting (.convertSToF 0 0 0))

/-- The explicit executing list is the derived difference: served minus the named float surface. -/
theorem executingOpcodes_eq :
    executingOpcodes = servedOpcodes.removeAll floatEpilogueOpcodes := by
  decide

/-- The served set admits the executing set. -/
theorem mem_served_of_mem_executing {o : Nat} (h : o ∈ executingOpcodes) : o ∈ servedOpcodes := by
  rw [executingOpcodes_eq] at h
  simp [List.removeAll] at h
  exact h.1

/-- The fragment admits its own executing core. -/
theorem fragment_of_executing {m : Module} (h : ∀ i ∈ m.code, IsExecuting i) :
    IsDp4aFragment m := by
  intro i hi
  exact mem_served_of_mem_executing (h i hi)

/-- `OpUMod` (137) is a named rejection, not an omission: it is outside the admitted set. -/
theorem umod_not_admitted : 137 ∉ servedOpcodes := by
  decide

/-- Barriers (`OpControlBarrier` 224) are outside the admitted set (phase 2). -/
theorem controlBarrier_not_admitted : 224 ∉ servedOpcodes := by
  decide

end Spirv
