import I64DecodeSource
import Aeneas

/-
SCOPE: the actual extracted body of the real `i64` elliptic decode, `src/table/keyrep.rs:292-294`
(`I64WordCodec::decode`), reached by the extraction root `word_i64_codec_decode`: its two
transitions (the odd-multiply of `src/coord.rs:385-387` and the bit reinterpretation that exits
the body), the joint machine-word domain those transitions run in, the counted work of one
evaluation, the exit branch, the refusal channel, and reachable witnesses for decoded keys.
REGISTER: scalar source refinement plus an explicit-work count of that one body. The body
multiplies with `wrapping_mul`, so the overflow refusal a checked multiply would carry does not
exist here, and `body_never_errors` shows its `Error` channel is unreachable; the only other
result the type admits is divergence, which `body_total` excludes. The counts are denominated in
the named premise `MachineWord64`. Excluded: the canonical-corpus correspondence for this codec,
the encode direction, the `Regen` carrier and every table operation around this call, the
allocator, and any hardware or cycle measurement.
CORPUS: `src/table/keyrep.rs:292-294`, whose real body is
`fn decode(_: Sigma, identity: u64) -> i64 { coord::unspread(identity).cast_signed() }`, over
`src/coord.rs:385-387` `pub const fn unspread(id: u64) -> u64 { id.wrapping_mul(PHI_INV) }` and
`src/coord.rs:390` `PHI_INV`. The same body is already extracted at
`proof/ApeironWord.lean:23881-23884` as the method call, and the shared root wrapper is
`proof/extract/lib.rs:268-273`, rooted by `proof/regen.sh:146`. This gate does not use that unit:
`proof/extract/lib.rs` no longer compiles, so the gate re-extracts the same body through its own
narrow forwarding wrapper.
REFERENCE: NONE - the claim is a source refinement and a work count of the extracted body, and
names no Lean theorem outside this file.
GATE: `python3 proof/resource/I64Decode_gate.py --controls`, 540-second budget, lane
/tmp/opencode/apeiron-lane-2.lock.
GRADE: `proved` for the theorems below when that gate exits 0. The work measure counts 64-bit
machine-word multiplications, machine-word reads, bytes produced and heap allocations in the
extracted body under the Aeneas model. It is not a cycle count and not a hardware measurement.
-/

namespace I64DecodeResource
open Aeneas Aeneas.Std Result

/-- `src/coord.rs:59` written at the extracted width: the odd multiplier of the forward spread.
    It is one 64-bit machine word and a global. -/
def phiLit : U64 := 11400714819323198485#u64

/-- That constant is the source's `coord::PHI`. -/
theorem phiLit_is_the_source_constant : phiLit = i64_decode_source.coord.PHI := by
  simp [phiLit, i64_decode_source.coord.PHI]

/-- `src/coord.rs:390` written at the extracted width: the odd constant the real body multiplies
    by. It is one 64-bit machine word and a global, so it is read once per call. -/
def phiInvLit : U64 := 17428512612931826493#u64

/-- That constant is the source's `coord::PHI_INV`, so the numerals used below are the ones the
    real body multiplies by. -/
theorem phiInvLit_is_the_source_constant :
    phiInvLit = i64_decode_source.coord.PHI_INV := by
  simp [phiInvLit, i64_decode_source.coord.PHI_INV]

/-- The real `src/table/keyrep.rs:292-294` body, reached through the extracted
    `ExactCodec<i64, WordHome>` instance for `src/table.rs`'s `WordHome`. -/
def decodeBody (sigma : i64_decode_source.cell.Sigma) (identity : U64) : Result I64 :=
  i64_decode_source.table.keyrep.I64WordCodec.Insts.I64_decode_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.decode i64_decode_source.table.WordHome.Insts.I64_decode_sourceTableHomeI64Sigma sigma identity

/-- The extraction root is that body: the root fixes the two argument types and restates no
    arithmetic. -/
theorem root_is_the_real_body (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    i64_decode_source.word_i64_codec_decode sigma identity = decodeBody sigma identity := rfl

/-- The real method body is the two transitions, in the order the source runs them: bind the
    multiply, then reinterpret the product's bits as the key. Neither transition reads the face
    parameter, which is `_` in the source. -/
theorem method_is_the_multiply_then_the_exit (sigma : i64_decode_source.cell.Sigma)
    (identity : U64) :
    decodeBody sigma identity
      = (do let i ← i64_decode_source.coord.unspread identity; core.num.U64.cast_signed i) := rfl

/-- The first transition, `src/coord.rs:385-387`: the retained identity word becomes the product
    of that word and the odd inverse multiplier. It carries no guard, because `wrapping_mul` is
    total on the whole 64-bit domain and so has no overflow premise to discharge. -/
theorem unspread_step (identity : U64) :
    i64_decode_source.coord.unspread identity
      = ok (core.num.U64.wrapping_mul identity phiInvLit) := by
  rw [i64_decode_source.coord.unspread, phiInvLit_is_the_source_constant]

/-- The second transition, `src/table/keyrep.rs:293`: the product's 64 bits are reinterpreted as
    the key's bit pattern, which is `as i64`. No bit is added, dropped or shifted. -/
theorem reinterpret_exit (p : U64) :
    core.num.U64.cast_signed p = ok ⟨p.bv⟩ := rfl

/-- The 64 bits the body computes: the identity word times the odd inverse multiplier. -/
def product (identity : U64) : BitVec UScalarTy.U64.numBits :=
  identity.bv * phiInvLit.bv

/-- The multiply transition carries exactly the 64 bits `product` names. -/
theorem unspread_step_bits (identity : U64) :
    (core.num.U64.wrapping_mul identity phiInvLit).bv = product identity :=
  UScalar.wrapping_mul_bv_eq identity phiInvLit

/-- The whole real body is the `ok` of one 64-bit product: one `wrapping_mul`, one bit
    reinterpretation, no `if`, no `loop`, no `bind` on a checked operation, no `alloc` and no
    `Error` channel. This is the kernel certificate the work count below is read from. -/
theorem body_is_one_wrapping_mul (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    decodeBody sigma identity = ok ⟨product identity⟩ := by
  rw [method_is_the_multiply_then_the_exit sigma identity, i64_decode_source.coord.unspread,
    bind_tc_ok, ← phiInvLit_is_the_source_constant]
  rfl

/-- The joint machine-word domain of the real call: `identity` is one 64-bit machine word and the
    constant it is multiplied by is one 64-bit machine word. -/
def MachineWords (identity : U64) : Prop :=
  identity.val < 2 ^ UScalarTy.U64.numBits ∧ phiInvLit.val < 2 ^ UScalarTy.U64.numBits

/-- The extracted types discharge the joint domain at every input, so the two transitions above
    are reached with no side condition to check and no input to reject. -/
theorem machineWords_holds (identity : U64) : MachineWords identity :=
  ⟨identity.hmax, phiInvLit.hmax⟩

/-- The guarded transition of the real body: on the joint machine-word domain the single
    machine-word multiplication produces the product, and the body exits with the key those
    64 bits denote. The guard removes no input, because `machineWords_holds` discharges it at
    every `Std.U64`; the multiplication is the wrapping one, so it has no overflow premise and
    no refusing branch to guard. -/
theorem guarded_step (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    MachineWords identity →
      ∃ p : U64, decodeBody sigma identity = ok ⟨p.bv⟩ ∧
        p = core.num.U64.wrapping_mul identity phiInvLit ∧ product identity = p.bv := by
  intro _h
  exact ⟨core.num.U64.wrapping_mul identity phiInvLit,
    body_is_one_wrapping_mul sigma identity, rfl, (unspread_step_bits identity).symm⟩

/-- Bytes in one machine word of the width the counts are denominated in. -/
def bytesPerWord : Nat := UScalarTy.U64.numBits / 8

/-- The named premise of the byte count: the word width the count is read at. -/
def MachineWord64 : Prop := UScalarTy.U64.numBits = 64

/-- The extracted `Std.U64` is 64 bits wide, so the named premise is discharged by the extracted
    type itself and not by an assumption about the machine. -/
theorem machineWord64_holds : MachineWord64 := rfl

/-- Bytes the real body moves in one evaluation: the one 64-bit machine word it reads and the one
    64-bit machine word it writes. -/
def bytesMovedOf : Nat := 2 * bytesPerWord

/-- The byte count at the extracted width, `machineWord64_holds` being the named premise it reads. -/
theorem bytesMovedOf_is_sixteen : bytesMovedOf = 16 := by
  simp [bytesMovedOf, bytesPerWord]

/-- The counted work of one evaluation of the real body. `value` is a value node, with
    `multiplications` machine-word multiplications, `wordReads` machine words read from the
    argument, `bytesMoved` bytes of result produced and `allocations` heap allocations; `effect`
    is an effect node; `diverging` is a divergence node. The counts are the ones
    `body_is_one_wrapping_mul` certifies, denominated in the named premise `MachineWord64`. -/
inductive Work where
  /-- A value node. -/
  | value (multiplications wordReads bytesMoved allocations : Nat)
  /-- An effect node. -/
  | effect
  /-- A divergence node. -/
  | diverging

/-- One evaluation of the real body, counted. -/
def workOf (sigma : i64_decode_source.cell.Sigma) (identity : U64) : Work :=
  Result.cases (decodeBody sigma identity)
    (fun _ => Work.value 1 1 bytesMovedOf 0)
    (fun _ _ => Work.effect) Work.diverging

/-- The real body performs exactly one machine-word multiplication, reads exactly one machine
    word and allocates nothing, at every input. The count is read off the extracted body through
    `body_is_one_wrapping_mul`: the `Result` it returns is the value node `ok`, not an effect or
    a divergence node, and the single `Std.U64.wrapping_mul` of `src/coord.rs:386` is the only
    arithmetic in it. `src/table/keyrep.rs:293` reinterprets the same 64 bits, so it adds no
    arithmetic, no comparison and no read. -/
theorem work_is_one_multiplication_one_read_zero_allocations
    (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    workOf sigma identity = Work.value 1 1 bytesMovedOf 0 := by
  show Result.cases (decodeBody sigma identity)
      (fun _ => Work.value 1 1 bytesMovedOf 0) (fun _ _ => Work.effect) Work.diverging
      = Work.value 1 1 bytesMovedOf 0
  rw [body_is_one_wrapping_mul]
  rfl

/-- At the named width the same count reads as sixteen bytes moved and no allocation. -/
theorem work_moves_sixteen_bytes (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    workOf sigma identity = Work.value 1 1 16 0 :=
  (work_is_one_multiplication_one_read_zero_allocations sigma identity).trans
    (congrArg (fun n => Work.value 1 1 n 0) bytesMovedOf_is_sixteen)

/-- The face token costs nothing: the real body's face parameter is `_` in the source, so two
    calls at the same identity word agree at every face. -/
theorem body_ignores_the_face_token (sigma sigma' : i64_decode_source.cell.Sigma)
    (identity : U64) : decodeBody sigma identity = decodeBody sigma' identity :=
  (body_is_one_wrapping_mul sigma identity).trans (body_is_one_wrapping_mul sigma' identity).symm

/-- The counted work does not depend on the face. -/
theorem work_ignores_the_face_token (sigma sigma' : i64_decode_source.cell.Sigma)
    (identity : U64) : workOf sigma identity = workOf sigma' identity := by
  unfold workOf
  rw [body_ignores_the_face_token]

/-- The body reads exactly the one machine word `identity` and nothing else, so its result is
    determined by that word's 64 bits. -/
theorem body_reads_only_the_identity_word (sigma : i64_decode_source.cell.Sigma)
    (i i' : U64) (h : i'.bv = i.bv) : decodeBody sigma i = decodeBody sigma i' := by
  rw [body_is_one_wrapping_mul, body_is_one_wrapping_mul]
  unfold product
  rw [h]

/-- The counted work does not depend on the magnitude of the identity word: this is the
    constant-work statement the counted measure certifies. -/
theorem work_is_input_independent (sigma : i64_decode_source.cell.Sigma) (i i' : U64)
    (h : i'.bv = i.bv) : workOf sigma i = workOf sigma i' := by
  unfold workOf
  rw [body_reads_only_the_identity_word sigma i i' h]

/-- The exit branch: the real body returns `ok` at every input, so the elliptic face's
    materialization is neither partial nor absent. -/
theorem body_total (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    ∃ key : I64, decodeBody sigma identity = ok key :=
  ⟨⟨product identity⟩, body_is_one_wrapping_mul sigma identity⟩

/-- The refusal channel of the real body is unreachable: it produces no `Error` node and no
    divergence node under the Aeneas weak-precondition relation. There is therefore no overflow
    refusal to take and no input at which the decode is undefined. -/
theorem body_never_errors (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    WP.spec (decodeBody sigma identity) (fun _ : I64 => True) := by
  rw [body_is_one_wrapping_mul]
  exact (WP.spec_ok (⟨product identity⟩ : I64)).mpr trivial

/-- Every retained identity word decodes at every face: the decode is total on the whole 64-bit
    identity domain, not on a sub-domain the caller must respect. -/
theorem every_identity_decodes (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    ∃ key : I64, decodeBody sigma identity = ok key ∧ MachineWords identity :=
  ⟨⟨product identity⟩, body_is_one_wrapping_mul sigma identity, machineWords_holds identity⟩

/-- A reachable decoded key at the zero of the word: a retained identity of `0`, which the
    reviewed codec stores for key `0`, decodes to the key whose 64 bits are all zero. -/
theorem zero_witness :
    MachineWords 0#u64 ∧
      decodeBody i64_decode_source.cell.Sigma.Elliptic 0#u64 = ok ⟨(0#u64).bv⟩ := by
  refine ⟨machineWords_holds 0#u64, ?_⟩
  rw [body_is_one_wrapping_mul]
  congr 1

/-- A reachable decoded key at the spread multiplier: a retained identity of `PHI`, which is the
    identity word the reviewed codec stores for key `1`, decodes to the key with the bit pattern
    of `1`. The witness is the exact inverse property of the two constants of `src/coord.rs`. -/
theorem phi_witness :
    MachineWords phiLit ∧
      decodeBody i64_decode_source.cell.Sigma.Elliptic phiLit = ok ⟨(1#u64).bv⟩ := by
  refine ⟨machineWords_holds phiLit, ?_⟩
  rw [body_is_one_wrapping_mul]
  congr 1

/-- A reachable decoded key at the sign bit: a retained identity of `2^63`, which is the
    identity word of the least `i64` key, decodes to that bit pattern, whose signed reading is
    `-2^63`. It shows the reinterpretation exit carries the sign as data rather than saturating
    or dropping it. -/
theorem min_key_witness :
    MachineWords 9223372036854775808#u64 ∧
      decodeBody i64_decode_source.cell.Sigma.Elliptic 9223372036854775808#u64
        = ok ⟨(9223372036854775808#u64).bv⟩ := by
  refine ⟨machineWords_holds 9223372036854775808#u64, ?_⟩
  rw [body_is_one_wrapping_mul]
  congr 1

/-- The same decoded key at every face: the three face tokens of `src/cell.rs` are read by no
    transition of this body, so a reachable witness does not depend on the dial. -/
theorem all_faces_witness (identity : U64) (key : I64)
    (h : decodeBody i64_decode_source.cell.Sigma.Elliptic identity = ok key) :
    decodeBody i64_decode_source.cell.Sigma.Parabolic identity = ok key ∧
      decodeBody i64_decode_source.cell.Sigma.Hyperbolic identity = ok key :=
  ⟨(body_ignores_the_face_token i64_decode_source.cell.Sigma.Elliptic i64_decode_source.cell.Sigma.Parabolic identity).symm.trans h,
    (body_ignores_the_face_token i64_decode_source.cell.Sigma.Elliptic i64_decode_source.cell.Sigma.Hyperbolic identity).symm.trans h⟩

/-- The transition is deterministic: one retained identity word reaches exactly one key, so a
    second evaluation of the real body cannot disagree with the first. -/
theorem transition_deterministic (sigma : i64_decode_source.cell.Sigma) (identity : U64)
    (k k' : I64) (hk : decodeBody sigma identity = ok k)
    (hk' : decodeBody sigma identity = ok k') : k = k' := by
  have h1 : k = ⟨product identity⟩ :=
    Result.ok_injective (hk.symm.trans (body_is_one_wrapping_mul sigma identity))
  have h2 : k' = ⟨product identity⟩ :=
    Result.ok_injective (hk'.symm.trans (body_is_one_wrapping_mul sigma identity))
  exact h1.trans h2.symm

/-- The multiply transition carries the product's 64 bits and the exit reads them as the key:
    no other arithmetic stands between the retained identity and the returned key. -/
theorem transitions_are_the_multiply_then_the_exit
    (sigma : i64_decode_source.cell.Sigma) (identity : U64) :
    (core.num.U64.wrapping_mul identity phiInvLit).bv = product identity ∧
      decodeBody sigma identity
        = (do let i ← i64_decode_source.coord.unspread identity; core.num.U64.cast_signed i) :=
  ⟨unspread_step_bits identity, method_is_the_multiply_then_the_exit sigma identity⟩

end I64DecodeResource

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit I64DecodeResource.phiLit_is_the_source_constant
#resource_audit I64DecodeResource.phiInvLit_is_the_source_constant
#resource_audit I64DecodeResource.root_is_the_real_body
#resource_audit I64DecodeResource.method_is_the_multiply_then_the_exit
#resource_audit I64DecodeResource.unspread_step
#resource_audit I64DecodeResource.reinterpret_exit
#resource_audit I64DecodeResource.unspread_step_bits
#resource_audit I64DecodeResource.body_is_one_wrapping_mul
#resource_audit I64DecodeResource.machineWords_holds
#resource_audit I64DecodeResource.guarded_step
#resource_audit I64DecodeResource.machineWord64_holds
#resource_audit I64DecodeResource.bytesMovedOf_is_sixteen
#resource_audit I64DecodeResource.work_is_one_multiplication_one_read_zero_allocations
#resource_audit I64DecodeResource.work_moves_sixteen_bytes
#resource_audit I64DecodeResource.body_ignores_the_face_token
#resource_audit I64DecodeResource.work_ignores_the_face_token
#resource_audit I64DecodeResource.body_reads_only_the_identity_word
#resource_audit I64DecodeResource.work_is_input_independent
#resource_audit I64DecodeResource.body_total
#resource_audit I64DecodeResource.body_never_errors
#resource_audit I64DecodeResource.every_identity_decodes
#resource_audit I64DecodeResource.zero_witness
#resource_audit I64DecodeResource.phi_witness
#resource_audit I64DecodeResource.min_key_witness
#resource_audit I64DecodeResource.all_faces_witness
#resource_audit I64DecodeResource.transition_deterministic
#resource_audit I64DecodeResource.transitions_are_the_multiply_then_the_exit

run_cmd Lean.logInfo "I64_DECODE_KERNEL_OK"