import CoordFinishSource
import Aeneas

/-
SCOPE: the extracted `src/coord.rs:158-164` `coord::finish` chain: its certified shape,
the eight primitive `U64` transitions with the model's guards, the closed-form work
value, the exit branch, and a reachable success witness.
REGISTER: scalar source refinement and a straight-line resource measure. Excluded: the
bit-level avalanche claim, latency, allocation totals, the `spread`/`unspread`
identities, the table key codec, and every other root of `proof/extract/lib.rs`.
CORPUS: `src/coord.rs:158-164`, extracted as `coord_finish_source.coord.finish` by
`proof/resource/CoordFinish_gate.py`; the same real body is rooted in the shared unit at
`proof/extract/lib.rs:491-493` (`word_coord_finish`), which this gate does not use.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean, `apeiron_word::coord::finish`
GATE: proof/resource/CoordFinish_gate.py --controls, 540 s budget, lane
/tmp/opencode/apeiron-lane-1.lock, MemoryMax=12G, MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: `proved` for every theorem below, at the Aeneas model of `U64` operations. The
work measure counts model primitives, not cycles; no hardware timing is claimed.
-/

namespace CoordFinishResource

open Aeneas Aeneas.Std Result

/-- The `U64` modulus, which every `core.num.U64.wrapping_mul` of the real body reduces
    its product modulo. The model's `UScalar.size` is an `irreducible_def`, so the
    concrete modulus is stated and used to read the two multiplies as `Nat` products. -/
theorem u64_size : U64.size = 18446744073709551616 := by
  simp [U64.size, U64.numBits]

/-- The word-value closed form of the real `src/coord.rs:158-164` chain, in the source's
    order: right shift by 30, xor, wrapping multiply by `0xBF58_476D_1CE4_E5B9`, right
    shift by 27, xor, wrapping multiply by `0x94D0_49BB_1331_11EB`, right shift by 31,
    xor. The two multiplicands are the Rust literals as the pinned extractor prints
    them, and each product is reduced modulo `2 ^ 64`, which is exactly what
    `u64::wrapping_mul` computes. -/
def finishValue (n : Nat) : Nat :=
  let h1 := n ^^^ (n >>> 30)
  let h2 := (h1 * 13787848793156543929) % 18446744073709551616
  let h3 := h2 ^^^ (h2 >>> 27)
  let h4 := (h3 * 10723151780598845931) % 18446744073709551616
  h4 ^^^ (h4 >>> 31)

/-- Shape certificate for the work measure. The real body is exactly these eight
    primitive `U64` operations of the Aeneas model -- three right shifts, three xors and
    two wrapping multiplies -- with no loop, no data-dependent branch, no allocation and
    no failure path; `finish_spec` below is the value of precisely those eight. Any
    further or missing primitive in the extracted body changes this equation and makes it
    fail to be checked. -/
theorem finish_shape (h : U64) :
    coord_finish_source.coord.finish h =
      (do
        let i ← h >>> 30#i32
        let h1 ← lift (h ^^^ i)
        let h2 ← lift (core.num.U64.wrapping_mul h1 13787848793156543929#u64)
        let i1 ← h2 >>> 27#i32
        let h3 ← lift (h2 ^^^ i1)
        let h4 ← lift (core.num.U64.wrapping_mul h3 10723151780598845931#u64)
        let i2 ← h4 >>> 31#i32
        ok (h4 ^^^ i2)) := rfl

/-- Transition 1 of 8: the first operation of the real body is a logical right shift by
    30. The model guards the shift by its width, and `30 < 64` discharges that guard, so
    the bind cannot fail and the shifted word is `h.val >>> 30`. -/
theorem step_shift30 (h : U64) :
    ∃ i : U64, (h >>> 30#i32) = ok i ∧ i.val = h.val >>> 30 := by
  obtain ⟨i, hi, hiv, _⟩ :=
    WP.spec_imp_exists (U64.ShiftRight_IScalar_spec h (30#i32) (by norm_num) (by norm_num))
  exact ⟨i, hi, by simpa using hiv⟩

/-- Transition 2 of 8: the first xor of the real body, `src/coord.rs:159`. -/
theorem step_xor30 (h i : U64) :
    ∃ h1 : U64, lift (h ^^^ i) = ok h1 ∧ h1.val = h.val ^^^ i.val := by
  obtain ⟨h1, h1r, h1v, _⟩ := WP.spec_imp_exists (UScalar.xor_spec h i)
  exact ⟨h1, h1r, by simpa using h1v⟩

/-- Transition 3 of 8: the first wrapping multiply, `src/coord.rs:160`, by the odd
    constant `0xBF58_476D_1CE4_E5B9`; the model reduces the product modulo `2 ^ 64`. -/
theorem step_mul1 (h1 : U64) :
    ∃ h2 : U64, lift (core.num.U64.wrapping_mul h1 13787848793156543929#u64) = ok h2 ∧
      h2.val = (h1.val * 13787848793156543929) % 18446744073709551616 := by
  refine ⟨core.num.U64.wrapping_mul h1 13787848793156543929#u64, rfl, ?_⟩
  calc (core.num.U64.wrapping_mul h1 13787848793156543929#u64).val
      = h1.val * (13787848793156543929#u64).val % UScalar.size .U64 :=
        core.num.U64.wrapping_mul_val_eq h1 13787848793156543929#u64
    _ = (h1.val * 13787848793156543929) % 18446744073709551616 := by
        norm_num [u64_size]

/-- Transition 4 of 8: the second right shift, `src/coord.rs:161`; its guard is
    `27 < 64`, so the bind cannot fail. -/
theorem step_shift27 (h2 : U64) :
    ∃ i1 : U64, (h2 >>> 27#i32) = ok i1 ∧ i1.val = h2.val >>> 27 := by
  obtain ⟨i1, hi, hiv, _⟩ :=
    WP.spec_imp_exists (U64.ShiftRight_IScalar_spec h2 (27#i32) (by norm_num) (by norm_num))
  exact ⟨i1, hi, by simpa using hiv⟩

/-- Transition 5 of 8: the second xor of the real body, `src/coord.rs:161`. -/
theorem step_xor27 (h2 i1 : U64) :
    ∃ h3 : U64, lift (h2 ^^^ i1) = ok h3 ∧ h3.val = h2.val ^^^ i1.val := by
  obtain ⟨h3, h3r, h3v, _⟩ := WP.spec_imp_exists (UScalar.xor_spec h2 i1)
  exact ⟨h3, h3r, by simpa using h3v⟩

/-- Transition 6 of 8: the second wrapping multiply, `src/coord.rs:162`, by the odd
    constant `0x94D0_49BB_1331_11EB`. -/
theorem step_mul2 (h3 : U64) :
    ∃ h4 : U64, lift (core.num.U64.wrapping_mul h3 10723151780598845931#u64) = ok h4 ∧
      h4.val = (h3.val * 10723151780598845931) % 18446744073709551616 := by
  refine ⟨core.num.U64.wrapping_mul h3 10723151780598845931#u64, rfl, ?_⟩
  calc (core.num.U64.wrapping_mul h3 10723151780598845931#u64).val
      = h3.val * (10723151780598845931#u64).val % UScalar.size .U64 :=
        core.num.U64.wrapping_mul_val_eq h3 10723151780598845931#u64
    _ = (h3.val * 10723151780598845931) % 18446744073709551616 := by
        norm_num [u64_size]

/-- Transition 7 of 8: the third right shift, `src/coord.rs:163`; its guard is
    `31 < 64`. -/
theorem step_shift31 (h4 : U64) :
    ∃ i2 : U64, (h4 >>> 31#i32) = ok i2 ∧ i2.val = h4.val >>> 31 := by
  obtain ⟨i2, hi, hiv, _⟩ :=
    WP.spec_imp_exists (U64.ShiftRight_IScalar_spec h4 (31#i32) (by norm_num) (by norm_num))
  exact ⟨i2, hi, by simpa using hiv⟩

/-- Transition 8 of 8: the third xor of the real body, `src/coord.rs:163`, and the word
    it produces is the value the body returns. -/
theorem step_xor31 (h4 i2 : U64) :
    ∃ v : U64, lift (h4 ^^^ i2) = ok v ∧ v.val = h4.val ^^^ i2.val := by
  obtain ⟨v, hvr, hv, _⟩ := WP.spec_imp_exists (UScalar.xor_spec h4 i2)
  exact ⟨v, hvr, by simpa using hv⟩

/-- The eight transitions of the certified shape, composed in the real body's order,
    evaluate the body to the closed form. The proof rewrites the unfolded real body with
    the eight transition equations above, so the equation and the closed form cannot both
    survive a change to the extracted body. -/
theorem finish_spec (h : U64) :
    ∃ v : U64, coord_finish_source.coord.finish h = ok v ∧ v.val = finishValue h.val := by
  obtain ⟨i, hi, hival⟩ := step_shift30 h
  obtain ⟨h1, h1r, h1val⟩ := step_xor30 h i
  obtain ⟨h2, h2r, h2val⟩ := step_mul1 h1
  obtain ⟨i1, h1i, hi1val⟩ := step_shift27 h2
  obtain ⟨h3, h3r, h3val⟩ := step_xor27 h2 i1
  obtain ⟨h4, h4r, h4val⟩ := step_mul2 h3
  obtain ⟨i2, h4i, hi2val⟩ := step_shift31 h4
  obtain ⟨v, hvr, hvval⟩ := step_xor31 h4 i2
  have hval : v.val = finishValue h.val := by
    simp only [finishValue]
    -- Substitute each transition's value in the order the real body performs them.
    rw [← hival, ← h1val, ← h2val, ← hi1val, ← h3val, ← h4val, ← hi2val, ← hvval]
  refine ⟨v, ?_, hval⟩
  unfold coord_finish_source.coord.finish
  simp only [bind_tc_ok, hi, h1r, h2r, h1i, h3r, h4i, h4r]
  exact hvr

/-- The work measure, read as a count on the real body: for every input word the body
    performs exactly the eight primitives of `finish_shape`, in that order, every one of
    them succeeding, and the returned word is the value of the eighth. This is the
    resource claim: three shifts, three xors and two wrapping multiplies, no loop to
    rank, no data-dependent branch, no allocation, no partial operation. -/
theorem finish_work (h : U64) :
    ∃ (i h1 h2 i1 h3 h4 i2 v : U64),
      (h >>> 30#i32) = ok i ∧ lift (h ^^^ i) = ok h1 ∧
      lift (core.num.U64.wrapping_mul h1 13787848793156543929#u64) = ok h2 ∧
      (h2 >>> 27#i32) = ok i1 ∧ lift (h2 ^^^ i1) = ok h3 ∧
      lift (core.num.U64.wrapping_mul h3 10723151780598845931#u64) = ok h4 ∧
      (h4 >>> 31#i32) = ok i2 ∧ lift (h4 ^^^ i2) = ok v ∧
      coord_finish_source.coord.finish h = ok v ∧ v.val = finishValue h.val := by
  obtain ⟨v, hrun, hv⟩ := finish_spec h
  obtain ⟨i, hi, _⟩ := step_shift30 h
  obtain ⟨h1, h1r, _⟩ := step_xor30 h i
  obtain ⟨h2, h2r, _⟩ := step_mul1 h1
  obtain ⟨i1, h1i, _⟩ := step_shift27 h2
  obtain ⟨h3, h3r, _⟩ := step_xor27 h2 i1
  obtain ⟨h4, h4r, _⟩ := step_mul2 h3
  obtain ⟨i2, h4i, _⟩ := step_shift31 h4
  obtain ⟨v0, hv0, _⟩ := step_xor31 h4 i2
  -- The real body, evaluated through those eight equations, returns the last xor's word.
  have hbody : coord_finish_source.coord.finish h = ok (h4 ^^^ i2) := by
    unfold coord_finish_source.coord.finish
    simp only [bind_tc_ok, hi, h1r, h2r, h1i, h3r, h4i, h4r]
  have heq : v0 = v := Result.ok_injective (by
    calc ok v0 = lift (h4 ^^^ i2) := hv0.symm
      _ = coord_finish_source.coord.finish h := hbody.symm
      _ = ok v := hrun)
  refine ⟨i, h1, h2, i1, h3, h4, i2, v, hi, h1r, h2r, h1i, h3r, h4r, h4i, ?_, hrun, hv⟩
  rw [← heq]
  exact hv0

/-- The exit branch of the real body is the single `ok` of `src/coord.rs:163`: for every
    input word the body returns, and it returns the value of the eight transitions. -/
theorem finish_exit (h : U64) : ∃ v : U64, coord_finish_source.coord.finish h = ok v := by
  obtain ⟨v, hrun, _⟩ := finish_spec h
  exact ⟨v, hrun⟩

/-- The `Error` exit of the real body is unreachable: no input word reaches a `fail`. -/
theorem finish_no_fail (h : U64) (e : Error) :
    coord_finish_source.coord.finish h ≠ fail e := by
  obtain ⟨v, hrun, _⟩ := finish_spec h
  rw [hrun]
  exact ok_not_fail

/-- The stuck exit of the real body is unreachable: no input word reaches `div`. -/
theorem finish_no_div (h : U64) : coord_finish_source.coord.finish h ≠ (div : Result U64) := by
  obtain ⟨v, hrun, _⟩ := finish_spec h
  rw [hrun]
  exact ok_not_div

/-- Reachable success witness at the zero word: the real body takes all eight transitions
    and returns the zero word, so the `ok` exit is reached. -/
theorem finish_zero_witness : ∃ v : U64, coord_finish_source.coord.finish 0#u64 = ok v ∧ v.val = 0 := by
  obtain ⟨v, hrun, hv⟩ := finish_spec 0#u64
  refine ⟨v, hrun, ?_⟩
  rw [hv]
  norm_num [finishValue]

/-- Reachable success witness at a nonzero word: the real body takes all eight
    transitions on `1` and returns `6238072747940578789`, the value the closed form gives
    for that input, so the `ok` exit is reached with a changed word. -/
theorem finish_one_witness :
    ∃ v : U64, coord_finish_source.coord.finish 1#u64 = ok v ∧ v.val = 6238072747940578789 := by
  obtain ⟨v, hrun, hv⟩ := finish_spec 1#u64
  refine ⟨v, hrun, ?_⟩
  rw [hv]
  have hone : (1#u64).val = 1 := by decide
  rw [hone]
  decide

end CoordFinishResource

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

#resource_audit CoordFinishResource.u64_size
#resource_audit CoordFinishResource.finish_shape
#resource_audit CoordFinishResource.step_shift30
#resource_audit CoordFinishResource.step_xor30
#resource_audit CoordFinishResource.step_mul1
#resource_audit CoordFinishResource.step_shift27
#resource_audit CoordFinishResource.step_xor27
#resource_audit CoordFinishResource.step_mul2
#resource_audit CoordFinishResource.step_shift31
#resource_audit CoordFinishResource.step_xor31
#resource_audit CoordFinishResource.finish_spec
#resource_audit CoordFinishResource.finish_work
#resource_audit CoordFinishResource.finish_exit
#resource_audit CoordFinishResource.finish_no_fail
#resource_audit CoordFinishResource.finish_no_div
#resource_audit CoordFinishResource.finish_zero_witness
#resource_audit CoordFinishResource.finish_one_witness

run_cmd Lean.logInfo "COORD_FINISH_KERNEL_OK"
