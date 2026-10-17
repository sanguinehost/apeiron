import ApeironWord
import Learn.Solve.WsyrkSpec
import Learn.Grad.DenseLinearSpec
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted `src/learn/solve/syrk.rs`: refinement

`solve_port.syrk.{all_below, within_bound, round32, round64, hadamard_into, wsyrk}` and the shared
`solve_port.q32.narrow` are
refined to the independent `ℤ`/`List ℤ` model in `Learn.Solve.WsyrkSpec`: the accepted-domain bound
check (`wsyrkAllBelowRef`, `withinBoundRef`) against the strict `±2³⁶` boundary, the Q32.32 / Q96.96
rounders, and the elementwise in-place Hadamard fold. The loop refinements are stated over the
extracted `_loop` bodies via `loop.spec_decr_nat`, never by kernel evaluation.

Trust boundary (§4 obligation 11): rustc/LLVM codegen, the kernel ABI, the `unsafe` container and
the vector layout are not covered; only the extracted integer body. One further, named boundary:
`solve_port.syrk.round32` and `round64` route their shift through the **opaque boundary axiom**
`core.option.Option.map` (`proof/ApeironWord.lean`; admitted in `proof/boundary-axioms.txt` as a
std operation absent from the pinned Aeneas model). Its `some`-branch law is therefore stated as an
explicit premise `MapSome`, not derived; the theorems that consume it are conditional and labelled.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## 1. Value lemmas -/

theorem syrk_bound_bits_val : solve_port.syrk.BOUND_BITS.val = 36 := by
  unfold solve_port.syrk.BOUND_BITS
  rfl

theorem syrk_block_val : solve_port.syrk.BLOCK.val = 64 := by
  unfold solve_port.syrk.BLOCK
  rfl

theorem syrk_shift32_toNat : (32#i32).toNat = 32 := by decide

theorem syrk_shift64_toNat : (64#i32).toNat = 64 := by decide

/-- The `u64` carrier of `BitVec.ofNat 64 m` reads back `m` modulo `2⁶⁴`. -/
theorem u64_ofNat_val (m : ℕ) : (⟨BitVec.ofNat 64 m⟩ : U64).val = m % 2 ^ 64 := rfl

/-- An `i64` magnitude fits `u64`. -/
theorem syrk_i64_natAbs_lt (z : I64) : z.val.natAbs < 2 ^ 64 := by
  have h := z.hBounds
  norm_num [IScalarTy.numBits] at h
  have h3 : (z.val.natAbs : ℤ) < (2 : ℤ) ^ 64 := by
    rw [Int.natCast_natAbs, abs_lt]
    constructor <;> linarith [h.1, h.2]
  exact_mod_cast h3

/-- The magnitude `core.num.I64.unsigned_abs` returns, as a `u64` word. -/
def uabs (z : I64) : U64 := ⟨BitVec.ofNat 64 z.val.natAbs⟩

theorem uabs_eq (z : I64) : core.num.I64.unsigned_abs z = ok (uabs z) := rfl

theorem uabs_val (z : I64) : (uabs z).val = z.val.natAbs := by
  unfold uabs
  rw [u64_ofNat_val, Nat.mod_eq_of_lt (syrk_i64_natAbs_lt z)]

/-- The `u64` limit `1 <<< 36` the accepted-domain check compares against. -/
def boundLimit : U64 := ⟨BitVec.ofNat 64 (2 ^ 36)⟩

theorem boundLimit_val : boundLimit.val = 2 ^ 36 := by
  unfold boundLimit
  rw [u64_ofNat_val, Nat.mod_eq_of_lt (by norm_num)]

/-- The extracted `1#u64 <<< BOUND_BITS` is total and equals `2³⁶`. -/
theorem syrk_bound_limit : ∃ l : U64, (1#u64 <<< solve_port.syrk.BOUND_BITS) = ok l ∧
    l.val = 2 ^ 36 := by
  refine ⟨boundLimit, ?_, boundLimit_val⟩
  show UScalar.shiftLeft_UScalar 1#u64 solve_port.syrk.BOUND_BITS = ok boundLimit
  unfold UScalar.shiftLeft_UScalar UScalar.shiftLeft
  rw [syrk_bound_bits_val, if_pos (by decide : 36 < UScalarTy.U64.numBits)]
  rfl

/-- The extracted `1#i128 <<< 63#i32` equals `2⁶³`. -/
theorem syrk_shl63 : (1#i128 <<< 63#i32) ⦃ z => z.val = (2 : ℤ) ^ 63 ⦄ := by
  step
  rw [z_post]
  norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]

/-! ## 2. The accepted-domain magnitude bound -/

/-- Every entry of `xs` has a magnitude strictly below `limit`. -/
def wsyrkAllBelowRef (xs : Slice I64) (limit : U64) : Prop :=
  ∀ t < xs.val.length, (uabs (xs.val.getD t default)).val < limit.val

/-- The `all_below` loop invariant: `b` is the conjunction of the comparisons below `i`. -/
def wsyrkAllBelowInv (xs : Slice I64) (limit : U64) (b : Bool) (i : Usize) : Prop :=
  i.val ≤ xs.val.length ∧
  (b = true ↔ ∀ t < i.val, (uabs (xs.val.getD t default)).val < limit.val)

theorem wsyrkAllBelowRef_congr {xs : Slice I64} {l l' : U64} (h : l.val = l'.val) :
    wsyrkAllBelowRef xs l ↔ wsyrkAllBelowRef xs l' := by
  unfold wsyrkAllBelowRef
  rw [h]

/-- The `all_below` body on the continuing path. -/
theorem all_below_loop_body_exec_cont (xs : Slice I64) (limit : U64) (i : Usize)
    (i1 : Usize) (hb : i < Slice.len xs) (hlen : i.val < xs.val.length)
    (hi1 : i + 1#usize = ok i1) :
    solve_port.syrk.all_below_loop.body xs limit true i =
      ok (ControlFlow.cont (decide (uabs (xs.val.getD i.val default) < limit), i1)) := by
  have hidx := dense_slice_index_getD xs i hlen
  unfold solve_port.syrk.all_below_loop.body
  dsimp only
  rw [if_pos hb, if_pos rfl]
  simp only [hidx, uabs_eq, hi1, bind_tc_ok]

/-- One iteration of the `all_below` loop advances the model. -/
theorem all_below_loop_body_step (xs : Slice I64) (limit : U64) (i : Usize)
    (hlen : i.val < xs.val.length) (hinv : wsyrkAllBelowInv xs limit true i) :
    WP.spec (solve_port.syrk.all_below_loop.body xs limit true i)
      (fun cf => ∃ b' i', cf = ControlFlow.cont (b', i') ∧ i'.val = i.val + 1 ∧
        wsyrkAllBelowInv xs limit b' i') := by
  obtain ⟨hile, hiff⟩ := hinv
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize)
      (by show i.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          have := Slice.property xs
          omega))
  have hi1_val : i1.val = i.val + 1 := by rw [hi1v]; rfl
  have hb : i < Slice.len xs :=
    (UScalar.lt_equiv i (Slice.len xs)).2 (by rw [Slice.len_val]; exact hlen)
  have hexec := all_below_loop_body_exec_cont xs limit i i1 hb hlen hi1
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (decide (uabs (xs.val.getD i.val default) < limit), i1), hexec,
    decide (uabs (xs.val.getD i.val default) < limit), i1, rfl, hi1_val, ?_⟩
  refine ⟨by omega, ?_⟩
  rw [hi1_val]
  constructor
  · intro hbo
    have hp : (uabs (xs.val.getD i.val default)).val < limit.val :=
      (UScalar.lt_equiv _ _).1 (decide_eq_true_iff.mp hbo)
    intro t ht
    rcases Nat.lt_succ_iff_lt_or_eq.mp ht with hlt | heq
    · exact (hiff.mp rfl) t hlt
    · rw [heq]; exact hp
  · intro hall
    exact decide_eq_true_iff.mpr ((UScalar.lt_equiv _ _).2 (hall i.val (by omega)))

/-- Full specification of the `all_below` loop. -/
theorem all_below_loop_spec (xs : Slice I64) (limit : U64) (b : Bool) (i : Usize)
    (hinv0 : wsyrkAllBelowInv xs limit b i) :
    WP.spec (solve_port.syrk.all_below_loop xs limit b i)
      (fun r => r = true ↔ wsyrkAllBelowRef xs limit) := by
  unfold solve_port.syrk.all_below_loop
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Usize => xs.val.length - s.2.val)
    (inv := fun s => wsyrkAllBelowInv xs limit s.1 s.2)
    (post := fun r : Bool => r = true ↔ wsyrkAllBelowRef xs limit)
  · rintro ⟨ok', i'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hile, hiff⟩ := hinv
    by_cases hok : ok' = true
    · subst hok
      by_cases hlt : i'.val < xs.val.length
      · obtain ⟨cf, hstep, b'', i'', hcf, hi''_val, hinv''⟩ := WP.spec_imp_exists
          (all_below_loop_body_step xs limit i' hlt ⟨hile, hiff⟩)
        rw [hcf] at hstep
        simp only [hstep]
        refine ⟨hinv'', ?_⟩
        dsimp only
        omega
      · have hbody : solve_port.syrk.all_below_loop.body xs limit true i' =
            ok (ControlFlow.done true) := by
          unfold solve_port.syrk.all_below_loop.body
          dsimp only
          rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hlt)]
        simp only [hbody]
        rw [WP.spec_ok]
        dsimp only
        have hi'_eq : i'.val = xs.val.length := by omega
        rw [hi'_eq] at hiff
        exact hiff
    · have hok' : ok' = false := by
        cases ok' with
        | false => rfl
        | true => exact absurd rfl hok
      have hbody : solve_port.syrk.all_below_loop.body xs limit ok' i' =
            ok (ControlFlow.done false) := by
        unfold solve_port.syrk.all_below_loop.body
        dsimp only
        by_cases hlt : i' < Slice.len xs
        · rw [if_pos hlt, if_neg hok]
        · rw [if_neg hlt, hok']
      simp only [hbody]
      rw [WP.spec_ok]
      dsimp only
      rw [hok', Bool.false_eq_true] at hiff
      rw [Bool.false_eq_true, false_iff]
      intro hall
      exact hiff.mpr (fun t ht => hall t (by omega))
  · exact hinv0

/-- `all_below` decides the accepted-domain magnitude bound. -/
theorem all_below_spec (xs : Slice I64) (limit : U64) :
    ∃ b : Bool, solve_port.syrk.all_below xs limit = ok b ∧
      (b = true ↔ wsyrkAllBelowRef xs limit) := by
  have hinv0 : wsyrkAllBelowInv xs limit true 0#usize := by
    refine ⟨?_, ?_⟩
    · show (0#usize).val ≤ xs.val.length
      rw [show (0#usize).val = 0 from rfl]
      omega
    · rw [show (0#usize).val = 0 from rfl]
      simp
  have hspec := all_below_loop_spec xs limit true 0#usize hinv0
  obtain ⟨r, hexec, hpost⟩ := (WP.spec_equiv_exists _ _).mp hspec
  exact ⟨r, by unfold solve_port.syrk.all_below; exact hexec, hpost⟩

/-! ## 3. `within_bound` -/

/-- `within_bound`'s reference: `x` and, when present, `w` are inside the strict `±2³⁶` boundary. -/
def withinBoundRef (x : Slice I64) (w : Option (Slice I64)) : Prop :=
  wsyrkAllBelowRef x boundLimit ∧ ∀ ws, w = some ws → wsyrkAllBelowRef ws boundLimit

theorem syrk_within_bound_true (x : Slice I64) (w : Option (Slice I64))
    (h : withinBoundRef x w) : solve_port.syrk.within_bound x w = ok true := by
  obtain ⟨l, hl, hlv⟩ := syrk_bound_limit
  have hcongr : ∀ {y : Slice I64}, wsyrkAllBelowRef y l ↔ wsyrkAllBelowRef y boundLimit :=
    fun {y} => wsyrkAllBelowRef_congr (xs := y) (by rw [hlv, boundLimit_val])
  obtain ⟨bx, hbx, hbxiff⟩ := all_below_spec x l
  have hbxT : bx = true := hbxiff.mpr (hcongr.mpr h.1)
  cases w with
  | none =>
    unfold solve_port.syrk.within_bound
    rw [hl]
    simp only [hbx, bind_tc_ok, hbxT, if_true]
  | some ws =>
    have hws : wsyrkAllBelowRef ws l := hcongr.mpr (h.2 ws rfl)
    obtain ⟨bw, hbw, hbwiff⟩ := all_below_spec ws l
    have hbwT : bw = true := hbwiff.mpr hws
    unfold solve_port.syrk.within_bound
    rw [hl]
    simp only [hbx, bind_tc_ok, hbxT, if_true]
    rw [hbw, hbwT]

theorem syrk_within_bound_false (x : Slice I64) (w : Option (Slice I64))
    (h : ¬ withinBoundRef x w) : solve_port.syrk.within_bound x w = ok false := by
  obtain ⟨l, hl, hlv⟩ := syrk_bound_limit
  have hcongr : ∀ {y : Slice I64}, wsyrkAllBelowRef y l ↔ wsyrkAllBelowRef y boundLimit :=
    fun {y} => wsyrkAllBelowRef_congr (xs := y) (by rw [hlv, boundLimit_val])
  obtain ⟨bx, hbx, hbxiff⟩ := all_below_spec x l
  cases w with
  | none =>
    have hnx : ¬ wsyrkAllBelowRef x l :=
      fun hc => h ⟨hcongr.mp hc, by intro ws hws; exact absurd hws (by simp)⟩
    have hbxF : bx = false := by
      cases bx with
      | false => rfl
      | true => exact absurd (hbxiff.mp rfl) hnx
    unfold solve_port.syrk.within_bound
    rw [hl]
    simp only [hbx, bind_tc_ok, hbxF, Bool.false_eq_true, if_false]
  | some ws =>
    by_cases hx : wsyrkAllBelowRef x boundLimit
    · have hws : ¬ wsyrkAllBelowRef ws l :=
        fun hc => h ⟨hx, fun ws' hws' => by
          obtain rfl := Option.some.inj hws'
          exact hcongr.mp hc⟩
      obtain ⟨bw, hbw, hbwiff⟩ := all_below_spec ws l
      have hbwF : bw = false := by
        cases bw with
        | false => rfl
        | true => exact absurd (hbwiff.mp rfl) hws
      have hbxT : bx = true := hbxiff.mpr (hcongr.mpr hx)
      unfold solve_port.syrk.within_bound
      rw [hl]
      simp only [hbx, bind_tc_ok, hbxT, if_true]
      rw [hbw, hbwF]
    · have hnx : ¬ wsyrkAllBelowRef x l := fun hc => hx (hcongr.mp hc)
      have hbxF : bx = false := by
        cases bx with
        | false => rfl
        | true => exact absurd (hbxiff.mp rfl) hnx
      unfold solve_port.syrk.within_bound
      rw [hl]
      simp only [hbx, bind_tc_ok, hbxF, Bool.false_eq_true, if_false]

/-- `within_bound` decides the strict `±2³⁶` accepted-domain bound. -/
theorem syrk_within_bound_spec (x : Slice I64) (w : Option (Slice I64)) :
    ∃ b : Bool, solve_port.syrk.within_bound x w = ok b ∧
      (b = true ↔ withinBoundRef x w) := by
  by_cases h : withinBoundRef x w
  · exact ⟨true, syrk_within_bound_true x w h, by simp [h]⟩
  · refine ⟨false, syrk_within_bound_false x w h, ?_⟩
    rw [Bool.false_eq_true, false_iff]
    exact h

/-! ## 4. The rounders and the `core.option.Option.map` boundary -/

/-- The `some`-branch law of the **opaque boundary axiom** `core.option.Option.map`
(`proof/ApeironWord.lean`; admitted in `proof/boundary-axioms.txt` as a std operation absent from
the pinned Aeneas model): on `some u` it runs the closure and wraps its value, on `none` it is
`ok none`. The model patch supplies no body for `map`, so this law is a premise, not a theorem;
every statement below that consumes it is conditional on it and labelled. -/
def MapSome : Prop :=
  (∀ {T U F : Type} (inst : core.ops.function.FnOnce F T U) (f : F) (u : T),
      core.option.Option.map inst (some u) f =
        (inst.call_once f u >>= fun x => Result.ok (some x)))
  ∧ (∀ {T U F : Type} (inst : core.ops.function.FnOnce F T U) (f : F),
      core.option.Option.map inst none f = Result.ok none)

/-- `Usize` is at least 32-bit, so `18 < Usize.numBits`. The `Usize` width is the opaque
`System.Platform.numBits`, which `decide` cannot reduce; the case split is the model's own route. -/
theorem syrk_usize_numBits_gt18 : 18 < UScalarTy.Usize.numBits := by
  rw [UScalarTy.Usize_numBits_eq]
  rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> decide

/-- The extracted `MAX_O = 1#usize <<< 18` is total and equals `2¹⁸`. -/
theorem syrk_max_o : ∃ m : Usize, solve_port.syrk.MAX_O = ok m ∧ m.val = 2 ^ 18 := by
  refine ⟨⟨(1#usize).bv.shiftLeft 18⟩, ?_, ?_⟩
  · unfold solve_port.syrk.MAX_O
    show UScalar.shiftLeft_IScalar 1#usize 18#i32 = ok _
    unfold UScalar.shiftLeft_IScalar UScalar.shiftLeft
    rw [if_pos (by decide : (18#i32).val ≥ 0),
        show (18#i32).toNat = 18 from by decide,
        if_pos syrk_usize_numBits_gt18]
  · show ((1#usize).bv.shiftLeft 18).toNat = 2 ^ 18
    rw [BitVec.shiftLeft_eq, BitVec.toNat_shiftLeft]
    have h1 : (1#usize).bv.toNat = 1 := rfl
    rw [h1, Nat.shiftLeft_eq, one_mul]
    apply Nat.mod_eq_of_lt
    exact Nat.pow_lt_pow_right (by norm_num) syrk_usize_numBits_gt18

/-- The extracted `round32` is the Q32.32 rounding by `2³²` of `(v + 2³¹)` on the fitting domain.
Conditional on `MapSome`, the unwitnessable `core.option.Option.map` boundary law. -/
theorem syrk_round32_spec (hmap : MapSome) (v : I128)
    (h : I128.min ≤ v.val + 2 ^ 31 ∧ v.val + 2 ^ 31 ≤ I128.max) :
    ∃ w : I128, solve_port.syrk.round32 v = ok (some w) ∧ w.val = rnd32 v.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists dense_shl31
  obtain ⟨s, hs, hsv⟩ := dense_checked_add_some (a := v) (b := c) (by rw [hcv]; exact h)
  obtain ⟨w, hw, hwv⟩ := WP.spec_imp_exists
    (IScalar.ShiftRight_IScalar_spec (x := s) (y := 32#i32) (by decide) (by decide))
  refine ⟨w, ?_, ?_⟩
  · unfold solve_port.syrk.round32
    rw [hc, bind_tc_ok]
    simp only [lift, hs, bind_tc_ok]
    rw [hmap.1 solve_port.syrk.round32.closure.Insts.CoreOpsFunctionFnOnceTupleI128I128 () s]
    show ((s >>> 32#i32) >>= fun x => Result.ok (some x)) = ok (some w)
    simp only [hw, bind_tc_ok]
  · rw [hwv.1, dense_shift32_toNat, hsv, hcv, rnd32]

/-- The extracted `round64` is the Q96.96 rounding by `2⁶⁴` of `(v + 2⁶³)` on the fitting domain.
Conditional on `MapSome`, the unwitnessable `core.option.Option.map` boundary law. -/
theorem syrk_round64_spec (hmap : MapSome) (v : I128)
    (h : I128.min ≤ v.val + 2 ^ 63 ∧ v.val + 2 ^ 63 ≤ I128.max) :
    ∃ w : I128, solve_port.syrk.round64 v = ok (some w) ∧ w.val = rnd64 v.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists syrk_shl63
  obtain ⟨s, hs, hsv⟩ := dense_checked_add_some (a := v) (b := c) (by rw [hcv]; exact h)
  obtain ⟨w, hw, hwv⟩ := WP.spec_imp_exists
    (IScalar.ShiftRight_IScalar_spec (x := s) (y := 64#i32) (by decide) (by decide))
  refine ⟨w, ?_, ?_⟩
  · unfold solve_port.syrk.round64
    rw [hc, bind_tc_ok]
    simp only [lift, hs, bind_tc_ok]
    rw [hmap.1 solve_port.syrk.round64.closure.Insts.CoreOpsFunctionFnOnceTupleI128I128 () s]
    show ((s >>> 64#i32) >>= fun x => Result.ok (some x)) = ok (some w)
    simp only [hw, bind_tc_ok]
  · rw [hwv.1, syrk_shift64_toNat, hsv, hcv, rnd64]

/-! ## 5. `narrow` -/

/-- The extracted `i64::MAX` widened to `i128` is `2⁶³ - 1`. -/
theorem syrk_i64max_val :
    (core.convert.num.FromI128I64.from core.num.I64.MAX).val = (2 : ℤ) ^ 63 - 1 := by
  rw [dense_fromI64_val]
  decide

/-- The extracted `i64::MIN` widened to `i128` is `-2⁶³`. -/
theorem syrk_i64min_val :
    (core.convert.num.FromI128I64.from core.num.I64.MIN).val = -((2 : ℤ) ^ 63) := by
  rw [dense_fromI64_val]
  decide

/-- The extracted `narrow` returns the value exactly on the `i64` window. Unconditional: the
`i128 → i64` cast is total. -/
theorem syrk_narrow_ok (v : I128)
    (h : -((2 : ℤ) ^ 63) ≤ v.val ∧ v.val ≤ (2 : ℤ) ^ 63 - 1) :
    ∃ q : I64, solve_port.q32.narrow v = ok (some q) ∧ q.val = v.val := by
  refine ⟨IScalar.cast .I64 v, ?_, dense_cast_i64_val v h⟩
  have hnotmax : ¬ (v > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, syrk_i64max_val] at hc
    omega
  have hnotmin : ¬ (v < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, syrk_i64min_val] at hc
    omega
  unfold solve_port.q32.narrow
  simp only [lift, bind_tc_ok, hnotmax, hnotmin, ↓reduceIte]

/-- `narrow` refuses above the `i64` window. -/
theorem syrk_narrow_none_hi (v : I128) (h : (2 : ℤ) ^ 63 - 1 < v.val) :
    solve_port.q32.narrow v = ok none := by
  have hgt : v > core.convert.num.FromI128I64.from core.num.I64.MAX := by
    rw [gt_iff_lt, IScalar.lt_equiv, syrk_i64max_val]
    exact h
  unfold solve_port.q32.narrow
  simp only [lift, bind_tc_ok, hgt, ↓reduceIte]

/-- `narrow` refuses below the `i64` window. -/
theorem syrk_narrow_none_lo (v : I128) (h : v.val < -((2 : ℤ) ^ 63)) :
    solve_port.q32.narrow v = ok none := by
  have hnotmax : ¬ (v > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, syrk_i64max_val] at hc
    omega
  have hlt : v < core.convert.num.FromI128I64.from core.num.I64.MIN := by
    rw [IScalar.lt_equiv, syrk_i64min_val]
    exact h
  unfold solve_port.q32.narrow
  simp only [lift, bind_tc_ok, hnotmax, hlt, ↓reduceIte]

/-! ## 6. The in-place Hadamard factor -/

/-- The `hadamard_into` loop invariant: `k` entries are folded, the slice keeps its length and its
`ℤ` image is the Hadamard model `hadStep`. -/
def hadInv (a0 h0 : Slice I64) (a : Slice I64) (k : Usize) : Prop :=
  k.val ≤ a0.val.length ∧ a.val.length = a0.val.length ∧
  qvals a = hadStep (qvals a0) (qvals h0) k.val

/-- `qvals` of an in-place write. -/
theorem qvals_set (a : Slice I64) (k : Usize) (v : I64) :
    qvals (a.set k v) = (qvals a).set k.val v.val := by
  simp only [qvals, Slice.set_val_eq, List.map_set]

/-- One `hadamard_into` iteration rounds and narrows the elementwise product and writes it in
place; the fold still reads the original entry at the write index. Consumes the `MapSome` boundary
law through `round32`. -/
theorem syrk_hadamard_body_step (hmap : MapSome) (a0 h0 : Slice I64) (n : Usize)
    (hn : n.val = a0.val.length) (hlen : h0.val.length = a0.val.length)
    (hfit : ∀ k : ℕ, k < a0.val.length → hadFit (qvals a0) (qvals h0) k)
    (a : Slice I64) (k : Usize) (hk : k.val < n.val) (hinv : hadInv a0 h0 a k) :
    ∃ s : Slice I64, ∃ k1 : Usize,
      solve_port.syrk.hadamard_into_loop.body h0 n a k = ok (ControlFlow.cont (s, k1)) ∧
      k1.val = k.val + 1 ∧
      qvals s = (hadStep (qvals a0) (qvals h0) k.val).set k.val
        (rnd32 ((qvals a0).getD k.val 0 * (qvals h0).getD k.val 0)) ∧
      s.val.length = a0.val.length := by
  obtain ⟨hkle, halen, hq⟩ := hinv
  have hkL : k.val < a0.val.length := by omega
  have hkal : k.val < a.val.length := by omega
  have hkh : k.val < h0.val.length := by omega
  have hia := dense_slice_index_getD a k hkal
  have hih := dense_slice_index_getD h0 k hkh
  have hiaval : (a.val.getD k.val default).val = (qvals a0).getD k.val 0 := by
    rw [← hadStep_getD_self (qvals a0) (qvals h0) k.val (by rw [qvals_length]; exact hkL), ← hq]
    exact (qvals_getD a hkal).symm
  have hihval : (h0.val.getD k.val default).val = (qvals h0).getD k.val 0 := by
    exact (qvals_getD h0 hkh).symm
  obtain ⟨pv, hpv, hpvv⟩ := dense_checked_mul_some
    (a := core.convert.num.FromI128I64.from (a.val.getD k.val default))
    (b := core.convert.num.FromI128I64.from (h0.val.getD k.val default))
    (by rw [dense_fromI64_val, dense_fromI64_val]; exact dense_i64_mul_fits _ _)
  have hpvL : pv.val = (qvals a0).getD k.val 0 * (qvals h0).getD k.val 0 := by
    rw [hpvv, dense_fromI64_val, dense_fromI64_val, hiaval, hihval]
  obtain ⟨hf1, hf2, hf3, hf4⟩ := hfit k.val hkL
  obtain ⟨w, hw, hwv⟩ := syrk_round32_spec hmap pv (by rw [hpvL]; exact ⟨hf1, hf2⟩)
  have hwvL : w.val = rnd32 ((qvals a0).getD k.val 0 * (qvals h0).getD k.val 0) := by
    rw [hwv, hpvL]
  obtain ⟨q, hnar, hqv⟩ := syrk_narrow_ok w (by rw [hwvL]; exact ⟨hf3, hf4⟩)
  have hqvv : q.val = rnd32 ((qvals a0).getD k.val 0 * (qvals h0).getD k.val 0) := by
    rw [hqv, hwvL]
  obtain ⟨s0, hsu, hsv⟩ := WP.spec_imp_exists (Slice.update_spec a k q hkal)
  subst hsv
  obtain ⟨k1, hk1, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize)
      (by rw [show (1#usize).val = 1 from rfl]
          have hp := Slice.property a0
          omega))
  have hk1_val : k1.val = k.val + 1 := by rw [hk1v]; rfl
  refine ⟨a.set k q, k1, ?_, hk1_val, ?_, ?_⟩
  · unfold solve_port.syrk.hadamard_into_loop.body
    rw [if_pos ((UScalar.lt_equiv k n).2 hk)]
    simp only [hia, hih, lift, hpv, hw, hnar, hsu, hk1, bind_tc_ok]
  · rw [qvals_set, hq, hqvv]
  · rw [Slice.set_val_eq, List.length_set]
    exact halen

/-- Full specification of the `hadamard_into` loop: under the elementwise fit predicate the fold
completes and the slice is the Hadamard model. Consumes `MapSome` through the body step. -/
theorem syrk_hadamard_loop_spec (hmap : MapSome) (a0 h0 : Slice I64) (n : Usize)
    (hn : n.val = a0.val.length) (hlen : h0.val.length = a0.val.length)
    (hfit : ∀ k : ℕ, k < a0.val.length → hadFit (qvals a0) (qvals h0) k) :
    WP.spec (solve_port.syrk.hadamard_into_loop a0 h0 n 0#usize)
      (fun r : (core.result.Result Unit solve_port.syrk.SyrkError) × Slice I64 =>
        r.1 = core.result.Result.Ok () ∧
        qvals r.2 = hadStep (qvals a0) (qvals h0) n.val) := by
  have hinv0 : hadInv a0 h0 a0 0#usize := by
    refine ⟨?_, rfl, ?_⟩
    · show (0#usize).val ≤ a0.val.length
      rw [show (0#usize).val = 0 from rfl]
      omega
    · show qvals a0 = hadStep (qvals a0) (qvals h0) (0#usize).val
      simp [hadStep, pushN]
  unfold solve_port.syrk.hadamard_into_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × Usize => n.val - s.2.val)
    (inv := fun s : Slice I64 × Usize => hadInv a0 h0 s.1 s.2)
    (post := fun r : (core.result.Result Unit solve_port.syrk.SyrkError) × Slice I64 =>
      r.1 = core.result.Result.Ok () ∧
      qvals r.2 = hadStep (qvals a0) (qvals h0) n.val)
  · rintro ⟨a, k⟩ hinv
    dsimp only at hinv ⊢
    by_cases hk : k.val < n.val
    · obtain ⟨s, k1, hstep, hk1v, hqvals, hlenS⟩ :=
        syrk_hadamard_body_step hmap a0 h0 n hn hlen hfit a k hk hinv
      have hkL : k.val < (qvals a0).length := by
        rw [qvals_length]; omega
      rw [hstep, WP.spec_ok]
      dsimp only
      refine ⟨⟨by omega, hlenS, ?_⟩, by omega⟩
      rw [hqvals, hk1v, hadStep_succ (qvals a0) (qvals h0) k.val hkL]
    · obtain ⟨hkle, halen, hq⟩ := hinv
      have hbody : solve_port.syrk.hadamard_into_loop.body h0 n a k =
            ok (ControlFlow.done (core.result.Result.Ok (), a)) := by
        unfold solve_port.syrk.hadamard_into_loop.body
        rw [if_neg (fun hc => hk ((UScalar.lt_equiv k n).1 hc))]
      rw [hbody, WP.spec_ok]
      dsimp only
      refine ⟨rfl, ?_⟩
      have hk_eq : k.val = n.val := by
        have hp := Slice.property a0
        omega
      rw [hq, hk_eq]
  · exact hinv0

/-- `hadamard_into` folds `a` elementwise with `h` in place under the elementwise fit predicate.
Conditional on `MapSome` (the `round32` boundary law); the shape refusal is unconditional below. -/
theorem word_hadamard_ok (hmap : MapSome) (a h : Slice I64)
    (hln : Slice.len a = Slice.len h)
    (hfit : ∀ k : ℕ, k < a.val.length → hadFit (qvals a) (qvals h) k) :
    ∃ v : Slice I64,
      solve_port.syrk.hadamard_into a h = ok (core.result.Result.Ok (), v) ∧
      qvals v = hadStep (qvals a) (qvals h) a.val.length := by
  have hleneq : h.val.length = a.val.length := by
    have h1 : (Slice.len a).val = (Slice.len h).val := by rw [hln]
    rw [Slice.len_val, Slice.len_val] at h1
    exact h1.symm
  have hspec := syrk_hadamard_loop_spec hmap a h (Slice.len a) rfl hleneq hfit
  obtain ⟨r, hexec, hpost⟩ := (WP.spec_equiv_exists _ _).mp hspec
  have hr : r = (core.result.Result.Ok (), r.2) := by rw [← hpost.1]
  have hbne : (Slice.len a != Slice.len h) = false := by
    cases hbl : (Slice.len a != Slice.len h) with
    | false => rfl
    | true => exact absurd (bne_iff_ne.mp hbl) (fun he => he hln)
  refine ⟨r.2, ?_, hpost.2⟩
  unfold solve_port.syrk.hadamard_into
  dsimp only
  rw [hbne]
  simp only [Bool.false_eq_true, if_false]
  rw [hexec, hr]

/-- `hadamard_into` refuses a length mismatch with `ShapeMismatch` and returns `a` unwritten. -/
theorem word_hadamard_refuses (a h : Slice I64) (hne : Slice.len a ≠ Slice.len h) :
    solve_port.syrk.hadamard_into a h =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.ShapeMismatch, a) := by
  have hbne : (Slice.len a != Slice.len h) = true := bne_iff_ne.mpr hne
  unfold solve_port.syrk.hadamard_into
  dsimp only
  rw [hbne]
  simp only [if_true]

/-! ## 7. The `wsyrk` accepted domain and its refusal surface -/

/-- The accepted domain of `wsyrk`: neither size is zero, both `checked_mul`s succeed, the `x` length
matches `o·n`, a present weight slice has length `o`, `o` is below `MAX_O = 2¹⁸`, and the input is
inside the strict `±2³⁶` boundary. Its negation is the refusal surface. -/
def wsyrkDomain (x : Slice I64) (w : Option (Slice I64)) (o n : Usize) : Prop :=
  n ≠ 0#usize ∧ o ≠ 0#usize ∧
  Usize.checked_mul n n ≠ none ∧
  (∀ on, Usize.checked_mul o n = some on → Slice.len x = on) ∧
  (∀ ws, w = some ws → Slice.len ws = o) ∧
  o.val < 2 ^ 18 ∧ withinBoundRef x w

/-- The `Slice.len x` shape check passes on the continuing path. -/
theorem syrk_bne_len_x {x : Slice I64} {on : Usize} (h : Slice.len x = on) :
    (Slice.len x != on) = false := by
  rw [Bool.eq_false_iff]
  exact fun hc => (bne_iff_ne.mp hc) h

/-- `wsyrk` refuses `n = 0` with `ZeroN`. -/
theorem syrk_wsyrk_refuse_zeroN (x : Slice I64) (w : Option (Slice I64)) (o n : Usize)
    (h : n = 0#usize) :
    solve_port.syrk.wsyrk x w o n =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.ZeroN) := by
  unfold solve_port.syrk.wsyrk
  dsimp only
  rw [if_pos h]

/-- `wsyrk` refuses `o = 0` with `ZeroO`. -/
theorem syrk_wsyrk_refuse_zeroO (x : Slice I64) (w : Option (Slice I64)) (o n : Usize)
    (hn : n ≠ 0#usize) (h : o = 0#usize) :
    solve_port.syrk.wsyrk x w o n =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.ZeroO) := by
  unfold solve_port.syrk.wsyrk
  dsimp only
  rw [if_neg hn, if_pos h]

/-- `wsyrk` refuses an overflowing `n·n` with `Overflow`. -/
theorem syrk_wsyrk_refuse_mulNN (x : Slice I64) (w : Option (Slice I64)) (o n : Usize)
    (hn : n ≠ 0#usize) (ho : o ≠ 0#usize) (h : Usize.checked_mul n n = none) :
    solve_port.syrk.wsyrk x w o n =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.Overflow) := by
  unfold solve_port.syrk.wsyrk
  dsimp only
  rw [if_neg hn, if_neg ho]
  simp only [lift, h, bind_tc_ok]

/-- `wsyrk` refuses an overflowing `o·n` with `ShapeMismatch`. -/
theorem syrk_wsyrk_refuse_mulON (x : Slice I64) (w : Option (Slice I64)) (o n : Usize)
    {nn : Usize} (hn : n ≠ 0#usize) (ho : o ≠ 0#usize)
    (hnn : Usize.checked_mul n n = some nn) (h : Usize.checked_mul o n = none) :
    solve_port.syrk.wsyrk x w o n =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.ShapeMismatch) := by
  unfold solve_port.syrk.wsyrk
  dsimp only
  rw [if_neg hn, if_neg ho]
  simp only [lift, hnn, h, bind_tc_ok]

/-- `wsyrk` refuses an `x` length that does not match `o·n` with `ShapeMismatch`. -/
theorem syrk_wsyrk_refuse_len_x (x : Slice I64) (w : Option (Slice I64)) (o n : Usize)
    {nn on : Usize} (hn : n ≠ 0#usize) (ho : o ≠ 0#usize)
    (hnn : Usize.checked_mul n n = some nn) (hon : Usize.checked_mul o n = some on)
    (h : Slice.len x ≠ on) :
    solve_port.syrk.wsyrk x w o n =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.ShapeMismatch) := by
  have hb : (Slice.len x != on) = true := bne_iff_ne.mpr h
  unfold solve_port.syrk.wsyrk
  dsimp only
  rw [if_neg hn, if_neg ho]
  simp only [lift, hnn, hon, bind_tc_ok, hb, if_true]

/-- `wsyrk` refuses a weight slice whose length does not match `o` with `ShapeMismatch`. -/
theorem syrk_wsyrk_refuse_len_w (x ws : Slice I64) (o n : Usize)
    {nn on : Usize} (hn : n ≠ 0#usize) (ho : o ≠ 0#usize)
    (hnn : Usize.checked_mul n n = some nn) (hon : Usize.checked_mul o n = some on)
    (hlen : Slice.len x = on) (hw : Slice.len ws ≠ o) :
    solve_port.syrk.wsyrk x (some ws) o n =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.ShapeMismatch) := by
  have hb : (Slice.len x != on) = false := syrk_bne_len_x hlen
  have hbw : (Slice.len ws != o) = true := bne_iff_ne.mpr hw
  unfold solve_port.syrk.wsyrk
  dsimp only
  rw [if_neg hn, if_neg ho]
  simp only [lift, hnn, hon, bind_tc_ok]
  rw [hb]
  simp only [Bool.false_eq_true, if_false]
  rw [hbw]
  simp only [if_true]

/-- `wsyrk` refuses `o ≥ MAX_O = 2¹⁸` with `Overflow` once the shape checks pass. -/
theorem syrk_wsyrk_refuse_maxO (x : Slice I64) (w : Option (Slice I64)) (o n : Usize)
    {nn on : Usize} (hn : n ≠ 0#usize) (ho : o ≠ 0#usize)
    (hnn : Usize.checked_mul n n = some nn) (hon : Usize.checked_mul o n = some on)
    (hlen : Slice.len x = on) (hwlen : ∀ ws, w = some ws → Slice.len ws = o)
    (h18 : 2 ^ 18 ≤ o.val) :
    solve_port.syrk.wsyrk x w o n =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.Overflow) := by
  obtain ⟨m, hm, hmv⟩ := syrk_max_o
  have hge : o ≥ m := by
    show m ≤ o
    rw [UScalar.le_equiv, hmv]
    exact h18
  have hb : (Slice.len x != on) = false := syrk_bne_len_x hlen
  unfold solve_port.syrk.wsyrk
  dsimp only
  rw [if_neg hn, if_neg ho]
  simp only [lift, hnn, hon, bind_tc_ok]
  rw [hb]
  simp only [Bool.false_eq_true, if_false]
  cases w with
  | none =>
    dsimp only
    rw [hm]
    simp only [bind_tc_ok]
    rw [if_pos hge]
  | some ws =>
    have hbw : (Slice.len ws != o) = false := syrk_bne_len_x (hwlen ws rfl)
    dsimp only
    rw [hbw]
    simp only [Bool.false_eq_true, if_false]
    rw [hm]
    simp only [bind_tc_ok]
    rw [if_pos hge]

/-- `wsyrk` refuses an input outside the strict `±2³⁶` boundary with `InputBound` once every shape
and size check passes. -/
theorem syrk_wsyrk_refuse_bound (x : Slice I64) (w : Option (Slice I64)) (o n : Usize)
    {nn on : Usize} (hn : n ≠ 0#usize) (ho : o ≠ 0#usize)
    (hnn : Usize.checked_mul n n = some nn) (hon : Usize.checked_mul o n = some on)
    (hlen : Slice.len x = on) (hwlen : ∀ ws, w = some ws → Slice.len ws = o)
    (h18 : o.val < 2 ^ 18) (hwb : ¬ withinBoundRef x w) :
    solve_port.syrk.wsyrk x w o n =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.InputBound) := by
  obtain ⟨m, hm, hmv⟩ := syrk_max_o
  have hlt : ¬ (o ≥ m) := by
    intro hc
    have hc' : m.val ≤ o.val := (UScalar.le_equiv m o).mp hc
    rw [hmv] at hc'
    omega
  have hb : (Slice.len x != on) = false := syrk_bne_len_x hlen
  unfold solve_port.syrk.wsyrk
  dsimp only
  rw [if_neg hn, if_neg ho]
  simp only [lift, hnn, hon, bind_tc_ok]
  rw [hb]
  simp only [Bool.false_eq_true, if_false]
  cases w with
  | none =>
    dsimp only
    rw [hm]
    simp only [bind_tc_ok]
    rw [if_neg hlt]
    have hwbf := syrk_within_bound_false x none hwb
    rw [hwbf]
    simp only [bind_tc_ok, Bool.false_eq_true, if_false]
  | some ws =>
    have hbw : (Slice.len ws != o) = false := syrk_bne_len_x (hwlen ws rfl)
    dsimp only
    rw [hbw]
    simp only [Bool.false_eq_true, if_false]
    rw [hm]
    simp only [bind_tc_ok]
    rw [if_neg hlt]
    have hwbf := syrk_within_bound_false x (some ws) hwb
    rw [hwbf]
    simp only [bind_tc_ok, Bool.false_eq_true, if_false]

/-- `wsyrk` on the refusal surface: outside `wsyrkDomain` the driver returns an error, never a
result. -/
theorem word_wsyrk_refuses (x : Slice I64) (w : Option (Slice I64)) (o n : Usize)
    (h : ¬ wsyrkDomain x w o n) :
    ∃ e : solve_port.syrk.SyrkError,
      solve_port.syrk.wsyrk x w o n = ok (core.result.Result.Err e) := by
  by_cases hn : n = 0#usize
  · exact ⟨_, syrk_wsyrk_refuse_zeroN x w o n hn⟩
  by_cases ho : o = 0#usize
  · exact ⟨_, syrk_wsyrk_refuse_zeroO x w o n hn ho⟩
  cases hnn : Usize.checked_mul n n with
  | none => exact ⟨_, syrk_wsyrk_refuse_mulNN x w o n hn ho hnn⟩
  | some nn =>
    cases hon : Usize.checked_mul o n with
    | none => exact ⟨_, syrk_wsyrk_refuse_mulON x w o n hn ho hnn hon⟩
    | some on =>
      by_cases hlen : Slice.len x = on
      · have hon' : ∀ on', Usize.checked_mul o n = some on' → Slice.len x = on' := by
          intro on' h'
          have : on' = on := Option.some.inj (h'.symm.trans hon)
          rw [this]
          exact hlen
        cases w with
        | none =>
          have hwn : ∀ ws, (none : Option (Slice I64)) = some ws → Slice.len ws = o :=
            fun ws hws => absurd hws (by simp)
          by_cases h18 : o.val < 2 ^ 18
          · by_cases hwb : withinBoundRef x none
            · exact absurd ⟨hn, ho, (by rw [hnn]; simp), hon', hwn, h18, hwb⟩ h
            · exact ⟨_, syrk_wsyrk_refuse_bound x none o n hn ho hnn hon hlen hwn
                h18 hwb⟩
          · exact ⟨_, syrk_wsyrk_refuse_maxO x none o n hn ho hnn hon hlen hwn
              (by omega)⟩
        | some ws =>
          by_cases hwl : Slice.len ws = o
          · have hws' : ∀ ws', (some ws : Option (Slice I64)) = some ws' → Slice.len ws' = o := by
              intro ws' hws
              rw [← Option.some.inj hws]
              exact hwl
            by_cases h18 : o.val < 2 ^ 18
            · by_cases hwb : withinBoundRef x (some ws)
              · exact absurd ⟨hn, ho, (by rw [hnn]; simp), hon', hws', h18, hwb⟩ h
              · exact ⟨_, syrk_wsyrk_refuse_bound x (some ws) o n hn ho hnn hon hlen
                  hws' h18 hwb⟩
            · exact ⟨_, syrk_wsyrk_refuse_maxO x (some ws) o n hn ho hnn hon hlen hws'
                (by omega)⟩
          · exact ⟨_, syrk_wsyrk_refuse_len_w x ws o n hn ho hnn hon hlen hwl⟩
      · exact ⟨_, syrk_wsyrk_refuse_len_x x w o n hn ho hnn hon hlen⟩

end apeiron_word
