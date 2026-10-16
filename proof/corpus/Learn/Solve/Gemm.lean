import ApeironWord
import Learn.Grad.DenseLinearSpec
import Learn.Solve.GemmSpec
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted `src/learn/solve/gemm.rs` products: refinement

`solve_port.word_solve_gemm_nn` / `_nt` are the extracted blocked Q32.32 products. This module
refines the shared refusal path: the bound scan (`solve_port.syrk.all_below`, and its user
`within_bound`) and the leading zero-dimension check of `solve_port.gemm.check`, to the independent
`ℤ` models in `Learn.Solve.GemmSpec` (`allBelowRef`, `withinRef`, `withinBound`, `gemmRefuse`).

The `all_below` loop is proved by the well-founded loop rule (`loop.spec_decr_nat`), not by kernel
evaluation of the extracted body.

Trust boundary: rustc/LLVM codegen, the kernel ABI, the `Vec`/`Slice` container layout and the
threaded driver `block_rows_par` (an extraction `axiom`) are not covered. What is NOT proved here:
the accepted-path induction through `block_rows` (`word_gemm_nn_ok` / `_nt_ok`) and the conditional
`threads > 1` theorem. The refusal chain is covered branch by branch (section 6) on the `MAX_K` probe
value; discharging `solve_port.gemm.MAX_K = ok (2⁵⁵)` needs a 64-bit `usize` and is open.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## 1. Constants and `i64::unsigned_abs` -/

/-- `BOUND_BITS` is `36`. -/
theorem gemm_BOUND_BITS_val : (solve_port.syrk.BOUND_BITS).val = 36 := by
  unfold solve_port.syrk.BOUND_BITS; decide

/-- `i64::unsigned_abs` returns the magnitude as a `u64`. -/
theorem gemm_unsigned_abs_val (z : I64) :
    ∃ u : U64, core.num.I64.unsigned_abs z = ok u ∧ u.val = z.val.natAbs := by
  refine ⟨⟨BitVec.ofNat 64 z.val.natAbs⟩, rfl, ?_⟩
  have hb : z.val.natAbs < 2 ^ 64 := by
    have h := z.hBounds
    norm_num [IScalarTy.numBits] at h
    have h1 : ((z.val.natAbs : ℕ) : ℤ) = |z.val| := Int.natCast_natAbs _
    have h2 : ((z.val.natAbs : ℕ) : ℤ) ≤ (2 : ℤ) ^ 63 := by rw [h1, abs_le]; omega
    have h3 : ((z.val.natAbs : ℕ) : ℤ) < (2 : ℤ) ^ 64 := by
      have : (2 : ℤ) ^ 63 < (2 : ℤ) ^ 64 := by norm_num
      linarith
    exact_mod_cast h3
  show (BitVec.ofNat 64 z.val.natAbs).toNat = z.val.natAbs
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hb]

/-! ## 2. The bound scan `all_below` -/

/-- The reference of `all_below`: every entry of `xs` below `limit` in magnitude. -/
def allBelowRef (limit : ℕ) : List I64 → Bool
  | [] => true
  | z :: zs => (z.val.natAbs < limit) && allBelowRef limit zs

/-- The scan is `true` exactly when every entry is below `limit` in magnitude. -/
theorem gemm_allBelowRef_eq_true (limit : ℕ) :
    ∀ l : List I64, allBelowRef limit l = true ↔ ∀ z ∈ l, z.val.natAbs < limit
  | [] => by simp [allBelowRef]
  | z :: zs => by
    simp only [allBelowRef, Bool.and_eq_true, decide_eq_true_iff,
      gemm_allBelowRef_eq_true limit zs]
    constructor
    · rintro ⟨h1, h2⟩ w hw
      rcases List.mem_cons.mp hw with rfl | hw
      · exact h1
      · exact h2 w hw
    · intro h
      exact ⟨h z List.mem_cons_self, fun w hw => h w (List.mem_cons_of_mem z hw)⟩

theorem gemm_allBelowRef_true {limit : ℕ} {zs : List I64} (h : allBelowRef limit zs) :
    ∀ z ∈ zs, z.val.natAbs < limit :=
  (gemm_allBelowRef_eq_true limit zs).mp h

theorem gemm_allBelowRef_append (limit : ℕ) (l1 l2 : List I64) :
    allBelowRef limit (l1 ++ l2) = (allBelowRef limit l1 && allBelowRef limit l2) := by
  induction l1 with
  | nil => simp [allBelowRef]
  | cons a tl ih =>
    simp only [List.cons_append, allBelowRef, ih, Bool.and_assoc]

/-- The scan's state: `ok1` is the conjunction over the entries already read. -/
def allBelowInv (xs : Slice I64) (limit : U64) (ok1 : Bool) (i : Usize) : Prop :=
  ok1 = allBelowRef limit.val (xs.val.take i.val)

/-- One iteration of `all_below_loop` on the succeeding path. -/
theorem gemm_all_below_loop_body_exec
    (xs : Slice I64) (limit : U64) (i : Usize) (u : U64) (i1 : Usize)
    (hi : i.val < xs.val.length)
    (hidx : Slice.index_usize xs i = ok (xs.val.getD i.val default))
    (hab : core.num.I64.unsigned_abs (xs.val.getD i.val default) = ok u)
    (had : i + 1#usize = ok i1) :
    solve_port.syrk.all_below_loop.body xs limit true i =
      ok (ControlFlow.cont (decide (u < limit), i1)) := by
  have hlen : i < Slice.len xs := by rw [UScalar.lt_equiv, Slice.len_val]; exact hi
  rw [solve_port.syrk.all_below_loop.body]
  simp only []
  rw [if_pos hlen, if_pos trivial]
  simp only [hidx, hab, had, bind_tc_ok]

/-- One iteration of `all_below_loop` advances the reference scan. -/
theorem gemm_all_below_loop_body_step
    (xs : Slice I64) (limit : U64) (ok1 : Bool) (i : Usize)
    (hok : ok1 = true) (hi : i.val < xs.val.length)
    (hinv : allBelowInv xs limit ok1 i) :
    WP.spec (solve_port.syrk.all_below_loop.body xs limit ok1 i)
      (fun cf => ∃ i', cf = ControlFlow.cont
          (decide ((xs.val.getD i.val default).val.natAbs < limit.val), i') ∧
        i'.val = i.val + 1 ∧ allBelowInv xs limit
          (decide ((xs.val.getD i.val default).val.natAbs < limit.val)) i') := by
  obtain ⟨u, hu, huv⟩ := gemm_unsigned_abs_val (xs.val.getD i.val default)
  have hidx := dense_slice_index_getD xs i hi
  have hxsmax := Slice.length_ineq xs
  obtain ⟨i1, had, hadv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) (by
      show i.val + (1#usize).val ≤ Usize.max
      rw [show (1#usize).val = 1 from rfl]; omega))
  have hadv' : i1.val = i.val + 1 := by rw [hadv]; rfl
  have hbool : decide (u < limit) =
      decide ((xs.val.getD i.val default).val.natAbs < limit.val) := by
    rw [decide_eq_decide, UScalar.lt_equiv, huv]
  have hexec := gemm_all_below_loop_body_exec xs limit i u i1 hi hidx hu had
  rw [hok, WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont
      (decide ((xs.val.getD i.val default).val.natAbs < limit.val), i1),
    ?_, i1, rfl, hadv', ?_⟩
  · rw [hbool] at hexec; exact hexec
  · rw [allBelowInv] at hinv ⊢
    rw [hok] at hinv
    have htake : xs.val.take i1.val =
        xs.val.take i.val ++ [xs.val.getD i.val default] := by
      rw [hadv', List.take_add_one]
      congr 1
      rw [List.getElem?_eq_getElem hi, Option.toList_some, List.getElem_eq_getD (default : I64)]
    rw [htake, gemm_allBelowRef_append, ← hinv]
    simp only [allBelowRef, Bool.true_and, Bool.and_true]

/-- Full specification of `all_below_loop`: it returns the reference scan of the whole slice. -/
theorem gemm_all_below_loop_spec (xs : Slice I64) (limit : U64) (ok1 : Bool) (i : Usize)
    (hinv : allBelowInv xs limit ok1 i) :
    WP.spec (solve_port.syrk.all_below_loop xs limit ok1 i)
      (fun r => r = allBelowRef limit.val xs.val) := by
  unfold solve_port.syrk.all_below_loop
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Usize => xs.val.length - s.2.val)
    (inv := fun s => allBelowInv xs limit s.1 s.2)
    (post := fun r => r = allBelowRef limit.val xs.val)
  · rintro ⟨ok', i'⟩ hinv'
    simp only [] at hinv' ⊢
    rw [allBelowInv] at hinv'
    by_cases hlt : i'.val < xs.val.length
    · by_cases hok : ok' = true
      · rw [hok] at hinv' ⊢
        obtain ⟨cf, hstep, i'', hcf, hi''v, hinv''⟩ := WP.spec_imp_exists
          (gemm_all_below_loop_body_step xs limit true i' rfl hlt
            (by rw [allBelowInv]; exact hinv'))
        rw [hcf] at hstep
        simp only [hstep]
        refine ⟨hinv'', by show xs.val.length - i''.val < xs.val.length - i'.val; omega⟩
      · have hof : ok' = false := by cases hb : ok' <;> simp_all
        have hfalse : allBelowRef limit.val xs.val = false := by
          have hsplit : allBelowRef limit.val xs.val =
              (allBelowRef limit.val (xs.val.take i'.val) &&
                allBelowRef limit.val (xs.val.drop i'.val)) := by
            conv_lhs => rw [← List.take_append_drop i'.val xs.val]
            rw [gemm_allBelowRef_append]
          rw [hsplit, ← hinv', hof, Bool.false_and]
        have hlen : i' < Slice.len xs := by rw [UScalar.lt_equiv, Slice.len_val]; exact hlt
        have hbody : solve_port.syrk.all_below_loop.body xs limit ok' i' =
            ok (ControlFlow.done false) := by
          rw [solve_port.syrk.all_below_loop.body]
          rw [if_pos hlen, if_neg hok]
        rw [hbody, WP.spec_equiv_exists]
        refine ⟨ControlFlow.done false, rfl, ?_⟩
        exact hfalse.symm
    · have hw : ¬ (i' < Slice.len xs) := by
        rw [UScalar.lt_equiv, Slice.len_val]; exact hlt
      have hbody : solve_port.syrk.all_below_loop.body xs limit ok' i' =
          ok (ControlFlow.done ok') := by
        rw [solve_port.syrk.all_below_loop.body]
        rw [if_neg hw]
      rw [hbody, WP.spec_equiv_exists]
      refine ⟨ControlFlow.done ok', rfl, ?_⟩
      show ok' = allBelowRef limit.val xs.val
      rw [hinv', List.take_of_length_le (by omega)]
  · exact hinv

/-- `all_below` returns the reference scan of the whole slice. -/
theorem gemm_all_below_spec (xs : Slice I64) (limit : U64) :
    solve_port.syrk.all_below xs limit = ok (allBelowRef limit.val xs.val) := by
  have hone : allBelowInv xs limit true 0#usize := by
    rw [allBelowInv, show (0#usize).val = 0 from rfl, List.take_zero]
    simp only [allBelowRef]
  have hspec := gemm_all_below_loop_spec xs limit true 0#usize hone
  obtain ⟨r, hloop, hr⟩ := (WP.spec_equiv_exists _ _).mp hspec
  unfold solve_port.syrk.all_below
  rw [hloop, hr]

/-! ## 3. The shared bound `within_bound` -/

/-- The reference of `within_bound`: every entry of `x`, and of `w` when present, below `2³⁶`. -/
def withinRef (x : Slice I64) (w : Option (Slice I64)) : Bool :=
  allBelowRef ((2 : ℕ) ^ 36) x.val &&
    (match w with | none => true | some ws => allBelowRef ((2 : ℕ) ^ 36) ws.val)

/-- The extracted shift `1 <<< BOUND_BITS` evaluates to `2³⁶`. -/
theorem gemm_limit_val :
    ∃ v : U64, (1#u64 <<< solve_port.syrk.BOUND_BITS) = ok v ∧ v.val = (2 : ℕ) ^ 36 := by
  unfold solve_port.syrk.BOUND_BITS
  have h : (1#u64 <<< (36#u32)) ⦃ fun z => z.val = (2 : ℕ) ^ 36 ⦄ := by
    step
    rw [z_post]
    norm_num [U64.size, U64.numBits, Nat.shiftLeft_eq]
  obtain ⟨v, hv, hvv⟩ := WP.spec_imp_exists h
  exact ⟨v, hv, hvv⟩

/-- `within_bound` returns the reference scan of both operands. -/
theorem gemm_within_bound_spec (x : Slice I64) (w : Option (Slice I64)) :
    solve_port.syrk.within_bound x w = ok (withinRef x w) := by
  obtain ⟨lm, hlm, hlmv⟩ := gemm_limit_val
  unfold solve_port.syrk.within_bound withinRef
  rw [hlm]
  simp only [bind_tc_ok]
  rw [gemm_all_below_spec]
  simp only [bind_tc_ok, hlmv]
  cases w with
  | none => cases allBelowRef ((2 : ℕ) ^ 36) x.val <;> simp_all
  | some ws =>
    simp only []
    rw [gemm_all_below_spec]
    simp only [hlmv]
    cases allBelowRef ((2 : ℕ) ^ 36) x.val <;> simp_all

/-! ## 4. The magnitude bound as a list predicate -/

/-- The extracted bound scan is `true` exactly when every entry is within `BND`. -/
theorem gemm_allBelowRef_iff (l : List I64) :
    allBelowRef ((2 : ℕ) ^ 36) l = true ↔ ∀ z ∈ l, |z.val| < BND := by
  rw [gemm_allBelowRef_eq_true, BND_eq]
  constructor
  · intro h z hz
    have hz' : ((z.val.natAbs : ℕ) : ℤ) < (((2 : ℕ) ^ 36 : ℕ) : ℤ) := by
      exact_mod_cast h z hz
    rwa [Int.natCast_natAbs] at hz'
  · intro h z hz
    have hz' : ((z.val.natAbs : ℕ) : ℤ) < (((2 : ℕ) ^ 36 : ℕ) : ℤ) := by
      rw [Int.natCast_natAbs]
      rw [show (((2 : ℕ) ^ 36 : ℕ) : ℤ) = (2 : ℤ) ^ 36 by norm_num]
      exact h z hz
    exact_mod_cast hz'

/-! ## 5. The leading zero-dimension refusal of `check` -/

/-- A zero dimension refuses with `ZeroDim`. -/
theorem gemm_check_zero_dim (a b : Slice I64) (m k n : Usize)
    (h : m = 0#usize ∨ k = 0#usize ∨ n = 0#usize) :
    solve_port.gemm.check a b m k n =
      ok (core.result.Result.Err solve_port.gemm.GemmError.ZeroDim) := by
  unfold solve_port.gemm.check
  rcases h with hm | hk | hn
  · rw [if_pos hm]
  · by_cases hm0 : m = 0#usize
    · rw [if_pos hm0]
    · rw [if_neg hm0, if_pos hk]
  · by_cases hm0 : m = 0#usize
    · rw [if_pos hm0]
    · by_cases hk0 : k = 0#usize
      · rw [if_neg hm0, if_pos hk0]
      · rw [if_neg hm0, if_neg hk0, if_pos hn]

/-! ## 6. The remaining refusal branches of `check`

After the three zero-dimension tests, `check` reads the contraction bound from
`solve_port.gemm.MAX_K`, then the three `Usize::checked_mul` products, then the two buffer lengths,
then the magnitude scan. Every branch past the zero-dimension gate is reachable and returns its
documented error, parametrised by the `MAX_K` probe value `v`: the second branch is `k ≥ v`, the
later branches `k < v`. `MAX_K` computes `1 <<< (127 − 2·36)`, which requires a 64-bit `usize`; its
discharge to `ok (2⁵⁵)` is the residual obligation recorded in the module header, so these theorems
carry `v` as a parameter rather than baking in a platform width.

`solve_port.gemm.MAX_K` is marked `irreducible`, so `#print axioms` on these theorems lists the
closed cone and no platform axiom. `usize`'s `!=` is the extraction's `bne`; `gemm_usize_bne_true_iff`
relates it to the reference's `=`.
-/

/-- The extraction's `usize` `!=` is the reference's `≠`. -/
theorem gemm_usize_bne_true_iff (x y : Usize) : (x != y) = true ↔ x ≠ y := by
  simp [bne_iff_ne]

/-- The `MAX_K` length bound fires: `k ≥ v` refuses with `Overflow`. -/
theorem gemm_check_overflow_k (a b : Slice I64) (m k n v : Usize)
    (hm : m ≠ 0#usize) (hk : k ≠ 0#usize) (hn : n ≠ 0#usize)
    (hMK : solve_port.gemm.MAX_K = ok v) (hkv : v ≤ k) :
    solve_port.gemm.check a b m k n =
      ok (core.result.Result.Err solve_port.gemm.GemmError.Overflow) := by
  unfold solve_port.gemm.check
  rw [if_neg hm, if_neg hk, if_neg hn, hMK]
  simp only [bind_tc_ok]
  rw [if_pos hkv]

/-- `m·k` overflowing `usize` refuses with `Overflow`. -/
theorem gemm_check_overflow_mk (a b : Slice I64) (m k n v : Usize)
    (hm : m ≠ 0#usize) (hk : k ≠ 0#usize) (hn : n ≠ 0#usize)
    (hMK : solve_port.gemm.MAX_K = ok v) (hkv : k < v)
    (hmk : Usize.checked_mul m k = none) :
    solve_port.gemm.check a b m k n =
      ok (core.result.Result.Err solve_port.gemm.GemmError.Overflow) := by
  unfold solve_port.gemm.check
  rw [if_neg hm, if_neg hk, if_neg hn, hMK]
  simp only [bind_tc_ok]
  rw [if_neg (not_le.mpr hkv), hmk]
  simp only [lift, bind_tc_ok]

/-- `k·n` overflowing `usize` refuses with `Overflow`. -/
theorem gemm_check_overflow_kn (a b : Slice I64) (m k n v ak : Usize)
    (hm : m ≠ 0#usize) (hk : k ≠ 0#usize) (hn : n ≠ 0#usize)
    (hMK : solve_port.gemm.MAX_K = ok v) (hkv : k < v)
    (hmk : Usize.checked_mul m k = some ak)
    (hkn : Usize.checked_mul k n = none) :
    solve_port.gemm.check a b m k n =
      ok (core.result.Result.Err solve_port.gemm.GemmError.Overflow) := by
  unfold solve_port.gemm.check
  rw [if_neg hm, if_neg hk, if_neg hn, hMK]
  simp only [bind_tc_ok]
  rw [if_neg (not_le.mpr hkv), hmk, hkn]
  simp only [lift, bind_tc_ok]

/-- `n·m` overflowing `usize` refuses with `Overflow`. -/
theorem gemm_check_overflow_nm (a b : Slice I64) (m k n v ak kn : Usize)
    (hm : m ≠ 0#usize) (hk : k ≠ 0#usize) (hn : n ≠ 0#usize)
    (hMK : solve_port.gemm.MAX_K = ok v) (hkv : k < v)
    (hmk : Usize.checked_mul m k = some ak)
    (hkn : Usize.checked_mul k n = some kn)
    (hnm : Usize.checked_mul n m = none) :
    solve_port.gemm.check a b m k n =
      ok (core.result.Result.Err solve_port.gemm.GemmError.Overflow) := by
  unfold solve_port.gemm.check
  rw [if_neg hm, if_neg hk, if_neg hn, hMK]
  simp only [bind_tc_ok]
  rw [if_neg (not_le.mpr hkv), hmk, hkn, hnm]
  simp only [lift, bind_tc_ok]
  rfl

/-- A first operand whose length differs from `m·k` refuses with `ShapeMismatch`. -/
theorem gemm_check_shape_a (a b : Slice I64) (m k n v ak kn nm : Usize)
    (hm : m ≠ 0#usize) (hk : k ≠ 0#usize) (hn : n ≠ 0#usize)
    (hMK : solve_port.gemm.MAX_K = ok v) (hkv : k < v)
    (hmk : Usize.checked_mul m k = some ak)
    (hkn : Usize.checked_mul k n = some kn)
    (hnm : Usize.checked_mul n m = some nm)
    (hlen : (Slice.len a != ak) = true) :
    solve_port.gemm.check a b m k n =
      ok (core.result.Result.Err solve_port.gemm.GemmError.ShapeMismatch) := by
  unfold solve_port.gemm.check
  rw [if_neg hm, if_neg hk, if_neg hn, hMK]
  simp only [bind_tc_ok]
  rw [if_neg (not_le.mpr hkv), hmk, hkn, hnm]
  simp only [lift, bind_tc_ok, hlen]
  rfl

/-- A second operand whose length differs from `k·n` refuses with `ShapeMismatch`. -/
theorem gemm_check_shape_b (a b : Slice I64) (m k n v ak kn nm : Usize)
    (hm : m ≠ 0#usize) (hk : k ≠ 0#usize) (hn : n ≠ 0#usize)
    (hMK : solve_port.gemm.MAX_K = ok v) (hkv : k < v)
    (hmk : Usize.checked_mul m k = some ak)
    (hkn : Usize.checked_mul k n = some kn)
    (hnm : Usize.checked_mul n m = some nm)
    (hlena : (Slice.len a != ak) = false)
    (hlenb : (Slice.len b != kn) = true) :
    solve_port.gemm.check a b m k n =
      ok (core.result.Result.Err solve_port.gemm.GemmError.ShapeMismatch) := by
  unfold solve_port.gemm.check
  rw [if_neg hm, if_neg hk, if_neg hn, hMK]
  simp only [bind_tc_ok]
  rw [if_neg (not_le.mpr hkv), hmk, hkn, hnm]
  simp only [lift, bind_tc_ok, hlena, hlenb]
  rfl

/-- A matching shape with a magnitude scan that reads `false` refuses with `InputBound`. -/
theorem gemm_check_input_bound (a b : Slice I64) (m k n v ak kn nm : Usize)
    (hm : m ≠ 0#usize) (hk : k ≠ 0#usize) (hn : n ≠ 0#usize)
    (hMK : solve_port.gemm.MAX_K = ok v) (hkv : k < v)
    (hmk : Usize.checked_mul m k = some ak)
    (hkn : Usize.checked_mul k n = some kn)
    (hnm : Usize.checked_mul n m = some nm)
    (hlena : (Slice.len a != ak) = false)
    (hlenb : (Slice.len b != kn) = false)
    (hb : solve_port.syrk.within_bound a (some b) = ok false) :
    solve_port.gemm.check a b m k n =
      ok (core.result.Result.Err solve_port.gemm.GemmError.InputBound) := by
  unfold solve_port.gemm.check
  rw [if_neg hm, if_neg hk, if_neg hn, hMK]
  simp only [bind_tc_ok]
  rw [if_neg (not_le.mpr hkv), hmk, hkn, hnm]
  simp only [lift, bind_tc_ok, hlena, hlenb, hb]
  rfl

/-- A matching shape and a passing magnitude scan accepts, returning `Ok ()`. -/
theorem gemm_check_ok (a b : Slice I64) (m k n v ak kn nm : Usize)
    (hm : m ≠ 0#usize) (hk : k ≠ 0#usize) (hn : n ≠ 0#usize)
    (hMK : solve_port.gemm.MAX_K = ok v) (hkv : k < v)
    (hmk : Usize.checked_mul m k = some ak)
    (hkn : Usize.checked_mul k n = some kn)
    (hnm : Usize.checked_mul n m = some nm)
    (hlena : (Slice.len a != ak) = false)
    (hlenb : (Slice.len b != kn) = false)
    (hb : solve_port.syrk.within_bound a (some b) = ok true) :
    solve_port.gemm.check a b m k n = ok (core.result.Result.Ok ()) := by
  unfold solve_port.gemm.check
  rw [if_neg hm, if_neg hk, if_neg hn, hMK]
  simp only [bind_tc_ok]
  rw [if_neg (not_le.mpr hkv), hmk, hkn, hnm]
  simp only [lift, bind_tc_ok, hlena, hlenb, hb]
  rfl

end apeiron_word
