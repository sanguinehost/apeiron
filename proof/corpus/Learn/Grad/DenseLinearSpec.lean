import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted `src/learn/grad/dense.rs` linear passes: independent spec

The independent `ℤ`/`List ℤ` denotation the extracted `src/learn/grad/dense.rs` linear passes are
refined against: the Q32.32 rounding `(v + 2³¹) >>> 32` (`rnd32`), the exact `i128` accumulation
`dsum`, the row-major output shapes (`pushN`, `fwdCols`, `fwdRows`, `rndAll`), and the checked
`i64 → i128` / `usize` arithmetic lemmas the refinement proofs consume. No extracted definition is
referenced by the models here.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the vector container layout are not covered;
only the extracted integer body.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result
/-! ## 1. Index and checked-arithmetic helpers -/

/-- `Slice.index_usize` reads the backing list at the in-range index. -/
theorem dense_slice_index_getD {α : Type} [Inhabited α] (x : Slice α) (k : Usize)
    (h : k.val < x.val.length) :
    Slice.index_usize x k = ok (x.val.getD k.val default) := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]

/-- A `checked_mul` that fits returns its mathematical product. -/
theorem dense_checked_mul_some {a b : I128}
    (h : I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max) :
    ∃ z, I128.checked_mul a b = some z ∧ z.val = a.val * b.val := by
  have hspec : (match I128.checked_mul a b with
      | some z => I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max
          ∧ z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => ¬ (I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max)) :=
    I128.checked_mul_bv_spec a b
  cases hm : I128.checked_mul a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

/-- A `checked_add` that fits returns its mathematical sum. -/
theorem dense_checked_add_some {a b : I128}
    (h : I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max) :
    ∃ z, I128.checked_add a b = some z ∧ z.val = a.val + b.val := by
  have hspec : (match I128.checked_add a b with
      | some z => I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max
          ∧ z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max)) :=
    I128.checked_add_bv_spec a b
  cases hm : I128.checked_add a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

/-- The `i64 → i128` widening preserves the value. -/
theorem dense_fromI64_val (z : I64) : (core.convert.num.FromI128I64.from z).val = z.val :=
  core.convert.num.FromI128I64.from_val_eq z

/-- Two `i64` values multiply into `i128` (`|a·b| ≤ 2¹²⁶`). -/
theorem dense_i64_mul_fits (a b : I64) :
    I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max := by
  have ha := a.hBounds
  have hb := b.hBounds
  norm_num [IScalarTy.numBits] at ha hb
  rw [I128.min_eq, I128.max_eq]
  have ha1 : -9223372036854775808 ≤ a.val := ha.1
  have ha2 : a.val ≤ 9223372036854775807 := by omega
  have hb1 : -9223372036854775808 ≤ b.val := hb.1
  have hb2 : b.val ≤ 9223372036854775807 := by omega
  constructor <;> nlinarith

/-- The extracted `1 <<< 31` evaluates to `2^31`. -/
theorem dense_shl31 : (1#i128 <<< 31#i32) ⦃ z => z.val = (2 : ℤ) ^ 31 ⦄ := by
  step
  rw [z_post]
  norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]

/-- The extracted shift amount `32#i32` has `toNat = 32`. -/
theorem dense_shift32_toNat : (32#i32).toNat = 32 := by decide

/-- The Q32.32 rounding `(v + 2³¹) >>> 32` of a Q64.64 value. -/
def rnd32 (v : ℤ) : ℤ := (v + 2 ^ 31) >>> 32

theorem rnd32_eq_div (v : ℤ) : rnd32 v = (v + 2 ^ 31) / 2 ^ 32 := by
  rw [rnd32, Int.shiftRight_eq_div_pow]
  norm_num

/-- The extracted `round32` is total on the fitting domain and returns the rounding. -/
theorem dense_round32_spec (v : I128)
    (h : I128.min ≤ v.val + 2 ^ 31 ∧ v.val + 2 ^ 31 ≤ I128.max) :
    ∃ w : I128, learn_port.dense.round32 v = ok (some w) ∧ w.val = rnd32 v.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists dense_shl31
  obtain ⟨s, hs, hsv⟩ := dense_checked_add_some (a := v) (b := c) (by rw [hcv]; exact h)
  obtain ⟨w, hw, hwv⟩ := WP.spec_imp_exists
    (IScalar.ShiftRight_IScalar_spec (x := s) (y := 32#i32) (by decide) (by decide))
  refine ⟨w, ?_, ?_⟩
  · unfold learn_port.dense.round32
    rw [hc, bind_tc_ok]
    simp only [lift, hs, bind_tc_ok]
    rw [hw, bind_tc_ok]
  · rw [hwv.1, dense_shift32_toNat, hsv, hcv, rnd32]

/-! ## 2. `narrow` and `expect_len` -/

/-- The extracted `i64::MAX` cast to `i128` is `2^63 - 1`. -/
theorem dense_i64max_val :
    (IScalar.cast .I128 core.num.I64.MAX).val = (2 : ℤ) ^ 63 - 1 := by
  rw [IScalar.cast_val_eq]
  have h : core.num.I64.MAX.val = (2 : ℤ) ^ 63 - 1 := by decide
  rw [h]
  norm_num [IScalarTy.numBits]
  exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2 : ℤ) ^ 63 - 1)
    (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])

/-- The extracted `i64::MIN` cast to `i128` is `-2^63`. -/
theorem dense_i64min_val :
    (IScalar.cast .I128 core.num.I64.MIN).val = -((2 : ℤ) ^ 63) := by
  rw [IScalar.cast_val_eq]
  have h : core.num.I64.MIN.val = -((2 : ℤ) ^ 63) := by decide
  rw [h]
  norm_num [IScalarTy.numBits]
  exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-((2 : ℤ) ^ 63))
    (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])

/-- The extracted `i128 → i64` narrowing cast is the value when it fits. -/
theorem dense_cast_i64_val (v : I128)
    (h : -((2 : ℤ) ^ 63) ≤ v.val ∧ v.val ≤ (2 : ℤ) ^ 63 - 1) :
    (IScalar.cast .I64 v).val = v.val := by
  rw [IScalar.cast_val_eq]
  norm_num [IScalarTy.numBits]
  exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 v.val
    (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)

/-- The extracted `narrow` returns the value exactly on the `i64` window. -/
theorem dense_narrow_spec (v : I128)
    (h : -((2 : ℤ) ^ 63) ≤ v.val ∧ v.val ≤ (2 : ℤ) ^ 63 - 1) :
    ∃ q : I64, learn_port.dense.narrow v = ok (some q) ∧ q.val = v.val := by
  refine ⟨IScalar.cast .I64 v, ?_, dense_cast_i64_val v h⟩
  have hnotmax : ¬ (v > IScalar.cast .I128 core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, dense_i64max_val] at hc
    omega
  have hnotmin : ¬ (v < IScalar.cast .I128 core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, dense_i64min_val] at hc
    omega
  unfold learn_port.dense.narrow
  simp only [lift, bind_tc_ok, hnotmax, hnotmin, ↓reduceIte]

/-- A `usize` `checked_mul` that fits returns its mathematical product. -/
theorem dense_usize_checked_mul_some (a b : Usize) (h : a.val * b.val ≤ Usize.max) :
    ∃ t, Usize.checked_mul a b = some t ∧ t.val = a.val * b.val := by
  have hspec := Usize.checked_mul_bv_spec a b
  cases hm : Usize.checked_mul a b with
  | none => rw [hm] at hspec; omega
  | some t => rw [hm] at hspec; exact ⟨t, rfl, hspec.2.1⟩

/-- `expect_len` accepts a matching `rows × cols` buffer. -/
theorem dense_expect_len_ok (rows cols len : Usize) (hr : rows ≠ 0#usize) (hc : cols ≠ 0#usize)
    (hprod : rows.val * cols.val ≤ Usize.max) (heq : len.val = rows.val * cols.val) :
    learn_port.dense.expect_len rows cols len = ok (core.result.Result.Ok ()) := by
  obtain ⟨t, ht, htv⟩ := dense_usize_checked_mul_some rows cols hprod
  unfold learn_port.dense.expect_len
  rw [if_neg hr, if_neg hc, ht]
  simp only [lift, bind_tc_ok]
  rw [if_pos (UScalar.eq_of_val_eq (by omega))]

/-- `expect_len` refuses a zero dimension with `Empty`. -/
theorem dense_expect_len_empty (rows cols len : Usize) (hr : rows = 0#usize) :
    learn_port.dense.expect_len rows cols len =
      ok (core.result.Result.Err learn_port.dense.DenseError.Empty) := by
  unfold learn_port.dense.expect_len
  simp only [hr, if_pos]

/-- `expect_len` refuses a mismatched length with `Shape`. -/
theorem dense_expect_len_shape (rows cols len : Usize) (hr : rows ≠ 0#usize) (hc : cols ≠ 0#usize)
    (hprod : rows.val * cols.val ≤ Usize.max) (hne : len.val ≠ rows.val * cols.val) :
    learn_port.dense.expect_len rows cols len =
      ok (core.result.Result.Err learn_port.dense.DenseError.Shape) := by
  obtain ⟨t, ht, htv⟩ := dense_usize_checked_mul_some rows cols hprod
  unfold learn_port.dense.expect_len
  rw [if_neg hr, if_neg hc, ht]
  simp only [lift, bind_tc_ok]
  rw [if_neg (fun heq' => hne (by rw [← heq', htv]))]

/-- `expect_len` refuses a dimension product above `Usize.max` with `Overflow`. -/
theorem dense_expect_len_overflow (rows cols len : Usize) (hr : rows ≠ 0#usize)
    (hc : cols ≠ 0#usize) (hmul : rows.checked_mul cols = none) :
    learn_port.dense.expect_len rows cols len =
      ok (core.result.Result.Err learn_port.dense.DenseError.Overflow) := by
  unfold learn_port.dense.expect_len
  rw [if_neg hr, if_neg hc, hmul]
  simp only [lift, bind_tc_ok]

/-! ## 3. The pure `ℤ` models -/

/-- The Q32.32 value list of an `i64` slice. -/
def qvals (x : Slice I64) : List ℤ := x.val.map (fun z => z.val)

/-- The Q64.64 value list of an `i128` slice. -/
def ivals (a : Slice I128) : List ℤ := a.val.map (fun z => z.val)

/-- The value list of an `i64` vector. -/
def vvals (v : alloc.vec.Vec I64) : List ℤ := v.val.map (fun z => z.val)

/-- `n` terms of a dense kernel's inner accumulation, read at loop-derived positions. -/
def dsum (L R : List ℤ) (gL gR : ℕ → ℕ → ℕ → ℕ) (a b : ℕ) : ℕ → ℤ
  | 0 => 0
  | k + 1 => dsum L R gL gR a b k + L.getD (gL a b k) 0 * R.getD (gR a b k) 0

theorem dsum_succ (L R : List ℤ) (gL gR : ℕ → ℕ → ℕ → ℕ) (a b k : ℕ) :
    dsum L R gL gR a b (k + 1) =
      dsum L R gL gR a b k + L.getD (gL a b k) 0 * R.getD (gR a b k) 0 := rfl

/-- The `linear_fwd` inner accumulation `Σ_{t<k} x[r·i + t] · w[c·i + t]`. -/
def fwdK (L R : List ℤ) (i r c : ℕ) : ℕ → ℤ :=
  dsum L R (fun a _ t => a * i + t) (fun _ b t => b * i + t) r c

/-- Append `n` values to a list. -/
def pushN (l : List ℤ) (f : ℕ → ℤ) : ℕ → List ℤ
  | 0 => l
  | n + 1 => pushN l f n ++ [f n]

/-- The `linear_fwd` column list of row `r`: `[fwdK r 0, …, fwdK r (o-1)]`. -/
def fwdCols (L R : List ℤ) (i r o : ℕ) : List ℤ := pushN [] (fun c => fwdK L R i r c i) o

/-- The `linear_fwd` output after `n` rows. -/
def fwdRows (L R : List ℤ) (i o : ℕ) : ℕ → List ℤ
  | 0 => []
  | r + 1 => fwdRows L R i o r ++ fwdCols L R i r o

/-- Row-major `rows × cols` output whose entry `(p, c)` is `elem p c`. -/
def rowsOf (cols : ℕ) (elem : ℕ → ℕ → ℤ) : ℕ → List ℤ
  | 0 => []
  | r + 1 => rowsOf cols elem r ++ pushN [] (elem r) cols

/-- The `linear_bwd_input` inner accumulation `Σ_{t<c} dy[r·o + t] · w[t·i + k]`. -/
def bwdIK (L R : List ℤ) (i o r k : ℕ) : ℕ → ℤ :=
  dsum L R (fun a _ t => a * o + t) (fun _ b t => t * i + b) r k

theorem bwdIK_succ (L R : List ℤ) (i o r k c : ℕ) :
    bwdIK L R i o r k (c + 1) =
      bwdIK L R i o r k c + L.getD (r * o + c) 0 * R.getD (c * i + k) 0 := rfl

/-- The `linear_bwd_weight` inner accumulation `Σ_{t<r} dy[t·o + c] · x[t·i + k]`. -/
def bwdWK (L R : List ℤ) (i o c k : ℕ) : ℕ → ℤ :=
  dsum L R (fun a _ t => t * o + a) (fun _ b t => t * i + b) c k

theorem bwdWK_succ (L R : List ℤ) (i o c k r : ℕ) :
    bwdWK L R i o c k (r + 1) =
      bwdWK L R i o c k r + L.getD (r * o + c) 0 * R.getD (r * i + k) 0 := rfl


theorem qvals_length (x : Slice I64) : (qvals x).length = x.val.length := by simp [qvals]

theorem ivals_length (a : Slice I128) : (ivals a).length = a.val.length := by simp [ivals]

theorem pushN_length (l : List ℤ) (f : ℕ → ℤ) (n : ℕ) :
    (pushN l f n).length = l.length + n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [pushN, List.length_append, ih, List.length_singleton]
    omega

/-- `pushN` leaves the first `l.length` entries alone. -/
theorem pushN_getD_left (l : List ℤ) (f : ℕ → ℤ) :
    ∀ n j, j < l.length → (pushN l f n).getD j 0 = l.getD j 0 := by
  intro n
  induction n with
  | zero => intro j _; rfl
  | succ k ih =>
    intro j hj
    rw [pushN, List.getD_append _ _ _ _ (by rw [pushN_length]; omega)]
    exact ih j hj

/-- The `k`-th appended entry is `f k`. -/
theorem pushN_getD_push (l : List ℤ) (f : ℕ → ℤ) :
    ∀ n k, k < n → (pushN l f n).getD (l.length + k) 0 = f k := by
  intro n
  induction n with
  | zero => intro k hk; omega
  | succ m ih =>
    intro k hk
    rcases Nat.lt_succ_iff_lt_or_eq.mp hk with hlt | heq
    · rw [pushN, List.getD_append _ _ _ _ (by rw [pushN_length]; omega)]
      exact ih k hlt
    · subst heq
      rw [pushN]
      rw [List.getD_append_right _ _ _ _ (by rw [pushN_length])]
      rw [pushN_length, Nat.sub_self]
      rfl

theorem ivals_getD (a : Slice I128) {k : ℕ} (h : k < a.val.length) :
    (ivals a).getD k 0 = (a.val.getD k 0#i128).val := by
  rw [ivals, List.getD_eq_getElem _ _ (by rwa [List.length_map]), List.getElem_map,
    List.getD_eq_getElem _ _ h]

theorem qvals_getD (x : Slice I64) {k : ℕ} (h : k < x.val.length) :
    (qvals x).getD k 0 = (x.val.getD k 0#i64).val := by
  rw [qvals, List.getD_eq_getElem _ _ (by rwa [List.length_map]), List.getElem_map,
    List.getD_eq_getElem _ _ h]

theorem fwdCols_length (L R : List ℤ) (i r o : ℕ) : (fwdCols L R i r o).length = o := by
  rw [fwdCols, pushN_length]
  simp

theorem fwdRows_length (L R : List ℤ) (i o : ℕ) (n : ℕ) :
    (fwdRows L R i o n).length = n * o := by
  induction n with
  | zero => simp [fwdRows]
  | succ k ih =>
    rw [fwdRows, List.length_append, ih, fwdCols_length, Nat.succ_mul]

/-- Entry `(p, c)` of the `linear_fwd` output model. -/
theorem fwdRows_getD (L R : List ℤ) (i o : ℕ) :
    ∀ n p c, p < n → c < o →
      (fwdRows L R i o n).getD (p * o + c) 0 = fwdK L R i p c i := by
  intro n
  induction n with
  | zero => intro p c hp _; omega
  | succ m ih =>
    intro p c hp hc
    rw [fwdRows]
    rcases Nat.lt_succ_iff_lt_or_eq.mp hp with hlt | heq
    · rw [List.getD_append _ _ _ _ (by
        rw [fwdRows_length]
        calc p * o + c < p * o + o := by omega
          _ = (p + 1) * o := (Nat.succ_mul p o).symm
          _ ≤ m * o := Nat.mul_le_mul_right o (Nat.succ_le_of_lt hlt))]
      exact ih p c hlt hc
    · rw [heq]
      rw [List.getD_append_right _ _ _ _ (by rw [fwdRows_length]; omega)]
      rw [fwdRows_length, fwdCols]
      rw [show m * o + c - m * o = c by omega]
      simpa using pushN_getD_push [] (fun c => fwdK L R i m c i) o c hc

theorem rowsOf_succ (cols : ℕ) (elem : ℕ → ℕ → ℤ) (r : ℕ) :
    rowsOf cols elem (r + 1) = rowsOf cols elem r ++ pushN [] (elem r) cols := rfl

theorem rowsOf_length (cols : ℕ) (elem : ℕ → ℕ → ℤ) (n : ℕ) :
    (rowsOf cols elem n).length = n * cols := by
  induction n with
  | zero => simp [rowsOf]
  | succ k ih => rw [rowsOf, List.length_append, ih, pushN_length]; simp [Nat.succ_mul]

/-- Entry `(p, c)` of a `rowsOf` output. -/
theorem rowsOf_getD (cols : ℕ) (elem : ℕ → ℕ → ℤ) :
    ∀ n p c, p < n → c < cols → (rowsOf cols elem n).getD (p * cols + c) 0 = elem p c := by
  intro n
  induction n with
  | zero => intro p c hp _; omega
  | succ m ih =>
    intro p c hp hc
    rw [rowsOf]
    rcases Nat.lt_succ_iff_lt_or_eq.mp hp with hlt | heq
    · rw [List.getD_append _ _ _ _ (by
        rw [rowsOf_length]
        calc p * cols + c < p * cols + cols := by omega
          _ = (p + 1) * cols := (Nat.succ_mul p cols).symm
          _ ≤ m * cols := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hlt))]
      exact ih p c hlt hc
    · rw [heq]
      rw [List.getD_append_right _ _ _ _ (by rw [rowsOf_length]; omega)]
      rw [rowsOf_length]
      rw [show m * cols + c - m * cols = c by omega]
      simpa using pushN_getD_push [] (elem m) cols c hc

/-! ## 4. The `round_all` loop -/

/-- Every `usize` is at most `Usize.max`. -/
theorem dense_usize_le_max (x : Usize) : x.val ≤ Usize.max := by
  have h := UScalar.hrBounds x
  rwa [UScalar.rMax_eq_max, UScalar.max_USize_eq] at h

/-- The value list of an `i128` vector. -/
def ovals (v : alloc.vec.Vec I128) : List ℤ := v.val.map (fun z => z.val)

theorem ovals_length (v : alloc.vec.Vec I128) : (ovals v).length = v.val.length := by
  simp [ovals]

/-- The `round_all` output model: `[rnd32 L[0], …, rnd32 L[n-1]]`. -/
def rndAll (L : List ℤ) (n : ℕ) : List ℤ := pushN [] (fun c => rnd32 (L.getD c 0)) n

theorem rndAll_succ (L : List ℤ) (n : ℕ) :
    rndAll L (n + 1) = rndAll L n ++ [rnd32 (L.getD n 0)] := by
  unfold rndAll
  rw [pushN]

theorem rndAll_length (L : List ℤ) (n : ℕ) : (rndAll L n).length = n := by
  rw [rndAll, pushN_length]; simp

theorem rndAll_getD (L : List ℤ) {n j : ℕ} (h : j < n) :
    (rndAll L n).getD j 0 = rnd32 (L.getD j 0) := by
  rw [rndAll]
  simpa using pushN_getD_push [] (fun c => rnd32 (L.getD c 0)) n j h

/-- The `round_all` loop invariant. -/
def roundLoopInv (acc : Slice I128) (n : Usize) (out : alloc.vec.Vec I64) (k : Usize)
    (ok1 : Bool) : Prop :=
  ok1 = true ∧ k.val ≤ n.val ∧ vvals out = rndAll (ivals acc) k.val

/-- The extracted `round_all` loop body on the succeeding path. -/
theorem round_all_loop_body_exec
    (acc : Slice I128) (n : Usize) (out : alloc.vec.Vec I64) (k : Usize)
    (v : I128) (q : I64) (out1 : alloc.vec.Vec I64) (k1 : Usize)
    (hk : k < n)
    (hidx : Slice.index_usize acc k = ok (acc.val.getD k.val 0#i128))
    (hr : learn_port.dense.round32 (acc.val.getD k.val 0#i128) = ok (some v))
    (hn : learn_port.dense.narrow v = ok (some q))
    (hp : alloc.vec.Vec.push out q = ok out1)
    (hk1 : k + 1#usize = ok k1) :
    learn_port.dense.round_all_loop.body acc n out k true =
      ok (ControlFlow.cont (out1, k1, true)) := by
  rw [learn_port.dense.round_all_loop.body]
  rw [if_pos hk, if_pos rfl]
  simp only [hidx, hr, hn, hp, hk1, bind_tc_ok]
  simp

/-- One iteration of `round_all_loop` advances the model and preserves the invariant. -/
theorem round_all_loop_body_step
    (acc : Slice I128) (n : Usize) (out : alloc.vec.Vec I64) (k : Usize) (ok1 : Bool)
    (hlen : acc.val.length = n.val)
    (hround : ∀ t < n.val,
      I128.min ≤ (ivals acc).getD t 0 + 2 ^ 31 ∧ (ivals acc).getD t 0 + 2 ^ 31 ≤ I128.max)
    (hnar : ∀ t < n.val,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((ivals acc).getD t 0) ∧
        rnd32 ((ivals acc).getD t 0) ≤ (2 : ℤ) ^ 63 - 1)
    (hkn : k.val < n.val)
    (hinv : roundLoopInv acc n out k ok1) :
    WP.spec (learn_port.dense.round_all_loop.body acc n out k ok1)
      (fun cf => ∃ out' k', cf = ControlFlow.cont (out', k', true) ∧
        k'.val = k.val + 1 ∧ roundLoopInv acc n out' k' true) := by
  obtain ⟨hok, _hkle, hout⟩ := hinv
  have hklt : k.val < acc.val.length := by rw [hlen]; exact hkn
  have hidx := dense_slice_index_getD acc k hklt
  have hval : (acc.val.getD k.val 0#i128).val = (ivals acc).getD k.val 0 :=
    (ivals_getD acc hklt).symm
  obtain ⟨v, hr, hv⟩ := dense_round32_spec (acc.val.getD k.val 0#i128) (by rw [hval]; exact hround k.val hkn)
  obtain ⟨q, hn, hq⟩ := dense_narrow_spec v (by rw [hv, hval]; exact hnar k.val hkn)
  have hlenout : out.val.length < Usize.max := by
    have h1 : (vvals out).length = out.val.length := by simp [vvals]
    rw [← h1, hout, rndAll_length]
    have := dense_usize_le_max n
    omega
  obtain ⟨out1, hp, hout1⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out q hlenout)
  obtain ⟨k1, hk1, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize)
      (by show k.val + (1#usize).val ≤ Usize.max
          rw [show (1#usize).val = 1 from rfl]
          have := dense_usize_le_max n
          omega))
  have hk1_val : k1.val = k.val + 1 := by rw [hk1v]; rfl
  have hkProp : k < n := (UScalar.lt_equiv k n).2 hkn
  have hexec := round_all_loop_body_exec acc n out k v q out1 k1 hkProp hidx hr hn hp hk1
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out1, k1, true), hexec, out1, k1, rfl, hk1_val, ?_⟩
  refine ⟨rfl, by omega, ?_⟩
  have hval1 : vvals out1 = vvals out ++ [q.val] := by simp [vvals, hout1]
  rw [hval1, hout, hq, hv, hval, hk1_val, rndAll_succ]

/-- Full specification of `round_all_loop`. -/
theorem round_all_loop_spec
    (acc : Slice I128) (n : Usize) (out : alloc.vec.Vec I64) (k : Usize)
    (hlen : acc.val.length = n.val)
    (hround : ∀ t < n.val,
      I128.min ≤ (ivals acc).getD t 0 + 2 ^ 31 ∧ (ivals acc).getD t 0 + 2 ^ 31 ≤ I128.max)
    (hnar : ∀ t < n.val,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((ivals acc).getD t 0) ∧
        rnd32 ((ivals acc).getD t 0) ≤ (2 : ℤ) ^ 63 - 1)
    (hinv0 : roundLoopInv acc n out k true) :
    WP.spec (learn_port.dense.round_all_loop acc n out k true)
      (fun (out', ok') => ok' = true ∧ vvals out' = rndAll (ivals acc) n.val) := by
  unfold learn_port.dense.round_all_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool => n.val - s.2.1.val)
    (inv := fun s => roundLoopInv acc n s.1 s.2.1 s.2.2)
    (post := fun b : alloc.vec.Vec I64 × Bool =>
      b.2 = true ∧ vvals b.1 = rndAll (ivals acc) n.val)
  · rintro ⟨out', k', ok'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', hk'le, hout'⟩ := hinv
    rw [hok']
    by_cases hklt : k'.val < n.val
    · obtain ⟨cf, hstep, out'', k'', hcf, hk''_val, hinv''⟩ := WP.spec_imp_exists
        (round_all_loop_body_step acc n out' k' true hlen hround hnar hklt ⟨rfl, hk'le, hout'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (k' < n) := fun hc => hklt ((UScalar.lt_equiv k' n).1 hc)
      have hbody : learn_port.dense.round_all_loop.body acc n out' k' true =
          ok (ControlFlow.done (out', true)) := by
        rw [learn_port.dense.round_all_loop.body, if_neg hnot]
      simp only [hbody]
      have hk'_eq : k'.val = n.val := by omega
      refine ⟨rfl, ?_⟩
      rw [hout', hk'_eq]
  · exact hinv0

/-- `round_all` rounds every accumulator to Q32.32 under the explicit no-overflow domain. -/
theorem round_all_spec (acc : Slice I128)
    (hround : ∀ t < acc.val.length,
      I128.min ≤ (ivals acc).getD t 0 + 2 ^ 31 ∧ (ivals acc).getD t 0 + 2 ^ 31 ≤ I128.max)
    (hnar : ∀ t < acc.val.length,
      -((2 : ℤ) ^ 63) ≤ rnd32 ((ivals acc).getD t 0) ∧
        rnd32 ((ivals acc).getD t 0) ≤ (2 : ℤ) ^ 63 - 1) :
    ∃ out, learn_port.dense.round_all acc = ok (core.result.Result.Ok out) ∧
      vvals out = rndAll (ivals acc) acc.val.length := by
  have hlen : acc.val.length = (Slice.len acc).val := by rw [Slice.len_val]
  have hinv0 : roundLoopInv acc (Slice.len acc) (alloc.vec.Vec.with_capacity I64 (Slice.len acc))
      0#usize true := by
    refine ⟨rfl, by show (0#usize).val ≤ (Slice.len acc).val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    show (alloc.vec.Vec.with_capacity I64 (Slice.len acc)).val.map (fun z => z.val) =
      pushN [] (fun c => rnd32 ((ivals acc).getD c 0)) (0#usize).val
    rw [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]
    rfl
  have hspec := round_all_loop_spec acc (Slice.len acc) (alloc.vec.Vec.with_capacity I64 (Slice.len acc))
    0#usize hlen (by rwa [hlen] at hround) (by rwa [hlen] at hnar) hinv0
  obtain ⟨⟨out1, ok1⟩, hloop, hok1, hout1⟩ := (WP.spec_equiv_exists _ _).mp hspec
  subst hok1
  refine ⟨out1, ?_, by rw [hout1, hlen]⟩
  unfold learn_port.dense.round_all
  dsimp only
  rw [hloop]
  simp only [bind_tc_ok]
  simp

end apeiron_word
