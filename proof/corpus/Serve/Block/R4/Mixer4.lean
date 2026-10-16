/-
  `Serve/Block/R4/Mixer4.lean` — the extraction-level refinement of `mixer.l2_normalize`
  (`src/engine/mixer.rs:1194-1235`), the Q32.32 L2 normalisation used by the serving path.  It
  continues `Serve/Block/Mixer.lean` (refusal propagation), `Serve/Block/Mixer2.lean` (the Q32.32
  leaf specs) and `Serve/Block/Mixer3.lean` (the `div_q32` value theorem and the `delta_head_row`
  discharge); the two loops and the driver are closed here.

  What the Rust body computes (`l2_normalize_loop0` → `isqrt` → `l2_normalize_loop1`):

  * `sum = Σ xᵢ²` over the slice, in an `i128` accumulator that refuses (`Overflow`) on the first
    prefix sum outside `i128`;
  * `norm = round(isqrt sum)` to nearest, `den = max norm eps`; `den ≤ 0` is `BadConfig`;
  * each output is the half-biased Q32.32 quotient `xᵢ·2³² / den` truncated toward zero, narrowed
    to `i64` (`Overflow` when the quotient does not fit).

  The references below are plain-`Int`/`Nat` definitions of that *meaning* — the sum of squares, the
  integer square root `Nat.sqrt`, the round-to-nearest scale, `max`, and the Q32.32 quotient — never
  a restatement of the Rust loop.  The loop invariants are `loop.spec_decr_nat` on the remaining
  index, in the style of `Serve/Block/Mixer3.lean` and `Numeric/Engine/Rsqrt.lean`.

  Boundary, stated rather than papered over:

  * `isqrt` is characterised by `isqrt_spec` (`Numeric/Engine/Rsqrt.lean`), which is why
    `mixer4_isqrtRef` is `Nat.sqrt` and the driver carries the `2^116` bound that `isqrt_spec`
    requires (`hdeep`);
  * the two `i64`-narrowing sites and the `den ≤ 0` message conversion are the `#print axioms`
    boundaries of `Serve/Block/Mixer.lean` §Boundary: `mixer.l2_normalize`'s own closure reaches
    `I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from` and
    `alloc.string.String.Insts.CoreConvertFromShared0Str.from`, so every theorem whose statement
    mentions `mixer.l2_normalize` or its loops carries those in its cone;
  * the `Vec` capacity growth is modelled by `alloc.vec.Vec.push`'s spec, which needs
    `out.val.length < Usize.max`; that is the stated `hcap` hypothesis;
  * not covered: rustc/LLVM codegen, the F32/F64 carriers, AVX2/`unsafe`/opaque bodies.
-/

import Serve.Block.Mixer3
import Numeric.Engine.Rsqrt

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## §1 · References — what the kernel computes

`mixer4_sumSq` is the exact integer sum of squares, `mixer4_isqrtRef` the integer square root,
`mixer4_roundRef` the nearest-integer rounding of the root, `mixer4_denRef` the `max` with `eps`,
and `mixer4_divRef` the half-biased Q32.32 quotient (shared with `mixer.div_q32`). -/

/-- The exact sum of squares `Σ vᵢ²` of a `Slice`'s words, as an integer. -/
def mixer4_sumSq (l : List Std.I64) : ℤ := (l.map (fun v => v.val * v.val)).sum

/-- The integer square root of a nonnegative integer: `isqrt`'s specification, in reference form. -/
def mixer4_isqrtRef (n : ℤ) : ℤ := (Nat.sqrt n.toNat : ℤ)

/-- `round(isqrt sum)`: `s` or `s + 1`, whichever is nearer to the real root. -/
def mixer4_roundRef (s sum : ℤ) : ℤ := if sum - s * s > s then s + 1 else s

/-- The `l2_normalize` scale: the integer square root of the sum of squares, rounded to nearest. -/
def mixer4_normRef (sum : ℤ) : ℤ := mixer4_roundRef (mixer4_isqrtRef sum) sum

/-- The denominator `max ‖x‖₂ eps` of the scaling quotient. -/
def mixer4_denRef (sum eps : ℤ) : ℤ := max (mixer4_normRef sum) eps

/-- The per-element quotient `xᵢ·2³² / den`, half-biased and truncated toward zero (the same
reference as `mixer.div_q32`, whose divisor is `den` here). -/
def mixer4_divRef (num den : ℤ) : ℤ := mixer2_divQ32Ref num den

/-- The output word for the element `v`: the reference quotient narrowed to `i64`. -/
def mixer4_outRef (den : ℤ) (v : Std.I64) : Std.I64 :=
  ⟨BitVec.ofInt 64 (mixer4_divRef v.val den)⟩

/-- The whole output of `l2_normalize`: `x / max(‖x‖₂, eps)` elementwise, in Q32.32. -/
def mixer4_expected (den : ℤ) (l : List Std.I64) : List Std.I64 := l.map (mixer4_outRef den)

/-! ## §2 · List and integer helpers -/
/-- `Σ vᵢ²` on a cons. -/
theorem mixer4_sumSq_cons (v : Std.I64) (l : List Std.I64) :
    mixer4_sumSq (v :: l) = v.val * v.val + mixer4_sumSq l := by
  simp only [mixer4_sumSq, List.map_cons, List.sum_cons]

/-- `Σ vᵢ²` on the empty list. -/
theorem mixer4_sumSq_nil : mixer4_sumSq [] = 0 := by
  simp only [mixer4_sumSq, List.map_nil, List.sum_nil]

/-- `Σ vᵢ²` on a singleton. -/
theorem mixer4_sumSq_singleton (v : Std.I64) : mixer4_sumSq [v] = v.val * v.val := by
  rw [mixer4_sumSq_cons, mixer4_sumSq_nil]; ring

/-- `Σ vᵢ²` is nonnegative.  Temporal cost `O(len)`, no allocation. -/
theorem mixer4_sumSq_nonneg (l : List Std.I64) : 0 ≤ mixer4_sumSq l := by
  induction l with
  | nil => rw [mixer4_sumSq_nil]
  | cons v t ih => rw [mixer4_sumSq_cons]; nlinarith [mul_self_nonneg v.val, ih]

/-- **The integer square root is unique.**  A value whose square is at most `n` while `(s+1)²`
exceeds it is exactly `Nat.sqrt n`.  This turns `isqrt_spec`'s inequalities into the reference. -/
theorem mixer4_isqrt_unique {n s : ℤ} (hn : 0 ≤ n) (h1 : s * s ≤ n)
    (h2 : n < (s + 1) * (s + 1)) : s = mixer4_isqrtRef n := by
  have hs0 : 0 ≤ s := by nlinarith [h1, h2]
  have hs' : (s.toNat : ℤ) = s := Int.toNat_of_nonneg hs0
  have hn' : (n.toNat : ℤ) = n := Int.toNat_of_nonneg hn
  have h1n : s.toNat * s.toNat ≤ n.toNat := by
    have h : (((s.toNat * s.toNat : ℕ)) : ℤ) ≤ ((n.toNat : ℕ) : ℤ) := by
      push_cast
      rw [hs', hn']
      exact h1
    exact_mod_cast h
  have h2n : n.toNat < (s.toNat + 1) * (s.toNat + 1) := by
    have h : ((n.toNat : ℕ) : ℤ) < ((((s.toNat + 1) * (s.toNat + 1) : ℕ)) : ℤ) := by
      push_cast
      rw [hs', hn']
      exact h2
    exact_mod_cast h
  have hle : s.toNat ≤ Nat.sqrt n.toNat := Nat.le_sqrt.mpr h1n
  have hlt : Nat.sqrt n.toNat < s.toNat + 1 := Nat.sqrt_lt.mpr h2n
  have heq : s.toNat = Nat.sqrt n.toNat := by omega
  rw [mixer4_isqrtRef, ← heq, hs']

/-- `round(isqrt sum)` and the reference agree whenever `s` satisfies the `isqrt` inequalities. -/
theorem mixer4_roundRef_eq_normRef {sum s : ℤ} (hn : 0 ≤ sum) (h1 : s * s ≤ sum)
    (h2 : sum < (s + 1) * (s + 1)) : mixer4_roundRef s sum = mixer4_normRef sum := by
  rw [mixer4_normRef, ← mixer4_isqrt_unique hn h1 h2]

/-- The integer square root is nonnegative. -/
theorem mixer4_isqrtRef_nonneg (n : ℤ) : 0 ≤ mixer4_isqrtRef n := by
  rw [mixer4_isqrtRef]; exact Int.natCast_nonneg _

/-- `Nat.sqrt` never exceeds its argument, as an integer statement. -/
theorem mixer4_isqrtRef_le_self {sum : ℤ} (h : 0 ≤ sum) : mixer4_isqrtRef sum ≤ sum := by
  rw [mixer4_isqrtRef, ← Int.toNat_of_nonneg h]
  exact_mod_cast Nat.sqrt_le_self sum.toNat

/-- Rounding to nearest raises the root by at most one. -/
theorem mixer4_normRef_le_isqrtRef_succ (sum : ℤ) :
    mixer4_normRef sum ≤ mixer4_isqrtRef sum + 1 := by
  rw [mixer4_normRef, mixer4_roundRef]
  split <;> omega

/-- The scale is at most `sum + 1` for a nonnegative sum. -/
theorem mixer4_normRef_le_succ {sum : ℤ} (h : 0 ≤ sum) : mixer4_normRef sum ≤ sum + 1 := by
  have h1 := mixer4_isqrtRef_le_self h
  have h2 := mixer4_normRef_le_isqrtRef_succ sum
  omega

/-- `1 ≤ sum` makes the scale at least `1`. -/
theorem mixer4_normRef_pos {sum : ℤ} (h : 1 ≤ sum) : 1 ≤ mixer4_normRef sum := by
  have h0 : 0 ≤ sum := by omega
  have h1 : 1 ≤ mixer4_isqrtRef sum := by
    rw [mixer4_isqrtRef]
    have hsum1 : (1 : ℤ) ≤ (sum.toNat : ℤ) := by rw [Int.toNat_of_nonneg h0]; exact h
    have h1' : 1 ≤ sum.toNat := by exact_mod_cast hsum1
    have hs1 : 1 * 1 ≤ sum.toNat := by simpa using h1'
    have hsq : 1 ≤ Nat.sqrt sum.toNat := Nat.le_sqrt.mpr hs1
    exact_mod_cast hsq
  have h2 : mixer4_isqrtRef sum ≤ mixer4_normRef sum := by
    rw [mixer4_normRef, mixer4_roundRef]
    split <;> omega
  omega

/-- The denominator `max norm eps` stays inside `i128` for a bounded sum and an `i64` `eps`. -/
theorem mixer4_denRef_le_max (sum eps : ℤ) (h0 : 0 ≤ sum) (hsum : sum ≤ (2 : ℤ) ^ (116 : ℕ))
    (heps : eps ≤ I64.max) : mixer4_denRef sum eps ≤ IScalar.max IScalarTy.I128 := by
  rw [mixer4_denRef, mixer2_i128_max_eq]
  apply max_le
  · have h1 := mixer4_normRef_le_succ h0
    have h2 : (2 : ℤ) ^ (116 : ℕ) + 1 ≤ mixer2_i128Max := by norm_num [mixer2_i128Max]
    linarith
  · rw [I64.max_eq] at heps
    have h2 : (2 : ℤ) ^ (63 : ℕ) - 1 ≤ mixer2_i128Max := by norm_num [mixer2_i128Max]
    linarith

/-! ## §3 · Word-level leaves

The `i128 → i64` narrowing, the shifts, the `Ord::max` trait default and the `Vec` capacity are the
only primitives the driver needs beyond the Mixer2/Mixer3 leaves. -/

/-- **`i128::checked_add` succeeds exactly inside `i128`** and returns the integer sum. -/
theorem mixer4_checked_add_i128 (a b : Std.I128)
    (hlo : IScalar.min IScalarTy.I128 ≤ a.val + b.val)
    (hhi : a.val + b.val ≤ IScalar.max IScalarTy.I128) :
    ∃ c : Std.I128, I128.checked_add a b = some c ∧ c.val = a.val + b.val := by
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (IScalar.add_spec (x := a) (y := b) hlo hhi)
  exact ⟨c, by rw [I128.checked_add, core.num.checked_add_IScalar, hc]; rfl, hcv⟩

/-- The `i128` arithmetic right shift by `1` is `floor(·/2)`. -/
theorem mixer4_shiftRight1_i128 (d : Std.I128) :
    ∃ z : Std.I128, d >>> (1#i32) = ok z ∧ z.val = Int.ediv d.val 2 := by
  obtain ⟨z, hz, hzv, -⟩ := WP.spec_imp_exists
    (IScalar.ShiftRight_IScalar_spec (x := d) (y := (1#i32)) (by norm_num)
      (by norm_num [I128.numBits]))
  refine ⟨z, hz, ?_⟩
  rw [hzv, show (1#i32).toNat = 1 from by decide, Int.shiftRight_eq_div_pow]
  simp only [pow_one, Nat.cast_ofNat]
  rfl

/-- **`Ord::max` at `i128`** (through the trait default used by the extraction) returns the larger
value. -/
theorem mixer4_ord_max_default_i128 (x y : Std.I128) :
    ∃ z : Std.I128, core.cmp.Ord.max.default core.cmp.OrdI128 x y = ok z ∧
      z.val = Max.max x.val y.val := by
  refine ⟨if y.val < x.val then x else y, ?_, ?_⟩
  · unfold core.cmp.Ord.max.default core.cmp.Ord.max_body
    simp [core.cmp.impls.PartialOrdI128.lt, liftFun2]
    split <;> rfl
  · by_cases h : y.val < x.val
    · rw [if_pos h, max_def, if_neg (not_le.mpr h)]
    · rw [if_neg h, max_def, if_pos (le_of_not_gt h)]

/-- The `i64` word of the reference quotient carries that value when it fits `i64`. -/
theorem mixer4_outRef_val (den : ℤ) (v : Std.I64)
    (hlo : I64.min ≤ mixer4_divRef v.val den) (hhi : mixer4_divRef v.val den ≤ I64.max) :
    (mixer4_outRef den v).val = mixer4_divRef v.val den := by
  have hlo' : -(2 : ℤ) ^ (63 : ℕ) ≤ mixer4_divRef v.val den := by
    have h := hlo; rw [I64.min_eq] at h
    linarith
  have hhi' : mixer4_divRef v.val den < (2 : ℤ) ^ (63 : ℕ) := by
    have h := hhi; rw [I64.max_eq] at h
    have h2 : (2 : ℤ) ^ (63 : ℕ) = 9223372036854775808 := by norm_num
    linarith
  show (BitVec.ofInt 64 (mixer4_divRef v.val den)).toInt = mixer4_divRef v.val den
  exact BitVec.toInt_ofInt_eq_self (by norm_num) hlo' hhi'

/-- `Vec::with_capacity` starts empty. -/
theorem mixer4_with_capacity_val (T : Type) (n : Std.Usize) :
    (alloc.vec.Vec.with_capacity T n).val = [] := rfl

/-! ## §4 · `mixer.l2_normalize_loop0` — the sum of squares

The accumulator loop.  On a state whose running sum is `total - Σ_{i ≥ pos} xᵢ²`, one turn adds
`x_pos²` and advances the cursor; the turn is defined exactly when the new running sum stays inside
`i128`, which the stated domain (`total` itself inside `i128`) guarantees. -/

/-- **One `l2_normalize_loop0` turn.**  For a cursor inside the slice and a running sum equal to
`total` minus the sum of squares of the remaining suffix, the body adds the current square, keeps
`err = none`, and advances the cursor. -/
theorem mixer4_loop0_body_step (x : Slice Std.I64) (acc sum : Std.I128) (pos : Std.Usize)
    (hposlt : pos.val < x.val.length)
    (hsum0 : 0 ≤ sum.val)
    (hsumv : sum.val = (acc.val + mixer4_sumSq x.val) - mixer4_sumSq (x.val.drop pos.val))
    (hfit : acc.val + mixer4_sumSq x.val ≤ IScalar.max IScalarTy.I128) :
    ∃ (sum' : Std.I128) (pos' : Std.Usize),
      mixer.l2_normalize_loop0.body x (none : Option step.StepError) sum pos
          = ok (ControlFlow.cont (none, sum', pos')) ∧
      sum'.val = (acc.val + mixer4_sumSq x.val) - mixer4_sumSq (x.val.drop pos'.val) ∧
      0 ≤ sum'.val ∧ pos'.val = pos.val + 1 := by
  unfold mixer.l2_normalize_loop0.body
  dsimp only
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hposlt)]
  rw [if_pos (by rfl : core.option.Option.is_none (none : Option step.StepError) = true)]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec x pos hposlt)
  rw [hi1]
  simp only [lift, bind_tc_ok]
  obtain ⟨hb1, hb2⟩ := mixer2_i64_val_bounds i1
  have hv1 : (core.convert.num.FromI128I64.from i1).val = i1.val := mixer3_from_val i1
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
    (IScalar.mul_spec (x := core.convert.num.FromI128I64.from i1)
      (y := core.convert.num.FromI128I64.from i1)
      (by rw [mixer2_i128_min_eq, hv1]; simp only [mixer2_i128Min]; nlinarith [mul_self_nonneg i1.val])
      (by rw [mixer2_i128_max_eq, hv1]; simp only [mixer2_i128Max]; nlinarith [hb1, hb2, sq_nonneg i1.val]))
  rw [hi2]
  simp only [bind_tc_ok]
  have hi2nn : 0 ≤ i2.val := by rw [hi2v, hv1]; exact mul_self_nonneg i1.val
  have hdrop : x.val.drop pos.val =
      x.val[pos.val] :: x.val.drop (pos.val + 1) := List.drop_eq_getElem_cons hposlt
  have hsq : sum.val + i2.val =
      (acc.val + mixer4_sumSq x.val) - mixer4_sumSq (x.val.drop (pos.val + 1)) := by
    rw [hsumv, hdrop, mixer4_sumSq_cons]
    have hx : (x.val[pos.val]).val = i1.val := by rw [hi1v]
    rw [hi2v, hv1, hx]
    ring
  have hdropnn : 0 ≤ mixer4_sumSq (List.drop (↑pos + 1) ↑x) := mixer4_sumSq_nonneg _
  obtain ⟨tot, htot, htotv⟩ := mixer4_checked_add_i128 sum i2
    (by rw [mixer2_i128_min_eq]; simp only [mixer2_i128Min]; linarith [hsum0, hi2nn])
    (by
      simp only [mixer2_i128_max_eq, mixer2_i128Max] at hfit ⊢
      linarith [hdropnn, hsq, hfit])
  rw [htot]
  simp only [bind_tc_ok]
  have hmax : pos.val + (1#usize).val ≤ Usize.max := by
    have := Slice.length_ineq x
    have h1v : (1#usize).val = 1 := by decide
    omega
  obtain ⟨pos1, hpos1, hpos1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := pos) (y := 1#usize) hmax)
  rw [hpos1]
  simp only [bind_tc_ok]
  have h1v : (1#usize).val = 1 := by decide
  refine ⟨tot, pos1, rfl, ?_, ?_, ?_⟩
  · rw [htotv, hsq, hpos1v, h1v]
  · rw [htotv]; linarith [hsum0, hi2nn]
  · rw [hpos1v, h1v]

/-- **`l2_normalize_loop0` from cursor zero computes the sum of squares.**  For every accumulator
and slice whose final running sum `acc + Σ xᵢ²` is inside `i128`, the loop returns `err = none` and
that sum.  `proved`, relative to the boundary closure of `mixer.l2_normalize_loop0` (it contains no
boundary operation of its own). -/
theorem mixer4_loop0_spec (x : Slice Std.I64) (acc : Std.I128)
    (hacc : 0 ≤ acc.val)
    (hfit : acc.val + mixer4_sumSq x.val ≤ IScalar.max IScalarTy.I128) :
    mixer.l2_normalize_loop0 x (none : Option step.StepError) acc 0#usize ⦃ r =>
      ∃ sum : Std.I128, r = (none, sum) ∧ sum.val = acc.val + mixer4_sumSq x.val ⦄ := by
  unfold mixer.l2_normalize_loop0
  apply loop.spec_decr_nat
    (measure := fun s : (Option step.StepError) × Std.I128 × Std.Usize => x.val.length - s.2.2.val)
    (inv := fun s : (Option step.StepError) × Std.I128 × Std.Usize =>
      s.1 = none ∧ s.2.2.val ≤ x.val.length ∧ 0 ≤ s.2.1.val ∧
      s.2.1.val = (acc.val + mixer4_sumSq x.val) - mixer4_sumSq (x.val.drop s.2.2.val))
    (post := fun y : (Option step.StepError) × Std.I128 =>
      ∃ sum : Std.I128, y = (none, sum) ∧ sum.val = acc.val + mixer4_sumSq x.val)
  · rintro ⟨err, sum, pos⟩ ⟨herr, hposle, hsum0, hsumv⟩
    dsimp only at herr hposle hsum0 hsumv
    cases herr
    by_cases hlt : pos < Slice.len x
    · have hposlt : pos.val < x.val.length := by
        rw [UScalar.lt_equiv, Slice.len_val] at hlt; exact hlt
      obtain ⟨sum', pos1, hbody, hsum'v, hsum'0, hpos1v⟩ :=
        mixer4_loop0_body_step x acc sum pos hposlt hsum0 hsumv hfit
      dsimp only
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨trivial, by omega, hsum'0, hsum'v⟩, by omega⟩
    · have hge : x.val.length ≤ pos.val :=
        le_of_not_gt (by simpa only [UScalar.lt_equiv, Slice.len_val] using hlt)
      have hdrop : x.val.drop pos.val = [] := List.drop_eq_nil_of_le hge
      dsimp only
      unfold mixer.l2_normalize_loop0.body
      dsimp only
      rw [if_neg hlt]
      simp only [WP.spec_ok]
      refine ⟨sum, rfl, ?_⟩
      rw [hsumv, hdrop, mixer4_sumSq_nil]
      ring
  · refine ⟨rfl, by rw [show (0#usize).val = 0 from by decide]; exact Nat.zero_le _, hacc, ?_⟩
    rw [show (0#usize : Std.Usize).val = 0 from by decide, List.drop_zero]
    show (acc.val : ℤ) = acc.val + mixer4_sumSq x.val - mixer4_sumSq x.val
    ring

/-! ## §5 · `mixer.l2_normalize_loop1` — the per-element quotient

The scaling loop.  Its reference is the prefix map: after `pos` turns the output vector is the
initial vector followed by the quotients of `x₀ … x_{pos-1}`. -/

/-- **One `l2_normalize_loop1` turn.**  For a cursor inside the slice, a positive denominator, and a
reference quotient of the current element that fits `i64`, the body pushes that quotient, keeps
`err = none`, and advances the cursor. -/
theorem mixer4_loop1_body_step (x : Slice Std.I64) (den half : Std.I128)
    (out : alloc.vec.Vec Std.I64) (pos : Std.Usize)
    (hposlt : pos.val < x.val.length)
    (hden : 0 < den.val) (hdenle : den.val ≤ IScalar.max IScalarTy.I128)
    (hhalf : half.val = Int.ediv den.val 2)
    (hfit : I64.min ≤ mixer4_divRef (x.val[pos.val]'hposlt).val den.val ∧
      mixer4_divRef (x.val[pos.val]'hposlt).val den.val ≤ I64.max)
    (hcap : out.val.length < Usize.max) :
    ∃ (out1 : alloc.vec.Vec Std.I64) (pos1 : Std.Usize),
      mixer.l2_normalize_loop1.body x den half (none : Option step.StepError) out pos
          = ok (ControlFlow.cont (none, out1, pos1)) ∧
      pos1.val = pos.val + 1 ∧
      out1.val = out.val ++ [mixer4_outRef den.val (x.val[pos.val]'hposlt)] := by
  unfold mixer.l2_normalize_loop1.body
  dsimp only
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hposlt)]
  rw [if_pos (by rfl : core.option.Option.is_none (none : Option step.StepError) = true)]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec x pos hposlt)
  rw [hi1]
  simp only [lift, bind_tc_ok]
  have hi1val : i1.val = (x.val[pos.val]'hposlt).val := by rw [hi1v]
  have hv1 : (core.convert.num.FromI128I64.from i1).val = i1.val := mixer3_from_val i1
  have h95 : |i1.val| ≤ (2 : ℤ) ^ (95 : ℕ) - 1 := by
    have h := mixer3_i64_abs_le i1
    have h2 : (2 : ℤ) ^ (63 : ℕ) ≤ (2 : ℤ) ^ (95 : ℕ) - 1 := by norm_num
    linarith
  obtain ⟨num, hnum, hnumv⟩ :=
    mixer3_shiftLeft32_val (core.convert.num.FromI128I64.from i1) (by rw [hv1]; exact h95)
  rw [hnum]
  simp only [bind_tc_ok]
  have hnumval : num.val = i1.val * 4294967296 := by rw [hnumv, hv1]
  have hnumabs : |num.val| ≤ (2 : ℤ) ^ (95 : ℕ) := by
    rw [hnumval, abs_mul, abs_of_nonneg (by norm_num : (0 : ℤ) ≤ 4294967296)]
    have hb' : |i1.val| ≤ (2 : ℤ) ^ (63 : ℕ) := mixer3_i64_abs_le i1
    have h2 : (2 : ℤ) ^ (63 : ℕ) * 4294967296 = (2 : ℤ) ^ (95 : ℕ) := by norm_num
    nlinarith [hb', abs_nonneg i1.val]
  obtain ⟨hnlo, hnhi⟩ := abs_le.mp hnumabs
  have hhalf0 : 0 ≤ half.val := by
    rw [hhalf]; exact Int.ediv_nonneg (le_of_lt hden) (by norm_num)
  have hhalf_le : half.val ≤ (2 : ℤ) ^ (126 : ℕ) := by
    rw [hhalf]
    have hq : (↑den) / 2 < (2 : ℤ) ^ (126 : ℕ) := by
      rw [Int.ediv_lt_iff_lt_mul (by norm_num : (0 : ℤ) < 2)]
      rw [mixer2_i128_max_eq] at hdenle
      have h2 : mixer2_i128Max < (2 : ℤ) ^ (126 : ℕ) * 2 := by norm_num [mixer2_i128Max]
      linarith
    exact le_of_lt hq
  have hq : ∃ q : Std.I128,
      (if num >= 0#i128
       then (do let i3 ← num + half; i3 / den)
       else (do let i3 ← -.num; let i4 ← i3 + half; let i5 ← i4 / den; -.i5))
        = ok q ∧ q.val = mixer4_divRef (x.val[pos.val]'hposlt).val den.val := by
    by_cases hge : num ≥ 0#i128
    · have hnn : 0 ≤ num.val := by
        have h := (IScalar.le_equiv (0#i128) num).1 hge
        rwa [show (0#i128).val = 0 from by decide] at h
      have hxnn : 0 ≤ (x.val[pos.val]'hposlt).val := by
        rw [← hi1val]
        exact mixer3_nonneg_of_mul_nonneg (by rw [← hnumval]; exact hnn)
      rw [if_pos hge]
      obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
        (IScalar.add_spec (x := num) (y := half)
          (by rw [mixer2_i128_min_eq]; simp only [mixer2_i128Min]; linarith [hnn, hhalf0])
          (by rw [mixer2_i128_max_eq]; simp only [mixer2_i128Max]; linarith [hnhi, hhalf_le]))
      rw [hi3]
      simp only [bind_tc_ok]
      have hi3nn : 0 ≤ i3.val := by rw [hi3v]; linarith [hnn, hhalf0]
      obtain ⟨qd, hqd, hqdv⟩ := IScalar.div_spec (x := i3) (y := den)
        (by omega)
        (by rintro ⟨-, h⟩; omega)
      rw [hqd]
      refine ⟨qd, rfl, ?_⟩
      rw [hqdv, Int.tdiv_eq_ediv_of_nonneg hi3nn, hi3v, hnumval, hi1val, hhalf,
        mixer4_divRef, mixer2_divQ32Ref, if_pos hxnn]
      rfl
    · have hneg : num.val < 0 := by
        by_contra hc
        exact hge ((IScalar.le_equiv (0#i128) num).2 (by
          rw [show (0#i128).val = 0 from by decide]; omega))
      have hxneg : ¬ ((0 : ℤ) ≤ (x.val[pos.val]'hposlt).val) := by
        rw [← hi1val]
        intro hc
        have hnumpos : 0 ≤ num.val := by rw [hnumval]; nlinarith [hc]
        omega
      rw [if_neg hge]
      obtain ⟨xn, hxn, hxnv⟩ := mixer3_neg_val num
        (by rw [mixer2_i128_min_eq]; simp only [mixer2_i128Min]; linarith [hnhi])
        (by rw [mixer2_i128_max_eq]; simp only [mixer2_i128Max]; linarith [hnlo])
      rw [hxn]
      simp only [bind_tc_ok]
      have hxnpos : 0 < xn.val := by rw [hxnv]; omega
      obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
        (IScalar.add_spec (x := xn) (y := half)
          (by rw [mixer2_i128_min_eq, hxnv]; simp only [mixer2_i128Min]; linarith [hnhi, hhalf0])
          (by rw [mixer2_i128_max_eq, hxnv]; simp only [mixer2_i128Max]; linarith [hnlo, hhalf_le]))
      rw [hi3]
      simp only [bind_tc_ok]
      have hi3pos : 0 < i3.val := by rw [hi3v]; linarith [hxnpos, hhalf0]
      obtain ⟨i4, hi4, hi4v⟩ := IScalar.div_spec (x := i3) (y := den)
        (by omega)
        (by rintro ⟨-, h⟩; omega)
      rw [hi4]
      simp only [bind_tc_ok]
      have hi4v' : i4.val = Int.ediv i3.val den.val := by
        rw [hi4v]
        exact Int.tdiv_eq_ediv_of_nonneg (le_of_lt hi3pos)
      have hi4nn : 0 ≤ i4.val := by
        rw [hi4v']; exact Int.ediv_nonneg (le_of_lt hi3pos) (le_of_lt hden)
      have hi4le : Int.ediv i3.val den.val ≤ (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (126 : ℕ) := by
        calc Int.ediv i3.val den.val ≤ i3.val := Int.ediv_le_self den.val (le_of_lt hi3pos)
          _ = ↑xn + ↑half := hi3v
          _ ≤ (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (126 : ℕ) := by
              have hxnle : ↑xn ≤ (2 : ℤ) ^ (95 : ℕ) := by rw [hxnv]; linarith [hnlo]
              linarith [hxnle, hhalf_le]
      have hpow : (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (126 : ℕ) < (2 : ℤ) ^ (127 : ℕ) := by
        norm_num
      obtain ⟨q, hq', hq'v⟩ := mixer3_neg_val i4
        (by rw [mixer2_i128_min_eq, hi4v']; simp only [mixer2_i128Min]; linarith [hi4le, hpow])
        (by rw [mixer2_i128_max_eq, hi4v']; simp only [mixer2_i128Max]; linarith [hi4nn])
      rw [hq']
      refine ⟨q, rfl, ?_⟩
      rw [hq'v, hi4v', hi3v, hxnv, hnumval, hi1val, hhalf, mixer4_divRef, mixer2_divQ32Ref,
        if_neg hxneg]
      ring_nf
  obtain ⟨q, hqeq, hqval⟩ := hq
  rw [hqeq]
  simp only [bind_tc_ok]
  obtain ⟨w, hw, hwv⟩ := mixer2_try_from_val q (by rw [hqval]; exact hfit.1)
    (by rw [hqval]; exact hfit.2)
  rw [hw]
  simp only [bind_tc_ok]
  obtain ⟨out2, hout2, hout2v⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out w hcap)
  rw [hout2]
  simp only [bind_tc_ok]
  have hmax : pos.val + (1#usize).val ≤ Usize.max := by
    have := Slice.length_ineq x
    have h1v : (1#usize).val = 1 := by decide
    omega
  obtain ⟨pos1, hpos1, hpos1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := pos) (y := 1#usize) hmax)
  rw [hpos1]
  simp only [bind_tc_ok]
  have h1v : (1#usize).val = 1 := by decide
  have hwref : w = mixer4_outRef den.val (x.val[pos.val]'hposlt) := by
    have hwval : w.val = mixer4_divRef (x.val[pos.val]'hposlt).val den.val := by rw [hwv, hqval]
    apply IScalar.eq_of_val_eq
    rw [hwval]
    exact (mixer4_outRef_val den.val (x.val[pos.val]'hposlt) hfit.1 hfit.2).symm
  refine ⟨out2, pos1, rfl, ?_, ?_⟩
  · rw [hpos1v, h1v]
  · rw [hout2v, hwref]

/-- **`l2_normalize_loop1` from cursor zero maps the reference quotient over the slice.**  For a
positive denominator whose per-element reference quotients all fit `i64`, the loop returns
`err = none` and the initial vector extended by `xᵢ ↦ ⌊xᵢ·2³²/den⌉`. -/
theorem mixer4_loop1_spec (x : Slice Std.I64) (den half : Std.I128)
    (out : alloc.vec.Vec Std.I64)
    (hden : 0 < den.val) (hdenle : den.val ≤ IScalar.max IScalarTy.I128)
    (hhalf : half.val = Int.ediv den.val 2)
    (hfit : ∀ (j : ℕ) (hj : j < x.val.length),
      I64.min ≤ mixer4_divRef (x.val[j]'hj).val den.val ∧
      mixer4_divRef (x.val[j]'hj).val den.val ≤ I64.max)
    (hcap : out.val.length + x.val.length < Usize.max) :
    mixer.l2_normalize_loop1 x (none : Option step.StepError) den half out 0#usize ⦃ r =>
      ∃ out1 : alloc.vec.Vec Std.I64, r = (none, out1) ∧
        out1.val = out.val ++ x.val.map (mixer4_outRef den.val) ⦄ := by
  unfold mixer.l2_normalize_loop1
  apply loop.spec_decr_nat
    (measure := fun s : (Option step.StepError) × alloc.vec.Vec Std.I64 × Std.Usize =>
      x.val.length - s.2.2.val)
    (inv := fun s : (Option step.StepError) × alloc.vec.Vec Std.I64 × Std.Usize =>
      s.1 = none ∧ s.2.2.val ≤ x.val.length ∧
      s.2.1.val = out.val ++ (x.val.take s.2.2.val).map (mixer4_outRef den.val))
    (post := fun y : (Option step.StepError) × alloc.vec.Vec Std.I64 =>
      ∃ out1 : alloc.vec.Vec Std.I64, y = (none, out1) ∧
        out1.val = out.val ++ x.val.map (mixer4_outRef den.val))
  · rintro ⟨err, out1, pos⟩ ⟨herr, hposle, houtv⟩
    dsimp only at herr hposle houtv
    cases herr
    by_cases hlt : pos < Slice.len x
    · have hposlt : pos.val < x.val.length := by
        rw [UScalar.lt_equiv, Slice.len_val] at hlt; exact hlt
      have hcap' : out1.val.length < Usize.max := by
        rw [houtv, List.length_append, List.length_map, List.length_take]
        omega
      obtain ⟨out2, pos1, hbody, hpos1v, hout2v⟩ :=
        mixer4_loop1_body_step x den half out1 pos hposlt hden hdenle hhalf
          (hfit pos.val hposlt) hcap'
      dsimp only
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
      rw [hout2v, houtv, hpos1v, List.take_succ_eq_append_getElem hposlt, List.map_append,
        List.map_cons, List.map_nil, List.append_assoc]
    · have hge : x.val.length ≤ pos.val :=
        le_of_not_gt (by simpa only [UScalar.lt_equiv, Slice.len_val] using hlt)
      have htake : x.val.take pos.val = x.val := List.take_of_length_le hge
      dsimp only
      unfold mixer.l2_normalize_loop1.body
      dsimp only
      rw [if_neg hlt]
      simp only [WP.spec_ok]
      refine ⟨out1, rfl, ?_⟩
      rw [houtv, htake]
  · refine ⟨rfl, by rw [show (0#usize).val = 0 from by decide]; exact Nat.zero_le _, ?_⟩
    rw [show (0#usize).val = 0 from by decide, List.take_zero, List.map_nil, List.append_nil]

/-! ## §6 · `mixer.l2_normalize` — the driver

The loops composed: `sum = Σ xᵢ²`, `norm = round(isqrt sum)`, `den = max norm eps`, and the
scaling loop with `half = den / 2`. -/

/-- **`l2_normalize` computes `x / max(‖x‖₂, eps)` in Q32.32.**  Under the stated domain — the sum
of squares inside `i128` (`hfit`, the loop's no-overflow condition), at least `1` and at most `2¹¹⁶`
(`hnz`, `hdeep`, the `isqrt_spec` window), a slice shorter than `Usize.max` (`hcap`), and every
reference quotient fitting `i64` — the extracted body returns `Ok` with the reference output.
`proved`, relative to the boundary closure of `mixer.l2_normalize`. -/
theorem word_l2_normalize_ok (x : Slice Std.I64) (eps : Std.I64)
    (hfit : mixer4_sumSq x.val ≤ IScalar.max IScalarTy.I128)
    (hnz : 1 ≤ mixer4_sumSq x.val)
    (hdeep : mixer4_sumSq x.val ≤ (2 : ℤ) ^ (116 : ℕ))
    (hcap : x.val.length < Usize.max)
    (hquot : ∀ (j : ℕ) (hj : j < x.val.length),
      I64.min ≤ mixer4_divRef (x.val[j]'hj).val (mixer4_denRef (mixer4_sumSq x.val) eps.val) ∧
      mixer4_divRef (x.val[j]'hj).val (mixer4_denRef (mixer4_sumSq x.val) eps.val) ≤ I64.max) :
    ∃ out : alloc.vec.Vec Std.I64,
      mixer.l2_normalize x eps = ok (core.result.Result.Ok out) ∧
      out.val = mixer4_expected (mixer4_denRef (mixer4_sumSq x.val) eps.val) x.val := by
  have h0 : (0#i128).val = 0 := by decide
  unfold mixer.l2_normalize
  obtain ⟨y0, hy0, hpost0⟩ := WP.spec_imp_exists
    (mixer4_loop0_spec x 0#i128 (by rw [h0])
      (by rw [h0, zero_add]; exact hfit))
  obtain ⟨sum, hy0', hsumv⟩ := hpost0
  rw [hy0'] at hy0
  simp only [hy0, bind_tc_ok]
  have hsumv' : sum.val = mixer4_sumSq x.val := by rw [hsumv, h0, zero_add]
  -- isqrt
  obtain ⟨s, hs, hs1, hs2⟩ := WP.spec_imp_exists
    (isqrt_spec sum (by rw [hsumv']; exact hnz) (by rw [hsumv']; exact hdeep))
  simp only [Aeneas.Std.uncurry_apply_pair, hs]
  simp only [bind_tc_ok]
  have hsref : s.val = mixer4_isqrtRef sum.val :=
    mixer4_isqrt_unique (by rw [hsumv']; omega) hs1 hs2
  have hs0 : 0 ≤ s.val := by nlinarith [mul_self_nonneg s.val, hs1, hsumv', hdeep]
  have hs_hi : s.val ≤ (2 : ℤ) ^ (58 : ℕ) := by
    have hsq : s.val * s.val ≤ mixer4_sumSq x.val := by rw [← hsumv']; exact hs1
    nlinarith [hsq, hdeep, hs0, sq_nonneg (s.val - (2 : ℤ) ^ (58 : ℕ))]
  -- i = s * s
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (IScalar.mul_spec (x := s) (y := s)
      (by rw [mixer2_i128_min_eq]; simp only [mixer2_i128Min]; nlinarith [mul_self_nonneg s.val])
      (by rw [mixer2_i128_max_eq]; simp only [mixer2_i128Max]; nlinarith [hs_hi, hs0]))
  rw [hi]
  simp only [bind_tc_ok]
  -- i1 = sum - i
  have hsum_hi : sum.val ≤ (2 : ℤ) ^ (116 : ℕ) := by rw [hsumv']; exact hdeep
  have hsum_lo : (1 : ℤ) ≤ sum.val := by rw [hsumv']; exact hnz
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (IScalar.sub_spec (x := sum) (y := i)
      (by rw [mixer2_i128_min_eq, hiv]; simp only [mixer2_i128Min]; nlinarith [hsum_lo, hsum_hi, hs_hi, hs0])
      (by rw [mixer2_i128_max_eq, hiv]; simp only [mixer2_i128Max]; nlinarith [hsum_hi, hs_hi, hs0]))
  rw [hi1]
  simp only [bind_tc_ok]
  have hi1v' : i1.val = sum.val - s.val * s.val := by rw [hi1v, hiv]
  -- norm = if i1 > s then s + 1 else ok s
  obtain ⟨norm, hnorm, hnormv⟩ : ∃ norm : Std.I128,
      (if i1 > s then (s + 1#i128) else ok s) = ok norm ∧
        norm.val = mixer4_roundRef s.val sum.val := by
    by_cases hgt : i1 > s
    · have hgtv : s.val < sum.val - s.val * s.val := by
        rw [gt_iff_lt] at hgt
        rw [IScalar.lt_equiv] at hgt
        rw [hi1v'] at hgt
        exact hgt
      have hlo : IScalar.min IScalarTy.I128 ≤ s.val + (1#i128).val := by
        rw [mixer2_i128_min_eq, show (1#i128).val = 1 from by decide]
        simp only [mixer2_i128Min]
        linarith [hs0]
      have hhi : s.val + (1#i128).val ≤ IScalar.max IScalarTy.I128 := by
        rw [mixer2_i128_max_eq, show (1#i128).val = 1 from by decide]
        simp only [mixer2_i128Max]
        linarith [hs_hi]
      obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists (IScalar.add_spec (x := s) (y := 1#i128) hlo hhi)
      exact ⟨z, by rw [if_pos hgt]; exact hz,
        by rw [hzv, show (1#i128).val = 1 from by decide, mixer4_roundRef,
          if_pos (by omega)]⟩
    · have hgtv : ¬ (s.val < sum.val - s.val * s.val) := by
        intro hc
        exact hgt ((IScalar.lt_equiv s i1).2 (by rw [hi1v']; exact hc))
      refine ⟨s, by rw [if_neg hgt], ?_⟩
      rw [mixer4_roundRef, if_neg (by omega)]
  rw [hnorm]
  simp only [bind_tc_ok]
  have hnormRef : norm.val = mixer4_normRef sum.val := by
    rw [hnormv, mixer4_roundRef_eq_normRef (by rw [hsumv']; omega) hs1 hs2]
  -- den = max norm eps
  simp only [lift, bind_tc_ok]
  obtain ⟨den, hden_eq, hdenv⟩ :=
    mixer4_ord_max_default_i128 norm (core.convert.num.FromI128I64.from eps)
  rw [hden_eq]
  simp only [bind_tc_ok]
  have heps : (core.convert.num.FromI128I64.from eps).val = eps.val := mixer3_from_val eps
  have hdenval : den.val = mixer4_denRef (mixer4_sumSq x.val) eps.val := by
    rw [hdenv, heps, hnormRef, hsumv', mixer4_denRef]
  have hdenpos : 0 < den.val := by
    rw [hdenval, mixer4_denRef]
    have := mixer4_normRef_pos hnz
    omega
  -- the refusal branch is not taken
  rw [if_neg (by
    rw [IScalar.le_equiv, show (0#i128).val = 0 from by decide]
    exact not_le.mpr hdenpos)]
  -- half
  obtain ⟨half, hhalf, hhalfv⟩ := mixer4_shiftRight1_i128 den
  rw [hhalf]
  simp only [bind_tc_ok]
  have hhalfv' : half.val = Int.ediv den.val 2 := hhalfv
  have hdenle : den.val ≤ IScalar.max IScalarTy.I128 := by
    rw [hdenval]
    exact mixer4_denRef_le_max (mixer4_sumSq x.val) eps.val (mixer4_sumSq_nonneg x.val) hdeep
      (by rw [I64.max_eq]; exact (mixer2_i64_val_bounds eps).2)
  -- the scaling loop
  obtain ⟨y1, hy1, hpost1⟩ := WP.spec_imp_exists
    (mixer4_loop1_spec x den half (alloc.vec.Vec.with_capacity Std.I64 (Slice.len x))
      hdenpos hdenle hhalfv'
      (by intro j hj; rw [hdenval]; exact hquot j hj)
      (by rw [mixer4_with_capacity_val, List.length_nil]; omega))
  obtain ⟨out1, hy1', hout1⟩ := hpost1
  rw [hy1'] at hy1
  rw [hy1]
  simp only [bind_tc_ok]
  refine ⟨out1, rfl, ?_⟩
  rw [hout1, mixer4_with_capacity_val, List.nil_append, mixer4_expected, hdenval]

/-! ## §7 · The refusal — an exact condition

The scale can be non-positive only when the sum of squares is `0` (no element contributes) and
`eps` does not lift it: `den = max 0 eps ≤ 0` exactly then.  The empty slice is the concrete
witness, and `isqrt 0 = 0` is proved from the extraction. -/

/-- **`isqrt 0 = 0`**: the seed reduction walks `bit` to `0` and the extraction loop exits at once. -/
theorem mixer4_isqrt_zero : fixed_rsqrt.isqrt 0#i128 = ok 0#i128 := by
  unfold fixed_rsqrt.isqrt
  obtain ⟨bit0, hbit0, hbit0v, -⟩ := WP.spec_imp_exists
    (I128.ShiftLeft_IScalar_spec (x := 1#i128) (y := 126#i32) (by norm_num)
      (by norm_num [I128.numBits]))
  rw [hbit0]
  simp only [bind_tc_ok]
  have hbit0' : bit0.val = (2 : ℤ) ^ (126 : ℕ) := by
    rw [hbit0v, show (1#i128).val = 1 from by decide,
      show (126#i32).toNat = 126 from by decide, Int.shiftLeft_eq]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.bmod]
  have hloop0 : fixed_rsqrt.isqrt_loop0 0#i128 bit0 ⦃ out => out.val = 0 ⦄ := by
    unfold fixed_rsqrt.isqrt_loop0
    apply loop.spec_decr_nat
      (measure := fun b : I128 => (b.val).toNat)
      (inv := fun b : I128 => 0 ≤ b.val)
      (post := fun out : I128 => out.val = 0)
    · rintro b hb
      unfold fixed_rsqrt.isqrt_loop0.body
      by_cases hgt : b > (0#i128)
      · rw [if_pos hgt]
        obtain ⟨b1, hb1, hb1v, -⟩ := WP.spec_imp_exists
          (I128.ShiftRight_IScalar_spec (x := b) (y := 2#i32) (by norm_num)
            (by norm_num [I128.numBits]))
        rw [hb1]
        simp only [bind_tc_ok]
        refine (WP.spec_ok (ControlFlow.cont b1)).mpr ?_
        dsimp only
        have hbv : 0 < b.val := (IScalar.lt_equiv (0#i128) b).mp hgt
        have hb1v' : b1.val = b.val / 4 := by
          rw [hb1v, show (2#i32).toNat = 2 from by decide, Int.shiftRight_eq_div_pow]
          norm_num
        refine ⟨?_, ?_⟩
        · rw [hb1v']; exact Int.ediv_nonneg (le_of_lt hbv) (by norm_num)
        · rw [hb1v', Int.toNat_lt_toNat hbv]
          exact (Int.ediv_lt_iff_lt_mul (by norm_num : (0 : ℤ) < 4)).mpr (by omega)
      · rw [if_neg hgt]
        simp only [WP.spec_ok]
        have hbv : b.val ≤ 0 := by
          have h := not_lt.mp ((not_congr (IScalar.lt_equiv (0#i128) b)).mp hgt)
          rwa [show (0#i128).val = 0 from by decide] at h
        omega
    · rw [hbit0']; norm_num
  obtain ⟨b1, hb1, hb1post⟩ := WP.spec_imp_exists hloop0
  rw [hb1]
  simp only [bind_tc_ok]
  have hloop1 : fixed_rsqrt.isqrt_loop1 0#i128 0#i128 b1 ⦃ out => out.val = 0 ⦄ := by
    unfold fixed_rsqrt.isqrt_loop1
    apply loop.spec_decr_nat
      (measure := fun _ : I128 × I128 × I128 => 0)
      (inv := fun s : I128 × I128 × I128 => s.2.2.val = 0 ∧ s.2.1.val = 0)
      (post := fun r : I128 => r.val = 0)
    · rintro ⟨rem, res, bit⟩ ⟨hbit, hres⟩
      dsimp only at hbit hres
      have hbit0 : bit = 0#i128 :=
        IScalar.eq_of_val_eq (by rw [hbit]; decide)
      rw [hbit0]
      unfold fixed_rsqrt.isqrt_loop1.body
      dsimp only
      rw [if_neg (by decide : ¬ ((0#i128) != 0#i128))]
      simp only [WP.spec_ok]
      rw [hres]
    · refine ⟨hb1post, ?_⟩
      show (0#i128 : Std.I128).val = 0
      decide
  obtain ⟨b2, hb2, hb2v⟩ := WP.spec_imp_exists hloop1
  rw [hb2]
  exact congrArg ok (IScalar.eq_of_val_eq hb2v)

/-- **`l2_normalize` refuses on a non-positive scale.**  The exact condition: the sum of squares is
`0` (here the empty slice, the concrete witness) and `eps ≤ 0`, so `den = max 0 eps ≤ 0` and the body
returns `Err (BadConfig "l2 norm eps")` with the boundary-converted message. -/
theorem word_l2_normalize_refuses_badconfig (eps : Std.I64) (hle : eps.val ≤ 0) (msg : String)
    (hmsg : alloc.string.String.Insts.CoreConvertFromShared0Str.from (toStr "l2 norm eps")
      = ok msg) :
    mixer.l2_normalize (Slice.new Std.I64) eps
      = ok (core.result.Result.Err (step.StepError.BadConfig msg)) := by
  have h0 : (0#i128).val = 0 := by decide
  unfold mixer.l2_normalize
  obtain ⟨y0, hy0, hpost0⟩ := WP.spec_imp_exists
    (mixer4_loop0_spec (Slice.new Std.I64) 0#i128 (by rw [h0])
      (by rw [h0, zero_add, Slice.new_val, mixer4_sumSq_nil, mixer2_i128_max_eq]; norm_num [mixer2_i128Max]))
  obtain ⟨sum, hy0', hsumv⟩ := hpost0
  rw [hy0'] at hy0
  simp only [hy0, bind_tc_ok]
  have hsumv' : sum.val = 0 := by rw [hsumv, h0, zero_add, Slice.new_val, mixer4_sumSq_nil]
  have hsumz : sum = 0#i128 := IScalar.eq_of_val_eq (by rw [hsumv']; decide)
  simp only [Aeneas.Std.uncurry_apply_pair, hsumz, mixer4_isqrt_zero]
  simp only [bind_tc_ok]
  have hmul : (0#i128) * (0#i128) = ok (0#i128) := by
    obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists (IScalar.mul_spec (x := (0#i128)) (y := (0#i128))
      (by rw [mixer2_i128_min_eq]; simp only [mixer2_i128Min]; norm_num)
      (by rw [mixer2_i128_max_eq]; simp only [mixer2_i128Max]; norm_num))
    have hz0 : z = 0#i128 := IScalar.eq_of_val_eq (by rw [hzv]; norm_num)
    rw [hz0] at hz; exact hz
  rw [hmul]
  simp only [bind_tc_ok]
  have hsub : (0#i128) - (0#i128) = ok (0#i128) := by
    obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists (IScalar.sub_spec (x := (0#i128)) (y := (0#i128))
      (by rw [mixer2_i128_min_eq]; simp only [mixer2_i128Min]; norm_num)
      (by rw [mixer2_i128_max_eq]; simp only [mixer2_i128Max]; norm_num))
    have hz0 : z = 0#i128 := IScalar.eq_of_val_eq (by rw [hzv]; norm_num)
    rw [hz0] at hz; exact hz
  rw [hsub]
  simp only [bind_tc_ok]
  rw [if_neg (by
    rw [gt_iff_lt, IScalar.lt_equiv]
    simp)]
  simp only [lift, bind_tc_ok]
  obtain ⟨d, hd, hdv⟩ :=
    mixer4_ord_max_default_i128 (0#i128) (core.convert.num.FromI128I64.from eps)
  rw [hd]
  simp only [bind_tc_ok]
  have hdv' : d.val ≤ 0 := by
    rw [hdv, h0, mixer3_from_val]
    exact max_le (le_refl _) hle
  rw [if_pos (by rw [IScalar.le_equiv, h0]; exact hdv')]
  rw [hmsg]
  simp only [bind_tc_ok]

/-! ## §8 · Non-vacuity

The empty slice inhabits the refusal's exact condition, and `[1]` with `eps = 1` inhabits the
success domain end to end: the loops run, and the single output is `2³²`. -/

/-- `1 ≤ Usize.max`: the platform word is at least 32 bits, so `1` is a valid slice length. -/
lemma mixer4_one_le_usize_max : (1 : Nat) ≤ Usize.max := by
  rcases Usize.bounds_eq with h | h <;> rw [h]
  · rw [U32.max_eq]; norm_num
  · rw [U64.max_eq]; norm_num

/-- `1 < Usize.max`: strict, for the `Vec.push` capacity hypothesis `length < Usize.max`. -/
lemma mixer4_one_lt_usize_max : (1 : Nat) < Usize.max := by
  rcases Usize.bounds_eq with h | h <;> rw [h]
  · rw [U32.max_eq]; norm_num
  · rw [U64.max_eq]; norm_num

/-- **The refusal condition is inhabited.**  With the empty slice and `eps = 0` the body returns the
`BadConfig` message. -/
example : ∃ msg : String,
    mixer.l2_normalize (Slice.new Std.I64) 0#i64
      = ok (core.result.Result.Err (step.StepError.BadConfig msg)) := by
  refine ⟨_, word_l2_normalize_refuses_badconfig 0#i64 (by decide) _ rfl⟩

/-- **The success domain is inhabited and the output is attained.**  For `x = [1]` and `eps = 1` the
sum of squares is `1`, `isqrt 1 = 1`, `norm = den = 1`, and the single output value is `2³²`. -/
example : ∃ out : alloc.vec.Vec Std.I64,
    mixer.l2_normalize (Slice.from [1#i64] mixer4_one_le_usize_max) 1#i64
      = ok (core.result.Result.Ok out) ∧
    out.val.map (fun w => w.val) = [4294967296] := by
  have hxval : (Slice.from [1#i64] mixer4_one_le_usize_max).val = [1#i64] := Slice.from_val _ _
  have h1v : (1#i64).val = 1 := by decide
  have hsumSq : mixer4_sumSq [1#i64] = 1 := by
    rw [mixer4_sumSq_singleton, h1v]; norm_num
  have hisq : mixer4_isqrtRef 1 = 1 :=
    (mixer4_isqrt_unique (by norm_num) (by norm_num) (by norm_num)).symm
  have hnorm : mixer4_normRef 1 = 1 := by
    rw [mixer4_normRef, mixer4_roundRef, hisq]; norm_num
  have hden : mixer4_denRef 1 1 = 1 := by
    rw [mixer4_denRef, hnorm]; norm_num
  have hdiv : mixer4_divRef 1 1 = 4294967296 := by
    rw [mixer4_divRef, mixer2_divQ32Ref, if_pos (by norm_num)]; norm_num
  have hsqx : mixer4_sumSq (Slice.from [1#i64] mixer4_one_le_usize_max).val = 1 := by
    rw [hxval, hsumSq]
  obtain ⟨out, hout, houtv⟩ := word_l2_normalize_ok (Slice.from [1#i64] mixer4_one_le_usize_max) 1#i64
    (by rw [hsqx, mixer2_i128_max_eq]; simp only [mixer2_i128Max]; norm_num)
    (by rw [hsqx])
    (by rw [hsqx]; norm_num)
    (by rw [hxval]; simpa using mixer4_one_lt_usize_max)
    (by
      intro j hj
      rw [hxval] at hj
      have hj0 : j = 0 := by simp only [List.length_cons, List.length_nil] at hj; omega
      subst hj0
      simp only [hxval, List.getElem_cons_zero, hsumSq, h1v, hden, hdiv, I64.min_eq, I64.max_eq]
      norm_num)
  refine ⟨out, hout, ?_⟩
  rw [houtv, hxval, hsumSq, h1v, hden, mixer4_expected]
  simp only [List.map_cons, List.map_nil]
  rw [mixer4_outRef_val 1 1#i64
      (by rw [h1v, hdiv, I64.min_eq]; norm_num)
      (by rw [h1v, hdiv, I64.max_eq]; norm_num),
    h1v, hdiv]

end apeiron_word
