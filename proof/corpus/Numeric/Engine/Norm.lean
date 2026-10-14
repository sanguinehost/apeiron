/-
Refinement of the extracted fixed-point `RMSNorm` kernel (
plan the model-algebra note , gap G3).

The Rust body is `src/engine/norm.rs`; `i64` Q32.32. The public entry point is
`apeiron_word.word_rms_norm_fixed`, the extracted wrapper over `norm.rms_norm_fixed`.
The documented specification (`norm.rs:23-44`) is

* an empty row is `Err EmptyRow`;
* otherwise the mean square is the single round-to-nearest
  `mean = (sum x_i^2 + (n·2^32)/2) / (n·2^32)` in Q32.32, accumulated in `i128`;
* `arg = mean + eps_fixed` must lie in the `fixed_rsqrt` window `[2^-20, 2^20]`;
* `root = fixed_rsqrt arg`, and each output is `round(x_i · root / 2^32)`;
* every `i128` accumulation is checked and every output is narrowed only after a range check;
  a refusal, an out-of-window `arg` or an out-of-range output is `Err OutOfRange`.

`Numeric.Engine.Silu` (green, `89fb4a6e…`) is imported for the proved
`apeiron_word.div_round_nearest_abs_bound` rounding lemma this refinement composes with; the local
restatement is deleted.

This module lands the specification (`rms_norm_domain`, `rmsReal`, `rms_norm_refines`), the pure
rounding lemma, and the refusal equalities available by structural reduction of the extracted body
(`word_rms_norm_fixed_empty`, `word_rms_norm_fixed_unfold`, and the `sum_ok`/`arg`/`rsqrt`/`out_ok`
refusal arms).
-/
import Numeric.Engine.Silu
import Numeric.Engine.Rsqrt

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The documented mean square and the reference

`value.div_round_nearest num den = (num + den/2)/den` (half-biased, round-to-nearest) on `i128`. -/

/-- Half-biased integer quotient `(n + d/2)/d`, the value of `value.div_round_nearest n d` when the
division is exact and in range. -/
def divRoundNearest (n d : ℤ) : ℤ := (n + d / 2) / d

/-- The documented integer mean square in Q32.32:
`round(Σ x_i² / (n·2^32))` with one round-to-nearest, as an exact `ℤ` expression. -/
def rmsMean (row : Slice I64) : ℤ :=
  divRoundNearest ((row.val.map (fun x => x.val * x.val)).sum)
    ((row.val.length : ℤ) * 2 ^ (32 : ℕ))

/-- Output-window condition, the case excluded from `rms_norm_domain` that would otherwise make
`loop1` refuse: for the `fixed_rsqrt` result `r` of `a`, every coordinate's epilogue
`divRoundNearest (x_i · r) 2^32` fits `i64`. -/
def rms_output_window (row : Slice I64) (a : I64) : Prop :=
  ∀ (r : I64), word_fixed_rsqrt a = ok (core.result.Result.Ok r) →
    ∀ (i : ℕ) (hi : i < row.val.length),
      -((2 : ℤ) ^ (63 : ℕ)) ≤ divRoundNearest (row.val[i].val * r.val) ((2 : ℤ) ^ (32 : ℕ)) ∧
      divRoundNearest (row.val[i].val * r.val) ((2 : ℤ) ^ (32 : ℕ)) ≤ (2 : ℤ) ^ (63 : ℕ) - 1

/-- The checked-arithmetic window of the mean computation: `Σxᵢ² + n·2³¹ ≤ 2¹²⁷−1`, the
intermediate `num + den/2` of `value.div_round_nearest` (`num = Σxᵢ²`, `den = n·2³²`). Without it
that extracted checked add fails and the body returns a `Result` failure rather than `Ok`. The
denominator itself always fits: `n = row.val.length < 2⁶⁴`, so `n·2³² < 2⁹⁶ < 2¹²⁷−1`. -/
def rms_arith_window (row : Slice I64) : Prop :=
  (row.val.map (fun x => x.val * x.val)).sum + (row.val.length : ℤ) * (2:ℤ)^31 ≤ (2:ℤ)^127 - 1

/-- The documented domain. `a` is the Q32.32 `fixed_rsqrt` argument actually computed by the body:
`a.val = rmsMean row + eps.val`, so `rsqrt_window a` is exactly `arg ∈ [2^-20, 2^20]`. The two
`i128` bounds are the accumulated sum-of-squares range check of `loop0`; `rms_arith_window` is the
`value.div_round_nearest` intermediate; `rms_output_window` is the `i64` epilogue window that keeps
`loop1`'s `out_ok` true (its failure is `word_rms_norm_fixed_out_refuses`). -/
def rms_norm_domain (row : Slice I64) (eps a : I64) : Prop :=
  row.val ≠ [] ∧
  a.val = rmsMean row + eps.val ∧
  rsqrt_window a ∧
  rms_output_window row a ∧
  rms_arith_window row ∧
  -((2 : ℤ) ^ (127 : ℕ)) ≤ (row.val.map (fun x => x.val * x.val)).sum ∧
  (row.val.map (fun x => x.val * x.val)).sum ≤ (2 : ℤ) ^ (127 : ℕ) - 1

/-- The real reference `y = x · rsqrt(toReal a)` for a coordinate `x` and the Q32.32 `fixed_rsqrt`
argument `a` (`toReal` is the signed Q32.32 denotation from `Numeric.Scale`). -/
def rmsReal (a x : I64) : ℝ := toReal x * (1 / Real.sqrt (toReal a))

/-! ## Refusal equalities

`word_rms_norm_fixed` reduces definitionally to the extracted `norm.rms_norm_fixed`; the branches
below are the extracted refusal arms, proved as equalities. -/

/-- **Empty row.** `row.val = []` returns `Err EmptyRow` exactly. -/
theorem word_rms_norm_fixed_empty (row : Slice I64) (eps : I64) (h : row.val = []) :
    word_rms_norm_fixed row eps = ok (core.result.Result.Err norm.NormError.EmptyRow) := by
  have hn : Slice.len row = 0#usize := by
    apply UScalar.eq_of_val_eq
    simp [Slice.len, h, UScalar.ofNatCore_val_eq]
  rw [word_rms_norm_fixed, norm.rms_norm_fixed, hn]
  simp

/-- Nonempty rows do not take the empty branch: `Slice.len row ≠ 0`. -/
theorem slice_len_ne_zero_of_ne_nil (row : Slice I64) (h : row.val ≠ []) :
    Slice.len row ≠ 0#usize := by
  intro hc
  apply h
  have hv : row.val.length = 0 := by
    have := congrArg UScalar.val hc
    simpa [Slice.len, UScalar.ofNatCore_val_eq] using this
  exact List.length_eq_zero_iff.mp hv

/-- **`i128` accumulation refusal.** If `loop0` stops with the failure flag set, the body returns
`Err OutOfRange` exactly. -/
theorem word_rms_norm_fixed_sum_refuses (row : Slice I64) (eps : I64) (sum_sq : I128)
    (h : row.val ≠ [])
    (hloop : norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, false)) :
    word_rms_norm_fixed row eps = ok (core.result.Result.Err norm.NormError.OutOfRange) := by
  have hn := slice_len_ne_zero_of_ne_nil row h
  rw [word_rms_norm_fixed, norm.rms_norm_fixed]
  simp only [hn, if_false]
  rw [hloop]
  simp

/-- The extracted body after the empty-row test and a successful `loop0`, as a `do` block. The
remaining refusal arms of `norm.rs` are read off this block. -/
theorem word_rms_norm_fixed_unfold (row : Slice I64) (eps : I64) (sum_sq : I128)
    (h : row.val ≠ [])
    (hloop : norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true)) :
    word_rms_norm_fixed row eps =
      (do
        let n_i128 ← lift (UScalar.hcast .I128 (Slice.len row))
        let i ← 1#i128 <<< engine.FIXED_FRAC_BITS
        let o ← lift (I128.checked_mul n_i128 i)
        match o with
        | none => ok (core.result.Result.Err norm.NormError.OutOfRange)
        | some den =>
          let mean ← value.div_round_nearest sum_sq den
          let i1 ← lift (core.convert.num.FromI128I64.from eps)
          let o1 ← lift (mean.checked_add i1)
          match o1 with
          | none => ok (core.result.Result.Err norm.NormError.OutOfRange)
          | some arg =>
            let i2 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if arg > i2
            then ok (core.result.Result.Err norm.NormError.OutOfRange)
            else
              let i3 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if arg < i3
              then ok (core.result.Result.Err norm.NormError.OutOfRange)
              else
                let arg_i64 ← lift (IScalar.cast .I64 arg)
                let r ← fixed_rsqrt.fixed_rsqrt arg_i64
                match r with
                | core.result.Result.Ok root =>
                  let root1 ← lift (core.convert.num.FromI128I64.from root)
                  let out := alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)
                  let (out1, out_ok) ←
                    norm.rms_norm_fixed_loop1 row (Slice.len row) root1 out 0#usize true
                  if out_ok
                  then ok (core.result.Result.Ok out1)
                  else ok (core.result.Result.Err norm.NormError.OutOfRange)
                | core.result.Result.Err _ =>
                  ok (core.result.Result.Err norm.NormError.OutOfRange)) := by
  have hn := slice_len_ne_zero_of_ne_nil row h
  rw [word_rms_norm_fixed, norm.rms_norm_fixed]
  simp only [hn, if_false]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- The target refinement statement. On `rms_norm_domain` the extracted body returns the length-`n`
vector whose `i`-th Q32.32 value is within the composed bound of the real reference `rmsReal a x_i`:
the `fixed_rsqrt` relative/absolute error scaled by the coordinate plus the `1/(2·2^32)` epilogue
rounding. -/
def rms_norm_refines : Prop :=
  ∀ (row : Slice I64) (eps a : I64),
    rms_norm_domain row eps a →
      ∃ out : alloc.vec.Vec I64,
        word_rms_norm_fixed row eps = ok (core.result.Result.Ok out) ∧
        out.val.length = row.val.length ∧
        ∀ (i : ℕ) (hi : i < row.val.length),
          ∃ (hi' : i < out.val.length),
            |toReal (out.val[i]'hi') - rmsReal a (row.val[i]'hi)| ≤
              epsExp * |toReal row.val[i]| * (1 / Real.sqrt (toReal a))
                + (4 / q32) * |toReal row.val[i]| + 1 / (2 * q32)

/-- **Upper `i64` narrowing refusal.** If the computed `arg` exceeds `i64::MAX`, the body returns
`Err OutOfRange` exactly. -/
theorem word_rms_norm_fixed_arg_high_refuses (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128)
    (h : row.val ≠ [])
    (hloop : norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< engine.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hhigh : arg > IScalar.cast .I128 core.num.I64.MAX) :
    word_rms_norm_fixed row eps = ok (core.result.Result.Err norm.NormError.OutOfRange) := by
  rw [word_rms_norm_fixed_unfold row eps sum_sq h hloop]
  simp only [hshift, lift, bind_tc_ok, hden, hmean, harg, ↓reduceIte, hhigh]

/-- **Lower `i64` narrowing refusal.** If the computed `arg` is below `i64::MIN`, the body returns
`Err OutOfRange` exactly. -/
theorem word_rms_norm_fixed_arg_low_refuses (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128)
    (h : row.val ≠ [])
    (hloop : norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< engine.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX))
    (hlow : arg < IScalar.cast .I128 core.num.I64.MIN) :
    word_rms_norm_fixed row eps = ok (core.result.Result.Err norm.NormError.OutOfRange) := by
  rw [word_rms_norm_fixed_unfold row eps sum_sq h hloop]
  simp only [hshift, lift, bind_tc_ok, hden, hmean, harg, hmax, ↓reduceIte, hlow]

/-- **`fixed_rsqrt` refusal.** If the narrowed argument is outside the `fixed_rsqrt` window, the
body's `Err` arm returns `NormError::OutOfRange` exactly (the `From<RsqrtError>` conversion). -/
theorem word_rms_norm_fixed_rsqrt_refuses (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128) (e : fixed_rsqrt.RsqrtError)
    (h : row.val ≠ [])
    (hloop : norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< engine.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX))
    (hmin : ¬ (arg < IScalar.cast .I128 core.num.I64.MIN))
    (hroot : fixed_rsqrt.fixed_rsqrt (IScalar.cast .I64 arg) = ok (core.result.Result.Err e)) :
    word_rms_norm_fixed row eps = ok (core.result.Result.Err norm.NormError.OutOfRange) := by
  rw [word_rms_norm_fixed_unfold row eps sum_sq h hloop]
  simp only [hshift, lift, bind_tc_ok, hden, hmean, harg, hmax, hmin, hroot, ↓reduceIte]

/-- **Output-window refusal.** The case excluded by `rms_output_window`: if the epilogue loop stops
with `out_ok = false` (a coordinate's `i64` narrowing fails), the body returns `Err OutOfRange`
exactly. -/
theorem word_rms_norm_fixed_out_refuses (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128) (root : I64) (h : row.val ≠ [])
    (hloop : norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< engine.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX))
    (hmin : ¬ (arg < IScalar.cast .I128 core.num.I64.MIN))
    (hroot : fixed_rsqrt.fixed_rsqrt (IScalar.cast .I64 arg) = ok (core.result.Result.Ok root))
    (out1 : alloc.vec.Vec I64)
    (hloop1 : norm.rms_norm_fixed_loop1 row (Slice.len row)
        (core.convert.num.FromI128I64.from root)
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true
        = ok (out1, false)) :
    word_rms_norm_fixed row eps = ok (core.result.Result.Err norm.NormError.OutOfRange) := by
  rw [word_rms_norm_fixed_unfold row eps sum_sq h hloop]
  simp only [hshift, lift, bind_tc_ok, hden, hmean, harg, hmax, hmin, hroot, hloop1, ↓reduceIte]
  rfl


/-! ## The `value.round_shift_32` epilogue bridge

`value.round_shift_32 v = (v + 2^31) >>> 32`; the BitVec arithmetic shift `>>>` is `Int` floor
division, so the epilogue value is exactly `divRoundNearest v.val 2^32` and
`div_round_nearest_abs_bound` applies. -/

/-- The extracted `x >>> 32#i32` is `Int` floor division by `2^32`. -/
theorem I128_shiftRight_32_val (x : I128) :
    ∃ y : I128, (x >>> 32#i32) = ok y ∧ y.val = x.val / 2^32 := by
  refine ⟨⟨x.bv.sshiftRight 32⟩, ?_, ?_⟩
  · show IScalar.shiftRight x (32#i32).toNat = ok _
    have h32 : (32#i32).toNat = 32 := by decide
    rw [h32]
    unfold IScalar.shiftRight
    rw [if_pos (by norm_num [IScalarTy.numBits, I128.numBits])]
  · show (⟨x.bv.sshiftRight 32⟩ : I128).val = x.val / 2^32
    rw [show (⟨x.bv.sshiftRight 32⟩ : I128).val = (x.bv.sshiftRight 32).toInt from rfl]
    rw [BitVec.toInt_sshiftRight, Int.shiftRight_eq_div_pow]
    norm_num

/-- The extracted constant `1#i128 <<< 31#i32 = 2^31`. -/
theorem I128_shiftLeft_31 : (1#i128 <<< 31#i32) = ok (2147483648#i128) := by
  have h : (1#i128 <<< 31#i32) ⦃ z => z = (2147483648#i128) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted constant `1#i128 <<< engine.FIXED_FRAC_BITS = 2^32`. -/
theorem I128_shiftLeft_32 : (1#i128 <<< engine.FIXED_FRAC_BITS) = ok (4294967296#i128) := by
  have h : (1#i128 <<< engine.FIXED_FRAC_BITS) ⦃ z => z = (4294967296#i128) ⦄ := by
    unfold engine.FIXED_FRAC_BITS
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- **Epilogue semantics.** `value.round_shift_32 v` is the half-biased rounding
`divRoundNearest v.val 2^32`, when `v + 2^31` fits `i128`. -/
theorem round_shift_32_spec (v : I128)
    (hlo : I128.min ≤ v.val + 2^31) (hhi : v.val + 2^31 ≤ I128.max) :
    ∃ y : I128, value.round_shift_32 v = ok y ∧ y.val = divRoundNearest v.val ((2:ℤ)^32) := by
  unfold value.round_shift_32
  rw [I128_shiftLeft_31]
  simp only [bind_tc_ok]
  have h31 : (2147483648#i128).val = (2:ℤ)^31 := by norm_num
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (I128.add_spec (x := v) (y := 2147483648#i128)
      (by rw [h31]; exact hlo) (by rw [h31]; exact hhi))
  rw [hi1]
  simp only [bind_tc_ok]
  obtain ⟨y, hy, hyv⟩ := I128_shiftRight_32_val i1
  refine ⟨y, hy, ?_⟩
  rw [hyv, hi1v, h31]
  unfold divRoundNearest
  norm_num

def sqPrefix (row : Slice I64) (k : Nat) : ℤ :=
  ((row.val.take k).map (fun x => x.val * x.val)).sum

theorem map_sq_sum_nonneg (l : List I64) :
    0 ≤ (l.map (fun x => x.val * x.val)).sum := by
  induction l with
  | nil => simp
  | cons x xs ih =>
      simp only [List.map_cons, List.sum_cons]
      exact add_nonneg (mul_self_nonneg x.val) ih

theorem sqPrefix_nonneg (row : Slice I64) (k : Nat) : 0 ≤ sqPrefix row k := by
  unfold sqPrefix; exact map_sq_sum_nonneg _

theorem sqPrefix_total (row : Slice I64) :
    sqPrefix row row.val.length = (row.val.map (fun x => x.val * x.val)).sum := by
  unfold sqPrefix; rw [List.take_length]

theorem sqPrefix_le_total (row : Slice I64) (k : Nat) :
    sqPrefix row k ≤ (row.val.map (fun x => x.val * x.val)).sum := by
  unfold sqPrefix
  conv_rhs => rw [← List.take_append_drop k row.val]
  rw [List.map_append, List.sum_append]
  have := map_sq_sum_nonneg (row.val.drop k)
  linarith

theorem sqPrefix_succ (row : Slice I64) {k : Nat} (hk : k < row.val.length) :
    sqPrefix row (k + 1) = sqPrefix row k + row.val[k].val * row.val[k].val := by
  unfold sqPrefix
  rw [List.take_add_one, List.getElem?_eq_getElem hk]
  simp only [List.map_append, List.sum_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Option.toList_some]
  ring

theorem I128_checked_add_some (x y : I128)
    (hlo : I128.min ≤ x.val + y.val) (hhi : x.val + y.val ≤ I128.max) :
    ∃ z, x.checked_add y = some z ∧ z.val = x.val + y.val := by
  have hspec := I128.checked_add_bv_spec x y
  cases hc : core.num.checked_add_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I128.checked_add] using hc, hspec.2.2.1⟩

def loop0Inv (row : Slice I64) (n : Usize) (s : I128 × Usize × Bool) : Prop :=
  s.2.2 = true ∧ s.2.1.val ≤ n.val ∧ s.1.val = sqPrefix row s.2.1.val

theorem loop0_body_step (row : Slice I64) (n : Usize) (sum_sq : I128) (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = row.val.length)
    (hsum : sum_sq.val = sqPrefix row i.val)
    (hfit : (row.val.map (fun x => x.val * x.val)).sum ≤ I128.max) :
    ∃ total i2, norm.rms_norm_fixed_loop0.body row n sum_sq i true
        = ok (ControlFlow.cont (total, i2, true)) ∧
      total.val = sqPrefix row i2.val ∧ i2.val = i.val + 1 := by
  have hbnd : i.val < row.val.length := by omega
  have hI128min : I128.min = -(2:ℤ)^127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2:ℤ)^127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec row i hbnd)
  have hxval : i1.val = row.val[i.val].val := by rw [hi1v]
  have hi1b := i1.hBounds
  rw [IScalarTy.I64_numBits_eq] at hi1b
  have hvv : (core.convert.num.FromI128I64.from i1).val = i1.val :=
    core.convert.num.FromI128I64.from_val_eq i1
  have hmullo : I128.min ≤ (core.convert.num.FromI128I64.from i1).val *
      (core.convert.num.FromI128I64.from i1).val := by
    rw [hvv, hI128min]; nlinarith [mul_self_nonneg i1.val]
  have hmulhi : (core.convert.num.FromI128I64.from i1).val *
      (core.convert.num.FromI128I64.from i1).val ≤ I128.max := by
    have hb1 : -(2:ℤ)^63 ≤ i1.val := by have h := hi1b.1; norm_num at h; linarith
    have hb2 : i1.val < (2:ℤ)^63 := by have h := hi1b.2; norm_num at h; exact h
    rw [hvv, hI128max]
    nlinarith [hb1, hb2, sq_nonneg i1.val,
      sq_nonneg (i1.val - (2:ℤ)^63), sq_nonneg (i1.val + (2:ℤ)^63)]
  obtain ⟨sq, hsq, hsqv⟩ := I128_checked_mul_some _ _ hmullo hmulhi
  have hsumnn : 0 ≤ sum_sq.val := by rw [hsum]; exact sqPrefix_nonneg row i.val
  have hsuc : sum_sq.val + sq.val = sqPrefix row (i.val + 1) := by
    rw [hsqv, hsum, sqPrefix_succ row hbnd]
    rw [hvv, hxval]
  have hsumhi : sum_sq.val + sq.val ≤ I128.max := by
    rw [hsuc]; exact le_trans (sqPrefix_le_total row (i.val + 1)) hfit
  obtain ⟨total, htot, htotv⟩ := I128_checked_add_some sum_sq sq
    (by rw [hI128min]; nlinarith [hsumnn, sq_nonneg sq.val]) hsumhi
  have hiaddmax : i.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hb := row.property
    rw [h1]; omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hiaddmax)
  refine ⟨total, i2, ?_, ?_, hi2v⟩
  · rw [norm.rms_norm_fixed_loop0.body, hi1]
    simp only [lift, bind_tc_ok]
    rw [hsq]
    dsimp only
    rw [htot]
    dsimp only
    rw [hi2]
    simp only [bind_tc_ok]
    have hilt' : i < n := (UScalar.lt_equiv i n).2 hilt
    rw [if_pos hilt', if_pos trivial]
    rfl
  · rw [htotv, hsuc]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hi2v, h1]


theorem rms_norm_fixed_loop0_spec (row : Slice I64)
    (hfit : (row.val.map (fun x => x.val * x.val)).sum ≤ I128.max) :
    norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true ⦃ b =>
      ∃ S : I128, b = (S, true) ∧ S.val = (row.val.map (fun x => x.val * x.val)).sum ⦄ := by
  unfold norm.rms_norm_fixed_loop0
  apply loop.spec_decr_nat
    (measure := fun s : I128 × Usize × Bool => (Slice.len row).val - s.2.1.val)
    (inv := loop0Inv row (Slice.len row))
    (post := fun b : I128 × Bool =>
      ∃ S : I128, b = (S, true) ∧ S.val = (row.val.map (fun x => x.val * x.val)).sum)
  · intro s hs
    rcases s with ⟨sum_sq, i, sum_ok⟩
    dsimp only at hs ⊢
    obtain ⟨hok, hile, hsum⟩ := hs
    dsimp only at hok hile hsum
    rw [hok]
    by_cases hlt : i.val < (Slice.len row).val
    · obtain ⟨total, i2, hbody, htot, hi2⟩ :=
        loop0_body_step row (Slice.len row) sum_sq i hlt (by rw [Slice.len_val]) hsum hfit
      rw [hbody]
      have hi2le : i2.val ≤ (Slice.len row).val := by omega
      refine ⟨⟨rfl, hi2le, htot⟩, ?_⟩
      show (Slice.len row).val - i2.val < (Slice.len row).val - i.val
      omega
    · have hnot : ¬ (i < Slice.len row) := by
        intro hc; exact hlt ((UScalar.lt_equiv i (Slice.len row)).1 hc)
      have hie : i.val = (Slice.len row).val := by omega
      have hlen : (Slice.len row).val = row.val.length := Slice.len_val row
      have hbody : norm.rms_norm_fixed_loop0.body row (Slice.len row) sum_sq i true
          = ok (ControlFlow.done (sum_sq, true)) := by
        rw [norm.rms_norm_fixed_loop0.body, if_neg hnot]
      rw [hbody]
      exact ⟨sum_sq, rfl, by rw [hsum, hie, hlen, sqPrefix_total]⟩
  · refine ⟨rfl, ?_, ?_⟩
    · simp
    · show (0#i128).val = sqPrefix row (0#usize).val
      rw [sqPrefix]
      simp

theorem rms_norm_fixed_loop0_eq (row : Slice I64)
    (hfit : (row.val.map (fun x => x.val * x.val)).sum ≤ I128.max) :
    ∃ S : I128,
      S.val = (row.val.map (fun x => x.val * x.val)).sum ∧
      norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true = ok (S, true) := by
  obtain ⟨y, hy, S, hyS, hS⟩ := WP.spec_imp_exists (rms_norm_fixed_loop0_spec row hfit)
  exact ⟨S, hS, by rw [hy, hyS]⟩


def epilogue (root : I128) (x : I64) : ℤ :=
  divRoundNearest (x.val * root.val) ((2:ℤ)^32)

def loop1Inv (row : Slice I64) (n : Usize) (root : I128)
    (s : alloc.vec.Vec I64 × Usize × Bool) : Prop :=
  s.2.2 = true ∧ s.2.1.val ≤ n.val ∧
  s.1.val.map (fun y => y.val) = (row.val.take s.2.1.val).map (epilogue root)

theorem rms_norm_fixed_loop1_body_step (row : Slice I64) (n : Usize) (root : I128)
    (out : alloc.vec.Vec I64) (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = row.val.length)
    (hout : out.val.map (fun y => y.val) = (row.val.take i.val).map (epilogue root))
    (hrootb : -((2:ℤ)^61) ≤ root.val ∧ root.val ≤ (2:ℤ)^61)
    (hprod : ∀ (j : ℕ) (hj : j < row.val.length),
        I128.min ≤ row.val[j].val * root.val ∧ row.val[j].val * root.val ≤ I128.max)
    (hfit : ∀ (j : ℕ) (hj : j < row.val.length),
        -((2:ℤ)^63) ≤ epilogue root row.val[j] ∧ epilogue root row.val[j] ≤ (2:ℤ)^63 - 1) :
    ∃ out2 i2, norm.rms_norm_fixed_loop1.body row n root out i true
        = ok (ControlFlow.cont (out2, i2, true)) ∧
      out2.val.map (fun y => y.val) = (row.val.take i2.val).map (epilogue root) ∧
      i2.val = i.val + 1 := by
  have hbnd : i.val < row.val.length := by omega
  have hI128min : I128.min = -(2:ℤ)^127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2:ℤ)^127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec row i hbnd)
  have hxval : i1.val = row.val[i.val].val := by rw [hi1v]
  have hvv : (core.convert.num.FromI128I64.from i1).val = i1.val :=
    core.convert.num.FromI128I64.from_val_eq i1
  have hpr := hprod i.val hbnd
  have hmullo : I128.min ≤ (core.convert.num.FromI128I64.from i1).val * root.val := by
    rw [hvv, hxval]; exact hpr.1
  have hmulhi : (core.convert.num.FromI128I64.from i1).val * root.val ≤ I128.max := by
    rw [hvv, hxval]; exact hpr.2
  obtain ⟨prod, hprod_eq, hprodv⟩ := I128_checked_mul_some _ _ hmullo hmulhi
  have hfitv := hfit i.val hbnd
  have hprodval : prod.val = row.val[i.val].val * root.val := by rw [hprodv, hvv, hxval]
  have hxb := (row.val[i.val]).hBounds
  rw [IScalarTy.I64_numBits_eq] at hxb
  norm_num at hxb
  have hlo31 : I128.min ≤ prod.val + 2^31 := by
    rw [hI128min, hprodval]
    nlinarith [hxb.1, hxb.2, hrootb.1, hrootb.2]
  have hhi31 : prod.val + 2^31 ≤ I128.max := by
    rw [hI128max, hprodval]
    nlinarith [hxb.1, hxb.2, hrootb.1, hrootb.2]
  obtain ⟨normed, hnormed_eq, hnormedv⟩ := round_shift_32_spec prod hlo31 hhi31
  have hnormed_epi : normed.val = epilogue root row.val[i.val] := by
    rw [hnormedv, hprodval]; rfl
  have hfitlo : -((2:ℤ)^63) ≤ normed.val := by rw [hnormed_epi]; exact hfitv.1
  have hfithi : normed.val ≤ (2:ℤ)^63 - 1 := by rw [hnormed_epi]; exact hfitv.2
  -- narrow checks
  have hmaxv : (IScalar.cast .I128 core.num.I64.MAX).val = (2:ℤ)^63 - 1 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MAX.val = (2:ℤ)^63 - 1 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2:ℤ)^63 - 1)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hminv : (IScalar.cast .I128 core.num.I64.MIN).val = -(2:ℤ)^63 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MIN.val = -(2:ℤ)^63 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2:ℤ)^63)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hnotmax : ¬ (normed > IScalar.cast .I128 core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv] at hc
    omega
  have hnotmin : ¬ (normed < IScalar.cast .I128 core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, hminv] at hc
    omega
  have hcastval : (IScalar.cast .I64 normed).val = normed.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 normed.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  have hpushmax : out.val.length < Usize.max := by
    have hlen : out.val.length = i.val := by
      have h := congrArg List.length hout
      simp only [List.length_map, List.length_take] at h
      rw [Nat.min_eq_left (le_of_lt hbnd)] at h
      exact h
    have hb := row.property
    rw [← hn] at hb
    omega
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out (IScalar.cast .I64 normed) hpushmax)
  have hiaddmax : i.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hb := row.property
    rw [h1]; omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hiaddmax)
  refine ⟨out2, i2, ?_, ?_, ?_⟩
  · rw [norm.rms_norm_fixed_loop1.body, hi1]
    simp only [lift, bind_tc_ok]
    rw [hprod_eq]
    dsimp only
    rw [hnormed_eq]
    simp only [bind_tc_ok]
    rw [if_neg hnotmax, if_neg hnotmin]
    rw [hpush, hi2]
    simp only [bind_tc_ok]
    have hilt' : i < n := (UScalar.lt_equiv i n).2 hilt
    rw [if_pos hilt', if_pos trivial]
    rfl
  · rw [hpushv, List.map_append, hout]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hi2v, h1]
    have hsucc : (row.val.take (i.val + 1)).map (epilogue root) =
        (row.val.take i.val).map (epilogue root) ++ [epilogue root row.val[i.val]] := by
      rw [List.take_add_one, List.getElem?_eq_getElem hbnd]
      simp only [List.map_append, List.map_cons, List.map_nil, Option.toList_some]
    rw [hsucc]
    simp only [List.map_cons, List.map_nil]
    congr 1
    rw [hcastval, hnormed_epi]
  · rw [hi2v]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]


theorem rms_norm_fixed_loop1_spec (row : Slice I64) (root : I128)
    (hrootb : -((2:ℤ)^61) ≤ root.val ∧ root.val ≤ (2:ℤ)^61)
    (hprod : ∀ (j : ℕ) (hj : j < row.val.length),
        I128.min ≤ row.val[j].val * root.val ∧ row.val[j].val * root.val ≤ I128.max)
    (houtfit : ∀ (j : ℕ) (hj : j < row.val.length),
        -((2:ℤ)^63) ≤ epilogue root row.val[j] ∧ epilogue root row.val[j] ≤ (2:ℤ)^63 - 1) :
    norm.rms_norm_fixed_loop1 row (Slice.len row) root
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true ⦃ b =>
      ∃ out : alloc.vec.Vec I64, b = (out, true) ∧
        out.val.map (fun y => y.val) = (row.val.map (epilogue root)) ⦄ := by
  unfold norm.rms_norm_fixed_loop1
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool => (Slice.len row).val - s.2.1.val)
    (inv := loop1Inv row (Slice.len row) root)
    (post := fun b : alloc.vec.Vec I64 × Bool =>
      ∃ out : alloc.vec.Vec I64, b = (out, true) ∧
        out.val.map (fun y => y.val) = (row.val.map (epilogue root)))
  · intro s hs
    rcases s with ⟨out, i, out_ok⟩
    dsimp only at hs ⊢
    obtain ⟨hok, hile, hout⟩ := hs
    dsimp only at hok hile hout
    rw [hok]
    by_cases hlt : i.val < (Slice.len row).val
    · obtain ⟨out2, i2, hbody, hout2, hi2⟩ :=
        rms_norm_fixed_loop1_body_step row (Slice.len row) root out i hlt
          (by rw [Slice.len_val]) hout hrootb hprod houtfit
      rw [hbody]
      have hi2le : i2.val ≤ (Slice.len row).val := by omega
      refine ⟨⟨rfl, hi2le, hout2⟩, ?_⟩
      show (Slice.len row).val - i2.val < (Slice.len row).val - i.val
      omega
    · have hnot : ¬ (i < Slice.len row) := by
        intro hc; exact hlt ((UScalar.lt_equiv i (Slice.len row)).1 hc)
      have hie : i.val = (Slice.len row).val := by omega
      have hlen : (Slice.len row).val = row.val.length := Slice.len_val row
      have hbody : norm.rms_norm_fixed_loop1.body row (Slice.len row) root out i true
          = ok (ControlFlow.done (out, true)) := by
        rw [norm.rms_norm_fixed_loop1.body, if_neg hnot]
      rw [hbody]
      exact ⟨out, rfl, by rw [hout, hie, hlen, List.take_length]⟩
  · refine ⟨rfl, ?_, ?_⟩
    · simp
    · show (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)).val.map (fun y => y.val) =
        (row.val.take (0#usize).val).map (epilogue root)
      simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val]

theorem rms_norm_fixed_loop1_eq (row : Slice I64) (root : I128)
    (hrootb : -((2:ℤ)^61) ≤ root.val ∧ root.val ≤ (2:ℤ)^61)
    (hprod : ∀ (j : ℕ) (hj : j < row.val.length),
        I128.min ≤ row.val[j].val * root.val ∧ row.val[j].val * root.val ≤ I128.max)
    (houtfit : ∀ (j : ℕ) (hj : j < row.val.length),
        -((2:ℤ)^63) ≤ epilogue root row.val[j] ∧ epilogue root row.val[j] ≤ (2:ℤ)^63 - 1) :
    ∃ out : alloc.vec.Vec I64,
      out.val.map (fun y => y.val) = (row.val.map (epilogue root)) ∧
      norm.rms_norm_fixed_loop1 row (Slice.len row) root
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true = ok (out, true) := by
  obtain ⟨y, hy, out, hyS, hS⟩ :=
    WP.spec_imp_exists (rms_norm_fixed_loop1_spec row root hrootb hprod houtfit)
  exact ⟨out, hS, by rw [hy, hyS]⟩


theorem div_round_nearest_spec (num den : I128) (hden : 0 < den.val) (hnum : 0 ≤ num.val)
    (hadd : num.val + den.val / 2 ≤ I128.max) :
    value.div_round_nearest num den ⦃ z => z.val = divRoundNearest num.val den.val ⦄ := by
  have hI128min : I128.min = -(2:ℤ)^127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2:ℤ)^127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  unfold value.div_round_nearest
  refine WP.spec_bind (I128.div_spec (x := den) (y := 2#i128)
    (by norm_num) (by intro h; exact absurd h.2 (by norm_num))) ?_
  intro i hi_val
  have hi_val' : i.val = den.val / 2 := by
    rw [hi_val]
    exact Int.tdiv_eq_ediv_of_nonneg (le_of_lt hden)
  have hinn : 0 ≤ i.val := by rw [hi_val']; exact Int.ediv_nonneg (le_of_lt hden) (by norm_num)
  refine WP.spec_bind (I128.add_spec (x := num) (y := i)
    (by rw [hI128min]; linarith [hnum, hinn])
    (by rw [hi_val']; simpa [hI128max] using hadd)) ?_
  intro i1 hi1_val
  refine WP.spec_mono (I128.div_spec (x := i1) (y := den)
    (by omega) (by intro h; exact absurd h.2 (by omega))) ?_
  intro z hz_val
  rw [hz_val, hi1_val, hi_val']
  unfold divRoundNearest
  exact Int.tdiv_eq_ediv_of_nonneg (by positivity)

theorem rmsElement_bound (a r x y : I64)
    (hrsq : |toReal r - 1 / Real.sqrt (toReal a)| ≤
      epsExp * (1 / Real.sqrt (toReal a)) + 4 / q32)
    (hy : y.val = divRoundNearest (x.val * r.val) ((2:ℤ)^32)) :
    |toReal y - rmsReal a x| ≤
      epsExp * |toReal x| * (1 / Real.sqrt (toReal a))
        + (4 / q32) * |toReal x| + 1 / (2 * q32) := by
  set Z : ℤ := y.val * (2:ℤ)^32 - x.val * r.val with hZdef
  have hint : |Z| ≤ (2:ℤ)^31 := by
    have h := div_round_nearest_abs_bound (x.val * r.val) ((2:ℤ)^32) (by norm_num)
    rw [show (2:ℤ)^32 / 2 = (2:ℤ)^31 from by norm_num] at h
    rw [hZdef, hy]
    exact h
  have hround : |toReal y - toReal x * toReal r| ≤ 1 / (2 * q32) := by
    have hZid : toReal y - toReal x * toReal r = ((Z : ℤ) : ℝ) / ((2:ℝ)^64) := by
      rw [hZdef]
      unfold toReal q32
      push_cast
      ring
    have hZR : |((Z : ℤ) : ℝ)| ≤ (2:ℝ)^31 := by exact_mod_cast hint
    rw [hZid, abs_div, abs_of_pos (by positivity : (0:ℝ) < (2:ℝ)^64)]
    calc |((Z : ℤ) : ℝ)| / (2:ℝ)^64 ≤ (2:ℝ)^31 / (2:ℝ)^64 :=
          div_le_div_of_nonneg_right hZR (by positivity)
      _ = 1 / (2 * q32) := by rw [show q32 = (2:ℝ)^32 from rfl]; norm_num
  have hrel : |toReal x * toReal r - toReal x * (1 / Real.sqrt (toReal a))| ≤
      |toReal x| * (epsExp * (1 / Real.sqrt (toReal a)) + 4 / q32) := by
    rw [← mul_sub, abs_mul]
    exact mul_le_mul_of_nonneg_left hrsq (abs_nonneg _)
  have htri := abs_sub_le (toReal y) (toReal x * toReal r)
    (toReal x * (1 / Real.sqrt (toReal a)))
  have hmain : |toReal y - toReal x * (1 / Real.sqrt (toReal a))| ≤
      1 / (2 * q32) + |toReal x| * (epsExp * (1 / Real.sqrt (toReal a)) + 4 / q32) := by
    linarith [htri, hround, hrel]
  rw [rmsReal]
  calc |toReal y - toReal x * (1 / Real.sqrt (toReal a))|
      ≤ |toReal x| * (epsExp * (1 / Real.sqrt (toReal a)) + 4 / q32) + 1 / (2 * q32) := by
        linarith [hmain]
    _ = epsExp * |toReal x| * (1 / Real.sqrt (toReal a)) + (4 / q32) * |toReal x|
          + 1 / (2 * q32) := by ring


theorem rms_norm_refines_proved : rms_norm_refines := by
  intro row eps a hdom
  obtain ⟨hne, ha, hwin, houtwin, harith, hsumlo, hsumhi⟩ := hdom
  have hI128min : I128.min = -(2:ℤ)^127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2:ℤ)^127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  obtain ⟨S, hSval, hloop0⟩ := rms_norm_fixed_loop0_eq row (by rw [hI128max]; exact hsumhi)
  have hlen_lt : row.val.length < 2^64 := by
    have h := UScalar.hBounds (Slice.len row)
    rw [Slice.len_val] at h
    rw [UScalarTy.Usize_numBits_eq] at h
    rcases System.Platform.numBits_eq with h32 | h64
    · rw [h32] at h; norm_num at h; omega
    · rw [h64] at h; exact h
  have hcastval : (UScalar.hcast .I128 (Slice.len row)).val = (Slice.len row).val := by
    rw [UScalar.hcast_val_eq, Slice.len_val]
    apply Int.bmod_eq_of_le
    · have : (0:ℤ) ≤ row.val.length := by exact_mod_cast Nat.zero_le _
      norm_num
    · norm_num
      have hlen : row.val.length < 2^127 := by omega
      exact_mod_cast hlen
  have hq32val : (4294967296#i128).val = (2:ℤ)^32 := by norm_num
  have hlen_lt' : (row.val.length : ℤ) < 2^64 := by exact_mod_cast hlen_lt
  obtain ⟨den, hden, hdenv⟩ := I128_checked_mul_some
    (UScalar.hcast .I128 (Slice.len row)) (4294967296#i128)
    (by
      rw [hI128min, hcastval, Slice.len_val, hq32val]
      have h1 : (0:ℤ) ≤ row.val.length := by exact_mod_cast Nat.zero_le _
      nlinarith [h1, show (0:ℤ) < 2^32 from by norm_num])
    (by
      rw [hI128max, hcastval, Slice.len_val, hq32val]
      have h2 : (row.val.length : ℤ) * 2^32 < (2:ℤ)^96 := by nlinarith [hlen_lt']
      nlinarith [h2, show ((2:ℤ)^96) < 2^127 - 1 from by norm_num])
  have hdenpos : 0 < den.val := by
    rw [hdenv, hcastval, hq32val, Slice.len_val]
    have hpos : 0 < row.val.length := List.length_pos_iff.mpr hne
    exact mul_pos (by exact_mod_cast hpos) (by norm_num)
  have hdenhalf : den.val / 2 = (row.val.length : ℤ) * (2:ℤ)^31 := by
    rw [hdenv, hcastval, Slice.len_val, hq32val]
    rw [show ((row.val.length : ℤ) * 2^32) = ((row.val.length : ℤ) * 2^31) * 2 from by ring]
    exact Int.mul_ediv_cancel _ (by norm_num)
  have hadd : S.val + den.val / 2 ≤ I128.max := by
    rw [hSval, hdenhalf, hI128max]
    exact harith
  obtain ⟨mean, hmean, hmeanv⟩ :=
    WP.spec_imp_exists (div_round_nearest_spec S den hdenpos
      (by rw [hSval, ← sqPrefix_total]; exact sqPrefix_nonneg row row.val.length) hadd)
  have hmeanval : mean.val = rmsMean row := by
    rw [hmeanv, hSval, hdenv, hcastval, Slice.len_val, hq32val]
    unfold rmsMean divRoundNearest
    rfl
  have harglo : I128.min ≤ mean.val + (core.convert.num.FromI128I64.from eps).val := by
    rw [hI128min, hmeanval, core.convert.num.FromI128I64.from_val_eq, ← ha]
    have := a.hBounds; rw [IScalarTy.I64_numBits_eq] at this
    norm_num at this; linarith [this.1]
  have harghi : mean.val + (core.convert.num.FromI128I64.from eps).val ≤ I128.max := by
    rw [hI128max, hmeanval, core.convert.num.FromI128I64.from_val_eq, ← ha]
    have := a.hBounds; rw [IScalarTy.I64_numBits_eq] at this
    norm_num at this; linarith [this.2]
  obtain ⟨arg, harg, hargv⟩ := I128_checked_add_some mean (core.convert.num.FromI128I64.from eps) harglo harghi
  have hargval : arg.val = a.val := by
    rw [hargv, hmeanval, core.convert.num.FromI128I64.from_val_eq, ← ha]
  have hmaxv : (IScalar.cast .I128 core.num.I64.MAX).val = (2:ℤ)^63 - 1 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MAX.val = (2:ℤ)^63 - 1 := by decide
    rw [h]; norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2:ℤ)^63 - 1)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hminv : (IScalar.cast .I128 core.num.I64.MIN).val = -(2:ℤ)^63 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MIN.val = -(2:ℤ)^63 := by decide
    rw [h]; norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2:ℤ)^63)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hab : -(2:ℤ)^63 ≤ a.val ∧ a.val < (2:ℤ)^63 := by
    have := a.hBounds; rw [IScalarTy.I64_numBits_eq] at this; norm_num at this; omega
  have hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX) := by
    intro hc; rw [gt_iff_lt, IScalar.lt_equiv, hmaxv] at hc; rw [hargval] at hc; omega
  have hmin : ¬ (arg < IScalar.cast .I128 core.num.I64.MIN) := by
    intro hc; rw [IScalar.lt_equiv, hminv] at hc; rw [hargval] at hc; omega
  have hcasteq : IScalar.cast .I64 arg = a := by
    apply IScalar.eq_of_val_eq
    obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists
      (IScalar.cast_inBounds_spec .I64 arg (by
        rw [hargval]
        exact ⟨by rw [IScalar.min_IScalarTy_I64_eq, I64.min_eq]; exact hab.1,
               by rw [IScalar.max_IScalarTy_I64_eq, I64.max_eq]; omega⟩))
    have hz2 : IScalar.cast .I64 arg = z := by
      have hz' : ok (IScalar.cast .I64 arg) = ok z := by simpa only [lift] using hz
      rwa [Result.ok.injEq] at hz'
    rw [hz2, hzv, hargval]
  obtain ⟨r, hr, s0, hs1, hs2, hs3, hs4, hsv, hr0, hr61⟩ := word_fixed_rsqrt_success a hwin
  obtain ⟨r2, hr2, hrb2⟩ := fixed_rsqrt_refines_proved a hwin
  have hr_eq : r2 = r := by
    have hok : core.result.Result.Ok r = core.result.Result.Ok r2 :=
      Result.ok.injEq.mp (hr.symm.trans hr2)
    injection hok with h
    exact h.symm
  have hbound : |toReal r - 1 / Real.sqrt (toReal a)| ≤
      epsExp * (1 / Real.sqrt (toReal a)) + 4 / q32 := by rw [← hr_eq]; exact hrb2
  have hroot : fixed_rsqrt.fixed_rsqrt (IScalar.cast .I64 arg) = ok (core.result.Result.Ok r) := by
    rw [hcasteq]
    simpa [word_fixed_rsqrt] using hr
  have hrootb : -((2:ℤ)^61) ≤ (core.convert.num.FromI128I64.from r).val ∧
      (core.convert.num.FromI128I64.from r).val ≤ (2:ℤ)^61 := by
    rw [core.convert.num.FromI128I64.from_val_eq]; exact ⟨by linarith [hr0], hr61⟩
  have hprod : ∀ (j : ℕ) (hj : j < row.val.length),
      I128.min ≤ row.val[j].val * (core.convert.num.FromI128I64.from r).val ∧
        row.val[j].val * (core.convert.num.FromI128I64.from r).val ≤ I128.max := by
    intro j hj
    rw [core.convert.num.FromI128I64.from_val_eq, hI128min, hI128max]
    have hxb := (row.val[j]).hBounds; rw [IScalarTy.I64_numBits_eq] at hxb; norm_num at hxb
    have hx0 : -(2:ℤ)^63 ≤ (row.val[j]).val := by linarith [hxb.1]
    have hx1 : (row.val[j]).val ≤ 2^63 - 1 := by omega
    constructor
    · have h1 : (-(2:ℤ)^63) * r.val ≤ (row.val[j]).val * r.val :=
        mul_le_mul_of_nonneg_right hx0 hr0
      have h2 : -((2:ℤ)^127) ≤ (-(2:ℤ)^63) * r.val := by nlinarith [hr0, hr61]
      linarith
    · have h1 : (row.val[j]).val * r.val ≤ ((2:ℤ)^63 - 1) * r.val :=
        mul_le_mul_of_nonneg_right hx1 hr0
      have h2 : ((2:ℤ)^63 - 1) * r.val ≤ ((2:ℤ)^63 - 1) * 2^61 :=
        mul_le_mul_of_nonneg_left hr61 (by norm_num)
      have h3 : ((2:ℤ)^63 - 1) * 2^61 ≤ 2^127 - 1 := by norm_num
      linarith
  have houtfit : ∀ (j : ℕ) (hj : j < row.val.length),
      -((2:ℤ)^63) ≤ epilogue (core.convert.num.FromI128I64.from r) row.val[j] ∧
        epilogue (core.convert.num.FromI128I64.from r) row.val[j] ≤ (2:ℤ)^63 - 1 := by
    intro j hj
    have h := houtwin r hr j hj
    simpa only [epilogue, core.convert.num.FromI128I64.from_val_eq] using h
  obtain ⟨out, houtmap, hloop1⟩ :=
    rms_norm_fixed_loop1_eq row (core.convert.num.FromI128I64.from r) hrootb hprod houtfit
  have hlenout : out.val.length = row.val.length := by
    have h := congrArg List.length houtmap; simpa using h
  refine ⟨out, ?_, hlenout, ?_⟩
  · rw [word_rms_norm_fixed_unfold row eps S hne hloop0]
    rw [I128_shiftLeft_32]
    simp only [lift, bind_tc_ok, hden, hmean, harg, hmax, hmin, hroot, hloop1, ↓reduceIte]
    rfl
  · intro i hi
    have hi' : i < out.val.length := by rw [hlenout]; exact hi
    refine ⟨hi', ?_⟩
    have hentry : (out.val[i]'hi').val =
        epilogue (core.convert.num.FromI128I64.from r) (row.val[i]'hi) := by
      have h2 : (out.val.map (fun y => y.val))[i]? =
          (row.val.map (epilogue (core.convert.num.FromI128I64.from r)))[i]? := by rw [houtmap]
      simp only [List.getElem?_map, List.getElem?_eq_getElem hi, List.getElem?_eq_getElem hi',
        Option.map_some] at h2
      exact Option.some.inj h2
    have hy : (out.val[i]'hi').val = divRoundNearest ((row.val[i]'hi).val * r.val) ((2:ℤ)^32) := by
      rw [hentry]; unfold epilogue; rw [core.convert.num.FromI128I64.from_val_eq]
    exact rmsElement_bound a r (row.val[i]'hi) (out.val[i]'hi') hbound hy

end

end apeiron_word
