import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Numeric.Engine.Attn

/-!
# The extracted vocabulary head: KL and TV losses, their logit gradients, and the chunked head

`src/learn/grad/head.rs` holds the head of the ported one-bit transfer: softmax rows in Q32.32 with the
`fixed_exp` underflow flush, `KL(p_t ‖ p_s)` and `TV(p_s, p_t)` with their logit gradients formed in
the forward, and `head_chunked`, the chunked `W_head` backward. This module refines the extracted
bodies (`apeiron_word.learn_port.head.*`) against closed forms stated as exact integer (`ℤ`)
expressions.

## What is stated

The head's arithmetic splits in two.

* The **exact integer part**: the rounding (`round_div`), the sign and magnitude helpers, the KL
  accumulator `Σ p_t·(log p_t − log p_s)`, the gradient `p_s − p_t`, the TV scan
  `(Σ|p_s − p_t|, Σ p_s·sign)`, the TV gradient `½ p_s (s − Σ p_s s)`, the checked logit dot product,
  and the `d_hidden`/`d_w_head` accumulations. Each is a total function of its integer inputs; the
  theorems below state that the extracted body returns exactly that value, under the explicit
  no-overflow hypotheses that make every checked operation admit its result.
* The **engine part**: `fixed_exp` and `fixed_ln`, reached through `exp_entry` and `softmax_row`.
  Their refinement lives in `Numeric.Engine.Attn` (`fixed_exp_refines_proved` and the
  refusal/underflow lemmas) and is not restated here. The head-level theorems that consume a softmax
  row are stated **conditionally** on the extracted `softmax_row` returning the pair the caller
  consumes, so the head arithmetic is proved here and the engine bound stays where it is proved.

## Grade and boundary

Each theorem is `proved` (Lean kernel, cone `{propext, Classical.choice, Quot.sound}`). Not stated
here: the real-valued error bound of a softmax row or a loss, and the composition of `head_chunked`
through the engine calls of `row_objective`. The trust boundary is the Aeneas extraction and the
engine primitives' own layer.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_head

open apeiron_word
open Aeneas Aeneas.Std Result

/-! ## 1. List and Aeneas helpers -/

/-- Reading a list after one element is written. -/
theorem hl_getD_set {α : Type} (l : List α) (i j : ℕ) (v : α) (h : i < l.length) (d : α) :
    (l.set i v).getD j d = if i = j then v else l.getD j d := by
  by_cases heq : i = j
  · subst heq
    rw [List.getD_eq_getElem?_getD, List.getElem?_set_self h, Option.getD_some]
    simp
  · rw [List.getD_eq_getElem?_getD, List.getElem?_set, if_neg heq, ← List.getD_eq_getElem?_getD]
    simp [heq]

/-- `Slice.index_usize` reads the backing list at the in-range index. -/
theorem slice_index_getD {α : Type} [Inhabited α] (x : Slice α) (k : Usize)
    (h : k.val < x.val.length) :
    Slice.index_usize x k = ok (x.val.getD k.val default) := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]

/-- `Slice.update` writes the backing list at the in-range index. -/
theorem slice_update_val {α : Type} (sl sl' : Slice α) (i : Usize) (x : α)
    (h : Slice.update sl i x = ok sl') : sl'.val = sl.val.set i.val x := by
  unfold Slice.update at h
  split at h
  · simp at h
  · have := (Result.ok_injective h).symm
    subst this
    simp

/-- The backing list length is unchanged by `Slice.update`. -/
theorem slice_update_length {α : Type} (sl sl' : Slice α) (i : Usize) (x : α)
    (h : Slice.update sl i x = ok sl') : sl'.val.length = sl.val.length := by
  rw [slice_update_val sl sl' i x h, List.length_set]

/-- `i64` values lie in their type bounds. -/
theorem i64_bounds (x : I64) : I64.min ≤ x.val ∧ x.val ≤ I64.max := by
  have hb := x.hBounds
  have hbits : IScalarTy.I64.numBits = 64 := by decide
  rw [hbits] at hb
  constructor
  · norm_num [I64.min, IScalar.min, I64.numBits]
    omega
  · have : I64.max = (2 : ℤ) ^ (63 : ℕ) - 1 := by norm_num [I64.max, IScalar.max, I64.numBits]
    rw [this]
    norm_num at hb ⊢
    omega

/-- `i128` values lie in their type bounds. -/
theorem i128_bounds (x : I128) : I128.min ≤ x.val ∧ x.val ≤ I128.max := by
  have hb := x.hBounds
  have hbits : IScalarTy.I128.numBits = 128 := by decide
  rw [hbits] at hb
  have hmin : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hmax : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  rw [hmin, hmax]
  norm_num at hb ⊢
  exact ⟨hb.1, by omega⟩

/-- A checked `i128` multiply that fits returns its mathematical product. -/
theorem checked_mul_some_i128 {a b : I128}
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

/-- A checked `i128` add that fits returns its mathematical sum. -/
theorem checked_add_some_i128 {a b : I128}
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

/-- A checked `usize` multiply that fits returns its mathematical product. -/
theorem checked_mul_some_usize {a b : Usize}
    (h : a.val * b.val ≤ Usize.max) :
    ∃ z, Usize.checked_mul a b = some z ∧ z.val = a.val * b.val := by
  have hspec : (match Usize.checked_mul a b with
      | some z => a.val * b.val ≤ Usize.max ∧ z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => Usize.max < a.val * b.val) :=
    Usize.checked_mul_bv_spec a b
  cases hm : Usize.checked_mul a b with
  | none => rw [hm] at hspec; exact absurd h (by omega)
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.1⟩

/-- The `some` branch of `checked_add` carries the fitting bound. -/
theorem checked_add_bounds {a b s : I128} (hm : I128.checked_add a b = some s) :
    I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max := by
  have hspec : (match I128.checked_add a b with
      | some z => I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max
          ∧ z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max)) :=
    I128.checked_add_bv_spec a b
  rw [hm] at hspec
  exact ⟨hspec.1, hspec.2.1⟩

/-- The `some` branch of `checked_mul` carries the fitting bound. -/
theorem checked_mul_bounds {a b s : I128} (hm : I128.checked_mul a b = some s) :
    I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max := by
  have hspec : (match I128.checked_mul a b with
      | some z => I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max
          ∧ z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => ¬ (I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max)) :=
    I128.checked_mul_bv_spec a b
  rw [hm] at hspec
  exact ⟨hspec.1, hspec.2.1⟩

/-- The `some` branch of the `usize` `checked_mul` carries the fitting bound. -/
theorem checked_mul_usize_bounds {a b s : Usize} (hm : Usize.checked_mul a b = some s) :
    a.val * b.val ≤ Usize.max := by
  have hspec : (match Usize.checked_mul a b with
      | some z => a.val * b.val ≤ Usize.max ∧ z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => Usize.max < a.val * b.val) :=
    Usize.checked_mul_bv_spec a b
  rw [hm] at hspec
  exact hspec.1

/-! ## 2. The integer specifications

Stated on `ℤ` alone. `Int.tdiv`/`Int.tmod` are the truncated (toward zero) division and remainder the
Rust `/` and `%` on signed integers perform. -/

/-- `|x|` on `ℤ`. -/
def absZ (x : ℤ) : ℤ := if x < 0 then -x else x

/-- The rounding of `round_div`: `round n / d` to nearest, ties away from zero. -/
def rndDiv (n d : ℤ) : ℤ :=
  let q := Int.tdiv n d
  let r := Int.tmod n d
  if 2 * absZ r ≥ d then (if 0 ≤ n then q + 1 else q - 1) else q

/-- Sign of `x` as `-1`, `0`, `1`; the value of `sgn`. -/
def sgnZ (x : ℤ) : ℤ := if 0 < x then 1 else if x < 0 then -1 else 0

/-- Fold a list with `max` from a starting accumulator; the value of `row_max`'s loop. -/
def maxInit (l : List ℤ) (m : ℤ) : ℤ := l.foldl (fun a x => if x > a then x else a) m

/-- `kl_acc`'s accumulator after `n` terms: `Σ_{i<n} p_t[i]·(lp_t[i] − lp_s[i])`. -/
def klAccAux (p lp lp_s : List ℤ) (a : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => klAccAux p lp lp_s (a + p.getD n 0 * (lp.getD n 0 - lp_s.getD n 0)) n

/-- `dz_diff`'s written vector after `n` entries: `dz[i] = p_s[i] − p_t[i]` for `i < n`. -/
def dzVals (p_s p_t : List ℤ) (dz : List ℤ) : ℕ → List ℤ
  | 0 => dz
  | n + 1 => (dzVals p_s p_t dz n).set n (p_s.getD n 0 - p_t.getD n 0)

/-- `tv_scan`'s absolute-sum accumulator after `n` entries. -/
def tvAbsAux (p_s p_t : List ℤ) (a : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => tvAbsAux p_s p_t (a + absZ (p_s.getD n 0 - p_t.getD n 0)) n

/-- `tv_scan`'s signed accumulator `Σ p_s·sign(p_s − p_t)` after `n` entries. -/
def tvSignedAux (p_s p_t : List ℤ) (s : ℤ) : ℕ → ℤ
  | 0 => s
  | n + 1 => tvSignedAux p_s p_t (s + p_s.getD n 0 * sgnZ (p_s.getD n 0 - p_t.getD n 0)) n

/-- `tv_scan`'s written sign vector after `n` entries. -/
def sgVals (p_s p_t : List ℤ) (sg : List ℤ) : ℕ → List ℤ
  | 0 => sg
  | n + 1 => (sgVals p_s p_t sg n).set n (sgnZ (p_s.getD n 0 - p_t.getD n 0))

/-- `tv_dz`'s written gradient after `n` entries:
`dz[i] = round(p_s[i]·(sg[i]·2^32 − signed) / 2^33)`. -/
def tvDzVals (p_s sg : List ℤ) (signed : ℤ) (dz : List ℤ) : ℕ → List ℤ
  | 0 => dz
  | n + 1 => (tvDzVals p_s sg signed dz n).set n
      (rndDiv (p_s.getD n 0 * (sg.getD n 0 * 2 ^ (32 : ℕ) - signed)) (2 * 2 ^ (32 : ℕ)))

/-- `dot_round`'s accumulator over `d` terms of `a[abase..]·b[bbase..]`. -/
def dotAux (a : List ℤ) (ab : ℕ) (b : List ℤ) (bb : ℕ) (acc : ℤ) : ℕ → ℤ
  | 0 => acc
  | n + 1 => dotAux a ab b bb (acc + a.getD (ab + n) 0 * b.getD (bb + n) 0) n

/-- `logits_row`'s written logit vector after `n` entries. -/
def logitsVals (hidden w_head : List ℤ) (r d : ℕ) (zs : List ℤ) : ℕ → List ℤ
  | 0 => zs
  | n + 1 => (logitsVals hidden w_head r d zs n).set n
      (rndDiv (dotAux hidden (r * d) w_head (n * d) 0 d) (2 ^ (32 : ℕ)))

/-- `dw_fill`'s `d_w_head` accumulator after `n` of the `v·d` entries:
`dw[k·d + j] += dz[k]·hidden[hbase + j]`. -/
def dwVals (dz hidden : List ℤ) (hbase d : ℕ) (dw : List ℤ) : ℕ → List ℤ
  | 0 => dw
  | n + 1 =>
    (dwVals dz hidden hbase d dw n).set n
      ((dwVals dz hidden hbase d dw n).getD n 0 + dz.getD (n / d) 0 * hidden.getD (hbase + n % d) 0)

/-- `dw_out_fill`'s written vector after `n` entries: `out[j] = round(dw[j] / n_scale)`. -/
def dwOutVals (dw : List ℤ) (nscale : ℤ) (out : List ℤ) : ℕ → List ℤ
  | 0 => out
  | n + 1 => (dwOutVals dw nscale out n).set n (rndDiv (dw.getD n 0) nscale)

/-- `dh_entry`'s accumulator after `n` terms: `Σ_{k<n} dz[k]·w_head[k·d + j]`.

`logits_row` computes `z_k = Σ_j hidden[r·d+j]·w_head[k·d+j]` with `w_head` row-major (`V × d`), so
entry `(k, j)` of `W_head` is `w_head[k·d + j]` and the backward `dh[dbase+j] = Σ_{k<v} dz[k]·w_head[k·d+j]`
runs over `k` at the fixed column `j`. This is the accumulator the extracted `dh_entry` computes. -/
def dhAux (dz w_head : List ℤ) (d j : ℕ) (acc : ℤ) : ℕ → ℤ
  | 0 => acc
  | n + 1 => dhAux dz w_head d j (acc + dz.getD n 0 * w_head.getD (n * d + j) 0) n

/-- `dh_entry`'s value at column `j`: `round((Σ_{k<v} dz[k]·w_head[k·d+j]) / nscale)`, `v = dz.length`. -/
def dhEntryVal (dz w_head : List ℤ) (d j : ℕ) (nscale : ℤ) : ℤ :=
  rndDiv (dhAux dz w_head d j 0 dz.length) nscale

/-- `dh_fill`'s written vector after `n` entries of the `d`-block: entry `base + n` is
`dh_entry`'s value at column `n`. -/
def dhVals (dz w_head : List ℤ) (d : ℕ) (nscale : ℤ) (base : ℕ) (dh : List ℤ) : ℕ → List ℤ
  | 0 => dh
  | n + 1 => (dhVals dz w_head d nscale base dh n).set (base + n)
      (dhEntryVal dz w_head d n nscale)

/-! ## 3. Bound lemmas and the non-loop refinements -/

/-- `i64` type bounds as numerals. -/
theorem i64_min_eq : I64.min = -(2 : ℤ) ^ (63 : ℕ) := by
  norm_num [I64.min, IScalar.min, I64.numBits]

/-- `i64` type bounds as numerals. -/
theorem i64_max_eq : I64.max = (2 : ℤ) ^ (63 : ℕ) - 1 := by
  norm_num [I64.max, IScalar.max, I64.numBits]

/-- `i128` type bounds as numerals. -/
theorem i128_min_eq : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
  norm_num [I128.min, IScalar.min, I128.numBits]

/-- `i128` type bounds as numerals. -/
theorem i128_max_eq : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
  norm_num [I128.max, IScalar.max, I128.numBits]

/-- `core.num.I64.MIN` widened to `i128` has the value `I64.min`. -/
theorem fromMIN_val : (core.convert.num.FromI128I64.from core.num.I64.MIN).val = I64.min := by
  rw [core.convert.num.FromI128I64.from_val_eq]
  norm_num [core.num.I64.MIN, IScalar.ofInt, IScalar.rMin, I64.rMin, I64.min, IScalar.min,
    I64.numBits]

/-- `core.num.I64.MAX` widened to `i128` has the value `I64.max`. -/
theorem fromMAX_val : (core.convert.num.FromI128I64.from core.num.I64.MAX).val = I64.max := by
  rw [core.convert.num.FromI128I64.from_val_eq]
  norm_num [core.num.I64.MAX, IScalar.ofInt, IScalar.rMax, I64.rMax, I64.max, IScalar.max,
    I64.numBits]

/-- The narrowing cast `i128 → i64` preserves the value in the `i64` window. -/
theorem cast_inBounds (x : I128) (h : I64.min ≤ x.val ∧ x.val ≤ I64.max) :
    ∃ y, IScalar.cast .I64 x = y ∧ y.val = x.val := by
  obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists (IScalar.cast_inBounds_spec .I64 x (by
    rw [i64_min_eq, i64_max_eq] at h
    simp only [IScalar.min, IScalar.max]
    norm_num at h ⊢
    exact h))
  have hy' : ok (IScalar.cast .I64 x) = ok y := by simpa only [lift] using hy
  exact ⟨y, Result.ok_injective hy', hyv⟩

/-- The lower `to64` test is false above `i64::MIN`. -/
theorem not_lt_MIN (x : I128) (h : I64.min ≤ x.val) :
    ¬ (x < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
  intro hc
  have := (IScalar.lt_equiv x (core.convert.num.FromI128I64.from core.num.I64.MIN)).mp hc
  rw [fromMIN_val] at this
  omega

/-- The upper `to64` test is false below `i64::MAX`. -/
theorem not_gt_MAX (x : I128) (h : x.val ≤ I64.max) :
    ¬ (x > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
  intro hc
  have := (IScalar.lt_equiv (core.convert.num.FromI128I64.from core.num.I64.MAX) x).mp hc
  rw [fromMAX_val] at this
  omega

/-- **`abs128` success.** Above `I128.min` the body returns `|x|`. -/
theorem abs128_spec (x : I128) (h : I128.min < x.val) :
    ∃ z, learn_port.head.abs128 x = ok z ∧ z.val = absZ x.val := by
  unfold learn_port.head.abs128
  by_cases hx : x < (0#i128)
  · rw [if_pos hx]
    have hxlt : x.val < 0 := by
      have := (IScalar.lt_equiv x (0#i128)).mp hx
      simpa using this
    have h0 : (↑(0#i128) : ℤ) = 0 := rfl
    obtain ⟨z, hz, hzval, _⟩ := WP.spec_imp_exists (I128.sub_bv_spec (x := (0#i128)) (y := x)
      (by rw [h0, i128_min_eq]; norm_num; omega)
      (by rw [h0, i128_max_eq, i128_min_eq] at *; norm_num at *; omega))
    refine ⟨z, hz, ?_⟩
    rw [hzval, absZ, if_pos hxlt, h0]
    ring
  · rw [if_neg hx]
    refine ⟨x, rfl, ?_⟩
    have hxle : ¬ x.val < 0 := by
      intro hc
      exact hx ((IScalar.lt_equiv x (0#i128)).mpr (by simpa using hc))
    rw [absZ, if_neg hxle]

/-- **`sgn`.** The body returns `-1`, `0` or `1` by the sign of the value. -/
theorem sgn_spec (x : I128) :
    ∃ z, learn_port.head.sgn x = ok z ∧ z.val = sgnZ x.val := by
  unfold learn_port.head.sgn
  rcases lt_trichotomy (x.val : ℤ) 0 with hlt | heq | hgt
  · have hcmp : core.cmp.impls.OrdI128.cmp x (0#i128) = Ordering.lt := compare_lt_iff_lt.mpr hlt
    rw [hcmp]
    simp only [lift, bind_tc_ok]
    refine ⟨(-1)#i128, rfl, ?_⟩
    simp only [IScalar.val]
    simp [sgnZ, not_lt.mpr (le_of_lt hlt), hlt]
  · have hcmp : core.cmp.impls.OrdI128.cmp x (0#i128) = Ordering.eq := compare_eq_iff_eq.mpr heq
    rw [hcmp]
    simp only [lift, bind_tc_ok]
    refine ⟨0#i128, rfl, ?_⟩
    simp only [IScalar.val]
    simp [sgnZ, heq]
  · have hcmp : core.cmp.impls.OrdI128.cmp x (0#i128) = Ordering.gt :=
      compare_gt_iff_gt.mpr hgt
    rw [hcmp]
    simp only [lift, bind_tc_ok]
    refine ⟨1#i128, rfl, ?_⟩
    simp only [IScalar.val]
    simp [sgnZ, hgt]

/-- **`to64` success.** Inside the `i64` window the body returns the narrowed value. -/
theorem to64_ok (x : I128) (h : I64.min ≤ x.val ∧ x.val ≤ I64.max) :
    ∃ z, learn_port.head.to64 x = ok (core.result.Result.Ok z) ∧ z.val = x.val := by
  unfold learn_port.head.to64
  simp only [lift, bind_tc_ok]
  rw [if_neg (not_lt_MIN x h.1), if_neg (not_gt_MAX x h.2)]
  obtain ⟨y, hcast, hyv⟩ := cast_inBounds x h
  exact ⟨y, by rw [hcast], hyv⟩

/-- **`to64` refusal below `i64::MIN`.** -/
theorem to64_low (x : I128) (h : x.val < I64.min) :
    learn_port.head.to64 x =
      ok (core.result.Result.Err learn_port.head.HeadError.Overflow) := by
  unfold learn_port.head.to64
  simp only [lift, bind_tc_ok]
  rw [if_pos]
  exact (IScalar.lt_equiv x _).mpr (by rw [fromMIN_val]; exact h)

/-- **`to64` refusal above `i64::MAX`.** -/
theorem to64_high (x : I128) (h : I64.max < x.val) :
    learn_port.head.to64 x =
      ok (core.result.Result.Err learn_port.head.HeadError.Overflow) := by
  unfold learn_port.head.to64
  simp only [lift, bind_tc_ok]
  rw [if_neg (not_lt_MIN x (by
        have hh := h
        rw [i64_max_eq] at hh
        rw [i64_min_eq]
        norm_num at hh ⊢
        omega))]
  rw [if_pos]
  exact (IScalar.lt_equiv _ x).mpr (by rw [fromMAX_val]; exact h)

/-- **`add128` success.** -/
theorem add128_ok (a b : I128) (h : I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max) :
    ∃ s, learn_port.head.add128 a b = ok (core.result.Result.Ok s) ∧ s.val = a.val + b.val := by
  unfold learn_port.head.add128
  simp only [lift, bind_tc_ok]
  obtain ⟨s, hs, hsv⟩ := checked_add_some_i128 h
  rw [hs]
  exact ⟨s, rfl, hsv⟩

/-- **`add128` refusal.** -/
theorem add128_refuse (a b : I128)
    (h : ¬ (I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max)) :
    learn_port.head.add128 a b =
      ok (core.result.Result.Err learn_port.head.HeadError.Overflow) := by
  unfold learn_port.head.add128
  simp only [lift, bind_tc_ok]
  cases hm : I128.checked_add a b with
  | none => rfl
  | some s => exact absurd (checked_add_bounds hm) h

/-- **`mul128` success.** -/
theorem mul128_ok (a b : I128) (h : I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max) :
    ∃ s, learn_port.head.mul128 a b = ok (core.result.Result.Ok s) ∧ s.val = a.val * b.val := by
  unfold learn_port.head.mul128
  simp only [lift, bind_tc_ok]
  obtain ⟨s, hs, hsv⟩ := checked_mul_some_i128 h
  rw [hs]
  exact ⟨s, rfl, hsv⟩

/-- **`mul128` refusal.** -/
theorem mul128_refuse (a b : I128)
    (h : ¬ (I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ I128.max)) :
    learn_port.head.mul128 a b =
      ok (core.result.Result.Err learn_port.head.HeadError.Overflow) := by
  unfold learn_port.head.mul128
  simp only [lift, bind_tc_ok]
  cases hm : I128.checked_mul a b with
  | none => rfl
  | some s => exact absurd (checked_mul_bounds hm) h

/-- A `usize` value fits the `i128` window on any platform width. -/
theorem usize_val_le_i128max (a : Usize) : (a.val : ℤ) ≤ IScalar.max IScalarTy.I128 := by
  simp only [IScalar.max_IScalarTy_I128_eq]
  have h : a.val < 2 ^ (64 : ℕ) := by
    have h := a.hBounds
    rw [UScalarTy.Usize_numBits_eq] at h
    rcases System.Platform.numBits_eq with hp | hp <;> rw [hp] at h <;> norm_num at h <;> omega
  scalar_tac

/-- **`dim128`.** Every `usize` widens to `i128` with its value unchanged. -/
theorem dim128_spec (a : Usize) : ∃ z, learn_port.head.dim128 a = ok z ∧ z.val = a.val := by
  unfold learn_port.head.dim128
  obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists (UScalar.hcast_inBounds_spec .I128 a
    (usize_val_le_i128max a))
  have hy' : UScalar.hcast .I128 a = y := Result.ok_injective (by simpa only [lift] using hy)
  exact ⟨y, by rw [hy'], hyv⟩

/-- **`mul_dims` success.** -/
theorem mul_dims_ok (a b : Usize) (h : a.val * b.val ≤ Usize.max) :
    ∃ s, learn_port.head.mul_dims a b = ok (core.result.Result.Ok s) ∧ s.val = a.val * b.val := by
  unfold learn_port.head.mul_dims
  simp only [lift, bind_tc_ok]
  obtain ⟨s, hs, hsv⟩ := checked_mul_some_usize h
  rw [hs]
  exact ⟨s, rfl, hsv⟩

/-- **`mul_dims` refusal.** -/
theorem mul_dims_refuse (a b : Usize) (h : Usize.max < a.val * b.val) :
    learn_port.head.mul_dims a b =
      ok (core.result.Result.Err learn_port.head.HeadError.Shape) := by
  unfold learn_port.head.mul_dims
  simp only [lift, bind_tc_ok]
  cases hm : Usize.checked_mul a b with
  | none => rfl
  | some s => exact absurd (checked_mul_usize_bounds hm) (by omega)

/-- **`lp_entry` success.** `log p_i = (z_i − m) − l`, computed in `i128` and narrowed once. -/
theorem lp_entry_ok (zi m l : I64)
    (h1 : I128.min ≤ zi.val - m.val ∧ zi.val - m.val ≤ I128.max)
    (h2 : I128.min ≤ zi.val - m.val - l.val ∧ zi.val - m.val - l.val ≤ I128.max)
    (h3 : I64.min ≤ zi.val - m.val - l.val) (h4 : zi.val - m.val - l.val ≤ I64.max) :
    ∃ v, learn_port.head.lp_entry zi m l = ok (core.result.Result.Ok v) ∧
      v.val = zi.val - m.val - l.val := by
  unfold learn_port.head.lp_entry
  simp only [lift, bind_tc_ok]
  obtain ⟨i2, hi2, hi2v, _⟩ := WP.spec_imp_exists (I128.sub_bv_spec
    (x := core.convert.num.FromI128I64.from zi) (y := core.convert.num.FromI128I64.from m)
    (by simpa only [core.convert.num.FromI128I64.from_val_eq] using h1.1)
    (by simpa only [core.convert.num.FromI128I64.from_val_eq] using h1.2))
  rw [hi2]
  simp only [bind_tc_ok]
  obtain ⟨i4, hi4, hi4v, _⟩ := WP.spec_imp_exists (I128.sub_bv_spec
    (x := i2) (y := core.convert.num.FromI128I64.from l)
    (by rw [hi2v]; simpa only [core.convert.num.FromI128I64.from_val_eq] using h2.1)
    (by rw [hi2v]; simpa only [core.convert.num.FromI128I64.from_val_eq] using h2.2))
  rw [hi4]
  simp only [bind_tc_ok]
  obtain ⟨v, hv, hvv⟩ := to64_ok i4 (by
    rw [hi4v, hi2v]
    simp only [core.convert.num.FromI128I64.from_val_eq]
    exact ⟨h3, h4⟩)
  exact ⟨v, hv, by
    rw [hvv, hi4v, hi2v]
    simp only [core.convert.num.FromI128I64.from_val_eq]⟩

/-- `rndDiv` unfolded: the body's own expression. -/
theorem rndDiv_eq (n d : ℤ) :
    rndDiv n d =
      (if 2 * absZ (Int.tmod n d) ≥ d then
        (if 0 ≤ n then Int.tdiv n d + 1 else Int.tdiv n d - 1)
      else Int.tdiv n d) := rfl

/-- `absZ` is never negative. -/
theorem absZ_nonneg (x : ℤ) : 0 ≤ absZ x := by
  rw [absZ]
  split <;> omega

/-- `i128::MIN` is a lower bound for every nonnegative integer. -/
theorem i128_min_le_of_nonneg {x : ℤ} (h : 0 ≤ x) : I128.min ≤ x := by
  rw [i128_min_eq]
  norm_num
  omega

/-- For a positive divisor the truncated remainder lies strictly above `i128::MIN`. -/
theorem i128_min_lt_tmod (n d : I128) (hd : 0 < d.val) :
    I128.min < Int.tmod n.val d.val := by
  have h1 := Int.lt_tmod_of_pos n.val hd
  have h2 := (i128_bounds d).2
  rw [i128_max_eq] at h2
  rw [i128_min_eq]
  norm_num at h2 ⊢
  omega

/-- **`round_div`.** Round `n / d` to nearest, ties away from zero.
Each checked step carries the hypothesis that makes it admit a result: `d > 0`, the doubled remainder
fits, and the rounded quotient fits. -/
theorem round_div_spec (n d : I128) (hd : 0 < d.val)
    (hmul : 2 * absZ (Int.tmod n.val d.val) ≤ I128.max)
    (hadd : I128.min ≤ Int.tdiv n.val d.val + 1 ∧ Int.tdiv n.val d.val + 1 ≤ I128.max)
    (hsub : I128.min ≤ Int.tdiv n.val d.val - 1 ∧ Int.tdiv n.val d.val - 1 ≤ I128.max) :
    ∃ v, learn_port.head.round_div n d = ok v ∧ v.val = rndDiv n.val d.val := by
  have h2i : (↑(2#i128) : ℤ) = 2 := rfl
  have h1i : (↑(1#i128) : ℤ) = 1 := rfl
  unfold learn_port.head.round_div
  obtain ⟨q, hq, hqv, _⟩ :=
    I128.div_bv_spec (x := n) (y := d) (by omega) (by intro hc; omega)
  rw [hq]
  simp only [bind_tc_ok]
  obtain ⟨r, hr, hrv, _⟩ := WP.spec_imp_exists (I128.rem_bv_spec (x := n) (y := d) (by omega))
  rw [hr]
  simp only [bind_tc_ok]
  obtain ⟨i, hi, hiv⟩ := abs128_spec r (by rw [hrv]; exact i128_min_lt_tmod n d hd)
  rw [hi]
  simp only [bind_tc_ok]
  have hival : (↑i : ℤ) = absZ (Int.tmod n.val d.val) := by rw [hiv, hrv]
  obtain ⟨i1, hi1, hi1v, _⟩ := WP.spec_imp_exists (I128.mul_bv_spec (x := 2#i128) (y := i)
    (by rw [h2i, hival]; exact i128_min_le_of_nonneg (by have h := absZ_nonneg (Int.tmod n.val d.val); omega))
    (by rw [h2i, hival]; exact hmul))
  rw [hi1]
  simp only [bind_tc_ok]
  have hi1val : (↑i1 : ℤ) = 2 * absZ (Int.tmod n.val d.val) := by rw [hi1v, h2i, hival]
  rw [rndDiv_eq]
  by_cases hge : i1 >= d
  · have hge' : d.val ≤ i1.val := (IScalar.le_equiv d i1).mp hge
    have hsge : d.val ≤ 2 * absZ (Int.tmod n.val d.val) := by rw [← hi1val]; exact hge'
    rw [if_pos hge, if_pos hsge]
    by_cases hn : n >= 0#i128
    · have hn' : 0 ≤ n.val := (IScalar.le_equiv (0#i128) n).mp hn
      rw [if_pos hn, if_pos hn']
      obtain ⟨v, hv, hvv, _⟩ := WP.spec_imp_exists (I128.add_bv_spec (x := q) (y := 1#i128)
        (by rw [hqv, h1i]; exact hadd.1) (by rw [hqv, h1i]; exact hadd.2))
      exact ⟨v, hv, by rw [hvv, hqv, h1i]⟩
    · have hn' : ¬ 0 ≤ n.val := by
        intro hc
        exact hn ((IScalar.le_equiv (0#i128) n).mpr (by simpa using hc))
      rw [if_neg hn, if_neg hn']
      obtain ⟨v, hv, hvv, _⟩ := WP.spec_imp_exists (I128.sub_bv_spec (x := q) (y := 1#i128)
        (by rw [hqv, h1i]; exact hsub.1) (by rw [hqv, h1i]; exact hsub.2))
      exact ⟨v, hv, by rw [hvv, hqv, h1i]⟩
  · have hge' : ¬ d.val ≤ i1.val := by
      intro hc
      exact hge ((IScalar.le_equiv d i1).mpr (by simpa using hc))
    have hsge : ¬ d.val ≤ 2 * absZ (Int.tmod n.val d.val) := by
      intro hc
      exact hge' (by rw [hi1val]; exact hc)
    rw [if_neg hge, if_neg hsge]
    exact ⟨q, rfl, hqv⟩

/-! ## 5. The public composition: `log_softmax_row`

`log_softmax_row` is `softmax_row` with the pair projected. Its refinement is stated **conditionally**
on the extracted `softmax_row` returning the pair the caller consumes, so the engine bound (the
`fixed_exp`/`fixed_ln` refinement) stays in `Numeric.Engine.Attn` and is not restated. -/

/-- **`log_softmax_row`, success arm.** The log-softmax row is the first component of the pair the
softmax row returns. -/
theorem log_softmax_row_spec (z : Slice I64) (lp p : alloc.vec.Vec I64)
    (h : learn_port.head.softmax_row z = ok (core.result.Result.Ok (lp, p))) :
    learn_port.head.log_softmax_row z = ok (core.result.Result.Ok lp) := by
  unfold learn_port.head.log_softmax_row
  rw [h]
  simp only [bind_tc_ok]
  rfl

/-- **`log_softmax_row`, refusal arm.** A softmax-row refusal is passed through. -/
theorem log_softmax_row_err (z : Slice I64) (x : learn_port.head.HeadError)
    (h : learn_port.head.softmax_row z = ok (core.result.Result.Err x)) :
    learn_port.head.log_softmax_row z = ok (core.result.Result.Err x) := by
  unfold learn_port.head.log_softmax_row
  rw [h]
  simp only [bind_tc_ok]

/-! ## 6. Witnesses

Every refined body is exhibited on a value. `abs128` reduces in the kernel; the bodies that go through
the Aeneas division, ordering or cast models are exhibited against their own refinement theorem. -/

example : learn_port.head.abs128 (-7)#i128 = ok 7#i128 := rfl
example : learn_port.head.abs128 7#i128 = ok 7#i128 := rfl

/-- Witness: `to64` narrows `5` to `i64`. -/
theorem to64_witness_ok :
    learn_port.head.to64 5#i128 = ok (core.result.Result.Ok 5#i64) := by
  have h5 : (↑(5#i128) : ℤ) = 5 := rfl
  obtain ⟨z, hz, hzv⟩ := to64_ok 5#i128 (by rw [h5, i64_min_eq, i64_max_eq]; norm_num)
  rw [hz]
  exact congrArg (fun w => ok (core.result.Result.Ok w))
    (IScalar.eq_imp _ _ (by rw [hzv, h5]; rfl))

/-- Witness: `add128` adds `5` and `7`. -/
theorem add128_witness_ok :
    learn_port.head.add128 5#i128 7#i128 = ok (core.result.Result.Ok 12#i128) := by
  have h5 : (↑(5#i128) : ℤ) = 5 := rfl
  have h7 : (↑(7#i128) : ℤ) = 7 := rfl
  have h12 : (↑(12#i128) : ℤ) = 12 := rfl
  obtain ⟨s, hs, hsv⟩ := add128_ok 5#i128 7#i128 (by rw [h5, h7, i128_min_eq, i128_max_eq]; norm_num)
  rw [hs]
  exact congrArg (fun w => ok (core.result.Result.Ok w))
    (IScalar.eq_imp _ _ (by rw [hsv, h12, h5, h7]; norm_num))

/-- Witness: `mul128` multiplies `5` by `7`. -/
theorem mul128_witness_ok :
    learn_port.head.mul128 5#i128 7#i128 = ok (core.result.Result.Ok 35#i128) := by
  have h5 : (↑(5#i128) : ℤ) = 5 := rfl
  have h7 : (↑(7#i128) : ℤ) = 7 := rfl
  have h35 : (↑(35#i128) : ℤ) = 35 := rfl
  obtain ⟨s, hs, hsv⟩ := mul128_ok 5#i128 7#i128 (by rw [h5, h7, i128_min_eq, i128_max_eq]; norm_num)
  rw [hs]
  exact congrArg (fun w => ok (core.result.Result.Ok w))
    (IScalar.eq_imp _ _ (by rw [hsv, h35, h5, h7]; norm_num))

/-- Witness: `lp_entry` gives `(5 − 2) − 1 = 2`. -/
theorem lp_entry_witness_ok :
    learn_port.head.lp_entry 5#i64 2#i64 1#i64 = ok (core.result.Result.Ok 2#i64) := by
  have h5 : (↑(5#i64) : ℤ) = 5 := rfl
  have h2 : (↑(2#i64) : ℤ) = 2 := rfl
  have h1 : (↑(1#i64) : ℤ) = 1 := rfl
  obtain ⟨v, hv, hvv⟩ := lp_entry_ok 5#i64 2#i64 1#i64
    (by rw [h5, h2, i128_min_eq, i128_max_eq]; norm_num)
    (by rw [h5, h2, h1, i128_min_eq, i128_max_eq]; norm_num)
    (by rw [h5, h2, h1, i64_min_eq]; norm_num)
    (by rw [h5, h2, h1, i64_max_eq]; norm_num)
  rw [hv]
  exact congrArg (fun w => ok (core.result.Result.Ok w))
    (IScalar.eq_imp _ _ (by
      rw [hvv]
      simp only [h5, h2, h1]
      norm_num))

end apeiron_head
