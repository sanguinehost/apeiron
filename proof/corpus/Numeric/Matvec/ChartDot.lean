/-
Apeiron — integer chart dot: activation quantizer spec and chart factoring.

Spec: `docs/epics/engine-throughput.md`, section "W2 — integer chart dot"; task
`APEIRON-TPUT-W2-T7`. Pure integer mathematics over `ℤ` (Mathlib); no extraction and no Rust
dependency. The module fixes the activation quantizer the `Q8Act` fast path uses and proves the
facts the portable/AVX2/DP4A kernels rely on:

* `qcode_range` — every code lies in `[−127, 127]`;
* `round_err` and `qcode_err` — round-to-nearest, ties away from zero, has absolute error at most
  `S/2`; `scale_eq_zero_block` discharges the `S = 0` block;
* `qcode_smul`, `quantBlock_idem`, `dequantBlock_idem` — idempotence on the code grid `|k| ≤ 127`;
* `chart_factor` and `dequant_block_factor` — the exact distribution identity
  `Σ_j (W·q_j − M)·x_j = W·Σ_j q_j·x_j − M·Σ_j x_j` and its block form;
* `row_err_bound` — the per-block activation error over one row is at most `(S/2)·Σ_j |w_j|`,
  stated with the factor `2`.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom`, `opaque`,
`native_decide`, `bv_decide`, `decide`/`rfl` over a large expression, or raised heartbeat bound
appears.
-/
import Mathlib

set_option autoImplicit false
set_option warningAsError true

namespace ChartDot

/-! ## Definitions -/

/-- Largest `Int.natAbs` in a block; `0` on the empty block. -/
def absmax (xs : List ℤ) : ℕ := (xs.map Int.natAbs).foldr max 0

/-- Per-block activation scale `ceil(absmax xs / 127)`, as an integer. -/
def scale (xs : List ℤ) : ℤ := ((absmax xs : ℤ) + 126) / 127

/-- Round-to-nearest, ties away from zero, of `x / S` (`0` when `S = 0`). -/
def qcode (S x : ℤ) : ℤ := if S = 0 then 0 else Int.sign x * ((2 * |x| + S) / (2 * S))

/-- Activation code of a block. -/
def quantBlock (xs : List ℤ) : List ℤ := xs.map (qcode (scale xs))

/-- Dequantized block `x̂_j = S·qcode S x_j`, elementwise. -/
def dequantBlock (xs : List ℤ) : List ℤ := (quantBlock xs).map (scale xs * ·)

/-! ## `absmax` and `scale` -/

theorem absmax_nonneg (xs : List ℤ) : 0 ≤ absmax xs := Nat.zero_le _

theorem natAbs_le_absmax {x : ℤ} {xs : List ℤ} (hx : x ∈ xs) : x.natAbs ≤ absmax xs := by
  induction xs with
  | nil => cases hx
  | cons a xs ih =>
      rw [absmax, List.map_cons, List.foldr_cons]
      rcases List.mem_cons.mp hx with rfl | hx
      · exact le_max_left _ _
      · exact le_trans (ih hx) (le_max_right _ _)

theorem le_absmax {x : ℤ} {xs : List ℤ} (hx : x ∈ xs) : |x| ≤ (absmax xs : ℤ) := by
  rw [Int.abs_eq_natAbs]
  exact_mod_cast natAbs_le_absmax hx

theorem scale_nonneg (xs : List ℤ) : 0 ≤ scale xs := by
  rw [scale]
  apply Int.ediv_nonneg
  · have := absmax_nonneg xs; omega
  · norm_num

theorem absmax_le_scale_mul (xs : List ℤ) : (absmax xs : ℤ) ≤ 127 * scale xs := by
  rw [scale]
  have h := Int.lt_ediv_add_one_mul_self ((absmax xs : ℤ) + 126)
    (show (0 : ℤ) < 127 by norm_num)
  omega

theorem scale_eq_zero_iff (xs : List ℤ) : scale xs = 0 ↔ absmax xs = 0 := by
  constructor
  · intro h
    have hA := absmax_le_scale_mul xs
    rw [h, mul_zero] at hA
    omega
  · intro h
    rw [scale, h]
    norm_num

theorem scale_pos_of_absmax_pos (xs : List ℤ) (h : 0 < absmax xs) : 0 < scale xs := by
  have h1 := absmax_le_scale_mul xs
  have h2 : (0 : ℤ) < absmax xs := by exact_mod_cast h
  nlinarith [h1, h2]

/-! ## Range -/

theorem qcode_range (xs : List ℤ) (x : ℤ) (hx : x ∈ xs) :
    |qcode (scale xs) x| ≤ 127 := by
  by_cases hS : scale xs = 0
  · have hA : absmax xs = 0 := (scale_eq_zero_iff xs).mp hS
    have hx0 : x = 0 := by
      have hle := natAbs_le_absmax hx
      rw [hA] at hle
      exact Int.natAbs_eq_zero.mp (Nat.le_zero.mp hle)
    subst x
    rw [qcode, if_pos hS, abs_zero]
    norm_num
  · have hSpos : 0 < scale xs := lt_of_le_of_ne' (scale_nonneg xs) hS
    have ha : |x| ≤ (absmax xs : ℤ) := le_absmax hx
    have hA : (absmax xs : ℤ) ≤ 127 * scale xs := absmax_le_scale_mul xs
    have hq_nonneg : 0 ≤ (2 * |x| + scale xs) / (2 * scale xs) :=
      Int.ediv_nonneg (by nlinarith [abs_nonneg x, hSpos]) (by nlinarith [le_of_lt hSpos])
    have hq_lt : (2 * |x| + scale xs) / (2 * scale xs) < 128 := by
      apply (Int.ediv_lt_iff_lt_mul (show (0 : ℤ) < 2 * scale xs by nlinarith)).mpr
      nlinarith [ha, hA, hSpos]
    have hq_le : (2 * |x| + scale xs) / (2 * scale xs) ≤ 127 := by omega
    have hsign : |Int.sign x| ≤ 1 := by
      rw [Int.abs_eq_natAbs, Int.natAbs_sign]
      split <;> norm_num
    rw [qcode, if_neg hS, abs_mul, abs_of_nonneg hq_nonneg]
    have hm := mul_le_mul_of_nonneg_right hsign hq_nonneg
    nlinarith [hm, hq_le]

/-! ## Error bound -/

theorem round_err (S t : ℤ) (hS : 0 < S) :
    2 * |t - S * ((2 * t + S) / (2 * S))| ≤ S := by
  set q := (2 * t + S) / (2 * S) with hq
  set rem := (2 * t + S) % (2 * S) with hrem
  have hden : (2 * S) ≠ 0 := by nlinarith
  have hdec : rem + (2 * S) * q = 2 * t + S := by
    rw [hrem, hq]
    exact Int.emod_add_mul_ediv _ _
  have hrem_nn : 0 ≤ rem := Int.emod_nonneg _ hden
  have hrem_lt : rem < 2 * S := Int.emod_lt_of_pos _ (by nlinarith)
  have h2 : 2 * (t - S * q) = rem - S := by linarith only [hdec]
  have hbound : |rem - S| ≤ S := by rw [abs_le]; omega
  have h2abs : 2 * |t - S * q| = |rem - S| := by
    rw [show 2 * |t - S * q| = |2 * (t - S * q)| by
          rw [abs_mul, abs_of_nonneg (show (0 : ℤ) ≤ 2 by norm_num)]]
    rw [h2]
  linarith [h2abs, hbound]

theorem qcode_err (S x : ℤ) (hS : 0 < S) : 2 * |x - S * qcode S x| ≤ S := by
  rw [qcode, if_neg (ne_of_gt hS)]
  rcases lt_trichotomy x 0 with hlt | heq | hgt
  · have hsign : Int.sign x = -1 := Int.sign_eq_neg_one_of_neg hlt
    have habs : |x| = -x := abs_of_neg hlt
    rw [hsign, habs, neg_one_mul]
    have hdecomp : x - S * (-((2 * -x + S) / (2 * S)))
        = -((-x) - S * ((2 * -x + S) / (2 * S))) := by ring
    rw [hdecomp, abs_neg]
    exact round_err S (-x) hS
  · subst x
    simp only [Int.sign_zero, zero_mul, mul_zero, sub_zero, abs_zero]
    exact hS.le
  · have hsign : Int.sign x = 1 := Int.sign_eq_one_of_pos hgt
    have habs : |x| = x := abs_of_pos hgt
    rw [hsign, one_mul, habs]
    exact round_err S x hS

theorem scale_eq_zero_block (xs : List ℤ) (h : scale xs = 0) :
    (∀ x ∈ xs, x = 0) ∧ dequantBlock xs = xs.map (fun _ => 0) := by
  have hA : absmax xs = 0 := (scale_eq_zero_iff xs).mp h
  constructor
  · intro x hx
    have hle := natAbs_le_absmax hx
    rw [hA] at hle
    exact Int.natAbs_eq_zero.mp (Nat.le_zero.mp hle)
  · unfold dequantBlock quantBlock
    rw [h]
    have hq : ∀ y : ℤ, qcode (0 : ℤ) y = 0 := fun y => by rw [qcode, if_pos rfl]
    rw [show xs.map (qcode (0 : ℤ)) = xs.map (fun _ => (0 : ℤ)) from
          List.map_congr_left (fun y _ => hq y)]
    rw [List.map_map]
    apply List.map_congr_left
    intro y _
    show (0 : ℤ) * 0 = 0
    ring

/-! ## Idempotence on the code grid -/

theorem qcode_smul (S k : ℤ) (hS : 0 < S) (_hk : |k| ≤ 127) : qcode S (S * k) = k := by
  rw [qcode, if_neg (ne_of_gt hS)]
  have habs : |S * k| = S * |k| := by rw [abs_mul, abs_of_pos hS]
  rw [habs]
  have hnum : 2 * (S * |k|) + S = S * (2 * |k| + 1) := by ring
  rw [hnum]
  rw [show 2 * S = S * 2 by ring]
  rw [Int.mul_ediv_mul_of_pos (2 * |k| + 1) 2 hS]
  have hdiv : (2 * |k| + 1) / 2 = |k| := by
    rw [show 2 * |k| + 1 = 1 + 2 * |k| by ring]
    rw [Int.add_mul_ediv_left (1 : ℤ) (b := (2 : ℤ)) (c := |k|) (by norm_num)]
    rw [show (1 : ℤ) / 2 = 0 by norm_num]
    ring
  rw [hdiv]
  rw [Int.sign_mul, Int.sign_eq_one_of_pos hS, one_mul]
  exact Int.sign_mul_abs k

theorem dequantBlock_eq (xs : List ℤ) :
    dequantBlock xs = xs.map (fun x => scale xs * qcode (scale xs) x) := by
  simp only [dequantBlock, quantBlock, List.map_map, Function.comp_def]

theorem quantBlock_idem (xs : List ℤ) (hS : 0 < scale xs)
    (hscale : scale (dequantBlock xs) = scale xs) :
    quantBlock (dequantBlock xs) = quantBlock xs := by
  rw [quantBlock, hscale, dequantBlock_eq, List.map_map]
  apply List.map_congr_left
  intro x hx
  show qcode (scale xs) (scale xs * qcode (scale xs) x) = qcode (scale xs) x
  rw [qcode_smul (scale xs) (qcode (scale xs) x) hS (qcode_range xs x hx)]

theorem dequantBlock_idem (xs : List ℤ) (hS : 0 < scale xs)
    (hscale : scale (dequantBlock xs) = scale xs) :
    dequantBlock (dequantBlock xs) = dequantBlock xs := by
  change (quantBlock (dequantBlock xs)).map (scale (dequantBlock xs) * ·)
      = (quantBlock xs).map (scale xs * ·)
  rw [hscale, quantBlock_idem xs hS hscale]

/-! ## Chart factoring -/

theorem chart_factor (q x : List ℤ) (hlen : q.length = x.length) (W M : ℤ) :
    (q.zipWith (fun qj xj => (W * qj - M) * xj) x).sum
      = W * (q.zipWith (fun a b => a * b) x).sum - M * x.sum := by
  induction q generalizing x with
  | nil =>
      cases x with
      | nil => simp
      | cons b x => simp at hlen
  | cons a q ih =>
      cases x with
      | nil => simp at hlen
      | cons b x =>
          simp only [List.length_cons] at hlen
          have hlen' : q.length = x.length := by simpa using hlen
          simp only [List.zipWith_cons_cons, List.sum_cons]
          rw [ih x hlen']
          ring

theorem dequant_block_factor (w c : List ℤ) (hlen : w.length = c.length) (S : ℤ) :
    (w.zipWith (fun wj cj => wj * (S * cj)) c).sum
      = S * (w.zipWith (fun a b => a * b) c).sum := by
  induction w generalizing c with
  | nil =>
      cases c with
      | nil => simp
      | cons b c => simp at hlen
  | cons a w ih =>
      cases c with
      | nil => simp at hlen
      | cons b c =>
          simp only [List.length_cons] at hlen
          have hlen' : w.length = c.length := by simpa using hlen
          simp only [List.zipWith_cons_cons, List.sum_cons]
          rw [ih c hlen']
          ring

/-! ## Row error bound -/

theorem dot_diff_weighted (w x : List ℤ) (S : ℤ) (hlen : w.length = x.length) :
    (w.zipWith (fun a b => a * b) x).sum
        - (w.zipWith (fun a b => a * b) (x.map (fun xj => S * qcode S xj))).sum
      = (w.zipWith (fun wj xj => wj * (xj - S * qcode S xj)) x).sum := by
  induction w generalizing x with
  | nil =>
      cases x with
      | nil => simp
      | cons b x => simp at hlen
  | cons a w ih =>
      cases x with
      | nil => simp at hlen
      | cons b x =>
          simp only [List.length_cons] at hlen
          have hlen' : w.length = x.length := by simpa using hlen
          simp only [List.zipWith_cons_cons, List.sum_cons, List.map_cons]
          rw [← ih x hlen']
          ring

theorem weighted_err_bound (w x : List ℤ) (S : ℤ) (hS : 0 < S)
    (hlen : w.length = x.length) :
    2 * |(w.zipWith (fun wj xj => wj * (xj - S * qcode S xj)) x).sum|
      ≤ S * (w.map (fun a => |a|)).sum := by
  induction w generalizing x with
  | nil =>
      cases x with
      | nil => simp
      | cons b x => simp at hlen
  | cons a w ih =>
      cases x with
      | nil => simp at hlen
      | cons b x =>
          simp only [List.length_cons] at hlen
          have hlen' : w.length = x.length := by simpa using hlen
          have ih' := ih x hlen'
          have herr := qcode_err S b hS
          simp only [List.zipWith_cons_cons, List.sum_cons, List.map_cons]
          have htri := abs_add_le (a * (b - S * qcode S b))
            (w.zipWith (fun wj xj => wj * (xj - S * qcode S xj)) x).sum
          rw [abs_mul] at htri
          have hmul : |a| * (2 * |b - S * qcode S b|) ≤ |a| * S :=
            mul_le_mul_of_nonneg_left herr (abs_nonneg a)
          nlinarith [htri, hmul, ih']

theorem row_err_bound (w x : List ℤ) (hlen : w.length = x.length)
    (hS : 0 < scale x) :
    2 * |(w.zipWith (fun a b => a * b) x).sum
        - (w.zipWith (fun a b => a * b) (dequantBlock x)).sum|
      ≤ scale x * (w.map (fun a => |a|)).sum := by
  rw [dequantBlock_eq]
  rw [dot_diff_weighted w x (scale x) hlen]
  exact weighted_err_bound w x (scale x) hS hlen

end ChartDot
