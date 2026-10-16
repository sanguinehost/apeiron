/-
Integer and index refinement of the extracted `chart::QuantAct::magnitude` loop
(`chart.QuantAct.magnitude_loop`), the seventh group of the `kern::gemm` refinement.

Serving path: `Engine::next_token_state` -> `step::cols` -> `kern::gemm::pack_group` /
`kern::gemm::fast_columns_ok`, which read a `chart::QuantAct` through `chart.QuantAct.len`,
`chart.QuantAct.impl.scales`, `chart.QuantAct.magnitude` and `chart.q8act_fast_bound_ok`.
`Serve/Kern/R4/Gemm6.lean` refined the `chart.QuantAct` accessors and stated Goal 1 as open: the value
of `chart.QuantAct.magnitude`, i.e. the value of `chart.QuantAct.magnitude_loop` (`I64.checked_mul`,
`I128.checked_add`, `chart.ActMagnitude.{Exact,NoQ32,TotalOverflow}`). This module closes it.

The extracted body, for `idx < codes.len`, reads `scale = scales[idx / 32]` (absent -> `NoQ32`), reads
`codes[idx]`, multiplies the two on the checked `i64` success arm (overflow -> `NoQ32`), widens the
product to `i128`, takes `I128.abs`, and adds it to the accumulator on the checked `i128` success arm
(overflow -> `TotalOverflow`); the absent scale and the `i64` overflow share the `NoQ32` arm. At
`idx >= codes.len` it returns `Exact acc`.

The reference is plain Lean over `Int`/`Nat` and states WHAT the loop computes: `gemm7_term` is the
absolute value of the `i64` product of a code with its block scale, `gemm7_sumFrom` sums it over a code
window, `gemm7_magnitude` sums it over every code. `word_quantact_magnitude_loop_ok` states the loop
value on the domain where every block scale is present, every `i64` product fits and the accumulator
stays in `i128`; the loop's own termination measure is `codes.len - idx`.

The partiality is stated, not skirted: `gemm7_body_noq32_scale`, `gemm7_body_noq32_mul` and
`gemm7_body_totaloverflow` give the one-step refusal values on their exact conditions (absent block
scale, `i64` product overflow, `i128` accumulator overflow), and
`word_quantact_magnitude_loop_refuses` lifts them to the loop: from a state whose prefix `[idx0,
istop)` is in range and whose accumulation is bounded, a body refusal at `istop` returns that
refusal unchanged. Each arm carries a checked non-vacuity witness.

NOT refined here, stated rather than skirted:

* `kern.gemm.fast_columns_ok` (the brief's Goal 2): its body calls `chart.QuantAct.len`,
  `chart.QuantAct.impl.scales`, `chart.QuantAct.magnitude` and `chart.q8act_fast_bound_ok`. This
  module supplies the `chart.QuantAct.magnitude` value; `chart.q8act_fast_bound_ok`'s own loop
  (`core.num.I128.abs` / `I128.saturating_mul` over a slice) is a separate refinement and stays open.
* `chart.QuantAct.dequantizes*`, `chart.quant_act*` and the `q8act_plan`/`q8act` consumers of the
  magnitude value.

Trust boundary: rustc/LLVM codegen, the `F32`/AVX2 layers, the `axiom` pool driver and the
`R[x]/(x^2 - sigma)` word kernels are not covered here. No `sorry`/`admit`/`axiom`/`native_decide`/
`bv_decide`/`maxHeartbeats 0`; `decide` appears only on literal values and literal integer bounds.
-/
import Serve.Kern.R4.Gemm6

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## References (plain Lean)

`gemm7_term` is the magnitude one code contributes: the absolute value of its `i64` product with the
block scale `scales[j / 32]`, as an integer. `gemm7_sumFrom` is the sum over the code indices
`lo, ..., lo + k - 1` and `gemm7_magnitude` the sum over every code. -/

/-- The magnitude contribution of the code `code` under the block scale `scale`. -/
def gemm7_term (scale : Std.I64) (code : Std.I8) : Int :=
  ((scale.val * code.val).natAbs : Int)

/-- The sum of the `k` magnitude terms at code indices `lo, ..., lo + k - 1`. -/
def gemm7_sumFrom (scales : List Std.I64) (codes : List Std.I8) (lo : Nat) : Nat → Int
  | 0 => 0
  | k + 1 =>
    gemm7_sumFrom scales codes lo k +
      gemm7_term (scales.getD ((lo + k) / 32) 0#i64) (codes.getD (lo + k) 0#i8)

/-- The magnitude of every code under its block scale. -/
def gemm7_magnitude (scales : List Std.I64) (codes : List Std.I8) : Int :=
  gemm7_sumFrom scales codes 0 codes.length

/-- The joint domain of one loop step at code index `j`: block `j / 32` is present and the `i64`
product of its scale with code `j` fits `i64`. -/
def gemm7_inRange (self : chart.QuantAct) (j : Nat) : Prop :=
  j / 32 < self.scales.val.length ∧
  I64.min ≤ (self.scales.val.getD (j / 32) 0#i64).val * (self.codes.val.getD j 0#i8).val ∧
  (self.scales.val.getD (j / 32) 0#i64).val * (self.codes.val.getD j 0#i8).val ≤ I64.max

/-- The `i128` with value `v`; `gemm7_i128_val` is its value on the in-range arguments. -/
def gemm7_i128 (v : Int) : Std.I128 := ⟨BitVec.ofInt 128 v⟩

/-! ## Value lemmas -/

/-- `chart.ACT_QK` is 32, the block stride `idx / 32`. -/
theorem gemm7_ACT_QK_val : (chart.ACT_QK : Std.Usize).val = 32 := by
  unfold chart.ACT_QK
  rfl

/-- In-bounds `List.getD` of an `i8` list is its `getElem`. -/
theorem gemm7_getD_i8 (l : List Std.I8) (i : Nat) (h : i < l.length) :
    l.getD i 0#i8 = l[i]'h := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

/-- The `getD` term at an in-range index is the in-bounds term. -/
theorem gemm7_term_getD (self : chart.QuantAct) (j : Nat)
    (hs : j / 32 < self.scales.val.length) (hc : j < self.codes.val.length) :
    gemm7_term (self.scales.val.getD (j / 32) 0#i64) (self.codes.val.getD j 0#i8) =
      gemm7_term (self.scales.val[j / 32]'hs) (self.codes.val[j]'hc) := by
  unfold gemm7_term
  rw [gemm7_getD_i8 self.codes.val j hc, gemm6_getD_getElem self.scales.val (j / 32) hs]

/-- One unfolding of `gemm7_sumFrom` at the successor index. -/
theorem gemm7_sumFrom_succ (scales : List Std.I64) (codes : List Std.I8) (lo k : Nat) :
    gemm7_sumFrom scales codes lo (k + 1) =
      gemm7_sumFrom scales codes lo k +
        gemm7_term (scales.getD ((lo + k) / 32) 0#i64) (codes.getD (lo + k) 0#i8) := rfl

/-- Every magnitude term is nonnegative. -/
theorem gemm7_term_nonneg (scale : Std.I64) (code : Std.I8) : 0 ≤ gemm7_term scale code := by
  unfold gemm7_term
  exact Int.natCast_nonneg _

/-- The partial magnitudes are nonnegative. -/
theorem gemm7_sumFrom_nonneg (scales : List Std.I64) (codes : List Std.I8) (lo k : Nat) :
    0 ≤ gemm7_sumFrom scales codes lo k := by
  induction k with
  | zero => exact Int.le_refl 0
  | succ k ih =>
    rw [gemm7_sumFrom_succ]
    exact Int.add_nonneg ih (gemm7_term_nonneg _ _)

/-- The partial magnitudes are monotone in the window length. -/
theorem gemm7_sumFrom_mono (scales : List Std.I64) (codes : List Std.I8) (lo : Nat) :
    ∀ {k₁ k₂ : Nat}, k₁ ≤ k₂ →
      gemm7_sumFrom scales codes lo k₁ ≤ gemm7_sumFrom scales codes lo k₂ := by
  intro k₁ k₂ h
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  induction d with
  | zero => simp
  | succ d ih =>
    rw [Nat.add_succ, gemm7_sumFrom_succ]
    have hterm := gemm7_term_nonneg (scales.getD ((lo + (k₁ + d)) / 32) 0#i64)
      (codes.getD (lo + (k₁ + d)) 0#i8)
    omega

/-- Past the end of `codes` the terms vanish, so the window sum is unchanged. -/
theorem gemm7_sumFrom_beyond (scales : List Std.I64) (codes : List Std.I8) (lo k : Nat)
    (h : codes.length ≤ lo + k) :
    gemm7_sumFrom scales codes lo (k + 1) = gemm7_sumFrom scales codes lo k := by
  rw [gemm7_sumFrom_succ]
  have hz : codes.getD (lo + k) 0#i8 = 0#i8 := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by simpa using h)]
    rfl
  rw [hz]
  unfold gemm7_term
  rw [show (0#i8 : Std.I8).val = 0 from rfl]
  simp

/-- The `i128` literal's value on the in-range arguments. -/
theorem gemm7_i128_val {v : Int} (h1 : I128.min ≤ v) (h2 : v ≤ I128.max) :
    (gemm7_i128 v).val = v := by
  show (BitVec.ofInt 128 v).toInt = v
  apply BitVec.toInt_ofInt_eq_self (w := 128)
  · norm_num
  · rw [I128.min_eq] at h1
    norm_num
    exact h1
  · rw [I128.max_eq] at h2
    norm_num
    omega

/-- `I64.checked_mul` on its success arm, as a value. -/
theorem gemm7_checked_mul_some {a b : Std.I64}
    (h : I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max) :
    ∃ z, I64.checked_mul a b = some z ∧ z.val = a.val * b.val := by
  have hspec : (match I64.checked_mul a b with
      | some z => I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max ∧
          z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => ¬ (I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max)) :=
    I64.checked_mul_bv_spec a b
  cases hm : I64.checked_mul a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

/-- `I128.checked_add` on its success arm, as a value. -/
theorem gemm7_checked_add_some {a b : Std.I128}
    (h : I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max) :
    ∃ z, I128.checked_add a b = some z ∧ z.val = a.val + b.val := by
  have hspec : (match I128.checked_add a b with
      | some z => I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max ∧
          z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max)) :=
    I128.checked_add_bv_spec a b
  cases hm : I128.checked_add a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

/-- `I64.checked_mul` on its refusal arm. -/
theorem gemm7_checked_mul_none {a b : Std.I64}
    (h : ¬ (I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max)) :
    I64.checked_mul a b = none := by
  have hspec : (match I64.checked_mul a b with
      | some z => I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max ∧
          z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => ¬ (I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max)) :=
    I64.checked_mul_bv_spec a b
  cases hm : I64.checked_mul a b with
  | none => rfl
  | some z => rw [hm] at hspec; exact absurd ⟨hspec.1, hspec.2.1⟩ h

/-- `I128.checked_add` on its refusal arm. -/
theorem gemm7_checked_add_none {a b : Std.I128}
    (h : ¬ (I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max)) :
    I128.checked_add a b = none := by
  have hspec : (match I128.checked_add a b with
      | some z => I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max ∧
          z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ I128.max)) :=
    I128.checked_add_bv_spec a b
  cases hm : I128.checked_add a b with
  | none => rfl
  | some z => rw [hm] at hspec; exact absurd ⟨hspec.1, hspec.2.1⟩ h

/-- `i128::abs` never overflows on an `i64`-ranged argument and is the value absolute value. -/
theorem gemm7_abs_spec (x : Std.I128) (hlo : I64.min ≤ x.val) (hhi : x.val ≤ I64.max) :
    core.num.I128.abs x ⦃ r => r.val = (x.val.natAbs : Int) ⦄ := by
  have hne : x.val ≠ I128.min := by
    intro h
    rw [h, I128.min_eq] at hlo
    rw [I64.min_eq] at hlo
    norm_num at hlo
  have hb1 : I128.min ≤ (x.val.natAbs : Int) := by
    have h := I64.min
    have hz : I128.min ≤ (0 : Int) := by rw [I128.min_eq]; norm_num
    omega
  have hb2 : (x.val.natAbs : Int) ≤ I128.max := by
    rw [I64.min_eq] at hlo
    rw [I64.max_eq] at hhi
    rw [I128.max_eq]
    rcases Int.natAbs_eq x.val with h | h
    · omega
    · omega
  rw [show core.num.I128.abs x = ok (⟨BitVec.ofInt 128 x.val.natAbs⟩ : Std.I128) from by
        unfold core.num.I128.abs
        rw [if_neg hne]]
  simp only [WP.spec_ok]
  exact gemm7_i128_val hb1 hb2

/-! ## One loop step

The body reads the block scale and the code, takes both checked operations on their success arms and
returns the accumulator advanced by the block's magnitude term. -/

/-- One body step of `chart.QuantAct.magnitude_loop`: on `idx < codes.len`, a present block scale and
a fitting `i64` product, the body returns `cont` with the accumulator advanced by `gemm7_term`. -/
theorem gemm7_body_cont (self : chart.QuantAct) (acc : Std.I128) (idx : Std.Usize)
    (hlt : idx.val < self.codes.val.length)
    (hscale : idx.val / 32 < self.scales.val.length)
    (hfit : I64.min ≤ (self.scales.val[idx.val / 32]'hscale).val *
        (self.codes.val[idx.val]'hlt).val ∧
      (self.scales.val[idx.val / 32]'hscale).val * (self.codes.val[idx.val]'hlt).val ≤ I64.max)
    (hadd : I128.min ≤ acc.val + gemm7_term (self.scales.val[idx.val / 32]'hscale)
          (self.codes.val[idx.val]'hlt) ∧
      acc.val + gemm7_term (self.scales.val[idx.val / 32]'hscale)
          (self.codes.val[idx.val]'hlt) ≤ I128.max) :
    ∃ (total : Std.I128) (idx1 : Std.Usize),
      chart.QuantAct.magnitude_loop.body self acc idx = ok (ControlFlow.cont (total, idx1)) ∧
      idx1.val = idx.val + 1 ∧
      total.val = acc.val + gemm7_term (self.scales.val[idx.val / 32]'hscale)
        (self.codes.val[idx.val]'hlt) := by
  have hQK : (chart.ACT_QK : Std.Usize).val = 32 := gemm7_ACT_QK_val
  have hnz : (↑chart.ACT_QK : Nat) ≠ 0 := by rw [hQK]; norm_num
  obtain ⟨i1, hi1, hi1v⟩ :=
    WP.spec_imp_exists (Usize.div_spec (x := idx) (y := chart.ACT_QK) hnz)
  have hi1val : i1.val = idx.val / 32 := by rw [hi1v, hQK]
  have hget : core.slice.Slice.get (core.slice.index.SliceIndexUsizeSlice Std.I64)
      (alloc.vec.Vec.deref self.scales) i1
      = ok (some (self.scales.val[idx.val / 32]'hscale)) := by
    show ok ((alloc.vec.Vec.deref self.scales)[i1]?) = ok (some _)
    rw [Slice.getElem?_Usize_eq, alloc.vec.Vec.deref, Slice.from_val, hi1val,
      List.getElem?_eq_getElem hscale]
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec self.codes idx hlt)
  have hi2C : i2 = self.codes.val[idx.val]'hlt := hi2v
  have hfitMul : I64.min ≤ (self.scales.val[idx.val / 32]'hscale).val *
        (core.convert.num.FromI64I8.from i2).val ∧
      (self.scales.val[idx.val / 32]'hscale).val * (core.convert.num.FromI64I8.from i2).val ≤
        I64.max := by
    rw [core.convert.num.FromI64I8.from_val_eq, hi2C]
    exact hfit
  obtain ⟨v, hmul, hvval⟩ := gemm7_checked_mul_some hfitMul
  have hbnd : I64.min ≤ (core.convert.num.FromI128I64.from v).val ∧
      (core.convert.num.FromI128I64.from v).val ≤ I64.max := by
    rw [core.convert.num.FromI128I64.from_val_eq, hvval]
    exact hfitMul
  obtain ⟨m, habs, hmv⟩ :=
    WP.spec_imp_exists (gemm7_abs_spec (core.convert.num.FromI128I64.from v) hbnd.1 hbnd.2)
  have hmterm : m.val = gemm7_term (self.scales.val[idx.val / 32]'hscale)
      (self.codes.val[idx.val]'hlt) := by
    rw [hmv, core.convert.num.FromI128I64.from_val_eq, hvval,
      core.convert.num.FromI64I8.from_val_eq, hi2C]
    rfl
  have hadd' : I128.min ≤ acc.val + m.val ∧ acc.val + m.val ≤ I128.max := by
    rw [hmterm]
    exact hadd
  obtain ⟨t, ht, htv⟩ := gemm7_checked_add_some hadd'
  obtain ⟨i5, hi5, hi5v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := idx) (y := 1#usize) (by scalar_tac))
  refine ⟨t, i5, ?_, ?_, ?_⟩
  · rw [chart.QuantAct.magnitude_loop.body]
    simp only []
    rw [if_pos (by rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]; exact hlt)]
    rw [hi1]
    simp only [lift, bind_tc_ok]
    rw [hget]
    simp only [lift, bind_tc_ok]
    rw [alloc.vec.Vec.index_slice_index, hi2]
    simp only [lift, bind_tc_ok]
    rw [hmul]
    simp only [lift, bind_tc_ok]
    rw [habs]
    simp only [lift, bind_tc_ok]
    rw [ht]
    simp only [lift, bind_tc_ok]
    rw [hi5]
    simp only [lift, bind_tc_ok]
  · rw [hi5v]
    have h1 : ((1#usize : Std.Usize) : Nat) = 1 := rfl
    rw [h1]
  · rw [htv, hmterm]

/-- The body on `idx >= codes.len`: it returns `done` with `Exact acc` and the state unchanged. -/
theorem gemm7_body_done (self : chart.QuantAct) (acc : Std.I128) (idx : Std.Usize)
    (hge : ¬ idx < alloc.vec.Vec.len self.codes) :
    chart.QuantAct.magnitude_loop.body self acc idx
      = ok (ControlFlow.done (chart.ActMagnitude.Exact acc)) := by
  rw [chart.QuantAct.magnitude_loop.body]
  simp only []
  rw [if_neg hge]

/-! ## One loop step, refusal arms

The three ways one step declines to advance: the block scale `idx / 32` is absent (`NoQ32`), the
`i64` product of the block scale with the code overflows (`NoQ32`), or the `i128` accumulation
overflows (`TotalOverflow`). -/

/-- One body step of `chart.QuantAct.magnitude_loop` when the block scale is absent: the body returns
`done NoQ32` without reading the code. -/
theorem gemm7_body_noq32_scale (self : chart.QuantAct) (acc : Std.I128) (idx : Std.Usize)
    (hlt : idx.val < self.codes.val.length)
    (hscale : ¬ idx.val / 32 < self.scales.val.length) :
    chart.QuantAct.magnitude_loop.body self acc idx
      = ok (ControlFlow.done chart.ActMagnitude.NoQ32) := by
  have hQK : (chart.ACT_QK : Std.Usize).val = 32 := gemm7_ACT_QK_val
  have hnz : (↑chart.ACT_QK : Nat) ≠ 0 := by rw [hQK]; norm_num
  obtain ⟨i1, hi1, hi1v⟩ :=
    WP.spec_imp_exists (Usize.div_spec (x := idx) (y := chart.ACT_QK) hnz)
  have hi1val : i1.val = idx.val / 32 := by rw [hi1v, hQK]
  have hlen : self.scales.val.length ≤ idx.val / 32 := Nat.not_lt.mp hscale
  have hget : core.slice.Slice.get (core.slice.index.SliceIndexUsizeSlice Std.I64)
      (alloc.vec.Vec.deref self.scales) i1 = ok none := by
    show ok ((alloc.vec.Vec.deref self.scales)[i1]?) = ok none
    rw [Slice.getElem?_Usize_eq, alloc.vec.Vec.deref, Slice.from_val, hi1val,
      List.getElem?_eq_none hlen]
  rw [chart.QuantAct.magnitude_loop.body]
  simp only []
  rw [if_pos (by rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]; exact hlt)]
  rw [hi1]
  simp only [lift, bind_tc_ok]
  rw [hget]
  simp only [lift, bind_tc_ok]

/-- One body step of `chart.QuantAct.magnitude_loop` when the `i64` product of the block scale with
the code overflows: the body returns `done NoQ32`. -/
theorem gemm7_body_noq32_mul (self : chart.QuantAct) (acc : Std.I128) (idx : Std.Usize)
    (hlt : idx.val < self.codes.val.length)
    (hscale : idx.val / 32 < self.scales.val.length)
    (hover : ¬ (I64.min ≤ (self.scales.val[idx.val / 32]'hscale).val *
          (self.codes.val[idx.val]'hlt).val ∧
        (self.scales.val[idx.val / 32]'hscale).val * (self.codes.val[idx.val]'hlt).val
          ≤ I64.max)) :
    chart.QuantAct.magnitude_loop.body self acc idx
      = ok (ControlFlow.done chart.ActMagnitude.NoQ32) := by
  have hQK : (chart.ACT_QK : Std.Usize).val = 32 := gemm7_ACT_QK_val
  have hnz : (↑chart.ACT_QK : Nat) ≠ 0 := by rw [hQK]; norm_num
  obtain ⟨i1, hi1, hi1v⟩ :=
    WP.spec_imp_exists (Usize.div_spec (x := idx) (y := chart.ACT_QK) hnz)
  have hi1val : i1.val = idx.val / 32 := by rw [hi1v, hQK]
  have hget : core.slice.Slice.get (core.slice.index.SliceIndexUsizeSlice Std.I64)
      (alloc.vec.Vec.deref self.scales) i1
      = ok (some (self.scales.val[idx.val / 32]'hscale)) := by
    show ok ((alloc.vec.Vec.deref self.scales)[i1]?) = ok (some _)
    rw [Slice.getElem?_Usize_eq, alloc.vec.Vec.deref, Slice.from_val, hi1val,
      List.getElem?_eq_getElem hscale]
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec self.codes idx hlt)
  have hi2C : i2 = self.codes.val[idx.val]'hlt := hi2v
  have hover' : ¬ (I64.min ≤ (self.scales.val[idx.val / 32]'hscale).val *
        (core.convert.num.FromI64I8.from i2).val ∧
      (self.scales.val[idx.val / 32]'hscale).val *
        (core.convert.num.FromI64I8.from i2).val ≤ I64.max) := by
    rw [core.convert.num.FromI64I8.from_val_eq, hi2C]
    exact hover
  have hmulnone : I64.checked_mul (self.scales.val[idx.val / 32]'hscale)
      (core.convert.num.FromI64I8.from i2) = none := gemm7_checked_mul_none hover'
  rw [chart.QuantAct.magnitude_loop.body]
  simp only []
  rw [if_pos (by rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]; exact hlt)]
  rw [hi1]
  simp only [lift, bind_tc_ok]
  rw [hget]
  simp only [lift, bind_tc_ok]
  rw [alloc.vec.Vec.index_slice_index, hi2]
  simp only [lift, bind_tc_ok]
  rw [hmulnone]

/-- One body step of `chart.QuantAct.magnitude_loop` when the `i128` accumulation overflows: the
body returns `done TotalOverflow`. -/
theorem gemm7_body_totaloverflow (self : chart.QuantAct) (acc : Std.I128) (idx : Std.Usize)
    (hlt : idx.val < self.codes.val.length)
    (hscale : idx.val / 32 < self.scales.val.length)
    (hfit : I64.min ≤ (self.scales.val[idx.val / 32]'hscale).val *
        (self.codes.val[idx.val]'hlt).val ∧
      (self.scales.val[idx.val / 32]'hscale).val * (self.codes.val[idx.val]'hlt).val ≤ I64.max)
    (hno : ¬ (I128.min ≤ acc.val + gemm7_term (self.scales.val[idx.val / 32]'hscale)
          (self.codes.val[idx.val]'hlt) ∧
        acc.val + gemm7_term (self.scales.val[idx.val / 32]'hscale)
          (self.codes.val[idx.val]'hlt) ≤ I128.max)) :
    chart.QuantAct.magnitude_loop.body self acc idx
      = ok (ControlFlow.done chart.ActMagnitude.TotalOverflow) := by
  have hQK : (chart.ACT_QK : Std.Usize).val = 32 := gemm7_ACT_QK_val
  have hnz : (↑chart.ACT_QK : Nat) ≠ 0 := by rw [hQK]; norm_num
  obtain ⟨i1, hi1, hi1v⟩ :=
    WP.spec_imp_exists (Usize.div_spec (x := idx) (y := chart.ACT_QK) hnz)
  have hi1val : i1.val = idx.val / 32 := by rw [hi1v, hQK]
  have hget : core.slice.Slice.get (core.slice.index.SliceIndexUsizeSlice Std.I64)
      (alloc.vec.Vec.deref self.scales) i1
      = ok (some (self.scales.val[idx.val / 32]'hscale)) := by
    show ok ((alloc.vec.Vec.deref self.scales)[i1]?) = ok (some _)
    rw [Slice.getElem?_Usize_eq, alloc.vec.Vec.deref, Slice.from_val, hi1val,
      List.getElem?_eq_getElem hscale]
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec self.codes idx hlt)
  have hi2C : i2 = self.codes.val[idx.val]'hlt := hi2v
  have hfitMul : I64.min ≤ (self.scales.val[idx.val / 32]'hscale).val *
        (core.convert.num.FromI64I8.from i2).val ∧
      (self.scales.val[idx.val / 32]'hscale).val * (core.convert.num.FromI64I8.from i2).val ≤
        I64.max := by
    rw [core.convert.num.FromI64I8.from_val_eq, hi2C]
    exact hfit
  obtain ⟨v, hmul, hvval⟩ := gemm7_checked_mul_some hfitMul
  have hbnd : I64.min ≤ (core.convert.num.FromI128I64.from v).val ∧
      (core.convert.num.FromI128I64.from v).val ≤ I64.max := by
    rw [core.convert.num.FromI128I64.from_val_eq, hvval]
    exact hfitMul
  obtain ⟨m, habs, hmv⟩ :=
    WP.spec_imp_exists (gemm7_abs_spec (core.convert.num.FromI128I64.from v) hbnd.1 hbnd.2)
  have hmterm : m.val = gemm7_term (self.scales.val[idx.val / 32]'hscale)
      (self.codes.val[idx.val]'hlt) := by
    rw [hmv, core.convert.num.FromI128I64.from_val_eq, hvval,
      core.convert.num.FromI64I8.from_val_eq, hi2C]
    rfl
  have hadd' : ¬ (I128.min ≤ acc.val + m.val ∧ acc.val + m.val ≤ I128.max) := by
    rw [hmterm]; exact hno
  have haddnone : I128.checked_add acc m = none := gemm7_checked_add_none hadd'
  rw [chart.QuantAct.magnitude_loop.body]
  simp only []
  rw [if_pos (by rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]; exact hlt)]
  rw [hi1]
  simp only [lift, bind_tc_ok]
  rw [hget]
  simp only [lift, bind_tc_ok]
  rw [alloc.vec.Vec.index_slice_index, hi2]
  simp only [lift, bind_tc_ok]
  rw [hmul]
  simp only [lift, bind_tc_ok]
  rw [habs]
  simp only [lift, bind_tc_ok]
  rw [haddnone]

/-! ## The magnitude loop -/

/-- `chart.QuantAct.magnitude_loop` returns `Exact` of the code window's magnitude: on the domain
where every block scale is present, every `i64` product fits and the accumulation stays in `i128`,
the loop from `(acc0, idx0)` terminates at `codes.len` with the accumulator equal to
`acc0 + gemm7_sumFrom idx0 (codes.len - idx0)`.

The loop's measure is `codes.len - idx.val`; the invariant is the running window sum plus the index
bounds that keep the block read and the accumulator add on their success arms. -/
@[step]
theorem word_quantact_magnitude_loop_ok (self : chart.QuantAct) (acc0 : Std.I128) (idx0 : Std.Usize)
    (hin : ∀ j, idx0.val ≤ j → j < self.codes.val.length → gemm7_inRange self j)
    (hacc0 : 0 ≤ acc0.val)
    (hsum : acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
        (self.codes.val.length - idx0.val) ≤ I128.max)
    (hidx0 : idx0.val ≤ self.codes.val.length) :
    chart.QuantAct.magnitude_loop self acc0 idx0 ⦃ r =>
      r = chart.ActMagnitude.Exact (gemm7_i128 (acc0.val + gemm7_sumFrom self.scales.val
        self.codes.val idx0.val (self.codes.val.length - idx0.val))) ⦄ := by
  unfold chart.QuantAct.magnitude_loop
  apply loop.spec_decr_nat
    (measure := fun st : Std.I128 × Std.Usize => self.codes.val.length - st.2.val)
    (inv := fun st : Std.I128 × Std.Usize =>
      idx0.val ≤ st.2.val ∧ st.2.val ≤ self.codes.val.length ∧
      st.1.val = acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
        (st.2.val - idx0.val))
    (post := fun r : chart.ActMagnitude =>
      r = chart.ActMagnitude.Exact (gemm7_i128 (acc0.val + gemm7_sumFrom self.scales.val
        self.codes.val idx0.val (self.codes.val.length - idx0.val))))
  · rintro ⟨a, i⟩ ⟨hile, hile2, hval⟩
    dsimp only at *
    by_cases hlt : i.val < self.codes.val.length
    · have hin_i : gemm7_inRange self i.val := hin i.val hile hlt
      have hscale : i.val / 32 < self.scales.val.length := hin_i.1
      have hsv : (self.scales.val.getD (i.val / 32) 0#i64).val =
          (self.scales.val[i.val / 32]'hscale).val := by
        rw [gemm6_getD_getElem self.scales.val (i.val / 32) hscale]
      have hcv : (self.codes.val.getD i.val 0#i8).val = (self.codes.val[i.val]'hlt).val := by
        rw [gemm7_getD_i8 self.codes.val i.val hlt]
      have hfit : I64.min ≤ (self.scales.val[i.val / 32]'hscale).val *
            (self.codes.val[i.val]'hlt).val ∧
          (self.scales.val[i.val / 32]'hscale).val * (self.codes.val[i.val]'hlt).val ≤
            I64.max := by
        have h1 := hin_i.2.1
        have h2 := hin_i.2.2
        rw [hsv, hcv] at h1 h2
        exact ⟨h1, h2⟩
      have hsucc : a.val + gemm7_term (self.scales.val[i.val / 32]'hscale)
            (self.codes.val[i.val]'hlt) =
          acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
            (i.val - idx0.val + 1) := by
        rw [hval, ← gemm7_term_getD self i.val hscale hlt, gemm7_sumFrom_succ,
          show idx0.val + (i.val - idx0.val) = i.val from by omega, Int.add_assoc]
      have hadd : I128.min ≤ a.val + gemm7_term (self.scales.val[i.val / 32]'hscale)
            (self.codes.val[i.val]'hlt) ∧
          a.val + gemm7_term (self.scales.val[i.val / 32]'hscale)
            (self.codes.val[i.val]'hlt) ≤ I128.max := by
        constructor
        · have h1 : 0 ≤ a.val := by
            rw [hval]
            exact Int.add_nonneg hacc0 (gemm7_sumFrom_nonneg self.scales.val self.codes.val
              idx0.val (i.val - idx0.val))
          have h2 := gemm7_term_nonneg (self.scales.val[i.val / 32]'hscale)
            (self.codes.val[i.val]'hlt)
          have h3 : I128.min ≤ (0 : Int) := by rw [I128.min_eq]; norm_num
          omega
        · rw [hsucc]
          have hmono := gemm7_sumFrom_mono self.scales.val self.codes.val idx0.val
            (k₁ := i.val - idx0.val + 1) (k₂ := self.codes.val.length - idx0.val) (by omega)
          omega
      obtain ⟨total, i1, hbody, hi1v, htot⟩ := gemm7_body_cont self a i hlt hscale hfit hadd
      rw [hbody, WP.spec_ok]
      dsimp only
      refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
      · omega
      · omega
      · rw [htot, hval, hi1v]
        rw [show i.val + 1 - idx0.val = (i.val - idx0.val) + 1 from by omega]
        rw [gemm7_sumFrom_succ]
        rw [show idx0.val + (i.val - idx0.val) = i.val from by omega]
        rw [gemm7_term_getD self i.val hscale hlt]
        ring
      · omega
    · have hge : self.codes.val.length ≤ i.val := by omega
      have hnot : ¬ i < alloc.vec.Vec.len self.codes := by
        rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]
        exact not_lt.mpr hge
      rw [gemm7_body_done self a i hnot, WP.spec_ok]
      have hlen : i.val = self.codes.val.length := le_antisymm hile2 hge
      have hV : a.val = acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
          (self.codes.val.length - idx0.val) := by
        rw [hval, hlen]
      have hb1 : I128.min ≤ acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
          (self.codes.val.length - idx0.val) := by
        have h1 := gemm7_sumFrom_nonneg self.scales.val self.codes.val idx0.val
          (self.codes.val.length - idx0.val)
        have h2 : I128.min ≤ (0 : Int) := by rw [I128.min_eq]; norm_num
        omega
      have heq : a = gemm7_i128 (acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
          (self.codes.val.length - idx0.val)) := by
        apply IScalar.eq_of_val_eq
        rw [gemm7_i128_val hb1 hsum]
        exact hV
      rw [heq]
  · refine ⟨le_refl _, hidx0, ?_⟩
    simp [gemm7_sumFrom, Nat.sub_self]

/-- `chart.QuantAct.magnitude` is the whole-code magnitude on the same domain. -/
theorem word_quantact_magnitude_ok (self : chart.QuantAct)
    (hin : ∀ j, j < self.codes.val.length → gemm7_inRange self j)
    (hsum : gemm7_sumFrom self.scales.val self.codes.val 0 self.codes.val.length ≤ I128.max) :
    chart.QuantAct.magnitude self ⦃ r => r = chart.ActMagnitude.Exact
      (gemm7_i128 (gemm7_sumFrom self.scales.val self.codes.val 0 self.codes.val.length)) ⦄ := by
  rw [show chart.QuantAct.magnitude self
      = chart.QuantAct.magnitude_loop self 0#i128 0#usize from rfl]
  have h := word_quantact_magnitude_loop_ok self 0#i128 0#usize
    (by intro j hj hlt; exact hin j hlt)
    (by rw [show (0#i128 : Std.I128).val = 0 from rfl])
    (by
      rw [show (0#i128 : Std.I128).val = 0 from rfl,
        show (0#usize : Std.Usize).val = 0 from rfl, Nat.sub_zero]
      simpa using hsum)
    (by rw [show (0#usize : Std.Usize).val = 0 from rfl]; exact Nat.zero_le _)
  simpa only [show (0#i128 : Std.I128).val = 0 from rfl,
    show (0#usize : Std.Usize).val = 0 from rfl, Nat.sub_zero, Int.zero_add] using h

/-- `chart.QuantAct.magnitude_loop` returns the refusal `r` when one body step refuses at the first
out-of-domain index `istop`: on the prefix `[idx0, istop)` every block scale is present and every
`i64` product fits, the accumulation stays in `i128`, and at `istop` the body returns `done r` — an
absent block scale or an `i64` product overflow (`NoQ32`) or an `i128` accumulator overflow
(`TotalOverflow`). The loop stops at the refusal and never reads a code beyond `istop`.

The measure is `istop - idx`; the invariant carries the running window sum and the index bounds. -/
@[step]
theorem word_quantact_magnitude_loop_refuses (self : chart.QuantAct) (acc0 : Std.I128)
    (idx0 istop : Std.Usize) (r : chart.ActMagnitude)
    (hle : idx0.val ≤ istop.val)
    (histop : istop.val ≤ self.codes.val.length)
    (hin : ∀ j, idx0.val ≤ j → j < istop.val → gemm7_inRange self j)
    (hacc0 : 0 ≤ acc0.val)
    (hbound : acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
        (istop.val - idx0.val) ≤ I128.max)
    (hstop : ∀ accS : Std.I128, accS.val = acc0.val + gemm7_sumFrom self.scales.val
        self.codes.val idx0.val (istop.val - idx0.val) →
        chart.QuantAct.magnitude_loop.body self accS istop = ok (ControlFlow.done r)) :
    chart.QuantAct.magnitude_loop self acc0 idx0 ⦃ res => res = r ⦄ := by
  unfold chart.QuantAct.magnitude_loop
  apply loop.spec_decr_nat
    (measure := fun st : Std.I128 × Std.Usize => istop.val - st.2.val)
    (inv := fun st : Std.I128 × Std.Usize =>
      idx0.val ≤ st.2.val ∧ st.2.val ≤ istop.val ∧
      st.1.val = acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
        (st.2.val - idx0.val))
    (post := fun res : chart.ActMagnitude => res = r)
  · rintro ⟨a, i⟩ ⟨hile, hile2, hval⟩
    dsimp only at *
    by_cases hlt : i.val < istop.val
    · have hcode : i.val < self.codes.val.length := by omega
      have hin_i : gemm7_inRange self i.val := hin i.val hile (by omega)
      have hscale : i.val / 32 < self.scales.val.length := hin_i.1
      have hsv : (self.scales.val.getD (i.val / 32) 0#i64).val =
          (self.scales.val[i.val / 32]'hscale).val := by
        rw [gemm6_getD_getElem self.scales.val (i.val / 32) hscale]
      have hcv : (self.codes.val.getD i.val 0#i8).val = (self.codes.val[i.val]'hcode).val := by
        rw [gemm7_getD_i8 self.codes.val i.val hcode]
      have hfit : I64.min ≤ (self.scales.val[i.val / 32]'hscale).val *
            (self.codes.val[i.val]'hcode).val ∧
          (self.scales.val[i.val / 32]'hscale).val * (self.codes.val[i.val]'hcode).val ≤
            I64.max := by
        have h1 := hin_i.2.1
        have h2 := hin_i.2.2
        rw [hsv, hcv] at h1 h2
        exact ⟨h1, h2⟩
      have hsucc : a.val + gemm7_term (self.scales.val[i.val / 32]'hscale)
            (self.codes.val[i.val]'hcode) =
          acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
            (i.val - idx0.val + 1) := by
        rw [hval, ← gemm7_term_getD self i.val hscale hcode, gemm7_sumFrom_succ,
          show idx0.val + (i.val - idx0.val) = i.val from by omega, Int.add_assoc]
      have hadd : I128.min ≤ a.val + gemm7_term (self.scales.val[i.val / 32]'hscale)
            (self.codes.val[i.val]'hcode) ∧
          a.val + gemm7_term (self.scales.val[i.val / 32]'hscale)
            (self.codes.val[i.val]'hcode) ≤ I128.max := by
        constructor
        · have h1 : 0 ≤ a.val := by
            rw [hval]
            exact Int.add_nonneg hacc0 (gemm7_sumFrom_nonneg self.scales.val self.codes.val
              idx0.val (i.val - idx0.val))
          have h2 := gemm7_term_nonneg (self.scales.val[i.val / 32]'hscale)
            (self.codes.val[i.val]'hcode)
          have h3 : I128.min ≤ (0 : Int) := by rw [I128.min_eq]; norm_num
          omega
        · rw [hsucc]
          have hmono := gemm7_sumFrom_mono self.scales.val self.codes.val idx0.val
            (k₁ := i.val - idx0.val + 1) (k₂ := istop.val - idx0.val) (by omega)
          omega
      obtain ⟨total, i1, hbody, hi1v, htot⟩ := gemm7_body_cont self a i hcode hscale hfit hadd
      rw [hbody, WP.spec_ok]
      dsimp only
      refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
      · omega
      · omega
      · rw [htot, hval, hi1v]
        rw [show i.val + 1 - idx0.val = (i.val - idx0.val) + 1 from by omega]
        rw [gemm7_sumFrom_succ]
        rw [show idx0.val + (i.val - idx0.val) = i.val from by omega]
        rw [gemm7_term_getD self i.val hscale hcode]
        ring
      · omega
    · have heqi : i.val = istop.val := le_antisymm hile2 (not_lt.mp hlt)
      have haval : a.val = acc0.val + gemm7_sumFrom self.scales.val self.codes.val idx0.val
          (istop.val - idx0.val) := by rw [hval, heqi]
      have hi : i = istop := by apply UScalar.eq_of_val_eq; exact heqi
      rw [hi, hstop a haval]
      simp only [WP.spec_ok]
  · refine ⟨le_refl _, hle, ?_⟩
    simp [gemm7_sumFrom, Nat.sub_self]

/-! ## Non-vacuity -/

/-- Non-vacuity of the term reference: `|(-3) * 5| = 15`. -/
example : gemm7_term ((-3)#i64) (5#i8) = 15 := by
  unfold gemm7_term
  decide

/-- Non-vacuity of the term reference: a zero code contributes nothing. -/
example : gemm7_term (3#i64) (0#i8) = 0 := by
  unfold gemm7_term
  decide

/-- Non-vacuity of the window sum: `|3·2| + |3·1| = 9` over a two-code window whose two indices
`0` and `1` share the block scale `scales[1 / 32] = scales[0]`. -/
example : gemm7_sumFrom ([3#i64, 1#i64] : List Std.I64) ([2#i8, 1#i8] : List Std.I8) 0 2 = 9 := by
  simp [gemm7_sumFrom, gemm7_term]

/-- Non-vacuity of the whole-code magnitude: the code index selects its block scale `j / 32`. -/
example : gemm7_magnitude ([2#i64] : List Std.I64) ([3#i8] : List Std.I8) = 6 := by
  simp only [gemm7_magnitude, gemm7_sumFrom, gemm7_term, List.getD_cons_zero, List.length_cons,
    List.length_nil, Nat.reduceAdd, Nat.zero_div, Nat.add_zero]
  decide

/-- Non-vacuity of the hypothesis bundle: `codes` of length 1 with `scales` of length 1 satisfies
`gemm7_inRange` at every index, and the accumulated sum is in `i128`. -/
example : ∃ (scales : List Std.I64) (codes : List Std.I8),
    (∀ j, j < codes.length →
      j / 32 < scales.length ∧
      I64.min ≤ (scales.getD (j / 32) 0#i64).val * (codes.getD j 0#i8).val ∧
      (scales.getD (j / 32) 0#i64).val * (codes.getD j 0#i8).val ≤ I64.max) ∧
    gemm7_sumFrom scales codes 0 codes.length ≤ I128.max := by
  refine ⟨([1#i64] : List Std.I64), ([1#i8] : List Std.I8), ?_, ?_⟩
  · intro j hj
    have hj0 : j = 0 := by
      simp only [List.length_cons, List.length_nil] at hj
      omega
    subst hj0
    have hs : ([1#i64] : List Std.I64).getD (0 / 32) 0#i64 = 1#i64 := by
      simp only [Nat.zero_div, List.getD_cons_zero]
    have hc : ([1#i8] : List Std.I8).getD 0 0#i8 = 1#i8 := by
      simp only [List.getD_cons_zero]
    have hv : ((1#i64 : Std.I64).val * (1#i8 : Std.I8).val) = 1 := by decide
    refine ⟨by norm_num, ?_, ?_⟩
    · rw [hs, hc, hv, I64.min_eq]
      norm_num
    · rw [hs, hc, hv, I64.max_eq]
      norm_num
  · have h2 : ([1#i64] : List Std.I64).getD (0 / 32) 0#i64 = 1#i64 := by
      simp only [Nat.zero_div, List.getD_cons_zero]
    have h3 : ([1#i8] : List Std.I8).getD 0 0#i8 = 1#i8 := by
      simp only [List.getD_cons_zero]
    have hv : ((1#i64 : Std.I64).val * (1#i8 : Std.I8).val) = 1 := by decide
    simp only [gemm7_sumFrom, gemm7_term, List.length_cons, List.length_nil, Nat.reduceAdd,
      Nat.zero_div, Nat.add_zero, List.getD_cons_zero, hv]
    rw [I128.max_eq]
    norm_num

/-- Non-vacuity of the `i64` product-overflow refusal precondition: the block scale `I64.max` times
the code `2` does not fit `i64`. -/
example : ¬ (I64.min ≤ I64.max * ((2#i8 : Std.I8)).val ∧
    I64.max * ((2#i8 : Std.I8)).val ≤ I64.max) := by
  have hv : ((2#i8 : Std.I8)).val = 2 := by decide
  rw [hv, I64.max_eq, I64.min_eq]
  norm_num

/-- Non-vacuity of the `i128` accumulator-overflow refusal precondition: `I128.max + 1` does not fit
`i128`. -/
example : ¬ (I128.min ≤ I128.max + ((1#i128 : Std.I128)).val ∧
    I128.max + ((1#i128 : Std.I128)).val ≤ I128.max) := by
  have hv : ((1#i128 : Std.I128)).val = 1 := by decide
  rw [hv, I128.max_eq, I128.min_eq]
  norm_num

/-- Non-vacuity of the absent-scale body refusal: one step over a code whose block scale is absent
returns `NoQ32` without reading the code. -/
example : chart.QuantAct.magnitude_loop.body
      { scales := alloc.vec.Vec.from ([] : List Std.I64) (by simp)
        codes := alloc.vec.Vec.from ([5#i8] : List Std.I8) (by scalar_tac) }
      0#i128 0#usize = ok (ControlFlow.done chart.ActMagnitude.NoQ32) := by
  apply gemm7_body_noq32_scale
  · simp
  · simp

/-- Non-vacuity of the loop refusal: the loop over one code with no block scale returns `NoQ32` at
`istop = 0`, never advancing. -/
example :
    chart.QuantAct.magnitude_loop
      { scales := alloc.vec.Vec.from ([] : List Std.I64) (by simp)
        codes := alloc.vec.Vec.from ([5#i8] : List Std.I8) (by scalar_tac) }
      0#i128 0#usize ⦃ res => res = chart.ActMagnitude.NoQ32 ⦄ := by
  apply word_quantact_magnitude_loop_refuses
    (istop := 0#usize) (r := chart.ActMagnitude.NoQ32)
  · simp
  · simp
  · intro j hj hj2; omega
  · simp
  · rw [show (0#i128 : Std.I128).val = 0 from rfl]
    simp only [alloc.vec.Vec.from_val, Usize.ofNatCore_val_eq, Nat.sub_self, List.length_cons,
      List.length_nil, gemm7_sumFrom]
    rw [I128.max_eq]
    norm_num
  · intro accS _
    apply gemm7_body_noq32_scale
    · simp
    · simp

end apeiron_word
