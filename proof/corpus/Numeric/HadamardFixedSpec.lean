/-
Normalized exact-integer Walsh–Hadamard refinement (`APEIRON-TERNARY-W3-T6c`; spec
`docs/epics/ternary-hadamard.md`, contract rows "integer FWHT" and "normalized FWHT").

Extends `Numeric.HadamardFixed` (items 1–3, 6: the rounding law, the `C` table and the validation
refusals). This module lands items 4 and 5:

* `fwht_fixed_loop3_spec`, the normalization-loop law, and `fwht_fixed_spec`, the composed
  `hadamard.fwht_fixed` law on the checked signs domain;
* `fwht_fixed_refines`, the real-arithmetic error bound against `fwhtRef k (s⊙x)`, and its
  aligned-segment form through `fwhtRef_segment`/`fwhtRef_getElem`.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom` or `decide`
over an extracted body appears.
-/
import Numeric.HadamardFixed
import Numeric.HadamardStage

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word
open Aeneas Aeneas.Std Result
noncomputable section

theorem fromI64_max_val :
    (core.convert.num.FromI128I64.from core.num.I64.MAX).val = (2 : ℤ) ^ 63 - 1 := by
  rw [core.convert.num.FromI128I64.from_val_eq]
  have h : core.num.I64.MAX.val = (2 : ℤ) ^ 63 - 1 := by decide
  rw [h]

theorem fromI64_min_val :
    (core.convert.num.FromI128I64.from core.num.I64.MIN).val = -(2 : ℤ) ^ 63 := by
  rw [core.convert.num.FromI128I64.from_val_eq]
  have h : core.num.I64.MIN.val = -(2 : ℤ) ^ 63 := by decide
  rw [h]

theorem cast_i64_val (r : I128)
    (hlo : -((2 : ℤ) ^ 63) ≤ r.val) (hhi : r.val ≤ (2 : ℤ) ^ 63 - 1) :
    (IScalar.cast .I64 r).val = r.val := by
  rw [IScalar.cast_val_eq]
  norm_num [IScalarTy.numBits]
  exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 r.val
    (by norm_num [IScalarTy.numBits]; omega)
    (by norm_num [IScalarTy.numBits]; omega)

def NormWinAt (coeff v : ℤ) : Prop :=
  I128.min ≤ v * coeff ∧ v * coeff ≤ I128.max ∧
  |v * coeff| < (2 : ℤ) ^ 126 ∧
  -((2 : ℤ) ^ 63) ≤ roundQ (v * coeff) ∧ roundQ (v * coeff) ≤ (2 : ℤ) ^ 63 - 1

theorem normAt_set (y x1 : Slice I64) (coeff : ℤ) (lo : ℕ) (i5 : I64) (iv : Usize)
    (hinv : normAt y coeff x1 lo iv.val)
    (hloiv : lo ≤ iv.val)
    (hivlt : iv.val < y.val.length)
    (hval : i5.val = roundQ ((y.val[iv.val]'hivlt).val * coeff)) :
    normAt y coeff (x1.set iv i5) lo (iv.val + 1) := by
  obtain ⟨hlen, hmid, hout⟩ := hinv
  have hivx1 : iv.val < x1.val.length := by rw [hlen]; exact hivlt
  refine ⟨?_, ?_, ?_⟩
  · rw [Slice.set_val_eq, List.length_set]; exact hlen
  · intro j hjlo hjhi hjy
    have hjle : j ≤ iv.val := by omega
    rcases Nat.lt_or_eq_of_le hjle with hjlt | hjeq
    · rw [Slice.set_val_eq, List.getElem?_set, if_neg (by omega : ¬ (iv.val = j))]
      exact hmid j hjlo hjlt hjy
    · subst hjeq
      have hval' : i5.val = roundQ ((y.val[iv.val]'hjy).val * coeff) := by
        rw [hval]
      rw [Slice.set_val_eq, List.getElem?_set_self hivx1]
      simp only [Option.map_some]
      rw [hval']
  · intro j hjout hjy
    rw [Slice.set_val_eq, List.getElem?_set]
    rw [if_neg (by omega : ¬ (iv.val = j))]
    rcases hjout with hlt | hge
    · exact hout j (Or.inl hlt) hjy
    · exact hout j (Or.inr (by omega)) hjy

theorem fwht_fixed_loop3_body_step (y x1 : Slice I64) (coeff : I128) (idx iv : Usize)
    (hinv : normAt y coeff.val x1 idx.val iv.val)
    (hidxle : idx.val ≤ iv.val) (hilt : iv.val < y.val.length)
    (hwin : NormWinAt coeff.val (y.val[iv.val]'hilt).val) :
    ∃ x1u inext, hadamard.fwht_fixed_loop3.body coeff x1 iv true none
        = ok (ControlFlow.cont (x1u, inext, true, none)) ∧
      inext.val = iv.val + 1 ∧ normAt y coeff.val x1u idx.val inext.val := by
  obtain ⟨hlen, hmid, hout⟩ := hinv
  obtain ⟨hwlo, hwhi, hwabs, hwrlo, hwrhi⟩ := hwin
  have hivx1 : iv.val < x1.val.length := by rw [hlen]; exact hilt
  have heq : x1.val[iv.val]'hivx1 = y.val[iv.val]'hilt := by
    have h := hout iv.val (Or.inr le_rfl) hilt
    rw [List.getElem?_eq_getElem hivx1, List.getElem?_eq_getElem hilt] at h
    exact Option.some.inj h
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec x1 iv hivx1)
  have hi1val : i1.val = (y.val[iv.val]'hilt).val := by
    rw [hi1v]
    exact congrArg (fun z : I64 => z.val) heq
  have hfrom : (core.convert.num.FromI128I64.from i1).val = i1.val :=
    core.convert.num.FromI128I64.from_val_eq i1
  have hpmul : (core.convert.num.FromI128I64.from i1).val * coeff.val
      = (y.val[iv.val]'hilt).val * coeff.val := by rw [hfrom, hi1val]
  obtain ⟨prod, hprod_eq, hprodv⟩ := I128_checked_mul_some
    (core.convert.num.FromI128I64.from i1) coeff
    (by rw [hpmul]; exact hwlo) (by rw [hpmul]; exact hwhi)
  have hprodval : prod.val = (y.val[iv.val]'hilt).val * coeff.val := by rw [hprodv, hpmul]
  obtain ⟨rounded, hround_eq, hroundval⟩ :=
    round_q32_signed_val prod (by rw [hprodval]; exact hwabs)
  have hroundedval : rounded.val = roundQ ((y.val[iv.val]'hilt).val * coeff.val) := by
    rw [hroundval, hprodval]
  have hnotgt : ¬ (rounded > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, fromI64_max_val, hroundedval] at hc
    omega
  have hnotlt : ¬ (rounded < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, fromI64_min_val, hroundedval] at hc
    omega
  have hcl : -((2 : ℤ) ^ 63) ≤ rounded.val := by rw [hroundedval]; exact hwrlo
  have hch : rounded.val ≤ (2 : ℤ) ^ 63 - 1 := by rw [hroundedval]; exact hwrhi
  have hcastval : (IScalar.cast .I64 rounded).val
      = roundQ ((y.val[iv.val]'hilt).val * coeff.val) := by
    rw [cast_i64_val rounded hcl hch, hroundedval]
  obtain ⟨x1u, hupd_eq, hupd⟩ := WP.spec_imp_exists
    (Slice.update_spec x1 iv (IScalar.cast .I64 rounded) hivx1)
  have himax : iv.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by simp
    rw [h1]
    have hlenmax : y.val.length ≤ Usize.max := Slice.length_ineq y
    omega
  obtain ⟨inext, hadd_eq, haddv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := iv) (y := 1#usize) himax)
  have hinextv : inext.val = iv.val + 1 := by rw [haddv]; simp
  refine ⟨x1u, inext, ?_, hinextv, ?_⟩
  · rw [hadamard.fwht_fixed_loop3.body]
    have hltx : iv < Slice.len x1 :=
      (UScalar.lt_equiv iv (Slice.len x1)).2 (by rw [Slice.len_val]; exact hivx1)
    rw [if_pos hltx, if_pos rfl]
    rw [hi1]
    simp only [lift, bind_tc_ok]
    rw [hprod_eq]
    simp only [core.option.Option.ok_or_some, bind_tc_ok]
    rw [hround_eq]
    simp only [bind_tc_ok]
    rw [if_neg hnotgt]
    rw [if_neg hnotlt]
    rw [hupd_eq]
    simp only [bind_tc_ok]
    rw [hadd_eq]
    simp only [bind_tc_ok]
  · rw [hupd, hinextv]
    exact normAt_set y x1 coeff.val idx.val (IScalar.cast .I64 rounded) iv
      ⟨hlen, hmid, hout⟩ hidxle hilt hcastval


theorem fwht_fixed_loop3_spec (y : Slice I64) (coeff : I128) (idx : Usize)
    (hidx : idx.val ≤ y.val.length)
    (hdom : ∀ j (_hj : idx.val ≤ j) (hjy : j < y.val.length),
      NormWinAt coeff.val (y.val[j]'hjy).val) :
    ∃ x', hadamard.fwht_fixed_loop3 y coeff idx true none = ok (x', none) ∧
      normAt y coeff.val x' idx.val y.val.length := by
  have hspec : hadamard.fwht_fixed_loop3 y coeff idx true none ⦃ p =>
      p.2 = none ∧ normAt y coeff.val p.1 idx.val y.val.length ⦄ := by
    unfold hadamard.fwht_fixed_loop3
    apply loop.spec_decr_nat
      (measure := fun st : Slice I64 × Usize × Bool × Option hadamard.HadamardError =>
        y.val.length - st.2.1.val)
      (inv := fun st => normAt y coeff.val st.1 idx.val st.2.1.val ∧
        idx.val ≤ st.2.1.val ∧ st.2.1.val ≤ y.val.length ∧
        st.2.2.1 = true ∧ st.2.2.2 = none)
    · intro st hst
      rcases st with ⟨x1, iv, ok1, err1⟩
      obtain ⟨hinv, hidxle, hile, hok, herr⟩ := hst
      dsimp only at hinv hidxle hile hok herr
      dsimp only
      by_cases hlt : iv < Slice.len y
      · have hilt : iv.val < y.val.length := by
          have h := (UScalar.lt_equiv iv (Slice.len y)).mp hlt
          rwa [Slice.len_val] at h
        have hwin : NormWinAt coeff.val (y.val[iv.val]'hilt).val := hdom iv.val hidxle hilt
        obtain ⟨x1u, inext, hbody, hinext, hinv'⟩ :=
          fwht_fixed_loop3_body_step y x1 coeff idx iv hinv hidxle hilt hwin
        have hbody' : hadamard.fwht_fixed_loop3.body coeff x1 iv ok1 err1
            = ok (ControlFlow.cont (x1u, inext, true, none)) := by
          rw [hok, herr]; exact hbody
        rw [hbody']
        refine ⟨⟨hinv', ?_, ?_, rfl, rfl⟩, ?_⟩
        · rw [hinext]; omega
        · rw [hinext]; omega
        · show y.val.length - inext.val < y.val.length - iv.val
          rw [hinext]; omega
      · have hnot : ¬ (iv < Slice.len y) := hlt
        have hive : iv.val = y.val.length := by
          have h : ¬ (iv.val < y.val.length) := by
            intro hh
            exact hnot ((UScalar.lt_equiv iv (Slice.len y)).mpr (by rw [Slice.len_val]; exact hh))
          omega
        have hlenx : Slice.len x1 = Slice.len y := by
          apply UScalar.eq_of_val_eq
          rw [Slice.len_val, Slice.len_val]
          exact hinv.1
        have hbody : hadamard.fwht_fixed_loop3.body coeff x1 iv ok1 err1
            = ok (ControlFlow.done (x1, err1)) := by
          rw [hadamard.fwht_fixed_loop3.body, if_neg (by rw [hlenx]; exact hnot)]
        rw [hbody]
        refine ⟨herr, ?_⟩
        rw [hive] at hinv
        exact hinv
    · refine ⟨?_, le_rfl, hidx, rfl, rfl⟩
      dsimp only
      unfold normAt
      refine ⟨rfl, ?_, ?_⟩
      · intro j hjlo hjhi hjy; omega
      · intro j hjout hjy; rfl
  obtain ⟨⟨x', err⟩, hp, hpost⟩ := WP.spec_imp_exists hspec
  obtain ⟨herr, hmap⟩ := hpost
  dsimp only at herr
  refine ⟨x', ?_, hmap⟩
  rw [hp, herr]


theorem signsGood_good (s : Slice I8) (h : signsGood s) :
    ∀ (j : ℕ) (a : I8), s.val[j]? = some a → a.val = 1 ∨ a.val = -1 := by
  intro j a hja
  have hall : s.val.all signGood = true := by
    simp only [signsGood, signsGoodFrom] at h
    simpa using h
  have hall' : ∀ b ∈ s.val, signGood b = true := List.all_eq_true.mp hall
  have hg : signGood a = true := hall' a (List.mem_iff_getElem?.mpr ⟨j, hja⟩)
  rw [signGood, Bool.or_eq_true] at hg
  rcases hg with h1 | h1
  · exact Or.inl (by rw [beq_iff_eq.mp h1]; decide)
  · exact Or.inr (by rw [beq_iff_eq.mp h1]; decide)

theorem i64_ne_min_of_win (x : Slice I64) (k : Usize) (hk1 : 1 ≤ k.val)
    (hwin : ∀ i (hi : i < x.val.length), |(x.val[i]).val| * 2 ^ k.val ≤ i64Ceil)
    {j : ℕ} {b : I64} (hjb : x.val[j]? = some b) : b.val ≠ I64.min := by
  intro hmin
  have hjlt : j < x.val.length := by
    by_contra hc
    rw [List.getElem?_eq_none (by omega)] at hjb
    simp at hjb
  rw [List.getElem?_eq_getElem hjlt] at hjb
  have hbv : b.val = (x.val[j]'hjlt).val := by
    rw [Option.some.injEq] at hjb
    rw [← hjb]
  have h2k : 2 ≤ 2 ^ k.val := by
    have h := Nat.pow_le_pow_right (show 1 ≤ (2 : ℕ) by norm_num) hk1
    simpa using h
  have hmul2 : |b.val| * 2 ≤ |b.val| * 2 ^ k.val :=
    mul_le_mul_of_nonneg_left (by exact_mod_cast h2k) (abs_nonneg _)
  have habs : |b.val| * 2 ≤ i64Ceil := le_trans hmul2 (by rw [hbv]; exact hwin j hjlt)
  rw [hmin, I64.min_eq] at habs
  norm_num [i64Ceil] at habs

theorem smulList_length (a b : List ℤ) :
    (smulList a b).length = min a.length b.length := by
  unfold smulList; rw [List.length_zipWith]

theorem normAt_map (y x : Slice I64) (coeff : ℤ) (h : normAt y coeff x 0 y.val.length) :
    x.val.map (·.val) = y.val.map (fun v => roundQ (v.val * coeff)) := by
  apply List.ext_getElem?
  intro j
  rw [List.getElem?_map, List.getElem?_map]
  by_cases hj : j < y.val.length
  · rw [List.getElem?_eq_getElem hj]
    simp only [Option.map_some]
    exact h.2.1 j (Nat.zero_le j) hj hj
  · rw [List.getElem?_eq_none (by omega : y.val.length ≤ j),
      List.getElem?_eq_none (by rw [h.1]; exact (by omega : y.val.length ≤ j))]
    simp

theorem C_len : hadamard.C.val.length = 17 := by
  have h := congrArg List.length hadamard_C_val
  rw [List.length_map] at h
  rw [h]; rfl

theorem smulList_getElem?_some (ss xs : List ℤ) (i : ℕ) {a b : ℤ}
    (h1 : ss[i]? = some a) (h2 : xs[i]? = some b) :
    (smulList ss xs)[i]? = some (a * b) := by
  unfold smulList
  rw [List.getElem?_zipWith, h1, h2]

theorem smulList_window (ss xs : List ℤ) (hgood : ∀ v ∈ ss, |v| = 1) (k : ℕ)
    (hwin : ∀ i (hi : i < xs.length), |xs[i]'hi| * 2 ^ k ≤ i64Ceil) :
    ∀ i (hi : i < (smulList ss xs).length), |(smulList ss xs)[i]'hi| * 2 ^ k ≤ i64Ceil := by
  intro i hi
  have hmin : (smulList ss xs).length = min ss.length xs.length := by
    unfold smulList; rw [List.length_zipWith]
  have hxs : i < xs.length := by rw [hmin] at hi; omega
  have hss : i < ss.length := by rw [hmin] at hi; omega
  have h1 : ss[i]? = some (ss[i]'hss) := List.getElem?_eq_getElem hss
  have h2 : xs[i]? = some (xs[i]'hxs) := List.getElem?_eq_getElem hxs
  have hM := smulList_getElem?_some ss xs i h1 h2
  have heq : (smulList ss xs)[i]'hi = ss[i]'hss * xs[i]'hxs := by
    have hM' := hM
    rw [List.getElem?_eq_getElem hi] at hM'
    exact Option.some.inj hM'
  rw [heq, abs_mul, hgood _ (List.getElem_mem hss), one_mul]
  exact hwin i hxs

theorem map_round_of_map_val (l : List I64) (M : List ℤ) (h : l.map (·.val) = M) (c : ℤ) :
    l.map (fun v => roundQ (v.val * c)) = M.map (fun v => roundQ (v * c)) := by
  rw [← h]
  exact (List.map_map (f := fun v : I64 => v.val) (g := fun v : ℤ => roundQ (v * c)) (l := l)).symm

theorem window_of_smulList_eq (M ss xs : List ℤ) (hM : M = smulList ss xs)
    (hgood : ∀ v ∈ ss, |v| = 1) (k : ℕ)
    (hwin : ∀ i (hi : i < xs.length), |xs[i]'hi| * 2 ^ k ≤ i64Ceil) :
    ∀ i (hi : i < M.length), |M[i]'hi| * 2 ^ k ≤ i64Ceil := by
  subst hM
  exact smulList_window ss xs hgood k hwin

theorem fwht_fixed_norm_spec (x2 : Slice I64) (k : Usize) (hk1 : 1 ≤ k.val) (hk16 : k.val ≤ 16)
    (hnarrow : ∀ (j : ℕ) (hj : j < x2.val.length),
      NormWinAt (cValue k.val) (x2.val[j]'hj).val) :
    ∃ (i2 : I64) (x3 : Slice I64),
      Array.index_usize hadamard.C k = ok i2 ∧
      hadamard.fwht_fixed_loop3 x2 (core.convert.num.FromI128I64.from i2) 0#usize true none
        = ok (x3, none) ∧
      x3.val.map (·.val) = x2.val.map (fun v => roundQ (v.val * cValue k.val)) := by
  have hClt : k.val < hadamard.C.val.length := by rw [C_len]; omega
  obtain ⟨i2, hidx, hi2v⟩ := WP.spec_imp_exists (Array.index_usize_spec hadamard.C k hClt)
  have hi2val : i2.val = cValue k.val := by
    have hc := C_val k.val hk1 hk16
    rw [List.getElem?_eq_getElem hClt] at hc
    simp only [Option.map_some, Option.some.injEq] at hc
    rw [hi2v]; exact hc
  have hcoeffval : (core.convert.num.FromI128I64.from i2).val = cValue k.val := by
    rw [core.convert.num.FromI128I64.from_val_eq, hi2val]
  obtain ⟨x3, hloop3, hnorm⟩ := fwht_fixed_loop3_spec x2
    (core.convert.num.FromI128I64.from i2) 0#usize (by simp)
    (by
      intro j _ hjy
      rw [hcoeffval]
      exact hnarrow j hjy)
  refine ⟨i2, x3, hidx, hloop3, ?_⟩
  have := normAt_map x2 x3 (cValue k.val) (by rw [← hcoeffval]; exact hnorm)
  exact this

theorem fwht_fixed_spec (x : Slice I64) (s : Slice I8) (block k : Usize)
    (hb : block.val = 2 ^ k.val) (hk1 : 1 ≤ k.val) (hk16 : k.val ≤ 16)
    (hlen : s.val.length = x.val.length)
    (hsgood : signsGood s)
    (hdiv : block.val ∣ x.val.length)
    (hwin : ∀ i (hi : i < x.val.length), |(x.val[i]).val| * 2 ^ k.val ≤ i64Ceil)
    (hnarrow : ∀ (j : ℕ) (hj : j < (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).length),
      NormWinAt (cValue k.val) ((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[j]'hj)) :
    ∃ x' : Slice I64,
      hadamard.fwht_fixed x (some s) block = ok (core.result.Result.Ok (), x') ∧
      x'.val.map (·.val)
        = (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).map
            (fun v => roundQ (v * cValue k.val)) := by
  set M : List ℤ := smulList (s.val.map (·.val)) (x.val.map (·.val)) with hMdef
  have hexp := block_exponent_spec_pow block k hb hk1 hk16
  have hmod : (Slice.len x).val % block.val = 0 := by
    rw [Slice.len_val]; exact Nat.dvd_iff_mod_eq_zero.mp hdiv
  have hmul : core.num.Usize.is_multiple_of (Slice.len x) block = ok true := by
    unfold core.num.Usize.is_multiple_of UScalar.is_multiple_of
    rw [hmod]; rfl
  have hslen : ¬ (((Slice.len s) != (Slice.len x)) = true) := by
    rw [bne_iff_ne]
    intro hc
    apply hc
    apply UScalar.eq_of_val_eq
    rw [Slice.len_val, Slice.len_val]; exact hlen
  have hloop1 : hadamard.fwht_fixed_loop1 s 0#usize true = ok true := by
    rw [fwht_fixed_loop1_spec s 0#usize true (by simp)]
    have hg : signsGoodFrom s (0#usize).val = true := by
      rw [show (0#usize).val = 0 by simp]
      unfold signsGood at hsgood
      exact hsgood
    rw [hg]; simp
  have hsabs : ∀ v ∈ s.val.map (·.val), |v| = 1 := by
    intro v hv
    obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hv
    obtain ⟨i, hi⟩ := List.mem_iff_getElem?.mp ha
    rcases signsGood_good s hsgood i a hi with h | h <;> rw [h] <;> norm_num
  obtain ⟨x1, hloop2, hmap2⟩ := fwht_fixed_loop2_spec x s hlen
    (signsGood_good s hsgood)
    (by
      intro j a b hja hjb _
      exact i64_ne_min_of_win x k hk1 hwin hjb)
  have hMlen : M.length = x.val.length := by
    rw [hMdef, smulList_length]
    simp only [List.length_map]
    rw [hlen, Nat.min_self]
  have hMlen' : M.length = x1.val.length := by
    have h := congrArg List.length hmap2
    rw [List.length_map] at h
    have h2 : M.length = (smulList (s.val.map (·.val)) (x.val.map (·.val))).length := by rw [hMdef]
    rw [h2, h]
  have hx1div : block.val ∣ x1.val.length := by rw [← hMlen', hMlen]; exact hdiv
  have hx1win : ∀ i (hi : i < x1.val.length), |(x1.val[i]).val| * 2 ^ k.val ≤ i64Ceil := by
    intro i hi
    have hile : i < M.length := by rw [hMlen']; exact hi
    have hmapi : (x1.val.map (·.val))[i]? = some (x1.val[i]'hi).val := by
      rw [List.getElem?_map, List.getElem?_eq_getElem hi]
      rfl
    rw [hmap2] at hmapi
    have heq : (x1.val[i]'hi).val = M[i]'hile := by
      have h' := hmapi
      rw [List.getElem?_eq_getElem hile] at h'
      exact (Option.some.inj h').symm
    rw [heq]
    exact window_of_smulList_eq M (s.val.map (·.val)) (x.val.map (·.val)) hMdef hsabs k.val
      (by
        intro j hjj
        have hjj' : j < x.val.length := by simpa only [List.length_map] using hjj
        rw [List.getElem_map]
        exact hwin j hjj')
      i hile
  obtain ⟨x2, hint, hmapint⟩ := fwht_int_spec x1 block k hb hk1 hk16 hx1div hx1win
  have hmapint' : x2.val.map (·.val) = fwhtRef k.val M := by rw [hmapint, hmap2]
  have hx2len : x2.val.length = M.length := by
    have h := congrArg List.length hmapint'
    rw [List.length_map, fwhtRef_length] at h
    exact h
  obtain ⟨i2, x3, hidx, hloop3, hmap3⟩ := fwht_fixed_norm_spec x2 k hk1 hk16 (by
    intro j hj
    have hx2 : (x2.val[j]'hj).val
        = (fwhtRef k.val M)[j]'(by rw [fwhtRef_length, ← hx2len]; exact hj) := by
      have hc := congrArg (fun l => l[j]?) hmapint'
      rw [List.getElem?_map, List.getElem?_eq_getElem hj, Option.map_some] at hc
      rw [List.getElem?_eq_getElem (by rw [fwhtRef_length, ← hx2len]; exact hj)] at hc
      exact Option.some.inj hc
    rw [hx2]
    exact hnarrow j (by rw [fwhtRef_length, ← hx2len]; exact hj))
  refine ⟨x3, ?_, ?_⟩
  · rw [hadamard.fwht_fixed.eq_def, hexp]
    simp only [bind_tc_ok, hmul]
    rw [if_true, if_neg hslen]
    try simp only [bind_tc_ok, if_true]
    rw [hloop1]
    try simp
    rw [hloop2]
    try simp
    rw [hint]
    try simp
    rw [hidx]
    simp only [lift, bind_tc_ok]
    rw [hloop3]
    try simp
    try rfl
  · rw [hmap3]
    exact map_round_of_map_val x2 (fwhtRef k.val M) hmapint' (cValue k.val)

theorem roundQ_bound (n : ℤ) :
    |roundQ n * (2 : ℤ) ^ 32 - n| ≤ (2 : ℤ) ^ 31 := by
  unfold roundQ divRoundNearest
  by_cases h : n < 0
  · rw [if_pos h]
    have hb := div_round_nearest_abs_bound (-n) ((2 : ℤ) ^ 32) (by norm_num)
    rw [show (2 : ℤ) ^ 32 / 2 = (2 : ℤ) ^ 31 from by norm_num] at hb
    have he : -((-n + (2 : ℤ) ^ 32 / 2) / (2 : ℤ) ^ 32) * (2 : ℤ) ^ 32 - n
        = -((((-n + (2 : ℤ) ^ 32 / 2) / (2 : ℤ) ^ 32)) * (2 : ℤ) ^ 32 - (-n)) := by ring
    rw [he, abs_neg, show (2 : ℤ) ^ 32 / 2 = (2 : ℤ) ^ 31 from by norm_num]
    exact hb
  · rw [if_neg h]
    have hb := div_round_nearest_abs_bound n ((2 : ℤ) ^ 32) (by norm_num)
    rw [show (2 : ℤ) ^ 32 / 2 = (2 : ℤ) ^ 31 from by norm_num] at hb
    exact hb

/-! ## Item 5 · the real-arithmetic refinement -/

/-- **Rounding-scale bound.** The single `roundQ` reduction is within half a Q32.32 unit of the
exact quotient: `|roundQ n / 2^32 − n / 2^64| ≤ 1/(2·2^32)`. -/
theorem roundQ_real_bound (n : ℤ) :
    |(roundQ n : ℝ) / q32 - (n : ℝ) / q32 ^ 2| ≤ 1 / (2 * q32) := by
  set Z : ℤ := roundQ n * (2 : ℤ) ^ 32 - n with hZdef
  have hint : |Z| ≤ (2 : ℤ) ^ 31 := by rw [hZdef]; exact roundQ_bound n
  have hZid : (roundQ n : ℝ) / q32 - (n : ℝ) / q32 ^ 2 = ((Z : ℤ) : ℝ) / (2 : ℝ) ^ 64 := by
    rw [hZdef]; unfold q32; push_cast; ring
  have hZR : |((Z : ℤ) : ℝ)| ≤ (2 : ℝ) ^ 31 := by exact_mod_cast hint
  rw [hZid, abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ 64)]
  calc |((Z : ℤ) : ℝ)| / (2 : ℝ) ^ 64 ≤ (2 : ℝ) ^ 31 / (2 : ℝ) ^ 64 :=
        div_le_div_of_nonneg_right hZR (by positivity)
    _ = 1 / (2 * q32) := by rw [show q32 = (2 : ℝ) ^ 32 from rfl]; norm_num

/-- **Pointwise error decomposition.** For a value `yv` and coefficient `c`, the rounded real
`roundQ (yv·c)/2^32` is within `1/(2·2^32) + |yv|/2^32 · |c/2^32 − 1/√(2^k)|` of
`yv/(2^32·√(2^k))`: the rounding half-ulp plus the coefficient's half-ulp times the magnitude. -/
theorem roundQ_refines_pointwise (k : ℕ) (yv c : ℤ) :
    |(roundQ (yv * c) : ℝ) / q32 - (yv : ℝ) / (q32 * Real.sqrt ((2 : ℝ) ^ k))| ≤
      1 / (2 * q32) +
        |(yv : ℝ)| / q32 * |(c : ℝ) / q32 - 1 / Real.sqrt ((2 : ℝ) ^ k)| := by
  have hs : (0 : ℝ) < Real.sqrt ((2 : ℝ) ^ k) := Real.sqrt_pos.mpr (by positivity)
  have hsne : Real.sqrt ((2 : ℝ) ^ k) ≠ 0 := ne_of_gt hs
  have hround := roundQ_real_bound (yv * c)
  have hround' : |(roundQ (yv * c) : ℝ) / q32 - (yv : ℝ) * (c : ℝ) / q32 ^ 2| ≤ 1 / (2 * q32) := by
    simpa only [Int.cast_mul] using hround
  have h2 : (yv : ℝ) * (c : ℝ) / q32 ^ 2 - (yv : ℝ) / (q32 * Real.sqrt ((2 : ℝ) ^ k))
      = (yv : ℝ) / q32 * ((c : ℝ) / q32 - 1 / Real.sqrt ((2 : ℝ) ^ k)) := by
    field_simp [q32_ne_zero, hsne]
  calc |(roundQ (yv * c) : ℝ) / q32 - (yv : ℝ) / (q32 * Real.sqrt ((2 : ℝ) ^ k))|
      = |((roundQ (yv * c) : ℝ) / q32 - (yv : ℝ) * (c : ℝ) / q32 ^ 2)
          + ((yv : ℝ) * (c : ℝ) / q32 ^ 2
            - (yv : ℝ) / (q32 * Real.sqrt ((2 : ℝ) ^ k)))| := by
            congr 1; ring
    _ ≤ |(roundQ (yv * c) : ℝ) / q32 - (yv : ℝ) * (c : ℝ) / q32 ^ 2|
          + |(yv : ℝ) * (c : ℝ) / q32 ^ 2
            - (yv : ℝ) / (q32 * Real.sqrt ((2 : ℝ) ^ k))| := abs_add_le _ _
    _ = |(roundQ (yv * c) : ℝ) / q32 - (yv : ℝ) * (c : ℝ) / q32 ^ 2|
          + |(yv : ℝ) / q32 * ((c : ℝ) / q32 - 1 / Real.sqrt ((2 : ℝ) ^ k))| := by rw [h2]
    _ = |(roundQ (yv * c) : ℝ) / q32 - (yv : ℝ) * (c : ℝ) / q32 ^ 2|
          + |(yv : ℝ)| / q32 * |(c : ℝ) / q32 - 1 / Real.sqrt ((2 : ℝ) ^ k)| := by
            rw [abs_mul, abs_div, abs_of_pos q32_pos]
    _ ≤ 1 / (2 * q32)
          + |(yv : ℝ)| / q32 * |(c : ℝ) / q32 - 1 / Real.sqrt ((2 : ℝ) ^ k)| :=
          add_le_add hround' le_rfl

/-- **Item 5 · normalized FWHT refinement.** On the checked signs domain with the no-overflow and
`i64` narrowing windows, the extracted `hadamard.fwht_fixed` returns `Ok ()` with values within
`1/(2·2^32) + |y_i|/2^32 · |C_k/2^32 − 2^{−k/2}|` of `y_i/(2^32·√(2^k))`, where
`y = fwhtRef k (s⊙x)`. -/
theorem fwht_fixed_refines (x : Slice I64) (s : Slice I8) (block k : Usize)
    (hb : block.val = 2 ^ k.val) (hk1 : 1 ≤ k.val) (hk16 : k.val ≤ 16)
    (hlen : s.val.length = x.val.length)
    (hsgood : signsGood s)
    (hdiv : block.val ∣ x.val.length)
    (hwin : ∀ i (hi : i < x.val.length), |(x.val[i]).val| * 2 ^ k.val ≤ i64Ceil)
    (hnarrow : ∀ (j : ℕ)
      (hj : j < (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).length),
      NormWinAt (cValue k.val) ((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[j]'hj)) :
    ∃ x' : Slice I64,
      hadamard.fwht_fixed x (some s) block = ok (core.result.Result.Ok (), x') ∧
      (∀ (j : ℕ) (hj : j < x'.val.length)
          (hjM : j < (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).length),
        |toReal (x'.val[j]'hj)
            - (((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[j]'hjM : ℤ) : ℝ)
                / (q32 * Real.sqrt ((2 : ℝ) ^ k.val))| ≤
          1 / (2 * q32) +
            |(((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[j]'hjM : ℤ) : ℝ)|
                / q32
              * |(cValue k.val : ℝ) / q32 - 1 / Real.sqrt ((2 : ℝ) ^ k.val)|) := by
  obtain ⟨x', hfix, hmap⟩ := fwht_fixed_spec x s block k hb hk1 hk16 hlen hsgood hdiv hwin hnarrow
  refine ⟨x', hfix, ?_⟩
  intro j hj hjM
  have hpw : (x'.val[j]'hj).val
      = roundQ (((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[j]'hjM)
          * cValue k.val) := by
    have hc := congrArg (fun l => l[j]?) hmap
    rw [List.getElem?_map, List.getElem?_map] at hc
    rw [List.getElem?_eq_getElem hj, List.getElem?_eq_getElem hjM] at hc
    simp only [Option.map_some, Option.some.injEq] at hc
    exact hc
  have hbound := roundQ_refines_pointwise k.val
    ((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[j]'hjM) (cValue k.val)
  unfold toReal
  rw [hpw]
  exact hbound
/-- `Matrix.mulVec` is insensitive to the `Fin` proof: equal index values give equal entries. -/
theorem mulVec_fin_apply_eq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℤ) (v : Fin n → ℤ)
    (a b : Fin n) (h : a.val = b.val) : (A.mulVec v) a = (A.mulVec v) b :=
  congrArg (fun z => (A.mulVec v) z) (Fin.ext h)

/-- **Item 5b · aligned-segment refinement.** The `fwht_fixed_refines` bound with the block value
read through `fwhtRef_segment`/`fwhtRef_getElem` as `(sylvesterH k).mulVec` applied to the aligned
`2^k` segment of `s⊙x`. -/
theorem fwht_fixed_refines_sylvester (x : Slice I64) (s : Slice I8) (block k : Usize)
    (hb : block.val = 2 ^ k.val) (hk1 : 1 ≤ k.val) (hk16 : k.val ≤ 16)
    (hlen : s.val.length = x.val.length)
    (hsgood : signsGood s)
    (hdiv : block.val ∣ x.val.length)
    (hwin : ∀ i (hi : i < x.val.length), |(x.val[i]).val| * 2 ^ k.val ≤ i64Ceil)
    (hnarrow : ∀ (j : ℕ)
      (hj : j < (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).length),
      NormWinAt (cValue k.val) ((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[j]'hj))
    (m : ℕ) (hm : x.val.length = m * 2 ^ k.val) (i : ℕ) (hi : i < m) :
    ∃ x' : Slice I64,
      hadamard.fwht_fixed x (some s) block = ok (core.result.Result.Ok (), x') ∧
      x'.val.length = m * 2 ^ k.val ∧
      (∀ (r : Fin (2 ^ k.val)) (hx : i * 2 ^ k.val + r.val < x'.val.length),
        let M := smulList (s.val.map (·.val)) (x.val.map (·.val))
        let seg := (M.drop (i * 2 ^ k.val)).take (2 ^ k.val)
        let yv := (sylvesterH k.val).mulVec
          (fun l : Fin (2 ^ k.val) => seg.get
            (Fin.cast (sylvester_seg_length k.val m M
              (by rw [smulList_length]; simp only [List.length_map]; rw [hlen, hm, Nat.min_self])
              i hi).symm l)) r
        |toReal (x'.val[i * 2 ^ k.val + r.val]'hx)
            - ((yv : ℤ) : ℝ) / (q32 * Real.sqrt ((2 : ℝ) ^ k.val))| ≤
          1 / (2 * q32) + |((yv : ℤ) : ℝ)| / q32
            * |(cValue k.val : ℝ) / q32 - 1 / Real.sqrt ((2 : ℝ) ^ k.val)|) := by
  have hM : (smulList (s.val.map (·.val)) (x.val.map (·.val))).length = m * 2 ^ k.val := by
    rw [smulList_length]
    simp only [List.length_map]
    rw [hlen, hm, Nat.min_self]
  obtain ⟨x', hfix, hmap⟩ := fwht_fixed_spec x s block k hb hk1 hk16 hlen hsgood hdiv hwin hnarrow
  have hx'len : x'.val.length = (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).length := by
    have h := congrArg List.length hmap
    rw [List.length_map, List.length_map] at h
    exact h
  have hx'lenm : x'.val.length = m * 2 ^ k.val := by rw [hx'len, fwhtRef_length, hM]
  refine ⟨x', hfix, hx'lenm, ?_⟩
  intro r hx
  dsimp only
  have hidxM : i * 2 ^ k.val + r.val
      < (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).length := by
    rw [← hx'len]; exact hx
  have hpw : (x'.val[i * 2 ^ k.val + r.val]'hx).val
      = roundQ (((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[i * 2 ^ k.val + r.val]'hidxM)
          * cValue k.val) := by
    have hc := congrArg (fun l => l[i * 2 ^ k.val + r.val]?) hmap
    rw [List.getElem?_map, List.getElem?_map] at hc
    rw [List.getElem?_eq_getElem hx, List.getElem?_eq_getElem hidxM] at hc
    simp only [Option.map_some, Option.some.injEq] at hc
    exact hc
  have hlen2 : (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop (i * 2 ^ k.val)).take
      (2 ^ k.val)).length = 2 ^ k.val := sylvester_seg_length k.val m _ hM i hi
  have hllen : r.val < (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop
      (i * 2 ^ k.val)).take (2 ^ k.val)).length := lt_of_lt_of_eq r.isLt hlen2.symm
  have hr : r.val < (fwhtRef k.val (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop
      (i * 2 ^ k.val)).take (2 ^ k.val))).length :=
    lt_of_lt_of_eq hllen (fwhtRef_length k.val _).symm
  have hpi := fwhtRef_getElem k.val
    (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop (i * 2 ^ k.val)).take (2 ^ k.val))
    hlen2 r.val hllen
  have e1 : ((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).drop
      (i * 2 ^ k.val))[r.val]?
      = (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[i * 2 ^ k.val + r.val]? := by
    rw [List.getElem?_drop]
  have e2 : (((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).drop
      (i * 2 ^ k.val)).take (2 ^ k.val))[r.val]?
      = ((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val)))).drop
        (i * 2 ^ k.val))[r.val]? := by
    rw [List.getElem?_take, if_pos r.isLt]
  have h1 : some ((fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[i * 2 ^ k.val + r.val]'hidxM)
      = some ((fwhtRef k.val (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop
          (i * 2 ^ k.val)).take (2 ^ k.val)))[r.val]'hr) := by
    rw [← List.getElem?_eq_getElem hidxM]
    rw [← e1, ← e2]
    rw [fwhtRef_segment k.val m (smulList (s.val.map (·.val)) (x.val.map (·.val))) hM i hi]
    rw [List.getElem?_eq_getElem hr]
  have hget : (fwhtRef k.val (smulList (s.val.map (·.val)) (x.val.map (·.val))))[i * 2 ^ k.val + r.val]'hidxM
      = (sylvesterH k.val).mulVec
          (fun l : Fin (2 ^ k.val) =>
            (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop (i * 2 ^ k.val)).take
                (2 ^ k.val)).get (Fin.cast hlen2.symm l))
          r := by
    rw [Option.some.inj h1]
    have hpi' : (fwhtRef k.val (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop
        (i * 2 ^ k.val)).take (2 ^ k.val)))[r.val]'hr
        = (sylvesterH k.val).mulVec
            (fun l : Fin (2 ^ k.val) =>
              (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop (i * 2 ^ k.val)).take
                  (2 ^ k.val)).get (Fin.cast hlen2.symm l))
            ⟨r.val, lt_of_lt_of_eq hllen hlen2⟩ := hpi
    rw [hpi']
  have hpw' : (x'.val[i * 2 ^ k.val + r.val]'hx).val
      = roundQ (((sylvesterH k.val).mulVec
            (fun l : Fin (2 ^ k.val) =>
              (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop (i * 2 ^ k.val)).take
                  (2 ^ k.val)).get (Fin.cast hlen2.symm l))
            r) * cValue k.val) := by
    rw [hpw, hget]
  have hbound := roundQ_refines_pointwise k.val
    ((sylvesterH k.val).mulVec
      (fun l : Fin (2 ^ k.val) =>
        (((smulList (s.val.map (·.val)) (x.val.map (·.val))).drop (i * 2 ^ k.val)).take
            (2 ^ k.val)).get (Fin.cast hlen2.symm l))
      r) (cValue k.val)
  unfold toReal
  rw [hpw']
  exact hbound

end

end apeiron_word
