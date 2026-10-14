import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Perf.FaceTypedE8

/-!
# The extracted E8 codebook

Refinement of `learn_port.e8` (`src/learn/code/e8.rs`, extracted into `ApeironWord`) against integer
specifications over `ℤ`, with `ONE = 2^32` and `SQRT8 = 12148002000`.

* `popcount_parity_spec`, `spinor_row_spec`, `spinor_key_parity`: the spinor row for index `r < 128` is
  the sign vector of the `r`-th even-parity key `2r + parity r`, with an even number of `-ONE` entries.
* `e8_codeword_spec`: `codeword i` is `codewordSpec i` for `i < 256` and `IndexOutOfRange` otherwise.
* `e8_codeword_spin8`: a spinor row is `ONE` times the signs of an element of
  `Perf.FaceTypedE8.Spin8`.
* `e8_nearest_spec`: for `|x_k| ≤ 2^60`, `nearest x` is the least index minimising
  `Σ (x_k - c_k)^2` over `ℤ`.
* `witness_*`: one reachable input per success branch and per `Err` arm.

The loops are proved by `loop.spec_decr_nat` with explicit invariants. Two bounded `decide` calls
close `rootPairs_length` (a list of 28) and `root_pair_link` (28 pairs × 7 starts); no 256-case table is
evaluated.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Learn.E8Code

open Aeneas Aeneas.Std Result
open apeiron_word

def ONE : ℤ := 4294967296
def SQRT8 : ℤ := 12148002000

def bitN (n j : ℕ) : ℕ := n / 2 ^ j % 2

def parityNat (r : ℕ) : ℕ := (((List.range 8).map (fun j => bitN r j)).sum) % 2

theorem parityNat_eq (r : ℕ) :
    parityNat r = (r % 2 + r / 2 % 2 + r / 4 % 2 + r / 8 % 2 + r / 16 % 2 + r / 32 % 2
      + r / 64 % 2 + r / 128 % 2) % 2 := by
  unfold parityNat bitN
  simp [List.range_succ]
  omega

theorem engine_ONE_eq : engine.ONE = ok (4294967296#i64) := by
  have h : engine.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
    unfold engine.ONE
    rw [engine.FIXED_FRAC_BITS]
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

theorem popcount_parity_spec (r : Std.Usize) (hr : r.val < 256) :
    learn_port.e8.popcount_parity r ⦃ p => p.val = parityNat r.val ⦄ := by
  unfold learn_port.e8.popcount_parity
  step*
  rw [parityNat_eq]
  simp only [p_post, i21_post, i20_post, i19_post, i18_post, i17_post, i16_post, i15_post, i14_post, i13_post, i12_post, i11_post, i10_post, i9_post, i8_post, i7_post, i6_post, i5_post, i4_post, i3_post, i2_post, i1_post, i_post]


theorem eq_of_spec' {α : Type} {x : Result α} {v : α} (h : x ⦃ z => z = v ⦄) : x = ok v := by
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

theorem NEG_ONE_eq : learn_port.e8.NEG_ONE = ok ((-4294967296)#i64) := by
  unfold learn_port.e8.NEG_ONE
  rw [engine_ONE_eq]
  simp only [bind_tc_ok]
  rfl

theorem TWO_ONE_eq : learn_port.e8.TWO_ONE = ok (8589934592#i64) := by
  unfold learn_port.e8.TWO_ONE
  rw [engine_ONE_eq]
  simp only [bind_tc_ok]
  rfl

theorem NEG_TWO_ONE_eq : learn_port.e8.NEG_TWO_ONE = ok ((-8589934592)#i64) := by
  unfold learn_port.e8.NEG_TWO_ONE
  rw [engine_ONE_eq]
  simp only [bind_tc_ok]
  rfl

theorem SQRT8_eq : learn_port.e8.SQRT8_Q32 = 12148002000#i64 := by
  unfold learn_port.e8.SQRT8_Q32
  rfl

theorem NEG_SQRT8_eq : learn_port.e8.NEG_SQRT8_Q32 = ok ((-12148002000)#i64) := by
  unfold learn_port.e8.NEG_SQRT8_Q32
  rw [SQRT8_eq]
  rfl

/-! ## Spinor rows -/

theorem parity_double (m : ℕ) (hm : m < 128) : parityNat (2 * m) = parityNat m := by
  rw [parityNat_eq, parityNat_eq]; omega

theorem parity_double_succ (m : ℕ) (hm : m < 128) :
    parityNat (2 * m + 1) = (parityNat m + 1) % 2 := by
  rw [parityNat_eq, parityNat_eq]; omega

/-- The even-popcount indices below `256`, ascending. -/
def evenKeys : List ℕ := (List.range 256).filter (fun i => parityNat i = 0)

theorem evenKeys_filter (m : ℕ) (hm : m ≤ 128) :
    (List.range (2 * m)).filter (fun i => parityNat i = 0) =
      (List.range m).map (fun r => 2 * r + parityNat r) := by
  induction m with
  | zero => simp
  | succ m ih =>
    have hlt : m < 128 := by omega
    have e : 2 * (m + 1) = 2 * m + 1 + 1 := by ring
    rw [e, List.range_succ, List.range_succ, List.filter_append, List.filter_append,
      ih (by omega), List.range_succ, List.map_append]
    have h0 := parity_double m hlt
    have h1 := parity_double_succ m hlt
    by_cases hp : parityNat m = 0
    · simp [hp, h0, h1]
    · have hp1 : parityNat m = 1 := by
        have := Nat.mod_lt (((List.range 8).map (fun j => bitN m j)).sum) (by norm_num : 0 < 2)
        unfold parityNat at hp ⊢
        omega
      simp [hp1, h0, h1]

theorem evenKeys_getD (r : ℕ) (hr : r < 128) : evenKeys.getD r 0 = 2 * r + parityNat r := by
  have h : evenKeys = (List.range 128).map (fun r => 2 * r + parityNat r) :=
    evenKeys_filter 128 (le_refl _)
  rw [h, List.getD_eq_getElem _ _ (by simpa using hr), List.getElem_map, List.getElem_range]

def signAt (key k : ℕ) : ℤ := if key / 2 ^ (7 - k) % 2 = 1 then -ONE else ONE

theorem pow_div_128 (k : ℕ) (hk : k ≤ 7) : 128 / 2 ^ k = 2 ^ (7 - k) := by
  interval_cases k <;> norm_num

theorem list_getD_set_eq {α : Type} (l : List α) (i j : ℕ) (v : α) (h : i < l.length) (d : α) :
    (l.set i v).getD j d = if i = j then v else l.getD j d := by
  by_cases heq : i = j
  · subst heq
    rw [List.getD_eq_getElem?_getD, List.getElem?_set_self h, Option.getD_some]
    simp
  · rw [List.getD_eq_getElem?_getD, List.getElem?_set, if_neg heq, ← List.getD_eq_getElem?_getD]
    simp [heq]

theorem spinor_row_loop_spec (key : Std.Usize) (out : Array Std.I64 8#usize) (k w : Std.Usize)
    (hk : k.val ≤ 8) (hw : w.val = 128 / 2 ^ k.val)
    (hd : ∀ j < k.val, (out.val.getD j 0#i64).val = signAt key.val j) :
    learn_port.e8.spinor_row_loop key out k w ⦃ o =>
      ∀ j < 8, (o.val.getD j 0#i64).val = signAt key.val j ⦄ := by
  unfold learn_port.e8.spinor_row_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.I64 8#usize × Std.Usize × Std.Usize => 8 - s.2.1.val)
    (inv := fun s => s.2.1.val ≤ 8 ∧ s.2.2.val = 128 / 2 ^ s.2.1.val ∧
      ∀ j < s.2.1.val, (s.1.val.getD j 0#i64).val = signAt key.val j)
    (post := fun o : Array Std.I64 8#usize =>
      ∀ j < 8, (o.val.getD j 0#i64).val = signAt key.val j)
  · rintro ⟨o, k', w'⟩ ⟨hk', hw', hd'⟩
    dsimp only at hk' hw' hd' ⊢
    unfold learn_port.e8.spinor_row_loop.body
    by_cases hlt : k' < 8#usize
    · rw [if_pos hlt]
      have hk8 : k'.val < 8 := by scalar_tac
      have hwpos : 0 < w'.val := by
        rw [hw', pow_div_128 k'.val (by omega)]; positivity
      have hw7 : w'.val = 2 ^ (7 - k'.val) := by rw [hw', pow_div_128 _ (by omega)]
      step*
      have hsign : ∃ v : Std.I64,
          (if i1 = 1#usize then learn_port.e8.NEG_ONE else engine.ONE) = ok v ∧
            v.val = signAt key.val k'.val := by
        by_cases h1 : i1 = 1#usize
        · have h1v : i1.val = 1 := by rw [h1]; rfl
          refine ⟨_, by rw [if_pos h1, NEG_ONE_eq], ?_⟩
          unfold signAt
          rw [← hw7, ← i_post, ← i1_post, if_pos h1v]
          rfl
        · have h1v : i1.val ≠ 1 := fun h => h1 (by scalar_tac)
          refine ⟨_, by rw [if_neg h1, engine_ONE_eq], ?_⟩
          unfold signAt
          rw [← hw7, ← i_post, ← i1_post, if_neg h1v]
          rfl
      obtain ⟨v, hv, hvval⟩ := hsign
      rw [hv]
      simp only [bind_tc_ok]
      step*
      have hlen : k'.val < o.val.length := by
        rw [o.property]
        scalar_tac
      refine ⟨by omega, ?_, ?_, by omega⟩
      · rw [w1_post, hw', k1_post, pow_succ, Nat.div_div_eq_div_mul]
      · intro j hj
        rw [a_post, Array.set_val_eq, list_getD_set_eq _ _ _ _ hlen]
        by_cases hjk : k'.val = j
        · rw [if_pos hjk, hvval, hjk]
        · rw [if_neg hjk]
          exact hd' j (by omega)
    · rw [if_neg hlt]
      have hk8 : k'.val = 8 := by scalar_tac
      simp only [WP.spec_ok]
      intro j hj
      exact hd' j (by omega)
  · exact ⟨hk, hw, hd⟩

theorem parityNat_lt (r : ℕ) : parityNat r < 2 := Nat.mod_lt _ (by norm_num)

/-- The `r`-th even-popcount index below `256`. -/
def spinorKey (r : ℕ) : ℕ := evenKeys.getD r 0

theorem spinorKey_eq (r : ℕ) (hr : r < 128) : spinorKey r = 2 * r + parityNat r :=
  evenKeys_getD r hr

theorem spinor_row_spec (r : Std.Usize) (hr : r.val < 128) :
    learn_port.e8.spinor_row r ⦃ o =>
      ∀ j < 8, (o.val.getD j 0#i64).val = signAt (spinorKey r.val) j ⦄ := by
  unfold learn_port.e8.spinor_row
  have hp := parityNat_lt r.val
  step as ⟨i, hi⟩
  step with popcount_parity_spec r (by omega) as ⟨i1, hi1⟩
  step as ⟨key, hkey⟩
  have hk : key.val = spinorKey r.val := by
    rw [spinorKey_eq _ hr]; omega
  have hl := spinor_row_loop_spec key (Array.repeat 8#usize 0#i64) 0#usize 128#usize
    (by simp) (by simp) (by simp)
  refine WP.spec_mono hl ?_
  intro o ho j hj
  rw [← hk]
  exact ho j hj

/-! ## D8 root rows -/

/-- Number of pairs `(a, b)`, `a < b < 8`, whose first index is below `t`. -/
def startR (t : ℕ) : ℕ := (15 * t - t * t) / 2

theorem startR_succ (a : ℕ) (ha : a < 7) : startR (a + 1) = startR a + (7 - a) := by
  interval_cases a <;> simp [startR]

theorem startR_seven : startR 7 = 28 := by simp [startR]

theorem root_row_loop_spec (pair : Std.Usize) (a t : Std.Usize)
    (ht1 : 1 ≤ t.val) (ht7 : t.val ≤ 7) (hat : a.val < t.val) (hs : startR a.val ≤ pair.val)
    (hgt : ∀ s, a.val < s → s < t.val → pair.val < startR s) :
    learn_port.e8.root_row_loop pair a t ⦃ a' =>
      a'.val < 7 ∧ startR a'.val ≤ pair.val ∧ ∀ s, a'.val < s → s < 7 → pair.val < startR s ⦄ := by
  unfold learn_port.e8.root_row_loop
  apply loop.spec_decr_nat
    (measure := fun s : Std.Usize × Std.Usize => 7 - s.2.val)
    (inv := fun s => 1 ≤ s.2.val ∧ s.2.val ≤ 7 ∧ s.1.val < s.2.val ∧ startR s.1.val ≤ pair.val ∧
      ∀ u, s.1.val < u → u < s.2.val → pair.val < startR u)
    (post := fun a' : Std.Usize =>
      a'.val < 7 ∧ startR a'.val ≤ pair.val ∧ ∀ s, a'.val < s → s < 7 → pair.val < startR s)
  · rintro ⟨a', t'⟩ ⟨h1, h7, hlt, hst, hg⟩
    dsimp only at h1 h7 hlt hst hg ⊢
    unfold learn_port.e8.root_row_loop.body
    by_cases hc : t' < 7#usize
    · rw [if_pos hc]
      have ht : t'.val < 7 := by scalar_tac
      have hsq : t'.val * t'.val ≤ 15 * t'.val := by nlinarith
      step*
      have hi3 : i3.val = startR t'.val := by
        simp only [startR, i3_post, i2_post, i_post, i1_post]
      by_cases hle : i3 ≤ pair
      · rw [if_pos hle]
        have hle' : startR t'.val ≤ pair.val := by rw [← hi3]; scalar_tac
        step*
        refine ⟨by omega, by omega, by omega, hle', ?_, by omega⟩
        intro u hu1 hu2
        omega
      · rw [if_neg hle]
        have hle' : pair.val < startR t'.val := by rw [← hi3]; scalar_tac
        step*
        refine ⟨by omega, by omega, by omega, hst, ?_, by omega⟩
        intro u hu1 hu2
        by_cases hu : u < t'.val
        · exact hg u hu1 hu
        · have : u = t'.val := by omega
          subst this; exact hle'
    · rw [if_neg hc]
      have ht : t'.val = 7 := by scalar_tac
      simp only [WP.spec_ok]
      refine ⟨by omega, hst, ?_⟩
      intro s h1' h2'
      exact hg s h1' (by omega)
  · exact ⟨ht1, ht7, hat, hs, hgt⟩

theorem repeat_getD (j : ℕ) : ((Array.repeat 8#usize 0#i64).val.getD j 0#i64).val = 0 := by
  by_cases hj : j < 8
  · interval_cases j <;> rfl
  · have hl : ¬ j < (Array.repeat 8#usize 0#i64).val.length := by
      rw [(Array.repeat 8#usize 0#i64).property]; scalar_tac
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (Nat.not_lt.mp hl)]; rfl

/-! ## Specifications of the three row families -/

/-- Pairs `a < b < 8` in lexicographic order. -/
def rootPairs : List (ℕ × ℕ) :=
  (List.range 8).flatMap fun a => ((List.range 8).filter (a < ·)).map fun b => (a, b)

/-- Row `j` (`0 ≤ j < 112`) of the D8 roots: signs `±2` at the `(j/4)`-th pair. -/
def rootSpec (j k : ℕ) : ℤ :=
  let ab := rootPairs.getD (j / 4) (0, 0)
  if k = ab.1 then (if j % 4 / 2 = 0 then -(2 * ONE) else 2 * ONE)
  else if k = ab.2 then (if j % 2 = 0 then -(2 * ONE) else 2 * ONE)
  else 0

/-- Row `j` (`0 ≤ j < 16`) of the O8 vertices: `±√8 · e_{j/2}`. -/
def o8Spec (j k : ℕ) : ℤ :=
  if k = j / 2 then (if j % 2 = 0 then -SQRT8 else SQRT8) else 0

/-- Spinor row `i < 128`: the `i`-th even-parity sign vector, scaled by `ONE`. -/
def spinorSpec (i k : ℕ) : ℤ := signAt (spinorKey i) k

/-- The 256-row codebook, row-wise. -/
def codewordSpec (i k : ℕ) : ℤ :=
  if i < 128 then spinorSpec i k
  else if i < 240 then rootSpec (i - 128) k
  else o8Spec (i - 240) k

theorem rootPairs_length : rootPairs.length = 28 := by decide

theorem root_pair_link : ∀ p : Fin 28, ∀ a : Fin 7, startR a.val ≤ p.val →
    (∀ s : Fin 7, a.val < s.val → p.val < startR s.val) →
    (a.val, a.val + 1 + (p.val - startR a.val)) = rootPairs.getD p.val (0, 0) := by
  decide

theorem root_row_spec (k : Std.Usize) (hk : k.val < 112) :
    learn_port.e8.root_row k ⦃ o =>
      ∀ j < 8, (o.val.getD j 0#i64).val = rootSpec k.val j ⦄ := by
  unfold learn_port.e8.root_row
  step as ⟨pair, hpair⟩
  step as ⟨sub, hsub⟩
  step with root_row_loop_spec pair 0#usize 1#usize (by simp) (by simp) (by simp)
    (by simp [startR]) (by intro s h1 h2; simp at h1 h2; omega) as ⟨a, ha7, hsa, hgt⟩
  have hp28 : pair.val < 28 := by omega
  have hsa' : (15 * a.val - a.val * a.val) / 2 ≤ pair.val := hsa
  have hsq : a.val * a.val ≤ 15 * a.val := by nlinarith
  step*
  have hi4 : i4.val = startR a.val := by
    simp only [startR, i4_post, i3_post, i1_post, i2_post]
  have hlt7 : pair.val < startR a.val + (7 - a.val) := by
    by_cases ha6 : a.val + 1 < 7
    · have := hgt (a.val + 1) (by omega) ha6
      rwa [startR_succ a.val (by omega)] at this
    · have ha6' : a.val = 6 := by omega
      rw [ha6']; simp [startR]; omega
  have hbv : b.val = a.val + 1 + (pair.val - startR a.val) := by
    rw [b_post, i_post, i5_post, hi4]
  have hb8 : b.val < 8 := by omega
  have hlink := root_pair_link ⟨pair.val, hp28⟩ ⟨a.val, ha7⟩ hsa
    (fun s h1 => hgt s.val h1 s.isLt)
  have hab : (a.val, b.val) = rootPairs.getD (k.val / 4) (0, 0) := by
    rw [hbv, ← hpair]; exact hlink
  obtain ⟨v7, hv7, hv7v⟩ : ∃ v : Std.I64,
      (if i6 = 0#usize then learn_port.e8.NEG_TWO_ONE else learn_port.e8.TWO_ONE) = ok v ∧
      v.val = (if k.val % 4 / 2 = 0 then -(2 * ONE) else 2 * ONE) := by
    have h6 : i6.val = k.val % 4 / 2 := by rw [i6_post, hsub]
    by_cases h : i6 = 0#usize
    · have h0 : k.val % 4 / 2 = 0 := by rw [← h6, h]; rfl
      refine ⟨_, by rw [if_pos h, NEG_TWO_ONE_eq], ?_⟩
      rw [if_pos h0]; simp [ONE]
    · have h0 : ¬ k.val % 4 / 2 = 0 := fun h' => h (by scalar_tac)
      refine ⟨_, by rw [if_neg h, TWO_ONE_eq], ?_⟩
      rw [if_neg h0]; simp [ONE]
  rw [hv7]
  simp only [bind_tc_ok]
  step as ⟨out1, hout1⟩
  step as ⟨i8, hi8⟩
  obtain ⟨v9, hv9, hv9v⟩ : ∃ v : Std.I64,
      (if i8 = 0#usize then learn_port.e8.NEG_TWO_ONE else learn_port.e8.TWO_ONE) = ok v ∧
      v.val = (if k.val % 2 = 0 then -(2 * ONE) else 2 * ONE) := by
    have h8 : i8.val = k.val % 2 := by rw [hi8, hsub]; omega
    by_cases h : i8 = 0#usize
    · have h0 : k.val % 2 = 0 := by rw [← h8, h]; rfl
      refine ⟨_, by rw [if_pos h, NEG_TWO_ONE_eq], ?_⟩
      rw [if_pos h0]; simp [ONE]
    · have h0 : ¬ k.val % 2 = 0 := fun h' => h (by scalar_tac)
      refine ⟨_, by rw [if_neg h, TWO_ONE_eq], ?_⟩
      rw [if_neg h0]; simp [ONE]
  rw [hv9]
  simp only [bind_tc_ok]
  step as ⟨o, ho⟩
  rename_i j hj
  have hlen1 : a.val < (Array.repeat 8#usize 0#i64).val.length := by
    rw [(Array.repeat 8#usize 0#i64).property]; scalar_tac
  have hlen2 : b.val < out1.val.length := by rw [out1.property]; scalar_tac
  have hab1 : a.val = (rootPairs.getD (k.val / 4) (0, 0)).1 := by rw [← hab]
  have hab2 : b.val = (rootPairs.getD (k.val / 4) (0, 0)).2 := by rw [← hab]
  unfold rootSpec
  simp only []
  rw [← hab1, ← hab2, ho, Array.set_val_eq, list_getD_set_eq _ _ _ _ hlen2, hout1,
    Array.set_val_eq, list_getD_set_eq _ _ _ _ hlen1]
  by_cases hjb : b.val = j
  · rw [if_pos hjb, hv9v]
    have : ¬ j = a.val := by omega
    rw [if_neg this, if_pos hjb.symm]
  · rw [if_neg hjb]
    by_cases hja : a.val = j
    · rw [if_pos hja, hv7v, if_pos hja.symm]
    · rw [if_neg hja, if_neg (Ne.symm hja), if_neg (Ne.symm hjb)]
      exact repeat_getD j

theorem o8_row_spec (k : Std.Usize) (hk : k.val < 16) :
    learn_port.e8.o8_row k ⦃ o =>
      ∀ j < 8, (o.val.getD j 0#i64).val = o8Spec k.val j ⦄ := by
  unfold learn_port.e8.o8_row
  step as ⟨i, hi⟩
  obtain ⟨v, hv, hvv⟩ : ∃ v : Std.I64,
      (if i = 0#usize then learn_port.e8.NEG_SQRT8_Q32 else ok learn_port.e8.SQRT8_Q32) = ok v ∧
      v.val = (if k.val % 2 = 0 then -SQRT8 else SQRT8) := by
    by_cases h : i = 0#usize
    · have h0 : k.val % 2 = 0 := by rw [← hi, h]; rfl
      refine ⟨_, by rw [if_pos h, NEG_SQRT8_eq], ?_⟩
      rw [if_pos h0]; simp [SQRT8]
    · have h0 : ¬ k.val % 2 = 0 := fun h' => h (by scalar_tac)
      refine ⟨_, by rw [if_neg h], ?_⟩
      rw [if_neg h0, SQRT8_eq]; simp [SQRT8]
  rw [hv]
  simp only [bind_tc_ok]
  step as ⟨i2, hi2⟩
  step as ⟨o, ho⟩
  rename_i j hj
  have hlen : i2.val < (Array.repeat 8#usize 0#i64).val.length := by
    rw [(Array.repeat 8#usize 0#i64).property]; scalar_tac
  unfold o8Spec
  rw [ho, Array.set_val_eq, list_getD_set_eq _ _ _ _ hlen, hi2]
  by_cases hjk : k.val / 2 = j
  · rw [if_pos hjk, if_pos hjk.symm, hvv]
  · rw [if_neg hjk, if_neg (Ne.symm hjk)]
    exact repeat_getD j

theorem codeword_unchecked_spec (j : Std.Usize) (hj : j.val < 256) :
    learn_port.e8.codeword_unchecked j ⦃ o =>
      ∀ k < 8, (o.val.getD k 0#i64).val = codewordSpec j.val k ⦄ := by
  unfold learn_port.e8.codeword_unchecked
  by_cases h1 : j < 128#usize
  · rw [if_pos h1]
    have : j.val < 128 := by scalar_tac
    refine WP.spec_mono (spinor_row_spec j this) ?_
    intro o ho k hk
    rw [ho k hk]
    unfold codewordSpec spinorSpec
    rw [if_pos this]
  · rw [if_neg h1]
    have h1' : ¬ j.val < 128 := fun h => h1 (by scalar_tac)
    by_cases h2 : j < 240#usize
    · rw [if_pos h2]
      have h2' : j.val < 240 := by scalar_tac
      step as ⟨i, hi⟩
      refine WP.spec_mono (root_row_spec i (by omega)) ?_
      intro o ho k hk
      rw [ho k hk]
      unfold codewordSpec
      rw [if_neg h1', if_pos h2', hi]
    · rw [if_neg h2]
      have h2' : ¬ j.val < 240 := fun h => h2 (by scalar_tac)
      step as ⟨i, hi⟩
      refine WP.spec_mono (o8_row_spec i (by omega)) ?_
      intro o ho k hk
      rw [ho k hk]
      unfold codewordSpec
      rw [if_neg h1', if_neg h2', hi]

theorem e8_codeword_spec (idx : Std.Usize) :
    learn_port.e8.codeword idx ⦃ r =>
      (idx.val < 256 → ∃ a, r = core.result.Result.Ok a ∧
        ∀ k < 8, (a.val.getD k 0#i64).val = codewordSpec idx.val k) ∧
      (256 ≤ idx.val → r = core.result.Result.Err learn_port.e8.E8Error.IndexOutOfRange) ⦄ := by
  unfold learn_port.e8.codeword
  have hc : learn_port.e8.CODEBOOK_LEN = 256#usize := by
    unfold learn_port.e8.CODEBOOK_LEN; rfl
  rw [hc]
  by_cases h : idx < 256#usize
  · rw [if_pos h]
    have h' : idx.val < 256 := by scalar_tac
    step with codeword_unchecked_spec idx h' as ⟨a, ha⟩
    exact ⟨fun _ => ⟨a, rfl, ha⟩, fun h2 => absurd h2 (by omega)⟩
  · rw [if_neg h]
    have h' : ¬ idx.val < 256 := fun h2 => h (by scalar_tac)
    simp only [WP.spec_ok]
    exact ⟨fun h2 => absurd h2 h', fun _ => trivial⟩

/-! ## Spinor key and parity -/

/-- Number of negative entries among the first eight coordinates of `f`. -/
def negCount (f : ℕ → ℤ) : ℕ := ((List.range 8).map fun k => if f k < 0 then 1 else 0).sum

theorem spinorKey_even (r : ℕ) (hr : r < 128) : parityNat (spinorKey r) = 0 := by
  have h2 := parityNat_lt r
  rw [spinorKey_eq _ hr]
  rcases (by omega : parityNat r = 0 ∨ parityNat r = 1) with h | h
  · rw [h, Nat.add_zero, parity_double r hr, h]
  · rw [h, parity_double_succ r hr, h]

theorem spinorKey_lt (r : ℕ) (hr : r < 128) : spinorKey r < 256 := by
  have h2 := parityNat_lt r
  rw [spinorKey_eq _ hr]; omega

theorem signAt_neg (n : ℕ) : (if (if n % 2 = 1 then -ONE else ONE) < 0 then 1 else 0 : ℕ) = n % 2 := by
  have h := Nat.mod_two_eq_zero_or_one n
  rcases h with h | h <;> simp [h, ONE]

theorem negCount_signAt (key : ℕ) : negCount (signAt key) % 2 = parityNat key := by
  unfold negCount signAt
  simp only [signAt_neg]
  rw [parityNat_eq]
  simp [List.range_succ]
  omega

theorem spinor_key_parity (r : Std.Usize) (hr : r.val < 128) :
    ∃ p o, learn_port.e8.popcount_parity r = ok p ∧ p.val = parityNat r.val ∧
      learn_port.e8.spinor_row r = ok o ∧
      (∀ j < 8, (o.val.getD j 0#i64).val = signAt (2 * r.val + p.val) j) ∧
      negCount (fun j => (o.val.getD j 0#i64).val) % 2 = 0 := by
  obtain ⟨p, hp, hpv⟩ := WP.spec_imp_exists (popcount_parity_spec r (by omega))
  obtain ⟨o, ho, hov⟩ := WP.spec_imp_exists (spinor_row_spec r hr)
  have hkey : 2 * r.val + p.val = spinorKey r.val := by
    rw [spinorKey_eq _ hr, hpv]
  refine ⟨p, o, hp, hpv, ho, ?_, ?_⟩
  · intro j hj; rw [hkey]; exact hov j hj
  · have : negCount (fun j => (o.val.getD j 0#i64).val) = negCount (signAt (spinorKey r.val)) := by
      unfold negCount
      congr 1
      apply List.map_congr_left
      intro k hk
      have hk' : k < 8 := by simpa using hk
      dsimp only
      simp only [hov k hk']
    rw [this, negCount_signAt, spinorKey_even _ hr]

/-- Every spinor row is `ONE` times an element of `Perf.FaceTypedE8.Spin8`. -/
theorem e8_codeword_spin8 (i : ℕ) (hi : i < 128) :
    ∃ s : Perf.FaceTypedE8.Spin8, ∀ k : Fin 8, codewordSpec i k.val = ONE * ((s.1 k : ℤˣ) : ℤ) := by
  have hev := spinorKey_even i hi
  refine ⟨⟨fun k => (-1 : ℤˣ) ^ (spinorKey i / 2 ^ (7 - k.val) % 2), ?_⟩, ?_⟩
  · show ∏ p : Fin 8, (-1 : ℤˣ) ^ (spinorKey i / 2 ^ (7 - p.val) % 2) = 1
    have hsum : Even (∑ p : Fin 8, spinorKey i / 2 ^ (7 - p.val) % 2) := by
      have hF : (∑ p : Fin 8, spinorKey i / 2 ^ (7 - p.val) % 2) =
          spinorKey i / 128 % 2 + spinorKey i / 64 % 2 + spinorKey i / 32 % 2 +
          spinorKey i / 16 % 2 + spinorKey i / 8 % 2 + spinorKey i / 4 % 2 +
          spinorKey i / 2 % 2 + spinorKey i % 2 := by
        simp [Fin.sum_univ_eight]
      rw [hF, Nat.even_iff]
      rw [parityNat_eq] at hev
      generalize spinorKey i % 2 = a0 at *
      generalize spinorKey i / 2 % 2 = a1 at *
      generalize spinorKey i / 4 % 2 = a2 at *
      generalize spinorKey i / 8 % 2 = a3 at *
      generalize spinorKey i / 16 % 2 = a4 at *
      generalize spinorKey i / 32 % 2 = a5 at *
      generalize spinorKey i / 64 % 2 = a6 at *
      generalize spinorKey i / 128 % 2 = a7 at *
      omega
    exact (Finset.prod_pow_eq_pow_sum _ _ _).trans hsum.neg_one_pow
  · intro k
    unfold codewordSpec spinorSpec signAt
    rw [if_pos hi]
    rcases Nat.mod_two_eq_zero_or_one (spinorKey i / 2 ^ (7 - k.val)) with h | h
    · simp [h, ONE]
    · simp [h, ONE]
/-! ## i128 checked-arithmetic helpers -/

theorem checked_mul_some128 {a b : Std.I128}
    (h : Std.I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ Std.I128.max) :
    ∃ z, Std.I128.checked_mul a b = some z ∧ z.val = a.val * b.val := by
  have hspec : (match Std.I128.checked_mul a b with
      | some z => Std.I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ Std.I128.max
          ∧ z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => ¬ (Std.I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ Std.I128.max)) :=
    Std.I128.checked_mul_bv_spec a b
  cases hm : Std.I128.checked_mul a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

theorem checked_add_some128 {a b : Std.I128}
    (h : Std.I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I128.max) :
    ∃ z, Std.I128.checked_add a b = some z ∧ z.val = a.val + b.val := by
  have hspec : (match Std.I128.checked_add a b with
      | some z => Std.I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I128.max
          ∧ z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (Std.I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I128.max)) :=
    Std.I128.checked_add_bv_spec a b
  cases hm : Std.I128.checked_add a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

theorem checked_sub_some128 {a b : Std.I128}
    (h : Std.I128.min ≤ a.val - b.val ∧ a.val - b.val ≤ Std.I128.max) :
    ∃ z, Std.I128.checked_sub a b = some z ∧ z.val = a.val - b.val := by
  have hspec : (match Std.I128.checked_sub a b with
      | some z => Std.I128.min ≤ a.val - b.val ∧ a.val - b.val ≤ Std.I128.max
          ∧ z.val = a.val - b.val ∧ z.bv = a.bv - b.bv
      | none => ¬ (Std.I128.min ≤ a.val - b.val ∧ a.val - b.val ≤ Std.I128.max)) :=
    Std.I128.checked_sub_bv_spec a b
  cases hm : Std.I128.checked_sub a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

theorem checked_add_none128 {a b : Std.I128}
    (h : ¬ (Std.I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I128.max)) :
    Std.I128.checked_add a b = none := by
  have hspec : (match Std.I128.checked_add a b with
      | some z => Std.I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I128.max
          ∧ z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (Std.I128.min ≤ a.val + b.val ∧ a.val + b.val ≤ Std.I128.max)) :=
    Std.I128.checked_add_bv_spec a b
  cases hm : Std.I128.checked_add a b with
  | none => rfl
  | some z => rw [hm] at hspec; exact absurd hspec.1 (fun h1 => h ⟨h1, hspec.2.1⟩)

theorem checked_mul_none128 {a b : Std.I128}
    (h : ¬ (Std.I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ Std.I128.max)) :
    Std.I128.checked_mul a b = none := by
  have hspec : (match Std.I128.checked_mul a b with
      | some z => Std.I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ Std.I128.max
          ∧ z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => ¬ (Std.I128.min ≤ a.val * b.val ∧ a.val * b.val ≤ Std.I128.max)) :=
    Std.I128.checked_mul_bv_spec a b
  cases hm : Std.I128.checked_mul a b with
  | none => rfl
  | some z => rw [hm] at hspec; exact absurd hspec.1 (fun h1 => h ⟨h1, hspec.2.1⟩)


theorem shl_one_31 : ∃ i2, (1#i128 <<< 31#i32) = ok i2 ∧ i2.val = 2 ^ 31 := by
  have h := Std.I128.ShiftLeft_IScalar_spec (1#i128) (31#i32) (by scalar_tac) (by scalar_tac)
  obtain ⟨z, hz, hv, _⟩ := WP.spec_imp_exists h
  refine ⟨z, hz, ?_⟩
  rw [hv]
  simp [Int.bmod, Int.shiftLeft_eq]
  have hsz : ((Std.I128.size : ℕ) : ℤ) = 2 ^ 128 := by
    simp [Std.I128.size, Std.I128.numBits_def]
  rw [hsz]
  norm_num

theorem shr_32 (t : Std.I128) : ∃ i4, (t >>> 32#i32) = ok i4 ∧ i4.val = t.val / 2 ^ 32 := by
  have h := Std.I128.ShiftRight_IScalar_spec t (32#i32) (by scalar_tac) (by scalar_tac)
  obtain ⟨z, hz, hv, _⟩ := WP.spec_imp_exists h
  refine ⟨z, hz, ?_⟩
  rw [hv]
  simp [Int.shiftRight_eq_div_pow]

theorem scaled_term_spec (x c scale : Std.I64) (hs : scale.val = 4294967296)
    (hx : |x.val| ≤ 2 ^ 63) (hc : |c.val| ≤ 2 ^ 35) :
    ∃ z, learn_port.e8.scaled_term x c scale = ok (some z) ∧ z.val = (x.val - c.val) ^ 2 := by
  unfold learn_port.e8.scaled_term
  simp only [lift, bind_tc_ok]
  have hxl := abs_le.mp hx
  have hcl := abs_le.mp hc
  have hI : Std.I128.min = -2 ^ 127 ∧ Std.I128.max = 2 ^ 127 - 1 := by
    constructor <;> simp [Std.I128.min_def, Std.I128.max_def, Std.I128.numBits_def]
  have hf1 := core.convert.num.FromI128I64.from_val_eq scale
  have hf2 := core.convert.num.FromI128I64.from_val_eq c
  have hf3 := core.convert.num.FromI128I64.from_val_eq x
  obtain ⟨sc, hsc, hscv⟩ := checked_mul_some128 (a := core.convert.num.FromI128I64.from scale)
    (b := core.convert.num.FromI128I64.from c) (by
      rw [hf1, hf2, hs, hI.1, hI.2]
      constructor <;> nlinarith [hcl.1, hcl.2])
  rw [hsc]
  simp only []
  obtain ⟨i2, hi2, hi2v⟩ := shl_one_31
  rw [hi2]
  simp only [bind_tc_ok]
  rw [hf1, hf2, hs] at hscv
  obtain ⟨t, ht, htv⟩ := checked_add_some128 (a := sc) (b := i2) (by
    rw [hscv, hi2v, hI.1, hI.2]
    constructor <;> nlinarith [hcl.1, hcl.2])
  rw [ht]
  simp only []
  obtain ⟨i4, hi4, hi4v⟩ := shr_32 t
  rw [hi4]
  simp only [bind_tc_ok]
  have hi4c : i4.val = c.val := by
    rw [hi4v, htv, hscv, hi2v]; omega
  obtain ⟨e, he, hev⟩ := checked_sub_some128 (a := core.convert.num.FromI128I64.from x) (b := i4) (by
    rw [hf3, hi4c, hI.1, hI.2]
    constructor <;> nlinarith [hcl.1, hcl.2, hxl.1, hxl.2])
  rw [he]
  simp only []
  rw [hf3, hi4c] at hev
  have he1 : -(2 ^ 63 + 2 ^ 35) ≤ x.val - c.val := by linarith [hcl.1, hcl.2, hxl.1, hxl.2]
  have he2 : x.val - c.val ≤ 2 ^ 63 + 2 ^ 35 := by linarith [hcl.1, hcl.2, hxl.1, hxl.2]
  obtain ⟨z, hz, hzv⟩ := checked_mul_some128 (a := e) (b := e) (by
    rw [hev, hI.1, hI.2]
    constructor
    · nlinarith [mul_self_nonneg (x.val - c.val)]
    · nlinarith [mul_nonneg (sub_nonneg.2 he2) (by linarith : (0:ℤ) ≤ 2 ^ 63 + 2 ^ 35 + (x.val - c.val))])
  rw [hz]
  refine ⟨z, rfl, ?_⟩
  rw [hzv, hev]; ring


/-- Entry `k` of an eight-word row as an integer. -/
def xv (x : Array Std.I64 8#usize) (k : ℕ) : ℤ := (x.val.getD k 0#i64).val

/-- Squared Euclidean distance over the first `n` coordinates. -/
def tsum (x row : Array Std.I64 8#usize) (n : ℕ) : ℤ :=
  ∑ k ∈ Finset.range n, (xv x k - xv row k) ^ 2

theorem term_bound (a b : ℤ) (ha : |a| ≤ 2 ^ 60) (hb : |b| ≤ 2 ^ 35) :
    0 ≤ (a - b) ^ 2 ∧ (a - b) ^ 2 ≤ 2 ^ 122 := by
  have h1 := abs_le.mp ha
  have h2 := abs_le.mp hb
  refine ⟨sq_nonneg _, ?_⟩
  nlinarith [h1.1, h1.2, h2.1, h2.2]

theorem tsum_bound (x row : Array Std.I64 8#usize)
    (hx : ∀ k < 8, |xv x k| ≤ 2 ^ 60) (hr : ∀ k < 8, |xv row k| ≤ 2 ^ 35) :
    ∀ n ≤ 8, 0 ≤ tsum x row n ∧ tsum x row n ≤ n * 2 ^ 122 := by
  intro n
  induction n with
  | zero => intro _; simp [tsum]
  | succ m ih =>
    intro hm
    have h := ih (by omega)
    have hb := term_bound (xv x m) (xv row m) (hx m (by omega)) (hr m (by omega))
    unfold tsum at h ⊢
    rw [Finset.sum_range_succ]
    push_cast
    constructor <;> nlinarith [h.1, h.2, hb.1, hb.2]

theorem scaled_dist2_loop_spec (x row : Array Std.I64 8#usize) (scale : Std.I64)
    (hs : scale.val = 4294967296)
    (hx : ∀ k < 8, |xv x k| ≤ 2 ^ 60) (hr : ∀ k < 8, |xv row k| ≤ 2 ^ 35)
    (acc : Std.I128) (i : Std.Usize) (hi : i.val ≤ 8) (hacc : acc.val = tsum x row i.val) :
    learn_port.e8.scaled_dist2_loop x row scale acc i true ⦃ r =>
      r.2 = true ∧ r.1.val = tsum x row 8 ⦄ := by
  unfold learn_port.e8.scaled_dist2_loop
  have hI : Std.I128.min = -2 ^ 127 ∧ Std.I128.max = 2 ^ 127 - 1 := by
    constructor <;> simp [Std.I128.min_def, Std.I128.max_def, Std.I128.numBits_def]
  apply loop.spec_decr_nat
    (measure := fun s : Std.I128 × Std.Usize × Bool => 8 - s.2.1.val)
    (inv := fun s => s.2.1.val ≤ 8 ∧ s.2.2 = true ∧ s.1.val = tsum x row s.2.1.val)
    (post := fun r : Std.I128 × Bool => r.2 = true ∧ r.1.val = tsum x row 8)
  · rintro ⟨a, i', ok'⟩ ⟨hi', hok, ha'⟩
    dsimp only at hi' hok ha' ⊢
    subst hok
    unfold learn_port.e8.scaled_dist2_loop.body
    by_cases hlt : i' < 8#usize
    · rw [if_pos hlt]
      have hk8 : i'.val < 8 := by scalar_tac
      simp only [if_true]
      step as ⟨i1, hi1⟩
      step as ⟨i2, hi2⟩
      have hlx : i'.val < x.val.length := by rw [x.property]; scalar_tac
      have hlr : i'.val < row.val.length := by rw [row.property]; scalar_tac
      have e1 : i1.val = xv x i'.val := by
        rw [hi1]; unfold xv; rw [List.getD_eq_getElem _ _ hlx]
      have e2 : i2.val = xv row i'.val := by
        rw [hi2]; unfold xv; rw [List.getD_eq_getElem _ _ hlr]
      have hxi : |i1.val| ≤ 2 ^ 60 := by
        rw [e1]; exact hx _ hk8
      have hri : |i2.val| ≤ 2 ^ 35 := by
        rw [e2]; exact hr _ hk8
      obtain ⟨z, hz, hzv⟩ := scaled_term_spec i1 i2 scale hs (le_trans hxi (by norm_num)) hri
      rw [hz]
      simp only [bind_tc_ok]
      have hb := term_bound _ _ hxi hri
      have hts := tsum_bound x row hx hr i'.val (by omega)
      obtain ⟨s, hsum, hsv⟩ := checked_add_some128 (a := a) (b := z) (by
        rw [ha', hzv, hI.1, hI.2]
        constructor <;> nlinarith [hts.1, hts.2, hb.1, hb.2])
      rw [hsum]
      simp only [lift, bind_tc_ok]
      step as ⟨i3, hi3⟩
      have hi3' : i3.val = i'.val + 1 := hi3
      refine ⟨by omega, ?_, by omega⟩
      rw [hsv, hzv, ha', hi3', e1, e2]
      unfold tsum
      rw [Finset.sum_range_succ]
    · rw [if_neg hlt]
      simp only [WP.spec_ok]
      have h8 : i'.val = 8 := by scalar_tac
      refine ⟨trivial, ?_⟩
      rw [ha', h8]
  · exact ⟨hi, rfl, hacc⟩


theorem codeword_bound (j k : ℕ) : |codewordSpec j k| ≤ 2 ^ 35 := by
  rw [abs_le]
  unfold codewordSpec
  split_ifs <;>
    simp only [spinorSpec, signAt, rootSpec, o8Spec, ONE, SQRT8] <;>
    split_ifs <;> norm_num

/-- Squared distance from `x` to codeword `j`, over `ℤ`. -/
def dist (x : Array Std.I64 8#usize) (j : ℕ) : ℤ :=
  ∑ k ∈ Finset.range 8, (xv x k - codewordSpec j k) ^ 2

theorem row_dist_spec (x : Array Std.I64 8#usize) (j : Std.Usize) (scale : Std.I64)
    (hs : scale.val = 4294967296) (hx : ∀ k < 8, |xv x k| ≤ 2 ^ 60) (hj : j.val < 256) :
    learn_port.e8.row_dist x j scale ⦃ r => r.2 = true ∧ r.1.val = dist x j.val ⦄ := by
  unfold learn_port.e8.row_dist
  step with codeword_unchecked_spec j hj as ⟨row, hrow⟩
  have hr : ∀ k < 8, |xv row k| ≤ 2 ^ 35 := by
    intro k hk
    have : xv row k = codewordSpec j.val k := hrow k hk
    rw [this]; exact codeword_bound _ _
  unfold learn_port.e8.scaled_dist2
  refine WP.spec_mono (scaled_dist2_loop_spec x row scale hs hx hr 0#i128 0#usize (by simp)
    (by simp [tsum])) ?_
  rintro ⟨d, g⟩ ⟨hg, hd⟩
  refine ⟨hg, ?_⟩
  rw [hd]
  unfold tsum dist
  refine Finset.sum_congr rfl ?_
  intro k hk
  have hk8 : k < 8 := Finset.mem_range.mp hk
  have : xv row k = codewordSpec j.val k := hrow k hk8
  rw [this]


theorem nearest_loop_spec (x : Array Std.I64 8#usize) (scale : Std.I64)
    (hs : scale.val = 4294967296) (hx : ∀ k < 8, |xv x k| ≤ 2 ^ 60)
    (best : Std.Usize) (best_d : Std.I128) (j : Std.Usize)
    (hj1 : 1 ≤ j.val) (hj : j.val ≤ 256) (hb : best.val < j.val)
    (hbd : best_d.val = dist x best.val)
    (hle : ∀ i < j.val, dist x best.val ≤ dist x i)
    (hlt : ∀ i < best.val, dist x best.val < dist x i) :
    learn_port.e8.nearest_scaled_loop x scale best best_d true j ⦃ r =>
      r.2 = true ∧ r.1.val < 256 ∧ (∀ i < 256, dist x r.1.val ≤ dist x i) ∧
        (∀ i < r.1.val, dist x r.1.val < dist x i) ⦄ := by
  unfold learn_port.e8.nearest_scaled_loop
  apply loop.spec_decr_nat
    (measure := fun s : Std.Usize × Std.I128 × Bool × Std.Usize => 256 - s.2.2.2.val)
    (inv := fun s => 1 ≤ s.2.2.2.val ∧ s.2.2.2.val ≤ 256 ∧ s.2.2.1 = true ∧
      s.1.val < s.2.2.2.val ∧ s.2.1.val = dist x s.1.val ∧
      (∀ i < s.2.2.2.val, dist x s.1.val ≤ dist x i) ∧
      (∀ i < s.1.val, dist x s.1.val < dist x i))
    (post := fun r : Std.Usize × Bool =>
      r.2 = true ∧ r.1.val < 256 ∧ (∀ i < 256, dist x r.1.val ≤ dist x i) ∧
        (∀ i < r.1.val, dist x r.1.val < dist x i))
  · rintro ⟨b, bd, ok', j'⟩ ⟨hj1', hj', hok, hb', hbd', hle', hlt'⟩
    dsimp only at hj1' hj' hok hb' hbd' hle' hlt' ⊢
    subst hok
    unfold learn_port.e8.nearest_scaled_loop.body
    have hc : learn_port.e8.CODEBOOK_LEN = 256#usize := by
      unfold learn_port.e8.CODEBOOK_LEN; rfl
    rw [hc]
    by_cases hlt : j' < 256#usize
    · rw [if_pos hlt]
      have hk : j'.val < 256 := by scalar_tac
      step with row_dist_spec x j' scale hs hx hk as ⟨d, g, hg, hd⟩
      subst hg
      simp only [if_true, bind_tc_ok]
      by_cases hdl : d < bd
      · rw [if_pos hdl]
        simp only [bind_tc_ok]
        step as ⟨j1, hj1v⟩
        have hdl' : d.val < bd.val := by scalar_tac
        have hj1' : j1.val = j'.val + 1 := hj1v
        refine ⟨by omega, by omega, by omega, hd, ?_, ?_, by omega⟩
        · intro i hi
          rcases Nat.lt_succ_iff_lt_or_eq.mp (by omega : i < j'.val + 1) with h | h
          · have := hle' i h
            rw [hbd'] at hdl'
            omega
          · subst h; omega
        · intro i hi
          have := hle' i (by omega)
          rw [hbd'] at hdl'
          omega
      · rw [if_neg hdl]
        simp only [bind_tc_ok]
        step as ⟨j1, hj1v⟩
        have hdl' : ¬ d.val < bd.val := by scalar_tac
        have hj1' : j1.val = j'.val + 1 := hj1v
        refine ⟨by omega, by omega, by omega, hbd', ?_, hlt', by omega⟩
        intro i hi
        rcases Nat.lt_succ_iff_lt_or_eq.mp (by omega : i < j'.val + 1) with h | h
        · exact hle' i h
        · subst h; rw [hbd', hd] at *; omega
    · rw [if_neg hlt]
      simp only [WP.spec_ok]
      have h256 : j'.val = 256 := by scalar_tac
      refine ⟨trivial, by omega, ?_, hlt'⟩
      intro i hi
      exact hle' i (by omega)
  · exact ⟨hj1, hj, rfl, hb, hbd, hle, hlt⟩


/-- The nearest codeword: the least index minimising the squared distance over `ℤ`. -/
theorem e8_nearest_spec (x : Array Std.I64 8#usize) (hx : ∀ k < 8, |xv x k| ≤ 2 ^ 60) :
    ∃ j : Std.Usize, learn_port.e8.nearest x = ok (core.result.Result.Ok j) ∧ j.val < 256 ∧
      (∀ i < 256, dist x j.val ≤ dist x i) ∧ (∀ i < j.val, dist x j.val < dist x i) := by
  unfold learn_port.e8.nearest
  rw [engine_ONE_eq]
  simp only [bind_tc_ok]
  unfold learn_port.e8.nearest_scaled
  have hs : (4294967296#i64).val = 4294967296 := by rfl
  have hneg : ¬ (4294967296#i64 <= 0#i64) := by scalar_tac
  generalize (4294967296#i64) = S at hs hneg ⊢
  rw [if_neg hneg]
  have h0 := row_dist_spec x 0#usize S hs hx (by scalar_tac)
  obtain ⟨⟨d0, ok0⟩, hr0, hp0⟩ := WP.spec_imp_exists h0
  obtain ⟨hok0, hd0⟩ := hp0
  dsimp only at hok0 hd0
  subst hok0
  rw [hr0]
  simp only [bind_tc_ok]
  have hl := nearest_loop_spec x S hs hx 0#usize d0 1#usize (by scalar_tac)
    (by scalar_tac) (by scalar_tac) hd0
    (by intro i hi; have : i = 0 := by scalar_tac
        subst this; exact le_refl _)
    (by intro i hi; scalar_tac)
  obtain ⟨⟨b, okb⟩, hrb, hpb⟩ := WP.spec_imp_exists hl
  obtain ⟨hokb, hb256, hble, hblt⟩ := hpb
  dsimp only at hokb hb256 hble hblt
  subst hokb
  refine ⟨b, ?_, hb256, hble, hblt⟩
  simp [hrb]


theorem scaled_term_total (x c scale : Std.I64) :
    ∃ o, learn_port.e8.scaled_term x c scale = ok o := by
  unfold learn_port.e8.scaled_term
  simp only [lift, bind_tc_ok]
  rcases hm : Std.I128.checked_mul (core.convert.num.FromI128I64.from scale)
      (core.convert.num.FromI128I64.from c) with _ | sc
  · exact ⟨_, rfl⟩
  · simp only []
    obtain ⟨i2, hi2, _⟩ := shl_one_31
    rw [hi2]
    simp only [bind_tc_ok]
    rcases ht : Std.I128.checked_add sc i2 with _ | t
    · exact ⟨_, rfl⟩
    · simp only []
      obtain ⟨i4, hi4, _⟩ := shr_32 t
      rw [hi4]
      simp only [bind_tc_ok]
      rcases he : Std.I128.checked_sub (core.convert.num.FromI128I64.from x) i4 with _ | e
      · exact ⟨_, rfl⟩
      · exact ⟨_, rfl⟩

theorem scaled_dist2_loop_total (x row : Array Std.I64 8#usize) (scale : Std.I64)
    (acc : Std.I128) (i : Std.Usize) (ok1 : Bool) (hi : i.val ≤ 8) :
    learn_port.e8.scaled_dist2_loop x row scale acc i ok1 ⦃ _ => True ⦄ := by
  unfold learn_port.e8.scaled_dist2_loop
  apply loop.spec_decr_nat
    (measure := fun s : Std.I128 × Std.Usize × Bool => 8 - s.2.1.val)
    (inv := fun s => s.2.1.val ≤ 8)
    (post := fun _ : Std.I128 × Bool => True)
  · rintro ⟨a, i', ok'⟩ hi'
    dsimp only at hi' ⊢
    unfold learn_port.e8.scaled_dist2_loop.body
    by_cases hlt : i' < 8#usize
    · rw [if_pos hlt]
      have hk8 : i'.val < 8 := by scalar_tac
      by_cases hok : ok' = true
      · subst hok
        simp only [if_true]
        step as ⟨i1, hi1⟩
        step as ⟨i2, hi2⟩
        obtain ⟨o, ho⟩ := scaled_term_total i1 i2 scale
        rw [ho]
        simp only [bind_tc_ok]
        rcases o with _ | z
        · simp only []
          step as ⟨i3, hi3⟩
          have hi3' : i3.val = i'.val + 1 := hi3
          refine ⟨by omega, by omega⟩
        · simp only []
          rcases ha : Std.I128.checked_add a z with _ | s
          · simp only [lift, bind_tc_ok]
            step as ⟨i3, hi3⟩
            have hi3' : i3.val = i'.val + 1 := hi3
            refine ⟨by omega, by omega⟩
          · simp only [lift, bind_tc_ok]
            step as ⟨i3, hi3⟩
            have hi3' : i3.val = i'.val + 1 := hi3
            refine ⟨by omega, by omega⟩
      · have : ok' = false := by simpa using hok
        subst this
        simp
    · rw [if_neg hlt]
      simp only [WP.spec_ok]
  · exact hi


theorem row_dist_total (x : Array Std.I64 8#usize) (j : Std.Usize) (scale : Std.I64)
    (hj : j.val < 256) : learn_port.e8.row_dist x j scale ⦃ _ => True ⦄ := by
  unfold learn_port.e8.row_dist
  step with codeword_unchecked_spec j hj as ⟨row, hrow⟩
  unfold learn_port.e8.scaled_dist2
  exact scaled_dist2_loop_total x row scale 0#i128 0#usize true (by simp)

/-- Once the overflow flag is false the scan returns false, whatever the rows are. -/
theorem nearest_loop_false (x : Array Std.I64 8#usize) (scale : Std.I64)
    (best : Std.Usize) (best_d : Std.I128) (j : Std.Usize) (hj : j.val ≤ 256) :
    learn_port.e8.nearest_scaled_loop x scale best best_d false j ⦃ r => r.2 = false ⦄ := by
  unfold learn_port.e8.nearest_scaled_loop
  apply loop.spec_decr_nat
    (measure := fun s : Std.Usize × Std.I128 × Bool × Std.Usize => 256 - s.2.2.2.val)
    (inv := fun s => s.2.2.2.val ≤ 256 ∧ s.2.2.1 = false)
    (post := fun r : Std.Usize × Bool => r.2 = false)
  · rintro ⟨b, bd, ok', j'⟩ ⟨hj', hok⟩
    dsimp only at hj' hok ⊢
    subst hok
    unfold learn_port.e8.nearest_scaled_loop.body
    have hc : learn_port.e8.CODEBOOK_LEN = 256#usize := by
      unfold learn_port.e8.CODEBOOK_LEN; rfl
    rw [hc]
    by_cases hlt : j' < 256#usize
    · rw [if_pos hlt]
      have hk : j'.val < 256 := by scalar_tac
      step with row_dist_total x j' scale hk as ⟨d, g⟩
      simp only [Bool.false_eq_true, if_false, bind_tc_ok]
      by_cases hdl : d < bd
      · rw [if_pos hdl]
        simp only [bind_tc_ok]
        step as ⟨j1, hj1v⟩
        have hj1' : j1.val = j'.val + 1 := hj1v
        refine ⟨by omega, by omega⟩
      · rw [if_neg hdl]
        simp only [bind_tc_ok]
        step as ⟨j1, hj1v⟩
        have hj1' : j1.val = j'.val + 1 := hj1v
        refine ⟨by omega, by omega⟩
    · rw [if_neg hlt]
      simp only [WP.spec_ok]
  · exact ⟨hj, rfl⟩


/-- `x` with every coordinate `i64::MIN`, against a row of `ONE`: the third addition of the
squared distance leaves `i128`, so the overflow flag clears. -/
theorem overflow_dist2_loop (x row : Array Std.I64 8#usize) (scale : Std.I64)
    (hs : scale.val = 4294967296)
    (hxk : ∀ k < 8, xv x k = -2 ^ 63) (hrk : ∀ k < 8, xv row k = 4294967296)
    (acc : Std.I128) (i : Std.Usize)
    (h0 : i.val = 0 ∧ acc.val = 0) :
    learn_port.e8.scaled_dist2_loop x row scale acc i true ⦃ r => r.2 = false ⦄ := by
  unfold learn_port.e8.scaled_dist2_loop
  have hI : Std.I128.min = -2 ^ 127 ∧ Std.I128.max = 2 ^ 127 - 1 := by
    constructor <;> simp [Std.I128.min_def, Std.I128.max_def, Std.I128.numBits_def]
  apply loop.spec_decr_nat
    (measure := fun s : Std.I128 × Std.Usize × Bool => 8 - s.2.1.val)
    (inv := fun s => s.2.1.val ≤ 8 ∧ (s.2.2 = false ∨ (s.2.2 = true ∧
      ((s.2.1.val = 0 ∧ s.1.val = 0) ∨
       (s.2.1.val = 1 ∧ s.1.val = (-2 ^ 63 - 4294967296) ^ 2)))))
    (post := fun r : Std.I128 × Bool => r.2 = false)
  · rintro ⟨a, i', ok'⟩ ⟨hi', hinv⟩
    dsimp only at hi' hinv ⊢
    unfold learn_port.e8.scaled_dist2_loop.body
    by_cases hlt : i' < 8#usize
    · rw [if_pos hlt]
      have hk8 : i'.val < 8 := by scalar_tac
      rcases hinv with hok | ⟨hok, hcase⟩
      · subst hok
        simp
      · subst hok
        simp only [if_true]
        step as ⟨i1, hi1⟩
        step as ⟨i2, hi2⟩
        have hlx : i'.val < x.val.length := by rw [x.property]; scalar_tac
        have hlr : i'.val < row.val.length := by rw [row.property]; scalar_tac
        have e1 : i1.val = -2 ^ 63 := by
          rw [← hxk i'.val hk8, hi1]; unfold xv; rw [List.getD_eq_getElem _ _ hlx]
        have e2 : i2.val = 4294967296 := by
          rw [← hrk i'.val hk8, hi2]; unfold xv; rw [List.getD_eq_getElem _ _ hlr]
        obtain ⟨z, hz, hzv⟩ := scaled_term_spec i1 i2 scale hs (by rw [e1]; norm_num)
          (by rw [e2]; norm_num)
        rw [hz]
        simp only [bind_tc_ok]
        rw [e1, e2] at hzv
        have hzv' : z.val = (-2 ^ 63 - 4294967296) ^ 2 := by rw [hzv]
        rcases hcase with ⟨hi0, ha0⟩ | ⟨hi1', ha1⟩
        · obtain ⟨s, hsum, hsv⟩ := checked_add_some128 (a := a) (b := z) (by
            rw [ha0, hzv', hI.1, hI.2]; norm_num)
          simp only [lift, bind_tc_ok]
          rw [hsum]
          simp only []
          step as ⟨i3, hi3⟩
          have hi3' : i3.val = i'.val + 1 := hi3
          refine ⟨by omega, Or.inr (Or.inr ⟨by omega, by rw [hsv, ha0, hzv']; ring⟩), by omega⟩
        · have hnone := checked_add_none128 (a := a) (b := z) (by
            rw [ha1, hzv', hI.1, hI.2]; norm_num)
          simp only [lift, bind_tc_ok]
          rw [hnone]
          simp only []
          step as ⟨i3, hi3⟩
          have hi3' : i3.val = i'.val + 1 := hi3
          refine ⟨by omega, Or.inl trivial, by omega⟩
    · rw [if_neg hlt]
      simp only [WP.spec_ok]
      have h8 : i'.val = 8 := by scalar_tac
      rcases hinv with hok | ⟨hok, hcase⟩
      · exact hok
      · rcases hcase with ⟨h, _⟩ | ⟨h, _⟩ <;> omega
  · exact ⟨by show i.val ≤ 8; omega, Or.inr ⟨rfl, Or.inl h0⟩⟩


/-! ## Witnesses: one reachable input per success branch and per `Err` arm -/

/-- The all-`i64::MIN` input. -/
def xmin : Array Std.I64 8#usize := Array.repeat 8#usize ((-9223372036854775808)#i64)

/-- The all-zero input. -/
def xzero : Array Std.I64 8#usize := Array.repeat 8#usize 0#i64

theorem xmin_val (k : ℕ) (hk : k < 8) : xv xmin k = -2 ^ 63 := by
  unfold xv xmin
  simp [Array.repeat, List.getD_eq_getElem?_getD, hk]
  interval_cases k <;> rfl

theorem xzero_val (k : ℕ) (hk : k < 8) : xv xzero k = 0 := by
  unfold xv xzero
  simp [Array.repeat, List.getD_eq_getElem?_getD, hk]
  interval_cases k <;> rfl


theorem witness_codeword_ok :
    (∃ a, learn_port.e8.codeword 0#usize = ok (core.result.Result.Ok a)) ∧
    (∃ a, learn_port.e8.codeword 127#usize = ok (core.result.Result.Ok a)) ∧
    (∃ a, learn_port.e8.codeword 128#usize = ok (core.result.Result.Ok a)) ∧
    (∃ a, learn_port.e8.codeword 239#usize = ok (core.result.Result.Ok a)) ∧
    (∃ a, learn_port.e8.codeword 240#usize = ok (core.result.Result.Ok a)) ∧
    (∃ a, learn_port.e8.codeword 255#usize = ok (core.result.Result.Ok a)) := by
  have key : ∀ idx : Std.Usize, idx.val < 256 →
      ∃ a, learn_port.e8.codeword idx = ok (core.result.Result.Ok a) := by
    intro idx h
    obtain ⟨r, hr, hp⟩ := WP.spec_imp_exists (e8_codeword_spec idx)
    obtain ⟨a, ha, _⟩ := hp.1 h
    exact ⟨a, by rw [hr, ha]⟩
  exact ⟨key _ (by scalar_tac), key _ (by scalar_tac), key _ (by scalar_tac),
    key _ (by scalar_tac), key _ (by scalar_tac), key _ (by scalar_tac)⟩

theorem witness_codeword_err :
    learn_port.e8.codeword 256#usize =
      ok (core.result.Result.Err learn_port.e8.E8Error.IndexOutOfRange) := by
  obtain ⟨r, hr, hp⟩ := WP.spec_imp_exists (e8_codeword_spec 256#usize)
  rw [hr, hp.2 (by scalar_tac)]

theorem witness_nonpositive_scale (x : Array Std.I64 8#usize) :
    learn_port.e8.nearest_scaled x 0#i64 =
      ok (core.result.Result.Err learn_port.e8.E8Error.NonPositiveScale) := by
  unfold learn_port.e8.nearest_scaled
  have h : (0#i64 <= 0#i64) := by scalar_tac
  rw [if_pos h]

theorem witness_nearest_ok :
    ∃ j : Std.Usize, learn_port.e8.nearest xzero = ok (core.result.Result.Ok j) := by
  obtain ⟨j, hj, _⟩ := e8_nearest_spec xzero (by
    intro k hk
    rw [xzero_val k hk]; norm_num)
  exact ⟨j, hj⟩

theorem witness_overflow :
    learn_port.e8.nearest xmin =
      ok (core.result.Result.Err learn_port.e8.E8Error.Overflow) := by
  unfold learn_port.e8.nearest
  rw [engine_ONE_eq]
  simp only [bind_tc_ok]
  unfold learn_port.e8.nearest_scaled
  have hs : (4294967296#i64).val = 4294967296 := by rfl
  have hneg : ¬ (4294967296#i64 <= 0#i64) := by scalar_tac
  generalize (4294967296#i64) = S at hs hneg ⊢
  rw [if_neg hneg]
  have h0 : learn_port.e8.row_dist xmin 0#usize S ⦃ r => r.2 = false ⦄ := by
    unfold learn_port.e8.row_dist
    step with codeword_unchecked_spec 0#usize (by scalar_tac) as ⟨row, hrow⟩
    unfold learn_port.e8.scaled_dist2
    refine overflow_dist2_loop xmin row S hs (fun k hk => xmin_val k hk) ?_ 0#i128 0#usize
      ⟨rfl, rfl⟩
    intro k hk
    have : xv row k = codewordSpec 0 k := hrow k hk
    rw [this]
    have hk0 : spinorKey 0 = 0 := by
      rw [spinorKey_eq 0 (by norm_num)]; simp [parityNat_eq]
    simp [codewordSpec, spinorSpec, signAt, hk0, ONE]
  obtain ⟨⟨d0, ok0⟩, hr0, hp0⟩ := WP.spec_imp_exists h0
  dsimp only at hp0
  subst hp0
  rw [hr0]
  simp only [bind_tc_ok]
  have hl := nearest_loop_false xmin S 0#usize d0 1#usize (by scalar_tac)
  obtain ⟨⟨b, okb⟩, hrb, hpb⟩ := WP.spec_imp_exists hl
  dsimp only at hpb
  subst hpb
  simp [hrb]

end Learn.E8Code
