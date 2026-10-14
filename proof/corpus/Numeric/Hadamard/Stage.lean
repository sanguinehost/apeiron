/-
Pure integer Walsh–Hadamard reference and its helper laws (
spec the ternary-hadamard note).

Carved verbatim from `Numeric.HadamardRefinement` to keep both files under the source-size cap. It
holds the extractor-independent reference — `fwhtStage s` (one in-place stage `(u, v) ↦ (u + v,
u − v)` at span `2^s`), `fwhtRef k` (the stages `0, …, k-1` applied left to right), the concatenation
and block-butterfly lemmas, the FWHT/Sylvester identity `fwhtRef_getElem` and the aligned-segment law
`fwhtRef_segment` — together with the extracted `block_exponent` laws and the integer window bounds
`i64Ceil`, `fwhtStage_abs_le` and `fwhtRef_abs_le`. The extracted `hadamard.fwht_int` law and the loop
assemblies stay in `Numeric.HadamardRefinement`.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom` or `decide` over an
extracted body appears.
-/
import Numeric.Hadamard.Sylvester
import Numeric.Engine.Norm
import Numeric.Scale

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

open scoped Matrix

noncomputable section

/-! ## The pure integer reference -/

/-- One in-place FWHT stage at span `2^s` over the whole list. Each base multiple of `2^(s+1)`
carries a block of `2^(s+1)` consecutive entries; lane `l < 2^s` at `base + l` (the `u`) and
`base + 2^s + l` (the `v`) map to `u + v` and `u - v`. A trailing partial block shorter than
`2^(s+1)` is left unchanged. -/
def fwhtStage (s : ℕ) (xs : List ℤ) : List ℤ :=
  if _h : xs.length < 2 ^ (s + 1) then xs
  else
    (xs.take (2 ^ s)).zipWith (· + ·) ((xs.drop (2 ^ s)).take (2 ^ s)) ++
      (xs.take (2 ^ s)).zipWith (· - ·) ((xs.drop (2 ^ s)).take (2 ^ s)) ++
        fwhtStage s (xs.drop (2 ^ (s + 1)))
termination_by xs.length
decreasing_by
  have hpos : 0 < 2 ^ s := by positivity
  simp only [List.length_drop]
  omega

/-- The integer FWHT reference: the stages `0, …, k-1` applied left to right. -/
def fwhtRef (k : ℕ) (xs : List ℤ) : List ℤ :=
  (List.range k).foldl (fun acc s => fwhtStage s acc) xs

theorem pow_two_succ (s : ℕ) : 2 ^ (s + 1) = 2 ^ s + 2 ^ s := by
  rw [pow_succ]
  ring

/-! ## One stage -/

theorem fwhtStage_of_lt (s : ℕ) {xs : List ℤ} (h : xs.length < 2 ^ (s + 1)) :
    fwhtStage s xs = xs := by
  conv_lhs => rw [fwhtStage.eq_def]
  rw [dif_pos h]

theorem fwhtStage_of_ge (s : ℕ) {xs : List ℤ} (h : 2 ^ (s + 1) ≤ xs.length) :
    fwhtStage s xs =
      (xs.take (2 ^ s)).zipWith (· + ·) ((xs.drop (2 ^ s)).take (2 ^ s)) ++
        (xs.take (2 ^ s)).zipWith (· - ·) ((xs.drop (2 ^ s)).take (2 ^ s)) ++
          fwhtStage s (xs.drop (2 ^ (s + 1))) := by
  conv_lhs => rw [fwhtStage.eq_def]
  rw [dif_neg (by omega)]

@[simp]
theorem fwhtStage_nil (s : ℕ) : fwhtStage s [] = [] :=
  fwhtStage_of_lt s (by simp)

theorem fwhtStage_length_aux (s : ℕ) :
    ∀ (n : ℕ) (xs : List ℤ), xs.length = n → (fwhtStage s xs).length = xs.length := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro xs hlen
    by_cases h : xs.length < 2 ^ (s + 1)
    · rw [fwhtStage_of_lt s h]
    · have hle : 2 ^ (s + 1) ≤ xs.length := by omega
      have hp0 : 0 < 2 ^ (s + 1) := by positivity
      rw [fwhtStage_of_ge s hle]
      have h1 : (xs.take (2 ^ s)).length = 2 ^ s := by
        rw [List.length_take]
        exact Nat.min_eq_left (by omega)
      have h2 : ((xs.drop (2 ^ s)).take (2 ^ s)).length = 2 ^ s := by
        rw [List.length_take, List.length_drop]
        exact Nat.min_eq_left (by omega)
      have hz1 : ((xs.take (2 ^ s)).zipWith (· + ·) ((xs.drop (2 ^ s)).take (2 ^ s))).length =
          2 ^ s := by
        rw [List.length_zipWith, h1, h2, Nat.min_self]
      have hz2 : ((xs.take (2 ^ s)).zipWith (· - ·) ((xs.drop (2 ^ s)).take (2 ^ s))).length =
          2 ^ s := by
        rw [List.length_zipWith, h1, h2, Nat.min_self]
      have hrec : (fwhtStage s (xs.drop (2 ^ (s + 1)))).length = xs.length - 2 ^ (s + 1) := by
        rw [ih (xs.drop (2 ^ (s + 1))).length (by rw [List.length_drop]; omega)
          (xs.drop (2 ^ (s + 1))) rfl, List.length_drop]
      rw [List.length_append, List.length_append, hz1, hz2, hrec]
      have hp := pow_two_succ s
      omega

theorem fwhtStage_length (s : ℕ) (xs : List ℤ) : (fwhtStage s xs).length = xs.length :=
  fwhtStage_length_aux s xs.length xs rfl

/-! ## Concatenation and the block butterfly -/

theorem fwhtStage_append_aux (s : ℕ) :
    ∀ (n : ℕ) (xs ys : List ℤ), xs.length = n →
      2 ^ (s + 1) ∣ xs.length → 2 ^ (s + 1) ∣ ys.length →
      fwhtStage s (xs ++ ys) = fwhtStage s xs ++ fwhtStage s ys := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro xs ys hlen hx hy
    by_cases h0 : xs.length = 0
    · have hnil : xs = [] := List.length_eq_zero_iff.mp h0
      rw [hnil, List.nil_append, fwhtStage_nil, List.nil_append]
    · have hpos : 0 < xs.length := Nat.pos_of_ne_zero h0
      have hle : 2 ^ (s + 1) ≤ xs.length := Nat.le_of_dvd hpos hx
      have h2s : 2 ^ s ≤ xs.length := by have := pow_two_succ s; omega
      have hle' : 2 ^ s ≤ (xs.drop (2 ^ s)).length := by
        rw [List.length_drop]; omega
      have hb := pow_two_succ s
      conv_lhs => rw [fwhtStage_of_ge s (xs := xs ++ ys) (by rw [List.length_append]; omega)]
      conv_rhs => rw [fwhtStage_of_ge s (xs := xs) (by rw [hb]; omega)]
      rw [List.take_append_of_le_length h2s, List.drop_append_of_le_length h2s,
        List.take_append_of_le_length hle', List.drop_append_of_le_length hle]
      have hx' : 2 ^ (s + 1) ∣ (xs.drop (2 ^ (s + 1))).length := by
        rw [List.length_drop]
        exact Nat.dvd_sub hx (dvd_refl _)
      rw [ih (xs.drop (2 ^ (s + 1))).length (by rw [List.length_drop]; omega)
        (xs.drop (2 ^ (s + 1))) ys rfl hx' hy]
      simp only [List.append_assoc]

theorem fwhtStage_append (s : ℕ) (xs ys : List ℤ)
    (hx : 2 ^ (s + 1) ∣ xs.length) (hy : 2 ^ (s + 1) ∣ ys.length) :
    fwhtStage s (xs ++ ys) = fwhtStage s xs ++ fwhtStage s ys :=
  fwhtStage_append_aux s xs.length xs ys rfl hx hy

/-- The top stage `k` on two `2^k`-blocks is the half-combining butterfly. -/
theorem fwhtStage_block_butterfly (k : ℕ) (a b : List ℤ)
    (ha : a.length = 2 ^ k) (hb : b.length = 2 ^ k) :
    fwhtStage k (a ++ b) = a.zipWith (· + ·) b ++ a.zipWith (· - ·) b := by
  have hp := pow_two_succ k
  have htot : 2 ^ (k + 1) ≤ (a ++ b).length := by
    rw [List.length_append, ha, hb, ← hp]
  rw [fwhtStage_of_ge k htot]
  have ht : (a ++ b).take (2 ^ k) = a := by
    rw [List.take_append_of_le_length (by omega), ← ha, List.take_length]
  have hd : (a ++ b).drop (2 ^ k) = b := by
    rw [List.drop_append_of_le_length (by omega), ← ha, List.drop_length, List.nil_append]
  have htz : ((a ++ b).drop (2 ^ k)).take (2 ^ k) = b := by
    rw [hd, ← hb, List.take_length]
  have hd2 : (a ++ b).drop (2 ^ (k + 1)) = [] := by
    apply List.length_eq_zero_iff.mp
    rw [List.length_drop, List.length_append, ha, hb, ← hp, Nat.sub_self]
  simp only [ht, htz, hd2, fwhtStage_nil, List.append_nil]

/-! ## The stage fold -/

@[simp]
theorem fwhtRef_zero (xs : List ℤ) : fwhtRef 0 xs = xs := rfl

theorem fwhtRef_succ (k : ℕ) (xs : List ℤ) :
    fwhtRef (k + 1) xs = fwhtStage k (fwhtRef k xs) := by
  unfold fwhtRef
  rw [List.range_succ, List.foldl_append]
  rfl

@[simp]
theorem fwhtRef_length (k : ℕ) (xs : List ℤ) : (fwhtRef k xs).length = xs.length := by
  induction k with
  | zero => simp
  | succ k ih => rw [fwhtRef_succ, fwhtStage_length, ih]

theorem fwhtRef_append (k : ℕ) (xs ys : List ℤ)
    (hx : 2 ^ k ∣ xs.length) (hy : 2 ^ k ∣ ys.length) :
    fwhtRef k (xs ++ ys) = fwhtRef k xs ++ fwhtRef k ys := by
  induction k with
  | zero => simp
  | succ k ih =>
    have hx' : 2 ^ k ∣ xs.length := dvd_trans ⟨2, by rw [pow_succ]⟩ hx
    have hy' : 2 ^ k ∣ ys.length := dvd_trans ⟨2, by rw [pow_succ]⟩ hy
    rw [fwhtRef_succ, ih hx' hy',
      fwhtStage_append k (fwhtRef k xs) (fwhtRef k ys) (by rw [fwhtRef_length]; exact hx)
        (by rw [fwhtRef_length]; exact hy),
      ← fwhtRef_succ k xs, ← fwhtRef_succ k ys]

/-- **Segment independence.** On an aligned boundary `p` (a multiple of `2^k`) the transform splits
into the transform of the two sides. -/
theorem fwhtRef_split (k : ℕ) (xs : List ℤ) (p : ℕ) (hp : 2 ^ k ∣ p) (hpx : p ≤ xs.length)
    (hxs : 2 ^ k ∣ xs.length) :
    fwhtRef k xs = fwhtRef k (xs.take p) ++ fwhtRef k (xs.drop p) := by
  conv_lhs => rw [← List.take_append_drop p xs]
  rw [fwhtRef_append k (xs.take p) (xs.drop p)
    (by rw [List.length_take, Nat.min_eq_left hpx]; exact hp)
    (by rw [List.length_drop]; exact Nat.dvd_sub hxs hp)]

/-! ## The FWHT/Sylvester identity

The `k`-stage reference on a `2^k`-long segment is `(sylvesterH k).mulVec` indexed through `Fin (2^k)`.
The induction splits the segment into its two `2^k` halves: the first `k` stages act on each half
(`fwhtRef_append`), and stage `k` is the half-combining butterfly (`fwhtStage_block_butterfly`),
which is exactly `sylvester_mulVec_butterfly`. -/

/-- The top index of the doubled face. -/
def hTop (k : ℕ) (i : Fin (2 ^ k)) : Fin (2 ^ (k + 1)) :=
  ⟨i.val, by have := i.isLt; have := pow_two_succ k; omega⟩

/-- The bottom index of the doubled face. -/
def hBot (k : ℕ) (i : Fin (2 ^ k)) : Fin (2 ^ (k + 1)) :=
  ⟨2 ^ k + i.val, by have := i.isLt; have := pow_two_succ k; omega⟩

@[simp]
theorem hTop_val (k : ℕ) (i : Fin (2 ^ k)) : (hTop k i).val = i.val := rfl

@[simp]
theorem hBot_val (k : ℕ) (i : Fin (2 ^ k)) : (hBot k i).val = 2 ^ k + i.val := rfl

theorem sylvesterBlockEquiv_inl_val (k : ℕ) (i : Fin (2 ^ k)) :
    (sylvesterBlockEquiv k (Sum.inl i)).val = i.val := by
  simp [sylvesterBlockEquiv]

theorem sylvesterBlockEquiv_inr_val (k : ℕ) (i : Fin (2 ^ k)) :
    (sylvesterBlockEquiv k (Sum.inr i)).val = 2 ^ k + i.val := by
  simp [sylvesterBlockEquiv]
  omega

/-- The Sylvester butterfly, top half: at a top index, `H_{k+1}·x` is `H_k x₀ + H_k x₁`. -/
theorem sylvesterH_mulVec_top (k : ℕ) (x : Fin (2 ^ (k + 1)) → ℤ) (i : Fin (2 ^ k)) :
    (sylvesterH (k + 1)).mulVec x (hTop k i) =
      (sylvesterH k).mulVec (fun j : Fin (2 ^ k) => x (hTop k j)) i +
        (sylvesterH k).mulVec (fun j : Fin (2 ^ k) => x (hBot k j)) i := by
  have hb := congrFun (sylvester_mulVec_butterfly k x) (Sum.inl i)
  simp only [Sum.elim_inl, Pi.add_apply] at hb
  have he : sylvesterBlockEquiv k (Sum.inl i) = hTop k i :=
    Fin.ext (sylvesterBlockEquiv_inl_val k i)
  rw [he] at hb
  have hx0 : (x ∘ sylvesterBlockEquiv k ∘ Sum.inl) = fun j : Fin (2 ^ k) => x (hTop k j) := by
    funext j
    exact congrArg x (Fin.ext (sylvesterBlockEquiv_inl_val k j))
  have hx1 : (x ∘ sylvesterBlockEquiv k ∘ Sum.inr) = fun j : Fin (2 ^ k) => x (hBot k j) := by
    funext j
    exact congrArg x (Fin.ext (sylvesterBlockEquiv_inr_val k j))
  rw [hx0, hx1] at hb
  exact hb

/-- The Sylvester butterfly, bottom half: at a bottom index, `H_{k+1}·x` is `H_k x₀ − H_k x₁`. -/
theorem sylvesterH_mulVec_bottom (k : ℕ) (x : Fin (2 ^ (k + 1)) → ℤ) (i : Fin (2 ^ k)) :
    (sylvesterH (k + 1)).mulVec x (hBot k i) =
      (sylvesterH k).mulVec (fun j : Fin (2 ^ k) => x (hTop k j)) i -
        (sylvesterH k).mulVec (fun j : Fin (2 ^ k) => x (hBot k j)) i := by
  have hb := congrFun (sylvester_mulVec_butterfly k x) (Sum.inr i)
  simp only [Sum.elim_inr, Pi.sub_apply] at hb
  have he : sylvesterBlockEquiv k (Sum.inr i) = hBot k i :=
    Fin.ext (sylvesterBlockEquiv_inr_val k i)
  rw [he] at hb
  have hx0 : (x ∘ sylvesterBlockEquiv k ∘ Sum.inl) = fun j : Fin (2 ^ k) => x (hTop k j) := by
    funext j
    exact congrArg x (Fin.ext (sylvesterBlockEquiv_inl_val k j))
  have hx1 : (x ∘ sylvesterBlockEquiv k ∘ Sum.inr) = fun j : Fin (2 ^ k) => x (hBot k j) := by
    funext j
    exact congrArg x (Fin.ext (sylvesterBlockEquiv_inr_val k j))
  rw [hx0, hx1] at hb
  exact hb

/-- A list equality carries `List.getElem` across. -/
theorem getElem_of_eq {l₁ l₂ : List ℤ} (h : l₁ = l₂) (i : ℕ) (hi : i < l₁.length) :
    l₁[i]'hi = l₂[i]'(h ▸ hi) := by
  cases h
  rfl

theorem get_append_left_get (a b : List ℤ) (j : Fin a.length) :
    (a ++ b).get (Fin.cast (by rw [List.length_append]) (Fin.castAdd b.length j)) = a.get j := by
  rw [List.get_eq_getElem, List.get_eq_getElem]
  simp only [Fin.val_cast, Fin.val_castAdd]
  rw [List.getElem_append_left j.isLt]

theorem get_append_right_get (a b : List ℤ) (j : Fin b.length) :
    (a ++ b).get (Fin.cast (by rw [List.length_append]) (Fin.natAdd a.length j)) = b.get j := by
  rw [List.get_eq_getElem, List.get_eq_getElem]
  simp only [Fin.val_cast, Fin.val_natAdd]
  rw [List.getElem_append_right (by omega : a.length ≤ a.length + j.val)]
  simp only [Nat.add_sub_cancel_left]

/-- **The FWHT/Sylvester identity.** On a `2^k`-long segment the `k`-stage reference is
`(sylvesterH k).mulVec`, indexed through `Fin (2^k)`. -/
theorem fwhtRef_getElem (k : ℕ) : ∀ (seg : List ℤ) (hlen : seg.length = 2 ^ k) (i : ℕ)
    (hi : i < seg.length),
    (fwhtRef k seg)[i]'(by rw [fwhtRef_length]; exact hi) =
      (sylvesterH k).mulVec
        (fun j : Fin (2 ^ k) => seg.get (Fin.cast hlen.symm j))
        ⟨i, by rw [← hlen]; exact hi⟩ := by
  induction k with
  | zero =>
    intro seg hlen i hi
    have hi0 : i = 0 := by have := hlen; omega
    subst hi0
    rw [sylvester_zero, Matrix.one_mulVec]
    rfl
  | succ k ih =>
    intro seg hlen i hi
    obtain ⟨a, b, rfl, ha, hb⟩ :
        ∃ a b : List ℤ, seg = a ++ b ∧ a.length = 2 ^ k ∧ b.length = 2 ^ k :=
      ⟨seg.take (2 ^ k), seg.drop (2 ^ k), (List.take_append_drop (2 ^ k) seg).symm,
        by rw [List.length_take]; exact Nat.min_eq_left (by omega),
        by rw [List.length_drop]; omega⟩
    have hAlen : ((fwhtRef k a).zipWith (· + ·) (fwhtRef k b)).length = 2 ^ k := by
      rw [List.length_zipWith, fwhtRef_length, fwhtRef_length, ha, hb, Nat.min_self]
    have hstep : fwhtRef (k + 1) (a ++ b) =
        ((fwhtRef k a).zipWith (· + ·) (fwhtRef k b)) ++
          ((fwhtRef k a).zipWith (· - ·) (fwhtRef k b)) := by
      rw [fwhtRef_succ,
        fwhtRef_append k a b ⟨1, by rw [mul_one, ha]⟩ ⟨1, by rw [mul_one, hb]⟩,
        fwhtStage_block_butterfly k (fwhtRef k a) (fwhtRef k b) (by rw [fwhtRef_length, ha])
          (by rw [fwhtRef_length, hb])]
    have hfunA : (fun j : Fin (2 ^ k) =>
        (a ++ b).get (Fin.cast hlen.symm (hTop k j))) =
          fun j : Fin (2 ^ k) => a.get (Fin.cast ha.symm j) := by
      funext j
      have h1 : (Fin.cast hlen.symm (hTop k j) : Fin (a ++ b).length) =
          Fin.cast (by rw [List.length_append])
            (Fin.castAdd b.length (Fin.cast ha.symm j)) :=
        Fin.ext (by simp only [Fin.val_cast, Fin.val_castAdd, hTop_val])
      rw [h1]
      exact get_append_left_get a b (Fin.cast ha.symm j)
    have hfunB : (fun j : Fin (2 ^ k) =>
        (a ++ b).get (Fin.cast hlen.symm (hBot k j))) =
          fun j : Fin (2 ^ k) => b.get (Fin.cast hb.symm j) := by
      funext j
      have h1 : (Fin.cast hlen.symm (hBot k j) : Fin (a ++ b).length) =
          Fin.cast (by rw [List.length_append])
            (Fin.natAdd a.length (Fin.cast hb.symm j)) :=
        Fin.ext (by simp only [Fin.val_cast, Fin.val_natAdd, hBot_val]; omega)
      rw [h1]
      exact get_append_right_get a b (Fin.cast hb.symm j)
    rw [getElem_of_eq hstep]
    by_cases hisplit : i < 2 ^ k
    · rw [List.getElem_append_left (by rw [hAlen]; exact hisplit), List.getElem_zipWith,
        ih a ha i (by omega), ih b hb i (by omega),
        show (⟨i, by rw [← hlen]; exact hi⟩ : Fin (2 ^ (k + 1))) = hTop k ⟨i, hisplit⟩ from
          Fin.ext (by simp),
        sylvesterH_mulVec_top k (fun j : Fin (2 ^ (k + 1)) => (a ++ b).get (Fin.cast hlen.symm j))
          ⟨i, hisplit⟩,
        hfunA, hfunB]
    · have hge : 2 ^ k ≤ i := by omega
      have hidx : i - 2 ^ k < 2 ^ k := by omega
      rw [List.getElem_append_right (by rw [hAlen]; exact hge)]
      simp only [hAlen]
      rw [List.getElem_zipWith, ih a ha (i - 2 ^ k) (by rw [ha]; exact hidx),
        ih b hb (i - 2 ^ k) (by rw [hb]; exact hidx),
        show (⟨i, by rw [← hlen]; exact hi⟩ : Fin (2 ^ (k + 1))) = hBot k ⟨i - 2 ^ k, hidx⟩ from
          Fin.ext (by simp only [hBot_val]; omega),
        sylvesterH_mulVec_bottom k (fun j : Fin (2 ^ (k + 1)) => (a ++ b).get (Fin.cast hlen.symm j))
          ⟨i - 2 ^ k, hidx⟩,
        hfunA, hfunB]

/-- **Aligned segment.** On a length-`m·2^k` list the `i`-th aligned `2^k` segment of the result is
`fwhtRef k` of the `i`-th segment of the input. -/
theorem fwhtRef_segment (k m : ℕ) (xs : List ℤ) (hm : xs.length = m * 2 ^ k)
    (i : ℕ) (hi : i < m) :
    ((fwhtRef k xs).drop (i * 2 ^ k)).take (2 ^ k) =
      fwhtRef k ((xs.drop (i * 2 ^ k)).take (2 ^ k)) := by
  have hxs : 2 ^ k ∣ xs.length := ⟨m, by rw [hm, mul_comm]⟩
  have hp1le : (i + 1) * 2 ^ k ≤ xs.length := by
    rw [hm]; exact Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hi)
  have hlen_take : (xs.take ((i + 1) * 2 ^ k)).length = (i + 1) * 2 ^ k := by
    rw [List.length_take, Nat.min_eq_left hp1le]
  have hp0' : i * 2 ^ k ≤ (xs.take ((i + 1) * 2 ^ k)).length := by
    rw [hlen_take]; exact Nat.mul_le_mul_right _ (Nat.le_succ i)
  have hd0 : 2 ^ k ∣ i * 2 ^ k := ⟨i, by ring⟩
  have hd1 : 2 ^ k ∣ (i + 1) * 2 ^ k := ⟨i + 1, by ring⟩
  have hsplit1 : fwhtRef k xs =
      fwhtRef k (xs.take ((i + 1) * 2 ^ k)) ++ fwhtRef k (xs.drop ((i + 1) * 2 ^ k)) :=
    fwhtRef_split k xs ((i + 1) * 2 ^ k) hd1 hp1le hxs
  have hsplit2 : fwhtRef k (xs.take ((i + 1) * 2 ^ k)) =
      fwhtRef k ((xs.take ((i + 1) * 2 ^ k)).take (i * 2 ^ k)) ++
        fwhtRef k ((xs.take ((i + 1) * 2 ^ k)).drop (i * 2 ^ k)) :=
    fwhtRef_split k (xs.take ((i + 1) * 2 ^ k)) (i * 2 ^ k) hd0 hp0'
      (by rw [hlen_take]; exact hd1)
  have hAlen : (fwhtRef k ((xs.take ((i + 1) * 2 ^ k)).take (i * 2 ^ k))).length = i * 2 ^ k := by
    rw [fwhtRef_length, List.length_take, Nat.min_eq_left hp0']
  have hBlen : (fwhtRef k ((xs.take ((i + 1) * 2 ^ k)).drop (i * 2 ^ k))).length = 2 ^ k := by
    have h1 : (i + 1) * 2 ^ k = i * 2 ^ k + 2 ^ k := by ring
    rw [fwhtRef_length, List.length_drop, hlen_take, h1, Nat.add_sub_cancel_left]
  have hA0drop : (fwhtRef k ((xs.take ((i + 1) * 2 ^ k)).take (i * 2 ^ k))).drop (i * 2 ^ k) = [] :=
    List.drop_eq_nil_iff.mpr (le_of_eq hAlen)
  have hdrop : (fwhtRef k xs).drop (i * 2 ^ k) =
      fwhtRef k ((xs.take ((i + 1) * 2 ^ k)).drop (i * 2 ^ k)) ++
        fwhtRef k (xs.drop ((i + 1) * 2 ^ k)) := by
    rw [hsplit1, hsplit2,
      List.drop_append_of_le_length (by rw [List.length_append, hAlen]; omega),
      List.drop_append_of_le_length (le_of_eq hAlen.symm), hA0drop, List.nil_append]
  have hdt : (xs.take ((i + 1) * 2 ^ k)).drop (i * 2 ^ k) =
      (xs.drop (i * 2 ^ k)).take (2 ^ k) := by
    rw [List.drop_take]
    have hp : (i + 1) * 2 ^ k - i * 2 ^ k = 2 ^ k := by
      have h1 : (i + 1) * 2 ^ k = i * 2 ^ k + 2 ^ k := by ring
      rw [h1, Nat.add_sub_cancel_left]
    rw [hp]
  have hBtake : (fwhtRef k ((xs.take ((i + 1) * 2 ^ k)).drop (i * 2 ^ k))).take (2 ^ k) =
      fwhtRef k ((xs.take ((i + 1) * 2 ^ k)).drop (i * 2 ^ k)) :=
    List.take_of_length_le (le_of_eq hBlen)
  rw [hdrop, List.take_append_of_le_length (le_of_eq hBlen.symm), hBtake, hdt]

/-! ## The extracted `block_exponent` (item 1)

`block_exponent block = some k` exactly when `block = 2^k` with `1 ≤ k ≤ 16`. The loop scans
`k = 0, 1, …, 16` and stops at the first `k` with `2^k ≥ block`, so its result is the ceiling
exponent: `2^j < block` for all smaller `j`. -/

theorem usize_numBits_gt16 : 16 < UScalarTy.Usize.numBits := by
  rw [UScalarTy.Usize_numBits_eq]
  rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num

theorem usize_max_ge : 16 ≤ Usize.max := by
  have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
    rw [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq]
  rw [hmax]
  rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num

/-- `1#usize <<< k = 2^k` for every `k` below the word width. -/
theorem usize_one_shift (k : Usize) (hk : k.val < UScalarTy.Usize.numBits) :
    ∃ y : Usize, (1#usize <<< k) = ok y ∧ y.val = 2 ^ k.val := by
  obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists
    (UScalar.ShiftLeft_spec (ty0 := UScalarTy.Usize) (ty1 := UScalarTy.Usize)
      (1#usize) k (2 ^ UScalarTy.Usize.numBits) hk (by rw [UScalar.size]))
  refine ⟨y, hy, ?_⟩
  rw [hyv.1]
  have h1 : (1#usize).val = 1 := by simp
  rw [h1, Nat.shiftLeft_eq, one_mul,
    Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) hk)]

/-- The loop's stopping condition for `block`: `kf` is the least `j ≤ 16` with `block ≤ 2^j`, or
`16` when `block > 2^16`. -/
def BExpOk (block : Usize) (kf : Usize) : Prop :=
  kf.val ≤ 16 ∧ (∀ j, j < kf.val → 2 ^ j < block.val) ∧
    (kf.val = 16 ∨ block.val ≤ 2 ^ kf.val)

theorem block_exponent_loop_spec (block : Usize) :
    ∃ kf : Usize, hadamard.block_exponent_loop block 0#usize = ok kf ∧ BExpOk block kf := by
  have hspec : hadamard.block_exponent_loop block 0#usize ⦃ kf => BExpOk block kf ⦄ := by
    unfold hadamard.block_exponent_loop
    apply loop.spec_decr_nat
      (measure := fun k : Usize => 16 - k.val)
      (inv := fun k : Usize => k.val ≤ 16 ∧ ∀ j, j < k.val → 2 ^ j < block.val)
      (post := BExpOk block)
    · intro k hk
      rcases hk with ⟨hk16, hprev⟩
      by_cases hM : k < hadamard.MAX_K
      · have hklt : k.val < 16 := by
          have h := (UScalar.lt_equiv k hadamard.MAX_K).mp hM
          simpa [hadamard.MAX_K] using h
        obtain ⟨i, hi, hival⟩ := usize_one_shift k (by have := usize_numBits_gt16; omega)
        by_cases hib : i < block
        · have hibv : 2 ^ k.val < block.val := by
            have h := (UScalar.lt_equiv i block).mp hib
            rw [hival] at h
            exact h
          have h1u : (1#usize).val = 1 := by simp
          have hk1max : k.val + (1#usize).val ≤ Usize.max := by
            rw [h1u]
            have := usize_max_ge
            omega
          obtain ⟨k1, hk1, hk1val⟩ :=
            WP.spec_imp_exists (Usize.add_spec (x := k) (y := 1#usize) hk1max)
          have hk1val' : k1.val = k.val + 1 := by
            rw [h1u] at hk1val
            exact hk1val
          have hbody : hadamard.block_exponent_loop.body block k = ok (ControlFlow.cont k1) := by
            unfold hadamard.block_exponent_loop.body
            rw [if_pos hM, hi]
            simp only [bind_tc_ok]
            rw [if_pos hib, hk1]
            simp only [bind_tc_ok]
          rw [hbody]
          refine ⟨⟨by omega, ?_⟩, by omega⟩
          intro j hj
          by_cases hjk : j < k.val
          · exact hprev j hjk
          · have hje : j = k.val := by omega
            rw [hje]
            exact hibv
        · have hibv : block.val ≤ 2 ^ k.val := by
            have hle : block ≤ i := le_of_not_gt hib
            have h := (UScalar.le_equiv block i).mp hle
            rw [hival] at h
            exact h
          have hbody : hadamard.block_exponent_loop.body block k = ok (ControlFlow.done k) := by
            unfold hadamard.block_exponent_loop.body
            rw [if_pos hM, hi]
            simp only [bind_tc_ok]
            rw [if_neg hib]
          rw [hbody]
          exact ⟨hk16, hprev, Or.inr hibv⟩
      · have hk16' : k.val = 16 := by
          have hnot : ¬ k.val < 16 := by
            intro h
            exact hM ((UScalar.lt_equiv k hadamard.MAX_K).mpr (by simpa [hadamard.MAX_K] using h))
          omega
        have hbody : hadamard.block_exponent_loop.body block k = ok (ControlFlow.done k) := by
          unfold hadamard.block_exponent_loop.body
          rw [if_neg hM]
        rw [hbody]
        exact ⟨hk16, hprev, Or.inl hk16'⟩
    · refine ⟨by simp, ?_⟩
      intro j hj
      simp at hj
  obtain ⟨kf, hkf, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨kf, hkf, hpost⟩

/-- **Power of two.** For `block = 2^k` with `1 ≤ k ≤ 16`, `block_exponent block = some k`. -/
theorem block_exponent_spec_pow (block K : Usize) (hb : block.val = 2 ^ K.val)
    (hK1 : 1 ≤ K.val) (hK16 : K.val ≤ 16) :
    hadamard.block_exponent block = ok (some K) := by
  obtain ⟨kf, hloop, hkf16, hprev, hlast⟩ := block_exponent_loop_spec block
  have h2K : 2 ≤ block.val := by
    rw [hb]
    have h := Nat.pow_le_pow_right (by norm_num : 0 < 2) hK1
    simpa using h
  have hkfge : K.val ≤ kf.val := by
    by_contra hc
    have hlt : kf.val < K.val := by omega
    have hb2 : block.val ≤ 2 ^ kf.val := by
      rcases hlast with h | h
      · rw [h, hb]
        exact Nat.pow_le_pow_right (by norm_num : 0 < 2) hK16
      · exact h
    have hlt2 : 2 ^ kf.val < 2 ^ K.val := Nat.pow_lt_pow_right (by norm_num) hlt
    omega
  have hkfle : kf.val ≤ K.val := by
    by_contra hc
    have hlt : K.val < kf.val := by omega
    have h := hprev K.val hlt
    rw [hb] at h
    omega
  have hkfeq : kf.val = K.val := le_antisymm hkfle hkfge
  have hkeq : kf = K := UScalar.eq_of_val_eq hkfeq
  have hshift := usize_one_shift kf (by have := usize_numBits_gt16; omega)
  obtain ⟨i, hi, hival⟩ := hshift
  have hieq : i = block := by
    apply UScalar.eq_of_val_eq
    rw [hival, hb, ← hkfeq]
  have hnot2 : ¬ block < 2#usize := by
    intro hc
    have h := (UScalar.lt_equiv block (2#usize)).mp hc
    have h2 : (2#usize).val = 2 := by simp
    rw [h2] at h
    omega
  rw [hadamard.block_exponent, if_neg hnot2, hloop]
  simp only [bind_tc_ok]
  rw [hi]
  simp only [bind_tc_ok]
  rw [if_pos hieq]
  rw [hkeq]

/-- **Not a power of two.** For `2 ≤ block ≤ 2^16` that is not a power of two,
`block_exponent block = none`. -/
theorem block_exponent_spec_not_pow (block : Usize) (hge : 2 ≤ block.val)
    (hle : block.val ≤ 2 ^ 16) (hpow : ∀ j, block.val ≠ 2 ^ j) :
    hadamard.block_exponent block = ok none := by
  obtain ⟨kf, hloop, hkf16, hprev, hlast⟩ := block_exponent_loop_spec block
  have hble : block.val ≤ 2 ^ kf.val := by
    rcases hlast with h | h
    · rw [h]
      exact hle
    · exact h
  have hne : block.val ≠ 2 ^ kf.val := hpow kf.val
  have hgt : block.val < 2 ^ kf.val := lt_of_le_of_ne hble hne
  have hshift := usize_one_shift kf (by have := usize_numBits_gt16; omega)
  obtain ⟨i, hi, hival⟩ := hshift
  have hine : ¬ i = block := by
    intro hc
    have h := congrArg UScalar.val hc
    rw [hival] at h
    omega
  have hnot2 : ¬ block < 2#usize := by
    intro hc
    have h := (UScalar.lt_equiv block (2#usize)).mp hc
    have h2 : (2#usize).val = 2 := by simp
    rw [h2] at h
    omega
  rw [hadamard.block_exponent, if_neg hnot2, hloop]
  simp only [bind_tc_ok]
  rw [hi]
  simp only [bind_tc_ok]
  rw [if_neg hine]

/-! ## Integer bounds for the unnormalized butterflies

The window `|x_i| · 2^k ≤ 2^63 − 1` bounds every entry of `fwhtRef s x`. The scaled form
`|y| · 2^(k−s) ≤ 2^63 − 1` carries no division: one `fwhtStage` doubles the accumulated factor, so
the window at scale `k` descends to scale `k−s` after `s` stages. -/

/-- `2^63 − 1`, the `i64` ceiling as an integer. -/
def i64Ceil : ℤ := 2 ^ 63 - 1

theorem i64Ceil_eq_max : i64Ceil = I64.max := by
  rw [I64.max_eq]
  norm_num [i64Ceil]

theorem i64Ceil_nonneg : 0 ≤ i64Ceil := by norm_num [i64Ceil]

/-- `[-i64Ceil, i64Ceil]` lies inside the `i64` range. -/
theorem i64_range_of_abs_le {z : ℤ} (h : |z| ≤ i64Ceil) :
    I64.min ≤ z ∧ z ≤ I64.max := by
  rw [abs_le] at h
  rw [i64Ceil_eq_max, I64.max_eq] at h
  rw [I64.min_eq, I64.max_eq]
  omega

/-- A member of a `zipWith` is `f a b` for a left and a right member. -/
theorem zipWith_mem {α β γ : Type} {f : α → β → γ} {y : γ} {a : List α} {b : List β}
    (h : y ∈ a.zipWith f b) : ∃ x ∈ a, ∃ z ∈ b, f x z = y := by
  rw [List.mem_iff_getElem?] at h
  obtain ⟨i, hi⟩ := h
  rw [List.getElem?_zipWith_eq_some] at hi
  obtain ⟨x, z, hx, hz, hf⟩ := hi
  exact ⟨x, List.mem_iff_getElem?.mpr ⟨i, hx⟩, z, List.mem_iff_getElem?.mpr ⟨i, hz⟩, hf⟩

/-- **The stage bound.** If every entry of `l` satisfies `|x| · 2^(k−s) ≤ 2^63 − 1` and `s < k`,
every entry of `fwhtStage s l` satisfies `|y| · 2^(k−s−1) ≤ 2^63 − 1`. -/
theorem fwhtStage_abs_le (s k : ℕ) (hsk : s < k) :
    ∀ (n : ℕ) (l : List ℤ), l.length = n →
      (∀ x ∈ l, |x| * 2 ^ (k - s) ≤ i64Ceil) →
      ∀ y ∈ fwhtStage s l, |y| * 2 ^ (k - s - 1) ≤ i64Ceil := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro l hlen h
    have hpow : (2 : ℤ) ^ (k - s) = 2 * 2 ^ (k - s - 1) := by
      have h1 : (2 : ℤ) ^ (k - s - 1 + 1) = 2 ^ (k - s - 1) * 2 := pow_succ _ _
      have h2 : k - s - 1 + 1 = k - s := by omega
      rw [h2, mul_comm] at h1
      exact h1
    have hT : (0 : ℤ) ≤ 2 ^ (k - s - 1) := by positivity
    have hsum : ∀ a b : ℤ, |a| * 2 ^ (k - s) ≤ i64Ceil → |b| * 2 ^ (k - s) ≤ i64Ceil →
        |a + b| * 2 ^ (k - s - 1) ≤ i64Ceil := by
      intro a b ha hb
      have h1 : 2 * (|a| * 2 ^ (k - s - 1)) ≤ i64Ceil := by
        have he : 2 * (|a| * 2 ^ (k - s - 1)) = |a| * 2 ^ (k - s) := by rw [hpow]; ring
        rw [he]; exact ha
      have h2 : 2 * (|b| * 2 ^ (k - s - 1)) ≤ i64Ceil := by
        have he : 2 * (|b| * 2 ^ (k - s - 1)) = |b| * 2 ^ (k - s) := by rw [hpow]; ring
        rw [he]; exact hb
      have hab : |a + b| * 2 ^ (k - s - 1) ≤ (|a| + |b|) * 2 ^ (k - s - 1) :=
        mul_le_mul_of_nonneg_right (abs_add_le a b) hT
      have hAB : (|a| + |b|) * 2 ^ (k - s - 1) ≤ i64Ceil := by
        have he : (|a| + |b|) * 2 ^ (k - s - 1) =
            |a| * 2 ^ (k - s - 1) + |b| * 2 ^ (k - s - 1) := by ring
        rw [he]
        linarith
      exact le_trans hab hAB
    have hdiff : ∀ a b : ℤ, |a| * 2 ^ (k - s) ≤ i64Ceil → |b| * 2 ^ (k - s) ≤ i64Ceil →
        |a - b| * 2 ^ (k - s - 1) ≤ i64Ceil := by
      intro a b ha hb
      have habs : |a - b| ≤ |a| + |b| := by
        rw [sub_eq_add_neg, ← abs_neg b]
        exact abs_add_le a (-b)
      have hab : |a - b| * 2 ^ (k - s - 1) ≤ (|a| + |b|) * 2 ^ (k - s - 1) :=
        mul_le_mul_of_nonneg_right habs hT
      have h1 : 2 * (|a| * 2 ^ (k - s - 1)) ≤ i64Ceil := by
        have he : 2 * (|a| * 2 ^ (k - s - 1)) = |a| * 2 ^ (k - s) := by rw [hpow]; ring
        rw [he]; exact ha
      have h2 : 2 * (|b| * 2 ^ (k - s - 1)) ≤ i64Ceil := by
        have he : 2 * (|b| * 2 ^ (k - s - 1)) = |b| * 2 ^ (k - s) := by rw [hpow]; ring
        rw [he]; exact hb
      have hAB : (|a| + |b|) * 2 ^ (k - s - 1) ≤ i64Ceil := by
        have he : (|a| + |b|) * 2 ^ (k - s - 1) =
            |a| * 2 ^ (k - s - 1) + |b| * 2 ^ (k - s - 1) := by ring
        rw [he]
        linarith
      exact le_trans hab hAB
    by_cases hlt : l.length < 2 ^ (s + 1)
    · rw [fwhtStage_of_lt s hlt]
      intro y hy
      have hy' := h y hy
      have hle : (2 : ℤ) ^ (k - s - 1) ≤ (2 : ℤ) ^ (k - s) :=
        pow_le_pow_right₀ (by norm_num : (1:ℤ) ≤ 2) (by omega)
      exact le_trans (mul_le_mul_of_nonneg_left hle (abs_nonneg y)) hy'
    · rw [fwhtStage_of_ge s (by omega)]
      intro y hy
      rw [List.mem_append, List.mem_append] at hy
      rcases hy with (hyA | hyB) | hyC
      · obtain ⟨a, ha, b, hb, rfl⟩ := zipWith_mem hyA
        exact hsum a b (h a (List.mem_of_mem_take ha))
          (h b (List.mem_of_mem_drop (List.mem_of_mem_take hb)))
      · obtain ⟨a, ha, b, hb, rfl⟩ := zipWith_mem hyB
        exact hdiff a b (h a (List.mem_of_mem_take ha))
          (h b (List.mem_of_mem_drop (List.mem_of_mem_take hb)))
      · have hpos : 0 < 2 ^ (s + 1) := by positivity
        exact ih (l.drop (2 ^ (s + 1))).length (by rw [List.length_drop, hlen]; omega)
          (l.drop (2 ^ (s + 1))) rfl (fun x hx => h x (List.mem_of_mem_drop hx)) y hyC

/-- **The stage-fold bound.** After `s ≤ k` stages every entry satisfies the window scaled by
`2^(k−s)`. -/
theorem fwhtRef_abs_le (k : ℕ) : ∀ (s : ℕ), s ≤ k →
    ∀ (l : List ℤ), (∀ x ∈ l, |x| * 2 ^ k ≤ i64Ceil) →
      ∀ y ∈ fwhtRef s l, |y| * 2 ^ (k - s) ≤ i64Ceil := by
  intro s
  induction s with
  | zero =>
    intro hs l h y hy
    simpa using h y hy
  | succ s ih =>
    intro hs l h y hy
    have hs' : s ≤ k := by omega
    have hsk : s < k := by omega
    have h1 : ∀ x ∈ fwhtRef s l, |x| * 2 ^ (k - s) ≤ i64Ceil := ih hs' l h
    have h2 := fwhtStage_abs_le s k hsk (fwhtRef s l).length (fwhtRef s l) rfl h1
    rw [fwhtRef_succ] at hy
    have hk : k - (s + 1) = k - s - 1 := by omega
    rw [hk]
    exact h2 y hy

end

end apeiron_word
