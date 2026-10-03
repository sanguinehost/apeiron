/-
  `AlgebraShrExactOpen.lean` — AR-43: the open obligations of AR-23 (`AlgebraShrExact.lean`),
  closed in a new file so AR-23's source binding stays valid.

  Owned file: this one plus `AlgebraShrExactOpen.md`. No production, pin, API, corpus or AR-23
  edit; no commit. AR-23's file is read-only and is NOT imported: every fact re-used from it is
  re-proved here.

  Closures, in the order the task names them:
  OPEN 4  the width-32 exactness set `{n | 32 ∣ n}` as a set equality (`shr32_exact_amounts`),
          instantiated from the width-generic `shr_exact_amounts` re-proved here; its proof has
          one `ext` and no enumeration of the word space.
  OPEN 3  the amount-10 injection instance (`shr32_10_agreement_inj`) and the amount-3 instance
          (`shr32_3_agreement_inj`), instantiated from the width- and amount-generic
          `shr_agree_inj`; the only `decide`s are the numeric side conditions, no ground
          reduction over `ZMod (2^32)` and no enumeration of the 2^32 words.
  OPEN 2  the per-word agreement closed form for a general multiplier (`shr_agree_iff`) and its
          zero-multiplier vanishing corollary (`shr_agree_zero_iff`).
  OPEN 1  the two statements blocked on AR-23's `Nat.Coprime` route: `shr_agree_inj` (at most one
          word per residue class mod 2^m, with the injection corollary `shr_agree_at_most`) and
          `shr_agree_multiples` (only 0 agrees among the multiples of 2^m, with the candidate
          refutation `shr_agree_multiples_not_family`), proved from first principles by a `Nat`
          induction on the two-adic cancellation `dvd_pow_two_of_dvd_mul_odd`; no `Nat.Coprime`
          is constructed and no gcd lemma is used.

  Grades: `proved` for each named theorem whose `#print axioms` row at the end is inside
  {propext, Classical.choice, Quot.sound} with no `sorryAx`. Bounded-domain statements only for
  SHA-2's widths; no security, performance or mining claim.
-/

import Apeiron.Trichotomy.ChannelsAnyRing

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

namespace AlgebraShrExactOpen

open Apeiron.Trichotomy.AnyRing

/-! ## §1 The width-generic bodies, re-proved from the corpus definitions

`Apeiron.Crypto.Sha2.shrx` is `x >>> (n % 32)`; the same body at width `k`. `den` is the word
read into `ZMod (2 ^ k)`. -/

/-- The word read into `ZMod (2 ^ k)`, the crypto-width denotation at width `k`. -/
def den {k : Nat} (x : BitVec k) : ZMod (2 ^ k) := (x.toNat : ZMod (2 ^ k))

/-- `Apeiron.Crypto.Sha2.shrx` at width `k`: the corpus body with `32` replaced by `k`. -/
def tshrx {k : Nat} (x : BitVec k) (n : Nat) : BitVec k := x >>> (n % k)

/-- **The two-slot read is one multiplication by `a + r * b`, for every σ and every `r`.** -/
theorem ringUpper_mul_eval {R : Type*} [CommRing R] (σ r a b x : R) :
    ringUpper r (ringMul σ (x, 0) (a, b)) = x * (a + r * b) := by
  simp only [ringUpper, ringMul]
  ring

/-- The `ringUpper`/`ringMul` family applied to the word on the line `x ↦ (x, 0)`. -/
def twoSlotRead {k : Nat} (σ r a b : ZMod (2 ^ k)) (x : BitVec k) : ZMod (2 ^ k) :=
  ringUpper r (ringMul σ (den x, 0) (a, b))

/-- The family is the homogeneous family `x ↦ x·(a + r·b)`. -/
theorem twoSlotRead_eq {k : Nat} (σ r a b : ZMod (2 ^ k)) (x : BitVec k) :
    twoSlotRead σ r a b x = den x * (a + r * b) := by
  rw [twoSlotRead, ringUpper_mul_eval]

/-- **The closed form of the shift.** `den (x >>> n) = ⌊x / 2^(n mod k)⌋`. -/
theorem tshrx_toNat {k : Nat} (x : BitVec k) (n : Nat) :
    (tshrx x n).toNat = x.toNat / 2 ^ (n % k) := by
  simp [tshrx, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow]

/-- `2 ∣ 2^m` for `m > 0`. -/
theorem two_dvd_two_pow {m : Nat} (hm : 0 < m) : (2 : Nat) ∣ 2 ^ m := by
  obtain ⟨j, rfl⟩ : ∃ j, m = j + 1 := ⟨m - 1, by omega⟩
  exact ⟨2 ^ j, by rw [pow_succ, Nat.mul_comm]⟩

/-- Strict monotonicity of `2 ^ ·`: the only width arithmetic the witnesses need. -/
theorem two_pow_lt_two_pow {k m : Nat} (hm : m < k) : 2 ^ m < 2 ^ k :=
  (Nat.pow_lt_pow_iff_right (a := 2) (n := m) (m := k) (by decide)).2 hm

/-- The word `1`. -/
def oneWord (k : Nat) (hk : 0 < k) : BitVec k := ⟨1, Nat.one_lt_two_pow (Nat.ne_of_gt hk)⟩

theorem oneWord_toNat {k : Nat} (hk : 0 < k) : (oneWord k hk).toNat = 1 := rfl

/-- The all-ones word, whose width representative is `2^k - 1`. -/
def topWord (k : Nat) : BitVec k :=
  ⟨2 ^ k - 1, by have h1 : 0 < 2 ^ k := Nat.two_pow_pos k; omega⟩

theorem topWord_toNat {k : Nat} : (topWord k).toNat = 2 ^ k - 1 := rfl

/-- The word whose value is `2^m`, for `m < k`. -/
def powWord {k : Nat} (m : Nat) (hm : m < k) : BitVec k := ⟨2 ^ m, two_pow_lt_two_pow hm⟩

theorem powWord_toNat {k : Nat} {m : Nat} (hm : m < k) : (powWord m hm).toNat = 2 ^ m := rfl

theorem bitVec_zero_toNat {k : Nat} : (0 : BitVec k).toNat = 0 := by simp

/-- The bridge that turns a `.val` equation back into a ring equation. -/
theorem natCast_val_eq {k : Nat} (c : ZMod (2 ^ k)) : ((c.val : Nat) : ZMod (2 ^ k)) = c :=
  ZMod.val_injective (2 ^ k) (ZMod.val_natCast_of_lt (ZMod.val_lt c))

/-! ## §2 The exactness set of the family over the whole word space

The same content as AR-23's §5, re-proved here because OPEN 4 must be discharged from the
width-parametric statement and AR-23's file is not imported. -/

/-- **The shift is nonzero at the all-ones word for every non-identity amount.** -/
theorem shr_read_top_ne_zero {k n : Nat} (hk : 0 < k) (hm : 0 < n % k) :
    den (tshrx (topWord k) n) ≠ 0 := by
  have htop : (topWord k).toNat = 2 ^ k - 1 := topWord_toNat
  have hsub0 : 2 ^ k - 1 < 2 ^ k := by have h1 : 0 < 2 ^ k := Nat.two_pow_pos k; omega
  have hQlt : (2 ^ k - 1) / 2 ^ (n % k) < 2 ^ k := Nat.div_lt_of_lt hsub0
  rw [den, tshrx_toNat, htop]
  intro hz
  have hdvd : 2 ^ k ∣ (2 ^ k - 1) / 2 ^ (n % k) :=
    (ZMod.natCast_eq_zero_iff ((2 ^ k - 1) / 2 ^ (n % k)) (2 ^ k)).mp hz
  have hQ0 : (2 ^ k - 1) / 2 ^ (n % k) = 0 := Nat.eq_zero_of_dvd_of_lt hdvd hQlt
  have hmle : 2 ^ (n % k) ≤ 2 ^ k - 1 := by
    have h1 : 2 ^ (n % k) < 2 ^ k :=
      two_pow_lt_two_pow (k := k) (m := n % k) (Nat.mod_lt (x := n) (y := k) hk)
    omega
  have hQ1 : 1 ≤ (2 ^ k - 1) / 2 ^ (n % k) :=
    (Nat.le_div_iff_mul_le (k := 2 ^ (n % k)) (x := 1) (y := 2 ^ k - 1)
      (Nat.two_pow_pos (n % k))).2 (by simpa using hmle)
  omega

/-- **No member of the two-slot family is the shift**, at any width, any face, any amount with
`k ∤ n`. The witness pair is `1` and `2^k - 1`. -/
theorem shr_not_family {k n : Nat} (hk : 0 < k) (hne : n % k ≠ 0) :
    ¬ ∃ (σ r a b : ZMod (2 ^ k)), ∀ x : BitVec k,
      twoSlotRead σ r a b x = den (tshrx x n) := by
  rintro ⟨σ, r, a, b, hfam⟩
  have hm : 0 < n % k := Nat.pos_of_ne_zero hne
  have h1lt : 1 < 2 ^ k := Nat.one_lt_two_pow (Nat.ne_of_gt hk)
  have hone : (oneWord k hk).toNat = 1 := oneWord_toNat hk
  have h1zero : den (tshrx (oneWord k hk) n) = 0 := by
    rw [den, tshrx_toNat, hone]
    have hlt : 1 < 2 ^ (n % k) := Nat.one_lt_two_pow (Nat.ne_of_gt hm)
    rw [Nat.div_eq_of_lt hlt]
    exact Nat.cast_zero
  have h1 := hfam (oneWord k hk)
  rw [twoSlotRead_eq, h1zero] at h1
  have hvden1 : (den (oneWord k hk)).val = 1 := by
    rw [den, ZMod.val_natCast, hone]
    exact Nat.mod_eq_of_lt h1lt
  have hv := congrArg ZMod.val h1
  rw [ZMod.val_mul, ZMod.val_zero, hvden1, Nat.one_mul] at hv
  have hdvd : 2 ^ k ∣ (a + r * b).val := (Nat.dvd_iff_mod_eq_zero).mpr hv
  have hc0 : a + r * b = 0 := by
    rw [← natCast_val_eq (a + r * b),
      Nat.eq_zero_of_dvd_of_lt hdvd (ZMod.val_lt (a + r * b)), Nat.cast_zero]
  have h2 := hfam (topWord k)
  rw [twoSlotRead_eq, hc0, mul_zero] at h2
  exact absurd h2.symm (shr_read_top_ne_zero hk hm)

/-- **The exactness set of the shift over the whole word space is exactly `{n | k ∣ n}`.** The
`ext` is the one extensionality step of the width-32 instance below. -/
theorem shr_exact_amounts {k : Nat} (hk : 0 < k) :
    {n : Nat | ∃ (σ r a b : ZMod (2 ^ k)), ∀ x : BitVec k,
      twoSlotRead σ r a b x = den (tshrx x n)} = {n : Nat | k ∣ n} := by
  ext n
  constructor
  · rintro ⟨σ, r, a, b, hfam⟩
    by_cases hz : n % k = 0
    · exact Nat.dvd_of_mod_eq_zero hz
    · exact absurd ⟨σ, r, a, b, hfam⟩ (shr_not_family hk hz)
  · rintro ⟨j, hj⟩
    have h0 : n % k = 0 := Nat.mod_eq_zero_of_dvd ⟨j, hj⟩
    refine ⟨0, 0, 1, 0, fun x => ?_⟩
    calc twoSlotRead (0 : ZMod (2 ^ k)) 0 1 0 x = den x * (1 + 0 * 0) := twoSlotRead_eq 0 0 1 0 x
      _ = den x := by ring
      _ = den (tshrx x n) := by rw [tshrx, h0, BitVec.ushiftRight_zero]

/-- **OPEN 4: the width-32 amount-level statement.** The exactness set of the family at the
SHA-256 word width is exactly the multiples of `32`, as a named set equality. -/
theorem shr32_exact_amounts :
    {n : Nat | ∃ (σ r a b : ZMod (2 ^ 32)), ∀ x : BitVec 32,
      twoSlotRead σ r a b x = den (tshrx x n)} = {n : Nat | (32 : Nat) ∣ n} :=
  shr_exact_amounts (k := 32) (by decide)

/-! ## §3 The per-word agreement condition in closed form (OPEN 2)

With `x = 2^m·q + r` (`r` the low `m` bits, `q` the rest), agreement of `c·x` with `x >>> m` is
exactly `c.val·r + q·2^m·c.val = q + 2^k·t` for some `t`. -/

/-- **The agreement condition in divisibility form.** -/
theorem shr_agree_dvd {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m) (hmk : m < k)
    (c : ZMod (2 ^ k)) (x : BitVec k) (h : c * den x = den (tshrx x m)) :
    ∃ t : Nat, c.val * x.toNat = x.toNat / 2 ^ m + 2 ^ k * t := by
  have hv : (c * den x).val = (den (tshrx x m)).val := congrArg ZMod.val h
  rw [ZMod.val_mul, den, ZMod.val_natCast_of_lt x.isLt] at hv
  rw [den, tshrx_toNat, ZMod.val_natCast_of_lt (Nat.div_lt_of_lt x.isLt), Nat.mod_eq_of_lt hmk]
    at hv
  refine ⟨c.val * x.toNat / 2 ^ k, ?_⟩
  have := Nat.mod_add_div (c.val * x.toNat) (2 ^ k)
  omega

/-- **OPEN 2: the per-word agreement set in closed form, for a general multiplier.** With
`x = 2^m·q + r`, `c·x = x >>> m` holds exactly when `c.val·r + q·2^m·c.val = q + 2^k·t` for
some `t`. -/
theorem shr_agree_iff {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m) (hmk : m < k)
    (c : ZMod (2 ^ k)) (x : BitVec k) :
    c * den x = den (tshrx x m) ↔
      ∃ t : Nat, c.val * (x.toNat % 2 ^ m) + (x.toNat / 2 ^ m) * 2 ^ m * c.val
        = x.toNat / 2 ^ m + 2 ^ k * t := by
  have hdec : x.toNat = x.toNat % 2 ^ m + 2 ^ m * (x.toNat / 2 ^ m) :=
    (Nat.mod_add_div x.toNat (2 ^ m)).symm
  have heq : c.val * x.toNat
      = c.val * (x.toNat % 2 ^ m) + (x.toNat / 2 ^ m) * 2 ^ m * c.val := by
    conv_lhs => rw [hdec]
    ring
  constructor
  · intro h
    obtain ⟨t, ht⟩ := shr_agree_dvd hk hm hmk c x h
    exact ⟨t, heq ▸ ht⟩
  · rintro ⟨t, ht⟩
    refine ZMod.val_injective (2 ^ k) ?_
    rw [ZMod.val_mul, den, ZMod.val_natCast_of_lt x.isLt]
    rw [den, tshrx_toNat, ZMod.val_natCast_of_lt (Nat.div_lt_of_lt x.isLt), Nat.mod_eq_of_lt hmk]
    have hlt : x.toNat / 2 ^ m < 2 ^ k := Nat.div_lt_of_lt x.isLt
    rw [heq, ht, Nat.add_mod, Nat.mul_mod_right, Nat.add_zero, Nat.mod_mod,
      Nat.mod_eq_of_lt hlt]

/-- **OPEN 2, the vanishing case.** The zero multiplier agrees with `x >>> m` exactly on the
words below the cut `2^m`: the per-word vanishing statement of the shift. -/
theorem shr_agree_zero_iff {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m) (hmk : m < k)
    (x : BitVec k) :
    (0 : ZMod (2 ^ k)) * den x = den (tshrx x m) ↔ x.toNat < 2 ^ m := by
  rw [zero_mul, eq_comm, den, tshrx_toNat, Nat.mod_eq_of_lt hmk, ZMod.natCast_eq_zero_iff]
  constructor
  · intro hdvd
    have hQlt : x.toNat / 2 ^ m < 2 ^ k := Nat.div_lt_of_lt x.isLt
    have hQ0 : x.toNat / 2 ^ m = 0 := Nat.eq_zero_of_dvd_of_lt hdvd hQlt
    have h1 : x.toNat / 2 ^ m < 1 := by omega
    have h2 := (Nat.div_lt_iff_lt_mul (k := 2 ^ m) (x := x.toNat) (y := 1)
      (Nat.two_pow_pos m)).mp h1
    simpa using h2
  · intro hlt
    rw [Nat.div_eq_of_lt hlt]
    exact ⟨0, (Nat.mul_zero (2 ^ k)).symm⟩

/-! ## §4 The two-adic cancellation by `Nat` induction (OPEN 1 route)

`2^k ∣ q·w` with `w` odd forces `2^k ∣ q`, without `Nat.Coprime` and without any gcd lemma. -/

/-- `A` even and at least `2` forces `A - 1` odd. -/
theorem even_sub_one_odd {A : Nat} (hA : A % 2 = 0) (hA2 : 2 ≤ A) : (A - 1) % 2 = 1 := by
  obtain ⟨q, hq⟩ : ∃ q, A = 2 * q := ⟨A / 2, by have := Nat.mod_add_div A 2; omega⟩
  obtain ⟨p, rfl⟩ : ∃ p, q = p + 1 := ⟨q - 1, by omega⟩
  rw [hq]
  have heq : 2 * (p + 1) - 1 = 2 * p + 1 := by omega
  rw [heq, Nat.add_mod, Nat.mul_mod_right, Nat.zero_add, Nat.mod_mod,
    Nat.mod_eq_of_lt (by decide)]

/-- **The cancellation.** `2^k ∣ w·q` with `w` odd forces `2^k ∣ q`, by induction on `k` using
only `Nat` mod/mul arithmetic. No `Nat.Coprime` is constructed. -/
theorem dvd_pow_two_of_dvd_mul_odd {k w q : Nat} (hw : w % 2 = 1) (hd : 2 ^ k ∣ w * q) :
    2 ^ k ∣ q := by
  induction k generalizing q with
  | zero => exact Nat.one_dvd q
  | succ k ih =>
    have h2dvd : (2 : Nat) ∣ 2 ^ (k + 1) := two_dvd_two_pow (Nat.succ_pos k)
    have hd2 : (2 : Nat) ∣ w * q := Nat.dvd_trans h2dvd hd
    have hq2 : q % 2 = 0 := by
      have h0 : (w * q) % 2 = 0 := (Nat.dvd_iff_mod_eq_zero).mp hd2
      rw [Nat.mul_mod, hw, Nat.one_mul, Nat.mod_mod] at h0
      exact h0
    obtain ⟨q', rfl⟩ : ∃ q', q = 2 * q' := by
      obtain ⟨q', hq'⟩ := (Nat.dvd_iff_mod_eq_zero).mpr hq2
      exact ⟨q', hq'⟩
    have hd' : 2 ^ k ∣ w * q' := by
      have h1 : 2 ^ (k + 1) ∣ 2 * (w * q') := by
        rwa [show w * (2 * q') = 2 * (w * q') by ring] at hd
      rw [pow_succ, Nat.mul_comm (2 ^ k) 2] at h1
      exact (Nat.mul_dvd_mul_iff_left (by decide : 0 < 2)).mp h1
    obtain ⟨s, hs⟩ := ih hd'
    refine ⟨s, ?_⟩
    rw [hs, pow_succ]
    ring

/-! ## §5 The injection and the "low m bits zero" refutation (OPEN 1)

At most one word per residue class mod `2^m` agrees with a fixed expression, and among the
multiples of `2^m` the only word that agrees is `0`. -/

/-- The agreement condition read on two words with the same low `m` bits. -/
theorem shr_agree_inj_of_le {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m) (hmk : m < k)
    (c : ZMod (2 ^ k)) {x y : BitVec k}
    (hx : c * den x = den (tshrx x m)) (hy : c * den y = den (tshrx y m))
    (hr : x.toNat % 2 ^ m = y.toNat % 2 ^ m)
    (hle : y.toNat / 2 ^ m ≤ x.toNat / 2 ^ m) : x = y := by
  obtain ⟨tx, htx⟩ := shr_agree_dvd hk hm hmk c x hx
  obtain ⟨ty, hty⟩ := shr_agree_dvd hk hm hmk c y hy
  have hxd : x.toNat = x.toNat % 2 ^ m + 2 ^ m * (x.toNat / 2 ^ m) :=
    (Nat.mod_add_div x.toNat (2 ^ m)).symm
  have hyd : y.toNat = y.toNat % 2 ^ m + 2 ^ m * (y.toNat / 2 ^ m) :=
    (Nat.mod_add_div y.toNat (2 ^ m)).symm
  by_cases hc : c.val = 0
  · have hxq : x.toNat / 2 ^ m = 0 := by
      have h0 : 0 = x.toNat / 2 ^ m + 2 ^ k * tx := by
        rw [hc, Nat.zero_mul] at htx
        exact htx
      exact (Nat.add_eq_zero_iff.mp h0.symm).1
    have hyq : y.toNat / 2 ^ m = 0 := by
      have h0 : 0 = y.toNat / 2 ^ m + 2 ^ k * ty := by
        rw [hc, Nat.zero_mul] at hty
        exact hty
      exact (Nat.add_eq_zero_iff.mp h0.symm).1
    have hx' : x.toNat = x.toNat % 2 ^ m := by
      calc x.toNat = x.toNat % 2 ^ m + 2 ^ m * (x.toNat / 2 ^ m) := hxd
        _ = x.toNat % 2 ^ m := by rw [hxq, Nat.mul_zero, Nat.add_zero]
    have hy' : y.toNat = y.toNat % 2 ^ m := by
      calc y.toNat = y.toNat % 2 ^ m + 2 ^ m * (y.toNat / 2 ^ m) := hyd
        _ = y.toNat % 2 ^ m := by rw [hyq, Nat.mul_zero, Nat.add_zero]
    exact BitVec.toNat_inj.mp (by omega)
  · have hc1 : 1 ≤ c.val := by omega
    have hxy : y.toNat ≤ x.toNat := by
      have hmul : 2 ^ m * (y.toNat / 2 ^ m) ≤ 2 ^ m * (x.toNat / 2 ^ m) :=
        Nat.mul_le_mul_left _ hle
      omega
    have h1 : c.val * x.toNat = c.val * y.toNat + c.val * (x.toNat - y.toNat) := by
      conv_lhs => rw [← Nat.add_sub_of_le hxy]
      rw [Nat.mul_add]
    have hxyd : x.toNat - y.toNat = 2 ^ m * (x.toNat / 2 ^ m - y.toNat / 2 ^ m) := by
      have e1 : x.toNat - y.toNat
          = (x.toNat % 2 ^ m + 2 ^ m * (x.toNat / 2 ^ m))
            - (y.toNat % 2 ^ m + 2 ^ m * (y.toNat / 2 ^ m)) := by
        congr 1
      rw [e1]
      have e2 : (x.toNat % 2 ^ m + 2 ^ m * (x.toNat / 2 ^ m))
          - (y.toNat % 2 ^ m + 2 ^ m * (y.toNat / 2 ^ m))
          = 2 ^ m * (x.toNat / 2 ^ m) - 2 ^ m * (y.toNat / 2 ^ m) := by omega
      rw [e2, ← Nat.mul_sub]
    rw [hxyd] at h1
    have key : c.val * (2 ^ m * (x.toNat / 2 ^ m - y.toNat / 2 ^ m))
        + 2 ^ k * ty = (x.toNat / 2 ^ m - y.toNat / 2 ^ m) + 2 ^ k * tx := by
      have hq : x.toNat / 2 ^ m
          = y.toNat / 2 ^ m + (x.toNat / 2 ^ m - y.toNat / 2 ^ m) :=
        (Nat.add_sub_of_le hle).symm
      omega
    rcases le_total tx ty with htx_ty | hty_tx
    · have hd : x.toNat / 2 ^ m - y.toNat / 2 ^ m
          = c.val * (2 ^ m * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)) + 2 ^ k * (ty - tx) := by
        have hmul : 2 ^ k * ty = 2 ^ k * tx + 2 ^ k * (ty - tx) := by
          rw [← Nat.mul_add, Nat.add_sub_of_le htx_ty]
        omega
      have hd0 : x.toNat / 2 ^ m - y.toNat / 2 ^ m = 0 := by
        by_contra hne
        have hdpos : 0 < x.toNat / 2 ^ m - y.toNat / 2 ^ m := Nat.pos_of_ne_zero hne
        have hm2 : 2 ≤ 2 ^ m := by
          have h := Nat.one_lt_two_pow (Nat.ne_of_gt hm)
          omega
        have hge : 2 * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)
            ≤ c.val * (2 ^ m * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)) := by
          calc 2 * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)
              = 1 * (2 * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)) := by ring
            _ ≤ c.val * (2 ^ m * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)) :=
                Nat.mul_le_mul hc1 (Nat.mul_le_mul_right _ hm2)
        omega
      have hqeq : x.toNat / 2 ^ m = y.toNat / 2 ^ m := by omega
      have hx' : x.toNat = x.toNat % 2 ^ m + 2 ^ m * (y.toNat / 2 ^ m) := by
        calc x.toNat = x.toNat % 2 ^ m + 2 ^ m * (x.toNat / 2 ^ m) := hxd
          _ = x.toNat % 2 ^ m + 2 ^ m * (y.toNat / 2 ^ m) := by rw [hqeq]
      exact BitVec.toNat_inj.mp (by omega)
    · have hd : (c.val * 2 ^ m - 1) * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)
          = 2 ^ k * (tx - ty) := by
        have hmul : 2 ^ k * tx = 2 ^ k * ty + 2 ^ k * (tx - ty) := by
          rw [← Nat.mul_add, Nat.add_sub_of_le hty_tx]
        have hC : c.val * (2 ^ m * (x.toNat / 2 ^ m - y.toNat / 2 ^ m))
            = (x.toNat / 2 ^ m - y.toNat / 2 ^ m) + 2 ^ k * (tx - ty) := by omega
        calc (c.val * 2 ^ m - 1) * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)
            = (c.val * 2 ^ m) * (x.toNat / 2 ^ m - y.toNat / 2 ^ m)
              - (x.toNat / 2 ^ m - y.toNat / 2 ^ m) := by rw [Nat.sub_mul, one_mul]
          _ = c.val * (2 ^ m * (x.toNat / 2 ^ m - y.toNat / 2 ^ m))
              - (x.toNat / 2 ^ m - y.toNat / 2 ^ m) := by rw [Nat.mul_assoc]
          _ = ((x.toNat / 2 ^ m - y.toNat / 2 ^ m) + 2 ^ k * (tx - ty))
              - (x.toNat / 2 ^ m - y.toNat / 2 ^ m) := by rw [hC]
          _ = 2 ^ k * (tx - ty) := by omega
      have hdvd : 2 ^ k ∣ (c.val * 2 ^ m - 1) * (x.toNat / 2 ^ m - y.toNat / 2 ^ m) :=
        ⟨tx - ty, hd⟩
      have hodd : (c.val * 2 ^ m - 1) % 2 = 1 := by
        apply even_sub_one_odd
        · have h2m : 2 ^ m % 2 = 0 := Nat.mod_eq_zero_of_dvd (two_dvd_two_pow hm)
          rw [Nat.mul_mod, h2m, Nat.mul_zero, Nat.zero_mod]
        · have hm2 : 2 ≤ 2 ^ m := by
            have h := Nat.one_lt_two_pow (Nat.ne_of_gt hm)
            omega
          calc 2 ≤ 2 ^ m := hm2
            _ = 1 * 2 ^ m := (one_mul (2 ^ m)).symm
            _ ≤ c.val * 2 ^ m := Nat.mul_le_mul_right (2 ^ m) hc1
      have hkq : 2 ^ k ∣ x.toNat / 2 ^ m - y.toNat / 2 ^ m :=
        dvd_pow_two_of_dvd_mul_odd hodd hdvd
      have hlt : x.toNat / 2 ^ m - y.toNat / 2 ^ m < 2 ^ k := by
        have h1 : x.toNat / 2 ^ m - y.toNat / 2 ^ m ≤ x.toNat / 2 ^ m := Nat.sub_le _ _
        have h2 : x.toNat / 2 ^ m ≤ x.toNat := Nat.div_le_self _ _
        omega
      have hd0 : x.toNat / 2 ^ m - y.toNat / 2 ^ m = 0 := Nat.eq_zero_of_dvd_of_lt hkq hlt
      have hqeq : x.toNat / 2 ^ m = y.toNat / 2 ^ m := by omega
      have hx' : x.toNat = x.toNat % 2 ^ m + 2 ^ m * (y.toNat / 2 ^ m) := by
        calc x.toNat = x.toNat % 2 ^ m + 2 ^ m * (x.toNat / 2 ^ m) := hxd
          _ = x.toNat % 2 ^ m + 2 ^ m * (y.toNat / 2 ^ m) := by rw [hqeq]
      exact BitVec.toNat_inj.mp (by omega)

/-- **OPEN 1: at most one word per residue class mod `2^m`.** Two words agreeing with the same
expression and with the same low `m` bits are equal. This is the maximality half of AR-23's §(2),
re-proved without `Nat.Coprime`. -/
theorem shr_agree_inj {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m) (hmk : m < k)
    (c : ZMod (2 ^ k)) (x y : BitVec k)
    (hx : c * den x = den (tshrx x m)) (hy : c * den y = den (tshrx y m))
    (hr : x.toNat % 2 ^ m = y.toNat % 2 ^ m) : x = y := by
  rcases le_total (y.toNat / 2 ^ m) (x.toNat / 2 ^ m) with h | h
  · exact shr_agree_inj_of_le hk hm hmk c hx hy hr h
  · exact (shr_agree_inj_of_le hk hm hmk c hy hx hr.symm h).symm

/-- **OPEN 1: the agreement set is at most `2^m` words**, by the injection `x ↦ x mod 2^m`. -/
theorem shr_agree_at_most {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m) (hmk : m < k)
    (c : ZMod (2 ^ k)) :
    ∃ f : {x : BitVec k // c * den x = den (tshrx x m)} → Fin (2 ^ m),
      Function.Injective f := by
  refine ⟨fun x => ⟨x.1.toNat % 2 ^ m, Nat.mod_lt _ (Nat.two_pow_pos m)⟩, ?_⟩
  intro a b hab
  have hr : a.1.toNat % 2 ^ m = b.1.toNat % 2 ^ m := congrArg Fin.val hab
  exact Subtype.ext (shr_agree_inj hk hm hmk c a.1 b.1 a.2 b.2 hr)

/-- **OPEN 1: only `0` agrees among the multiples of `2^m`.** Agreement at a word whose low `m`
bits are zero forces the word to be `0`. -/
theorem shr_agree_multiples {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m) (hmk : m < k)
    (c : ZMod (2 ^ k)) {x : BitVec k}
    (hx : c * den x = den (tshrx x m)) (hr : x.toNat % 2 ^ m = 0) : x = 0 := by
  obtain ⟨t, ht⟩ := shr_agree_dvd hk hm hmk c x hx
  have hxd : x.toNat = 2 ^ m * (x.toNat / 2 ^ m) := by
    have := Nat.mod_add_div x.toNat (2 ^ m)
    omega
  by_cases hc : c.val = 0
  · have hxq : x.toNat / 2 ^ m = 0 := by
      have h0 : 0 = x.toNat / 2 ^ m + 2 ^ k * t := by
        rw [hc, Nat.zero_mul] at ht
        exact ht
      exact (Nat.add_eq_zero_iff.mp h0.symm).1
    have hx' : x.toNat = 0 := by
      calc x.toNat = 2 ^ m * (x.toNat / 2 ^ m) := hxd
        _ = 0 := by rw [hxq, Nat.mul_zero]
    exact BitVec.toNat_inj.mp (by rw [hx', bitVec_zero_toNat])
  · have hc1 : 1 ≤ c.val := by omega
    have hstep : c.val * x.toNat = c.val * (2 ^ m * (x.toNat / 2 ^ m)) := by
      conv_lhs => rw [hxd]
    have hC : c.val * (2 ^ m * (x.toNat / 2 ^ m)) = x.toNat / 2 ^ m + 2 ^ k * t := by
      rw [hstep] at ht
      exact ht
    have hA : (c.val * 2 ^ m - 1) * (x.toNat / 2 ^ m) = 2 ^ k * t := by
      calc (c.val * 2 ^ m - 1) * (x.toNat / 2 ^ m)
          = (c.val * 2 ^ m) * (x.toNat / 2 ^ m) - (x.toNat / 2 ^ m) := by
            rw [Nat.sub_mul, one_mul]
        _ = c.val * (2 ^ m * (x.toNat / 2 ^ m)) - (x.toNat / 2 ^ m) := by
            rw [Nat.mul_assoc]
        _ = ((x.toNat / 2 ^ m) + 2 ^ k * t) - (x.toNat / 2 ^ m) := by rw [hC]
        _ = 2 ^ k * t := by omega
    have hdvd : 2 ^ k ∣ (c.val * 2 ^ m - 1) * (x.toNat / 2 ^ m) := ⟨t, hA⟩
    have hodd : (c.val * 2 ^ m - 1) % 2 = 1 := by
      apply even_sub_one_odd
      · have h2m : 2 ^ m % 2 = 0 := Nat.mod_eq_zero_of_dvd (two_dvd_two_pow hm)
        rw [Nat.mul_mod, h2m, Nat.mul_zero, Nat.zero_mod]
      · have hm2 : 2 ≤ 2 ^ m := by
          have h := Nat.one_lt_two_pow (Nat.ne_of_gt hm)
          omega
        calc 2 ≤ 2 ^ m := hm2
          _ = 1 * 2 ^ m := (one_mul (2 ^ m)).symm
          _ ≤ c.val * 2 ^ m := Nat.mul_le_mul_right (2 ^ m) hc1
    have hkq : 2 ^ k ∣ x.toNat / 2 ^ m := dvd_pow_two_of_dvd_mul_odd hodd hdvd
    have hlt : x.toNat / 2 ^ m < 2 ^ k := by
      have h := Nat.div_le_self x.toNat (2 ^ m)
      omega
    have hq0 : x.toNat / 2 ^ m = 0 := Nat.eq_zero_of_dvd_of_lt hkq hlt
    have hx' : x.toNat = 0 := by
      calc x.toNat = 2 ^ m * (x.toNat / 2 ^ m) := hxd
        _ = 0 := by rw [hq0, Nat.mul_zero]
    exact BitVec.toNat_inj.mp (by rw [hx', bitVec_zero_toNat])

/-- **OPEN 1: the "low `m` bits zero" candidate is refuted.** No expression agrees with
`x >>> m` on the whole set of multiples of `2^m`: the witness `2^m` forces `2^m = 0`. -/
theorem shr_agree_multiples_not_family {k : Nat} (hk : 0 < k) {m : Nat} (hm : 0 < m)
    (hmk : m < k) :
    ¬ ∃ c : ZMod (2 ^ k), ∀ x : BitVec k,
      x.toNat % 2 ^ m = 0 → c * den x = den (tshrx x m) := by
  rintro ⟨c, hc⟩
  have hx : (powWord m hmk).toNat % 2 ^ m = 0 := by
    rw [powWord_toNat]
    exact Nat.mod_self (2 ^ m)
  have h0 := shr_agree_multiples hk hm hmk c (hc (powWord m hmk) hx) hx
  have hne : (powWord (k := k) m hmk) ≠ 0 := by
    intro h
    have h2 : (powWord (k := k) m hmk).toNat = (0 : BitVec k).toNat := congrArg BitVec.toNat h
    rw [powWord_toNat, bitVec_zero_toNat] at h2
    have := Nat.two_pow_pos m
    omega
  exact hne h0

/-! ## §6 The width-32 instances (OPEN 3)

The generic injection instantiated at the SHA-256 width for the two FIPS shift amounts. Numeric
side conditions are closed by `decide`; no `ZMod` or word-space ground reduction occurs. -/

/-- **OPEN 3: the amount-3 injection instance at width 32.** -/
theorem shr32_3_agreement_inj (c : ZMod (2 ^ 32)) (x y : BitVec 32)
    (hx : c * den x = den (tshrx x 3)) (hy : c * den y = den (tshrx y 3))
    (hr : x.toNat % 2 ^ 3 = y.toNat % 2 ^ 3) : x = y :=
  shr_agree_inj (k := 32) (by decide) (m := 3) (by decide) (by decide) c x y hx hy hr

/-- **OPEN 3: the amount-10 injection instance at width 32.** -/
theorem shr32_10_agreement_inj (c : ZMod (2 ^ 32)) (x y : BitVec 32)
    (hx : c * den x = den (tshrx x 10)) (hy : c * den y = den (tshrx y 10))
    (hr : x.toNat % 2 ^ 10 = y.toNat % 2 ^ 10) : x = y :=
  shr_agree_inj (k := 32) (by decide) (m := 10) (by decide) (by decide) c x y hx hy hr

/-- The joint hypotheses of the injection at the amount-10 instance are inhabited. -/
theorem shr_agree_inj_hyp_witness :
    (0 : ZMod (2 ^ 32)) * den (0 : BitVec 32) = den (tshrx (0 : BitVec 32) 10) ∧
      (0 : BitVec 32).toNat % 2 ^ 10 = (0 : BitVec 32).toNat % 2 ^ 10 := by
  refine ⟨?_, rfl⟩
  simp [den, tshrx_toNat, bitVec_zero_toNat]

/-! ## Axiom cone -/

#print axioms AlgebraShrExactOpen.ringUpper_mul_eval
#print axioms AlgebraShrExactOpen.tshrx_toNat
#print axioms AlgebraShrExactOpen.twoSlotRead_eq
#print axioms AlgebraShrExactOpen.two_dvd_two_pow
#print axioms AlgebraShrExactOpen.two_pow_lt_two_pow
#print axioms AlgebraShrExactOpen.bitVec_zero_toNat
#print axioms AlgebraShrExactOpen.natCast_val_eq
#print axioms AlgebraShrExactOpen.shr_read_top_ne_zero
#print axioms AlgebraShrExactOpen.shr_not_family
#print axioms AlgebraShrExactOpen.shr_exact_amounts
#print axioms AlgebraShrExactOpen.shr32_exact_amounts
#print axioms AlgebraShrExactOpen.shr_agree_dvd
#print axioms AlgebraShrExactOpen.even_sub_one_odd
#print axioms AlgebraShrExactOpen.dvd_pow_two_of_dvd_mul_odd
#print axioms AlgebraShrExactOpen.shr_agree_inj_of_le
#print axioms AlgebraShrExactOpen.shr_agree_inj
#print axioms AlgebraShrExactOpen.shr_agree_at_most
#print axioms AlgebraShrExactOpen.shr_agree_multiples
#print axioms AlgebraShrExactOpen.shr_agree_multiples_not_family
#print axioms AlgebraShrExactOpen.shr_agree_iff
#print axioms AlgebraShrExactOpen.shr_agree_zero_iff
#print axioms AlgebraShrExactOpen.shr32_3_agreement_inj
#print axioms AlgebraShrExactOpen.shr32_10_agreement_inj
#print axioms AlgebraShrExactOpen.shr_agree_inj_hyp_witness

end AlgebraShrExactOpen
