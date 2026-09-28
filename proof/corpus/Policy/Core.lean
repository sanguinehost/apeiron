/- Canonical definition of the bounded authorization decision core
(docs/epics/policy-engine.md, "Canonical definition"), plus the bit, guard and fold
toolkit the laws need. Lean core only, no external libraries; every declaration here
is constructive.

Bitwise operations are fuel-bounded (64 steps), not core `Nat.land`/`Nat.lor`: core
`Nat.land` depends on `propext` under Lean 4.31.0, which would put `Atom.eval`,
`guardEval`, `mask`, `decide`, `granted` and `Completes` outside the empty axiom cone.
The construction follows `Slang.Word.bitwise` (proof/corpus/Slang/Word.lean:37), with
the per-bit Boolean test written `decide (left % 2 = 1)` (`Nat.beq` on `Nat` is the
same test). Bit characterisations of `band`/`bor` may use the closed cone. -/
namespace Apeiron.Policy

/-! ## Terms -/

inductive Tri where
  | t | f | u
  deriving DecidableEq

inductive Op where
  | eq | le | ge | all | any
  deriving DecidableEq

structure Atom where
  op : Op
  slot : Nat
  value : Nat
  deriving DecidableEq

inductive Effect where
  | permit | forbid | ask
  deriving DecidableEq

structure Rule where
  id : Nat
  effect : Effect
  action : Nat
  rights : Nat
  guard : Option (List Atom)
  deriving DecidableEq

structure Req where
  facts : Nat → Option Nat
  action : Nat
  want : Nat

inductive Cause where
  | forbid (id : Nat) | noPermit
  deriving DecidableEq

inductive Verdict where
  | allow (g byId : Nat)
  | attenuate (g byId : Nat)
  | ask (g pending byId : Nat)
  | deny (c : Cause)
  deriving DecidableEq

/-! ## Width and the fuel-bounded 64-step bitwise operations -/

def W : Nat := 64

def mask (x : Nat) : Nat := x % 2 ^ W

/-- Fuel-bounded bitwise fold: `bitwise operation fuel left right` is the low `fuel`
bits of the pointwise `operation` of `left` and `right`. Temporal O(fuel) recursive
steps, each O(1) division/mod and one `operation` call; spatial O(fuel) stack (the
recursive result is used twice). The callers pass `fuel = W = 64`, so each is O(1)
with a 64-step constant. Follows `Slang.Word.bitwise`. -/
def bitwise (operation : Bool → Bool → Bool) : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, left, right =>
      let higher := bitwise operation fuel (left / 2) (right / 2)
      let low := operation (decide (left % 2 = 1)) (decide (right % 2 = 1))
      if low then higher + higher + 1 else higher + higher

/-- 64-step bitwise and (the lattice meet on rights). -/
def band (x y : Nat) : Nat := bitwise Bool.and W x y

/-- 64-step bitwise or (the lattice join on rights). -/
def bor (x y : Nat) : Nat := bitwise Bool.or W x y

/-- 64-step complement `(2^W - 1) - mask x`. -/
def ncomp (x : Nat) : Nat := (2 ^ W - 1) - mask x

theorem bitwise_lt (operation : Bool → Bool → Bool) :
    ∀ (fuel left right : Nat), bitwise operation fuel left right < 2 ^ fuel
  | 0, l, r => by simp [bitwise]
  | fuel + 1, l, r => by
      rw [bitwise]
      have ih : bitwise operation fuel (l / 2) (r / 2) < 2 ^ fuel :=
        bitwise_lt operation fuel (l / 2) (r / 2)
      split <;> simp only [Nat.pow_succ] <;> omega

theorem bitwise_div_two (operation : Bool → Bool → Bool) (fuel left right : Nat) :
    bitwise operation (fuel + 1) left right / 2 = bitwise operation fuel (left / 2) (right / 2) := by
  rw [bitwise]
  split <;> omega

theorem bitwise_mod_two (operation : Bool → Bool → Bool) (fuel left right : Nat) :
    bitwise operation (fuel + 1) left right % 2 =
      (if operation (decide (left % 2 = 1)) (decide (right % 2 = 1)) then 1 else 0) := by
  rw [bitwise]
  split <;> omega

theorem bitwise_testBit (operation : Bool → Bool → Bool) (i : Nat) :
    ∀ (fuel left right : Nat), i < fuel →
      (bitwise operation fuel left right).testBit i =
        operation (left.testBit i) (right.testBit i)
  | 0, l, r, h => by omega
  | fuel + 1, l, r, h => by
      by_cases h0 : i = 0
      · subst h0
        rw [Nat.testBit_zero, bitwise_mod_two, Nat.testBit_zero, Nat.testBit_zero]
        cases operation (decide (l % 2 = 1)) (decide (r % 2 = 1)) <;> rfl
      · obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
        rw [Nat.testBit_succ, bitwise_div_two, bitwise_testBit operation j fuel (l/2) (r/2) (by omega),
          ← Nat.testBit_succ, ← Nat.testBit_succ]

theorem testBit_bitwise (operation : Bool → Bool → Bool) (fuel left right i : Nat) :
    (bitwise operation fuel left right).testBit i =
      (decide (i < fuel) && operation (left.testBit i) (right.testBit i)) := by
  by_cases h : i < fuel
  · rw [bitwise_testBit operation i fuel left right h]; simp [h]
  · have hfi : fuel ≤ i := by omega
    have hpow : 2 ^ fuel ≤ 2 ^ i := Nat.pow_le_pow_right (by omega) hfi
    have hlt : bitwise operation fuel left right < 2 ^ i :=
      Nat.lt_of_lt_of_le (bitwise_lt operation fuel left right) hpow
    rw [Nat.testBit_lt_two_pow hlt]; simp [h]

theorem testBit_mask (x i : Nat) : (mask x).testBit i = (decide (i < W) && x.testBit i) := by
  unfold mask; rw [Nat.testBit_mod_two_pow]
theorem testBit_mask_lt {x i : Nat} (h : i < W) : (mask x).testBit i = x.testBit i := by
  rw [testBit_mask, decide_eq_true h, Bool.true_and]
theorem testBit_mask_ge {x i : Nat} (h : W ≤ i) : (mask x).testBit i = false := by
  rw [testBit_mask, decide_eq_false (Nat.not_lt.mpr h), Bool.false_and]

theorem testBit_band (x y i : Nat) :
    (band x y).testBit i = (decide (i < W) && (x.testBit i && y.testBit i)) :=
  testBit_bitwise Bool.and W x y i
theorem testBit_bor (x y i : Nat) :
    (bor x y).testBit i = (decide (i < W) && (x.testBit i || y.testBit i)) :=
  testBit_bitwise Bool.or W x y i
theorem testBit_band_lt {x y i : Nat} (h : i < W) :
    (band x y).testBit i = (x.testBit i && y.testBit i) := by
  rw [testBit_band, decide_eq_true h, Bool.true_and]
theorem testBit_band_ge {x y i : Nat} (h : W ≤ i) : (band x y).testBit i = false := by
  rw [testBit_band, decide_eq_false (Nat.not_lt.mpr h), Bool.false_and]
theorem testBit_bor_lt {x y i : Nat} (h : i < W) :
    (bor x y).testBit i = (x.testBit i || y.testBit i) := by
  rw [testBit_bor, decide_eq_true h, Bool.true_and]
theorem testBit_bor_ge {x y i : Nat} (h : W ≤ i) : (bor x y).testBit i = false := by
  rw [testBit_bor, decide_eq_false (Nat.not_lt.mpr h), Bool.false_and]

theorem band_lt (x y : Nat) : band x y < 2 ^ W := bitwise_lt Bool.and W x y
theorem bor_lt (x y : Nat) : bor x y < 2 ^ W := bitwise_lt Bool.or W x y

theorem mask_of_lt {x : Nat} (h : x < 2 ^ W) : mask x = x := Nat.mod_eq_of_lt h

theorem band_comm (x y : Nat) : band x y = band y x := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_band_lt h, testBit_band_lt h, Bool.and_comm]
  · rw [testBit_band_ge (Nat.le_of_not_lt h), testBit_band_ge (Nat.le_of_not_lt h)]

theorem bor_comm (x y : Nat) : bor x y = bor y x := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_bor_lt h, testBit_bor_lt h, Bool.or_comm]
  · rw [testBit_bor_ge (Nat.le_of_not_lt h), testBit_bor_ge (Nat.le_of_not_lt h)]

theorem band_assoc (x y z : Nat) : band (band x y) z = band x (band y z) := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_band_lt h, testBit_band_lt h, testBit_band_lt h, testBit_band_lt h,
      Bool.and_assoc]
  · rw [testBit_band_ge (Nat.le_of_not_lt h), testBit_band_ge (Nat.le_of_not_lt h)]

theorem bor_assoc (x y z : Nat) : bor (bor x y) z = bor x (bor y z) := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_bor_lt h, testBit_bor_lt h, testBit_bor_lt h, testBit_bor_lt h,
      Bool.or_assoc]
  · rw [testBit_bor_ge (Nat.le_of_not_lt h), testBit_bor_ge (Nat.le_of_not_lt h)]

theorem band_self (x : Nat) : band x x = mask x := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_band_lt h, testBit_mask_lt h, Bool.and_self]
  · rw [testBit_band_ge (Nat.le_of_not_lt h), testBit_mask_ge (Nat.le_of_not_lt h)]

theorem bor_self (x : Nat) : bor x x = mask x := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_bor_lt h, testBit_mask_lt h, Bool.or_self]
  · rw [testBit_bor_ge (Nat.le_of_not_lt h), testBit_mask_ge (Nat.le_of_not_lt h)]

theorem band_zero_left (x : Nat) : band 0 x = 0 := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_band_lt h, Nat.zero_testBit, Bool.false_and]
  · rw [testBit_band_ge (Nat.le_of_not_lt h), Nat.zero_testBit]

theorem band_zero_right (x : Nat) : band x 0 = 0 := by rw [band_comm, band_zero_left]

theorem bor_zero_left (x : Nat) : bor 0 x = mask x := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_bor_lt h, Nat.zero_testBit, Bool.false_or, testBit_mask_lt h]
  · rw [testBit_bor_ge (Nat.le_of_not_lt h), testBit_mask_ge (Nat.le_of_not_lt h)]

theorem bor_zero_right (x : Nat) : bor x 0 = mask x := by rw [bor_comm, bor_zero_left]

theorem band_bor_absorb (x y : Nat) : band x (bor x y) = mask x := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_band_lt h, testBit_bor_lt h, testBit_mask_lt h]
    cases x.testBit i <;> cases y.testBit i <;> rfl
  · rw [testBit_band_ge (Nat.le_of_not_lt h), testBit_mask_ge (Nat.le_of_not_lt h)]

theorem bor_band_absorb (x y : Nat) : bor x (band x y) = mask x := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_bor_lt h, testBit_band_lt h, testBit_mask_lt h]
    cases x.testBit i <;> cases y.testBit i <;> rfl
  · rw [testBit_bor_ge (Nat.le_of_not_lt h), testBit_mask_ge (Nat.le_of_not_lt h)]

theorem comp_mod_two (W m : Nat) (hW : 1 ≤ W) (hm : m < 2 ^ W) :
    (2 ^ W - 1 - m) % 2 = 1 - m % 2 := by
  have hle : m ≤ 2 ^ W - 1 := Nat.le_sub_one_of_lt hm
  have hsum : (2 ^ W - 1 - m) + m = 2 ^ W - 1 := Nat.sub_add_cancel hle
  have hmod : ((2 ^ W - 1 - m) % 2 + m % 2) % 2 = 1 := by
    have h := congrArg (fun t => t % 2) hsum
    rw [Nat.add_mod] at h
    rw [Nat.two_pow_sub_one_mod_two, Nat.mod_eq_of_lt (Nat.one_lt_two_pow (by omega : W ≠ 0))] at h
    exact h
  have hr : (2 ^ W - 1 - m) % 2 = 0 ∨ (2 ^ W - 1 - m) % 2 = 1 := Nat.mod_two_eq_zero_or_one _
  have hm1 : m % 2 = 0 ∨ m % 2 = 1 := Nat.mod_two_eq_zero_or_one m
  rcases hr with hr | hr <;> rcases hm1 with hm1 | hm1 <;> omega

theorem testBit_compl (W m i : Nat) (hm : m < 2 ^ W) (hi : i < W) :
    (2 ^ W - 1 - m).testBit i = !(m.testBit i) := by
  induction i generalizing W m with
  | zero =>
    have hW : 1 ≤ W := hi
    rw [Nat.testBit_zero, Nat.testBit_zero]
    have hmod := comp_mod_two W m hW hm
    rw [hmod]
    have hm1 : m % 2 = 0 ∨ m % 2 = 1 := Nat.mod_two_eq_zero_or_one m
    rcases hm1 with h | h <;> rw [h] <;> simp
  | succ j ih =>
    rw [Nat.testBit_succ, Nat.testBit_succ]
    cases W with
    | zero => omega
    | succ W' =>
      have hdiv : (2 ^ (W' + 1) - 1 - m) / 2 = 2 ^ W' - 1 - m / 2 := by
        rw [Nat.pow_succ]; omega
      rw [hdiv]
      have hm2 : m / 2 < 2 ^ W' := by
        rw [Nat.pow_succ] at hm
        have hm' : m < 2 * 2 ^ W' := by rw [Nat.mul_comm]; exact hm
        exact Nat.div_lt_of_lt_mul hm'
      have hj : j < W' := by omega
      exact ih W' (m / 2) hm2 hj

theorem ncomp_lt (x : Nat) : ncomp x < 2 ^ W := by
  unfold ncomp
  exact Nat.lt_of_le_of_lt (Nat.sub_le (2 ^ W - 1) (mask x))
    (Nat.sub_one_lt (Nat.ne_of_gt (Nat.two_pow_pos W)))

theorem testBit_ncomp (x i : Nat) :
    (ncomp x).testBit i = (decide (i < W) && !(x.testBit i)) := by
  unfold ncomp
  by_cases hi : i < W
  · rw [testBit_compl W (mask x) i (Nat.mod_lt x (Nat.two_pow_pos W)) hi, testBit_mask_lt hi,
      decide_eq_true hi, Bool.true_and]
  · have hlt : 2 ^ W - 1 - mask x < 2 ^ i := by
      have h := ncomp_lt x
      unfold ncomp at h
      exact Nat.lt_of_lt_of_le h (Nat.pow_le_pow_right (by omega) (Nat.le_of_not_lt hi))
    rw [Nat.testBit_lt_two_pow hlt, decide_eq_false hi, Bool.false_and]

theorem testBit_ncomp_lt {x i : Nat} (h : i < W) : (ncomp x).testBit i = !(x.testBit i) := by
  rw [testBit_ncomp, decide_eq_true h, Bool.true_and]
theorem testBit_ncomp_ge {x i : Nat} (h : W ≤ i) : (ncomp x).testBit i = false := by
  rw [testBit_ncomp, decide_eq_false (Nat.not_lt.mpr h), Bool.false_and]

theorem band_ncomp (x : Nat) : band x (ncomp x) = 0 := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases h : i < W
  · rw [testBit_band_lt h, testBit_ncomp_lt h, Nat.zero_testBit, Bool.and_not_self]
  · rw [testBit_band_ge (Nat.le_of_not_lt h), Nat.zero_testBit]

theorem testBit_band_left {x y i : Nat} (h : (band x y).testBit i = true) :
    x.testBit i = true := by
  rw [testBit_band] at h
  exact (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp h).2).1

theorem testBit_band_right {x y i : Nat} (h : (band x y).testBit i = true) :
    y.testBit i = true := by
  rw [testBit_band] at h
  exact (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp h).2).2

theorem exists_testBit_of_ne_zero {x : Nat} (h : x ≠ 0) : ∃ i, x.testBit i = true := by
  by_cases hc : ∃ i, x.testBit i = true
  · exact hc
  · exfalso
    have hall : ∀ i, x.testBit i = false := by
      intro i
      cases hi : x.testBit i with
      | false => rfl
      | true => exact absurd ⟨i, hi⟩ hc
    exact h (Nat.eq_of_testBit_eq (fun i => by rw [hall i, Nat.zero_testBit]))

theorem band_right_mask (x y : Nat) : band x (mask y) = band x y := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · rw [testBit_band_lt hi, testBit_band_lt hi, testBit_mask_lt hi]
  · rw [testBit_band_ge (Nat.le_of_not_lt hi), testBit_band_ge (Nat.le_of_not_lt hi)]

theorem one_testBit (i : Nat) : (1 : Nat).testBit i = decide (i = 0) := by
  rw [show (1 : Nat) = 2 ^ 0 from rfl, Nat.testBit_two_pow]
  by_cases h : i = 0 <;> simp [h, eq_comm]

theorem band_one (x : Nat) : band x 1 = x % 2 := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · rw [testBit_band_lt hi, one_testBit,
      show (x % 2).testBit i = (x % 2 ^ 1).testBit i from rfl, Nat.testBit_mod_two_pow]
    by_cases h0 : i = 0
    · rw [decide_eq_true h0, decide_eq_true (by omega : i < 1), Bool.and_true, Bool.true_and]
    · rw [decide_eq_false h0, decide_eq_false (by omega : ¬ i < 1), Bool.and_false,
        Bool.false_and]
  · have hW : 1 ≤ W := by unfold W; omega
    have h1lt : ¬ i < 1 := by omega
    rw [testBit_band_ge (Nat.le_of_not_lt hi),
      show (x % 2).testBit i = (x % 2 ^ 1).testBit i from rfl, Nat.testBit_mod_two_pow,
      decide_eq_false h1lt, Bool.false_and]

/-! ## Submask order `Sub x y` := every set bit of `x` is set in `y` -/

def Sub (x y : Nat) : Prop := band x y = mask x

theorem bool_and_sub2 {a b c : Bool} (h1 : a = true → b = true) (h2 : b = true → c = true) :
    (a && c) = a := by
  cases a with
  | false => simp
  | true =>
    have hb : b = true := h1 rfl
    have hc : c = true := h2 hb
    simp [hc]
theorem bool_or_sub {a b c : Bool} (h : a = true → b = true) :
    ((a || c) && (b || c)) = (a || c) := by
  cases a with
  | false => cases c <;> simp
  | true => rw [h rfl]; simp
theorem bool_and_and_sub {a b c : Bool} (h : b = true → c = true) :
    ((a && b) && (a && c)) = (a && b) := by
  cases a with
  | false => simp
  | true =>
    cases b with
    | false => simp
    | true => rw [h rfl]; simp
theorem bool_and_not_sub {a b : Bool} (h : a = true → b = true) : (a && !b) = false := by
  cases a with
  | false => simp
  | true => rw [h rfl]; simp
theorem bool_not_not_sub {a b : Bool} (h : a = true → b = true) : (!b && !a) = !b := by
  cases a with
  | false => simp
  | true => rw [h rfl]; simp
theorem bool_and_and_mul_sub {a b c d : Bool} (h1 : a = true → b = true)
    (h2 : c = true → d = true) : ((a && c) && (b && d)) = (a && c) := by
  cases a with
  | false => simp
  | true =>
    have hb : b = true := h1 rfl
    rw [hb]
    cases c with
    | false => simp
    | true => rw [h2 rfl]; simp

theorem sub_refl (x : Nat) : Sub x x := band_self x

theorem sub_testBit {x y : Nat} (h : Sub x y) {i : Nat} (hi : i < W)
    (hx : x.testBit i = true) : y.testBit i = true := by
  have hh := congrArg (fun t => t.testBit i) h
  simp only [testBit_band_lt hi, testBit_mask_lt hi, hx, Bool.true_and] at hh
  exact hh

theorem sub_trans {x y z : Nat} (h1 : Sub x y) (h2 : Sub y z) : Sub x z := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · simp only [testBit_band_lt hi, testBit_mask_lt hi]
    rw [bool_and_sub2 (fun ha => sub_testBit h1 hi ha) (fun hb => sub_testBit h2 hi hb)]
  · simp only [testBit_band_ge (Nat.le_of_not_lt hi), testBit_mask_ge (Nat.le_of_not_lt hi)]

theorem sub_or_right (x y : Nat) : Sub x (bor x y) := band_bor_absorb x y

theorem sub_or {x y z : Nat} (h : Sub x y) : Sub (bor x z) (bor y z) := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · simp only [testBit_mask_lt hi, testBit_band_lt hi, testBit_bor_lt hi]
    rw [bool_or_sub (fun ha => sub_testBit h hi ha)]
  · simp only [testBit_band_ge (Nat.le_of_not_lt hi), testBit_mask_ge (Nat.le_of_not_lt hi)]

theorem sub_and_right (x y : Nat) : Sub (band x y) x := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · simp only [testBit_mask_lt hi, testBit_band_lt hi]
    cases x.testBit i <;> cases y.testBit i <;> simp
  · simp only [testBit_band_ge (Nat.le_of_not_lt hi), testBit_mask_ge (Nat.le_of_not_lt hi)]

theorem sub_and_left {y z : Nat} (h : Sub y z) (x : Nat) : Sub (band x y) (band x z) := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · simp only [testBit_mask_lt hi, testBit_band_lt hi]
    rw [bool_and_and_sub (fun hb => sub_testBit h hi hb)]
  · simp only [testBit_band_ge (Nat.le_of_not_lt hi), testBit_mask_ge (Nat.le_of_not_lt hi)]

theorem sub_and_mul {a b c d : Nat} (h1 : Sub a b) (h2 : Sub c d) :
    Sub (band a c) (band b d) := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · simp only [testBit_mask_lt hi, testBit_band_lt hi]
    rw [bool_and_and_mul_sub (fun ha => sub_testBit h1 hi ha) (fun hc => sub_testBit h2 hi hc)]
  · simp only [testBit_band_ge (Nat.le_of_not_lt hi), testBit_mask_ge (Nat.le_of_not_lt hi)]

theorem sub_ncomp {y z : Nat} (h : Sub y z) : Sub (ncomp z) (ncomp y) := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · simp only [testBit_mask_lt hi, testBit_band_lt hi, testBit_ncomp_lt hi]
    rw [bool_not_not_sub (fun hy => sub_testBit h hi hy)]
  · simp only [testBit_band_ge (Nat.le_of_not_lt hi), testBit_mask_ge (Nat.le_of_not_lt hi)]

theorem sub_ncomp_zero {x y : Nat} (h : Sub x y) : band x (ncomp y) = 0 := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · simp only [testBit_band_lt hi, testBit_ncomp_lt hi, Nat.zero_testBit]
    rw [bool_and_not_sub (fun hx => sub_testBit h hi hx)]
  · simp only [testBit_band_ge (Nat.le_of_not_lt hi), Nat.zero_testBit]

theorem sub_zero_left (x : Nat) : Sub 0 x := by
  unfold Sub; rw [band_comm, band_zero_right]; rfl

theorem sub_of_band_ncomp_zero {x y : Nat} (h : band x (ncomp y) = 0) : Sub x y := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · rw [testBit_band_lt hi, testBit_mask_lt hi]
    have hh := congrArg (fun t => t.testBit i) h
    rw [testBit_band_lt hi, testBit_ncomp_lt hi, Nat.zero_testBit] at hh
    by_cases hx : x.testBit i
    · have hy : y.testBit i = true := by
        cases hb : y.testBit i
        · rw [hx, hb] at hh; simp at hh
        · rfl
      simp [hx, hy]
    · simp [hx]
  · rw [testBit_band_ge (Nat.le_of_not_lt hi), testBit_mask_ge (Nat.le_of_not_lt hi)]

theorem sub_apply {x y : Nat} (h : Sub x y) (hx : x < 2 ^ W) : band x y = x := by
  rw [h, mask_of_lt hx]

/-! ## Guard evaluation -/

def Atom.eval (f : Nat → Option Nat) (a : Atom) : Tri :=
  if a.slot < W then
    match f a.slot with
    | none => Tri.u
    | some fact =>
      match a.op with
      | Op.eq  => if fact = a.value then Tri.t else Tri.f
      | Op.le  => if fact ≤ a.value then Tri.t else Tri.f
      | Op.ge  => if a.value ≤ fact then Tri.t else Tri.f
      | Op.all => if band fact a.value = a.value then Tri.t else Tri.f
      | Op.any => if band fact a.value = 0 then Tri.f else Tri.t
  else Tri.u

def guardGo (f : Nat → Option Nat) : List Atom → Tri → Tri
  | [], acc => acc
  | a :: rest, acc =>
    match Atom.eval f a with
    | Tri.f => Tri.f
    | Tri.u => guardGo f rest Tri.u
    | Tri.t => guardGo f rest acc

def guardEval (f : Nat → Option Nat) : Option (List Atom) → Tri
  | none => Tri.u
  | some as => guardGo f as Tri.t

/-! ## Guard characterisation and monotonicity over `Completes` -/

theorem guardGo_t_iff (f : Nat → Option Nat) (as : List Atom) (acc : Tri) :
    guardGo f as acc = Tri.t ↔ acc = Tri.t ∧ ∀ a ∈ as, Atom.eval f a = Tri.t := by
  induction as generalizing acc with
  | nil => simp [guardGo]
  | cons a rest ih =>
    cases h : Atom.eval f a with
    | t =>
      simp only [guardGo, h]
      rw [ih]
      constructor
      · rintro ⟨hacc, hall⟩
        exact ⟨hacc, fun b hb => by
          rcases List.mem_cons.mp hb with rfl | hb
          · exact h
          · exact hall b hb⟩
      · rintro ⟨hacc, hall⟩
        exact ⟨hacc, fun b hb => hall b (List.mem_cons.mpr (Or.inr hb))⟩
    | f =>
      simp only [guardGo, h]
      constructor
      · intro hh; exact absurd hh (by decide)
      · intro hh; exact absurd (hh.2 a (by simp)) (by rw [h]; decide)
    | u =>
      simp only [guardGo, h]
      rw [ih]
      constructor
      · intro hh; exact absurd hh.1 (by decide)
      · intro hh; exact absurd (hh.2 a (by simp)) (by rw [h]; decide)

theorem guardGo_f_of_mem (f : Nat → Option Nat) (as : List Atom) (acc : Tri)
    (h : ∃ a ∈ as, Atom.eval f a = Tri.f) : guardGo f as acc = Tri.f := by
  induction as generalizing acc with
  | nil => rcases h with ⟨a, ha, _⟩; exact absurd ha (List.not_mem_nil)
  | cons a rest ih =>
    rcases h with ⟨b, hb, he⟩
    rcases List.mem_cons.mp hb with rfl | hb'
    · simp only [guardGo, he]
    · cases hh : Atom.eval f a with
      | t => simp only [guardGo, hh]; exact ih acc ⟨b, hb', he⟩
      | f => simp only [guardGo, hh]
      | u => simp only [guardGo, hh]; exact ih Tri.u ⟨b, hb', he⟩

theorem guardGo_f_iff (f : Nat → Option Nat) (as : List Atom) (acc : Tri) :
    guardGo f as acc = Tri.f ↔
      (∃ a ∈ as, Atom.eval f a = Tri.f) ∨ (acc = Tri.f ∧ ∀ a ∈ as, Atom.eval f a = Tri.t) := by
  induction as generalizing acc with
  | nil => simp [guardGo]
  | cons a rest ih =>
    cases h : Atom.eval f a with
    | f =>
      constructor
      · intro _; exact Or.inl ⟨a, by simp, h⟩
      · intro _; simp only [guardGo, h]
    | t =>
      simp only [guardGo, h]
      rw [ih]
      constructor
      · rintro (hx | ⟨hacc, hall⟩)
        · exact Or.inl ⟨hx.choose, List.mem_cons.mpr (Or.inr hx.choose_spec.1), hx.choose_spec.2⟩
        · exact Or.inr ⟨hacc, fun b hb => by
            rcases List.mem_cons.mp hb with rfl | hb
            · exact h
            · exact hall b hb⟩
      · rintro (⟨b, hb, he⟩ | ⟨hacc, hall⟩)
        · rcases List.mem_cons.mp hb with rfl | hb
          · rw [h] at he; exact absurd he (by decide)
          · exact Or.inl ⟨b, hb, he⟩
        · exact Or.inr ⟨hacc, fun b hb => hall b (List.mem_cons.mpr (Or.inr hb))⟩
    | u =>
      simp only [guardGo, h]
      rw [ih]
      constructor
      · rintro (⟨b, hb, he⟩ | ⟨hacc, _⟩)
        · exact Or.inl ⟨b, List.mem_cons.mpr (Or.inr hb), he⟩
        · exact absurd hacc (by decide)
      · rintro (⟨b, hb, he⟩ | ⟨_, hall⟩)
        · rcases List.mem_cons.mp hb with rfl | hb
          · rw [h] at he; exact absurd he (by decide)
          · exact Or.inl ⟨b, hb, he⟩
        · exact absurd (hall a (by simp)) (by rw [h]; decide)

def Completes (q q' : Req) : Prop :=
  q'.action = q.action ∧ q'.want = q.want ∧
    ∀ s v, q.facts s = some v → q'.facts s = some v

theorem Atom.eval_t_completes {q q' : Req} (hc : Completes q q') (a : Atom)
    (h : Atom.eval q.facts a = Tri.t) : Atom.eval q'.facts a = Tri.t := by
  unfold Atom.eval at h ⊢
  by_cases hsl : a.slot < W
  · rw [if_pos hsl] at h ⊢
    cases hf : q.facts a.slot with
    | none => rw [hf] at h; simp at h
    | some v =>
      rw [hf] at h
      rw [hc.2.2 a.slot v hf]
      exact h
  · rw [if_neg hsl] at h; exact absurd h (by decide)

theorem Atom.eval_f_completes {q q' : Req} (hc : Completes q q') (a : Atom)
    (h : Atom.eval q.facts a = Tri.f) : Atom.eval q'.facts a = Tri.f := by
  unfold Atom.eval at h ⊢
  by_cases hsl : a.slot < W
  · rw [if_pos hsl] at h ⊢
    cases hf : q.facts a.slot with
    | none => rw [hf] at h; simp at h
    | some v =>
      rw [hf] at h
      rw [hc.2.2 a.slot v hf]
      exact h
  · rw [if_neg hsl] at h; exact absurd h (by decide)

theorem guardEval_t_completes {q q' : Req} (hc : Completes q q') (g : Option (List Atom))
    (h : guardEval q.facts g = Tri.t) : guardEval q'.facts g = Tri.t := by
  cases g with
  | none => exact absurd h (by simp [guardEval])
  | some as =>
    simp only [guardEval] at h ⊢
    have hh : ∀ a ∈ as, Atom.eval q.facts a = Tri.t := ((guardGo_t_iff q.facts as Tri.t).mp h).2
    exact (guardGo_t_iff q'.facts as Tri.t).mpr
      ⟨rfl, fun a ha => Atom.eval_t_completes hc a (hh a ha)⟩

theorem guardEval_f_completes {q q' : Req} (hc : Completes q q') (g : Option (List Atom))
    (h : guardEval q.facts g = Tri.f) : guardEval q'.facts g = Tri.f := by
  cases g with
  | none => exact absurd h (by simp [guardEval])
  | some as =>
    simp only [guardEval] at h ⊢
    have hx : ∃ a ∈ as, Atom.eval q.facts a = Tri.f := by
      rcases (guardGo_f_iff q.facts as Tri.t).mp h with hx | ⟨hacc, _⟩
      · exact hx
      · exact absurd hacc (by decide)
    rcases hx with ⟨a, ha, he⟩
    exact guardGo_f_of_mem q'.facts as Tri.t ⟨a, ha, Atom.eval_f_completes hc a he⟩

/-! ## Rules, the three accumulators and the decision -/

abbrev Permits (q : Req) (r : Rule) : Prop :=
  r.action = q.action ∧ r.effect = Effect.permit ∧ guardEval q.facts r.guard = Tri.t ∧
    band r.rights q.want ≠ 0

abbrev Forbids (q : Req) (r : Rule) : Prop :=
  r.action = q.action ∧ r.effect = Effect.forbid ∧ guardEval q.facts r.guard ≠ Tri.f

abbrev Asks (q : Req) (r : Rule) : Prop :=
  r.action = q.action ∧ r.effect = Effect.ask ∧ guardEval q.facts r.guard ≠ Tri.f

def permStep (q : Req) (acc : Nat) (r : Rule) : Nat :=
  if Permits q r then bor acc r.rights else acc

def forbStep (q : Req) (acc : Nat) (r : Rule) : Nat :=
  if Forbids q r then bor acc r.rights else acc

def askStep (q : Req) (acc : Nat) (r : Rule) : Nat :=
  if Asks q r then bor acc r.rights else acc

def permRights (rs : List Rule) (q : Req) : Nat := rs.foldl (permStep q) 0
def forbRights (rs : List Rule) (q : Req) : Nat := rs.foldl (forbStep q) 0
def askRights (rs : List Rule) (q : Req) : Nat := rs.foldl (askStep q) 0

theorem permStep_sub (q : Req) (acc : Nat) (r : Rule) : Sub acc (permStep q acc r) := by
  unfold permStep
  by_cases h : Permits q r
  · rw [if_pos h]; exact sub_or_right acc r.rights
  · rw [if_neg h]; exact sub_refl acc

theorem forbStep_sub (q : Req) (acc : Nat) (r : Rule) : Sub acc (forbStep q acc r) := by
  unfold forbStep
  by_cases h : Forbids q r
  · rw [if_pos h]; exact sub_or_right acc r.rights
  · rw [if_neg h]; exact sub_refl acc

theorem askStep_sub (q : Req) (acc : Nat) (r : Rule) : Sub acc (askStep q acc r) := by
  unfold askStep
  by_cases h : Asks q r
  · rw [if_pos h]; exact sub_or_right acc r.rights
  · rw [if_neg h]; exact sub_refl acc

theorem foldl_sub_of_step (g : Nat → Rule → Nat) (hg : ∀ acc r, Sub acc (g acc r)) :
    ∀ (l : List Rule) (acc : Nat), Sub acc (l.foldl g acc) := by
  intro l
  induction l with
  | nil => intro acc; exact sub_refl acc
  | cons r rest ih =>
    intro acc
    rw [List.foldl_cons]
    exact sub_trans (hg acc r) (ih (g acc r))

theorem permRights_sub_of_mem (rs : List Rule) (q : Req) (r : Rule) (hmem : r ∈ rs)
    (hp : Permits q r) : Sub r.rights (permRights rs q) := by
  suffices aux : ∀ (l : List Rule), r ∈ l → Permits q r →
      ∀ acc, Sub r.rights (l.foldl (permStep q) acc) from aux rs hmem hp 0
  intro l
  induction l with
  | nil => intro hmem acc; exact absurd hmem (List.not_mem_nil)
  | cons r' rest ih =>
    intro hmem hp acc
    rw [List.foldl_cons]
    rcases List.mem_cons.mp hmem with heq | hmem'
    · subst heq
      have hstep : permStep q acc r = bor acc r.rights := by rw [permStep, if_pos hp]
      rw [hstep]
      exact sub_trans (by rw [bor_comm]; exact sub_or_right r.rights acc)
        (foldl_sub_of_step (permStep q) (permStep_sub q) rest (bor acc r.rights))
    · exact ih hmem' hp (permStep q acc r')

theorem forbRights_sub_of_mem (rs : List Rule) (q : Req) (r : Rule) (hmem : r ∈ rs)
    (hf : Forbids q r) : Sub r.rights (forbRights rs q) := by
  suffices aux : ∀ (l : List Rule), r ∈ l → Forbids q r →
      ∀ acc, Sub r.rights (l.foldl (forbStep q) acc) from aux rs hmem hf 0
  intro l
  induction l with
  | nil => intro hmem acc; exact absurd hmem (List.not_mem_nil)
  | cons r' rest ih =>
    intro hmem hf acc
    rw [List.foldl_cons]
    rcases List.mem_cons.mp hmem with heq | hmem'
    · subst heq
      have hstep : forbStep q acc r = bor acc r.rights := by rw [forbStep, if_pos hf]
      rw [hstep]
      exact sub_trans (by rw [bor_comm]; exact sub_or_right r.rights acc)
        (foldl_sub_of_step (forbStep q) (forbStep_sub q) rest (bor acc r.rights))
    · exact ih hmem' hf (forbStep q acc r')

theorem Permits_completes {q q' : Req} (hc : Completes q q') (r : Rule)
    (h : Permits q r) : Permits q' r := by
  rcases h with ⟨ha, he, hg, hw⟩
  refine ⟨?_, he, ?_, ?_⟩
  · rw [hc.1]; exact ha
  · exact guardEval_t_completes hc r.guard hg
  · rw [hc.2.1]; exact hw

theorem Forbids_completes {q q' : Req} (hc : Completes q q') (r : Rule)
    (h : Forbids q' r) : Forbids q r := by
  rcases h with ⟨ha, he, hg⟩
  refine ⟨?_, he, ?_⟩
  · rw [← hc.1]; exact ha
  · intro hf; exact hg (guardEval_f_completes hc r.guard hf)

theorem permStep_sub_completes (q q' : Req) (hc : Completes q q') (r : Rule) :
    ∀ acc acc', Sub acc acc' → Sub (permStep q acc r) (permStep q' acc' r) := by
  intro acc acc' h
  unfold permStep
  by_cases hp : Permits q r
  · rw [if_pos hp, if_pos (Permits_completes hc r hp)]
    exact sub_or h
  · rw [if_neg hp]
    by_cases hp' : Permits q' r
    · rw [if_pos hp']
      exact sub_trans h (sub_or_right acc' r.rights)
    · rw [if_neg hp']
      exact h

theorem forbStep_sub_completes (q q' : Req) (hc : Completes q q') (r : Rule) :
    ∀ acc acc', Sub acc acc' → Sub (forbStep q' acc r) (forbStep q acc' r) := by
  intro acc acc' h
  unfold forbStep
  by_cases hf : Forbids q' r
  · rw [if_pos hf, if_pos (Forbids_completes hc r hf)]
    exact sub_or h
  · rw [if_neg hf]
    by_cases hf' : Forbids q r
    · rw [if_pos hf']
      exact sub_trans h (sub_or_right acc' r.rights)
    · rw [if_neg hf']
      exact h

theorem permRights_sub_completes (rs : List Rule) (q q' : Req) (hc : Completes q q') :
    Sub (permRights rs q) (permRights rs q') := by
  suffices aux : ∀ (l : List Rule) acc acc', Sub acc acc' →
      Sub (l.foldl (permStep q) acc) (l.foldl (permStep q') acc') from aux rs 0 0 (sub_refl 0)
  intro l
  induction l with
  | nil => intro acc acc' h; exact h
  | cons r rest ih =>
    intro acc acc' h
    rw [List.foldl_cons, List.foldl_cons]
    exact ih (permStep q acc r) (permStep q' acc' r) (permStep_sub_completes q q' hc r acc acc' h)

theorem forbRights_sub_completes (rs : List Rule) (q q' : Req) (hc : Completes q q') :
    Sub (forbRights rs q') (forbRights rs q) := by
  suffices aux : ∀ (l : List Rule) acc acc', Sub acc acc' →
      Sub (l.foldl (forbStep q') acc) (l.foldl (forbStep q) acc') from aux rs 0 0 (sub_refl 0)
  intro l
  induction l with
  | nil => intro acc acc' h; exact h
  | cons r rest ih =>
    intro acc acc' h
    rw [List.foldl_cons, List.foldl_cons]
    exact ih (forbStep q' acc r) (forbStep q acc' r) (forbStep_sub_completes q q' hc r acc acc' h)

def minOpt (best : Option Nat) (id : Nat) : Option Nat :=
  match best with
  | none => some id
  | some b => some (min b id)

def minStep (c : Rule → Prop) [DecidablePred c] (acc : Option Nat) (r : Rule) : Option Nat :=
  if c r then minOpt acc r.id else acc

def permBy (rs : List Rule) (q : Req) : Option Nat :=
  rs.foldl (minStep (Permits q)) none

def askBy (rs : List Rule) (q : Req) : Option Nat :=
  rs.foldl (minStep (Asks q)) none

abbrev forbCond (q : Req) (P : Nat) (r : Rule) : Prop :=
  r.action = q.action ∧ r.effect = Effect.forbid ∧ band (band r.rights q.want) P ≠ 0 ∧
    guardEval q.facts r.guard ≠ Tri.f

def forbBy (rs : List Rule) (q : Req) (P : Nat) : Option Nat :=
  rs.foldl (minStep (forbCond q P)) none

def gOf (rs : List Rule) (q : Req) : Nat :=
  band (band (mask q.want) (permRights rs q)) (ncomp (forbRights rs q))

theorem gOf_lt (rs : List Rule) (q : Req) : gOf rs q < 2 ^ W := band_lt _ _

theorem gOf_sub_mask_want (rs : List Rule) (q : Req) : Sub (gOf rs q) (mask q.want) := by
  unfold gOf
  exact sub_trans (sub_and_right (band (mask q.want) (permRights rs q)) (ncomp (forbRights rs q)))
    (sub_and_right (mask q.want) (permRights rs q))

/-- The decision. Time O(R·(1+A)) atom evaluations and O(1) words of space, R = `rs.length`,
A = the longest guard: six folds over `rs` (three rights accumulators, three by-id minima), each
rule costing at most one `guardEval` of O(A) atoms; each atom and each `band`/`bor` is at most
W = 64 fuel steps. Total; no panic path.
The allow arm compares `gOf rs q` with `mask q.want` (not the raw `q.want`), so `want` bits above
the 64-bit width cannot turn an allow into an attenuate. -/
def decide (rs : List Rule) (q : Req) : Verdict :=
  if gOf rs q = 0 then
    if band (mask q.want) (permRights rs q) = 0 then
      Verdict.deny Cause.noPermit
    else
      Verdict.deny (Cause.forbid ((forbBy rs q (permRights rs q)).getD 0))
  else if band (gOf rs q) (askRights rs q) ≠ 0 then
    Verdict.ask (gOf rs q) (band (gOf rs q) (askRights rs q)) ((askBy rs q).getD 0)
  else if gOf rs q = mask q.want then
    Verdict.allow (gOf rs q) ((permBy rs q).getD 0)
  else
    Verdict.attenuate (gOf rs q) ((permBy rs q).getD 0)

def granted : Verdict → Nat
  | Verdict.allow g _ => g
  | Verdict.attenuate g _ => g
  | Verdict.ask g _ _ => g
  | Verdict.deny _ => 0

theorem granted_decide (rs : List Rule) (q : Req) : granted (decide rs q) = gOf rs q := by
  unfold decide granted
  by_cases h : gOf rs q = 0
  · rw [if_pos h]
    by_cases h2 : band (mask q.want) (permRights rs q) = 0
    · rw [if_pos h2]; exact h.symm
    · rw [if_neg h2]; exact h.symm
  · rw [if_neg h]
    by_cases h2 : band (gOf rs q) (askRights rs q) ≠ 0
    · rw [if_pos h2]
    · rw [if_neg h2]
      by_cases h3 : gOf rs q = mask q.want
      · rw [if_pos h3]
      · rw [if_neg h3]

end Apeiron.Policy
