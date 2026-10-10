/-
Canonical definition of the bounded authorization decision core
(docs/epics/policy-engine.md, "Canonical definition"), plus the small bit, guard
and fold toolkit the laws need. Lean core only, no external libraries; every
declaration here is constructive.

The 64-bit meet/join are the fuel-bounded `bitwise` of `Iota.Word.bitwise`
(proof/corpus/Iota/Word.lean:37-50), not core `Nat.land`/`Nat.lor`: the latter
depend on `propext` under Lean 4.31.0, which would put the definitions outside
the empty axiom cone.
-/
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

/-! ## Width -/

def W : Nat := 64

def mask (x : Nat) : Nat := x % 2 ^ W

def ncomp (x : Nat) : Nat := (2 ^ W - 1) - mask x

/-! ## Fuel-bounded bitwise operations -/

-- Temporal O(fuel): one recursive step per fuel decrement, each doing O(1) division/mod and one
-- `operation` call, so fuel bounds the steps. Spatial O(fuel) stack: the recursive result is bound
-- to `higher` before the conditional, so the recursion is not tail-recursive. Operations counted:
-- fuel decrements / bit extractions (`left % 2`, `left / 2`), one `operation` call per bit. The
-- callers `band`/`bor` pass `W = 64`, so each is O(1) with a 64-step constant.
def bitwise (operation : Bool → Bool → Bool) : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, left, right =>
      let higher := bitwise operation fuel (left / 2) (right / 2)
      let low := operation (left % 2 == 1) (right % 2 == 1)
      if low then higher + higher + 1 else higher + higher

/-- 64-bit meet (`⋀`). -/
def band (x y : Nat) : Nat := bitwise Bool.and W x y

/-- 64-bit join (`⋁`). -/
def bor (x y : Nat) : Nat := bitwise Bool.or W x y

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

/-! ## Bit toolkit -/

theorem bitwise_succ (operation : Bool → Bool → Bool) (n left right : Nat) :
    bitwise operation (n + 1) left right =
      (if operation (left % 2 == 1) (right % 2 == 1)
       then bitwise operation n (left / 2) (right / 2) + bitwise operation n (left / 2) (right / 2) + 1
       else bitwise operation n (left / 2) (right / 2) + bitwise operation n (left / 2) (right / 2)) := rfl

theorem bitwise_div_two (operation : Bool → Bool → Bool) (n left right : Nat) :
    bitwise operation (n + 1) left right / 2 = bitwise operation n (left / 2) (right / 2) := by
  rw [bitwise_succ]
  split <;> omega

theorem bitwise_mod_two (operation : Bool → Bool → Bool) (n left right : Nat) :
    bitwise operation (n + 1) left right % 2 =
      (if operation (left % 2 == 1) (right % 2 == 1) then 1 else 0) := by
  rw [bitwise_succ]
  split <;> omega

theorem bitwise_lt (operation : Bool → Bool → Bool) (n left right : Nat) :
    bitwise operation n left right < 2 ^ n := by
  induction n generalizing left right with
  | zero => simp [bitwise]
  | succ n ih =>
      rw [bitwise_succ]
      have h := ih (left / 2) (right / 2)
      have h2 : 2 ^ (n + 1) = 2 ^ n + 2 ^ n := by
        rw [Nat.pow_succ, Nat.mul_comm, Nat.two_mul]
      split <;> omega

theorem bitwise_testBit (operation : Bool → Bool → Bool) (n left right i : Nat) :
    (bitwise operation n left right).testBit i =
      (decide (i < n) && operation (left.testBit i) (right.testBit i)) := by
  induction n generalizing left right i with
  | zero =>
      rw [show bitwise operation 0 left right = 0 from rfl, Nat.zero_testBit]
      simp
  | succ n ih =>
      cases i with
      | zero =>
          rw [Nat.testBit_zero, bitwise_mod_two]
          have h0 : decide ((if operation (left % 2 == 1) (right % 2 == 1) then 1 else 0) = 1)
              = operation (left % 2 == 1) (right % 2 == 1) := by
            by_cases hl : operation (left % 2 == 1) (right % 2 == 1) <;> simp [hl]
          rw [h0, Nat.testBit_zero, Nat.testBit_zero]
          rfl
      | succ j =>
          rw [Nat.testBit_succ, bitwise_div_two, ih (left / 2) (right / 2) j,
              ← Nat.testBit_succ left j, ← Nat.testBit_succ right j]
          have hp : j + 1 < n + 1 ↔ j < n := by omega
          simp only [hp]

theorem band_lt (x y : Nat) : band x y < 2 ^ W := bitwise_lt Bool.and W x y

theorem bor_lt (x y : Nat) : bor x y < 2 ^ W := bitwise_lt Bool.or W x y

theorem testBit_band (x y i : Nat) :
    (band x y).testBit i = (decide (i < W) && (x.testBit i && y.testBit i)) := by
  unfold band
  rw [bitwise_testBit]

theorem testBit_bor (x y i : Nat) :
    (bor x y).testBit i = (decide (i < W) && (x.testBit i || y.testBit i)) := by
  unfold bor
  rw [bitwise_testBit]

theorem testBit_mask (x i : Nat) : (mask x).testBit i = (decide (i < W) && x.testBit i) :=
  Nat.testBit_mod_two_pow x W i

theorem mask_lt (x : Nat) : mask x < 2 ^ W := Nat.mod_lt x (Nat.two_pow_pos W)

theorem mask_mask (x : Nat) : mask (mask x) = mask x := Nat.mod_eq_of_lt (mask_lt x)

theorem mask_band (x y : Nat) : mask (band x y) = band x y := Nat.mod_eq_of_lt (band_lt x y)

theorem mask_bor (x y : Nat) : mask (bor x y) = bor x y := Nat.mod_eq_of_lt (bor_lt x y)

theorem testBit_band_of_lt {x y i : Nat} (hi : i < W) :
    (band x y).testBit i = (x.testBit i && y.testBit i) := by
  rw [testBit_band]; simp [hi]

theorem testBit_bor_of_lt {x y i : Nat} (hi : i < W) :
    (bor x y).testBit i = (x.testBit i || y.testBit i) := by
  rw [testBit_bor]; simp [hi]

theorem testBit_mask_of_lt {x i : Nat} (hi : i < W) : (mask x).testBit i = x.testBit i := by
  rw [testBit_mask]; simp [hi]

theorem eq_of_testBit_lt {x y : Nat} (hx : x < 2 ^ W) (hy : y < 2 ^ W)
    (h : ∀ i, i < W → x.testBit i = y.testBit i) : x = y := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < W
  · exact h i hi
  · have hWi : W ≤ i := Nat.le_of_not_lt hi
    have hxi : x < 2 ^ i := Nat.lt_of_lt_of_le hx (Nat.pow_le_pow_right (by omega) hWi)
    have hyi : y < 2 ^ i := Nat.lt_of_lt_of_le hy (Nat.pow_le_pow_right (by omega) hWi)
    rw [Nat.testBit_lt_two_pow hxi, Nat.testBit_lt_two_pow hyi]

theorem band_comm (x y : Nat) : band x y = band y x := by
  apply eq_of_testBit_lt (band_lt x y) (band_lt y x)
  intro i hi
  rw [testBit_band_of_lt hi, testBit_band_of_lt hi]
  exact Bool.and_comm _ _

theorem band_assoc (x y z : Nat) : band (band x y) z = band x (band y z) := by
  apply eq_of_testBit_lt (band_lt (band x y) z) (band_lt x (band y z))
  intro i hi
  rw [testBit_band_of_lt hi, testBit_band_of_lt hi, testBit_band_of_lt hi, testBit_band_of_lt hi]
  exact Bool.and_assoc _ _ _

theorem band_assoc_comm (a b c : Nat) : band (band a b) c = band (band a c) b := by
  apply eq_of_testBit_lt (band_lt (band a b) c) (band_lt (band a c) b)
  intro i hi
  rw [testBit_band_of_lt hi, testBit_band_of_lt hi, testBit_band_of_lt hi, testBit_band_of_lt hi]
  cases a.testBit i <;> cases b.testBit i <;> cases c.testBit i <;> simp

theorem band_idem (x : Nat) : band x x = mask x := by
  apply eq_of_testBit_lt (band_lt x x) (mask_lt x)
  intro i hi
  rw [testBit_band_of_lt hi, testBit_mask_of_lt hi]
  cases x.testBit i <;> simp

theorem band_mask_left (x y : Nat) : band (mask x) y = band x y := by
  apply eq_of_testBit_lt (band_lt (mask x) y) (band_lt x y)
  intro i hi
  rw [testBit_band_of_lt hi, testBit_mask_of_lt hi, testBit_band_of_lt hi]

theorem band_zero (x : Nat) : band x 0 = 0 := by
  apply Nat.eq_of_testBit_eq; intro i
  rw [testBit_band, Nat.zero_testBit]
  simp

theorem zero_band (x : Nat) : band 0 x = 0 := by
  apply Nat.eq_of_testBit_eq; intro i
  rw [testBit_band, Nat.zero_testBit]
  simp

theorem bor_comm (x y : Nat) : bor x y = bor y x := by
  apply eq_of_testBit_lt (bor_lt x y) (bor_lt y x)
  intro i hi
  rw [testBit_bor_of_lt hi, testBit_bor_of_lt hi]
  exact Bool.or_comm _ _

theorem bor_assoc (x y z : Nat) : bor (bor x y) z = bor x (bor y z) := by
  apply eq_of_testBit_lt (bor_lt (bor x y) z) (bor_lt x (bor y z))
  intro i hi
  rw [testBit_bor_of_lt hi, testBit_bor_of_lt hi, testBit_bor_of_lt hi, testBit_bor_of_lt hi]
  exact Bool.or_assoc _ _ _

/-- The join is width-masked, so the zero operand leaves the mask of the other one:
`bor x 0 = mask x`. The retired statement `bor x 0 = 0` is false, with the joint witness
`x = 3` (`bor 3 0 = 3 ≠ 0`), and had no consumer in this package. -/
theorem bor_zero_eq_mask (x : Nat) : bor x 0 = mask x := by
  apply eq_of_testBit_lt (bor_lt x 0) (mask_lt x)
  intro i hi
  rw [testBit_bor_of_lt hi, testBit_mask_of_lt hi, Nat.zero_testBit]
  simp [Bool.or_false]

theorem bor_swap (z a b : Nat) : bor (bor z a) b = bor (bor z b) a := by
  rw [bor_assoc, bor_assoc, bor_comm a b]

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

theorem testBit_ncomp (x i : Nat) :
    (ncomp x).testBit i = (decide (i < W) && !(x.testBit i)) := by
  unfold ncomp mask
  by_cases hi : i < W
  · rw [testBit_compl W (x % 2 ^ W) i (Nat.mod_lt x (Nat.two_pow_pos W)) hi]
    simp [hi]
  · have hlt : (2 ^ W - 1 - x % 2 ^ W) < 2 ^ W := by
      have h1 : x % 2 ^ W < 2 ^ W := Nat.mod_lt x (Nat.two_pow_pos W)
      omega
    have h := Nat.testBit_mod_two_pow (2 ^ W - 1 - x % 2 ^ W) W i
    rw [Nat.mod_eq_of_lt hlt] at h
    simp [hi] at h
    rw [h]
    simp [hi]

/-! ## Submask order `Sub x y` := `band x y = mask x` -/

def Sub (x y : Nat) : Prop := band x y = mask x

-- The bound variables are implicit so that the `Iff` projection `.mp`/`.mpr` resolves; with
-- explicit binders `sub_iff.mpr` is not a constant.
theorem sub_iff {x y : Nat} :
    Sub x y ↔ ∀ i, i < W → x.testBit i = true → y.testBit i = true := by
  constructor
  · intro h i hi hx
    have hm : (mask x).testBit i = true := by rw [testBit_mask_of_lt hi]; exact hx
    have hb : (band x y).testBit i = true := by rw [h]; exact hm
    rw [testBit_band_of_lt hi] at hb
    rw [Bool.and_eq_true] at hb
    exact hb.2
  · intro h
    apply eq_of_testBit_lt (band_lt x y) (mask_lt x)
    intro i hi
    rw [testBit_band_of_lt hi, testBit_mask_of_lt hi]
    cases hx : x.testBit i with
    | false => simp
    | true => simp [h i hi hx]

theorem sub_refl (x : Nat) : Sub x x := by
  apply sub_iff.mpr
  intro i _ hx
  exact hx

theorem sub_testBit {x y : Nat} (h : Sub x y) {i : Nat} (hi : i < W) :
    x.testBit i = true → y.testBit i = true := (sub_iff.mp h) i hi

theorem sub_trans {x y z : Nat} (h1 : Sub x y) (h2 : Sub y z) : Sub x z :=
  sub_iff.mpr (fun _ hi hx => sub_testBit h2 hi (sub_testBit h1 hi hx))

theorem sub_or_right (x y : Nat) : Sub x (bor x y) := by
  apply sub_iff.mpr
  intro i hi hx
  rw [testBit_bor_of_lt hi]
  simp [hx]

theorem sub_or_left (x y : Nat) : Sub y (bor x y) := by
  apply sub_iff.mpr
  intro i hi hy
  rw [testBit_bor_of_lt hi]
  simp [hy]

theorem sub_or {x y z : Nat} (h : Sub x y) : Sub (bor x z) (bor y z) := by
  apply sub_iff.mpr
  intro i hi hx
  rw [testBit_bor_of_lt hi] at hx
  rw [testBit_bor_of_lt hi]
  rw [Bool.or_eq_true] at hx ⊢
  rcases hx with hx | hx
  · exact Or.inl (sub_testBit h hi hx)
  · exact Or.inr hx

theorem sub_and_right (x y : Nat) : Sub (band x y) x := by
  apply sub_iff.mpr
  intro i hi hx
  rw [testBit_band_of_lt hi] at hx
  rw [Bool.and_eq_true] at hx
  exact hx.1

theorem sub_and_left {y z : Nat} (h : Sub y z) (x : Nat) : Sub (band x y) (band x z) := by
  apply sub_iff.mpr
  intro i hi hx
  rw [testBit_band_of_lt hi] at hx
  rw [Bool.and_eq_true] at hx
  rw [testBit_band_of_lt hi, Bool.and_eq_true]
  exact ⟨hx.1, sub_testBit h hi hx.2⟩

theorem sub_and_mul {a b c d : Nat} (h1 : Sub a b) (h2 : Sub c d) :
    Sub (band a c) (band b d) := by
  apply sub_iff.mpr
  intro i hi hx
  rw [testBit_band_of_lt hi] at hx
  rw [Bool.and_eq_true] at hx
  rw [testBit_band_of_lt hi, Bool.and_eq_true]
  exact ⟨sub_testBit h1 hi hx.1, sub_testBit h2 hi hx.2⟩

theorem sub_ncomp {y z : Nat} (h : Sub y z) : Sub (ncomp z) (ncomp y) := by
  apply sub_iff.mpr
  intro i hi hx
  rw [testBit_ncomp] at hx ⊢
  simp only [hi, decide_true, Bool.true_and] at hx ⊢
  have hz : z.testBit i = false := by
    cases hzi : z.testBit i with
    | false => rfl
    | true =>
        rw [hzi] at hx
        exact absurd hx (by simp)
  have hy : y.testBit i = false := by
    cases hyi : y.testBit i with
    | false => rfl
    | true =>
        have hz' := sub_testBit h hi hyi
        rw [hz'] at hz
        exact absurd hz (by simp)
  rw [hy]
  rfl

theorem sub_comp_zero {x y : Nat} (h : Sub x y) : band x (ncomp y) = 0 := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.zero_testBit]
  by_cases hi : i < W
  · rw [testBit_band_of_lt hi, testBit_ncomp]
    simp only [hi, decide_true, Bool.true_and]
    cases hx : x.testBit i with
    | false => simp
    | true => simp [sub_testBit h hi hx]
  · rw [testBit_band]
    simp [hi]

theorem band_ncomp_self (x : Nat) : band x (ncomp x) = 0 := sub_comp_zero (sub_refl x)

theorem band_bor (a x y : Nat) : band a (bor x y) = bor (band a x) (band a y) := by
  apply eq_of_testBit_lt (band_lt a (bor x y)) (bor_lt (band a x) (band a y))
  intro i hi
  rw [testBit_band_of_lt hi, testBit_bor_of_lt hi, testBit_bor_of_lt hi,
      testBit_band_of_lt hi, testBit_band_of_lt hi]
  cases a.testBit i <;> cases x.testBit i <;> cases y.testBit i <;> simp

theorem band_eq_self_of_band_ncomp_eq_zero {a y : Nat} (ha : a < 2 ^ W)
    (h : band a (ncomp y) = 0) : band a y = a := by
  apply eq_of_testBit_lt (band_lt a y) ha
  intro i hi
  rw [testBit_band_of_lt hi]
  have hb := congrArg (fun t => t.testBit i) h
  rw [testBit_band_of_lt hi, Nat.zero_testBit, testBit_ncomp] at hb
  simp only [hi, decide_true, Bool.true_and] at hb
  cases ha_i : a.testBit i with
  | false => simp
  | true =>
      rw [ha_i] at hb
      simp only [Bool.true_and] at hb
      cases hy_i : y.testBit i with
      | false => rw [hy_i] at hb; simp at hb
      | true => simp

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
      exact sub_trans (sub_or_left acc r.rights)
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
      exact sub_trans (sub_or_left acc r.rights)
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

def permBy (rs : List Rule) (q : Req) : Option Nat :=
  rs.foldl (fun acc r => if Permits q r then minOpt acc r.id else acc) none

def askBy (rs : List Rule) (q : Req) : Option Nat :=
  rs.foldl (fun acc r => if Asks q r then minOpt acc r.id else acc) none

def forbBy (rs : List Rule) (q : Req) (P : Nat) : Option Nat :=
  rs.foldl (fun acc r =>
    if r.action = q.action ∧ r.effect = Effect.forbid ∧ band (band r.rights q.want) P ≠ 0 ∧
        guardEval q.facts r.guard ≠ Tri.f
    then minOpt acc r.id else acc) none

def gOf (rs : List Rule) (q : Req) : Nat :=
  band (band (mask q.want) (permRights rs q)) (ncomp (forbRights rs q))

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
