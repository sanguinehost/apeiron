/-
APEIRON-POLICY-W1-T4: integer-row encoding of the 1-bit policy analysis.

`docs/epics/policy-engine.md` ("Analysis layer (D7)") fixes the canonical decision
`Apeiron.Policy.decide`; this file encodes the "some completion allows with want = 1" question as a
finite integer-row system over `Fragment.Row` and decides it with `Apeiron.Decider.completeSat`.
The encoding is exact for `Op.eq`, `Op.le`, `Op.ge` on slots below the column count; `Op.all`,
`Op.any` and out-of-range slots are not encodable and yield the empty DNF.
-/
import Policy.Core
import Apeiron.Decider.CompleteDecision

-- This module carries the statement of each declaration in prose; the canonical corpus'
-- per-symbol gloss linter is disabled here as in the other bridge modules that cite rows.
set_option linter.apeiron.notation false

namespace Apeiron.Policy.Analysis

open Apeiron.Policy
open Apeiron.Fragment (Row)
open Apeiron.Decider (SatRows completeSat completeSat_eq_true_iff completeSat_eq_false_iff)

/-- The 64-bit maximum value a fact column can take. -/
def M : ℤ := (2 : ℤ) ^ 64 - 1

/-- The row `c + f_s ≥ 0`, that is the lower bound `f_s ≥ -c` on column `s`. -/
def colRow {n : Nat} (s : Fin n) (c : ℤ) : Row n :=
  ⟨c, fun i => if i = s then 1 else 0⟩

/-- The row `c - f_s ≥ 0`, that is the upper bound `f_s ≤ c` on column `s`. -/
def negColRow {n : Nat} (s : Fin n) (c : ℤ) : Row n :=
  ⟨c, fun i => if i = s then -1 else 0⟩

/-- The value of `colRow s c` at `ρ` is `c + ρ s`. -/
theorem colRow_value {n : Nat} (s : Fin n) (c : ℤ) (ρ : Fin n → ℤ) :
    (colRow s c).value ρ = c + ρ s := by
  have hsum : (∑ i, (if i = s then (1 : ℤ) else 0) * ρ i) = ρ s := by
    simp only [ite_mul, zero_mul]
    rw [Finset.sum_ite_eq']
    simp
  simp only [Apeiron.Fragment.Row.value, colRow, hsum]

/-- The value of `negColRow s c` at `ρ` is `c - ρ s`. -/
theorem negColRow_value {n : Nat} (s : Fin n) (c : ℤ) (ρ : Fin n → ℤ) :
    (negColRow s c).value ρ = c - ρ s := by
  have hsum : (∑ i, (if i = s then (-1 : ℤ) else 0) * ρ i) = -ρ s := by
    simp only [ite_mul, zero_mul]
    rw [Finset.sum_ite_eq']
    simp
  simp only [Apeiron.Fragment.Row.value, negColRow, hsum]
  ring

/-- Membership in a one-branch system is the single row conjunction. -/
theorem satRows_pair {n : Nat} (ρ : Fin n → ℤ) (r₁ r₂ : Row n) :
    SatRows ρ [r₁, r₂] ↔ Row.Holds ρ r₁ ∧ Row.Holds ρ r₂ := by
  simp [SatRows]

/-- Membership in a one-row system is that row. -/
theorem satRows_single {n : Nat} (ρ : Fin n → ℤ) (r : Row n) :
    SatRows ρ [r] ↔ Row.Holds ρ r := by
  simp [SatRows]

/-- The bound system: `0 ≤ f_s ≤ 2^64 - 1` for every one of the `n` fact columns. -/
def boundRows (n : Nat) : List (Row n) :=
  (List.finRange n).flatMap fun s => [colRow s 0, negColRow s M]

/-- The DNF of `a` being true. `Op.eq` is the one branch `f_s ≥ v ∧ f_s ≤ v`, `Op.le` is
    `f_s ≤ v`, `Op.ge` is `f_s ≥ v`; `Op.all`, `Op.any` and `a.slot ≥ n` are not encodable and
    contribute no branch. -/
def atomRows {n : Nat} (a : Atom) : List (List (Row n)) :=
  if h : a.slot < n then
    let s : Fin n := ⟨a.slot, h⟩
    match a.op with
    | Op.eq => [[colRow s (-(a.value : ℤ)), negColRow s (a.value : ℤ)]]
    | Op.le => [[negColRow s (a.value : ℤ)]]
    | Op.ge => [[colRow s (-(a.value : ℤ))]]
    | Op.all => []
    | Op.any => []
  else []

/-- The DNF of `a` being false, using integer strictness: `¬(f ≤ v)` is `f ≥ v + 1`, `¬(v ≤ f)`
    is `f ≤ v - 1`, `¬(f = v)` is the two branches `f ≤ v - 1` and `f ≥ v + 1`. -/
def negAtomRows {n : Nat} (a : Atom) : List (List (Row n)) :=
  if h : a.slot < n then
    let s : Fin n := ⟨a.slot, h⟩
    match a.op with
    | Op.eq => [[negColRow s ((a.value : ℤ) - 1)], [colRow s (-((a.value : ℤ) + 1))]]
    | Op.le => [[colRow s (-((a.value : ℤ) + 1))]]
    | Op.ge => [[negColRow s ((a.value : ℤ) - 1)]]
    | Op.all => []
    | Op.any => []
  else []

/-- Conjunction of two DNFs: the branch product. -/
def dnfProduct {n : Nat} (xs ys : List (List (Row n))) : List (List (Row n)) :=
  xs.flatMap fun x => ys.map fun y => x ++ y

/-- The DNF of a guard being true; the empty guard and the absent guard are not true. -/
def guardRows {n : Nat} : Option (List Atom) → List (List (Row n))
  | none => []
  | some as => as.foldr (fun a acc => dnfProduct (atomRows a) acc) [[]]

/-- The DNF of a guard being false: a disjunction of the negated atoms. -/
def negGuardRows {n : Nat} : Option (List Atom) → List (List (Row n))
  | none => []
  | some as => as.flatMap negAtomRows

/-- The permits relevant to `action` whose rights bit 0 is set. -/
def relevantPermits (rs : List Rule) (action : Nat) : List Rule :=
  rs.filter fun p => p.action = action ∧ p.effect = Effect.permit ∧ band p.rights 1 ≠ 0

/-- The forbids relevant to `action` whose rights bit 0 is set. -/
def relevantForbids (rs : List Rule) (action : Nat) : List Rule :=
  rs.filter fun f => f.action = action ∧ f.effect = Effect.forbid ∧ band f.rights 1 ≠ 0

/-- The DNF of "every relevant forbid has a false guard": the product of their negated guards.
    An empty forbid set gives the single empty branch. -/
def forbidBranches {n : Nat} (rs : List Rule) (action : Nat) : List (List (Row n)) :=
  (relevantForbids rs action).foldr
    (fun f acc => dnfProduct (negGuardRows (n := n) f.guard) acc) [[]]

/-- The 1-bit analysis: `some` relevant permit has a true guard and every relevant forbid has a
    false guard, over facts in the 64-bit bounds.

    Time: one `completeSat` call per (relevant permit, true-DNF branch, false-forbid product)
    triple, so `|relevantPermits| · max_p |guardRows p.guard| · Π_f |negGuardRows f.guard|` calls,
    each on a row list of size `2n + |branch|`. `completeSat`'s own finite residue search has no
    stated bound (`Apeiron/Decider/CompleteDecision.lean:16-17`), so the total is not bounded by
    this file. Space: one DNF branch at a time. -/
def canAllow (rs : List Rule) (action : Nat) (n : Nat) : Bool :=
  (relevantPermits rs action).any fun p =>
    (guardRows (n := n) p.guard).any fun pb =>
      (forbidBranches (n := n) rs action).any fun fb =>
        completeSat n (boundRows n ++ pb ++ fb)

/-! ## Atom encoding link -/

/-- A two-valued `Tri` test is `Tri.t` exactly on its condition. -/
theorem ite_tri {c : Prop} [Decidable c] : (if c then Tri.t else Tri.f) = Tri.t ↔ c := by
  by_cases h : c <;> simp [h]

/-- The DNF encoding of an `Op.eq`/`Op.le`/`Op.ge` atom on a slot below the column count is exact
    for bounded integer fact assignments. -/
theorem atomRows_iff {n : Nat} (hn : n ≤ W) {a : Atom} (hsl : a.slot < n)
    (hop : a.op = Op.eq ∨ a.op = Op.le ∨ a.op = Op.ge)
    (ρ : Fin n → ℤ) (hρ : ∀ s, 0 ≤ ρ s ∧ ρ s ≤ M) :
    Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none) a = Tri.t ↔
      ∃ br ∈ atomRows a, SatRows ρ br := by
  obtain ⟨op, slot, value⟩ := a
  dsimp only at hsl ⊢
  have hif : slot < W := lt_of_lt_of_le hsl hn
  rcases hop with h | h | h
  · subst h
    have hL : Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none)
        (⟨Op.eq, slot, value⟩ : Atom) =
        (if (ρ ⟨slot, hsl⟩).toNat = value then Tri.t else Tri.f) := by
      simp only [Atom.eval, if_pos hif, dif_pos hsl]
    have hR : atomRows (⟨Op.eq, slot, value⟩ : Atom) =
        [[colRow ⟨slot, hsl⟩ (-(value : ℤ)), negColRow ⟨slot, hsl⟩ (value : ℤ)]] := by
      simp only [atomRows, dif_pos hsl]
    rw [hL, ite_tri, hR]
    constructor
    · intro hval
      have h2 : ((ρ ⟨slot, hsl⟩).toNat : ℤ) = (value : ℤ) := by rw [hval]
      rw [Int.toNat_of_nonneg (hρ _).1] at h2
      refine ⟨[colRow ⟨slot, hsl⟩ (-(value : ℤ)), negColRow ⟨slot, hsl⟩ (value : ℤ)],
        by simp, ?_⟩
      rw [satRows_pair]
      constructor
      · rw [Row.Holds, colRow_value]; omega
      · rw [Row.Holds, negColRow_value]; omega
    · rintro ⟨br, hbr, hsat⟩
      rw [List.mem_singleton] at hbr
      subst hbr
      rw [satRows_pair] at hsat
      obtain ⟨h1, h2⟩ := hsat
      rw [Row.Holds, colRow_value] at h1
      rw [Row.Holds, negColRow_value] at h2
      have hval : ρ ⟨slot, hsl⟩ = (value : ℤ) := by omega
      rw [hval, Int.toNat_natCast]
  · subst h
    have hL : Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none)
        (⟨Op.le, slot, value⟩ : Atom) =
        (if (ρ ⟨slot, hsl⟩).toNat ≤ value then Tri.t else Tri.f) := by
      simp only [Atom.eval, if_pos hif, dif_pos hsl]
    have hR : atomRows (⟨Op.le, slot, value⟩ : Atom) =
        [[negColRow ⟨slot, hsl⟩ (value : ℤ)]] := by
      simp only [atomRows, dif_pos hsl]
    rw [hL, ite_tri, hR]
    constructor
    · intro hval
      refine ⟨[negColRow ⟨slot, hsl⟩ (value : ℤ)], by simp, ?_⟩
      rw [satRows_single, Row.Holds, negColRow_value]
      omega
    · rintro ⟨br, hbr, hsat⟩
      rw [List.mem_singleton] at hbr
      subst hbr
      rw [satRows_single, Row.Holds, negColRow_value] at hsat
      exact (Int.toNat_le).mpr (by omega)
  · subst h
    have hL : Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none)
        (⟨Op.ge, slot, value⟩ : Atom) =
        (if value ≤ (ρ ⟨slot, hsl⟩).toNat then Tri.t else Tri.f) := by
      simp only [Atom.eval, if_pos hif, dif_pos hsl]
    have hR : atomRows (⟨Op.ge, slot, value⟩ : Atom) =
        [[colRow ⟨slot, hsl⟩ (-(value : ℤ))]] := by
      simp only [atomRows, dif_pos hsl]
    rw [hL, ite_tri, hR]
    constructor
    · intro hval
      refine ⟨[colRow ⟨slot, hsl⟩ (-(value : ℤ))], by simp, ?_⟩
      rw [satRows_single, Row.Holds, colRow_value]
      have hle : (value : ℤ) ≤ ρ ⟨slot, hsl⟩ := (Int.le_toNat (hρ ⟨slot, hsl⟩).1).mp hval
      omega
    · rintro ⟨br, hbr, hsat⟩
      rw [List.mem_singleton] at hbr
      subst hbr
      rw [satRows_single, Row.Holds, colRow_value] at hsat
      exact (Int.le_toNat (hρ _).1).mpr (by omega)

/-! ## Concrete witnesses -/

/-- A permit with one atom guard and rights bit 0 set. -/
def permitRule (id action : Nat) (a : Atom) : Rule :=
  ⟨id, Effect.permit, action, 1, some [a]⟩

/-- A forbid with one atom guard and rights bit 0 set. -/
def forbidRule (id action : Nat) (a : Atom) : Rule :=
  ⟨id, Effect.forbid, action, 1, some [a]⟩

/-- The `f_s ≤ v` atom. -/
def leAtom (slot v : Nat) : Atom := ⟨Op.le, slot, v⟩

/-- The `f_s ≥ v` atom. -/
def geAtom (slot v : Nat) : Atom := ⟨Op.ge, slot, v⟩

/-- Shadowed permit: permit `f0 ≤ 10`, forbid `f0 ≤ 20`; their conjunction is infeasible. -/
def shadowedRules : List Rule :=
  [permitRule 1 7 (leAtom 0 10), forbidRule 2 7 (leAtom 0 20)]

/-- The single allow branch of `shadowedRules`. -/
def shadowedBranch : List (Row 1) :=
  boundRows 1 ++ [negColRow (0 : Fin 1) 10] ++ [colRow (0 : Fin 1) (-21)]

/-- `canAllow` on `shadowedRules` is the decision of its single branch. -/
theorem canAllow_shadowed_eq :
    canAllow shadowedRules 7 1 = completeSat 1 shadowedBranch := rfl

/-- The shadowed permit is refused, and no bounded fact assignment meets its allow branch. -/
theorem shadowed_witness :
    canAllow shadowedRules 7 1 = false ∧ ¬ ∃ ρ : Fin 1 → ℤ, SatRows ρ shadowedBranch := by
  have h : completeSat 1 shadowedBranch = false := by decide
  exact ⟨by rw [canAllow_shadowed_eq, h],
    (completeSat_eq_false_iff 1 shadowedBranch).mp h⟩

/-- Live permit: permit `f0 ≤ 10`, forbid `f0 ≥ 5`; the permit branch survives. -/
def liveRules : List Rule :=
  [permitRule 1 7 (leAtom 0 10), forbidRule 2 7 (geAtom 0 5)]

/-- The single allow branch of `liveRules`: `0 ≤ f0 ≤ 4`. -/
def liveBranch : List (Row 1) :=
  boundRows 1 ++ [negColRow (0 : Fin 1) 10] ++ [negColRow (0 : Fin 1) 4]

/-- `canAllow` on `liveRules` is the decision of its single branch. -/
theorem canAllow_live_eq : canAllow liveRules 7 1 = completeSat 1 liveBranch := rfl

/-- The live permit is granted, with an explicit bounded witness. -/
theorem live_witness :
    canAllow liveRules 7 1 = true ∧ ∃ ρ : Fin 1 → ℤ, SatRows ρ liveBranch := by
  have h : completeSat 1 liveBranch = true := by decide
  exact ⟨by rw [canAllow_live_eq, h], (completeSat_eq_true_iff 1 liveBranch).mp h⟩

/-- Exactness under an unknown fact: permits `f0 ≤ 5` and `f0 ≥ 6` cover every bounded value. -/
def unknownRules : List Rule :=
  [permitRule 1 7 (leAtom 0 5), permitRule 2 7 (geAtom 0 6)]

/-- The allow branch of the first permit (`f0 ≤ 5`), read off the encoding. -/
def unknownBranch : List (Row 1) :=
  boundRows 1 ++ (guardRows (n := 1) (some [leAtom 0 5])).flatten

/-- The branch where both permits are false: `f0 ≥ 6 ∧ f0 ≤ 5`, which is unsatisfiable. It is the
    conjunction of the encoded negations of both permit guards. -/
def unknownNegBranch : List (Row 1) :=
  boundRows 1
    ++ (negGuardRows (n := 1) (some [leAtom 0 5])).flatten
    ++ (negGuardRows (n := 1) (some [geAtom 0 6])).flatten

/-- A request with every fact slot unknown, `want = 1`. -/
def queryUnknown : Req := ⟨fun _ => none, 7, 1⟩

/-- The analysis allows, the "no permit true" system over the bounds is unsatisfiable, and the
    canonical Kleene decision on the same request denies with no permit. -/
theorem unknown_exact_witness :
    canAllow unknownRules 7 1 = true ∧
      completeSat 1 unknownNegBranch = false ∧
      (¬ ∃ ρ : Fin 1 → ℤ, SatRows ρ unknownNegBranch) ∧
      Apeiron.Policy.decide unknownRules queryUnknown = Verdict.deny Cause.noPermit := by
  have htrue : canAllow unknownRules 7 1 = true := by decide
  have hfalse : completeSat 1 unknownNegBranch = false := by decide
  exact ⟨htrue, hfalse, (completeSat_eq_false_iff 1 unknownNegBranch).mp hfalse, by rfl⟩

theorem satRows_append {n : Nat} (ρ : Fin n → ℤ) (l₁ l₂ : List (Row n)) :
    SatRows ρ (l₁ ++ l₂) ↔ SatRows ρ l₁ ∧ SatRows ρ l₂ := by
  constructor
  · intro h
    exact ⟨fun r hr => h r (List.mem_append.mpr (Or.inl hr)),
      fun r hr => h r (List.mem_append.mpr (Or.inr hr))⟩
  · rintro ⟨h1, h2⟩ r hr
    rcases List.mem_append.mp hr with hr | hr
    · exact h1 r hr
    · exact h2 r hr

theorem satRows_boundRows {n : Nat} (ρ : Fin n → ℤ) (hρ : ∀ s, 0 ≤ ρ s ∧ ρ s ≤ M) :
    SatRows ρ (boundRows n) := by
  intro r hr
  rw [boundRows] at hr
  simp only [List.mem_flatMap, List.mem_cons, List.not_mem_nil, or_false] at hr
  obtain ⟨s, _, hcases⟩ := hr
  rcases hcases with rfl | rfl
  · rw [Row.Holds, colRow_value]; linarith [(hρ s).1]
  · rw [Row.Holds, negColRow_value]; linarith [(hρ s).2]

theorem mem_dnfProduct {n : Nat} {xs ys : List (List (Row n))} {b : List (Row n)} :
    b ∈ dnfProduct xs ys ↔ ∃ x ∈ xs, ∃ y ∈ ys, b = x ++ y := by
  simp only [dnfProduct, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨x, hx, y, hy, h⟩; exact ⟨x, hx, y, hy, h.symm⟩
  · rintro ⟨x, hx, y, hy, h⟩; exact ⟨x, hx, y, hy, h.symm⟩

theorem exists_sat_foldr {n : Nat} {α : Type} (ρ : Fin n → ℤ) (F : α → List (List (Row n)))
    (as : List α) :
    ∀ (acc : List (List (Row n))),
      (∀ a ∈ as, ∃ b ∈ F a, SatRows ρ b) →
      (∃ y ∈ acc, SatRows ρ y) →
      ∃ pb ∈ as.foldr (fun a acc => dnfProduct (F a) acc) acc, SatRows ρ pb := by
  induction as with
  | nil => intro acc _ hy; exact hy
  | cons a rest ih =>
      intro acc ha hy
      rw [List.foldr_cons]
      have hb := ha a (by simp)
      have hrest : ∀ c ∈ rest, ∃ b ∈ F c, SatRows ρ b := fun c hc => ha c (by simp [hc])
      obtain ⟨b, hbF, hbs⟩ := hb
      obtain ⟨y, hy', hys⟩ := ih acc hrest hy
      exact ⟨b ++ y, (mem_dnfProduct).mpr ⟨b, hbF, y, hy', rfl⟩,
        (satRows_append ρ b y).mpr ⟨hbs, hys⟩⟩

theorem satRows_nil {n : Nat} (ρ : Fin n → ℤ) : SatRows ρ ([] : List (Row n)) := by
  simp [SatRows]

theorem exists_sat_guardRows {n : Nat} (ρ : Fin n → ℤ) (as : List Atom)
    (h : ∀ a ∈ as, ∃ b ∈ atomRows a, SatRows ρ b) :
    ∃ pb ∈ guardRows (n := n) (some as), SatRows ρ pb := by
  simpa [guardRows] using
    exists_sat_foldr ρ (fun a => atomRows a) as [[]] h ⟨[], by simp, satRows_nil ρ⟩

theorem exists_sat_negGuardRows {n : Nat} (ρ : Fin n → ℤ) (as : List Atom)
    (h : ∃ a ∈ as, ∃ b ∈ negAtomRows a, SatRows ρ b) :
    ∃ pb ∈ negGuardRows (n := n) (some as), SatRows ρ pb := by
  simp only [negGuardRows, List.mem_flatMap]
  obtain ⟨a, ha, b, hb, hbs⟩ := h
  exact ⟨b, ⟨a, ha, hb⟩, hbs⟩

theorem exists_sat_forbidBranches {n : Nat} (ρ : Fin n → ℤ) (rs : List Rule) (action : Nat)
    (h : ∀ (f : Rule), f ∈ relevantForbids rs action →
      ∃ b ∈ negGuardRows (n := n) f.guard, SatRows ρ b) :
    ∃ fb ∈ forbidBranches (n := n) rs action, SatRows ρ fb := by
  unfold forbidBranches
  exact exists_sat_foldr ρ (fun f => negGuardRows (n := n) f.guard) (relevantForbids rs action)
    [[]] h ⟨[], by simp, satRows_nil ρ⟩

theorem canAllow_true_of_branch {n : Nat} (rs : List Rule) (action : Nat) (ρ : Fin n → ℤ)
    (p : Rule) (hp : p ∈ relevantPermits rs action)
    (pb : List (Row n)) (hpb : pb ∈ guardRows (n := n) p.guard)
    (fb : List (Row n)) (hfb : fb ∈ forbidBranches (n := n) rs action)
    (hsat : SatRows ρ (boundRows n ++ pb ++ fb)) :
    canAllow rs action n = true := by
  have hcs : completeSat n (boundRows n ++ pb ++ fb) = true :=
    (completeSat_eq_true_iff n _).mpr ⟨ρ, hsat⟩
  unfold canAllow
  rw [List.any_eq_true]
  refine ⟨p, hp, ?_⟩
  rw [List.any_eq_true]
  refine ⟨pb, hpb, ?_⟩
  rw [List.any_eq_true]
  exact ⟨fb, hfb, hcs⟩

/-! ## Bit and comparison plumbing -/

theorem ite_tri_f {c : Prop} [Decidable c] : (if c then Tri.t else Tri.f) = Tri.f ↔ ¬c := by
  by_cases h : c <;> simp [h]

/-- The single low bit of `mask 1`. Used before the reverse-extraction section by
    `neverAllows_of_canAllow_false`. -/
theorem mask_one : mask 1 = 1 := by unfold mask W; norm_num

theorem and_one_ne_zero_of_testBit_zero (x : Nat) (h : x.testBit 0 = true) : band x 1 ≠ 0 := by
  have h2 : x % 2 = 1 := by
    have hh := h
    rw [Nat.testBit_zero] at hh
    exact of_decide_eq_true hh
  rw [band_one, h2]; decide

theorem testBit_zero_of_and_one_ne_zero (x : Nat) (h : band x 1 ≠ 0) : x.testBit 0 = true := by
  have hmod : x % 2 = 1 := by
    have h1 : band x 1 = x % 2 := band_one x
    have h2 : x % 2 ≠ 0 := by rw [← h1]; exact h
    have h3 : x % 2 < 2 := Nat.mod_lt x (by norm_num)
    omega
  rw [Nat.testBit_zero]; simp [hmod]

theorem ncomp_testBit_zero (x : Nat) : (ncomp x).testBit 0 = !(x.testBit 0) :=
  testBit_ncomp_lt (by decide : 0 < W)

theorem permRights_testBit_zero {rs : List Rule} {q : Req}
    (h : (permRights rs q).testBit 0 = true) :
    ∃ r ∈ rs, Permits q r ∧ r.rights.testBit 0 = true := by
  have aux : ∀ (l : List Rule) (acc : Nat),
      (l.foldl (permStep q) acc).testBit 0 = true →
      acc.testBit 0 = true ∨ ∃ r ∈ l, Permits q r ∧ r.rights.testBit 0 = true := by
    intro l
    induction l with
    | nil => intro acc h; exact Or.inl h
    | cons r rest ih =>
        intro acc h
        rw [List.foldl_cons] at h
        by_cases hp : Permits q r
        · rw [show permStep q acc r = bor acc r.rights from by rw [permStep, if_pos hp]] at h
          rcases ih (bor acc r.rights) h with h1 | hw
          · rw [testBit_bor_lt (by decide : 0 < W)] at h1
            rcases Bool.or_eq_true_iff.mp h1 with h1 | h1
            · exact Or.inl h1
            · exact Or.inr ⟨r, List.mem_cons.mpr (Or.inl rfl), hp, h1⟩
          · exact Or.inr ⟨hw.choose, List.mem_cons.mpr (Or.inr hw.choose_spec.1),
              hw.choose_spec.2⟩
        · rw [show permStep q acc r = acc from by rw [permStep, if_neg hp]] at h
          rcases ih acc h with h1 | hw
          · exact Or.inl h1
          · exact Or.inr ⟨hw.choose, List.mem_cons.mpr (Or.inr hw.choose_spec.1),
              hw.choose_spec.2⟩
  rcases aux rs 0 h with h0 | hw
  · simp at h0
  · exact hw

theorem forbRights_testBit_zero_eq_false {rs : List Rule} {q : Req}
    (h : (forbRights rs q).testBit 0 = false) {r : Rule} (hrs : r ∈ rs)
    (hr : Forbids q r) : r.rights.testBit 0 = false := by
  have hsub := forbRights_sub_of_mem rs q r hrs hr
  have hb := congrArg (fun x : Nat => x.testBit 0) hsub
  rw [testBit_band_lt (by decide : 0 < W), testBit_mask_lt (by decide : 0 < W), h] at hb
  simpa using hb.symm

theorem decide_allow_facts {rs : List Rule} {q : Req} {g b : Nat}
    (h : Apeiron.Policy.decide rs q = Verdict.allow g b) :
    gOf rs q = mask q.want ∧ gOf rs q ≠ 0 ∧ band (gOf rs q) (askRights rs q) = 0 := by
  unfold Apeiron.Policy.decide at h
  by_cases h0 : gOf rs q = 0
  · rw [if_pos h0] at h
    by_cases h1 : band (mask q.want) (permRights rs q) = 0
    · rw [if_pos h1] at h; cases h
    · rw [if_neg h1] at h; cases h
  · rw [if_neg h0] at h
    by_cases h2 : band (gOf rs q) (askRights rs q) ≠ 0
    · rw [if_pos h2] at h; cases h
    · rw [if_neg h2] at h
      by_cases h3 : gOf rs q = mask q.want
      · rw [if_pos h3] at h
        exact ⟨h3, h0, not_not.mp h2⟩
      · rw [if_neg h3] at h; cases h

/-! ## Evaluation agreement -/

theorem atom_eval_congr_facts {n : Nat} (q : Req) (ρ : Fin n → ℤ)
    (hfacts : ∀ s (hs : s < n), q.facts s = some (ρ ⟨s, hs⟩).toNat)
    {a : Atom} (hsl : a.slot < n) :
    Atom.eval q.facts a =
      Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none) a := by
  have hf : q.facts a.slot = some (ρ ⟨a.slot, hsl⟩).toNat := hfacts a.slot hsl
  have hg : (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none) a.slot =
      some (ρ ⟨a.slot, hsl⟩).toNat := by simp only [dif_pos hsl]
  unfold Atom.eval
  rw [hf, hg]

/-! ## Negation encoding link -/

theorem negAtomRows_iff {n : Nat} (hn : n ≤ W) {a : Atom} (hsl : a.slot < n)
    (hop : a.op = Op.eq ∨ a.op = Op.le ∨ a.op = Op.ge)
    (ρ : Fin n → ℤ) (hρ : ∀ s, 0 ≤ ρ s ∧ ρ s ≤ M) :
    Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none) a = Tri.f ↔
      ∃ br ∈ negAtomRows a, SatRows ρ br := by
  obtain ⟨op, slot, value⟩ := a
  dsimp only at hsl ⊢
  have hif : slot < W := lt_of_lt_of_le hsl hn
  rcases hop with h | h | h
  · subst h
    have hL : Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none)
        (⟨Op.eq, slot, value⟩ : Atom) =
        (if (ρ ⟨slot, hsl⟩).toNat = value then Tri.t else Tri.f) := by
      simp only [Atom.eval, if_pos hif, dif_pos hsl]
    have hR : negAtomRows (⟨Op.eq, slot, value⟩ : Atom) =
        [[negColRow ⟨slot, hsl⟩ ((value : ℤ) - 1)],
          [colRow ⟨slot, hsl⟩ (-((value : ℤ) + 1))]] := by
      simp only [negAtomRows, dif_pos hsl]
    rw [hL, ite_tri_f, hR]
    constructor
    · intro hne
      have hne' : ρ ⟨slot, hsl⟩ ≠ (value : ℤ) := by
        intro he; exact hne (by rw [he, Int.toNat_natCast])
      by_cases hlt : ρ ⟨slot, hsl⟩ ≤ (value : ℤ) - 1
      · refine ⟨[negColRow ⟨slot, hsl⟩ ((value : ℤ) - 1)], by simp, ?_⟩
        rw [satRows_single, Row.Holds, negColRow_value]; omega
      · refine ⟨[colRow ⟨slot, hsl⟩ (-((value : ℤ) + 1))], by simp, ?_⟩
        rw [satRows_single, Row.Holds, colRow_value]; omega
    · rintro ⟨br, hbr, hsat⟩
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hbr
      intro hc
      have hval : ρ ⟨slot, hsl⟩ = (value : ℤ) := by
        have := congrArg (fun z : Nat => (z : ℤ)) hc
        rwa [Int.toNat_of_nonneg (hρ _).1] at this
      rcases hbr with rfl | rfl
      · rw [satRows_single, Row.Holds, negColRow_value] at hsat; omega
      · rw [satRows_single, Row.Holds, colRow_value] at hsat; omega
  · subst h
    have hL : Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none)
        (⟨Op.le, slot, value⟩ : Atom) =
        (if (ρ ⟨slot, hsl⟩).toNat ≤ value then Tri.t else Tri.f) := by
      simp only [Atom.eval, if_pos hif, dif_pos hsl]
    have hR : negAtomRows (⟨Op.le, slot, value⟩ : Atom) =
        [[colRow ⟨slot, hsl⟩ (-((value : ℤ) + 1))]] := by
      simp only [negAtomRows, dif_pos hsl]
    rw [hL, ite_tri_f, hR]
    constructor
    · intro hnf
      refine ⟨[colRow ⟨slot, hsl⟩ (-((value : ℤ) + 1))], by simp, ?_⟩
      rw [satRows_single, Row.Holds, colRow_value]
      have hle : ¬ ((ρ ⟨slot, hsl⟩) ≤ (value : ℤ)) := fun hle => hnf ((Int.toNat_le).mpr hle)
      omega
    · rintro ⟨br, hbr, hsat⟩
      rw [List.mem_singleton] at hbr; subst hbr
      rw [satRows_single, Row.Holds, colRow_value] at hsat
      intro hc
      have hρle : ρ ⟨slot, hsl⟩ ≤ (value : ℤ) := (Int.toNat_le).mp hc
      omega
  · subst h
    have hL : Atom.eval (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none)
        (⟨Op.ge, slot, value⟩ : Atom) =
        (if value ≤ (ρ ⟨slot, hsl⟩).toNat then Tri.t else Tri.f) := by
      simp only [Atom.eval, if_pos hif, dif_pos hsl]
    have hR : negAtomRows (⟨Op.ge, slot, value⟩ : Atom) =
        [[negColRow ⟨slot, hsl⟩ ((value : ℤ) - 1)]] := by
      simp only [negAtomRows, dif_pos hsl]
    rw [hL, ite_tri_f, hR]
    constructor
    · intro hnf
      refine ⟨[negColRow ⟨slot, hsl⟩ ((value : ℤ) - 1)], by simp, ?_⟩
      rw [satRows_single, Row.Holds, negColRow_value]
      have hle : ¬ ((value : ℤ) ≤ ρ ⟨slot, hsl⟩) :=
        fun hle => hnf ((Int.le_toNat (hρ _).1).mpr hle)
      omega
    · rintro ⟨br, hbr, hsat⟩
      rw [List.mem_singleton] at hbr; subst hbr
      rw [satRows_single, Row.Holds, negColRow_value] at hsat
      intro hc
      have hge : (value : ℤ) ≤ ρ ⟨slot, hsl⟩ := (Int.le_toNat (hρ _).1).mp hc
      omega

/-! ## Stretch: soundness of the analysis against the canonical decision -/

theorem neverAllows_of_canAllow_false
    (rs : List Rule) (action n : Nat) (hn : n ≤ W)
    (hrules : ∀ r ∈ rs, ∀ a ∈ r.guard.getD [],
      a.op = Op.eq ∨ a.op = Op.le ∨ a.op = Op.ge)
    (hslots : ∀ r ∈ rs, ∀ a ∈ r.guard.getD [], a.slot < n)
    (hcan : canAllow rs action n = false)
    (q : Req) (ha : q.action = action) (hw : q.want = 1)
    (ρ : Fin n → ℤ) (hρ : ∀ s, 0 ≤ ρ s ∧ ρ s ≤ M)
    (hfacts : ∀ s (hs : s < n), q.facts s = some (ρ ⟨s, hs⟩).toNat) :
    ∀ g b, Apeiron.Policy.decide rs q ≠ Verdict.allow g b := by
  intro g b hdec
  obtain ⟨hg, _hne0, _hask⟩ := decide_allow_facts hdec
  have hgo : gOf rs q = 1 := by rw [hg, hw, mask_one]
  have hPQ : band (band (mask q.want) (permRights rs q)) (ncomp (forbRights rs q)) = 1 := by
    have hh := hg; unfold gOf at hh; rw [hh, hw, mask_one]
  have hbit : (band (band (mask q.want) (permRights rs q)) (ncomp (forbRights rs q))).testBit 0 = true := by
    rw [hPQ, Nat.testBit_one_zero]
  have hbit' := hbit
  rw [testBit_band_lt (by decide : 0 < W)] at hbit'
  obtain ⟨hPbit, hQbit⟩ := Bool.and_eq_true_iff.mp hbit'
  have hpermbit : (permRights rs q).testBit 0 = true := by
    have hh := hPbit; rw [testBit_band_lt (by decide : 0 < W)] at hh
    exact (Bool.and_eq_true_iff.mp hh).2
  have hforbbit : (forbRights rs q).testBit 0 = false := by
    have hh := hQbit; rw [ncomp_testBit_zero] at hh; simpa using hh
  obtain ⟨p, hprs, hpPerm, hpbit⟩ := permRights_testBit_zero hpermbit
  obtain ⟨hpa, hpe, hpg, _hprw⟩ := hpPerm
  have hp_mem : p ∈ relevantPermits rs action := by
    refine List.mem_filter.mpr ⟨hprs, ?_⟩
    rw [decide_eq_true_eq]
    exact ⟨by rw [hpa, ha], hpe, and_one_ne_zero_of_testBit_zero _ hpbit⟩
  have hp_pb : ∃ pb ∈ guardRows (n := n) p.guard, SatRows ρ pb := by
    cases hgrd : p.guard with
    | none => rw [hgrd, guardEval] at hpg; exact absurd hpg (by decide)
    | some as =>
        rw [hgrd, guardEval] at hpg
        have hall : ∀ a ∈ as, Atom.eval q.facts a = Tri.t :=
          ((guardGo_t_iff q.facts as Tri.t).mp hpg).2
        have henc : ∀ a ∈ as, ∃ bb ∈ atomRows a, SatRows ρ bb := by
          intro a ha
          have hsl : a.slot < n := hslots p hprs a (by simpa [hgrd] using ha)
          have hop := hrules p hprs a (by simpa [hgrd] using ha)
          have heval : Atom.eval
              (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none) a = Tri.t := by
            rw [← atom_eval_congr_facts q ρ hfacts hsl]; exact hall a ha
          exact (atomRows_iff hn hsl hop ρ hρ).mp heval
        exact exists_sat_guardRows ρ as henc
  obtain ⟨pb, hpb_mem, hpb_sat⟩ := hp_pb
  have hfb : ∃ fb ∈ forbidBranches (n := n) rs action, SatRows ρ fb := by
    apply exists_sat_forbidBranches
    intro f hfmem
    obtain ⟨hfrs, hfpred⟩ := List.mem_filter.mp hfmem
    rw [decide_eq_true_eq] at hfpred
    obtain ⟨hfa, hfe, hfrw⟩ := hfpred
    have hguardf : guardEval q.facts f.guard = Tri.f := by
      by_contra hne
      have hforb : Forbids q f := ⟨by rw [hfa, ha], hfe, hne⟩
      have hbitf := forbRights_testBit_zero_eq_false hforbbit hfrs hforb
      have htrue : f.rights.testBit 0 = true := testBit_zero_of_and_one_ne_zero f.rights hfrw
      rw [hbitf] at htrue
      cases htrue
    cases hgrd : f.guard with
    | none => rw [hgrd, guardEval] at hguardf; exact absurd hguardf (by decide)
    | some as =>
        rw [hgrd, guardEval] at hguardf
        have hsome : ∃ a ∈ as, Atom.eval q.facts a = Tri.f := by
          rcases (guardGo_f_iff q.facts as Tri.t).mp hguardf with hx | ⟨hc, _⟩
          · exact hx
          · exact absurd hc (by decide)
        obtain ⟨a, ha_mem, hevalf⟩ := hsome
        have hsl : a.slot < n := hslots f hfrs a (by simpa [hgrd] using ha_mem)
        have hop := hrules f hfrs a (by simpa [hgrd] using ha_mem)
        have heval : Atom.eval
            (fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none) a = Tri.f := by
          rw [← atom_eval_congr_facts q ρ hfacts hsl]; exact hevalf
        exact exists_sat_negGuardRows ρ as
          ⟨a, ha_mem, (negAtomRows_iff hn hsl hop ρ hρ).mp heval⟩
  obtain ⟨fb, hfb_mem, hfb_sat⟩ := hfb
  have hbounds : SatRows ρ (boundRows n) := satRows_boundRows ρ hρ
  have hsat : SatRows ρ (boundRows n ++ pb ++ fb) :=
    (satRows_append ρ (boundRows n ++ pb) fb).mpr
      ⟨(satRows_append ρ (boundRows n) pb).mpr ⟨hbounds, hpb_sat⟩, hfb_sat⟩
  have htrue := canAllow_true_of_branch rs action ρ p hp_mem pb hpb_mem fb hfb_mem hsat
  rw [hcan] at htrue
  cases htrue

/-! ## Reverse extraction (converse witness direction) -/

theorem and_one_eq_one_of_testBit_zero (x : Nat) (h : x.testBit 0 = true) : band x 1 = 1 := by
  rw [band_one]
  have hh := h; rw [Nat.testBit_zero] at hh
  exact of_decide_eq_true hh

theorem one_and_eq_one_of_testBit_zero (x : Nat) (h : x.testBit 0 = true) : band 1 x = 1 := by
  rw [band_comm]; exact and_one_eq_one_of_testBit_zero x h

theorem testBit_zero_eq_false_of_and_one_eq_zero (x : Nat) (h : band x 1 = 0) :
    x.testBit 0 = false := by
  have hmod : x % 2 = 0 := by rw [band_one] at h; exact h
  rw [Nat.testBit_zero]; simp [hmod]

theorem bounds_of_satRows_boundRows {n : Nat} (ρ : Fin n → ℤ)
    (h : SatRows ρ (boundRows n)) : ∀ s, 0 ≤ ρ s ∧ ρ s ≤ M := by
  intro s
  have h1 : Row.Holds ρ (colRow s 0) :=
    h _ (List.mem_flatMap.mpr ⟨s, List.mem_finRange s, by simp⟩)
  have h2 : Row.Holds ρ (negColRow s M) :=
    h _ (List.mem_flatMap.mpr ⟨s, List.mem_finRange s, by simp⟩)
  constructor
  · rw [Row.Holds, colRow_value] at h1; linarith
  · rw [Row.Holds, negColRow_value] at h2; linarith

theorem sat_atoms_of_mem_guardRows {n : Nat} (ρ : Fin n → ℤ) (as : List Atom) :
    ∀ (pb : List (Row n)), pb ∈ guardRows (n := n) (some as) → SatRows ρ pb →
      ∀ a ∈ as, ∃ b ∈ atomRows a, SatRows ρ b := by
  induction as with
  | nil => intro pb hpb _ a ha; exact absurd ha (by simp)
  | cons a rest ih =>
      intro pb hpb hsat
      simp only [guardRows, List.foldr_cons] at hpb
      rw [mem_dnfProduct] at hpb
      obtain ⟨b, hb, y, hy, rfl⟩ := hpb
      rw [satRows_append] at hsat
      intro c hc
      rcases List.mem_cons.mp hc with rfl | hc
      · exact ⟨b, hb, hsat.1⟩
      · exact ih y hy hsat.2 c hc

theorem sat_of_mem_forbidFold {n : Nat} (ρ : Fin n → ℤ) :
    ∀ (l : List Rule) (fb : List (Row n)),
      fb ∈ l.foldr (fun f acc => dnfProduct (negGuardRows (n := n) f.guard) acc) [[]] →
      SatRows ρ fb → ∀ f ∈ l, ∃ b ∈ negGuardRows (n := n) f.guard, SatRows ρ b := by
  intro l
  induction l with
  | nil => intro fb hfb _ f hf; exact absurd hf (by simp)
  | cons f rest ih =>
      intro fb hfb hsat g hg
      simp only [List.foldr_cons] at hfb
      rw [mem_dnfProduct] at hfb
      obtain ⟨b, hb, y, hy, rfl⟩ := hfb
      rw [satRows_append] at hsat
      rcases List.mem_cons.mp hg with rfl | hg
      · exact ⟨b, hb, hsat.1⟩
      · exact ih y hy hsat.2 g hg

theorem sat_of_mem_forbidBranches {n : Nat} (ρ : Fin n → ℤ) (rs : List Rule) (action : Nat)
    (fb : List (Row n)) (hfb : fb ∈ forbidBranches (n := n) rs action) (hsat : SatRows ρ fb) :
    ∀ f ∈ relevantForbids rs action, ∃ b ∈ negGuardRows (n := n) f.guard, SatRows ρ b := by
  unfold forbidBranches at hfb
  exact sat_of_mem_forbidFold ρ (relevantForbids rs action) fb hfb hsat

theorem exists_branch_of_canAllow_true {rs : List Rule} {action n : Nat}
    (h : canAllow rs action n = true) :
    ∃ p ∈ relevantPermits rs action, ∃ pb ∈ guardRows (n := n) p.guard,
      ∃ fb ∈ forbidBranches (n := n) rs action,
        completeSat n (boundRows n ++ pb ++ fb) = true := by
  unfold canAllow at h
  rw [List.any_eq_true] at h
  obtain ⟨p, hp, h⟩ := h
  rw [List.any_eq_true] at h
  obtain ⟨pb, hpb, h⟩ := h
  rw [List.any_eq_true] at h
  obtain ⟨fb, hfb, hcs⟩ := h
  exact ⟨p, hp, pb, hpb, fb, hfb, hcs⟩

theorem decide_allow_of {rs : List Rule} {q : Req}
    (h0 : gOf rs q ≠ 0) (h1 : band (gOf rs q) (askRights rs q) = 0)
    (h2 : gOf rs q = mask q.want) :
    Apeiron.Policy.decide rs q = Verdict.allow (gOf rs q) ((permBy rs q).getD 0) := by
  unfold Apeiron.Policy.decide
  rw [if_neg h0, if_neg (by simp [h1]), if_pos h2]

theorem forbRights_testBit_zero_eq_false_of_guards {rs : List Rule} {q : Req}
    (h : ∀ r ∈ rs, r.action = q.action → r.effect = Effect.forbid → band r.rights 1 ≠ 0 →
      guardEval q.facts r.guard = Tri.f) :
    (forbRights rs q).testBit 0 = false := by
  have aux : ∀ (l : List Rule) (acc : Nat),
      (∀ r ∈ l, r.action = q.action → r.effect = Effect.forbid → band r.rights 1 ≠ 0 →
        guardEval q.facts r.guard = Tri.f) →
      (l.foldl (forbStep q) acc).testBit 0 = acc.testBit 0 := by
    intro l
    induction l with
    | nil => intro acc _; rfl
    | cons r rest ih =>
        intro acc hl
        have hstep : (forbStep q acc r).testBit 0 = acc.testBit 0 := by
          unfold forbStep
          by_cases hf : Forbids q r
          · rw [if_pos hf, testBit_bor_lt (by decide : 0 < W)]
            have hbitf : r.rights.testBit 0 = false := by
              by_cases hrw : band r.rights 1 ≠ 0
              · have hg := hl r (by simp) hf.1 hf.2.1 hrw
                exact absurd hf.2.2 (by rw [hg]; decide)
              · exact testBit_zero_eq_false_of_and_one_eq_zero r.rights (not_not.mp hrw)
            rw [hbitf]; simp
          · rw [if_neg hf]
        rw [List.foldl_cons, ih (forbStep q acc r) (fun r' hr' => hl r' (by simp [hr'])), hstep]
  have hh := aux rs 0 (fun r hr => h r hr)
  unfold forbRights
  simpa using hh

theorem askRights_testBit_zero_eq_false_of_noask {rs : List Rule} {q : Req}
    (h : ∀ r ∈ rs, r.action = q.action → r.effect = Effect.ask → band r.rights 1 = 0) :
    (askRights rs q).testBit 0 = false := by
  have aux : ∀ (l : List Rule) (acc : Nat),
      (∀ r ∈ l, r.action = q.action → r.effect = Effect.ask → band r.rights 1 = 0) →
      (l.foldl (askStep q) acc).testBit 0 = acc.testBit 0 := by
    intro l
    induction l with
    | nil => intro acc _; rfl
    | cons r rest ih =>
        intro acc hl
        have hstep : (askStep q acc r).testBit 0 = acc.testBit 0 := by
          unfold askStep
          by_cases ha : Asks q r
          · rw [if_pos ha, testBit_bor_lt (by decide : 0 < W)]
            have hbit0 : band r.rights 1 = 0 := hl r (by simp) ha.1 ha.2.1
            have hbitf : r.rights.testBit 0 = false :=
              testBit_zero_eq_false_of_and_one_eq_zero r.rights hbit0
            rw [hbitf]; simp
          · rw [if_neg ha]
        rw [List.foldl_cons, ih (askStep q acc r) (fun r' hr' => hl r' (by simp [hr'])), hstep]
  have hh := aux rs 0 (fun r hr => h r hr)
  unfold askRights
  simpa using hh

theorem exists_allow_of_canAllow_true
    (rs : List Rule) (action n : Nat) (hn : n ≤ W)
    (hrules : ∀ r ∈ rs, ∀ a ∈ r.guard.getD [],
      a.op = Op.eq ∨ a.op = Op.le ∨ a.op = Op.ge)
    (hslots : ∀ r ∈ rs, ∀ a ∈ r.guard.getD [], a.slot < n)
    (hnoask : ∀ r ∈ rs, r.action = action → r.effect = Effect.ask → band r.rights 1 = 0)
    (hcan : canAllow rs action n = true) :
    ∃ (g b : Nat) (q : Req), q.action = action ∧ q.want = 1 ∧
      Apeiron.Policy.decide rs q = Verdict.allow g b := by
  obtain ⟨p, hp_mem, pb, hpb, fb, hfb, hcs⟩ := exists_branch_of_canAllow_true hcan
  obtain ⟨ρ, hsat⟩ := (completeSat_eq_true_iff n _).mp hcs
  rw [satRows_append, satRows_append] at hsat
  obtain ⟨⟨hbounds, hpb_sat⟩, hfb_sat⟩ := hsat
  have hρ : ∀ s, 0 ≤ ρ s ∧ ρ s ≤ M := bounds_of_satRows_boundRows ρ hbounds
  obtain ⟨hprs, hppred⟩ := List.mem_filter.mp hp_mem
  rw [decide_eq_true_eq] at hppred
  obtain ⟨hpa, hpe, hprw⟩ := hppred
  have hpbit : p.rights.testBit 0 = true := testBit_zero_of_and_one_ne_zero p.rights hprw
  let ρf : Nat → Option Nat := fun s => if h : s < n then some (ρ ⟨s, h⟩).toNat else none
  have hPerm : Permits (⟨ρf, action, 1⟩ : Req) p := by
    refine ⟨?_, hpe, ?_, ?_⟩
    · show p.action = action; exact hpa
    · cases hguard : p.guard with
      | none => exact absurd hpb (by simp [guardRows, hguard])
      | some as =>
          have hatoms := sat_atoms_of_mem_guardRows ρ as pb (by simpa [hguard] using hpb) hpb_sat
          have hall : ∀ a ∈ as, Atom.eval ρf a = Tri.t := by
            intro a ha
            obtain ⟨b, hb, hbs⟩ := hatoms a ha
            have hsl : a.slot < n := hslots p hprs a (by simpa [hguard] using ha)
            have hop := hrules p hprs a (by simpa [hguard] using ha)
            exact (atomRows_iff hn hsl hop ρ hρ).mpr ⟨b, hb, hbs⟩
          show guardEval ρf (some as) = Tri.t
          rw [guardEval]
          exact (guardGo_t_iff ρf as Tri.t).mpr ⟨rfl, hall⟩
    · show band p.rights 1 ≠ 0; exact hprw
  have hpermbit : (permRights rs (⟨ρf, action, 1⟩ : Req)).testBit 0 = true :=
    sub_testBit (permRights_sub_of_mem rs _ p hprs hPerm) (by decide : 0 < W) hpbit
  have hforbguard : ∀ f ∈ rs, f.action = (⟨ρf, action, 1⟩ : Req).action → f.effect = Effect.forbid →
      band f.rights 1 ≠ 0 → guardEval (⟨ρf, action, 1⟩ : Req).facts f.guard = Tri.f := by
    intro f hfrs hfa hfe hfrw
    have hfmem : f ∈ relevantForbids rs action := by
      refine List.mem_filter.mpr ⟨hfrs, ?_⟩
      rw [decide_eq_true_eq]
      exact ⟨hfa, hfe, hfrw⟩
    obtain ⟨b, hb, hbs⟩ := sat_of_mem_forbidBranches ρ rs action fb hfb hfb_sat f hfmem
    cases hguard : f.guard with
    | none => exact absurd hb (by simp [negGuardRows, hguard])
    | some as =>
        obtain ⟨a, ha, hba⟩ := List.mem_flatMap.mp (by simpa [hguard, negGuardRows] using hb)
        have hsl : a.slot < n := hslots f hfrs a (by simpa [hguard] using ha)
        have hop := hrules f hfrs a (by simpa [hguard] using ha)
        have hevalf : Atom.eval ρf a = Tri.f :=
          (negAtomRows_iff hn hsl hop ρ hρ).mpr ⟨b, hba, hbs⟩
        show guardEval ρf (some as) = Tri.f
        rw [guardEval]
        exact guardGo_f_of_mem ρf as Tri.t ⟨a, ha, hevalf⟩
  have hforbbit : (forbRights rs (⟨ρf, action, 1⟩ : Req)).testBit 0 = false :=
    forbRights_testBit_zero_eq_false_of_guards hforbguard
  have haskbit : (askRights rs (⟨ρf, action, 1⟩ : Req)).testBit 0 = false :=
    askRights_testBit_zero_eq_false_of_noask (fun r hr => hnoask r hr)
  have hgo : gOf rs (⟨ρf, action, 1⟩ : Req) = 1 := by
    unfold gOf
    have hP : band (mask (⟨ρf, action, 1⟩ : Req).want) (permRights rs (⟨ρf, action, 1⟩ : Req)) = 1 := by
      rw [mask_one, one_and_eq_one_of_testBit_zero _ hpermbit]
    have hQbit : (ncomp (forbRights rs (⟨ρf, action, 1⟩ : Req))).testBit 0 = true := by
      rw [ncomp_testBit_zero]; simp [hforbbit]
    rw [hP, band_comm, and_one_eq_one_of_testBit_zero _ hQbit]
  refine ⟨gOf rs (⟨ρf, action, 1⟩ : Req), (permBy rs (⟨ρf, action, 1⟩ : Req)).getD 0,
    ⟨ρf, action, 1⟩, rfl, rfl, ?_⟩
  refine decide_allow_of ?_ ?_ ?_
  · rw [hgo]; decide
  · have haskmod : askRights rs (⟨ρf, action, 1⟩ : Req) % 2 = 0 := by
      have hne : ¬ (askRights rs (⟨ρf, action, 1⟩ : Req) % 2 = 1) :=
        of_decide_eq_false (by simpa [Nat.testBit_zero] using haskbit)
      have hlt : askRights rs (⟨ρf, action, 1⟩ : Req) % 2 < 2 := Nat.mod_lt _ (by norm_num)
      omega
    rw [hgo, band_comm, band_one, haskmod]
  · rw [hgo, mask_one]

end Apeiron.Policy.Analysis
