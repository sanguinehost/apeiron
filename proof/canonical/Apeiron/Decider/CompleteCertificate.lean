import Apeiron.Decider.CompleteDecision

/-!
# Finite residue-refutation certificates

@name Complete Integer-Row Certificates
@grade proved

A leaf names a negative constant row after every column has been eliminated. A split supplies
one child for each residue assignment, in `residueChoices` order. The checker recomputes the
normalized projection and checks the corresponding child; missing and extra children fail.
Certificates contain only natural-number indices and finite lists of certificates.

`synthesizeResidueCertificate` constructs this tree by recursion on the number of columns.
Its leaf search selects a negative row if one exists. The checker validates the resulting index
independently. `residue_certificate_complete` states that a checked tree exists exactly when the
input system has no integer assignment, with no coordinate caps.

## Root

`completeSat_eq_false_iff` supplies the exact integer infeasibility semantics. Its proof uses
`exists_satRows_residue` and `satRows_unitProject_iff`, rooted in the existing integer-row
predicate and `Fragment.unit_elimination_exact`. This file checks finite proof data for the same
projection and introduces no additional semantic premise.
-/

namespace Apeiron.Decider

open Fragment

/-- A serializable refutation tree containing only row indices and ordered child trees.

    ResidueCertificate                     [a finite certificate for residue projection] -/
inductive ResidueCertificate : Type
  /-- At zero columns, name a row whose constant is negative. -/
  | leaf (row : ℕ)
  /-- Supply one child per residue assignment in the prescribed enumeration order. -/
  | split (children : List ResidueCertificate)
  deriving Repr, BEq

mutual

/-- Compare two finite certificate trees by their constructors and data.

    (a b : ResidueCertificate) →           [takes the two certificate trees]
    Decidable (a = b)                      [an executing equality decision] -/
private def residueCertificateDecEq : (a b : ResidueCertificate) → Decidable (a = b)
  | .leaf i, .leaf j =>
      match decEq i j with
      | .isTrue h => .isTrue (congrArg ResidueCertificate.leaf h)
      | .isFalse h => .isFalse (fun heq => h (ResidueCertificate.leaf.inj heq))
  | .split cs, .split ds =>
      match residueCertificateListDecEq cs ds with
      | .isTrue h => .isTrue (congrArg ResidueCertificate.split h)
      | .isFalse h => .isFalse (fun heq => h (ResidueCertificate.split.inj heq))
  | .leaf _, .split _ => .isFalse (fun h => by cases h)
  | .split _, .leaf _ => .isFalse (fun h => by cases h)

/-- Compare ordered lists of finite certificate trees.

    (as bs : List ResidueCertificate) →    [takes the two child lists]
    Decidable (as = bs)                    [an executing equality decision] -/
private def residueCertificateListDecEq :
    (as bs : List ResidueCertificate) → Decidable (as = bs)
  | [], [] => .isTrue rfl
  | a :: as, b :: bs =>
      match residueCertificateDecEq a b with
      | .isFalse h => .isFalse (fun heq => h (List.cons.inj heq).1)
      | .isTrue h =>
          match residueCertificateListDecEq as bs with
          | .isFalse htail => .isFalse (fun heq => htail (List.cons.inj heq).2)
          | .isTrue htail => .isTrue (by cases h; cases htail; rfl)
  | [], _ :: _ => .isFalse (fun h => by cases h)
  | _ :: _, [] => .isFalse (fun h => by cases h)

end

/-- Certificate equality is decidable by the finite-tree comparison. -/
instance : DecidableEq ResidueCertificate := residueCertificateDecEq

/-- Check ordered residue branches in lockstep, requiring exactly one child per branch.

    {α : Type}                            [the type of residue assignments]
    (f : α → ResidueCertificate → Bool)    [the checker for one assignment and its child]
    List α → List ResidueCertificate → Bool [the assignments, children, and acceptance result] -/
def checkResidueBranches {α : Type} (f : α → ResidueCertificate → Bool) :
    List α → List ResidueCertificate → Bool
  | [], [] => true
  | x :: xs, c :: cs => f x c && checkResidueBranches f xs cs
  | _, _ => false

/-- Every prescribed assignment has a checked child when the branch checker accepts.

    {α : Type}                            [the type of residue assignments]
    (f : α → ResidueCertificate → Bool)    [the checker for an assignment and child]
    (xs : List α) (cs : List ResidueCertificate) [the prescribed assignments and supplied children]
    checkResidueBranches f xs cs = true     [all ordered branches were accepted]
    ∀ x ∈ xs, ∃ c ∈ cs, f x c = true     [each assignment has an accepted supplied child] -/
theorem checkResidueBranches_sound {α : Type} (f : α → ResidueCertificate → Bool)
    (xs : List α) (cs : List ResidueCertificate) (h : checkResidueBranches f xs cs = true) :
    ∀ x ∈ xs, ∃ c ∈ cs, f x c = true := by
  induction xs generalizing cs with
  | nil => simp
  | cons x xs ih =>
      cases cs with
      | nil => simp [checkResidueBranches] at h
      | cons c cs =>
          obtain ⟨hhead, htail⟩ := Bool.and_eq_true_iff.mp h
          intro y hy
          rcases List.mem_cons.mp hy with rfl | hy
          · exact ⟨c, by simp, hhead⟩
          · obtain ⟨d, hd, hchecked⟩ := ih cs htail y hy
            exact ⟨d, List.mem_cons_of_mem _ hd, hchecked⟩

/-- Accepted branch lists have exactly the prescribed number of children.

    {α : Type}                            [the type of residue assignments]
    (f : α → ResidueCertificate → Bool)    [the branch checker]
    (xs : List α) (cs : List ResidueCertificate) [the assignments and supplied children]
    checkResidueBranches f xs cs = true     [the checker accepted]
    cs.length = xs.length                  [the two lists have the same length] -/
theorem checkResidueBranches_length {α : Type} (f : α → ResidueCertificate → Bool)
    (xs : List α) (cs : List ResidueCertificate) (h : checkResidueBranches f xs cs = true) :
    cs.length = xs.length := by
  induction xs generalizing cs with
  | nil => cases cs <;> simp_all [checkResidueBranches]
  | cons x xs ih =>
      cases cs with
      | nil => simp [checkResidueBranches] at h
      | cons c cs =>
          have ht := (Bool.and_eq_true_iff.mp h).2
          simpa only [List.length_cons] using congrArg Nat.succ (ih cs ht)

/-- Mapping an accepted child constructor over every prescribed branch passes the checker.

    {α : Type}                            [the type of residue assignments]
    (f : α → ResidueCertificate → Bool)    [the branch checker]
    (xs : List α) (g : α → ResidueCertificate) [the assignments and child constructor]
    (∀ x ∈ xs, f x (g x) = true)         [every constructed child passes]
    checkResidueBranches f xs (xs.map g) = true [the complete ordered list passes] -/
theorem checkResidueBranches_map {α : Type} (f : α → ResidueCertificate → Bool)
    (xs : List α) (g : α → ResidueCertificate) (h : ∀ x ∈ xs, f x (g x) = true) :
    checkResidueBranches f xs (xs.map g) = true := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      exact Bool.and_eq_true_iff.mpr ⟨h x (by simp), ih (fun y hy => h y (by simp [hy]))⟩

/-- Replay a finite refutation tree, recomputing each residue projection and checking its leaf.

    (n : ℕ)                               [the number of input columns]
    List (Row n) → ResidueCertificate → Bool [the input rows, finite tree, and acceptance result] -/
def checkResidueCertificate : (n : ℕ) → List (Row n) → ResidueCertificate → Bool
  | 0, rs, .leaf k => match rs[k]? with
      | some r => decide (r.const < 0)
      | none => false
  | 0, _, .split _ => false
  | _ + 1, _, .leaf _ => false
  | n + 1, rs, .split cs =>
      checkResidueBranches
        (fun s c => checkResidueCertificate n
          (unitProject (rs.map (residueRow (residueModulus rs) s))) c)
        (residueChoices (residueModulus rs) n) cs

/-- An accepted tree forces the complete satisfiability decision to return false.

    (n : ℕ) (rs : List (Row n))            [the number of columns and input rows]
    (cert : ResidueCertificate)             [a finite refutation tree]
    checkResidueCertificate n rs cert = true [the checker accepted it]
    completeSat n rs = false                [the complete decision reports infeasibility] -/
theorem checkResidueCertificate_decision (n : ℕ) (rs : List (Row n))
    (cert : ResidueCertificate) (h : checkResidueCertificate n rs cert = true) :
    completeSat n rs = false := by
  induction n generalizing cert with
  | zero =>
      cases cert with
      | split cs => simp [checkResidueCertificate] at h
      | leaf k =>
          simp only [checkResidueCertificate] at h
          split at h
          · next r hk =>
              have hneg : r.const < 0 := of_decide_eq_true h
              apply Bool.eq_false_iff.mpr
              intro hsat
              have hrow := List.all_eq_true.mp hsat r (List.mem_of_getElem? hk)
              have hnonneg : 0 ≤ r.const := of_decide_eq_true hrow
              omega
          · simp at h
  | succ n ih =>
      cases cert with
      | leaf k => simp [checkResidueCertificate] at h
      | split cs =>
          apply Bool.eq_false_iff.mpr
          intro hsat
          obtain ⟨s, hs, hsat⟩ := List.any_eq_true.mp hsat
          obtain ⟨c, _, hc⟩ := checkResidueBranches_sound _ _ cs h s hs
          have hfalse := ih _ c hc
          rw [hfalse] at hsat
          cases hsat

/-- A checked refutation tree excludes every integer assignment to the input system.

    (n : ℕ) (rs : List (Row n))            [the number of columns and input rows]
    (cert : ResidueCertificate)             [a finite refutation tree]
    checkResidueCertificate n rs cert = true [the checker accepted it]
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs       [no integer assignment satisfies the input] -/
theorem checkResidueCertificate_sound (n : ℕ) (rs : List (Row n))
    (cert : ResidueCertificate) (h : checkResidueCertificate n rs cert = true) :
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs :=
  (completeSat_eq_false_iff n rs).mp (checkResidueCertificate_decision n rs cert h)

/-- Construct a finite refutation candidate by projecting every residue branch.

    (n : ℕ)                               [the number of input columns]
    List (Row n) → ResidueCertificate      [the input rows and finite certificate candidate]

The zero-column leaf searches for a negative row. On a satisfiable system the resulting
candidate fails validation; construction itself makes no acceptance claim. -/
def synthesizeResidueCertificate : (n : ℕ) → List (Row n) → ResidueCertificate
  | 0, rs => .leaf (rs.findIdx fun r => decide (r.const < 0))
  | n + 1, rs => .split ((residueChoices (residueModulus rs) n).map fun s =>
      synthesizeResidueCertificate n
        (unitProject (rs.map (residueRow (residueModulus rs) s))))

/-- Certificate synthesis succeeds for every infeasible integer-row system.

    (n : ℕ) (rs : List (Row n))            [the number of columns and input rows]
    completeSat n rs = false                [the complete procedure reports infeasibility]
    checkResidueCertificate n rs           [the checker validates the generated finite tree]
      (synthesizeResidueCertificate n rs) = true -/
theorem synthesizeResidueCertificate_accepts (n : ℕ) (rs : List (Row n))
    (h : completeSat n rs = false) :
    checkResidueCertificate n rs (synthesizeResidueCertificate n rs) = true := by
  induction n with
  | zero =>
      obtain ⟨r, hr, hneg⟩ := List.all_eq_false.mp h
      have hlt : r.const < 0 := by
        simp only [decide_eq_true_eq, not_le] at hneg
        exact hneg
      have hex : ∃ r ∈ rs, decide (r.const < 0) = true :=
        ⟨r, hr, decide_eq_true hlt⟩
      have hidx := List.findIdx_lt_length_of_exists hex
      simp only [synthesizeResidueCertificate, checkResidueCertificate,
        List.getElem?_eq_getElem hidx]
      exact List.findIdx_getElem (w := hidx)
  | succ n ih =>
      apply checkResidueBranches_map
      intro s hs
      apply ih
      apply Bool.eq_false_iff.mpr
      exact List.any_eq_false.mp h s hs

/-- Checked finite certificates exist exactly for integer-infeasible row systems.

    (n : ℕ) (rs : List (Row n))            [the number of columns and input rows]
    (∃ cert : ResidueCertificate,          [some finite plain-data tree exists]
      checkResidueCertificate n rs cert = true) [and passes independent validation]
    ↔ ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs     [exactly when the input has no integer assignment] -/
theorem residue_certificate_complete (n : ℕ) (rs : List (Row n)) :
    (∃ cert : ResidueCertificate, checkResidueCertificate n rs cert = true) ↔
      ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs := by
  constructor
  · rintro ⟨cert, h⟩
    exact checkResidueCertificate_sound n rs cert h
  · intro h
    exact ⟨synthesizeResidueCertificate n rs,
      synthesizeResidueCertificate_accepts n rs ((completeSat_eq_false_iff n rs).mpr h)⟩

-- The generated tree proves the integer-only contradiction.
#guard checkResidueCertificate 1 [⟨-1, ![2]⟩, ⟨1, ![-2]⟩]
  (synthesizeResidueCertificate 1 [⟨-1, ![2]⟩, ⟨1, ![-2]⟩])
-- Missing, extra, or premature branches are rejected.
#guard !checkResidueCertificate 1 [⟨-1, ![2]⟩, ⟨1, ![-2]⟩] (.split [])
#guard !checkResidueCertificate 1 [⟨-1, ![2]⟩, ⟨1, ![-2]⟩]
  (.split [.leaf 0, .leaf 0])
#guard !checkResidueCertificate 1 [⟨-1, ![2]⟩, ⟨1, ![-2]⟩] (.leaf 0)
#guard !checkResidueCertificate 0 [⟨-1, Fin.elim0⟩] (.leaf 1)
#guard !checkResidueCertificate 1 [⟨-2, ![2]⟩, ⟨2, ![-2]⟩]
  (synthesizeResidueCertificate 1 [⟨-2, ![2]⟩, ⟨2, ![-2]⟩])

end Apeiron.Decider
