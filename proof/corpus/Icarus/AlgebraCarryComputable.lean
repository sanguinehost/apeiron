/-
# A computable algebraic normal form for modular addition, over `F2`

`work/proofs/AlgebraCarryGeneral.lean` proves, at every width, that output bit `j` of `x + y` has
algebraic normal form of degree exactly `j + 1`. Its five ANF objects are `MvPolynomial` values and
are therefore not executable: Mathlib builds the coefficient algebra of a monoid algebra inside a
non-computability section (`Mathlib/Algebra/MonoidAlgebra/Defs.lean:58`, instance
`addAddCommMonoid` at line 159), so the compiler rejects
`MvPolynomial.monomial (Finsupp.single (i, false) 1) 1 : MvPolynomial (Fin k × Bool) (ZMod 2)` as a
definition of code, naming `AddMonoidAlgebra.addAddCommMonoid` in the error. Every definition in this
file is executable code instead, and this file carries no Lean non-computability marker of any kind:
that is the point of the file, and the check is that it compiles without one.

## The representation

A polynomial over `F2` in Boolean variables is a *set of squarefree monomials*: the coefficient of a
monomial is `0` or `1`, so the polynomial is determined by its support and no coefficient map is
needed. A monomial is a set of variables and a polynomial is a set of monomials. Product of monomials
is union of the variable sets (Boolean variables are idempotent, so a repeated factor collapses);
sum of polynomials is symmetric difference; product of polynomials is the set of all products of
monomials. Evaluation multiplies the variable values and adds the monomial values in `ZMod 2`.

| object | type | note |
|---|---|---|
| variable | `Fin k × Bool` | `(i, false)` is bit `i` of the first word, `(i, true)` bit `i` of the second |
| monomial | `Monomial k := Finset (Fin k × Bool)` | squarefree, as a set of variables |
| polynomial | `Anf k := Finset (Monomial k)` | the ANF is its own support |

## What is proved here, and what is not

| claim | statement | form |
|---|---|---|
| the five ANF objects | `xVar`, `yVar`, `carryPoly`, `bitPoly` as `Anf k`; `anfEval`, `evalPoly` | definitions, executable |
| output bit `j` is the ANF value | `anfEval (bitPoly k j) x y = sumBit x y j` | kernel-checked, every pair at `k = 2, 4` |
| the carry identity | `bvXor k x y + (bvAnd k x y <<< 1) = x + y` | kernel-checked, every pair at `k = 2, 4` |
| degree of output bit `j` | `anfDegree (bitPoly k j) = j.val + 1` | proved at every `k`; also kernel-checked at `k = 2, 4` |

`k`, `x`, `y` and `j` are universally quantified in every general headline; no width is instantiated
in a general proof, and no enumeration over pairs of words appears in a general statement. Nothing
here is connected to SHA-2, SHA-3, ARX, a hash round or attack cost.

Grades, and they differ:

- **`#guard` and `#eval` are evaluator evidence.** `#guard e = e'` elaborates `decide (e = e')` and
  evaluates it with `evalExpr` (`Lean/Elab/Tactic/Guard.lean:162`), that is with the compiler, not
  with the kernel; `#eval` likewise runs the compiled definition. Neither is a proof of a
  proposition, and a `#guard` line is not a substitute for a theorem.
- **`decide +kernel` is a kernel check.** The tactic builds the `Decidable` instance of the closed
  proposition and hands it to the kernel (`Lean/Elab/Tactic/Decide.lean`, `doKernel`, which calls
  `mkAuxLemma`); the kernel then decides the proposition. It is used rather than plain `decide`
  because the `decide` tactic's own `whnf` pre-check cannot reduce `Finset`'s quotient recursion
  while the kernel can. Nothing here trusts the compiler for a proof.
- **`#print axioms` rows** at the end of the file are the cone record: every one of them is inside
  `{propext, Classical.choice, Quot.sound}`.

There is no proof gap, no compiler-trusting decision procedure, no raised recursion depth and no
disabled heartbeat limit in this file.
-/
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.BitVec
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Image
import Mathlib.Data.Finset.Union
import Mathlib.Data.Finset.Fold
import Mathlib.Data.Finset.SDiff
import Mathlib.Data.Finset.Card
import Mathlib.Tactic.Positivity

set_option autoImplicit false

namespace AlgebraCarryComputable

/-! ## 1 · The computable representation -/

/-- A squarefree monomial over `F2` in the `2k` input bits, as the set of variables it contains:
`(i, false)` is bit `i` of the first word, `(i, true)` bit `i` of the second word. -/
abbrev Monomial (k : ℕ) := Finset (Fin k × Bool)

/-- A polynomial over `F2` in the `2k` input bits, as the set of its nonzero monomials. This *is* its
algebraic normal form: the coefficient of a monomial is `1` exactly when it is a member. -/
abbrev Anf (k : ℕ) := Finset (Monomial k)

/-- A single bit, read as an element of `F2`. -/
def bitVal (b : Bool) : ZMod 2 := if b then 1 else 0

/-- **Product of two monomials**: the union of their variable sets, because a Boolean variable is
idempotent and a variable occurring in both factors appears once in the product.

Temporal and spatial cost: one `Finset.union`, that is `|m| + |n|` membership tests and one
allocation of `|m ∪ n|` variable slots. -/
def monoMul {k : ℕ} (m n : Monomial k) : Monomial k := m ∪ n

/-- **Sum of two ANFs**: the symmetric difference of their monomial sets, because `1 + 1 = 0` in
`F2`.

Temporal and spatial cost: two `Finset` differences and one union, `2 (|p| + |q|)` membership tests
and one allocation of at most `|p| + |q|` monomial slots. -/
def anfAdd {k : ℕ} (p q : Anf k) : Anf k := (p \ q) ∪ (q \ p)

/-- **Product of two ANFs**: every product of a monomial of `p` and a monomial of `q`. Products
coinciding as variable sets are one monomial, which is the reduction modulo `xᵢ² - xᵢ`.

Temporal and spatial cost: `|p| * |q|` unions, `|p| * |q| * (|m| + |n|)` membership tests and one
allocation of at most `|p| * |q|` monomial slots. -/
def anfMul {k : ℕ} (p q : Anf k) : Anf k := p.biUnion fun m => q.image (monoMul m)

/-- **Value of one monomial** at the assignment given by the two words: the product of the values of
its variables.

Temporal and spatial cost: `|m|` `BitVec.getLsb` reads and `|m|` products in `F2`. -/
def monoEval {k : ℕ} (m : Monomial k) (x y : BitVec k) : ZMod 2 :=
  m.prod fun v => bitVal (if v.2 then y.getLsb v.1 else x.getLsb v.1)

/-- **Value of an ANF** at the assignment given by the two words: the sum of the values of its
monomials in `F2`.

Temporal and spatial cost: `|p|` products of at most `max |m|` factors and `|p|` additions in `F2`,
that is `O(|p| * (1 + max |m|))` operations. -/
def anfEval {k : ℕ} (p : Anf k) (x y : BitVec k) : ZMod 2 :=
  p.sum fun m => monoEval m x y

/-- **Value of a polynomial over `F2`**: the computable counterpart of `evalPoly` in
`work/proofs/AlgebraCarryGeneral.lean:425`, which evaluates a `MvPolynomial` and is therefore not
executable. -/
def evalPoly {k : ℕ} (p : Anf k) (x y : BitVec k) : ZMod 2 := anfEval p x y

/-- Bit `j` of the sum of two `k`-bit words, as an element of `F2`. -/
def sumBit {k : ℕ} (x y : BitVec k) (j : Fin k) : ZMod 2 := bitVal ((x + y).getLsb j)

/-! ## 2 · The five ANF objects

The same structural recursion over the bit index as the sibling file, over `Anf` instead of
`MvPolynomial`. -/

/-- The monomial of `xVar k i`, the singleton variable `xᵢ`. -/
def xVarMono (k : ℕ) (i : Fin k) : Monomial k := ({(i, false)} : Monomial k)

/-- The monomial of `yVar k i`, the singleton variable `yᵢ`. -/
def yVarMono (k : ℕ) (i : Fin k) : Monomial k := ({(i, true)} : Monomial k)

/-- **Bit `i` of the first word**, as an ANF: the single monomial `xᵢ`. -/
def xVar (k : ℕ) (i : Fin k) : Anf k := {xVarMono k i}

/-- **Bit `i` of the second word**, as an ANF: the single monomial `yᵢ`. -/
def yVar (k : ℕ) (i : Fin k) : Anf k := {yVarMono k i}

/-- **The carry into bit `j`, as an ANF.** The carry into bit `0` is the zero polynomial. The carry
into bit `j + 1` is the carry into bit `j` multiplied by the sum of the two input bits, plus the
product of the two input bits: over `F2` that is the full-adder relation `c' = (x + y) * c + x * y`,
that is the majority of `x`, `y`, `c`. Each bit re-expands the previous carry into a product, and
that multiplication is what raises the degree.

Structural recursion on the bit index: `Fin k → Anf k`, the successor case calling at `j`. -/
def carryPoly (k : ℕ) : Fin k → Anf k
  | ⟨0, _⟩ => ∅
  | ⟨j + 1, h⟩ =>
      anfAdd (anfMul (anfAdd (xVar k ⟨j, Nat.lt_of_succ_lt h⟩) (yVar k ⟨j, Nat.lt_of_succ_lt h⟩))
        (carryPoly k ⟨j, Nat.lt_of_succ_lt h⟩))
        (anfMul (xVar k ⟨j, Nat.lt_of_succ_lt h⟩) (yVar k ⟨j, Nat.lt_of_succ_lt h⟩))

/-- **Output bit `j` of `x + y`, as an ANF**: the two input bits plus the carry into that bit. -/
def bitPoly (k : ℕ) (j : Fin k) : Anf k := anfAdd (anfAdd (xVar k j) (yVar k j)) (carryPoly k j)

/-! ### 2.1 · The recursion equations, at every width -/

/-- The carry into bit `0` is the zero polynomial. -/
theorem carryPoly_zero (k : ℕ) (h : 0 < k) : carryPoly k ⟨0, h⟩ = ∅ := by
  simp [carryPoly]

/-- The carry recursion, at every width and every `j + 1 < k`. -/
theorem carryPoly_succ (k : ℕ) (j : ℕ) (hj : j + 1 < k) :
    carryPoly k ⟨j + 1, hj⟩
      = anfAdd
          (anfMul (anfAdd (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (yVar k ⟨j, Nat.lt_of_succ_lt hj⟩))
            (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩))
          (anfMul (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (yVar k ⟨j, Nat.lt_of_succ_lt hj⟩)) := by
  simp [carryPoly]

/-- The carry recursion at the literal index `1`, where the equation compiler's equation lemma is not
usable as a rewrite rule on a numeral index. -/
theorem carryPoly_one_eq (k : ℕ) (h : 1 < k) :
    carryPoly k ⟨1, h⟩
      = anfAdd
          (anfMul (anfAdd (xVar k ⟨0, by omega⟩) (yVar k ⟨0, by omega⟩))
            (carryPoly k ⟨0, by omega⟩))
          (anfMul (xVar k ⟨0, by omega⟩) (yVar k ⟨0, by omega⟩)) := by
  simp [carryPoly]

/-! ## 3 · Laws of the representation

Each is a `Finset` fact, proved at every width `k`, with no enumeration. -/

theorem mem_monoMul {k : ℕ} {m n : Monomial k} {v : Fin k × Bool} :
    v ∈ monoMul m n ↔ v ∈ m ∨ v ∈ n := Finset.mem_union

theorem mem_anfAdd {k : ℕ} {p q : Anf k} {m : Monomial k} :
    m ∈ anfAdd p q ↔ (m ∈ p ∧ m ∉ q) ∨ (m ∈ q ∧ m ∉ p) := by
  simp only [anfAdd, Finset.mem_union, Finset.mem_sdiff]

theorem mem_anfMul {k : ℕ} {p q : Anf k} {m : Monomial k} :
    m ∈ anfMul p q ↔ ∃ m₁ ∈ p, ∃ m₂ ∈ q, monoMul m₁ m₂ = m := by
  simp [anfMul]

theorem mem_xVarMono {k : ℕ} (i : Fin k) {v : Fin k × Bool} :
    v ∈ xVarMono k i ↔ v = (i, false) := by
  simp [xVarMono]

theorem mem_yVarMono {k : ℕ} (i : Fin k) {v : Fin k × Bool} :
    v ∈ yVarMono k i ↔ v = (i, true) := by
  simp [yVarMono]

theorem xVarMono_ne_yVarMono {k : ℕ} (i : Fin k) : xVarMono k i ≠ yVarMono k i := by
  intro h
  have h2 : (i, false) ∈ yVarMono k i := h ▸ Finset.mem_singleton_self _
  rw [mem_yVarMono] at h2
  simp at h2

theorem anfMul_empty {k : ℕ} (p : Anf k) : anfMul p ∅ = ∅ := by
  ext m
  rw [mem_anfMul]
  simp

theorem anfAdd_empty {k : ℕ} (p : Anf k) : anfAdd p ∅ = p := by
  ext m
  rw [mem_anfAdd]
  simp

theorem anfAdd_empty' {k : ℕ} (p : Anf k) : anfAdd ∅ p = p := by
  ext m
  rw [mem_anfAdd]
  simp

/-- **`xVar` evaluates to bit `i` of the first word**, at every width. -/
theorem anfEval_xVar {k : ℕ} (i : Fin k) (x y : BitVec k) :
    anfEval (xVar k i) x y = bitVal (x.getLsb i) := by
  simp [anfEval, monoEval, xVar, xVarMono, bitVal]

/-- **`yVar` evaluates to bit `i` of the second word**, at every width. -/
theorem anfEval_yVar {k : ℕ} (i : Fin k) (x y : BitVec k) :
    anfEval (yVar k i) x y = bitVal (y.getLsb i) := by
  simp [anfEval, monoEval, yVar, yVarMono, bitVal]

/-! ## 4 · Degree

The degree of a polynomial over `F2` is the largest cardinality of one of its monomials. -/

/-- **Degree of an ANF**: the largest number of variables in one of its monomials, `0` for the zero
polynomial.

Temporal and spatial cost: `|p|` cardinalities, `|p|` comparisons and one accumulator; no
allocation. -/
def anfDegree {k : ℕ} (p : Anf k) : Nat := p.fold max 0 fun m => m.card

/-- Every monomial of `p` has degree at most `anfDegree p`. -/
theorem anfDegree_ge {k : ℕ} {p : Anf k} {m : Monomial k} (hm : m ∈ p) : m.card ≤ anfDegree p := by
  have h := (Finset.le_fold_max (f := fun m => m.card) (b := (0 : Nat)) (c := m.card)).2
    (Or.inr ⟨m, hm, le_rfl⟩)
  simpa [anfDegree] using h

/-- The induction behind a degree *upper* bound. -/
theorem anfDegree_le {k : ℕ} {p : Anf k} {d : Nat} (h : ∀ m ∈ p, m.card ≤ d) : anfDegree p ≤ d := by
  induction p using Finset.induction_on with
  | empty => simp [anfDegree]
  | insert a s ha ih =>
      have hins : anfDegree (insert a s) = max a.card (anfDegree s) := by
        rw [anfDegree, Finset.fold_insert ha]
        rfl
      rw [hins]
      exact max_le (h a (Finset.mem_insert_self a s))
        (ih (fun m hm => h m (Finset.mem_insert_of_mem hm)))

@[simp] theorem anfDegree_empty {k : ℕ} : anfDegree (∅ : Anf k) = 0 := rfl

theorem anfDegree_xVar {k : ℕ} (i : Fin k) : anfDegree (xVar k i) = 1 := by
  have hmem : xVarMono k i ∈ xVar k i := Finset.mem_singleton_self _
  have h1 : 1 ≤ anfDegree (xVar k i) := anfDegree_ge hmem
  have h2 : anfDegree (xVar k i) ≤ 1 :=
    anfDegree_le fun m hm => by
      have hme : m = xVarMono k i := by simpa [xVar] using hm
      rw [hme, xVarMono]
      simp
  omega

theorem anfDegree_yVar {k : ℕ} (i : Fin k) : anfDegree (yVar k i) = 1 := by
  have hmem : yVarMono k i ∈ yVar k i := Finset.mem_singleton_self _
  have h1 : 1 ≤ anfDegree (yVar k i) := anfDegree_ge hmem
  have h2 : anfDegree (yVar k i) ≤ 1 :=
    anfDegree_le fun m hm => by
      have hme : m = yVarMono k i := by simpa [yVar] using hm
      rw [hme, yVarMono]
      simp
  omega

/-- **Degree of the sum is at most the maximum of the degrees.** -/
theorem anfDegree_anfAdd_le {k : ℕ} (p q : Anf k) :
    anfDegree (anfAdd p q) ≤ max (anfDegree p) (anfDegree q) :=
  anfDegree_le fun m hm => by
    rw [mem_anfAdd] at hm
    rcases hm with ⟨h, _⟩ | ⟨h, _⟩
    · exact (anfDegree_ge h).trans (le_max_left _ _)
    · exact (anfDegree_ge h).trans (le_max_right _ _)

/-- **Degree of a product is at most the sum of the degrees**: the union of two variable sets has
cardinality at most the sum of the cardinalities. -/
theorem anfDegree_anfMul_le {k : ℕ} (p q : Anf k) :
    anfDegree (anfMul p q) ≤ anfDegree p + anfDegree q :=
  anfDegree_le fun m hm => by
    rcases mem_anfMul.mp hm with ⟨m₁, h₁, m₂, h₂, rfl⟩
    exact (Finset.card_union_le _ _).trans (Nat.add_le_add (anfDegree_ge h₁) (anfDegree_ge h₂))

/-! ### 4.1 · A witness monomial of degree `j + 1`

`bitPoly k j` contains a monomial of `j + 1` variables for every `j < k`: `y₀` for `j = 0`, and
`y₀ x₀ x₁ ⋯ x_{j-1}` for `j ≥ 1`, which picks `x` at every position but the lowest, where it picks
`y`. A degree upper bound alone would not give the theorem; this monomial is the lower bound. -/

/-- **The witness monomial of `bitPoly k j`**: `y₀` for `j = 0`, and
`{(0, true)} ∪ {(i, false) : i < j}` for `j > 0`. -/
def witMono : (k : ℕ) → (j : ℕ) → j < k → Monomial k
  | k, 0, h => { ((⟨0, h⟩ : Fin k), true) }
  | k, j + 1, h =>
      insert ((⟨j, Nat.lt_of_succ_lt h⟩ : Fin k), false) (witMono k j (Nat.lt_of_succ_lt h))

/-- The successor case of `witMono` at a variable index, where the equation compiler's equation
lemma cannot be used as a rewrite rule. -/
theorem witMono_succ (k : ℕ) (j : ℕ) (h : j + 1 < k) :
    witMono k (j + 1) h = insert ((⟨j, Nat.lt_of_succ_lt h⟩ : Fin k), false)
      (witMono k j (Nat.lt_of_succ_lt h)) := rfl

/-- **Every element of the witness monomial is `y₀` or an `x` variable below `j`**: the only `y`
variable it contains is `y₀`. -/
theorem mem_witMono {k : ℕ} {j : ℕ} (hj : j < k) {v : Fin k × Bool} (hv : v ∈ witMono k j hj) :
    (v.2 = true ∧ v.1.val = 0) ∨ (v.2 = false ∧ v.1.val < j) := by
  induction j with
  | zero =>
      rw [witMono, Finset.mem_singleton] at hv
      refine Or.inl ⟨congrArg Prod.snd hv, ?_⟩
      simpa using congrArg (fun w : Fin k × Bool => w.1.val) hv
  | succ j ih =>
      have hj' : j < k := by omega
      rw [witMono, Finset.mem_insert] at hv
      rcases hv with h | hv
      · rcases h with ⟨v', rfl⟩
        exact Or.inr ⟨rfl, by simp⟩
      · rcases ih hj' hv with ⟨h1, h2⟩ | ⟨h1, h2⟩
        · exact Or.inl ⟨h1, h2⟩
        · exact Or.inr ⟨h1, by omega⟩

/-- The variable at index `j` is not in the witness monomial of bit `j`. -/
theorem not_mem_witMono {k : ℕ} {j : ℕ} (hj : j < k) :
    ((⟨j, hj⟩ : Fin k), false) ∉ witMono k j hj := by
  intro hv
  rcases mem_witMono hj hv with ⟨h, _⟩ | ⟨h, h2⟩
  · simp at h
  · simp at h2

/-- **The witness monomial of bit `j` has degree `j + 1`.** -/
theorem witMono_card {k : ℕ} {j : ℕ} (hj : j < k) : (witMono k j hj).card = j + 1 := by
  induction j with
  | zero => simp [witMono]
  | succ j ih =>
      have hj' : j < k := by omega
      rw [witMono, Finset.card_insert_of_notMem (not_mem_witMono hj'), ih hj']

/-! ### 4.2 · The degree theorem at every width

The sibling file proves `totalDegree_bitPoly : (bitPoly k j).totalDegree = j.val + 1` at every width
by induction on the bit index, using the exact additivity of the degree of a product. Here a product
of overlapping monomials can lose degree, so the upper bound uses only `|m ∪ n| ≤ |m| + |n|` and the
lower bound uses the witness monomial of §4.1. -/

/-- **The carry into bit `j` has degree at most `j + 1`**, at every width `k > j`. Induction on the
bit index: multiplying by the linear form `xⱼ + yⱼ` adds at most one to the degree, and `xⱼ yⱼ` has
degree two, at most `j + 1`. -/
theorem anfDegree_carryPoly_le (k : ℕ) : ∀ (j : ℕ) (hj : j < k),
    anfDegree (carryPoly k ⟨j, hj⟩) ≤ j + 1 := by
  intro j hj
  induction j using Nat.strong_induction_on with
  | h j ih =>
    rcases Nat.eq_zero_or_pos j with hj0 | hj0
    · subst hj0
      simp only [carryPoly_zero, anfDegree_empty]
      omega
    · obtain ⟨m, rfl, hmK⟩ : ∃ m, j = m + 1 ∧ m < k := ⟨j - 1, by omega, by omega⟩
      have hmj : m < k := hmK
      simp only [carryPoly_succ]
      have h1 := anfDegree_anfMul_le (anfAdd (xVar k ⟨m, hmj⟩) (yVar k ⟨m, hmj⟩))
        (carryPoly k ⟨m, hmj⟩)
      have h2 := anfDegree_anfAdd_le (xVar k ⟨m, hmj⟩) (yVar k ⟨m, hmj⟩)
      have h3 := ih m (by omega) hmj
      have h4 := anfDegree_anfMul_le (xVar k ⟨m, hmj⟩) (yVar k ⟨m, hmj⟩)
      have h5 := anfDegree_anfAdd_le (anfMul (anfAdd (xVar k ⟨m, hmj⟩) (yVar k ⟨m, hmj⟩))
        (carryPoly k ⟨m, hmj⟩)) (anfMul (xVar k ⟨m, hmj⟩) (yVar k ⟨m, hmj⟩))
      rw [anfDegree_xVar, anfDegree_yVar] at h2 h4
      omega

/-- **The witness monomial of bit `j` lies in the carry into bit `j`**, for `1 ≤ j < k`, by induction
on the bit index: for `j = 1` it is the product `x₀ y₀`, and for `j + 1` it is `xⱼ` times the witness
of `j`, which is one of the products of the recursion. -/
theorem mem_witMono_carryPoly {k : ℕ} {j : ℕ} (hj : 1 ≤ j) (hjk : j < k) :
    witMono k j hjk ∈ carryPoly k ⟨j, hjk⟩ := by
  induction j using Nat.strong_induction_on with
  | h j ih =>
    rcases Nat.eq_zero_or_pos j with hj0 | hj0
    · omega
    · obtain ⟨m, rfl, hmK⟩ : ∃ m, j = m + 1 ∧ m < k := ⟨j - 1, by omega, by omega⟩
      have hmk : m < k := hmK
      rcases Nat.eq_zero_or_pos m with hmz | hmz
      · -- `j = 1`: the witness is `x₀ y₀`, the second product of the recursion
        subst hmz
        have hk0 : 0 < k := by omega
        have h0 : carryPoly k ⟨0, hk0⟩ = ∅ := carryPoly_zero k hk0
        simp only [witMono, carryPoly_succ, h0]
        rw [mem_anfAdd]
        refine Or.inr ⟨?_, ?_⟩
        · rw [mem_anfMul]
          refine ⟨xVarMono k ⟨0, hk0⟩, Finset.mem_singleton_self _,
            yVarMono k ⟨0, hk0⟩, Finset.mem_singleton_self _, ?_⟩
          show monoMul (xVarMono k ⟨0, hk0⟩) (yVarMono k ⟨0, hk0⟩)
            = insert ((⟨0, hk0⟩ : Fin k), false) {((⟨0, hk0⟩ : Fin k), true)}
          rw [monoMul, xVarMono, yVarMono]
          refine (Finset.ext fun v => ?_).symm
          simp only [Finset.mem_insert, Finset.mem_union, Finset.mem_singleton]
        · intro hcon
          rw [mem_anfMul] at hcon
          rcases hcon with ⟨_, _, m₂, hm₂, _⟩
          simp at hm₂
      · -- `j = m + 1` with `1 ≤ m`: the witness is `xₘ` times the witness of `m`
        simp only [witMono, carryPoly_succ]
        rw [mem_anfAdd]
        refine Or.inl ⟨?_, ?_⟩
        · rw [mem_anfMul]
          refine ⟨xVarMono k ⟨m, hmk⟩,
            (mem_anfAdd).2 (Or.inl ⟨Finset.mem_singleton_self _,
              by simp [xVarMono, yVar, yVarMono]⟩),
            witMono k m hmk, ih m (by omega) (by omega) hmk, ?_⟩
          show monoMul (xVarMono k ⟨m, hmk⟩) (witMono k m hmk)
            = insert ((⟨m, hmk⟩ : Fin k), false) (witMono k m hmk)
          rw [monoMul, xVarMono]
          refine (Finset.ext fun v => ?_).symm
          simp only [Finset.mem_insert, Finset.mem_union, Finset.mem_singleton]
        · intro hcon
          rw [mem_anfMul] at hcon
          rcases hcon with ⟨m₁, hm₁, m₂, hm₂, heq⟩
          simp only [xVar, Finset.mem_singleton] at hm₁
          simp only [yVar, Finset.mem_singleton] at hm₂
          subst hm₁
          subst hm₂
          have hn : ((⟨m, hmk⟩ : Fin k), true) ∉ witMono k m hmk := by
            intro hc
            rcases mem_witMono hmk hc with ⟨_, hv⟩ | ⟨h, _⟩
            · exact absurd (by simpa using hv) (Nat.ne_of_gt hmz)
            · simp at h
          have hn' : ((⟨m, hmk⟩ : Fin k), true)
              ∉ insert ((⟨m, hmk⟩ : Fin k), false) (witMono k m hmk) := by
            intro hc
            rcases Finset.mem_insert.mp hc with hc | hc
            · exact absurd hc (by simp)
            · exact hn hc
          have hymem : ((⟨m, hmk⟩ : Fin k), true)
              ∈ monoMul (xVarMono k ⟨m, hmk⟩) (yVarMono k ⟨m, hmk⟩) := by
            refine mem_monoMul.mpr (Or.inr ?_)
            show ((⟨m, hmk⟩ : Fin k), true) ∈ ({(⟨m, hmk⟩, true)} : Monomial k)
            exact Finset.mem_singleton.mpr rfl
          exact absurd (heq ▸ hymem) hn'

/-- For a successor bit, the witness monomial is neither input bit: it has `j + 1 ≥ 2` variables
while a single-variable monomial has one. -/
theorem witMono_ne_xVarMono {k : ℕ} {j : ℕ} (hj : 0 < j) (hjk : j < k) :
    witMono k j hjk ≠ xVarMono k ⟨j, hjk⟩ := by
  intro h
  have h1 := witMono_card hjk
  have h2 : (xVarMono k ⟨j, hjk⟩).card = 1 := by simp [xVarMono]
  rw [h, h2] at h1
  omega

/-- For a successor bit, the witness monomial is not `yⱼ` either, for the same reason. -/
theorem witMono_ne_yVarMono {k : ℕ} {j : ℕ} (hj : 0 < j) (hjk : j < k) :
    witMono k j hjk ≠ yVarMono k ⟨j, hjk⟩ := by
  intro h
  have h1 := witMono_card hjk
  have h2 : (yVarMono k ⟨j, hjk⟩).card = 1 := by simp [yVarMono]
  rw [h, h2] at h1
  omega

/-- For a successor bit, the witness monomial is in neither the `xⱼ` summand nor the `yⱼ` summand of
the output bit. -/
theorem witMono_not_mem_anfAdd {k : ℕ} {j : ℕ} (hj : 0 < j) (hjk : j < k) :
    witMono k j hjk ∉ anfAdd (xVar k ⟨j, hjk⟩) (yVar k ⟨j, hjk⟩) := by
  rw [mem_anfAdd, xVar, yVar]
  intro hcon
  rcases hcon with ⟨h, _⟩ | ⟨h, _⟩
  · have hx : witMono k j hjk = xVarMono k ⟨j, hjk⟩ := Finset.mem_singleton.mp h
    exact absurd hx (witMono_ne_xVarMono hj hjk)
  · have hy : witMono k j hjk = yVarMono k ⟨j, hjk⟩ := Finset.mem_singleton.mp h
    exact absurd hy (witMono_ne_yVarMono hj hjk)

/-- **The witness monomial of bit `j` lies in `bitPoly k j`**, at every width. For `j = 0` the
witness is `y₀`, the second input bit; for `j > 0` it is in neither input bit, so it must come from
the carry, and it lies in the carry by `mem_witMono_carryPoly`. -/
theorem mem_witMono_bitPoly (k : ℕ) (j : Fin k) : witMono k j.val j.isLt ∈ bitPoly k j := by
  rcases Nat.eq_zero_or_pos j.val with hj0 | hj0
  · rw [show j = ⟨0, by omega⟩ from Fin.ext hj0, witMono, bitPoly, carryPoly_zero, mem_anfAdd]
    refine Or.inl ⟨?_, by simp⟩
    rw [mem_anfAdd]
    exact Or.inr ⟨Finset.mem_singleton_self _, by simp [xVar, xVarMono]⟩
  · rw [bitPoly, mem_anfAdd]
    exact Or.inr ⟨mem_witMono_carryPoly (by omega) j.isLt,
      witMono_not_mem_anfAdd (by omega) j.isLt⟩

/-- **THE DEGREE THEOREM, at every width.** The ANF of output bit `j` of the sum of two `k`-bit words
has degree exactly `j + 1`, for every `k` and every `j < k`. The upper bound is the induction of
§4.2, the lower bound the witness monomial of §4.1. -/
theorem anfDegree_bitPoly (k : ℕ) (j : Fin k) : anfDegree (bitPoly k j) = j.val + 1 := by
  have hupper : anfDegree (bitPoly k j) ≤ j.val + 1 := by
    have h1 := anfDegree_anfAdd_le (anfAdd (xVar k j) (yVar k j)) (carryPoly k j)
    have h2 := anfDegree_anfAdd_le (xVar k j) (yVar k j)
    have h3 := anfDegree_carryPoly_le k j.val j.isLt
    rw [anfDegree_xVar, anfDegree_yVar] at h2
    have h2' : anfDegree (anfAdd (xVar k j) (yVar k j)) ≤ 1 := by simpa using h2
    unfold bitPoly
    calc anfDegree (anfAdd (anfAdd (xVar k j) (yVar k j)) (carryPoly k j))
        ≤ max (anfDegree (anfAdd (xVar k j) (yVar k j))) (anfDegree (carryPoly k j)) := h1
      _ ≤ max 1 (j.val + 1) :=
        max_le (h2'.trans (le_max_left _ _)) (h3.trans (by simp))
      _ = j.val + 1 := by simp
  have hlower : j.val + 1 ≤ anfDegree (bitPoly k j) := by
    have h1 : (witMono k j.val j.isLt).card ≤ anfDegree (bitPoly k j) :=
      anfDegree_ge (mem_witMono_bitPoly k j)
    rw [witMono_card (k := k) (j := j.val) j.isLt] at h1
    omega
  omega

/-! ## 5 · Evaluator evidence

`#eval` and `#guard` run the compiled definitions: they are evaluator evidence, not proofs. The
`decide +kernel` lines that accompany them are kernel checks of the same equalities. -/

/-! ### 5.1 · Output bit `j` is bit `j` of the sum, at `k = 2` and `k = 4` -/

-- width two, carry-critical: 1 + 1 = 2 sets bit 1 by the carry and clears bit 0
#guard anfEval (bitPoly (2 : ℕ) (1 : Fin 2)) 1 1 = sumBit 1 1 (1 : Fin 2)
#guard anfEval (bitPoly (2 : ℕ) (0 : Fin 2)) 1 1 = sumBit 1 1 (0 : Fin 2)
#guard anfEval (bitPoly (2 : ℕ) (1 : Fin 2)) 3 3 = sumBit 3 3 (1 : Fin 2)
#guard anfEval (bitPoly (2 : ℕ) (0 : Fin 2)) 3 3 = sumBit 3 3 (0 : Fin 2)
#guard anfEval (bitPoly (2 : ℕ) (1 : Fin 2)) 2 3 = sumBit 2 3 (1 : Fin 2)
#guard anfEval (bitPoly (2 : ℕ) (0 : Fin 2)) 2 3 = sumBit 2 3 (0 : Fin 2)

-- width four: 1 + 1 carries into bit 1; 15 + 1 = 16 clears every bit; 15 + 15 = 30 sets bits 1 to 3
#guard anfEval (bitPoly (4 : ℕ) (3 : Fin 4)) 1 1 = sumBit 1 1 (3 : Fin 4)
#guard anfEval (bitPoly (4 : ℕ) (1 : Fin 4)) 1 1 = sumBit 1 1 (1 : Fin 4)
#guard anfEval (bitPoly (4 : ℕ) (0 : Fin 4)) 1 1 = sumBit 1 1 (0 : Fin 4)
#guard anfEval (bitPoly (4 : ℕ) (3 : Fin 4)) 15 1 = sumBit 15 1 (3 : Fin 4)
#guard anfEval (bitPoly (4 : ℕ) (2 : Fin 4)) 15 1 = sumBit 15 1 (2 : Fin 4)
#guard anfEval (bitPoly (4 : ℕ) (3 : Fin 4)) 15 15 = sumBit 15 15 (3 : Fin 4)
#guard anfEval (bitPoly (4 : ℕ) (1 : Fin 4)) 9 6 = sumBit 9 6 (1 : Fin 4)

/-! ### 5.2 · The content of the carry cell

The sibling file proves the second slot of the carry cell is `2 * denoteBV (x &&& y)`
(`AlgebraCarryGeneral.lean:191`, `carryCell_snd`): the deviation of the sum from the exclusive-or sum
is the conjunction of the two words, shifted up by one bit. In the ANF the conjunction is the carry
chain, and the carry into bit `1` is exactly the conjunction of the two low bits as the single
monomial `x₀ y₀`. -/

#guard anfEval (carryPoly (2 : ℕ) (1 : Fin 2)) 1 1 = 1
#guard anfEval (carryPoly (2 : ℕ) (1 : Fin 2)) 3 3 = 1
#guard anfEval (carryPoly (2 : ℕ) (1 : Fin 2)) 3 2 = 0
#guard anfEval (carryPoly (4 : ℕ) (1 : Fin 4)) 15 15 = 1
#guard anfEval (carryPoly (4 : ℕ) (1 : Fin 4)) 15 2 = 0
#guard anfEval (anfMul (xVar (2 : ℕ) (0 : Fin 2)) (yVar (2 : ℕ) (0 : Fin 2))) 3 3 = 1

/-- **The carry into bit `1` is the single monomial `x₀ y₀`**, at every width `k > 1`: the conjunction
of the two low bits, which is the content of the second slot of the carry cell at the lowest bit. -/
theorem carryPoly_one (k : ℕ) (h : 1 < k) :
    carryPoly k ⟨1, h⟩ = anfMul (xVar k ⟨0, by omega⟩) (yVar k ⟨0, by omega⟩) := by
  rw [carryPoly_one_eq, carryPoly_zero, anfMul_empty, anfAdd_empty']

/-! ### 5.3 · Kernel checks of representative instances, at `k = 2` and `k = 4`

The same instances as §5.1, decided by the kernel rather than by the compiler. -/

example : anfEval (bitPoly (2 : ℕ) (1 : Fin 2)) 1 1 = sumBit 1 1 (1 : Fin 2) := by decide +kernel

example : anfEval (bitPoly (2 : ℕ) (0 : Fin 2)) 3 3 = sumBit 3 3 (0 : Fin 2) := by decide +kernel

example : anfEval (bitPoly (4 : ℕ) (3 : Fin 4)) 1 1 = sumBit 1 1 (3 : Fin 4) := by decide +kernel

example : anfEval (bitPoly (4 : ℕ) (3 : Fin 4)) 15 1 = sumBit 15 1 (3 : Fin 4) := by decide +kernel

example : anfEval (carryPoly (2 : ℕ) (1 : Fin 2)) 1 1 = 1 := by decide +kernel

example : anfEval (anfMul (xVar (2 : ℕ) (0 : Fin 2)) (yVar (2 : ℕ) (0 : Fin 2))) 3 3 = 1 := by
  decide +kernel

/-! ### 5.4 · `#eval`: the evaluator, on the same instances -/

#eval anfEval (bitPoly (4 : ℕ) (3 : Fin 4)) 15 15
#eval anfEval (bitPoly (4 : ℕ) (1 : Fin 4)) 9 6
#eval anfDegree (bitPoly (4 : ℕ) (3 : Fin 4))
#eval anfDegree (carryPoly (4 : ℕ) (2 : Fin 4))
#eval (anfEval (carryPoly (4 : ℕ) (2 : Fin 4)) 15 15, anfEval (carryPoly (4 : ℕ) (2 : Fin 4)) 15 2)

/-! ## 6 · The carry identity and the sum, executable at fixed widths

`bvXor` and `bvAnd` are the bitwise operations of §5.2 as functions on words, and `anfSumBV` rebuilds
a word from the per-bit ANFs. The statements below are the sibling's carry identity
`x + y = (x ^^^ y) + ((x &&& y) <<< 1)` and its `bitPoly` identification, decided by the kernel at
widths two and four: every pair of words, every output bit. -/

/-- Bitwise exclusive-or of two `k`-bit words, assembled from the bits.

Temporal and spatial cost: `k` bit reads, `k` additions, one allocation of `k` bits. -/
def bvXor (k : ℕ) (x y : BitVec k) : BitVec k :=
  ⟨((Finset.univ.filter fun j : Fin k => x.getLsb j != y.getLsb j).sum fun j => 2 ^ j.val) % 2 ^ k,
    Nat.mod_lt _ (by positivity)⟩

/-- Bitwise conjunction of two `k`-bit words, assembled from the bits.

Temporal and spatial cost: `k` bit reads, `k` additions, one allocation of `k` bits. -/
def bvAnd (k : ℕ) (x y : BitVec k) : BitVec k :=
  ⟨((Finset.univ.filter fun j : Fin k => x.getLsb j && y.getLsb j).sum fun j => 2 ^ j.val) % 2 ^ k,
    Nat.mod_lt _ (by positivity)⟩

/-- The word rebuilt from the per-bit ANFs `bitPoly k j`.

Temporal and spatial cost: `k` ANF evaluations of at most `2^(k-1) + 1` monomials, `k` additions, one
allocation of `k` bits. -/
def anfSumBV (k : ℕ) (x y : BitVec k) : BitVec k :=
  ⟨((Finset.univ.filter fun j : Fin k => anfEval (bitPoly k j) x y == 1).sum fun j => 2 ^ j.val)
      % 2 ^ k, Nat.mod_lt _ (by positivity)⟩

/-! ### 6.1 · Width two: every pair of words -/

theorem carry_identity_two : ∀ (x y : BitVec 2), bvXor 2 x y + (bvAnd 2 x y <<< 1) = x + y := by
  decide +kernel

theorem bitPoly_eval_two : ∀ (x y : BitVec 2) (j : Fin 2),
    anfEval (bitPoly 2 j) x y = sumBit x y j := by
  decide +kernel

theorem anfSumBV_two : ∀ (x y : BitVec 2), anfSumBV 2 x y = x + y := by
  decide +kernel

/-! ### 6.2 · Width four: every pair of words -/

theorem carry_identity_four : ∀ (x y : BitVec 4), bvXor 4 x y + (bvAnd 4 x y <<< 1) = x + y := by
  decide +kernel

theorem bitPoly_eval_four : ∀ (x y : BitVec 4) (j : Fin 4),
    anfEval (bitPoly 4 j) x y = sumBit x y j := by
  decide +kernel

theorem anfSumBV_four : ∀ (x y : BitVec 4), anfSumBV 4 x y = x + y := by
  decide +kernel

/-! ## 7 · The degree of `bitPoly`, decided at `k = 2` and `k = 4`

§4.2 proves this at every width. These two lines check it by kernel reduction at the two widths where
the sibling file enumerated, every output bit: the induction and the enumeration agree. -/

theorem anfDegree_bitPoly_two : ∀ j : Fin 2, anfDegree (bitPoly 2 j) = j.val + 1 := by
  decide +kernel

theorem anfDegree_bitPoly_four : ∀ j : Fin 4, anfDegree (bitPoly 4 j) = j.val + 1 := by
  decide +kernel

/-! ## 8 · The bridge to the `MvPolynomial` formulation

`work/proofs/AlgebraCarryGeneral.lean` formulates the same five objects over
`AddPoly k := MvPolynomial (Fin k × Bool) (ZMod 2)` and marks `xVar`, `yVar`, `carryPoly`, `bitPoly`
and `evalPoly` as non-executable. A function from `Anf k` to `AddPoly k` cannot be defined in this
file at all: Mathlib declares the coefficient algebra of a monoid algebra inside a
non-computability section, so `MvPolynomial.monomial (Finsupp.single (i, false) 1) 1` is rejected by
the compiler as a definition of code, and every `AddPoly` value reached from it is equally rejected.
The correspondence is therefore a *statement*, not a function, and it belongs in a file that is
allowed to hold non-executable definitions. The obligations are exactly that statement, one per
object, in the notation of the sibling file (`xVar` at line 252, `yVar` at 256, `carryPoly` at 264,
`bitPoly` at 272, `evalPoly` at 425):

```text
bridge_anfToAddPoly : Anf k → AlgebraCarryGeneral.AddPoly k
  satisfying, for every (x y : BitVec k) and every (j : Fin k),
  (1) AlgebraCarryGeneral.evalPoly (bridge_anfToAddPoly (xVar k j)) x y = anfEval (xVar k j) x y
  (2) AlgebraCarryGeneral.evalPoly (bridge_anfToAddPoly (yVar k j)) x y = anfEval (yVar k j) x y
  (3) AlgebraCarryGeneral.evalPoly (bridge_anfToAddPoly (carryPoly k j)) x y
      = anfEval (carryPoly k j) x y
  (4) AlgebraCarryGeneral.evalPoly (bridge_anfToAddPoly (bitPoly k j)) x y
      = anfEval (bitPoly k j) x y
```

(1) and (2) discharge on `MvPolynomial.eval_monomial`; (3) and (4) follow from (1), (2), the two
recursion equations `carryPoly_zero` and `carryPoly_succ`, and the two evaluation laws of the sibling
file (`evalPoly_add`, `evalPoly_mul`), by induction on `j`. This file proves the ANF side of (1) and
(2) in `anfEval_xVar` and `anfEval_yVar`, and the ANF side of (3) and (4) for `k = 2, 4` by kernel
reduction in §6.

The consequence, once the bridge is discharged: the sibling's degree theorem `totalDegree_bitPoly`
transfers to this file as `anfDegree_bitPoly`, already proved here at every width. The bridge
therefore carries no new mathematics at width `k`; it carries the identification of two
representations of one algebraic normal form. -/

/-! ## 9 · The cone

Every headline above is checked inside `{propext, Classical.choice, Quot.sound}`. -/

#print axioms mem_monoMul
#print axioms mem_anfAdd
#print axioms mem_anfMul
#print axioms anfEval_xVar
#print axioms anfEval_yVar
#print axioms anfDegree_ge
#print axioms anfDegree_le
#print axioms anfDegree_anfAdd_le
#print axioms anfDegree_anfMul_le
#print axioms carryPoly_zero
#print axioms carryPoly_succ
#print axioms carryPoly_one
#print axioms mem_witMono
#print axioms not_mem_witMono
#print axioms witMono_card
#print axioms anfDegree_carryPoly_le
#print axioms mem_witMono_carryPoly
#print axioms mem_witMono_bitPoly
#print axioms witMono_ne_xVarMono
#print axioms witMono_ne_yVarMono
#print axioms witMono_not_mem_anfAdd
#print axioms anfDegree_bitPoly
#print axioms carry_identity_two
#print axioms bitPoly_eval_two
#print axioms anfSumBV_two
#print axioms carry_identity_four
#print axioms bitPoly_eval_four
#print axioms anfSumBV_four
#print axioms anfDegree_bitPoly_two
#print axioms anfDegree_bitPoly_four

end AlgebraCarryComputable