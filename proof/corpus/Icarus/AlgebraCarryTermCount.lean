/-
# AR-62 — the exact ANF term count of modular addition, at every width

`work/proofs/AlgebraCarryComputable.lean` (AR-29, frozen at sha256
`99b82870d78de7144181b23cf36d03be0da7bbbb81877a4ef4695698c3f88689`) proves the degree of
`bitPoly k j` is exactly `j + 1` at every width. This file closes the count that pairs with that
degree: `(bitPoly k j).card = 2 ^ j.val + 1` for every `k` and every `j : Fin k`. The carrier
`Anf k` *is* the support of the `F2` polynomial — a `Finset` of squarefree monomials — so `card` is
the number of nonzero monomials, i.e. the term count.

## The recurrence

With `C_j = carryPoly k ⟨j,_⟩`, `L_j = xVar k j + yVar k j` and `Q_j = x_j * y_j`,

    C_{j+1} = L_j * C_j + Q_j,   and since `L_j = {x_j} ⊎ {y_j}` with `x_j`, `y_j` fresh to `C_j`,

    C_{j+1} = (x_j * C_j) ⊎ (y_j * C_j) ⊎ {x_j y_j},   card C_{j+1} = 2 * card C_j + 1.

`carryPoly_succ_support` states the three pairwise disjoint classes: the `x_j`-lift of every old
monomial, the `y_j`-lift, and the one fresh monomial `x_j y_j` (whose variable set is
`monoMul (xVarMono j) (yVarMono j)`). `carryPoly_succ_card` is the count recurrence that carries
the induction; `carryPoly_card` gives `card C_j + 1 = 2^j`, and `bitPoly_succ_eq` expands output
bit `j + 1` through the same recurrence. The two linear monomials of bit `j` are disjoint from the
carry support, so `bitPoly k j = x_j + y_j + C_j` has `2 + (2^j - 1) = 2^j + 1` monomials.

## Domain and what is not claimed

Every statement is total and universally quantified over `k` and `j : Fin k`; no width is
instantiated in the general proofs. `bitPoly_card_two` and `bitPoly_card_four` re-decide the
count at the widths where `work/proofs/AlgebraCarry.lean` enumerated (`add_k2_bit*_terms`,
`add_k4_bit*_terms`): kernel checks, not samples. The count is a cardinality statement about an
explicit `F2` polynomial. It constrains encodings whose monomials are degree-bounded or sparse; it
is **not** a lower bound on computation, and it makes no claim about extraction, Rust, SHA-2, ARX
or attack cost. The `MvPolynomial` bridge named in AR-29 §8 is not used: this file is
self-contained over a verbatim re-declaration of the AR-29 carrier, so the count is stated against
the executable representation. No `noncomputable` marker, no `sorry`, no raised limits.
-/
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.BitVec
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Disjoint
import Mathlib.Data.Finset.Image
import Mathlib.Data.Finset.Union
import Mathlib.Data.Finset.Fold
import Mathlib.Data.Finset.SDiff
import Mathlib.Data.Finset.Card
import Mathlib.Tactic.Positivity

set_option autoImplicit false

namespace AlgebraCarryTermCount

/-! ## 1 · The AR-29 carrier, re-declared verbatim

The five ANF objects are re-declared with the AR-29 bodies so that every statement below holds
definitionally of the executable carrier; `AlgebraCarryComputable.lean` is read-only and is not
imported. -/

/-- A squarefree monomial over `F2` in the `2k` input bits, as the set of variables it contains:
`(i, false)` is bit `i` of the first word, `(i, true)` bit `i` of the second word. -/
abbrev Monomial (k : ℕ) := Finset (Fin k × Bool)

/-- A polynomial over `F2` in the `2k` input bits, as the set of its nonzero monomials. This *is*
its algebraic normal form: the coefficient of a monomial is `1` exactly when it is a member. -/
abbrev Anf (k : ℕ) := Finset (Monomial k)

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

/-- The monomial of `xVar k i`, the singleton variable `xᵢ`. -/
def xVarMono (k : ℕ) (i : Fin k) : Monomial k := ({(i, false)} : Finset (Fin k × Bool))

/-- The monomial of `yVar k i`, the singleton variable `yᵢ`. -/
def yVarMono (k : ℕ) (i : Fin k) : Monomial k := ({(i, true)} : Finset (Fin k × Bool))

/-- **Bit `i` of the first word**, as an ANF: the single monomial `xᵢ`. -/
def xVar (k : ℕ) (i : Fin k) : Anf k := {xVarMono k i}

/-- **Bit `i` of the second word**, as an ANF: the single monomial `yᵢ`. -/
def yVar (k : ℕ) (i : Fin k) : Anf k := {yVarMono k i}

/-- **The carry into bit `j`, as an ANF.** The carry into bit `0` is the zero polynomial. The carry
into bit `j + 1` is the carry into bit `j` multiplied by the sum of the two input bits, plus the
product of the two input bits: over `F2` that is the full-adder relation `c' = (x + y) * c + x * y`.
Structural recursion on the bit index, exactly as in AR-29. -/
def carryPoly (k : ℕ) : Fin k → Anf k
  | ⟨0, _⟩ => ∅
  | ⟨j + 1, h⟩ =>
      anfAdd (anfMul (anfAdd (xVar k ⟨j, Nat.lt_of_succ_lt h⟩) (yVar k ⟨j, Nat.lt_of_succ_lt h⟩))
        (carryPoly k ⟨j, Nat.lt_of_succ_lt h⟩))
        (anfMul (xVar k ⟨j, Nat.lt_of_succ_lt h⟩) (yVar k ⟨j, Nat.lt_of_succ_lt h⟩))

/-- **Output bit `j` of `x + y`, as an ANF**: the two input bits plus the carry into that bit. -/
def bitPoly (k : ℕ) (j : Fin k) : Anf k := anfAdd (anfAdd (xVar k j) (yVar k j)) (carryPoly k j)

/-- The number of monomials of an ANF, i.e. the cardinality of the support of the corresponding
`F2` polynomial. The carrier is its own support, so this is `p.card`. -/
def anfTermCount {k : ℕ} (p : Anf k) : Nat := p.card

/-! ## 2 · Membership laws and the recursion equations -/

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

theorem carryPoly_zero (k : ℕ) (h : 0 < k) : carryPoly k ⟨0, h⟩ = ∅ := by
  simp [carryPoly]

theorem carryPoly_succ (k : ℕ) (j : ℕ) (hj : j + 1 < k) :
    carryPoly k ⟨j + 1, hj⟩
      = anfAdd
          (anfMul (anfAdd (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (yVar k ⟨j, Nat.lt_of_succ_lt hj⟩))
            (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩))
          (anfMul (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (yVar k ⟨j, Nat.lt_of_succ_lt hj⟩)) := by
  simp [carryPoly]

/-! ## 3 · Set-level laws of the two operations -/

theorem disjoint_xVar_yVar {k : ℕ} (i : Fin k) : Disjoint (xVar k i) (yVar k i) := by
  rw [Finset.disjoint_left]
  intro m hm hm'
  rw [xVar, Finset.mem_singleton] at hm
  rw [yVar, Finset.mem_singleton] at hm'
  exact xVarMono_ne_yVarMono i (hm.symm.trans hm')

theorem anfAdd_eq_union_of_disjoint {k : ℕ} {p q : Anf k} (h : Disjoint p q) :
    anfAdd p q = p ∪ q := by
  ext m
  rw [mem_anfAdd, Finset.mem_union]
  constructor
  · rintro (⟨hm, _⟩ | ⟨hm, _⟩)
    · exact Or.inl hm
    · exact Or.inr hm
  · rintro (hm | hm)
    · exact Or.inl ⟨hm, Finset.disjoint_left.mp h hm⟩
    · exact Or.inr ⟨hm, Finset.disjoint_left.mp h.symm hm⟩

theorem anfMul_union_left {k : ℕ} (p q c : Anf k) :
    anfMul (p ∪ q) c = anfMul p c ∪ anfMul q c := by
  ext m
  simp only [mem_anfMul, Finset.mem_union]
  constructor
  · rintro ⟨m₁, hm₁, m₂, hm₂, heq⟩
    rcases hm₁ with hm₁ | hm₁
    · exact Or.inl ⟨m₁, hm₁, m₂, hm₂, heq⟩
    · exact Or.inr ⟨m₁, hm₁, m₂, hm₂, heq⟩
  · rintro (⟨m₁, hm₁, m₂, hm₂, heq⟩ | ⟨m₁, hm₁, m₂, hm₂, heq⟩)
    · exact ⟨m₁, Or.inl hm₁, m₂, hm₂, heq⟩
    · exact ⟨m₁, Or.inr hm₁, m₂, hm₂, heq⟩

theorem mem_anfMul_xVar {k : ℕ} {i : Fin k} {c : Anf k} {m : Monomial k} :
    m ∈ anfMul (xVar k i) c ↔ ∃ m' ∈ c, monoMul (xVarMono k i) m' = m := by
  rw [mem_anfMul]
  constructor
  · rintro ⟨m₁, hm₁, m₂, hm₂, heq⟩
    rw [xVar, Finset.mem_singleton] at hm₁
    subst hm₁
    exact ⟨m₂, hm₂, heq⟩
  · rintro ⟨m', hm', rfl⟩
    exact ⟨xVarMono k i, by rw [xVar]; exact Finset.mem_singleton_self _, m', hm', rfl⟩

theorem mem_anfMul_yVar {k : ℕ} {i : Fin k} {c : Anf k} {m : Monomial k} :
    m ∈ anfMul (yVar k i) c ↔ ∃ m' ∈ c, monoMul (yVarMono k i) m' = m := by
  rw [mem_anfMul]
  constructor
  · rintro ⟨m₁, hm₁, m₂, hm₂, heq⟩
    rw [yVar, Finset.mem_singleton] at hm₁
    subst hm₁
    exact ⟨m₂, hm₂, heq⟩
  · rintro ⟨m', hm', rfl⟩
    exact ⟨yVarMono k i, by rw [yVar]; exact Finset.mem_singleton_self _, m', hm', rfl⟩

theorem mem_anfMul_xVar_yVar {k : ℕ} {i : Fin k} {m : Monomial k} :
    m ∈ anfMul (xVar k i) (yVar k i) ↔ m = monoMul (xVarMono k i) (yVarMono k i) := by
  rw [mem_anfMul]
  constructor
  · rintro ⟨m₁, hm₁, m₂, hm₂, heq⟩
    rw [xVar, Finset.mem_singleton] at hm₁
    rw [yVar, Finset.mem_singleton] at hm₂
    subst hm₁; subst hm₂
    exact heq.symm
  · rintro rfl
    exact ⟨xVarMono k i, by rw [xVar]; exact Finset.mem_singleton_self _,
      yVarMono k i, by rw [yVar]; exact Finset.mem_singleton_self _, rfl⟩

/-! ## 4 · Cardinality of the lifted classes

The two lifts of a carry support whose monomials avoid the index under multiplication, and the one
fresh monomial class `x_j y_j`. -/

/-- Union with a fixed singleton is injective on monomials that do not contain the singleton. -/
theorem singleton_union_inj {k : ℕ} {a : Fin k × Bool} {m₁ m₂ : Monomial k}
    (h₁ : a ∉ m₁) (h₂ : a ∉ m₂)
    (h : ({(a)} : Monomial k) ∪ m₁ = {(a)} ∪ m₂) : m₁ = m₂ := by
  ext v
  have hv := Finset.ext_iff.mp h v
  simp only [Finset.mem_union, Finset.mem_singleton] at hv
  by_cases hva : v = a
  · subst hva
    simp [h₁, h₂]
  · simpa [hva] using hv

theorem monoMul_xVarMono_inj {k : ℕ} {i : Fin k} {m₁ m₂ : Monomial k}
    (h₁ : (i, false) ∉ m₁) (h₂ : (i, false) ∉ m₂)
    (h : monoMul (xVarMono k i) m₁ = monoMul (xVarMono k i) m₂) : m₁ = m₂ := by
  rw [monoMul, xVarMono] at h
  exact singleton_union_inj h₁ h₂ h

theorem monoMul_yVarMono_inj {k : ℕ} {i : Fin k} {m₁ m₂ : Monomial k}
    (h₁ : (i, true) ∉ m₁) (h₂ : (i, true) ∉ m₂)
    (h : monoMul (yVarMono k i) m₁ = monoMul (yVarMono k i) m₂) : m₁ = m₂ := by
  rw [monoMul, yVarMono] at h
  exact singleton_union_inj h₁ h₂ h

/-- The `x_i`-lift of `c` has the same number of monomials as `c`, when no monomial of `c`
contains `x_i`. -/
theorem card_anfMul_xVar {k : ℕ} (i : Fin k) (c : Anf k)
    (h : ∀ m ∈ c, (i, false) ∉ m) :
    (anfMul (xVar k i) c).card = c.card := by
  have himg : anfMul (xVar k i) c = c.image (fun m => monoMul (xVarMono k i) m) := by
    ext m
    rw [mem_anfMul_xVar, Finset.mem_image]
  rw [himg]
  exact Finset.card_image_of_injOn fun m₁ hm₁ m₂ hm₂ heq =>
    monoMul_xVarMono_inj (h m₁ hm₁) (h m₂ hm₂) heq

/-- The `y_i`-lift of `c` has the same number of monomials as `c`, when no monomial of `c`
contains `y_i`. -/
theorem card_anfMul_yVar {k : ℕ} (i : Fin k) (c : Anf k)
    (h : ∀ m ∈ c, (i, true) ∉ m) :
    (anfMul (yVar k i) c).card = c.card := by
  have himg : anfMul (yVar k i) c = c.image (fun m => monoMul (yVarMono k i) m) := by
    ext m
    rw [mem_anfMul_yVar, Finset.mem_image]
  rw [himg]
  exact Finset.card_image_of_injOn fun m₁ hm₁ m₂ hm₂ heq =>
    monoMul_yVarMono_inj (h m₁ hm₁) (h m₂ hm₂) heq

/-- The fresh class `x_i * y_i` is the single monomial `{x_i, y_i}`. -/
theorem card_anfMul_xVar_yVar {k : ℕ} (i : Fin k) :
    (anfMul (xVar k i) (yVar k i)).card = 1 := by
  have hsingle : anfMul (xVar k i) (yVar k i) = {monoMul (xVarMono k i) (yVarMono k i)} := by
    ext m
    rw [mem_anfMul_xVar_yVar, Finset.mem_singleton]
  rw [hsingle, Finset.card_singleton]

/-! ### 4.1 · The three classes are pairwise disjoint -/

theorem anfMul_xVar_fst_mem {k : ℕ} {i : Fin k} {c : Anf k} {m : Monomial k}
    (hm : m ∈ anfMul (xVar k i) c) : (i, false) ∈ m := by
  rw [mem_anfMul_xVar] at hm
  rcases hm with ⟨m', _, rfl⟩
  exact Finset.mem_union_left _ (Finset.mem_singleton_self _)

theorem anfMul_yVar_snd_mem {k : ℕ} {i : Fin k} {c : Anf k} {m : Monomial k}
    (hm : m ∈ anfMul (yVar k i) c) : (i, true) ∈ m := by
  rw [mem_anfMul_yVar] at hm
  rcases hm with ⟨m', _, rfl⟩
  exact Finset.mem_union_left _ (Finset.mem_singleton_self _)

theorem anfMul_xVar_snd_notMem {k : ℕ} {i : Fin k} {c : Anf k} {m : Monomial k}
    (h : ∀ m' ∈ c, (i, true) ∉ m') (hm : m ∈ anfMul (xVar k i) c) : (i, true) ∉ m := by
  rw [mem_anfMul_xVar] at hm
  rcases hm with ⟨m', hm', rfl⟩
  intro hmem
  rw [monoMul, Finset.mem_union] at hmem
  rcases hmem with hmem | hmem
  · rw [xVarMono] at hmem
    exact absurd (Finset.mem_singleton.mp hmem) (by simp)
  · exact h m' hm' hmem

theorem anfMul_yVar_fst_notMem {k : ℕ} {i : Fin k} {c : Anf k} {m : Monomial k}
    (h : ∀ m' ∈ c, (i, false) ∉ m') (hm : m ∈ anfMul (yVar k i) c) : (i, false) ∉ m := by
  rw [mem_anfMul_yVar] at hm
  rcases hm with ⟨m', hm', rfl⟩
  intro hmem
  rw [monoMul, Finset.mem_union] at hmem
  rcases hmem with hmem | hmem
  · rw [yVarMono] at hmem
    exact absurd (Finset.mem_singleton.mp hmem) (by simp)
  · exact h m' hm' hmem

/-- An index-`j` variable is not in a monomial all of whose variables have index below `j`. -/
theorem fst_notMem_of_vars_lt {k : ℕ} {j : ℕ} {hj : j < k} {m : Monomial k}
    (h : ∀ v ∈ m, v.1.val < j) : ((⟨j, hj⟩ : Fin k), false) ∉ m := fun hmem =>
  absurd (by simpa using h _ hmem) (Nat.lt_irrefl j)

theorem snd_notMem_of_vars_lt {k : ℕ} {j : ℕ} {hj : j < k} {m : Monomial k}
    (h : ∀ v ∈ m, v.1.val < j) : ((⟨j, hj⟩ : Fin k), true) ∉ m := fun hmem =>
  absurd (by simpa using h _ hmem) (Nat.lt_irrefl j)

/-- The two lifts and the fresh class are pairwise disjoint. The fresh monomial is `{x_i, y_i}`:
an `x_i`-lift contains `x_i` and not `y_i`, a `y_i`-lift contains `y_i` and not `x_i`. -/
theorem disjoint_anfMul_xy_fresh {k : ℕ} (i : Fin k) (c : Anf k)
    (hxf : ∀ m ∈ c, (i, false) ∉ m) (hyf : ∀ m ∈ c, (i, true) ∉ m) :
    Disjoint (anfMul (xVar k i) c ∪ anfMul (yVar k i) c) (anfMul (xVar k i) (yVar k i)) := by
  rw [Finset.disjoint_left]
  intro m hm hmQ
  rw [mem_anfMul_xVar_yVar] at hmQ
  subst hmQ
  rw [Finset.mem_union] at hm
  rcases hm with hm | hm
  · exact anfMul_xVar_snd_notMem hyf hm
      (by rw [monoMul, Finset.mem_union]
          exact Or.inr (by rw [yVarMono]; exact Finset.mem_singleton_self _))
  · exact anfMul_yVar_fst_notMem hxf hm
      (by rw [monoMul, Finset.mem_union]
          exact Or.inl (by rw [xVarMono]; exact Finset.mem_singleton_self _))

theorem disjoint_anfMul_x_y {k : ℕ} (i : Fin k) (c : Anf k)
    (hyf : ∀ m ∈ c, (i, true) ∉ m) :
    Disjoint (anfMul (xVar k i) c) (anfMul (yVar k i) c) := by
  rw [Finset.disjoint_left]
  intro m hm hmPy
  exact anfMul_xVar_snd_notMem hyf hm (anfMul_yVar_snd_mem hmPy)

/-! ## 5 · The recurrence, stated

`carryPoly_succ_support` is the content of the recursion as three explicit, pairwise disjoint
classes; `carryPoly_succ_card` is the count recurrence it induces, and it is what the induction on
the bit index uses. The invariant `carryPoly_vars_lt` — every variable of every monomial of the
carry into bit `j` has index below `j` — is what makes the two lifts injective and the fresh class
disjoint from them. -/

/-- Every variable of every monomial of the carry into bit `j` has index below `j`: the index `j` is
fresh at each recursion step. Induction on the bit index. -/
theorem carryPoly_vars_lt (k : ℕ) : ∀ (j : ℕ) (hj : j < k),
    ∀ m ∈ carryPoly k ⟨j, hj⟩, ∀ v ∈ m, v.1.val < j := by
  intro j
  induction j using Nat.strong_induction_on with
  | h j ih =>
    intro hj m hm v hv
    rcases Nat.eq_zero_or_pos j with hj0 | hj0
    · subst hj0
      rw [carryPoly_zero k hj] at hm
      exact absurd hm (Finset.notMem_empty m)
    · obtain ⟨m0, rfl, hmK⟩ : ∃ m0, j = m0 + 1 ∧ m0 < k := ⟨j - 1, by omega, by omega⟩
      simp only [carryPoly_succ, mem_anfAdd] at hm
      rcases hm with ⟨hm, _⟩ | ⟨hm, _⟩
      · rw [mem_anfMul] at hm
        rcases hm with ⟨m₁, hm₁, m₂, hm₂, rfl⟩
        rw [mem_anfAdd] at hm₁
        have h2 := ih m0 (by omega) (Nat.lt_of_succ_lt hj) m₂ hm₂
        rcases hm₁ with ⟨hx, _⟩ | ⟨hy, _⟩
        · have hx' : m₁ = xVarMono k ⟨m0, by omega⟩ := by simpa [xVar] using hx
          rw [hx', monoMul, Finset.mem_union, mem_xVarMono] at hv
          rcases hv with hv | hv
          · subst hv; simp
          · have := h2 v hv; omega
        · have hy' : m₁ = yVarMono k ⟨m0, by omega⟩ := by simpa [yVar] using hy
          rw [hy', monoMul, Finset.mem_union, mem_yVarMono] at hv
          rcases hv with hv | hv
          · subst hv; simp
          · have := h2 v hv; omega
      · rw [mem_anfMul] at hm
        rcases hm with ⟨m₁, hm₁, m₂, hm₂, rfl⟩
        have hm₁' : m₁ = xVarMono k ⟨m0, by omega⟩ := by simpa [xVar] using hm₁
        have hm₂' : m₂ = yVarMono k ⟨m0, by omega⟩ := by simpa [yVar] using hm₂
        rw [hm₁', hm₂', monoMul, Finset.mem_union, mem_xVarMono, mem_yVarMono] at hv
        rcases hv with hv | hv
        · subst hv; simp
        · subst hv; simp

/-- **THE RECURRENCE.** The carry into bit `j + 1` is the union of the `x_j`-lift of the carry into
bit `j`, its `y_j`-lift, and the one fresh monomial `x_j y_j` — the three classes being pairwise
disjoint. -/
theorem carryPoly_succ_support (k : ℕ) (j : ℕ) (hj : j + 1 < k) :
    carryPoly k ⟨j + 1, hj⟩
      = (anfMul (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩)
          ∪ anfMul (yVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩))
        ∪ anfMul (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (yVar k ⟨j, Nat.lt_of_succ_lt hj⟩) := by
  have hC := carryPoly_vars_lt k j (Nat.lt_of_succ_lt hj)
  have hxf : ∀ m ∈ carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩,
      ((⟨j, Nat.lt_of_succ_lt hj⟩ : Fin k), false) ∉ m :=
    fun m hm => fst_notMem_of_vars_lt (hC m hm)
  have hyf : ∀ m ∈ carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩,
      ((⟨j, Nat.lt_of_succ_lt hj⟩ : Fin k), true) ∉ m :=
    fun m hm => snd_notMem_of_vars_lt (hC m hm)
  rw [carryPoly_succ k j hj,
      anfAdd_eq_union_of_disjoint (disjoint_xVar_yVar ⟨j, Nat.lt_of_succ_lt hj⟩), anfMul_union_left,
      anfAdd_eq_union_of_disjoint (disjoint_anfMul_xy_fresh ⟨j, Nat.lt_of_succ_lt hj⟩
        (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩) hxf hyf)]

/-- **THE COUNT RECURRENCE.** `card C_{j+1} = 2 * card C_j + 1`: each old monomial has exactly two
index-`j` lifts, `x_j * m` and `y_j * m`, and the fresh monomial `x_j y_j` adds one. -/
theorem carryPoly_succ_card (k : ℕ) (j : ℕ) (hj : j + 1 < k) :
    (carryPoly k ⟨j + 1, hj⟩).card = 2 * (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩).card + 1 := by
  have hC := carryPoly_vars_lt k j (Nat.lt_of_succ_lt hj)
  have hxf : ∀ m ∈ carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩,
      ((⟨j, Nat.lt_of_succ_lt hj⟩ : Fin k), false) ∉ m :=
    fun m hm => fst_notMem_of_vars_lt (hC m hm)
  have hyf : ∀ m ∈ carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩,
      ((⟨j, Nat.lt_of_succ_lt hj⟩ : Fin k), true) ∉ m :=
    fun m hm => snd_notMem_of_vars_lt (hC m hm)
  rw [carryPoly_succ_support k j hj,
      Finset.card_union_of_disjoint (disjoint_anfMul_xy_fresh ⟨j, Nat.lt_of_succ_lt hj⟩
        (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩) hxf hyf),
      Finset.card_union_of_disjoint (disjoint_anfMul_x_y ⟨j, Nat.lt_of_succ_lt hj⟩
        (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩) hyf),
      card_anfMul_xVar ⟨j, Nat.lt_of_succ_lt hj⟩ (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩) hxf,
      card_anfMul_yVar ⟨j, Nat.lt_of_succ_lt hj⟩ (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩) hyf,
      card_anfMul_xVar_yVar ⟨j, Nat.lt_of_succ_lt hj⟩]
  omega

/-! ## 6 · The count at every width -/

/-- **The carry into bit `j` has `2^j - 1` monomials**, stated additively as `card C_j + 1 = 2^j`
to avoid truncated subtraction: induction on the bit index with the count recurrence
`carryPoly_succ_card`. -/
theorem carryPoly_card (k : ℕ) : ∀ (j : ℕ) (hj : j < k),
    (carryPoly k ⟨j, hj⟩).card + 1 = 2 ^ j := by
  intro j
  induction j using Nat.strong_induction_on with
  | h j ih =>
    intro hj
    rcases Nat.eq_zero_or_pos j with hj0 | hj0
    · subst hj0
      rw [carryPoly_zero k hj]
      simp
    · obtain ⟨m0, rfl, hmK⟩ : ∃ m0, j = m0 + 1 ∧ m0 < k := ⟨j - 1, by omega, by omega⟩
      rw [carryPoly_succ_card k m0 hj]
      have hrec := ih m0 (by omega) (Nat.lt_of_succ_lt hj)
      have htwo : 2 * (carryPoly k ⟨m0, Nat.lt_of_succ_lt hj⟩).card + 1 + 1
          = 2 * ((carryPoly k ⟨m0, Nat.lt_of_succ_lt hj⟩).card + 1) := by omega
      rw [pow_succ, htwo, hrec]
      exact Nat.mul_comm _ _

/-- Output bit `j + 1`, expanded through the carry recurrence: its two linear monomials plus the
three classes of `carryPoly_succ_support`. -/
theorem bitPoly_succ_eq (k : ℕ) (j : ℕ) (hj : j + 1 < k) :
    bitPoly k ⟨j + 1, hj⟩
      = anfAdd (anfAdd (xVar k ⟨j + 1, hj⟩) (yVar k ⟨j + 1, hj⟩))
          ((anfMul (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩)
            ∪ anfMul (yVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩))
           ∪ anfMul (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩) (yVar k ⟨j, Nat.lt_of_succ_lt hj⟩)) := by
  rw [bitPoly, carryPoly_succ_support k j hj]

/-- **THE TERM COUNT, at every width.** The ANF of output bit `j` of the sum of two `k`-bit words
has exactly `2^j + 1` monomials, for every `k` and every `j < k`. The two linear monomials `x_j`,
`y_j` are disjoint from the carry support (whose monomials avoid index `j`), and the carry itself
has `2^j - 1` monomials. -/
theorem bitPoly_card (k : ℕ) (j : Fin k) : (bitPoly k j).card = 2 ^ j.val + 1 := by
  have hC := carryPoly_vars_lt k j.val j.isLt
  have hCcard : (carryPoly k j).card + 1 = 2 ^ j.val := by
    simpa using carryPoly_card k j.val j.isLt
  have hxnot : xVarMono k j ∉ carryPoly k j := by
    intro hmem
    have hlt := hC (xVarMono k j) hmem (j, false)
      (by rw [xVarMono]; exact Finset.mem_singleton_self _)
    exact absurd (by simp at hlt) (Nat.lt_irrefl j.val)
  have hynot : yVarMono k j ∉ carryPoly k j := by
    intro hmem
    have hlt := hC (yVarMono k j) hmem (j, true)
      (by rw [yVarMono]; exact Finset.mem_singleton_self _)
    exact absurd (by simp at hlt) (Nat.lt_irrefl j.val)
  have hdisj : Disjoint (anfAdd (xVar k j) (yVar k j)) (carryPoly k j) := by
    rw [Finset.disjoint_left]
    intro m hm hmC
    rw [mem_anfAdd] at hm
    rcases hm with ⟨hmx, _⟩ | ⟨hmy, _⟩
    · rw [xVar, Finset.mem_singleton] at hmx
      subst hmx
      exact hxnot hmC
    · rw [yVar, Finset.mem_singleton] at hmy
      subst hmy
      exact hynot hmC
  have hLcard : (anfAdd (xVar k j) (yVar k j)).card = 2 := by
    rw [anfAdd_eq_union_of_disjoint (disjoint_xVar_yVar j),
        Finset.card_union_of_disjoint (disjoint_xVar_yVar j)]
    simp [xVar, yVar]
  rw [bitPoly, anfAdd_eq_union_of_disjoint hdisj, Finset.card_union_of_disjoint hdisj, hLcard]
  omega

/-- The term count in the `anfTermCount` spelling; the carrier is its own support, so this is the
support cardinality. -/
theorem anfTermCount_eq_card {k : ℕ} (p : Anf k) : anfTermCount p = p.card := rfl

theorem anfTermCount_bitPoly (k : ℕ) (j : Fin k) :
    anfTermCount (bitPoly k j) = 2 ^ j.val + 1 := bitPoly_card k j

/-! ## 7 · The finite instances at `k = 2` and `k = 4`, decided by the kernel -/

theorem bitPoly_card_two : ∀ j : Fin 2, (bitPoly 2 j).card = 2 ^ j.val + 1 := by decide +kernel

theorem bitPoly_card_four : ∀ j : Fin 4, (bitPoly 4 j).card = 2 ^ j.val + 1 := by decide +kernel

theorem carryPoly_card_two : ∀ j : Fin 2, (carryPoly 2 j).card + 1 = 2 ^ j.val := by
  decide +kernel

theorem carryPoly_card_four : ∀ j : Fin 4, (carryPoly 4 j).card + 1 = 2 ^ j.val := by
  decide +kernel

/-! ## 8 · The cone -/

#print axioms mem_anfAdd
#print axioms mem_anfMul
#print axioms anfAdd_eq_union_of_disjoint
#print axioms anfMul_union_left
#print axioms mem_anfMul_xVar
#print axioms mem_anfMul_yVar
#print axioms mem_anfMul_xVar_yVar
#print axioms singleton_union_inj
#print axioms card_anfMul_xVar
#print axioms card_anfMul_yVar
#print axioms card_anfMul_xVar_yVar
#print axioms disjoint_anfMul_xy_fresh
#print axioms disjoint_anfMul_x_y
#print axioms carryPoly_vars_lt
#print axioms carryPoly_succ_support
#print axioms carryPoly_succ_card
#print axioms carryPoly_card
#print axioms bitPoly_succ_eq
#print axioms bitPoly_card
#print axioms anfTermCount_bitPoly
#print axioms bitPoly_card_two
#print axioms bitPoly_card_four
#print axioms carryPoly_card_two
#print axioms carryPoly_card_four

end AlgebraCarryTermCount
