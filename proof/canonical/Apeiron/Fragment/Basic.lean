import Apeiron.Bridge.Cell
import Apeiron.Trichotomy.Channels
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fintype.Pi
import Mathlib.Tactic.Linarith

/-!
# Fragment containment

@name Fragment
@grade proved
@cite farkas-1902
@cite presburger-1929
@cite matiyasevich-1970

A decider that answers in finite time answers about a small language. `../song`'s prover answers
about rows: a whole-number constant, plus whole-number multiples of columns, asserted to be at
least zero. Nothing in a row multiplies one column by another. That is what makes the question
decidable, and until now it was a reading of the source rather than a theorem.

## Root

The fragment sits on the split face, and the face is forced rather than chosen. A row is an
order: `c + a.x >= 0`. A cap is two orders, one from each side, and two orders need two
independent readings of a point. `Trichotomy/Channels.lean`'s `two_channels_iff_disc_pos` says an
algebra has two of those exactly when σ is above zero, so `two_caps_iff_disc_pos` below
puts the whole row format on `0 < σ`, and `../song` fixes the scale at `σ = 1`, which is
`Bridge/Cell.lean`'s `ofBounds`.

Three theorems of the trichotomy are instantiated, and each has an edge theorem here derived from
it rather than restated beside it.

- `two_channels_iff_disc_pos` (`Trichotomy/Channels.lean`). Edge: `two_caps_iff_disc_pos`. The
  upper and lower readings are two distinct channels exactly above zero, which is the face a
  two-sided cap can be stated on at all.
- `le_iff_exists_sq` and `int_le_iff_exists_sum_four_sq` (`Trichotomy/NullBasis.lean`). Edge:
  `holds_iff_four_squares`. The order a row asserts is definable in the ring, one square over the
  reals and four over the whole numbers, so "at least zero" is a ring statement and not a
  predicate the fragment carries on the side.
- `crossing_ofBounds` (`Bridge/Cell.lean`), built on `upper` and `lower`, and
  `crossing_independent_of_cone` (`Trichotomy/NullBasis.lean`). Edges: `holds_iff_not_crossing`,
  `crossingRow_is_the_crossing`, `norm_not_a_row` and `verdict_needs_no_norm`. A row holding is a
  cell of the algebra not crossing, the verdict is one such row, and the cone `normForm = lo * hi`
  is the quadratic the row format cannot state and the verdict never asks for.

## The grammar, and the constructor that is missing

Start with what a law or spec row is allowed to say. A constant. A column. A constant multiple of
something already built. A sum. A difference. Six constructors, and the sixth is the interesting
one: a product, a shift or a mask enters as `opaqueCol`, which carries an operator tag, two
operands, and **a column**. Its denotation is that column and nothing about the operands.

There is no constructor taking two expressions and returning their product as a value. `Expr` has
`scale`, which multiplies by a literal, and it has `opaqueCol`, which throws the product away and
keeps a fresh variable. It does not have `mul`. This is `Capability/Basic.lean`'s pattern: the
absence of a constructor is the theorem, and `product_not_a_row` is where the absence bites. No
row's value is `x` times `y`, for any two distinct columns, because a row is additive in the
assignment and a product is not.

## Two readings of an opaque column, and why the abstraction is sound

An opaque column has two readings and the gap between them is the whole point.

- `denote` sends `opaqueCol op c a b` to the row `c`, forgetting `op`, `a` and `b`.
- `Expr.eval` sends it to `μ op` applied to the operand values, the number the machine computes.

`Keyed` is the bridge: an assignment is keyed for an expression when every opaque column in it
carries the value the machine would compute there. `denote_eval` says that on a keyed assignment
the two readings agree, so a row proved for every assignment holds for the machine's own. Nothing
about `μ` is used, which is why assuming nothing about a product is sound.

## What is proved here

`eval_is_a_row`: every expression the grammar builds has one row that computes it on keyed
assignments. `product_not_a_row`: no row computes a product of two columns. `monomial_cancels`
and `scale_distributes`: `../song/src/prover/face/form.slang`'s two free-standing laws,
`fragment_monomial_consistent` and `fragment_scale_linear`, restated over the model.
`combo_holds` and `unsat_of_negative_combo`: Farkas, which is `arith.slang`'s `combovalid_nonneg`
and `farkas_bound` plus the refutation they are for. `sat_iff_finite_search`: satisfiability of a
finite conjunction of rows inside declared caps is decidable, by search over the box.

## What this does not settle

Two things, and both are named rather than hedged.

The Slang side. That `../song`'s extractor emits only this grammar is a claim about that
extractor, checkable by reading `src/prover/face/form.slang`, and is not established here. It is
the same scope `Apeiron/Capability/Basic.lean` states for its own grammar.

The unbounded fragment. `sat_iff_finite_search` decides satisfiability when every column carries
declared caps, which is the case `facec` emits and the case `ncone0` runs under `PROP_CAP`.
Satisfiability over all of the whole numbers, with no caps, is Presburger's decidable fragment and
is **not** proved here. Mathlib carries `presburger.definable_iff_isSemilinearSet` and
`presburger.mul_not_definable`, which say which sets are definable rather than how to decide one,
and no decision procedure for the fragment exists in Lean today. `unit_elimination_exact` is the
one step of it that does not need the machinery: eliminating a column whose coefficient is one is
exact over the whole numbers. `theory/DEAD_ENDS.md` carries the three routes tried.
-/

namespace Apeiron
namespace Fragment

open Trichotomy Bridge

/-- A row: a constant, and one whole-number coefficient per column.

    Row (n : ℕ)              [a row over n columns]
    const : ℤ                [the constant term]
    coeff : Fin n → ℤ        [one whole-number coefficient per column]

    So: `../song/src/prover/cone/ncone0.slang`'s row layout, `R_C` then `R_A` for `MAXV`
    columns, read as the assertion `const + coeff . x >= 0`. -/
structure Row (n : ℕ) where
  /-- The constant term. -/
  const : ℤ
  /-- One coefficient per column. -/
  coeff : Fin n → ℤ

namespace Row

variable {n : ℕ}

/-- What a row works out to at an assignment.

    (ρ : Fin n → ℤ)                          [a whole number for each of the n columns]
    (r : Row n)                              [a row]
    : ℤ                                      [the row's value there] -/
def value (ρ : Fin n → ℤ) (r : Row n) : ℤ :=
  r.const + ∑ i, r.coeff i * ρ i

/-- A row holds at an assignment when its value is at least zero.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    (r : Row n)                              [a row]
    : Prop                                   [a statement that is true or false] -/
def Holds (ρ : Fin n → ℤ) (r : Row n) : Prop := 0 ≤ r.value ρ

/-- Whether a row holds is decidable at any assignment.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    DecidablePred (Holds ρ)                  [holding is a check, not just a statement] -/
instance decidableHolds (ρ : Fin n → ℤ) : DecidablePred (Holds ρ) :=
  fun r => inferInstanceAs (Decidable (0 ≤ r.value ρ))

/-- The row asserting a constant.

    (c : ℤ)                                  [a whole number]
    : Row n                                  [the row with that constant and no columns] -/
def ofConst (c : ℤ) : Row n := ⟨c, fun _ => 0⟩

/-- The row that is one column on its own.

    (i : Fin n)                              [which column]
    : Row n                                  [the row whose value is that column's value] -/
def ofCol (i : Fin n) : Row n := ⟨0, fun k => if k = i then 1 else 0⟩

/-- Two rows added coefficient by coefficient.

    (r s : Row n)                            [two rows]
    : Row n                                  [their sum] -/
def add (r s : Row n) : Row n := ⟨r.const + s.const, fun i => r.coeff i + s.coeff i⟩

/-- One row taken away from another, coefficient by coefficient.

    (r s : Row n)                            [two rows]
    : Row n                                  [the first minus the second] -/
def sub (r s : Row n) : Row n := ⟨r.const - s.const, fun i => r.coeff i - s.coeff i⟩

/-- A row multiplied through by a whole number.

    (r : Row n)                              [a row]
    (c : ℤ)                                  [a whole number to multiply by]
    : Row n                                  [every coefficient and the constant, times c] -/
def scale (r : Row n) (c : ℤ) : Row n := ⟨c * r.const, fun i => c * r.coeff i⟩

/-- A row with its constant moved.

    (r : Row n)                              [a row]
    (d : ℤ)                                  [how far to move the constant]
    : Row n                                  [the same coefficients, constant shifted by d] -/
def shift (r : Row n) (d : ℤ) : Row n := ⟨r.const + d, r.coeff⟩

/-- A constant row is worth its constant.

    (ρ : Fin n → ℤ)                          [an assignment]
    (c : ℤ)                                  [a whole number]
    value ρ (ofConst c) = c                  [the row is worth that number] -/
@[simp] theorem value_ofConst (ρ : Fin n → ℤ) (c : ℤ) : value ρ (ofConst c) = c := by
  simp [value, ofConst]

/-- A single-column row is worth that column.

    (ρ : Fin n → ℤ)                          [an assignment]
    (i : Fin n)                              [which column]
    value ρ (ofCol i) = ρ i                  [the row is worth what that column holds] -/
@[simp] theorem value_ofCol (ρ : Fin n → ℤ) (i : Fin n) : value ρ (ofCol i) = ρ i := by
  simp [value, ofCol, ite_mul]

/-- Adding rows adds their values.

    (ρ : Fin n → ℤ)                          [an assignment]
    (r s : Row n)                            [two rows]
    value ρ (add r s) = value ρ r + value ρ s [the sum's value is the sum of the values] -/
@[simp] theorem value_add (ρ : Fin n → ℤ) (r s : Row n) :
    value ρ (add r s) = value ρ r + value ρ s := by
  simp only [value, add, add_mul, Finset.sum_add_distrib]
  ring

/-- Subtracting rows subtracts their values.

    (ρ : Fin n → ℤ)                          [an assignment]
    (r s : Row n)                            [two rows]
    value ρ (sub r s) = value ρ r - value ρ s [the difference's value is the difference] -/
@[simp] theorem value_sub (ρ : Fin n → ℤ) (r s : Row n) :
    value ρ (sub r s) = value ρ r - value ρ s := by
  simp only [value, sub, sub_mul, Finset.sum_sub_distrib]
  ring

/-- Multiplying a row through multiplies its value.

    (ρ : Fin n → ℤ)                          [an assignment]
    (r : Row n)                              [a row]
    (c : ℤ)                                  [a whole number]
    value ρ (scale r c) = c * value ρ r      [scaling the row scales the value]

    So: the one multiplication `facec` puts into a row, `formScaleNew`, and the constant-factor
    arm of `formMulNew`. The factor is a literal, so the row stays linear in the columns. -/
@[simp] theorem value_scale (ρ : Fin n → ℤ) (r : Row n) (c : ℤ) :
    value ρ (scale r c) = c * value ρ r := by
  have h : ∑ i, c * r.coeff i * ρ i = c * ∑ i, r.coeff i * ρ i := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  simp only [value, scale, h]
  ring

/-- Moving a row's constant moves its value.

    (ρ : Fin n → ℤ)                          [an assignment]
    (r : Row n)                              [a row]
    (d : ℤ)                                  [how far to move the constant]
    value ρ (shift r d) = value ρ r + d      [the value moves by the same amount] -/
@[simp] theorem value_shift (ρ : Fin n → ℤ) (r : Row n) (d : ℤ) :
    value ρ (shift r d) = value ρ r + d := by
  simp only [value, shift]
  ring

/-- A row is additive in the assignment, up to the constant it repeats.

    (r : Row n)                                        [a row]
    (ρ τ : Fin n → ℤ)                                  [two assignments]
    value (ρ + τ) r + r.const = value ρ r + value τ r   [adding assignments adds values]

    So: this is what "linear in the columns" means as a checkable statement, and it is the
    property a product fails. The constant appears once on the left and twice on the right,
    which is the only way an affine map differs from a linear one. -/
theorem value_add_valuation (r : Row n) (ρ τ : Fin n → ℤ) :
    value (ρ + τ) r + r.const = value ρ r + value τ r := by
  simp only [value, Pi.add_apply, mul_add, Finset.sum_add_distrib]
  ring

/-- The assignment that puts one at a single column and zero everywhere else.

    (i : Fin n)                              [which column]
    : Fin n → ℤ                              [one there, zero at every other column] -/
def unitAt (i : Fin n) : Fin n → ℤ := fun k => if k = i then 1 else 0

/-- No row computes a fixed multiple of a product of two different columns.

    {i j : Fin n}                                              [two columns]
    (hij : i ≠ j)                                              [they are different columns]
    {k : ℤ}                                                    [a whole-number factor]
    (hk : k ≠ 0)                                               [the factor is not zero]
    ¬ ∃ r : Row n, ∀ ρ : Fin n → ℤ,                            [no row works at every assignment]
      value ρ r = k * (ρ i * ρ j)                              [worth that multiple of the product]

    So: the containment theorem. A product escapes the row format, whatever coefficients are
    tried, because `value_add_valuation` makes a row additive in the assignment and the product
    of the two unit assignments is one where the sum of the two products is zero. -/
theorem scaled_product_not_a_row {i j : Fin n} (hij : i ≠ j) {k : ℤ} (hk : k ≠ 0) :
    ¬ ∃ r : Row n, ∀ ρ : Fin n → ℤ, value ρ r = k * (ρ i * ρ j) := by
  rintro ⟨r, hr⟩
  have hji : j ≠ i := Ne.symm hij
  have hconst : r.const = 0 := by
    have h := hr (fun _ => 0)
    simpa [value] using h
  have hi : value (unitAt i) r = 0 := by
    rw [hr]; simp [unitAt, hji]
  have hj : value (unitAt j) r = 0 := by
    rw [hr]; simp [unitAt, hij]
  have hij' : value (unitAt i + unitAt j) r = k := by
    rw [hr]; simp [unitAt, hij, hji]
  have hsum := value_add_valuation r (unitAt i) (unitAt j)
  rw [hi, hj, hij', hconst] at hsum
  exact hk (by linarith)

/-- No row computes a product of two different columns.

    {i j : Fin n}                                              [two columns]
    (hij : i ≠ j)                                              [they are different columns]
    ¬ ∃ r : Row n, ∀ ρ : Fin n → ℤ,                            [no row works at every assignment]
      value ρ r = ρ i * ρ j                                    [worth the two columns multiplied]

    So: `scaled_product_not_a_row` at a factor of one. -/
theorem product_not_a_row {i j : Fin n} (hij : i ≠ j) :
    ¬ ∃ r : Row n, ∀ ρ : Fin n → ℤ, value ρ r = ρ i * ρ j := by
  intro h
  refine scaled_product_not_a_row hij (k := 1) one_ne_zero ?_
  obtain ⟨r, hr⟩ := h
  exact ⟨r, fun ρ => by rw [hr ρ]; ring⟩

end Row

/-- What a law or spec row is allowed to state.

    Expr (n : ℕ)                             [an expression over n columns]
    const (c : ℤ)                            [a whole-number constant]
    var (i : Fin n)                          [one column, read]
    opaqueCol (op : ℕ) (c : Fin n)           [a nonlinear operation, and the column standing for it]
      (a b : Expr n)                         [its two operands, recorded and not used]
    scale (c : ℤ) (a : Expr n)               [a whole-number multiple of something already built]
    add (a b : Expr n)                       [two expressions added]
    sub (a b : Expr n)                       [one expression taken away from another]

    So: six constructors and no seventh. Nothing takes two expressions and returns their product
    as a value. A product, a shift or a mask reaches a row only through `opaqueCol`, whose
    denotation is a column. -/
inductive Expr (n : ℕ) : Type
  /-- A whole-number constant. -/
  | const (c : ℤ)
  /-- One column, read. -/
  | var (i : Fin n)
  /-- A nonlinear operation, taken as the column that stands for it. -/
  | opaqueCol (op : ℕ) (c : Fin n) (a b : Expr n)
  /-- A whole-number multiple. -/
  | scale (c : ℤ) (a : Expr n)
  /-- Two expressions added. -/
  | add (a b : Expr n)
  /-- One expression taken away from another. -/
  | sub (a b : Expr n)

variable {n : ℕ}

/-- The row an expression turns into.

    Expr n → Row n                           [takes an expression, gives one row]

    So: `../song/src/prover/face/form.slang`'s lowering, with `formSetVarNew (monomialColumn ..)`
    as the `opaqueCol` case. The operands are dropped, which is what "carrying no value property"
    means there. -/
def denote : Expr n → Row n
  | .const c => Row.ofConst c
  | .var i => Row.ofCol i
  | .opaqueCol _ c _ _ => Row.ofCol c
  | .scale c a => (denote a).scale c
  | .add a b => (denote a).add (denote b)
  | .sub a b => (denote a).sub (denote b)

/-- What the machine computes, with the nonlinear operations supplied.

    (μ : ℕ → ℤ → ℤ → ℤ)                      [what each nonlinear operator tag does]
    (ρ : Fin n → ℤ)                          [an assignment to the columns]
    Expr n → ℤ                               [takes an expression, gives the number]

    So: the reading `denote` throws away. Nothing below assumes anything about `μ`. -/
def Expr.eval (μ : ℕ → ℤ → ℤ → ℤ) (ρ : Fin n → ℤ) : Expr n → ℤ
  | .const c => c
  | .var i => ρ i
  | .opaqueCol op _ a b => μ op (Expr.eval μ ρ a) (Expr.eval μ ρ b)
  | .scale c a => c * Expr.eval μ ρ a
  | .add a b => Expr.eval μ ρ a + Expr.eval μ ρ b
  | .sub a b => Expr.eval μ ρ a - Expr.eval μ ρ b

/-- An assignment is keyed for an expression when each opaque column holds what it stands for.

    (μ : ℕ → ℤ → ℤ → ℤ)                      [what each nonlinear operator tag does]
    (ρ : Fin n → ℤ)                          [an assignment to the columns]
    Expr n → Prop                            [takes an expression, gives a statement]

    So: `monomialColumn`'s congruence as a condition on the assignment. The same operands map to
    the same column, so one assignment can satisfy every occurrence at once. -/
def Keyed (μ : ℕ → ℤ → ℤ → ℤ) (ρ : Fin n → ℤ) : Expr n → Prop
  | .const _ => True
  | .var _ => True
  | .opaqueCol op c a b =>
      ρ c = μ op (Expr.eval μ ρ a) (Expr.eval μ ρ b) ∧ Keyed μ ρ a ∧ Keyed μ ρ b
  | .scale _ a => Keyed μ ρ a
  | .add a b => Keyed μ ρ a ∧ Keyed μ ρ b
  | .sub a b => Keyed μ ρ a ∧ Keyed μ ρ b

/-- On a keyed assignment the row and the machine agree.

    (μ : ℕ → ℤ → ℤ → ℤ)                              [what each nonlinear operator tag does]
    (ρ : Fin n → ℤ)                                  [an assignment to the columns]
    (e : Expr n)                                     [an expression]
    (h : Keyed μ ρ e)                                [each opaque column holds what it stands for]
    ∀ e : Expr n                                     [for every single expression]
    Row.value ρ (denote e) = Expr.eval μ ρ e         [the row is worth what the machine computes]

    So: the soundness of the opaque-column abstraction. A row proved for every assignment is proved
    for the keyed ones, which are the machine's own, and nothing about `μ` was used to get there. -/
theorem denote_eval (μ : ℕ → ℤ → ℤ → ℤ) (ρ : Fin n → ℤ) :
    ∀ e : Expr n, Keyed μ ρ e → Row.value ρ (denote e) = Expr.eval μ ρ e := by
  intro e
  induction e with
  | const c => intro _; simp [denote, Expr.eval]
  | var i => intro _; simp [denote, Expr.eval]
  | opaqueCol op c a b _ _ => intro h; simp [denote, Expr.eval, h.1]
  | scale c a ih => intro h; simp [denote, Expr.eval, ih h]
  | add a b iha ihb => intro h; simp [denote, Expr.eval, iha h.1, ihb h.2]
  | sub a b iha ihb => intro h; simp [denote, Expr.eval, iha h.1, ihb h.2]

/-- Every expression the grammar builds is computed by one row.

    (e : Expr n)                                     [any expression]
    ∃ r : Row n, ∀ (μ : ℕ → ℤ → ℤ → ℤ)               [there is one row, for every nonlinear reading]
      (ρ : Fin n → ℤ), Keyed μ ρ e →                 [and every keyed assignment]
      Row.value ρ r = Expr.eval μ ρ e                [the row is worth what the machine computes]

    So: fragment containment. The row is chosen before `μ` is, so no constructor of the grammar
    escapes into arithmetic the row format cannot state. -/
theorem eval_is_a_row (e : Expr n) :
    ∃ r : Row n, ∀ (μ : ℕ → ℤ → ℤ → ℤ) (ρ : Fin n → ℤ),
      Keyed μ ρ e → Row.value ρ r = Expr.eval μ ρ e :=
  ⟨denote e, fun μ ρ h => denote_eval μ ρ e h⟩

/-- The grammar is exactly its six constructors.

    (e : Expr n)                                     [any expression]
    (∃ c, e = Expr.const c) ∨ (∃ i, e = Expr.var i)  [a constant, or one column]
    ∨ (∃ op c a b, e = Expr.opaqueCol op c a b)      [or a nonlinear operation taken as a column]
    ∨ (∃ c a, e = Expr.scale c a)                    [or a whole-number multiple]
    ∨ (∃ a b, e = Expr.add a b)                      [or a sum]
    ∨ (∃ a b, e = Expr.sub a b)                      [or a difference]

    So: the lemma that breaks first if a product constructor is ever added, which is why it is
    stated. `Apeiron/Capability/Basic.lean`'s `cexpr_exhaustive` does the same job there. -/
theorem expr_exhaustive (e : Expr n) :
    (∃ c, e = Expr.const c) ∨ (∃ i, e = Expr.var i) ∨
      (∃ op c a b, e = Expr.opaqueCol op c a b) ∨ (∃ c a, e = Expr.scale c a) ∨
      (∃ a b, e = Expr.add a b) ∨ (∃ a b, e = Expr.sub a b) := by
  cases e with
  | const c => exact Or.inl ⟨c, rfl⟩
  | var i => exact Or.inr (Or.inl ⟨i, rfl⟩)
  | opaqueCol op c a b => exact Or.inr (Or.inr (Or.inl ⟨op, c, a, b, rfl⟩))
  | scale c a => exact Or.inr (Or.inr (Or.inr (Or.inl ⟨c, a, rfl⟩)))
  | add a b => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨a, b, rfl⟩))))
  | sub a b => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨a, b, rfl⟩))))

/-- No expression the grammar builds computes a product of two different columns.

    {i j : Fin n}                                    [two columns]
    (hij : i ≠ j)                                    [they are different columns]
    (e : Expr n)                                     [any expression]
    ¬ ∀ ρ : Fin n → ℤ,                               [there is no assignment-wide agreement]
      Row.value ρ (denote e) = ρ i * ρ j             [with the two columns multiplied]

    So: the forging theorem. Grammar plus denotation cannot reach a nonlinear row, so the question
    the decider is asked stays inside the fragment it decides. -/
theorem no_expr_denotes_a_product {i j : Fin n} (hij : i ≠ j) (e : Expr n) :
    ¬ ∀ ρ : Fin n → ℤ, Row.value ρ (denote e) = ρ i * ρ j :=
  fun h => Row.product_not_a_row hij ⟨denote e, h⟩

/-- Multiplying by a literal distributes over a sum.

    (c : ℤ)                                          [a whole-number factor]
    (a b : Expr n)                                   [two expressions]
    (ρ : Fin n → ℤ)                                  [an assignment]
    Row.value ρ (denote                              [the row a multiple of a sum turns into]
      (Expr.scale c (Expr.add a b)))                 [the multiple of the sum]
      = Row.value ρ (denote                          [is worth the same as the row]
        (Expr.add (Expr.scale c a) (Expr.scale c b))) [of the sum of the two multiples]

    So: `../song/src/prover/face/form.slang`'s `law fragment_scale_linear`, over the model. The
    only multiplication that reaches a row is by a literal, and it stays linear. -/
theorem scale_distributes (c : ℤ) (a b : Expr n) (ρ : Fin n → ℤ) :
    Row.value ρ (denote (Expr.scale c (Expr.add a b)))
      = Row.value ρ (denote (Expr.add (Expr.scale c a) (Expr.scale c b))) := by
  simp [denote]
  ring

/-- The same opaque column on both sides of a row cancels.

    (op : ℕ) (c : Fin n)                             [an operator tag, and its column]
    (x y r : Expr n)                                 [two operands, and the rest of the row]
    (ρ : Fin n → ℤ)                                  [an assignment]
    Row.value ρ (denote (Expr.sub                    [the row this difference turns into]
      (Expr.add (Expr.opaqueCol op c x y) r)         [one operand order, added to the rest]
      (Expr.opaqueCol op c y x)))                    [minus the other operand order]
      = Row.value ρ (denote r)                       [is worth what the rest alone is worth]

    So: `../song/src/prover/face/form.slang`'s `law fragment_monomial_consistent`, over the model.
    `monomialFind` matches an unordered pair, so `x * y` and `y * x` take the same column, and the
    cancellation is then an identity of rows rather than a fact about multiplication. It holds
    whatever `x` and `y` are, and for every reading of the operator. -/
theorem monomial_cancels (op : ℕ) (c : Fin n) (x y r : Expr n) (ρ : Fin n → ℤ) :
    Row.value ρ
        (denote (Expr.sub (Expr.add (Expr.opaqueCol op c x y) r) (Expr.opaqueCol op c y x)))
      = Row.value ρ (denote r) := by
  simp [denote]

/-- The comparisons a law or spec row may use.

    Cmp                                      [a comparison]
    le                                       [is at most]
    lt                                       [is below]
    eq                                       [is equal to]
    ge                                       [is at least]
    gt                                       [is above] -/
inductive Cmp : Type
  /-- Is at most. -/
  | le
  /-- Is below. -/
  | lt
  /-- Is equal to. -/
  | eq
  /-- Is at least. -/
  | ge
  /-- Is above. -/
  | gt

/-- What a comparison asserts about two numbers.

    Cmp → ℤ → ℤ → Prop                       [takes a comparison and two numbers, gives a claim] -/
def Cmp.holds : Cmp → ℤ → ℤ → Prop
  | .le, x, y => x ≤ y
  | .lt, x, y => x < y
  | .eq, x, y => x = y
  | .ge, x, y => y ≤ x
  | .gt, x, y => y < x

/-- The rows a comparison lowers to.

    Cmp → Expr n → Expr n → List (Row n)     [takes a comparison and two sides, gives the rows]

    So: an equality splits into two rows, which is `ncone0`'s `e` line split at load time. A strict
    comparison moves the constant by one, which is sound because the columns hold whole numbers. -/
def Cmp.rows : Cmp → Expr n → Expr n → List (Row n)
  | .le, a, b => [(denote b).sub (denote a)]
  | .lt, a, b => [((denote b).sub (denote a)).shift (-1)]
  | .eq, a, b => [(denote b).sub (denote a), (denote a).sub (denote b)]
  | .ge, a, b => [(denote a).sub (denote b)]
  | .gt, a, b => [((denote a).sub (denote b)).shift (-1)]

/-- The rows a comparison lowers to hold exactly when the comparison does.

    (μ : ℕ → ℤ → ℤ → ℤ)                              [what each nonlinear operator tag does]
    (ρ : Fin n → ℤ)                                  [an assignment to the columns]
    (cmp : Cmp) (a b : Expr n)                       [a comparison and its two sides]
    (ha : Keyed μ ρ a) (hb : Keyed μ ρ b)            [both sides keyed at this assignment]
    (∀ r ∈ Cmp.rows cmp a b, Row.Holds ρ r)          [every row it lowers to holds]
    ↔                                                [exactly when, true in both directions]
    Cmp.holds cmp (Expr.eval μ ρ a) (Expr.eval μ ρ b) [the comparison holds of the two values]

    So: the lowering is faithful, so a decision about the rows is a decision about the comparison.
    The strict cases use that the columns hold whole numbers, where `x < y` and `y - x - 1 >= 0`
    are the same statement. -/
theorem rows_iff (μ : ℕ → ℤ → ℤ → ℤ) (ρ : Fin n → ℤ) (cmp : Cmp) (a b : Expr n)
    (ha : Keyed μ ρ a) (hb : Keyed μ ρ b) :
    (∀ r ∈ Cmp.rows cmp a b, Row.Holds ρ r)
      ↔ Cmp.holds cmp (Expr.eval μ ρ a) (Expr.eval μ ρ b) := by
  have hA := denote_eval μ ρ a ha
  have hB := denote_eval μ ρ b hb
  cases cmp <;>
    simp only [Cmp.rows, Cmp.holds, Row.Holds, List.mem_cons, List.not_mem_nil, or_false,
      forall_eq_or_imp, forall_eq, Row.value_sub, Row.value_shift, hA, hB] <;>
    omega

/-- A nonnegative-weighted sum of rows.

    List (ℤ × Row n) → Row n                 [takes weighted rows, gives their combination]

    So: `../song/src/prover/cone/certcheck.slang`'s `c` op iterated, which is the certificate. -/
def combo : List (ℤ × Row n) → Row n
  | [] => Row.ofConst 0
  | (c, r) :: rest => (r.scale c).add (combo rest)

/-- Multiplying a row that holds by a nonnegative number gives a row that holds.

    {ρ : Fin n → ℤ} {r : Row n} {c : ℤ}      [an assignment, a row, a factor]
    (hc : 0 ≤ c)                             [the factor is at least zero]
    (h : Row.Holds ρ r)                      [the row holds there]
    Row.Holds ρ (Row.scale r c)              [so does the multiplied row] -/
theorem holds_scale {ρ : Fin n → ℤ} {r : Row n} {c : ℤ} (hc : 0 ≤ c) (h : Row.Holds ρ r) :
    Row.Holds ρ (Row.scale r c) := by
  simpa [Row.Holds, Row.value_scale] using mul_nonneg hc h

/-- Adding two rows that hold gives a row that holds.

    {ρ : Fin n → ℤ} {r s : Row n}            [an assignment and two rows]
    (hr : Row.Holds ρ r) (hs : Row.Holds ρ s) [both rows hold there]
    Row.Holds ρ (Row.add r s)                [so does their sum] -/
theorem holds_add {ρ : Fin n → ℤ} {r s : Row n} (hr : Row.Holds ρ r) (hs : Row.Holds ρ s) :
    Row.Holds ρ (Row.add r s) := by
  simpa [Row.Holds, Row.value_add] using add_nonneg hr hs

/-- A nonnegative-weighted sum of rows that hold is a row that holds.

    (ρ : Fin n → ℤ)                          [an assignment]
    (ws : List (ℤ × Row n))                  [weighted rows]
    (hw : ∀ p ∈ ws, 0 ≤ p.1)                 [every weight is at least zero]
    (hr : ∀ p ∈ ws, Row.Holds ρ p.2)         [every row holds at that assignment]
    Row.Holds ρ (combo ws)                   [the combination holds too]

    So: `../song/src/prover/cone/arith.slang`'s `combovalid_nonneg` and `farkas_bound` in one
    statement, over the model rather than over a memory array. -/
theorem combo_holds (ρ : Fin n → ℤ) : ∀ ws : List (ℤ × Row n),
    (∀ p ∈ ws, 0 ≤ p.1) → (∀ p ∈ ws, Row.Holds ρ p.2) → Row.Holds ρ (combo ws)
  | [], _, _ => by simp [combo, Row.Holds]
  | (c, r) :: rest, hw, hr => by
      refine holds_add (holds_scale (hw (c, r) (by simp)) (hr (c, r) (by simp))) ?_
      exact combo_holds ρ rest (fun p hp => hw p (by simp [hp])) (fun p hp => hr p (by simp [hp]))

/-- A combination that comes out a negative constant refutes the system.

    (rs : List (Row n))                              [the rows of a system]
    (ws : List (ℤ × Row n))                          [weighted rows drawn from it]
    (hw : ∀ p ∈ ws, 0 ≤ p.1)                         [every weight is at least zero]
    (hm : ∀ p ∈ ws, p.2 ∈ rs)                        [every weighted row is one of the system's]
    (hz : ∀ i, (combo ws).coeff i = 0)               [the combination has no columns left]
    (hn : (combo ws).const < 0)                      [and its constant is below zero]
    ¬ ∃ ρ : Fin n → ℤ, ∀ r ∈ rs, Row.Holds ρ r       [no assignment satisfies the system]

    So: Farkas' lemma as `certcheck` uses it. The `c` op accumulates the combination and the `x`
    op finds it impossible, and this is why finding it settles the system rather than one row. -/
theorem unsat_of_negative_combo (rs : List (Row n)) (ws : List (ℤ × Row n))
    (hw : ∀ p ∈ ws, 0 ≤ p.1) (hm : ∀ p ∈ ws, p.2 ∈ rs)
    (hz : ∀ i, (combo ws).coeff i = 0) (hn : (combo ws).const < 0) :
    ¬ ∃ ρ : Fin n → ℤ, ∀ r ∈ rs, Row.Holds ρ r := by
  rintro ⟨ρ, hρ⟩
  have h : Row.Holds ρ (combo ws) :=
    combo_holds ρ ws hw (fun p hp => hρ p.2 (hm p hp))
  rw [Row.Holds, Row.value] at h
  simp only [hz, zero_mul, Finset.sum_const_zero, add_zero] at h
  exact absurd h (not_le.2 hn)

/-- The row saying the bottom endpoint has passed the top one.

    (lo hi : Fin n)                          [the columns holding the two endpoints]
    : Row n                                  [the row `lo - hi - 1 >= 0`]

    So: `ncone0`'s crossing, as one row of the same format as every other. -/
def crossingRow (lo hi : Fin n) : Row n :=
  ((Row.ofCol lo).sub (Row.ofCol hi)).shift (-1)

/-- The crossing row holds exactly when the endpoints have passed each other.

    (lo hi : Fin n)                                  [the columns holding the two endpoints]
    (ρ : Fin n → ℤ)                                  [an assignment]
    Row.Holds ρ (crossingRow lo hi)                  [the crossing row holds]
    ↔                                                [exactly when, true in both directions]
    ρ hi < ρ lo                                      [the top endpoint has fallen below the bottom]

    So: the verdict is a row of the fragment, with no product in it. -/
theorem crossingRow_holds_iff (lo hi : Fin n) (ρ : Fin n → ℤ) :
    Row.Holds ρ (crossingRow lo hi) ↔ ρ hi < ρ lo := by
  simp only [crossingRow, Row.Holds, Row.value_shift, Row.value_sub, Row.value_ofCol]
  omega

/-- Every whole number between two, as a finite collection.

    (a b : ℤ)                                [a bottom and a top]
    : Finset ℤ                               [the whole numbers from a up to b, finitely many]

    So: a declared cap's range, built from `Finset.range` so that the search below runs. -/
def colRange (a b : ℤ) : Finset ℤ :=
  (Finset.range (b + 1 - a).toNat).map
    (⟨fun k : ℕ => a + (k : ℤ), by intro x y h; simp only at h; omega⟩ : ℕ ↪ ℤ)

/-- A number is in a cap's range exactly when it sits between the caps.

    {a b x : ℤ}                              [a bottom, a top, and a number]
    x ∈ colRange a b                         [the number is one of the range's]
    ↔                                        [exactly when, true in both directions]
    a ≤ x ∧ x ≤ b                            [it sits between the two caps] -/
theorem mem_colRange {a b x : ℤ} : x ∈ colRange a b ↔ a ≤ x ∧ x ≤ b := by
  simp only [colRange, Finset.mem_map, Finset.mem_range, Function.Embedding.coeFn_mk]
  constructor
  · rintro ⟨k, hk, rfl⟩
    omega
  · rintro ⟨h1, h2⟩
    exact ⟨(x - a).toNat, by omega, by omega⟩

/-- Every column sits between its declared caps.

    (lo hi : Fin n → ℤ)                      [a bottom and a top cap for each column]
    (ρ : Fin n → ℤ)                          [an assignment]
    : Prop                                   [a statement that is true or false]

    So: the caps a `given` row declares, which `facec` requires of every column it emits. -/
def Box (lo hi : Fin n → ℤ) (ρ : Fin n → ℤ) : Prop := ∀ i, lo i ≤ ρ i ∧ ρ i ≤ hi i

/-- Inside declared caps, satisfiability is a finite search.

    (lo hi : Fin n → ℤ)                                      [a bottom and a top cap per column]
    (rs : List (Row n))                                      [the rows of a system]
    (∃ ρ : Fin n → ℤ, Box lo hi ρ ∧ ∀ r ∈ rs, Row.Holds ρ r) [some capped assignment satisfies it]
    ↔                                                        [exactly when, true in both directions]
    (∃ ρ ∈ Fintype.piFinset fun i => colRange (lo i) (hi i), [some assignment of the finite box]
      ∀ r ∈ rs, Row.Holds ρ r)                               [satisfies it]

    So: the caps turn the search space into a finite product of finite ranges, and membership in it
    is exactly the cap condition. -/
theorem sat_iff_finite_search (lo hi : Fin n → ℤ) (rs : List (Row n)) :
    (∃ ρ : Fin n → ℤ, Box lo hi ρ ∧ ∀ r ∈ rs, Row.Holds ρ r)
      ↔ (∃ ρ ∈ Fintype.piFinset fun i => colRange (lo i) (hi i),
          ∀ r ∈ rs, Row.Holds ρ r) := by
  constructor
  · rintro ⟨ρ, hb, hr⟩
    exact ⟨ρ, Fintype.mem_piFinset.2 fun i => mem_colRange.2 (hb i), hr⟩
  · rintro ⟨ρ, hm, hr⟩
    exact ⟨ρ, fun i => mem_colRange.1 (Fintype.mem_piFinset.1 hm i), hr⟩

/-- Inside declared caps, satisfiability is decidable.

    (lo hi : Fin n → ℤ)                                      [a bottom and a top cap per column]
    (rs : List (Row n))                                      [the rows of a system]
    Decidable                                                [this is a check, not just a statement]
      (∃ ρ : Fin n → ℤ, Box lo hi ρ ∧ ∀ r ∈ rs, Row.Holds ρ r) [some capped assignment satisfies it]

    So: the bounded fragment `facec` emits and `ncone0` runs under `PROP_CAP` is decidable, by
    search. The unbounded fragment is Presburger's and is not settled here. -/
instance boxSatDecidable (lo hi : Fin n → ℤ) (rs : List (Row n)) :
    Decidable (∃ ρ : Fin n → ℤ, Box lo hi ρ ∧ ∀ r ∈ rs, Row.Holds ρ r) :=
  decidable_of_iff _ (sat_iff_finite_search lo hi rs).symm

/-- Eliminating a column whose coefficient is one loses nothing over the whole numbers.

    (L U : Finset ℤ)                                 [the lower bounds, and the upper bounds]
    (hL : L.Nonempty)                                [there is at least one lower bound]
    (∃ x : ℤ, (∀ l ∈ L, l ≤ x) ∧ (∀ u ∈ U, x ≤ u))   [some whole number sits between them]
    ↔                                                [exactly when, true in both directions]
    (∀ l ∈ L, ∀ u ∈ U, l ≤ u)                        [every lower bound is at most every upper one]

    So: the exact case of Fourier-Motzkin elimination, which is `ncone0`'s phase two at a unit
    coefficient. The witness is the largest lower bound, and it is a whole number because the
    bounds are, which is why no rounding enters. At a coefficient other than one the equivalence
    needs the integer tightening `certcheck`'s `t` op performs. -/
theorem unit_elimination_exact (L U : Finset ℤ) (hL : L.Nonempty) :
    (∃ x : ℤ, (∀ l ∈ L, l ≤ x) ∧ (∀ u ∈ U, x ≤ u)) ↔ (∀ l ∈ L, ∀ u ∈ U, l ≤ u) := by
  constructor
  · rintro ⟨x, hlx, hxu⟩ l hl u hu
    exact le_trans (hlx l hl) (hxu u hu)
  · intro h
    exact ⟨L.max' hL, fun l hl => L.le_max' l hl, fun u hu => h _ (L.max'_mem hL) u hu⟩

/-- An even multiple never lands on an odd number.

    ¬ ∃ x : ℤ, 2 * x = 1                     [no whole number doubles to one]

    So: why `unit_elimination_exact` carries its unit coefficient rather than dropping it. Over the
    fractions the same system has a solution, so an elimination step at a coefficient other than
    one has to round, which is the cutting plane `certcheck`'s `t` op applies. -/
theorem no_integer_halves_one : ¬ ∃ x : ℤ, 2 * x = 1 := by
  rintro ⟨x, hx⟩
  omega

-- ## The edges to the trichotomy

/-- A two-sided cap needs two channels, and two channels exist exactly above zero.

    (σ : ℝ)                                          [the parameter]
    Channel σ (upper σ) ∧ Channel σ (lower σ)        [both readings are channels]
    ∧ upper σ ≠ lower σ                              [and they are different readings]
    ↔                                                [exactly when, true in both directions]
    0 < σ                                            [σ is above zero]

    So: `two_channels_iff_disc_pos` read on the two readings a cap uses. A row states one order
    and a declared cap states two, one from each side, so the row format lives on the face that
    has two independent readings and on no other. `../song` fixes the scale at `σ = 1`. -/
theorem two_caps_iff_disc_pos (σ : ℝ) :
    (Channel σ (upper σ) ∧ Channel σ (lower σ) ∧ upper σ ≠ lower σ) ↔ 0 < σ := by
  constructor
  · rintro ⟨hu, hl, hne⟩
    exact (two_channels_iff_disc_pos σ).mp ⟨upper σ, lower σ, hu, hl, hne⟩
  · intro hσ
    refine ⟨channel_upper hσ.le, channel_lower hσ.le, ?_⟩
    intro h
    have hz : upper σ ((0 : ℝ), (1 : ℝ)) = lower σ ((0 : ℝ), (1 : ℝ)) := congrFun h _
    have hs : 0 < Real.sqrt σ := Real.sqrt_pos.mpr hσ
    simp only [upper, lower] at hz
    norm_num at hz
    linarith

/-- A row holding is the order the null channel carries, definable in the ring.

    (ρ : Fin n → ℤ)                                  [an assignment to the columns]
    (r : Row n)                                      [a row]
    Row.Holds ρ r                                    [the row holds there]
    ↔                                                [exactly when, true in both directions]
    ∃ w x y z : ℤ, Row.value ρ r                     [four whole numbers exist whose squares]
      = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2                [add up to the row's value]

    So: `int_le_iff_exists_sum_four_sq` at zero. What a row asserts is not an extra predicate
    bolted to the fragment, it is the channel's own order written in the ring's operations, which
    is why the assertion stays inside the language the decider works in. Over the reals the same
    fact needs one square, `le_iff_exists_sq`. -/
theorem holds_iff_four_squares (ρ : Fin n → ℤ) (r : Row n) :
    Row.Holds ρ r ↔ ∃ w x y z : ℤ, Row.value ρ r = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2 := by
  rw [Row.Holds, int_le_iff_exists_sum_four_sq 0 (Row.value ρ r)]
  simp

/-- A row holding is a cell of the algebra not crossing.

    (ρ : Fin n → ℤ)                                       [an assignment to the columns]
    (r : Row n)                                           [a row]
    Row.Holds ρ r                                         [the row holds there]
    ↔                                                     [exactly when, true in both directions]
    ¬ Crossing 1 (ofBounds 0 (Row.value ρ r))             [that cell holds its endpoints in order]

    So: `crossing_ofBounds` with the bottom endpoint at zero. The row format is the algebra's
    channel order and not a separate notion of assertion. -/
theorem holds_iff_not_crossing (ρ : Fin n → ℤ) (r : Row n) :
    Row.Holds ρ r ↔ ¬ Crossing 1 (ofBounds 0 (Row.value ρ r)) := by
  rw [crossing_ofBounds]
  simp only [Row.Holds, Int.cast_zero, not_lt]
  constructor
  · intro h; exact_mod_cast h
  · intro h; exact_mod_cast h

/-- The verdict row is the crossing of the algebra.

    (lo hi : Fin n)                                       [the columns holding the two endpoints]
    (ρ : Fin n → ℤ)                                       [an assignment]
    Row.Holds ρ (crossingRow lo hi)                       [the crossing row holds]
    ↔                                                     [exactly when, true in both directions]
    Crossing 1 (ofBounds (ρ lo) (ρ hi))                   [that cell has its endpoints passed]

    So: `crossing_ofBounds` again, now on the verdict itself. The unsatisfiability report is one
    row of the same format as every other row, so reporting it takes the decider nowhere new. -/
theorem crossingRow_is_the_crossing (lo hi : Fin n) (ρ : Fin n → ℤ) :
    Row.Holds ρ (crossingRow lo hi) ↔ Crossing 1 (ofBounds (ρ lo) (ρ hi)) := by
  rw [crossingRow_holds_iff, crossing_ofBounds]
  constructor
  · intro h; exact_mod_cast h
  · intro h; exact_mod_cast h

/-- The cone is not a row.

    {i j : Fin n}                                         [two columns]
    (hij : i ≠ j)                                         [they are different columns]
    ¬ ∃ r : Row n, ∀ ρ : Fin n → ℤ,                       [no row agrees at every assignment]
      ((Row.value ρ r : ℤ) : ℝ)                           [with the size of the cell]
        = normForm 1 (ofBounds (ρ i) (ρ j))               [built from those two columns]

    So: `normForm_ofBounds` says that size is four times the endpoints multiplied, and
    `scaled_product_not_a_row` says no row is a multiple of a product. The cone is quadratic and
    the fragment is not, which is the precise sense in which it escapes. -/
theorem norm_not_a_row {i j : Fin n} (hij : i ≠ j) :
    ¬ ∃ r : Row n, ∀ ρ : Fin n → ℤ,
      ((Row.value ρ r : ℤ) : ℝ) = normForm 1 (ofBounds (ρ i) (ρ j)) := by
  rintro ⟨r, hr⟩
  refine Row.scaled_product_not_a_row hij (k := 4) (by norm_num) ⟨r, fun ρ => ?_⟩
  have h := hr ρ
  rw [normForm_ofBounds] at h
  have h2 : Row.value ρ r = 4 * ρ i * ρ j := by exact_mod_cast h
  rw [h2]
  ring

/-- The verdict is a row and the cone is not, so the verdict never needs the cone.

    (lo hi : Fin n)                                       [the columns holding the two endpoints]
    (hij : lo ≠ hi)                                       [they are different columns]
    (∀ ρ : Fin n → ℤ, Row.Holds ρ (crossingRow lo hi)     [the crossing row decides]
      ↔ Crossing 1 (ofBounds (ρ lo) (ρ hi)))              [the crossing, at every assignment]
    ∧ ¬ ∃ r : Row n, ∀ ρ : Fin n → ℤ,                     [and no row agrees at every assignment]
      ((Row.value ρ r : ℤ) : ℝ)                           [with the size of the cell]
        = normForm 1 (ofBounds (ρ lo) (ρ hi))             [built from those two columns]

    So: `crossing_independent_of_cone` says the crossing and the cone are different loci, and this
    is the consequence for the decider. The verdict is stated by a row of the fragment, and the
    quantity that is outside the fragment is one the verdict never reads. `../song`'s
    `plans/REFLECT.md` states the same position as the modulus being verdict-irrelevant. -/
theorem verdict_needs_no_norm (lo hi : Fin n) (hij : lo ≠ hi) :
    (∀ ρ : Fin n → ℤ, Row.Holds ρ (crossingRow lo hi)
        ↔ Crossing 1 (ofBounds (ρ lo) (ρ hi))) ∧
      ¬ ∃ r : Row n, ∀ ρ : Fin n → ℤ,
        ((Row.value ρ r : ℤ) : ℝ) = normForm 1 (ofBounds (ρ lo) (ρ hi)) :=
  ⟨fun ρ => crossingRow_is_the_crossing lo hi ρ, norm_not_a_row hij⟩

end Fragment
end Apeiron
