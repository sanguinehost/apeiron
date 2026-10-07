import Mathlib.Tactic

/-!
# The identity floor of the bit-identical coopmat epilogue

The cooperative-matrix K-block epilogue (`src/spirv/coopmat_gemm.comp`, `#if EPI`) spends five float
issue slots per element per block. The epic (`docs/epics/resource-guided-evolution.md`, W2d spec,
"Identity floor") states the premise: the bit-identical contract preserves the rounding sequence, so an
epilogue that agrees with it bit for bit must perform each fp32 operation, and the slot count of that
epilogue is bounded below by the number of rounding operations.

This module formalises the counting half of that premise over the carrier `Int` (no Mathlib `Float`):

* a **rounding model** `rnd k : Int → Int` that rounds half-down to the lattice `2^k · ℤ` (`k ≥ 1`),
  proved idempotent (`rnd_idem`) and monotone (`rnd_mono`). It is a `model` of one fp32 rounding, not
  IEEE-754: no exponent, signs, subnormals, NaN or infinity, and the lattice is absolute rather than
  relative.
* an expression type `E` over `+`, `*` with a `rnd` node wrapping each operation, its `eval`, and
  `rcount` counting `rnd` nodes.
* `drop_changes` for nodes 1, 4 and 5: an input assignment (`W1`, `W4`, `W5`) on which deleting that
  node's `rnd` changes the value, for every setting of the other deletion bits
  (`drop1_changes`, `drop4_changes`, `drop5_changes`).
* `redundant_two`, `redundant_three`: an input-independent proof that deleting a scale-product node
  changes nothing under this model, with the cause named.
* `identity_floor`: within the family of trees obtained from `epi` by deleting `rnd` nodes, any tree
  that agrees with `epi` on every input keeps at least `3` rounding nodes, and `3` is attained
  (`tight_agrees`).

## Shader mapping

Per float op of the `#if EPI` block (`src/spirv/coopmat_gemm.comp:263-268`) against the tree:

| shader op | node | matches? |
|---|---|---|
| `intBitsToFloat(q[e]) - MAGIC_F` | 1 `rnd(var 3 + (-1))` | shape matches; the `intBitsToFloat` reinterpret and the concrete `MAGIC_F` are not modelled (a leaf plus a grid-unit constant) |
| `fd[e] * qf` (scale product) | 2 `rnd(node1 * var 4)` | shape matches; **not observable** in this model (see below) |
| `* fsx[j][e]` (scale product) | 3 `rnd(node2 * var 5)` | shape matches; **not observable** in this model (see below) |
| `fma(flo[e], fxs[j][e], dqs)` | 4 `rnd(var 1 * var 2 + node3)` | matches the fused fma (one rounding); the two SPIR-V ops `FMul`/`FAdd` of the non-fused form are one node here |
| `a[e] = a[e] + s2` (accumulate) | 5 `rnd(var 0 + node4)` | matches |

Mismatches stated, not hidden: (a) nodes 2 and 3 are unobservable under an absolute fixed grid
(`redundant_two`, `redundant_three`), so the model floor is `3`, not the shader's `5`; (b) `k = 1`
(grid `2`) is used for the concrete witnesses, a degenerate rounding in which every odd integer is a
tie; (c) the `#else` branch (`a[e] = a[e] + intBitsToFloat(q[e])`) is a different epilogue and is not
modelled; (d) the carrier is `Int`, so fp32 overflow, subnormals, signed zero and rounding-mode
effects are outside the statement.

## What this does **not** establish

The fixed-grid model cannot separate the two scale-product roundings. The reason is arithmetic, not a
gap in the proof: on a lattice `g·ℤ`, the product of an on-grid value by any integer is again on-grid,
so `rnd (rnd x * d) = rnd x * d` for every integer `d` (`redundant_two`, `redundant_three`). A
relative-precision (floating-exponent) rounding is required to make those two nodes necessary; that
model, and therefore a floor of `5`, is open. The Shader mapping above records where the model
diverges.

Axiom cone: `{propext, Classical.choice, Quot.sound}` (audited in `AxiomAudit.lean`).
-/

namespace Perf

/-! ## 1 · Rounding to a fixed grid -/

/-- The grid step `2^k` as an `Int`. For `k ≥ 1` this is `2` or more. -/
def grid (k : Nat) : Int := (2 : Int) ^ k

/-- The half-step `2^(k-1)`. For `k ≥ 1`, `grid k = 2 * half k`. -/
def half (k : Nat) : Int := (2 : Int) ^ (k - 1)

/-- Half-down rounding of `x` to the lattice `2^k·ℤ`: `grid k * ⌊(x + (half k - 1)) / grid k⌋`.
Because the divisor is positive, the Euclidean quotient `Int.ediv` on a nonnegative divisor is the
floor, so this is "round half down" (a tie goes to the smaller lattice point). This is a `model` of one
fp32 rounding: an absolute fixed grid, not IEEE-754's relative grid. -/
def rnd (k : Nat) (x : Int) : Int := grid k * ((x + (half k - 1)) / (grid k))

theorem grid_pos (k : Nat) : 0 < grid k := by unfold grid; positivity

theorem half_pos (k : Nat) : 0 < half k := by unfold half; positivity

/-- For `k ≥ 1`, `2^k = 2 · 2^(k-1)`. -/
theorem grid_eq (k : Nat) (hk : 1 ≤ k) : grid k = 2 * half k := by
  unfold grid half
  conv_lhs => rw [show k = (k - 1) + 1 by omega]
  rw [pow_succ]
  ring

/-- `0 ≤ half k - 1 < grid k` for `k ≥ 1`: the offset `half k - 1` is one grid step short of the step,
so it contributes nothing to the quotient. -/
theorem csmall (k : Nat) (hk : 1 ≤ k) : 0 ≤ half k - 1 ∧ half k - 1 < grid k := by
  have hp := half_pos k
  have hg := grid_eq k hk
  omega

/-- `rnd` is floor division by the positive grid: the Euclidean quotient equals `Int.fdiv`. -/
theorem rnd_eq_fdiv (k : Nat) (x : Int) :
    rnd k x = grid k * ((x + (half k - 1)).fdiv (grid k)) := by
  unfold rnd
  rw [Int.fdiv_eq_ediv_of_nonneg _ (le_of_lt (grid_pos k))]

/-- Floor division by a positive divisor is monotone. -/
theorem fdiv_le_fdiv {x y g : Int} (h : x ≤ y) (hg : 0 < g) : x.fdiv g ≤ y.fdiv g := by
  by_contra hc
  have hlt : y.fdiv g < x.fdiv g := lt_of_not_ge hc
  have h1 : y < (y.fdiv g + 1) * g := Int.lt_fdiv_add_one_mul_self y hg
  have h2 : (y.fdiv g + 1) * g ≤ (x.fdiv g) * g := by
    apply Int.mul_le_mul_of_nonneg_right _ (le_of_lt hg)
    omega
  have h3 : (x.fdiv g) * g ≤ x := by
    have hfm : 0 ≤ x.fmod g := Int.fmod_nonneg_of_pos x hg
    have hh := Int.fdiv_mul_add_fmod x g
    omega
  omega

/-- **`rnd` is monotone.** A rounding to a fixed grid preserves order. -/
theorem rnd_mono (k : Nat) : Monotone (rnd k) := by
  intro x y hxy
  rw [rnd_eq_fdiv, rnd_eq_fdiv]
  apply Int.mul_le_mul_of_nonneg_left _ (le_of_lt (grid_pos k))
  apply fdiv_le_fdiv _ (grid_pos k)
  omega

/-- **`rnd` is idempotent.** Rounding an already-rounded value changes nothing. -/
theorem rnd_idem (k : Nat) (hk : 1 ≤ k) (x : Int) : rnd k (rnd k x) = rnd k x := by
  rw [rnd_eq_fdiv, rnd_eq_fdiv]
  congr 1
  set m := (x + (half k - 1)).fdiv (grid k)
  have hcf : (half k - 1).fdiv (grid k) = 0 :=
    Int.fdiv_eq_zero_of_lt (csmall k hk).1 (csmall k hk).2
  have : (grid k * m + (half k - 1)).fdiv (grid k) = m := by
    rw [show grid k * m + (half k - 1) = (half k - 1) + grid k * m by ring]
    rw [Int.add_mul_fdiv_left (half k - 1) m (ne_of_gt (grid_pos k)), hcf]
    omega
  rw [this]

/-- **An on-grid value is fixed, and the lattice is closed under integer scaling.** For `k ≥ 1`,
`rnd k (grid k * t) = grid k * t`: the image of `rnd` is exactly the lattice. -/
theorem rnd_of_grid_mul (k : Nat) (hk : 1 ≤ k) (t : Int) : rnd k (grid k * t) = grid k * t := by
  rw [rnd_eq_fdiv]
  congr 1
  have hcf : (half k - 1).fdiv (grid k) = 0 :=
    Int.fdiv_eq_zero_of_lt (csmall k hk).1 (csmall k hk).2
  rw [show grid k * t + (half k - 1) = (half k - 1) + grid k * t by ring]
  rw [Int.add_mul_fdiv_left (half k - 1) t (ne_of_gt (grid_pos k)), hcf]
  omega

/-! ## 2 · Expressions with a rounding node per operation -/

/-- Expressions over `+`, `*`, integer constants and free `Int` variables, with a `rnd` node wrapping
each operation. `roundingCount` counts the `rnd` nodes. -/
inductive E where
  | var (n : Nat)
  | const (c : Int)
  | add (a b : E)
  | mul (a b : E)
  | rnd (a : E)

/-- The value of an expression at an assignment `ρ : Nat → Int` over the grid `2^k`. -/
def eval (k : Nat) : E → (Nat → Int) → Int
  | .var n, ρ => ρ n
  | .const c, _ => c
  | .add a b, ρ => eval k a ρ + eval k b ρ
  | .mul a b, ρ => eval k a ρ * eval k b ρ
  | .rnd a, ρ => rnd k (eval k a ρ)

/-- **The rounding count**: the number of `rnd` nodes. -/
def rcount : E → Nat
  | .var _ => 0
  | .const _ => 0
  | .add a b => rcount a + rcount b
  | .mul a b => rcount a + rcount b
  | .rnd a => rcount a + 1

/-! ## 3 · The epilogue-shaped tree and its node-deletion family

The five rounding nodes follow the `#if EPI` block of `src/spirv/coopmat_gemm.comp` (lines 263-268):

| node | shader op | expression |
|---|---|---|
| 1 | `intBitsToFloat(q[e]) - MAGIC_F` | `rnd((var 3) + (-1))` |
| 2 | `fd[e] * qf` (scale product) | `rnd(node1 * (var 4))` |
| 3 | `* fsx[j][e]` (scale product) | `rnd(node2 * (var 5))` |
| 4 | `fma(flo[e], fxs[j][e], dqs)` | `rnd((var 1) * (var 2) + node3)` |
| 5 | `a[e] = a[e] + s2` (accumulate) | `rnd((var 0) + node4)` |

Free variables: `0` the accumulator `a`, `1`/`2` the `lo`/`xs` factors, `3` the quantized code `q`,
`4` the scale `fd`, `5` the activation scale `fsx`. The constant `-1` models `MAGIC_F` in grid units.
`N i d p` is node `i` over the prefix `p`, with `d = true` deleting its `rnd`. -/

def N1 : Bool → E
  | true => E.add (E.var 3) (E.const (-1))
  | false => E.rnd (E.add (E.var 3) (E.const (-1)))

def N2 : Bool → E → E
  | true, p => E.mul p (E.var 4)
  | false, p => E.rnd (E.mul p (E.var 4))

def N3 : Bool → E → E
  | true, p => E.mul p (E.var 5)
  | false, p => E.rnd (E.mul p (E.var 5))

def N4 : Bool → E → E
  | true, p => E.add (E.mul (E.var 1) (E.var 2)) p
  | false, p => E.rnd (E.add (E.mul (E.var 1) (E.var 2)) p)

def N5 : Bool → E → E
  | true, p => E.add (E.var 0) p
  | false, p => E.rnd (E.add (E.var 0) p)

/-- The tree with the `rnd` of node `i` deleted exactly when `dᵢ = true`. -/
def variant (d1 d2 d3 d4 d5 : Bool) : E := N5 d5 (N4 d4 (N3 d3 (N2 d2 (N1 d1))))

/-- The identity-preserving epilogue tree: all five roundings present. -/
def epi : E := variant false false false false false

/-! ## 4 · Node 1, 4 and 5 are deletion-necessary

Three assignments witness that deleting node 1, node 4 or node 5 changes the value. The witnesses are
robust: the value at each is a closed form in the single relevant deletion bit, proved over all
`2^4` settings of the other bits. -/

/-- Witness for node 1: `q = 2`, `d = 2`, `sx = 1`, `lo = xs = a = 0`. -/
def W1 : Nat → Int
  | 0 => 0 | 1 => 0 | 2 => 0 | 3 => 2 | 4 => 2 | 5 => 1 | _ => 0

/-- Witness for node 4: `a = 1`, `lo = xs = 1`, `q = 0`, `d = sx = 1`. -/
def W4 : Nat → Int
  | 0 => 1 | 1 => 1 | 2 => 1 | 3 => 0 | 4 => 1 | 5 => 1 | _ => 0

/-- Witness for node 5: `a = 1`, `lo = xs = 0`, `q = 0`, `d = sx = 1`. -/
def W5 : Nat → Int
  | 0 => 1 | 1 => 0 | 2 => 0 | 3 => 0 | 4 => 1 | 5 => 1 | _ => 0

/-- **`drop_changes` for node 1**: at `W1`, deleting node 1 (the `MAGIC_F` subtract rounding) changes
the value, whatever the other deletion bits are. -/
theorem W1_ne (b2 b3 b4 b5 : Bool) : eval 1 (variant true b2 b3 b4 b5) W1 ≠ eval 1 epi W1 := by
  rcases b2 <;> rcases b3 <;> rcases b4 <;> rcases b5 <;>
    norm_num [epi, variant, N1, N2, N3, N4, N5, eval, rnd, grid, half, W1]

/-- At `W4`, deleting node 4 (the fma rounding) changes the value, whatever the other bits are. -/
theorem W4_ne (b1 b2 b3 b5 : Bool) : eval 1 (variant b1 b2 b3 true b5) W4 ≠ eval 1 epi W4 := by
  rcases b1 <;> rcases b2 <;> rcases b3 <;> rcases b5 <;>
    norm_num [epi, variant, N1, N2, N3, N4, N5, eval, rnd, grid, half, W4]

/-- At `W5`, deleting node 5 (the accumulate rounding) changes the value, whatever the other bits are. -/
theorem W5_ne (b1 b2 b3 b4 : Bool) : eval 1 (variant b1 b2 b3 b4 true) W5 ≠ eval 1 epi W5 := by
  rcases b1 <;> rcases b2 <;> rcases b3 <;> rcases b4 <;>
    norm_num [epi, variant, N1, N2, N3, N4, N5, eval, rnd, grid, half, W5]

/-- **`drop_changes` for node 1**: deleting the `MAGIC_F` subtract rounding changes the value. -/
theorem drop1_changes :
    ∃ ρ, eval 1 (variant true false false false false) ρ ≠ eval 1 epi ρ :=
  ⟨W1, W1_ne false false false false⟩

/-- **`drop_changes` for node 4**: deleting the fma rounding changes the value. -/
theorem drop4_changes :
    ∃ ρ, eval 1 (variant false false false true false) ρ ≠ eval 1 epi ρ :=
  ⟨W4, W4_ne false false false false⟩

/-- **`drop_changes` for node 5**: deleting the accumulate rounding changes the value. -/
theorem drop5_changes :
    ∃ ρ, eval 1 (variant false false false false true) ρ ≠ eval 1 epi ρ :=
  ⟨W5, W5_ne false false false false⟩

/-! ## 5 · Nodes 2 and 3 are redundant under the fixed-grid model

On the lattice `g·ℤ` the product of an on-grid value by any integer is on-grid, so a `rnd` node whose
argument is `(on-grid) * (var)` is the identity for every input. Nodes 2 and 3 have exactly that shape
(node 1's output and node 2's output are on-grid), so deleting them changes nothing. This is the
arithmetic reason the fixed-grid model cannot see the two scale-product roundings. -/

/-- A `rnd` node over `(on-grid) * d` is the identity. -/
theorem N2_irrel (p : E) (ρ : Nat → Int) (z : Int) (hz : eval 1 p ρ = grid 1 * z) :
    eval 1 (N2 true p) ρ = eval 1 (N2 false p) ρ := by
  have h : rnd 1 (eval 1 p ρ * ρ 4) = eval 1 p ρ * ρ 4 := by
    rw [hz, show grid 1 * z * ρ 4 = grid 1 * (z * ρ 4) by ring,
      rnd_of_grid_mul 1 (by norm_num)]
  show eval 1 p ρ * ρ 4 = rnd 1 (eval 1 p ρ * ρ 4)
  exact h.symm

/-- A `rnd` node over `(on-grid) * sx` is the identity. -/
theorem N3_irrel (p : E) (ρ : Nat → Int) (z : Int) (hz : eval 1 p ρ = grid 1 * z) :
    eval 1 (N3 true p) ρ = eval 1 (N3 false p) ρ := by
  have h : rnd 1 (eval 1 p ρ * ρ 5) = eval 1 p ρ * ρ 5 := by
    rw [hz, show grid 1 * z * ρ 5 = grid 1 * (z * ρ 5) by ring,
      rnd_of_grid_mul 1 (by norm_num)]
  show eval 1 p ρ * ρ 5 = rnd 1 (eval 1 p ρ * ρ 5)
  exact h.symm

/-- Node 1's output is on-grid. -/
theorem N1_grid (ρ : Nat → Int) :
    eval 1 (N1 false) ρ = grid 1 * ((ρ 3 + (-1) + (half 1 - 1)) / grid 1) := rfl

/-- Node 2's output (with node 1 present) is on-grid. -/
theorem N2_grid (ρ : Nat → Int) :
    eval 1 (N2 false (N1 false)) ρ
      = grid 1 * ((eval 1 (N1 false) ρ * ρ 4 + (half 1 - 1)) / grid 1) := rfl

/-- Deleting node 2 changes nothing, for every input. -/
theorem redundant_two : ∀ ρ, eval 1 (variant false true false false false) ρ = eval 1 epi ρ := by
  intro ρ
  have h2 : eval 1 (N2 true (N1 false)) ρ = eval 1 (N2 false (N1 false)) ρ :=
    N2_irrel (N1 false) ρ _ (N1_grid ρ)
  have h3 : eval 1 (N3 false (N2 true (N1 false))) ρ
          = eval 1 (N3 false (N2 false (N1 false))) ρ := by
    show rnd 1 (eval 1 (N2 true (N1 false)) ρ * ρ 5)
       = rnd 1 (eval 1 (N2 false (N1 false)) ρ * ρ 5)
    rw [h2]
  have h4 : eval 1 (N4 false (N3 false (N2 true (N1 false)))) ρ
          = eval 1 (N4 false (N3 false (N2 false (N1 false)))) ρ := by
    show rnd 1 (ρ 1 * ρ 2 + eval 1 (N3 false (N2 true (N1 false))) ρ)
       = rnd 1 (ρ 1 * ρ 2 + eval 1 (N3 false (N2 false (N1 false))) ρ)
    rw [h3]
  have h5 : eval 1 (N5 false (N4 false (N3 false (N2 true (N1 false))))) ρ
          = eval 1 (N5 false (N4 false (N3 false (N2 false (N1 false))))) ρ := by
    show rnd 1 (ρ 0 + eval 1 (N4 false (N3 false (N2 true (N1 false)))) ρ)
       = rnd 1 (ρ 0 + eval 1 (N4 false (N3 false (N2 false (N1 false)))) ρ)
    rw [h4]
  simpa only [variant, epi] using h5

/-- Deleting node 3 changes nothing, for every input. -/
theorem redundant_three : ∀ ρ, eval 1 (variant false false true false false) ρ = eval 1 epi ρ := by
  intro ρ
  have h3 : eval 1 (N3 true (N2 false (N1 false))) ρ
          = eval 1 (N3 false (N2 false (N1 false))) ρ :=
    N3_irrel (N2 false (N1 false)) ρ _ (N2_grid ρ)
  have h4 : eval 1 (N4 false (N3 true (N2 false (N1 false)))) ρ
          = eval 1 (N4 false (N3 false (N2 false (N1 false)))) ρ := by
    show rnd 1 (ρ 1 * ρ 2 + eval 1 (N3 true (N2 false (N1 false))) ρ)
       = rnd 1 (ρ 1 * ρ 2 + eval 1 (N3 false (N2 false (N1 false))) ρ)
    rw [h3]
  have h5 : eval 1 (N5 false (N4 false (N3 true (N2 false (N1 false))))) ρ
          = eval 1 (N5 false (N4 false (N3 false (N2 false (N1 false))))) ρ := by
    show rnd 1 (ρ 0 + eval 1 (N4 false (N3 true (N2 false (N1 false)))) ρ)
       = rnd 1 (ρ 0 + eval 1 (N4 false (N3 false (N2 false (N1 false)))) ρ)
    rw [h4]
  simpa only [variant, epi] using h5

/-- **The tight variant: `3` roundings, agreeing with the identity epilogue on every input.** Deleting
both scale-product nodes keeps node 1, 4 and 5. -/
theorem tight_agrees : ∀ ρ, eval 1 (variant false true true false false) ρ = eval 1 epi ρ := by
  intro ρ
  have h2 : eval 1 (N2 true (N1 false)) ρ = eval 1 (N2 false (N1 false)) ρ :=
    N2_irrel (N1 false) ρ _ (N1_grid ρ)
  have h3 : eval 1 (N3 true (N2 true (N1 false))) ρ
          = eval 1 (N3 true (N2 false (N1 false))) ρ := by
    show eval 1 (N2 true (N1 false)) ρ * ρ 5 = eval 1 (N2 false (N1 false)) ρ * ρ 5
    rw [h2]
  have h3' : eval 1 (N3 true (N2 false (N1 false))) ρ
           = eval 1 (N3 false (N2 false (N1 false))) ρ :=
    N3_irrel (N2 false (N1 false)) ρ _ (N2_grid ρ)
  have h4 : eval 1 (N4 false (N3 true (N2 true (N1 false)))) ρ
          = eval 1 (N4 false (N3 false (N2 false (N1 false)))) ρ := by
    show rnd 1 (ρ 1 * ρ 2 + eval 1 (N3 true (N2 true (N1 false))) ρ)
       = rnd 1 (ρ 1 * ρ 2 + eval 1 (N3 false (N2 false (N1 false))) ρ)
    rw [h3, h3']
  have h5 : eval 1 (N5 false (N4 false (N3 true (N2 true (N1 false))))) ρ
          = eval 1 (N5 false (N4 false (N3 false (N2 false (N1 false))))) ρ := by
    show rnd 1 (ρ 0 + eval 1 (N4 false (N3 true (N2 true (N1 false)))) ρ)
       = rnd 1 (ρ 0 + eval 1 (N4 false (N3 false (N2 false (N1 false)))) ρ)
    rw [h4]
  simpa only [variant, epi] using h5

/-! ## 6 · The identity floor -/

/-- `b2n b = 1` when the deletion bit is set, `0` otherwise. -/
def b2n : Bool → Nat
  | true => 1
  | false => 0

theorem b2n_le_one (b : Bool) : b2n b ≤ 1 := by cases b <;> norm_num [b2n]

/-- The rounding count with only nodes 2 and 3 optionally deleted. -/
theorem rcount_tight (d2 d3 : Bool) :
    rcount (variant false d2 d3 false false) = 5 - (b2n d2 + b2n d3) := by
  cases d2 <;> cases d3 <;> rfl

/-- **The identity floor.** Any tree in the node-deletion family that agrees with the five-rounding
epilogue on every input keeps at least three rounding nodes. Nodes 1, 4 and 5 are forced present by
`W1_ne`/`W4_ne`/`W5_ne`; nodes 2 and 3 may be absent (`redundant_two`, `redundant_three`). The floor is
over node deletions only; arbitrary rewriting is not claimed. -/
theorem identity_floor (d1 d2 d3 d4 d5 : Bool)
    (h : ∀ ρ, eval 1 (variant d1 d2 d3 d4 d5) ρ = eval 1 epi ρ) :
    3 ≤ rcount (variant d1 d2 d3 d4 d5) := by
  have hd1 : d1 = false := by
    cases d1 with
    | false => rfl
    | true => exact absurd (h W1) (W1_ne d2 d3 d4 d5)
  have hd4 : d4 = false := by
    cases d4 with
    | false => rfl
    | true => exact absurd (h W4) (W4_ne d1 d2 d3 d5)
  have hd5 : d5 = false := by
    cases d5 with
    | false => rfl
    | true => exact absurd (h W5) (W5_ne d1 d2 d3 d4)
  subst hd1
  subst hd4
  subst hd5
  rw [rcount_tight]
  have h2 := b2n_le_one d2
  have h3 := b2n_le_one d3
  omega

end Perf
