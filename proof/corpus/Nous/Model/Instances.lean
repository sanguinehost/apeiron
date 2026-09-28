import WordRefinement
import Nous.Model.Decomposition

/-!
# Fixed-σ instances of the canonical two-slot algebra

The canonical object `Apeiron.Trichotomy.AnyRing.ringMul`/`ringNormForm` is parameterised by a
dial `σ`. This module fixes the dial at its three concrete values over the integers, links the
executing word face at `ZMod (2 ^ 64)` to the extracted kernels through `denoteSigma`, and records
the 2-adic obstruction that keeps the hyperbolic channel split off that ring.

## Faces and their grades

| face | carrier | `σ` | declarations | grade |
|---|---|---|---|---|
| hyperbolic | `ℤ` | `1` | `hyperbolic_ringMul`, `hyperbolic_ringNormForm`, `hyperbolic_ringMul_assoc`, `hyperbolic_ringMul_comm`, `hyperbolic_ringOne_mul`, `hyperbolic_ringNormForm_mul` | proved |
| parabolic | `ℤ` | `0` | `parabolic_ringMul`, `parabolic_ringNormForm`, `parabolic_ringMul_assoc`, `parabolic_ringMul_comm`, `parabolic_ringOne_mul`, `parabolic_ringNormForm_mul` | proved |
| elliptic | `ℤ` | `-1` | `elliptic_ringMul`, `elliptic_ringNormForm`, `elliptic_ringMul_assoc`, `elliptic_ringMul_comm`, `elliptic_ringOne_mul`, `elliptic_ringNormForm_mul` | proved |
| executing word | `ZMod (2 ^ 64)` | `denoteSigma Hyperbolic/Parabolic/Elliptic` | `word_mul_denote_*`, `word_norm_denote_*` | proved |

Each `proved` row is a Lean check: the three formulas and laws are stated and closed, and each word
row rewrites the extracted `word_mul_denote`/`word_norm_denote` to the concrete dial value. The
`#guard`s at the end of each section evaluate the concrete faces on small inputs.

## Excluded: the ternary face (anchor)

The ternary face (`ZMod 3`, balanced trits) is not instantiated here. Its excluded population is
every `ZMod 3` and balanced-trit carrier; the reason is that this tree carries no extracted ternary
kernel, so there is no `word_*_denote`-style result to link and no Lean check exists. The face is an
`anchor` (a stated mapping onto a planned carrier), not a `proved` face, and is kept out rather than
named as if proved.

## The 2-adic caveat at the word face

`ringMul` and `ringNormForm` instantiate at `ZMod (2 ^ 64)`. The channel decomposition
(`channelPair`, `hyperbolic_decomposition_of_isUnit`) does not: it needs `IsUnit (2 : R)` and
`IsUnit r`, and `zmod_two_not_unit` shows `2` is not a unit in `ZMod (2 ^ 64)`. The word ring
carries no channel split, and no energy or nonnegativity of `ringNormForm` is claimed there.

## Trust boundary

No claim reaches `rustc`/LLVM, the kernel ABI or the `unsafe` container. `ringNormForm` is
indefinite over `ℤ`; the elliptic word row states the algebra at `σ = -1` only, not a root of `-1`
in `ZMod (2 ^ 64)`.
-/

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing
open apeiron_word
open Aeneas Aeneas.Std Result

section Hyperbolic

/-- The hyperbolic multiplication formula at `σ = 1` over `ℤ`: the split-complex product
`(a + b·j) * (c + d·j) = (ac + bd) + (ad + bc)·j`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem hyperbolic_ringMul (z w : RingCell ℤ) :
    ringMul (1 : ℤ) z w = (z.1 * w.1 + z.2 * w.2, z.1 * w.2 + z.2 * w.1) := by
  simp only [ringMul]
  ext <;> ring

/-- The hyperbolic norm form at `σ = 1` over `ℤ`: `N(a + b·j) = a ^ 2 - b ^ 2`, the quadratic form
of the split-complex numbers.

    RingCell     [a point: two ring elements side by side, a + b·x]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem hyperbolic_ringNormForm (z : RingCell ℤ) :
    ringNormForm (1 : ℤ) z = z.1 ^ 2 - z.2 ^ 2 := by
  simp only [ringNormForm]
  ring

/-- Associativity of the hyperbolic multiplication, the canonical `ringMul_assoc` instantiated at
`σ = 1` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem hyperbolic_ringMul_assoc (z w v : RingCell ℤ) :
    ringMul (1 : ℤ) (ringMul (1 : ℤ) z w) v = ringMul (1 : ℤ) z (ringMul (1 : ℤ) w v) :=
  ringMul_assoc (1 : ℤ) z w v

/-- Commutativity of the hyperbolic multiplication, the canonical `ringMul_comm` instantiated at
`σ = 1` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem hyperbolic_ringMul_comm (z w : RingCell ℤ) :
    ringMul (1 : ℤ) z w = ringMul (1 : ℤ) w z :=
  ringMul_comm (1 : ℤ) z w

/-- The hyperbolic unit law, the canonical `ringOne_mul` instantiated at `σ = 1` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way]
    ringOne  [the point (1, 0), which changes nothing when you multiply by it] -/
theorem hyperbolic_ringOne_mul (z : RingCell ℤ) :
    ringMul (1 : ℤ) ringOne z = z :=
  ringOne_mul (1 : ℤ) z

/-- Multiplicativity of the hyperbolic norm form, the canonical `ringNormForm_mul` instantiated at
`σ = 1` over `ℤ`: `N(z * w) = N(z) * N(w)`.

    RingCell     [a point: two ring elements side by side, a + b·x]
    ringMul      [multiply two points together, the σ way]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem hyperbolic_ringNormForm_mul (z w : RingCell ℤ) :
    ringNormForm (1 : ℤ) (ringMul (1 : ℤ) z w) =
      ringNormForm (1 : ℤ) z * ringNormForm (1 : ℤ) w :=
  ringNormForm_mul (1 : ℤ) z w

end Hyperbolic

section Parabolic

/-- The parabolic multiplication formula at `σ = 0` over `ℤ`: the dual-number product
`(a + b·ε) * (c + d·ε) = ac + (ad + bc)·ε`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem parabolic_ringMul (z w : RingCell ℤ) :
    ringMul (0 : ℤ) z w = (z.1 * w.1, z.1 * w.2 + z.2 * w.1) := by
  simp only [ringMul]
  ext <;> ring

/-- The parabolic norm form at `σ = 0` over `ℤ`: `N(a + b·ε) = a ^ 2`.

    RingCell     [a point: two ring elements side by side, a + b·x]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem parabolic_ringNormForm (z : RingCell ℤ) :
    ringNormForm (0 : ℤ) z = z.1 ^ 2 := by
  simp only [ringNormForm]
  ring

/-- Associativity of the parabolic multiplication, the canonical `ringMul_assoc` instantiated at
`σ = 0` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem parabolic_ringMul_assoc (z w v : RingCell ℤ) :
    ringMul (0 : ℤ) (ringMul (0 : ℤ) z w) v = ringMul (0 : ℤ) z (ringMul (0 : ℤ) w v) :=
  ringMul_assoc (0 : ℤ) z w v

/-- Commutativity of the parabolic multiplication, the canonical `ringMul_comm` instantiated at
`σ = 0` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem parabolic_ringMul_comm (z w : RingCell ℤ) :
    ringMul (0 : ℤ) z w = ringMul (0 : ℤ) w z :=
  ringMul_comm (0 : ℤ) z w

/-- The parabolic unit law, the canonical `ringOne_mul` instantiated at `σ = 0` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way]
    ringOne  [the point (1, 0), which changes nothing when you multiply by it] -/
theorem parabolic_ringOne_mul (z : RingCell ℤ) :
    ringMul (0 : ℤ) ringOne z = z :=
  ringOne_mul (0 : ℤ) z

/-- Multiplicativity of the parabolic norm form, the canonical `ringNormForm_mul` instantiated at
`σ = 0` over `ℤ`.

    RingCell     [a point: two ring elements side by side, a + b·x]
    ringMul      [multiply two points together, the σ way]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem parabolic_ringNormForm_mul (z w : RingCell ℤ) :
    ringNormForm (0 : ℤ) (ringMul (0 : ℤ) z w) =
      ringNormForm (0 : ℤ) z * ringNormForm (0 : ℤ) w :=
  ringNormForm_mul (0 : ℤ) z w

end Parabolic

section Elliptic

/-- The elliptic multiplication formula at `σ = -1` over `ℤ`: the complex product
`(a + b·i) * (c + d·i) = (ac - bd) + (ad + bc)·i`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem elliptic_ringMul (z w : RingCell ℤ) :
    ringMul (-1 : ℤ) z w = (z.1 * w.1 - z.2 * w.2, z.1 * w.2 + z.2 * w.1) := by
  simp only [ringMul]
  ext <;> ring

/-- The elliptic norm form at `σ = -1` over `ℤ`: `N(a + b·i) = a ^ 2 + b ^ 2`, the sum of two
squares.

    RingCell     [a point: two ring elements side by side, a + b·x]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem elliptic_ringNormForm (z : RingCell ℤ) :
    ringNormForm (-1 : ℤ) z = z.1 ^ 2 + z.2 ^ 2 := by
  simp only [ringNormForm]
  ring

/-- Associativity of the elliptic multiplication, the canonical `ringMul_assoc` instantiated at
`σ = -1` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem elliptic_ringMul_assoc (z w v : RingCell ℤ) :
    ringMul (-1 : ℤ) (ringMul (-1 : ℤ) z w) v = ringMul (-1 : ℤ) z (ringMul (-1 : ℤ) w v) :=
  ringMul_assoc (-1 : ℤ) z w v

/-- Commutativity of the elliptic multiplication, the canonical `ringMul_comm` instantiated at
`σ = -1` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way] -/
theorem elliptic_ringMul_comm (z w : RingCell ℤ) :
    ringMul (-1 : ℤ) z w = ringMul (-1 : ℤ) w z :=
  ringMul_comm (-1 : ℤ) z w

/-- The elliptic unit law, the canonical `ringOne_mul` instantiated at `σ = -1` over `ℤ`.

    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul  [multiply two points together, the σ way]
    ringOne  [the point (1, 0), which changes nothing when you multiply by it] -/
theorem elliptic_ringOne_mul (z : RingCell ℤ) :
    ringMul (-1 : ℤ) ringOne z = z :=
  ringOne_mul (-1 : ℤ) z

/-- Multiplicativity of the elliptic norm form, the canonical `ringNormForm_mul` instantiated at
`σ = -1` over `ℤ`: the two-squares identity.

    RingCell     [a point: two ring elements side by side, a + b·x]
    ringMul      [multiply two points together, the σ way]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem elliptic_ringNormForm_mul (z w : RingCell ℤ) :
    ringNormForm (-1 : ℤ) (ringMul (-1 : ℤ) z w) =
      ringNormForm (-1 : ℤ) z * ringNormForm (-1 : ℤ) w :=
  ringNormForm_mul (-1 : ℤ) z w

end Elliptic

-- The three integer faces on a small input; each row is a concrete computation.
#guard (ringMul (1 : ℤ) (3, 1) (5, 2) = (17, 11))
#guard (ringNormForm (1 : ℤ) (3, 1) = (8 : ℤ))
#guard (ringMul (0 : ℤ) (3, 1) (5, 2) = (15, 11))
#guard (ringNormForm (0 : ℤ) (3, 1) = (9 : ℤ))
#guard (ringMul (-1 : ℤ) (3, 1) (5, 2) = (13, 11))
#guard (ringNormForm (-1 : ℤ) (3, 1) = (10 : ℤ))

section WordFace

/-- The hyperbolic dial as an element of the word ring: `denoteSigma Hyperbolic = 1`. -/
theorem word_denoteSigma_hyperbolic :
    denoteSigma cell.Sigma.Hyperbolic = (1 : ZMod (2 ^ 64)) := by
  simp [denoteSigma, sigmaI64]

/-- The parabolic dial as an element of the word ring: `denoteSigma Parabolic = 0`. -/
theorem word_denoteSigma_parabolic :
    denoteSigma cell.Sigma.Parabolic = (0 : ZMod (2 ^ 64)) := by
  simp [denoteSigma, sigmaI64]

/-- The machine word `-1` denotes `-1` in the word ring.

    denote [a machine word read as its bit pattern in the word ring] -/
theorem word_denote_neg_one : denote (-1)#i64 = (-1 : ZMod (2 ^ 64)) := by
  have h : (-1)#i64 = core.num.I64.wrapping_sub 0#i64 1#i64 := by decide
  rw [h, denote_wrapping_sub, denote_zero, denote_one]
  ring

/-- The elliptic dial as an element of the word ring: `denoteSigma Elliptic = -1`. -/
theorem word_denoteSigma_elliptic :
    denoteSigma cell.Sigma.Elliptic = (-1 : ZMod (2 ^ 64)) := by
  simp only [denoteSigma, sigmaI64, word_denote_neg_one]

/-- The extracted multiplication at the hyperbolic dial refines the canonical multiplication at
`σ = 1` in the word ring.

    ringMul [multiply two points together, the σ way] -/
theorem word_mul_denote_hyperbolic (a b : cell.Cell I64) :
    (do let c ← word_mul a cell.Sigma.Hyperbolic b; ok (denoteCell c)) =
      ok (ringMul (1 : ZMod (2 ^ 64)) (denoteCell a) (denoteCell b)) := by
  rw [word_mul_denote, word_denoteSigma_hyperbolic]

/-- The extracted multiplication at the parabolic dial refines the canonical multiplication at
`σ = 0` in the word ring.

    ringMul [multiply two points together, the σ way] -/
theorem word_mul_denote_parabolic (a b : cell.Cell I64) :
    (do let c ← word_mul a cell.Sigma.Parabolic b; ok (denoteCell c)) =
      ok (ringMul (0 : ZMod (2 ^ 64)) (denoteCell a) (denoteCell b)) := by
  rw [word_mul_denote, word_denoteSigma_parabolic]

/-- The extracted multiplication at the elliptic dial refines the canonical multiplication at
`σ = -1` in the word ring.

    ringMul [multiply two points together, the σ way] -/
theorem word_mul_denote_elliptic (a b : cell.Cell I64) :
    (do let c ← word_mul a cell.Sigma.Elliptic b; ok (denoteCell c)) =
      ok (ringMul (-1 : ZMod (2 ^ 64)) (denoteCell a) (denoteCell b)) := by
  rw [word_mul_denote, word_denoteSigma_elliptic]

/-- The extracted norm form at the hyperbolic dial refines the canonical norm form at `σ = 1` in
the word ring.

    denote       [a machine word read as its bit pattern in the word ring]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem word_norm_denote_hyperbolic (a : cell.Cell I64) :
    (do let r ← word_norm a cell.Sigma.Hyperbolic; ok (denote r)) =
      ok (ringNormForm (1 : ZMod (2 ^ 64)) (denoteCell a)) := by
  rw [word_norm_denote, word_denoteSigma_hyperbolic]

/-- The extracted norm form at the parabolic dial refines the canonical norm form at `σ = 0` in
the word ring.

    denote       [a machine word read as its bit pattern in the word ring]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem word_norm_denote_parabolic (a : cell.Cell I64) :
    (do let r ← word_norm a cell.Sigma.Parabolic; ok (denote r)) =
      ok (ringNormForm (0 : ZMod (2 ^ 64)) (denoteCell a)) := by
  rw [word_norm_denote, word_denoteSigma_parabolic]

/-- The extracted norm form at the elliptic dial refines the canonical norm form at `σ = -1` in
the word ring.

    denote       [a machine word read as its bit pattern in the word ring]
    ringNormForm [the norm form: a point times its conjugate, which can come out negative] -/
theorem word_norm_denote_elliptic (a : cell.Cell I64) :
    (do let r ← word_norm a cell.Sigma.Elliptic; ok (denote r)) =
      ok (ringNormForm (-1 : ZMod (2 ^ 64)) (denoteCell a)) := by
  rw [word_norm_denote, word_denoteSigma_elliptic]

-- The three dial values in the ring the extracted arithmetic runs in, and the word-ring product.
#guard (denoteSigma cell.Sigma.Hyperbolic = (1 : ZMod (2 ^ 64)))
#guard (denoteSigma cell.Sigma.Parabolic = (0 : ZMod (2 ^ 64)))
#guard (denoteSigma cell.Sigma.Elliptic = (-1 : ZMod (2 ^ 64)))
#guard (ringMul (1 : ZMod (2 ^ 64)) (3, 1) (5, 2) = (17, 11))
#guard (ringNormForm (-1 : ZMod (2 ^ 64)) (3, 1) = (10 : ZMod (2 ^ 64)))

/-- The 2-adic obstruction at the word face: `2` is not a unit in `ZMod (2 ^ 64)`, so the channel
split's `IsUnit (2 : R)` premise fails and `hyperbolic_decomposition_of_isUnit` does not transport
to the executing ring.

    IsUnit [can be divided by, meaning something multiplies with it to give 1]
    ¬      [this is not true] -/
theorem zmod_two_not_unit : ¬ IsUnit (2 : ZMod (2 ^ 64)) := by
  simpa using ZMod.prime_natCast_not_isUnit_pow (p := 2) (d := 64) Nat.prime_two (by norm_num)

end WordFace

end Nous.Model
