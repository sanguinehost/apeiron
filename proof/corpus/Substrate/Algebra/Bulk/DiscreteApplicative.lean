import Substrate.Ski.PolyIotaTermAlgebra
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Field.ZMod

/-!
# Path (ii) — Discrete Applicatives for the Polymorphic Term
@name When the Math Just Becomes Ordinary Algebra
@child If you plug in a simple combine-rule that already has an undo (like flipping a bit, which you undo by flipping again), the fancy machinery collapses to plain school algebra — and that's an honest dead end, not a magic crypto-breaker.
@scientist At any discrete applicative with an algebraic inverse (groups, fields) the bulk's apparatus reduces to straight inversion — a sharp negative result for "stationary phase in the discrete setting," and the reason crypto primitives aren't `app(B,U)` for any group/field `app`.
@engineer `bool_K_inversion` recovers `U = B xor y` (XOR self-inverse) and `zmod_K_inversion` recovers `U = B/y` in a prime field; `state_congr`/`map_state` transfer the K/S/ι structure but add no inversion mechanism over algebra.

`Polymorphic.Term α` is parametric in any binary applicative `app : α → α → α`. The continuous
setting (`α = ℝ → ℝ`, `app = pointwiseDiv`) located critical points of the lift. Path (ii) swaps
in a discrete applicative and asks: does a discrete analog of stationary phase emerge, or does the
discrete setting collapse to algebraic inversion?

Two cases:

* **α = `Bool`, app = `xor`** — a single-bit field with XOR.
* **α = `ZMod p` (prime), app = field division** — a finite-field analog of the bulk's `/`.

## What this file establishes

At a single-level K-term, both collapse to direct algebraic recovery:

* **`bool_K_inversion`** — given `B xor U = y`, the unique `U` is `B xor y` (XOR self-inverse).
* **`zmod_K_inversion`** — given `B/U = y` with `B, y ≠ 0`, the unique `U` is `B/y`.

No stationary-phase analog is needed; the applicative is its own inverse-finder.

## Conclusion

At any discrete applicative with an algebraic inverse (groups, fields), the bulk's apparatus
reduces to standard algebra — a sharp negative result for "stationary phase in the discrete
setting." The K/S/ι structure transfers (`state_congr`, `map_state`) but adds no inversion
mechanism over straight algebra.

This closes path (ii) as a candidate for novel cryptographic inversion: cryptographic primitives
(S-boxes, hash compression, AES rounds) are not `app(B, U)` for any group/field `app`, and the
trivial K-term inversion gives no recovery without already holding the inverse as the applicative.
The polymorphic Term classifies algebraic structures; it does not manufacture inversion.

## What is proved here, and what is read onto the world

* **Proved (native):** `bool_K_inversion` and `bool_K_inversion_unique` — over `Bool` with `boolXor`,
  `U := B.xor y` hits `y`, and it is the only `U` that does. By case split on three booleans.
* **Proved (Mathlib):** `zmod_K_inversion` and `zmod_K_inversion_unique` — over `ZMod p` with `p`
  prime and `zmodDiv`, `U := B/y` hits `y` for `B, y ≠ 0`, and is the only `U ≠ 0` that does. The
  field structure on `ZMod p` and the cancellation lemmas are Mathlib's.
* **Proved (native):** `K_state_bool` and `K_state_zmod`, both `rfl` — the K-term's state under an
  applicative is that applicative applied to `B` and `U`. Definitional, and the whole reason the
  two inversions are one-liners.
* **Open:** the conclusion's quantifier. "At any discrete applicative with an
  algebraic inverse (groups, fields), the bulk's apparatus reduces to standard algebra" ranges over
  all such applicatives; two are exhibited (`boolXor`, `zmodDiv`) and no statement here quantifies
  over a group or a field.
* **Open:** that "cryptographic primitives (S-boxes, hash compression, AES rounds)
  are not `app(B, U)` for any group/field `app`". No S-box, compression function or AES round
  appears in any statement, and no non-representability theorem is proved.
* **Anchor:** reading the two inversions as a *negative* result about stationary
  phase in the discrete setting. What is shown is that at a single-level K-term the inverse is
  immediate; that no discrete stationary-phase analogue exists is the reading, not a theorem —
  nothing here quantifies over candidate selection rules.

-/

noncomputable section

namespace Sanguine

namespace Polymorphic

-- ═══════════════════════════════════════════════════════════════════
-- DISCRETE APPLICATIVE 1: α = Bool, app = XOR
-- ═══════════════════════════════════════════════════════════════════

/-- The XOR applicative on `Bool`. -/
def boolXor : Bool → Bool → Bool := fun b u => b.xor u

@[simp] theorem boolXor_apply (B U : Bool) :
    boolXor B U = B.xor U := rfl

/-- The state of a K-term over `Bool` with XOR applicative is the XOR. -/
theorem K_state_bool (B U : Bool) :
    (Term.K (α := Bool) B U).state boolXor = B.xor U := rfl

/-- **Bool/XOR inversion.** XOR is self-inverse: for any `B y : Bool`,
    setting `U := B.xor y` gives `(K B U).state boolXor = y`. -/
theorem bool_K_inversion (B y : Bool) :
    (Term.K (α := Bool) B (B.xor y)).state boolXor = y := by
  rw [K_state_bool]
  cases B <;> cases y <;> rfl

/-- **Uniqueness.** If `(K B U).state boolXor = y`, then `U = B.xor y`. -/
theorem bool_K_inversion_unique (B U y : Bool) :
    (Term.K (α := Bool) B U).state boolXor = y → U = B.xor y := by
  rw [K_state_bool]
  intro h
  cases B <;> cases U <;> cases y <;> simp_all

-- ═══════════════════════════════════════════════════════════════════
-- DISCRETE APPLICATIVE 2: α = ZMod p, app = FIELD DIVISION
-- ═══════════════════════════════════════════════════════════════════

variable {p : ℕ} [Fact p.Prime]

/-- The bulk's applicative at a finite field: division in ZMod p. -/
def zmodDiv (a b : ZMod p) : ZMod p := a / b

@[simp] theorem zmodDiv_apply (a b : ZMod p) : zmodDiv a b = a / b := rfl

/-- The state of a K-term over `ZMod p` with field-division applicative
    is `B/U`. -/
theorem K_state_zmod (B U : ZMod p) :
    (Term.K (α := ZMod p) B U).state zmodDiv = B / U := rfl

/-- **ZMod p / field-division inversion.** Given `B y : ZMod p` with
    `B ≠ 0` and `y ≠ 0`, setting `U := B / y` gives
    `(K B U).state zmodDiv = y`. -/
theorem zmod_K_inversion (B y : ZMod p) (hB : B ≠ 0) (hy : y ≠ 0) :
    (Term.K (α := ZMod p) B (B / y)).state zmodDiv = y := by
  rw [K_state_zmod]
  field_simp

/-- **Uniqueness.** If `(K B U).state zmodDiv = y` with `U ≠ 0` and
    `y ≠ 0`, then `U = B/y`. -/
theorem zmod_K_inversion_unique (B U y : ZMod p) (hU : U ≠ 0) (hy : y ≠ 0) :
    (Term.K (α := ZMod p) B U).state zmodDiv = y → U = B / y := by
  rw [K_state_zmod]
  intro h
  -- h : B / U = y
  -- B = y * U (multiply both sides by U), then U = B / y (divide by y)
  have h1 : B = y * U := by
    rw [← h]; exact (div_mul_cancel₀ B hU).symm
  rw [h1, mul_comm y U, mul_div_assoc, div_self hy, mul_one]

end Polymorphic
end Sanguine
