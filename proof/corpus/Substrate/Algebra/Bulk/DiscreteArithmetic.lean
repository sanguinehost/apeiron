import Substrate.Ski.PolyIotaTermAlgebra
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Field.ZMod

/-!
# Discrete Arithmetic for LLVM Intrinsics

@name Subtracting and Comparing in One Step
@child Computers usually count one tick at a time, which is slow. Here, subtracting two numbers and checking if two things are equal each happen in a single step — no counting down a chain.
@scientist Treating integers as discrete field elements rather than Peano successor chains, `zmodSub` and `boolEq` are the K combinator's state under the subtraction and equality applicatives — O(1), not O(N) Church-numeral `pred` loops.
@engineer `boolEq`/`zmodSub` are the applicatives; `K_state_bool_eq` ((K B U).state boolEq = B == U) and `K_state_zmod_sub` ((K B U).state zmodSub = B − U) prove `Compare_eq` and `Sub` are just `K_Comb` evaluated under them.

This file proves that the Sanguine architecture can natively express
`Compare_eq` and `Sub` in O(1) time without falling back to O(N) 
Church numeral `pred` loops. 

Instead of treating integers as discrete Peano structures, we instantiate
the `Polymorphic.Term` at discrete field applicatives. The LLVM `sub`
and `Compare_eq` operations correspond exactly to the `K` combinator's
state under the respective subtraction and equality applicatives.

## What is proved here, and what is read onto the world

* **Proved (native):** `K_state_bool_eq`, `K_state_zmod_sub`, `K_state_mul`, `K_state_add` — the
  K-term's state under `boolEq` / `zmodSub` / `mulApp` / `addApp` is `B == U`, `B − U`, `B * U`,
  `B + U`. All four are `rfl`: the applicative is a parameter of `Term.state`, so instantiating it
  *is* the equation. `K_state_mul` and `K_state_add` are generic over any `Mul` / `Add` carrier.
* **Proved (native):** `cannonballRoot_eq` — `cannonballRoot = 70 ^ 2`, by `decide`. The
  25-step `addApp`/`mulApp` fold over `List.range 25` evaluates to `4900`.
* **Anchor:** every **O(1)** claim in this file, including its title. No statement
  here mentions a cost, a step count or a machine: there is no cost model, no reduction relation
  and no fuel bound in any theorem. That `mulApp`/`addApp` are "the metal's MUL/ADD, one
  instruction" is the hardware-projection reading of the four `rfl` equations, of the same status
  as reading `zmodSub` as LLVM `sub`.
* **Open:** the contrast the file is built on — that Church-numeral arithmetic
  does *not* run, "`Nat.mul 2 3` does NOT collapse at fuel 3000, measured". That is a measurement
  recorded in a comment; no theorem here or in the imports states a Church-path lower bound, so
  "O(1), not O(N)" is an assertion at both ends.
* **Anchor:** that `4900 = 70²` is the geometric root of `Σmν = 70.1 meV`. The
  file's own docstring already flags the `meV` scale as empirical input, not computed; the link
  from the cannonball sum to the neutrino sum is not stated as a theorem anywhere in this file.

-/

noncomputable section

namespace Sanguine
namespace Polymorphic

-- ═══════════════════════════════════════════════════════════════════
-- O(1) EQUALITY IN THE BOOLEAN APPLICATIVE
-- ═══════════════════════════════════════════════════════════════════

/-- The equality applicative on `Bool`. -/
def boolEq (b u : Bool) : Bool := b == u

@[simp] theorem boolEq_apply (B U : Bool) :
    boolEq B U = (B == U) := rfl

/-- **O(1) Equality.**
    The state of a K-term over `Bool` with the equality applicative
    is exactly the classical equality comparison `B == U`.
    Therefore, `Compare_eq` is simply `K_Comb` evaluated under `boolEq`. -/
theorem K_state_bool_eq (B U : Bool) :
    (Term.K (α := Bool) B U).state boolEq = (B == U) := rfl

-- ═══════════════════════════════════════════════════════════════════
-- O(1) SUBTRACTION IN THE FINITE FIELD APPLICATIVE
-- ═══════════════════════════════════════════════════════════════════

variable {p : ℕ} [Fact p.Prime]

/-- The subtraction applicative on `ZMod p`. -/
def zmodSub (a b : ZMod p) : ZMod p := a - b

@[simp] theorem zmodSub_apply (a b : ZMod p) : 
    zmodSub a b = a - b := rfl

/-- **O(1) Subtraction.**
    The state of a K-term over a finite field with the subtraction
    applicative is exactly `B - U`. 
    Therefore, LLVM `sub` is simply `K_Comb` evaluated under `zmodSub`. -/
theorem K_state_zmod_sub (B U : ZMod p) :
    (Term.K (α := ZMod p) B U).state zmodSub = B - U := rfl

-- ═══════════════════════════════════════════════════════════════════
-- O(1) MULTIPLICATION AND ADDITION — the metal's MUL/ADD, not a Church loop
-- ═══════════════════════════════════════════════════════════════════
-- The discrete applicative had `sub` and `eq` (above); it lacked **multiplication**, which is
-- exactly the operation the lift's Church-numeral path cannot run (measured: `Nat.mul 2 3` does
-- NOT collapse at fuel 3000, because Church numerals are unary — O(value) depth). These add it:
-- multiplication and addition are the `K`-form's state under the product/sum applicative — ONE
-- applicative call (the metal's MUL/ADD instruction, Axiom XI/XV: addresses are coordinates, not
-- tallies), not a reduction of `n` Church combinators. Generic over any `Mul`/`Add` carrier, so the
-- same primitive serves ℕ, ℤ, `ZMod 2ⁿ` (the wrapping machine word), or a finite field.

/-- The multiplication applicative — the metal's MUL. -/
def mulApp {R : Type*} [Mul R] (a b : R) : R := a * b

@[simp] theorem mulApp_apply {R : Type*} [Mul R] (a b : R) : mulApp a b = a * b := rfl

/-- **O(1) Multiplication.** The state of a `K`-term under the product applicative is exactly
    `B * U` — multiplication is one applicative call (HP-dispatched, the metal multiplies), not a
    Church add-loop. The product counterpart of `K_state_zmod_sub`. -/
theorem K_state_mul {R : Type*} [Mul R] (B U : R) :
    (Term.K (α := R) B U).state mulApp = B * U := rfl

/-- The addition applicative — the metal's ADD. -/
def addApp {R : Type*} [Add R] (a b : R) : R := a + b

@[simp] theorem addApp_apply {R : Type*} [Add R] (a b : R) : addApp a b = a + b := rfl

/-- **O(1) Addition.** The state of a `K`-term under the sum applicative is exactly `B + U`. -/
theorem K_state_add {R : Type*} [Add R] (B U : R) :
    (Term.K (α := R) B U).state addApp = B + U := rfl

-- ═══════════════════════════════════════════════════════════════════
-- THE CANNONBALL ROOT RUNS THROUGH THE DISCRETE APPLICATIVE
-- ═══════════════════════════════════════════════════════════════════
-- The geometric root of the neutrino prediction (`Σmν = 70.1 meV`, `TwentySixDescent`) is the
-- cannonball `∑_{k=0}^{24} k² = 70² = 4900` — the transverse cap's timelike norm. Reconnaissance
-- (`sang fidelity`) located the break: this lifts clean (Gate 1) but does NOT run under the lift's
-- unary Church-numeral arithmetic (Gate 2 — `Nat.mul` won't collapse at fuel). Folded through the
-- discrete applicative's `mulApp`/`addApp` (each O(1), the metal's op), it is 25 bounded steps and
-- reaches the root exactly — the representation that RUNS where Church breaks.

/-- The cannonball root computed through the discrete applicative: `∑_{k<25} k·k`, the square via
    `mulApp` and the accumulation via `addApp`. -/
def cannonballRoot : ℕ :=
  (List.range 25).foldl (fun acc k => addApp acc (mulApp k k)) 0

/-- **The discrete applicative computes the cannonball root: `4900 = 70²`.** Where the Church path
    breaks at fuel (`Nat.mul 2 3` ✗ at 3000, measured), the discrete `mulApp`/`addApp` fold reaches
    the geometric root of `Σmν = 70.1 meV` in 25 bounded O(1) steps. The runtime realization is the
    *generic* HP dispatch (`HardwareProjectionDispatch.dispatchHP_eq_K_state` fires **any** registered
    applicative as `(K va vb).state f`, so `mulApp` slots in beside `zmodSub`/SAT — no new mechanism,
    only the missing primitive). *(Labeled, Axiom XIII: the O(1) is the HP-dispatch reading, the same
    status as `K_state_zmod_sub`; seeding `Nat.mul → mulApp` in the lift resolver and running the fold
    end-to-end on the K=4 evaluator is the named next rung. `4900 = 70²` is the geometric root; the
    `meV` scale of `Σmν` is empirical input, not computed.)* -/
theorem cannonballRoot_eq : cannonballRoot = 70 ^ 2 := by decide

end Polymorphic
end Sanguine
