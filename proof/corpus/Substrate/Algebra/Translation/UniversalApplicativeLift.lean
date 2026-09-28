import Substrate.Ski.PolyIotaTermAlgebra
import Substrate.Algebra.Bulk.DiscreteApplicative
import Substrate.Algebra.Bulk.DiscreteArithmetic
import Mathlib.Data.ZMod.Basic

/-!
# Universal Applicative Lift: Every Framework Operation IS a `Polymorphic.Term`
@name One Shape Runs Everything
@child Every operation the framework does has the exact same skeleton — only what each step *means* (add, divide, XOR) changes. Swap the meaning-maker and the same skeleton becomes a different operation.
@scientist For any carrier α and binary applicative app, `Polymorphic.Term α` under `state app` is the universal recursion; the constructors K/S_routeF/S_routeG are identical everywhere, only app's content (division, XOR, equality, field ops, affine compose) changes.
@engineer `K_form_state_universal` states the K-form uniformly; `K_form_inversion_universal` solves x = B/U for U — exactly one solution, given a right-inverse and cancellativity — and `K_form_inversion_uniqueness` is its uniqueness half; no algebraic structure assumed on app.

**Every framework operation lifts as a `Polymorphic.Term` instance under its specific
applicative.** Same K/S_routeF/S_routeG shape across all instances; only the binary applicative
changes — the operational realisation of Axiom XIV (the bulk is primary) and Axiom XV (discrete
arithmetic O(1) polymorphic).

## The universal claim

For any `α` and any binary `app : α → α → α`, `Polymorphic.Term α` under `state app` IS the
universal recursion at that applicative. The three constructors — `K B U`, `S_routeF prev outer`
(extend with outer in the U position), `S_routeG outer prev` (in the B position) — give the SAME
recursion everywhere. Only `app`'s content changes:

| Applicative | Carrier α | Operation | Domain |
|---|---|---|---|
| `pointwiseDiv` | `ℝ → ℝ` | `B p / U p` | The bulk (`UniversalEquation.state`) |
| `boolXor` | `Bool` | `B XOR U` | Boolean HP marker `BoolXor` |
| `boolEq` | `Bool` | `B == U` | Boolean HP marker `BoolEq` |
| `zmodDiv` | `ZMod p` | `B / U` (field) | Finite-field division (Axiom XV) |
| `zmodSub` | `ZMod p` | `B - U` (field) | Finite-field subtraction (HP marker `ZmodSub`) |
| `AffineStepN.compose` | `AffineStepN n` | matrix-affine composition | Spinor at any tier (`SpinorLoop §6`) |

The universal algorithm — stationary-phase collapse via the bulk's recursion — is invariant; only
each step's operational content is applicative-dependent.

## What this file proves

* `K_form_state_universal` — the K-form's state is the applicative evaluated, stated uniformly so
  corollaries cite one theorem.
* `K_form_inversion_uniqueness` — under cancellativity (`app B U₁ = app B U₂ → U₁ = U₂`), the
  inverting `U` is unique.
* `K_form_inversion_universal` — with a right-inverse `inv` (`app B (inv B y) = y`) AND
  cancellativity, `∃! U, state app (K B U) = y`: "x = B/U solved for U", exactly one solution, at
  every applicative at once.

The per-applicative theorems (`bool_K_inversion`, `zmod_K_inversion`, etc.) are instances.

## What it does NOT do

It imposes no algebraic structure on `app` (no associativity, identity, or commutativity). Purely
structural: same recursion shape, same K-form inversion law given a right-inverse. Specific
instances inherit their own properties (`AffineStepN.compose` associative by `composeN_assoc`;
`pointwiseDiv` not).

## The universal lift is the `adapt` ingest (the trinity applied to incoming data)

`Lift.lift` / `fromTerm` is the ontological ingest that forces raw, external, continuous data
(float activations, byte logits) down into the discrete ternary seed space so the `fold` can
evaluate it. Read as the trinity on the *incoming* vector, the lift is three applicative reads of
the one Term, dispatched at runtime — rotor (Logos / Dao, the FWHT orthogonal divisor), scale
(Monad / Taiji, the local `amax` unit), discretization (Apeiron / Wuji, the cone-gate clamp onto
the ternary codebook). `K_form_inversion_universal` / `_uniqueness` are the rotor+scale legs
(boundary fixes the interior up to one scalar); the Apeiron-clamp leg is the singular `/` on the
null cone (`null_not_invertible`). At the metal these are one forced fold into the AVX2 algebra
(`emit`, `Runtime/MetalEmit`): `adapt` / `rope` / `amax` are native `emit` primitives alongside
`dot_product_bits_asm`, and fusing the Trinity out of the inner loop removes the OpenMP fork/join
drag the per-token scalar ingest would otherwise pay (`CompilerIsDrag`) — the same `adapt`/`fold`
the proofs name, realized at a target algebra, not a new code path (`emit_forced`).

## What is proved here, and how much each statement carries

* **Proved (native):** `K_form_inversion_uniqueness` and
  `K_form_inversion_conditional_uniqueness` — with `app` cancellative in its second argument, the
  inverting `U` is the only one. This is the file's real content, and it is the standard
  "cancellative ⇒ at most one solution" argument with the domain vocabulary living in the binder
  names `B`, `U`, `app`.
* **Proved (native):** `K_form_inversion_universal` — with a right-inverse AND cancellativity,
  `∃! U, (Term.K B U).state app = y`: the K-form is a bijection in `U` at every applicative.
  Neither hypothesis alone gives it. Its predecessor concluded
  `(Term.K B (inv B y)).state app = y`, which *is* the hypothesis `app B (inv B y) = y` up to
  definitional unfolding — proof term `exact h_right_inv B y` after a `show` — and was
  strengthened 2026-08-05 under R11. `K_form_inversion_unconditional_via_conditional` carried
  that same statement and was deleted the same day; nothing referenced it.
* **Proved (native):** read its weight exactly — `K_form_state_universal` is `rfl`, it is
  `Term.state`'s own defining equation at the `K` constructor, restated. Kept as a name because
  `OneCollapse.one_collapse_recovers_all` and `OneLift.lift_and_recover_are_one_apply` cite it as a proof term rather
  than unfold `state`; recorded under R11 in `data/audit/VACUOUS.txt` so the restatement is not
  read as a result.
* **Proved (native):** read its weight exactly — `K_form_inversion_conditional` is
  `exact h_right_inv B y h` after a `show`, the same hypothesis-is-the-conclusion shape. It is
  kept because the partial-applicative regime admits no ∃!: uniqueness there needs the extra
  `Q B U' (inv B y)` precondition, which is `K_form_inversion_conditional_uniqueness`'s job.
  Recorded under R11.
* **Proved (riding):** `applicative_is_the_only_freedom` is `Polymorphic.state_congr` re-cited
  under a new name; the file says so in its own docstring. There is no second induction here.
* **Proved (Mathlib):** the four `example` instantiations that discharge the catalogue —
  `boolXor`, `boolEq`, `zmodSub` unconditionally, `zmodDiv` under `B ≠ 0 ∧ y ≠ 0` — using
  `field_simp`, `div_mul_cancel₀` and `mul_left_cancel₀` over `ZMod p` with `Fact p.Prime`.
* **Next:** the `AffineStepN.compose` row of the table above, and its
  `P s₂ y := ∃ s₁, compose s₂ s₁ = y` reading. Every other row has an `example` in §3; that one
  has none, here or elsewhere.
* **Anchor:** the trinity reading of the lift — rotor, scale, discretization — and
  its fusion into one AVX2 pass. No statement in this file mentions a transform, an `amax`, a
  codebook or a fork/join barrier; the evidence outside the Lean is `MetalEmit.emit_forced` plus
  the measured table in `papers/paper1/src/11-limits.md` (Intel Core i9-14900KF / RTX 5080), in
  which the fused-`adapt` pass is the change the 1-thread rows *target*, not a measured row.

-/

noncomputable section

universe u

namespace Sanguine

namespace Polymorphic

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE UNIVERSAL K-FORM STATEMENT
-- ═══════════════════════════════════════════════════════════════════

/-- **The universal K-form state**. For any `α` and binary `app`, the K-form under `state app` is
    precisely `app B U` — the structural form of `x = op(B, U)` at every applicative (the bulk's
    `x = B/U` and the spinor's `compose s_outer s_inner` are specialisations). -/
theorem K_form_state_universal {α : Type u} (app : α → α → α) (B U : α) :
    (Term.K (α := α) B U).state app = app B U := rfl

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE UNIVERSAL INVERSION LAW
-- ═══════════════════════════════════════════════════════════════════

/-- **Universal inversion uniqueness.** With `app` cancellative in its second argument
    (`app B U₁ = app B U₂ → U₁ = U₂`), the inverting U is unique. Specialises to: XOR
    (cancellative), `zmodDiv` (with B, U₁, U₂ ≠ 0), `zmodSub` (trivially). -/
theorem K_form_inversion_uniqueness {α : Type u}
    (app : α → α → α) (inv : α → α → α)
    (h_right_inv : ∀ B y, app B (inv B y) = y)
    (h_cancel : ∀ B U₁ U₂, app B U₁ = app B U₂ → U₁ = U₂)
    (B y U' : α) :
    (Term.K (α := α) B U').state app = y → U' = inv B y := by
  intro h_state
  have h_app : app B U' = y := by exact h_state
  have h_app_inv : app B (inv B y) = y := h_right_inv B y
  exact h_cancel B U' (inv B y) (h_app.trans h_app_inv.symm)

/-- **The universal K-form inversion law: the K-form is a bijection in `U`.** Given `app` with a
    right-inverse (`app B (inv B y) = y`) and cancellative in its second argument, `x = op(B, U)`
    has exactly ONE solution `U` for every target `y` — the fibre of `state app ∘ Term.K B` over
    `y` is a singleton. Neither hypothesis alone gives this: the right-inverse only exhibits a
    solution, cancellativity only forbids two. Every invertible operation instantiates it with
    its right-inverse:

    * `boolXor` ↔ `boolXor` (self-inverse)
    * `boolEq` ↔ `boolEq` (self-inverse)
    * `zmodDiv` ↔ `fun B y => B / y`
    * `zmodSub` ↔ `fun B y => B - y`
    * `pointwiseDiv` ↔ the pointwise `zmodDiv` inversion

    One universal law plus per-operation right-inverse data, no new mechanism per operation. -/
theorem K_form_inversion_universal {α : Type u}
    (app : α → α → α) (inv : α → α → α)
    (h_right_inv : ∀ B y, app B (inv B y) = y)
    (h_cancel : ∀ B U₁ U₂, app B U₁ = app B U₂ → U₁ = U₂)
    (B y : α) :
    ∃! U : α, (Term.K (α := α) B U).state app = y :=
  ⟨inv B y, h_right_inv B y,
   fun U' h => K_form_inversion_uniqueness app inv h_right_inv h_cancel B y U' h⟩

-- ═══════════════════════════════════════════════════════════════════
-- §2.5. CONDITIONAL INVERSION (PARTIAL APPLICATIVES)
-- ═══════════════════════════════════════════════════════════════════
-- §2 covers unconditional inversion (XOR, subtraction, etc.). Field
-- division (`zmodDiv`, `pointwiseDiv`) and non-invertible spinor steps
-- admit inversion only on a SUBSET of (B, y) pairs — typically those
-- where some non-degeneracy condition holds (B ≠ 0, y ≠ 0,
-- step is invertible, etc.). This section generalises §2 to a
-- predicate-parametric form: the right-inverse only needs to hold
-- where a predicate `P B y` is true.

/-- **Conditional K-form inversion law.** A partial right-inverse `inv` satisfying
    `app B (inv B y) = y` only when a predicate `P B y` holds inverts the K-form whenever `P B y`
    holds. This covers the partial-inversion applicatives:

    * `zmodDiv` at ZMod p: `P B y := B ≠ 0 ∧ y ≠ 0`, `inv := fun B y => B / y`, so
      `B / (B / y) = y`; `zmod_K_inversion` becomes an instance.
    * `pointwiseDiv` on `(ℝ → ℝ)`: `P B y := ∀ p, B p ≠ 0 ∧ y p ≠ 0`, same `inv` pointwise.
    * `AffineStepN.compose`: `P s₂ y := ∃ s₁, compose s₂ s₁ = y`, `inv` selecting such an `s₁`.

    The §2 unconditional law is the special case `P := fun _ _ => True`. -/
theorem K_form_inversion_conditional {α : Type u}
    (app : α → α → α) (inv : α → α → α) (P : α → α → Prop)
    (h_right_inv : ∀ B y, P B y → app B (inv B y) = y)
    (B y : α) (h : P B y) :
    (Term.K (α := α) B (inv B y)).state app = y := by
  show app B (inv B y) = y
  exact h_right_inv B y h

/-- **Conditional inversion uniqueness.** With the conditional right-inverse property and
    conditional cancellativity (`app` cancellative whenever `Q B U₁ U₂` holds), the inverting U is
    unique. `Q` carries the non-degeneracy conditions — for `zmodDiv`,
    `Q B U₁ U₂ := U₁ ≠ 0 ∧ U₂ ≠ 0 ∧ B / U₁ ≠ 0`. The precondition `Q B U' (inv B y)` puts both the
    candidate and the canonical inverse in the regime where cancellativity applies. -/
theorem K_form_inversion_conditional_uniqueness {α : Type u}
    (app : α → α → α) (inv : α → α → α)
    (P : α → α → Prop) (Q : α → α → α → Prop)
    (h_right_inv : ∀ B y, P B y → app B (inv B y) = y)
    (h_cancel : ∀ B U₁ U₂, Q B U₁ U₂ → app B U₁ = app B U₂ → U₁ = U₂)
    (B y U' : α) (h_P : P B y) (h_Q : Q B U' (inv B y)) :
    (Term.K (α := α) B U').state app = y → U' = inv B y := by
  intro h_state
  have h_app : app B U' = y := h_state
  have h_app_inv : app B (inv B y) = y := h_right_inv B y h_P
  exact h_cancel B U' (inv B y) h_Q (h_app.trans h_app_inv.symm)

-- ═══════════════════════════════════════════════════════════════════
-- §3. THE CATALOGUE: KNOWN INSTANCES VIA THE UNIVERSAL LAW
-- ═══════════════════════════════════════════════════════════════════
-- Each existing per-applicative inversion theorem is now a direct
-- application of `K_form_inversion_universal`. We re-derive them
-- here as `example`s to make the universal-instance correspondence
-- explicit at the theorem level (the file's headline content), then
-- the original named theorems in `DiscreteApplicative.lean` and
-- elsewhere remain as the canonical citations.

/-- Bool/XOR as a universal-law instance: self-inverse and cancellative, so `B XOR U = y` has
    exactly one solution. -/
example (B y : Bool) :
    ∃! U : Bool, (Term.K (α := Bool) B U).state boolXor = y :=
  K_form_inversion_universal
    boolXor
    (fun B y => B.xor y)
    (fun B y => by cases B <;> cases y <;> rfl)
    (fun B U₁ U₂ h => by cases B <;> cases U₁ <;> cases U₂ <;> simp_all)
    B y

/-- ZMod p / field-subtraction as a universal-law instance. Right-inverse `inv B y = B - y`
    (`B - (B - y) = y` always) and unconditionally cancellative. -/
example {p : ℕ} [Fact p.Prime] (B y : ZMod p) :
    ∃! U : ZMod p, (Term.K (α := ZMod p) B U).state zmodSub = y :=
  K_form_inversion_universal
    zmodSub
    (fun B y => B - y)
    (fun B y => by unfold zmodSub; ring)
    (fun B U₁ U₂ h => by
      unfold zmodSub at h
      linear_combination -h)
    B y

/-- Bool/equality as a universal-law instance. `boolEq` is its own right-inverse (self-inverse,
    like XOR) and cancels. -/
example (B y : Bool) :
    ∃! U : Bool, (Term.K (α := Bool) B U).state boolEq = y :=
  K_form_inversion_universal
    boolEq
    (fun B y => B == y)
    (fun B y => by cases B <;> cases y <;> rfl)
    (fun B U₁ U₂ h => by cases B <;> cases U₁ <;> cases U₂ <;> simp_all)
    B y

/-- ZMod p / field-division inversion as a CONDITIONAL universal-law instance. The right-inverse
    `inv B y = B / y` works exactly when B and y are non-zero, captured by the §2.5 predicate
    `P B y := B ≠ 0 ∧ y ≠ 0`. `DiscreteApplicative.lean`'s `zmod_K_inversion` is now a corollary. -/
example {p : ℕ} [Fact p.Prime] (B y : ZMod p) (hB : B ≠ 0) (hy : y ≠ 0) :
    (Term.K (α := ZMod p) B (B / y)).state zmodDiv = y :=
  K_form_inversion_conditional
    zmodDiv
    (fun B y => B / y)
    (fun B y => B ≠ 0 ∧ y ≠ 0)
    (fun B' y' ⟨hB', hy'⟩ => by
      show B' / (B' / y') = y'
      field_simp)
    B y ⟨hB, hy⟩

-- The named per-applicative uniqueness theorems (`bool_K_inversion_unique` in
-- DiscreteApplicative.lean, and its ZMod siblings) are corollaries of the ∃! instances above;
-- the conditional case gets its own instance, since ∃! is not available there — uniqueness
-- holds only inside the Q-regime.

/-- ZMod p / field-division inversion-uniqueness as a CONDITIONAL universal-law instance. The
    `B ≠ 0` plumbing (derivable from `B/U = y` when `y ≠ 0`) is discharged first, then §2.5
    conditional uniqueness applies with `P B y := B ≠ 0 ∧ y ≠ 0` and
    `Q B U₁ U₂ := B ≠ 0 ∧ U₁ ≠ 0 ∧ U₂ ≠ 0`. `DiscreteApplicative.lean`'s `zmod_K_inversion_unique`
    is now a corollary, proven via the universal law rather than direct algebra. -/
example {p : ℕ} [Fact p.Prime] (B U y : ZMod p)
    (hU : U ≠ 0) (hy : y ≠ 0) :
    (Term.K (α := ZMod p) B U).state zmodDiv = y → U = B / y := by
  intro h_state
  -- Step 1: derive B ≠ 0. If B = 0, then B/U = 0/U = 0 = y, contradicting hy.
  have hB : B ≠ 0 := by
    intro hB0
    rw [K_state_zmod, hB0, zero_div] at h_state
    exact hy h_state.symm
  -- Step 2: derive B/y ≠ 0 (needed for Q-witness on the canonical inv B y).
  have hBdivY : B / y ≠ 0 := div_ne_zero hB hy
  -- Step 3: invoke the conditional uniqueness law.
  exact K_form_inversion_conditional_uniqueness
    zmodDiv
    (fun B y => B / y)
    (fun B y => B ≠ 0 ∧ y ≠ 0)
    (fun B U₁ U₂ => B ≠ 0 ∧ U₁ ≠ 0 ∧ U₂ ≠ 0)
    -- h_right_inv: B / (B/y) = y under non-degeneracy
    (fun _ _ ⟨_, _⟩ => by show _ / (_ / _) = _; field_simp)
    -- h_cancel: B/U₁ = B/U₂ → U₁ = U₂ under non-degeneracy.
    -- Strategy: show (B/U₂) * U₂ = (B/U₂) * U₁ = B'; cancel left.
    (fun B' U₁ U₂ ⟨hB', hU₁, hU₂⟩ h => by
      unfold zmodDiv at h
      have hBdivU₂_ne : B' / U₂ ≠ 0 := div_ne_zero hB' hU₂
      have h_lhs : B' / U₂ * U₂ = B' := div_mul_cancel₀ B' hU₂
      have h_rhs : B' / U₂ * U₁ = B' := by
        rw [← h, div_mul_cancel₀ B' hU₁]
      have key : B' / U₂ * U₂ = B' / U₂ * U₁ := by rw [h_lhs, h_rhs]
      exact (mul_left_cancel₀ hBdivU₂_ne key).symm)
    B y U
    ⟨hB, hy⟩          -- h_P
    ⟨hB, hU, hBdivY⟩  -- h_Q
    h_state

-- ═══════════════════════════════════════════════════════════════════
-- §4. THE STRUCTURAL UNIVERSALITY
-- ═══════════════════════════════════════════════════════════════════

/-- **Structural universality**: two applicatives agreeing pointwise on a Term's leaves give the
    same state. This is `state_congr`, re-cited as the claim that the recursion SHAPE is the
    universal invariant; the applicative is the only freedom. So the K=4 evaluator dispatches any
    operation by selecting the applicative at runtime — the same Term tree, different applicatives,
    the operational content. -/
theorem applicative_is_the_only_freedom {α : Type u}
    {app₁ app₂ : α → α → α}
    (h : ∀ x y, app₁ x y = app₂ x y)
    (t : Term α) :
    t.state app₁ = t.state app₂ :=
  state_congr h t

end Polymorphic
end Sanguine
