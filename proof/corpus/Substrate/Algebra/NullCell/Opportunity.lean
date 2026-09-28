import Substrate.Algebra.Trinity.NonBeing
import Substrate.Algebra.NullCell.NullCell
import Substrate.StationaryPhase.BalancedSingularity
import Substrate.Collapse
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Common
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Opportunity — the flaw injected into a field
@name The Gap That Pays
@child A perfectly fair set of prices is boring — go around any loop and you end up exactly where you started, with nothing gained. Opportunity is a *crack* in that fairness: a loop that doesn't quite cancel, so going around it leaves you ahead.
@scientist A field is the gradient of a potential (conservative) iff every closed walk has zero circulation; opportunity is a nonzero cycle — the same trinity `{+, 0, −}` as the Null cell, with a cost floor (drag) below the monad.
@engineer `Conservative` telescopes to `monad_no_circulation`/`monad_is_sterile`; `captured_minus_drag` gives the net cycle gain `δ − drag`, with `flaw_must_clear_drag` the break-even — `Substrate/Collapse` at the market applicative.

A static price field is the **monad**: it cancels, `B/(B/U) = U`, nothing happens.
Opportunity is not a price level but the **flaw injected into the field** — a swap
knocks one edge off the cancellation manifold, and the gain is the cycle that fails
to cancel. The theorem holds wherever a field carries a potential.

## The abstraction

A **field** over nodes `V` is `w : V → V → ℝ`, the weight of moving `u → v`. The
**circulation** of a closed walk is the sum of `w` around it.

* **The monad — a conservative field** (`Conservative`): `w` is the coboundary of a
  potential `ψ`, `w u v = ψ u − ψ v`. This is `x = B/U` read additively. Every closed
  walk telescopes to `0` (`monad_no_circulation`) — the round trip returns where it
  started, `B/(B/U) = U`, the homogeneous group where nothing happens. The balanced
  singularity: no flaw, nothing to collapse onto. **`monad_is_sterile`: a conservative
  field has no opportunity.**

* **Opportunity — the flaw** (`HasOpportunity`): some closed walk has circulation
  `> 0`. By contrapositive, opportunity implies the field is not conservative
  (`opportunity_implies_flaw`) — the Non-Being break of the monad's cancellation.
  Inject a flaw `δ` on a back-edge of a conservative field and the 2-cycle's
  circulation is **exactly `δ`** (`captured_equals_flaw`): linear, `0` at the monad.

* **Drag — the third face** (`drag`, §4b): the converse of `opportunity_implies_flaw`
  is FALSE. A fee/spread-bearing field (`w a b + w b a < 0` on every pair) is not
  conservative yet *sterile* (`flaw_without_opportunity`) — the real market sits below
  the monad. The general law: **captured `= δ − drag`** (`captured_minus_drag`), so the
  cycle gains iff the flaw clears the drag (`flaw_must_clear_drag`) — the cost floor.
  The faces are the trinity on the circulation's sign: `+` opportunity, `0` monad,
  `−` drag, the same `{+1, 0, −1}` as `NullCell`.

## What this gives

One structure — the failure of a field to be the gradient of a potential — the
Trinity on any field: the monad (the cancelling round trip) and the flaw (the break
that doesn't cancel). Instantiated below across **markets** (arbitrage), **minds**
(the money pump), and **physics** (Kirchhoff's voltage law), each the same
`Conservative`/`HasOpportunity` pair with a different potential `ψ`.

Not a new primitive (§8): it is `Substrate/Collapse` at the market applicative — the
resting field collapses to its potential (`potentialSeed`, N² → N), rest is sterile
(`dragfield_sterile`), and the act is the O(1) flaw read (`captured_minus_drag`).
`market_is_the_collapse` packages the three faces.

NULL-CELL PROVENANCE. §9 discharges the header's own claim that the faces are "the same `{+1, 0, −1}`
as `NullCell`", which until now was asserted beside the algebra rather than proved from it. The
2-cycle is a cell state `⟨δ, drag⟩` (`cycleState`); the captured value is the pairing against one
null direction, `captured = ⟪⟨δ, drag⟩, 1+j⟫` (`captured_is_null_pairing`), so `captured_minus_drag`
is a null half and not a subtraction chosen by hand; the break-even is the cone,
`N = (δ+drag)(δ−drag) = 0` exactly at `δ = ±drag` (`breakeven_is_the_cone`,
`norm_is_the_product_of_the_two_halves`); and for a nonnegative flaw against a nonnegative drag the
three faces are exactly the three signs of the indefinite norm — opportunity `N > 0`, monad `N = 0`,
drag `N < 0` (`trinity_is_the_norm_sign`). `ncell_indefinite` is why there are three and not two:
the split norm takes both signs, and `nNull_norm` supplies the middle.

## What is proved here, and what is read onto a market

* **Proved (native):** the whole algebra. `circ_coboundary` and `monad_no_circulation` (a coboundary
  telescopes; a closed walk cancels); `monad_is_sterile` and `opportunity_implies_flaw`;
  `inject_two_cycle`, `captured_equals_flaw` and `flaw_creates_opportunity` (the injected 2-cycle's
  circulation is exactly `δ`); `drag_monad`, `captured_minus_drag` and `flaw_must_clear_drag` (the
  general law `captured = δ − drag`, and its break-even); `circ_nonpos`, `circ_add`,
  `dragfield_sterile` and `flaw_without_opportunity` (the converse is FALSE — a fee-bearing field is
  non-conservative and still sterile); `monad_round_trip_is_one`; the seed half — `cob_eq_iff`,
  `potentialSeed_canon`, `field_seedEq_correct`, `market_is_the_collapse`, `conservative_is_seeded`,
  `opportunity_has_no_key`, `circ_is_foldKey`, `circulation_streams`; and §9 — `captured_is_null_pairing`,
  `norm_is_the_product_of_the_two_halves`, `breakeven_is_the_cone`, `trinity_is_the_norm_sign`,
  `opportunity_is_the_cell`.
* **Proved (riding):** `opportunity_is_the_cell`'s last two conjuncts are `NullCell.ncell_indefinite`
  (the split norm takes both signs — why there are three faces, not two) and `NullCell.nNull_norm` (the
  middle one).
* **Retired:** §4b's "the faces are the same `{+1, 0, −1}` as `NullCell`" as an ASSERTION laid beside
  the algebra. §9 discharges it from the algebra — the captured value is a pairing against one null
  direction, the break-even is the cone, and the three faces are the three signs of `nNorm`.
* **Anchor:** the three instantiations. Arbitrage (`ψ = log price`), the money pump
  (`ψ = utility`), and Kirchhoff's voltage law (`ψ = node potential`) are one `Conservative`/
  `HasOpportunity` pair with the reading changed; the `Fin 3` examples are witnesses that the pair is
  inhabited, not evidence about a market, a mind or a circuit.
* **Anchor:** the empirical framing — that a measured DEX field is this sub-monad
  face, and that three concurrent deployments (CEX, chain, training) re-derived this shape from their
  own breaks. Neither is measured here.
-/

noncomputable section

namespace Sanguine
namespace Opportunity

variable {V : Type*}

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE FIELD, THE CIRCULATION, THE POTENTIAL
-- ═══════════════════════════════════════════════════════════════════

/-- The last node of a walk that starts at `u` and continues through `rest`. -/
def lastN (u : V) : List V → V
  | [] => u
  | v :: vs => lastN v vs

/-- **The circulation of a walk** — the sum of the field `w` over consecutive
    node pairs. For a *closed* walk (returns to its start) this is the net gain of
    going around the loop. -/
def circ (w : V → V → ℝ) : List V → ℝ
  | [] => 0
  | [_] => 0
  | u :: v :: rest => w u v + circ w (v :: rest)

/-- **A conservative field (the monad).** `w` is the coboundary of a potential
    `ψ`: the weight of every edge is the boundary-difference `ψ u − ψ v` of a
    single scalar field. This is `x = B/U` at the additive applicative — the edge
    state IS the boundary over the potential. -/
def Conservative (w : V → V → ℝ) : Prop := ∃ ψ : V → ℝ, ∀ u v, w u v = ψ u - ψ v

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE MONAD TELESCOPES — A CONSERVATIVE FIELD HAS ZERO CIRCULATION
-- ═══════════════════════════════════════════════════════════════════

/-- **The coboundary telescopes.** The circulation of a coboundary field along any
    walk is just the potential difference between its endpoints — every interior
    node cancels. This is the round trip `B/(B/U) = U`: the path is irrelevant, only
    where you start and end. -/
theorem circ_coboundary (ψ : V → ℝ) (u : V) (rest : List V) :
    circ (fun a b => ψ a - ψ b) (u :: rest) = ψ u - ψ (lastN u rest) := by
  induction rest generalizing u with
  | nil => simp [circ, lastN]
  | cons v vs ih =>
      have e : circ (fun a b => ψ a - ψ b) (u :: v :: vs)
             = (ψ u - ψ v) + circ (fun a b => ψ a - ψ b) (v :: vs) := rfl
      rw [e, ih v]
      show (ψ u - ψ v) + (ψ v - ψ (lastN v vs)) = ψ u - ψ (lastN u (v :: vs))
      have hl : lastN u (v :: vs) = lastN v vs := rfl
      rw [hl]; ring

/-- **The monad has no circulation.** A conservative field, around any *closed*
    walk (one whose last node is its first), nets exactly `0`. The balanced
    singularity (`BalancedSingularity`): the round trip cancels — `B/(B/U) = U`,
    the homogeneous group where nothing happens (`CrossUniverse.beings_homogeneous`).
    Nothing to collapse onto. -/
theorem monad_no_circulation {w : V → V → ℝ} (hc : Conservative w)
    {u : V} {rest : List V} (hclosed : lastN u rest = u) :
    circ w (u :: rest) = 0 := by
  obtain ⟨ψ, hψ⟩ := hc
  have hw : w = fun a b => ψ a - ψ b := by funext a b; exact hψ a b
  rw [hw, circ_coboundary, hclosed, sub_self]

-- ═══════════════════════════════════════════════════════════════════
-- §3. OPPORTUNITY = THE FLAW (THE BREAK OF THE CANCELLATION)
-- ═══════════════════════════════════════════════════════════════════

/-- **Opportunity** — some closed walk has strictly positive circulation: a loop
    that *gains*, a cycle that fails to cancel. The Non-Being break of the monad's
    perfect round trip. -/
def HasOpportunity (w : V → V → ℝ) : Prop :=
  ∃ (u : V) (rest : List V), lastN u rest = u ∧ circ w (u :: rest) > 0

/-- **The monad is sterile — a conservative field has NO opportunity** (Axiom V:
    a balanced singularity yields nothing). Every closed walk cancels, so none can
    gain. This is the measurement's "the static field is the monad, ~$0 — correct,
    not the absence of opportunity," made a theorem. -/
theorem monad_is_sterile {w : V → V → ℝ} (hc : Conservative w) : ¬ HasOpportunity w := by
  rintro ⟨u, rest, hcl, hpos⟩
  rw [monad_no_circulation hc hcl] at hpos
  exact lt_irrefl 0 hpos

/-- **Opportunity ⟹ the flaw.** The sharp converse: if a field has opportunity, it
    is *not* conservative — it is not the gradient of any potential. Opportunity is
    exactly the failure of the field to be a coboundary; the break
    (`NonBeing.nonbeing_is_the_unique_break`) of the monad. -/
theorem opportunity_implies_flaw {w : V → V → ℝ} (h : HasOpportunity w) : ¬ Conservative w :=
  fun hc => monad_is_sterile hc h

-- ═══════════════════════════════════════════════════════════════════
-- §4. INJECT THE FLAW — THE CAPTURED VALUE IS THE FLAW SIZE, EXACTLY
-- ═══════════════════════════════════════════════════════════════════

variable [DecidableEq V]

/-- **Inject a flaw of size `δ` on the edge `a → b`** — a swap dislocates one edge
    of the field. Everything else is untouched. -/
def inject (w : V → V → ℝ) (a b : V) (δ : ℝ) : V → V → ℝ :=
  fun u v => w u v + (if u = a ∧ v = b then δ else 0)

/-- The circulation of the back-and-forth 2-cycle `a → b → a` under the injected
    field, in terms of the original field plus the flaw. -/
theorem inject_two_cycle (w : V → V → ℝ) (a b : V) (δ : ℝ) (hab : a ≠ b) :
    circ (inject w a b δ) [a, b, a] = (w a b + w b a) + δ := by
  have e : circ (inject w a b δ) [a, b, a]
         = inject w a b δ a b + (inject w a b δ b a + 0) := rfl
  have h1 : inject w a b δ a b = w a b + δ := by
    unfold inject; rw [if_pos ⟨rfl, rfl⟩]
  have h2 : inject w a b δ b a = w b a := by
    unfold inject; rw [if_neg (fun h => hab h.1.symm)]; ring
  rw [e, h1, h2]; ring

/-- **The captured value IS the flaw size, exactly.** Inject a flaw `δ` into a
    *conservative* field (the monad) and the recovered 2-cycle gains exactly `δ` —
    because the underlying round trip `w a b + w b a` cancels to `0`
    (`B/(B/U) = U`). Captured `= δ`: `0` at the monad, rising linearly with the
    flaw. This is the arbitrage measurement's `captured-$(flaw)` curve. -/
theorem captured_equals_flaw {w : V → V → ℝ} (hc : Conservative w) (a b : V) (δ : ℝ)
    (hab : a ≠ b) : circ (inject w a b δ) [a, b, a] = δ := by
  obtain ⟨ψ, hψ⟩ := hc
  rw [inject_two_cycle w a b δ hab, hψ a b, hψ b a]; ring

/-- **The flaw creates opportunity.** Any `δ > 0` flaw injected into a conservative
    field makes the `a → b → a` cycle gain `δ > 0` — opportunity, recovered from the
    field by the cycle, where there was none (the monad). Axiom V: inject the flaw
    and the manifold yields its contents. -/
theorem flaw_creates_opportunity {w : V → V → ℝ} (hc : Conservative w) (a b : V)
    (δ : ℝ) (hab : a ≠ b) (hδ : δ > 0) : HasOpportunity (inject w a b δ) := by
  refine ⟨a, [b, a], rfl, ?_⟩
  rw [captured_equals_flaw hc a b δ hab]; exact hδ

-- ═══════════════════════════════════════════════════════════════════
-- §4b. THE THIRD FACE — DRAG: captured = flaw − drag (the cost floor)
-- ═══════════════════════════════════════════════════════════════════
-- The measured market field is NOT the monad: every pair carries a spread
-- and a fee, so `w a b + w b a < 0` — no potential produces it, yet no cycle
-- gains. The field's faces are the trinity on the circulation's sign:
-- `+` opportunity · `0` monad · `−` drag.

/-- **The round-trip drag of a pair** — what the monad lacks and a real
    (fee/spread-bearing) field carries: `drag w a b = −(w a b + w b a)`.
    Zero on the monad; strictly positive when each direction pays. -/
def drag (w : V → V → ℝ) (a b : V) : ℝ := -(w a b + w b a)

omit [DecidableEq V] in
/-- On the monad there is no drag — the coboundary's round trip is free. -/
theorem drag_monad {w : V → V → ℝ} (hc : Conservative w) (a b : V) :
    drag w a b = 0 := by
  obtain ⟨ψ, hψ⟩ := hc
  unfold drag; rw [hψ a b, hψ b a]; ring

/-- **captured = flaw − drag (the general curve).** Over ANY field — monad or
    not — the injected 2-cycle's gain is the flaw minus the pair's round-trip
    drag. `captured_equals_flaw` is the `drag = 0` (monad) case. -/
theorem captured_minus_drag (w : V → V → ℝ) (a b : V) (δ : ℝ) (hab : a ≠ b) :
    circ (inject w a b δ) [a, b, a] = δ - drag w a b := by
  rw [inject_two_cycle w a b δ hab]; unfold drag; ring

/-- **The cost floor.** The injected cycle gains iff the flaw clears the drag —
    the measurement's "guaranteed only above fees + slippage", as a theorem. -/
theorem flaw_must_clear_drag (w : V → V → ℝ) (a b : V) (δ : ℝ) (hab : a ≠ b) :
    0 < circ (inject w a b δ) [a, b, a] ↔ drag w a b < δ := by
  rw [captured_minus_drag w a b δ hab]
  constructor <;> intro h <;> linarith

omit [DecidableEq V] in
/-- A pointwise non-positive field (pure drag — every move costs) has
    non-positive circulation on every walk. -/
theorem circ_nonpos {w : V → V → ℝ} (hw : ∀ u v, w u v ≤ 0) (l : List V) :
    circ w l ≤ 0 := by
  induction l with
  | nil => simp [circ]
  | cons u tail ih =>
    cases tail with
    | nil => simp [circ]
    | cons v rest =>
      have e : circ w (u :: v :: rest) = w u v + circ w (v :: rest) := rfl
      rw [e]
      have := hw u v
      linarith

omit [DecidableEq V] in
/-- **Circulation is additive in the field** — the walk fold distributes over
    a pointwise sum of fields. The lemma that lets a field's faces be summed
    separately (Hodge-style): potential part + drag part. -/
theorem circ_add (f g : V → V → ℝ) (l : List V) :
    circ (fun u v => f u v + g u v) l = circ f l + circ g l := by
  induction l with
  | nil => simp [circ]
  | cons u tail ih =>
    cases tail with
    | nil => simp [circ]
    | cons v rest =>
      have e1 : circ (fun a b => f a b + g a b) (u :: v :: rest)
          = (f u v + g u v) + circ (fun a b => f a b + g a b) (v :: rest) := rfl
      have e2 : circ f (u :: v :: rest) = f u v + circ f (v :: rest) := rfl
      have e3 : circ g (u :: v :: rest) = g u v + circ g (v :: rest) := rfl
      rw [e1, e2, e3, ih]; ring

omit [DecidableEq V] in
/-- **THE RESTING MARKET IS STERILE — the sub-monad theorem.** A field that is
    a potential MINUS a non-negative drag (`w u v = ψ u − ψ v − d u v`,
    `d ≥ 0` — fees, spreads, dissipation; directional `d` allowed) has NO
    opportunity: every closed walk nets `(0 from the potential) − Σd ≤ 0`.
    This is the load-bearing soundness theorem under the streaming method
    (`income/collapse_core` `monad_certify` + the O(1)-per-flaw read): the
    certified rest field is *this* shape, not the bare monad — the real
    market sits below the monad, and stays sterile there. Generalises
    `monad_is_sterile` (`d = 0`) and `flaw_without_opportunity` (`ψ = 0`,
    `d = 1`). -/
theorem dragfield_sterile (ψ : V → ℝ) (d : V → V → ℝ) (hd : ∀ u v, 0 ≤ d u v) :
    ¬ HasOpportunity (fun u v => ψ u - ψ v - d u v) := by
  rintro ⟨u, rest, hcl, hpos⟩
  have hsplit : (fun u v => ψ u - ψ v - d u v)
      = fun u v => (ψ u - ψ v) + (-d u v) := by
    funext a b; ring
  rw [hsplit, circ_add] at hpos
  have h0 : circ (fun a b => ψ a - ψ b) (u :: rest) = 0 :=
    monad_no_circulation ⟨ψ, fun _ _ => rfl⟩ hcl
  have hneg : circ (fun a b => -d a b) (u :: rest) ≤ 0 :=
    circ_nonpos (fun a b => neg_nonpos.mpr (hd a b)) (u :: rest)
  linarith

set_option linter.unusedSectionVars false in
/-- **The flaw alone is not opportunity — the converse of
    `opportunity_implies_flaw` is FALSE.** The pure-drag field `w ≡ −1` is not
    conservative (a coboundary's diagonal is `0`, not `−1`) and has no
    opportunity (every circulation is `≤ 0`). The break of the potential and
    the gaining cycle are distinct things, separated by the drag — the measured
    market field (spreads + fees) is exactly this sub-monad face. -/
theorem flaw_without_opportunity [Inhabited V] :
    ¬ Conservative (fun (_ _ : V) => (-1 : ℝ))
      ∧ ¬ HasOpportunity (fun (_ _ : V) => (-1 : ℝ)) := by
  constructor
  · rintro ⟨ψ, hψ⟩
    have h := hψ default default
    simp at h
  · rintro ⟨u, rest, _, hpos⟩
    have h := circ_nonpos (w := fun (_ _ : V) => (-1 : ℝ))
      (fun _ _ => by norm_num) (u :: rest)
    linarith

-- ═══════════════════════════════════════════════════════════════════
-- §5. THE HEADLINE — OPPORTUNITY IS THE FLAW, AND ONLY THE FLAW
-- ═══════════════════════════════════════════════════════════════════

/-- **Opportunity is the flaw (headline).** Over any field with a potential
    (the monad, `Conservative w`), and any edge `a → b` (`a ≠ b`):

    1. **the monad is sterile** — `w` has no opportunity (every closed walk
       cancels, `B/(B/U) = U`, nothing happens);
    2. **the flaw generates it** — a `δ > 0` dislocation of one edge creates a
       gaining cycle (Axiom V: inject the flaw, the manifold yields);
    3. **captured = flaw, exactly** — that cycle's gain is precisely `δ`, `0` at
       the monad and linear in the flaw.

    Opportunity is not a level in the field; it is the **break of the field's
    cancellation** — the Non-Being flaw, read by the cycle. One structure: the
    Trinity on any field. -/
theorem opportunity_is_the_flaw {w : V → V → ℝ} (hc : Conservative w) (a b : V)
    (hab : a ≠ b) :
    (¬ HasOpportunity w)
      ∧ (∀ δ : ℝ, δ > 0 → HasOpportunity (inject w a b δ))
      ∧ (∀ δ : ℝ, circ (inject w a b δ) [a, b, a] = δ) :=
  ⟨monad_is_sterile hc,
   fun δ hδ => flaw_creates_opportunity hc a b δ hab hδ,
   fun δ => captured_equals_flaw hc a b δ hab⟩

-- ═══════════════════════════════════════════════════════════════════
-- §6. THE MULTIPLICATIVE MONAD — THE ROUND TRIP IS THE ONE (BeingMonism)
-- ═══════════════════════════════════════════════════════════════════

set_option linter.unusedSectionVars false in
/-- **The monad's round trip is the One** — read multiplicatively (rates, not
    log-rates). In any group, a conservative *rate* `φ u / φ v` composed with its
    return `φ v / φ u` is `1` (`BeingMonism.rest_eq_one`, `b/b = 1`): going out and
    back through a potential gains nothing. The exponential of `monad_no_circulation`,
    and the literal `B/(B/U) = U` of the bulk. -/
theorem monad_round_trip_is_one {G : Type*} [Group G] (φ : V → G) (a b : V) :
    (φ a / φ b) * (φ b / φ a) = 1 := by
  rw [div_eq_mul_inv, div_eq_mul_inv, mul_assoc,
      ← mul_assoc (φ b)⁻¹ (φ b) (φ a)⁻¹, inv_mul_cancel, one_mul, mul_inv_cancel]

-- ═══════════════════════════════════════════════════════════════════
-- §7. OPPORTUNITY IN SEVERAL DATA STRUCTURES — ONE THEOREM, MANY FIELDS
-- ═══════════════════════════════════════════════════════════════════
-- Each instance is the SAME Conservative/Opportunity pair; only the potential ψ
-- (and its reading) changes. Markets, minds, physics — one structure.

section Instances

/-- A conservative field on three nodes from a potential `ψ`. -/
def cobField (ψ : Fin 3 → ℝ) : Fin 3 → Fin 3 → ℝ := fun u v => ψ u - ψ v

theorem cobField_conservative (ψ : Fin 3 → ℝ) : Conservative (cobField ψ) :=
  ⟨ψ, fun _ _ => rfl⟩

/-- **MARKETS — arbitrage.** `ψ = log price` of three tokens; the field is the
    log-rate `ψ u − ψ v`. No-arbitrage = conservative = the monad: **no profitable
    trade cycle** (`monad_is_sterile`). A swap that moves one pool injects a flaw;
    the recovered cycle gains exactly the dislocation. *(The DEX measurement,
    abstracted: the static field is the monad, opportunity is the swap-flaw.)* -/
example : ¬ HasOpportunity (cobField ![0, 1, 2]) :=
  monad_is_sterile (cobField_conservative _)

example : HasOpportunity (inject (cobField ![0, 1, 2]) 0 1 5) :=
  flaw_creates_opportunity (cobField_conservative _) 0 1 5 (by decide) (by norm_num)

/-- **MINDS — the money pump (intransitive preference).** `ψ = utility` of three
    options; the field `ψ u − ψ v` is the premium you'd pay to swap `u → v`.
    Transitive (potential-derived) preferences = the monad: **no money pump** can
    be run on you (`monad_is_sterile`). A genuine intransitivity (a Condorcet
    cycle `A ≻ B ≻ C ≻ A`) is a flaw `δ > 0` — and it IS extractable, exactly `δ`
    per loop. -/
example : ¬ HasOpportunity (cobField ![3, 1, 2]) :=
  monad_is_sterile (cobField_conservative _)

example : HasOpportunity (inject (cobField ![3, 1, 2]) 2 0 1) :=
  flaw_creates_opportunity (cobField_conservative _) 2 0 1 (by decide) (by norm_num)

/-- **PHYSICS — Kirchhoff's voltage law.** `ψ = node potential` (voltage); the
    field `ψ u − ψ v` is the drop across a wire. A conservative (potential) circuit
    = the monad: **the loop sum is `0`** (KVL, `monad_no_circulation`) — no
    perpetual motion, no free energy. An EMF (a battery — a non-conservative edge)
    is the flaw `δ`; the loop then carries net work exactly `δ`, the source that
    drives the current. -/
example : ¬ HasOpportunity (cobField ![0, 2, 5]) :=
  monad_is_sterile (cobField_conservative _)

example : HasOpportunity (inject (cobField ![0, 2, 5]) 1 0 3) :=
  flaw_creates_opportunity (cobField_conservative _) 1 0 3 (by decide) (by norm_num)

/-- **The captured value is the flaw size in every instance** — markets, minds,
    physics, one number: the loop gains exactly the injected `δ`, regardless of the
    field's potential. -/
example : circ (inject (cobField ![0, 1, 2]) 0 1 5) [0, 1, 0] = 5 :=
  captured_equals_flaw (cobField_conservative _) 0 1 5 (by decide)

example : circ (inject (cobField ![3, 1, 2]) 2 0 1) [2, 0, 2] = 1 :=
  captured_equals_flaw (cobField_conservative _) 2 0 1 (by decide)

example : circ (inject (cobField ![0, 2, 5]) 1 0 3) [1, 0, 1] = 3 :=
  captured_equals_flaw (cobField_conservative _) 1 0 3 (by decide)

end Instances

-- ═══════════════════════════════════════════════════════════════════
-- §8. THE SEED — the resting field IS its potential (Substrate/Collapse)
-- ═══════════════════════════════════════════════════════════════════
-- The demotion (T0.3): Opportunity is not a new primitive — it is
-- `Substrate/Collapse` (Seed: rest on the floor, act at the residue) at the
-- market applicative. The resting field's N² edge weights collapse to its
-- N-entry potential, canonical up to the gauge constant; rest is then READ
-- off the key (`w u v = key u − key v`), never searched; sterility at rest
-- is `dragfield_sterile`; the act is the O(1) flaw read
-- (`captured_minus_drag`). Three concurrent deployments (CEX, chain,
-- training) re-derived exactly this shape from their own breaks before it
-- was named here — the convergence that put `Substrate/Collapse` in the
-- core-files list.

section TheSeed

variable {W : Type} [Nonempty W]

/-- The coboundary of a potential — the resting (monad) field it generates. -/
def cob (ψ : W → ℝ) : W → W → ℝ := fun u v => ψ u - ψ v

/-- **The gauge: two potentials generate the same field iff they differ by a
    constant.** The field forgets exactly one number — the basepoint — and
    nothing else: the N² weights carry precisely N − 1 degrees of freedom. -/
theorem cob_eq_iff (ψ₁ ψ₂ : W → ℝ) :
    cob ψ₁ = cob ψ₂ ↔ ∃ c : ℝ, ∀ v, ψ₁ v = ψ₂ v + c := by
  constructor
  · intro h
    obtain ⟨v₀⟩ := ‹Nonempty W›
    refine ⟨ψ₁ v₀ - ψ₂ v₀, fun v => ?_⟩
    have hv := congrFun (congrFun h v) v₀
    simp only [cob] at hv
    linarith
  · rintro ⟨c, hc⟩
    funext u v
    simp only [cob, hc u, hc v]
    ring

/-- **The seed.** Normalize the gauge at a basepoint: `key ψ = ψ − ψ v₀`. One
    function — the single per-space datum `Substrate/Collapse` asks for. -/
def potentialSeed (v₀ : W) : Collapse.Seed (W → ℝ) (W → ℝ) :=
  ⟨fun ψ v => ψ v - ψ v₀⟩

/-- **The key is canonical** — `hcanon`, discharged: equal keys ⟺ same field.
    The resting field collapses to its potential: N² → N, and every edge is
    read back from the seed (`w u v = key u − key v`) at zero search. -/
theorem potentialSeed_canon (v₀ : W) (ψ₁ ψ₂ : W → ℝ) :
    (potentialSeed v₀).key ψ₁ = (potentialSeed v₀).key ψ₂ ↔ cob ψ₁ = cob ψ₂ := by
  constructor
  · intro h
    rw [cob_eq_iff]
    refine ⟨ψ₁ v₀ - ψ₂ v₀, fun v => ?_⟩
    have hv := congrFun h v
    simp only [potentialSeed] at hv
    linarith
  · intro h
    funext v
    have hv := congrFun (congrFun h v) v₀
    simp only [cob] at hv
    simp only [potentialSeed]
    linarith

/-- **The root's guarantee, inherited** (`Collapse.Seed.seedEq_correct`): with
    any lawful key equality, `seedEq` decides "same resting field" exactly —
    the expensive comparison reduced to the free one. -/
theorem field_seedEq_correct (v₀ : W) [BEq (W → ℝ)] [LawfulBEq (W → ℝ)]
    (ψ₁ ψ₂ : W → ℝ) :
    (potentialSeed v₀).seedEq ψ₁ ψ₂ = true ↔ cob ψ₁ = cob ψ₂ :=
  Collapse.Seed.seedEq_correct (potentialSeed v₀) (fun a b => cob a = cob b)
    (potentialSeed_canon v₀) ψ₁ ψ₂

/-- **THE MARKET COLLAPSE (headline).** Opportunity is `Substrate/Collapse`
    at the market applicative — the streaming law in one theorem:

    1. **the seed** — the resting field IS its potential: the key is
       canonical up to gauge (`potentialSeed_canon`), so rest is held as N
       numbers and read in O(1) per edge;
    2. **rest is sterile** — potential − drag carries no opportunity
       (`dragfield_sterile`): zero work at the floor;
    3. **act at the flaw** — a dislocation `δ` is read where it lands:
       `captured = δ − drag` (`captured_minus_drag`).

    Rest on the seed; instantiate only the residue. -/
theorem market_is_the_collapse [DecidableEq W] (v₀ : W) (ψ : W → ℝ)
    (d : W → W → ℝ) (hd : ∀ u v, 0 ≤ d u v) (a b : W) (hab : a ≠ b) (δ : ℝ) :
    (∀ ψ₁ ψ₂ : W → ℝ,
        (potentialSeed v₀).key ψ₁ = (potentialSeed v₀).key ψ₂ ↔ cob ψ₁ = cob ψ₂)
      ∧ ¬ HasOpportunity (fun u v => ψ u - ψ v - d u v)
      ∧ circ (inject (fun u v => ψ u - ψ v - d u v) a b δ) [a, b, a]
          = δ - drag (fun u v => ψ u - ψ v - d u v) a b :=
  ⟨potentialSeed_canon v₀,
   dragfield_sterile ψ d hd,
   captured_minus_drag _ a b δ hab⟩

end TheSeed

-- ═══════════════════════════════════════════════════════════════════
-- §8b. THE WIRES CLOSED — Conservative is the seed's image; the
--      circulation is the root's `foldKey` (the streaming face)
-- ═══════════════════════════════════════════════════════════════════
-- Two wires §8 left open. (1) The file's own §1–§3 objects join the seed
-- language: `Conservative` is exactly "some potential's coboundary reaches
-- `w`", and `HasOpportunity` is the key failing to exist — the flaw is
-- keylessness. (2) The streaming law applies to the market *verbatim*, not
-- by analogy: `circ` IS `Collapse.foldKey` over the walk's edge list, so
-- `streaming_exact` is the measured method's law (certify the rest once,
-- read each flaw at its own edge), worn directly.

section RootWiring

variable {N : Type}

/-- **`Conservative` is the seed's image** — the §1 definition and the §8 seed
    language are one: a field is the monad iff some potential's coboundary
    reaches it. Definitional on each side; the wire made visible. -/
theorem conservative_is_seeded (w : N → N → ℝ) :
    Conservative w ↔ ∃ ψ : N → ℝ, cob ψ = w := by
  constructor
  · rintro ⟨ψ, h⟩
    exact ⟨ψ, by funext u v; exact (h u v).symm⟩
  · rintro ⟨ψ, h⟩
    exact ⟨ψ, fun u v => by rw [← h]; rfl⟩

/-- **Opportunity is the key failing to exist.** A field with a gaining cycle
    is outside the seed's image — no potential presents it.
    `opportunity_implies_flaw` wired through `conservative_is_seeded`;
    nothing re-proven. -/
theorem opportunity_has_no_key {w : N → N → ℝ} (h : HasOpportunity w) :
    ¬ ∃ ψ : N → ℝ, cob ψ = w :=
  fun hk => opportunity_implies_flaw h ((conservative_is_seeded w).mpr hk)

/-- The edge list of a walk — the per-item coordinates the circulation folds. -/
def walkEdges : List N → List (N × N)
  | [] => []
  | [_] => []
  | u :: v :: rest => (u, v) :: walkEdges (v :: rest)

/-- **The circulation IS the root's `foldKey`** — the market lane's key is a
    fold of per-edge contributions, exactly the shape the streaming law
    governs. Not an analogy: the same function. -/
theorem circ_is_foldKey (w : N → N → ℝ) (l : List N) :
    circ w l = Collapse.foldKey (fun e : N × N => w e.1 e.2) (walkEdges l) := by
  induction l with
  | nil => simp [circ, walkEdges, Collapse.foldKey]
  | cons u tail ih =>
    cases tail with
    | nil => simp [circ, walkEdges, Collapse.foldKey]
    | cons v rest =>
      have e1 : circ w (u :: v :: rest) = w u v + circ w (v :: rest) := rfl
      have e2 : walkEdges (u :: v :: rest) = (u, v) :: walkEdges (v :: rest) := rfl
      rw [e1, e2, ih]
      simp [Collapse.foldKey]

/-- **The market lane streams** — the read over `rest ++ flaw` edges is the
    rest's read (certified once, `dragfield_sterile`) plus the flaw's own:
    never re-touch the rest. `Collapse.streaming_exact`, worn directly —
    the law under the measured 10,087×. -/
theorem circulation_streams (w : N → N → ℝ) (rest flaw : List (N × N)) :
    Collapse.foldKey (fun e : N × N => w e.1 e.2) (rest ++ flaw)
      = Collapse.foldKey (fun e : N × N => w e.1 e.2) rest
        + Collapse.foldKey (fun e : N × N => w e.1 e.2) flaw :=
  Collapse.streaming_exact _ rest flaw

end RootWiring

-- ═══════════════════════════════════════════════════════════════════
-- §9. THE THREE FACES ARE THE THREE SIGNS OF THE CELL'S NORM
-- ═══════════════════════════════════════════════════════════════════
-- §4b named the faces `+ opportunity · 0 monad · − drag` and said they are the
-- Null cell's `{+1, 0, −1}`. That was a reading laid on the algebra. Here it is
-- the algebra: the 2-cycle is a cell state, the captured value is a null half,
-- and the break-even is the cone.

open Sanguine.CrossUniverse

/-- The injected 2-cycle as a Null-cell state: the flaw `δ` on one coordinate, the pair's
    round-trip drag on the other. -/
def cycleState (δ dr : ℝ) : NCell := (δ, dr)

/-- **The captured value is a null pairing.** `captured = δ − drag = ⟪⟨δ, drag⟩, 1+j⟫`: the
    quantity `captured_minus_drag` reports is the state paired against one null direction — the
    cell's own half, not a subtraction chosen for the occasion. -/
theorem captured_is_null_pairing (w : V → V → ℝ) (a b : V) (δ : ℝ) (hab : a ≠ b) :
    circ (inject w a b δ) [a, b, a] = nForm (cycleState δ (drag w a b)) nNull := by
  rw [captured_minus_drag w a b δ hab]
  simp only [nForm, cycleState, nNull]; ring

/-- **The norm is the product of the two halves**: `N⟨δ, drag⟩ = (δ+drag)(δ−drag)` — the sum face
    times the captured face. -/
theorem norm_is_the_product_of_the_two_halves (δ dr : ℝ) :
    nNorm (cycleState δ dr) = (δ + dr) * (δ - dr) := by
  simp only [nNorm, cycleState]; ring

/-- **The break-even is the cone.** The state is null exactly when the flaw equals the drag up to
    sign — `flaw_must_clear_drag`'s boundary is `nNorm = 0`, the locus where `/` goes singular. -/
theorem breakeven_is_the_cone (δ dr : ℝ) :
    nNorm (cycleState δ dr) = 0 ↔ (δ = dr ∨ δ = -dr) := by
  rw [norm_is_the_product_of_the_two_halves, mul_eq_zero]
  constructor
  · rintro (h | h)
    · exact Or.inr (by linarith)
    · exact Or.inl (by linarith)
  · rintro (h | h)
    · exact Or.inr (by rw [h]; ring)
    · exact Or.inl (by rw [h]; ring)

/-- **The three faces ARE the three signs of the indefinite norm.** For a nonnegative flaw against
    a nonnegative drag, the cycle gains iff `N > 0`, breaks even iff `N = 0` (the cone), and loses
    iff `N < 0`. `flaw_must_clear_drag` is the sign of the split norm; `ncell_indefinite` is why
    there are exactly three faces and not two, and `nNull_norm` is the middle one. -/
theorem trinity_is_the_norm_sign (w : V → V → ℝ) (a b : V) (δ : ℝ) (hab : a ≠ b)
    (hδ : 0 ≤ δ) (hd : 0 ≤ drag w a b) :
    (0 < circ (inject w a b δ) [a, b, a] ↔ 0 < nNorm (cycleState δ (drag w a b)))
      ∧ (circ (inject w a b δ) [a, b, a] = 0 ↔ nNorm (cycleState δ (drag w a b)) = 0)
      ∧ (circ (inject w a b δ) [a, b, a] < 0 ↔ nNorm (cycleState δ (drag w a b)) < 0) := by
  rw [captured_minus_drag w a b δ hab, norm_is_the_product_of_the_two_halves]
  refine ⟨⟨fun h => by nlinarith, fun h => by nlinarith⟩,
          ⟨fun h => by rw [h]; ring, fun h => by nlinarith⟩,
          ⟨fun h => by nlinarith, fun h => by nlinarith⟩⟩

/-- **Opportunity is the cell's `NB` face (headline, in the cell).** The captured value is one null
    half of the 2-cycle state; the break-even is the null cone; the three faces are the three signs
    the split norm takes. The trinity on a field's circulation is not like the Null cell's — it is
    the Null cell's, read on `⟨δ, drag⟩`. -/
theorem opportunity_is_the_cell (w : V → V → ℝ) (a b : V) (δ : ℝ) (hab : a ≠ b) :
    circ (inject w a b δ) [a, b, a] = nForm (cycleState δ (drag w a b)) nNull
      ∧ nNorm (cycleState δ (drag w a b))
          = (δ + drag w a b) * (δ - drag w a b)
      ∧ (∃ z y : NCell, nNorm z > 0 ∧ nNorm y < 0)
      ∧ nNorm nNull = 0 :=
  ⟨captured_is_null_pairing w a b δ hab,
   norm_is_the_product_of_the_two_halves δ (drag w a b),
   ncell_indefinite, nNull_norm⟩

end Opportunity
end Sanguine

end
