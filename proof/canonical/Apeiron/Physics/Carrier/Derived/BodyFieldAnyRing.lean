import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.CarrierBodyWorkAnyRing

/-!
# Finite basis momentum field and body exchange over any commutative ring

@name Carrier Body Field Accounting Over Any Commutative Ring
@grade proved

`CarrierBodyWorkAnyRing` derives the local equal-mass exchange of two ring points. This file
places one such exchange at a selected site of a finite field. The field is `Fin n → RingCell R`
for both positions and momenta, so every site carries one two-slot point over an arbitrary
commutative ring `R` and the whole state is finite by construction. Taking `R = Z/2^64` is an
instance rather than an analogy.

Contact is a separate, explicit adaptor. A body position is compared with the selected site's
basis position, and the comparison is an authored reading of the two slots as positions. It is not
forced by `ringNormForm`, and `ringNormForm` never sees a position. The exchange runs only when
contact holds and the supplied impulse is the derived `ringExchangeImpulse`.

The updates replace one site only. The total `RingCell R` momentum of the whole field
plus the body is unchanged, and the finite sum of `ringNormForm σ` over the field plus the body is
unchanged at every dial value. That sum is a norm-form ledger, not a kinetic energy: `ringNormForm`
is indefinite in a general ring, so no order and no `1/2` appear here. The kinetic-energy reading
belongs to the nonpositive-`σ` faces of the real development.

Rejection is transactional. The accepted result is `some` of the updated field and body; a
rejected contact or a non-derived impulse returns `none`, and `ringExchangeOrRollback` then returns
the supplied field and body untouched. Both transaction definitions are computable `def`s: the
executable branches take an explicit `[DecidableEq R]`, so no `Classical` choice enters them.

The whole field exchange is proved from the generic exchange theorems of `CarrierBodyWorkAnyRing`.
Locality uses the swap at the selected site, total momentum uses
`ringTotalFieldMomentum_conserved`, the ledger uses `ringNormFormLedger_conserved`, and rollback
uses the rejection theorems. No impulse or swap formula is authored a second time.

## Root

| from | here |
|---|---|
| `AnyRing.RingCell` | `ringBasisPositionField`, `ringBasisMomentumField` |
| `CarrierBodyWorkAnyRing.ringExchangeImpulse` | `ringFieldMomentumAfter` |
| `CarrierBodyWorkAnyRing.ringTotalMomentum_conserved` | `ringTotalFieldMomentum_conserved` |
| `CarrierBodyWorkAnyRing.ringNormFormLedger_conserved` | `ringNormFormFieldLedger_conserved` |

The field is `Fin n → RingCell R` for both positions and momenta, so every site carries one
two-slot point. `ringFieldMomentumAfter` replaces the selected site with the derived exchange and
leaves the rest alone, `ringTotalFieldMomentum_conserved` and `ringNormFormFieldLedger_conserved`
account for the two whole-field totals, and `ringExchangeTransaction` with
`ringExchangeOrRollback` makes rejection transactional. Contact is an authored reading of the two
slots as positions: `ringNormForm` never sees a position and cannot check it. What does not
transfer is the kinetic-energy reading, which needs an order and a half and belongs to the
nonpositive-`σ` faces of the real development.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

variable {R : Type*} [CommRing R]
variable {n : ℕ}

/-- A finite basis position field: one ring point at every site.

    (n : ℕ)                         [the counting numbers 0, 1, 2, 3, and so on]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    →                               [takes the left and gives the right]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    The position field and the momentum field are both `Fin n → RingCell R`; the role is the
    interpretation, not a second representation. -/
abbrev ringBasisPositionField (R : Type*) [CommRing R] (n : ℕ) : Type _ := Fin n → RingCell R

/-- A finite basis momentum field: one ring point at every site.

    (n : ℕ)                         [the counting numbers 0, 1, 2, 3, and so on]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    →                               [takes the left and gives the right]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    The same finite shape as the position field, read as momentum. -/
abbrev ringBasisMomentumField (R : Type*) [CommRing R] (n : ℕ) : Type _ := Fin n → RingCell R

/-- The basis position the field places at a site: the explicit spatial adaptor.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (s : Fin n)                     [the selected site]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringBasisPositionField`        [one ring point at every site, read as position]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    This names the reading of a site's `RingCell R` as a position. `ringNormForm` does not supply
    it and cannot check it. -/
def ringSitePosition (pos : ringBasisPositionField R n) (s : Fin n) : RingCell R := pos s

/-- Contact holds exactly when the body position equals the selected site's basis position.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (s : Fin n)                     [the selected site]
    (bodyPos : RingCell R)          [the body position, a ring point]
    Prop                            [a claim that is true or false]
    `ringBasisPositionField`        [one ring point at every site, read as position]
    `ringSitePosition`              [the explicit spatial adaptor at a site]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    Contact is a spatial comparison, not a norm-form condition. -/
abbrev ringContactAt (pos : ringBasisPositionField R n) (s : Fin n) (bodyPos : RingCell R) : Prop :=
  bodyPos = ringSitePosition pos s

/-- The momentum field after the exchange at the selected site.

    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    ringBasisMomentumField R n        [one ring point at every site, read as momentum]
    `ringCarrierMomentumAfter`      [the carrier momentum with the derived impulse removed]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    Only site `s` changes, and it changes through `ringCarrierMomentumAfter`. -/
def ringFieldMomentumAfter (mom : ringBasisMomentumField R n) (s : Fin n) (bodyMom : RingCell R) :
    ringBasisMomentumField R n :=
  fun i => if i = s then ringCarrierMomentumAfter (mom s) bodyMom else mom i

/-- The selected site of the updated field holds the derived carrier result.

    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringFieldMomentumAfter`        [the momentum field with the selected site updated]
    `ringCarrierMomentumAfter`      [the carrier momentum with the derived impulse removed]

    The selected site is the only site the update touches. -/
theorem ringFieldMomentumAfter_site (mom : ringBasisMomentumField R n) (s : Fin n)
    (bodyMom : RingCell R) :
    ringFieldMomentumAfter mom s bodyMom s = ringCarrierMomentumAfter (mom s) bodyMom := by
  unfold ringFieldMomentumAfter
  rw [if_pos rfl]

/-- The selected site of the updated field holds the body momentum exactly.

    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringFieldMomentumAfter`        [the momentum field with the selected site updated]
    `ringBodyMomentumAfter`         [the body momentum with the derived impulse added]

    Equal masses make the selected site a swap. -/
theorem ringFieldMomentumAfter_site_eq_body (mom : ringBasisMomentumField R n) (s : Fin n)
    (bodyMom : RingCell R) : ringFieldMomentumAfter mom s bodyMom s = bodyMom := by
  rw [ringFieldMomentumAfter_site, ringCarrierMomentumAfter_eq]

/-- Every other site of the updated field is unchanged.

    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    (i : Fin n)                     [any other site]
    (h : i ≠ s)                     [the other site is not the selected one]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    ≠                               [is not equal to]
    `ringFieldMomentumAfter`        [the momentum field with the selected site updated]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    A site away from `s` keeps the momentum it had. -/
theorem ringFieldMomentumAfter_other (mom : ringBasisMomentumField R n) (s i : Fin n)
    (bodyMom : RingCell R) (h : i ≠ s) : ringFieldMomentumAfter mom s bodyMom i = mom i := by
  unfold ringFieldMomentumAfter
  rw [if_neg h]

/-- The update is exactly the single-site replacement of `mom` at `s` by the body momentum.

    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    ringBasisMomentumField R n        [one ring point at every site, read as momentum]
    `ringFieldMomentumAfter`        [the momentum field with the selected site updated]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    This is the statement that no site but `s` is disturbed. -/
theorem ringFieldMomentumAfter_eq_update (mom : ringBasisMomentumField R n) (s : Fin n)
    (bodyMom : RingCell R) :
    ringFieldMomentumAfter mom s bodyMom = Function.update mom s bodyMom := by
  funext i
  by_cases hi : i = s
  · subst hi
    simp [ringFieldMomentumAfter, Function.update, ringCarrierMomentumAfter_eq]
  · simp [ringFieldMomentumAfter, Function.update, hi]

/-- The body momentum after the exchange at the selected site.

    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringBodyMomentumAfter`         [the body momentum with the derived impulse added]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    The body reaction is defined from the site momentum through `ringBodyMomentumAfter`. -/
def ringBodyMomentumFieldAfter (mom : ringBasisMomentumField R n) (s : Fin n)
    (bodyMom : RingCell R) : RingCell R :=
  ringBodyMomentumAfter (mom s) bodyMom

/-- The whole-region `RingCell R` momentum: every site plus the body.

    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringBasisMomentumField`        [one ring point at every site, read as momentum]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    The sum is finite because the site set is `Fin n`. -/
def ringTotalMomentum (mom : ringBasisMomentumField R n) (bodyMom : RingCell R) : RingCell R :=
  (∑ i, mom i) + bodyMom

/-- The sum of one updated site in a finite field, over any additive commutative group.

    (g : Fin n → A)                 [any finite family of group elements]
    (s : Fin n)                     [the selected site]
    (v : A)                         [the value the selected site takes]
    A                               [an additive commutative group]
    `A`                             [an additive commutative group]

    Replacing one term of a finite sum removes the old value and adds the new one. -/
theorem ringSum_single_site {A : Type*} [AddCommGroup A] (g : Fin n → A) (s : Fin n) (v : A) :
    (∑ i, (if i = s then v else g i)) = (∑ i, g i) - g s + v := by
  have hsingle : (∑ i : Fin n, (if i = s then v - g s else 0)) = v - g s := by
    rw [Finset.sum_eq_single s]
    · simp
    · intro b _ hb; simp [hb]
    · intro h; exact absurd (Finset.mem_univ s) h
  calc
    (∑ i, (if i = s then v else g i))
        = ∑ i, (g i + (if i = s then v - g s else 0)) := by
          apply Finset.sum_congr rfl
          intro i _
          by_cases hi : i = s
          · subst hi; simp
          · simp [hi]
    _ = (∑ i, g i) + ∑ i, (if i = s then v - g s else 0) := by
          rw [Finset.sum_add_distrib]
    _ = (∑ i, g i) - g s + v := by
          rw [hsingle]
          abel

/-- The whole-region momentum is conserved by the selected-site exchange.

    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringTotalMomentum`             [every site plus the body, as one ring point]
    `ringFieldMomentumAfter`        [the momentum field with the selected site updated]
    `ringBodyMomentumFieldAfter`    [the body momentum after the selected-site exchange]

    The updated field plus the updated body equals the original field plus body. -/
theorem ringTotalFieldMomentum_conserved (mom : ringBasisMomentumField R n) (s : Fin n)
    (bodyMom : RingCell R) :
    ringTotalMomentum (ringFieldMomentumAfter mom s bodyMom)
        (ringBodyMomentumFieldAfter mom s bodyMom)
      = ringTotalMomentum mom bodyMom := by
  have hsum : (∑ i, ringFieldMomentumAfter mom s bodyMom i)
      = (∑ i, mom i) - mom s + ringCarrierMomentumAfter (mom s) bodyMom := by
    change (∑ i, (if i = s then ringCarrierMomentumAfter (mom s) bodyMom else mom i))
        = (∑ i, mom i) - mom s + ringCarrierMomentumAfter (mom s) bodyMom
    exact ringSum_single_site mom s (ringCarrierMomentumAfter (mom s) bodyMom)
  unfold ringTotalMomentum ringBodyMomentumFieldAfter
  rw [hsum, ringCarrierMomentumAfter_eq, ringBodyMomentumAfter_eq]
  abel

/-- The norm-form ledger of the whole field plus the body.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    R                               [the commutative ring the points are built over]
    `ringNormForm`                  [the norm form, which can come out negative]
    `ringBasisMomentumField`        [one ring point at every site, read as momentum]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    This is universal accounting: `ringNormForm` may be negative or non-invertible in a general
    ring, so the sum is named a ledger rather than an energy. -/
def ringNormFormFieldLedger (σ : R) (mom : ringBasisMomentumField R n) (bodyMom : RingCell R) : R :=
  (∑ i, ringNormForm σ (mom i)) + ringNormForm σ bodyMom

/-- The norm-form ledger is conserved by the selected-site exchange at every dial value.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    R                               [the commutative ring the points are built over]
    `ringNormFormFieldLedger`       [the finite norm-form sum over every site plus the body]
    `ringNormForm`                  [the norm form, which can come out negative]
    `ringFieldMomentumAfter`        [the momentum field with the selected site updated]
    `ringBodyMomentumFieldAfter`    [the body momentum after the selected-site exchange]

    The ledger after the exchange equals the ledger before it, because the two norm forms at the
    selected site are only permuted. -/
theorem ringNormFormFieldLedger_conserved (σ : R) (mom : ringBasisMomentumField R n) (s : Fin n)
    (bodyMom : RingCell R) :
    ringNormFormFieldLedger σ (ringFieldMomentumAfter mom s bodyMom)
        (ringBodyMomentumFieldAfter mom s bodyMom)
      = ringNormFormFieldLedger σ mom bodyMom := by
  have hsum : (∑ i, ringNormForm σ (ringFieldMomentumAfter mom s bodyMom i))
      = (∑ i, ringNormForm σ (mom i)) - ringNormForm σ (mom s)
          + ringNormForm σ (ringCarrierMomentumAfter (mom s) bodyMom) := by
    have hpoint : (∑ i, ringNormForm σ (ringFieldMomentumAfter mom s bodyMom i))
        = ∑ i, (if i = s then ringNormForm σ (ringCarrierMomentumAfter (mom s) bodyMom)
                 else ringNormForm σ (mom i)) := by
      apply Finset.sum_congr rfl
      intro i _
      by_cases hi : i = s
      · subst hi; simp [ringFieldMomentumAfter]
      · simp [ringFieldMomentumAfter, hi]
    rw [hpoint]
    exact ringSum_single_site (fun i => ringNormForm σ (mom i)) s
      (ringNormForm σ (ringCarrierMomentumAfter (mom s) bodyMom))
  have hpt : ringNormForm σ (ringCarrierMomentumAfter (mom s) bodyMom)
      + ringNormForm σ (ringBodyMomentumAfter (mom s) bodyMom)
      = ringNormForm σ (mom s) + ringNormForm σ bodyMom := by
    simpa only [ringNormFormLedger] using ringNormFormLedger_conserved σ (mom s) bodyMom
  unfold ringNormFormFieldLedger ringBodyMomentumFieldAfter
  rw [hsum]
  linear_combination hpt

/-- A transactional exchange: contact and the derived impulse, or no update.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : RingCell R)          [the body position, a ring point]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    (claimed : RingCell R)          [the impulse the caller claims to hand over]
    Option (ringBasisMomentumField R n × RingCell R)
                                     [either the updated field and body, or nothing]
    ×                               [paired with, so R × R means two ring elements side by side]
    `ringContactAt`                 [the body position equals the selected site's basis position]
    `ringExchangeImpulse`           [the impulse derived from the two basis momenta]
    `ringFieldMomentumAfter`        [the momentum field with the selected site updated]
    `ringBodyMomentumFieldAfter`    [the body momentum after the selected-site exchange]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    Both contact and the exact derived impulse are required before any update is produced. The
    branches are executable: `[DecidableEq R]` decides the equalities, so no `Classical` choice
    enters the definition. -/
def ringExchangeTransaction [DecidableEq R] (pos : ringBasisPositionField R n)
    (mom : ringBasisMomentumField R n) (s : Fin n) (bodyPos bodyMom claimed : RingCell R) :
    Option (ringBasisMomentumField R n × RingCell R) :=
  if ringContactAt pos s bodyPos ∧ claimed = ringExchangeImpulse (mom s) bodyMom then
    some (ringFieldMomentumAfter mom s bodyMom, ringBodyMomentumFieldAfter mom s bodyMom)
  else
    none

/-- The canonical contact and derived impulse are accepted.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : RingCell R)          [the body position, a ring point]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    (claimed : RingCell R)          [the impulse the caller claims to hand over]
    (hcontact : ringContactAt pos s bodyPos)
                                     [the body position equals the selected site's basis position]
    (himpulse : claimed = ringExchangeImpulse (mom s) bodyMom)
                                     [the claim is the derived impulse]
    `ringExchangeTransaction`       [the contact-and-impulse transactional exchange]
    `ringFieldMomentumAfter`        [the momentum field with the selected site updated]
    `ringBodyMomentumFieldAfter`    [the body momentum after the selected-site exchange]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    The accepted result is the updated field and body, not the inputs. -/
theorem ringExchangeTransaction_accepts [DecidableEq R] (pos : ringBasisPositionField R n)
    (mom : ringBasisMomentumField R n) (s : Fin n) (bodyPos bodyMom claimed : RingCell R)
    (hcontact : ringContactAt pos s bodyPos)
    (himpulse : claimed = ringExchangeImpulse (mom s) bodyMom) :
    ringExchangeTransaction pos mom s bodyPos bodyMom claimed
      = some (ringFieldMomentumAfter mom s bodyMom, ringBodyMomentumFieldAfter mom s bodyMom) := by
  unfold ringExchangeTransaction
  rw [if_pos ⟨hcontact, himpulse⟩]

/-- A body position off the selected site is rejected.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : RingCell R)          [the body position, a ring point]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    (claimed : RingCell R)          [the impulse the caller claims to hand over]
    (h : ¬ ringContactAt pos s bodyPos)
                                     [contact does not hold]
    ¬                               [this is not true]
    Option (ringBasisMomentumField R n × RingCell R)
                                     [either the updated field and body, or nothing]
    `ringContactAt`                 [the body position equals the selected site's basis position]
    `ringExchangeTransaction`       [the contact-and-impulse transactional exchange]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    No update is produced when contact fails, whatever the claimed impulse. -/
theorem ringExchangeTransaction_rejects_position [DecidableEq R] (pos : ringBasisPositionField R n)
    (mom : ringBasisMomentumField R n) (s : Fin n) (bodyPos bodyMom claimed : RingCell R)
    (h : ¬ ringContactAt pos s bodyPos) :
    ringExchangeTransaction pos mom s bodyPos bodyMom claimed = none := by
  unfold ringExchangeTransaction
  rw [if_neg]
  intro hc
  exact h hc.1

/-- A claimed impulse that is not the derived impulse is rejected.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : RingCell R)          [the body position, a ring point]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    (claimed : RingCell R)          [the impulse the caller claims to hand over]
    (h : claimed ≠ ringExchangeImpulse (mom s) bodyMom)
                                     [the claim differs from the derived impulse]
    ≠                               [is not equal to]
    Option (ringBasisMomentumField R n × RingCell R)
                                     [either the updated field and body, or nothing]
    `ringExchangeImpulse`           [the impulse derived from the two basis momenta]
    `ringExchangeTransaction`       [the contact-and-impulse transactional exchange]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    Contact alone does not license an independently supplied impulse. -/
theorem ringExchangeTransaction_rejects_impulse [DecidableEq R] (pos : ringBasisPositionField R n)
    (mom : ringBasisMomentumField R n) (s : Fin n) (bodyPos bodyMom claimed : RingCell R)
    (h : claimed ≠ ringExchangeImpulse (mom s) bodyMom) :
    ringExchangeTransaction pos mom s bodyPos bodyMom claimed = none := by
  unfold ringExchangeTransaction
  rw [if_neg]
  intro hc
  exact h hc.2

/-- The rollback result: the accepted update, or the supplied field and body.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : RingCell R)          [the body position, a ring point]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    (claimed : RingCell R)          [the impulse the caller claims to hand over]
    ringBasisMomentumField R n × RingCell R
                                     [the field paired with the body]
    ×                               [paired with, so R × R means two ring elements side by side]
    `ringExchangeTransaction`       [the contact-and-impulse transactional exchange]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    On rejection the supplied field and body are returned unchanged. The definition is a
    computable `def` because `ringExchangeTransaction` is. -/
def ringExchangeOrRollback [DecidableEq R] (pos : ringBasisPositionField R n)
    (mom : ringBasisMomentumField R n) (s : Fin n) (bodyPos bodyMom claimed : RingCell R) :
    ringBasisMomentumField R n × RingCell R :=
  match ringExchangeTransaction pos mom s bodyPos bodyMom claimed with
  | some state => state
  | none => (mom, bodyMom)

/-- Rejected contact leaves the supplied field and body unchanged.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : RingCell R)          [the body position, a ring point]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    (claimed : RingCell R)          [the impulse the caller claims to hand over]
    (h : ¬ ringContactAt pos s bodyPos)
                                     [contact does not hold]
    ¬                               [this is not true]
    ringBasisMomentumField R n × RingCell R
                                     [the field paired with the body]
    `ringExchangeOrRollback`        [the accepted update or the supplied state]
    `ringExchangeTransaction_rejects_position`
                                     [the rejection theorem for a position off the selected site]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    The rollback returns exactly the inputs. -/
theorem ringExchangeOrRollback_of_rejected_position [DecidableEq R]
    (pos : ringBasisPositionField R n)
    (mom : ringBasisMomentumField R n) (s : Fin n) (bodyPos bodyMom claimed : RingCell R)
    (h : ¬ ringContactAt pos s bodyPos) :
    ringExchangeOrRollback pos mom s bodyPos bodyMom claimed = (mom, bodyMom) := by
  unfold ringExchangeOrRollback
  rw [ringExchangeTransaction_rejects_position pos mom s bodyPos bodyMom claimed h]

/-- A non-derived impulse leaves the supplied field and body unchanged.

    (pos : ringBasisPositionField R n)
                                     [the finite position field]
    (mom : ringBasisMomentumField R n)
                                     [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : RingCell R)          [the body position, a ring point]
    (bodyMom : RingCell R)          [the body momentum, a ring point]
    (claimed : RingCell R)          [the impulse the caller claims to hand over]
    (h : claimed ≠ ringExchangeImpulse (mom s) bodyMom)
                                     [the claim differs from the derived impulse]
    ≠                               [is not equal to]
    ringBasisMomentumField R n × RingCell R
                                     [the field paired with the body]
    `ringExchangeOrRollback`        [the accepted update or the supplied state]
    `ringExchangeTransaction_rejects_impulse`
                                     [the rejection theorem for a non-derived impulse]
    `RingCell R`                    [a point: two ring elements side by side, a + b·x]

    The rollback returns exactly the inputs. -/
theorem ringExchangeOrRollback_of_rejected_impulse [DecidableEq R]
    (pos : ringBasisPositionField R n)
    (mom : ringBasisMomentumField R n) (s : Fin n) (bodyPos bodyMom claimed : RingCell R)
    (h : claimed ≠ ringExchangeImpulse (mom s) bodyMom) :
    ringExchangeOrRollback pos mom s bodyPos bodyMom claimed = (mom, bodyMom) := by
  unfold ringExchangeOrRollback
  rw [ringExchangeTransaction_rejects_impulse pos mom s bodyPos bodyMom claimed h]

end Physics
end Apeiron
