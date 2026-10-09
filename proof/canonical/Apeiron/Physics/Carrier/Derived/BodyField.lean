import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.Carrier.Derived.BodyWork
import Apeiron.Physics.Carrier.Derived.BodyFieldAnyRing

/-!
# Finite basis momentum field and body exchange over the reals

@name Carrier Body Field Accounting
@grade proved

`CarrierBodyFieldAnyRing` places one equal-mass exchange at a selected site of a finite field over
an arbitrary commutative ring. This file is its real specialization: every field name below is the
`ℝ` instance of the ring-universal definition or theorem, and `Cell` is `RingCell ℝ`. The impulse
and swap formulas are authored once, in `CarrierBodyWorkAnyRing`, and no second copy appears here.

Contact is a separate, explicit adaptor. A body position is compared with the selected site's basis
position, and the comparison is an authored reading of the two `Cell` coordinates as positions. It
is not forced by `normForm`, and `normForm` never sees a position. The exchange runs only when
contact holds and the supplied impulse is the derived `exchangeImpulse`.

The updates replace one site only. The total `Cell` momentum of the whole field plus
the body is unchanged, and the finite sum of `normForm σ` over the field plus the body is unchanged
at every `σ`. That sum is a norm-form ledger, not a kinetic energy: `normForm` is indefinite on the
hyperbolic face, so the physical energy reading stays reserved exactly as in `CarrierBodyWork`.

Rejection is transactional. The accepted result is `some` of the updated field and body; a
rejected contact or a non-derived impulse returns `none`, and `exchangeOrRollback` then returns the
supplied field and body untouched.

## Supported regime, assumptions and scope

A finite local collision experiment. One selected site, one body, an instantaneous exchange, and no
time step. There is no shared fluid velocity, no pressure, no boundary or volume term, no position
update, and no field-to-field coupling. The finite field is a bookkeeping device for the region
totals, not a production body-fluid solver.

## Root

`BasisPositionField` and `BasisMomentumField` are `Fin n → Cell`. `ContactAt` is the explicit
spatial adaptor, `fieldMomentumAfter` replaces only the selected site, and
`totalFieldMomentum_conserved` with `normFormFieldLedger_conserved` accounts for the two
whole-region totals. `exchangeTransaction` is the acceptance result and `exchangeOrRollback` is the
rollback. Each is the `ℝ` instance of the generic definition or theorem in
`CarrierBodyFieldAnyRing`.
-/

namespace Apeiron
namespace Physics

open Trichotomy

variable {n : ℕ}

/-- A finite basis position field: one trichotomy point at every site.

    (n : ℕ)                         [the counting numbers 0, 1, 2, 3, and so on]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    The position field and the momentum field are both `Fin n → Cell`; the role is the
    interpretation, not a second representation. This is the real instance of
    `ringBasisPositionField`. -/
abbrev BasisPositionField (n : ℕ) : Type := ringBasisPositionField ℝ n

/-- A finite basis momentum field: one trichotomy point at every site.

    (n : ℕ)                         [the counting numbers 0, 1, 2, 3, and so on]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    The same finite shape as the position field, read as momentum, and the real instance of
    `ringBasisMomentumField`. -/
abbrev BasisMomentumField (n : ℕ) : Type := ringBasisMomentumField ℝ n

/-- The basis position the field places at a site: the explicit spatial adaptor.

    (pos : BasisPositionField n)    [the finite position field]
    (s : Fin n)                     [the selected site]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    This names the reading of a site's `Cell` as a position, and it is the real instance of
    `ringSitePosition`. `normForm` does not supply it and cannot check it. -/
abbrev sitePosition (pos : BasisPositionField n) (s : Fin n) : Cell :=
  ringSitePosition (R := ℝ) pos s

/-- Contact holds exactly when the body position equals the selected site's basis position.

    (pos : BasisPositionField n)    [the finite position field]
    (s : Fin n)                     [the selected site]
    (bodyPos : Cell)                [the body position, a basis point]
    Prop                            [a claim that is true or false]
    `sitePosition`                  [the explicit spatial adaptor at a site]
    `Cell`                          [a point: two real numbers written side by side, a + b·x]

    Contact is a spatial comparison, not a norm-form condition, and the real instance of
    `ringContactAt`. -/
abbrev ContactAt (pos : BasisPositionField n) (s : Fin n) (bodyPos : Cell) : Prop :=
  ringContactAt (R := ℝ) pos s bodyPos

/-- The momentum field after the exchange at the selected site.

    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : Cell)                [the body momentum, a basis point]
    BasisMomentumField n            [one trichotomy point at every site, read as momentum]
    `carrierMomentumAfter`          [the carrier momentum with the derived impulse removed]
    `Cell`                          [a point: two real numbers written side by side, a + b·x]

    Only site `s` changes, and it changes through `carrierMomentumAfter`; this is the real
    instance of `ringFieldMomentumAfter`. -/
abbrev fieldMomentumAfter (mom : BasisMomentumField n) (s : Fin n) (bodyMom : Cell) :
    BasisMomentumField n :=
  ringFieldMomentumAfter (R := ℝ) mom s bodyMom

/-- The selected site of the updated field holds the derived carrier result.

    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : Cell)                [the body momentum, a basis point]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `carrierMomentumAfter`          [the carrier momentum with the derived impulse removed]

    The selected site is the only site the update touches. This is the real instance of
    `ringFieldMomentumAfter_site`. -/
theorem fieldMomentumAfter_site (mom : BasisMomentumField n) (s : Fin n) (bodyMom : Cell) :
    fieldMomentumAfter mom s bodyMom s = carrierMomentumAfter (mom s) bodyMom :=
  ringFieldMomentumAfter_site (R := ℝ) mom s bodyMom

/-- The selected site of the updated field holds the body momentum exactly.

    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : Cell)                [the body momentum, a basis point]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumAfter`             [the body momentum with the derived impulse added]

    Equal masses make the selected site a swap, and this is the real instance of
    `ringFieldMomentumAfter_site_eq_body`. -/
theorem fieldMomentumAfter_site_eq_body (mom : BasisMomentumField n) (s : Fin n)
    (bodyMom : Cell) : fieldMomentumAfter mom s bodyMom s = bodyMom :=
  ringFieldMomentumAfter_site_eq_body (R := ℝ) mom s bodyMom

/-- Every other site of the updated field is unchanged.

    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : Cell)                [the body momentum, a basis point]
    (i : Fin n)                     [any other site]
    (h : i ≠ s)                     [the other site is not the selected one]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ≠                               [is not equal to]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]

    A site away from `s` keeps the momentum it had. This is the real instance of
    `ringFieldMomentumAfter_other`. -/
theorem fieldMomentumAfter_other (mom : BasisMomentumField n) (s i : Fin n) (bodyMom : Cell)
    (h : i ≠ s) : fieldMomentumAfter mom s bodyMom i = mom i :=
  ringFieldMomentumAfter_other (R := ℝ) mom s i bodyMom h

/-- The update is exactly the single-site replacement of `mom` at `s` by the body momentum.

    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : Cell)                [the body momentum, a basis point]
    BasisMomentumField n            [one trichotomy point at every site, read as momentum]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `Cell`                          [a point: two real numbers written side by side, a + b·x]

    This is the statement that no site but `s` is disturbed, and the real instance of
    `ringFieldMomentumAfter_eq_update`. -/
theorem fieldMomentumAfter_eq_update (mom : BasisMomentumField n) (s : Fin n) (bodyMom : Cell) :
    fieldMomentumAfter mom s bodyMom = Function.update mom s bodyMom :=
  ringFieldMomentumAfter_eq_update (R := ℝ) mom s bodyMom

/-- The body momentum after the exchange at the selected site.

    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : Cell)                [the body momentum, a basis point]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `bodyMomentumAfter`             [the body momentum with the derived impulse added]

    The body reaction is defined from the site momentum through `bodyMomentumAfter`, and this is
    the real instance of `ringBodyMomentumFieldAfter`. -/
abbrev bodyMomentumFieldAfter (mom : BasisMomentumField n) (s : Fin n) (bodyMom : Cell) : Cell :=
  ringBodyMomentumFieldAfter (R := ℝ) mom s bodyMom

/-- The whole-region `Cell` momentum: every site plus the body.

    (mom : BasisMomentumField n)    [the finite momentum field]
    (bodyMom : Cell)                [the body momentum, a basis point]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    The sum is finite because the site set is `Fin n`, and this is the real instance of
    `ringTotalMomentum`. -/
abbrev totalMomentum (mom : BasisMomentumField n) (bodyMom : Cell) : Cell :=
  ringTotalMomentum (R := ℝ) mom bodyMom

/-- The sum of one updated site in a finite field.

    (g : Fin n → Cell)              [any finite family of basis points]
    (s : Fin n)                     [the selected site]
    (v : Cell)                      [the value the selected site takes]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    Replacing one term of a finite sum removes the old value and adds the new one. This is the
    `Cell` instance of `ringSum_single_site`. -/
theorem sum_single_site (g : Fin n → Cell) (s : Fin n) (v : Cell) :
    (∑ i, (if i = s then v else g i)) = (∑ i, g i) - g s + v :=
  ringSum_single_site g s v

/-- The same single-site sum for a real-valued finite family.

    (g : Fin n → ℝ)                 [any finite family of real numbers]
    (s : Fin n)                     [the selected site]
    (v : ℝ)                         [the value the selected site takes]
    ℝ                               [the real numbers: every number on the number line]

    The norm-form ledger sums real values, so this is the scalar instance of
    `ringSum_single_site`. -/
theorem sum_single_site_real (g : Fin n → ℝ) (s : Fin n) (v : ℝ) :
    (∑ i, (if i = s then v else g i)) = (∑ i, g i) - g s + v :=
  ringSum_single_site g s v

/-- The whole-region momentum is conserved by the selected-site exchange.

    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : Cell)                [the body momentum, a basis point]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `totalMomentum`                 [every site plus the body, as one basis point]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumFieldAfter`        [the body momentum after the selected-site exchange]

    The updated field plus the updated body equals the original field plus body. This is the real
    instance of `ringTotalFieldMomentum_conserved`. -/
theorem totalFieldMomentum_conserved (mom : BasisMomentumField n) (s : Fin n)
    (bodyMom : Cell) :
    totalMomentum (fieldMomentumAfter mom s bodyMom) (bodyMomentumFieldAfter mom s bodyMom)
      = totalMomentum mom bodyMom :=
  ringTotalFieldMomentum_conserved (R := ℝ) mom s bodyMom

/-- The norm-form ledger of the whole field plus the body.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (bodyMom : Cell)                [the body momentum, a basis point]
    ℝ                               [the real numbers: every number on the number line]
    `normForm`                      [the norm form, which can come out negative]

    This is universal accounting: `normForm` may be negative on hyperbolic faces, so the sum is
    named a ledger rather than an energy, and this is the real instance of
    `ringNormFormFieldLedger`. -/
abbrev normFormFieldLedger (σ : ℝ) (mom : BasisMomentumField n) (bodyMom : Cell) : ℝ :=
  ringNormFormFieldLedger (R := ℝ) σ mom bodyMom

/-- The norm-form ledger is conserved by the selected-site exchange at every σ.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyMom : Cell)                [the body momentum, a basis point]
    ℝ                               [the real numbers: every number on the number line]
    `normFormFieldLedger`           [the finite norm-form sum over every site plus the body]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumFieldAfter`        [the body momentum after the selected-site exchange]

    The ledger after the exchange equals the ledger before it, because the two norm forms at the
    selected site are only permuted. This is the real instance of
    `ringNormFormFieldLedger_conserved`. -/
theorem normFormFieldLedger_conserved (σ : ℝ) (mom : BasisMomentumField n) (s : Fin n)
    (bodyMom : Cell) :
    normFormFieldLedger σ (fieldMomentumAfter mom s bodyMom)
        (bodyMomentumFieldAfter mom s bodyMom)
      = normFormFieldLedger σ mom bodyMom :=
  ringNormFormFieldLedger_conserved (R := ℝ) σ mom s bodyMom

/-- A transactional exchange: contact and the derived impulse, or no update.

    (pos : BasisPositionField n)    [the finite position field]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : Cell)                [the body position, a basis point]
    (bodyMom : Cell)                [the body momentum, a basis point]
    (claimed : Cell)                [the impulse the caller claims to hand over]
    Option (BasisMomentumField n × Cell)
                                     [either the updated field and body, or nothing]
    ×                               [paired with, so Cell × Cell means two points side by side]
    `ContactAt`                     [the body position equals the selected site's basis position]
    `exchangeImpulse`               [the impulse derived from the two basis momenta]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumFieldAfter`        [the body momentum after the selected-site exchange]
    `Cell`                          [a point: two real numbers written side by side, a + b·x]

    Both contact and the exact derived impulse are required before any update is produced. This
    is the real instance of `ringExchangeTransaction`; equality of reals is not decidable, so the
    real specialization is `noncomputable`, while the generic definition takes an explicit
    `[DecidableEq R]` and is executable. -/
noncomputable abbrev exchangeTransaction (pos : BasisPositionField n) (mom : BasisMomentumField n)
    (s : Fin n) (bodyPos bodyMom claimed : Cell) : Option (BasisMomentumField n × Cell) :=
  ringExchangeTransaction (R := ℝ) pos mom s bodyPos bodyMom claimed

/-- The canonical contact and derived impulse are accepted.

    (pos : BasisPositionField n)    [the finite position field]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : Cell)                [the body position, a basis point]
    (bodyMom : Cell)                [the body momentum, a basis point]
    (claimed : Cell)                [the impulse the caller claims to hand over]
    (hcontact : ContactAt pos s bodyPos)
                                     [the body position equals the selected site's basis position]
    (himpulse : claimed = exchangeImpulse (mom s) bodyMom)
                                     [the claim is the derived impulse]
    `exchangeTransaction`           [the contact-and-impulse transactional exchange]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumFieldAfter`        [the body momentum after the selected-site exchange]

    The accepted result is the updated field and body, not the inputs, and this is the real
    instance of `ringExchangeTransaction_accepts`. -/
theorem exchangeTransaction_accepts (pos : BasisPositionField n) (mom : BasisMomentumField n)
    (s : Fin n) (bodyPos bodyMom claimed : Cell) (hcontact : ContactAt pos s bodyPos)
    (himpulse : claimed = exchangeImpulse (mom s) bodyMom) :
    exchangeTransaction pos mom s bodyPos bodyMom claimed
      = some (fieldMomentumAfter mom s bodyMom, bodyMomentumFieldAfter mom s bodyMom) :=
  ringExchangeTransaction_accepts (R := ℝ) pos mom s bodyPos bodyMom claimed hcontact himpulse

/-- A body position off the selected site is rejected.

    (pos : BasisPositionField n)    [the finite position field]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : Cell)                [the body position, a basis point]
    (bodyMom : Cell)                [the body momentum, a basis point]
    (claimed : Cell)                [the impulse the caller claims to hand over]
    (h : ¬ ContactAt pos s bodyPos) [contact does not hold]
    ¬                               [this is not true]
    Option (BasisMomentumField n × Cell)
                                     [either the updated field and body, or nothing]
    `ContactAt`                     [the body position equals the selected site's basis position]
    `exchangeTransaction`           [the contact-and-impulse transactional exchange]

    No update is produced when contact fails, whatever the claimed impulse. This is the real
    instance of `ringExchangeTransaction_rejects_position`. -/
theorem exchangeTransaction_rejects_position (pos : BasisPositionField n)
    (mom : BasisMomentumField n) (s : Fin n) (bodyPos bodyMom claimed : Cell)
    (h : ¬ ContactAt pos s bodyPos) :
    exchangeTransaction pos mom s bodyPos bodyMom claimed = none :=
  ringExchangeTransaction_rejects_position (R := ℝ) pos mom s bodyPos bodyMom claimed h

/-- A claimed impulse that is not the derived impulse is rejected.

    (pos : BasisPositionField n)    [the finite position field]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : Cell)                [the body position, a basis point]
    (bodyMom : Cell)                [the body momentum, a basis point]
    (claimed : Cell)                [the impulse the caller claims to hand over]
    (h : claimed ≠ exchangeImpulse (mom s) bodyMom)
                                     [the claim differs from the derived impulse]
    ≠                               [is not equal to]
    Option (BasisMomentumField n × Cell)
                                     [either the updated field and body, or nothing]
    `exchangeImpulse`               [the impulse derived from the two basis momenta]
    `exchangeTransaction`           [the contact-and-impulse transactional exchange]

    Contact alone does not license an independently supplied impulse. This is the real instance
    of `ringExchangeTransaction_rejects_impulse`. -/
theorem exchangeTransaction_rejects_impulse (pos : BasisPositionField n)
    (mom : BasisMomentumField n) (s : Fin n) (bodyPos bodyMom claimed : Cell)
    (h : claimed ≠ exchangeImpulse (mom s) bodyMom) :
    exchangeTransaction pos mom s bodyPos bodyMom claimed = none :=
  ringExchangeTransaction_rejects_impulse (R := ℝ) pos mom s bodyPos bodyMom claimed h

/-- The rollback result: the accepted update, or the supplied field and body.

    (pos : BasisPositionField n)    [the finite position field]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : Cell)                [the body position, a basis point]
    (bodyMom : Cell)                [the body momentum, a basis point]
    (claimed : Cell)                [the impulse the caller claims to hand over]
    BasisMomentumField n × Cell     [the field paired with the body]
    ×                               [paired with, so Cell × Cell means two points side by side]
    `exchangeTransaction`           [the contact-and-impulse transactional exchange]

    On rejection the supplied field and body are returned unchanged. This is the real instance of
    `ringExchangeOrRollback`. -/
noncomputable abbrev exchangeOrRollback (pos : BasisPositionField n) (mom : BasisMomentumField n)
    (s : Fin n) (bodyPos bodyMom claimed : Cell) : BasisMomentumField n × Cell :=
  ringExchangeOrRollback (R := ℝ) pos mom s bodyPos bodyMom claimed

/-- Rejected contact leaves the supplied field and body unchanged.

    (pos : BasisPositionField n)    [the finite position field]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : Cell)                [the body position, a basis point]
    (bodyMom : Cell)                [the body momentum, a basis point]
    (claimed : Cell)                [the impulse the caller claims to hand over]
    (h : ¬ ContactAt pos s bodyPos) [contact does not hold]
    ¬                               [this is not true]
    BasisMomentumField n × Cell     [the field paired with the body]
    `exchangeOrRollback`            [the accepted update or the supplied state]

    The rollback returns exactly the inputs, and this is the real instance of
    `ringExchangeOrRollback_of_rejected_position`. -/
theorem exchangeOrRollback_of_rejected_position (pos : BasisPositionField n)
    (mom : BasisMomentumField n) (s : Fin n) (bodyPos bodyMom claimed : Cell)
    (h : ¬ ContactAt pos s bodyPos) :
    exchangeOrRollback pos mom s bodyPos bodyMom claimed = (mom, bodyMom) :=
  ringExchangeOrRollback_of_rejected_position (R := ℝ) pos mom s bodyPos bodyMom claimed h

/-- A non-derived impulse leaves the supplied field and body unchanged.

    (pos : BasisPositionField n)    [the finite position field]
    (mom : BasisMomentumField n)    [the finite momentum field]
    (s : Fin n)                     [the selected site]
    (bodyPos : Cell)                [the body position, a basis point]
    (bodyMom : Cell)                [the body momentum, a basis point]
    (claimed : Cell)                [the impulse the caller claims to hand over]
    (h : claimed ≠ exchangeImpulse (mom s) bodyMom)
                                     [the claim differs from the derived impulse]
    ≠                               [is not equal to]
    BasisMomentumField n × Cell     [the field paired with the body]
    `exchangeOrRollback`            [the accepted update or the supplied state]

    The rollback returns exactly the inputs, and this is the real instance of
    `ringExchangeOrRollback_of_rejected_impulse`. -/
theorem exchangeOrRollback_of_rejected_impulse (pos : BasisPositionField n)
    (mom : BasisMomentumField n) (s : Fin n) (bodyPos bodyMom claimed : Cell)
    (h : claimed ≠ exchangeImpulse (mom s) bodyMom) :
    exchangeOrRollback pos mom s bodyPos bodyMom claimed = (mom, bodyMom) :=
  ringExchangeOrRollback_of_rejected_impulse (R := ℝ) pos mom s bodyPos bodyMom claimed h

/-- The selected site of the concrete two-site witness. -/
def witnessSite : Fin 2 := 0

/-- The position field of the concrete two-site witness. -/
def witnessPosition : BasisPositionField 2 := fun i => if i = 0 then (0, 0) else (1, 0)

/-- The momentum field of the concrete two-site witness. -/
def witnessMomenta : BasisMomentumField 2 := fun i => if i = 0 then (3, 1) else (5, -2)

/-- The body position of the concrete witness, placed on the selected site.

    Cell                            [a point: two real numbers written side by side, a + b·x]

    The first coordinate is the site's own basis position. -/
def witnessBodyPosition : Cell := (0, 0)

/-- The body momentum of the concrete witness.

    Cell                            [a point: two real numbers written side by side, a + b·x]

    Distinct from the selected site momentum, so the exchange is nontrivial. -/
def witnessBodyMomentum : Cell := (1, 2)

/-- The claimed impulse of the concrete witness: the derived one.

    Cell                            [a point: two real numbers written side by side, a + b·x]

    It is defined as `exchangeImpulse`, so the witness exercises the accepted path. -/
def witnessClaim : Cell := exchangeImpulse (witnessMomenta witnessSite) witnessBodyMomentum

/-- A body position off the selected site, the negative control.

    Cell                            [a point: two real numbers written side by side, a + b·x]

    The first coordinate is not the selected site's basis position. -/
def witnessWrongPosition : Cell := (7, 0)

/-- The witness body is in contact with the selected site.

    `ContactAt`                     [the body position equals the selected site's basis position]

    The body position and the site position are both the zero point. -/
theorem witness_contact : ContactAt witnessPosition witnessSite witnessBodyPosition := rfl

/-- The selected witness site holds the momentum `(3, 1)`.

    A concrete evaluation of the witness momentum field at its selected site. -/
theorem witness_selected_momentum : witnessMomenta witnessSite = (3, 1) := by
  unfold witnessMomenta witnessSite
  rw [if_pos rfl]

/-- The derived witness impulse is `(2, -1)`.

    `exchangeImpulse`               [the impulse derived from the two basis momenta]

    A concrete evaluation of the derived impulse at the witness. -/
theorem witness_derived_impulse :
    exchangeImpulse (witnessMomenta witnessSite) witnessBodyMomentum = (2, -1) := by
  rw [witness_selected_momentum]
  norm_num [ringExchangeImpulse, witnessBodyMomentum]

/-- The witness impulse is nonzero.

    ≠                               [is not equal to]
    `exchangeImpulse`               [the impulse derived from the two basis momenta]

    The two witness momenta differ, so the exchange moves both. -/
theorem witness_impulse_nonzero :
    exchangeImpulse (witnessMomenta witnessSite) witnessBodyMomentum ≠ 0 := by
  rw [witness_derived_impulse]
  intro h
  have hf : (2 : ℝ) = 0 := congrArg Prod.fst h
  norm_num at hf

/-- The witness selected site swaps its momentum and the body receives it.

    ∧                               [both are true]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumFieldAfter`        [the body momentum after the selected-site exchange]

    The selected site ends at the body momentum and the body ends at the site momentum. -/
theorem witness_selected_site :
    fieldMomentumAfter witnessMomenta witnessSite witnessBodyMomentum witnessSite
        = witnessBodyMomentum ∧
      bodyMomentumFieldAfter witnessMomenta witnessSite witnessBodyMomentum = (3, 1) := by
  constructor
  · rw [fieldMomentumAfter_site_eq_body]
  · unfold bodyMomentumFieldAfter ringBodyMomentumFieldAfter
    rw [ringBodyMomentumAfter_eq, witness_selected_momentum]

/-- The other witness site is unchanged by the exchange.

    `fieldMomentumAfter`            [the momentum field with the selected site updated]

    The site away from the selected one keeps its momentum. -/
theorem witness_other_site :
    fieldMomentumAfter witnessMomenta witnessSite witnessBodyMomentum 1 = witnessMomenta 1 := by
  apply fieldMomentumAfter_other
  decide

/-- The witness transaction accepts the canonical contact and derived impulse.

    `exchangeTransaction`           [the contact-and-impulse transactional exchange]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumFieldAfter`        [the body momentum after the selected-site exchange]

    The concrete instance of the accepted path. -/
theorem witness_transaction_accepts :
    exchangeTransaction witnessPosition witnessMomenta witnessSite witnessBodyPosition
        witnessBodyMomentum witnessClaim
      = some (fieldMomentumAfter witnessMomenta witnessSite witnessBodyMomentum,
              bodyMomentumFieldAfter witnessMomenta witnessSite witnessBodyMomentum) := by
  apply exchangeTransaction_accepts
  · rfl
  · rfl

/-- The witness rejects a body position off the selected site.

    `exchangeTransaction`           [the contact-and-impulse transactional exchange]

    The concrete instance of the rejected position. -/
theorem witness_wrongPosition_rejected :
    exchangeTransaction witnessPosition witnessMomenta witnessSite witnessWrongPosition
        witnessBodyMomentum witnessClaim = none := by
  apply exchangeTransaction_rejects_position
  intro h
  have hf : (7 : ℝ) = 0 := by
    simpa [witnessWrongPosition, ringSitePosition, witnessPosition, witnessSite] using
      congrArg Prod.fst h
  norm_num at hf

/-- The witness rejects an independently supplied impulse.

    `exchangeTransaction`           [the contact-and-impulse transactional exchange]

    The claim `(0, 1)` is not the derived `(2, -1)`. -/
theorem witness_wrongImpulse_rejected :
    exchangeTransaction witnessPosition witnessMomenta witnessSite witnessBodyPosition
        witnessBodyMomentum (0, 1) = none := by
  apply exchangeTransaction_rejects_impulse
  rw [witness_derived_impulse]
  intro h
  have hf : (0 : ℝ) = 2 := congrArg Prod.fst h
  norm_num at hf

/-- The witness conserves the whole-region basis momentum.

    `totalMomentum`                 [every site plus the body, as one basis point]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumFieldAfter`        [the body momentum after the selected-site exchange]

    The concrete instance of `totalFieldMomentum_conserved`. -/
theorem witness_totalFieldMomentum :
    totalMomentum (fieldMomentumAfter witnessMomenta witnessSite witnessBodyMomentum)
        (bodyMomentumFieldAfter witnessMomenta witnessSite witnessBodyMomentum)
      = totalMomentum witnessMomenta witnessBodyMomentum :=
  totalFieldMomentum_conserved witnessMomenta witnessSite witnessBodyMomentum

/-- The witness conserves the norm-form ledger at every σ.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    ℝ                               [the real numbers: every number on the number line]
    `normFormFieldLedger`           [the finite norm-form sum over every site plus the body]
    `fieldMomentumAfter`            [the momentum field with the selected site updated]
    `bodyMomentumFieldAfter`        [the body momentum after the selected-site exchange]

    The concrete instance of `normFormFieldLedger_conserved`. -/
theorem witness_ledger (σ : ℝ) :
    normFormFieldLedger σ (fieldMomentumAfter witnessMomenta witnessSite witnessBodyMomentum)
        (bodyMomentumFieldAfter witnessMomenta witnessSite witnessBodyMomentum)
      = normFormFieldLedger σ witnessMomenta witnessBodyMomentum :=
  normFormFieldLedger_conserved σ witnessMomenta witnessSite witnessBodyMomentum

end Physics
end Apeiron
