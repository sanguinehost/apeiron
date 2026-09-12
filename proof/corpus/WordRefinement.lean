import ApeironWord
import Denotation
import Apeiron.Trichotomy.AnyRing
import Apeiron.Physics.CarrierAnyRing
import Apeiron.Physics.CarrierBodyWorkAnyRing
import Apeiron.Physics.CarrierScalarKernelAnyRing

/-!
# The extracted word kernels refine the canonical algebra

`word_mul` and `word_norm` are the extracted `Cell<i64>::mul` and `Cell<i64>::norm_form`. On
`ZMod (2 ^ 64)` — the ring the generated `i64` arithmetic actually runs in — they agree with the
canonical `Apeiron.Trichotomy.AnyRing.ringMul` and `ringNormForm` instantiated at that ring.

The dial is carried through symbolically as `denoteSigma`; nothing here needs to know its three
numeric values. Splitting on the three constructors makes the extracted monadic `ok` chains reduce,
so each generated result is the total `ok` of a concrete word computation.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-- The dial's value as a machine word, definitionally equal to what `cell.Sigma.value` returns. -/
def sigmaI64 : cell.Sigma → I64
  | cell.Sigma.Elliptic => (-1)#i64
  | cell.Sigma.Parabolic => 0#i64
  | cell.Sigma.Hyperbolic => 1#i64

/-- The dial as an element of `ZMod (2 ^ 64)`. -/
def denoteSigma (s : cell.Sigma) : ZMod (2 ^ 64) := denote (sigmaI64 s)

/-- A cell's denotation: the two slots in `ZMod (2 ^ 64)`. -/
def denoteCell (z : cell.Cell I64) : Apeiron.Trichotomy.AnyRing.RingCell (ZMod (2 ^ 64)) :=
  (denote z.a, denote z.b)

theorem sigma_value_eq (s : cell.Sigma) : cell.Sigma.value s = ok (sigmaI64 s) := by
  cases s <;> rfl

/-- The extracted multiplication refines the canonical `ringMul`. -/
theorem word_mul_denote (a : cell.Cell I64) (s : cell.Sigma) (b : cell.Cell I64) :
    (do let c ← word_mul a s b; ok (denoteCell c)) =
      ok (Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell a) (denoteCell b)) := by
  cases s <;>
    simp [word_mul, cell.Cell.mul, cell.Sigma.value, sigmaI64, denoteCell, denoteSigma,
      I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub,
      I64.Insts.Apeiron_wordScalarRing.mul, I64.Insts.Apeiron_wordScalarRing.from_i64,
      Apeiron.Trichotomy.AnyRing.ringMul]

/-- The extracted norm form refines the canonical `ringNormForm`. -/
theorem word_norm_denote (a : cell.Cell I64) (s : cell.Sigma) :
    (do let r ← word_norm a s; ok (denote r)) =
      ok (Apeiron.Trichotomy.AnyRing.ringNormForm (denoteSigma s) (denoteCell a)) := by
  cases s <;>
    simp [word_norm, cell.Cell.norm_form, cell.Sigma.value, sigmaI64, denoteCell, denoteSigma,
      I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.mul, I64.Insts.Apeiron_wordScalarRing.sub,
      I64.Insts.Apeiron_wordScalarRing.from_i64,
      Apeiron.Trichotomy.AnyRing.ringNormForm] <;>
    congr 1 <;>
    ring

/-- Conjugation by negating the second slot. -/
def cellConj (a : cell.Cell I64) : cell.Cell I64 :=
  { a := a.a, b := core.num.I64.wrapping_sub 0#i64 a.b }

/-- The extracted conjugation denotes the canonical `AnyRing.ringConj`. -/
theorem denoteCell_cellConj (a : cell.Cell I64) :
    denoteCell (cellConj a) = Apeiron.Trichotomy.AnyRing.ringConj (denoteCell a) := by
  simp [cellConj, denoteCell, Apeiron.Trichotomy.AnyRing.ringConj, zero_sub]

/-- The extracted multiplication is total: it always returns `ok` a cell. -/
theorem word_mul_total (a : cell.Cell I64) (s : cell.Sigma) (b : cell.Cell I64) :
    ∃ d, word_mul a s b = ok d := by
  refine ⟨{ a := core.num.I64.wrapping_add
              (core.num.I64.wrapping_mul a.a b.a)
              (core.num.I64.wrapping_mul (core.num.I64.wrapping_mul (sigmaI64 s) a.b) b.b),
            b := core.num.I64.wrapping_add
              (core.num.I64.wrapping_mul a.a b.b)
              (core.num.I64.wrapping_mul a.b b.a) }, ?_⟩
  cases s <;>
    simp [word_mul, cell.Cell.mul, cell.Sigma.value, sigmaI64,
      I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.mul,
      I64.Insts.Apeiron_wordScalarRing.from_i64]

/-- The extracted norm form is total: it always returns `ok` a word. -/
theorem word_norm_total (a : cell.Cell I64) (s : cell.Sigma) :
    ∃ r, word_norm a s = ok r := by
  refine ⟨core.num.I64.wrapping_sub
            (core.num.I64.wrapping_mul a.a a.a)
            (core.num.I64.wrapping_mul
              (core.num.I64.wrapping_mul (sigmaI64 s) a.b) a.b), ?_⟩
  cases s <;>
    simp [word_norm, cell.Cell.norm_form, cell.Sigma.value, sigmaI64,
      I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.mul, I64.Insts.Apeiron_wordScalarRing.sub,
      I64.Insts.Apeiron_wordScalarRing.from_i64]

/-- Transfer of `Apeiron.Trichotomy.AnyRing.ringNormForm_mul` through the link: the norm form of
the extracted product is the product of the extracted norm forms.

The proof composes `word_mul_denote` and `word_norm_denote` with the ring-generic
`ringNormForm_mul`; it does not re-derive the polynomial identity. -/
theorem word_norm_mul (a : cell.Cell I64) (s : cell.Sigma) (b : cell.Cell I64) :
    (do
      let c ← word_mul a s b
      let n ← word_norm c s
      ok (denote n)) =
    (do
      let na ← word_norm a s
      let nb ← word_norm b s
      ok (denote na * denote nb)) := by
  obtain ⟨c, hc⟩ := word_mul_total a s b
  obtain ⟨na, hna⟩ := word_norm_total a s
  obtain ⟨nb, hnb⟩ := word_norm_total b s
  have hdc : denoteCell c =
      Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell a) (denoteCell b) := by
    have h := word_mul_denote a s b
    rw [hc] at h
    simpa only [bind_ok, bind_tc_ok, Result.ok.injEq] using h
  have hda : denote na =
      Apeiron.Trichotomy.AnyRing.ringNormForm (denoteSigma s) (denoteCell a) := by
    have h := word_norm_denote a s
    rw [hna] at h
    simpa only [bind_ok, bind_tc_ok, Result.ok.injEq] using h
  have hdb : denote nb =
      Apeiron.Trichotomy.AnyRing.ringNormForm (denoteSigma s) (denoteCell b) := by
    have h := word_norm_denote b s
    rw [hnb] at h
    simpa only [bind_ok, bind_tc_ok, Result.ok.injEq] using h
  rw [hc, hna, hnb]
  simp only [bind_ok, bind_tc_ok]
  rw [word_norm_denote c s, hdc, Apeiron.Trichotomy.AnyRing.ringNormForm_mul, ← hda, ← hdb]

/-- Transfer of `Apeiron.Trichotomy.AnyRing.ringMul_conj` through the link: the extracted product
with the conjugate has the extracted norm form in the first slot and zero in the second.

The proof composes `word_mul_denote` with `denoteCell_cellConj` and the ring-generic
`ringMul_conj`; it does not re-derive the identity. -/
theorem word_mul_conj (a : cell.Cell I64) (s : cell.Sigma) :
    (do let c ← word_mul a s (cellConj a); ok (denoteCell c)) =
      ok (Apeiron.Trichotomy.AnyRing.ringNormForm (denoteSigma s) (denoteCell a),
        (0 : ZMod (2 ^ 64))) := by
  rw [word_mul_denote a s (cellConj a), denoteCell_cellConj,
    Apeiron.Trichotomy.AnyRing.ringMul_conj]

/-!
## The extracted point-density and point-flux kernels refine the canonical carrier readout

`word_point_density` and `word_point_flux` are the extracted `i64` specialisations of the generic
`carrier::point_density` / `carrier::point_flux`. On `ZMod (2 ^ 64)` they agree with
`Apeiron.Physics.pointDensity` and `Apeiron.Physics.pointFlux`, the carrier readout over an
arbitrary commutative ring, with the dial carried as `denoteSigma`. The proofs compose the existing
`word_mul_denote` and `word_norm_denote`; they do not re-expand the polynomial. At the unit
direction the dial drops out, matching `pointDensity_ringOne_independent` /
`pointFlux_ringOne_independent`.
-/

/-- The zero cell the density kernel builds: amplitude in the first slot, `R::ZERO` in the
second. -/
def zeroCell (a : I64) : cell.Cell I64 :=
  { a := a, b := I64.Insts.Apeiron_wordScalarRing.ZERO }

/-- The zero cell denotes `(denote a, 0)`. -/
theorem denoteCell_zeroCell (a : I64) :
    denoteCell (zeroCell a) = (denote a, (0 : ZMod (2 ^ 64))) := by
  simp [zeroCell, denoteCell, I64.Insts.Apeiron_wordScalarRing.ZERO, denote_zero]

/-- The unit direction as a machine cell: `(1, 0)`. -/
def unitCell : cell.Cell I64 :=
  { a := 1#i64, b := 0#i64 }

/-- The unit machine direction denotes the canonical `ringOne`. -/
theorem denoteCell_unitCell :
    denoteCell unitCell = Apeiron.Trichotomy.AnyRing.ringOne := by
  simp [unitCell, denoteCell, denote_zero, denote_one, Apeiron.Trichotomy.AnyRing.ringOne]

/-- The extracted multiplication followed by the extracted norm form denotes the canonical norm
form of the canonical product.

The proof composes `word_mul_denote` and `word_norm_denote`; it does not re-derive the polynomial
identity. This is the shared step of both point readouts, which are norm forms of a product. -/
theorem word_norm_of_mul_denote (a : cell.Cell I64) (s : cell.Sigma) (b : cell.Cell I64) :
    (do
      let c ← word_mul a s b
      let r ← word_norm c s
      ok (denote r)) =
    ok (Apeiron.Trichotomy.AnyRing.ringNormForm (denoteSigma s)
      (Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell a) (denoteCell b))) := by
  obtain ⟨c, hc⟩ := word_mul_total a s b
  have hdc : denoteCell c =
      Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell a) (denoteCell b) := by
    have h := word_mul_denote a s b
    rw [hc] at h
    simpa only [bind_ok, bind_tc_ok, Result.ok.injEq] using h
  rw [hc]
  simp only [bind_ok, bind_tc_ok]
  rw [word_norm_denote c s, hdc]

/-- The extracted density kernel is the composition of the extracted multiplication and norm form
on the cell `(a, 0)`. This is the production `carrier::point_density` body specialised to `i64`;
the proof only unfolds definitions, it does not restate the algebra. -/
theorem word_point_density_unfold (s : cell.Sigma) (a : I64) (u : cell.Cell I64) :
    word_point_density s a u =
      (do let c ← word_mul (zeroCell a) s u; word_norm c s) := by
  simp only [word_point_density, carrier.point_density, zeroCell, cell.Cell.new,
    I64.Insts.Apeiron_wordScalarRing, bind_ok, bind_tc_ok, bind_assoc_eq]
  rfl

/-- The extracted point density refines the canonical `pointDensity`, at every dial value
`denoteSigma s`. The kernel builds `(a, 0)`, multiplies by the direction and takes the norm form;
the proof composes the two existing refinements rather than restating the algebra. -/
theorem word_point_density_denote (s : cell.Sigma) (a : I64) (u : cell.Cell I64) :
    (do let r ← word_point_density s a u; ok (denote r)) =
      ok (Apeiron.Physics.pointDensity (denoteSigma s) (denote a) (denoteCell u)) := by
  rw [word_point_density_unfold]
  simp only [bind_assoc_eq]
  rw [word_norm_of_mul_denote]
  congr 1
  rw [denoteCell_zeroCell]
  simp only [Apeiron.Physics.pointDensity]

/-- The extracted point density is total: it always returns `ok` a word. -/
theorem word_point_density_total (s : cell.Sigma) (a : I64) (u : cell.Cell I64) :
    ∃ r, word_point_density s a u = ok r := by
  rw [word_point_density_unfold]
  obtain ⟨c, hc⟩ := word_mul_total (zeroCell a) s u
  obtain ⟨r, hr⟩ := word_norm_total c s
  refine ⟨r, ?_⟩
  rw [hc]
  simpa only [bind_tc_ok] using hr

/-- The extracted point density factors into the amplitude's norm form and the direction's norm
form, the transfer of the canonical `pointDensity_eq` through `word_norm_denote`. -/
theorem word_point_density_eq (s : cell.Sigma) (a : I64) (u : cell.Cell I64) :
    (do let r ← word_point_density s a u; ok (denote r)) =
      ok (Apeiron.Trichotomy.AnyRing.ringNormForm (denoteSigma s)
            (denote a, (0 : ZMod (2 ^ 64))) *
          Apeiron.Trichotomy.AnyRing.ringNormForm (denoteSigma s) (denoteCell u)) := by
  rw [word_point_density_denote]
  congr 1
  rw [Apeiron.Physics.pointDensity_eq]

/-- The extracted point flux is the extracted density followed by the forward phase difference.
This is the production `carrier::point_flux` body specialised to `i64`. -/
theorem word_point_flux_unfold (s : cell.Sigma) (a : I64) (u : cell.Cell I64) (p n : I64) :
    (do let r ← word_point_flux s a u p n; ok (denote r)) =
      (do
        let t ← word_point_density s a u
        let t1 ← I64.Insts.Apeiron_wordScalarRing.sub n p
        let r ← I64.Insts.Apeiron_wordScalarRing.mul t t1
        ok (denote r)) := by
  simp only [word_point_flux, carrier.point_flux, bind_assoc_eq]
  rfl

/-- The extracted point flux refines the canonical `pointFlux`, at every dial value
`denoteSigma s`. The kernel takes the point density and multiplies by the forward phase difference
`next - previous`; the proof composes the density refinement with the wrapping `sub`/`mul`
denotations. -/
theorem word_point_flux_denote (s : cell.Sigma) (a : I64) (u : cell.Cell I64) (p n : I64) :
    (do let r ← word_point_flux s a u p n; ok (denote r)) =
      ok (Apeiron.Physics.pointFlux (denoteSigma s) (denote a) (denoteCell u)
        (denote p) (denote n)) := by
  rw [word_point_flux_unfold]
  obtain ⟨d, hd⟩ := word_point_density_total s a u
  have hdd : denote d =
      Apeiron.Physics.pointDensity (denoteSigma s) (denote a) (denoteCell u) := by
    have h := word_point_density_denote s a u
    rw [hd] at h
    simpa only [bind_ok, bind_tc_ok, Result.ok.injEq] using h
  rw [hd]
  simp only [bind_ok, bind_tc_ok, I64.Insts.Apeiron_wordScalarRing.sub,
    I64.Insts.Apeiron_wordScalarRing.mul, denote_wrapping_sub, denote_wrapping_mul]
  rw [hdd]
  simp [Apeiron.Physics.pointFlux]

/-- At the unit direction the extracted density is the amplitude squared, at any dial: the transfer
of `pointDensity_ringOne`. -/
theorem word_point_density_ringOne (s : cell.Sigma) (a : I64) :
    (do let r ← word_point_density s a unitCell; ok (denote r)) = ok (denote a ^ 2) := by
  rw [word_point_density_denote, denoteCell_unitCell,
    Apeiron.Physics.pointDensity_ringOne]

/-- At the unit direction the extracted density does not consult the dial: the transfer of
`pointDensity_ringOne_independent`. -/
theorem word_point_density_ringOne_independent (s s' : cell.Sigma) (a : I64) :
    (do let r ← word_point_density s a unitCell; ok (denote r)) =
    (do let r ← word_point_density s' a unitCell; ok (denote r)) := by
  rw [word_point_density_ringOne, word_point_density_ringOne]

/-- At the unit direction the extracted flux is the amplitude squared times the phase difference,
at any dial: the transfer of `pointFlux_ringOne`. -/
theorem word_point_flux_ringOne (s : cell.Sigma) (a p n : I64) :
    (do let r ← word_point_flux s a unitCell p n; ok (denote r)) =
      ok (denote a ^ 2 * (denote n - denote p)) := by
  rw [word_point_flux_denote, denoteCell_unitCell, Apeiron.Physics.pointFlux_ringOne]

/-- At the unit direction the extracted flux does not consult the dial: the transfer of
`pointFlux_ringOne_independent`. -/
theorem word_point_flux_ringOne_independent (s s' : cell.Sigma) (a p n : I64) :
    (do let r ← word_point_flux s a unitCell p n; ok (denote r)) =
    (do let r ← word_point_flux s' a unitCell p n; ok (denote r)) := by
  rw [word_point_flux_ringOne, word_point_flux_ringOne]

/-!
## The extracted exchange kernels refine the canonical ring exchange

`word_exchange_impulse` and `word_elastic_exchange` are the extracted `i64` exchange kernels. On
`ZMod (2 ^ 64)` they agree with `Apeiron.Physics.ringExchangeImpulse` and the two canonical
momentum updates from `Apeiron.Physics.CarrierBodyWorkAnyRing`. The exchange is additive and never
consults the dial, so the transfers below carry `denoteSigma` symbolically and need no numeric
case analysis; the norm-form ledger conservation is the canonical theorem instantiated at
`ZMod (2 ^ 64)`.

The checked-claim kernel `accept_derived_impulse` is extracted: production now selects the branch
with an explicit `if`, so Charon translates the derived `Cell` equality and the branch rather than
the external `core.bool.Bool.then` combinator. The generated equality is componentwise on the two
word slots. The theorems below select the branch by that **source-word equality** and carry the
result over with `word_elastic_exchange_denote`; no `ZMod (2^64)` equality is read back as word
equality.
-/

/-- The extracted derived impulse refines the canonical `ringExchangeImpulse`. -/
theorem word_exchange_impulse_denote (p q : cell.Cell I64) :
    (do let r ← word_exchange_impulse p q; ok (denoteCell r)) =
      ok (Apeiron.Physics.ringExchangeImpulse (denoteCell p) (denoteCell q)) := by
  simp [word_exchange_impulse, exchange.exchange_impulse, cell.Cell.new,
    I64.Insts.Apeiron_wordScalarRing, I64.Insts.Apeiron_wordScalarRing.sub,
    denoteCell, Apeiron.Physics.ringExchangeImpulse]

/-- The extracted elastic exchange refines the canonical derived exchange, field by field. -/
theorem word_elastic_exchange_denote (p q : cell.Cell I64) :
    (do
      let r ← word_elastic_exchange p q
      ok (denoteCell r.impulse, denoteCell r.carrier_after, denoteCell r.body_after)) =
    ok (Apeiron.Physics.ringExchangeImpulse (denoteCell p) (denoteCell q),
      Apeiron.Physics.ringCarrierMomentumAfter (denoteCell p) (denoteCell q),
      Apeiron.Physics.ringBodyMomentumAfter (denoteCell p) (denoteCell q)) := by
  simp [word_elastic_exchange, exchange.elastic_exchange, exchange.exchange_impulse,
    cell.Cell.new, I64.Insts.Apeiron_wordScalarRing,
    I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub,
    denoteCell, Apeiron.Physics.ringExchangeImpulse, Apeiron.Physics.ringCarrierMomentumAfter,
    Apeiron.Physics.ringBodyMomentumAfter]

/-- The concrete `I64` scalar instance, named for the extracted kernel. -/
abbrev i64ScalarRing : scalar.Ring I64 := I64.Insts.Apeiron_wordScalarRing

/-- The derived impulse computation is total: it always returns `ok` a cell. -/
theorem word_exchange_impulse_total (p q : cell.Cell I64) :
    ∃ d, word_exchange_impulse p q = ok d := by
  exact ⟨{ a := core.num.I64.wrapping_sub p.a q.a,
           b := core.num.I64.wrapping_sub p.b q.b }, by
    simp [word_exchange_impulse, exchange.exchange_impulse, cell.Cell.new,
      i64ScalarRing, I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.sub]⟩

/-- The elastic exchange computation is total: it always returns `ok` an exchange. -/
theorem word_elastic_exchange_total (p q : cell.Cell I64) :
    ∃ e, word_elastic_exchange p q = ok e := by
  exact ⟨{ impulse := { a := core.num.I64.wrapping_sub p.a q.a,
                        b := core.num.I64.wrapping_sub p.b q.b },
           carrier_after :=
             { a := core.num.I64.wrapping_sub p.a (core.num.I64.wrapping_sub p.a q.a),
               b := core.num.I64.wrapping_sub p.b (core.num.I64.wrapping_sub p.b q.b) },
           body_after :=
             { a := core.num.I64.wrapping_add q.a (core.num.I64.wrapping_sub p.a q.a),
               b := core.num.I64.wrapping_add q.b (core.num.I64.wrapping_sub p.b q.b) } }, by
    simp [word_elastic_exchange, exchange.elastic_exchange, exchange.exchange_impulse,
      cell.Cell.new, i64ScalarRing, I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub]⟩

/-- Transfer of `ringCarrierMomentumAfter_eq`: the extracted carrier ends holding the body
momentum. The proof composes `word_elastic_exchange_denote` with the ring-generic swap. -/
theorem word_elastic_exchange_carrier (p q : cell.Cell I64) :
    (do let r ← word_elastic_exchange p q; ok (denoteCell r.carrier_after)) = ok (denoteCell q) := by
  obtain ⟨e, he⟩ := word_elastic_exchange_total p q
  have hden := word_elastic_exchange_denote p q
  rw [he] at hden
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq, Prod.mk.injEq] at hden
  obtain ⟨_, hcar, _⟩ := hden
  rw [he]
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq]
  rw [hcar, Apeiron.Physics.ringCarrierMomentumAfter_eq]

/-- Transfer of `ringBodyMomentumAfter_eq`: the extracted body ends holding the carrier
momentum. The proof composes `word_elastic_exchange_denote` with the ring-generic swap. -/
theorem word_elastic_exchange_body (p q : cell.Cell I64) :
    (do let r ← word_elastic_exchange p q; ok (denoteCell r.body_after)) = ok (denoteCell p) := by
  obtain ⟨e, he⟩ := word_elastic_exchange_total p q
  have hden := word_elastic_exchange_denote p q
  rw [he] at hden
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq, Prod.mk.injEq] at hden
  obtain ⟨_, _, hbody⟩ := hden
  rw [he]
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq]
  rw [hbody, Apeiron.Physics.ringBodyMomentumAfter_eq]

/-- Transfer of `ringTotalMomentum_conserved`: the extracted exchange conserves the total basis
momentum. The proof composes `word_elastic_exchange_denote` with the ring-generic conservation. -/
theorem word_elastic_exchange_totalMomentum (p q : cell.Cell I64) :
    (do
      let r ← word_elastic_exchange p q
      ok (denoteCell r.carrier_after + denoteCell r.body_after)) =
    ok (denoteCell p + denoteCell q) := by
  obtain ⟨e, he⟩ := word_elastic_exchange_total p q
  have hden := word_elastic_exchange_denote p q
  rw [he] at hden
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq, Prod.mk.injEq] at hden
  obtain ⟨_, hcar, hbody⟩ := hden
  rw [he]
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq]
  rw [hcar, hbody, Apeiron.Physics.ringTotalMomentum_conserved]

/-- Transfer of `ringNormFormLedger_conserved`: the extracted exchange conserves the canonical
norm-form ledger at every symbolic dial value `denoteSigma s`. The proof composes
`word_elastic_exchange_denote` with the ring-generic ledger conservation. -/
theorem word_elastic_exchange_normFormLedger (s : cell.Sigma) (p q : cell.Cell I64) :
    (do
      let r ← word_elastic_exchange p q
      ok (Apeiron.Physics.ringNormFormLedger (denoteSigma s) (denoteCell r.carrier_after)
        (denoteCell r.body_after))) =
    ok (Apeiron.Physics.ringNormFormLedger (denoteSigma s) (denoteCell p) (denoteCell q)) := by
  obtain ⟨e, he⟩ := word_elastic_exchange_total p q
  have hden := word_elastic_exchange_denote p q
  rw [he] at hden
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq, Prod.mk.injEq] at hden
  obtain ⟨_, hcar, hbody⟩ := hden
  rw [he]
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq]
  rw [hcar, hbody, Apeiron.Physics.ringNormFormLedger_conserved]

/-- Transfer of `ringExchangeImpulse_antisymm`: swapping the extracted momenta negates the
denoted derived impulse. The proof composes `word_exchange_impulse_denote` with the ring-generic
antisymmetry. -/
theorem word_exchange_impulse_antisymm (p q : cell.Cell I64) :
    (do let r ← word_exchange_impulse p q; ok (denoteCell r)) =
      (do let r ← word_exchange_impulse q p; ok (- denoteCell r)) := by
  obtain ⟨r, hr⟩ := word_exchange_impulse_total q p
  have hden := word_exchange_impulse_denote q p
  rw [hr] at hden
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq] at hden
  rw [word_exchange_impulse_denote p q, hr]
  simp only [bind_ok, bind_tc_ok, Result.ok.injEq]
  rw [hden, Apeiron.Physics.ringExchangeImpulse_antisymm]

/-!
### The checked-claim kernel

`word_accept_derived_impulse` is the extracted `accept_derived_impulse`. The generated
`cell.Cell.Insts.CoreCmpPartialEqCell.eq` compares the two slots in turn and the generated `if`
selects the branch, so acceptance is decided by **source-word equality** of the two `I64` slots.
The spec below therefore reads word equality directly; result correspondence is carried over from
`word_elastic_exchange_denote` rather than inferred from `ZMod (2^64)` equality.
-/

/-- Source-word equality of two cells is componentwise equality of the two `I64` slots. -/
theorem word_cell_ext_iff (a b : cell.Cell I64) :
    a = b ↔ a.a = b.a ∧ a.b = b.b := by
  constructor
  · intro h
    exact ⟨congrArg cell.Cell.a h, congrArg cell.Cell.b h⟩
  · intro h
    cases a
    cases b
    simp_all

/-- The generated derived equality compares the two word slots in turn; this is the exact branch
condition the extracted acceptance kernel selects on. -/
theorem word_cell_eq_eq (a b : cell.Cell I64) :
    cell.Cell.Insts.CoreCmpPartialEqCell.eq
        i64ScalarRing.corecmpPartialEqInst a b =
      ok (decide (a.a = b.a) && decide (a.b = b.b)) := by
  simp [cell.Cell.Insts.CoreCmpPartialEqCell.eq, i64ScalarRing,
    I64.Insts.Apeiron_wordScalarRing, core.cmp.PartialEqI64,
    core.cmp.impls.PartialEqI64.eq, liftFun2]
  by_cases h : a.a = b.a <;> simp [h]

/-- The checked-claim kernel returns `some` exactly when the claim is the derived word impulse
`d`; the branch is selected by source-word equality, not by `ZMod (2^64)` equality. -/
theorem word_accept_derived_impulse_eq_some_iff (p q claimed d : cell.Cell I64)
    (hd : word_exchange_impulse p q = ok d) :
    (∃ r, word_accept_derived_impulse p q claimed = ok (some r)) ↔ claimed = d := by
  have hd' : exchange.exchange_impulse i64ScalarRing p q = ok d := hd
  rw [word_accept_derived_impulse, exchange.accept_derived_impulse, hd']
  simp only [cell.Cell.Insts.CoreCmpPartialEqCell.eq, core.cmp.PartialEqI64,
    core.cmp.impls.PartialEqI64.eq, liftFun2]
  by_cases ha : claimed.a = d.a <;> by_cases hb : claimed.b = d.b <;>
    simp [ha, hb, hd', word_cell_ext_iff, word_elastic_exchange, exchange.elastic_exchange,
      cell.Cell.new, i64ScalarRing, I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub]

/-- The single specification of the checked-claim kernel: it returns `some` exactly when the
claim equals the impulse the extracted computation derives from the two momenta. -/
theorem word_accept_derived_impulse_spec (p q claimed : cell.Cell I64) :
    (∃ r, word_accept_derived_impulse p q claimed = ok (some r)) ↔
      ∃ d, word_exchange_impulse p q = ok d ∧ claimed = d := by
  constructor
  · rintro ⟨r, hr⟩
    obtain ⟨d, hd⟩ := word_exchange_impulse_total p q
    exact ⟨d, hd, (word_accept_derived_impulse_eq_some_iff p q claimed d hd).mp ⟨r, hr⟩⟩
  · rintro ⟨d, hd, hcd⟩
    exact (word_accept_derived_impulse_eq_some_iff p q claimed d hd).mpr hcd

/-- Every claim unequal to the derived word impulse is rejected with `ok none`. -/
theorem word_accept_derived_impulse_rejects (p q claimed d : cell.Cell I64)
    (hd : word_exchange_impulse p q = ok d) (hne : claimed ≠ d) :
    word_accept_derived_impulse p q claimed = ok none := by
  have hd' : exchange.exchange_impulse i64ScalarRing p q = ok d := hd
  rw [word_accept_derived_impulse, exchange.accept_derived_impulse, hd']
  simp only [cell.Cell.Insts.CoreCmpPartialEqCell.eq, core.cmp.PartialEqI64,
    core.cmp.impls.PartialEqI64.eq, liftFun2]
  by_cases ha : claimed.a = d.a
  · by_cases hb : claimed.b = d.b
    · exact absurd ((word_cell_ext_iff claimed d).mpr ⟨ha, hb⟩) hne
    · simp [ha, hb]
  · simp [ha]

/-- The non-parameterised rejection form: if no `ok d` produced by the impulse equals the claim,
the kernel rejects it. -/
theorem word_accept_derived_impulse_rejects_of_ne (p q claimed : cell.Cell I64)
    (h : ∀ d, word_exchange_impulse p q = ok d → claimed ≠ d) :
    word_accept_derived_impulse p q claimed = ok none := by
  obtain ⟨d, hd⟩ := word_exchange_impulse_total p q
  exact word_accept_derived_impulse_rejects p q claimed d hd (h d hd)

/-- The exact derived word impulse is accepted and returns the elastic exchange. -/
theorem word_accept_derived_impulse_accepts (p q d : cell.Cell I64)
    (hd : word_exchange_impulse p q = ok d) :
    word_accept_derived_impulse p q d =
      (do let e ← word_elastic_exchange p q; ok (some e)) := by
  have hd' : exchange.exchange_impulse i64ScalarRing p q = ok d := hd
  rw [word_accept_derived_impulse, exchange.accept_derived_impulse, hd']
  simp only [cell.Cell.Insts.CoreCmpPartialEqCell.eq, core.cmp.PartialEqI64,
    core.cmp.impls.PartialEqI64.eq, liftFun2]
  simp [word_elastic_exchange]

/-- The same acceptance, with the derived impulse bound inside the computation. -/
theorem word_accept_derived_impulse_accepts_derived (p q : cell.Cell I64) :
    (do
      let d ← word_exchange_impulse p q
      let r ← word_accept_derived_impulse p q d
      ok r) =
    (do
      let r ← word_elastic_exchange p q
      ok (some r)) := by
  obtain ⟨d, hd⟩ := word_exchange_impulse_total p q
  rw [hd]
  have hred :
      (do let d' ← (ok d : Result (cell.Cell I64))
          let r ← word_accept_derived_impulse p q d'
          ok r) =
      (do let r ← word_accept_derived_impulse p q d; ok r) := by simp
  rw [hred, word_accept_derived_impulse_accepts p q d hd]
  simp

/-- The accepted exchange denotes exactly what `word_elastic_exchange` denotes. -/
theorem word_accept_derived_impulse_denote (p q d : cell.Cell I64)
    (hd : word_exchange_impulse p q = ok d) :
    (do
      let r ← word_accept_derived_impulse p q d
      ok (r.map (fun e => (denoteCell e.impulse, denoteCell e.carrier_after,
        denoteCell e.body_after)))) =
    (do
      let r ← word_elastic_exchange p q
      ok (some (denoteCell r.impulse, denoteCell r.carrier_after,
        denoteCell r.body_after))) := by
  rw [word_accept_derived_impulse_accepts p q d hd]
  simp

/-- Composing the acceptance denotation with the existing `word_elastic_exchange_denote`: the
accepted exchange denotes the canonical ring exchange impulse and momentum updates. -/
theorem word_accept_derived_impulse_denote_canonical (p q d : cell.Cell I64)
    (hd : word_exchange_impulse p q = ok d) :
    (do
      let r ← word_accept_derived_impulse p q d
      ok (r.map (fun e => (denoteCell e.impulse, denoteCell e.carrier_after,
        denoteCell e.body_after)))) =
    ok (some (Apeiron.Physics.ringExchangeImpulse (denoteCell p) (denoteCell q),
      Apeiron.Physics.ringCarrierMomentumAfter (denoteCell p) (denoteCell q),
      Apeiron.Physics.ringBodyMomentumAfter (denoteCell p) (denoteCell q))) := by
  rw [word_accept_derived_impulse_denote p q d hd]
  obtain ⟨e, he⟩ := word_elastic_exchange_total p q
  rw [he]
  have hred :
      (do let r ← (ok e : Result (exchange.Exchange I64))
          ok (some (denoteCell r.impulse, denoteCell r.carrier_after,
            denoteCell r.body_after))) =
      ok (some (denoteCell e.impulse, denoteCell e.carrier_after,
        denoteCell e.body_after)) := by simp
  rw [hred]
  have hden := word_elastic_exchange_denote p q
  rw [he] at hden
  have hred2 :
      (do let r ← (ok e : Result (exchange.Exchange I64))
          ok (denoteCell r.impulse, denoteCell r.carrier_after,
            denoteCell r.body_after)) =
      ok (denoteCell e.impulse, denoteCell e.carrier_after,
        denoteCell e.body_after) := by simp
  rw [hred2] at hden
  rw [Result.ok_injective hden]

/-!
## The extracted scalar carrier step and exact-half flux refine the canonical kernel

`word_conservative_density_step` and `word_diffusive_flux_half` are the extracted `i64`
specialisations of the real `src/carrier.rs` kernels. On `ZMod (2 ^ 64)` the conservative step
agrees with `Apeiron.Physics.scalarCarrierStep` and the exact half-flux is the
`Apeiron.Physics.exactHalfFlux` witness `2 * f = left - right`.

Both kernels are **partial**: an intermediate that does not fit `i64`, or an odd face difference,
is refused as `ok none`. The theorems below state the exact generated no-overflow preconditions
and the representability/evenness hypotheses that make success possible; the refusal witnesses at
the end show the partiality is real. No theorem claims every integer state is iterable or that the
updated value remains a norm-form square.

`word_conservative_total_mass` then composes the per-cell refinement with the ring-generic
shared-face telescoping `Apeiron.Physics.sum_scalarCarrierStep_eq`: when every one of the `n`
calls actually returns `ok (some (result i))`, the denoted finite total changes only by the two
endpoint faces plus the sources. `word_conservative_total_mass_closed` is the closed zero-source
corollary.
-/

/-- The bit pattern of a signed integer is its value cast into `ZMod`. -/
theorem zmod_bmod (x : Int) (m : Nat) : ((x.bmod m : Int) : ZMod m) = (x : ZMod m) := by
  rw [Int.bmod_eq_emod]
  split_ifs <;> push_cast <;> simp

/-- The bit pattern of a machine word is its signed value cast into `ZMod (2 ^ 64)`. -/
theorem denote_eq_intCast (x : I64) : denote x = ((x.val : Int) : ZMod (2 ^ 64)) := by
  simp only [denote, IScalar.val, BitVec.toInt_eq_toNat_bmod]
  rw [zmod_bmod]
  exact Int.cast_natCast _

/-- A successful checked subtraction is ring subtraction under `denote`. -/
theorem denote_checked_sub {x y z : I64} (h : I64.checked_sub x y = some z) :
    denote z = denote x - denote y := by
  have hspec := I64.checked_sub_bv_spec x y
  have hb : core.num.checked_sub_IScalar x y = some z := by simpa only [I64.checked_sub] using h
  rw [hb] at hspec
  obtain ⟨_, _, hval, _⟩ := hspec
  rw [denote_eq_intCast z, denote_eq_intCast x, denote_eq_intCast y, hval]
  push_cast
  ring

/-- A successful checked addition is ring addition under `denote`. -/
theorem denote_checked_add {x y z : I64} (h : I64.checked_add x y = some z) :
    denote z = denote x + denote y := by
  have hspec := I64.checked_add_bv_spec x y
  have hb : core.num.checked_add_IScalar x y = some z := by simpa only [I64.checked_add] using h
  rw [hb] at hspec
  obtain ⟨_, _, hval, _⟩ := hspec
  rw [denote_eq_intCast z, denote_eq_intCast x, denote_eq_intCast y, hval]
  push_cast
  ring

/-- A checked subtraction of `i64` succeeds exactly on the representable differences. This is the
generated no-overflow precondition, stated over the two signed values. -/
theorem checked_sub_eq_some_iff (x y : I64) :
    (∃ z, I64.checked_sub x y = some z) ↔ I64.min ≤ x.val - y.val ∧ x.val - y.val ≤ I64.max := by
  have hspec := I64.checked_sub_bv_spec x y
  rw [show core.num.checked_sub_IScalar x y = I64.checked_sub x y from rfl] at hspec
  cases h : I64.checked_sub x y with
  | none =>
    simp only [h] at hspec
    constructor
    · rintro ⟨z, hz⟩; exact absurd hz (by simp)
    · intro hb; exact (hspec ⟨hb.1, hb.2⟩).elim
  | some z =>
    simp only [h] at hspec
    exact ⟨fun _ => ⟨hspec.1, hspec.2.1⟩, fun _ => ⟨z, rfl⟩⟩

/-- The extracted conservative step refines the canonical `scalarCarrierStep` whenever it
succeeds. The proof reads the three generated checked results and carries each across with the
wrapping denotation; it does not restate the update. -/
theorem word_conservative_density_step_denote
    (density incoming outgoing source r : I64)
    (h : word_conservative_density_step density incoming outgoing source = ok (some r)) :
    denote r = Apeiron.Physics.scalarCarrierStep
      (denote density) (denote outgoing) (denote incoming) (denote source) := by
  rw [Apeiron.Physics.scalarCarrierStep]
  simp only [word_conservative_density_step, carrier.conservative_density_step, lift, bind_tc_ok]
    at h
  cases hnet : I64.checked_sub outgoing incoming with
  | none => simp [hnet] at h
  | some net =>
    cases hafter : I64.checked_sub density net with
    | none => simp [hnet, hafter] at h
    | some after =>
      simp only [hnet, hafter] at h
      have hadd : I64.checked_add after source = some r := by simpa using h
      rw [denote_checked_add hadd, denote_checked_sub hafter, denote_checked_sub hnet]

/-- The extracted conservative step succeeds exactly when all three generated checked
intermediates do. This is the exact no-overflow precondition; the step is not total. -/
theorem word_conservative_density_step_eq_some_iff
    (density incoming outgoing source : I64) :
    (∃ r, word_conservative_density_step density incoming outgoing source = ok (some r)) ↔
      ∃ net after, I64.checked_sub outgoing incoming = some net ∧
        I64.checked_sub density net = some after ∧
        ∃ r, I64.checked_add after source = some r := by
  constructor
  · rintro ⟨r, h⟩
    simp only [word_conservative_density_step, carrier.conservative_density_step, lift, bind_tc_ok]
      at h
    cases hnet : I64.checked_sub outgoing incoming with
    | none => simp [hnet] at h
    | some net =>
      cases hafter : I64.checked_sub density net with
      | none => simp [hnet, hafter] at h
      | some after =>
        simp only [hnet, hafter] at h
        exact ⟨net, after, rfl, hafter, r, by simpa using h⟩
  · rintro ⟨net, after, hnet, hafter, r, hadd⟩
    simp only [word_conservative_density_step, carrier.conservative_density_step, lift, bind_tc_ok,
      hnet, hafter, hadd]
    exact ⟨r, rfl⟩

/-- Under the generated no-overflow preconditions, the extracted conservative step succeeds and
denotes the canonical `scalarCarrierStep`. -/
theorem word_conservative_density_step_success
    (density incoming outgoing source net after r : I64)
    (hnet : I64.checked_sub outgoing incoming = some net)
    (hafter : I64.checked_sub density net = some after)
    (hadd : I64.checked_add after source = some r) :
    word_conservative_density_step density incoming outgoing source = ok (some r) ∧
      denote r = Apeiron.Physics.scalarCarrierStep
        (denote density) (denote outgoing) (denote incoming) (denote source) := by
  have hsome : word_conservative_density_step density incoming outgoing source = ok (some r) := by
    simp only [word_conservative_density_step, carrier.conservative_density_step, lift, bind_tc_ok,
      hnet, hafter, hadd]
  exact ⟨hsome, word_conservative_density_step_denote density incoming outgoing source r hsome⟩

/-- `x != 0` is false exactly when `x` is the zero word. -/
theorem bne_zero_eq_false_of_val_eq_zero {x : I64} (h : x.val = 0) : (x != 0#i64) = false := by
  have hbv : x.bv = (0#i64).bv := by
    rw [show (0#i64).bv = 0 from by simp]
    apply BitVec.toInt_inj.mp
    rw [show x.bv.toInt = 0 from h]
    rfl
  rw [show x = 0#i64 from (IScalar.mk.injEq (ty := .I64) x.bv (0#i64).bv).mpr hbv]
  simp

/-- The truncating remainder by two vanishes exactly on the even integers. -/
theorem tmod_two_eq_zero_iff (n : Int) : n.tmod 2 = 0 ↔ n % 2 = 0 := by
  rw [Int.tmod_eq_emod]
  by_cases h0 : 0 ≤ n
  · rw [if_pos (Or.inl h0)]; simp
  · by_cases hd : (2 : Int) ∣ n
    · rw [if_pos (Or.inr hd)]; simp [Int.dvd_iff_emod_eq_zero.mp hd]
    · rw [if_neg (by simp [h0, hd])]
      rcases Int.emod_two_eq_zero_or_one n with h | h
      · exact absurd (Int.dvd_iff_emod_eq_zero.mpr h) hd
      · rw [h]; norm_num

/-- Checked remainder by two never overflows: the divisor is nonzero. -/
theorem checked_rem_two_isSome (x : I64) : ∃ r, I64.checked_rem x 2#i64 = some r := by
  have hs := I64.checked_rem_bv_spec x 2#i64
  cases h : I64.checked_rem x 2#i64 with
  | none =>
    rw [show core.num.checked_rem_IScalar x 2#i64 = I64.checked_rem x 2#i64 from rfl, h] at hs
    norm_num at hs
  | some r => exact ⟨r, rfl⟩

/-- Checked division by two never overflows: the divisor is nonzero and not `-1`. -/
theorem checked_div_two_isSome (x : I64) : ∃ h, I64.checked_div x 2#i64 = some h := by
  have hs := I64.checked_div_bv_spec x 2#i64
  cases h : I64.checked_div x 2#i64 with
  | none =>
    rw [show core.num.checked_div_IScalar x 2#i64 = I64.checked_div x 2#i64 from rfl, h] at hs
    rcases hs with h0 | ⟨_, hneg⟩
    · norm_num at h0
    · norm_num at hneg
  | some r => exact ⟨r, rfl⟩

/-- The negation `0 - x` succeeds whenever `x` is not the minimum word, which is the only word
whose negation is unrepresentable. -/
theorem checked_sub_zero_isSome {x : I64} (hlo : I64.min < x.val) (hhi : x.val ≤ I64.max) :
    ∃ r, I64.checked_sub 0#i64 x = some r := by
  rw [checked_sub_eq_some_iff]
  have h0 : (0#i64).val = 0 := by simp
  rw [h0]
  constructor <;> simp only [I64.min, I64.max, I64.numBits] at * <;> omega

/-- The extracted exact half-flux satisfies the canonical `exactHalfFlux` witness whenever it
succeeds. The result is `-(right - left) / 2`; the parity of the difference makes the halved
value an exact integer, and the witness records that without dividing in `ZMod (2 ^ 64)`. -/
theorem word_diffusive_flux_half_denote
    (left right f : I64)
    (h : word_diffusive_flux_half left right = ok (some f)) :
    Apeiron.Physics.exactHalfFlux (denote f) (denote left) (denote right) := by
  rw [Apeiron.Physics.exactHalfFlux, Apeiron.Physics.DoublesTo]
  simp only [word_diffusive_flux_half, carrier.diffusive_flux_half, lift, bind_tc_ok] at h
  cases hsub : I64.checked_sub right left with
  | none => simp [hsub] at h
  | some difference =>
    cases hrem : I64.checked_rem difference 2#i64 with
    | none => simp [hsub, hrem] at h
    | some remainder =>
      simp only [hsub, hrem] at h
      by_cases hcond : remainder != 0#i64
      · simp [hcond] at h
      · simp only [hcond] at h
        cases hdiv : I64.checked_div difference 2#i64 with
        | none => simp [hdiv] at h
        | some half =>
          simp only [hdiv] at h
          have hf : I64.checked_sub 0#i64 half = some f := by simpa using h
          have hrem0 : remainder.val = 0 := by simpa using hcond
          have hremval : remainder.val = difference.val.tmod 2 := by
            have hs := I64.checked_rem_bv_spec difference 2#i64
            have hb : core.num.checked_rem_IScalar difference 2#i64 = some remainder := by
              simpa only [I64.checked_rem] using hrem
            rw [hb] at hs
            simpa using hs.2.1
          have htmod : difference.val.tmod 2 = 0 := by
            rw [← hremval, hrem0]
          have hmod : difference.val % 2 = 0 := (tmod_two_eq_zero_iff difference.val).mp htmod
          have hdvd : (2 : Int) ∣ difference.val := Int.dvd_iff_emod_eq_zero.mpr hmod
          have hhalfval : half.val = difference.val.tdiv 2 := by
            have hs := I64.checked_div_bv_spec difference 2#i64
            have hb : core.num.checked_div_IScalar difference 2#i64 = some half := by
              simpa only [I64.checked_div] using hdiv
            rw [hb] at hs
            simpa using hs.2.2.1
          have hhalf : half.val = difference.val / 2 := by
            rw [hhalfval, Int.tdiv_eq_ediv, if_pos (Or.inr hdvd), add_zero]
          have htwice : 2 * half.val = difference.val := by
            have he := Int.mul_ediv_add_emod difference.val 2
            rw [hmod, add_zero] at he
            rw [← hhalf] at he
            exact he
          have hdenhalf : (2 : ZMod (2 ^ 64)) * denote half = denote difference := by
            rw [denote_eq_intCast half, denote_eq_intCast difference, ← Int.cast_ofNat,
              ← Int.cast_mul, htwice]
          have hfval : denote f = (0 : ZMod (2 ^ 64)) - denote half := by
            rw [denote_checked_sub hf]; simp
          calc
            2 * denote f = 2 * ((0 : ZMod (2 ^ 64)) - denote half) := by rw [hfval]
            _ = -(2 * denote half) := by ring
            _ = -(denote difference) := by rw [hdenhalf]
            _ = denote left - denote right := by
                  have hd := denote_checked_sub hsub
                  rw [hd]; ring

/-- The extracted exact half-flux succeeds under explicit representability and evenness of the
face difference, and the result is the canonical exact-half witness. The representability
hypothesis is `right - left ∈ [i64::MIN, i64::MAX]`; the evenness is what makes a half-unit flux
exist at all. -/
theorem word_diffusive_flux_half_success
    (left right : I64)
    (hfit : I64.min ≤ right.val - left.val ∧ right.val - left.val ≤ I64.max)
    (heven : Even (right.val - left.val)) :
    ∃ f, word_diffusive_flux_half left right = ok (some f) ∧
      Apeiron.Physics.exactHalfFlux (denote f) (denote left) (denote right) := by
  obtain ⟨difference, hdiff⟩ := (checked_sub_eq_some_iff right left).mpr hfit
  obtain ⟨remainder, hrem⟩ := checked_rem_two_isSome difference
  obtain ⟨half, hdiv⟩ := checked_div_two_isSome difference
  have hdiffval : difference.val = right.val - left.val := by
    have hs := I64.checked_sub_bv_spec right left
    have hb : core.num.checked_sub_IScalar right left = some difference := by
      simpa only [I64.checked_sub] using hdiff
    rw [hb] at hs
    simpa using hs.2.2.1
  have hremval : remainder.val = difference.val.tmod 2 := by
    have hs := I64.checked_rem_bv_spec difference 2#i64
    have hb : core.num.checked_rem_IScalar difference 2#i64 = some remainder := by
      simpa only [I64.checked_rem] using hrem
    rw [hb] at hs
    simpa using hs.2.1
  have hmod : difference.val % 2 = 0 := by
    have he : (right.val - left.val) % 2 = 0 := Int.even_iff.mp heven
    rw [hdiffval]; exact he
  have htmod : difference.val.tmod 2 = 0 := (tmod_two_eq_zero_iff _).mpr hmod
  have hrem0 : remainder.val = 0 := by rw [hremval, htmod]
  have hcond : (remainder != 0#i64) = false := bne_zero_eq_false_of_val_eq_zero hrem0
  have hhalfval : half.val = difference.val.tdiv 2 := by
    have hs := I64.checked_div_bv_spec difference 2#i64
    have hb : core.num.checked_div_IScalar difference 2#i64 = some half := by
      simpa only [I64.checked_div] using hdiv
    rw [hb] at hs
    simpa using hs.2.2.1
  have hdvd : (2 : Int) ∣ difference.val := Int.dvd_iff_emod_eq_zero.mpr hmod
  have hhalf : half.val = difference.val / 2 := by
    rw [hhalfval, Int.tdiv_eq_ediv, if_pos (Or.inr hdvd), add_zero]
  have hbound : I64.min < half.val ∧ half.val ≤ I64.max := by
    rw [hhalf, hdiffval]
    constructor <;> simp only [I64.min, I64.max, I64.numBits] at * <;> omega
  obtain ⟨f, hsub0⟩ := checked_sub_zero_isSome hbound.1 hbound.2
  have hrun : word_diffusive_flux_half left right = ok (some f) := by
    simp only [word_diffusive_flux_half, carrier.diffusive_flux_half, lift, bind_tc_ok,
      hdiff, hrem, hcond, hdiv, hsub0]
    simp
  exact ⟨f, hrun, word_diffusive_flux_half_denote left right f hrun⟩

/-- The exact half of `0` and `1` does not exist: the odd difference is refused, so the kernel is
not total over `i64`. -/
theorem word_diffusive_flux_half_odd_refusal :
    word_diffusive_flux_half 0#i64 1#i64 = ok none := by
  have h1 : I64.checked_sub 1#i64 0#i64 = some 1#i64 := by decide
  have h2 : I64.checked_rem 1#i64 2#i64 = some 1#i64 := by decide
  have h3 : (1#i64 != 0#i64) = true := by decide
  simp only [word_diffusive_flux_half, carrier.diffusive_flux_half, lift, bind_tc_ok, h1, h2, h3]
  simp

/-- An unrepresentable face difference is refused rather than wrapped: the kernel is not total over
`i64`. -/
theorem word_diffusive_flux_half_unrepresentable_refusal :
    word_diffusive_flux_half 9223372036854775807#i64 (-9223372036854775808)#i64 = ok none := by
  have h1 : I64.checked_sub (-9223372036854775808)#i64 9223372036854775807#i64 = none := by
    decide
  simp only [word_diffusive_flux_half, carrier.diffusive_flux_half, lift, bind_tc_ok, h1]

/-- The conservative step refuses an update whose intermediate does not fit `i64`, rather than
leaving the earlier steps as a partial result. -/
theorem word_conservative_density_step_overflow_refusal :
    word_conservative_density_step 9223372036854775807#i64 1#i64 0#i64 0#i64 = ok none := by
  have h1 : I64.checked_sub 0#i64 1#i64 = some (-1)#i64 := by decide
  have h2 : I64.checked_sub 9223372036854775807#i64 (-1)#i64 = none := by decide
  simp only [word_conservative_density_step, carrier.conservative_density_step, lift, bind_tc_ok,
    h1, h2]

/-- Pointwise successful extracted conservative steps on one shared face array change the denoted
finite total only by the endpoint flux plus the sources.

The hypothesis requires the actual success of every call, with the result carried explicitly, not
a conditional property of a partial result. The proof composes the per-cell denotation
`word_conservative_density_step_denote` with the ring-generic shared-face telescoping
`Apeiron.Physics.sum_scalarCarrierStep_eq`; it does not restate the arithmetic. -/
theorem word_conservative_total_mass
    (density face source result : ℕ → I64) (n : ℕ)
    (hok : ∀ i, i < n →
      word_conservative_density_step (density i) (face i) (face (i + 1)) (source i)
        = ok (some (result i))) :
    (∑ i ∈ Finset.range n, denote (result i))
      = (∑ i ∈ Finset.range n, denote (density i))
        - (denote (face n) - denote (face 0))
        + ∑ i ∈ Finset.range n, denote (source i) := by
  have hper : ∀ i ∈ Finset.range n,
      denote (result i) = Apeiron.Physics.scalarCarrierStep (denote (density i))
        (denote (face (i + 1))) (denote (face i)) (denote (source i)) := by
    intro i hi
    exact word_conservative_density_step_denote (density i) (face i) (face (i + 1))
      (source i) (result i) (hok i (Finset.mem_range.mp hi))
  calc
    (∑ i ∈ Finset.range n, denote (result i))
        = ∑ i ∈ Finset.range n, Apeiron.Physics.scalarCarrierStep (denote (density i))
            (denote (face (i + 1))) (denote (face i)) (denote (source i)) := by
          apply Finset.sum_congr rfl
          intro i hi
          rw [hper i hi]
    _ = (∑ i ∈ Finset.range n, denote (density i))
          - (denote (face n) - denote (face 0))
          + ∑ i ∈ Finset.range n, denote (source i) :=
          Apeiron.Physics.sum_scalarCarrierStep_eq (fun i => denote (density i))
            (fun i => denote (face i)) (fun i => denote (source i)) n

/-- A closed extracted line with no net source conserves the denoted total.

The closed endpoint faces and the vanishing source total are the hypotheses; the telescoping
`word_conservative_total_mass` supplies the rest. -/
theorem word_conservative_total_mass_closed
    (density face source result : ℕ → I64) (n : ℕ)
    (hok : ∀ i, i < n →
      word_conservative_density_step (density i) (face i) (face (i + 1)) (source i)
        = ok (some (result i)))
    (hLeft : denote (face 0) = 0) (hRight : denote (face n) = 0)
    (hSource : (∑ i ∈ Finset.range n, denote (source i)) = 0) :
    (∑ i ∈ Finset.range n, denote (result i))
      = ∑ i ∈ Finset.range n, denote (density i) := by
  rw [word_conservative_total_mass density face source result n hok, hLeft, hRight, hSource]
  ring

end apeiron_word
