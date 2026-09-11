import ApeironWord
import Denotation
import Apeiron.Trichotomy.AnyRing
import Apeiron.Physics.CarrierBodyWorkAnyRing

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

/-- Transfer of `Apeiron.Trichotomy.AnyRing.ringNormForm_mul` through the link: the norm form of
the extracted product is the product of the extracted norm forms. -/
theorem word_norm_mul (a : cell.Cell I64) (s : cell.Sigma) (b : cell.Cell I64) :
    (do
      let c ← word_mul a s b
      let n ← word_norm c s
      ok (denote n)) =
    (do
      let na ← word_norm a s
      let nb ← word_norm b s
      ok (denote na * denote nb)) := by
  cases s <;>
    simp [word_mul, word_norm, cell.Cell.mul, cell.Cell.norm_form, cell.Sigma.value, sigmaI64,
      I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub,
      I64.Insts.Apeiron_wordScalarRing.mul, I64.Insts.Apeiron_wordScalarRing.from_i64] <;>
    ring

/-- Transfer of `Apeiron.Trichotomy.AnyRing.ringMul_conj` through the link: the extracted product
with the conjugate has the extracted norm form in the first slot and zero in the second. -/
theorem word_mul_conj (a : cell.Cell I64) (s : cell.Sigma) :
    (do let c ← word_mul a s (cellConj a); ok (denoteCell c)) =
      ok (Apeiron.Trichotomy.AnyRing.ringNormForm (denoteSigma s) (denoteCell a),
        (0 : ZMod (2 ^ 64))) := by
  cases s <;>
    simp [word_mul, cell.Cell.mul, cell.Sigma.value, sigmaI64, denoteCell, denoteSigma, cellConj,
      I64.Insts.Apeiron_wordScalarRing,
      I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub,
      I64.Insts.Apeiron_wordScalarRing.mul, I64.Insts.Apeiron_wordScalarRing.from_i64,
      Apeiron.Trichotomy.AnyRing.ringNormForm] <;>
    constructor <;> ring

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

/-- Transfer of `ringCarrierMomentumAfter_eq`: the extracted carrier ends holding the body
momentum. -/
theorem word_elastic_exchange_carrier (p q : cell.Cell I64) :
    (do let r ← word_elastic_exchange p q; ok (denoteCell r.carrier_after)) = ok (denoteCell q) := by
  simp [word_elastic_exchange, exchange.elastic_exchange, exchange.exchange_impulse,
    cell.Cell.new, I64.Insts.Apeiron_wordScalarRing,
    I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub,
    denoteCell]

/-- Transfer of `ringBodyMomentumAfter_eq`: the extracted body ends holding the carrier
momentum. -/
theorem word_elastic_exchange_body (p q : cell.Cell I64) :
    (do let r ← word_elastic_exchange p q; ok (denoteCell r.body_after)) = ok (denoteCell p) := by
  simp [word_elastic_exchange, exchange.elastic_exchange, exchange.exchange_impulse,
    cell.Cell.new, I64.Insts.Apeiron_wordScalarRing,
    I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub,
    denoteCell]

/-- Transfer of `ringTotalMomentum_conserved`: the extracted exchange conserves the total basis
momentum. -/
theorem word_elastic_exchange_totalMomentum (p q : cell.Cell I64) :
    (do
      let r ← word_elastic_exchange p q
      ok (denoteCell r.carrier_after + denoteCell r.body_after)) =
    ok (denoteCell p + denoteCell q) := by
  simp [word_elastic_exchange, exchange.elastic_exchange, exchange.exchange_impulse,
    cell.Cell.new, I64.Insts.Apeiron_wordScalarRing,
    I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub,
    denoteCell]
  constructor <;> ring

/-- Transfer of `ringNormFormLedger_conserved`: the extracted exchange conserves the canonical
norm-form ledger at every symbolic dial value `denoteSigma s`. -/
theorem word_elastic_exchange_normFormLedger (s : cell.Sigma) (p q : cell.Cell I64) :
    (do
      let r ← word_elastic_exchange p q
      ok (Apeiron.Physics.ringNormFormLedger (denoteSigma s) (denoteCell r.carrier_after)
        (denoteCell r.body_after))) =
    ok (Apeiron.Physics.ringNormFormLedger (denoteSigma s) (denoteCell p) (denoteCell q)) := by
  simp [word_elastic_exchange, exchange.elastic_exchange, exchange.exchange_impulse,
    cell.Cell.new, I64.Insts.Apeiron_wordScalarRing,
    I64.Insts.Apeiron_wordScalarRing.add, I64.Insts.Apeiron_wordScalarRing.sub,
    denoteCell, Apeiron.Physics.ringNormFormLedger,
    Apeiron.Trichotomy.AnyRing.ringNormForm]
  ring

/-- Transfer of `ringExchangeImpulse_antisymm`: swapping the extracted momenta negates the
denoted derived impulse. -/
theorem word_exchange_impulse_antisymm (p q : cell.Cell I64) :
    (do let r ← word_exchange_impulse p q; ok (denoteCell r)) =
      (do let r ← word_exchange_impulse q p; ok (- denoteCell r)) := by
  simp [word_exchange_impulse, exchange.exchange_impulse, cell.Cell.new,
    I64.Insts.Apeiron_wordScalarRing, I64.Insts.Apeiron_wordScalarRing.sub,
    denoteCell]

/-!
### The checked-claim kernel

`word_accept_derived_impulse` is the extracted `accept_derived_impulse`. The generated
`cell.Cell.Insts.CoreCmpPartialEqCell.eq` compares the two slots in turn and the generated `if`
selects the branch, so acceptance is decided by **source-word equality** of the two `I64` slots.
The spec below therefore reads word equality directly; result correspondence is carried over from
`word_elastic_exchange_denote` rather than inferred from `ZMod (2^64)` equality.
-/

/-- The concrete `I64` scalar instance, named for the extracted kernel. -/
abbrev i64ScalarRing : scalar.Ring I64 := I64.Insts.Apeiron_wordScalarRing

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

end apeiron_word
