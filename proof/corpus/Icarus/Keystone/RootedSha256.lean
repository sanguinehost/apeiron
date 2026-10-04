/-
  AR-74 — the keystone: one composed theorem for the SHA-256 round step as a rooted circuit over
  the trichotomy carrier.

  Corpus port of `work/proofs/AlgebraRootedSha256.lean` (AR-74,
  sha256 dcfc822ff05761526f47208409c78bd38fb23cfbb68ab9eb26fb667ea8d12424). Every declaration name
  and every statement is unchanged; only the import lines are mapped onto the corpus modules.

  The bound pieces are imported read-only, composed, and nothing here re-proves them. The pieces and
  their roles:

  * AR-60 `Icarus.AlgebraBoolCarrier32Round` — the width-32 carrier round `carrierRound`, its
    canonical round body `roundStepFun`, the consistency statement `carrierRound_coord`, the
    coordinate wires (`wiringRotR`, `wiringShr`, the packed wiring links, the Σ composites), the
    first-axis product read `round_products_face_free`, and the AR-53 carry step
    (`carrierT1Stage`, `SchedInv`, `carrierT1Stage_inv`, `carrierT1Stage_hi`).
  * AR-19 `Icarus.AlgebraFaceComposition`, `Icarus.AlgebraFaceCompositionRound`,
    `Icarus.AlgebraFaceCompositionSwitch` — `one_face_serves_the_round`, the dial's first-slot-only
    appearance, the elliptic no-hom-out verdict.
  * AR-36 `Icarus.AlgebraRotBridge32` — no equivariant multiplier repairs the halves pairing at
    width 32. [ported: AR-36, commit 6d53961.]
  * AR-23 `Icarus.AlgebraShrExact` — no ring polynomial reads either FIPS shift amount.
  * AR-26 `Icarus.AlgebraSigmaHalves` — a rotation split serves exactly its own amount.
  * AR-33 `Icarus.RingModelWeb` — no member of the two-slot read family reads a rotation. [ported: AR-33, commit 5b643d0.]
  * AR-14 `Icarus.Composition.AlgebraSha256d` — every `+`/`ringMul` expression preserves the bit-1
    congruence; ROTR, SHR and both big Σ do not.
  * AR-41 `Icarus.Cost.AlgebraCostSource` — the source-bound one-round field figure.
  * AR-58 `Icarus.Cost.AlgebraCostBridge` — the `CostAlgebra.Cost` image of the round plan. [ported: AR-58, commit dd2bd71; CostAlgebra vendored at Icarus/Cost/CostAlgebra.lean.]

  `sha256_round_rooted` is one conjunction of nine named clauses; the clause table is in
  `AlgebraRootedSha256.md`. Not in scope, named exactly: the message-schedule injection, the
  64-round fold, the compression feed-forward, the digest/store, and the executing `Cell<i64>`
  refinement. No security, collision, preimage, constant-time, performance or mining claim.

  No `sorry`, no `admit`, no axiom, no `native_decide`, no raised `maxRecDepth`, no
  `maxHeartbeats 0`.
-/
import Icarus.AlgebraBoolCarrier32Round
import Icarus.AlgebraFaceComposition
import Icarus.AlgebraFaceCompositionRound
import Icarus.AlgebraFaceCompositionSwitch
import Icarus.AlgebraShrExact
import Icarus.AlgebraSigmaHalves
import Icarus.AlgebraRotBridge32
import Icarus.RingModelWeb
import Icarus.Composition.AlgebraSha256d
import Icarus.Cost.AlgebraCostBridge
import Icarus.Cost.AlgebraCostCompletePart1
import Icarus.Cost.AlgebraCostCompletePart2
import Icarus.Cost.AlgebraCostSource
import Mathlib.Tactic
import Lean

set_option autoImplicit false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
-- `RefutationWall`'s rotation clause carries `hn : 0 < n` because AR-36's `no_equivariant_multiplier32`
-- binds it; the statement itself does not reference it. Deliberate, so the binder stays.
set_option linter.unusedVariables false

namespace AlgebraRootedSha256

open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
open AlgebraBoolCarrier32Round

/-! ## 0 · Fail-closed axiom cone

`#audit` is `AlgebraChi.lean`'s command as re-declared by AR-60: it collects each constant's
transitive axioms and throws on anything outside `{propext, Classical.choice, Quot.sound}`, so a
`sorryAx` in any clause below is a build failure rather than a line in a log. -/

open Lean Elab Command in
private def auditClosedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
elab "#audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !auditClosedAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the closed cone " ++
        m!"(propext, Classical.choice, Quot.sound): {bad.toList}"

/-! ## 1 · The nine clauses, as named propositions

Each clause below is the exact statement of a bound component, so its discharge is that component
applied (the clause table in `AlgebraRootedSha256.md` is the mapping). The clauses are stated as
`abbrev`s only so that `sha256_round_rooted` reads as the composition instead of as a page of
formulas; every one of them unfolds definitionally. -/

/-- **(V) Value.** Coordinate `j` of word `i` of the carrier round is the canonical `Crypto.Sha2`
round body `roundStepFun` read at the packed inputs, at every state, constant, schedule word, word
and bit. No premise, no partial branch. Discharge: AR-60 `carrierRound_coord`. -/
abbrev RoundValue : Prop :=
  ∀ (s : CState) (kt wt : CWord) (i : Fin 8) (j : Fin 32),
    carrierRound s kt wt i j
      = b2f ((roundStepFun (fun l : Fin 8 => pack 32 (s l)) (pack 32 kt) (pack 32 wt) i).getLsbD
          j.val)

/-- **(P) First-axis products, and the dial only in the first-slot product.** Every product the
round uses (the two Ch products and the three Maj products, over first-axis cells) reads as the
carrier product at every face `σ`; and a change of face changes a product only by a multiple of
the two second slots. Discharge: AR-60 `round_products_face_free`; AR-19
`ringMul_dial_only_in_first_slot`. -/
abbrev ProductPrimitives : Prop :=
  (∀ (σ x y z : CWord),
      (ringMul σ (bCell x) (bCell y)).1 = x * y
      ∧ (ringMul σ (bCell x) (bCell z)).1 = x * z
      ∧ (ringMul σ (bCell y) (bCell z)).1 = y * z)
  ∧ (∀ (σ τ : CWord) (z w : RingCell CWord),
      ringMul τ z w - ringMul σ z w = ((τ - σ) * z.2 * w.2, 0))

/-- **(W) Coordinate wiring.** ROTR and SHR are coordinate maps whose packing is the canonical
word operation, at every amount (`n < 32` for the rotation), and the round's two Σ composites
pack to the canonical `Sigma0`/`Sigma1` at their FIPS amounts (2/13/22 and 6/11/25). Discharge:
AR-60 `unpack_rotr_wiring_32`, `unpack_shr_wiring_32`, `pack_carrierSigma0`,
`pack_carrierSigma1`. -/
abbrev WirePrimitives : Prop :=
  (∀ (n : ℕ) (hn : n < 32) (x : BitVec 32),
      wiringRotR n (unpack x) = unpack (x.rotateRight n))
  ∧ (∀ (n : ℕ) (x : BitVec 32), wiringShr (unpack x) n = unpack (x >>> n))
  ∧ (∀ x : CWord, pack 32 (carrierSigma0 x) = Sigma0 (pack 32 x))
  ∧ (∀ x : CWord, pack 32 (carrierSigma1 x) = Sigma1 (pack 32 x))

/-- **(C) Carry addition.** Every addition of the round runs through the AR-53 two-input carry
step; the T1 five-term chain (four `addStep`s) preserves the schedule invariant, so its integer
overflow component is the exact whole-word quotient of the five-term total, carried in `ℕ` and
identified with no `ZMod (2^32)` slot. Discharge: AR-60 `carrierT1Stage_inv`,
`carrierT1Stage_hi`. -/
abbrev CarryPrimitives : Prop :=
  (∀ hh se ch kt wt : CWord,
      SchedInv 32 5
        ((pack 32 hh).toNat + (pack 32 se).toNat + (pack 32 ch).toNat + (pack 32 kt).toNat
          + (pack 32 wt).toNat)
        (carrierT1Stage hh se ch kt wt))
  ∧ (∀ hh se ch kt wt : CWord,
      (carrierT1Stage hh se ch kt wt).hi
        = ((pack 32 hh).toNat + (pack 32 se).toNat + (pack 32 ch).toNat + (pack 32 kt).toNat
            + (pack 32 wt).toNat) / 2 ^ 32)

/-- **(O) Readout and write-back.** The carrier word is the packed word: both round trips hold at
width 32, so the carrier read/write is an equivalence and the (V) coordinate read is the bit read
`b2f ∘ getLsbD`. Discharge: AR-60 `pack_unpack_32`, `unpack_pack_32`. -/
abbrev ReadoutPrimitives : Prop :=
  (∀ x : BitVec 32, pack 32 (unpack x) = x)
  ∧ (∀ v : CWord, unpack (pack 32 v) = v)

/-- **(F) Face verdict.** The AR-19 verdict at width 32: no one-`ringMul` read reaches the
rotation's wrapped bits; the additive readout exists exactly at `σ = 1` (the read is the modular
sum); the quadratic term is `σ`-free; and the elliptic face has no operation-preserving map out to
either other face. Discharge: AR-19 `one_face_serves_the_round`, `elliptic_face_no_hom_out`. -/
abbrev FaceVerdict : Prop :=
  (¬ ∃ (σ r : ZMod (2 ^ 32)) (a b : ZMod (2 ^ 32)),
      ∀ (v : ℕ) (_hv : v < 2 ^ 32),
        AlgebraFaceComposition.FaceRound.rotWrap 32 (32 - 1) v
          = ringUpper r (ringMul σ (((v : ℕ) : ZMod (2 ^ 32)), 0) (a, b)))
  ∧ (∃ r : ZMod (2 ^ 32),
      r ^ 2 = 1 ∧ ∀ z : RingCell (ZMod (2 ^ 32)), ringUpper r z = z.1 + z.2)
  ∧ (∀ x y z : ZMod (2 ^ 32),
      (x, 0) + (z, 0) + ringMul 1 (y, 0) (z, 0) = (x + z + y * z, 0))
  ∧ (¬ Nonempty (AlgebraFaceComposition.FaceSwitch.FH 32 (-1) 0))
  ∧ (¬ Nonempty (AlgebraFaceComposition.FaceSwitch.FH 32 (-1) 1))

/-- **(R) The refutations, respected.** (i)+(ii) every `+`/`ringMul` expression over a `RingCell`
of `ZMod (2^64)` preserves the `2^j`-congruence, while ROTR, SHR and both big Σ do not
(AR-14); (iii) no multiplier repairs the halves pairing at any positive amount below 32, in the
elliptic reading (AR-36); (iv) no ring polynomial reads either FIPS SHR amount at 32 (AR-23);
(v) a rotation split serves exactly its own amount (AR-26); (vi) no member of the two-slot read
family reads a rotation at 32 (AR-33). Together these are why clause (W) is a wire and not a
product. -/
abbrev RefutationWall : Prop :=
  (∀ (σ : ZMod (2 ^ 64)) (t : AlgebraSha256d.CTerm (ZMod (2 ^ 64)) σ) (j : ℕ),
      AlgebraSha256d.CongPres j (fun x => AlgebraSha256d.CTerm.eval t x))
  ∧ (¬ (∀ (u v : Word),
      AlgebraSha256d.Cmod1 u v → AlgebraSha256d.Cmod1 (rotr u 2) (rotr v 2)))
  ∧ (¬ (∀ (u v : Word),
      AlgebraSha256d.Cmod1 u v → AlgebraSha256d.Cmod1 (shrx u 1) (shrx v 1)))
  ∧ (¬ (∀ (u v : Word),
      AlgebraSha256d.Cmod1 u v → AlgebraSha256d.Cmod1 (Sigma0 u) (Sigma0 v)))
  ∧ (¬ (∀ (u v : Word),
      AlgebraSha256d.Cmod1 u v → AlgebraSha256d.Cmod1 (Sigma1 u) (Sigma1 v)))
  ∧ (∀ (n : ℕ) (hn : 0 < n) (hnk : n < 32),
      ¬ ∃ u : ZMod (2 ^ 32) × ZMod (2 ^ 32), ∀ a : ℕ, a < 2 ^ 32 →
        ringMul (-1) (AlgebraRotBridge32.halves (k := 32) n a) u
          = AlgebraRotBridge32.halves (k := 32) n (AlgebraRotBridge32.rotrVal (k := 32) n a))
  ∧ (¬ AlgebraShrExact.RingPolyRead 32 (fun x => AlgebraShrExact.den (AlgebraShrExact.tshrx x 3)))
  ∧ (¬ AlgebraShrExact.RingPolyRead 32 (fun x => AlgebraShrExact.den (AlgebraShrExact.tshrx x 10)))
  ∧ (∀ (m n : ℕ) (hm : 0 < m) (hmk : m < 32) (hn : 1 ≤ n) (hnk : n < 32),
      AlgebraSigmaHalves.Span2 (k := 32) n (AlgebraSigmaHalves.rotrVal (k := 32) m) ↔ m = n)
  ∧ (∀ (σ r a b : ZMod (2 ^ 32)),
      ¬ ∀ k : ℕ, ringUpper r (ringMul σ (((k : ℕ) : ZMod (2 ^ 32)), 0) (a, b))
          = RingModelWeb.bvNat 32 ((BitVec.ofNat 32 k).rotateLeft 2))

/-- **(K) Cost.** Two records, not summed: (i) the `CostAlgebra.Cost` image of the round plan
through AR-58's `shaToCost` — 99 operations, no bytes, one image for every face; (ii) AR-41's
source-bound one-round field figure in the 23-field measure, with `mul = 0` in the word layer (the
word layer contains no ring multiplication). Discharge: AR-58 `shaToCost_roundCost`; AR-41
`roundCost_fields`. The AR-41 refusals (no wall-clock, cycle, joule, watt, hash-rate, throughput,
USD or profitability figure; the C5 rotation reading is a model premise and the word layer pays
none of it) stay intact and are restated in the receipt. -/
abbrev CostClause : Prop :=
  (∀ σ : ZMod (2 ^ 32),
      algebra_cost_bridge.shaToCost σ
          (algebra_cost_complete.ShaCost.ofRound (algebra_cost_complete.roundCost σ))
        = ({ ops := 99, pack := 0, alloc := 0, xfer := 0, dispatch := 0, peak := 0, path := 0 }
            : CostAlgebra.Cost))
  ∧ (∀ σ : ZMod (2 ^ 32),
      algebra_cost_source.roundCost σ
        = algebra_cost_source.ShaCost.op σ 0 7 0 0 0 0 7 5 1 12 6)

/-- **(X) Reachable witnesses.** Two joint instances on the delivered bodies: the all-zero
carrier state, zero constant and zero schedule word give the all-zero round at every coordinate;
and the five-term chain at five all-ones words carries out `4` whole 32-bit units, so the integer
component is exercised nonzero. Discharge: AR-60 `carrierRound_zero`, `carrierT1Stage_full_hi`. -/
abbrev Witnesses : Prop :=
  (∀ (i : Fin 8) (j : Fin 32), carrierRound (fun _ : Fin 8 => 0) 0 0 i j = 0)
  ∧ (carrierT1Stage (unpack (full 32)) (unpack (full 32)) (unpack (full 32)) (unpack (full 32))
      (unpack (full 32))).hi = 4

/-! ## 2 · The composed theorem -/

/-- **AR-74 keystone.** The SHA-256 round step is a total circuit over the trichotomy carrier
`RingCell (Fin 32 → ZMod 2)`: its products are first-axis `ringMul` reads of first-axis cells at
every face, its additions are the AR-53 two-input carry chain with the integer overflow count kept
outside the words, its ROTR/SHR/Σ leaves are coordinate wires equal to the canonical word
operations, its readout is the exact packing bit read, the AR-19 face verdict holds at width 32
(including the elliptic no-hom-out), the AR-14/23/26/33/36 refutations hold (so no ring expression
reaches the wires), and its cost is the AR-58 `CostAlgebra.Cost` image together with AR-41's
source-bound field figure. Every clause is discharged by a bound component; the clause table is in
`AlgebraRootedSha256.md`.

Short of the message-schedule injection and the 64-round fold this is the strongest composition
the bound pieces support: `carrierRound` takes the round constant and schedule word as arbitrary
carrier inputs, and no theorem here folds `carrierRound` against `compress_eq_rounds`. -/
theorem sha256_round_rooted :
    RoundValue ∧ ProductPrimitives ∧ CarryPrimitives ∧ WirePrimitives ∧ ReadoutPrimitives
      ∧ FaceVerdict ∧ RefutationWall ∧ CostClause ∧ Witnesses := by
  refine ⟨carrierRound_coord, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact ⟨round_products_face_free,
      fun σ τ z w => AlgebraFaceComposition.FaceTable.ringMul_dial_only_in_first_slot σ τ z w⟩
  · exact ⟨carrierT1Stage_inv, carrierT1Stage_hi⟩
  · exact ⟨fun n hn x => unpack_rotr_wiring_32 n hn x,
      fun n x => unpack_shr_wiring_32 n x, pack_carrierSigma0, pack_carrierSigma1⟩
  · exact ⟨pack_unpack_32, unpack_pack_32⟩
  · refine
      ⟨(AlgebraFaceComposition.FaceRound.one_face_serves_the_round (k := 32) (by omega)).1,
        (AlgebraFaceComposition.FaceRound.one_face_serves_the_round (k := 32) (by omega)).2.1,
        (AlgebraFaceComposition.FaceRound.one_face_serves_the_round (k := 32) (by omega)).2.2,
        (AlgebraFaceComposition.FaceSwitch.elliptic_face_no_hom_out (k := 32) (by omega)).1,
        (AlgebraFaceComposition.FaceSwitch.elliptic_face_no_hom_out (k := 32) (by omega)).2⟩
  · refine
      ⟨fun σ t j => AlgebraSha256d.CTerm.cong_preserves (σ := σ) t j,
        AlgebraSha256d.rot_not_cong_pres,
        AlgebraSha256d.shr_not_cong_pres,
        AlgebraSha256d.Sigma0_not_cong_pres,
        AlgebraSha256d.Sigma1_not_cong_pres,
        fun n hn hnk => AlgebraRotBridge32.no_equivariant_multiplier32 n hn hnk,
        AlgebraShrExact.shr32_3_not_ringpoly,
        AlgebraShrExact.shr32_10_not_ringpoly,
        fun m n hm hmk hn hnk =>
          AlgebraSigmaHalves.span_rotr_iff (k := 32) (m := m) (n := n) (by omega) hm hmk hn hnk,
        fun σ r a b =>
          RingModelWeb.no_read_computes_rotation 32 2 (by norm_num) (by norm_num) σ r a b⟩
  · exact ⟨fun σ => algebra_cost_bridge.shaToCost_roundCost σ,
      fun σ => algebra_cost_source.roundCost_fields σ⟩
  · exact ⟨carrierRound_zero, carrierT1Stage_full_hi⟩

/-! ## 3 · Cone -/

#audit AlgebraRootedSha256.CarryPrimitives
#audit AlgebraRootedSha256.CostClause
#audit AlgebraRootedSha256.FaceVerdict
#audit AlgebraRootedSha256.ProductPrimitives
#audit AlgebraRootedSha256.ReadoutPrimitives
#audit AlgebraRootedSha256.RefutationWall
#audit AlgebraRootedSha256.RoundValue
#audit AlgebraRootedSha256.WirePrimitives
#audit AlgebraRootedSha256.Witnesses
#audit AlgebraRootedSha256.sha256_round_rooted

#print axioms AlgebraRootedSha256.CarryPrimitives
#print axioms AlgebraRootedSha256.CostClause
#print axioms AlgebraRootedSha256.FaceVerdict
#print axioms AlgebraRootedSha256.ProductPrimitives
#print axioms AlgebraRootedSha256.ReadoutPrimitives
#print axioms AlgebraRootedSha256.RefutationWall
#print axioms AlgebraRootedSha256.RoundValue
#print axioms AlgebraRootedSha256.WirePrimitives
#print axioms AlgebraRootedSha256.Witnesses
#print axioms AlgebraRootedSha256.sha256_round_rooted

end AlgebraRootedSha256