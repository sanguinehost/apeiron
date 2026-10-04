/-
  `CarrierSigma32.lean` — AR-68's residual R1: the `zxor3` composition of the six SHA-2 amount
  wires on the **Boolean carrier** `BoolRing 32`, and the packing equalities that tie it to
  `BitVec 32` and to the corpus's own `Apeiron.Crypto.Sha2.{sigma0,sigma1}` declarations.

  AR-68 (`work/proofs/AlgebraSigma32.md` §6, R1) read the composite on `RingCell (ZMod (2^32))`
  with the forced bit-halves pairing, where the `⊕` layer is a residual costing two exclusive-ors
  and zero ring-side price. AR-53 (`work/proofs/AlgebraBoolCarrier32.md` §5 row 1) wired ROTR and
  SHR on the Boolean carrier `B_32 = Fin 32 → ZMod 2` and left the three-fold composition open.
  This file is that composition: `zxor3 a b c = a + b + c` in `B_32`, applied to the ROTR-7,
  ROTR-18, SHR-3 wires (`σ₀`) and to the ROTR-17, ROTR-19, SHR-10 wires (`σ₁`), proved equal to
  `unpack` of the corresponding `BitVec 32` word and to `unpack` of the corpus's own declaration.

  The structural difference from AR-68 is the point: on this carrier the `⊕` layer **is** the
  carrier's addition, so the layer needs no operator beyond `+` and introduces no residual term.
  The statements here are equality statements on the carrier and on `BitVec 32`; no cost figure,
  no hardware measurement, no security, performance or mining claim is made.

  Trust boundary. Statements about `ZMod 2`, `Fin 32 → ZMod 2`, `BoolCell 32`, `BitVec 32` and the
  corpus's own `Apeiron.Crypto.Sha2` declarations. Nothing here reaches rustc/LLVM, the kernel ABI,
  GPU/SPIR-V, the `unsafe` container, or any running hash.

  No `sorry`, no `admit`, no axiom, no `native_decide`, no raised `maxRecDepth`, no
  `maxHeartbeats 0`, no `decide` over a word space.
-/
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sha2
import Icarus.AlgebraBoolCarrier32
import Mathlib.Data.BitVec
import Mathlib.Tactic

set_option autoImplicit false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace IcarusSha256CarrierSigma32

open Apeiron.Trichotomy.AnyRing
open AlgebraBoolCarrier32

/-! ## Fail-closed axiom cone

`#audit` is `Icarus/AlgebraBoolCarrier32.lean`'s command verbatim: it collects each constant's
transitive axioms and throws on anything outside `{propext, Classical.choice, Quot.sound}`, so a
`sorry` below is a build failure rather than a line in a log. -/

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

/-! ## §1 · The carrier and the six amount wires

The Boolean carrier at width 32 is AR-53's `B_32 = Fin 32 → ZMod 2`; the two-slot algebra is
`BoolCell 32 = B_32 × B_32`. Neither is re-declared here — `Icarus.AlgebraBoolCarrier32` is
imported and read, so this file adds only the composition. -/

/-- The Boolean carrier at width 32, AR-53's `BoolRing32`. -/
abbrev BoolR32 : Type := BoolRing 32

/-- The two-slot algebra at width 32, AR-53's `BoolCell32`. -/
abbrev BoolC32 : Type := BoolCell 32

/-- **`ROTR n` as a coordinate wiring at width 32**: the coordinate permutation of AR-53's
`wiringRotR`, read at the fixed amount `n`. One coordinate test per output coordinate. -/
def rotR32 (n : ℕ) (x : BitVec 32) : BoolR32 := wiringRotR n (unpack x)

/-- **`SHR n` as a coordinate wiring at width 32**: AR-53's `wiringShr`, read at the fixed amount
`n`; coordinates above the shifted-out boundary receive `0`. -/
def shr32 (n : ℕ) (x : BitVec 32) : BoolR32 := wiringShr (unpack x) n

/-- **The `zxor3` composition on the Boolean carrier**: the three-fold exclusive-or of FIPS 180-4
§4.1.2 as a sum in the carrier's own ring structure. This is the residual layer of AR-68, and the
definition is what discharges it: on `B_32` the layer *is* `+`, so it introduces no operator
beyond the addition the two-slot algebra already provides, and no residual term of its own. -/
def zxor3 (a b c : BoolR32) : BoolR32 := a + b + c

/-- **FIPS 180-4 §4.1.2 `σ₀ = ROTR^7 ⊕ ROTR^18 ⊕ SHR^3` on the Boolean carrier**, as `zxor3` of the
three amount wires at width 32. -/
def sigma0Bool32 (x : BitVec 32) : BoolR32 := zxor3 (rotR32 7 x) (rotR32 18 x) (shr32 3 x)

/-- **FIPS 180-4 §4.1.2 `σ₁ = ROTR^17 ⊕ ROTR^19 ⊕ SHR^10` on the Boolean carrier**, as `zxor3` of the
three amount wires at width 32. -/
def sigma1Bool32 (x : BitVec 32) : BoolR32 := zxor3 (rotR32 17 x) (rotR32 19 x) (shr32 10 x)

/-- **`σ₀` as a two-slot cell operation**: the sum of the three first-axis cells of its terms. The
`zxor3` layer is the addition, so the cell is read on the first axis throughout. -/
def sigma0Cell32 (x : BitVec 32) : BoolC32 :=
  bCell (rotR32 7 x) + bCell (rotR32 18 x) + bCell (shr32 3 x)

/-- **`σ₁` as a two-slot cell operation**: the sum of the three first-axis cells of its terms. -/
def sigma1Cell32 (x : BitVec 32) : BoolC32 :=
  bCell (rotR32 17 x) + bCell (rotR32 19 x) + bCell (shr32 10 x)

/-! ## §2 · The wires are AR-53's checked wiring, at the six amounts -/

/-- **`ROTR n` at width 32 is the packing of the extracted rotation**, for every `n < 32`. This is
AR-53's `unpack_rotr_wiring_32` read through `rotR32`. -/
theorem rotR32_eq_unpack_rotateRight (n : ℕ) (hn : n < 32) (x : BitVec 32) :
    rotR32 n x = unpack (x.rotateRight n) := unpack_rotr_wiring_32 n hn x

/-- **`SHR n` at width 32 is the packing of the extracted logical shift**, for every amount. -/
theorem shr32_eq_unpack_ushiftRight (n : ℕ) (x : BitVec 32) :
    shr32 n x = unpack (x >>> n) := unpack_shr_wiring_32 n x

/-- **`σ₀`'s ROTR-7 wire** at width 32. -/
theorem rotR32_7 (x : BitVec 32) : rotR32 7 x = unpack (x.rotateRight 7) :=
  rotR32_eq_unpack_rotateRight 7 (by norm_num) x

/-- **`σ₀`'s ROTR-18 wire** at width 32. -/
theorem rotR32_18 (x : BitVec 32) : rotR32 18 x = unpack (x.rotateRight 18) :=
  rotR32_eq_unpack_rotateRight 18 (by norm_num) x

/-- **`σ₀`'s SHR-3 wire** at width 32. -/
theorem shr32_3 (x : BitVec 32) : shr32 3 x = unpack (x >>> 3) := shr32_eq_unpack_ushiftRight 3 x

/-- **`σ₁`'s ROTR-17 wire** at width 32. -/
theorem rotR32_17 (x : BitVec 32) : rotR32 17 x = unpack (x.rotateRight 17) :=
  rotR32_eq_unpack_rotateRight 17 (by norm_num) x

/-- **`σ₁`'s ROTR-19 wire** at width 32. -/
theorem rotR32_19 (x : BitVec 32) : rotR32 19 x = unpack (x.rotateRight 19) :=
  rotR32_eq_unpack_rotateRight 19 (by norm_num) x

/-- **`σ₁`'s SHR-10 wire** at width 32. -/
theorem shr32_10 (x : BitVec 32) : shr32 10 x = unpack (x >>> 10) :=
  shr32_eq_unpack_ushiftRight 10 x

/-! ## §3 · The two-slot reading: the `zxor3` layer stays on the first axis -/

/-- **The first axis of `σ₀`'s cell is `σ₀` on the carrier.** -/
theorem sigma0Cell32_fst (x : BitVec 32) :
    (sigma0Cell32 x).1 = sigma0Bool32 x := by
  unfold sigma0Cell32 sigma0Bool32
  rw [bCell_add, bCell_add, bCell_fst]
  rfl

/-- **The second axis of `σ₀`'s cell is zero**: the `⊕` layer is addition, so it generates no
second-slot residual. This is the statement that closes AR-68's residual R1 on this carrier. -/
theorem sigma0Cell32_snd (x : BitVec 32) : (sigma0Cell32 x).2 = 0 := by
  unfold sigma0Cell32
  simp only [bCell_add, bCell_snd]

/-- **The first axis of `σ₁`'s cell is `σ₁` on the carrier.** -/
theorem sigma1Cell32_fst (x : BitVec 32) :
    (sigma1Cell32 x).1 = sigma1Bool32 x := by
  unfold sigma1Cell32 sigma1Bool32
  rw [bCell_add, bCell_add, bCell_fst]
  rfl

/-- **The second axis of `σ₁`'s cell is zero.** -/
theorem sigma1Cell32_snd (x : BitVec 32) : (sigma1Cell32 x).2 = 0 := by
  unfold sigma1Cell32
  simp only [bCell_add, bCell_snd]

/-! ## §4 · The packing equalities: `zxor3` of three packed words is the packing of their XOR

The layer statement AR-53 §5 row 1 asks for. `zxor3` is the carrier's addition, and AR-53's
`unpack_add_xor` reads the carrier's addition as the packed exclusive-or, so the layer needs no
operator of its own. -/

/-- **`zxor3` of three packed words is the packing of their three-fold exclusive-or**, at width 32,
from AR-53's `unpack_add_xor` applied twice. No hypothesis on the words. -/
theorem zxor3_unpack (p q r : BitVec 32) :
    zxor3 (unpack p) (unpack q) (unpack r) = unpack ((p ^^^ q) ^^^ r) := by
  unfold zxor3
  rw [unpack_add_xor, unpack_add_xor]

/-- **`σ₀` on the Boolean carrier is the packing of the `BitVec 32` word
`ROTR 7 ⊕ ROTR 18 ⊕ SHR 3`** — the width-32 `zxor3` composition of the three amount wires. -/
theorem sigma0Bool32_eq_unpack (x : BitVec 32) :
    sigma0Bool32 x =
      unpack ((x.rotateRight 7 ^^^ x.rotateRight 18) ^^^ (x >>> 3)) := by
  unfold sigma0Bool32
  rw [rotR32_7, rotR32_18, shr32_3]
  exact zxor3_unpack _ _ _

/-- **`σ₁` on the Boolean carrier is the packing of the `BitVec 32` word
`ROTR 17 ⊕ ROTR 19 ⊕ SHR 10`**. -/
theorem sigma1Bool32_eq_unpack (x : BitVec 32) :
    sigma1Bool32 x =
      unpack ((x.rotateRight 17 ^^^ x.rotateRight 19) ^^^ (x >>> 10)) := by
  unfold sigma1Bool32
  rw [rotR32_17, rotR32_19, shr32_10]
  exact zxor3_unpack _ _ _

/-- **`σ₀`'s cell is the first-axis cell of the packed word.** The `zxor3` layer is the addition,
so the whole composite occupies slot one. -/
theorem sigma0Cell32_eq (x : BitVec 32) :
    sigma0Cell32 x = bCell (sigma0Bool32 x) := by
  unfold sigma0Cell32
  rw [bCell_add, bCell_add]
  unfold sigma0Bool32 zxor3 bCell
  rfl

/-- **`σ₁`'s cell is the first-axis cell of the packed word.** -/
theorem sigma1Cell32_eq (x : BitVec 32) :
    sigma1Cell32 x = bCell (sigma1Bool32 x) := by
  unfold sigma1Cell32
  rw [bCell_add, bCell_add]
  unfold sigma1Bool32 zxor3 bCell
  rfl

/-- **`σ₀`'s cell, read, is the packing of the `BitVec 32` word
`ROTR 7 ⊕ ROTR 18 ⊕ SHR 3`.** -/
theorem sigma0Cell32_read_eq_unpack (x : BitVec 32) :
    (sigma0Cell32 x).1 =
      unpack ((x.rotateRight 7 ^^^ x.rotateRight 18) ^^^ (x >>> 3)) := by
  rw [sigma0Cell32_fst, sigma0Bool32_eq_unpack]

/-- **`σ₁`'s cell, read, is the packing of the `BitVec 32` word
`ROTR 17 ⊕ ROTR 19 ⊕ SHR 10`.** -/
theorem sigma1Cell32_read_eq_unpack (x : BitVec 32) :
    (sigma1Cell32 x).1 =
      unpack ((x.rotateRight 17 ^^^ x.rotateRight 19) ^^^ (x >>> 10)) := by
  rw [sigma1Cell32_fst, sigma1Bool32_eq_unpack]

/-! ## §5 · The corpus's own declarations

The statements above are about re-declared wiring. These link the composition to
`Apeiron.Crypto.Sha2.sigma0` and `.sigma1` as those are declared in `Crypto/Sha2.lean`, so no
re-declaration gap remains for the two small σ composites. -/

/-- **The corpus's `Sha2.rotr x m` is `x.rotateRight m`** for `0 < m < 32` (AR-12's
`rotr_eq_rotateRight`, at the corpus's own width). -/
theorem sha2_rotr_eq_rotateRight (x : BitVec 32) (m : ℕ) (hm : 0 < m) (hmk : m < 32) :
    Apeiron.Crypto.Sha2.rotr x m = x.rotateRight m := by
  have h1 : m % 32 = m := Nat.mod_eq_of_lt hmk
  have h2 : (32 - m) % 32 = 32 - m := Nat.mod_eq_of_lt (by omega)
  simp only [Apeiron.Crypto.Sha2.rotr, h1, h2, BitVec.rotateRight_def]

/-- **The corpus's `Sha2.shrx x n` is `x >>> n`** at width 32, for every amount `n < 32`. The
amount bound is the same one AR-12 records: `shrx` reduces modulo 32, so the identity is a
statement about `n < 32` and not a claim about all amounts. -/
theorem sha2_shrx_eq_ushiftRight (x : BitVec 32) (n : ℕ) (hnk : n < 32) :
    Apeiron.Crypto.Sha2.shrx x n = x >>> n := by
  have h1 : n % 32 = n := Nat.mod_eq_of_lt hnk
  simp only [Apeiron.Crypto.Sha2.shrx, h1]

/-- **`σ₀` on the Boolean carrier at width 32 is the packing of the corpus's own `Sha2.sigma0`.** -/
theorem sigma0Bool32_eq_sha2 (x : BitVec 32) :
    sigma0Bool32 x = unpack (Apeiron.Crypto.Sha2.sigma0 x) := by
  rw [sigma0Bool32_eq_unpack, Apeiron.Crypto.Sha2.sigma0]
  have h7 := sha2_rotr_eq_rotateRight x 7 (by norm_num) (by norm_num)
  have h18 := sha2_rotr_eq_rotateRight x 18 (by norm_num) (by norm_num)
  have h3 := sha2_shrx_eq_ushiftRight x 3 (by norm_num)
  rw [h7, h18, h3, BitVec.xor_assoc]

/-- **`σ₁` on the Boolean carrier at width 32 is the packing of the corpus's own `Sha2.sigma1`.** -/
theorem sigma1Bool32_eq_sha2 (x : BitVec 32) :
    sigma1Bool32 x = unpack (Apeiron.Crypto.Sha2.sigma1 x) := by
  rw [sigma1Bool32_eq_unpack, Apeiron.Crypto.Sha2.sigma1]
  have h17 := sha2_rotr_eq_rotateRight x 17 (by norm_num) (by norm_num)
  have h19 := sha2_rotr_eq_rotateRight x 19 (by norm_num) (by norm_num)
  have h10 := sha2_shrx_eq_ushiftRight x 10 (by norm_num)
  rw [h17, h19, h10, BitVec.xor_assoc]

/-- **`σ₀`'s cell at width 32 is the first-axis cell of the packing of `Sha2.sigma0`.** -/
theorem sigma0Cell32_eq_sha2 (x : BitVec 32) :
    sigma0Cell32 x = bCell (unpack (Apeiron.Crypto.Sha2.sigma0 x)) := by
  rw [sigma0Cell32_eq, sigma0Bool32_eq_sha2]

/-- **`σ₁`'s cell at width 32 is the first-axis cell of the packing of `Sha2.sigma1`.** -/
theorem sigma1Cell32_eq_sha2 (x : BitVec 32) :
    sigma1Cell32 x = bCell (unpack (Apeiron.Crypto.Sha2.sigma1 x)) := by
  rw [sigma1Cell32_eq, sigma1Bool32_eq_sha2]

/-- **The `zxor3` layer is the carrier's addition, so it costs nothing on the ring side.** Both
slots of the composite's cell are accounted for: slot one is the packed word, slot two is zero.
This is the statement AR-68 could not make on `RingCell (ZMod (2^32))`, where the layer had a
two-`⊕` residual. -/
theorem zxor3_is_addition :
    ∀ a b c : BoolR32, zxor3 a b c = a + b + c := fun _ _ _ => rfl

/-! ## §6 · Coordinate reading of the layer

`zxor3` is pointwise addition, so at each of the 32 coordinates the layer reads the three
corresponding coordinates and nothing else. -/

/-- **Coordinate `i` of the `zxor3` layer is the sum of the three coordinates.** -/
theorem zxor3_apply (a b c : BoolR32) (i : Fin 32) : zxor3 a b c i = a i + b i + c i := rfl

/-- **The `zxor3` layer is zero iff it is zero at every coordinate.** Each coordinate is a genuine
equation in `ZMod 2`, so no coordinate is dropped by the layer. -/
theorem zxor3_eq_zero_iff (a b c : BoolR32) :
    zxor3 a b c = 0 ↔ ∀ i : Fin 32, a i + b i + c i = 0 := by
  constructor
  · intro h i
    have := congrFun h i
    simpa [zxor3] using this
  · intro h
    funext i
    exact h i

#audit IcarusSha256CarrierSigma32.BoolR32
#audit IcarusSha256CarrierSigma32.BoolC32
#audit IcarusSha256CarrierSigma32.rotR32
#audit IcarusSha256CarrierSigma32.shr32
#audit IcarusSha256CarrierSigma32.zxor3
#audit IcarusSha256CarrierSigma32.sigma0Bool32
#audit IcarusSha256CarrierSigma32.sigma1Bool32
#audit IcarusSha256CarrierSigma32.sigma0Cell32
#audit IcarusSha256CarrierSigma32.sigma1Cell32
#audit IcarusSha256CarrierSigma32.rotR32_eq_unpack_rotateRight
#audit IcarusSha256CarrierSigma32.shr32_eq_unpack_ushiftRight
#audit IcarusSha256CarrierSigma32.rotR32_7
#audit IcarusSha256CarrierSigma32.rotR32_18
#audit IcarusSha256CarrierSigma32.shr32_3
#audit IcarusSha256CarrierSigma32.rotR32_17
#audit IcarusSha256CarrierSigma32.rotR32_19
#audit IcarusSha256CarrierSigma32.shr32_10
#audit IcarusSha256CarrierSigma32.sigma0Cell32_fst
#audit IcarusSha256CarrierSigma32.sigma0Cell32_snd
#audit IcarusSha256CarrierSigma32.sigma1Cell32_fst
#audit IcarusSha256CarrierSigma32.sigma1Cell32_snd
#audit IcarusSha256CarrierSigma32.zxor3_unpack
#audit IcarusSha256CarrierSigma32.sigma0Bool32_eq_unpack
#audit IcarusSha256CarrierSigma32.sigma1Bool32_eq_unpack
#audit IcarusSha256CarrierSigma32.sigma0Cell32_eq
#audit IcarusSha256CarrierSigma32.sigma1Cell32_eq
#audit IcarusSha256CarrierSigma32.sigma0Cell32_read_eq_unpack
#audit IcarusSha256CarrierSigma32.sigma1Cell32_read_eq_unpack
#audit IcarusSha256CarrierSigma32.sha2_rotr_eq_rotateRight
#audit IcarusSha256CarrierSigma32.sha2_shrx_eq_ushiftRight
#audit IcarusSha256CarrierSigma32.sigma0Bool32_eq_sha2
#audit IcarusSha256CarrierSigma32.sigma1Bool32_eq_sha2
#audit IcarusSha256CarrierSigma32.sigma0Cell32_eq_sha2
#audit IcarusSha256CarrierSigma32.sigma1Cell32_eq_sha2
#audit IcarusSha256CarrierSigma32.zxor3_is_addition
#audit IcarusSha256CarrierSigma32.zxor3_apply
#audit IcarusSha256CarrierSigma32.zxor3_eq_zero_iff

end IcarusSha256CarrierSigma32
