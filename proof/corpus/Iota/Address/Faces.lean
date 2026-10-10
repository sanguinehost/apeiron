/-
# Face-correct addressing

An address is one 64-bit source word computed from a memo key's face, its algebra
parameter `σ` and its address cell. Each face has its own formula:

* the elliptic face packs its two bounded coordinates into one word,
  `ellipticPack`;
* the parabolic face takes the linear form `a + σ · b`, `parabolicCoord`;
* the hyperbolic face takes the norm form `a · a − σ · b · b`, `hyperbolicCoord`.

`coord` dispatches on the face, so a key of one face is never addressed by another
face's formula. `homeShard` reduces the coordinate modulo 256 to one of the fixed
shards. There is no fallback shard: `checkInput` refuses an out-of-domain elliptic
cell before any key exists, and the parabolic and hyperbolic faces admit every
cell.

`ellipticUnpack` inverts `ellipticPack` on the `.elliptic` domain. That inverse
rests on `Nat` division and modulus. The required facts are proved in this module
with an empty axiom cone because the library's `Nat.div` and `Nat.mod` lemmas
reach `propext`; `divGoFuelCongr` is the fuel-independence lemma the definition of
`Nat.div` needs, proved by structural recursion on the fuel exactly as the word
ring's `modCoreGoFuelCongr` is.

`hyperbolicCoord` is the norm form, and `hyperbolicCoord_denotes_ringNormForm`
links it to the canonical `Apeiron.Trichotomy.AnyRing.ringNormForm` through the
constructive `Iota.Refinement.{wordDenote,cellDenote,cellNormForm}` chain.
-/
import Iota.Memo.Key
import Iota.Refinement.WordDenotation

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.unusedVariables false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Address

open Iota

/-- The address word: one 64-bit source word. -/
abbrev Coord := Iota.Word

/-- One of the 256 fixed home shards. -/
abbrev Shard := Fin 256

/-- Why an input was refused before it became a key. -/
inductive AddressRefusal where
  | ellipticOutOfDomain

/-- The packing base `2 ^ 32`: the bound both elliptic coordinates stay below. -/
abbrev packBase : Nat := Nat.pow 2 32

/-- `packBase` is positive. -/
theorem packBase_pos : 0 < packBase := by
  decide

/-- `packBase` is below the word modulus, so a packed pair fits one word. -/
theorem packBase_lt_wordModulus : packBase < Iota.wordModulus := by
  have h : packBase * 1 < packBase * packBase :=
    Nat.mul_lt_mul_of_pos_left (by decide : 1 < packBase) packBase_pos
  rw [Nat.mul_one packBase] at h
  exact h

/-- Two coordinates below the packing base pack into a word below the modulus. -/
theorem packLt {left right : Nat} (hleft : left < packBase) (hright : right < packBase) :
    left * packBase + right < Iota.wordModulus := by
  have hstep : left * packBase + right < (left + 1) * packBase := by
    rw [Nat.succ_mul]
    exact Nat.add_lt_add_left hright (left * packBase)
  have hbound : (left + 1) * packBase ≤ packBase * packBase :=
    Nat.mul_le_mul_right packBase (Nat.succ_le_of_lt hleft)
  have hword : packBase * packBase = Iota.wordModulus := rfl
  rw [hword] at hbound
  exact Nat.lt_of_lt_of_le hstep hbound

/-- `Nat.div.go` does not depend on its fuel. The definition of `Nat.div` recurses by subtracting
the divisor, so a structural recursion on the fuel mirrors the definition exactly. -/
theorem divGoFuelCongr (value modulus fuel₁ fuel₂ : Nat) (positive : 0 < modulus)
    (bound₁ : value < fuel₁) (bound₂ : value < fuel₂) :
    Nat.div.go modulus positive fuel₁ value bound₁ =
      Nat.div.go modulus positive fuel₂ value bound₂ := by
  match fuel₁, fuel₂ with
  | 0, _ => exact False.elim (Nat.not_lt_zero _ bound₁)
  | _, 0 => exact False.elim (Nat.not_lt_zero _ bound₂)
  | fuel₁ + 1, fuel₂ + 1 =>
      rw [Nat.div.go.eq_1 modulus positive value fuel₁ bound₁,
        Nat.div.go.eq_1 modulus positive value fuel₂ bound₂]
      match Nat.decLe modulus value with
      | isTrue lower =>
          rw [dif_pos lower, dif_pos lower]
          exact congrArg (fun total => total + 1)
            (divGoFuelCongr (value - modulus) modulus fuel₁ fuel₂ positive
              (Nat.div_rec_fuel_lemma positive lower bound₁)
              (Nat.div_rec_fuel_lemma positive lower bound₂))
      | isFalse _ => rfl
termination_by structural fuel₁

/-- The division step: above the divisor, one subtraction turns into one unit. -/
theorem divEqSubDiv {modulus value : Nat} (positive : 0 < modulus) (bound : modulus ≤ value) :
    value / modulus = (value - modulus) / modulus + 1 := by
  change Nat.div value modulus = Nat.div (value - modulus) modulus + 1
  rw [Nat.div.eq_1, dif_pos positive]
  rw [Nat.div.go.eq_1 modulus positive value value (Nat.lt_succ_self value), dif_pos bound]
  rw [Nat.div.eq_1 (value - modulus), dif_pos positive]
  exact congrArg (fun total => total + 1)
    (divGoFuelCongr (value - modulus) modulus value (value - modulus + 1) positive
      (Nat.div_rec_fuel_lemma positive bound (Nat.lt_succ_self value))
      (Nat.lt_succ_self (value - modulus)))

/-- Below the divisor, the quotient is zero. -/
theorem divSmall {modulus value : Nat} (positive : 0 < modulus) (bound : value < modulus) :
    value / modulus = 0 := by
  change Nat.div value modulus = 0
  rw [Nat.div.eq_1, dif_pos positive]
  rw [Nat.div.go.eq_1 modulus positive value value (Nat.lt_succ_self value),
    dif_neg (Nat.not_le_of_gt bound)]

/-- Base-`packBase` division: the high coordinate of a packed pair. -/
theorem divMulAdd (modulus : Nat) (positive : 0 < modulus) :
    ∀ (left right : Nat), right < modulus →
      (left * modulus + right) / modulus = left := by
  intro left
  induction left with
  | zero =>
      intro right hright
      rw [Nat.zero_mul, Nat.zero_add]
      exact divSmall positive hright
  | succ left ih =>
      intro right hright
      rw [Nat.succ_mul]
      have hle : modulus ≤ left * modulus + modulus + right :=
        Nat.le_trans (Nat.le_add_left modulus (left * modulus))
          (Nat.le_add_right (left * modulus + modulus) right)
      rw [divEqSubDiv positive hle]
      have hsub : left * modulus + modulus + right - modulus = left * modulus + right := by
        rw [Nat.add_assoc, Nat.add_comm modulus right, ← Nat.add_assoc]
        exact Iota.Refinement.addSubCancel (left * modulus + right) modulus
      rw [hsub, ih right hright]

/-- Base-`packBase` modulus: the low coordinate of a packed pair. -/
theorem modMulAdd (modulus : Nat) :
    ∀ (left right : Nat), right < modulus →
      (left * modulus + right) % modulus = right := by
  intro left
  induction left with
  | zero =>
      intro right hright
      rw [Nat.zero_mul, Nat.zero_add]
      exact Iota.Refinement.modEqOfLt hright
  | succ left ih =>
      intro right hright
      rw [Nat.succ_mul]
      have hrearr : left * modulus + modulus + right = (left * modulus + right) + modulus := by
        rw [Nat.add_assoc, Nat.add_comm modulus right, ← Nat.add_assoc]
      rw [hrearr, Iota.Refinement.modAddModulus (left * modulus + right) modulus,
        ih right hright]

/-- Pack an elliptic cell: the high coordinate shifted by `2 ^ 32`, plus the low one. `σ` is
carried so every executing signature names the algebra parameter; the packing itself does not
read it. -/
def ellipticPack (sigma : Iota.Word)
    (cell : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word) : Coord :=
  Iota.Word.ofNat (cell.1.val * packBase + cell.2.val)

/-- Unpack an elliptic coordinate: divide and take the remainder by `2 ^ 32`. -/
def ellipticUnpack (sigma : Iota.Word)
    (coord : Coord) : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word :=
  (Iota.Word.ofNat (coord.val / packBase), Iota.Word.ofNat (coord.val % packBase))

/-- `ellipticUnpack` inverts `ellipticPack` on the `.elliptic` domain: both coordinates below
`2 ^ 32` fit one word, and division and remainder recover them exactly. -/
theorem ellipticUnpack_pack (sigma : Iota.Word)
    (cell : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word)
    (domain : Iota.Memo.FaceDomain .elliptic cell) :
    ellipticUnpack sigma (ellipticPack sigma cell) = cell := by
  have hleft : cell.1.val < packBase := domain.1
  have hright : cell.2.val < packBase := domain.2
  have hval : (ellipticPack sigma cell).val = cell.1.val * packBase + cell.2.val :=
    Iota.Refinement.modEqOfLt (packLt hleft hright)
  have hdiv : (ellipticPack sigma cell).val / packBase = cell.1.val := by
    rw [hval]
    exact divMulAdd packBase packBase_pos cell.1.val cell.2.val hright
  have hmod : (ellipticPack sigma cell).val % packBase = cell.2.val := by
    rw [hval]
    exact modMulAdd packBase cell.1.val cell.2.val hright
  apply Prod.ext
  · show Iota.Word.ofNat ((ellipticPack sigma cell).val / packBase) = cell.1
    rw [hdiv]
    apply Fin.ext
    rw [Iota.Word.ofNat_val]
    exact Iota.Refinement.modEqOfLt (Nat.lt_trans hleft packBase_lt_wordModulus)
  · show Iota.Word.ofNat ((ellipticPack sigma cell).val % packBase) = cell.2
    rw [hmod]
    apply Fin.ext
    rw [Iota.Word.ofNat_val]
    exact Iota.Refinement.modEqOfLt (Nat.lt_trans hright packBase_lt_wordModulus)

/-- The parabolic coordinate: the linear form `a + σ · b`, which at `σ = 0` is the second slot of
the dual-number face. -/
def parabolicCoord (sigma : Iota.Word)
    (cell : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word) : Coord :=
  Iota.Word.add cell.1 (Iota.Word.mul sigma cell.2)

/-- The hyperbolic coordinate: the norm form `a · a − σ · b · b`. It is the quantity the
constructive `Iota.Refinement.cellNormForm` computes and the canonical
`Apeiron.Trichotomy.AnyRing.ringNormForm` denotes. -/
def hyperbolicCoord (sigma : Iota.Word)
    (cell : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word) : Coord :=
  Iota.Refinement.cellNormForm sigma cell

/-- The address of a key, by face. -/
def coord {sources : Iota.Source.SourceArena} {stateCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    (key : Iota.Memo.MemoKey sources states) : Coord :=
  match key.face with
  | .elliptic => ellipticPack key.sigma key.addressCell
  | .parabolic => parabolicCoord key.sigma key.addressCell
  | .hyperbolic => hyperbolicCoord key.sigma key.addressCell

/-- The one home shard of a key: the coordinate reduced modulo 256. There is no second probe and no
fallback shard. -/
def homeShard {sources : Iota.Source.SourceArena} {stateCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    (key : Iota.Memo.MemoKey sources states) : Shard :=
  ⟨(coord key).val % 256, Nat.mod_lt _ (Nat.zero_lt_succ 255)⟩

/-- Admit an input on a face: the elliptic face checks both coordinates are below `2 ^ 32` and
refuses otherwise; the parabolic and hyperbolic faces admit every cell. The result carries `σ` and
the face, so the admitted input refines the memo key it will become. -/
def checkInput {sources : Iota.Source.SourceArena} {stateCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    (sigma : Iota.Word) (face : Iota.Memo.Face)
    (input : Iota.Memo.MemoInput sources states) :
    Except AddressRefusal (Iota.Memo.AdmittedInput sources states sigma face) :=
  match face with
  | .elliptic =>
      if h : input.addressCell.1.val < packBase ∧ input.addressCell.2.val < packBase then
        .ok { input := input, admitted := h }
      else
        .error .ellipticOutOfDomain
  | .parabolic => .ok { input := input, admitted := trivial }
  | .hyperbolic => .ok { input := input, admitted := trivial }

/-- An observation of hyperbolic cells is in the hyperbolic domain when it factors through the
hyperbolic coordinate: equal coordinates force equal observations. -/
def HyperbolicObservationDomain {α : Type} (sigma : Iota.Word)
    (observe : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word → α) : Prop :=
  ∀ (left right : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word),
    hyperbolicCoord sigma left = hyperbolicCoord sigma right → observe left = observe right

/-- Two hyperbolic cells with the same coordinate are observed identically, provided the
observation is in the hyperbolic domain. -/
theorem hyperbolic_observation_sound {α : Type} {sigma : Iota.Word}
    {observe : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word → α}
    (domain : HyperbolicObservationDomain sigma observe)
    {left right : Apeiron.Trichotomy.AnyRing.RingCell Iota.Word}
    (equal : hyperbolicCoord sigma left = hyperbolicCoord sigma right) :
    observe left = observe right :=
  domain left right equal

/-- The constructive norm form denotes the canonical ring norm form: the `wordDenote` of
`cellNormForm` is `ringNormForm` at the `wordDenote`d parameter and the `cellDenote`d cell. -/
theorem hyperbolicCoord_denotes_ringNormForm (sigma : Iota.Word) (cell : Iota.Word × Iota.Word) :
    Iota.Refinement.wordDenote (Iota.Refinement.cellNormForm sigma cell) =
      Apeiron.Trichotomy.AnyRing.ringNormForm (Iota.Refinement.wordDenote sigma)
        (Iota.Refinement.cellDenote cell) :=
  Iota.Refinement.cellNormForm_denote sigma cell

end Iota.Address
