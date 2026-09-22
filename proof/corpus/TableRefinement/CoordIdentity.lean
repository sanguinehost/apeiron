import ApeironWord
import Apeiron.Table.WordIdentity

/-!
# Coordinate identity-word refinement

The extracted `coord.spread` widens its input and `coord.PHI` to `U128`, multiplies them, and
truncates the product to the `U64` identity field. This module identifies that field with the
canonical low-word operation and transfers `Apeiron.Table.phi64_lowWord_injective`.
-/

namespace Apeiron.TableRefinement

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- The extracted hexadecimal spread multiplier is the canonical natural-number multiplier. -/
theorem coord_PHI_val : coord.PHI.val = phi64 := by
  rw [coord.PHI]
  rfl

/-- Widened multiplication followed by truncation is the canonical low 64-bit product. -/
theorem spread_low_word (x : U64) :
    (UScalar.cast .U64
      (core.num.U128.wrapping_mul (UScalar.cast .U128 x)
        (UScalar.cast .U128 coord.PHI))).val =
      lowWord 64 (x.val * phi64) := by
  rw [UScalar.cast_val_eq, core.num.U128.wrapping_mul_val_eq]
  simp only [U64.cast_U128_val_eq, lowWord]
  rw [coord_PHI_val]
  have hx := x.hBounds
  have hfit := phi64_product_fits_u128 hx
  rw [UScalar.size]
  change ((x.val * phi64) % 2 ^ 128) % 2 ^ 64 = (x.val * phi64) % 2 ^ 64
  rw [Nat.mod_eq_of_lt hfit]

/-- The expression assigned to the identity field by the extracted body. -/
def coordSpreadId (x : U64) : U64 :=
  UScalar.cast .U64
    (core.num.U128.wrapping_mul (UScalar.cast .U128 x)
      (UScalar.cast .U128 coord.PHI))

/-- The real extracted `coord.spread` identity field denotes the canonical low-word product. -/
theorem coord_spread_id_low_word (x : U64) :
    coord.spread x ⦃ addr => addr.id.val = lowWord 64 (x.val * phi64) ⦄ := by
  unfold coord.spread
  step
  step
  step
  step
  step
  step
  step
  step
  rw [lo_post, w_post, i_post, i1_post]
  exact spread_low_word x

/-- The real extracted `coord.spread` body returns `coordSpreadId` in its identity field. -/
theorem coord_spread_returns_id (x : U64) :
    coord.spread x ⦃ addr => addr.id = coordSpreadId x ⦄ := by
  unfold coord.spread
  step
  step
  step
  step
  step
  step
  step
  step
  rw [lo_post, w_post, i_post, i1_post]
  rfl

/-- The extracted identity-field expression is injective on `U64`. -/
theorem coordSpreadId_injective : Function.Injective coordSpreadId := by
  intro x y hid
  apply UScalar.eq_of_val_eq
  apply phi64_lowWord_injective x.hBounds y.hBounds
  rw [← spread_low_word x, ← spread_low_word y]
  exact congrArg UScalar.val hid

/-- Equal identity fields produced by two real `coord.spread` calls imply equal inputs. -/
theorem coord_spread_id_injective (x y : U64) :
    coord.spread x ⦃ ax => coord.spread y ⦃ ay => ax.id = ay.id → x = y ⦄ ⦄ := by
  unfold coord.spread
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  intro hid
  apply coordSpreadId_injective
  subst_vars
  simpa [coordSpreadId] using lo_post

end Apeiron.TableRefinement
