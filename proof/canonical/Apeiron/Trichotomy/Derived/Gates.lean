import Apeiron.Trichotomy.NullBasis
import Apeiron.Trichotomy.ChannelsAnyRing

/-!
# Logic gates on the two channels

@name Logic Gates
@cite boole-1847
@cite stone-1936
@cite peirce-1870
@grade proved

Above zero the algebra is two independent copies of the reals (`NullBasis`). Put a truth
value on it by sending `true` to `idemPlus` and `false` to `idemMinus`. In coordinates those
are the two corners that carry all of their weight in one channel:

```
                      first channel    second channel
   true   idemPlus          1                0
   false  idemMinus         0                1
```

Now multiply a whole list of them together. Multiplication is componentwise, so the first
channel of the answer is the product of the first coordinates and the second channel is the
product of the second coordinates. A product of zeroes and ones is `1` exactly when every
factor is `1`. So:

```
   first channel  = 1  exactly when every input is true      AND
   second channel = 1  exactly when every input is false     NOR
```

One multiplication per element, and both gates come out at once, on two wires that never
communicate. `fold_is_and_and_nor` is that statement.

The second channel carries `NOR` rather than `OR` because sending `false` to the second
corner puts the negation in. `OR` is its complement, and the complement is the swap of the
two channels rather than an operation added from outside.

Because each channel is an ordinary product of reals, reordering the list cannot change the
answer (`fold_perm`). Nothing in the fold depends on the order the inputs arrive.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

/-- A truth value as a real number.

    (b : Bool)              [a truth value]
    if b then 1 else 0      [one for true, zero for false]
    : ℝ                     [a plain real number] -/
def boolVal (b : Bool) : ℝ := if b then 1 else 0

/-- Multiplying two truth values is AND.

@cite boole-1847

    (b c : Bool)                          [two truth values]
    boolVal b * boolVal c                 [their real values, multiplied]
    = boolVal (b && c)                    [the value of their conjunction]

    So: `1·1 = 1` and every other product has a zero in it. This is `AnyRing.ringBoolVal_mul`
    at `R = ℝ`. -/
theorem boolVal_mul (b c : Bool) : boolVal b * boolVal c = boolVal (b && c) :=
  AnyRing.ringBoolVal_mul ℝ b c

/-- A truth value carried by the algebra: true is one corner, false is the other.

    (σ : ℝ)                                        [the parameter]
    (b : Bool)                                     [a truth value]
    if b then idemPlus σ else idemMinus σ          [true goes to one corner, false the other]
    : Cell                                         [a point of the algebra] -/
def cellOfBool (σ : ℝ) (b : Bool) : Cell := if b then idemPlus σ else idemMinus σ

/-- Carrying a truth value is the generic construction at the real root and half.

    (σ : ℝ)                    [the parameter]
    cellOfBool σ               [the map sending true and false to the two corners]
    Real.sqrt σ                [the number that squares to give sigma]
    cellOfBool σ = AnyRing.ringCellOfBool σ (Real.sqrt σ) (1 / Real.sqrt σ) (1 / 2)
                               [the real carried map is the generic one]

    So: `idemPlus_eq_ringIdemPlus` and `idemMinus_eq_ringIdemMinus` read through the two
    branches. This is what lets the gate theorems below call the generic gates. -/
theorem cellOfBool_eq_ringCellOfBool (σ : ℝ) :
    cellOfBool σ =
      AnyRing.ringCellOfBool σ (Real.sqrt σ) (1 / Real.sqrt σ) (1 / 2) := by
  funext b
  cases b
  · simp [cellOfBool, AnyRing.ringCellOfBool, idemMinus_eq_ringIdemMinus]
  · simp [cellOfBool, AnyRing.ringCellOfBool, idemPlus_eq_ringIdemPlus]

/-- The first coordinate of a carried truth value is its real value.

    {σ : ℝ} (hσ : 0 < σ)                             [the parameter, above zero]
    (b : Bool)                                       [a truth value]
    upper σ (cellOfBool σ b) = boolVal b             [the first channel carries it directly]

    So: `AnyRing.ringUpper_ringCellOfBool` at the real witnesses. -/
theorem upper_cellOfBool {σ : ℝ} (hσ : 0 < σ) (b : Bool) :
    upper σ (cellOfBool σ b) = boolVal b := by
  rw [cellOfBool_eq_ringCellOfBool]
  exact AnyRing.ringUpper_ringCellOfBool
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num) b

/-- The second coordinate of a carried truth value is the value of its negation.

    {σ : ℝ} (hσ : 0 < σ)                                [the parameter, above zero]
    (b : Bool)                                          [a truth value]
    lower σ (cellOfBool σ b) = boolVal (!b)             [the second channel carries the negation]

    So: `AnyRing.ringLower_ringCellOfBool` at the real witnesses. -/
theorem lower_cellOfBool {σ : ℝ} (hσ : 0 < σ) (b : Bool) :
    lower σ (cellOfBool σ b) = boolVal (!b) := by
  rw [cellOfBool_eq_ringCellOfBool]
  exact AnyRing.ringLower_ringCellOfBool
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num) b

-- ═══════════════════════════════════════════════════════════════════
-- MULTIPLYING A WHOLE LIST
-- ═══════════════════════════════════════════════════════════════════

/-- Multiply every point in a list together, starting from the unit.

    (σ : ℝ)                       [the parameter]
    (zs : List Cell)              [a list of points]
    zs.foldr (mul σ) one          [multiply them all, right to left, starting at the unit]
    : Cell                        [a point of the algebra] -/
def fold (σ : ℝ) (zs : List Cell) : Cell := zs.foldr (mul σ) one

/-- The first coordinate of a product is the product of the first coordinates.

    {σ : ℝ} (hσ : 0 ≤ σ)                             [the parameter, at zero or above]
    (zs : List Cell)                                 [a list of points]
    upper σ (fold σ zs)                              [multiply everything, then read channel one]
    = (zs.map (upper σ)).prod                        [read channel one of each, then multiply]

    So: `upper_mul` carried through the whole list by induction. This is `AnyRing.ringUpper_fold`
    at the real root. -/
theorem upper_fold {σ : ℝ} (hσ : 0 ≤ σ) (zs : List Cell) :
    upper σ (fold σ zs) = (zs.map (upper σ)).prod :=
  AnyRing.ringUpper_fold (Real.sq_sqrt hσ) zs

/-- The second coordinate of a product is the product of the second coordinates.

    {σ : ℝ} (hσ : 0 ≤ σ)                             [the parameter, at zero or above]
    (zs : List Cell)                                 [a list of points]
    lower σ (fold σ zs)                              [multiply everything, then read channel two]
    = (zs.map (lower σ)).prod                        [read channel two of each, then multiply]

    So: `AnyRing.ringLower_fold` at the real root. -/
theorem lower_fold {σ : ℝ} (hσ : 0 ≤ σ) (zs : List Cell) :
    lower σ (fold σ zs) = (zs.map (lower σ)).prod :=
  AnyRing.ringLower_fold (Real.sq_sqrt hσ) zs

/-- The size of a whole product is the two channel products, closed by one multiplication.

    {σ : ℝ} (hσ : 0 ≤ σ)                                   [the parameter, at zero or above]
    (zs : List Cell)                                              [a list of points]
    normForm σ (fold σ zs)                                            [the size of the product]
    = (zs.map (upper σ)).prod * (zs.map (lower σ)).prod           [the two channels, multiplied]

    So: no point in the list is ever squared. Two running products and one final
    multiplication. This is `AnyRing.ringNormForm_fold` at the real root. -/
theorem normForm_fold {σ : ℝ} (hσ : 0 ≤ σ) (zs : List Cell) :
    normForm σ (fold σ zs) = (zs.map (upper σ)).prod * (zs.map (lower σ)).prod :=
  AnyRing.ringNormForm_fold (Real.sq_sqrt hσ) zs

/-- Reordering the list cannot change the product.

    {σ : ℝ} (hσ : 0 ≤ σ)                       [the parameter, at zero or above]
    {zs ws : List Cell}                        [two lists]
    (h : zs.Perm ws)                           [one is a reordering of the other]
    upper σ (fold σ zs) = upper σ (fold σ ws)  [same first channel]
    ∧                                          [and]
    lower σ (fold σ zs) = lower σ (fold σ ws)  [same second channel]

    So: each channel is an ordinary product of real numbers, and multiplication of reals does
    not care about order. This is `AnyRing.ringFold_perm` at the real root. -/
theorem fold_perm {σ : ℝ} (hσ : 0 ≤ σ) {zs ws : List Cell} (h : zs.Perm ws) :
    upper σ (fold σ zs) = upper σ (fold σ ws) ∧
      lower σ (fold σ zs) = lower σ (fold σ ws) :=
  AnyRing.ringFold_perm (Real.sq_sqrt hσ) h

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO GATES
-- ═══════════════════════════════════════════════════════════════════

/-- The first channel of a fold over truth values is AND.

    {σ : ℝ} (hσ : 0 < σ)                                    [the parameter, above zero]
    (bs : List Bool)                                        [a list of truth values]
    upper σ (fold σ (bs.map (cellOfBool σ)))                [carry them, multiply, read channel one]
    = boolVal (bs.all id)                                   [one exactly when all are true]

    So: `AnyRing.ringUpper_fold_bool` at the real root and half. -/
theorem upper_fold_bool {σ : ℝ} (hσ : 0 < σ) (bs : List Bool) :
    upper σ (fold σ (bs.map (cellOfBool σ))) = boolVal (bs.all id) := by
  rw [cellOfBool_eq_ringCellOfBool]
  exact AnyRing.ringUpper_fold_bool (Real.sq_sqrt hσ.le)
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num) bs

/-- The second channel of a fold over truth values is NOR.

    {σ : ℝ} (hσ : 0 < σ)                                    [the parameter, above zero]
    (bs : List Bool)                                        [a list of truth values]
    lower σ (fold σ (bs.map (cellOfBool σ)))                [carry them, multiply, read channel two]
    = boolVal (!(bs.any id))                                [one exactly when none is true]

    So: `AnyRing.ringLower_fold_bool` at the real root and half. -/
theorem lower_fold_bool {σ : ℝ} (hσ : 0 < σ) (bs : List Bool) :
    lower σ (fold σ (bs.map (cellOfBool σ))) = boolVal (!(bs.any id)) := by
  rw [cellOfBool_eq_ringCellOfBool]
  exact AnyRing.ringLower_fold_bool (Real.sq_sqrt hσ.le)
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num) bs

/-- One pass over the list computes both gates at once.

    {σ : ℝ} (hσ : 0 < σ)                                       [the parameter, above zero]
    (bs : List Bool)                                           [a list of truth values]
    upper σ (fold σ (bs.map (cellOfBool σ))) = boolVal (bs.all id)         [channel one is AND]
    ∧                                                          [and, at the same time]
    lower σ (fold σ (bs.map (cellOfBool σ))) = boolVal (!(bs.any id))      [channel two is NOR]

    So: one multiplication per input, two answers. The two channels are independent, so
    neither gate costs anything the other did not already pay for. -/
theorem fold_is_and_and_nor {σ : ℝ} (hσ : 0 < σ) (bs : List Bool) :
    upper σ (fold σ (bs.map (cellOfBool σ))) = boolVal (bs.all id) ∧
      lower σ (fold σ (bs.map (cellOfBool σ))) = boolVal (!(bs.any id)) :=
  ⟨upper_fold_bool hσ bs, lower_fold_bool hσ bs⟩

/-- Carrying a truth value into the quotient.

    (σ : ℝ)                             [the parameter]
    (b : Bool)                          [a truth value]
    toQuot σ (cellOfBool σ b)           [the carried value, sent into the quotient]
    : AdjoinRoot (rel σ)                [an element of the quotient ring] -/
def quotOfBool (σ : ℝ) (b : Bool) : AdjoinRoot (rel σ) := toQuot σ (cellOfBool σ b)

/-- Multiplying carried truth values in the quotient computes AND on one channel and NOR on
the other.

    {σ : ℝ} (hσ : 0 < σ)                     [the parameter, above zero]
    (bs : List Bool)                         [a list of truth values]
    (bs.map (quotOfBool σ)).prod             [carry each into the ring, multiply them all]
    upper σ (... ) = boolVal (bs.all id)     [channel one reads AND]
    ∧                                        [and]
    lower σ (... ) = boolVal (!(bs.any id))  [channel two reads NOR]

    So: `fold_is_and_and_nor` with the product taken in the quotient ring rather than by the
    hand-written fold. The gates are a fact about `ℝ[X]/(X² − σ)`. -/
theorem quot_fold_is_and_and_nor {σ : ℝ} (hσ : 0 < σ) (bs : List Bool) :
    upper σ ((toQuotEquiv σ).symm ((bs.map (quotOfBool σ)).prod)) = boolVal (bs.all id) ∧
      lower σ ((toQuotEquiv σ).symm ((bs.map (quotOfBool σ)).prod))
        = boolVal (!(bs.any id)) := by
  have key : (bs.map (quotOfBool σ)).prod = toQuot σ (fold σ (bs.map (cellOfBool σ))) := by
    induction bs with
    | nil => simpa [fold] using (toQuot_one σ).symm
    | cons b bs ih =>
      simp only [List.map_cons, List.prod_cons, ih, quotOfBool, fold, List.foldr_cons,
        mul_eq_quotient_mul]
  have back : (toQuotEquiv σ).symm (toQuot σ (fold σ (bs.map (cellOfBool σ))))
      = fold σ (bs.map (cellOfBool σ)) := by
    rw [← toQuotEquiv_apply]; exact (toQuotEquiv σ).symm_apply_apply _
  rw [key, back]
  exact fold_is_and_and_nor hσ bs

end Trichotomy
end Apeiron

end
