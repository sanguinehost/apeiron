import Apeiron.Physics.Kepler.Enclosure
import Mathlib.Tactic

/-!
# Conditional interpretation of a Kepler certificate

`FiniteBinary64` records the sign, biased exponent and fraction of an admitted
finite binary64 bit pattern. Its rational value is exact; positive and negative
zero have different signs but the same rational value. These fields are a
mathematical representation, not a decoder of Rust `f64` or an Aeneas refinement.

For the returned patterns `⟦m⟧`, `⟦e⟧`, `⟦x⟧`, `⟦d⟧` and `⟦r⟧`, `Sound` demands
a supplied SOUND outward bound on the exact-real residual and a supplied
outward error bound. A rounded evaluation of sine is not this premise. The
conditional theorem does not establish that Rust constructed either bound.
If those proofs cannot be supplied, `certificateFailure` carries no value.
The executing refusal and bit-pattern conversion remain open obligations.

With `m e x d` naming the exact-real denotations of the four patterns, the
supplied premise is `|x - e * Real.sin x - m| ≤ d` and the canonical error
bound is `d / (1 - |e|)`. A returned finite `r` must bound that quotient.
-/

namespace Kepler.Certificate

/-- Decoded fields of an admitted finite binary64 bit pattern. The exponent
    `Fin 2047` excludes the all-ones exponent used by infinity and NaN. -/
structure FiniteBinary64 where
  negative : Bool
  exponent : Fin 2047
  fraction : Fin (2 ^ 52)

/-- The exact rational value of a finite binary64 bit pattern. The normal
    exponent is biased by 1023; subnormals use the fixed exponent minus 1074. -/
def FiniteBinary64.denoteRational (b : FiniteBinary64) : ℚ :=
  (if b.negative then -1 else 1) *
    (if b.exponent.val = 0 then
       (b.fraction.val : ℚ) * (2 : ℚ) ^ (-1074 : ℤ)
     else
       ((2 ^ 52 + b.fraction.val : ℕ) : ℚ) *
         (2 : ℚ) ^ ((b.exponent.val : ℤ) - 1075))

/-- Embed the exact rational represented by a finite pattern in the reals.

    ℝ [the real numbers: every number on the number line] -/
def FiniteBinary64.denoteReal (b : FiniteBinary64) : ℝ := b.denoteRational

/-- Either sign bit represents a distinct zero pattern.

    Bool [true or false, nothing else] -/
def FiniteBinary64.zeroBits (negative : Bool) : FiniteBinary64 :=
  ⟨negative, 0, 0⟩

/-- Signed zeros have the same exact rational denotation; their sign bits remain distinct.

    Bool [true or false, nothing else] -/
theorem signed_zero_denotes_zero (negative : Bool) :
    (FiniteBinary64.zeroBits negative).denoteRational = 0 := by
  change (if negative then (-1 : ℚ) else 1) *
    ((0 : ℚ) * (2 : ℚ) ^ (-1074 : ℤ)) = 0
  rw [zero_mul, mul_zero]

/-- The five admitted finite patterns of a candidate certificate. -/
structure CertifiedBounds where
  mean : FiniteBinary64
  eccentricity : FiniteBinary64
  candidate : FiniteBinary64
  defect : FiniteBinary64
  error : FiniteBinary64

/-- Soundness is supplied, not inferred from rounded evaluation. The first
    inequality bounds the exact-real sine residual from above; the second
    bounds the quotient from above by the returned finite error pattern. -/
def CertifiedBounds.Sound (c : CertifiedBounds) : Prop :=
  0 ≤ c.defect.denoteReal ∧
  |c.candidate.denoteReal - c.eccentricity.denoteReal *
      Real.sin c.candidate.denoteReal - c.mean.denoteReal| ≤ c.defect.denoteReal ∧
  c.defect.denoteReal / (1 - |c.eccentricity.denoteReal|) ≤ c.error.denoteReal

/-- A success is possible only with a finite-pattern certificate and its
    supplied soundness proof. Failure contains no default candidate or bound. -/
inductive CertifiedResult where
  | accepted (c : CertifiedBounds) (he : |c.eccentricity.denoteReal| < 1)
      (hsound : c.Sound)
  | certificateFailure

/-- The exact-real defect premise gives the enclosure at the constructed root.
    This applies `Apeiron.Physics.Kepler.absolute_defect_enclosure` at denoted
    `m e x d`, without another sine Lipschitz proof.

    < [is smaller than]
    ≤ [is smaller than or equal to] -/
theorem kepler_abs_defect_enclosure_denoted (c : CertifiedBounds)
    (he : |c.eccentricity.denoteReal| < 1) (hsound : c.Sound) :
    |c.candidate.denoteReal - Apeiron.Physics.Kepler.eccentricAnomaly
      c.mean.denoteReal c.eccentricity.denoteReal he| ≤
      c.defect.denoteReal / (1 - |c.eccentricity.denoteReal|) := by
  rcases hsound with ⟨hd, hbound, _⟩
  exact Apeiron.Physics.Kepler.absolute_defect_enclosure
    c.mean.denoteReal c.eccentricity.denoteReal c.candidate.denoteReal
    c.defect.denoteReal he hd hbound

/-- A successful interpretation bounds the error by the returned finite pattern.

    < [is smaller than]
    ≤ [is smaller than or equal to] -/
theorem accepted_error_encloses (c : CertifiedBounds)
    (he : |c.eccentricity.denoteReal| < 1) (hsound : c.Sound) :
    |c.candidate.denoteReal - Apeiron.Physics.Kepler.eccentricAnomaly
      c.mean.denoteReal c.eccentricity.denoteReal he| ≤ c.error.denoteReal :=
  (kepler_abs_defect_enclosure_denoted c he hsound).trans hsound.2.2

/-- Certificate failure cannot be reinterpreted as an accepted result.

    < [is smaller than]
    ≠ [is not equal to] -/
theorem refusal_has_no_value (c : CertifiedBounds)
    (he : |c.eccentricity.denoteReal| < 1) (hsound : c.Sound) :
    CertifiedResult.certificateFailure ≠ CertifiedResult.accepted c he hsound := by
  intro h
  cases h

/-- Circular zero patterns jointly inhabit the certificate's soundness and domain.

    ∃ [there is at least one]
    ∧ [both are true]
    < [is smaller than] -/
theorem circular_joint_witness :
    ∃ c : CertifiedBounds, c.mean.denoteReal = 0 ∧
      c.eccentricity.denoteReal = 0 ∧ c.candidate.denoteReal = 0 ∧
      c.defect.denoteReal = 0 ∧ c.error.denoteReal = 0 ∧
      |c.eccentricity.denoteReal| < 1 ∧ c.Sound := by
  let z := FiniteBinary64.zeroBits false
  refine ⟨⟨z, z, z, z, z⟩, ?_⟩
  have hz : z.denoteReal = 0 := by
    change ((FiniteBinary64.zeroBits false).denoteRational : ℝ) = 0
    rw [signed_zero_denotes_zero]
    norm_num
  simp [CertifiedBounds.Sound, hz]

/-- Positive eccentricity also jointly inhabits the finite-pattern certificate domain.

    ∃ [there is at least one]
    ∧ [both are true]
    < [is smaller than] -/
theorem positive_joint_witness :
    ∃ c : CertifiedBounds, 0 < c.eccentricity.denoteReal ∧
      |c.eccentricity.denoteReal| < 1 ∧ c.Sound := by
  let z := FiniteBinary64.zeroBits false
  let half : FiniteBinary64 := ⟨false, 1022, 0⟩
  have hh : half.denoteReal = (1 / 2 : ℝ) := by
    norm_num [half, FiniteBinary64.denoteReal, FiniteBinary64.denoteRational]
  refine ⟨⟨z, half, z, z, z⟩, ?_⟩
  have hz : z.denoteReal = 0 := by
    change ((FiniteBinary64.zeroBits false).denoteRational : ℝ) = 0
    rw [signed_zero_denotes_zero]
    norm_num
  simp [CertifiedBounds.Sound, hz, hh]
  norm_num

end Kepler.Certificate
