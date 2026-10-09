import Apeiron.Decider.Cert

/-!
# Finite cutting-plane derivations and certificate compilation

@name Cutting-Plane Derivation Compiler
@grade proved

The equation `2x = 1` supplies one input row. A gcd cut derives `1 ≤ x ≤ 0`, so closing
that row gives the certificate `r 0; t 0; x 1`. A derivation records the rows and the checked
side conditions; its compiler emits only the operation names, indices, and multipliers.

`CuttingPlaneInference` uses the existing row operations. Its indices name earlier rows,
combination multipliers are positive integers, and a meet carries equal coefficients.
`CuttingPlaneDerivation` is a finite sequence of these inferences. The compiler preserves
the exact emitted rows, and `compile_accepts` supplies a checked log from a derived crossing.

## Root

| from | here |
|---|---|
| `checkCert_sound`, through `crMeet_holds` and `crImpossible_empty` | `refutes` |

The compiler is a fact about the certificate syntax. It introduces no algebraic primitive.
The soundness dependency above reaches `inCell_meet_iff` and `crossing_ofBounds` through the
existing checker. This file does not prove that every integer-infeasible system has a finite
derivation, or simulate cuts by arbitrary divisors.
-/

namespace Apeiron
namespace Decider

variable {n : ℕ}

/-- A valid inference from the input system and rows already derived.

    (raw : List (Line n))                    [the input system]
    (rs : List (CRow n))                     [rows already derived]
    CRow n → Type                            [an inference indexed by its result row] -/
inductive CuttingPlaneInference (raw : List (Line n)) (rs : List (CRow n)) : CRow n → Type
  /-- Read a row from the input system. -/
  | input (k : Fin raw.length) : CuttingPlaneInference raw rs (certRaw (raw.get k))
  /-- Add positive integer multiples of two earlier rows. -/
  | combine (i j : Fin rs.length) (a b : ℤ) (ha : 1 ≤ a) (hb : 1 ≤ b) :
      CuttingPlaneInference raw rs (crCombo a b (rs.get i) (rs.get j))
  /-- Negate an earlier row. -/
  | negate (i : Fin rs.length) : CuttingPlaneInference raw rs (crNeg (rs.get i))
  /-- Intersect two earlier rows with identical coefficients. -/
  | intersect (i j : Fin rs.length) (hc : (rs.get i).coeff = (rs.get j).coeff) :
      CuttingPlaneInference raw rs (crMeet (rs.get i) (rs.get j))
  /-- Normalize the sign and divide by the coefficient gcd, rounding bounds inward. -/
  | cut (i : Fin rs.length) : CuttingPlaneInference raw rs (crTidy (rs.get i))

namespace CuttingPlaneInference

/-- Erase an inference's membership and arithmetic proofs to a certificate operation.

    {raw : List (Line n)} {rs : List (CRow n)} {x : CRow n} [input, earlier rows, result]
    CuttingPlaneInference raw rs x → CertOp                 [the emitted operation] -/
def compile {raw : List (Line n)} {rs : List (CRow n)} {x : CRow n} :
    CuttingPlaneInference raw rs x → CertOp
  | .input k => .load k
  | .combine i j a b _ _ => .combo i j a b
  | .negate i => .neg i
  | .intersect i j _ => .meet i j
  | .cut i => .tidy i

/-- The compiled inference appends exactly its indexed result.

    {raw : List (Line n)} {rs : List (CRow n)} {x : CRow n} [input, earlier rows, result]
    (s : CuttingPlaneInference raw rs x)                    [a valid inference]
    certStep raw rs s.compile = some (rs ++ [x])           [replay appends its result] -/
theorem compile_correct {raw : List (Line n)} {rs : List (CRow n)} {x : CRow n}
    (s : CuttingPlaneInference raw rs x) :
    certStep raw rs s.compile = some (rs ++ [x]) := by
  cases s <;> simp_all [compile, certStep, List.get_eq_getElem]
  assumption

/-- A compiled inference preserves the checker's result on every continuation.

    {raw : List (Line n)} {rs : List (CRow n)} {x : CRow n} [input, earlier rows, result]
    (s : CuttingPlaneInference raw rs x)                    [a valid inference]
    (tail : List CertOp)                                    [the remaining certificate]
    checkCert raw rs (s.compile :: tail)                   [check after the compiled step]
      = checkCert raw (rs ++ [x]) tail                     [check after appending its result] -/
theorem compile_continuation {raw : List (Line n)} {rs : List (CRow n)} {x : CRow n}
    (s : CuttingPlaneInference raw rs x) (tail : List CertOp) :
    checkCert raw rs (s.compile :: tail) = checkCert raw (rs ++ [x]) tail := by
  have hs := s.compile_correct
  cases s <;> simp only [compile] at hs ⊢ <;> simp only [checkCert, hs]

end CuttingPlaneInference

/-- A finite sequence of valid inferences, starting with no derived rows.

    (raw : List (Line n))                    [the input system]
    List (CRow n) → Type                     [a derivation indexed by all its emitted rows] -/
inductive CuttingPlaneDerivation (raw : List (Line n)) : List (CRow n) → Type
  /-- The empty derivation starts with no derived rows. -/
  | nil : CuttingPlaneDerivation raw []
  /-- Append one inference using only earlier rows. -/
  | snoc {rs : List (CRow n)} {x : CRow n} (d : CuttingPlaneDerivation raw rs)
      (s : CuttingPlaneInference raw rs x) : CuttingPlaneDerivation raw (rs ++ [x])

namespace CuttingPlaneDerivation

/-- Compile a finite derivation in emission order.

    {raw : List (Line n)} {rs : List (CRow n)} [the input and all derived rows]
    CuttingPlaneDerivation raw rs → List CertOp [the replay operations, without a close] -/
def compile {raw : List (Line n)} {rs : List (CRow n)} :
    CuttingPlaneDerivation raw rs → List CertOp
  | .nil => []
  | .snoc d s => d.compile ++ [s.compile]

/-- Compilation supplies exactly the derived rows to any certificate continuation.

    {raw : List (Line n)} {rs : List (CRow n)} [the input and all derived rows]
    (d : CuttingPlaneDerivation raw rs)       [a finite derivation]
    (tail : List CertOp)                      [the remaining certificate]
    checkCert raw [] (d.compile ++ tail)     [replay the derivation before the continuation]
      = checkCert raw rs tail               [the continuation sees the derived rows] -/
theorem compile_correct {raw : List (Line n)} {rs : List (CRow n)}
    (d : CuttingPlaneDerivation raw rs) (tail : List CertOp) :
    checkCert raw [] (d.compile ++ tail) = checkCert raw rs tail := by
  induction d generalizing tail with
  | nil => rfl
  | snoc d s ih =>
      simp only [compile, List.append_assoc, List.singleton_append]
      rw [ih, s.compile_continuation]

/-- A derived crossing compiles to an accepted certificate.

    {raw : List (Line n)} {rs : List (CRow n)} [the input and all derived rows]
    (d : CuttingPlaneDerivation raw rs)       [a finite derivation]
    (i : Fin rs.length)                      [an index of a derived row]
    (h : crImpossible (rs.get i) = true)     [the row's bounds cross]
    checkCert raw []                         [the checker starts with no replay rows]
      (d.compile ++ [.close i]) = true       [and accepts the compiled closed derivation] -/
theorem compile_accepts {raw : List (Line n)} {rs : List (CRow n)}
    (d : CuttingPlaneDerivation raw rs) (i : Fin rs.length)
    (h : crImpossible (rs.get i) = true) :
    checkCert raw [] (d.compile ++ [.close i]) = true := by
  rw [d.compile_correct]
  simpa [checkCert, List.get_eq_getElem] using h

/-- A derived crossing refutes every integer assignment to the input system.

    {raw : List (Line n)} {rs : List (CRow n)} [the input and all derived rows]
    (d : CuttingPlaneDerivation raw rs)       [a finite derivation]
    (i : Fin rs.length)                      [an index of a derived row]
    (h : crImpossible (rs.get i) = true)     [the row's bounds cross]
    ¬ ∃ ρ : Fin n → ℤ, SatLines ρ raw       [no integer assignment satisfies the input] -/
theorem refutes {raw : List (Line n)} {rs : List (CRow n)}
    (d : CuttingPlaneDerivation raw rs) (i : Fin rs.length)
    (h : crImpossible (rs.get i) = true) : ¬ ∃ ρ : Fin n → ℤ, SatLines ρ raw :=
  checkCert_sound raw _ (d.compile_accepts i h)

end CuttingPlaneDerivation

/-- The integer-infeasible equation used by the compiler witness.

    Line 1                                   [one equation over one integer column] -/
private def oddEquality : Line 1 := .eq ⟨-1, fun _ => 2⟩

/-- Load the odd equality and apply its gcd cut.

    CuttingPlaneDerivation [oddEquality]     [a derivation from the odd equality]
      [certRaw oddEquality,                 [the initial row]
       crTidy (certRaw oddEquality)]         [the row after the gcd cut] -/
private def oddDerivation : CuttingPlaneDerivation [oddEquality]
    [certRaw oddEquality, crTidy (certRaw oddEquality)] :=
  .snoc (.snoc .nil (.input ⟨0, by decide⟩)) (.cut ⟨0, by decide⟩)

#guard oddDerivation.compile = [.load 0, .tidy 0]
#guard checkCert [oddEquality] [] (oddDerivation.compile ++ [.close 1])
#guard !checkCert [oddEquality] [] (oddDerivation.compile ++ [.close 0])
#guard !checkCert [oddEquality] [] (oddDerivation.compile ++ [.close 2])

/-- The witness inhabits the joint hypotheses of the refutation theorem.

    ¬ ∃ ρ : Fin 1 → ℤ, SatLines ρ [oddEquality] [no integer satisfies the odd equality] -/
example : ¬ ∃ ρ : Fin 1 → ℤ, SatLines ρ [oddEquality] :=
  oddDerivation.refutes ⟨1, by decide⟩ (by decide)

end Decider
end Apeiron
