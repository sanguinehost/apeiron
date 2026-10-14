/-
Aggregator for the fixed-point `exp` refinement proofs (`APEIRON-SIZE-E1`).

Declarations, statements, namespace and order are unchanged from the pre-split
`Numeric/Engine/Exp.lean`; the body lives in the linearly chained part
modules `Numeric.Exp.Part1` .. `Numeric.Exp.Part4`.
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.ExponentialBounds
import ApeironWord
import Numeric.Scale
import Numeric.Engine.Exp.Part1
import Numeric.Engine.Exp.Part2
import Numeric.Engine.Exp.Part3
import Numeric.Engine.Exp.Part4
