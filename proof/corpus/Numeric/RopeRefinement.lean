/-
Aggregator for the fixed-point `sin`/`cos` refinement. The original 3524-line body is split by
section into `Numeric.Rope.Part1` .. `Numeric.Rope.Part4`; every declaration, statement, constant,
namespace and declaration order is unchanged. Load order is the linear chain
`Part1 <- Part2 <- Part3 <- Part4`.
-/
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Numeric.Scale
import Numeric.Rope.Part1
import Numeric.Rope.Part2
import Numeric.Rope.Part3
import Numeric.Rope.Part4
