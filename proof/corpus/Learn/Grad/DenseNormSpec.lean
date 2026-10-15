import Mathlib

/-
# The `ℤ` specification of the extracted `src/learn/grad/dense.rs` passes

This module states the `RMSNorm`/`SwiGLU` forward and backward maps of the W19 dense backward
file as exact integer formulas over `ℤ` and `List ℤ`, independently of the extracted body
(`ApeironWord.lean`, module `learn_port.dense`). The Rust values are Q32.32 `i64`; every product
of two Q32.32 words is rounded once by `round32`, and the division by the row length uses the
engine's signed round-to-nearest-ties-away-from-zero `div_round`.

The two integer divisions the Rust performs are distinguished here because they disagree on
negative arguments:

* `rdiv a b = a.tdiv b` is Rust's `/` on signed integers (truncation toward zero);
* `rshr a n = a / 2^n` is Rust's `>>` on a signed integer (arithmetic shift, i.e. `ℤ` floor
  division by a power of two).

Nothing in this file mentions the extraction; the refinement in `Learn.Grad.DenseNorm` links the
extracted bodies to these definitions.
-/

namespace apeiron_norm

/-! ## The rounding primitives -/

/-- `2^32`, the Q32.32 unit. -/
def q32 : ℤ := 2 ^ (32 : ℕ)

/-- Rust's `/` on signed integers: truncation toward zero (`Int.tdiv`). -/
def rdiv (a b : ℤ) : ℤ := a.tdiv b

/-- Rust's `>>` on a signed integer: arithmetic shift, i.e. `ℤ` floor division by `2^n`. -/
def rshr (a : ℤ) (n : ℕ) : ℤ := a / 2 ^ n

/-- The extracted `dense::round32` value: `(v + 2³¹) >> 32`, single half-biased rounding of a
Q64.64 `i128` product to Q32.32. -/
def round32 (v : ℤ) : ℤ := rshr (v + 2 ^ (31 : ℕ)) 32

/-- The Q32.32 product chain of `rmsnorm_bwd`'s `t`: `(v + 2⁶³) >> 64`, one rounding of a
Q96.32 `i128` intermediate to Q32.32. -/
def round64 (v : ℤ) : ℤ := rshr (v + 2 ^ (63 : ℕ)) 64

/-- The extracted `dense::mul_round`: `round32 (a · b)`. -/
def mulRound (a b : ℤ) : ℤ := round32 (a * b)

/-- The extracted `dense::div_round`: signed round-to-nearest with ties away from zero,
`(v + ⌊d/2⌋) / d` for `v ≥ 0` and `(v − ⌊d/2⌋) / d` for `v < 0`, `d ≥ 1`. -/
def divRound (v d : ℤ) : ℤ :=
  if 0 ≤ v then rdiv (v + rdiv d 2) d else rdiv (v - rdiv d 2) d

/-- The engine's `value::div_round_nearest` on the non-negative mean numerator:
`(n + ⌊d/2⌋) / d`. -/
def meanRound (n d : ℤ) : ℤ := rdiv (n + rdiv d 2) d

/-! ## `RMSNorm` -/

/-- The `RMSNorm` argument `mean(x²) + eps` in Q32.32, the value `fixed_rsqrt` is called on. -/
def rmsMean (x : List ℤ) (eps : ℤ) : ℤ :=
  meanRound ((x.map (fun xi => xi * xi)).sum) ((x.length : ℤ) * q32) + eps

/-- The normalised row `round(x_i · r / 2³²)` of `engine::rms_norm_fixed`, for the Q32.32 inverse
root `r` its `fixed_rsqrt` call returns. -/
def rmsNorm (x : List ℤ) (r : ℤ) : List ℤ := x.map (fun xi => round32 (xi * r))

/-- `rmsnorm_fwd`: the gain applied to the normalised row. -/
def rmsFwd (gamma x : List ℤ) (r : ℤ) : List ℤ := List.zipWith mulRound gamma (rmsNorm x r)

/-- `a_j = round(g_j · dy_j / 2³²)`, the first accumulator of `rmsnorm_bwd`. -/
def bwdA (g dy : ℤ) : ℤ := mulRound g dy

/-- `S = round32 (Σ_j a_j · x_j)`, the `i128` sum's single rounding. -/
def bwdS (x a : List ℤ) : ℤ := round32 ((List.zipWith (fun xi ai => ai * xi) x a).sum)

/-- `p = round(r · a_i / 2³²)`. -/
def bwdP (r a : ℤ) : ℤ := mulRound r a

/-- `t = round64 (x_i · r² · r)`, the unrounded `r³` product chain. -/
def bwdT (x r r2 : ℤ) : ℤ := round64 (x * r2 * r)

/-- `q = round(round(t · S / 2³²) / n)`, the `r³` correction. -/
def bwdQ (t s n : ℤ) : ℤ := divRound (mulRound t s) n

/-- `dx_i = p − q`. -/
def bwdDx (r a t s n : ℤ) : ℤ := bwdP r a - bwdQ t s n

/-- `dgamma_i = round(round(dy_i · x_i / 2³²) · r / 2³²)`. -/
def bwdDg (dy x r : ℤ) : ℤ := mulRound (mulRound dy x) r

/-! ## `SwiGLU` -/

/-- `silu'(g) = round(s · (2³² + round(g·(2³² − s)/2³²)) / 2³²)` for `s = σ(g)`. -/
def siluDeriv (g s : ℤ) : ℤ := round32 (s * (q32 + mulRound g (q32 - s)))

/-- `swiglu_fwd`: `m_i = round(silu(g_i) · u_i / 2³²)`. -/
def swigluFwd (sv u : List ℤ) : List ℤ := List.zipWith mulRound sv u

/-- `dgate = round(round(dy · u / 2³²) · silu'(g) / 2³²)`. -/
def swigluDgate (dy u ds : ℤ) : ℤ := mulRound (mulRound dy u) ds

/-- `dup = round(dy · silu(g) / 2³²)`. -/
def swigluDup (dy s : ℤ) : ℤ := mulRound dy s

/-! ## Evaluation

Concrete values are `norm_num` facts about these definitions and are stated where they are used
(`Learn.Grad.DenseNorm`, the witness section). -/

end apeiron_norm
