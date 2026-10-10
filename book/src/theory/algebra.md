# The algebra

## Definition

A cell is an ordered pair `(a, b)` over a commutative ring `R`, read as `a + b·x` in
`R[x]/(x² − σ)`. The dial `σ` is an element of `R`. Multiplication follows from the relation
`x² = σ`.

$$(a_1 + b_1 x)(a_2 + b_2 x) = (a_1 a_2 + \sigma\, b_1 b_2) + (a_1 b_2 + b_1 a_2)\,x$$

- **say:** the first slot is `a₁a₂` plus `σ` times `b₁b₂`; the second slot is the cross terms.
- **means:** ordinary polynomial multiplication, with every `x²` replaced by `σ`.
- **vars:** `a₁, b₁, a₂, b₂` are elements of `R`; `σ` is the dial, one element of `R` fixed per computation.
- **apeiron:** `Apeiron.Trichotomy.AnyRing.ringMul`; Rust `Cell::mul`.
- **physics:** none. This is the quotient ring `R[x]/(x² − σ)` (`Apeiron.Trichotomy.mul_eq_quotient_mul`).

`σ` enters at one place, the `σ·b₁b₂` term. The product is commutative and associative for every
`σ` and every commutative `R` (`Apeiron.Trichotomy.AnyRing.ringMul_comm`,
`Apeiron.Trichotomy.AnyRing.ringMul_assoc`). The identity is `(1, 0)`.

## Norm form

$$N(a + bx) = a^2 - \sigma\, b^2$$

- **say:** `a` squared minus `σ` times `b` squared.
- **means:** the size of a cell. It is the product of the cell and its conjugate `a − bx`.
- **vars:** `a, b` are the slots of the cell; `σ` is the dial.
- **apeiron:** `Apeiron.Trichotomy.AnyRing.ringNormForm`; Rust `Cell::norm_form`.
- **physics:** the principal symbol of a second-order operator (next section).

The norm form is the quadratic form that Gauss studied in the *Disquisitiones Arithmeticae*
(1801). It is multiplicative:

$$N(z\,w) = N(z)\,N(w)$$

- **say:** the norm of a product is the product of the norms.
- **means:** multiplying cells multiplies their sizes. Euclid, *Elements* II.5, is the case `σ = 1`.
- **vars:** `z, w` are cells.
- **apeiron:** `Apeiron.Trichotomy.AnyRing.ringNormForm_mul`.
- **physics:** none.

The identity is polynomial with integer coefficients, so it holds in every commutative ring.

## Inverses

$$z \text{ has an inverse} \iff N(z) \text{ is a unit of } R$$

- **say:** a cell can be divided by exactly when its norm form can.
- **means:** the inverse is the conjugate scaled by the inverse of the norm, `z⁻¹ = (a, −b)/N(z)`. No square root and no ordering of `R` is used.
- **vars:** `z` is a cell; "unit of `R`" means an element with a multiplicative inverse in `R`.
- **apeiron:** `Apeiron.Trichotomy.AnyRing.exists_inverse_iff_isUnit_ringNormForm`; over `ℝ`, `Apeiron.Trichotomy.isUnit_iff_normForm_ne_zero`.
- **physics:** none.

Over `ℝ` the units of `R` are the non-zero reals, so a cell is invertible off the cone `N = 0`.
Over `Z/2^64` the units are the odd words, so a cell is invertible when its norm form is odd.
One theorem, two readings.

## Three faces

Over `ℝ`, `σ` can be scaled to `−1`, `0` or `+1` without changing the algebra.

$$\sigma \in \{-1,\ 0,\ +1\}$$

- **say:** every two-dimensional unital commutative associative real algebra is `ℝ[x]/(x² − σ)` for one of three `σ`.
- **means:** the three-way split is the complete list. Complete the square, then rescale the generator by `1/√|σ|`.
- **vars:** `σ` is the dial; the sign is the only invariant.
- **apeiron:** `Apeiron.Trichotomy.classification`.
- **physics:** the classification of second-order linear operators into elliptic, parabolic and hyperbolic.

| `σ` | name | algebra | attribution |
|---|---|---|---|
| `< 0` | elliptic | complex numbers | |
| `= 0` | parabolic | dual numbers | Clifford (1873) |
| `> 0` | hyperbolic | split-complex numbers (`j² = +1`, `j ≠ ±1`) | Cockle (1848) |

The split-complex numbers here are Cockle's. They are not the complex numbers with a renamed unit:
there are zero divisors, a nontrivial idempotent pair and a null cone. [The three faces](faces.md) states
each difference.

## The same classification for differential operators

A second-order linear operator in two variables has principal part

$$A\,u_{xx} + 2B\,u_{xy} + C\,u_{yy}$$

- **say:** `A` times the second `x` derivative, plus `2B` times the mixed derivative, plus `C` times the second `y` derivative.
- **means:** the terms that decide the operator's type. Lower-order terms do not.
- **vars:** `u(x, y)` is the unknown; `A, B, C` are coefficients.
- **apeiron:** none. This is an `anchor`: the reading of the dial onto a domain, not a theorem about operators.
- **physics:** the general second-order operator.

The type is set by the sign of the discriminant:

$$\sigma = B^2 - A\,C$$

- **say:** `B` squared minus `A` times `C`.
- **means:** `σ < 0` is elliptic (Laplace, `u_xx + u_yy`), `σ = 0` is parabolic (heat, `u_t − u_xx`), `σ > 0` is hyperbolic (wave, `u_tt − u_xx`).
- **vars:** `A, B, C` as above.
- **apeiron:** none. `anchor`.
- **physics:** the characteristic equation `A λ² + 2B λ + C = 0` has roots `λ = (−B ± √σ)/A`. Two real roots (hyperbolic) are the two characteristic directions. In the hyperbolic case the null coordinates are the two channels of the split-complex numbers.

The algebra of the dial and the type of the operator are decided by the same quantity, so the
channel count of an algebra is the characteristic count of the operator: 0, 1 or 2
(`Sigma::channels`).

## Worked by hand

Take `z = 3 + 2x` and `w = 1 + 5x`, so `(a₁, b₁) = (3, 2)` and `(a₂, b₂) = (1, 5)`.

At `σ = +1`:

- first slot: `3·1 + 1·2·5 = 13`
- second slot: `3·5 + 2·1 = 17`
- product `(13, 17)`

Norms at `σ = +1`: `N(3, 2) = 9 − 4 = 5`, `N(1, 5) = 1 − 25 = −24`, `N(13, 17) = 169 − 289 = −120`.
Check: `5 · (−24) = −120`.

At `σ = −1` the same inputs give `(3 − 10, 15 + 2) = (−7, 17)`, norms `13`, `26` and `338`, and
`13 · 26 = 338`.

```rust
extern crate apeiron;
use apeiron::{Cell, Sigma};

let x = Cell::<i64>::of(3, 2);
let y = Cell::<i64>::of(1, 5);

let h = x.mul(Sigma::Hyperbolic, y);
assert_eq!((h.a, h.b), (13, 17));
assert_eq!(x.norm_form(Sigma::Hyperbolic), 5);
assert_eq!(y.norm_form(Sigma::Hyperbolic), -24);
assert_eq!(h.norm_form(Sigma::Hyperbolic), -120);

let e = x.mul(Sigma::Elliptic, y);
assert_eq!((e.a, e.b), (-7, 17));
assert_eq!(e.norm_form(Sigma::Elliptic), 13 * 26);
```
