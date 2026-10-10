# The three faces

The dial `σ` selects a face. Each face is a different algebra with different units, zero
divisors, idempotents and channels. A channel is a ring homomorphism from the cell to the base
ring `R`: a lane that multiplies componentwise.

| | elliptic `σ < 0` | parabolic `σ = 0` | hyperbolic `σ > 0` |
|---|---|---|---|
| algebra | complex numbers | dual numbers | split-complex numbers |
| norm form `a² − σb²` | positive definite | `a²` | indefinite |
| non-zero zero divisors | none | `(0, b)` | `(r, ±1)` with `r² = σ` |
| idempotents | `0`, `1` | `0`, `1` | `0`, `1` and a complementary pair |
| channels (`Sigma::channels`) | 0 | 1 | 2 |
| can discard (`Sigma::discards`) | no | no | yes |

## Elliptic

Below zero the norm form is `a² + |σ|b²`, which vanishes only at the origin. So every non-zero
cell is invertible over `ℝ`, there are no zero divisors, and the only idempotents are `0` and `1`.

$$\sigma < 0 \implies \big(z\,w = 0 \Rightarrow z = 0 \ \lor\ w = 0\big)$$

- **say:** below zero, a product is zero only if a factor is zero.
- **means:** nothing can be thrown away. Every operation that loses information needs a non-zero cell that multiplies to zero.
- **vars:** `z, w` are cells.
- **apeiron:** `Apeiron.Trichotomy.noZeroDivisors_iff_disc_neg`; no channel exists, `Apeiron.Trichotomy.no_channel_of_disc_neg`.
- **physics:** Laplace's equation has no real characteristics.

## Parabolic

At `σ = 0` the norm form is `a²` and the second slot is nilpotent: `(0, 1)·(0, 1) = (0, 0)`. The
only channel is the first slot.

$$\varphi \text{ is a channel} \iff \forall z,\ \varphi(z) = a$$

- **say:** a channel at `σ = 0` is exactly the first projection.
- **means:** the second slot is carried through products but cannot be read as an independent lane.
- **vars:** `φ` is a function from cells to `R`; `a` is the first slot of `z`.
- **apeiron:** `Apeiron.Trichotomy.channel_iff_fst_of_disc_zero`.
- **physics:** the heat equation has one characteristic direction, and `x` is nilpotent as in forward-mode differentiation.

## Hyperbolic

Above zero the norm form factors into two lanes. For `σ = 1` the lanes are the null coordinates
`upper = a + b` and `lower = a − b`.

$$N(z) = \mathrm{upper}(z)\cdot \mathrm{lower}(z), \qquad \mathrm{upper}(zw) = \mathrm{upper}(z)\,\mathrm{upper}(w)$$

- **say:** the norm form is the product of the two lanes, and each lane multiplies on its own.
- **means:** a cell is on the null cone exactly when one lane is zero. Multiplying two cells multiplies lane by lane, so the two lanes can be accumulated separately and combined at the end.
- **vars:** `z, w` are cells; `upper` and `lower` are the two channels at `σ = 1`.
- **apeiron:** `Apeiron.Trichotomy.normForm_eq_upper_mul_lower`, `Apeiron.Trichotomy.upper_mul`, `Apeiron.Trichotomy.lower_mul`.
- **physics:** null (light-cone) coordinates `x ± t` for the wave equation.

The idempotent pair is `(½, ½)` and `(½, −½)` at `σ = 1`. Each projects onto one lane and sends the
other to zero. Over `ℝ`, a nontrivial idempotent exists exactly above zero.

$$(\exists\, e\ \text{idempotent},\ e \ne 0,\ e \ne 1) \iff \sigma > 0$$

- **say:** a nontrivial idempotent exists if and only if `σ` is positive.
- **means:** the elliptic and parabolic faces have no projection other than the identity and zero.
- **vars:** `e` is a cell with `e·e = e`.
- **apeiron:** `Apeiron.Trichotomy.exists_nontrivial_idempotent_iff_disc_pos`, `Apeiron.Trichotomy.idempotent_iff_trivial_of_disc_nonpos`.
- **physics:** the two idempotents split a wave into right-moving and left-moving parts.

Discarding is a projection that keeps one thing and kills another. It exists only above zero.

$$\text{a discarding projection exists} \iff \sigma > 0$$

- **say:** a projection that keeps one lane and destroys the other exists if and only if `σ` is positive.
- **means:** every lossy operation in a table (collapsing a key to a fold, culling, demoting detail) is hyperbolic. The elliptic and parabolic faces refuse it by construction, and so does the Rust API: `upper` and `lower` return `None` off the hyperbolic face.
- **vars:** `σ` is the dial.
- **apeiron:** `Apeiron.Trichotomy.discards_iff_disc_pos`.
- **physics:** none.

## Order-free accumulation

A fold of cells is independent of order on every face, because the product is commutative and
associative (`Apeiron.Trichotomy.AnyRing.ringMul_comm`, `Apeiron.Trichotomy.AnyRing.ringMul_assoc`).
That licenses splitting an accumulation across threads and combining at a sync point. The dial
changes what a table may discard. It does not change the merge.

The Lean statement `Apeiron.Trichotomy.fold_perm` assumes `0 ≤ σ`, because it is proved
through the channels. The Rust test `tests/three_and_one.rs` checks all 120 permutations of five
cells on each of the three faces, including elliptic.

## Worked by hand

```rust
extern crate apeiron;
use apeiron::{Cell, Sigma};

// Elliptic: i * i = -1.
let i = Cell::<i64>::of(0, 1);
let m1 = i.mul(Sigma::Elliptic, i);
assert_eq!((m1.a, m1.b), (-1, 0));

// Parabolic: the second slot squares to zero.
let eps = Cell::<i64>::of(0, 1);
let z = eps.mul(Sigma::Parabolic, eps);
assert_eq!((z.a, z.b), (0, 0));

// Hyperbolic: (1, 1) * (1, -1) = 0, two non-zero cells with a zero product.
let p = Cell::<i64>::of(1, 1);
let q = Cell::<i64>::of(1, -1);
let zero = p.mul(Sigma::Hyperbolic, q);
assert_eq!((zero.a, zero.b), (0, 0));

// Hyperbolic channels: a + b and a - b multiply lane by lane.
let x = Cell::<i64>::of(3, 2);
let y = Cell::<i64>::of(1, 5);
let xy = x.mul(Sigma::Hyperbolic, y);
let s = Sigma::Hyperbolic;
assert_eq!(xy.upper(s), Some(x.upper(s).unwrap() * y.upper(s).unwrap()));
assert_eq!(xy.lower(s), Some(x.lower(s).unwrap() * y.lower(s).unwrap()));
assert_eq!(xy.norm_form(s), xy.upper(s).unwrap() * xy.lower(s).unwrap());

// The other faces have no such lanes.
assert_eq!(x.upper(Sigma::Elliptic), None);
assert_eq!(x.upper(Sigma::Parabolic), None);
assert_eq!(x.channel(Sigma::Parabolic), Some(3));
```

By hand at `σ = 1`: `x = (3, 2)` has `upper = 5` and `lower = 1`; `y = (1, 5)` has `upper = 6`
and `lower = −4`. The product `(13, 17)` has `upper = 30 = 5·6` and `lower = −4 = 1·(−4)`.
