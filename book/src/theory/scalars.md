# Scalars

A cell is built over a commutative ring. The ring decides what the cell can hold. The crate has
three scalar types, and the algebra of the previous chapters is the same over each.

| scalar | ring | role |
|---|---|---|
| `i64` | `Z/2^64` (wrapping) | the fast word lattice; the type the table and the engine use |
| `Dyadic` | `m / 2^e`, `m: i128`, `e: u32`, exact when the result fits | holds the idempotents `(1 ± j)/2` |
| `i8` | `Z/2^8` | a narrow ring for exhaustive ring-law tests (`tests/the_ring_laws.rs`) |

## `i64` is `Z/2^64`

`Ring for i64` uses `wrapping_add`, `wrapping_sub` and `wrapping_mul`. Arithmetic wraps modulo
`2^64`, so the value is an element of `Z/2^64` and overflow does not trap.

```rust
extern crate apeiron;
use apeiron::{Cell, Sigma};

let big = Cell::<i64>::of(i64::MAX, 0);
let two = Cell::<i64>::of(2, 0);
let p = big.mul(Sigma::Parabolic, two);
// (2^63 - 1) * 2 = 2^64 - 2, which is -2 modulo 2^64.
assert_eq!((p.a, p.b), (-2, 0));
```

The algebra's laws are polynomial identities with integer coefficients, so they hold in every
commutative ring. The ring-universal statements live in `Apeiron.Trichotomy.AnyRing` and are
proved once for an arbitrary commutative ring `R`. `Z/2^64` is an instance of that theory.

What changes between `ℝ` and `Z/2^64` is the meaning of "can be divided by". Two is not
invertible in `Z/2^64`, and the sign of `σ` does not exist. The units are the odd words.

$$z \text{ is invertible in } (Z/2^{64})[x]/(x^2-\sigma) \iff N(z) \text{ is odd}$$

- **say:** a word-valued cell has an inverse exactly when its norm form is an odd word.
- **means:** the inverse is the conjugate times the modular inverse of the norm, which exists for odd words.
- **vars:** `z` is a cell with `i64` slots; `N` is the norm form, computed with wrapping arithmetic.
- **apeiron:** `Apeiron.Trichotomy.AnyRing.exists_inverse_iff_isUnit_ringNormForm`, instantiated at `ZMod (2^64)`.
- **physics:** none.

```rust
extern crate apeiron;
use apeiron::{Cell, Sigma};

// z = 3 + 2x at sigma = 1 has N = 5, which is odd, so z is invertible.
let z = Cell::<i64>::of(3, 2);
let s = Sigma::Hyperbolic;
let n = z.norm_form(s);
assert_eq!(n, 5);

// Newton iteration for n^-1 modulo 2^64: each step doubles the correct bits.
let mut inv = n;
for _ in 0..6 {
    inv = inv.wrapping_mul(2i64.wrapping_sub(n.wrapping_mul(inv)));
}
assert_eq!(n.wrapping_mul(inv), 1);

// The inverse is the conjugate (a, -b) scaled by n^-1.
let w = Cell::<i64>::of(inv.wrapping_mul(z.a), inv.wrapping_mul(z.b).wrapping_neg());
let one = z.mul(s, w);
assert_eq!((one.a, one.b), (1, 0));
```

### Idempotents in `Z/2^n`

The classification by sign does not exist in `Z/2^n`. What replaces it is narrower. In `Z/2^n`
every odd element is a unit, `2a − 1` is odd for every `a`, and one of `a` and `a − 1` is odd.
So the only idempotents of `Z/2^n` are `0` and `1`, and the only cell idempotents are the two
that every ring has.

$$z\cdot z = z \implies z \in \{(0,0),\ (1,0)\}\quad \text{in } (Z/2^{n})[x]/(x^2-\sigma)$$

- **say:** over a word, the only cell that squares to itself is zero or one.
- **means:** the pair `(1 ± j)/2` does not exist over `i64`, because it needs a half. The theorem assumes that `2a − 1` is a unit and that one of `a`, `a − 1` is a unit; both hold for every `a` in `Z/2^n`, so the conclusion is unconditional there and holds at every `σ`.
- **vars:** `z = (a, b)` is a cell with word slots; `σ` is any word.
- **apeiron:** `Apeiron.Trichotomy.AnyRing.ringIdempotent_trivial`.
- **physics:** none.

The crate states this boundary in the types. `Cell::idem_plus`, `Cell::idem_minus` and
`Cell::of_halves` require `R: Halve`. `i64` does not implement `Halve`, so
`Cell::<i64>::idem_plus()` does not compile. The refinement of the executed word body is in
[Evidence and grades](../verification/evidence.md).

## `Dyadic` holds the idempotents

`Dyadic` is `m / 2^e`: an integer mantissa and a power-of-two scale. It has exact halves, so
`(1 ± j)/2` lives in it.

```rust
extern crate apeiron;
use apeiron::scalar::Dyadic;
use apeiron::{Cell, Sigma};

let e = Cell::<Dyadic>::idem_plus();
let s = Sigma::Hyperbolic;
// e * e = e: an idempotent that is neither zero nor one.
assert_eq!(e.mul(s, e), e);
assert!(e != Cell::<Dyadic>::zero());
assert!(e != Cell::<Dyadic>::one());

// The complementary idempotent annihilates it, and the pair sums to one.
let f = Cell::<Dyadic>::idem_minus();
assert_eq!(e.mul(s, f), Cell::<Dyadic>::zero());
assert_eq!(e.add(f), Cell::<Dyadic>::one());
```

`Dyadic` is not a total ring. The mantissa is `i128` and the scale is `u32`, so aligning,
multiplying or growing the scale can exceed the representation. The `checked_*` methods return
`None` there. The `Ring` and `Halve` implementations panic on an unrepresentable result and do
not drop bits.

```rust
extern crate apeiron;
use apeiron::scalar::Dyadic;

let half = Dyadic::new(1, 1);
assert_eq!(half.checked_add(half), Some(Dyadic::whole(1)));
// 2^100 * 2^100 does not fit an i128 mantissa.
let huge = Dyadic::new(1i128 << 100, 0);
assert_eq!(huge.checked_mul(huge), None);
```

## Which scalar to use

| need | scalar |
|---|---|
| speed, hashing, the engine, the executed table | `i64` |
| the null basis and its idempotents, exact | `Dyadic` |
| an exhaustive check at narrow width | `i8` |

An operation that can lose bits does not have an infallible signature over a type that cannot
hold the result. That is the reason `Dyadic` has `checked_*` methods and `i64` is stated as
`Z/2^64`: the wrap is the specification, not a defect.
