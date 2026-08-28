# apeiron

`R[x]/(x^2 - sigma)`, and the table that falls out of it.

One ordered pair, one real dial. The sign of `sigma` is the only thing that varies, and it decides
what a face can do:

| sigma | face | channels | zero divisors | discard |
|---|---|---|---|---|
| `< 0` | elliptic | 0 | none (a field) | no |
| `= 0` | parabolic | 1 | nilpotent line | no |
| `> 0` | hyperbolic | 2 | null cone | **yes** |

Every row is a theorem in `../sanguine-apeiron/proof/Apeiron/Trichotomy/`, cited by name at each
definition, and every row is a test in `tests/the_dial_is_real.rs`.

## Why this crate exists

`hashtrinity` names three storage faces after the trichotomy and branches on no discriminant
anywhere: `sigma` appears in no signature, no type and no test, so nothing there would fail if the
trichotomy were wrong, absent, or replaced with three other words. It is vocabulary.

Here the dial is a value the operations take. The channel accessors return `None` on faces that
have no channel, `discards()` is false off the hyperbolic face, and the fold's order-freedom is
stated as a property of `sigma` rather than assumed. What the algebra forbids is unavailable
rather than merely undocumented.

## The scalar is a choice, not a limit

`Cell<R>` is generic over the ring its slots live in, because which ring you pick decides what the
cell can hold:

- **`i64`** -- the fast lattice. Everything structural holds: commutativity, associativity, the
  unit, multiplicativity of the norm, the channel count, componentwise multiplication in the null
  basis, order-free folding above zero, zero divisors at and above zero.
- **`Dyadic`** -- an exact `m / 2^e`, two integers, the same shape a float has without the
  rounding. This is where the nontrivial idempotents `(1 +/- j)/2` live, so
  `exists_nontrivial_idempotent_iff_disc_pos` holds here rather than being written off.

The halves were never an obstruction in the algebra, only in the container. In the null basis the
idempotents are the unit vectors `(1, 0)` and `(0, 1)` -- the fraction is an artefact of the
coordinates.

The split is in the bounds. `Ring` is what the algebra needs; `Halve` is the extra thing the
idempotents need; `i64` implements the first and not the second, so `Cell::<i64>::idem_plus()`
does not compile. That is the theorem's scope enforced by the type rather than described in a
comment.

## The coordinate fold, which is three folds

Injectivity is required on one face, irrelevant on the second and impossible on the third, so
asking for "the" collision rate asks three questions at once:

| face | wants | `coord.rs` gives it |
|---|---|---|
| elliptic (**B**) | the address determines the key | `Packed` -- concatenation, exactly invertible, only for keys that fit in 64 bits |
| parabolic (**B/U**) | spreading; the key is carried and compared | `Coord` -- multiply-add, judged on distribution |
| hyperbolic (**NB**) | a discard; the key is gone | `Coord` -- same fold, and losing information is the intent |

Two things this corrects. hashtrinity's `a.wrapping_mul(PHI) ^ b` leaves the ring for `GF(2)` and
collides on 49% of a signed pair grid and 72% of a signed triple grid -- bad *distribution*, on a
face where distribution is the whole job. And multiply-add, which fixes that, is **not** a
bijection either: `(2971215073, 50920843)` and `(0, 0)` fold to the same address, because PHI is
the golden ratio and its convergents are Fibonacci. So it cannot serve the elliptic face, and
`Packed` is a separate trait rather than a refinement of `Coord` -- a bijection and a discard are
different operations and should not share a name.
