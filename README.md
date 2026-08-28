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

## The coordinate fold

`coord.rs` mixes with `wrapping_mul(PHI)` then `wrapping_add`, both in `Z/2^64`, because a
coordinate fold is ring arithmetic and leaving the ring costs correctness. hashtrinity's
`a.wrapping_mul(PHI) ^ b` mixes `Z/2^64` with `GF(2)^64` and collides on 49% of a signed pair grid
and 72% of a signed triple grid; `tests/coord_is_injective.rs` is the gate it never had.
