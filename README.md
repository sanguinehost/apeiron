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

## Scope, stated

The corpus is over the reals; this crate is over wrapping `i64`. Most of the structure transfers
and some does not, so the tests assert the integer facts:

- **Transfers:** commutativity, associativity, the unit, multiplicativity of the norm, the channel
  count, componentwise multiplication in the null basis, the norm factoring into the channels,
  order-free folding above zero, zero divisors at and above zero.
- **Does not:** the nontrivial idempotents are `(1 +/- j)/2` and are not integral, so over `Z` no
  face has one. What survives is the discard itself -- `(1, 1)` is `2e+`, and multiplying by it
  kills the lower lane the way the idempotent would.

## The coordinate fold

`coord.rs` mixes with `wrapping_mul(PHI)` then `wrapping_add`, both in `Z/2^64`, because a
coordinate fold is ring arithmetic and leaving the ring costs correctness. hashtrinity's
`a.wrapping_mul(PHI) ^ b` mixes `Z/2^64` with `GF(2)^64` and collides on 49% of a signed pair grid
and 72% of a signed triple grid; `tests/coord_is_injective.rs` is the gate it never had.
