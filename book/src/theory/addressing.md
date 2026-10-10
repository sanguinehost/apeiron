# Addressing

A table needs one 64-bit word per key, the address. The three faces ask different things of it.

| face | the address | so the fold must be |
|---|---|---|
| elliptic | determines the key; the key is regenerated from it | a bijection (`CoordInv::from_coord`) |
| parabolic | need not determine the key; the key is carried beside it | any spreading fold; a miss is caught by comparing the stored key |
| hyperbolic | discards information; the key is retained | a scrambler judged on distribution, never on injectivity |

Injectivity is a requirement on the first face, a non-requirement on the second and impossible on
the third. One fold cannot be right for all three, so a single collision rate measures three
different things at once.

## Integers are their own address

For `u8` through `u64`, `usize` and the signed types, `coord(&k)` is `k as u64`. Signed keys widen
by sign extension, so `-1` has every bit set. The map is injective and `from_coord` inverts it.

```rust
extern crate apeiron;
use apeiron::freq::Coord;

assert_eq!(5i32.coord(), 5);
assert_eq!((-1i32).coord(), u64::MAX);
assert!(<i32 as Coord>::INJECTIVE);
```

## From address to slot

The home slot of a coordinate `c` in a table of `2^bits` slots is the top `bits` bits of `c·PHI`,
with `PHI = 0x9E3779B97F4A7C15`, the golden ratio times `2^64`.

$$\mathrm{home}(c) = \big\lfloor (c \cdot \varphi \bmod 2^{64}) \,/\, 2^{64-\mathrm{bits}} \big\rfloor$$

- **say:** multiply by `φ` modulo `2^64`, then keep the top bits.
- **means:** consecutive and strided keys land far apart, and no key is lost before the top bits are taken.
- **vars:** `c` is a coordinate; `φ` is `PHI`; `bits` is the log of the slot count.
- **apeiron:** none — anchor. Fibonacci hashing (Knuth, *TAOCP* vol. 3, §6.4). The bijection is the statement that an odd word is a unit of `Z/2^64`, tested below.
- **physics:** none.

`PHI` is odd, so it is a unit of `Z/2^64` and `c ↦ c·PHI` is a bijection. Splitting the product into
top bits (the slot) and low bits (the residual) loses nothing. The inverse is `PHI_INV`.

```rust
extern crate apeiron;
use apeiron::freq::PHI;

let phi_inv = (0..6).fold(1u64, |x, _| x.wrapping_mul(2u64.wrapping_sub(PHI.wrapping_mul(x))));
assert_eq!(PHI & 1, 1);
assert_eq!(PHI.wrapping_mul(phi_inv), 1);
for c in [0u64, 1, 2, 1_000_003, u64::MAX] {
    assert_eq!(c.wrapping_mul(PHI).wrapping_mul(phi_inv), c);
}
```

A coordinate key skips the multiply: `CoordMap` takes `home(c) = c`, so a key's probe length is 1
for any pattern of keys.

## Composite keys

A pair `(a, b)` folds to one word. The executed fold in `apeiron::freq::Coord` is
`a.coord()·PHI ^ b.coord()`. XOR is addition in `GF(2)^64`, not in `Z/2^64`, so the crossing
loses the multiplicative spread. A signed coordinate has its whole high half set when negative, and
XOR cannot tell one sign flip from the matching flip in the other component. Measured on the grid
`-64..64` squared, then `-512..512` squared:

| key set | pairs | XOR fold distinct | `wrapping_add` fold distinct |
|---|---:|---:|---:|
| `-64..64` squared | 16384 | 8320 | 16384 |
| `-512..512` squared | 1048576 | 525312 | 1048576 |

```rust
extern crate apeiron;
use apeiron::freq::{Coord, PHI};
use std::collections::HashSet;

fn distinct(range: std::ops::Range<i32>) -> (usize, usize) {
    let mut xor = HashSet::new();
    let mut add = HashSet::new();
    for a in range.clone() {
        for b in range.clone() {
            let (ca, cb) = (a.coord(), b.coord());
            xor.insert((a, b).coord());
            add.insert(ca.wrapping_mul(PHI).wrapping_add(cb));
        }
    }
    (xor.len(), add.len())
}

assert_eq!(distinct(-64..64), (8320, 16384));
assert_eq!(distinct(-512..512), (525312, 1048576));
```

The table stores the pair beside the address on the parabolic face, so a collision costs a probe
and never a wrong answer. The count above is therefore a cost on the executed fold, and the
`wrapping_add` column is the in-ring alternative.

## Addition is not injective either

Staying in the ring is necessary and not sufficient. `f(a, b) = a·PHI + b` is not injective on
`u32 × u32`: it collides whenever `d·PHI ≡ −e (mod 2^64)` with `|d|, |e| < 2^32`. The convergents of
the golden ratio are Fibonacci numbers, and `d = 2971215073` gives `e = 50920843`.

```rust
extern crate apeiron;
use apeiron::freq::PHI;

let f = |a: u64, b: u64| a.wrapping_mul(PHI).wrapping_add(b);
assert_eq!(f(2_971_215_073, 50_920_843), f(0, 0));
```

So a composite fold is a spreading hash for the parabolic face, with the stored key as the
decision. It is not an address for the elliptic face, which would need `pack` (a bijection that
refuses keys too wide to fit).

## Where this is checked

| statement | grade | artifact |
|---|---|---|
| integer coordinates are injective and invert | tested | the doctests above and `tests/hashdos.rs`, run by `cargo test --test hashdos --locked` |
| the executed fold on pairs is XOR | read from source | `src/hashtrinity/algebra.rs`, `impl Coord for (A, B)` |
| counts above | tested | the doctests above, run by `cargo test --test hashdos --locked` and `mdbook test book` |
| the readout from the full coordinate is exact | proved in the corpus | `readout_beats_compare_by_construction` in `proof/corpus/Closures/Engineering/CoordinateReadout.lean` |
