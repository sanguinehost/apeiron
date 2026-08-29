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

## What it costs

`benches/spatial_binning.rs`: a clustered scene of signed world coordinates binned into cells,
counted, then probed for every occupied cell. `n = 1,000,000`, 16 threads, criterion mean, one
uninterrupted run.

| arm | ms | exact | deterministic across processes | HashDoS |
|---|---|---|---|---|
| **apeiron_par16** | **3.16** | yes | **yes** | **structural** |
| fxhash_par16 | 3.99 | yes | yes | none |
| foldhash_par16 | 4.14 | yes | no | none advertised |
| std_siphash_par16 | 6.81 | yes | no | by random seed |
| hashtrinity_packed | 10.27 | yes | yes | none |
| apeiron_tuple_par16 | 10.42 | yes | yes | structural |
| hashtrinity_tuple | 12.09 | **no, off by 8236** | yes | none |
| apeiron_serial | 32.70 | yes | yes | structural |
| hashbrown_foldhash | 33.71 | yes | no | none advertised |
| hashbrown_fxhash | 36.00 | yes | yes | none |
| std_siphash | 102.67 | yes | no | by random seed |

Every arm ending `par16` is sharded and threaded the same way: the same shard function, the same
two-level parallel partition, one map per shard, and probes grouped exactly as the build was.
That is not a courtesy. A parallel arm measured against a serial one measures the scheduling and
not the table, and this benchmark made that mistake in apeiron's favour for one commit -- the
reference arms were still scattering a million keys on one thread after apeiron's partition was
threaded, which is worth 4.5 ms and made a 7% lead look like 2.3x.

The whole point is the last three columns. foldhash's seed is the stack pointer and ASLR and its
own comment says it does not advertise HashDoS resistance; std seeds SipHash randomly. Both are
therefore non-deterministic across processes, so neither can back a replay.

### A correction to that last column

The claim used to read: apeiron is seedless, and what defends it is the shape of the probe rather
than a secret -- `table::group_of` descends successive windows of the whole 64-bit word, so
extending a chain by one step costs the attacker another `gbits` of control and an unbounded chain
costs all 64.

**That was true, and then it stopped being true.** "Bits of control" is not the same as cost while
the address is a public *bijection*: all the windows are slices of one word, so matching every one
of them was a single call to `unspread`. Measured, 200,000 keys aimed that way built **988x** slower
than an honest set.

`map` then split its address in two (`coord::Addr`). The **identity** word is still a bijection --
the elliptic face reads keys back off it -- but the **placement** word, the one the descent and the
tag actually read, is a fold of both halves of a widening multiply and has no inverse. An attacker
can still evaluate it; they can no longer solve it. So keys agreeing on `j` windows must be *found*,
at `2^(j * gbits)` trials each:

| windows matched | trials per key | search | mean probe |
|---|---|---|---|
| honest | — | — | 1.02 |
| 1 | 254 | instant | 2.00 |
| 2 | 67,682 | 0.10 s | 2.99 |
| 3 | > 4 × 10⁹ | unaffordable | — |

Exponential cost, linear damage — which is what the original claim always meant and could not
deliver while the word was invertible. `CountTable` still addresses through `finish`, a bijection,
so it keeps the weaker story; that is a gap and is written down as one.

What defends it is a secret, as it is for SipHash. `map::Salt` is that secret and `pepper` is how it
gets set:

```rust
// once, at startup, before any table exists
apeiron::pepper::set_process_pepper(config.pepper);
tracing::info!(pepper = config.pepper, "address salt");   // write it down
```

The ratio goes to **1.0x**. It stays compatible with replay because a pepper is a process-wide
*constant* rather than a per-run random seed: a recorded pepper reproduces a recorded session
exactly, which is the property a random seed denies and the reason foldhash and SipHash cannot back
one. `Exposed<K, V>` takes a `NonZeroU64`, so a table fed from outside the process can be made
unable to exist without one; `Map::stats()` reports `longest_probe` so an attack is visible whether
or not anybody remembered.

`CountTable` has no pepper yet and still carries this exposure. That is a gap, written down here
rather than left as a claim.

## The map and the set

`table` is the hyperbolic face -- it keeps a fold and a count and throws the key away. A map cannot,
so `map` is the other two faces, and which one a key gets is decided by the key:

| key | its fold | face | a slot holds | a hit costs |
|---|---|---|---|---|
| fits in a word (`Packed`) | a bijection | elliptic, **B** | the address | one compare |
| wider (`Coord` only) | a discard | parabolic, **B/U** | address **and** key | a compare, then the key |

`Map<K, V>`, `Words<K, V>`, `Set<K>`, `WordSet<K>`, `Exposed<K, V>`, `ExposedSet<K>`. The std shape,
including `entry`, `extract_if`, `get_disjoint_mut`, `try_reserve` and `Index`, plus a borrowed-key
surface so `Map<String, V>` is probed with `&str` and `Map<(String, u16), V>` with `(&str, u16)`,
allocating nothing on a hit. `#[derive(Coord)]` and `#[derive(Packed)]` behind the `derive` feature,
which is off by default so the ordinary build keeps zero dependencies.

Per lookup, 694 keys, L1-resident, `perf stat` on one core:

| | instr/lookup | cycles/lookup | deterministic | DoS |
|---|---|---|---|---|
| apeiron, `u64` | 35.9 | 6.5 | yes | by pepper |
| hashbrown + fxhash | 32.2 | 5.7 | yes | none |
| std (SipHash) | 105.2 | 26.2 | no | by random seed |

The gap was 70 instructions until `get` and `find` were made to actually inline; IPC was already as
high as hashbrown's, which is what said the problem was instruction count rather than stalls.
`String` keys stay about 22% behind hashbrown, which is the price of carrying the address beside the
key -- growth is then a probe per entry rather than a re-fold of every byte.

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
