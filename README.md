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
counted, then probed for every occupied cell. Criterion, 20 samples, median and 95% interval.

**Single-threaded**, where the measurement is trustworthy -- every interval below is within 3.5%,
because one thread does not compete with whatever else the machine is doing:

| arm | 100k | 1M | exact | deterministic across processes | HashDoS |
|---|---|---|---|---|---|
| **apeiron_serial** | **1.06 ms** | **27.7 ms** | yes | **yes** | **structural** |
| hashbrown_foldhash | 1.44 | 35.2 | yes | no | none advertised |
| hashbrown_fxhash | 1.45 | 35.4 | yes | yes | none |
| std_siphash | 4.25 | 108.2 | yes | no | by random seed |

21% to 26% ahead, deterministic, and the only arm with a structural answer on the last column.

**The 16-thread comparison is owed a quiet machine and is not published here.** The run these
numbers come from had a background process holding about 3.6 cores, and it shows: the `par16` arms
came back with intervals from 9% to 62% wide, against 1-3% for the serial arms, so their *ordering*
is not resolvable from it. What can be said is that the spread is not evenly distributed --
`apeiron_par16` was the widest arm in the 1M group at +/-50%, against +/-12% for `fxhash_par16` --
which suggests its two-level partition is more sensitive to core contention than the competitors'
flat sharding, and is worth measuring on its own rather than assuming.

One thing does survive the noise, because it is a comparison against apeiron's own serial arm in
the same run: **at 100k the serial build beats every parallel arm, including apeiron's**, at 1.06 ms
against a best case of 1.42. The partition is a fixed cost that has not started paying at that size.
At 1M it has: 27.7 ms serial against 7.6 parallel. The crossover sits between, and
`CountTable::build_parallel` does not currently know about it -- it partitions at every size.

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
deliver while the word was invertible. `CountTable` took the same split, and the attack there has to
aim at both ends of the placement word — the shard is its bottom eight bits, the descent walks down
from bit 57 — so crowding one region costs `2^(8 + j·gbits)` trials per key: 247 for the shard alone
(which is a *speed-up*, 0.71×, because a region is sized to its load), 63,839 for one further
window, and unaffordable for two.

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

`CountTable` has one too (`CountTable::with_pepper`, or the process floor). Measured structurally
rather than by the clock: the same aimed keys reach **one** shard of 256 under the salt they were
aimed at, and **all 256** under a peppered table — against 255 for honest traffic.

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

Per lookup, 694 keys, L1-resident, `perf stat` on one core. Instructions retired rather than
cycles: the count is per-process and reproduces on a machine that is doing other work, which cycles
do not.

| | instr/lookup | deterministic | DoS |
|---|---|---|---|
| **apeiron, `u64`** | **34.8** | yes | by pepper |
| hashbrown + foldhash | 35.3 | no | none advertised |
| hashbrown + fxhash | 34.2 | yes | none |
| std (SipHash) | 128.2 | no | by random seed |
| **apeiron, `String`** | 93.4 | yes | by pepper |
| hashbrown + foldhash, `String` | 83.9 | no | none advertised |
| hashbrown + fxhash, `String` | 92.3 | yes | none |

On `u64` keys the probe is half an instruction under the flat competitor and within one of fxhash,
which buys its count by having no resistance to pay for. On a quiet machine that read 6.84 cycles
against foldhash's 6.62.

The `u64` gap was 70 instructions until `get` and `find` were made to actually inline; IPC was
already as high as hashbrown's, which is what said the problem was instruction count rather than
stalls.

The `String` gap is 9.4 instructions and **its cause is not known**. It is not the fold: the two
folds are at instruction parity on this corpus, 23.1 against 24.1. The obvious suspect was tested
and did not survive -- a map with a zero-sized value brings the slot to 32 bytes, the same as
hashbrown's, and is no faster (93.2 against 93.4). It is recorded as open rather than attributed,
because a wrong cause forecloses the next measurement. Note also that this is the lookup column
only: on build-then-look-up, apeiron takes every row from 8 keys to 500,000, which is where
carrying the address beside the key pays -- growth is a probe per entry rather than a re-fold of
every key's bytes.

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
