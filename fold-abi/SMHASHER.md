# Running SMHasher against `apeiron::fold`

The fold had never faced a statistical battery. This is how to run one.

```sh
git clone --depth 1 https://github.com/rurban/smhasher
cd smhasher && git submodule update --init --recursive --depth 1   # SHA-Intrinsics, prvhash
cargo build --release -p apeiron-fold-abi                          # libapeiron_fold_abi.a
```

Three edits to SMHasher:

- `Hashes.h`: `extern "C" void apeiron_fold(const void*, int, uint32_t, void*);` and an
  `apeiron_test` wrapper forwarding to it.
- `main.cpp`: a row in the hash table --
  `{ apeiron_test, 64, 0x00000000, "apeiron", "apeiron fold (mix2 ladder)", POOR, {}},`
- `CMakeLists.txt`: add `.../target/release/libapeiron_fold_abi.a` to
  `target_link_libraries(SMHasher ...)`.

Then `cmake -DCMAKE_BUILD_TYPE=Release . && make -j8 SMHasher && ./SMHasher apeiron`.

## Runs, 2026-08-30

| run | fails | exact 64-bit collisions |
|---|---|---|
| 1, as found | 8 | PerlinNoise 14,741,760 and 493,636; DiffDist 3,570 |
| 2, `c` key-derived at 1-3 bytes | 7 | PerlinNoise 493,636; DiffDist 3,570 |
| 3, pepper multiplied, constants dropped | 8 | Sparse 9/279/36, Permutation 225, TwoBytes 1/19, DiffDist 1,000 |
| 4, constants and length restored | 6 | DiffDist 100 |
| 5, finalizer added | **2** | **none** |
| 6, guard changed to `a - b` | **0** | **none** |
| 7, finalizer routed by length | **0** | **none** |

Run 5 passes Sanity, AppendedZeroes, Avalanche, Sparse, Permutation, TwoBytes, Text,
PerlinNoise, DiffDist, Diff (8.3e9 tests), Bic, Zeroes, Seed, Window, Cyclic, Prng, and runs bulk
at ~6.9 bytes/cycle. **Zero exact 64-bit collisions in every keyset.**

## Run 6, 2026-08-31: **zero failures**

`mix2`'s guard became `a - b` from `a + rotl(b, 32)`, and `MomentChi2` went from the crate's last
failing test to passing:

| | run 5 | run 6 |
|---|---|---|
| `MomentChi2` deriv b1 / b0 | 145,171 / 143,659 | **354.5 / 439.9** |
| `DiffDist` bit 63, 35-bit window | 81/63 (1.27x) | **71/64 (1.11x)** |
| failures | 1 | **0** |
| exact 64-bit collisions | none | none |

The derivative was measuring the guard. At four and eight bytes the fold reads the same bytes twice,
so both `mix2` operands are `t + seed` and the guard restricted to that line is `alpha*t + k`; with
`alpha` odd the guard reproduces the input faithfully and the derivative sees the structure. Both
coefficients odd makes `alpha` even and the line collapses. See `examples/moment_chi2.rs`, which
reproduces the statistic in seconds, and the four proofs beside `fold::mix2`.

The same change made the guard **thirty-one bits more swap-resistant**: `a + rotl(b, 32)` is
symmetric on every pair of words with the same half difference -- classes of size `2^32`, since
`f(x) = x - rotl(x,32)` is `(h-l)*(2^32 - 1)` and `2^32 - 1` is odd, hence invertible -- and `a - b`
is symmetric on exactly two points.

## Run 7, 2026-08-31: the finalizer only where it is needed

`settle` is not repairing the fold, it is repairing the **dependence between the two `mix2`
operands**. Below sixteen bytes the reads `b[..8]` and `b[len-8..]` overlap or coincide; from
sixteen to thirty-two the last `mix2` still takes raw key words, so its guard is affine in the key;
at thirty-three and above, and in `fold_long`, the last mix is `mix2(x, y + NU)` -- affine only in
already-folded state, and the finalizer is dead weight.

Routing it there costs nothing and buys four instructions on every key of 33 bytes or more:
**42-byte paths fold in 57.07 instructions against `foldhash`'s 73.07**, and the whole map lookup
in **121.79 against hashbrown + foldhash's 130.32**.

Routing at *sixteen* instead passed `examples/diffdist.rs` at 17, 24 and 32 bytes and then failed
`Avalanche` at twenty bytes with a **13.62%** worst bias. `examples/avalanche_bias.rs` exists
because of that, runs at `SMHasher`'s own 300,000 reps so the numbers compare directly, and reads
0.59-0.85% at every length on the shipped fold against `SMHasher`'s 0.618-0.911%.

**Two screens, then this file.** `diffdist.rs` alone is not sufficient and neither is
`avalanche_bias.rs`; each caught something the other missed.

## Screening without an eleven-minute run

`examples/diffdist.rs` reproduces `DiffDist` in **0.94 seconds** and is calibrated against this
table: it reads 83/64 for run 5's fold where `SMHasher` reads 81/63, and 155/64 for a rejected
seed-by-rotation variant where `SMHasher` reads 152/63. Within 2% on both.

That matters because `examples/top_bit.rs` is **not** a sufficient screen and said both rejected
candidates were free. Use `diffdist.rs` to search and this file to decide.

Not diagnosed. The suspicion is that a four-byte key gives `mix2` two affinely related operands,
since `a` reads `b[..4]` and `c` reads `b[len-4..]` and at `len == 4` those are the same bytes, so
the fold is quadratic in the key and consecutive differences are linear in it. **That is a
hypothesis and it is not confirmed**: a local check of the difference popcount over 200,000
consecutive keys comes out at mean 32.005 and variance 16.86 against an ideal 32 and 16, which is
too coarse to see what a moment test over 2^32 keys sees.

Worth noting for severity: apeiron does not address integers through the byte fold. `Coord` for an
integer is the identity and `map` sends it through `spread`. This keyset is four-byte *strings*.

### What run 1 found, and what it cost to fix

The pepper entered additively into one operand: `mix2(a + seed, c + NU)`. The hash is then a
function of `a + seed` whenever two keys share `c`, so key and pepper are interchangeable. At one
to three bytes `c` was always zero, so every pair collided. `examples/perlin.rs` reproduces the
count exactly and now reads zero; `examples/sparse_keys.rs` covers the keyset run 3 broke.

Fixed by giving each lane its own odd multiplier of the pepper, plus a constant and a length term.
Dropping any of those pieces reintroduces collisions somewhere else -- see the run table.

### Still failing, run 4

`Avalanche`, `DiffDist`, `Text`, `TwoBytes`, `MomentChi2` x2. Only `DiffDist` has exact
collisions, 100, and they are concentrated at the **top input bit**:

| flipped bit | high-32 collisions vs 512 | distribution bias |
|---|---|---|
| 62 | 2,617 | 70.6% |
| 63 | 642,947 | 99.7% |

That is the shape of a multiply carrying entropy upward: flipping the top bit of an operand moves
the product only in its high half, so the induced *difference* is nearly a function of the other
operand rather than of the flipped one, and the differences cluster.

The obvious repair is a finalizer on the fold's output. Note that `map` currently has none for
string keys: `already_mixed` returns the fold's result as the address unchanged, because
`Coord::SPREAD` says it is mixed. That skips a multiply on every string lookup, which is exactly
what would fix this, so the fix and the optimisation are in tension and the trade needs measuring.

Not attempted yet.

## First run, 2026-08-30
## First run, 2026-08-30

Passes: sanity, appended zeroes, Diff (8.3e9 tests, 0 collisions), Prng, Bic, Zeroes, Seed,
Permutation, Window, Cyclic, TwoBytes low windows, bulk speed 6.9 bytes/cycle.

**Eight sections fail.** Not yet diagnosed, listed so they are not lost:

| test | worst reading |
|---|---|
| Avalanche | 0.802% bias, 1024-bit keys (threshold is 0.75%) |
| Keyset 'Sparse' | 2048-bit keys, up to 2 bits set |
| Keyset 'TwoBytes' | 42-bit window: 14,741,760 collisions against 31 |
| Keyset 'Text' | -- |
| Keyset 'PerlinNoise' | 64-bit: 493,636 collisions against 0 |
| DiffDist | bit 63: 1,856,794 collisions against 512; 99.958% bias at bit 54 |
| MomentChi2 | derivative moments 1.5e8 and 2.4e8 |

DiffDist's bias climbs with the input bit position -- 55% at bit 58, 99.958% at bit 63 -- which
looks like a multiply carrying entropy upward and the fold not bringing it back down. That reading
is **not confirmed**: the obvious consequence would be a weak tag, and the tag is measured at
0.991-0.993 probability of changing per flipped input bit, against an ideal of 127/128 = 0.9922,
at every length from 8 to 64 bytes and every bit position. So the cause is elsewhere.

None of the crate's own tests catch any of this, which is the point of running the battery.
