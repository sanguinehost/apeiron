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
