# Hashing and table comparison for apeiron

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. The survey below, its registry and its per-face applicability columns describe that removed harness; the named paths are absent and no command naming them runs now.

Comparison plan and property matrix for the hashing/table landscape apEIRON is measured against. It
grounds the acceptance family of the R0 G1/REG-1 matrix and the "good enough for the roadmap"
reading. The programme authority is [`direction.md`](direction.md); the boundary is
[`kernel-boundary.md`](kernel-boundary.md).

Evidence convention used here, per `AGENTS.md` §5:

- **[S#]** — sourced: the statement is in a named primary artifact, listed in [Sources](#sources)
  and read for this note.
- **anchor** — a reading, a design projection, or an external/unread assertion. No command in this
  note measured it.
- **measured** — a benchmark or `perf` command run and read. **No entry in this note is
  `measured`.** Every throughput number is the cited project's own number, read, not rerun.
- `proved` is reserved for a named Lean kernel check. Theorem names here are pointers (`anchor`)
  until the corpus gate is run.

apEIRON makes no current benchmark claim for its table or crate [S1]. The comparison below therefore
names what each incumbent states about itself, plus what apEIRON's source states about itself, and
marks the rest `anchor`.

## Comparators

### Hashers

| comparator | kind | keying | quality | determinism | throughput (anchor) | HashDOS posture | no_std | stream | license |
|---|---|---|---|---|---|---|---|---|---|
| XXH3 | non-crypto | 64-bit seed; unkeyed by default | SMHasher pass, dispersion/randomness [S25] | fixed output, same on all platforms/endian [S25] | 31.5 GB/s SSE2 XXH3-64, 29.6 XXH128, i7-9700K [S25] | seed is caller data, not a secret PRF; no DoS claim [S25] | `XXH_NO_STDLIB`; one-shot/static state work [S25] | yes | BSD-2 (library files) [S25] |
| XXH64 | non-crypto | 64-bit seed | SMHasher pass [S25] | fixed cross-platform output [S25] | 19.4 GB/s i7-9700K [S25] | as XXH3 | as XXH3 | yes | BSD-2 [S25] |
| wyhash | non-crypto | secret `_wyp[4]` from `make_secret` | SMHasher pass; ~62-bit collision, not 64 [S26] | secret-dependent | author claims fastest for short keys, no number [S26] | salted "to avoid intended attack"; author lists limits [S26] | portable 64/32; no explicit no_std claim [S26] | anchor | Unlicense (LICENSE not read) anchor |
| ahash 0.8.12 | non-crypto | keyed; `RandomState` from OS RNG or compile-time RNG [S16] | SMHasher full suite pass [S16] | output unstable across versions/platforms; not for persistence [S16] | cites its own compare page (not read) anchor | claims DOS resistance by keying [S16]; mechanism page not read anchor | `std` feature; core path anchor | yes | MIT/Apache [S16] |
| foldhash-f | non-crypto | per-instance secret seed [S17] | strong on full 64-bit collisions; fails avalanche [S17] | not stable across versions/platforms [S17] | u64 hashonly 0.79 ns Xeon 8481C, 0.60 ns M2; word lookup 2.6–8.6 ns [S17] | "minimally DoS-resistant"; explicitly not vs interactive attackers; not crypto [S17] | `no_std` via `std` off [S17] | `Hasher` | Zlib [S17] |
| foldhash-q | non-crypto | per-instance secret seed [S17] | passes SMHasher3, avalanches [S17] | not stable across versions/platforms [S17] | ~1.3–1.35× foldhash-f on hashonly [S17] | as foldhash-f | as foldhash-f | `Hasher` | Zlib [S17] |
| fxhash / FxHasher | non-crypto | unkeyed | no collision/DoS resistance; poor avalanche; `u64hibits` collides [S23][S27] | deterministic | rustc-hash 2.x is polynomial + wyhash-inspired compression [S23] | none; README disclaims [S27] | rustc-hash `no_std` via `std` off [S23] | `Hasher` | MIT/Apache [S23] |
| FNV-1a | non-crypto | unkeyed | poor avalanche; fast for small keys [S18][S25] | deterministic | 1.2 GB/s i7-9700K [S25] | none; README warns [S18] | core lib anchor | `Hasher` | Apache-2.0/MIT [S18] |
| SipHash-1-3 (std `DefaultHasher`) | crypto PRF | 128-bit key; std seeds per process | PRF design | deterministic given key; std key random per process | 3.0 GB/s class [S25]; 9.09 ns u64 hashonly [S17] | resistance by secret keying; reduced rounds vs 2-4 | core | `Hasher` | MIT/Apache [S19] |
| SipHash-2-4 (`siphasher`) | crypto PRF | 128-bit key, explicit | reference SipHash | deterministic given key | anchor | by keying | core [S19] | yes, 64- and 128-bit tags | MIT/Apache [S19] |
| BLAKE3 1.8.7 | crypto | keyed MAC/PRF mode; unkeyed hash [S21] | secure, length-extension resistant; test vectors [S21] | deterministic | "much faster than SHA-2/BLAKE2"; no number read anchor [S21] | cryptographic | SIMD; no_std claim anchor | yes, XOF/verified streaming [S21] | CC0-1.0 OR Apache-2.0 OR Apache-2.0-LLVM [S21] |
| BLAKE2b | crypto | keyed / salt / personalization | RFC 7693 anchor | deterministic | README carries no number [S28] | cryptographic | reference implementation [S28] | yes | CC0/OpenSSL/Apache-2.0 [S28] |
| BLAKE2s | crypto | keyed / salt / personalization | RFC 7693 anchor | deterministic | README carries no number [S28] | cryptographic | reference implementation [S28] | yes | CC0/OpenSSL/Apache-2.0 [S28] |
| SHA-256 | crypto | unkeyed; HMAC to key | FIPS 180-4 anchor | deterministic | no number read anchor | cryptographic | RustCrypto `no_std` anchor [S22] | yes | MIT/Apache [S22] |

### Table layers

| comparator | keying | placement/probe | growth/removal | HashDOS | no_std | license |
|---|---|---|---|---|---|---|
| std `HashMap`/`HashSet` | SipHash-1-3, process-random key | SwissTable port; 1 control byte/entry; SIMD group probe; load factor 7/8; `DELETED` tombstone [S20] | doubling growth; tombstone removal | resistance by secret keying | std/alloc | std |
| hashbrown 0.15.5 (apeiron dev-dep pin [S2]) | foldhash default, pluggable | SwissTable; 1 byte overhead; SIMD; empty map does not allocate [S20] | load 7/8; doubling; `rayon` feature [S20] | default hasher weaker than SipHash, by its own README [S20] | `no_std` + `alloc` [S20] | MIT/Apache [S20] |
| hashbrown + rayon | per-instance seeds | parallel `par_iter` | as hashbrown | as hashbrown | `no_std` unlikely with rayon anchor | MIT/Apache [S20][S29] |
| apeiron `coord::spread` / direct `Coord` | optional process pepper, default `NO_PEPPER = 0` [S3][S6] | `id = x·PHI` bijection; `place = rotl(hi,32) ^ lo` discard; `finish` is splitmix bijection [S4] | direct lane is exact-coordinate candidate [S5] | coordinate key is its own address; lossy fold uses pepper [S4][S6] | std (`AtomicU64`) [S6] | MPL-2.0 [S2] |
| apeiron `fold_bytes` (`mix2`/`settle`) | default unkeyed; process pepper optional [S3] | length ladder; two independent 16-byte lanes; folded multiply plus `a - b` guard [S3] | one-shot | guard closes seed-cancel; pepper de-amortizes offline search [S3][S6]; no collision-resistance promise, 2^32 birthday [S3] | std | MPL-2.0 [S2] |
| apeiron `Table` | `Home`/`Merge` callbacks; optional global pepper [S5][S6] | 256 growable regions; 7-bit tag group scan; direct lane then 7 descent windows then cyclic sweep [S5] | region reallocation and re-placement; `ShardFull` retained only for compatibility [S5] | deterministic placement; finite first-fit order; no proved adversary bound anchor | std + alloc (`Vec`) [S5] | MPL-2.0 [S2] |
| hashtrinity `Table`/`FreqMap`/`CoordMap` (pinned predecessor [S2]) | unseeded by design; deterministic [S11] | keyless 2^−64 false-positive; `CoordMap` worst-case probe 1 [S11] | capacity-presized; merges only | claims "by construction" via `k·PHI` bijection and radix descent [S11]; `tests/hashdos.rs` states the worst-case probe bound is measured, and `AggregationBound.descent_can_be_wholly_blocked` fills a descent's whole window set [S14] | uses std anchor | MIT/Apache [S11] |

Source alignment note: the corrected `README.md` and `kernel-boundary.md` now name the executing
growable flat/spill `Table` (a flat certified-coordinate resident area plus grouped spill regions,
keyed by `(Sigma, K)` with `Stored`/`Regen` carriers) and record the fixed-shard `Table` (one home
shard, `ShardFull` refusal, `N = 0` refusal, no probing) as revision-bound historical evidence
[S1][S5][S9]. `src/table.rs` is that growable, key-partitioned representation with a finite
first-fit probe and checked sizing/reservation; `ShardFull` is retained as a compatibility variant
the growable table does not return. The divergence this note previously recorded is resolved in the
local boundary; `docs/epics/storage-refinement.md` still carries stale fixed-shard-as-executing rows
and is corrected under a separate owner.

## Fair comparisons by face

| face | fair comparators | informational only | load-bearing incumbent | bar |
|---|---|---|---|---|
| Coordinate-key count/dedup | apEIRON `coord::spread`/direct `Coord`; hashtrinity `CoordMap`; identity-hashed `HashMap<u64,·>`; std `HashMap<u64,·>` | string/byte hashers (xxHash, wyhash, ahash, foldhash, BLAKE3): they add a hash a coordinate does not need | registered: `hashtrinity_table` on current-strata serial, `hashbrown_foldhash`/`hashbrown_fxhash` on the opaque-key serial face; no `CoordMap` or serial-`FreqMap` adapter is registered (coverage open) | worst-case probe and the direct-index cell of the R0 matrix [S8] |
| Opaque-key exact map | std `HashMap`; hashbrown + foldhash; hashbrown + SipHash; ahash; fxhash; apEIRON fold + `Table`; hashtrinity exact `Table` | crypto hashers only as the hasher behind an existing table, never as the table | registered: `hashbrown_foldhash`, `hashbrown_fxhash`, `hashtrinity_table`, `std_siphash` [S8] | hit, miss, update, fresh insert, refusal timed separately [S8] |
| Opaque-key keyless count/dedup | hashtrinity `FreqMap`; `HashSet<u64>` over the fold; Bloom filter; apEIRON `Table` with the count merge | owning `HashSet<String>` alone (a different resource point, not the same face) [S12] | registered spatial only: `hashtrinity_count_par`/`hashtrinity_get_each`; no serial `FreqMap` adapter is registered | per-pair false-positive rate and RSS at fixed distinct count [S12] |
| Parallel spatial build/probe | hashbrown + rayon; hashtrinity parallel `CountTable` at `71942db`; apEIRON parallel build | single-thread hashers as parallel baselines | registered: `hashbrown_rayon`, `hashtrinity_count_par`, `hashtrinity_get_each`, `foldhash_par16`, `fxhash_par16` (all load-bearing on the spatial face) [S12] | preserve input, packing, partitioning, threads, timed region [S8] |
| Serial map/lookup baseline | std `HashMap` (SipHash-1-3); hashbrown + foldhash | non-map hasher microbenchmarks | registered: `std_siphash` (correctness/determinism), `hashbrown_foldhash`/`hashbrown_fxhash` (speed), `hashtrinity_table` [S8] | `selected/incumbent <= 1.0` on every applicable cell [S8] |
| Keyed PRF / cryptographic face | SipHash-1-3/2-4; BLAKE3 keyed/MAC; BLAKE2b/s keyed; SHA-256 + HMAC | non-crypto hashers for authentication | none in-repo; the claim must name its own security contract | a named contract, not a speed cell |

### Acceptance registry (`examples/reg1/acceptance.rs`, read — historical; the harness was removed in c9a2d0c)

The registry below records the removed harness. `examples/reg1/acceptance.rs` and
`examples/reg1/acceptance/registry_tests.rs` are absent from the tree, so the classes, the
applicability reasons and the `selected/incumbent <= 1.0` bound describe that harness, not a
registry a reader can open now.

The actual per-cell class is the registry below, not a recommendation; it corrects the
design-intent "load-bearing incumbent" column in "Fair comparisons by face" above. A pair is
load-bearing or informational, and a load-bearing pair is also either `Applicable` on a face or
`AuthorizedNotApplicable` per cell with a registered reason.

| incumbent | class | applicability (registered reason) |
|---|---|---|
| `hashtrinity_table` | load-bearing | applicable: current-strata serial map incumbent on the same stored-key face |
| `hashtrinity_count_par` | load-bearing | authorized-not-applicable per cell: packed spatial count workload only; strata cells are the serial-map face |
| `hashtrinity_get_each` | load-bearing | authorized-not-applicable per cell: packed spatial probe only; strata cells are the serial-map face |
| `hashbrown_foldhash` | load-bearing | applicable: opaque-key exact map face (hit/miss/update/insert) |
| `hashbrown_fxhash` | load-bearing | applicable: opaque-key exact map face (hit/miss/update/insert) |
| `hashbrown_rayon` | load-bearing | authorized-not-applicable per cell: packed spatial workload only |
| `std_siphash` | load-bearing | applicable: correctness/determinism reference on the same total-map face |
| `foldhash_par16` | load-bearing | authorized-not-applicable per cell: packed spatial workload only |
| `fxhash_par16` | load-bearing | authorized-not-applicable per cell: packed spatial workload only |
| `xxhash_xxh3` | load-bearing | applicable: opaque-key exact map face (hit/miss/update/insert) |
| `xxhash_xxh64` | load-bearing | applicable: opaque-key exact map face (hit/miss/update/insert) |
| `wyhash` | load-bearing | applicable: opaque-key exact map face (hit/miss/update/insert) |
| `ahash` | load-bearing | applicable: opaque-key exact map face (hit/miss/update/insert) |
| `fnv` | load-bearing | applicable: opaque-key exact map face (hit/miss/update/insert) |
| `blake3` | informational | applicable: informational cryptographic comparator; no named authentication/integrity contract in this harness |
| `sha256` | informational | applicable: informational cryptographic comparator; no named authentication/integrity contract in this harness |

**T1 landed.** [Direction](direction.md#required-r0-acceptance-g1--reg-1) names XXH3/XXH64,
`wyhash`, `ahash` and `fnv` as required load-bearing incumbents. Task `APEIRON-REANCHOR-W1-T1` made
the five load-bearing in `examples/reg1/acceptance.rs`: each is the same `hashbrown::HashMap`
surface as `hashbrown_foldhash` with a different `BuildHasher`, measured over the same admitted key
set, lookup trace, update trace and canonical readback, so each is applicable on the opaque-key
exact map face and subject to `selected/incumbent <= 1.0`. Each is listed in
`REQUIRED_LOAD_BEARING`; demotion to informational or omission rejects. The registry-integrity
controls are `examples/reg1/acceptance/registry_tests.rs` — `required_incumbents_are_present_load_bearing_and_built`,
`every_registered_incumbent_is_classified`, `required_incumbent_classification_is_correct`,
`omitted_required_pair_is_reported_missing`, `get_each_combined_arm_is_concurrency_unmatched` and
`acceptance_threshold_is_one` — reached by `cargo test --locked --example reg1 acceptance::`; they
read the compiled registry and the pure classification helpers and run no benchmark and no adapter.
No run of that command is claimed by this note. `blake3` and `sha256` stay
informational (no named authentication/integrity contract in the R0 matrix) and are carried in
`INFORMATIONAL_COMPARATORS` with that reason; `blake2b`/`blake2s` are not registered in the harness.

**`get_each` combined arm rejected.** The `hashtrinity_get_each` adapter is timed through the
combined `count_par` build plus grouped probe, so its build phase runs on the hashtrinity global
pool while the reported policy names only the single-core probe. `spatial_incumbent_workers` returns
the build pool width and `pair_concurrency` therefore returns `Mismatch` against the single-core
selected run: the pair is counted in `concurrency_unmatched` and rejected, not matched to the serial
selected run. The `acceptance-concurrency-mismatch` control and the
`get_each_combined_arm_is_concurrency_unmatched` unit test enforce this.

**Coverage open.** The registry has no `hashtrinity::CoordMap` or serial `hashtrinity::FreqMap`
adapter. [Direction](direction.md#required-r0-acceptance-g1--reg-1) requires independently checking
coverage of `hashtrinity` `Table`, `CoordMap` and `FreqMap` and the historical serial/spatial
surfaces; the missing adapters are an open coverage gap, not a `not-applicable` finding.

## Good enough by domain

| domain | property it actually needs | bar-setter | apEIRON reading |
|---|---|---|---|
| General Rust std replacement | hasher quality and speed; HashDOS only if keys are untrusted; permissive license; optional no_std | hashbrown/foldhash for speed; std SipHash for untrusted keys | licence is `MPL-2.0` [S2]; no licence blocker; no current speed claim [S1] |
| Operating systems / kernels | `no_std`, no hidden allocation, deterministic, no panics, bounded worst case | hashbrown `no_std`+`alloc`; an inline fixed table | grows regions with `Vec` and uses `std` atomics [S5][S6]; not a no_std drop-in |
| Programming-language runtimes / compilers | small-integer throughput, deterministic across runs, trusted inputs | FxHash/rustc-hash and foldhash-f | coordinate direct index is the right shape but unmeasured; license applies |
| Proof assistants / proof kernels | reproducibility and cross-process determinism, stable names, keyed collision resistance; speed secondary | a deterministic fold with an explicit key, or keyed SipHash | deterministic default and logged pepper fit replay [S3][S6]; the pepper is process-global state, not per-map |
| Video games (broadphase / interest management) | replay and netcode determinism, coordinate direct indexing, no per-run seed, low latency, parallel build, console no_std | direct coordinate indexing; `CoordMap`; parallel build | apEIRON is aimed here (replay-preserving pepper, coordinate addressing) [S3][S6]; no measured claim |
| Servers (untrusted keys) | HashDOS resistance under adversarial keys; a secret the attacker cannot precompute against | std SipHash-1-3 with a per-process random key; ahash per-instance seed | pepper is a process-wide constant, so an attacker who learns it precomputes once [S6]; per-instance keying is stronger for this domain |
| Aerospace / satellites / aircraft | determinism, worst-case execution time, no hidden allocation, qualification evidence, security only if networked | a bounded, allocation-free, deterministic table with a documented worst case | the governing docs now name the growable flat/spill form as executing and the fixed-shard form as historical [S1][S5][S9]; it grows regions with `Vec`, uses `std` and has no WCET evidence |

Cryptographic face: warranted for authentication (MAC), integrity digests, key derivation, tokens
and content addressing across a trust boundary, and — at the weaker PRF level — for untrusted-key
server maps where a secret key must defeat precomputation. Unwarranted for internal compiler tables,
coordinate/dense indexing, replay-deterministic game maps, and proof-assistant name maps, where
determinism, not secrecy, is the requirement. Password hashing is not in this set; BLAKE3's own
README directs that to Argon2 [S21].

## Sources

Local apEIRON:

- [S1] `README.md`.
- [S2] `Cargo.toml` and `Cargo.lock`: `license = "MPL-2.0"`; pins foldhash 0.1.5, hashbrown 0.15.5,
  rustc-hash 2.1.3, criterion 0.5.1; hashtrinity git rev
  `4cc14471ae24f3f204fb0b0e91736ef75c909358`.
- [S3] `src/fold.rs`.
- [S4] `src/coord.rs`.
- [S5] `src/table.rs`, `src/table/scan.rs`.
- [S6] `src/pepper.rs`.
- [S7] `src/table/probe.rs` ordinary finite read/locator regression tests; symbolic Kani harnesses
  were retired under `docs/epics/kani-retirement.md`.
- [S8] `docs/direction.md`, section "Required R0 acceptance: G1 / REG-1".
- [S9] `docs/kernel-boundary.md`.
- [S10] `docs/plan.md`.

Pinned predecessor, `file:///home/socol/Workspace/fold` at the [S2] revision:

- [S11] `hashtrinity/README.md`.
- [S12] `hashtrinity/BENCHMARK.md`.
- [S13] `hashtrinity/proof/THEOREMS.md`.
- [S14] `hashtrinity/tests/hashdos.rs`.
- [S15] `hashtrinity/src/seed.rs`.

Cargo registry (crate version pinned by checksum; README and `Cargo.toml` read):

- [S16] ahash 0.8.12.
- [S17] foldhash 0.1.5 (and 0.2.0 for comparison).
- [S18] fnv 1.0.7.
- [S19] siphasher 1.0.3.
- [S20] hashbrown 0.17.1 README; `src/raw.rs:189` (`bucket_mask_to_capacity`, 7/8),
  `src/control/tag.rs:9,12` (`EMPTY`, `DELETED`); hashbrown 0.15.5 as the apEIRON pin.
- [S21] blake3 1.8.7.
- [S22] sha2 0.10.9.
- [S23] rustc-hash 2.1.3.
- [S24] twox-hash 2.1.4 (XXH3/XXH64 Rust binding).
- [S29] rayon 1.12.0.

External, fetched and read at the stated revision:

- [S25] `Cyan4973/xxHash` `dev` `e103f5f1ed345acc83de15632e04d50e6dbdbfa1`, `README.md`.
- [S26] `wangyi-fudan/wyhash` `master` `e4764a0b637d34d3421a7760affada9288b625a8`, `README.md`.
- [S27] `cbreeden/fxhash` `master` `a03cbf42dde010250adef75908f109be197d982b`, `README.md`.
- [S28] `BLAKE2/BLAKE2` `master` `ed1974ea83433eba7b2d95c5dcd9ac33cb847913`, `README.md`.

Unread; every claim resting on these is `anchor`: the SipHash paper (Aumasson–Bernstein), FIPS
180-4, RFC 7693 (BLAKE2), rapidhash, and the Abseil SwissTable design note.
