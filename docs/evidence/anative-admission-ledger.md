# Evidence — algebra-native addressing admission and retirement ledger

- **Task:** `APEIRON-ANATIVE-W0-F1`, census, READ-ONLY on `src/`.
- **Spec:** `docs/spec/algebra-native-addressing.md` (per-face address contract, summary deletion).
- **Base revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied checkout is dirty.
- **Grade:** `tested` for the completeness check below (it executes and its rejecting control is red).
  Every disposition is a planning decision at grade `anchor`; no Lean theorem, build or benchmark
  ran here.
- **Scope:** this file only. No `src/`, `tests/`, `proof/`, spec/plan/intent or `Cargo.toml` edit.

## 0 · Provenance and source digests

The census binds to the working-tree bytes below, not to HEAD alone. `sha256` of the files read:

```text
e568d0c318801304d17ad304a1f719d509de28b35b33764ad6980619e3e4856d  src/table.rs
8dad06425b7ae68b47657e97d7f31ece917b0dd87b7c520d7e1b4f321bf64f8a  src/table/state.rs
11249962faf8131325a248c43840e8f5d66b22c190f53ae09115fb4ac94c0ce0  src/table/keyrep.rs
bd843ddcf5fb8074736a52ffbca4fa0bc1bf872839d770100e54c81f59ab1290  src/table/parallel.rs
7cf4724447b5ef8de8c4d4c2a175432d2284e3c1152b1f41c7f374cd13688c64  src/table/partition.rs
a1d6e242c912706752375027c21b255677830ec74254d89fc8d5dd1324c06009  src/table/resources.rs
ec633507a8ad456a7cb11553f559cf1abc08d3d4984a118e145eb5aa1ad91e97  src/table/scan.rs
4cb247c46827abf207aa14a24f433621a048e53e04beb338eef0d59279cd3435  src/table/tests.rs
3cf98b30bf12aaf6081e025677a8232c4148c2bac202b28f5075286294e585c1  src/coord.rs
7607330cd7784879b008a12d8f74973c13cc72b6012209df57c4502680be1be9  tests/table.rs
93d9257b03cdca48658ddca13e6192cfcdccb407cc1c0a0ffa4689e5b8683bf7  tests/parallel.rs
7b2e5747d9d49a85fa57dfe92c3e7056e2728f8cd3ddb9ce914498118b42f387  examples/micro_spill.rs
```

`src/table/state.rs` is `include!`-ed at `src/table.rs:145`; its citations name the physical file.

### 0.1 `rg -n` counts recorded

```text
rg -n 'summary_index'          src | wc -l   -> 11    proof -> 4
rg -n 'summary_set'            src | wc -l   -> 11    proof -> 3
rg -n 'summary_maybe'          src | wc -l   -> 12    proof -> 5
rg -n 'summary_words'          src | wc -l   ->  6    proof -> 0
rg -n 'summary_trailing_zeros' src | wc -l   ->  7    proof -> 15
rg -n 'SUMMARY_BITS_PER_SLOT'  src | wc -l   ->  3
rg -n 'SUMMARY_MIN_BITS'       src | wc -l   ->  3
rg -n 'publish_live'           src | wc -l   ->  7    proof -> 6
rg -n 'SpillSummary'           src | wc -l   ->  2    proof -> 0
rg -n 'SpillFingerprintExact'  src tests examples proof -> 0
rg -n '\.summary\b|summary:'   src | wc -l   -> 28    tests -> 0   examples -> 0
rg -n '\bsummary\b'            src | wc -l   -> 104   examples -> 7
rg -n 'PHI'                    src -> 56   tests -> 5   examples -> 27   proof -> 133
rg -n 'coord::PHI'             src+tests+examples -> 8
rg -n 'TAG_LIVE|TAG_FINGERPRINT|fn tag\(' src -> 5
rg -n 'Fingerprint|fingerprint' src -> 13    proof -> 0
rg -n 'slots_for'              src -> 12
rg -n 'is_power_of_two'        src -> 5     examples -> 1
rg -n 'byte_len_for'           src -> 3
rg -n 'DESCENT_MASK'           src -> 2     'WINDOW_MASK' src -> 2
rg -n '>> 8'                   src -> 7     tests -> 4
rg -n '& 255'                  src -> 4     tests -> 4
```

The bare-word `summary` count (104 in `src/`) includes doc prose and test names; the per-symbol
counts above are the precise deletion surface. `proof/storage-controls.py` uses an unrelated
`summary: str` control-description field (`proof/storage-controls.py:191`) and is **not** a
`Region.summary` consumer.

## 1 · Generic-API admission ledger

Dispositions: **admit** (the new contract keeps it as certified), **narrow** (kept but constrained to
a declared face/profile/domain), **block** (cannot be certified on the named face and must be
refused or removed from the admitted population). The certified faces are: elliptic
`Z/2^64` exact packing with a checked at-most-word domain; parabolic `Z/2^64` stable fold with a
faithful retained identity; hyperbolic `Z/q` (`q = p^k`, `p` odd) explicit encoding into the
validated profile with a faithful retained identity. A row citing `Sigma` states that the entry point
takes the face as a value and today ignores or under-specifies it.

<!-- rows:generic_api expected=45 -->
| Entry point | file:line | Declared signature | Wider-than-certified admission | Disposition |
|---|---|---|---|---|
| `Home` trait | `src/table.rs:75` | `pub trait Home<K> { fn home(&self,&K)->u64; fn address(&self,Sigma,&K)->TableAddress }` | Any `K` and any `u64` placement; no face, no profile, no codec law | narrow |
| `Home::home` | `src/table.rs:77` | `fn home(&self, key: &K) -> u64` | Arbitrary placement word; no face selection and no profile identity | narrow |
| `Home::address` default | `src/table.rs:79` | `fn address(&self, _sigma: Sigma, key: &K) -> TableAddress` | Ignores `Sigma`; `direct_word: None`; cannot fabricate an elliptic exactness certificate or an odd-modulus ring from an arbitrary `u64` | narrow |
| blanket `Home` closure adapter | `src/table.rs:86` | `impl<K, F: Fn(&K) -> u64> Home<K> for F` | Admits an arbitrary opaque closure as a placement-only home on every face; the spec's named wider population | block |
| `WordHome` | `src/table.rs:104` | `impl Home<i64> for WordHome` | `home` is `spread(word).place`; ignores `Sigma`; not an authoritative all-face fallback | narrow |
| `WordHome::address` | `src/table.rs:114` | `fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress` | Certifies `direct_word` only below `2^32`; ignores `Sigma`; elliptic/parabolic only | narrow |
| `Merge` trait | `src/table.rs:93` | `pub trait Merge<V> { fn merge(&self, new: V, old: V) -> V }` | Value callback only; no `(K,H,R)` admission | admit |
| blanket `Merge` closure adapter | `src/table.rs:97` | `impl<V, F: Fn(V, V) -> V> Merge<V> for F` | Value callback only | admit |
| `WordMerge` | `src/table.rs:127` | `pub struct WordMerge; impl Merge<i64>` | Value callback only | admit |
| `ReplacementMerge` | `src/table.rs:135` | `pub struct ReplacementMerge; impl Merge<i64>` | Value callback only | admit |
| `Table` default `R` | `src/table.rs:557` | `pub struct Table<K, V, H, const N: usize = 16, R = Stored<K>>` | Default admits `Stored<K>` on every face, including hyperbolic, with no address law | narrow |
| `Table::new` | `src/table.rs:571` | `pub fn new(default: V, home: H) -> Self` | Generic over any `H: Home<K>`, `R: KeyRep<K,H>`; establishes no face/profile law | narrow |
| `Table::search_decision_at` | `src/table.rs:595` | `pub fn search_decision_at(&self, sigma: Sigma, key: &K) -> SearchDecision` | Any `(K,H,R)`; `sigma` accepted but `home.address` may ignore it | narrow |
| `Table::get_at` | `src/table.rs:649` | `pub fn get_at(&self, sigma: Sigma, key: &K) -> &V` | Total and borrowed over any `(K,H,R)`; absence uses the summary certificate today | narrow |
| `Table::get` | `src/table.rs:682` | `pub fn get(&self, key: &K) -> &V` | Default face only; same wider carrier/home admission | narrow |
| `Table::len` | `src/table.rs:689` | `pub fn len(&self) -> usize` | No key/face admission | admit |
| `Table::is_empty` | `src/table.rs:694` | `pub fn is_empty(&self) -> bool` | No key/face admission | admit |
| `Table::insert_with_at` | `src/table.rs:1403` | `pub fn insert_with_at<F: Merge<V>>(&mut self, merge: &F, sigma: Sigma, key: K, value: V) -> Result<(), StorageError>` | Any `(K,H,R)`; `R::hold` may drop the key on a many-to-one face | narrow |
| `Table::insert_with` | `src/table.rs:1465` | `pub fn insert_with<F: Merge<V>>(&mut self, merge: &F, key: K, value: V) -> Result<(), StorageError>` | Default face only; same carrier admission | narrow |
| `Table::build_at` | `src/table.rs:1477` | `pub fn build_at<F: Merge<V>>(merge: &F, default: V, kvs: &[((Sigma,K),V)], home: H) -> Result<Self, StorageError>` | Any `(K,H,R)`; partitioned path routes by the same low-byte shard | narrow |
| `Table::build` | `src/table.rs:1593` | `pub fn build<F: Merge<V>>(merge: &F, default: V, kvs: &[(K,V)], home: H) -> Result<Self, StorageError>` | Default face only; same wider carrier/home admission | narrow |
| `KeyRep` trait | `src/table/keyrep.rs:22` | `pub trait KeyRep<K, H>: sealed::Sealed + Clone` | Any `R`; the trait fixes no face/profile and permits a lossy carrier | narrow |
| `KeyRep::hold` | `src/table/keyrep.rs:24` | `fn hold(sigma: Sigma, key: &K, address: &TableAddress) -> Self` | May discard the key with no exact-identity obligation at the trait level | narrow |
| `Stored<K>` | `src/table/keyrep.rs:38` | `pub struct Stored<K> { key: K }` | Retains the full key; can serve any face with a valid address law | admit |
| `KeyRep for Stored` | `src/table/keyrep.rs:44` | `impl<K: Clone + Eq, H: Home<K>> KeyRep<K, H> for Stored<K>` | Full-key identity; no face/profile check | admit |
| `ExactCodec` trait | `src/table/keyrep.rs:80` | `pub trait ExactCodec<K, H>: sealed::Sealed` | Sealed; only reviewed codecs; no profile check | narrow |
| `Regen<C>` | `src/table/keyrep.rs:88` | `pub struct Regen<C> { identity: u64, codec: PhantomData<C> }` | Retains a single identity word; generic over `Sigma`, so it can be selected on a many-to-one face | narrow |
| `KeyRep for Regen` | `src/table/keyrep.rs:104` | `impl<K, H: Home<K>, C: ExactCodec<K,H>> KeyRep<K,H> for Regen<C>` | `matches` compares `C::encode`; `key_owned`/`address_held` call `C::decode`, which the spec forbids as a many-to-one-face identity | narrow |
| `I64WordCodec` | `src/table/keyrep.rs:135` | `pub struct I64WordCodec; impl<H: Home<i64>> ExactCodec<i64,H>` | `encode` ignores `Sigma` and `address`; elliptic `i64` word bijection only | narrow |
| `build_at_parallel` | `src/table/parallel.rs:234` | `pub fn build_at_parallel<F: Merge<V>+Sync>(merge,default,kvs,home,workers) -> Result<Self,StorageError>` | Any `(K,H,R)` with `Sync`/`Send`; same low-byte shard route | narrow |
| `build_at_parallel_phased` | `src/table/parallel.rs:249` | `pub fn build_at_parallel_phased<F: Merge<V>+Sync>(...) -> Result<(Self,PhaseNanos),StorageError>` | Same as above, plus a measurement return | narrow |
| `build_parallel` | `src/table/parallel.rs:372` | `pub fn build_parallel<F: Merge<V>+Sync>(merge,default,kvs,home,workers) -> Result<Self,StorageError>` | Default face only; same carrier/home admission | narrow |
| `build_parallel_phased` | `src/table/parallel.rs:387` | `pub fn build_parallel_phased<F: Merge<V>+Sync>(...) -> Result<(Self,PhaseNanos),StorageError>` | Same as above | narrow |
| `get_each_at_parallel` | `src/table/parallel.rs:415` | `pub fn get_each_at_parallel<T,F>(&self,queries:&[(Sigma,K)],workers,each:F)->Vec<T>` | Any `(K,H,R)` with `Sync`; grouped by the same shard route | narrow |
| `get_each_at_parallel_phased` | `src/table/parallel.rs:433` | `pub fn get_each_at_parallel_phased<T,F>(...) -> (Vec<T>,PhaseNanos)` | Same as above | narrow |
| `get_each_parallel` | `src/table/parallel.rs:545` | `pub fn get_each_parallel<T,F>(&self,queries:&[K],workers,each:F)->Vec<T>` | Default face only | narrow |
| `get_each_parallel_phased` | `src/table/parallel.rs:554` | `pub fn get_each_parallel_phased<T,F>(...) -> (Vec<T>,PhaseNanos)` | Default face only | narrow |
| `sum_parallel` | `src/table/parallel.rs:592` | `pub fn sum_parallel(&self, queries: &[i64], workers: usize) -> i64` | Concrete `Table<i64,i64,H,N,R>`; same admission | narrow |
| `PhaseNanos` | `src/table/parallel.rs:39` | `pub struct PhaseNanos { ... }` | Measurement only; no key/face admission | admit |
| `reservation_failure::arm` | `src/table/parallel.rs:105` | `pub fn arm(shard: usize)` | Thread-local test seam; no key/face admission | admit |
| `Coord` trait | `src/coord.rs:53` | `pub trait Coord { fn coord(&self)->u64; const SPREAD: bool; const CHEAP_EQ: bool; fn coord_with(&self,pepper:u64)->u64 }` | A lossy one-word key encoding; no face/profile; the table's `Home` may call it | narrow |
| `Coord::SPREAD` | `src/coord.rs:71` | `const SPREAD: bool = false` | Only the parabolic face may act on it, but the trait carries no face | narrow |
| `Coord::CHEAP_EQ` | `src/coord.rs:101` | `const CHEAP_EQ: bool = false` | Performance hint; the default is the safe one | narrow |
| `Packed` trait | `src/coord.rs:465` | `pub trait Packed: Sized { fn packed(&self)->u64; fn unpack(c:u64)->Self }` | Bijection on keys that fit a word; the elliptic exact codec | admit |
| `i21` | `src/coord.rs:566` | `pub struct i21(i32)` | Bounded 21-bit signed coordinate for the elliptic triple | narrow |

## 2 · Carrier / REG-1 row ledger

Carrier choice is compile-time (`Stored<K>` / `Regen<C>`). `Sigma` is outside the retained carrier.
REG-1 (`examples/reg1/`) never names `Stored` or `Regen`; every table there uses the default
`R = Stored<StrataKey>` at `examples/reg1/strata.rs:133`, `examples/reg1/matrix.rs:187` and
`examples/reg1/adapters/spatial.rs:56`. `Regen` appears only in `tests/table.rs`,
`src/table/tests.rs`, `src/table/keyrep.rs` and the extraction root `proof/extract/lib.rs:29`.
Therefore the summary deletion's REG-1 blast radius is the `Stored` read/write/probe path only.

### 2.1 Carrier × public operation

<!-- rows:carriers expected=14 -->
| Carrier | Public operation | Exercised by | file:line | Disposition |
|---|---|---|---|---|
| `Stored<K>` (default) | `new`, `get`, `get_at` | `tests/table.rs`, `src/table/tests.rs`, `examples/reg1`, `examples/micro_spill.rs` | `tests/table.rs:80` | retain |
| `Stored<K>` (default) | `insert_with`, `insert_with_at` | `tests/table.rs`, `tests/parallel.rs`, `examples/reg1`, `examples/micro_spill.rs` | `tests/table.rs:87` | retain |
| `Stored<K>` (default) | `build`, `build_at` | `tests/table.rs`, `tests/parallel.rs` | `tests/table.rs:125` | retain |
| `Stored<K>` (default) | `build_parallel`, `get_each_parallel` | `tests/parallel.rs`, `examples/reg1/adapters/spatial.rs` | `tests/parallel.rs:112` | retain |
| `Stored<K>` (default) | `build_at_parallel`, `get_each_at_parallel` | `tests/parallel.rs` | `tests/parallel.rs:174` | retain |
| `Stored<K>` (default) | `sum_parallel` | `tests/parallel.rs` | `tests/parallel.rs:269` | retain |
| `Stored<K>` (default) | `search_decision_at` | `tests/table.rs`, `src/table/tests.rs` | `tests/table.rs:514` | retain |
| `Regen<I64WordCodec>` | `new`, `insert_with_at`, `get_at` (elliptic) | `tests/table.rs`, `src/table/tests.rs` | `tests/table.rs:816` | retain |
| `Regen<I64WordCodec>` | `insert_with_at` at unequal `Sigma` | `tests/table.rs` | `tests/table.rs:843` | migrate |
| `Regen<I64WordCodec>` | relocation rebuild | `tests/table.rs` | `tests/table.rs:1050` | migrate |
| `Regen<PanicDecodeCodec>` | direct-hit no-materialization control | `src/table/keyrep.rs` | `src/table/keyrep.rs:181` | retain |
| `Regen<C>` | extraction root (elliptic word) | `proof/extract/lib.rs` | `proof/extract/lib.rs:30` | migrate |
| `Regen<C>` | `build`, `build_at`, `*_parallel` | not exercised | `src/table/keyrep.rs:104` | retain |
| `Stored`/`Regen` | `KeyRep::address_held` relocation | `src/table.rs` | `src/table.rs:848` | migrate |

### 2.2 REG-1 rows and their public operations

The speed strata are the product `OCCUPANCIES × Distribution::ALL × Operation::ALL × MergeMode`
= `5 × 4 × 5 × 2` = 200 cells (`examples/reg1/strata.rs:58`, `examples/reg1/strata.rs:180`,
`examples/reg1/strata.rs:217`, `examples/reg1/strata.rs:242`). `PresentHit`/`AbsentMiss` call `get`
(`examples/reg1/strata.rs:688`); `ExistingUpdate`/`AdmittedFreshInsert`/`GrowthInsert` call
`insert_with` (`examples/reg1/strata.rs:693`); every cell is built by `new` + `insert_with`
(`examples/reg1/strata.rs:639`). The parallel adapter calls `build_parallel` + `get_each_parallel`
(`examples/reg1/adapters/spatial.rs:283`, `examples/reg1/adapters/spatial.rs:286`).

<!-- rows:carriers_reg1 expected=8 -->
| REG-1 row | Public operation | file:line | Disposition |
|---|---|---|---|
| 200 speed cells `occ{1,2,4,8,16}-{direct-index-resident,offset-beyond-direct,seeded-random,clustered-skewed}-{present-hit,absent-miss,existing-update,admitted-fresh-insert,growth-insert}-{replacement,wrapping-add}` | `new`, `get`, `insert_with` | `examples/reg1/strata.rs:251` | migrate |
| `zero-group-width-sizeoverflow` resource cell | `new`, `get`, `insert_with` (`N = 0`) | `examples/reg1/matrix.rs:886` | retain |
| `sizing-overflow-not-forceable` resource cell | registered `missing` | `examples/reg1/matrix.rs:924` | retain |
| `injected-reservation-failure-not-forceable` resource cell | registered `missing` | `examples/reg1/matrix.rs:936` | retain |
| `current_api` rows `new`,`get`,`insert_with`,`build` | the admitted `apeiron::Table` surface | `examples/reg1/coverage.json:41` | migrate |
| bootstrap identity witness | `new`, `insert_with`, `get` | `examples/reg1/workload.rs:148` | migrate |
| parallel spatial adapter | `build_parallel`, `get_each_parallel` | `examples/reg1/adapters/spatial.rs:283` | migrate |
| registered comparison matrix (`matrix.json`) | all of the above | `examples/reg1/matrix.rs:962` | retain |

## 3 · `PHI` / summary consumer census

Two distinct objects share the word "summary":

- `Region.summary`, the per-region Bloom filter over `place >> 8`, with `summary_index`,
  `summary_set`, `summary_maybe`, `summary_words`, `summary_trailing_zeros`, the
  `RegionReservation.summary` capacity bound, the `publish_live` summary write and every
  `summary_maybe` absence branch. **Delete.**
- The shifted seven-bit tag fingerprint `tag()` (`src/table/state.rs:315`), `TAG_LIVE`
  (`src/table.rs:38`), `TAG_FINGERPRINT_SHIFT` (`src/table.rs:47`), `TAG_FINGERPRINT_MASK`
  (`src/table.rs:49`) and `scan::match_tag` (`src/table/scan.rs:77`). **Retain.** The spec names the
  obligation `SpillFingerprintExact`; it has **no** occurrence in `src/`, `tests/`, `examples/` or
  `proof/` (only in the spec's "Summary deletion" table).

`coord::PHI` (`src/coord.rs:50`) is independent of the summary and is **retained**: the parabolic
fold and the elliptic codec use it. The literal `0x9E37_79B9_7F4A_7C15` inside `summary_index`
(`src/table/state.rs:140`) is the summary's own mixer and is deleted with the summary.

### 3.1 Region summary core

<!-- rows:summary_core expected=14 -->
| Reference | file:line | Role | Disposition |
|---|---|---|---|
| `SUMMARY_BITS_PER_SLOT` | `src/table/state.rs:7` | summary sizing constant | delete |
| `SUMMARY_MIN_BITS` | `src/table/state.rs:9` | summary sizing constant | delete |
| `summary_trailing_zeros` | `src/table/state.rs:17` | summary bit-count loop | delete |
| `summary_words` | `src/table/state.rs:34` | summary word sizing | delete |
| `Region.summary` field | `src/table/state.rs:95` | Bloom storage | delete |
| `Region::empty` summary init | `src/table/state.rs:102` | `summary: vec![]` | delete |
| `summary_index` | `src/table/state.rs:127` | Bloom bit index | delete |
| summary mixer literal | `src/table/state.rs:140` | `0x9E37_79B9_7F4A_7C15` inside `summary_index` | delete |
| `summary_set` | `src/table/state.rs:147` | Bloom write | delete |
| `summary_maybe` | `src/table/state.rs:161` | Bloom absence filter | delete |
| `publish_live` summary write | `src/table/state.rs:179` | calls `summary_set` before the tag | delete |
| `reserve_empty` summary sizing | `src/table/state.rs:191` | `summary_words(n)` reservation | delete |
| `reserve_empty` summary resize | `src/table/state.rs:199` | `summary.resize(words, 0)` | delete |
| `SpillSummaryShape`/`SpillSummaryMissSound` comment | `src/table/state.rs:158` | names the mooted obligations | delete |

### 3.2 Read / write / probe summary branches

<!-- rows:summary_paths expected=13 -->
| Reference | file:line | Role | Disposition |
|---|---|---|---|
| `spill_locate` `summary_maybe` | `src/table.rs:171` | write-path absence short-circuit | delete |
| `spill_read` `summary_maybe` | `src/table.rs:358` | read-path absence short-circuit | delete |
| `spill_locate_held` `summary_maybe` | `src/table.rs:415` | relocation absence short-circuit | delete |
| `read_trace::SUMMARY_MISSES` | `src/table.rs:212` | test-only summary counter | delete |
| `read_trace::record_summary_miss` | `src/table.rs:223` | test-only summary counter | delete |
| `read_trace::summary_misses` | `src/table.rs:233` | test-only summary counter | delete |
| `ReadStats.summary_tests`/`summary_hits`/`summary_misses` | `src/table.rs:258` | `read-stats` stage counters | delete |
| `read_stats::record_summary` | `src/table.rs:288` | `read-stats` stage counter | delete |
| `observe_receipt` summary doc | `src/table.rs:994` | receipt description | delete |
| `observe_receipt` `summary_required` | `src/table.rs:1023` | checked summary target | delete |
| `observe_receipt` summary bound | `src/table.rs:1027` | `capacity_bound(summary_required, ...)` | delete |
| `RegionReservation.summary` | `src/table/resources.rs:27` | receipt summary bound | delete |
| `publication_trace::SUMMARY` | `src/table/state.rs:61` | test-only publication step | delete |

### 3.3 Summary tests

<!-- rows:summary_tests expected=13 -->
| Reference | file:line | Role | Disposition |
|---|---|---|---|
| `summary_trailing_zeros_matches_the_standard_intrinsic` | `src/table/tests.rs:606` | summary helper test | delete |
| `zero_length_summary_is_total_and_disables_filtering` | `src/table/tests.rs:626` | summary totality test | delete |
| `malformed_summary_lengths_are_total` | `src/table/tests.rs:652` | summary totality test | delete |
| `every_live_spill_tag_has_its_summary_bit_recorded` | `src/table/tests.rs:686` | summary recording test | delete |
| `spill_publication_order_is_payload_then_summary_then_tag` | `src/table/tests.rs:722` | publication-order test | delete |
| `read_absence_summary_miss_certificate_skips_the_sweep` | `src/table/tests.rs:846` | summary-miss certificate test | delete |
| `read_absence_coverage_certificate_forces_complete_sweep` | `src/table/tests.rs:890` | forced-collision coverage test | delete |
| `read_forced_summary_collision_has_no_false_negative` | `src/table/tests.rs:926` | forced-collision test | delete |
| `summary_sizing_overflow_refuses_before_any_hold_or_clone` | `src/table/tests.rs:523` | summary-refusal injection | delete |
| `prepared_adapter_materializes_vacant_storage...` receipt asserts | `src/table/tests.rs:474` | summary receipt length | delete |
| `read_clones_and_merges_nothing...` summary capacity | `src/table/tests.rs:1021` | read-invariance snapshot | delete |
| `read_finds_a_late_spill_hit...` summary zeroing | `src/table/tests.rs:967` | fixture reset | delete |
| `spill_sweep_finds_a_late_match...` `summary_set` | `src/table/tests.rs:137` | fixture publication | delete |

### 3.4 Generated proof / extraction summary dependencies

`proof/ApeironWord.lean` is regenerated by `proof/regen.sh`; its summary definitions and call sites
disappear when the Rust source is edited. `proof/corpus/Growable/Scan.lean` and
`proof/corpus/AxiomAudit.lean` are hand-carried and must be edited/deleted.

<!-- rows:summary_proof expected=7 -->
| Reference | file:line | Role | Disposition |
|---|---|---|---|
| `table.summary_index` | `proof/ApeironWord.lean:1568` | generated extraction | regenerate |
| `table.summary_set` | `proof/ApeironWord.lean:1627` | generated extraction | regenerate |
| `table.summary_maybe` | `proof/ApeironWord.lean:1648` | generated extraction | regenerate |
| `table.summary_trailing_zeros` | `proof/ApeironWord.lean:1563` | generated extraction | regenerate |
| `table.Region.publish_live` summary call | `proof/ApeironWord.lean:1697` | generated extraction | regenerate |
| `Apeiron.GrowableScan.summary_trailing_zeros_eq` | `proof/corpus/Growable/Scan.lean:36` | hand-carried summary theorem | delete |
| `#audit Apeiron.GrowableScan.summary_trailing_zeros_eq` | `proof/corpus/AxiomAudit.lean:181` | audit registration | delete |

### 3.5 Retained `PHI` / fingerprint consumers

<!-- rows:summary_retain expected=11 -->
| Reference | file:line | Role | Disposition |
|---|---|---|---|
| `coord::PHI` | `src/coord.rs:50` | golden-ratio odd multiplier, parabolic fold and elliptic codec | retain |
| `coord::spread` | `src/coord.rs:357` | identity/placement split | retain |
| `coord::unspread` | `src/coord.rs:376` | elliptic key regeneration | retain |
| `I64WordCodec::encode` | `src/table/keyrep.rs:140` | elliptic exact identity | retain |
| `I64WordCodec::decode` | `src/table/keyrep.rs:146` | elliptic exact inverse | retain |
| `coord::mix` | `src/coord.rs:440` | parabolic multi-word fold | retain |
| `tag()` | `src/table/state.rs:315` | shifted seven-bit fingerprint | retain |
| `TAG_LIVE` | `src/table.rs:38` | spill occupancy bit | retain |
| `TAG_FINGERPRINT_SHIFT` | `src/table.rs:47` | fingerprint shift | retain |
| `TAG_FINGERPRINT_MASK` | `src/table.rs:49` | fingerprint mask | retain |
| `scan::match_tag` | `src/table/scan.rs:77` | fingerprint equality filter | retain |

`SpillFingerprintExact` itself has no `src/` site; the retained obligation is the fingerprint
implementation above. Its spec row is the `SpillFingerprintExact` row of the spec's "Summary
deletion" table.

## 4 · Free-placement census (retire on the hyperbolic face only)

These sites make a low-byte or load-rounded power-of-two word the final routing authority. The
successor replaces them with a bounded physical locator on the hyperbolic face; they are retained on
elliptic/parabolic until that locator exists. `separate-locator` marks a site that must become a
locator-specific certificate rather than a cast of an address value. `delete-with-summary` marks the
summary's own mask.

<!-- rows:free_placement expected=26 -->
| Site | file:line | Role | Disposition |
|---|---|---|---|
| `shard()` selector | `src/table/state.rs:303` | `direct_word & 255` or `place & 255` picks one of 256 regions | retire-hyperbolic |
| `shard()` direct branch | `src/table/state.rs:305` | `(word as usize) & 255` | retire-hyperbolic |
| `shard()` place branch | `src/table/state.rs:306` | `(address.place as usize) & 255` | retire-hyperbolic |
| `group()` descent | `src/table/state.rs:357` | `(place >> 8) & DESCENT_MASK` group walk | retire-hyperbolic |
| `group()` window mask | `src/table/state.rs:360` | `WINDOW_MASK` / `mask` group selection | retire-hyperbolic |
| `group()` sweep wrap | `src/table/state.rs:362` | `(d as usize) & mask` cyclic coverage | retire-hyperbolic |
| `DESCENT_MASK` | `src/table.rs:50` | 49-bit descent slice | retire-hyperbolic |
| `WINDOW_MASK` | `src/table.rs:51` | seven-bit window | retire-hyperbolic |
| `WINDOWS` | `src/table.rs:52` | fixed window count | retire-hyperbolic |
| `spill_read` power-of-two groups | `src/table.rs:357` | `groups.is_power_of_two()` gate | retire-hyperbolic |
| `spill_locate` power-of-two groups | `src/table.rs:162` | `groups.is_power_of_two()` gate | retire-hyperbolic |
| `spill_locate_held` power-of-two groups | `src/table.rs:410` | `groups.is_power_of_two()` gate | retire-hyperbolic |
| `slots_for` load rounding | `src/table/state.rs:328` | `load * 2` rounded up to a power-of-two group count | retire-hyperbolic |
| `slots_for` doubling | `src/table/state.rs:336` | `groups.checked_mul(2)` | retire-hyperbolic |
| `valid_n` | `src/table/state.rs:295` | group width must be a power of two | retire-hyperbolic |
| `flat_index_for` | `src/table.rs:1107` | `word >> 8` lane and `word & 255` shard | retire-hyperbolic |
| `Flat::index` | `src/table/state.rs:444` | `word >> 8` lane and `word & 255` shard | retire-hyperbolic |
| `insert_with_at` lane | `src/table.rs:1415` | `(word >> 8) as usize` | retire-hyperbolic |
| `byte_len_for` | `src/table/state.rs:456` | `lanes * 256` byte budget | retire-hyperbolic |
| `Flat::reserve`/`initialize` | `src/table/state.rs:465` | `lanes * 256` slot count | retire-hyperbolic |
| `density_at` bitmap | `src/table.rs:712` | `lanes * 256` density limit | retire-hyperbolic |
| `selected_lanes` budget | `src/table.rs:768` | power-of-two lane doubling to `1 << MAX_LANE_BITS` | retire-hyperbolic |
| `MAX_LANE_BITS` | `src/table.rs:30` | 24-bit lane ceiling | retire-hyperbolic |
| `SHARDS` | `src/table/partition.rs:14` | 256-entry physical directory | retain |
| `summary_index` mask | `src/table/state.rs:131` | summary power-of-two mask | delete-with-summary |
| `direct_word` certificate | `src/table.rs:71` | `Option<u32>` resident-admission only | separate-locator |

## 5 · Completeness check

The check executes from the repository root. It (a) resolves every backticked `path:line` citation
whose path starts with `src/`, `tests/`, `examples/`, `proof/`, `docs/` or `Cargo`, (b) verifies
each `<!-- rows:KEY expected=N -->` marker's table row count equals `N`, and (c) requires every
census row's final cell to begin with a declared disposition token. A missing citation target, an
out-of-range line, a wrong row count, or an unknown disposition exits nonzero.

```sh
python3 - docs/evidence/anative-admission-ledger.md <<'PY'
import re, sys, pathlib

ledger = pathlib.Path(sys.argv[1])
root = pathlib.Path.cwd()
text = ledger.read_text()
lines = text.splitlines()

allowed = {
    "admit", "narrow", "block",
    "retain", "delete", "migrate", "regenerate", "none",
    "retire-hyperbolic", "delete-with-summary", "separate-locator",
}

cite = re.compile(r'`([A-Za-z0-9_./-]+):(\d+)`')
prefixes = ("src/", "tests/", "examples/", "proof/", "docs/", "Cargo")
bad = []
resolved = 0
for m in cite.finditer(text):
    path, line = m.group(1), int(m.group(2))
    if not path.startswith(prefixes):
        continue
    target = root / path
    if not target.is_file():
        bad.append(f"missing citation target {path} (cited {path}:{line})")
        continue
    n = len(target.read_text(errors="replace").splitlines())
    resolved += 1
    if line < 1 or line > n:
        bad.append(f"line out of range {path}:{line} (file has {n} lines)")

marker = re.compile(r'<!-- rows:([a-z0-9_]+) expected=(\d+) -->')
sep = re.compile(r'^:?-{2,}:?$')
counts = {}
dispositions = {}
i = 0
while i < len(lines):
    m = marker.search(lines[i])
    if not m:
        i += 1
        continue
    key, expected = m.group(1), int(m.group(2))
    j = i + 1
    block = []
    while j < len(lines) and lines[j].startswith("|"):
        block.append((j + 1, [c.strip() for c in lines[j].strip().strip("|").split("|")]))
        j += 1
    # Drop the header row (immediately before the first separator) and the separator itself.
    first_sep = next((k for k, (_, cells) in enumerate(block) if all(sep.match(c) for c in cells)), None)
    if first_sep is not None:
        block = [r for k, r in enumerate(block) if k != first_sep and k != first_sep - 1]
    counts[key] = len(block)
    if len(block) != expected:
        bad.append(f"section {key}: {len(block)} rows != expected {expected}")
    for lineno, cells in block:
        last = cells[-1].lower()
        token = re.split(r'[ (\u2014:.,;]', last, maxsplit=1)[0]
        if token not in allowed:
            bad.append(f"line {lineno}: disposition {cells[-1]!r} not allowed")
        else:
            dispositions[token] = dispositions.get(token, 0) + 1
    i = j

print(f"citations_resolved={resolved}")
for k in sorted(counts):
    print(f"rows_{k}={counts[k]}")
print("dispositions=" + ",".join(f"{k}={v}" for k, v in sorted(dispositions.items())))
if bad:
    print("FAIL")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("PASS")
PY
```

### 5.1 Gate result

Run on this file: `citations_resolved=173`; row counts `generic_api=45`, `carriers=14`,
`carriers_reg1=8`, `summary_core=14`, `summary_paths=13`, `summary_tests=13`, `summary_proof=7`,
`summary_retain=11`, `free_placement=26`; dispositions
`admit=11, block=1, delete=42, delete-with-summary=1, migrate=8, narrow=33, regenerate=5,
retain=26, retire-hyperbolic=23, separate-locator=1`; `PASS` (exit 0).

### 5.2 Rejecting control

Copy this file to a throwaway path, delete exactly one census row, run the same check. The row count
for that marker no longer matches `expected`, so the check exits nonzero. The working tree is not
modified. Control: `/tmp/opencode/anative-ledger-control.md` with the `summary_core` row
`summary_words` removed → `section summary_core: 13 rows != expected 14` → exit 1.

## 6 · Residual obligations

- Every disposition is a planning decision, not a proof or a measured result. The successor
  `FaceRepR` headlines and the per-face absence certificates remain absent proof targets.
- The ledger records the current dirty working tree; `proof/regen.sh --check`, `cargo kani` and
  `lake build` did not run, so no Aeneas/Kani/Lean receipt is fresh.
- `SpillFingerprintExact` is retained but has no `src/` implementation to cite; the retained sites
  are the fingerprint implementation in §3.5. The obligation itself lives in the spec.
- The exact finite hyperbolic locator and its bounds/coverage/cost theorem are not selected here;
  §4 only names the sites that must be replaced.
- `docs/spec/algebra-native-addressing.md` is under concurrent edit (it grew from 288 to 306 lines
  during this census), so the spec is cited by section/row name and not by line; the `src/`, `tests/`
  and `examples/` citations bind to the digests in §0.
