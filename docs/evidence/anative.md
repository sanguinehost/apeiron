# Algebra-native addressing evidence (rolled up)

Rolled up from: `evidence/anative-admission-ledger.md`, `evidence/anative-extraction-preset-probe.md`, `evidence/anative-w4-e-i7-extraction.md`.

## Evidence — algebra-native addressing admission and retirement ledger

- **Task:** `APEIRON-ANATIVE-W0-F1`, census, READ-ONLY on `src/`.
- **Spec:** `docs/epics/algebra-native-addressing.md` (per-face address contract, summary deletion).
- **Base revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied checkout is dirty.
- **Grade:** `tested` for the completeness check below (it executes and its rejecting control is red).
  Every disposition is a planning decision at grade `anchor`; no Lean theorem, build or benchmark
  ran here.
- **Scope:** this file only. No `src/`, `tests/`, `proof/`, spec/plan/intent or `Cargo.toml` edit.

### 0 · Provenance and source digests

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

#### 0.1 `rg -n` counts recorded

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

### 1 · Generic-API admission ledger

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

### 2 · Carrier / REG-1 row ledger

Carrier choice is compile-time (`Stored<K>` / `Regen<C>`). `Sigma` is outside the retained carrier.
REG-1 (`examples/reg1/`) never names `Stored` or `Regen`; every table there uses the default
`R = Stored<StrataKey>` at `examples/reg1/strata.rs:133`, `examples/reg1/matrix.rs:187` and
`examples/reg1/adapters/spatial.rs:56`. `Regen` appears only in `tests/table.rs`,
`src/table/tests.rs`, `src/table/keyrep.rs` and the extraction root `proof/extract/lib.rs:29`.
Therefore the summary deletion's REG-1 blast radius is the `Stored` read/write/probe path only.

#### 2.1 Carrier × public operation

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

#### 2.2 REG-1 rows and their public operations

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

### 3 · `PHI` / summary consumer census

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

#### 3.1 Region summary core

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

#### 3.2 Read / write / probe summary branches

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

#### 3.3 Summary tests

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

#### 3.4 Generated proof / extraction summary dependencies

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

#### 3.5 Retained `PHI` / fingerprint consumers

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

### 4 · Free-placement census (retire on the hyperbolic face only)

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

### 5 · Completeness check

The check executes from the repository root. It (a) resolves every backticked `path:line` citation
whose path starts with `src/`, `tests/`, `examples/`, `proof/`, `docs/` or `Cargo`, (b) verifies
each `<!-- rows:KEY expected=N -->` marker's table row count equals `N`, and (c) requires every
census row's final cell to begin with a declared disposition token. A missing citation target, an
out-of-range line, a wrong row count, or an unknown disposition exits nonzero.

```sh
python3 - docs/evidence/anative.md <<'PY'
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

#### 5.1 Gate result

Run on this file: `citations_resolved=173`; row counts `generic_api=45`, `carriers=14`,
`carriers_reg1=8`, `summary_core=14`, `summary_paths=13`, `summary_tests=13`, `summary_proof=7`,
`summary_retain=11`, `free_placement=26`; dispositions
`admit=11, block=1, delete=42, delete-with-summary=1, migrate=8, narrow=33, regenerate=5,
retain=26, retire-hyperbolic=23, separate-locator=1`; `PASS` (exit 0).

#### 5.2 Rejecting control

Copy this file to a throwaway path, delete exactly one census row, run the same check. The row count
for that marker no longer matches `expected`, so the check exits nonzero. The working tree is not
modified. Control: `/tmp/opencode/anative-ledger-control.md` with the `summary_core` row
`summary_words` removed → `section summary_core: 13 rows != expected 14` → exit 1.

### 6 · Residual obligations

- Every disposition is a planning decision, not a proof or a measured result. The successor
  `FaceRepR` headlines and the per-face absence certificates remain absent proof targets.
- The ledger records the current dirty working tree; `proof/regen.sh --check`, `cargo kani` and
  `lake build` did not run, so no Aeneas/Kani/Lean receipt is fresh.
- `SpillFingerprintExact` is retained but has no `src/` implementation to cite; the retained sites
  are the fingerprint implementation in §3.5. The obligation itself lives in the spec.
- The exact finite hyperbolic locator and its bounds/coverage/cost theorem are not selected here;
  §4 only names the sites that must be replaced.
- `docs/epics/algebra-native-addressing.md` is under concurrent edit (it grew from 288 to 306 lines
  during this census), so the spec is cited by section/row name and not by line; the `src/`, `tests/`
  and `examples/` citations bind to the digests in §0.

## Extraction probe — `--preset aeneas` on the current source

- **Task:** `APEIRON-ANATIVE-SPIKE-PRESET` (extraction feasibility probe)
- **Revision:** `c450c222905c050f793a23332708c230c658f82c`; working tree dirty
- **Grade:** `anchor` — a probe receipt. The generated file carries 23 axioms and is **not** an
  accepted extraction. No source, governing document or committed artifact was modified.

### Question

Does the current source (per-face address rings, scalar-arena locator, growable table) translate
under the reproducible historical Charon configuration `--preset aeneas`, or does the failure that
motivated the explicit monomorphizing option set (`APEIRON-ANATIVE-AEN-PIPE`) persist?

### Result

`--preset aeneas` **succeeds structurally** on the current source. The explicit monomorphizing
option set is the cause of the supplied red, not the locator or a generic-binder gap.

| Stage | Command (throwaway output) | Exit | Result |
|---|---|---|---|
| Charon | `charon rustc --preset aeneas --dest-file /tmp/.../preset.llbc <43 --start-from roots> -- --edition 2021 --crate-type lib --crate-name apeiron_word proof/extract/lib.rs` | 0 | `preset.llbc` (11 983 294 bytes) |
| Aeneas | `aeneas -backend lean -dest /tmp/.../out -warnings-as-errors -no-progress-bar /tmp/.../preset.llbc` | 0 | `out/Preset.lean`, 525 467 bytes, 24.4 s |

Generated content: **450 top-level definitions, 23 axioms.** The real production bodies are present,
including `table.Table.get_at`, `table.Table.reserve_rebuild`, `table.Table.materialize_rebuild`,
`table.Table.insert_with_at`, and the scalar-arena locator (`locator_core` referenced 878 times,
`Directory` 237, `reserve_for` 51, `walk_residue` 5). There is no `type_var_id` failure and no
`TArray`/`TSlice` failure under the preset.

### Why the explicit option set failed

The option set of `APEIRON-ANATIVE-AEN-PIPE` omits `--ops-to-function-calls`,
`--index-to-function-calls` and `--no-gen-tuple-structs`. Those three are load-bearing for **every**
array/slice index and array repeat in the tree, not only for the locator. The recorded E1 first
errors are all of this one class:

| Error | Site (real source) |
|---|---|
| `Could not find core::array::repeat` | `src/coord/locator_core.rs:1024` (`[0u32; MAX_LEVELS + 1]`), `src/table/scan.rs:65`, `src/table/partition.rs:51` |
| `expand_symbolic_value_no_branching: unexpected type: TSlice` | `src/coord/locator_core.rs:657` (`words[index]`), `:662` (store), `:159`, `:395`, `:414`; `src/table.rs:212`, `:246`, `:260` |
| `expand_symbolic_value_no_branching: unexpected type: TArray` | `src/table/partition.rs:72`, `:78`, `:104`; `src/table.rs:1495` |

These are pre-existing production sites, including the historically accepted word kernels; the
candidate configuration breaks array lowering globally. The spec's own prediction for
`index_to_function_calls` ("unsupported indexing blocks the candidate") is confirmed.

### The remaining 23 axioms

Aeneas models `Vec` as a functional `Slice`-backed value (`Vec.new`/`push`/`index_usize`/`set`/`len`;
`alloc.vec.Vec.with_capacity` ignores capacity). It does not model reservation, capacity or the std
combinators the growable table and resource path reach, so it emits an `axiom` per external.

| Class | Axioms | Source family |
|---|---|---|
| Vec reservation/growth | `alloc.vec.Vec.capacity`, `try_reserve`, `try_reserve_exact`, `truncate`, `clear`, `is_empty`, `alloc.collections.TryReserveError` | `src/table/resources.rs`, `src/coord/locator_core.rs` rollback/materialize |
| allocator glue | `core.mem.size_of`, `core.mem.drop` | `Vec` growth |
| integer conversion | `Usize…TryFromU64…try_from`, `Usize…TryFromU32…try_from` | `u32::try_from`/`usize::try_from` sites |
| `Option` combinators | `is_none_or`, `as_ref`, `and_then`, `filter`, `OptionShared0T.copied`, `Try.branch` | table/probe |
| `Result` combinators | `is_err`, `ok`, `and_then` | table |
| `usize` arithmetic | `saturating_mul`, `div_ceil`, `checked_next_power_of_two` | `src/table/state.rs` (`slots_for`), `src/table/partition.rs` |

No `sorry`/`admit`; the 23 are the fail-closed audit target. The historical accepted
`proof/ApeironWord.lean` (HEAD, 1029 lines; worktree, 3511 lines) has **zero** axioms, so the
current reach set is what introduces these.

### Implication for the direction

1. The reproducible `--preset aeneas` configuration must be restored; the explicit monomorphizing
   candidate is rejected by its own gate. This removes the rationale for a Charon
   monomorphization/preset bypass.
2. The dominant axiom class is the **growable-table resource machinery** (reserve/capacity/truncate),
   which the canonical fixed-shard object does not have: refusal on a full fiber needs no growth.
   This is independent, tool-level support for the quotient principle's fixed-shard bounded-fiber
   realization — it retires the largest, least-modelable axiom class rather than modelling it.
3. The remaining classes (`TryFrom`, `Option`/`Result` combinators, `usize` ops) are removable by
   source reshaping to explicit matches and arithmetic, or by a checked model.
4. The scalar-arena locator itself translates under the preset, so "retire the trie" is an algebraic
   and cost decision, **not** an extraction prerequisite. The corrected claim is narrower than the
   handoff's.

### Digests

| Input | SHA-256 |
|---|---|
| `src/coord/locator_core.rs` | `b4699816521d89623988e4e92ee687b33d7243b3a70634bfbff1c8cd7e00eee8` |
| `src/coord/address.rs` | `a8c5c08164c1c007a72e3089f964cb2cc2167f864f76414e641c2b9e563c70e6` |
| `src/table.rs` | `b253c60f442939a6553800b1b68cf05fc73db1947db491272d280f4a18e9baeb` |
| `proof/regen.sh` | `d39c7ef7f120b1ef118922c0a735670bde6c43cb1a942f8a4053a9c70e7d4600` |
| `proof/extract/lib.rs` | `94cc97cabe97d790b669dd44958025735694bd6a2c60b9fa7849cedba2b7e5b1` |
| `target/aeneas/charon` | `747e165bc62bb963f8d741e8c193c721a4b94555bcb1f9f86c79090fd7a59d0e` |
| `target/aeneas/charon-driver` | `207b223af35cf9a05227b329819b73532e94f655ccdefd64e77b755a15fdf297` |
| `target/aeneas/aeneas` | `33d0ce1a1cc2f1d3935c0cd2ab34f283797da61c531ed6d4e09d5c97b0d0228a` |
| generated `out/Preset.lean` | `27202a968e268a5172fd4fc270d857d3ec3d4633af07a661e852da239c673892` |
| `aeneas.log` | `dfd55e4f5c3948d472dca8e24d4def7a193711b68b37934a2d9e27b49b6845b6` |
| `charon.log` | `e6e5144892651085ecd93452f1a2db1c369254d0d8aa9cfbaeb0ed297c8c2756` |

Tools: charon `0.1.254 (b104e24fea7d721b71e6c39fd70f26ff20bc0980)`, aeneas `505b6ca-dirty`.

### Exact axiom-to-site map (reached source)

`-split-files` puts all 23 in `FunsExternal_Template.lean` / `TypesExternal_Template.lean`, so a
model file is the intended hook. Sites below are the reached uses.

| Axiom | Reached sites |
|---|---|
| `Vec.try_reserve` | `coord/locator_core.rs:951,954,956,1133`; `coord/locator.rs:677`; `table/locator.rs:422` |
| `Vec.try_reserve_exact` | `table/resources.rs:198-203` (central helper); `coord/locator_core.rs:224,963` |
| `Vec.capacity` | `table.rs:876,877,903,907,911`; `table/locator.rs:424` |
| `Vec.truncate` | `coord/locator_core.rs:1159-1161` |
| `Vec.clear` | `coord/locator_core.rs:1060,1067`; `table/state.rs:71` |
| `Vec.is_empty` | `coord/locator_core.rs:1002`; `coord/locator.rs:415`; `table/locator.rs:41`; `table/partition.rs:38,67` |
| `TryReserveError` | error type of the above |
| `core.mem.size_of` | `table/keyrep.rs:297,332` (`size_of::<u64>`); `table/state.rs:459` (`size_of::<FlatSlot<R,V>>`) |
| `Option.is_none_or` | `table.rs:840` |
| `Option.as_ref` | `table/locator.rs:564,605` |
| `Option.and_then` | `coord/locator_core.rs:923,924,943`; `table.rs:1501`; `table/parallel.rs:201` |
| `Option.filter` | `table.rs:804,825` |
| `Option.copied` | `table.rs:366`; `table/parallel.rs:151`; `table/locator.rs:155` |
| `Try.branch` | residual `?` conversion on `Option` (one site) |
| `Result.is_err` | `coord/locator_core.rs:935-937,951,955,956,1133` |
| `Result.ok` | `table.rs:1034`; `table/locator.rs:118` |
| `usize::try_from` | `table.rs:606,1029,1033`; `table/locator.rs:99,112` |
| `u32::try_from` | `coord/locator_core.rs:515,519,935-937,1040,1141`; `coord/locator.rs:486,495,537,674`; `table/locator.rs:93,118,420` |
| `Usize.saturating_mul` | `table.rs:680` |
| `Usize.div_ceil` | `table.rs:607,1483`; `table/state.rs:43` |
| `checked_next_power_of_two` | `table/state.rs:44` |
| `core.mem.drop` | drop glue on owned `R`/`V` in `Region`/`Flat` |

Classes: **allocation/growth** (the first eight rows) is the fixed-shard object's removed machinery;
**combinators/conversions** (the rest) is exact-match reshaping.

### Reshape round 1 — pure combinator/conversion class removed (23 → 11)

Source reshaping replaced the unmodeled **pure** operations with exact equivalents. All Rust gates
pass on this state: `cargo test --workspace --all-features --locked` (258 passed, 0 failed),
`cargo clippy … -D warnings` exit 0, `cargo fmt --all -- --check` exit 0. Re-extraction under
`--preset aeneas` translates (charon exit 0, aeneas exit 0) with **444 defs and 11 axioms**,
generated file sha `b3644dbf789ce6a2e9d72a0d44c3177642fa3efe00b973eaa623c7b30519d5e3`.

Removed (12 axiom names): `Usize…TryFromU64…try_from`, `Usize…TryFromU32…try_from`,
`core.num.Usize.saturating_mul`, `core.num.Usize.div_ceil`,
`core.num.Usize.checked_next_power_of_two`, `core.option.Option.is_none_or`,
`core.option.Option.and_then`, `core.option.Option.filter`, `OptionShared0T.copied`,
`core.result.Result.ok`, `core.result.Result.and_then`, `alloc.vec.Vec.is_empty`.

Mechanisms: `x / n + usize::from(x % n != 0)` for `div_ceil`; an explicit doubling loop for
`checked_next_power_of_two`; `checked_mul(k)` + fallback for `saturating_mul`; `(x as usize)` after
a `> u32::MAX as usize` guard for `try_from`; explicit `match`/`if` for the `Option`/`Result`
combinators and iterator `first().copied()`; `Option` `?` replaced by `match`; a named
`Table::keep_unoccupied` for `Option::filter` so the body stays out of the joined loop context.
Extractor-shaped forms are kept and the corresponding clippy lints are allowed, with a reason, at
the top of `src/table.rs`, `src/table/parallel.rs`, `src/table/partition.rs`,
`src/coord/locator.rs` and `src/coord/locator_core.rs`.

#### Aeneas shape is load-bearing

The original source was deliberately shaped for the extractor, so a semantically exact replacement
is not automatically translatable. Three naive replacements regressed translation and were repaired
by restoring the single-exit form:

| Change | Aeneas failure | Repair |
|---|---|---|
| `let slots = slots_for(..)?;` inside `build_at`'s `while` loop | `Early returns inside of loops` | restored the in-expression `match` (no `?` in the loop) |
| guarded `match flat { Some(i) if !occupied[i] => .. }` inside the nested planning loop | `Could not match the contexts` | named `keep_unoccupied` call (opaque body, like the original closure) |
| `match &self.directory` for `Option::as_ref` | macro-origin `Not implemented yet` in `linked_handle` | restored `self.directory.as_ref()?` |

This confirms the class the earlier bridge hit: loop-context joins, in-loop early returns and inline
guards are the extractor's shape boundaries, independent of the algebra.

### Residual — 11 axioms (Path 2b)

| Axiom | Class |
|---|---|
| `Vec.capacity`, `try_reserve`, `try_reserve_exact`, `truncate`, `clear`, `TryReserveError`, `core.mem.size_of`, `core.mem.drop`, `Result.is_err` | **allocator/growth** — removed by the fixed-shard, bounded-fiber realization (no reserve, no capacity, no growth, no rollback truncation) |
| `Option.as_ref`, `Option.…Try.branch` | the `self.directory.as_ref()?` site in `TableLocator::linked_handle`/`linked_handle_query`; a `&self.directory` match regressed translation, so it stays until the re-founding removes the field |

The allocator class is not reducible by exact substitution: Aeneas models `Vec` as a functional
sequence with no reservation, capacity or length-shrink. Removing it is the storage re-founding.

### Reshape digests (round 1)

| Input | SHA-256 |
|---|---|
| `src/table.rs` | `9ec504cec3d244683936779e2e07f5d83aafcb9b8ae8e9ab2043ec89e55ac548` |
| `src/table/locator.rs` | `72c9d11ce5699c8ad1cd9ab35d855416617bd2bf8a646248bccb2e2bb7ce7c83` |
| `src/table/parallel.rs` | `08f42f76867e80e8ca4696e045b84461a08c8e3c3185670cc10f2bceed5dbdac` |
| `src/table/partition.rs` | `54a80b996ea9df07fff9077c54a20f8e436cc8843c2daae543f8d77116708edd` |
| `src/coord/locator.rs` | `93833607fe434b5c2f4788e200800388ed13c9b47bd39e832e0f4d0dd859fd2d` |
| `src/coord/locator_core.rs` | `1d85a66893cd20fbdff03c5d58e649b1844fb70f3edc42fc591638caf144c95d` |
| `src/table/state.rs` | `4a72f3834f067b62642daeeb48e4cbffde5b617e5b9132dd2eebabf176c423a0` |
| `src/table/keyrep.rs` | `2b0db4d24967322b0cc95dd16fd68ec7dae734f7502c24c81db2a7ea242ee03c` |
| `proof/regen.sh` | `88ebc85ccb46f1f5a686cbbe1e51a20401aecf8cdfe7184cae1b814d653089a8` |
| generated `out/Preset.lean` | `b3644dbf789ce6a2e9d72a0d44c3177642fa3efe00b973eaa623c7b30519d5e3` |

### Adapter-out-of-cone probe — zero-axiom read/kernel extraction

Removing the write/growth adapter roots (`*_table_build_at`, `*_prepared_shard_apply`,
`*_apply_insert`, `*_relocate_*`, `*_apply_pending_and_commit`, `*_publish_prepared`,
`word_capacity_bound_holds`, `word_schedule_is_complete`; 27 of 43 roots) and re-extracting under
`--preset aeneas` gives:

| Metric | Full 43 roots | Reduced 27 roots |
|---|---|---|
| definitions | 444 | **154** |
| axioms | 11 | **0** |
| `table.Table.get_at` | present | present |
| `reserve_rebuild` / `try_reserve` / `Directory::link` | present | absent |

Generated file sha `9ead8f3319ee0b236aad4a56d4bf7fd2b2755c61dcab3d0499e778b23abbaee4`. This is the
design's adapter-out-of-cone boundary: the **read/query face and the word/locator kernels extract
axiom-free**, while the fallible write/growth adapter is the named `ResourceAdapterSimulates`
residual. It is evidence for that boundary, not an accepted extraction (the production root set and
the Lean bridge are unchanged; the write face is not established).

### Verdicts on the residual (not this probe's work)

Two escalations were consulted on how to remove the 11 axioms.

- **Astra (`anchor`)**: select hybrid C. The resource family (`Vec.capacity/try_reserve/
  try_reserve_exact/TryReserveError`) is blocked because the pinned `Vec` model identifies a vector
  by its element list only, so it cannot carry capacity or allocation outcome; a definition alone
  cannot preserve two observations of equal-contents/different-capacity vectors. `size_of` needs
  target-layout evidence. The ordinary names (`is_err`, `as_ref`, `Try.branch`, `truncate`, `clear`,
  `drop`) are simplifiable or modelable.
- **Sol (contract/attribution)**: the fixed-shard selection in `docs/kernel-boundary.md:258-272` is
  historical ("Prior selection"); the current boundary (`:279-285`) and `docs/direction.md:39-45`
  select the growable table. Adopting fixed shards therefore requires amending `docs/direction.md`,
  `docs/kernel-boundary.md`, `docs/plan.md` and the ANATIVE lifecycle, and retiring
  `ReserveBeforeRetainContract`, the eight `LocatorReservationReceipt` categories,
  `NativeLocator.TableStorageCoverage`, `Growable/{Relocation,Rebuild,Resources,Representation}`, the
  `injection` failure controls and W7 — with a recorded performance blocker
  (`docs/plan.md:45-52`). **No third zero-axiom path exists** for the growable table's
  allocating methods under the pinned backend; excluding the adapter gives a zero-axiom core but
  leaves `ResourceAdapterSimulates` unproved.

Consequence: the 11-axiom residual is not a source-shape defect and is not removed by the quotient
storage re-founding (which the current ANATIVE spec does not select). The decision is the lead's:
accept the adapter exclusion at the existing `ResourceAdapterSimulates` boundary (zero-axiom read/
kernel cone, write face open), or authorize a governing-document amendment plus the resource/
backend representation work.

### Trust boundary

A clean structural translation is entry-encoding feasibility, not semantic preservation, generic
refinement or end-to-end correctness. The generated file's 23 axioms keep the fail-closed cone red;
no claim of `proved` follows. The probe command used a throwaway destination; no production artifact
was written.

## ANATIVE W4-E-I7 extraction-lane feasibility

Result: the lane is **red**. Charon translates the new typed-face/locator unit and emits the LLBC;
Aeneas aborts in `translate_type_decls` on an arrow type. No `proof/ApeironWord.lean` was written.

```text
SCOPE:      proof/regen.sh SOURCE_FILES and two --start-from roots; proof/ApeironWord.lean left
            byte-identical; this receipt. Completion: the lane reaches or fails on the new
            typed-face/locator modules with a recorded first error.
REGISTER:   local obligation (refinement/extraction lane). Exclusions: src/, tests/, examples/,
            docs/, Cargo.toml; the Growable summary-shape debt is not touched.
CORPUS:     proof/corpus/Native/LocatorSpec.lean, extraction_targets_exact (lines 874-906), read;
            no theorem, grade or corpus declaration changed.
REFERENCE:  LocatorSpec.requiredExtractionTargets; 13 of its wrapper roots do not exist in
            proof/extract/lib.rs and cannot be expressed without new wrappers.
GATE:       flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=6G
            -p MemorySwapMax=0 -- env LEAN_NUM_THREADS=1 timeout 2400 bash -lc './proof/regen.sh'
            -> exit 2; first Aeneas error table.rs 560:4-615:5; fatal table/locator.rs 268:0-271:1.
GRADE:      measured (one lane-locked run); extraction red, not proved, not tested.
```

### Delta

`proof/regen.sh` `SOURCE_FILES` adds the five missing modules and the `include!`/`#[path]` comment is
corrected:

- `src/table/probe.rs`, `src/table/locator.rs` (textual `include!` from `src/table.rs`)
- `src/coord/address.rs`, `src/coord/locator.rs`, `src/coord/locator_core.rs` (`#[path]` from
  `src/coord.rs`, then `src/coord/locator.rs`)

`--start-from` adds the only two unrooted wrappers in `extraction_targets_exact` that already exist
in `proof/extract/lib.rs`:

- `crate::word_stored_table_build_at`, `crate::word_regen_table_build_at`

Thirteen targets in `extraction_targets_exact` cannot be expressed: `word_face_address`; the eight
`word_{stored,regen}_locator_{decide,publish,relink,entry}`; and the four
`word_{stored,regen}_{build_at,get_each_at}_parallel`. No wrapper for them exists in the extraction
unit. The typed table paths are nevertheless exercised: the existing
`word_stored_prepared_shard_apply`/`word_regen_prepared_shard_apply` roots reach
`apply_prepared_shard_at` -> `reserve_locator_path` -> `Locator::reserve_for`, and
`word_*_table_build_at` reaches the build and parallel paths.

### Command and outcome

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=6G \
  -p MemorySwapMax=0 -- env LEAN_NUM_THREADS=1 timeout 2400 bash -lc './proof/regen.sh'
# exit=2
```

Tools: charon `0.1.254 (b104e24fea7d721b71e6c39fd70f26ff20bc0980)`, aeneas `505b6ca-dirty`,
rust `nightly-2026-08-18`, lean `v4.31.0`. Charon succeeded and wrote `proof/apeiron_word.llbc`
(12,066,259 bytes); the script aborted before its normal `rm`, so that artifact was removed after
the run. All `[Error]` lines below are Aeneas reading that LLBC.

Aeneas errors, in order:

1. `Early returns inside of loops are not supported yet` — `src/table.rs` 560:4-615:5,
   `Table::selected_lanes`.
2. `Early returns inside of loops are not supported yet` — `src/table.rs` 814:4-857:5,
   `Table::observe_receipt`.
3. `Returns inside of nested loops are not supported yet` — `src/coord/locator_core.rs`
   659:4-706:5, `Directory::stage_reserved`.
4. `Arrow types are not supported yet` — `src/table/locator.rs` 268:0-271:1, type decl
   `apeiron_word::table::QueryIdentity`, then `Could not translate type decl ... because of previous
   error`, then uncaught `Aeneas.Errors.CFailure` in `SymbolicToPure.translate_type_decls`.

The arrow type is the field `_carrier: PhantomData<fn() -> (H, R)>` on `QueryIdentity`. Type-decl
translation is crate-wide and independent of the `--start-from` roots, so this blocks every locator
extraction regardless of root selection.

### thread_local / injection-seam risk

No `thread_local`, `LocalKey`, or `LocalKey::new` error appears. The two seams
(`src/coord/locator_core.rs:311`, `src/table/parallel.rs:101`) are statically reachable from the
existing and new roots, but Aeneas aborts in `translate_type_decls` before function translation, so
its `LocalKey::new` rejection is **not reached and not confirmed**. This is a residual obligation,
not a confirmed blocker.

### Generated file

`proof/ApeironWord.lean` is unchanged by this run: sha256
`9af54e7d2467a5a19ebccfd7fd70691072a17fda76e78e2366c3832675749d3e`, identical to the pre-run
working-tree file. No new digest exists because regeneration failed. `proof/regen.sh --check` was not
run; it applies only after successful regeneration. The Growable summary-shape debt in `proof/corpus`
was not exercised — the run stops before any `lake build`.

### Residual and next bounded task

- First blocker (source, out of this task's scope): replace the `PhantomData<fn() -> (H, R)>` marker
  on `src/table/locator.rs::QueryIdentity` with an Aeneas-translatable variance marker.
- Then the loop-return bodies: `Table::selected_lanes`, `Table::observe_receipt`,
  `Directory::stage_reserved`.
- Then the `reservation_failure` seams, expected to need a source-side feature gate like
  `read-stats`.

Next bounded task: a new source assignment to remove the `fn()` phantom from `QueryIdentity`, then
re-run this same lane-locked command; the injection-seam gate is a later slice.
