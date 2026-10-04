# Bounded Storage Refinement

Merged epic record: `intent/storage-refinement.md`, `spec/storage-refinement.md`, `plan/storage-refinement.md` (one file per epic; see `docs/README.md`).

## Intent - Bounded Storage Refinement

- **Owner:** OpenCode
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean
- **Status:** approved
- **Spec:** `docs/epics/storage-refinement.md`
- **Plan:** `docs/epics/storage-refinement.md`

### Problem

`Map` implements lookup, replacement and removal with open addressing, `MaybeUninit`, SIMD tag
scans and growth. Random differential tests and Miri exercise it, but no safe executable state
machine defines its finite failure semantics and no proof links storage transitions to
`Apeiron.Table.insertWith`.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Extract the complete optimized map first | Aeneas does not cover its unsafe allocation, SIMD and generic closure surface. |
| Treat `std::collections::HashMap` as the specification | It supplies an oracle but not the canonical Table theorem link or fixed-capacity refusal. |
| Restate table equations beside Rust | A second formula is not a refinement of the executing body. |
| Replace the optimized map immediately | Existing measured paths remain useful until an equivalent implementation is proved and measured. |

### Completion condition

A bounded safe reference is extracted from real Rust, proved to implement canonical Table reads and
writes on successful transitions, and the production map passes a discriminating transition trace
against it. The optimized map remains labelled tested until the relation itself is universal.

```sh
sh -c 'cargo test --workspace --all-features --locked && proof/regen.sh --check && cd proof/corpus && lake build'
```

### Invariants

1. Full capacity refuses only insertion of an absent key; replacement at capacity remains valid.
2. Refusal returns no successor state and leaves the input state available to the caller.
3. Lookup and removal preserve full key identity; physical slot order is unobservable.
4. Canonical merge argument order is new value then old value.
5. Algebraic merge claims instantiate `Apeiron.Trichotomy.AnyRing`; generic replacement semantics
   are identified as table semantics rather than attributed to the trichotomy.
6. No proof about the safe reference is reported as proof of the optimized map.
7. **Aggregate reserve before the first merge.** Before the first merge for a batch commits, checked
   sizing and all fallible locator reservations for the batch's worst-case fresh inserts are
   complete; no later merge or directory link may fail for lack of locator allocation.

   The reserve point is the batch entry point, which determines the batch's worst-case fresh-insert
   count before it merges anything: `insert_with_at` for a one-key batch, `build`, `build_at` and
   `build_at_sequential` for `kvs.len()` entries. `TableLocator::reserve_batch_paths` is the whole
   reservation — checked sizing, one `u32` handle bound, and one capacity-only `try_reserve` of the
   reverse-link registry. It stages no path, creates no
   directory node and reads no key, so its cost does not scale with the 64 residue levels a link
   costs.

   The per-entry merge (`Table::insert_with_reserved`, and `apply_insert`, `grow_spill` and
   `apply_pending_and_commit` beneath it) therefore performs no locator reservation and no directory
   walk or link. Its only locator effect is `TableLocator::record_admitted_link`, a write into the
   already-reserved registry extent, and that write is fail-closed: with no room in the reserved
   extent the merge returns `StorageError::AllocationFailed` before its payload effect rather than
   allocating after a commit. That refusal is the invariant's reachable witness, so removing the
   reserve is observable rather than unobservable — the negative control is
   `batch_reservation_admits_a_fresh_insert_and_its_absence_refuses_one` in `src/table/locator.rs`.

   Directory content is not maintained on the merge path. A reverse link admitted by the merge path
   carries `UNRESOLVED_IDENTITY_NODE`, and `check_storage_coverage` issues no certificate over an
   unresolved link. `Table::assemble_directory` — a cold pass over the live payloads, off every read
   and write path — links each payload's complete path, resolves its identity node and records its
   location; `build` and `build_at` run it before returning, so a table they return is covered, and
   a serially inserted one becomes covered when that pass runs.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| The safe finite reference implements its stated transitions. | tested | `cargo test --locked --test safe_table` |
| Extracted successful transitions refine canonical `Apeiron.Table` operations. | proved | Named bridge theorems and fail-closed audits. |
| The optimized map agrees on selected transition traces. | tested | Differential production/reference test and mutation control. |
| The optimized unsafe storage universally refines the reference. | open | Requires a separate universal representation proof. |

### Non-goals

- Proving allocation, SIMD, threads, rustc/LLVM or machine execution.
- Covering every `Map` convenience API in the first reference.
- Replacing the optimized implementation before equivalent behavior and cost are established.
- Claiming that generic table replacement is derived from the trichotomy.

### Trust boundary

The safe Rust extraction and Lean theorem do not cover the optimized map's unsafe slots, allocation,
SIMD scan, concurrency, compiler backend or hardware. Differential traces are tests, not a universal
refinement.

### Open questions

- Which bounded relation can cover the optimized slot representation without importing unsafe
  allocation into the Aeneas proof? This is answered by a later wave after the reference is fixed.

## Specification - Bounded Storage Refinement

> **Kani retirement.** The [user-directed retirement](kani-retirement.md) supersedes
> every Kani run, preservation task and barrier requirement below. Historical receipts and
> experiments remain revision-bound evidence; current work uses ordinary Rust gates and semantic
> mutation controls.

- **Owner:** OpenCode
- **Intent:** `docs/epics/storage-refinement.md`
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean before this epic
- **Status:** approved

> **Record status.** The two-container content below (`SafeTable<K, V, N>`, `Map<K, V>`, the
> fixed-capacity-four reference) records retired source. The executing representation is the growable
> flat/spill `Table<K, V, H, N, R>` in `src/table.rs`, with `Stored` and `Regen` carrier faces. The
> fixed-shard `Table<K, V, H, N>` and its proofs are revision-bound historical evidence. Face/key
> semantics, layer separation and cost boundaries are stated here; historical rows remain labelled.

### Face and key semantics

The canonical object is `Apeiron.Table.HMap` (`K → V`) with `Apeiron.Table.get`,
`Apeiron.Table.insertWith` and `Apeiron.Table.build`. A face is one sign of `Sigma`, and it fixes how
a key becomes an address and what the stored entry carries. Face semantics interpret the canonical
object; they do not replace it.

| Face | `Sigma` | Address | Identity |
|---|---|---|---|
| Elliptic | `sigma < 0` | exact packing on an explicit admitted domain | the address determines the key on that domain |
| Parabolic | `sigma = 0` | a lossy spreading fold | the complete key is carried beside the address |
| Hyperbolic | `sigma > 0` | a scramble and a discard | the key is discarded on an explicit observation domain |

- **Elliptic: exact packing on an explicit admitted domain.** The address is a bijection
  (`Packed`/`unpack`) exactly for keys that fit the packed width. Outside that admitted domain the
  packing does not exist and the key is refused rather than packed lossily.
- **Parabolic: carried complete key.** The address spreads the key; it never decides identity. A
  coordinate or hash equality is a placement probe; the stored complete key is compared, and the
  coordinate never replaces that comparison.
- **Hyperbolic: scramble/discard on an explicit observation domain.** The address is a scrambler
  judged on distribution, and the discard is defined only on the observations the face retains. It
  makes no identity claim outside that domain.

Every executing address signature carries `Sigma` as a value. `Cell<R>` operations take the dial, and
`Coord` addressing is correct per face; a signature that omits `Sigma` selects no face and is not an
executing address signature.

**Fixed-shard contract (historical).** `H: Home<K>` selects exactly one home shard,
`home(key) mod 256`; a miss in the home shard is a miss and there is no fallback probing. Insertion
refuses with `ShardFull` exactly when the key is absent and its home shard has no vacant slot;
`N = 0` refuses every write and keeps reads total. Refusal changes no field and invokes neither clone
nor merge. The growable contract below replaces this home-only miss with a finite first-fit order and
a growable refusal contract.

### Three layers

- **Canonical semantics.** `Apeiron.Table.HMap`, `get`, `insertWith`, `build`: the shared total map,
  its new-first write and its ordered build fold.
- **Physical fixed-shard invariant.** The inline `[Option<[Option<(K, V)>; N]>; 256]` state with
  `Rep`: one home shard per key, no fallback probing, `ShardFull`/`N = 0` refusal. A physical state
  denotes the canonical map under `Rep`; `Rep` is the private invariant, not the public contract.
- **Source-checker memo store.** The source checker's memo store is a separate store with its own
  refinement obligation.

A result at one layer does not transfer to another without its own refinement.

### Fixed-N cost boundary

`N` is a compile-time constant, callbacks are bounded, and key length is not constant. Lookup and
update are bounded per operation at fixed `N` with bounded callbacks and nonconstant key length.
Reading a whole source remains at least linear in its bytes at fixed `N`, so constant-time table
operations at fixed `N` do not imply constant-time whole-source checking or constant-size proof
terms. No performance measurement is claimed here.

### Revised production contract: reworked growable key-partitioned table

Recorded by `APEIRON-PERF-RESTORE-W4-T1`. This contract replaces the bounded fixed-shard contract
above. The growable flat/spill `Table<K,V,H,N,R>` is the executing implementation; its extraction and
refinement integration wave remains open. The fixed-shard `Table<K,V,H,N>` is revision-bound
historical evidence. No extraction, refinement or measurement acceptance is claimed here. Trust, pin
transport and the ratified resource/refusal contract are in the
[local boundary](../kernel-boundary.md#resourcerefusal-contract-for-the-growable-table); the dispatch
order is in the [plan](storage-refinement.md).

**First cut rejected.** The first growable `Table` at `src/table.rs` sha256
`c34ddf17ef2ed7a81fd4b3e16447479b55d737af8f5253b99dd397414d9b1dc1` is rejected as the production
result and reworked in place. The bounded fixed-shard table is not restored and the benchmark-only
candidate `examples/reg1/candidate.rs` is not promoted; one growable representation is kept. The
measured red and the defect list are in the
[local boundary](../kernel-boundary.md#reworked-growable-representation-astra-decision-first-cut-rejected).

**Representation (ratified).** A fixed directory of 256 region descriptors within one generic
`Table`, generic over `V` and merge `f`. Tags are the sole occupancy authority; a payload slot
carries `((Sigma, K), V)`, not `Option<((Sigma,K),V)>`. `coord::Addr` retains its `id`/`place`
meaning: `id` is the word-coordinate identity and `place` is placement; this contract makes no
blanket injectivity claim for `place` or for generic `K`. Shard = `place` bits [0,8); tag =
`0x80 | (place >> 57)`; descent windows = bits [8,57); group width 16. Region sizing uses checked arithmetic:
`groups = checked_pow2_ceil(max(2, ceil_div(2*load, N)))`, `slots = groups * N`; an empty region
allocates nothing. `N` is the group width in lanes, not keys per home, so no positive `N` imposes a
`256·N` ceiling. The contract refines canonical `Apeiron.Table.get`/`insertWith`/`build` under the
canonical schedule licence `Apeiron.Table.build_eq_of_key_streams_eq`.

**Address interface and laws (Sol contract).** The optional coordinate specialization stays inside
the generic table:

```text
struct TableAddress { place: u64, direct_word: Option<u32> }
trait Home<K> { fn address(&self, sigma: Sigma, key: &K) -> TableAddress; }
```

For all admitted `sigma` and `key`, `address` terminates and returns one value (totality), repeated
calls on unchanged arguments return the same value (stability), and equal `(sigma, key)` pairs return
equal addresses (equality-respect). A placement-only adapter returns `direct_word = None`; it cannot
claim the direct lane from `place`. `WordHome` returns `place = coord::spread(x).place` and
`direct_word = Some(x as u32)` exactly when the represented unsigned word `x < 2^32`, otherwise
`None`. This changes the executing contract: the old premise supplied one stable placement word for
`K`; the new premise supplies a `Sigma`-indexed placement plus an optional, certified exact word.

Let `π(d) = d*PHI mod 256`, and let directory slot `d` describe physical shard `π(d)`. Under
`x < 2^32`, `WordHome.address(sigma,k) = { place = spread(x).place, direct_word = Some(x as u32) }`,
and faithful representation of `k` by `x`, the routing equation is
`directoryIndex(place) = π⁻¹(place & 255) = x & 255`. The direct lane is
`u64::from(direct_word) >> 8`, never `place >> 8`, and is admitted only when it is below the routed
region's slot count. Directory, tag, window and direct lane are hints; identity is always equality of
the full `(Sigma, K)` payload.

**Window and mask rule (Sol contract).** Set
`D = (place >> 8) & ((1 << 49) - 1)`, `WINDOW_MASK = (1 << 7) - 1`, and, for
`j in {0,...,6}`,
`w[j] = (D >> (49 - 7*(j+1))) & WINDOW_MASK`. The prefix group is
`w[j] & (G - 1)`. `G - 1` reduces a seven-bit window to the region; it is not the window extraction
mask. Thus a region with `G > 128` intentionally reaches at most groups `0..127` in the prefix and
does not consume a tag bit or a previous window. The coverage sweep uses
`((D & (G - 1)) + s + 1) & (G - 1)` for `0 <= s < G`; it visits every group once. This supersedes
`group_of` masking the unbounded shifted value directly by `gmask`.

**Probe and absence rule (Sol contract).** The finite order is the admitted direct candidate,
seven windows, then the cyclic coverage sweep. A reachable state starts at the real constructor and
is closed under successful insertion and transactional relocation; tags are authoritative
(`tag = 0` iff vacant), logical keys are unique, insertion and relocation use this order, occupancy
is monotone between relocations, and relocation atomically re-establishes these facts. It also
satisfies `DirectCover`: if a live logical key has an admitted direct lane `l`, then lane `l` is
occupied, though not necessarily by that key. Under `DirectCover`, an empty direct candidate proves
the key absent and may return that vacancy before any window. An occupied direct candidate proves
nothing about identity, so search continues.

Without that certificate, finding a window vacancy does not yet license absence. The implementation
may record the first vacancy and skip the remaining, later windows, but the sweep must still inspect
all `G` groups for a full-key match. The prefix need not complete before that sweep: every skipped
prefix position follows the already recorded vacancy, and the sweep supplies finite coverage. If no
earlier proved certificate applies, absence requires complete finite coverage. No other early absence
is admitted merely because historical first-fit stopped at a first vacant group; that rule requires
its own reachable-state preservation theorem. Removal requires tombstones or a transactional rebuild
that preserves the certificate.

The required theorem `search_complete` states, for every query `q` and reachable well-formed region:
`Found i` iff lane `i` is live with payload key `q`; `Vacant i` implies `q` is absent and `i` is the
first vacant lane in the defined order; `Exhausted` implies `q` is absent and no lane is vacant.
`locate_first_fit` supplies the returned-index clauses, termination and bounds. `b = 0` has no
admissible insertion.

**Payload key.** For the extracted word instance the raw exact coordinate word is sound as the
payload key, via canonical `Apeiron.Table.phi64_lowWord_injective` and bridge
`Apeiron.TableRefinement.coord_spread_id_injective`. Generic `K` must carry a faithful full `K::eq`,
because `Home<K>` may be lossy.

**Slot initialization and invariant.** Unused payloads may hold clones of an available key and the
default; tags are the occupancy authority, so `¬occupied → payload ignored`. Premises: `clone`
terminates, preserves denotation and is pure; equality is total, faithful and pure; no
`Classical.choice` fabricates an initializer Rust does not have.

**Transaction point.** locate; compute the complete checked layout; reserve every allocation; then
initialize/relocate without merge; apply the merge exactly once; commit. `insert_err_stutters`: on
`Err` the state is unchanged, no clone/merge/relocate runs and no partial table is observable.
Relocation failure is proved unreachable for the checked layout, not aliased to `AllocationFailed`.
`BuildPre` is the build's checked pre-state; build failure returns no table.

**Admissible domain and partiality (ratified).** The growable representation can fail allocation. Its
resource/refusal contract is ratified in the
[local boundary](../kernel-boundary.md#resourcerefusal-contract-for-the-growable-table) (option (b),
checked reservation/allocation-failure result, transactional): `insert`/`build` return `Result`, `Err`
leaves the table unchanged and returns no partial table, and `ShardFull` and positive-`N`
`BuildAdmissibleN` are retired from the growable production contract. Reads remain total on the
admitted domain; allocation failure is a stated outcome, never hidden through `Vec::push`. Allocator
success/failure correspondence is an unextracted trust boundary with a simulation statement and must
not enter the `Slang.*` empty-cone chain.

**`Sigma` (mandatory).** The logical key is exactly `(Sigma, K)`; payloads carry both components,
equality compares both, reads and writes receive `sigma`, and `Home::address` receives it. Build
consumes `&[((Sigma, K), V)]`. The relation is `Rep t (HMap (Sigma × K) V)`. For every `k` and
distinct `sigma1 != sigma2`, updates and reads at `(sigma1,k)` leave `(sigma2,k)` unchanged even when
their addresses are identical; this is a required test and refinement theorem. A `Sigma`-free
interface cannot be proved first with `Sigma` deferred.

#### Flat-resident and grouped-spill contract (Sol decision)

This section supersedes the region-only representation and payload-key clauses above where they
differ. It records the Sol contract for the Astra boundary; both decisions are `anchor`, not
`proved`. No executing or performance claim follows from this record.

**Composite representation.** One table owns:

- a flat resident area in shard-major order, `256 × L`, whose slot is
  `{ live, sigma, key, value }`; `live` is the sole flat occupancy authority; and
- independently load-sized grouped spill regions whose tags are the sole spill occupancy authority.

All flat and spill lengths, byte counts, offsets and allocation budgets use checked arithmetic. A
key has an admitted certified flat coordinate exactly when `direct_word = Some(w)` and its derived
`lane < L`. The table-level density test decides whether that coordinate is resident. A failed
density or budget test routes the key to spill; it is not insertion refusal. Allocation or sizing
failure for the complete transaction remains the ratified `Result` refusal.

There is exactly one representative of each logical `(Sigma,K)` under the composite invariant. Its
coverage component is `DirectCover` plus this stronger extent clause: if a live key has an admitted
certified coordinate, the corresponding flat slot is occupied, even when that key's unique
representative is in spill. Occupancy in that clause need not be by the same key. Increasing `L`
reclassifies every previously spilled key that newly enters the extent and rebuilds the composite
state; extent growth is not an append.

`Site` is the exhaustive locate result:

```text
FlatFound | FlatVacant | SpillFound | SpillVacant | NeedSpillGrowth
```

`FlatFound` and `SpillFound` identify the unique representative. `FlatVacant` is an absence
certificate only under composite `DirectCover`; otherwise locate must complete finite spill
coverage. `SpillVacant` records a proved first available spill position after complete required
matching coverage. `NeedSpillGrowth` carries the checked growth requirement, not a refusal.

**Key representation and identity.** Storage policy is explicit:

```text
KeyRep<K,H> = Stored<K> | Regen<C>
Stored<K>::Held = K
Regen<C>::Held = u64
```

`Stored<K>` retains the full key. `Regen<C>` retains the identity word and is sealed to reviewed
exact codecs `C`. Both retain `Sigma` separately. On the admitted codec domain,
`decode(sigma, encode(sigma, k)) = k`; encode/decode totality, stability and equality-respect are
premises. Regenerated identity is recovered only through the verified inverse of `Addr.id`, never
from `Addr.place`. Dropping `K` therefore requires the exact codec law and its joint admitted-domain
witness. `direct_word = Some(w)` alone is only a coordinate certificate and never licenses key
dropping.

**Search geometry.** Direct-empty absence is sound only in a reachable state satisfying composite
`DirectCover`; without it, complete finite flat-and-spill coverage is required. The historical
variable-width descent and the current seven-bit window descent are different algorithms. Restoring
the historical geometry, or changing the current descent, requires a separate Sol adjudication of
the search order, first-empty certificate and reachable-state preservation premises.

**Transactional insertion and growth.** The only admitted order is: locate; construct a complete
checked plan; reserve all fallible storage; initialize or relocate without merge; invoke merge
exactly once; commit. On `Err`, no committed field changes and no clone, merge or relocation has
run. Extent and spill growth use the same transaction and relocation never invokes merge.

**Required bridge set.** Each row is an `anchor` now. Its headline may carry `proved` only when the
real implementing bodies are included from `src/` by `#[path]`, the generated definitions are
fresh under `proof/regen.sh --check`, the named refinement is checked by a bounded `lake build`, and
the headline is fail-closed in `proof/corpus/AxiomAudit.lean` under
`{propext, Classical.choice, Quot.sound}`.

| Module | Required headline and named premise | Canonical definition or theorem refined |
|---|---|---|
| `GrowableGeometry` | `geometry_wellFormed`: joint constructor witness for checked `256 × L` sizing, density threshold, budgets, group width, spill divisibility and every computed bound | physical representation premise for `Apeiron.Table.HMap` |
| `GrowableDirectory` | `site_partition`: stable routing and exhaustive, disjoint `Site` classification with admitted coordinate `direct_word = Some(w) ∧ lane < L` | key-routing premise for `Apeiron.Table.get` and `insertWith` |
| `GrowableIdentity` | `keyRep_identity`: exact sealed-codec round trip, separate `Sigma`, faithful `Stored`, and regeneration only through the inverse of `Addr.id` | equality on `Apeiron.Table.HMap (Sigma × K) V` |
| `GrowableScan` | `scan_exact`: `live` and tags are their respective sole occupancy authorities; flat and grouped matches/vacancies are exact and read no vacant payload | observation premise for `Apeiron.Table.get` |
| `GrowableSearch` | `locate_complete`: composite `DirectCover`, reachable first-empty certification, finite coverage, termination and bounds make every `Site` result sound | `Apeiron.Table.get_insertWith_self` and `get_insertWith_ne` |
| `GrowableRelocation` | `rebuild_preserves`: extent expansion reclassifies newly admitted spill keys and relocation preserves uniqueness, coverage, identity, values and no-merge execution | preservation of the represented `Apeiron.Table.HMap`; `GrowableRefinement` composes the pending apply with `Apeiron.Table.insertWith` |
| `GrowableRefinement` | `operations_refine`: constructor establishes `Rep`; get, insert and build refine the canonical operations; sizing/allocation `Err` stutters | `Apeiron.Table.{HMap,get,insertWith,build}` and `get_build_acc` |
| `GrowableWitnesses` | `reachable_joint_witnesses`: one actual-type witness covers constructor, direct/spill success, spill growth, extent rebuild, overflow and injected reservation-failure branches | admissible-domain and preservation premises of the four canonical operations |
| `GrowableSchedule` | `partition_preserves_key_streams`: stable full-logical-key partition and disjoint complete assembly preserve each key's value subsequence | `Apeiron.Table.build_eq_of_key_streams_eq` (`Table/Basic.lean:167-174`) |
| `GrowableBatch` | `grouped_readback_refines`: grouped readback order and assembly preserve canonical build; any partial reduction states and proves its merge law | `Apeiron.Table.build`; `build_eq_of_key_streams_eq`, or `build_gauge_invariance` when same-key order changes |

`build_eq_of_key_streams_eq` requires equality of each key's value subsequence; stable partition and
assembly must supply that exact premise. It does not license same-key reordering or combining partial
accumulators. Grouped readback order and any reduction law remain explicit premises.

**Pending Sol adjudications.** Code must not select among these open contract alternatives:

1. flat geometry and constructor joint witnesses;
2. composite `DirectCover`, extent expansion, relocation and removal preservation;
3. exact codec/regeneration identity and separate `Sigma` preservation;
4. reachable first-empty-group certification and any descent-geometry change;
5. stable partition/assembly supplying the exact premise of
   `Apeiron.Table.build_eq_of_key_streams_eq`;
6. grouped readback order and any reduction law; and
7. allocation-simulation premises, successful and refusing witnesses, final claim grades and audit
   coverage.

**End-to-end verification criterion.** “Formally verified end to end” for this table means all real
performance-bearing sequential bodies are extracted from `src/` through `#[path]` wrappers that only
specialize and call them; the ten audited headlines above refine those generated bodies to
`Apeiron.Table.{HMap,get,insertWith,build}` on one reachable admitted domain; the bounded bridge build
passes under the closed cone; `proof/regen.sh --check` proves generated-file freshness; and mutations
of identity, occupancy, search, relocation, merge order, partition order and refusal handling turn
their named proof or test red before the restored source passes.

That phrase excludes allocator/global-allocator realization, thread/executor realization including
`std::thread::scope`, rustc/LLVM code generation, the kernel ABI, GPU/SPIR-V and the `unsafe`
container. Each requires a separate realization refinement. The Slang source-refinement chain stays
separate and requires an empty axiom cone; no theorem depending on the Rust/canonical cone enters it.
REG-1/G1 remains **OPEN**. This contract creates no `proved`, `tested` or `measured` claim.

**Extractability constraints for new source.** Source on the extracted path is admitted only when:

- `locate` is an explicit search state with bounded loops and the result returned after the loops;
  no `return` inside nested loops.
- no GATs or higher-ranked borrowed associated types.
- `Vec` storage is initialized, with explicit index loops and result matching.
- length and byte arithmetic is checked, with an explicit overflow/allocation-failure policy.
- extraction wrappers only specialize and call real bodies.
- window descent is not replaced by a linear scan for extraction convenience.

**Unextracted execution boundary.** Threading is behind an unextracted orchestration boundary; the
extracted roots are the sequential partition, worker and assembly bodies. The schedule obligation
composes the implementation-specific per-key subsequence equation with
`Apeiron.Table.build_eq_of_key_streams_eq`. This covers sequential worker semantics and schedule
preservation only; thread memory semantics, exclusive-borrow realization, spawn/join failure, panic
propagation, liveness, allocator behaviour and rustc/LLVM codegen remain open.

**Identity and theorem placement.** The canonical odd-multiply injectivity theorem and the Rust
refinement paths are named in the
[local boundary](../kernel-boundary.md#reworked-growable-representation-astra-decision-first-cut-rejected),
together with the full required bridge theorem set and its audit placement. This contract requires
the refinement to unfold the generated `coord.spread` low-word truncation and instantiate at `PHI`.

**Decision grades.** The Astra verdict and this Sol contract are `anchor`, not `proved`. Focused Rust
tests may later establish `tested` only for their named finite cases. A Rust semantic headline may
carry `proved` only under the end-to-end criterion above. Canonical
`Apeiron.Table.build_eq_of_key_streams_eq` retains its existing `proved` grade; the
implementation-specific stable-partition premise remains an `anchor` until `GrowableSchedule` is
checked. Performance remains separately `measured`; it cannot upgrade a semantic clause.

### Extraction-shaped representation contract (Astra, APEIRON-EXTRACT-REP)

Recorded by `APEIRON-EXTRACT-REP-W1-T1`; documentation only, `anchor`, no source, extraction, proof or
performance claim. This contract supersedes the `KeyRep`/`Held` carrier and `rebuild_resident` wording
above where they differ. The boundary is in the
[local boundary](../kernel-boundary.md#extraction-shaped-representation-boundary-astra-apeiron-extract-rep);
the dispatch order is in the [plan](storage-refinement.md#extraction-shaped-representation-dispatch-order-apeiron-extract-rep).

**Carrier, not policy.** The fifth type parameter is the stored carrier itself:
`Stored<K> { key: K }` and `Regen<C> { identity: u64, codec: PhantomData<C> }` with a manual `Clone`
and no `C: Clone` bound. `FlatSlot<R,V>` contains `R`; spill slots contain `(Sigma,R,V)`. No `R::Held`
projection exists in storage, relocation temporaries or worker results. Public spellings
`Table<…,Stored<K>>` and `Table<…,Regen<I64WordCodec>>` are preserved. Heterogeneity is between
compile-time instantiations, never a runtime enum or heterogeneous vector. `Sigma` is retained
separately on both faces; representation is selected by `R`, not switched by `Sigma`. `KeyRep` keeps
`hold`/`matches`/`key_owned` and adds `address_held`, so relocation does not clone arbitrary carried
keys.

**Both carrier faces, generic mathematics.** The proof-root candidates are the real shared
prepared-plan sequential core/apply operation, specialized at
`Table<i64,i64,WordHome,16,Stored<i64>>` and
`Table<i64,i64,WordHome,16,Regen<I64WordCodec>>`. The wrappers supply an already prepared plan and
reserved storage and call the same core used by production; they do not restate its transition.
Planning, reservation and allocating public `build`/`insert` methods are not roots under the resource
boundary below. Canonical `HMap (Sigma × K) V` and the per-key-stream lemmas stay generic.
Word-instance proofs alone do not establish arbitrary-`K` refinement. Denotation is
`decodeStored(s,Stored(k)) = k`,
`decodeRegen(s,Regen(id)) = C.decode(s,id)`, `logicalKey = (slot.sigma, decodeR(...))`, and `RepR`
composites the physical invariant with the decoded observations. `GrowableIdentity.keyRep_identity`
must establish faithful carried-key equality/clone, codec round-trip on the joint admitted domain,
`matches` iff decoded logical equality, separate `Sigma`, and address agreement for reconstructed
carriers. `proof/corpus/TableRefinement/CoordIdentity.lean` supplies the `spread`/identity injectivity
link (`coord_spread_id_injective`), not the codec inverse theorem.

**Required restructuring.** `rebuild_resident` becomes checked-plan, then reserve-all, then
apply/commit, with the pending insertion inside the transaction: no payload clone, relocation or merge
runs before all reservations, and no intermediate commit precedes spill growth. Bounds and prepared
capacity make relocation failure unreachable; an internal placement failure is not `AllocationFailed`.
The factored build body returns a single `Result<Region,StorageError>` consumed at one branch
boundary. The sequential worker/assembly/probe bodies the parallel executor uses are callable
functions shared by the executor and the extraction roots; `spawn`/`join`/`unwind` orchestration stays
outside the extracted claim.

**Allocator boundary.** `Region::reserve` and `Flat::reserve` call `Vec::try_reserve_exact`, a
reachable unmodeled primitive; it blocks an axiom-free accepted extraction of the complete allocating
public methods.

#### Prepared-plan resource interface (Sol decision, APEIRON-EXTRACT-REP-W5-T1)

**Decision and grade.** The extraction boundary includes the real shared prepared-plan sequential
core/apply operation on both carrier faces. `plan_rebuild`, `reserve_rebuild`, every
`try_reserve*` call and the allocating public `build`/`insert` methods are the unextracted resource
adapter. This is a contract decision at grade `anchor`. Digest history: the generated
`proof/ApeironWord.lean` sha256
`58d0ba3e871cd75ae84d5cae7d7966242d3309b1a14ec1b032a26cdd5606e64b` was an uncommitted checkpoint
and is unrecoverable; it is not a landed four-root regeneration. The then-current clean extraction
checkpoint was `89c58fdd35df102d62dd07231a8c8d52d5b102cb684a40923030c6ac2768a9de`: it retained the two real
per-entry relocation roots, while `relocate_all` rooting was in flight. Rooting the old
`apply_pending_and_commit` body was rejected by the fail-closed audit described below; option (b)
supersedes that body. No
combined `relocate_all` plus pending-commit regeneration had landed at that revision. None of these checkpoints
contains or proves composite `Table::locate`, `Site` or `DirectCover`.

**Joint prepared-resource handoff (current decision).** The adapter result is exactly one of the
following proposed source shapes; these declarations do not exist yet:

```text
CapacityBound {
  required_len: usize,
  observed_capacity: usize,
}
ReservationReceipt {
  flat: CapacityBound,
  directory: CapacityBound,
  regions: [ { tags: CapacityBound, payloads: CapacityBound }; 256 ],
}
PreparedOutcome<R> {
  reservation: ReservationReceipt,
  pending_held: R,
}
PreparedResult<R,V> =
    Prepared {
      plan: RebuildPlan,
      storage: RebuildReserved<R,V>,
      outcome: PreparedOutcome<R>,
    }
  | Refused(StorageError)
```

The fixed-size receipt introduces no fallible receipt allocation after reservation. For each bound,
`required_len` is the checked target vector length and `observed_capacity` is the corresponding real
`Vec::capacity()` read after all reservations succeed and before initialization. The targets are:

- flat: checked `256 * plan.lanes`;
- region directory: `256`; and
- each region's tags and payloads: checked `slots_for(plan.spill_loads[shard], N)`, or zero when the
  planned spill load is zero.

Every comparison calls the production helper
`capacity_bound_holds(required_len: usize, observed_capacity: usize) -> bool`, whose body is exactly
`required_len <= observed_capacity`; Stored/Regen publication wrappers do not restate it, and the
helper is rooted separately from publication. Capacity reads and the correspondence between their
values and actual Rust allocations remain outside extraction. Capacity is not length, and neither a
generated `Vec` length nor construction of a receipt proves successful reservation. A violated
comparison after successful reservation is an internal invariant failure, not `AllocationFailed`
after initialization.

Input-side data are the unchanged pre-state and pending `(sigma,key,value,address)`, the checked plan,
the plan's checked target lengths, and the default. Adapter-produced data are the actual capacity
observations, `ReservationReceipt`, `R::hold(sigma,key,address)` as `pending_held`, and the
materialized vacant `RebuildReserved`. The receipt remains attached to the exact storage whose
capacities it records. It is not caller-supplied, a public arena, a reservation token, a second
storage representation, or a pending search decision.

The only admitted order is:

```text
checked planning -> reserve every flat/directory/region allocation
  -> read capacities and check every receipt bound
  -> R::hold plus Flat::initialize plus every Region::initialize
  -> publish Prepared -> relocate -> locate pending decision
  -> merge once and commit
```

All fallible planning, byte sizing, scratch reservation and destination reservation complete before
`R::hold` or payload/default clones. Initialization is part of the unextracted resource adapter,
after reserve-all and before publication. On publication, `storage.resident.lanes = plan.lanes`, the
flat slot length is its target, the region-directory length is 256, each tag/payload length is its
target, every flat `live` bit is false, and every spill tag is zero. Initialization uses the real
pending held carrier and table default, replaces no reserved vector and performs no allocation beyond
its checked capacity. The extracted relocation/apply core performs no reserve, resize, push or
initialization.

This is not the current source order. `src/table.rs:1180-1187` still calls `R::hold`,
`Flat::initialize` and every `Region::initialize` inside `Table::apply_prepared_rebuild`; meanwhile
`reserve_rebuild` returns capacity-reserved storage. The source lane must move those operations into
the adapter after `reserve_rebuild` has completed all reservations, carry `pending_held` forward, and
remove the second hold from apply. `RebuildReserved` and the existing relocation/commit signatures
remain unchanged; the new internal envelope is consumed by production before relocation.

The prospective production publication helper has shape

```text
publishPrepared(
  Result<(RebuildPlan, RebuildReserved<R,V>, PreparedOutcome<R>), StorageError>
) -> PreparedResult<R,V>
```

It is nonallocating and only moves the exact success tuple or maps `Err(e)` to `Refused(e)`; it does
not plan, reserve, read capacity, initialize or certify a predicate. Production calls and consumes
this helper. Thin Stored/Regen extraction wrappers specialize and call the same helper. The error arm
carries no plan, storage, receipt or held carrier. Independently assembled plan/storage/outcome values
are admissible theorem inputs but are not evidence that the production adapter reached `Prepared`.

**Shape — `PreparedResources`.** This predicate is defined in `proof/corpus/Growable/Resources.lean`
(W3/W4), with the contract relations carried as `PreparedResourceClauses` parameters. For carrier
`R`, the recorded shape is

```text
PreparedResources_R(t, pending, adapter_result, prepared) :=
  extractedPublish_R(adapter_result) = ok prepared /\
  prepared = Prepared { plan := p, storage := r,
                        outcome := { reservation := receipt,
                                     pending_held := held } } /\
  checkedReceiptTargets(p, receipt) /\
  everyScalarCapacityCheck(receipt) = true /\
  materializedVacant(p, r) /\
  pendingHeldCorrect(t, pending, held)
```

`checkedTargets` and all plan facts are input-side arithmetic/shape conditions.
`everyScalarCapacityCheck`, the receipt observations, `materializedVacant` and
`pendingHeldCorrect` describe adapter-produced values. The extracted publication equation proves
only faithful packaging of those explicit values; it does not prove planning, allocator behavior,
capacity-read fidelity, initialization execution, clone termination or actual adapter reachability.
`Refused(SizeOverflow | AllocationFailed)` has no success witness and exposes no partial tuple.

A non-model constructor witness becomes reachable only when the real production adapter supplies its
completed result to `publishPrepared` and production consumes that returned `Prepared` before
relocation. A source test/simulation must therefore traverse the actual adapter call path and observe
the same envelope; a proof-side call of an extraction wrapper on a hand-built tuple is only a
conditional input witness. This production-call/consumption property is the missing link that
previous model tuples did not establish.

**Named simulation statement — `resource_adapter_simulates`.** If the adapter returns `Prepared`
and `PreparedResources` holds, the extracted core consumes the supplied plan/storage and its success
transition refines the applicable canonical `Apeiron.Table.{get,insertWith,build}` denotation under
`RepR`, which composes the physical invariant with decoded carrier observations. If it returns
`Refused`, `insert_err_stutters` requires byte-for-observation unchanged committed table state and no
clone, merge or relocation; build returns no table. These refusal effects are supplied Rust test
evidence at grade `tested`, not a Lean proof or allocator simulation. The canonical operation remains
defined; resource refusal is an implementation stutter.

Resource option (b) remains checked transactional reservation with `SizeOverflow` and
`AllocationFailed`; this handoff is not reservation option (c). Pending-commit option (b) also
remains unchanged: after relocation the unextracted `locate_rebuilt` computes `SearchDecision`, and
the extracted `apply_pending_and_commit` consumes that precomputed decision. `PreparedOutcome` is not
`SearchDecision`.

#### Relocation entry interface (Sol decision, APEIRON-EXTRACT-REP-RELOCATION-CONTRACT-T1)

**Decision and grade.** The selected relocation root is the production-called per-entry operation
below; this is the required source factor, not a model transition or a second relocation
implementation:

```rust
pub(crate) fn relocate_entry(
    &mut self,
    plan: &RebuildPlan,
    reserved: &mut RebuildReserved<R, V>,
    entry_at: usize,
)
```

`src/table.rs` factors this function as `Table::relocate_entry`, and
`Table::apply_prepared_rebuild` calls it from its relocation loop.
Called by that production loop, under `entry_at < plan.entries.len()`, it clones the selected live
source payload from either the old resident area or old spill region. For a planned flat destination it writes one live
`FlatSlot { sigma, held, value }`. For a planned spill destination it recomputes the retained
carrier's address, requires `spill_locate_held` to return the plan-certified `Vacant(index)`, writes
the payload there and writes `tag(address.place)`. It does not initialize vectors, allocate, merge,
insert the pending value, change `self.len`, or commit `reserved` into `self`. The production
`apply_prepared_rebuild` must iterate this helper before its pending-insertion and commit phases.

This is a legitimate prospective straight-line extraction root under the prepared-resource boundary.
The supplied throwaway result is: rooting the whole current `apply_prepared_rebuild` exits 1 in Aeneas
`505b6ca-dirty` with `Could not find: type_var_id: 0/2` at the generic `Region::initialize` loop;
removing that loop makes the core translate; and a scratch `relocate_entry` specialized only at
`Stored<i64>` exits 0 with no generated `sorry` or `axiom`. That scratch result predates the landed
production factor (`src/table.rs:1046-1084`) and does not establish `Regen<I64WordCodec>` acceptance. `plan_rebuild`,
`reserve_rebuild`, `try_reserve*`, the rebuild decision loop, pending-insertion decision, commit
executor and allocating public methods remain unextracted. The contract and boundary decision are
`anchor`; the supplied single-carrier extraction result is `tested` tool evidence, not semantic proof.

**Shape — `RebuildPlanCorrect`.** This predicate is defined in `proof/corpus/Growable/Resources.lean`
(W3/W4), with the plan relations carried as `RebuildPlanClauses` parameters. It is input-side and
is indexed by the same `Prepared` value exposed by the extracted publication root:

```text
RebuildPlanCorrect_R(t, pending, adapter_result, prepared) :=
  PreparedResources_R(t, pending, adapter_result, prepared) /\
  prepared = Prepared { plan := p, storage := r, outcome := o } /\
  sourceEnumeration(t, p) /\
  pendingSymbolicCertificate(t, pending, p) /\
  destinationAssignment(t, p, r) /\
  prefixVacancyCertificates(t, p, r) /\
  perEntryObservationCertificates(t, p)
```

For pre-state `t`, pending logical key/value `(q,v)`, plan `p`, materialized reserved storage `r` and
resource outcome `o`, every clause is checkable before relocation from those inputs, pure
address/codec calculations and the statically bound core operations. It states jointly:

1. **Source enumeration (input-side).** `p.entries` bijectively classifies all and only old live
   source representatives of `t`; every source index is live and in bounds, and no source occurs
   twice.
2. **Pending certificate (input-side).** The pending operation is not an old-source entry in
   `p.entries`. The proof predicate derives exactly one found-or-vacant symbolic certificate by
   running the pure locate specification over the plan's symbolic destination occupancy, with enough
   reserved spill load for the vacant case. This certificate is checked without executing relocation;
   it is not stored in `PreparedOutcome`, which contains only the receipt and `pending_held`.
3. **Destination assignment (input-side).** Every assigned flat/spill index is in bounds, assignments
   are pairwise disjoint, and each flat assignment equals the source carrier's certified direct
   coordinate under `p.lanes`; the assignment therefore records every required extent
   reclassification.
4. **Prefix vacancy certificate (input-side).** For every spill-bound entry and every plan prefix,
   the occupancy map computed solely from earlier destination assignments makes
   `spill_locate_held` return the plan-certified `Vacant(i)`; the symbolic result is neither `Found`
   nor `Grow`.
5. **Per-entry observation certificate (input-side).** For each source/destination pair, decoding the
   source carrier and the carrier to be written gives the same full `(Sigma,K)`, the planned value is
   the source value, and `Stored`/`Regen` codec and reconstructed-address equations hold. The planned
   logical-key assignment is injective; its planned live/tag bits are exactly those induced by the
   destination list; and each direct-coordinate coverage obligation names an occupied planned
   destination. These are facts about `p`, not facts about the state produced by execution.
6. **Materialized-resource certificate (input-side).** `PreparedResources` establishes vacant initial
   occupancy, initialized payload lengths, destination bounds and capacity for every assignment.
   The bound relocation operation contains no allocation, initialization or merge call.

| Clause | Checked before execution from | Input-side | Excluded theorem postcondition |
|---|---|---|---|
| 1. Source enumeration | `t`, `p.entries` | yes | no claim about the resulting reserved state |
| 2. Pending certificate | `(q,v)`, `p`, symbolic planned occupancy | yes | no claim that pending apply has executed; not a resource-outcome field |
| 3. Destination assignment | `p`, address/direct-coordinate calculations, reserved bounds | yes | no resulting-state uniqueness claim |
| 4. Prefix vacancy certificate | `p` and the pure prefix-occupancy recurrence | yes | no assumption that relocation steps succeeded |
| 5. Per-entry observation certificate | source payloads, planned writes, codec/address laws | yes | does not assume `ReservedRepR`, `unique` or `DirectCover` of execution output |
| 6. Materialized-resource certificate | `r`, `PreparedResources`, bound operation call graph | yes | no claim about merge count observed after execution |

`ReservedRepR r' h`, `unique(r')`, `DirectCover(r')` and `mergeCalls = 0` remain conclusions of
`rebuild_preserves`; none is a `RebuildPlanCorrect` premise.

**Whole-table representation decision (landed, `APEIRON-EXTRACT-REP-RESOURCE-W5-T1`; extended by the
W6-T4 interface repair and the W6-T5 production-dictionary binding).**
`proof/corpus/Growable/Representation.lean` (sha256
`4dc7df700d8fc9eb50a88755934f5e632c702f77b7de0992114611a70c3139c1`) defines whole-table
`RepR`/`ReservedRepR`/`PendingCommittedR` over `Occupied`, `LiveSetEqual`, `ObservationCorrect`,
`UniqueRepresentation`, `PhysicalCoverage`, `DirectIndexCorrect`, `AddressCorrect`,
`SpillDirectoryConsistent` and `LengthCorrect`, and proves `repR_liveSet`, `repR_unique`,
`repR_coverage`, `repR_directCover`, `repR_directory`, `reservedRepR_iff_rebuiltView`,
`pendingCommitted_reserved`, `pendingCommitted_committed`, `repR_singleSource` and
`reservedRepR_singleDestination`. W6-T4 added `reservedRepR_of_components` and the
`reservedRepR_{liveSet,unique,coverage,directCover,directory,length}` projections. All declarations are
registered in `proof/corpus/AxiomAudit.lean` and build in the closed cone
`{propext, Classical.choice, Quot.sound}` (`lake build` 3473 jobs); the uniqueness mutation control and
the W6-T4 flat-extent `256 × lanes + 1` mutation are red on the mutated source and green on the
restored source. Existing `Growable/Relocation.lean` relations observe one selected
source/destination and must not silently gain complete-live-set, decoded-uniqueness, default,
length or composite-`DirectCover` meaning; W5 connects them by projection rather than redefinition.
The reachable constructor/adapter witnesses are not established here: W6-T1 landed the disjoint-write
frames (proved and audited in `Growable/Relocation.lean`), and the W6-T3 certificate redesign below
replaced the opaque `Prop`-field `RebuildPlanClauses` with concrete structures and eliminators.
W6-T4 also strengthened the target-side certificates in `Growable/Resources.lean` (sha256
`687c0e6d381e18481017f9dac92535bb77cf4010473f9d89743a148bf00a089e`): `ReceiptMatchesStorage` now
carries the target lanes, the flat extent `256 × plan.lanes`, the 256-entry storage and receipt
directories and each region's tag/payload lengths, exposed by `receiptMatches_flatExtent`,
`receiptMatches_spillDirectory` and `receiptMatches_regionLengths`; and `TargetGeometryCertificate`
carries the target `committedLive`, uniqueness, coverage, addressing, the target `directAfter` with
`DirectIndexCorrect`/`DirectCover`, directory consistency and length, exposed by the
`rebuildPlan_target*` projections. W6-T5 landed: `RebuildPlanClauses` binds
`cloneHeld`/`addressHeld`/`locateHeld` to the generated carrier clone, `address_held state.home` and
`table.spill_locate_held … 16#usize` (`cloneHeld_eq`/`addressHeld_eq`/`locateHeld_eq`);
`RebuildPlanCorrect_R` requires the production `keyRep` dictionary; `ResourceAdapterSimulates.prepared`
carries the same equality; and the `rebuildPlan_stored_operations_binding`/
`rebuildPlan_regen_operations_binding` projections expose the three bindings for both carriers,
registered in `proof/corpus/AxiomAudit.lean`, closed cone `{propext, Classical.choice, Quot.sound}`,
`lake build` 3473 jobs, and the `locateHeld := Vacant 1` mutation is red then green
(`Growable/Resources.lean` current sha256
`31d17cca7bc25707be6d96628ab4730f65e67da8bbdbb741adf368611dacc576`). W6-T2 landed:
`Growable/Rebuild.lean` (sha256
`8db24388133248ef29ab483c740e1b0462e10d826bc31cdbdc1d2c1238c19e9a`) proves
`word_stored_rebuild_preserves` and `word_regen_rebuild_preserves` over
`word_{stored,regen}_relocate_all`, from source `RepR`, the publication equality,
`PreparedResources_R` and `RebuildPlanCorrect_R`; relocation takes no merge or callback parameter, so
no merge is invoked. Both are registered in `proof/corpus/AxiomAudit.lean` (sha256
`56d73dbe2bebda144f2d7a13a46fbb7b5fb471a6e28a03c46b6084472e69fa85`), with `lakefile.lean` sha256
`bdd14f9fa9fdbd79ce72b02d2c52da320c9950f103847a7ceb1361056899600a`, build in the closed cone
`{propext, Classical.choice, Quot.sound}` (`lake build` 3474 jobs), and the relocation-write mutation
is red on the mutated source and green on the restored source. Dependency order: operation bindings
→ exact per-entry symbolic writes → disjoint-write framing → traversal equals the final symbolic
prefix → source enumeration derives the target observations → `reservedRepR_of_components`. The W7-T2
premise interface is landed. `Growable/Search.lean` (sha256
`d349cd481bdc2a33d68c7b1487b64e72ff8bfc98be89f17d3029e585382c7600`) defines `QueryAddressCorrect`
with `queryAddressCorrect_vacant_firstFit`, `FreshLengthBound` with `freshLengthBound_vacant`,
`GrowableCallbackLaws` with `growableCallbackLaws_identity`/`growableCallbackLaws_generated_once`,
and `wordMerge_newFirst` as the genuine trivial discharge; `Growable/Resources.lean` (sha256
`3708d69cbdc85616ac69a35b58e81915844ba91961cbfa99f95416f98a812461`) defines
`DecisionSymbolicCorrespondence` with `decisionSymbolic_{found,vacant,vacant_valid}` and proves
`resourceAdapter_refusal_stutter`. All are registered in `proof/corpus/AxiomAudit.lean` and build in
the closed cone `{propext, Classical.choice, Quot.sound}` (`lake build` 3474 jobs); the spill-shard
mutation is red on the mutated source and green on the restored source. Per Sol, `FreshLengthBound`
and the arbitrary-merge semantic identity are input-side; `DecisionSymbolicCorrespondence`,
`QueryAddressCorrect` and `GrowableCallbackLaws.exactlyOnce` require source binding/extraction; and
`ResourceAdapterSimulates.refused` carries the no-clone-on-`Err` obligation. Astra approved all four
as explicit conditional trust-boundary premises, with discharge paths in the
[pending-composition premise boundary](../kernel-boundary.md#pending-composition-premise-boundary);
no existing headline or grade changed. The W7 retry `prepared_rebuild_refines_insertWith`, composing
`rebuild_preserves` with the commit equality, `SearchDecisionCorrect` and `CallbackLaws` to canonical
`Apeiron.Table.insertWith`, is in flight and is not asserted. The
focused gate is bounded `lake build Growable.Representation AxiomAudit` in `proof/corpus/`; a missing
live entry, duplicate decoded key or incorrect default/coverage rejects its named projection or witness.

The resource predicates and publication transport are landed in
`proof/corpus/Growable/Resources.lean` (current sha256
`3708d69cbdc85616ac69a35b58e81915844ba91961cbfa99f95416f98a812461`; the W6-T5 digest was
`31d17cca7bc25707be6d96628ab4730f65e67da8bbdbb741adf368611dacc576`; `APEIRON-EXTRACT-REP-RESOURCE-W3-T1`,
`W4-T1`, the W6-T3 certificate redesign, the W6-T4 interface repair, the W6-T5 production-dictionary
binding and the W7-T2 premise interface). The focused gate is bounded
`lake build Growable.Resources AxiomAudit` in
`proof/corpus/`; a wrong scalar receipt comparison or swapped/discarded plan, storage or outcome
must reject its named equation or witness. No capacity accessor or allocator axiom is admissible.
The module defines `CapacityBound_R`, `EveryScalarCapacityCheck_R`, `PreparedResources_R`,
`RebuildPlanCorrect_R` and `ResourceAdapterSimulates`, and proves `word_capacity_bound_holds_iff`,
`word_stored_publish_prepared_carries`, `word_regen_publish_prepared_carries`,
`preparedResources_of_publication` and the two `word_{stored,regen}_reachable_publication_witness`
headlines. W6-T5 adds the `rebuildPlan_stored_operations_binding`/
`rebuildPlan_regen_operations_binding` projections. All are registered in
`proof/corpus/AxiomAudit.lean` and build in the closed cone
`{propext, Classical.choice, Quot.sound}`. The two reachable-publication headlines are `proved`
only as implications conditional on `ResourceAdapterSimulates`; that premise is an explicit
unproved named trust-boundary premise at the
[local boundary](../kernel-boundary.md#resource-adapter-simulation-premise), following the
`SearchDecisionCorrect` pattern, and the headlines prove no actual adapter reachability.

**Concrete certificate redesign (landed, `APEIRON-EXTRACT-REP-RESOURCE-W6-T3`).**
`PreparedResourceClauses` and `RebuildPlanClauses` are concrete proof-carrying structures over the
real generated types, not opaque `Prop`-field parameters. `PreparedResourceClauses` carries the
exact generated state, `PendingInput`, `RebuildPlan`, materialized `RebuildReserved` storage,
`ReservationReceipt`, held carrier, `decode`/`addressHeld`, receipt/storage length agreement
(`ReceiptMatchesStorage`), materialized flat/spill vacancy (`MaterializedVacant`), pending decoded
identity and address, and a `CoreCallTrace` with `NoCoreEffects` (zero allocation, initialization
and merge calls). `RebuildPlanClauses` carries the source/live-entry bijection
(`SourceEnumerationCertificate`), the unique pending decision (`PendingDecisionValid`), a bounded
pairwise-disjoint destination assignment (`DestinationInBounds`, `destinationDisjoint`), the
symbolic prefix write/spill-vacancy equations (`applyPlannedWrite`, `prefixInitial`, `prefixStep`,
`prefixVacancy`), decoded identity/value/carrier/address/tag observations
(`EntryObservationCertificate`), logical-key injectivity, coverage (`PlannedCoverage`) and
direct-coordinate coverage (`PlannedDirectCover`). Eliminators:
`rebuildPlan_{destination,destinations_disjoint,prefix_vacancy,entry_observation,logicalKey_injective,coverage,directCover}`
and `preparedResources_{materializedVacant,zero_allocation_merge}`. All are registered in
`proof/corpus/AxiomAudit.lean` and build in the closed cone `{propext, Classical.choice, Quot.sound}`
(`lake build` 3473 jobs); the certificate negative control is red on the mutated source and green on
the restored source. The headline statements `word_capacity_bound_holds_iff` and
`word_{stored,regen}_publish_prepared_carries` are unchanged; `preparedResources_of_publication` and
the two reachable-publication witnesses now consume the concrete certificates, and reachability
stays conditional on the unproved `ResourceAdapterSimulates` premise.

**Source binding of composition notation.** The theorem notation below names current production code.
Both composites are now production-factored functions and production-called by
`Table::apply_prepared_rebuild`; neither is yet a standalone extraction root:

| Contract notation | Real `src/table.rs` body denoted | Extraction status and smallest missing artifact |
|---|---|---|
| `relocateAll(relocate_entry,t,p,r)` | `Table::relocate_all(&mut self, plan: &RebuildPlan, reserved: &mut RebuildReserved<R, V>)` (`src/table.rs`), the production-factored traversal of `plan.entries` that calls the production-factored `Table::relocate_entry` once per entry; its flat/spill writes move one checked-plan entry. `Table::apply_prepared_rebuild` calls it after payload materialization. | Production-factored and production-called. Both monomorphic `relocate_all` roots are in the landed `proof/ApeironWord.lean` sha256 `c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2`, fresh for `src/table.rs` sha256 `2a60a19dae103fbd6981824c11ca7b436ef344071cd979905746c9ec531d5856`. The per-entry roots remain present and their four flat/spill headlines remain proved. The missing semantic artifact is the fold/composition theorem over the per-entry calls. |
| `applyPendingAndCommit(rebuilt,decision,f,q,v)` | Production source signature: `Table::apply_pending_and_commit<F: Merge<V>>(&mut self, reserved: RebuildReserved<R,V>, decision: SearchDecision, merge: &F, sigma: Sigma, value: V, held: R, address: &TableAddress)`. `Table::apply_prepared_rebuild` computes `decision` after `relocate_all` through the production rebuilt-state decision loop, then calls this real body. The body matches only the supplied well-formed class/location, merges or inserts once, updates `len` only on vacancy, and commits `reserved`. | **Option (b), implemented at grade `anchor`.** `locate_rebuilt` returns the value-only `SearchDecision` and the private `enum Site` is removed; `locate_rebuilt` and `Flat::index` remain in the unextracted `apply_prepared_rebuild` orchestration. The stored/regen wrappers taking `SearchDecision` are present and rooted in the landed `proof/ApeironWord.lean` sha256 `c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2` (audit clean, `proof/` `lake build` 1708 jobs). The wrappers that root the old body are rejected because `usize::try_from(...).ok()?` reaches four external axioms. |

These bindings do not treat line slices as extracted functions. `relocate_entry` is called by
`Table::relocate_all`, which `apply_prepared_rebuild` calls; `apply_pending_and_commit` remains the
production commit body. The selected signature and both carrier wrappers have passed strict
regeneration into the landed `proof/ApeironWord.lean` above, so `applyPendingAndCommit` is the
production-factored extraction phase; the fold/composition theorem over the traversal stays open.

Reconciliation (`APEIRON-EXTRACT-REP-RELOCATION-CONTRACT-T3`; implementation status recorded by
`APEIRON-EXTRACT-REP-RELOCATION-RECORD-T4` and `APEIRON-EXTRACT-REP-RELOCATION-RECORD-T5`): the status
above is bound to current `src/table.rs`; no premise, gate, grade or requirement changes. The current
source still runs payload initialization in `Table::apply_prepared_rebuild`, before relocation. The
joint handoff decision above requires the next source slice to move it after reserve-all into the
unextracted adapter and pass the held carrier through the published envelope.
Strict extraction acceptance for both carriers (`Stored<i64>` and `Regen<I64WordCodec>`) was recorded
at the per-entry checkpoint above; that checkpoint is superseded by the current source revision. On
current `src/table.rs` sha256
`2a60a19dae103fbd6981824c11ca7b436ef344071cd979905746c9ec531d5856`, the traversal, pending insertion
and commit are production-factored as `Table::relocate_all` and `Table::apply_pending_and_commit`.
Option (b) is implemented in source: `locate_rebuilt` returns `SearchDecision`, the private
`enum Site` is removed, and `apply_pending_and_commit` consumes the precomputed decision. Both
`apply_pending_and_commit` roots are present in the landed `proof/ApeironWord.lean` sha256
`c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2` (fresh for that source, audit
clean, `proof/` `lake build` 1708 jobs). The old pending-commit root was rejected by the fail-closed
audit.

#### Pending apply/commit extraction decision (Sol, APEIRON-EXTRACT-REP-RELOCATION-CONTRACT-T4)

**Selected option (b), grade `anchor`.** The pending commit consumes a precomputed
`SearchDecision`; it does not call `locate_rebuilt`. This is the same decision/apply split already
used by `Table::apply_get` and `Table::apply_insert` and refined in
`proof/corpus/Growable/Search.lean`. It changes no canonical operation, admission domain, callback
law or resource/refusal contract.

The exact source change is:

```rust
pub(crate) fn apply_pending_and_commit<F: Merge<V>>(
    &mut self,
    mut reserved: RebuildReserved<R, V>,
    decision: SearchDecision,
    merge: &F,
    sigma: Sigma,
    value: V,
    held: R,
    address: &TableAddress,
)
```

`locate_rebuilt` changes its return type from private `Site` to the value-only `SearchDecision` and
constructs the corresponding `Found`/`Vacant`/`Exhausted` plus `Flat`/`Spill`/`NoIndex` pair; remove
`Site` when it has no remaining caller. `Table::apply_prepared_rebuild` calls `locate_rebuilt` after
`relocate_all` and passes its result to `apply_pending_and_commit`. The commit matches the five well-formed
class/location combinations; spill branches use the shard carried by `SearchLocation::Spill`, and
`address` remains only for the vacant spill tag. `key` leaves the commit signature because search,
not apply, consumes it. The `Exhausted/NoIndex` branch remains unreachable under the pending vacancy
certificate in `RebuildPlanCorrect`. `word_stored_apply_pending_and_commit` and
`word_regen_apply_pending_and_commit` add a `SearchDecision` argument, remove their `key` argument,
and call this real production body; their existing `proof/regen.sh` root names remain unchanged.
They are accepted only after the strict generated audit is clean. This source change is implemented
on `src/table.rs` sha256
`2a60a19dae103fbd6981824c11ca7b436ef344071cd979905746c9ec531d5856`: `locate_rebuilt` returns
`SearchDecision`, the private `enum Site` is removed, and `locate_rebuilt`/`Flat::index` remain in the
unextracted `apply_prepared_rebuild` orchestration. The two wrappers are present and rooted in the
landed `proof/ApeironWord.lean` sha256
`c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2` (audit clean, `proof/`
`lake build` 1708 jobs).

Option (a) is rejected. Replacing `usize::try_from(...).ok()?` in `Flat::index` with casts would
either reduce the portable target domain or add a `usize`-width premise; modeling checked conversion
would enlarge the extracted primitive surface. Neither change is needed for the commit mutation.
Option (c) is rejected because it would leave the real merge, occupancy write, `len` update and final
resident/region commit outside extraction; `prepared_rebuild_refines_insertWith` could then prove
only relocation preservation plus a conditional model-level composition, not refinement of the Rust
commit body.

**Premises and theorem impact.** `SearchDecisionCorrect` is unchanged. For the rebuilt storage the
proof uses the pure `rebuiltView(t,reserved)` table value obtained by replacing only `resident` and
`shards`; this lets the existing `entryAt`, `VacantAt` and `SearchDecisionCorrect` contract apply
without introducing a second source transition. The new extracted apply theorem assumes
`SearchDecisionCorrect (rebuiltView t r') ... q decision ...`. The unextracted production decision
loop and the proof of that premise remain outside the extracted-apply claim.

`RebuildPlanCorrect` is unchanged: its pending certificate already supplies one symbolic
found-or-vacant result, sufficient reserved spill load and the exclusion of exhaustion. Relating that
certificate and `rebuild_preserves` to the actual post-relocation `SearchDecisionCorrect` witness is
a composition obligation, not a new plan premise. `CallbackLaws` is unchanged and still supplies
purity, totality and denotation for the one new-first merge call; relocation invokes no merge.

The source-linked composition statement becomes:

```text
prepared_rebuild_refines_insertWith:
  RepR t h -> extractedPublish_R(adapter_result) = ok prepared ->
  PreparedResources_R t (q,v) adapter_result prepared ->
  RebuildPlanCorrect_R t (q,v) adapter_result prepared ->
  prepared = Prepared { plan := p, storage := r, outcome := outcome } ->
  relocateAll(relocate_entry,t,p,r) = r' ->
  SearchDecisionCorrect(rebuiltView(t,r'),q,decision) -> CallbackLaws f ->
  extractedApplyPendingAndCommit(t,r',decision,f,sigma,v,held,address) = t' ->
  RepR t' (Apeiron.Table.insertWith f q v h)
```

The query `q` is related to `sigma`, `held` and `address` by the existing carrier/address clauses of
`RebuildPlanCorrect`; it need not remain a runtime commit argument. This statement adds no premise
beyond the already recorded relocation result, `SearchDecisionCorrect` and `CallbackLaws`; it makes
the real extracted commit call explicit. Its canonical target remains
`Apeiron.Table.insertWith` at
`/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/Basic.lean:80-81`.

**Audit evidence and digest correction.** Rooting the old pending-commit body with Aeneas
`505b6ca-dirty` exits 1 at `proof/regen.sh`'s audit. `Flat::index` contributes
`Usize.Insts.CoreConvertTryFromU32TryFromIntError.try_from`,
`core.option.Option.Insts.CoreOpsTry_traitTry.branch`, `core.result.Result.ok` and one bare external
`axiom`; no external axiom is admitted. The dirty generated artifact has sha256
`eedf281fbc82a5ff8efb9366f1a5659c229a24eb4fc4310494751b76efae64c5`. The pre-task
`58d0ba3e871cd75ae84d5cae7d7966242d3309b1a14ec1b032a26cdd5606e64b` checkpoint was uncommitted and
is unrecoverable. On the current `src/table.rs` sha256
`2a60a19dae103fbd6981824c11ca7b436ef344071cd979905746c9ec531d5856`, option (b) removes the
`locate_rebuilt`/`Flat::index` path from the commit root: the two `apply_pending_and_commit` roots are
present in the landed `proof/ApeironWord.lean` sha256
`c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2` (audit clean, `proof/`
`lake build` 1708 jobs). The
`89c58fdd35df102d62dd07231a8c8d52d5b102cb684a40923030c6ac2768a9de` checkpoint, the prior
`relocate_all`-in-flight status and the four-root combined regeneration bound to the earlier source
revision are historical.

**Trust boundary and existing headlines.** The selected root covers the real pending merge, selected
flat/spill write, occupancy update, fresh-key `len` update and final vector commit under
`SearchDecisionCorrect`, `RebuildPlanCorrect`, `PreparedResources` and `CallbackLaws`. It does not
cover the decision loop or its correctness, `Flat::index`, planner/reserver, allocator or payload
materialization realization, executor/`std::thread::scope`, callback panic freedom outside
`CallbackLaws`, rustc/LLVM, ABI, GPU/SPIR-V or the `unsafe` container. The Rust/canonical closed cone
and separate Slang empty cone do not change. The four existing per-entry relocation headlines are
unaffected: their source body, wrappers, premises, theorem statements and `proved` scope do not use
the pending decision or commit signature.

**Theorem split.** The earlier table row is too strong if `rebuild_preserves` names only the extracted
per-entry relocation: relocation has not yet applied the pending canonical write. The exact
relocation headline is therefore:

```text
rebuild_preserves:
  RepR t h ->
  extractedPublish_R(adapter_result) = ok prepared ->
  PreparedResources_R t (q,v) adapter_result prepared ->
  RebuildPlanCorrect_R t (q,v) adapter_result prepared ->
  prepared = Prepared { plan := p, storage := r, outcome := outcome } ->
  relocateAll(relocate_entry,t,p,r) = r' ->
  ReservedRepR r' h /\ unique(r') /\ DirectCover(r') /\ mergeCalls = 0
```

`ReservedRepR r' h` means every old observation is represented exactly once with the same decoded
key and value; it is representation preservation of the canonical `Apeiron.Table.HMap`, not yet an
`insertWith` result. The insertion theorem is the separate source-linked composition headline stated
in the pending apply/commit decision above.

Its canonical target is exactly `Apeiron.Table.insertWith` at
`/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/Basic.lean:80-81`, under
`RepR t (Apeiron.Table.HMap (Sigma × K) V)`. New-first merge occurs exactly once in the pending apply;
relocation invokes none. `rebuild_preserves` is now proved in `Growable/Rebuild.lean`; the
composition headline is not. This premise extension weakens no existing
`proved` result: the growable relocation/refinement rows were already anchors, canonical
`insertWith` retains its proved grade, and the historical fixed-shard headlines have a different
revision-bound representation.

**Per-entry relocation result (proved at flat and spill scope).** `proof/corpus/Growable/Relocation.lean`
proves the per-entry step for both carriers and both planned destinations:

- `word_stored_relocate_entry_flat_preserves` and `word_regen_relocate_entry_flat_preserves` cover a
  planned flat destination;
- `word_stored_relocate_entry_spill_preserves` and `word_regen_relocate_entry_spill_preserves` cover a
  planned spill destination, with `nested_region_set_readback` and `nested_region_payload_set_readback`
  carrying the nested `Vec::set` payload/tag readback through the outer region vector.

Each states that one extracted `word_stored_relocate_entry`/`word_regen_relocate_entry` call leaves the
source table unchanged and writes the selected canonical observation to its planned destination
(`ReservedRepR` at `.flat index` or `.spill shard index`), with no merge parameter and therefore no
merge. All four headlines are registered in `proof/corpus/AxiomAudit.lean` and build in `proof/corpus`
under the closed cone `{propext, Classical.choice, Quot.sound}` (`lake build` 3470 jobs at the
four-headline checkpoint). The
rejecting control `proof/controls/relocation.sh` (receipt `docs/evidence/relocation-controls.md`)
mutates `tag(address.place) -> 0` in the generated spill write: step 2 reddens the four spill
headlines and the restored source rebuilds green. The default control's baseline and restored
freshness steps are green on the landed `proof/ApeironWord.lean` sha256
`c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2`, which roots the six
`relocate_entry`/`relocate_all`/`apply_pending_and_commit` wrappers for both carriers. Grade `proved`
at this per-entry scope only.

**Traversal and commit results (proved; composition open).** Four further headlines are proved in the
same closed cone and registered in `proof/corpus/AxiomAudit.lean`:

- `word_stored_relocate_all_preserves` and `word_regen_relocate_all_preserves`
  (`Growable/Relocation.lean`) state that the rooted traversal preserves any invariant that is
  initially established and preserved by each rooted per-entry step. The traversal takes no callback
  argument and the invariant is a parameter, so this is the fold skeleton, not the semantic
  postcondition.
- `word_stored_apply_pending_and_commit_eq_apply_insert` and
  `word_regen_apply_pending_and_commit_eq_apply_insert` (`Growable/Refinement.lean`, now a
  `lakefile.lean` root) identify the rebuild commit root with the ordinary extracted `apply_insert` on
  `rebuiltView`, under the input-side carrier equation `hhold` for the pending carrier.

These four build in `proof/corpus` under `{propext, Classical.choice, Quot.sound}` (`lake build` 3471
jobs). Digests: `Growable/Relocation.lean` sha256 `0604166a…`, `Growable/Refinement.lean` sha256
`3e21b9de…`. Grade `proved` at this traversal/commit-equation scope only.

The handoff requires no statement change to these eight proved headlines: the four per-entry
flat/spill results, the two traversal results and the two commit equalities retain their exact scopes.
They establish neither `PreparedResources` nor a whole-table relation. After the source lane moves
initialization and adds publication, strict regeneration must preserve all existing roots and
`spill_locate`/`spill_locate_held`; the bounded generated audit/build, bounded corpus build with all
eight `#audit` registrations, and existing relocation control must revalidate them on the new
source/generated identity. Any statement change returns to Sol rather than being accepted as a
resource-handoff consequence.

Landed: the production-factored `Table::relocate_all` and the selected decision-consuming
`Table::apply_pending_and_commit` roots for both carriers in `proof/ApeironWord.lean` sha256
`c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2` (audit clean, `proof/`
`lake build` 1708 jobs). W6-T2 `rebuild_preserves` over the traversal landed in
`Growable/Rebuild.lean` (sha256 `8db24388…`) as
`word_stored_rebuild_preserves`/`word_regen_rebuild_preserves`; still open:
`prepared_rebuild_refines_insertWith`, which additionally needs `SearchDecisionCorrect` and
`CallbackLaws`; payload initialization remains at the mismatched current source location in
`Table::apply_prepared_rebuild` until the next source task moves it. `PreparedResources_R` and
`RebuildPlanCorrect_R` are now Lean predicates in
`Growable/Resources.lean`, with the reachable publication witness conditional on the explicit
unproved `ResourceAdapterSimulates` premise; the W6-T1 disjoint-write frames are landed; and the
W6-T3 certificate redesign (that checkpoint's `Growable/Resources.lean` sha256 `563f8196…`) replaced
the opaque `Prop`-field clauses with concrete structures and eliminators. W6-T4 interface repair is
landed: the missing artifacts named above are supplied by `reservedRepR_of_components` and the
`reservedRepR_{liveSet,unique,coverage,directCover,directory,length}` projections in
`Growable/Representation.lean` (current sha256
`4dc7df700d8fc9eb50a88755934f5e632c702f77b7de0992114611a70c3139c1`), and by the strengthened
`ReceiptMatchesStorage` and `TargetGeometryCertificate` with the `receiptMatches_*`/`rebuildPlan_target*`
projections in `Growable/Resources.lean` (current sha256
`31d17cca7bc25707be6d96628ab4730f65e67da8bbdbb741adf368611dacc576`); the bounded gates are green in
the closed cone (`lake build` 3473 jobs) and the flat-extent `256 × lanes + 1` mutation is red then
green. W6-T5 landed: `RebuildPlanClauses` binds `cloneHeld`/`addressHeld`/`locateHeld` to the generated
carrier clone, `address_held state.home` and `table.spill_locate_held … 16#usize`;
`RebuildPlanCorrect_R` requires the production `keyRep` dictionary; `ResourceAdapterSimulates.prepared`
carries the same equality; and the `rebuildPlan_stored_operations_binding`/
`rebuildPlan_regen_operations_binding` projections expose the bindings for both carriers, closed cone
and `lake build` 3473 jobs, with the `locateHeld := Vacant 1` mutation red then green. W6-T2
`rebuild_preserves` landed in `Growable/Rebuild.lean` (sha256
`8db24388133248ef29ab483c740e1b0462e10d826bc31cdbdc1d2c1238c19e9a`) as
`word_stored_rebuild_preserves`/`word_regen_rebuild_preserves`, registered in
`proof/corpus/AxiomAudit.lean` (sha256
`56d73dbe2bebda144f2d7a13a46fbb7b5fb471a6e28a03c46b6084472e69fa85`) with `lakefile.lean` sha256
`bdd14f9fa9fdbd79ce72b02d2c52da320c9950f103847a7ceb1361056899600a`, closed cone, `lake build` 3474
jobs, relocation-write mutation red then green. Whole-table
`RepR` is now landed in `Growable/Representation.lean` (W5,
`APEIRON-EXTRACT-REP-RESOURCE-W5-T1`), so that clause is no longer a blocker; the
traversal invariant is instantiated to the semantic postcondition by `word_stored_rebuild_preserves`
and `word_regen_rebuild_preserves`. The composition headline `prepared_rebuild_refines_insertWith`
stays unproved (W7 retry in flight) and `RebuildPlanCorrect` remains a premise. Kani stays `tested`.

The semantic adjudication gate is **MISSING** until `GrowableRelocation.rebuild_preserves` and
`GrowableRefinement.prepared_rebuild_refines_insertWith` exist, build under the bounded Lean gate,
are registered in `AxiomAudit.lean`, and fail their relocation/merge controls. The traversal
preservation skeleton, the commit-equation headline and `GrowableRebuild.rebuild_preserves`
(`word_{stored,regen}_rebuild_preserves`) are present and registered; the W7 retry composition
`prepared_rebuild_refines_insertWith` is in flight, so the gate stays open.

**Extraction roots and freshness (present).** The two monomorphic wrappers are present and rooted:
`word_stored_relocate_entry` and `word_regen_relocate_entry`, for
`Table<i64,i64,WordHome,16,Stored<i64>>` and
`Table<i64,i64,WordHome,16,Regen<I64WordCodec>>`. Each takes the real table, `&RebuildPlan`, a
materialized `RebuildReserved<carrier,i64>` and `entry_at`, calls the real `relocate_entry`, and
returns the table and reserved storage. The production source is `src/table.rs` sha256
`2a60a19dae103fbd6981824c11ca7b436ef344071cd979905746c9ec531d5856`, where option (b) is implemented
and the traversal, pending insertion and commit are production-factored as `Table::relocate_all` and
`Table::apply_pending_and_commit`. The landed `proof/ApeironWord.lean` sha256
`c8de66e3b760c45b8dd12aa4eaa74841c4cb9b7eb2656af5d4999bb9e5548fe2` is fresh for that source, audit
clean, and roots all six wrappers: `word_stored_relocate_entry`, `word_regen_relocate_entry`,
`word_stored_relocate_all`, `word_regen_relocate_all`, `word_stored_apply_pending_and_commit` and
`word_regen_apply_pending_and_commit` (`proof/` `lake build` 1708 jobs). The earlier
`89c58fdd35df102d62dd07231a8c8d52d5b102cb684a40923030c6ac2768a9de` checkpoint at source
`52d7f8ec4fca9367aaeb4c7ab31317084c81f04c33740f47ab8cd02163ab0a22` retained both per-entry wrappers
and the then in-flight stored/regen `relocate_all` definitions without a generated
`axiom`/`sorry`/`admit`; it was a clean working checkpoint, not a landed combined regeneration, and
the old pending-commit wrappers failed the audit through their transitive `Flat::index` path. The
smallest missing artifacts are the traversal fold/composition theorem and its refinement.

**Trust consequence.** Under the selected boundary, the unextracted prepared-resource adapter owns
both capacity reservation and payload materialization. The reviewed source still performs
`Flat::initialize` and `Region::initialize` at the start of `Table::apply_prepared_rebuild`; moving
those calls into the adapter before any rooted relocation call is a source residual. The Lean premise
checks the resulting materialized shape; it does not prove allocator behavior, initialization
execution, clone purity/termination or panic freedom. Those remain adapter/callback realization
obligations. The executor, planner, search decision, rustc/LLVM, ABI, GPU/SPIR-V and the `unsafe`
container remain outside this claim.

#### Search decision/apply interface (Sol decision, APEIRON-EXTRACT-REP-W11-T2)

**Decision and grade.** Composite `Table::locate` is not selected as an extraction root in its current
shape. It has the same `PinnedJoinBoundary` trigger as `plan_rebuild`: a probe loop carries a borrow of
a generic-carrier field, tests an inner occupancy guard, and branches on computed match and
classification state. No tested whole-body reshaping removes that trigger without changing the
search algorithm. The search is therefore factored into one production decision and one production
apply; this contract is grade `anchor`, while the supplied context-join reproducer is grade `tested`.

The unextracted decision walks exactly the admitted direct candidate, window prefix and mandatory
cyclic sweep. Its output is the concrete pair

```text
SearchDecision = (Found | Vacant | Exhausted,
                  Flat(index) | Spill(shard,index) | NoIndex)
```

Only `(Found,Flat(i))`, `(Vacant,Flat(i))`, `(Found,Spill(s,i))`,
`(Vacant,Spill(s,i))` and `(Exhausted,NoIndex)` are well formed; these map respectively to
`Site::{FlatFound,FlatVacant,SpillFound,SpillVacant,NeedSpillGrowth}`. The decision carries no value
or mutable reference. The rootable straight-line apply receives the decision, the selected flat/spill
region, `sigma`, key/address, default and, for insertion, value/merge. Read returns the indexed real
value only for `Found` and the real default otherwise. Insert updates that indexed real value for
`Found`, initializes that indexed real slot and occupancy authority for `Vacant`, and routes
`Exhausted` to the already prepared growth/resource outcome. It invokes merge exactly once and uses
the same region access and commit body called by production; no reference or model transition is
introduced.

**Named premise — `SearchDecisionCorrect`.** For a reachable `RepR` state and query `q`:

1. `Found(r,i)` is returned iff `q` has its unique live representative at in-bounds `(r,i)`.
2. `Vacant(r,i)` is returned iff `q` is absent and `(r,i)` is the first fit in the defined direct,
   prefix and sweep order. A direct vacancy proves absence only under composite `DirectCover`.
3. `Exhausted` is returned iff `q` is absent and the applicable flat/spill search space has no vacant
   lane; insertion therefore requires prepared growth rather than treating exhaustion as semantic
   refusal.
4. An occupied direct non-match does not prove absence. A window vacancy records the first-fit
   candidate and may stop later prefix windows, but it never stops the all-group match sweep.
5. Every absence result uses either the `DirectCover` direct-empty certificate or complete finite
   coverage of every possible representative location; tag/live authority, full `(Sigma,K)` identity,
   uniqueness, bounds and the seven-bit/cyclic geometry are preserved.

`search_decision_simulates` is the composition law: `SearchDecisionCorrect` plus the extracted apply
refinement yields canonical total-by-default `Apeiron.Table.get`; for insertion it yields
`Apeiron.Table.get_insertWith_self` at `q` and `Apeiron.Table.get_insertWith_ne` at every unequal key.
Focused Rust tests can exercise direct/spill found, first-fit vacancy, window-vacancy-plus-late-hit,
direct-empty with and without `DirectCover`, full exhaustion and unequal-`Sigma` cases. Kani can
exhaust bounded geometries and mutations of sweep or first-fit order. Those results support grade
`tested`/bounded evidence only; they do not prove `SearchDecisionCorrect` for the generic domain.

**Bounded decision evidence.** Three exhaustive small-model tests run the real `Stored<i64>`
decision body over every ordered prefix of a fixed three-key family and every query at every prefix,
comparing the decision to a model written from the geometry:

- `tests/table.rs::search_decision_matches_the_two_lane_spill_oracle_for_every_small_prefix` uses a
  placement-only, constant-address `N = 1` path (no direct candidate) and checks `Exhausted/NoIndex`
  on the empty and full absent cases, `Vacant/Spill(0,1)` as first fit after one write, exact
  `Found/Spill(0,i)` locations, and canonical default/present readback.
- `tests/table.rs::search_decision_matches_the_direct_resident_oracle_for_every_small_prefix` uses
  `WordHome` with keys `0`, `1`, `2`, whose certified coordinates own flat slots `0`, `1`, `2`. It
  checks `Exhausted/NoIndex` on the fresh table, `Found/Flat(k)` for a present key, first-fit
  `Vacant/Flat(k)` for an absent key, the exact slot, well-formedness and default readback.
- `tests/table.rs::search_decision_matches_the_direct_collision_spill_oracle_for_every_small_prefix`
  uses `SameLaneHome` at `N = 1`, where the first key occupies flat slot `0` and later keys spill.
  It checks `Found/Flat(0)`, `Found/Spill(0, j)`, first-fit `Vacant/Spill(0, j)`, exhaustion while
  the spill is empty and once it is full, unequal-`Sigma` identity and default readback.

The private unit control `spill_sweep_finds_a_late_match_after_a_window_vacancy` separately checks
that a prefix vacancy does not truncate a late match in the mandatory sweep. At the historical
revision recorded below, the bounded Kani harness
`search_decision_classifies_a_bounded_two_lane_spill` was present in `src/proofs.rs` with an explicit
`#[kani::unwind(257)]`. The attribute sets one global bound for every loop, so the bound is not the
two-lane scan width: `spill_locate` calls `scan::match_tag`/`scan::match_empty` at the per-group
width `n = N = 1`, at most two iterations. The floor is `Table::new`'s fixed construction loop over
`SHARDS = 256` (`src/table/partition.rs`), which every table constructor runs, plus one for the
unwinding assertion.

**Per-loop bound experiment (`APEIRON-PERF-KANI-T1`), result `tested`.** CBMC 6.8.0 supports
`--unwindset` and Kani 0.67 exposes it through `-Z unstable-options --cbmc-args`; the loop IDs are
read from the generated goto binary with `cbmc --show-loops`. Two bounded, single-threaded re-runs
were made. (a) Lowering only the reachable hot loops (`least_set_lane` 65, `spill_locate` /
`spill_locate_held` 4/16, `scan::match_tag` / `match_empty` 4, `grow_spill` / `Region::initialize` /
`slots_for` / `Vec::extend_with` 8) while `Table::new` stayed at 257 still did not terminate within
900 s (exit 124). (b) Lowering the default to 16 with `Table::new` and `drop_in_place::<[Region]>` at
257 and `least_set_lane` at 65 also did not terminate within 300 s. CBMC logs `Not unwinding loop
... iteration 4/16/8`, so the per-loop bounds took effect; the residual is `goto-symex` path
explosion dominated by slice-index panic paths (`core::slice::index::slice_index_fail`,
`Range<usize> as SliceIndex<[u8]>::index`) and allocation modelling
(`kani::mem::cbmc::same_allocation`, `ToISize::to_isize`), not loop-bound inflation. `--cbmc-args` is
an unstable Kani option and the loop IDs are mangled and change with monomorphisation, so
`--unwindset` is not stable enough to record as the acceptance gate. The reduced-shard real
constructor that would remove the 256-lane floor requires changing `src/table/partition.rs`
(`SHARDS`), `Table::new` and the hard-coded `& 255` in `src/table.rs::shard`, which are outside this
task's scope; it is returned as a Sol escalation request. The harness therefore stays `tested` /
bounded evidence and no mutation control could be run against it, because it does not terminate under
any in-scope bound. The exhaustive runtime oracle
`tests/table.rs::search_decision_matches_the_two_lane_spill_oracle_for_every_small_prefix` is the
`tested` evidence for the same property.
The direct lane cannot be reached without the first certified insertion running
`selected_lanes`/`rebuild_resident`, the unextracted resource path, so a Kani harness for
`Found/Vacant Flat` would include that path and the exhaustive runtime tests above carry it instead.
This is `tested`/bounded evidence only. The generic `SearchDecisionCorrect` premise remains open for
arbitrary reachable `RepR` states, both carriers, direct coverage, every admissible geometry and the
independent order/coverage lists. Its smallest missing proof artifact is a real-source monomorphic
extraction of `Table::search_decision` and `spill_locate` for both carriers, with a refinement
theorem deriving every field of `SearchDecisionCorrect` from reachable `RepR` and composite
`DirectCover`.

`GrowableSearch` refines the real apply under `SearchDecisionCorrect` rather than claiming the
decision loop is extracted. `GrowableRelocation`, `GrowableRefinement` and `GrowableWitnesses` carry
the same premise through rebuilt-state application, operation composition and joint witnesses.
`GrowableDirectory`, `GrowableSchedule` and `GrowableBatch` are unchanged by this split and retain
their routing, per-key-subsequence and grouped-assembly obligations. `GrowableSchedule` still
composes its implementation-specific per-key value-subsequence equation with canonical
`Apeiron.Table.build_eq_of_key_streams_eq`
(`Table/Basic.lean:167-174`). The external parallel-reduce twin is `Song/kernel/Web.lean`
`preduce_schedule_free`/`preduce_append`, read at
`/home/socol/Workspace/song-ref/lean/Song/kernel/Web.lean` revision `ae688f66`; it is a pointer to
Song-side licence, not an import into Apeiron's theorem or axiom cone.

**Spike evidence, grade `tested` (supplied; not rerun here).** Aeneas `505b6ca-dirty` with Charon
`0.1.254` fails `Could not match the contexts` at `Table::plan_rebuild`
(`src/table.rs:678:16-680:17`). A minimal generic `FlatSlot<T>` borrowed in a loop reproduces the
failure when an inner `;x.live` guard is followed by a branch on a computed `Option`/enum
classification from `T::address_held`; the scalar-guard form passes. Straight-line `Option`/`if let`,
`while` with an accumulated `Option`, and `usize` cast variants pass. `-loops-to-rec`,
`-loops-no-rec` and `-strict-joins` do not remove the failure. This classifies the join as pinned
Aeneas limitation (b), not unmodeled allocator case (c); further generic-planner reshaping is not a
dispatch strategy. Independently, reachable `try_reserve_exact` remains unmodeled.

**Refinement and trust boundary.** No claim covers the planner/reserver, allocator or global
allocator, allocation success/failure realization, executor or `thread::scope`, rustc/LLVM, ABI,
GPU/SPIR-V or the `unsafe` container. The search decision loop and `SearchDecisionCorrect` are also
outside the proved apply claim. This is not a model-versus-real-body substitution after factoring:
the extracted apply is the real executed region read/update and commit path, while the decision is a
controlled classification with a separately tested/Kani-checked contract. The separate empty-cone
Slang source-refinement chain is unchanged and imports none of this Rust/canonical resource
interface. A checked Aeneas primitive/tool model is required before an allocating public method can
become a root.

**Open.** REG-1/G1 remains **OPEN**. Extraction freshness was not rerun here. The current artifact is
spill-only and does not establish the decision/apply interface; a whole composite search retains the
same pinned context-join blocker. No proof or speed claim follows; only the bounded spike and refusal
observations carry grade `tested`.

#### Partition and spill-filter contract (Sol decisions, APEIRON-PERF-CONTRACT-T1/T2)

**Decision and grade.** This section ratifies the bounded source contract requested by the current
partition/validation performance cut. It is grade `anchor`: no source, extraction, theorem, test or
measurement is accepted by this text. The transactional refusal contract, fail-closed generated audit,
both carrier faces and `W7-SearchCommitBoundary` remain unchanged.

##### Partition validator

**`APEIRON-PERF-CONTRACT-T2` decision: option (b), retarget extraction.** Remove
`word_partition_is_intact` from the extraction roots. Do not reshape `Parts::is_intact` again for
Aeneas. The current intact-flag body, split-guard variants and a hoisted `buf_len` all fail in Aeneas
`505b6ca-dirty` with `Could not match the contexts`, moving the failure to the next borrowed
field/index/address construct. The former nested-loop early-return body fails with `Continue to outer
loops are not supported yet`. The validator's complete `&self` access pattern is outside this pinned
tool's translatable subset; another source rewrite has no evidenced terminating target.

The observed red gate is `proof/regen.sh --check` from the repository root: exit 2 at
`src/table/partition.rs:98:25-100:44`, before Lean emission. Charon `0.1.254` succeeds; Aeneas is
`505b6ca-dirty`. No option-(a) candidate is accepted, so there is no candidate generated-Lean result.

`Parts::is_intact` remains the production fail-closed runtime validator with this exact tested
contract:

1. initialize `intact` with `ends[0] = 0`, `ends[SHARDS] = buf.len()` and
   `buf.len() = input.len()`;
2. run the shard loop only while `intact`; for shard `s`, require
   `ends[s] <= ends[s+1] <= buf.len()`;
3. run its index loop only while `intact`; require every index to be in `input`, strictly greater than
   the preceding index in that shard, and routed by a fresh `home.address(sigma,key)` calculation to
   `s`; and
4. return `intact` after the loops. No nested loop contains `return`, `break` or `continue`, and no
   `seen` vector or coverage scan remains.

Strict local order excludes duplicate indexes in one shard. If one index occurred in two different
shards, deterministic equality-respecting routing of
that same input element would have to equal both shard numbers, contradicting their difference.
Therefore all `buf` indexes are distinct. The endpoint and length equations give exactly
`input.len()` in-range indexes; an injection between two finite sets of that cardinality is a
bijection, so no input index is omitted. Recomputed routing excludes misrouting. Consequently no
omitted, duplicated or misrouted index escapes.

`partition_chunk` satisfies the checker: its prefix sums tile a buffer of the input length; the
source loop visits each input index once in increasing order and writes it to the cursor for its
recomputed shard; each cursor advances once per routed index. Thus each shard run is strictly
ascending, every endpoint is the corresponding prefix sum, and the empty input satisfies all endpoint
and cardinality clauses. This is the required constructor-completeness argument. It is not yet a Lean
theorem. A same-shard duplicate and a same-shard reversal are mandatory rejecting tests; different-
shard routing rejection alone does not test the strict-order clause.

The source definition has temporal `O(input.len() + SHARDS)` cost in comparisons/address calls and
`O(1)` auxiliary space. The former `seen` allocation and coverage pass are removed; the partition
buffer itself is unchanged. Its positive and malformed-output cases, including same-shard duplicate
and reversal, remain `tested`: `cargo test --locked --lib partition_integrity` exits 0 with 9 passed,
0 failed. The source-mutation control remains required; this positive run does not waive the runtime
obligation or promote it to `proved`.

The extraction/refinement claim excludes the validator and therefore does not prove that this
runtime guard accepts every valid partition or rejects every malformed one. This is the same kind of
realization boundary as allocator and executor checks: tests exercise the executing guard, while a
future source-linked partition theorem must prove the partition output relation directly. The
canonical `Apeiron.Table.build` refinement needs stable per-key value subsequences and disjoint,
complete assembly; it does not need a theorem about this defensive Boolean checker. Removing the
checker root does not remove `GrowableSchedule.partition_preserves_key_streams` or permit that premise
to be assumed from a successful runtime check.

###### Extraction and bridge impact

| Artifact | Required effect of T2 |
|---|---|
| `proof/extract/lib.rs` | Delete the `word_partition_is_intact` wrapper so it cannot be mistaken for an accepted root. Other partition/parallel wrappers remain real-body specializations. |
| `proof/regen.sh` | Delete only `--start-from crate::word_partition_is_intact`; retain source digests, audit, freshness diff and the nonallocating F2/F4 roots. F3 reserve-all has no accepted root. |
| `proof/ApeironWord.lean` | Regeneration removes `word_partition_is_intact` and any `Parts::is_intact`/`Parts` declarations reachable only through it. Declarations reached by the schedule, summary/tag, `spill_locate`, relocation and publication roots remain; reserve-all stays unextracted. |
| `proof/corpus/Growable/{Geometry,Scan,Search,Relocation,Representation,Resources,Rebuild}.lean` | No current declaration references `word_partition_is_intact`, `Parts::is_intact` or `table.partition.Parts`; no statement or grade changes for this decision. Regeneration may still require mechanical repairs for changed generated structures from F2/F3/F4. |
| `proof/corpus/AxiomAudit.lean` | It audits no validator or partition-integrity headline; no line is removed. The fail-closed cone remains mandatory for every retained and new headline. |
| `proof/corpus/lakefile.lean` | No validator module/root exists; no manifest change follows from T2. A future `Growable.Schedule` module remains separately required and must be added and audited when implemented. |

No existing `proved` headline changes statement or grade. In particular, the relocation, rebuild,
resource, representation, search-apply, scan and canonical table headlines retain their exact scopes.
`SpillSummaryShape`, `SpillSummaryRecorded`, `SpillSummaryMissSound` and `SpillFingerprintExact`
remain the explicit `anchor` obligations below: they concern spill publication and filtered absence,
not partition-output validation, so removing the validator root neither discharges nor weakens them.
Extraction freshness is currently red, so historical green receipts are not fresh evidence for the
current source. The implementation-specific stable-partition/assembly theorem remains open; the
validator itself is `tested` only.

###### Flash extraction edit and order

One bounded Flash task owns exactly `proof/extract/lib.rs`, `proof/regen.sh` and
`proof/ApeironWord.lean`:

1. remove the wrapper and selector above;
2. run `proof/regen.sh` from the repository root to emit the retained roots and require the generated
   `axiom`/`sorry`/`admit` audit plus the `proof/` package build to pass;
3. inspect the generated file for absence of the removed validator root and presence of every
   currently retained root;
4. run `proof/regen.sh --check`; this is the task's single gate and must exit 0; and
5. in a throwaway copy, restore the removed wrapper and selector against the unchanged source and
   require the recorded Aeneas context-match failure, then discard the copy and rerun the unchanged
   gate green.

The control proves that the green extraction results from the declared boundary rather than an
unnoticed validator translation. Audit cleanliness remains part of both regeneration commands. After
this task, execute F2 and F3, then apply the primitive and extraction disposition below. F4 retains
the roots that already reach summary publication, `tag` and both `spill_locate` bodies; it does not
root reservation or allocator orchestration. Inspect the retained bodies before F4's
`proof/regen.sh --check`. Then run
the bounded full corpus build and existing semantic controls, followed by a final
`proof/regen.sh --check`. A green freshness gate does not upgrade the runtime validator or the open
partition theorem.

##### Summary resource and publication lifecycle

The following clauses are one contract. A partial implementation is not a transactional success.

1. **Checked shape.** Replace saturating or panicking summary sizing by checked multiplication,
   round-up and power-of-two arithmetic. The result is either a word count satisfying
   `bits = trailingZeros(wordCount) + 6 < 64`, or `SizeOverflow`. Include summary bytes in checked
   byte sizing.
2. **Fallible reservation.** Every nonempty region reserves its `summary : Vec<u64>` through the
   same `try_reserve_exact` error mapping as tags and payloads, then zero-initializes within that
   capacity. No `vec![0; words]` allocation remains. `RegionReservation` contains `summary :
   CapacityBound`; `ReservationReceipt`, `ReceiptMatchesStorage`, `EveryScalarCapacityCheck_R` and
   `PreparedResources_R` bind its checked target, observed capacity and materialized length. Empty
   regions may use an empty summary, which disables filtering.
3. **Reserve-all transaction.** A build computes every shard's checked tag, payload and summary
   target and successfully reserves all of them, plus any fallible worker-result storage, before a
   worker starts probing or applying entries. Before that common success point there is no
   `R::hold`, key/value/default clone, merge, relocation or publication. A reservation refusal starts
   no worker and returns no partial table. `build_prepared_shard` is therefore split into preparation
   of already-reserved/materialized storage and a nonallocating apply phase; a worker does not reserve
   its own region.
4. **Live publication.** For every fresh spill write and relocation, write the payload, set the
   summary bit, then publish the nonzero tag. The tag transition `0 -> tag(place)` is the occupancy
   publication point. No fallible operation or search-visible handoff occurs between summary
   recording and tag publication. Found-value updates change neither tag nor summary. Constructor and
   materialization publish only zero tags. Completed public states, prepared-region returns, worker
   results, rebuilt views and committed tables therefore satisfy `SpillSummaryRecorded`; a summary
   bit set for a still-vacant lane is harmless.
5. **Refusal.** `SizeOverflow` or `AllocationFailed` before the common reservation point leaves the
   committed table byte-for-observation unchanged, invokes no hold/clone/merge/relocation and returns
   no partial build. Allocator correspondence remains the explicit `ResourceAdapterSimulates`
   boundary; it is not inferred from vector length or a receipt value.

##### Exact spill-filter invariants

The following notation is proof-facing. `R` ranges over
`Stored<I64>` and `Regen<I64WordCodec>`; `r : table.Region R I64`;
`decode : Sigma -> R -> Option I64`; and
`addressHeld : Sigma -> R -> Result table.TableAddress` is the generated production carrier
operation. For a nonempty summary let `w = r.summary.length`,
`bits = Nat.trailingZeros w + 6`, and

```text
summaryIndex(place,bits) =
  ((((place >> 8) * 0x9E3779B97F4A7C15) xor
      (((place >> 8) * 0x9E3779B97F4A7C15) >> 29)) mod 2^bits).
summaryBit(r,place) = bit r.summary[summaryIndex(place,bits) / 64]
                              (summaryIndex(place,bits) % 64).
```

- **`SpillSummaryShape(r)`.** `r.tags.length = r.slots.length` and either
  `r.summary.length = 0`, or `w > 0`, `w` is a power of two,
  `bits < 64`, and for every `place : U64`,
  `summaryIndex(place,bits) < 64*w`, its word index is below `w`, and its bit index is below `64`.
  For resource-produced regions there is additionally a `RegionReservation.summary` whose
  `required_len = w`, whose observed capacity passes `word_capacity_bound_holds`, and whose checked
  target is the successful result of summary sizing for `r.slots.length`.

- **`SpillSummaryRecorded(r, addressHeld)`.** For every `i : Nat`, `sigma`, `held`, `value`, `tagAt`
  and `address`, if `i < r.tags.length`, `r.tags[i] = tagAt != 0`,
  `r.slots[i] = (sigma,held,value)`, and `addressHeld sigma held = ok address`, then
  `r.summary.length = 0` or `summaryBit(r,address.place) = true`.

- **`SpillSummaryMissSound(r, decode, addressHeld, querySigma, queryKey, queryAddress)`.** Under
  `SpillSummaryShape`, `SpillSummaryRecorded`, generated routing/address stability and equality-
  respect, `queryAddress` being the generated address of `(querySigma,queryKey)`, and the region being
  `shard(queryAddress)`,
  `summary_maybe(r,queryAddress.place) = false` implies there do not exist `i,held,value` with a
  nonzero tag at `i`, payload `(querySigma,held,value)` at `i`, and
  `decode querySigma held = some queryKey`. The held-query form replaces the last equation by the
  generated `matches_held` relation and requires generated address agreement. This conclusion is
  routed-spill absence only; it does not by itself prove flat absence or global absence. A summary
  hit proves only “maybe”.

- **`SpillFingerprintExact(r, addressHeld)`.** For every live index and reconstructed address as in
  `SpillSummaryRecorded`,
  `r.tags[i] = tag(address.place) = 0x80 OR ((address.place >> 8) AND 0x7f)`, hence the tag is nonzero.
  Equal decoded logical keys have equal generated addresses and therefore equal tags; a different tag
  excludes logical equality. For every scanned group, bit `lane` of `match_tag(tags,wanted)` is set
  iff that lane's byte equals `wanted`, and `match_empty` is exact for zero bytes. Equal fingerprints
  do not imply key equality: a hit still performs the full `Sigma` plus carrier comparison.

`SpillSummaryShape`, `SpillSummaryRecorded` and `SpillFingerprintExact` become fields of the
whole-state representation (and its reserved/rebuilt counterpart), not permanent assumptions hidden
inside search. Their constructor and preservation theorems must cover prepared-shard fresh writes,
ordinary fresh spill writes, spill growth, both relocation carriers and pending commit. Found updates
preserve them by framing. `SpillSummaryMissSound` is a derived theorem from shape, recordedness and the
existing carrier/address laws; it must not be introduced as an axiom or independent premise.

Until those proofs land, the four names remain explicit `anchor` premises at
`W7-SearchCommitBoundary`. Their discharge path is: generated summary arithmetic and
`summary_set`/`summary_maybe` lemmas; receipt/materialization transport under
`ResourceAdapterSimulates`; per-write and frame proofs for both carriers; whole-state representation
composition; then the miss-soundness derivation and locator composition.

The filtered locator changes `SearchDecisionCorrect`. Its non-found certificate is the disjoint sum
of (a) direct-empty under `DirectCover`, (b) complete finite coverage, or (c) the direct candidate
having been checked and a `SpillSummaryMissSound` certificate for the routed spill. The existing
`sweepComplete`/`vacancyDoesNotStopSweep` clauses apply only when the summary says “maybe”; they are
not asserted for a sound summary miss. First-fit vacancy order is unchanged. Both `spill_locate` and
`spill_locate_held` must be linked to this sum. Consequently these invariants discharge part of
`SearchDecisionCorrect`; they do not establish `DecisionSymbolicCorrespondence`, callback laws,
resource reachability or `prepared_rebuild_refines_insertWith`. W7 stays frozen until its full
reopening condition is met.

##### Bridge impact

The canonical `Apeiron.Table` object, logical `(Sigma,I64)` key, direct index, spill shard and carrier
decoding equations do not change. The bridge nevertheless requires **re-proof**, not only
re-validation: generated `table.Region` gains `summary`; spill writes change; receipt structures gain
a summary bound; `RepR` is strengthened; and filtered absence changes `SearchDecisionCorrect`.

Required new or revised proofs are the four named spill-filter invariants;
`word_{stored,regen}_search_decision_simulates` and `search_decision_simulates`; both
`word_{stored,regen}_relocate_entry_spill_preserves` headlines;
`word_{stored,regen}_apply_pending_and_commit_eq_apply_insert`;
`reservedRepR_of_components` and all `RepR`/`ReservedRepR` projections;
`word_capacity_bound_holds_iff`, `word_{stored,regen}_publish_prepared_carries`,
`preparedResources_of_publication` and both reachable-publication witnesses; and
`word_{stored,regen}_rebuild_preserves`.
The open `prepared_rebuild_refines_insertWith` statement must consume the strengthened
`SearchDecisionCorrect`; it gains no shortcut around the other W7 premises. Canonical corpus
theorems, `scan_exact`, the two flat relocation headlines, the generic relocation-traversal
headlines, `resourceAdapter_refusal_stutter` and the `WordHome` addressing/codec theorems require
re-validation only at their unchanged statements. They do not establish the new fields.

##### Serialized implementation and proof order

Each row is a separate task and file owner. Shared extraction and Lean lanes are serialized. A
mutation runs in a throwaway copy and the unchanged source is rerun; no mutation edits this working
tree.

| Order / owner | Files (at most three) | Single gate | Rejecting control |
|---|---|---|---|
| Flash F1 | `proof/extract/lib.rs`, `proof/regen.sh`, generated `proof/ApeironWord.lean` | `proof/regen.sh --check` from repository root after one successful regeneration; extraction audit and the package build must exit 0 | In a throwaway copy, restore the removed validator wrapper and selector: Aeneas must reproduce the context-match failure; discard it and rerun the retargeted gate green. |
| Flash F2 | `src/table.rs`, `src/table/resources.rs` | `cargo test --locked --lib table` | Independently omit a live-write `summary_set` and omit the summary capacity bound; the actual-path recordedness and injected summary-reservation-refusal tests must fail, then pass restored. |
| Flash F3 | `src/table/parallel.rs`, `tests/parallel/main.rs` | `cargo test --locked --test parallel` | Move one shard reservation behind worker start in a throwaway copy; injected final-shard refusal with counted hold/clone/merge must fail the zero-effects assertion, then pass restored. A deterministic reservation-failure seam is **MISSING** and must be added by this gate-first task before the ordering claim can close. |
| Flash F4 | `proof/ApeironWord.lean` | `proof/regen.sh --check` | Mutate one generated summary bit index or fingerprint shift; freshness must reject. This is a freshness control only; the semantic controls below remain required. No allocator or reserve-all root is added. |
| Sol S1 | `proof/corpus/Growable/Scan.lean`, `proof/corpus/AxiomAudit.lean` | bounded `lake build Growable.Scan AxiomAudit` in `proof/corpus/` | Mutate tag shift or one `match_tag` lane equation; the exactness headline must fail. |
| Sol S2 | `proof/corpus/Growable/Representation.lean`, `proof/corpus/AxiomAudit.lean` | bounded `lake build Growable.Representation AxiomAudit` | Remove recordedness/fingerprint from one live spill witness; a `RepR` constructor or projection must fail. |
| Sol S3 | `proof/corpus/Growable/Resources.lean`, `proof/corpus/AxiomAudit.lean` | bounded `lake build Growable.Resources AxiomAudit` | Drop or swap one summary receipt bound; receipt transport or a reachable-publication implication must fail. |
| Sol S4 | `proof/corpus/Growable/Relocation.lean`, `proof/corpus/AxiomAudit.lean` | bounded `lake build Growable.Relocation AxiomAudit` | Remove `summary_set` from a generated spill relocation; both carrier spill-preservation headlines must fail. |
| Sol S5 | `proof/corpus/Growable/Search.lean`, `proof/corpus/AxiomAudit.lean` | bounded `lake build Growable.Search AxiomAudit` | Treat a summary hit as presence or a miss as flat absence, or bypass the full-key compare on a fingerprint collision; the miss/search theorem must fail. |
| Sol S6 | `proof/corpus/Growable/Refinement.lean`, `proof/corpus/Growable/Rebuild.lean`, `proof/corpus/AxiomAudit.lean` | bounded `lake build Growable.Refinement Growable.Rebuild AxiomAudit` | Omit pending summary recording or reverse merge order; the corresponding commit/rebuild headline must fail. |

Run only one Sol row at a time because every row owns `AxiomAudit.lean` and the shared corpus lane.
After S6, run the bounded full `lake build` in `proof/corpus/`, then the existing relocation controls,
then `proof/regen.sh --check`. Only after those pass may a separately authorized W7 task attempt the
strengthened composition. The Rust barrier and strict REG-1 campaign follow on the same recorded
source/tool identity; neither is replaced by these focused gates.

##### Extraction primitive and reserve-all disposition (`APEIRON-PERF-CONTRACT-T3`)

**Decision: avoid both unmodeled primitives in the executing source; do not admit models or generated
axioms.** The failed F4 candidate (supplied sha256 `84f844bf…`) reaches the real
`Region::publish_live`, `summary_index`, `summary_set`, `summary_maybe`, `tag`, `spill_locate` and
`spill_locate_held` bodies, but the fail-closed audit rejects
`core.num.Usize.trailing_zeros` and `alloc.vec.Vec.is_empty`. Excluding the summary bodies would remove
the source link needed by `SpillSummaryRecorded` and `SpillSummaryMissSound`; recording the block would
leave the authorized source repair unused. Adding either declaration to an accepted external set is
rejected: it would enlarge the generated trust boundary and would not prove correspondence to the
Rust implementation.

The minimal source repair is confined to `src/table.rs`:

```rust
fn summary_trailing_zeros(mut value: usize) -> u32 {
    let mut zeros = 0;
    while zeros < usize::BITS && value & 1 == 0 {
        zeros += 1;
        value >>= 1;
    }
    zeros
}
```

Replace both `self.summary.is_empty()` calls by `self.summary.len() == 0`, and replace all three
executing `trailing_zeros()` calls in `summary_words`, `summary_set` and `summary_maybe` by
`summary_trailing_zeros(...)`. The loop is total, including zero, and executes at most
`usize::BITS` iterations with `O(1)` auxiliary storage. Keep the existing checked
`summary_trailing_zeros(words) + 6 < u64::BITS` guard. Add a same-file unit test comparing the helper
with `usize::trailing_zeros` at zero, odd/even values, powers of two through the top bit and
`usize::MAX`. This test is Rust evidence for semantic preservation; it is not a Lean primitive model.

No hand-written Lean definition replaces either rejected axiom. Regeneration must emit the real
`table.summary_trailing_zeros` loop and the summary callers without `axiom`, `sorry` or `admit`.
`Growable/Scan.lean` must then prove an audited `summary_trailing_zeros_eq` loop specification: on
every generated `Usize`, the returned `U32` is the fixed-width trailing-zero count (word width at
zero, otherwise the least set-bit index). The generated summary arithmetic uses that lemma; on the
nonzero power-of-two lengths in `SpillSummaryShape` it specializes to the existing
`Nat.trailingZeros w + 6` formula. This adds a source-linked arithmetic lemma, not an accepted
primitive or an axiom, so the admissible external set and closed cone remain unchanged.

###### F4 roots and transactionality boundary

Add no dedicated root for `Region::reserve_empty`, the serial reserve loop in `Table::build_at`, or
parallel `reserve_all_shards`. They call `try_reserve_exact` and allocator-backed vector operations;
rooting them would cross the ratified `ResourceAdapterSimulates` boundary. Rooting `build_at` or
`build_at_parallel_phased` is also invalid as a workaround because both reach the deliberately
unrooted `Parts::is_intact`; the parallel body additionally reaches `std::thread::scope`. A wrapper
that restates only the reserve order would be a model, not the real transaction.

The retained relocation, prepared-shard and decision-conditioned apply roots already reach
`publish_live`, `summary_set`, `summary_maybe`, `tag`, `spill_locate` and `spill_locate_held` for both
carriers. F4 therefore needs no new wrapper or selector after the source repair: regeneration of
those transitive bodies is the accepted F2 summary-kernel extraction. F3's serial reserve-all before
worker start carries grade `tested` only after `cargo test --locked --test parallel` and its injected
final-shard-refusal control establish zero hold/clone/merge effects; this decision does not rerun
them. The serial `build_at` reserve loop likewise has no accepted extraction claim. Allocation success/failure,
reservation capacity, reserve-all transactionality, worker start ordering and the public build's
refusal path remain outside the `proved` cone under `ResourceAdapterSimulates`; executor,
`std::thread::scope`, allocator/global allocator and rustc/LLVM realization remain excluded. This is
an explicit `tested`/trusted boundary, not an admissible-domain reduction.

###### Invariant and search consequence

The primitive repair changes no summary bit, index, disabled-filter or fingerprint semantics.
`SpillSummaryShape`, `SpillSummaryRecorded`, `SpillSummaryMissSound` and `SpillFingerprintExact`
retain the statements above and remain `anchor` until their source-linked proofs and controls pass.
The helper lemma discharges the trailing-zero arithmetic inside shape; `publish_live` extraction
supplies the payload -> summary -> tag body needed by recordedness. It does not establish
materialization, reserve-all reachability or allocator correspondence.

`SearchDecisionCorrect` retains the same three-way non-found certificate: direct-empty under
`DirectCover`, complete finite coverage, or a checked direct candidate plus
`SpillSummaryMissSound` for the routed spill. A summary miss certifies only routed-spill absence; a
summary hit does not certify presence, and fingerprint hits still require full `(Sigma,K)` comparison.
First-fit vacancy order is unchanged. No search theorem may replace the miss certificate with the
helper equation alone.

###### Bounded Flash edits and regeneration order

The four-file repair is two serialized tasks because a task owns at most three files:

1. **Primitive source repair:** own `src/table.rs`; add `summary_trailing_zeros`, the replacements and
   same-file equivalence test above. Single gate: `cargo test --locked --lib table`. In a throwaway
   copy, change the helper's zero count by one; the equivalence test must fail, then the restored gate
   must pass.
2. **F4 regeneration:** own generated `proof/ApeironWord.lean` only. Do not change
   `proof/extract/lib.rs` or `proof/regen.sh`: their retained selectors already reach the required
   summary/tag/locator bodies, and neither file may gain a reservation/build selector. Run
   `proof/regen.sh` once to replace the generated file, inspect that both summary callers, both
   locators, `publish_live` and `tag` are present and that the audit is clean, then run the single
   gate `proof/regen.sh --check`. In a throwaway copy, change one generated summary bit index or tag
   shift; freshness must fail, then the restored gate must pass.

F2 and F3 source/test gates precede these tasks. After F4, run Sol S1 through S6 in the serialized
order above, the bounded full corpus build, semantic controls, and a final
`proof/regen.sh --check`. The current executable status remains red: the source repair and regenerated
artifact have not run in this decision task. The candidate translatability gate after the exact
source repair is **MISSING** until the primitive source task executes it.

### Read-path specialization reconciliation

The lifecycle artifacts
[`intent`](read-path-specialization.md),
[`spec`](read-path-specialization.md) and
[`plan`](read-path-specialization.md) govern the authorized read-only specialization. This
section reconciles that contract with the storage contract above; it changes no representation,
admission, refusal, proof grade or acceptance threshold.

`get_at`/`get` remain total-by-default reads of `HMap (Sigma × K) V`. A read may return default only
from one of three certificates: direct empty under composite `DirectCover`; checked direct exclusion
plus `SpillSummaryMissSound` for the routed spill; or complete finite match coverage. Summary-positive
misses retain the complete sweep and full `(Sigma,K)` comparison. A false negative is forbidden; a
summary or fingerprint false positive may cost that bounded sweep. No spill vacancy is an absence
certificate.

The production read may inline resident address/index/classification/application and use a
match-only spill path. It must not compute insertion first fit, vacancy or exhaustion. The insertion
path retains the five well-formed `SearchDecision` cases, exact first fit, growth and transactional
refusal. Reads allocate, hold, clone and merge nothing and do not mutate state. Found-key values,
occupied-direct spill fallback, disabled-summary behavior, carrier identity and separate `Sigma`
remain unchanged.

The generated Stored/Regen public-read bodies therefore change and require source-linked bridge
repair. `Growable/Scan` supplies summary/fingerprint exactness;
`Growable/Representation` carries those fields with uniqueness and `DirectCover`; and
`Growable/Search` repairs `SearchDecisionCorrect` to expose the direct/summary/coverage absence sum
and proves the specialized reads against canonical `Apeiron.Table.get`. Existing insertion,
relocation, resources, rebuild and pending-commit headlines retain their exact statements and grades
unless regeneration requires a mechanical dependency repair. Every revised/new headline remains
fail-closed in `AxiomAudit.lean` under `{propext, Classical.choice, Quot.sound}`.

Verification retains the focused Rust tests, micro harness, strict full-profile per-pair command and
semantic red/restore controls in the new plan. The strict ceiling remains interval maximum
`selected/incumbent <= 1.0` for every applicable pair. The recorded `pass=421 fail=1029` result stays
red; attainability is unestablished. No benchmark-specific dispatch, second container, cached summary
width, geometry change, first-empty rule, reduced matrix or W7 reopening is authorized here.

#### Parallel stable-scatter reconciliation

The [parallel stable-scatter specification](parallel-stable-scatter.md) resolves the pending stable
partition/assembly mechanism without changing the representation or any acceptance bar. The parallel
entry points retain serial `Parts::of` as the exact output oracle. They count and plan contiguous
input chunks on the calling thread, reserve and materialize all chunk-private index buffers, the final
initialized `Parts` destination, result metadata and every nonempty table shard before the first
`thread::scope`, then parallelize only:

1. stable scatter of each contiguous input chunk into its private pre-sized shard runs; and
2. copying each shard's runs into disjoint initialized destination slices in increasing input-chunk
   order.

The result has the same per-shard global input-index order, `ends`, loads and deterministic LPT bins
as `Parts::of`, independent of worker count or completion order. Scratch beyond final `Parts` is one
index per input plus fixed 256/257-entry metadata per chunk: `O(n + 256p)` words for `n` inputs and
`p` chunks, with no key/value clone or per-element allocation. No raw concurrent write or
`MaybeUninit` is admitted.

`Parts::is_intact` remains the production fail-closed validator at grade `tested` and remains outside
extraction. A sizing/allocation refusal occurs before worker start, invokes no hold/clone/merge/
relocation, and returns no table. Every started scoped task is joined before panic propagation. The
per-shard build/probe kernels, full-logical-key routing, LPT policy, worker/crossover budgets,
query-order callbacks, grouped readback and reservation/refusal variants are unchanged.

This source slice does not change extraction or an existing Lean headline. `Parts`, reserve-all and
`std::thread::scope` remain tested/unextracted. `GrowableSchedule.partition_preserves_key_streams`
remains the open implementation-specific theorem that must link exact per-key subsequences to proved
canonical `Apeiron.Table.build_eq_of_key_streams_eq`; the canonical theorem does not prove Rust
scatter, allocation, safe slice ownership or threads. All other `Growable/*` premises, statements,
grades and closed-cone audits remain unchanged.

Evidence is exact-output partition unit tests, the non-commutative and query-order equivalence tests
in `tests/parallel/main.rs` across worker counts, the reserve-all refusal control, a rerun of
`examples/micro_spatial.rs`, and the unchanged strict full-profile command. The supplied 46–55%
kernel and about 28% partition-plus-validation split remains an anchor until rerun. Every spatial pair
must still satisfy interval maximum `selected/incumbent <= 1.0`; missing, invalid, inconclusive,
concurrency-unmatched or address-face-unmatched pairs reject. G1/REG-1 remains **OPEN**.

### Historical record (superseded): two-container bounded reference

The sections below describe the retired `SafeTable`/`Map` reference. They are historical evidence,
not the current table.

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | Safe fixed-capacity lookup, replace and removal |
| Domain | `K: Eq`, arbitrary `V`, compile-time slot count `N` |
| Totality | Lookup/removal total; insertion returns `Result` on full absent-key insertion |
| Preconditions | None; allocation failure is absent because storage is an inline array |
| Postcondition | Successful observations equal a total function `K -> Option V` |

The first executable instance uses word keys and values at a small fixed capacity for extraction.
This bounds proof cost without narrowing the generic Rust reference's domain.

### Representation relation

| Layer | Symbol | Path |
|---|---|---|
| Safe Rust reference | `SafeTable<K, V, N>` | `src/safe_table.rs` |
| Production comparison | `Map<K, V>` | `src/map.rs` |
| Extraction wrapper | Monomorphic word-table wrappers | `proof/extract/lib.rs` |
| Extracted body | Generated safe-table operations | `proof/ApeironWord.lean` |
| Canonical definition | `Apeiron.Table.get`, `insertWith`, `replace` | Pinned corpus |
| Refinement theorem | Named safe-table denotation theorems | `proof/corpus/WordRefinement.lean` |

The wrappers fix word types and capacity. They call the reference body and do not restate its
transitions.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| Lookup | `get k` returns the unique stored value or `None`. | proved | Safe-table lookup refinement. |
| Replacement | Existing `k` changes to `v` and returns the old value. | proved | Canonical `insertWith replace`. |
| Fresh insertion | Absent `k` stores `v` when a slot is free and returns no old value. | proved | Canonical `insertWith replace`. |
| Full refusal | Absent `k` with `len = N` returns the original key/value and no successor. | proved | Extracted branch characterization and witness. |
| Removal | Present `k` becomes absent and returns its value; other keys are unchanged. | proved | Canonical replacement by `None`. |
| Production agreement | `Map` matches the safe reference after every selected operation. | tested | Differential trace test. |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Fresh key with free slot | `Ok(None)` and committed successor | `len < N` |
| Existing key at any occupancy | `Ok(Some(old))` and committed successor | key present |
| Fresh key at capacity | `Err(Full { key, value })` | key absent and `len = N` |
| Missing removal | `None`, state unchanged | key absent |

The optimized `Map::insert` remains allocation-backed and infallible except process allocation
failure. Its comparison uses a pre-reserved capacity and does not equate allocation failure with the
reference's logical full result.

### Evidence plan

1. Add focused nominal, full, replacement-at-full, removal and trace tests for the safe reference.
2. Compare production `Map` with the reference after every admitted trace operation.
3. Extract only monomorphic safe operations after the Rust API and tests are fixed.
4. Prove denotation equalities against the pinned corpus and audit every headline.
5. Mutate full handling, replacement order and removal in throwaway copies; each unchanged gate must
   fail and the original must pass.

### Trust boundary

Proof of the safe transition system does not transfer to optimized storage. Until its representation
relation is universal, the production comparison grade is tested.

### Open questions

- The optimized relation may require Kani over bounded layouts, a proof-friendly probe extraction or
  a redesign that shares a safe transition core. Select only after measuring the reference boundary.

## Plan - Single Table Refinement

> **Kani retirement.** The [user-directed retirement](kani-retirement.md) supersedes
> every Kani run, preservation task and barrier requirement below. Historical receipts remain
> revision-bound evidence; current work uses ordinary Rust gates and semantic mutation controls.

### Performance-preserving consolidation: lifecycle annex

Owner Astra; task `APEIRON-PERF-RESTORE-PLAN-W1-T1`; base
`c450c222905c050f793a23332708c230c658f82c`. Lead-authorized documentation exception in the supplied
checkout; no build lane, commit or production edit. Preserve existing dirty REG-1/spinor artifacts.
This annex governs current performance restoration and dispatch; earlier implementation sequences
below are historical checkpoints, not instructions to restart proofs of the current scan table.

```text
SCOPE: direction.md, plan.md, storage-refinement.md; performance-preserving consolidation
REGISTER: planning/measurement/refinement; no production implementation or speed claim
CORPUS: canonical Apeiron.Table/AnyRing; historical CountTable/Map at named revisions
REFERENCE: 71942db, 3a95bb5, 00a3629, dde9d58, f6db050; RFC 0001 A2/A7/G1
GATE: git diff --check; RFC renderer; performance/refinement requirement consistency search
GRADE: owner-approved requirement; historical numbers remain historical until rerun
```

#### Intent

Fold/hashtrinity -> Apeiron consolidation must preserve performance as well as canonical meaning.
The proof subject is the high-performance executing implementation, not a slower proof-only
replacement or a reference alongside unproved production. Extraction convenience, sampled
SwissTable/hashbrown-style confidence and historical speed alone cannot select an accepted
primitive. Recover historical performance; do not resurrect deleted source as accepted production.
Fixed N, bounded callbacks and linear whole-source reading remain qualifications, not exemptions.
Non-goals: production changes in this task, new roadmap, identity/admission reduction, proof-time
claims presented as runtime, or a universal slowdown claim about fixed shards.

#### Spec and evidence register

Owner definition of done (apeiron core): formally verified end to end (real-source extraction,
refinement to canonical `Apeiron.Table`/`AnyRing`, closed cone `{propext, Classical.choice,
Quot.sound}` with an empty cone for the Slang chain, mutation controls); competitive-or-SOTA speed;
not HashDOS-vulnerable; cryptographic strength not mandatory but achievable as a face. Full statement
in [direction](../direction.md#owner-definition-of-done-for-the-apeiron-core).

The [G1 matrix](../direction.md#required-r0-acceptance-g1--reg-1) is the contract: original spatial
parallel CountTable; serial operation/lookup cost; occupancy 1/2/4/8/16 at fixed `N = 16` with
direct-index-resident, offset, skewed and seeded-random coordinates; separate hit/miss/update/
insert/refusal; hashtrinity, hashbrown/foldhash and std baselines where historically applicable.
Keep identity (`10^8` colliding keys), round-trip sentinels, cross-process determinism, growth/remove,
ordered/commutative merge and Aeneas/Kani coverage. Missing operations and unmatched admission are
explicit open rows, not passes or silently narrowed populations. Timing requires full-key oracle
agreement first; default/new-first semantics, overflow, callbacks, effects and refusal remain explicit.

| Source / artifact | Registered role; grade and limits |
|---|---|
| `71942db` | Original spatial parallel CountTable versus packed hashtrinity: archival 3.16/10.27 ms. `--check history` reproduces `cargo bench --bench spatial_binning --offline` in an isolated `71942db` worktree with a pinned `4cc14471` Fold worktree and the 15 registered source digests; all 10 arms are oracle-checked before timing and 9 arms (18 cells at both sizes) are promoted to `measured` with the exact correctness-line digest. `hashtrinity_tuple` reproduces WRONG (off by 92 at 100k, 8236 at 1e6) and stays historical. i9-14900KF (32 threads) / rustc 1.97.0-nightly (17584a181 2026-04-13). No CountTable proof. |
| `3a95bb5` | Serial Map campaign, machine as above. `map_sweep` ns/key reproduced in an isolated `3a95bb5` worktree; it asserts no oracle, so those 16 rows stay historical. `probe_cost` instructions/lookup reproduced under `taskset -c 2 perf stat -e instructions:u -- target/release/examples/probe_cost`, three reps; presence oracle is the printed hit count `138800000` (every key found). Reproduced u64: apeiron 34.82, hashbrown+foldhash 35.14, hashbrown+fxhash 34.17, std 128.27; String: apeiron 93.35, hashbrown+foldhash 84.43, hashbrown+fxhash 92.34 (intervals in `matrix.json` `fresh.result`). The 7 probe cells are promoted; `fold_instr` has no independent oracle and stays historical. Not hashtrinity comparisons. |
| `00a3629`, `dde9d58`, `f6db050` | Supplied history: Map/Set/SafeTable deletion; CountTable plus 25 benchmark/example targets deletion; current fixed-shard scan Table installation, respectively. These revisions are retrieval inputs, not proof/performance acceptance. |
| `examples/reg1/{main,workload,correctness,performance,coverage}.rs`, `examples/reg1/coverage.json` | Existing uncommitted partial gate: identity/operation boundaries, correctness, determinism, full16 timing and coverage. Preserve and adapt, not replace with a one-workload optimization. |
| `examples/reg1/candidate.rs` | Benchmark-only cached-placement candidate; supplied 2.65–2.69x hashtrinity result is rejected. No production/refinement status. |
| Current REG-1 and verification receipts | Fresh strict full-profile receipt [`perf-remeasure-2026-09-19-post-spill.md`](../evidence/perf-ledger.md), sha256 `946b5d40…`: HEADLINE `registered=2592 load_bearing=2268 applicable=1450 pass=387 fail=1063 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`, exit 1; spill absent-miss and growth-insert cells fixed (lifecycle annex below). Kani 448.58 s, 62/62 green, one unreachable, largest harness ~94.23 s; freshness red in 3.28 s on line-comment drift. Reported measurements, not rerun here; details and missing receipts in `docs/plan.md`. |

Historical numbers are hypotheses for a registered rerun, never copied absolute thresholds.
Acceptance threshold (owner decision): for every applicable cell, `selected/incumbent <= 1.0` against
**every** named incumbent, never behind and ideally ahead. Named incumbents:
`hashtrinity::{Table,CoordMap,FreqMap}` serial and packed-parallel (`count_par`/`get_each`),
`hashbrown+foldhash`, `hashbrown+fxhash`, `hashbrown+rayon`, `std` SipHash, `foldhash`/`fxhash`
par16, and — per [hash-comparison.md](../hash-comparison.md) — `xxHash` XXH3/XXH64, `wyhash`, `ahash`
and `fnv`. The five named hashers are load-bearing on the opaque-key exact map face in
`examples/reg1/acceptance.rs`; demotion to informational or omission rejects. Applicability is
registered per cell/incumbent pair before timing with a semantic or API reason; no family-wide
exemption; missing, duplicate, invalid or inconclusive pairs reject.
`--accept --profile smoke` is diagnostic only; full acceptance requires all registered pairs. This
supersedes and is stricter than the RFC elapsed-median floor `selected/hashtrinity <= 1.25`, and
replaces the former historical-envelope question: there is no separate envelope tolerance. Cells
where the incumbent has no corresponding path (for example refusal) are reported without the bound.
`docs/hash-comparison.md` carries the axes (crypto versus non-crypto, throughput
small/large, latency, collisions/avalanche, HashDOS resistance, determinism, `no_std`, licensing,
verification) and the per-domain "good enough" reading for OS, languages, provers, games, servers
and aerospace. Historical Apeiron values are direction evidence, not a second gate.

Current strict result: the fresh post-spill full-profile receipt
([`perf-remeasure-2026-09-19-post-spill.md`](../evidence/perf-ledger.md),
sha256 `946b5d40…`) gives HEADLINE `registered=2592 load_bearing=2268 applicable=1450 pass=387
fail=1063 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false`, exit 1.
Spill absent-miss cells fell ~15–21x and growth-insert ~145–152x; the worst cell moved to the spill
present-hit path (~30.9x). No speed recovery is claimed; G1/REG-1 remains OPEN. The current
performance dispatch is the re-anchored bounded order in
[../roadmap.md#performance-re-anchor-and-bounded-dispatch](../roadmap.md#performance-re-anchor-and-bounded-dispatch);
its W1-T1 (five hashers load-bearing) has landed.
Pre-register sample/repetition counts, timed region, thread/cache conditions and uncertainty rule;
report raw samples and relative intervals. Inconclusive or failed required cells keep G1 open, even
if an aggregate is fast. Use matched before/after machine, compiler/features, inputs and baseline;
record every unavoidable mismatch rather than pooling incompatible samples.

#### Lineage: the performant structure against the algebra

The canonical denotation is `Apeiron.Table.build f d kvs = kvs.foldl (insertWith f) (fun _ => d)` with
`insertWith f k v m x = if x = k then f v (m x) else m x` (`proof/Apeiron/Table/Basic.lean:80,121`).
`get_build_acc` (`:134`) reads a key as the stream-order fold of `f` over that key's values.
`build_eq_of_key_streams_eq` (`:169`) licenses any regrouping that preserves each key's value
subsequence, for **every** merge; `build_gauge_invariance` (`:191`) licenses permutation only when
`f` is commutative and associative, and `build_gauge_invariance_mul` (`:224`) instantiates the
algebra's own `mul σ`.

The historical `CountTable` (`71942db:src/table.rs:91`, `3a95bb5:src/table.rs:103`) computes
`get a = |{i · addresses[i] = a}|`, which is exactly `build (fun new old => new + old) 0
(addresses.map (·, 1))` under the additive `i64` merge with default `0` (`src/merge.rs:110`). Its
shard-and-chunk schedule is `build_eq_of_key_streams_eq`: chunking is contiguous and regrouping is
by key, so each key's value subsequence is preserved and no commutativity is used. Region sizing,
the window descent and the tag line are placement, not semantics. The historical header's citation
is therefore correct, and the performant parallel structure is an instance of the canonical build.

The load-bearing premise is identity: the slot stores the *identity* word (`3a95bb5:src/table.rs:726`)
so slot equality equals key equality; `finish`/`spread` must be injective. That premise is asserted at
grade tested/bounded — `tests/the_counts_are_exact.rs` and the Kani lemmas cited at
`71942db:src/table.rs:36` — not `proved` in Lean.

Divergence: the current `Table` (`src/table.rs:6,74`) is a *bounded* 256xN fixed inline-shard
structure with `ShardFull`, total at most `256·N` keys (N=16 -> 4096). It neither holds the historical
1M-key workload nor shares the growable, load-sized structure. The general performant form
(`map`/`Map`) was deleted at `00a3629`; `CountTable` at `dde9d58`. No checked Lean refinement exists
for the counting/growable structure, for `spread`/`finish` injectivity, or for `locate` termination
(`region_slots_for`, `group_at` at `3a95bb5:src/table.rs:780,791` are exposed "for proof" but unproved
here); `proof/corpus/TableRefinement*.lean` refines the bounded fixed-shard Table only, and
`src/lib.rs:10` states that refinement pending.

Consequence: the equally-good solution is a growable, key-partitioned, cache-resident table generic
in `V` and merge `f`, whose `get` is proved equal to `get (build f d kvs)` under the injectivity
premise — the general `map` lineage generalized over `V`/`f`, not the bounded fixed-shard Table. It
is simultaneously the semantic refinement and the performance-bearing mechanism, so the choice does
not trade proof for speed. Selecting it replaces the current representation/ownership contract and
is a separately approved boundary decision.

#### Selection: Astra boundary and Sol contract

Target: one growable, key-partitioned `Table` with load-sized home regions, tag-line scan and
window-descent probe, generic over `V` and `f`, superseding the bounded fixed-shard representation;
no second production mode. Every operation carries `Sigma`; `N` becomes the lanes examined by one
tag-group primitive, not keys per home, so no positive `N` imposes a `256·N` ceiling. The canonical
schedule licence `build_eq_of_key_streams_eq` is `proved` and sufficient; no canonical theorem or
attribution changes, and an implementation-specific scheduling lemma must supply its per-key
subsequence premise. The Rust representation-to-canonical refinement is open, grade `tested`/bounded.

Qualifications that must survive: cache residency is a workload-dependent target, not an invariant;
a fixed probe-group width does not establish constant time, so the current universal O(1)-at-fixed-`N`
claim does not transfer to the growable representation and must not be renamed onto it. Load-bearing
open obligations: extracted `finish`/`spread` identity-word injectivity, `locate` termination and
first-fit specification, allocation/ownership semantics, SIMD/SWAR scan equivalence, and thread/join
simulation are unproved; the real widened multiply `id = ((x as u128).wrapping_mul(PHI as u128)) as u64`
has no corpus link yet. `build_gauge_invariance` is unnecessary unless same-key values are reordered
or partial worker accumulators are recombined. The bridge `proof/corpus/lakefile.lean` pins canonical
`8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f`, not the dirty sibling base; no sibling theorem enters
without a separately gated pin update. Aeneas/Charon translation of allocation, ownership, SIMD and
threads is a feasibility barrier to probe before implementation expands. The acceptance threshold is
match-or-beat the applicable incumbent per cell (`selected/incumbent <= 1.0`, owner decision), not a
historical-envelope tolerance.

Measured mechanism (first-cut diagnostic, superseded by the spill/bitmap rework and the fresh
receipt): the fixed-16 loss was the conjunction of direct-index-resident coordinates, `absent-miss`
and high-occupancy `present-hit`, and the fixed 16-slot linear scan. Under clustered probing the
first-cut Table was faster than hashtrinity (0.10-0.53x); under seeded-random it was at parity
(1.0-1.46x); the direct-path `absent-miss` cell was the worst at 5.16-6.76x. The selected mechanism
resolves a direct-index coordinate in O(1) while retaining bounded probes on clustered input: the
historical `coord::Addr` identity/placement split plus window descent, not a linear scan. Current
strict run (post-spill receipt, sha256 `946b5d40…`): HEADLINE `registered=2592 load_bearing=2268
applicable=1450 pass=387 fail=1063 missing=0 concurrency_unmatched=2 address_face_unmatched=4
full_acceptance=false`, exit 1; spill absent-miss fell ~15–21x and growth-insert ~145–152x, and the
worst cell is now spill present-hit (~30.9x, ~24.7 ns/op). No speed recovery is claimed.

Extraction feasibility (spike, Charon `0.1.254` / Aeneas `505b6ca-dirty`): the `coord.spread`/`finish`
body translates to Lean with no `axiom`/`sorry` — the identity-word arithmetic is extractable, so a
proved descendant is reachable for that premise. Unsupported as written: `std::thread::scope` (HRTB
lifetime constraint in `scoped.rs`), `return` inside nested loops in `locate` (`71942db:src/table.rs`
709-738), and the `Held::Ref<'a>` GAT (`3a95bb5:src/map.rs:394`); `Vec` allocation/push translates.
Identity injectivity is not derivable from the extracted model (Aeneas/Std has no odd-multiply lemma):
it needs a hand-proved `ZMod 2^64`/`BitVec` odd-multiply injectivity theorem plus a refinement link
unfolding `coord.spread`. Consequence: extract the sequential per-worker bodies, hoist threading behind
an unextracted execution boundary, restructure `locate` for extraction, and avoid GATs in the interface.

Extraction readiness (spike, same tools): the rendered growable core is structurally extractable.
Extractable as-is: `Vec` region allocation/zero-init/indexing, the explicit-search-state `locate`
with no `return` inside nested loops, the SWAR tag scan, sequential `insert_into`/`build`, and the
real `coord::spread`/`finish`/`unspread`. Rewrite required: `usize::next_power_of_two` and
`{u64,u16,usize}::trailing_zeros` surface as `axiom`; replaced by bounded loops the restructured
`build` extracts with **zero axioms**. Boundary: the SSE2 intrinsics and raw-pointer ops
(`_mm_loadu_si128`, `_mm_cmpeq_epi8`, `_mm_set1_epi8`, `_mm_movemask_epi8`, `Slice::as_ptr`,
`const_ptr::cast`) surface as axioms; keep the axiom-free SWAR kernel as the extracted implementation
and carry SSE2 as an unextracted optimization with an equivalence obligation anchored at
`tests/both_tag_paths_agree.rs`. `std::thread::scope` and the `Held::Ref<'a>` GAT stay outside the
sequential core. Aeneas already lowers scalar `+ - *` to checked monadic operations, so no
unchecked-arithmetic rewrite is needed. Verdict: in a position to reimplement with proper Aeneas
integration.

Every performance claim carries source/harness/input SHA-256, resolved revisions/dependencies,
actual rustc/Charon/Aeneas/Lean/Mathlib identities, machine, command, metric and interval. Register
the actual selected implementation symbol/build profile, not just HEAD on a dirty tree. For each
claim run baseline/red/restored controls: wrong oracle/result digest; wrong workload identity
(including occupancy/threads); deliberately slowed executing path or selected-implementation
mismatch. Each must reject for the intended reason; corrupted sample metadata alone does not
establish sensitivity to runtime regression. Controls mutate throwaway copies, never production.

#### Plan: serialized waves, not a new roadmap

Lead records integrated base and transfers file ownership before each dispatch; REG-1 paths remain
with their current owner until transfer. Use `worktree-agent-<task-id>` and exclusive R/E/B lanes
defined below. Routine harness/measurement tasks belong to Flash; Sol reviews proof/contract
sufficiency. A representation boundary exceeding three files requires Astra architecture
decomposition before implementation; routine measurement does not. Other AGENTS escalation
triggers, including governing-boundary changes, remain in force.

The existing Rust `reg1` executable remains the single aggregate gate: no Python/shell runner or
second gate executable. Its Rust history module invokes child `cargo`/git commands for historical
builds in isolated `/tmp/opencode` directories, never mutates the current or Fold working trees,
and removes every temporary worktree, including on failure. Source-file line limits remain binding.

| Wave / task | Owner; owned files (at most three per task) | Completion, one focused gate and control |
|---|---|---|
| `APEIRON-PERF-RESTORE-W1-T1` — first gate-first dispatch | Flash; edit `examples/reg1/main.rs`, create `examples/reg1/history.rs` and `examples/reg1/matrix.json` (exactly three files); isolated benchmark outputs | Restore/adapt the original spatial benchmark from `71942db` in temporary revision-pinned builds; register original serial targets from `3a95bb5` and every required matrix cell. No production optimization. Implement the history mode (currently MISSING) in the existing Rust gate: `cargo run --release --locked --example reg1 -- --check history --controls`; oracle-checked original spatial reproduction, exact source/command/digest receipts, matrix inventory and wrong-oracle/digest/workload controls. Exit 0 means framework valid; omitted matrix results remain MISSING and G1 false. |
| `APEIRON-PERF-RESTORE-W2-T1` — matrix module split | Flash; edit `examples/reg1/main.rs` and `examples/reg1/performance.rs`, create `examples/reg1/matrix.rs`; R | Move shared workload, measurement and matrix-cell machinery into `matrix.rs`; preserve every existing performance result and control while bringing `performance.rs` below 1,500 lines. Gate: the existing full performance command with controls. No matrix cell changes in this task. |
| `APEIRON-PERF-RESTORE-W2-T2` — current matrix adapter | Flash; `examples/reg1/matrix.rs`, `examples/reg1/performance.rs`, `examples/reg1/matrix.json`; R, after W2-T1 | Measure occupancy, coordinate and separated-operation strata; preserve correctness/coverage modules. Extend the existing full performance gate to enumerate registered cells and reject workload/digest mismatch. Every owned source file remains below 1,500 lines. Done as W2-T2 (160 strata cells measured in `strata.rs`) and W2-T3 (registration and enforcement, 3 new rejecting controls; `matrix.rs`, `performance.rs`, `matrix.json`). |
| `APEIRON-PERF-RESTORE-W2-T4` — threshold enforcement | Flash; `examples/reg1/matrix.rs`, `examples/reg1/performance.rs`, `examples/reg1/strata.rs` | Apply the owner threshold: each applicable cell passes only when `selected/incumbent <= 1.0`; report per-cell pass/fail and keep `reg1_pass=false` until every applicable cell passes. Gate: `--check performance --profile full --controls`. |
| `APEIRON-PERF-RESTORE-W3-T1` — historical/baseline campaign | Flash; `examples/reg1/history.rs`, `examples/reg1/matrix.json`, this plan, only after W2 releases `matrix.json`; isolated R | Complete historical serial and applicable hashbrown/foldhash/std adapters and matched before/after receipts. Extend W1's Rust history mode; gate `cargo run --release --locked --example reg1 -- --check history --controls`; run real slowed-path or implementation-mismatch control per claim. If dependency edits are needed, stop for a separate at-most-three-file manifest task. Full matrix acceptance mode is MISSING until implemented; measurement completion is not G1. Done: the gate exits 0 with 25 promoted cells (18 spatial, 7 serial probe), 20 historical, 2 missing, 26 boundary; 7 controls red at `child_exit=2`, two of them targeting promoted receipts; `history.rs` 1,498 lines, under the 1,500-line bound. `fold_instr` and the 16 `map_sweep` rows remain historical; the current `performance`/`correctness` boundary cells are unchanged. |
| Selection barrier | Lead selects from same-run matrix; Sol reviews domains, callbacks, admission and proposed canonical relation; no production edits | **Threshold (owner decision): match or beat every named applicable incumbent on every applicable cell, `selected/incumbent <= 1.0` (see the lifecycle annex for the incumbent list and per-pair applicability); no family-wide exemption, no historical-envelope tolerance.** Registered comparisons identify the mechanism (direct-index O(1) plus bounded clustered probes) and the selection is the growable key-partitioned structure above, with the Astra boundary and Sol contract recorded. Blocking G1: the 2 missing rows (identity 10^8, coverage census) and the current applicable cells above 1.0. No representation migration before this evidence. |
| Production recovery, then extraction/refinement | Flash implementation; Sol proof/contract review; separate serialized tasks each at most three named files, assigned after selection | Selected real build path, then `proof/regen.sh --check`, then `lake build` in the bridge as separate focused tasks. Real constructor witnesses, invariant preservation, successful/refusing branches and canonical `Apeiron.Table`/`AnyRing` denotation; fail-closed axiom audits and source/generated mutation controls. Unsupported Aeneas constructs require a bounded tool/refinement task, not a slower substitute. |
| Joint terminal barrier | Lead; no new implementation ownership | Same selected source identity passes the complete performance matrix, canonical extraction/refinement, S1–S8 and all AGENTS §6 integration gates and controls. No proof-only or speed-only closure. |

Existing partial commands (root) remain evidence inputs, not all-row acceptance:
`cargo run --release --locked --example reg1 -- --check correctness --profile smoke --controls`,
`cargo run --release --locked --example reg1 -- --check performance --profile full --controls`,
and `cargo run --release --locked --example reg1 -- --check coverage --controls`.
The all-row matrix acceptance command remains **MISSING**; the threshold is the owner's match-or-beat
rule above, not a historical-envelope tolerance.
No performance harness deletion without a replacement gate retaining its useful tested surface.

**Terminal acceptance:** the selected implementation is on the real build path; its actual Aeneas
extraction and canonical refinement close under the allowed axiom cone; the performance matrix is
green on that same source identity with matched before/after and rejecting controls. An unsupported
operation/admission row cannot disappear into a faster workload. Historical proofs stay attributed
to historical bodies. The current boundary document describes the executing fixed-shard layout;
changing its representation/ownership contract requires a separately approved boundary task.
Source refinement does not cover rustc/LLVM codegen, allocator/unsafe/SIMD/thread realization,
kernel ABI or GPU merely by association. Any selected implementation using them needs explicit
simulation obligations and external trust inventory; no desired simulation may be assumed.
All Lean gates use AGENTS §6's serialized, bounded wrapper; Kani/Lean latency is not runtime speed.

### Fixed-shard representation: current boundary and evidence

Owner Astra; base `d3423b2ec3dea6ddd0a8a668381d7749ecfdfff4`, initially clean main.
The lead assigns this supplied checkout, representation implementation, build/test acceptance,
per-body extraction verdicts and a building main commit/push. The unrelated `one-allocation`
branch is untouched. This section and `docs/kernel-boundary.md` supersede the growable contract
below and the obsolete storage statements in other documents; no epic closure is claimed.

**Intent/spec:** one executing `Table<K,V,H,N>` with 256 inline optional shards, `N` optional
full-key entries each, default `N = 16`. Checked `ShardFull` replaces allocator failure, not total
lookup. `Home<K>`/`Merge<V>` have concrete word instances; adapters call the same generic methods.
Build borrows a slice and clones entries in input order. Exact admission, representation relation,
callback premises, rejected alternatives and stack/translation trust cost are in the boundary.
`Home.lean` does not specify shard capacity, allocation or Rust simulation. Its collision facts do
not license discarding keys. Fixed capacity is an explicit resource restriction, not a reduced
statement of `HMap` or an alternative table outside the executing path.

Three serialized owned slices, IDs `APEIRON-TABLE-W8-T1` through `T3`, owner Astra:
`src/table.rs`, `src/merge.rs`, `tests/table.rs` (R, build/test);
`proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` (E, regeneration);
the two governing documents (no build lane). They share the authorized representation commit.

```text
SCOPE:      the three slices above; fixed inline storage, real-body extraction, bounded proof plan
REGISTER:   representation/extraction/test/planning; excludes completed Rust-to-HMap refinement
CORPUS:     Apeiron.Table "get_insertWith_self", "get_build_acc", "get_build_append"; native proved
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean
GATE:       cargo build/test --workspace --all-features --locked; proof/regen.sh --check (root)
            supplied red: capacity/try_reserve_exact/TryReserveError axioms after arrow removal
GRADE:      tested representation; clean extraction/elaboration, not proved table refinement
```

#### Open obligations: stable Sigma/address interface and the executing path

The fixed-shard `Table<K, V, H, N>` and its word refinement are the current evidence. The following
obligations are open; each names a gap to the O(1)-at-fixed-N canonical object, not a completed
result.

1. **Stable Sigma/address interface (open).** A read or write takes `(sigma, face, admittedInput)`.
   The stored full key carries `sigma`. `Apeiron.Table.get` stays total: a missing key returns the
   default, never a new refusal. No new refusal is introduced under `ShardFull`; the only refusals
   remain the absent-key full-shard refusal and the `N = 0` write refusal. The present
   `Table<K, V, H, N>` signatures take no `Sigma`, so this interface is not implemented.
2. **Face-routing of `WordHome` (open).** `WordHome` routes by the word bit pattern only and selects
   no face. Routing each face's admitted input to its home shard — elliptic pack, parabolic carried
   complete key, hyperbolic observation domain — is not implemented.
3. **Real `Cell` multiplication merge (open).** `WordMerge` is wrapping addition. A merge that calls
   the real `Cell::mul` at the selected `sigma` and refines `Apeiron.Table.insertWith` is not wired
   into the executing table.
4. **Fixed-capacity cost model (open).** The cost boundary at fixed `N`, bounded callbacks and
   nonconstant key length is stated in the spec. No cost model or measurement binds the executing
   path.
5. **Hyperbolic observation domain (open).** The discard face's observation domain, and its refusal
   of identity outside that domain, are not defined in the executing path.

Each obligation is named open; none is closed by the existing word-table refinement.

#### Observed gates and extraction verdict

| Command | Exit/result |
|---|---|
| `cargo build --workspace --all-features --locked` | 0 |
| `cargo test --workspace --all-features --locked` | 0; 94 tests and 2 doctests |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 |
| `cargo fmt --all -- --check` | 0 |
| `proof/regen.sh`; `proof/regen.sh --check` | 0 each; fresh, no `axiom`/`sorry`/`admit`, 1708-job build |
| `lake build` in `proof/corpus` | 1; obsolete `TableRefinement.lean:56:28` error below; WordRefinement builds |

Per-body command, root cwd, `b` each of `new get insert build`, existing writable `$out`:

```sh
target/aeneas/charon rustc --preset aeneas --dest-file "$out/$b.llbc" \
  --start-from "crate::word_table_$b" -- --edition 2021 --crate-type lib \
  --crate-name apeiron_word proof/extract/lib.rs
target/aeneas/aeneas -backend lean -dest "$out" -warnings-as-errors \
  -no-progress-bar "$out/$b.llbc"
```

Each Charon and Aeneas exit is 0; each generated file has the real method `def`, zero forbidden
declarations and passes `lake env lean <absolute-generated-file>` from `proof/` (exit 0).
Verbatim Aeneas output excerpts, ANSI color removed:

```text
[Info ] Generated: /tmp/opencode/apeiron-fixed-verdicts/New.lean
[Info ] Generated: /tmp/opencode/apeiron-fixed-verdicts/Get.lean
[Info ] Generated: /tmp/opencode/apeiron-fixed-verdicts/Insert.lean
[Info ] Generated: /tmp/opencode/apeiron-fixed-verdicts/Build.lean
```

Full local per-body logs: `/tmp/opencode/apeiron-fixed-verdicts/{new,get,insert,build}-{charon,aeneas,lean}.log`.
No remaining unsupported construct in these roots. Trials exposed nested const-repeat failure,
early return inside an outer optional-shard match, then loop-context joining failure. The final
source uses a single const-repeat per construction, an isolated array search loop, and no local
`table` identifier shadowing the generated namespace. `proof/regen.sh` strips backend-emitted
trailing blanks identically during regeneration and freshness checks; it does not rewrite Lean.

Control: `python3 /tmp/opencode/control-fixed-table.py` compiles throwaway `src/table.rs` copies
with `rustc --edition=2021 --test`, then runs them. Baseline 0, replace both fresh-write merges
with `let merged = value;` 101, restored 0. Verbatim failing assertion: `left: 3`, `right: 13` in
`full_shard_refuses_unchanged_but_existing_key_still_updates`. This is a Rust test control, not a
refinement control. The temporary runner is evidence only; the durable control task remains below.

#### Source/tool identity

Corpus read/resolved: `50c94192399add3b9f8e64bd14e36d8cba270705`;
Mathlib manifest: `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`; Lean `4.31.0`.
Rust tests use `rustc 1.97.0-nightly (17584a181 2026-04-13)`; Charon extraction uses
`nightly-2026-08-18`. Charon reports `0.1.254 (b104e24fea7d721b71e6c39fd70f26ff20bc0980)`;
Aeneas reports `aeneas 505b6ca-dirty`. Existing const-repeat patches, not an unmodified release:

| Artifact | SHA-256 |
|---|---|
| `src/table.rs` | `f30e31e46c18680283d0333ff04a679d519e904cf0958e86763372a639b074e3` |
| `src/merge.rs` | `7d4ce7656b38da200ba9c5265eca972e7cc475f133ae03efb6a1fa24c728c498` |
| `tests/table.rs` | `177ff01a04bc731f2ff7ffe6a6e1845e156bfdd3a29de95dcbfd3ef5d7f1a207` |
| `proof/extract/lib.rs` | `a2ecf3b1a01ca4b77fc575ff21254c0f1f3fe1e7fdc83532b110c996ab051f4e` |
| `proof/regen.sh` | `1625b0147f458f709fd46f7504a0e41258d0660b79fd60fee00f7086e783dc2d` |
| `proof/ApeironWord.lean` | `2237cf4040ce7ecafa5f528afab1e9539cc0b922d8043f31adfc2e56a8fb43dd` |
| `target/aeneas/charon` | `747e165bc62bb963f8d741e8c193c721a4b94555bcb1f9f86c79090fd7a59d0e` |
| `target/aeneas/charon-driver` | `207b223af35cf9a05227b329819b73532e94f655ccdefd64e77b755a15fdf297` |
| `target/aeneas/aeneas` | `33d0ce1a1cc2f1d3935c0cd2ab34f283797da61c531ed6d4e09d5c97b0d0228a` |

Source `table.rs` is 246 lines; generated `ApeironWord.lean` 991. This is not a complete trusted
perimeter inventory. No performance measurement or new proved table theorem is claimed.

#### Remaining serial waves

IDs `APEIRON-TABLE-W<number>-T1`. Lead records integrated base and creates
`worktree-agent-<id>` before dispatch. **R** = `target/`; **E** = R + generated extraction +
`proof/.lake/`; **B** = `proof/corpus/.lake/` + `proof/.lake/` + backend cache. Exclusive lanes;
controls use throwaway files and separate writable outputs. All proofs quantify over `N`, with
explicit pure/total, equality-respecting home, faithful equality/clone and merge premises.

| Wave | Owner; lane | Owned files (≤3) / result | One gate | Rejecting control |
|---|---|---|---|---|
| 9 | Sol; B | `proof/corpus/TableRefinement.lean`, `proof/corpus/AxiomAudit.lean`: replace obsolete error witnesses, define denotation/Rep, prove real constructor initialization and joint default/zero-capacity refusal witnesses | `lake build` in `proof/corpus` | Mutate generated constructor default; unchanged constructor/default witness must fail. No invented reservation axiom or deleted canonical contract. |
| 10 | Sol; B | `proof/corpus/TableReadRefinement.lean`, `proof/corpus/TableRefinement.lean`, `proof/corpus/AxiomAudit.lean`: terminating `find_slots` invariant at measure `N-i`, routing, get = canonical get | `lake build` in `proof/corpus` | Generated search always misses; present-key read theorem fails. |
| 11 | Sol; B | `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/TableReadRefinement.lean`, `proof/corpus/AxiomAudit.lean`: exact admission/refusal, successful insertWith, Rep preservation; N=0, collision/full and full-existing witnesses | `lake build` in `proof/corpus` | Bypass `f new default`, or refuse an existing key in a full shard; unchanged theorem/witness fails. |
| 12 | Sol; B | `proof/corpus/TableBuildRefinement.lean`, `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/AxiomAudit.lean`: exact per-shard distinct-key admission, ordered build refinement and canonical order/gauge corollaries | `lake build` in `proof/corpus` | Reverse same-key stream; replace/append control fails. Gauge corollary retains commutativity/associativity. |
| 13 | Flash; E+B | `proof/storage-controls.py`, this plan: durable fail-closed baseline/red/restored runner for the preceding source/generated controls; runner present (W13-T2/T4, W14-T3), twelve encoded controls with name-based locators | `python3 proof/storage-controls.py --jobs 3` | Runner must reject a surviving mutation and missing tools. |
| 14 | Flash/Sol; B/E | split the generic admission module (W14-T1); discharge the abstract callback contract (W14-T2/T5); persist the toolchain patches (W14-T4); name-based control runner (W14-T3) | `lake build` in `proof/corpus`; `proof/install-patches.sh --check`; `python3 proof/storage-controls.py --jobs 3` | Each slice's throwaway mutation red at a named declaration; one-byte patch mutation red. |
| 15 | Flash; B/E | run the epic barrier (W15-T1); real `cfg(kani)` harnesses in `src/proofs.rs` (W15-T2) | §6 barrier set | `Cell::add` → `sub` fails `word_cell_add_matches_native_wrapping`. |

#### Generic-N refinement slices (waves 9-11 progress)

Waves 9-10 are integrated at `1403767e671c81737525702946a3eb1eae4f0b0b` (`TableRefinement.lean` carries
the constructor/read/refusal witnesses at that commit). Wave 11 is discharged serially in three
slices on top of it; the planned `TableWriteRefinement.lean`/`TableReadRefinement.lean` module split
was not used, and the obligations landed in `TableRefinement.lean`. Wave 12 (build) is not started.

Owner Sol; lane B; owned files `proof/corpus/TableRefinement.lean`,
`proof/corpus/AxiomAudit.lean`.

| Slice | Obligation | Headline results | Gate |
|---|---|---|---|
| `APEIRON-TABLE-W11-T1` | generic-`N` present-key write preservation | `insert_with_present_exact`, `stateRead_replacePresent`, `insert_with_present_preserves`, `insert_with_present_rep_preservation`, `insert_with_present_joint_witness` | `lake build`, cwd `proof/corpus`, exit 0, 3358 jobs |
| `APEIRON-TABLE-W11-T2` | generic-`N` executing read refinement | `find_slots_spec_generic`, `word_shard_index_spec`, `generatedRoute_exists`, `generatedLocator_routes`, `table_get_eq_stateRead_of_route`, `table_get_eq_stateRead`, `functionalRep_iff_readRep_of_routes`, `generatedFunctionalRep_iff_readRep`, `readRep_iff_Rep` | same |
| `APEIRON-TABLE-W11-T3` | generic-`N` fresh-key insertion and vacant-slot loop | `stateRead_insertVacant`, `stateRead_insertEmptyShard`, `insert_with_empty_shard_exact`, `insert_with_loop0_vacant_body_exact`, `insert_with_loop0_vacant_spec`, `insert_with_loop0_full_spec_generic`, `insert_with_vacant_exact`, `insert_with_vacant_rep_preservation`, `insert_with_vacant_readRep_preservation`, `insert_with_absent_joint_witness` | same |

`stateRead` is the total physical-state denotation at a locator; `FunctionalRep` is extensional
equality with it. A state-independent locator is unsound because the generated `find` searches
`t.shards`, so the read relation is state-indexed: `ReadRep t h` states the generated `get` result
directly, and `readRep_iff_Rep` shows it equals the existing extensional `Rep` at `N = 16` in both
directions. `generatedLocator t` is derived from the generated `shard_index`/`find` through
`generatedRoute_exists`, so `table_get_eq_stateRead` is total over every generated state and key.

Controls: mutate the generated present-key branch to store `value`; `insert_with_present_exact` fails
at `TableRefinement.lean:630`. Mutate generated `get`'s present branch to return `self.default`;
`table_get_eq_stateRead_of_route` fails at `TableRefinement.lean:963` with goal `t.default = value`.
Mutate the throwaway generated `insert_with_loop0.body` vacant branch to store `value` instead of
`wrapping_add value default`; `insert_with_loop0_vacant_body_exact` fails at
`TableRefinement.lean:1240:69`. Restore builds green (3358 jobs). Every new headline `#audit`s inside
`{propext, Classical.choice, Quot.sound}`; no forbidden token. Source, generated file and pins
unchanged: `src/table.rs` `83597ecf...`, `proof/ApeironWord.lean` `958b1ab5...`.
`TableRefinement.lean` `fcf322ccb1c743097105514bfef422c03ee9d58ee690512a9cf0d3760b1a9100` (587 lines),
`TableRefinementGeneric.lean` `5db832f56256a22efcd0455dcba73f6e6fdb03d4a953c37f9d4d19481f306720` (1319
lines), `TableRefinementBuild.lean` `7a8121d7ad7c1e64459f24e3dc09418837b8ccf6366a583bb21ba122daccbbbb`
(1377 lines), `TableRefinementAdmission.lean`
`1a6146c728c15660feab1594d56e07fa7b91636341d2c91b762874863422cd42` (1345 lines), `AxiomAudit.lean`
`a8e5d11a5fadef761611d222423a0f62abc234913b77a9180e8a5672040e96f9` (249 lines).

Module split (`APEIRON-TABLE-W11-T4`): the generic-`N` material moved from `TableRefinement.lean` to
`TableRefinementGeneric.lean` under the same namespace (audit names unchanged), with
`TableRefinementGeneric` added to `lakefile.lean` roots and imported by `AxiomAudit.lean`.
`lakefile.lean` `09c2fb38f143a2ee9eaee9485800bdd0e220cd6bc500512481b6b18d84f2a88b`. Gate `lake build`,
cwd `proof/corpus`, exit 0, 3359 jobs. Control: `#check Apeiron.TableRefinement.insert_with_present_exact`
fails `lean.unknownIdentifier` with only `import TableRefinement` (exit 1) and resolves with
`import TableRefinementGeneric` (exit 0).

Unconditional post-state routing (`APEIRON-TABLE-W11-T5`): `word_shard_index_state_independent`,
`find_slots_eq_of_matches`, `find_slots_insertVacant_ne`, `find_slots_insertVacant_self`,
`table_find_insertVacant_ne`, `table_find_replacePresent`, `routes_replacePresent` and
`routes_insertVacant` show the exact generated updates preserve generated routing;
`insert_with_present_readRep_preservation` and `insert_with_vacant_readRep_preservation_routed` then
derive `ReadRep` from `Routes t locate` alone, with no post-state routing premise. Gate `lake build`,
cwd `proof/corpus`, exit 0, 3359 jobs; cone `{propext, Classical.choice, Quot.sound}`. Control:
generated `insert_with_loop0.body` vacant branch `if b` → `if false`; red at
`TableRefinementGeneric.lean:1062:17`; restore green. No forbidden token.

Ordered build refinement, first slice (`APEIRON-TABLE-W12-T1`): `proof/corpus/TableRefinementBuild.lean`
defines `sliceEntries`, `orderedFold` (`orderedFold_eq_build` shows it is `Apeiron.Table.build`
definitionally) and the per-step physical predicate `BuildInsertPremises`;
`build_insert_premises_success` discharges one real `insert_with` and preserves `ReadRep`;
`table_build_loop_result_readRep` and `table_build_loop_readRep` give the `build_loop`
`loop.spec_decr_nat` result (Ok returns the ordered fold, Err is `ShardFull`); `word_table_build_readRep`
composes with the real empty constructor. `TableRefinementBuild` added to `lakefile.lean` roots and
imported by `AxiomAudit.lean`. Gate `lake build`, cwd `proof/corpus`, exit 0, 3360 jobs; cone
`{propext, Classical.choice, Quot.sound}`. Control: generated `build_loop.body` guard `i < i1` →
`i.val + 1 < i1.val` skips the last element; red at `TableRefinementBuild.lean:163`; restore green.

Physical invariant and unconditional build result (`APEIRON-TABLE-W12-T2`): `TableInv` states that
every populated shard has an occupied prefix and vacant suffix (`OccupiedPrefix`);
`emptyWordTable_tableInv` and `tableInv_insert_with` establish initialization and preservation by the
real generated `insert_with` in every branch (present, vacant, untouched, and full-shard refusal via
`insert_with_full_exact`). `buildInsertResult_of_readRep` and `allBuildInsertResults` expose the
Err-allowing per-step result, and `word_table_build_result_readRep` is unconditional:
`word_table_build default kvs = ok (Err ShardFull) ∨ ∃ final, = ok (Ok final) ∧ ReadRep final
(Apeiron.Table.build core.num.I64.wrapping_add default (sliceEntries kvs))`. Gate `lake build`, cwd
`proof/corpus`, exit 0, 3360 jobs; cone `{propext, Classical.choice, Quot.sound}`. Control: generated
`insert_with_loop0.body` writes at `i + 1` instead of the first vacant slot; `tableInv_insert_with`
fails to rebuild; restore green.

Per-shard capacity admission (`APEIRON-TABLE-W12-T3`): `PrefixShardCapacity` is an explicit
physical-state capacity contract (every occupied-prefix witness of a populated shard is shorter than
16); `allBuildInsertsAdmitted_of_prefixShardCapacity` derives `AllBuildInsertsAdmitted`, and
`word_table_build_readRep_of_prefixShardCapacity` gives the Ok-only canonical-fold refinement. Gate
`lake build`, cwd `proof/corpus`, exit 0, 3360 jobs; cone `{propext, Classical.choice, Quot.sound}`.
Control: generated present-key-as-absent mutation turns the build red (the first failure is the
pre-existing present-key theorem at `TableRefinement.lean:220`, so localization to a new headline was
not achieved); restore green.

Physical no-duplicate invariant (`APEIRON-TABLE-W12-T4`): `ShardKeysDistinct`/`TableInv2` require
pairwise-distinct keys in each populated shard's occupied prefix; `emptyWordTable_tableInv2` and
`tableInv2_insert_with` establish initialization and preservation across every generated branch;
`occupiedPrefix_distinct_key_count` counts them. `DistinctKeysPerShard` is the stream hypothesis;
`prefixShardCapacity_of_distinctKeysPerShard` and `word_table_build_readRep_of_distinctKeysPerShard`
close the capacity bridge conditional on `PrefixShardKeyCoverage`. Gate `lake build`, cwd
`proof/corpus`, exit 0, 3360 jobs; cone `{propext, Classical.choice, Quot.sound}`. Control: generated
present branch writes at the first vacant slot instead of the matching slot (duplicate key);
`tableInv2_insert_with` fails at `TableRefinementBuild.lean:1013`; restore green.

Stream-capacity admission (`APEIRON-TABLE-W12-T5`): `proof/corpus/TableRefinementAdmission.lean`
proves the Ok-only canonical-fold build from the stream hypothesis `DistinctKeysPerShard` alone.
`ReachablePrefixCoverage` is indexed by the carried state and prefix (not quantified over arbitrary
`ReadRep` representatives); `emptyWordTable_reachablePrefixCoverage` and `reachablePrefix_insert`
carry it through the real generated `insert_with`; `reachablePrefixCoverage_capacity` bounds each
shard load by the distinct routed stream keys; and
`word_table_build_readRep_of_distinctKeysPerShard_alone` composes with the real constructor. This
supersedes the `PrefixShardCapacity`/`PrefixShardKeyCoverage` chain (those declarations remain). Gate
`lake build`, cwd `proof/corpus`, exit 0, 3361 jobs; cone `{propext, Classical.choice, Quot.sound}`.
Control: generated present-as-vacant duplicate mutation turns the build red (first failure
`tableInv2_insert_with` at `TableRefinementBuild.lean:1013`, upstream of the new module); restore
green.

Exact admission characterization (`APEIRON-TABLE-W12-T6`/`T7`): `ReachablePrefixExact` strengthens the
loop coverage to a bidirectional locator↔routed-keys equivalence (`emptyWordTable_reachablePrefixExact`,
`reachablePrefixExact_insert`). `BuildAdmissible` is the tight per-step stream condition (the next key
is already present in its routed shard, or that shard holds fewer than 16 distinct prefix keys).
`word_table_build_readRep_of_buildAdmissible` gives the Ok branch; `buildAdmissible_of_distinctKeysPerShard`
shows the earlier strict condition is sufficient but stronger than necessary;
`word_table_build_success_implies_buildAdmissible`, `word_table_build_refuses_of_not_buildAdmissible`
and `word_table_build_refuses_iff_not_buildAdmissible` close
`word_table_build default kvs = ok (Err ShardFull) ↔ ¬ BuildAdmissible default kvs`. Gate `lake build`,
cwd `proof/corpus`, exit 0, 3361 jobs; cone `{propext, Classical.choice, Quot.sound}`. Control:
generated full-shard Err branch mutated to `Ok` with unchanged state; red at
`TableRefinementAdmission.lean:1018`; restore green.

Generic-`N` build and durable controls (`APEIRON-TABLE-W13-T1`/`T2`). W13-T1 adds
`proof/corpus/TableRefinementBuildGeneric.lean` (1090 lines): the ordered build refinement at general
`N` with the concrete word callbacks — `build_insert_premises_successN`, `buildInsertResult_of_readRepN`,
`table_build_loop_result_readRepN` (unconditional `Err ShardFull ∨ ∃ final, Ok final ∧ ReadRep …`),
`table_build_result_readRepN`, `table_build_readRepN`, `table_build_readRepN_of_distinctKeysPerShardN` —
plus an abstract-callback loop layer whose one-step contract `BuildInsertResultG`/`BuildInsertOkG` and
clone identities are explicit hypotheses (`table_build_loop_result_readRepG`, `table_build_loop_readRepG`).
Gate `lake build`, cwd `proof/corpus`, exit 0, 3362 jobs; cone `{propext, Classical.choice, Quot.sound}`.
Control: `BuildInsertResultN`'s `Ok` branch mutated to `ReadRep result.2 h`; red at
`TableRefinementBuildGeneric.lean:497:12` (`table_build_loop_result_readRepN`); restore green. W13-T2
adds `proof/storage-controls.py` (sha256 `16d6d4289b32ea8c872f3e81e6caf5f51c6f14c21c8547962c2907b0ba92b854`):
a stdlib-only runner with `--list`, `--only`, `--jobs`, `--self-test` and 7 encoded generated-body
controls (`w11-t1`, `w11-t2`, `w11-t3`, `w11-t5`, `w12-t1`, `w12-t6`, `s6-fresh-write`); each copies
`proof/corpus`, applies one exact-string mutation, asserts the first `lake build` error locator,
restores and re-asserts green. Gate: `python3 -m py_compile` exit 0, `--list` exit 0 (7 controls),
`--self-test` exit 0 (10/10), `--only w11-t5` red at `TableRefinementGeneric.lean:1062:17` then green,
`--jobs 3` 7/7. The repository tree and owned source digests are unchanged by the runner.

Exact admission at general `N` and the durable runner (`APEIRON-TABLE-W13-T3`/`T4`). W13-T3 adds
`proof/corpus/TableRefinementAdmissionGeneric.lean` (1500 lines, the file-size limit):
`ReachablePrefixCoverageN`, `ReachablePrefixExactN`, `BuildAdmissibleN`,
`emptyWordTableN_reachablePrefixExactN`, `reachablePrefixExact_insertN`,
`reachablePrefixExact_admissible_of_insert_successN`, `buildAdmissibleN_of_distinctKeysPerShardN`,
`table_build_readRepN_of_buildAdmissibleN`, `table_build_success_implies_buildAdmissibleN`,
`table_build_refusesN_of_not_buildAdmissibleN`, and `table_build_refusesN_iff_not_buildAdmissibleN`
(`wordBuildN default kvs = ok (Err ShardFull) ↔ ¬ BuildAdmissibleN default kvs`, `0 < N`). Gate
`lake build`, cwd `proof/corpus`, exit 0, 3363 jobs; cone `{propext, Classical.choice, Quot.sound}`.
Control: generated full-shard `Err` → `Ok ()`; the whole-package build fails first at
`TableRefinement.lean:517` (`insert_full_loop_spec`), so the new headline is falsified transitively,
not with a new-file red line. W13-T4 hardens `proof/storage-controls.py` (917 lines, sha256
`ffde2f60e96de3b432cece34f9b1c4eb906b78dbed6181d153aba6ae8c626213`): nine encoded generated-body
controls (`w11-t1`, `w11-t2`, `w11-t3`, `w11-t5`, `w12-t1`, `w12-t2`, `w12-t3`, `w12-t6`,
`s6-fresh-write`), each a whole-package `lake build` in a throwaway copy with a current recorded
locator (no `(stale)`), plus `--list`/`--only`/`--jobs`/`--self-test`; self-test 15/15 including
missing-tool rejection (exit 3). Gate: `py_compile` exit 0, `--list` exit 0, `--self-test` exit 0,
`--jobs 3` 9/9. Repository tree and source digests unchanged by the runner.

Contract characterization, module split, runner hardening, toolchain patches (`APEIRON-TABLE-W14-T1`…`T5`).
W14-T1 split the 1500-line `TableRefinementAdmissionGeneric.lean` into
`TableRefinementReachablePrefixExactN.lean` (1271 lines) and `TableRefinementAdmissionGeneric.lean`
(247 lines), byte-for-byte reconstruction verified; gate exit 0, 3364 jobs. W14-T2/W14-T5 discharged
the abstract callback contract: `cloneI64_identity`, `buildInsertResultG_word_iff`,
`buildInsertResultG_word`, `buildInsertOkG_word`, `table_build_loop_result_readRepG_word`,
`table_build_loop_readRepG_word`, `WordBuildOneStepContractN`, `wordBuildOneStepContractN_iff_concrete`,
`wordBuildOneStepContractN_sufficient`, plus the Aeneas `wpSpec_bind_iff` and the necessity
characterization `BuildLoopSpecG`, `BuildInsertSpecG`, `AllBuildInsertSpecsG`,
`buildLoopSpecG_iff_allBuildInsertSpecsG`, `buildLoopSpecG_true_iff_buildInsertResultG`,
`buildLoopSpecG_false_iff_buildInsertOkG`; `TableRefinementBuildGeneric.lean` 1468 lines. W14-T4
persisted the toolchain patches: `proof/patches/{charon-repeat-optional-trait.patch,
aeneas-array-repeat.patch,PINS}`, `proof/install-patches.sh`, a `proof/README.md` pin section; gate
`proof/install-patches.sh --check --aeneas-src /tmp/opencode/aeneas-src` exit 0, one-byte patch
mutation red. W14-T3 hardened `proof/storage-controls.py` (1149 lines, sha256
`06079130309505b8df59ad6e732e8e55969592fbb8c5e2480251e50588701914`): name-based locators, twelve
encoded controls, `w12-t5` a decided rejection, and two in-file red lines (`w14-admission-generic` in
`TableRefinementAdmissionGeneric.lean`, `w14-reachable-exact` in `TableRefinementReachablePrefixExactN.lean`);
`--jobs 3` 12/12, `--self-test` 34/34.

Epic barrier (`APEIRON-TABLE-W15-T1`/`T2`). All ten §6 commands exit 0: `git diff --check`,
`cargo test --workspace --all-features --locked` (96 passed, 0 failed), `cargo clippy … -D warnings`,
`cargo fmt --all -- --check`, `cargo kani`, `proof/regen.sh --check` (1708 jobs), `lake build` in
`proof/` (1708) and `proof/corpus/` (3364), `proof/install-patches.sh --check` (patches apply to the
pinned bases), `python3 proof/storage-controls.py --jobs 3` (12/12). The barrier mutated nothing:
`git status --porcelain` and `sha256(proof/ApeironWord.lean)` (`958b1ab5…`) identical before and after.
W15-T2 replaced the vacuous `cargo kani` gate: `src/proofs.rs` (958 lines, sha256
`3be901668aee40abd04a1a1f5d99b140663f02a76d659aa7ea3a672692b464c5`) now carries 57 bounded
`#[kani::proof]` harnesses (`Complete - 57 successfully verified harnesses, 0 failures, 57 total`);
the control mutating `Cell::add` to `sub` fails `word_cell_add_matches_native_wrapping`; `cargo test`,
`clippy`, `fmt` stay green.

Residual (S4 open): the `Table` Kani behavioral harnesses use concrete edge keys because a symbolic key
builds a 256-way mux that CBMC does not decide in reasonable time; `i64` multiplication
associativity/distributivity is verified at the `i8` width `src/scalar.rs` prescribes, not at full
`i64`. Both are bounded-model-checking limits, recorded, not correctness gaps. No rustc/LLVM codegen,
ABI or GPU claim is established.

**First gate (historical):** `lake build`, cwd `proof/corpus`; observed exit 1:

```text
error: TableRefinement.lean:56:28: Unknown constant `apeiron_word.table.StorageError.CapacityOverflow`
```

The next task is Sol's constructor/resource-premise proof, not another extraction redesign.
Required Sol verdict: are the fixed-resource domain and callback contracts sufficient, and do
actual constructor/refusal witnesses inhabit them jointly? Astra's decision does not supply it.
Terra has no open implementation assignment: all four real bodies extract. If a proof reveals an
executing-source defect, stop and assign Terra a separate ≤3-file source/caller repair, gate
`cargo test --workspace --all-features --locked`, before re-extraction in its own E lane. Do not
silently widen a proof task into source/tool redesign.

The lead's full Rust, both Lake builds, controls and trusted-perimeter gates remain required
for epic closure. The present authorization accepts a building representation and extraction
verdict, not closure of the red bridge. Other governing-doc corrections are a separately scoped
owner task; old growable and deleted-container claims below are historical only.

### Historical binding boundary replacement (superseded by fixed shards)

Owner Astra; base `da01485d24be8900824750431a70f19100bee200`, initially clean. The lead authorizes
source deletion and a red boundary commit in the supplied checkout. No branch/worktree is created;
the existing `one-allocation` branch is unrelated and is not modified. The two-container decision
below is superseded, not a prerequisite to this sequence. No new markdown or other governing-doc
edits are authorized. `docs/kernel-boundary.md` contains the replacement contract and trust cut.

**Intent:** the executing table is the thesis's proof subject. A proved reference beside an unproved
container does not satisfy it. Delete both old containers and their wrappers rather than preserving
that boundary under aliases. Compatibility breakage is accepted; no performance claim transfers.
**Spec:** one `Table<K,V>`, default-valued total lookup, new-first `insertWith`, finite-stream `build`,
and home-sharded finite overrides, as specified in the boundary document. No fixed-capacity-four
restriction, automatic `Option` lifting or commutativity assumption for ordinary builds survives.
The old intent/spec documents below describe retired source, not the replacement's contract.

The boundary change is three serial owned slices: `APEIRON-TABLE-W0-T1` owns these two documents;
`APEIRON-TABLE-W0-T2` deletes `src/map.rs`, `src/safe_table.rs`, `src/set.rs`;
`APEIRON-TABLE-W0-T3` owns `src/table.rs`, `src/lib.rs`, `src/merge.rs`. Astra owns all three for this
explicit lead-assigned deletion. Only the last slice takes lane R. They share one authorized
boundary commit, not a claim of implementation completion.

```text
SCOPE:      the three slices above; remove competing containers, expose only Table, freeze its API
REGISTER:   planning/API removal; excludes completed implementation, refinement and performance
CORPUS:     Apeiron.Table "get_build_acc", "build_eq_of_key_streams_eq",
            "build_gauge_invariance"; Adjacency and Home, native grade proved
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean
GATE:       root: cargo check --lib --locked baseline 0; cargo test --doc --locked after deletion
            101: E0599, Table::new missing at src/merge.rs:47; git diff --check 0
GRADE:      anchor; declaration only, no methods or executing container; no new proof/measurement
```

The merge doctest is rewired to the required API and deliberately fails without its implementation.
The library alone compiles, with unread-field warnings. Do not suppress those warnings or add a
compatibility implementation to manufacture green. The old integration tests, benchmarks, examples,
fuzz targets and storage Kani harnesses are retained as red migration inputs, not silently disabled.
`proof/extract/lib.rs` still names deleted source; generated Lean and its old bridge are stale.
A successful build of those cached definitions alone cannot establish freshness or new-table proof.

Observed boundary checks (root):

| Command | Outcome |
|---|---|
| `cargo test --doc --locked` | 101; one passes, the Table example fails with E0599 (`Table::new` missing). |
| `cargo test --workspace --all-features --locked` | 101; E0432/E0433 on removed `map`, `safe_table`, `set`, `CountTable` and scanner imports. |
| `proof/regen.sh --check` | 1; source digest stage rejects deleted `src/safe_table.rs`, before extraction. Charon `0.1.254` (`b104e24f`), Aeneas `505b6ca-dirty`, Rust extraction toolchain `nightly-2026-08-18`; script reports Lean pin `4.31.0`. |
| `git diff --check` | 0. Formatting evidence only; no storage proof or mutation-control closure. |
| `cargo fmt --all -- --check` | 0 after correcting the changed re-export formatting. |

#### Identity

Corpus read: clean `apeiron-rebuild` at `50c94192399add3b9f8e64bd14e36d8cba270705`.
The bridge currently resolves `d9aefc93522e2109bc96158526de6dbd52477276`.

| Source | SHA-256 at boundary edit |
|---|---|
| `src/lib.rs` | `56c0ce5d9fcf14a7075e286cc7710461dc2b986aaf3cecea91844b6b5c15c7a7` |
| `src/table.rs` | `18c9ce400f8fb8f872e6c9cfe370d4f0415081589833d58f0668240c3feea72f` |
| `src/merge.rs` | `f1a056ad3d19c460ff49ec565ad1e22d06ac77ac08da3d24efdff48dc63d1874` |
| corpus `Table/Basic.lean` | `e7558bac1d255643f58100f5a1adf32926a423fe6b49e7c991f8d3a3b0eb26b9` |
| corpus `Table/Adjacency.lean` | `5593d39db83213694ef51b9cb4d73686b9e2b5e8b548a2840e61b5a94d382732` |
| corpus `Table/Home.lean` | `bab40ae04d1bd19cde8c97e09eb2453652cbddf253969ebad03ac582cf5d599b` |

No Lean/toolchain acceptance is inferred from those digests. Each execution task records its actual
Rust, Charon/Aeneas binaries (including patches), Lean, Mathlib and corpus identities. Tool pins
and the historical constructor patch are in `proof/README.md`; do not assume the patch solves
growable vectors, fallible reservation, function pointers or generic callback translation.

#### Extraction feasibility stop

`APEIRON-TABLE-W4-T1`, Sol, base `dde9d58`, lanes E+B. Charon emits the real bodies for all four
monomorphic probes. Aeneas rejects each LLBC before translating a method because
`Table.home : fn(&K) -> u64` is an arrow-valued struct field. Verbatim:
`Arrow types are not supported yet`; `Could not translate type decl
'apeiron_word::table::Table because of previous error`. The backend therefore does not reach a
separate verdict on `Vec`, allocation, 256-shard initialization, `usize`, default values, iterator
loops or callback invocation. `StorageError` and both constructors extract. Wave 4 stops here;
waves 5-7 remain blocked on a generated `Table` representation.

```text
SCOPE:      proof extraction roots/generated file, TableRefinement/audits/pin, storage status docs
REGISTER:   extraction/refinement; excludes compiler, allocator and ungenerated Table transitions
CORPUS:     Apeiron.Table Basic/Adjacency/Home at 50c94192399add3b9f8e64bd14e36d8cba270705
REFERENCE:  /home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean
GATE:       proof/regen.sh --check; lake build in proof/corpus; initial stale source digest exit 1
GRADE:      proved canonical contracts/error constructors; Rust Table implementation link open
```

#### Serial waves and lanes

IDs are `APEIRON-TABLE-W<n>-T1`; one owner per file. Before dispatch the lead records the integrated
base and creates `worktree-agent-<id>`. Each row is a serial wave; a rename is one owned artifact.
**R** = `target/`; **E** = R + `proof/ApeironWord.lean` + `proof/.lake/`;
**B** = `proof/corpus/.lake/` + `proof/.lake/` + Aeneas backend cache. Controls use throwaway copies
and separate writable build outputs. No simultaneous lane occupants. Future filenames denote
required artifacts, not already existing tests or proofs.

| Wave | Owned artifacts / obligation | Owner; lane | One focused gate | One rejecting control |
|---|---|---|---|---|
| 1 | `src/table.rs`, `src/merge.rs`: implement the declared constructor/get/insert_with/build, initialized growable shards and fallible resources; expand executable doctests for nonzero default, collisions and directional merges | Terra; R | `cargo test --doc --locked` (root) | First insertion bypasses `f v d` in a throwaway real body; nonidentity-default fixture must fail. |
| 2 | `tests/safe_table.rs` renamed to `tests/table.rs`: replace the two-container trace with single-table default/write/build tests, growth, exact error witnesses and replace/append order fixtures | Flash; R | `cargo test --locked --test table` (root) | Compare only home addresses instead of full keys in a throwaway real body; collision fixture must fail. |
| 3 | `proof/corpus/lakefile.lean`, `proof/corpus/lake-manifest.json`: pin the read gen2 revision, check Basic/Adjacency/Home on the bridge toolchain | Sol; B | `lake build Apeiron.Table.Basic Apeiron.Table.Adjacency Apeiron.Table.Home` (cwd `proof/corpus`; preparation: `lake update apeiron`) | Nonexistent corpus revision in a throwaway package must fail resolution, never fall back to sibling source. |
| 4 | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: replace `word_safe_table4_*` roots with actual Table constructor/get/insert_with/build roots, no capacity bound; include executing routing/growth and concrete merge instances | Sol; E | `proof/regen.sh --check` (root; first regenerate with `proof/regen.sh`) | Mutate the real default-read branch; unchanged generated output must fail freshness. |
| 5 | `proof/corpus/SafeTableRefinement.lean` renamed/reshaped to `TableRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: remove retired roots/audits; prove constructor, joint representation witnesses, terminating home-shard lookup and total default read | Sol; B | `lake build` (cwd `proof/corpus`) | Generated lookup returns the default for a present key; unchanged lookup theorem must fail. |
| 6 | `proof/corpus/TableRefinement.lean`, `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/AxiomAudit.lean`: writes/growth refine canonical insertWith, preserve Rep, characterize success/resource refusal with witnesses; audit imports the new module | Sol; B | `lake build` (cwd `proof/corpus`) | Generated first-write branch stores `v` instead of `f v d`; unchanged insertWith theorem must fail. |
| 7 | `proof/corpus/TableBuildRefinement.lean`, `proof/corpus/TableWriteRefinement.lean`, `proof/corpus/AxiomAudit.lean`: actual build = canonical build; compose get_build_acc, per-key stream equality and conditional gauge invariance; exact Home attribution and order counterexamples | Sol; B | `lake build` (cwd `proof/corpus`) | Reverse same-key contributions in the generated build; unchanged replace/append build theorem or witness must fail. |

All gates are required with baseline/red/restored controls; the proof tasks target `proved`, tests
only `tested`. A type or empty module that elaborates is not a completed row. No `sorry`, admission,
opaque reservation axiom, assumed desired simulation, or reduced-capacity proof can close a row.
If proof size would exceed 1,500 lines, split the obligation into another serialized ≤3-file task
before editing beyond the declared paths. Sol must approve the resource and callback premises;
this architecture decision is not that proof-sufficiency verdict.

**First gate:** `cargo test --doc --locked`, cwd repository root. It currently exits 101 on the
missing `Table::new`. Implement the same path; do not ignore the doctest. The extraction row is an
explicit feasibility stop: after three materially different unsupported shapes, return the exact
unsupported primitive and next tool-model task. Never restore a separate reference to evade it.

#### Caller migration and integration barrier

The seven rows do not repair the whole compatibility surface. After the API lands, Flash migrates
retained callers in serialized ≤3-file batches, without introducing old-name aliases:

- `tests/a_map_is_a_map.rs`, `tests/a_set_is_a_set.rs`, `tests/the_counts_are_exact.rs`: one initial
  batch, gate `cargo test --locked --test a_map_is_a_map --test a_set_is_a_set --test the_counts_are_exact`;
  control: change a new-first merge to old-first and require the unchanged directional fixture red.
  Preserve corresponding semantic/boundary cases; entry/iterator-only contracts are explicitly
  retired, not claimed for the new minimal API.
- `src/proofs.rs`: separate Kani batch, gate `cargo kani`; replace retired storage/scan claims with
  bounds/routing/collision harnesses for Table; a routing-mask mutation must fail. Preserve unrelated
  algebra harnesses. Bounded Kani is supplementary, not the storage refinement theorem.
- Remaining test/example/bench/fuzz callers need an exact path manifest and ≤3-file, single-command
  batches selected by their owners; no unbounded "repair all callers" dispatch. That manifest is
  **MISSING** and blocks whole-workspace closure, not the seven focused gates. Benchmarks must be
  remeasured on the new executing path before old numbers are reused.

Integration order: declaration/deletion → implementation and tests → corpus pin → actual extraction
→ constructor/read → write/resource → build/layout → caller migrations → full assurance barrier.
The lead, not an individual task, runs all current AGENTS §6 Rust commands, regeneration
freshness, both Lake builds and the controls on one source/tool identity. Existing S7/S8 requirements
remain: claim/witness/audit coverage, trusted checker ≤5,000 lines, owned source ≤1,500 lines and
≤12 immediate source files. Missing assurance gates remain missing; deletion alone does not close
them. Corrections to out-of-scope README, direction, handoff and old lifecycle documents need a
separate lead-owned documentation batch; their retired-container claims are not current acceptance.

**Residual:** no executing Table, no new extraction/refinement, no runtime measurement. SIMD, raw
slot lifecycle and workers are removed rather than left in a tested-only path. Source-level proof
still trusts translation/model fidelity, compiler/runtime/allocator realization and machine execution
as explicitly bounded in `docs/kernel-boundary.md`; it may not assume the table semantics themselves.

### Historical bounded-reference plan and evidence (superseded)

- **Owner:** OpenCode
- **Intent:** `docs/epics/storage-refinement.md`
- **Spec:** `docs/epics/storage-refinement.md`
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean before this epic
- **Status:** approved

### Goal and completion condition

Establish one safe finite storage semantics, prove its successful transitions against the canonical
Table definitions, then relate `Map` without promoting sampled agreement to proof.

```sh
sh -c 'cargo test --workspace --all-features --locked && proof/regen.sh --check && cd proof/corpus && lake build'
```

### Non-goals

- Whole-container extraction in one task.
- Proof by differential testing.
- Performance changes before semantic equivalence.

### Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | Fixed-capacity container and differential trace | `src/safe_table.rs`, `src/lib.rs`, `tests/safe_table.rs` | OpenCode | `cargo test --locked --test safe_table` | Full-check or removal mutation makes the trace fail. |
| W2 | Monomorphic extraction | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | OpenCode | `proof/regen.sh --check` | Generated transition mutation breaks freshness or proof. |
| W2 | Extraction evidence | `proof/README.md` | OpenCode | `git diff --check` | Not applicable: extraction freshness owns the executable control. |
| W3 | Canonical Table refinement | `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean` | OpenCode | `lake build` in `proof/corpus` | Transition mutation breaks unchanged theorem. |
| W4 | Optimized relation decision | `docs/kernel-boundary.md`, `docs/epics/storage-refinement.md` | Astra | `git diff --check` | Requirement review; no executable equivalence claim. |

### Task blocks

#### W1-T1 - Safe finite reference

- **Scope:** Implement fixed-capacity lookup/replace/remove and compare against `Map` on
  the same bounded traces.
- **Files:** `src/safe_table.rs`, `src/lib.rs`, `tests/safe_table.rs`
- **Depends on:** None
- **Grade:** tested
- **Gate:** `cargo test --locked --test safe_table`
- **Negative control:** In a throwaway copy, treat a full-table replacement as refusal or leave a
  removed key readable; the unchanged test must fail and the original must pass.
- **Evidence to record:** Test population, full/refusal witnesses and source digest.

#### W2-T1 - Extract the safe transition body

- **Scope:** Add concrete word/capacity wrappers that call the real safe implementation and regenerate
  the accepted Aeneas output.
- **Files:** `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`
- **Depends on:** W1-T1
- **Grade:** tested
- **Gate:** `proof/regen.sh --check`
- **Negative control:** A generated body mutation must make freshness or W3 fail.
- **Evidence to record:** Source/tool digests, selected wrappers and extraction result.

#### W2-T2 - Extraction evidence

- **Scope:** Record the selected safe-table bodies, source digest, result and exclusions.
- **Files:** `proof/README.md`
- **Depends on:** W2-T1
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. W2-T1 owns extraction freshness.
- **Evidence to record:** Generated names, source digest and unsupported operations.

#### W3-T1 - Canonical refinement

- **Scope:** Prove successful word-table transitions denote `Apeiron.Table` lookup and replacement,
  with explicit full and missing-key witnesses.
- **Files:** `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean`
- **Depends on:** W2-T1
- **Grade:** proved
- **Gate:** `lake build` in `proof/corpus`
- **Negative control:** Mutate replacement or removal in a throwaway generated file; unchanged
  refinement must fail and the original must pass.
- **Evidence to record:** Theorem names and complete axiom cones.

#### W4-T1 - Independent relation decision

- **Scope:** Freeze the independent relation and trust cuts; neither decision nor reference proof authorizes removing `Map`.
- **Files:** `docs/kernel-boundary.md`, `docs/epics/storage-refinement.md`
- **Depends on:** W3-T1
- **Grade:** anchor (architecture), not implementation equivalence
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. No universal optimized-map claim closes in this task.
- **Evidence to record:** Selected representation, excluded layers and next red command.

### Barrier checklist

- [ ] Container tests and controls pass.
- [ ] Extraction is fresh and free of `axiom`, `sorry` and `admit`.
- [ ] Canonical refinement and fail-closed audits pass.
- [ ] Independent optimized relation decision and complete controls satisfy S6; no `Map` retirement authorized.
- [ ] Universal optimized representation/transition relation is checked before any optimized proof claim; remains open independently of reference closure.
- [ ] S7 integration and S8 ledger/perimeter enforcement pass; see `docs/plan.md`.
- [ ] Rust tests, clippy and formatting pass.
- [ ] Source and tool digests and trust boundaries are recorded.

### Stop conditions

- Stop extraction after three unsupported Aeneas representations and record the smallest missing
  safe construct; do not rewrite formulas in wrappers.
- Stop optimized-map promotion if the relation remains sampled; retain tested status and name the
  exact universal obligation.

### Historical W3 blocker record

`W1` and `W2` are closed: the safe fixed-capacity reference passes its focused tests and the
10,000-step differential comparison, and `word_safe_table4_get/insert/remove` are extracted from the
real `src/safe_table.rs` with a fresh, axiom-free generated file. `W3` is blocked before the first
refinement theorem.

Three materially distinct attempts were made at the refinement:

| Attempt | Change | Result |
|---|---|---|
| 1 | Iterator/`for`-loop reference | Charon refused outer-loop breaks and iterator closures. |
| 2 | Bounded recursive `find_from`/`empty_from` reference | Extracts, but the generated body is a `partial_fixpoint`; direct `simp`/`rw` cannot unfold the recursion and the scalar `Result` binds are not `[simp]`. |
| 3 | Aeneas `fun_induction` / `step` on the generated body | `fun_induction` reports no functional induction theorem; `step` accepts only `WP` specs, not pure `Eq` goals. |

The precise unresolved goal: a total Lean function for the safe search, or a canonical refinement
stated as a `WP.dspec` Hoare triple. The extracted search is a `partial_fixpoint`; the only supplied
reasoning principle is the auto-generated `.fixpoint_induct`, which targets `WP.dspec` goals and
requires admissibility over a `Result` CCPO. A pure structural characterization additionally needs a
denotation from the Aeneas array state to `Apeiron.Table.HMap`.

Next bounded task, unblocked now: give the Rust search a form Aeneas translates to a structurally or
well-founded recursive function (for example recursion on a slice/list tail rather than a numeric
index, with the index carried as an accumulator), so the extracted body has equation lemmas and
`fun_induction` support. If that fails, state the refinement through `Apeiron.WP`/`loop.spec_decr_nat`
with the array denotation above.

#### W3 unblock attempt (2026-09-12)

Two further materially distinct shapes were tried.

| Attempt | Change | Result |
|---|---|---|
| 4 | Recursion on a slice tail via `split_first` | The generated body is still a `partial_fixpoint`, and `core.slice.Slice.split_first` enters as an unmodeled external `axiom`. Rejected. |
| 5 | Fixed-array `while index < N` search with an early `return` | Charon translates each search to a total `loop` combinator marked `@[rust_loop]`; no `partial_fixpoint`, no external axiom. The supplied `Aeneas.Std.loop.spec_decr_nat` measure principle now applies. |

`src/safe_table.rs::find_from` and `empty_from` are now the `while` loops of attempt 5. The
extraction is regenerated: `word_safe_table4_get/insert/remove` call the loop-based generated
bodies, `proof/regen.sh --check` exits 0, and the throwaway source mutation (`stored == key` to
`stored != key`) makes `--check` exit 1. `cargo test --locked --test safe_table` passes 4/4 and
`cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` exits 0.

The remaining obligation is the canonical refinement itself: a `loop.spec_decr_nat` invariant
relating the generated `Array (Option (K × V)) N` state to a denotation over that array, then
`Apeiron.Table.get`/`insertWith`/`replace` lemmas for `get`, `try_insert` and `remove`. That is a
loop-invariant proof, not an extraction blocker; the first four attempts failed because no total
generated search existed, and one does now.

#### W3 first slice (2026-09-12)

`proof/corpus/SafeTableRefinement.lean` proves the generated search semantics with
`Aeneas.Std.loop.spec_decr_nat` at measure `4 - index` and invariant `NoMatchIn key l start index`:

- `find_from_loop_spec`: a returned `some j` is the first key match in `[start, 4)`, and `none` means
  no match there. `find_from_spec` lifts it to the `@[reducible]` `find_from`.
- The body is discharged through `Array.index_usize_spec` and a local `@[step]` `I64` equality lemma.
- Cone `{propext, Classical.choice, Quot.sound}`; `#audit`ed in `proof/corpus/AxiomAudit.lean`.
- Control: a throwaway copy with the generated `done (some index)` changed to `done none` fails the
  same proof. The real `proof/regen.sh --check` and the 3277-job bridge build stay green.

#### W3 write links

`try_insert_spec` and `remove_spec` in `proof/corpus/SafeTableRefinement.lean` link the extracted
`try_insert`/`remove` to the canonical table writes, under
`WellFormed t := t.len.val = List.countP (fun e => e.isSome) t.slots.val ∧ KeysUnique t.slots.val`:

- `try_insert_spec`: `Ok old` gives `old = tableHMap self key` and
  `tableHMap t' = Apeiron.Table.insertWith (fun new _ => new) key (some value) (tableHMap self)`;
  `Err _` gives `t' = self`.
- `remove_spec`: `old = tableHMap self key` and
  `tableHMap t' = Apeiron.Table.insertWith (fun _ _ => none) key none (tableHMap self)`.

The invariant supplies the `self.len + 1`/`self.len - 1` scalar bounds (`Aeneas.Std.Usize.add_spec`
needs `len + 1 ≤ max`; `Usize.sub_spec` needs `1 ≤ len`) and, for `remove`, the key uniqueness
that makes clearing the first match equal to the canonical `none` write. Cones inside
`{propext, Classical.choice, Quot.sound}`; `#audit`ed. Control: a throwaway copy with `try_insert`'s
generated `Ok none` changed to `Ok (some value)` fails `try_insert_spec` at
`SafeTableRefinement.lean:786`; the unmutated bridge builds 3351 jobs.

Both specs now conclude `WellFormed t'`. Preservation is proved by the `countP` step
(`countP_set_some`/`countP_set_none`, `one_le_countP_of_get`) and the uniqueness step
(`keysUnique_set_some`/`keysUnique_set_none`). Still open: `WellFormed` at the initial state
(`SafeTable::new`) at this historical checkpoint. The constructor and exact outcomes are now closed;
neither result authorizes removing `Map`.

### Optimized representation decision and remaining sequence

**Historical (superseded by the fixed-shard representation above).** This section names the retired
`SafeTable`/`Map` source; it is kept as the record of that decision, not as current acceptance.

`APEIRON-STORAGE-W4-T1`, owner Astra; base `7d6100c450e5da1fa469ffe9adc16a65c7e720ba`.
User-authorized supplied-checkout documentation exception; initially clean. No branch/worktree created.
The frozen relation is in `docs/kernel-boundary.md` §4. `SafeTable` is the checked reference;
whole unsafe/SIMD/threaded extraction is rejected. Share safe executing scalar bodies where extraction
permits; an abstract layout proof is not a Rust heap proof. S4/S5 stand; S6-S8 and optimized refinement
remain separate open obligations. The earlier W1-W4 table is historical, not renewed acceptance.

```text
SCOPE:      docs/kernel-boundary.md, docs/epics/storage-refinement.md; boundary and bounded sequence
REGISTER:   planning; excludes Rust/Lean edits, optimized promotion and retirement
CORPUS:     Apeiron.Table.Basic: "get_insertWith_self", "get_insertWith_ne",
            "build_eq_of_key_streams_eq"; native grade proved
REFERENCE:  /home/socol/Workspace/apeiron/proof/corpus/.lake/packages/apeiron/proof/Apeiron/Table/Basic.lean
GATE:       git diff --check (root), baseline exit 0; optimized refinement gate MISSING
GRADE:      anchor; bridge baseline lake build exit 0, 3351 jobs; no new refinement or measurement
```

Baseline source SHA-256 (bridge build only; extraction freshness not rerun):

| Path | SHA-256 |
|---|---|
| `src/map.rs` | `4226acc5282ce7626eefb955dff55ce3023d6e5e80167fd536bec982d86c1961` |
| `src/table.rs` | `5f6a62cb47a791162c0173f41d62342f7a0981c650ceed004e8cf0c83f0b9b9d` |
| `src/safe_table.rs` | `c09c5419456d355b2856064479114cf1c575db5016d9787078c0645a663a0479` |
| `proof/extract/lib.rs` | `e5a86d5add7d83a4c9b72dba045dd25cf6e9aa8924442cdc88faa23ae7a79fc8` |
| `proof/ApeironWord.lean` | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` |
| `proof/corpus/SafeTableRefinement.lean` | `6d089d616581d958b8379d1c68b69fdca01d1fa481d63c9424f23528ed278685` |

Resolved corpus checkout: `d9aefc93522e2109bc96158526de6dbd52477276`; manifest Mathlib:
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. Full source/tool identity acceptance remains S7.

#### Dispatch and lanes

Each row below is its own serial wave, task T1, with ID `APEIRON-STORAGE-W<number>-T1`.
Flash owns ordinary execution; Sol owns the explicitly marked proof-sufficiency tasks. Lead records
the integrated base and assigns `worktree-agent-<id>` before dispatch. This is a decomposition, not
permission to edit all listed paths now. Proposed files/commands marked **MISSING** require gate-first
implementation; exit 0 from a missing-test filter is never acceptance.

Lanes: **R** = `target/`; **E** = `proof/ApeironWord.lean` plus R and `proof/.lake/`;
**B** = `proof/corpus/.lake/` plus `proof/.lake/` and the Aeneas backend cache under R.
Any task running both extraction and bridge takes E+B exclusively. Controls use throwaway source and
generated copies and separate build outputs; immutable dependency caches may be reused read-only.
No sibling task shares an occupied lane. Every control records baseline/red/restored exit statuses,
source/generated digests, actual patched tool binaries and resolved Lean/Mathlib/corpus identities.

| Wave | Owned files (maximum three) / result | Owner; lane | Single gate (root unless stated) | Control |
|---|---|---|---|---|
| 5 | `proof/storage-controls.py`, this plan: durable S6 proof-control runner and record; **MISSING runner** | Flash; E+B | `python3 proof/storage-controls.py` | Independently mutate generated search hit, fresh write/old return, removal, constructor len, full payload/state, and force refusal on free/present keys. Each unchanged bridge build must fail semantically, not merely fail freshness; restored copies pass. Runner must reject a surviving mutation and missing tools. |
| 6 | `tests/safe_table.rs`, this plan: I64/capacity-four common-domain traces, compare all mappings/old values; separate fifth-key refusal; retain generic tests | Flash; R | `cargo test --locked --test safe_table` | In a throwaway `src/map.rs`, replace the displaced-value return with `None`; unchanged trace must fail. Safe full-refusal test must not call the unbounded Map insert as if it could refuse. |
| 7 | `proof/corpus/OptimizedTableRelation.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: formal logical projection/common-domain relation, joint witnesses and capacity mismatch counterexample; **MISSING module** | Sol; B | `lake build` (cwd `proof/corpus`) | Mutate the logical projection to drop a live entry; unchanged witnessed lookup/size correspondence fails. Grade is proved mathematics about layouts, only model/anchor for actual Map. No assumed Map transition theorem. |
| 8 | `src/table.rs`, `src/table/scan.rs`, `tests/both_tag_paths_agree.rs`: move the existing scanner unchanged to a shared module; preserve exports and target selection; **MISSING module** | Flash; R | `cargo test --locked --test both_tag_paths_agree` | Corrupt one returned lane bit in the executing scanner; unchanged scalar oracle fails. Fix fixtures to include actual EMPTY/TOMB, one-live and one-free lines. No algorithm or performance claim. |
| 9 | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: extract actual shared SWAR match/free bodies only | Flash; E | `proof/regen.sh --check` | Mutate a real SWAR mask constant; unchanged generated artifact must fail freshness. Unsupported calls/axioms stop the task, not trigger a proof-only scalar rewrite. |
| 10 | `proof/corpus/ScanRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`: exact masks for all 16 lanes/all byte values; **MISSING module** | Sol; B | `lake build` (cwd `proof/corpus`) | Mutate generated gather or free polarity; unchanged exact-mask theorem fails; restored copy passes. This proves SWAR only, not SSE2 dispatch or Map. |
| 11 | `docs/epics/storage-refinement.md`, this plan: gate-first design for the actual scalar probe/transition core and heap adapter, constructor/preservation/refusal obligations | Astra owner, Sol read-only verdict; no build lane | `git diff --check` | Requirement review must reject a proposal that assumes the desired heap simulation, omits unwind, equates full with OOM, or selects a proof-only path. Semantic acceptance command **MISSING** until the memory semantics/tool route is selected; this row cannot close optimized refinement. |

The scanner sequence is a follow-up after S6 and the shared assurance barrier, not a prerequisite
invented for S4/S5 or a substitute for the heap bridge. Scalar probe geometry (`Geom`, `Descent`,
`group_at`) is the next candidate after scan, but contains `assert_unchecked` and finite counters:
do not allocate its implementation before the last design row specifies actual-source extraction,
overflow bounds and rejecting controls. Serial `CountTable` partition/count refinement follows that
as a distinct contract; live workers/SSE2/unsafe heap semantics need separately approved tool routes.

#### Integration barriers and exact blockers

1. Run `lake build` in `proof/corpus` as the single next baseline gate; then implement the S6 runner,
   execute its controls, integrate the focused trace tests, and obtain Sol's relation/premise verdict.
2. Before optimized-source work, complete S7 on that identity: all four AGENTS §6 Rust commands,
   `proof/regen.sh --check`, `lake build` in both packages, and named controls, serialized. Missing
   tools block. The constructor needs the patched tools documented in `proof/README.md`; record
   `AENEAS_HOME`, do not substitute the unpatched release silently.
3. S8 is a separate gate-design dependency, not waived: contrary to the handoff's installed wording,
   `tools/check-claims.py` and `tools/check-assurance.py` are absent at this base. The latter command
   `python3 tools/check-assurance.py` exits 2 (file not found). Lead must authorize a separate
   assurance task and correction to the out-of-scope handoff/AGENTS wiring. Proposed separate tasks:
   - Claim gate: `tools/check-claims.py` plus the existing ledger selected by its owner; gate
     `python3 tools/check-claims.py`; delete a required joint witness/audit binding in a throwaway
     fixture and require rejection. Gate/ledger design is **MISSING**, not dispatch-ready proof work.
   - Perimeter gate: `tools/check-assurance.py` plus this plan; gate
     `python3 tools/check-assurance.py`; inject unlisted production source, a 1,501-line owned file,
     a thirteenth immediate source file and a >5,000-line trusted checker population, requiring
     rejection. Bind all production `src/`/`tests/`, inventory external trust separately; do not
     exclude optimized files to turn the size check green. Existing over-limit populations require
     a separately frozen structural plan, not an exemption here.
   - Gate wiring: `AGENTS.md`, `docs/plan.md`; gate `git diff --check`; requirement review
     checks that missing checker execution cannot be reported as S8 closed. No semantic grade.
   These are separate serial waves owned by Flash except boundary/claim decisions, which require
   Astra/Sol. Checker tasks take no Rust/Lean lane unless their approved design invokes those tools.
4. Integrate shared scanner, extraction, then mask proof in order. Re-run S7/S8 and controls on one
   identity. Stop before claiming the optimized relation: no real heap-to-layout simulation exists.

**Sol trigger:** the relation changes theorem premises/attribution. Required decision: are the
joint `MemoryValid`/`PlacementValid` witnesses and operation domains sufficient and non-vacuous,
and which actual Rust semantics could discharge the heap simulation without an extra axiom?
No Sol verdict is supplied by this architecture decision. Source observations needing scrutiny:
`Map::clear` drops before blanking tags; `rehome` temporarily removes arrays while counters remain;
generic callbacks may unwind. These are proof obligations, not a completed safety audit.

**Residual claim:** only the existing safe/kernel refinements are checked. No performance measurement,
whole-Map equivalence, generic trait correctness, CountTable overflow/scheduling theorem, compiled
execution guarantee or ring-0 guarantee is added. A stored-state kernel consumer also needs its real
read/compute/write body and exact failure contract selected before composition work is dispatchable.

### S6 baseline/red/restore controls

**Historical (superseded by the fixed-shard representation above).** These controls bind the retired
`SafeTable` source and its generated bodies; they are the record of that checkpoint.

Throwaway root `.mut-s6/` (btrfs reflink of `proof/`, `target/aeneas` symlink; same filesystem as the
repository). Baseline `cd .mut-s6/proof/corpus && lake build` exit 0, 3351 jobs. Toolchain: the single
patched `target/aeneas` symlink, no parallel toolchain directory.

| Obligation | Mutated file (throwaway) | Mutation | Red command (cwd) | Exit | Red line | Restore sha256 | Restore gate |
|---|---|---|---|---|---|---|---|
| Search `find_from_loop_spec` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.find_from_loop.body`: `if index < N` -> `if index > N` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:103:10` (`find_from_loop_spec`) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Write `try_insert_fresh_iff`, `try_insert_err_iff` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.try_insert`: fresh-branch `Ok none` -> `Ok (some value)` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:877:16` (`try_insert_spec`; write link the exactness headlines consume) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Initialization `safe_table_new_eq_emptyTable` | `.mut-s6/proof/ApeironWord.lean` | `SafeTable.new`: `len := 0#usize` -> `len := 1#usize` | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:1067:2` (`safe_table_new_eq_emptyTable`) | `e509d24e634e457895baa49fda6bec6264e1e149f96594e00a7c703093ec8391` | exit 0, 3351 jobs |
| Full handling `try_insert_fullTable_refusal` | `.mut-s6/proof/corpus/SafeTableRefinement.lean` | `fullTable`: slot 0 `some (k0, v0)` -> `none` (falsifies `NoEmptyIn`) | `lake build` (`.mut-s6/proof/corpus`) | 1 | `SafeTableRefinement.lean:1222:67` (`fullTable_noEmpty`; `NoEmptyIn` witness the refusal theorem consumes) | `6d089d616581d958b8379d1c68b69fdca01d1fa481d63c9424f23528ed278685` | exit 0, 3351 jobs |

Restored repository gate: `cd proof/corpus && lake build` exit 0, 3351 jobs; `proof/regen.sh --check`
exit 0, fresh extraction, `ApeironWord` 1708 jobs.

### Revised production boundary and dispatch order

Owner: lead; recorded by `APEIRON-PERF-RESTORE-W4-T1`; base
`c450c222905c050f793a23332708c230c658f82c`. Documentation only; gate `git diff --check` and
`python3 tools/check-assurance.py --check`. This section records the Astra architecture decision that
replaces the earlier production decomposition and the lead-ratified resource/refusal contract
(`APEIRON-PERF-RESTORE-W4-T2`). It supersedes the
"Production recovery, then extraction/refinement" row and the fixed-shard "Remaining serial waves"
above as the dispatch sequence; those sections remain the historical record, not current
instructions.

#### Acceptance threshold (owner decision)

The selected implementation must match or beat **every** named applicable incumbent on every
applicable cell, `selected/incumbent <= 1.0`, never behind and ideally ahead. The named incumbents
and the per-cell/incumbent applicability registration are in the
[lifecycle annex](#performance-preserving-consolidation-lifecycle-annex) above; there is no
family-wide exemption and missing, duplicate, invalid or inconclusive pairs reject. There is no
historical-envelope tolerance. This supersedes the RFC elapsed-median floor
`selected/hashtrinity <= 1.25`. The superseded 1.25 floor and historical-envelope language still
appears in `docs/direction.md` and `docs/plan.md`; it is reconciled in
`APEIRON-PERF-RESTORE-W5-T1`, not here. Within this plan the threshold above is the only acceptance
rule.

#### Staging: serialized representation integration wave

The growable-representation change is a multi-task serialized integration wave, not a three-file
edit. Each task owns at most three files and keeps `cargo test`, `cargo clippy` and `cargo fmt`
green plus its focused gate; the wave closes only when extraction, refinement, Kani, the REG-1
harness and the shared gates pass on the integrated identity. An intermediate proof failure is not
an accepted wave result. Reason: changing the representation simultaneously changes admission
(`src/table.rs` positive-`N` `ShardFull`), the extracted array representation
(`proof/ApeironWord.lean`), the bounded-admission refinements, `src/proofs.rs` Kani predicates and
the REG-1 workloads/oracles.

#### Revised dispatch order prefix

| Order | Task | Owned scope |
|---|---|---|
| 1 | `APEIRON-PERF-RESTORE-W4-T1` | this documentation task: revised production boundary |
| 2 | `APEIRON-PERF-RESTORE-W5-T1` | reconcile `docs/direction.md`, `docs/plan.md`, `proof/README.md` |
| 3 | `APEIRON-PERF-RESTORE-W6-T1` | root real `spread`/`finish`: `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` |
| 4 | canonical corpus task (separately assigned) | `Table/WordIdentity.lean`, registered in `Audit.lean` |
| 5 | `APEIRON-PERF-RESTORE-W7-T1` | bridge pin transport |
| 6 | `APEIRON-PERF-RESTORE-W8-T1` | `proof/corpus/TableRefinement/CoordIdentity.lean` plus `AxiomAudit` |
| 7 | interface expand / migrate / contract batches | separately scoped tasks, at most three files each |
| 8 | representation integration wave | the serialized wave above |

The resource/refusal contract is ratified in `docs/kernel-boundary.md` (option (b), checked
reservation/allocation failure, transactional); the identity theorem, bridge pin transport and
representation wave are unblocked. The canonical release/commit remains separately authorized; the
exact remaining proof/task inventory is no longer blocked on the resource decision.

### Rework dispatch order

Owner Astra; recorded by the rework contract; base
`c450c222905c050f793a23332708c230c658f82c`. Documentation only; gate `git diff --check` and
`python3 tools/check-assurance.py --check`. This order supersedes the "Revised dispatch order
prefix" above. The first growable `Table` at `src/table.rs` sha256
`c34ddf17ef2ed7a81fd4b3e16447479b55d737af8f5253b99dd397414d9b1dc1` is rejected; its measured red
and defect list are in the
[local boundary](../kernel-boundary.md#reworked-growable-representation-astra-decision-first-cut-rejected).
The bounded fixed-shard table is not restored and `examples/reg1/candidate.rs` is not promoted.

```text
SCOPE:      docs/kernel-boundary.md, docs/epics/storage-refinement.md, docs/epics/storage-refinement.md;
            adjudicate address, Sigma, search and refinement premises
REGISTER:   refinement/state/planning; no source, proof, benchmark or extraction change
CORPUS:     Apeiron.Table.{get_insertWith_self,get_insertWith_ne,get_build_acc,
            build_eq_of_key_streams_eq}, Basic.lean:87-174; native proved
REFERENCE:  src/{table.rs,table/scan.rs,coord.rs}; 3a95bb5:src/{table.rs,coord.rs}; baseline receipt
GATE:       git diff --check, repository root; no task-specific red for this documentation gate
GRADE:      specification/anchor; downstream tested/proved grades require their named gates
```

#### Revised representation boundary: flat resident area and spills

The Astra verdict revises the selected structure from a single growable key-partitioned `Table` to
two coupled regions:

- A **table-level flat resident area**: shard-major `256 x L`, admission
  `direct_word = Some(w) ∧ lane < L`, checked sizing, a stated byte budget and a density test.
  Failures spill to the grouped regions rather than refuse.
- **Independently load-sized grouped spill regions** beside the flat area.

One representative per key is held under a composite `DirectCover`; extent growth must reclassify
newly eligible spills. Key-dropping uses a compile-time `KeyRep` policy (`Stored<K>` or `Regen`
with an exact codec law), not a sibling count table and not a lossy `FreqMap`. Restoration also
includes the key-partitioned load-sized shard build, deterministic schedule, grouped probe and
key-dropping count/dedup; the proof, extraction, Kani, registration and measurement work follows on
the same source identity. The boundary is a recorded plan, not a built, extracted or measured
result, and no speed recovery is claimed. REG-1/G1 stays OPEN.

#### Sol contract adjudication (`APEIRON-PERF-REWORK-W5-T2`)

The Astra boundary is accepted with these proof premises:

1. One generic `Table` uses `TableAddress { place: u64, direct_word: Option<u32> }` and
   `Home::address(sigma,key)`. `WordHome` supplies `Some(x as u32)` exactly below `2^32`; all
   placement-only adapters supply `None`. The direct lane is `u64::from(x) >> 8`, not
   `place >> 8`, and full `(Sigma,K)` equality decides identity.
2. The logical key and payload key are `(Sigma,K)`; reads/writes receive `sigma`; build consumes
   `&[((Sigma,K),V)]`. Equal `K` at unequal `Sigma` remains two keys through every transition.
3. Direct-empty early absence is admitted only from reachable `DirectCover`, monotone occupancy and
   tag authority. A first window vacancy may stop the prefix, not the coverage sweep. Every absence
   without an earlier proved certificate requires finite all-group coverage. Window extraction masks
   each seven-bit value by `0x7f` before group reduction; `gmask` never extracts a window.
4. `GrowableGeometry`, `GrowableDirectory`, `GrowableScan`, `GrowableSearch`,
   `GrowableRefinement` and `GrowableSchedule` carry the theorem set in the spec and boundary.
   `GrowableSchedule` composes its per-key subsequence equation with
   `Apeiron.Table.build_eq_of_key_streams_eq` (`Basic.lean:167-174`).

This is a contract change. Relative to `Home::home(&K) -> u64`, refinement adds `Sigma`-indexed
address laws, an optional exact-word certificate and its routing equation, full-pair identity,
reachable search certificates and relocation preservation. These are anchors until the named
real-source bridge gates pass.

The strict 126/160 failures at that adjudication are against the serial `hashtrinity::Table`
incumbent at `4cc14471`; the current strict run is the fresh post-spill full-profile receipt
(lifecycle annex above; sha256 `946b5d40…`), which still reports `full_acceptance=false`. Serial
recovery remains open and no speed recovery is claimed. Partitioned/parallel build plus grouped probe for the spatial packed
family is distinct production work, not evidence supplied by serial recovery. REG-1/G1 remains OPEN.

#### Gate-first tasks

`APEIRON-PERF-REWORK-W1-T1` and `W2-T1` are gate-first and must land before the storage rework;
`W3-T1` captures the unchanged first-cut baseline before any rework edit.

| Task | Owned files (at most three) | Completion, one focused gate and control |
|---|---|---|
| `APEIRON-PERF-REWORK-W1-T1` | `examples/reg1/strata.rs`, `examples/reg1/matrix.rs`, `examples/reg1/matrix.json` | Focused registration tests; re-register refusal/resource cells and archive the old fixed-shard receipts. Control: the old `ShardFull` digest rejects. |
| `APEIRON-PERF-REWORK-W2-T1` | `examples/reg1/performance.rs`, `examples/reg1/strata.rs`, `examples/reg1/main.rs` | Strict performance-acceptance mode: a per-cell failure makes the command exit nonzero. Control: a valid report with one failing cell exits nonzero. |
| `APEIRON-PERF-REWORK-W3-T1` | new `docs/evidence/` receipt | Capture the unchanged first-cut before-receipt under `docs/evidence/` before rework. |
| `APEIRON-PERF-REWORK-W4-T1` | `src/table.rs`, new `src/table/scan.rs`, `tests/table.rs` | Geometry, exact SWAR, bounded first-fit, non-optional slots. |
| `APEIRON-PERF-REWORK-W5-T1` | `src/table.rs`, `tests/table.rs` | Direct candidate/directory and transactional relocation. |
| `APEIRON-PERF-REWORK-W6-T1` | `src/table.rs`, `tests/table.rs` | Implement `TableAddress`, `Home::address`, full `(Sigma,K)` identity/build input and direct-word routing; test unequal-`Sigma`/equal-`K` separation and placement-only `None`. Consumer migrations are separate at-most-three-file tasks. |
| `APEIRON-PERF-REWORK-W7-T1` | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | Root the real address, search, relocation and sequential build bodies and regenerate. |
| `APEIRON-PERF-REWORK-W8-T1` | `proof/corpus` bridge theorems (serialized) | `GrowableGeometry`, `GrowableDirectory`, `GrowableScan`, including the seven-bit mask and direct routing. |
| `APEIRON-PERF-REWORK-W9-T1` | `proof/corpus` bridge theorems plus `AxiomAudit.lean` (serialized) | `GrowableSearch`, `GrowableRefinement`, audited under `{propext, Classical.choice, Quot.sound}`. |
| `APEIRON-PERF-REWORK-W10-T1` | sequential partition/worker/assembly source plus tests, separately decomposed | Restore partitioned/parallel build and grouped probe as spatial production work; preserve per-key subsequences. |
| `APEIRON-PERF-REWORK-W11-T1` | `proof/corpus` schedule bridge plus `AxiomAudit.lean` (serialized) | `GrowableSchedule` and composition with `build_eq_of_key_streams_eq`; thread realization remains outside the extracted claim. |
| `APEIRON-PERF-REWORK-W12-T1` | `src/proofs.rs`, `examples/reg1/coverage.rs`, `examples/reg1/coverage.json` | Kani harnesses for the reworked geometry and locate. |
| `APEIRON-PERF-REWORK-W13-T1` | `examples/reg1/workload.rs`, `examples/reg1/correctness.rs`, `examples/reg1/matrix.rs` | Workload and correctness re-registration. |
| `APEIRON-PERF-REWORK-W14-T1` | `examples/reg1/matrix.json` plus serial and spatial before/after receipts | Every applicable cell `selected/incumbent <= 1.0`; serial and spatial incumbent families remain distinct. |

#### Refusal-cell re-registration

Do not merely mark the 8 rejected cells not-applicable. Archive the old fixed-shard refusal receipts;
register successful growable insertion and update traces with fresh oracle digests and explicit
operation names; add genuine zero-width, sizing-overflow and injected-reservation-failure rows with
unchanged-state and no-callback observations. The strict performance-acceptance mode from `W2-T1` is
the per-cell authority; the aggregate exit status is not.

### Extraction-shaped representation dispatch order (APEIRON-EXTRACT-REP)

Owner Astra; recorded by `APEIRON-EXTRACT-REP-W1-T1`; documentation only; gate `git diff --check`.
This order supersedes the "Rework dispatch order" and "Gate-first tasks" above as the dispatch
sequence; those remain the historical record. The representation boundary is in the
[local boundary](../kernel-boundary.md#extraction-shaped-representation-boundary-astra-apeiron-extract-rep);
the contract is in the
[spec](storage-refinement.md#extraction-shaped-representation-contract-astra-apeiron-extract-rep).

| Wave | Obligation |
|---|---|
| `W1` | Projection-free carrier on both faces: `Stored<K>{key}`, `Regen<C>{identity,codec}` with manual `Clone`, `FlatSlot<R,V>` and spill `(Sigma,R,V)`, no `R::Held`; preserve `Table<…,Stored<K>>` and `Table<…,Regen<I64WordCodec>>`. |
| `W2` | Digest fix: `proof/regen.sh` drops the deleted `src/table/resident.rs` from the hashed source list. |
| `W3` | Resource interface decision: keep planning, reservation and allocation unextracted; allocating public methods stay unrooted until a checked allocator primitive/tool model exists. Decision recorded by `APEIRON-EXTRACT-REP-W5-T1`. |
| `W4` | Prepared resource adapter in production: `Prepared { plan, storage, outcome } | Refused(StorageError)`; checked-plan, reserve-all, then apply/commit; pending insertion inside the transaction; refusal stutters. This remains unextracted. |
| `W5`–`W10` | Prepared-spill extraction checkpoint: `proof/ApeironWord.lean` sha256 `faa257a8720c71ee7a08497621ddecd40495cf3729ee10c56f18119f5778113a` contains the real spill-only roots. It does not contain composite `Table::locate`, `Site` or `DirectCover`; no composite-search refinement follows. |
| `W11` | Search boundary decision, `APEIRON-EXTRACT-REP-W11-T2`: keep the probe-order decision unextracted under `PinnedJoinBoundary`; root the real straight-line apply under `SearchDecisionCorrect`. Contract grade `anchor`; supplied context-join evidence `tested`. |
| `W12` | **Next bounded task:** factor the production search in `src/table.rs` into a decision returning `(Found|Vacant|Exhausted, Flat(i)|Spill(shard,i)|NoIndex)` and a straight-line read/insert apply; keep direct, seven-window and mandatory-sweep order unchanged. Add focused cases in `tests/table.rs`. Gate and a late-hit/sweep mutation control must be assigned in the source task. |
| `W13` | Root both carrier specializations of the real apply in `proof/extract/lib.rs`, add only those roots to `proof/regen.sh`, and regenerate `proof/ApeironWord.lean`. Gate `proof/regen.sh --check`; control mutates an apply branch in a throwaway generated copy. |
| `W14` | Prove `GrowableSearch` over the extracted apply under `SearchDecisionCorrect`, with `search_decision_simulates` targeting total-by-default `Apeiron.Table.get`, `get_insertWith_self` and `get_insertWith_ne`; register the headline and rejecting control under the closed cone. |
| `W15`–`W17` | Prove, in order, `GrowableRelocation`, `GrowableRefinement` and `GrowableWitnesses`, each carrying `SearchDecisionCorrect`, `RepR` and `PreparedResources` and each with its own rejecting control. |
| `W18` onward | `GrowableDirectory`, `GrowableSchedule` and `GrowableBatch` retain their existing routing, per-key-subsequence and grouped-assembly contracts; then controls, Kani, coverage, matched correctness and performance. An allocator primitive-model task is added only if allocating public methods are later required as roots. |

**W3 Sol contract.** Premise `PreparedResources` requires checked arithmetic, exact old/pending entry
classification, in-bounds destinations, sufficient initialized reserved storage and no further
fallible allocation in the core. The simulation obligation `resource_adapter_simulates` has two
branches: prepared core success preserves `RepR` and denotes canonical
`Apeiron.Table.{get,insertWith,build}`; refusal supplies no core input, leaves insert state unchanged
with no clone/merge/relocation, and returns no build table. Allocator/global-allocator realization is
outside the claim. Grade: contract `anchor`; supplied refusal observations and context-join spike
`tested`; no proof.

**W11 Sol contract.** Premise `SearchDecisionCorrect` requires `Found` iff the queried full
`(Sigma,K)` key is present at the returned in-bounds unique representative; `Vacant` only for an
absent key at the first-fit lane; and `Exhausted` iff absent with no vacancy. Direct-empty absence
requires `DirectCover`; a window vacancy may stop the prefix but never the mandatory all-group match
sweep; every other absence requires complete finite coverage. `GrowableSearch` refines the real apply
under this premise. `GrowableRelocation`, `GrowableRefinement` and `GrowableWitnesses` carry it;
`GrowableDirectory`, `GrowableSchedule` and `GrowableBatch` are unchanged. Rust/Kani can support only
`tested`/bounded evidence for the decision; the apply becomes `proved` only after real-source
extraction, canonical refinement, bounded bridge build, audit and rejecting control.

`GrowableSchedule` still proves the implementation-specific per-key value-subsequence equation and
composes it with canonical `Apeiron.Table.build_eq_of_key_streams_eq`
(`Table/Basic.lean:167-174`). `Song/kernel/Web.lean`
`preduce_schedule_free`/`preduce_append`, read at
`/home/socol/Workspace/song-ref/lean/Song/kernel/Web.lean` revision `ae688f66`, is the external Song
twin only; no Song declaration enters the Apeiron Rust/canonical cone.

**Prerequisites and open gates.**

- `W2` removed the deleted `src/table/resident.rs` digest path. The current generated artifact remains
  bound to the live source list and is spill-only.
- `proof/corpus/lakefile.lean` still lists the fixed-shard and joint consumers; a separate archival
  manifest is required before a full corpus build.
- Supplied spike evidence, grade `tested`: Aeneas `505b6ca-dirty` / Charon `0.1.254` fails
  `Could not match the contexts` at `Table::plan_rebuild` (`src/table.rs:678:16-680:17`). The generic
  borrowed-carrier/live-guard/computed-classification shape reproduces; scalar and tested
  straight-line/accumulated-`Option`/cast variants pass; `-loops-to-rec`, `-loops-no-rec` and
  `-strict-joins` do not remove it. Classification is pinned Aeneas limitation (b), not allocator
  case (c); further planner reshaping is not scheduled.
- Reachable `try_reserve_exact` is independently unmodeled. The planner/reserver, allocator/global
  allocator, executor/`thread::scope`, rustc/LLVM, ABI, GPU and `unsafe` container stay outside the
  proved claim. Slang's empty cone remains separate.
- The current generated artifact is spill-only. Composite search, the decision contract and its real
  apply root remain open. No proof or speed is claimed. REG-1/G1 remains **OPEN**.

## Coordinate-keyed counting-container contract

Recorded by `APEIRON-COUNT-W1-T1` from `/home/socol/oc-tmp/escalation/APEIRON-PARITY-REVIEW5.verdict.md`
and `/home/socol/oc-tmp/escalation/APEIRON-REORIENT.verdict.md` §3. Grade `anchor`: this is an
architecture and contract-selection record. It adds no source, extraction, test or measurement result,
and REG-1/G1 remains **OPEN**.

REVIEW5 is the **selected counting-face architecture**: one single-region, coordinate-keyed,
open-addressed counting container with no locator. The generic growable flat/spill
`Table<K, V, H, N, R>` (`src/table.rs:237-268`) is **retained unchanged** as the map face. This
contract supersedes the T3 clause "not an exemption or a second container" at
`docs/direction.md:522` and the "a new container" non-goal at `docs/direction.md:631-632`
**for the counting face only**; no other
non-goal, obligation or acceptance criterion is superseded. Both citations are bound to the committed
revision `e6e9d13` (`docs/direction.md` sha256 `c07fa77926cba3003e34928e6483b494ab10895c8cfc9d8feeb9ee1a9b9a03da`),
which is where both phrases are: `522` reads "not an exemption or a second container", and the
non-goal list runs `a new container` at `631` into `632`. The uncommitted working-tree
`docs/direction.md` (sha256 `255997a2c2d6c4bd20bf11485d208828f8ca129c467c157e5f6394a9a23e93e3`,
78 insertions) renumbers the same text to `:518-519` and `:699-700` and adds the
**Counting-face supersession** grant at `:517` and `:581`, which does not exist at `e6e9d13`. So the
grant that licenses taking a second representation into the tree is itself uncommitted, and is owned
by `APEIRON-COUNT-W1-T3`; a citation against either tree is checkable only against its named digest.
The canonical object is unchanged:
`Apeiron.Table.{HMap,get,insertWith,build}` in
`/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/Basic.lean` (`HMap:63`, `get:71`,
`insertWith:80-81`, `build:121`, `get_build_acc:134`, `get_insertWith_self:89`,
`get_insertWith_ne:97`, `build_eq_of_key_streams_eq:169`).

### Intent

**Problem.** The counting face cannot reach the owner's target ratio because the resident-coordinate
admission and the addressing decision are separate losses. `Flat::index` (`src/table/state.rs:492`)
admits only a `u32` resident coordinate, and `examples/reg1/adapters/native_strata.rs:93-98`
certifies residency only when the whole coordinate word fits `u32`, so a key whose coordinate is
outside that domain is spilled while the resident area, 256 spill regions and the locator are still
allocated (`src/table.rs:237-268`). `Table::shard` (`src/table/state.rs:366-370`) routes on
`direct_word` when present and on `place` otherwise, so changing one home adapter cannot force a
single region. HashDOS resistance is a separate obligation and concentrating keys supplies none of it.

**Alternatives rejected.**

| Alternative | Why not |
|---|---|
| Route every counting key to one shard of the existing table | It is the same sharded `Table`: the resident area, 256 spill regions and the locator remain, and `Table::shard` has two arms, so one-region routing is not obtained by changing a home. |
| Literal coordinate-as-array-index | It requires a `2^63` array; a compact open-addressed coordinate table does not. REVIEW5 distinguishes the two proposals. |
| Benchmark-only substitution for `StrataHome` or `RawStrataHome` | Forbidden: REVIEW5 admits a one-shard placement policy only as an explicitly contracted reusable production policy, never as a benchmark-only substitution. |
| Provisioning surgery on resident payloads | `examples/reg1/adapters/native_strata.rs:93-98` certifies residency by coordinate width; changing payloads alone does not move spill-routed keys. |
| A second example-only container | An example competitor is not a production implementation and discharges no acceptance row. |
| R1-only as the next step | REVIEW5 rejects R1-only as the selected next step; R1 follows the representation evidence. |

**Success condition.** The counting container denotes canonical `Apeiron.Table.HMap` on its admitted
domain, its reads and writes are proved from real extracted source under the closed cone, and the
registered full-profile acceptance reports `full_acceptance=true` with the owner target applied.

**Non-goals.** Removing or weakening the generic `Table`, its `Sigma`-separated logical key, its
transactional refusal or any existing acceptance obligation; changing the canonical corpus; a
benchmark-only path; HashDOS by key concentration; universal O(1)/WCET claims; the generic-Table
proofs discharging any counting-face obligation.

### Spec

Ten frozen clauses. A clause with no executable gate is recorded **MISSING** and no implementation
task may open against it.

**1. Selected physical representation, public operations, consumers.** One single region of
open-addressed slots holding `(Sigma, u32)` coordinate words and counts, with tags as the sole
occupancy authority, a growth policy and no `TableLocator` and no 256-region directory. Public
operations are exactly: construction, `get`, new-first `insertWith` at one coordinate, and ordered
`build`.

**Production consumers, measured.** No counting container and no production call site reaching one
exists. `rg -ni "counting|count_par|get_each|CountMap|CountingTable|FreqMap" src/` returns only
`Table::get_each` / `get_each_at` (`src/table/geteach.rs:115,87`) and the parallel batch and build
(`src/table/parallel.rs:482,497,793,811,930`); `rg -n "fn n\b|fn n_at|fn n_parallel" src/` returns no
match, so the generic `Table` has no counting surface to redirect.
`examples/reg1/adapters/native_strata.rs` is under `examples/`, included by
`examples/reg1/strata.rs:94-95`, and is a harness key/home adapter: its `address` (`:93-98`,
`:141-146`) certifies `direct_word: u32::try_from(key.tag).ok()` and refuses nothing.
`src/table/state.rs` is the generic `Table`'s private state — the representation this contract replaces
on the counting face, not a consumer of it.

The **minimum production path that becomes the consumer** is the coordinate→count aggregate that
already has a production form in `src/` and no other:

| production call site | operation | incumbent arm it must match |
|---|---|---|
| `Table::sum_parallel`, `src/table/parallel.rs:979-983` | grouped reduction over default-face probes | `hashtrinity::FreqMap::count_par` + `get_each`, `examples/reg1/acceptance.rs:491-492` |
| `Table::get_each_at`, `src/table/geteach.rs:87-109` | ordered batched read at one face | `hashtrinity::FreqMap::get_each`, `examples/reg1/acceptance.rs:501-502` |

`sum_parallel` is the binding one and the reason is structural rather than convenient: at
`src/table/parallel.rs:958-962` it clones every query key into a `Vec<((Sigma, K), ())>` before
probing, so the generic `Table` pays a retained-`K` widening per query to reach a plain `i64`. The
counting container replaces that widening with the packed `(Sigma, u32)` word of clause 2 and retains
no key, which is the per-query identity loss clause 2 forbids on the map face.
`examples/reg1/adapters/spatial.rs` remains the registered
measurement harness, not a production consumer, and cannot discharge this clause alone.

**2. Admitted coordinate domain, identity and `Sigma` separation.** The admitted domain is the
packed `u32` coordinate word, the same domain `Flat::index` (`src/table/state.rs:492`) and
`examples/reg1/adapters/native_strata.rs:93-98` already certify; a coordinate outside it is refused,
not truncated or wrapped. The logical key is exactly `(Sigma, coordinate)`: equality compares both
components, reads and writes receive `Sigma`, and for distinct `Sigma` at one coordinate an update
or read at `(Sigma_1, c)` leaves `(Sigma_2, c)` unchanged even under identical placement. No lossy
tuple identity, no coordinate-equality identity claim and no sentinel alias: the codec
`encode/decode : (Sigma, u32) → (Sigma, u32)` must be total, stable, equality-respecting and satisfy
`decode(s, encode(s, c)) = c` on the admitted domain, and two admitted keys may never share a stored
word. Where a key is reconstructed rather than carried, the reconstruction uses only a verified codec
inverse and never `Addr.place`.

**Frozen codec: `CountKeyCodec`, instance `U32CoordWord`.** A sealed trait with exactly one admitted
instance, in the shape of `ExactCodec` / `I64WordCodec` (`src/table/keyrep.rs:182,281`), so no
unreviewed instance can be selected. Three operations, no others:

| operation | definition | role |
|---|---|---|
| `encode(sigma, coord: u32) -> u64` | `(u64::from((sigma.value() + 1) as u32) << 32) + u64::from(coord)` | the only producer of a stored word; total |
| `matches(sigma, stored: u64, coord: u32) -> bool` | `stored == encode(sigma, coord)` | the hot identity operation; materializes nothing |
| `decode(sigma, stored: u64) -> u32` | `stored as u32`, reached only after `matches` returned true | the verified inverse on the image; never reads `Addr.place` |

`Sigma::value()` (`src/cell.rs:39`) is `-1`, `0`, `1` on `Elliptic`, `Parabolic`, `Hyperbolic`, so
`value() + 1` is the injective face tag `0`, `1`, `2`. That is the `Sigma`-separation encoding: the
face half occupies bits 32–33 and the coordinate half bits 0–31, two disjoint bit ranges, so the sum
above is the bitwise-or packing and the image is exactly `{0,1,2} × 2^32` — `12_884_901_888` distinct
stored words, none reused. The injectivity obligation is therefore
`encode(s1, c1) = encode(s2, c2) ⟹ s1 = s2 ∧ c1 = c2`, and the law is
`decode(s, encode(s, c)) = c` for every `s` and every `c < 2^32`. `decode` is total on that image; the
container must never present a face-mismatched word to it, because identity is the `matches` equality
first — the order `FlatSlot` identity already uses at `src/table.rs:65`, `flat.live && flat.sigma ==
sigma && R::matches(sigma, &flat.held, key, address)`. This is what discharges "two admitted keys may
never share a stored word": the stored word carries the face, so `(s, c)` and `(s', c)` differ for
`s ≠ s'` at one coordinate and one placement.

**Sentinels and boundary values, frozen.**

| value | admitted | why |
|---|---|---|
| `0` | yes | `encode(Elliptic, 0) = 0`, so a stored word of `0` is a **live** coordinate and never a vacancy marker; occupancy is the tag, per clause 1 |
| `2^31` | yes | the top bit of the `u32` coordinate word: `word >> SHARD_BITS` is `2^23` (`src/table/state.rs:493`) and an `i32` reading of the word changes sign here |
| `2^32 - 1` | yes | the admitted-domain edge. `Flat::index` takes a `u32` (`src/table/state.rs:492`) and `u32::try_from` accepts it |
| `2^32` | **no — the first refused coordinate** | the first `u64` tag `u32::try_from(key.tag).ok()` rejects (`examples/reg1/adapters/native_strata.rs:96,144`), so `direct_word = None` and the key routes to spill today. The counting container refuses it: no truncation to `0`, no wrap to `2^32 - 1` |

Bits 34–63 of a stored word lie outside the image of `encode` for every admitted key, so no word above
`0x2_FFFF_FFFF` is ever a live coordinate and none may be written as one.

**3. Denotation and refinement targets.** The relation is
`RepC t (Apeiron.Table.HMap (Sigma × Nat32) Int64)` where `Nat32` is the admitted coordinate word.
The refinement targets are exactly `Apeiron.Table.HMap`, total-by-default `Apeiron.Table.get`, the
**new-first** `Apeiron.Table.insertWith` (new value then old value, as in
`insertWith f k v m = fun x => if x = k then f v (m x) else m x`), and the ordered
`Apeiron.Table.build`, with `get_insertWith_self` at the queried key, `get_insertWith_ne` at every
unequal key, and `build_eq_of_key_streams_eq` supplied through an implementation-specific per-key
value-subsequence equation. Word-instance results do not establish arbitrary-carrier refinement.

**4. Establishment, preservation, bounds and witnesses.** The real constructor establishes `RepC`
from checked empty state. Successful insertion and growth preserve `RepC`, uniqueness, tag authority,
the coverage clause and the count value. Absence is returned only under a proved coverage certificate
or complete finite probe coverage; an occupied non-matching slot proves nothing. A recorded first
vacancy and the mandatory sweep both obey the probe order of the extracted body. Every probe loop
terminates within the region width, and the bounds are stated in slot comparisons. **Joint reachable
witnesses** are required at the actual claimed types: fresh successful insert, replacement at
capacity, growth, `Sigma`-separation, and each refusal branch, each with a checked witness term, not
separate incompatible witnesses. Which of these witnesses the extraction can reach is currently
**MISSING**.

**5. Checked sizing, reserve-all, initialization, transaction, refusal.** Every region length, byte
count, offset and budget is checked arithmetic with an explicit overflow policy, consistent with the
existing `SizeOverflow`/`AllocationFailed` option (b) contract in the
[local boundary](../kernel-boundary.md#resourcerefusal-contract-for-the-growable-table). The only
admitted order is: locate; compute the complete checked plan; reserve **every** fallible allocation;
initialize vacant slots; apply or relocate without merge; invoke the count merge exactly once;
commit. The transaction point is the commit. On refusal the committed state is unchanged, no
partial table is observable, no merge and no clone of a payload has run, and build returns no table.
Relocation failure is proved unreachable for the checked layout, not aliased to
`AllocationFailed`.

**Frozen before construction.** The count carrier is `i64` (`Int64`, clause 3's `RepC`); the count
default is `0`; the overflow policy is **checked refusal, never wrapping** — a count overflow is a
refusal at the commit point on option (b)'s transactional terms, and a refused merge leaves the
committed state unchanged and writes nothing. The merge is `f(new, old) = new + old`, in the new-first
argument order of `Apeiron.Table.insertWith` (`insertWith f k v m = fun x => if x = k then f v (m x)
else m x`, `/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/Basic.lean:80-81`); because
`i64` addition is commutative the new-first order is not observable through the merge itself, and the
order clause 6 fixes is the input order of the build.

Checked rather than wrapping, for three stated reasons.

1. Ratified option (b) admits exactly `StorageError::SizeOverflow` for a checked layout and
   `StorageError::AllocationFailed` for a failed reservation
   (`docs/kernel-boundary.md:333-352`, variants at `src/table.rs:99-106`). A wrap is neither, and a
   silently wrong count is the silent domain reduction obligation 2 forbids.
2. `checked_add` returns an `Option`, the refusal shape the existing sizing path already uses —
   `checked_mul` → `SizeOverflow` (`src/table/state.rs:414,420,423`) and `try_reserve_exact` →
   `AllocationFailed` (`src/table/state.rs:515`) — so the transaction point does not move.
3. The behaviour being replaced is profile-dependent, not an overflow contract. `sum_parallel`
   reduces with `Iterator::sum()` over `i64` (`src/table/parallel.rs:980-982`), and that operator wraps
   silently in release and aborts in debug: `rustc -O` on `[i64::MAX, 1]` printed
   `sum=-9223372036854775808` and exited 0, and the same source without `-O` panicked `attempt to add
   with overflow` and exited 101, on `rustc 1.97.0-nightly (17584a181 2026-04-13)` with no
   `[profile]` override in `Cargo.toml`. A wrap is a wrong count and a panic is not a refusal; the
   deferred `sum_parallel` overflow task (`docs/roadmap.md:394`, `docs/kernel-boundary.md:80`) needs a
   contract, and this is it for the counting face. The `wrapping_mul` at `src/table/state.rs:379` (a
   tag fingerprint) and the `saturating_add` chain at `src/table/geteach.rs:167-174` (a
   non-authoritative footprint estimate) are neither sizing arithmetic nor a count and license
   nothing here.

The refusal type is `CountRefusal { SizeOverflow, AllocationFailed, CountOverflow }`, a total 1:1
widening of `StorageError` by `CountOverflow`. It lives in the counting module so `APEIRON-COUNT-W2-T1`
touches at most three files and `src/table.rs` is not edited for an error variant. The default is a
parameter, matching `Apeiron.Table.build`'s own default argument
(`build f d kvs = kvs.foldl (insertWith …) (fun _ => d)`,
`/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Table/Basic.lean:121-122`), frozen here at `0`:
`HMap K V = K → V` (`Basic.lean:63`) is a total function whose absence is the default, and `0` is the
additive identity of `Int64`, so `get` at a coordinate never written answers `0` and a build of an
empty stream is the constant-zero table.

This freeze is scoped to the counting container. It does not discharge the deferred `sum_parallel`
overflow task, which needs `sum_parallel`'s own source changed and is not owned by `APEIRON-COUNT-W2-T1`.

**6. Parallel ownership, per-key ordering, achieved workers, readback.** Ownership of a slot is
exclusive for the operation holding it and is fixed by the coordinate before any probe, so two
workers cannot own the same slot for one operation. Per-key merge order is the input order, and the
stable partition plus disjoint assembly supplies the exact premise of
`Apeiron.Table.build_eq_of_key_streams_eq`; same-key reordering is not licensed. The **achieved**
worker count is what the run reports and is compared against the requested width, not assumed from
it; a requested width with a lower achieved width is a measurement finding. Readback is in query
order and must reproduce the canonical `build` for the ordered stream.

**7. Extraction feasibility, theorem dependency graph, audits, trust boundary.** Real-source
extraction by `#[path]` wrappers that only specialize and call the production bodies; the
pinned-JoinBoundary and `try_reserve_exact` limits recorded above apply to the counting roots
unchanged. The theorem dependency graph is ordered: codec/identity → constructor establishment →
insertion/growth preservation → absence/probe bounds → read refinement to `Apeiron.Table.get` →
update refinement to `Apeiron.Table.insertWith` → ordered-build refinement to `Apeiron.Table.build` →
joint witnesses. Every headline is registered in `proof/corpus/AxiomAudit.lean` and builds under the
closed cone `{propext, Classical.choice, Quot.sound}`; the Slang empty-cone chain is separate and no
Rust/canonical-cone theorem enters it. Excluded: allocator/global-allocator realization,
`try_reserve_exact`, thread/executor realization including `std::thread::scope`, rustc/LLVM code
generation, the kernel ABI, GPU/SPIR-V and the `unsafe` container. Existing generic-`Table`
headlines prove nothing about this container.

**8. Oracle, populations and controls.** An independent **full-key** oracle — not the container's own
readback and not a hash — decides every population: sentinel and boundary coordinates, aliasing
pairs, `2^31`, `2^32-1`, the admitted-domain edge and the first refused coordinate, equal coordinates
at distinct `Sigma`, adversarial/HashDOS populations, and the serialized and packed counting
populations. The harness's existing limitation that `apeiron::Table` exposes no encode/decode
surface (`examples/reg1/workload.rs:161`, `examples/reg1/correctness.rs:78`,
`examples/reg1/coverage.rs:1022`) does not carry over: the counting face must supply the codec surface
the oracle needs, or the population is recorded **MISSING** coverage rather than passed.
**Attributable controls, in a disposable copy, never the working tree:** truncate the coordinate
identity, or alias two sentinel/boundary keys, and require the full-key oracle and the corresponding
refinement to reject, then pass restored; reservation-failure controls must detect any effect before
refusal, not only the refusal itself. Every control names the defective cell, the source identity and
the responsible arm — a `digest_fail` is not attributed to Apeiron by default, since the comparator or
the harness may own it.

**9. Measurement protocol.** Matched operations, populations, packing, partitioning, threads and
timed regions; the timed build and probe phases must be the phases the arm label names. Report the
mean over at least three runs with the observed spread; a movement inside the spread already measured
on unchanged code is not a change. Bind every figure to source and tool identity (sha256 of the
measured sources, the Aeneas/Charon/Lean/Mathlib revisions, the machine and the interval) and force a
rebuild or use a private target directory so a run never reads a sibling worktree's objects. Every
registered comparator participates: the three hashtrinity arms
(`hashtrinity_table`, `hashtrinity_count_par`, `hashtrinity_get_each`), the apeiron map arms
(`apeiron_table`, `apeiron_serial`, `apeiron_parallel`, `apeiron_tuple_par16`),
`hashbrown_foldhash`, `hashbrown_fxhash`, `hashbrown_rayon`, `std_siphash_par16`, `foldhash_par16`,
`fxhash_par16`, `xxHash`/`wyhash`/`ahash`/`fnv` on the opaque-key exact face, and the informational
cryptographic comparators that carry no ratio bound. Every applicable pair retains its bound.

**10. Tasks, owners, dependencies, gates, budgets.** See the plan below. Each task owns at most three
files, names one gate and one negative control, and every gate runs under an enforced wall-clock
budget strictly below 600 s; a run at or beyond 600 s fails the budget and is not a pass.

**Blocking rule.** **Implementation stays blocked until this contract names an executable gate and
that gate's rejecting control is established.** The contract-level gate is
`python3 tools/check-claims.py` (repository root, 60 s), scoped by `COVERED_SECTIONS`
(`tools/check-claims.py:110-112`) to this file's `## Coordinate-keyed counting-container contract`
heading through the next heading of the same or higher level. Its rejecting control is established:
`python3 tools/check-claims.py --self-test` (60 s) runs the fixture *nonexistent path cited in the epic
contract rejected* (`tools/check-claims.py:854-860`), which requires exit 1 with a finding naming
`docs/epics/storage-refinement.md` and the seeded path, and the fixture *renamed epic contract section
rejected* (`tools/check-claims.py:869-874`), which requires exit 2, so a renamed heading is a refusal
and not a silent pass. Both hold at grade
`tested` — `python3 tools/check-claims.py` at exit 0 with 2 graded claims and 0 findings on this
section, `python3 tools/check-claims.py --self-test` at exit 0 with 11/11 controls — measured on the
pinned corpus population at
`proof/corpus/.lake/packages/apeiron`; without that population the same gate reports 36 findings in
other governing documents and none in this file.

Whether that opens a build lane is the lead's decision and is not taken by this record. The
task-specific implementation gate of `APEIRON-COUNT-W2-T1` is now **named** — the `cargo test` and its
control in the plan row below — and its rejecting control is **not established**, because no counting
module exists to mutate. Per the blocking rule, implementation therefore stays blocked on exactly one
thing: the first observed red/restored pair for that row's coordinate-identity and sentinel-alias
mutations. Every other gate named here and not run is recorded
**MISSING**, never waived by a similar gate elsewhere.

**Owner target scope.** `REG1_TARGET_RATIO` is read at `examples/reg1/acceptance.rs:30`. The owner's
`half` value (`examples/reg1/acceptance.rs:34`, parsed at `examples/reg1/acceptance.rs:159`) applies
**`<= 0.5` to the three hashtrinity arms** `hashtrinity_table`, `hashtrinity_count_par` and
`hashtrinity_get_each` — `TARGET_SCOPED_INCUMBENTS` at `examples/reg1/acceptance.rs:43` — and
**`<= 1.0` (`RATIO_LIMIT`, `examples/reg1/acceptance.rs:20`) to every other incumbent**. The scope is
closed over the registry, not over a name prefix, so a newly registered hashtrinity arm is a scope
failure. Neither "`<= 0.5` everywhere" nor "default `1.0` everywhere" describes this target: the
registered gate with the variable unset stays `<= 1.0` on every applicable pair, and an unparsable
value refuses the invocation with `USAGE_EXIT` rather than silently running the registered gate. The
targeted invocation is

```sh
REG1_TARGET_RATIO=half cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

`concurrency_unmatched` (`examples/reg1/acceptance.rs:739`) and `address_face_unmatched`
(`examples/reg1/acceptance.rs:741`) must both be zero at acceptance, and `full_acceptance`
(`examples/reg1/acceptance.rs:754`) must be true.

### Plan

| Task | Owned files (≤3) | Gate (budget) | Negative control | Depends on |
|---|---|---|---|---|
| `APEIRON-COUNT-W1-T1` | `docs/epics/storage-refinement.md`, `tools/check-claims.py` | `python3 tools/check-claims.py` and `python3 tools/rooting/read_gate.py --check`, repository root (60 s each) | Citation to a nonexistent path inside this contract, added under a `tested` sentence: exit 1 naming `docs/epics/storage-refinement.md` and the line; removed: exit 0. **Coverage established**, scoped: `COVERED_SECTIONS` in `tools/check-claims.py` scans the `## Coordinate-keyed counting-container contract` heading of this file to the next heading of the same or higher level, reports file line numbers, and exits 2 if that heading is absent, so a renamed section is a refusal and not a silent pass. Two coverage residuals stay **open**: whole-file coverage of this epic, because the 35 pre-existing claims above this contract are superseded records and redden the gate at 48 findings re-measured with 37 graded claims whole-file against 2 in this section; and markdown link destinations, which are not citations to this gate. | REVIEW5 selection |
| `APEIRON-COUNT-W1-T2` | `examples/reg1/matrix.rs`, `examples/reg1/correctness.rs`, `examples/reg1/strata.rs` | the strict full-profile command above, bounded below 600 s per run with repetitions and spread reported | Seeded wrong-output cells; every one of the six `digest_fail` cells must go red then green on restore | exact failing cells reproduced |
| `APEIRON-COUNT-W1-T3` | `docs/kernel-boundary.md`, `docs/direction.md`, `docs/README.md` | `python3 tools/check-claims.py` and `python3 tools/rooting/check.py --check` (60 s each) | `python3 tools/rooting/check.py --self-test` rejection fixtures must reject | `APEIRON-COUNT-W1-T1` |
| `APEIRON-COUNT-W2-T1` | the counting module `src/table/count.rs`, its `include!` line in `src/table.rs`, and the one production consumer `Table::sum_parallel` in `src/table/parallel.rs` | `cargo test --locked --lib table::count::tests::` (300 s) — the container's own cells, run on the build path, filtered as `docs/kernel-boundary.md:76` filters `table::partition::tests::partition_integrity_` — plus `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` and `cargo fmt --all -- --check` (§6, 300 s each) | In a disposable copy under `/home/socol/oc-tmp`, never the working tree. (i) **Coordinate-identity mutation:** replace `CountKeyCodec::encode` with `u64::from(coord)`, dropping the face half; require `cargo test --locked --lib table::count::tests::` red with `sigma_separation_update_leaves_the_other_face_unchanged` and `counting_words_are_injective_across_faces` failing; restore, require green. (ii) **Sentinel alias:** write the vacant marker as the stored word `0`, which is `encode(Elliptic, 0)`; require `zero_word_is_a_live_coordinate_not_a_vacancy` red. The oracle is the independent full-key map over `(Sigma, u32)` in the container's `full_key_oracle_agrees_with_the_counting_container` cell — not the container's own readback and not a hash. A mutation that has never been observed red leaves this control **MISSING** and the row open. | clause 1 consumers and clause 2 codec frozen; clause 5 overflow policy, count carrier and default frozen |
| `APEIRON-COUNT-W2-T2` | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | `proof/regen.sh --check` (570 s) | Generated-body mutation in a throwaway copy, red then restored green | `APEIRON-COUNT-W2-T1` |
| `APEIRON-COUNT-W2-T3` | `proof/corpus/Counting/*.lean` and `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build` under `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`, `LEAN_NUM_THREADS=1`, `timeout 570`, `flock /tmp/opencode/apeiron-lane.lock` | Codec/identity mutation reddens the named headline; restored source green | `APEIRON-COUNT-W2-T2` |
| `APEIRON-COUNT-W3-T1` | the parallel executor and readback bodies (≤3 files) | the strict full-profile command with `--controls`, bounded below 600 s per run | Achievable-worker and query-order readback mutations red then green | `APEIRON-COUNT-W2-T1` |
| `APEIRON-COUNT-W4-T1` | `docs/plan.md`, `docs/roadmap.md` | `python3 tools/rooting/check.py --check` (60 s) | `python3 tools/rooting/check.py --controls` | `APEIRON-COUNT-W3-T1` |
| `APEIRON-COUNT-W5-T1` | no edit; acceptance run | `REG1_TARGET_RATIO=half cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`, bounded below 600 s per run, plus `proof/regen.sh --check` and bounded `cd proof/corpus && lake build` | the command's own baseline, red and restore controls | all of the above |

No task opens before its dependency closes, every wave is a barrier with its control run, and no task
owns a file another task owns. `docs/plan.md` records the order and the board rows; a row is created
only after this contract is accepted.

**Open obligations carried.** G1 is **OPEN**; the composition headline
`prepared_rebuild_refines_insertWith` for the generic `Table` stays open and is untouched here; the
generic `Table`'s `SearchDecisionCorrect`, `ResourceAdapterSimulates` and
`GrowableSchedule.partition_preserves_key_streams` premises stay open; `hashtrinity::CoordMap` and the
serial `FreqMap` coverage gap stays open; the six `digest_fail` cells stay open until
`APEIRON-COUNT-W1-T2` attributes them. Historical fixed-shard and single-shard results cannot
discharge any obligation above.
