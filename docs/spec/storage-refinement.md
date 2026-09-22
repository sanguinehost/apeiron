# Specification - Bounded Storage Refinement

- **Owner:** OpenCode
- **Intent:** `docs/intent/storage-refinement.md`
- **Source revision:** `f3cac24df6263a3c64eb125064cb7b1cb9b73fcc`; working tree clean before this epic
- **Status:** approved

> **Record status.** The two-container content below (`SafeTable<K, V, N>`, `Map<K, V>`, the
> fixed-capacity-four reference) records retired source. The executing representation is the growable
> flat/spill `Table<K, V, H, N, R>` in `src/table.rs`, with `Stored` and `Regen` carrier faces. The
> fixed-shard `Table<K, V, H, N>` and its proofs are revision-bound historical evidence. Face/key
> semantics, layer separation and cost boundaries are stated here; historical rows remain labelled.

## Face and key semantics

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

## Three layers

- **Canonical semantics.** `Apeiron.Table.HMap`, `get`, `insertWith`, `build`: the shared total map,
  its new-first write and its ordered build fold.
- **Physical fixed-shard invariant.** The inline `[Option<[Option<(K, V)>; N]>; 256]` state with
  `Rep`: one home shard per key, no fallback probing, `ShardFull`/`N = 0` refusal. A physical state
  denotes the canonical map under `Rep`; `Rep` is the private invariant, not the public contract.
- **Source-checker memo store.** The source checker's memo store is a separate store with its own
  refinement obligation.

A result at one layer does not transfer to another without its own refinement.

## Fixed-N cost boundary

`N` is a compile-time constant, callbacks are bounded, and key length is not constant. Lookup and
update are bounded per operation at fixed `N` with bounded callbacks and nonconstant key length.
Reading a whole source remains at least linear in its bytes at fixed `N`, so constant-time table
operations at fixed `N` do not imply constant-time whole-source checking or constant-size proof
terms. No performance measurement is claimed here.

## Revised production contract: reworked growable key-partitioned table

Recorded by `APEIRON-PERF-RESTORE-W4-T1`. This contract replaces the bounded fixed-shard contract
above. The growable flat/spill `Table<K,V,H,N,R>` is the executing implementation; its extraction and
refinement integration wave remains open. The fixed-shard `Table<K,V,H,N>` is revision-bound
historical evidence. No extraction, refinement or measurement acceptance is claimed here. Trust, pin
transport and the ratified resource/refusal contract are in the
[local boundary](../kernel-boundary.md#resourcerefusal-contract-for-the-growable-table); the dispatch
order is in the [plan](../plan/storage-refinement.md).

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

### Flat-resident and grouped-spill contract (Sol decision)

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

## Extraction-shaped representation contract (Astra, APEIRON-EXTRACT-REP)

Recorded by `APEIRON-EXTRACT-REP-W1-T1`; documentation only, `anchor`, no source, extraction, proof or
performance claim. This contract supersedes the `KeyRep`/`Held` carrier and `rebuild_resident` wording
above where they differ. The boundary is in the
[local boundary](../kernel-boundary.md#extraction-shaped-representation-boundary-astra-apeiron-extract-rep);
the dispatch order is in the [plan](../plan/storage-refinement.md#extraction-shaped-representation-dispatch-order-apeiron-extract-rep).

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

### Prepared-plan resource interface (Sol decision, APEIRON-EXTRACT-REP-W5-T1)

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

### Relocation entry interface (Sol decision, APEIRON-EXTRACT-REP-RELOCATION-CONTRACT-T1)

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

### Pending apply/commit extraction decision (Sol, APEIRON-EXTRACT-REP-RELOCATION-CONTRACT-T4)

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

### Search decision/apply interface (Sol decision, APEIRON-EXTRACT-REP-W11-T2)

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
that a prefix vacancy does not truncate a late match in the mandatory sweep. The bounded Kani harness
`search_decision_classifies_a_bounded_two_lane_spill` is present in `src/proofs.rs` with an explicit
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

### Partition and spill-filter contract (Sol decisions, APEIRON-PERF-CONTRACT-T1/T2)

**Decision and grade.** This section ratifies the bounded source contract requested by the current
partition/validation performance cut. It is grade `anchor`: no source, extraction, theorem, test or
measurement is accepted by this text. The transactional refusal contract, fail-closed generated audit,
both carrier faces and `W7-SearchCommitBoundary` remain unchanged.

#### Partition validator

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

##### Extraction and bridge impact

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

##### Flash extraction edit and order

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

#### Summary resource and publication lifecycle

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

#### Exact spill-filter invariants

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

#### Bridge impact

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

#### Serialized implementation and proof order

Each row is a separate task and file owner. Shared extraction and Lean lanes are serialized. A
mutation runs in a throwaway copy and the unchanged source is rerun; no mutation edits this working
tree.

| Order / owner | Files (at most three) | Single gate | Rejecting control |
|---|---|---|---|
| Flash F1 | `proof/extract/lib.rs`, `proof/regen.sh`, generated `proof/ApeironWord.lean` | `proof/regen.sh --check` from repository root after one successful regeneration; extraction audit and the package build must exit 0 | In a throwaway copy, restore the removed validator wrapper and selector: Aeneas must reproduce the context-match failure; discard it and rerun the retargeted gate green. |
| Flash F2 | `src/table.rs`, `src/table/resources.rs` | `cargo test --locked --lib table` | Independently omit a live-write `summary_set` and omit the summary capacity bound; the actual-path recordedness and injected summary-reservation-refusal tests must fail, then pass restored. |
| Flash F3 | `src/table/parallel.rs`, `tests/parallel.rs` | `cargo test --locked --test parallel` | Move one shard reservation behind worker start in a throwaway copy; injected final-shard refusal with counted hold/clone/merge must fail the zero-effects assertion, then pass restored. A deterministic reservation-failure seam is **MISSING** and must be added by this gate-first task before the ordering claim can close. |
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

#### Extraction primitive and reserve-all disposition (`APEIRON-PERF-CONTRACT-T3`)

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

##### F4 roots and transactionality boundary

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

##### Invariant and search consequence

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

##### Bounded Flash edits and regeneration order

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

## Read-path specialization reconciliation

The lifecycle artifacts
[`intent`](../intent/read-path-specialization.md),
[`spec`](read-path-specialization.md) and
[`plan`](../plan/read-path-specialization.md) govern the authorized read-only specialization. This
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

### Parallel stable-scatter reconciliation

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
in `tests/parallel.rs` across worker counts, the reserve-all refusal control, a rerun of
`examples/micro_spatial.rs`, and the unchanged strict full-profile command. The supplied 46–55%
kernel and about 28% partition-plus-validation split remains an anchor until rerun. Every spatial pair
must still satisfy interval maximum `selected/incumbent <= 1.0`; missing, invalid, inconclusive,
concurrency-unmatched or address-face-unmatched pairs reject. G1/REG-1 remains **OPEN**.

## Historical record (superseded): two-container bounded reference

The sections below describe the retired `SafeTable`/`Map` reference. They are historical evidence,
not the current table.

## Scope and admissible domain

| Item | Value |
|---|---|
| Operation | Safe fixed-capacity lookup, replace and removal |
| Domain | `K: Eq`, arbitrary `V`, compile-time slot count `N` |
| Totality | Lookup/removal total; insertion returns `Result` on full absent-key insertion |
| Preconditions | None; allocation failure is absent because storage is an inline array |
| Postcondition | Successful observations equal a total function `K -> Option V` |

The first executable instance uses word keys and values at a small fixed capacity for extraction.
This bounds proof cost without narrowing the generic Rust reference's domain.

## Representation relation

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

## Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| Lookup | `get k` returns the unique stored value or `None`. | proved | Safe-table lookup refinement. |
| Replacement | Existing `k` changes to `v` and returns the old value. | proved | Canonical `insertWith replace`. |
| Fresh insertion | Absent `k` stores `v` when a slot is free and returns no old value. | proved | Canonical `insertWith replace`. |
| Full refusal | Absent `k` with `len = N` returns the original key/value and no successor. | proved | Extracted branch characterization and witness. |
| Removal | Present `k` becomes absent and returns its value; other keys are unchanged. | proved | Canonical replacement by `None`. |
| Production agreement | `Map` matches the safe reference after every selected operation. | tested | Differential trace test. |

## Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Fresh key with free slot | `Ok(None)` and committed successor | `len < N` |
| Existing key at any occupancy | `Ok(Some(old))` and committed successor | key present |
| Fresh key at capacity | `Err(Full { key, value })` | key absent and `len = N` |
| Missing removal | `None`, state unchanged | key absent |

The optimized `Map::insert` remains allocation-backed and infallible except process allocation
failure. Its comparison uses a pre-reserved capacity and does not equate allocation failure with the
reference's logical full result.

## Evidence plan

1. Add focused nominal, full, replacement-at-full, removal and trace tests for the safe reference.
2. Compare production `Map` with the reference after every admitted trace operation.
3. Extract only monomorphic safe operations after the Rust API and tests are fixed.
4. Prove denotation equalities against the pinned corpus and audit every headline.
5. Mutate full handling, replacement order and removal in throwaway copies; each unchanged gate must
   fail and the original must pass.

## Trust boundary

Proof of the safe transition system does not transfer to optimized storage. Until its representation
relation is universal, the production comparison grade is tested.

## Open questions

- The optimized relation may require Kani over bounded layouts, a proof-friendly probe extraction or
  a redesign that shares a safe transition core. Select only after measuring the reference boundary.
