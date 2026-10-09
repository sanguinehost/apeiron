# apeiron

Kani is retired from this repository's workflow. Use Rust tests/clippy/fmt, real-source mutation
controls and the Aeneas/Lean gates in [AGENTS.md](AGENTS.md#6--local-commands). Historical Kani
receipts are not current acceptance and no missing proof is discharged by retirement.

`R[x]/(x^2 - sigma)`: one two-slot algebra over a commutative ring, and one corpus-specified table
whose executing representation is the growable flat/spill `Table`; its earlier fixed-shard form is
revision-bound historical evidence.

Apeiron owns the canonical definitions, the reusable Rust primitives and their refinements. The
programme thesis, ownership, outcomes, primitive correspondence, migration and execution order are
in the [programme direction](docs/direction.md); that document is the programme authority. This
README is the local repository description: it states the crate's actual behavior and links to the
[local boundary](docs/kernel-boundary.md) and the [proof evidence](proof/README.md). It makes no
current benchmark or performance claim.

The canonical mathematics is Lean, physically at `../sanguine-apeiron/proof/Apeiron/` with unchanged
pins; theorem names in the source doc comments resolve there. Rust representation theorems live in
`proof/corpus/`.

## Purpose

Apeiron is the O(1)-primitive basis at fixed N with bounded callbacks and nonconstant key length,
evolved from hashtrinity; `fold/hashtrinity` is the naive predecessor. The basis is
`R[x]/(x^2 - sigma)`, `Cell<R>` with `Sigma` as a value in every executing signature, `Coord`
addressing correct per face, and the executing growable `Table`: a flat certified-coordinate resident
area plus grouped spill regions, keyed by `(Sigma, K)` with `Stored`/`Regen` carriers. The historical
fixed-shard `Table<K, V, H, N>` (one home shard, `ShardFull` refusal, `N = 0` refusal, no probing)
and its bounded-admission proofs are revision-bound historical evidence, not proofs of the executing
layout. Every executing face refines the canonical object; anything not rooted in it is not accepted. The canonical object is
`Apeiron.Table.{HMap,get,insertWith,build}` and
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` with the extracted `Cell<i64>` word
bodies (`word_mul_denote`, `word_norm_denote`).

The O(1) qualifier is the parameter boundary. Fixed N, bounded callbacks and nonconstant key length
bound the table; reading a whole source remains at least linear in its bytes at fixed N, so
constant-time table operations at fixed N do not imply constant-time whole-source checking or
constant-size proof terms.

## The scalar is a choice, not a limit

`Cell<R>` is generic over the ring its slots live in, because which ring you pick decides what the
cell can hold:

- **`i64`** -- the fast lattice. Its operations are `Z/2^64`, and what holds there is the
  *structural* half: commutativity, associativity, the unit, multiplicativity of the norm,
  componentwise multiplication in the null basis, order-free folding. The **classification does
  not transfer**: `Z/2^64` has only the two idempotents `0`/`1`; it has a nonzero nilpotent at
  *every* dial, so it is a zero divisor on the elliptic face the real ring calls a field; and it
  has no `Discards` projection at any dial. Naming a word object by the real trichotomy's
  character is an analogy, not a theorem. `tests/the_dial_is_real.rs` tests the part that does
  hold on its concrete inputs; the canonical word-versus-real boundary is stated at
  `Trichotomy/AnyRing.lean:13-46`.
- **`Dyadic`** -- `m / 2^e`, two integers, the same shape a float has without the rounding while the
  result fits the fixed mantissa and scale. This is where the nontrivial idempotents `(1 +/- j)/2`
  live, so `exists_nontrivial_idempotent_iff_disc_pos` holds here rather than being written off. It
  is not a total ring: `checked_*` returns `None` on overflow and the `Ring` operations panic rather
  than silently dropping bits.

The halves were never an obstruction in the algebra, only in the container. In the null basis the
idempotents are the unit vectors `(1, 0)` and `(0, 1)` -- the fraction is an artefact of the
coordinates.

The split is in the bounds. `Ring` is what the algebra needs; `Halve` is the extra thing the
idempotents need; `i64` implements the first and not the second, so `Cell::<i64>::idem_plus()`
does not compile. That is the theorem's scope enforced by the type rather than described in a
comment.

## The cell

`Cell<R>` is an ordered pair `(a, b)` over `R: Ring`. `Cell::mul(self, sigma, other) -> Self` is
`(a1 a2 + sigma b1 b2, a1 b2 + b1 a2)`; the canonical `mul` is commutative and associative on every
face, which licenses order-free folding (`fold`, `fold_is_order_free`), and the extracted `i64` body
is refined to it (`word_mul_denote`). The `Ring` trait itself promises no laws: a partial scalar such
as `Dyadic` panics on an unrepresentable operation, so those laws hold only where the operations are
defined. `add` is componentwise and `norm_form` is `a^2 - sigma b^2`.

`Sigma` is the dial: `Elliptic` (`-1`), `Parabolic` (`0`), `Hyperbolic` (`1`), with `value()`,
`channels()` (`0`, `1`, `2`) and `discards()` (hyperbolic only). Channel accessors are partial by
face: `upper`/`lower` return `Some` only on `Hyperbolic`, `channel` only on `Parabolic`; the
idempotent constructions `idem_plus`, `idem_minus` and `of_halves` require `R: Halve`.

## The table

The executing representation is the growable `Table<K, V, H, const N: usize = 16, R = Stored<K>>`
(`src/table.rs`): a flat certified-coordinate resident area plus grouped spill regions, keyed by
`(Sigma, K)` with `Stored`/`Regen` carriers. `H: Home<K>` selects placement and an optional certified
coordinate (`WordHome` uses the word bit pattern) and `F: Merge<V>` combines new-first (`WordMerge` is
wrapping addition). `get`/`insert_with`/`build` use the default face; `get_at`/`insert_with_at`/
`build_at` take `Sigma` explicitly. Generic `Home`/`Eq`/`Clone`/`Merge`/drop operations may allocate,
panic or diverge.

The historical fixed-shard `Table<K, V, H, const N: usize = 16>` stored an inline
`[Option<[Option<(K, V)>; N]>; 256]` — 256 fixed shards, each an optional length-`N` array, with no
`Vec` — and its `Rep` invariant and bounded-admission proofs are revision-bound historical evidence.
Stack availability for that inline object was not proved.

Public API:

| method | behavior |
|---|---|
| `new(default, home)` | empty table; every key reads the default |
| `get(&K) -> &V` | the stored override, or the default; total for every key |
| `insert_with(&F, k, v) -> Result<(), StorageError>` | applies `merge(v, old)`, including the default on the first write |
| `build(&F, default, kvs, home) -> Result<Self, StorageError>` | applies each key's stream in order; no partial table on refusal |

Admission and refusal are explicit. The growable table returns
`Err(StorageError::SizeOverflow)` on a checked-sizing failure and
`Err(StorageError::AllocationFailed)` when reservation fails; `Err` changes no committed field and
runs no clone or merge, and `build` returns no partial table. `StorageError::ShardFull` is a
compatibility variant the growable table does not return. The historical fixed-shard table refused
with `ShardFull` exactly when the key was absent and its home shard had no vacant slot; refusal
invoked neither clone nor merge and changed no field, while an existing key still updated in a full
shard, and `N = 0` kept reads total and refused every write. `tests/table.rs` covers the total
default read, new-first replacement, per-key stream order, checked group-width refusal with an
unchanged table, growth past the historical fixed ceiling, `Sigma` separation and the
`Stored`/`Regen` round trips.

The historical fixed-shard table is tested and refined at its revision. The generated generic
refinement covers arbitrary `Table<K,V,H,N>` instances
whose equality, clone, home and merge callbacks satisfy `CallbackLawsG`; the laws state their total,
stable denotations rather than assuming them from Rust trait names. For the concrete `I64` word instance (`EqI64`, `CloneI64`,
`WordHome`, wrapping-add `WordMerge`) the real constructor establishes `PhysicalInv` at every `N`;
`get` reads the represented total map; `insert_with` preserves `PhysicalInv` and canonical
`insertWith` on success and returns the identical representative on `ShardFull`, with exact refusal at
`N = 0`. For positive `N`, `BuildAdmissibleN` exactly characterizes build success versus `ShardFull`,
and success establishes canonical ordered `ReadRep` and `PhysicalInv` through
`table_build_physicalInv_of_buildAdmissibleN`. The zero-capacity build is characterized exactly by
`wordBuildN_zero_of_empty` and `wordBuildN_zero_refuses_of_nonempty`
(`wordBuildN_zero_ok_iff_empty`). Generic insertion preserves `PhysicalInvG` under
`CallbackInsertPostG`; admitted builds preserve `PhysicalInvG`, with exact zero-capacity behavior
and an all-`N` sufficient routed-key admission theorem. The joint replacement-map profile remains the concrete `N = 16`
instance. These results are revision-bound historical evidence about the fixed-shard layout, not
proofs of the executing growable table. Evidence and the open obligations are in
[proof/README.md](proof/README.md) and the [local boundary](docs/kernel-boundary.md). No current
table benchmark or performance claim is made; the removed `Map`/`CountTable`/`Set` measurements and
their benches concerned source that no longer exists, and the deletions are recorded at the
[local boundary](docs/kernel-boundary.md).

## What this does not claim

- No audited cryptography. `aes`, `chacha`, `sha256` and `crypto` are unaudited, tested against
  known-answer vectors only, and not side-channel resistant (`aes` indexes its S-box by secret
  bytes). Do not use them to protect real data. Their rooting onto `R[x]/(x² − σ)` is in progress.
- No current benchmark or performance number for the table or the crate.
- No semantics for callbacks that violate `CallbackLawsG`, including callbacks that allocate,
  panic or diverge. The proved generic refinement is conditional on those explicit callback laws and
  its stated insertion/build admission predicates.
- No claim that the wrapping `i64` ring carries the real trichotomy's classification, the real
  energy/nonnegativity results, or an order.
- No claim about rustc/LLVM code generation, the kernel ABI, a GPU backend, or the generic
  `Home`/`Eq`/`Clone`/`Merge`/drop callbacks, which may allocate, panic or diverge.
- A theorem name, a sibling path or a citation is a pointer, not evidence; it is `proved` only when
  the named theorem exists and the in-repo link is checked.
