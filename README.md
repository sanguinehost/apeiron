# apeiron

`R[x]/(x^2 - sigma)`: one two-slot algebra over a commutative ring, and one corpus-specified
fixed-shard table.

Apeiron owns the canonical definitions, the reusable Rust primitives and their refinements. The
programme thesis, ownership, outcomes, primitive correspondence, migration and execution order are
in the [programme direction](the design notes); that document is the programme authority. This
README is the local repository description: it states the crate's actual behavior and links to the
[local boundary](the design notes) and the [proof evidence](proof/README.md). It makes no
current benchmark or performance claim.

The canonical mathematics is Lean, physically at `../sanguine-apeiron/proof/Apeiron/` with unchanged
pins; theorem names in the source doc comments resolve there. Rust representation theorems live in
`proof/corpus/`.

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
  hold and `src/proofs.rs` boundedly checks the part that does not, the same boundary the corpus
  states at `Trichotomy/AnyRing.lean:13-46`.
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

`Table<K, V, H, const N: usize = 16>` (`src/table.rs`) is the sole storage type. Storage is inline:
an outer `[Option<[Option<(K, V)>; N]>; 256]`, 256 fixed shards each an optional length-`N` array.
An outer `None` is an untouched empty shard; the table owns no `Vec` or heap-backed storage. Generic
`Home`/`Eq`/`Clone`/`Merge`/drop operations may allocate, panic or diverge, and stack availability
for the inline object is not proved.
`H: Home<K>` selects placement (`WordHome` uses the word bit pattern) and `F: Merge<V>` combines
new-first (`WordMerge` is wrapping addition).

Public API:

| method | behavior |
|---|---|
| `new(default, home)` | empty table; every key reads the default |
| `get(&K) -> &V` | the stored override, or the default; total for every key |
| `insert_with(&F, k, v) -> Result<(), StorageError>` | applies `merge(v, old)`, including the default on the first write |
| `build(&F, default, kvs, home) -> Result<Self, StorageError>` | applies the stream in order; no partial table on refusal |

Admission and refusal are explicit. `insert_with` returns `Err(StorageError::ShardFull)` exactly
when the key is absent and its home shard has no vacant slot; refusal invokes neither clone nor merge
and changes no field, while an existing key still updates in a full shard. `N = 0` is valid: reads
return the default and every insertion refuses. `build` returns no partial table on refusal and
preserves per-key stream order. `tests/table.rs` covers the total default read, new-first
replacement and build order; the `src/table.rs` unit tests cover full-shard refusal with an unchanged
table, existing-key update, `N = 0`, and that refusal does not invoke merge.

The table is tested, and named word-instance witnesses refine scoped behavior of the extracted
source. Universal physical `Rep` preservation and admission completeness remain pending. The current
word-instance witnesses and the open universal `Rep` obligation are in
[proof/README.md](proof/README.md) and the [local boundary](the design notes). No current
table benchmark or performance claim is made; the removed `Map`/`CountTable`/`Set` measurements and
their benches concerned source that no longer exists, and the deletions are recorded at the
[local boundary](the design notes).

## What this does not claim

- No current benchmark or performance number for the table or the crate.
- No universal `Rep` preservation, admission completeness or physical subsumption for arbitrary
  fixed-shard states, keys and streams; the checked table witnesses are scoped to the word instance.
- No claim that the wrapping `i64` ring carries the real trichotomy's classification, the real
  energy/nonnegativity results, or an order.
- No claim about rustc/LLVM code generation, the kernel ABI, a GPU backend, or the generic
  `Home`/`Eq`/`Clone`/`Merge`/drop callbacks, which may allocate, panic or diverge.
- A theorem name, a sibling path or a citation is a pointer, not evidence; it is `proved` only when
  the named theorem exists and the in-repo link is checked.
