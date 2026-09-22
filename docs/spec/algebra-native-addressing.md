# Specification — Algebra-native addressing and channel decomposition

- **Owner:** Astra architecture; Sol mathematical/refinement ratification; lead admission/closure
- **Intent:** `docs/intent/algebra-native-addressing.md`, approved per-face direction
- **Base:** `c450c222905c050f793a23332708c230c658f82c`; dirty supplied checkout
- **Extraction decision:** `APEIRON-ANATIVE-AEN-PIPE`; explicit monomorphizing candidate plus
  mutable-reference-free aggregate wiring; real-source acceptance OPEN, one production implementation.
- **Decision:** `APEIRON-ANATIVE-W4-A4`; lead selects the explicit-capability public API break,
  not preservation of opaque `Eq`/arbitrary placement closures. Both-face wiring precedes held
  summary deletion; source-linked adapter readiness remains open. Architecture
  grade `anchor`. Earlier prospective API paragraphs retain their founding scope; the standalone
  address/locator implementation is not table integration evidence.

## Extraction pipeline decision — APEIRON-ANATIVE-AEN-PIPE

**Select explicit Charon monomorphization AND remove mutable references from translated aggregate
types.** Grade `anchor`, not a green extraction receipt. This supersedes AEN-DIR's next dispatch,
old loop-error status and prohibition on selecting a configuration change; its scalar arenas,
typed faces, coverage, reservation, summary-deletion hold and W7 freeze remain. The assignment
reports that scalar-arena conversion clears directory errors. The remaining supplied failures are
`Could not find: type_var_id: 0/2 from ExtractBase.Item` at `reserve_rebuild`'s generic region loop
and `materialize_rebuild`'s generic initialization loop, and `SymbolicToPure.translate_type_decls`
CFailure on `Option`/`Result`/`ControlFlow` containing mutable references. These are tool failures,
not counterexamples to the contracts. No heavy build runs in this decision.

### Exact configuration and semantics delta

Replace **only** `--preset aeneas` in `proof/regen.sh::run_extract` with this candidate:

```text
--lift-associated-types '*' \
--treat-box-as-builtin \
--reconstruct-fallible-operations \
--reconstruct-asserts \
--reconstruct-matches \
--hide-marker-traits \
--hide-allocator \
--remove-unused-self-clauses \
--remove-adt-clauses \
--unbind-item-vars \
--deallocate-all-locals \
--monomorphize \
--monomorphize-mut=all
```

Keep every existing `--start-from`, `--dest-file`, rustc argument and real-source inclusion; keep
the Aeneas invocation `-backend lean -dest ... -warnings-as-errors -no-progress-bar`. Do not combine
the preset with these options. At Charon `b104e24fea7d721b71e6c39fd70f26ff20bc0980`,
`charon/src/options.rs:482-497` sets fourteen preset fields. Eleven remain above; the three omitted
flags are `--no-gen-tuple-structs`, `--ops-to-function-calls`, `--index-to-function-calls`.
`validate():572-582` forbids those with full monomorphization. CLI validation precedes driver
`apply_preset`, so apparent acceptance of `--preset aeneas --monomorphize` is not compatibility.
Quote `'*'` to prevent shell expansion. No tool revision, opacity policy or warning change is selected.

| Change | Extraction-semantics delta and required check |
|---|---|
| Omit `no_gen_tuple_structs` | Generate actual tuple declarations instead of the special opaque UNIT declaration with field types carried as generic arguments. Check tuple field/order correspondence and generated names; old Lean definitional equalities are not assumed. |
| Omit `ops_to_function_calls` | Array-to-slice unsizing and repeat remain LLBC operations rather than that pass's standard-library calls. This flag is not a general arithmetic lowering switch. Check all reached operations against pinned Aeneas support, including the existing Repeat patch. |
| Omit `index_to_function_calls` | Array/slice indexing remains LLBC indexing rather than synthetic standard-library calls. Preserve bounds/panic behavior and mutable updates in translation; unsupported indexing blocks the candidate. Vec trait calls are not all erased by this switch. |
| Add `monomorphize` | Specialize the transitive closure of concrete roots, resolving R/V/const and trait instances before Aeneas extracts generic loop bodies. Generic crate items without an instantiated root are skipped; coverage must be enumerated rather than inferred from file inclusion. |
| Add `monomorphize-mut=all` | Partially specialize items, including type declarations, according to mutable-borrow shape. This exposes references in fields; it does **not** remove them or prove Aeneas supports those declarations. It is not a replacement for full R/V specialization or the source constraints below. |

These changes select a different entry encoding of the **same Rust**, not a proof that either
encoding preserves semantics. Full-width words, checked overflow, bounds failures, successful and
refusing branches, ownership/update results, callback order and canonical observations must survive.
Tuple/name changes and specialized dictionaries can invalidate existing bridge scripts. No
`axiom`, `sorry`, `admit`, opaque required body, unknown external, partial output, disabled warning,
`--no-typecheck` or reduced root inventory is an acceptable repair.

The local source read is
`/tmp/opencode/charon-b104e24fea7d721b71e6c39fd70f26ff20bc0980/charon/src/options.rs`,
with `bin/charon/main.rs:225-246`, `bin/charon-driver/driver.rs:325` and
`transform/normalize/partial_monomorphization.rs:1-12`. A lightweight installed-binary probe using
the exact options and rustc `--version` accepts the option syntax and reaches rustc
`1.100.0-nightly (8fa1c96cf 2026-08-17)`, but exits **2**, `ERROR Code failed to compile` (no
translation unit/artifact). Adding either incompatible tuple or indexing flag exits **1** in
validation with the documented incompatibility. This tests `--monomorphize-mut=all` parsing only;
its real-source pass behavior is UNTESTED. Do not report the probe as a green pipeline or the
README's historical Rust pin as the version observed here.

### Mutable-reference-free aggregate wiring

“Free” refers to **type declarations containing mutable references**, not a ban on Rust `&mut self`
or direct mutable function arguments. Required concrete mutation helpers may take `&mut Directory`
or `&mut Vec<T>` and return owned/scalar results. Inspect the entire reached LLBC declaration and
signature closure, including compiler-generated standard-library and closure types, not just public
Rust signatures or a source-text search.

- In `src/table/locator.rs::reserve_bytes`, replace
  `directory.as_mut().ok_or(...)?` by direct matching on the owned `Option<Directory>` field through
  a local reborrow. Branch-local calls return owned `Result<LocatorReservationReceipt, StorageError>`;
  no helper returns a reference. Preserve None's exact refusal, staging and receipt checks.
- In `publish_for_query`, directly match/reborrow the field in each face branch, call the same
  `Directory::link`, and end the borrow before recording the reverse link. Do not use
  `Option::as_mut`, `get_or_insert`, `Result<&mut Directory, _>` or `?` on a borrowed success result.
  Preserve Elliptic's locator-free behavior and the profile guard; admitted publication must have
  the reserved directory and valid profile, not silently skip a link and still claim coverage.
- In `set_link_location` and `retarget_held`, use copied integer indexes, explicit `at < len`, a
  copied old `(identity_node, location)`, and an indexed whole-pair write. No `get_mut` result,
  `iter_mut`, mutable-reference tuple or reference-capturing closure. Missing handles remain no-ops;
  retarget changes only a link naming `old`; `old == new` remains a no-op. Bounds guards precede writes.
- Apply this discipline to the reached wiring in `src/table.rs`, including reserve/materialize
  calls: scalar loop state, short direct reborrows, owned results, no borrow-bearing aggregate
  crossing a call/loop join. Ordinary `Result<T, StorageError>` and `?` with **owned** T need not be
  banned; reject `ControlFlow<Result<Infallible, StorageError>, &mut T>` and recursively nested
  mutable-reference payloads wherever the compiler creates them. Direct pattern matching is a
  selected source shape, not accepted until the generated closure is checked.
- Keep `Option<Directory>`'s allocation-free empty construction; do not clone/take and rebuild an
  arena to avoid a borrow, change to unsafe pointers, split a proof-only directory, duplicate keys/
  values, introduce a scan fallback or remove mutation roots. Shared borrowed public reads remain.

Preserve **every current field** of `NativeLocator.TableStorageCoverage`: `arenaShape`,
`uniqueDecodedKeys`, `everyLiveRepresentativeReachable`, `completePathsUseScalarArenas`,
`completePathSound`, `reverseLinksSound`. The old four-field description predates the scalar-arena
extension. Cover default-valued entries, both carriers, distinct faces, constructor, found/fresh
insert, prepared build, growth, relocation and assembly/rebase; filtered coverage alone is not
Elliptic exact-address coverage. `Unknown` remains neither absence nor vacancy. The independent
fingerprint and full `(Sigma,K)` comparison remain required.

### Generic bodies and reservation contract

Use **pipeline specialization**, not handwritten concrete copies of `reserve_rebuild` or
`materialize_rebuild`. Existing thin `WordStoredTable`/`WordRegenTable` wrappers already fix
K=V=i64, H=WordHome, N=16 and R=Stored<i64>/Regen<I64WordCodec>, but the old pipeline still extracts
generic definitions. The selected pipeline must produce fully substituted versions of the real
two loops and their callees for both carriers, with no unresolved R/V binder. Concrete wrappers
alone under the unchanged preset do not solve the supplied failure. Mutable-reference reshaping
alone does not address that independent generic-binder error, so reshape-only is not selected.

Keep reserve-all → observe/check capacities → reserve/check locator path → hold/materialize →
publish/relocate → pending merge once → commit. All eight locator receipt categories and real
observed capacities survive; late refusal releases scratch/restores staged logical lengths,
stutters on represented committed state, performs zero hold/clone/merge/relocation and returns no
partial table. Raw allocator capacity equality is not required. Materialization cannot move before
the last fallible reservation. Preserve the current `ReserveBeforeRetainContract` committed/scratch
projection and require Sol to relate its phase list to actual source staging, not assume the result.

Full monomorphization covers only enumerated instances. **It does not establish arbitrary K/V/H/R/N
or arbitrary-merge refinement.** Keep the Rust generic API and every existing generic theorem's
quantifiers/domain. Sol must map current theorem consumers to generated specializations and identify
the remaining parametric source-refinement route; deleting generic claims/roots or replacing them
by N=16 word claims is not re-baselining. An unsupported allocator/model primitive or generic proof
transport blocks acceptance and requires a separately scoped pinned-tool/model repair, not a second
implementation or an enlarged trusted source boundary. “Everything provable” remains the goal;
neither this selection nor a green extractor establishes it.

### Re-baseline protocol

1. Release source/cache owners; freeze all included source, wrapper, script, tool/backend/patch,
   Lean and Mathlib identities. Record old generated digest and root-to-instance inventory. Compare
   identities against actual binaries; this probe's rustc differs from the historical README pin.
2. In the serialized extraction task, apply the selected wiring/configuration, retain all roots and
   inspect LLBC for both concrete loops, reference-free aggregate declarations and supported native
   index/repeat/unsize. Run `proof/regen.sh`; stop on the first unsupported construct. Never hand-edit
   generated bodies or overwrite a last accepted baseline with a partial translation.
3. Review the **entire** generated declaration/root/dictionary delta, then run
   `proof/regen.sh --check`. Normal regeneration and freshness must use identical options. This
   establishes reproducibility, not canonical refinement. A generated-body drift control must fail
   freshness and the unchanged restored copy pass.
4. Transport all affected bridge consumers without weaker statements, in ≤3-file Sol tasks. Run
   bounded `lake build AxiomAudit` and full `lake build` in `proof/corpus`; `proof/` also builds.
   Every registered headline stays fail-closed in `{propext, Classical.choice, Quot.sound}`; the
   Slang empty cone is untouched. Run actual-source/generated semantic mutations against the named
   theorem, not merely freshness. Retain old evidence as revision-bound; attach new claims only to
   the new source/tool/generated identity after these checks. An already-red bridge is still open.
5. Integration still requires the complete Rust/Kani/proof/control barrier. No source coverage,
   end-to-end table correctness, HashDOS bound or speed claim follows from extraction alone.
   Summary deletion remains held; W7's original reopening condition is unchanged.

## Historical directory extraction decision — APEIRON-ANATIVE-AEN-DIR

**Decision: one production directory, scalar arenas; translation feasibility OPEN.** This section
supersedes the physical directory shape and next extraction dispatch in A4, not its typed face API,
coverage, reservation contract, deletion hold or W7 freeze. Grade `anchor`. No build is authorized
in this documentation task. No supplied successful extraction establishes that the replacement
below is accepted by the pinned tools; calling it an accepted Aeneas shape would invent evidence.
The plan makes conversion plus real-source extraction the next bounded feasibility wave, before
further integration. Everything executing remains a refinement target, not a tested-only escape.

### Failure boundary and selected shape

The supplied Charon `0.1.254` / Aeneas `505b6ca-dirty` failure is
`Could not compute a loop fixed point in 2 iterations` in `walk_residue`. The read source has
`nodes: [Vec<DirectoryNode>; 2]`, `DirectoryNode.children: [u32; 3]`, and nested
`IdentityNode.edges: Vec<Edge>`. Even `walk_residue(nodes: &[DirectoryNode], ...)` performs successive
index borrows followed by a field/array projection. Rust's copied `u32` loop variables do not imply
that Aeneas's symbolic context contains only scalars. Its loop interpreter joins reentry contexts
and compares the reduced contexts; failure of that comparison is not an out-of-bounds counterexample,
a Rust borrow-check failure, or evidence that the mathematical trie is unprovable. Earlier
`AEndedSharedLoan` / context-join failures are consistent with this boundary, but without the
failing context trace they do not establish which loan causes nonconvergence. Merely changing
`Option` to a sentinel, adding braces, or spelling each index afresh has already failed to establish
translation. Flattening is a selected hypothesis, not a diagnosis proved by those errors.

Use this concrete internal target (names are prospective):

```text
Directory {
  residue_words: Vec<u32>,   // node h: words[4*h .. 4*h+4]
  identity_words: Vec<u32>,  // node h: words[2*h .. 2*h+2]
  edge_words: Vec<u32>,      // edge h: words[3*h .. 3*h+3]
  hyperbolic_nodes: usize, parabolic_nodes: usize
}
residue record = [child0, child1, child2, identity_root]
identity record = [terminal_handle, first_edge]
edge record     = [byte_label_as_u32, child_identity_node, next_edge]
NONE = u32::MAX; every live node/edge/terminal handle < NONE
```

- One shared residue arena, disjoint roots: Hyperbolic node 0, Parabolic node 1; all descendants
  retain root ownership. Empty vectors represent two implicit empty roots (logical counts 1/1).
  First reservation materializes both root records before any effect; `new` allocates nothing.
  Node 0, byte 0, residue 0 and terminal handle 0 remain ordinary data. Parabolic child2 is NONE.
  `StagedPath.residue_path[0]` must use the selected root, not an unconditional zero.
- Identity edges form sorted, unique, acyclic sibling chains of at most 256 labels. There is no
  per-node `Vec`, 256-wide dense identity allocation, fiber scan, second key/value store or handle
  registry search. Terminal is separate from byte labels, preserving empty and prefix keys.
  Edges are append-only records; insertion splices one reserved record, not a population-wide shift.
- Handles are logical record numbers, not addresses or flattened word offsets. Checked sizing
  proves `stride * record_count`, byte counts, additions and casts fit `usize`, `isize::MAX` allocation
  bounds and the sentinel domain before effects. Lengths are multiples of their stride. Every
  followed handle is initialized, in range and in the correct arena/root. Vec growth may move bytes
  but cannot invalidate these handles; growth/moves and initialization remain charged work. This
  clarifies earlier “nonmoving nodes”: stable handles, not a claim of pointer-stable Vec allocation.

**Traversal pattern:** a single-exit scalar `while`, with scalar depth, handle, missing flag and
stopping depth; immutable owner/key inputs stay fixed. Read one digit, call a production per-level
helper returning a copied child handle, then advance the scalar state. Its scalar-load helper has
shape `read_word(&Vec<u32>, usize) -> u32`, contains one indexed copy, and returns no reference.
The caller computes `4 * node + digit` only after the sentinel/bounds guard. No `&node`, subslice,
`&children`, iterator, closure-captured arena or borrowed result survives a step. Identity lookup
uses the same scalar-access discipline for `(edge,label,child,next)` and a scalar bounded sibling
scan; key byte access occurs once per identity level. Query and retained access supply bytes to the
same path algorithm, not separate traversals. Keep helper boundaries in the Charon input and inspect
the generated dependency graph; a Rust `inline` attribute is not evidence of symbolic separation.

Writes likewise read scalar handles first, finish the read helper call, then call a no-borrow-return
scalar store helper. No loop-carried mutable node reference, nested `IndexMut` projection, early
return/`?`/outer continue inside a loop, or recursive replacement of the loop is selected. Failure
is scalar control state with one return after the loop. These are implementation constraints to test
on the pinned translator, not a claim that it supports all such Rust programs. Do not unroll 64
levels, clone an arena per step, lower capacity to a fixed-size array or change admission to make
translation pass. Fixed arrays remain appropriate only for the existing bounded residue path.

### Contracts that conversion must preserve

There is exactly one body for each production directory operation. Both standalone `Locator` and
`TableLocator` delegate to it. Extraction wrappers call those bodies via the existing `#[path]`
inclusion; no proof-only implementation, cfg-selected substitute, reference traversal or mock may
replace them. Cold structural validators may inspect scalar records independently, but cannot
serve as a second lookup engine or as an assumed source proof. Replace borrowed `identity_nodes()`
exposure with copied terminal/edge/count accessors; do not reconstruct the old nested arena as a
compatibility copy. Public face tokens, carrier encoding, H3_40 and borrowed read types do not change.

Preserve all four fields of `NativeLocator.TableStorageCoverage`, including sound complete paths
and reverse links, not just forward reachability. Table owns payloads once, including default-valued
entries. The current diagnostic rejects Elliptic payloads; it is not already an all-face C witness.
The adapter must combine filtered-face coverage with Elliptic exact-address coverage and explicit
exclusion of other storage. `Absent` needs that relation; `Unknown` is neither absence, presence nor
vacancy. Keep fingerprint/full `(Sigma,K)` confirmation and independent Regen identity. No read
cutover, summary deletion or W7 composition is released here.

`LocatorReservationReceipt` retains all eight logical categories. Physical capacity is translated
to record capacity by stride, not confused with word length. Reserve complete additional node/edge
records, reverse links, residue/identity staging and any actual retained-byte buffers before
retain/clone/merge/relocation. Account for already committed lengths, root materialization, scratch
and every reservation; a fresh-path estimate is not a cumulative capacity proof. Empty receipt is
valid only for a genuinely locator-free operation. Observe actual storage, not `required` copied
into `observed`. Stage privately without modifying committed sibling links; publish only after
all checks. On failure restore staged lengths/counters and release staging, leaving committed
links/payloads/counts unchanged with zero callbacks. Found updates allocate no duplicate terminal.

Sol must reconcile `ReserveBeforeRetainContract`'s exact phase order and `after = before` with the
concrete state projection: current code stages scratch during reservation and may change Vec
capacities on failure. Neither bitwise equality of raw vectors nor completed source refinement
follows from logical rollback. Preserve the stated contract; explicitly relate private reservation
scratch to its phases and prove stutter on the represented committed state, or report the mismatch.
The transient `QueryIdentity` currently supplies unit retention and `matches_retained = false`;
do not instantiate the full faithful-retention law with that view. Separate byte-access requirements
from owning retention requirements within the same engine if required by the source adapter.

Target lookup work is `O(d + 256*ell)` scalar loads/comparisons for `d=64/40` and `ell` exact bytes,
plus declared encoding/equality costs, from one residue loop and at most 256 sibling steps per byte.
Space is proportional to occupied nodes and edges, plus `O(d+ell)` staged handles. These are design
recurrences, not source-proved bounds or latency measurements. Vec growth and the aggregate insertion/
rebuild recurrence remain obligations; neither copying all arenas nor a reachable population sweep
can be hidden in the allocator exclusion. Strict REG-1 and the adversarial cost proposition stand.

### Tool and warning decision

Keep `proof/regen.sh`'s `-warnings-as-errors`. Specifically, the warnings for missing
`core::iter::range::Step::{forward_overflowing,backward_overflowing}` builtin trait fields are gate
failures for this crate until the actual root closure is shown not to require them or the pinned
model/registration is repaired. They are not the loop fixed-point error. The installed Lean
`Aeneas/Std/Core/Iter.lean` contains scalar definitions for those names; that alone does not repair
the extractor's builtin trait-field mapping or establish the semantics of every Step instance.
Do not globally disable warnings, suppress unknown externals, accept partial generated files or
introduce axioms to accommodate the mismatch. Removing unused range machinery from the real bodies
may remove a warning without changing the gate; required range operations need a checked model.

Residual: the replacement's read AND mutation loops have no accepted extraction receipt. No
unshapeable limitation is established by the supplied failures. If scalar loads/stores still fail,
stop with their real-source LLBC/context trace; the next task is a pinned translator repair, not
another production implementation. The inspected pinned-base `src/Config.ml` hard-codes
`loop_fixed_point_max_num_iters = 2`; its comment treats nonconvergence as likely a bug. No supported
iteration CLI knob is established here. Increasing that constant is a tool-source change, not a
Lean heartbeat fix, and cannot be assumed to fix context joins. Require a bounded reproducer and
regression/control receipt before selecting a minimal patch or version change. Tool/model changes
need their own owner, reproducible pin/digests and source-semantics review. Closed-cone audits and
all source-link obligations remain unchanged. No tool/config edit is authorized by this decision.

## Explicit-capability API (A4, authoritative migration contract)

This section supersedes earlier compatibility holds and prohibitions on adding identity bounds in
this spec and the A3 plan. It does not claim the interface is implemented. One `Table`, canonical
`HMap (Sigma × K) V`, compile-time carriers and borrowed total reads are retained.
There is **no final generic compatibility table, implicit hash/sweep fallback, or runtime
UnsupportedFace result**. A home selects a sealed face domain: `Sigma` (all three faces) or new
`LocatorSigma` (Parabolic/Hyperbolic only). The latter is an explicit capability-constrained path
on the **same Table**, not the rejected generic fallback. Both domains default to Parabolic.
Logical identity still carries the ordinary `Sigma` obtained from the domain token.

### Concrete traits and bounds

Use the existing `coord::locator::ExactIdentity`, not a second byte-identity trait. I2 may re-export
it from the shared core without changing its public path. Its `encoding_len`/`encoding_byte` supply
query bytes; the Table directory borrows these and never calls `retain` to create another key store.
Require finite, stable, allocation-free indexed byte access on `0..encoding_len`, equality-faithful
bytes and an explicit cost for length/byte access. Constant-time byte access is required for the
stated trie recurrence; an implementation that rescans a prefix per byte needs a different proved
cost and is not silently admitted to that bound. `Eq`, trait implementation and tests alone do not
prove these laws.

Add the following capability to `src/coord/address.rs`; strengthen the existing `FaceCoord`, rather
than introducing a competing address enum. Names/signatures below are implementation targets:

```rust
pub trait FaceResidueCapability<K> {
    fn parabolic_dual(&self, key: &K) -> ParabolicDual;
    fn hyperbolic_scalar(&self, profile: H3_40, key: &K) -> Residue;
}
pub trait EllipticPacking<K> {
    fn elliptic_pack(&self, key: &K) -> u64;
    fn elliptic_unpack(&self, word: u64) -> Option<K>;
}
pub enum LocatorSigma { Parabolic, Hyperbolic }
pub trait FaceDomain<K, H: ?Sized>: sealed::Sealed + Copy {
    const DEFAULT: Self; // Parabolic for both admitted domains
    fn sigma(self) -> Sigma;
    fn from_sigma(sigma: Sigma) -> Option<Self>;
    fn encode(self, home: &H, profile: H3_40, key: &K) -> FaceTableAddress;
}
pub trait FaceCoord<K>: FaceResidueCapability<K> {
    type Faces: FaceDomain<K, Self>;
    fn coord(&self, profile: H3_40, face: Self::Faces, key: &K) -> FaceTableAddress;
}
pub trait FaceHome<K>: FaceCoord<K> {
    fn address(&self, profile: H3_40, face: Self::Faces, key: &K) -> FaceTableAddress;
}
pub trait Home<K>: FaceHome<K> {
    fn home(&self, key: &K) -> u64;
    fn address(&self, sigma: Sigma, key: &K) -> TableAddress;
}
```

The only `FaceDomain` implementations are `Sigma` for
`H: FaceResidueCapability<K> + EllipticPacking<K>`, and `LocatorSigma` for
`H: FaceResidueCapability<K>`. Domain encoding is a shared exhaustive match, not a user callback.
`FaceCoord::coord` defaults to `face.encode(self,profile,key)`; `FaceHome::address` retains default
delegation to it. The equations in `FaceHomeLaws` apply on the selected domain: exact packed word;
`ParabolicWord(parabolic_dual(k).residue())`; and
`HyperbolicResidue(profile, hyperbolic_scalar(profile,k))`. For the all-face domain require
`elliptic_unpack(elliptic_pack(k)) == Some(k)`, stable equality-respecting encodings and the frozen
H3_40 version. These are refinement obligations, not marker-trait proof certificates.

Delete `impl<K,F:Fn(&K)->u64> Home<K> for F`; add **no replacement blanket Home implementation**.
Named homes explicitly implement both capability traits and `Home`. Remove the placement-only
default body of `Home::address`. `Home::{home,address}` remain a **physical storage adapter** during
flat/spill migration, with their existing signatures; they are not semantic encoders or absence
certificates. Qualify calls as `Home::address` versus `FaceHome::address` to avoid method ambiguity.
The directory consumes only `FaceHome::address`; physical routing must be stable, equality-respecting
and consistent between query/held/rebuild paths, but it cannot mint face or coverage evidence.
`direct_word` retains its independent resident-certification law. No claim that low-bit physical
placement is the final algebra-native layout follows from this interface step.

`WordHome` implements the same semantic capability (`Faces=Sigma`) as `WordFaceHome`, by delegation rather than
another word encoding. Its old physical adapter stays separate. Table configuration retains one
immutable validated H3_40 token; `new(default, home)` keeps its arguments. The address module supplies
the fixed token through a validated constant constructor (no per-query validation, mutable profile
or unchecked arbitrary modulus). No new profile argument or fallible read is introduced.

The final declaration and operational bounds are:

```rust
pub struct Table<K, V, H, const N: usize = 16,
                 R: KeyRep<K, H> = Stored<K>> { /* existing storage + directory */ }
pub trait KeyRep<K, H>: sealed::Sealed + Clone {
    // Existing hold/matches/matches_held/key_owned/address_held signatures retained.
    fn held_encoding_len(carrier: &Self) -> usize;
    fn held_encoding_byte(carrier: &Self, index: usize) -> u8;
    fn query_encoding_len(key: &K) -> usize;
    fn query_encoding_byte(key: &K, index: usize) -> u8;
    fn face_address_held(home: &H, profile: H3_40, face: H::Faces,
                         carrier: &Self) -> FaceTableAddress where H: FaceHome<K>;
}
```

Keep the existing operational `H: Home<K>,R: KeyRep<K,H>` bounds. The sealed carrier instances,
not a blanket identity implementation, enforce `K: ExactIdentity` at construction. Generic code
with `R: KeyRep<K,H>` obtains query access through R's delegation to K; it need not propagate a new
redundant K bound across every raw storage helper. This is a type-level requirement: no sealed
carrier implementation exists for opaque Eq-only keys. Preserve every existing `Clone`, `Send`,
`Sync`, `Merge` and callback bound as well. `Stored`'s instance becomes
`impl<K: ExactIdentity + Clone,H: Home<K>> KeyRep<K,H> for Stored<K>`. It borrows its sole stored
`K` for both byte and face queries; it does not cache a duplicate `K::Retained`.
`Regen`'s instance becomes `impl<K: ExactIdentity,H: Home<K>,C: ExactCodec<K,H>> ...`.
Strengthen sealed `ExactCodec<K,H>` implementations to require `K: ExactIdentity,H: Home<K>`, with
`fn identity_encoding_len(identity:u64)->usize` and
`fn identity_encoding_byte(identity:u64,index:usize)->u8`. Retain its existing `encode`/`decode`
signatures, but require these byte accessors to equal the query's canonical bytes after encode.
`Regen` delegates held bytes to these accessors, never reconstructing `K` on the hot identity path.
Its cold `key_owned`/held face-address recovery may decode the **independent exact identity**, not
the face residue. `I64WordCodec` keeps `word*PHI`/`unspread`; original-bit big-endian bytes are
recovered by word arithmetic, not by reusing the residue or allocating a key. `Sigma` stays outside
the carrier and must be checked alongside identity.

Every public `_at` face argument changes from `Sigma` to `H::Faces`, including input/query tuples
in serial/parallel build/read APIs; arguments for `Faces=Sigma` stay source-compatible. Default
operations use `<H::Faces as FaceDomain<K,H>>::DEFAULT`. The type parameter order/default carrier
does not change. Private payloads may retain erased `Sigma`, but constructor/preservation must
establish domain membership. Relocation recovers the token through `from_sigma` **before effects**;
malformed/out-of-domain storage cannot issue a certificate and must not be mapped to a different
face or default-valued success. This is an internal invariant check, not a new public refusal for
admitted keys. Prepared/parallel code must carry the domain token until the shared internal boundary
erases it; it cannot accept an arbitrary Sigma on behalf of a locator-only table.

`Coord`, `Coord::{coord,coord_with,SPREAD,CHEAP_EQ}`, `Packed`, and `i21` keep their Rust signatures.
`Coord` is only a possible explicit **parabolic** encoding adapter; it does not imply any of the new
capabilities. `SPREAD`/`CHEAP_EQ` do not authorize omission of bytes, face checks or full equality.
`Packed` may discharge elliptic packing only on its exact domain; it supplies neither a residue
encoding nor finite identity access automatically. No `Coord → FaceHome` blanket conversion exists.

### Face and population policy

| Population / face | Requirement and migration outcome |
|---|---|
| Elliptic | Exact at-most-word packing with inverse on image; exact-address coverage, no locator. No field instance for `Z/2^64`. |
| Parabolic | Stable specified dual encoding; index only `a=fstHom(a+bε)` for 64 binary levels, plus independent exact bytes and faithful carrier. |
| Hyperbolic | Specified H3_40 encoding, canonical residue and 40 ternary levels, plus independent exact bytes and faithful carrier. No old hash-then-reduce adapter. |
| All `Table` calls, including default-Parabolic | Exact bytes plus both locator residues. `Faces=Sigma` additionally requires exact packing; `Faces=LocatorSigma` cannot be passed Elliptic at compile time. No runtime unsupported-face branch. |
| Opaque `K: Eq` without an encoding | Rejected at compile time until the owner supplies an equality-faithful `ExactIdentity` and a law-bearing face home. No scan-only generic route is retained. |
| Arbitrary `Fn(&K)->u64` home | Blanket admission removed. Replace with a named explicit home and the required encodings/laws; wrapping the closure without those laws is not migration. |
| `Regen` decode-as-identity from residue | Removed from the admitted set. Sealed exact codec plus query/held-byte agreement is required on every face. Correct decoding of the independent `I64WordCodec` word remains admitted on all faces. |
| Finite byte sequences | `ExactIdentity` plus a specified residue home admits `Faces=LocatorSigma`. Unrestricted `Vec<u8>` cannot inject into a `u64`, so `Faces=Sigma` is rejected, not silently truncated. No byte residue algorithm is inferred from Eq or an old hash. Missing explicit encoding/cost evidence blocks that concrete adapter, not the typed path. |
| Ledger's REG-1 word-wrapper keys | Retain the full original equality domain and inputs; supply canonical bytes and explicit face encodings. Raw placement may remain a physical adapter, not the semantic hyperbolic encoding. Missing or impossible adapters block that row and full acceptance; never drop/relabel it as not applicable merely to pass. |

`examples/reg1/workload.rs::ShardKey { tag:u32, home:u64 }` derives equality on both fields: it
needs 12 exact bytes, not a tag-only codec or an invented 64-bit packing. Migrate it with
`Faces=LocatorSigma`, preserve both fields, and change explicit face tuples to the corresponding
locator tokens without changing inputs or logical faces. StrataKey/SharedKey are word wrappers
and can implement `Faces=Sigma`. This prevents an all-face bound from silently deleting a real row.

This is an explicit reduction of the **old** unconstrained API, authorized by the lead; it is not
an exclusion from the newly stated joint admitted domain. No member satisfying its domain laws is
silently excluded. `table_admission_map_exact`'s byte-key row is conditional on its advertised
face home: `FaceResidueCapability Key Exact` leaves `Exact` abstract. It is not a witness of
unbounded bytes packed into Rust `u64`. Sol must supply the domain-restricted adapter for the two
filtered faces without asserting the all-face contract for these keys. Likewise `api_migration_is_pending` records S1's old decision
state, not a veto of the lead's new selection; Sol updates that record, not its mathematical laws.

### Public-surface break inventory (all 45 generic_api rows accounted for)

The ledger is `docs/evidence/anative-admission-ledger.md`, §1; row names below, not its moved line
numbers, identify the surface. “Bound” includes an inherited trait/impl bound, even with unchanged
argument lists. No `admit` disposition in that historical census overrides this decision.

| Ledger items | Exact change |
|---|---|
| `Home`; `Home::home`; `Home::address` default; blanket `Home` closure adapter | `Home: FaceHome`; both methods inherit that capability; physical-only interpretation. Remove address default and closure blanket implementation. |
| `WordHome`; `WordHome::address` | Add the semantic supertraits; existing physical signature remains, no longer an authoritative all-face address. |
| `Table` default `R`; `Table::new`; `Table::search_decision_at`; `Table::get_at`; `Table::get`; `Table::len`; `Table::is_empty`; `Table::insert_with_at`; `Table::insert_with`; `Table::build_at`; `Table::build` | Existing H/R bounds become capability-bearing; the sealed R implementation requires `K: ExactIdentity`. `Stored<K>` remains default. `_at` arguments/input tuples use `H::Faces`; other arguments/returns and default Parabolic are unchanged. Even len/is_empty cannot be called on an unconstructible Eq-only carrier instance. No redundant raw-type K bound is added. |
| `KeyRep`; `KeyRep::hold`; `KeyRep for Stored`; `ExactCodec`; `KeyRep for Regen`; `I64WordCodec` | Strengthened concrete K/H instance bounds and required query/held/codec byte methods above. `hold` keeps its arguments but is available only through the stronger instances. `I64WordCodec` supplies query/held-byte agreement, not merely encode/decode. |
| `build_at_parallel`; `build_at_parallel_phased`; `build_parallel`; `build_parallel_phased`; `get_each_at_parallel`; `get_each_at_parallel_phased`; `get_each_parallel`; `get_each_parallel_phased`; `sum_parallel` | Same capability-bearing H/R bounds, plus existing thread/callback bounds and `H::Faces: Send + Sync` where transported. `_at` input/query tuples use `H::Faces`; serial/parallel domains agree. |
| `Merge`; blanket `Merge` closure adapter; `WordMerge`; `ReplacementMerge`; `Stored<K>` declaration; `Regen<C>` declaration; `PhaseNanos`; `reservation_failure::arm`; `Coord`; `Coord::SPREAD`; `Coord::CHEAP_EQ`; `Packed`; `i21` | No signature/bound change to these 13 items. Stored/Regen **instances**, not their payload declarations, change. Coord/i21 lose no standalone functionality; neither grants table admission. |

Also affected beyond those 45 rows: existing `KeyRep::{matches,matches_held,key_owned,address_held}`
and `ExactCodec::{encode,decode}` inherit the stronger trait bounds; `FaceCoord` gains its supertrait,
and therefore `FaceHome`/`WordFaceHome` require the new implementation. `TableAddress` remains the
physical result type with unchanged fields, never the directory's semantic input. No other public
signature change is authorized implicitly; a newly found consumer is a separately owned slice.

## Per-face address contract

**Decision:** one `Table`, face-indexed addresses, unchanged canonical
`Apeiron.Table.HMap (Sigma × K) V`. `Cell<i64>` remains the value/key word kernel, with its
`ZMod (2^64)` denotation and existing `word_mul_denote` / `word_norm_denote` obligations.
The wholesale scalar pivot is rejected. Face selection is the explicit `Sigma` enum, not an order
on a finite ring. Neither a word-ring field classification nor a homomorphism from the word ring
to an odd-modulus ring is required or asserted.

| Address face | Address ring and map | Identity/admission | Absence certificate after summary removal |
|---|---|---|---|
| Elliptic | `Z/2^64`; exact `pack` / `Packed::packed` | `unpack(pack(k)) = k` on the declared packable domain; bijection onto its image (onto all words only for full-width codecs). `Regen` retains this exact word; `Stored` may retain the key | Empty exact-address entry under complete exact-address coverage, or complete candidate search with decoded full-key exclusion. An unallocated page alone is insufficient without the coverage invariant |
| Parabolic | Dual numbers over `Z/2^64`; index `a = fstHom(a+bε)` by a 64-level base-2 residue trie. Word encoding supplies `a = spread(word).place` | Full key retained, or an independently lossless codec; neither `a` nor `b` decides identity. No half/idempotent split is imported | Missing covered binary residue/identity path; `Unknown` continues complete full-key search, never default by itself |
| Hyperbolic | `Z/q`, `q = 3^40 = 12157665459056928801`; quadratic `e±` channels at `sigma=1`. No nontrivial scalar CRT split in this profile | Reduction is many-to-one; retain the full key or independently lossless decoding | Missing covered residue path, then exact identity-trie lookup; marginal channels never establish presence |

This replaces the earlier all-face homomorphic-address / real-field layout requirement. Packing and
spreading are encodings, not unital ring homomorphisms. The hyperbolic CRT reductions are ring
homomorphisms **within `Z/m`**; an arbitrary key encoding into `Z/m` is not one. Norm multiplicativity
does not make the norm additive, injective or an absence certificate. No norm-level index is required.

### Coord / Home / TableAddress boundary

The current `Coord::coord() -> u64`, default `Home::address` and `WordHome` are not this interface:
`WordHome.address` ignores `Sigma`, and `TableAddress` contains only `place: u64` plus
`direct_word: Option<u32>`. `Addr.id` and `Addr.place` do not denote two CRT channels.

Proposed semantic interface, **not implemented Rust declarations**:

```text
AddressValue = EllipticWord(u64) | ParabolicWord(u64) | HyperbolicResidue(H3_40, residue)
FaceTableAddress = { value: AddressValue }   // value determines Sigma; no physical index
CoordFace(profile,sigma,K): admitted key encoding into the selected address ring
FaceHome.address(profile,sigma,k): FaceTableAddress satisfying CoordFace(profile,sigma,K)
Admissible(profile,sigma,K,H,R): face encoding + faithful carrier + stable Home laws
                              + exact identity discrimination for the chosen locator
```

- The only initial hyperbolic profile is sealed `H3_40`, defined below. Its identity includes the
  reduction/encoding version; it is not a modulus supplied by a query. Residues satisfy `0 ≤ a < q`.
  Address constructors do not accept a face tag inconsistent with the variant. A future second
  profile requires explicit constructor admission, never implicit conversion at lookup.
- `Coord` must gain a face-aware operation/capability (with `Packed` for the elliptic case), or be
  retained only as the parabolic encoding adapter. `Home::home -> u64` cannot be the authoritative
  all-face fallback. The blanket closure adapter must acquire explicit face/profile laws; it cannot
  fabricate an elliptic exactness certificate or an odd-modulus ring from an arbitrary `u64`.
- `Home.address` is total on the admitted key-face domain, deterministic/stable for the table's
  lifetime and equality-respecting. Its denotation equals the specified face encoding. Held/query
  addresses agree after decoding; clone and relocation preserve profile, `Sigma`, identity and value.
  Thread sharing requires the existing callback stability/thread-safety laws as well.
- Physical indexes are separate bounded locators, not casts of `AddressValue`. The finite
  hyperbolic locator below replaces low-byte shard / load-rounded power-of-two placement; its
  source-linked coverage and cost theorems are MISSING. No hash fallback or full-address allocation.
- `direct_word: Some(u32)` currently certifies resident admission only. It must be replaced or
  refined by a locator-specific certificate, not promoted to full-domain injectivity. A future
  direct-empty shortcut requires its own complete cover invariant across all storage locations.
- Carrier choice remains compile-time (`Stored<K>` / `Regen<C>`), not a runtime enum of payload
  types. Address variants may be runtime values because `Sigma` already is. `Sigma` stays outside
  the retained carrier and participates in every equality; equal addresses across faces never alias
  keys. `Stored` can serve any face with a valid address law. `Regen` can serve a many-to-one address
  only with a separate exact identity, never by decoding that address.

**Transition, not a new table.** W1 adds `coord::address::{AddressValue, FaceTableAddress,
HyperbolicProfile, Residue, FaceCoord, FaceHome, WordFaceHome}` as contract vocabulary and pure
constructors. These names are prospective. Keep existing `Coord`, `Packed`, `Addr`, `Home`,
`WordHome`, `TableAddress { place, direct_word }`, carriers and every table call unchanged.
Do not add a default face method to existing `Home`: that would either fabricate a law or silently
migrate existing callers. `FaceHome` has no blanket implementation for `Fn(&K)->u64`.
W4 replaces the old table address result with this vocabulary and makes its face-aware operation
authoritative, after consumer/admission and proof feasibility gates. It must consume the W1
definitions, not keep a parallel address implementation. W1 establishes no table refinement.

`FaceCoord`/`FaceHome` are semantic capability laws, not proof certificates minted by implementing a
Rust trait. The W1 concrete instance covers `u64` and `i64`: elliptic is the exact unsigned word
(`Packed` image), parabolic is `spread(word).place`, hyperbolic is `reduce_word(word)` below.
For `i64`, `word` is its two's-complement bit pattern, not signed remainder. `Sigma` remains part
of logical identity even when two variants contain the same numeric value. `Regen` retains its
independent exact identity; no optional scalar field of an address proves that identity law.
Other key types require explicit instances and admission evidence; no W1 narrowing of the old API.

**Admissible key-face condition:** elliptic keys have a checked at-most-word exact packing;
parabolic keys have a total stable fold and faithful retained identity; hyperbolic keys have an
explicit encoding into the validated `Z/m` and faithful retained identity. For a word, unsigned
bit-pattern interpretation followed by integer reduction modulo `m` is a possible set encoding,
not a word-ring homomorphism. General byte/key encoding needs a specified total algorithm and cost.
If algebra operations on keys are claimed to commute with encoding, that is a separate law over
the declared key algebra; it cannot be assumed from `Coord`.

Reads remain total and borrowed on the admitted API; constructor profiles/type capabilities must
establish these laws before reads. No silent default, panic, fallback hash or `UnsupportedFace`
error is authorized. A4 explicitly replaces the old wider public population with the capability
domains above, including its public break list. Both carriers, defaults and every REG-1 row require
adapters and evidence; selecting a domain is not evidence for an omitted API or workload.

## Hyperbolic layout and identity

### Concrete profile H3_40

Select `p=3`, `k=40`, `q=12157665459056928801`, the largest power of the smallest odd prime that
fits a `u64`: `2^63 < q < 2^64 < 3^41 = 36472996377170786403`. A residue occupies one 64-bit
word, not a `usize` or a signed `i64`. For `sigma=1`, `r=s=1` and
`h=(q+1)/2=6078832729528464401`; `2*h=q+1`, so `2` and `r` are units. The primality witness is
for `3`, not a large probable-prime test. `k>1` deliberately exercises a prime power, not only a field.
The choice gives a fixed 40-step ternary locator and a one-subtraction word reduction. It is an
engineering profile, not a theorem that this modulus is uniquely forced or fastest.

For any unsigned word `w`, `reduce_word(w) = w` if `w<q`, otherwise `w-q`.
Since `w<2^64<2*q`, this is exactly integer remainder modulo `q`, with no underflow or wrapping.
`u64::MAX` reduces to `6289078614652622814`. A checked canonical-residue constructor accepts
exactly `[0,q)` and rejects `q` and above; distinguish it from this total reduction constructor.
No truncating cast, masking or `spread` substitutes for this reduction. Signed words first preserve
their bits. General keys need an explicit integer/byte encoding before reduction; applying the old
unkeyed hash and then reducing is not admitted as the hyperbolic successor encoding.

If residue arithmetic executes later, widen before arithmetic: add in `u128`; subtract as
`a + q - b` in `u128`; multiply in `u128`, then take integer `% q` and checked-narrow the bounded
remainder. `(q-1)^2 < 2^128`. Reduce each product before adding products in a quadratic multiply:
the sum of two unreduced products is **not** guaranteed to fit `u128`. This is total on valid
residues. Allocation counts/byte offsets instead use checked `usize` arithmetic and fail with
`SizeOverflow`; neither arithmetic class silently wraps. W1 needs reduction and constructors only,
not a new scalar `Ring` implementation or a replacement for `Cell<i64>`.

Profile fields are private, with one admitted versioned token `H3_40`; validation checks the exact
tuple `(3,40,q,1,1,h)`, 40 checked multiplications by 3, width, and the unit/root equations using
wide integers. Wrong modulus, exponent, root, half or version is rejected before an admitted table
exists. Validation occurs at configuration construction, not each read. The validated token is
immutable and copied by value into configuration/addresses; there is no mutable global modulus,
per-query modulus, reconfiguration setter or load-dependent profile. Raw residues cannot forge the
token. Internal construction from the frozen constant must satisfy the same validation law.

### Finite hyperbolic locator

Select a **sparse ternary residue trie followed by an exact-identity trie**, inside the same `Table`.
The first trie is the canonical chain of quotients `Z/3^40 → Z/3^j`, not a low-byte shard function.
At depth `j=0..39`, read digit `floor(a/3^j) mod 3`; operationally repeat remainder/division by 3
exactly 40 times, including leading zero digits. Each node has three optional child handles. A
missing edge certifies an empty residue prefix only under the coverage invariant below. Zero digits
and residue zero are ordinary data; optional handles/live flags, never numeric zero, mark vacancy.

At a residue leaf, discriminate **all** keys in that fiber by an equality-faithful finite byte
encoding `E(k)` (`E(k)=E(k') iff k=k'`), not by a linear collision list. An identity node has an
optional terminal representative and a sorted, duplicate-free sibling list of byte-labelled edges,
at most 256 edges. A terminal marker is separate from every byte, so empty/prefix/zero-containing
keys differ. Read each query byte once, scan at most 256 labels, follow the matching edge. At the
terminal, compare the same retained/decoded `(Sigma,K)` before returning the single stored value.
Word identities are exactly eight big-endian bytes of the original bit pattern, never the residue.
For the word profile a lookup traverses at most 40 residue edges and 8 identity edges, examines at
most 2048 byte-edge labels, and performs at most one full-key comparison. These are design bounds,
not measured latency or source-linked proved bounds. Prefix compression is not selected in this cut.

Node handles refer to initialized, nonmoving reserved nodes, with constant-work handle dereference;
no linear handle registry or whole-tree relocation on insert. There is one hyperbolic root scoped
by the table profile and face. Missing paths are constructed privately, all new nodes/payload/link
storage reserved before hold/clone/merge, then published once. The source feasibility task must
realize this nonmoving handle/resource interface with safe storage and checked failure, or stop;
substituting a relocating vector without a charged recurrence is not an implementation detail.
With `n` live keys and `S=Σ_live |E(k)|`, occupied residue nodes are at most `1+40*n`, identity
nodes at most `n+S`, edges no more than those node totals, and payloads exactly `n`. Account also
for transient staged nodes, retained-key/value bytes and any allocator metadata. Check all count,
byte-size and handle bounds against `usize` and the allocation object's `isize::MAX` limit before
effects. Do not allocate `q` or `q²` entries. No fiber-capacity refusal or resurrected `ShardFull`.

**Coverage and decisions.** Every live representative has its entire canonical 40-digit path and
entire exact-identity path ending at that representative; every terminal points back to one live
payload with that path, profile and face; full logical keys and path edges are unique. Empty
construction establishes the invariant; insertion, relocation and any future removal must preserve
both directions. A missing edge/terminal then proves absence. A reached terminal proves presence
only after full identity confirmation. Fresh insertion at a missing path is not a physical
first-fit slot; W4 must transport the old `Vacant`/first-fit obligations to unique path extension,
not feed fabricated indexes to the old apply. Resource refusal leaves all committed links/payloads
and counts unchanged with zero hold/clone/merge/relocation, and build returns no partial table.

**Cost obligation, still MISSING.** Proposed `Native.Locator.coverage`, `lookup_complete`,
`lookup_work_bound`, `insert_work_bound` and `sequence_work_bound` are names for absent targets,
not declarations or supplied premises. Count digit arithmetic, labels inspected, handle loads,
key bytes inspected/copied, allocated/initialized nodes and link writes. For key encoding length
`ell`, require internal lookup/insert work `≤ C*(40+256*ell+1) + E_cost + Eq_cost`, plus separately
charged allocator and callback costs; no term may depend on residue-fiber population. Repeated key
comparisons/copies must be charged, not hidden as one input read. Require an aggregate bound from
the real constructor compatible with the unchanged adversarial proposition below and a linear
bound in `T` for fixed-width keys, including node initialization/reservation and failing operations.
The traversal argument alone does not prove source storage growth costs or allocator correspondence.

For generic opaque `K: Eq` plus an arbitrary `Home` closure, no equality-faithful bounded byte
encoding is supplied by those traits. A4 resolves that **admission blocker** with the explicit
capability break, not a scan fallback or relabelled REG-1 rows. The ledger's population map and
Sol's law/adapter ratification remain required. The finite
locator is specified; its safe-storage/extraction feasibility, coverage/preservation and adversarial
cost theorems block **behavior migration**, not W1's additive address types/law tests. Constants
40/256 may be expensive: performance parity remains unestablished and strict REG-1 is unchanged.

**Prime-power base (the deciding case).** Let `q = p^k`, `p` an odd prime, `k ≥ 1`, so `2` is a unit in
`Z/q`. The quadratic algebra splits through the canonical witnesses `r*s = 1`, `2*h = 1`, `r*r = sigma`,
with `e_plus = (h, h*s)` and `e_minus = (h, -h*s)`; no coprime factorization of a composite modulus is
required. `(Z/9)[x]/(x²−1) ≅ Z/9 × Z/9` realizes this at `r = s = 1`, `h = 5`. Locality forbids
nontrivial *scalar* idempotents in `Z/p^k`; it does not forbid the quadratic extension's `e±`.

**Composite extension (optional scalar CRT, not H3_40).** For a different profile `m = ∏ q_i`, `d ≥ 2`, each
`q_i = p_i^{k_i}`, `p_i` odd, the `q_i` pairwise coprime, then `R_H = Z/m ≃ ∏ i, Z/q_i` is a ring
product (not numerical equality of a ring and an integer factorization). Define `e_i` as the inverse CRT
image of the unit vector at `i`:

```text
e_i^2 = e_i;   e_i * e_j = 0 (i ≠ j);   ∑ i, e_i = 1
pi_i(a) = a mod q_i;   P_i(a) = e_i * a
P_i(P_j(a)) = 0 (i ≠ j);   a = ∑ i, P_i(a)
```

`pi_i` is unital into `Z/q_i`. `P_i` is unital only into the ideal ring `e_i R_H` whose unit
is `e_i`, not into the ambient ring with unit `1`. This algebraic noninterference is not memory
isolation, independence of adversarial inputs or a bound on collision occupancy.

**Do not identify two different splits.** When `m` is composite, CRT decomposes the scalar `R_H` into
`d` factors; the quadratic split below needs none of that and holds for a single odd prime power.
The quadratic algebra `A_sigma(R_H)` splits as `R_H × R_H` under `r*r = sigma`, `r*s = 1`,
`2*h = 1`, using `e_plus=(h,h*s)` and `e_minus=(h,-h*s)`. Its per-factor refinement has
`2d` channels indexed by `(i,+/-)`, not `d`. The CRT `e_i` are not literally the quadratic
`e_plus/e_minus` even when `d=2`. At the executing hyperbolic dial `sigma=1`, use `r=s=1` and
the inverse of `2` in the odd ring. A different dial needs its own unit-root witness.
The selected address is one scalar residue, not a cell or two independent scalar CRT channels.
Its explicit embedding is `a ↦ (a,0)` in `RingCell (Z/q)`; both quadratic reads are then `a`.
The projected parts `e±*(a,0)` multiply to zero and sum to `(a,0)`; they need not themselves be
idempotents. Their two marginal indexes would be redundant. H3_40 stores one residue directory,
not duplicate links sold as independent exclusion.
A non-diagonal cell encoding or multi-prime profile needs a separate contract; no lossless cell
reconstruction or channel selectivity improvement follows from this scalar address. The accepted
single-prime-power quadratic split remains `proved`; this encoding/locator is `anchor`.

**Logical layout:** one live full-key representative per `(Sigma,K)`, value stored once, and one
projected-value association per required factor, all referring to that same representative.
For encoded address `a`, association `(i, pi_i(a), representative)` is present, including at zero.
Zero residue is data, not vacancy. The profile/factor ordering is fixed across reads and writes.
For H3_40 the required scalar association is just the unique residue/identity path above; redundant
quadratic marginal links are not materialized. The multi-factor relation is a future extension,
still `anchor`; a scanned posting list is not the selected collision-resolution algorithm.

**Exact-map decision: the key is carried, not truly discarded.** The original key object may be
replaced by an independently proved lossless codec, but information sufficient to recover/compare
the whole key remains. A CRT address alone cannot distinguish different keys with the same residue.
Literal deletion with lossy reduction would change the observation to a quotient map and is not
authorized for `HMap (Sigma × K) V`. Key retention is decided by the corrected intent and this
assignment; the obsolete literal-erasure question is closed, not a pending lead decision.

Write `Live(t,r,q)` for a live representative `r` decoding to query `q=(sigma,k)` and
`Link(t,i,v,r)` for a live, in-bounds factor association. Require coverage
`Live(t,r,q) → ∀ i, Link(t,i,pi_i(address(q)),r)`, sound links back to the same live payload,
and uniqueness of decoded keys. Then:

```text
Present(t,q) iff ∃ r, Live(t,r,q) ∧ ∀ i, Link(t,i,pi_i(address(q)),r)
(∃ i, ¬ ∃ r, Link(t,i,pi_i(address(q)),r)) → ¬ Present(t,q)
```

The second implication follows from coverage, not from orthogonality alone. All projected values
being present proves nothing about a common representative. For `m=15`, factors `3,5`, stored
addresses `7` and `9` have projections `(1,2)` and `(0,4)`; query address `4` has `(1,4)` but
is absent. Even matching the whole address is insufficient: word keys `4` and `19` reduce to the
same residue. These are control fixtures, not a production modulus selection.

Construction establishes empty coverage. Fresh insert publishes all links and payload atomically
at the logical commit point; found update merges once, not once per factor. Relocation preserves
links without merge, rebuilding destinations before commit. Removal, if exposed, must preserve
coverage and remove/retarget stale links. Incomplete or stale channel indexes cannot certify absence;
an explicitly disabled optional index must use a complete, cost-admitted identity traversal instead.
This permits no linear collision-list fallback or bypass of the required H3_40 coverage invariant.

## Summary deletion: moots and retains

The deletion target is `Region.summary`, `summary_words`, `summary_trailing_zeros`, `summary_index`,
`summary_set`, `summary_maybe`, summary receipt/reservation fields and summary branches in every
read/write/prepared-shard/relocation/build consumer. Delete the `PHI` **summary hash**, not every
occurrence of `coord::PHI`: the parabolic fold and existing exact codec use it independently.
Their equations remain until explicitly replaced and re-proved.

| Obligation | Successor status and exact retirement condition |
|---|---|
| `summary_index place len = ok (some index) → index/64 < len` | Moot for the successor only after all summary reads **and writes** and their generated dependencies disappear. This includes shift bounds, word/bit indexes and the summary-only checked-sizing proof; not other indexing arithmetic |
| `SpillSummaryShape` | Moot after summary storage, constructor/receipt sizing and all accesses are removed |
| `SpillSummaryRecorded` | Moot after summary publication in fresh insert, prepared apply, growth, relocation and pending commit is removed |
| `SpillSummaryMissSound` | Moot after every absence branch ceases using `summary_maybe`; replaced by the per-face certificates above with source-bound coverage/preservation |
| `SpillFingerprintExact` | FingerprintExact is NOT mooted by summary deletion alone. The shifted seven-bit tag is independent. It remains while any tag-based equality filter executes; it becomes historical only after that filter and its writes are deleted/replaced and successor occupancy/scan laws pass |

**W4 scope (`APEIRON-ANATIVE-W4-A3`).** Summary deletion is HELD until both filtered faces are
wired and the lead releases the deletion slices. Retain `tag`, `TAG_LIVE`, `TAG_FINGERPRINT_SHIFT`,
`TAG_FINGERPRINT_MASK`, `scan::match_tag` and `SpillFingerprintExact`. Fingerprint retirement is not
part of this migration; the retained tag filter still requires full `(Sigma,K)` confirmation.
`coord::PHI`, spreading and the independent exact codec also remain. Delete the summary mixer,
not these consumers. Disabling filtering is not deleting the summary implementation. Retirement
requires call-graph/root inventory, fresh extraction, successor proofs and rejecting controls on
one identity, not merely a source-text search or a documentation edit.

### W4 certificate integration boundary

**A3 supersedes the hyperbolic-only and deletion-first boundary.** The standalone implementation
now has separate binary Parabolic and ternary Hyperbolic roots and refuses Elliptic locator writes.
`Absence::Absent(CoveredScope)` covers its own contents under its laws; `Unknown` includes present
keys and unavailable coverage. Before a table query can consume Absent, coverage must include
**every** possible representative of that `(Sigma,K)` in the claimed scope, plus explicit exclusion
of storage outside it. A fresh empty locator beside an already populated table is not coverage.

The consumed S0 contract is `proof/corpus/Native/LocatorSpec.lean`:
`TableStorageCoverage`, `TableCoverageTransitions`, `CertifiedDecision`,
`ReserveBeforeRetainContract`, `LocatorReservationReceipt` and `summaryReplacement`.
`table_absent_of_complete_path_missing` is conditional on table coverage; it does not establish
that coverage from Rust. The source adapter must provide constructor and per-transition preservation,
sound reverse handles, identity uniqueness and missing-path reflection for both carriers. Bind these
to `GrowableSearch.entryAt`/`Absent` on the summary-free successor; establish source directory facts
against S0's parameterized observation before read cutover, then prove that observation's equality
to Growable at the integration adapter. Do not import the known-red old summary proof as a
prerequisite, duplicate its search vocabulary or assume the desired output invariant.

**Per-face query rule:** Parabolic uses the complete 64-bit `a` residue from `fstHom`, never `b`;
Hyperbolic uses the complete 40-digit H3_40 scalar residue. For the existing word instance,
`ParabolicWord(a)` embeds as `ParabolicDual(a,0)`. This does not make the key-to-`a` encoding a ring
homomorphism. Hyperbolic scalar embedding `(a,0)` has equal channel reads, so one directory suffices.
Elliptic retains exact packing and exact-address coverage without a secondary locator; no finite
word-ring field instance follows from S0's abstract `EllipticExactAddress` field parameter.

On **both** filtered faces, covered Absent may return the default on reads after all other storage
is excluded. Unknown must continue exact identity search; it is neither presence, absence nor an
insertion vacancy. A negative query result supplies no first-fit destination: insertion must still
select a valid vacant slot/path extension. Keep full `(Sigma,K)` comparison after the retained
fingerprint; keep late-hit coverage, exact live length and default-valued occupancy. Unknown may
not fall back to the retired summary. A transitional finite sweep is correct under coverage, but
does not satisfy the final population-independent lookup bound; exact-terminal-to-payload access
and a charged insertion/rebuild recurrence remain necessary for the final cost claim.

**Storage/commit interface:** Table remains the sole owner of each payload/value. Refactor the
standalone trie into a shared directory engine with opaque bounded payload handles; Table uses
that engine, not a nested second `Locator<K,V>` value store. Reserve directory/path/reverse-link
storage together with destination payloads and retain buffers. Rebuilds stage the replacement
directory privately, relocate without merge, retarget handles and commit directory/payload together;
found updates preserve links. Prepared-shard and parallel assembly paths have the same obligation,
including handle rebasing and reservation before any worker callback. A boolean coverage flag is
not the source relation. Directory staging before query cutover is not accepted dual-map storage.

**Resource acceptance still needed:** the A4 read of I1 stages/reserves the edge path before
`retain_reserved` and exposes a late-edge seam. Consume I1's gate/control receipt; source presence
alone does not accept ordering, arena growth or the future Table adapter. A rollback does not erase
callback effects. The late-edge failure control must observe unchanged links/payloads/counts plus
zero retain/clone/merge/relocation on the actual path.
Account for binary nodes, ternary nodes, identity nodes, identity edges, representatives/handles,
reverse links, staging and retained-identity buffers in checked plans and capacity receipts.
Unextracted allocator realization is separate from this source-order requirement.

S0 alone does not resolve API admission; the lead's A4 explicit-capability decision above does.
Its strengthened sealed carrier instances and typed face domains replace the compatibility hold.
Automatic closure encoding, duplicate value stores, always-Unknown final adapters and reduced REG-1
populations remain unauthorized. Sol supplies the source-pinned coverage/cost adapter; a Rust trait
or the lead's API decision is not a theorem premise establishing it. Deletion remains held until
both filtered faces are wired with source evidence and rejecting controls.

The staged cutover keeps the genuine old Elliptic summary-assisted spill branches until the atomic
deletion slice, so no dead helper or artificial caller is needed between waves. It adds no Elliptic
locator and changes no exact result. The final slice removes those branches and their last helpers
together, leaving exact-address coverage or complete identity search; check its runtime cost rather
than assuming unchanged timings. The two filtered faces do not use summary after their cutovers.

The strict full-profile gate retains all incumbents and intervals `<=1.0`. In addition, matched
Parabolic absent-miss must not regress from the pinned summary baseline; Hyperbolic gains the
locator mechanism; Elliptic exact behavior stays unchanged with no locator work. These are acceptance
targets, not measured speed claims. Default Sigma strata are Parabolic; missing explicit other-face
populations block measurement readiness rather than being relabelled. W7 remains frozen.

Retained independently: canonical default/new-first/ordered semantics; faithful codec and `Sigma`
separation; uniqueness, initialized storage and bounds; complete candidate coverage; search termination,
found/absent/first-fit correctness; no duplicate merge; relocation and extent reclassification wherever
direct residency remains; exact live length; resources, reserve-all/no-clone-on-Err and refusal stutter;
partition/schedule/order; public-method linkage; full-state `RepR` preservation and source freshness.
The seven `W7-SearchCommitBoundary` entries remain in the discharge ledger:
`ResourceAdapterSimulates`, `SearchDecisionCorrect`, `DecisionSymbolicCorrespondence`,
`QueryAddressCorrect`, `FreshLengthBound`, `GrowableCallbackLaws`, `SpillVacantDirectGuard`.
If direct/spill geometry disappears, transport the last guard's coverage purpose to successor cases;
do not retain meaningless old indexes or silently erase the obligation.

## Representation and failure semantics

Proposed `FaceRepR` (absent target, not a renamed proved `RepR`) contains typed address denotations,
faithful carrier decoding, profile consistency, unique live representatives, finite complete per-face
candidate coverage, parabolic/hyperbolic live-link agreement, initialized bounded storage, exact count and
`∀ q, observe(t,q)=h(q)`. Stored default-valued keys still have occupancy; `get != default` is not
membership. No HashDOS assumption is a field of `FaceRepR`.

Required source-linked headlines: real constructor establishes `FaceRepR t (fun _ => default)`;
`get_at` denotes canonical `get`; successful `insert_with_at` preserves the entire relation to
new-first `insertWith`; relocation preserves the old map; successful `build_at` denotes ordered
`build`. `SizeOverflow` / `AllocationFailed` leave the committed state unchanged, invoke no
hold/clone/merge/relocation and return no partial build. Reserve all destination/link storage before
callbacks; the allocator realization exclusion does not excuse incorrect source ordering.

Require joint actual-constructor witnesses for both carriers at admitted faces: empty, found, fresh,
collision, growth, default-valued entry, unequal `Sigma`, overflow and late reservation refusal.
Translate old flat/spill cases explicitly. Per-key readback is not whole-state preservation.
`build_eq_of_key_streams_eq` requires identical per-key value sequences; channel independence does not
license reordered or duplicated arbitrary merges. Parallel consumers must preserve this same relation.

## Sol ratification obligations

Existing inputs below were read at resolved corpus `3d66ea3aa2fe1de5997b6b757247454a2f771358`
under `proof/corpus/.lake/packages/apeiron/proof/`. They keep their corpus-native `proved` scope;
no build is rerun here. The established word-ring obstruction, no cross-characteristic unital hom
and norm non-additivity are not reopened. Proposed declarations below are **absent proof targets**,
not new claimed theorems; Sol fixes their Lean types and joint witnesses before proof dispatch.

| Input declarations / exact hypotheses | Required consumer edge |
|---|---|
| `Apeiron.Trichotomy.AnyRing.ringIdempotent_trivial` in `Trichotomy/AnyRing.lean`: `IsUnit (2*z.1-1)`, `IsUnit z.1 ∨ IsUnit (z.1-1)`, `RingIdempotent sigma z` | Retain the established all-dial word specialization; do not request word halves or use it as an elliptic-field theorem |
| `AnyRing.ringUpper_channel`, `ringLower_channel` in `Trichotomy/ChannelsAnyRing.lean`: `[CommRing R]`, `r^2=sigma` | Ratify scalar CRT reductions separately from quadratic channels. If quadratic channels execute, source encoding must land in `RingCell R_H` and respect `ringMul`, not the componentwise product instance |
| `AnyRing.ringOfHalves_upper_lower`: `r*s=1`, `2*h=1`; `AnyRing.ringComplementary_idem` in `Trichotomy/IdempotentsAnyRing.lean`: additionally `r*r=sigma` | Instantiate the odd-ring joint witness at `sigma=1`; prove the per-factor transport and distinguish `d` scalar from `2d` quadratic channels |
| `AnyRing.ringNormForm_mul` in `Trichotomy/AnyRing.lean`: `[CommRing R]`; `Apeiron.Trichotomy.normForm_eq_zero_iff_of_neg` in `Trichotomy/Basic.lean`: `sigma:ℝ`, `sigma<0` | Multiplicativity remains applicable to the word kernel; real negative-norm definiteness does not become a word-address premise |
| Quadratic split over a single odd prime power `Z/pᵏ` (`r*s=1`, `2*h=1`, `r*r=sigma`), Sol T2 ratification accepted at grade `proved` | `AnyRing.ringUpper_channel`, `ringLower_channel`, `ringComplementary_idem`; `(Z/9)[x]/(x²−1) ≅ Z/9×Z/9`, `r=s=1`, `h=5`. The H3_40 source specialization remains open |
| Proposed canonical `Apeiron.Table.NativeAddress.crt_equiv`, `crt_idempotents`, `crt_projection_reconstruct` | `d≥2` odd primes, positive exponents, pairwise coprime powers, product modulus; ring equivalence, units, orthogonality, sum-one and reconstruction. Joint `m=15` witness; scalar CRT only, no claim the quadratic theorems prove it. Anchor until a canonical declaration exists |
| Proposed bridge `Native.Address.pack_roundtrip`, `home_face_denote`, `held_address_agrees` | Actual `pack` width/shift bounds, encode/decode on image, immutable valid profile, stable equality-respecting encodings, faithful clone/matches; prove generated bodies, not formula copies. Both carriers at each admitted face |
| Proposed canonical `Apeiron.Table.NativeMembership.absent_of_projected_missing`, `present_iff_common_witness` | Exact live/key relation, unique representatives, sound links, every representative covered in every factor; presence iff one common full-key witness. Source bridge must establish these from constructor and preserve them, not assume the post-state |
| `Apeiron.Table.{HMap,get,insertWith,build,get_insertWith_self,get_insertWith_ne,get_build_acc,build_eq_of_key_streams_eq}` in `Table/Basic.lean` | Source-linked `Native.Read.get_at_refines`, `Native.Write.apply_preserves`, `Native.Rebuild.preserves`, `Native.Build.refines` targets for both carriers, whole-state success/refusal/public links. Exact dependencies and audit registration required before W7 reopens |
| No adversarial table cost declaration in these inputs | The cost proposition below, source-linked to the chosen finite locator; mathematics and runtime measurements remain separate |

`AnyRing` abbreviates `Apeiron.Trichotomy.AnyRing` only in this table. Canonical generic CRT/membership
edges belong to the canonical owner, not a second generic mathematics library in the Rust bridge.
Any new release/pin transport is a separate authorized task; no dirty-sibling import is permitted.

## Adversarial cost proposition

Ratification must choose and prove an explicit worst-case bound, not “no hash ⇒ no HashDOS”. A
concrete sufficient target is: for every valid profile, every reachable state with `n` live keys and
every adaptive sequence of `T` admitted operations, independent of key distribution, projections,
addresses, insertion order or pepper knowledge, the source work excluding explicitly charged external
callbacks/allocation satisfies

```text
W(sequence) + Phi(t_final) ≤ Phi(t_initial)
    + C(profile) * T * (1 + ceil(log2(n_max + 1))) + C(profile) * B + A + F
```

Here `n_max` is the maximum live-key count during the sequence. The nonnegative, source-linked
potential `Phi` has value zero at the real empty constructor and an explicit storage-dependent bound;
it accounts for prepaid rebuilding in an arbitrary reachable initial state. This is not the deleted
`PHI` multiplier. It cannot be an assumed cost budget. From empty, dropping the final nonnegative
potential gives the aggregate bound without initial credit.
`W` counts comparisons, bytes inspected/copied, index steps, link maintenance and rebuild work;
`B` counts input key bytes read once, `A` separately charges allocator realization, and `F` actual
callback work under registered callback bounds. Repeated comparisons/copies of long keys are **not**
hidden in `B`, `A` or `F`. The proposal is a sufficient target, not an established complexity claim;
Sol may state a different explicit subquadratic-in-`T` guarantee with the same threat/domain and
lead approval, but not an occupancy assumption or an average-case statement.

Also require worst-case individual lookup work independent of an attacker-controlled fiber size,
and a charged aggregate rebuild/insertion recurrence (a linear single rebuild is permissible only
with a proved aggregate bound). For fixed-size keys/callbacks, building `T` distinct adversarial keys
must not take `Θ(T²)` internal work. Constants may depend on a frozen profile's declared size, not
hide occupancy or an exponential full-address-space allocation. State space use and modulus/channel
cost explicitly. Exercise same-residue keys, all-positive mixed witnesses, zero components, skew,
repeated keys, long common prefixes and resource refusals. CRT alone bounds none of these costs.
The pepper stays until this theorem is source-linked and accepted; its presence is not the theorem.

## Verification and trust

No Rust test, Lean build or benchmark runs in founding. All new boundaries are `anchor` / open.
Real-source extraction, fail-closed `{propext, Classical.choice, Quot.sound}` audit, actual reachable
witnesses and semantic mutation controls are required. Slang's empty cone remains separate.
Complete AGENTS §6 barriers, the `10^8` adversarial identity row and strict full-profile REG-1 with
every applicable interval maximum `<= 1.0` remain mandatory. No comparison row is removed because
its face has unresolved admission. Measurement of modular arithmetic/storage overhead is missing.

Allocator/capacity realization, executor/`std::thread::scope`, arbitrary callback panic/allocation,
rustc/LLVM, ABI, GPU/SPIR-V and the unsafe container remain outside the proof claim. Sequential
search/commit, source reservation order and full-state preservation are not excused by those exclusions.

## Lead decision boundary

No new lead design question remains: explicit-capability admission, per-face rings,
single-prime-power quadratic splitting and full-key retention are settled inputs. Concrete
population adapters, source-linked exclusion/presence/cost, safe resource handles, extraction
feasibility and all W7 obligations remain open. An impossible adapter returns its exact population
and missing law; it does not silently change equality, weaken the cost proposition or reopen the
rejected generic compatibility design.
