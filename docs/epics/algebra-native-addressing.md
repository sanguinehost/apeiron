# Algebra-native addressing and channel decomposition

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

Merged epic record: `intent/algebra-native-addressing.md`, `spec/algebra-native-addressing.md`, `plan/algebra-native-addressing.md` (one file per epic; see `docs/README.md`).

## Intent — Algebra-native addressing and channel decomposition

- **Owner:** lead (direction); Astra (architecture founding); Sol (ratification)
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; working tree dirty
- **Status:** approved for founding — per-face address rings
- **Spec:** `docs/epics/algebra-native-addressing.md` (Astra founding; prime-power split and key-retention corrections applied)
- **Plan:** `docs/epics/algebra-native-addressing.md` (Astra founding; implementation rows are reservations)

### Problem

The growable table places a key by free geometry: the shard is the low byte of the coordinate
(`src/table/state.rs`, `shard`); a region is rounded to a power-of-two group count from an observed
load (`slots_for`, `src/table/state.rs:328`); and absence is filtered by a Bloom summary over
`place >> 8` with a golden-ratio multiplier and a runtime word width (`summary_index`,
`src/table/state.rs:127`). The canonical object is a total function whose semantics leave all of this
unobserved: "an implementation may sort by home address, shard across cores, or take arrival order,
and get the same function" (`proof/corpus` canonical `Apeiron/Table/Basic.lean:11,25-26,71`). Every
geometry value that reaches a shift or an index therefore becomes a partial operation in the Aeneas
rendering with no algebraic statement to discharge it at the use site. The class has recurred: the
unmodeled `Vec::is_empty`/`usize::trailing_zeros` intrinsics (avoided in `src/table/state.rs:13`), the
`summary_index` shift overflow (totalized, which moved the obligation rather than removing it), the
nested-loop return in `spill_read` (single-exit), the borrowed-carrier context-match failure, and the
`summary_index` dynamic-shift bound now blocking `Growable/Search` and `Growable/Relocation`. The
placement is also an adversary-influenceable hash surface, which is exactly the property the
programme claims the primitive removes.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Patch each Aeneas obstruction in the source | Relocates a free-geometry safety obligation; the next geometry feature regenerates it. The recurrence is the evidence. |
| Prove the dynamic-shift bound with a one-off spec lemma | Same bucket-2 obligation, and leaves the golden-ratio hash of `place` on the proved read path. |
| Type the geometry (const-generic region/summary width) | Removes the shift but keeps arbitrary logical partitioning; the partition and its security properties stay un-derivable from the primitive. |
| Scope the geometry executor as tested/unverified | Renegotiates "every executing face refines the canonical object" and keeps adversary-influenceable placement. |
| Compact hot lookup directory (Astra T1) | Adds locator/coverage/collision contracts on top of the same arbitrary partition; a performance patch over free geometry, not a derivation. |

### Completion condition

Every executing face places a key by the primitive's canonical reduction, with no hash and no free
power-of-two geometry, and the read/write faces refine `Apeiron.Table` under the closed axiom cone.

```sh
cargo test --workspace --all-features --locked \
  && cargo clippy --workspace --all-features --all-targets --locked -- -D warnings \
  && proof/regen.sh --check \
  && (cd proof/corpus && lake build) \
  && cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

### Invariants

1. The primitive is the Clifford algebra of the rank-one form `q = ⟨σ⟩`, `Cl(σ) ≅ R[x]/(x²−σ)`, with
   norm form `N(a+bx) = a²−σb²`, multiplicative: `N(uv) = N(u)·N(v)`.
2. The address structure is per face, as `src/coord.rs:36-47` already states ("a single fold cannot be
   right for all three"):
   - **elliptic** (σ<0, regenerated): the address determines the key, so the address ring stays
     `Z/2⁶⁴` and the fold is the bijection `pack`. A modulus-based reduction cannot be injective on the
     full key space, so it is excluded here.
   - **parabolic** (σ=0, carried): the key is carried beside the address, so the address ring stays
     `Z/2⁶⁴` with the spreading fold.
   - **hyperbolic** (σ>0, discard): the address does not determine the key, so the address ring is an
     odd prime power `Z/q` (`q = pᵏ`, `p` odd, `2` a unit) and the quadratic idempotents
     `e± = (1 ± x/r)/2` (`r` a unit, `r² = σ`) are the two crosstalk-free channels, `e₊e₋ = 0`. A
     composite `m = ∏ qᵢ` of coprime factors additionally gives scalar CRT channels; the `≥2`-factor
     condition is required for the scalar decomposition only, not for the quadratic split. The full key,
     or a lossless codec regenerating it, is retained: "discard" is an address property, not key
     erasure, because exact-map identity requires the full `(Sigma, K)` witness.
3. The split exists exactly where the algebra supplies the witnesses `2 ∈ R^×` and a unit `r² = σ`.
   `Z/2ᵏ` has nonunit `2`, so the split is unrealizable there for every σ. A single odd prime power
   splits: `(Z/9)[x]/(x²−1) ≅ Z/9 × Z/9` via `r = s = 1`, `h = 5`; locality forbids nontrivial *scalar*
   idempotents, not the quadratic extension's `e±`. σ=0 stays local (`R[ε]/ε²`, a filtration); an
   irreducible `x²−σ` over a prime field is the unsplit field case.
4. On the hyperbolic face the component strata are necessary-condition descriptors: absence of the
   query's projected component implies absence of the query (sound exclusion), while presence of every
   component does not establish presence. Presence requires a common full-key witness on one stored
   representative, so components of different keys cannot be combined into a false positive. On the
   elliptic and parabolic faces presence is decided by the carried or decoded `(Sigma, K)` identity.
5. The `PHI` summary hash over `Z/2⁶⁴` (`summary_index`, `src/table/state.rs:127`) is deleted: it
   filters against free geometry with no algebraic statement and is the source of the dynamic-shift
   class.
6. Full `(Sigma, K)` identity decides presence. Address, channel, level and any fingerprint are
   placement or exclusion filters only; a false negative is forbidden.
7. The canonical semantics is unchanged: `get`/`insertWith`/`build` and the absence default
   (`Apeiron/Table/Basic.lean`); `Cell<i64>` remains the value/key word kernel (`word_mul`,
   `word_norm_denote`).
8. Every HashDOS claim carries a source-linked adversarial worst-case cost theorem. Removing the
   unkeyed hash does not by itself bound adversarial occupancy, and the pepper stays until the theorem
   exists.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| `N` is multiplicative (`N(uv) = N(u)·N(v)`). | proved | `ringNormForm` and its multiplication law. |
| `N` is not additive, so it is not an address homomorphism. | proved | counterexample `N(1+1) = 4 ≠ 2`. |
| `Z/2ᵏ` is local for every σ, with only `0,1` as idempotents; no CRT split. | proved | canonical `Aeneas`-adjacent `AnyRing.ringIdempotent_trivial`; Sol ratified. |
| An odd prime power `Z/pᵏ` (p odd, unit `r² = σ`) splits the quadratic algebra through `e±`; a composite odd `m` additionally gives scalar CRT channels. | proved | `AnyRing.ringUpper_channel`/`ringLower_channel`/`ringComplementary_idem`; `(Z/9)[x]/(x²−1) ≅ Z/9×Z/9`; Sol ratified. |
| Per-face address rings: elliptic/parabolic `Z/2⁶⁴`; hyperbolic `Z/m`. | anchor/open | `Coord`/`Home` contract change; Sol ratification. |
| Hyperbolic absence of a query's projected channel implies absence of the query (sound exclusion). | proved target | projection is an endomorphism and `eᵢeⱼ = 0`; shared-representative obligation. |
| Presence requires a common full-key witness, not the conjunction of components. | proved target | `RepR`/`get_at`; `(Sigma,K)` identity. |
| Adversarial worst-case cost is bounded without an unkeyed hash. | anchor/open | source-linked worst-case cost theorem; not implied by CRT or by removing the hash. |
| Parity without the `PHI` summary. | measured target | strict full-profile command, named machine and interval. |

### Non-goals

- No threshold relaxation, historical envelope, aggregate pass, or reduced matrix.
- No change to the canonical object: `Apeiron.Table` stays a function.
- No GPU/SIMD/ISA specialization.
- No discharge of `W7-SearchCommitBoundary` from performance evidence.
- No new arbitrary partition (shards by low bits, power-of-two regions, Bloom summaries) as placement.

### Trust boundary

Evidence will not reach the allocator, executor/`std::thread::scope`, rustc/LLVM codegen, the kernel
ABI, GPU/SPIR-V, or the `unsafe` container. The spinor/double-cover reading of the unit-group action on
`N` is an `anchor` until a named corpus theorem states it; the algorithmic realization and its resource
bounds are not proved by the algebra alone.

### Open questions

- **Decided (lead):** per-face address rings. `Z/2⁶⁴` for the elliptic and parabolic faces; an odd
  composite `Z/m` with CRT idempotent channels for the hyperbolic face only; the `PHI` summary deleted;
  `Cell<i64>` kept as the value/key word kernel. The wholesale scalar pivot is rejected: a modulus-based
  address cannot be the elliptic bijection, a finite ring has no sign trichotomy, and modular reduction
  costs the hot path.
- Sol: ratify the CRT idempotent split at its exact hypotheses, the hyperbolic sound-exclusion
  obligation (shared representative, live links, complete coverage, preservation), and the adversarial
  worst-case cost statement, against the pinned corpus; name the declarations.
- Astra: revise `docs/epics/algebra-native-addressing.md` and `docs/epics/algebra-native-addressing.md`
  to the per-face boundary, the `Coord`/`Home` contract change, the serialized tasks with one gate and
  one rejecting control each, and the deletion of the summary read path.
- Lead: sequencing against the frozen `W7` and the pending summary bridge repair.

## Specification — Algebra-native addressing and channel decomposition

> **Kani retirement.** The [user-directed retirement](kani-retirement.md) supersedes
> every Kani run, preservation task and barrier requirement below. Historical receipts remain
> revision-bound evidence; current work uses ordinary Rust gates and semantic mutation controls.

- **Owner:** Astra architecture; Sol mathematical/refinement ratification; lead admission/closure
- **Intent:** `docs/epics/algebra-native-addressing.md`, approved per-face direction
- **Base:** `c450c222905c050f793a23332708c230c658f82c`; dirty supplied checkout
- **Extraction decision:** `APEIRON-ANATIVE-AEN-PIPE`; explicit monomorphizing candidate plus
  mutable-reference-free aggregate wiring; real-source acceptance OPEN, one production implementation.
- **Decision:** `APEIRON-ANATIVE-W4-A4`; lead selects the explicit-capability public API break,
  not preservation of opaque `Eq`/arbitrary placement closures. Both-face wiring precedes held
  summary deletion; source-linked adapter readiness remains open. Architecture
  grade `anchor`. Earlier prospective API paragraphs retain their founding scope; the standalone
  address/locator implementation is not table integration evidence.

### Extraction pipeline decision — APEIRON-ANATIVE-AEN-PIPE

**Select explicit Charon monomorphization AND remove mutable references from translated aggregate
types.** Grade `anchor`, not a green extraction receipt. This supersedes AEN-DIR's next dispatch,
old loop-error status and prohibition on selecting a configuration change; its scalar arenas,
typed faces, coverage, reservation, summary-deletion hold and W7 freeze remain. The assignment
reports that scalar-arena conversion clears directory errors. The remaining supplied failures are
`Could not find: type_var_id: 0/2 from ExtractBase.Item` at `reserve_rebuild`'s generic region loop
and `materialize_rebuild`'s generic initialization loop, and `SymbolicToPure.translate_type_decls`
CFailure on `Option`/`Result`/`ControlFlow` containing mutable references. These are tool failures,
not counterexamples to the contracts. No heavy build runs in this decision.

#### Exact configuration and semantics delta

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

#### Mutable-reference-free aggregate wiring

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

#### Generic bodies and reservation contract

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

#### Re-baseline protocol

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
5. Integration still requires the complete Rust/proof/control barrier. No source coverage,
   end-to-end table correctness, HashDOS bound or speed claim follows from extraction alone.
   Summary deletion remains held; W7's original reopening condition is unchanged.

### Historical directory extraction decision — APEIRON-ANATIVE-AEN-DIR

**Decision: one production directory, scalar arenas; translation feasibility OPEN.** This section
supersedes the physical directory shape and next extraction dispatch in A4, not its typed face API,
coverage, reservation contract, deletion hold or W7 freeze. Grade `anchor`. No build is authorized
in this documentation task. No supplied successful extraction establishes that the replacement
below is accepted by the pinned tools; calling it an accepted Aeneas shape would invent evidence.
The plan makes conversion plus real-source extraction the next bounded feasibility wave, before
further integration. Everything executing remains a refinement target, not a tested-only escape.

#### Failure boundary and selected shape

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

#### Contracts that conversion must preserve

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

#### Tool and warning decision

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

### Explicit-capability API (A4, authoritative migration contract)

This section supersedes earlier compatibility holds and prohibitions on adding identity bounds in
this spec and the A3 plan. It does not claim the interface is implemented. One `Table`, canonical
`HMap (Sigma × K) V`, compile-time carriers and borrowed total reads are retained.
There is **no final generic compatibility table, implicit hash/sweep fallback, or runtime
UnsupportedFace result**. A home selects a sealed face domain: `Sigma` (all three faces) or new
`LocatorSigma` (Parabolic/Hyperbolic only). The latter is an explicit capability-constrained path
on the **same Table**, not the rejected generic fallback. Both domains default to Parabolic.
Logical identity still carries the ordinary `Sigma` obtained from the domain token.

#### Concrete traits and bounds

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

#### Face and population policy

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

#### Public-surface break inventory (all 45 generic_api rows accounted for)

The ledger is `docs/evidence/anative.md`, §1; row names below, not its moved line
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

### Per-face address contract

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

#### Coord / Home / TableAddress boundary

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

### Hyperbolic layout and identity

#### Concrete profile H3_40

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

#### Finite hyperbolic locator

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

### Summary deletion: moots and retains

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

#### W4 certificate integration boundary

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

### Representation and failure semantics

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

### Sol ratification obligations

Existing inputs below were read at resolved corpus `3d66ea3aa2fe1de5997b6b757247454a2f771358`
under `proof/corpus/.lake/packages/apeiron/proof/`. They keep their corpus-native `proved` scope;
no build is rerun here. The established word-ring obstruction, no cross-characteristic unital hom
and norm non-additivity are not reopened. Rows below distinguish existing conditional declarations
from absent proof targets. Existing declarations retain only their exact audited scope; absent
declarations and premises remain open.

| Input declarations / exact hypotheses | Required consumer edge |
|---|---|
| `Apeiron.Trichotomy.AnyRing.ringIdempotent_trivial` in `Trichotomy/AnyRing.lean`: `IsUnit (2*z.1-1)`, `IsUnit z.1 ∨ IsUnit (z.1-1)`, `RingIdempotent sigma z` | Retain the established all-dial word specialization; do not request word halves or use it as an elliptic-field theorem |
| `AnyRing.ringUpper_channel`, `ringLower_channel` in `Trichotomy/ChannelsAnyRing.lean`: `[CommRing R]`, `r^2=sigma` | Ratify scalar CRT reductions separately from quadratic channels. If quadratic channels execute, source encoding must land in `RingCell R_H` and respect `ringMul`, not the componentwise product instance |
| `AnyRing.ringOfHalves_upper_lower`: `r*s=1`, `2*h=1`; `AnyRing.ringComplementary_idem` in `Trichotomy/IdempotentsAnyRing.lean`: additionally `r*r=sigma` | Instantiate the odd-ring joint witness at `sigma=1`; prove the per-factor transport and distinguish `d` scalar from `2d` quadratic channels |
| `AnyRing.ringNormForm_mul` in `Trichotomy/AnyRing.lean`: `[CommRing R]`; `Apeiron.Trichotomy.normForm_eq_zero_iff_of_neg` in `Trichotomy/Basic.lean`: `sigma:ℝ`, `sigma<0` | Multiplicativity remains applicable to the word kernel; real negative-norm definiteness does not become a word-address premise |
| Quadratic split over a single odd prime power `Z/pᵏ` (`r*s=1`, `2*h=1`, `r*r=sigma`), Sol T2 ratification accepted at grade `proved` | `AnyRing.ringUpper_channel`, `ringLower_channel`, `ringComplementary_idem`; `(Z/9)[x]/(x²−1) ≅ Z/9×Z/9`, `r=s=1`, `h=5`. The H3_40 source specialization remains open |
| Proposed canonical `Apeiron.Table.NativeAddress.crt_equiv`, `crt_idempotents`, `crt_projection_reconstruct` | `d≥2` odd primes, positive exponents, pairwise coprime powers, product modulus; ring equivalence, units, orthogonality, sum-one and reconstruction. Joint `m=15` witness; scalar CRT only, no claim the quadratic theorems prove it. Anchor until a canonical declaration exists |
| Existing `Native.Address.pack_roundtrip` proves generated `I64` pack/unpack; `home_face_denote` is proved and its generated `H3_40` profile premise is inhabited by `h3_40_new_ok`, with `home_face_joint_witness` covering all three faces at a common `I64` key; `held_address_agrees` (`Native/LocatorSpec.lean:1134`) and `held_address_joint_witness` (`:1161`, the same Parabolic key across Stored and Regen through the private `spread_zero`) are proved under the closed cone, and the module gate `lake build Native.LocatorSpec` is green (3350 jobs) with a generated-address mutation control (mutated `ParabolicDual.new a.place 0#u64` to `ParabolicDual.new 1#u64 0#u64`, rejected on the zero key, restored) | width/shift bounds beyond pack/unpack, stable equality-respecting encodings, faithful clone/matches, and joint-witness coverage at the other admitted faces remain open |
| Proposed canonical `Apeiron.Table.NativeMembership.absent_of_projected_missing`, `present_iff_common_witness` | Exact live/key relation, unique representatives, sound links, every representative covered in every factor; presence iff one common full-key witness. Source bridge must establish these from constructor and preserve them, not assume the post-state |
| `Apeiron.Table.{HMap,get,insertWith,build,get_insertWith_self,get_insertWith_ne,get_build_acc,build_eq_of_key_streams_eq}` in `Table/Basic.lean` | Source-linked `Native.Read.get_at_refines`, `Native.Write.apply_preserves`, `Native.Rebuild.preserves`, `Native.Build.refines` targets for both carriers, whole-state success/refusal/public links. Exact dependencies and audit registration required before W7 reopens |
| No adversarial table cost declaration in these inputs | The cost proposition below, source-linked to the chosen finite locator; mathematics and runtime measurements remain separate |

`AnyRing` abbreviates `Apeiron.Trichotomy.AnyRing` only in this table. Canonical generic CRT/membership
edges belong to the canonical owner, not a second generic mathematics library in the Rust bridge.
Any new release/pin transport is a separate authorized task; no dirty-sibling import is permitted.

### Adversarial cost proposition

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

### Verification and trust

No Rust test, Lean build or benchmark runs in founding. All new boundaries are `anchor` / open.
Real-source extraction, fail-closed `{propext, Classical.choice, Quot.sound}` audit, actual reachable
witnesses and semantic mutation controls are required. Slang's empty cone remains separate.
Complete AGENTS §6 barriers, the `10^8` adversarial identity row and strict full-profile REG-1 with
every applicable interval maximum `<= 1.0` remain mandatory. No comparison row is removed because
its face has unresolved admission. Measurement of modular arithmetic/storage overhead is missing.

Allocator/capacity realization, executor/`std::thread::scope`, arbitrary callback panic/allocation,
rustc/LLVM, ABI, GPU/SPIR-V and the unsafe container remain outside the proof claim. Sequential
search/commit, source reservation order and full-state preservation are not excused by those exclusions.

### Lead decision boundary

No new lead design question remains: explicit-capability admission, per-face rings,
single-prime-power quadratic splitting and full-key retention are settled inputs. Concrete
population adapters, source-linked exclusion/presence/cost, safe resource handles, extraction
feasibility and all W7 obligations remain open. An impossible adapter returns its exact population
and missing law; it does not silently change equality, weaken the cost proposition or reopen the
rejected generic compatibility design.

## Plan — Algebra-native addressing and channel decomposition

> **Kani retirement.** The [user-directed retirement](kani-retirement.md) supersedes
> every Kani run, preservation task and barrier requirement below. Historical receipts remain
> revision-bound evidence; current work uses ordinary Rust gates and semantic mutation controls.

- **Owner:** Astra architecture; Sol contracts/proofs; Flash bounded implementation; lead closure
- **Intent:** `docs/epics/algebra-native-addressing.md`
- **Spec:** `docs/epics/algebra-native-addressing.md`
- **Task:** `APEIRON-ANATIVE-AEN-PIPE` (previous decision: `APEIRON-ANATIVE-AEN-DIR`)
- **Base:** `c450c222905c050f793a23332708c230c658f82c`; supplied checkout dirty
- **Status:** directory conversion clears its errors per assignment; generic-body/type-declaration
  extraction BLOCKED; explicit monomorphizing candidate selected; deletion HELD; W7 frozen

### Current extraction dispatch — APEIRON-ANATIVE-AEN-PIPE

Select the exact option list and source constraints in the spec's `Extraction pipeline decision`.
This supersedes AEN-DIR's current-status/next-dispatch and lead-edit map, not its contracts or
historical receipts. Do not repeat the completed scalar-arena conversion or S0 contract definitions.
One executing generic implementation remains; translation specializes it, not a handwritten clone.
No full-source translation, Lean build, runtime test or performance measurement runs here.

```text
SCOPE: docs/epics/algebra-native-addressing.md; docs/epics/algebra-native-addressing.md;
       extraction configuration, aggregate-borrow constraints, serial repair and lead edit map.
REGISTER: architecture/planning; excludes production, proof, tool and governing-document edits.
CORPUS: Native/LocatorSpec.lean, table_absent_of_complete_path_missing:
        TableStorageCoverage + missing complete path -> no live representative;
        conditional contract, not source-linked coverage.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Native/LocatorSpec.lean:548-602
GATE: complete-file whitespace wrapper below, repository root; baseline exit 0.
      Supplied regen red: type_var_id 0/2; translate_type_decls CFailure for mutable-reference ADTs.
GRADE: anchor before/after; no new proved/tested/measured implementation result.
```

Owner Astra; supplied-checkout documentation exception; base
`c450c222905c050f793a23332708c230c658f82c` plus preserved unrelated dirty work. Only the two lifecycle
files are owned. No worktree/cache owner, permission change, commit or governing edit is made.

#### Evidence and decision limits

- Supplied failure locations match the read real bodies: `src/table.rs:803-815` allocates generic
  `Region<R,V>` in `reserve_rebuild`; `:918-921` initializes them in `materialize_rebuild`.
  `src/table/locator.rs:435-438` constructs a fallible mutable-reference result; `:480,493` use
  `as_mut`; `:616,644` use `get_mut`. Changing these wrappers does not itself eliminate R/V binders.
- Current `Native/LocatorSpec.lean:548-563` has six coverage fields, including scalar shape/path
  fields; `:1242-1269` already distinguishes committed state from scratch/capacity changes. Its
  conditional witnesses are not evidence that Rust establishes the relation. Preserve these actual
  declarations rather than dispatching the obsolete four-field/S0 request again.
- Read pinned Charon source in
  `/tmp/opencode/charon-b104e24fea7d721b71e6c39fd70f26ff20bc0980/charon/src/`: `options.rs:68-85`,
  `:205-227`, `:482-497`, `:549-590`; `bin/charon/main.rs:225-246`;
  `bin/charon-driver/driver.rs:325`; `transform/normalize/partial_monomorphization.rs:1-12`.
  The explicit set retains eleven preset flags and omits precisely the three incompatible flags.
- Lightweight probe command: `timeout 30s target/aeneas/charon rustc <the spec's exact option list>
  -- --version`. It reaches rustc `1.100.0-nightly (8fa1c96cf 2026-08-17)` and exits **2** with
  `ERROR Code failed to compile`; no source unit is compiled and no extraction is accepted. Adding
  `--no-gen-tuple-structs` exits **1**, rejecting mono/tuple incompatibility; adding
  `--index-to-function-calls` exits **1**, rejecting mono/ops/index incompatibility. The candidate
  includes `--monomorphize-mut=all`; its syntax is recognized, but actual transformation acceptance
  remains MISSING. No inference from this probe to source-provability is permitted.

Read-input SHA-256 (dirty identity, not an execution receipt):

| Input | SHA-256 |
|---|---|
| `src/table.rs` | `78bc20c91b6b6f29503e702dd0df3a25613e9c936741899683b8cacecee51ecd` |
| `src/table/locator.rs` | `5c61f02bcc634d66bf3312bdc5cb7cbc776d2ad3dfe9d51aa2b9a8cbb72a438a` |
| `src/coord/locator_core.rs` | `b4699816521d89623988e4e92ee687b33d7243b3a70634bfbff1c8cd7e00eee8` |
| `proof/regen.sh` | `898aa30927b149bd49af6e38ce5b9f5def096ac8e3157146c1e0baf5ffdabdf6` |
| `proof/extract/lib.rs` | `94cc97cabe97d790b669dd44958025735694bd6a2c60b9fa7849cedba2b7e5b1` |
| `proof/corpus/Native/LocatorSpec.lean` | `8de10bc5361d6f2268c2e4e63c21edd5b4df651a2a8ebc83f9ac959f549acfa6` |
| `target/aeneas/charon` | `747e165bc62bb963f8d741e8c193c721a4b94555bcb1f9f86c79090fd7a59d0e` |
| `target/aeneas/charon-driver` | `207b223af35cf9a05227b329819b73532e94f655ccdefd64e77b755a15fdf297` |
| `target/aeneas/aeneas` | `33d0ce1a1cc2f1d3935c0cd2ab34f283797da61c531ed6d4e09d5c97b0d0228a` |

#### Serialized repair waves

Every row below is a singleton wave, prefix `APEIRON-ANATIVE-AEN-PIPE-`. Sequential reuse of a path
requires explicit owner release; there are **no overlapping active owners**. Implementation uses
isolated task worktrees/private targets, after the lead captures the dirty source identity. Shared
extraction/Lean products serialize on `/tmp/opencode/apeiron-lane.lock`; single thread, 4G memory,
zero swap, 900s timeout as E/L below. No unbounded reduction. A timeout/OOM is inconclusive failure,
not a rejecting semantic control. New controls must execute a nonzero test/theorem population.

| Wave / task / owner | Owned files and completion | One gate | One rejecting control / invariant |
|---|---|---|---|
| 0 / S1 / Sol | These two lifecycle docs only, after Astra release: ratify specialization semantics/root inventory, generic-claim transport obligation and actual coverage/reservation projection. Name exact affected bridge targets before their dispatch; do not weaken any theorem. | D below plus the explicit signed-off verdict in the plan | Documentary fixture deleting the generic-quantifier preservation requirement must be rejected in verdict review; not a kernel proof. Keep C/T, full domain and both axiom cones. |
| 1 / F1 / Flash | `src/table.rs`, `src/table/locator.rs`, `src/table/tests.rs`: apply the spec's local direct-match/index mutation wiring. First move unchanged cold rebuild adapter methods into the existing included locator file if necessary to meet 1,500 lines (read parent is 1,578; locator is 1,091); no new module or implementation. Inline tests cover missing/out-of-range handles, mismatched-old no-op, mixed faces, both carriers and late refusal. | R below | Omit a relocated live payload's reverse-link retarget; actual constructor/growth coverage/readback assertion fails, restored passes. Preserve every C field, callback counts and T; no root deletion or read cutover. |
| 2 / E1 / Flash | `proof/regen.sh`, `proof/extract/lib.rs`, `proof/ApeironWord.lean`: use the explicit option set, retain root inventory and all source digests, regenerate both carriers; inspect LLBC/declarations for forbidden reference ADTs, native operations and fully substituted loops. Thin wrappers only if an enumerated instance is missing. | E below | Modify one generated link-write result in a disposable copy; `--check` rejects drift, restored passes. Freshness control only, not semantic proof. First unsupported primitive stops dispatch. |
| 3 / S2 / Sol | These two lifecycle docs only: from the successful generated diff, freeze the exact old-to-new declaration/root/instance map, unchanged theorem statements and ≤3-file bridge transport tasks. Include a finite parametric-refinement discharge map; monomorphic word proofs cannot replace generic obligations. | D plus recorded source-pinned verdict | Review fixture replacing an all-N/generic claim by word/N=16-only acceptance must reject. No silent contraction of the claimed domain. |
| 4 / P-resource / Sol, conditional on S2 | `proof/corpus/Growable/Resources.lean`, `proof/corpus/Growable/Representation.lean`, `proof/corpus/AxiomAudit.lean`: transport generated type/dictionary consumers at unchanged statements and register every changed headline. If S2 identifies other blocking modules, sequence separate ≤3-file rows before this one, not unowned edits. | L(`lake build Growable.Resources Growable.Representation AxiomAudit`) | Change a generated publication receipt to the wrong capacity/extent; a named resource theorem rejects semantically, restored passes. Preserve actual receipt and committed-state interpretation. |
| 5 / P-native / Sol, conditional on S2 | new `proof/corpus/Native/LocatorSource.lean`, `proof/corpus/Native/LocatorCost.lean`, `proof/corpus/lakefile.lean`: existing source-link obligation, not a new axiom; inline fail-closed headline audits, constructors/transition coverage and charged work for the accepted bodies. | L(`lake build Native.LocatorSource Native.LocatorCost`) | Omit a generated terminal relink but keep its live payload; source coverage preservation rejects, restored passes. Input invariants cannot assume output C or the desired cost bound. |
| 6 / P-audit / Sol | `proof/corpus/AxiomAudit.lean` only: register Native headlines and run complete bridge regression after every S2 transport slice is accepted. | L(`lake build`) | Substitute an extra axiom for a registered headline in a disposable copy; fail-closed audit rejects, restored passes. Rust cone exactly bounded by `{propext, Classical.choice, Quot.sound}`. |

**Conditional rows are not dispatch-ready proof tasks.** The generated diff does not yet exist;
exact additional bridge filenames, statements and semantic control targets are MISSING until S2.
Sol must explicitly account for `Denotation`, `WordRefinement`, identity, Search/Relocation/Rebuild
and all other existing consumers, not just the two illustrative resource modules. If `proof/`'s
own build fails inside E because of changed tuple/numeric representations, return that exact goal
and an owned ≤3-file transport task before retrying E; do not accept a translation-only E exit or
edit a fourth file. Likewise an unavailable native operation, allocator primitive or loss of a
parametric source link calls for a bounded pinned-tool/model task, not endless source reshaping.

F1 and E1 form one unaccepted repair checkpoint: the supplied extraction red may persist after F1,
but no unrelated source task runs between them or beyond a failed E1. Lead application of the
roadmap qualification below is required before using this narrow scheduling exception. Recheck
source-size caps after formatting; if moving the unchanged adapters plus wiring cannot fit the
three files, stop and request a size-only slice rather than exceed ownership. Do not rerun AEN-DIR
F1 or opportunistically rewrite its accepted scalar loops.

R, root (one conjunctive command):

```sh
sh -ec 'cargo test --workspace --all-features --locked; cargo clippy --workspace --all-features --all-targets --locked -- -D warnings; cargo fmt --all -- --check'
```

E, root (normal generation, audit/build and freshness must all succeed):

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s sh -ec 'proof/regen.sh; proof/regen.sh --check'
```

L(command), cwd `proof/corpus`:

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s <command>
```

After E1, re-baseline only by the spec's five-step protocol: root/declaration delta review,
same-option freshness, statement-preserving bridge transport, closed-cone audit, semantic controls
and full Rust/both-package integration on one identity. Preserve warnings-as-errors and the
generated axiom/sorry/admit rejection. A green E1 is entry-encoding feasibility, not generic
refinement, end-to-end correctness or a W7 release. Existing read-cutover/summary-deletion,
source-linked HashDOS and strict REG-1 tasks stay held behind their own unchanged prerequisites.

#### Sol ratification and lead-only governing edits

Required verdicts: **S1** before implementation (eleven retained/three omitted flags; native-operation
semantics and tuple transport; concrete instance completeness versus generic claims; short-borrow
wiring preserves C/T); **S2** after generation (finite source-pinned theorem transport/discharge map,
including allocator/model blockers and the committed/scratch reservation projection). These are
not permission to admit postconditions or call a tested directory proved. No fresh Sol verdict is
claimed in this architecture task.

Apply none of the following here. They replace only the superseded AEN-DIR pipeline/status edit
proposals; A4's API boundary and unrelated governing changes remain. Exact insertion text:

| File + section + location | Exact edit for lead |
|---|---|
| `docs/direction.md`, `Execution order`, immediately before `Serial read specialization and stable parallel scatter` | Insert: “**Single-implementation extraction repair.** Follow APEIRON-ANATIVE-AEN-PIPE in `docs/epics/algebra-native-addressing.md` and `docs/epics/algebra-native-addressing.md`: explicit monomorphizing Charon configuration plus mutable-reference-free aggregate wiring, preserving the one production implementation. The scalar directory errors are reported cleared; generic-loop and mutable-reference type-declaration extraction remains open. Configuration selection is not translation or refinement acceptance. Preserve all admitted domains, generic proof obligations, reservation/coverage contracts and strict performance requirements. Re-baseline generated Lean only through fresh regeneration/check, statement-preserving bridge transport, closed-cone audit and rejecting controls. W7 remains frozen and summary deletion held.” |
| `docs/kernel-boundary.md`, `Canonical object and executing root`, immediately before `Read specialization and scatter boundary` | Insert: “**Monomorphizing extraction boundary.** ANATIVE-AEN-PIPE replaces the Aeneas preset with its explicitly enumerated compatible flags, full monomorphization and the candidate mutable-shape specialization. Native indexing, unsizing/repeat and tuple declarations require checked support and bridge transport; option validation is not evidence of semantic preservation. Short mutable function borrows remain; no translated aggregate declaration may contain a mutable reference. Both carrier instances retain the same production reserve/materialize/link bodies, TableStorageCoverage and committed-state refusal contract. Finite word roots prove no arbitrary-generic claim; those quantifiers and their source-link obligations remain required. Unsupported primitives block acceptance rather than become opaque assumptions. No historical planner/reserver exclusion is a permanent exemption from sequential source refinement. Allocator/executor realization, codegen/ABI/GPU exclusions and the separate Rust closed/Slang empty cones remain unchanged.” |
| `docs/roadmap.md`, `Serialized table owner and spatial plan`, immediately before `Ownership sequence` | Insert: “ANATIVE-AEN-PIPE supersedes the next extraction-repair dispatch only. After owner release and Sol S1, serialize F1 short-borrow/index wiring and E1 explicit-configuration regeneration/freshness as one repair checkpoint; the existing extraction red may persist between those two tasks only, with no unrelated source work interleaved. Stop on failed E1. Do not repeat scalar-arena conversion. Sol S2 then freezes all affected bridge transport slices, each at most three files with one gate/control; generic claims cannot be replaced by concrete word claims. Complete AxiomAudit and both-package/Rust/Kani/control barriers remain mandatory on one identity, even if an already-red old-summary dependency awaits its separately authorized successor proof. This schedules repair, not acceptance of an intermediate representation. No W7 reopening or summary-deletion release follows.” |

The extraction task must also make its three-file script/wrapper/generated identity reproducible;
updating `proof/README.md`'s old preset recipe/tool receipt is a subsequent separately owned
documentation task, not a fourth E1 file or an applied edit here.

#### Documentary acceptance and unresolved goal

D, repository root (includes full untracked lifecycle contents):

```sh
sh -ec 'git diff --check -- "$@"; for f do rc=0; out=$(git diff --no-index --check /dev/null "$f" 2>&1) || rc=$?; test "$rc" -le 1; test -z "$out"; done' -- docs/epics/algebra-native-addressing.md docs/epics/algebra-native-addressing.md
```

Baseline and post-edit D exit 0. A trailing-space stdin fixture produces a diagnostic and raw
no-index exit 3; D's exit/diagnostic rule rejects it with wrapper exit 1. Unchanged files pass D
again, exit 0. These validate documentation only. Heavy commands E/L/R are unrun by scope.
The precise unresolved goal is warning-free, complete translation/build/freshness of every existing
real-source root with specialized reserve/materialize bodies and no mutable-reference aggregate
declarations, followed by unchanged-domain audited refinement. Native operation/backend support,
generic theorem transport and reachable allocator/model support remain risks, not assumed results.
No mathematical impossibility or guaranteed green pipeline is inferred from the supplied errors.

**Single lead question:** authorize S1 → F1 → E1 with the quoted governing qualifications, and stop
for a separately scoped pinned-tool/model repair if this exact candidate fails? No second
implementation, narrower contract, axiom-cone enlargement or W7 retry is proposed.

### Historical directory dispatch — APEIRON-ANATIVE-AEN-DIR

**Select one scalar-arena production implementation; accepted translation remains BLOCKED.** The
spec's `Directory extraction decision` is the representation contract. This dispatch supersedes
A4's next-task status only for repairing the directory extraction lane. Do not repeat completed
capability/maintenance tasks from the historical schedule, release summary deletion or reopen W7.
The read tree already contains TableLocator maintenance; its presence is not source proof or a fresh
test receipt. No Rust/Lean/extraction build or performance command runs in this task.

```text
SCOPE: docs/epics/algebra-native-addressing.md; docs/epics/algebra-native-addressing.md;
       production-only directory shape, bounded conversion, tool residual and lead-only edits.
REGISTER: architecture/planning; excludes implementation, theorem changes and measurement.
CORPUS: Native/LocatorSpec.lean, table_absent_of_complete_path_missing:
        TableStorageCoverage + missing complete path -> no live representative;
        conditional theorem, not source-linked table coverage.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Native/LocatorSpec.lean
GATE: complete-file whitespace wrapper below, repository root; baseline exit 0.
      Supplied extraction red: Could not compute a loop fixed point in 2 iterations.
      No fresh extraction run authorized; accepted replacement shape MISSING.
GRADE: anchor before/after; no new proved/tested/measured implementation result.
```

Owner Astra; supplied-checkout documentation exception; base
`c450c222905c050f793a23332708c230c658f82c` plus dirty inputs. No new worktree/build-cache lane,
source edit, governing edit or commit. Release current file owners before any implementation task.

#### Evidence identity and exact blocker

| Read input | SHA-256 |
|---|---|
| `src/coord/locator_core.rs` | `ad2fb37847b319fe711ddfa5567947d7ded41b051ee1f4fa61982366a7c34404` |
| `src/coord/locator.rs` | `67da2e1895955858e22604d001d1ca8751c54067d961c4f3f24b1ea785c7aa35` |
| `src/table/locator.rs` | `5c61f02bcc634d66bf3312bdc5cb7cbc776d2ad3dfe9d51aa2b9a8cbb72a438a` |
| `src/table.rs` | `78bc20c91b6b6f29503e702dd0df3a25613e9c936741899683b8cacecee51ecd` |
| `src/table/state.rs` | `411949407965312fd2fa5c4ac1efefdf096fde9fa084760210af3449f3de30da` |
| `proof/corpus/Native/LocatorSpec.lean` | `3cd68d4695b2242716c072b10fa4f50c460701ec24912b5814282aceb0af8601` |
| `proof/regen.sh` | `898aa30927b149bd49af6e38ce5b9f5def096ac8e3157146c1e0baf5ffdabdf6` |
| `target/aeneas/aeneas` | `33d0ce1a1cc2f1d3935c0cd2ab34f283797da61c531ed6d4e09d5c97b0d0228a` |
| `target/aeneas/charon` | `747e165bc62bb963f8d741e8c193c721a4b94555bcb1f9f86c79090fd7a59d0e` |

Tool versions `0.1.254` / `505b6ca-dirty` and the walk fixed-point failure are assignment inputs,
not reruns. The binary hashes match the patched identities recorded in `proof/README.md`.
`walk_residue` now spans core lines 889–907; changing line numbers is not a different failure.
The read historical `/tmp/opencode/locshape/aeneas-cur.log:10-16` names both Step-field model
warnings; that run also has an unsupported raw-pointer operation and a partial output, so it is
not a successful shape receipt. `/tmp/opencode/locshape/aeneas-full.log` reports nested-loop returns
and scoped-thread/higher-ranked-lifetime failures for a wider root selection; no inference that
the entire crate is now translatable follows from repairing the directory.

Mechanism source read: `/tmp/opencode/patchtest/aeneas-base/src/interp/InterpLoopsFixedPoint.ml`
149–224 (join/reduce/equivalence), `src/Config.ml:176-182` (hard-coded limit 2),
`src/extract/Extract.ml:2859-2875` (missing builtin-field warning) and `src/Main.ml:146-148`
(`-warnings-as-errors`). This is the locally available pinned-base source, not a new tool build.
Installed `target/aeneas/backends/lean/Aeneas/Std/Core/Iter.lean:283-327` contains scalar overflowing
definitions. Absence of a trait-field registration is distinct from absence of those definitions.
Keep both warnings fatal; no exception or change to the axiom cone is proposed.

#### Conversion tasks and stop conditions

Each row is a singleton serial task, at most three owned files, prefix `APEIRON-ANATIVE-AEN-DIR-`.
Flash owns source/extraction, Sol owns contracts/proofs. Use isolated task worktrees and private
Rust targets; serialize extraction and Lean caches on the existing lane. Pin sources after ownership
release. R and E below are single conjunctive gate commands, not claims of current passes. A semantic
control must fail an assertion with nonzero test count, not compilation, OOM or timeout. Perform
baseline/red/restored in a disposable copy; never mutate another owner's checkout. New fixtures
and proof targets are deliverables, not existing acceptance evidence.

| Task / predecessor | Owned files and completion | One gate | Rejecting control / preserved invariant |
|---|---|---|---|
| S0 / owner release | `proof/corpus/Native/LocatorSpec.lean` only: Sol freezes scalar-record denotation, four-field C transport, cumulative eight-category receipt interpretation, committed-state stutter and scratch-phase mapping; distinguish byte access from faithful retention for QueryIdentity. Add joint empty/success/late-failure witnesses, no assumed output C. | L(`lake build Native.LocatorSpec`) | Attempt to certify a live payload with its terminal omitted; the missing-path/coverage witness rejects. Logical contract only, not source coverage. |
| F1 / S0 | `src/coord/locator_core.rs`, `src/coord/locator.rs`, `src/table/locator.rs`: atomic replacement of nested arenas by the spec's three scalar vectors and copied observer API; all walk/stage/link/rollback paths and both cold validators consume it. Preserve public APIs and TableLocator fields. Include inline tests for root separation, first/last missing depth, prefix/empty bytes, fanout 256, edge splicing and rollback. Update physical-to-logical receipt observation in the same cut; no compatibility arena or second traversal. | R | Change the Parabolic root handle to the Hyperbolic root; an explicit disjoint-root/descendant-ownership assertion after mixed-face insertion rejects (readback alone need not detect shared prefixes). C/Q/T, sentinel/bounds and one implementation survive. |
| E1 / F1, before any further source task | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`: fresh real-source translation, retain all existing roots; add production directory read/write/constructor/reservation/observer roots as needed for both faces and both carriers. Thin monomorphic callers only. Inspect generated calls to scalar helpers; publication as well as lookup must translate. | E | Corrupt a generated child-link write in a disposable copy; `--check` rejects drift, restored passes. This is freshness, not a semantic theorem. Any failed root stops the wave. |
| F2 / E1 | `src/table/locator.rs`, `src/table/resources.rs`, `src/table/tests.rs`: reconcile cumulative checked plans and actual record capacities, root/scratch/retained buffers and late refusal cleanup with S0. Keep all eight receipt categories and no effects on refusal. Extend actual-path counter/receipt witnesses for both carriers; do not use target-as-observed capacities. | R | Remove the final edge-record reservation and force that late refusal; zero-callback/stutter assertions reject. No fallible operation after retention. |
| F3 / E-F2 | `src/table.rs`, `src/table/locator.rs`, `src/table/tests.rs`: consume accepted engine/receipt on constructor, found/fresh, prepared apply, growth, spill-to-flat relocation and pending commit. Retain TableLocator payload-location handles; fix any caller assumptions of root 0 or old arena identity. Add both-carrier/default-valued/mixed-face reachability witnesses; no query cutover. | R | Omit relocation retarget while keeping the moved payload; C/readback rejects. Exact live length, one merge, typed faces and full identity after fingerprint remain. |
| F4 / E-F3 | `src/table/parallel.rs`, `src/table/locator.rs`, `tests/parallel/main.rs`: consume the same engine in preassembly/rebase and prepared workers. Reserve all before worker callbacks and preserve per-key input order and face tokens; no separate parallel directory algorithm. | R | Drop one assembled terminal's location rebase; mixed-face both-carrier coverage/readback rejects. C/T and callback order survive. |
| P1 / E-F4 | new `proof/corpus/Native/LocatorSource.lean`, `proof/corpus/Native/LocatorCost.lean`, `proof/corpus/lakefile.lean`: source-linked constructor/transition facts and charged scalar-step/arena-growth obligations with inline fail-closed audits, under S0 and the existing S2 interface. No assumption of preserved output C or full cost bound. | L(`lake build Native.LocatorSource Native.LocatorCost`) | Wrong generated child store rejects reachable-key preservation semantically. Return any undischarged field as open; a green file with only conditional lemmas does not close S2. |

F1 is the single selected new shape experiment after the supplied unsuccessful reshaping campaign,
not authorization for three more syntax variations. F1/E1 jointly establish feasibility only if
both gates pass. If F1 cannot fit the three files/size bounds, stop and request a size-only cut;
do not add a fourth file. F3's read `src/table.rs` is already 1,578 lines, above the 1,500-line cap:
before F3, separately assign a size-only task owning `src/table.rs`, `src/table/locator.rs`,
`src/table/tests.rs`, moving unchanged locator maintenance bodies into the existing included file.
Gate R; control omit a moved retarget call and require the relocation coverage assertion to fail.
This creates no new module or alternative body. Recheck actual sizes before dispatch, then run E.

After every accepted F row (including the size cut), an E-row instance owns the same three proof
files and uses E1's gate/control. Full bridge regression remains required; the lead must apply the
narrow scheduling edit below before deferring an already-red dependency. No downstream source row
runs past a new extraction error. If E1 is blocked by reserve/capacity, sorting, iterator or callback
models, return the exact required primitive and its real call site; do not drop roots or call it an
allocator exclusion. If P1 needs another proof module, return a new bounded proof slice rather than
assuming the result. P3/audit registration, full barriers and measurement from the existing plan
remain necessary; these rows do not close the epic or reopen W7.

R, repository root:

```sh
sh -ec 'cargo test --workspace --all-features --locked; cargo clippy --workspace --all-features --all-targets --locked -- -D warnings; cargo fmt --all -- --check'
```

E, repository root (unchanged warnings/audit/freshness):

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s sh -ec 'proof/regen.sh; proof/regen.sh --check'
```

L(command), cwd `proof/corpus`, expands to:

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s <command>
```

#### Residual tool decision and lead-only governing edits

The exact unresolved goal is extraction of the real scalar-arena directory's read, reserve,
publication and relink closure with all current production roots, without warning, unknown external
or partial body. No no-build architecture review can certify that pinned Aeneas accepts it. If
F1/E1 still fails loop-context convergence, freeze the smallest real-source LLBC and trace and hand
off a translator regression/repair task. Do not advertise a nonexistent iteration flag; changing
`loop_fixed_point_max_num_iters` requires a reproducible tool patch and does not establish soundness
or convergence. A Step registry mismatch separately requires a model/registration repair or removal
of genuinely unused range machinery in production, not dropping `-warnings-as-errors`.

Apply none of these edits here. They supplement A4's typed-API edit map and supersede only its next
directory dispatch. Exact text is quoted; governing owners and the lead apply it after release.

| File + section + insertion point | Exact edit |
|---|---|
| `docs/direction.md`, `Execution order`, before `Serial read specialization and stable parallel scatter` | Insert: “**Single-implementation directory feasibility.** Follow APEIRON-ANATIVE-AEN-DIR in `docs/epics/algebra-native-addressing.md` before further locator integration. The production directory itself takes the scalar-arena shape in the spec; no proof-only or parallel traversal is admitted. Translation feasibility is open until real-source read/write/reservation roots pass the pinned gate. Preserve typed faces, complete table coverage, all reservation categories, retained fingerprint and full identity. No source-proof, HashDOS or performance acceptance follows from shape selection. W7 remains frozen.” |
| `docs/kernel-boundary.md`, `Canonical object and executing root`, before `Read specialization and scatter boundary` | Insert: “**Directory extraction boundary.** One production handle-only directory serves standalone Locator and TableLocator. Scalar arenas are selected for translation feasibility, not accepted by assertion. Stable integer handles may survive Vec relocation; pointer stability is not claimed and copying/growth remains charged. TableStorageCoverage and ReserveBeforeRetainContract require actual constructor/transition refinement, including sound reverse links, private staging and late-refusal stutter. No always-Unknown substitute, assumed output invariant or tested-only directory perimeter discharges those obligations. Keep the Rust closed cone, Slang empty cone and existing realization exclusions.” |
| `docs/plan.md`, `Current status`, at section start | Insert: “Current locator extraction is blocked by the supplied pinned-Aeneas loop-context fixed-point failure in `src/coord/locator_core.rs::walk_residue`; historical green extraction receipts do not cover this dirty identity. Dispatch the bounded ANATIVE-AEN-DIR feasibility wave only after owner release and Sol contract ratification. Stop on its first unresolved tool/model primitive. Keep W7 frozen and summary deletion held; shape conversion is not read-cutover authorization.” |
| `docs/roadmap.md`, `Serialized table owner and spatial plan`, before `Ownership sequence` | Insert: “ANATIVE-AEN-DIR is the next directory feasibility wave, not a restart of completed A4 capability tasks. Serialize its S0 → F1 → E1 sequence and stop if either production conversion or real-source extraction fails. Only then dispatch its receipt and consumer slices, each followed by E. For this bounded repair, the known-red old-summary full bridge regression may await successor P completion; no new extraction failure is deferred and no intermediate representation is accepted. Full Rust/Kani, both proof packages, audits and controls remain required on one identity. W7 is unchanged.” |
| `proof/README.md`, `Stage 1: extraction`, after the paragraph describing `regen.sh --check` freshness | Insert: “Directory shape feasibility includes the real read, reservation and mutation closure; wrappers may not substitute a second implementation. Keep Aeneas warnings fatal, including missing Step overflowing-method builtin fields, until a pinned model/registration repair or a checked removal of unused production range dependencies removes them. Lean scalar definitions alone do not certify extractor trait registration. Unknown externals, partial bodies and axiom/sorry/admit remain failures.” |

**Single lead question:** authorize the bounded S0/F1/E1 feasibility wave, with a pinned
translator/model repair decision only if that one selected shape fails? The one-implementation
requirement is settled; this question requests execution outside the present no-build scope, not
permission to weaken a contract. There is no established unshapeable limitation or approved tool
change yet. Allocator/executor, codegen/ABI/GPU exclusions are unchanged, not proof of their behavior.

#### Documentary acceptance

Use the existing complete-file gate, including these untracked lifecycle files (repository root):

```sh
sh -ec 'git diff --check -- "$@"; for f do rc=0; out=$(git diff --no-index --check /dev/null "$f" 2>&1) || rc=$?; test "$rc" -le 1; test -z "$out"; done' -- docs/epics/algebra-native-addressing.md docs/epics/algebra-native-addressing.md
```

Baseline and post-edit gate exit 0. A trailing-space stdin fixture produces a whitespace diagnostic
and raw no-index exit 3; the same exit/diagnostic checks reject it with wrapper exit 1. Unchanged
lifecycle files pass the gate again. No control file or source is modified. This checks documentation
only. Mathematical proof, implementation refinement, runtime tests and measurement are not performed
or claimed. Next bounded task: S0 after owner release; F1/E1 is
conditional on that contract, source-size feasibility and lead application of the scheduling edit.

### Historical dispatch decision — APEIRON-ANATIVE-W4-A4

The lead selects **explicit capability**, rejects compatibility preservation, and resolves A3's
single lead question. This section and the spec's `Explicit-capability API` supersede A3's admission
hold, old I3 row, old next dispatch and governing-edit map. A3's C/Q/T invariants, wiring/deletion
order, E/P gates, measurement requirements and W7 freeze survive. A3/A2 historical receipts below
are not A4 acceptance. No production, proof, evidence, manifest or governing file is edited here.

```text
SCOPE: docs/epics/algebra-native-addressing.md; docs/epics/algebra-native-addressing.md;
       capability API, explicit break inventory, bounded migration and owner-applied edits.
REGISTER: architecture/planning; not implementation, refinement proof or runtime measurement.
CORPUS: Native/LocatorSpec.lean, FaceLocatorCapability and table_admission_map_exact;
        conditional capability/admission definitions, not source-linked table coverage.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Native/LocatorSpec.lean
GATE: A4 documentary gate below, repository root; initial raw no-index command exits 1
      for untracked additions with no whitespace diagnostic; wrapper checks that distinction.
GRADE: anchor before/after; no new proved/tested/measured implementation claim.
```

Owner Astra; supplied-checkout documentation exception; base
`c450c222905c050f793a23332708c230c658f82c` plus dirty inputs. No build/cache lane acquired.
I1's read source stages/reserves edges before retention and has the late-edge seam; this is a source
observation, not a rerun or acceptance of I1. I2 owns the shared-directory cut until its owner returns
the gate/control receipt. Do not repeat either task or edit its files concurrently. Re-pin the I2
result before I3; the old source hashes below do not identify that future result.

#### Interface decision and migration invariant

The spec fixes the signatures, laws and all **45** `generic_api` dispositions. In brief:

- Reuse `ExactIdentity`; add `FaceResidueCapability`, `EllipticPacking` and sealed `FaceDomain`.
  `FaceCoord::Faces` is either `Sigma` (requires packing) or `LocatorSigma` (two filtered faces).
  Both default to Parabolic. `FaceHome` agrees with these maps; `Home: FaceHome` keeps only its
  separate physical storage interface. Remove the blanket closure impl and the placement-only
  address default. No replacement blanket Home impl or inferred Eq/Coord capability.
- Sealed `KeyRep` implementations require `K: ExactIdentity` and the face home; add delegated query
  bytes, held bytes and held semantic address. `Stored` owns K once; `Regen` keeps its independent
  exact word and codec byte agreement. Generic helpers retain `R: KeyRep<K,H>` rather than requiring
  an additional K bound that Rust cannot infer from it. No duplicate retained identity/value store.
- Every public `_at` argument/input/query tuple uses `H::Faces`; defaults still mean Parabolic.
  Erased internal Sigma has a domain-membership invariant. Parallel transports the same tokens;
  token conversion must not allocate/copy all keys before checked reservation or alter callback order.
- Opaque Eq-only keys and arbitrary closures fail admission at compile time; lossy Regen codecs
  remain unavailable. Unbounded bytes and the actual 96-bit `ShardKey` use the explicit
  locator-only domain with a specified encoding, not a false word-packing witness. No generic
  compatibility fallback exists. Neither byte identity alone nor a Rust trait proves the laws.
- All original REG-1 keys, equality fields, streams, logical faces, baselines and registered cells
  remain. Add real adapters, not benchmark-type shortcuts. A missing encoding/cost/face population
  blocks readiness. Do not demote or delete the row, or quote its old timing for the new mechanism.

#### I3 preparation and atomic activation

Each row is a singleton serial task prefixed `APEIRON-ANATIVE-W4-`, Flash unless marked Sol.
Exactly the listed files are owned, at most three. **R** and **L** below retain A3's exact commands
and resource bounds. No source task can claim acceptance before its gate and one semantic rejecting
control pass baseline/red/restored. New tests are deliverables authored first; zero tests, compile
failure, timeout or OOM are not a semantic red. Compile-fail admission witnesses are separately
required where named; their expected rejection is an API test, not a theorem or semantic mutation.

Preparatory named homes implement both the old physical API and the new explicit capability.
This allows each consumer slice to compile before removal of the blanket adapter. It does **not**
activate a locator or provide a final compatibility route. Generic old `_at(Sigma)` signatures
remain only until I3-K; no locator-only type may consume algebraic absence in that interval.
All stages use the real Table and shared encodings, never a shadow table or a default fake law.

| Task / predecessors | Owned files and completion condition | One gate | One rejecting control / invariant |
|---|---|---|---|
| I3-A / accepted I1, I2 | `src/coord/address.rs`, `tests/native_address/main.rs`: add residue/packing/domain traits, `LocatorSigma`, associated Faces and shared dispatch; WordFaceHome uses Sigma. Provide the immutable validated fixed token. Update address law tests and compile-fail unencodable Elliptic token example. | R | Encode Elliptic word q as reduced zero; exact-pack boundary assertion rejects. No storage behavior changes. |
| I3-B / I3-A | `src/coord/locator.rs`, `tests/native_locator/main.rs`: implement the existing finite identity interface for the actual small integer test/Kani keys (`i8`, `i32`, and any used unsigned counterparts); fixed-width original-bit bytes and exact retained representation. Keep I2's public trait path/re-export and shared engine. | R | Drop the top identity byte; distinct same-residue/high-bit keys must fail equality/byte-law tests. No new approximate codec. |
| I3-C / I3-B | `tests/table.rs`, `src/table/tests.rs`, `tests/parallel/main.rs`: replace closure homes with named explicit homes; implement exact identity for Counted/CountedKey and semantic maps independent of the collision-only physical route. Preserve every late-hit, callback, refusal, merge-order and both-carrier test. Explicitly qualify physical address calls. | R | Encode Counted's distinct equality keys identically; an added byte-exactness assertion rejects even before locator activation. Counter fields not used by Eq are not identity. |
| I3-D / I3-C (historical; Kani portion retired) | `src/table/partition.rs`, deleted `src/proofs.rs`, `src/merge.rs`: the original task migrated inline-test/Kani homes and the closure doctest to explicit homes. Current work retains ordinary source-path tests; no Kani harness or separate Kani barrier remains. | R for retained Rust paths | Corrupt the partition-test home's semantic face while leaving its physical placement intact; its face-law assertion rejects. |
| I3-E / I3-D | `examples/micro_spill.rs`, `examples/micro_spatial.rs`: explicit identity and face homes for the original keys, same physical diagnostic workloads and counters. Add inline encoding round-trip assertions/tests. | R | Encode two distinct diagnostic keys to identical bytes; the example test rejects. No retiming or rewritten workload. |
| I3-F / I3-E | `examples/reg1/strata.rs`, new `examples/reg1/adapters/native_strata.rs`: move key/home definitions into the included child and re-export them, preserving access and equality. Supply full tag bytes, exact packing and both residues; keep raw placement separate. This cut makes room below the source-size cap rather than adding lines to the nearly full strata file. | R | Ignore the high tag bits in identity; inline full-width round-trip test rejects. All 200 strata definitions/inputs unchanged. |
| I3-G / I3-F | `examples/reg1/matrix.rs`, new `examples/reg1/adapters/native_matrix.rs`: same source-size cut and explicit capability for SharedKey/SharedHome, preserving the full tag and every matrix row/resource outcome. | R | Truncate SharedKey's high tag bits in its hyperbolic encoding; the u32::MAX widened-remainder oracle rejects. No row demotion. |
| I3-H / I3-G | `examples/reg1/workload.rs`, new `examples/reg1/adapters/native_workload.rs`: move the key/home definitions, preserve both ShardKey fields in a 12-byte encoding; choose `Faces=LocatorSigma` for ShardHome and a specified dual/H3_40 encoding, with widened modular arithmetic and charged byte cost. Preserve ConstantKey's full tag. | R | Omit ShardKey.home from bytes; same-tag/different-home identity test rejects. No fabricated u64 packing; original stream/oracle digest unchanged. |
| I3-J / I3-H | `examples/reg1/candidate.rs`, `examples/reg1/adapters/spatial.rs`, `tests/native_address/main.rs`: migrate remaining named test homes and adapter uses; audit qualified physical versus semantic calls. Reuse migrated word-wrapper keys; add the 96-bit locator-domain compile-positive witness. Do not make the benchmark-only candidate the production implementation. | R | Use only residue bytes in the adapter witness; 0/q collision identity assertion rejects. Every supplied consumer has an explicit domain before activation. |
| I3-Q / I3-J | `src/table.rs`, `src/table/keyrep.rs`, `src/table/parallel.rs`: qualify all existing physical calls as `Home::address` before introducing the same-named FaceHome method through the supertrait. No encoding or query behavior change. Other consumer qualification is owned by its preparation row above. | R | Flip bit 8 of the held physical placement only during spill relocation; both-carrier growth readback rejects the inconsistent fingerprint. No ambiguous method lookup remains at activation. |
| I3-I / I3-Q | `src/table.rs`, `src/table/keyrep.rs`, `tests/table.rs`: implement WordHome capability by delegation; add required carrier/query/held and sealed codec byte accessors; strengthen concrete instances. Set `Home: FaceHome`, remove closure blanket/address default. Retain physical TableAddress signatures. Add both-carrier agreement and Eq-only/closure compile-fail doctests. `_at` token cutover is the next row, not a locator release here. | R | Use reduced H3_40 residue as Regen held bytes; 0/q both-carrier identity agreement rejects. All callers compile with no compatibility Home blanket. |
| I3-K / I3-I | `src/table.rs`, `src/table/parallel.rs`, `src/table/partition.rs`: make all serial/parallel `_at` arguments and tuple inputs use H::Faces; adapt shared prepared/build/partition consumers to carry tokens without pre-reservation key cloning. Default entry points use domain DEFAULT; erase only at the internal storage boundary. Add inline actual-constructor mixed-face and locator-only success/refusal tests plus compile-fail Elliptic-on-locator-domain use. | R | Erase Hyperbolic as Parabolic; same-key/different-face serial and parallel readback rejects. Token admission holds on every public path, no runtime UnsupportedFace branch. |

The symbolic order deliberately leaves I3-I as the carrier/interface activation after consumer
preparation; task IDs are identities, not permission to dispatch alphabetically. The old single I3
is retired, not assigned in parallel. File-size/directory caps apply to each row, including the
three new included key/home files; if an actual diff needs another path, stop and re-slice rather
than editing it unowned. New helper files count toward the existing 12-file directory cap.

**Gate feasibility:** R includes tests, doctests and all-target compilation/lints, not Kani or the
REG-1 campaign. New inline example tests must execute under `cargo test --all-targets --all-features
--locked` as part of R4 below, so an example-only assertion is not accepted merely because R compiles
it. For I3-E/F/G/H/J use **R4** instead of R: the same single R command with its test invocation
replaced by `cargo test --workspace --all-targets --all-features --locked`, followed by
`cargo test --workspace --doc --all-features --locked`; clippy/fmt unchanged. The documented row
control runs under that same gate. No changes to Cargo.toml are part of these slices.

After each source row, serialize **E-<row>** with the existing three-file extraction lane and
bounded regeneration/check command below. Preserve all production roots; add direct byte/face
accessors and both carriers' directory roots from `Native.LocatorSpec.requiredExtractionTargets`.
A required unsupported primitive stops the next row; do not add formula-copy wrappers or assumptions.
I3-D additionally needs focused ordinary tests of its concrete homes and a real-source semantic
mutation control before I3 closes. The full barrier is not collapsed into R.

#### Post-I3 wiring and Sol verification

After I3-K, E-I3-K and Sol S3 below, the next production tasks are A3's **I4, I5, I6, I7**, unchanged file ownership:

| Task | Files | One gate | Rejecting control / invariant |
|---|---|---|---|
| I4 | `src/table.rs`, `src/table/state.rs`, new `src/table/locator.rs` | R | Empty directory beside a live payload must fail coverage. Table owns the only payload; no read consumes the directory. |
| I5 | `src/table.rs`, `src/table/resources.rs`, `src/table/locator.rs` | R | Omit a required late edge reserve and force failure; zero-callback/stutter test rejects. All directory, identity, handle and scratch reservation precedes effects. |
| I6 | `src/table.rs`, `src/table/locator.rs`, `src/table/tests.rs` | R | Omit a relocation retarget; both-carrier constructor/growth coverage rejects. Serial fresh/found, prepared, rebuild and pending commit maintain C/T. |
| I7 | `src/table/parallel.rs`, `src/table/locator.rs`, `tests/parallel/main.rs` | R | Drop one worker's links but retain payloads; actual parallel readback/coverage rejects. Same domains, ordered effects and jointly reserved assembly. |

**Required Sol task: `APEIRON-ANATIVE-W4-S2`, source-linked coverage/cost adapter instantiation.**
It follows I3, but cannot finish immediately after an interface-only change: its actual-state
predecessors are I4–I7 and their E receipts. Own new `proof/corpus/Native/LocatorSource.lean`, new
`proof/corpus/Native/LocatorCost.lean`, `proof/corpus/lakefile.lean`; inline fail-closed audits.
One gate, cwd `proof/corpus`:

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s lake build Native.LocatorSource Native.LocatorCost
```

Bind generated query/held bytes and face-domain interpretation to
`FiniteExactIdentityContract` and `FaceHomeLaws` (all-face `FaceLocatorCapability` only where packing
exists). Instantiate `TableEntryAtCoverageAdapter`/`SourceCoverageObligations` from the real table
constructor and transitions, not from standalone Locator state, a boolean flag or an assumed
committed invariant. Supply the restricted two-face relation for LocatorSigma. Establish cost
charges from actual directory branches/handles, including byte access, copies, reservations,
growth/relink and failures. Rejecting control: omit one generated terminal relink while retaining
the relocated payload; the both-carrier coverage-preservation witness must fail semantically.

`SourceLinkedCostObligation.populationIndependentLookup` cannot be inhabited while a reachable
Unknown path does an unbounded population sweep. S2 must either source-prove exact-terminal access
with the required bound, or explicitly return that field and the aggregate recurrence as unresolved.
It may prove scoped coverage and charged finite costs without claiming the whole cost structure.
This distinction blocks final HashDOS/pepper release, not a claim that a hypothesis was proved.
If a source exact-terminal task is necessary, it needs a new bounded assignment; I8/I9 are not
permission to add it silently. S2's three-file budget is a stop condition, not a demand to assume
the result when more proof modules are needed.

Immediately after I3, **S3** (Sol, `proof/corpus/Native/LocatorSpec.lean` only) updates the stale
pending decision and freezes the generated domain/byte adapter signatures and joint all-face versus
locator-only witnesses. Gate L(`lake build Native.LocatorSpec`); control: replace ShardKey's exact
bytes with tag-only packing; the equal-tag/different-home witness must reject that identity
instance. This is contract ratification, **not** S2 source coverage.
Do not edit `api_migration_is_pending` to assert Rust integration or source cost is complete.

Then S2 → **I8** (Parabolic) → **I9** (Hyperbolic), each owning
`src/table.rs`, `src/table/probe.rs`, `src/table/tests.rs`, gate R. Their respective Unknown→Absent
mutation must fail a live late-hit/0-q-collision read and insertion-no-duplicate test. Re-run E and
S2 on each cutover identity. Elliptic has no locator. Only after both real cutovers and their controls
may the lead release **D1–D3**; the summary deletion hold remains. Retain fingerprints/full identity,
E/P1–P3/audit, all resource/semantic controls and strict full REG-1. **W7 stays frozen** under its
unchanged full-state-preservation reopening condition; neither S3 nor S2 reopens it.

#### Exact governing-document edits — A4, lead only

Apply none here. These replace the A3 proposed edit map, not historical receipts. Owner release and
lead application are required before dispatch under a changed governing boundary.

| File + section + location | Exact edit |
|---|---|
| `docs/direction.md`, `Execution order`, before `Serial read specialization and stable parallel scatter` | Insert: “**Explicit-capability algebra-native migration.** The lead selects the breaking capability API in `docs/epics/algebra-native-addressing.md`, Explicit-capability API. Eq-only keys and arbitrary placement closures are not admitted to locator-backed absence. A home declares all faces with exact word packing, or the explicit Parabolic/Hyperbolic token domain with exact bytes and residue maps; no generic compatibility fallback is retained. Follow the A4 I3 preparation/activation and I4–I9 sequence in `docs/epics/algebra-native-addressing.md`. Keep summary deletion held until both filtered faces are wired with source coverage and controls, then release D1–D3 and revalidate extraction/proofs. Elliptic has no locator; the fingerprint remains a prefilter. W7, source-linked HashDOS, G1 and every strict comparison obligation remain open.” |
| `docs/kernel-boundary.md`, `Canonical object and executing root`, before `Read specialization and scatter boundary` | Insert: “**Capability and certificate boundary.** The selected successor uses `Home: FaceHome`, sealed capability-bearing KeyRep instances, exact query/held bytes and a typed face domain as specified by ANATIVE A4. Sigma-domain homes additionally provide exact packing; LocatorSigma-domain homes cannot receive Elliptic. Old Home placement results are physical adapters, never semantic absence certificates. Absent requires actual TableStorageCoverage over every possible payload in scope and exclusion of other storage. Unknown is neither presence, absence nor vacancy. Table owns payloads once; directory handles do not own a second key/value map. Constructor, insert, prepared/parallel build, growth, relocation and commit must establish/preserve coverage and reserve all resources before effects. Trait implementation alone proves none of these laws or costs. Generic callback, allocator/executor, codegen, ABI, GPU and unsafe-container exclusions remain; W7 is unchanged.” |
| `docs/kernel-boundary.md`, `Layer separation`, face-specific key-interpretation bullet | Replace that bullet with: “Face-specific key interpretation: elliptic exact word packing; parabolic dual residue; hyperbolic H3_40 scalar residue. Every exact-map face retains full identity directly or through an independently lossless codec; no lossy scramble/discard identity is admitted.” |
| `docs/kernel-boundary.md`, `Spill-filter obligations and partition cut`, after the four-row ledger | Insert: “Summary obligations remain live while their old source executes. Successor retirement requires both-face wiring, lead release, complete Rust/generated read/write/storage/receipt removal, source-linked coverage and rejecting controls. SpillFingerprintExact and all identity, occupancy, coverage, first-fit and refusal obligations survive. A finite Unknown sweep is not population-independent lookup evidence.” |
| `docs/roadmap.md`, `Serialized table owner and spatial plan`, before `Ownership sequence` | Insert: “ANATIVE A4 supersedes the admission/dispatch choice for locator integration. Accept I1/I2 receipts, then serialize the explicit-capability I3 slices in `algebra-native-addressing.md`; no second owner of their paths. Every old public combination has an explicit keep/constrain/reject disposition; no silent REG-1 row reduction. Follow S3 contract ratification, I4–I7 maintenance, S2 source-linked coverage/cost instantiation, I8/I9 cutover, then lead-held D1–D3, E/P and strict full-profile measurement. A standalone locator or S0 build is not table coverage. W7 stays frozen.” |
| Same roadmap section, immediately after `Mandatory serialized extraction/proof barrier after each source row` paragraph | Insert: “For the bounded ANATIVE capability/wiring/deletion chain only, E/freshness remains mandatory after every source slice. Require source-linked certificate evidence before query activation; defer the already-red old-summary full bridge regression to D3/P completion. No intermediate representation is accepted by this scheduling exception. Extraction blockers stop dispatch; all Rust/Kani, both proof packages, audits and semantic controls must pass on the integrated identity before acceptance/final measurement. This changes no theorem, evidence grade, domain law or W7 reopening condition.” |

#### A4 acceptance, residual obligations and handoff

One documentary gate (root cwd; includes the full untracked file contents):

```sh
sh -ec 'git diff --check -- "$@"; for f do rc=0; out=$(git diff --no-index --check /dev/null "$f" 2>&1) || rc=$?; test "$rc" -le 1; test -z "$out"; done' -- docs/epics/algebra-native-addressing.md docs/epics/algebra-native-addressing.md
```

Raw no-index checks exit 1 for additions even without diagnostics; that is not a whitespace red or
a baseline pass. The wrapper accepts only exit 0/1 with empty diagnostics; other errors reject.
This checks whitespace only. Read the 45-row inventory against
the source ledger and the concrete ShardKey population; no source/proof/runtime gate runs here.
Observed wrapper exit 0; the same full-file check rejects a disposable trailing-space fixture
with exit 1. The unchanged lifecycle inputs pass restored. These are documentary checks, not
runtime tests or a semantic mutation receipt. No governing edit is applied.
Remaining risks: I2's actual storage/extraction interface and growth charges; typed token transport
through all prepared/parallel paths; all-source adapter coverage; Unknown successful-read work;
finite aggregate insertion/rebuild cost; caps/consumer completeness; strict REG-1 and full barriers.
An unavailable primitive or a needed fourth owned file returns a bounded blocker, not a weaker law.
The governing edits and Sol ratification remain required. **No new lead design question**: capability
versus compatibility is settled. Next bounded dispatch is I3-A after I1/I2 ownership/receipt release.

### Historical A3 dispatch record (A4 supersedes admission and I3)

This section and the A3 certificate boundary in the spec supersede **all A2 dispatch, next-task,
measurement-expectation and governing-edit text below**. A2 is retained as a historical file-cut
record, not permission for deletion-first execution. The lead holds summary deletion until both
Parabolic and Hyperbolic use the locator on the real table paths. Elliptic gets no locator.
`W7-SearchCommitBoundary` stays frozen, with its seven premises and reopening condition unchanged.

```text
SCOPE: docs/epics/algebra-native-addressing.md; docs/epics/algebra-native-addressing.md;
       wiring-before-deletion order, adapter deliverables and lead-only governing edits.
REGISTER: planning/architecture; no src/tests/proof/evidence/manifest/governing edits or builds.
CORPUS: Native/LocatorSpec.lean, table_absent_of_complete_path_missing:
        TableStorageCoverage + missing complete path -> no live representative.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Native/LocatorSpec.lean:340-387;
           ReserveBeforeRetainContract:513-526; integration_replacement_exact:574-586.
GATE: Integration exit's complete-file whitespace command, repository root; baseline PASS, exit 0.
GRADE: anchor before/after; S0 definitions/implications are not source-linked table proofs.
```

Owner Astra; assigned-checkout documentation exception, base
`c450c222905c050f793a23332708c230c658f82c`, dirty inputs. No build/cache lane acquired. Concurrent
Flash `APEIRON-ANATIVE-W4-F1` exclusively owns `src/table.rs` and `src/table/probe.rs`; do not repeat,
renumber or edit its active factoring task. All later table owners start from its accepted snapshot.
The old F2–F4 deletions become D1–D3 **after wiring**. No source/proof gate is run in A3.

Read-input SHA-256 (not execution receipts; re-pin after any concurrent change):

| Input | SHA-256 |
|---|---|
| `src/coord/locator.rs` | `6e4c54a8f2f600692b98e3634df217a0faab578758878e7469f5e37dc7dcbf3e` |
| `src/coord/address.rs` | `d434488c9ee897327fe5e6c426a6e59c38d9ded9ebe905cf57fe76d05056b7c3` |
| `proof/corpus/Native/LocatorSpec.lean` | `f8fe4284602201f7a05ab015ec71d52e20cdb70d17f2406c65c437d1855217db` |
| `src/table/state.rs` | `8dad06425b7ae68b47657e97d7f31ece917b0dd87b7c520d7e1b4f321bf64f8a` |
| `src/table/resources.rs` | `a1d6e242c912706752375027c21b255677830ec74254d89fc8d5dd1324c06009` |

#### Read-source blockers and dispatch readiness

S0 now exists. Consume `TableStorageCoverage`, `TableCoverageTransitions`, `CertifiedDecision`,
`ReserveBeforeRetainContract`, `LocatorReservationReceipt` and `summaryReplacement`; do not request
S0 again. `parabolicResidueHom` is `TrivSqZeroExt.fstHom`; the Rust binary path reads `dual.residue()`
(`a`), never `b`. Hyperbolic uses one scalar H3_40 residue, not two independent channel filters.
`WordFaceHome` still encodes a word's parabolic `a` as `spread(word).place`; projection from that
dual element to `a` is a homomorphism, not a theorem that the key encoding itself is one.

Three prerequisites remain real, rather than implied by standalone tests:

1. **Admission:** S0 explicitly rejects opaque `Eq` and blanket placement closures as sufficient
   capabilities. The ledger's `block`/`narrow` entries are not lead authorization to break those
   APIs. `Home`/`KeyRep` still lack a population-complete face/identity-byte interface. `Regen`'s
   exact codec is independent of the residue and must remain so. A word-only adapter cannot close
   the opaque-key/REG-1 rows. No automatic closure encoding, global new `ExactIdentity` bound,
   always-`Unknown` final adapter or silently excluded comparison population is allowed.
2. **Resources:** `insert` now calls `reserve_for` before `retain_reserved`, and staged path storage
   is pre-reserved. Nevertheless `stage` and `new_identity_node` still call `edges.try_reserve(1)`
   **after** retention (`src/coord/locator.rs`). Rollback of node lengths does not undo retention
   effects. The entry seam only fails the first reservation. Table integration needs a late-edge
   failure control and no remaining fallible step after retain. The file itself records this residual.
3. **Coverage/cost:** `Locator<K,V>` owns its own values; `coverage_established` covers that store,
   not `Table`. It cannot be nested as a duplicate map. Native has no actual-source coverage/cost
   adapter. A present key deliberately yields `Absence::Unknown`; always sweeping on that outcome
   leaves occupancy-dependent successful lookup work. A miss filter alone cannot establish the
   final `HashDOSBound`, even if miss timings improve.

**Readiness rule:** F1 can finish its assigned size-only work. S1 below is the next bounded Sol
task. Later rows are exact conditional ownership/gate/control specifications, dispatchable only
after their named predecessors and S1's source interface are accepted. Calling the entire chain
unconditionally dispatch-ready now would invent the missing admission and source-cost evidence.
If S1 cannot supply the interface without breaking a current population, stop there for the single
lead decision below; do not expand a three-file wiring task into a public-API migration.

#### Gates, ownership and invariant vocabulary

Every row is a singleton serial wave. Task names below have prefix `APEIRON-ANATIVE-W4-`.
Every mutable path is listed; tests placed inside an owned source file count as that same file.
No concurrent `table.rs`, locator, test, extraction, Lake-cache or measurement-host owner. Each
source task gets an isolated worktree/private target; pin dirty input hashes and actual tool identity
at dispatch. An ownership or 1,500-line/12-immediate-source-file overflow stops for a separate cut.
No shim, dead-code allowance, proof-only production body or second value store makes a cut pass.

**R** (root cwd), one conjunctive command per source row:

```sh
sh -ec 'cargo test --workspace --all-features --locked; cargo clippy --workspace --all-features --all-targets --locked -- -D warnings; cargo fmt --all -- --check'
```

**L(command)** (cwd `proof/corpus`):

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s <command>
```

Each row's rejecting control uses a disposable copy, then the unchanged restored copy, under the
same gate. Rust/proof controls must fail a named semantic assertion, not compilation, missing tests,
OOM or timeout. Author new witnesses before implementation; record an intended red and nonzero test
counts. New module/test names below are deliverables, not assertions that those artifacts exist.

**C (coverage):** one authoritative table payload/value per `(Sigma,K)`; complete face/identity
paths and sound reverse handles cover every live representative in the certified scope, including
default-valued keys. If a directory covers only spill, flat exclusion and routed-spill completeness
are explicit premises; never present a region-local certificate as whole-table absence.
**Q (query):** `Absent` under C may skip matching, not manufacture a destination; `Unknown` must
continue exact search. Fingerprint equality is only a candidate prefilter. Unknown late hits,
same-residue unequal keys, unequal Sigma and first-fit order remain correct.
**T (transaction):** checked layout → reserve every destination, directory, path, edge and identity
buffer → check capacities → retain/initialize → private link/payload publication → merge once →
commit. No failure/search-visible partial publication after retain; refusal stutters with zero
hold/clone/merge/relocation. Existing allocator/executor exclusions do not weaken source ordering.

#### Adapter prerequisite and wiring slices

| Order / owner / task | Owned files and exact completion condition | One gate | One rejecting control | Invariant after acceptance |
|---|---|---|---|---|
| 0 / Flash / F1 (already assigned) | `src/table.rs`, `src/table/probe.rs`: unchanged A2 textual probe cut, including read counters. No locator/deletion work. | R | Terminate the moved sweep at an early vacancy; existing late-hit tests reject. | Same namespace/bodies; summary remains functional; exclusive parent owner released. |
| 1 / Sol / S1 | `proof/corpus/Native/LocatorSpec.lean`, `docs/epics/algebra-native-addressing.md`, `docs/epics/algebra-native-addressing.md`: freeze the source adapter deliverables below, exact signatures and ≤3-file proof targets; add joint contract witnesses/negative examples to the existing audited module. No production edit. | L(`lake build Native.LocatorSpec`) | Attempt the missing-path absence conclusion for a live but unlinked representative without coverage; the explicit counterexample must reject it. | Non-vacuous interface, population map and finite source-link/cost discharge route; logical compilation alone does not release read activation. |
| 2 / Flash / I1 | `src/coord/locator.rs`, `tests/native_locator/main.rs`: move every edge-node allocation/check into reserve-all; use allocation-free empty roots without changing the constructor signature. Add late-reservation failure injection at the real edge reservation, not just the entry seam. | R | Move one edge reservation after `retain_reserved`; injected late failure rejects zero-retain/unchanged-state assertions on a long-prefix insertion. | T for the locator, including staged edges and retained byte buffers; no claim of table coverage. |
| 3 / Flash / I2 | `src/coord/locator.rs`, new `src/coord/locator_core.rs`, `tests/native_locator/main.rs`: factor one shared directory engine whose terminals are bounded opaque payload handles. Standalone Locator delegates to it; Table will borrow query/held identity through S1's interface. The engine owns no `V` and does not retain a second key. Reserve/path-extension/relink operations share the same code, with charged arena growth. | R | Link a terminal to a different equal-residue key's handle; collision/coverage and original standalone readback witnesses reject. | Single trie implementation; two face roots, no elliptic root; no pointer invalidation, linear handle registry or uncharged copying. |
| 4 / Flash / I3, blocked on admission verdict | `src/table.rs`, `src/table/keyrep.rs`, `tests/table.rs`: implement only S1's approved face/query/held-byte capability and exact codec agreement. Keep physical placement separate from semantic face index. Cover every ledger population or stop; consumer edits outside these files require separately assigned slices before this gate. | R | Use the hyperbolic residue as Regen identity; `0`/`q` and unequal-Sigma readback reject. | Held/query addresses and exact identity agree; no accidental API narrowing; elliptic exact packing unchanged. |
| 5 / Flash / I4 | `src/table.rs`, `src/table/state.rs`, new `src/table/locator.rs`: attach the handle-only directory and private staged destination state to Table/rebuild storage; allocation-free empty constructor; implement C checker/tests in the new included file. No read consumes the directory yet. | R | Certify an empty directory against a manually live table payload; the checker rejects. | Table remains sole payload owner; unlinked intermediate staging cannot issue certificates; old query behavior remains. |
| 6 / Flash / I5 | `src/table.rs`, `src/table/resources.rs`, `src/table/locator.rs`: add checked locator reservation plans and observed receipts for binary/ternary nodes, identity nodes/edges, handles/reverse links, staging and retained identity buffers. Extend reserve/rebuild publication before callbacks; keep summary receipts for now. Add inline receipt/refusal witnesses. | R | Omit one late identity-edge reservation while forcing that allocation to fail; T witness rejects. | Actual capacities match every checked target, including scratch; T covers table and directory jointly, no locator failure after a table merge. |
| 7 / Flash / I6 | `src/table.rs`, `src/table/locator.rs`, `src/table/tests.rs`: maintain C on `apply_insert`, `apply_prepared_shard[_at]`, `grow_spill`, `relocate_entry/all`, `apply_pending_and_commit` and both serial build branches. Stage paths before effects; relocation retargets handles without merge; found updates keep links. No read cutover. | R | Omit the retarget after spill-to-flat relocation; actual-constructor both-carrier growth readback/coverage assertions reject. | C and T at each completed serial operation; canonical new-first/default/ordered behavior and exact live length unchanged. |
| 8 / Flash / I7 | `src/table/parallel.rs`, `src/table/locator.rs`, `tests/parallel/main.rs`: reserve all worker/link/assembly storage before worker callbacks; fill through the shared prepared core; assemble and rebase handles without a late allocation or duplicate merge. Preserve Parts validation, stable per-key order, worker budget and query-order callbacks. | R | Drop one worker's directory links while retaining its payloads; mixed-face, both-carrier parallel build/coverage/readback witnesses reject. | C covers assembled tables, not only worker fragments; T on late reservation refusal; serial and parallel admitted populations identical. |
| 9 / Sol / S2 (after E of I7) | new `proof/corpus/Native/LocatorSource.lean`, new `proof/corpus/Native/LocatorCost.lean`, `proof/corpus/lakefile.lean`: establish S1's source-bound directory coverage, publication/relink and charged cost results on the maintained table paths, with inline closed-cone audits. Keep the old-summary Search import out of this predecessor; use S0's parameterized observation and exact generated storage projection. | L(`lake build Native.LocatorSource Native.LocatorCost`) | Omit a generated terminal relink; the both-carrier coverage preservation witness rejects. | Accepted source evidence for the certificate core and cost route, not assumed output coverage; no claim of whole-map W7 preservation. An unrooted transition or uncharged scan blocks I8. |
| 10 / Flash / I8 | `src/table.rs`, `src/table/probe.rs`, `src/table/tests.rs`: Parabolic cutover at `get_at`, prepared reads, `spill_locate` and `spill_locate_held`/rebuilt decisions. Use binary `a` path `Absent`/`Unknown`; retain real first-vacancy selection for insertion. Add live-Unknown, equal-`a`/different-`b`, default-valued, refused-write and both-carrier witnesses. | R | Replace Unknown with default/absence on a live late-hit fixture; public read and insertion-no-duplicate assertions reject. | Q on Parabolic; no parabolic summary read is authoritative; Hyperbolic still uses its old query until I9. |
| 11 / Flash / I9 | `src/table.rs`, `src/table/probe.rs`, `src/table/tests.rs`: same cutover on Hyperbolic using canonical H3_40 scalar paths; bind covered scopes to the routed/excluded storage. Add `0`/`q`, zero residue, unequal Sigma and relocation witnesses. Elliptic gets no locator and retains its old exact results; its existing summary-assisted spill branch remains until D3. | R | Replace Unknown with default for a live hyperbolic collision key after growth; both-carrier public/helper assertions reject. | Q on both filtered faces; neither consumes summary; elliptic result semantics and fingerprint/full-identity confirmation retained. |

I4–I7 are private staging/metadata maintenance, **not a second value store or an accepted dual-map
runtime**. Summary remains the old read mechanism until the individual face cutovers; it is not used
as an extra fallback after a locator Unknown. Manual prepared tuples must carry the same scoped
coverage inputs as production. A wrapper cannot turn an uninitialized directory into covered storage.

**Compile-coupled hold:** do not delete the last summary caller in I9. Elliptic's genuine old spill
read/write/relocation branches keep `summary_maybe` and its counters live until D1/D3; they are not
new dummy callers. I8/I9's counters count only summary operations that still execute, never locator
outcomes relabelled as summary hits. D3 removes that final elliptic shortcut together with its helper,
using exact-address/DirectCover exclusion or full finite identity coverage instead. This temporary
legacy branch is not the summary-free endpoint or an additional elliptic locator. Its measured
post-deletion cost must still satisfy the unchanged Elliptic expectation; a regression stays open.

**S1's exact deliverables (Sol, not Flash):**

- Source-pinned relation from Table's decoded flat/spill observations and handle lifetime to
  `TableStorageCoverage`; constructor, fresh/found writes, prepared shards, grow/rebuild/relink,
  pending commit and parallel assembly each get a real-body/statement/root/owner/control entry.
  Reverse links and uniqueness are required in addition to forward coverage. Both carriers and
  each face have joint reachable success/refusal/default-valued witnesses.
- The face/index/identity signatures consumed by I2–I7, including `ParabolicWord(a)` →
  `ParabolicDual(a,0)` for the existing word encoding, general admitted dual inputs via `fstHom`,
  H3_40 reduction and immutable scope/profile, and exact packed elliptic access. Explicitly reconcile
  S0's abstract `[Field Carrier]` elliptic vocabulary with Rust `Z/2^64`: it is not a field and the
  adapter may use only exact packing/coverage, not a false field instance.
- A population-complete admission decision using the existing ledger, not just i64 witnesses;
  one shared handle-only directory interface and reserved publication/relink signatures; receipts
  that include every binary/ternary/identity/edge/handle/staging allocation. Fix the mapping of
  locator refusals to existing resource errors without inventing positive-capacity `ShardFull`.
- Actual-body extraction feasibility for both carriers and both filtered faces, with no accepted
  unmodeled allocation/seam intrinsic. Allocation stays in the explicit adapter, not in an assumed
  pure reserve model. Freeze wrapper signatures for E; a failure names the smallest unavailable
  primitive and stops before read activation. S1 states the route; E supplies the executed receipt.
- `WorkCounts` linked to actual digit/byte/label/handle/link operations; lookup bound with depth
  64 for Parabolic and 40 for Hyperbolic, bounded identity fanout, retained-key comparisons/copies,
  staging, node initialization, Vec growth/moves, rebuild/relink and refusals charged explicitly.
  An actual-constructor zero potential and a proved aggregate recurrence are required; count
  repeated work, not only input bytes. An assumed `HashDOSBound` argument is not its inhabitant.
  The cost deliverable must distinguish this wiring's finite sweep on Unknown from the final
  population-independent lookup target. If the sweep remains reachable for present keys, name
  the later exact-terminal-to-payload read/write task; do not claim HashDOS closed or release pepper.
- An acyclic proof schedule: generated directory/coverage facts before cutover, then the
  summary-free `GrowableSearch.entryAt`/`Absent` equality/adapter at P3. Do not fix the old summary
  theorem, duplicate Growable search definitions, assume an output invariant, or restart W7 to
  break the cycle. If S2 needs more than its two modules, return ≤3-file proof sub-tasks before
  dispatch, not a single unbounded “prove the adapter” assignment.

**Release of the deletion hold:** lead must receive I8/I9's actual-path controls plus fresh E,
S2's accepted scoped coverage/cost evidence and the population map. A standalone Locator test or
green S0 build does not suffice. Full HashDOS/end-to-end acceptance remains a later joint condition;
the scoped cost evidence must state every remaining sweep rather than conceal it.
Rerun S2's bounded gate/control on each cutover's regenerated identity as separate read-only Sol
validation tasks (no changed files); any changed proof needs a newly owned ≤3-file repair before
release. A proof about the pre-cutover directory does not automatically refine the changed query.

#### Held deletion slices, renumbered after wiring

F1 is already before wiring because it is size-only. The old F2/F3/F4 become D1/D2/D3, singleton
Flash waves after explicit lead release. Their source couplings remain those found in A2, but their
replacement invariant is now the wired locator, **not an unfiltered replacement table**.

| Task | Owned files / change | One gate | One rejecting control | Invariant after acceptance |
|---|---|---|---|---|
| D1 (old F2) | `src/table/probe.rs`, `examples/micro_spill.rs`: delete summary-only feature counters, calls and format arguments; retain sweep/tag/full-compare counters. Locator behavior unchanged. | R | Suppress a genuine spill match; retained public late-hit test rejects. | All-feature diagnostics compile without summary counters; no semantic deletion yet. |
| D2 (old F3) | `src/table.rs`, `src/table/resources.rs`, `src/table/tests.rs`: remove only summary receipt fields, calculations and assertions; retain the complete locator receipt from I5. Keep actual summary allocation/writes until D3. | R | Set a required locator-edge capacity target to zero; the actual prepared receipt witness rejects. | Tags/payloads/locator resources fully accounted; summary receipt alone retired, no proof claim for this transitional shape. |
| D3 (old F4) | `src/table/probe.rs`, `src/table/state.rs`, `src/table/tests.rs`: remove the final elliptic summary branches together with summary storage, sizing, initialization, reservation, methods, remaining calls and test-only counters/publication event atomically. Keep locator calls and live fingerprint publication. | R | Corrupt only a live tag's fingerprint bits; tag/payload agreement and public readback reject. | No summary in executing Rust; payload/locator links committed before a query can observe the live tag; C/Q/T retained. |

D3 retains late-hit, first-fit, both-carrier, unequal-Sigma and no-effects refusal assertions. Replace
the `1usize << 58` summary-sizing fixture with retained checked `slots_for` overflow (for example
`usize::MAX` before doubling), not an allocator failure. Remove summary setup in manual late-hit
fixtures and the ignored timing-test query loop; use an explicitly absent full key. Publication
tests cover payload → tag locally **and** the directory/table atomic handoff; deleting the old
summary event must not delete link-order testing. `coord::PHI`, `tag`, `TAG_LIVE`, shifted
fingerprint bits and `scan::match_tag` remain. D3 must find no new summary dependency outside its
three files; otherwise stop and split before editing. No dead summary shims.

#### Extraction E and proof P

After F1 and each I/D source slice, **E-<source-task>**, Flash, owns exactly
`proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`. One root-cwd gate:

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s sh -ec 'proof/regen.sh; proof/regen.sh --check'
```

Add each real included source to `SOURCE_FILES`: probe first, then address/locator/core/table-locator
as they enter the root cone. Preserve every old root listed in `proof/regen.sh`, especially both
`word_{stored,regen}_table_get_at`, prepared shard get/apply, apply get/insert, relocation entry/all,
pending commit, publication, capacity, schedule and codec/coord roots. Add S1's monomorphic real-body
roots for face encoding, directory decisions, prepared link publication and relink for **both**
carriers; symbolic Sigma or explicit roots must exercise both filtered faces. An extraction wrapper
only calls the production body. Retire an old root only with an audited production replacement map.
There is no top-level summary root to drop: D3's regeneration removes transitive summary helpers,
`Region.summary` and the receipt field. Never hand-edit generated Lean to simulate removal.
Control: corrupt one generated link/write in a disposable copy; `--check` rejects drift and passes
restored. Invariant: source freshness and clean generated audit, not semantic correctness.

The following are separate serialized **Sol** proof tasks after E-D3. Existing statements/grades
stay unchanged; broadened representation needs a proved transport, not extra output assumptions.

| Task | Owned files | One gate, cwd `proof/corpus` | One rejecting control / invariant |
|---|---|---|---|
| P1 | `proof/corpus/Growable/Scan.lean`, `proof/corpus/AxiomAudit.lean` | L(`lake build Growable.Scan`) | Wrong generated equality-scan mask rejects `extracted_eq_swar_exact`; remove only summary-specific lemmas/audits, preserve fingerprint scan laws. |
| P2 | `proof/corpus/Growable/Search.lean`, `proof/corpus/Growable/Relocation.lean`, `proof/corpus/AxiomAudit.lean` | L(`lake build Growable.Search Growable.Relocation AxiomAudit`) | Generated spill tag → zero rejects the both-carrier spill-preservation headlines; revalidate summary-free reads/relocations with unchanged domains. |
| P3 | new `proof/corpus/Native/TableAdapter.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean` | L(`lake build Native.TableAdapter AxiomAudit`) | Generated certificate branch Unknown → Absent rejects the live-key exclusion witness; instantiate `AbsentAdapter`/S0 coverage against actual `GrowableSearch.entryAt`/`Absent`, with the exact S2 observation equality and public-query linkage. Register Native source/cost/adapter headlines in the fail-closed audit. |

S2/P3 deliverables are new targets, not existing theorems. A failing dependency outside P2's two
modules requires a separate ≤3-file regression slice before P2 closes; `Resources`, `Representation`,
`Rebuild` and commit equations are not implicitly owned by P2. The source-linked cost theorem and
all public/full-state links stay required even if these focused modules build. Only after inventory,
fresh regeneration and P acceptance do the summary bound/Shape/Recorded/MissSound become historical
for the successor; `SpillFingerprintExact` remains live.

**Scheduling authorization still required:** the governing roadmap requires immediate full bridge
regression after each source row, but the supplied old-summary Search/Relocation goal is red. The
lead-only edit below proposes a narrowly bounded exception: E/freshness after every slice, S2 before
read activation, and full regression after D3/P. No intermediate source/extraction pass is called
bridge acceptance. Without lead application, that inherited rule blocks the dependent dispatches.
Preserve the old owner's frozen source/result; do not repair its summary goal for this successor.

#### Measurement M and integrated exit

**M0 readiness (lead/harness owner, no implicit edit):** freeze a matched pre-wiring full-profile
receipt on the current summary source before changing timed behavior. Existing default `get`/
`insert_with` strata are **Parabolic**, not Hyperbolic. The admission ledger reports Stored opaque
Strata/Shared/spatial keys, not Regen coverage. Require registered explicit Hyperbolic and Elliptic
populations and admitted exact-identity adapters; if missing, M1 is blocked pending a separately
scoped harness/registration task, not a relabelled default row. Its concrete ≤3-file edits cannot be
finalized before S1 resolves those population capabilities. Preserve all original cells meanwhile.

**M1**, Flash after the full Rust/proof/control barrier, owns exactly
`docs/evidence/anative-w4-reg1.md`, `docs/evidence/anative-w4-reg1.stdout.log`,
`docs/evidence/anative-w4-reg1.stderr.log`. One root-cwd gate, also used for the pinned M0 baseline:

```sh
timeout 3600s cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

Rejecting control: the harness's selected-path slowdown must reject strict acceptance and the
unchanged restored campaign must run; retain all oracle/digest/omission/face/concurrency controls.
Invariant: every registered applicable incumbent pair has interval maximum `<=1.0`; missing or
inconclusive pairs reject. A receipt recording exit 1 is useful evidence, not acceptance.

Expected movements are hypotheses: Parabolic absent-miss must **not regress** against its pinned
summary baseline; Hyperbolic gains the locator exclusion and should reduce covered absent-miss
work; Elliptic has no locator work and unchanged exact behavior, with matched timings checked for
regression. Require the pre-registered before/after uncertainty rule (no-regression upper ratio
bound `<=1.0`, inconclusive stays open), in addition to the unchanged all-incumbent bar. A failing
Parabolic row is not accepted as the price of deleting summary. Present hits/Unknown, updates,
fresh/growth insert, resident misses, memory and parallel cells remain load-bearing; maintenance
can increase their costs. No promise of speed follows from the 64/40-level bound.

Record full source/harness/input/dependency/tool hashes, actual compiler/Lean/Aeneas/Charon identity,
machine, target/features, workers, cache conditions, repetitions, raw samples, intervals and elapsed
time. Full AGENTS §6 Rust, extraction, both bounded Lake builds and controls remain the joint
integration barrier; `10^8` identity and coverage census remain required. No heavy gate ran in A3.

#### Exact governing-document edits — lead only, A3 replacement map

Do not apply the A2 edit map below. Apply these only after each governing file owner releases it;
retain historical receipts and every acceptance/reopening bar.

| File + section + insertion point | Exact text to insert |
|---|---|
| `docs/direction.md`, `Execution order`, before `Serial read specialization and stable parallel scatter` | **Algebra-native wiring before summary retirement.** “The lead holds summary deletion until the real Table uses the face locator on both filtered faces. Follow `docs/epics/algebra-native-addressing.md`, Current dispatch decision: active F1 size-only factoring → Sol source coverage/cost and admission adapter → reserve-all/shared handle directory → serial and parallel maintenance → source evidence → Parabolic and Hyperbolic query cutover → lead release → D1–D3 deletion → E/P revalidation → strict full-profile REG-1. Parabolic indexes `fstHom(a+bε)=a` by a base-2 trie; Hyperbolic indexes one H3_40 scalar residue by a base-3 trie; Elliptic retains exact packing without a locator. Keep the fingerprint as a prefilter requiring full identity. No standalone test or abstract S0 implication is table coverage/cost evidence. This supersedes summary optimization and deletion-first dispatch, not historical receipts. W7, G1 and HashDOS closure remain open.” |
| `docs/kernel-boundary.md`, `Canonical object and executing root`, before `Read specialization and scatter boundary` | **Locator-backed certificate boundary.** “On Parabolic and Hyperbolic, Absent may skip matching only under actual-source TableStorageCoverage for the queried scope, with all excluded storage accounted for. Unknown requires exact search and supplies neither presence nor an insertion vacancy; insertion still needs a valid destination. Elliptic has no secondary locator and retains exact-address coverage. Table owns each payload/value once; directory terminals and reverse links name those payloads, not a duplicate Locator value store. Constructor, fresh/found writes, prepared shards, growth, relocation, pending commit and parallel assembly establish/preserve coverage. Reserve every node, edge, identity, handle and staging buffer before retain/clone/merge; entry-seam refusal alone does not test late allocations. Preserve both carriers and existing API populations unless a separately ratified compatibility decision says otherwise. An Unknown sweep has charged finite cost, not population-independent HashDOS evidence. W7 and all realization exclusions remain unchanged.” |
| `docs/kernel-boundary.md`, `Spill-filter obligations and partition cut`, after its four-row ledger | “The summary obligations remain live on the old source and during staged maintenance. Retire summary-specific bounds, Shape, Recorded and MissSound for the successor only after both-face wiring, lead release, complete Rust/generated storage/read/write/receipt removal and replacement source-linked absence evidence. Preserve old goals/receipts. SpillFingerprintExact, full identity, initialized occupancy, complete coverage, first fit and resource/refusal obligations survive; do not add old summary fields to make the successor bridge compile.” |
| `docs/roadmap.md`, `Serialized table owner and spatial plan`, before `Ownership sequence` | “ANATIVE W4 A3 owns the locator integration schedule. Active Flash F1 finishes its unchanged probe factoring first; no other task owns table.rs concurrently. The ordered singleton task map is in `algebra-native-addressing.md`, Current dispatch decision. S0 is the existing logical contract, not a completed source adapter; Sol S1/S2 and admission feasibility gate query cutover. Both filtered faces must be wired before the lead releases D1–D3 (old F2–F4). No deletion-first fallback dispatch is authorized. Every E retains both carriers and real production roots; P revalidates the summary-free Search/Relocation and Native adapter/audit. M uses the unchanged strict full profile and checks Parabolic no-regression, Hyperbolic locator gains and Elliptic unchanged behavior.” |
| Same roadmap section, immediately after `Mandatory serialized extraction/proof barrier after each source row` paragraph | “For the bounded ANATIVE wiring/deletion chain only, retain extraction/freshness after every source slice, require source-linked certificate evidence before read activation, and defer the already-red old-summary full bridge regression to D3/P completion. This schedules a red integration lane; it does not accept an intermediate representation, waive a proof/control, change a grade or reopen W7. Any new extraction blocker stops the next source slice. Full Rust/Kani, both proof packages, audit and semantic controls must pass on the integrated identity before acceptance or final measurement.” |

#### A3 acceptance and handoff

Documentation grade `anchor`; complete-file whitespace gate passes before and after (exit 0).
Scoped `git diff --check` also exits 0, but the complete-file gate is authoritative for these
untracked lifecycle files. These are documentary checks only.
No runtime, proof, extraction, source mutation or performance acceptance is reported. No governing
edits are applied. Source observations bind to the read dirty files, not HEAD alone; F1 can move
probe line numbers, so dispatch re-pins symbols and hashes after its release.

**Single lead question:** if Sol cannot construct the required face/identity capabilities for the
ledger's arbitrary `Home`/opaque-`Eq` populations, will the lead authorize a separately specified
explicit-capability API migration, or require a compatibility design preserving that entire API?
Until resolved, I3 and the full chain remain blocked; neither answer is silently assumed here.
The scheduling edit is proposed for lead application, not another deletion-order question: the
existing hold is retained. Next bounded task is S1, then I1's late-edge resource repair; no W7 retry.

### Historical founding and A2 records (superseded for current dispatch)

### Arc block

```text
SCOPE: docs/epics/algebra-native-addressing.md; docs/epics/algebra-native-addressing.md;
       concrete modulus/locator, semantic interface, W1 task and exact governing-doc edit map.
REGISTER: architecture/planning; no source, proof, evidence, pin or governing-file edits.
CORPUS: AnyRing.ringComplementary_idem:
        r*r=σ → r*s=1 → 2*h=1 → RingComplementary σ e+ e−;
        accepted corpus-native proved input, not a locator/cost theorem.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/.lake/packages/apeiron/proof/Apeiron/Trichotomy/IdempotentsAnyRing.lean:406-411.
GATE: complete-file whitespace wrapper in Integration exit, repository root.
      Pre-edit scoped git diff --check exit 0; no semantic documentary red invented.
      W1 semantic test target MISSING until authored by its task.
GRADE: anchor before/after; numeric sanity check only, no new Lean/runtime/performance claim.
```

Assigned-checkout documentation exception: own only these two lifecycle files; no build lane,
new branch/worktree, commit or push. Governing edits below are exact owner-applied edits, not
changes across dirty active paths. `APEIRON-ANATIVE-W0-F1` owns
`docs/evidence/anative.md` in parallel; this task neither edits nor accepts it.
Its eventual receipt is input to migration, not invented evidence of current API coverage.

### Decision and evidence identity

**Concrete modulus:** `q = 3^40 = 12157665459056928801`, one 64-bit unsigned residue,
`r=s=1`, `h=6078832729528464401`. `H3_40` is a sealed, validated, immutable profile. Word reduction
is `w<q ? w : w-q`, exactly remainder because `2^63<q<2^64<2*q`; no cast or hash reduction.
Elliptic packing and parabolic spreading remain over `Z/2^64`; `Cell<i64>` is unchanged.

**Finite locator:** sparse 40-level ternary residue trie (canonical quotients by powers of 3),
then a terminal-marked exact-identity byte trie, at most 256 outgoing labels per identity node.
Retain the full key/independently lossless codec, one value and representative per `(Sigma,K)`.
Word lookup has at most 40 residue edges, 8 identity edges, 2048 byte-label comparisons and one
full-key comparison. Sparse storage bounds, complete bidirectional coverage, immutable profile,
checked nonmoving handles and transactional refusal are specified in the spec. No low-byte shards,
load-rounded placement, `q`-sized allocation or collision-list fallback is selected.

These are **design bounds**, not source-linked cost theorems. The absent targets
`Native.Locator.{coverage,lookup_complete,lookup_work_bound,insert_work_bound,sequence_work_bound}`,
safe resource/handle realization and extraction feasibility block W4. `K: Eq` and arbitrary `Home`
closures alone do not provide the equality-faithful byte encoding this locator requires. Do not
silently narrow the current public population to word keys. Address types can be implemented and
law-tested without admitting those populations to a new runtime or claiming HashDOS resistance.

Sol T2 ratification is accepted: single-odd-prime-power **quadratic** splitting is `proved`;
multi-prime scalar CRT is `anchor`, optional and not a predecessor for H3_40. Source-linked sound
exclusion, common-witness presence, adversarial cost and W7 remain open. The scalar embedding
`a ↦ (a,0)` has equal upper/lower reads; no independent marginal selectivity is claimed. Key
retention is decided; the founding literal-erasure question is removed.

| Read input | SHA-256 |
|---|---|
| `docs/epics/algebra-native-addressing.md` | `46d606c3c31bfa998502d13768417552415b97ecae774ff57080f86e61aca6ce` |
| `src/coord.rs` | `3cf98b30bf12aaf6081e025677a8232c4148c2bac202b28f5075286294e585c1` |
| `src/table.rs` | `e568d0c318801304d17ad304a1f719d509de28b35b33764ad6980619e3e4856d` |
| `src/table/state.rs` | `8dad06425b7ae68b47657e97d7f31ece917b0dd87b7c520d7e1b4f321bf64f8a` |
| `src/table/keyrep.rs` | `11249962faf8131325a248c43840e8f5d66b22c190f53ae09115fb4ac94c0ce0` |
| `proof/corpus/lake-manifest.json` | `90acc49824c65ac8b49b7ac9ac9fd76003aa533a2ce74e8a3975be44d8335c7b` |

The manifest resolves canonical `3d66ea3aa2fe1de5997b6b757247454a2f771358`, Mathlib
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. Read `ChannelsAnyRing.lean` and
`IdempotentsAnyRing.lean` distinguish quadratic reads and complementary idempotents.
`proof/README.md` records Lean `4.31.0`, Aeneas `505b6ca35217e7be5c96c3e2f8045edfbdf47291`,
Charon `b104e24fea7d721b71e6c39fd70f26ff20bc0980` / `0.1.254`, extraction Rust
`nightly-2026-08-18` and patches. These are read pins, not freshly executed tool identities.

Supplied bridge goal: `summary_index place len = ok (some index) → index/64 < len` in
Search/Relocation. Prior unmodeled intrinsics, dynamic shifts and context joining remain attributed
to their frozen source. No new attempt, changed premise, summary optimization or receipt is supplied.
The historical strict `pass=561 fail=889` report is not rerun or attributed to the successor.

### W1 founding dispatch record

**ID:** `APEIRON-ANATIVE-W1-T1`. **Owner:** Flash, additive implementation of this fixed contract;
Sol reviews any proposed change to its laws. Pin the above dirty source digests at dispatch; use
`worktree-agent-APEIRON-ANATIVE-W1-T1` with a private target after the active source/proof owner
releases its snapshot/lane. If source digests changed, inspect the affected diff rather than copying
these identities. Do not disturb the active summary bridge to start W1.

**Exactly three owned files:**

| File | Exact change |
|---|---|
| `src/coord.rs` | Add the explicit child-module declaration for `coord::address`; retain existing bodies/signatures and all `Coord`/`Packed` behavior. Correct only the nearby obsolete literal-key-erasure description, stating that exact-map identity is retained |
| `src/coord/address.rs` (new) | Public `HyperbolicProfile`, private-field `Residue`, `AddressValue`, `FaceTableAddress`, additive `FaceCoord`/`FaceHome` capabilities and `WordFaceHome`; only `u64`/`i64` concrete word instances. Profile validation, canonical-residue rejection and total word reduction exactly as specified; read-only accessors and face reporting. No storage, locator, new scalar kernel or old-Home blanket adapter |
| `tests/native_address/main.rs` (new, Cargo auto-discovered) | Required laws below against those library bodies, plus old `WordHome`/`TableAddress` compatibility fixtures. No manifest or old test edits |

**Semantic interface:** `FaceCoord` maps an admitted `(profile,Sigma,key)` to `FaceTableAddress`;
`FaceHome` supplies that same stable/equality-respecting map and cannot mint capabilities from a
placement-only closure. Word instances support all three faces: exact original bit pattern for
elliptic; existing `spread(word).place` for parabolic; canonical H3_40 remainder for hyperbolic.
`AddressValue`'s variant determines its face; `FaceTableAddress` has no `usize` locator and no
unproved `exact_identity` flag. W4 will make this result authoritative for the real `Home`/table
consumer. W1 must not add fields to old `TableAddress`, switch `WordHome`, change carrier encoding,
route a table operation through the new types, or touch `src/table.rs`/`keyrep.rs`/proof roots.
These are staged library API types, not a second table or a proof of the still-old executing path.

**One gate, repository root:**

```sh
cargo test --locked --lib --test native_address --test table
```

The target does not exist before W1; author it first, then confirm the intended missing-API red
before implementation. A compile failure is that baseline only, **not** the semantic mutation.
The finished gate must execute all eight named tests below (none ignored, no name filter), plus
the existing lib/table suites. Record counts and baseline/red/restored exits; zero selected laws
or missing test names block acceptance even if Cargo exits 0.

1. `profile_is_valid_and_rejects_other_parameters`: exact tuple/version, checked `3^40`, width,
   `r*r=1`, `r*s=1`, `2*h mod q=1`; reject altered modulus/exponent/root/half/version.
2. `residue_constructor_rejects_noncanonical`: accepts `0,1,q-1`, rejects `q,q+1,u64::MAX`.
3. `word_reduction_matches_integer_remainder`: `0,1,q-1,q,q+1,2^63,u64::MAX`, deterministic
   boundary neighborhoods and sampled full-width words; oracle is widened `% q`, not the branch.
4. `elliptic_full_word_is_not_reduced`: packing/unpacking the entire fixture including `q`,
   `q+1`, `2^63` and `u64::MAX`; elliptic word `q` stays `q` while hyperbolic `q` becomes zero.
5. `parabolic_uses_existing_spread`: compare to the existing library fold, with high-bit words.
6. `faces_and_profile_are_preserved`: `FaceHome` agrees with `FaceCoord`; deterministic repeats,
   clones and all three variants, including equal numeric values across distinct faces.
7. `signed_words_keep_bits_and_collisions_keep_identity`: `i64::MIN`, `-1`, `0`, `i64::MAX`;
   distinct original word keys `0` and `q` share a hyperbolic residue but remain unequal keys.
   Do not claim this fixture proves table membership or the `Regen` codec.
8. `legacy_word_home_is_unchanged`: existing `WordHome.address` still uses `spread(word).place`
   and the original sub-`2^32` `direct_word` rule at every Sigma, including boundary values.

**One rejecting control:** in a disposable W1 copy, change the new word instance's **elliptic**
constructor from the unchanged `word` to `reduce_word(word)`, leaving the enum tag elliptic.
The same gate must compile and fail `elliptic_full_word_is_not_reduced` at `word=q`
(`expected q`, `got 0`); restored source passes. Do not mutate the real checkout, change tests,
use an absent-symbol compile failure or count a timeout as rejection.

**Completion/grade:** `tested` address laws only. `src/table.rs` currently has 1,619 lines, so
W1 deliberately does not own or enlarge it; size-only factoring is a W4 predecessor, not an extra
W1 file. Keep each owned source ≤1,500 lines and each immediate source directory ≤12 files.
No accepted source-linked address theorem, locator, new production behavior or HashDOS claim results.

### Migration and ordering against W7 and the bridge

This supersedes the founding reservation numbers (in particular its former W1 ratification and
W7 interface row), not their semantic obligations. `W7-SearchCommitBoundary` always means the
existing frozen composition boundary, **not** an ANATIVE interface implementation wave.

1. **Frozen owner first.** The summary bridge owner finishes only its assigned bounded attempt
   on its pinned snapshot, or returns exact goal/log/digests and releases the lane. Its result and
   source remain untouched. Current summary obligations stay live while their source executes.
2. **W0 decisions and W1 additive types/laws.** Consume the accepted T2 algebraic ratification;
   keep W0-F1's admission ledger independently owned. W1 needs neither a new scalar CRT proof nor
   a solved locator cost theorem: it changes no storage behavior or admitted old API.
3. **W2 compatibility and feasibility; W3 source/proof dispatch contract.** Sol must ratify
   source encoding, sound exclusion/common witness, exact-key discrimination, the nonmoving
   resource interface, coverage/preservation and adversarial cost against every current population.
   Freeze bounded actual-source extraction feasibility, its gate/control and a source-linked cost
   route. Missing statements, primitives or successful feasibility receipts block W4. The finite
   locator in the spec supplies an algorithm, not those proofs. No second container or hash fallback.
4. **W4 behavior migration, only after these barriers.** Size-factor oversized `table.rs` first;
   then migrate address/carrier consumers, storage/search/writes, resources and partition/parallel
   consumers in separately owned ≤3-file tasks. W4 is a migration milestone, not permission for
   one giant task. W3 must expand it into serial task waves with exact files, one gate and one
   rejecting control each; no overlapping `table.rs` ownership or dual-write runtime.
5. **Retire by reachability and replacement evidence.** Remove all summary reads/writes,
   reservations/receipts and generated dependencies, then retire the summary word bound,
   `SpillSummaryShape`, `SpillSummaryRecorded`, `SpillSummaryMissSound` for that successor.
   `SpillFingerprintExact` survives until its separate tag filter/writes disappear. Occupancy,
   initialized bounds, complete coverage, identity, resources and search/commit obligations survive.
6. **W7 stays frozen.** Only lead receipt of Sol's source-pinned full-state `apply_insert`
   preservation task for both carriers, joint found/fresh and old-flat/spill-to-successor witnesses,
   rejecting controls, acyclic composition and finite discharge map can reopen it. Every entry
   names real body, exact statement, gate, owner and checked primitive/tool dependency. No assumed
   output invariant, summary deletion or performance pass satisfies this condition.
7. **Joint integration last.** Source-linked adversarial cost, all API/coverage rows, constructor/
   read/write/refusal/rebuild/ordered-build/public links, Rust/proof/control barriers and strict
   full-profile REG-1 on one identity. G1 stays OPEN, `report_complete=false`; freezing W7 does
   not waive its eventual discharge. The pepper stays pending the accepted source cost theorem.

The W4 decomposition below supersedes the founding next-dispatch reservation. Each missing
gate/primitive still requires a separate bounded task, not a fictional ready command. Optional
scalar CRT is not required to instantiate the already-ratified quadratic split.

### W4 ordered slices — APEIRON-ANATIVE-W4-A2

**Decision:** delete the summary from real Rust and regenerate its representation; do not repair
`Growable.Relocation`/`Growable.Search` to accommodate `Region.summary`. Preserve the frozen
bridge owner's source, result, goals and receipts. Start only after that owner releases the lane.
The supplied goal `summary_index place len = ok (some index) → index/64 < len` becomes unnecessary
for the summary-free successor, not proved and not retroactively moot for the old snapshot.

This is a documentation-only assigned-checkout exception, base
`c450c222905c050f793a23332708c230c658f82c`, dirty source. No build lane is used. The ledger records
42 `delete` and 26 `retain` dispositions, plus separate `delete-with-summary`/`regenerate` rows;
those are ledger rows, not an exhaustive count of textual occurrences. Inspection additionally
finds the feature-gated `examples/micro_spill.rs` summary counter reads and the ignored timing
test's `summary_maybe` loop. Both must be handled for all-target/all-feature compilation.

```text
SCOPE: docs/epics/algebra-native-addressing.md; docs/epics/algebra-native-addressing.md;
       serial deletion slices, certificate boundary, root change and lead-only edit map.
REGISTER: architecture/planning; no source, proof, evidence, manifest or governing-file edits.
CORPUS: Native/LocatorSpec.lean, absent_of_projected_missing:
        AbsentAdapter + LocatorInvariant + missing association -> absent;
        SourceLinkObligation is not instantiated for the table.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Native/LocatorSpec.lean:74-90,144-157.
GATE: Integration exit's complete-file whitespace command; repository root; baseline exit 0.
       Known-red Search/Relocation shape debt is supplied, not rerun in this task.
GRADE: anchor before/after; no new proof, Rust test or measurement claim.
```

#### Compile-coupled deletion prefix

Each row is a **singleton serial source wave**, owned by Flash, with an exclusive private target
and `worktree-agent-<full task ID>` after lead assignment. No rows overlap concurrently. Snapshot
the actual dirty inputs before dispatch; never overwrite an owner's changes from HEAD. Use one
exit-status gate **R**, repository root, for each source row:

```sh
sh -ec 'cargo test --workspace --all-features --locked; cargo clippy --workspace --all-features --all-targets --locked -- -D warnings; cargo fmt --all -- --check'
```

R is one conjunctive acceptance command: compile, tests, lint and format must all pass. No new
`allow(dead_code)`, dummy call, renamed dead field, `cfg(test)` production replacement or visibility
widening to suppress warnings. Existing extraction-only allowances are not permission for new
shims. Every control below mutates a disposable copy, must compile and fail a semantic assertion,
then pass restored under R. A missing test, compile failure or exhausted resource cap is not that
control. Add the stated witnesses in the owned test file before changing behavior.

| Order / task | Exact owned files (only these) and edit | Single gate | Rejecting control | Invariant after acceptance |
|---|---|---|---|---|
| 1 / `APEIRON-ANATIVE-W4-F1` | `src/table.rs`, new `src/table/probe.rs`: move the contiguous `spill_locate` through `spill_locate_held` block, including `read_trace`, `read_stats` and `spill_read`, into a textual `include!("table/probe.rs")`. No module namespace, signature or behavior change. | R | Change the moved sweep to terminate at its first vacancy; `spill_sweep_finds_a_late_match_after_a_window_vacancy` and the read late-hit witness reject. | One definition per body; existing tests still access the same `table` names; both files below 1,500 lines (parent loses about 300 from the read 1,619); directory grows from 7 to 8 source files. Summary still fully functional. |
| 2 / `APEIRON-ANATIVE-W4-F2` | `src/table/probe.rs`, `examples/micro_spill.rs`: delete only feature-gated `ReadStats.summary_tests/summary_hits/summary_misses`, their initializer fields, `record_summary` and its calls; remove both diagnostic format fields/arguments. Keep test-only `read_trace` summary counters until F4. | R | Suppress a real spill match in `spill_read`; the existing prepared/public late-hit tests reject. Review the diff additionally for diagnostic-only deletion. | No all-feature example references deleted fields; production summary reads/writes and test witnesses remain unchanged; retained sweep/tag/full-compare counters stay live. |
| 3 / `APEIRON-ANATIVE-W4-F3` | `src/table.rs`, `src/table/resources.rs`, `src/table/tests.rs`: remove `RegionReservation.summary`, both `observe_receipt` initializers/calculation and only receipt-related summary assertions. Keep real summary reservation in `Region::reserve_empty`, and its live-bit tests. Update parent comments to distinguish the forthcoming summary-free certificate path. | R | Change the surviving payload capacity receipt's required length to zero; `prepared_adapter_materializes_vacant_storage_and_carries_the_pending_held` rejects at its existing target assertion. | Tags/payloads still have exact checked receipts; summary reservation still precedes callbacks although no longer carried in the receipt. No unused helpers: `summary_words` remains called by real reservation. This is transitional Rust, not an accepted full resource refinement. |
| 4 / `APEIRON-ANATIVE-W4-F4` | `src/table/probe.rs`, `src/table/state.rs`, `src/table/tests.rs`: atomically remove all summary reads, methods, constants, field, initialization, reservation, publication event and test-only read counters. Publish payload then tag. Remove summary-only tests/imports; replace collision/absence/publication coverage with summary-free witnesses, retaining both-carrier, unequal-Sigma, late-hit, first-fit and refusal tests. | R | Stop `spill_read` at an early empty group while a matching payload is live in the late group; the public and helper late-hit assertions reject. Also corrupt only the seven fingerprint bits of a published live tag; the new tag/payload-agreement assertion rejects. | No summary allocation, access or helper remains; every legacy spill absence uses empty-region or full finite coverage after flat exclusion. Tags, exact fingerprint scan, `DirectCover`, first fit, resources and logical results remain. No locator certificate is claimed yet. |

F4 must also replace `prepared_shard_reserves_before_hold_on_sizing_refusal`'s `1usize << 58`
fixture: its old `SizeOverflow` depended on summary sizing even though its name does not. Use a
load such as `usize::MAX` that overflows the retained `slots_for` checked doubling **before** any
allocation, and retain zero-hold/clone observations. Preserve a separate payload-sizing/refusal
witness rather than relabel allocator refusal as size overflow. Remove summary setup from manual
late-hit fixtures, summary capacity snapshots and the ignored timing test's query-selection loop;
choose an explicitly absent full key for that diagnostic. Replace the publication-order assertion
with payload → tag, and check every live tag equals `tag(address_held(...).place)` without using
summary state. Do not delete general semantic/resource tests along with the summary-only ones.

**Why this compiles without shims:** F2 removes the external diagnostic coupling first. F3 cuts
the receipt/test coupling while the summary itself is still used. F4 then owns every remaining
executable reference in exactly three files; `summary_maybe` and its last callers disappear together,
as do `summary_set`, its publication call and helper tests. Keeping only unused private summary
helpers for a later cleanup is explicitly rejected.

#### Extraction and shape-debt retirement

Each source identity gets a separate serialized extraction task (suffix `E1` through `E4`),
Flash-owned files exactly `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`.
The gate, root cwd, is:

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s sh -ec 'proof/regen.sh; proof/regen.sh --check'
```

Add `src/table/probe.rs` to `SOURCE_FILES` at E1; retain all existing root selections and both
carriers. Negative control: change a generated spill payload/tag write in a disposable copy;
`proof/regen.sh --check` must reject drift, restore must pass. This is a freshness control, not
a semantic proof. Invariant: real source/generated bodies match and generated audit is clean.

There is **no top-level summary extraction root to remove** in the read `proof/regen.sh`.
`table.summary_{index,set,maybe,trailing_zeros}` are transitive dependencies. E4 removes them,
`table.Region.summary` and the summary receipt field by regeneration, never by editing generated
Lean or dropping public-read/relocation roots to hide reachability. Keep
`word_{stored,regen}_table_get_at`, prepared-shard get/apply, apply get/insert,
relocate entry/all, pending commit, publication, capacity, schedule and coord/codec roots.
Keep `table.tag` and scan dependencies. F3/E3 alone does **not** retire the summary debt.

After E4, serialize these proof tasks; they are **Sol-owned**, not Flash proof repairs:

| Task | Exact owned files | One gate (cwd `proof/corpus`, bounded wrapper above) | Control / invariant |
|---|---|---|---|
| `APEIRON-ANATIVE-W4-P1` | `proof/corpus/Growable/Scan.lean`, `proof/corpus/AxiomAudit.lean` | `lake build Growable.Scan` | Remove only the retired `summary_trailing_zeros_eq` and its summary-only support/audit entries; keep scan declarations/audits. A wrong generated equality-scan mask must reject `extracted_eq_swar_exact`. Scan compiles on the summary-free artifact; this focused gate does not claim the whole audit imports build. |
| `APEIRON-ANATIVE-W4-P2` | `proof/corpus/Growable/Relocation.lean`, `proof/corpus/Growable/Search.lean`, `proof/corpus/AxiomAudit.lean` | `lake build Growable.Relocation Growable.Search AxiomAudit` | Generated spill tag write → zero must reject the both-carrier spill-preservation headlines, restored pass. Revalidate existing statements on the summary-free shape; no summary field/premise repair, no narrowed domain. An error outside owned modules stops for a new bounded regression task. |

The generated removal moots **summary-specific** shape/index/set/miss repairs for the successor.
It does not guarantee P2 succeeds: fingerprint, public-read linkage and other existing obligations
remain. `SpillSummaryShape`, `SpillSummaryRecorded` and `SpillSummaryMissSound` become historical
only when successor absence evidence and the removal inventory pass. `SpillFingerprintExact`
remains live. Do not remove its audit/tests or change a headline grade to get a green build.

The older after-every-source full bridge barrier cannot be reported green on the supplied red
Search/Relocation snapshot. The lead-only governing edit below must authorize this deletion-first
ordering before dispatch; until then these rows are ready specifications, **not permission to
skip the governing barrier**. No intermediate source/extraction green closes the migration wave.
Full bridge/control acceptance is required after E4/P1/P2 and again after locator integration.

#### Locator wiring: exact blocked edge, not a fictitious Flash task

The standalone implementation is not the required table adapter. Three read-source blockers prevent
an honest ≤3-file wiring task with an established green acceptance gate:

1. `Table<K,V,H,N,R>` admits arbitrary `Home` closures/opaque `K`; `Locator<K,V>` requires
   `K: ExactIdentity`, with implementations only for `i64`, `u64`, `Vec<u8>` in the read file.
   `WordHome` still ignores Sigma. Neither the ledger's `block`/`narrow` dispositions nor the
   separate `FaceHome` trait authorize breaking existing consumers or silently reducing old home
   words modulo q. A population-complete compatibility/interface decision is missing.
2. `Locator` owns representatives and values. Nesting it next to an independently written table
   value store does not establish `SourceLinkObligation.reachableInvariant`. Every fresh write,
   prepared-shard apply, growth, relocation, pending commit and parallel assembly must maintain the
   same coverage relation. Unknown cannot return default; a fresh missing path cannot be passed as
   an invented old `SearchDecision::Vacant`. One authoritative payload/handle interface is missing.
3. `Locator::insert` retains at line 548 before fallible reservation at 549; `stage` allocates
   its scratch path with `Vec::with_capacity` at 733. The Vec arenas also need a charged growth
   recurrence, not an assertion of nonmoving storage. Standalone coverage tests do not establish
   reserve-all/no-clone-on-Err, safe-resource realization, source extraction or `HashDOSBound`.

**Next bounded task:** `APEIRON-ANATIVE-W4-S0`, Sol contract verdict after Astra's boundary,
owned files `docs/epics/algebra-native-addressing.md`, `docs/epics/algebra-native-addressing.md`,
`proof/corpus/Native/LocatorSpec.lean` (no production edits). Pin an exact admitted encoding/carrier
interface, table-wide coverage and resource handoff, and the generated-body extraction feasibility
target/control. Existing module gate: bounded `lake build Native.LocatorSpec` in `proof/corpus`;
reject an attempted absent conclusion for an unpublished live representative without the required
coverage premise. A successful logical-module gate is not acceptance of source wiring. Missing
actual-source feasibility/cost receipts must remain explicit blockers, with their own bounded tasks.

Only after S0 and its required feasibility receipts can a follow-up architecture slice assign exact
source files for publication/build/parallel integration. Do not dispatch a nominal
`table.rs + locator.rs + tests.rs` edit that hides changes to `Home`, carriers, resources and parallel
construction. The required final read branch is fixed: on Hyperbolic only, table-linked
`Absent(scope)` returns default; `Unknown` continues complete identity search; an exact common live
representative alone supplies presence. Elliptic/parabolic retain their existing non-summary
certificates. Tests must force Unknown with a live late representative, collide word keys `0`/`q`
at residue zero,
separate Sigma, exercise both carriers/default-valued keys and refuse late allocation without effects.
Their rejecting mutation is `Unknown -> default` on that live-key fixture. New roots must call
these real production bodies, with both-carrier certificate/public-read/publication wrappers and
address/locator source digests; wrapper names and prepared interfaces are frozen by S0, not invented
here. Old roots stay until an explicit production-path replacement is audited.

`Apeiron.NativeLocator.absent_of_projected_missing` in `Native/LocatorSpec.lean` is a logical
implication, not the missing Rust
adapter. Instantiate its `AbsentAdapter` against `GrowableSearch.Absent` only on the summary-free
representation. Preserve `CompleteCoverage`, `DirectCover`, full identity and first-fit obligations.
W7's seven premises and reopening condition stay unchanged; no shape cleanup or locator test reopens
`W7-SearchCommitBoundary`. All whole-state preservation/public links and source-linked cost remain.

#### Strict measurement slice

`APEIRON-ANATIVE-W4-M1`, Flash, exclusive measurement host after the integrated source/proof/control
barrier. Own exactly new `docs/evidence/anative-w4-reg1.md`,
`docs/evidence/anative-w4-reg1.stdout.log`, `docs/evidence/anative-w4-reg1.stderr.log`.
Single root-cwd gate (finite campaign bound):

```sh
timeout 3600s cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

Control: existing oracle/digest, workload/face/concurrency, omitted-pair and slowed-path controls
must reject and the unchanged restored campaign must run. Invariant: unchanged registered population,
incumbents and all-pair interval ceiling `<=1.0`, with source/harness/input/tool hashes, machine,
thread/cache settings, raw samples and uncertainty. Record a nonzero strict result as an open gate,
never a green diagnostic. No measurement runs in A2.

**Expected mechanism, not a speed promise:** covered hyperbolic absent-miss cells gain the locator
shortcut; elliptic/parabolic gain no locator and keep their identity/certificate behavior. The
read source's default Sigma is **Parabolic** (`src/cell.rs`), and REG-1 strata call default `get`/
`insert_with`; those cells cannot be relabelled hyperbolic or credited with a locator gain.
Register matched explicit hyperbolic populations in a separately owned harness task if absent;
missing face coverage blocks measurement readiness. Deleting the all-face summary removes an old
parabolic spill-miss shortcut, so unchanged elliptic/parabolic **timings** cannot be promised.
Measure regression as well as improvement. Present-hit, update, fresh/growth insert, resident miss,
memory and parallel cells remain in the campaign; link maintenance can move their costs too.

**Lead question:** authorize the deletion-first prefix and delayed full bridge revalidation (without
closing any barrier), while table-locator wiring remains blocked on S0, or hold all source dispatch
until that contract and feasibility evidence arrive?

### Retained integration reservations

These founding reservations remain for the blocked integration beyond the W4 deletion prefix;
they are not assignments.
Each owns at most three actual files, one real gate and one semantic rejecting control. Existing
source owners release their paths first; no proof attempt uses a changing source identity.

| Arc | Required gate/control retained from founding |
|---|---|
| Consumer/admission ledger | All old `Home` closures, opaque keys, defaults, both carriers, all Sigma and REG-1 populations; a missing population or marginal-as-hit witness rejects readiness |
| Membership/locator/cost | Canonical generic law owner and actual-source bridge identified; missing projection/late exact-key hit and adversarial same-residue sequence controls. Name the exact bounded Lean command when modules exist; targets currently absent |
| Canonical release/pin transport if new authorities are required | Separately authorized release; ≤3 manifest/README files; bounded bridge build; pre-authority pin with required import rejects |
| Size-only factoring | `cargo test --locked --lib --test table`; moved incorrect full-key comparison rejects; no semantic tuning |
| Storage/read/write and summary retirement | Same focused Rust gate; first unequal colliding key must not imply absence; late-hit control, same-residue/full-key and unequal-Sigma witnesses |
| Resources | Same focused Rust gate; a late reservation moved after hold rejects no-clone-on-Err; reserve-all and checked-byte/handle bounds before publication |
| Partition/parallel | `cargo test --locked --lib --test parallel`; reverse same-key chunk order and reject using noncommutative merge and query-order fixtures |
| Address/identity/constructor/read refinements | Generated actual-body semantic controls, closed-cone audit, both-carrier joint witnesses; explicit source/statement links, not formula copies |
| Write/rebuild/build and W7 discharge | Full-state, not pending-key-only readback; finite premise map before lead reopening; no unknown proof work hidden inside one task |
| Strict measurement | `cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls`; unchanged pair/oracle/workload/slowed-path controls and restored campaign |

The inherited rule is: after every source wave, including W1, serialize extraction/freshness and regression
task instances before the next source owner. Extraction owns `proof/extract/lib.rs`,
`proof/regen.sh`, `proof/ApeironWord.lean`; new arithmetic roots require their own approved
selection, never an implied W1 edit. Regression owns at most two affected bridge modules plus
`AxiomAudit.lean`, selected from the actual generated diff. A dependency outside that set blocks
dispatch. For W1, unchanged old generated bodies are expected, not presumed; new uncalled types
have no extraction/refinement credit. W3 must bind them to production for successor acceptance.
The W4 deletion-first exception is proposed above and in the lead-only edit map; until the lead
ratifies that governing change, this inherited ordering still blocks source dispatch on the red bridge.

Lean commands use `flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope
-p MemoryMax=4G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s <command>`, in the named
package; no unlimited heartbeats, raised recursion escape or OOM/timeout-as-red. Shared `target/`,
generated Lean, both Lake caches and the measurement host are exclusive lanes.

### Exact governing-document edits for the lead

Apply only after the relevant file owner releases it. No governing file is modified in this task;
the edit map is its deliverable. Retain old receipts and the unchanged acceptance/reopening bars.

| File and exact section/insertion point | Exact edit |
|---|---|
| `docs/direction.md`, `Execution order`, before `Serial read specialization and stable parallel scatter` | Insert heading **Algebra-native summary retirement** and: “ANATIVE W4 selects deletion of the `PHI` summary, not further summary-bound/shape repair. After the active bridge owner returns its frozen-source result and releases the lane, the deletion-first sequence in `docs/epics/algebra-native-addressing.md` removes diagnostics coupling, receipts, summary consumers and generated summary storage. The 40-level H3_40 residue trie plus exact-identity byte trie is the selected hyperbolic certificate mechanism, but its table admission, resource ordering and source-linked coverage/cost remain blocked as recorded there. Elliptic/parabolic retain their non-summary certificates; the shifted tag fingerprint and `SpillFingerprintExact` remain. This supersedes T3's summary optimization dispatch, not its receipts or acceptance bars. W7 stays frozen; G1 and HashDOS acceptance stay open.” |
| `docs/kernel-boundary.md`, `Canonical object and executing root`, before `Read specialization and scatter boundary` | Insert heading **Summary-free certificate boundary** and: “W4 deletes `Region.summary` and its receipt from the generated representation rather than extending Growable Search/Relocation to the old summary shape. Hyperbolic `Absent(covered scope)` may certify a table miss only under the actual-source `NativeLocator.SourceLinkObligation` adapter and constructor/preservation linkage covering every possible representative. `Unknown` must continue complete identity search, not return default or an insertion vacancy. Elliptic/parabolic retain `DirectCover` direct-empty, empty routed spill after flat exclusion, or complete finite coverage. No duplicate value store, automatic opaque-key encoding or reduced public population is authorized. Locator existence and a coverage flag are not refinement evidence. Both-carrier identity, reserve-all, full-state preservation and all trust exclusions remain.” |
| `docs/kernel-boundary.md`, `Spill-filter obligations and partition cut`, immediately after the four-row ledger | Insert: “For the summary-free successor only, the summary word bound, Shape, Recorded and MissSound retire after all Rust/generated reads, writes, storage and receipts disappear and replacement absence evidence passes. Their old-source receipts and open goals remain historical, untouched. Keep `tag`, `TAG_LIVE`, `TAG_FINGERPRINT_*`, `scan::match_tag` and the live `SpillFingerprintExact` obligation; W4 does not retire this filter. Do not repair the old summary shape merely to unblock `Native.LocatorSpec`'s missing Growable adapter. W7 reopening is unchanged.” |
| `docs/roadmap.md`, `Serialized table owner and spatial plan`, before `Ownership sequence` and qualifying `Mandatory serialized extraction/proof barrier after each source row` | Insert: “ANATIVE W4 is the deletion-first exception to immediate full bridge regression: frozen-owner release → F1 probe factoring → F2 diagnostic-counter retirement → F3 receipt retirement → F4 summary removal, with E1–E4 real-source extraction/freshness after their respective source rows → P1 Scan cleanup → P2 Search/Relocation/audit revalidation. Exact files, controls and conjunctive Rust gates are in `algebra-native-addressing.md`. Each source slice is a singleton wave with no overlapping owner. The known-red old-summary bridge is not repaired between deletion slices and no intermediate state is bridge-accepted; the full bridge/control barrier remains mandatory after deletion. Source wiring waits for S0 admission/coverage/resource/extraction feasibility, followed by its own bounded source/proof tasks and strict M1 REG-1. Preserve the active owner's frozen result; no W7 retry and no barrier or grade waiver.” |

The supplied intent still has an older “odd composite” phrase under Open questions; its corrected
invariants and accepted single-prime-power decision govern this task. Do not reopen the rejected
wholesale scalar pivot or key-retention decision. Intent/handoff historical cleanup is not owned here.

### Integration exit, evidence and next task

Required at eventual integration: full source-linked constructor/read/success/refusal/relocation/
ordered-build/public-method links for both carriers; source-linked cost theorem and semantic controls;
full `cargo test --workspace --all-features --locked`, `cargo clippy --workspace --all-features
--all-targets --locked -- -D warnings`, `cargo fmt --all -- --check`; bounded
`proof/regen.sh --check` and `lake build` in both packages; `10^8` identity row, coverage census,
all named REG-1 incumbents and each interval maximum `<=1.0` on one source/tool identity.
No opaque-key row becomes inapplicable because its encoding is unresolved. Keep source/harness/
input/tool digests, actual versions, machine, metric, interval, uncertainty and finite timeout.
Owned files ≤1,500 lines, ≤12 immediate source files per directory, trusted checker inventory
≤5,000 lines; inventory external trust separately. No proof or source-size waiver.

Unchanged exclusions: allocator/capacity realization, executor/spawn/join/unwind, arbitrary callback
panic/allocation, rustc/LLVM, ABI, GPU/SPIR-V and unsafe container. These do not excuse sequential
search/commit, source resource ordering or full-state preservation. Rust/canonical closed cone and
Slang empty cone remain separate. A cost theorem is not an ns/op measurement; neither replaces REG-1.

This documentation task runs no Rust, Lean, extraction or benchmark gate. Arithmetic sanity checking
with Python confirms `3^40`, 64-bit width, `3^41`, half, maximum-word remainder, one-subtraction
bound and wide-product bound (exit 0); it is not a Lean witness or Rust refinement. The Arc gate
below checks complete files, including untracked content, for whitespace only. `git diff --no-index
--check` returns 1 with no diagnostics for each clean added file; the wrapper explicitly accepts
that difference status, but rejects any diagnostic or unexpected status. W1's gate and semantic
control are specified, **not run or passed** here. The founding temporary clause checker receipt
belongs to the prior document version and does not certify this architecture.

```sh
python3 -c 'import subprocess,sys; paths=["docs/epics/algebra-native-addressing.md","docs/epics/algebra-native-addressing.md"]; results=[subprocess.run(["git","diff","--no-index","--check","/dev/null",p],capture_output=True,text=True) for p in paths]; ok=all(r.returncode in (0,1) and not r.stdout and not r.stderr for r in results); print("complete-file whitespace:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)'
```

**A2 acceptance:** the complete-file whitespace command passes before and after the scoped edits;
this checks documentary whitespace only. No heavy builds, source mutation controls or timing run.
The source/gate/control rows are dispatch specifications, not reports of green execution. No
mathematical proof or implementation refinement is added. The proposed governing edits are not applied.

**Next task:** lead resolves the deletion-first scheduling question above; release the frozen owner
before F1, and request S0 before any locator wiring. The full W4 implementation sequence cannot yet
be called dispatch-ready without inventing the missing admission/resource interface. Three materially
different failed approaches end an attempt with the exact goal/log/digests, not a weaker contract or
W7 retry.
