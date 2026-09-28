# Extraction requirements — what Aeneas needs from the source

Analysis only. No decision, no root-set change, no claim of `proved`. Derived from the record:
`proof/README.md`, `proof/regen.sh`, `proof/patches/`, `docs/evidence/anative.md`,
`docs/epics/storage-refinement.md`, and the storage-extraction history (`f6db050`, `57a83ba`, `f7d77e4`,
`48333f0`, `d7cf517`, `e848256`, `9a80543`).

## The pipeline and its single acceptance test

Rust real source (`#[path]` from `proof/extract/lib.rs`) → Charon (LLBC) → Aeneas (Lean) →
`regen.sh` audit → bridge (`proof/corpus`, `#audit`, closed cone).

The extraction stage has exactly one acceptance test: `regen.sh`'s audit rejects any generated
`axiom`/`sorry`/`admit`. So:

> **Every function in the reached call closure must be either modeled or lowered away.** A reached
> external with no model becomes a generated `axiom`, and the audit is red. There is no "declare and
> justify an axiom" path in the Rust/Aeneas cone; the Slang cone is stricter still (empty axioms).

This is the invariant the design must satisfy. It is not negotiable at task time.

## Two requirement classes

### A · Lowering / shape (Charon → Aeneas)

The reached bodies must be in a form the lowering passes handle. Recorded boundaries, with the exact
failure the record shows:

| Requirement | Failure if violated | Recorded site |
|---|---|---|
| Keep `--ops-to-function-calls`, `--index-to-function-calls`, `--no-gen-tuple-structs` | `Could not find core::array::repeat`; `unexpected type: TSlice`/`TArray` on every index/repeat | preset probe (`anative-extraction-preset-probe.md:32-45`) |
| Native array repeat `[const { e }; N]` | `Unknown trait found: … Option<(K,V)>: Copy … Ambiguity` | patched Charon `Repeat` + Aeneas `array::repeat` (`proof/patches/`, `d7cf517`) |
| No `?` on `Option`/`Result` on the reached path | desugars through `Try::branch`, unmodeled | `proof/README.md:164-166` |
| No `?` inside a loop body | `Early returns inside of loops` | probe round 1 (`:163`) |
| No in-loop guarded `match` join | `Could not match the contexts` | probe round 1 (`:164`) |
| Avoid inline shared-ref `match &x` | macro-origin `Not implemented yet` | probe round 1 (`:165`) |
| No iterator closures / desugared `for` | unsupported iterator lowering | `48333f0` |
| Drop unmodeled supertraits | `Aeneas.Std.core.fmt.Formatter` axiom via `Debug` | `proof/README.md:469-472` |

Pure combinators and integer helpers (`Option.{is_none_or,and_then,filter,copied}`,
`Result.{ok,and_then}`, `Usize.{saturating_mul,div_ceil,checked_next_power_of_two}`, `TryFrom`) are
removable by **exact source substitution** — done in round 1 (12 of 23 axioms).

### B · Modeled externals

What remains after round 1 (11 names) is not shape-removable; it needs a model:

| Class | Names | Status |
|---|---|---|
| allocator/growth | `Vec.{capacity,try_reserve,try_reserve_exact,truncate,clear}`, `TryReserveError`, `mem.{size_of,drop}` | the pinned Aeneas `Vec` is `structure Vec (α) where slice : Slice α` (`Std/Vec.lean:28`) — no capacity field, so reservation outcome and capacity are **not expressible** without changing the representation |
| adapter combinators | `Result.is_err`, `Option.{as_ref,Try.branch}` | modelable over the existing representation (Astra verdict) |

## Historical precedent: the fixed-shard design reached only modeled things

The fixed-shard `SafeTable<K,V,N>` was extracted **including its write path** and fully refined:
inline `[const { None }; N]` arrays, `ShardFull` refusal, `PhysicalInv`, `CallbackLawsG`,
`insert_with` preservation (`proof/README.md:261-286`). Its extraction difficulty was the array
repeat, resolved by **patching both upstream tools** (`proof/patches/`). One residual was *not*
resolved: the **generic** `SafeTable::<K,V,N>::new` could not resolve `Option<(K,V)>: Copy`, the root
was reverted, and bridge witnesses construct the generated state directly (`d7cf517`). That is a
recorded obligation transfer, not a proof — and it is the precedent that should have bounded
expectations, not a licence to transfer without recording.

The growable flat/spill table replaced fixed arrays with `Vec` growth. Growth reaches the unmodeled
`Vec` reservation family; that is the whole origin of the 11-axiom residual. The design choice, not a
shape defect, is what moved the write path out of the modeled closure.

The ratified spec already encodes the consequence: allocation is "an unextracted trust boundary with
a simulation statement" (`docs/epics/storage-refinement.md:171-178`), and "planning, reservation and
allocating public build/insert methods are not roots under the resource [contract]"
(`docs/epics/storage-refinement.md:352-353`, `:373`). So the inner-cone root set is the **current
contract**, not a new idea — but its cost must stay visible.

## Three legitimate responses — and the one that is not

1. **Patch the tool.** Precedent: the two repeat patches in `proof/patches/`. Appropriate when the
   external is generic and belongs in the backend model.
2. **Reshape the source exactly.** Precedent: round 1. Appropriate only for pure operations with an
   exact equivalent; the extractor-shaped forms are kept with reasoned lint allows.
3. **Model the external.** A `FunsExternal`/`TypesExternal` model file is Aeneas's intended hook
   (`-split-files` emits those templates; probe `:101`), or a Std extension. Appropriate for
   effectful operations defined over the existing representation.

**Removing the root is not a fourth option.** It does not discharge the obligation; it deletes the
extracted artifact and the refinement theorem. If a root is removed, the obligation must be recorded
as an **explicit transfer** to a named, testable premise — never presented as a resolution. That is
what "the boundary" must mean.

## The decision the design owes (before any root-set change)

The growable write path cannot be both in the cone and axiom-free without one of:

- **(a) extend the backend `Vec` model** to carry capacity and reservation outcome (a third patch,
  in the existing spirit but deeper: it changes the shared Std representation and its laws);
- **(b) restructure the resource interface** so the reached closure calls only modeled functions —
  allocation hoisted to a single checked reserve entry or passed as a parameter with a Lean spec, so
  the write path extracts axiom-free and the concrete-allocator conformance is one named, testable
  lemma. The `one-allocation` branch (`f78ce71`) is the nearest attempt ("sound, Miri-clean, and not
  yet a net win");
- **(c) adopt bounded-fiber (no allocation)** — provable by construction, at the contract/performance
  cost Sol recorded (`docs/kernel-boundary.md` "Prior selection"; `docs/flash-handoff.md:45-52`).

Dropping the write roots is (a/b/c) deferred, not chosen.

## Process gate this implies

A design change that adds an external call to the reached closure must name, **in the plan before
implementation**, which of (a/b/c) resolves it. A task that discovers an unmodeled external
mid-implementation is a planning failure: stop, extend this document, and escalate — do not reshape
the corpus's claims to fit the extractor.

## Trust boundary

This document makes no `proved`, `tested` or `measured` claim. It records a structural constraint and
a decision point. Whether (a) or (b) is achievable on the pinned toolchain is itself unestablished;
ascertaining it is a spike with a named command, not an assumption.

## Cost determination (a / b / c)

Method: read the pinned backend `target/aeneas/backends/lean` and the performance record. No build
was run. **Runtime cost and proof/tooling cost are separated; only runtime cost affects "fast".**

### (a) capacity-carrying `Vec` model

- **Runtime: zero.** The executing `src/` is untouched; only the Lean model changes. This is the
  direct answer to the performance concern.
- **Feasibility: measured.** A throwaway patch (one file, `Aeneas/Std/Vec.lean`, 79/24 lines) makes
  the entire Aeneas Lean library compile green (`lake build Aeneas`, 1704 jobs, rc 0), with no other
  file changed. See `docs/evidence/vec-capacity-model-spike.md` and
  `docs/evidence/vec-capacity-model.patch`. The remaining work is the reservation WP spec, the filled
  `FunsExternal.lean`, and the refinement — not feasibility.
- **Mechanism (determined, not guessed).** Aeneas `-split-files` emits
  `FunsExternal_Template.lean` with the header "rename it to `FunsExternal.lean` and fill the holes"
  and every external's exact signature. Filling a hole replaces its `axiom`. `regen.sh` must adopt
  `-split-files` (it currently emits the axioms inline in `ApeironWord.lean`).
- **7 of 11 fill non-vacuously over `{slice}`:** `TryReserveError` (opaque type), `Vec.truncate`,
  `Vec.clear`, `core.mem.drop` (with a no-destructor bound), `Result.is_err`, `Option.as_ref`,
  `Option.Try.branch`. The eleventh, `core.mem.size_of`, is layout, not `{slice}`.
- **3 cannot, over `{slice}`:** `Vec.capacity`, `Vec.try_reserve`, `Vec.try_reserve_exact`.
  Proof: `Vec` is `structure Vec (α) where slice : Slice α` and `Vec.eq_iff` identifies by element
  list, so any `capacity` is a function of length. A non-vacuous `try_reserve additional` must
  establish `capacity ≥ len + additional`; with `capacity ≡ len` and `additional > 0` that is
  unsatisfiable on success, so every real reservation would have to return `AllocationFailed` and the
  write path would be provably dead. A non-vacuous model therefore requires `Vec` to **carry
  capacity** — a representation change, exactly Astra's finding.
- **Patch surface (counted):** `Aeneas/Std/Vec.lean` is 720 lines with **7** `{slice := …}`
  construction sites, **20** `rust_fun` definitions, and only **12** uses of
  `Vec.ext`/`Vec.eq_iff`/`Vec.val_from`/`Vec.from_val` across the whole backend. The change is
  contained to `Vec.lean`; the bridge's own `Vec` use is confined to the unfinished `Growable/*`
  files plus 2 uses in `WordRefinement.lean`.
- **Unavoidable cost:** `proof/lakefile.lean` `require`s the Aeneas Lean backend, which is built from
  source (no cached `.olean`; Mathlib `fabf563a…` is its dependency). A model patch invalidates that
  build, so the next `lake build` pays the backend rebuild — bounded, one-time per model revision.
- **Residual trust boundary:** the model carries capacity and the reservation *outcome*; "the real
  allocator conforms" is then one named, testable premise (`ResourceAdapterSimulates`), not an
  axiom. `size_of` stays a layout assumption (`size_of::<u64>` is 8; `size_of::<FlatSlot<R,V>>`
  needs a layout bound or a source restriction to a fixed-width element).
- **Formal effect:** the write/growth path stays in the extraction cone and keeps a refinement
  theorem.

### (b) parameterize/hoist allocation

- **Runtime: uncertain.** The nearest experiment, `one-allocation` (`f78ce71`), changed the
  allocation layout and measured **not a net win** (58 win / 13 tie / 37 loss, against the
  baseline's 58 / 18 / 30; the outlier it targeted was not causal). A refactor that only moves *where*
  the fallible call happens, without touching layout or the hash/placement path, can be
  perf-neutral — but that is not established.
- Source: resource-path restructure, >3 files. Formal: write path in cone; adapter conformance a lemma.

### (c) bounded-fiber / no allocation

- **Runtime: the recorded blocker** (`docs/kernel-boundary.md` "Prior selection"; the `256·N` ceiling
  and reduced admission are fixed-shard costs). The growable design was selected *for* performance.
- Formal: provable by construction.

### Summary

| option | runtime perf | proof/tool cost | write path |
|---|---|---|---|
| (a) capacity-carrying model | **0** (allocation code untouched) | **landed**: model patch + `strip-modeled-axioms.py`; 0 axioms over all 43 roots, `regen.sh --check` green | in cone, proved |
| (b) parameterize/hoist | unknown (possibly 0) | medium: resource-path refactor | in cone, conditional |
| (c) bounded-fiber | the recorded cost | design reversal + contract retirement | in cone, proved |

(a) is the only option whose runtime cost is zero by construction. It is also the only one that keeps
the write path proved without changing the design. Recommendation: **(a)**.

### The spike that would determine (a) — proposed, not run

In a throwaway backend copy: add `capacity : Usize` to `Aeneas.Std.alloc.vec.Vec`; define
`capacity`/`try_reserve`/`try_reserve_exact`/`truncate`/`clear`/`with_capacity` with specs; adjust
`Vec.eq_iff`/`Vec.ext` and the `@[simp]` lemmas; `lake build` the backend under the lane lock;
re-extract with `-split-files` and filled externals; assert the generated axiom count is 0 and that a
reservation-failure branch is inhabited (non-vacuity control). Cost: the Aeneas + Mathlib rebuild,
exclusive on the shared build lane. This determines the patch cost rather than estimating it; it is
excluded from this note.
