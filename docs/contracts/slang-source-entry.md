# Slang Source-Entry Fidelity Specification

This file is the specification of the Slang source-entry fidelity mechanism for
. It fixes, for each selected Song Slang body, the exact source identity,
the represented program, the observation relation, the consumed arguments and input grammar, the
state and effects, the refusal behaviour and the rejecting control. It records that a copied
implementation model, a differential suite and equal outputs are **not** a source-connected
refinement, and that changing a selected actual body without changing its represented program must be
rejected.

Authority: [the design notes](../direction.md#primitive-correspondence) is the programme
authority; [the design notes](../kernel-boundary.md) owns the Rust representation and trust
boundary; [`proof/README.md`](../../proof/README.md) owns extraction and bridge scope. The canonical
corpus remains at `../sanguine-apeiron/proof/` at Git
`8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f`, Mathlib `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.
The Apeiron sources were read at revision `9a805439640ff63b5a2028aeb887f50e31cf6351` and the Song
sources at revision `47869e76f982cfa3085a8618a82b80463e22ef46`; the per-file digests are recorded
under [Pins and source digests](#pins-and-source-digests).

This document specifies obligations. It does not claim the correspondence exists. Every entry is
`selected-open`. The machine-readable form is
[`tools/joint/source_entry_cases.json (unpublished)`](../../tools/joint/source_entry_cases.json); the checker
[`tools/joint/source_entry_check.py (unpublished)`](../../tools/joint/source_entry_check.py) establishes
specification consistency only and never reports correspondence.

## Active route and acceptance

The active source-refinement chain is: source arena → checked references → complete memo key →
face-correct addressing → memo value → fixed-shard store → ordered fold → the unconditional
`Slang.SourceAudit.source_binding`. The chain is rooted in the canonical map `Apeiron.Table.HMap`,
`Apeiron.Table.get`, `Apeiron.Table.insertWith` and `Apeiron.Table.build`, and in the canonical
algebra `Apeiron.Trichotomy.AnyRing.RingCell`, `Apeiron.Trichotomy.AnyRing.ringMul` and
`Apeiron.Trichotomy.AnyRing.ringNormForm`.

The `Slang/Derivation` `Layer`/`Derives` tower is retired. The `Slang/Generated/Trace` generated
decoder traces are retired. The `DecodeTrace.Run` representation is retired, the named proof DAG is
retired, and the segment lemma is retired; none is the active mechanism and none is an acceptance
condition. A reference to a retired route carries a retirement marker and is not acceptance.

Specification consistency is this document with `tools/joint/source_entry_check.py (unpublished)`: the text,
identities and records pass green, and no entry closes. Implementation acceptance is the
`tools/slang/refine.py` exact-declaration probe on the unconditional
`Slang.SourceAudit.source_binding`, red until that theorem exists. A green consistency run
establishes neither the correspondence nor the source binding nor any entry's executing refinement.

## Required fields

Every selected entry carries these nine fields. An entry that drops one is incomplete and the checker
rejects it.

| Field | Obligation |
|---|---|
| `song` | The exact Song Slang source file, function, repository revision and sha256 digest of the selected executing body. |
| `represented_program` | The abstract operation the selected body implements, stated independently of the body so a body mutation can be judged against it. |
| `observation_relation` | The relation between the selected body's result and the canonical Lean definition, with the exact canonical declaration names it refines. |
| `canonical_targets` | The exact fully-qualified canonical Lean declarations the represented program is stated against. |
| `input_grammar` | The arguments and byte grammar the selected body consumes, and its trailing-input policy. |
| `state_effects` | The caller-visible state the body reads or mutates and whether any effect can be rolled back. |
| `refusal` | The refusal, unsupported-input and resource-exhaustion behaviour, including the raw return encoding. |
| `rejecting_control` | The named control that mutates the actual body while leaving the represented program unchanged and must be rejected. |
| `missing` | The checked Slang source-semantics link and the executing refinement that are absent for this entry. |

## Selected entries

The selected boundary is the executing Song bodies for which a canonical Lean definition exists and a
profile row requires a source-connected refinement: the table merge, read, write and init bodies in
[`song/src/table/table.slang`](../../../song/src/table/table.slang), and the evaluator bodies in
[`song/src/prover/ring/prover0.slang`](../../../song/src/prover/ring/prover0.slang). No entry is closed.

| Entry | Song file | Song entry | Canonical target | Status |
|---|---|---|---|---|
| SE1 | `song/src/table/table.slang` | `mergeAcc` | `Apeiron.Table.get_build_acc` | selected-open |
| SE2 | `song/src/table/table.slang` | `tableGet` | `Apeiron.Table.get` | selected-open |
| SE3 | `song/src/table/table.slang` | `tableInsertWith` | `Apeiron.Table.insertWith` | selected-open |
| SE4 | `song/src/table/table.slang` | `tableInit` | `Apeiron.Table.build` | selected-open |
| SE5 | `song/src/prover/ring/prover0.slang` | `headReduce` | `Apeiron.Combinator.Comb.headReduce` | selected-open |
| SE6 | `song/src/prover/ring/prover0.slang` | `normalize` | `Apeiron.Combinator.Comb.normalize` | selected-open |
| SE7 | `song/src/prover/ring/prover0.slang` | `hasRedex` | `Apeiron.Combinator.Comb.hasRedex` | selected-open |
| SE8 | `song/src/prover/ring/prover0.slang` | `verdict` | `Apeiron.Combinator.Comb.verdict` | selected-open |

A source-connected refinement of a selected entry is a checked relation between the actual executing
body and the canonical declaration, on a matched domain, under an empty axiom cone. The relation is a
function of the actual body: the recorded digest binds the entry to the bytes the relation is stated
about, and a body change with an unchanged represented program is a change the relation must fail to
accept. None of the relations below is checked; every entry states its relation as an anchor.

### SE1 · mergeAcc scalar addition

**Song.** `song/src/table/table.slang::mergeAcc`, Song revision
`47869e76f982cfa3085a8618a82b80463e22ef46`, sha256
`bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d`.

**Represented program.** The binary merge parameter `f : V -> V -> V` of
`Apeiron.Table.insertWith` and `Apeiron.Table.build`, instantiated at scalar addition:
`mergeAcc(left, right) = left + right` on bounded machine integers. The represented program is the
merge function, not the table that folds it.

**Observation relation.** For a key `x` and a stream `kvs`,
`Apeiron.Table.get (Apeiron.Table.build mergeAcc d kvs) x` is the fold of `mergeAcc` over that key's
values in stream order, the equation named by `Apeiron.Table.get_build_acc`. The relation is stated as
an anchor, not checked; no denotation from the Song machine integer to the canonical scalar exists.

**Canonical targets.** `Apeiron.Table.insertWith`, `Apeiron.Table.get_build_acc`.

**Input grammar.** Two arguments `left` and `right`, each a bounded machine integer. No byte input, no
input consumption and no trailing-input policy; the body is a pure call, not a stream reader.

**State and effects.** None. The body reads no `memory[]` cell and writes no `memory[]` cell. The
in-tree laws `mergeacc_comm` and `mergeacc_assoc` are Song laws, not canonical Lean.

**Refusal and unsupported input.** None. The body admits every pair of machine integers. The in-tree
laws carry the domain `0 <= left, right <= COORD_CEILING`, but the executing body has no guard and
computes for every pair; an out-of-domain value is outside the matched domain but is not refused.

**Rejecting control.** SC1.

**Missing.** MISSING: a checked Slang source-semantics link from the executing `mergeAcc` body to the
canonical merge parameter, and any executing refinement of the fold. `mergeacc_comm` and
`mergeacc_assoc` are in-tree Song laws and do not establish the link.

### SE2 · tableGet read

**Song.** `song/src/table/table.slang::tableGet`, Song revision
`47869e76f982cfa3085a8618a82b80463e22ef46`, sha256
`bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d`.

**Represented program.** The total canonical read `Apeiron.Table.get`: return the value stored at
`key`. The body locates the slot with `tableFind` and reads it with `tableSlotVal`; the represented
program is the read at an occupied key.

**Observation relation.** At an occupied key the raw word returned by `tableGet` is intended to equal
the value read by `Apeiron.Table.get`; `Apeiron.Table.get_insertWith_self` names the read-back
equation at a written key. The canonical read is total with a stored default, while the Song read
returns the sentinel `-1` for a missing key; the relation is stated, not checked.

**Canonical targets.** `Apeiron.Table.get`, `Apeiron.Table.get_insertWith_self`.

**Input grammar.** Arguments `base`, `cap` and `key`; `base` and `cap` describe a caller-owned
`memory[]` extent of `2*cap` cells. No byte input and no input consumption.

**State and effects.** None. A read does not write; `tableFind` and `tableSlotVal` read `memory[]`
only.

**Refusal and unsupported input.** A missing key, an empty probe slot, an invalid capacity or the
reserved key `-1` all return the raw word `-1`, which is overloaded across those cases and is not a
status. The canonical read has no refusal.

**Rejecting control.** SC2.

**Missing.** MISSING: a checked Slang source-semantics link from `tableGet` to `Apeiron.Table.get` and
any executing refinement. The missing-key sentinel and the canonical stored default are unresolved.

### SE3 · tableInsertWith replacement and accumulator write

**Song.** `song/src/table/table.slang::tableInsertWith`, Song revision
`47869e76f982cfa3085a8618a82b80463e22ef46`, sha256
`bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d`.

**Represented program.** The canonical `Apeiron.Table.insertWith f k v m`: for an existing key store
and return `f value old`; for a fresh key store and return `value`. The selected merge is the
accumulator face (`old + value`) on the existing-key path, while the fresh-key path stores `value`
without adding a default. The body is the write operation, not `Apeiron.Table.build`.

**Observation relation.** For an existing key,
`Apeiron.Table.get (Apeiron.Table.insertWith mergeAcc k v m) k = mergeAcc v (m k)`; for a different key
the map is unchanged. The corpus equations `Apeiron.Table.get_insertWith_self` and
`Apeiron.Table.get_insertWith_ne` name the relation. The relation is stated, not checked.

**Canonical targets.** `Apeiron.Table.insertWith`, `Apeiron.Table.get_insertWith_self`,
`Apeiron.Table.get_insertWith_ne`.

**Input grammar.** Arguments `base`, `cap`, `key`, `value` and `merge`; the caller owns a `memory[]`
extent of `2*cap` cells. No byte input; the merge code is an integer in the range
`MERGE_SET..MERGE_ACC`.

**State and effects.** The body writes the located slot in place: `tableSlotSet` for a fresh key and
`tableSlotValSet` for an existing key. The caller memory is mutated and there is no rollback.

**Refusal and unsupported input.** Invalid capacity, the reserved key `-1`, an out-of-range merge code
or probe exhaustion returns `-1` before any slot write. Earlier successful calls in a sequence remain
visible, so the multi-insert is not transactional.

**Rejecting control.** SC3.

**Missing.** MISSING: a checked Slang source-semantics link from `tableInsertWith` to
`Apeiron.Table.insertWith` and any executing refinement. The absent default on a fresh write and the
non-transactional post-state are unresolved.

### SE4 · tableInit empty layout

**Song.** `song/src/table/table.slang::tableInit`, Song revision
`47869e76f982cfa3085a8618a82b80463e22ef46`, sha256
`bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d`.

**Represented program.** The constant-default canonical table `Apeiron.Table.build f d [] = fun _ =>
d`: initialize `cap` slots to the empty key and the zero value. The represented program is the empty
table construction, not a build over a stream.

**Observation relation.** After `tableInit` returns `0`, every slot key reads the reserved empty key;
the canonical object is the total function `Apeiron.Table.HMap K V`. The relation is stated as an
anchor, not checked; no denotation from the caller `memory[]` to `Apeiron.Table.HMap` exists.

**Canonical targets.** `Apeiron.Table.HMap`, `Apeiron.Table.build`.

**Input grammar.** Arguments `base` and `cap`; the caller owns a `memory[]` extent of `2*cap` cells.
No byte input and no input consumption.

**State and effects.** Writes `2*cap` cells: each slot key is set to the empty key and each value to
`0` through `tableInitGo` and `tableSlotSet`. The caller memory is mutated and there is no rollback.

**Refusal and unsupported input.** `tableInit` returns the raw word `-1` without writes when
`tableCapacityOk(cap)` is `0`, that is for a non-positive or non-power-of-two capacity. The raw word is
overloaded and is not a status.

**Rejecting control.** SC4.

**Missing.** MISSING: a checked Slang source-semantics link from `tableInit` to the canonical
constant-default table and any executing refinement. The caller's address extent is a separate caller
obligation.

### SE5 · headReduce one head step

**Song.** `song/src/prover/ring/prover0.slang::headReduce`, Song revision
`47869e76f982cfa3085a8618a82b80463e22ef46`, sha256
`1a59bd3cb3417cce185c70926062d56b057618001a0f3eb8b3ecc867bc21fafb`.

**Represented program.** The canonical `Apeiron.Combinator.Comb.headReduce`: fire one head redex on
already-reduced children. The `TAG_I`, `TAG_IOTA`, `TAG_K` and `S`-head shapes are the four arms; a
term with no head redex is rebuilt.

**Observation relation.** `Apeiron.Combinator.Comb.headReduce_sound` states
`Reduces c (headReduce c)`, so the canonical one-step reduction reaches the reduced term. The Song
body is intended to implement that one step, but no denotation from the Song arena to
`Apeiron.Combinator.Comb` is established.

**Canonical targets.** `Apeiron.Combinator.Comb.headReduce`,
`Apeiron.Combinator.Comb.headReduce_sound`.

**Input grammar.** Arguments `fl`, `fr` and `orig`: two already-reduced child node indices and the
original node index in the Song arena. No byte input and no input consumption at this entry.

**State and effects.** Allocates arena nodes through `newApp` and `newNode` and may set the `SPENT`
flag on arena exhaustion. The input node's children are not mutated and there is no rollback.

**Refusal and unsupported input.** Arena exhaustion is not a refusal at this entry: `newNode` sets
`SPENT` and returns the inert `NIL` node (tag `0`), and the caller `verdict` maps `SPENT` to `FUEL`.
Unsupported input: none at this entry.

**Rejecting control.** SC5.

**Missing.** MISSING: a checked Slang source-semantics link from `headReduce` to
`Apeiron.Combinator.Comb.headReduce` and any executing refinement of the Song arena to
`Apeiron.Combinator.Comb`. The `SPENT` and `NIL` arm is unresolved.

### SE6 · normalize budgeted composition

**Song.** `song/src/prover/ring/prover0.slang::normalize`, Song revision
`47869e76f982cfa3085a8618a82b80463e22ef46`, sha256
`1a59bd3cb3417cce185c70926062d56b057618001a0f3eb8b3ecc867bc21fafb`.

**Represented program.** The canonical `Apeiron.Combinator.Comb.normalize fuel c`: apply `reduceDeep`
at most `fuel` times. The Song body composes `pass` to a fixed point or the source budget
`FUEL_PASSES`.

**Observation relation.** `Apeiron.Combinator.Comb.reduces_normalize` states
`Reduces c (normalize fuel c)`. The Song pass composition is intended to implement the same iteration;
`FUEL_PASSES` is a source constant, not a Lean `fuel` value, and no relation is checked.

**Canonical targets.** `Apeiron.Combinator.Comb.normalize`,
`Apeiron.Combinator.Comb.reduces_normalize`.

**Input grammar.** Arguments `root`, a Song node index, and `fuel`, a pass count. No byte input and no
input consumption.

**State and effects.** Increments the pass counter, resets and reuses the walk stack, and mutates the
memo and memo-stamp arrays. There is no rollback.

**Refusal and unsupported input.** When the source budget is exhausted or `SPENT` is set, the body
returns the current root and the caller reports `FUEL`. Unsupported input: none at this entry.

**Rejecting control.** SC6.

**Missing.** MISSING: a checked Slang source-semantics link from `normalize` to
`Apeiron.Combinator.Comb.normalize` and any executing refinement. The source budget is not a Lean
`fuel` value.

### SE7 · hasRedex floor detector

**Song.** `song/src/prover/ring/prover0.slang::hasRedex`, Song revision
`47869e76f982cfa3085a8618a82b80463e22ef46`, sha256
`1a59bd3cb3417cce185c70926062d56b057618001a0f3eb8b3ecc867bc21fafb`.

**Represented program.** The canonical `Apeiron.Combinator.Comb.hasRedex`: report whether the term
carries a redex. The Song body walks the arena and reports `1` for the `TAG_I`, `TAG_IOTA`,
`TAG_K`-head or `S`-head application shapes.

**Observation relation.** `Apeiron.Combinator.Comb.normal_of_hasRedex_false` states that a clean
report is a `NormalForm`. The Song body is intended to decide the same predicate; the relation is
stated, not checked.

**Canonical targets.** `Apeiron.Combinator.Comb.hasRedex`,
`Apeiron.Combinator.Comb.normal_of_hasRedex_false`.

**Input grammar.** Argument `root`, a Song node index. No byte input and no input consumption.

**State and effects.** Resets and reuses the walk stack. On stack exhaustion the body sets `SPENT` and
the caller reports a redex. There is no rollback.

**Refusal and unsupported input.** Stack exhaustion reports a redex (`1`) through `SPENT`; it is not a
refusal. Unsupported input: none at this entry.

**Rejecting control.** SC7.

**Missing.** MISSING: a checked Slang source-semantics link from `hasRedex` to
`Apeiron.Combinator.Comb.hasRedex` and any executing refinement. The `SPENT` report is unresolved.

### SE8 · verdict normalize-and-compare

**Song.** `song/src/prover/ring/prover0.slang::verdict`, Song revision
`47869e76f982cfa3085a8618a82b80463e22ef46`, sha256
`1a59bd3cb3417cce185c70926062d56b057618001a0f3eb8b3ecc867bc21fafb`.

**Represented program.** The canonical `Apeiron.Combinator.Comb.verdict fuel a b`: normalize two
terms, detect remaining redexes and compare the floors, reporting `equal`, `distinct` or `fuel`. The
Song body parses two prefix terms, normalizes both, detects redexes and structurally compares.

**Observation relation.** `Apeiron.Combinator.Comb.verdict_decided_exact` states that when
`verdict fuel a b` is not `fuel`, `equal` holds exactly when `Joinable a b` holds. The Song `EQUAL`,
`DISTINCT`, `FUEL` and `PARSE` reports are intended to implement that decision; no denotation from the
byte stream and arena to `Apeiron.Combinator.Comb` is established.

**Canonical targets.** `Apeiron.Combinator.Comb.verdict`,
`Apeiron.Combinator.Comb.verdict_decided_exact`.

**Input grammar.** Stdin bytes: `*` application, `k`, `s`, `i` and `u` atoms, digits `0..9` for
variables, whitespace skipped. `readToken` consumes one byte per call and increments the position
counter; `verdict` reads exactly two prefix terms and leaves trailing bytes unconsumed.

**State and effects.** Mutates the arena, memo, stamps, counters and stacks in place. There is no
rollback.

**Refusal and unsupported input.** `PARSE <pos>` with exit `2` on an unknown byte, an incomplete term
or parse-stack overflow; `FUEL` with exit `3` on `SPENT` or a remaining redex; `RING` with exit `4` on
startup ring-claim failure.

**Rejecting control.** SC8.

**Missing.** MISSING: a checked Slang source-semantics link from `verdict` to
`Apeiron.Combinator.Comb.verdict` and any executing refinement. The byte grammar and the arena
denotation are unresolved.

## Non-refinement

These statements are part of the specification, not commentary. A copied implementation model, a
differential suite and equal outputs are **not** a source-connected refinement, and a body change with
an unchanged represented program must be rejected.

| Statement | Record |
|---|---|
| `copied_model` | A copied implementation model is NOT a source-connected refinement: a Rust or Lean re-implementation of the selected Song body, however faithful, shares no checked denotation with the executing Slang source and closes no row. |
| `differential_suite` | A differential suite is NOT a source-connected refinement: equal or differing outputs on sampled inputs are evidence about two executions, not a checked denotation of the Slang body by the canonical definition. |
| `equal_outputs` | Equal outputs are NOT a source-connected refinement: agreement on a test corpus does not establish that the executing body denotes the canonical object on its matched domain. |
| `body_change_rule` | Changing a selected actual Song body without changing its represented program must be rejected: the source-semantics link is a function of the actual body, so a body mutation that preserves the stated represented program is exactly the change the link must fail to accept. |

## Missing artifacts

- MISSING: a checked Slang source-semantics link for every selected entry. No denotation from the
  executing Song Slang body to its canonical Lean definition exists.
- MISSING: an executing refinement for every selected entry. No Song-to-Lean refinement theorem for
  any selected body exists, and every entry stays `selected-open`.

## Rejecting controls

Each control is a specification-level mutation to run in a throwaway copy. The checker verifies only
that the control is named, well formed and attached to an entry; it does not execute the control and
does not claim the control goes red. A control that cannot fail has no weight. Every control mutates
the actual body while leaving the represented program unchanged; the source-semantics link must reject
it.

| ID | Entry | Control |
|---|---|---|
| SC1 | SE1 | In `song/src/table/table.slang::mergeAcc`, replace the addition with a subtraction. |
| SC2 | SE2 | In `song/src/table/table.slang::tableGet`, read the key cell instead of the value cell at the located slot. |
| SC3 | SE3 | In `song/src/table/table.slang::tableInsertWith`, drop the existing-key accumulator addition. |
| SC4 | SE4 | In `song/src/table/table.slang::tableInitGo`, stop writing the value cell so only the key cell is initialized. |
| SC5 | SE5 | In `song/src/prover/ring/prover0.slang::headReduce`, change the `TAG_I` arm to return the left child. |
| SC6 | SE6 | In `song/src/prover/ring/prover0.slang::normalize`, remove the fixed-point test so the body always recurses on the budget. |
| SC7 | SE7 | In `song/src/prover/ring/prover0.slang::isHeadRedex`, drop the `TAG_IOTA` redex case. |
| SC8 | SE8 | In `song/src/prover/ring/prover0.slang::verdict`, swap the `sayEqual` and `sayDistinct` branches. |

## Not covered

- Semantic correspondence itself: this profile and its checker establish specification consistency
  only and never report correspondence, refinement or proof.
- A checked Slang source-semantics link for any selected entry; none exists.
- An executing refinement for any selected entry; none exists.
- rustc/LLVM code generation, the generated machine code and the kernel ABI.
- Runtime linkage between Apeiron and Song; the required result is a checked correspondence, not
  linkage.
- Song local registers, ontology and commands.

## Pins and source digests

The checker resolves and verifies these pins; a drift in any revision or digest fails `--check`.

| Item | Revision |
|---|---|
| Canonical corpus (`../sanguine-apeiron/proof`) | `8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f` |
| Mathlib | `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f` |
| Apeiron sources | `9a805439640ff63b5a2028aeb887f50e31cf6351` |
| Song sources | `47869e76f982cfa3085a8618a82b80463e22ef46` |

| Source | sha256 |
|---|---|
| `song/src/table/table.slang` | `bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d` |
| `song/src/prover/ring/prover0.slang` | `1a59bd3cb3417cce185c70926062d56b057618001a0f3eb8b3ecc867bc21fafb` |
| `Apeiron/Table/Basic.lean` | `e7558bac1d255643f58100f5a1adf32926a423fe6b49e7c991f8d3a3b0eb26b9` |
| `Apeiron/Combinator/Basic.lean` | `8b5629081130e7b53057b2b797b11e8f8340a34a0e0cfca5ca91282f39523c5c` |
| `Apeiron/Combinator/Verdict.lean` | `083c46cd1e78287e761da2703c485ac9c9566586987a57067d1d5a4d0878c455` |

## Checker

`python3 tools/joint/source_entry_check.py (unpublished) --check` validates
`tools/joint/source_entry_cases.json (unpublished)` against this document and the pinned sources: every selected
entry carries all nine fields and stays `selected-open`; every named Song source exists, its recorded
sha256 matches the working-tree bytes and its recorded revision matches the resolved Song checkout
and the blob at that revision; every named canonical target resolves as the exact fully-qualified
declaration in the pinned corpus under its real namespace; every named reference, including control
targets and any file/symbol reference in the entry text, is parsed under the strict fail-closed
grammar and resolves; the corpus, Mathlib, Apeiron and Song revisions match the resolved checkout; the
non-refinement and missing-artifact records are present; and the spec carries the actual entry
records and fields, not incidental mentions. It exits nonzero on a missing field, entry, control,
target or source, on a digest or revision drift, on an unresolved or malformed reference, on a
Markdown field or status that disagrees with the JSON, or on an entry that is not `selected-open`.

`--controls` and `--self-test` mutate a deep copy of the loaded cases and the spec text in memory and
require each mutation to be rejected, including a removed field or entry, a changed digest, a changed
revision, a missing or law-only Song symbol, a renamed or mislabelled canonical target, an unresolved
or malformed named reference, a closed entry, and a removed non-refinement or missing-artifact record.
The working tree is never touched. The checker establishes specification consistency only; it never
reports correspondence.
