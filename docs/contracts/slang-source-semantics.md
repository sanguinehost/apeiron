# Selected Slang Source Semantics

This specification fixes the source language fragment, the exact selected bytes, and the re-rooted
refinement contract used to interpret the selected table bodies in
`~/song/src/table/table.slang`. It is an obligation for later checked artifacts,
not an executable semantics, proof, source-to-Lean refinement, or primitive correspondence result.
`tools/slang/spec_check.py` checks this document's consistency as text; it does not read the working
implementation headline. Implementation acceptance is the `tools/slang/refine.py`
exact-declaration probe on the unconditional existential.

Authority: [the design notes](../direction.md#primitive-correspondence),
[`joint-primitives.md`](joint-primitives.md), [`joint-table.md`](joint-table.md), and
[`slang-source-entry.md`](slang-source-entry.md). The canonical corpus is the checkout
`../sanguine-apeiron/proof` at Git `3d66ea3aa2fe1de5997b6b757247454a2f771358`, with Mathlib
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.

## Required obligations

Every later definition and theorem MUST retain all rows. <!-- OBLIGATIONS -->

| ID | Obligation | Required checked result |
|---|---|---|
| O01 | Word representation | One 64-bit word carrier, two's-complement signed view, and exact wrapping arithmetic. |
| O02 | Operators | Signed comparisons; exact bitwise `and`, `or`, `xor`; strict arithmetic evaluation; and rejection of reachable shifts, division, and remainder. |
| O03 | Binding and sequencing | Function-local binding/rebinding, statement order, calls, recursion, and early-return propagation. |
| O04 | Conditional evaluation | Lazy inline and statement conditionals; eager bitwise `and`/`or` with both operands evaluated. |
| O05 | Memory | Initialized raw `memory[...]`, valid extents, exact loads/stores, and unchanged surrounding memory; raw memory is decoded-fragment state, not the active refinement object. |
| O06 | Termination | Well-founded source derivations for initialization and both bounded probes. |
| O07 | Resources | Separate source counts for operations, probes, memory actions, calls, and maximum call-stack entries. |
| O08 | Exceptional boundaries | Invalid backing memory, machine-stack exhaustion, and arena-capacity refusal are separate outcomes, never encoded as table `-1`. |
| O09 | Timeout | Interpreter or checker timeout is inconclusive and proves neither success nor refusal. |
| O10 | Source identity | Exact bytes, repository identity, imports, constants, declaration boundaries, and reachable call closure. |
| O11 | Parsing fidelity | Grammar-derived parsing consumes every selected body; unsupported reachable constructs reject the binding. |
| O12 | Non-substitutes | A digest alone, ISA records, sampled agreement, or a handwritten Lean algorithm with a matching filename proves no source semantics. |
| O13 | Representation relation | `RepMemo(store, h)` relates the executing memo store to `Apeiron.Table.HMap MemoKey MemoValue` through the pointwise invariant; the retired `RepSong` relation is not the active refinement. |
| O14 | State-aware lookup | A memo hit returns the checked evidence-bearing value; absence is distinguished from a present value equal to the default; the retired `observeRead` relation is not the active refinement. |
| O15 | Classification | Success/refusal follows checked execution plus the operation admission predicate, never `return != -1`. |
| O16 | State and result | Every theorem preserves both the raw return word and the resulting state. |
| O17 | Memo refinement shape | Constructor establishment, read correctness, replacement-write correctness, exact admission, and termination over the memo store, including `N = 0`, `ShardFull`, and the non-storage refusals with joint witnesses. |
| O18 | Non-vacuity | Joint hypotheses have checked constructor, success, and refusal witnesses at the claimed types. |

## Selected source and identity

The binding input is the byte sequence of each file below, not normalized text and not a filename.
Line endings, comments, declaration text, and imported bytes are part of identity. The Song checkout
is Git `aeaf3baac434b2e3c01154d275bea65fdadc69bf`.

| Role | Exact source | sha256 |
|---|---|---|
| selected execution | `song/src/table/table.slang` | `bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d` |
| imported constant | `song/src/kernel/std.slang` | `e24b0e360f26794fab0fe1332c32455143d22d2059e871b8c012b4e8a89a6cae` |
| textual import edge | `song/src/kernel/mem/cell.slang` | `35e3e46c5b7820c67f9c92f2af07a16bf581964b90acf7c508c5b13570d9b3cc` |
| grammar evidence | `song/src/front/parser.slang` | `e643bd85ca34b195888a0c64ec3662ce75c0c770d365b360285a9e0af2e4a882` |
| expression parsing | `song/src/front/expr.slang` | `6034b4e3b3aa60bd5bb9d767db24a5c5b33804473b4b4aba5be1dde8e2b865d4` |
| AST evidence | `song/src/front/node.slang` | `db096abda00e40ddf6114db7d8da8590b4e03e46390eb4fd38ce0477679b2d6c` |
| prover-form bridge, not execution lowering | `song/src/front/bridge.slang` | `6d3aef63833ac4c2eaf99c71b044b7ae22346a7020c414ceb73ad03f5b4466c0` |
| excluded ISA evidence | `song/src/emit/isa.slang` | `da75ad47bbbbb4ddb8a22c2d5d2729910583f0e37fcec6a198aa7257dd5e79f9` |

`song/` in this table denotes the sibling checkout `~/song/`; it is not an
Apeiron subdirectory. The selected file imports `src/kernel/mem/cell.slang`, which imports
`src/kernel/std.slang`; `COORD_CEILING = 8796093022208` resolves from that closure. The selected table
bodies access raw `memory[...]` in `tableSlotKey`, `tableSlotVal`, `tableSlotSet`, and
`tableSlotValSet`; they do **not** call `cellGet` or `cellSet`. Raw memory is decoded-fragment state:
it is parsed and checked for source identity, and it is not the active refinement object.

## Declaration closure

The source binder MUST recover exact byte spans for these constants:

`TABLE_PRESENT`, `TABLE_EMPTY`, `TABLE_STRIDE`, `MERGE_SET`, `MERGE_MAP`, `MERGE_ACC`,
`MERGE_GRAPH`, `TABLE_PHI`, and `COORD_CEILING`.

It MUST recover exact parameter lists and complete brace-delimited bodies for this closed reachable
function set:

`tableSpread`, `tableSlotKey`, `tableSlotVal`, `tableSlotSet`, `tableSlotValSet`, `tableInitGo`,
`tableInit`, `tableCapacityOk`, `tableFindSlotGo`, `tableLookupSlotGo`, `tableFind`, `tableGet`, and
`tableInsertWith`.

Reachability starts independently at `tableInit`, `tableFind`, `tableGet`, and `tableInsertWith` and
closes over calls in the selected bodies. Contract, carrier, metadata, test, compatibility wrapper,
and `mergeAcc` declarations remain outside execution closure. Their intervening bytes still matter to
declaration-boundary parsing. Calls resolve in Slang's flat namespace after textual `use` expansion.

## Grammar and source binding

The decoder MUST be derived from the Slang grammar, not from ISA records or an instruction lowering.
For each selected declaration it MUST tokenize the exact bytes, pair delimiters, parse the declared
parameter list and entire body, and end exactly at the matching closing brace. Skipping an opaque
statement, accepting a parsed prefix, treating an unknown token as whitespace, or retaining trailing
tokens inside a selected body is rejection. Imports MUST resolve repo-root-relative quoted paths and
MUST reject cycles, duplicate conflicting declarations, missing files, and unresolved names.

Parsing provenance is layered: declaration and statement parsing reside in
`song/src/front/parser.slang`, and expression parsing resides in `song/src/front/expr.slang`.
`song/src/front/bridge.slang` translates AST arena nodes into prover linear forms; it is a
prover-form bridge, not execution lowering, and it MUST NOT supply the source semantics. Frontend
call and load nodes can retain opaque source spans; the decoder MUST fully parse those spans into the
admitted fragment and MUST NOT inherit their opacity.

The admitted fragment is decimal integer constants; names; calls; raw memory indexing and the store
statement `memory[address] = value`; local `=` binding/rebinding; `return`; expression statements;
semicolon/newline sequencing; braced `if` with an optional `else`; lazy `a if c else b`; parentheses;
`+`, `-`, unary `0 - x`, `*`, `and`, `or`, `xor`; and `is`, `is not`, `<`, `<=`, `>`, `>=`. A store
`memory[address] = value` is a statement: it evaluates `address` first and `value` second,
left-to-right, then writes the value at that coordinate. Only constructs reachable
from the selected entries need semantics. Any unsupported reachable construct MUST reject source binding.
Top-level `carrier`, `law`, `spec`, `measure`, metadata, comments, tests, and nonselected declarations
are parsed only far enough to establish nonoverlapping declaration boundaries; they never add an
execution premise.

A source digest establishes byte equality only. It proves neither grammar fidelity nor semantics.
The records in `song/src/emit/isa.slang` describe a later carrier and are not a shortcut from source
bytes to behavior. A handwritten Lean algorithm that resembles the table and is stored under a
matching source filename is unacceptable: the checked decoder, declaration spans, closure, and
machine relation must depend on the exact selected bytes.

### Entry and closed-program indices

`ParsesClosure bundle entryProgram` is indexed only by the table-entry program returned by
`Decode.decodeTable`. Its exact constant-name list has eight entries and excludes
`COORD_CEILING`. It MUST NOT be formed at the closed program returned by `Decode.decodeClosure`;
that program has nine selected constants and includes `COORD_CEILING`.

The binding interface is frozen to these declaration shapes. `ParsesClosure` carries only the entry
decode and entry-source checks. `ClosedBinding` composes that entry binding with the checked
table-to-cell-to-std import path and the imported coordinate:

```lean
inductive ParsesClosure (bundle : SourceBundle) (entryProgram : Program) : Prop where
  | intro (entryBytes : SourceBytes)
      (entryLookup : bundle.fileBytes? bundle.entry = some entryBytes)
      (entryDecoded : CertifiedDecoded)
      (entryDecodes : Decode.decodeTable (encodeBytes entryBytes) = .ok entryDecoded)
      (programEq : entryDecoded.val.program = entryProgram)
      (selectedSource : SelectedSourceExact entryBytes)
      (selectedProgram : SelectedProgramExact entryProgram)
      (importsResolved : ∀ path, path ∈ importPaths entryBytes →
        (bundle.fileBytes? path).isSome = true) : ParsesClosure bundle entryProgram

inductive ClosedBinding (bundle : SourceBundle) (value : CertifiedDecoded) : Prop where
  | intro (entryBytes cellPath cellBytes stdImportPath stdBytes : SourceBytes)
      (entryDecoded : CertifiedDecoded)
      (stdTokens : List Token) (stdConsumed : Nat) (coord : ConstantDecl)
      (entryLookup : bundle.fileBytes? bundle.entry = some entryBytes)
      (entryBinding : ParsesClosure bundle entryDecoded.val.program)
      (tableImports : importPaths entryBytes = [cellPath])
      (cellLookup : bundle.fileBytes? cellPath = some cellBytes)
      (cellImports : importPaths cellBytes = [stdImportPath])
      (stdLookup : bundle.fileBytes? stdImportPath = some stdBytes)
      (coordDerivation : ImportedCoordDerivation stdBytes stdTokens stdConsumed coord)
      (coordValue : coord.value = .literal 8796093022208)
      (composed : value = Decode.appendImportedCoord entryDecoded coord) :
      ClosedBinding bundle value

theorem source_binding_of_closed_binding {value : Decode.CertifiedDecoded}
    (binding : ClosedBinding entryBundle value)
    (selectedSyntax : ExpectedSelectedSyntax value.val.program = true) :
    Decode.decodeClosure (encodeBytes tableSource) (encodeBytes stdSource) = .ok value ∧
    value.val.certificate.Valid ∧
    ExpectedSelectedSyntax value.val.program = true ∧
    ClosedBinding entryBundle value

theorem source_binding :
  ∃ value : Decode.CertifiedDecoded,
    Decode.decodeClosure (encodeBytes tableSource) (encodeBytes stdSource) = .ok value ∧
    value.val.certificate.Valid ∧
    ExpectedSelectedSyntax value.val.program = true ∧
    ClosedBinding entryBundle value
```

The unconditional `source_binding` declaration is the only source-binding headline. The corrected
conditional composition lemma remains under `source_binding_of_closed_binding` with the exact shape
above; it is not an acceptance target. The former theorem requiring both
`ExpectedSelectedSyntax value.val.program = true` and
`ParsesClosure entryBundle value.val.program` MUST be removed, not renamed, because those premises
force the same constant-name list to contain respectively nine and eight entries.

The concrete theorem MUST be a named proof term whose transitive `Lean.collectAxioms` set is empty.
Whole-source `#guard` checks remain compiled regression checks; they are neither a `tested` grade nor
source-binding acceptance. The re-rooted memo-store contract is the active refinement; generated
decoder traces are not an acceptance condition. Implementation acceptance is the exact-declaration
probe on the unconditional existential, not a shape check of a working headline.

### Bounded source-witness composition

The existing `tools/slang/export.py` MUST emit named bounded derivations of the existing structural
source relations, with shared data/index equations and kernel-checked composition. This is the
selected generator contract, not current whole-source acceptance. The composition lemmas are in
commit `37b95c385ae0e58d020346a0e697053199af5940`; names below are relative to
`Slang.Source.Tokenizer` or `Slang.Source.Parsing`.

| Source module | Composition interface |
|---|---|
| `proof/corpus/Slang/Source/Tokenizer/Utf8.lean` | `Utf8Scan.continue`: a bounded prefix implication applied to its tail witness. |
| `proof/corpus/Slang/Source/Tokenizer/Scan.lean` | `CharsScan.continue`; `ReverseTokens.of_eq` takes the checked accumulator-reversal equality. |
| `proof/corpus/Slang/Source/Tokenizer/Helpers.lean` | `DigitsFold.append`, `Utf8LengthFold.append`: the first output accumulator is the second input. |
| `proof/corpus/Slang/Source/Parsing/Selection.lean` | `DropNewlinesScan.append`, `TakeLineScan.append`, `ScanSelected.continue`. |
| `proof/corpus/Slang/Source/Parsing/Validation.lean` | `DelimitersBalancedDerivation.continue`, `SpansValidDerivation.append`, `DistinctDerivation.append`. |
| `proof/corpus/Slang/Source/Parsing/Validation.lean` | `ExpressionsCallsResolveDerivation.append`, `StatementsCallsResolveDerivation.append`, `FunctionCallsResolveDerivation.append`, `CallsResolveDerivation.of_parts`. |
| `proof/corpus/Slang/Source/Parsing/Validation.lean` | `ExpressionsNamesResolveDerivation.append`, `StatementsNamesResolveDerivation.continue`, `FunctionNamesResolveDerivation.append`, `NamesResolveDerivation.append`. |
| `proof/corpus/Slang/Source/Parsing/Tokens.lean` | `DecodeTokensDerivation.assemble`: exact consumption, delimiters, selection, counts, distinctness, spans, calls, names and result equality. |

Composition MUST preserve these premises; chunking does not weaken them:

- `TakeWhileScan` and `SkipCommentScan` in the tokenizer helpers are maximal-prefix relations
  with no cross-boundary append. Keep identifiers and comments within one lexer chunk.
- Newline completeness means an empty first remainder, not a newline-terminated first chunk:
  `DropNewlinesScan.append` requires `DropNewlinesScan a []` (all first-chunk tokens are newlines);
  `TakeLineScan.append` requires `TakeLineScan a lineA []` (no terminating newline in the first
  chunk). A retained newline belongs to the second scan's remainder.
- `SpansValidDerivation.append` requires both directed cross guards, `crossAB` and `crossBA`,
  in addition to validity of each chunk at the same byte size.
- `DistinctDerivation.append` requires explicit `a.Nodup`, `b.Nodup` and
  `∀ name ∈ a, name ∉ b`. The generator MUST produce `List.Nodup` proofs directly; the accepted
  layer supplies `DistinctDerivation.of_nodup`, not a `DistinctDerivation → List.Nodup` converse.
  The reported `eraseDups` converse uses rejected axioms and is not an admissible dependency;
  this is a layer restriction, not a mathematical impossibility theorem.
- Call- and name-resolution appends require no cross condition: both chunks use the same global
  `available`/`constants`. `CallsResolveDerivation.of_parts` uses the full combined function-name
  list. Statement-name continuation carries the changed environment and the `fuel0`/`fuel1`
  recursion indices explicitly; these are not a combined source-resource budget.

The retired `Slang/Derivation/` named proof DAG / segment-lemma evaluator and retired
`Slang/Generated/Trace/` outputs remain binned. Shared data/index equations and bounded constructor
proof terms over the existing structural relations are a different mechanism, not a replacement
evaluator or revived trace acceptance. The generator is untrusted; Lean checks each witness and
its empty axiom cone under default recursion depth and bounded heartbeat/time/memory limits.
Neither compiled `#guard` checks nor a digest replaces those witnesses, and byte binding alone
establishes no memo-store refinement, codegen or ABI claim.

Retirement precedes generation. `tools/rooting/retire_check.py (unpublished)` rule R6, configured in
`tools/rooting/retire_cases.json (unpublished)`, rejects producer references to the retired
`Slang/Generated/Trace/` and retired `Slang/Derivation/` routes and emission of `maxHeartbeats 0`
or `maxRecDepth 1000000`. It is a textual gate; the default-depth requirement is stronger than those
two option checks. The current exporter emits `maxRecDepth 10000`; replacing that output with
bounded shared data definitions remains required. The
[barrier-probe measurements](slang-source-refinement-plan.md#barrier-probe-measurements) motivate
chunking but do not prove whole-source feasibility.

## Source arena and parsed records

The binding input is an immutable source arena that owns the exact bytes and the bounded parsed
records derived from them. A `SourceRef` is a file identity with a checked byte interval and a checked token interval, together with in-bounds and correspondence proofs that the interval lies in
the owned bytes and that the record's tokens are exactly the tokens of that byte interval. No copied text, filename-only identity, unchecked offset or prefix parse. A reference that omits the bytes it names, that names a file without an interval, or that parses a prefix of a selected body is rejected. Full-body consumption and unsupported-syntax rejection are properties of this arena.

## State and result arenas

Execution state and results live in arenas of bounded handles with checked membership. Every handle
is a checked index into its arena; a handle outside the current capacity is rejected. Records in
these arenas are immutable once allocated. Allocation returns a fresh handle or an explicit arena-capacity refusal; it never reuses a live handle and never returns a partial record. The
arena-capacity refusal is a non-storage outcome distinct from `ShardFull` and from a memory or stack
fault.

## Memo key and memo value

A memo key is the complete computation identity: the `SourceRef`, the operation, the
context/environment, the precedence, every recursion, token, byte, stack and result limit, every
input state handle, `Sigma`, and the selected addressing face. A digest or coordinate alone is not key equality. Two computations that differ in any component of that identity are different keys.

A memo value is a bounded result handle together with the resulting state handles, the consumed
source interval, and checked evidence that the result denotes the specified transition. A value without evidence is not a memo hit. Storing or returning a result handle whose transition is not checked is rejected.

## Addressing faces and Sigma

Every executing signature carries `Sigma`. The three faces are distinguished:

- elliptic exact reversible packing on an explicit admitted domain;
- parabolic carried complete key, where a coordinate never replaces key comparison;
- hyperbolic scramble/discard only on an explicit observation domain.

One home shard; no fallback probing. A coordinate that is not exact on the admitted domain is
refused, not silently widened.

## Memo storage and fold

The memo store is fixed-shard storage: 256 shards, capacity `N` per shard, and full-key comparison.
There is no growth, fallback container, silent eviction or unrelated-key replacement. The canonical
type is `Apeiron.Table.HMap MemoKey MemoValue`; reads, updates and builds denote the canonical
`get`, `insertWith` and `build` definitions named under the canonical algebraic link.

Execution is an ordered fold of checked transitions. A successful execution denotes the same ordered
update stream under the canonical `build`; reordering, skipping or synthesizing a transition is forbidden. The fold records each checked transition in source order and exposes the resulting state
handles and consumed intervals.

## Pointwise invariant, success and refusal

`RepMemo(store, h)` is the pointwise invariant: every occupied handle resolves to its memo key and
memo value, the store's canonical meaning is `h`, and every read denotes the canonical `get`. The
obligations include `N = 0`: construction and total default reads are valid, every insertion refuses,
an empty fold succeeds, and any storage-requiring fold refuses without a partial table. `ShardFull`
and the non-storage refusals are distinct, and each carries joint witnesses at the claimed types.

## Canonical algebraic link

Canonical targets resolve in `Apeiron/Table/Basic.lean` as the axiom-free definitions
`Apeiron.Table.HMap`, `Apeiron.Table.get`, `Apeiron.Table.insertWith` and `Apeiron.Table.build`; the
memo store's reads, updates and builds denote those same canonical `get`, `insertWith` and `build`
definitions. The canonical theorems `Apeiron.Table.get_insertWith_self`,
`Apeiron.Table.get_insertWith_ne`, `Apeiron.Trichotomy.AnyRing.ringMul_comm`, `ringMul_assoc` and
`ringNormForm_mul` are attribution references only; the specializations this chain needs are
re-proved locally, because those theorems carry `propext`/`Quot.sound` and are inadmissible as
dependencies.

The constructive algebraic link is `Slang.Refinement.wordDenote`, `Slang.Refinement.cellDenote`,
`Slang.Refinement.cellDenote_mul` and `Slang.Refinement.cellNormForm_denote`, which match
`Apeiron.Trichotomy.AnyRing.ringMul` and `Apeiron.Trichotomy.AnyRing.ringNormForm`. The link is
constructive and locally supplied. `word_mul_denote` and any `ZMod` declaration MUST NOT enter the empty-cone chain; `ZMod`, `Finset`, `Quot`, `noncomputable` and `classical` are rejected there.

The Slang source-refinement chain requires an **empty** axiom cone: every registered declaration,
instance, witness and headline audits with `Lean.collectAxioms` empty
(`proof/corpus/Slang/Audit/Empty.lean`, `tools/slang/zero_axioms.py`). A `#guard` is a compiled
evaluator test, not a proof term and not acceptance: an obligation is `proved` only when a named
declaration audits empty.

## Word and expression semantics

Let `Word = Fin (2^64)`. `bits(w)` is its representative in `[0, 2^64)`. The signed view is
`signed(w) = bits(w)` below `2^63` and `bits(w) - 2^64` otherwise. Source integer literals in the
selected closure are embedded modulo `2^64`; `TABLE_EMPTY` is the all-ones word and has signed view
`-1`.

- `+`, `-`, and `*` compute in `Word`, reducing modulo `2^64` after the mathematical operation.
  Negating the minimum signed word therefore wraps to itself.
- `<`, `<=`, `>`, and `>=` compare `signed`; `is` and `is not` compare all 64 bits. Comparisons return
  exactly word `1` or word `0`. A guard takes the then branch exactly when its word is nonzero.
- `and`, `or`, and `xor` are bitwise operations on all 64 bits. They are strict, evaluate the left
  operand and then the right operand, and are not short-circuit Boolean connectives. Division,
  remainder, and shifts are outside this selected execution closure and MUST reject if made reachable.
- Ordinary binary expressions and call arguments evaluate left-to-right. Calls may expose memory
  effects in that order. Inline `a if c else b` evaluates `c` first and exactly one branch. Statement
  `if` has the same lazy branch rule.

In `tableSpread`, `pos` is computed in `Word`; then `pos * TABLE_PHI` is a **64-bit wrapping multiplication**,
and only then is the product bitwise-masked by `4611686018427387903` (`2^62 - 1`).
The multiplication MUST NOT be interpreted as unbounded integer multiplication followed by masking.

## Local state, calls, and control

A configuration contains the current function, a function-local environment, shared memory, and a
call stack. Parameters are initialized from left-to-right argument values. `name = expression`
evaluates the expression, then creates or replaces `name` in the current frame. Rebinding does not
mutate the caller's frame or a top-level constant. Constants are immutable.

Statements execute in source order. A call creates a fresh frame and shares memory. A normal callee
return supplies its raw word to the caller; its memory changes remain. `return e` evaluates `e` and
immediately exits the current function: no later statement or untaken branch executes. A bare call
statement evaluates the call and discards only its raw return. Recursion is ordinary call recursion;
its meaning is the finite derivation generated by these rules, not a bounded approximation.

## Memory and extents

Memory is a partial map from nonnegative representable coordinates to initialized `Word` values,
paired with an owned valid-coordinate set. A load succeeds only at an owned initialized coordinate;
an uninitialized or unowned load is a memory fault. A store succeeds at any owned coordinate,
replaces exactly that coordinate, initializes it, and leaves every other coordinate unchanged; an
owned store MAY initialize a previously uninitialized coordinate and is not a fault. An unowned load
or store produces a memory fault outcome with the pre-fault state; it is not a word return. This
decoded-fragment memory semantics supports source binding; the active refinement object is the memo
store, not this machine.

## Checked execution outcomes and admission

Checked execution yields one of: `returned(raw, memory')`, `memoryFault(memory')`, or
`stackExhausted(memory')`. The last two are machine/source-resource outcomes outside the table's raw
return carrier. Invalid backing memory and machine-stack exhaustion have **NO checked table refusal encoding**.
In particular they MUST NOT be modeled as a returned `TABLE_EMPTY`/`-1`. An interpreter,
checker, or proof search timeout is inconclusive and MUST establish no branch of this outcome type.

Unbounded source execution and a stack-limited checker are distinct artifacts. The source relation is
the finite-derivation relation: a `returned` outcome requires only a finite derivation and is not
restricted by any fixed call depth. A stack-limited checker's claim that it finds a `returned`
outcome requires an explicit premise that its stack capacity suffices for that derivation, and its
failure to find one is not by itself a refusal.

For each operation `op`, `Admit_op(input, memory)` is defined independently of the raw return. The
memo-store admission adds the arena-capacity refusal and the `ShardFull` refusal as separate explicit
outcomes. `Classify_op` first requires a checked `returned(raw, memory')` derivation, then reports
success iff `Admit_op` held, and otherwise reports only the operation's specified explicit
refusal/miss. It never tests `raw != TABLE_EMPTY`: a present read and successful MAP/ACC write may
return `-1`. Every explicit refusal leaves the refused call's memory unchanged. Earlier successful
calls remain visible. A `returned` execution outside the initialized-extent or other table-domain
hypotheses is not automatically an explicit refusal: classification requires the table-domain
hypotheses to hold and the executed branch to be the actual refusing branch.

## Required theorem shapes

The later checked package MUST provide these theorem shapes over decoded selected bodies and the
memo store, not copied algorithms:

| Theorem obligation | Required conclusion |
|---|---|
| constructor establishment | From joint valid-capacity/backing hypotheses, checked construction establishes `RepMemo(store, const default)`, and `N = 0` construction and total default reads are valid. |
| read correctness | From `RepMemo`, checked lookup preserves state and agrees with the memo key's complete identity and the canonical `get h key`. |
| replacement write | On an admitted update, checked insertion preserves `RepMemo` for the extended key set and the canonical `insertWith`, with exact raw result and post-state. |
| exact admission | Checked success is equivalent to its stated admission predicate; `ShardFull`, `N = 0`, arena-capacity and non-storage refusals are distinct; faults are excluded, not recoded. |
| fold denotation | The ordered fold of checked transitions denotes the same ordered update stream under `build`; reordering, skipping or synthesizing is excluded. |
| termination and resources | The bounded transition measures terminate under their joint hypotheses, with exact transition-derived counts. |
| success witness | One concrete initialized store, present/empty layout, key/value, checked execution, and `RepMemo` jointly inhabit a successful read and write premise. |
| refusal witness | One concrete initialized full layout with an absent key jointly inhabits the write-refusal premises and refuses without a partial table. |

## Termination and source resources

Termination uses natural measures derived from decoded arguments: `cap - i` for `tableInitGo` and
`cap - step` for each probe function, under `0 <= i, step <= cap`. Each recursive edge increments its
cursor exactly once and the guard exits at the bound. The proof MUST connect these facts to the
decoded call edge and wrapping-word side conditions; source annotations alone are not premises. The
memo store adds a bounded transition measure over its checked fold.

Every checked execution carries a trace. Resource projections separately count arithmetic
operations (including the wrapping multiply before `tableSpread` masking), comparisons, bitwise
operations, loads, stores, calls, probe iterations, and maximum simultaneous source call-stack
entries. Initialization performs `cap` slot writes, hence `2*cap` raw stores. Each probe performs at
most `cap` iterations. The projection also retains exactly zero allocated nodes and exactly zero
external stream consumption: the selected execution allocates no arena or heap nodes and reads or
writes no stream other than its owned memory. Counts are not converted into one scalar budget and do
not imply a native machine-stack bound. Machine-stack exhaustion remains the separate outcome above.

## Retired route

The `Slang/Derivation/` `Layer`/`Derives` tower is retired. The `Slang/Generated/Trace/` generated
decoder traces are retired. The `DecodeTrace.Run` representation is retired, the named proof DAG is
retired, and the segment lemma is retired; none MUST be presented as the active architecture, and
their generation is not an acceptance condition. `RepSong`, `observeRead`, the raw `memory[]`
relation and `MERGE_MAP` are retired as the primary refinement and are replaced by `RepMemo`,
state-aware memo lookup, the source arena and the memo store. Historical references to a retired
route carry a retirement marker and are not acceptance.

## Not covered

- No correspondence, Song-to-Lean refinement, theorem, or source execution is established here.
- No ISA, emitter, compiler, assembler, linker, kernel ABI, native stack, rustc/LLVM, GPU, or SPIR-V
  correctness is covered.
- No Rust evaluator is introduced and no runtime linkage to Rust or Aeneas is permitted.
- No semantics is claimed for unreachable Slang constructs or nonselected table wrappers/tests.
- `tools/slang/spec_check.py` checks text, identities, references, contract statements and mutation
  rejection only; it is not the decoder, source machine, or refinement checker.

## Checker

`python3 tools/slang/spec_check.py --check` verifies this specification's obligations, sections,
source identities, canonical references, re-rooted contract statements, retired-route markers and
staged plan rows as text. It does not read the working `proof/corpus/Slang/SourceAudit.lean`
declaration; implementation acceptance is the `tools/slang/refine.py` exact-declaration probe on the
unconditional existential. The manifest gates the selected execution/import set and the listed
grammar, AST, bridge, and ISA evidence files; `song/src/front/expr.slang` is recorded here as parsing
provenance and is not yet digest-gated. `--controls` and `--self-test` mutate in-memory copies and
require each removed obligation, row, stage, gate, control, digest, symbol, theorem, contract
statement and retired-route reintroduction to be rejected. The working tree is not mutated.
