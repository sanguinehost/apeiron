# Selected Slang Source Semantics

This specification fixes the source language fragment and execution closure used to interpret the
selected table bodies in `/home/socol/Workspace/song/src/table/table.slang`. It is an obligation for
later checked artifacts, not an executable semantics, proof, source-to-Lean refinement, or primitive
correspondence result. `tools/slang/spec_check.py` checks this document's consistency only.

Authority: [`docs/direction.md`](../direction.md#primitive-correspondence),
[`joint-primitives.md`](joint-primitives.md), [`joint-table.md`](joint-table.md), and
[`slang-source-entry.md`](slang-source-entry.md). The canonical corpus is the checkout
`../sanguine-apeiron/proof` at Git `8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f`, with Mathlib
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.

## Required obligations

Every later definition and theorem MUST retain all rows. <!-- OBLIGATIONS -->

| ID | Obligation | Required checked result |
|---|---|---|
| O01 | Word representation | One 64-bit word carrier, two's-complement signed view, and exact wrapping arithmetic. |
| O02 | Operators | Signed comparisons; exact bitwise `and`, `or`, `xor`; strict arithmetic evaluation; and rejection of reachable shifts, division, and remainder. |
| O03 | Binding and sequencing | Function-local binding/rebinding, statement order, calls, recursion, and early-return propagation. |
| O04 | Conditional evaluation | Lazy inline and statement conditionals; eager bitwise `and`/`or` with both operands evaluated. |
| O05 | Memory | Initialized raw `memory[...]`, valid extents, exact loads/stores, and unchanged surrounding memory. |
| O06 | Termination | Well-founded source derivations for initialization and both bounded probes. |
| O07 | Resources | Separate source counts for operations, probes, memory actions, calls, and maximum call-stack entries. |
| O08 | Exceptional boundaries | Invalid backing memory and machine-stack exhaustion are separate outcomes, never encoded as table `-1`. |
| O09 | Timeout | Interpreter or checker timeout is inconclusive and proves neither success nor refusal. |
| O10 | Source identity | Exact bytes, repository identity, imports, constants, declaration boundaries, and reachable call closure. |
| O11 | Parsing fidelity | Grammar-derived parsing consumes every selected body; unsupported reachable constructs reject the binding. |
| O12 | Non-substitutes | A digest alone, ISA records, sampled agreement, or a handwritten Lean algorithm with a matching filename proves no source semantics. |
| O13 | Representation relation | `RepSong(memory, base, cap, default, O, h)` retains occupied keys `O` as well as extensional map `h`. |
| O14 | State-aware read | `observeRead` distinguishes absence from a present value equal to `-1` or `default`. |
| O15 | Classification | Success/refusal follows checked execution plus the operation admission predicate, never `return != -1`. |
| O16 | State and result | Every theorem preserves both the raw return word and the resulting memory state. |
| O17 | Table refinement shape | Constructor establishment, read correctness, replacement-write correctness, exact admission, and termination. |
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

`song/` in this table denotes the sibling checkout `/home/socol/Workspace/song/`; it is not an
Apeiron subdirectory. The selected file imports `src/kernel/mem/cell.slang`, which imports
`src/kernel/std.slang`; `COORD_CEILING = 8796093022208` resolves from that closure. The selected table
bodies access raw `memory[...]` in `tableSlotKey`, `tableSlotVal`, `tableSlotSet`, and
`tableSlotValSet`; they do **not** call `cellGet` or `cellSet`.

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
or store produces a memory fault outcome with the pre-fault state; it is not a word return.

For a table call, `ValidExtent(memory, base, cap)` requires signed `base >= 0`, signed `cap > 0`, no
word or coordinate overflow in `base + TABLE_STRIDE * cap`, the explicit premise
`base + 2*cap <= COORD_CEILING`, and ownership of every coordinate in the
half-open interval `[base, base + 2*cap)`. Table slots are exactly key at `base + 2*slot` and value at
`base + 2*slot + 1`. Valid table states initialize both cells of every slot. The caller's memory below
`base` and at or above `base + 2*cap` is frame state and MUST remain bit-for-bit unchanged by every
selected table operation. `tableCapacityOk` checks only positive power-of-two capacity; it does not
establish backing validity.

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

For each operation `op`, `Admit_op(input, memory)` is defined independently of the raw return:

- init admission: positive power-of-two `cap` and `ValidExtent`; invalid capacity is the explicit
  source refusal and performs no write; valid capacity with invalid backing faults separately;
- read admission: valid capacity, valid initialized extent, nonreserved key, and an occupied matching
  key reached before the first empty slot; a checked miss is an explicit read refusal/miss;
- write admission: valid initialized extent, positive power-of-two capacity, key not `TABLE_EMPTY`,
  `MERGE_SET <= merge <= MERGE_ACC`, and a probe reaches that key or an empty slot in at most `cap`
  probes. An absent key with no empty slot is the explicit full-table refusal.

`Classify_op` first requires a checked `returned(raw, memory')` derivation, then reports success iff
`Admit_op` held, and otherwise reports only the operation's specified explicit refusal/miss. It never
tests `raw != TABLE_EMPTY`: a present read and successful MAP/ACC write may return `-1`. Every explicit
init/read/write refusal leaves the refused call's memory unchanged. Earlier successful calls remain
visible. A `returned` execution outside the initialized-extent or other table-domain hypotheses is
not automatically an explicit refusal: classification requires the table-domain hypotheses to hold
and the executed branch to be the actual refusing branch.

## Representation and observation

`RepSong(memory, base, cap, default, O, h)` relates a valid initialized table extent to
`h : Word -> Word` and `O : Finset Word`. It requires:

1. positive power-of-two `cap`, valid owned extent, and unchanged outside-frame ownership;
2. every slot key is `TABLE_EMPTY` or belongs to `O`; occupied keys are unique and never
   `TABLE_EMPTY`;
3. every `k in O` occurs in its `tableSpread(k) and (cap - 1)` linear-probe sequence before the first
   empty slot, and no key outside `O` occurs;
4. an occupied slot for `k` stores `h(k)`; for `k` outside `O`, `h(k) = default`.

The occupied-key set is essential: an extensional map alone cannot distinguish an unwritten key from
a written key whose value equals `default`.

`observeRead(memory, base, cap, default, O, h, key, raw)` is state-aware. Under `RepSong`, it is
`present(h(key), raw)` when `key in O`, requiring `raw = h(key)`; otherwise it is
`absent(default, raw)`, requiring `raw = TABLE_EMPTY`. The constructor/read/write theorems preserve
both the raw return and resulting state, even where the canonical observation ignores the raw word.

Canonical targets resolve in `Apeiron/Table/Basic.lean` as `Apeiron.Table.HMap`,
`Apeiron.Table.get`, `Apeiron.Table.insertWith`, `Apeiron.Table.build`,
`Apeiron.Table.get_insertWith_self`, and `Apeiron.Table.get_insertWith_ne`.

## Required theorem shapes

The later checked package MUST provide these theorem shapes over decoded selected bodies, not copied
algorithms:

| Theorem obligation | Required conclusion |
|---|---|
| constructor establishment | From joint valid-capacity/backing hypotheses, checked `tableInit` returns raw `0`, establishes `RepSong(memory', base, cap, default, empty, const default)`, and performs exactly `2*cap` stores, setting keys to `TABLE_EMPTY` and values to `0`, preserving other coordinates and ownership; already-correct cells need not change value. |
| read correctness | From `RepSong`, checked `tableGet` preserves memory and `observeRead` agrees with occupied membership and `Apeiron.Table.get h key`. |
| replacement write | On admitted `MERGE_MAP`, checked `tableInsertWith` preserves `RepSong` for `O insert key` and `Apeiron.Table.insertWith (fun new old => new) key value h`, with exact raw return and post-memory. |
| exact admission | Checked init/read/write success is equivalent to its stated admission predicate; each explicit refusing branch returns its raw word and unchanged memory; memory/stack faults are excluded, not recoded. |
| termination and resources | `tableInitGo`, `tableFindSlotGo`, and `tableLookupSlotGo` terminate under their joint hypotheses, with exact trace-derived bounds and maximum source call depth. |
| success witness | One concrete initialized extent, present/empty layout, key/value/merge, checked execution, and `RepSong` jointly inhabit a successful read and write premise. |
| refusal witness | One concrete initialized full layout with an absent nonreserved key jointly inhabits the write-refusal premises and returns raw `-1` with unchanged state. |

Replacement is the first canonical link. Accumulator semantics remain a later, separate specification:
the source fresh-key ACC path stores `value`, while canonical `insertWith (+)` combines with `default`.
No theorem may hide that mismatch by reducing the domain silently.

## Termination and source resources

Termination uses natural measures derived from decoded arguments: `cap - i` for `tableInitGo` and
`cap - step` for each probe function, under `0 <= i, step <= cap`. Each recursive edge increments its
cursor exactly once and the guard exits at the bound. The proof MUST connect these facts to the
decoded call edge and wrapping-word side conditions; source annotations alone are not premises.

Every checked execution carries a trace. Resource projections separately count arithmetic
operations (including the wrapping multiply before `tableSpread` masking), comparisons, bitwise
operations, loads, stores, calls, probe iterations, and maximum simultaneous source call-stack
entries. Initialization performs `cap` slot writes, hence `2*cap` raw stores. Each probe performs at
most `cap` iterations. The projection also retains exactly zero allocated nodes and exactly zero
external stream consumption: the selected execution allocates no arena or heap nodes and reads or
writes no stream other than its owned memory. Counts are not converted into one scalar budget and do
not imply a native machine-stack bound. Machine-stack exhaustion remains the separate outcome above.

## Not covered

- No correspondence, Song-to-Lean refinement, theorem, or source execution is established here.
- No ISA, emitter, compiler, assembler, linker, kernel ABI, native stack, rustc/LLVM, GPU, or SPIR-V
  correctness is covered.
- No Rust evaluator is introduced and no runtime linkage to Rust or Aeneas is permitted.
- No semantics is claimed for unreachable Slang constructs or nonselected table wrappers/tests.
- `tools/slang/spec_check.py` checks text, identities, references, and mutation rejection only; it is
  not the decoder, source machine, or refinement checker.

## Checker

`python3 tools/slang/spec_check.py --check` verifies every obligation and section, the
harness-manifest source hashes, imports, selected constants/functions, canonical declarations, and
all staged plan rows. The manifest gates the selected execution/import set and the listed grammar,
AST, bridge, and ISA evidence files; `song/src/front/expr.slang` is recorded here as parsing
provenance and is not yet digest-gated. `--controls`
and `--self-test` mutate in-memory copies and require each removed obligation, row, stage, gate,
control, digest, symbol, theorem, and boundary statement to be rejected. The working tree is not
mutated.
