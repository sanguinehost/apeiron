# Joint Table Accumulator and Default Specification

This document specifies the accumulator (`MERGE_ACC`) face of the decoded selected
`tableInsertWith` body and the default adaptation a later correspondence must carry. It is an
obligation for a later checked theorem, not that theorem: it states no correspondence and proves
nothing. [`slang-source-semantics.md`](slang-source-semantics.md) is normative for the decoded
semantics, admission and resources; [`slang-source-entry.md`](slang-source-entry.md) SE3 records the
same entry at the specification level.

Authority: [`../direction.md`](../direction.md), [`joint-table.md`](joint-table.md),
[`slang-source-semantics.md`](slang-source-semantics.md). The canonical corpus is the pinned checkout
`../sanguine-apeiron/proof`; the declarations named below resolve in `Apeiron/Table/Basic.lean` at
that pin.

## Decoded selected body

The decoded `tableInsertWith(base, cap, key, value, merge)` selects one of two accumulator paths
after admission. The merge codes are `MERGE_SET = 1`, `MERGE_MAP = 2`, `MERGE_ACC = 3`.

| Path | Guard | Stored value | Write | Raw return |
|---|---|---|---|---|
| fresh key | located slot key `is TABLE_EMPTY` | `value`; the `is MERGE_SET` test selects `TABLE_PRESENT` only for `MERGE_SET` | `tableSlotSet(base, slot, key, value)` | `value` |
| present key | located slot key is not `TABLE_EMPTY` | `oldVal + value`, with `oldVal = tableSlotVal(base, slot)` | `tableSlotValSet(base, slot, oldVal + value)` | `oldVal + value` |

Both paths run only after the admission checks below. The `+` is `Slang.Word.add`, wrapping modulo
`2^64`. The fresh-key path never reads `oldVal` and never combines with a default: it stores the
incoming `value` unchanged. The present-key path combines with the located value in the order
`oldVal + value`.

## Absence and default adaptation

The canonical table is total:

- `Apeiron.Table.HMap K V` is `K → V`.
- `Apeiron.Table.get m k` is `m k`.
- `Apeiron.Table.insertWith f k v m` is `fun x => if x = k then f v (m x) else m x`, with the merge
  applied new-value-first.
- `Apeiron.Table.get_insertWith_self` gives
  `Apeiron.Table.get (Apeiron.Table.insertWith f k v m) k = f v (m k)`.

Canonical absence is the default stored in `m k`; there is no separate "not found". A fresh source
key has no slot value and no default, so the two sides differ:

- source fresh key with `MERGE_ACC`: stores and returns `value`;
- canonical `Apeiron.Table.insertWith (+)` at the same key: `Apeiron.Table.get (...) k = value + default`.

The source fresh-key ACC path stores `value`, while canonical `Apeiron.Table.insertWith (+)` combines
with `default`. The two agree exactly when the canonical value at the written key is `Word.ofNat 0`,
because `value + 0 = value`; they differ whenever `default` is nonzero. Equivalently, the source
fresh-key path is the first fold step of `Apeiron.Table.build f (fun _ => 0) kvs` at a key, not of
`Apeiron.Table.build f default kvs` at a nonzero default.

The present-key path has no such gap. Under `RepSong(memory, base, cap, default, O, h)` the located
`oldVal` is `h key`, and `oldVal + value = value + oldVal` by commutativity of `Slang.Word.add`, which
is the canonical new-value-first merge read back at `key`.

## Admission

The decoded write is admitted under the same predicate as the replacement face; the merge code
changes only the stored value, never admission. The write is admitted when:

- `tableCapacityOk(cap)` is nonzero, that is positive power-of-two capacity;
- the valid initialized extent holds (owned, initialized, `2*cap` cells);
- `key` is not `TABLE_EMPTY`;
- `MERGE_SET <= merge <= MERGE_ACC`;
- the probe reaches `key` or an empty slot within `cap` probes.

An absent key with no empty slot is the explicit full-table refusal; the probe-exhaustion branch
returns the raw word `0 - 1` before any slot write. Invalid capacity, the reserved key, an
out-of-range merge code and probe exhaustion are explicit refusals with unchanged memory. A memory or
machine-stack fault is a separate outcome and is never recoded as a table refusal.

## State and effects

The write mutates the located slot in place and has no rollback. The fresh-key path calls
`tableSlotSet`, which writes the key cell and the value cell; the present-key path calls
`tableSlotValSet`, which writes the value cell only. Earlier successful calls in a sequence remain
visible. The raw return is the stored value, not a status: a successful `MERGE_ACC` write may return
`-1` when `oldVal + value` wraps to the all-ones word, and that is success, not refusal.

## Resources

The write performs one `tableFindSlotGo` probe bounded by `cap` iterations, a bounded number of
arithmetic and bitwise operations (including the wrapping multiply in `tableSpread`), one call to
`tableSlotSet` or `tableSlotValSet`, and exactly one slot store: two raw stores on the fresh path,
one raw store on the present path. It allocates no nodes and consumes no external stream. The
projections are the `Slang.Machine.resources_*` fields; counts are not collapsed into one scalar
budget.

## Canonical declarations and the admissible proof route

The later accumulator theorem names:

- `Apeiron.Table.HMap`, `Apeiron.Table.get`, `Apeiron.Table.insertWith`,
  `Apeiron.Table.get_insertWith_self`, `Apeiron.Table.get_insertWith_ne`;
- the local computation lemmas `insertWith_apply_self` and `insertWith_apply_ne` in
  `proof/corpus/Slang/Table/TableWrite.lean`.

The admissible route is a theorem over the same joint hypotheses as the W29 replacement link
(`proof/corpus/Slang/Joint/SongTable.lean`, `songTable_replacement_link`), with the merge argument
fixed to `MERGE_ACC` and the canonical merge `fun new old => Slang.Word.add old new`:

- present branch: conclude the raw return and the read-back equal
  `Apeiron.Table.get (Apeiron.Table.insertWith (fun new old => Slang.Word.add old new) key value h) key`
  from `insertWith_apply_self` and commutativity of `Slang.Word.add`; no default hypothesis is
  needed;
- fresh branch: conclude the same read-back only under the explicit hypothesis
  `h key = Slang.Word.ofNat 0`, equivalently `default = Slang.Word.ofNat 0`. Without it the
  conclusion is false at every nonzero default.

The fresh-branch hypothesis is part of the contract. A theorem that drops it, or that silently
restricts the domain to zero defaults without naming the restriction, is not admissible. The
nonzero-default fresh case is recorded as a refuted profile row, not as a theorem.

## Profile check and rejecting control

`proof/corpus/Slang/Joint/AccumulatorProfile.lean` carries the profile as Lean data, the compiled
`#guard` check and the empty-axiom audit. The `accumulator-profile` stage of
`tools/slang/zero_axioms.json` registers exactly the audited declarations. The profile admits the
present-key match and the fresh-key store, and rejects the row that equates fresh-key ACC insertion
with canonical addition at an arbitrary nonzero default. The rejecting control changes that row's
verdict from rejected to admitted in a throwaway copy; `#guard profileCheck profile` then fails and
the copy does not build.

## Not covered

- No accumulator correspondence, refinement or theorem is established here; this is a specification
  and a profile check.
- No correspondence for `MERGE_SET` or `MERGE_MAP` is claimed; the `MERGE_MAP` replacement link is
  the separate W29 artifact.
- No source execution, machine-stack bound, rustc/LLVM, kernel ABI, GPU/SPIR-V or performance claim.
