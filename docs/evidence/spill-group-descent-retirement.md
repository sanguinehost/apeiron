# Retirement receipt — the spill group-descent scanner (`src/table/scan.rs`)

`src/table/scan.rs`, 303 lines, 11034 bytes, sha256
`ec633507a8ad456a7cb11553f559cf1abc08d3d4984a118e145eb5aa1ad91e97` — the digest already recorded in
`proof/README.md:457` and in the spike receipts under `artifacts/`, so the removed bytes are the
same ones those digests name.

## What was removed and why it was reachable only by itself

The module is the SwissTable group-descent scanner: an 8-lane SWAR tag compare, an SWAR
zero-lane compare, the gather that packs matched lanes into a byte mask, the word index, and the
scalar tail that scans a partial word.

| item | line | role |
|---|---|---|
| `LANES_PER_WORD` | 14 | lanes per `u64` group word |
| `LO` / `HI` / `GATHER` | 17 / 19 / 26 | SWAR constants |
| `eq_swar` / `zero_swar` | 35 / 44 | the two SWAR predicates |
| `gather` | 55 | matched-lane mask packing |
| `word_at` | 64 | group word load |
| `match_tag` | 77 | fingerprint equality filter |
| `match_empty` | 103 | vacancy scan |
| `least_set_lane` | 138 | least set-bit index |

Its five production callers were three `scan::match_tag` and two `scan::match_empty`, all inside
`src/table/probe.rs` (`docs/epics/read-path-specialization.md:770`). The flat open-addressed port
replaced that call shape: the tag travels inside `SpillSlot` in `src/table/state.rs`, the probe is
one bounded linear cycle, and a `tag == 0` test *is* the vacancy test. `docs/epics/read-path-specialization.md:183`
states the preservation argument for removing vacancy work from reads, and `:255` lists
`scan::match_tag` among the helpers the Aeneas audit covered.

`cargo clippy --release --example spill_read_probe -- -D warnings` named all eleven items as
`never used` at the lines above, each in `src/table/scan.rs` and nowhere else. That is the
executing-path statement of unreachability; the consumer inventory below is the static one.

## Consumer inventory

| consumer | kind | action |
|---|---|---|
| `src/table.rs:42-43` | `#[path]` module declaration | removed with the file |
| `proof/regen.sh:104` | extraction `SOURCE_FILES` entry | removed with the file; the loop at `:145` exits 2 on a missing source, so leaving it would have broken `regen.sh` |
| `proof/resource/Capacity_gate.py:194`, `Schedule_gate.py:203`, `I64Codec_gate.py:264`, `CoordUnspread_gate.py:197` | receipt `source_paths` digest list | entries removed; the receipt comprehension filters on `p.is_file()`, so a stale entry would have been silently skipped rather than refused |
| `proof/resource/StoredRelocateAllSource.lean:2178-2220` | generated extraction source, quoting `src/table/scan.rs` lines 14, 17, 19, 26, 44, 55, 64 | **not** hand-edited: it is regeneration output and the regeneration owns the extracted symbols |
| `proof/README.md:457`, `docs/hash-comparison.md:161`, `docs/roadmap.md:249`, `docs/plan.md:730`, `docs/epics/read-path-specialization.md:730,771`, `docs/reviews/post-compaction-continuation-10.md:86` | prose and digest records | retained as the historical record of the pre-port scanner; not rewritten, because they describe the state at the revision that had it |
| `artifacts/*/receipt.json` | spike digests | retained; the digests are the record of what those spikes measured |

## Reachability

`rg -n "scan::" src/ tests/ examples/ benches/ fuzz/ derive/` returns nothing after the removal, and
`rg -n "table/scan\.rs" src/ proof/regen.sh` returns nothing. The gate that refuses an unregistered
removal is `python3 tools/rooting/retire_check.py --check`; the record is
`spill-group-descent-scanner-retirement` in `tools/rooting/removals.json`.

## Retained obligation

Nothing is deleted that a caller still names, and no replacement is claimed for the extracted
symbols: `proof/ApeironWord.lean` and the `proof/resource/*_gate.py` extraction units still carry the
pre-port scanner's extracted definitions, and they are discharged by the extraction re-link
(`APEIRON-E6-W3-T3` in `docs/plan.md`), not by this removal. `proof/regen.sh --check` is red for
that reason and for the unrelated pin-assert blocker, both recorded before this change.

## What this receipt does not establish

`rustc`/LLVM codegen; that the SWAR constants were correct (their own unit tests, four of them,
are removed with the module and asserted nothing about the table's observable behaviour); and the
generated extraction artefacts above, which are the regeneration's to own.
