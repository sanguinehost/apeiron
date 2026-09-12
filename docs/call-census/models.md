# Hashtrinity call census — `modelfold` (schema 2)

Static, syntax-level census of how `modelfold` consumes `hashtrinity`. This is
**schema 2 / tool 0.2.0**. Reports from 0.1 are incompatible and must not be
merged or compared (see `meta.json` `compatibility_note`). The earlier 0.1 TSVs
in this directory are marked OBSOLETE.

## 0. Provenance and reproduction

| item | value |
| --- | --- |
| snapshot | `/tmp/opencode/fold-call-census-20260911` |
| revision | `4cc14471ae24f3f204fb0b0e91736ef75c909358` (`git status --porcelain` = 0) |
| consumer root | `<SNAP>/modelfold` (all files) |
| target sources | `<SNAP>/hashtrinity`, `<SNAP>/hashtrinity/hashtrinity_derive` |
| binary | `/tmp/opencode/call-census-target/release/call-census` |
| `tool_binary_sha256` | `5600cfc46cacd0ed871b10960a97cc8bd5dfb36f13f0d30760f8f7ab320824a1` |
| `tool_source_sha256` | `7de0f968c93b7367c6fc8b98f36ebaa2f627f60d6b2b4870ad7b7a6c12413fa9` |
| raw output | `/tmp/opencode/census-v2/models/` (unknown/raw kept in `/tmp`) |
| final rerun / exit | `audit_reports.py --rerun`, all 34 runs exit `0`; published ids/spans/callees reconciled |
| current summary hash / provenance | [`data/census-audit.json`](data/census-audit.json), models run |

```sh
SNAP=/tmp/opencode/fold-call-census-20260911
/tmp/opencode/call-census-target/release/call-census \
  --root "$SNAP/modelfold" \
  --target-src "$SNAP/hashtrinity" \
  --target-src "$SNAP/hashtrinity/hashtrinity_derive" \
  --output /tmp/opencode/census-v2/models
```

Artifacts under `docs/call-census/data/`:

| file | rows | content |
| --- | ---: | --- |
| `models-calls.tsv` | 1009 | **full attributed call-site ledger** (verbatim `calls.tsv`: id, operation_kind, confidence, cfg, source_location, caller, callee, receiver) |
| `models-symbols.tsv` | 78 | per-callee attributed counts with operation_kind breakdown |
| `models-manual-unresolved.tsv` | 41 | manually adjudicated unresolved sites (separate; not added to totals) |
| `models-bypass.tsv` | 26 | production `std` container use in `src/` |
| `models-{alias-hidden,false-positives,missed-calls}.tsv` | — | OBSOLETE 0.1; annotated, do not use |

## 1. Core counts

409 files (241 examples, 149 src, 15 src/bin, 4 tests), 409 parsed, 0 parse errors.

**Recorded call expressions: 8962.**

| class | count |
| --- | ---: |
| `method_attributed` | 624 |
| `target_assoc_fn_call` | 239 |
| `target_direct_call` | 146 |
| **syntactically attributed (high)** | **1009** |
| `method_candidate` | 7912 |
| `method_trait_heuristic` | 23 |
| `method_ambiguous` | 16 |
| `method_dynamic_dispatch` | 2 |
| **recorded total** | **8962** |

Separately aggregated (not individual records): **39139** method calls on a
known non-target receiver or otherwise unclassified.

**`operation_kind` over all 8962 records:** `method` 8577, `associated_fn` 203,
`constructor` 36, `free_fn` 99, `unclassified` 47.

**`operation_kind` over the 1009 attributed records:** `method` 624,
`associated_fn` 203, `constructor` 36, `free_fn` 99, `unclassified` 47.
`constructor` here is inventory-derived (e.g. `par::SendPtr`, a tuple struct).
`codec::syntax::SyntaxTerm::Bu` (18) and the other `SyntaxTerm`/`Atom` variant
paths are **enum-variant constructor expressions labelled `unclassified` by the
tool**, not functions; 31 of the 116 codec expressions are such paths.

**`source_location` (all / attributed):** src 5504/735, examples 2812/271,
src_bin 637/3, tests 9/0. `reachability` is always `not_determined`.

**`cfg` on attributed records:** 74 carry a non-empty cfg — 49 a `cfg(test)`
predicate (47 single, 2 nested), 20 `cfg(feature = "gpu")`, 5
`cfg(feature = "codec")`; `is_test_context` is true for 49.

Other: **78 distinct target symbols** and **78 distinct callees** (40 non-empty
`callee_path`; method records carry the symbol), **441 distinct callers**;
200 target imports, 0 re-exports; 37 `Term` derives, 1 attribute macro, 23
`macro_rules!` defs, 11308 unexpanded macro invocations.

### 1.1 What "attributed" means

`attributed` is a **syntactic match**, not compiler resolution. It is **not a
mathematically assured lower bound**, not a count of executed calls, and not a
completeness proof. Macro bodies are not expanded; generic instantiations, trait
objects, `Deref` chains and untyped receivers are not attributed. The 0.1 word
"resolved" and any "floor" claim are withdrawn.

No global attribution rate is stated: the denominator is incomplete because free
calls inside macro bodies, dynamically dispatched calls, and calls whose receiver
is untyped are missing from both the attributed and the aggregated sets. Only
counts, never a percentage.

## 2. Per-callee attributed counts

Full dataset: `models-symbols.tsv` (all 78 symbols with operation_kind
breakdown). Top symbols:

| target_symbol | attributed | op kinds |
| --- | ---: | --- |
| `Table::new` | 123 | associated_fn |
| `Table::get` | 98 | method |
| `Table::iter` | 77 | method |
| `Table::put` | 68 | method |
| `Table::add` | 56 | method |
| `codec::syntax::field` | 41 | free_fn |
| `Table::count` | 40 | method |
| `par::SendPtr` | 36 | constructor |
| `Table::with_capacity` | 30 | associated_fn |
| `Table::len` | 26 | method |
| `Table::neighbors` | 24 | method |
| `Table::insert_with` | 21 | method |
| `CoordMap::iter` | 20 | method |
| `CoordMap::get` | 18 | method |
| `codec::syntax::SyntaxTerm::Bu` | 18 | unclassified (variant path) |
| `codec::syntax::parse_surface` | 17 | free_fn |
| `CoordMap::coordinate` | 16 | associated_fn |
| `CoordMap::insert_with` | 15 | method |
| `par::Pool::global` | 15 | unclassified |
| `Dense::zeros` | 14 | associated_fn |
| `Table::get_exact` | 12 | method |
| `CoordMap::accumulate` | 11 | method |

New in schema 2 vs 0.1 (same snapshot): local aliases (`Flat`/`Dist`/`Link`/
`Tab`) are now attributed (`Table::add` 56 vs 46, `Table::get` 98 vs 78,
`CoordMap::get` 18 vs 11, `Table::link` 4 new); trait-provided methods are
recorded as heuristic (`FoldExt`/`Monoid`/`AsTerm`); and the local
`phasefold::Table`/`FlatTree` false positives are gone from the candidate set.

## 3. Role aggregate

Symbol-derived, over the 1009 attributed expressions (not filename-inferred):

| role | attributed | composition |
| --- | ---: | --- |
| table / state / count / index | 824 | `Table` 665, `CoordMap` 120, `Dense` 38, `FreqMap` 1 |
| generic codec / serialization | 120 | `codec` 116 (85 free fns + 31 variant paths), `S3` 3, `tregex` 1 |
| parallel / runtime | 51 | `par::SendPtr` 36, `par::Pool::global` 15 |
| deferred crypto | 14 | `crypto::blake3` 7, `crypto::to_hex` 7 |
| **total** | **1009** | |

An essential-vs-incidental split (e.g. registry/CAS vs count store) would have to
be inferred from filenames; it is **not** given as a ratio here. The 0.1
369/349 heuristic split is withdrawn. What is defensible from the tool alone:
the table role dominates (824/1009), and within it the count/merge faces
(`add` 66, `count` 43, `accumulate` 14, `fuse` 15, `insert_with` 36, `link` 4,
`neighbors` 24, `push` 14) are a large share.

## 4. Why — representative families (source-read)

- **Count fold.** `src/countlm.rs:26,31` `Table<(u8,u64,u16), i64>`;
  `fold_range` `:200` `add(…,1)`; `merge` `:213` `add(k,c)`; `unfold` `:226`
  `add(k,-c)`. `src/arch.rs:9309/9426` `TokenLM::fold`/`add`;
  `:11481/11492` `TokenCountLM::merge` builds a `CoordMap` and iterates `Link`.
  `src/data.rs:2537–2591` `DnaFold` `coordinate_with_capacity`/`add`/`iter`;
  `:2800–2905` `AaFold`. `src/cast/direct.rs:279/436–447/543/548–549`
  `DirectFold` `coordinate_with_capacity`/`accumulate`/`fuse`/`iter`. The
  **program function** is the order-free count/merge state; the value type is an
  integer count.
- **Merge algebra.** `src/cast/phasefold/table.rs:10,98–102` wraps three
  `Dense` columns; `merge` `:153–155` `Dense::fuse`; `unlearn` `:171,173`
  `Dense::try_unfuse`, `:187` `Dense::unfuse`. `src/boost/quant.rs:183–217`
  builds `Table<u32, QLeafStat, Regen<u32>>` with `coordinate_with_capacity` +
  `accumulate`, reads with `get`; the integer `QLeafStat` type is a `Monoid`
  (`:129–139`) and the module's own tests assert order-exactness. **The census
  does not verify that algebra**; it records the call.
- **Universal-term codec.** `src/vision/wire.rs` contributes 74 codec
  expressions: `SyntaxTerm::B(Atom::Num/Str/Bool)` constructors at `:22–38`,
  `field`/`field_str` decoders at `:71–438`, `render_surface` `:454`,
  `parse_surface` `:459`. Same library at `src/tokenizer.rs:399`,
  `src/data.rs:1056`, `src/runtime.rs:1032`, `src/pipeline_numeric.rs:138`.
- **Registry / CAS (bookkeeping).** `src/runtime.rs:163–169` `ModelRegistry`
  (`put`/`get`/`remove`/`iter` only, `:183,196,210,380`);
  `src/checkpoint.rs:664–668` five `Table::new()` for `TestLake`;
  `src/vision/corpus.rs:554–596` `CorpusCatalog`.
- **Parallel.** `src/arch.rs:3387,3579,3724,3877,3934,4472`
  `par::SendPtr`/`Pool::global` around f32 GEMM/norm/attention/AdamW;
  `src/runtime.rs:3477–3478,4129–4131` in `SemStore`.
- **Deferred crypto.** `crypto::blake3`/`to_hex` (14) at
  `src/checkpoint.rs:672,687`, `src/vision/corpus.rs:408,448`,
  `src/vision/coco.rs:29,815`, `src/auto/artifact.rs:32–40`.

## 5. Counterfactual: could a `std` container replace it?

This is **not proved** either way. The defensible statement is narrow:

- The abstract algorithm at these sites is mostly conventional key→value plus
  integer counting. Replacing `Table`/`CoordMap` with `HashMap` would leave the
  *shape* of that algorithm recognizable, but exact preservation is **not
  established**, because the target types carry semantics a `HashMap` does not:
  coordinate-key collision handling (`get` vs `get_exact`,
  `src/text.rs:182–192`), iteration order (`Table::iter` is hash order, relied on
  only with explicit sorts), refusal/panic behavior (`Dense::try_unfuse`
  checks before writing, `phasefold/table.rs:32–36`), and serialization order
  (`.seed` bytes, `countlm.rs:269–284`). Any of these can change observable
  output.
- The integer count columns require overflow and subtraction-domain contracts;
  the `f64` columns are not order-free (`phasefold/table.rs:54–58` says so).
  The census does not verify these numeric contracts.
- One place where the *container API* carries a correctness behavior is
  `Dense::try_unfuse`'s check-before-write; a hand-rolled subtraction loop would
  not preserve that behavior without re-implementing the check. However, this
  per-column property does not automatically compose: the [retained probe](primitive-findings.md)
  reproduces a rejected `Table::unlearn` that has already changed `cnt`.

## 6. Bypasses, duplicates, nominated primitives

- **Quantization.** The numeric kernel bodies of `src/quant.rs` — `gptq_ternary`
  (`:93`), `lqquant` (`:490`), `q20` (`:1019`), `seal` fit (`:1504`), `veccode`
  (`:1935`) — are `Vec<f32>`/`Vec<f64>` and call no Hashtrinity face. The module
  **as a whole** does call Hashtrinity: `Seal::pack`/`unpack` at `:1802`/`:1822`
  call `codec::lz::compress`/`decompress`. So the correct claim is "the numeric
  kernels bypass the table faces", not "quant.rs calls nothing".
- **Linear algebra / training.** `src/foldtrain.rs` `solve_spd` (`:458`),
  `cholesky` (`:499`), `chol_solve` (`:544`), `FoldTrainer::fold/merge`
  (`:296–311`) are `Vec`/`f64`.
- **GPU / SIMT.** `src/simt/*`, `src/simt_nn/*` emit SPIR-V `Vec<u32>`
  (`simt/gemv.rs:27`, `simt_nn/norm.rs:18`); `src/gpu/vulkan.rs` dlopens Vulkan.
  No attributed table call in those files; the device side is raw buffers.
- **Production `std` containers in `src/`** (`models-bypass.tsv`): documented
  `LOOP-EXEMPT` — `arch.rs:9233/10142` `HashMap` (measured 1.24×/1.15×),
  `text.rs:46/241` `BinaryHeap` (ordered merge queue); undocumented —
  `auto/synth.rs`, `audio/whisper.rs`, `auto/dual.rs`, `auto/mod.rs`,
  `auto/trace.rs`, `bin/gift_eval.rs`.
- **Duplicates / primitives.** 38 local `SyntaxTerm` accessor defs across 12
  files while `codec::syntax::field` exists (41 attributed); ≥5 Lloyd/
  nearest-centroid codebooks (`boost/quant.rs:62`, `cast/lloyd.rs:131,231`,
  `quant.rs:676,2021`, `cast/quantize.rs:155`); Cholesky/`solve_spd` in
  `foldtrain.rs` plus `quant.rs:379` and six examples; ~30 hand-rolled
  `fold`/`merge`/`unfold` trios with no shared trait; two parallel mechanisms
  (`par` 51 attributed vs `rayon` in 26 files); two BPE tokenizers.
  Nominated: a `FoldState` trait, one deterministic Lloyd codebook, one
  `solve_spd`, a shared typed-accessor module, one parallel primitive, and the
  already-existing `FoldExt` collectors.

## 7. Programs (brief)

15 binaries (`src/bin/`): `eval` (YAML harness, codec), the seven `gift_eval*`
(forecasting, cast/ridge/Table), `harness` (countlm gates), `marketfold`,
four `nht_*`, `plantfold`. 241 examples: `agent_*` (8 files, 130 attributed)
and `rsna_*` (9 files, 54) dominate; the rest are mostly codec/`Table` counters
or have no attributed call. `modelfold` depends on `hashtrinity` with features
`codec, compress, syntax, derive, hex, base64, syncfold` (default off), plus
`rayon`, `libc`, `serde`/`serde_json`, `transcode`, `cast-cli-macro`;
`parquet`/`arrow` only under `mrnet`.

## 8. Manually adjudicated unresolved sites

`models-manual-unresolved.tsv` (41 rows, each with its tool `id`; **not added to
the 1009**):

- **16 `method_ambiguous`** — all `src/cast/phasefold/fold.rs` (`:687–926`,
  `:1321–1324`) on `self.l0..l3`: the local `cast::phasefold::table::Table`
  struct, **NON-TARGET** (correctly ambiguous; a non-target glob is in scope).
- **2 `method_dynamic_dispatch`** — `src/arch.rs:2286,2288` `write_all` on
  `&mut dyn Write`, **NON-TARGET**.
- **23 `method_trait_heuristic`** — imported target trait, untyped receiver:
  `FoldExt` 17 (`count_by`/`group_by`/`index_by`), `Monoid` 4 (`combine`),
  `AsTerm` 2 (`to_term`). **TARGET, but heuristic**, not typed; listed
  separately rather than folded into the attributed total.

Known false negatives of the syntax census (not candidates): calls generated by
macros (11308 unexpanded invocations, 37 `Term` derives), `Deref`/trait-object
dispatch, function-pointer references, and enum-variant/const *paths* that are
not call expressions (e.g. `Surface::Json`, `PHI`).

## 9. Independent verification

Recomputed from `records.json` (8962 records) and `summary.json`:

- categories and `operation_kind` histograms match `summary.json` exactly;
- `calls.tsv` has exactly `attributed_calls` = 1009 rows, categories
  `{method_attributed 624, target_assoc_fn_call 239, target_direct_call 146}`;
- `unattributed_methods.tsv` has 7953 rows = 7912 + 23 + 16 + 2;
- `by_source_location` all-record totals match `summary.json`
  (src 5504, examples 2812, src_bin 637, tests 9).

The 0.1 report's "lower bound", alias-hidden table, false-positive list and
missed-`FoldExt` list are superseded; their TSVs are annotated OBSOLETE.
