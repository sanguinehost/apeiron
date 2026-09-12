# Call-level census — first-party Fold consumers (excluding `bedrock`, `modelfold`)

Scope tag: `other-fold`. Schema **2** / tool **0.2.0**. This is the half of the Fold
consumer census not owned by the `bedrock` / `modelfold` agents.

Every count in this report and in `data/other-*` is from schema 2. Reports produced by
call-census `< 0.2.0` used different field names and different confidence semantics
(`resolved` rather than `attributed`) and are **not** comparable; no schema-1 count is
carried into any dataset here. The tool's own `compatibility_note` says the same.

**Terminology.** A call is **syntactically attributed** when the scanner's syntax match
identifies it as a target call. That is a count of source syntax: it is **not exhaustive**,
not a count of executed calls, not a compiler-resolved lower bound, and not a completeness
proof. Macro bodies are not expanded; generic instantiations, trait objects, `Deref` chains,
wrapped receivers and untyped receivers are not attributed. "Zero attributed sites" below
always means *zero attributed sites plus a manual source search*, never "unreachable".

## 0. Provenance, scope, method

| | |
|---|---|
| Snapshot | `/tmp/opencode/fold-call-census-20260911` |
| Revision | `4cc14471ae24f3f204fb0b0e91736ef75c909358` (git HEAD; working tree clean per `meta.revision.git_status_porcelain_lines = 0`) |
| Tool binary | `/tmp/opencode/call-census-target/release/call-census` |
| `tool_binary_sha256` | `5600cfc46cacd0ed871b10960a97cc8bd5dfb36f13f0d30760f8f7ab320824a1` |
| `tool_source_sha256` | `7de0f968c93b7367c6fc8b98f36ebaa2f627f60d6b2b4870ad7b7a6c12413fa9` |
| Target srcs | `SNAP/hashtrinity`, `SNAP/hashtrinity/hashtrinity_derive` |
| Target crates | default set: `hashtrinity`, `hashtrinity_derive`, `hashtrinity_crypto`, `hashtrinity_codegen` |
| Output root | `/tmp/opencode/census-v2/other/<crate>/` |
| Ledger | `docs/call-census/data/other-calls.tsv` (4,059 rows) |
| Sums | `docs/call-census/data/other-sums.json` |

**Run (one `--root` per canonical top-level subtree):**

```sh
BIN=/tmp/opencode/call-census-target/release/call-census
SNAP=/tmp/opencode/fold-call-census-20260911
mkdir -p /tmp/opencode/census-v2/other
for c in transcode netfold webfold opsfold appfold investfold sangsec kleis \
         memoryfold personafold spbench; do
  mkdir -p /tmp/opencode/census-v2/other/$c
  "$BIN" --root "$SNAP/$c" \
    --target-src "$SNAP/hashtrinity" \
    --target-src "$SNAP/hashtrinity/hashtrinity_derive" \
    --output /tmp/opencode/census-v2/other/$c
done
```

All 11 runs exited 0.

**Exclusions and absences.**
- Excluded: `bedrock/`, `modelfold/` (other agents).
- `hashtrinity/` is target inventory only; its own implementation, tests, examples, benches
  and `hashtrinity/syncfold` are not counted as consumers.
- `winfold/` and `forgefold/` are absent; `forgefold` exists only as the appfold binary
  `appfold/src/bin/forgefold.rs`.
- `benchmarks/`, `models/`, `scripts/` are outside the enumerated scope and contain no
  `hashtrinity` reference.
- `kleis/provider/` (13 vendored `rauthy-*` packages) is censused and classified separately
  (§5); it is excluded from the workspace (`exclude = ["kleis/provider"]`).

**Parse health.** 586 `.rs` files, 586 parsed, **0 parse errors**.

## 1. Quantitative summary (schema 2 / tool 0.2.0)

### 1.1 Per-crate totals

| crate | files | records | syntactically attributed | unattributed candidates | heuristics | ambiguous | dynamic | other method calls | target imports | derives | attr macros |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transcode | 53 | 401 | 24 | 377 | 0 | 0 | 0 | 5,092 | 6 | 0 | 0 |
| netfold | 69 | 502 | 78 | 423 | 1 | 0 | 0 | 2,390 | 25 | 45 | 0 |
| webfold | 35 | 1,105 | 131 | 974 | 0 | 0 | 0 | 5,444 | 6 | 0 | 0 |
| opsfold | 63 | 5,787 | 706 | 5,057 | 16 | 0 | 8 | 14,619 | 184 | 195 | 0 |
| appfold | 31 | 171 | 75 | 95 | 1 | 0 | 0 | 635 | 7 | 0 | 0 |
| investfold | 17 | 422 | 61 | 361 | 0 | 0 | 0 | 1,930 | 9 | 0 | 0 |
| sangsec | 13 | 852 | 50 | 802 | 0 | 0 | 0 | 1,934 | 27 | 0 | 0 |
| kleis | 291 | 5,802 | 2,886 | 2,830 | 84 | 2 | 0 | 12,846 | 204 | 51 | 216 |
| memoryfold | 2 | 110 | 12 | 98 | 0 | 0 | 0 | 177 | 1 | 0 | 0 |
| personafold | 2 | 23 | 6 | 17 | 0 | 0 | 0 | 55 | 3 | 0 | 0 |
| spbench | 10 | 132 | 30 | 102 | 0 | 0 | 0 | 326 | 14 | 1 | 0 |
| **TOTAL** | **586** | **15,307** | **4,059** | **11,136** | **102** | **2** | **8** | **45,448** | **486** | **292** | **216** |

`heuristics` are labelled attributions where a method name is provided by an imported target
trait but the receiver type is unknown (e.g. `FoldExt::index_by`). They are counted
separately and are still candidates; they are **not** in the 4,059.

### 1.2 Source location (all records)

| crate | src | src_bin | tests | examples | benches | other |
|---|---:|---:|---:|---:|---:|---:|
| transcode | 401 | 0 | 0 | 0 | 0 | 0 |
| netfold | 502 | 0 | 0 | 0 | 0 | 0 |
| webfold | 654 | 451 | 0 | 0 | 0 | 0 |
| opsfold | 5,714 | 33 | 36 | 4 | 0 | 0 |
| appfold | 131 | 9 | 13 | 18 | 0 | 0 |
| investfold | 422 | 0 | 0 | 0 | 0 | 0 |
| sangsec | 343 | 286 | 0 | 223 | 0 | 0 |
| kleis | 5,706 | 0 | 96 | 0 | 0 | 0 |
| memoryfold | 110 | 0 | 0 | 0 | 0 | 0 |
| personafold | 23 | 0 | 0 | 0 | 0 | 0 |
| spbench | 132 | 0 | 0 | 0 | 0 | 0 |

### 1.3 `operation_kind`: constructors separated from functions and methods

The frozen tool exposes `operation_kind ∈ {free_fn, associated_fn, constructor, method,
unclassified}`. Its constructor classifier returns **0 for this whole corpus**, which is a
defect, not evidence of absence: enum-variant construction is plainly present
(`SyntaxTerm::Bu`, `Atom::Num`, …). The tool records those with `target_type = null` and
`operation_kind = unclassified`, so the 0 must not be read as "no constructors". Per the
README, constructors are supposed to come from the inventory (enum variants / tuple structs).

Two independent views are therefore given:

**Tool's own `by_operation_kind` (all records, not only attributed):**

| crate | free_fn | associated_fn | constructor | method | unclassified |
|---|---:|---:|---:|---:|---:|
| transcode | 1 | 4 | 0 | 396 | 0 |
| netfold | 9 | 34 | 0 | 453 | 6 |
| webfold | 2 | 31 | 0 | 1,060 | 12 |
| opsfold | 269 | 98 | 0 | 5,266 | 154 |
| appfold | 0 | 23 | 0 | 148 | 0 |
| investfold | 14 | 10 | 0 | 397 | 1 |
| sangsec | 13 | 2 | 0 | 806 | 31 |
| kleis | 1,479 | 17 | 0 | 3,270 | 1,036 |
| memoryfold | 0 | 4 | 0 | 106 | 0 |
| personafold | 0 | 1 | 0 | 22 | 0 |
| spbench | 16 | 0 | 0 | 104 | 12 |
| **TOTAL** | **1,803** | **224** | **0** | **12,028** | **1,252** |

(These sum to the 15,307 records; they are not the attributed-only view. The tool's `method`
count includes unattributed candidate method calls, which is why it is far larger than the
780 attributed methods in the next table.)

**Inventory-join reclassification of the 4,059 attributed calls** (join each
`target_symbol`'s last two segments against `inventory.json` `variants` / `tuple_structs` /
`fns` / `types`; method kept from the tool). This is the separation the tool fails to make:

| crate | constructor (variant) | constructor (tuple) | free_fn | associated_fn | method |
|---|---:|---:|---:|---:|---:|
| transcode | 0 | 0 | 1 | 4 | 19 |
| netfold | 1 | 0 | 9 | 39 | 29 |
| webfold | 12 | 0 | 2 | 31 | 86 |
| opsfold | 154 | 0 | 269 | 98 | 185 |
| appfold | 0 | 0 | 0 | 23 | 52 |
| investfold | 1 | 0 | 14 | 10 | 36 |
| sangsec | 31 | 0 | 13 | 2 | 4 |
| kleis | 234 | 0 | 1,479 | 819 | 354 |
| memoryfold | 0 | 0 | 0 | 4 | 8 |
| personafold | 0 | 0 | 0 | 1 | 5 |
| spbench | 12 | 0 | 16 | 0 | 2 |
| **TOTAL** | **445** | **0** | **1,803** | **1,031** | **780** |

The 445 variant constructors are exactly the `SyntaxTerm::{B,Bu,Fourth,Nb}` and
`Atom::{Bool,Num,Str}` sites (`SyntaxTerm::Bu` 193, `SyntaxTerm::B` 17, `SyntaxTerm::Fourth`
7, `Atom::Str` 12, `Atom::Num` 4, `Atom::Bool` 1 in kleis; the rest in opsfold/webfold/
sangsec/spbench/netfold/investfold). Caveat: the parent-type test uses the `hashtrinity`
inventory only, so a few `hashtrinity_crypto` associated functions whose leaf name (`new`)
also exists as a target free function can fall into `free_fn`; the join is a documented
manual correction, not tool output.

### 1.4 Syntactically attributed calls by operation role

| role | trans | net | web | ops | app | inv | san | kleis | mem | pers | spb | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| map construct | 4 | 34 | 31 | 97 | 23 | 10 | 2 | 0 | 4 | 1 | 0 | 206 |
| map read/index | 11 | 4 | 40 | 86 | 28 | 23 | 2 | 0 | 2 | 4 | 1 | 201 |
| map mutate/set | 8 | 25 | 46 | 72 | 24 | 12 | 2 | 0 | 2 | 0 | 1 | 192 |
| map count/monoid | 0 | 0 | 0 | 17 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 19 |
| map graph | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 11 |
| map other (2 residual FPs + `Dense`) | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| codec / term | 1 | 14 | 14 | 388 | 0 | 15 | 43 | 1,927 | 0 | 0 | 28 | 2,430 |
| API substitute | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 883 | 0 | 0 | 0 | 883 |
| runtime / platform | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| secret access | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| crypto / encoding (deferred) | 0 | 0 | 0 | 32 | 0 | 0 | 1 | 76 | 0 | 0 | 0 | 109 |
| **total** | **24** | **78** | **131** | **706** | **75** | **61** | **50** | **2,886** | **12** | **6** | **30** | **4,059** |

The exact call sites are in `data/other-calls.tsv`; per-callee counts are in
`data/other-symbol-counts.tsv`; role sums are in `data/other-role-counts.tsv`. The `map other`
row is not all real: 2 of its 3 opsfold rows are the residual `Rc::clone` false positives
(§2.2, marked `false_positive` in `data/other-manual-confirmed.tsv`); the third is
`Dense::zeros`. Excluding the 2 known false positives leaves 4,057 syntactic
attributions; this does not verify the remaining attributions.

### 1.5 Per-crate + subsystem sums (aligned JSON)

`docs/call-census/data/other-sums.json` carries, per crate: files/parse errors, records,
attributed, unattributed candidates, heuristics, ambiguous, dynamic, other method calls,
target imports, derives, attribute macros, `by_source_location` (all records),
`by_operation_kind` (tool, all records), `by_operation_kind_attributed_inventory_join`,
`by_subsystem` (all records) and `by_subsystem_attributed`. It is generated and the
generator asserts reconciliation: per-crate records sum to the total, per-crate attributed
sum to the total, per-crate unattributed sum to the total, and the inventory-join
operation-kind sums, the role sums and the attributed-subsystem sums each equal the total
attributed (4,059). Top attributed subsystems per crate:

| crate | attributed | top attributed subsystems |
|---|---:|---|
| transcode | 24 | doc 22, aac 1, content_pack 1 |
| netfold | 78 | tests 34, netbox 28, extract 8, compliance 4, parse 2 |
| webfold | 131 | crawl_index 19, engine 16, flaresolverr 14, webfold_mcp 13, cdp 12 |
| opsfold | 706 | datalake 199, vision 96, estate 90, platform 79, thingsboard 40 |
| appfold | 75 | delta 25, handlers 9, agent_tools 7, client 7, spec_render 6 |
| investfold | 61 | portfolio 23, store 23, provider 12, dispatch 1, freds 1 |
| sangsec | 50 | clawhub_structural 34, talon 4, crate 4, clawhub_codebench 2, tests 2 |
| kleis | 2,886 | users 668, pam 426, oidc 355, clients 265, auth_providers 241 |
| memoryfold | 12 | crate 12 |
| personafold | 6 | crate 6 |
| spbench | 30 | task 22, persona 6, report 2 |

### 1.6 Surface narrowness (qualified)

Union of distinct target symbols with at least one attributed site = **100**; distinct
`Table` methods = **24**. The target inventory has 340 types, 635 methods, 954 fns, 34
traits, 30 trait-method maps, 54 variant maps and 22 tuple structs. So the attributed
surface is a small slice. For the faces below there are **no attributed sites in this
census and none found by manual source search** across the 11 subtrees — this is not a
claim that they are unreachable or unused elsewhere:

| primitive face | attributed sites + manual search |
|---|---|
| `FreqMap`, `CoordMap` | none |
| `encircle` / `encircle_floor` | none |
| `get_all` / `get_each` / `get_reduce` | none |
| `FreqMap::count_par` | none |
| `Dense::fuse` / `unfuse` / `span` | none (only `Dense::zeros`, opsfold vision NMS) |
| `entry` / `get_exact` / `remove_at` | none |
| `FoldExt` builders | 6 heuristic sites (netfold ×1, opsfold ×5) — not attributed |

## 2. Scanner verification (schema 2)

The 0.2 tool fixed the three v1 defects; two residual classes remain, and the residual
false positive is flagged and excluded from the verified attributed count.

### 2.1 Fixed in 0.2 (confirmed on this corpus)

- **Shadowed receiver type.** v1 recorded `opsfold/src/vymanager.rs:95` `devices.clone()` as
  `Table::clone` after `let devices = Arc::new(devices)`. In v2 those 19 sites are gone
  (`Table::clone` attributed only 2 corpus-wide; see 2.2).
- **`&mut TargetType` receivers.** v1 dropped them. v2 attributes
  `opsfold/src/thingsboard/service.rs:558` `last_forecast_row.get(&device)` and `:560` `.put`,
  `:671`/`:674` (receiver `&mut Table<String, usize>`), and
  `spbench/src/report.rs:33` `table.insert_with(key, 1, |old, new| old + new)` (receiver
  `&mut SignalTable`).
- **`FoldExt` trait methods.** v1 did not even list them. v2 inventories the trait and records
  the six sites as `heuristic` (`netfold/src/lib.rs:1321` `index_by`;
  `opsfold/src/platform.rs:515` `set_by`, `netops.rs:7719/7789/7843` `index_by`,
  `estate.rs:2279` `set_by`), counted in the 102 heuristics and not in the 4,059.

### 2.2 Residual false positive (2 sites, flagged, excluded from the verified count)

`opsfold/src/bizops.rs:5196/5197`: `store.clone()` where
`store: Rc<RefCell<Table<String, String>>>`. The tool peels the wrapper, sees `Table`, and
attributes `Table::clone`; the actual method is `Rc::clone`. These 2 rows are in the
attributed ledger but are marked `false_positive` in
`data/other-manual-confirmed.tsv`; the real attributed count excluding them is 4,057. This is
the only residual false-positive class found; other `store.clone()` sites in the same file are
unattributed because their receiver type is not written down.

### 2.3 Residual recall gaps (manual candidates, kept separate)

Not attributed and not added to any total; each has a stable manual id in
`data/other-manual-confirmed.tsv`:

| id | crate | file:line | callee | why the tool misses it |
|---|---|---|---|---|
| `manual:transcode:src/aac.rs:1049` | transcode | `src/aac.rs:1049` | `Table::get` | receiver `MutexGuard<Table<usize, &Twiddles>>` |
| `manual:transcode:src/aac.rs:1085` | transcode | `src/aac.rs:1085` | `Table::put` | same wrapped receiver |
| `manual:opsfold:src/bizops.rs:5198` | opsfold | `src/bizops.rs:5198` | `Table::get_at` | receiver `Ref<Table<String,String>>` |
| `manual:opsfold:src/bizops.rs:5200` | opsfold | `src/bizops.rs:5200` | `Table::insert_at` | receiver `RefMut<Table<String,String>>` |
| `manual:spbench:src/report.rs:40` | spbench | `src/report.rs:40` | `Table::new` | local alias `SignalTable = Table<String,i64>` in a path call |

These are hand-confirmed only; they are **not** an estimate added to the 4,059.

### 2.4 `hashtrinity_crypto` is absent from the snapshot

The default target-crate list contains `hashtrinity_crypto`, and the provider imports
`hashtrinity_crypto::{argon2, ed25519_compact, rsa, sha256, sha384, sha512}` (16 files). No
`hashtrinity_crypto` package exists in the snapshot; `hashtrinity/Cargo.toml` says the crate
is *subsumed* into the `hashtrinity::crypto` module. The 63 attributed calls are therefore
attributed **by crate name only**; their real link is unverified. `kleis/provider/` has no
workspace root (its 13 crates use `*.workspace = true`), so the provider is not a buildable
part of this snapshot.

## 3. Per-crate findings (schema 2 counts)

### 3.1 `transcode` — sovereign audio/image codec (24 attributed)

Entrypoints `src/lib.rs`, `src/main.rs`. Two unrelated uses:

- **Incidental cache (AAC).** `src/aac.rs:1046` `hashtrinity::Table::new()` inside
  `mdct_twiddles`, a `OnceLock<Mutex<Table<usize, &'static Twiddles>>>` memo of MDCT twiddle
  factors. The cache's read/write (`map.get` :1049, `map.put` :1085) are manual candidates
  (§2.3). This is the "AAC Table just cache" case; `aac_tables.rs` constants are spec tables,
  not `Table`.
- **Essential index + counter (PDF).** `src/doc/pdf/mod.rs:35` re-exports `hashtrinity::Table`
  as `crate::Table` (the corpus's only target re-export). `Document.objects: Table<u32,Object>`
  is the xref object table (`put`/`get`/`len`/`iter`); `RegionSeed.votes` uses the arbitrary
  merge `insert_with` (`:57`) as a `(table, ¬table)` vote accumulator, read by `get`. Now 22 of
  24 attributed calls sit in `doc/`. `src/doc/html.rs` uses the set face (`has`/`mark`).
- **Codec substitute.** `src/content_pack.rs:49` `parse_surface(Surface::Json, …)`.
- Numeric work (MDCT/DCT, entropy coding) is local, not hashtrinity algebra.

### 3.2 `netfold` — vendor-neutral network-model IR (78 attributed, 45 derives)

Entrypoints `src/lib.rs` (`NetworkModel`), `src/netbox/client.rs`, `analyze/`, `parse-edgeos/`,
`parse-unifi/`. An IR/analysis stack, not packet transport.

- **Essential state/index.** `Table` fields on containers (`interfaces`, `vlans`, `zones`,
  `to_zone`, `vrrp_groups`, `dhcp`): `new` 34, `put` 17, `insert` 1, `get_mut_at` 1, `has` 3,
  `mark` 7. Representative: `analyze/src/compliance.rs:788` `Table::new`; `:789` `insert`;
  `parse-edgeos/src/extract.rs:121` `put`; `src/lib.rs:900` `get_mut_at`;
  `src/netbox/drift.rs:98/138` `mark`/`has`. 34 of 78 attributed calls are in the `tests`
  subsystem (test fixtures building models).
- **Codec/term.** `#[derive(Term)]` ×10, `#[derive(Collapse)]` ×22, `#[derive(Error)]` ×13.
  Hand-written `AsTerm` (`src/netbox/client.rs:56/69/71/72`), JSON read `:155`
  `parse_surface`, write `src/netbox/timeline.rs:31` `render_surface`, Pod round-trip
  `src/lib.rs:1101/1102` `CollapseElem::collapse_pack/open`.
- **Runtime/platform.** `src/netbox/client.rs:149` `hashtrinity::sync::http::request` (the
  NetBox client's HTTP transport).
- **Documented limits** (`lib.rs:10–35`, `config.rs:8–17`): see §6.

### 3.3 `webfold` — web extraction (131 attributed)

Entrypoints `src/lib.rs`, `src/main.rs`, `src/bin/{bench,flaresolverr,webfold_mcp}.rs`. The
&mut/alias fixes raised this from 68 to 131.

- **Essential state/index.** Crawl BFS seen-set (`src/main.rs:1114` `has`/`mark`); MCP index
  `Table<String,bool>` (`webfold_mcp.rs:1071` `new`, `:1088` `iter`), by-url maps
  (`:1111/1258`). Now `map read` 40 and `map mutate` 46.
- **Evaluation harness (bench).** `src/bin/bench.rs` n-gram sets and truth-by-url.
- **Codec substitute.** `src/bin/flaresolverr.rs` uses `parse_surface`/`render_surface` and
  `SyntaxTerm`/`Atom` as its JSON codec.
- **Bypass.** 9 files keep `std::collections::HashMap` for page/exchange/robots/import maps
  (annotated `LOOP-EXEMPT: get_mut/remove needed`), although `Table::get_mut` and
  `Table::remove` exist.

### 3.4 `opsfold` — platform-ops invariant (706 attributed)

Entrypoints `src/lib.rs` and seven bins. The largest real primitive consumer and the only
real algebra consumer.

- **Essential state/index.** `new` 97, read/index 86, mutate/set 72, graph 7. Representative:
  `src/alerting.rs:2048` `new`/`:2182` `put`; `src/collect/pve.rs:81` `get`/`:37` `iter`;
  `src/vymanager.rs:72` `get_at`; `src/session_episodes/transcript.rs:299` `remove`;
  `src/platform.rs:118` `neighbors`; `src/integrations.rs:1688` `values`;
  `src/thingsboard/service.rs:558/560/671/674` now attributed.
- **Real numeric/algebra kernel.** `impl Monoid` for `Counter` (`flowops.rs:686`),
  `SeenStamp` (`vision/post.rs:15`), `IfaceCounters` (`collect/vyos_flow.rs:45`);
  `accumulate` 10 (`datalake.rs:2227`, `flowops.rs:807`), `fold` 1 (`flowops.rs:813`),
  `fuse` 3 (`session_episodes/trace.rs:149/150/160`), `add` 2,
  `Dense::<SeenStamp>::zeros` (`vision/post.rs:96`) + graph-face `push`/`neighbors` for NMS
  broad-phase. The arithmetic (IoU, NMS ordering, deltas) is local; hashtrinity supplies the
  address/merge substrate. The algebra is the inline `Monoid::combine`/merge closure, so
  absence of a named "sigma" call does not mean absence of algebra.
- **Codec/term (388).** `field` 113, `SyntaxTerm::Bu` 100, `field_str` 47, `parse_surface` 46,
  `render_surface` 22, `AsTerm`; `#[derive(Collapse)]` ×178, `#[derive(Term)]` ×17.
- **Runtime/platform.** `sync::http::{request,request_plain}` ×3 (`estate.rs:2958/2968`);
  `SecretString::expose_secret` ×1 (`estate.rs:4858`).
- **Deferred crypto (32).** See §7.
- **Heuristics (16).** `FoldExt::set_by`/`index_by` (§2.1) plus other trait-provided methods.

### 3.5 `appfold` — full-stack app framework (75 attributed)

Entrypoints `src/lib.rs`, `examples/counter.rs`, bin `forgefold`. Per-client session state,
agent-proposal store, dirty-var delta maps (`delta` subsystem 25). Representative:
`examples/counter.rs:18/20`; `src/agent_tools.rs:75/78/88`.
**Bypass:** `src/agent_tools.rs:78` removes a proposal by rebuilding the table instead of
`Table::remove` (which returns the removed value).

### 3.6 `investfold` — OpenBB terminal, derived (61 attributed)

Entrypoint `src/lib.rs`. Storage "five faces of one `Table`": `portfolio` 23 (`iter`/`put`/
`get`/`new`), `store` 23 (`add` column sum at `store.rs:306`), `provider` 12 (`parse_surface`
at `dispatch.rs:292`). Heavy analysis is delegated to `sangdb`/`modelfold`.

### 3.7 `sangsec` — SangSec ported to Rust (50 attributed)

Entrypoint `src/lib.rs`, bin `talon`. 43 of 50 are codec/term, and 34 are in the
`clawhub_structural` **example**, not the library (`SyntaxTerm::B` 12, `parse_surface` 11,
`Atom::Str` 8, `SyntaxTerm::Fourth` 6, `msgpack::encode` 1). Quarantine store in the `talon`
bin (`Table<i64,QuarantineRow>`: `new`/`put`/`iter`). `crypto::md5` (`talon.rs:736`).
**Bypass:** 9 files use `HashMap` for rule indexes/token counts/SBOM.

### 3.8 `memoryfold` — episodic memory retrieval (12 attributed)

Entrypoint `src/lib.rs`, bin `memorybench`. Cleanest adoption: token/entity postings are
graph-face `Table<K,Vec<usize>>` (`with_capacity` 4, `push` 2, `neighbors` 2) plus a
`store_ids` map (`get`/`put`).

### 3.9 `personafold` — persona enumeration (6 attributed)

Entrypoint `src/lib.rs`. Implements `Coord`/`CoordInv` for `PersonaKey`; `SignalTable {
inner: Table<PersonaKey,i64> }` uses `default`/`add`/`count`/`len` — the count face driven by
a custom coordinate key.
**Bypass:** `ValueCatalogue = Vec<HashMap<u32,String>>` and `labels: HashMap<String,String>`
keep `HashMap` for the label catalogue.

### 3.10 `spbench` — persona benchmark runner (30 attributed)

Entrypoint `src/lib.rs`. 28 codec/term calls (`term_str`, `SyntaxTerm::B`, `term_num`,
`field`, `parse_surface`, `term_bool`) parse YAML/JSON task bundles. Its declared core —
signal accumulation into `SignalTable = Table<String,i64>` — is now partly attributed
(`report.rs:33` `insert_with`, `:75` `iter`); the constructor `SignalTable::new()`
(`report.rs:40`) remains a manual candidate (§2.3).

### 3.11 `kleis` — credential plane on sangdb (2,886 attributed, all in the vendored provider)

**Core `kleis` package:** 14 files, 89 records, **0 attributed**, 1 target import
(`src/oidc_wire.rs:179` `use hashtrinity::base64::Engine;`, in the `oidc`-feature module) and
3 records under `cfg(feature = "columnar")` (`Account` derives `hashtrinity::Collapse`;
`idx: sangorm::Db`). So core kleis's direct hashtrinity use is entirely behind non-default
features; its runtime credential plane uses `sangorm`/`sangdb` (bedrock-owned). The 2,886
attributed calls all belong to the 13 vendored `rauthy-*` packages (§5).

## 4. Operation-role grouping (global)

- **Incidental cache / bookkeeping.** `transcode/src/aac.rs:1046` (MDCT twiddle cache);
  `webfold/src/bin/bench.rs` (evaluation harness); `with_capacity`-then-`put` build loops.
- **Essential state / count / index.** transcode PDF object table + vote accumulator;
  netfold model containers + drift sets; webfold crawl/index; opsfold flow/datalake/vision/
  quarantine/vymanager; appfold sessions/proposals/deltas; investfold row/column/edge;
  sangsec quarantine; memoryfold postings; personafold signal counts.
- **Codecs / errors / API substitutes.** All `codec::syntax::*`, `codec::web::*`, `codec::msgpack`;
  derives `Term`/`Collapse`/`Error`/`Validate`/`ToSchema`/`IntoParams`; the entire
  kleis-provider openapi/validate/codec/base64/hex/crypto surface.
- **Runtime / platform.** `sync::http::{request,request_plain}` (netfold ×1, opsfold ×3);
  `SecretString::expose_secret` (opsfold ×1).
- **Real numeric / algebra kernel.** opsfold `Monoid` + `accumulate`/`fold`/`fuse` + `Dense`
  (flow accounting, episode census, vision NMS); investfold `add`; personafold count face.
- **Deferred crypto.** §7.
- **Unknown.** 11,136 unattributed candidates and 45,448 other method calls; dominated by
  std-shaped names (`map`/`push`/`collect`/`iter`/`len`/`is_empty`/`get`/`insert`/`put`).
  Kleis alone contributes 2,830 candidates and 12,860 other calls, almost all std/tokio
  methods in the vendored provider. The scanner cannot attribute them; they are a work queue,
  not target calls.

## 5. Vendored upstream provider (`kleis/provider`, Rauthy) — precise API-substitute summary

The `kleis` subtree contains the core `kleis` package plus 13 vendored Rauthy packages. The
`by_crate` split confirms the classification:

| package | files | records | attributed | target imports |
|---|---:|---:|---:|---:|
| `kleis` (core) | 14 | 89 | 0 | 1 |
| `rauthy-api-types` | 22 | 4,101 | 2,726 | 99 |
| `rauthy-data` | 121 | 824 | 103 | 51 |
| `rauthy-error` | 2 | 43 | 28 | 18 |
| `rauthy-handlers` | 27 | 479 | 10 | 25 |
| `rauthy` (bin) | 33 | 118 | 5 | 4 |
| `rauthy-common` | 10 | 32 | 10 | 4 |
| `rauthy-service` | 27 | 90 | 4 | 1 |
| `rauthy-jwt` | 3 | 25 | 0 | 1 |
| `rauthy-derive` / `rauthy-middlewares` / `rauthy-notify` / `rauthy-schedulers` / `rauthy-wasm-modules` | 32 | 1 | 0 | 0 |
| **provider total** | **277** | **5,713** | **2,886** | **203** |

The 2,886 attributed calls break down by the corrected `operation_kind` separation
(§1.3): **1,479 free_fn, 819 associated_fn, 354 method, 234 enum-variant constructors, 0
tuple constructors**. Do not conflate these:

- **Enum-variant construction (234):** `SyntaxTerm::{B,Bu,Fourth,Nb}` (217) and
  `Atom::{Bool,Num,Str}` (17). These are datatype construction via the `hashtrinity` codec
  AST, e.g. `provider/src/api_types/src/api_keys.rs:106` `SyntaxTerm::Bu(...)`.
- **Method calls (354):** a receiver method, e.g.
  `provider/src/api_types/src/api_keys.rs:350` `__errors.add(...)` on
  `hashtrinity::validate::ValidationErrors`, `:364` `into_result`.
- **Free functions (1,479):** e.g. `codec::syntax::field` (586), `SyntaxError::new` (457),
  `term_str` (410), `openapi::schema_object` (171).
- **Associated functions (819):** e.g. `validate::ValidationError::new` (223),
  `ValidationErrors::new`, `rsa::RsaPrivateKey::new`, `argon2::Params::new`.

Grouped by substitute role: **codec/serde 1,927** (24 files), **validate/validator 676**
(19 files), **openapi/utoipa 207** (28 files), **crypto 63**, **base64/hex 13**. Attribute/
derive surface: 216 `#[hashtrinity::openapi::path(...)]`, 26 `ToSchema`, 16 `Term`, 8
`Validate`, 1 `Collapse`. The provider has **zero references to `Table`, `Dense`, `FreqMap`
or `CoordMap`** — verified by attributed sites and by manual grep over `kleis/provider`; the
map primitive is unexercised there. The `hashtrinity` paths were spot-checked against the
inventory and all exist, so these are real syntactic substitute calls, not phantom names.

**Datatype use is not call use.** The provider also names hashtrinity types in field/parameter
positions and derives (e.g. `ValidationErrors` as a type, `SyntaxTerm` as a return type); those
are not calls and are not counted as such. The counts above are call records only.

## 6. Primitive-gap interpretation (reviewed)

**(a) Documented capability limits — findings, not automatically features to build.**

1. **`Collapse` cannot express `Table` fields** (`netfold/src/lib.rs:15–20`). Containers that
   own maps (`NetworkModel`, `Interface`, `Bond`, `FirewallModel`, `FirewallZone`, `DhcpModel`)
   do not derive `Collapse`; they use a manual leaf-column encode/rebuild. Whether the fix is a
   `CollapseElem` impl for `Table`, a nested-schema form, or leaving it explicit is a design
   decision, not an obvious missing function.
2. **`Option<T>` has no `Pod`/`CollapseElem` impl** (`netfold/src/lib.rs:30–35`). The consumer
   wraps optionals in local `OptU8`/`OptU32`/`OptIpAddr`/… types. This is **not** automatically
   a feature to implement: `Option<T>` has no guaranteed layout, and `bool`/enum validity and
   padding are not something a `#[repr]`-less generic `Pod` impl can assume. Satisfying an
   arbitrary wire format by transmuting `Option`/enum bytes is unsound; a blanket `unsafe Pod`
   impl is **not** the nominee. The safe direction is an **explicit schema encoding with a
   round-trip test** (tagged presence + payload encoded by an explicit `CollapseElem`
   implementation, or an opt-in derive), never struct-byte reinterpretation.
3. **No insertion-order-preserving table** (`netfold/src/config.rs:8–17`): ordered first-match
   config semantics use `Vec<(String, ConfigNode)>` because `Table` is a hash table. This is a
   real shape mismatch; an ordered/linked face or a documented iteration-order contract would
   address it, but the current workaround is correct and needs no unsafe change.

**(b) Faces with no attributed sites + manual search (not "unreachable").** `FreqMap`,
`CoordMap`, `encircle`, `get_all`/`get_each`/`get_reduce`, `count_par`, `Dense::fuse/unfuse/span`,
`entry`/`get_exact`/`remove_at`, and the `FoldExt` builders (6 heuristic sites, not
attributed). The README's recommended NB count/join/dedup engine (`FreqMap`) has no
attributed first-party consumer; that is a measured adoption gap in this corpus, not a proof
it is unused in code the census did not walk.

**(c) Bypasses where a primitive exists.** `appfold/src/agent_tools.rs:78` rebuilds a table to
remove one key instead of `Table::remove`; `personafold` keeps `HashMap` for labels;
`webfold` keeps `HashMap` for page/exchange/robots/import maps (annotated `LOOP-EXEMPT`,
though `get_mut`/`remove` exist); `sangsec` keeps `HashMap` for rule indexes/token counts/SBOM
(9 files); `spbench` used `insert_with(k,1,|a,b| a+b)` rather than the dedicated `Table::add`
(this one is now attributed).

**(d) Provider non-use.** The absorbed identity provider uses hashtrinity only as an
API-substitute bundle (§5); the map primitive is unexercised there.

## 7. Crypto status — deferred (exact gate)

Cryptographic use is **deferred**, and adoption/verification is gated exactly as stated:
**only after everything else is mature, and then only after full end-to-end verification plus
independent external analysis.** This census establishes no cryptographic correctness,
key-management correctness, constant-time behaviour or side-channel resistance.

Deferred surface: opsfold `hashtrinity::crypto`/`hex`/`base64` (32 calls: `sha256`, `to_hex`,
`hex::encode`/`decode`, `blake3`, `hmac_sha256`, `ed25519_pubkey`/`sign`/`verify`); sangsec
`crypto::md5` (1); kleis/provider `hashtrinity_crypto` + base64/hex (76, with the crate-name
mismatch of §2.4). Core `kleis`'s own sealing/OTP design is deliberately deferred in its
lib.rs ("Left to the next pass so this one ships correct rather than crypto-shaped").

## 8. What is proven, tested, and open

- **Census evidence (schema 2):** 586 files, 0 parse errors; the attributed call syntax and
  roles above; the two residual false positives and five residual recall gaps identified with
  stable ids; the core-kleis / vendored-provider split; the provider's zero map-primitive use.
- **Hand-verified:** representative `file:line` for every attributed callee group; the
  `Rc::clone` false positive read at source; the wrapper-receiver misses read at source; the
  `FoldExt` heuristic sites and imports read at source; core kleis's feature-gated use.
- **Open / not proven:** compiler resolution; macro-expanded calls; generic/`Deref`/trait
  dispatch; whether `hashtrinity_crypto` resolves (absent from snapshot, no provider
  workspace root); whether further wrapper-peeling false positives exist beyond the 2 found;
  and all cryptographic claims (§7). The 445-constructor separation is a manual inventory
  join, not tool output, and carries the caveat in §1.3.

## 9. Reproduction and artifacts

```sh
# raw schema-2 census
ls /tmp/opencode/census-v2/other/<crate>/{meta,summary,records,calls,unattributed_methods}.json 2>/dev/null
# durable attributed callsite ledger (4,059 rows)
head -1 docs/call-census/data/other-calls.tsv
# aligned per-crate + subsystem sums (reconciliation asserted by the generator)
python3 -c "import json;d=json.load(open('docs/call-census/data/other-sums.json'));print(d['totals'])"
# manual candidates and residual false positives (separate, not added to totals)
cat docs/call-census/data/other-manual-confirmed.tsv
```

Report generated 2026-09-11 against snapshot `4cc14471ae24f3f204fb0b0e91736ef75c909358`,
schema 2 / tool 0.2.0 (binary sha256 `5600cfc46cacd0ed871b10960a97cc8bd5dfb36f13f0d30760f8f7ab320824a1`).
