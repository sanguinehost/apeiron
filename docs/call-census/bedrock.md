# Bedrock → Hashtrinity call census (syntactic attribution, schema v2)

Scope: **`bedrock` only**, all `src` / `src_bin` / `examples` / `tests` contexts separated.
Method: `call-census` 0.2.0 (AST / `syn`). **"Attributed" means a syntactic match, not compiler
resolution.** It is not a lower bound, not a proof, not a call graph, and not a count of executed
calls. `source_location` is where the text lives; `reachability` is always `not_determined` — the
tool does not build a module graph and never asserts a file is compiled. `cfg` predicates are
propagated verbatim from the item and every enclosing module/impl, but are **not evaluated**.

## Provenance / reproducibility

| field | value |
|---|---|
| snapshot | `/tmp/opencode/fold-call-census-20260911` |
| SHA | `4cc14471ae24f3f204fb0b0e91736ef75c909358` (branch `main`, 0 porcelain lines) |
| tool | `/tmp/opencode/call-census-target/release/call-census` (`tool_version 0.2.0`, `schema_version 2`) |
| final rerun | all 34 schema-2 runs rechecked after scanner formatting; current hashes in [`data/census-audit.json`](data/census-audit.json) |
| output | `/tmp/opencode/census-v2/bedrock/` |
| per-call TSV | [`data/bedrock-calls.tsv`](data/bedrock-calls.tsv) (copy of `calls.tsv`) |

```sh
SNAP=/tmp/opencode/fold-call-census-20260911
/tmp/opencode/call-census-target/release/call-census \
  --root $SNAP/bedrock \
  --target-src $SNAP/hashtrinity \
  --target-src $SNAP/hashtrinity/hashtrinity_derive \
  --output /tmp/opencode/census-v2/bedrock
```

Parse coverage (from `summary.json` / `files.json`): **429/429 `.rs` parsed, 0 parse errors, 0
target-crate files** — `src` 251, `src_bin` 58, `examples` 108, `tests` 12. Line/column are
`proc-macro2` span-accurate.

## 0. Summary totals

| metric | count |
|---|---|
| recorded call expressions | 13178 |
| **syntactically attributed** (high 1580, medium 0) | **1580** |
| — `method_attributed` | 771 |
| — `target_direct_call` | 523 (free_fn 324 + unclassified 199) |
| — `target_assoc_fn_call` | 286 (associated_fn 251 + constructor 35) |
| **constructors** (`operation_kind = constructor`) | **35** (all `par::SendPtr`) |
| unattributed: candidate / ambiguous / heuristic / dynamic | 11428 / 102 / 31 / 37 |
| **unknown total** (unattributed) | **11598** |
| other (known non-target) method calls, aggregated only | 49077 |
| distinct target symbols attributed | 108 |
| distinct callers | 781 |

`13178 = 1580 attributed + 11598 unattributed`; `11598 = 11428 + 102 + 31 + 37`. All aggregate
tables in `summary.json` re-sum to these totals (category, operation-kind, source-location,
subsystem, target-symbol).

## 1. Exact attributed sites by callee and subsystem

Attributed by capability group (all 1580):

| group | sites | notes |
|---|---|---|
| `Table::*` | **931** | 217 `new`; 714 method calls (see §6) |
| `codec::syntax::*` | 323 | `parse_surface` 79, `render_surface` 31, `SyntaxError::new` 45, `SyntaxTerm::*` 90, `Atom::*` 35, `field*` 26, … |
| `crypto::*` | 136 | sha256 18, blake3 26, to_hex 30, ed25519 21, x25519 6, hmac 5, hkdf 5, chacha 6, aes-gcm 4, ct_eq 2, rand 9, quic 3 |
| `CoordMap::*` / `Coord::coord` | 87 | `coordinate` 23, `put` 9, `insert_with` 9, `get` 9, `len` 8, `iter` 8, `into_iter` 7, … |
| `par::*` | 55 | `Pool::global` 20, `SendPtr` 35 (all constructors) |
| other `codec::*` | 38 | msgpack, `bitpack_i64`/`bitunpack_i64`, compress, `encode_i64`, `web::*` |
| `sync::http::request` | 3 | HTTP masquerade |
| `rustls::provider` | 2 | TLS provider masquerade |
| `S3::get` / `S3::put` | 2 | qualified `hashtrinity::sync::s3::S3` at `src/sangdb/mod.rs:5660,5668` |
| `Sha256::{update,finalize}` | 2 | target type direct |
| `hashtrinity_crypto::ed25519_pubkey` | 1 | cross-crate path call |

Top callees: `Table::new` 217, `Table::get` 205, `Table::put` 143, `Table::iter` 108,
`codec::syntax::parse_surface` 79, `SyntaxError::new` 45, `SyntaxTerm::Bu` 41, `Table::remove` 40,
`par::SendPtr` 35, `Table::insert_with` 34, `render_surface` 31, `crypto::to_hex` 30.

Per-subsystem totals (imports / attributed / candidates / ambiguous / dynamic / heuristic) are in
[`data/bedrock-subsystem-coverage.tsv`](data/bedrock-subsystem-coverage.tsv); the full
callee × subsystem × source-location × operation-kind cross-product is in
[`data/bedrock-callee-subsystem.tsv`](data/bedrock-callee-subsystem.tsv). Largest attributed:
`sangdb` 367, `bin` 158, `transport` 136, `serve` 123, `bindings` 113, `ontology` 100,
`appfold` 69, `runtime` 65, `domain` 61, `htmlfold` 52.

## 2. `operation_kind`: constructors vs functions/methods

| operation_kind | sites | what it covers |
|---|---|---|
| `method` | 12369 | all method calls (attributed + candidate + dynamic + heuristic) |
| `free_fn` | 324 | path free functions (`codec::syntax::parse_surface`, `crypto::*`) |
| `associated_fn` | 251 | `Table::new` 217, `CoordMap::coordinate` 23, `Table::with_capacity` 6, `from_iter` 2, `coord` 1 |
| `unclassified` | 199 | target path calls with no inventory match — mostly enum-variant constructors (`SyntaxTerm::Bu` 41, `SyntaxTerm::Fourth` 27, `SyntaxTerm::B` 22, `SyntaxError::new` 45, `Atom::*` 35, `par::Pool::global` 20) |
| **`constructor`** | **35** | **all `par::SendPtr`** (tuple-struct constructor), e.g. `serve/model/matvec.rs` 13, `serve/gpu.rs` 11, `serve/model/batch.rs` 7 |

Of the 1580 attributed: 771 `method`, 324 `free_fn`, 251 `associated_fn`, 199 `unclassified`,
35 `constructor`. Constructor and associated-fn counts are exact from the `operation_kind` column
of [`data/bedrock-calls.tsv`](data/bedrock-calls.tsv).

## 3. cfg propagation and source location

- **137 attributed calls carry a non-empty `cfg`** (104 `cfg(test)`, 18 `cfg(feature="native")`,
  10 `cfg_attr(feature="wasm", wasm_bindgen)`, …); 2203 unattributed records carry a `cfg`.
- `source_location` (attributed): `src` 1320, `src_bin` 158, `examples` 82, `tests` 20.
  All 13178 records: `src` 9619, `src_bin` 2317, `examples` 1142, `tests` 100.
- **None of this means the code is compiled.** `reachability` is `not_determined` for every record;
  `cfg` is recorded verbatim and not evaluated.

## 4. Unattributed quantification and the manual ledger

- **11428** `method_candidate` (name matches a target method, receiver unresolved);
  **11382** of these have **no receiver type recorded**.
- **102** `method_ambiguous`; **31** `method_trait_heuristic`; **37** `method_dynamic_dispatch`.
- Top candidate method names and category split:
  [`data/bedrock-unattributed-top.tsv`](data/bedrock-unattributed-top.tsv). Leaders: `map` 1811,
  `collect` 1304, `iter` 914, `len` 808, `push` 758, `parse` 543, `get` 422, `is_empty` 389.

**Manually confirmed unknown-operation ledger** — kept separate from the attributed count:
[`data/bedrock-manual-ledger.tsv`](data/bedrock-manual-ledger.tsv).

| class | sites | basis | disposition |
|---|---|---|---|
| `method_ambiguous` | 102 | target-name collision with a non-target glob in scope: `Engine` 101 (bedrock `serve::Engine`, `impl Engine` at `src/serve/probe.rs:348`, `src/serve/walker.rs:87`; target `Engine` is the base64 **trait** at `hashtrinity/src/base64.rs:106`), `mut Response` 1 | not a target call |
| `method_trait_heuristic` | 31 | imported target trait name, receiver type unknown: `FoldExt` 13, `AsTerm` 8, `Engine` 7 (base64 encode/decode), `Coord` 3 | labelled heuristic only |
| `method_dynamic_dispatch` | 37 | trait object / generic bound | not resolvable syntactically |
| `candidate_const_receiver` | 2 | `ZBASE32.encode`/`decode` `src/transport/signed_dns.rs:100,108`; `hashtrinity::base32::ZBASE32` imported at `:40` | manually confirmed target call; not attributed |
| `candidate_noncompiled_copy` | 19 | `CoordMap` in `src/native_lib/proxy.rs` 9, `http_types.rs` 7, `router.rs` 3; live merged file `src/native_lib.rs:42` imports `CoordMap` | dead source copies; **cannot** be added to live misses |
| `candidate_unknown_receiver` | 11382 | no receiver type recorded (of 11428 candidates) | unresolved |
| `macro_generated` | 13818 | unqualified macro invocations, not expanded | generated calls invisible to this census |

## 5. Manual inspection of misclassification

- **Aliases and `&mut` receivers are now attributed by the tool**; the old manual "+20 ambiguous
  alias" adjustment is obsolete and is **not** added anywhere in this report.
  `countlm::Flat`/`Dist` (12 alias sites) and `sangdb::resp` `RowTable`/`ListTable`/`ZSetTable`/
  `HashTable` now attribute as `Table` calls with basis "local type alias …".
- **The 102 ambiguous are name collisions** (§4): 101 `Engine` + 1 `mut Response`, not target calls.
- **Non-compiled orphan source copies.** `src/lib.rs:51` does `include!("native_lib.rs")`, and six
  files under `src/native_lib/` (`client.rs`, `http_types.rs`, `proxy.rs`, `router.rs`, `wire.rs`,
  `tests.rs`) are not referenced by any `mod`/`include!`; their content is merged into
  `src/native_lib.rs` (e.g. `pub struct Client` at `:1562`). The walker still censuses them
  (14 attributed, 368 candidates). Their **19 unqualified `CoordMap` candidates** are dead copies
  of code that the live `native_lib.rs` attributes; they are not missed live calls.
- **Const-receiver base64 calls are heuristic, not attributed.** `STANDARD.encode`/`decode`
  (`src/appfold.rs:4069`, `src/httpsrv.rs:5789,5888`, `src/bin/epihogue/wire.rs:382-386`) appear in
  the 31 heuristic records; `ZBASE32` (no imported `Engine` trait) remains a candidate.
- **Target-inventory pollution still inflates candidates.** The target type set includes test-local
  structs (`Model` `codec.rs:505`, `Document` `crypto.rs:5919`, `SecretKey` `crypto.rs:780`, `Rng`
  `rustls.rs:559`), associated types (`Block`, `Item`, `Store`), traits (`Engine` `base64.rs:106`,
  `Context` `error.rs:70`) and generic params (`P`). The `Engine` collision is the dominant
  residual.

## 6. `Table`: essential storage state vs incidental registries

The earlier blanket phrase "818 Table sites are bookkeeping" is **retracted**. A reproducible split
of the 931 attributed `Table` sites by receiver form:

| form | sites | meaning |
|---|---|---|
| `Table::new` constructions | 217 | allocation of a table |
| methods on `self.<field>` | 381 | struct-owned state — the storage-state proxy (`self.manifests` 21, `self.pages` 19, `self.points` 18, `self.hosts` 17, `self.rows` 9, `self.chunks` 8, `self.edges` 7, `self.logs` 6, …) |
| methods on local variables | 333 | mixed: store handles (`db` 35, `table` 8) and incidental short-lived registries/accumulators (`seen` 32, `reject` 12, `guessable` 8, `by_type` 6, `agg` 6, `children_of` 6, …) |

By subsystem (self-field / local / new): `sangdb` 96/129/68 = 293, `transport` 63/6/17 = 86,
`bin` 0/57/22 = 79, `appfold` 39/10/20 = 69, `runtime` 45/9/11 = 65, `domain` 38/7/5 = 50,
`htmlfold` 9/24/17 = 50, `serve` 13/22/13 = 48, `countlm` 11/9/7 = 27, `ontology` 0/16/10 = 26.

**DB significance is concentrated, not flattened:** `sangdb` alone holds 293 `Table` sites
(96 field-state + 68 constructions) plus its own `SqlTable`/`Lorelake`/`ChunkStore` state;
`ontology` 26, `countlm::Flat`/`Dist` 27, `runtime` worker/node registries 65. Server/app/UI
`Table` uses are predominantly local registries, but they are not all incidental either — the
split above is the evidence, not a blanket label.

## 7. Call traces: aliases, wrappers, field calls, generic helpers

- **Alias chain:** `hashtrinity::CoordMap<K,V>` is `pub type CoordMap<K,V> = Table<K,V,Regen<K>>`
  (`hashtrinity/src/lib.rs:478`). Bedrock builds aliases on top: `RowTable`/`ListTable`/`ZSetTable`/
  `HashTable` (`sangdb/resp.rs:25-31`), `SharedSqlDb = Arc<Mutex<Table<String,SqlTable>>>`
  (`sangdb/mod.rs:1541`), `SharedFold = Versioned<HashMap<String,Table<String,i64>>>`
  (`sangdb/http.rs:59`), `countlm::Flat`/`Dist`. These now attribute with basis
  "local type alias …".
- **Field calls:** 381 attributed `Table` method sites are on `self.<field>` (§6).
- **Generic helpers over `Table`:** `fold_group<K: Eq + Coord, A>` (`sangdb/mod.rs:74`),
  `group_sum<K: Eq + Coord>(…) -> Table<K,i64>` (`:85`), `ingest<K: Eq + Coord, A>` (`:97`),
  `fn tbl<M: Model>(&self) -> Option<&Table<M::Key,M>>` (`:7944`), `fn tbl_mut` (`:7948`).
- **Wrappers, not reimplementations:** `epihogue/wire.rs:381 fn base64_decode` calls
  `general_purpose::STANDARD.decode`; `sangdb` `Store`/`Versioned` wrap `Table`.

## 8. What `sangdb` really computes vs a simple `Table` alias

`sangdb` is the largest consumer (367 attributed, 1830 candidates). Its **storage faces are `Table`
aliases**, but the module owns real computation:

- **Streaming log / consumer offsets:** `Log<T>`, `Law<T>`, `Polled`, `ConsumerGroup` (`mod.rs:113-221`).
- **Inverted index:** `PostingIndex<Id>::build/search/search_prefix`, `LexSegment` (`mod.rs:305-673`).
- **ANN / spatial:** `KdTree<Id>::build/nearest/k_nearest` (`mod.rs:825-903`), `LshIndex<Id>`
  (`sangdb/lsh.rs:87`), `IndexSegment` (`mod.rs:3196`).
- **SQL:** `SqlValue`, `SqlTable`, `SqlQuery`, `SqlExec`; `sql_lex`, `sql_plan_of_tokens`,
  `sql_query_rows`, `sql_run`, `sql_exec` (`mod.rs:1188-1905`).
- **Chunking / content addressing:** `find_boundaries`, `chunk`, `ChunkStore::put_chunk/get_chunk`,
  `MissingChunks`, `Diff`, `Versioned::{pin,write}` (`mod.rs:2956-3342`, `1038`).
- **Lake / versions:** `Lorelake` rows/edges/posting/counts (`mod.rs:3426-3443`).
- **Adapters/protocols:** `resp.rs`, `sqlite.rs`, `meili.rs`, `mqtt.rs`, `s3.rs`, `gossip.rs`,
  `partition.rs`, `prisma.rs`, `crud.rs`, `client.rs`.

No name-text counts are used anywhere in this report. The above are source symbols inspected by
hand; only the attributed call sites in §1/§6 are census quantities.

## 9. What `bedrock` runtime really contributes vs tokio/std wrappers

`src/runtime.rs` is one file holding 11 merged submodules. **Attributed `Table` sites by inline
submodule sum exactly to the runtime total of 65:**

| inline submodule | attributed | symbol |
|---|---|---|
| `work` | 42 | `Table::{iter 17, new 11, put 10, get 9, neighbors 6, push 4, remove 2, len 2, is_empty 2, get_mut 1, add 1}` |
| `cluster` | 14 | `Table::{iter, new, put, get, len, is_empty}` |
| `fiber` | 5 | `Table::{new, put, get, remove}` |
| `metrics` | 3 | `Table::new` |
| `cache` | 1 | `Table::new` |
| **total** | **65** | (42 + 14 + 5 + 3 + 1 = 65) |

`alloc` (mimalloc/jemalloc + `ProfilingAllocator`), `cron` (full cron parser), `daemon`,
`graceful` (`serve_gracefully`, `bind_reuseport`, `ConnectionCounter`), `task_metrics`,
`tracecontext` have **zero attributed target calls** and are std/tokio plumbing. Runtime's
hashtrinity contribution is Table-backed worker/node/task registries, not a tokio replacement.

## 10. Purpose groups, rolecount, and functions that do not reach Hashtrinity

Attributed sites by role (sums to 1580):

| role | sites | subsystems |
|---|---|---|
| DB / storage / index | 554 | `sangdb` 367, `ontology` 100, `countlm` 29, `experiment` 16, `feeds` 15, `spend` 14, `dom` 7, `context` 3, `security` 3 |
| server / HTTP / net | 442 | `transport` 136, `serve` 123, `bindings` 113, `native_lib` 43, `httpsrv` 27 |
| ops / app / UI | 259 | `appfold` 69, `domain` 61, `htmlfold` 52, `ui` 41, `platform` 28, `techdocs` 8 |
| executables (`src/bin`) | 158 | server/inference/ops binaries |
| runtime | 65 | `src/runtime.rs` |
| examples / tests | 102 | 15 example files + `tests/market_store.rs`, `tests/bench_shuffle.rs`, `tests/self_update_e2e.rs` |

**Neural math, qualified:** in the inspected kernel bodies of
`serve/model/{matvec,batch,attn,cold,hybrid}.rs` the only attributed target symbols are
`par::Pool::global` and `par::SendPtr`; `serve/model/forward.rs` attributes only
`codec::web::{Wire::open, pack}`; `serve/model/rowdot.rs` and `serve/model/loaders.rs` attribute
nothing. This is a per-file inspection of those kernel bodies, **not** a transitive-closure claim.

**53 files mention `hashtrinity` in source but have zero attributed calls.** Notable non-aggregator
ones (mentions, 0 attributed): `src/ontology/foundry.rs` (12), `src/ontology/mod.rs` (10),
`src/web/mod.rs` (6), `src/transport/mod.rs` (4), `src/bindings/mod.rs` (4), `src/serve/mod.rs` (3),
`src/serve/model/rowdot.rs` (2), `src/guard.rs`, `src/crud.rs`, `src/bin/epihogue/wire.rs` (has
heuristic, not attributed), `src/sangdb/http.rs`, `src/sangdb/codebook.rs`, `src/model/tokenizer.rs`.
Modules with no target import and no attributed call: `src/model/`, `src/web/`, `src/ops.rs`,
`src/crud.rs`, `src/guard.rs`, `src/compliance.rs`. This does not prove absence of use
(macros/generics could hide it); there is simply no attributed static evidence.

## 11. Same helper duplicated / reimplemented inside Bedrock

Source-level duplication confirmed by reading definitions (not call counts):

| capability | hashtrinity path used | local reimplementations |
|---|---|---|
| base64 | `hashtrinity::base64` at `appfold.rs:3919`, `httpsrv.rs:5721`, `bindings/nextcloud.rs:566`, `bin/epihogue/wire.rs:11` | `fn base64_decode` `native_lib.rs:1856`; `fn base64` `textfold/image/graphics.rs:19`; `base64_decode/encode` `httpsrv.rs:2553,5155,5169`; `bindings/tlog.rs:362,375`; `native_lib/wire.rs:201` |
| hex | `crypto::to_hex` (30 attributed sites) | `fn hex_encode` `bin/syndesi_ui/fleet.rs:919`, `bin/syndesi_ui_v2.rs:1020`, `bindings/kleis_p2p.rs:33` |
| sha256 | `crypto::sha256` (18 attributed) | `fn sha256_hex` `bindings/kv_credential.rs:40`; `sha256_file`/`sha256_bytes` `sangdb/mod.rs:5928,5943` |
| JSON escaping | — | `json_escape` `sangdb/http.rs:116`, `bin/syndesi_ui/fleet.rs:929`; `escape_json` `ontology/mod.rs:1354` |
| wall-clock seconds | — | `now_secs` `appfold.rs:3560`, `domain.rs:767`, `techdocs/mod.rs:616`, `compliance.rs:25`, `httpsrv.rs:5123`; `now_ms` `domain.rs:2675`; `now_millis` `ontology/store/mod.rs:89`; `unix_now` `context.rs:479` |

## 12. Primitive extraction recommendations (with proof boundaries)

Candidate shared primitives, based on attributed call sites plus source inspection:

1. **Content-addressed chunk/version store** — `ChunkStore::{put_chunk,get_chunk}`,
   `find_boundaries`, `chunk`, `MissingChunks`, `Diff`, `Versioned::{pin,write}`, `Lorelake`.
   Boundary: these currently sit on `Table`; no proof they match any hashtrinity primitive.
2. **Inverted + ANN index** — `PostingIndex`, `KdTree`, `LshIndex`, `IndexSegment`.
3. **Query / exec** — `SqlQuery`, `SqlExec`, `sql_lex`, `sql_plan_of_tokens`, `sql_query_rows`,
   `sql_run`, `sql_exec`.
4. **Record / update / log** — `SqlTable`/`SqlValue`, `Log`/`Law`/`ConsumerGroup`, `ingest`,
   `fold_group`, `group_sum`.
5. **Scheduling / capability** — runtime `Fiber` registry (`provide`, `provide_as`, `revoke`,
   `provider_of`, `epoch_for`, `is_declared`, `would_violate`), `Cluster::{preduce,encircled,
   pmap_at,hedged_pmap,pipelined_pmap}`, hashtrinity `par::Pool::{global,dispatch}` / `SendPtr`.
6. **Capability / ACL** — `SangDbMayActChecker`, `reachable`, `security.rs`.

Proof boundary for all six: a shared name and a static call site are not a shared contract.
Promotion requires end-to-end verification. (The stricter crypto requirement below is specific to
crypto and is **not** imposed on these non-crypto primitives.)

## 13. Crypto — deferred (not excluded)

Crypto is deferred until **everything else is mature**; at that point crypto adoption requires
**full end-to-end formal verification AND independent external analysis**. It is not permanently
excluded. Current static evidence: 136 attributed `crypto::*` sites plus 1
`hashtrinity_crypto::ed25519_pubkey`, concentrated in `bindings` (61) and `transport` (33), plus
`httpsrv`, `sangdb`, `domain`, `bin`. The census makes no correctness or side-channel claim.

## 14. Limitations and parse coverage

- 429/429 files parsed, 0 errors; `source_location` is `src` 251 / `src_bin` 58 / `examples` 108 /
  `tests` 12; `reachability` always `not_determined` — a file may not be compiled.
- Attributed is a syntactic count (1580/13178), not a lower bound or proof. 11428 candidates,
  11382 with no receiver type; 49077 other method calls aggregated.
- **Macro expansion absent:** 13818 unexpanded macro invocations, 24 `macro_rules!` defs,
  58 derives (Term 47, Error 7, Collapse 4), 3 `path` attribute macros, 2 `bitflags!` invocations.
- **Inventory pollution** (test structs / associated types / traits / generic params) still yields
  the 102 `Engine`-style ambiguous flags.
- **Non-compiled files walked:** the six `src/native_lib/*` orphans contribute 368 candidates;
  their 19 `CoordMap` candidates are dead copies and are not added to live misses.
- `include!` chains (`lib.rs` → `native_lib.rs`) are not followed; parent imports are not propagated
  into child files.
- Const receivers and local aliases: aliases and `&mut` are now attributed; const receivers are
  heuristic (31) or candidate (2 `ZBASE32`).
- `cfg` recorded verbatim, not evaluated; function-pointer references (`let f = Table::get;`) are
  not calls and are not recorded.
