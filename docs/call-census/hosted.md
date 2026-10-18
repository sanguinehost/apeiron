# Hosted Hashtrinity call census — 21 remote consumers (tool 0.2 / schema 2)

AST-based, syntax-level census of 21 hosted repositories against the canonical
`hashtrinity` source, using `call-census 0.2.0` (schema 2). It counts call
expressions only; imports, re-exports, types, derives and macro sites are
reported in their own columns and are **not** calls. It performs no name
resolution and no type inference.

## 1. Acquisition and canonical link

| Item | Value |
|---|---|
| Canonical source (target) | GitLab `twn/RnD/fold` @ `4cc14471ae24f3f204fb0b0e91736ef75c909358`, branch `main`, 2026-09-09 |
| Canonical snapshot path | `/tmp/opencode/fold-call-census-20260911` |
| Target roots passed | `hashtrinity/`, `hashtrinity/hashtrinity_derive/` |
| Consumer cohort | `/tmp/opencode/hosted-consumers-20260911` (21 repos) |
| Census output (raw) | `/tmp/opencode/census-v2/hosted/<namespace>-<repo>/` |
| Tool | `/tmp/opencode/call-census-target/release/call-census` (`call-census 0.2.0`, schema 2, syn 2.0.119, proc-macro2 1.0.107) |
| Apeiron revision | `0b786453cc9ada1bd49fb7cb8b0d2efbe4fd85ae` (working tree had pre-existing edits; tool source was not edited) |

Per-repo full SHAs are in `data/hosted-repos.tsv` and on every row of
`data/hosted-calls.tsv`. **Coverage honesty:** this is a selected cohort of 21
hosted repos — **not all 427 repos**. GitLab `twn/RnD` siblings (`webfold`,
`chunkfold`, `transcode`, `pdffold`, `textfold`, `cerialize`, `qwalk`, `song`,
`talon`) and other namespaces were not deep-censused. Absence here is not
evidence of absence of calls.

## 2. Method and semantics (v0.2)

- **syntactically attributed** (`high`/`medium`) = a callee path rooted at a
  configured target crate, or a method whose receiver type is syntactically known
  to be a target type. This counts **source syntax**, not executed calls; it is
  **not** a proof, a completeness claim, a "lower bound" or a "floor".
- **unattributed candidates** = method calls whose receiver type is unknown; a
  work queue, never attributed.
- **operation_kind** separates `free_fn`, `associated_fn`, `method` and
  `unclassified`. Enum-variant constructors are `unclassified` and are **not**
  counted as functions or methods.
- **source_location** (`src`, `src_bin`, `tests`, `examples`, `other`) is where
  the text lives. **reachability** is always `not_determined`: no module graph is
  built and no file is asserted to be compiled.
- `cfg` predicates are recorded verbatim and **not evaluated**. 104 attributed
  rows carry `cfg(test)`, plus feature/platform cfgs. The column captures
  item/module/impl cfg, **not block-level** cfg: `webtrainer`'s crypto calls sit
  behind a block-level `#[cfg(not(target_os = "solana"))]` and show empty `cfg`,
  so that gate is known only from reading the source. An attributed call may
  therefore not be reachable in a given build.
- **0 attributed calls is not "no runtime link".** A repo may depend on
  `hashtrinity` transitively, re-export it, or call it through macros the tool
  does not expand.

## 3. Per-repo results (v0.2)

Files parsed 1,389; recorded call expressions 9,364; **syntactically attributed
1,311** (all `high`; by target crate `hashtrinity` 1,269, `hashtrinity_crypto`
42). Unattributed candidates 7,992; ambiguous 3; heuristic 45; dynamic 13; other
method calls 92,278; target imports 316; re-exports 7. Nine of 21 repos have at
least one attributed call.

| repo | SHA | files | calls | attr | unattr | amb | heur | dyn | imp | reexp | callees | callers | syms | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ptyktos/syndesi | 43fd837f437a | 176 | 6486 | 839 | 5617 | 0 | 18 | 12 | 95 | 0 | 43 | 346 | 43 | direct: Table+syntax+`hashtrinity::crypto` |
| Tasmanian-Cloud/bedrock | b043bbc8edee | 100 | 1605 | 198 | 1402 | 0 | 5 | 0 | 79 | 4 | 44 | 126 | 44 | direct: Table+syntax+old `hashtrinity_crypto` |
| Ptyktos/pvefold | 9ae54103b4d3 | 49 | 428 | 95 | 311 | 0 | 21 | 1 | 50 | 0 | 14 | 49 | 14 | direct: Table+codec::syntax |
| Tasmanian-Cloud/Galvanize | eb6b6eb9b6a4 | 70 | 339 | 95 | 244 | 0 | 0 | 0 | 41 | 0 | 13 | 43 | 13 | direct: codec::syntax+Table |
| Tasmanian-Cloud/saga | e4fe92208ef0 | 48 | 111 | 32 | 79 | 0 | 0 | 0 | 19 | 0 | 8 | 26 | 8 | direct: codec::syntax only |
| Tasmanian-Cloud/flexprice-sdk | 0f44f74e996f | 16 | 73 | 26 | 47 | 0 | 0 | 0 | 20 | 3 | 16 | 16 | 16 | direct: Table+syntax+Error |
| Ptyktos/autofold-standalone | 9f83cdc06eae | 22 | 265 | 21 | 243 | 0 | 1 | 0 | 9 | 0 | 11 | 14 | 11 | direct; **vendors divergent copy** |
| Tasmanian-Cloud/flexprice-collector | 9cd8cb80fd8c | 1 | 12 | 3 | 9 | 0 | 0 | 0 | 3 | 0 | 3 | 2 | 3 | **Error-only** + old crypto |
| sanguinehost/webtrainer | 30da6b9879a2 | 337 | 18 | 2 | 16 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 1 | old `hashtrinity_crypto` only (cfg-gated) |
| Tasmanian-Cloud/hosting-runtime | d8c8d5db763e | 145 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | declares dep; no member attributed |
| Tasmanian-Cloud/sauthy | b0ad9eebf891 | 319 | 17 | 0 | 15 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | transitive via `transcode` |
| Tasmanian-Cloud/flexprice-sidecar | 3be58427646f | 1 | 7 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | transitive via `bedrock` |
| gitlab/vocalize-rs | 3b137d5f1067 | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | transitive via `transcode` |
| Ptyktos/pii-svc | 49f780ea3b8b | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **local copy of primitive** |
| gitlab/sdirstat | 6dec1ebac7ac | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **local copy of primitive** |
| Ptyktos/aim | e33d866f1565 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no direct dep/attributed found |
| Ptyktos/agentvfs | 6a61577ec344 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no direct dep/attributed found |
| Ptyktos/corpusforge | f8fcaa5b142b | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no direct dep/attributed found |
| Ptyktos/lexigraph | cff565dffe6f | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no direct dep/attributed found |
| sanguinehost/webtrainer-core | f634c50504a8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no direct dep/attributed found |
| gitlab/sanguine-datastores | ac7e43fa8a80 | 6 | 3 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | doc cite only; no `Cargo.toml` |

The 7,992 unattributed candidates are dominated by non-target methods (`map`,
`collect`, `push`, `get`, `iter`, `len`, `is_empty`); they are a recall gap, not
attribution. "No direct dep/attributed found" is weaker than "no relation": those
repos may still link `hashtrinity` transitively (see §7).

## 4. Attributed calls by operation kind

| operation_kind | sites |
|---|---|
| method | 717 |
| associated_fn | 249 |
| free_fn | 132 |
| unclassified | 213 |
| **total** | **1,311** |

The 213 `unclassified` split into **167 enum-variant constructors**
(`SyntaxTerm::Bu` 59, `::B` 46, `::Fourth` 13, `Atom::Str` 24, `::Num` 13,
`::Bool` 12) and **46 unmatched associated functions** (`SyntaxError::new` 44,
`ProtoRegistry::new` 1, `ProtoSchema::default` 1). Constructors are therefore
reported separately from functions and methods.

By `source_location` (attributed only): `src` 1,210, `src_bin` 76, `other` 13,
`tests` 12 — totals 1,311. `reachability` is `not_determined` for all rows.

## 5. Per-callee: static sites vs distinct callers vs repos

Caller frequency and site counts are now separate columns; no figure below is a
caller count. Full 77 callees in `data/hosted-callee-summary.tsv`.

| callee | kind | static sites | distinct callers | repos |
|---|---|---|---|---|
| `Table::new` | associated_fn | 228 | 128 | 5 |
| `Table::put` | method | 171 | 119 | 6 |
| `Table::get` | method | 159 | 103 | 6 |
| `Table::iter` | method | 158 | 92 | 5 |
| `SyntaxTerm::Bu` | constructor | 59 | 47 | 6 |
| `SyntaxTerm::B` | constructor | 46 | 25 | 6 |
| `SyntaxError::new` | associated_fn | 44 | 16 | 3 |
| `Table::mark` | method | 32 | 24 | 1 |
| `Table::remove` | method | 30 | 27 | 3 |
| `codec::syntax::parse_surface` | free_fn | 26 | 26 | 5 |
| `Table::has` | method | 25 | 21 | 1 |
| `Table::len` | method | 24 | 19 | 4 |
| `Atom::Str` | constructor | 24 | 20 | 6 |
| `codec::syntax::render_surface` | free_fn | 19 | 17 | 5 |
| `Table::push` | method | 17 | 17 | 3 |
| `Table::insert_with` | method | 15 | 9 | 2 |
| `SyntaxTerm::get` | method | 14 | 7 | 1 |
| `Atom::Num` | constructor | 13 | 8 | 5 |

The prior hosted report conflated these: its "`Table::new` (128)" was the
distinct-caller count while its "190×" was a v1 site count. The v0.2 site count is
**228**.

## 6. Genuine callers — concrete caller → callee

- **syndesi** (839): `Table` + `codec::syntax` + `hashtrinity::crypto`
  (`audit_id`, `content_address`, `content_hash_u64`, `KeyVault::new`,
  `op_digest`, `term_bit` → `blake3`; `sign_result` → `hmac_sha256`).
- **bedrock** (198): `SyntaxError::new`, `parse_surface`, `CoordMap::coordinate`,
  `Table`; plus old `hashtrinity_crypto` (`Ed25519Signer`, `MembershipLog`,
  `ChunkStore::put_chunk`, `FileStore::put`, `Version::content_hash`,
  `derive_response_secrets`, `decrypt_response`/`encrypt_response`).
- **pvefold** (95): `Table`; `Term`/`Coord` (`impl hashtrinity::Coord for
  ApiMethod`, `foldapi/src/path.rs:29`).
- **Galvanize** (95): `SyntaxTerm` constructors, `parse_surface`,
  `render_surface`.
- **saga** (32): `SyntaxTerm` constructors, `render_surface`
  (`crates/runner/src/ansible.rs:307`).
- **flexprice-sdk** (26): `Properties`/`ListResponse` `to_term`/`from_term` over
  `Table<String, String>`.
- **autofold-standalone** (21): `Table` in `autofold/src/{artifact,trace,ledger}.rs`
  and `boostfold`.
- **flexprice-collector** (3): `hashtrinity::Error::new` in `err()` only, plus
  old crypto (`derive_secret`).
- **webtrainer** (2): `hashtrinity_crypto::sha256` in `sha256`/`sha256v` under
  `#[cfg(not(target_os = "solana"))]`.

## 7. Older API inventory gap (`hashtrinity-crypto`)

Canonical `hashtrinity/Cargo.toml` states the sub-crates were subsumed
(*"Subsumed crates: hashtrinity-crypto … One crate, one fold."*). The separate
`hashtrinity-crypto` crate is **absent** from the canonical snapshot, yet three
consumers declare a path dep `../../fold/hashtrinity-crypto`:

- `Tasmanian-Cloud/bedrock` — 38 attributed (35 crypto ops + 3 `to_hex`)
- `Tasmanian-Cloud/flexprice-collector` — 2 (1 crypto op + 1 `to_hex`)
- `sanguinehost/webtrainer/shared/core` — 2 (cfg-gated)

`Ptyktos/syndesi` instead uses the **current** subsumed module
`hashtrinity::crypto` (7: `blake3` 6, `hmac_sha256` 1). The old crate accounts
for 42 of 1,311 attributed calls; the split is the cohort's largest API-version
gap.

## 8. Vendored copy (autofold-standalone)

`Ptyktos/autofold-standalone` embeds its own `hashtrinity/` as a workspace member
(`hashtrinity = { path = "hashtrinity" }`); it is **not an external dependent**.
The copy is divergent and was excluded from counts via `--skip hashtrinity`:

| file | canonical | autofold vendored |
|---|---|---|
| `src/lib.rs` | `f8c8fd12…` | `7db27c2c…` |
| `Cargo.toml` | `f78f9383…` | `74b376cc…` |
| `src/codec.rs` | present | **absent** |
| `src/crypto/keccak.rs` | present | **absent** |
| `src/*.rs` | 25 files | 13 files |

## 9. Duplicated primitive functions (validated exact usage)

Two near-verbatim copies of the same `HashTrinity` operation over
`std::collections`, neither depending on the crate. Usage validated by call
expression, not by line count (`data/hosted-duplicates.tsv`).

- **`Ptyktos/pii-svc` `crates/pii-core/src/hash.rs`** — `insert_with`,
  `insert_with_ord`, `set_merge`, `first_merge`, `map_merge`, `acc_merge_add`,
  `acc_merge_max`, `acc_merge_min`, `graph_merge`. **Actual calls: 10**
  `insert_with_ord(...)` expressions in `detect.rs`; `first_merge` is passed as a
  **function value 10×** (not called); 1 `use` import. The B/U `acc_merge_*`,
  `set_merge`, `map_merge`, `graph_merge` have **0 call expressions**.
- **`gitlab/sdirstat` `src/hash.rs`** — same scalar set (minus `acc_merge_min`,
  with `link_merge`), plus `fold`, `struct Acc` (B/U byte accumulator) and
  `struct Csr` (Graph face). **Actual calls: 5** (`Acc::with_capacity` 3,
  `Csr::from_parents` 2); **14 type-reference tokens** (`hash::Acc` 9,
  `hash::Csr` 5). Scalar merge fns have **0 call expressions**. The prior "12
  uses" was an `rg` line count that included 2 comments.
- **`Ptyktos/syndesi` `src/meta.rs:6381`** — local `fn insert_with` delegating to
  `hashtrinity::Table`; a thin wrapper, not a reimplementation.

## 10. Crypto (deferred, exact groups)

Crypto is deferred until everything else is mature; it then requires full
end-to-end formal verification **and** independent external analysis. The crypto
standard must not be attached to non-crypto callees.

- **45 cryptographic operations**: `blake3` 13 (7 old-crate + 6
  `hashtrinity::crypto`), `sha256` 7, `ed25519_pubkey` 6, `ed25519_sign` 4,
  `hmac_sha256` 4, `ed25519_verify` 3, `hkdf_expand_sha256` 3,
  `hkdf_extract_sha256` 2, `aes_gcm` 2, `ct_eq` 1.
- **4 `to_hex` encoding-helper sites** — `to_hex` is an encoding helper **housed
  in** crypto, **not** a cryptographic operation.
- **Crypto-housed total: 49** attributed calls (42 `hashtrinity_crypto` + 7
  `hashtrinity::crypto`). Non-crypto attributed: **1,262** (1,311 − 49).

## 11. Program purpose and primitive classification

| repo | purpose | classification |
|---|---|---|
| syndesi | textfold TUI coding agent | **reusable primitive consumer** — `Table` + `codec::syntax` (+ crypto deferred) |
| bedrock | runtime foundation (HTTP/router/proxy) | **reusable primitive consumer** — `Table` + `codec::syntax` + old crypto |
| pvefold | Proxmox VE replacement workspace | **reusable primitive consumer** — `Table` + `Term`/`Coord` |
| Galvanize | IaC umbrella | **reusable primitive consumer** — `codec::syntax` + `Table` |
| flexprice-sdk | billing API SDK | **wrapper** — models derive `Term`; thin `Table` use |
| saga | hosting orchestration runner | **wrapper** — serializes provider payloads via `codec::syntax` |
| autofold-standalone | distill LLM behavior to small model | **reusable primitive consumer** (vendored copy) |
| flexprice-collector | event collector/batcher | **Error-only** core consumer + old crypto |
| webtrainer | decentralized training network | old-crypto only (cfg-gated) |
| pii-svc | PII detection/redaction | **local reimplementation** (app policy over copied primitive) |
| sdirstat | disk-usage analyzer | **local reimplementation** (`Acc`/`Csr` carriers) |
| hosting-runtime | TasCloud hosting services | declares dep; no member attributed |
| sauthy / flexprice-sidecar / vocalize-rs | SSO / sidecar / speech-to-text | **transitive only**; no direct target use found |
| aim / agentvfs / corpusforge / lexigraph / sanguine-datastores / webtrainer-core | identity, VFS, corpus, code-understanding, data-plane docs, gradient exchange | no direct dep/attributed found |

**Reusable primitive candidates** (cross-program, actual shared callers):
`Table` (`new/put/get/iter/len/is_empty/push/neighbors/add/count/mark/has/remove/
insert_with/accumulate/coordinate_with_capacity`), `Coord`/`CoordMap`, `FoldExt`,
and `codec::syntax` (`Term`/`SyntaxTerm`/`Atom`/`AsTerm`/`Surface`/`parse_surface`/
`render_surface`/`term_*`/`SyntaxError`), plus `Error`.

**Wrapper / app policy** (not primitive): `flexprice-sdk`'s `Properties`/
`ListResponse` derives, `saga`'s provider-payload serialization,
`flexprice-collector`'s `CollectorError`, and per-app `to_term`/`from_term`.
`pii-svc`'s and `sdirstat`'s local `hash.rs` are app-local copies that should
migrate to the primitive or be justified as carriers (`Acc`/`Csr`).

## 12. Corrections to the prior hosted report

- v0.1 counts replaced by v0.2; "resolved lower bound" wording dropped for
  "syntactically attributed" (source syntax, not proof/floor).
- Caller frequency and site counts separated: `Table::new` is **228 sites / 128
  callers**, not a single "128" figure, and "190×" was a v1 artifact.
- Enum-variant constructors (`SyntaxTerm::Bu`, `Atom::Str`, …) separated from
  functions and methods.
- `to_hex` separated from cryptographic operations (encoding helper).
- "No relation" softened: `vocalize-rs`/`sauthy` (via `transcode`) and
  `flexprice-sidecar` (via `bedrock`) are transitive; others are "no direct
  dep/attributed found", not proven unconnected.
- `pii-svc`/`sdirstat` usage validated as call expressions (10 / 5) rather than
  `rg` line counts (11 / 12).
- 0 attributed calls explicitly does **not** mean "no runtime link".

## 13. Unknowns and limitations

- **7,992 unattributed candidates** remain unadjudicated; the
  `get/iter/len/put/push` subset is the recall risk and needs manual receiver
  tracing. `syndesi` alone contributes 5,617.
- 3 ambiguous, 45 heuristic, 13 dynamic-dispatch method calls.
- No macro expansion, no name resolution, no reachability; `cfg` recorded but not
  evaluated (`webtrainer`'s crypto sits behind `cfg(not(target_os = "solana"))`).
- Cohort is 21 selected repos, not the full hosted surface.

## 14. Arithmetic verification

- `hosted-calls.tsv`: **1,311** data rows + header = sum of `attributed_calls`
  over the 21 `summary.json` files = **1,311**.
- Operation kinds: 717 + 249 + 132 + 213 = **1,311**; unclassified 167 + 46 =
  **213**.
- Source location: 1,210 + 76 + 13 + 12 = **1,311**.
- Crypto: 45 ops + 4 `to_hex` = **49**; target crate 42 + 7 = **49**;
  non-crypto 1,311 − 49 = **1,262**.
- Global: files 1,389; call expressions 9,364; attributed 1,311; unattributed
  7,992; ambiguous 3; heuristic 45; dynamic 13; other method calls 92,278;
  imports 316; re-exports 7.

## 15. Files

- `docs/call-census/hosted.md` (this report)
- `docs/call-census/data/hosted-repos.tsv` (per-repo counts, full SHA, purpose)
- `docs/call-census/data/hosted-calls.tsv` (1,311 attributed callsites)
- `docs/call-census/data/hosted-callee-summary.tsv` (77 callees: sites/callers/repos)
- `docs/call-census/data/hosted-duplicates.tsv` (validated local-copy usage)
- Raw per-repo census: `/tmp/opencode/census-v2/hosted/<namespace>-<repo>/`
