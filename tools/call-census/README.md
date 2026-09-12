# call-census

A reproducible, AST-based call-site census for Rust consumers of a target crate
(default: `hashtrinity`). It reports, lexically:

- which target functions/methods are called, where, and in which caller;
- which imports, re-exports, types, variants and derives come from the target;
- which method calls are **candidates** that a human must resolve, and why they
  were not attributed;
- which calls are **labelled heuristics** (method name provided by an imported
  target trait, receiver type unknown).

It is **not** a compiler. It uses `syn` for syntax and `proc-macro2`
`span-locations` for exact line/column positions. There is no name resolution,
no type inference, no macro expansion and no trait selection. Every record says
what it knows and how it knows it.

**Schema version 2 / tool 0.2.0.** Reports from earlier versions use different
field names and different confidence semantics (`resolved` rather than
`attributed`) and MUST be regenerated. Do not merge or compare counts across
schema versions; `meta.json` carries `schema_version`, `tool_version` and a
`compatibility_note`.

## Build and test

The crate is its own workspace root and is pinned to crates already in the local
registry cache, so it builds offline:

```sh
cd tools/call-census
cargo build --offline --release
cargo test --offline
```

`cargo test` runs `tests/fixture.rs`, which asserts, among others:

- comments/strings are not counted;
- alias, grouped and glob imports are tracked; re-exports recorded;
- two calls on one line get distinct stable ids;
- `Arc::new(x)` shadowing invalidates a stale `Table` binding;
- a local type named `Table` is not read as the target `Table`;
- `&mut Table` receivers attribute;
- a non-target `.get` is not a candidate; an unknown-receiver `.get` is;
- enum/tuple-struct constructors are separated from free/associated functions;
- trait-provided methods (`FoldExt::index_by`, …) are inventoried and recorded
  as labelled heuristics;
- local type aliases to target types are attributed with an alias basis;
- module/impl/fn `cfg` predicates propagate; `cfg(test)` is test context;
- test-only target items are excluded from the inventory.

## Run

```sh
target/release/call-census \
  --root  <consumer-snapshot> \
  --output <report-dir> \
  --target-src <hashtrinity-crate-dir> \
  --target-src <hashtrinity/hashtrinity_derive> \
  --target-crate hashtrinity \
  --target-crate hashtrinity_crypto \
  --target-crate hashtrinity_codegen
```

- `--root` is the whole consumer folder snapshot. It is walked recursively;
  `target/`, `.git/`, `node_modules/`, `vendor/`, `build/`, `dist/`, `.claude/`
  and hidden directories are skipped.
- `--target-src` should point at a **crate directory** (the `src/` subtree is
  used). It adds the package name and its symbols to the inventory. Passing the
  `hashtrinity_derive` crate is what lets an unqualified `#[derive(Term)]` be
  recognised.
- `--target-crate` / `--target-type` / `--target-method` extend or replace the
  built-in defaults.

For an apeiron snapshot, use `--target-crate apeiron --target-crate apeiron_derive`.

## Output

| file | contents |
| --- | --- |
| `meta.json` | schema/tool versions, root, git revision, tool source + binary SHA-256, limitations |
| `files.json` | every `.rs` file: sha256, bytes, lines, crate, source location, parsed/parse error, reachability |
| `inventory.json` | target crates, types, traits, trait methods, methods, free functions, macros, variants, tuple structs, field types |
| `records.json` | every call/method record with evidence |
| `calls.tsv` | syntactically attributed calls (confidence high/medium) |
| `unattributed_methods.tsv` | candidate / ambiguous / dynamic / heuristic method calls |
| `imports.json` / `imports.tsv` | target imports, globs and re-exports |
| `macros.json` / `macros.tsv` | target macro sites, derives, attribute macros, `macro_rules!` defs |
| `summary.json` / `summary.md` | aggregates by crate, source location, operation kind, subsystem, symbol, category |

Stable identifiers are `call:<file>:<line>:<col>-<end_line>:<end_col>`,
`import:<file>:<line>:<col>:<path>`, `derive:...:<name>`, `macro:...`,
`macro_rules:...:<name>`. The same static call expression always gets the same
id; two calls on one line get different ids because their columns differ.

## Confidence and operation kind

`confidence`:

- `high` — the callee path is rooted at a target crate, or the receiver's type is
  written down as an import, parameter, field, `impl` self type, narrow
  constructor binding, or a local alias whose right-hand side attributes.
- `medium` — a target glob import plus a name from the target inventory.
- `heuristic` — the method name is provided by a target trait imported in scope
  but the receiver type is unknown. Counted separately; still `candidate: true`.
- `ambiguous` — a local type alias whose target is unclear, or a name reachable
  from both a target and a non-target glob.
- `unattributed` — a candidate; never attributed.
- `dynamic` — trait object / generic bound dispatch.

`operation_kind` for call records is one of `free_fn`, `associated_fn`,
`constructor`, `unclassified`. Constructors come from the target inventory
(enum variants, tuple structs), never from capitalisation. Method records use
`method`.

`source_location` (`src`, `src_bin`, `tests`, `benches`, `examples`, `build`,
`other`) says where the text lives. `reachability` is always `not_determined`:
the tool never asserts that a file is compiled or reachable.

## What "attributed" means

`summary.attributed_calls` is a count of **source syntax**, not executed calls,
not a mathematically assured lower bound, and not a completeness proof. Macro
bodies are not expanded; generic instantiations, trait objects, `Deref` chains
and untyped receivers are not attributed. The summary states this explicitly.

## Observed attribution gaps

The portfolio audit found concrete residual limits in 0.2.0:

- Smart-pointer normalization can attribute `Rc<RefCell<Table>>::clone` to
  `Table::clone`. Two such sites are flagged in the consolidated reviewed ledger;
  `high` confidence is not a guarantee of correctness.
- Deeply qualified or re-exported enum constructors can remain `unclassified`.
  `audit_reports.py` joins their suffixes to the inventory and retains both the
  scanner kind and the joined kind; the latter is still a syntax-level join.
- Block-level cfg attributes are not propagated. Empty `cfg` does not prove
  unconditional compilation.
- Cross-file imports, macro expansion, wrapper dispatch and reachability still
  require the consumer's actual build/type information.

See `docs/call-census/README.md` at the repository root. The postprocessor checks
published site identities, spans, symbols and totals against the raw schema-2
evidence, marks known false positives, and emits a reproducible run manifest.

## How to resolve candidates

`unattributed_methods.tsv` is the work queue. It carries `file`, `line`, `column`,
`caller`, `method`, `receiver_text`, `receiver_type`, `candidate`, `dynamic` and
an `attribution_basis`. `summary.md` lists the top 50 candidate method names.
Use the receiver text and the caller's file to decide by hand; the tool will not
label every `.get` as `Table`.

## Why not rust-analyzer

`rust-analyzer` was checked: it is not installed as a rustup component on this
machine (`rustup` reports `Unknown binary 'rust-analyzer'`), and the tool must
run on a snapshot without a full compiler/toolchain install. `syn` parsing is
robust, fast (≈240k LOC in ≈7s) and dependency-light; the cost is that type
resolution is approximated and reported as such.

## Reproducibility

`meta.json` records:

- `tool_source_sha256` — SHA-256 over the tool's `Cargo.toml` + `src/` + `tests/`
  (sorted), so the exact scanner logic is identifiable;
- `tool_binary_sha256` — SHA-256 of the running executable;
- `revision` — git HEAD/branch/describe and dirty-line count of the walked tree;
- per-file `sha256` in `files.json`.
