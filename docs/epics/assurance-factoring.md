# assurance factoring

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

Merged epic record: `intent/assurance-factoring.md`, `spec/assurance-factoring.md`, `plan/assurance-factoring.md` (one file per epic; see `docs/README.md`).

## Intent — assurance factoring

- Owner: Astra architecture; implementation and closure: lead.
- Source: `262bb0a`; accepted dirty baseline `target/open-areas/baseline.json`, SHA-256 `818593f0b9ebbb33e2f020b1345e8001e0a8778665321b2af2996ae09962ff52`.
- Status: implementation under the [factoring evidence](../evidence/assurance-factoring.md); all seven size breaches pass; proof integration results remain explicit there.
- [Spec](assurance-factoring.md); [plan](assurance-factoring.md).

### Problem

`python3 tools/check-assurance.py --check` reports five oversized files and two overcrowded directories. Its limits remain 1,500 physical lines per owned source file, 12 immediate source files per directory, and 5,000 local trusted-checking lines. The observed trusted total is 3,276. Generated Slang source remains counted.

| Breach | Observed | Required factoring |
|---|---:|---|
| `examples/reg1/acceptance.rs` | 1,799 lines | Reporting and registry tests become child modules. |
| `proof/corpus/Slang/Generated/Table.lean` | 14,414 lines | Dependency-ordered declaration shards behind the existing import root. |
| `src/table.rs` | 1,678 lines | Existing free prepared-region functions become a textual include. |
| `tools/slang/export.py` | 2,299 lines | Separate pure decoder and structural witness emitter. |
| `tools/slang/spec_check.py` | 1,562 lines | Separate frozen requirement data and fixtures. |
| `proof/corpus/Slang/Source/` | 14 files | Group byte/lexing audit modules in `Source/Audit/`. |
| `tests/` | 15 files | Three existing Cargo integration targets use directory entry points. |

The initial isolated-checkout check also reported the missing empty `benches/` root. After the lead restored that declared directory, the repeated checker reports exactly seven source breaches, exit 1. No checker exemption changed.

### Rejected alternatives

| Alternative | Reason |
|---|---|
| Exempt generated Slang, alter limits/extensions, move outside owned roots | Removes the rejecting requirement instead of satisfying it. |
| Minify terms, comments or tests | Does not establish an architectural boundary and risks losing evidence. |
| One generator rewrite plus all output files | Exceeds three files and prevents reviewable intermediate freshness. |
| Move Rust methods into a new namespace | Changes extracted identities unnecessarily. |
| Drop integration targets or audit roots | Reduces acceptance coverage. |
| Update Song/canonical pins to simplify regeneration | Changes the source identity; requires separate authorization and gates. |

### Completion condition

Every task gate and control in the plan passes; the unchanged assurance checker and its self-tests pass; all Rust/extraction/Lean barrier commands pass on one recorded source/tool identity. Generated output is reproducible, audit coverage is unchanged, and no theorem premise, admissible input, rejection criterion or benchmark threshold changes.

```sh
python3 tools/check-assurance.py --check
python3 tools/check-assurance.py --self-test
```

These commands close size acceptance only after the semantic barrier; size acceptance does not close storage, W7 composition, Slang source binding or REG-1/G1.

### Invariants and evidence

- Preserve actual `#[path]` Rust extraction, `table` item identities, callback order, checked refusal, payload initialization and public API.
- Preserve all source bytes, source digests, source-part names, append-tree order, theorem statements and the Slang empty axiom cone.
- Preserve the acceptance registry, 1.0 interval ceiling, reporting semantics and all rejecting controls.
- Preserve Cargo target names, test functions and assertions.
- Architecture is `anchor`. Rust execution is `tested` only after its gates. Existing Lean declarations remain proof candidates until bounded rebuild/audits pass. No speedup is claimed.

### Trust boundary

The exporter remains an untrusted candidate renderer. Freshness establishes byte equality to its output, not source semantics. Lean checks terms; the empty-cone runner checks registered declarations. Rust tests do not establish universal refinement, allocator/executor realization, rustc/LLVM correctness, ABI, GPU/SPIR-V or unsafe-container correctness. Existing open premises remain open.

## Specification — assurance factoring

Current result: all seven size breaches pass the unchanged checker; eleven shard migrations and
both audit relocations pass their focused gates/controls. Old paths below identify completed
migration ownership, not live consumers. [Integration evidence](../evidence/assurance-factoring.md)
records final proof checks and remaining programme obligations.


- Owner: Astra architecture; proof sufficiency and attribution: Sol before Lean relocation.
- Source and dirty identity: [intent](assurance-factoring.md).
- Status: proposed, `anchor`; [plan](assurance-factoring.md).

### Contract

| Population | Required equality | Failure semantics |
|---|---|---|
| Rust table | Same tokens in each moved free function; same names in module `table`; actual wrappers still call these bodies | Same `StorageError`, callback effects and transaction boundary |
| REG-1 | Same registry and evaluator decisions; same report fields and control dispatch | Same exit codes for missing, failing, unauthorized and mismatched pairs |
| Slang exporter | Same selected Song closure, bytes, paths, SHA-256, decode JSON and exported declarations | Cycles, missing imports, invalid UTF-8 and unexpected closure still reject |
| Slang proof files | Same public declaration names/types; same terms except module-private helper identity | Same bounded elaboration and empty-cone rejection |
| Spec checker | Same constants, fixtures, validators and complete mutation set | Same positive/negative exits and diagnostics |
| Integration tests | Same Cargo target names, test names and assertions | Same test failure conditions |

No new precondition or reduced input domain is permitted. This is source organization, not a change to `Apeiron.Table.{HMap,get,insertWith,build}`. `proof/extract/lib.rs` includes `src/table.rs`; `proof/regen.sh` has an explicit source digest list. Add every production include to that list before accepting its extraction.

### Rust boundaries

`src/table/prepared.rs` contains the existing free functions from `reserve_plan_regions` through `build_prepared_shard`, including their comments, and is included textually at their old location. The observed span begins with the documentation following `include!("table/locator.rs")` and ends before the `Table` documentation. Do not split method `impl` blocks or create another implementation. This removes about 200 lines; require the resulting root to be at most 1,500. Existing nine immediate files become ten. Future additions need fresh count checks; the small root margin is not a waiver.

`examples/reg1/acceptance/report.rs` contains `print_registration` through `print_ratio_summary`, including `print_pair_table`, `print_row`, `interval_text` and `print_headline`. Retain parent-facing exports and use `pub(super)` only where needed. `acceptance/registry_tests.rs` contains the existing test module body. Parent tests keep their existing module name. Parent declarations and control functions remain in place; explicit `adapters/serial.rs` routing is unchanged. About 460 lines leave the parent; no new file appears directly in `examples/reg1/`.

Move `tests/native_address.rs`, `tests/native_locator.rs`, and `tests/parallel.rs` byte-for-byte to `tests/native_address/main.rs`, `tests/native_locator/main.rs`, and `tests/parallel/main.rs`. Cargo auto-discovery retains these integration target names. Confirm target enumeration and test lists before/after; no `autotests = false` or selective target omission. Root source count becomes 12; each new directory contains one source file.

### Python boundaries

Use counted modules under `tools/slang/export_parts/`:

1. `decode.py`: `_TOKEN_KINDS`, selected-name constants, `_STEM_PATHS`, `_DecodeError`, lexical/parser/validation functions and `_token_to_json` (the current decoder section before `_source_bytes`). It imports only standard-library dependencies, never the CLI.
2. `scan.py`: structural scan constants, render helpers, `_StructuralScan`, packing and witness-module rendering. It imports the selected-name constants from `decode.py`. Its public witness function accepts the source-data provider and repository root explicitly; it must not import `export.py`.
3. `table.py`: source declaration grouping and shard presentation, used only after the generator-interface gate. Accept source closure/revision and presentation helpers explicitly, with no reverse import to the CLI.

`export.py` keeps its public `source_data`, `render_source_witness`, `render_utf8_witness`, source resolution, default paths and CLI behavior. Explicitly re-export moved names still used by callers; do not use `globals().update`, executable strings or circular imports. The UTF-8 emitter reuses packing through an explicit import. Moving the decoder alone leaves roughly 1,600 lines; the second extraction is required. After both extractions, the root and all helpers must be under 1,500; the decoder/scan spans are each below 900 before small interface additions. The new directory has at most three `.py` files; a package initializer, if necessary, is a separate owned path and must fit its task.

`tools/slang/spec_constants.py` owns the constant/fixture block from `OBLIGATIONS` through `STAGE_COLUMNS`, with its `re` import. Keep `ROOT`, `SONG`, `CORPUS`, path resolution and revision constants in `spec_check.py`; no helper-relative root discovery. Preserve explicit names used by validators and mutation constructors. Expected root size is under 1,300, helper under 400; inspect actual physical lines.

### Reproducible generated-table transition

The current `render` emits one string, while witness renderers already return output maps. Add `render_table_outputs` as an exact path-to-text map, with deterministic chunk layout and a persisted prefix count in `export_parts/table.py`. The CLI and freshness check consume the whole map. Initially the count is zero and the generated bytes are identical. The final count is eleven. Intermediate counts change presentation only and keep the existing import root operational.

Group complete dependency-ordered declaration blocks, including each part and its length theorem; never split a declaration, literal, UTF-8 code point or theorem. The observed table/cell/std body packed at 1,400 physical body lines yields eleven groups, with body sizes:

`1397, 1397, 1400, 1400, 1395, 1395, 1395, 1395, 1398, 1398, 429`.

These are source-text packing observations, not generated-file or Lean acceptance. Full rendered files must independently be at most 1,500 lines, including imports, helper, namespace and footer. Reject an indivisible oversized block or more than twelve direct shards; do not compress it. Frozen names are `proof/corpus/Slang/Generated/Table/Chunk01.lean` through `Chunk11.lean`; the directory contains eleven files.

Each chunk imports `TableSupport` and the previous chunk where applicable, uses `namespace Slang.Generated.Table`, and contains its complete block group. `Table.lean` imports the last installed chunk and retains the unsharded suffix and final closure definitions. On the last step it becomes the small aggregation/metadata root. Dependencies are only toward preceding chunks; no chunk imports `Slang.Generated.Table` itself. Existing downstream imports remain valid through this root.

Repeat the private `sourceByte` helper locally where needed, with its existing `Nat.mod_lt` proof. Its Lean private name changes with its module; all public source-part names and terms must remain definitionally equal. Do not replace it with `Fin` numeral notation, introduce `propext`, or use whole-source `rfl`/`decide`/unbounded reduction. Keep the balanced append tree and source-part width 32 unchanged. Sol must adjudicate the private-helper relocation and declaration/type inventory before shard dispatch.

The existing tokenizer-witness registry explicitly scans `Generated/Table.lean`. Before moving any declarations, extend that stage with a registered child-import closure, rooted at `Slang.Generated.Table` and restricted to `Slang.Generated.Table.*`, so the same forbidden-token scan visits every imported shard. Keep every existing audit registration; a missing or cyclic child import rejects. This gate-first change owns the audit driver, registry and its synthetic controls separately from shard migrations. It avoids silently leaving newly moved unused forbidden declarations outside the scan.

Freshness compares the complete expected set, rejects missing/mutated outputs and stale numbered shards, and does not silently delete them. Alternate `--output` handling must place its shard family next to that output and never overwrite a second default tree. Controls run in a throwaway with the same selected Song revision/bytes. Compare the concatenated ordered public declaration blocks to the old monolith, modulo headers/imports/private helper, and compare decoded source JSON/digests exactly. The table source digests remain unchanged; generated file digests necessarily change.

`tools/slang/controls.json` does not regenerate these artifacts in its retained source-stage control. Do not pretend the existing source-stage gate becomes green through factoring; its unconditional source-binding target remains separate. Use dedicated freshness/declaration controls, bounded existing witness audits, and the full corpus barrier.

### Source audit relocation

Move the bodies of `Source/BytesAudit.lean` and `Source/LexingAudit.lean` into `Source/Audit/Bytes.lean` and `Source/Audit/Lexing.lean`. Keep namespaces `Slang.Source.BytesAudit` and `Slang.Source.LexingAudit` and every audit statement unchanged. First leave old import-only forwarding modules and update `tools/slang/zero_axioms.json` for the `bytes`/`lexing` stages: audit path, source files, declaration module/path and mutation anchors all resolve to the new body. Then update `proof/corpus/lakefile.lean` roots and remove the two old forwarding files in separate tasks. Do not leave shims at final acceptance: root count must become 12.

Live Lean searches found those old module names in their own audit bodies and Lake roots; rerun the complete consumer search before removal, including JSON, tools and active documentation. A newly found executable consumer requires its own sequenced task before deletion; it cannot be omitted from coverage.

### Identity, gates and limits

Record SHA-256 for all changed/moved/generated sources and tools; preserve Song/canonical pins. Source manifests for actual extraction must include the new Rust file. Prior profile source digests and theorem receipts remain historical; an uncommitted factoring change must not be attributed solely to HEAD, nor may a commit-bound profile be silently repinned. No current performance assertion follows.

Each implementation task has one gate command, controls and at most three owned paths, counting both old and new paths in a move. Shared Rust and Lean build lanes are serialized; every Lean subprocess uses the repository lock, one thread, memory/swap cap and timeout. Full Rust tests/clippy/fmt, extraction freshness, both Lake package builds and both applicable axiom-cone controls remain required at the final barrier.

## Plan — assurance factoring

Current result: all seven size breaches pass the unchanged checker; eleven shard migrations and
both audit relocations pass their focused gates/controls. Old paths below identify completed
migration ownership, not live consumers. [Integration evidence](../evidence/assurance-factoring.md)
records final proof checks and remaining programme obligations.


- Owner: lead; architecture owner: Astra; proof/attribution verdict: Sol.
- [Intent](assurance-factoring.md); [specification](assurance-factoring.md).
- Source: `262bb0a` plus accepted dirty baseline `target/open-areas/baseline.json` (SHA-256 in intent).
- Status: architecture handback, `anchor`; implementation gates below are not run.
- Spike: `APEIRON-OPEN-SPIKE-size-design`, bounded to twenty minutes; assigned documentation checkout exception at `/home/socol/Workspace/worktree-agent-APEIRON-OPEN-SPIKE-size-design`, branch `codex/open-size-design`. Own only these three lifecycle files. No production/build lane acquired.

### Design arc and observed gate

```text
SCOPE: docs/epics/assurance-factoring.md, docs/epics/assurance-factoring.md,
       docs/epics/assurance-factoring.md; seven-breach factoring architecture.
REGISTER: planning; excludes production modification, proof and performance closure.
CORPUS: Slang.Source.BytesAudit.represented_encoding_eq read;
        theorem represented_encoding_eq {bytes bytes' : SourceBytes}
          (h : ByteRep bytes (encodeBytes bytes')) : bytes = bytes';
        existing theorem pointer, no fresh Lean acceptance.
REFERENCE: /home/socol/Workspace/worktree-agent-APEIRON-OPEN-SPIKE-size-design/tools/check-assurance.py
           /home/socol/Workspace/worktree-agent-APEIRON-OPEN-SPIKE-size-design/tools/slang/export.py
GATE: git diff --check in the assigned checkout; source/path/count review.
      Repeated assurance check: exit 1, seven source breaches; benches/ provisioned.
GRADE: anchor; design before/after, no semantic acceptance or speed measurement.
```

Lead accepts these lifecycle artifacts for the user-authorized uncommitted workflow. Use existing focused commands with immutable source/output inventories and repeatable throwaway controls; do not add a broad orchestration layer. Generator freshness and imported-source coverage require narrow fail-closed extensions before sharding. No commits are authorized.

### Focused verification

Pure factoring uses existing commands: REG-1 and named integration Cargo tests, spec checker
self-test, and exporter freshness. Capture the baseline before edits, compare complete test/API/
constant/output inventories, run each named real mutation in a disposable copy, and restore green.
A pre-existing red gate stays open until its separately owned repair passes.

Generated-table migration adds output-map freshness and explicit imported-child source scanning.
Before the first shard, reject missing/changed/extra outputs, invalid output placement, missing/cyclic/
escaped child imports and unused forbidden tokens. Compare ordered public declaration bodies, source
JSON and digests against the frozen monolith. These controls are necessary beyond line counting.

All Cargo/Lean execution serializes on `/tmp/opencode/apeiron-lane.lock`. Lean uses8GiB/no-swap,
`LEAN_NUM_THREADS=1` and a 540-second whole-invocation cap; the zero-axiom runner retains its
6GiB inner memory bound and distinct per-corpus lock inside the outer global lock. Resource exhaustion
is inconclusive, never an intended mutation rejection.

### Exact ownership and dependency order

Every row owns at most three paths, including deleted source paths. Owner names are role assignments; the lead replaces each with one actual assignee/base revision/build lane before dispatch. Default worker is Flash if available; do not invent unavailable model access. Each task gets `worktree-agent-<task-id>`. No task edits this plan while another owns it.

| Task ID | Owner | Owned paths | Dependency / change | One gate |
|---|---|---|---|---|
| `APEIRON-SIZE-W2-T1` | REG-1 worker | `examples/reg1/acceptance.rs`; `examples/reg1/acceptance/report.rs`; `examples/reg1/acceptance/registry_tests.rs` | Move reporting/tests as specified; wait for existing REG-1 owner | `cargo test --locked --example reg1` |
| `APEIRON-SIZE-W2-T2` | table worker | `src/table.rs`; `src/table/prepared.rs`; `proof/regen.sh` | Wait for active frozen-source Sol owner; include free bodies and extend digest list | `cargo test --workspace --all-features --locked` |
| `APEIRON-SIZE-W2-T3` | test-address worker | `tests/native_address.rs`; `tests/native_address/main.rs` | Preserve target and body | `cargo test --locked --test native_address` |
| `APEIRON-SIZE-W2-T4` | test-locator worker | `tests/native_locator.rs`; `tests/native_locator/main.rs` | Preserve target and body | `cargo test --locked --test native_locator` |
| `APEIRON-SIZE-W2-T5` | test-parallel worker | `tests/parallel.rs`; `tests/parallel/main.rs` | Preserve target and body | `cargo test --locked --test parallel` |
| `APEIRON-SIZE-W2-T6` | spec worker | `tools/slang/spec_check.py`; `tools/slang/spec_constants.py` | Extract frozen data; retain path anchors in CLI | `python3 tools/slang/spec_check.py --self-test` |
| `APEIRON-SIZE-W2-T7` | exporter worker | `tools/slang/export.py`; `tools/slang/export_parts/decode.py` | Extract pure decoder and explicit imports; root still oversized | `python3 tools/slang/export.py --check` |
| `APEIRON-SIZE-W3-T1` | exporter worker | `tools/slang/export.py`; `tools/slang/export_parts/scan.py` | After decoder extraction; inject provider/root, no cycle; root now below cap | `python3 tools/slang/export.py --check` |
| `APEIRON-SIZE-W4-T1` | exporter worker; Sol verdict required | `tools/slang/export.py`; `tools/slang/export_parts/table.py`; `tools/slang/tests/test_table_outputs.py` | After scan extraction; output-map interface, prefix count zero, exact legacy bytes | `python3 -m unittest discover -s tools/slang/tests -p test_table_outputs.py` |

Immediate dispatch priority: REG-1 reporting/tests, spec constants, and the three target moves. These do not wait for generator tooling. Existing commands are their positive gates; run the corresponding throwaway control and restore with the same command, plus byte-identical moved-test/target inventory checks. For a moved integration test, change one existing expected result so that its named test fails, then restore; deleting a test is separately rejected by the inventory comparison.

Disjoint source edits in the second group may run concurrently, but Cargo gate runs serialize the `target/` lane. The table row additionally hands off a source identity to a dedicated extraction task; no worker starts another extraction while the lead's proof owner holds the lane. Generator rows are serial because they share `export.py`.

#### Generated-source token-scan coverage prerequisite

`APEIRON-SIZE-W4-T2` owns exactly `tools/slang/zero_axioms.py`, `tools/slang/zero_axioms.json` and `tools/slang/tests/test_source_closure.py`; owner audit-gate worker, Sol verdict mandatory. The existing tokenizer-witness registry lists `Generated/Table.lean` but cannot yet cover moved leaf definitions through its explicit source-file list. Before sharding, add an explicit import-closure scope to that stage rooted at `Slang.Generated.Table`, restricted to its `Slang.Generated.Table.*` child modules. Traverse imports from the registered root and apply the same forbidden-token scan to every reached shard; retain all existing explicit files, declarations, types and checks. Missing/repeated/cyclic/escaped module paths must reject; do not make unchecked filesystem globbing a source population. Other existing imports remain governed by their existing registered files.

Its focused gate is `python3 -m unittest discover -s tools/slang/tests -p test_source_closure.py`.
Fixtures exercise the real source scanner and reject unused `noncomputable`, `sorry`, missing/cyclic
imports and escaped module paths. The existing tokenizer-witness audit and its controls must also
pass before sharding. Sol conditionally accepts the relocation architecture; actual gates remain
required. This task adds coverage and preserves every declaration/type registration.

#### Generated shard migrations

The following are eleven separate serial barriers. Every row owns `tools/slang/export_parts/table.py`, `proof/corpus/Slang/Generated/Table.lean` and exactly the listed new shard. Increase the persisted prefix count by exactly one, regenerate, verify that all prior shard bytes remain unchanged, and leave the remaining suffix in the root. Do not write additional generated files: abort on unexpected drift and report the exact affected paths. This is the same production renderer at every step; no dead prototype outputs.

| Task ID | New shard under `proof/corpus/Slang/Generated/Table/` | Gate case |
|---|---|---|
| `APEIRON-SIZE-W5-T1` | `Chunk01.lean` | `table-shard-01` |
| `APEIRON-SIZE-W6-T1` | `Chunk02.lean` | `table-shard-02` |
| `APEIRON-SIZE-W7-T1` | `Chunk03.lean` | `table-shard-03` |
| `APEIRON-SIZE-W8-T1` | `Chunk04.lean` | `table-shard-04` |
| `APEIRON-SIZE-W9-T1` | `Chunk05.lean` | `table-shard-05` |
| `APEIRON-SIZE-W10-T1` | `Chunk06.lean` | `table-shard-06` |
| `APEIRON-SIZE-W11-T1` | `Chunk07.lean` | `table-shard-07` |
| `APEIRON-SIZE-W12-T1` | `Chunk08.lean` | `table-shard-08` |
| `APEIRON-SIZE-W13-T1` | `Chunk09.lean` | `table-shard-09` |
| `APEIRON-SIZE-W14-T1` | `Chunk10.lean` | `table-shard-10` |
| `APEIRON-SIZE-W15-T1` | `Chunk11.lean` | `table-shard-11` |

Owner: exporter worker for each. Focused command for each: bounded `lake build Slang.Generated.Table`. Exact exporter freshness, ordered declaration/source-data comparisons, real leaf/order mutations and restoration are also mandatory. Prefix zero is temporary migration state; final acceptance requires all eleven shards, no body suffix and a below-cap aggregator. The eleven new files count normally.

#### Audit grouping and extraction identity

These rows are sequenced after generated-source work to avoid corpus build collisions. No proof statements or grades change; Sol's relocation verdict precedes the first row.

| Task ID | Owner | Owned paths | Change | One gate |
|---|---|---|---|---|
| `APEIRON-SIZE-W16-T1` | audit worker | `proof/corpus/Slang/Source/BytesAudit.lean`; `proof/corpus/Slang/Source/Audit/Bytes.lean`; `tools/slang/zero_axioms.json` | Move body, retain old shim, update every bytes-stage path/anchor; preserve namespace | `python3 tools/slang/zero_axioms.py --stage bytes --controls` |
| `APEIRON-SIZE-W17-T1` | audit worker | `proof/corpus/Slang/Source/BytesAudit.lean`; `proof/corpus/lakefile.lean` | Change Lake root, delete shim after consumer search | `python3 tools/slang/zero_axioms.py --stage bytes --controls` |
| `APEIRON-SIZE-W18-T1` | audit worker | `proof/corpus/Slang/Source/LexingAudit.lean`; `proof/corpus/Slang/Source/Audit/Lexing.lean`; `tools/slang/zero_axioms.json` | Move body, retain old shim, update every lexing-stage path/anchor; preserve namespace | `python3 tools/slang/zero_axioms.py --stage lexing --controls` |
| `APEIRON-SIZE-W19-T1` | audit worker | `proof/corpus/Slang/Source/LexingAudit.lean`; `proof/corpus/lakefile.lean` | Change Lake root, delete shim after consumer search | `python3 tools/slang/zero_axioms.py --stage lexing --controls` |
| `APEIRON-SIZE-W20-T1` | extraction owner | `proof/ApeironWord.lean` only if regeneration changes provenance text | After table source and digest-list change; source/body identity inspection; no wrapper restatement | `proof/regen.sh --check` after any required normal regeneration |
| `APEIRON-SIZE-W21-T1` | lead | `docs/evidence/assurance-factoring.md` | Record complete integrated receipts; no new production edits | unchanged assurance check plus listed integration gates |

Extraction is a regression gate, not permission to reopen W7 composition. A changed extracted body/name rather than expected source-location/provenance differences stops this task for a separately owned Sol decision. Do not broaden its one-file ownership to repair bridge files opportunistically.

### Barrier and handback

The lead executes and records:

- `python3 tools/check-assurance.py --check` and `--self-test`, unchanged limits/populations, no missing declared root.
- `cargo test --workspace --all-features --locked`, `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`, `cargo fmt --all -- --check`.
- `proof/regen.sh --check`; bounded `lake build` in both `proof/` and `proof/corpus/`.
- `python3 tools/slang/export.py --song-root <frozen-Song-checkout> --check`, exact source-data/digest equality and all freshness controls.
- `python3 tools/slang/spec_check.py --self-test`; `zero_axioms.py --stage bytes --controls`, `--stage lexing --controls`, `--stage tokenizer-witness --controls`, `--stage parsing-witness --controls`; existing Rust/Aeneas closed-cone audit and relevant storage mutation controls.
- Baseline/new Cargo target and test inventories, REG-1 registry/control inventory, Lean declaration/type inventory, all moved/generated file digests, tool versions and resolved corpus revision.

Do not update a source pin to obtain green. If current Song bytes or other retained gates are red before factoring, record that baseline and assign a separate source-identity repair; factoring is not semantically accepted until the required positive is real. No benchmark rerun is claimed by this plan; REG-1 performance acceptance remains its own open obligation.

Every slice returns changed files, exact command/exit, control diagnostic and restoration, counts/digests, `proved` versus `tested` versus open, and the next bounded task. Preserve unrelated dirty work. Only the lead may integrate, commit or push when authorized. On completion inventory worktrees/branches, remove task worktrees through `git worktree remove` and prune records; no task branch survives closure.

### Remaining decisions

- Lead: accept architecture under the user-authorized uncommitted workflow, appoint actual file owners and release frozen source/build lanes.
- Sol: adjudicate empty-cone/public-declaration preservation for moved audit bodies and private byte helpers; certify the gate-first negative controls. No premise or theorem-grade weakening is authorized.
- Generator worker: establish narrow output-map freshness and imported-source scanner controls before shard migrations. Independent factoring uses existing gates.
- Missing installed Aeneas/Lean dependencies in the isolated checkout require setup before acceptance; design review is not a substitute.
