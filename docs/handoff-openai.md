# Handoff: continue Apeiron on the OpenAI stack

You are an **Astra (Light Thinking)** orchestrator. You do not implement; you decompose, sequence,
dispatch, and close. You are picking up a Rust + Lean repository at a **green-but-open** checkpoint
and driving it to the next wave barrier. Read this whole document before acting.

---

## 0 · Standing rules (do not rediscover)

- **One in-progress owner per file.** Every task declares its owned files; tasks in one wave touch
  disjoint files. If a task needs a file another owns, sequence it.
- **A task is ≤3 files, one gate, one dedicated negative control** where soundness is at stake, and
  is closed by a command exit status, never by an agent saying so.
- **Parallel branches** are allowed only when file sets are disjoint and no shared build product is
  contested. Shared lanes (serialize):
  `flock /tmp/opencode/apeiron-lane.lock`, `proof/ApeironWord.lean`, `target/`,
  `proof/.lake/`, `proof/corpus/.lake/`.
- **Lean runs are bounded and serialized**: `systemd-run --user --scope -p MemoryMax=<cap>
  -p MemorySwapMax=0` with `LEAN_NUM_THREADS=1` and a `timeout`, on the lane lock. `set_option
  maxHeartbeats 0`, whole-source `decide`/`rfl`, and raised `maxRecDepth` are **forbidden** in
  committed proofs and in agent runs. A job that exhausts the cap is a failed run, not a red gate.
- **Do not commit, push, branch, worktree, stash, reset, checkout, restore or clean** without
  explicit lead authorization. The prior session committed the checkpoint; the tree is clean.
- **A mutation control edits a throwaway copy, never the working tree.**
- **Never generate URLs** except to retrieve a cited paper/repo for a comparison; retrieve it
  locally, read it, cite the exact file/revision/theorem.
- The sibling `song`/`song-ref` tree is a different repository; its ontology, `R1`–`R5` registers,
  `//@` semantics and `./song` gates **do not apply here**.
- Escalation by artifact, not by difficulty: **Astra** for architecture/integration (>3 files,
  repo/governing-doc boundary, overlapping paths), **Sol** for proof/contract/claim-grade change,
  **Terra** for a lead-assigned large build. When both Astra and Sol triggers hold, both verdicts
  are required.

---

## 1 · Repository and reading order

- Repo: `/home/socol/Workspace/apeiron` (Rust crate + Lean extraction + Lean bridge).
- HEAD: `dbc1bc8` — *storage: axiom-free extraction, green bridge, bounded Kani probe fixtures*.
  Working tree **clean**.
- Canonical corpus (pinned, do not move): `/home/socol/Workspace/sanguine-apeiron` at
  `3d66ea3aa2fe1de5997b6b757247454a2f771358`. `proof/corpus` depends on it.
- Read, in order, before acting: `AGENTS.md` → `docs/direction.md` → `docs/kernel-boundary.md` →
  `docs/flash-handoff.md` → `docs/plan/programme-roadmap.md` →
  `docs/evidence/kani-table-harness-audit.md` → `docs/extraction-requirements.md` →
  `proof/README.md`. Source and gates beat documents; fix documents that disagree.

Machine: `i9-14900KF`, 32 threads, 62 GiB RAM, `/home` ~431 GB free. Mathlib oleans are cached under
`proof/.lake/packages`; builds are fast — do not treat them as a resource concern.

---

## 2 · Environment and pins

- Rust: `nightly-2026-08-18`. Lean: `v4.31.0`.
- Charon `0.1.254 (b104e24f)`, charon-driver `207b223a`, Aeneas `505b6ca-dirty`.
- `AENEAS_HOME` unset → `target/aeneas`, a **symlink to another checkout**; never mutate it.
  The patched backend lives at `target/aeneas-model/lean` and is built by `proof/model-patches.sh`
  from `proof/patches/aeneas-lean-model.patch` (sha256
  `81ae0e16f88ceac7b1a4d7cceaa0211aa255ce7221c1e7f2fd8e9795b3f6a7b4`). `proof/lakefile.lean`
  requires it. A fresh clone must run `proof/model-patches.sh` before `lake build`.
- Kani `0.67.0`, CBMC `6.8.0`.
- `/tmp` is a 32 GB tmpfs (currently ~76% full from prior scratch).

---

## 3 · What is established (with grade)

- **Rust: `tested`.** `cargo test --workspace --all-features --locked` rc 0;
  `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` rc 0;
  `cargo fmt --all -- --check` rc 0.
- **Extraction: `proved` cone, fresh.** `proof/regen.sh` rc 0 and `proof/regen.sh --check` rc 0.
  `proof/ApeironWord.lean` is 10923 lines, **0 `axiom`/`sorry`/`admit`**, sha256
  `e1385a4d2927b7b47c4454e16a30ce365d505c97606ecf3429622dac26cd5be9`, from **43 roots** including
  the write/growth path (`*_table_build_at`, `*_prepared_shard_apply`, `*_apply_insert`,
  `*_relocate_*`, `*_apply_pending_and_commit`, `*_publish_prepared`, `word_capacity_bound_holds`,
  `word_schedule_is_complete`). `lake build` in `proof/`: 1708 jobs rc 0.
- **Bridge: builds green, some statements conditional.** `lake build` in `proof/corpus/`: 3482 jobs
  rc 0; `AxiomAudit` reached under `{propext, Classical.choice, Quot.sound}`. The historical
  fixed-shard set (`TableRefinement*`, `Joint.TableInstance*`) is **retired from the roots** (files
  kept on disk). Relocation/rebuild headlines use `PayloadStateEq` and require
  `TraversalEffectsPrepared`.
- **Kani: 58/67 verified, 9 timeouts, no counterexamples.** Four new real-`spill_read` fixture
  harnesses in `src/table/probe.rs` are green (`tested`/bounded) with a passing mutation control.
  Reduced `cfg(kani)` geometry is landed (`SHARD_BITS`/`SHARDS`/`SHARD_MASK` = `2`/`4`/`3` under
  `cfg(kani)`, `8`/`256`/`255` otherwise, compile-time assertions bind both).
- **Security track: staged, not actioned.** Intent/plan written; a rebase premise is partly false
  (see §5·G).

---

## 4 · Level-1 acceptance reference (exact commands)

Rust (serialize with `nice -n 10` + `timeout`):

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
cargo kani                 # bounded harnesses; see §5·A for the capped form
```

Kani, bounded and parallel (the form that completes without a single serialized hang):

```sh
nice -n 10 timeout 1800s cargo kani -j 16 --output-format terse \
  -Z unstable-options --harness-timeout 300s
```

Proof regeneration (needs Aeneas/Charon under `target/aeneas` or `AENEAS_HOME`):

```sh
proof/model-patches.sh          # build target/aeneas-model/lean (patched backend)
proof/regen.sh                  # extract, strip modeled axiom stubs, audit, build
proof/regen.sh --check          # fail on drift from the committed ApeironWord.lean
```

Bridge build:

```sh
cd proof/corpus && lake build
```

Barrier set = all of the above plus the named controls, on one source/tool identity.

---

## 5 · Open obligations, ranked, with the branch that owns each

### A · Kani whole-`Table` harnesses (the only red gate) — Branch A
9 harnesses time out: `search_decision_classifies_a_bounded_two_lane_spill`,
`table_build_folds_a_multi_key_stream_in_order`,
`table_default_is_returned_for_a_key_never_written`,
`table_update_merges_new_with_the_stored_value`,
`table_insert_then_get_round_trips_a_present_key`,
`table_fresh_read_is_the_default_for_every_key`,
`a_reduced_shard_table_searches_symbolic_keys`,
`table_symbolic_byte_keys_fill_one_shard`,
`table_symbolic_key_round_trips_a_present_key`.
Root cause recorded: `table::probe::spill_read` (and the growable/locator machine) has loops bounded
by a symbolic `Vec` length and a data-dependent bitmask; CBMC cannot disgorge them. Astra's standing
decision: **do not stub, mirror, retire or disable the real path**; the timeout is a feasibility
blocker. The probe's own correctness is now covered by the fixture harnesses; the remaining
obligation is the whole-`Table` wiring plus `search_decision`/`spill_locate`.
Required: a Sol-ratified decidable harness form (Astra's "directly check the real helper on
concretely sized, initialized fixtures, without carrying whole allocation/growth/locator
construction"), reachable positive and negative cases, and a real-source mutation that fails.

### B · The missing universal read-path links (the real hole) — Branch B
No bridge theorem proves the generated matcher returns the represented key's location or sound
absence. `word_{stored,regen}_search_decision_simulates`
(`proof/corpus/Growable/Search.lean:338,397`) conclude about `apply_get` and take
`SearchDecisionCorrect` as a premise. Canonical `get` is `m k` (`Apeiron/Table/Basic.lean:69`).
Named missing goals (in `docs/spec/read-path-specialization.md`):
- `Growable/Scan.lean`: source-linked summary/fingerprint exclusion facts.
- `Growable/Representation.lean`: representation facts and their construction/preservation.
- `Growable/Search.lean`: prove generated `spill_read` and both `word_{stored,regen}_table_get_at`
  roots refine canonical `get`; separately establish `SearchDecisionCorrect` for the generated
  insertion classifier including `spill_locate`.
Do **not** manufacture the read result by assuming the desired classifier postcondition.

### C · Extraction/bridge premises — Branch C
`TraversalEffectsPrepared` is a premise the unextracted planner/resource boundary must construct.
`SearchDecisionCorrect` and `ResourceAdapterSimulates` are unproved. `core.mem.size_of` was replaced
by a documented budget constant (`ASSUMED_SLOT_BYTES = 32`), not modelled — a layout assumption.

### D · `Vec.try_reserve` model soundness — Branch D
The model patch makes `try_reserve` deterministic (cannot fail except at the `Usize` ceiling). The
sound form needs may-fail WP semantics plus a Kani harness that exercises a real refusal, with
`ResourceAdapterSimulates` as the named conformance premise. Astra flagged a possible separate task
on `src/table/resources.rs`.

### E · Performance gate `G1`/`REG-1` — Branch E
Strict acceptance harness `examples/reg1/` plus `examples/micro_spill.rs`, `examples/micro_spatial.rs`.
Contract: every applicable `(cell × incumbent)` pair `<= 1.0`, never behind; stretch `<= 0.8`;
incumbents `hashtrinity::{Table,CoordMap,FreqMap}` (`4cc14471`), `hashbrown+{foldhash,fxhash,rayon}`,
`std`, par16, `xxHash`, `wyhash`, `ahash`, `fnv`; `blake3`/`sha256` informational. A number is a
measurement only with a named command, machine, metric and interval, before and after.

### F · Storage epic waves — Branch F
Intent/spec/plan already exist for `algebra-native-addressing`, `parallel-stable-scatter`,
`read-path-specialization`, `spinor-*`. Implement the next wave's tasks (≤3 files, one gate, one
control each). Do not start an epic without a written intent, spec and plan.

### G · Security/algebra rebase — Branch G
`docs/intent/security-algebra-rebase.md` + `docs/plan/security-algebra-rebase.md`. The premise is
partly false: SEC-R1..R5 assume canonical security targets that do not exist. The canonical corpus
has only the trichotomy algebra and no `Universe`/`Term`/`Polymorphic` substrate; federation needs
that substrate built in canonical first. This requires an **Astra re-decomposition** before any
SEC-R* dispatch. Doctrine: the documents must let a claim be decided either way; importing priors
pro/con is unacceptable; each load-bearing claim gets a confirm/refute pair; unresolved is a valid
result.

### H · Housekeeping — Branch H
`src/table.rs` exceeds the 1500-line cap (open). `proof/README.md`, `docs/direction.md`,
`docs/plan/programme-roadmap.md` need the current extraction digests, the 43-root axiom-free
extraction, the retired fixed-shard bridge set, and the `PayloadStateEq`/`TraversalEffectsPrepared`
statements. Stale "unextracted" claims in `Growable/Search.lean` and `proof/extract/lib.rs:323-325`
must be corrected (the probe is extracted but unlinked).

---

## 6 · Parallel branch plan for the first wave

Dispatch these **concurrently** (disjoint file sets; serialize only on shared lanes):

| Branch | Owns | Gate | Control | Depends on |
|---|---|---|---|---|
| A1 | `src/proofs.rs`, `docs/evidence/kani-table-harness-audit.md` | `cargo kani` (capped) | wrong-shard-mask + omitted-relocation-publication mutations | Sol ratification of the decidable harness form |
| A2 | `src/table/probe.rs`, `docs/evidence/kani-table-harness-audit.md` | `cargo kani` on new fixtures | `WINDOWS+groups → WINDOWS` mutant | A1's file lock (A2 owns probe.rs; A1 owns proofs.rs) |
| B1 | `proof/corpus/Growable/Scan.lean`, `proof/corpus/AxiomAudit.lean` | `lake build` in `proof/corpus` | empty-axiom audit rejection | canonical pins |
| B2 | `proof/corpus/Growable/Representation.lean`, `AxiomAudit.lean` | `lake build` | representation-mutation reject | B1 |
| B3 | `proof/corpus/Growable/Search.lean` | `lake build` | `SearchDecisionCorrect` counterexample control | B1, B2 |
| C1 | `proof/corpus/Growable/{Relocation,Rebuild,Resources}.lean` | `lake build`, `proof/regen.sh --check` | premise-omission mutant | A freeze |
| D1 | `proof/patches/aeneas-lean-model.patch`, `proof/strip-modeled-axioms.py` | `proof/regen.sh --check`, `lake build` both packages | `try_reserve` failure harness | Sol contract verdict |
| E1 | `examples/reg1/*`, `examples/micro_spill.rs` | strict `--accept` run + measurement | incumbent swap | none |
| F1 | one epic's next wave (≤3 files) | that epic's gate | its named control | doc exists |
| H1 | `src/table.rs` split, `proof/README.md`, `docs/*` | `cargo test`, `cargo clippy`, `cargo fmt` | line-count check | none |

**Serialization constraints:** `proof/ApeironWord.lean` and `target/` are single-owner lanes; only
one regeneration task at a time. `proof/.lake/` and `proof/corpus/.lake/` each take one agent.
Branches A1/A2 both cite `docs/evidence/kani-table-harness-audit.md`; serialize the doc edits (A1
first, A2 appends). Keep `proof/corpus/AxiomAudit.lean` edits serialized across B1/B2.

**Do not start:** any task without a declared gate; any Lean task with `maxHeartbeats 0`; any Kani
task that disables checks, stubs the real path, or retires a harness without a Sol verdict; fixed
shards (that decision is historical; adopting it needs direction/kernel-boundary/flash-handoff
amendment and a recorded performance blocker).

---

## 7 · Barrier and closure

A wave barrier requires **every** child gate green with its control run, plus the shared gates in
§4 on one source/tool identity. Record the sha256 of the bound sources and the Aeneas/Charon/Lean/
Mathlib revisions; evidence from a dirty tree is attributed to those digests, not HEAD alone. The
epic exits are owned by `docs/flash-handoff.md`. Report format for every slice: **changed files,
acceptance result (command + exit status), proved vs tested vs open, next bounded task.** No
`proved` grade for anything without a Lean kernel check and, if about Rust, a refinement theorem
from the real extracted source under the closed cone.

---

## 8 · First actions (do these before dispatching)

1. `git log --oneline -1` → confirm `dbc1bc8`; `git status --short` → confirm clean.
2. Read §1 documents; run `proof/regen.sh --check`, `lake build` in `proof/` and `proof/corpus/`,
   and the capped `cargo kani` to re-establish the reported identities (do not trust the numbers in
   this document without re-running).
3. Open **A1** with a Sol ratification request for the decidable whole-`Table` harness form; open
   **B1**, **D1** (Sol), **E1** and **H1** in parallel; hold **A2** on A1's Sol verdict and **B2/B3**
   on B1. **C1** stays frozen until A1/A2 settle the Kani boundary.
4. Report the wave-1 branch table with owners, files, gates and controls, then execute.
