# Plan — Growable table read-path specialization

- **Owner:** Astra architecture; Flash source/extraction; Sol bridge
- **Intent:** `docs/intent/read-path-specialization.md`
- **Spec:** `docs/spec/read-path-specialization.md`
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; supplied working tree dirty
- **Status:** approved for serialized dispatch

## Arc block

```text
SCOPE:      production read specialization, real-source extraction, bridge repair and matched read measurement; complete when every child gate/control is green and strict affected pairs are recorded without weakening G1
REGISTER:   state/refinement/test/measurement; excludes layout, cached summary state, second containers, spatial scatter and W7 composition
CORPUS:     Apeiron/Table/Basic.lean `get`, `get_insertWith_self`, `get_insertWith_ne` (proved); Growable/Search.lean `SearchDecisionCorrect` and Stored/Regen apply-get headlines at their current conditional scopes
REFERENCE:  /home/socol/Workspace/apeiron/src/table.rs and src/table/state.rs read from the supplied checkout; mechanism derived from docs/direction.md `Serial read specialization and stable parallel scatter`
GATE:       focused commands below; strict REG-1 baseline is red at pass=421 fail=1029, and no task may convert that observed red to acceptance by reducing coverage
GRADE:      lifecycle documents are anchor; implementation targets tested, bridge targets proved, runtime targets measured
```

The documentation task is the supplied-checkout exception and owns four documentation files. Every
implementation/proof task below owns at most three files. Use an isolated
`worktree-agent-APEIRON-PERF-READ-<id>` at the recorded base revision and serialize `target/`,
`proof/ApeironWord.lean`, `proof/.lake/` and `proof/corpus/.lake/`.

## Goal and completion condition

The production read performs no insertion-only vacancy work, exact found/default behavior is tested
and source-refined for both carriers, and before/after micro plus strict full-profile results are
recorded on one source/tool identity. These local gates do not close G1 unless the strict command
itself exits 0 for every applicable pair.

```sh
cargo test --workspace --all-features --locked && cargo clippy --workspace --all-features --all-targets --locked -- -D warnings && cargo fmt --all -- --check && cargo kani && proof/regen.sh --check
```

Run the two bounded Lean package builds separately under the required memory cap, timeout,
single-thread setting and `/tmp/opencode/apeiron-lane.lock`; they are barrier gates, not part of the
shell chain above.

## Non-goals

- No threshold, applicability, workload, incumbent, interval or control change.
- No benchmark-specific dispatch, second representation, cached summary width, flat geometry,
  first-empty rule, payload replacement, spatial scatter or W7 retry.

## Wave map

| Wave | Task | Files | Owner | Gate | Rejecting control |
|---|---|---|---|---|---|
| W0 | `APEIRON-PERF-READ-W0-T1` micro observability and frozen-source baseline | `examples/micro_spill.rs` | Flash | `cargo run --release --locked --example micro_spill -- --controls` | Corrupt one expected readback digest in a throwaway copy; the gate fails before timing, then restored gate passes. |
| W1 | `APEIRON-PERF-READ-W1-T1` production read specialization | `src/table.rs`, `src/table/state.rs`, `tests/table.rs` | Flash | `cargo test --locked --lib --test table` | In a throwaway copy, stop a summary-positive match scan at the first vacancy; the late-hit test fails, then restored gate passes. |
| W2 | `APEIRON-PERF-READ-W2-T1` real-read extraction | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | Flash | `proof/regen.sh --check` | In a throwaway copy, mutate one generated read exclusion branch; freshness fails, then restored gate passes. This is not the later semantic proof control. |
| W3 | `APEIRON-PERF-READ-W3-T1` summary/representation bridge repair | `proof/corpus/Growable/Scan.lean`, `proof/corpus/Growable/Representation.lean`, `proof/corpus/AxiomAudit.lean` | Sol | bounded `lake build Growable.Scan Growable.Representation AxiomAudit` in `proof/corpus/` | Remove recordedness from one live spill witness; the representation/summary headline fails, then restored build passes. |
| W4 | `APEIRON-PERF-READ-W4-T1` read/search bridge repair | `proof/corpus/Growable/Search.lean`, `proof/corpus/AxiomAudit.lean` | Sol | bounded `lake build Growable.Search AxiomAudit` in `proof/corpus/` | In a throwaway generated copy, let summary miss imply global absence without direct exclusion; the read theorem fails, then restored build passes. |
| W5 | `APEIRON-PERF-READ-W5-T1` focused post-change measurement | no edit; read `examples/micro_spill.rs` | Flash | `cargo run --release --locked --example micro_spill -- --controls` | The W0 digest/control set reruns unchanged; any accepted corruption rejects the result. |
| W6 | `APEIRON-PERF-READ-W6-T1` strict pair/full-matrix measurement | no source edit; existing `examples/reg1/` | Flash | strict full-profile command below | `--controls` must report every registered corruption red; any surviving control rejects the run. |

W1 starts only after W0 freezes the harness and records the unchanged-source baseline. W2 starts only
after W1 is green/restored. W3 starts only after W2 fixes the generated identity. W4 starts after W3
because both Sol tasks own `AxiomAudit.lean` and the corpus build lane. W5 starts from the
proof-repaired identity and uses the byte-identical W0 harness. W6 starts only after W5 records its
diagnostic result. No source or proof edit may overlap these waves.

## Task blocks

### APEIRON-PERF-READ-W0-T1 · Freeze the focused read micro harness and baseline

- **Scope:** Preserve the existing spill absent-miss/present-hit 4096-operation schedules; add the
  corresponding resident hit/miss schedules, independent expected readback digests, raw sample
  output and fail-closed `--controls`; run the baseline before any timed source edit.
- **Files:** `examples/micro_spill.rs`
- **Depends on:** none; run after the current frozen-source owner releases `src/table.rs`
- **Grade:** measured baseline plus tested harness controls
- **Gate:**

```sh
cargo run --release --locked --example micro_spill -- --controls
```

- **Rejecting control:** In a throwaway copy, corrupt one expected readback digest. The same gate must
  exit nonzero before timing; restore and rerun green. The control validates harness rejection, not
  table correctness.
- **Evidence to record:** harness/source/input SHA-256, host/compiler/governor/cache/thread conditions,
  all raw samples, medians/ns per op, digest results and baseline/red/restored exits.

### APEIRON-PERF-READ-W1-T1 · Production read specialization

- **Scope:** Inline resident address/index/classification/read application; route occupied non-matches
  to spill; add one read-only spill match path with summary-miss absence and complete
  summary-positive match coverage; leave insertion `SearchDecision`/first-fit/growth unchanged.
- **Files:** `src/table.rs`, `src/table/state.rs`, `tests/table.rs`
- **Depends on:** W0 and completion of the active frozen-source Sol owner
- **Grade:** tested target
- **Gate:**

```sh
cargo test --locked --lib --test table
```

- **Rejecting control:** In a throwaway worktree, terminate the summary-positive read match scan when
  the first empty tag is observed. The real-path late-hit test must fail on a wrong default; restore
  and rerun green. The positive suite also asserts a summary-negative read performs zero vacancy
  scans and zero hold/clone/merge/allocation, so restoring insertion-locator vacancy bookkeeping is a
  separate required test failure.
- **Evidence to record:** source SHA-256; named tests and counts; baseline/red/restored exits; exact
  call graph showing `get_at` uses the production read path and insertion still uses
  `search_decision`/`spill_locate`.

### APEIRON-PERF-READ-W2-T1 · Extract the real specialized read

- **Scope:** Add/retarget monomorphic Stored/Regen wrappers and selectors for the production read,
  regenerate once, and retain every existing root. Wrappers specialize and call; they do not restate
  resident or spill matching.
- **Files:** `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean`
- **Depends on:** W1
- **Grade:** extraction audit; not semantic proof
- **Gate:**

```sh
proof/regen.sh --check
```

- **Rejecting control:** In a throwaway copy of the generated artifact, change the summary-negative
  read branch or a full-key comparison. The freshness diff must fail; restore and rerun green. W4
  supplies the semantic control, because freshness alone proves no read theorem.
- **Evidence to record:** all source/generated SHA-256 values, actual Aeneas/Charon versions, generated
  body inventory for both carriers, audit output and baseline/red/restored exits. A context-join or
  external-axiom failure is a blocker, not permission to add a model root.

### APEIRON-PERF-READ-W3-T1 · Repair summary and representation premises

- **Scope:** Source-link/audit spill-summary shape, recordedness, miss soundness and fingerprint
  exactness and carry them through `RepR`/`ReservedRepR`, preserving all existing projections.
- **Files:** `proof/corpus/Growable/Scan.lean`, `proof/corpus/Growable/Representation.lean`,
  `proof/corpus/AxiomAudit.lean`
- **Depends on:** W2
- **Grade:** proved target under the closed Rust/canonical cone
- **Gate:** bounded per repository policy:

```sh
lake build Growable.Scan Growable.Representation AxiomAudit
```

- **Rejecting control:** Remove a required recorded summary bit from one live spill witness in a
  throwaway proof copy. The named constructor/projection or miss-soundness dependency fails; restore
  and rebuild green.
- **Evidence to record:** source/generated/proof digests, Lean/Mathlib revisions, named headlines,
  closed axiom sets and baseline/red/restored exits.

### APEIRON-PERF-READ-W4-T1 · Repair the read/search refinement

- **Scope:** Give `SearchDecisionCorrect` the direct/summary/coverage non-found sum; prove both
  carrier public-read found and absence branches against `Apeiron.Table.get`; preserve every
  insertion/apply headline and premise.
- **Files:** `proof/corpus/Growable/Search.lean`, `proof/corpus/AxiomAudit.lean`
- **Depends on:** W3
- **Grade:** proved target, conditional exactly on the named reachable representation/search laws
- **Gate:** bounded per repository policy:

```sh
lake build Growable.Search AxiomAudit
```

- **Rejecting control:** In a throwaway generated/proof copy, remove direct exclusion from the
  summary-miss branch or omit one summary-positive late group. The corresponding public-read theorem
  must fail semantically; restore and rebuild green. Compile errors from deleting a declaration do
  not count.
- **Evidence to record:** named theorem statements, joint Stored/Regen witnesses, canonical `get`
  attribution, axiom audit and baseline/red/restored exits.

### APEIRON-PERF-READ-W5-T1 · Focused post-change measurement

- **Scope:** Measure the unchanged resident/spill hit/miss 4096-operation schedules with the
  byte-identical W0 harness and compare against its frozen-source baseline; use no result as an
  acceptance substitute.
- **Files:** none; read `examples/micro_spill.rs`
- **Depends on:** W4
- **Grade:** measured target
- **Gate:**

```sh
cargo run --release --locked --example micro_spill -- --controls
```

- **Rejecting control:** Rerun the W0 digest corruption against the byte-identical harness in a
  throwaway copy; it must still exit nonzero before timing, then the restored gate passes. W6's
  independent oracle, workload identity, selected-implementation and slowed/failing-pair controls
  remain mandatory before any acceptance reading.
- **Evidence to record:** host, compiler, source/harness/input digests, governor/cache/thread
  conditions, raw samples, median ns/op and interval/protocol before and after.

### APEIRON-PERF-READ-W6-T1 · Strict affected-pair and full-matrix result

- **Scope:** Run the unchanged full registry and record every resident/spill present-hit and
  absent-miss pair plus the global headline. Do not filter to passing pairs.
- **Files:** none; read existing `examples/reg1/`
- **Depends on:** W5
- **Grade:** measured target; acceptance only if the command exits 0
- **Gate:**

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

- **Rejecting control:** The command's wrong workload/oracle, omitted incumbent, selected
  implementation, interval-crossing and strict-single-failing-cell controls must all report red. A
  surviving control or any missing/inconclusive pair rejects the campaign.
- **Evidence to record:** complete pair table, affected-cell intervals, full headline, exit status,
  raw log hash, machine/tool/source/harness/input identities. Any interval maximum above `1.0` stays
  red; no scoped improvement closes G1.

## Barrier checklist

- [ ] W1–W4 focused gates exit 0; each control is red and restored green.
- [ ] Both carriers have joint direct hit/absence, spill hit/absence, collision and unequal-`Sigma`
  witnesses; no separate incompatible witnesses.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` and `cargo kani` pass.
- [ ] `proof/regen.sh --check`, bounded `lake build` in `proof/`, and bounded full `lake build` in
  `proof/corpus/` pass on one source/tool identity.
- [ ] Existing relocation/resource/search controls are rerun; W7 remains frozen.
- [ ] Micro and strict full-profile outputs bind the same final source/harness identity.
- [ ] Every claim is labelled tested, proved, measured or open at its actual scope.
- [ ] No acceptance threshold, incumbent, applicability, workload, interval or exclusion changed.

## Stop and rollback conditions

- Stop on a false negative, late-hit loss, unequal-`Sigma` alias, found-value change, read callback or
  allocation, insertion-search change, surviving semantic mutation, generated axiom, or extraction
  wrapper that restates lookup.
- Stop and return to Sol/Astra if the real specialized read cannot be extracted under the pinned
  toolchain; retain the production correctness fix only if its Rust gate/control pass, but claim no
  proved read refinement.
- After three materially different failed implementation shapes, report the smallest unresolved
  source body and generated/proof goal. Do not change geometry, add cached state/another container,
  weaken complete coverage or relax `<= 1.0`.
