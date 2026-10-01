# Epic: gate-driver — delete every per-task Python gate

## 1 · Intent

**Problem.** `proof/resource` holds 33 `*_gate.py`, 6,513 lines, every one re-implementing the same
harness: `flock` on a lane, `systemd-run -p MemoryMax=12G -p MemorySwapMax=0`, a private lake
package, `charon`/`aeneas` extraction, a drift check against the committed `*Source.lean`, a `lake
build` of `*Resource.lean`, a cone audit, a real-source mutation in a throwaway copy, a restore, a
receipt JSON, and sha256 of every input. Only 18 lines are byte-identical across the 33;
identifier-blinding any two still leaves 116 differing lines. Each gate is un-reviewed boilerplate,
the audit surface grows with every cost-model task, and the swarm mints one more per task.

**Rejected alternatives.** Lint the existing scripts: the duplication is semantic, not textual
(18 shared lines), so linting leaves 33 divergent copies. Delete the scripts and run the steps by
hand: removes the mutation control, which is the acceptance evidence. A shared Python library: keeps
the interpreter dependency and the per-task drift, and adds an indirection the audit must now cover.
Rust driver: root-cause fix, and the precedent already exists — `tools/dispatch/admit.py` is a thin
entry into the Rust core `tools/dispatch/{main.rs,core.rs}`.

**Measurable success condition.** Every `proof/resource/*_gate.py` is deleted, and `tools/gate`
reproduces, per module, the same green markers and the same named mutation red the deleted gate
produced, inside the same 540 s budget, recorded as a retirement receipt. After the freeze date, no
new `*_gate.py` is added.

**Non-goals.** No Lean statement, cone, or claimed headline changes. No extraction root changes. Not
`proof/regen.sh`, not `tools/slang/zero_axioms.py`, not `tools/dispatch/admit.py` — those keep their
paths and move logic only under their own tasks.

## 2 · Spec

**Driver.** `tools/gate`, a Rust binary. One binary, N specs.

**Spec.** `proof/resource/<module>.gate.toml`, one per module:

| field | meaning |
|---|---|
| `module` | `Fwht`, `Argmax`, … |
| `root` | the extract root (`crate::word_…`) |
| `sources` | real source paths with expected sha256 |
| `wiring` | the `#[path]` mirror lines the private package needs |
| `wrapper` | the typed forwarding `pub fn` |
| `headline` | the named claimed theorem the mutation must break |
| `mutation` | file, exact needle, replacement |
| `markers` | the green marker sequence the gate prints |
| `lane`, `budget_s`, `cone` | lane lock, wall-clock ceiling, axiom cone |

**Behaviour.** Acquire the lane → materialise the private package → extract with charon/aeneas →
drift-check against the committed `*Source.lean` → `lake build` `*Resource.lean` → audit the cone →
apply the mutation in a scratch copy → assert the **named** red → restore → assert green → write the
receipt.

**Partiality.** The driver exits non-zero on any stage failure with the stage and the tool named. It
never reports success without the mutation red, and a mutation that leaves the proof green is exit 1
(the check could not fail). Budget expiry is exit 124 with elapsed time and the incomplete stage.

**Trust boundary.** Aeneas/Charon, the compiler, IO/process/hash, and the lane scheduler remain
trusted. The claim the receipt supports is the source-level one only: no cycle, wall-clock, GPU or
machine-code claim.

## 3 · Plan

`epic → wave → task`; each task owns ≤3 files, carries one gate and one control.

- **Wave 0 — freeze (read-only).** Harvest each gate's declared markers, its source digests and its
  last attributable mutation red into `proof/resource/conformance/`. This is the suite the driver
  must reproduce. Touches no gate; reversible.
- **Wave 1 — driver.** `tools/gate` core plus the `Fwht` spec, reproducing Fwht's green and its named
  red end to end. One module proves the pattern before any migration.
- **Wave 2 — migrate.** The remaining 32, batched; each module gets its `.gate.toml` and a retirement
  receipt showing driver output equals the frozen conformance record.
- **Wave 3 — delete and guard.** Remove the 33 `*_gate.py`; add the check that a new `*_gate.py` is
  rejected in review.

**Retirement rule (failure register: consolidation row).** A removal is a migration or an authorized
retirement: it names the replacement (the `.gate.toml` plus the driver receipt), inventories the
consumers (every doc that cites the gate path), and gates reachability (the conformance record). No
bare `rm`.

**Acceptance for the epic.** The 33 deletions each carry a receipt whose driver output matches the
frozen conformance record; `tools/gate` exits 0 on the batch; no `*_gate.py` remains under
`proof/resource`.
