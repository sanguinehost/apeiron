# Escalation — Apeiron↔Intent lane: restart, root, or contain

**Type.** Architecture escalation with a verification component (Intent's executing harness, and
whether it must refine the canonical object).

**Authority.** Owner decision. A worker may not act on this: §9.1 — an architecture escalation
defines the boundary and decomposes work; it does not waive a register, bound or control, and no
worker unilaterally replaces the harness.

**Affected repositories.** `intent` (harness under question), `apeiron` (canonical object and the
primitives Intent must refine), `song` (reference shape, independent), `penglai` (application
policy, local).

**Governing documents in tension.**

- `apeiron/AGENTS.md` §Purpose and `docs/direction.md`: every executing face must refine the
  canonical object; anything not rooted in it is not accepted.
- `intent/docs/PROOF.md`: every pure decision function has one register row.

Intent meets the second and does not meet the first. The two rules are not the same strength.

## Evidence at issue

1. **Selection signal is gameable.** Verified: `fullforge` 15/15 → 0/15 after the relay split plus
   `admit_frame`, so frame/verdict forgery is **closed**. `MAX-PAYLOAD-FORGE` is still 15/15: a
   candidate that embeds the public case table (`oracle_cases.rs`) scores full marks implementing
   nothing. Selection integrity is **open**. This is the apeiron failure-register class *Measurement
   instrument defects*: `S` has no control that separates implementing the task from memorising it.
2. **Rooting gap.** `intent/proof/register.toml`: the registered cores are plain integer/bool
   predicates — `clamp_budget`, `stall_at`, `in_window`, `novelty_count`, `stays_inside`,
   `family_factor_table`, `stale_ratio_permille`, `admit_frame`. **None is an object of
   `R[x]/(x²−σ)`.** The `admit_frame` proof is a local refinement of a boolean/enum function.
3. **Runtime linkage absent.** `intent/Cargo.toml:17-20`; the `HistoryStore` trait is the seam.
   Apeiron ships as crate `apeiron`, MPL-2.0, direct linking permitted; Intent does not depend on it.
4. **Harness defects.** Failure catalogue: the path jail refuses its own work dir; `/tmp` is a fresh
   tmpfs per command; no wall clock on a tool call (one `shell` ran 271 s); compaction drops tool
   outputs.
5. **Debt.** `intent` carries 242 Python files / 13,239 lines (average 55 lines/file).

## Exact decision requested (one)

Given (a) verdict integrity is closed but the selection instrument is gameable by construction, and
(b) no Intent core refines `R[x]/(x²−σ)`, choose one:

- **A — Repair.** Keep the lane. Make Ring 6 load-bearing (randomised or reference-based grading, or
  secret held-out cases); add the `apeiron` runtime linkage; root the admission/selection cores in
  the trichotomy (`Cell<R>` with explicit `Sigma`; the two capabilities as `e_±`; refusal set as the
  null cone; the σ→0 collapse as the guarded failure), then refine the executing Rust to that object.
- **B — Restart.** Rebuild the Intent harness in Rust and Lean with the trichotomy as the object;
  Python only as the workload interpreter. Name what carries over.
- **C — Contain.** Stop Intent's selection loop; keep Apeiron's primitives only; make no selection
  claim; withdraw the RSII deliverable.

State the chosen option, the option rejected, and why, in ≤400 words. If B or C, name the artifact
that must be preserved and the first bounded task. If A, name the Ring 6 gate and the control that
makes it a measurement.

**Do not return:** a general review, a request for more context, or a code change. Historical green
receipts are scoped evidence, not fresh acceptance.

## Resolution — owner, option A: Repair, with a precise rooting boundary

Retain the lane and its existing Rust/Lean artifacts; **hold candidate acceptance** until the
selection instrument passes rejecting controls.

- **Rejected B.** Intent is already Rust-native with extracted decision cores; the Python file count
  does not establish that replacing the harness fixes selection integrity.
- **Rejected C.** Withdrawing self-improvement abandons the central Apeiron–Intent objective while
  the identified defects have bounded repair paths.

### Ring 6 gate — currently MISSING

- Artifact: `crates/intent-cli/tests/ring6_selection.rs`.
- Command: `cargo test -p intent-cli --locked --test ring6_selection -- --test-threads=1`.
- Budget: 540 s whole invocation.
- Contract: exercise the production evaluator with fresh generated **evolve-split** inputs, trusted
  reference outputs, and reproducible seed receipts. Candidates receive inputs, never expected
  answers. Preserve Intent's held-out reporting/checkpoint policy (`intent/docs/ARCHITECTURE.md:120`)
  — making held-out data the ordinary selection signal changes its authority.
- Required controls:
  1. a correct reference implementation passes;
  2. the public-table memorizer fails the registered acceptance threshold;
  3. wrong-result and frame-forgery candidates fail;
  4. a disposable mutation restoring static-table grading makes the memorizer-rejection test fail;
     restoration passes.
- Grade: **tested discrimination against these attacks.** Measured selection improvement
  additionally requires the registered paired calibration campaign; it proves no universal immunity
  to gaming.

### Rooting boundary

Direct linkage supplies no refinement. "capabilities = `e_±`, refusals = null cone, guarded failure =
σ→0" remain **proposed mappings** until their carrier, hypotheses and connecting theorems exist.
Require the explicit canonical construction and executing-source refinement; preserve local admission
proofs at their current scope.

### Sequence

Ring 6 specification/gate → evaluator and isolation repairs → explicit Apeiron linkage and canonical
refinement tasks → calibration and candidate screens → one accepted improvement cycle.

**Preserve** the campaign dataset, the forgery controls, the frozen proofs, and the current harness
for comparison. Missing gates remain open.

**Drain note.** `StoredRelocateAll` integrated at `6267778`; its focused result closes no Intent
selection obligation.

## Fallback if no decision

The lane holds at its current state (no new dispatch). Open obligations stand:
Ring 6 (epistemic); Ring 2 (`/proc/<relay>/fd/1` needs nested `--unshare-pid`); calibration (≥30
paired incumbent reruns, `δ_1` per class); 5 candidate screens; 4 harness defects; the rooting gap;
the runtime linkage. The untracked campaign dataset
`intent/artifacts/space-bunny-evolution/sb-2026-10-01/` and the closed forgery result are preserved
whichever option is chosen.
