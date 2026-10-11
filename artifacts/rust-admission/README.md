# Rust admission gate: scoped acceptance

The public `python3 tools/dispatch/admit.py --launch <proposal.json>` entry executes
`tools/dispatch/run` and the Rust adapter. No inference optimization contract is registered.

| claim | grade | evidence |
|---|---|---|
| Real Rust admission state machine refines ordered, successful observations with equal identities | proved, conditional | `proof/dispatch/Dispatch/Refinement.lean`, `permit_iff`; nine audited headlines; actual Rust success/refusal witnesses |
| JSON, filesystem/hash, process and deadline adapter | tested/trusted | thirteen unit tests; one constructor compile-fail check; fifteen synthetic rejection controls and restoration |
| Unregistered production worker does not start through this entry | tested | `public-unregistered-launch.log`: exit1, `ADMISSION_UNREGISTERED_CONTRACT`, marker absent |
| Inference speed or a synthesis compiler | unmeasured/design | No new inference kernel or benchmark; typed combinatory synthesis remains a proposed direction |

`integration-gate.json` records the exact commands, cwd, exit codes and elapsed times.
The whole integrated run passes in21.715431s within540s. Rust tests, Clippy, formatting,
fresh extraction/refinement/source controls and public self-test exit0; the expected production
refusal exits1. Gate durations on the Intel Core i9-14900KF are not inference speed measurements.

The real Rust certificate-exit bypass produces fresh extraction and an attributable
`advance_refines` failure; restored source yields byte-identical extraction and passes.
The `proof-*-receipt.json` files bind source/tool hashes and retain baseline/red/restore logs.
Additional logs cover injected-axiom rejection, toolchain-pin rejection, and parent-deadline
guard mutation with restoration. Synthetic controls accept marker fixtures only.

Independent review decisions: `proof-review.jsonl` ACCEPT for the conditional core;
`adapter-review.jsonl` ACCEPT for the public adapter and policy clarification.
The closed axiom cone is `{propext, Classical.choice, Quot.sound}`.

`acceptance.json` inventories source hashes, actual tool versions, tool hashes, local source
sizes and external trust. The Aeneas binary reports `505b6ca-dirty`; its digest identifies the
actual patched binary. The local inventory is scoped to this auxiliary harness, not the
whole programme's trusted checking perimeter or mandatory size enforcement.

Excluded from the proof: authenticity of observations supplied by callers of the pure core,
IO/process provenance, cryptographic collision resistance, rustc/LLVM, OS behavior, concurrent
source mutation, protected checker/registry enforcement, native bypass and worker sandboxing.
Registry population requires a real source/refinement/domain/cost contract and a production
positive with attributable refusal/restoration. The root/canonical inference epic barriers
and hardware cost correspondence remain open.

`tasks/` retains uncommitted source recovery archives; `task-cleanup.json` records removal
of this slice's completed worktrees/branches. Unrelated dirty work is preserved. No commit/push.
