# Flash Operator Handoff

2026-09-11. Execution sequence for the Apeiron/Penglai consolidation.
Read `docs/direction.md` for the decisions and evidence. This document specifies
the order of work, acceptance gates and where to stop, not a completed migration.

## Stage Status

- **Stage 0 — done**: source/evidence inventory and tool pins recorded.
- **Stage 1 — done**: `proof/` (`apeironWordProof`), reproducible extraction, 1708 jobs.
- **Stage 2 — done**: `proof/corpus/` (`apeironWordBridge`), full-width refinement linked to
  `AnyRing`, mutation control. (The historical run reported 2609 jobs; the current bridge reports
  3276 jobs.)
- **Stage 3 — done**: bounded dyadic arithmetic, ordered-vs-associative reduction,
  scaled-vs-idempotent projection, identity/region contracts and container scope. 143 tests at the
  time; the workspace now reports 164. Evidence in `docs/direction.md`.
- **Stage 4 — done**: corpus pinned at `16112030` (Lean `4.31.0`, Mathlib `v4.31.0`); the carrier
  point readout, the scalar carrier step and the exchange are refined to the ring-generic corpus;
  the corpus-connected carrier lab exercises the production path. Bridge `lake build` exit 0
  (3276 jobs). Evidence: `proof/README.md`, `docs/plan/carrier-refinement.md`.
- **Stage 5 — current**: constructive material experiment. The corpus conservation and energy
  results (`CarrierEvolve.lean`, `CarrierDissipation.lean`, `CarrierVolume.lean`,
  `CarrierAuthority.lean`, `CarrierCirculation.lean`, `CarrierBodyWork*.lean`) are Lean-proved; the
  Rust production integration, float mapping and live scheduling remain **tested, not refined**.
  Keep that tested/proved split explicit until the fixed-line carrier realization/refinement proof
  lands. Evidence: `docs/autonomous-run-log.md`, `docs/direction.md`.

## First Session

**Execute stages 0 and 1 only, then report.** Do not dispatch a specialist until
the user approves a concrete assignment. Do not start a whole-corpus migration,
repair every known defect, or rewrite Penglai during the extraction setup.

You are the primary operator, configured as `deepseek-flash` using
`opencode-go/deepseek-v4.1-flash`. Select that agent, not just the model under a
different agent: its delegation permissions are part of the workflow.
Use normal permissions, not `--auto`. Approve specialist requests once rather
than always if you want to retain the per-invocation approval gate.

## Budget And Ownership

- Work directly. No swarm, generic exploration agents or automatic fan-out.
- `openai-sol` is the default candidate for a bounded implementation/proof task;
  `openai-astra` is for difficult mathematical or architectural reasoning.
- Ask approval for each specialist invocation or resume. State agent, scope,
  artifacts, unresolved question and gate first. Run one specialist at a time.
- Do not launch agents through shell/API calls or modify permissions to bypass
  the gate. Approving one task does not approve the remainder of this roadmap.
- After three materially different failed approaches to one obstacle, stop and
  report its smallest unresolved goal. Do not cycle tactics indefinitely.
- The configured Astra limit is 32 agent iterations, not a dollar/token cap.
  Keep the assigned context small; stop when the task is complete or blocked.
- Preserve all existing work. Do not commit, push, post issues, rename/move repos,
  change licenses or modify Song's active workstream without explicit permission.
- Serialize builds using the same output/cache directories. A worktree is not
  permission to mix incompatible Lean/Mathlib artifacts in a shared build cache.

## Stage 0: Freeze The Input

**Owner:** Flash. **Output:** exact source/evidence inventory for this slice.

Read applicable repository instructions, `docs/direction.md`,
`../penglai/proof/README.md`, and the exact Rust/Lean files needed below. Do not
re-read the entire old conversation or resurvey all repositories.

Check branches, dirty files and relevant upstream updates. Reference checkpoint:

| Repository | Known state; recheck rather than assume |
| --- | --- |
| Apeiron | Local/main `d043ad3`, many pre-existing uncommitted edits. |
| Canonical corpus | `../sanguine-apeiron` local `c7339cce`; upstream `apeiron-rebuild` at `6213271f`. |
| Song | Local gen2 `50f8e47d`; upstream gen2 `857e9434`, compiler/main `5596c18f`. Another agent owns this work. |
| Penglai | `1222477`, with working Aeneas integer-formula proofs. |
| Old Sanguine | Local `proof/null-transformer` `eaf22284`, dirty; upstream main `795c7a8f`. Migration reference, not the canonical dependency. |

Sanguine and sanguine-apeiron are worktrees of one Git repository. Fetching updates
refs but does not update either checkout. The previously missing parser commits
are now published in `6213271f`; do not report that historical blocker again.

Record the actual working-source digests, not only HEAD. A proof extracted from
dirty files cannot be attributed to the committed revision alone. Preserve the
unrelated work rather than stashing or committing it to obtain a clean tree.

**Gate:** input source, tool versions, baseline commands and known dirty state are
identified. Do not silently pull/rebase an active worktree to match this table.

## Stage 1: Maintainable Extraction

**Owner:** Flash. **Depends on:** 0. **Output:** a reproducible, development-only
extraction/build entry point in Apeiron for two real Rust operations.

Target `src/cell.rs::Cell<i64>::mul` and `norm_form`, with the actual `Ring for i64`
operations in `src/scalar.rs`. Thin calling wrappers may expose concrete types;
they must call the real implementation, not restate its formulas. No proof-only
arithmetic feature or alternate implementation.

The prior successful probe is `/tmp/opencode/apeiron-direction-extract.rs` and its
generated `/tmp/opencode/Apeiron-word.lean`. Treat these as examples, not durable
dependencies. If absent, reproduce them from the real modules.

The essential Charon selection was:

```text
--preset aeneas --start-from crate::word_mul --start-from crate::word_norm
```

Penglai's tools currently live under `../penglai/target/aeneas/`, with Aeneas
`505b6ca` and the Charon Rust toolchain documented in its proof README. The backend
uses Lean 4.31.0. Use its recorded archive/checksum or an explicitly configured
tool installation; do not require another game's ignored target directory on a
fresh checkout. Pin the Lean backend and transitive dependencies too.

Whole-module extraction previously emitted unknown external axioms for hashing,
`i128::trailing_zeros` and iterator folding, and failed Lean elaboration in dyadic
code. The two reachable word functions extracted and elaborated without those
axioms. Do not solve the entire dyadic/iterator surface for this stage.

Keep the setup minimal: extraction root, pinned proof package, one verification
entry point and a short status/command description. Use repository-relative paths
or explicit tool configuration, not `/home/socol` embedded in release artifacts.

**Gate:** regenerate from the same Rust source the crate builds, elaborate with
the pinned backend, and fail on stale generated code or an unmodeled operation
in the accepted dependency chain. Record flags and source/tool identities.
`-warnings-as-errors` alone did not reject generated axioms in the broad probe.
Run `cargo test --workspace --all-features --locked` and report its actual result.

**Stop:** report stage 1 as extraction/elaboration, not a proved algebra. Present
the stage 2 assignment for approval if specialist help is needed.

## Stage 2: The First Refinement Theorem

**Owner:** Flash integrates; one approved Sol/Astra task if needed.
**Depends on:** 1. **Output:** actual word operations linked to canonical AnyRing.

Import the canonical corpus at an immutable revision. The currently published
corpus uses Lean 4.30.0-rc2, so settle compatibility with the 4.31.0 Aeneas backend
in an isolated build. Test and propose a coordinated corpus/toolchain update;
do not silently bump Song's active dependency or transcribe substitute theorems.
Any compatibility patch must be explicit and checked before publishing a new pin.

Define the denotation of signed machine words in `ZMod (2^64)`. Prove generated
wrapping add/subtract/multiply and casts agree with that denotation. Then prove
the extracted cell operations agree with:

- `Apeiron.Trichotomy.AnyRing.ringMul`;
- `Apeiron.Trichotomy.AnyRing.ringNormForm`.

Compose those equalities with `ringNormForm_mul` and `ringMul_conj`.

**Gate:** total success for every admitted full-width input and all three Sigma
values, exact denotation equality and checked axiom cones. No reduced 4-bit domain,
`sorry`, admitted operation, or conditional-on-success substitute. Mutating the
multiplication's sigma term without changing the specification must break the
refinement proof; the unchanged original must pass again. Record concrete inputs
inhabiting any preconditions and the complete source-to-checked-verdict command.

This verifies the selected kernel, not map storage, rustc/LLVM or GPU execution.

## Stage 3: Numeric And Storage Contracts

**Owner:** Flash on small repairs, approved specialist for unresolved proofs.
**Depends on:** 2 for the transferred algebraic guarantees.

Do separate reviewed slices, each with a failing regression first:

1. Correct bounded dyadic arithmetic. The current source can compute
   `Dyadic::new(1,127) + Dyadic::whole(2) == Dyadic::new(1,127)` by losing alignment
   bits. Check representability of shifted values, intermediate sums/products and
   exponent growth, not just shift counts. Represent expected failure explicitly;
   do not advertise a total finite exact ring. Inventory callers before API changes.
2. Separate ordered approximate accumulation from genuinely associative reductions.
   Existing f32/f64 `Monoid` implementations do not justify reassociation. Do not
   transfer real channel/projection classification to wrapping words, or call
   multiplication by `(1,1)` an idempotent projector.
3. Verify the identity/region operations the next consumer needs: packing round
   trips, negative-coordinate Euclidean division, bounds, collision-safe identity,
   get/insert/remove, merge order and explicit capacity/failure outcomes.

**Gate:** each claimed operation has its own specification and evidence. The
unsafe/SIMD/concurrent map is not Aeneas-verified by proximity. A safe reference
needs a concrete storage refinement; tests against it alone do not prove an
optimized implementation equivalent. Preserve useful optimized paths as explicitly
tested implementations until their own obligations are discharged.

Do not complete every std-shaped collection API before advancing the carrier.
Fold's first consumer migration can follow the required table/identity gates
independently of the physics work. Hashtrinity adapters stay only where needed;
TLS, Tokio and iroh replacement do not gate this stage.

## Stage 4: Corpus-Connected Carrier

**Owner:** Flash integrates small kernels; scoped mathematical work escalates.
**Depends on:** 2 and the numerical/index contracts needed from 3.

Re-derive the minimal old-corpus density/phase/finite-boundary dependency closure
under canonical Apeiron. Check which statements already exist before adding any.
Generic Rust kernels belong in Apeiron; Penglai depends on them rather than copies
them. Game-specific adapters and physical interpretation stay in Penglai.

Connect the generated functions to the actual imported mathematical definitions.
Penglai's current three proofs establish integer formulas but import no physical
corpus. Its `u16` amplitude result does not establish a signed-amplitude contract.
Retain amplitude sign where required and prove intermediate/region-sum bounds.

**Gate:** a small carrier lab executes the refined operations, unique-face reads
and complete boundary accounting. Distinguish amplitude balance, density
continuity and energy. Correct outputs and rejecting mutations use the same source
revision. A readout identity alone must not be called a water solver.

## Stage 5: Constructive Material Experiment

**Owner:** a deliberately scoped derivation/implementation task, not broad research.
**Depends on:** 4. **Output:** one nontrivial supported evolving material regime.

Choose one physical behavior and its independently justified fixture before
expanding the API. Construct evolution and boundary/source handling, then derive
the claimed invariants from that construction. Do not assume zero residuals or
stationarity to claim the update satisfies them.

Use the newly published `ClassicalWaves` and `WaveTime` where applicable; neither
supplies the missing coupled material evolution. State the supported circulation,
dissipation and body reaction/work model. Extend to three spatial factors and
six-side boundary accounting before claiming volumetric cave/water behavior.

Specify world authority as versioned seed/content plus accepted causal inputs.
Prove any history summary sufficient before compacting it. Measure cold/revisit
evaluation and its dependency radius; finite local stencils do not make arbitrary
time evaluation constant-cost. Caches must not create another authoritative field.

**Gate:** a nontrivial evolution refines its specification, passes the selected
physical fixture and has explicit error/work bounds and measured cost. If the
candidate fails, identify the missing derivation rather than silently replacing
the architecture with an independent MAC/global-pressure or mutable-height solver.

## Stage 6: Bank, Valley, Island

**Owner:** Flash implements bounded gameplay slices after their kernel gates.
**Depends on:** the specific supported material regimes from 5, not all physics.

First connect the existing combat clearing to one physical bank/pool and item/body
interaction with one collision/material authority. Preserve combat, controller
support, peaceful progression, pause/reload and explicit failed-update behavior.
Keep policy in RON/Luau and bulk simulation in Rust; no replacement ECS or AI model.

Then expand the same game to a settlement, river/lake/outlet and woodland. The
playable loop is fight/negotiate, fell a tree, use timber, ignite/wet/extinguish,
and revisit the consequences. Use `paperboygold/ascii-venture` at `e933813` as the
recovered fire/water presentation reference, not as an authoritative physics port.
Do not import its terminal-size-dependent world or hardcoded combat rules.

Scale active entities/regions and geography only with observed whole-scene cost:
queries, history/residency, collision, scripts, snapshots and rendering. Report
p95/p99, memory, disturbed cases and failure budgets on named hardware. Preserve
offscreen causal effects and identities across activity/LOD transitions. No island
or mobile-performance guarantee from a hashmap microbenchmark.

Use Penglai's README verification commands for runtime changes. Physical models,
discretization, implementation refinement, tests and playtesting remain separate
evidence. Ordinary combat/content improvements need not wait for all of stage 5.

## Specialist Assignment Template

```text
AGENT: openai-sol or openai-astra; one invocation, user approval required
SCOPE: one specific result; research/review or implementation explicitly stated
SOURCE: repository, branch/revision plus dirty-source identity
FILES: exact files it may edit; other files read-only
CORPUS: exact definitions/theorems and their hypotheses
BLOCKER: current proof goal/error and up to three materially different attempts
GATE: exact command/property that must pass; negative control where applicable
DO NOT: broaden scope, weaken contracts, delegate, commit, push or post
RETURN: artifacts, commands/results, remaining assumptions or exact blocker
```

At the end of every slice, report: stage and status; changed files; actual checks
and exit codes; proved versus tested versus open; next bounded assignment. Update
the existing evidence record instead of producing another parallel roadmap.
