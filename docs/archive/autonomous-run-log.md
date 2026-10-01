# Autonomous Run Log

Owner authorized autonomous multi-agent execution across all stages and slices on 2026-09-11,
with decisions documented here and no per-step check-ins. This file is the record of what was
done, why, by which agent, and what remains.

## Operating rules I adopted

- **No issues or repo moves.** Work stayed as reviewable working-tree changes until the owner
  explicitly authorized commits and pushes on 2026-09-11.
- **Serialize Lean builds that share a `.lake`.** The corpus worktree `.lake` is written by both
  its own 4.30.0-rc2 toolchain and the Aeneas bridge's 4.31.0 toolchain; concurrent builds collide.
- **Cost ladder for agents:** use `deepseek-flash` for routine investigation, implementation,
  tests and bounded Lean work; `openai-sol` for hard implementation; `openai-astra` only for
  difficult mathematics I have actually attempted and failed. One agent at a time. Do not use
  `deepseek-pro`, Gemini or GLM agents.
- **Evidence standard:** a claim is `proved` only when it is a Lean/Aeneas theorem with a checked
  axiom cone; Kani is bounded; tests are tested. Container storage is tested, not verified.
- **Preserve existing work**, including uncommitted edits in every sibling repo.

## Known state at the start of the run

- Stage 3 complete. `docs/direction.md`, `docs/plan.md` updated.
- Stage 4 slice 1, corpus side, done:
  - `sanguine-apeiron/proof/Apeiron/Physics/Carrier.lean` — `density` **defined** as
    `normForm σ (R • u)` (literal trichotomy root), `density_eq_amplitude_sq` proved from
    `Trichotomy.normForm_smul`; integer `wordDensity`/`wordFluxFirst`. Builds at 4.30.0-rc2
    (native) and 4.31.0 (bridge).
  - `sanguine-apeiron/proof/Apeiron/Physics/CarrierFlow.lean` — flow instantiation via
    `Trichotomy.normForm_flow`.
- **Blocker:** the corpus's `Apeiron/Trichotomy/Flow.lean:310` does not elaborate on Lean 4.31.0
  (`Prod.snd ∘ flow` vs `fun u => (flow σ u).2`, `NormedField.toNormedSpace.toModule` /
  `Semiring.toModule` instance mismatch). The corpus pin bump therefore needs source migration,
  not just a manifest update.

## Work plan

| Run | Slice | Goal | Candidate agent |
| --- | --- | --- | --- |
| R1 | Stage 4.1b | Penglai link: extracted kernels → `wordDensity`/`wordFluxFirst` | gemini-flash |
| R2 | pre-4 | Corpus 4.31.0 analysis migration (Flow + cascade) | openai-sol |
| R3 | 4.2 | Carrier lab: refined ops, unique-face reads, boundary accounting | glm-flash |
| R4 | 4.3 | Old-corpus density/phase/finite-boundary closure under canonical Apeiron | openai-sol |
| R5 | 5 | Constructive material experiment (one regime + fixture) | openai-sol / openai-astra |
| R6 | 6 | Bank → valley → island gameplay slices | glm-flash / gemini-flash |

## Decision log

- **D1.** The Rust↔corpus link is the product of this program; tests and Kani are supporting
  evidence only. Therefore the corpus must build at the Aeneas backend's 4.31.0, and R2 is a
  prerequisite for anything that imports the calculus part of the corpus.
- **D2.** Root `density` on `Trichotomy.Basic` alone (`normForm` + `smul`) and keep the `flow`
  instantiation in a separate file, so the link does not drag the broken analysis files into
  4.31.0. This is what let R1 proceed before R2.
- **D3.** The corpus pin bump (toolchain + manifest) is applied only after R2 shows the source
  builds at 4.31.0 end to end.
- **D4.** Agent pool is restricted to the flash agents (`deepseek-flash`, `glm-flash`,
  `gemini-flash`) and the OpenAI agents (`openai-sol`, `openai-terra`, `openai-astra`).
  `deepseek-pro` is excluded by owner instruction. Prefer flash for bounded/mechanical work and
  OpenAI for hard implementation or mathematics.
- **D5.** The `deepseek-flash` subagent was configured against the `opencode-go` provider, which
  returned `Insufficient balance`; the agent file was repointed to the DeepSeek provider
  (`deepseek/deepseek-v4-flash`). Config is not hot-reloaded, so subagent dispatch only works
  after an opencode restart. Owner restarted on 2026-09-11.
- **D7.** New material authority is admissible when it is **derived from the trichotomy basis**.
  Momentum, circulation, force/work or dissipative state is therefore not rejected merely for being
  additional state; its construction, update and acceptance relation must be rooted in that basis
  and proved/accounted for rather than independently authored.
- **D8.** Owner authorized committing and pushing the completed consolidation on 2026-09-11.
  Commits remain repository-local; unrelated pre-existing Apeiron benchmark/example edits are not
  staged. Exact extraction inputs are included so pushed generated proofs remain fresh.

## Runs

### R1 — Penglai link (agent: `deepseek-flash`) — DONE

- **Goal:** state Penglai's Aeneas-extracted `physics_math.density` / `phase_flux` /
  `reverse_phase_flux` against the canonical corpus definitions.
- **Changed:** `penglai/proof/{lakefile.lean,lake-manifest.json}` (corpus `«apeiron»` +
  `mathlib @ v4.31.0` last), `penglai/proof/PhysicsProofs.lean` (import
  `Apeiron.Physics.Carrier`; new `density_refines`, `phase_flux_refines`,
  `reverse_phase_flux_refines` as `WP.spec` statements against `wordDensity`/`wordFluxFirst`,
  plus docstrings/glosses).
- **Result:** `cd /home/socol/Workspace/penglai/proof && lake build` → exit 0 (2605 jobs).
  No `sorry`, no statement weakened, corpus untouched.
- **Side note:** importing the corpus activates its custom `linter.apeiron.notation` inside
  `penglaiProof` (which sets `warningAsError`), so downstream declarations need docstrings/glosses.

### R2 — Corpus Lean 4.31.0 migration (agent: `openai-sol`) — DONE

- **Goal:** make the whole canonical corpus elaborate on the 4.31.0 backend.
- **Changed (migration-only, no statement weakened):**
  - `Trichotomy/Flow.lean:296,310` — `convert` to bridge 4.31 instance diamonds.
  - `Combinator/Basic.lean:462–463` — explicit `if True`/substitution simplification.
  - `Capability/MaskedTable.lean:72` — unfold `Table.get`.
  - `Doubling/Sedenion.lean:107,114` — mark Real-dependent definitions `noncomputable`.
  - `Physics/Quadratic.lean:8,469–583` — direct ODE import + instance-safe derivative conversions.
  - `Physics/Wick.lean:217,221` — instance-safe derivative conversions.
- **Result:** `cd /home/socol/Workspace/apeiron/proof/corpus && lake build Apeiron` → exit 0.
  The corpus is now 4.31.0-clean. Harness untouched.

### R3 — Corpus finite-boundary closure (agent: `openai-sol`) — DONE

- Added `sanguine-apeiron/proof/Apeiron/Physics/CarrierBoundary.lean` re-derived under canonical
  Apeiron from the old `Substrate.Algebra.ProductCellBoundary`. Theorems:
  `sum_forward_difference`, `rectangle_boundary_balance`, `closed_rectangle_flux_cancels`.
- Built at 4.31.0: `cd /home/socol/Workspace/apeiron/proof/corpus && lake build
  Apeiron.Physics.CarrierBoundary` → exit 0; no `sorry`. (Agent: `openai-sol`.)

### R4 — Penglai carrier lab (agent: `deepseek-flash`) — DONE

- Added `penglai/tests/carrier_lab.rs` (readout/boundary lab; not a solver). Six tests:
  `density_is_amplitude_squared`, `density_is_zero_iff_amplitude_is_zero`,
  `phase_flux_is_density_times_forward_difference`, `reverse_face_is_exactly_the_negative`,
  `closed_boundary_flux_cancels` (4×4 single-valued phase; interior faces cancel, boundary sum 0),
  `independent_flux_is_rejected` (`accept_face_flux` only accepts the phase-derived value).
- `cargo test --locked --test carrier_lab` → exit 0 (6 passed); `cargo test --locked --all-targets`
  → exit 0; clippy `-D warnings` and rustfmt clean. No production code changed.

**Stage 4 status: gate met.** Rust↔corpus link (R1), corpus 4.31.0 (R2), corpus density/phase
(R1) + finite-boundary (R3) closure, and the carrier lab (R4). Next: Stage 5.

### R5 — Stage 5 constructive material experiment (agent: `openai-sol`) — DONE

- Corpus: `sanguine-apeiron/proof/Apeiron/Physics/CarrierEvolve.lean` — discrete conservative
  density update (`rho' = rho - div J`, `J = rho · Δφ`); proved `evolveCarrierDensity_continuity`,
  `sum_carrierDivergence`, `evolveCarrierDensity_total_mass` (zero-boundary telescoping),
  `evolveCarrierDensity_nonnegative_iff_cfl`. Built at 4.31.0 (exit 0), no `sorry`.
- Rust: `penglai/tests/carrier_evolve.rs` — `zero_boundary_flux_conserves_total_mass_exactly`,
  `open_boundary_and_source_match_net_mass_accounting`,
  `nonconservative_update_fails_the_conservation_check`; `cargo test --locked --test
  carrier_evolve` exit 0. Readout-carrier evolution, not a water solver.

### R6 — Stage 6 first bank slice (agent: `openai-sol`) — DONE

- Surfaces found: `PhysicsWorld` voxel pools/bodies, separate `ItemWorld`, `simulation::fighting`
  collision queries; existing pool state not directly carrier-authoritative.
- Added `penglai/src/physics/bank.rs`, `penglai/tests/bank.rs`;
  `tests/bank.rs::carrier_bank_carries_item_from_cleared_cell_into_pool`,
  `independent_height_and_velocity_fixture_is_rejected_transactionally`.
- `cargo test --locked --all-targets` → exit 0; no combat/ECS/script changes.

### R7 — Stage 6 material consequence loop (agent: `openai-sol`) — DONE

- Added `penglai/tests/valley.rs`; no production files changed.
  `tree_timber_ignites_is_wet_extinguished_and_revisits_changed`,
  `independent_fire_wet_claim_is_rejected_transactionally`.
- `cargo test --locked --all-targets` → exit 0. Fire/wet read from the carrier authority via
  `Bank::readout`/`move_body`, not an independent authored field.

### R8 — Stage 6 scale / identity slice (agent: `openai-sol`) — DONE

- Added `penglai/tests/scale.rs`; no production changes. Tests:
  `stable_ids_and_offscreen_causes_survive_activity_transition`,
  `reused_id_for_a_different_region_is_rejected`. Models 1,024 entities across 64 regions with a
  64-update deterministic budget.
- `cargo fmt --all -- --check && cargo test --locked --all-targets` → exit 0.

## Session summary — where to resume

**Completed this session**
- R1 Stage 4 link; R2 corpus 4.31.0 migration (whole corpus now builds on the Aeneas backend);
  R3 finite-boundary closure; R4 carrier lab; R5 constructive carrier evolution; R6 bank;
  R7 valley material loop; R8 scale/identity; R9 three-axis/six-face conservation.
- Corpus additions under `sanguine-apeiron/proof/Apeiron/Physics/`: `Carrier.lean`,
  `CarrierFlow.lean`, `CarrierBoundary.lean`, `CarrierEvolve.lean`.
- Penglai: `proof/PhysicsProofs.lean` link theorems; `tests/{carrier_lab,carrier_evolve,bank,valley,scale}.rs`.

**Evidence classes**
- Proved (Lean, 4.31.0): density/phase root, boundary telescoping, discrete conservation.
- Linked (Aeneas): extracted `physics_math.*` = corpus `wordDensity`/`wordFluxFirst`.
- Tested (Rust): carrier lab, evolution conservation, bank/valley/scale behaviour.

**Deferred / open**
- D6: corpus native toolchain/manifest bump (avoided the hours-long Mathlib rebuild; 4.31.0 is
  established through the bridge harness).
- Penglai `transfer_conservative` remains open (pre-existing).
- Stage 5 now has bounded dissipation, basis-derived circulation accounting and equal-mass body
  reaction/work. Circulation time evolution, general masses, viscosity/pressure and a physical
  damping calibration remain open.
- Stage 6 now has an opt-in production carrier-contact impulse channel, but shipped content remains
  disabled and full playable integration plus whole-scene p95/p99 benchmarks are not done.
- Stage 3 item 5 (Aeneas storage refinement for map/table) remains tested, not verified.

**Rollback / review**
- All changes are uncommitted working-tree edits in `apeiron`, `sanguine-apeiron`, `penglai`,
  except `docs/archive/autonomous-run-log.md` and `docs/direction.md` in `apeiron`. No commits/pushes.
- Agent pool: `deepseek-flash` subagents (provider repointed to `deepseek/deepseek-v4-flash`) and
  OpenAI; `deepseek-pro` excluded by owner instruction.

### R9 — Stage 5 three-axis / six-face extension (agent: `deepseek-flash`) — DONE

- Added canonical corpus `Apeiron/Physics/CarrierVolume.lean`: extended/ghost sampled amplitude
  and phase fields; norm-form `carrierDensity3`; one shared x/y/z face value; six-face divergence;
  forward-Euler update; local continuity; finite-box boundary balance; closed-box cancellation;
  total-density conservation; explicit CFL equivalence. No `sorry`/`admit`.
- Added `penglai/tests/carrier_volume.rs`: exact-integer closed-box conservation, open-boundary plus
  source accounting, and rejection of misoriented/independent faces. This is tested accounting,
  not a proof or fluid solver.
- Independent review rejected the first draft: Lean sampled face `x+1` from amplitude/phase
  `x+1 → x+2`, while Rust correctly treated it as cell `x`'s outgoing face. The agent revised Lean
  to an explicit ghost convention: logical density reads `(x+1,y+1,z+1)` and face coordinate `q`
  reads the extended amplitude at `q` and phase difference `q → q+1`. This now matches Rust's
  lower-cell face anchor; boundary ghost amplitude remains explicit (clamped in the fixture).
- Independent gates: `lake build Apeiron.Physics.CarrierVolume` → exit 0 (2304 jobs);
  `cargo fmt --all -- --check` → exit 0; `cargo test --locked --test carrier_volume` → exit 0
  (3 passed); `cargo test --locked --all-targets` → exit 0 (one pre-existing ignored benchmark).
- Evidence: conservation and boundary results are Lean-proved; Rust correspondence here is tested.
  Remaining assumptions: unit spacing, first-order time step, single-valued phase-gradient flux,
  positivity only under `carrierCFL3`, and no dissipation/body-work/viscosity/pressure law.

### R10 — Stage 5 constitutive-closure inventory (agent: `deepseek-flash`) — DONE

- Corpus inventory found no canonical energy, force/work, viscosity, curl or vorticity law.
  `Transport.Basic` supplies only convex-mixing bounds; the old corpus contains useful obstruction
  and zero-circulation results but is migration input, not authority.
- A single-valued real phase has identically zero plaquette circulation. Nonzero vorticity therefore
  needs a branch-cut/bundle phase or a separate rotational field: an architectural decision, not a
  bounded proof slice.
- Body work needs body momentum/velocity and a coupling law; `BankBody` deliberately has neither.
  Adding them would be a new authority decision rather than closing the existing carrier algebra.
- Selected next slice: conservative 1-D diffusive relaxation of carrier density, with mass
  conservation, a proved Dirichlet-energy non-increase under an explicit stability bound, and an
  anti-diffusive negative control. The coefficient remains an explicit gradient-flow step parameter,
  not a physically derived damping constant.

### R11 — Conservative carrier dissipation (agent: `deepseek-flash`) — DONE

- Added corpus `Apeiron/Physics/CarrierDissipation.lean`: trichotomy-derived `carrierState`, shared
  down-gradient flux, conservative update, finite Dirichlet energy, total-mass conservation,
  nonnegative energy, and energy non-increase for the derived stable range `0 ≤ ν ≤ 1/2`.
  `overlarge_step_increases_energy` proves the `[0,1,0]`, `ν=1` control raises energy `2 → 8`.
- Added `penglai/tests/carrier_dissipation.rs`: five exact-integer tests covering conservation,
  strict stable decrease, the overlarge-step control, unequal energy at equal mass, and rejection of
  misoriented/independent faces. Rust evidence is tested, not proved.
- Independent gates: `lake build Apeiron.Physics.CarrierDissipation` → exit 0 (2304 jobs);
  `cargo fmt --all -- --check` → exit 0; `cargo test --locked --test carrier_dissipation` → exit 0
  (5 passed); `cargo test --locked --all-targets` → exit 0 (one pre-existing ignored benchmark).
- D7 follow-up: the input is definitionally the trichotomy norm-form density, but this slice does
  not yet prove that every updated scalar field is represented again by a trichotomy-basis field.

### R12 — Trichotomy authority realization (agent: `deepseek-flash`) — DONE

- Added `Apeiron/Physics/CarrierAuthority.lean`: `basisRealization d = sqrt(d) • one`, exact
  norm-form readback for nonnegative amounts, pointwise 3-D field reconstruction, and constructive
  realization/existence results for CFL-admissible volume updates and stable diffusive updates.
- Extended `CarrierDissipation.lean` with explicit boundary/interior stencil equations and
  `diffusiveUpdate_nonnegative` for pointwise nonnegative state and `0 ≤ ν ≤ 1/2`.
- Negative controls prove the `ν=1` witness becomes negative and cannot be represented by the
  canonical real-amplitude/unit-direction realization; the claim explicitly does not exclude
  negative hyperbolic norm-form values.
- Independent `lake build Apeiron.Physics.CarrierAuthority` → exit 0 (2306 jobs). A scratch axiom
  audit initially selected the stale partial Apeiron olean tree because `lake env lean` rewrites
  `LEAN_PATH`; rerunning the Lean 4.31 binary with the canonical corpus olean directory first
  confirmed all six audited headlines use only `propext`, `Classical.choice`, `Quot.sound`.

### R13 — Basis-derived circulation (agent: `openai-sol`) — DONE

- Added corpus `Apeiron/Physics/CarrierCirculation.lean`: a `Cell` field whose two basis coordinates
  are explicitly adapted to x/y oriented face values; exact readback; derived plaquette circulation;
  scalar-gradient zero circulation; a rotational witness with circulation `2`; rectangle discrete
  Stokes; and closed-boundary cancellation. The spatial adaptor is explicit material interpretation,
  not claimed to be forced by `normForm`.
- Added `penglai/tests/carrier_circulation.rs`, using the real public `apeiron::Cell<i64>` API. Four
  tests cover gradient-zero with nonzero edges, exact nonzero rotation, rectangle boundary equality,
  and rejecting an independent circulation claim.
- Independent gates: Lean build exit 0 (2303 jobs); all three audited theorem cones contain only
  `propext`, `Classical.choice`, `Quot.sound`; targeted Rust 4/4 and full all-target suite pass.
- Remaining: this is kinematic accounting. A basis-derived time law for circulation, stretching,
  forcing and coupling remains open.

### R14 — Basis-derived body reaction/work (agent: `deepseek-flash`) — DONE

- Added corpus `Apeiron/Physics/CarrierBodyWork.lean`: `exchangeImpulse p q = p-q`; exact carrier
  and body swap; impulse antisymmetry; total `Cell` momentum conservation; universal norm-form
  ledger conservation; opposite work; and a derived-impulse acceptance relation with controls.
  Nonnegative unit-mass kinetic-energy language is correctly restricted to `σ ≤ 0`.
- Added `penglai/tests/carrier_body_work.rs` using `Cell<i64>`/`Sigma`: seven tests cover the swap,
  impulse, momentum, all three norm-form ledgers, opposite work, exact witness values and rejection.
- Independent gates: Lean build exit 0 (2301 jobs); audited theorem cones contain only the standard
  three axioms; targeted Rust 7/7 and full all-target suite pass.
- Scope remains equal-mass instantaneous local exchange: no contact geometry, momentum field or
  time composition yet.

### R15 — Field-local body exchange (agent: `deepseek-flash`) — DONE

- Added corpus `Apeiron/Physics/CarrierBodyField.lean`: finite basis position/momentum fields,
  explicit Cell-valued contact, one-site exchange, selected/other-site locality, whole-field plus
  body momentum conservation, all-σ norm-form ledger conservation, and accepted/rejected rollback.
- Added `penglai/tests/carrier_body_field.rs`: seven tests cover accepted locality, whole-region
  ledgers, wrong position, independent/near-miss impulse, and out-of-range transactional rollback.
- Independent gates: Lean build exit 0 (2302 jobs); locality/conservation/rollback cones contain
  only the standard three axioms; targeted Rust 7/7 and full all-target suite pass.
- Production gap: the exchange remains duplicated in test helpers and is not scheduled in the live
  world; an explicit exact-Cell to live-float mapping is still required.

### R16 — Canonical Rust exchange kernel (agent: `deepseek-flash`) — DONE

- Added Apeiron `src/exchange.rs`, exported from `lib.rs`: generic `Exchange<R>`, derived impulse,
  elastic exchange and exact-claim acceptance. Added six focused tests including wrapping extremes.
- Refactored Penglai body-work/body-field fixtures to consume the canonical kernel; duplicate
  impulse/carrier/body formulas were removed without reducing their assertions.
- Independent review narrowed an overstatement: `Ring` is a behavioral contract, not a type-level
  proof, and partial `Dyadic` operations can panic. Docs now condition generic identities on lawful,
  successful operations; only `i64` wrapping execution is tested here.
- Independent gates: Apeiron workspace tests pass; changed library/test clippy clean with
  `-D warnings`; Penglai targeted 14/14 and full all-target suite pass. Full Apeiron all-target
  clippy still has unrelated pre-existing examples/bench `too_many_lines`/semicolon findings.
- Proof status: production Rust implemented/tested; full-width Aeneas refinement remains open.

### R17 — Aeneas exchange refinement (agent: `deepseek-flash`) — DONE

- Added generic corpus `CarrierBodyWorkAnyRing.lean` and extracted the real production
  `exchange_impulse`/`elastic_exchange` through monomorphic wrappers. Seven bridge theorems prove
  exact full-width `ZMod (2^64)` denotation, carrier/body swap, total basis momentum, symbolic-σ
  norm-form ledger and impulse antisymmetry.
- Extraction freshness passed with scalar/cell/exchange source digests; bridge build passed (2610
  jobs). Extracted cones use the standard three axioms; generic-ring theorems use `propext` and
  `Quot.sound`. A throwaway sign mutation broke all exchange refinements; restored source passes.
- Independent verification: `regen.sh --check` passed (1708 jobs), correct bridge `lake build`
  passed (2610 jobs), targeted production Rust 6/6 passed. One operator command accidentally
  chained `lake build` at the repository root after the successful freshness check and exited 1
  because no root Lake package exists; it did not modify files and the correct bridge command was
  then run successfully.
- Open operation: `accept_derived_impulse` used `bool::then`, which Charon emitted as an audited
  external unknown. It was intentionally not admitted or hand-modeled.

### R18 — Checked-acceptance refinement (operator + agent: `deepseek-flash`) — DONE

- Operator semantic-preserving source change: replace `bool::then` in `accept_derived_impulse` with
  an explicit `if`, removing the unsupported library combinator before extraction.
- Extracted the real checked path and generated componentwise `Cell<I64>` equality with no external
  `bool::then` axiom. Proved source-word equality characterization, totality, exact some/none
  acceptance/rejection, derived-claim acceptance and canonical result denotation.
- Independent `regen.sh --check` passed (1708 jobs); bridge passed (2610 jobs). Every acceptance
  theorem reports only `propext`, `Classical.choice`, `Quot.sound`. Branch inversion and one-slot
  equality mutation both broke the relevant theorems before restoration.
- Remaining trust boundary: Aeneas Std wrapping/equality models and source-level Rust only; no
  rustc/LLVM/machine-code claim. The sibling corpus dependency is still an uncommitted worktree.

### R19 — Live basis-mapping inventory (agent: `deepseek-flash`) — DONE

- Live Avian bodies store `f32` velocity and derive arbitrary mass; the exact grid stores water
  amplitude but no exact phase or momentum field. `Bank` remains test-only. No existing fixed-point
  momentum scale or Rust contact-event bridge exists.
- Making an exact Cell sidecar authoritative now would either conflict with Avian or require a new
  force/time law and persistence migration. Keeping live `f32` authoritative while exposing a
  read-only exact ledger adds no authority and cannot satisfy D7 positively, but is the smallest
  honest production call site for the proved kernel.
- Selected slice: checked exact-subset `f32 ↔ Cell<i64>` mapping, carrier basis from exact amplitude
  with zero second slot (because live phase is absent), and a read-only live contact audit. No
  writeback, force or solver claim.

### R20 — Live read-only carrier ledger (agent: `deepseek-flash`) — DONE

- Added Penglai production `physics/carrier.rs`: checked fixed-scale planar `BasisMomentum`, exact
  amplitude-derived `carrier_basis`, canonical checked audit, safe water-cell lookup and
  `PhysicsWorld::audit_body_contact` by stable ID. Slot b remains explicitly unavailable because
  the live world has no exact phase; zero is not claimed as physical zero flux.
- Added eight integration tests plus one unit test for mapping refusals, exact round trip, live wet
  acceptance, wrong/near-miss/dry/unknown/out-of-bounds rejection and snapshot immutability.
- Independent review corrected “bit-for-bit” to Rust `f32` value equality (`-0.0 == 0.0`). Changed
  library/test clippy clean; targeted 8/8 and full all-target suite pass.
- Evidence: Apeiron integer acceptance is Aeneas-proved; float mapping and live lookup are tested
  only. This is read-only and adds no authoritative sidecar or Avian writeback.

### R21 — Exact live phase/momentum authority design (agent: `openai-sol`) — DONE

- Selected an opt-in exact carrier impulse channel. Exact phase/site/body coupling momenta own only
  carrier-contact impulses; Avian remains authoritative for position, velocity, collision, combat,
  buoyancy and drag. Avian contact nominates a site but never writes basis momentum back.
- V1 is a bounded 2-D x/z plane with integer phase vertices, amplitude provenance and one `Cell<i64>`
  momentum per site derived as density times x/z phase differences. Contact entry performs the
  Aeneas-refined equal-mass exchange atomically and projects its accepted impulse one-way to Avian.
- Existing `water`, `flow`, `surface` and MAC fields do not refine this authority and remain disjoint
  legacy authorities. Full 3-D, phase evolution, pressure/viscosity and total-body kinetic claims
  remain excluded.
- Persistence/replay, bounded no-wrap proof, initialization refinement, lookup refinement and float
  projection error are explicit later obligations; they are not implied by the architecture.

### R22 — Opt-in live carrier impulse (agent: `openai-sol`) — DONE

- Added an optional bounded/versioned `CarrierAuthority` to `PhysicsWorld`, disabled for shipped
  content. It derives integer phase vertices and site momenta from seed/config plus initial exact
  water amplitude, tracks exact opted-in body coupling Cells and rising-edge contacts, and records
  accepted events in stable body order.
- A tick clones the complete authority, validates every observation/exchange/projection, then commits
  the exact batch and submits Avian linear impulses. Any error discards the candidate and applies no
  impulse; ordinary existing physics continues with a typed last error. Avian never feeds back.
- Added six production integration tests: deterministic seed/provenance, one-way first contact and
  no retrigger, leave/re-enter, disabled equivalence, invalid/dry/unknown/oversized rollback and
  stable event ordering. Existing snapshot schema remains unchanged and persistence is unclaimed.
- Independent changed-target clippy, targeted carrier authority/contact/body tests and full all-target
  suite pass. Existing content remains carrier-disabled.
- Evidence: live scheduling/float projection are tested only. Initialization/lookup/no-wrap and
  field-transaction refinement, persistence/replay and projection error proof remain open.

### R23 — Site-momentum initialization refinement (agent: `deepseek-flash`) — DONE

- Added production `physics::math::site_momentum`, calling the proved `phase_flux` once per x/z
  axis; live authority initialization now uses this kernel after checked `i16` conversion.
- Regenerated `PhysicsMath.lean`; proved total exact x/z values, canonical `wordFluxFirst`
  refinements and the R22 signed no-wrap bound. A swapped-z mutation broke only the expected z
  theorems and was restored.
- Independent review found the first no-wrap hypotheses used phase sample bounds for differences.
  Corrected: samples are `[-2,2]`, differences `[-4,4]`, and components remain within
  `±4,194,304 = 4·FULL²`. Added direct and real initialization fixtures reaching the exact bound.
- Independent proof build passed (2605 jobs, 19 standard axiom cones); changed-target clippy and
  targeted authority/contact tests pass (8+8). Phase hashing/indexing, bound enforcement, Avian
  projection and scheduling remain tested rather than refined.

### R24 — Carrier checkpoint and replay validation (agent: `openai-sol`) — DONE

- Added owned serde V1 checkpoint DTOs plus `export_carrier_checkpoint_v1` and transactional
  `restore_carrier_checkpoint_v1`. `Cell` uses an explicit local `{a,b}` DTO; Apeiron was unchanged.
- Restore validates version/algorithm, exact float-bit definition identity, tick/seed/body set,
  dimensions/lengths/bounds/contact indices/event caps and sequence. It regenerates phases and
  initial `site_momentum`, replays every canonical accepted event, and requires replayed final state
  to equal stored momenta before replacing the authority.
- Added JSON round-trip at nonzero tick, persistent-contact preservation, tamper matrix, mismatched
  tick/seed/definition/body rejection and state/snapshot preservation. Authority amplitudes are
  accepted checkpoint provenance because legacy water can evolve; authenticity is not cryptographic.
- Independent gates: changed library/test clippy clean; targeted carrier tests pass (11 authority,
  8 contact, 7 body-work, 7 body-field); full all-target suite passes with one ignored benchmark.
- Scope: carrier state only. Full future replay still requires separately restoring matching Avian
  positions/velocities, water and game state; serde/replay/contact correspondence is Rust-tested,
  not Lean/Aeneas-proved.

## Continuation summary (R10-R24)

- Lean-proved: stable diffusive energy decrease; trichotomy basis realization; nonzero-capable
  basis circulation plus discrete Stokes; equal-mass body work; finite-field locality/rollback;
  generic-ring exchange laws.
- Aeneas-proved: production `i64` impulse, elastic exchange and checked acceptance over full-width
  `ZMod (2^64)`; Penglai x/z `site_momentum` refines canonical `wordFluxFirst` with an R22 no-wrap
  bound covering actual `[-4,4]` phase differences.
- Rust-tested production: canonical Apeiron exchange consumed by Penglai; read-only live ledger;
  disabled-by-default exact carrier authority; atomic contact-entry impulse projection; validated
  carrier-only checkpoint replay.
- No issues, dependency pin changes or shipped-content enablement were performed. Commits/pushes
  were authorized after this continuation; see D8 and the final repository histories.

## Deferred decision

### Decision D6 — native corpus pin bump deferred

Applying the corpus's own toolchain/manifest bump (`lean-toolchain` → v4.31.0, `lake update`)
would make the corpus workspace fetch and rebuild Mathlib from scratch (hours, large network),
while the Apeiron bridge already provides Mathlib v4.31.0 and now builds the full corpus. The
bump is therefore deferred as a release/bookkeeping step; 4.31.0 validity is established through
the bridge harness. The corpus's native 4.30.0-rc2 `.lake` is intentionally left unbuilt to avoid
clobbering the 4.31.0 artifacts (shared cache).

## Fixed-line carrier realization

- Production kernel: `src/carrier.rs` gained `DensityLine3` and `diffusive_line3_step`, the closed
  three-cell synchronous diffusive step at `nu = 1/2`. It derives the two interior faces once each
  through `diffusive_flux_half`, fixes the endpoint faces at zero, and calls
  `conservative_density_step` per cell. Every local is bound before `Some`, so refusal is
  transactional by value; no loop, allocation, float or rounding.
- Extraction: `word_diffusive_line3_step` added to `proof/extract/lib.rs` and the Charon
  `--start-from` set; `proof/regen.sh --check` is fresh (`src/carrier.rs` `4fcdbeba…`,
  `ApeironWord.lean` `95542402…`).
- Lean: `WordRefinement.lean` proves the success characterization, the component relation to
  `scalarCarrierStep`, the two `exactHalfFlux` face witnesses, the total-mass equation (reusing
  `word_conservative_total_mass`), its zero-source corollary and the odd-difference refusal; all
  `#audit` clean in `{propext, Classical.choice, Quot.sound}`.
- Rust: nine tests in `tests/the_diffusive_line3_step_is_exact.rs` cover success, exact source
  accounting, zero-source conservation, odd and unrepresentable refusal, transactional refusal,
  the orientation control and multi-step partiality.
- Mutation control: swapping the generated left cell's shared face in a throwaway copy of
  `ApeironWord.lean` turns `WordRefinement` red; restoring it passes. The working tree is not
  modified for the control.
- Gates: `cargo test --workspace --all-features --locked` exit 0 (173 including 9 doc-tests);
  clippy `-D warnings` exit 0; `cargo fmt --all -- --check` exit 0; `cargo kani` exit 0 (60
  harnesses); `proof/regen.sh --check` exit 0; `cd proof/corpus && lake build` exit 0 (3276 jobs)
  with the closed axiom cone.
- Corpus: the pin advanced from `16112030` to `d9aefc93`
  (`d9aefc93522e2109bc96158526de6dbd52477276`), which adds
  `Apeiron.Physics.diffusiveUpdate_half_three_anyRing` (the any-ring three-cell half step) and the
  real exact quarter-energy identity `dirichletEnergy_halfStep_three` with its strict form
  `dirichletEnergy_halfStep_three_lt`. The bridge was rebuilt against it with every `#audit` in
  `{propext, Classical.choice, Quot.sound}`. The `1/4` factor needs the order of `ℝ`, so the ring
  theorem and the word refinement claim no energy, nonnegativity or ordering.
