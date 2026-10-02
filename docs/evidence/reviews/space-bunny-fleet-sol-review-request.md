# Review request — deployed agent fleet and the forward schedule (independent direction)

To: independent verifier / director (OpenAI Sol).
From: Apeiron lead.
Type: architecture escalation under `AGENTS.md` §9.1. This document is the handoff; it carries no
history and claims nothing on its own.

## Decision requested

Two things, in order:

1. **Review the currently deployed fleet** — the ~20 `general` subagents running a Space Bunny Alpha
   wave on the Apeiron cost model — and its orchestration: task decomposition, gate shape, lane
   schedule, integration protocol, and observed failure modes.
2. **Confirm (or replace) the forward schedule** for the Space Bunny fleet from now to the
   **2026-10-05** cutoff, when Space Bunny Alpha stops being free. Direct which tasks run, in what
   order and batch sizes, on which lanes and models, with what gates and acceptance — to extract the
   most durable value before the cutoff.

Space Bunny Alpha is **free but mid-tier** (cloaked weights, MMLU-Pro ≈ 75%, fast, token-efficient).
It is not the most capable model available; it is the only high-throughput lane currently free. The
schedule should use it where parallelism + bounded verifiable tasks pay, and escalate where they do
not.

## Deadline economics

Today is **2026-10-01**; the free window ends **2026-10-05** (≈4 days). Value before the cutoff =
verified, committed, composable artifacts — not dispatched tasks. Every task that does not reach a
green gate + lead commit by the cutoff contributes nothing durable.

## Where

Repo `/home/socol/Workspace/apeiron`, `main`, HEAD `2bdb66d10a443f7ea23433da20091a3c2464ecce`, Lean
`v4.31.0`. Read `AGENTS.md` (authority), `docs/direction.md`, `docs/kernel-boundary.md`,
`proof/README.md`, `docs/epics/resource-guided-evolution.md` (the cost-model epic), and the template
`proof/resource/ActScaleResource.lean` + `proof/resource/gate.py`.

## Objective of the wave

Build the **source-cost model** for the executing primitives: for each real extracted body, an
Aeneas extraction + Lean resource theorem (loop transition; strictly-decreasing rank or closed-form
work count; exit/refusal branch; a reachable success witness) + a gate carrying a **real-source
mutation** that must go red naming a headline and restore green. These per-primitive counts are meant
to feed `proof/resource/CostAlgebra.lean` (composition + dominance) so that an optimization can be
certified as a source-connected strict improvement — the model the programme iterates on.

## Committed and green (verified by the lead on HEAD)

| artifact | gate |
|---|---|
| `CostAlgebra.lean` (`49ca93a`) composition + fieldwise dominance + nonvacuous witness; `dominates_improves` gives the strict `Nat` reduction `DispatchAdmission` needs | `CostAlgebraGate.py --check --controls` exit 0, 10.2 s; 3 defects red-naming-headline, restored |
| `RowDotResource.lean` (`2bdb66d`) Θ(n) checked i128 multiply-adds in n+1 visits for the extracted `matvec::row_dot_fixed` | `RowDot_gate.py --controls` exit 0, 14.6 s; real `matvec.rs` mutation red, restored |

## Deployed fleet under review (this session, `agent: general`, `model: openrouter/stealth/space-bunny-alpha`)

| task (target = extracted root / source) | on disk | status |
|---|---|---|
| cost-algebra (`CostAlgebra.lean`) | yes | committed green |
| row_dot (`matvec::row_dot_fixed`) | yes | committed green |
| rms_norm, argmax, f16_split, fixed_exp, fixed_rsqrt, gguf_parse_header, policy_decide, route_topk, silu, capacity_bound, q6_k_codes | Source+Resource+gate | produced; **not yet lead-verified** |
| rope_rotate, fixed_sin_cos, fwht | Source only | partial (fwht redo in flight) |
| kv_append | gate only | partial |
| exact_dot, matvec, packing_decode_block, i64_codec | none yet | running / not produced |
| shader epilogue (`proof/native/ShaderEpilogue.lean`, worktree `wt-w7-epilogue`) | partial | replanned after `xiaomi-pro` ran out of credit |

Orchestration: 4 Lean lanes (`apeiron-lane.lock` + `-1/-2/-3`), each `flock` + `systemd-run
MemoryMax=12G/MemorySwapMax=0` + `LEAN_NUM_THREADS=1` + `timeout`, one gate per task ≤540 s; one
writer per file; each task owns 3 new files under `proof/resource/`; lead verifies every gate and
commits. Machine: 32 cores, 62 GiB, 48 GiB available.

## Backlog not yet dispatched

`q4_k_codes`, `i64_codec_decode`, `check_fragment`, `lm_head`, `attn_head`, `kv_state_new`,
`packing_decode_block`, `schedule_is_complete`, and the table roots: `coord_{finish,spread,unspread,
id,place}`, `stored/regen_relocate_{entry,all}`, `apply_pending_and_commit`, `publish_prepared`,
`prepared_shard_{get,apply}`, `table_{get,build}_at`, `policy_decide` siblings; then the connecting
theorems: source-derived counts → `Cost` fields, the universal `Admission.preserves`, and the epilogue
hoist certificate.

## Discovered blocker (weigh into the schedule)

`proof/extract/lib.rs` is stale against `src/engine/attn.rs`: the extraction unit no longer compiles
(`crate::engine::mixer` missing, `f32_core` path unresolvable), so `proof/regen.sh --check` cannot
pass and the committed attn/KV headlines refine a body that no longer exists. Details and repro:
`docs/evidence/attn-extraction-drift.md`. Therefore **do not schedule `attn_head`, `kv_state_new`,
`kv_append`, or any attn-family cost task** until the root is repaired; and a second, separate
Aeneas limitation ("returns inside nested loops") blocks `KvState::append` even then. The non-attn
roots (matvec, table/resources, numeric, format) are unaffected.

## Observed failure modes (weigh these)

- One `general` worker returned with no text and an incomplete artifact pair (`FwhtSource.lean` +
  gate, no `FwhtResource.lean`); re-dispatched. Empty handbacks happen.
- Some workers begin with scratch files in the owned path (e.g. a probe-only `Resource.lean`) before
  the real proof; the lead must reject non-proofs.
- Lane waits: with 4 lanes and ~23 tasks, gate runs bunch and can exhaust a 540 s budget on
  `flock` wait alone (`LANE_BUSY`). Batching matters.
- `xiaomi-pro` (the intended proof-ceiling fallback) is out of credit; there is no strong fallback.

## Questions

**R1 — Deployment review.** Given the roster, lane schedule and gate template above, is this a sound
use of a free mid-tier frontier fleet? Name any task that is the wrong shape for Space Bunny (too
hard, too coupled, unverifiable) and any that should have been batched, serialized, or escalated.

**R2 — Gate sufficiency.** Is the per-primitive gate (extraction + resource theorem + one real-source
mutation) sufficient to accept a cost count, or does it need a stronger control (e.g. a count-mutation
that changes the closed form, a refusal-reachability witness)? The current weaknesses are known:
space bounds are not claimed, refusals are conditioned equalities not reachable refusals, and the
counts are not yet connected to `CostAlgebra`.

**R3 — The forward schedule.** Produce the schedule for 2026-10-01 → 2026-10-05: an ordered
allocation of the backlog across lanes/models with batch sizes, per-task gate and acceptance, and a
stop rule. State what to run on Space Bunny, what to escalate off it, and what to drop entirely as
not worth the remaining free budget. Include the integration/commit cadence and the expected
committed output per day.

**R4 — Value ordering.** On your schedule, what is the single highest-value artifact that can be
committed by the cutoff, and what is the smallest set of tasks that makes the cost model *usable to
iterate on* (source-derived count → `Cost` → dominance → `Admission.preserves`)?

**R5 — Escalation triggers.** Which tasks must NOT be handed to Space Bunny (independent-verification
triggers: changing a contract premise, a theorem, a claim grade, a canonical-corpus theorem; or
exceeding 3 files / one gate; or overlapping another task)? Name them.

**R6 — Cutoff risk.** What in the current wave is most likely to waste the free window, and what is
the minimal change to the orchestration that raises the committed-artifact yield before 2026-10-05?

## Required return

- A findings list for R1–R2: artifact/line where possible, and a counterexample where the deployment
  is wrong.
- For R3: a concrete allocation table (task → model → lane → batch → gate → acceptance) covering the
  backlog, in order, with the stop rule.
- For R4–R6: the named highest-value artifact, the escalation list, and the one orchestration change.
- A verdict: is the deployed wave sound to continue as-is, or does the schedule change first? Do not
  waive a gate or a §4-pillar obligation.

## Disclosed deviations

Lanes widened 1→3 (12 GiB each, within 48 GiB available) for throughput. Artifacts authored by
free/other-model workers; the lead re-runs every gate. `xiaomi-pro` lane out of credit. The
cost-algebra worker reported a possible mid-slice commit by another actor; the lead checked `reflog`
and tracked status — only the lead's own commits, modified files intact.

## Not claimed

No hardware speed or codegen cost; the per-primitive counts are source-level work counts, not
latency; the `Admission` link in `CostAlgebra` is a text+digest binding, not a Lean elaboration; the
§6 epic barrier was not run for these slices.
