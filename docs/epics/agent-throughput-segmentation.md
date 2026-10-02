# Agent throughput segmentation plan

Status: proposed operating plan, no gate run by this document. Owner: lead. Basis: measured
repository rates from `git` (see the 2026-09-24 throughput measurement in this session), not task
budgets. Deadline target: July 2027, with 3x throughput against the measured baseline.

```text
SCOPE:      lane map, queue protocol, interface freezes, file-split wave, milestone backcast
REGISTER:   planning/execution; excludes claim grades, gate results and scope promises
CORPUS:     none - no Lean theorem is claimed here
REFERENCE:  measured rates: apeiron 279 commits/14 active days of 27; song 1845 commits/79 of 86;
            penglai 48 commits/8 of 16; sanguine-apeiron bursts up to 146/day
GATE:       task-ledger closure counts per week (to be instrumented); no compiler gate
GRADE:      anchor - a scheduling reading of measured rates, not evidence of correctness
```

## 1 · Measured baseline (2026-09-24)

| repo | span | commits | active days | duty | notable |
|---|---|---|---|---|---|
| apeiron | Aug 29 → Sep 24 | 279 | 14 of 27 | 52% | 152 Lean modules, 506 audits; read rows 1–5c, 6 partial, 7 partial; G1 308/1450 |
| song | Jun 20 → Sep 14 | 1,845 | 79 of 86 | 92% | K0–K14 done; A0 6 implemented / A0.4 partial / A0.6+A0.9 blocked / A0.10 open |
| penglai | Sep 8 → Sep 24 | 48 | 8 of 16 | 50% | playable Bevy slice; substrate milestones 2–5 closed |
| sanguine-apeiron | May → Sep 24 | 2,253 | bursts | — | 154 canonical modules; lands on demand |

Song already runs near the target operating envelope: numbered items, file-disjoint modules, fast
local gates (`./song check/test/controls`), high commit cadence. Apeiron's gates are comparably fast
now (incremental Lean builds 2–5 s); its gap is concurrency, task readability and serialization, not
verification cost.

## 2 · Levers and expected multipliers

| # | lever | measured basis | multiplier |
|---|---|---|---|
| 1 | Concurrency 2–3 → 6–8 file-disjoint lanes | per-lane speed roughly unchanged in the 2026-09-23/24 wave; agents are API-bound, builds are seconds | 2.0–2.3x |
| 2 | Duty cycle 52% → 85–90% (queued 24/7, never idle) | apeiron 14/27 vs song 79/86 | 1.6–1.7x |
| 3 | Shovel-ready task specs (exact goal state, adjacent lemmas, file map, expected diagnostics) | recent partial slices hit budget mid-exploration (row 6 twice, W7-T3 pending) | 1.3–1.5x |
| 4 | Remove serialization points (per-module audit files, lakefile globs, batched controls, frozen extraction per wave) | every new module currently edits `AxiomAudit.lean`/`lakefile.lean` through one writer | 1.1–1.2x |
| 5 | Decomposition + independent verification pair + kernel arbiter (replaces the Sol-class lever) | the OpenAI lanes are decommissioned and no separate escalation model is available; contract/proof verdicts run through decomposition and independent verification, and Gemini Flash (Vertex) carries the high-throughput work; the hard row-5–7 slices landed under decomposition | 1.1–1.3x |
| | product before Amdahl | | 5–8x |
| | realised estimate after integration-lane and dependency-chain caps | row 8 after row 7; A1–A4 after A0; contract/proof verdicts run the decomposed-verification protocol | **2.5–3.5x** |

The realised 3x requires all five; dropping one (for example skipping independent verification)
keeps the programme at ~1.5–2x because contract/proof verdicts queue.

## 2.1 · Roster (measured availability, 2026-09-24)

| model lane | cost | role | constraint |
|---|---|---|---|
| `gemini-flash` (Vertex, Gemini 3.8 Flash) | prepaid Vertex balance | default producer (fastest, ~300 tok/s): edits, scans, proof slices, verification replay | balance is the binding budget; Claude on Vertex is out (Messages endpoint rejects API keys; the ADC path did not authenticate) |
| `deepseek-flash` | ~$0.27/task | conserving speed lane: mechanical edits, splits, audit moves, verification replay | lower proof-search ceiling; pair with exact specs |

With Gemini Flash as the default, spend draws the Vertex prepaid balance rather than per-task
billing; at the recorded ~$1.24/task rate the reported $1,230.34 covers on the order of a thousand
tasks, so the balance is a governed budget and DeepSeek is the overflow lane. The binding
constraint remains dispatch and verification throughput, not model spend. Verify the balance before
relying on the number.

## 2.2 · Hard-node protocol

1. Hard nodes are decomposed before dispatch into casework with exact goal states, adjacent lemmas and
   expected diagnostics. A node that cannot be decomposed is a lead decision, not an agent task.
2. Every headline gets an independent verification session on a different model from the writer; the
   Lean kernel and the mutation control are the arbiters, never the writer's report.
3. Contract and claim-grade questions queue to the lead; every headline gets an independent
   verification session on a different model. They do not block the worker lanes.
4. A node that fails three materially different decomposed approaches becomes a spike with a named
   blocker and a 45-minute cap (existing rule), not a repeated retry.

## 3 · Lane map (file ownership; one writer per file)

Integration lane (serial, lead): `proof/ApeironWord.lean` regeneration, package roots, root audit,
barrier runs, wave-barrier controls. Everything below is file-disjoint by construction.

Apeiron, after Wave 0 splits:
- `R1` read stored partitions (`Growable/Read/Stored.lean`) — W5-T13 family
- `R2` read regen partitions (`Growable/Read/Regen.lean`) — W5-T13 family
- `C1` classifier decision fields (`Growable/Classifier/*.lean`, per field group)
- `A4` row 8 per branch (`Growable/Apply/FoundFlat.lean`, `FoundSpill.lean`, `VacantFlat.lean`,
  `VacantSpill.lean`, `Exhausted.lean`) — 5 writers once the interface is frozen
- `G1` Rust performance cells (`src/table/*`, one phase per lane) — measurement lane only on a quiet
  machine
- `SEC` security transport (`Security/*`)
- `ALLOC` allocator (blocked on contract)

Song lanes (its own gates): `S1` A0.6 emitted stream, `S2` A0.6.P3/P4, `S3` A0.10 read groups,
`S4` correspondence with Apeiron, `S5` K2.1 driver (secondary). Penglai is its own repo and its own
queue; content production is not agent-bound.

AI architecture lane (`AI-ARCH`, own harness on the RTX 5080; independent of the corpus lane):
- small-model mechanism experiments only: forward-only training, test-time training, recurrent and
  state-space topologies, measured on fixed task suites with capability-per-FLOP and per-watt as the
  metric; no pretraining-scale work and no data pipeline.
- every candidate topology carries one mechanism lemma from the algebra corpus (norm-form growth
  bounds, exact scalar arithmetic, update-operator invariants) with a mutation control on the
  mechanism, so the architecture search is guided by proofs rather than by guessing.
- the verified stack supplies checkers, exact arithmetic and certified search; results are reported
  as measured capability per compute, never as frontier parity.
- prior art and the staged order: [Verified-AI sequence and prior-art index](../roadmap.md#verified-ai-sequence-and-prior-art-index)
  (paper1, paper2, spinor-ttl, the Neural corpus). The Neural rebase is stage 2, not tail-end;
  historical results keep their scopes and are not fresh acceptance.

## 4 · Queue protocol

1. Every dispatch carries: exact owned files, the exact statement to prove (goal state), adjacent
   lemmas to reuse, the gate command, one control, a 30–45 min budget, and the expected first red.
2. Two staged tasks per running task; the queue never empties. Lead or an orchestrator agent
   refreshes the queue at every return.
3. Tasks are ≤30 min atomic wherever possible; a wave batches 4–8 atoms and runs controls once per
   shared gate, per-task controls only where soundness is at stake.
4. Interface-first waves: freeze the structure (`RepR`, `SearchDecisionCorrect`,
   `RelocationEffectsPrepared`, plan model), then producers run in parallel against it. No producer
   starts on an unfrozen interface.
5. One Rust/extraction change wave at a time; regenerate once per wave and freeze the generated
   digest for all proof lanes until the next wave.
6. Evidence discipline unchanged: digests, controls, cone, audit registration per landed headline.
7. A lane's gate is run only when its dependency modules compile; a blocked gate waits for the owning
   lane instead of acquiring the lock over a mid-edit tree. Lanes keep their edit sessions compiling
   at their checkpoint where possible.

## 5 · Wave 0 (interface and serialization removal, immediately after the in-flight tasks return)

1. Split `Growable/Read.lean` by carrier/partition; split `Classifier.lean` per field group; split
   row 8 per branch. Names/statements preserved; controls re-anchored to the frozen extraction.
2. Per-module audit files under `proof/corpus/Audit/`; the root `AxiomAudit.lean` imports them, so
   lanes stop queueing on one registration writer.
3. Lakefile roots via globs (or a documented root convention) so new modules stop editing one file.
4. Freeze decisions: repair `RelocationEffectsPrepared.publish`; locator well-formedness as the
   `hstep` supplier; G1 acceptance protocol unchanged.
5. Roster to 6–8 concurrent sessions; route hard nodes through decomposition plus an independent verification session.

## 6 · Backcast to July 2027 (3x)

| window | apeiron | song | other |
|---|---|---|---|
| now–Oct | Wave 0; read P2–P3 | A0.6 stream probe; A0.10 children | penglai content; G1 cells on quiet machine |
| Oct–Dec | classifier; row 8 branches; core barrier | A0.6.P3/P4; correspondence draft | security transport; extraction waves |
| Jan–Mar | core DoD (G1 last) | A1/A2 slices; correspondence matched domain | bounded translation pipeline; T6/G8 inference probe |
| Apr–Jun | closure and audit | A3; A4 freeze candidate | AI learning-update probe; penglai sellable slice |
| Jul | deadline review against the exit table | | |

Scope statement at 3x: apeiron core, Song A0–A4 plus correspondence, a bounded verified translation
domain, a bounded end-to-end verified inference/learning probe, and a penglai slice fit the window
at the optimistic end. Full frontier-capability verified AI does not fit the window at 3x from the
measured base; it stays research-gated, and the July target is the functional, workable artifact.

## 7 · Open decisions for the lead

1. Vertex lane: Gemini Flash runs on the prefunded balance ($1,230.34 reported). Claude on Vertex is
   out (the Anthropic Messages endpoint rejects API keys and the ADC path did not authenticate), so
   escalation is the decomposed-verification path at 2.5–3.5x.
2. Concurrency budget: 6–8 sessions on one host, or add a second host for Lean/compile lanes; the
   cheap roster puts 30–60 tasks/day in the $5–20/day range.
3. Scope confirmation of the July deliverable set in §6.
4. Authorization for the Wave 0 file splits and audit/lakefile restructuring (paths and digests
   change; controls re-anchor once).
