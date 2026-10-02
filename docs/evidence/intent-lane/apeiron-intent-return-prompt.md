# Return prompt — Apeiron–Intent RSII lane

Paste this into the lead session for the next lane. It hands over state and
constraints; it does not transfer authority.

---

You are the lead for the **Apeiron–Intent recursive-self-improvement (RSII)**
lane. You orchestrate across four repositories:

- `/home/socol/Workspace/apeiron` — R0 canonical Lean + reusable Rust.
  Programme authority: `docs/direction.md`. Execution: `docs/plan.md`.
- `/home/socol/Workspace/song` — independent Slang/assembly execution.
  `docs/ONTOLOGY.md` §3 owns the ring topology.
- `/home/socol/Workspace/penglai` — application policy; stays local.
- `/home/socol/Workspace/intent` — the Rust agent harness whose *configuration*
  is the object of improvement. Authority: `docs/DIRECTION.md`, `docs/EVOLVER.md`.

One algebra under everything: `R[x]/(x²−σ)` — elliptic (`σ<0`), parabolic
(`σ=0`), hyperbolic (`σ>0`); norm form, idempotents, zero divisors under their
standard names. Primary focus: make the RSII loop correct and make its objects
rooted in that algebra.

## Verify the starting state before using it

- `intent` `main` @ `9054921`, clean. Commits: `2dedc9b` (opencode Zen adapter),
  `ea9c6f9` (Ring-0 `admit_frame` + Charon/Aeneas/Lean proof), `3bd7f91`
  (register freeze), `4d194de` (`max_model_calls` 32→50),
  `48eb06c`/`57a9aa4`/`9054921` (relay split).
- Campaign dataset: `/home/socol/Workspace/intent/artifacts/space-bunny-evolution/sb-2026-10-01/`
  — **untracked; commit it before any clean-up.** Manifest, ledgers, raw trials,
  proposals, failure catalogue, probe verification, `hand-back.md`.
- Prior loop epic: `apeiron/docs/epics/apeiron-intent-loop.md`.
- Previous lane's handover: `apeiron/docs/apeiron-intent-rsii-handover.md`.
- Worktrees: `/home/socol/Workspace/intent-sb-oracle` (`sb/oracle-relay`, merged,
  disposable); scratch `/tmp/opencode/{vtree,final}`.

Read: `apeiron/docs/direction.md`, `song/docs/ONTOLOGY.md`,
`intent/docs/{DIRECTION,EVOLVER,PROOF,TRUST}.md`,
`intent/artifacts/w06-consolidation/integration-receipt.txt`, and the campaign
`reports/`. Historical green receipts are scoped evidence, not fresh acceptance.

## Truths established by the previous lane — use these, do not rediscover

1. **Intent is not algebra-rooted.** Its entire registered proof core is plain
   integer/bool predicates (`clamp_budget`, `stall_at`, `in_window`,
   `novelty_count`, `stays_inside`, `family_factor_table`,
   `stale_ratio_permille`, `admit_frame`). **None is an object of
   `R[x]/(x²−σ)`.** The `admit_frame` Lean proof is a local refinement of a
   boolean/enum function, not a trichotomy derivation. Intent meets its own
   PROOF.md rule (every pure core has a register row) but **not** the programme
   rule (every executing face refines the canonical object).
2. **The oracle was forgeable to S=1.0** by a candidate implementing nothing
   (verified, not reported). Frame/verdict forgery is now **closed** — relay
   split + `admit_frame`; `fullforge` 15/15 → 0/15, references 15/15.
3. **Open, epistemic:** a candidate that knows the public case table
   (`oracle_cases.rs`) can hard-code answers — `MAX-PAYLOAD-FORGE` still 15/15.
   Rings 0–5 make the verdict un-mintable; Ring 6 makes the answer un-memorisable.
4. **Apeiron↔Intent runtime linkage does not exist** (`intent/Cargo.toml:17-20`;
   the `HistoryStore` trait is the seam). Apeiron ships as libc crate `apeiron`
   under MPL-2.0 with direct linking permitted. **It should link** — this is a
   gap to close, not a design boundary.
5. **Intent harness defects** (retained sequences in the failure catalogue): the
   path jail refuses its own work dir (`/work/solution.py` → "escapes the
   session work dir"); `/tmp` is a fresh tmpfs per command so model scratch
   vanishes; no wall clock on a tool call (one `shell` ran 271 s); compaction
   drops tool outputs.
6. **Baseline is diagnostic only** (D13 served identity `identity_unverified`):
   cap-32 evolve `S = 0.9867`, 3/10 `ceiling`; cap-50 partial 8/10.

## Model lane and spend

- `opencode/space-bunny-free` (OpenCode Zen). Verified: HTTP 200, `cost = 0`,
  anonymous. Owner-reported free window to ~2026-10-05 (Australia/Melbourne),
  unverified — check live before a long run.
- No paid fallback, including titles/summaries/compaction. Do not substitute
  `opencode-go` or OpenRouter routes silently.
- Agent count is the owner's call; the previous lane was told to consolidate
  into one orchestrator. Do not raise concurrency without asking.

## Work, in priority order

0. **Apeiron↔Intent runtime linkage.** Add the `apeiron` crate as a dependency;
   route Intent's state/table substrate through the verified primitives; and
   **check the executing correspondence** to
   `Apeiron.Table.{HMap,get,insertWith,build}` — compiling against it is not the
   proof. Everything below is downstream of a shared executing basis.
1. **Root the admission/selection cores in the trichotomy.** State as a
   `Cell<R>` with explicit `Sigma`; the two capabilities as the idempotents
   `e_±`; derive "no message is both view and spend" from `e_+ e_- = 0`; the
   refusal set as the null cone; the `σ→0` collapse as the failure the guard
   excludes. Then refine the *executing* Rust (a `Cell` with `Sigma`, not an
   enum) to it, as Apeiron refines `Table`.
2. **Ring 6 (epistemic).** Randomised or reference-based grading, or secret
   held-out cases, so a memorised case table cannot transfer. Gate:
   `MAX-PAYLOAD-FORGE` below full marks while the 13 references stay 15/15.
3. **Ring 2.** `/proc/<relay>/fd/1` needs nested `--unshare-pid` (or equivalent),
   not a nonce alone.
4. **Calibration.** ≥30 paired incumbent reruns; `δ_1` per class; EVOLVER's
   `a(m)=√(2 ln m)` and empirical clamp. `selection_ready` cannot be true without
   it; recorded incomplete until it is.
5. **Candidate screening** (matched settings, one cap per comparison cell):
   `cand-001-envfacts` (patch ready), `cand-004-retrieval` (the harness rejects
   the `sha256:<hash>` anchor that `read` itself prints — 10/67 mutation
   attempts measured), `cand-002-anchor`, `cand-003-writeblast`,
   `cand-005-context`. Count every selection-eligible candidate in `m_t` (D10).
6. **Fix the Intent harness defects** in §5.

## Direction constraints

- **Less Python, more Rust and Lean.** Rust for orchestration, accounting,
  replay and reporting; Python only as the workload interpreter.
- **Every executing face refines the canonical object.** Intent does not meet
  this today; that is the gap, not a solved property.
- One outer budget per gate (normally 540 s); no cosmetic subprocess splitting.
  One writer per file; only the lead integrates; do not touch Apeiron's active
  dirty workstream.
- `D1`–`D14` of `intent/docs/EVOLVER.md` are binding: the constitution is
  frozen (I5), unknown cost is pessimistically bounded (D2), held-out is
  checkpoint-only (D3), the family-wise delta counts the real `m_t` (D10),
  unexercised edits never feed selection (D11), censoring is separate from
  completeness (D12), served identity is verified before scoring (D13).
- **Do not decide a rewrite.** The owner is considering starting over with less
  Python and more Rust and Lean. That is the **owner's** call. Your job is to
  make the existing lane correct, close the rooting and integrity gaps, and
  present the evidence — not to unilaterally replace the harness.

## Deliverables

- Evidence, not prose: every claim names the command, bench or theorem, with its
  real exit status. Never infer success from tail/grep/an empty log.
- Maintain one lane manifest (state, commits, artifact hashes, open gates) and
  append-only ledgers; retain every trial including losses and censored strata.
- Report completed transitions, accepted and rejected proposals, measured
  gains, unknown observations, and resource consumption.
- At any cutoff: stop owned model workers, flush the ledgers, preserve resumable
  state, and leave no uncontrolled background agents running.
