# Post-compaction continuation — programme dispatch

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md` (E1–E6), `docs/reviews/multi-repo-fleet-brief.md`.
Source and gates are the record; a document is corrected to them. Live mirror:
`/home/socol/oc-tmp/fleet-state.md`. Escalation verdicts: `/home/socol/oc-tmp/escalation/`.

## 1 · Routes and discipline

- **Workers**: `openrouter/stealth/space-bunny-alpha` via `agent: general`, free until **2026-10-05**.
  `deepseek-flash` is the paid lane on a mandatory trigger only; the owner authorized it once, for
  the webfold PR merge.
- **Sol**: `/home/socol/oc-tmp/bin/sol.sh <prompt-file> [pro|<model>]`, answers under
  `/home/socol/oc-tmp/escalation/`. Default `openai/gpt-6-luna-pro` ($0.10/Mtok in, $0.50 out).
  **The lead does not idle waiting for the owner — escalate, record, act, report.**
- ≤3 simultaneous heavy Lean checks under `flock /tmp/opencode/apeiron-lane.lock`.
- Every gate declares a numeric wall-clock budget; 600 s is a failure ceiling.
- **`find` and `grep` are broken shims** (`~/.local/bin/grep` symlinks `rg`, where `-r` means
  replace; `find` is a 2009 binary that returns nothing on a populated directory, exit 0). Use `rg`,
  `rg --files`, `/usr/bin/find`. A claim of absence names its command. The owner's answer on this:
  **a permanent architectural fix, not a patch** — recorded in `docs/plan.md` under "Enforcement is
  Rust"; no gate may depend on a `PATH`-resolved conventional tool.
- Song's commit hook: subject ≤72 chars with an `<area>: ` prefix, **one paragraph of ≤12 lines**,
  bans phrases including "not yet", and flags ` -- ` used as an aside. Write messages to a file and
  `git commit -F`.
- **zsh aborts a command line whose glob matches nothing** — `rm -rf .review-*` silently did nothing
  and cost a push. Use `setopt NULL_GLOB` or check first.

## 2 · Owner decisions in force

1. **Kalman reading: yes** — belongs in provenance, formalise when suitable. Task in flight.
2. **PATH shims: permanent architectural fix** (`c00b0b6`), not a local patch.
3. **Sustained width is a requirement on Intent** (`c00b0b6`), **testable within 48 hours**
   (`0eedb33`): 50 concurrent workers, per-worker identity, a receipt each, and a killed worker
   re-leased rather than lost. Harness in flight.
4. **webfold's callers are untrusted** (hosted crawler on a rented box) — this selects the untrusted
   branch of Sol's SSRF answer: the initial destination is address-checked too, the direct-loopback
   row is a vulnerability, and any test authority must be server-side configuration unreachable
   from anything a client sends.
5. **The hashtrinity gap is the highest-value item** — the conversion was never done; Penglai's
   fluid simulation depends on the table and 9× is "runnable on a potato". Now **`docs/plan.md` E6**
   and the roadmap item, committed `d05e22d`. Waves: W1 attribution, W2 the port, W3 refinement,
   W4 the gate of record.

## 3 · Measured state that decides work

**The table gap** (`cargo run --release --features read-stats --example table_cost_probe`, i9-14900KF,
192 keys, `N = 16`): present lookup **9.73 ns/key vs hashtrinity's 1.04 (9.35×)** on the **plain**
baseline; 31.52 vs 2.12 low-byte; 292.35 vs 0.78 adversarial. Insert 480.98 vs 4.04 but that column
is **allocation-dominated and not evidence about the probe** — and the insert locators carry no
counter, which is its own task. On the distinct baseline the table does 1.00 sweep step, 1.00 tag
match, 1.00 full compare per key, so **the 9.73 ns is outside the probe loop**; E6.W1 attributes it
by building variants, not by argument. The adversarial column is a **filter** defect: `tag(place)`
reads the same low 16 bits the shard selector does, so the population matches every lane (22,032 tag
lanes for 1,377 steps) while group steps stay bounded at 7.17. A **mixing** home fixes it
(`key ^ (key>>16)`, 292 → 9.95 ns); a multiply-only home does not (2-adic valuation preserved).

**Aeneas prepass**: the limiter is loop **shape** (a `return` crossing the function boundary from
inside a loop nest), not `?` or `Result`; **`zip`** is a second independent blocker. `KvState::append`
is reshaped and cleared (`8fc50a5`). `attn_head_f32` still refuses (`Breaks to outer loops`).
**`f32_core.rs` is blocked by the value domain, not shape — the Aeneas model has no float type at
all**; it needs either a float-capable model or an f32-through-`u32`-bits rewrite. That decision is
**open and is the owner's or Sol's**.

**Rule D** (`tools/rooting/check.py`, named in the enforcement map): 477 findings at HEAD → **31**
(D 472 → 26) with the committed fix (`3aa8ee4`); the diagnosis predicted 7 genuine, so 26-vs-7 is
**unreconciled** — a task is classifying all 26 and finishing the "does a bare numeral count as a
measure" question the interrupted session left open.

## 4 · Landed since the restart

- **apeiron** (11): `c00b0b6` shim + fleet-width decisions · `0eedb33` width testable in 48 h ·
  `5b7246c` failure register gains the inert-gate instance and the rule "a gate is not accepted until
  its refusal has been observed firing under the harness that runs it" · `1857194` evidence index ·
  `4030c95` LmHead prose · `d942939`+`5b71476` the Lean-model route named · `3aa8ee4` rule D ·
  `074ff79` the table-cost harness · `8fc50a5` the append reshape · `d05e22d` E6.
- **song** (18): the merge is a **compile-time name** (`e9583471`) · `intern.slang` rebased onto the
  canonical table (`83055a6f`), with first-wins justified and `keepFirst`'s vacant-path hazard found
  · the adversarial probe (`1b0188ab`) · the symbol gate made able to **fire** (`5263e8d4`) · A5.1 ·
  grantGet laws · a59 closed · hashdos · sliced-grant fixture · merge docs.
- **penglai** (8): three-axis carrier · citation resolver out of `/tmp` with a symbol-clause census
  (1 of 26 → 21 of 26 inspected, which then caught a **real bad citation**) · six module headers ·
  globe/margin kernel + fixtures.
- **webfold** (8): units checker names its monitor; ships the monitor it defaults to; wasm handles
  bound to an origin and declared limits honoured; `lift.rs` refuses what has no faithful Term
  (ternary residue still open); the CLI module split shipped; ledger CORE-15 + RT-13/14.
- **webfold-bench** (4) and **intent** (3): path repairs, READMEs; D2 enforced cap + D4 class floors,
  Ring 3 leaf (forge 15/15 → 0/15), Ring 6 D4 arm.

## 5 · webfold: the push is blocked, precisely

PR #6 is **MERGEABLE**, remote `ef5eb17d`. Local is ahead at `2bb62360`; the merge and push are
ready but **gated on a green build**, which fails only because `review_bundle.py --materialize` is
**not resumable**: after an interrupted run the next invocation dies with
`FAIL: [Errno 17] File exists: '.review-…'` (or `No such file or directory` once the temp dir is
gone), so the `include_str!` fixtures under `benchmarks/review/` are never complete and
`cargo check --all-targets` fails on missing fixtures. A fix task is in flight; when it lands, run
the merge in `/home/socol/oc-tmp/wf-merge2`, materialize, `cargo check --workspace --all-targets
--offline`, and push **only if exit 0**. Worktrees `/home/socol/oc-tmp/wf-merge{,2}` exist and are
disposable.

Also open in webfold: the merged tree's six loopback-test failures (3 in `webfold_mcp.rs`, 3 of 4 in
`fetch_metadata_review.rs`) are correct behaviour under untrusted clients and need the server-side
destination policy; `proxy_metadata_…` failing on `metadata.example.invalid` is **pre-existing and
unrelated**; `check_repair_ledger.py` still fails on `CORE-16` (index 15, our rows are at index 42),
then 36 stale finding rows, then 26 of 753 `source_files` byte counts.

## 6 · In flight at compaction (10 workers)

Kalman provenance (Song) · Intent 50-worker fleet harness · table fingerprint independence ·
`attn_head_f32` reshape · Penglai headers slice 4 · rooting "remaining 26 + numeral decision" ·
Intent cost-interval bypass closure · review-bundle resumability · SSRF policy under untrusted
clients · lift ternary/assignment.

## 7 · Next actions

1. Integrate returns; re-run each gate before committing; keep other owners' hunks unstaged by
   constructing the staged blob as HEAD plus the new lines (`git hash-object -w` +
   `git update-index --cacheinfo`) — three files today carried foreign WIP.
2. Dispatch **E6.W1** (attribution of the 9.73 ns) — the owner's highest-value item.
3. Decide the **f32 value-domain** route (float-capable model vs `u32` bits) — owner or Sol.
4. Unblock the webfold push once the materializer is resumable, then finish the SSRF policy so the
   six loopback tests are proven against the deployment model.
5. Reconcile 26-vs-7 on rule D, then fix the genuine sites (three files, one line each).
