# Post-compaction continuation — hashtrinity parity (14)

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

Supersedes `post-compaction-continuation-13.md` (retired; it superseded 12, retained as an archived
receipt at `docs/evidence/reviews/post-compaction-continuation-12.md`). Repo
`/home/socol/Workspace/apeiron`, branch `main`,
HEAD `4ae9813` (density hoist + acceptance instrument + gate hooks + `read_gate`), working tree dirty
(the red `SpillSlot` corpus migration plus cited receipts). Sibling `intent` at `3acaac0`.
**Deadline: 2026-10-05 00:00 UTC** (Space Bunny free window). Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md` before acting. Source and gates are the record; this file is
navigation, not evidence.

## 0 · Standing owner direction

**Hashtrinity-first.** Every priority-1 lane must read `../fold`'s hashtrinity **Rust and its LLVM
output**, and prove from the trichotomy basis (`R[x]/(x² − σ)`) end-to-end, until apeiron beats
hashtrinity in everything. No creative/IDE/documentation work opens before that. Do not dispatch a
lane that does not read the incumbent.

**Isolation.** Every lane delivers a **staged set plus an install command**; the lead installs. Two
lanes wrote the live trees today against instruction and broke them. Never let a worker write live.

## 0a · Base identity (re-run before every dispatch)

The live tree is the **pre-rebase** base. Two corpus lines are in play and must not be conflated — a
lane dispatched against the wrong one measures a phantom and wastes its slot.

| artefact | live (authoritative) | un-landed rebase (scratch only) |
|---|---|---|
| `proof/ApeironWord.lean` | `aaa69bedf5f799d9…`: `ResidentIndex` 0, `index_at` 0, `FaceIndex` 0, `flatIndex` 0; carries `table.Flat.index` | `90191cba8281e7e1…` in `/home/socol/oc-tmp/hyp-geom/` |
| `proof/corpus/Growable/Geometry.lean` | `627ab73df9cd8812…` = HEAD | `012bdb66…` (per-face), scratch only |
| `src/map/table/state.rs` | `fn index` at `:426`; no `index_at` | — |

- **Line A (live, active):** the `SpillSlot` corpus migration (`APEIRON-W3-T2*`), HT-FUSE-W1's
  measured read win, and the extraction route.
- **Line B (scratch, un-landed):** the algebraic rebase (`ResidentIndex`/`FaceIndex`/`index_at`) —
  `APEIRON-HYP-*`. It cannot be graded until its extraction lands and its `Growable/Geometry.lean`
  section rebases onto Line A's `Growable/Geometry.lean` (**one file, one owner**).

A dispatch states which line it edits and carries `python3 tools/rooting/read_gate.py --receipt`.

## 1 · Lane state at handoff

| lane | state |
|---|---|
| `APEIRON-G1-REGRESSION-GATE-W1` | running — add regression-vs-registered-row check + seeded control |
| `APEIRON-HT-PROBE-W1` | running — spill-probe attribution + **owes the registered-row reconciliation** |
| `APEIRON-HT-FREQ-IR-W1` | running — `FreqMap`/`Coord`-fold IR for the spatial cells |
| `APEIRON-W3-T2c` | running — ReadLoop/LocateLoop probe layer + loop induction |
| `APEIRON-W3-T2d` | running — Growable Identity wave |
| `APEIRON-W3-T2e` | running — Classifier schema completion |
| `APEIRON-CORPUS-AUDIT-W1` | running — AxiomAudit registration repair |
| `APEIRON-TRAIN-GRAM-W1-T3` | running — gram-fold extraction link |
| `APEIRON-HYP-DERIVED-PIN-W1` | running — derived arm pinned to generated body |
| `APEIRON-GPU-COSTBAR-W1` | running — resolved-term rule replacing the 2×2 bars |

Parked/deferred: `INTENT-BOARD-JOIN-W2` (needs 5 files, `LaunchRequest` frozen across 4 unowned test
files; fix = separate board map + `tick_with_board`), `INTENT-IDE-DESIGN` (creative — waits on parity).

## 2 · Fresh measurements (11:22 UTC, load 1.75)

- **G1:** `--profile full --accept --controls` exit 1, **641 pass / 859 fail**, 1500 applicable,
  concurrency 6, address 6, `full_acceptance=false`.
  - worst serial hashtrinity `occ1-direct-index-resident-admitted-fresh-insert-replacement`
    **5382 / 1028 ns = 5.2354×** (stable: 5.26× on the prior run).
  - worst spatial `spatial/n10000` vs `hashtrinity_get_each` ~5.48×; `spatial/n1000000` vs
    `fxhash_par16` ~5.5×.
- **Intent:** `w03b_live_session` 1/1, `mcp_transport` 13/13, `fleet` 27/27, `budget_fleet` 9/9.

**Open quantitative question (do not paper over):** `examples/reg1/matrix.json` registers that same
serial cell at **1653 / 1339 ns = 1.2345, `within-floor`** at revision `c450c222…`, while the live
acceptance reports **5.2354×** against baseline `4cc14471`. The incumbent barely moved (1339→1028);
the candidate is ~3.3× slower than its own registered row. Either a **candidate regression since
`c450c222`** (bisect: reserve-before-first-merge, cold-registry, density hoist) or a **protocol
difference**. `APEIRON-HT-PROBE-W1` must answer it with both commands before any serial ratio is
quoted. Nothing currently compares the candidate to its own registered rows — that is
`APEIRON-G1-REGRESSION-GATE-W1`.

## 3 · The measured work order (from incumbent IR)

`APEIRON-HT-LLVM-W1` receipt `/home/socol/oc-tmp/ht-llvm/receipt.md`:

| operation | incumbent `4cc14471` | apeiron |
|---|---|---|
| resident hit | **10 instrs, 2 loads** — one `sc == coord` decides liveness+identity+hit | **29 instrs, 4 loads** (`live`, per-slot σ, key `matches`, value) + `mul` + 5 `checked_*` + bound |
| fresh insert | **111** | **836** (7.5×) |

Ranked: **1)** fuse {liveness, identity, hit}, drop per-slot σ and key compare — `APEIRON-HT-FUSE-W1`
running; **2)** `reserve_locator_batch(1)` + `record_admitted_link` (`src/map/table.rs:1316`, `:1356`), no
incumbent counterpart; **3)** per-insert `u32::try_from` + σ plumbing.

Incumbent identity: G1 compares to `4cc14471`, not fold HEAD `d497161`; `freq/face/par/dense/algebra`
are byte-identical, `lib.rs` differs by 2 lines.

## 4 · Landed today (lead merges, uncommitted)

- **apeiron:** density hoist (`src/map/table.rs` `+=` extraction-clean, `state.rs`, `tests.rs`,
  `proof/ApeironWord.lean`) — `regen.sh --check` fresh; gate fixes (`tests/table.rs` oracle,
  `src/map/table/locator.rs` injection test); acceptance instrument + **fair timed region**
  (`examples/reg1/{acceptance.rs,adapters/spatial.rs,acceptance/registry_tests.rs}`), spatial worst
  36.55× → ~5.3×; corpus `Growable/{Scan,Relocation,Search,Representation,Refinement,Resources,ReadState}`
  **modified in the working tree — uncommitted and still red** until their lanes land (only `Identity`
  is measured green; `Geometry` is clean at HEAD, i.e. no per-face rebase is landed);
  `Native/HyperbolicFaceNullCell.lean` + root + 33 `#audit` rows, **gated green** (3351 jobs).
- **intent:** inbound MCP `intent_state` + `intent_return` + `intent_decide`; durable-reservation
  interface (`payloads.rs`, `fleet_ledger.rs`, `budget_lease.rs`, `fleet.rs`) — **629 passed / 0 failed**.
- **tooling:** `.githooks/{pre-commit,commit-msg,pre-push}` + `tools/hooks/{lane.py,install.sh,README.md}`
  (installed; `lane.py self-test` 9/9) + `tools/board/board.py` driver; `AGENTS.md` §1a rows added.
- **board:** 8 milestones + 14 milestone-bound tasks in Operately `apeiron` project
  (`apeiron-ExNAtv3hObMmq01Ab20DfA`).

## 5 · Open / broken

- Live corpus still red: `Growable.ReadLoop`/`LocateLoop` (T2c), `Identity` (T2d), `Classifier` (T2e),
  and `Numeric.AttnLoopOutput` / `Policy.*` — so `lake build AxiomAudit` is red and **no corpus axiom
  cone is currently machine-checkable**.
- `proof/corpus/Native/HashtrinityBasis.lean` — written live, **does not compile** (12 errors,
  `Invalid 'end'` at `:423`), **unrooted** and not imported; needs a real delivery + gate. It was never
  green anywhere and no backup exists.
- Staged, not installed: `intent-wire` patch (3 files, gate 632/0) → reserve-before-launch;
  `extract-root` patch (`regen.sh`, `lib.rs`, `README`, `ApeironWord.lean`).
- `matrix.json` registered revision `c450c222` vs acceptance baseline `4cc14471`; reconciliation open.

## 6 · Gotchas measured today

- `find` is a broken shim here; use `rg --files` / shell globs. Absence claims from `find` are unsound.
- `update_task_status` / `create_task` return a response-schema error **for writes that land** — always
  read back.
- Bare `lake build A B C` can silently expand to defaults; use `+Module` names.
- **`APEIRON_LAKE=0 proof/regen.sh --check` skips the Lean build** — "extraction is fresh" says nothing
  about whether the corpus elaborates against it.
- Timed runs need a **private** `CARGO_TARGET_DIR` + forced rebuild; scratch `target/` symlinked at the
  live tree contaminated earlier timings (2.02 vs 9.26 ns/key). See
  `/home/socol/oc-tmp/shared/BUILD-CACHE.md`.
- No dispatch into a tree whose barrier has not been read green on that same source identity.

## 7 · Next move

1. Receive the spill-probe **reconciliation** and the **regression gate**; quote no serial ratio until
   then.
2. Land the staged patches; get `AxiomAudit` green by landing T2c/T2d/T2e and the audit repair.
3. Attack the ranked work order above; each change carries its extraction/refinement, not just a ratio.
4. Keep 10 lanes; fill only with hashtrinity-first foundational work; lead installs every staged set.
