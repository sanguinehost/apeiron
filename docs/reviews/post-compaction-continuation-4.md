# Post-compaction continuation — programme dispatch (clean restart)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md` (owner layer:
§Authority/hold/momentum, §Escalation ladder), `docs/direction.md`, `docs/kernel-boundary.md`,
`docs/plan.md`, and `docs/reviews/multi-repo-fleet-brief.md` (§9 programme phasing, §9.1–9.3
decisions). Source and gates are the record; a document is corrected to them.

## 1 · Worker lane — the thing that was wrong

**Default worker: `openrouter/stealth/space-bunny-alpha`.** Free, OpenRouter, and it is the lane the
owner uses. **The free window ends 2026-10-05.**

Measured, same prompt, 400 tokens (both routes live now):

| route | wall | TTFT | tok/s |
|---|---|---|---|
| `openrouter/stealth/space-bunny-alpha` | 2.70 s | 1.19 s | **147.9** |
| `opencode-go/space-bunny-free` (OpenCode Zen) | 6.95 s | 6.95 s | 57.5 |

The OpenCode Zen route is the same family but slower — do not default to it. Other free Zen ids
exist (`mimo-v2.5-free`, `mimo-v2.6-flash-free`, `nous/nemotron-3.5-lightning-free`,
`muse-spark-1.2/1.3-contributor-free`, `longcat-2.5-preview-free`, `jev-1.13-free`) if Space Bunny is
rate-limited.

- **Dispatch shape:** `agent: general`, `model: openrouter/stealth/space-bunny-alpha`.
- **Escalation:** `deepseek-flash` (`opencode-go/deepseek-v4-flash`) only when a mandatory trigger
  holds or Space Bunny fails the same bounded task twice. Anything beyond escalates **manually to the
  owner**.
- **Non-default lanes:** `gemini-flash` (Vertex prepaid) and `xiaomi-pro`; their balances are not
  spent on ordinary work.
- **Do not dispatch without the owner's explicit go.** The last wave was dispatched while
  `~/Workspace/HOLD` was still on disk; that is the failure not to repeat.
- Credentials live in the OpenCode **credential store**, not `auth.json` (which holds only the legacy
  set): `GET /api/credential` (Basic `opencode:<service.json password>`) shows the active `openrouter`
  key. `auth.json` providers: `opencode-go`, `zai-coding-plan`, `deepseek`, `google`, `openai`.

## 2 · State at compaction

- **No agents running.** All prior subagent sessions were cancelled; the cancelled agents wrote
  nothing (no repo file modified in their window).
- **Worktrees cleaned.** 20 stale agent worktrees removed — penglai `worktree-agent-*` (16),
  `penglai-water-control`, `intent-sb-oracle`, `webfold-bench-parity-docs`,
  `sanguine/.worktrees/review-novelty`, `oc-tmp/torchlean-de3192df`. Every diff captured first to
  `/home/socol/oc-tmp/agent-trees-cleanup-1790860389/` (`<name>.receipt.txt`, `<name>.diff`). Left
  alone (not agent trees): `opencode-v2`, `song-ref`, `sanguine-apeiron`.
- **Hold:** `~/Workspace/HOLD` removed; `gate hold` → `no hold in force`, exit 0.
- **RowDotAdmission:** gate of record exit 0 (`48 headlines`, mutant red on `widened_product_fits` /
  `srcOps_of_cont`, restored green). Its three files are in `proof/resource/` **uncommitted**:
  `RowDotAdmission.lean`, `RowDotProposedSource.lean`, `RowDotAdmission_gate.py`. `src/engine/matvec.rs`
  unchanged. Commit it when the next slice opens.
- **Lane free:** no `Matvec_gate`/`RowDotAdmission_gate`/`LmHead_gate` process holds
  `/tmp/opencode/apeiron-lane.lock`.
- **Resume queue** (`docs/plan.md` §Phasing E3): LmHead stopped on the cycle cap (`LmHeadResource.lean:557`,
  bind zeta-reduction); `regen_relocate_entry`/`regen_relocate_all` returned without handbacks;
  `exact_dot` stopped (bind over a pattern binder); `matvec` outstanding.
- **Extraction root:** `proof/extract/lib.rs` one error from compiling; fix is a three-line `src/`
  move (`dequant_f32_row` → `src/engine/f32_convert.rs`). `regen.sh --check` also needs the E0514
  sysroot cleared and `ApeironWord.lean` regenerated.

## 3 · First wave (ready; dispatch only on the owner's go)

Every row is ≤3 files, one gate, one control. `F`=files, `G`=gate, `C`=control, budget in seconds.

| id | task | F | G | C | s |
|---|---|---|---|---|---|
| E4-W1-T1 | move `dequant_f32_row` into `src/engine/f32_convert.rs` | `src/engine/{mod,f32_convert,loader}.rs` | `cargo build --manifest-path proof/extract/Cargo.toml` exit 0 | current red error captured, then green; `cargo test --locked --lib` | 420 |
| E1-W3-T1 | manifest schema + adapter | `docs/artifacts.toml`, `tools/gate/adapter.rs`, `tools/gate/main.rs` | `cargo test --manifest-path tools/gate/Cargo.toml` | 7 core controls green + duplicate-id/digest-mismatch rejects | 420 |
| E1-W1-T1 | conformance record for the 36 `proof/resource/*_gate.py` | `tools/gate/conformance.json`, `tools/gate/conformance_check.py` | `python3 tools/gate/conformance_check.py --check` | `--self-test` seeds a bad record, rejects | 420 |
| E5-W1-T1 | Intent D7 rebase-epoch correction | `intent/docs/EVOLVER.md` | the named control (old red, new green) | the control itself | 300 |
| E5-W1-T2 | Intent D2 cost-bound correction | `intent/crates/intent-core/src/selection.rs` | `cargo test -p intent-core` | new test red on old bound | 300 |
| E5-W1-T3 | loop-brief target correction | `docs/epics/apeiron-intent-loop.md` | cites the code lines | — | 300 |
| S-A1 | Song inventory: memory authority | read-only | receipt | — | 540 |
| S-A2/A3 | Song inventory: executing boundary + translation trust | read-only | receipt | — | 540 |
| S-K2.1 | Song inventory: Slang gate driver K2.1.1–K2.1.4 | read-only | receipt | — | 540 |
| S-A4..B6 | Song inventory: A4/A5/A6/A7/B6 | read-only | receipt | — | 540 |
| P-SPIKE-7 | Penglai milestone-7 bounded spike | read-only | contract / counterexample / smallest missing derivation | — | 540 |
| F-INV | Webfold: dedupe ledger by mechanism; first repair group | read-only | receipt | — | 540 |

Inventories write receipts under `/home/socol/oc-tmp/` (persistent), not into the repos.

## 4 · Programme frame

Fleet brief §9 (Song 10 waves, Penglai 6, Webfold 5, Apeiron E1–E5, Intent) with §9.1–9.3 decisions
and corrections; apeiron lane detail in `docs/plan.md` §Phasing. Sizing: ~50 agents, 10 lanes, ≤3
simultaneous heavy Lean checks. Task totals are estimates — forecast from a drained wave's observed
turnaround, and "one full pass" means every obligation gets a disposition.

## 5 · Rules

Routine proof and Rust tasks continue without re-escalating to the owner. Every task ≤3 files, one
gate, one control, one Arc block; the lead re-runs each gate before commit; no `sorry`/`axiom`/`admit`
outside the stated cone; a numeric wall-clock budget per gate (≤540 s). One owner per file; one writer
per shared build product. A worker stops on 3 materially different approaches, 5 patch-and-rebuild
cycles, budget expiry, or a contract/premise/grade/trust-boundary blocker, and returns the failing goal
verbatim, the approaches tried, what is on disk, and the one decision requested. Do not start a
document that does not replace one. Report results and evidence, not narration.
