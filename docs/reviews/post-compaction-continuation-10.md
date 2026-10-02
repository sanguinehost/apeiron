# Post-compaction continuation — programme dispatch (4)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`,
`docs/direction.md`, `docs/plan.md`, `docs/reviews/multi-repo-fleet-brief.md`. Source and gates are
the record. Live mirror: `/home/socol/oc-tmp/fleet-state.md` (eighteen deltas). Escalation verdicts:
`/home/socol/oc-tmp/escalation/`. This file replaces `post-compaction-continuation-9.md`.

## 0 · Dispatch floor and the supervisor

Owner direction: **20-50 agents running at all times; six is the floor.** Refill on every return;
a turn that ends below the floor with dispatchable work queued is a failed turn. The durable layer
now partly exists: `intent fleet check|tick` landed (`2c43f0c`, `243355e`, `708e62d`) with the
state file, the closed refusal/blocker cause sets, reserve-before-launch under a flock, and the
`<6` urgent path. Still missing from the supervisor: a mutation control for the tick (largest
gap), the `FleetTask`/`FleetLease`/`FleetReceipt` EventLog rows, the systemd unit and the 150-task
acceptance experiment (lead slots, spike.md (d)). Until it runs, the orchestrator refills.

## 1 · Routes, traps, lanes

- Workers: `openrouter/stealth/space-bunny-alpha` (free until 2026-10-05). Sol:
  `/home/socol/oc-tmp/bin/sol.sh <prompt-file>`; confirm the file exists beforehand (two launches
  raced the write and logged "No such file" while still answering; re-run clean for load-bearing
  verdicts). Verdicts land under `escalation/`.
- Lean: `flock /tmp/opencode/apeiron-lane.lock` + `systemd-run` caps, 540-600 s budgets. Corpus
  builds run in warm copies: `/home/socol/oc-tmp/corpus-t1-f32` (carries the narrowed artifact),
  `corpus-fix-scan`, `corpus-fix-attn`, `corpus-fix-sampler` (warm `.lake`; sync live files in).
- **Same-crate serialization is real**: one worker editing `src/fetch.rs` compile-blocks the whole
  webfold crate; E4/D4/headless gates sat blocked behind C3b. Never dispatch two source tasks in
  one crate concurrently; queue them.
- Timed runs never share `CARGO_TARGET_DIR` (register row); force rebuilds. libtest filters on
  test *names*. Traps: `rg` only (`find`/`grep` are shims); `local path=x` clobbers zsh `$PATH`;
  `read IFS=$'\t'` collapses tabs; `echo ===` is zsh `=cmd` expansion.
- Song commit hook: subject ≤72 with `<area>: `, body **one paragraph**, no dates (a toolchain id
  like `nightly-2025-11-21` counts as a date), no "not yet", no ` -- `. Penglai's hook runs its own
  check suite on commit. Webfold/apeiron/intent take multi-paragraph SCOPE/REGISTER/GATE messages.
- `AENEAS_HOME` for model builds = `/home/socol/Workspace/penglai/target/aeneas` (charon
  `747e165b`, aeneas `33d0ce1a`); the patched model lives at `/home/socol/oc-tmp/f32model`; the
  environment variable points at a backend that cannot consume the model patch.

## 2 · Landed since continuation-9 (commits)

| repo | commits |
|---|---|
| apeiron | `5a8953c` E1.W1 conformance record; `cf2b0ad` evidence ledger fixes; `0a2b14d` REANCHOR W1-T3; `caacbf3` + `4a4400c` + `677e48c` gate-record/digest maintenance; `8378951` tests fmt; `0069dcc` + `7b6c7b7` index_check enforcer + map; `5265e47` RowDot row; `313963a` `gate run` verb; `677e48c` run-verb cwd fix + E1.W5 pilot evidence |
| webfold (origin tip `1f59e9e9`) | `38425b9a` C1 transport; `eab111d1` B3 MCP; `8da2af32` CI round 2; `86b03f10` proxied status-line/meta test; `400aa82a` CI round 3 (gitlab mirror, deny current); `cf46cf9b` D1 runtime routing; `542ccfdc` C3 adapter; `d99b5fdd` deny triage; `1f59e9e9` D2 readiness |
| song | `9bd32b70` A0.10 census; `a2869f13` B6.1 profile; `e971fbc8` A5.10 platform validator; `aa20e7b8` A0.8 `song format --check` |
| intent | `a9c61a4` W06R w10 group + SWARM §8; `2c43f0c` fleet check; `243355e` fleet controls; `708e62d` fleet refill tick; `e07bc6d` W25 docs |
| penglai | `6edac67` surface metadata; `30cd29a` reference kernels; `34870ab` geometry fixture tool; `f6161a6` capture verdict; `18d994c` README bench docs |

## 3 · Formal chain (the point)

- **Artifacts**: base `bd7c612a` (reconciliation target); **narrowed `c16a68f0`** (with_carrier
  transparent after the `src/engine/attn.rs` `alloc_f32_storage` refactor; single new opaque item;
  4 axioms total; byte-identical on two regens). Live `proof/ApeironWord.lean` is stale
  (`800c0f9e`) — never build the live corpus; build in copies carrying `bd7c612a`/`c16a68f0`.
- **Green against `bd7c612a`**: `Growable/Scan` (`fcf22d4b`), `Growable/Representation` (reuse of
  `GrowableScan.tag_ne_zero`), `Numeric/EngineAttnRefinement` (`a46be5e0`; reshapes: `carrier` +
  `storage`, loop0/2/3 arities, `score_at`/`plane_at`/`value_component*` deleted,
  `attn_head_loop3_body_epilogue` renamed).
- **Sampler**: `sampler2` is re-deriving the seven `word_kv_state_new_*` against `c16a68f0`
  (probe: 6/6 refusals close; geometry is a do-block peel through `WP.spec_imp_exists` per bind
  then `injection`, the `word_kv_append_ok` shape).
- **Attn family**: B (`AttnHeadWindowsValue`/`LoopOutput`/`SoftmaxPerturb`) went quiet ~10:51 with
  no receipt; D (`AttnLoopScore`/`ScoreStage`) has an empty scratch — both need re-dispatch. A
  (`AttnHeadRefines`/`Score`) is blocked on B/D plus one content blocker. C adapted
  `AxiomAudit/Remaining.lean` (4 lines) and left `KvReachable` blocked on decisions that dissolve
  under `c16a68f0` — re-dispatch after sampler2.
- **Sol verdict on the attention weight path (route 1 accepted)**: the executed `recip = some r`
  arm computes a u128 multiply-shift estimate, so the old per-weight ideal (`divRoundNearest`) is
  withdrawn as an executable-refinement claim; the revised statement is an unproved specification
  until the reciprocal-error bound is established; `attn_head_refines_v2_proved` is blocked. Task
  order: specify `recip_u64`/`weight_at_recip` integer semantics, prove the reciprocal error lemma,
  widen `attnErrBoundV2`, re-prove end-to-end and update `AttnHeadRefines`. Independent of the
  `alloc_f32_storage` boundary. Verdict: `escalation/attn-v2-weight-path-answer.md`.
- **Landing unit** (`/home/socol/oc-tmp/f32bound/landing`, 8 files + artifact) needs a refresh
  before it lands: artifact `c16a68f0`, the `regen.sh` opaque list (`crate::attn::alloc_f32_storage`),
  `boundary-axioms.txt` rename, `model-patches.sh` header nine files, and the F-1
  `chmod 755 proof/audit-extraction.py`. Land only after the port and a green corpus.

## 4 · E6 port landing

- Live tree carries `M src/table.rs` + `M src/table/probe.rs`; `e6-land/port.patch` (100 KB) also
  covers locator, parallel, prepared, probe, resources, state, tests. The worker was last active
  11:03, still iterating.
- Barrier markers: `lib_test=0` (43 passed, 251 s), `fmt=0`, `probe_ratio=0` (ratio 1.6629 ≤ 2.2,
  parity pass); **`clippy=101`**: eleven `src/table/scan.rs` dead-code items after the summary
  retirement, plus a duplicate-key error in the `fold` checkout manifest. Control logs
  (baseline/slow_read/restored) exist.
- The scan.rs dead-code condition is shared by the live tree (any `cargo clippy -D warnings`
  fails). Fix: a documented `#[allow(dead_code)]` tied to the subtraction S1-S6, or land the
  subtraction first; the register says scan.rs stays until subtraction.
- After green: apply the patch to the live tree, commit as ≤3-file tasks, regenerate with the
  narrowed opaque list, re-reconcile the corpus against the new artifact, refresh and land the
  extraction unit. Engine T2/T3 and E3.W2/W5 unblock when `src/table*` settles.

## 5 · CI (the owner's complaint)

- **Five-Pillar**: chrome protected (`large-packages: false`), private deps authenticated
  (`CARGO_SUBSTRATE_TOKEN` repo secret, set from the owner's `gh` token — recommend rotating to a
  fine-grained read-only PAT), audit config schema fixed, eight committed fmt violations fixed,
  the gitlab `ptyktos` mirror rewritten to github. Runs reached the long Nix Charon+Aeneas build
  (the former disk wall) without the earlier failures; newest runs are on `1f59e9e9`.
- **Supply-chain**: SBOM green; advisories/bans/sources ok. **One red left**: the pinned apeiron
  rev carries GPL-3.0-or-later while `deny.toml` excludes strong copyleft — owner decision:
  re-pin to an MPL-2.0 rev (recommended) or amend the policy. lopdf RUSTSEC-2026-0187 is ignored
  with usage evidence and an expiry tied to the `superdocuments` pin; six wildcards pinned in
  place; transcode clarified MIT.
- Tip-known red test: `headless_response_review` (loopback fixture vs the destination guard; fixer
  in flight). After it lands, run a local workspace-test sweep (`cargo test --workspace --locked
  --offline`, warm target) to enumerate every remaining red before calling CI green. `cargo check`
  also fails at `src/sandbox/mod.rs:120` from another stream's WIP.

## 6 · Fleet queue at writing (~11:06)

Live: C3b (`src/fetch.rs` per-hop credential scoping), the headless fixer, E4 (implemented; re-run
its gate + controls once C3b unblocks the crate), D4 (same), penglai T4b (fixture loader; its
mutant copy is rebuilding Bevy cold), sampler2, and the E6 port worker. Quiet and to be
re-checked: attn-B, attn-D.

Queued dispatches: attn-B/D re-dispatch; attn-A re-run; KvReachable after sampler2; the
`AttnHeadRefines` reciprocal work (heavy — fresh session on a different worker model per the
ceiling rule); E1.W6 batch 1 (the `run` verb now exists); a scan.rs clippy task; webfold E1 after
D4; song B6.2/B6.3; penglai T5/W2; engine T2/T3.

Integrations owed: C3b commit+push; E4/D4/headless commits; the port commit; corpus files
accumulate uncommitted for the landing unit.

## 7 · Open decisions

- **CI licence**: re-pin apeiron to an MPL-2.0 rev vs accept GPL in `deny.toml`.
- Penglai headers: owner-committable patch staged (`/home/socol/oc-tmp/penglai-headers/`); drop
  `src/view/field.rs`; route B for the dirty tree.
- Webfold ledger inventory: 31 stale `source_files` rows (owner decision).
- scan.rs: allow-with-comment vs subtraction timing (port gate).
- Song: `tools/song/controls.txt` is still owned by the A0.10 stream; A0.8's `a232` control entry
  and A5.10's fallback entry are written out in their receipts for whoever owns it.

## 8 · Next actions, ordered

1. Owner: the GPL call. Then push the final CI commit and watch the newest runs to completion.
2. C3b returns → commit+push; re-run and commit E4, D4, headless; run the workspace-test sweep;
   fix every red before reporting CI status.
3. Wake attn-B/D (re-dispatch if silent); then attn-A; sampler2 → KvReachable; then the reciprocal
   specification work.
4. Port: resolve the scan.rs clippy condition; barrier green; apply the patch; commit; regenerate
   with the narrowed boundary; re-reconcile; refresh the landing unit; land.
5. Refill on every return toward the floor; mirror updates; never end a turn below the floor while
   dispatchable work exists.
