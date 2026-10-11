# Consolidated stopped tracks

Decision: preserve the complete Claude source stack as one checked, recoverable candidate.
At the preservation checkpoint main runtime is unchanged. The owner subsequently authorises
merging the gains and planning GPU/SPIR-V execution and proof-guided evolution. The auxiliary Rust admission gate has separate scoped acceptance.

| track | retained result | remaining boundary |
|---|---|---|
| Rust admission | `../rust-admission/README.md`: actual Rust extraction, nine audited headlines, thirteen unit tests, one compile-fail test, fifteen synthetic controls, actual unregistered worker refusal | Conditional observations; IO trusted/tested; registry empty; no inference speed claim |
| Claude committed prefill | `task/w6-integ` at `04d206013f70f245dfdaf2619c0b1846d10b9922`, retained in `claude-history.bundle` | Main is `e33977a`; five recorded binary32 baseline suite failures and whole proof/bridge barriers remain open |
| Claude RoPE fix | Dirty loader permutation and tests preserved and included in the candidate; four focused tests pass | Sequential oracle reads the original file, bypassing the permuted resident buffer; full source-linked claim open |
| Claude T03 | Aligned activation pack and merged-multiplier kernel are wired; focused kernel tests pass | Measurement instrument defects; full source/cost refinement and matched quiet acceptance open |
| Claude T04 | Batched attention, shared tile helpers and pending cols wiring assembled together; focused attention and prefill checks pass | Independent engine parity/refusal controls and end-to-end timing including scatter/allocation open |
| Intent evolver | Separate thread `01a0f458-538e-7470-ae35-6c7779b474cc`, title `Review Evolver evaluation changes`, proceeds with repairs | Snapshot reports a hanging invalid-input case receiving full reward; no Intent source edited here or acceptance verdict claimed |

## Source and checks

`source-manifest.json` binds26 changed/new files to their original worktrees and combined hashes.
`combined-candidate.tar.gz` contains those files over committed6e; `rope.patch`, `t03.patch`,
`t04.patch` and `t04-cols.patch` preserve the separate changes. The original worktrees are unchanged.
The assembled checkout is `/home/socol/oc-tmp/track-consolidation-candidate`.

Recover by restoring committed6e from the bundle with its existing `e33977a` prerequisite, then
overlaying `combined-candidate.tar.gz`. `history-verify.log` records successful Git bundle verification.
Archive restoration matches every source hash; changing one recovered byte is rejected, and
restoration passes. This is a preservation control, not a semantic mutation control.

`combined-checks.json` records one180s invocation with `APEIRON_F32_CARRIER=1`:
attention17, kernel43, RoPE4 and prefill1 tests pass,65 total; Clippy and formatting exit0.
These focused tests do not constitute the whole workspace barrier. The prefill fixture compares
two paths sharing the current implementation; it is not an independent oracle for the new wiring.
`candidate-inventory.json` retains source sizes and files exceeding the1500-line bound.

## Review findings

- T03 is not still mid-edit: `kern/gemm.rs` calls `merged_operands` and `x2_shifted` in both
  projection paths, and `merged_kernel_equals_the_shifted_kernel` tests eligible/fallback cases.
  The live dense logs contain measurements, contrary to the earlier stop report. Their interpretation
  remains limited by the following instrument defects.
- `kern/gemm_attrib.rs:1168` chooses a best attempt and retains one even if every foreign-load value
  exceeds0.05; the quiet observation is discarded. This cannot establish the required five paired
  quiet-round result. `:1145` treats `Some(Err(_))` from Swiglu as a successful call through
  `.is_some()`. Its FNV equality is a digest comparison, not proof of identical output words.
- `attn/f32_block.rs:818` times the batched path with `scatter=false`, although the proposed
  `step/cols.rs` path gathers the outputs and allocates intermediate/result buffers. The layer
  timing cannot establish its end-to-end effect. The synthetic harness now lists Qwen2.5-0.5B,
  Qwen2.5-1.5B and Qwen3.5 at200/512; an earlier single-shape report is stale. Listing shapes does
  not establish a completed measurement population.
- T04 comparisons share refactored tile helpers with the reference. Require an independent
  baseline, output/live-state/refusal equality, constructor-reachable fixtures, a wiring mutation
  that is actually detected, and restored source before claiming implementation equivalence.
- RoPE's report acknowledges that `Sched::Sequential` sources tensor bytes from the file while
  the permutation updates the resident buffer. A Llama comparison against that oracle needs an
  explicit representation/source repair. Unchanged `rope.rs` extraction does not prove the new loader.

The source locations above refer to the preserved combined candidate, whose digests are recorded.
No new universal correctness theorem or hardware bound is claimed for these candidates.

## Archived prefill evidence

`archived-ratios.json` recomputes ratios from the retained raw JSON; no fresh timing ran.
Host: Intel Core i9-14900KF. Comparator: llama.cpp `0324696b8`, best measured thread count8/16/24/32.
Source: clean committed6e; GGUF and prompt hashes and exact protocols remain in each receipt.

| model | prompt tokens | quiet rounds | median prefill ratio |
|---|---:|---:|---:|
| Qwen3.5-0.8B Q8_0 | 200 | 5 | 2.17x |
| Qwen3.5-0.8B Q8_0, later receipt | 512 | 5 | 2.45x |
| Qwen2.5-0.5B Q8_0, later receipt | 200 | 5 | 1.58x |
| Qwen2.5-1.5B Q8_0, later receipt | 200 | 5 | 1.77x |

The hybrid-path2x target is met on the two listed hybrid workloads. Dense gains of1.58x and1.77x
transfer from hybrid-path optimisation without a separate dense optimisation campaign; they have no
2x acceptance requirement. Explaining that transfer through shared primitive costs is a research
target. Three-round or nonquiet receipts are retained and labelled,
not substituted for the five-round quiet population. Generated-token comparisons remain separate.
`evidence/` preserves67 reports/logs/receipts with hashes in `evidence-digests.json`.

## Stop boundary

`stopped-processes.json` records five confirmed surviving Claude gate descendants terminated and
an empty rescan. Two earlier orphaned dense scripts were also observed waiting outside the540s
invocation ceiling; their wrapper uses `flock -w900` before `timeout540`. The stop report's
claim that no descendants survived was false at inspection. Other projects and the Intent repair
thread are preserved. Historical subagents of this thread are completed.

Runtime integration remains blocked by the open shared gates, independent engine controls,
source/refinement/cost correspondence, source-size acceptance and reviewed architecture boundary.
Independent consolidation review is inconclusive: Gemini exits124 at150s with a provider transport
error and no verdict; the MiMo packet review exits124 at90s without a verdict. Their raw logs are
retained. Neither timeout is an architecture approval or a semantic rejection. These are historical stop receipts;
the owner now authorises a bounded integration review and gain finalisation.
The preservation slice changes no source contract/premise, canonical definition, extraction pin,
registry entry, commit or push.
Next work is the named admission/measurement/oracle repair, not another timing retry under load.
