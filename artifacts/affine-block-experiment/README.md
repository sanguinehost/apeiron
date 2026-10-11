# Blocked affine head experiment

Decision: retain the isolated numerical experiment; do not integrate or claim a speed win. Real-input capture, algebra and source mutation controls pass. Five-round timing and numerical quality remain open.

| evidence | result | grade |
|---|---|---|
| Whole executing gate | `timeout --kill-after=5 540 python3 /tmp/opencode/affine-block/gate.py`, exit0,48.063827s; scoped correctness passes, timing incomplete | tested; no performance acceptance |
| Capture | Qwen3.5-0.8B Q8_0, first affine layer, heads0/7/15, docs/code fixtures; zero constructor128tokens and reachable prefix64/suffix64; value widths128/32; all selected gated readouts and exit-state bits reproduce executing recurrence | tested, raw scratch comparison indirect |
| Block coverage | chunks8/16/32/64, full128/64 and partial127/63,480output/state cases plus240one-token continuation cases | tested arithmetic diagnostics |
| Maximum raw-output difference | 4.842877388e-8 over2,347,008coordinates;2,041,194bit mismatches | tested sampled discrepancy; no quality allowance |
| Maximum exit-state difference | 8.344650269e-7 over3,145,728coordinates;2,348,866bit mismatches | tested sampled discrepancy; no uniform bound |
| Maximum resumed readout/state differences | 1.490116119e-8 /8.344650269e-7 | tested one-token continuation only |
| Independent algebra | separate f64 chronological recurrence; exact dyadic/basis fixture and real-input arithmetic smoke checks | tested; smoke tolerance1e-3*max(1,maxabs oracle) is not production quality |
| Real-source mutation | zero scalar and AVX final-state outer-product coefficients in disposable copy; independent state assertion rejects exit101/5.468343s; exact source restoration exits0/5.468096s | tested attributable red/restoration |
| Timing |0/96cases complete five quiet rounds; all eventually exceed5% host-busy threshold | incomplete, no accepted ratio |
| Partial timing | one32-value,64-token head slice: recurrent/blocked0.6071,0.5418,0.4838 for chunks8/16/64; one pair each | exploratory only, excludes loading/projection/capture/reset/allocation; no generalization |

Source base3f29041c23697ab7aebab23da95fa900d3ff27b4, Intel Core i9-14900KF. Model SHA25637ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f. Full source/tool/environment/log digests are in the archive receipt and manifest. `summary.json` is derived from retained raw logs; the original runner recognizes AFFINE_F64 but does not collect the probe's ERROR-prefixed records, so use the summary and raw capture log for numerical statistics.

The initial root bridge fails compilation because a vector of layer indices is passed instead of one layer index (exit1/7.784542s). The corrected executing gate binds the one-index source; both receipts remain. Candidate preparation publication exceeds565/540seconds and remains unaccepted; passing the independently planned executing gate does not close that preparation task. The narrow read-only numeric review completes150.061/180seconds.

Source inspection: the implementation computes key Gram triangles and solves causal deltas with repeated small vector updates. Blocking adds O(C²*(dk+w)) operations per chunk; removing intermediate state writes does not remove that work. The existing recurrence already retains head state in cache. ExLlama/FLA's high-throughput matrix execution is a separate mechanism; exact-ring chunk/WY identities do not provide it or binary32 equivalence. Four32-value slices independently recompute the same key/query Gram data; preparing that data once per complete head is a possible next structural task. A second independent avenue is checkpoint-prepared projection tiles with immutable scales compiled once. Neither has a fresh measurement here.

Changed experimental paths: mixer/tests.rs; mixer/affine_block_probe.rs; mixer/affine_block_candidate.rs; step/tests.rs; step/affine_block_inputs.rs, each with declared separate ownership. Main runtime and foreign workstreams remain untouched. Private worktrees /tmp/apeiron-affine-block/worktree-agent-APEIRON-STRUCTURAL-SPIKE-affine-{capture,block} are retained dirty with complete recovery in the archive; no forced cleanup, commit, merge or push.

Trust/acceptance exclusions: Aeneas extraction, Lean source link and contraction/local-error premises; model-wide quality/logits/greedy choices; longer continuation; arbitrary geometry/nonfinite/overflow refusal parity; compiler/ABI/hardware FMA and SIMD; GPU; full Rust test/clippy/fmt and proof integration barrier. No epic closes. Next bounded task: prepare one shared Gram/causal operator per real head, apply it across value slices, and compare all readouts/state plus five quiet paired kernel rounds without changing numerical quality claims.
