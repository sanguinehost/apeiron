APEIRON-FINALISE-W1-T1

SCOPE: `src/engine/tests.rs` only. Pin the four fixed-point attention contracts to `KvState::with_carrier(..., super::mixer::Carrier::Q32)` and preserve `engine_step_greedy_decode_is_deterministic` under both environment-selected carriers with the carrier-specific first-checksum assertion. Complete both-carrier focused positives and fixture/golden mutation controls.
REGISTER: test obligation; excludes runtime implementation and all non-test source.
CORPUS: `src/engine/tests.rs`, `src/engine/attn.rs`, and `src/engine/mixer.rs` read; no applicable Lean theorem.
REFERENCE: NONE - test fixture carrier selection follows explicit source behavior; no theorem-derived claim.
GATE: `timeout -k3 150 python3 /home/socol/Workspace/apeiron/artifacts/gain-integration/fixture-gate.py /tmp/worktree-agent-APEIRON-FINALISE-W1-T1 /home/socol/Workspace/apeiron/artifacts/gain-integration/fixture-finalise`; cwd `/tmp/worktree-agent-APEIRON-FINALISE-W1-T1`; task cap 540 s. Baseline receipt `../fixture-baseline/receipt.json`: Q32 all five tests exit 0; F32 four fixed fixtures and decode test exit 101. Shared build lane: `/home/socol/oc-tmp/wt/target-w6-integ`.
GRADE: tested, baseline before change has expected carrier mismatch; after change requires all ten carrier/test pairs green. Mutation controls: explicit fixture carrier flip and golden checksum bit flip must each fail at the intended assertion; restored source must pass.

Base: `04d206013f70f245dfdaf2619c0b1846d10b9922`. Owner: fixture_finalise.
