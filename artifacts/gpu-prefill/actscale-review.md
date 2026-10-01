# Actual activation-scale refinement review

SCOPE: review `gate.py`, `ActScaleSource.lean`, `ActScaleResource.lean`, production `pow2.rs`, and the `chart/mod.rs` caller; finish with a scoped verdict and concrete gate fixes.
REGISTER: scalar source refinement and resource bound; excludes GPU latency, full caller refinement, and canonical `RingCell` correspondence.
CORPUS: reviewed the actual extracted `pow2.act_scale` and `pow2.act_scale_loop.body`; no canonical theorem is claimed by this artifact.
REFERENCE: `/tmp/worktree-agent-APEIRON-RESOURCE-W1-T2/src/engine/kern/pow2.rs`, production `act_scale` body.
GATE: latest main-checkout receipt `/tmp/opencode/actscale-resource/source-n6xbipa5-receipt.json`; no build run during the follow-up review.
GRADE: scalar refinement is `proved` for this receipt’s source/tool identity; mutation and Rust source tests are `tested`; resource conclusion is source-level, not a hardware measurement.

## Decision

Accept the theorem content and hardened gate at the stated scalar scope. The refreshed receipt records the enforced lane, memory/swap bound, thread count, outer timeout budget, top-level argv, and Lean/Rust compiler versions. No defect was found in the scalar theorem statement or its source binding.

## Source binding and evidence

`gate.py` writes a temporary crate whose only source entry is `#[path = <production pow2.rs>] pub mod pow2`; its `scale(a)` wrapper calls `pow2::act_scale(a)`. Charon starts extraction from `crate::scale`, so the generated call closure contains that forwarding wrapper and the real production body. The committed generated file is compared byte-for-byte with fresh extraction after normalizing only its source-location comment. The proof imports that generated module and names `act_scale_source.pow2.act_scale_loop.body` and `act_scale_source.pow2.act_scale`; it does not reproduce the Rust formula or narrow the input type/domain.

The real caller is `src/engine/chart/mod.rs:264`, where `quant_act` passes its `absmax: u64` to `super::kern::pow2::act_scale`. This proves the called scalar body’s behavior for its full `u64` domain. It does not prove the caller’s complete quantization behavior or its preceding `absmax` loop.

The latest main-checkout receipt records 12.68 seconds total and these outcomes: fresh Rust tests, Charon, Aeneas, and `lake build ActScaleResource` exit 0; the temporary `s *= 3` mutation makes Rust tests fail and the Lean proof exit 1 at `body_step` with a specific mismatch; restored Rust tests, extraction, and Lean proof all exit 0. The Lean output reports each registered theorem’s axiom set as exactly `{propext, Classical.choice, Quot.sound}` and prints `ACT_SCALE_KERNEL_OK`. Thus the negative control is attributable to the mutated multiplication in the extracted body, and restoration returns to the positive result.

## Theorem scope and numeric premises

For positive `a`, `ceilNeed(a) = a / 127 + [a % 127 ≠ 0]`. Since `a ≤ 2^64−1`, this value is at most `2^58`; the proof derives that bound from `U64.le_max`. Starting from `s = 1 = 2^0`, every extracted continuation doubles `s` and increments `steps` while `s < need` and `steps < 62`. The invariant proves `s = 2^steps`, `steps ≤ 58`, and the predecessor/least-cover condition. The loop’s well-founded rank is `58 − steps`, which strictly decreases on each continuation. It therefore takes at most 58 source doublings and returns the least power of two covering `ceilNeed(a)`.

The zero branch has an explicit witness `act_scale(0) = Some(0)`. The positive branch proves a successful `Some(s)` result with the cover and leastness conditions. `scale_never_none` proves `None` is unreachable for every `u64`, so the source’s refusal branch exists in the body but is outside the actual input population. This is a Lean proof about source-level loop transitions and arithmetic. It is not a measurement of CPU instruction latency or GPU execution, and it does not establish the full `quant_act` refinement or any `R[x]/(x²−σ)` / canonical `RingCell` correspondence.

## Gate and receipt disposition

The earlier review held acceptance for gate-wrapper hardening. The refreshed entry now sets the deadline before acquiring `/tmp/opencode/apeiron-lane.lock`, then runs under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`, `LEAN_NUM_THREADS=1`, and nested whole-invocation timeouts. It records top-level argv, lane/cap settings, Rust and Lean versions, child argv/statuses, and source/tool hashes. The refreshed receipt shows the wrapper completing within budget. No gate-wrapper residual remains at this scalar scope.

The production Rust comment in `pow2.rs` says “at most 63 doublings”; the proved bound for actual `u64` inputs is 58. That wording should be aligned with the theorem when the owning implementation task permits it. It does not invalidate the theorem or mutation control.

## Receipt identities

Latest successful main-checkout receipt: `/tmp/opencode/actscale-resource/source-n6xbipa5-receipt.json` (exit 0, 12.682 s; command `/usr/bin/python3 /home/socol/Workspace/apeiron/proof/resource/gate.py --check --controls`; outer 540-second timeout; lane `/tmp/opencode/apeiron-lane.lock`; memory 12G, swap 0, one Lean thread). It records Lean 4.31.0-nightly-2026-04-13, rustc 1.97.0-nightly (17584a181, 2026-04-13), and Cargo 1.97.0-nightly (a12f2bc1b, 2026-04-08). Recorded SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `src/engine/kern/pow2.rs` | `1f005c6807c1c87513c8c16654da3664614d9ced783616bd040fbfec8548d3b8` |
| `proof/resource/gate.py` | `7aa523f89ff97f9a37a3d6581be6331e90f590801c17b05dddea5ef083b3597f` |
| `proof/resource/ActScaleSource.lean` | `65e36c2f0673aea5855b4f06c8ae5f57f0944cd1b93f2597aba02128aa52e8fd` |
| `proof/resource/ActScaleResource.lean` | `64e4391987209a095920321d0d12c3cfb3f91247bcf17de887a73577b2bbe809` |
| `target/aeneas/charon-driver` | `207b223af35cf9a05227b329819b73532e94f655ccdefd64e77b755a15fdf297` |
| `target/aeneas/charon` | `747e165bc62bb963f8d741e8c193c721a4b94555bcb1f9f86c79090fd7a59d0e` |
| `target/aeneas/aeneas` | `33d0ce1a1cc2f1d3935c0cd2ab34f283797da61c531ed6d4e09d5c97b0d0228a` |
| `target/aeneas-model/lean/lean-toolchain` and `proof/dispatch/lean-toolchain` | `efac0b94923b2d8b6840cd35be9177ad0fc5ab2332f4f4311c98712cee92fdee` |
| `proof/dispatch/lake-manifest.json` | `9ccd1a96f2830caad41af223939c42ed7fede9e8ae6e2ce0223e8b6e86e54314` |

The latest main-checkout receipt supersedes the preceding candidate-worktree receipt. The candidate and main proof files were byte-identical at review: gate SHA-256 `7aa523f89ff97f9a37a3d6581be6331e90f590801c17b05dddea5ef083b3597f`, extracted source `65e36c2f0673aea5855b4f06c8ae5f57f0944cd1b93f2597aba02128aa52e8fd`, theorem `64e4391987209a095920321d0d12c3cfb3f91247bcf17de887a73577b2bbe809`. The receipt records no GPU timing. The full engine proof barrier remains open independently of this scalar result; this acceptance covers only the actual scalar `act_scale` extraction/refinement and its source-level loop bound.
