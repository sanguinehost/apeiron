# Prefill integration and GPU reuse

Main merge: `01de89d020954897f48e9863f806ac9df0bbf5af`.
`main-merge.json` confirms all preexisting dirty file bytes preserved and all 22 tested runtime/fixture
hashes unchanged. The completed integration worktree and branch are removed; integration.bundle
retains the merged stack. integration-receipt.json records the preceding isolated snapshot.

Luna repairs four Q32-specific attention fixtures with five explicit constructors and preserves both
exact greedy-decode checksum goldens. Sol reviews the Claude-written runtime, genuine extraction
requirements and next epic drafts. The owner's native Luna/Sol selection governs this work.

`rust-gates.json`: both full workspace/all-feature carrier suites, Clippy and formatting exit 0 in
508.645624 s within 540 s; 606 tests pass per carrier. `fixture-positive/receipt.json`: ten focused
cases pass. `fixture-controls/receipt.json`: carrier and golden-bit mutations fail on exact
assertions; restoration and source byte identity pass in 8.171184 s within 150 s.
`integration-sources.json` binds all 21 runtime files unchanged from measured 04d2060 plus the
fixture SHA. Later RoPE/T03/T04/cols overlays remain separate in artifacts/track-consolidation.

Historical five-quiet-round prefill: hybrid Qwen3.5-0.8B Q8_0 pp200 2.17x, pp512 2.45x; dense Qwen2.5
0.5B 1.58x and 1.5B 1.77x pp200 against llama.cpp 0324696b8, Intel Core i9-14900KF. Hybrid exceeds
its 2x target; dense improvements transfer from that work without a separate dense 2x requirement.
No new hardware timing runs in this consolidation.

GPU reuse reports: fold-gpu-receipts.md and fold-gpu-reuse.md, source Fold
`d497161f32e0d366220f2065ec36e680bd9aed86`. Retained RTX 5080 matched f16 GEMV is within 1.2%/5.1%
of cuBLAS time at the two recorded shapes; Q4 DP4A is 2.53x/2.47x versus that f16 baseline with a
different carrier. Reuse existing typed kernel selection, register tiling, batch operators, resident
dense prefill stack and hybrid decode recurrence. The additional work is Apeiron packing/state
correspondence and hybrid batched recurrence, with actual source-cost proofs.

`gpu-capabilities.json`: RTX 5080, 16303 MiB, Vulkan 1.4.351, shaderIntegerDotProduct=true.
`gpu-fragment-validation.json`: captured 676-word module matches its source checksum and passes
spirv-val for Vulkan 1.1. Capability/syntax evidence is distinct from runtime/numeric refinement.

Remaining proof work: `6e-extract-compile-fresh.json` exits 101, missing child f32_core path and mixer
aliases. Namespace repair alone does not connect the changed carrier/KvStorage constructor and
representation to existing integer theorems. sol-merge-review.md names the genuine-source migration
obligation. Regeneration/freshness, both Lean builds, named controls and size/trust acceptance remain
open; main merge completion does not claim those proof barriers passed.

Next epics: docs/epics/gpu-prefill.md reuses Fold's existing GPU execution; resource-guided-evolution.md
explains CPU/GPU reuse and develops source-connected cost certificates for typed schedule selection.
fold-plan-review.md records independent Luna review and document controls. Inference optimization
implementation remains behind actual admission; read-only design/refinement/cost tasks can establish
its prerequisites. Intent and sibling sources remain unchanged. No push.

SCOPE: this README; reconcile current merge/reuse status with retained historical receipts.
REGISTER: planning; runtime, proof and benchmark acceptance excluded.
CORPUS: source/refinement residuals read in sol-merge-review.md; no new theorem acceptance.
REFERENCE: /home/socol/Workspace/apeiron/artifacts/gain-integration/main-merge.json.
GATE: timeout 10s receipt/status/path check; historical pending-merge text fails current status.
GRADE: planning; exact runtime and historical measurement receipts retain their scope.
