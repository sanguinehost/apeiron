# Structural prefill dispatch

Stopped by user after all three agent handbacks. No next wave starts. No general2x prefill result is established.

| avenue | evidence | decision |
|---|---|---|
| compiled schedule | Q32/f32 hidden parity/refusals pass; swapped FFN bindings fail and restored source passes | retain isolated candidate; timing blocked by other active CPU tasks before the first sample |
| live-state diagnostics | gate exit0,318.379s; omitted replay fails at live KV occupancy201 versus202, restoration passes | retain diagnostics for prefill chunk equivalence; current population is decode after completed prefill |
| batched FFN certificate | gate exit0,47.391s; actual README200 prefill samples all24 layers;0/2688 blocks certify for each1–4 omitted low bits | reject this conservative certificate at this sample scope; tighter representations remain open |
| benchmark suite | full-prompt Rust/C++ timing probes and five-round runner drafted; no build/protocol/fixtures or timing run | untested recovery material only |

A source/proof representation gap needs its own refinement: [ChartDot.scale](../../proof/corpus/Numeric/ChartDot.lean) uses ceil(absmax/127); the captured executing activation chart rounds up to a power of two. At absmax300 the scales are3 and4. The generic partition/one-rounding laws remain reusable, but the concrete quantiser instance does not represent that executing scale. No new Lean proof is claimed. Smallest next target: extracted word_pow2_finish rounding/refusal, then word_act_scale.

The next structural prefill seam is staged-convolution borrowing, followed by activation buffers reused across projection lifetimes. Both need constructor-to-chunk live-state controls and matched prefill measurement. No arithmetic reassociation is licensed by this review.

[evidence.tar.gz](evidence.tar.gz) preserves exact patches,13 recovery source files, source digests, commands, gate runners, full logs and handbacks. SHA256: df72aeb598abe5f38916d127c9b92a9419a7d0a449f1c16bebf5e5a59c5d35fb. The archive root has compile-graph/, selective-state/, quant-certificate/ and benchmarks/; each recovery/ directory contains complete.patch and source files. Base:3f29041c23697ab7aebab23da95fa900d3ff27b4. dispatch-manifest.json records dirty overlays, ownership and rustc identity. restore by applying the task patch in a new isolated checkout at that base; do not combine candidate patches without integration review.

Archive verification exited0 in0.717s under60s. [cleanup.json](cleanup.json) records removal of this dispatch's four worktrees/branches after source hash verification. Other workstreams remain untouched. No candidate was merged, committed or pushed.

Changed supplied-checkout prose: docs/epics/structural-decode-search.md and one additive docs/README.md row. Source candidate paths remain only in recovery material. Full Rust tests/clippy/fmt, Aeneas regeneration freshness, bounded Lean builds and integration are unrun. Trust excludes compiler/ABI/GPU/unsafe realisation; source refinement is open. The certificate's universal interval soundness and exact fused/scalar internal equality were not established.
