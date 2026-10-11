# Structural decode review

Decision: develop selective multi-token verification from a cheap draft source. Four-column Q32 lookup reaches a median paired 2.14408x over llama.cpp greedy on the code fixture, up from1.45568x for Q32 greedy in the same campaign. README reaches1.79580x. This is one contextual 2x result, not a general 2x lead. The code continuations differ between engines.

## Contextual measurement

Intel Core i9-14900KF, CPU only; Qwen3.5-0.8B Q8_0; identical200-token prompt IDs;64 emitted predictions;63 decode steps timed after prefill and first prediction. Each process warms one identical workload before a fresh-context measurement. Five rotated rounds, CPU busy fraction below0.05 before each sample. No EOS early stop. No sample is discarded.

llama.cpp headers and libraries bind to build10152, revision0324696b8; the source checkout is newer. The comparator uses lowest-index greedy argmax, f16 KV, flash attention auto, context268, batch/ubatch512, no GPU or operation offload. Threads8/16/24/32 are measured in every round. Eight threads wins every round on both cases. Apeiron uses its existing resident worker pool; this is tuned throughput, not an equal-worker-budget or energy comparison. Apeiron lookup speculation is compared with llama.cpp greedy; llama.cpp speculation is not measured.

| Context / engine | Median tok/s | Median paired ratio to fastest llama arm | Median paired ratio to own greedy |
|---|---:|---:|---:|
| README / llama greedy | 51.3103 | 1 | 1 |
| README / Q32 greedy | 75.5830 | 1.42308 | 1 |
| README / Q32 lookup4 | 92.1430 | 1.79580 | 1.26191 |
| README / f32 greedy | 90.9610 | 1.68849 | 1 |
| README / f32 lookup2 | 94.7677 | 1.90823 | 1.06794 |
| Code / llama greedy | 51.0165 | 1 | 1 |
| Code / Q32 greedy | 74.2637 | 1.45568 | 1 |
| Code / Q32 lookup4 | 110.7579 | 2.14408 | 1.48519 |
| Code / f32 greedy | 79.8245 | 1.47952 | 1 |
| Code / f32 lookup2 | 98.9212 | 1.84617 | 1.13540 |

Ratios are medians of per-round ratios, not ratios of the displayed medians. Code Q32 lookup4 spans2.01112–2.22282x versus the fastest llama arm over the five rounds. README Q32 lookup4 spans1.60709–1.91751x. Code f32 lookup2 spans1.12149–2.00648x; that variability prevents a dependable2x claim.

Every lookup arm emits the same64 IDs as its own greedy arm in every measured round. README emits the same64 IDs across all three engines. Code Q32 first diverges from llama at zero-based token27;28/64 IDs match. Code f32 first diverges at token58;58/64 IDs match. Numeric-carrier quality and longer output equivalence need separate acceptance. All llama thread settings emit identical IDs within each round.

[Context receipt](/home/socol/Workspace/apeiron/artifacts/context-decode-compare/receipt.json), [source, headers, comparator, controls and logs](/home/socol/Workspace/apeiron/artifacts/context-decode-compare/evidence.tar.gz). Archive SHA256 b11521e67a55adbe8a94b66520dabc2ee71d89abdb1254993497e413333b7bfb. Acceptance: `timeout --kill-after=5 540 python3 /tmp/opencode/context-compare/gate.py`, exit0,437.669seconds. Invalid token input rejects. A source mutation emits63 IDs while reporting the64-ID workload; the parser rejects it with `token count,63`. Restored source emits64. Model, inputs, executable, linked-library and header digests are in the receipt.

## Executed seams

| Mechanism | Observation | Decision / evidence |
|---|---|---|
| Skip weight columns with exact zero activation codes | Only3.6–3.7% zeros; no all-zero32 blocks in three populations | Reject. [Support receipt](/home/socol/Workspace/apeiron/artifacts/decode-support-spike/review.md); gate exit0,55.595s; predicate mutation red and restoration green. |
| Demand the second speculative head only after acceptance | Q32 paired1.06364x, f32 paired1.02136x over full-head speculation | Retain as a later component; insufficient alone. [Head receipt](/home/socol/Workspace/apeiron/artifacts/demand-head-spike/review.md); gate exit0,211.591s; unconditional-acceptance mutation red and restoration green. |
| Execute small activation quantisations locally before fanout | Q32 paired0.98206x; f32 approximately1.00x or0.98838x | Reject. [Schedule receipt](/home/socol/Workspace/apeiron/artifacts/static-quant-spike/review.md); gate exit0,180.760s; truncated-input mutation red and restoration green. |
| Use suffix lookup only when a draft exists | Q32 README1.19766x and code1.21791x over greedy with suffix2 | Retain. [Lookup receipt](/home/socol/Workspace/apeiron/artifacts/selective-lookup-spike/review.md); gate exit0,458.310s; wrong acceptance red and restoration green. |
| Verify four columns; replay accepted prefixes after a later mismatch | Separate campaign: Q32 README1.28704x, code1.49355x over greedy; f32 width4 approximately0.90x | Retain carrier-aware width. [Window receipt](/home/socol/Workspace/apeiron/artifacts/lookup-window-spike/review.md); gate exit0,380.426s; omitted replay red and restoration green. |

These campaigns have different timing domains. Demand-head and quantisation measurements use fresh-state or one-seed decode; lookup measurements use their stated contexts. Their gains are not multiplied together. The repeated-prose lookup fixture is artificial and is not the headline population. All new grades are tested or measured; no new Lean proof is claimed.

## Where the algebra should change execution

The useful SKI direction is a compiled graph with shared pure nodes and an explicit state thread. [webfold substrate](/home/socol/Workspace/webfold/src/substrate.rs:274) implements S by cloning its argument twice into boxed application trees. Its bounded evaluator also clones the initial term. Copying that interpreter into every token step does not remove weight traffic. [webfold lifting](/home/socol/Workspace/webfold/src/lift.rs:1120) provides the compile-once route; shared graph storage is a proposed change, not an established implementation result.

[Apeiron.Combinator.denote_reduces](/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Combinator/Denotation.lean:344) gives reduction-to-route preservation. [kRoute_reads_nothing_of_its_argument](/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Combinator/Denotation.lean:144) supplies a weakening model. [denote_not_faithful](/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/Combinator/Denotation.lean:428) blocks the reverse inference from equal routes to convertible terms for its stated open-term environment. These declarations are read pointers, not fresh kernel receipts or cost theorems. The physical corpus and bridge manifest resolve880aa57e47e782ed7a9d9ee1974cfc87360b446b.

The graph should expose three interfaces:

| Interface | Structural rewrite | Required source relation |
|---|---|---|
| Immutable weights and pure numeric nodes | Share one quantisation with gate/up consumers; fuse admitted matrix columns; omit an unobserved head | Same exact code/scale blocks, rounding boundaries, tie rule and successful/refusing behaviour. Dropping a value cannot silently drop a public error. |
| Greedy state transition | Draft cheaply, verify several tokens, commit the accepted prefix | Each column equals the executing single step; KV and affine state reach the accepted boundary; later refusal and future outputs agree. |
| Numerical observation | Compute only enough intermediate information to certify the next quantised activation | Joint32-element scale certificate and every rounding bucket must be determined before low weight planes can be omitted. Uncertain blocks take the original full path. |

The implemented lookup/window probe instantiates the second row at tested-output grade. Prefix replay shares projection work without changing recurrence order. It needs no approximate inverse state update. A longer-lived drafter can index fixed-length suffix keys in a bounded dictionary, with full-key identity and checked resource outcomes; replacing the current linear reverse search is a separate representation task. Growable Table lookup is not automatically constant-time from fixed probe width.

## Proof review

| Read declaration / source | What it supports | Missing link or mismatch |
|---|---|---|
| [ColumnCover.column_cover_round](/home/socol/Workspace/apeiron/proof/corpus/Numeric/ColumnCover.lean:54), quant_locality:131, split_ffn_exact:176 | Exact integer partition sums with one final rounding; aligned block quantisation; FFN slicing with preserved quantiser boundaries | No licence to round each slice separately; no executor, allocation or speed proof. Earlier hidden-slice experiments are not assumed faster. |
| [BitplaneArgmax.bitplane_argmax_exact](/home/socol/Workspace/apeiron/proof/corpus/Numeric/BitplaneArgmax.lean:284) | Exact lowest-index argmax under sound bounds | SIMD bound construction, actual frontier and hot head source correspondence remain separate. Limite head spectra do not establish Qwen pruning. |
| [SpeculativeRollback.spec_round_state_eq_greedy](/home/socol/Workspace/apeiron/proof/corpus/Nous/Model/SpeculativeRollback.lean:192), [JacobiDecode.verify_accept_prefix](/home/socol/Workspace/apeiron/proof/corpus/Nous/Model/JacobiDecode.lean:267) | Arbitrary drafts are admissible under their abstract greedy verifier | The model is sequential. It does not certify the new parallel column executor, snapshot/truncate/replay or refusal branches.64-token parity is not full-state equality. |
| [GatedDelta.stepAffine](/home/socol/Workspace/apeiron/proof/corpus/Nous/Seq/GatedDelta.lean:127), affComp_assoc:76, recurrent_eq_chunked:210 | The actual gated-delta affine map includes the key-dependent rank-one factor; exact-ring composition is associative | Binary32 arithmetic is not a commutative ring. Exact-ring scan reassociation needs a numerical refinement before use on f32. |
| [EngineLaws.trans_assoc](/home/socol/Workspace/apeiron/proof/corpus/Nous/Model/EngineLaws.lean:58) | Composition of scalar decay plus a fixed additive matrix | Its element omits the state-dependent key correction. It is not the complete gated-delta transition used by the runtime. Use GatedDelta's matrix-valued map for that claim. |
| [EngineLaws.round_div_pow2_eq_shift](/home/socol/Workspace/apeiron/proof/corpus/Nous/Model/EngineLaws.lean:93) | Replace the half-power rounding offset | Both sides still divide by a power of two. The comment's hardware-divider elimination does not follow. An arithmetic-shift refinement, source link and emitted-code/measurement receipt are required. |
| [EngineLaws.head_col_split_k_one_identity](/home/socol/Workspace/apeiron/proof/corpus/Nous/Model/EngineLaws.lean:101) | Nat division/remainder at1 | The identity alone establishes no removal of worker branches, scheduling cost or decode latency. |
| [DeltaCarrier.delta_carrier_error_uniform](/home/socol/Workspace/apeiron/proof/corpus/Nous/Seq/DeltaCarrier.lean:119) | Uniform error from a supplied per-step error, unit keys, bounded beta, uniform decay below1 and equal initial states | Actual normalised float keys are not exactly unit real vectors; operational rounding and a reachable uniform gate bound must be connected. |
| [CarrierEpsilon.delta_relative_carrier_error_uniform](/home/socol/Workspace/apeiron/proof/corpus/Nous/Seq/CarrierEpsilon.lean:349) | Derives error for coordinatewise rounding of an exact recurrence under a relative-carrier hypothesis | This new read module narrows the epsilon gap. It still models one rounding after an exact update, not every intermediate runtime dot/update/FMA. Its normal-range carrier definition:108 is distinct from the headline's global relative-carrier hypothesis. Actual binary32, subnormals and key normalisation need an executing link. No fresh build ran here. |
| [BlockComposition trust boundary](/home/socol/Workspace/apeiron/proof/corpus/Nous/Model/BlockComposition.lean:1) | Conditional composition of dimension-changing operation contracts | Attention, multiplication and RMS-domain obligations remain named. The statement does not close whole-checkpoint equivalence. |

[Extraction selection](/home/socol/Workspace/apeiron/proof/extract/lib.rs:48) includes real scalar/numeric, attention and matrix bodies. It does not select step/cols.rs, step/spec.rs, mixer.rs or the hot multi-column SIMD/head path. The enlarged numerical proof surface is useful, but it is not an end-to-end refinement of this measured executor. Pin documentation reads Lean4.31.0, Aeneas505b6ca35217e7be5c96c3e2f8045edfbdf47291, Charonb104e24fea7d721b71e6c39fd70f26ff20bc0980, Mathlibfabf563a7c95a166b8d7b6efca11c8b4dc9d911f. These are read pins; no tool execution or fresh proof acceptance is implied.

[Read-source snapshot](/home/socol/Workspace/apeiron/artifacts/decode-structural-review/review-sources.tar.gz), SHA256 8e5d6b1087440771c99f1b4c40f68b5bf72e2e57fd60ee5043e1ffe3e7d23df7, and [digests](/home/socol/Workspace/apeiron/artifacts/decode-structural-review/source-digests.json) preserve the reviewed dirty proof files and sibling sources. webfold HEAD is23e727347950138aa9fefd36d6fee89689990465. No governing document or theorem is edited to resolve these findings.

## Further directions

| Priority | Direction | First falsifier |
|---|---|---|
| 1 | Carrier-aware selective verification, with draft width chosen from measured prefix acceptance and verifier/replay cost; combine with demanded heads only after separate parity/refusal gates | A nonrepetitive held-out workload regresses, or a damaged later draft still passes state/continuation checks. Compare against llama speculative modes as well as greedy before a broad lead claim. |
| 2 | Compile the fixed model graph once into shared pure nodes and an effect-linear state schedule. Treat SKI as the rewrite language; retain admitted numeric kernels as primitives | The generated schedule changes a refusal, quantised block, lowest-index tie or state boundary; lowering costs exceed removed dispatch/materialisation. |
| 3 | Observation-directed FFN projection: certify the32-element quantisation scale and codes from coarse weight-plane bounds, then avoid irrelevant fine planes | Too many blocks remain ambiguous after the coarse pass. Measure certificate coverage and bytes before implementing SIMD. SiLU and the shared block maximum can make bounds too loose. |
| 4 | Smaller integer or block-floating recurrent state with a sourced rounding/error contract | A real gate/key/rounding bound cannot be witnessed or the64-bit fallback dominates. Functional acceptance and a numerical quality budget precede speed claims. |

Priority3 differs from zero sparsity: dense values can still have a stable quantised observation. This could eliminate arithmetic that cannot affect the next admitted quantiser, while preserving Q8Act output. It is a design/anchor, not a theorem, implementation or performance result.

The fresh profile attributes approximately29.5% of traced token time to FFN,20.7% to affine input projection,17.6% to the head,10.9% to affine state heads and3.4% to the softmax core. Pool wait is an overlapping18.4% trace, not another additive phase. The head's printed byte count is a logical full-Q8 count, not measured DRAM traffic. These figures come from a separately traced run, not an Amdahl proof. The executable binds to the frozen W5 source archivec6f7052d8dd8786c6e23b3041942f35828591ecd64c2706d30e7b3edb2c731b9; the refresh harness printed current-main source identity because it ran from main, so that printed source identity must not be attributed to the binary. Binary SHA256 5fd885061bd7a6cb0faef2c132b7072778af8f9f5f8318b99046235433b41503 matches the archived campaign. Raw refresh receipts are retained in the read-source archive.

The measured Qwen3.5 checkpoint uses18 gated-delta layers and6 softmax layers. The separate experimental hyperbolic-attention branch is not this checkpoint's full architecture. Bounded memory remains useful, but the short-context trace makes dense weight streams the larger speed target. Native model/topology changes need their own training and quality acceptance; they cannot be counted as the same-checkpoint comparison.

## Handback

Changed production files: none. Saved artifacts contain five experiment sources/controls, the contextual comparator and this review. The previous two-column-head plan's timing description is corrected: its spec64 starts from one seed, not the separately timed200-token prefill. No timing number is changed. All five private experiment worktrees and their branches are removed after source-archive verification. Existing workstreams, dirty proof work, corpus pins and main HEAD e33977aa72534ad56775110be99cf002ca96fa37 are preserved. W5 integration advances independently; this review uses its frozen incoming source.

Next bounded task: establish full accepted-prefix state and refusal parity before adding the lookup API to the build path. Own step/spec.rs, step/tests.rs and one epic plan; use a private lane and source controls for omitted replay, wrong acceptance and miss-only fallback. Rebase against the current integration source rather than transplant the whole frozen overlay. Then separately link the executing verifier to the abstract rollback theorem. No new Lean/extraction or full Rust barrier runs in this slice; `proof/regen.sh --check`, bounded lake builds in both packages and the three Rust commands remain unrun integration gates. Compiler/codegen, ABI, unsafe-container, GPU and numerical-quality exclusions remain. No commit, merge, push or epic closure.

Artifact consistency gate: `timeout --kill-after=5 60 python3 /tmp/opencode/structural-review/verify.py`. It recomputes all contextual paired ratios from the five rounds, rejects a seeded5% ratio change, restores the baseline and checks archive source digests, receipts and local links. The first inventory attempt assumes the wrong receipt prefix for the earlier lookup archive; the accepted check uses its actual logs/receipt.json member. This is an artifact check, not another performance or proof run. [Result](/home/socol/Workspace/apeiron/artifacts/decode-structural-review/artifact-check.json).
