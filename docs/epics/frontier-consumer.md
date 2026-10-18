# Plan — frontier intelligence on consumer devices (cross-track roadmap)

- **Owner:** lead
- **Status:** draft; a roadmap across epics, not an epic. Each track opens its own intent/spec/plan
  before tasks are allocated (AGENTS §2).
- **Parents:**
  - the prior-art index in `docs/roadmap.md` (Paper I and II, spinor-ttl, the Neural corpus);
  - stage 6 `AI-ARCH` of `docs/roadmap.md`;
  - `docs/epics/ai-track-reanchor.md` (the trichotomy-first model surface);
  - `docs/epics/ternary-hadamard.md` (Track A, in progress).
- **Grade of every claim in this file:** `anchor` (a design reading), unless a row names a gate.

## Binding constraint

On a consumer device, autoregressive decode is bound by memory bandwidth, and model size is bound by
resident memory. Model this, then measure it:

- `tokens/s ≈ B / W_active`, where `B` is sustained memory bandwidth (bytes/s) and `W_active` is the
  bytes of weights read per token;
- `W_resident ≤ M_device − KV − activations`, where `M_device` is device memory (16–64 GB unified
  memory, or the GPU's VRAM).

Every track is judged by what it does to `W_active`, `W_resident`, the per-token KV growth, and
quality at fixed `W_resident`. The levers multiply, so the tracks are complementary.

A derived requirement follows from the bits-per-parameter bound in R2:
- knowledge capacity is measured at about 2 bits per parameter (Allen-Zhu and Li, arXiv 2404.05405);
- a ternary weight carries at most `log₂ 3 ≈ 1.58` bits;
- so a fully ternary model stores fewer knowledge bits per parameter than that measured capacity.

Knowledge must therefore move out of the weights (Track B) or into more parameters (Track A's MoE).
This is a mapping, not a measurement; the Track A falsification row measures it.

## Measurement host and protocol

- **Host:** the RTX 5080 box named in `../roadmap.md` stage 6, plus its CPU. A unified-memory
  consumer machine is added when available and named in the receipt.
- **Isolation:** exclusive quiet host, no Lean or agent lanes running (AGENTS §10; the G1 rule).
- **Report per run:** model file sha256, command, machine, metric (tokens/s, time to first token,
  resident bytes, top-1 agreement), interval, before/after.

## Product ladder (draft; the owner confirms rung order)

The aim is an entity, not an oracle: a persistent, self-consistent agent on owned hardware, many at
once. It learns by reading, looking things up, trying and remembering, so capability grows over its
life instead of being fixed at training time. Each rung is a shippable product if the next one is
not reached.

The keywords MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119/8174.

| rung | product | requirements |
|---|---|---|
| R0 | proved local engine | MUST generate coherently at ≥ 30 tokens/s (60 SHOULD) on the measurement host with a ternary-class model; MUST serve from the proved kernels; MUST NOT fall back to an unproved float path for a served token |
| R1 | a companion that remembers | R0, plus: MUST answer from a persistent memory (Apeiron table) that survives restart; MUST forget exactly on request (the entry is absent, not decayed); every memory write and forget MUST pass the policy engine |
| R2 | a living world (consumer: `../penglai`, whispering-tides) | R1 for ≥ 24 concurrent agents, each with its own memory and personality, on one GPU at interactive speed; per-agent state MUST be O(1) in lived tokens (recurrent state plus the table), not a growing KV cache; agent actions in the world MUST pass the policy engine |
| R3 | a continuous learner | R1, plus test-time learning (developed alongside R2, not after it; neither depends on the other); the state update MUST carry a proved contraction certificate (Lohmiller and Slotine, *On contraction analysis for non-linear systems*, Automatica 1998 — pointer, not retrieved), so a lifetime of updates stays bounded and perturbations decay |
| R4 | a worker (consumers: `../intent`, opencode) | MUST complete real coding and research tasks through tool use, lookup and iteration; one-shot capability is not required; every tool call MUST be a policy-engine decision with a named cause on deny |
| R5 | embodied | MUST acquire knowledge by reading, watching and playing, written to the store; actuator commands MUST pass the policy engine and carry the R3 contraction certificate |
| grail | one continuous entity with R0–R5 | — |

fold's `AC-PRODUCT` gate applies from R1 onward: coherent generation, store fetching and 30–60
tokens/s held at the same time, where a number that holds only two of them does not pass. So do the
failure modes its `STATE.md` records (mechanism ≠ product, component-met ≠ gate-met, proved ≠ the
right object). They are reference, not the surface: the trichotomy basis and the Aeneas link replace
the old basis.

## Dependency: the policy engine (another lane)

A policy engine is being built separately: study `docs/study/policy-engines.md`, epic
`docs/epics/policy-engine.md` (canonical `Apeiron.Policy.decide`,
Rust `src/policy`). This plan does not specify it; it consumes the verdict
`allow | attenuate | ask | deny(cause)` over a W-bit rights order. The consumer mapping:
- `allow` / `attenuate`: the effect proceeds with the granted rights;
- `ask`: the entity escalates to its owner and waits, which is how a Jarvis-style agent defers;
- `deny(cause)`: the effect is dropped and the cause is written to the agent's memory.

Lane coordination: that plan's W2 extraction and W3 `AxiomAudit` registration share
`proof/ApeironWord.lean`, `proof/regen.sh`, `proof/extract/lib.rs` and `AxiomAudit.lean` with this
plan's lanes. They are serialized by the lead between tasks, never run concurrently. It assumes the following interface properties, taken from the study's
recommendation:
- the decision point is a proved evaluator, and administration sits outside it;
- decisions are default-deny and fail-closed on unknowns, with a named cause on every deny;
- rights attenuate monotonically.

Design consequences for every track:
- **The entity is an enforcement-point client.** Every effect it takes goes through a decision:
  tool call, memory write or forget, world action, actuator command. No track may add an effect
  path that bypasses it.
- **Delegation attenuates.** Sub-agents and NPCs receive attenuated rights, never broader ones than
  their parent.
- **Denials are information.** A deny's named cause is recorded in the agent's memory, so the agent
  learns the boundary instead of retrying blindly.
- **Receipts are joined.** Decision receipts (the policy image digest per decision) join the
  engine's measurement receipts, so a transcript can be audited end to end.

## Prior work (reference, not the surface)

The programme has a prior AI corpus and measured results. They failed end to end on a basis without the trichotomy or an Aeneas link; mine them for pieces and failure modes, but do not constrain the design to them. Per `../roadmap.md` stages 2–3, none
of it is a current claim until it is re-bound to the canonical identity and, where it executes,
extracted. It is the starting point for every track below. The index of record is the prior-art
table in `docs/roadmap.md` ("Prior-art index").

| artifact | path (revision) | what it gives this plan |
|---|---|---|
| Paper I | `../sanguine-ref/release/paper1/INDEX.md` (`sanguine@e830d3fb`) | `spinor_subtrinity` with a forward-only exact gradient/Laplacian; `ridgeHead_unique`, `gram_stack`, `streaming_law`; the numerics seam (`flash_attention_exact`, the quant dual) — Tracks D and T |
| Paper II | `../sanguine-ref/release/paper2/INDEX.md` | closed-form loops (`constantStep_fixed_point`, `affine_orbit_closed_form`), determinism and confluence — Track C (looped cores) |
| Neural corpus | `../sanguine-ref/proof/Substrate/Algebra/Neural/` (43 modules) | `AttentionSpinor`, `SpinorRecall`, `SpeculativeGen`, `TrainingCollapse`, `TrainingBudget`, `MirasCountLM` (`hebb_comm`, `forget_hebb`, `surprise_eq_info`), `NullTransformer`, `SparseAttentionBudget`, `FrontierLadder` (the open seams) |
| fold | `../fold@d497161f`: `STATE.md`, `docs/ml-architecture-standard.md`, `modelfold/` (`examples/climbmix_train.rs`, `hellaswag.rs`, `piqa.rs`, `src/bin/harness.rs`), `results/` (61 records) | the served-model product gate `AC-PRODUCT` (coherent generation and store fetching together, at 30–60 tokens/s); the countlm harness; 8B/27B decode and 1M-context records |
| spinor-ttl | `../spinor-ttl@a05bb1b`: `PAPER.md`, `STATUS.md`, `TERNARY_FINDINGS.md`, `bench/` | measured on the RTX 5080: a ridge fold reaching the same optimum 16–71× faster than full-batch Adam (`bench/train_fold_vs_grad.py`); fold finetune vs LoRA with exact unlearning (`bench/ridge_vs_lora.py`, `finetune_race.py`); activation-weighted ternary descent from a 4-bit seed (`bench/ternary_descent.py`, `ternary_8b_vs_bnb.py`); seal vs NF4/Q4_K_M |
| data pins | `../sanguine-ref/data/MANIFEST.md` | ClimbMix (`nvidia/Nemotron-ClimbMix`); reasoning (`SupraLabs/reasoning-corpus-4K-5M-v1`); agentic traces (Kimi-K3, Fable-5); a Stack v3 subset; the stated starter mix for a small native-ternary conversational model |
| nursery | `../sanguine_nursery/` | reference-paper notes (`REFERENCES.md`, `EQUATIONS.md`) |

## Tracks

### Track T — transfer from existing models (the central path)

The most viable route to frontier capability on one RTX 5080 is not pretraining from scratch. It is
this:
1. convert existing open weights (Qwen, SmolLM and similar) into the proved basis;
2. distil from their weights and outputs;
3. train new models shaped to this programme's constraints (ternary, recurrent/hybrid, retrieval)
   from those teachers.

The spinor-ttl and fold records measure parts of this path.

- **Requirement:** a reproducible pipeline:
  1. teacher (a pinned open checkpoint);
  2. conversion into the basis (ternary descent, fold finetune, attention-to-recurrence transfer);
  3. student training on the pinned data mix (ClimbMix plus reasoning and agentic traces);
  4. served on the proved engine;
  5. scored by the pinned evaluation harness.
- **Deliverables:**
  - the conversion maps as extracted kernels with refinement theorems. Ternary descent's objective
    `tr[R G Rᵀ]` (`TERNARY_FINDINGS.md` §1) and the ridge fold's closed-form solve are stated and
    proved, and their Rust implementations refine them;
  - the distillation objective, stated;
  - the data mix, with each input's sha256 recorded in the receipt;
  - a teacher→student ladder at 135M → 0.5B → 1.5B → 8B, matching fold's ladder.
- **Falsification:**
  1. Conversion fidelity: a model converted into the basis must come within a stated margin of its
     source on the pinned suite, at no more resident bytes. Otherwise that conversion is rejected.
  2. Training economics: the fold's GPU-seconds advantage (16–71× at equal optimum) must reproduce
     on this host under Track E's protocol for each head type it is used on. Otherwise that head is
     trained by gradient.
  3. Student quality: each ladder rung must beat a same-size model trained or finetuned
     conventionally on the same data. Otherwise the rung is not promoted.
- **References:** spinor-ttl and fold above; Wang et al., *The Mamba in the Llama*, arXiv
  2408.15237; Zhang et al., *LoLCATs*, arXiv 2410.10254; Wu et al., arXiv 2508.09883
  (data-efficient reasoning distillation); DeepSeek-AI, arXiv 2501.12948 (distilling reasoning into
  small models).

### Track A — low-bit weights, rotation, fine-grained MoE (reproduce, then improve)

- **Requirement:** load and run a ternary-class GGUF end to end on the proved engine. First target:
  Bonsai 2 27B `PTQ1_0`/`PQ2_0`/`Q2_0` with its Walsh–Hadamard activation transform.
- **Deliverables:** `docs/epics/ternary-hadamard.md` W1–W5 (six dequant arms, exact integer FWHT,
  `prism.hadamard.*` loader, folded matmul path); then a fine-grained MoE router epic (top-k gating,
  shared experts) with a proved routing law.
- **Improvement hypotheses:**
  - an exact integer FWHT with one proved rounding, vs the fork's `log₂ n` float roundings;
  - ternary × integer activations as a multiplier-free inner product;
  - `H_2 = channelPair 1` at `σ = 1` (`Numeric/Hadamard/Sylvester.lean`), which puts the rotation inside the
    hyperbolic face.
- **Falsification:**
  1. Token parity: top-1 agreement with the fork below 99% on a fixed 256-prompt greedy set at equal
     weights means the engine or loader is wrong. It blocks the track until explained.
  2. Throughput: tokens/s on the host below the fork's CPU path on the same machine means the
     kernels are redesigned before any performance claim.
  3. Knowledge capacity: a closed-book fact-recall probe (fixed question set) comparing the ternary
     and fp16/Q8 versions of one model. A drop larger than the `log₂ 3 / 2` ratio predicts confirms
     R2's requirement and prioritizes Track B.
- **References:**
  - Ma et al., *The Era of 1-bit LLMs*, arXiv 2402.17764;
  - Ma et al., *BitNet b1.58 2B4T*, arXiv 2504.12285;
  - Ashkboos et al., *QuaRot*, arXiv 2404.00456 (Hadamard rotation for quantization);
  - Dai et al., *DeepSeekMoE*, arXiv 2401.06066;
  - `PrismML-Eng/llama.cpp@adfffbe` and `PrismML-Eng/Bonsai-demo@9ef3205` (read; notes in
    `/home/socol/oc-tmp/spike-prism/FINDINGS.md`);
  - upstream `../llama.cpp@ece963f41`.

### Track B — small reasoner, test-time compute, knowledge out of the weights

- **Requirement:** a small model (1–14B) whose reasoning quality comes from inference-time tokens,
  with factual knowledge served from an addressable store rather than the weights.
- **Deliverables:**
  1. Speculative decoding on the engine (draft model plus verifier). The exact integer engine makes
     acceptance deterministic; prove the acceptance rule preserves the target distribution.
  2. A verified retrieval memory on the Apeiron growable table (`Apeiron.Table` with its O(1)
     primitives and proved refinement) as the knowledge store the reasoner queries.
  3. A test-time compute harness: best-of-n and self-verification loops with fixed budgets.
- **Improvement hypothesis:** retrieval backed by a proved table removes the knowledge-capacity cap
  R2 derives for ternary cores. No other stack offers a proof-carrying memory.
- **Falsification:**
  1. Speculative decoding: speedup below 1.5× at the measured acceptance rate on the host means the
     draft/verifier pairing is redesigned.
  2. Retrieval: on a held-out fact set, a 1–3B reasoner with retrieval must beat the same model
     without it by a stated margin, and must match a 2–4× larger closed-book model. Failing both
     rejects the decoupling at that scale.
  3. Test-time compute: accuracy vs thinking-token budget on a fixed math/code set must rise
     monotonically to a plateau above the closed-book baseline. A flat curve rejects the reasoner
     choice.
- **References:**
  - Wu et al., *Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning*,
    arXiv 2508.09883 (0.8k curated examples; AIME, MATH-500, LiveCodeBench);
  - DeepSeek-AI, *DeepSeek-R1*, arXiv 2501.12948;
  - Allen-Zhu and Li, *Physics of Language Models 3.3*, arXiv 2404.05405;
  - Leviathan, Kalman and Matias, *Speculative Decoding*, arXiv 2211.17192;
  - Lightman et al., *Let's Verify Step by Step* (process supervision): arXiv ID `2305.20050` from
    memory, not verified; retrieve before citing.

### Track C — recurrent and hybrid cores by conversion (topology change without pretraining)

- **Requirement:** replace most attention layers of an existing open model with O(1)-state
  recurrent layers, keeping a few attention layers (hybrid). This gives decode KV that doesn't grow
  with context, and a core small enough to stay cache-resident when looped.
- **Deliverables:**
  1. A face-recurrence layer: a selective linear recurrence whose state transition is a `RingCell`
     product on a chosen face (elliptic: rotations, the RoPE and complex-diagonal SSM family;
     parabolic: dual-number shift updates; hyperbolic: two real decay channels, the idempotent
     split).
  2. Parallel training by associative scan, whose associativity is `token_step_fold` (`ringMul`
     associativity; `Nous/Model/State.lean`).
  3. Conversion of a 1–8B model by attention transfer plus LoRA.
  4. An extracted Rust kernel with a Lean refinement.
- **Falsification:**
  1. Quality: the converted hybrid (≥ 50% of layers recurrent) must come within 2 points of the
     teacher on a fixed eval set (5-shot MMLU subset plus a needle-in-a-haystack test at 8× the
     training length), within the published conversion budgets. Otherwise the conversion is
     rejected at that scale.
  2. Face value: the face-recurrence layer must beat a real-diagonal-decay baseline (Mamba-2-style
     scalar decay) at equal state size on multi-query associative recall. Otherwise the trichotomy
     parameterization adds nothing, and the track continues with the baseline.
  3. Looping: a looped core (one block iterated `r` times) must beat an unlooped model with the same
     `W_resident` on the reasoning set at equal wall-clock on the host, or looping is dropped.
- **References:**
  - Gu and Dao, *Mamba*, arXiv 2312.00752;
  - Dao and Gu, *Transformers are SSMs* (Mamba-2), arXiv 2405.21060;
  - Yang, Kautz and Hatamizadeh, *Gated Delta Networks*, arXiv 2412.06464;
  - Wang et al., *The Mamba in the Llama*, arXiv 2408.15237;
  - Zhang et al., *LoLCATs*, arXiv 2410.10254;
  - Geiping et al., *Recurrent Depth*, arXiv 2502.05171;
  - Kosowski et al., *The Dragon Hatchling*, arXiv 2509.26507 (local `../bdh@2b0d7a4`);
  - Katharopoulos et al., *Transformers are RNNs* (linear attention): arXiv ID `2006.16236` from
    memory, not verified.

### Track D — mechanisms from the algebra (research; proof first)

- **Requirement:** attention alternatives, and learning rules, whose state update is an associative
  operator with O(1) state. This makes training parallelizable by scan and inference O(1) per token.
  It also covers test-time training and forward-only learning rules stated as operators on
  `R[x]/(x²−σ)`, carried by `docs/epics/ai-track-reanchor.md`.
- **Deliverables:** for each candidate:
  1. the operator and its algebraic laws, proved in Lean (associativity, the norm-form invariant,
     the O(1) state bound);
  2. a Rust kernel refinement;
  3. a small-scale training run with a named baseline.
- **Falsification:** a candidate is archived, with its proof kept and its claim retired, if it fails
  either test:
  - multi-query associative recall at equal state size against gated delta rule and linear
    attention baselines;
  - loss at equal tokens on a fixed 125M–350M setup.

  Forward-only rules must also beat backpropagation-trained baselines at equal compute on the same
  setup, or stay research.
- **References:**
  - Sun et al., *Learning to (Learn at Test Time)*, arXiv 2407.04620;
  - Gated DeltaNet (above);
  - `TorchLean` scope reference (`../TorchLean@12f5c651`; arXiv 2602.22631 per
    `docs/epics/ai-track-reanchor.md`);
  - Hinton, *The Forward-Forward Algorithm*: arXiv ID `2212.13345` from memory, not verified;
  - the corpus's `softmax_attention_is_online_spinor` and `so2_spinor_recurrence` (existing online
    recurrences to build on).

### Track E — measurement and parity (gates every performance claim)

- **Requirement:** every tokens/s, memory or quality number above is `measured` under the host
  protocol, or it is not claimed.
- **Quality harness:** EleutherAI `lm-evaluation-harness` is the named quality gate. It is not
  installed on this host, so installing it with a pinned version is the first task. fold's
  `modelfold/src/bin/harness.rs` records that the product-quality grade is `lm_eval` scored on the
  *served* model against an fp reference, and that the grade is not yet reached. So the harness
  scores the proved engine through an adapter, not a separate float reference path.
  - **Task set** (hashed before any run): HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, MMLU,
    GSM8K, plus a long-context retrieval task.
  - **Survey first:** whether a newer harness has displaced `lm-evaluation-harness` needs retrieving
    before the pin; that is not established here.
- **Deliverables:**
  - G1 closure (`docs/evidence/open-area-performance.md`; 325 passing / 1125 failing pairs);
  - the ternary parity run;
  - a standing benchmark harness against the incumbents in the bench suite.
- **E0/E1 results (tested; scratch receipts in `/home/socol/oc-tmp/lm-eval/REPORT.md`):**
  - Harness: `lm-eval` 0.4.13 pinned in a Python 3.12 venv.
  - Engine adapter: `Engine::step_logits` (`src/engine/step.rs`), the JSON bridge
    `examples/lm_eval_bridge.rs` (removed; at 2f15d6a9), and an `@register_model("apeiron")` backend.
  - Parity on Qwen2.5-0.5B (`Q4_K_M` GGUF on the proved Q32.32 engine, vs the Hugging Face bf16
    reference at `060db649`): ARC-Easy and PIQA, first 20 documents each, identical accuracy
    (0.70/0.70) and **40/40 identical chosen answers**.
  - Control: logprob forced to 0 collapses ARC to the first-choice rate (0.15).
  - Limits: n = 20 is a smoke slice, not a metric, and the comparison bundles quantization with engine
    error.
  - Throughput 1.6 tok/s single-stream on the early CPU path (`E0/E1`); the GPU SPIR-V path has since closed the R0
    throughput gate at CUDA parity (`docs/plan.md` rows `DECODE-ATTRIB` and `INF-RESIDENT-PREFILL`; current
    numbers in `docs/evidence/perf-predictions.md` § Current, guarded by `SPEED-GUARD`
    `tools/gpu/speed_guard.py` (removed; at 2f15d6a9)).
- **Falsification:** the harness's own control, a deliberately slowed kernel, must show up in the
  measurement. Otherwise the instrument is invalid (AGENTS §10, instrument-defect row).

## Recursive self-improvement (RRSI) synthesis: why `1.18` b/w constructive transfer is the keystone

With the SPIR-V execution engine at CUDA parity (current numbers in `docs/evidence/perf-predictions.md` § Current)
and forward-only continuous learning (`LEARN-RIDGE` exact-Gram Cholesky fold and `LEARN-FOLDADAPT` exact-key table merge/unlearn
in `docs/plan.md` rows 244–245), `LEARN-ONEBIT` (`docs/evidence/onebit-w15.md`, `docs/evidence/cost-correspondence.md`,
`proof/corpus/Perf/CostCorrespondence.lean`) is the remaining piece required to run a local 27B–32B coding and research
entity (`../intent` RRSI loop) on a single 16 GB consumer GPU:

| Component | Artifact & status (`docs/plan.md`, `docs/evidence/onebit-w15.md`) | Role in local 27B RRSI |
|---|---|---|
| **Resident footprint & decode floor (`27B` at `1.18` b/w)** | `Perf.CostCorrespondence.{qwen27b_target_floor, tg_floor_ratio}` (`docs/evidence/cost-correspondence.md`): **`3.9825 GB`** weight footprint (`< 4.15` ms/tok roofline at `960` GB/s, `> 240` tok/s ceiling; `~140–180` tok/s at `60–75%` bandwidth efficiency) | Fits a `27B` model (`Qwen2.5/3-27B/32B`) plus `32k–64k` KV cache and workspace inside `16 GB` VRAM (`Q4_K_M` needs `~16.5 GB` weights alone and cannot fit) |
| **Scale collapse of the `1.18` b/w K-FAC floor (`0.5B → 8B → 32B`)** | `tools/onebit/rstar_{stream,fast}.py` (`docs/evidence/onebit-w15.md`): against `llama.cpp Q4_K_M` (`1.00×`), the `1.18` b/w separable floor drops from **`16.9× Q4_K_M`** (`0.5B`, `R*_sep = 3.1731` b/w) → **`5.27× Q4_K_M`** (`Qwen3-8B`, `R*_sep = 2.2530` b/w, `0.0989` nats) → **`3.78× Q4_K_M`** (`Qwen3-32B`, `R*_sep = 1.9927` b/w, `0.0913` nats, below 4.5-bit `NF4`'s `4.62× Q4_K_M`), while the 0-side-info oracle floor is **`2.11× Q4_K_M` on 8B** (`R*_orc = 1.5908` b/w) and **`~1.35×–1.50× Q4_K_M` on 27B/32B** | Because `R*_sep` and `R*_orc` fall with model width (`r99 / n` shrinks as width grows from `896` to `5120`), a constructive coder that sits near the K-FAC floor on `0.5B` lands inside the **`1.5×–2.5× Q4_K_M`** (`0.03–0.05` nats, `> 88%` top-1) envelope at `27B/32B` |
| **Constructive `1.18` b/w coder (`LEARN-ONEBIT` W15d §§9–20)** | `tools/onebit/fold_transfer.py` (removed; at 2f15d6a9) (`docs/evidence/onebit-w15.md`): 7-component trichotomy pipeline (`had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm` at **`1.1795` b/w**) combining `H₁₂₈/√128`, butterfly sign-group gauge, 2-plane full-attention hierarchical scales, the 1-byte `E₈ ∪ O₈` spinor + parabolic-sparse codebook, `beam=4` Cholesky trajectory search, MP-damped (`λ_icm = 0.20`) global 8-tuple ICM, Stage-1 drift attenuation (`inter_gamma_in = 0.25`), and Riemannian `RMSNorm` pullback sink gating + row-trust radial projection + coherence-anchored readout `fnorm` reaches **`0.3749` nats in-sample (`21.55× Q4_K_M`, `0.9646×` — `3.54%` BELOW the `0.3887` nats uniform-rate separable K-FAC floor; that `model` floor is `refuted` by `docs/evidence/receipts/onebit/t3-nb-3seed.json` at `0.35064`)** and **`1.3777` nats held-out (`69.93× Q4_K_M`, `42.19%` top-1, down from `14.86` nats on `gptq1-t` and `3.119` nats on `fold-block`)** in `93.9` s | Closes the in-sample constructive coding factor (`Factor 2: KL_in / D_sep = 0.9646×` uniform-rate); the remaining `0.5B` held-out gap is `72.8%` finite-sample cross-layer target drift (`Factor 1: KL_out / KL_in = 3.675×`) |
| **SPIR-V execution & forward-only adaptation** | `DECODE-ATTRIB`, `INF-RESIDENT-PREFILL` (current numbers in `docs/evidence/perf-predictions.md` § Current), `LEARN-RIDGE` (`FoldTrainer` closed-form Gram solve), `LEARN-FOLDADAPT` (`FoldAdapter` exact unlearning `fold(A+B) − fold(B) = fold(A)`) | Serves the `1.18` b/w core on Vulkan/SPIR-V and folds accepted candidate trajectories into the adapter/readout state in seconds without backprop |

## Sequencing

| order | work | why first |
|---|---|---|
| 0 | Track E quality harness (`lm-evaluation-harness` pinned, engine adapter) and the stage-2 rebind of the prior corpus | every later claim needs the harness; the prior work is the starting point |
| 1 | Track A through ternary-hadamard W5 (parity on Bonsai 2) | lowest risk; enables Track B's token budget; first consumer-device artifact |
| 1∥ | Track T conversion pipeline on the ladder's 135M–0.5B rungs (`LEARN-ONEBIT` W15d) | the central path; drives constructive `1.18` b/w transfer to the K-FAC floor |
| 2 | Track A MoE router; Track B speculative decoding | both reuse the engine; B's speedup multiplies A's |
| 3 | Track B retrieval memory on `Apeiron.Table` (`LEARN-FOLDADAPT`) | the repo's core object; answers R2's derived requirement |
| 4 | Track C conversion with the face-recurrence layer | Track T's pipeline applied to attention-to-recurrence transfer |
| 5 | Track D candidates | proof-first; empirical tests after Track C's harness exists |
| ∥ | Track E | runs alongside, on the quiet host only |

## Open questions

- **Training and conversion compute (resolved):** one RTX 5080. From-scratch pretraining is out of scope. Tracks
  C, D and T run as conversion, distillation, fold/closed-form heads (`LEARN-RIDGE`, `LEARN-ONEBIT` streaming block-wise
  Gram transfer) and `LEARN-FOLDADAPT` updates on 0.5B–32B models (`rstar_stream.py` and block-wise `fold_transfer.py`
  stream one block at a time on a single 16 GB GPU).
- **Product bar (adopted):** fold's `AC-PRODUCT` conjunction (coherent generation, store fetching,
  30–60 tokens/s at once) is the acceptance bar from R1 onward. R2 and R3 are concurrent (owner):
  Penglai drives R2 while test-time learning proceeds; neither is a hard dependency of the other.
- **Eval sets.** The fixed prompt, fact, math/code and recall sets named in the falsification rows
  must be chosen and hashed before any Track B or C run. Owner: lead.
- **Unverified references.** The three IDs marked "from memory" are retrieved and read before use.
  Owner: lead.
