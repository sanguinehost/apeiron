# Qwen3.5-0.8B hybrid decode on the exact engine

## Intent

- **Owner:** lead (AI track)
- **Status:** W0–W2 closed (exact path decodes llama.cpp's ids); W3 (speed) next
- **Source revision:** dirty tree over `74c037b`; model `ggml-org/Qwen3.5-0.8B-GGUF` `Qwen3.5-0.8B-Q8_0.gguf`
  (sha256 recorded by Q35-T0), config `Qwen/Qwen3.5-0.8B/config.json`

### Problem

The engine serves dense `qwen2` models only. Qwen3.5-0.8B is a natively trained hybrid: 24 layers, every
fourth a gated full-attention layer (6), the rest Gated DeltaNet (18) with a constant-size state. It is the
smallest checkpoint where an O(1)-state sequence mixer is trained, not converted (`docs/evidence/linearize-qwen05b.md`:
our conversion flipped 54–66% of ids and recalled 0/20 needles). Serving it on the exact integer path tests whether
the constant-size state keeps retrieval, and it is the base the sparse-MoE rung (OLMoE) builds on.

### Architecture (read from `llama.cpp` `src/models/qwen35.cpp` at `ece963f41`, and the GGUF metadata)

| part | shape and semantics |
|---|---|
| embedding / head | `token_embd` 248320 × 1024, tied (the LM head reads it every token: ≈1/3 of the weights) |
| block | `x += mixer(rms(x)·attn_norm)`; `x += ffn(rms(x)·post_attention_norm)`; SwiGLU 1024 → 3584 |
| gated attention (layers 3, 7, …, 23) | `attn_q` 1024 → 4096 = per head `[q(256) | gate(256)]`, 8 heads; `attn_k`, `attn_v` 1024 → 512 (2 KV heads × 256); per-head RMS `q_norm`, `k_norm`; RoPE on the first 64 of 256 dims, θ = 10⁷, sections [11, 11, 10] (text: all three position ids equal); `out = attn · sigmoid(gate)`, then `attn_output` 2048 → 1024 |
| Gated DeltaNet (other layers) | `attn_qkv` 1024 → 6144 = q(16×128) k(16×128) v(16×128); causal depthwise conv, kernel 4, state of 3 columns × 6144, then SiLU; L2-normalise q, k per head; `β = sigmoid(ssm_beta·x)`; `g = ssm_a · softplus(ssm_alpha·x + ssm_dt)` (`ssm_a` stores −exp(A_log)); per head `S ← e^g·S; S ← S + k ⊗ β(v − Sᵀk); o = Sᵀ(q/√128)`; `o ← rms(o)·ssm_norm ⊙ silu(z)` with `z = attn_gate·x` (2048); `ssm_out` 2048 → 1024 |
| MTP | `blk.24.nextn.*`: one extra full-attention block predicting token t+2; not executed in the main pass |

### Why the alternatives were rejected

| alternative | why not |
|---|---|
| convert Qwen2.5-0.5B to Gated DeltaNet ourselves | measured: 54–66% flips, 0/20 needle recall (`linearize-qwen05b.md`) |
| start with OLMoE | the user's ordering; a dense-FFN hybrid isolates the new mixer from routing |
| float GDN state | breaks the engine contract (no float on the executing path) and the bit-identity oracle |

### Completion condition

`target/release/examples/engine_decode <Qwen3.5-0.8B-Q8_0.gguf> <tokenizer.json> 64` decodes 64 tokens whose ids are
bit-identical across kernel modes, thread counts and E-core settings, and whose first-divergence index against the
float reference (`tools/qwen35/reference.py` (removed; at 2f15d6a9)) is recorded; tok/s measured against llama.cpp CPU on an idle host.

### Invariants

1. No float on the executing path; load-time float only where a rounding contract is stated (as `RopeTable`).
2. Every fast kernel is equal to the exact oracle (`matvec_fixed`, the scalar GDN step) on real payloads.
3. `qwen2` decode ids unchanged (`/tmp/opencode/l9/qwen-turns.txt`, `/tmp/opencode/idle/bonsai-1.txt`).
4. Every refusal is a typed error; no clamp, no wrap.

### Verification hypotheses

| claim | grade | evidence |
|---|---|---|
| float64 reference matches llama.cpp greedy ids | measured | Q35-T0 receipt |
| fixed-point `ln`/`softplus`/`sigmoid` within stated error on their windows | tested → proved | Q35-T2 tests; later Lean refinement as `fixed_exp` |
| GDN step: scalar exact step equals the chunked form | proved (model) | `Nous/Seq/GatedDelta.lean` `recurrent_eq_chunked` (exists); Rust link later |
| hybrid decode ids equal across kernels/threads | tested | Q35-T5 |
| tok/s vs llama.cpp (Q8_0 53.9, Q4_0 63.8 tok/s, 8 threads, load ≈2) | measured | idle-host A/B |

### Non-goals

- Vision tower, M-RoPE with distinct position ids, batching, prefill chunking.
- MTP speculative decoding (next rung: `Nous/Model/JacobiDecode.lean` `verify_accept_prefix` licenses it).
- `Q4_0` until the `Q8_0` path is green.

### Trust boundary

rustc/LLVM codegen, AVX2 intrinsics and the `unsafe` pool are tested, not proved. The GDN step is `tested` against
the scalar oracle until an extraction refinement lands. Float reference and llama.cpp are measurement instruments.

## Spec

- **Value domain.** Q32.32 `i64`, `i128` accumulation, one `round_shift_32` per product sum (engine convention).
- **GDN state.** Per layer: conv `3 × 6144` Q32.32, state `16 × 128 × 128` Q32.32 (2 MiB/layer, 36 MiB total, O(1)
  in context length). `‖q‖ = ‖k‖ = 1` after L2-norm and `β ∈ (0,1)`, `e^g ∈ (0,1]`, so `S` stays bounded by the
  sum of `|v|` over the window of non-decayed steps; overflow is a typed refusal.
- **New numerics.** `fixed_ln` on `(0, 2^31)`, `softplus(x) = x + ln(1 + e^{−x})` for `x ≥ 0` and `ln(1 + e^x)` for
  `x < 0`, `sigmoid(x) = 1/(1 + e^{−x})`, each with an error bound in units of `2^-32` stated and tested.
- **Partial RoPE.** Dims `0..64` of each 256-dim head rotate with the pairing llama.cpp uses for this rope type
  (Q35-T0 records it: NeoX halves `(i, i+32)` or adjacent pairs); dims `64..256` pass through. Angles come from the
  turn ring (`src/engine/rope.rs`), θ_j = 10⁷^(−2j/64).
- **Failure semantics.** Missing tensor, wrong shape, unsupported type → `EngineError`; arithmetic range → typed
  step error.

## Plan

| wave | task | owner | files | gate | control |
|---|---|---|---|---|---|
| W0 | Q35-T0 float64 reference + llama.cpp agreement | gemini-flash | `tools/qwen35/reference.py` (removed; at 2f15d6a9) (new), `docs/evidence/qwen35-reference.md` (new) | `python3 tools/qwen35/reference.py --check` exit 0 (ids = llama.cpp for 32 tokens) | swap RoPE pairing → ids diverge |
| W0 | Q35-T1 hybrid config + layer tensor index | deepseek-flash | `src/engine/hybrid/mod.rs` (new), `src/engine/hybrid/tests.rs` (new), `src/engine/mod.rs` | `cargo test --lib engine::hybrid` | drop a layer-kind rule → test red |
| W0 | Q35-T2 `fixed_ln`, `softplus`, `sigmoid` | gemini-flash | `src/numeric/fixed_ln.rs` (new), `src/numeric/mod.rs` | `cargo test --lib numeric::fixed_ln` | perturb a table constant → bound test red |
| W1 | Q35-T3 GDN scalar exact step + conv state | after T1, T2 | `src/engine/hybrid/gdn.rs` (new), `src/engine/hybrid/gdn_tests.rs` (new) | layer-0 real-weight step within tolerance of the T0 dump | drop the decay → red |
| W1 | Q35-T4 gated attention (partial RoPE, q/k norm, sigmoid gate) | after T1, T2 | `src/engine/hybrid/gattn.rs` (new), `src/engine/hybrid/gattn_tests.rs` (new) | layer-3 real-weight output within tolerance of the T0 dump | full-width RoPE → red |
| W2 | Q35-T5 hybrid step, decode example, ids | lead-assigned | `src/engine/hybrid/step.rs` (new), `examples/engine_decode.rs` (removed; at 2f15d6a9) | 64 ids identical across kernel modes; first divergence vs T0 recorded | — |
| W3 | speed: pooled GDN heads, chart kernels, LM head | later | — | idle-host A/B | — |

## Results

| task | artifact | gate | control | grade |
|---|---|---|---|---|
| Q35-T0 | `tools/qwen35/reference.py` (removed; at 2f15d6a9) (`a56c5d32`), `docs/evidence/qwen35-reference.md` | `python3 tools/qwen35/reference.py --check`: 32/32 ids = llama.cpp on both prompts | RoPE pairing adjacent → diverges at index 31 (worker); decay removed → diverges at index 0 (lead) | measured |
| Q35-T1 | `src/engine/hybrid/{mod,tests}.rs` | `cargo test engine::hybrid` 3 passed | kind rule `il % 4` (worker), KV-head count (lead) → red | tested |
| Q35-T2 | `src/numeric/fixed_ln.rs` (`e5113e68`) | ln ≤ 7, softplus ≤ 13, sigmoid ≤ 8 units of 2⁻³² on \|x\| ≤ 16 (≤ 484 in the flushed tail) | LN2 perturbed (worker), sigmoid denominator (lead) → red | tested |
| Q35-T3 | `src/engine/hybrid/gdn.rs` | real layer 0, 8 steps: output 1.3e-6, state 1.3e-9 vs f64; 22.9 ms/layer scalar | decay dropped (worker), q scale dropped (lead) → red | tested |
| Q35-T4 | `src/engine/hybrid/gattn.rs` (`cdc6115a`) | real layer 3, 8 positions: 2.0e-8 vs f64 | full-width RoPE (worker), gate halved (lead) → red | tested |
| Q35-T5 | `src/engine/hybrid/step.rs` (`64a94809`), `examples/hybrid_decode.rs` | 16/16 ids = llama.cpp on "The capital of France is"; lead: 32/32 ids = llama.cpp on the 200-token README prompt | FFN residual dropped → first token 466 ≠ 11751 | tested; measured |

| Q35-T6 | `step.rs`, `gattn.rs` (`gattn_step_with_mode`), `examples/hybrid_decode.rs --mode` | Q8Act ids = llama.cpp 32/32 on both prompts, identical at pool width 1 and full; `q8act_head_equals_dequant_oracle`; lead made `q8act_determinism_five_prompt_eight_gen` assert llama.cpp's ids | last head chunk skipped (worker) → oracle test red; K/V swapped in Q8Act attention (lead) → ids `[220, 21, 15, …]` | tested; measured |

Q8Act path (load 3.8–11): 3.50–3.79 tok/s; per token: DeltaNet 249–263 ms, FFN 9.2–14.0, head 3.7–4.4,
attention 1.5–3.4. DeltaNet is 93% of the token (Q35-T7).

Exact scalar path: 1.43–1.46 tok/s (683–700 ms/token, load 7.6); llama.cpp CPU Q8_0 53.9 tok/s.
Residuals: weights are loaded per state (`GdnState::open`, `GattnState::open`, the decoder's own store) instead of by
`HybridModel`; `GdnState` holds one layer, and the decoder swaps 18 per-layer states in.
