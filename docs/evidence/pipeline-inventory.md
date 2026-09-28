# Decode-pipeline inventory: one physical operation per row, both models

Task `APEIRON-DOM-MAP-T1`. Data for a visual map of the LLM decode step as implemented and proved
in this repository. One row per physical operation, in execution order, for both served models.
Every `path:line` cited here is checked to resolve by the gate script
`/home/socol/oc-tmp/dom-map-t1/check_paths.py` (see §9); a citation is a pointer, not a proof.

**Grade basis.** `proved` in the Lean-link column means the named theorem is kernel-checked in the
committed corpus under `{propext, Classical.choice, Quot.sound}`; this task re-ran no `lake build`
(§10 of the task: no builds). `tested` means a `cargo test` covers it; `measured` means a receipt
records a command's output; `model` is an algebraic reading, not correctness evidence. `NONE` is a
declared gap. Per AGENTS §5 a citation alone is not verified.

**Source identity.** `proof/ApeironWord.lean` sha256 `a4599646…` at task start. Recorded at the
final gate below; if the policy lane regenerates it, the gate is re-run on the new digest and both
reported.

## 1 · Verified model parameters

Read from the GGUF metadata by a direct header probe (read-only, no build), and cross-checked
against the loader's own key list: `src/engine/loader.rs:557` (`resolve_config`) reads
`{arch}.block_count`, `attention.head_count`, `attention.head_count_kv`, `embedding_length`,
`feed_forward_length`, `context_length`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`, and
derives `head_dim = embedding_length / head_count` (`src/engine/loader.rs:581`). The probe script is
`/home/socol/oc-tmp/dom-map-t1/gguf_probe.py`; its output is the source of the numbers below.

| symbol | Qwen2.5-0.5B Q4_K_M | Ternary-Bonsai-1.7B PQ2_0 | source |
|---|---|---|---|
| arch | qwen2 | qwen3 | `general.architecture` |
| L (block_count) | 24 | 28 | probe |
| E (embedding_length) | 896 | 2048 | probe |
| F (feed_forward_length) | 4864 | 6144 | probe |
| H (head_count) | 14 | 16 | probe |
| Hkv (head_count_kv) | 2 | 8 | probe |
| hd = E/H (head_dim) | 64 | 128 | derived |
| V (vocab) | 151936 | 151669 | `token_embd.weight` dims[1] |
| q_out = H·hd | 896 = E | 2048 = E | derived |
| kv_out = Hkv·hd | 128 | 1024 | derived |
| rope_theta | 1000000.0 | 1000000.0 | `rope.freq_base` |
| rms_eps | 1e-6 | 1e-6 | `attention.layer_norm_rms_epsilon` |
| context_length | 32768 | 32768 | probe |
| weight tie | yes (`output.weight` absent; LM head = `token_embd.weight`) | yes | probe (`src/engine/lm_head.rs:73`) |
| QK-norm | none (no `attn_q_norm`/`attn_k_norm`) | present (`blk.*.attn_q_norm.weight`, `attn_k_norm.weight`, width 128) | probe |
| attention bias | present (`attn_{q,k,v}.bias`, F32) | absent | probe |
| RoPE scaling | none | `rope.scaling.type = yarn`, factor 4.0, original_context_length 8192 | probe |
| Hadamard fold contract | absent | absent | probe (`prism.hadamard.*` not present) |

Verification note: the task statement gave `vocab 151936`, `14 heads/2 KV heads`, `head_dim 64`,
`24 layers`, `n_embd 896`, `n_ff 4864` for Qwen — all confirmed. For Bonsai it gave 28 layers,
2048, 6144, vocab 151669, QK-norm — confirmed; the head counts were omitted and are 16/8,
head_dim 128. The Bonsai `yarn` scaling is **not** on the executing path (see §6 gap): the engine
builds the plain `1/theta^(2j/hd)` table (`src/engine/step.rs:804`, `src/engine/rope.rs:56`).

## 2 · Execution order (both models, greedy, `KernelMode::Q8Act`, `Sched::Resident`)

The step is `Engine::compose_logits` (`src/engine/step.rs:1180`): geometry check, embedding, then
per layer `attn_norm → Q/K/V → (bias) → (QK-norm) → RoPE → KV append → attention → O → residual →
ffn_norm → gate/up → SwiGLU → down → residual`, then `output_norm → LM head → argmax`. This is the
fold `forward.rs::cpu_step_f32` / `attn.rs::cpu_hidden_with` order named in `src/engine/mod.rs:153`.

Notation: `t` = number of KV positions the query attends (`pos+1`), `E,F,L,V,H,Hkv,hd` as §1.
All values are `i64` Q32.32 (`ONE = 2^32`); accumulations are checked `i128`; each stage ends with
one `round_shift_32` (`src/engine/value.rs:10`), i.e. round-to-nearest `>> 32`.

| # | op | Rust fn (path:line) | Lean link · theorem, grade | shapes / integer domain | MACs/token | bytes/token | t-complexity | parallel dispatch (barrier) | measured ms/tok | algebraic reading |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | token-embedding lookup + row dequant | `Engine::token_embedding` `src/engine/step.rs:733`, `embedding_sched` `src/engine/step.rs:740`; `dequant_row_fixed` `src/engine/dequant.rs:93`; Q8_0 row `dequant_q8_0_row` `src/engine/dequant.rs:224` | `Numeric/DequantQ8_0.lean` · `dequant_row_fixed_q8_0` (proved); row-pointer arithmetic **NONE** | row `[E]` Q8_0 → `[E]` Q32.32; block 32 codes × exact f16 scale | 0 (lookup) | row `n_embd/32·34` = **952** Q; **544** B | O(E) | serial | 0.001 Q / 0.002 B | injective row read; dequant = exact lattice scale×code |
| 2 | `attn_norm` RMSNorm + gain | `rms_with_gain_slice` `src/engine/step.rs:234`; `rms_norm_fixed` `src/engine/norm.rs:55`; `fixed_rsqrt` `src/numeric/fixed_rsqrt.rs:73`; gain mult `src/engine/step.rs:242` | `Numeric/EngineNormRefinement.lean` · `rms_norm_refines_proved`; `Numeric/RmsNormIdeal.lean` · `rms_norm_refines_ideal`; `Numeric/RsqrtRefinement.lean` · `fixed_rsqrt_refines_proved` (proved) | `[E]`→`[E]`; mean square `round(Σx²/(n·2³²))`; window `[2⁻²⁰,2²⁰]` | 2E sq+out + E gain = 3E = **2,688** Q / **6,144** B | — (in-memory gain, read at load) | O(E) | serial | 0.071 Q / 0.181 B | radial projection to a sphere S^{E−1}; gain = anisotropic scaling |
| 3 | Q/K/V projections (fused) | `qkv_q8act_at` `src/engine/step.rs:533`; `quant_act_pooled` `src/engine/chart/ffn.rs:59`; `projections_q8act` `src/engine/chart/ffn.rs:511`; row cores `q8act_q5_0_row` `src/engine/chart/mod.rs:474`, `q8act_q8_0_row` `src/engine/chart/mod.rs:390` | matvec: `Numeric/EngineMatvecRefinement.lean` · `matvec_refines_proved`; `Numeric/EngineDotRefinement.lean` · `row_dot_refines_proved`; chunk: `Numeric/MatvecChunk.lean` · `matvec_fixed_chunks`; quant_act exact refinement **NONE**; chart-Q8Act equality `tested` (`src/engine/chart/tests.rs`) | `[E]`→`[q_out]`,`[kv_out]`,`[kv_out]`; weight rows Q5_0/Q8_0; codes `|xq|≤127`, `i128` acc | E·(q_out+2·kv_out) = **1,032,192** Q / **8,388,608** B | Q5_0 616·(896+128+896)=1,184,704 + Q8_0 952·128=121,856 → **1,306,560** Q; B 544·(2048+1024+1024)=**2,228,224** B | O(n_out·E/t) independent of t | `Pool::run` over `min(2·width,blocks)` then `projections_q8act_prepared_into` row chunks (barrier) | 0.412 Q / 1.506 B | linear map Z^E→Z^n_out with one round = nearest-point map |
| 4 | Qwen attention bias add | `add_assign_fixed` `src/engine/step.rs:220` | **NONE** | `[q_out]`,`[kv_out]` Q32.32; checked add | q_out+2·kv_out = **1,152** Q / **3,072** B | — (bias at load) | O(q_out) | serial | 0.028 Q (none B) | translation |
| 5 | Qwen3 per-head QK-norm | `qk_norm_and_rope` `src/engine/step.rs:357`; `qk_norm_heads` `src/engine/loader.rs:240`; `rms_norm_fixed` (`src/engine/norm.rs:55`) | `EngineNormRefinement` covers the per-row norm; the fused head split **NONE** | per head `[hd]`→`[hd]` ×(H+Hkv); gain `[hd]` | 3·(q_out+kv_out) = Q 0 / B **9,216** | — (gain at load) | O(q_out+kv_out) | inside the head dispatch (row 6) | 0.073 Q / 0.174 B (shared with RoPE) | radial projection per head to S^{hd−1} |
| 6 | RoPE rotation | `qk_norm_and_rope` `src/engine/step.rs:357`; `RopeTable::rotate_one` `src/engine/rope.rs:181`; `rotate_head` `src/engine/rope.rs:215`; `fixed_sin_cos` `src/numeric/fixed_trig.rs:148` | `Numeric/EngineRopeRefinement.lean` · `rope_pair_spec_proved`, `rotate_head_loop_spec`; `Numeric/Rope/Part4.lean` · `fixed_sin_cos_refines_proved`; `Numeric/RopeHeadContent.lean` · `rotate_head_content` (proved) | pair `(j,j+hd/2)`; angle `pos·freq_j`; trig window `[−4096,4096]` rad | 4·pairs + pairs = 5·(q_out+kv_out)/2 = **2,560** Q / **7,680** B mults | — (table at load) | O(q_out+kv_out) | `attn_heads` ×2 (query, key) — `src/engine/chart/ffn.rs:689`; barrier | 0.073 Q / 0.174 B | multiplication by unit complex numbers on each 2-plane = rotation on torus T^{hd/2} (**model**) |
| 7 | KV append | `KvState::append` `src/engine/attn.rs:359` | `Numeric/EngineSamplerRefinement.lean` · `word_kv_append_ok`; `Numeric/KvReachable.lean` · `KvReachable_append_geometry` (proved) | writes `[kv_out]` key + `[kv_out]` value at `pos`; capacity check | 0 | writes 2·kv_out Q32.32 = **1,024** Q / **8,192** B (write) | O(kv_out) | serial | 0.022 B / incl. resid Q | list append / ring buffer |
| 8 | attention score | `KvState::attn_head` `src/engine/attn.rs:432`; `score_at` `src/engine/attn.rs:101`; `row_dot_fixed` `src/engine/matvec.rs:86` | `Numeric/AttnLoopScore.lean` · `attn_head_loop0_spec`; `Numeric/AttnScoreStage.lean` · `attn_score_stage`; `Numeric/EngineAttnRefinement.lean` · `word_attn_head_ok` (proved) | `q[hd]·k[hd]` → Q32.32; scale `fixed_rsqrt(hd)`; `i128` acc | H·hd·t = q_out·t = **896t** Q / **2048t** B | reads keys `Hkv·hd·t·8` = **1024t** Q B / **8192t** B B | O(t) | `attn_heads` n_head tasks (`src/engine/chart/ffn.rs:689`) | 0.304 Q / 0.879 B (with softmax+value) | bilinear form / inner product |
| 9 | softmax (`fixed_exp`) | `attn_head` (`src/engine/attn.rs:432`); `exp_at` `src/engine/attn.rs:127`; `weight_at` `src/engine/attn.rs:143`; `fixed_exp` `src/numeric/fixed_exp.rs:176`; `div_round_nearest` `src/engine/value.rs:5` | `Numeric/EngineAttnRefinement.lean` · `exp_at_window`, `weight_at_spec`; `Numeric/Exp/Part4.lean` · `fixed_exp_refines_proved`; `Numeric/AttnSoftmaxPerturb.lean` · `attn_stage_compose` (proved) | `[t]` scores → weights summing `≈2³²`; window `[−16·ONE,0]`, underflow → 0 | exp calls = H·t = **14t** Q / **16t** B; adds 2·H·t | reads none (registers) | O(t) | inside the head task | see row 8 | Gibbs measure p∝exp(shifted score); `fixed_exp` = integer evaluation |
| 10 | value mix | `value_component` `src/engine/attn.rs:180` | `Numeric/AttnLoopOutput.lean` · `value_component_spec`, `attn_head_loop3_spec`; `Numeric/AttnHeadWindowsValue.lean` · `attn_value_components` (proved) | `Σ_s p_s·v_s` in `i128` → `[hd]` Q32.32 | H·hd·t = **896t** Q / **2048t** B | reads values `Hkv·hd·t·8` = **1024t** Q B / **8192t** B B | O(t) | inside the head task | see row 8 | convex combination (weights ≈ a partition of 1) |
| 11 | head dispatch (schedule) | `attn_heads` `src/engine/chart/ffn.rs:689`; `Pool::run` `src/engine/chart/pool.rs:206` | **NONE** (schedule; equality is `tested`) | `out[q_out]`, one task per query head | — | — | O(n_head) scheduling | n_head tasks round-robin across `width`; barrier = `Pool::run` spin-wait | — | — |
| 12 | O projection | `proj_q8act_at` `src/engine/step.rs:505`; `proj_q8act_quantified` `src/engine/step.rs:321`; `matvec_q8act_quant_chunked` `src/engine/chart/ffn.rs:223` | as row 3 (matvec/dot/chunk); quant_act refinement **NONE** | `[q_out]`→`[E]`; weight Q5_0 / PQ2_0 | E·E = **802,816** Q / **4,194,304** B | Q5_0 616·896 = **551,936** Q; B 544·2048 = **1,114,112** B | O(E²) | `Pool::run` quant + prepared-dispatch (barrier) | 0.287 Q / 0.752 B | linear map |
| 13 | residual O add | `add_assign_fixed` `src/engine/step.rs:220` (call `:1312`) | **NONE** | `[E]` checked add | E = **896** Q / **2,048** B | — | O(E) | serial | 0.021 Q | translation |
| 14 | `ffn_norm` RMSNorm + gain | `rms_with_gain_slice` `src/engine/step.rs:234` (call `:1318`) | as row 2 (proved) | `[E]`→`[E]` | 3E = **2,688** Q / **6,144** B | — | O(E) | serial | 0.070 Q / 0.178 B | radial projection to S^{E−1} |
| 15 | `quant_act(h)` + gate/up projections | `ffn_q8act_at` `src/engine/step.rs:585`; `ffn_q8act_prepared_into` `src/engine/chart/ffn.rs:863`; `quant_act_pooled` `src/engine/chart/ffn.rs:59`; row cores (row 3) | as row 3; quant_act exact refinement **NONE** | `[E]`→`[F]`,`[F]`; Q5_0 / PQ2_0 | 2·F·E = **8,716,288** Q / **25,165,824** B | Q5_0 616·F·2 = **5,992,448** Q; B 544·6144·2 = **6,684,672** B | O(F·E) | quant dispatch + fused pair dispatch (barrier) | 4.106 Q / 5.719 B (gate+up+swiglu+down, row 16/17) | linear map; gate/up are the two `R[x]/(x²−σ)`-free scalar branches |
| 16 | SwiGLU elementwise | `swiglu_hidden` `src/engine/chart/ffn.rs:148`; `swiglu_one` `src/engine/chart/ffn.rs:126`; `silu_fixed` `src/engine/ffn.rs:66` | `Numeric/EngineSiluRefinement.lean` · `silu_window_refines`; `Numeric/SiluSaturation.lean` · `silu_refines_total`; `Nous/Model/SiluChannel.lean` · `silu_gate_eq_channelWeight` (proved) | `h_i=round(silu(gate_i)·up_i/2³²)`; Silu window `−21·ONE<v<16·ONE` else saturates | F products + F silu (1 exp, 1 div each) = **4,864** Q / **6,144** B products; exp F×L = **116,736** Q / **172,032** B | — | O(F) | `pool.chunks_mut` `≈2·width` tasks (barrier) | see row 15 | gating; `silu(v)=v·σ(v)` is exactly a channel weight (proved identity) |
| 17 | FFN down projection | `ffn_q8act_prepared_into` `src/engine/chart/ffn.rs:863`; `projections_q8act_prepared_into` `src/engine/chart/ffn.rs:555` | as row 3 | `[F]`→`[E]`; Q6_K / PQ2_0 | E·F = **4,358,144** Q / **12,582,912** B | Q6_K 3990·896 = **3,575,040** Q; B 1632·2048 = **3,342,336** B | O(E·F) | quant(hidden) dispatch + down dispatch (barrier) | see row 15 | linear map |
| 18 | residual FFN add | `add_assign_fixed` `src/engine/step.rs:220` (call `:1360`) | **NONE** | `[E]` checked add | E = **896** Q / **2,048** B | — | O(E) | serial | 0.016 Q / 0.041 B | translation |
| 19 | `output_norm` RMSNorm + gain | `rms_with_gain_slice` `src/engine/step.rs:234` (call `:1368`) | as row 2 (proved) | `[E]`→`[E]` | 3E = **2,688** Q / **6,144** B | — | O(E) | serial | 0.003 Q / 0.006 B | radial projection to S^{E−1} |
| 20 | LM head | `lm_head_fixed_sched` `src/engine/step.rs:845`; `head_payload_len` `src/engine/lm_head.rs:73`; `proj_q8act_quantified` `src/engine/step.rs:321`; `matvec_q8act_quant_chunked` `src/engine/chart/ffn.rs:223` | `Numeric/EngineLmHeadRefinement.lean` · `lm_head_refines_proved` (proved); matvec/dot as row 3 | `[E]`→`[V]`; tied `token_embd` Q8_0 / PQ2_0 | V·E = **136,134,656** Q / **310,618,112** B | Q8_0 V·952 = **144,643,072** Q; B PQ2_0 V·544 = **82,507,936** B | O(V·E) | `Pool::run` quant + row chunks (barrier) | 2.174 Q / 1.367 B | linear map |
| 21 | argmax (greedy sampler) | `argmax_fixed` `src/engine/sampler.rs:41`; call `src/engine/step.rs:1130` | `Numeric/EngineSamplerRefinement.lean` · `word_argmax_fixed_spec`; `Numeric/PrunedArgmax.lean` · `argmax_fixed_returns_lowest`, `word_argmax_fixed_returns_lowest` (proved) | `[V]` Q32.32 → one index; ties to lowest index | V−1 comparisons = **151,935** Q / **151,668** B | — | O(V) | serial | ≤0.003 | zero-temperature limit of the softmax (Gibbs) measure (**model**); tie rule proved |
| 22 | PQ2_0 LUT row kernel (Bonsai only) | `Pq2Lut::from_pq2_0` `src/engine/kern/lut.rs:600`; `RowKernel::rows` `src/engine/kern/lut.rs:1207`; `matvec_lut_rows` `src/engine/kern/lut.rs:1179`; `rows_core` `src/engine/kern/lut.rs:1028`; `build_tables_pooled` `src/engine/kern/lut.rs:216` | `Numeric/LutPQ2.lean` · `block_dot_lut`, `planeSum_bound`, `lutEntry_mirror`; `Numeric/DequantQ1Q2.lean` · `dequant_row_fixed_pq2_0`; refusals `Numeric/DequantLowBitRefusals.lean` · `dequant_pq2_0_row_*` (proved) | 128-weight block, 2-bit codes, `inner≤508`; two byte planes; `i16`/`i64` lanes; AVX2 twin `tested` | one lookup sum per 4-group per row (not 4 mults) | reads packed planes: codes `n_in/4` + scales `n_in/16` ≈ **640 B/row** B (vs 544 raw) | O(n_in/4) per row | rows split as row 3; tables built once per dispatch | see row 3/12/20 | exact factorization `Σw·x̂=d_b·Σ S_b·inner_sb` over the activation lattice |
| 23 | folded activation (Hadamard) | `folded_activation` `src/engine/step.rs:260`; `fwht_fixed` `src/engine/hadamard.rs:195` | `Numeric/HadamardRefinement.lean` · `fwht_int_spec`; `Numeric/HadamardFixedSpec.lean` · `fwht_fixed_refines`; `Nous/Model/RotationStage.lean` · `hadamardRef_isometry_l2` (proved) | `[n]` in place, `n=2^k` block; no-op on both models (no contract) | 0 on both paths | 0 | O(n log n) if present | serial | 0 | hypercube Fourier transform (Sylvester matrix); on-path no-op |

Load-time-only operations (once per `Engine::open`, not per token): GGUF header parse
(`src/format/gguf.rs`), config resolve (`src/engine/loader.rs:557`), tensor table
(`src/engine/loader.rs:640`), `DecodeVectors` gains/biases (`src/engine/loader.rs:809`),
`RopeTable::new` (`src/engine/rope.rs:56`), `eps_fixed` (`src/engine/step.rs:930`),
`resolve_layers` (`src/engine/step.rs:435`), and the Bonsai `PQ2_0` repack
(`Pq2Lut::from_pq2_0`, `src/engine/kern/lut.rs:600`, cached by `Engine::pq2_lut`
`src/engine/loader.rs:860`). The LUT repack is `O(n_out·n_in)` once per tensor; every later token
reuses the cached table.

## 3 · Quantizer / dequantizer per format on the path

| role | Qwen Q4_K_M | Ternary-Bonsai PQ2_0 | Lean |
|---|---|---|---|
| weight formats in layer 0 | attn_q/k/output, ffn_gate/up Q5_0; attn_v, token_embd Q8_0; ffn_down Q6_K | all seven layer tensors + token_embd PQ2_0 | probe |
| weight dequant (Exact path / fallback) | `dequant_row_fixed` `src/engine/dequant.rs:93` → `dequant_q5_0_row` `:262`, `dequant_q8_0_row` `:224`, `dequant_q6_k_row` `:528` | `dequant_pq2_0_row` `src/engine/dequant_lowbit.rs:221` | `Numeric/DequantQ5_0.lean`, `DequantQ8_0.lean`, `DequantQ6K.lean`, `DequantQ1Q2.lean` (proved) |
| Q8-act row core (chart) | `q8act_q5_0_row` `src/engine/chart/mod.rs:474`, `q8act_q8_0_row` `:390`, `q8act_q6_k_row` `:673`; fused plan `q8act_rows` `:1334` | — (LUT path) | chart equality `tested` (`src/engine/chart/tests.rs`) |
| activation quantizer | `quant_act` `src/engine/chart/mod.rs:205`, pooled `quant_act_pooled` `src/engine/chart/ffn.rs:59`: 32-wide block, `S_b=ceil(absmax/127)`, codes `[−127,127]` | same | `Numeric/ChartDot.lean` · `qcode_range`, `row_err_bound` (proved); `Numeric/ColumnCover.lean` · `quant_locality` (integer spec, proved) |
| LUT kernel | — | `Pq2Lut::from_pq2_0` `src/engine/kern/lut.rs:600`; `matvec_lut` `:932`; `matvec_lut_chunked` `:1133` | `Numeric/LutPQ2.lean` (proved) |
| exact reference kernel | `matvec_fixed` `src/engine/matvec.rs:149` + `row_dot_fixed` `src/engine/matvec.rs:86` | same | `Numeric/EngineMatvecRefinement.lean`, `EngineDotRefinement.lean` (proved) |
| fused row (Exact resident) | `matvec_fused` `src/engine/fused.rs:663` (Q5_0/Q8_0/Q6_K) | falls back | equality `tested` |

Q8Act contract: `matvec_q8act = matvec_fixed ∘ dequant ∘ quant_act` bit for bit (module doc
`src/engine/chart/mod.rs:1`); the block-partitioned quant and the row/chunk split are bit-identical
to the single call by `Numeric/ColumnCover.lean` `quant_locality` and `column_cover_round`
(integer spec).

## 4 · Parallel schedule and barriers

- Pool: `Pool::global()` `src/engine/chart/pool.rs:172`; workers = `APEIRON_THREADS − 1` (default 7),
  width = workers + caller (`src/engine/chart/pool.rs:192`), capped at the P-core count read from
  `/sys/devices/cpu_core/cpus` (`performance_cpus` `src/engine/chart/pool.rs:334`). Worker `w` pins
  to the `(w+1)`-th P-core, the caller to the first (`src/engine/chart/pool.rs:122`).
- Dispatches: `Pool::run(n, f)` `src/engine/chart/pool.rs:206` partitions `0..n` round-robin
  (`run_partition` `src/engine/chart/pool.rs:92`); every participant runs its share, the caller
  first, then everybody spin-waits its `done` word. **Each `Pool::run` is a barrier.**
- Per layer, in order: `quant_act_pooled(x)` (tasks `min(2·width, blocks)`), Q/K/V
  `projections_q8act` (row chunks), `qk_norm_and_rope` → `attn_heads` on query then key (n_head/n_head_kv
  tasks), attention `attn_heads` (n_head tasks), O `proj_q8act` (quant + row chunks), FFN
  `quant_act(h2)`, gate/up dispatch, `swiglu_hidden` (`≈2·width` tasks), `quant_act(hidden)`, down
  dispatch.
- Measured dispatch count (throwaway instrumentation, `tput-t15.out:2030`, `:3081`):
  **267 Pool::run barriers/token for Qwen (24 layers), 311 for Bonsai (28 layers)** — ≈11/layer plus
  3 (LM head + adjacent) — and `caller-serial` 0.74 ms/tok Qwen, 1.35 ms/tok Bonsai
  (`tput-t15.out:2031`, `:3082`).
- Row chunk count: `row_chunk_budget(n_out)` `src/engine/par.rs:23` = `min(available_parallelism,
  n_out/64)`, capped at `Q8ACT_MAX_CHUNKS = 8` (`src/engine/chart/ffn.rs:201`) for the chart kernel.
- The three dispatches the T15 measurement removed as serial: `quant_act` (1.8 ms/tok Bonsai) and
  `build_tables` (1.2 ms/tok Bonsai) (`tput-t15.out:117`).

## 5 · Measured ms/token (receipts)

From `docs/evidence/decode-attribution.md` (i9-14900KF, 8 P-cores, host read 82.0 GB/s, host load
1.0–2.4): Qwen 128.2/137.7/133.0/127.2/128.1 tok/s → **7.3–7.8 ms/tok**; Bonsai
82.0/81.8/91.3/81.9/84.3 tok/s → **11.0–12.2 ms/tok** (`docs/evidence/decode-attribution.md:13`).
Ratios 1.07 (Qwen) and 0.87 (Bonsai) against the forks.

Per-phase wall (paired runs, `APEIRON_KERNEL=q8act`, 64 tokens):

| phase | Qwen ms/tok (`tput-t15.out:2016`) | Bonsai ms/tok (`tput-t15.out:1228`) |
|---|---:|---:|
| embedding | 0.0009 | 0.0017 |
| attn_norm | 0.0708 | 0.1808 |
| qkv | 0.4118 | 1.5059 |
| qkv_bias | 0.0279 | 0.0012 |
| qknorm_rope | 0.0729 | 0.1739 |
| attn_core | 0.3038 | 0.8894 |
| o_proj | 0.2866 | 0.7518 |
| resid_o | 0.0208 | — (in resid) |
| ffn_norm | 0.0696 | 0.1776 |
| ffn (gate+up+swiglu+down) | 4.1064 | 5.7191 |
| resid_ffn | 0.0156 | 0.0409 |
| out_norm | 0.0029 | 0.0064 |
| lm_head | 2.1740 | 1.3674 |
| **total wall** | **≈7.57** | **≈10.81** |

The earlier Bonsai split (before the T15 serial-time cut) is `tput-t9b.out:2064`:
qkv 2.8537, qknorm_bias_rope 0.9307, attn_core 0.8673, o_proj 1.4827, ffn_gate_up_down 11.1359,
lm_head 2.4447, total ≈16.9 ms/tok; its Step-0 table is `tput-t9b.out:245`. The `pool::worker` loop
is ≈48% of Bonsai decode cycles (`docs/evidence/decode-attribution.md:41`), so dispatch cost bounds
Bonsai there.

## 6 · Lean modules touching the pipeline

`Numeric` and `Nous` are globbed in `proof/corpus/lakefile.lean:116,121`, so `lake build
Numeric.<Name>` resolves. Grades are the corpus's recorded grades (not re-run here, §0).

**Numeric — dequant / format**
- `Numeric/DequantRefinement.lean` — `f16_to_fixed_exact`, `read_u16_ok`, `dequant_f16_row_exact`, `dequant_row_fixed_f16`, `matvec_domain_f16_witness`.
- `Numeric/FormatRefinement.lean` — `word_q4_k_codes_ok`, `word_q6_k_codes_ok`, `word_gguf_parse_header_*`.
- `Numeric/DequantQ8_0.lean` — `dequant_row_fixed_q8_0`, `q8_0BlockDomain_witness`.
- `Numeric/DequantQ5_0.lean` — `dequant_row_fixed_q5_0`.
- `Numeric/DequantQ4K.lean` / `Numeric/DequantQ4KBlock.lean` — `dequant_row_fixed_q4_k`, `q4_k_block_spec`.
- `Numeric/DequantQ6K.lean` — `dequant_row_fixed_q6_k`.
- `Numeric/DequantQ2K.lean`, `Numeric/DequantQ3K.lean`, `Numeric/DequantQ3KBlock.lean` — `dequant_row_fixed_q2_k`, `dequant_row_fixed_q3_k`.
- `Numeric/DequantQ1Q2.lean`, `Numeric/DequantQ1Q2Block.lean` — `dequant_row_fixed_q1_0`, `dequant_row_fixed_q2_0`, `dequant_row_fixed_pq2_0`, `pq2_0_block_ok`.
- `Numeric/DequantTQ1_0.lean`, `Numeric/DequantTQ1_0Row.lean`, `Numeric/DequantTQ2_0.lean`, `Numeric/DequantPTQ1_0.lean` — `dequant_row_fixed_tq1_0`, `dequant_tq1_0_row_spec`, `dequant_row_fixed_tq2_0`, `dequant_ptq1_0_row_spec`.
- `Numeric/DequantRefusals.lean`, `Numeric/DequantLowBitRefusals.lean` — the short-payload/non-finite-scale refusals per family.
- `Numeric/DequantKWitness.lean` — `q4kWitness_row`, `q6kWitness_row`.
- `Numeric/Scale.lean` — `q32_pos`, `q32_ne_zero`.

**Numeric — projections / dot / chart**
- `Numeric/EngineMatvecRefinement.lean` — `matvec_refines_proved`.
- `Numeric/EngineDotRefinement.lean` — `row_dot_refines_proved`, `dot_rounding_bound`.
- `Numeric/MatvecChunk.lean` — `matvec_fixed_chunks`, `matvec_fixed_rows`.
- `Numeric/MatvecBridge.lean` — `matvec_stage_proved`, `rowDotReal_eq_matvecRef`.
- `Numeric/MatvecBridgeGrid.lean` — `matvec_stage_grid_proved`, `gridOut_spec`.
- `Numeric/ChartDot.lean` — `quantBlock_idem`, `dequant_block_factor`, `qcode_range`, `row_err_bound`.
- `Numeric/ColumnCover.lean` — `column_cover_round`, `quant_locality`, `dequant_locality`, `split_ffn_exact`.

**Numeric — norms / rsqrt / silu / exp / trig**
- `Numeric/EngineNormRefinement.lean` — `rms_norm_refines_proved`.
- `Numeric/RmsNormIdeal.lean` — `rms_norm_refines_ideal`, `rms_ideal_element_bound`.
- `Numeric/RsqrtRefinement.lean` — `fixed_rsqrt_refines_proved`, `isqrt_spec`.
- `Numeric/EngineSiluRefinement.lean` — `silu_window_refines`, `silu_tail_refines`.
- `Numeric/SiluSaturation.lean` — `silu_refines_total`.
- `Numeric/ExpRefinement.lean` (statement) and `Numeric/Exp/Part1.lean`–`Part4.lean` — `fixed_exp_refuses_proved`, `word_fixed_exp_value`, `fixed_exp_refines_proved`.
- `Numeric/Rope/Part1.lean`–`Part4.lean` — `fixed_sin_cos_refuses_proved`, `word_fixed_sin_cos_value`, `fixed_sin_cos_refines_proved`.

**Numeric — rope / attention**
- `Numeric/EngineRopeRefinement.lean` — `rope_pair_spec_proved`, `rotate_head_loop_spec`.
- `Numeric/RopeRotate.lean` — `rotate_loop_spec`, `rope_two_head_witness`.
- `Numeric/RopeHeadContent.lean` — `rotate_head_content`, `rotate_head_bound`.
- `Numeric/RopeHeadWitness.lean` — `rope_head_domain_witness`, `rope_head_witness_ok`.
- `Numeric/RopeRefinement.lean` — (assembly module; no `^theorem`).
- `Numeric/EngineAttnRefinement.lean` — `word_attn_head_ok`, `softmaxWeights_shift_invariant`, `weight_at_spec`, `exp_at_window`.
- `Numeric/AttnLoopScore.lean` — `attn_head_loop0_spec`, `score_at_spec`.
- `Numeric/AttnLoopOutput.lean` — `attn_head_loop3_spec`, `value_component_spec`.
- `Numeric/AttnHeadWindowsScore.lean`, `Numeric/AttnHeadWindowsValue.lean` — `attn_v2_loop0_domain`, `attn_value_components`.
- `Numeric/AttnScoreStage.lean` — `attn_score_stage`, `attnScore_round_close`.
- `Numeric/AttnSoftmaxPerturb.lean` — `attn_stage_compose`, `attn_domain_v2_witness`.
- `Numeric/AttnHeadRefines.lean` — `attn_head_refines_v2_proved`.
- `Numeric/KvReachable.lean` — `attn_domain_v2_structural`, `attn_head_refines_reachable`.

**Numeric — LM head / sampler / Hadamard / LUT**
- `Numeric/EngineLmHeadRefinement.lean` — `lm_head_refines_proved`.
- `Numeric/EngineSamplerRefinement.lean` — `word_argmax_fixed_spec`, `word_kv_append_ok`.
- `Numeric/PrunedArgmax.lean` — `argmax_fixed_returns_lowest`, `word_argmax_fixed_returns_lowest`, `pruned_argmax_exact`.
- `Numeric/HadamardRefinement.lean` — `fwht_int_spec`, `fwht_int_sylvester`.
- `Numeric/HadamardFixed.lean`, `Numeric/HadamardFixedSpec.lean` — `fwht_fixed_spec`, `fwht_fixed_refines`.
- `Numeric/HadamardStage.lean` — `fwhtRef_getElem`, `fwhtRef_abs_le`.
- `Numeric/Sylvester.lean` — `sylvester_mul_self`, `sylvester_mulVec_butterfly`.
- `Numeric/LutPQ2.lean` — `block_dot_lut`, `row_dot_lut`, `planeSum_bound`, `lutEntry_mirror`.

**Numeric — adjacent, not on the dense decode path**
- `Numeric/MoeRouter.lean` — `route_topk_spec` (MoE; neither model is MoE — `docs/evidence/decode-attribution.md:51`).
- `Numeric/Packing*.lean` (5 modules) — table packing, not the LLM decode step.

**Nous/Model**
- `Nous/Model/Attention.lean` — `attention_channel_form`, `channelWeight_parabolic`, `channelAttnScan_eq`.
- `Nous/Model/AttnStage.lean` — `attnOut_spec`, `attnRef_lipschitz`, `dot_bridge`.
- `Nous/Model/AttnStageWitness.lean` — `attnHeadStage_witness`, `attnNZState_witness`.
- `Nous/Model/RmsVectorStage.lean` — `rmsVectorStage_dom_witness`, `rmsIdeal_eq_rmsIdealVec`.
- `Nous/Model/RotationStage.lean` — `hadamardRef_isometry_l2`, `hadamardRef_involutive`.
- `Nous/Model/RopeOrbit.lean` — `rope_orbit`, `rotMat_relative`.
- `Nous/Model/SiluChannel.lean` — `silu_gate_eq_channelWeight`, `siluCell_normForm`.
- `Nous/Model/BlockComposition.lean`, `Numeric`-adjacent block error bounds — `fullBlock_delta_eq`, `block_error_bound`.
- `Nous/Model/BlockGrid.lean`, `Nous/Model/BlockGrounded.lean`, `Nous/Model/BlockGroundedWitness.lean`, `Nous/Model/BlockGroundedWitness1.lean`, `Nous/Model/BlockMulBox.lean` — `fullBlockGrounded_delta_eq`, `fullBlockGrounded_dom_nonempty`.
- `Nous/Model/ErrorComposition.lean` — `decision_margin_preserved`, `siluGrid_bound`, `rmsGrid_bound`.
- `Nous/Model/Instances.lean` — `word_mul_denote_hyperbolic`, `word_mul_denote_parabolic`, `word_mul_denote_elliptic`, `word_norm_denote_*`.
- `Nous/Model/FaceContraction.lean`, `FaceRecurrence.lean`, `Weighting.lean`, `State.lean`, `Model.lean`, `Memory.lean`, `Decomposition.lean` — ring/algebra layer of the canonical basis (`euclid_contraction`, `hyper_contraction`, `para_contraction`, `tokenFold_append`, `hyperbolic_decomposition`).
- `Nous/Model/TopKRouter.lean` — `topkSel_spec`, `softmax_sum_eq_one`, `softmax_topk_renorm` (MoE router; off-path).
- `Nous/Model/JacobiDecode.lean`, `Nous/Model/SpeculativeAccept.lean` — `jacobi_exact`, `verify_accept_prefix`, `specStep_eq_target` (lookup/speculative decoding; off-path per `docs/evidence/decode-attribution.md:49`).

## 7 · Gaps (ops with no Lean link)

| gap | op | why |
|---|---|---|
| G1 | embedding row pointer arithmetic (`embedding_sched` `src/engine/step.rs:740`) | row offset/extent check unproved; only the Q8_0 dequant is linked (`Numeric/DequantQ8_0.lean`) |
| G2 | `quant_act` exact refinement (`src/engine/chart/mod.rs:205`) | only error/shape bounds (`Numeric/ChartDot.lean`, `Numeric/ColumnCover.lean`); no extracted `quant_act` refinement linking to the exact int8 quantization |
| G3 | chart Q8Act row kernels and their fused/AVX2 twins (`src/engine/chart/mod.rs:1334`, `src/engine/chart/x86.rs`) | equality with `matvec_fixed ∘ dequant ∘ quant_act` is `tested`, not extracted |
| G4 | LUT AVX2 twin (`src/engine/kern/lut.rs:837`) and the LUT table-build schedule | portable rows linked (`Numeric/LutPQ2.lean`); AVX2 equality `tested` |
| G5 | the persistent pool and every `Pool::run` barrier (`src/engine/chart/pool.rs:206`) | scheduling is `tested` only |
| G6 | head split of `qk_norm_and_rope` (`src/engine/step.rs:357`) and `attn_heads` (`src/engine/chart/ffn.rs:689`) | per-row norm linked; the fused head dispatch is not |
| G7 | residual/bias `add_assign_fixed` (`src/engine/step.rs:220`) | trivial checked add, no module |
| G8 | `YaRN` RoPE scaling | Bonsai metadata declares `yarn` (factor 4.0); the engine builds the plain table and ignores it (`src/engine/rope.rs:56`) — not on the path |
| G9 | Bonsai `rope.scaling` / long-context behaviour | same as G8; out of scope |
| G10 | stochastic sampler / temperature | `top_p_fixed` (`src/engine/sampler.rs:81`) exists but `step_with_kv` calls `argmax_fixed` only (`src/engine/step.rs:1130`) |
| G11 | `ffn_tp` (`src/engine/kern/tp.rs:91`) | new tensor-parallel FFN; equality to `ffn_q8act` is `tested` in `tp.rs`, not wired into `compose_logits` |
| G12 | rustc/LLVM codegen, AVX2 intrinsics, the `unsafe` pool container | outside the trust boundary (AGENTS §4 obligation 11) |
| G13 | no `Cell`/`Sigma` operation on the decode path | `src/engine/` carries scalar `i64` Q32.32; `Cell<R>`/`Sigma` live in `src/cell.rs` and the canonical corpus (`Nous/Model/Instances.lean`). Any face reading (elliptic/parabolic/hyperbolic) is grade `model` |

## 8 · Totals per model and the bandwidth floor

Symbols as §1. Bytes/token = layer weights + one embedding row + LM-head payload (tied).

| quantity | Qwen2.5-0.5B | Ternary-Bonsai-1.7B |
|---|---:|---:|
| layer weights/token | L·(Σ row_bytes·rows) = 24·10,872,064 = **260,929,536 B** | 28·13,369,344 = **374,341,632 B** |
| embedding row | 952 B | 544 B |
| LM-head payload (tied) | 151936·952 = **144,643,072 B** | 151669·544 = **82,507,936 B** |
| **bytes/token** | **405,573,560 B** (386.8 MiB) | **456,850,112 B** (435.7 MiB) |
| weight MACs/token (projections + head) | **493,961,216** (0.494 G) | **1,719,904,256** (1.720 G) |
| norm/rope/silu MACs | ≈ 324,864 | ≈ 995,328 |
| attention MACs/token | 2·q_out·t per layer·L = **43,008·t** | **114,688·t** |
| exp calls/token | silu F·L = 116,736; softmax H·L·t = 336·t | silu 172,032; softmax 448·t |
| bandwidth floor @ 82 GB/s | 405,573,560/82e9 = **4.95 ms → 202 tok/s** | 456,850,112/82e9 = **5.57 ms → 179 tok/s** |
| measured (median) | 128.2 tok/s → 7.80 ms | 84.3 tok/s → 11.86 ms |
| measured / floor | 0.63 | 0.47 |

The floor moves the whole weight payload once; the measured gap is compute + dispatch, not a
bandwidth measurement (obligation-10 metric is the named command above, machine i9-14900KF).

## 9 · Gate and control

Gate: the file exists; every `path:line` in it resolves (file exists, `≥ line` lines). The checker
is `/home/socol/oc-tmp/dom-map-t1/check_paths.py`; it resolves relative paths under the repository
root, `proof/corpus/` and the receipt directory `/home/socol/oc-tmp/ds-dispatch/ai/`. Control: one
line number in a scratch copy of this file is moved past the end of its file, which must make the
same checker report exactly 1 failure.

```
$ python3 /home/socol/oc-tmp/dom-map-t1/check_paths.py docs/evidence/pipeline-inventory.md
checked=90 failures=0 exit=0

$ cp docs/evidence/pipeline-inventory.md /home/socol/oc-tmp/dom-map-t1/pipeline-inventory.bad.md
$ # in the scratch copy, mutate one cited line number past the end of its file
$ python3 /home/socol/oc-tmp/dom-map-t1/check_paths.py /home/socol/oc-tmp/dom-map-t1/pipeline-inventory.bad.md
checked=90 failures=1 exit=1
FAIL src/engine/step.rs:<99999> (file has 1431 lines)
```

`proof/ApeironWord.lean` sha256 `a4599646…` unchanged at the final gate.
