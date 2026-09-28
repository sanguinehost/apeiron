# Decode dominance: beating llama.cpp by exactness, not by tuning

- **Owner:** lead (Claude track)
- **Source revision:** `74c037b`; working tree dirty
- **Status:** draft. L0 is measured (idle i9-14900KF, `qwen05b-Q4_K_M.gguf`, 64-token greedy decode, generated ids byte-identical throughout): 8.4 tok/s `Exact` → 35 (T6B raw-byte AVX2 cores) → 78–79 (persistent pinned pool, `src/engine/chart/pool.rs`) vs llama.cpp 113. T6C is removing the serial sections (attention, `dequant`, i128 division on the calling thread: ≈40% of wall time)
- **Epic ID:** `APEIRON-DOM`
- **Parent:** R0 of [`frontier-consumer`](frontier-consumer.md); follows [`engine-throughput`](engine-throughput.md) W2

## Intent

### The ceiling

Single-stream decode reads every active weight once per token. The time per token is bounded
below by `bytes_per_pass / BW`. This is a roofline model (grade `model`), not a theorem.

On the i9-14900KF with `qwen05b-Q4_K_M.gguf` (373.7 MiB), the two ceilings are measured, not
taken from a document:

| quantity | value | instrument |
|---|---|---|
| host sequential read | 82.0 GB/s at 8 threads, 94.3 at 32 (30.0 single-thread) | 4 GiB `u64` fold, best of 3, `rustc -O -C target-cpu=native` (scratch `bw.rs`) |
| bandwidth ceiling | 82.0 / 0.392 GB ≈ **209 tok/s** at 8 threads (≈240 at 32) | derived |
| Q8Act work per token | 988M instructions, 266M cycles (IPC 3.7), single thread, i.e. **2.0 instructions per weight** | `perf stat -e instructions:u,cycles:u`, 24-token minus 8-token run, `APEIRON_THREADS=1` |
| compute ceiling | 266M cycles / 8 P-cores ≈ 5.8 ms ≈ **171 tok/s** | derived, perfect parallelism assumed |
| llama.cpp `build-cpu` (`ece963f41`) tg64 | 113 ± 4.5 tok/s at 8 threads ≈ 44 GB/s, 54% of the 8-thread read | `llama-bench -p 0 -n 64 -r 3` |
| apeiron Q8Act, pooled | 76–81 tok/s ≈ 46% of the 171 compute ceiling | `engine_decode`, 64 tokens |

`fold`'s 66.0 GB/s host figure is superseded here, and 200 tok/s single-stream is not ruled out.
The model ranks the levers in order:
1. **Parallel efficiency** (46% → ~90%): the serial sections on the calling thread (T6C). This lever
   is worth up to ≈171 tok/s.
2. **Instructions per weight** (2.0 → ≤ 1): the compute ceiling passes the bandwidth ceiling, and
   decode becomes bandwidth-bound at ≈209. The exact lookup-table decomposition (L6) and tighter
   SIMD are the candidates.
3. **Beyond ≈209–240:** k-token passes (L1/L2) or fewer bytes (L4). Compute grows with k while
   bandwidth does not, so L1 pays only after lever 2. At 2.0 instructions per weight the kernel
   is already compute-bound at k = 1: **L6 is a prerequisite for L1/L2 on this CPU**, not an
   alternative to them.

Every such change today is a correctness risk. llama.cpp validates such changes by perplexity,
because its float reductions depend on batch shape and order. Batched and single-token logits
differ, and speculative output is not guaranteed equal to plain decoding.

### What the Aeneas link gives that fold and llama.cpp lack

1. **An exact oracle.** `matvec_q8act = matvec_fixed ∘ dequant ∘ quant_act` holds bit for bit.
   Every reordering, tiling, fusion, repacking or threading of an integer reduction is then an
   equality to prove or test, not a quality to re-measure. Integer addition is associative:
   any partition of the sum is exact.
2. **Batch-invariance as a theorem.** Take a k-column product that reads each weight once. Its
   column `c` equals the single-vector projection of `x_c`, exactly. Two uses follow:
   - speculative greedy decoding verified by that kernel emits **the identical token sequence** as
     plain greedy decoding;
   - a batch of agents served together gets **identical** per-agent outputs.

   llama.cpp cannot state either.
3. **Traffic as a statement about the extracted loop.** The bytes a kernel reads per call can be
   stated over the extracted body: each weight row slice is read exactly once, whatever k is.
   `bytes_proved / BW_measured` then predicts a lever's ceiling before it is built. The roofline
   stays a `model`; the traffic count becomes `proved`.

### Levers, ranked by expected gain on the same model and the same output

| id | lever | why it can beat llama.cpp | exactness argument | falsified by |
|---|---|---|---|---|
| L0 | parity: raw-byte AVX2 cores, one overflow bound per call, persistent worker pool instead of a `thread::scope` spawn per projection (≈169 projections/token) | up to the measured ceilings: ≈171 tok/s compute at 2.0 instructions per weight, ≈209 bandwidth at 8 threads | bit-identity tests (existing) | still < 60 tok/s after T6B plus the pool |
| L1 | **k-column integer product** `matmat_q8act(W, [x_1..x_k])`: weights streamed once, k activation vectors in L1 | time(k) ≈ max(bytes/BW, k·compute): throughput × k until compute-bound | column c = `matvec_q8act(W, x_c)` exactly; traffic independent of k | time(8) > 2·time(1) on the 0.5B model |
| L2 | **exact speculative greedy** on L1: draft k tokens, verify in one pass, keep the agreeing prefix plus one | tokens per pass = (1−α^{k+1})/(1−α) at argmax-agreement rate α (Leviathan et al., arXiv `2211.17192`, pointer; our `accept_rate`/`specStep_eq_target` cover the sampling case) | output sequence = plain greedy, by batch-invariance plus argmax verification | α < 0.5 for every draft source after draft cost |
| L2a | draft sources, cheapest first: prompt/context n-gram lookup (0 weight bytes; the product's "learns by lookup"); a low-bit copy of the same weights (TQ/Q2 with the proved integer FWHT); layer-skip self-draft | the draft costs little or no bandwidth | the draft's quality affects speed only, never output | measured α per source |
| L3 | **exact zero-code skip**: under `Q8Act`, a zero code contributes exactly nothing; store the down projection column-blocked and skip all-zero 32-blocks of its input | fewer bytes per pass with identical output | zero code ⇒ zero term (from `Exact ∘ quant`); column layout is a proved bijection of the row layout | all-zero-block fraction < 10% on real decode |
| L4 | **fewer bytes by rotation**: fold orthogonal Hadamard rotations into adjacent weights with the proved integer FWHT, then re-quantize at lower bits (QuaRot, arXiv `2404.00456`; SpinQuant, arXiv `2405.16406`; pointers, not yet retrieved) | fold measured bytes as the gap to llama.cpp (`vs-llamacpp-cpu-27b.md`: IQ2_M moves 35% fewer bytes) | the rotated network equals the original before quantization (orthogonality); after quantization the gate is lm-eval quality, not bit identity | lm-eval loss beyond the stated tolerance at the target bpw |
| L5 | **multi-agent batch** (the NPC product) on L1 | aggregate tok/s × B for B agents sharing one weight pass | per-agent output identical to serving it alone | as L1 |

L1 is the keystone: L2, L5 and batched prefill are the same kernel.

### Decompositions: structure the proofs make usable

Each row below is an exact factorization. The work is split into pieces that are O(1) per unit,
or shared across units, without losing information. Aeneas then links the fast face to the
proved one.

| structure | decomposition | where it pays | exactness |
|---|---|---|---|
| **reduction over a cover** | an integer sum over any partition equals the sum of its partial sums. Threads, tiles, k columns and split-K are all covers of one index set (`fold` `Web.preduceAt_schedule_free`; `Numeric/MatvecChunk.lean` `matvec_fixed_chunks`) | any schedule, any batch shape, any fusion | associativity of ℤ addition |
| **fiber over block scales** | `w = d·q + lo`: the integer core on the small-code fiber, lifted once per block (`chart_factor`, proved) | the i32 `maddubs`/DP4A core with one i128 lift per block (W2) | `Numeric/ChartDot.lean` |
| **finite alphabet → table (L6)** | a k-bit code is `Σ_b 2^b·bit_b` (bit planes). For a group of g activations, precompute all 2^g signed partial sums once per token, shared by every output row; each weight group then costs one lookup per plane instead of g multiply-adds (T-MAC, Wei et al., arXiv `2407.00088`; bitnet.cpp lookup-table kernels; pointers, not yet retrieved) | ternary/1–2-bit weights (TQ1_0/TQ2_0/Q1_0/Q2_0, already proved decoders) where multiply-adds, not bytes, bound the row | a finite sum evaluated by cases; the table entry equals the sum it caches |
| **fixed point in time (L2b)** | greedy decoding is the fixed point of the map `x_{t+1} = argmax f(x_≤t)`. Jacobi iteration updates a whole window at once, and each iteration makes at least one more prefix token final, so it reaches the exact greedy window in ≤ k passes (Jacobi / lookahead decoding, Santilli et al. 2023, Fu et al. 2024; pointers) | with L1, each iteration is one weight pass for k positions: speculation without a draft model | induction on position; the fixed point equals sequential greedy |
| **lookup as the draft (L2a)** | a context-suffix n-gram → continuation map stored in the proved `Apeiron.Table` (O(1) get/insert); drafts cost no weight bytes | spec decode on repetitive text, code and tool traces; the product's "learns by lookup" | verification makes draft quality affect speed only |
| **Fourier on the hypercube (L4)** | the Walsh–Hadamard transform is the Fourier transform on (ℤ/2)^n and diagonalizes the hypercube Laplacian. Applied to weights and activations, it spreads outlier mass across coordinates, so a coarser grid fits (`HadamardFixed`, `fwht_fixed_refines`, proved) | fewer bits per weight at equal quality: the bytes lever | orthogonality before quantization; lm-eval after |
| **associative scan (linear-recurrent layers)** | a linear recurrence is an affine-monoid product, so a parallel scan in O(log n) depth equals the sequential one (`Nous/Model/FaceRecurrence.lean`, proved) | prefill and multi-token verification through DeltaNet/SSM layers (`fold`'s Qwen3.8 hybrid) | monoid associativity |


### Invariants

1. `Exact` is untouched and remains the proved oracle.
2. Every lever except L4 preserves output bit for bit against its stated reference: `Exact`, or
   `Q8Act` single-stream. L4 changes the model and is graded by lm-eval.
3. Performance claims are `measured` only on an idle host, with the named machine, command and
   interval, and with llama.cpp run in the same session on the same GGUF.

### Non-goals

GPU (`engine-throughput` T10) until CPU parity holds; training; any lever whose exactness argument
is "perplexity did not move".

## Spec

| item | contract | grade target |
|---|---|---|
| `matmat_q8act(ggml_type, raw, n_out, n_in, qs: &[QuantAct]) -> Result<Vec<Vec<i64>>, MatvecError>` | column c = `matvec_q8act(…, &qs[c])` bit for bit, including `Err` | tested, then proved via the extracted portable body |
| traffic | the portable body reads each of the `n_out·row_bytes` weight bytes once per call, for every k | proved (resource statement over the extracted loop) |
| `spec_greedy(engine, draft, k)` | emitted ids = plain `Q8Act` greedy ids for every prompt and every draft | tested (differential), then proved from batch-invariance + argmax |
| zero skip | `down(x)` with skipped all-zero blocks = `down(x)` | tested, then proved (zero term) |
| roofline | `tok/s ≤ BW / bytes_per_pass × tokens_per_pass` | model; calibrated by a measured stream read |

## Plan

| wave | task | files | gate | control |
|---|---|---|---|---|
| W0 | T1 parity kernels (running as `engine-throughput` W2-T6B) | `src/engine/chart/{mod,x86,ffn}.rs` | bit-identity tests; ≥ 60 tok/s | f16 decode bit flip → red |
| W0 | T2 persistent worker pool for chart projections. Measured spawn cost of an empty `std::thread::scope`, 2000 reps, `rustc -O`: 86.5 µs at 8 threads, 160.2 at 16, 291.6 at 32. × 169 projections that is 14.6 / 27.1 / 49.3 ms per token, against llama.cpp's 8.8 ms for the whole token: spawn cost alone caps the current path at ≈20 tok/s | `src/engine/chart/pool.rs` (new), `chart/ffn.rs` | pool result == scoped result; tok/s before/after | a pool that drops a chunk → red |
| W0 | T3 measurement instruments: a stream-read bandwidth probe; per-projection all-zero-block and zero-code fractions; teacher-forced argmax agreement α for each draft source on a fixed text | `examples/decode_probe.rs` (new) | named command, idle host | a shuffled draft must show α ≈ 1/vocab |
| W1 | T1 `matmat_q8act` + AVX2 k-column core | `src/engine/chart/matmat.rs` (new) | column == `matvec_q8act`; time(k), k = 1..16 | swap two columns' scales → red |
| W1 | T2 Lean: batch-invariance and the traffic statement over the extracted portable `matmat_q8act` | `proof/corpus/Numeric/MatmatQ8.lean` (new) + extraction root (lead) | module gate | drop a column → red |
| W2 | T1 `spec_greedy` with the n-gram draft | `src/engine/chart/spec.rs` (new), `step.rs` batch entry | ids == plain greedy on the lm-eval slice prompts; tok/s vs llama.cpp `llama-speculative`/lookup on the same GGUF | accept one extra token → ids differ → red |
| W2 | T2 zero skip (only if W0-T3 ≥ 10%) | `src/engine/chart/skip.rs` (new) | `down` equality; bytes and tok/s | skip a nonzero block → red |
| W3 | L4 rotation and re-quantization | its own epic | lm-eval | — |

## L8 · Exact tensor-parallel layer: row-split fork-join → column-cover reduction

### Intent

The measured gap is parallel inefficiency, not bandwidth or kernel speed. One thread decodes Qwen at 26.5 tok/s
(37.7 ms/token of compute); 8 ideal P-cores would take 4.7 ms, about the 4.8 ms bandwidth floor. The measured
8-thread time is ≈8.0 ms, which is 59% efficiency. The perf profiles put 35–48% of worker cycles in the pool spin
loop: barrier tails and serial caller sections between the ≈5 dispatches per layer.

The topology in use is row-split fork-join. Every projection partitions output rows, and a barrier separates it
from the next projection because the next one needs the whole vector. The alternative is the column/row pairing of
Megatron-LM (Shoeybi et al., 2019, arXiv 1909.08053; not yet retrieved), applied across cores instead of GPUs:
- worker i owns a slice S_i of the FFN hidden dimension;
- it computes the gate/up rows of S_i, applies SwiGLU elementwise, quantizes its slice, then multiplies by the
  down projection's columns S_i;
- the result is a partial vector y_i = W_down[:, S_i]·ĥ_{S_i}, and the only synchronisation is the reduction
  Σ_i y_i.

Attention splits the same way by head: Q/K/V rows of head h, then the attention core of h, then the O-projection
columns of h.

In floating point this changes results with the thread count, which is why float engines keep row-split
barriers. Here it is bit-identical by construction:
- the row accumulator is an exact `i128` with one `round_shift_32` (`src/engine/chart/mod.rs:1378`), so partial sums
  over any column cover add exactly;
- the activation quantizer is per 32-block (`Numeric/ChartDot.lean` `quantBlock`), so a slice aligned to the block
  grid quantizes locally;
- SwiGLU is elementwise.

Result: one barrier per sublayer, at the reduction, which fuses with the residual add and the next RMSNorm sum of
squares. That replaces about 5 dispatches per layer.

Model prediction:
- 59% → ≈80% parallel efficiency gives ≈6 ms/token, ≈165 tok/s on Qwen;
- adding the 16 idle E-cores (`/sys/devices/cpu_atom/cpus` = 16–31) with a speed-weighted static split lifts the
  compute side to about 16 P-core equivalents, and host read reaches 94.3 GB/s at 32 threads, so the floor becomes
  ≈4.2 ms (≈240 tok/s).

Falsifier: if the fused FFN at 8 P-cores is not faster than `ffn_q8act_prepared` on the real Qwen shapes, the
barrier model is wrong.

### Spec

| item | contract | grade target |
|---|---|---|
| column cover | for a partition of the input blocks into slices, `round(Σ_i acc_i) = round(acc)` per output row, with each `acc_i` an exact `i128` partial | proved (spec level), tested (Rust) |
| quantizer locality | `quant_act(x_{S_1} ‖ … ‖ x_{S_k}) = quant_act(x_{S_1}) ‖ … ‖ quant_act(x_{S_k})` for 32-aligned slices | proved from `quantBlock` |
| `ffn_tp` | equals `ffn_q8act` bit for bit, including `Err` | tested |
| slice alignment | slice boundaries are multiples of the down projection's block width: 32 for `Q5_0`/`Q8_0`, 256 for `Q4_K`/`Q6_K`, 128 for `PQ2_0` | tested (refusal otherwise) |

### Plan

| task | files | gate | control |
|---|---|---|---|
| L8-T1 Lean: column cover, quantizer locality, FFN composition | `proof/corpus/Numeric/ColumnCover.lean` (new) | `lake build Numeric.ColumnCover` | drop one slice from the cover → red |
| L8-T2 `ffn_tp` for chart types, one dispatch + reduction; FFN timing vs `ffn_q8act_prepared` on Qwen shapes | `src/engine/kern/tp.rs` (new), `src/engine/kern/mod.rs`, `src/engine/chart/mod.rs` (a partial-row `i128` accumulator over a block range) | `cargo test` equality on every family + ignored timing test | offset one slice's column base → red |
| L8-T3 step integration, attention by head, reduction fused with residual + RMSNorm | `step.rs`, `step/tests.rs`, `chart/ffn.rs` | ids equal on both models; tok/s | — |
| L8-T4 PQ2_0 partial rows (Bonsai), E-core speed-weighted split | `kern/lut.rs`, `chart/pool.rs` | as above | — |

## L9 · RoPE on the turn ring: remove the 4096-position ceiling

### Intent

`fixed_sin_cos` (`src/numeric/fixed_trig.rs:4`) is total only on `[-4096, 4096]` radians, and `RopeTable`
forms the angle as `pos · freq` in Q32.32 radians. The base frequency is `theta^0 = 1` rad per position, so
the exact path refuses every position above 4096 while both served models declare `context_length`
32768. Radians are not a ring in binary (2π is irrational), so the path needs a window and a range
reduction. Turns are one: a frequency stored as a `u64` fraction of a turn makes the phase
`(f_j · pos) mod 2^64`, an element of `Z/2^64`, with exact wrapping arithmetic and no window.

- Invariant: the pair rotation, its rounding and `fixed_sin_cos` are unchanged. Only the angle source
  changes, and the new angle stays inside the proved window for every `u32` position.
- Success: `engine_decode` passes position 4096 on both models; `RopeTurns.lean` proves the phase is a
  group homomorphism `Z → Z/2^64` and bounds the angle error against `pos · theta_j`. The bound is at
  most `2π·pos·2^-55 + 2^-32` rad, below the radian path's frequency-rounding error `pos · 2^-33` for every `pos ≥ 3`.
- Not a goal: YaRN scaling (Bonsai metadata, not on the path); byte-identity with the old radian path
  (the angle bits change; the reference ids are re-baselined and the change recorded).
- Rejected: widening the radian window. It keeps range reduction by a rounded 2π, whose error grows with
  `|k|`, and it still refuses at some finite position.

### Spec

- Load time (declared float boundary, as today): `turn_j = round(theta^(-2j/hd) / (2π) · 2^64)` as `u64`,
  refused when not finite.
- Per position: `phase = ((turn_j as u128 · pos as u128) mod 2^64) as u64`; the signed lift
  `s = phase as i64 ∈ [-2^63, 2^63)`; `r = (s · TWO_PI_Q32 + 2^63) >> 64` (i128, round to nearest).
  Then `|r| ≤ π·2^32 + 1`, so `fixed_sin_cos(r)` never refuses. The rotation of the pair is as before.
- Rust function `turn_to_radians(phase: u64) -> i64` in `src/numeric/fixed_trig.rs` (already extracted by
  `#[path]`), total.
- The Lean spec `Numeric/RopeTurns.lean` covers: `phase_add` (homomorphism), the lift's range and
  congruence, `turn_to_radians_window`, `turn_angle_error` (real statement with `Real.pi_gt_d20` and
  `pi_lt_d20`), and a `#guard` witness.

### Plan

| task | files (owner) | gate | control | grade |
|---|---|---|---|---|
| L9-T1 | `proof/corpus/Numeric/RopeTurns.lean` (new) | `lake build Numeric.RopeTurns` | break the lift (`< 2^63` → `≤ 2^63`), or mutate a witness | proved (spec) |
| L9-T2 | `src/numeric/fixed_trig.rs`, `src/engine/rope.rs`, `src/engine/tests.rs` | `cargo test`; extraction in a copy exits 0 | drop the `mod 2^64` mask, or the rounding bias | tested |
| L9-T3 | `proof/ApeironWord.lean` (regen), the RoPE refinement modules | `proof/regen.sh --check`, `lake build` of the RoPE modules | per module | proved (Rust link) |

The wave barrier is the whole set, plus re-baselined ids for both models and a decode past position 4096.

**Status: closed on its Lean and Rust gates.**
- `proof/regen.sh --check` exits 0 with `AENEAS_HOME=target/aeneas` (70 s).
- `lake build AxiomAudit` exits 0 (9070 jobs) with `turn_to_radians_refines`, `turn_phase_refines` and `turn_angle_trig_window` registered.
- `RopeRotate`, `RopeHeadContent` and `RopeHeadWitness` build over `turns : Vec U64`. The lead's control (witness table set to a quarter turn) turns `RopeHeadWitness` red (exit 1).
- The Rust gates are green; the Qwen reference ids are re-baselined at index 34 (`/tmp/opencode/l9/qwen-turns.txt`); a 4,200-token Qwen decode passes position 4096.

## Status

Idle host (load ≈ 1), i9-14900KF, 8 threads, `APEIRON_KERNEL=q8act target/release/examples/engine_decode <gguf> /home/socol/models/qwen2.5-0.5b/tokenizer.json 64`, 3 reps; baseline `llama-bench -p 0 -n 64 -r 3 -t 8` run back to back. Grade `measured`.

| model | ours tok/s | baseline tg64 | ratio | ids |
|---|---|---|---|---|
| `qwen05b-Q4_K_M.gguf` | 101.0 / 113.1 / 100.7 | llama.cpp `0324696b8`: 116.04 ± 1.49 | 0.87 (median) | equal to `/tmp/opencode/t9/q8act-1.txt`, 3/3 |
| `Ternary-Bonsai-1.7B-PQ2_0.gguf` | 62.6 / 57.0 / 58.6 | PrismML fork `adfffbe`: 91.11 ± 2.86 | 0.64 (median) | equal across reps; equal to the fork's greedy ids for 32 tokens (DOM-W5 report, `tested`) |

| task | result | gate / control (lead re-run) |
|---|---|---|
| `engine-throughput` W2-T6D row cores | single-thread 1.925 → 1.377 instructions/weight, cycles/token 252.5M → 199.4M (worker `perf stat`) | Rust gates exit 0; Q5_0 high-bit mutant in `/home/socol/oc-tmp/tput-t6d/tree` → `cargo test --lib chart::tests` exit 101, `fast_row_bodies_match_their_portable_twins` red |
| DOM-W5 PQ2_0 decode (qwen3 QK-norm, per-tensor `Pq2Lut` cache, pooled LUT rows) | Bonsai decodes; ids equal the fork's | Rust gates exit 0; chunk-misalignment mutant in `/home/socol/oc-tmp/dom-w5/mutant` → exit 101, `pooled_row_chunked_lut_equals_the_unchunked_kernel` red |

After `engine-throughput` W2-T6E (4-block transpose-reduce; single-thread 1.3931 → 1.2398 instructions/weight, worker `perf stat`) and W2-T9 (one pool dispatch for Q/K/V and one for gate/up on both kernel paths, pooled SwiGLU), same command, load ≈ 2–3:

| model | ours tok/s | baseline tg64 (`-r 5`) | ratio | ids |
|---|---|---|---|---|
| `qwen05b-Q4_K_M.gguf` | 136.1 / 135.7 / 126.2 | llama.cpp: 118.42 ± 6.18 | 1.15 (median) | equal, 3/3 |
| `Ternary-Bonsai-1.7B-PQ2_0.gguf` | 73.6 / 70.8 / 73.0 | PrismML fork: 86.77 ± 3.92 | 0.84 (median) | equal to the pre-change ids, 3/3 |

Lead re-runs: Rust gates exit 0; T6E mutant (`reduce4` drops the high half) → `fast_row_bodies_match_their_portable_twins` exit 101; T9 mutant (second projection row base + 1) → `n_projection_dispatch_matches_separate_calls_and_refuses` exit 101.

Open: YaRN `RoPE` scaling; `src/engine/chart/mod.rs` is 2217 lines against the 1500-line cap (TPUT-T6E moves its tests out); per-projection dispatch count (TPUT-T9 fuses Q/K/V and gate/up into one pool dispatch each).
