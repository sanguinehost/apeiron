# Performance predictions

Row `perf-theory-formal-cost-HiI4vx6ezYPSHkfGWbq7YX` (W3). A prediction is committed before the measurement it concerns is read; the commit order is the evidence. The model under test is the roofline of `proof/corpus/Perf/Roofline.lean` (`timeLB f = max_r d_r(f)/c_r`, chain term inside the max per `Perf/Concurrency.lean`). Grade of every row until its measurement is read: `anchor`.

| id | case | cost equation | prediction | refuted if | measured | verdict |
|---|---|---|---|---|---|---|
| P1 | ATTN-HD128: online softmax with one K/V walk at hd=128, node ratio old/new | binding field is shared-slab bytes; walks 3 to 1; the FMA term is unchanged and bounds the ratio below 3 (`walk_mix`); hd=128 forces a smaller tile, which lowers reuse per staged slab | point 1.8, band [1.5, 1.95] | outside [1.4, 2.05] over at least 5 interleaved rounds | none: the old arm does not exist at hd=128 (exact tile K slab 65536 B > `WG_SHARED_MAX` 49152 B) | untestable as stated |
| P2 | DECODE-SPLITK: qwen05b Q4_K_M tg128, raise split-K / workgroup occupancy of the fused `qkv` and `gu` GEMVs (`src/gpu/decode/layer.rs`, `spec.rs`); `o`, `rope`, `flash`, `dn` untouched | per dispatch `t_i >= max(W_i/865 GB/s, floor_i)` with `floor_i = 1.14 us` (empty dependent dispatch) plus the dependent memory phases inside the kernel; the in-tree small kernel `o` (0.9 MB) already runs at 4.06 us, which sets the observed floor for a one-pass small GEMV at about 4 us; the serialized chain sum (1344 us at conc 1) binds | qkv 9.66 to 4.6 us each [3.5, 6.5]; gu 17.85 to 13.3 us each [12.0, 15.5]; `o` unchanged [3.5, 4.1]; token 1348 to 1118 us; tg128 steady 741.7 to 890 tok/s, band [800, 930] | tg128 steady mean outside [770, 960], or qkv below 3.0 us each, or `o` changed by more than 10% | qkv best 7.56 us (8192; outside); gu 16.3 to 17.8 us (outside); `o` 4.03 to 4.33 us (inside); tg128 shipped default 710.7 [706.6, 712.9], best arm 721.2 [701.5, 734.6] against 709.8 [687.5, 721.7] at HEAD dials (outside) | refuted |
| P3 | DECODE-UNFUSE: qwen05b Q4_K_M tg128, replace the fused RMSNorm+quant+GEMV (`make_fused_gemv`, one 256-thread WG recomputes the norm over x before streaming weights) for `attn_qkv` only by a norm+quant dispatch followed by the bare DP4A GEMV (`make_gemv`); `gu`, `o`, `dn`, `rope`, `flash` untouched | in-WG serial prologue (x load, sumsq and absmax reductions, quantize, barriers) is a dependent chain inside each WG that hides behind weight transport only when other WGs are resident to overlap it; a 1.16 MB `qkv` has about one wave, so the prologue is paid once in series; CUDA's separate `quantize_q8_1` (median 0.8 us) plus a Q5_0/Q8_0 GEMV (median 2.0 us) is the reference for the unfused pair (kernel-to-tensor mapping inferred from ggml type ids) | unfused `qkv` pair 5.5 us total [4.0, 8.0] against 9.57 fused; token 1348 to 1250 us; tg128 ratio after/before 1.078 [1.03, 1.12] (same-session interleaved) | ratio below 1.02 or above 1.15, or unfused pair outside [3.5, 9.0] us | unfused pair cold 8.995 us [8.893, 9.062] (warm 5.663) against 9.571 fused cold; tg128 interleaved 5 rounds 703.40 before, 691.78 after, ratio 0.984 (pairs 0.943, 0.996, 1.018, 1.026, 0.936) | refuted |
| P4 | DECODE-PROLOGUE: qwen05b Q4_K_M tg128, shorten the dependent chain inside the norm+quant prologue of `gemv_norm_module_sc` (`src/simt_nn/norm.rs`, emitters `rmsnorm_wg`, `emit_quantize_to_shared`, `tree_reduce` in `src/simt_nn/mod.rs`), order-preserving only: (a) issue every strided x/gamma load of the sumsq and scale loops before the first dependent add, keep x in registers across the two passes; (b) unroll the 32-element per-block absmax/sum loop so its shared loads issue back to back, same add order; (c, control arm) replace the six warp-internal tree barriers by subgroup-scope barriers; `gu`/`qkv` fused sites only, the f32 sum and max order unchanged | prologue = serial phases inside one WG: sumsq loop (4 dependent global-load iterations), 9 barriers, scale loop (4 iterations, x reloaded), absmax loop (32 serial shared loads per block, 28 lanes active), pack loop; phase cost = iterations x load latency while the loads cannot overlap (loop body has no unroll hint, `OpLoopMerge` flags 0). Warm 3.81 us = 4x0.15 + 0.45 (barriers) + 4x0.15 + 0.75 (absmax) + 0.2 (pack) +- 0.5 (latencies are estimates, the sum reproduces the measured warm figure); cold adds about 2.3 us on the two global-load phases. (a) removes about 3 of 4 serial load latencies in two loops, (b) cuts the 32x~45-cycle chain to about 32x4 cycles of adds | prologue cold 6.08 to 3.6 us [3.0, 4.2] (warm 3.81 to 2.0); per fused site -2.5 us cold or -1.8 us warm; 48 sites (24 layers x qkv, gu) give -85 to -120 us per token against 1344; tg128 741.7 to 795-817 tok/s (+7 to +10%), ratio to CUDA 856 of 0.93 to 0.95; arm (c) alone inert (< 0.2 us per site; consistent with the 2026-07-08 subgroup-reduce measurement in `tree_reduce`); 0 argmax mismatches and unchanged `SEQHASH` because the f32 order is unchanged | tg128 gain below 3% (below 764 tok/s) over at least 5 interleaved rounds, or cold prologue above 4.8 us after (a)+(b), or (c) alone above 0.5 us per site, or `SEQHASH` changes | cold prologue 6.14 to 5.44 us under (a), 7.43 to 6.42 under a+b+c (prologue-only, trunc=4); warm 3.804 to 3.621; tg128 702.9 to 728.7 (ratio 1.037 [0.999, 1.079], 5 interleaved rounds); `SEQHASH` `414708724a03be6e` to `27ee7624bee91cf8` under (a) | refuted: cold prologue above 4.8 us, tg128 gain 3.7% above the 3% floor but below the 7-10% band, `SEQHASH` changed |
| P5 | DECODE-SUMSQ-FLOOR: the sumsq phase of the qkv prologue (grid (5,16), LocalSize 256, shared allocation as `gemv_norm_module_sc`) is 4.695 us warm in P4 against a 1.2-1.5 us empty-dispatch floor; decompose it. Probe kernel: same grid and shared allocation, one x load per thread and one store, no loop, no barrier | if fixed dispatch plus first global load dominates, the probe costs close to the P4 sumsq phase minus its extra three dependent loads and the strided loop overhead; if the loop and codegen dominate, the probe is near the floor plus one load latency (~0.6 us) | warm T0 2.0 us [1.5, 2.8]; so the loop-and-codegen share of the P4 sumsq phase is about 2.7 us, of which arm (a) removed about 1 us | T0 above 3.0 us warm (fixed launch plus first-load cost dominates; the loop is not the lever), or below 1.5 us (the loop owns more than 3 us) | none yet | pending |

## P1 provenance

- Basis: hd=64 measured node 2214 us to 1160 us, ratio 1.91 (the epic's evidence table); same binding field.
- A first version of this prediction used a recalled walk count of 3 to 2 (band 1.0 to 1.5). It contradicted the hd=64 measurement and was corrected before any hd=128 number was read. The miss was a count recalled instead of read from the kernel.
- The measurement is `ATTN-HD128` (board row `inf-resident-prefill-GEB633UfORCK7XSmzJifHd`), paired, at least 5 rounds, forced rebuild, through `cargo-gate.sh`.

## P1 outcome

Board row `inf-resident-prefill-GEB633UfORCK7XSmzJifHd` (ATTN-HD128); receipt `/home/socol/oc-tmp/escalation/ATTN-HD128-RECEIPT.md`.

- Command (live tree, `touch src/lib.rs`, shared target via `cargo-gate.sh`, 5 interleaved rounds, qwen2.5-1.5b q8, `PREFILL_PROFILE=1 cargo run --release --features gpu --example gpu_decode_bench -- <gguf> --prefill 512 --reps 5 --carrier q8`; logs `/home/socol/oc-tmp/escalation/HD128-bench-{untiled,tiled}-{1..5}.log`):

| arm | attn node us, mean [min, max] | pp512 tok/s, mean [min, max] |
|---|---|---|
| `DEC_ATTN_TILE=0` (untiled) | 10491.2 [10270.3, 10815.4] | 16114.5 [15461.7, 16417.9] |
| `attn_tiled_16_4_4_kp4_false_m3` | 1805.9 [1799.8, 1814.8] | 22286.0 [21300.8, 23877.4] |
| ratio | 5.81 | 1.383 |

- The predicted pair (exact 3-walk tile against online one-walk tile) cannot be built at hd=128, so P1 is neither confirmed nor refuted. The measured pair is untiled against online dim-split tile (`kp=4`, `rows=16`); the model made no prediction for it, so 5.81 is a post-hoc figure and carries no predictive grade.
- Rule from P1: a prediction names a baseline that can be constructed on the measured configuration.
- The tile that exists at hd=128 (`rows=16`, `kp=4`) also changes reuse per staged byte (FMAs per staged byte `rows/kp` against `rows/2` for the fp16 slab); the model has no row for that term yet.

## P2 provenance

Board row `inf-decode-measure-E3CPdJamkoVAQSrj96zC6o`. Basis: DECODE-ATTRIB receipt `/home/socol/oc-tmp/escalation/DECODE-ATTRIB-RECEIPT.md` (apeiron `8836859`, 5 interleaved rounds, timestamp queries, no-timestamp control 1344 us/token against the bench's 1348 us).

- Baseline (steady, n=20): apeiron q4 741.7 tok/s [693.6, 766.9]; llama.cpp CUDA `ece963f41` 856.0 [845.6, 864.8]; gap 15.4%. Closing it needs about -176 us per token.
- The lane's own estimate for the same lever took `qkv` to the `lm_head` streaming rate (874 GB/s, 1.4 us each, -253 us, 917 tok/s). P2 rejects that rate for a 1.16 MB kernel: it is below the 4.06 us the 0.9 MB `o` kernel already takes, so the dependent-phase floor binds before bandwidth. Both figures are recorded; the measurement decides between them.
- P2's point 890 is 1.04x CUDA (856.0) and the top of the band, 930, is 1.09x. It makes no prediction for the other candidates (rope into flash, gate/up carrier).

## P2 outcome

Lane DECODE-SPLITK (receipt `/home/socol/oc-tmp/escalation/DECODE-SPLITK-RECEIPT.md`, diff `DECODE-SPLITK.diff` not installed). `DEC_SPLITK` swept over {4096 .. 131072}, then per-projection dials; host shared with the desktop (about 30% device utilisation), spreads as reported.

| class | predicted us | measured us | inside |
|---|---|---|---|
| qkv | 4.6 [3.5, 6.5] | 9.57 at HEAD; 7.56 best (`DEC_SPLITK_QKV=8192`, segs 8) | no |
| gu | 13.3 [12.0, 15.5] | 17.52 at HEAD; 16.3 best | no |
| o | unchanged [3.5, 4.1] | 4.03 to 4.33 | yes |
| tg128 steady | 890 [800, 930] | 710.7 shipped; 721.2 best arm | no |

- Verdict: refuted. Split-K occupancy moves `qkv` by at most 2 us per dispatch and `gu` by about 1 us; both stay above the band. The ~4 us floor for a one-pass 1.16 MB GEMV is not what binds: `qkv` at 7.56 us is 154 GB/s, 18% of the measured streaming rate.
- The lever that is faster is gated by fidelity: `qkv` at segs 8 turns `gpu_decode_step_q4` red (first token for "The author of Hamlet is": gpu 32671, cpu 264). Whether that is f32 reassociation or a constraint of `fused_rope_kv_fold` at segs 8 is unresolved.
- Model consequence: `Perf/Chain.lean` bounds a below-knee dispatch by `L` and says nothing about which field sets `L`; P2 assumed one (dependent phases at about 4 us) and the measurement rejects it for `qkv`. `L` per kernel class is a fitted parameter, not a constant.

## P3 provenance

Board row `inf-decode-measure-E3CPdJamkoVAQSrj96zC6o`. Basis: P2 refuted (`qkv` stays at 7.56 to 9.57 us whatever the split); CUDA census `DECATTR-kern.csv` (llama.cpp CUDA `ece963f41`, 8 generated tokens, medians: `quantize_q8_1` 0.8 us, `mul_mat_vec_q` type 6 and 8 about 2.0 us).

- Direction check against the proved model: `Perf.Chain.fuse_saves_L` says fusing a below-knee pair lowers the dependent-chain bound by `L`. P3 predicts the opposite measured direction (unfusing is faster), so a confirmed P3 means the chain bound is slack for the fused kernel by a term the model lacks: the serial phases inside one workgroup (an intra-WG chain, hidden only by occupancy). A refuted P3 leaves the chain model's direction intact and the `qkv` cost unexplained.
- The CUDA kernel-to-tensor mapping is inferred from type ids (6 = Q5_0, 8 = Q8_0, 12 = Q4_K, 14 = Q6_K); a short run of 8 tokens, so the figures are census medians, not steady state.

## P3 outcome

Lane `DECODE-UNFUSE` (DeepSeek Flash; return `/home/socol/oc-tmp/escalation/DECODE-UNFUSE.out`, diff `/home/socol/oc-tmp/escalation/DECODE-UNFUSE.diff`, sha256 `a87143baaf5270cf`, not installed). Row `inf-decode-measure-E3CPdJamkoVAQSrj96zC6o`. Lane figures, not yet re-run by the lead.

| quantity | prediction | measured (mean [min, max]) | verdict |
|---|---|---|---|
| unfused qkv pair, cold (96 MiB flush between iterations) | 5.5 [4.0, 8.0] us | 8.995 [8.893, 9.062] us | outside the interval, inside the refutation band [3.5, 9.0] by 0.005 |
| unfused qkv pair, warm (weights L2-resident) | 5.5 [4.0, 8.0] us | 5.663 [5.605, 5.706] us | inside; does not reproduce the chain figure |
| fused qkv, cold | 9.57 us | 9.571 [9.512, 9.627] us (chain probe: qkv class 9.249 us) | cold mode reproduces the chain |
| tg128 after/before | 1.078 [1.03, 1.12] | 0.984 (703.40 to 691.78 tok/s, 5 interleaved rounds) | refuted: below 1.02 |

- Mechanism the lane measured: cold fused 9.571 us is the bare GEMV stream 3.461 plus the norm+quant prologue 6.082 (norm+quant alone: 6.082 cold, 3.814 warm). The prologue is not hidden behind weight transport; unfusing removes nothing and adds one dependent dispatch (lane-reported ~3.7 us per layer for a compute-to-compute barrier, from `norm.rs`; not re-measured here).
- Correctness: `SEQHASH` `414708724a03be6e` unchanged, 0 argmax mismatches over 128 steps; `gpu_layer_match`, `gpu_gemv_match`, `gpu_decode_step` (ignored, 2 passed), clippy and fmt exit 0; negative control (drop the residual fold) turns `gpu_decode_step` red with "GPU text lacks Paris".
- Model consequence: the chain bound's direction survives (`Perf.Chain.fuse_saves_L`: fusing the below-knee pair is not slower). The `qkv` cost is attributed to a single-workgroup latency-bound norm of about 6 us cold, not to occupancy. The CUDA reference (quantize 0.8 us) is 7.6x lower for the same 896-element reduction, so the lever is the norm prologue's own latency (dependent global loads and barriers inside one WG), not fusion structure. A prediction for that lever is written before it is dispatched.
- Trust boundary: RTX 5080 shared desktop, per-run range 522.8 to 741.9 tok/s; `f16`/`F32` carriers and the direct-`Norm` variant not exercised; cold attribution assumes L2 eviction, not a counter.

## P4 provenance

Row `inf-decode-measure-E3CPdJamkoVAQSrj96zC6o`. Committed before dispatch.

- Source read before writing: `src/simt_nn/norm.rs:140-300` (`gemv_norm_module_sc`: residual fold, `rmsnorm_quant_wg`, then the DP4A fold; every WG of `m.div_ceil(256) x segments` repeats the whole prologue), `src/simt_nn/mod.rs:774-1010` (`wg_barrier`, `tree_reduce`, `rmsnorm_wg`, `emit_quantize_from_shared`), `loop_step` (`OpLoopMerge` with no unroll control).
- The phase accounting is an estimate that sums to the lane's measured warm 3.81 us; it is a model of where the time goes, not a measurement of each phase. The first lane task is to time the phases (stop the kernel after each phase, cold and warm).
- Same prologue shape appears in `silu_mul_q_module`, `quantize_vec_module` and `rmsnorm_module` (standalone), so a confirmed (a)/(b) transfers to the `dn` and `o` quantize sites as a second prediction, not claimed here.
- Order preservation is the constraint: `tree_reduce` records that a reordered f32 sum flipped a q4 argmax; levers that keep every add and max in its present order leave `SEQHASH` unchanged.
- The chain-reference cold figure (9.249 us) and the lane's cold probe (9.571) both use L2 eviction; in a real chain x is L2-resident, so the warm column is the likelier in-chain figure and the low end of the range (-85 us) the likelier token effect.


## P4 outcome

Lane `DECODE-PROLOGUE` (DeepSeek Flash; return `/home/socol/oc-tmp/escalation/DECODE-PROLOGUE.out`, diff `/home/socol/oc-tmp/escalation/DECODE-PROLOGUE.diff`, sha256 `81a0110a`, not installed). Row `inf-decode-measure-E3CPdJamkoVAQSrj96zC6o`. Lane figures; the lead has not re-run them.

Phase table (qkv, GPU timestamps, mean us, cumulative truncation; the first row includes a 1.2-1.5 us dispatch floor):

| phase | warm | cold | delta cold |
|---|---|---|---|
| sumsq | 4.695 | 5.710 | |
| +tree | 4.881 | 5.965 | 0.255 |
| +scale | 5.559 | 7.220 | 1.255 |
| +absmax | 5.805 | 7.277 | 0.057 |
| +pack | 5.919 | 7.430 | 0.153 |
| +fold | 6.270 | 9.653 | 2.223 |

Arms (qkv cold, us): base 9.653; (a) 8.539; (b) 9.055; (c) 9.575; a+b 7.897; a+b+c 7.883. The P4 phase estimate was wrong in the direction of the lever: sumsq 4x under (estimated 0.6-1.2), absmax 13x over (estimated 0.75; measured 0.057), tree 2x over, fold not estimated.

- Verdict: refuted by three of its own criteria (cold prologue above 4.8 us; `SEQHASH` changed; and the gain is below the predicted band although above the 3% floor).
- `SEQHASH` moved under (a) (`27ee7624bee91cf8`) and under (c) (`de7e288927a42731`); (b) alone left it unchanged. The mechanism for (a) (FMA contraction or scheduling of the unrolled form; the f32 add order is unchanged) is unproven: no disassembly was read.
- Arm (c) as specified is unsound: the six sub-32 levels lose the final workgroup barrier, so lanes 32-255 can read `red[0]` before lane 0 writes it. It is inert in time (-0.078 us), so there is no reason to repair it.
- The default build is byte-identical to HEAD (`emit` m=1152 k=896 sha256 `b6a3209fd7046c80`).
- Gates on the lane tree with dials unset: `gpu_gemv_match`, `gpu_layer_match`, `gpu_decode_step`, `spirv_storage_race`, clippy, fmt exit 0. Negative control `let cn = self.cu(n - wg);` under `DEC_PROLOGUE=1` turns `gpu_layer_match` red (exit 101). The brief's two literal mutations (drop the `i<n` guard, reverse the sum order) stayed green: `gpu_layer_match` tolerance 2.7e-2 absorbs ULP-level changes, so it does not guard f32 order. `SEQHASH` is the order guard.
- Consequence: about 2.4 us of the warm sumsq phase is unexplained by load serialisation (arm (a) removes about 1 us of about 3.5 us above the floor). P5 decomposes it before any further lever is built.
- Trust boundary: RTX 5080 shared desktop, per-run range 687-758 tok/s; only Q4 `qkv`/`gu` exercised; F16/F32 direct carriers not exercised; argmax over 128 steps; no counter or disassembly.

## Held-out cases still to predict

- the `b`-parameterised prefill GEMM (UNIFIED-CARRIER W3)
- a map layout change (compact tag array)
