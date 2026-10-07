# CHAIN-FUSE prediction (recorded before the first shader edit)

Worktree `/home/socol/oc-tmp/wt-attn2`, base 31dac0d. Baseline: 6 profile rounds, `PREFILL_PROFILE=1 gpu_decode_bench --prefill 512 --carrier q4`
(logs `/home/socol/oc-tmp/bench/chainfuse/base[1-6].log`, aggregate `agg.py`). GPU-active 7222.7 us [6735.9, 8184.1]; pp512 66998 tok/s [60104, 73650].

## `rms_1` 392 us vs `rms0_1` 7 us (resolved)
`PrefillStack::profile` (`src/gpu/decode/prefill.rs:679`) sums node time by `PNode.name`. The name is `rms_{dq}` for norm1 of layers 1..23,
norm2 of all 24 layers and the final norm (Q8 carrier, dpw 1): 23+24+1 = 48 dispatches. `rms_1` 365.7 us mean / 48 = 7.6 us per call;
`rms0_1` is one dispatch, 6.8 us. Same per-call cost; the 392 vs 7 is a count of 48 vs 1, not an anomaly.

## Lever 1 (A): swiglu + quant_{dff} -> one node `swiglu_quant`
New module in `src/simt_nn/batch.rs`: one workgroup per position reads `gu` (gate, up), computes `a = (g / (1 + exp(-g))) * u` (same op
sequence as `swiglu_batch_module`), stores `a` to workgroup shared (with a +1 float pad per 32 so the per-block quantize walk, stride 32 floats, is
not a 32-way bank conflict), barrier, then the existing `quantize_at` body reads shared. The f32 `a` is bit-identical because it passes through
shared memory (no fmul->fadd contraction into the quantizer's `sum += v`). `acts_buf` and `swiglu_batch_module` are deleted.
- Removed per prefill (n=512, dff=4864, 24 layers): global write + read of `acts`: 2 * 512*4864*4 B * 24 = 478.2 MB (L2 traffic); 24 dispatches and 24 barriers.
- Baseline nodes: swiglu 255.2 us (10.6 us/call, 29.9 MB/call = 2.8 TB/s, L2 resident), quant_4864 (_1 + _2) 398.8 us (16.6 us/call over 24 calls, 13.1 MB/call = 0.79 TB/s, latency-bound by the 32-iteration serial walk from global).
- Model: fused call = swiglu body without the act store (about 8 us) + quantize from shared (about 4 us) = 12 us [9, 18]; baseline 27.2 us/call.
- Predicted saving: 15 us/call * 24 = 360 us -> interval [150, 450] us of GPU-active (nodes 654 -> ~290 us). pp512 +3.5% [+1.5, +5.5].
- Refuted if: fused node > 21 us/call (saving < 6 us/call, < 150 us total), or any bit changes in prefill hidden/logits hash, or `gpu_prefill_match` goes red.

## Lever 2 (C): padded shared layout for the norm's quantize walk (`rmsnorm_batch_q_module`)
`nrm` is read by `quantize_at` phase 1 at index `b*32 + j` for 28 active lanes (n=896): all lanes hit one bank, 28-way conflict, 32 iterations.
Same +1-per-32 pad. Bit-identical (layout only). 48 dispatches.
- Model: conflict cost = 28 lanes * 32 iterations = ~900 shared cycles per workgroup, 6.1 workgroups per SM = ~5500 cycles = ~2 us per call at the
  ~2.5 GHz SM clock -> saving 2 us * 48 = 96 us, interval [30, 170] us. pp512 +1.3% [+0.4, +2.3].
- Refuted if: `rms_1` mean per call drops by < 0.6 us (< 29 us total), or bits change.

## Not done (declared files)
Lever B (kv_seed into rope; kv_seed 94.8 us = 4.0 us/call, launch-bound; model saving 70 us [30, 95]) needs `prefill_glue.rs` as a 4th file (the module is deleted)
and a second `spirv_storage_race.rs` list edit; it is the next task, not this one.

Declared files: `src/simt_nn/batch.rs`, `src/gpu/decode/prefill.rs`, `tests/spirv_storage_race.rs`.
Bandwidth reference: RTX 5080, 256-bit GDDR7 at 30 Gbps (nvidia-smi max memory clock 15001 MHz, x2) = 960 GB/s (spec, not a copy-kernel measurement);
L2 64 MB (spec). Nodes above 0.96 TB/s are served from L2, not DRAM.
