# Apeiron Q8 SPIR-V primitive latency at real projection shapes

Device **primitive compute time only**. Not whole-model throughput, not host transport, no
llama.cpp comparison. Command and output are the record.

- Command: `timeout 540 python3 tools/gpu/q8_shapes.py --reps 5` (cwd repository root)
- Result: exit 0, elapsed 2.46 s, `PASS`.
- Source identity: `tools/gpu/q8_driver.c` sha256 `6668974f…54b8b6f`, `src/spirv/q8_tile.comp`
  sha256 `0f971a8a…f24c60e`, compiled SPIR-V sha256 `5edba7dc…a8ba57`; git `e481943`.
- Driver: `tools/gpu/q8_shapes.py` drives the W5-T1 timestamp query path of `q8_driver.c` through
  the unchanged compile/input helpers of `q8_bench.py`; it does not modify `q8_bench.py`,
  `q8_bench_controls.py` or `q8_driver.c`.

## Instrument

- Device: **NVIDIA GeForce RTX 5080**, driverVersion `2580660800`, api `1.4.351`,
  device sha256 `880ac99b…aea1d`.
- `timestampPeriod = 1.00 ns`; `timestampValidBits = 64`; `timestampComputeAndGraphics = true`.
- Metric: `device_ns` = delta of `vkCmdWriteTimestamp` at `TOP_OF_PIPE` / `BOTTOM_OF_PIPE`
  around the single `vkCmdDispatch`, one query reset per repetition; `reps = 5`, median reported.
  It includes dispatch of the kernel only. `submit_ns` / `wait_ns` are host `CLOCK_MONOTONIC`
  around `vkQueueSubmit` / `vkQueueWaitIdle` and are reported separately, not folded into rates.

## Formula (per dispatch)

```text
weight_bytes = M * blocks * 32           # Q8_0 packed codes, 32 int8 codes per block
MACs         = M * tokens * blocks * 32  # one MAC per (row, token, block, lane)
GB/s         = weight_bytes / device_ns_med   # bytes/ns == GB/s
GMAC/s       = MACs         / device_ns_med   # MACs/ns  == GMAC/s   (x2 for flop/s)
```

`blocks = K/32`. `GB/s` is the packed-weight bytes for one dispatch divided by that dispatch's
device time; it is a per-dispatch weight-latency ratio, **not** a sustained memory bandwidth
(the same weights are re-read for every token, so it falls as `tokens` rises at fixed work).

## Results (`reps = 5`, median unless noted)

| shape | M | tokens | blocks | dev_min ns | dev_med ns | dev_max ns | submit_ns med | wait_ns med | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_q | 4096 | 200 | 32 | 4823520 | 4887744 | 5381536 | 8789 | 4987840 | 0.86 | 171.63 |
| attn_q | 4096 | 512 | 32 | 12129888 | 12168704 | 12995840 | 8311 | 12342060 | 0.34 | 176.48 |
| attn_k_v | 512 | 200 | 32 | 497120 | 503072 | 509088 | 1588 | 536699 | 1.04 | 208.43 |
| attn_k_v | 512 | 512 | 32 | 1222720 | 1223840 | 1236224 | 3973 | 1344745 | 0.43 | 219.34 |
| attn_output | 1024 | 200 | 64 | 1109024 | 1112128 | 1290080 | 4163 | 1230015 | 1.89 | 377.14 |
| attn_output | 1024 | 512 | 64 | 2999520 | 3045984 | 3333216 | 19043 | 3306477 | 0.69 | 352.51 |
| ffn_gate_up | 3584 | 200 | 32 | 3939232 | 4119136 | 4300736 | 8332 | 4210572 | 0.89 | 178.19 |
| ffn_gate_up | 3584 | 512 | 32 | 10128064 | 10348640 | 10658848 | 8752 | 10521132 | 0.35 | 181.57 |
| ffn_down | 1024 | 200 | 112 | 1426016 | 1449568 | 1943616 | 9799 | 1581846 | 2.53 | 506.36 |
| ffn_down | 1024 | 512 | 112 | 4073152 | 4180608 | 5145984 | 11384 | 4229667 | 0.88 | 449.47 |

Shapes are the Qwen3.5-0.8B Q8_0 projections from `comparator-shapes.md` §2; `attn_k`/`attn_v`
share `(K=1024, M=512)`. Prompt lengths `tokens ∈ {200, 512}`.

## Control

`device_ns` must strictly grow with `blocks` at fixed `(M, tokens)` (`attn_output` blocks=64 vs
`ffn_down` blocks=112, both M=1024) and with `tokens` at fixed shape. Both hold at every case;
outcome `CONTROL PASS`. The checker is itself controlled: `--control-self-test` feeds it a seeded
inverted (stuck-timer) series and requires rejection — `control self-test PASS: inverted series
rejected`.

## Boundary (obligation 11)

- This is device primitive compute time for one `q8_tile.comp` dispatch. It excludes activation
  quantization, weight upload/first-touch, descriptor/pipeline setup, host submit and wait, and
  the output epilogue's host read.
- No whole-model or whole-prefill number. No matched-comparator result; llama.cpp did not run.
- Residual: a matched comparator (same token list, timer boundary, state reset, warmup) and a
  whole-prefill number remain open. `§10` machine/metric/interval conditions hold for this
  primitive measurement only.
