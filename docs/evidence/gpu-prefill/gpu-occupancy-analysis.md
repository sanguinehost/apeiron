# Per-invocation cost attribution — Apeiron Q8 SPIR-V prefill kernel

Where the 30.8×–304× gap from the Apeiron Q8 SPIR-V primitive to llama.cpp CUDA MMQ goes, at
grade **measured**. This file resolves the citation at `matched-cuda-baseline.md:53` ("Apeiron's own
cost is further dominated by its per-invocation term (`gpu-occupancy-analysis.md`)"), which was
unreceipted.

- Command: `timeout 540 python3 tools/gpu/q8_occupancy.py --controls` (cwd repository root)
- Result: exit 0, elapsed **11.39 s**, 55 cases, control red observed under the harness; a second
  `--controls` run is likewise green at 13.17 s, so the figures below are one run and the run-to-run
  band is stated where it matters.
- Base: rsync of the working tree, git HEAD `ee180c13d5356f3c9721cc939b29a2b504a69885`
- Executing artifacts measured (sha256):
  - `src/spirv/q8_tile.comp` `0f971a8aa762676b5b12b32399ef749af7f8e3a7e7722e8ab157693f9f24c60e`
  - `tools/gpu/q8_driver.c` `6668974f92dbc06bc96c9529f09530c132e6521bd517455e9863e66fb6548b6f`
  - `tools/gpu/q8_occupancy.py` `b377390c948b509a8a942a20caa7265a434501bb0c96f9dacea91c29c211afa7`
- Device: NVIDIA GeForce RTX 5080, Vulkan driver `2580660800` (615.71.09), api `1.4.351`,
  `timestampPeriod 1.00 ns`, `timestampValidBits 64`. CUDA device-property probe (same physical
  GPU): 84 SMs, 1536 threads/SM → 2016 resident workgroups at 64 threads each, 256-bit bus,
  64 MiB L2. Peak bandwidth 960.1 GB/s `= 15001 MHz x 2 x 256/8` (anchor: nvidia-smi memory clock,
  not a bandwidth microbenchmark).

## 1 · Instrument

`tools/gpu/q8_driver.c` writes a `VK_QUERY_TYPE_TIMESTAMP` before and after `vkCmdDispatch` and
reads the pair back with `VK_QUERY_RESULT_WAIT_BIT`; `device_ns` is the GPU delta across one
dispatch. The harness reuses the existing compile path (`q8_bench.compile_shader/compile_driver`)
and the existing input synthesis, and adds no production path. Per dispatch:

- geometry (`q8_tile.comp`): `local_size 8x8` = 64 invocations/workgroup; each invocation a
  2-row x 4-token tile, so one workgroup covers 16 rows x 32 tokens.
- `ng = ceil(M/16) * ceil(tokens/32)` workgroups; `blocks = K/32`; `work_units = ng * blocks`.
- `MACs = M * tokens * blocks * 32`; `MACS_PER_WG_BLOCK = 16*32*32 = 16384`.

Four sweeps, `reps = 9`, medians: **occupancy** (blocks=64 fixed, M x tokens grid), **work**
(M=1024, tokens=256 → ng=512, blocks 1..224), **isolated** (ng=1, blocks 1 and 64) and the five
**real** Qwen3.5-0.8B projection shapes at tokens=200/512.

## 2 · Measured

Occupancy sweep, blocks = 64 (grid size `ng` varies; `device_ns_med` in ns):

| M | tokens | ng | dev_med | GMAC/s | uniq GB/s | intensity |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 1 | 222176 | 4.72 | 0.51 | 9.3 |
| 64 | 256 | 32 | 307616 | 109.08 | 2.98 | 36.6 |
| 256 | 256 | 128 | 518272 | 258.97 | 3.67 | 70.6 |
| 512 | 256 | 256 | 803424 | 334.11 | 4.00 | 83.6 |
| 1024 | 256 | 512 | 1445120 | 371.51 | 4.04 | 92.0 |
| 2048 | 256 | 1024 | 3331424 | 322.31 | 3.32 | 96.9 |
| 4096 | 256 | 2048 | 8373856 | 256.45 | 2.57 | 99.6 |

(`dev_med` grows monotonically in `ng` within 3% noise; throughput rises to ~371 GMAC/s near
ng≈512, then falls as the grid grows past ~1 wave of 2016 workgroups.)

Work sweep, M=1024, tokens=256 (ng=512), blocks varies:

| blocks | dev_med | GMAC/s | uniq GB/s | intensity |
|---:|---:|---:|---:|---:|
| 1 | 1049888 | 7.99 | 3.04 | 2.6 |
| 8 | 1085088 | 61.85 | 3.21 | 19.3 |
| 32 | 1221440 | 219.77 | 3.68 | 59.8 |
| 64 | 1436864 | 373.64 | 4.06 | 92.0 |
| 112 | 1857280 | 505.86 | 4.23 | 119.7 |
| 224 | 2874976 | 653.59 | 4.37 | 149.7 |

Isolated single workgroup (ng=1, M=16, tokens=32):

| blocks | dev_med | GMAC/s |
|---:|---:|---:|
| 1 | 5792 | 2.83 |
| 64 | 223200 | 4.70 |

Real projection shapes (Qwen3.5-0.8B):

| shape | tokens | ng | dev_med | GMAC/s | uniq GB/s | logic GB/s |
|---|---:|---:|---:|---:|---:|---:|
| attn_q | 200 | 1792 | 4875520 | 172.06 | 2.92 | 8.09 |
| attn_q | 512 | 4096 | 12196448 | 176.07 | 2.46 | 7.61 |
| attn_k_v | 200 | 224 | 503616 | 208.21 | 3.94 | 10.18 |
| attn_k_v | 512 | 512 | 1234944 | 217.37 | 3.45 | 9.82 |
| attn_output | 200 | 448 | 1119424 | 374.68 | 4.48 | 15.72 |
| attn_output | 512 | 1024 | 3014528 | 356.19 | 3.17 | 13.61 |
| ffn_gate_up | 200 | 1568 | 4029504 | 182.16 | 3.10 | 8.57 |
| ffn_gate_up | 512 | 3584 | 10363840 | 181.31 | 2.54 | 7.85 |
| ffn_down | 200 | 448 | 1465824 | 500.74 | 4.73 | 19.75 |
| ffn_down | 512 | 1024 | 4858272 | 386.77 | 2.48 | 13.81 |

## 3 · Attribution

1. **Dispatch overhead — negligible.** Isolated `ng=1, blocks=1` costs **5792 ns**; at the real
   shapes that is **0.047%–1.15%** of `device_ns`. The single-dispatch floor is ~6 µs, not the gap.
2. **Per-workgroup latency exposure — large and occupancy-dependent.** One isolated workgroup at
   ng=1 pays **3451 ns/block** (223200−5792 over 63 blocks; second run 3500 ns/block). At ng=512 the
   deep-K marginal cost (blocks ≥ 64) is **17.59 ns per (workgroup, block)**, fit R² **0.9999** → the
   isolated workgroup pays **~196×** the marginal per block. The serial b-loop (per block: 2 dm +
   4 e + 48 packed-word loads feeding eight dependent checked-i128 accumulations) exposes memory and
   arithmetic latency; concurrent workgroups are what hide it.
3. **Arithmetic plateau.** The deep-K slope gives **931.5 GMAC/s** (`= 16384 / 17.59 ns`,
   **1.07 ps/MAC**) in the cited run; a second run gives 908.8 GMAC/s (band **909–932 GMAC/s**).
   The best throughput observed anywhere is 653.6 GMAC/s (work, blocks=224); the real shapes peak
   at 500.7 GMAC/s = **54%** of the plateau. CUDA MMQ measures 6425–53644 GMAC/s, so the kernel's own
   plateau is **6.9×–59×** below CUDA.
4. **Occupancy.** Marginal per-workgroup cost over ng=1..2048 is 3731.5 ns/workgroup (linear fit
   R²=0.978; second run 3678 ns; the curve is convex, so this is a local bound). Throughput rises
   with grid size to ~371 GMAC/s at ng≈512 and falls past one wave (2016 resident workgroups), i.e.
   the grid is small at the low end and wave-quantized at the high end.
5. **Memory — not the limit.** Maximum achieved unique-byte bandwidth is **7.18 GB/s (0.75% of the
   960 GB/s peak)**; logical-byte (weight re-read per token tile) is 22.24 GB/s (2.32%). Minimum
   arithmetic intensity is 2.10 MAC/byte against a roofline knee of 0.97 MAC/byte, so every measured
   point has ≥2× bandwidth headroom. The 8.4 MB weight working set fits the 64 MiB L2, so the
   logical re-read is largely L2-served regardless.

## 4 · Verdict

**Compute/instruction-throughput-bound with exposed per-workgroup latency; not memory-bound and not
launch-bound.** At the real shapes the time is 99%+ kernel work; ≤0.75% of peak bandwidth is used;
dispatch is ≤1.15%. The gap to CUDA MMQ is the arithmetic rate: even when the grid is saturated
enough to hide latency, the kernel reaches ~909–932 GMAC/s, **6.9×–59×** below CUDA, because it
issues scalar `dotPacked4x8EXT` over 8 packed quads with an i128 epilogue per block instead of
tensor-core int8 MMA.

**Single next optimization target:** the **cooperative-matrix / tensor-core int8 path** — the int8
MMA the device exposes (`coopmat-design.md`, grade `anchor`; llama.cpp `mmq.cu` routes `tokens>8`
to `ggml_cuda_mul_mat_q`), carrying the checked i128 epilogue. Occupancy-aware tiling is the second
term (the isolated-workgroup penalty shows per-workgroup parallelism has value), but it cannot close
a 6.9×–59× arithmetic gap and is not the next target.

## 5 · Controls

- `--control-self-test` (pure): a seeded monotone, wide-range series is accepted; a seeded flat
  (stuck-timer) series and a flat control pair are rejected; a seeded inverted series is rejected.
- Live occupancy series: non-decreasing in `ng` within 3% per-rep noise, dynamic range 261344 →
  8373856 ns (32×) at tokens=256.
- Live control pair: work sweep blocks=16 → 1130368 ns vs blocks=224 → 2874976 ns = **2.54×**
  (threshold 1.50×).
- **Control observed firing under this harness:** the real occupancy results are copied and every
  `device_ns_med` set to their median (a stuck timer); the range check reddens with
  `occupancy series tokens=32 dynamic range 1.00x < 5.0x: instrument sees no change`; the unmutated
  series passes. A check that cannot fail has no weight, so the gate exits non-zero on any of the
  above.

## 6 · Boundary (obligation 11) and residuals

- Device-time kernel primitive only. It excludes host transport, whole-model throughput, attention,
  the LM head and state reset; the driver dispatches one projection.
- The CUDA comparator is from `cuda_q8_gate.py`, a different harness: the ratio is device-time-per-
  projection on both sides and is not epilogue-matched (Apeiron computes the checked i128 epilogue;
  CUDA a float epilogue plus a separate on-device activation quantization), and is not timer-matched.
- One run at `reps=9` medians; per-case `min..max` spread is ~5–15% and the coefficients are for this
  run's source/tool identity. The occupancy model is a convex-curve local bound, not a steady-state
  plateau.
- Peak bandwidth 960.1 GB/s is derived from the nvidia-smi memory clock and bus width (anchor); no
  bandwidth microbenchmark was run.
- The optimisation this analysis targets is blocked on the `AGENTS.md` §9.1 admission-model
  question: a GLSL/SPIR-V kernel has no Aeneas-extracted Rust executing source, so no admission
  certificate can be constructed under the current `proof/DispatchAdmission.lean` contract. This file
  supplies the cost-model half; it admits no worker and proves no theorem.
- No resident GPU prefill exists, so no whole-prefill speed claim follows.

## Reproduce

```sh
cd /home/socol/oc-tmp/apeiron-gpucost
timeout 540 python3 tools/gpu/q8_occupancy.py --controls --json /tmp/opencode/q8_occupancy_results.json
```
