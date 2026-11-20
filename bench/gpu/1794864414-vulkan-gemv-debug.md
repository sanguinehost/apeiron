# Vulkan Q4 GEMV: the zero was the harness, not the kernel

| field | value |
|---|---|
| criterion | **V1** (Vulkan/SPIR-V/Lean) — **advanced, not closed**: the Q4 DP4A GEMV now submits, matches its CPU reference and has a timed row; no 7B pack and no end-to-end logits-vs-CPU run exists |
| grade | **measured** (device timestamp rows, three process runs, exact integer face on 256/256 rows) |
| instance | vast.ai `60973058`, host `149975`, container hostname `e770db313f8d`; **destroyed** after the run |
| GPU | NVIDIA GeForce RTX 3060, 12288 MiB, Ampere; driver 580.173.02, proprietary module; SM max 2130 MHz, memory max 7501 MHz |
| host | AMD EPYC 7452, 8 allocated vCPUs (host exposes 64), 125 GiB RAM |
| container | `ubuntu:24.04`, `libvulkan1` + `vulkan-tools` + X/GL userland (`libx11-6 libxext6 libglvnd0 libgl1 libegl1 libxcb1 libxau6 libxdmcp6`), created with `--env '-e NVIDIA_DRIVER_CAPABILITIES=all'` |
| price | $0.0708/h |
| spend | **$0.042** over 0.544 h (GPU $0.030, storage $0.008, download $0.004); account credit 3.67 → 3.63; budget $0.25 |
| artifacts | `examples/gpu_gemv_bench.rs` sha256 `4de237f2…57fa80`, `examples/gpu_gemv_debug.rs` sha256 `666102f1…f730`; `src/gpu` unchanged |

## Result

The deterministic zero was **not** a kernel fault. Two harness defects, and two layout defects in the
harness controls, produced it:

1. **No submit before the readback.** The decode check created the six buffers, bound the pipeline
   and read `out` back without a `g.run(&disps)`. The only `g.run` in that loop was inside
   `run_timed`, which runs *after* the check. The readback therefore saw a freshly zeroed
   `MEMORY_DEVICE_LOCAL` buffer. `examples/gpu_gemv_bench_orig.rs --quick` on the GPU:

   ```
   check decode segs=8 row=0    got=0e0 want=-4.2433590087890625e3
   check decode segs=8 row=1792 got=0e0 want=-9.385692291259766e3
   check decode segs=8 row=3583 got=0e0 want=-7.822440856933594e3
   thread 'main' panicked at examples/gpu_gemv_bench_orig.rs:294:9:
   decode segs=8: rel 1e0 > 1e-4
   ```

2. **Consecutive activation words where the DP4A phase loop reads stride-2 words.** With the
   submit added, the check moved from 0 to `rel 4.6840743629096326e-1`. A Q4 weight word holds
   eight logical codes `8c..8c+7` in nibbles. Phase `l` masks `0x0F0F0F0F` (bytes =
   nibbles `0,2,4,6`) or shifts by 4 then masks (nibbles `1,3,5,7`), so phase `l` byte `b` carries
   logical code `8c + 2b + l`; it must dot activation word `2c + l` with byte `b` =
   `x[8c + 2b + l]`. The harness packed `codes[4w..4w+4]` into word `w`, pairing phase 0 with
   codes `0..3`.

   The served quantiser does the same deinterleave: `emit_quantize_from_shared`
   (`src/gpu/nn/mod.rs:1084`, pack loop at `1147`) computes `cc = c / dpw`, `ll = c % dpw`,
   `base = cc·4·dpw + ll`, inner index `base + b·dpw`. For `dpw = 2` that is exactly
   `8cc + ll + 2b`. The kernel was right; the benchmark's packer was not.

3. **The float control read the wrong weight layout.** `pack_proj_f32` writes DP4A words
   `words[(kk/8)·M + r]` with nibble `kk%8`; `gemv_bytes(..., Carrier::Q4)` (`gemv_module_impl`)
   reads `words[kk·(M/8) + r/8]` with nibble `r%8`. The harness passed the first buffer to the
   second decoder. A one-pass repack (`float_words_of`) fixes the control; the bytes and the rate
   accounting are unchanged.

4. **The debug harness's F32 control read row-major `chart.w`** where `gemv_module` reads
   `W[kk·M + r]` (column-major). The control now uploads the transpose.

After the fixes, all checks pass and all three benchmark runs exit 0:

```
check decode segs=8 row=0    got=-4.24336e3 want=-4.2433590087890625e3 diff=8.544921875e-4
check decode segs=8 row=1792 got=-9.385693e3 want=-9.385692291259766e3 diff=1.068115234375e-3
check decode segs=8 row=3583 got=-7.822442e3 want=-7.822440856933594e3 diff=1.03759765625e-3
check decode segs=8 f32 max_rel=1.1380249865741502e-7
check decode segs=8 int exact rows=3
check prefill bt=32 f32 max_rel=3.836368916718563e-7 cells=6
check float segs=8 f32 rel=8.630213393245425e-8
```

`gpu_gemv_debug` checks all 256 rows of a known `m=256, k=1024` Q4 chart at both production
split-K shapes, and every row is exact (`max_rel=0e0`):

| face | segs=1 | segs=8 |
|---|---|---|
| Q4 DP4A chart, f16 `(d, lo)` | 256/256, `max_rel=0` | 256/256, `max_rel=0` |
| Q4 DP4A unit face, exact integer dot | 256/256, `max_rel=0` | 256/256, `max_rel=0` |
| direct F32 `gemv_bytes` control (segs-independent) | 256/256, `max_rel=0` | 256/256, `max_rel=0` |

The first eight chart rows at `segs=8` (the same values at `segs=1`):

```
q4_chart row=0 got=7.595e2    want=7.595e2    diff=0e0
q4_chart row=1 got=1.740125e3 want=1.740125e3 diff=0e0
q4_chart row=2 got=5.1025e2   want=5.1025e2   diff=0e0
q4_chart row=3 got=-1.01325e3 want=-1.01325e3 diff=0e0
q4_chart row=4 got=1.026125e3 want=1.026125e3 diff=0e0
q4_chart row=5 got=-1.867625e3 want=-1.867625e3 diff=0e0
q4_chart row=6 got=6.06875e2  want=6.06875e2  diff=0e0
q4_chart row=7 got=1.19525e3  want=1.19525e3  diff=0e0
```

The unit face and F32 control also print `diff=0e0` on rows 0–7 and `max_rel=0e0` over all 256.

The zero is exact because that chart's `(d, lo)` values are chosen f16-exact and the integer sums
are below `2^24`; it is an identity, not a tolerance. The builder's module is also byte-identical to
the committed capture: `gemv_dp4a_module_dyn(4096, 14336, 256, 8, Q4, true)` is 676 words and
equals `CAPTURED_GEMV_DP4A_MODULE_DYN` in `tests/gpu_spv_check.rs`.

### Repository tests on the same GPU

| command | result |
|---|---|
| `cargo test --release --locked --features gpu --test gpu_spv_check` | **6 passed** |
| `cargo test --release --locked --features gpu --test gpu_gemv_match` | **4 passed** in 10.45 s, including Q5/Q6 chart and grid-exact tests on the 0.5B model |
| `cargo test --release --locked --features gpu --test gpu_splitk` | **1 passed** in 1.73 s |
| `cargo test --release --locked --features gpu --test gpu_kernel_timings -- --nocapture` | **1 passed**, 61 dispatch cells + 1 control, 7 runs each |

The tests ran in the order the task requires; the timings test ran only after the first three
passed.

## Timing

`gpu_gemv_bench --quick` (`m=3584`, `k=18944`, Q4, local 256), three process runs, each with 5
warmup + 5 timed dispatches. The median is over the three run medians; the spread is the union of
the three runs' `[min..max]` over their timed samples. `gpu_us` is the device timestamp interval
around the dispatch; host readback and pipeline creation are excluded.

| row | gpu_us median | spread µs | weight GB/s | all GB/s | tok/s |
|---|---|---|---|---|---|
| decode segs=8 | 126.976 | 125.952–129.024 | 334.19 | 336.30 | 7,875.5 |
| prefill bt=32, 512 tokens | 19,452.928 | 19,385.344–19,465.216 | 34.90 | 42.26 | 26,319.9 |
| float control segs=8 | 765.952 | 763.904–768.000 | 66.48 | 66.61 | 1,305.6 |

`tok_s` for decode is one token per dispatch of this one projection, and for prefill is 512 tokens
per dispatch; neither is a model rate. The prefill row reads 678.95 MB of weights in 19.45 ms
(34.90 GB/s), far below the decode row's rate, so it is a weight-reuse/compute measurement, not a
DRAM result.

`all GB/s` uses the benchmark's byte count: weights 42,434,560 B + one activation pass 23,680 B +
output 14,336 B + split-K partials 229,376 B = 42,701,952 B. The kernel's 14 workgroups each re-read
the activation, so the true traffic is 43,009,792 B and the true rate is **338.7 GB/s**; the
reported 336.30 is a 0.7% undercount, quoted as the conservative value.

### 7B ceiling

| quantity | value |
|---|---|
| measured all-traffic rate | **336.30 GB/s** (338.7 corrected) |
| RTX 3060 spec peak | 360 GB/s |
| fraction of spec | **93.4%** |
| 7B 1-bit pack | 0.998244352 GB/token |
| ceiling at the measured rate | **336.9 tok/s** (339.3 corrected) |
| vs T1 30 tok/s / working 20 tok/s | 11.2× / 16.8× |
| ceiling at spec peak | 360.6 tok/s |

The binding row is the measured 336.30 GB/s; the spec-peak row is arithmetic and not achieved.
This is a **projection**, not a 1-bit measurement: the kernel measured is Q4, and the 0.998 GB token
is the whole-model 1-bit pack. What the row establishes is that this GPU and this dispatch shape
sustain at least 336.3 GB/s of counted traffic on the Q4 chart, so the 1-bit pack's bytes alone do
not bind the 20 tok/s target. It does not establish that a 1-bit kernel reaches that rate, nor that
the rest of a 7B token (attention, KV, head) fits the remaining budget. The Q4 whole-model
arithmetic (4.3706 GB/token from the prior receipt) gives 76.9 tok/s at the same rate; that is
likewise arithmetic, not a model run.

## Diff

`src/gpu` is unchanged (`git diff --stat -- src/gpu` is empty). The changes are in the untracked
benchmark and debug harness. The three semantic changes (the full diff also carries `rustfmt`
reflow of untouched lines):

```diff
@@ decode check
+        // f32 face check — submit first: `run_timed` below is the only other caller of `g.run`,
+        // and it runs after this check, so without this line the readback sees an unwritten buffer.
+        g.run(&disps);
         let got = unsafe { readback(&g, out.buffer, M) };

@@ activation pack
-fn xq_of(codes: &[i8]) -> Vec<u32> {
-    codes
-        .chunks_exact(4)
-        .map(|c| {
-            u32::from(c[0] as u8)
-                | (u32::from(c[1] as u8) << 8)
-                | (u32::from(c[2] as u8) << 16)
-                | (u32::from(c[3] as u8) << 24)
-        })
-        .collect()
+fn xq_q4_of(codes: &[i8]) -> Vec<u32> {
+    codes
+        .chunks_exact(8)
+        .flat_map(|g| {
+            (0..2).map(move |l| {
+                u32::from(g[l] as u8)
+                    | (u32::from(g[2 + l] as u8) << 8)
+                    | (u32::from(g[4 + l] as u8) << 16)
+                    | (u32::from(g[6 + l] as u8) << 24)
+            })
+        })
+        .collect()
 }

@@ float control
+        let fwords = float_words_of(&words, M, K);
+        let fwb = g.upload(&fwords);
         let pipe1 = g.pipeline(&gemv_bytes(M as u32, K as u32, LOCAL, segs, Carrier::Q4), 5);
-        let ds1 = g.bind(&pipe1, &[&wb, &xf, &partial, &dmf, &xsb]);
+        let ds1 = g.bind(&pipe1, &[&fwb, &xf, &partial, &dmf, &xsb]);
```

The new `float_words_of` is the transpose loop quoted in full in the file; it maps
`words[(kk/8)·M + r]` nibble `kk%8` to `out[kk·(M/8) + r/8]` nibble `r%8`.

`examples/gpu_gemv_debug.rs` is new: the same six buffers and binding order as the decode bench,
`gemv_dp4a_dyn_bytes(m, k, 256, segs, Carrier::Q4)` at `segs = 1` and `8`, the first eight values
against an f64 reference, an exact integer face, a direct F32 control, and a SPIR-V header walk that
prints the entry point, local size and descriptor bindings:

```
dp4a_dyn Q4: words=676 entry="main" local=256x1x1
  bindings=[(16, 0), (17, 1), (18, 2), (19, 3), (20, 4), (36, 5)]
gemv F32:    words=303 entry="main" local=256x1x1 bindings=[(15, 0), (16, 1), (17, 2)]
```

## Commands

On instance `60973058`, in `/root/apeiron`:

```sh
cargo build --release --locked --features gpu \
  --example gpu_gemv_debug --example gpu_gemv_bench -j 8

./target/release/examples/gpu_gemv_debug --segs 1
./target/release/examples/gpu_gemv_debug --segs 8

cargo test --release --locked --features gpu --test gpu_spv_check
cargo test --release --locked --features gpu --test gpu_gemv_match
cargo test --release --locked --features gpu --test gpu_splitk
cargo test --release --locked --features gpu --test gpu_kernel_timings -- --nocapture

for r in 1 2 3; do ./target/release/examples/gpu_gemv_bench --quick; done
```

The Qwen 0.5B Q4_K_M fixture used by `gpu_gemv_match`/`gpu_splitk` was downloaded to make those
tests run the model path rather than SKIP: `qwen05b-Q4_K_M.gguf`, 491,400,032 B.

## What it does not cover

- **V1 is not closed.** GOAL.md's bar is a 7B pack end to end through the Vulkan path with logits
  agreeing with the CPU path. This receipt covers one synthetic `ffn_down` shape and per-projection
  tests on a 0.5B Q5/Q6 chart. No 7B pack was uploaded; no model logits were compared.
- **No NanoQuant pack was read.** The benchmark weights come from `pack_proj_f32` on random f32
  rows. The Vulkan path still uploads GGUF payloads; that V1 clause is untouched.
- **No 1-bit kernel ran.** The 336.9 tok/s row applies the measured Q4 traffic rate to the 1-bit
  pack's byte count. It is an arithmetic projection, not a measurement of a 1-bit carrier.
- **No speedup is claimed.** The "before" was a harness zero, not a slow kernel. There is no
  `phase_split --target` control and no A/B pair; this is a correctness measurement with a new
  timed row.
- **One device and one driver:** RTX 3060, 580.173.02, proprietary module, one host. No validation
  layers, no llvmpipe comparison (the local llvmpipe run still panics in `Vk::load` at
  `src/gpu/vulkan.rs:1570`, `vkCmdWriteTimestamp2 not found`, and was not changed).
- **V2/V3/V4 untouched:** no SPIR-V provenance change, no Aeneas extraction, no Lean theorem and no
  mutation control for the kernel. The only provenance check here is that the builder's words equal
  the committed capture.
- **Timing scope:** device timestamps around one dispatch, median of three process medians, 5
  warmup + 5 timed dispatches; it excludes pipeline creation, host readback and process startup.
  The allocation was `1x RTX 3060`; the GPU sharing fraction was not recorded, and no clock lock
  was applied.

## Spend

The instance was created for this run and destroyed at the end; `vastai show instances` reports
**0 instances**. Invoice for `60973058`: 0.544 h, GPU $0.030, storage $0.008, download $0.004, total
**$0.042**. Account credit moved 3.67 → 3.63; the budget was $0.25. No other instance was created
or touched.
