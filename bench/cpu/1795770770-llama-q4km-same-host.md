# The same-host 0.5× baseline: llama.cpp Q4_K_M and apeiron on the OVH box

Criterion: **T1** (decode) and **T2** (turn), whose 0.5× targets are ratios against llama.cpp.
Grade: `measured` on the OVH box. The 0.5× numbers are ratios, not claims. The laptop pair is
the target-host evidence and is still blocked (`MemAvailable` ~1.0 GiB against the 2 GiB guard);
this receipt is CPU-tier evidence for the box only.

## Result

Same host, same 8 threads, same 1,827-token prefill and 128-token decode, llama.cpp at
`049326a` with a `Q4_K_M` image quantized from the same F16 source, apeiron at `6cfb8eda`.

| phase | llama.cpp Q4_K_M | apeiron | ours / llama |
|---|---:|---:|---:|
| prefill 1,827 tokens | 4,161 ms (445.6 / 423.6 / 448.9 tok/s) | 3,645 ms (499.1 / 499.4 / 505.4 tok/s) | **0.876** |
| decode, 128 tokens | 7.13 ms/token (138.6 / 136.3 / 139.1 / 139.1 / 143.7 / 145.2 tok/s) | 5.79 ms/token (7 runs, 5.670–5.933) | **0.812** |

The decode arm is the shipped lookup path (draft 8, `lookup_accept`, 28 rounds, 102 accepted /
102 drafted, exact). llama.cpp has no speculative arm in `llama-bench`; its 7.13 ms/token is
plain decode, which is the same shape the goal's laptop number (10.92) measures.

The 0.5× targets are therefore **2,080 ms prefill** and **3.57 ms/token decode** on this host:
−43% and −38% from where the tree is.

## Correction to the working baseline

The checkpoint's "decode 0.53×" and "prefill 0.90×" divided box-measured apeiron walls by
laptop-measured llama walls (5.46 / 10.92 and 1.982 / 3.965). Those are cross-host ratios and
are not evidence for either host. This receipt replaces them with a pair taken on one host. The
laptop ratio is still unmeasured and is not inferred from this one.

## What the paired counts say

- Prefill: apeiron is 301.6e9 instructions and 105.1e9 cycles (IPC 2.87) for the 1,827 tokens
  (`/tmp/prefill_base`, `perf stat`). The NQ wide tile is 28% of those cycles at the AVX2
  sign-dot floor (0.117 uops/MAC); even a free tile leaves 76e9 cycles, so 0.5× is not reachable
  by micro-optimisation at the current rank. It needs about half the instructions — a lower-rank
  pack or a wider ISA host.
- Decode: llama reads the 396 MB `Q4_K_M` image once per token at ~55 GB/s and is bandwidth-bound.
  apeiron reads ~228 MB/token (pack 73 MB × 1.27 column-passes per accepted token + the 165 MB
  Q8_0 head × 29/102) at ~39 GB/s effective, so it is compute-bound in the projection pass near
  the same dot floor, not bandwidth-bound.
- The head is 1.14 ms/token of the 5.79 (19.6%); its pipeline ceiling is `1/(1 − 0.196) = 1.24×`.
  The projection pass is 4.5 ms (78%); `ffn` alone is 2.2 ms (38%).

## Machine, artifacts and commands

Box `ubuntu@139.99.223.161`: 8 vCPU EPYC Milan, 32 GiB, no AVX-512/VNNI, `~/bin/adg` with an
8-thread `APEIRON_THREADS`.

| artifact | detail |
|---|---|
| llama.cpp | `~/llama.cpp` at `049326a`, `llama-bench` built with `-DGGML_NATIVE=ON` |
| llama model | `~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base-Q4_K_M.gguf`, 396,704,448 B, `llama-quantize` 6,094 ms |
| our prefill | `/tmp/prefill_base`, sha256 `d9c04b94…`, `prefill_width --tokens 1827 --reps 1` |
| our decode | `/tmp/lookup_base`, sha256 `4f3d7e1e…`, `lookup_accept --first 42 --n 128 --ngram 2 --draft 8` |
| our pack | `nq06-v3.nqp`, 196 linears, 0.977441 b/w, rank sum 124,544 |

```sh
# llama
~/llama.cpp/build/bin/llama-bench -m Qwen3-0.6B-Base-Q4_K_M.gguf -p 1827 -n 0 -t 8 -r 1 -o json
~/llama.cpp/build/bin/llama-bench -m Qwen3-0.6B-Base-Q4_K_M.gguf -p 0 -n 128 -t 8 -r 1 -o json
# ours
APEIRON_THREADS=8 /tmp/prefill_base --model skeleton-q8head.gguf --pack nq06-v3.nqp --tokens 1827 --reps 1
APEIRON_THREADS=8 /tmp/lookup_base --model skeleton-q8head.gguf --pack nq06-v3.nqp --first 42 --n 128 --ngram 2 --draft 8
```

llama prefill, three runs: 4,100.2 / 4,313.0 / 4,070.0 ms. llama decode, six runs:
923.6 / 939.1 / 920.5 / 920.3 / 890.9 / 881.7 ms for 128 tokens. apeiron prefill:
3,660.7 / 3,658.3 / 3,614.8 ms. apeiron decode: 5.933 / 5.875 / 5.779 / 5.796 / 5.759 / 5.721 /
5.670 ms/token; one earlier run at 6.604 was a box-noise outlier and is excluded from the mean,
not the range.

## Scope

- Box only; no laptop row. The timed target is the laptop (i5-10210U, 15 W), where both engines
  scale differently and the ratio must be re-measured as a pair.
- llama.cpp's `llama-bench` figures include its own warmup in the process totals used for the
  instruction profile above; the walls in the table are its reported per-rep times.
- The quality of the two images is not compared here: the pack's PPL and the `Q4_K_M` PPL are a
  separate receipt.
- No code change. Rust/LLVM codegen, the kernel ABI and the `unsafe` container are outside the
  claim.
