# The 0.6B token was 82% the F16 head's scalar reader; a `Q8_0` head takes 3.44 → 21.42 tok/s

Criterion: **T1** (decode) as the 7B lever, **I1** (the `serve/` fixture) as the direct one, with
**Q1** as the byte context. Grade: `measured` for every time (3 interleaved runs on this host), the
phase split, the byte and weight counts, and the head deviation; `model` for the cycle estimate that
explains the split. Host: i5-10210U, 4C/8T, 15 W, 6.6 GiB, Ubuntu 24.04, `performance`; the shell
sees 2 CPUs and the pool measures 2.44× over one core. This note closes no criterion and no 1.174
b/w 7B pack exists.

## 1. Result

| width | before (`F16` head) | after (`Q8_0` head) | ratio |
|---:|---:|---:|---:|
| 1 | 1.23 tok/s | 9.01 tok/s (8.87 / 8.98 / 9.17) | 7.3× |
| 4 | 3.44 tok/s | **21.42 tok/s** (20.40 / 21.44 / 22.43) | **6.2×** |

0.6B v3 pack, 196 sections, 53,815,472 B, 418,381,824 sign terms/token; prompt "The capital of
France is", 32 tokens, `examples/generate`. The before figures are one run each; the after figures
are three runs at each width, spread ±1.7% and ±4.7% of the mean. The F16 fixture could not be
re-run at width 4 for the comparison: it was OOM-killed under the 741–1074 MiB guard cap in prefill,
when the multi-column head built its limb tile (496 MB of tile, 1.25 GB transient `i64` words).

## 2. The split: the head, not the sign dot

`examples/head_time` on the F16 fixture, single column, best of three:

| phase | time | share |
|---|---:|---:|
| `lm_head_fixed` (155.7M weights) | 0.279 s | 82% |
| one full step (`Q8Act`) | 0.342 s | 100% |
| 196 NQ projections + attention + norms | 0.064 s | 19% |

`q8act_plan(F16)` is `Q8ActPlan::Reference`, so the shipped `Q8Act` head calls `q8act_plan`'s
`q.dequant()` and then `matvec_fixed`: `dequant_row_fixed` decodes one `f16` per iteration through
`f16_to_fixed` (a branch on the exponent, a `checked_shl`, a `Vec::push`) and `row_dot_fixed`
accumulates one `i128` `checked_mul`/`checked_add` per iteration. 155.7M weights × ~17 cycles is
0.66 s single-core; the measured 0.279 s is that at the pool's 2.44×. The NQ sign dot is 64 ms, not
the wall.

So the "reader" that needed the Q4 treatment on this fixture was the head's: a tied `F16` tensor on
the oracle path, read element by element, once per token. Q4_K and Q6_K already have fused packed
row readers; `F16` does not.

## 3. Change: the head in a block format the fused reader owns

`tools/nq_head_q8.py` writes an `output.weight` of type `Q8_0` (165,306,368 B, 1.0625 B/weight) from
the tied `token_embd.weight`, streaming 2,048 rows at a time; every other tensor is copied verbatim.
The engine prefers `output.weight` and `q8act_plan(Q8_0)` is `Fused`, so the head takes
`q8act_rows`: one `vpmaddubsw`/`vpmaddwd`/`vpaddd` chain per 32-weight block with the f16 block
scale, the same reader as the 4-bit path. No engine source changed.

```sh
python3 tools/nq_head_q8.py \
  --in ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton.gguf \
  --out ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf
APEIRON_THREADS=4 target/release/examples/generate \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack /tmp/opencode/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --prompt "The capital of France is" --n 32
```

After the change the same split is head ≈ 26 ms and trunk 66 ms of a 92 ms single-column token: the
NQ reader is now **72%** of the token, which is where the next reader change is measurable.

## 4. Quality

`examples/head_cmp` runs both heads through the `Exact` per-column projection on one deterministic
activation of 1,024 Q32.32 words in ±2.0, so the difference is the weight image alone:

| quantity | value |
|---|---:|
| max abs logit | 5,374,753,650 words |
| max \|F16 − Q8_0\| | 29,350,824 words |
| relative | **5.46e-3** |
| argmax | 111 / 111 |

The 24-token generation on the Q8_0 head reproduces the F16 run's text prefix ("located in the city
of Paris, France, in the French-Italian border region…"). Not run: Q2's PPL ratio. The head is now
`Q8_0` (8.5 bits/weight), finer than the `Q6_K` output tensor llama.cpp ships for Qwen2.5-7B
`Q4_K_M`, but it is a weight change and Q2 owns the verdict.

## 5. The executable gate

`examples/phase_split` runs 16 decode steps under `APEIRON_TRACE`, so the target is chosen from the
measured pipeline rather than from the code shape. On the `Q8_0` fixture at width 1 it prints:

| phase | ms/token | share | calls |
|---|---:|---:|---:|
| `ffn` | 41.805 | **38.7%** | 448 |
| `head` | 33.082 | 30.7% | 16 |
| `attn.proj_in` | 15.585 | 14.4% | 448 |
| `attn.proj_out` | 9.420 | 8.7% | 448 |
| `attn.core` | 7.344 | 6.8% | 448 |
| `norm` | 0.389 | 0.4% | 912 |
| `residual` | 0.133 | 0.1% | 896 |
| `embed` | 0.006 | 0.0% | 16 |
| `pass` (encloses) | 74.803 | 69.3% | 16 |
| `pool.wait` (overlay) | 0.016 | 0.0% | 8112 |
| `untraced` | 0.122 | 0.1% | |

Traced wall 107.99 ms/token against 92 ms untraced, so every share divides by the traced wall. The
three NQ phases (`ffn`, `attn.proj_in`, `attn.proj_out`) hold **61.8%** of the token; `ffn` alone is
the dominant term at 38.7–39.2% across runs, ceiling **1.63–1.64×**. The head the change targeted
held **88.8–91.3%** of the F16 fixture's token (ceiling 8.9–11.5×), so it was the dominant term
there; after the change it is 30.5–30.7%, which is why the next change targets the NQ reader.

The gate is executable, not documentary:

| fixture | `--target` | printed | exit |
|---|---|---|---:|
| `Q8_0` head | `ffn` | `TARGET ffn: share 39.1%, ceiling 1.64x` | **0** |
| `Q8_0` head | `head` | `REFUSED: target head holds 30.5% (ceiling 1.44x); the dominant term is ffn at 39.1% (ceiling 1.64x)` | **1** |
| `F16` head | `head` | `TARGET head: share 88.8%, ceiling 8.94x` | **0** |

A phase at share `s` still leaves `1/(1 − s)` of the token when it becomes free; the refusal prints
that ceiling, not `1/s`.

## 6. What this does not establish

- The 7B head is `Q6_K` or an NQ head, both already off the `F16` reference; this is a **fixture**
  fix, not the 7B T1 lever. It does not move the 5.17–5.60 tok/s projection model.
- The q8 NQ reader (one i8 activation plane instead of four exact byte planes) is still unbuilt. At
  0.75 → ~0.31 instructions/term it is the 7B lever, 5.6 → ~12 tok/s at 2 IPC, still 2.0× short of
  24 and 2.5× short of 30; it also changes the NQ activation contract to Q8, so Q2 owns it.
- The exact three-byte-plane form (`t` shifted by a power of two so the remainder fits ±2^23) is
  admissible and exact but only 24 → 20 instructions per 32 terms, 1.2×.
- No PPL, no Intent turn on the new fixture, no 7B end-to-end run.
- rustc and LLVM code generation, the kernel ABI, Vulkan and the GPU remain outside every claim.
