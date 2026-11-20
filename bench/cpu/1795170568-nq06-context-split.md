# The 0.6B at 1,827 tokens: `attn.core` owns 49.8%, not the head

Criterion: **T1** (decode) and **C1** (context state). Grade: `measured` for the rates, the split and
the RSS; `model` for the byte budgets of the projected carriers. Machine: ThinkPad L13 Yoga 20R6,
i5-10210U (4 cores / 8 threads, 15 W, AVX2/BMI2/POPCNT, no AVX-512, no AVX-VNNI), 6.6 GiB,
Ubuntu 24.04, `performance`, `APEIRON_THREADS=4`. Load average 0.56 at the start of the clean runs;
the earlier run under two `npm install` processes is discarded.

Result: the Qwen3-0.6B pair decodes at **13.5 tok/s at 1,827 tokens** (three runs, 11.58–14.64) and
**38.5 tok/s at 5 tokens**. The phase split flips: at the empty prefix the Q8_0 head owns **36.0%**;
at 1,827 `attn.core` owns **49.8%** at 38.11 ms/token, and the head is 20.5%. The attention read is
the target at 2k, not the head.

## Commands

```sh
# rate, three runs each
target/release/examples/generate \
  --model /home/lucas/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack /home/lucas/models/nq-scratch/nq06-v3.nqp \
  --tokenizer /home/lucas/models/qwen3-0.6b-base-e8/tokenizer.json \
  --prompt "<5 tokens | 1,827 tokens>" --n 32

# split, 8 traced steps after the prefix
target/release/examples/phase_split \
  --model /home/lucas/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack /home/lucas/models/nq-scratch/nq06-v3.nqp \
  --steps 8 --prefix 1827
```

`--prefix N` is the new instrument: it appends `N` copies of token 1 without tracing, so the split is
taken at a real cache length. The prefix is a single repeated token, so the numbers are timing, not
quality; the KV bytes and the per-position dot count are the same as text.

The 4 GiB `MemoryMax` scope used for the rate runs is stated because the guard's derived cap at
1.6 GiB `MemAvailable` (589 MiB) is below the 875 MB peak of this configuration.

## The rates

| prefix | prefill | decode | ms/token | max RSS |
|---:|---:|---:|---:|---:|
| 5 | — | 38.49 tok/s (1 run) | 25.98 | — |
| 1,827, run 1 | 31.95 tok/s | 14.32 tok/s | 69.84 | 839 MB |
| 1,827, run 2 | 28.01 | 11.58 | 86.36 | 867 MB |
| 1,827, run 3 | 28.35 | 14.64 | 68.31 | 874 MB |
| **1,827, mean** | **29.44** | **13.51** | **74.8** | **860 MB** |

Decode spread 11.58–14.64 (26% of the mean); prefill spread 28.01–31.95. The spread is the run, not
the cap: the 875 MB peak is 89% of the 4 GiB scope and the guard was not in the queue.

## The split

| phase | prefix 0, ms/token | share | prefix 1,827, ms/token | share |
|---|---:|---:|---:|---:|
| **attn.core** | 2.395 | 9.2% | **38.109** | **49.8%** |
| head (Q8_0) | **9.350** | **36.0%** | 15.697 | 20.5% |
| ffn | 8.018 | 30.9% | 12.486 | 16.3% |
| attn.proj_in | 3.761 | 14.5% | 6.117 | 8.0% |
| attn.proj_out | 1.738 | 6.7% | 2.913 | 3.8% |
| norm | 0.417 | 1.6% | 0.644 | 0.8% |
| residual | 0.141 | 0.5% | 0.232 | 0.3% |
| untraced | 0.113 | 0.4% | 0.256 | 0.3% |
| **token** | **25.94** | | **76.46** | |
| `pool.wait` (overlay) | 1.174 | 4.5% | 5.544 | 7.2% |

Dominant term: `head` at prefix 0 (ceiling 1.56×), **`attn.core` at 1,827** (ceiling 1.99×). The 8
traced steps at 1,827 give 76.46 ms/token; the untraced 32-token mean is 74.8, so the trace is inside
the spread.

## The bytes

| term | bytes/token | source |
|---|---:|---|
| NQ pack (196 linears, 0.977441 b/w) | 53,808,384 | `nq::open` print |
| `output.weight`, Q8_0, 151,936×1,024 | 165,306,368 | GGUF tensor table |
| KV, F32, 28 layers × 8 kv heads × 128 × 2 | 229,376 per position | `KvState` geometry |
| KV at 1,827 positions (unique) | 419,069,952 | product |
| KV issued, 16 query heads over 8 kv heads | 838,139,904 | ×2 |
| **unique total** | **638,184,704** | |
| **issued total** | **1,057,254,656** | |

At 76.46 ms that is **8.35 GB/s unique** and **13.83 GB/s issued**. `attn.core` alone is 419.07 MB
unique / 838.14 MB issued in 38.109 ms = **11.00 / 22.00 GB/s**. The attention kernel is at 76% of the
28.8 GB/s 4-thread streaming mean on issued bytes; the NQ pack reader is the slow term at ~3.2 GB/s
(`ffn` 12.49 ms for its ~40 MB share).

The default KV carrier is **F32** (`carrier_from_env` selects F32 unless `APEIRON_F32_CARRIER=0`);
Q32 doubles these bytes and runs the scalar checked path. The certificate index is Q32-only, so it is
not available on this default.

## What it changes

20 tok/s is 50.0 ms/token. The measured split gives the order:

| change | attn.core | token | tok/s | grade |
|---|---:|---:|---:|---|
| today | 38.11 | 76.46 | 13.1 | measured |
| int8 KV, per-(kvh, pos) scale, GQA dedup | 4.9 | 43.3 | **23.1** | model, at the measured 22 GB/s issued rate |
| + head in the pack, rank 1,200, 23.0 MB | 4.9 | 28.9 | **34.6** | model, at the head's 17.7 GB/s |

The int8 KV is 28 × 8 × (2·128 + 8) = **59,136 B/token** at 0.6B geometry, 108.0 MB at 1,827
positions, and the GQA group read makes issued = unique. The carrier is the single change that moves
the dominant term. The head fit is the second: it is 165.3 MB of a 219.1 MB weight term at this
context.

At 1M the same pair is 240.5 GB of F32 KV, 62.0 GB int8 or 31.9 GB at 4 bit (Mi positions); no
dense carrier fits 6.6 GiB. The 1M path is the cold tier plus selection, and this receipt is its 2k
baseline.

## Scope

- One model pair, one machine, one power profile, 4 threads.
- The prefix is 1,827 copies of token 1; the split is timing, not quality.
- F32 KV, no cold tier, no selection, no 1M run.
- The 8B is not re-run here; its 2k point is 2.884 tok/s (346.8 ms) in
  `1794919269-nq8b-split.md`, with the position-dependent term ~62 ms.
- The head rank 1,200 is the 1.18 b/w arithmetic `151,936·1,200 + 1,200·1,024` bits = 22.9 MB; the
  fit is not built.
