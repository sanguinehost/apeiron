# The 0.5× pair at 2, 4 and 8 threads

Criterion: **T1** decode rate and **T2** turn latency, as same-host ratios against llama.cpp. Grade:
`measured` on the OVH box (CPU tier, 8 vCPU EPYC Milan); the laptop pair is blocked
(`MemAvailable` 821 MiB against the 1 GiB guard), so no laptop row exists.

## Result

llama.cpp `049326a`, `Q4_K_M` (596.05 M params, 372.65 MiB), `llama-bench`; apeiron
`nq06-v3.nqp` + `skeleton-q8head.gguf`, `prefill_width --tokens 1827` and
`lookup_accept --lookup-only --draft 8` (130 tokens, 102 accepted / 28 rounds, exact). Ratio is
ours / llama; 0.5 is the target.

Prefill, 1,827 tokens:

| threads | llama ms | ours ms | ratio | 0.5× target | gap |
|---:|---:|---:|---:|---:|---:|
| 2 | 14,148 | 12,829–12,885 | 0.91 | 7,074 | 1.82× |
| 4 | 7,233 | 6,253–6,267 | 0.865 | 3,617 | 1.73× |
| 8 | 4,161 | 3,594–3,645 | 0.864–0.876 | 2,081 | 1.74× |

Decode, ms/token:

| threads | llama | ours | ratio | 0.5× target | gap |
|---:|---:|---:|---:|---:|---:|
| 2 | 15.46 | 13.75–14.63 | 0.89–0.95 | 7.73 | 1.84× |
| 4 | 9.87 | 9.46–9.59 | 0.96–0.97 | 4.94 | 1.93× |
| 8 | 7.13 | 5.79–5.97 paired; 7.0 latest | 0.81–0.84; 0.98 | 3.57 | 1.62×; 1.98× |

llama: `llama-bench -p 1827 -n 0 -t T` and `-p 0 -n 128 -t T`; ours: `prefill_width --tokens 1827
--reps 1` and `lookup_accept … --lookup-only`, `APEIRON_THREADS=T`. The 2- and 4-thread rows were
measured in this session; the 8-thread row is the interleaved receipt
`bench/cpu/1795770770-llama-q4km-same-host.md`.

## What the width-gate commits bought

`594a26fe` (partial column group) and `61d65692` (tile at every participant count), receipt
`bench/cpu/1795774700-nq-tile-all-widths.md`:

| width | prefill | decode |
|---:|---:|---:|
| 2 | 19,777 → 12,829–12,885 (1.54×) | 16.82–17.48 → 13.75–14.63 (1.20–1.22×) |
| 4 | 10,121–10,341 → 6,253–6,267 (1.61×) | 10.41–10.52 → 9.46–9.59 (1.08×) |
| 8 | flat | flat |

The low-width pair moved from **1.40 prefill / 1.09–1.13 decode** to **0.91 / 0.89–0.95**. Exact
102/102 at every width; PPL 238.498293 in both tile and pair arms.

## Reading

- The distance to 0.5× is ~1.7–1.9× at every width. The gate find restored low-width parity; it
  did not change the width-8 ratio and did not close the structural gap.
- Our pack does 0.418 GMAC/token against llama's dense ~1.2 GMAC, 2.9× less projection math, yet
  total is only ~1.1× faster. At 2 threads the projections run at ~16.6 MACs/cycle/core against
  llama's ~31; the tile alone is 26–37, so the machinery around the fold is about half the
  projection time.
- Prefill shares: projections 57–59% (ffn 35, proj_in 16, proj_out 7.5), attention 33–36% at both
  widths. No single phase's `1/(1−s)` ceiling exceeds 1.56.
- Decode's dominant term moves with width: head 32.5% at 8 threads, **38.7% at 2** (5.53
  ms/token), then ffn 26.3%. `head_bench` cols 9 is 53.1 ms at 2 threads and 12.14 at 8
  (~4.4–4.8 MACs/cycle); the decode trace's head call is ~25 ms/round. Cost scales near-linearly
  in columns (10.66 → 53.11 for 1 → 9), so the per-column dot/reduce owns it, not the weight read.

## Scope

- Box only; no laptop row. The laptop is `nproc = 2`, so the 2-thread row is its shape, but the
  timed claim waits for the guard.
- Ours is speculative decode (draft 8, 3.64 accepted/round) against llama's plain `tg128`; the
  0.81 at 8 threads is the draft path's, while plain greedy is 19.2 ms/token.
- The box's 8-thread decode drifts 5.8 → 7.0 ms/token on one binary between sessions; only the
  interleaved pair is evidence, and the 2/4 rows are single-session pairs.
- Rust/LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.
