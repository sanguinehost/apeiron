# Qwen3-8B decode cost model: 20 tok/s needs 1.42 GB/token, not a faster kernel alone

Criterion: **Q1**. Grade: `measured` for the split, the bandwidth and the byte counts; `model` for
the projections the two bounds give. The result is a **refusal**: 20 tok/s is above the
bandwidth ceiling of this pack at 2k context (14.66 tok/s), and above it at short context
(17.36 tok/s). The kernel is issue-bound today at 0.834 cycles/term; the fast reader already in
the tree would move it to the bandwidth ceiling, not past it.

## Measured

`examples/phase_split` on the served pair, 8 decode steps, empty prefix, `--kernel q8act`:

```
decode split: 8 steps after a 0 token prefix, 5.042 s wall, 630.28 ms/token
| phase | ms/token | share |
| attn.proj_in | 167.931 | 26.6% |
| attn.core | 34.421 | 5.5% |
| attn.proj_out | 51.501 | 8.2% |
| ffn | 299.396 | 47.5% |
| head | 71.950 | 11.4% |
| pass (encloses) | 558.328 | 88.6% |
| pool.wait (overlay) | 310.723 | 49.3% |
| untraced | 0.395 | 0.1% |
dominant term: ffn at 47.5%, ceiling 1.90x
```

The NQ projections (`proj_in` + `proj_out` + `ffn`) are 518.8 ms, **82.3%** of the token. No single
phase owns enough of the token for a targeted change to reach 50 ms: the largest, `ffn`, caps at
1.90×.

Bandwidth, `tools/membw.c`, four threads pinned to the four physical cores, 128 MiB chunk, 25
passes: **28.42 GB/s mean, 30.99 GB/s best pass**.

## The two bounds

```
terms/token      6,945,767,424        (every fitted weight is one sign term; rank is near-full)
bytes/token      1,637,275,152        (976,049,680 pack + 661,225,472 Q8_0 head)
bandwidth        28.42e9 B/s          (measured)
issue            6.95e9 terms × 1.5 ops = 10.4e9 ops; at 0.63 s/token this is 16.5e9 ops/s
```

| bound | value |
|---|---:|
| bandwidth ÷ bytes, short context | **17.36 tok/s** |
| bandwidth ÷ bytes, 2k context (adds 301,989,888 B KV) | **14.66 tok/s** |
| measured | 1.59 tok/s (630 ms/token) |

The measured 0.834 cycles/term matches the exact reader's 1.5 operations/term at about 1.8 IPC. The
AVX2 byte selector (`sign_dot_planes_avx2`, `CodeTerms`) is 32 terms per 9–11 vector operations,
0.31 ops/term, and would put the projection stream at ~0.05–0.1 cycles/term, below the bandwidth
bound. So the kernel change is worth **4–10×**, and it lands on the bandwidth ceiling, not at 20
tok/s.

## The fast reader is not admitted at k = 1

`--kernel q8act` and `--kernel exact` decode the same 2 tokens in the same time on this pack:

```
q8act   prefill 6.343 s, 4.57 tok/s   decode 2.319 s, 0.86 tok/s
exact   prefill 12.562 s, 2.31 tok/s  decode 1.254 s, 1.59 tok/s
```

Prefill is 2.0× faster with `q8act` (multi-column takes the quantized reader); decode is not
(0.63 s/token in both, from the traced run). `CodeTerms::new` admits a stage only when
`norm·508 ≤ i32::MAX`, `norm = Σ_b 2^{shift_b}` over the stage's 32-term blocks
(`src/engine/kern/nq.rs:551`). A rank-2528 `V` stage is 79 blocks; any spread of block exponents
past ~2^16 in the sum takes it out of that domain, and `project_with` falls back to `Terms`, the
exact `i64` reader, for that stage. The conservative `i32` lane bound is what forces the 1.5-op
reader.

**Verification obligation.** Widen the lane bound so the pack's actual stages are admitted: prove
`508·Σ_b 2^{shift_b} ≤ i32::MAX` from the block scales the quantizer produces at rank 2528, or move
the selector's four-code lane to `i64` and prove the wider sum exact. This is a range theorem over
`quant_act`, not a kernel rewrite; it is the 4×.

## Byte budget for 20 tok/s

`20 tok/s` at 28.42 GB/s allows **1,421,000,000 B/token**. Current and candidate rows, 2k context:

| configuration | bytes/token | ceiling |
|---|---:|---:|
| today (Q8_0 head, F16 KV) | 1.939 GB | 14.66 |
| Q6_K head | 1.789 GB | 15.89 |
| Q5_K head | 1.706 GB | 16.66 |
| Q5_K head + 4-bit KV | 1.479 GB | 19.21 |
| 1.00 b/w pack + Q6_K head + 4-bit KV | 1.454 GB | 19.54 |
| **1.00 b/w pack + Q5_K head + 4-bit KV** | **1.372 GB** | **20.72** |

Only the last row clears 20 at 2k, and by 3.6%. The head alone cannot do it; the rate target and
the KV width are part of the same inequality. At short context the pack + a Q5_K head
(1.404 GB) gives 20.24.

The 1.00 b/w pack is Q3's target, so the rate work and the speed target are the same work. The
4-bit KV is C1's carrier. Nothing here needs a new kernel beyond the reader the tree already has.

## What this does not cover

- One pack, one machine, governor `powersave`, a shared box; the 28.42 GB/s mean is over 25 passes
  on an idle-ish machine, not the server's access pattern under load.
- The 2-token decode comparison is not a mean over three; the traced 630 ms/token is the anchor.
- `pool.wait` is 49.3% of the token: the work is cut too fine to schedule. Coarser dispatch is a
  constant-factor term this model does not price.
- The head change is a quality change. Q5_K against Q6_K on the same token stream is not measured,
  and Q2's same-head clause would need the reference re-run at the same head.
- The 4-bit KV row assumes the cold/hot split in C1; the hot ring's bytes are not priced here.

## Next

1. Admit the fast reader at k = 1: the range theorem above, then re-run `phase_split` with
   `--target ffn`; the expected landing is the bandwidth bound, 14–17 tok/s.
2. Price `pool.wait`: 310 ms/token of the 630 is spent waiting, not computing. A coarser dispatch
   is worth up to 2× before any kernel changes.
3. Then the byte budget: head carrier and KV width, in that order, against the same inequality.
