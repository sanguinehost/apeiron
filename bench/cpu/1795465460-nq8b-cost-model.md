# Qwen3-8B decode cost model: 20 tok/s needs 1.42 GB/token, not a faster kernel alone

Criterion: **Q1**. Grade: `measured` for the split, the bandwidth and the byte counts; `model` for
the projections the two bounds give. The result is a **refusal**: 20 tok/s is above the bandwidth
ceiling of this pack at 2k context (12.68 tok/s), and above it at short context (17.36 tok/s). A
perfect kernel lands on that bound, not past it. The kernel is **not** the binding term: the Q8
selector is live and 4.5× the exact reader, and the token reads 5.4 GB/s of a measured 28.42 GB/s.

Correction: the first issue of this note priced the KV plane at 2 B/value under an "F16 KV" label.
The engine has no F16 KV carrier — [`KvCarrier`](../../src/engine/attn.rs) is `Q32`, `F32` or `I8`,
and `F32` is the decode default — so the 2k rows below are re-derived at the carrier's 4 B/value.
The verdict is unchanged; the gap on the current configuration is larger than first stated.

## Measured

`examples/phase_split` on the served pair, 6 decode steps, empty prefix, 4 threads, three runs:
**262.72, 336.00, 312.52 ms/token** — mean **303.75 ms/token (3.29 tok/s)**, spread 12%. The
6-step trace at 280 ms/token:

```
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

That trace ran under load, so its 630 ms/token is not the baseline; the three-run mean above is.
The NQ projections (`proj_in` + `proj_out` + `ffn`) remain 82.3% of the token, and the largest
single phase, `ffn`, caps at 1.90×.

Bandwidth, `tools/membw.c`, four threads pinned to the four physical cores, 128 MiB chunk, 25
passes: **28.42 GB/s mean, 30.99 GB/s best pass**.

## The two bounds

```
terms/token      6,945,767,424        (rank is near-full: one sign term per fitted weight)
bytes/token      1,637,275,152        (976,049,680 pack + 661,225,472 Q8_0 head)
bandwidth        28.42e9 B/s          (measured)
today            303.75 ms/token → 5.39 GB/s, 0.402 cycles/term
```

| bound | value |
|---|---:|
| bandwidth ÷ bytes, short context | **17.36 tok/s** |
| bandwidth ÷ bytes, 2k context (adds 603,979,776 B f32 KV) | **12.68 tok/s** |
| measured, mean of three | 3.29 tok/s |

At 0.402 cycles/term the projection stream is arithmetic-bound, not byte-bound: the token reads
5.4 GB/s where the machine sustains 28.4. A kernel that removed every non-essential operation
would land at the bandwidth bound, 17.36 tok/s short-context and 12.68 at 2k — still short of 20.

`tools/decode-ceiling.py` is that inequality as a gate. Three passes, all exit 1:

```
# today's kernel: 0.31 ops/term at the measured 0.77 IPC
python3 tools/decode-ceiling.py --bytes 1637275152 --terms 6945767424 --bw 28.42 \
  --ops-per-term 0.31 --cores 4 --ghz 2.3 --ipc 0.77 --target 20
#   issue ceiling 3.29 tok/s, bandwidth 17.36: refuse
# the same body at IPC 2
#   issue ceiling 8.55 tok/s: refuse
# an ideal 8-term i32 dot, 0.125 ops/term at IPC 2
#   issue ceiling 21.19, bandwidth 17.36: refuse
```

The last row is the result: even a perfect kernel clears the issue bound and still fails the byte
bound. No kernel-only design reaches 20 tok/s on this pack.

## The readers

`--kernel q8act` against `--kernel exact`, 4 threads, 6 tokens:

```
q8act  prefill 7.559 s (3.84 tok/s)   decode 1.698 s, 3.53 tok/s
exact  prefill 20.020 s (1.45 tok/s)  decode 7.650 s, 0.78 tok/s
```

The Q8 byte selector (`CodeTerms`, `code_select_avx2`) is admitted on this pack and is **4.5×**
the exact `i64` reader. Its body is 32 terms per 9–11 vector operations, 0.31 ops/term; at the
measured 303.75 ms that is 7.1e9 ops/s over four cores, **0.77 IPC**. The remaining kernel headroom
is instruction-level parallelism and dispatch, not a different algorithm: at IPC 2 the same body is
about 2.5×, to ~8 tok/s, and at the measured bandwidth it stops at 17.36.

Thread scaling, same run: 1 → 851.63 ms, 2 → 462.46 ms (1.84×), 4 → 303.75 ms (2.80×),
8 → 655.12 ms with `pool.wait` at 66.2%. The pool is correctly sized at four; SMT doubles the
dispatch cost. `pool.wait` at four threads is 9.7%, not the 49.3% the loaded trace showed.

## Byte budget for 20 tok/s

`20 tok/s` at 28.42 GB/s allows **1,421,000,000 B/token**. Current and candidate rows, 2k context:

| configuration | bytes/token | ceiling |
|---|---:|---:|
| today (Q8_0 head, F32 KV) | 2.241 GB | 12.68 |
| Q6_K head | 2.091 GB | 13.59 |
| Q5_K head | 2.008 GB | 14.15 |
| Q5_K head + 4-bit KV | 1.479 GB | 19.21 |
| 1.00 b/w pack + Q6_K head + 4-bit KV | 1.454 GB | 19.54 |
| **1.00 b/w pack + Q5_K head + 4-bit KV** | **1.372 GB** | **20.72** |

Only the last row clears 20 at 2k, and by 3.6%. The head alone cannot do it; the rate target and
the KV width are part of the same inequality. At short context the pack plus a Q5_K head
(1.404 GB) gives 20.24. The 4-bit KV row is a 1.52× on the current row on its own
(12.68 → 19.21), so the KV carrier is the larger single lever at 2k, ahead of the head.

The 1.00 b/w pack is Q3's target, so the rate work and the speed target are the same work. The
4-bit KV is C1's carrier.

## What this does not cover

- One pack, one machine, governor `powersave`, a shared box. The 28.42 GB/s mean is 25 passes on an
  idle machine, not the server's access pattern under load.
- The three-run mean is 6 decode steps per run at empty prefix; it is not a 2k-context number.
- `phase_split` reports the 630 ms trace under load; only the three-run mean is used as the
  baseline, and the phase shares are from the loaded trace.
- The head change is a quality change. Q5_K against Q6_K on the same token stream is not measured,
  and Q2's same-head clause would need the reference re-run at the same head.
- The 4-bit KV row assumes the hot/cold split in C1; the hot ring's bytes are not priced here.

## Next, in cost order

1. **Dispatch and ILP** (no format change): the selector at 0.77 IPC to ~2 IPC is up to 2.5×, to
   ~8 tok/s. `pool.wait` is already 9.7% at four threads; do not add threads.
2. **Bytes**: the head carrier and the KV width, against the 1.421 GB budget. This is the only path
   past 17.36.
3. **Rate**: 1.00 b/w is both Q3 and the last 3.6% of the 20 tok/s margin.
