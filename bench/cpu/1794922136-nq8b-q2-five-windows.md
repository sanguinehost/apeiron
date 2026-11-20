# Q2's PPL half over five windows: the pack sits on the bar, and two windows could not have said so

Criterion **Q2**. Grade `measured`: both arms on this laptop, one protocol, five skips. The claim this
receipt withdraws is "the PPL half holds on two same-host windows"; the claim it establishes is the
pooled estimate over 5,115 scored positions and the window-level interval around it.

## The five windows

Protocol, identical on both arms: `ppl --tokens 2048 --skip <0|2048|4096|6144|8192> --ctx 2048
--head-batch 16 --second-half`, 4 threads, `MemoryMax=3600M`. Each window scores **1,023 positions**
(the second half of a 2,048-token chunk, as `llama.cpp` does). NQ arm:
`nq8b-late-v4/model.nqp` (1,023,456,080 B, 1.178785 b/w) on `skeleton-q6k-q4kembed.gguf`. Q4 arm:
`Qwen3-8B-Q4_K_M.gguf` (5,027,783,488 B, sha `d98cdcbd…c5785`). Same tokenizer, same text file.

| window | skip | NQ PPL | Q4_K_M PPL | ratio | margin to 1.5× |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 11.374839 | 7.587145 | 1.499226 | +0.005879 (+0.0517%) |
| 2 | 2048 | 17.361910 | 11.834074 | 1.467112 | +0.389201 (+2.1925%) |
| 3 | 4096 | 17.226503 | 10.462647 | **1.646477** | **−1.532533 (−9.7922%)** |
| 4 | 6144 | 14.057420 | 9.354884 | **1.502682** | **−0.025094 (−0.1786%)** |
| 5 | 8192 | 15.236382 | 10.927844 | 1.394272 | +1.155384 (+7.5755%) |

Three windows pass, two fail. The per-window ratio spans **1.394271 … 1.646477**, a spread of
**0.252206** — 17% of the bar.

## The pooled estimate

Wikitext-2 perplexity is a mean over scored positions, so the dataset-level estimate is
`exp(Σ nll_i · n_i / Σ n_i)` over the windows, not a mean of the windows' PPLs. With equal counts it is
the mean of the five printed `mean nll` values (`tools/ppl-pooled.py`):

| quantity | value |
|---|---:|
| pooled NQ nll | 2.699794 |
| pooled Q4_K_M nll | 2.294492 |
| pooled gap | 0.405301 nats |
| pooled NQ PPL | 14.876661 |
| pooled Q4_K_M PPL | 9.919400 |
| **pooled ratio** | **1.499754** |
| margin to the 1.5 bar | **+0.000246 PPL (+0.0164%)** |
| scored positions | 5,115 over 5 windows |

The pooled ratio is **0.016% under the bar**. It is not a pass: the window-level 95% interval on the
gap is `0.405301 ± 0.074794` nats (sd 0.060247, t 2.776, df 4), i.e. a ratio interval of
**[1.391674, 1.616228]**. The bar is inside it.

**The interval is text variance, not run noise.** Three repeats with the identical command reproduce
the first run to the last digit (`run13`, `determinism` lines in `status.txt`): NQ window 1
11.374839 → 11.374839, Q4 window 1 7.587145 → 7.587145, Q4 window 3 10.462647 → 10.462647, every
delta **0.000000**. The projections are integer, the head sum is f64, and the pool's reduction order
is stable across runs, so a window's ratio is a property of the text. More windows are the only way
to narrow the interval; more repeats are not.

## Why the two-window claim was wrong

The earlier statement — 1.499225 on window 1 and 1.467112 on window 2, "the PPL half holds on two
same-host windows" — is true of those two windows and says nothing about wikitext-2. The per-window
gap varies with sd 0.060 nats while the margin being claimed was 0.0059 PPL = **0.0005 nats**. Two
windows cannot resolve a 0.0005-nat margin when one window moves the gap by 0.06. The first two
windows happened to be two of the three that pass; windows 3 and 4, measured later, fail.

The same arithmetic retroactively bounds window 1's "pass": a 0.0517% margin is 1/116 of the
window-to-window spread. It is a measurement of that window, not evidence about the dataset.

## What would decide it

The window is the unit of the gap's variance: sd 0.060247 nats over 1,023-position windows. To put
the 95% half-width below the 0.0005 nats that separate window 1's estimate from the bar needs
`(1.96 × 0.060247 / 0.0005)² ≈ 55,800` windows — 57 M scored positions, ~14,000 h on this laptop.
Even a 0.005-nat half-width (0.5% in ratio) needs ~558 windows, 139 h. An earlier draft of this
paragraph said 516 windows; that is the count for a 0.0052-nat half-width, a factor of ten coarser
than the margin, and it is corrected here. **No feasible sample of this pack decides a margin this
small.** The route is not more sampling; it is a fit that moves the gap.

At 25 windows the half-width is ±0.025 nats, so the point estimate can be quoted to ±2.5% of itself.
That bounds what a better fit has to buy: the gap must fall from 0.405 to **≤0.380 nats** (a 6.3%
reduction in the pack's added nll) before a 25-window run can place the bar outside its interval.
The queue `bench/overnight/run14.sh` ran windows 6–25 for that bound: 20 windows spaced 11,776
tokens apart from skip 10,240 to 233,984 across the whole test file. The span is systematic, not
contiguous, because window 6 (skip 10,240) measures NQ PPL **5.450666** against 11–17 on windows 1–5:
the first 10 k tokens are much harder than the next 2 k, and a contiguous block would inherit that
difficulty. The laptop completed 17 of the 25 pairs (through skip 139,776) before memory pressure
made a Q4 window wait an hour for 3.6 GiB; it also ran window 6 (5.450666) alone. The remaining
windows and the clean same-host 25-window line run on the rented host of
`bench/cpu/1794937300-nq8b-q2-rented-host.md`. The five base windows are a 10 k-token block (4% of the file) but carry 20% of the
sample weight; even if the start's gap is 0.025 nats above the rest, that over-weight biases the
pooled gap by at most 0.004 nats, well inside the ±0.025-nat interval the span quotes.

## What this does not cover

The full test file (about 120 windows); the 0-shot half of Q2; whether the pack's added nll is a
function of text difficulty — the five gaps (0.405, 0.383, 0.499, 0.407, 0.332) do not order with the
Q4 arm's nll (2.026, 2.471, 2.348, 2.236, 2.391), so no difficulty model is fitted. The runs are repeats:
`run13` reran windows 1 and 3 on both arms and every delta is 0.000000, so the four-degree interval is
text variance. rustc/LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.
