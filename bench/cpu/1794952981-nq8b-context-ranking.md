# Where the 8B pack ranks: OG NanoQuant, llama.cpp 4-bit, and the release line

Criterion **Q2** in context; **Q1**, **T1** and **M1** for the release verdict. Grade `measured` for
the ladder PPL, the sizes, the in-tree PPL and the 0-shot screen; `cited` for the OG NanoQuant paper
(arXiv 2602.06694v3, ICML 2026) and the published llama.cpp table; `model` for the served-size
arithmetic and the interpolation.

## 1. The measured ladder on Qwen3-8B

One rented host (Intel Xeon E5-2696 v3, 36 cores, 72 threads, 126 GB), llama.cpp
`307e3f1551c2c3c6c0560c454ab46bb64baeb2df`, five full 2,048-token chunks of `wiki.test.raw`,
`-c 2048 --chunks 5 -t 36 -ngl 0`. The six files are the bartowski Qwen3-8B base quants plus the
official bar:

| file | bytes | sha256 | PPL | / bartowski Q4 | / official Q4 |
|---|---:|---|---:|---:|---:|
| `Qwen_Qwen3-8B-Q2_K.gguf` | 3,281,733,152 | `e89189c2…` | 10.8663 | **1.1296** | 1.0944 |
| `Qwen_Qwen3-8B-Q3_K_M.gguf` | 4,124,161,568 | `c2c61b55…` | 9.9451 | **1.0338** | 1.0016 |
| `Qwen_Qwen3-8B-Q4_K_M.gguf` | 5,027,784,224 | `54fffa05…` | 9.6202 | 1.0000 | — |
| `Qwen_Qwen3-8B-IQ4_XS.gguf` | 4,561,839,648 | `0f69fe02…` | 9.7463 | **1.0131** | — |
| `Qwen_Qwen3-8B-Q5_K_M.gguf` | 5,851,112,992 | `fa45032e…` | 9.5883 | **1.0033** | — |
| `Qwen3-8B-Q4_K_M.gguf` (official bar) | 5,027,783,488 | `d98cdcbd…` | 9.9290 | — | 1.0000 |

Two facts fall out. **IQ4_XS is not the 4-bit frontier on this model**: 1.0131× Q4_K_M, not better
than it; llama.cpp's own LLaMA-3-8B scoreboard has IQ4_XS at 1.036 and Q4_K_M at 1.024, the opposite
order. And **the official bar is 3.2 % worse in PPL than a standard Q4_K_M of the same checkpoint**
(9.9290 against 9.6202, both 5,027,78x,xxx B). The audit shows the bar carries the AWQ norm fold
(`1794921128-nq8b-tensor-audit.md`); the 3.2 % is a property of the two files on the same harness,
not of the protocol. Every ratio quoted against the bar in §4 is therefore bar-dependent; the
bartowski-anchored column is the standard-Q4 column.

The in-tree `ppl` on the same official file and the same five full windows pools to **10.717139**,
7.9 % above llama.cpp's 9.9290 on the same text. Absolute PPL is therefore not comparable across
the two harnesses; every cross-method statement below is a ratio, and the ratio's harness is named.

## 2. OG NanoQuant, published

Qwen3-8B-Base, their WikiText-2 protocol, Table 2 and Table 3 of arXiv 2602.06694v3:

| linear bits | served size | PPL | / BF16 | 0-shot six-task | / BF16 |
|---:|---:|---:|---:|---:|---:|
| 16.00 (BF16) | 16.38 GB | 7.00 | 1.00 | 71.22 | 1.00 |
| **1.00** | **3.35 GB** | **12.47** | **1.78** | **48.94** | **0.687** |
| 0.80 | — | 14.83 | 2.12 | — | — |
| 0.55 | — | 20.04 | 2.86 | — | — |

The 3.35 GB is not 1.00 b/w over all weights: it is 0.87 GB of 1-bit linears plus a **BF16
embedding and head** (the tied 151,936 × 4,096 pair is 2.49 GB). Their BPW is a linear-only rate, the
same denominator ours uses.

## 3. The llama.cpp published anchors

The repository's own scoreboard (`tools/perplexity/README.md` at master, revision `f364eb6f`, CUDA,
RTX 4090, LLaMA-3-8B, WikiText-2 test set), F16 **6.233160**:

| quant | imatrix | size GiB | PPL | / F16 | / q4_K_M |
|---|---|---:|---:|---:|---:|
| q6_K | — | 6.14 | 6.253382 | 1.0032 | 0.9797 |
| q4_K_M | WT 10m | 4.58 | 6.382937 | 1.0240 | 1.0000 |
| q4_K_M | — | 4.58 | 6.407115 | 1.0279 | 1.0038 |
| iq4_XS | WT 10m | 4.14 | 6.459705 | 1.0363 | 1.0120 |
| q3_K_M | WT 10m | 3.74 | 6.734255 | 1.0804 | 1.0550 |
| q2_K | WT 10m | 2.96 | 8.647825 | 1.3874 | 1.3548 |
| q2_K | — | 2.96 | 9.751568 | 1.5645 | 1.5220 |

Two readings. The **IQ4_XS/Q4_K_M inversion reproduces**: IQ4_XS is 1.012× Q4_K_M here and 1.013×
on our Qwen3 ladder, so it is a property of the quantisers on these models, not of our harness. And
**our 1.464× Q4 sits between LLaMA-3-8B's Q2_K with an importance matrix (1.355×) and without one
(1.522×)**, at 1.18 b/w and 1.89 GB: the pack's quality is that of a 2-bit K-quant on an 8B, at
58 % of Qwen3's own Q2_K storage. The measured Qwen3 ladder reproduces the shape throughout
(Q2_K 1.130, Q3_K_M 1.034, Q4_K_M 1.000, IQ4_XS 1.013, Q5_K_M 0.997); the model is not unusually
sensitive.

## 4. Our point

**1.178785 b/w**, pack 1,023,456,080 B, served **1,891,196,336 B = 1.76 GiB** with the Q4_K
embedding and Q6_K head. Q2 PPL **1.417631** [1.355945, 1.482123] against the official bar over 17
windows (58 % of the file); 0-shot **0.7534×** Q4.

| artifact | served bytes | PPL / standard Q4 | 0-shot / Q4 |
|---|---:|---:|---:|
| Q5_K_M | 5.85 GB | 1.003 | — |
| Q4_K_M (bartowski) | 5.03 GB | 1.000 | — |
| IQ4_XS | 4.56 GB | 1.013 | — |
| Q3_K_M | 4.12 GB | 1.034 | — |
| Q2_K | 3.28 GB | 1.130 | — |
| NQ 1.1788 (this pack) | **1.89 GB** | **1.464** | **0.7534** |
| OG NQ 1.00 | 3.35 GB | 1.78 (vs BF16) | 0.687 (vs BF16) |

Read the rows together, not the ranks: at 1.89 GB the pack is **1.74× smaller than Q2_K for 3.6×
its PPL excess** (0.464 against 0.130 over 1.0), and it is the only measured point below 2 GB. The
2.66× saving against a 4-bit quant costs 1.46× PPL and 25 % of the six-task accuracy. At equal
embedding/head precision the pack would be ~3.5 GB, essentially OG's 3.35 GB.

**The OG ratio comparison is not protocol-matched and is withdrawn as a win.** OG's 1.78× is
`12.47 / 7.00` under `SamsungLabs/NanoQuant`'s `evaluate_ppl`, whose BF16 Qwen3-8B-Base scores 7.00;
this checkpoint's near-lossless image in our harness is Q8_0, which at ctx 128 scores 10.676 against
Q4_K_M's 10.988 (`1794553293-nq8b-q4km-ppl.md`). The paper's BF16 is therefore ~25–35 % below the
same checkpoint's near-lossless PPL here, so the two denominators are different protocols, not two
measurements of one quantity. Converting our 1.464× Q4 through a BF16 distance is not licensed until
a Q8_0 (or BF16) arm is run on the same windows and span. The measured facts are: **1.464× Q4 at
1.18 b/w** here, and **1.78× BF16 at 1.00 b/w** there; the ordering between them is **unresolved**.
**It is the strongest measured sub-1.2-bit Qwen3-8B artifact here; whether it beats OG's published
point is the open measurement, not a conclusion.**

The bar correction is measured, not assumed. On the same five second-half windows and the same
binary, bartowski's Q4_K_M pools to **9.604552** against the official bar's **9.919400**, a factor
**0.96826**; the standard-Q4 ratio is therefore **1.4641**, not 1.4176. Scaling the 17-window
interval by that factor gives an upper end of **≈1.531**, so the 1.5× pass holds against the named
official bar and its upper end crosses the bar against a standard Q4_K_M. That is the bar the next
receipt has to quote, and the one the advertisement has to use.

## 5. Release verdict

| criterion | state | number | gap |
|---|---|---|---|
| Q1 rate | **pass** | 1.178785 b/w, 1,044,615 B headroom | — |
| Q2 PPL | **pass against the named official bar; bar-dependent** | 1.417631 [1.355945, 1.482123]; **1.4641** against a standard Q4_K_M, upper end ≈1.531 | 1.15× stretch is 1.24–1.27× away |
| Q2 0-shot | **fail** | 0.7534 against 0.90 | −0.1466 absolute, 19.6 % relative |
| T1 decode | **fail** | 2.884 tok/s at 2k on the i5-10210U | 10.4× short of 30, 8.3× short of 24 |
| M1 memory | **open** | 2,470 MiB RSS at 2k; no 200k allocation | 200k KV is 1.4746e10 values (36 layers × 8 KV heads × 128 × K,V); f32 59 GB, f16 29.5 GB, the 1 GiB cap 0.582 bits/value |
| T2 turn | **untested** | no 200k prefill | — |
| C1/C2 | **open** | classic f16 KV is the serving path | no replacement scored at 200k |
| I1/I2 | **partial/open** | one 25-token turn | no 20-task suite, no local/DeepSeek split |
| V1–V4 | **open** | Vulkan does not read the pack | no 7B end-to-end, no refinement proof |
| O1 | **partial** | Apache-2.0, builds from source | the 8B fit is not reproducible from a clone |

**Not releasable as a product.** The blocking pair is 0-shot (0.75 against 0.90) and decode (2.9
against 30). The 200k context claim has no measurement behind it at all: the 8B KV at 200k is 59 GB
f32 or 29.5 GB f16, and the C1 cap needs 0.58 bits/value. The pack is a credible research artifact
and the best measured sub-1.2-bit Qwen3-8B point.

## 6. What can and cannot be advertised

Truthful: **“Qwen3-8B at 1.18 b/w linear, 1.9 GB served, 1.42× the official Q4_K_M PPL (1.46× a
standard Q4_K_M), 0.75× its six-task 0-shot, CPU-only, 2.9 tok/s at 2k on a 4-core laptop.”**

Not: “Q4 quality” (1.4× PPL, 0.75× 0-shot), “20 tok/s” (2.9), “200k context” (untested, 59 GB of
f32 KV), “3× faster decode” (the 1-bit CPU path is instruction-bound; the older pack ran 4.0 tok/s
against Q4_K_M's 8.6 on the same EPYC), or **“beats NanoQuant”**: OG's 1.78× BF16 is a different
harness and no matched run exists. The earlier 0.27–0.32 “better” was a conversion through an
unmeasured BF16 distance and is withdrawn.

## 7. What moves it

The fix path in `1794948388-nq8b-q2-zeroshot-gap.md` is still the one with a budget: dedupe to
1.0735 b/w, add the 3.39 MB per-channel bias and a rank-16 residual to 1.1740 b/w, and gate on boolq
≥ 0.792 and winogrande ≥ 0.684 before the six-task screen. If that gate fails, the 0.90 bar at
1.18 b/w is above the published sub-1.2-bit frontier (OG's own 1-bit point is 0.687×) and the bar,
not the fit, is the thing to renegotiate.

For speed, the ceiling is already computed: 20 tok/s needs 190.6 G terms/s and 42.76 GB/s at 2k,
1.002× the spec peak, so it is refused at this rate and ISA, not deferred. For 200k, C1 is the
gate; no weight-only result changes the 59 GB f32 (29.5 GB f16) KV.

## 8. What this does not cover

The ladder is one host, five full chunks, llama.cpp; the NQ ratio is 17 second-half windows in this
tree. The 3.2 % bar difference is measured on five chunks and applied to the 17-window ratio as a
model. The OG 0-shot comparison is cross-harness: their per-task metrics mix `acc` and `acc_norm`
(HellaSwag 58.81 is `acc`, ours is `acc_norm` 74), so the 0.687/0.753 ordering is an anchor, not a
paired test. The llama.cpp scoreboard is LLaMA-3-8B and is cited, not measured here. The OG
comparison is **cross-protocol**: their BF16 7.00 is ~25–35 % below this checkpoint's near-lossless
Q8_0 in our harness, so the 1.78 vs 1.464 ordering is unresolved until Q8_0 and the candidate are
scored on the same windows and span (`bench/PROTOCOL.md`). rustc and LLVM codegen, the kernel ABI and
the `unsafe` container are outside the claim.

## 9. Artifacts

`bench/cpu/1794952981-nq8b-context-ranking-logs/`:

| file | content |
|---|---|
| `sha`, `sizes` | sha256 and byte count of all six GGUF files |
| `llama-commit` | `307e3f1551c2c3c6c0560c454ab46bb64baeb2df` |
| `ppl-Qwen_Qwen3-8B-{Q2_K,Q3_K_M,Q4_K_M,IQ4_XS,Q5_K_M}.log` | llama.cpp five-chunk PPL, one per file |
| `ppl-official-Q4_K_M.log` | the same for the official bar, 9.9290 |
| `apeiron-q4-full-{0..8192}.log` | in-tree `ppl` on the official bar, five full windows, pooled 10.717139 |
| `apeiron-bart-q4-half-{0..8192}.log` | in-tree `ppl` on bartowski Q4_K_M, five second-half windows, pooled 9.604552 |
| `apeiron-done`, `bart-done` | completion markers for the two in-tree loops |

Host: Intel Xeon E5-2696 v3, 72 visible threads, 36 used (`-t 36`, `APEIRON_THREADS=36`), 126 GB.
Commands: `llama-perplexity -m <file> -f wiki.test.raw -c 2048 --chunks 5 -t 36 -ngl 0`; in-tree
`target/release/examples/ppl --model <gguf> --tokenizer tokenizer.json --data wiki.test.raw
--tokens 2048 --skip <s> --ctx 2048 --head-batch 16 [--second-half]`. The ladder files are the six
listed in §1; the pack and skeleton were never on the box (the copy was killed when the ladder
finished), so the NQ arm is the 17-window line from `1794937300-nq8b-q2-rented-host.md`, not a
re-run here.
