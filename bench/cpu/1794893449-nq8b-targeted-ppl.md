# The 1.18 b/w residual: 1.598× at ctx 2048, 1.554× if spent uniformly, 1.067× for the full 3.13 b/w pack

Criterion: **Q2** for the quality and **Q1** for the rate. Grade: `measured` for every PPL and byte
count. Result: **not met at ≤1.18 b/w**. The qkv dedup frees 0.161 b/w; the 1.5× bar needs **0.337
b/w**, so the cap funds **48%**. The full two-plane residual at 2.006 b/w reaches **1.067×** at
ctx 2048, so the quality mechanism works and the rate is the only gap. The 19-section targeted pack
measures **1.598×**, 6.1× less efficient than a uniform spend of the same bytes.

## Artifacts

| file | bytes | sha256 |
|---|---:|---|
| `qwen3-8b-nanoquant.gguf` (source) | 2,214,297,824 | `75d3f4f150195e0c6a7ccd2ce3372f919534fb7ec609f6b9ba52eb71af194253` |
| `qwen3-8b-q4_k_m.gguf` (bar) | 5,027,783,488 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| `model.nqp` v2 (base) | 976,049,680 | `1f4cc4b43cd60f70cf52e110884f96e6a2eb2539cd0add5b353cc699cd8d0bac` |
| `model.nqp` v3, 2 planes everywhere | 2,718,095,872 | `9c7cd2a88b396f2f7a0e2ca52ec9c67350ecad9b7c5bf590062c2a7037f81823` |
| `model.nqp` targeted, 19 sections | 1,115,003,392 | `7f578b43de713be9121e38edd8668f61484b62df1bc1aadc832fb0657d9bf405` |
| `skeleton.gguf` | 1,329,625,696 | `3927454ca530ce26afdcd5dafe78f2d29d5ac006ea09151c8387d973a2ceaa28` |
| `tokenizer.json` | 4,790,085 | `9ac67dd678dec2bb53ba7ffcc4524d24fb50a70f3996e9db37d13e7552c16485` |

The targeted pack is `tools/nq_splice.py --base nq8b/model.nqp --residual nq8b-v3s-r2/model.nqp`.
The uniform residual is per-section and independent, so a pack that serves two planes on a subset is
a byte-level splice of the v2 and v3 packs: unselected sections come from v2 with `res_layers = 0`,
selected sections come from v3 with `res_layers = 2`. No factor is refit, no dense reference is read,
and every section is byte-identical to its source (`252/252` checked).

## Selection and rate

The ranker is the v2 `sal` field: `W_sal − (U diag(s_mid) V diag(s_pre))_sal` at the `k = 5..13`
columns the upstream pipeline marks as important. `rows·k·mean(sal²)` is the fit error sampled where
the activations are largest; the greedy takes sections by that mass per residual byte until the byte
budget.

| quantity | value |
|---|---:|
| selected sections | 19 / 252 |
| fitted weights with two planes | 7.9710% |
| salient-residual mass on those sections | 26.0641% |
| lift | 3.27× |
| residual payload | 138,952,704 B |
| pack | 1,115,003,392 B, **1.284229 b/w** |
| qkv duplicate in that pack | 91,450,016 B |
| pack after the dedup | 1,023,553,376 B, **1.178909 b/w** |

The budget is the 1.18 cap plus the dedup: `1,024,500,695 − 976,049,680 − 1,008 + 91,450,016 =
139,900,023 B`. The splice spends 138,952,704 B, leaving 947,319 B. The dedup is **not implemented**;
1.178909 b/w is the rate the same weights will store once the qkv group lands. PPL does not depend on
the layout, so the measurement below is the deduped pack's PPL.

## Protocol

Laptop `yoga`, Intel i5-10210U (4 cores, 8 threads), 6,747 MiB RAM, `APEIRON_THREADS=4`. The base
and targeted runs used `-p MemoryMax=3200M -p MemorySwapMax=0`; the 2.718 GB r2 pack needs
`-p MemoryMax=5000M -p MemorySwapMax=1G` and peaks at 3,697 MiB. The `ppl` example is built from the
current tree (`cargo build --release --locked --example ppl`, 2026-11-17 16:00). The command is

```
target/release/examples/ppl --model skeleton.gguf --pack <pack> --tokenizer tokenizer.json \
  --data wiki.test.raw --tokens 2048 --skip 0 --ctx 2048 --head-batch 16 \
  --kernel q8act --second-half
```

`--second-half` scores positions 1024..2046 of the first 2,048 tokens of raw wikitext-2 test, 1,023
positions, the protocol of `1794879503-nq8b-1124-ctx2048.md`. The v2 control reproduces that
receipt's box number to six decimals, so the laptop runs are the same measurement.

## Results

| pack | b/w | mean nll | PPL | PPL / Q4_K_M | tok/s | peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| v2 base | 1.124186 | 2.500577 | 12.189530 | 1.606603 | 1.4 | 2252.3 MiB |
| targeted, 19 sections | 1.284229 | 2.495234 | **12.124565** | **1.598041** | 1.3 | 2563.3 MiB |
| v3, 2 planes everywhere | 3.130640 | 2.091134 | **8.094085** | **1.066816** | 0.8 | 3697.0 MiB |
| Q4_K_M, official | ~4.62 | 2.026427 | 7.587145 | 1.000000 | — | — |

The bar is 11.380718. The targeted pack is 0.743847 above it, **6.54%**; the base is 0.808812 above,
**7.11%**; the full residual pack is **1.067×**, inside the bar by 40% and at 80.5% of Q4_K_M's
bytes. The targeted gain is **0.064965 PPL**, 0.005344 nats/token, **8.03% of the 0.808812 gap** for
**7.97%** of the weights.

The 63-token screen (`--tokens 128 --ctx 128 --second-half`, 63 positions) is **9.257322** for v2 and
**9.307707** for the targeted pack: the targeted pack is 0.54% *worse* on that window, the opposite
sign to the 1,023-token result. Both effects are below 1% and the window decides the sign; neither is
a quality claim.

## What the residual is worth per byte

| context | scored | r0 nll | r2 nll | Δnll | Δnll / b/w | Δratio / b/w |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1008 | 3.3427 | 2.7909 | 0.5518 | 0.2750 | 0.3713 |
| 2048 | 1023 | 2.5006 | 2.0911 | 0.4094 | **0.2041** | **0.2690** |

The two-plane residual keeps **74%** of its nats-per-byte value from ctx 128 to ctx 2048: the
mechanism is close to context-stable, and the earlier 0.353–0.371 ratio/bw model was not wrong about
the mechanism, only about which body of weights the 0.161 b/w would buy.

The bar needs `ln(12.189530 / 11.380718) = 0.068657` nats. At 0.2041 nats/bw that is **0.3364 b/w**
of two-plane residual, a pack at **1.4606 b/w**. The cap's 0.161 b/w is **47.8%** of the need; the
shortfall is 0.176 b/w. A uniform 8% spend at the cap's 0.160043 b/w gives 0.03266 nats,
**11.797864 PPL, 1.554981×**.

The 19-section *targeted* spend gives 0.005344 nats, **1.31%** of the full residual's 0.4094 at
7.971% coverage: **0.164× of proportional**. The salient-residual mass is anti-predictive — 26.1% of
that mass bought 1.3% of the gain — and a selection would need **2.10×** the uniform value per byte
to reach the bar inside the cap's 8%.

## What this means for Q2

- **The stock-file PPL comparison is met at 3.13 b/w, but Q2 is not**: 8.094085 against 7.587145,
  1.067×, at ctx 2048, on 1,023 scored tokens. The full residual pack is inside the 1.5× bar by 40%
  at 80.5% of Q4_K_M's bytes. The heads differ (`Q8_0` on the NQ skeleton, `Q6_K` on the stock
  file), so this is not the same-head comparison Q2 requires. The 0-shot half, the same-head run and
  the 1.18 rate are still open.
- **The 1.18 b/w product is 1.55–1.60×**: 1.554981 with a uniform 8% spend, 1.598041 with the
  salient-mass selection. The cap is 0.176 b/w short of the bar, not a rounding error.
- **The dedup is 48% of the Q2 need** (0.161 of 0.336 b/w) and a Q1/T1 lever (0.105 b/w, 9.5% of the
  sign terms). It makes the 1.598 pack shippable at 1.178909 b/w; it does not close Q2.
- **The next quality lever is the selection or the fit.** A selection with 2.10× the uniform value
  per byte would reach 1.5× inside the cap; the salient-residual mass is 0.164×, so the metric, not
  the byte count, is the immediate work. If no 8% selection reaches 2.10×, the fit must use the same
  bytes better, or the rate must rise to ~1.46 b/w.

## What this does not cover

- The qkv dedup's format and engine group: 1.178909 b/w is the arithmetic of removing the duplicate,
  not a built pack. The dedup is bit-exact by construction; its own test is not written here.
- The 8% selection space. One ranker is measured; a uniform 8% is modelled at 1.554981, not run, and
  the 2.10× lift a better selection needs is unproven.
- The ctx-2048 0-shot half of Q2 (piqa, arc_easy) is untouched; the upstream `quality.json` figures
  remain a pointer.
- One dataset, one tokenizer, one context. The residual is fit against the official Q8_0 image, so it
  absorbs that image's 8-bit quantisation noise.
- rustc and LLVM code generation, the kernel ABI, the GPU path and the `unsafe` container. No claim
  here is `proved`.

## Logs

`bench/cpu/1794893449-nq8b-targeted-ppl-logs/`: `ppl-v2-ctx2048.log`,
`ppl-targeted-ctx2048.log`, `ppl-r2-ctx2048.log`, `ppl-63-nq8b.log`, `ppl-63-nq8b-targeted.log`,
`targeted-sections.csv` (per-section mass, residual bytes, selection flag) and `sha256.txt`.
