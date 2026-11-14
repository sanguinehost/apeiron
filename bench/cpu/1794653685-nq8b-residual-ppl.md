# Dense sign residual on the 8B NanoQuant pack: perplexity

Rented vast.ai CPU instance `60602495`: Intel Xeon E5-2697 v4, 62 GB RAM, Ubuntu 24.04.5. The
container exposes 72 logical CPUs of the host; the host load average was 35–55 while the full runs
were in flight, so tok/s is descriptive and not a throughput claim. The source tree was rsynced
from the laptop and the harness was built on the instance with `cargo build --release --locked
--example ppl`. Every run below is that binary. `APEIRON_THREADS` is 10 for each residual full run,
4 for the Q4_K_M full run, and 2–12 for the first-window and bar runs.

## Artifacts

| file | bytes | sha256 |
|---|---:|---|
| `qwen3-8b-nanoquant.gguf` (source) | 2,214,297,824 | `75d3f4f150195e0c6a7ccd2ce3372f919534fb7ec609f6b9ba52eb71af194253` |
| `qwen3-8b-q8_0.gguf` (official dense reference) | 8,709,518,112 | `408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6` |
| `qwen3-8b-q4_k_m.gguf` (official bar) | 5,027,783,488 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| `skeleton.gguf` (all four packs) | 1,329,625,696 | `3927454ca530ce26afdcd5dafe78f2d29d5ac006ea09151c8387d973a2ceaa28` |
| `model.nqp` v2 (r0) | 976,049,680 | `1f4cc4b43cd60f70cf52e110884f96e6a2eb2539cd0add5b353cc699cd8d0bac` |
| `model.nqp` v3 refit, 1 residual | 1,847,073,280 | `f6b48309561877ec47d41df5ba7f185bbdd1680b541f8904e4bf4567ead985b1` |
| `model.nqp` v3 refit, 2 residual | 2,718,095,872 | `9c7cd2a88b396f2f7a0e2ca52ec9c67350ecad9b7c5bf590062c2a7037f81823` |
| `tokenizer.json` | | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

The v3 packs were produced by `tools/nq_convert.py --source qwen3-8b-nanoquant.gguf --dense-ref
qwen3-8b-q8_0.gguf --residual N`; the no-refit counterexamples are the same command without the
salient refit in that run. The converter reports 252 linears. The engine reports 1,847,064,176
factor bytes (2.127413 b/w) for r1 and 2,718,086,768 (3.130640 b/w) for r2, both on 6,945,767,424
fitted weights. The skeleton is byte-identical to the v2 pair, so v3 changes only the pack.

The earlier context-128 bar, 10.988133, was `qwen3-8b-base-q4_k_m.gguf`, sha256 `194c2d00…`, a
different checkpoint (`Qwen3-8B-Base`). The NQ source is the official instruct checkpoint: its
`output.weight` is byte-identical to `qwen3-8b-q8_0.gguf`, and the token embedding differs only in
the 636 special-token rows. The bar below is therefore re-measured on official
`Qwen3-8B-Q4_K_M.gguf` with the same harness and protocol. The file is 32 B smaller than the base
image and scores 16.124694, not 10.988133.

## Protocol

Raw wikitext-2 test file, Qwen3 tokenizer, consecutive non-overlapping windows, `--second-half`
scores positions `ctx/2 .. ctx-2`; `Q8Act` trunk and head, exact packed linears. The first window
is the first 128 tokens (`--tokens 128`), 63 scored positions; the full run is the first 2048
tokens (`--tokens 2048`), 1008 scored positions. The command is

```
target/release/examples/ppl --model <skeleton.gguf> [--pack <model.nqp>] \
  --tokenizer tokenizer.json --data wiki.test.raw \
  --tokens <128|2048> --skip 0 --ctx 128 --second-half
```

## Results

First window, 63 scored positions:

| config | b/w on linears | served bytes | % of Q4_K_M | mean nll | PPL | PPL / Q4_K_M |
|---|---:|---:|---:|---:|---:|---:|
| r0, v2 | 1.124186 | 2,305,675,376 | 45.9% | 2.236423 | 9.359791 | 1.857 |
| r1, v3 no refit | 2.127413 | 3,176,698,976 | 63.2% | 14.135396 | 1,376,970.012849 | 273,173 |
| r1, v3 salient refit | 2.127413 | 3,176,698,976 | 63.2% | 2.649433 | 14.146016 | 2.806 |
| r2, v3 no refit | 3.130640 | 4,047,721,568 | 80.5% | 1.939287 | 6.953788 | 1.380 |
| r2, v3 salient refit | 3.130640 | 4,047,721,568 | 80.5% | 1.755296 | 5.785161 | 1.148 |
| Q4_K_M, official | — | 5,027,783,488 | 100% | 1.617538 | 5.040663 | 1.000 |

Full window, 1008 scored positions:

| config | mean nll | PPL | PPL / Q4_K_M | Δ nll vs Q4_K_M |
|---|---:|---:|---:|---:|
| r0, v2 | — | 28.307562 | 1.7555 | +0.5628 |
| r1, v3 salient refit | 3.934826 | 51.153269 | 3.1724 | +1.1545 |
| r2, v3 salient refit | 2.790877 | 16.295301 | 1.0106 | +0.0105 |
| Q4_K_M, official | 2.780352 | 16.124694 | 1.0000 | 0 |

`r0` is the served v2 pack from `bench/cpu/1794553293-nq8b-q4km-ppl.md`; its artifacts are the two
files hashed above, and the current binary reproduces its first window at 9.359791. The target band
1.15–1.5× Q4_K_M is 18.543–24.187; r2 is 16.295, a 1.06% PPL excess over Q4_K_M, better than the
band's 15% lower edge. It is 0.980 GB (19.5%) smaller than the Q4_K_M image.

The 63-token first window is a screen, not a estimate of the full ratio: it puts r2 at 1.148× and
r1 at 2.806×, while the 1008-token row gives 1.011× and 3.172×. The ranking of r0 and r2 is stable;
the ratio is not.

## The sparse refit is mandatory

The pack stores `k = 5` (q/k/v/gate/up) or `13` (o/down) salient columns per row in f16 because
they dominate the loss. The sign residual adds `±α` at **every** column, including those, with
`α = mean|W − Ŵ|` per row. On the four sections sampled from the r1 pack, `α / mean|S|` is 4.98
(L0 gate), 6.87 (L0 down), 7.48 (L10 q) and 14.21 (L35 down). The first fit never undid that at
the sparse columns, so the one place the format keeps precise was moved by alpha; the r1 engine
run then measured 1.38M PPL on the first window and the Q8_0 bake 2.65M over 1008 tokens.

The version-3 fit refits the sparse values after the residual. Because the served product is
linear, `S = W_sal − (U·V)_sal − R_sal` restores the dense value at those columns. Decoding the r1
and r2 packs and comparing `(U·V)_sal + S + R_sal` with the official Q8_0 weights on L0 gate rows
0–7 gives a maximum error of 2.72e-5 (r1) and 2.67e-5 (r2), 0.13% of mean |W| = 0.02031. Before
the refit the same comparison is alpha-level, 5–14× the salient magnitude.

The refit is what makes the first layer finite; it does not make one layer good. Same engine, same
window: r1 no refit 1,376,970 → r1 refit 14.146, r2 no refit 6.954 → r2 refit 5.785. The refit
buys 1.168 PPL on r2 and five orders of magnitude on r1.

## One layer is worse than none; two layers recover

After the refit, r1 is 51.153 against r0's 28.308 over 1008 tokens, although its raw Frobenius
error is lower (L0 gate rel 0.2846 vs 0.4684; down 0.3354 vs 0.5506). The closed-form step is the
sign of the unweighted residual with a per-row mean scale; it is not aligned with the
calibration-weighted metric the served factor minimises. The first residual bit costs 0.592
nats/token. The second step fits the remaining residual and recovers: L0 gate rel 0.1730, down
0.2043, full PPL 16.295, 0.552 nats/token better than r0. The useful allocation is two residual
bits, not one; a one-bit section on this fit has negative value.

## Rate

A residual layer is 1 bit per fitted weight plus one bf16 row scale: **871,022,592 B** of factor
payload (868,220,928 B of signs and 2,801,664 B of row scales). Two layers add 1,742,045,184 B of
factors; the r2 pack is 1,742,046,192 B larger than r0, 34.6% of Q4_K_M's size.

| pack | linears | factor bytes | b/w | pack + skeleton | % of Q4_K_M | full PPL |
|---|---:|---:|---:|---:|---:|---:|
| r0 v2 | 252 | 976,041,584 | 1.124186 | 2,305,675,376 | 45.9% | 28.307562 |
| r1 v3 | 252 | 1,847,064,176 | 2.127413 | 3,176,698,976 | 63.2% | 51.153269 |
| r2 v3 | 252 | 2,718,086,768 | 3.130640 | 4,047,721,568 | 80.5% | 16.295301 |
| Q4_K_M | — | — | ~4.62 | 5,027,783,488 | 100% | 16.124694 |

r2 reaches Q4_K_M within 1.06% at 80.5% of its bytes. Including the 1.330 GB skeleton, r2 stores the
8,709,518,112 weights at 3.718 b/w, against Q4_K_M's 4.618 b/w: 0.90 b/w cheaper overall at a 1.06%
PPL excess. The last 0.980 GB of Q4_K_M buys 0.0105 nats/token over r2, while r0's 2.306 GB leaves
it 0.5628 nats/token behind Q4_K_M. In this cell the two residual layers are the efficient part of
the budget, not the base factor.

## Cross-checks

- The Q8_0 bake and the engine agree where both were run: r0 first 9.332351 vs 9.359791, full
  28.294362 vs 28.307562; r2-no-refit first 7.007880 vs 6.953788. The two implementations share
  the sign, scale and refit formulas but not the storage path.
- The bake reproduces the r1 catastrophe over 1008 tokens (2,646,480.188754), so it is not a Q8_0
  requantisation artifact.
- The kernel mode is not the cause: on the Q8_0 bake of the unrefit r1 image, Exact is 911,627 and
  Q8Act 1.38M on the first window; the engine v3 pack with Q8Act is 1,376,970.
- The 147 non-linear tensors are byte-identical across the r0, r1 and r2 baked images
  (`tools/nq_graft.py`, 0 differing hashes), so the bake isolates the linear fit.

## What this does not cover

- One dataset (raw wikitext-2 test), one tokenizer, one context length. The target is stated at
  ctx 128; ctx 512 was not re-run against the corrected bar.
- The residual is fit against the official Q8_0 weights, so it absorbs that image's 8-bit
  quantisation noise as if it were dense error. The fit is unweighted; a calibration-weighted
  refit or a teacher is a different objective and is not measured.
- The residual is two independent binary planes (2 b/w). A 2-bit four-level residual at the same
  2 b/w has not been fitted; it may do better or worse than two sign steps.
- The kernel is Rust; the extraction and the Lean refinement theorem for `src/engine/kern/nq.rs`
  are not run here (no pinned Aeneas/Charon toolchain). No claim in this receipt is `proved`.
- rustc and LLVM code generation, the kernel ABI, the GPU path and the `unsafe` container are
  outside the measurement. The wasm32 crate compiles against the v3 engine, but no v3 pack has
  been loaded there.

## Logs

The raw logs are in `bench/cpu/1794653685-nq8b-residual-ppl-logs/` (`ppl-v3s-r1-full.log`,
`ppl-v3s-r2-full.log`, `ppl-q4km-full.log`, `ppl-q4km-63.log`, the first-window and no-refit logs,
the bake logs, and `convert-v3s.log`).
