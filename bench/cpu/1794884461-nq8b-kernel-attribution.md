# Qwen3-8B at 1.124 b/w: the 1.607× gap is the pack, not the q8 reader

Criterion **Q2**. Grade: `measured` for every PPL and tok/s below; the instruction-exactness
argument is `model`, and the ratio interpolation in **Rate** is `model`, not a measurement. The
result is a **miss**: the pack is **1.0711×** the 1.5× bar at ctx 2048, and the exact reader does not
move it.

## Result

The same pack and tokenizer, one binary, two readers. `--kernel q8act` quantizes each stage
activation to int8 codes with one power-of-two scale per 32; `--kernel exact` uses the Q32.32 words.

| ctx | scored | q8act PPL | exact PPL | exact − q8act | q8act mean nll | exact mean nll |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1020 | 14.879110 | 14.903740 | **+0.024630** (+0.166%) | 2.699958 | 2.701612 |
| 2048 | 1023 | 12.189530 | 12.187527 | **−0.002003** (−0.016%) | 2.500577 | 2.500413 |

The exact reader changes sign with context and is at most 0.17% in magnitude. At ctx 2048 the
q8act reader costs **0.002003 PPL**, 0.016% of the PPL and 0.25% of the 6.6% gap. The bar is
11.380718; exact scores 12.187527, **1.0709×** the bar, and q8act scores 12.189530, **1.0711×**.
The reader is not the lever.

The Q4_K_M denominator 7.587145 is from `1794879503-nq8b-1124-ctx2048.md`, same stream and
`--second-half` protocol, a prior binary/box. The q8act ctx-2048 arm here reproduces that receipt's
NQ numerator bit-for-bit (12.189530 on 1,023 positions), so the ratio carries the same 0.2%
harness-revision uncertainty and not more.

## The assembly

`CodeTerms::new` and `quant_act` feed `code_select_avx2`; the shipped loop is nine vector
instructions per 32 terms (addresses from `target/release/examples/ppl`, sha256 `6db1fc27…`):

```
vpbroadcastd  ymm6, [word]        ; the 32 sign bits
vpsrlvd       ymm6, ymm6, [0,4,..,28]
vpshufb       ymm6, ymm6, repeat
vpand         ymm6, ymm6, 0x08040201
vpshufb       ymm6, select_one, ymm6
vpmaddubsw    ymm6, ymm6, [codes] ; 0/1 × int8, pairs into i16
vpmaddwd      ymm6, ymm6, ones    ; pairs into i32
vpsllvd       ymm6, ymm6, [shift] ; 2^(e_b − e_min)
vpaddd        ymm0, ymm6, ymm0
```

Every arithmetic instruction is exact on the admitted domain:

- `vpmaddubsw` takes the selector as unsigned bytes 0/1 and the code as signed bytes
  `−127..=127`; a pair sums to at most 254 in absolute value, so the i16 saturating add never
  saturates.
- `vpmaddwd` sums two such i16 pairs, at most 508, into i32.
- `vpsllvd` is an exact left shift by the normalized block exponent; it is where the power-of-two
  scale **enters**, not where it rounds.
- `vpaddd` accumulates the blocks. `CodeTerms::new` admits a reader only when
  `508·Σ 2^{shifts} ≤ i32::MAX`, so no lane wraps; the scalar tail and the final `e_min` shift are
  checked against `i64`.
- The widening/reduction (`vpmovsxdq`, `vpaddq`, `vpshufd`) is exact.

The lossy step is the scalar `quant_act` before the kernel, not the vector loop:
`S_b = 2^⌈log₂ ⌈absmax_b/127⌉⌉`, and each element is rounded to the nearest integer code in
`−127..=127`. The power-of-two choice wastes up to one bit per block: the largest code is in
`63..=127`, never always 127. That is the whole “warping” mechanism. It is a 6–7-bit code grid, not
an instruction that rounds, saturates or wraps. The A/B puts a bound on what removing it can buy:
at most 0.17%, with the sign depending on ctx.

## The other half of Q2

The upstream repository's `quality.json` for this exact GGUF (sha256 `75d3f4f1…`) reports 0-shot
accuracies against the dense base: piqa 0.800→0.535, arc_easy 0.805→0.400, arc_challenge
0.535→0.340, hellaswag 0.635→0.405, winogrande 0.705→0.570, boolq 0.855→0.660. That is the
upstream harness, not this one, so it is a pointer, not our measurement; it says the pack is far
from the base on the 0-shot half even where PPL is 1.6×. A 1.5× PPL point at 1.18 b/w would still
need a 0-shot check.

## Rate

The measured rate–quality points are:

| pack | b/w | ctx | PPL | ratio | receipt |
|---|---:|---:|---:|---:|---|
| r0 v2 | 1.124186 | 2048 | 12.189530 | 1.6066 | `1794879503-nq8b-1124-ctx2048.md` |
| r0 v2, exact | 1.124186 | 2048 | 12.187527 | 1.6063 | this receipt |
| r0 v2 | 1.124186 | 128 | 28.307562 | 1.7555 | `1794653685-nq8b-residual-ppl.md` |
| r1 v3, 1 residual bit | 2.127413 | 128 | 51.153269 | 3.1724 | same |
| r2 v3, 2 residual bits | 3.130640 | 128 | 16.295301 | 1.0106 | same |

The 1.124→1.18 band adds **0.055814 b/w**, 48,620,372 B on 6,945,767,424 weights. The r2 pack adds
**2.006454 b/w**, 1,742,045,354 B, to reach 1.0106×. The headroom is **2.8%** of that. A linear
interpolation between the ctx-128 endpoints (a `model`, not a measurement) puts 1.5× at about
**1.81 b/w**; between the ctx-2048 r0 point and the ctx-128 r2 point it is about **1.48 b/w**, but
those are not the same context. The r1 point is **worse** than r0 (3.17× vs 1.76× at ctx 128)
although it is 1.0 b/w above r0, so the real curve is not monotonic and neither interpolation is a
measured frontier.

The current pack is not rate-minimal. The source qkv is one fused linear; the converter writes
q/k/v as three independent sections and stores the same V payload in each. Layer 0's three V
payloads are byte-identical (sha256 `9497d1392a999830…`), 1,294,336 B each; across 36 layers the
duplicate is **90,505,216 B**, 9.27% of the pack. A shared-V section or reference would make the
factors 885,536,368 B (885,544,464 B with the header), **1.019953 b/w**, and the pair
2,215,170,160 B, 872,336 B from the 2,214,297,824 B source. The fused source also shares `s_pre`,
`s_mid` and the salient indices; deduping all four payloads is **91,450,016 B**, 884,599,664 B with
the header, **1.018865 b/w** (`bench/cpu/1794885786-nq8b-uv-signs.md`). The 1.18 cap would then have
**138,956,231 B** of headroom, not 48,620,372 B. That is still 12.5× short of the 1,742,045,354 B
r2 adds; it is room for a refit, not a quality gain by itself.

No allocation of 48.6 MB fixes a 6.6% PPL gap at this fit: the whole residual mechanism needs
1.74 GB to move r0 to 1.01×, and its first bit is negative. The gap is the weight approximation at
1.124 b/w, not the reader and not a removable instruction.

## Protocol

```
examples/ppl --model models/nq8b-r0/skeleton.gguf --pack models/nq8b-r0/model.nqp \
  --tokenizer models/tokenizer.json --data wiki.test.raw --tokens 2048 --skip 0 \
  --ctx <512|2048> --head-batch 16 --kernel <q8act|exact> --second-half
```

Host: vast.ai CPU instance `61002480`, Xeon E5-2680 v4, 28 cores, 31 GiB, Ubuntu 24.04,
`APEIRON_THREADS=24`; the instance is destroyed after the run. Raw wikitext-2 test, first 2,048
tokens, `--second-half` scores `ctx/2 .. ctx-2`: 1,020 positions at ctx 512 (4 windows) and 1,023
at ctx 2048 (1 window).

## Artifacts

| file | bytes | sha256 |
|---|---:|---|
| `qwen3-8b-nanoquant.gguf` (source) | 2,214,297,824 | `75d3f4f150195e0c6a7ccd2ce3372f919534fb7ec609f6b9ba52eb71af194253` |
| `model.nqp` | 976,049,680 | `1f4cc4b43cd60f70cf52e110884f96e6a2eb2539cd0add5b353cc699cd8d0bac` |
| `skeleton.gguf` | 1,329,625,696 | `3927454ca530ce26afdcd5dafe78f2d29d5ac006ea09151c8387d973a2ceaa28` |
| `tokenizer.json` | 4,790,085 | `9ac67dd678dec2bb53ba7ffcc4524d24fb50a70f3996e9db37d13e7552c16485` |
| `examples/ppl` (A/B binary) | 2,261,328 | `6db1fc279ea87577bed93a8e138c2709f304d2991dc4aabfcd867965bc32adb2` |

Logs: `bench/cpu/1794884461-nq8b-kernel-attribution-logs/nq-{512,2048}-{q8act,exact}.log`.

## What this does not cover

- One dataset (raw wikitext-2 test), one token stream, 2,048 tokens, one window at ctx 2048.
- The Q4_K_M denominator is from the prior receipt, not re-run on this box; the NQ numerator
  reproduces exactly, so the 1.0711× ratio is stable to the 0.2% harness shift.
- 0-shot accuracy (the other half of Q2) is not measured; no ≥8,192-token or 200k row.
- The pack is the upstream GPU pipeline's output, not `nq_fit`'s. Nothing here validates
  `learn::nanoquant`.
- The kernel is Rust; no extraction or Lean refinement is run here. No claim is `proved`.
- The 1.55 b/w interpolation is a two-point `model`; the r1 point already contradicts monotonicity.
