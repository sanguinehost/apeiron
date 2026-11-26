# nq: eight-row decode projection blocks — 1.135× decode, a kernel step

Criterion: **T1** (decode). This receipt records a kernel step on the rank-stage reader, not a
campaign speedup: the change targets the `U`/`V` projection stages, whose traced share is
21.2% (`proj_in` 10.5% + `proj_out` 7.4% + part of `ffn`), while the dominant term is `attn.core`
at 44.6%. `phase_split --target ffn` exits 1 on the unmodified tree; the refusal is quoted below
and the next lever is `attn.core`.

The measured result: default-carrier (F32 KV) decode **4.175 → 3.679 s** per 128 tokens,
30.67 → **34.80 tok/s** (1.135×, 3/3 paired rounds). The i8 carrier is unchanged
(54.65 → 54.60 tok/s, 0.999×). The first decoded token is identical on both carriers, so the row
tile is a scheduling change, not a values change.

Grade: `measured` for the walls, rates, split and binary hashes; `model` for the ratios, the
`1/(1 − s)` ceilings and the bytes-per-token arithmetic; `tested` for the blocked reader's
equality (`engine::kern::nq::tests`, 29 passed) and the workspace suite. Nothing is `proved`.

## Machine, binaries and command

Machine: OVH box `ubuntu@139.99.223.161` (8 vCPU EPYC Milan, 32 GiB), `APEIRON_THREADS=8`,
`~/bin/adg`. This is CPU-tier evidence; the timed target is the laptop (i5-10210U), which did not
run this comparison.

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, version 3, 0.977441 b/w,
53,808,384 B); tokenizer `tokenizer.json`.

| arm | binary | sha256 |
|---|---|---|
| O, committed tree `0a81876b` | `prefix_solve.unified2` | `8a740f0fee047101e1aebf96fd344b7c6bd171309053acb788ae66b7c31518ad` |
| N, this change | `prefix_solve.rows` | `6605f74c8b53e2fdb5ee29959edd89bfcdca0f9f9f228c2b1cc6d84e2fa4c4a6` |
| N4, four-row trial | `prefix_solve.rows4` | `8aeaa17341316ac3f2054acdde569ee42f55b6273a0000ffc4a253becb9f41f5` |

Command: `prefix_solve --len 1827 --differ 0 --decode 128 --mb 512`, untraced, three rounds
interleaved O/N on each carrier. Traced split: `phase_split --steps 32 --prefix 1795`. Raw evidence
in the `.log` beside this file.

## Paired decode

| carrier | O, s | N, s | O, tok/s | N, tok/s | N / O time | rounds N wins |
|---|---:|---:|---:|---:|---:|---:|
| F32 KV | 4.1747 | **3.6787** | 30.67 | **34.80** | **0.8810** | 3/3 |
| i8 KV | 2.3427 | 2.3443 | 54.65 | 54.60 | 1.0007 | 1/3 |

The F32 arm's per-round wall is 4.233/4.183/4.108 (O) against 3.649/3.692/3.695 (N); the i8 arm is
2.333/2.334/2.361 against 2.385/2.343/2.305. The i8 ratios (0.978–1.022) straddle 1, so the i8
carrier is neutral, not a regression: its `attn.core` (26.5%) and `head` (16.6%) carry more of the
token and the projection stages are a smaller share.

Prefill is unaffected (3.937 s O against 3.905 s N mean, inside the run spread): the blocked path
is decode's `k = 1` entry only.

## The change

`NqLinear::project_with`'s `V` loop and `project_rows`'s `U` loop now call
`Stage::dot_rows` in runs of `NQ_ROW_BLOCK = 8` instead of `Stage::dot` per row. `dot_rows`
dispatches to `code_select_block_avx2::<8>`: one load of the 32-lane code block and one load of the
block's shift vector serve every row in the run, the selector constants are built once per block,
and each row keeps its own accumulator chain and the same 5-op select. The eight-row entry is the
Q8 reader's existing row maximum (`NQ_BLOCK_MAX`), previously exercised by prefill's wider `k`; the
decode single-column path was the last caller on the per-row entry, whose assembly shows one serial
dependent chain per block behind a `call` with seven pushes per output row.

Four rows measured 3.735 s against eight's 3.679 s (1.5% slower, three rounds), so the tile stays
eight: the accumulators fit beside the five selector constants without a spill that costs more than
the extra loop turn saves.

Scale words are unchanged: the `V` run folds `rank` terms of the same `codes`/`scales` pair and the
`U` run folds `rank` terms, so the term count per token is the same **418,099,200** sign terms
(28 layers × (2048·640 q + 1024·480 k + 1024·480 v + 2048·640 o + 2·3072·736 gate/up +
1024·736 down)) and the pack read is still 53,808,384 B.

## Traced split, F32, before and after

| term | before, ms | share | after, ms | share |
|---|---:|---:|---:|---:|
| `attn.core` | 12.497 | 43.0% | 12.198 | 44.6% |
| `ffn` | 7.320 | 25.2% | 6.286 | 23.0% |
| `head` | 3.139 | 10.8% | 3.256 | 11.9% |
| `attn.proj_in` | 3.045 | 10.5% | 2.920 | 10.7% |
| `attn.proj_out` | 2.160 | 7.4% | 1.842 | 6.7% |
| wall | 29.05 | | 27.37 | |

The three rank-projection leaves fall 12.525 → 11.048 ms (−1.477, −11.8%); the untraced wall falls
1.68 ms/token, matching the paired wall difference (4.1747 − 3.6787)/128. `attn.core` is now
44.6% with a 1.81× ceiling.

## Dominant-term gate

On the unmodified tree with the change's target named:

```text
REFUSED: target ffn holds 23.0% (ceiling 1.30x); the dominant term is attn.core at 44.6%
(ceiling 1.81x). Target the dominant term, or instrument it if it is the untraced remainder.
exit 1
```

The refusal is the receipt's scope: a 1.135× step on a 21% share is capped at 1/0.79 = 1.27× by
the pipeline even if the reader became free, and this change takes only 12% of that leaf. The
campaign number does not move to 0.5× from here; `attn.core` owns the token.

## Ceilings and the campaign target

Decode bytes/token at F32 KV: 638,184,704 B (pack 53.8 MB + dense skeleton + 229,376 B/position
KV read); at the measured 100.74 GB/s STREAM mean that is **157.8 tok/s**, against the 0.5× decode
target **183.1 tok/s** (5.46 ms/token). The 34.80 tok/s measured is 22.1% of the F32 ceiling and
4.5× below the target; an F16 KV carrier moves the bytes to 428.6 MB/token and the ceiling to
**235.0 tok/s**, which is the precondition for the target, not this change.

## Refusals

1. **"This is a campaign speedup."** Refused: the target phase (`ffn`/rank stages) holds 21.2% and
   the gate exits 1; the 1.135× is inside the leaf's own 1.27× pipeline cap.
2. **"It helps the i8 carrier."** Refused: 0.999×, with per-round ratios straddling 1; i8 is
   neutral.
3. **"It reaches 0.5× on decode."** Refused: 34.80 against 183.1 tok/s is 5.3× short, and the
   F32-KV bandwidth ceiling (157.8) is below the target before any kernel is timed.
4. **"The assembly claim is measured for the blocked entry."** Not claimed: the per-row
   `code_select_avx2` disassembly is on disk (`/tmp/llvm/ours-decode.s` on the box); the blocked
   entry's instruction count is read from the source and is `model`. The wall is the evidence.
5. **"Laptop timing."** Refused: this ran on the OVH box; it is CPU-tier evidence, and no laptop
   wall is claimed.

## What this does not cover

The laptop's wall, the GPU path, the `unsafe` container (no `unsafe` block is added or changed;
`code_select_block_avx2` and its twin already exist), and quality: the first token is identical,
but no PPL run is made here because the arithmetic is unchanged and the first-token equality is the
check that the row tile did not reorder the sum.
