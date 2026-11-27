# LM-head carriers: `Q4_K` is refused for decode, `Q6_K` is quality-neutral but its pooled lane diverges

Criterion: **T1** decode rate; quality screening against the pack's PPL. Grade: `measured` on the
OVH box for walls, PPL and exactness; `tested` for the carrier equality tests the engine already
runs. The laptop pair is open; this is CPU-tier box evidence.

## Result

Three head images were built from the same tied F16 `token_embd.weight` and run against the same
pack and stream. **`Q4_K` is refused**: it halves the head bytes but does not move the wall, and
it costs PPL. **`Q6_K` is accepted on quality and bytes** (237.52 against 238.50 PPL, 23% fewer
bytes) but its multi-column lane disagrees with its single-column row: the decode stream diverges.
That is the open defect, and it is an engine bug rather than a tool bug.

| head | bytes | PPL (2,048 tok, ctx 2,048) | in-loop wall, draft 8, ms/token | head phase |
|---|---:|---:|---:|---:|
| `Q8_0` (shipped) | 165,306,368 | 238.498293 | 5.796 / 5.706 / 5.918 (mean 5.807) | 1.144 (19.8%) |
| `Q4_K` (9 per-column passes) | 87,515,136 | 244.134601 (+2.4%) | 5.747 / 5.744 / 5.812 (5.768) | 1.064 (19.0%) |
| `Q4_K` (fused, 1 pass) | 87,515,136 | — | 5.807 / 5.897 / 5.818 (5.841) | 1.105 (19.4%) |
| `Q6_K` | 127,626,240 | 237.524050 (−0.4%) | exactness fails, wall not taken | — |

The three `Q4_K` walls sit inside the Q8_0 arm's spread (5.706–5.918), so the carrier change is
not a wall change. The isolated head call says why: at `k = 1` both carriers cost **2.47 ms**;
the 165 MB Q8_0 read is at the box's memory floor (~67 GB/s) while the 87.5 MB Q4_K read is
compute-bound at only 35 GB/s. Halving the bytes does not halve the call, and at the decode
widths (`k ≈ 5`) the Q4_K lane's per-MAC cost offsets its byte saving.

## The `Q6_K` divergence

The control binary (unmodified `6cfb8eda`, whose `head_argmax_cols` already routes every carrier
through `matmat_q8act`) runs the greedy stream and the lookup stream over the same head:

```
Q6_K: lookup ids differ from greedy: 128 ids, first mismatch at Some(21)
```

`Q4_K` is exact on the same two streams (102/102). The greedy stream projects one column
(`matvec_rows_chunked` → `q8act_q6_k_row`), the lookup verification projects `k ≈ 5`
(`matmat_q8act` → the pooled lane). The two kernels read the same 210-byte blocks and must be
bit-equal; they are not.

The block layout is not the cause. ggml's `block_q6_K` at `049326a` is
`ql[128], qh[64], scales[16] (int8), d (f16)`, and the repo's `quant.rs` (`q6_k_codes`,
`q6_k_scales`, `q6_k_d`) and this tool's encoder agree with it; `head_cmp` against the Q8_0 head
reports a 2.0% max logit deviation with the same argmax, which the `Exact` per-column path
produces from those blocks. The suspect is the pooled lane: `Q6KRow::fill` or the packed panel's
signed scale spread (`q6_k_row_lanes_avx2` reads `sc.kd[b]` and `sc.bytes`; the `i128` fallback
`q6_k_row_kcols_avx2` sign-extends the 16 scale bytes explicitly and is the reference for the
lane). Q6_K's scales are **signed**, unlike Q4_K's unsigned 6-bit scales; a zero-extending fill
would corrupt exactly the blocks whose signed max is negative. Filed as the next task; the head
stays `Q8_0` until the lane is exact.

## Artifacts

| artifact | detail |
|---|---|
| `tools/nq_head_k.py` | Q4_K/Q6_K reference quantizers (ports of `quantize_row_q4_K_ref` / `quantize_row_q6_K_ref`, float32) and the GGUF head writer |
| `examples/head_bench.rs` | the head call in isolation, `--cols 1/9`, best of 20 |
| `skeleton-q4khead.gguf` | 404,881,664 B, `output.weight` `Q4_K` 87,515,136 B |
| `skeleton-q6khead.gguf` | 444,992,768 B, `output.weight` `Q6_K` 127,626,240 B |

```sh
python3 tools/nq_head_k.py --in skeleton-q8head.gguf --out skeleton-q4khead.gguf --type q4_k
python3 tools/nq_head_k.py --in skeleton-q8head.gguf --out skeleton-q6khead.gguf --type q6_k
APEIRON_THREADS=8 target/release/examples/head_bench skeleton-q4khead.gguf --cols 9 --reps 20
APEIRON_THREADS=8 target/release/examples/ppl --model skeleton-q4khead.gguf \
  --pack nq06-v3.nqp --tokenizer tokenizer.json --data wiki.test.raw \
  --tokens 2048 --ctx 2048 --head-batch 16 --kernel q8act
```

Isolated head, best of 20: `Q8_0` k=1 2.494 ms, k=9 12.100 ms; `Q4_K` k=1 2.473 ms, k=9
8.108 ms. The k=9 figures overstate the in-loop cost because each call allocates `k × 151,936`
`i64` logits (10.9 MB at k=9) and the allocator returns them to the OS; the in-loop head is
1.10–1.14 ms/token.

## Scope

- Box only; no laptop row and no laptop claim.
- The `Q4_K` head is refused: no wall move, +2.4% PPL. The experiment that routed `Q4_K` through
  the fused head in `head_cols_kernel` was reverted; `head_argmax_cols` is the decode path and it
  already fuses every carrier, so the routing change was inert on the measured path.
- The `Q6_K` divergence is observed, not diagnosed. No code change; the lane is not on the
  current product path (the pack is sign-rank, the head is `Q8_0`), but it is a correctness defect
  in a carrier used by `Q4_K_M` images and is the next task.
- The `Q6_K` PPL row is a head-image quality measurement on one 2,048-token window, not a model
  result; it does not become a product claim.
- Rust/LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.
