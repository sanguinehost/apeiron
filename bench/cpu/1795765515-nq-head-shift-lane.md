# Q8_0 head row kernel: per-block activation scale as a lane shift

Criterion: `GOAL.md` T1 decode rate (0.5× llama.cpp), Q1. Grade: `measured` on the OVH
box (8 vCPU EPYC Milan, 1 thread for the kernel bench, `APEIRON_THREADS=8` for the
engine). The laptop (i5-10210U) pair is open: `MemAvailable` is below the 2 GiB
guard, so no timed laptop row exists in this receipt.

## Result

The head verify pass was the decode token's largest single term after the lookup
rewrite: `1.793 ms/token` (27.7%) at `6.445 ms/token`. The new row kernel forms each
block term as `(m·dot) << (k_b − k_min + e_b − e_min)` in four `i64` lanes and applies
one shift per column at the end, replacing the per-block `i128` multiply-and-add
epilogue. It is bit-exact with the per-block path.

| quantity | before | after |
|---|---|---|
| `lookup_accept` wall, 3 runs | 6.415–6.485 ms/token | 5.995–6.024 ms/token |
| `--trace` run | 6.445 ms/token | 5.874 ms/token |
| `head` phase, 29 calls | 1.793 ms/token (27.7%) | 1.171 ms/token (19.9%) |
| greedy wall, 3 runs | 15.974–16.205 ms/token | 16.068–16.256 ms/token |

Decode target is `5.46 ms/token` (half of llama's `10.92`). `5.995–6.024` is
`1.10–1.13×` short; the head is no longer the dominant term. Phase split after the
change: `ffn 2.242 ms/token (38.2%)`, `head 1.171 (19.9%)`, `attn.proj_in 0.935
(15.9%)`, `attn.core 0.654 (11.1%)`, `attn.proj_out 0.450 (7.7%)`, `pool.wait 0.402
(6.8%)`, `norm 0.168 (2.9%)`, `residual 0.072 (1.2%)`, `embed 0.007 (0.1%)`,
`untraced 0.053 (0.9%)`; `pass 4.582 (78.0%)`. The next lever is `ffn`, not the head.

## Kernel bench

`bench/cpu/1795769200-head-kcols.rs`, standalone `rustc -O` with
`RUSTFLAGS="-C target-cpu=native"`, one thread, `k = 7`, `151936 × 1024` Q8_0 rows,
per-block activation scales `2^(28 + b mod 3)` (the head's `quant_act` shape):

| body | ms | cycles/unit | exact |
|---|---|---|---|
| `dyn` (shipped per-block `i128`) | 76.956 | 32.6 | yes (reference) |
| `afold` (uniform-scale fold) | 70.541 | 29.8 | no, per-block scales |
| `afold64` (uniform-scale lane fold) | 26.154 | 11.1 | no, per-block scales |
| `shift` (per-block scalar shifts, `i128`) | 71.375 | 30.2 | yes |
| `shlane` (per-block lane shifts) | 34.212 | 14.5 | yes |
| `dyn1` (`k = 1`, memory-bound floor) | 15.037 | 6.4 | — |

`shlane` is `2.25×` `dyn` at `k = 7`. `afold64` is faster (`11.1`) but needs one
activation scale for the whole row; the head's quantizer is per-block, so it does not
apply and was refused. `shift` (scalar shifts, no lanes) is neutral, so the epilogue is
latency-bound on the dependency chain, not on the multiplier port; the lane epilogue
removes the chain, which is the win.

## Exactness

A finite `f16` weight scale is exactly `d = m·2^k` with `|m| ≤ 2047` and
`k ∈ [8, 37]` (`kern::pow2::f16_split`), a `Q8_0` block dot obeys `|dot| ≤ 32·128·127 <
2^19`, so `|m·dot| < 2^30`. With `quant_act`'s `S_b = 2^{e_b}`, a block term is
`(m·dot) << (k_b + e_b)`. The kernel shifts by `k_b − k_min` and `e_b − e_min`, keeps
the sum in four `i64` lanes (each lane accumulates `blocks/4` terms), and applies
`2^{k_min + e_min}` once per column. The lane bound is
`30 + spread + ⌈log₂ blocks⌉ ≤ 63`; `ShiftPlan` sets `limit = 33 − ⌈log₂ blocks⌉`
(`28` at 32 blocks) and `admits(c, row_span)` refuses a row whose
`row_span + spread[c]` exceeds it. A refused row runs the per-block path, which
accumulates into the same zeroed `acc`; the lane body writes `acc` only after every
column is formed, so a refusal cannot leave a partial sum.

Checks: `cargo test --release --locked --lib chart` (57 passed, 7 ignored) and
`cargo test --release --locked --lib kern::matmat` (10 passed, 5 ignored) on the box;
the k-column contract tests compare every column against per-column `matvec_q8act`
through this path. `lookup_accept` prints `exact` and accepts 102/102 drafts.

## Scope

- Decode only, `k ≤ 8` columns (`LANE_COLS`); prefill's wide calls stay on the
  per-block path.
- Box timings are CPU-tier evidence, not laptop evidence. The box is AVX2 without
  AVX-VNNI, the same instruction set the laptop has.
- The kernel ABI, LLVM codegen and the `unsafe` container are outside the claim; the
  safe twin is the per-block path and the contract tests.
- The head tensor stays bit-exact with the `Q8_0` weights: the stream is unchanged
  (102/102 accepted). The refused per-row-scale head (`a6cba2fb`) remains refused.

## Files

- `src/engine/chart/mod.rs`: `ShiftPlan`, `q8_0_row_kcols_lanes{,_k}`,
  `q8_0_row_kcols_avx2(plan)`; the fallback's four terms now form a tree (same value,
  one add of dependency distance per group; `0.7%` decode, inside the spread, so it
  claims nothing).
- `src/engine/chart/x86.rs`: `dot32x4_codes_packed`, `dot32x4_signed`,
  `dot32_lanes_from_abs`.
- `src/engine/kern/matmat.rs`: `ShiftPlan::new(qs, geom.blocks)` per call,
  `PooledRowsCtx.plan`, `pooled_rows(plan)`, kernel dispatch.
- `bench/cpu/1795769200-head-kcols.rs`: the table above.
