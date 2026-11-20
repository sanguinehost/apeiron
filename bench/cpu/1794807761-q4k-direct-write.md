# `Q4_K` folds into the shared loop: direct-write band

Criterion: **T1** (decode/one path), **T2** context. Grade: `measured` for the paired wall, `tested`
for equality. Host: ThinkPad L13 Yoga, i5-10210U (4C/8T), 6.6 GiB, `performance`, no AVX-VNNI,
measured 27.97 GB/s. This receipt advances T1/T2's kernel path; it does not close either.

## Change

`matmat::pooled` staged its result as one interleaved `n_out × k` `i64` buffer, ran the row bands
into it, then copied it into `k` column `Vec`s, and refused any call with `n_out·k > 2^22`. The bands
now write each row value straight into its column through `chart::pool::PBand`/`for_pbands`; the
staging buffer, the transpose loop and `MAX_POOLED_WORDS` are deleted. `pooled_rows` is a
`BandTask<i64, ()>` and the `Q4_K` packed lane keeps one `k`-word row per band task before pushing
it. The 7B FFN up-projection (`18,944 × 3,584`, `k = 256`, `n_out·k = 4.85M`) now takes the shared
loop; the old staging cap refused that shape inside `pooled` (`gemm_q4_k` covered it before the
fold).

The `gemm_q4_k` pre-dispatch is deleted and `gemm_q4_k`/`Q4KRows`/`Q4_K_FAST_TOP` are removed from
`chart::kcols`; its admission (narrow bound, `k = 1`, `n_in % 256`) was a subset of the shared
carrier's (wide bound, `k ≤ 4096`, `n_in % 256`, same `pack_cols` power-of-two rule). `Q4_K` calls
above `MAX_POOLED_K = 4096`, and `k = 1`, take the per-column oracle; the production prefill block is
256 and every production `matmat_q8act` call has `k > 1`.

## Equality

- `pooled_q4_k_matches_per_column`: 3×512 and 2×1024 payloads, `k ∈ {2, 8, 33, 64}`, every column
  equals `matvec_q8act`. Exit 0.
- `q4_k_matmat_equals_the_per_column_contract` (retargeted from `gemm_q4_k`): `n_out = 7`,
  `n_in ∈ {256, 1024}`, `k ∈ {1, 3, 8, 9, 33}`, narrow and wide bound columns, non-finite scale,
  past-bound column, ragged 992-wide input. Exit 0.
- `q4_k_matmat_switches_to_the_i128_fold_past_the_lane_bound`: the lane takes `i64` at
  `kspread + spread = limit = 23` and the `i128` fold one past it; both equal the contract. Exit 0.

## Paired wall

Command, one release test process, three interleaved passes per arm, `n_out = n_in = 3584`,
best of 3, 3584-wide `Q4_K` payload:

```sh
agent-guard run -- cargo test -j 1 --release --lib --all-features --locked \
  -- --ignored --nocapture pooled_ > /tmp/opencode/q4k-direct.log
```

| `k` | before: pooled / gemm (ratio) | after the band, before the deletion (ratio) | payload pooled GB/s |
|---:|---:|---:|---:|
| 32 | 3.673 / 3.203 ms (1.15×) | **3.126 / 3.524 ms (0.89×)** | 2.31 |
| 128 | 15.861 / 13.118 ms (1.21×) | **12.666 / 13.407 ms (0.94×)** | 0.57 |
| 512 | 113.188 / 92.900 ms (1.22×) | **107.323 / 107.597 ms (1.00×)** | 0.07 |

The two runs are different processes on a shared laptop; the absolute walls moved (the `k = 512`
gemm arm is 92.9 ms before, 107.6 ms after) with load and package temperature. The ratio is the
comparable quantity because the arms are interleaved in the same process; the before/after rows were
not run interleaved against each other, so the 1.15–1.22× → 0.89–1.00× shift is the claim, not the
microseconds. The paired test that produced the "after" row was deleted with `gemm_q4_k`; its log is
`/tmp/opencode/q4k-direct.log` and the table is transcribed from it. The production-shape test
`q4_k_carrier_rate_at_production_shapes` replaces it and measures the surviving route.

## Production shapes

Same command, filter `q4_k_carrier_rate_at_production_shapes`; `pooled` and the production entry
`matmat_q8act` alternate order each round, best of 3. The `Q4_K` payload is random codes with
`d = 0x3800`, activation columns power-of-two scaled.

| `n_out` | `k` | pooled ms | `matmat_q8act` ms | m/p | payload GB/s |
|---:|---:|---:|---:|---:|---:|
| 3,584 | 32 | 3.826 | 3.426 | 0.90 | 2.11 |
| 3,584 | 128 | 15.302 | 13.122 | 0.86 | 0.55 |
| 3,584 | 512 | 107.142 | 111.650 | 1.04 | 0.06 |
| 18,944 | 256 | 275.696 | 285.911 | 1.04 | 0.13 |

The entry adds no systematic overhead over the carrier (0.86–1.04; the spread between runs is
larger than the difference). The 18,944-row row is the 7B FFN up-projection; `n_out·k = 4.85M` is
above the old staging cap.

Decode `k = 1` on the same 3,584² payload through `matvec_q8act`, best of 3: **2.893 ms**,
**4.4 G weights/s** single-call. The engine's `matvec_q8act_quant_chunked` splits rows over the
pool, so the four-thread projection is about 17.6 G weights/s, consistent with the 19.9 G weights/s
the 0.5B demo measures end to end (`bench/cpu/1794799742-decode-ceiling.md`). A 7B token at
1.174 b/w is 6.525G weights, so 20 tok/s needs **141 G weights/s**: 32× the single-call rate and
about 8× the four-thread rate. This carrier does not close that gap; the decode instruction count
does.

The decode step is the reason. `objdump` on the release test binary gives
`Q4_KDecoder::decode_step_avx2` **543 static instructions**; its full-step branch returns 8
32-lane groups (256 weights) and calls `q4_k_block_dots` once. At 4.4 G weights/s and 4 GHz the
core retires 1.1 weights/cycle, so the static stream implies about **2.3 IPC** at ~2.1
instructions/weight — the core is issuing near its limit, and the only route to 8.8 weights/cycle
is fewer instructions per weight, not a wider dot. The 4-bit `vpshufb` carrier is the candidate:
one shuffle per 16–32 weights replaces the scalar scale/min unpack and the eight `emit` calls.
Grade: `measured` for the static count, `model` for the per-weight and IPC arithmetic.

## Gate

On this tree, `cargo test --lib --all-features --locked`: **638 passed, 40 failed, 60 ignored**.
The 40 are the 39 known GPU/`glslc` cases plus `learn::nanoquant::tests::spd_inverse_matches_serial`
(`vkCmdWriteTimestamp2` absent), the same set as `bench/cpu/1794801248-g1-contributor-gate.md`;
every `Q4_K` equality test passes. `cargo clippy -j 1 --workspace --all-features --all-targets
--locked -- -D warnings` exit 0; `cargo fmt --all -- --check` exit 0; `tools/check-claims.py` exit
0 (34 claims); `tools/book-check.py` exit 0 (20 entries); `mdbook build book` exit 0.

## What remains

- `MAX_POOLED_K = 4096` leaves `Q4_K` calls above 4096 columns on the oracle. A production run that
  overrides `APEIRON_PREFILL_COLS` past 4096 is 5.8–9.6× slower per block than the shared kernel; no
  receipt measures that configuration. The default is 256.
- The 7B pack is not built; the FFN shape is measured on a synthetic payload.
- Decode (`k = 1`) does not take this band: `step::cols` calls `matvec_q8act_quant_chunked` per
  state, and the two-row tile in `q8act_rows` is the decode core. The decode rate above is the
  lever; this carrier is the prefill/multi-column route.
- `gemm_q6_k` is still a sibling in front of the shared loop; `Q6_K` does not yet have a carrier.
- The `Q4_K` packed lane has no Aeneas-verified safe twin; it is `tested` debt under the `## Unsafe`
  rule (`AGENTS.md`, `proof/README.md`).
- Not covered: rustc/LLVM codegen, the kernel ABI, AVX-VNNI (absent on this host), the GPU.
