# The partial column group takes the tile

Criterion: **T1** decode rate and **T2** turn latency, through the verify width. Grade: `measured`
on the OVH box (CPU tier, 8 vCPU EPYC Milan); the laptop pair is still blocked
(`MemAvailable` ~0.9 GiB against the 2 GiB guard), so this is box evidence and the ratios are
model.

## Result

The wide tile now runs on a **last group of 2–7 live columns** instead of dropping to the
two-column pair kernel; a single leftover column keeps the pair loop. Both fold stages take it
(`v_stage`, `u_stage`). Paired, interleaved, `APEIRON_THREADS=8`:

| shape | before, ms/token | after, ms/token | ratio |
|---|---:|---:|---:|
| k = 8 (band 1, control) | 3.27–3.39 | 3.30–3.39 | flat |
| k = 9 (band 2) | 4.71 | 3.81 | **1.24×** |
| k = 16 (band 2) | 2.94 | 2.43 | **1.21×** |

Decode wall, interleaved, same session:

| threads | before | after |
|---|---:|---:|
| 8 | 5.822–5.874 ms/token | 5.818–5.969 |
| 2 | 15.437–16.071 | 15.434–15.515 |

Exactness: `lookup_accept` reported `exact`, 102/102, at both widths; `engine::kern` 88 passed.
Decode at width 8 is unchanged because k ≈ 4.6 gives `band = 1`, where the tile already ran whole
groups; the ~2% at width 2 is the band-3 shape. This is a **kernel-rate step on band-2/3 shapes**,
not a decode-wall win.

## The tile beats the pair kernel at every width

`the_tile_beats_the_pair_kernel_at_a_verify_width` (ignored timing test), the `down` `V` fold,
736 rows × 3,072 bits, one thread, 60 passes, MACs/cycle at 3 GHz:

| k | tile | pair | per-column | tile/pair |
|---:|---:|---:|---:|---:|
| 2 | 24.51 | 16.60 | 12.70 | 1.48× |
| 3 | 30.79 | 15.39 | 12.74 | 2.00× |
| 4 | 33.26 | 16.66 | 12.74 | 2.00× |
| 5 | 35.82 | 15.87 | 12.59 | 2.26× |
| 6 | 36.97 | 16.09 | 12.46 | 2.30× |
| 7 | 37.99 | 16.10 | 12.39 | 2.36× |
| 8 | 37.20 | 16.69 | 12.45 | 2.23× |

A 2..=7-column partial group is admissible for the same reason an 8-column one is: `Stage::cols`
admits the same column references and `dot_row` bounds-checks `n ≤ 8`. The guard `ncols - base >=
2` leaves one column to the pair loop, which is where the pair kernel is still the right shape.

Two supporting tests are committed: `the_narrow_tile_serves_five_columns` (5-column tile equals the
per-column dots on both `V` and `U` geometries) and `an_empty_dispatch_costs_what` (an empty
8-task dispatch is **1.44 µs**, so the extra partial group does not explain a wall movement).

## Scope

- Box only; no laptop row. Timed claims name the laptop and this receipt does not make one.
- k=9/k=16 are `prefill_width` shapes (full layer pass, no head): a verify-width rate, not a
  decode wall. The decode wall is flat at width 8 and ~2% at width 2.
- Prefill 0.5× is 2,080 ms against 3,645: this step is 1.2× on the k=9 shape and does not close it.
- Rust/LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.
