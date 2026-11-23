# The ADMM sub-phase split

Status: `measured`. Machine: `nq-l4` (GCE L4, 23,034 MiB heap, 31 GB host), 2026-11-23. Tree:
`272397a5` plus the tape/frame release and the sub-phase counters. Criterion: Q3 campaign cost.
The split is the basis for the next speed target; it closes no criterion.

## The measurement

```sh
APEIRON_NQ_ADMM_TRACE=1 bash bench/gpu/campaign/trace-round.sh
```

One layer-0 round over all seven linears, 400 outer iterations each, inner 5, `rho` linear,
`reg = 3e-2`, seed 0, 128 samples at 2048 tokens. The run is rc 0, `round total 4587.4 s`
(`bench/gpu/1795396294-8b-sharing/logs/trace-round.log:189`). Each `NQ admm` line is one factorisation's split, drained from the
counters by [`print_admm_split`](../../src/learn/nanoquant.rs).

The counters are `Instant::elapsed` around the submits that already wait
(`src/gpu/limb/trace.rs`), so the trace adds no synchronisation. They are **sums over both
sides**: `cb1`, `inv`, `cb2` and `rank1` run once per side per iteration on two threads, `commit`
once per pair. A per-side time is half the print; the per-side share is `phase / (2 · loop)` for
those four and `commit / loop` for the pair-wide `commit`. The `inv` interval starts outside
`dev.lock()`, so `inv` includes the two threads' lock wait and the serialised device segments, not
only GPU time. The residual between the loop wall and the per-side sum is host work outside the
frames (draws, thread spawn, flag reads) and the readback.

| phase | what it times |
|---|---|
| `cb1` | upload `Z_x`, `Z`, `U`; plane build; the three limb products `Z_xᵀZ_x`, `W·Z_x`, `Rᵀ·H⁻¹`; the map to `hw` |
| `inv` | `hw⁻¹`: the device recursion above the cutoff and the host `spd_inverse` at each leaf |
| `cb2` | upload `H⁻¹`; `Rᵀ·H⁻¹`; the read map |
| `rank1` | the power steps forming the next target |
| `commit` | `U += X − Z`, `Z ← Z'`, the peer's `Z_x` |

## The seven linears

Two-side sums, seconds (`bench/gpu/1795396294-8b-sharing/logs/trace-round.log:57-177`). `loop` is the 400-iteration wall of the
factorisation; `ADMM` is the printed `ADMM time`, which also covers the frame build, the two
initial `rank1` and the export.

| linear | out×inn | r | loop | cb1 | inv | cb2 | rank1 | commit | ADMM | norm | per_el |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q | 4096×4096 | 2016 | 118.4 | 68.9 | **133.4** | 9.2 | 12.1 | 1.8 | 123.60 | 0.3217 | 2.2335e-4 |
| v | 4096×1024 | 800 | 22.7 | 7.4 | **31.2** | 1.5 | 2.7 | 0.4 | 23.95 | 0.3383 | 2.2442e-4 |
| o | 4096×4096 | 2016 | 118.7 | 69.4 | **133.6** | 9.2 | 12.2 | 1.8 | 123.40 | 0.2878 | 1.7921e-4 |
| k | 4096×1024 | 800 | 22.9 | 7.4 | **31.4** | 1.5 | 2.9 | 0.4 | 24.17 | 0.3364 | 2.8651e-4 |
| gate | 12288×4096 | 3040 | 493.4 | 333.3 | **512.5** | 57.1 | 37.5 | 5.3 | 508.62 | 0.2209 | 1.4947e-4 |
| up | 12288×4096 | 3040 | 494.7 | 333.9 | **514.0** | 57.3 | 37.7 | 5.3 | 509.13 | 0.2421 | 1.4651e-4 |
| down | 12288×4096 | 3040 | 492.2 | 332.4 | **511.8** | 56.4 | 37.5 | 5.3 | 507.43 | 0.3311 | 2.4644e-4 |

Per-side share of the loop wall, percent:

| linear | cb1 | inv | cb2 | rank1 | commit | untraced |
|---|---:|---:|---:|---:|---:|---:|
| q | 29.1 | **56.3** | 3.9 | 5.1 | 1.5 | 4.1 |
| v | 16.3 | **68.7** | 3.3 | 5.9 | 1.8 | 4.0 |
| o | 29.2 | **56.3** | 3.9 | 5.1 | 1.5 | 4.0 |
| k | 16.2 | **68.6** | 3.3 | 6.3 | 1.7 | 3.9 |
| gate | 33.8 | **51.9** | 5.8 | 3.8 | 1.1 | 3.6 |
| up | 33.7 | **52.0** | 5.8 | 3.8 | 1.1 | 3.6 |
| down | 33.8 | **52.0** | 5.7 | 3.8 | 1.1 | 3.6 |

`inv` is the largest phase in every linear and 54.2–72.2% of the summed phases. `inv` and `cb1`
together are 89–90% of the traced work. The two-side sum is 1.90× the loop wall on q and 1.92× on
gate, and the per-side sum is within 4.1% of it (q 284.0 ms/iter against 296.0, gate 1,188.8
against 1,233.5): each side is busy for nearly the whole wall. `ADMM time` exceeds `loop` by 5.2 /
1.25 / 4.7 / 1.27 / 15.22 / 14.43 / 15.23 s — the frame build, the two initial `rank1`, and the
export. The gate reproduces the probe's printed recon (raw 7523.1418, norm 0.2209, per_el
1.4947e-4; `bench/gpu/1795396294-8b-sharing/logs/probe-gate.log` against `bench/gpu/1795396294-8b-sharing/logs/trace-round.log:138`), so the trace does not perturb
the arithmetic at print precision.

## The layer

Wall seconds and share of the 4,587.4 s layer (`bench/gpu/1795396294-8b-sharing/logs/trace-round.log:40-189`):

| phase | wall s | share |
|---|---:|---:|
| ADMM (seven `ADMM time` values) | 1,820.3 | 39.7% |
| nonfact (seven rounds) | 1,297.4 | 28.3% |
| fact (seven rounds) | 1,365.0 | 29.8% |
| teacher and advance | 21.0 | 0.5% |
| residual (tape dump, streams, setup) | 83.7 | 1.8% |

Per side, `inv` is 934.0 s, 20.4% of the layer wall. That is the number a faster inverse acts on.

## Ceilings

Halving `inv` removes 10.18% of the layer wall, so the layer ceiling is `1/(1 − 0.1018) = 1.113×`
(the rounded `0.102` in the same formula evaluates to 1.114×).
Inside one factorisation the same halving is `1.392×` on the q loop (56.3% `inv`) and `1.351×` on
the gate loop (51.9%). Removing `inv` entirely is `1.256×` on the layer, `2.29×` on the q loop and
`2.08×` on the gate loop. These are pipeline bounds: the other 79.6% of the layer is nonfact, fact
and setup, and a change there is a separate target.

## The cutoff

The cutoff trades device nodes against host leaves. A node of order above the cutoff runs on the
device (block reads, `S = C − BᵀX`, assembly, four limb products); a node at or below it is a leaf:
the input is read back, the host `spd_inverse` inverts it, and the result is staged into the
parent. `DEV_INV_CUT` is 96, measured earlier on a 5080 at 100 iterations: host 3.3 s, cut 192
2.6 s, 96 2.4 s, 48 2.4–2.5 s, 32 2.9 s (`src/learn/nanoquant/step.rs`). The 8B r = 2016
recursion is a different shape, so `bench/gpu/campaign/inv-cut-sweep.sh` ran one q ADMM at each cut
on the final tree, restoring the same non-fact snapshot each time (`bench/gpu/1795396294-8b-sharing/logs/inv-cut-sweep.log`):

| cut | loop | cb1 | inv | cb2 | rank1 | commit | ADMM | per_el |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 105.7 | 68.2 | 109.2 | 9.5 | 11.6 | 1.8 | 112.01 | 2.2335e-4 |
| **48** | **104.8** | **67.8** | **107.3** | **9.5** | **12.7** | **1.8** | **110.96** | **2.2335e-4** |
| 64 | 116.4 | 67.7 | 130.7 | 9.7 | 12.0 | 1.8 | 121.68 | 2.2335e-4 |
| 96 | 118.4 | 70.0 | 132.3 | 9.6 | 11.7 | 1.8 | 123.58 | 2.2335e-4 |

All four print the same recon (`raw=3747.1287`, `norm=0.3217`, `per_el=2.2335e-4`), so the cut is a
route and not an arithmetic change. Cut 48 is 10.2% faster than 96 on the q ADMM and 11.5% on the
loop; its per-side `inv` is 53.7 s against 66.2 s, 18.9% lower. Cut 32 is 1% slower than 48 and cut
64 is 11.6% slower, so the response is not monotone and the sweep decides it. The campaign sets
`APEIRON_NQ_INV_CUT=48` for all seven linears. That is an engineering choice on the q sweep plus
the earlier 5080 `down` measurement, not a measured gain on `r = 3040`: gate/up/down are 82.35% of
the two-side `inv`, their sweep at n ≥ 3 is open, and no campaign speedup is claimed from the cut.
On the q linears the cut is 0.3% of the layer wall (q is 2.6% of the layer); the campaign's layer
wall is reported as measured.

## What this does not cover

- One 400-iteration run per linear, no repeat. The gate's `ADMM time` is 508.62 s here against
  505.34 s in the probe, 0.65% apart, which is the run-to-run scale.
- `inv` covers the inverse only. The frame build, pipeline creation and the export are inside
  `ADMM time` and outside `loop`; they are not attributed to a phase.
- The layer numbers are layer 0 of Qwen3-8B-Base at 128 samples, not the 36-layer campaign. The
  seven linears and their step counts are the campaign's per-layer work, but later layers' walls
  are not measured here and their checkpoints differ.
- The timings are CPU-side elapsed around submits, so they include the driver's wait and any
  scheduling delay on the two threads; they exclude GPU time after the last waited submit.
- The cutoff result is the q linears only. The gate, up and down (`r = 3040`) are 82.35% of the
  two-side `inv` and are not swept; the campaign runs 48 on them as an engineering choice, so no
  gain is claimed there and the campaign's layer wall is reported as measured.
- The sweep is one ADMM per cut, not a repeat. Cut 48's 110.96 s against 96's 123.58 s is 10.2%;
  the same cut 96 route reads 123.60 s in the trace round, 0.02% apart, and the gate's run-to-run
  scale is 0.6% (505.34 against 508.62 s). Cut 48's `rank1` reads 12.7 s against 11.6–12.0 s at the
  other cuts, and the 0.9% cut 32/48 order is a single run, so it is not resolved at this
  precision.

Review: `gemini-4-hydrogen-xhigh` (transcript
`bench/gpu/1795396294-8b-sharing/hydrogen-review-2.txt`) refused the `commit / (2 · loop)` division,
the rounded `1/(1 − 0.102) = 1.113×`, `inv` read as clean GPU time, and the cut-48 campaign setting
from a q-only single run. The first three are corrected above; the fourth is an engineering choice
with the `r = 3040` sweep open.
