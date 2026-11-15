# The NanoQuant fit's cost curve, and what a smaller exponent would cost

Status: `measured` unless a line says otherwise. Two machines: vast.ai `60743436`, 20 vCPU
allocated on an 80-vCPU host (E5-2673 v4, Broadwell), 251 GB; and vast.ai `60770801`, i5-13400F
(Raptor Lake, AVX-VNNI), 16 vCPU, 15.7 GB. Teacher `Qwen3-0.6B-Base-F16`, `down` (1024×3072),
rank 864, 1.18 b/w. The Broadwell box is shared and contended, so its absolute wall numbers are
not the point; the ratios and the model are. Each table names its box.

## What one iteration costs

`factorize` runs two `x_update`s and two `rank1`s per outer iteration. Per side, on a
`k × n` target at rank `r` (`k` the side's row count, `n` the contraction width):

| step | multiply-adds | note |
| --- | --- | --- |
| `wp.gemm(&zt, r)` — `W·Zᵀ` | `k·n·r` | the data pass |
| `gemm_q32(&rt, k, &hi, r, r)` — `Rᵀ·H⁻¹` | `k·r²` | latent solve |
| `syrk` — `ZᵀZ` | `r²·n/2` | ridge Gram, upper triangle |
| `spd_inverse` | `2r³/3` | host block recursion |
| `rank1` (5 power steps) | `10·k·r` | negligible |

`spd_inverse` is `2r³/3`, not the `r³/3` of a Cholesky factorisation the reference paper
advertises; the term is small here anyway (7%).

For `down` the sides are `1024×3072` and `3072×1024`, so each pays `2.72G` for its data
pass; the two latent solves are `2.29G` and `0.76G`, the two half-Grams `0.38G` and `1.15G`,
and the two inverses `0.86G`: `10.9G` multiply-adds per iteration, against `12.4G` before the
half-Gram change. Grade: `model` (from the source, no per-step timer).

## A routing cliff, and the 4.2× it cost

`projection` hands each product to `matmat_q8act`, whose shared-decode AVX2 kernel was
capped at `k ≤ MAX_K = 16` columns; past that it ran `per_column`, one `matvec_q8act` per
column, re-reading the whole packed weight payload every time. The fit's `X`-update projects
with **one column per rank** (`k = r = 576–1024`), so every projection took the per-column
path: 864 passes over 18.9 MB of payload per call. That, not arithmetic, was the 18 s.

The fix raises the pooled `Q8_0` kernel to `k ≤ MAX_POOLED_K = 4096` within a 32 MiB output
budget, with heap accumulators; the portable kernel keeps its `MAX_K` stack bound and anything over
budget falls to the same per-column reference. Values are unchanged — the pooled kernel is tested
bit-identical to the per-column contract, and
`engine::kern::matmat::tests::columns_past_max_k_match_per_column` exercises `k` past the
cap, asserting both that the pooled kernel takes the call and that every column equals
`matvec_q8act`.

Back-to-back A/B, same host, `APEIRON_THREADS=10`, layer-0 `down`, 10 iterations:

| binary | wall | `admm_loss` |
| --- | --- | --- |
| before (`matmat.rs` at `MAX_K = 16`) | 288.1 s | 0.48566 |
| after | **68.7 s** | 0.48566 |

**4.19×**, with the loss identical to five decimals. `limb_bench` (`gemm_q32` at
512×896×896, same host) moved 4.07 → 1.71 ns/MAC; the naive `i128` reference stayed at
4.2–4.9 ns/MAC, which is what "both arms are scalar" looked like.

`matmat.rs` is inside the extraction unit (`proof/extract/lib.rs` pulls in
`src/engine/kern/mod.rs`, which includes it), so this change is over the real source, not a
neighbouring copy: the maintainer gate is owed for the regenerated `ApeironWord.lean` and the
`Serve/Kern/R4/Matmat5.lean` / `R4/Matmat6.lean` shards that refine `fast_family`, `pooled_rows`,
`pooled_q8_0` and `matmat_q8act`.
`proof/regen.sh --check` and `lake build` are the two commands; neither has run since the change.

## What a second costs now

`APEIRON_NQ_ITERS ∈ {2,10,25,50}` before the fix, layer-0 `down`, concurrently with the
calibration prefill:

| iters | wall | `admm_loss` |
| --- | --- | --- |
| 2 | 18.1 s | 74.74971 |
| 10 | 81.7 s | 0.48566 |
| 25 | 362.7 s | 0.33786 |
| 50 | 800.1 s | 0.30741 |

Steady state was **13.6–18.7 s per iteration** (`0.43–0.91 G MAC/s`), the spread being host
contention. After the fix, the same 10-iteration fit is 6.7 s/iteration at `1.85 G MAC/s`.
Repeat runs of the *same* binary differed by 2.1× (81.7 s against 136.2 s, then 288.1 s), so
a single wall number on this box is not evidence; the A/B above was taken back-to-back.

## Where an iteration goes, after the fix

Three rank arms and two thread widths on the same host and cache, layer-0 `down`, five iterations
each (`APEIRON_NQ_ITERS=5`, `APEIRON_NQ_RELH=0`, back-to-back):

| arm | wall | per iteration | note |
| --- | --- | --- | --- |
| r 864, TH 10 | 24.9 s | 4.0 s | the reference rank at 1.18 b/w |
| r 432, TH 10 | 8.0 s | 1.3 s | |
| r 96, TH 10 | 1.8 s | 0.3 s | below the fixed setup, so an upper bound |
| r 864, TH 1 | 357.5 s | ~70 s | 17.5× fewer iterations/s than TH 10 |

The rank slope is 13× for a 9× rank: the data pass (`Θ(r)`) carries most of the marginal iteration
and the `r²` Gram/solve terms the rest. The `r³` inverse does not separate from a three-point fit;
the source's own receipt (`spd_inverse_base_wall`) puts the order-768 inverse at 12.2 ms at base 32,
which is under 1% of an iteration. Thread width is a real lever here: 17.5× at ten threads on a
20-vCPU quota, because the single-thread arm is latency-bound. The earlier "32 and 64 threads are
worse" runs were host contention, measured while another tenant held about 70 of the host's 80
CPUs.

## The pooled kernel's own rate

`engine::kern::matmat::tests::pooled_rate_at_fit_shape` (release, best of three) on the fit's
projection, `n_out 1024`, `n_in 3072`, `avxvnni false`:

| k | best_ms | G MAC/s | activation stream | note |
| --- | --- | --- | --- | --- |
| 864 | 146.4 | 18.5 | 2.6 GB | the fit's `X`-update, pool |
| 432 | 68.0 | 19.4 | 1.3 GB | |
| 96 | 19.2 | 14.9 | 288 MB | |
| 32 | 3.6 | 27.8 | 96 MB | L2-resident activation set |

The same binary at `APEIRON_THREADS=1`: 2.786 s at `k = 864`, **0.86 G MAC/s**. The pool buys
21.5×; the shape is cache-sensitive, since shrinking the activation stream from 2.6 GB to 96 MB
raises the rate from 18.5 to 27.8 G MAC/s at the same instruction mix. That is the 2-D tile's
headroom (a `R`-row tile divides the stream by `R`): ≤1.9× on this host, not 10×.

For calibration, `chart/pool.rs` records **20.76 G weight/s single-thread** for the `Q5_0`
4864×896 chart projection on an i9-14900KF P-core, against 0.86 G MAC/s here: 24× per core. That
gap is the machine — a dedicated 5.5 GHz core with AVX-VNNI against roughly 0.43 GHz of delivered
issue slots — not the kernel.

## The same fit on an AVX-VNNI host

Second box: i5-13400F (Raptor Lake, 16 vCPU, `avx_vnni` on every lane, 15.7 GB), same source,
same teacher, same Gram, `APEIRON_THREADS` set explicitly, layer-0 `down`, five iterations.

| arm | Broadwell box | i5-13400F | ratio |
| --- | --- | --- | --- |
| fit, 5 iterations, `APEIRON_THREADS` 10 vs 16 | 25.7 s | **2.4 s** | 10.7× |
| fit, 5 iterations, `APEIRON_THREADS=1` | 357.5 s | **4.1 s** | 87×, but 1 vs 5 participants |
| pooled kernel `k = 864`, `APEIRON_THREADS=1` | 0.86 G MAC/s | **141.5 G MAC/s** | 165×, 1 vs 5 participants |
| pooled kernel `k = 864`, pool | 18.5 G MAC/s | **371.9 G MAC/s** | 20× at equal width |

`admm_loss` is 2.64504 on both boxes and every arm: the VNNI kernel returns the same integers,
as it must. `APEIRON_THREADS` is not the participant count. The pool caps at the pinnable
performance cores and adds the efficiency cores by default, so on this i5-13400F the "one thread"
arm is five participants (the caller and four E-cores) and the sixteen-thread arm is ten (six
P-cores with the caller, plus four E-cores). The 87× and 165× rows therefore compare one Broadwell
participant with five Raptor Lake participants. The equal-width rows (ten participants on both
boxes) are the 10.7× and 20× figures; a true one-participant arm needs `APEIRON_ECORES=0` and is
not measured here.

## Where an iteration goes on the VNNI host

`x_update_phases` (release, best of three, the two sides of `down` after the wide-matrix
transpose, `r = 864`, `APEIRON_THREADS=16`, ten participants). Each row is the call `x_update`
makes, in order; `phase sum` is the sum of the lows and `x_update` is the whole function:

| phase | `k=3072, n=1024` ms | share | `k=1024, n=3072` ms | share |
| --- | ---: | ---: | ---: | ---: |
| `wp.gemm` (`W·Zᵀ`) | 43.87 | 31% | 61.52 | 39% |
| `syrk_q32` (`Z·Zᵀ`) | 19.67 | 14% | 48.40 | 31% |
| `spd_inverse` | 19.74 | 14% | 19.27 | 12% |
| `gemm_q32` (`Rᵀ·H⁻¹`) | 45.62 | 32% | 13.76 | 9% |
| three `grid_map`s | 10.91 | 8% | 12.81 | 8% |
| `par_fill` | 2.69 | 2% | 0.69 | 0.4% |
| phase sum | 142.50 | | 156.45 | |
| `x_update` | 150.37 | | 168.38 | |

The four kernel calls are 90% and 91% of the phase sum; the scalar maps and the final fill are
under 10%. The earlier reading divided the fit's total multiply-adds by the int8 kernel's rate and
called the remainder scalar, but the fit's products are *limb* products: six kept digit pairs per
Q32.32 multiply, and `wp.gemm` moves `2.72 G` Q32.32 products in 43.9–61.5 ms, i.e. `264–372 G`
byte-MAC/s against the measured pooled `Q8_0` rate of `371.9 G/s` (71–100%). The kernel is at its
own peak; the residue is not scalar.

What is left lossless is the two calls that run below it. `syrk_q32` does `1.15 G` Q32.32 products
in 19.7 ms (58 G/s) on the `n=1024` side but 48.4 ms (23.8 G/s) on the `n=3072` side, where its
16-column tiles re-read the activation rows; `spd_inverse` runs at about 22 G/s on both. Together
they are 26–45% of the phase sum at a third to a half of the kernel's rate, so a larger `syrk`
tile and a cheaper inverse are the next lossless moves. The iteration budget is the only lever
that changes the artifact.

## The second lossless move: half the Gram

`x_update` formed `Z·Zᵀ` with the general `gemm` on the prepared plane. The result is symmetric,
and `syrk_q32` computes the upper triangle and mirrors it: half the kernel calls, the same pair
grouping, the same words. `src/learn/nanoquant.rs` now calls it; the term is the largest of the
quadratic ones (2.29 G multiply-adds on the A side of `down` at `r = 864`).

Back-to-back, five iterations, `APEIRON_THREADS=10`, alternating old and new binaries:

| pair | `gemm` | `syrk` | ratio | `admm_loss` |
| --- | --- | --- | --- | --- |
| 1 | 30.4 s | 25.7 s | 1.18× | 2.64504 both |
| 2 | 99.7 s | 71.4 s | 1.40× | 2.64504 both |
| 3 | 109.4 s | 29.1 s | 3.76× | 2.64504 both |

The spread is the host, not the change; every pair favours the half square and every loss is
identical to five decimals. `src/learn/nanoquant.rs` is not in the extraction unit, so the swap
itself has no Lean statement. The kernel it calls is extracted and unproved:
`src/engine/kern/limb.rs::tests::syrk_matches_gemm_bit_for_bit` is the test that `syrk_q32(a)` is
`gemm_q32(a, a)` bit for bit and symmetric, and no corpus shard refines `kern.limb.syrk_q32`.

## What the 100× is made of

| lever | measured | grade |
| --- | --- | --- |
| `MAX_K` routing | 4.19× (288.1 → 68.7 s) | `measured` |
| `syrk` instead of `gemm` for `Z·Zᵀ` | 1.18× / 1.40× / 3.76× per back-to-back pair, `admm_loss` identical | `measured` |
| AVX-VNNI + delivered clock | 10.7× at equal width (ten participants); 87× at `APEIRON_THREADS=1` is 1 vs 5 participants, not per core | `measured` |
| iteration budget `T = 400 → 200` | 2×, 1.3% norm penalty on layer-0 `down` | `measured` |
| iteration budget `T = 400 → 50` | 8×, 7.4% norm penalty on layer-0 `down` | `measured` |

The routing fix, the first two uncontended half-Gram pairs (1.18–1.40×) and the equal-width host
factor multiply to 53–63× at the pool, with no change to a single output word. The direct
before/after is the 400-iteration fit: the pre-fix slope is `288.1 s / 10 = 28.8 s` per iteration,
so 400 iterations at that slope is `11.5 ks`; the measured 400-iteration fit here is **174.7 s**,
**66×**. That 66× crosses two hosts, so it is an arithmetic projection (`model`) over measured
slopes, not a measured ratio; the same-host A/B product is 53–63× and the measured same-host
routing ratio is 4.19×. The gap to 66× is the setup and the baseline host's contention.

The iteration budget is the fourth lever and the only one that changes the artifact. Measured on
layer-0 `down` against the same shipped factor:

| T | wall | `relH` (norm) | vs T=400 | vs the shipped factor | speed vs the pre-fix 400 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 23.2 s | 47.70% | +7.4% | 1.064× worse | 497× |
| 100 | 41.1 s | 45.98% | +3.5% | 1.025× worse | 280× |
| 200 | 83.8 s | 45.02% | +1.3% | 1.004× worse | 137× |
| 400 | 174.7 s | 44.42% | — | 1.010× better | 66× |
| shipped | — | 44.85% | −1.0% | 1.000× | — |

`T = 200` is the lossless 2× on this row: 137× overall for 1.3% more norm error. `T = 50` is
497× for 7.4%. That is where "100×" sits: in the constant, not the exponent.

## This host is shared, and delivers about a sixth of a core

A 10⁹-iteration `volatile` `imul+add` loop (`gcc -O1`) takes **2.42 / 2.70 / 2.54 s** at a
reported 2.3–2.65 GHz. The same loop on a dedicated core is ~0.4–0.8 s. `load average` was
**70** at the time with only `learn_calib` (1097 % CPU) among our processes — the rest is
another tenant on the 80 vCPUs. `cat /sys/fs/cgroup/cpu.max` does not exist, so there is no
quota to point at: the cycles are simply not delivered.

That factor is invisible to `nproc` and to `cpu MHz`, and it explains part of the distance
to the laptop receipt in `~/Workspace/apeiron/proof/corpus/Learn/Quant/NanoQuant.lean`
(`iters400Wall = [613, 613]` tenths for the blk.23 `ffn_down` fit, 400 iterations): same
kernel source, a host with AVX-VNNI and a dedicated clock.

Pool width is not a lever here: the same 10-iteration fit took 136.2 s at `APEIRON_THREADS=10`,
194.9 s at 32 and 681.7 s at 64. More workers on a kernel that is not scaling just steal
from each other. That saturation is the shared host's, not the pool's: on the VNNI box the
five-to-ten participant step buys 2.05× on one `x_update` (310–322 ms to 150–168 ms).

## The curve is Θ(P·n), and Ω(P) is a floor

`r = (o·n·β/(o+n) − 16)` rounded down to 32, so `r = Θ(min(o,n))` at fixed bits/weight —
`r/n = 0.84` for `down` at 1.18 b/w. With `k ≈ o`, one iteration is
`Θ(o·n·r + o·r²) = Θ(o·n²)` for a square layer, and a model of `L` layers with
`P = L·o·n` weights costs `Θ(400·P·n)`. A 4× wider model costs ~8× per weight. That is the
curve: cubic per layer, quadratic in the width at fixed parameters.

It is not exponential, and no fit of this artifact can be sublinear in the parameters: the
artifact itself is `r(o+n) + 16(o+n)` bits at ~1.15 bits per weight, i.e. `Θ(P)` bits, so
writing it is already `Ω(P)`. What the implementation spends is `O(P)` per pass with a
constant of ~800–1000 passes per layer (`2·400` data passes plus the Gram and solve terms).
**The exponent is already 1; the constant is the lever.**

### What O(n log n) would require

Only a *structured* artifact can beat `Θ(P·n)` per pass, because a dense rank-`r` factor
with `r = Θ(n)` has no small summary. Two candidates and their costs:

| factor | storage | apply per token | fit per pass |
| --- | --- | --- | --- |
| dense ±1 (shipped) | `r(o+n)` ≈ 1.15 b/w | `O(o·r + r·n)` | `Θ(P·n)` |
| Monarch `L·P·R`, `√n` blocks | `2n√n` bits | `O(n^1.5)` | `Θ(P·√n)` |
| butterfly, `log n` ±1 levels | `n·log n` bits | `O(n log n)` | `Θ(P log n)` |

The butterfly row is the `O(n log n)` asked for, and it cannot meet the 1.18 b/W bar: at
`n = 1024` it stores 10 bits per element against a 1.18-bit budget. So "O(n log n) at 1.18
b/w" is not a constant-factor gap, it is a different compression scheme with a different
rate. Monarch is the one that could pay: `Θ(P·√n)` per pass at 2–3 bits per element, which
is above 1.18 b/w too. Neither is a drop-in change to this card.

## Two cheaper optimisers, both worse (measured, synthetic)

`/tmp/opencode/nqfit/`. Synthetic 192×256, rank 128, anisotropic heavy-tailed Gram,
per-row/col scale variation; the score is `tr(EHEᵀ)/tr(WHWᵀ)` with the scales refitted by
exact coordinate descent for every method.

| method | E | note |
| --- | --- | --- |
| truncation floor (real rank-128) | 0.031 | unattainable with ±1 factors |
| `sign(SVD)` | 0.733 | rounding the factors to ±1 destroys the fit |
| ADMM 400 | 0.422 | |
| sign coordinate descent from random | 0.700 after 6 sweeps | |
| sign coordinate descent from `sign(SVD)` | 0.788 after 6 sweeps | converges, far too slowly |
| GPTQ-style sequential (error feedback) | 0.903 → 0.968 | degrades; the basis is not learned |
| ADMM 200 + **one** sign sweep | 0.316 | the sweep is a cheap polisher, not a replacement |

On a *less* anisotropic Gram the ADMM itself plateaus by 25 iterations (0.843 at 25, 0.847
at 400), so how much the iteration budget buys is data-dependent. The real Grams decide.

Two bugs found and fixed in the prototype on the way, both worth knowing before porting:
the V-side row objective needs `c = aa·g₀ − R_kᵀa_k` (sign), and the scale coordinate
descent drifts along the exact `sp → c·sp, sq → sq/c` ambiguity unless it is anchored.

## Levers, ranked by measured size

1. **The routing cliff** (done): `k > 16` sent every fit projection to the per-column
   reference. 4.19× measured, values bit-identical.
2. **Half the Gram** (done): `syrk_q32` instead of `gemm` for `Z·Zᵀ`. 1.18–3.76× per
   back-to-back pair, `admm_loss` identical.
3. **Run the fit where the multiply-adds are cheap.** 10.7× at equal width (ten participants)
   for the whole fit, on an i5-13400F against the contended Broadwell box; 87× at
   `APEIRON_THREADS=1`, which is one participant against five and is not a per-core ratio. The
   reference (arXiv 2602.06694, Table 4) compresses Llama-2-7B at 1.00 b/w in 1.7 H100-hours,
   ~1.8 T MAC/s.
4. **`syrk` and the inverse on the VNNI host** (measured phase split): the four kernel calls
   are 90–91% of one `x_update`; `wp.gemm` is at 71–100% of the pooled `Q8_0` rate, but
   `syrk_q32` is 58 G/s on the narrow side and 23.8 G/s on the wide side, and `spd_inverse`
   is ~22 G/s on both. Together they are 26–45% of the phase sum, so a larger `syrk` tile and
   a cheaper inverse are the next lossless moves.
5. **Iteration budget**: 2× for 1.3% norm error at T=200→400, and 8× for 7.4% at T=50→400,
   measured on layer-0 `down` against the same shipped factor. This is the only multiplier that
   changes the artifact.
6. **Cholesky instead of the explicit inverse**: `2r³/3 → r³/3` halves the product count and
   removes the `n/32` base Grams; `spd_inverse` is 12–14% of `x_update`, so the saving is
   bounded by that, not by the product term alone.

Not established: whether T can be cut on the real Grams (measurement queued), whether the
`relH` floor of the ±1 family sits near the ADMM's 400-iteration value, and whether
`VK_KHR_cooperative_matrix` exists on this instance's GTX 1660 Ti (the `gpu` feature targets
the limb path, not torch).
