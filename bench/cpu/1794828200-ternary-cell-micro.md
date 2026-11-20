# The 64-term ternary cell in a standalone loop: 1.02 popcnt/cycle, port-bound

Criterion: **T1** (decode), partial; PLAN W5's kernel-side bound. Grade: `measured` for the loop,
`model` for the 7B extrapolation. The engine loop, the pool, the head, the KV read and DRAM
contention are **not** in this measurement. No speedup over the tree is claimed: the cell is not in
the tree.

Host: Intel Core i5-10210U, 4C/8T, 15 W, Ubuntu 24.04, `rustc -O -C target-cpu=native`. One core,
`taskset -c 0`. Buffer 64 KiB of random weight words (L2-resident, 8,192 words = 8,192 groups).
Source `/tmp/opencode/ternary_cell.rs`, driver `/tmp/opencode/ternary_cell_run.sh`, raw output
`/tmp/opencode/ternary-cell-runs.txt`, binaries `/home/lucas/.cache/ternary-cell/`. Three
interleaved runs per build; build A is the source as written, build B disables the SLP and loop
vectorizers. The three kernels return the same checksum as the term-by-term reference on 64 groups.

## 1. The harness undercounts groups by 4×; the corrected rate is the port bound

`cell_reg1` walks `buf` (8,192 words, one 64-term group each), but `report` multiplies `REPS` by
`GROUPS = WORDS/4 = 2,048`. The printed `groups`, `ns/group` and tok/s are therefore 4× low; the
printed `cycles/group` and `IPC` are not usable. The seconds and the checksums are correct. True
groups per run: `400,000 × 8,192 = 3,276,800,000`.

| run | kernel | s | G groups/s | G terms/s | cycles/group | popcnt/cycle |
|---|---|---:|---:|---:|---:|---:|
| A1 | `reg1` | 1.7053 | 1.922 | 123.0 | 1.953 | 1.024 |
| A2 | `reg1` | 1.7411 | 1.882 | 120.4 | 1.976 | 1.012 |
| A3 | `reg1` | 1.7227 | 1.902 | 121.7 | 1.971 | 1.015 |
| B1 | `reg1`, scalar | 2.4824 | 1.320 | 84.5 | 2.838 | 0.705 |
| B2 | `reg1`, scalar | 2.4262 | 1.351 | 86.4 | 2.759 | 0.725 |
| B3 | `reg1`, scalar | 2.3759 | 1.379 | 88.3 | 2.724 | 0.734 |
| A1 | `reg4` (4× unroll) | 1.9597 | 1.672 | 107.0 | 2.245 | 0.891 |
| A2 | `reg4` | 2.0015 | 1.637 | 104.8 | 2.271 | 0.881 |
| A3 | `reg4` | 1.9638 | 1.669 | 106.8 | 2.247 | 0.890 |
| A1 | `reload1` (masks reloaded) | 1.9276 | 1.700 | 108.8 | 2.208 | 0.906 |
| A2 | `reload1` | 1.9469 | 1.683 | 107.7 | 2.209 | 0.905 |
| A3 | `reload1` | 1.9428 | 1.687 | 107.9 | 2.223 | 0.900 |

Cycle counts use the run's own 2-cycle-recurrence calibration (3.718–3.757 GHz; the second
calibration in each run reports 3.32–3.61 GHz, a −4.7 to −10.6% clock drift, so cycle figures carry
that uncertainty). `popcnt/cycle` is 2 popcounts per group ÷ cycles per group.

The plain loop sits at **1.95–1.98 cycles per 64 terms**, one percent from the two-`popcnt` port
floor of 2.0. That is the same bound `bench/cpu/1794822551-nq-loop-ports.md` §4 computed (62.9
tok/s at 7.94G terms/token on four cores at 3.9 GHz): the kernel is at it, not near it. The 4×
unroll is **12–14% slower** and the mask-reload variant **11–13% slower**; the model's "30.2
unrolled / 20.3 reloaded" ordering was wrong for this host, because the plain loop is already
port-bound and unrolling only adds retirement pressure. The scalar build is 1.4× off the port
(0.70–0.73 popcnt/cycle), so a compiler that refuses the vector/dual-issue shape costs 30% of the
rate but still clears the byte ceiling.

## 2. What that projects, and what binds

Terms per token: 7,613,448,192 sign terms (the 1.174 b/w pack's V plane); pack-plus-fitted-head
bytes 0.998 GB/token, measured DRAM 27.97 GB/s, byte ceiling **28.03 tok/s**.

| bound | four cores at 3.75 GHz | four cores at 2.6 GHz (sustained all-core) |
|---|---:|---:|
| `popcnt` port, 2 per 64 terms | 480 G terms/s → **63.0 tok/s** | 333 G terms/s → **43.7 tok/s** |
| scalar build | 346 G terms/s → 45.4 tok/s | 240 G terms/s → 31.5 tok/s |
| DRAM bytes, 0.998 GB/token | **28.03 tok/s** | 28.03 tok/s |
| T1 target | 30 at 2k, 24 at 200k | |

**Bytes bind first.** At the measured rate one core reads 1.92 G groups/s × 8 B = **15.4 GB/s** of
weight words; two cores would demand 30.7 GB/s, above the 27.97 GB/s measured DRAM. The cell
saturates bandwidth at **1.8 cores**, so the pool's width beyond two buys nothing for the
projections: 20 tok/s needs 20.0 GB/s (71% of DRAM) and 1.3 cores of cell work. The kernel headroom
to 20 is **2.2× at 2.6 GHz all-core** (43.7/20), not the 1.2× the 2.1-IPC front-end model implied.
The remaining risk is not the cell: it is the pool's attained bandwidth, the head, and the KV read.

## 3. What this does not cover

The loop is L2-resident and single-core; the engine's weights stream from DRAM while the pool and
the KV read compete, so this is the instruction-side ceiling, not the end-to-end rate. It is a
standalone `rustc` build, not the engine build: `src/` has no ternary kernel, and the engine's
actual codegen must be read from `project_q8`'s replacement with `phase_split` once built. The
head at rank 2080 (0.040 GB), the 200k KV read and the 2k/200k targets are outside. No claim that
the ternary **activation** meets Q2: its quality screen is separate (`bench/notes/
trichotomy-cell.md` §6).
