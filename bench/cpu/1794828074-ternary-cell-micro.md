# The compiled ternary cell is port-bound at 2.0 cycles/group: 62 tok/s, not 24

Criterion: **T1** (decode), PLAN W5's kernel-side bound. Grades: `measured` for every rate, cycle
count, clock and instruction count here (three interleaved runs each on two builds); `model` for the
four-core projection and the 7B extrapolation. The engine loop, the pool, the head, the KV read and
DRAM contention are not in the measurement.

Companion: `bench/cpu/1794828200-ternary-cell-micro.md` (same harness, first pass, no L1 control).
It found that the harness prints `groups` 4× low (`report` credits `REPS × WORDS/4` while each
kernel walks all 8,192 words), which makes its printed `ns/group` and tok/s 4× low too. All numbers
below are recomputed from the printed seconds and 3,276,800,000 groups per 64 KiB pass; the
seconds, checksums and cycle counts are unaffected.

Host: i5-10210U (Comet Lake-U, Skylake core), 4C/8T, 15 W, 32 KiB L1d, 256 KiB L2/core. One core,
`taskset -c 0`. Source `/tmp/opencode/ternary_cell.rs`, driver `/tmp/opencode/ternary_cell_run.sh`,
raw output `/tmp/opencode/ternary-cell-runs.txt`, binaries
`/home/lucas/.cache/ternary-cell/ternary_cell_{A,B}` (`/tmp` is a 3.3 GB tmpfs charged to the
guard's memory cgroup; the linker gets SIGBUS writing there).

Two builds of one source: **A** `rustc -O -C target-cpu=native`, **B** the same plus
`-C llvm-args=-vectorize-slp=false -C llvm-args=-vectorize-loops=false`. Kernels: `reg1` (masks in
registers, 1×), `reg4` (masks in registers, 4× unrolled), `reload1` (masks reloaded per block), and
`reg1_L1`/`reg4_L1`, the same kernels over an 8 KiB slice of the buffer. Three kernels return the
term-by-term reference on 64 groups in every run.

## 1. Measured, build A (the command the brief specifies)

Buffer 64 KiB (8,192 weight words = 8,192 groups), 400,000 passes. `ns/group` is one 64-term group,
`tok/s` is the four-core projection at 7.94G terms/token.

| run | kernel | s | ns/group | G terms/s | cycles/group | tok/s (4c) |
|---|---|---:|---:|---:|---:|---:|
| A1 | `reg1` | 1.7089 | 0.5215 | 122.72 | 1.986 | 61.8 |
| A2 | `reg1` | 1.6359 | 0.4993 | 128.20 | 1.898 | 64.6 |
| A3 | `reg1` | 1.6527 | 0.5044 | 126.89 | 1.913 | 63.9 |
| A1 | `reg4` | 1.8222 | 0.5561 | 115.09 | 2.118 | 58.0 |
| A2 | `reg4` | 1.9009 | 0.5802 | 110.32 | 2.205 | 55.6 |
| A3 | `reg4` | 1.8604 | 0.5679 | 112.73 | 2.153 | 56.8 |
| A1 | `reload1` | 1.8161 | 0.5543 | 115.48 | 2.111 | 58.2 |
| A2 | `reload1` | 1.7948 | 0.5478 | 116.85 | 2.082 | 58.9 |
| A3 | `reload1` | 1.8420 | 0.5622 | 113.85 | 2.132 | 57.4 |

Spread over three runs: `reg1` 0.499–0.522 ns/group (61.8–64.6 tok/s), `reg4` 0.556–0.580
(55.6–58.0), `reload1` 0.548–0.562 (57.4–58.9). The plain loop is the fastest and sits at **1.90–1.99
cycles per 64 terms**, within 5% of the two-`popcnt` port floor of 2.0.

## 2. What the compiler emits, and why the projections were low

The 11-instruction count of `bench/cpu/1794822551-nq-loop-ports.md` §4 is not what either build
emits. LLVM vectorizes `count_ones` into the `vpshufb` nibble-LUT ctpop: build A's `reg1` loop is
**87 instructions over 4 `vmovdqu` loads = 16 words**, i.e. 5.44 instructions per 64-term group
(`reg4`: 183 over 8 loads = 32 words, 5.72); the disassembly has 108 `vpshufb` and 26 `popcnt`. Build
B is scalar: `reg1` **37 instructions per 4 words = 9.25/group**, `reg4` 50 per 4 words = 12.5/group
(four bounds checks per iteration, from the `while i + 4 <= len` form).

Implied IPC on those counted instructions, at the measured cycles: A `reg1` 2.74, A `reg4` 2.70, B
`reg1` 3.45, B `reg4` 2.88. The projection's 2.1 IPC is the error: at IPC 3.0 the same 11/64 gives
34.3 tok/s, and the compiled loop clears that. A 4-core figure at 2.1 IPC and 11 instructions
(24.0) understates the cell by 2.6×.

## 3. Clock, port, and the L1 control

Clock, `cal_clock` (`x.rotate_left(7) ^ x`, a 2-cycle recurrence, so it cannot be fooled by the
compiler): 3.708–3.884 GHz across the six runs, drift between the two calibrations in a run −2.2% to
+2.0% (`reg4`-heavy B runs drift to +1.0%). `/proc/cpuinfo` processor 0 read 3.6–4.1 GHz over the
same window; the two agree, so the cycle and IPC columns carry the calibration's ~2% uncertainty and
not a clock question.

`cal_popcnt` (8 independent `popcnt` chains): scalar build **2.63–2.90 G/s = 0.70–0.76 popcnt per
cycle**; the same probe in build A reads 3.86–3.98 G/s because SLP vectorizes it. So the 1-per-cycle
port figure that gives 62.9 tok/s is reached by the **vectorized** ctpop path (1.0 vectors/cycle over
4 words) and not by scalar `popcnt`, which measures 0.73/cycle and caps the 11-instruction cell at
**45.5 tok/s**.

L1 control, 8 KiB slice (1,024 words, inside the 32 KiB L1d), 500,000 passes:

| run | kernel | ns/group | cycles/group | tok/s (4c) |
|---|---|---:|---:|---:|
| A1 | `reg1_L1` | 0.5148 | 1.961 | 62.6 |
| A2 | `reg1_L1` | 0.4885 | 1.857 | 66.0 |
| A3 | `reg1_L1` | 0.5184 | 1.966 | 62.2 |
| A1 | `reg4_L1` | 0.5807 | 2.212 | 55.5 |
| A2 | `reg4_L1` | 0.5758 | 2.189 | 56.0 |
| A3 | `reg4_L1` | 0.6576 | 2.494 | 49.0 |

1.96 cycles/group against 1.99 for the 64 KiB buffer: the weight stream costs ≤1.5%, so the rate is
the issue/port limit and not the buffer. The 64 KiB working set is L2-resident, not L1, and it makes
no difference — which is what makes this an instruction-side ceiling rather than a memory result.

## 4. Against the asm projection

| figure | source | tok/s (4c) |
|---|---|---:|
| `reg1`, 11 instr, 2.1 IPC | `1794822551` §4 | 24.0 |
| `reg4`, 8.75 instr, 2.1 IPC | same | 30.2 |
| `popcnt` port, 2 per 64 terms at 1/cycle | same | 62.9 |
| **`reg1` as compiled, vectorized (measured)** | this file | **61.8–64.6** |
| **`reg4` as compiled (measured)** | this file | **55.6–58.0** |
| **`reg1`, vectorizer off, scalar `popcnt` (measured)** | this file | **43.6–45.5** |

The plain loop reproduces the port bound to within 5%; the 4× unroll is 7–16% *slower* per run
(mean 12%), and the scalar build's rate is 26–31% lower than the vectorized one. The ordering the
model gave (unroll better, scalar close behind) does not hold here.

## 5. Scope

One core, `taskset -c 0`, weights L1/L2-resident, no pool, no head, no KV, no DRAM. At 1.92 G
groups/s one core reads 15.3 GB/s of weight words, so ~1.8 cores saturate the measured 27.97 GB/s;
the end-to-end decode rate is byte-bound at 28.0 tok/s and this is the instruction-side ceiling only.
The four-core column is a linear extrapolation (the pool measured 2.44–2.87× on this host, so ×4 is
optimistic; ×2.66 gives 41–43 tok/s for `reg1` and 29–30 for the scalar build). `7.94G` terms/token
is the brief's figure; the pack's V plane is 7.613G, 4.3% lower, which raises every tok/s by that
factor. Not covered: the engine build (no ternary kernel in `src/`), the attained pool bandwidth,
the head at rank 2080, the 200k KV read, and the ternary activation's Q2 quality.

## 6. Reproduce

```sh
~/Workspace/tools/agent-guard run -- rustc -O -C target-cpu=native \
  -o /home/lucas/.cache/ternary-cell/ternary_cell_A /tmp/opencode/ternary_cell.rs
~/Workspace/tools/agent-guard run -- rustc -O -C target-cpu=native \
  -C llvm-args=-vectorize-slp=false -C llvm-args=-vectorize-loops=false \
  -o /home/lucas/.cache/ternary-cell/ternary_cell_B /tmp/opencode/ternary_cell.rs
~/Workspace/tools/agent-guard run -- bash /tmp/opencode/ternary_cell_run.sh   # 3 runs x 2 builds
objdump -d --demangle --no-show-raw-insn /home/lucas/.cache/ternary-cell/ternary_cell_A \
  | awk '/ternary_cell::cell_reg1>:/{p=1;next} p&&/^$/{exit} p'
```

The program pulls `/proc/cpuinfo` before and after each kernel and calibrates the clock from the
2-cycle recurrence; a run is refused to be evidence if the three kernels disagree with the
term-by-term reference on 64 groups.
