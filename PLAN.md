# PLAN — a 7B at 20 tok/s on the 6.6 GiB laptop

This file tracks state. It is not a receipt. Every number below names the artifact behind it; when a
number moves, its row moves. Criteria are `GOAL.md`'s.

Target: **Qwen2.5-7B, 1.174 b/w, ≥20 tok/s decode at 2k, ≤1.5× Q4_K_M PPL**. The minimum line is
9.6 tok/s (1.5× Q4_K_M's 6.40 tok/s bytes ceiling on this host). Below 9.6 the product does not
exist on this laptop; 20 is the working target.

## 1. Where the time goes

The byte advantage is already spent. At 1.174 b/w the pack is 0.9576 GB and the fitted head
0.040 GB; 20 tok/s reads 20.0 GB/s against 27.97 GB/s measured, 71% of bandwidth
(`bench/cpu/1794811020-one-bit-vs-four-ceiling.md`, `1794810332-nq-head-in-pack.md`). For the q8
reader the binding constraint is instructions per sign term. A 64-term ternary-activation cell
removed that binding in a microbenchmark, but its activation failed the quality screen by
1,065,125×; W2 was an agent-opened exploration, not a requested path, so the CPU rate remains q8
11.0 and int4 14–18, both under 20.

| path | instructions / 32 terms | per term | 4 cores at 2.1 IPC | at 3.0 IPC |
|---|---:|---:|---:|---:|
| `i64` lanes `sign_dot_avx2` | 58 | 1.81 | 2.3 tok/s | 3.3 |
| four-plane `sign_dot_planes_avx2` | 24 | 0.75 | 5.5 | 7.8 |
| q8 block `code_select_avx2` (production) | 12 | 0.375 | 11.0 | 15.7 |
| ternary cell, exploratory and refuted | 2 cycles / 64 | 0.031 | 63.0 | 43.7 |
| budget for 20 tok/s | ≤9.4 / 32 at 3 IPC, ≤6.6 at 2.1 | ≤0.21–0.30 | | |

The ternary row is a measured rate ceiling from W2, not a candidate: its activation is
quality-refuted in section 2. The q8 loop is **12 static instructions** per 32-term block (13
execution uops, 11 front-end):
`vpsrlvd`, two `vpshufb`, `vpand`, `vpmaddubsw` with the activation folded, `vpmaddwd`, `vpsllvd`,
`vpaddd`, three loads, and the loop control; the two `vpshufb` are the selector expansion
(`bench/cpu/1794822551-nq-loop-ports.md`). Its 2.1 IPC ceiling is 11.0 tok/s, which brackets the
measured 8–10 tok/s band. **20 tok/s cannot be reached by improving that loop; the selector
expansion has to disappear.**

W2 tested a cell that removes the selector expansion. It is retained as a measured ceiling and a
negative quality result, not as a route:

- weights stay ±1, one bit per sign term, exactly as the 1.174 b/w pack stores them;
- the activation of a 64-term block is quantized to `t ∈ {−1, 0, +1}` with a per-block power-of-two
  scale, as two 64-bit masks `p` (positive) and `n` (negative), `p & n = 0`;
- with `w` the stored plane word (bit 1 is `+1`) and the kernel's sign coefficient
  `s_i = 1 − 2·bit_i(w)` (bit 1 contributes `−t`; `sign_dot!` computes exactly this, `nq.rs:382`,
  and U and V share the inversion), `Σ s_i t_i = 2·popcount(w & n) − 2·popcount(w & p) +
  popcount(p) − popcount(n)`. The last two counts are per activation block, shared by every output
  row. The dot is **two `and` + two `popcnt` + three adds**; the compiled loop adds the weight load,
  the two mask loads, the accumulates and the loop control. Checked exhaustively over 3- and 4-term
  cells and 200,000 random 64-term cells.

**Measured, standalone** (`bench/cpu/1794828200-ternary-cell-micro.md`): the plain loop runs
**1.95–1.98 cycles per 64-term group**, one percent from the two-`popcnt` port floor, on one core at
3.72–3.76 GHz — **120.4–123.0 G terms/s**, which is 1.00–1.02 `popcnt`/cycle. Four cores at
3.75 GHz project **63.0 tok/s**; at a sustained 2.6 GHz all-core, **43.7 tok/s**. The 4× unroll and
the mask reload are **12–14% slower**, not faster: the loop is already port-bound. The scalar
(no-vectorizer) build is 84.5–88.3 G terms/s, 1.4× off the port.

**Bytes bind first, not the kernel.** One core reads 15.4 GB/s of weight words; two would demand
30.7 GB/s, above the measured 27.97. The pack's 0.998 GB/token caps the token at **28.03 tok/s**,
and 20 tok/s needs 20.0 GB/s (71% of DRAM) and 1.3 cores of cell work. The kernel headroom to 20 is
**2.2× at 2.6 GHz all-core**. The remaining rate risk is the pool's attained bandwidth, the head and
the KV read, not the cell.

The cell did not meet Q2 (section 2), so the CPU fallback is int4 activations with the existing
selector (`~10 instr / 32`, 14–18 tok/s) and a sparse exact residual. That is above the 9.6 minimum
but below 20, so the laptop's 20 tok/s claim depends on V1 or a cheaper carrier that also passes
the quality screen.

**The GPU carrier reads 539.5 GB/s.** A CUDA probe of the two-popcount 1-bit GEMV on a fractional
RTX 2080 Ti measures 539.5 GB/s on a 1 GiB stream and 537.0 GB/s on the 0.998 GB fold, 87.6% of the
616 GB/s spec. The 7B pack's 0.998 GB/token is a **540.5 tok/s** ceiling, 27× the 20 target; 20
tok/s would use 3.7% of the card. That is the carrier's bytes, not the repository's Vulkan kernel.
The device now initializes on a rented RTX 3060 (`NVIDIA_DRIVER_CAPABILITIES=all` plus the X/GL
userspace, proprietary 580.173.02), the example compiles and submits dispatches, but the decode
GEMV writes zero on every checked row (`bench/gpu/1794836795-vulkan-bringup.md`): V1 is open, with
the failure localized to the first dispatch. The GPU is the route to 20+ if the kernel is fixed and
the rest of the token (attention, KV, head) fits the remaining 96%.

## 2. The quality gap that gates everything

| point | rate | ctx / tokens | PPL vs Q4_K_M | source |
|---|---|---|---|---|
| Qwen3-0.6B v3 | 0.978 b/w | 512 / 2,044 | 4.10× | `1794822341-nq06-ppl-vs-q4.md` |
| Qwen3-0.6B v3 | 0.978 b/w | 512 / 8,176 | 2.52× | same |
| Qwen3-8B v3 | 1.124 b/w | 128 | 1.755× | `1794740406-nq-118-distortion.md` |
| Qwen3-8B v3 | 3.13 b/w | 128 | 1.011× | `1794653685-nq8b-residual-ppl.md` |
| Qwen3-0.6B layer 0 | 1.153288 b/w, T=400, column-RMS | 512 / 2,044 | **185,327** | preconditioner failure, not a rate result; `1794834689` |
| Qwen3-0.6B layer 0 | 0.952344 b/w, T=400, column-RMS | 512 / 2,044 | **1,225,029** | same nominal rate as the 57.045 control, different preconditioner; erratum |
| Qwen3-0.6B layer 0 / 13 / 27 | 0.978 b/w, T=400 | 512 / 2,040 | **57.045 / 18.441 / 20.581** | layer controls, `ppl-q4-v3l{0,13,27}-2044.log` |
| Qwen3-0.6B v3, W2 ternary activation (exploratory, not requested) | 0.978 b/w | 512 / 2,044 | **77,741,390** vs same-binary q8act 72.988013 (**1,065,125×**) | negative result, `1794834226-ternary-ppl.md` |
| **Qwen2.5-7B** | **1.098871 b/w, 196 sections** | 512 / 2,044 | **not measured**: Q4 9.298465; NQ refused at the first norm because the T=2 o/down factors gain 10^4–10^6; `1794862546` |

0.978 b/w is below the Q1 band, so the 0.6B rows are lower bounds, not the production point. The
0.6B arm's head is the skeleton's `Q8_0` against Q4_K_M's `Q6_K`, so its ratio includes a head
difference; the 7B screen keeps the Q4_K_M head in both arms and scores the fitted head separately.
The full 7B pack now exists: **196 sections at 1.098871 b/w**, 896,312,736 B, fitted in 6.854 h on
an i9-12950HX for $0.653 (`bench/cpu/1794862546-nq7b-t2-ppl.md`). The Q4_K_M arm scores **9.298465**
on 2,044 tokens; the packed arm refuses at the first norm, and the cause is the **fit**, not the
kernel: the T=2 `admm_loss` for v/o is 9,094/10,546 against the 0.6B T=400 band of 0.22–0.35, and
layer-0 o/down outputs are 1.2e13/1.8e16 against the dense arm's 2.1e9. The exact reader gives the
same bands, so T=2 is not a quality upper bound either. **Q2 remains unknown.** The next lever is a
converged fit (T=200 is 504 h on CPU; a GPU fit) or a higher-rate o/down allocation, not a T=2 re-fit
and not a kernel change. An earlier T=2 campaign stopped at **12 of 196 sections** for cost and host
reasons (`bench/cpu/1794835100-nq7b-t2-abort.md`). W2's ternary activation, opened by the agent
rather than requested, is not a candidate: **77,741,390 PPL**, 1,065,125× the same-binary q8act
control (`1794834226-ternary-ppl.md`).

The 0.6B layer-0 1.174 screen did **not** measure the rate: both refits there used the example's
**column-RMS** preconditioner, while the 57.045 control came from the reference pipeline's
**activation Gram diagonal**. At the control's own rate the column-RMS arm is 1,225,029, and at
1.174 it is 185,327; the preconditioner, not the rate or a tree regression, owns that gap
(erratum in `1794834689-nq06-l0-1174.md`). D6 is **untested**. The next measurement is a 0.977441
column-RMS→diagonal fit that reproduces ≈57.045, then the same driver at 1.174.

The control is decisive: **layer 0 is uniquely sensitive**. At 0.978 b/w, layer 0 alone is
**57.045** (3.15× Q4), layer 13 alone **18.441** (1.018×), layer 27 alone **20.581** (1.136×). The
partial pack is sound; the full 196-layer v3 pack (74.297) is the layer-0 loss times **1.30** for the
other 27 layers. If layer 0 is served at a higher rate and the rest at the same rate, the full pack
could be ≈24 (1.32× Q4), inside Q2. The budget is cheap: layer 0 is 233M of the 7B's 6.525G weights
(3.6%), so raising it from 1.17 to 3.0 b/w costs 0.066 b/w on the average and the bulk drops only
from 1.174 to 1.108. But layer 0's `gate`/`up`/`down` are 18944-wide, and the dense sign-factor cap
there is **1.19 b/w**; a 2.0 or 3.0 b/w layer 0 needs a residual mechanism, which `nq_fit` does not
write. The executable layer-0 screen is therefore the 0.6B at 1.174 (its caps are 1.33 for `down`
and 2.0 for `q`), on the activation diagonal, before any 7B layer-0 plan.

W2 was opened by the agent, not requested. It was cheap on 0.6B: same v3 pack (0.978 b/w), replace
the q8 activation reader with the ternary cell, score 2,044 and 8,176 tokens against the q8 reader
on the same pack. The kill criterion was explicit: at the residual fraction that keeps the per-row
cost ≤8 instructions / 64 terms, PPL must be **within 1.25× of the q8 reader**; otherwise the cell
is refused. **Result: 77,741,390 against 72.988013, 1,065,125× — refused by a factor of ~850,000.**
The 1.5× Q4 bar is not decided here: the v3 pack itself is 2.52× Q4, so the activation cell cannot
close that gap. The 1.5× bar is tested on the 7B 1.174 b/w pack. The agent-opened W2 path is
retired; its receipt remains a negative result.

## 3. Workstreams

| id | work | owner | state | gate / artifact |
|---|---|---|---|---|
| W0 | `fro2` rescale: 7B `down`/`gate` fit | main | **fix confirmed end-to-end**: `fro2_carries_a_row_past_the_word_range` 1 passed; dead-coordinate 0.6B `q` fits/packs at 1.1543 b/w; 7B fit OOM at every rank | unit test + `/tmp/opencode/diag-synth/q.nqp`; box fit |
| W1 | 7B pack + PPL vs Q4_K_M, per-layer allocation | campaign agent | **pack measured, Q2 open**: 196 sections at 1.098871 b/w, 6.854 h, $0.653; Q4 9.298465; NQ refuses because the T=2 v/o factors are unconverged (o/down gain 10^4–10^6); next a converged/GPU fit or higher-rate o/down | no Q2; `1794862546` |
| W2 | ternary cell: quantizer, twin, PPL | design agent → K | **exploratory, not requested; done and refuted**: PPL 77,741,390 vs q8act 72.988013, 1,065,125× | `bench/cpu/1794834226-ternary-ppl.md`; retained as a negative result, not a product path |
| W3 | assembly audit: q8 loop ports, ternary loop count | asm agent | **done**: q8 12 instr/32; ternary port-bound | `bench/cpu/1794822551-nq-loop-ports.md` |
| W5 | ternary cell microbenchmark | micro agent | **done**: 1.95 cycles/group, 1.02 popcnt/cycle | `bench/cpu/1794828200-ternary-cell-micro.md` |
| W4 | formal ladder: extraction, twin, theorem, axioms | proof agent | **done**: drift confirmed, ladder written | `bench/notes/nq-proof-ladder.md` |
| W5 | 20 tok/s kernel: AVX2 `popcnt` cell + pool | K | **cancelled** by W2; int4 14–18 tok/s is the CPU fallback | — |
| W6 | 200k context / C1 | deferred | not started | KV ≤1.5 bits/value |
| W7 | serve/Intent integration | deferred | 0.6B loop works | I1 receipt |

W2 was agent-opened, not requested; its outputs are kept only as a negative result. In order: the
quantizer (per-64 block scale, zero threshold, residual policy), the safe twin
`ternary_dot(w, p, n, len) -> i128` in safe Rust, the equality test over lengths
{0,1,7,8,9,31,32,33,63,64,65,127,128,129,1000,1023}, the Lean statement, the 0.6B PPL arms.

W4's ladder, bottom up: `Cell<i64>` word kernel → `limb` placement/product → `matmat` → `nq`
factorization (`fro2`, `spd_inverse`, `export`) → the activation reader → the scale epilogue. The
popcount identity is not in Aeneas's std model, so the extracted body is the bit-recursive twin and
`count_ones` is the intrinsic fast path with a tested equality. That is the same shape as
`sign_dot_i64` / `sign_dot_avx2`.

## 4. Evidence ledger

| fact | value | artifact |
|---|---|---|
| host | i5-10210U 4C/8T, 15 W, 6.6 GiB, no usable GPU | — |
| measured DRAM | 27.97 GB/s (42.656 spec) | `1794811020` |
| pool scaling | 2.44–2.87× one core | `1794813322` |
| 7B weights / terms per token | 6,525,288,448 / 7,613,448,192 | `1794781140` |
| 1.174 b/w pack | 957,615,232 B | `1794781140` |
| Q4_K_M bytes / ceiling | 4.3706 GB, 6.40 tok/s | `1794811020` |
| q8 loop | 12 static instructions / 32 terms, 2.1 IPC measured | `1794822551` |
| ternary cell, static | 11 instructions / 64 terms; front-end model 24.0 tok/s at 2.1 IPC | `1794822551` |
| ternary cell, measured | 1.95–1.98 cycles/group; 120.4–123.0 G terms/s/core; 63.0 tok/s 4c at 3.75 GHz | `1794828200-ternary-cell-micro.md` |
| GPU 1-bit carrier read | 539.5 GB/s (CUDA fallback, RTX 2080 Ti, 87.6% of 616 spec); 7B 1-bit ceiling **540.5 tok/s**, 27× 20 | `bench/gpu/1794831271-gpu-gemv.md` |
| Vulkan on rented hosts | ICD `-3` on two 2080 Ti hosts, CDI on a 3090/A4000; no `src/gpu` kernel ran; V1 open | same |
| ternary cell, scalar | 84.5–88.3 G terms/s/core; 2.72–2.84 cycles/group | same |
| 0.6B v3 PPL | 54.806671 vs 21.715296 at 8,176 | `1794822341-nq06-ppl-vs-q4.md` |
| 0.6B v3 rate | 21.42 tok/s (width 4) | `1794813322` |
| vast.ai account | `lucasrosewinters@gmail.com` (139166), **$5.00** | `vastai show user` |
| old vast.ai key | account 841512, $0.278; replaced | `vastai show user` |
| box `no_such_ask` | CPU-only asks are unrentable; a fresh GPU `ask_contract_id` creates | `bench/cpu/1794825968-nq7b-box-probe.md` |
| 7B `down` T=2 on box | 1,089.98 s (481.7 s/iter, Broadwell, no VNNI); **fro2 Range gone** | same |
| 7B T=200 bill | VNNI host 2.52 h/iteration; **504 h at T=200**, not 18 h (per-layer figure); T=2 is 5.04 h | `1794828501` erratum, `tools/fit-budget.py` |

## 5. Blockers

1. **The 7B T=200 fit is 504 h, not a rented-hour problem.** The laptop's `down` fit OOMs at every
   rank (the matrix is copied), and local fits longer than ~50 min do not survive the shared lane.
   The AVX-VNNI box measures 94 s per `down` iteration; one layer's seven kinds are 324 s, 196
   linears are 2.52 h/iteration, and T=200 is **504 h** (the earlier 18 h figure was per layer).
   `tools/fit-budget.py` refuses it. The feasible arms are a working i9 at ~$0.8 for T=2 and a GPU
   fit.
2. **The decisive 7B quality point does not exist yet.** A full 196-linear T=2 campaign stopped at
   **12 of 196 sections**: the $0.222/h host's cap buys ~4 h against a 5.2–6.3 h fit, one host
   rebooted twice, the fallback hosts refused SSH. It is an upper bound on the converged ratio, not
   a T=400 verdict. The 3.0 b/w layer-0 arm is not executable by `nq_fit`: the dense sign-factor cap
   for 3584×18944 is 1.19 b/w and the tool writes no residual. The 0.6B layer-0 1.174 arm measured
   the example's column-RMS path, not the reference activation diagonal, so D6 is untested.
3. **W2's ternary activation was agent-opened, not requested, and is quality-refuted.** 0.5416 relative dot RMS
   against the q8 reader's 0.0119 (45.7×; 2086× SSE), designed cell 51.6×, best 4/64 residual 24.5×;
   matching q8 needs ~52/64 exact terms (`bench/notes/ternary-act-proxy.md`). The formal decider ran:
   on the 0.6B v3 pack the ternary reader scores **77,741,390 PPL** against the same-binary q8act
   control **72.988013**, **1,065,125×** (+13.88 nats/token), on the same model, pack, Q6_K head,
   2,044 tokens and ctx 512 (`bench/cpu/1794834226-ternary-ppl.md`). The 63 tok/s cell is a rate
   ceiling only; the CPU 20 tok/s route is int4 at 14–18 or a new carrier, not this one.
4. **Vulkan initializes; the first kernel returns zeros.** On an RTX 3060 with proprietary driver
   580.173.02 and the X/GL userspace, `vulkaninfo` lists the NVIDIA device and `Gpu::new()` selects
   it; the example compiles and submits dispatches, but the decode GEMV writes `0e0` on rows 0,
   1792 and 3583 and panics before timing (`bench/gpu/1794836795-vulkan-bringup.md`). V1 is not
   advanced. The next step is the repository's own GPU tests (`gpu_gemv_match`, `gpu_splitk`,
   `gpu_spv_check`) on that host, then the integer face at `segs=1` to separate the builder from
   the benchmark's binding order; a debug run is underway.
5. **`cargo fmt --all -- --check` is not yet rerun end to end.** K's `nq.rs` and the four changed
   library files pass `rustfmt --check`; the temporary `tests/tmp_nq_range.rs` that failed the
   workspace check is deleted. The full check timed out at 120 s and is rerun in the background.
6. **Maintainer gate unbootstrapped** (Aeneas/Charon/Lean); extraction is stale against the source.
   W4 measured it: `proof/ApeironWord.lean` has `MAX_INNER := 512·Q8_0_QK` against the source's
   `1024·Q8_0_QK`; `proof/README.md` records two `regen.sh --check` exits with 415 diff hunks. The
   first missing link is `word_limb_*_denote` (placement/product), and the q8 `code_select_twin` is
   unrooted beneath it. Bootstrap needs 14 GB RAM (Aeneas peaks 13.7 GB); this host has 6.7 GiB.

## 6. Agent roster

| agent | session | owns | returns |
|---|---|---|---|
| K | `ses_e1c7b5d42ffeDJDgv7LiNP3v7b` | `src/engine/kern/nq.rs` q8 A/B | rate + fmt |
| Q | `ses_e1c7b5d41ffehLT7UK9Ag4NYnA` | PPL receipt | `bench/cpu/*-nq06-ppl-vs-q4.md` |
| asm | `ses_e1c42c9ddffeYty2Gs2F9zIuh5` | W3 | `1794822551-nq-loop-ports.md` |
| cell | `ses_e1c41d4e3ffdwof0lui8MLa5L7` | W2 (exploratory, not requested) | `bench/notes/trichotomy-cell.md` + twin + PPL plan |
| box | `ses_e1c42c9d7ffeTE0Mjmq2zQESu0` | W1 | 7B `down`/`gate` probe + pack |
| proof | `ses_e1c42c9d6ffel8rgxApqTz0rry` | W4 | `bench/notes/nq-proof-ladder.md` |
| micro | `ses_e1c2e8f1dffeAIBBS6OwktMh4I` | W5 | ternary cell microbenchmark |
| act | `ses_e1c2e8f1dffd5Xz375bofjkUjM` | W2 (exploratory, not requested) | real-activation ternary proxy |

Rules for every agent: heavy commands through `~/Workspace/tools/agent-guard`; no edits to a file
another owner holds; one artifact and one gate command in the return; stop when the artifact is
written.

## 7. Queue (ordered)

1. **7B, next box: a converged pack, not another T=2.** The 196-linear T=2 pack is measured at
   1.098871 b/w but has no NQ PPL: its v/o factors are unconverged and the forward refuses at the
   first norm (`1794862546`). **Preflight:** fit layer 0, score an 8-token PPL on the one-layer pack;
   a nonzero exit or an `admm_loss` outside the reference band stops the campaign. Then T=200 on a
   GPU (CPU is 504 h and `tools/fit-budget.py` refuses it), or a higher-rate v/o/down allocation if
   convergence alone does not bound the gain. `--diag-dir diag-c6x29` is mandatory. PPL at ctx 512
   against the same Q4_K_M head, then ctx 2,048 if ≤1.5×.
2. **0.6B D6, reference path:** first reproduce 57.045 at 0.977441 with the activation Gram
   diagonal (needs `~/kld08/calib.c40.kld` or a `calib_diag_write` run — not local), then fit 1.174.
   The column-RMS arm is refused for quality points; D6 is untested.
3. **Box, running:** Vulkan bring-up on an AMD/Mesa host (RADV) and an NVIDIA proprietary-driver
   image, to test whether the ICD failure is host-specific. Success is `vulkaninfo` plus
   `gpu_gemv_bench --quick`; a failure is a receipt with the next lever. $0.35 cap.
4. **Retired — W2 ternary activation (exploratory, not requested):** 77,741,390 PPL vs 72.988013
   q8act, 1,065,125×, on the 0.6B v3 pack (`bench/cpu/1794834226-ternary-ppl.md`); the receipt is
   kept as a negative result. The CPU product is the **int4 selector** at 14–18 tok/s (above the 9.6
   minimum, below 20); 20+ stays V1 unless a new carrier appears.
5. On the 7B result: if the pack is >1.5× Q4, report the factor and the layer-0/bulk split; the next
   lever is a higher average rate or a different carrier, not a re-fit.
6. Re-run fmt / check-claims / book-check once the guard frees; no local fit longer than ~50 min.

## 8. Decision log

- **D1** Weights stay ±1 at 1 bit per term; ternary weights at 1.585 b/W would break Q1. The W2
  activation face was exploratory and is refuted; it is not a product path.
- **D2** The cell is 64 terms, not 32: the popcount and the mask are one `u64`, halving the
  per-term instruction count. 3,584 and 18,944 are both multiples of 64.
- **D3** The 7B quality point runs before the fast kernel is built. A 20 tok/s kernel over a pack
  that misses Q2 closes nothing.
- **D4** Fits go to the box when the box exists; the laptop runs only the probes the guard admits.
- **D5** No new `unsafe` without a safe twin in the crate and an extracted equality test; a new
  carrier core is safe Rust first, the intrinsic second.
- **D6** Bits are allocated per layer, not uniformly. Layer 0 alone is 3.15× Q4 at 0.978 b/w and the
  rest of the 0.6B pack is 1.30×; layer 0 is 3.6% of the 7B's weights, so it can be served at 3 b/w
  for 0.066 b/W on the average. Q1 is the pack's average, not every layer's rate. The 3.0 b/w arm
  needs a residual mechanism: the dense sign-factor cap for 3584×18944 is 1.19 b/w and `nq_fit`
  writes no residual, so the first executable test is the 0.6B layer-0 at 1.174 (running), then a
  residual layer on the 7B. A rank-only 3.0 b/w claim is refused by `tools/fit-budget.py`'s
  product-line rule in `AGENTS.md`.

## 9. What this plan does not cover

rustc/LLVM code generation, the kernel ABI, Vulkan/SPIR-V and the GPU, the `unsafe` container, the
200k context state (C1/C2), and the 0-shot half of Q2. The 20 tok/s claim is a decode claim at 2k;
200k is T1's second number and is not addressed here.
