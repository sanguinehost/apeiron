import Perf.Concurrency

/-!
# Occupancy: resident workgroups from the register file and the shared-memory budget

`Perf.Concurrency` amortizes a serial chain over `conc` independent in-flight chains,
`timeLBc m r conc = max (fieldMax m r) (ceilDiv (r.depth * m.latency) conc)`, but leaves `conc` a
free parameter. On a streaming multiprocessor `conc` is not free: the register file and the
shared-memory array are finite and shared by the resident workgroups, so the resident count is
`min (floor (regfile / (threads * regs))) (floor (sharedPerSM / sharedPerWG))`. This module derives
that count from the register file and the shared-memory budget (`residentWG`, `sharedWG`,
`residentMin`), proves the register bound's quantisation step (`residentWG_antitone`,
`residentWG_step`), bounds the occupancy gain on the latency term (`occupancy_gain_le`), and records
the P12 occupancy ceiling (`occupancy_ceiling`). It backfills P11, P12, P2 and P9 of
`docs/epics/resource-guided-evolution.md` (formalisation audit).

## What is proved here and what is not

* **Proved** under the closed axiom cone `{propext, Classical.choice, Quot.sound}` (the same cone as
  `Perf.Roofline` and `Perf.Concurrency`, to be confirmed by the lead's `#audit` lines):
  `residentWG_antitone` and `residentWG_step` (the register-file bound is antitone in the per-thread
  register count and drops across the exact boundary `regBoundary` — `>= k` up to it, `<= k - 1`
  above it); `residentMin_le_left` / `residentMin_le_right` (registers and shared memory each cap
  residency); the P12 register instances and the P11 shared-memory ones; `occupancy_gain_le` (more
  resident workgroups cannot push the bound below the field maximum, and divide the latency term by
  at most `ceil (c2 / c1)`); and `occupancy_ceiling` (going from 2 to 3 resident workgroups divides the latency term by at most `3 / 2`, up to one rounding cycle).
* **Not proved.** No hardware speed follows. `regfile`, `threads`, `regs`, `sharedPerSM` and
  `sharedPerWG` are declared parameters; the register-file size `65536`, the shared-memory size
  `102400` and the measured ratios are `model` inputs read from `docs/evidence/perf-predictions.md`
  (P11, P12 outcomes). rustc/LLVM and driver codegen, the register allocation and residency the
  driver actually achieves, register-file banking, shared-memory bank conflicts and the L1/shared
  carve-out are outside every statement. `Perf.timeLBc` is a lower bound on time, not a time.
-/

namespace Perf

/-! ## 1 · The register-file bound -/

/-- The resident workgroups a register file of `regfile` registers supports for workgroups of
`threads` threads each holding `regs` registers per thread: `floor (regfile / (threads * regs))`.
Registers are a finite file shared by the resident warps, so each workgroup claims `threads * regs`
of it, and the file admits that many whole workgroups. -/
def residentWG (regfile threads regs : Nat) : Nat := regfile / (threads * regs)

/-- **The register bound is antitone in the per-thread register count.** A workgroup whose threads
hold more registers claims more of the file, so it admits no more resident workgroups:
`regs ≤ regs'` gives `residentWG regfile threads regs' ≤ residentWG regfile threads regs`. The SM's
register allocator gives a workgroup a contiguous `threads * regs` slice; a wider slice fits fewer
times. -/
theorem residentWG_antitone {regfile threads regs regs' : Nat} (hT : 0 < threads)
    (hregs : 0 < regs) (h : regs ≤ regs') :
    residentWG regfile threads regs' ≤ residentWG regfile threads regs := by
  unfold residentWG
  exact Nat.div_le_div_left (Nat.mul_le_mul_left threads h) (Nat.mul_pos hT hregs)

/-- The register count at which the resident-workgroup count falls below `k`: the largest `regs`
with `k` workgroups still resident, `floor (regfile / (k * threads))`. The count is `>= k` for
`regs` at or below this boundary and `<= k - 1` above it. -/
def regBoundary (regfile threads k : Nat) : Nat := regfile / (k * threads)

/-- **The quantisation step of the register bound.** `regBoundary regfile threads k =
floor (regfile / (k * threads))` is the exact boundary: the register bound is `>= k` for every
`regs <= regBoundary` and `<= k - 1` for every `regs` above it. The count therefore holds constant
on the interval `[regBoundary k + 1, regBoundary (k-1)]` and drops by at least one across a
boundary. Physically the register file is only partitioned in whole `threads * regs` slices, so a
few extra registers per thread change nothing until a slice no longer fits. -/
theorem residentWG_step {regfile threads k regs : Nat} (hT : 0 < threads) (hk : 0 < k)
    (hregs : 0 < regs) :
    (regs ≤ regBoundary regfile threads k → k ≤ residentWG regfile threads regs) ∧
    (regBoundary regfile threads k < regs → residentWG regfile threads regs ≤ k - 1) := by
  have hd : 0 < threads * regs := Nat.mul_pos hT hregs
  have hkT : 0 < k * threads := Nat.mul_pos hk hT
  constructor
  · intro h
    have h' : regs ≤ regfile / (k * threads) := h
    show k ≤ regfile / (threads * regs)
    rw [Nat.le_div_iff_mul_le hd]
    have hmul : (k * threads) * regs ≤ (k * threads) * (regfile / (k * threads)) :=
      Nat.mul_le_mul_left _ h'
    have hle : (k * threads) * (regfile / (k * threads)) ≤ regfile :=
      Nat.mul_div_le regfile (k * threads)
    have hkey : k * (threads * regs) ≤ regfile := by
      have h2 := Nat.le_trans hmul hle
      simpa [Nat.mul_assoc] using h2
    exact hkey
  · intro h
    have h' : regfile / (k * threads) < regs := h
    show regfile / (threads * regs) ≤ k - 1
    have hlt : regfile / (threads * regs) < k := by
      by_contra hcon
      have hle : k ≤ regfile / (threads * regs) := not_lt.mp hcon
      have hmul : k * (threads * regs) ≤ regfile := (Nat.le_div_iff_mul_le hd).mp hle
      have h2 : (k * threads) * regs ≤ regfile := by simpa [Nat.mul_assoc] using hmul
      have hmul' : regs * (k * threads) ≤ regfile := by
        rw [Nat.mul_comm regs (k * threads)]; exact h2
      have hregs_le : regs ≤ regfile / (k * threads) := (Nat.le_div_iff_mul_le hkT).mpr hmul'
      exact absurd hregs_le (not_le_of_gt h')
    exact (Nat.le_sub_one_iff_lt hk).mpr hlt

/-- **The register bound is constant between boundaries.** On the interval
`(regBoundary regfile threads (k + 1), regBoundary regfile threads k]` of per-thread register counts
the resident count is exactly `k`: the file partitions into whole `threads * regs` slices, so the
count is constant between adjacent boundaries and steps down by one across each. -/
theorem residentWG_const {regfile threads k regs : Nat} (hT : 0 < threads) (hk : 0 < k)
    (hregs : 0 < regs) (hlo : regBoundary regfile threads (k + 1) < regs)
    (hhi : regs ≤ regBoundary regfile threads k) :
    residentWG regfile threads regs = k := by
  have h1 : k ≤ residentWG regfile threads regs :=
    (residentWG_step hT hk hregs).1 hhi
  have h2 : residentWG regfile threads regs ≤ (k + 1) - 1 :=
    (residentWG_step hT (Nat.succ_pos k) hregs).2 hlo
  omega

/-- The shared-memory bound: the resident workgroups a shared-memory budget of `sharedPerSM` bytes
supports at `sharedPerWG` bytes per workgroup, `floor (sharedPerSM / sharedPerWG)`. Shared memory is
a finite on-chip array shared by the resident workgroups; each claims `sharedPerWG` bytes. -/
def sharedWG (sharedPerSM sharedPerWG : Nat) : Nat := sharedPerSM / sharedPerWG

/-- The resident workgroups per SM as the minimum of the register-file bound and the shared-memory
bound: residency is capped by whichever resource runs out first. -/
def residentMin (regfileBound sharedBound : Nat) : Nat := min regfileBound sharedBound

/-- **The register bound caps residency.** `residentMin a b ≤ a`; no more workgroups are resident
than the register file admits (`model` capacities). -/
theorem residentMin_le_left (a b : Nat) : residentMin a b ≤ a := by
  unfold residentMin; exact Nat.min_le_left a b

/-- **The shared-memory bound caps residency.** `residentMin a b ≤ b`; no more workgroups are
resident than the shared-memory budget admits. -/
theorem residentMin_le_right (a b : Nat) : residentMin a b ≤ b := by
  unfold residentMin; exact Nat.min_le_right a b

/-! ## 2 · The P12 register-bound instance

`docs/evidence/perf-predictions.md` P12 (`--stats` outcome): the committed `gemm_9728_896_Q8_cm`
kernel uses `224` registers per thread at `128` threads on a `65536`-register file, so
`residentWG 65536 128 224 = 2`; the arm `t128x32` at `154` registers fits `3`. The register file is
the same physical array for every workgroup on the SM; the counts are `model` inputs. -/

-- P12: 65536 registers per SM, 128 threads per workgroup.
#guard residentWG 65536 128 224 = 2
#guard residentWG 65536 128 154 = 3

-- The boundary for `k = 3` is `regBoundary 65536 128 3 = floor (65536 / 384) = 170`. The arms at
-- 168 and 169 registers stay at 3; 171 drops to 2.
#guard regBoundary 65536 128 3 = 170
#guard residentWG 65536 128 168 = 3
#guard residentWG 65536 128 169 = 3
#guard residentWG 65536 128 171 = 2

/-- P12 default: `224` registers at `128` threads leave `2` resident workgroups per SM. -/
theorem residentWG_default : residentWG 65536 128 224 = 2 := by norm_num [residentWG]

/-- P12 `t128x32`: `154` registers at `128` threads leave `3` resident workgroups per SM. -/
theorem residentWG_t128x32 : residentWG 65536 128 154 = 3 := by norm_num [residentWG]

/-- At `168` registers the register bound is still `3`; the boundary for `k = 3` is `170`. -/
theorem residentWG_168 : residentWG 65536 128 168 = 3 := by norm_num [residentWG]

/-- At `169` registers the register bound is still `3`. -/
theorem residentWG_169 : residentWG 65536 128 169 = 3 := by norm_num [residentWG]

/-- At `171` registers the count drops to `2`: `171 > regBoundary 65536 128 3 = 170`. -/
theorem residentWG_171 : residentWG 65536 128 171 = 2 := by norm_num [residentWG]

/-- The `k = 3` register boundary for the P12 file: `floor (65536 / (3 * 128)) = 170`. -/
theorem regBoundary_3 : regBoundary 65536 128 3 = 170 := by norm_num [regBoundary]

-- Non-vacuity of `residentWG_step`: `169 <= 170` gives `3` resident, `171 > 170` gives `<= 2`.
example : 3 ≤ residentWG 65536 128 169 :=
  (residentWG_step (regfile := 65536) (threads := 128) (k := 3) (regs := 169)
    (by norm_num) (by norm_num) (by norm_num)).1 (by norm_num [regBoundary])

example : residentWG 65536 128 171 ≤ 3 - 1 :=
  (residentWG_step (regfile := 65536) (threads := 128) (k := 3) (regs := 171)
    (by norm_num) (by norm_num) (by norm_num)).2 (by norm_num [regBoundary])

/-! ## 3 · Occupancy as latency hiding

`Perf.Concurrency.timeLBc` amortizes the serial chain `r.depth * m.latency` over `conc` resident
chains while the throughput terms `fieldMax m r` are unchanged. Raising residency from `c1` to `c2`
therefore cannot move the bound below the field maximum (a latency-hidden kernel is transport-bound,
`latency_hidden`), and it can divide the latency term by at most the residency ratio. -/

/-- **The occupancy gain is bounded.** Raising the resident-workgroup count from `c1` to `c2 >= c1`
cannot push the bound below the throughput floor `fieldMax m r`, and divides the latency term by at
most the factor `c2 / c1`: `chainLB m r c1 ≤ chainLB m r c2 * ceilDiv c2 c1`. `ceilDiv c2 c1` is the
Nat-exact form of `c2 / c1`; without the ceiling arithmetic on `ceilDiv (depth * latency) conc` it
would be exactly `c2 / c1`. More resident workgroups hide latency, never bandwidth. -/
theorem occupancy_gain_le (m : Machine) (r : Res) {c1 c2 : Nat} (hc1 : 0 < c1) (hle : c1 ≤ c2) :
    fieldMax m r ≤ timeLBc m r c2 ∧
    chainLB m r c1 ≤ chainLB m r c2 * ceilDiv c2 c1 := by
  have hc2 : 0 < c2 := Nat.lt_of_lt_of_le hc1 hle
  constructor
  · show fieldMax m r ≤ max (fieldMax m r) (chainLB m r c2)
    exact Nat.le_max_left _ _
  · unfold chainLB
    have hDle : r.depth * m.latency ≤ ceilDiv (r.depth * m.latency) c2 * c2 :=
      le_ceilDiv_mul hc2
    calc ceilDiv (r.depth * m.latency) c1
        ≤ ceilDiv (ceilDiv (r.depth * m.latency) c2 * c2) c1 := ceilDiv_mono hDle
      _ ≤ ceilDiv (r.depth * m.latency) c2 * ceilDiv c2 c1 := ceilDiv_mul_le hc1

/-- **The P12 occupancy ceiling.** Going from `2` to `3` resident workgroups divides the latency
term `ceilDiv D c` by at most `3 / 2`, up to the one-cycle rounding of `ceilDiv`:
`2 * chainLB m r 2 ≤ 3 * chainLB m r 3 + 1`. Reading: a node that is wholly latency-bound can
speed up by at most the factor `1.5` from this axis; the measured P12 node ratios `1.09`-`1.14`
(`docs/evidence/perf-predictions.md` P12) lie below it, so they do not show the axis exhausted or the
node latency-bound; they are consistent with a node that is partly latency-bound. -/
theorem occupancy_ceiling (m : Machine) (r : Res) :
    2 * chainLB m r 2 ≤ 3 * chainLB m r 3 + 1 := by
  unfold chainLB ceilDiv
  generalize r.depth * m.latency = D
  omega

/-! ## 4 · The P11 shared-memory-bound instance

`docs/evidence/perf-predictions.md` P11 (`occ48k` outcome): the device offers `102400` bytes of
shared memory per SM (2 workgroups x `49152` bytes still fit `102400`; the shared pad is free up to
`32` KB). The committed kernel claims `22016` bytes per workgroup. Residency is the minimum of the
register bound and the shared-memory bound (`residentMin`); at `48` KB per workgroup the
shared-memory bound also reaches `2`, and at the committed `22016` bytes it is `4`, above the
register bound `2` that binds. The shared-memory sizes are `model` inputs. -/

#guard sharedWG 102400 49152 = 2
#guard sharedWG 102400 22016 = 4

/-- P11: at the device maximum `49152` bytes per workgroup, `102400 / 49152 = 2` — the shared-memory
bound equals the register bound `2`, so shared memory also admits `2` workgroups. -/
theorem sharedWG_occ48k : sharedWG 102400 49152 = 2 := by norm_num [sharedWG]

/-- P11: at the committed `22016` bytes per workgroup, `102400 / 22016 = 4` — shared memory admits
`4`, so the register bound `2` is the binding one. -/
theorem sharedWG_committed : sharedWG 102400 22016 = 4 := by norm_num [sharedWG]

/-- P11 `occ48k`: shared memory limits residency to `2`, tied with the register bound. -/
theorem residentMin_occ48k :
    residentMin (residentWG 65536 128 224) (sharedWG 102400 49152) = 2 := by
  norm_num [residentMin, residentWG, sharedWG]

/-- P11 committed: the register bound `2` binds, since shared memory admits `4`. -/
theorem residentMin_committed :
    residentMin (residentWG 65536 128 224) (sharedWG 102400 22016) = 2 := by
  norm_num [residentMin, residentWG, sharedWG]

/-!
## 5 · P9 (pipeline depth): the overlap model lives in `Perf/Overlap.lean`

`docs/evidence/perf-predictions.md` P9 (outcome) issued the MMAs of a `K32` block before the previous
block's epilogues and measured the node no faster (node ratio `0.947`, pp512 `-4.6%`), so the
dependency reading of the P8 residual is refuted for those two schedules. The overlap schedule the
audit records as missing is stated and decided in `Perf/Overlap.lean`: `Res.add` sums the per-field
demands and takes the maximum chain depth, `overlapLB m A B = timeLB m (A.add B)` is the overlapped
bound against `seqLB m A B = timeLB m A + timeLB m B`, and `overlap_le_seq`, `overlap_ge_max`,
`overlap_no_gain_of_shared_binding` (a shared binding field with exact step demands gains nothing)
and `overlap_full_of_disjoint` (disjoint phases overlap fully) decide the two-phase case. The P8 MMA
phase (`fMma`) and epilogue (`fFma`/`fAlu`) carry disjoint fields, so `overlap_full_of_disjoint`
predicts the full overlap `max 680.7 469.8 us` — a gain the P9 measurement did not show, so disjoint resource fields are not sufficient for overlap
(`p8_overlap_predicted_gain` is the model prediction, refuted by the measurement). The `385.0 us`
residual is the remainder of the arm decomposition (`residual_not_in_fields`, an arithmetic identity
over the arm means) and no field carries it.
No hardware speed follows; the model is over `Nat` field demands with declared machine parameters.
-/

end Perf
