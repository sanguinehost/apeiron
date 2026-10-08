import Perf.Chain
import Perf.Instances

/-!
# The measured moves as instances of the resource theorems

`Perf.Chain`, `Perf.Instances`, `Perf.Concurrency` and `Perf.Roofline` prove the resource theorems.
This module instantiates them at the numbers of `docs/evidence/perf-predictions.md` and
`docs/evidence/perf-move-ledger.md` (P3, P4/P5/P6, P10, P15, GPU-01, GPU-03), so each prediction row
of the formalisation audit carries a theorem that builds.

Every numeric constant below is a **`model`** input copied from the named prediction row, not a
measurement: `c`, `L`, the per-phase latency, the MMA peak and rate, and the P15 pair sum. The units
are stated at each definition. No theorem here produces a byte, a cycle or a microsecond; `Machine`
capacities and latencies are declared parameters (obligation 11).

## What is proved here, and by which theorem

* **P15 / P3 (dispatch fusion).** `p15_saves_L` (fusing a below-knee pair lowers the bound by exactly
  the floor `L`, `Chain.fuse_saves_L`) and `p3_unfuse_adds_L` (the reverse: splitting adds exactly
  `L`). `p15_measured` is the P15 pair sum and `p3_ratio_below_one` the measured direction.
* **P4/P5/P6 (prologue depth).** `prologue_depth_gain`: depth `4 -> 1` lowers the depth term by
  exactly `3 * latency` while `fieldMax` is unchanged; `prologue_gain_cap` caps the gain by the depth
  term share; `prologue_depth_gain_p4` is the P4 instance.
* **P10 (MMA ceiling).** `p10_mma_floor`: the node bound is at least the MMA term;
  `non_mma_move_ceiling`: while the MMA term binds, relieving any other field cannot lower the bound;
  `p10_mma_rate_ratio` states the measured `0.72` as a `Nat` ratio.
* **GPU-01 (DP4A -> int8 coopmat).** `mma_issue_ratio`: one `M16N16K32` MMA retires `64` times the
  MACs of one DP4A issue; `mma_issue_bound` is the issue-slot ratio bound on the unit machine.
* **GPU-03 (hd=128 tile refusal).** `hd128_tile_exceeds`: the tile is `98304 B > 49152 B`.

## What is not covered

No hardware speed follows from any theorem here. The counts are `model` inputs; rustc/LLVM codegen,
the GPU driver, cache replacement, clock behaviour and the pipeline's real issue scheduling are
outside every statement.
-/

namespace Perf

namespace Moves

/-! ## 1 · Dispatch fusion: P15 and P3 -/

/-- The unfused bound of a pair of dependent dispatches: the sum of the two per-dispatch floors. -/
def unfusedPair (c L a b : Nat) : Nat := Chain.dispLB c L a + Chain.dispLB c L b

/-- The fused bound of the pair: one dispatch carrying the summed demand. -/
def fusedPair (c L a b : Nat) : Nat := Chain.dispLB c L (a + b)

/-- The P15 pair sum of medians, unfused, in units of 0.1 us (P15 row: `638.6 us`). `model`. -/
def p15Unfused : Nat := 6386

/-- The P15 fused node, in units of 0.1 us (P15 row: `258.6 us`). `model`. -/
def p15Fused : Nat := 2586

/-- The prefill-side per-fused-pair floor the P15 saving removes, in units of 0.1 us. It is a stated
`model` constant, not the decode-side `Chain.decL`: the saving is over 24 fused pairs per prefill and
is dominated by the `478 MB` intermediate round trip, so `decL = 4 us` does not apply. -/
def p15L : Nat := 3800

/-- **P15 (swiglu + quant fused).** Fusing a below-knee pair of dependent dispatches lowers the bound
by exactly the floor `L`. Instance of `Chain.fuse_saves_L`; the physical reading is that a dispatch
boundary drains the machine and writes then re-reads the intermediate. -/
theorem p15_saves_L (c L a b : Nat) (hc : 0 < c) (h : a + b ≤ c * L) :
    fusedPair c L a b = unfusedPair c L a b - L := by
  unfold fusedPair unfusedPair
  have hf := Chain.fuse_saves_L c L a b hc h
  omega

/-- The P15 measured pair (`638.6 us -> 258.6 us`, units 0.1 us) saves `380.0 us`, the stated floor. -/
theorem p15_measured : p15Fused = p15Unfused - p15L := by decide

/-- **P3 (unfuse, refuted).** Splitting a below-knee pair into two dispatches raises the bound by
exactly `L`: the reverse of `Chain.fuse_saves_L`. The measured tg128 ratio `0.984` (after/before,
`p3MeasuredRatio`) is below one, the sign the model predicts; the prediction's `>1` is refuted. -/
theorem p3_unfuse_adds_L (c L a b : Nat) (hc : 0 < c) (h : a + b ≤ c * L) :
    unfusedPair c L a b = fusedPair c L a b + L := by
  unfold unfusedPair fusedPair
  exact Chain.fuse_saves_L c L a b hc h

/-- The P3 measured tg128 ratio after/before, in thousandths (P3 row: `0.984`). `model`. -/
def p3MeasuredRatio : Nat := 984

/-- The P3 measured direction: the unfused arm is slower (`0.984 < 1`), consistent with the sign. -/
theorem p3_ratio_below_one : p3MeasuredRatio < 1000 := by decide

#guard p15Unfused - p15Fused = p15L
#guard Chain.dispLB 2 1 1 + Chain.dispLB 2 1 1 = Chain.dispLB 2 1 2 + 1

/-! ## 2 · Prologue depth: P4/P5/P6 -/

/-- A demand record with field demand `dem` and serial chain depth `d`. -/
def prologueRes (dem : ResField → Nat) (d : Nat) : Res := ⟨dem, d⟩

/-- The depth axis does not enter `fieldMax`: only the field demand does. -/
theorem fieldMax_prologueRes (m : Machine) (dem : ResField → Nat) (d d' : Nat) :
    fieldMax m (prologueRes dem d) = fieldMax m (prologueRes dem d') := rfl

/-- `timeLB` of a `prologueRes` is the field maximum against the depth term. -/
theorem timeLB_prologueRes (m : Machine) (dem : ResField → Nat) (d : Nat) :
    timeLB m (prologueRes dem d) = max (fieldMax m (prologueRes dem d)) (d * m.latency) := rfl

/-- **P4/P5/P6 (prologue depth).** Shortening a serial prologue from depth `d` to `d'` (`d' ≤ d`)
lowers the depth term by exactly `(d - d') * latency` while `fieldMax` is unchanged, when the depth
term binds at `d'`. The physical reading: a dependent load chain inside a workgroup costs one latency
per iteration until the loads issue together. -/
theorem prologue_depth_gain (m : Machine) (dem : ResField → Nat) (d d' : Nat) (h : d' ≤ d)
    (hbind : fieldMax m (prologueRes dem d') ≤ d' * m.latency) :
    fieldMax m (prologueRes dem d) = fieldMax m (prologueRes dem d') ∧
    timeLB m (prologueRes dem d) = timeLB m (prologueRes dem d') + (d - d') * m.latency := by
  refine ⟨rfl, ?_⟩
  rw [timeLB_prologueRes, timeLB_prologueRes]
  have hF := fieldMax_prologueRes m dem d d'
  rw [hF]
  have hle : fieldMax m (prologueRes dem d') ≤ d * m.latency :=
    le_trans hbind (Nat.mul_le_mul_right m.latency h)
  rw [Nat.max_eq_right hle, Nat.max_eq_right hbind]
  calc d * m.latency = (d' + (d - d')) * m.latency := by rw [Nat.add_sub_of_le h]
    _ = d' * m.latency + (d - d') * m.latency := Nat.add_mul _ _ _

/-- **The depth gain is capped by the depth term share.** If the field term dominates, shortening the
prologue gains nothing beyond `(d - d') * latency`; the gain is never larger. -/
theorem prologue_gain_cap (m : Machine) (dem : ResField → Nat) (d d' : Nat) (h : d' ≤ d) :
    timeLB m (prologueRes dem d) ≤ timeLB m (prologueRes dem d') + (d - d') * m.latency := by
  rw [timeLB_prologueRes, timeLB_prologueRes]
  have hF := fieldMax_prologueRes m dem d d'
  rw [hF]
  apply Nat.max_le.mpr
  constructor
  · exact le_trans (Nat.le_max_left _ _) (Nat.le_add_right _ _)
  · have h1 : d' * m.latency ≤ max (fieldMax m (prologueRes dem d')) (d' * m.latency) :=
      Nat.le_max_right _ _
    calc d * m.latency = (d' + (d - d')) * m.latency := by rw [Nat.add_sub_of_le h]
      _ = d' * m.latency + (d - d') * m.latency := Nat.add_mul _ _ _
      _ ≤ max (fieldMax m (prologueRes dem d')) (d' * m.latency) + (d - d') * m.latency :=
          Nat.add_le_add_right h1 _

/-- The P4 model machine: one unit per field per step, per-step chain latency `150 ns`, the P4 row's
`4x0.15 us` dependent-load latency. `model`. -/
def p4Machine : Machine where
  cap := fun _ => 1
  capPos := fun _ => by omega
  latency := 150

/-- The P4 warm prologue, in ns (P4 row: `3.81 us = 3810 ns`). `model`. -/
def p4PrologueNs : Nat := 3810

/-- **The P4 instance.** Depth `4 -> 1` removes `3 * 150 ns = 450 ns` of the depth term. -/
theorem prologue_depth_gain_p4 :
    timeLB p4Machine (prologueRes (fun _ => 0) 4)
      = timeLB p4Machine (prologueRes (fun _ => 0) 1) + 3 * p4Machine.latency := by
  have h := (prologue_depth_gain p4Machine (fun _ => 0) 4 1 (by decide) (by decide)).2
  simpa using h

#guard timeLB p4Machine (prologueRes (fun _ => 0) 4) = 600
#guard timeLB p4Machine (prologueRes (fun _ => 0) 1) = 150

/-! ## 3 · The MMA ceiling: P10 -/

/-- **P10 (MMA floor).** The node time bound is at least the MMA term, for every record and machine:
`ceilDiv (demand fMma) (cap fMma) <= timeLB`. The tensor pipe's floor cannot be escaped. -/
theorem p10_mma_floor (m : Machine) (r : Res) :
    ceilDiv (r.demand fMma) (m.cap fMma) ≤ timeLB m r :=
  fieldLB_le_timeLB m r fMma

/-- The P10 MMA peak, in units of 0.1 TOPS (P10 row: `433.5 TOPS`). `model`. -/
def p10Peak : Nat := 4335

/-- The committed kernel's MMA-term rate, in units of 0.1 TOPS (P10 row: P8's `315 TOPS`). `model`. -/
def p10MmaRate : Nat := 3152

/-- **The measured `0.72` as a `Nat` ratio.** The MMA term runs at no less than `72/100` of the peak;
`72 * peak <= 100 * rate`. This bounds any MMA-rate lever, not a speed. -/
theorem p10_mma_rate_ratio : 72 * p10Peak ≤ 100 * p10MmaRate := by decide

/-- The P8 MMA term, in units of 0.1 us (P8 row: `680.6 us`). `model`. -/
def p8MmaTerm : Nat := 6806

/-- The P8 gate_up node, in units of 0.1 us (P8 row: `1860.6 us`). `model`. -/
def p8Node : Nat := 18606

/-- **The P10 floor at the P8 numbers.** The node is at least its MMA term (`680.6 <= 1860.6 us`). -/
theorem p10_mma_floor_p8 : p8MmaTerm ≤ p8Node := by decide

/-- The field maximum of a one-field record is that field's floor. -/
theorem fieldMax_singleField (m : Machine) (f : ResField) (n : Nat) :
    fieldMax m (singleField f n) = ceilDiv n (m.cap f) := by
  apply le_antisymm
  · unfold fieldMax
    refine maxVec_le (fun i => ?_)
    rw [fieldLB]
    by_cases hi : i = f
    · subst hi; rw [singleField_self]
    · rw [singleField_ne hi, Chain.ceilDiv_zero]
      exact Nat.zero_le _
  · have h := le_maxVec 9 (fun i => fieldLB m (singleField f n) i) f
    simpa [fieldMax, fieldLB, singleField_self] using h

/-- The lower bound of a one-field, zero-depth record is that field's floor. -/
theorem timeLB_singleField (m : Machine) (f : ResField) (n : Nat) :
    timeLB m (singleField f n) = ceilDiv n (m.cap f) := by
  unfold timeLB
  rw [fieldMax_singleField, singleField_depth, Nat.zero_mul, Nat.max_eq_left (Nat.zero_le _)]

/-- **The non-MMA move ceiling (P10).** While the MMA term binds, relieving any other field cannot
lower the bound: any record keeping `M` on `fMma` has a bound at least that of the MMA-only record.
Physical reading: one warp scheduler issues the shared port, so a move on a non-binding field leaves
the tensor-pipe floor in place. -/
theorem non_mma_move_ceiling (m : Machine) (M : Nat) (r' : Res)
    (hM : r'.demand fMma = M) :
    timeLB m (singleField fMma M) ≤ timeLB m r' := by
  rw [timeLB_singleField]
  calc ceilDiv M (m.cap fMma) = ceilDiv (r'.demand fMma) (m.cap fMma) := by rw [hM]
    _ = fieldLB m r' fMma := rfl
    _ ≤ timeLB m r' := fieldLB_le_timeLB m r' fMma

/-! ## 4 · Instruction width: GPU-01 -/

/-- MACs retired per `M16N16K32` int8 cooperative-matrix instruction: `16 * 16 * 32 = 8192`. -/
def mmaMacs : Nat := 16 * 16 * 32

/-- MACs retired per DP4A issue: `4` per lane-op times a `32`-lane warp = `128`. -/
def dp4aMacs : Nat := 4 * 32

/-- **GPU-01 (issue-slot ratio).** One MMA issue retires `64` times the MACs of one DP4A issue. This
is a ratio of per-instruction MACs, not a speedup claim. -/
theorem mma_issue_ratio : mmaMacs = 64 * dp4aMacs := by decide

/-- `ceilDiv a c <= K * ceilDiv a (K * c)` for `c, K > 0`: enlarging the divisor by `K` divides the
ceiling, so the instruction demand of the wider engine is at most `1/K` of the narrower one. -/
theorem ceilDiv_le_mul_ceilDiv (a c K : Nat) (hc : 0 < c) (hK : 0 < K) :
    ceilDiv a c ≤ K * ceilDiv a (K * c) := by
  have hc' : 0 < K * c := Nat.mul_pos hK hc
  apply Chain.ceilDiv_le_of_le_mul hc
  calc a ≤ ceilDiv a (K * c) * (K * c) := le_ceilDiv_mul hc'
    _ = (K * ceilDiv a (K * c)) * c := by ring

/-- The DP4A schedule for `W` MACs: `ceilDiv W 128` dot-product issues. -/
def dp4aIssueRes (W : Nat) : Res := singleField fDp4a (ceilDiv W dp4aMacs)

/-- The MMA schedule for the same `W` MACs: `ceilDiv W 8192` matrix issues. -/
def mmaIssueRes (W : Nat) : Res := singleField fMma (ceilDiv W mmaMacs)

/-- **GPU-01 (issue-slot bound).** For MAC-equal work the DP4A instruction demand is at most `64`
times the MMA instruction demand on the unit machine. The measured `2.1x` (`27.5k -> 57.0k tok/s`,
GPU-01 ledger) is far below `64x`, so the node was not issue-bound on MACs; the bound is on the
issue-slot ratio, not a speedup. -/
theorem mma_issue_bound (W : Nat) :
    timeLB unitMachine (dp4aIssueRes W) ≤ 64 * timeLB unitMachine (mmaIssueRes W) := by
  rw [dp4aIssueRes, mmaIssueRes, timeLB_unit_single, timeLB_unit_single]
  rw [mma_issue_ratio]
  exact ceilDiv_le_mul_ceilDiv W dp4aMacs 64 (by decide) (by decide)

/-! ## 5 · The hd=128 tile refusal: GPU-03 -/

/-- The hd=128 tile's key count (P1 row). `model`. -/
def hd128Keys : Nat := 64

/-- The head dimension at hd=128 (P1 row). `model`. -/
def hd128Dim : Nat := 128

/-- Bytes per f32 element. -/
def f32Bytes : Nat := 4

/-- The staged tensors of the attention tile (P1 row's `* 3`). `model`. -/
def hd128Tensors : Nat := 3

/-- The device shared-memory maximum, in bytes (P1 row: `WG_SHARED_MAX = 49152`). `model`. -/
def wgSharedMax : Nat := 49152

/-- **GPU-03 (tile refusal).** The hd=128 tile is `64 * 128 * 4 * 3 = 98304 B`, above the `49152 B`
shared maximum, so the tile is refused and the P1 one-walk gain does not transfer. `decide`-free. -/
theorem hd128_tile_exceeds :
    hd128Keys * hd128Dim * f32Bytes * hd128Tensors > wgSharedMax := by
  norm_num [hd128Keys, hd128Dim, f32Bytes, hd128Tensors, wgSharedMax]

#guard hd128Keys * hd128Dim * f32Bytes * hd128Tensors = 98304

/-! ## DECODE-Q5 (P40): the two-plane Q5 chart against the Q8 re-chart -/

/-- Global words per row of the Q8 re-chart at depth `k`: `k/4` code words plus `k/32` `dm` words,
1.125 B/w. `model` input read from `Carrier::Q8.code_words_per_row` and the per-32-block `dm`. -/
def q8RowWords (k : Nat) : Nat := k / 4 + k / 32

/-- Global words per row of the `Carrier::Q5` chart: plane A `k/8`, plane B `k/32`, `dm` `k/32`,
0.75 B/w. `model` input read from `Carrier::Q5.code_words_per_row` (`src/simt/mod.rs`). -/
def q5RowWords (k : Nat) : Nat := k / 8 + k / 32 + k / 32

/-- On every 32-aligned depth the Q5 chart streams exactly two thirds of the Q8 re-chart's words
(`k = 32 j`: `6 j` against `9 j`). -/
theorem q5_row_words_two_thirds (k : Nat) (h : 32 ∣ k) : 3 * q5RowWords k = 2 * q8RowWords k := by
  obtain ⟨j, rfl⟩ := h
  unfold q5RowWords q8RowWords
  omega

/-- **P40 (Q5 chart move).** The Q5 unpack adds ALU work, so the move is not componentwise
dominance and `timeLB_mono` does not apply. When the global-byte field binds the Q5 kernel
(`hbind`, the P38 arm A observation on gu, a measured premise), its bound is at most the Q8 kernel's
whatever the other fields do: the binding field's demand falls by a third, and every field's bound is
below the whole bound (`fieldLB_le_timeLB`). -/
theorem q5_chart_gain_when_bytes_bind (m : Machine) (r r' : Res) (rows k : Nat) (h32 : 32 ∣ k)
    (hg : r.demand fGlobal = 4 * (rows * q8RowWords k))
    (hg' : r'.demand fGlobal = 4 * (rows * q5RowWords k))
    (hbind : timeLB m r' = ceilDiv (r'.demand fGlobal) (m.cap fGlobal)) :
    timeLB m r' ≤ timeLB m r := by
  have h23 := q5_row_words_two_thirds k h32
  have hw : q5RowWords k ≤ q8RowWords k := by omega
  have hd : r'.demand fGlobal ≤ r.demand fGlobal := by
    rw [hg, hg']
    exact Nat.mul_le_mul_left 4 (Nat.mul_le_mul_left rows hw)
  rw [hbind]
  exact Nat.le_trans (ceilDiv_mono hd) (fieldLB_le_timeLB m r fGlobal)

#guard q5RowWords 896 = 168 ∧ q8RowWords 896 = 252

/-- **P41 (two-plane tile bytes).** `prefill_tile` (`src/gpu/decode/prefill.rs`) picks the 32-token
tile once a chart exceeds 32 MB; it now sizes against `chart_bytes = 4 m (code_words_per_row k + k/32)`,
which for `Carrier::Q5` is `4 m · q5RowWords k`. Every qwen05b Q5_0 group has `k = 896` and
`m ≤ 9728` (fused `gu`), so the corrected figure stays under the threshold: the tile is unchanged. -/
theorem q5_chart_under_tile_threshold (m k : Nat) (hm : m ≤ 16384) (hk : k ≤ 896) :
    4 * m * q5RowWords k ≤ 32 * 2 ^ 20 := by
  have hw : q5RowWords k ≤ 168 := by unfold q5RowWords; omega
  calc 4 * m * q5RowWords k ≤ 4 * 16384 * 168 := Nat.mul_le_mul (Nat.mul_le_mul_left 4 hm) hw
    _ ≤ 32 * 2 ^ 20 := by decide

/-- Largest |integer dot| over one 32-block for codes at most `qmax` against int8 activations. -/
def chartBlockDotBound (qmax : Nat) : Nat := 32 * qmax * 127

/-- **P42 (Q5 coopmat plane B).** `coopmat_gemm.comp` converts its int accumulator through the magic
float `1.5·2^23 + q`, exact for `|q| < 2^22`. The Q5 code (plane A nibble plus the plane-B fifth bit)
is at most 31, so one block's dot stays below `2^17`; the single-plane bound (code ≤ 255) holds too. -/
theorem q5_coopmat_dot_under_magic : chartBlockDotBound 31 < 2 ^ 17 ∧ chartBlockDotBound 255 < 2 ^ 22 := by
  unfold chartBlockDotBound; decide

/-- Global words per row of the `Carrier::Q6` chart: plane A `k/8` (low 4 bits), plane B `k/16` (high
2 bits, two words per 32-block), `dm` `k/32` (`f16(d) ‖ sc_a ‖ sc_b`), 0.875 B/w. `model` input read
from `Carrier::Q6.code_words_per_row` (`src/simt/mod.rs`). -/
def q6RowWords (k : Nat) : Nat := k / 8 + k / 16 + k / 32

/-- **P43 (Q6 chart).** On every 32-aligned depth the Q6 chart streams exactly seven ninths of the
Q8 re-chart's words (`k = 32 j`: `7 j` against `9 j`). -/
theorem q6_row_words_seven_ninths (k : Nat) (h : 32 ∣ k) : 9 * q6RowWords k = 7 * q8RowWords k := by
  obtain ⟨j, rfl⟩ := h
  unfold q6RowWords q8RowWords
  omega

#guard q6RowWords 4864 = 1064 ∧ q8RowWords 4864 = 1368

/-- **P44 (Q5 decode fold).** `Kb::dp4a_chart_fold` (`src/simt_nn/attn_dp4a.rs`) expands the four
plane-B bits of one DP4A phase with one multiply and one mask: `(nib * 0x0204_0810) & 0x1010_1010`
puts bit `i` of `nib` at bit 4 of byte `i`, with no partial-product collision, and the product never
wraps a u32. Checked over all sixteen nibbles. -/
theorem q5_spread_exact :
    (∀ n : Fin 16, (n.val * 0x02040810) &&& 0x10101010 =
      (n.val &&& 1) * 2 ^ 4 + (n.val >>> 1 &&& 1) * 2 ^ 12 + (n.val >>> 2 &&& 1) * 2 ^ 20 +
        (n.val >>> 3 &&& 1) * 2 ^ 28) ∧
    15 * 0x02040810 < 2 ^ 32 := by
  decide

/-- **The Q6 DP4A fold's signed byte and its exact block sum.** `pack_q6_k_raw` stores `u = code + 32`
in `0..63`; `(u + 0x60) ^ 0x80`, read as a two's-complement byte, is `u - 32` for every `u`, and
`u + 0x60 ≤ 0x9F` never carries into the next byte of the word. A block's integer is
`Σ_phases sc · SDot`, at most 32 codes of `|u - 32| ≤ 32` against `|x| ≤ 127` under `|sc| ≤ 128`,
which stays below `2^24`, so the int32 accumulator and its I2F conversion are exact. -/
theorem q6_signed_byte_exact :
    (∀ u : Fin 64, u.val + 0x60 < 0x100 ∧
      (((u.val + 0x60) ^^^ 0x80 : Nat) : Int) - (if (u.val + 0x60) ^^^ 0x80 < 0x80 then 0 else 0x100) =
        (u.val : Int) - 32) ∧
    32 * 128 * 32 * 127 < 2 ^ 24 := by
  decide

/-- `Σ_{k < n} f k`, left-folded. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

private theorem rsum_low (t : Nat → Int) (sa sb : Int) (m : Nat) :
    ∀ j, j ≤ m → rsum (fun k => (if k < m then sa else sb) * t k) j = sa * rsum t j := by
  intro j
  induction j with
  | zero => intro _; simp [rsum]
  | succ j ih =>
    intro h
    have hj : j < m := by omega
    simp only [rsum, ih (by omega), if_pos hj, Int.mul_add]

/-- **The Q6 cooperative-matrix half split.** Q6_K scales the two 16-code halves of a 32-block by
`sc_a` and `sc_b`; the GEMM stages a row as `[h0, 0, h1]` and runs one MMA on `[h0, 0]` and one on
`[0, h1]`, combining `sc_a·qa + sc_b·qb` in the integer ring. That combination is the per-position
scaled dot over the whole block: with `m = n = 16` the left side is `Σ_k sc(k)·v k·x k` and the
right side is `sc_a·qa + sc_b·qb`. -/
theorem q6_coopmat_half_split (t : Nat → Int) (sa sb : Int) (m n : Nat) :
    rsum (fun k => (if k < m then sa else sb) * t k) (m + n) =
      sa * rsum t m + sb * rsum (fun k => t (m + k)) n := by
  induction n with
  | zero =>
    rw [Nat.add_zero, rsum_low t sa sb m m (Nat.le_refl m)]
    simp [rsum]
  | succ n ih =>
    have hn : ¬ (m + n < m) := by omega
    show rsum _ (m + n) + (if m + n < m then sa else sb) * t (m + n) =
      sa * rsum t m + sb * (rsum (fun k => t (m + k)) n + t (m + n))
    rw [ih, if_neg hn, Int.mul_add, Int.add_assoc]

/-! ## P47: the `gu` cooperative prologue at the cooperative byte rate

`model` inputs from `DECODE-RANK` (021ab9d): the `gu` site streams `9805824` chart bytes per call;
the served row mapping runs at `537` B/ns, the cooperative mapping's microbench at `731` B/ns.
Time in ns is bytes over rate. -/

/-- `gu` chart bytes per call (P38 site table, Q4 arm), `model`. -/
def guBytes : Nat := 9805824

/-- At 537 B/ns the call takes at least 18260 ns; at 731 B/ns it fits in 13415 ns; over 24 layers
the difference is 116280 ns per token. Arithmetic on `model` inputs; no hardware rate follows. -/
theorem p47_gu_coop_floor :
    537 * 18260 ≤ guBytes ∧ guBytes ≤ 731 * 13415 ∧ 24 * (18260 - 13415) = 116280 := by
  decide

/-- FLASH-SPLIT-COMBINE (P48), `model` inputs from the FLASH-SCOPE digest at 8b57a7d: the combine
folded into the `o` gemv reads `3584 · 112` bytes per layer against the standalone combine's
`896 + 28672 + 3584`; a saving of 1670 ns per layer over 24 layers is 40080 ns per token, and on a
1341 us token that is a tg ratio of at least 1.031. Arithmetic only; no hardware rate follows. -/
theorem p48_flash_floor :
    3584 * 112 = 401408 ∧ 896 + 28672 + 3584 = 33152 ∧ 24 * 1670 = 40080 ∧
      1341 * 1000 ≤ 1031 * (1341 - 40) := by
  decide

/-- P49 (model): the qkv residual fold read once. Each of qkv's 80 row-tile workgroups re-reads the
previous layer's 32 split-K partials (`80 * 32 * 896 * 4` bytes); one standalone `resid_add` reads them
plus the stream once (`33 * 896 * 4`). The probe's per-call saving of 1880 ns over layers 1..23 is
43240 ns per token, and a 1341 us token shortened by 43 us is at least 1.03x faster. Arithmetic
only; no hardware rate follows. -/
theorem p49_resid_once_floor :
    80 * 32 * 896 * 4 = 9175040 ∧ 33 * 896 * 4 = 118272 ∧ 23 * 1880 = 43240 ∧
      103 * (1341 - 43) ≤ 100 * 1341 := by
  decide

/-- P50 (model): carrier selection. Per 32 weights the Q8 chart stores 32 code bytes and one 4-byte
scale (36 B, 1.125 B/w); the Q5_0 two-plane chart stores 16 nibble bytes, one high-bit word and one
`dm` word (24 B, 0.75 B/w); the Q6_K two-plane chart stores 16 nibble bytes, two high-bit words and
one `dm` word (28 B, 0.875 B/w). Selected groups stream 2/3 (Q5_0) and 7/9 (Q6_K) of their Q8 bytes.
Arithmetic only; no hardware rate follows. -/
theorem p50_selection_bytes :
    32 + 4 = 36 ∧ 16 + 4 + 4 = 24 ∧ 16 + 8 + 4 = 28 ∧ 3 * 24 = 2 * 36 ∧ 9 * 28 = 7 * 36 := by
  decide

/-- P51: the measured P50 means (tg128 756.7 → 866.1 tok/s, in tenths) sit inside P50's
registered ratio band [1.10, 1.22]. Arithmetic over measured values; grade stays `measured`. -/
theorem p51_sel_install_band :
    1100 * 7567 ≤ 1000 * 8661 ∧ 1000 * 8661 ≤ 1220 * 7567 := by
  decide

/-- P53 (model): ROPEFLASH removes one dispatch per layer, 150 → 126 per token over 24 layers. At
the P2 empty-dependent-dispatch floor (1.14 us, in hundredths) the floor alone is 27.36 us; on the
P50 token (1154.6 us, in tenths) that floor reaches the predicted lower ratio 1.02. Arithmetic only;
no hardware rate follows. -/
theorem p53_ropeflash_floor :
    150 - 24 = 126 ∧ 24 * 114 = 2736 ∧ 102 * (11546 - 274) ≤ 100 * 11546 := by
  decide

/-- P54: the measured P53 means (tg128 867.8 → 888.7 tok/s, in tenths) sit inside P53's
registered ratio band [1.02, 1.07]. Arithmetic over measured values; grade stays `measured`. -/
theorem p54_ropeflash_install_band :
    1020 * 8678 ≤ 1000 * 8887 ∧ 1000 * 8887 ≤ 1070 * 8678 := by
  decide

/-- P52 (model): the argmax tail pair. The lane probe priced the shipped pair (P = 256, serial
finish) at 20.53 us and the P = 2048 tree pair at 6.16 us (hundredths): one pair per token saves
14.37 us. On the P53 token (1125.2 us, in tenths) that saving reaches the predicted lower ratio
1.008 and stays under the upper 1.02. Arithmetic only; no hardware rate follows. -/
theorem p52_amax_tail :
    2053 - 616 = 1437 ∧ 1008 * (11252 - 144) ≤ 1000 * 11252 ∧
      1000 * 11252 ≤ 1020 * (11252 - 144) := by
  decide

/-- P55: the measured P52 means (tg128 872.0 → 890.1 tok/s, in tenths) clear P52's falsifier
floor 1.005, and sit 0.001 above its registered band top 1.02 (ratio 1.021). Arithmetic over
measured values; grade stays `measured`. -/
theorem p55_amaxp_install_floor :
    1005 * 8720 ≤ 1000 * 8901 ∧ 1020 * 8720 ≤ 1000 * 8901 ∧ 1000 * 8901 ≤ 1022 * 8720 := by
  decide

/-- P56: the qwen05b upload byte inventory measured by `qwen05b_upload_bytes_match_inventory`
(T1): llama.cpp stored bytes, the apeiron totals with and without plane-B selection, the dn `dm`
offset (24 layers of `896 · 4864 / 32` words), the per-group savings summing to the whole, and the
bandwidth roofline at 907 GB/s (907000 B/µs): 114 µs for the whole saving, 86 µs for gu. Arithmetic
over measured byte counts; the timing claim stays `measured`. -/
theorem p56_upload_inventory :
    246930432 + 144643072 = 391573504 ∧
    273014784 + 153151488 = 426166272 ∧
    376406016 + 153151488 = 529557504 ∧
    376406016 - 273014784 = 103391232 ∧
    24 * (896 * 4864 / 32) = 3268608 ∧
    4644864 + 7225344 + 78446592 + 13074432 = 103391232 ∧
    907000 * 113 ≤ 103391232 ∧ 103391232 ≤ 907000 * 114 ∧
    907000 * 86 ≤ 78446592 ∧ 78446592 ≤ 907000 * 87 := by
  decide

/-- P57: the per-call upload bytes of one Q6_K `ffn_down` GEMV (k = 4864, m = 896) in
`examples/dn_probe.rs`: Q6 two-plane `4 · (817152 + 136192)` and Q8 `4 · (1089536 + 136192)`
bytes. Their difference, 1089536 B, is 1.20 µs at 907 GB/s (907000 B/µs) and lies under the 1.5 µs
refutation floor, so the predicted 3.0–9.0 µs band is not a byte term. Arithmetic over the bound
byte counts; the timing verdict (refuted, +0.0047 µs) stays `measured`. -/
theorem p57_dn_byte_difference :
    4 * (817152 + 136192) = 3813376 ∧
    4 * (1089536 + 136192) = 4902912 ∧
    4902912 - 3813376 = 1089536 ∧
    907000 * 1 ≤ 1089536 ∧ 1089536 ≤ 907000 * 2 ∧
    2 * 1089536 < 3 * 907000 := by
  decide

/-- P58 (refuted) and P59: the arithmetic of the in-chain profile in centi-microseconds. The role
sums 20.15 + 25.06 + 23.27 + 24.10 + 24.53 + 12.25 + 4.84 make d = 134.20 us; the 12 Q6_K dn nodes
carry under a quarter of it (refutation clause); the mask-0 profiled rate 702.8 tok/s sits more than
5% under the unprofiled 802.3 (overhead clause); dn-Q6K − dn-Q4K is 12.28 us, against the 14.4 us
byte roofline of the 13074432 B dn difference at 907 GB/s. Arithmetic over measured values; the
timing verdict stays `measured`. -/
theorem p58_refutation_arithmetic :
    2015 + 2506 + 2327 + 2410 + 2453 + 1225 + 484 = 13420 ∧
    4 * 2453 < 13420 ∧
    7028 * 100 < 8023 * 95 ∧
    2453 - 1225 = 1228 ∧
    907000 * 14 ≤ 13074432 ∧ 13074432 ≤ 907000 * 15 := by
  decide

/-- P60: the `DEC_PROF` in-chain profiler overhead as a stamp count times a per-stamp drain cost,
and the sparse scheme's stamp arithmetic. P58 measured 176.5 µs overhead for the 127-stamp
`STAGE_ALL_COMMANDS` scheme on mask 0 (1.39 µs/stamp) and 24.1 µs on mask 8 (0.19 µs/stamp); the
sparse scheme stamps the embed, each layer's first and `dn` nodes and the five tail nodes
(1 + 24 + 24 + 5 = 54 stamped nodes, 55 stamps). At the mask-0 per-stamp cost that is 76.45 µs,
above 5 % of the 1246.5 µs mask-0 token; at the mask-8 cost it is 10.45 µs, below 5 % of the
1174.1 µs mask-8 token. Arithmetic over stated stamp counts and P58's measured per-stamp
constants; the overhead verdict stays `measured`. -/
theorem p60_overhead_arithmetic :
    1 + 24 + 24 + 5 + 1 = 55 ∧
    127 * 139 ≤ 17653 ∧ 17653 ≤ 127 * 140 ∧
    127 * 19 ≤ 2413 ∧ 2413 ≤ 127 * 20 ∧
    100 * 17653 > 5 * 124650 ∧
    100 * (55 * 139) > 5 * 124650 ∧
    100 * (55 * 19) < 5 * 117410 := by
  decide

/-- P61: the decomposition of P60's residual byte-identical delta `d` into stamp bias and a
chain-wide effect. P60 measured byte-identical d at 109.68 µs for 127 stamps (133.93 - 24.25)
and 32.31 µs for 55 stamps (56.66 - 24.35); over the 72-stamp difference that is 1.07–1.08 µs/stamp.
Two sparser schemes reduce stamps to 50 (dn and successor across 24 layers, 1 + 24 + 23 + 1 + 1 = 50)
and 9 (embed, layer-k dn and successor, 5 tail nodes, 1 + 1 + 1 + 5 + 1 = 9). The dn-Q6K carrier term
is 2.03–2.04 µs per layer across the 12 Q6_K layers (24.4 µs total). Arithmetic over stated stamp counts
and measured constants; the split verdict stays `measured`. -/
theorem p61_stamp_bias_arithmetic :
    1 + 24 + 23 + 1 + 1 = 50 ∧
    1 + 1 + 1 + 5 + 1 = 9 ∧
    13393 - 2425 = 10968 ∧
    5666 - 2435 = 3231 ∧
    10968 - 3231 = 7737 ∧
    127 - 55 = 72 ∧
    72 * 107 ≤ 7737 ∧ 7737 ≤ 72 * 108 ∧
    12 * 203 ≤ 2440 ∧ 2440 ≤ 12 * 204 ∧
    3231 < 55 * 108 := by
  decide

/-- P62: attribution of the tg128 gap between apeiron and CUDA llama.cpp.
Constants and bounds:
(a) qkv deficit against CUDA ≥ 100 µs/token (refutation floor: 50 µs);
    at ≥ 100 µs, qkv deficit is ≥ 25% of the 364 µs gap (1118 - 754 = 364);
(b) CUDA kernel-sum per token ≥ 90% of wall (refutation floor: 80%);
(c) apeiron idle fraction ≥ 10% of unprofiled wall.
Decide on small literals only; timing verdicts stay `measured`. -/
theorem p62_gap_attribution_bounds :
    1118 - 754 = 364 ∧
    50 < 100 ∧
    4 * 100 > 364 ∧
    80 < 90 ∧ 90 ≤ 100 ∧
    10 ≤ 100 := by
  decide

/-- P63: the barrier drain as the lever to 1.5× CUDA tg128.
The gap is 1118 − 754 = 364 µs/token. At ≥ 3 µs recovered per dispatch over 126 dispatches the
drains carry ≥ 378 µs, which covers the gap; prediction (a) puts the `DEC_BAR=9` floor at ≤ 800 µs,
so ≥ 290 µs of the 1093.6 µs token (written ×10 as 10936); refutation (a) at 950 µs leaves < 144 µs.
Decide on small literals only; timing verdicts stay `measured`. -/
theorem p63_drain_ceiling :
    1118 - 754 = 364 ∧
    364 < 126 * 3 ∧
    10936 - 8000 ≥ 2900 ∧
    10936 - 9500 < 1440 := by
  decide

end Moves

end Perf
