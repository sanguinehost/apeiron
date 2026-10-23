import ApeironWord

/-!
# The device limb product's route threshold (P106)

`ApeironWord` is imported for the FORM-ORPHAN reachability edge: this cost model prices a GPU
kernel that is waived from extraction (`src/gpu/limb.rs`, its SPIR-V `limb_gemm.comp`), so no
extracted definition is used here.

`engine::kern::limb` sends an `m×n · n×p` limb product of `W = m·p·n` multiply-accumulates to the
device or keeps it on the CPU. At `m·p = 1024·768` both walls are linear in `W`:

    routed(W) = F + b·W        cpu(W) = c·W

`F` is the fixed cost (dispatch, upload of the planes, readback of the lo/hi planes, the pooled host
finish), `b` the device rate, `c` the CPU kernel rate. The device wins exactly when
`(c − b)·W > F`. Units: tenths of a picosecond, so every constant is a natural number.
-/

namespace Perf.LimbDevice

/-- Routed wall `F + b·W`, tenths of a picosecond. -/
def routed (f b w : Nat) : Nat := f + b * w

/-- CPU wall `c·W`, tenths of a picosecond. -/
def cpu (c w : Nat) : Nat := c * w

/-- The device is faster exactly when the CPU's per-MAC excess over the device pays the fixed cost.
    Grade `model`: arithmetic over the named quantities. -/
theorem routed_lt_cpu_iff (f b c w : Nat) (h : b ≤ c) :
    routed f b w < cpu c w ↔ f < (c - b) * w := by
  unfold routed cpu
  rw [Nat.sub_mul]
  have : b * w ≤ c * w := Nat.mul_le_mul_right w h
  omega

/-- Fixed cost 0.89 ms, fit over the `src/gpu/limb.rs` test shapes at `m·p = 1024·768`
    (n = 128, 768, 3584; `gfp.log`). Grade `measured`. -/
def fixedF : Nat := 8_900_000_000
/-- Device rate 2.1e-9 ms per MAC = 2.1 ps. Grade `measured`. -/
def rateB : Nat := 21
/-- CPU kernel rate 1.0e-8 ms per MAC = 10 ps. Grade `measured`. -/
def rateC : Nat := 100

/-- At the route floor `n = 256` (`MIN_INNER`), `W = 1024·768·256`, the device is faster. -/
theorem routes_at_min_inner : routed fixedF rateB (1024 * 768 * 256) < cpu rateC (1024 * 768 * 256) := by
  decide

/-- At `n = 128`, `W = 1024·768·128`, the CPU is faster: the measured tie (1.10 against 1.07 ms)
    falls on the CPU side of the model, so `n = 128` stays on the CPU. -/
theorem cpu_at_half_min_inner :
    cpu rateC (1024 * 768 * 128) < routed fixedF rateB (1024 * 768 * 128) := by
  decide

/-- The break-even `n` at `m·p = 1024·768` lies between 143 and 144: `F / ((c − b)·m·p)` = 143.3. -/
theorem break_even_n :
    (rateC - rateB) * (1024 * 768 * 143) < fixedF ∧ fixedF < (rateC - rateB) * (1024 * 768 * 144) := by
  decide

/-! ## The device plane build (P107)

`src/gpu/spirv/limb_plane.comp` builds a `rows × n` limb plane in two passes over the words: pass 0
reads every word for the block exponents, pass 1 reads it again and writes `limbs` digit bytes per
lane of every 32-lane block. Its byte traffic is the words read twice plus the digits written; the
exponent atomics (`rows·⌈n/32⌉` words) are excluded. -/

/-- Bytes moved by the device plane build: `2·8·rows·n` word reads, `rows·⌈n/32⌉·limbs·32` digit
    writes. Grade `model`. -/
def planeBytes (rows n limbs : Nat) : Nat := 2 * 8 * rows * n + rows * ((n + 31) / 32) * limbs * 32

/-- At the fit frame (`W_n`, 3584 × 1024, 3 limbs) the build moves 69 730 304 bytes, of which
    11 010 048 are the digit bytes `plane_build_matches_cpu` compares: 72.6 µs at 960 GB/s. -/
theorem planeBytes_fit : planeBytes 3584 1024 3 = 69_730_304 ∧ 3584 * 32 * 3 * 32 = 11_010_048 := by
  decide

/-! ## The device placement (P108)

`src/gpu/spirv/limb_place.comp` computes the shift table from the block exponents in two passes over
the `segments·blocks` entries: each pass reads `e_a[b]` and `e_b[b]` (8 bytes), the second writes one
shift word (4 bytes). It replaces the readback of the exponents, the host placement and the upload of
the meta table. -/

/-- Bytes moved by the device placement over `segs·blocks` entries. Grade `model`. -/
def placeBytes (segs blocks : Nat) : Nat := segs * blocks * (2 * 8 + 4)

/-- At the fit frame (`n = 3584`, 112 blocks, 4 segments in two groups) the placement moves 8 960
    bytes: 9.3 ns at 960 GB/s, so its wall is the dispatch, not the traffic. -/
theorem placeBytes_fit : placeBytes 4 112 = 8_960 := by
  decide

/-! ## The device finish (P109)

`src/gpu/spirv/limb_finish.comp` reads the lo and hi `i32` planes of every pair group (`8·G` bytes per
entry) and writes one `i64` (8 bytes) per entry of the `mp·pp` plane. It replaces the readback of the
`2·G` planes and the pooled host finish. -/

/-- Bytes moved by the device finish over a `plane`-entry product of `groups` pair groups.
    Grade `model`. -/
def finishBytes (plane groups : Nat) : Nat := plane * (8 * groups + 8)

/-- At the fit frame (`W·Z`, `mp·pp = 1024·768`, two pair groups) the finish moves 18 874 368 bytes:
    19.7 µs at 960 GB/s. -/
theorem finishBytes_fit : finishBytes (1024 * 768) 2 = 18_874_368 := by
  decide

/-- Host↔device bytes of one resident `x_update` on a `k × n` target at rank `r`: up `[rho, reg]`
    (16), `Z_x`, `Z`, `U` and `H⁻¹` in binary64; down the 64-byte flag block, `hw` and `Fᵀ`. Every
    product, plane and map stays on the device. Grade `model`. -/
def admmXBytes (k n r : Nat) : Nat :=
  16 + 8 * (n * r + 2 * k * r + r * r) + 64 + 8 * (r * r + k * r)

/-- Side A of the blk.23 `ffn_down` fit (`k = 3584`, `n = 1024`, `r = 768`) moves 81 789 008 bytes per
    step: 3.3 ms at 25 GB/s over PCIe. -/
theorem admmXBytes_fit : admmXBytes 3584 1024 768 = 81_789_008 := by
  decide

/-! ## The resident `rank1` (P111)

`src/gpu/limb/rank1.rs` uploads the `rows × cols` target and the start vector of length `ls` in
binary64 and reads back the 1024-byte flag block and the `rows × cols` sign-product output; the power
steps, norms and planes stay on the device. -/

/-- Host↔device bytes of one resident `rank1`. Grade `model`. -/
def rank1Bytes (rows cols ls : Nat) : Nat := 8 * rows * cols + 8 * ls + 1024 + 8 * rows * cols

/-- Side A of the blk.23 `ffn_down` fit (3584 × 768, start on the 768 columns) moves 44 047 360 bytes
    and side B (1024 × 768, start on the 1024 rows) 12 592 128: 1.76 and 0.50 ms at 25 GB/s over PCIe,
    against 10.0 and 4.3 ms measured per call. -/
theorem rank1Bytes_fit :
    rank1Bytes 3584 768 768 = 44_047_360 ∧ rank1Bytes 1024 768 1024 = 12_592_128 := by
  decide

/-! ## The Schur inverse on the device, priced and not built (P112)

`learn::nanoquant::spd_inverse` at order 768 recurses to 32 bases of order 24 through 31 internal
nodes, each with four kernel products of half order `s`. Measured on the RTX 5080
(`gpu::limb::tests::device_unit_costs`): an empty `submit_wait` 19.9 µs; two plane builds plus one
product at `s` = 384, 192, 96, 48, 24: 457.4, 86.9, 37.5, 30.7, 29.2 µs. A device-resident inverse
pays every product there, three round trips per internal node (child inverse in, products out, the
base on the host) and about 0.5 ms of elementwise maps. Units: tenths of a microsecond. -/

/-- Products of one device-resident order-768 inverse: four per node, 1, 2, 4, 8 and 16 nodes at
    `s` = 384, 192, 96, 48, 24. Grade `model`. -/
def schurDeviceProducts : Nat := 4 * (4574 + 2 * 869 + 4 * 375 + 8 * 307 + 16 * 292)

/-- Projected wall: products, 31 · 3 round trips at 19.9 µs, 0.5 ms of maps. Grade `model`. -/
def schurDevice : Nat := schurDeviceProducts + 31 * 3 * 199 + 5000

/-- The projection is 8.33 ms against 10.21 ms measured for the pooled CPU inverse: 1.88 ms per call,
    and over the fit's 200 calls per 100 iterations 0.38 s, under 6% of its 6.8 s ADMM wall. -/
theorem schurDevice_fit :
    schurDeviceProducts = 59_760 ∧ schurDevice = 83_267 ∧ schurDevice < 102_100 ∧
    100 * (200 * (102_100 - schurDevice)) < 6 * 68_000_000 := by
  decide

/-! ## The two `x`-updates on two threads (P113)

`learn::nanoquant::factorize` computes `A`'s and `B`'s `x`-update from the previous `Z` and `U` only,
so the two run on two threads. Measured split per iteration at `629767bc` (`w6-split.txt`): `x_A`
22.20 ms, `x_B` 18.41 ms, each holding one 10.21 ms host `spd_inverse`; the rest is device work,
staging and readback under the device mutex. Overlap hides at most the smaller of the host and device
totals, and the pair cannot finish before the larger. Units: tenths of a microsecond. -/

/-- The serial `x` pair per iteration: 22.20 + 18.41 ms. Grade `measured`. -/
def xPairSerial : Nat := 222_000 + 184_100

/-- The pair's host inverses: two at 10.21 ms. Grade `measured`. -/
def xPairHost : Nat := 2 * 102_100

/-- The pair's device, staging and readback time: the serial pair less its inverses. -/
def xPairDevice : Nat := xPairSerial - xPairHost

/-- Measured saving per iteration: ADMM 6.40 → 5.63 s (means of 3 interleaved runs) over 100. -/
def xPairSaved : Nat := 76_700

/-- The saving is under its bound, the smaller of 20.42 ms host and 20.19 ms device, and 12.52 ms of
    the pair stays above the larger, the pair's floor under any overlap of these two stages. -/
theorem xPairOverlap_fit :
    xPairSerial = 406_100 ∧ xPairDevice = 201_900 ∧ xPairSaved ≤ min xPairHost xPairDevice ∧
    xPairSerial - xPairSaved - max xPairHost xPairDevice = 125_200 := by
  decide

/-! ## Each side's whole step on its own thread (P114)

At `4be01388` the two `rank1`s, `T = X + U` and the duals ran after the `x` pair joined. Each side's
`rank1`, `T` and dual read only its own `X` and `U`, so `factorize` runs the whole side step on the
side's thread, with both `rank1` starts drawn first in the serial order. Measured split per call at
`4be01388` (`w6-r1split.txt`): `x_A` 31.76 ms and `x_B` 26.54 ms (their inverses contend: 14.72 and
15.22 ms against 10.21 alone), `rank1` A 9.99 ms, `rank1` B 4.28 ms. `T` 2.84 ms and the duals
2.45 ms split by element count, `B`'s share `n/(o + n)` = 1024/4608. With `A`'s chain the longer one, the
iteration saves at most the work moved off it, `B`'s `rank1`, `T` and dual. Units: tenths of a
microsecond. -/

/-- `B`'s `rank1` per call: 0.30 staging + 3.59 device + 0.39 readback ms. Grade `measured`. -/
def sideRank1B : Nat := 3_000 + 35_900 + 3_900

/-- `B`'s share of `T` and the duals, (2.84 + 2.45) ms · 1024/4608. Grade `model`. -/
def sideHostB : Nat := (28_400 + 24_500) * 1024 / 4608

/-- Measured saving per iteration: ADMM 5.67 → 5.20 s (means of 3 interleaved runs) over 100. -/
def sideSaved : Nat := 46_700

/-- The saving is under the 5.46 ms of work moved off `A`'s chain; the 0.79 ms it falls short is the
two chains' contention for the device and the pool. -/
theorem sideStep_fit :
    sideRank1B + sideHostB = 54_555 ∧ sideSaved ≤ sideRank1B + sideHostB ∧
    sideRank1B + sideHostB - sideSaved = 7_855 := by
  decide

/-! ## The iterate resident between outer iterations (P115)

At `7ddb8eaa` every step staged `Z_x`, `Z`, `U` and `rank1`'s target to the device and read `Fᵀ` and
`rank1`'s output back, and the host formed `T = X + U` and the dual. With `[Z_x, Z, U, X]` resident,
`cb1` uploads only `[rho, reg]`, `cb2` and `rank1` read nothing back, `rank1`'s target is formed on the
device (`rank1_map.comp` mode 6) and the dual and the `Z` exchange run in one commit buffer (mode 7
and two copies). Moved off `A`'s chain per iteration, read from the device timestamps at `7ddb8eaa`
(`w6-gpuprof.txt`) and the host split at `4be01388` (`w6-r1split.txt`). Units: tenths of a
microsecond. -/

/-- `A`'s device transfers removed: `cb1` upload 4.534, `cb2` readback 1.838, `rank1` upload 1.791
and readback 1.698 ms. Grade `measured`. -/
def residentDevA : Nat := 45_340 + 18_380 + 17_910 + 16_980

/-- `A`'s host staging and readback removed: `x` stage 3.31 and back 0.98, `rank1` stage 0.92 and
back 1.02 ms. Grade `measured`. -/
def residentHostA : Nat := 33_100 + 9_800 + 9_200 + 10_200

/-- `A`'s share of `T` and the duals, (2.84 + 2.45) ms · 3584/4608. Grade `model`. -/
def residentTDualA : Nat := (28_400 + 24_500) * 3584 / 4608

/-- Measured saving per iteration: ADMM 5.37 → 3.43 s (means of 3 interleaved runs, arm 0
`APEIRON_ADMM_RESIDENT=0`) over 100. -/
def residentSaved : Nat := 194_000

/-- The saving is under the 20.21 ms of work moved off `A`'s chain, and falls 0.81 ms short of it;
34.3 ms per iteration remains. -/
theorem residentIterate_fit :
    residentDevA + residentHostA + residentTDualA = 202_054 ∧
    residentSaved ≤ residentDevA + residentHostA + residentTDualA ∧
    residentDevA + residentHostA + residentTDualA - residentSaved = 8_054 := by
  decide

/-! ## The padded tail block (P116)

`DigitsTask::body` splits a tail block (`hi − lo < 32` words) zero-padded to a full block in one
`split_block` call instead of lane by lane. Units: tenths of a microsecond. -/

/-- Order-768 `spd_inverse` saving: 10.06 → 9.33 ms (means of 3 interleaved pairs,
`digits-pad.log`). Grade `measured`. -/
def padInvSaved : Nat := 100_600 - 93_300

/-- ADMM saving per iteration: 3.50 → 3.33 s (means of 3 interleaved pairs, `t10pad-ab.log`) over
100. Grade `measured`; the wall prints at 0.1 s, so ±0.33 ms per iteration. -/
def padAdmmSaved : Nat := 17_000

/-- Two inverses per iteration account for 1.46 ms of the 1.70 ms saved; 0.24 ms, inside the
wall's print resolution, falls to the other padded planes. -/
theorem digitsPad_fit :
    2 * padInvSaved = 14_600 ∧ 2 * padInvSaved ≤ padAdmmSaved ∧
    padAdmmSaved - 2 * padInvSaved = 2_400 := by
  decide

/-! ## The device inverse above order 96 (P117)

`gpu::limb::inverse` runs every `spd_inverse` recursion node of order above 96 on the device (block
extracts, `S`, assembly, four limb products); nodes of order 96 are read back and inverted on the
host. Units: tenths of a microsecond per ADMM iteration. -/

/-- ADMM wall per iteration by cutoff on the blk.23 `down` fit, means of 3 interleaved runs
(`t11-devinv-cut.log`): host only 3.3 s, cut 192 2.6 s, 96 2.4 s, 48 2.47 s, 32 2.9 s per 100
iterations. Grade `measured`; the wall prints at 0.1 s. -/
def devInvHost : Nat := 330_000
def devInvCut192 : Nat := 260_000
def devInvCut96 : Nat := 240_000
def devInvCut48 : Nat := 246_667
def devInvCut32 : Nat := 290_000

/-- In-fit host inverse of side `A` before the change, 17.46 ms (`t10-prof.log`). Grade
`measured`. -/
def devInvHostA : Nat := 174_600

/-- Cut 96 is the least of the five; it saves 9.0 ms per iteration, under side `A`'s 17.46 ms host
inverse, and 8.46 ms of that inverse's time is not removed from `A`'s chain (grade `model`: the
per-phase split is not re-profiled). -/
theorem deviceInverse_fit :
    devInvCut96 ≤ devInvCut48 ∧ devInvCut96 ≤ devInvCut192 ∧ devInvCut96 ≤ devInvCut32 ∧
    devInvHost - devInvCut96 = 90_000 ∧ devInvHost - devInvCut96 ≤ devInvHostA ∧
    devInvHostA - (devInvHost - devInvCut96) = 84_600 := by
  decide

/-! ## Split block loop for thin limb products (P118)

`limb_gemm.comp` splits each tile's block loop across `Product::splits` workgroups; the finish sums a
group's splits in 64 bits before rounding. Units: tenths of a microsecond per ADMM iteration. -/

/-- ADMM wall per iteration on the blk.23 `down` fit, means of 3 interleaved runs (`t13-ab.log`):
one workgroup per tile 2.5 / 2.4 / 2.4 s, `SPLIT_WG` 192 and 768 1.9 s ×3 per 100 iterations.
Grade `measured`; the wall prints at 0.1 s. -/
def splitOne : Nat := 243_333
def split192 : Nat := 190_000
def split768 : Nat := 190_000

/-- `rank1`'s self-dot and matvec on side `A` before the change, 1.96 + 2.11 ms per iteration
(`w6-t12` timestamps). Grade `measured`. -/
def splitThinA : Nat := 40_700

/-- The split saves 5.33 ms per iteration, more than side `A`'s 4.07 ms of thin products: `B`'s
thin products and the inverse's small products split as well (grade `model`: not re-profiled). -/
theorem splitK_fit :
    split192 < splitOne ∧ split192 = split768 ∧ splitOne - split192 = 53_333 ∧
    splitThinA < splitOne - split192 := by
  decide

end Perf.LimbDevice
