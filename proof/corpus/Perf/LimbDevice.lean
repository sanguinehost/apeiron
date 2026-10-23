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

end Perf.LimbDevice
