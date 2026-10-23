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

end Perf.LimbDevice
