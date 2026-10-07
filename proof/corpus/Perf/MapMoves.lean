import Perf.Instances

/-!
# The kept map and count moves as instance theorems

the perf-move-ledger note records the map/count performance moves. This module backfills,
in the shape of `Perf/Moves.lean`, one theorem per row that carries a **numeric receipt** — MAP-02,
MAP-07, MAP-08, COUNT-01, COUNT-02, COUNT-03 — stated at the receipt's own numbers. Rows with no
numeric receipt get no theorem and are listed as OPEN below.

Every numeric constant is a **`model`** input copied from the named row, not a measurement: the units
are stated at each definition and the row id is in its docstring. No theorem here produces a byte, a
cycle, a nanosecond or a cache line (obligation 11).

## The binding field, and where `timeLB`/`fieldMax` do not apply

The ledger's `binding resource` is `cache-lines` for MAP-08, COUNT-01, COUNT-02 and COUNT-03;
MAP-02's row records `chain-depth` (the `Res.depth` axis, not a field) and MAP-07's records
`none-found`. **No receipt for these rows gives a resource count** — no line count, instruction count
or byte count. Each gives a ratio of measured per-operation times, so the cache-line field `fLine`
and the roofline bound `timeLB`/`fieldMax` have no measured demand to carry; each theorem below
therefore proves only the arithmetic relation the receipt states, and the binding field is named in
each docstring.

The ledger's `gain` column is the before/after ratio rounded to two decimals (`9.93 / 3.78 = 2.62698
-> 2.63`), so an integer cross-multiplication at the rounded value does not always hold; where a row
gives a before/after pair the theorem states the pair's measured direction, which is what the receipt
measures.

## Open obligations — no numeric receipt, no theorem

Each remaining map/count row is listed with the ledger's own status and reason; none is given an
invented number.

* **MAP-03** — `unattributed`, red: `CoordMap` columns `25/60 failing` against `hb_fx`, no binding
  resource named.
* **MAP-04** — `unattributed`, red: `0.06 -> 0.54` mispredicts/op bind the miss walk; no
  binding-resource receipt.
* **MAP-05** — `unattributed`, red: `benches/map.rs` baseline, `40/60` cell pairs above `1.0`, mixed.
* **MAP-06** — `cache-lines`, red: absent-miss `n=100000` dense pathological, `11215.319x` loss; a
  loss, not a kept move.
* **MAP-09** — refuted: prefetch of `vals[home]`, `none` outside spread; no numeric receipt.
* **MAP-10** — refuted: `TAGPLANE`; no numeric receipt.
* **MAP-11** — refuted: `MAP-GLANCE` variants; no numeric receipt.
* **MAP-12** — refuted: `INS` levers A and B; no numeric receipt.
* **MAP-13** — refuted: launch-path megakernel; no numeric receipt.
* **COUNT-04** — refuted: tag LUT, slot prefetch, 64 maps, interleaved ctrl+slot layout — null; no
  numeric receipt.
* **COUNT-05** — `unattributed`, red: count lookup `1.9-4.1x` behind `hb_fx`; no binding resource.

## What is not covered

No hardware speed follows from any theorem here. The ratios are `model` inputs read from the ledger
row, not re-measured; rustc/LLVM codegen, the CPU, cache replacement and clock behaviour are outside
every statement.
-/

namespace Perf

namespace MapMoves

/-! ## 1 · MAP-08 — flat spill read port (`cache-lines`) -/

/-- The canonical path's `8.14 ns/key`, in units of `0.01 ns/key`. `model`, ledger MAP-08. -/
def map08Canonical : Nat := 814

/-- The flat prototype's `1.34 ns/key`, in units of `0.01 ns/key`. `model`, ledger MAP-08. -/
def map08Prototype : Nat := 134

/-- The row's measured ratio `6.06`, in hundredths. `model`, ledger MAP-08. -/
def map08Ratio : Nat := 606

/-- **MAP-08 (flat spill read, `cache-lines`).** The prototype reads at least `6.06x` below the
canonical path: `1.34 * 6.06 <= 8.14` on the receipt's `0.01 ns/key` units. the design notes E6 /
the read-path-specialization note. -/
theorem map08_flatSpillRead :
    map08Prototype * map08Ratio ≤ map08Canonical * 100 := by decide

/-! ## 2 · MAP-02 — hashed `get_coord` (`chain-depth`) -/

/-- The absent-miss random `1e5` `apeiron/hb_fx` ratio before the move, `1.41x`, in hundredths.
`model`, ledger MAP-02. -/
def map02Before1e5 : Nat := 141

/-- The after ratio `1.01x`. `model`, ledger MAP-02. -/
def map02After1e5 : Nat := 101

/-- The `1e6` before ratio `3.32x`, in hundredths. `model`, ledger MAP-02. -/
def map02Before1e6 : Nat := 332

/-- The `1e6` after ratio `2.9x`. `model`, ledger MAP-02. -/
def map02After1e6 : Nat := 290

/-- **MAP-02 (hashed `get_coord`, `chain-depth`).** Skipping the home-`EMPTY` pre-check and loading
`window_steps` from a table lowers the `apeiron/hb_fx` absent-miss ratio at both sizes: `1.01 < 1.41`
and `2.9 < 3.32`. The row records `chain-depth`, so the binding resource is the serial-chain axis
(`Res.depth`), not a field. -/
theorem map02_hashedGetCoord :
    map02After1e5 < map02Before1e5 ∧ map02After1e6 < map02Before1e6 := by decide

/-! ## 3 · MAP-07 — subsumed incumbent parity (`none-found`) -/

/-- The parity ratio `ratio_mean 0.9757`, in ten-thousandths. `model`, ledger MAP-07 /
the map-parity-landed note. -/
def map07RatioMean : Nat := 9757

/-- The six-campaign parity ratio `0.9912`, in ten-thousandths. `model`, ledger MAP-07. -/
def map07SixCampaign : Nat := 9912

/-- Parity: an `apeiron/hb_fx` ratio of exactly `1.0`, in ten-thousandths. -/
def map07Parity : Nat := 10000

/-- **MAP-07 (subsumed parity).** The selected map arm's `apeiron/hb_fx` ratio is at or below parity
in both campaigns: the single run's mean `0.9757 < 1` and the six-campaign mean `0.9912 < 1`. The row
binds `none-found`; the measured direction is a ratio against the parity baseline. -/
theorem map07_subsumedParity :
    map07RatioMean < map07Parity ∧ map07SixCampaign < map07Parity := by decide

/-! ## 4 · COUNT-01 — bucket/tag windows (`cache-lines`) -/

/-- The random `1e6` `ap_par/hb_fx` ratio before the move, `9.93x`, in hundredths. `model`, ledger
COUNT-01. -/
def count01BeforeRandom : Nat := 993

/-- The after ratio `3.78x`. `model`, ledger COUNT-01. -/
def count01AfterRandom : Nat := 378

/-- The high-bits `1e7` before ratio `10.05x`, in hundredths. `model`, ledger COUNT-01. -/
def count01BeforeHigh : Nat := 1005

/-- The high-bits after ratio `2.84x`. `model`, ledger COUNT-01. -/
def count01AfterHigh : Nat := 284

/-- **COUNT-01 (bucket/tag windows, `cache-lines`).** Making the bucket and tag windows disjoint
from the home window lowers the `ap_par/hb_fx` ratio at both workloads: `3.78 < 9.93` and
`2.84 < 10.05`. The binding resource is `cache-lines`; the receipt gives the ratio, no line count. -/
theorem count01_bucketTagDisjoint :
    count01AfterRandom < count01BeforeRandom ∧ count01AfterHigh < count01BeforeHigh := by decide

/-! ## 5 · COUNT-02 — home-group fast path (`cache-lines`) -/

/-- The low end of the measured gain, `1.25x`, in hundredths. `model`, ledger COUNT-02. -/
def count02GainLow : Nat := 125

/-- The high end of the measured gain, `1.35x`, in hundredths. `model`, ledger COUNT-02. -/
def count02GainHigh : Nat := 135

/-- The low end of the residual `ap_par/hb_fx` ratio, `1.9x`, in hundredths. `model`, ledger
COUNT-02. -/
def count02ResidualLow : Nat := 190

/-- The high end of the residual ratio, `4.1x`, in hundredths. `model`, ledger COUNT-02. -/
def count02ResidualHigh : Nat := 410

/-- Parity, `1.0`, in hundredths. -/
def count02Parity : Nat := 100

/-- **COUNT-02 (home-group fast path, `cache-lines`).** The `FreqLine::get` fast path lowers the
ratio of every `27` lookup cells by `1.25-1.35x`, while the lookups remain above `hb_fx`
(`1.9-4.1x` > parity). The interval bounds and the still-above-parity floor are the receipt's facts;
the row binds `cache-lines`, with no line count given. -/
theorem count02_homeGroupFastPath :
    count02GainLow < count02GainHigh ∧ count02ResidualLow < count02ResidualHigh ∧
      count02Parity < count02ResidualLow := by decide

/-! ## 6 · COUNT-03 — one `k*PHI` product (`cache-lines`) -/

/-- The dense `ap_par/hb_fx` ratio before `count_get_arr`'s single `k*PHI` product and in-bounds
bucket index, `2.32`, in hundredths. `model`, ledger COUNT-03. -/
def count03BeforeDense : Nat := 232

/-- The dense after ratio `1.99`. `model`, ledger COUNT-03. -/
def count03AfterDense : Nat := 199

/-- The random before ratio `1.70`, in hundredths. `model`, ledger COUNT-03. -/
def count03BeforeRandom : Nat := 170

/-- The random after ratio `1.48`. `model`, ledger COUNT-03. -/
def count03AfterRandom : Nat := 148

/-- **COUNT-03 (`k*PHI` product, `cache-lines`).** One `k*PHI` product and an in-bounds bucket index
lower the `ap_par/hb_fx` ratio at both distributions: `1.99 < 2.32` (dense) and `1.48 < 1.70`
(random), the failing count moving `27 -> 26`. The row binds `cache-lines`; no line count is given. -/
theorem count03_readPathProduct :
    count03AfterDense < count03BeforeDense ∧ count03AfterRandom < count03BeforeRandom := by decide

-- Non-vacuity of the ratio scale: the exact MAP-08 pair ratio, in hundredths (ceiling).
#guard ceilDiv (map08Canonical * 100) map08Prototype = 608
-- The binding `fLine` field on the unit machine: a one-field record's roofline bound is its demand.
#guard timeLB unitMachine (singleField fLine map08Prototype) = 134

end MapMoves

end Perf
