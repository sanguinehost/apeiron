import Mathlib.Data.Nat.Basic

/-!
# FACES-R1 H10: a polar (elliptic) page box against Quest's Cartesian box, kept softmax mass at 524288

`docs/epics/spinor.md` ("FACES-R1 H10"): Qwen2.5-7B-Instruct-1M prose dump `g7-1m`, `t = 524288`, layers 0, 7, 14, 21, 27,
4 KV heads × 7 query heads, 4 of the 64 tail positions. Read set: 4 sinks, near keys and the 32 best 64-key pages by the
bound. Entries are `model` inputs copied from `FACES-R1-POLAR-BOUND.out` (`FACES-R1-POLAR-BOUND.py`, sha256
`8543edb97a8e`, output sha256 `f596437214bd`): mean kept softmax mass in units of `1e-4`. The instrument and
rustc/numpy are outside every statement.
-/

namespace Perf

namespace FacesPolarBound

def quest : ℕ := 5144

def polar : ℕ := 4912

def hybrid : ℕ := 5199

def pageOracle : ℕ := 6375

def tokenOracle : ℕ := 7881

/-- **P1 refuted.** The per-pair minimum of the Cartesian and polar bounds keeps `0.0055` more mass than Quest's
bound, under the registered `0.03`. -/
theorem p1_refuted : hybrid - quest = 55 ∧ ¬ 300 ≤ hybrid - quest := by decide

/-- **P2 held.** The hybrid bound stays `0.1177` below the page oracle (registered gap at least `0.05`). -/
theorem p2_held : pageOracle - hybrid = 1176 ∧ 500 ≤ pageOracle - hybrid := by decide

/-- **P3 held.** The page oracle stays `0.1506` below the token oracle (registered gap at least `0.05`). -/
theorem p3_held : tokenOracle - pageOracle = 1506 ∧ 500 ≤ tokenOracle - pageOracle := by decide

/-- **Finding.** The polar box alone keeps less than Quest's box. A tighter per-pair bound does not rank better:
the page granularity (`0.1506`) and the bound (`0.1176`) each cost more than the polar box returns. -/
theorem polar_alone_loses : polar < quest ∧ 55 < 1176 ∧ 55 < 1506 := by decide

end FacesPolarBound

end Perf
