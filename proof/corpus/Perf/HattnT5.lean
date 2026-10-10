import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# attention-cut T5: whole-model next-token KLD with page selection in the decode tail

the hattn-decode-tail note: Qwen2.5-0.5B Q4_K_M, the attention-cut T3 32,768-token dump as a dense prefill
cache, the last 8 positions before `t ∈ {8192, 32768}` recomputed through all 24 layers once per arm
(`KV_TAIL` mode of `examples/attn_cut_probe.rs`, output `tail.out`). Every figure
is a `model` input copied from that output; no theorem here produces a measurement. KLD and per-head KL in
ten-thousandths of a nat, reads in ten-thousandths of `t`, rel-L2 in units of `1e-8`. rustc, LLVM and the
instrument are outside every statement.

`cert_stop_sound` is the certified stop's soundness: if the unread mass `U` is at most `R` and
`(1 − ε) R ≤ ε Z`, the read mass `Z` is at least `1 − ε` of the total, so per-head KL ≤ `−ln(1 − ε)`.
-/

namespace Perf

namespace HattnT5

/-! ## Measured constants (mean over the 8 tail positions unless named) -/

/-- Dense-arm tail q, k, v against the dump, max rel-L2 over 24 layers, units of `1e-8`. -/
def ctl8 : ℕ := 357

def ctl32 : ℕ := 1505

/-- Certification violations (kept mass < `1 − ε`): the real bound, and the min-only mutation. -/
def certViol : ℕ := 0

def mutViol8 : ℕ := 1157

def mutViol32 : ℕ := 1146

/-- Oracle arm (per-head top keys to KL 0.05): whole-model KLD and mean per-head KL, at 8k and 32k. -/
def oracleKld8 : ℕ := 20

def oracleKld32 : ℕ := 7

def oracleHeadKl8 : ℕ := 471

def oracleHeadKl32 : ℕ := 475

/-- Certified Quest-64 stop: median reads per `t` at 32k, and its whole-model KLD. -/
def certReads32 : ℕ := 10309

def certKld32 : ℕ := 0

/-- Near set only (4 sinks + last ≥ 256 keys): whole-model KLD at 8k and 32k. -/
def nearKld8 : ℕ := 3930

def nearKld32 : ℕ := 83

/-- Quest-64 with a fixed 2048-token far budget at 32k: whole-model KLD and reads per `t`. -/
def fixKld32 : ℕ := 18

def fixReads32 : ℕ := 1031

/-- Quest-64 with the oracle stop at 32k: whole-model KLD and median reads per `t`. -/
def orderKld32 : ℕ := 6

def orderReads32 : ℕ := 601

/-- Certified arm top-1 agreement with dense over the 16 positions. -/
def certTop1 : ℕ := 16

/-! ## Verdicts on `PREDICTIONS-T5.md` (sha256 `26c24721…`) -/

/-- **PT1 held.** The dense tail reproduces the dump within `1e-4`. -/
theorem pt1_held : ctl8 ≤ 10000 ∧ ctl32 ≤ 10000 := by decide

/-- **PT2 held, and its control is red.** No certified head lost more than `ε`; the min-only bound broke it. -/
theorem pt2_held : certViol = 0 ∧ 0 < mutViol8 ∧ 0 < mutViol32 := by decide

/-- **PT3 held.** Per-head KL 0.05 in every head gives whole-model KLD 0.002 (8k) and 0.0007 (32k). -/
theorem pt3_held : oracleKld8 ≤ 500 ∧ oracleKld32 ≤ 500 := by decide

/-- **PT4 held.** The certified stop reads more than half of dense at the median. -/
theorem pt4_held : 5000 < certReads32 := by decide

/-- **The certified stop costs more than dense**: bounds plus nearly every page exceed `t` reads. -/
theorem cert_exceeds_dense : 10000 < certReads32 := by decide

/-- **PT5 refuted at 32k.** Dropping the far keys costs only 0.0083 nats there; at 8k it costs 0.39. -/
theorem pt5_refuted : ¬ 500 < nearKld32 := by decide

theorem pt5_holds_at_8k : 500 < nearKld8 := by decide

/-- **PT6 held.** A 2048-token far budget stays under KLD 0.05 at 32k, reading `0.103 t`. -/
theorem pt6_held : fixKld32 ≤ 500 := by decide

/-- **PT7 held.** The certified arm agrees with dense on top-1 at all 16 positions. -/
theorem pt7_held : certTop1 = 16 := by decide

/-- **Per-head KL does not compound; it attenuates.** Whole-model KLD is below a twentieth of the mean
per-head KL at 8k and below a fiftieth at 32k. -/
theorem compounding_attenuates : 20 * oracleKld8 < oracleHeadKl8 ∧ 50 * oracleKld32 < oracleHeadKl32 := by
  decide

/-! ## Soundness of the certified stop -/

/-- **The certified stop is sound.** With read mass `Z > 0`, unread mass `0 ≤ U ≤ R` and
`(1 − ε) R ≤ ε Z`, the kept share `Z / (Z + U)` is at least `1 − ε`. -/
theorem cert_stop_sound (Z U R ε : ℝ) (hZ : 0 < Z) (hU : 0 ≤ U) (hUR : U ≤ R) (hε1 : ε < 1)
    (hR : (1 - ε) * R ≤ ε * Z) : 1 - ε ≤ Z / (Z + U) := by
  rw [le_div_iff₀ (by linarith)]
  linarith [mul_le_mul_of_nonneg_left hUR (by linarith : (0 : ℝ) ≤ 1 - ε)]

end HattnT5

end Perf
