import Mathlib.Data.Nat.Basic

/-!
# attention-cut T7: bytes per KV head, and one page set per GQA group reused across positions

the hattn-decode-tail note (T7 section): the attention-cut T6 setup (64 tail positions before each
`t ∈ {4096, 8192, 16384, 32768}`) with the cost counted in bytes per (KV head, position):
`(2·|∪ over the 7 query heads of keys used| + bound vectors read) / (2t)`, so dense is exactly 1
(`KV_TAIL_GROUP=1`, output `~/hattn-long/group.out`). Every figure is a `model` input
copied from that output; no theorem here produces a measurement. Mean whole-model KLD and mean bytes/dense,
both in ten-thousandths; rel-L2 in units of `1e-8`. rustc, LLVM and the instrument are outside every
statement.
-/

namespace Perf

namespace HattnT7

/-- Dense-arm tail control, max rel-L2 over the four `t` and 24 layers, units of `1e-8`. -/
def ctlMax : ℕ := 1815

/-- Mean KLD at `t = 4096, 8192, 16384, 32768`. `a64-*`: Quest-64 pages per query head (T6). `grp-*-r1`: one
page set per KV head, pages ranked by the max over its 7 query heads of Quest's bound. `grp-*-r8`: that set
recomputed every 8 positions and reused between. `oracle`: per-head KL 0.25, union over the group. -/
def a64f1024Kld : List ℕ := [165, 368, 293, 89]

def a64f2048Kld : List ℕ := [36, 92, 158, 53]

def grp1024r1Kld : List ℕ := [196, 1551, 396, 145]

def grp2048r1Kld : List ℕ := [47, 227, 254, 104]

def grp1024r8Kld : List ℕ := [262, 1764, 455, 150]

def grp2048r8Kld : List ℕ := [53, 234, 342, 110]

def oracleKld : List ℕ := [259, 876, 497, 314]

/-- Mean bytes/dense at the same four `t`. -/
def a64f1024Bytes : List ℕ := [5675, 3263, 2042, 1161]

def a64f2048Bytes : List ℕ := [8279, 5221, 3302, 1892]

def grp2048r1Bytes : List ℕ := [5901, 3018, 1584, 870]

def grp2048r8Bytes : List ℕ := [5786, 2892, 1453, 736]

def oracleBytes : List ℕ := [1133, 712, 456, 210]

/-- `HATTN_MUT=2` (group ranking reversed): `grp-fix1024-r1` mean KLD at the four `t`. -/
def mutKld : List ℕ := [1730, 6065, 1136, 231]

/-- The figures the verdicts compare at a single `t`: bytes/dense at 32768 and the mutation's KLD at 8192. -/
def a64f1024B32 : ℕ := 1161

def a64f2048B32 : ℕ := 1892

def grp2048r1B32 : ℕ := 870

def grp2048r8B32 : ℕ := 736

def oracleB32 : ℕ := 210

def mutKld8 : ℕ := 6065

/-- The single-`t` figures are the lists' entries. -/
theorem singles_agree : a64f1024Bytes.getLast? = some a64f1024B32 ∧ a64f2048Bytes.getLast? = some a64f2048B32 ∧
    grp2048r1Bytes.getLast? = some grp2048r1B32 ∧ grp2048r8Bytes.getLast? = some grp2048r8B32 ∧
    oracleBytes.getLast? = some oracleB32 ∧ mutKld.tail.head? = some mutKld8 := by
  decide

/-- T6's mean key reads per `t` for `a64-fix1024` at 32768, counted per query head (`HattnT6.fix1024Reads32`). -/
def t6Reads32 : ℕ := 712

/-- Every entry of `l` is at most `0.05`. -/
def passes (l : List ℕ) : Prop := ∀ k ∈ l, k ≤ 500

instance (l : List ℕ) : Decidable (passes l) := by unfold passes; infer_instance

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T7.md`, sha256 `54a5d2ab…`) -/

/-- **PG1 held.** The dense tail reproduces the dump and the reversed ranking breaks group selection at 8192. -/
theorem pg1_held : ctlMax ≤ 10000 ∧ 2000 < mutKld8 := by decide

/-- **PG2 held.** Per-query-head selection costs at least `0.10` of dense bytes at 32k once the 7 heads' keys
are unioned. -/
theorem pg2_held : 1000 ≤ a64f1024B32 := by decide

/-- **PG3 refuted.** A 1024-token group budget fails at 8192 (mean KLD 0.155). -/
theorem pg3_refuted : ¬ passes grp1024r1Kld := by decide

/-- **PG4 held.** A 2048-token group budget passes at every `t` and costs at most `0.10` of dense at 32k. -/
theorem pg4_held : passes grp2048r1Kld ∧ grp2048r1B32 ≤ 1000 := by decide

/-- **PG5 held.** Reusing the group's page set for 8 positions moves mean KLD by at most `0.01` at every `t`. -/
theorem pg5_held : ∀ p ∈ grp2048r8Kld.zip grp2048r1Kld, p.1 ≤ p.2 + 100 ∧ p.2 ≤ p.1 + 100 := by decide

/-- **PG6 held.** The cheapest deployable arm passing at every `t` is `grp-fix2048-r8`, at most `0.08` of dense
bytes at 32k. The 1024-token arms fail; every other passing arm costs more at 32k. -/
theorem pg6_held :
    passes grp2048r8Kld ∧ grp2048r8B32 ≤ 800 ∧ ¬ passes grp1024r1Kld ∧ ¬ passes grp1024r8Kld ∧
      grp2048r8B32 < grp2048r1B32 ∧ grp2048r8B32 < a64f1024B32 ∧
      grp2048r8B32 < a64f2048B32 := by
  decide

/-! ## Findings -/

/-- **T6's 14× was per query head.** Counted per KV head with K and V, `a64-fix1024` is `0.116` of dense at
32k: between 8× and 9× cheaper, not 14×; the T6 read count was below the bytes figure. -/
theorem gqa_correction : 8 * a64f1024B32 < 10000 ∧ ¬ 9 * a64f1024B32 < 10000 ∧
    t6Reads32 < a64f1024B32 := by
  decide

/-- **Group selection with reuse is 13× cheaper than dense at 32k** at mean KLD at most `0.035` at every `t`. -/
theorem grp2048r8_thirteen_x : 13 * grp2048r8B32 < 10000 ∧ ∀ k ∈ grp2048r8Kld, k ≤ 350 := by decide

/-- **The gap to the oracle is 3.5×.** The oracle's group union at per-head KL 0.25 is `0.021` of dense at
32k, while `grp-fix2048-r8` reads `0.074`; the oracle does not pass at 8192 (0.088). -/
theorem oracle_gap : 3 * oracleB32 < grp2048r8B32 ∧ ¬ 4 * oracleB32 < grp2048r8B32 ∧
    ¬ passes oracleKld := by
  decide

end HattnT7

end Perf
