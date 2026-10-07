import Perf.Roofline

/-!
# The kept moves as instances of the move theorem

`Perf.Roofline` proves the resource carrier `Perf.Res`, the roofline lower bound `Perf.timeLB`, and
the move theorem `Perf.move_speedup_le` (`timeLB f ≤ K * timeLB g` when the baseline demand is at
most `K` times the after demand on every field and on the chain depth). This module instantiates
three kept move classes of the perf-move-ledger note on that carrier.

Every field count below is a **`model`** input read from the named ledger row (or derived from one),
not a measurement; `Machine.cap` is likewise a declared parameter. No theorem here produces a byte,
a cycle or a microsecond (obligation 11).

## What is proved, and by which theorem

* **Instruction width (GPU-01, measured 2.0-2.8x pp512).** DP4A at code width `b` emits
  `phases b = 8 / b` `OpSUDot` phases per weight word (`Spirv/Carrier.lean:53`, `carrier_split`;
  `src/simt/dp4a.rs:167` `dots_per_word`). An int8 cooperative-matrix instruction of tile `t`
  retires `t` MACs against DP4A's 4, so MAC-equal work has MMA instruction demand `D * 4 / t` where
  the DP4A demand is `D`. `instrWidth_speedup_le` proves the bound ratio `t / 4 = q` on the unit
  machine. The move *switches instruction class* (`fDp4a` to `fMma`); `width_no_DemandRatioBound`
  proves the baseline demand therefore cannot satisfy `DemandRatioBounded` across the switch, so
  `move_speedup_le` itself does not apply to this move and the bound is proved directly. This is
  the one move class here that is a field switch.
* **Walk count (GPU-02, measured node 1.91x).** Online softmax stages the K/V slabs in `3` walks;
  the one-walk schedule makes `1`. Shared-bytes demand is `walks * U` (`U` bytes per walk) and the
  FMA demand `F` is unchanged. `walk_demandRatio` proves `DemandRatioBounded` at ratio `3`, and
  `walk_speedup` is a literal instance of `move_speedup_le`. `walk_mix` states the resource mix the
  measured `1.91x` implies when the FMA demand also binds: if the realized ratio is strictly between
  `1` and `3`, then `U < F < 3 * U`. (On a general machine `U`, `F` are the field terms
  `ceilDiv U (cap fShared)`, `ceilDiv F (cap fFma)`; `walk_mix` is stated on the unit machine.)
* **Cache-line count (MAP-01; the control).** `map01_line_control` reuses the
  `scalar_not_sufficient` shape: a pair with fewer instructions but more cache lines has the larger
  `timeLB` on a line-costly machine, so a model that counts instructions only predicts the wrong
  sign. `map01_move_speedup` is the kept-direction instance: with model lines falling and FMA/ALU
  falling, `move_speedup_le` gives the bound ratio `2`.

## What is not covered

No hardware speed follows: capacities are parameters, the field counts are `model` inputs, and
rustc/LLVM codegen, the CPU and GPU drivers, cache replacement and clock behaviour are outside every
statement. The `phases` restatement is not imported from `Spirv.Carrier` (that module pulls the
extraction corpus, outside this task's build budget); the source line is cited instead. The
cooperative-matrix tile width `t`, the per-walk shared bytes `U`, the FMA demand `F` and the
MAP-01 line counts are declared model inputs, never measured here.
-/

namespace Perf

/-! ## 0 · Demand records with one or two nonzero fields

`mkRes` (`Perf/Roofline.lean:432`) is the general nine-field constructor; the moves below are
carried by one or two fields, for which the exact `timeLB` value is easier to state. -/

/-- A demand record with `n` units of field `f` and zero elsewhere, zero depth. -/
def singleField (f : ResField) (n : Nat) : Res where
  demand := fun i => if i = f then n else 0
  depth := 0

@[simp] theorem singleField_self (f : ResField) (n : Nat) : (singleField f n).demand f = n := by
  show (if f = f then n else 0) = n
  rw [if_pos rfl]

theorem singleField_ne {f i : ResField} (h : i ≠ f) (n : Nat) :
    (singleField f n).demand i = 0 := by
  show (if i = f then n else 0) = 0
  rw [if_neg h]

@[simp] theorem singleField_depth (f : ResField) (n : Nat) : (singleField f n).depth = 0 := rfl

/-- A demand record with `n1` units of field `f1`, `n2` units of the distinct field `f2`, and zero
elsewhere, zero depth. -/
def twoField (f1 f2 : ResField) (n1 n2 : Nat) : Res where
  demand := fun i => if i = f1 then n1 else if i = f2 then n2 else 0
  depth := 0

@[simp] theorem twoField_fst (f1 f2 : ResField) (n1 n2 : Nat) :
    (twoField f1 f2 n1 n2).demand f1 = n1 := by
  show (if f1 = f1 then n1 else if f1 = f2 then n2 else 0) = n1
  rw [if_pos rfl]

theorem twoField_snd {f1 f2 : ResField} (h : f2 ≠ f1) (n1 n2 : Nat) :
    (twoField f1 f2 n1 n2).demand f2 = n2 := by
  show (if f2 = f1 then n1 else if f2 = f2 then n2 else 0) = n2
  rw [if_neg h, if_pos rfl]

theorem twoField_other {f1 f2 i : ResField} (h1 : i ≠ f1) (h2 : i ≠ f2) (n1 n2 : Nat) :
    (twoField f1 f2 n1 n2).demand i = 0 := by
  show (if i = f1 then n1 else if i = f2 then n2 else 0) = 0
  rw [if_neg h1, if_neg h2]

@[simp] theorem twoField_depth (f1 f2 : ResField) (n1 n2 : Nat) :
    (twoField f1 f2 n1 n2).depth = 0 := rfl

/-- `ceilDiv n 1 = n`, the field term when the capacity is one unit per step. -/
@[simp] theorem ceilDiv_one' (n : Nat) : ceilDiv n 1 = n := by
  unfold ceilDiv; rw [Nat.add_sub_cancel, Nat.div_one]

/-- A vector bounded entrywise is bounded by its maximum. -/
theorem maxVec_le : ∀ {k : Nat} {v : Fin k → Nat} {n : Nat}, (∀ i, v i ≤ n) → maxVec k v ≤ n
  | 0, v, n, _ => by exact Nat.zero_le n
  | k + 1, v, n, h => by
      show max (v 0) (maxVec k fun i => v i.succ) ≤ n
      exact Nat.max_le.mpr ⟨h 0, maxVec_le (fun i => h i.succ)⟩

/-- On the unit machine the lower bound of a one-field record is that field's demand: the field
maximum is the demand itself. -/
theorem timeLB_unit_single (f : ResField) (n : Nat) :
    timeLB unitMachine (singleField f n) = n := by
  have hle : fieldMax unitMachine (singleField f n) ≤ n := by
    unfold fieldMax
    refine maxVec_le (fun i => ?_)
    rw [fieldLB, show unitMachine.cap i = 1 from rfl, ceilDiv_one']
    by_cases hi : i = f
    · subst hi; rw [singleField_self]
    · rw [singleField_ne hi]; exact Nat.zero_le n
  have hge : n ≤ fieldMax unitMachine (singleField f n) := by
    have := le_maxVec 9 (fun i => fieldLB unitMachine (singleField f n) i) f
    simpa [fieldMax, fieldLB, show unitMachine.cap f = 1 from rfl, ceilDiv_one', singleField_self]
      using this
  have hmax : timeLB unitMachine (singleField f n) =
      max (fieldMax unitMachine (singleField f n)) 0 := by
    unfold timeLB; rw [singleField_depth, Nat.zero_mul]
  rw [hmax]
  exact le_antisymm (Nat.max_le.mpr ⟨hle, Nat.zero_le n⟩)
    (Nat.le_trans hge (Nat.le_max_left _ _))

/-- On the unit machine the lower bound of a two-field record is the maximum of the two field
demands. -/
theorem timeLB_unit_two {f1 f2 : ResField} (h12 : f1 ≠ f2) (n1 n2 : Nat) :
    timeLB unitMachine (twoField f1 f2 n1 n2) = max n1 n2 := by
  have hle : fieldMax unitMachine (twoField f1 f2 n1 n2) ≤ max n1 n2 := by
    unfold fieldMax
    refine maxVec_le (fun i => ?_)
    rw [fieldLB, show unitMachine.cap i = 1 from rfl, ceilDiv_one']
    by_cases h1 : i = f1
    · subst h1; rw [twoField_fst]; exact Nat.le_max_left n1 n2
    · by_cases h2 : i = f2
      · subst h2; rw [twoField_snd h12.symm]; exact Nat.le_max_right n1 n2
      · rw [twoField_other h1 h2]; exact Nat.zero_le _
  have hge : max n1 n2 ≤ fieldMax unitMachine (twoField f1 f2 n1 n2) := by
    refine Nat.max_le.mpr ⟨?_, ?_⟩
    · have := le_maxVec 9 (fun i => fieldLB unitMachine (twoField f1 f2 n1 n2) i) f1
      simpa [fieldMax, fieldLB, show unitMachine.cap f1 = 1 from rfl, ceilDiv_one', twoField_fst]
        using this
    · have := le_maxVec 9 (fun i => fieldLB unitMachine (twoField f1 f2 n1 n2) i) f2
      simpa [fieldMax, fieldLB, show unitMachine.cap f2 = 1 from rfl, ceilDiv_one',
        twoField_snd h12.symm] using this
  have hmax : timeLB unitMachine (twoField f1 f2 n1 n2) =
      max (fieldMax unitMachine (twoField f1 f2 n1 n2)) 0 := by
    unfold timeLB; rw [twoField_depth, Nat.zero_mul]
  rw [hmax]
  exact le_antisymm (Nat.max_le.mpr ⟨hle, Nat.zero_le _⟩)
    (Nat.le_trans hge (Nat.le_max_left _ _))

/-! ## 1 · Instruction width (ledger GPU-01, `mma`)

`Spirv/Carrier.lean` states the emitted `OpSUDot` phase split once, indexed by the code width `b`:
`phases b = 8 / b` phases per weight word (`Spirv/Carrier.lean:53`; the served path is
`src/simt/dp4a.rs:167` `dots_per_word`, and `Spirv.carrier_split` is stated over `phases b`).

The row GPU-01 replaces a DP4A GEMM (4 MACs per instruction, the perf-move-ledger note)
by an int8 cooperative-matrix GEMM (tile `t` MACs per instruction). The field counts are `model`.
-/

/-- The `OpSUDot` phase count at code width `b`: `8 / b`. Restated from `Spirv/Carrier.lean:53`
(not imported, see the module note), which cites `src/simt/dp4a.rs:167`. -/
def phases (b : Nat) : Nat := 8 / b

/-- The DP4A instruction demand for `words` weight words at width `b`: `words * phases b`. -/
def dp4aInstr (words b : Nat) : Nat := words * phases b

/-- The MMA instruction demand for MAC-equal work: `D * 4 / t`, a tile-`t` instruction retiring
`t` MACs against DP4A's `4`. -/
def mmaInstr (D t : Nat) : Nat := D * 4 / t

/-- The DP4A schedule: `D` four-way dot-product instructions, nothing else. -/
def dp4aRes (D : Nat) : Res := singleField fDp4a D

/-- The MMA schedule: `mmaInstr D t` matrix instructions, nothing else. -/
def mmaRes (D t : Nat) : Res := singleField fMma (mmaInstr D t)

-- The width model's non-vacuity: the three admissible packings of `phases`.
#guard phases 8 = 1
#guard phases 4 = 2
#guard phases 2 = 4
-- `D` DP4A MAC-phases at width `b` and tile `t = 4 * q`: `Q4` words carry `2` phases.
#guard dp4aInstr 8 4 = 16

/-- **Instruction-width move (GPU-01).** With `t = 4 * q` and DP4A demand `D = q * n`, the
MAC-equal MMA schedule has instruction demand `D * 4 / t = n`, so the lower bound on the unit
machine is at most `q = t / 4` times the MMA lower bound. This is the move theorem's conclusion on
the switched instruction fields; `width_no_DemandRatioBound` shows `DemandRatioBounded` itself
cannot hold for the switch. -/
theorem instrWidth_speedup_le (q n : Nat) (hq : 0 < q) :
    timeLB unitMachine (dp4aRes (q * n)) ≤ q * timeLB unitMachine (mmaRes (q * n) (4 * q)) := by
  rw [dp4aRes, mmaRes, mmaInstr, timeLB_unit_single, timeLB_unit_single]
  have hdiv : q * n * 4 / (4 * q) = n := by
    rw [show q * n * 4 = (4 * q) * n by ring]
    exact Nat.mul_div_right n (by omega)
  rw [hdiv]

/-- **The field-switch obstruction.** The width move relieves `fDp4a` and pays `fMma`: the baseline
demand on `fDp4a` has no after demand on the same field, so no `K` satisfies `DemandRatioBounded`
while the baseline demand is nonzero. This is why the width move is proved directly rather than by
`move_speedup_le`. -/
theorem width_no_DemandRatioBound (D m K : Nat) (hD : 0 < D) :
    ¬ DemandRatioBounded (dp4aRes D) (singleField fMma m) K := by
  intro h
  have hf := h.1 fDp4a
  simp only [dp4aRes, singleField_self] at hf
  rw [singleField_ne (show fDp4a ≠ fMma by decide)] at hf
  omega

/-! ## 2 · Walk count (ledger GPU-02, `shared-bytes`)

The online-softmax hot loop stages the K/V slabs in three walks before the one-walk schedule; the
shared-bytes demand is `walks * U` for `U` bytes per walk and the FMA demand `F` is unchanged
. Both counts are `model` inputs. -/

/-- The online-softmax schedule: `walks * U` shared bytes on `fShared` and `F` FMA instructions on
`fFma`, zero depth. -/
def walkRes (walks U F : Nat) : Res := twoField fShared fFma (walks * U) F

@[simp] theorem walkRes_shared (w U F : Nat) : (walkRes w U F).demand fShared = w * U := by
  unfold walkRes; rw [twoField_fst]

@[simp] theorem walkRes_fma (w U F : Nat) : (walkRes w U F).demand fFma = F := by
  unfold walkRes; rw [twoField_snd (show fFma ≠ fShared by decide)]

theorem walkRes_other {i : ResField} (h1 : i ≠ fShared) (h2 : i ≠ fFma) (w U F : Nat) :
    (walkRes w U F).demand i = 0 := by
  unfold walkRes; rw [twoField_other h1 h2]

@[simp] theorem walkRes_depth (w U F : Nat) : (walkRes w U F).depth = 0 := rfl

theorem timeLB_unit_walkRes (w U F : Nat) :
    timeLB unitMachine (walkRes w U F) = max (w * U) F := by
  unfold walkRes
  rw [timeLB_unit_two (show fShared ≠ fFma by decide)]

/-- The walk-count move lowers the shared-bytes demand `3 * U` to `U` and leaves the FMA demand `F`
fixed, so the demand ratio is three on the shared field and one elsewhere. -/
theorem walk_demandRatio (U F : Nat) :
    DemandRatioBounded (walkRes 3 U F) (walkRes 1 U F) 3 := by
  refine ⟨fun i => ?_, ?_⟩
  · by_cases h1 : i = fShared
    · subst h1; simp only [walkRes_shared]; omega
    · by_cases h2 : i = fFma
      · subst h2; simp only [walkRes_fma]; omega
      · rw [walkRes_other h1 h2, walkRes_other h1 h2, Nat.mul_zero]
  · rw [walkRes_depth, walkRes_depth, Nat.mul_zero]

/-- **Walk-count move (GPU-02).** A literal instance of `move_speedup_le` at ratio `3`: the
three-walk lower bound is at most `3` times the one-walk lower bound. -/
theorem walk_speedup (m : Machine) (U F : Nat) :
    timeLB m (walkRes 3 U F) ≤ 3 * timeLB m (walkRes 1 U F) :=
  move_speedup_le m (walk_demandRatio U F)

/-- **The resource mix the measured `1.91x` implies.** If the one-walk schedule is strictly faster
(`1 < ratio`) but strictly less than the model's `3` (`ratio < 3`), the FMA demand `F` binds
against the shared term `U`: `U < F < 3 * U`. On the unit machine `U`, `F` are the field terms; in
general they are `ceilDiv U (cap fShared)` and `ceilDiv F (cap fFma)`. No measured number is
asserted here. -/
theorem walk_mix (U F : Nat)
    (hgain : timeLB unitMachine (walkRes 1 U F) < timeLB unitMachine (walkRes 3 U F))
    (hle3 : timeLB unitMachine (walkRes 3 U F) < 3 * timeLB unitMachine (walkRes 1 U F)) :
    U < F ∧ F < 3 * U := by
  rw [timeLB_unit_walkRes, timeLB_unit_walkRes] at hgain hle3
  simp only [one_mul] at hgain
  omega

-- Non-vacuity: three walks of five shared bytes and seven FMA instructions bound at 15 and 7, and
-- the mix witness `U = 5`, `F = 7` satisfies `5 < 7 < 15`.
#guard timeLB unitMachine (walkRes 3 5 7) = 15
#guard timeLB unitMachine (walkRes 1 5 7) = 7
example : (5 : Nat) < 7 ∧ 7 < 3 * 5 := walk_mix 5 7 (by decide) (by decide)

/-! ## 3 · Cache-line count (ledger MAP-01, `cache-lines`; the control)

MAP-01 splits the map probe into an inline hot path and an out-of-line cold path; the ledger row
records `121.4 -> 88.3` instructions per absent operation and `3.88 -> 2.91` ns
. The instruction figure is the `fAlu` demand and the ns
figure is the measured comparison; the line counts are `model` inputs. -/

/-- A probe model with `instr` (scaled `instr/op`) on `fAlu` and `lines` cache lines on `fLine`. -/
def lineInstrRes (instr lines : Nat) : Res := twoField fLine fAlu lines instr

@[simp] theorem lineInstrRes_line (instr lines : Nat) :
    (lineInstrRes instr lines).demand fLine = lines := by
  unfold lineInstrRes; rw [twoField_fst]

@[simp] theorem lineInstrRes_alu (instr lines : Nat) :
    (lineInstrRes instr lines).demand fAlu = instr := by
  unfold lineInstrRes; rw [twoField_snd (show fAlu ≠ fLine by decide)]

theorem lineInstrRes_other {i : ResField} (h1 : i ≠ fLine) (h2 : i ≠ fAlu) (instr lines : Nat) :
    (lineInstrRes instr lines).demand i = 0 := by
  unfold lineInstrRes; rw [twoField_other h1 h2]

@[simp] theorem lineInstrRes_depth (instr lines : Nat) : (lineInstrRes instr lines).depth = 0 := rfl

/-- A machine where an ALU instruction costs `1 / 1000` of a cache line: capacity `1000` ALU
instructions or `1` line per step, in units of a line. A declared `model` machine. -/
def lineMachine : Machine where
  cap := fun i => if i = fAlu then 1000 else 1
  capPos := fun i => by split <;> norm_num
  latency := 0

/-- The control view: the cold probe executes `1214` model instructions and opens `1` model line;
the split halves the instructions to `883` but its glance opens `8` model lines. On `lineMachine`
the second is strictly slower, so an instruction-only model predicts the wrong sign
(`scalar_not_sufficient`'s shape, `Perf/Roofline.lean:465`). -/
def map01Cold : Res := lineInstrRes 1214 1

/-- The split view of `map01Cold` (`model`). -/
def map01Hot : Res := lineInstrRes 883 8

/-- **The cache-line control (MAP-01).** Fewer instructions (`883 < 1214`) but more lines
(`8 > 1`) raises the lower bound on `lineMachine`. The field counts and the machine are `model`
inputs, not measurements. -/
theorem map01_line_control :
    totalInstr map01Hot < totalInstr map01Cold ∧
    timeLB lineMachine map01Cold < timeLB lineMachine map01Hot := by
  decide

/-- The kept-direction model pair: the cold view opens `8` lines at `1214` instructions, the split
opens `4` at `883` (`model`). -/
def map01KeptCold : Res := lineInstrRes 1214 8

/-- The split view of `map01KeptCold` (`model`). -/
def map01KeptHot : Res := lineInstrRes 883 4

/-- **The kept MAP-01 move.** Both demands fall, so `move_speedup_le` at ratio `2` bounds the
kept-direction lower-bound ratio (`1214 ≤ 2 * 883`, `8 ≤ 2 * 4`). The bound is an upper bound on
the lower-bound ratio and is loose by the Nat ceiling. -/
theorem map01_move_speedup :
    timeLB unitMachine map01KeptCold ≤ 2 * timeLB unitMachine map01KeptHot :=
  move_speedup_le unitMachine (K := 2) (by
    unfold map01KeptCold map01KeptHot
    refine ⟨fun i => ?_, ?_⟩
    · by_cases h1 : i = fLine
      · subst h1; simp only [lineInstrRes_line]; norm_num
      · by_cases h2 : i = fAlu
        · subst h2; simp only [lineInstrRes_alu]; norm_num
        · rw [lineInstrRes_other h1 h2, lineInstrRes_other h1 h2, Nat.mul_zero]
    · rw [lineInstrRes_depth, lineInstrRes_depth, Nat.mul_zero])

-- Non-vacuity of the control and the kept pair.
#guard totalInstr map01Hot = 883
#guard totalInstr map01Cold = 1214
#guard timeLB unitMachine map01KeptCold = 1214
#guard timeLB unitMachine map01KeptHot = 883

end Perf
