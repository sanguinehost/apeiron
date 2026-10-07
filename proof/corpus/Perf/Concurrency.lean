import Perf.Roofline

/-!
# Latency hiding on the roofline carrier

`Perf.Roofline.timeLB m r = max (fieldMax m r) (r.depth * m.latency)` charges a serial chain its
whole depth times latency, as if no independent work were in flight. This module adds the
concurrency axis: `conc` independent chains in flight, so the chain term amortizes to
`ceil (r.depth * m.latency / conc)` while the throughput terms are unchanged. It ports the
concurrency results of `Closures/Engineering/MemoryLevelParallelism.lean` (`MemModel`,
`latency_hidden`, `latency_hidden_saving`) and `Closures/Engineering/FusionHidesLatency.lean`
(`lone_reduction_is_latency_bound`, `fusion_hides_the_coreduction`,
`fused_strictly_beats_isolated`) — both at sanguine `e830d3fb` — onto `Perf.Res`.

## The critical-path question: inside the max, not additive

The chain term is **inside the max, not added to it**:
`timeLBc m r conc = max (fieldMax m r) (ceil (r.depth * m.latency / conc))`, the chain competing
with the throughput rather than summing with it. `latency_hidden` decides it: under
`r.depth * m.latency ≤ conc * fieldMax m r` the bound is exactly `fieldMax m r`, so the chain
contributes nothing. An additive form `fieldMax m r + ceil (depth*latency / conc)` cannot produce
that — it reports a strictly larger number whenever `depth*latency ≥ 1`. Counter-example numbers at
`L = 4` (all capacities one): with `fieldMax = 100`, `depth*latency = 16`, `conc = 1` (hidden,
`16 ≤ 100`) the inside form is `100` and the additive form is `116`; with the lone reduction
`fieldMax = 4`, `depth*latency = 12`, `conc = 1` (`latency_bound_below_hidden`) the inside form is
`12` and the additive form `16`. The additive form overcounts the hidden case and double-counts the
transport the chain rides on, so it is not the tight bound; only the max form reproduces
`latency_hidden` (`= fieldMax`) and `latency_bound_below_hidden` (`> fieldMax`).

## What is proved here and what is not

* **Proved** under the closed axiom cone `{propext, Classical.choice, Quot.sound}` (the same cone
  as `Perf.Roofline`, to be confirmed by the lead's `#audit` lines): `ceilDiv_le_iff` (the
  ceiling threshold `ceil (a/c) ≤ B ↔ a ≤ B*c`), `ceilDiv_antitone` and `timeLBc_mono_conc` (more
  concurrency never raises the bound), `latency_hidden` (a covered chain is not binding: the bound
  is the throughput bound), `latency_bound_below_hidden` (an uncovered chain binds strictly),
  `chainLB_one` / `timeLBc_one` (`timeLBc` at concurrency one is the serial `timeLB`),
  `fusion_hides_the_coreduction` (fusing a lone reduction into a covering stream removes exactly
  `depth*latency - fieldMax`) and `fused_strictly_beats_isolated`.
* **Not proved.** No hardware speed follows. `Machine.cap`, `Machine.latency` and `conc` are
  declared parameters, not measurements, and no theorem here produces a byte, a cycle or a
  microsecond. rustc/LLVM codegen, the GPU driver, the memory system's real outstanding-miss width
  and the `Res` field counts of the closing examples are outside every statement. The mapping of
  GPU decode's candidate binding resources (chain depth, launches) onto these fields is an
  `anchor`, not a result.
-/

namespace Perf

/-! ## 1 · The concurrency carrier -/

/-- `conc` independent in-flight chains: `Res` plus the concurrency axis. -/
structure MemRes extends Res where
  /-- The number of independent chains in flight. -/
  conc : Nat
  /-- At least one chain is in flight. -/
  concPos : 0 < conc

/-- Bundle a demand record with a concurrency and its positivity witness. -/
def mkMemRes (r : Res) (c : Nat) (hc : 0 < c) : MemRes where
  toRes := r
  conc := c
  concPos := hc

/-- The chain lower bound at concurrency `conc`: `ceil (depth * latency / conc)`. -/
def chainLB (m : Machine) (r : Res) (conc : Nat) : Nat :=
  ceilDiv (r.depth * m.latency) conc

/-- **The concurrent roofline lower bound.** The maximum over the throughput terms and the serial
chain amortized over `conc` independent chains. The chain term is a field of the maximum, not a
summand: `latency_hidden` shows it is absorbed by the throughput when the chains cover it. -/
def timeLBc (m : Machine) (r : Res) (conc : Nat) : Nat :=
  max (fieldMax m r) (chainLB m r conc)

/-- The bound on the bundled carrier, read at the carrier's own concurrency. -/
def MemRes.timeLBc (m : Machine) (r : MemRes) : Nat := Perf.timeLBc m r.toRes r.conc

/-- The bundled carrier at its own concurrency is the free bound at that concurrency. -/
theorem MemRes.timeLBc_eq (m : Machine) (r : Res) (c : Nat) (hc : 0 < c) :
    MemRes.timeLBc m (mkMemRes r c hc) = Perf.timeLBc m r c := by
  simp [MemRes.timeLBc, mkMemRes]

/-! ## 2 · The ceiling threshold -/

/-- `ceil (a / c) ≤ B` exactly when `a ≤ B * c`, for `c > 0`. The forward direction is
`le_ceilDiv_mul`; the backward direction is the floor bound `(a+c-1)/c ≤ (B*c+(c-1))/c = B`. This
threshold is the whole content of the hiding theorems. -/
theorem ceilDiv_le_iff {a c B : Nat} (hc : 0 < c) : ceilDiv a c ≤ B ↔ a ≤ B * c := by
  constructor
  · intro h
    calc a ≤ ceilDiv a c * c := le_ceilDiv_mul hc
      _ ≤ B * c := Nat.mul_le_mul_right c h
  · intro h
    calc ceilDiv a c = (a + c - 1) / c := rfl
      _ ≤ (B * c + (c - 1)) / c := by
          apply Nat.div_le_div_right
          have h3 : a + c - 1 ≤ B * c + c - 1 :=
            Nat.sub_le_sub_right (Nat.add_le_add_right h c) 1
          have h4 : B * c + c - 1 = B * c + (c - 1) := by omega
          omega
      _ = B := by
          apply Nat.div_eq_of_lt_le
          · omega
          · rw [Nat.add_mul, Nat.one_mul]
            omega

/-! ## 3 · Concurrency monotonicity -/

/-- Ceiling division is antitone in the divisor: more in-flight chains need no more steps. -/
theorem ceilDiv_antitone {a conc conc' : Nat} (h1 : 1 ≤ conc) (h : conc ≤ conc') :
    ceilDiv a conc' ≤ ceilDiv a conc := by
  have hc' : 0 < conc' := by omega
  rw [ceilDiv_le_iff hc']
  calc a ≤ ceilDiv a conc * conc := le_ceilDiv_mul (by omega)
    _ ≤ ceilDiv a conc * conc' := Nat.mul_le_mul_left _ h

/-- **More concurrency never raises the bound.** The throughput term is unchanged and the chain term
is amortized over more slots, so `timeLBc` is antitone in `conc`. -/
theorem timeLBc_mono_conc (m : Machine) (r : Res) {conc conc' : Nat}
    (h1 : 1 ≤ conc) (h : conc ≤ conc') : timeLBc m r conc' ≤ timeLBc m r conc := by
  have hchain : chainLB m r conc' ≤ chainLB m r conc := by
    unfold chainLB
    exact ceilDiv_antitone h1 h
  unfold timeLBc
  exact max_le (Nat.le_max_left _ _) (Nat.le_trans hchain (Nat.le_max_right _ _))

/-! ## 4 · The latency-hidden regime -/

/-- **The latency is hidden.** When `conc` independent chains cover the serial stall
(`depth * latency ≤ conc * fieldMax`), the amortized chain term is at most the throughput term, so
the bound is exactly the throughput bound: the kernel is transport-bound, not latency-bound. -/
theorem latency_hidden (m : Machine) (r : Res) {conc : Nat} (hc : 0 < conc)
    (h : r.depth * m.latency ≤ conc * fieldMax m r) : timeLBc m r conc = fieldMax m r := by
  unfold timeLBc
  rw [max_eq_left]
  unfold chainLB
  rw [ceilDiv_le_iff hc]
  calc r.depth * m.latency ≤ conc * fieldMax m r := h
    _ = fieldMax m r * conc := Nat.mul_comm _ _

/-- **When the chain is not hidden it binds.** If `conc` chains do not cover the stall, the chain
term strictly exceeds the throughput bound and the bound is latency-bound:
`fieldMax m r < timeLBc m r conc`. -/
theorem latency_bound_below_hidden (m : Machine) (r : Res) {conc : Nat} (hc : 0 < conc)
    (h : ¬ (r.depth * m.latency ≤ conc * fieldMax m r)) :
    fieldMax m r < timeLBc m r conc := by
  have hlt : conc * fieldMax m r < r.depth * m.latency := Nat.lt_of_not_le h
  have hchain : fieldMax m r < chainLB m r conc := by
    unfold chainLB
    apply Nat.lt_of_not_le
    intro hle
    have hle' := (ceilDiv_le_iff hc).mp hle
    rw [Nat.mul_comm (fieldMax m r) conc] at hle'
    exact absurd hle' (not_le_of_gt hlt)
  unfold timeLBc
  exact Nat.lt_of_lt_of_le hchain (Nat.le_max_right _ _)

/-! ## 5 · Fusion hides the coreduction -/

/-- **A lone reduction is a serial chain.** At concurrency one the amortized chain term is the whole
stall `depth * latency`. -/
theorem chainLB_one (m : Machine) (r : Res) : chainLB m r 1 = r.depth * m.latency := by
  unfold chainLB ceilDiv
  rw [Nat.add_sub_cancel, Nat.div_one]

/-- `timeLBc` at concurrency one is the serial bound `timeLB`. -/
theorem timeLBc_one (m : Machine) (r : Res) : timeLBc m r 1 = timeLB m r := by
  unfold timeLBc timeLB
  rw [chainLB_one]

/-- **Fusion hides the coreduction.** A lone reduction is latency-bound (`fieldMax ≤ depth*latency`);
issued beside a transport-bound stream with `conc` chains that cover its stall, the chain term is
hidden and the bound is the throughput floor. Fusing therefore removes exactly the gap
`depth*latency - fieldMax` from the serial bound. Port of
`FusionHidesLatency.fusion_hides_the_coreduction` (sanguine `e830d3fb`) on `Res`. -/
theorem fusion_hides_the_coreduction (m : Machine) (r : Res) {conc : Nat} (hc : 0 < conc)
    (hhide : r.depth * m.latency ≤ conc * fieldMax m r)
    (hbind : fieldMax m r ≤ r.depth * m.latency) :
    timeLBc m r conc + (r.depth * m.latency - fieldMax m r) = timeLB m r := by
  rw [latency_hidden m r hc hhide]
  unfold timeLB
  rw [max_eq_right hbind]
  omega

/-- **Fusing strictly beats the lone dispatch** when the chain strictly binds
(`fieldMax < depth*latency`): the hidden bound is strictly below the serial bound. Port of
`FusionHidesLatency.fused_strictly_beats_isolated` (sanguine `e830d3fb`) on `Res`. -/
theorem fused_strictly_beats_isolated (m : Machine) (r : Res) {conc : Nat} (hc : 0 < conc)
    (hhide : r.depth * m.latency ≤ conc * fieldMax m r)
    (hstrict : fieldMax m r < r.depth * m.latency) :
    timeLBc m r conc < timeLB m r := by
  rw [latency_hidden m r hc hhide]
  unfold timeLB
  rw [max_eq_right (le_of_lt hstrict)]
  exact hstrict

/-! ## 6 · Concrete instances

The capacities and latencies below are declared `model` inputs, not measurements. `latMachine L` is
one unit of every resource per step and per-step chain latency `L`; `mkRes` is `Roofline`'s
field-count constructor.
-/

/-- A machine at one unit of every resource per step and per-step chain latency `L`. -/
def latMachine (L : Nat) : Machine where
  cap := fun _ => 1
  capPos := fun _ => by omega
  latency := L

/-- A pure serial chain of depth `d` with no throughput demand. -/
def chainRes (d : Nat) : Res := ⟨fun _ => 0, d⟩

/-- Reduction: 16 cache lines, chain depth 4. At `L = 4` the chain term `16` equals `fieldMax`. -/
def hiddenRes : Res := mkRes 0 0 16 0 0 0 0 0 0 4

/-- Reduction: 100 cache lines, chain depth 4. At `L = 4` the chain `16` is far below `fieldMax`. -/
def slackRes : Res := mkRes 0 0 100 0 0 0 0 0 0 4

/-- The lone reduction: 4 cache lines, chain depth 3. At `L = 4`, `depth*latency = 12 > fieldMax = 4`. -/
def loneRes : Res := mkRes 0 0 4 0 0 0 0 0 0 3

-- `timeLBc_mono_conc`: tight (32/4 exact) and slack (ceil 32/7 = 5), and the bound is antitone.
#guard timeLBc (latMachine 4) (chainRes 8) 4 = 8
#guard timeLBc (latMachine 4) (chainRes 8) 7 = 5
#guard timeLBc (latMachine 4) (chainRes 8) 8 ≤ timeLBc (latMachine 4) (chainRes 8) 1

-- `latency_hidden`: tight (chain `16` equals throughput `16`) and slack (chain `16` < throughput `100`).
#guard timeLBc (latMachine 4) hiddenRes 1 = 16
#guard timeLBc (latMachine 4) slackRes 1 = 100

-- `latency_bound_below_hidden`: tight (chain `12`) and slack (ceil 20/3 = 7 > throughput `4`).
#guard timeLBc (latMachine 4) loneRes 1 = 12
#guard timeLBc (latMachine 4) (mkRes 0 0 4 0 0 0 0 0 0 5) 3 = 7

-- `fusion_hides_the_coreduction`: the identity at the lone reduction and the strict drop 12 → 4.
#guard timeLBc (latMachine 4) loneRes 4 + (12 - 4) = timeLB (latMachine 4) loneRes
#guard timeLBc (latMachine 4) loneRes 4 < timeLB (latMachine 4) loneRes

-- Non-vacuity: the theorems instantiated at the concrete records.
example : timeLBc (latMachine 4) (chainRes 8) 8 ≤ timeLBc (latMachine 4) (chainRes 8) 1 :=
  timeLBc_mono_conc (latMachine 4) (chainRes 8) (by omega) (by omega)

example : timeLBc (latMachine 4) hiddenRes 1 = fieldMax (latMachine 4) hiddenRes :=
  latency_hidden (latMachine 4) hiddenRes (by omega) (by decide)

example : fieldMax (latMachine 4) loneRes < timeLBc (latMachine 4) loneRes 1 :=
  latency_bound_below_hidden (latMachine 4) loneRes (by omega) (by decide)

example : timeLBc (latMachine 4) loneRes 4 + (loneRes.depth * 4 - fieldMax (latMachine 4) loneRes)
    = timeLB (latMachine 4) loneRes :=
  fusion_hides_the_coreduction (latMachine 4) loneRes (by omega) (by decide) (by decide)

example : timeLBc (latMachine 4) loneRes 4 < timeLB (latMachine 4) loneRes :=
  fused_strictly_beats_isolated (latMachine 4) loneRes (by omega) (by decide) (by decide)

-- The bundled carrier reads the same bound at its own concurrency.
example : MemRes.timeLBc (latMachine 4) (mkMemRes loneRes 4 (by omega)) = 4 := by
  rw [MemRes.timeLBc_eq, latency_hidden (latMachine 4) loneRes (conc := 4) (by omega) (by decide)]
  decide

end Perf
