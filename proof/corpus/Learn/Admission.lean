import Learn.QuantVerdicts

/-!
# Transfer admission: Amdahl refusal over the per-layer wall

The per-layer transfer wall is a sum of phase terms (`Learn.QuantVerdicts` §15 TEQ, §16 ATTRIB). A lane
that edits one phase can lower only that phase's term, so the projection after any speedup of term `t`
is bounded below by the projection with `t` set to zero (`no_speedup_of_refused`). A lane is admitted
against a bar only when zeroing every term it touches meets the bar; otherwise no measured speedup of
those terms can meet it, and the lane is refused before dispatch rather than after a run.

Units: tenths of a second per qwen2.5-0.5B layer; a bar is a whole-model wall over `layers` layers.
Every wall below is `model` (measured phase walls rescaled by measured per-term ratios); the refusal
theorems are `proved` over those inputs and say nothing about the inputs' accuracy.

- `attribWall`: ATTRIB §16 phases; `other` closes the TEQ recorded layer (545.0 s): objective 6.9 +
  unphased residue 24.0 + layer outside the fit 11.9.
- `nowWall`: the same with the four lanes installed at `7e3c4dbd`, `4aa0ec23`, `7ee5d382` applied as
  their measured ratios to their whole term: sweep × 6733/26527 (§17, `down`), `nb_refine` ×
  2464/20292 (§23 at (896, 896), live re-run), `gptq_e8` × 18262/22090 (§22 at (896, 896)); steps and
  `other` unchanged. No end-to-end layer has been re-measured since ATTRIB; this row is a projection.
-/

set_option linter.apeiron.notation false

namespace Learn.Admission

/-- The per-layer wall by phase, tenths of a second. -/
structure Wall where
  sweep : ℕ
  nbRefine : ℕ
  gptq : ℕ
  steps : ℕ
  other : ℕ
  deriving DecidableEq, Repr

/-- A phase term a lane can edit. -/
inductive Term
  | sweep
  | nbRefine
  | gptq
  | steps
  | other
  deriving DecidableEq, Repr

def Wall.total (w : Wall) : ℕ := w.sweep + w.nbRefine + w.gptq + w.steps + w.other

/-- The wall with term `t` replaced by `k`: what a lane on `t` alone can produce. -/
def Wall.set (w : Wall) : Term → ℕ → Wall
  | .sweep, k => { w with sweep := k }
  | .nbRefine, k => { w with nbRefine := k }
  | .gptq, k => { w with gptq := k }
  | .steps, k => { w with steps := k }
  | .other, k => { w with other := k }

/-- Every term in `ts` set to zero: the best case of a lane touching exactly `ts`. -/
def Wall.zeroAll (w : Wall) (ts : List Term) : Wall := ts.foldl (fun v t => v.set t 0) w

/-- The admission predicate: zeroing the lane's terms brings `layers` layers under `bar`. -/
def admits (w : Wall) (ts : List Term) (layers bar : ℕ) : Bool :=
  decide (layers * (w.zeroAll ts).total ≤ bar)

theorem total_zero_le_set (w : Wall) (t : Term) (k : ℕ) : (w.set t 0).total ≤ (w.set t k).total := by
  cases t <;> simp only [Wall.set, Wall.total] <;> omega

/-- **Amdahl refusal.** If zeroing term `t` misses the bar, every value a lane on `t` can reach misses
it too. A measured speedup of a refused term is not a path to the bar. -/
theorem no_speedup_of_refused (w : Wall) (t : Term) (layers bar : ℕ)
    (h : bar < layers * (w.set t 0).total) (k : ℕ) : bar < layers * (w.set t k).total :=
  lt_of_lt_of_le h (Nat.mul_le_mul_left _ (total_zero_le_set w t k))

/-- The four fit phases: any values at all leave `other` in the total. -/
theorem other_le_total (a b c d : ℕ) (w : Wall) :
    w.other ≤ (Wall.mk a b c d w.other).total := by
  simp only [Wall.total]; omega

def attribWall : Wall := ⟨2465, 1157, 855, 546, 427⟩

def nowWall : Wall := ⟨626, 141, 707, 546, 427⟩

/-- The 0.5B transfer bar (93.9 s, 24 layers) and the single-run ceiling (600 s) that re-opens the
Rust finish (V2), in tenths of a second. -/
def bar : ℕ := QuantVerdicts.teqTransferBar

def ceiling : ℕ := QuantVerdicts.teqRunCeiling

/-- **The model closes on the record.** `attribWall` sums to TEQ's recorded layer and its four phases
are ATTRIB's. -/
theorem attrib_wall_is_the_record :
    attribWall.total = QuantVerdicts.teqLayerRec ∧ attribWall.sweep = QuantVerdicts.attribSweep ∧
      attribWall.nbRefine = QuantVerdicts.attribNbRefine ∧ attribWall.gptq = QuantVerdicts.attribGptq ∧
      attribWall.steps = QuantVerdicts.attribSteps := by decide

/-- **Today's four speed lanes moved the projection 2.2×, to 62× the bar.** 545.0 s → 244.7 s per
layer; 24 layers at 244.7 s is 5873 s against 93.9 s. -/
theorem now_projection :
    nowWall.total = 2447 ∧ 2 * nowWall.total < attribWall.total ∧
      62 * bar < 24 * nowWall.total ∧ 24 * nowWall.total < 63 * bar := by decide

/-- **Every single-term lane was refusable before dispatch.** At the ATTRIB wall, zeroing any one of
the five terms leaves 24 layers above both the bar and the 600 s ceiling. SWEEP-LB, NB-JOINT,
NB-JOINT-2, NB-JOINT-3, BEAM-CVP, GFACT and GFEED each touch one of these terms. -/
theorem attrib_every_single_term_refused :
    ∀ t : Term, admits attribWall [t] 24 ceiling = false := by
  intro t; cases t <;> decide

/-- The same at today's wall: no single-term lane admits against the ceiling, let alone the bar. -/
theorem now_every_single_term_refused :
    ∀ t : Term, admits nowWall [t] 24 ceiling = false := by
  intro t; cases t <;> decide

/-- **The four fit phases together cannot reach the ceiling.** With sweep, `nb_refine`, `gptq_e8` and
the steps all at zero, `other` alone (42.7 s per layer) is 1025 s at 24 layers, over the 600 s
ceiling and 10.9× the bar. Any route that leaves the objective, the unphased residue and the
non-fit layer untouched is refused. -/
theorem fit_phases_cannot_admit (a b c d : ℕ) :
    ceiling < 24 * (Wall.mk a b c d nowWall.other).total := by
  have h := other_le_total a b c d nowWall
  have : ceiling < 24 * nowWall.other := by decide
  omega

/-- `nowWall` after GFEED: `gptq_e8` × 12856/50191 (whole `gptq_e8` at (896, 4864), restructured
12.856 s against a base-equivalent 50.191 s, three interleaved pairs, ratios 3.90–3.94×). -/
def gfeedWall : Wall := ⟨626, 141, 181, 546, 427⟩

/-- **GFEED moved the projection 1.27×, to 49× the bar.** 244.7 s → 192.1 s per layer. -/
theorem gfeed_projection :
    gfeedWall.total = 1921 ∧ 5 * gfeedWall.total < 4 * nowWall.total ∧
      10 * nowWall.total < 13 * gfeedWall.total ∧
      49 * bar < 24 * gfeedWall.total ∧ 24 * gfeedWall.total < 50 * bar := by decide

/-- After GFEED no single-term lane admits against the ceiling either (board row
). -/
theorem gfeed_every_single_term_refused :
    ∀ t : Term, admits gfeedWall [t] 24 ceiling = false := by
  intro t; cases t <;> decide

/-- **NB-JOINT-3 KILL.** Bound-pruning of the joint pair scan kept 8737 of every 10000 pairs (87.37 %),
so its speedup ceiling is 10000/8737 = 1.145×, under the brief's 2×; measured 0.93× (100 per 93) at
(896, 896). Basis points. -/
def nbJoint3Kept : ℕ := 8737

theorem nbjoint3_kill :
    10000 < 2 * nbJoint3Kept ∧ 114 * nbJoint3Kept < 100 * 10000 ∧
      100 * 10000 < 115 * nbJoint3Kept ∧ (93 : ℕ) < 100 := by decide

/-- **Only a whole-layer route admits.** TEQ's R5 (every product at the Q8_0 rate) is a rescaling of
every term at once: 2.9 s per layer, 69.6 s at 24 layers, under the bar. R4 fits the ceiling and
misses the bar. -/
theorem whole_layer_route_admits :
    24 * QuantVerdicts.teqR5 ≤ bar ∧ 24 * QuantVerdicts.teqR4 ≤ ceiling ∧
      bar < 24 * QuantVerdicts.teqR4 := by decide

end Learn.Admission
