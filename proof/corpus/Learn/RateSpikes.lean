import Learn.QuantVerdicts

/-!
# Rate spikes on qwen2.5-0.5B: every clause against its measured constant

| spike | receipt | sha256 |
|---|---|---|
| M1 (plane walk, KL against the streamed fraction `a`) | `lead-1021/PW1/receipt.md` | `82bab1e69d6714b0` |
| KW1 (sign walker at three rates, per-row water-fill at 1.18 b/w) | `lead-1021/KW1/receipt.md` | `526c604d76f7d58a` |
| MT1 (trellis-coded E8, Sturmian rates, per-row water-fill at 1.18 b/w) | `lead-1021/MT1/receipt.md` | `6b94b3b9997de7bd` |
| KW1 + MT1 same-scale attribution (rows re-read, no new run) | `lead-1021/MT1/attr/attr.out` | `f04177880afb495e` |
| MT2 (beam 4 per state, radius grid, per-row water-fill at 1.18 b/w, production scale) | `lead-1021/MT2/receipt.md` | `5f82d9aa88fdebc8` |
| MT2 per-linear ratios / allocation log | `~/mt2-table.tsv` / `~/mt2-alloc.log` | `0345663b0ec2a6a1` / `992087e5beee94ee` |

Every figure is a `model` input copied from its receipt: a KL in millionths of a nat over chunks 0..3
of `train400k` (`n_ctx` 512), a rate in ten-thousandths of a bit per weight over the 357.83 M scoped
trunk weights. No theorem here produces a measurement. The KL figures are measured on the crate's
`KernelMode::Q8Act` engine path; rustc, the kernel ABI and the `unsafe` pool are outside every
statement. The module extends the spike-verdict family of `Learn.QuantVerdicts`; no clause here is
linked to an extracted body, because the measured plane walk is a spike and is not on the crate.

## M1 · plane walk

`y = W·x_S + W_t·x_¬S`: `S` is the top-`a` fraction of input columns by `|S_b·xq_j|`, `W` the file's
`Q8_0` payload, `W_t` a resident base of `r_res` bits per weight. The arm `Ak` rounds `q / 2^k`; `B`
keeps the sign. `eff b/w = r_res + a·r_stream` counts the bits **read** per token, not the bits
**stored**.
-/

namespace Learn

namespace RateSpikes

/-! ### Constants (receipt § KL(a), scope `all`) -/

/-- The 0.5B KL target, millionths of a nat (0.1038). -/
def klTarget : Nat := 103800

/-- The 0.5B KL acceptance bar, millionths of a nat (0.2645). -/
def klBar : Nat := 264500

/-- KL of the `a = 1` cell, the unmodified engine on the same slice, millionths of a nat. -/
def klExact : Nat := 2052

/-- `r_res` of arm A5 (`q / 2^5`) at one 16-bit scale per 128 weights, ten-thousandths of a b/w. -/
def rResA5 : Nat := 29254

/-- `r_res` of arm A6 (`q / 2^6`), ten-thousandths of a b/w. -/
def rResA6 : Nat := 20020

/-- `r_res` of arm A7 (`q / 2^7`), ten-thousandths of a b/w. -/
def rResA7 : Nat := 11442

/-- `r_res` of arm A8 (scale only), ten-thousandths of a b/w. -/
def rResA8 : Nat := 1250

/-- `r_res` of arm B (sign base), ten-thousandths of a b/w. -/
def rResB : Nat := 11854

/-- `r_stream` of arm A5, the residual plane read for a column in `S`, ten-thousandths of a b/w. -/
def rStreamA5 : Nat := 50000

/-- `r_stream` of arm A6, ten-thousandths of a b/w. -/
def rStreamA6 : Nat := 60000

/-- `r_stream` of arm A7, ten-thousandths of a b/w. -/
def rStreamA7 : Nat := 70000

/-- `r_stream` of arm A8, ten-thousandths of a b/w. -/
def rStreamA8 : Nat := 80000

/-- `r_stream` of arm B, ten-thousandths of a b/w. -/
def rStreamB : Nat := 80000

/-- The GO ceiling on the resident base rate, ten-thousandths of a b/w (1.6). -/
def rGoCeiling : Nat := 16000

/-- The mission rate, ten-thousandths of a b/w (1.18: 27B in about 4 GB). -/
def rMission : Nat := 11800

/-- A5 KL at `a = 0`, millionths of a nat. -/
def klA5_0 : Nat := 519107

/-- A5 KL at `a = 0.05`, millionths of a nat. -/
def klA5_05 : Nat := 106379

/-- A5 KL at `a = 0.1`, millionths of a nat. -/
def klA5_10 : Nat := 74840

/-- A5 KL at `a = 0.15`, millionths of a nat. -/
def klA5_15 : Nat := 54771

/-- A5 KL at `a = 0.3`, millionths of a nat. -/
def klA5_30 : Nat := 24491

/-- A6 KL at `a = 0.1`, millionths of a nat. -/
def klA6_10 : Nat := 372438

/-- A6 KL at `a = 0.3`, millionths of a nat. -/
def klA6_30 : Nat := 93031

/-- A7 KL at `a = 0.1`, its best measured cell at `a ≤ 0.15`, millionths of a nat. -/
def klA7_10 : Nat := 3887718

/-- A8 KL at `a = 0.1`, its best measured cell at `a ≤ 0.15`, millionths of a nat. -/
def klA8_10 : Nat := 12228903

/-- B KL at `a = 0.15`, its best measured cell at `a ≤ 0.15`, millionths of a nat. -/
def klB_15 : Nat := 2264210

/-- Scope `ffn` (attention exact), A5 at `a = 0.2`: KL, millionths of a nat. -/
def klA5Ffn_20 : Nat := 33749

/-- Scope `ffn`, A5 at `a = 0.2`: all-in effective rate, ten-thousandths of a b/w. -/
def rA5Ffn_20 : Nat := 44280

/-- Scope `all`, A5 at `a = 0.3`: effective rate, ten-thousandths of a b/w. -/
def rA5All_30 : Nat := 44254

/-- Control 1: bottom-`a` selection at A5, `a = 0.1`, KL in millionths of a nat. -/
def klA5Bottom_10 : Nat := 520568

/-- The standard error (sd/√n) of the top-`a` A5 cell at `a = 0.1`, millionths of a nat. -/
def sdA5_10 : Nat := 9370

/-! ### Verdicts -/

/-- **GO fails.** Every arm whose resident base is at most 1.6 b/w (A7, A8, B) misses the 0.1038
target at every measured `a ≤ 0.15`, and the one arm that meets it there (A5, 0.0548) holds a
2.93 b/w base.
`∧` [both are true]; `<` [is smaller than]; `≤` [is smaller than or equal to]. -/
theorem m1_go_fails :
    rResA7 ≤ rGoCeiling ∧ rResA8 ≤ rGoCeiling ∧ rResB ≤ rGoCeiling ∧
      klTarget < klA7_10 ∧ klTarget < klA8_10 ∧ klTarget < klB_15 ∧
      klA5_15 ≤ klTarget ∧ rGoCeiling < rResA5 := by decide

/-- **PROMISING fires**: A5 and A6 reach the 0.2645 bar at `a = 0.3`, so KILL does not.
`∧` [both are true]; `≤` [is smaller than or equal to]. -/
theorem m1_promising : klA5_30 ≤ klBar ∧ klA6_30 ≤ klBar := by decide

/-- **KL falls monotonically in `a`** for A5, down to the unmodified engine at `a = 1`.
`∧` [both are true]; `<` [is smaller than]. -/
theorem m1_a5_monotone :
    klExact < klA5_30 ∧ klA5_30 < klA5_15 ∧ klA5_15 < klA5_10 ∧ klA5_10 < klA5_05 ∧
      klA5_05 < klA5_0 := by decide

/-- **The selection is what carries the quality**: the bottom-`a` control rises above the top-`a`
cell by more than 47 of its standard errors (the brief required 3).
`+` [plus]; `*` [times]; `<` [is smaller than]. -/
theorem m1_selection_control : klA5_10 + 47 * sdA5_10 < klA5Bottom_10 := by decide

/-- **At equal all-in rate the `all` scope beats the `ffn` scope**: A5 at `a = 0.3` over the whole
trunk reads no more bits than A5 at `a = 0.2` with attention exact, and has lower KL.
`∧` [both are true]; `<` [is smaller than]; `≤` [is smaller than or equal to]. -/
theorem m1_all_scope_wins : rA5All_30 ≤ rA5Ffn_20 ∧ klA5_30 < klA5Ffn_20 := by decide

/-- `model`: **the stored footprint of every arm is at least 7.9 b/w.** With streaming from host RAM
or disk ruled out, every residual plane is resident, so the footprint is `r_res + r_stream`, not
`eff b/w`. A5, the smallest, stores more than 6.7× the mission's 1.18 b/w: the plane walk
reduces bits read per token, never bits stored.
`∧` [both are true]; `<` [is smaller than]; `+` [plus]; `*` [times]. -/
theorem m1_footprint_not_rate :
    79000 < rResA5 + rStreamA5 ∧ 79000 < rResA6 + rStreamA6 ∧ 79000 < rResA7 + rStreamA7 ∧
      79000 < rResA8 + rStreamA8 ∧ 79000 < rResB + rStreamB ∧
        rMission * 67 < (rResA5 + rStreamA5) * 10 := by
  decide

/-! ## KW1 and MT1 · same-scale attribution

`J(W, Wq) = tr((W − Wq) H (W − Wq)ᵀ)` per linear on layers 0, 11, 23 (21 linears), `H` the `Mean` Gram of
256 calibration tokens. `E` is `gptq_e8` beam 4 with `wscale` (1.125 b/w all-in), `E1` beam 1 without
`wscale`. Two scales occur in the receipts: **production**, `E`'s group scale `a` on the `E8_SCALE` word
(suffix `p`, and `E` itself), and **refit**, a per-128-group diagonal-block least-squares refit (`Er` is
`E` under it). Every figure below is a median over the 21 linears of a ratio whose two arms share one
scale, in thousandths, with the count of linears below 1. The figures are printed by
`lead-1021/MT1/attr/attr.sh` from KW1's `table.txt` and MT1's
`rows/manifest.tsv`; its output `attr.out` has sha256 `f04177880afb495e`.

The verdicts `kw1_kill`, `kw1_no_linear_won`, `mt1_kill`, `mt1_mix_is_e` and `mt1_state_costs` are
withdrawn: each compared a refit-scale arm with production-scale `E`, so the refit's own cost
(`refit_defect`) was read as the walker's.
-/

/-- Median `J_Er / J_E`: the refit against production on `E` itself, thousandths. -/
def spErE : Nat := 1915

/-- Linears on which the refit lowers `J_E` (layer 11 `k`). -/
def spErEWon : Nat := 1

/-- The linears measured. -/
def spLinears : Nat := 21

/-- **The refit is an instrument defect**: it nearly doubles `E`'s `J` in the median and lowers it on one
linear of 21, so a refit-scale arm against production `E` carries a factor of about 1.9 that is not the arm's.
`∧` [both are true]; `<` [is smaller than]; `*` [times]. -/
theorem refit_defect : 1900 < spErE ∧ spErEWon * 20 < spLinears := by decide

/-! ### KW1 · sign walker

`S1_L` emits ±1 from the newest bit of an `L`-bit register at 1 code bit per weight; `S2` emits {±1, ±3}
from the newest two bits (2.125 b/w); `mix` is the per-row water-fill over {Z, Sh, S1_8, S2} at 1.18 b/w.
Every KW1 arm is refit scale, so each is read against `Er`. -/

/-- Median `J_S1_1 / J_E` as KW1 reported it (refit arm over production `E`), thousandths; only its ratio
to `kwS1_8` is read, where the common `J_E` cancels per linear. -/
def kwS1_1 : Nat := 3177

/-- Median `J_S1_8 / J_E` as KW1 reported it, thousandths; read only against `kwS1_1`. -/
def kwS1_8 : Nat := 2881

/-- Median `J_S1_8 / J_Er`, thousandths. -/
def kwS1_8r : Nat := 1503

/-- Smallest `J_S1_8 / J_Er` (layer 11 `o`), thousandths. -/
def kwS1_8rMin : Nat := 981

/-- Linears on which `S1_8` is under `Er`. -/
def kwS1_8rWon : Nat := 1

/-- Median `J_S2 / J_Er` (2.125 b/w against 1.125), thousandths. -/
def kwS2r : Nat := 264

/-- Linears on which `S2` is under `Er`. -/
def kwS2rWon : Nat := 21

/-- Median `J_mix / J_Er` at 1.18 b/w, thousandths. -/
def kwMixR : Nat := 1265

/-- Linears on which the KW1 water-fill is under `Er`. -/
def kwMixRWon : Nat := 9

/-- **At one bit the ±1 walker loses to `E` by 1.50×, not 2.88×**, and wins one linear (layer 11 `o`).
`∧` [both are true]; `<` [is smaller than]. -/
theorem kw1_one_bit_loses :
    1000 < kwS1_8r ∧ kwS1_8r < 1600 ∧ kwS1_8rMin < 1000 ∧ kwS1_8rWon = 1 := by decide

/-- **The register does no coding work**: eight state bits move the median `J` by less than 10% against one,
because the emitted value reads only the newest bit, so every sign sequence is reachable at every `L`.
`*` [times]; `<` [is smaller than]. -/
theorem kw1_register_idle : kwS1_1 * 9 < kwS1_8 * 10 := by decide

/-- **The walk is on the rate–distortion slope; the ±1 alphabet carries the one-bit loss**: one more bit per
weight returns `2^-2` within 10% (`D(R) ∝ 2^-2R`, Shannon (1959), Gaussian source: a `model` reading), and
`S2` is under `Er` on every linear.
`∧` [both are true]; `≤` [is smaller than or equal to]. -/
theorem kw1_alphabet_on_slope : 225 ≤ kwS2r ∧ kwS2r ≤ 275 ∧ kwS2rWon = spLinears := by decide

/-- **The water-fill is under `E` on 9 of 21 linears**, above it in the median.
`∧` [both are true]; `<` [is smaller than]. -/
theorem kw1_mix_nine : 1000 < kwMixR ∧ kwMixRWon = 9 := by decide

/-! ### MT1 · trellis-coded E8

Block value `a·ρ_s·c[idx]`, `c` the 256-word E8 ∪ O8 codebook, `s` the low `Lr` bits of the previous block's
`idx`; `T_Lr` is a beam-1 Viterbi over `2^Lr` states with per-survivor whitened feedback and no `wscale`,
radii a Lloyd fit to the `E1` gains. `R_x` reads `⌈8(m+1)x⌉ − ⌈8mx⌉` bits for block `m`. -/

/-- Median `J_T0p / J_E1`: one state, one radius, against `E` at the same beam, thousandths. -/
def mtT0pE1 : Nat := 953

/-- Linears on which `T0p` is under `E1`. -/
def mtT0pE1Won : Nat := 20

/-- Smallest `J_T0p / J_E1` (layer 0 `q`), thousandths. -/
def mtT0pE1Min : Nat := 526

/-- Median `J_E1 / J_E`: what beam 4 and `wscale` buy `E`, thousandths. -/
def mtE1E : Nat := 1283

/-- Smallest `J_E1 / J_E`, thousandths. -/
def mtE1EMin : Nat := 1089

/-- Median `J_T0p / J_E`, thousandths. -/
def mtT0pE : Nat := 1150

/-- Linears on which `T0p` is under `E` (layer 0 `q`, layer 11 `k`, layer 0 `k`). -/
def mtT0pEWon : Nat := 3

/-- Smallest `J_T0p / J_E` (layer 0 `q`), thousandths. -/
def mtT0pEMin : Nat := 724

/-- Median `J_T1p / J_T0p`, thousandths. -/
def mtT1T0p : Nat := 1017

/-- Linears on which `T1p` is under `T0p`. -/
def mtT1T0pWon : Nat := 9

/-- Median `J_T2p / J_T0p`, thousandths. -/
def mtT2T0p : Nat := 1058

/-- Linears on which `T2p` is under `T0p`. -/
def mtT2T0pWon : Nat := 4

/-- Median `J_R9/8p / J_E` (1.25 b/w all-in against 1.125), thousandths. -/
def mtR98p : Nat := 973

/-- Linears on which `R9/8p` is under `E`. -/
def mtR98pWon : Nat := 15

/-- Median `J_R5/4p / J_E` (1.375 b/w all-in), thousandths. -/
def mtR54p : Nat := 906

/-- Linears on which `J(R_17/16)` lies strictly between `J(R_1)` and `J(R_9/8)` and below their chord,
refit scale. -/
def mtDialHolds : Nat := 3

/-- The same count at production scale. -/
def mtDialHoldsP : Nat := 0

/-- **At equal beam, `E1` at a fitted radius beats `E1`** on 20 of 21 linears, by up to 1.9× (layer 0 `q`).
`T0p` is one state and one radius, so this is a radius finding, not a walker finding.
`∧` [both are true]; `<` [is smaller than]. -/
theorem mt1_radius_beats_beam1 :
    mtT0pE1 < 1000 ∧ mtT0pE1Won + 1 = spLinears ∧ mtT0pE1Min < 550 := by decide

/-- **Against full `E` the beam-1 walker wins three linears and loses the median**; the median gap (1.150)
is smaller than what beam 4 and `wscale` buy `E` (1.283), which they buy on every linear. Whether the
remaining gap is the beam is the open question of .
`∧` [both are true]; `<` [is smaller than]. -/
theorem mt1_gap_under_beam :
    mtT0pEMin < 1000 ∧ mtT0pEWon = 3 ∧ 1000 < mtT0pE ∧ mtT0pE < mtE1E ∧ 1000 < mtE1EMin := by decide

/-- **At beam 1 with radii fitted to `E1`'s gains, two states are neutral and four cost 5.8%**: `T1p`
within 2% of `T0p` (9 of 21 under it), `T2p` above it (4 of 21 under).
`∧` [both are true]; `<` [is smaller than]. -/
theorem mt1_state_at_beam1 :
    mtT1T0p < 1020 ∧ mtT1T0pWon = 9 ∧ 1050 < mtT2T0p ∧ mtT2T0pWon = 4 := by decide

/-- **Rate buys `J`**: `R_9/8` (1.25 b/w all-in) is under `E` in the median and on 15 of 21 linears;
`R_5/4` (1.375) is under it by 9.4%.
`∧` [both are true]; `<` [is smaller than]. -/
theorem mt1_rate_pays : mtR98p < 1000 ∧ mtR98pWon = 15 ∧ mtR54p < 1000 := by decide

/-- **The Sturmian dial is not a convex rate curve here**: the 17/16 rung sits below the chord on 3 of 21
linears at refit scale and on none at production scale.
`∧` [both are true]; `*` [times]; `<` [is smaller than]. -/
theorem mt1_dial_not_convex : mtDialHolds * 2 < spLinears ∧ mtDialHoldsP = 0 := by decide

/-! ### Verdicts: a KILL or a GO is a term, and its constructor takes the comparison's witnesses

KW1's and MT1's KILLs failed every field below: refit challengers against a production incumbent, beam 1
without `wscale` against beam 4 with it, no arm reproducing the incumbent, and a part's loss read as the
concept's. `Killed c` and `Goes c bar` both require `c.Fair`; a KILL also requires the fair mix to give the
part no rows, because a part the mix buys rows from is not dead at that rate. `c.part` names the component
under test, so a KILL is of that component and never of the concept. `#verdict_guard` refuses any
declaration whose name says `kill` or ends in `_go` unless its type is one of the two. -/

/-- The scale an arm's codewords are scored at: `E`'s group scale on the `E8_SCALE` word, or the per-group
diagonal-block refit. -/
inductive Scale
  | production
  | refit
  deriving DecidableEq

/-- The encoder search an arm ran: survivors per state, the `wscale` projection, and the encodings offered
per row (1 for one arm, the menu size for a water-fill). -/
structure Search where
  beam : Nat
  wscale : Bool
  menu : Nat
  deriving DecidableEq

/-- One challenger-against-incumbent comparison over a population of linears, read off a receipt. `ratio` is
the median `J_challenger / J_incumbent` in thousandths, `won` the linears below 1, `degen` the linears on
which the challenger's degenerate setting reproduced the incumbent bit for bit, `mixRows` the rows the
fair water-fill at equal rate gives the challenger. -/
structure Comparison where
  part : String
  population : Nat
  ratio : Nat
  won : Nat
  challengerScale : Scale
  incumbentScale : Scale
  challengerSearch : Search
  incumbentSearch : Search
  degen : Nat
  mixRows : Nat

/-- The witnesses a decisive verdict needs: one scale, equal search, the incumbent reproduced on every
linear. -/
structure Comparison.Fair (c : Comparison) : Prop where
  sameScale : c.challengerScale = c.incumbentScale
  equalSearch : c.challengerSearch = c.incumbentSearch
  degenerate : c.degen = c.population
  nonempty : 0 < c.population

/-- A KILL of `c.part`: fair, not better at the median, under the incumbent on at most half the linears,
and given no rows by the fair mix. -/
structure Killed (c : Comparison) : Prop where
  fair : c.Fair
  loses : 1000 ≤ c.ratio
  minority : 2 * c.won ≤ c.population
  unselected : c.mixRows = 0

/-- A GO of `c.part` at `bar` thousandths: fair, and the median ratio at or under the bar. -/
structure Goes (c : Comparison) (bar : Nat) : Prop where
  fair : c.Fair
  wins : c.ratio ≤ bar

/-- PROMISING needs no witness: under the incumbent in the median and on a majority of linears. -/
abbrev Promising (c : Comparison) : Prop := c.ratio < 1000 ∧ c.population < 2 * c.won

open Lean Elab Command in
/-- `#verdict_guard ns` throws on every theorem under `ns` whose last name component contains `kill` (any
case) or ends in `_go`, unless its type is `Killed _` or `Goes _ _`. Theorems only: the recursors, constructor
and projections Lean generates for `Killed` itself are not verdicts. Cost: one fold over the environment's
constants, `O(|env|)` name tests, each `O(|name|)`. -/
elab "#verdict_guard " ns:ident : command => do
  let pre := ns.getId
  let bad := (← getEnv).constants.fold (init := (#[] : Array Name)) fun acc n ci =>
    let s := (match n with | .str _ s => s | _ => "").toLower
    let verdict := (s.splitOn "kill").length > 1 || s.endsWith "_go"
    if pre.isPrefixOf n && !n.isInternal && ci matches .thmInfo _ && verdict then
      match ci.type.getAppFn.constName? with
      | some k => if k == ``Killed || k == ``Goes then acc else acc.push n
      | none => acc.push n
    else acc
  unless bad.isEmpty do
    throwError "verdict_guard: {bad.size} verdict(s) under {pre} are not Killed/Goes terms: {bad.toList}"
  logInfo m!"verdict_guard: every kill/go under {pre} is a Killed/Goes term"

/-! ### MT2 · the walker at beam 4 per state, production scale

`T_Lr K4`: `2^Lr` states, `K = 4` survivors per state, `wscale` on the final survivors, all at `E`'s scale.
Arm C sweeps one state's radius over {0.88, …, 1.24}; arm D fits `T1`/`T2` radii by Lloyd to arm C's argmin
gains. `T0K4` at `ρ = 1` is `gptq_e8(…, true, true, 4)` bit for bit (test
`the_beam_tube_is_gptq_e8_beam_four`, and 21/21 row dumps sha-equal to MT1's `E`). -/

/-- `E`'s search: beam 4, `wscale`, one encoding per row. -/
def searchE : Search := ⟨4, true, 1⟩

/-- Arm C: one state, radius `ρ*` per linear. -/
def mt2Radius : Comparison :=
  ⟨"one state, K = 4 + wscale, radius rho* per linear from the 7-point grid", 21, 993, 13,
    .production, .production, searchE, searchE, 21, 1179⟩

/-- Arm D at two states. -/
def mt2State1 : Comparison :=
  ⟨"two states, K = 4 per state + wscale, Lloyd radii fitted to arm C's argmin gains", 21, 1010, 9,
    .production, .production, searchE, searchE, 21, 6940⟩

/-- Arm D at four states. -/
def mt2State2 : Comparison :=
  ⟨"four states, K = 4 per state + wscale, Lloyd radii fitted to arm C's argmin gains", 21, 1040, 4,
    .production, .production, searchE, searchE, 21, 3983⟩

/-- The best of arms C and D per linear, against `E`. -/
def mt2Best : Comparison :=
  ⟨"min over T0K4(rho*), T1K4, T2K4 per linear", 21, 993, 13,
    .production, .production, searchE, searchE, 21, 12102⟩

/-- Arms B–D added to MT1's walker-free per-row menu, against that menu, both at 1.18 b/w all-in. The
challenger chooses among 19 encodings per row, the incumbent among 7. -/
def mt2Mix : Comparison :=
  ⟨"arms B-D in the per-row water-fill at 1.18 b/w", 21, 947, 21,
    .production, .production, ⟨4, true, 19⟩, ⟨4, true, 7⟩, 21, 38016⟩

/-- Median `J_T2K4 / J_T0K4(ρ*)`, thousandths. -/
def mt2T2T0 : Nat := 1064

/-- Median `J_T0K4(ρ*) / J_E` over the 12 attention linears (q, k, v, o), thousandths. -/
def mt2RadiusAttn : Nat := 973

/-- Wide linears (gate, up, down). -/
def mt2Wide : Nat := 9

/-- Wide linears with `ρ* = 1.00`. -/
def mt2WideRho1 : Nat := 8

/-- **Arms C and D are fair comparisons against `E`**: one scale, `E`'s search, `E` reproduced on 21/21.
`∧` [both are true]. -/
theorem mt2_fair : mt2Radius.Fair ∧ mt2State1.Fair ∧ mt2State2.Fair :=
  ⟨⟨rfl, rfl, rfl, by decide⟩, ⟨rfl, rfl, rfl, by decide⟩, ⟨rfl, rfl, rfl, by decide⟩⟩

/-- **The walker at beam 4 is PROMISING, not GO**: median 0.993, under `E` on 13 of 21.
`∧` [both are true]; `¬` [this is not true]. -/
theorem mt2_promising : Promising mt2Best ∧ ¬ Goes mt2Best 950 :=
  ⟨by decide, fun h => absurd h.wins (by decide)⟩

/-- **The radius survives beam 4 only on attention linears**: median 0.973 there, `ρ* = 1.00` on 8 of the
9 wide linears. `∧` [both are true]; `<` [is smaller than]. -/
theorem mt2_radius_attention : Promising mt2Radius ∧ mt2RadiusAttn < 980 ∧ mt2WideRho1 + 1 = mt2Wide := by
  refine ⟨by decide, by decide, by decide⟩

/-- **The state loses the per-linear median and is not killed**: `T1K4` 1.010 and `T2K4` 1.040 against `E`,
`T2K4` 1.064 against `T0K4(ρ*)`, yet the fair mix gives the state 6940 and 3983 of 38016 rows, so no
`Killed` term exists for either. `∧` [both are true]; `<` [is smaller than]; `¬` [this is not true]. -/
theorem mt2_state_open :
    1000 < mt2State1.ratio ∧ 1000 < mt2State2.ratio ∧ 1060 < mt2T2T0 ∧
      ¬ Killed mt2State1 ∧ ¬ Killed mt2State2 :=
  ⟨by decide, by decide, by decide, fun h => absurd h.unselected (by decide),
    fun h => absurd h.unselected (by decide)⟩

/-- **The mix gain is not a GO**: arms B–D lower the equal-rate water-fill by a median 5.3%, under it on
21/21, past the 0.95 bar, but the challenger chooses among 19 encodings per row against 7, so the
comparison is not fair and no `Goes` term exists. Whether the rows are bought by the state or by a
second search is the open question.
`∧` [both are true]; `≤` [is smaller than or equal to]; `¬` [this is not true]. -/
theorem mt2_mix_unequal_search : Promising mt2Mix ∧ mt2Mix.ratio ≤ 950 ∧ ¬ Goes mt2Mix 950 :=
  ⟨by decide, by decide, fun h => absurd h.fair.equalSearch (by decide)⟩

#verdict_guard Learn.RateSpikes

end RateSpikes

end Learn

/-!
# First-order held-out quality model on qwen2.5-0.5B layer 0 (`Learn.QualityModel`)

Receipt: `~/gen-w1-t01/gen-n8.json` (GEN-, `ncal` 8, arm `plain`, one row per
linear of layer 0). Every constant is a `model` input copied from that file: `KL_out` is the held-out
KL of the tensor quantised alone, in millionths of a nat.

## The model

`KL_total ≈ Σ_t κ_t · J_t`, with `J_t` the tensor's Hessian-weighted output error and `κ_t` a per-tensor
constant (`KL_out / J_out` in the receipt, a 5× spread across tensors). A change that scales `J_t` by
`r_t / 1000` moves the predicted KL by the factor `r_t / 1000` on that tensor's term. The model is
derived to first order and is **not** validated for additivity across tensors (GEN-is the test);
the theorems below are consequences of it, graded `model`.

## What it says about the walker

The walker's measured `J` ratios on layer 0 (`q 0.759`, `k 0.946`, `v 0.970`, `o 0.936`) apply to the
four tensors that carry 29.2% of the summed held-out KL. Under the model the gate/up/down tensors carry
the other 70.8%, and the walker leaves them unchanged, so its layer-0 gain is bounded by that 29.2% and
is 1.9% at the measured ratios.
-/

namespace Learn

namespace QualityModel

/-! ### Constants (`gen-n8.json`, millionths of a nat) -/

/-- Held-out KL of `q` alone. -/
def klQ : Nat := 8137

/-- Held-out KL of `k` alone. -/
def klK : Nat := 8635

/-- Held-out KL of `v` alone. -/
def klV : Nat := 29956

/-- Held-out KL of `o` alone. -/
def klO : Nat := 189236

/-- Held-out KL of `gate` alone. -/
def klGate : Nat := 124482

/-- Held-out KL of `up` alone. -/
def klUp : Nat := 241787

/-- Held-out KL of `down` alone. -/
def klDown : Nat := 207099

/-- The model's per-tensor `J` scale factors, per mille, in the order `q k v o gate up down`. -/
structure Ratios where
  q : Nat
  k : Nat
  v : Nat
  o : Nat
  gate : Nat
  up : Nat
  down : Nat

/-- Predicted summed held-out KL times 1000: `Σ_t KL_t · r_t`, `r_t` per mille. -/
def predicted (r : Ratios) : Nat :=
  klQ * r.q + klK * r.k + klV * r.v + klO * r.o + klGate * r.gate + klUp * r.up + klDown * r.down

/-- The unchanged encoder: every ratio is 1000. -/
def unity : Ratios := ⟨1000, 1000, 1000, 1000, 1000, 1000, 1000⟩

/-- The walker at its measured layer-0 ratios on `q k v o`, unchanged on `gate up down`. -/
def walker : Ratios := ⟨759, 946, 970, 936, 1000, 1000, 1000⟩

/-- Pointwise order on ratios. -/
def Ratios.le (a b : Ratios) : Prop :=
  a.q ≤ b.q ∧ a.k ≤ b.k ∧ a.v ≤ b.v ∧ a.o ≤ b.o ∧ a.gate ≤ b.gate ∧ a.up ≤ b.up ∧ a.down ≤ b.down

/-- Pointwise minimum: the hybrid that takes the better arm per tensor. -/
def Ratios.min (a b : Ratios) : Ratios :=
  ⟨Nat.min a.q b.q, Nat.min a.k b.k, Nat.min a.v b.v, Nat.min a.o b.o,
    Nat.min a.gate b.gate, Nat.min a.up b.up, Nat.min a.down b.down⟩

/-! ### Structure of the model -/

/-- **Lower ratios never raise the prediction.** `≤` [is smaller than or equal to]; `∧` [and]. -/
theorem predicted_mono {a b : Ratios} (h : a.le b) : predicted a ≤ predicted b := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  unfold predicted
  exact Nat.add_le_add (Nat.add_le_add (Nat.add_le_add (Nat.add_le_add (Nat.add_le_add
    (Nat.add_le_add (Nat.mul_le_mul_left _ h1) (Nat.mul_le_mul_left _ h2))
    (Nat.mul_le_mul_left _ h3)) (Nat.mul_le_mul_left _ h4)) (Nat.mul_le_mul_left _ h5))
    (Nat.mul_le_mul_left _ h6)) (Nat.mul_le_mul_left _ h7)

/-- **The per-tensor hybrid is no worse than either arm** under the model.
`∧` [both are true]; `≤` [is smaller than or equal to]. -/
theorem hybrid_dominates (a b : Ratios) :
    predicted (a.min b) ≤ predicted a ∧ predicted (a.min b) ≤ predicted b := by
  refine ⟨predicted_mono ?_, predicted_mono ?_⟩
  · exact ⟨Nat.min_le_left _ _, Nat.min_le_left _ _, Nat.min_le_left _ _, Nat.min_le_left _ _,
      Nat.min_le_left _ _, Nat.min_le_left _ _, Nat.min_le_left _ _⟩
  · exact ⟨Nat.min_le_right _ _, Nat.min_le_right _ _, Nat.min_le_right _ _, Nat.min_le_right _ _,
      Nat.min_le_right _ _, Nat.min_le_right _ _, Nat.min_le_right _ _⟩

/-! ### The receipt's numbers -/

/-- The summed held-out KL, millionths of a nat, times 1000 under `unity`. -/
theorem total_unity : predicted unity = 809332 * 1000 := by decide

/-- **`gate`, `up`, `down` carry more than half of the summed KL** (573 368 of 809 332), so a method
confined to `q k v o` cannot recover more than the other share. `<` [is smaller than]. -/
theorem gate_up_down_majority : 809332 < 2 * (klGate + klUp + klDown) := by decide

/-- **`q`, `k`, `v`, `o` carry 235 964 of 809 332** (29.2%). Even `J = 0` on all four leaves at least
70.8% of the prediction. `∧` [both are true]. -/
theorem attention_share : klQ + klK + klV + klO = 235964 ∧ klGate + klUp + klDown = 573368 := by
  exact ⟨by decide, by decide⟩

/-- **Zeroing `J` on `q k v o` caps the model's gain at 29.2%**: the prediction stays above 70% of the
unchanged value. `≤` [is smaller than or equal to]. -/
theorem attention_zero_floor :
    700 * predicted unity ≤ 1000 * predicted ⟨0, 0, 0, 0, 1000, 1000, 1000⟩ := by decide

/-- **The walker's predicted layer-0 gain is 15 437 091 / 1000 millionths, under 2%** of the summed KL:
`predicted walker` is within 2% of `predicted unity`. `∧` [both are true]; `<` [is smaller than]. -/
theorem walker_gain : predicted unity - predicted walker = 15437091 ∧
    98 * predicted unity < 100 * predicted walker := by
  exact ⟨by decide, by decide⟩

end QualityModel

end Learn
