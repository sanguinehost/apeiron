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
# Layer-0 hybrid verdict on qwen2.5-0.5B (`Learn.HybridVerdict`)

Receipt: `lead-1022/HYB/receipt.md` (`ncal` 8,
layer 0). Arms: `F` = `fold_fit` (sign plane plus the r/c/d pack), `E` = `gptq_e8` beam 4, `W` = the
walker arm selected by calibration `J` on `q k v o`. Every constant is a `measured` input from that
receipt: held-out `KL_out` in millionths of a nat, `J_in` in millionths unless stated, bits as counted.

Verdict: the hybrid (walker on `q k v o`, best of F/E/menu on `gate up down`) is refuted as an
improvement. `F` alone is lower in held-out KL than the walker arm on every attention tensor and lower
than the hybrid on the joint layer at fewer bits. The walker had been ranked only against `E`, which is
not the measured best. The premise `KL_out ∝ J` per tensor (the earlier `Learn.QualityModel`) is
refuted by two pairs below and removed.
-/

namespace Learn

namespace HybridVerdict

/-! ### Per tensor, one linear replaced (held-out `KL_out`, millionths of a nat) -/

/-- Held-out KL of `q` alone under `fold_fit` (F), millionths of a nat. -/
def klF_q : Nat := 8137
/-- Held-out KL of `q` alone under `gptq_e8` (E), millionths of a nat. -/
def klE_q : Nat := 9062
/-- Held-out KL of `q` alone under the walker arm (W), millionths of a nat. -/
def klW_q : Nat := 9548
/-- Held-out KL of `k` alone under `fold_fit` (F), millionths of a nat. -/
def klF_k : Nat := 8635
/-- Held-out KL of `k` alone under `gptq_e8` (E), millionths of a nat. -/
def klE_k : Nat := 9587
/-- Held-out KL of `k` alone under the walker arm (W), millionths of a nat. -/
def klW_k : Nat := 9790
/-- Held-out KL of `v` alone under `fold_fit` (F), millionths of a nat. -/
def klF_v : Nat := 29956
/-- Held-out KL of `v` alone under `gptq_e8` (E), millionths of a nat. -/
def klE_v : Nat := 35102
/-- Held-out KL of `v` alone under the walker arm (W), millionths of a nat. -/
def klW_v : Nat := 35471
/-- Held-out KL of `o` alone under `fold_fit` (F), millionths of a nat. -/
def klF_o : Nat := 189236
/-- Held-out KL of `o` alone under `gptq_e8` (E), millionths of a nat. -/
def klE_o : Nat := 188630
/-- Held-out KL of `o` alone under the walker arm (W), millionths of a nat. -/
def klW_o : Nat := 189522
/-- Held-out KL of `gate` alone under `fold_fit` (F), millionths of a nat. -/
def klF_gate : Nat := 124482
/-- Held-out KL of `gate` alone under `gptq_e8` (E), millionths of a nat. -/
def klE_gate : Nat := 138000
/-- Held-out KL of `up` alone under `fold_fit` (F), millionths of a nat. -/
def klF_up : Nat := 241787
/-- Held-out KL of `up` alone under `gptq_e8` (E), millionths of a nat. -/
def klE_up : Nat := 235375
/-- Held-out KL of `down` alone under `fold_fit` (F), millionths of a nat. -/
def klF_down : Nat := 207099
/-- Held-out KL of `down` alone under `gptq_e8` (E), millionths of a nat. -/
def klE_down : Nat := 257301
/-- Held-out KL of `down` alone under the walker arm (W), millionths of a nat. -/
def klW_down : Nat := 275207

/-- `J_in` of `q` under `fold_fit` (F), millionths. -/
def jF_q : Nat := 1891548
/-- `J_in` of `q` under `gptq_e8` (E), millionths. -/
def jE_q : Nat := 5694486
/-- `J_in` of `q` under the walker arm (W), millionths. -/
def jW_q : Nat := 4192838

/-- `J_in` of `down` under `gptq_e8` (E), ten-thousandths (the receipt's precision). -/
def jE_down : Nat := 11781
/-- `J_in` of `down` under the walker arm (W), ten-thousandths. -/
def jW_down : Nat := 11782

/-! ### Joint layer, all seven linears replaced -/

/-- Joint-layer held-out KL, all seven linears under F, millionths of a nat. -/
def jointF : Nat := 1080841
/-- Joint-layer held-out KL of the hybrid, millionths of a nat. -/
def jointHyb : Nat := 1081356
/-- Joint-layer held-out KL, all seven linears under E, millionths of a nat. -/
def jointE : Nat := 1236013
/-- Joint-layer held-out KL when each tensor takes its lowest-`KL_in` arm, millionths of a nat. -/
def jointMinKlin : Nat := 1387591
/-- Joint-layer calibration `KL_in` of the hybrid, millionths of a nat. -/
def klinHyb : Nat := 493901
/-- Joint-layer calibration `KL_in` of the lowest-`KL_in` selection, millionths of a nat. -/
def klinMinKlin : Nat := 360754
/-- Joint-layer bits under F. -/
def bitsF : Nat := 15590272
/-- Joint-layer bits under the hybrid. -/
def bitsHyb : Nat := 16413536
/-- Joint-layer bits under E. -/
def bitsE : Nat := 16773120

/-! ### Verdicts -/

/-- **`F` beats the walker arm on every attention tensor** in held-out KL. `∧` [and]; `<` [is
smaller than]. -/
theorem fit_beats_walker :
    klF_q < klW_q ∧ klF_k < klW_k ∧ klF_v < klW_v ∧ klF_o < klW_o := by
  exact ⟨by decide, by decide, by decide, by decide⟩

/-- **The hybrid does not improve on `F` for the joint layer**: `F` is lower in held-out KL at fewer
bits. `∧` [and]; `<` [is smaller than]. -/
theorem fit_beats_hybrid_joint : jointF < jointHyb ∧ bitsF < bitsHyb := by
  exact ⟨by decide, by decide⟩

/-- **`F` beats `E` on the joint layer** in held-out KL at fewer bits, and on the summed per-tensor
held-out KL. `E` is lower only on `o` and `up`. `∧` [and]; `<` [is smaller than]. -/
theorem fit_beats_e :
    jointF < jointE ∧ bitsF < bitsE ∧
    klF_q + klF_k + klF_v + klF_o + klF_gate + klF_up + klF_down <
      klE_q + klE_k + klE_v + klE_o + klE_gate + klE_up + klE_down ∧
    klE_o < klF_o ∧ klE_up < klF_up := by
  exact ⟨by decide, by decide, by decide, by decide, by decide⟩

/-- **`KL_out ∝ J` is refuted on `q`**: the walker arm has the lower `J_in` and the higher held-out
KL against `E`, so no positive per-tensor constant maps `J` to held-out KL. `∧` [and]; `<` [is
smaller than]. -/
theorem proportional_refuted_q : jW_q < jE_q ∧ klE_q < klW_q := by
  exact ⟨by decide, by decide⟩

/-- **`KL_out ∝ J` is refuted on `down`**: `J_in` differs by one part in 11 781 between `E` and the
walker arm, while held-out KL differs by more than 6%. `∧` [and]; `≤` [is smaller than or equal
to]; `<` [is smaller than]. -/
theorem proportional_refuted_down :
    jW_down ≤ jE_down + 1 ∧ 106 * klE_down < 100 * klW_down := by
  exact ⟨by decide, by decide⟩

/-- **Per-tensor calibration KL does not compose across the layer**: choosing each tensor by lowest
`KL_in` lowers the joint `KL_in` and raises the joint held-out KL by more than 28%. `∧` [and];
`<` [is smaller than]. -/
theorem min_klin_refuted :
    klinMinKlin < klinHyb ∧ 128 * jointHyb < 100 * jointMinKlin := by
  exact ⟨by decide, by decide⟩

end HybridVerdict

/-! ## Gate-code verdicts on Qwen3.5-0.8B: cost model and held-out KL, clause by clause

| run | receipt (`lead-1022/DOWN/`) | sha256 |
|---|---|---|
| `down`, chunks 8, 9 / 10 | `gramfix2/kl-down-8.log` / `gramfix2/kl-down-10.log` | `b40b9b5d4d00826c` / `b7d13666ee3132bd` |
| `up`, chunks 8, 9 / 10 | `gramfix2/kl-up-8.log` / `gramfix2/kl-up-10.log` | `588f8a5168029bb5` / `4930afe36e969bdf` |
| `gate`, chunks 8, 9 / 10 | `gramfix2/kl-gate-8.log` / `gramfix2/kl-gate-10.log` | `92ca0d41a9d53ef7` / `8d9ebea7ac816bf7` |
| `o` / `o-res2` / `o-cols256x8` | `t11/kl-o.log` / `t12/kl-o.log` / `t17/kl-ocols.log` | `695d167349603179` / `7328fd18034e7324` / `6bd3a35dd4bb1f14` |
| `up-res2all` / `up-rows-sal` | `t07/kl-res2all.log` / `t08/kl-sal.log` | `78bcd5284e2a02d7` / `4ec8e2b1a3ce9c30` |
| `down-res2all` / `down-mix` | `t12/kl-down.log` / `t13/kl-mix.log` | `342cd2898a678e25` / `8e61691c65aca175` |
| `down-x29` (37 calibration chunks) | `t18/kl-down-x29.log` | `17354a48ff9aac6c` |
| `gate+up+down` / `gate+up+down+o` | `t09/plain.log` / `t14/kl-base4.log` | `29860e29444960d7` / `481fce0016312d53` |
| t14 `gate+up-res2all+down-res2all+o-b` / t17 `…+o-c` | `t14/kl-budget.log` / `t17/kl-oc.log` | `14f23063c195addd` / `13108916a4afd4f6` |
| t14x `gate-x29+up-x29b+down-x29b+o-x29b`, chunks 8, 9 / 10 | `t19/kl-t14x.log` / `t19/kl-t14x-10.log` | `625c005f16678646` / `d2579c80ca3c9679` |
| held-out energy, `o` arms / `down-x29` | `t18/he-o.log` / `t18/he-down-x29.log` | `9f7f3648a3a0cbed` / `62aa64aa70a1363f` |
| SpQR masked refit on `up`, `gate` | commit `7619e8e9` body | — |
| gated second-order metric on `up` | commit `8c0d7812` body | — |

Every figure is a `model` input copied from its receipt. KL is in millionths of a nat on held-out chunks
8, 9, 10 of the 0.8B base file (`n_ctx` 512), measured on the crate's `KernelMode::Q8Act` engine path by
`tests/learn_gate_code.rs` `gate_code_kl`. Held-out energy `tr(E H Eᵀ)` is in units of `1e-8`, `H` the
Gram of chunks 8..10. Bits are counted from the `FITK`/`RES` lines of `fold_fit` (sign, row, column and
diagonal scale bits). No theorem produces a measurement; rustc, the kernel ABI and the `unsafe` pool are
outside every statement. Scope: the fitted layers 0, 3, 7, 11, 15, 19, 23, tensors `ffn_gate`, `ffn_up`,
`ffn_down`, `attn_output` (`o`).

The goal (the design notes QUALITY-118, board row ): held-out KL ≤ 0.0913
(gate), ≤ 0.0266 (target) at ≤ 1.18 b/w.
-/


namespace GateCode

/-! ### Cost model: bits per plane and the 1.18 b/w budget over the scope -/

/-- Weights of one `ffn_gate`, `ffn_up` or `ffn_down` (3584 × 1024). -/
def ffWeights : Nat := 3584 * 1024

/-- Weights of one `attn_output` (1024 × 2048). -/
def oWeights : Nat := 1024 * 2048

/-- Fitted layers. -/
def layers : Nat := 7

/-- The scope: four tensors on each fitted layer. -/
def scope : Nat := layers * (3 * ffWeights + oWeights)

/-- One `gate`/`up` plane: sign + row/column/diagonal scales (`143424/6176/5152`). -/
def gateUpPlane : Nat := ffWeights + 143424 + 6176 + 5152

/-- One `down` plane: sign + scales (`122944/21536/17952`). -/
def downPlane : Nat := ffWeights + 122944 + 21536 + 17952

/-- One `o` plane: sign + scales (`73792/12320/10272`). -/
def oPlane : Nat := oWeights + 73792 + 12320 + 10272

/-- The base fit: one plane per tensor per layer. -/
def baseBits : Nat := layers * (2 * gateUpPlane + downPlane + oPlane)

/-- `100 · bits ≤ 118 · weights` is `bits / weights ≤ 1.18`. -/
def fits (bits : Nat) : Prop := 100 * bits ≤ 118 * scope

/-- `fits` is decided by `Nat.decLe`. -/
instance (bits : Nat) : Decidable (fits bits) := by unfold fits; infer_instance

/-- t14's extra planes: `up` L23, `down` L23, `o` L19 and L23 (residual pass, same bits as the base). -/
def t14Extra : Nat := gateUpPlane + downPlane + 2 * oPlane

/-- `down-mix` extra planes: `down-res2` on L0, 3, 15, 19 and `down-res3` (two planes) at L23. -/
def downMixExtra : Nat := 6 * downPlane

/-- `o-res2` on every fitted layer. -/
def oRes2AllExtra : Nat := layers * oPlane

/-- The scope is 91 750 400 weights; the plane counts are the receipts' totals. `∧` [both are true]. -/
theorem plane_bits :
    scope = 91750400 ∧ gateUpPlane = 3824768 ∧ downPlane = 3832448 ∧ oPlane = 2193536 := by
  decide

/-- The base fit is 95 728 640 bits (1.0434 b/w) and leaves 12 536 832 bits under 1.18 b/w. `∧` [both are true]. -/
theorem base_headroom : baseBits = 95728640 ∧ 118 * scope / 100 - baseBits = 12536832 ∧
    fits baseBits := by
  decide

/-- t14 fits: base + four residual planes ≤ 1.18 b/w, with 492 544 bits left. `∧` [both are true]. -/
theorem t14_fits : fits (baseBits + t14Extra) ∧
    118 * scope / 100 - (baseBits + t14Extra) = 492544 := by
  decide

/-- After t14 no further plane fits: the smallest plane (`o`, 2 193 536 bits) exceeds what is left. `¬` [this is not true]. -/
theorem t14_no_further_plane : ¬ fits (baseBits + t14Extra + oPlane) := by decide

/-- `down-mix` (six extra `down` planes) does not fit 1.18 b/w on its own. `¬` [this is not true]. -/
theorem downMix_over_budget : ¬ fits (baseBits + downMixExtra) := by decide

/-- `o-res2` on all seven layers does not fit 1.18 b/w on its own. `¬` [this is not true]. -/
theorem oRes2All_over_budget : ¬ fits (baseBits + oRes2AllExtra) := by decide

/-- Refitting on 37 calibration chunks instead of 8 (`down-x29`) costs no bits: the bit count is a
function of the shape alone. -/
theorem x29_same_bits : baseBits + 0 = baseBits := rfl

/-! ### Held-out KL, chunks 8, 9, 10 (millionths of a nat) -/

/-- Held-out KL of `down`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klDown : List Nat := [102195, 107066, 230160]
/-- Held-out KL of `down-x29` (37 calibration chunks), chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klDownX29 : List Nat := [110124, 120432, 160106]
/-- Held-out KL of `up`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klUp : List Nat := [101351, 98623, 184029]
/-- Held-out KL of `gate`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klGate : List Nat := [69760, 72221, 158222]
/-- Held-out KL of `o`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klO : List Nat := [79845, 72782, 154403]
/-- Held-out KL of `o-res2` (L19, L23 residual planes on every layer), chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klORes2 : List Nat := [28545, 25342, 73530]
/-- Held-out KL of `o-cols256x8`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klOCols : List Nat := [40270, 36140, 119878]
/-- Held-out KL of `up-res2all`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klUpRes2All : List Nat := [73273, 71666, 146639]
/-- Held-out KL of `up-rows-sal`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klUpRowsSal : List Nat := [87138, 89203, 160191]
/-- Held-out KL of `down-res2all`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klDownRes2All : List Nat := [74145, 76914, 200042]
/-- Held-out KL of `down-mix`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klDownMix : List Nat := [33184, 36460, 88015]
/-- Held-out KL of `gate+up+down`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klGateUpDown : List Nat := [268367, 310386, 484157]
/-- Held-out KL of `gate+up+down+o`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klBase4 : List Nat := [371324, 399189, 664152]
/-- Held-out KL of t14 `gate+up-res2all+down-res2all+o-b`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klT14 : List Nat := [289309, 330574, 587555]
/-- Held-out KL of t17 `gate+up-res2all+down-res2all+o-c`, chunks 8, 9, 10, millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klT17 : List Nat := [292347, 330443, 596437]
/-- Held-out KL of t14x, t14's arms each refitted on 37 calibration chunks (`-x29`), chunks 8, 9, 10,
millionths of a nat. `List` [a finite sequence of things, in order]. -/
def klT14x : List Nat := [330531, 391982, 539456]

/-- The gate, three chunks summed (3 × 0.0913). -/
def gateSum : Nat := 3 * 91300

/-- **`down-x29` beats `down` on the mean** at the same bits: 0.146474 → 0.130221 (0.889×). `∧` [both are true]; `<` [is smaller than]. -/
theorem downX29_beats_mean : klDownX29.sum < klDown.sum ∧ 1000 * klDownX29.sum < 890 * klDown.sum := by
  decide

/-- **…but not on every chunk**: `down-x29` loses on chunks 8 and 9 and wins on chunk 10. `∧` [both are true]; `<` [is smaller than]. -/
theorem downX29_per_chunk :
    klDown.getD 0 0 < klDownX29.getD 0 0 ∧ klDown.getD 1 0 < klDownX29.getD 1 0 ∧
      klDownX29.getD 2 0 < klDown.getD 2 0 := by
  decide

/-- **Columns overfit on `o`**: full residual planes beat 256 × 8 column planes on held-out KL, and both
beat the base. `∧` [both are true]; `<` [is smaller than]. -/
theorem oRes2_beats_cols : klORes2.sum < klOCols.sum ∧ klOCols.sum < klO.sum := by decide

/-- The `up` L23 residual plane beats the salience rows plane, and both beat the base. `∧` [both are true]; `<` [is smaller than]. -/
theorem upRes2_beats_rows : klUpRes2All.sum < klUpRowsSal.sum ∧ klUpRowsSal.sum < klUp.sum := by
  decide

/-- `down-res2all` beats `down`; `down-mix` beats both but is over budget (`downMix_over_budget`). `∧` [both are true]; `<` [is smaller than]. -/
theorem down_residual_order : klDownMix.sum < klDownRes2All.sum ∧ klDownRes2All.sum < klDown.sum := by
  decide

/-- **SpQR masked refit fails**: every variant spends bits and raises held-out KL on `up` and `gate`
(three-chunk means: up 0.128001 → 0.136794 / 0.131462 / 0.131416; gate 0.100068 → 0.102440). `∧` [both are true]; `<` [is smaller than]. -/
theorem spqr_fails : 128001 < 136794 ∧ 128001 < 131462 ∧ 128001 < 131416 ∧ 100068 < 102440 := by
  decide

/-- **The gated second-order metric fails on `up`**: both arms raise held-out KL (0.128001 → 0.131957,
0.128513). `∧` [both are true]; `<` [is smaller than]. -/
theorem gated_metric_fails : 128001 < 131957 ∧ 128001 < 128513 := by decide

/-- **t14 beats the four-tensor base** within budget (`t14_fits`): 0.478222 → 0.402479. `<` [is smaller than]. -/
theorem t14_beats_base4 : klT14.sum < klBase4.sum := by decide

/-- The `o-c` allocation (t17) loses to t14's `o-b`. `<` [is smaller than]. -/
theorem t17_loses_to_t14 : klT14.sum < klT17.sum := by decide

/-- **t14x loses to t14** at equal bits (`x29_same_bits`): 1.207438 → 1.261969 summed. It wins chunk 10
(0.587555 → 0.539456) and loses chunks 8 and 9. Held-out chunks 8, 9 and the start of 10 continue the
article that calibration chunks 6 and 7 begin, so the 8-chunk fit has seen their text and the 37-chunk
fit dilutes it (board row ). `∧` [both are true]; `<` [is smaller than]. -/
theorem t14x_loses_to_t14 :
    klT14.sum < klT14x.sum ∧
    klT14x.getD 2 0 < klT14.getD 2 0 ∧
    klT14.getD 0 0 < klT14x.getD 0 0 ∧ klT14.getD 1 0 < klT14x.getD 1 0 := by
  decide

/-- **Adding `o` is near-additive**: the four-tensor KL is within 1% of the sum of the four single-tensor
KLs; three FFN tensors together are sub-additive (below the sum of their singles). `∧` [both are true]; `<` [is smaller than]; `≤` [is smaller than or equal to]. -/
theorem base4_near_additive :
    100 * klBase4.sum ≤ 101 * (klGate.sum + klUp.sum + klDown.sum + klO.sum) ∧
    klGate.sum + klUp.sum + klDown.sum + klO.sum ≤ klBase4.sum ∧
    klGateUpDown.sum < klGate.sum + klUp.sum + klDown.sum := by
  decide

/-- The open gap: t14 is above the 0.0913 gate. `<` [is smaller than]. -/
theorem t14_above_gate : gateSum < klT14.sum := by decide

/-! ### Held-out energy (`1e-8`): the instrument and its per-chunk blind spot -/

/-- Per-layer held-out energy of `down` and `down-x29`, layers 0, 3, 7, 11, 15, 19, 23. `List` [a finite sequence of things, in order]. -/
def heDown : List Nat := [6779009, 16385250, 12692530, 12979590, 55694280, 68244230, 128678200]
/-- Per-layer held-out energy of `down-x29`, same layers. `List` [a finite sequence of things, in order]. -/
def heDownX29 : List Nat := [6356296, 15376400, 11653570, 8667492, 41929810, 55802230, 131663200]

/-- Calibration (chunks 0..8) energy sum of `down`. -/
def ceDown : Nat := 196203900
/-- Calibration (chunks 0..8) energy sum of `down-x29`. -/
def ceDownX29 : Nat := 266770200

/-- **`down-x29` trades calibration energy for held-out energy**: its 8-chunk calibration energy rises
1.36×, its held-out energy falls to 0.9005×, the direction of the mean KL (`downX29_beats_mean`). `∧` [both are true]; `<` [is smaller than]. -/
theorem downX29_energy : heDownX29.sum < heDown.sum ∧ ceDown < ceDownX29 ∧
    10000 * heDownX29.sum < 9006 * heDown.sum := by
  decide

/-- **`down-x29` wins held-out energy on layers 0–19 and loses at L23**: the next arm keeps plain `down`
at L23. `∧` [both are true]; `<` [is smaller than]. -/
theorem downX29_loses_l23 :
    ((heDownX29.zip heDown).take 6).all (fun p => decide (p.1 < p.2)) = true ∧
    heDown.getD 6 0 < heDownX29.getD 6 0 := by
  decide

/-- Per-layer energy of `o-res2` and `o-cols256x8`: calibration and held-out. `List` [a finite sequence of things, in order]. -/
def ceORes2 : List Nat := [4104458, 1956503, 2412911, 1535993, 3430321, 7444158, 29739740]
/-- Per-layer calibration energy of `o-cols256x8`. `List` [a finite sequence of things, in order]. -/
def ceOCols : List Nat := [2789894, 1877993, 2306217, 1437984, 3081559, 5665881, 24846430]
/-- Per-layer held-out energy of `o-res2`. `List` [a finite sequence of things, in order]. -/
def heORes2 : List Nat := [5860646, 2646844, 3546589, 3051808, 5422556, 11298760, 36441880]
/-- Per-layer held-out energy of `o-cols256x8`. `List` [a finite sequence of things, in order]. -/
def heOCols : List Nat := [7384390, 3767718, 5100205, 4702234, 8374974, 18542500, 47560800]

/-- **The calibration order inverts on held-out on 7/7 layers**: column planes have lower calibration
energy than `o-res2` on every layer and higher held-out energy on every layer. `∧` [both are true]; `<` [is smaller than]. -/
theorem oCols_invert_7of7 :
    (ceOCols.zip ceORes2).all (fun p => decide (p.1 < p.2)) = true ∧
    (heORes2.zip heOCols).all (fun p => decide (p.1 < p.2)) = true := by
  decide

end GateCode

end Learn
