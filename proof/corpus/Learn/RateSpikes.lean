import Learn.QuantVerdicts

/-!
# Rate spikes on qwen2.5-0.5B: every clause against its measured constant

| spike | receipt | sha256 |
|---|---|---|
| M1 (plane walk, KL against the streamed fraction `a`) | `/home/socol/oc-tmp/escalation/lead-1021/PW1/receipt.md` | `82bab1e69d6714b0` |
| KW1 (sign walker at three rates, per-row water-fill at 1.18 b/w) | `/home/socol/oc-tmp/escalation/lead-1021/KW1/receipt.md` | `526c604d76f7d58a` |
| MT1 (trellis-coded E8, Sturmian rates, per-row water-fill at 1.18 b/w) | `/home/socol/oc-tmp/escalation/lead-1021/MT1/receipt.md` | `6b94b3b9997de7bd` |
| KW1 + MT1 same-scale attribution (rows re-read, no new run) | `/home/socol/oc-tmp/escalation/lead-1021/MT1/attr/attr.out` | `f04177880afb495e` |

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
`/home/socol/oc-tmp/escalation/lead-1021/MT1/attr/attr.sh` from KW1's `table.txt` and MT1's
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

/-- **At equal beam the walker with a radius beats `E`** on 20 of 21 linears, by up to 1.9× (layer 0 `q`).
`∧` [both are true]; `<` [is smaller than]. -/
theorem mt1_walker_beats_beam1 :
    mtT0pE1 < 1000 ∧ mtT0pE1Won + 1 = spLinears ∧ mtT0pE1Min < 550 := by decide

/-- **Against full `E` the beam-1 walker wins three linears and loses the median**; the median gap (1.150)
is smaller than what beam 4 and `wscale` buy `E` (1.283), which they buy on every linear. Whether the
remaining gap is the beam is the open question of `APEIRON-MT-SPIKE-W2`.
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

end RateSpikes

end Learn
