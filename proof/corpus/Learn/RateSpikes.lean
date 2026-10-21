import Learn.QuantVerdicts

/-!
# Rate spikes on qwen2.5-0.5B: every clause against its measured constant

| spike | receipt | sha256 |
|---|---|---|
| M1 (plane walk, KL against the streamed fraction `a`) | `lead-1021/PW1/receipt.md` | `82bab1e69d6714b0` |
| KW1 (sign walker at three rates, per-row water-fill at 1.18 b/w) | `lead-1021/KW1/receipt.md` | `526c604d76f7d58a` |
| MT1 (trellis-coded E8, Sturmian rates, per-row water-fill at 1.18 b/w) | `lead-1021/MT1/receipt.md` | `6b94b3b9997de7bd` |

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

/-! ## KW1 · sign walker

`J(W, Wq) = tr((W − Wq) H (W − Wq)ᵀ)` per linear on layers 0, 11, 23 (21 linears), `H` the `Mean` Gram of
256 calibration tokens. Every figure is a median over the 21 linears of `J_arm / J_E`, in thousandths, where
`E` is `gptq_e8` beam 4 (1.125 b/w all-in). `S1_L` emits ±1 from the newest bit of an `L`-bit register at 1 code
bit per weight; `S2` emits {±1, ±3} from the newest two bits; `mix` is the per-row water-fill over
{Z, Sh, S1_8, S2} at 1.18 b/w.
-/

/-- Median `J_S1_1 / J_E`, thousandths. -/
def kwS1_1 : Nat := 3177

/-- Median `J_S1_8 / J_E`, thousandths. -/
def kwS1_8 : Nat := 2881

/-- Median `J_S2 / J_E` (2.125 b/w), thousandths. -/
def kwS2 : Nat := 528

/-- Median `J_mix / J_E` at 1.18 b/w, thousandths. -/
def kwMix : Nat := 2423

/-- Smallest `J_S1_8 / J_E` over the 21 linears (layer 11 `o`), thousandths. -/
def kwS1_8Min : Nat := 1441

/-- **KILL**: both KILL clauses hold (median `S1_8` at or above `E`, median `mix` at or above 0.95 of
`E`), and neither GO clause (0.95 and 0.80).
`∧` [both are true]; `≤` [is smaller than or equal to]; `<` [is smaller than]. -/
theorem kw1_kill :
    1000 ≤ kwS1_8 ∧ 950 ≤ kwMix ∧ 950 < kwS1_8 ∧ 800 < kwMix := by decide

/-- **No linear is won**: the best `S1_8` cell over all 21 linears is still 1.44× `E`.
`<` [is smaller than]. -/
theorem kw1_no_linear_won : 1000 < kwS1_8Min := by decide

/-- **The register does no coding work**: eight state bits move the median `J` by less than 10% against one,
because the emitted value reads only the newest bit, so every sign sequence is reachable at every `L`.
`*` [times]; `<` [is smaller than]. -/
theorem kw1_register_idle : kwS1_1 * 9 < kwS1_8 * 10 := by decide

/-- **The alphabet carries the loss**: the two-bit alphabet at 2.125 b/w is under `E` in the median.
`<` [is smaller than]. -/
theorem kw1_alphabet : kwS2 < 1000 := by decide

/-! ## MT1 · trellis-coded E8

The same 21 linears, Grams and `J` as KW1 (`J_E` reproduced on 21/21 at relative deviation 0). Block
value `a·ρ_s·c[idx]`, `c` the 256-word E8 ∪ O8 codebook, `s` the low `Lr` bits of the previous block's
`idx`; `T_Lr` is the Viterbi over `2^Lr` states with per-survivor whitened feedback, radii a Lloyd fit
to the `E1` gains. `R_x` reads `⌈8(m+1)x⌉ − ⌈8mx⌉` bits for block `m` (7: half codebook; 9, 10: explicit
radii). Every figure is a median over the 21 linears, in thousandths unless named otherwise; `p` marks
`E`'s production scale, the others the diagonal-block refit.
-/

/-- Median `J_T_best / J_E`, refit scale (the brief's decision figure), thousandths. -/
def mtTBest : Nat := 2369

/-- Median `J_mix / J_E`, the nine-option water-fill at 1.18 b/w, thousandths. -/
def mtMix : Nat := 993

/-- Median `J_mix / J_mixE` (the menu with the trellis against the menu without it), ten-thousandths. -/
def mtMixOverMixE : Nat := 9999

/-- Median `J_T_2 / J_T_0`, refit scale, thousandths. -/
def mtT2T0 : Nat := 1087

/-- Median `J_T_2 / J_T_0`, production scale, thousandths. -/
def mtT2T0p : Nat := 1058

/-- Median `J_T_1 / J_T_0`, production scale, thousandths. -/
def mtT1T0p : Nat := 1017

/-- Rows of all 21 linears for which the water-fill bought `E`. -/
def mtRowsE : Nat := 36648

/-- Rows of all 21 linears. -/
def mtRows : Nat := 38016

/-- Median `J_R_5/4 / J_E`, production scale, at 1.375 b/w all-in, thousandths. -/
def mtR54p : Nat := 906

/-- Linears on which `J(R_17/16)` lies strictly between `J(R_1)` and `J(R_9/8)` and below their chord. -/
def mtDialHolds : Nat := 3

/-- The linears measured. -/
def mtLinears : Nat := 21

/-- **KILL**: both KILL clauses hold (the best trellis arm at or above `E`, and the trellis buying
less than 3% over the menu without it), and neither GO clause (0.95 and 0.80).
`∧` [both are true]; `≤` [is smaller than or equal to]; `<` [is smaller than]. -/
theorem mt1_kill :
    1000 ≤ mtTBest ∧ 9700 ≤ mtMixOverMixE ∧ 950 < mtTBest ∧ 800 < mtMix := by decide

/-- **The state costs `J` at both scales**: four states lose to one, and two lose to one, so a radius
chosen by the previous codeword's low bits costs the direction more than the radius returns.
`∧` [both are true]; `<` [is smaller than]. -/
theorem mt1_state_costs : 1000 < mtT2T0 ∧ 1000 < mtT2T0p ∧ 1000 < mtT1T0p := by decide

/-- **The water-fill is `E`**: more than 95% of rows buy `E`.
`*` [times]; `<` [is smaller than]. -/
theorem mt1_mix_is_e : 95 * mtRows < 100 * mtRowsE := by decide

/-- **Rate buys `J`, at a price**: `R_5/4` (1.375 b/w all-in) is under `E` (1.125) in the median.
`<` [is smaller than]. -/
theorem mt1_rate_pays : mtR54p < 1000 := by decide

/-- **The Sturmian dial is not a convex rate curve here**: the 17/16 rung sits below the chord on
fewer than half the linears.
`*` [times]; `<` [is smaller than]. -/
theorem mt1_dial_not_convex : mtDialHolds * 2 < mtLinears := by decide

end RateSpikes

end Learn
