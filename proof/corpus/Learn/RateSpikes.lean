import Learn.QuantVerdicts

/-!
# Rate spikes on qwen2.5-0.5B: every clause against its measured constant

| spike | receipt | sha256 |
|---|---|---|
| M1 (plane walk, KL against the streamed fraction `a`) | `/home/socol/oc-tmp/escalation/lead-1021/PW1/receipt.md` | `82bab1e69d6714b0` |

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

end RateSpikes

end Learn
