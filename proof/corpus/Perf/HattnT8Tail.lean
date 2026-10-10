import Mathlib.Data.Nat.Basic

/-!
# HATTN-T8 tail: which positions fail, and a deployable trigger for them (code prompt, 524288)

`docs/evidence/hattn-decode-tail.md` (tail-trigger section): the 64 tail positions of the code prompt at `t = 524288`
(`g7-code`, Qwen2.5-7B-Instruct-1M Q4_K_M) under `grp-fix2048-r8`, with per-position whole-model KLD and five
per-position proxies from `KV_TAIL_POS=1` (`attn_cut_probe.hd128qp`; outputs `g17-tailpos.out`, `g18-tailwide-8192.out`,
`g18-tailwide-32768.out`). Every list entry is a `model` input copied from those outputs and rounded to the stated unit;
the verdicts are `decide` computations over the lists. Predictions: `PREDICTIONS-T8-TAIL.md`, sha256 `c9d7b8a6…`
(rounds 1 to 3 are appended to one file; round 2 was registered after round 1's results, round 3 after round 2's).
A position fails when its KLD exceeds `0.05` (`5000` in units of `1e-5`). The composition in round 3 takes each
position's KLD from the wide arm's separate whole-model run, so it omits coupling through the tail keys. rustc, LLVM,
the f32 dump kernels and the instrument are outside every statement; one prompt, one model; the entropy threshold was
chosen on this data.
-/

namespace Perf

namespace HattnT8Tail

/-- Whole-model KLD at the 64 tail positions, `grp-fix2048-r8`, in units of `1e-5` (`g17-tailpos.out`, `POS ... kld`). -/
def kld : List ℕ :=
  [1, 1, 127, 17846, 3, 190, 4, 0, 13224, 40, 351, 122, 21, 345, 387, 4228, 12031, 1575, 64, 35177, 74892, 12, 290, 7, 16, 34065, 57, 3619, 1199, 252, 980, 30881, 13, 3134, 58, 0, 17, 32, 10, 0, 76, 245809, 88, 1, 69297, 7175, 251, 11224, 168059, 269, 145, 180, 0, 42, 4, 27, 25339, 19, 2298, 11642, 2518, 35066, 13, 4]

/-- Mean over 28 layers and 28 heads of the head KL `-ln(kept mass)`, per position, `1e-5` (undeployable). -/
def hkl : List ℕ :=
  [83851, 92068, 92562, 78443, 68434, 76065, 74206, 93549, 106086, 94452, 98856, 103491, 90752, 106165, 99204, 102866, 99786, 113133, 131782, 127701, 110819, 79914, 80361, 88249, 79094, 102854, 103278, 102000, 90583, 93567, 92271, 85765, 81115, 77034, 81205, 74032, 86100, 94296, 92382, 101076, 85430, 93776, 93077, 84759, 83974, 99681, 88162, 104247, 100547, 107649, 89008, 83129, 105591, 82548, 92260, 105791, 84233, 92122, 86481, 80889, 96065, 99094, 139497, 145112]

/-- Max over layers and heads of the head KL, per position, `1e-4` (undeployable). -/
def hklMax : List ℕ :=
  [86891, 94228, 79523, 90033, 84973, 71906, 87245, 103879, 84846, 105680, 84173, 96014, 100587, 75794, 88509, 88242, 79673, 81723, 76595, 89924, 79362, 88646, 72926, 82673, 100843, 77195, 93662, 92992, 84812, 85522, 93852, 86837, 79682, 72512, 94632, 98135, 80366, 94482, 82750, 97280, 80070, 94442, 77917, 92044, 77281, 93204, 87983, 79166, 101723, 89175, 86195, 88359, 95969, 80238, 91938, 84773, 82453, 61868, 86111, 83883, 87885, 72869, 82596, 83789]

/-- P1: mean over heads of `ln(1 + B/Z_read)` from the unread page bounds, `1e-5`. -/
def pMean : List ℕ :=
  [2930708, 3010135, 3087978, 3168859, 3118178, 3116728, 2999661, 2952857, 2986686, 3027574, 3031592, 2977448, 3196515, 3060234, 3009725, 3052767, 3039597, 3004283, 2984854, 3105207, 3116665, 3101850, 3077348, 3037408, 3181897, 3021083, 3015042, 3011080, 3042016, 2958102, 3042243, 3121255, 2923512, 3038924, 2975612, 3055165, 3047320, 3106265, 3064193, 3160954, 3014453, 3092831, 3132936, 3162528, 3081307, 3090548, 3066184, 2989339, 2974330, 3001695, 3021436, 2980785, 3079542, 3032881, 3029773, 3088059, 3089799, 3033373, 2971342, 3053451, 3043175, 3132396, 2960426, 2986534]

/-- P2: mean over heads of the best unread bound minus the best read score, `1e-5`. -/
def dMean : List ℕ :=
  [2373920, 2469140, 2556390, 2664860, 2607550, 2600990, 2473990, 2409160, 2419230, 2480330, 2488050, 2431810, 2668130, 2521070, 2466850, 2512960, 2472040, 2451210, 2443330, 2561700, 2585820, 2583720, 2563180, 2511030, 2632470, 2485110, 2482860, 2481810, 2518920, 2431630, 2517500, 2612640, 2373370, 2534890, 2469530, 2547890, 2543490, 2602550, 2552720, 2652260, 2452830, 2572740, 2611780, 2665050, 2550060, 2571220, 2549280, 2475150, 2415370, 2460280, 2486790, 2455050, 2563360, 2507010, 2497120, 2550680, 2527450, 2491870, 2426190, 2511800, 2502140, 2634830, 2441340, 2461240]

/-- Output entropy of the approximate arm (nats), per position, `1e-4`. -/
def ent : List ℕ :=
  [11, 3, 142, 1394, 24, 291, 29, 2, 5532, 5, 360, 147, 44, 334, 327, 1922, 4574, 1508, 77, 15793, 8138, 24, 365, 10, 9, 10678, 115, 2134, 1200, 269, 777, 9945, 70, 1648, 89, 0, 98, 278, 12, 0, 319, 5217, 109, 2, 8154, 2630, 1643, 7634, 6499, 232, 308, 229, 0, 97, 10, 121, 378, 52, 1162, 4032, 3353, 12367, 29, 12]

/-- `10^6` minus the approximate arm top-1 margin (top-1 minus top-2 log-probability, `1e-4`): larger is flatter. -/
def margInv : List ℕ :=
  [897047, 879492, 925100, 961841, 908480, 939347, 900383, 874132, 975043, 886913, 929986, 923486, 910944, 934364, 940392, 967992, 977913, 944296, 918468, 976609, 999578, 911997, 942353, 896496, 886959, 986139, 915738, 961328, 939751, 935345, 946498, 989212, 922381, 965868, 926288, 855139, 913904, 942201, 904895, 836651, 930832, 975726, 928057, 878485, 999485, 953714, 962662, 973338, 984805, 941164, 921885, 937972, 858166, 916857, 899475, 924373, 940919, 907758, 962513, 978139, 977566, 989958, 911607, 893785]

/-- `grp-fix8192-r8` per-position KLD, `1e-5` (`g18-tailwide-8192.out`). -/
def kld8192 : List ℕ :=
  [1, 0, 10, 12419, 10, 339, 4, 0, 2, 30, 0, 3, 13, 2, 0, 23, 1, 0, 1, 13, 1273, 3, 15, 0, 18, 9, 72, 134, 55, 3, 0, 46968, 13, 137, 13, 0, 5, 65, 1, 0, 36, 1164, 3, 0, 859, 3814, 907, 1909, 1770, 7, 15, 28, 0, 7, 1, 3, 11420, 1, 40, 159, 787, 2680, 1, 0]

/-- `grp-fix32768-r8` per-position KLD, `1e-5` (`g18-tailwide-32768.out`). -/
def kld32768 : List ℕ :=
  [0, 0, 2, 552, 7, 37, 1, 0, 2, 39, 0, 3, 11, 1, 1, 3, 1, 0, 0, 51, 119, 4, 33, 0, 17, 2, 8, 0, 12, 1, 0, 956, 2, 10, 5, 0, 3, 3, 1, 0, 13, 68, 5, 0, 15, 759, 563, 477, 526, 2, 4, 9, 0, 3, 0, 4, 7543, 1, 23, 33, 341, 2577, 1, 0]

/-- A position fails the gate. -/
def isFail (k : ℕ) : Bool := decide (5000 < k)

/-- Twice the Mann-Whitney count of `(failing, passing)` pairs in which the failing position has the larger score,
plus one per tie: `aucNum / (2 · #fail · #pass)` is the area under the ROC curve of the score for the failures. -/
def aucNum (l : List (ℕ × ℕ)) : ℕ :=
  (l.filter (fun p => isFail p.1)).foldl
    (fun a f =>
      a + (l.filter (fun p => !isFail p.1)).foldl
        (fun b n => b + (if n.2 < f.2 then 2 else if n.2 = f.2 then 1 else 0)) 0) 0

/-- Positions flagged by the threshold `τ` on a score, and failing positions among them. -/
def flagged (l : List (ℕ × ℕ)) (τ : ℕ) : ℕ := (l.filter (fun p => τ ≤ p.2)).length

def caught (l : List (ℕ × ℕ)) (τ : ℕ) : ℕ := ((l.filter (fun p => τ ≤ p.2)).filter (fun p => isFail p.1)).length

/-- The 15 failing and 49 passing positions: `2 · 15 · 49 = 1470` is the AUC denominator. -/
theorem population : (kld.filter isFail).length = 15 ∧ (kld.filter (fun k => !isFail k)).length = 49 ∧
    kld.length = 64 := by
  decide +kernel

/-! ## Round 1 and 2 verdicts (`PREDICTIONS-T8-TAIL.md`) -/

/-- **TP1 held.** The instrument reproduces the flat arm: mean KLD in `[0.1268, 0.1278]` (sum over 64 positions, units
of `1e-5`) and 15 of 64 over the gate. -/
theorem tp1_held : 1268 * 640 ≤ kld.sum ∧ kld.sum ≤ 1278 * 640 ∧ (kld.filter isFail).length = 15 := by decide +kernel

/-- **TP2 refuted.** The mean head KL, an oracle quantity, ranks the failing positions at AUC `0.600`, not `0.9`. -/
theorem tp2_refuted : aucNum (kld.zip hkl) = 882 ∧ ¬ 9 * 1470 ≤ 10 * aucNum (kld.zip hkl) := by decide +kernel

/-- **TP3 refuted.** `P1`, the bound-mass proxy, has AUC `0.656`. -/
theorem tp3_refuted : aucNum (kld.zip pMean) = 964 ∧ ¬ 8 * 1470 ≤ 10 * aucNum (kld.zip pMean) := by decide +kernel

/-- **TP4 refuted.** `P2`, the bound gap, has AUC `0.611`. -/
theorem tp4_refuted : aucNum (kld.zip dMean) = 898 ∧ ¬ 8 * 1470 ≤ 10 * aucNum (kld.zip dMean) := by decide +kernel

/-- **TP5 refuted.** For every threshold on `P1` or `P2` that catches 12 failing positions, more than 25 of 64 are
flagged. -/
theorem tp5_refuted :
    (∀ τ ∈ pMean, 12 ≤ caught (kld.zip pMean) τ → 25 < flagged (kld.zip pMean) τ) ∧
      ∀ τ ∈ dMean, 12 ≤ caught (kld.zip dMean) τ → 25 < flagged (kld.zip dMean) τ := by
  decide +kernel

/-- **TP6 refuted.** The max head KL over layers and heads has AUC `0.407`, below chance: a few extreme heads do not
mark the failing positions. -/
theorem tp6_refuted : aucNum (kld.zip hklMax) = 598 ∧ aucNum (kld.zip hklMax) < 1470 := by decide +kernel

/-- **TP7 held.** The approximate arm's output entropy separates failing from passing positions at AUC `0.978`. -/
theorem tp7_held : aucNum (kld.zip ent) = 1438 ∧ 8 * 1470 ≤ 10 * aucNum (kld.zip ent) := by decide +kernel

/-- **TP8 held.** The approximate arm's top-1 margin separates them at AUC `0.965`. -/
theorem tp8_held : aucNum (kld.zip margInv) = 1418 ∧ 8 * 1470 ≤ 10 * aucNum (kld.zip margInv) := by decide +kernel

/-- **TP9 held.** The threshold `0.4` nats on the output entropy flags 12 of 64 positions and all 12 are failures. -/
theorem tp9_held : flagged (kld.zip ent) 4000 = 12 ∧ caught (kld.zip ent) 4000 = 12 := by decide +kernel

/-- The 52 positions below the `0.4` threshold carry KLD sum `73450` (mean `0.0115` over 64 positions): the remaining
error if the flagged positions were recomputed exactly. -/
theorem unflagged_mean :
    (((kld.zip ent).filter (fun p => p.2 < 4000)).map Prod.fst).sum = 73450 := by decide +kernel

/-! ## Round 3: a wider far budget at the flagged positions -/

/-- Bytes per dense per position of the three budgets, units of `1e-4`: `b = 2048, 8192, 32768`. -/
def bytes2048 : ℕ := 64

def bytes8192 : ℕ := 181

def bytes32768 : ℕ := 650

/-- Position KLD under the hybrid: the wide arm's value where `ent ≥ τ`, else the `2048` arm's. -/
def hybrid (wide : List ℕ) (τ : ℕ) : List ℕ :=
  ((kld.zip ent).zip wide).map (fun p => if τ ≤ p.1.2 then p.2 else p.1.1)

/-- **TP10 held.** At `b = 32768`, 14 of the 15 failing positions pass. -/
theorem tp10_held :
    ((kld.zip kld32768).filter (fun p => isFail p.1 && !isFail p.2)).length = 14 ∧
      12 ≤ ((kld.zip kld32768).filter (fun p => isFail p.1 && !isFail p.2)).length := by
  decide +kernel

/-- **TP11 held.** At `b = 8192`, 12 of the 15 failing positions pass. -/
theorem tp11_held :
    ((kld.zip kld8192).filter (fun p => isFail p.1 && !isFail p.2)).length = 12 ∧
      8 ≤ ((kld.zip kld8192).filter (fun p => isFail p.1 && !isFail p.2)).length := by
  decide +kernel

/-- **TP12 held.** The hybrid with `b = 32768` at the 12 flagged positions has mean KLD `0.0122` (sum `78277`), 3
positions over the gate, and costs `(64 · 64 + 12 · 650) / 64` in `1e-4` of dense: every flagged position pays the
first pass and the wide pass. -/
theorem tp12_held :
    (hybrid kld32768 4000).sum = 78277 ∧ (hybrid kld32768 4000).sum ≤ 500 * 640 ∧ 64 * bytes2048 + 12 * bytes32768 ≤ 500 * 64 := by
  decide +kernel

/-! ## Findings -/

/-- **The tail is where the model is uncertain.** With the entropy trigger at `0.4` nats, 12 positions are flagged
and 3 of the 15 failures lie below the threshold: 12 of the 15 failures are positions where the approximate arm's
next-token distribution is flat. -/
theorem failures_are_uncertain :
    (((kld.zip ent).filter (fun p => p.2 < 4000)).filter (fun p => isFail p.1)).length = 3 ∧
      ((kld.zip ent).filter (fun p => 4000 ≤ p.2)).length = 12 := by
  decide +kernel

/-- **Flat `b = 8192` passes on code at 524288** (mean KLD `0.0136`, sum `87266`; 3 positions over) at `0.0181` of
dense bytes, where the prose prompt at the same `t` and `b` has mean `0.2612` (`g10-t524288-b8192-a1.out`). -/
theorem flat8192_passes_on_code : kld8192.sum = 87266 ∧ kld8192.sum ≤ 500 * 640 ∧
    (kld8192.filter isFail).length = 3 ∧ 5 * 640 * 100 < 2612 * 640 := by
  decide +kernel

/-- **The trigger halves the bytes of uniform `b = 8192` at a mean under the gate.** The hybrid (`b = 8192` at the
flagged positions) has mean KLD `0.0203` (sum `130257`, 4 over the gate) and `64 · 64 + 12 · 181 = 6268` over `64`
positions, `0.0098` of dense: against uniform `b = 8192` at `0.0181`, 1.85x fewer bytes. -/
theorem hybrid8192_cheaper :
    (hybrid kld8192 4000).sum = 130257 ∧ (hybrid kld8192 4000).sum ≤ 500 * 640 ∧
      (hybrid kld8192 4000).length = 64 ∧ 64 * bytes2048 + 12 * bytes8192 = 6268 ∧
      6268 < 64 * bytes8192 ∧ 1 * 64 * bytes8192 < 2 * 6268 := by
  decide +kernel

end HattnT8Tail

end Perf
