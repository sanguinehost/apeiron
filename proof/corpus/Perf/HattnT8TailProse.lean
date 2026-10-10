import Mathlib.Data.Nat.Basic
import Perf.HattnT8Tail

/-!
# HATTN-T8 tail trigger on the prose prompt (524288)

`docs/evidence/hattn-decode-tail.md` (tail-trigger section): the code prompt's entropy trigger (`Perf.HattnT8Tail`)
re-measured on the prose dump `g7-1m` at `t = 524288` (Qwen2.5-7B-Instruct-1M Q4_K_M), same instrument
(`attn_cut_probe.hd128qp`, `KV_TAIL_POS=1`), three far budgets in one run each (`g19-prose-2048.out`,
`g19-prose-8192.out`, `g19-prose-32768.out`). List entries are `model` inputs copied from those outputs: KLD in units
of `1e-5`, entropy in `1e-4` nats. Predictions: `PREDICTIONS-T8-TAIL.md` round 4, sha256 `e045e2e7…`.
A position fails when its KLD exceeds `0.05`. Composition as in `Perf.HattnT8Tail`. rustc, LLVM and the instrument
are outside every statement; one prompt, one model.
-/

namespace Perf

namespace HattnT8TailProse

open HattnT8Tail (isFail aucNum flagged caught)

/-- `grp-fix2048-r8` per-position whole-model KLD, `1e-5`. -/
def kld : List ℕ :=
  [26819, 37635, 26271, 17274, 25139, 15423, 107141, 54210, 73612, 21322, 33992, 30323, 32805, 44404, 50119, 35308, 15657, 20142, 12408, 11421, 9455, 29845, 20503, 42191, 3274, 9657, 15012, 23864, 33180, 26826, 11546, 84279, 70731, 19985, 17457, 32752, 12328, 52074, 20854, 11123, 15185, 13885, 14665, 18880, 17614, 16060, 45968, 45942, 45511, 36979, 179437, 34785, 31455, 23577, 11715, 69163, 22882, 51062, 18492, 25692, 23487, 22721, 132444, 604]

/-- Output entropy of the `2048` arm, `1e-4` nats. -/
def ent : List ℕ :=
  [55504, 67811, 27287, 56936, 44153, 38362, 53610, 57714, 47104, 57066, 69781, 75155, 58681, 62794, 68822, 27350, 39068, 56175, 51970, 71809, 64451, 75450, 84046, 10673, 13836, 66723, 67965, 81156, 54156, 60001, 54001, 71372, 54596, 62588, 83023, 56488, 68124, 69137, 68918, 59907, 70029, 84321, 77704, 53968, 83171, 50927, 68676, 2551, 71861, 72012, 36530, 68256, 49193, 69048, 57670, 62442, 64600, 59971, 55196, 53965, 64794, 78937, 40156, 207]

def kld8192 : List ℕ :=
  [15677, 15864, 18985, 10896, 15523, 13171, 89271, 32553, 70727, 16855, 28565, 26713, 28240, 32971, 46134, 28539, 16515, 18354, 11031, 6752, 7214, 27931, 18245, 21166, 6794, 6041, 14392, 18216, 22268, 17087, 7169, 34055, 50425, 15034, 12953, 30292, 8209, 39329, 14005, 7718, 15204, 12384, 11278, 16963, 15592, 13226, 14436, 42826, 21799, 28573, 162417, 23376, 18399, 15120, 10846, 87837, 23196, 42817, 16587, 16788, 17458, 17665, 104044, 842]

def kld32768 : List ℕ :=
  [9608, 12963, 8864, 9360, 14249, 7738, 52593, 22731, 51139, 15982, 16525, 13702, 20686, 25373, 47937, 19634, 9982, 11336, 11344, 3907, 4540, 10322, 10307, 9889, 4976, 4911, 9381, 8895, 8138, 5771, 2748, 7232, 34086, 9803, 8345, 22087, 4933, 23118, 8107, 4257, 9691, 7987, 6961, 10661, 11431, 7752, 10636, 28144, 10408, 22005, 93613, 11159, 9011, 8888, 7230, 81768, 16292, 25752, 12898, 8641, 11305, 12925, 66253, 481]

/-- Position KLD under the hybrid: the wide arm's value where `ent ≥ τ`, else the `2048` arm's. -/
def hybrid (wide : List ℕ) (τ : ℕ) : List ℕ :=
  ((kld.zip ent).zip wide).map (fun p => if τ ≤ p.1.2 then p.2 else p.1.1)

/-- **TQ1 held.** The instrument reproduces the prose flat arm: mean KLD in `[0.3360, 0.3380]` and 62 of 64 over. -/
theorem tq1_held : 3360 * 640 ≤ kld.sum ∧ kld.sum ≤ 3380 * 640 ∧ (kld.filter isFail).length = 62 := by
  decide +kernel

/-- **The prose population has two passing positions.** `2 · 62 · 2 = 248` is the AUC denominator, so a separation
score is computed over two negatives. -/
theorem population : (kld.filter isFail).length = 62 ∧ (kld.filter (fun k => !isFail k)).length = 2 ∧
    kld.length = 64 := by
  decide +kernel

/-- **TQ2 held on a degenerate population.** The arm entropy ranks the 62 failures above the 2 passing positions
at `aucNum = A` of `248`; with two negatives the figure carries little weight. -/
theorem tq2_held_degenerate : aucNum (kld.zip ent) = 244 ∧ 8 * 248 ≤ 10 * aucNum (kld.zip ent) := by
  decide +kernel

/-- **TQ3 held, and the trigger is not selective.** The `0.4` nat threshold flags 62 of 64 positions and catches 61
of the 62 failures: it flags everything. -/
theorem tq3_held_unselective : flagged (kld.zip ent) 4000 = 62 ∧ caught (kld.zip ent) 4000 = 61 ∧
    8 * 62 ≤ 10 * caught (kld.zip ent) 4000 := by
  decide +kernel

/-- **TQ4 held.** The hybrid with `b = 8192` has mean KLD `2616.2969` (sum 1674430) on prose: no better than flat `b = 8192`
(`0.2612`, sum 1671552), 63 positions over. -/
theorem tq4_held : (hybrid kld8192 4000).sum = 1674430 ∧ 500 * 640 < (hybrid kld8192 4000).sum ∧
    (kld8192.filter isFail).length = 63 := by
  decide +kernel

/-- **TQ5 refuted.** Flat `b = 32768` on prose has mean KLD `0.1687` (sum 1079391), 56 of 64 positions over:
a 16-fold larger far budget does not bring the prose prompt under the gate. -/
theorem tq5_refuted : kld32768.sum = 1079391 ∧ 500 * 640 < kld32768.sum ∧
    (kld32768.filter isFail).length = 56 := by
  decide +kernel

/-- **The far budget closes the gap by shrinking it, not by crossing the gate.** Mean KLD falls with the budget
(`0.3370`, `0.2612`, `0.1687`) and stays above `0.05` at every budget measured on prose. -/
theorem budget_monotone_not_enough : kld.sum = 2156566 ∧ kld8192.sum < kld.sum ∧ kld32768.sum < kld8192.sum ∧
    500 * 640 < kld32768.sum := by
  decide +kernel

end HattnT8TailProse

end Perf
