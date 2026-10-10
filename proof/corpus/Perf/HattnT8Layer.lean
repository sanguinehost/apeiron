import Perf.HattnT8

/-!
# HATTN-T8 layer attribution of the 524288 failure on Qwen2.5-7B-Instruct-1M

`docs/evidence/hattn-decode-tail.md` (layer attribution section): the flat arm `grp-fix2048-r8` at `t = 524288`, the 64
tail positions of the dump `hattn-long/g7-1m`, instrument `attn_cut_probe.hd128ql`
(`KV_TAIL_DENSE`, `KV_TAIL_SPARSE`, `layer.sh`, outputs `g13-t524288-<run>.out`). `D`: layers 14 and 21 dense, flat
elsewhere. `Q1`..`Q4`: flat only in layers 0-6, 7-13, 14-20, 21-27, dense elsewhere. Mean whole-model KLD in
ten-thousandths; `allFlat` is `Perf.HattnT8Q9`'s flat entry. Predictions: `PREDICTIONS-T8-LAYER.md`, sha256
`3534046b…`. rustc, LLVM, the f32 dump kernels and the instrument are outside every statement; one prompt, one model.
-/

namespace Perf

namespace HattnT8Layer

def allFlat : ℕ := 3370

def denseOneFour : ℕ := 3202

/-- Flat only in layers 0-6, 7-13, 14-20, 21-27. -/
def quarters : List ℕ := [1821, 479, 1149, 666]

/-- **L1 refuted.** Making layers 14 and 21 dense moves the KLD from `0.3370` to `0.3202`, not under `0.1685`. -/
theorem l1_refuted : ¬ (denseOneFour ≤ 1685) ∧ allFlat - denseOneFour = 168 := by decide

/-- **L2 held.** `D` does not pass the `0.05` gate. -/
theorem l2_held : ¬ (denseOneFour ≤ 500) := by decide

/-- **L3 refuted.** The damage is not where H9 puts it: the first quarter (layers 0-6) alone is `0.1821`, above the
third (`0.1149`) and fourth (`0.0666`); only the comparison with the second quarter (`0.0479`) holds. -/
theorem l3_refuted :
    ¬ (quarters.getD 0 0 < quarters.getD 2 0) ∧ ¬ (quarters.getD 0 0 < quarters.getD 3 0) ∧
      quarters.getD 1 0 < quarters.getD 2 0 ∧ quarters.getD 1 0 < quarters.getD 3 0 := by
  decide

/-- **L4 held (control).** Every single-quarter run is below the all-layer flat arm. -/
theorem l4_held : ∀ k ∈ quarters, k < allFlat := by decide

/-! ## Findings -/

/-- **Flat selection at 2048 tokens holds the gate in layers 7-13 alone** (`0.0479`), fails it in every other
quarter, and the first seven layers carry the largest share (`0.1821` of `0.3370`). -/
theorem early_layers_dominate :
    quarters.getD 1 0 ≤ 500 ∧ 500 < quarters.getD 0 0 ∧ 500 < quarters.getD 2 0 ∧ 500 < quarters.getD 3 0 ∧
      2 * quarters.getD 0 0 > allFlat := by
  decide

/-- The quarters' KLD sums to more than the all-flat KLD: attribution is sub-additive, so a per-layer share is an
upper bound on what removing the layer buys. -/
theorem quarters_overshoot : allFlat < quarters.sum := by decide


/-! ## Dense prefixes (`PREDICTIONS-T8-LAYER2.md`, sha256 `025488e1…`; `layer2.sh`, `g14-t524288-E<n>.out`)

`E1`..`E4`: layers `{0,1}`, `{0..3}`, `{0..6}`, `{0..6, 14..20}` dense, flat elsewhere. -/

/-- Mean KLD of `E1`..`E4`. -/
def prefixes : List ℕ := [3019, 3181, 3071, 1033]

/-- **M1 refuted.** Dense layers 0 and 1 give `0.3019`, not under `0.247`. -/
theorem m1_refuted : ¬ (prefixes.getD 0 0 ≤ 2470) := by decide

/-- **M2 refuted.** Dense layers 0-6 give `0.3071`, not in `[0.10, 0.20]`. -/
theorem m2_refuted : ¬ (prefixes.getD 2 0 ≤ 2000) := by decide

/-- **M3 held.** No dense-prefix run passes the `0.05` gate, including half the stack dense (`E4`, `0.1033`). -/
theorem m3_held : ∀ k ∈ prefixes, 500 < k := by decide

/-- **M4 held.** `E1` removes at least half of what `E3` removes from the all-flat `0.3370`. -/
theorem m4_held : 2 * (allFlat - prefixes.getD 0 0) ≥ allFlat - prefixes.getD 2 0 := by decide

/-- **M5 refuted.** The KLD is not monotone in the dense prefix: `E2` (`0.3181`) is above `E1` (`0.3019`). -/
theorem m5_refuted : ¬ (prefixes.getD 1 0 ≤ prefixes.getD 0 0) := by decide

/-- **Finding: the loss is distributed, not localised.** Each quarter alone costs `0.0479`-`0.1821` (`quarters`), yet
making any prefix of up to 7 layers dense moves the all-flat `0.3370` by at most `0.0351`; half the stack dense
(`E4`) still fails the gate. Per-layer errors compound through depth, so a per-layer fix does not restore the gate. -/
theorem compounding :
    ∀ k ∈ prefixes.take 3, allFlat - 351 ≤ k ∧ 500 < prefixes.getD 3 0 := by decide

end HattnT8Layer

end Perf
