import Mathlib.Data.Nat.Basic

/-!
# attention-cut T8 token-oracle selection at 524288: ranking versus budget

the hattn-decode-tail note (token-oracle section): Qwen2.5-7B-Instruct-1M Q4_K_M, `t = 524288`, 64 tail
positions. Arm `grp-fix<b>-r8` with `KV_TAIL_ORACLE=1`: each query head keeps its own exact top-`n_far` far tokens by
true score, `n_far` the count the page set selects (`attn_cut_probe.hd128or`, `oracle.sh`, outputs `g21-or-*.out`).
Sets are per query head, not a KV-group union, so the figures are quality diagnostics and carry no byte claim.
Entries are `model` inputs copied from those outputs: mean whole-model KLD in units of `1e-4`. Predictions:
`PREDICTIONS-T8-ORACLE.md`, sha256 `c6992a2e3d23`. rustc, LLVM and the instrument are outside every statement; one
prompt per class, one model.
-/

namespace Perf

namespace HattnT8Oracle

/-- Oracle selection: prose at 2048 and 8192, code at 2048. -/
def proseOr2048 : ℕ := 553

def proseOr8192 : ℕ := 275

def codeOr2048 : ℕ := 5

/-- Flat Quest-bound arm of the same budget (`g19-prose-*`, `HattnT8TailProse`; `HattnT8Tail`). -/
def proseFlat2048 : ℕ := 3370

def proseFlat8192 : ℕ := 2612

def codeFlat2048 : ℕ := 1273

/-- **OR1 refuted.** Exact top-2048 per head on prose gives mean KLD 0.0553, above the `0.05` gate. -/
theorem or1_refuted : ¬ proseOr2048 ≤ 500 := by decide

/-- **OR2 held.** The oracle more than halves the flat figure: 0.0553 against 0.3370, a factor above six. -/
theorem or2_held : proseOr2048 < 1500 ∧ 6 * proseOr2048 < proseFlat2048 := by decide

/-- **OR3 held.** Exact top-8192 per head on prose passes the gate (0.0275). -/
theorem or3_held : proseOr8192 ≤ 500 := by decide

/-- **OR4 held.** Exact top-2048 per head on code gives 0.0005, far under the `0.02` bound. -/
theorem or4_held : codeOr2048 ≤ 200 := by decide

/-- **OR5 held.** The oracle never exceeds the flat arm of the same budget. -/
theorem or5_held : proseOr2048 ≤ proseFlat2048 ∧ proseOr8192 ≤ proseFlat8192 ∧ codeOr2048 ≤ codeFlat2048 := by
  decide

/-- **Finding: on prose, selection quality is the bottleneck.** Perfect selection at 2048 beats the flat arm at
32768 (0.1687) by a factor above three, so the Quest-bound ranking, not the budget, sets the prose gap. -/
theorem selection_beats_budget : 3 * proseOr2048 < 1687 := by decide

end HattnT8Oracle

end Perf
