import Substrate.Algebra.Neural.NullTransformer.Core
import Substrate.Algebra.Neural.NullTransformer.Layers

/-!
# The Transformer, Built From The Null Cell — forward, loss, gradient, step

@name A Whole Modern Language Model, Rebuilt Out of the Nothing-Cell
@child The big AI models are made of a handful of moves stacked up. This file writes one out in
  full — the words going in, the layers, the answer coming out, the score for being wrong, and the
  nudge that makes it less wrong — and shows every one of those moves is the same tiny two-part
  object doing its job.
@scientist A Llama-class pre-norm transformer (RMSNorm · RoPE · causal multi-head attention ·
  SwiGLU · residual · tied-shape unembedding · mean next-token cross-entropy) defined over ℝ as a
  function of one flat parameter vector `θ : Fin P → ℝ`, with its two nonlinearities DERIVED from
  the split-complex null cell: `exp` is the null-cell boost's `+`-lightcone channel, softmax is
  `B/U` on that channel, and the sigmoid inside SiLU is the SAME ratio on one cell. The whole loss
  is proved `Differentiable ℝ` in `θ` — so the gradient exists everywhere, backprop is the chain
  rule on it, and a small step down it strictly decreases the loss.
@engineer `nExp`/`lPlus`/`lMinus` (the cell diagonalized); `softmax_is_boost_channel`,
  `sigmoid_is_cell_ratio`, `sigmoid_is_two_point_softmax`, `every_op_is_the_cell`;
  `composing_weight_is_the_boost` / `softmax_is_the_only_composing_weight` (the exponential is
  FORCED — Cauchy, closed by continuity); the one
  instruction `mv`/`bil` and its five laws (`mv_comp`, `mv_stack`, `mv_split`, `mv_reindex`,
  `mv_add`) plus `bilinear_expand`; `qkv_is_one_contraction`, `head_score_is_relative`,
  `qk_is_one_circuit`, `ov_is_one_circuit`, `qk_gauge_freedom`, `attention_is_two_circuits`;
  `headScore`/`torusPoint`, `head_separates_offsets_iff_torus`,
  `headScore_aliases_at_common_period`, `torusSep_eq`/`torusSep_zero_iff`; `netLogits`, `netLoss`;
  `netLoss_differentiable` (the headline — the whole pipeline is differentiable in θ);
  `crossEntropy_eq_logSumExp`; `grad`, `sgdStep`, `sgd_step_decreases`, `exists_decreasing_step`,
  `batchLoss`, `trainRun`, `trainRun_nonincreasing`; and §7's inference replacement —
  `attnOut_is_bulk_ratio`, `attnOut_streams_in_O1_state` (exact, O(1) state),
  `attnOut_budget_gap` / `budget_attention_gap_is_context_free` (the error is the dropped-mass
  fraction, with no `T` in it); and the bundle `transformer_from_the_null_cell`.

## Why this file exists

The corpus proved the transformer's ops one at a time — `MLPrimitivesBridge` (contraction,
softmax, the flash chart, sort, Bellman), `AttentionSpinor` (the carry, the online-softmax
recurrence, the bounded-KV gap), `TrainingCollapse` (the ridge solve, the Newton landing).
`Closures/Applied/TransformerGeometry` is explicitly NOT a transformer: it says so in its own
header ("no softmax, no token, no sequence and no matrix occurs anywhere in this file"). Nothing
carried the whole object: embeddings → L blocks → logits → loss → gradient → update, as one
Mathlib-checked chain. This file is that chain, and it is written so the two places a transformer
could hide a non-derived primitive — `exp` and the gate — are both discharged onto the null cell
first (§1), before any layer is defined (§3).

## The spine

* **§1 — the cell gives the nonlinearity.** In the boost-idempotent (lightcone) coordinates
  `L± z = z.1 ± z.2` the null cell's multiplication is coordinatewise (`lPlus_nmul`,
  `lMinus_nmul`) and its norm is the product of the channels (`nNorm_eq_lPlus_mul_lMinus`) — the
  cell is two independent real channels. The boost `e^{jφ} = cosh φ + j sinh φ` has channels
  `L₊ = e^{φ}`, `L₋ = e^{−φ}` (`lPlus_nExp`, `lMinus_nExp`), is the one-parameter group
  (`nExp_add`), and sits on the unit hyperbola (`nNorm_nExp`). Hence **softmax is `B/U` on the
  boost's `+`-channel** (`softmax_is_boost_channel`) and **the sigmoid is that same ratio inside
  ONE cell** — `σ(x) = L₊/(L₊+L₋)` at `φ = x/2` (`sigmoid_is_cell_ratio`), equivalently the
  two-point softmax (`sigmoid_is_two_point_softmax`). The transformer's two nonlinearities are one
  object: the boost, read as a ratio.
* **§1c — and no other weight was available.** Fix the one law a weighting must obey — scores add,
  weights multiply, because that is what makes the normalized read a likelihood and not a vote —
  and the exponential is the ONLY continuous positive solution (`composing_weight_is_the_boost`,
  Cauchy's equation closed by continuity). Normalizing any such weight IS softmax at temperature
  `c = log (w 1)` (`softmax_is_the_only_composing_weight`), and the architecture already spends
  that one degree of freedom as `1/√d_head`. What is ASSUMED is the composition law; what is
  PROVED is that it has one solution.
* **§1b — the rest of the ops are the same cell.** The contraction is the cell's own
  multiply-accumulate, `L₊` of accumulated `nmul`s (`contraction_is_cell_mac`); the RMSNorm drag
  is the sum of the BEING cells' norms (`rms_drag_is_being_cell_norm`) where the softmax drag was
  a sum of NULL-cell boost channels — two drags, two cells. Bundled as `every_op_is_the_cell`:
  contraction, attention weight, gate, norm, rotor, residual, and nothing else is used below.
* **§2 — the graph.** A node is `Node P = (Fin P → ℝ) → ℝ`: a scalar as a function of the flat
  parameter vector. Every op is a combinator on nodes, each with its differentiability lemma. This
  is autograd's own data structure, typed.
* **§2b — there is ONE arithmetic instruction, and five laws.** Every parameter is consumed by the
  contraction `mv` and by nothing else, so the machine's whole licence is: composition
  (`mv_comp` — two contractions are one weight), stacking (`mv_stack` — projections of the same
  input are one instruction), split-K (`mv_split`), schedule-freedom (`mv_reindex` — the
  accumulation order is not information), and linearity (`mv_add`, `mv_smul`). `bilinear_expand`
  is the sixth, used by §3b.
* **§3b — QKVO condenses to TWO circuits.** Q, K, V read the same vector, so they are one
  contraction (`qkv_is_one_contraction`). `W_Q`/`W_K` meet only inside a score and `W_V`/`W_O`
  only inside the residual write, so each pair collapses by `mv_comp` into one `d×d` matrix of
  rank ≤ d_head: `qk_is_one_circuit` (`W_Qᵀ·R(u−t)·W_K`) and `ov_is_one_circuit` (`W_O·W_V`).
  RoPE does not block the first — it sits INSIDE the circuit at the relative offset, which is
  also the head-level step `RopeRelative` declares open in its own header
  (`head_score_is_relative`). Consequence: the four-matrix parametrization carries gauge, and any
  refactorization preserving the two circuits changes nothing (`qk_gauge_freedom`).
* **§3 — the network.** `Block` and `Net` hold index maps into `Fin P` (the weights ARE slots of
  θ), and `blockFwd` is pre-norm attention + pre-norm SwiGLU with both residuals. Attention is
  causal by construction: position `t` sums over the prefix `pre t`, never a mask of `−∞`.
* **§3c — the shape, and what it forces on the KV cache.** The residual stream is `H·dp` Being
  cells; a head's `dp` frequencies send an offset to ONE point of the `dp`-torus; the score is a
  trigonometric polynomial in the offset with exactly those frequencies. **The two cells split the
  two jobs:** the Being cell (rotor, a circle — COMPACT) carries position, the null cell (boost, a
  hyperbola — non-compact) carries weight. A head separates two offsets **iff** their torus points
  differ (`head_separates_offsets_iff_torus`) — `←` is a ceiling no weight and no training can lift,
  `→` says the positional channel is exactly `2·dp` reals per head at any context length. Hence
  `headScore_aliases_at_common_period`: rationally-related frequencies close the winding line and
  cap the distinguishable context at their common period, so the RoPE frequency set — not a
  hyperparameter, a choice made before any weight exists — fixes the KV cache's injectivity radius.
  `torusSep_eq` turns the ceiling into a computable curve: positional separation is
  `∑_p 2·(1 − cos((Δ−Δ')·ω_p))`, a function of the offset difference alone, vanishing exactly where
  the head is blind (`torusSep_zero_iff`).
* **Anchor:** the aliasing ceiling is not binding for standard RoPE. Evaluating `torusSep` on
  the frequency sets a served model actually has (`head_dim = 64`; `base = 10⁶` as Qwen2.5-0.5B's
  `config.json` gives it, and `base = 10⁴`): the minimum separation over every offset in `[1, T]` is
  attained at `Δ = 1` and is CONSTANT — `1.6308` at base 10⁶, `2.1663` at base 10⁴ — for every `T`
  from 16 to 1 048 576. The winding line never returns near a visited point, so the aliasing ceiling
  is real but never approached out to 1M tokens: **positional resolution is not what limits a
  served context.** Two further readings, both measured: the curve is NOT monotonic in `Δ` (base
  10⁶ gives 53.8% of plateau at `Δ = 128` and 41.3% at `Δ = 256`), so "further apart" does not mean
  "more distinguishable"; and `base` is a dial between local resolution and global period — 10⁴
  separates adjacent positions better (2.17 vs 1.63) while its slowest cell wraps in 47 117 tokens
  where 10⁶'s takes 4 080 185. Rig: `~/Workspace/coldkv/bench/torus.py`.
* **§4/§5 — loss and the headline.** Mean next-token cross-entropy, proved equal to the
  log-sum-exp form (`crossEntropy_eq_logSumExp`), and `netLoss_differentiable`: the entire
  pipeline is differentiable in θ at every θ, for every network, every token stream, every depth.
* **§6 — training.** `grad` is the coordinate gradient (`fderiv` against `Pi.single`),
  `fderiv_eq_sum_grad` says the derivative is the gradient's inner product (so backprop's
  reverse-mode sum IS the derivative), and `sgd_step_decreases` proves a small enough step down
  `−grad` strictly lowers the loss whenever the gradient is nonzero.
* **§6b — the loop.** `batchLoss` (the dataset enters here and nowhere else),
  `exists_decreasing_step` (off a stationary point a strictly-decreasing rate always EXISTS),
  `trainRun` (n steps of a schedule) and `trainRun_nonincreasing` (descent composes). Data →
  forward → loss → gradient → step → repeat, as one object.
* **§7 — inference.** What replaces the `T²` softmax read, with the theorems that license it.

## What is proved, what rides, and what is anchored

* **Proved (native):** everything in §1–§6 over ℝ on Mathlib — the cell's diagonalization and its
  boost; the derivation of softmax and the sigmoid from it; §1c's forcing of the exponential;
  `contraction_is_cell_mac`,
  `rms_drag_is_being_cell_norm`, `every_op_is_the_cell`; every combinator's differentiability;
  `netLoss_differentiable` for arbitrary `P, H, dp, dff, V, T` and arbitrary block lists;
  the instruction algebra of §2b and the circuit collapse of §3b, including the whole-head
  relative-score identity `RopeRelative` names as its own open "Next";
  `crossEntropy_eq_logSumExp`; `fderiv_eq_sum_grad`; `sgd_step_decreases`;
  `exists_decreasing_step`; `trainRun_nonincreasing`; `attnOut_is_bulk_ratio`,
  `attnOut_eq_softmax_weighted`, `att_maskedScore`; §3c's resolution law and its aliasing ceiling.
* **Proved (riding):** §3b's per-pair rotor identity is `RopeRelative.rope_score_is_relative`
  (the head-level sum over cells and lanes is new here); §7's streaming exactness is `AttentionSpinor.softmax_attention_is_online_spinor`
  and `attnOut_budget_gap` rides `AttentionSpinor.bounded_support_gap` — both instantiated at this
  file's own `attnOut`, on its own constructed stream and its own causal prefix, not restated; the
  null cell's algebra is `NullCell.lean`'s.
* **Anchor:** that the budget bound is a serving guarantee. `attnOut_budget_gap` bounds the
  numeric deviation by the dropped mass; that a real top-`A` gather actually keeps that mass small
  is measured, not proved (`AttentionSpinor` §11's `A=6`, the cold-store's 1M-token recall), and
  the step from a numeric bound to an unchanged emitted token is
  `SparseSelection.sparse_emits_same_token`'s margin hypothesis. Measured 2026-08-14 (`coldkv`,
  Llama-3.2-3B and Qwen2.5-14B on real chat, teacher-forced): a FIXED gather does not hold that
  mass as context grows. At exactly 2048 tokens folded, max |Δlogit| against exact attention runs
  0.440 / 1.022 / 1.691 / 2.191 at 4k / 8k / 16k / 32k — about 0.58 per doubling, linear in
  log(1/fraction), so holding the deviation fixed needs `A` proportional to `T`. No `T` appears in
  the bound and none is claimed to; the empirical content is that dropped mass at fixed `A` grows
  with `T`, which is exactly what this anchor left open. Reading "no `T` in the bound" as "`A` need
  not grow with `T`" is the inference that fails.
* **Anchor:** `Composing` — that evidence adds and weights multiply. That is the modelling
  commitment §1c rests on; nothing here derives it from the task. The theorem is conditional on it
  and says exactly one thing: the commitment admits a single solution family.
* **Anchor:** that this architecture is "a modern transformer". It is the Llama/Qwen shape —
  pre-norm, RMSNorm, RoPE, causal MHA, SwiGLU, residual — written out; nothing here reads a
  reference implementation, and no theorem says the two agree. Dropout, weight decay, bias terms,
  attention-sink and KV quantization are absent by choice; adding them changes no theorem below.
* **Anchor:** that differentiability is trainability. `netLoss_differentiable` says the gradient
  EXISTS everywhere and `sgd_step_decreases` says descent is possible; neither says descent
  converges, that the minimum is good, or that it fits in any amount of memory. Convergence rate,
  budget and the measured `param·tok/s` constant are not in this file.
* **Open:** the cost. `mv` carries no cost model. `SplitReduce` holds the emitted kernel's
  address algebra (unit stride, exactly-once traffic, two-stage split-K at the emitted bounds) and
  `mv_eq_range_sum` is only the INDEXING bridge into that coordinate — no bandwidth, occupancy or
  FLOP count crosses from here. Nor is the gauge group of §3b sized: `qk_gauge_freedom` proves
  "same circuit ⇒ same scores"; how many parameters that frees is a dimension count nobody proved.
* **Anchor:** that streaming exactness buys constant WORK. `attnOut_streams_in_O1_state` gives an
  O(1) STATE — verified numerically to 2.2e-16 against dense softmax with the keys shuffled, at 1,
  2, 3, 7 and 64 rounds (`coldkv` E10), so the partition genuinely does not matter. It still touches
  every key it folds. A decode step over a cold store is O(T) to rank plus O(A) to gather, and the
  measurement above forbids constant `A` at fixed accuracy, so constant work is not reachable by
  folding alone. Sub-linear is: rank block summaries first and descend only into survivors
  (`SparseSelection`'s tree gather), giving O(T/B) + O(candidates).
* **Open:** float. Every statement is over ℝ. The ternary/int8 basis the stack actually runs
  (`TernaryDP4A`) is a separate quantization of these same functions, and the rounding residue is
  not closed here.
-/
