# Cost correspondence: the performance model and the quantisation model from one floor

Module: `proof/corpus/Perf/CostCorrespondence.lean` (`proved`, closed cone, audited in `proof/corpus/AxiomAudit.lean`).

The performance model predicts because it has a floor on the binding resource (`Perf.Roofline.timeLB`). The
quantisation model had none: it counted bits per weight. This module derives both floors from one lemma and states
the rows where the two models agree and where they differ.

## The correspondence

| role | performance | quantisation | theorem |
|---|---|---|---|
| floor | bytes ≤ steps · capacity (counting measure) | `r^d ≤ \|C\| · ε^d` (Haar measure) | `cover_bound`, `bytes_floor`, `covering_floor` |
| what is counted | the binding field | the stiff directions of the loss | `stiff_floor` |
| free rewrites | fusion preserves denotation | a gauge preserves the function | `gauge_lift_bound`, `perm_gauge_le`, `gauge_null` |
| composition, upper | overlap ≤ sequence (ℓ∞ triangle over fields) | `√K` subadditive (ℓ² triangle) | `roofline_subadd`, `kl_sqrt_subadd`, `kl_joint_le` |
| composition, lower | overlap ≥ max | none: scores cancel | `roofline_nocancel`, `kl_cancels` |
| serial composition | chain cost adds (ℓ¹) | — | `chain_append` |
| two-block form | — | discriminant `σ = ⟪v,w⟫² − ‖v‖²‖w‖² ≤ 0` | `score_form_disc_nonpos`, `kl_pair_sub_iff` |
| partition | rounds `⌈k/p⌉`; greedy ≤ `W/S + w/s` | water level across tensors (not built) | `greedy_makespan`, `refine_gain_le` |
| carrier ↔ coder | bytes per token `N·R/8` | rate `R` | `tg_floor_ratio` |

- **One floor.** `cover_bound`: a set covered by `|I|` cells of measure `≤ c` has measure `≤ |I|·c`. With the counting
  measure on bytes it is the roofline field bound (`bytes_floor` gives `⌈|B|/cap⌉ ≤ n`). With Haar measure on a
  `d`-dimensional space it is the covering converse (`covering_floor`: `r^d ≤ |I|·ε^d`).
- **Stiff dimension, not parameter count.** `stiff_floor`: if a 1-Lipschitz map sends the admissible weights onto
  a ball of a `k`-dimensional stiff space, any `ε`-codebook has `≥ (r/ε)^k` words. `k` is the stiff dimension; the
  ambient dimension does not enter.
- **Gauge.** A finite group `G` that preserves the function and the measure lowers the codebook by at most the factor
  `|G|` (`gauge_lift_bound`). Permutations of `n` channels cost `≤ b·n` bits when `n ≤ 2^b` (`perm_gauge_le`). A
  continuous exact gauge direction `x` (`xᵀFx = 0`, `F ⪰ 0`) leaves every quadratic error unchanged
  (`gauge_null`, via `Matrix.PosSemidef.dotProduct_mulVec_zero_iff`): Fisher water-filling already gives it zero rate,
  so a gauge choice moves the coder's gap, never the floor.
- **Composition.** Both models compose by a triangle inequality; the norm differs (ℓ∞ over fields, ℓ¹ over a serial
  chain, ℓ² over score samples), and so does the sign of the summands. Demands are non-negative counts and never
  cancel (`roofline_nocancel`); layer scores are signed and do (`kl_cancels`). Joint/sum below 1 is exactly a negative
  score correlation (`kl_pair_sub_iff`).
- **Not hyperbolic.** The form `‖αv + βw‖²` of two layer scores has discriminant `≤ 0` (Cauchy–Schwarz): elliptic or
  parabolic. The W14 reading "additivity ≈ 1 is the hyperbolic face" (`anchor` in `onebit-w14.md`) is refuted by this
  theorem: no additivity ratio places a pair of layers on the hyperbolic face.

## Verdicts on measured inputs

| verdict | inputs | result | theorem |
|---|---|---|---|
| bits per stiff direction at Q4_K-type layer error `ρ = 0.0045` | `onebit-w13.md` | `> 31/8 = 3.875` bits per stiff direction (`model` reading of the worst-case covering converse); at 1.18 b/w the stiff fraction must be `≤ 0.3046` | `q4_bits_per_stiff_dof` |
| stiff fraction on Qwen2.5-0.5B `ffn_down` | `r99(WHWᵀ)` 99 / 770 / 441 of 896 on blocks 3 / 11 / 23 (`docs/plan.md` LEARN-ONEBIT W10, `/home/socol/oc-tmp/escalation/2026-10-11-dof-spectra.json`) | 0.110 below the threshold; 0.859 and 0.492 above | `stiff_fraction_vs_w10` |
| permutation gauge, Qwen2.5-0.5B | 24 layers, 4864 FFN channels, 494,032,768 weights | `≤ 1,517,568` bits, `< 0.004` b/w | `qwen05b_perm_gauge` |
| W14 joint KL within the Minkowski bound | per-block KL from `/home/socol/oc-tmp/escalation/2026-10-11-stiffness.json` | joint / `(Σ√k_l)²` = 0.055 `IQ1_S`, 0.052 `IQ1_M`, 0.060 Q4_K_M, 0.062 RTN, 0.042 `had-gptq1`; the 21 block scores compose near the orthogonal (ℓ²) sum, not the aligned (ℓ¹) one | `w14_within_minkowski` |
| partition `0bf8b4d` (64 → 256 tasks, 24 workers) | GMAC/s ffn_down 1096 → 1211–1218, attn_output 1047 → 1177–1185, ssm_out 1054 → 1197–1201 | homogeneous rounds model `⌈64/24⌉ = 3`, `⌈256/24⌉ = 11`, gain `1.0909`; measured `1.105–1.139` exceeds it (homogeneous model refuted; heterogeneous P/E cores are the `model` reading, speeds not measured); under the greedy bound `1.375` | `partition_0bf8b4d`, `refine_gain_le` |
| 27B decode floor at the target rate | `27·10⁹` weights, 1.18 b/w, 960 GB/s | 3.9825 GB, floor `< 4.15` ms per token | `qwen27b_target_floor`, `tg_floor_ratio` |

## Gates and controls

| check | command | result |
|---|---|---|
| module | `cd proof/corpus && lake build Perf.CostCorrespondence` (lane lock, 12 G cap, `LEAN_NUM_THREADS=1`) | exit 0 |
| axiom cone | `cd proof/corpus && lake build AxiomAudit` | exit 0, 9172 jobs, 24 `#audit` lines; every theorem within `{propext, Classical.choice, Quot.sound}` (`stiff_fraction_vs_w10` empty) |
| mutation controls | `lake env lean` on throwaway copies under `/home/socol/oc-tmp/costcorr-mut/`; log `/home/socol/oc-tmp/costcorr-controls.log` | 5/5 exit 1 at the mutated theorem: `cover_bound` flipped (calc step), `kl_cancels` negation removed, `score_form_disc_nonpos` sign flipped (`nlinarith` fails), `partition_0bf8b4d` order flipped, `stiff_fraction_vs_w10` block 11 claimed below threshold (`decide` refutes) |

## What this says for the onebit target

- The rate floor is set by the stiff dimension of the function, not the weight count. At Q4_K-type layer error each
  stiff direction costs more than 3.875 bits, so 1.18 b/w is reachable only where at most 30.5% of the directions are
  stiff (`model`: worst-case covering, `ρ` as the distortion, r99 as the stiff count).
- On the three measured `ffn_down` blocks, block 3 meets that (0.110) and blocks 11 and 23 do not (0.859, 0.492). The
  same split appeared in the W13 isotropic floor (Q4_K-type `ρ` at 1 b/w on block 3 only), now from a statement that
  does not assume isotropic rows.
- So post-training coding of blocks with high stiff fraction cannot reach Q4-level error at 1.18 b/w; a route to the
  target either lowers the stiff fraction of the function (training) or spends the rate unevenly across tensors (the
  partition row, water level not built).

## Not covered

- `covering_floor` is a worst-case covering bound for a ball. A source distribution and an average distortion need
  the Shannon lower bound (`anchor`, not retrieved); the KL is a quadratic form only in the small-perturbation regime.
- The stiff count `r99` is a 99%-energy rank of the layer Gram (`G = I`), not of the full Fisher `G ⊗ H`.
- The water level across tensors (the quantisation analogue of the partition row) is not built.
- No byte, cycle or nat follows from a theorem here; numbers in the verdicts are measured inputs.
