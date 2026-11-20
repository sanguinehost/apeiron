# Q2 0-shot gap: the pack attenuates the candidate margin, and the loss is the prior

Criterion **Q2** (0-shot half). Grade `measured` for every table, `model` for the bit arithmetic in
§5. Data: the twelve per-item JSONL files in
`bench/cpu/1794937300-nq8b-q2-rented-host-logs/`, reproduced by
`python3 tools/quality-gap.py --dir bench/cpu/1794937300-nq8b-q2-rented-host-logs --items bench/quality/items`.

## 1. Verdict

NQ **0.5600** against Q4_K_M **0.7433**, ratio **0.7534**, paired difference **−0.1833 ± 0.0812**
(95 % t, df 5), allowance −0.0743: refuted. 134 items correct under Q4 and wrong under NQ, 24 the
other way, 110 net of 600.

## 2. The margin is attenuated, not scaled

For each item the score is `ll` (acc) or `ll / norm_len` (acc_norm); `m` is the gold score minus the
best other. `m_nq = a + b·m_q4 + e`:

| task | n | acc NQ | acc Q4 | ratio | Q4→NQ | NQ→Q4 | mean m Q4 | sd m Q4 | mean m NQ | sd m NQ | b | sd e | Φ fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boolq | 100 | 0.630 | 0.880 | 0.716 | 29 | 4 | +5.226 | 4.129 | +1.209 | 2.442 | 0.26 | 2.198 | 0.693 |
| piqa | 100 | 0.710 | 0.770 | 0.922 | 11 | 5 | +0.113 | 0.277 | +0.050 | 0.188 | 0.58 | 0.100 | 0.623 |
| hellaswag | 100 | 0.570 | 0.740 | 0.770 | 17 | 0 | +0.117 | 0.206 | +0.019 | 0.185 | 0.78 | 0.091 | 0.556 |
| winogrande | 100 | 0.490 | 0.760 | 0.645 | 33 | 6 | +2.218 | 4.073 | +0.303 | 1.751 | 0.16 | 1.631 | 0.569 |
| arc_easy | 100 | 0.580 | 0.790 | 0.734 | 23 | 2 | +0.349 | 0.573 | +0.048 | 0.483 | 0.46 | 0.407 | 0.537 |
| arc_challenge | 100 | 0.380 | 0.520 | 0.731 | 21 | 7 | +0.005 | 0.503 | −0.172 | 0.426 | 0.20 | 0.414 | 0.343 |

Slope median **0.36**, range 0.16–0.78. The residual sd is the size of the signal itself. A positive
scale on all logits leaves `argmax` unchanged, so this is not a temperature: the NQ arm's
candidate-score *difference* has changed direction, item by item, and kept about a third of its
spread.

The last column is the accuracy predicted by the same affine-plus-Gaussian relation,
`mean_i Φ((a + b·m_q4,i)/sd e)`. It lands within 9 points of every observed task accuracy, so one
mechanism accounts for most of the six-task gap.

## 3. Where the contrast is smallest, accuracy is the prior

| task | what decides it | NQ | Q4 |
|---|---|---:|---:|
| boolq | the single token ` yes` or ` no` | 0.630 | 0.880 |
| winogrande | which of two names fits | 0.490 | 0.760 |
| piqa, long tercile (~45 tokens) | a whole sentence | 0.880 | 0.850 |
| piqa, short tercile (~7 tokens) | a short phrase | 0.520 | 0.700 |

boolq makes the mechanism explicit. The yes/no contrast is `ll(" yes") − ll(" no")`:

| arm | mean | sd | P(yes) |
|---|---:|---:|---:|
| Q4_K_M | +1.390 | 6.514 | 0.580 |
| NQ | +1.939 | 1.915 | **0.870** |

The label prior in the sample is 0.560, so the NQ arm's 0.630 is 7 points above always-yes: the
yes/no axis has been compressed to **0.29×** and shifted +0.55 nats. Winogrande is the same collapse
without a dominant label: 0.490 on two choices is chance. On piqa's long tercile, where the two
candidates are different sentences, the NQ arm still ranks the gold first more often than Q4 does.

## 4. It is not boundary noise alone

| Q4 margin, task sd units | n | NQ acc | Q4→NQ flips | NQ→Q4 flips |
|---|---:|---:|---:|---:|
| [−∞, −1.0) | 34 | 0.059 | 0 | 2 |
| [−1.0, −0.3) | 67 | 0.119 | 0 | 8 |
| [−0.3, 0.0) | 53 | 0.264 | 0 | 14 |
| [0.0, +0.3) | 92 | 0.467 | 49 | 0 |
| [+0.3, +1.0) | 172 | 0.721 | 48 | 0 |
| [+1.0, +3.0) | 172 | 0.808 | 33 | 0 |
| [+3.0, +∞) | 10 | 0.600 | 4 | 0 |

Half the losses sit where Q4's margin is under 0.3 sd, which a small perturbation explains. But 19 %
of the 1–3 sd items and 4 of 10 above 3 sd also flip: the NQ error is large in absolute terms, not
only near the boundary.

## 5. What closes it

The rate is fixed at **1.178785 b/w** and the cap is 1.18, so there is 1,044,615 B of headroom —
0.1 %, not enough for anything. The format's own duplication is the budget. The qkv `V`, `s_pre`,
`s_mid` and `salient_idx` are stored three times (`bench/cpu/1794885786-nq8b-uv-signs.md`:
91,450,016 B, 9.369 % of the 976 MB pack it measured), and a shared-section format at identical
factors is the same weights with one copy. Applied to this pack's 1,023,456,080 B, that is
**932,006,064 B = 1.073466 b/w**, freeing **0.1053 b/w** with no quality change (confirm against the
pack's section table before building; the arithmetic is `model`).

| correction | bytes | b/w | running total |
|---|---:|---:|---:|
| dedupe qkv V / s_pre / s_mid / salient_idx | −91,450,016 | −0.1053 | 1.0735 |
| per-output-channel f16 bias, all 252 linears | +3,391,488 | +0.0039 | 1.0774 |
| rank-12 f16 residual, every linear | +65,470,464 | +0.0754 | 1.1489 |
| rank-16 f16 residual, every linear | +87,293,952 | +0.1005 | **1.1740** |

The residual is the SVD of `W − Ŵ`, the top-r error directions, which is the same `R_0` that
`tools/nq_residual_bake.py` computes before its dense sign correction; the f16 rank-r form captures
the same leading directions for about 1/200 of the bytes a second ±1 layer would cost (a sign layer
is +1 b/w). Rank 16 is the most that fits under 1.18, and it leaves 4.2 MB.

The bias targets §3's +0.55-nat shift; the residual targets §2's attenuation. Neither is fitted as
yet.

**The no-new-bits alternative.** Refit the existing 1-bit factors against the next-token
distribution (a KL/logit loss) instead of the activation MSE that `nq_fit` minimises. An MSE fit is
exactly the objective that tolerates a shrunken margin, so this is the principled fix; whether a KL
fit recovers the contrast at the same rate is untested, and it is the experiment that separates "the
1-bit capacity is insufficient" from "the fit spends the capacity on the wrong directions".

## 6. The experiment, in gate order

1. **Dedupe only.** The scored values must be bit-identical to today's; this proves the rewrite is
   inert. One boolq arm, 100 items, ~13 min.
2. **Bias.** Score boolq. The pass signal is `P(yes)` moving from 0.87 toward 0.58 and the contrast
   mean from +1.94 toward +1.39.
3. **Rank-8, then rank-16 residual.** Score boolq and winogrande (26 min). The gate is the Q2 bar on
   these two tasks: boolq ≥ 0.792 and winogrande ≥ 0.684 (0.90 × the Q4 values), because they are
   where the contrast is smallest.
4. Only then the six-task screen, ~45 min on one 40-CPU box against the saved Q4 denominators.

## 7. What this does not cover

One pack, 100 items per task. The per-layer attribution is not measured: the analysis localises the
loss to the *kind* of decision, not to a layer, so the residual is proposed for every linear rather
than a chosen subset. The bit arithmetic is a size calculation, not a built pack. The KL fit is
untested. rustc and LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.
