# Two papers: the fold is linear attention, and capacity is not sequence length

Criterion: **C1/C2** and the fold line. Grade: `anchor` for everything taken from the two papers
(a published theorem or measurement about another implementation, not a measurement here);
`measured` for the decode split this note reuses (`bench/cpu/1794919269-nq8b-split.md`); arithmetic
otherwise. The result is two **refusals** and one **endorsement**: the fold's TTT read is not a new
mechanism, a covariance state is the worst carrier per byte, and the block gather is the only tier
the capacity theorem leaves standing.

- Liu, J., Elflein, S., Litany, O., Gojcic, Z., Li, R. *Test-Time Training with KV Binding Is
  Secretly Linear Attention.* arXiv:2602.21204 (v1 2026-02-24, v4 2026-05-12).
  <https://arxiv.org/abs/2602.21204>
- Wang, K. A., Shi, J., Fox, E. B. *Test-time regression: a unifying framework for designing
  sequence models with associative memory.* arXiv:2501.12352 (v1 2025-01-21, v3 2025-05-02).
  <https://arxiv.org/abs/2501.12352>

`2602.21204` is the TTT paper. `2502.21204` is *Halfspace Representations of Path Polytopes of
Trees* (Goel, Maraj, Ribot), a combinatorics paper, and is not this work.

## 2602.21204: TTT-KVB is a linear attention operator

Theorem 5.1: a TTT-KVB inner loop whose final layer is linear and bias-free, taking one gradient
step per token, is exactly a linear attention operator with effective query, key and value built
from the inner-loop features. Theorem 5.2 unrolls it over the sequence; Theorem 5.3 extends it to
momentum. Nothing in the inner loop stores a key-value map.

The empirical case is that the memorization reading makes predictions the models violate (Table 1,
LaCT-LLM perplexity / LaCT-NVS PSNR / ViTTT top-1):

| condition | LaCT-LLM | LaCT-NVS | ViTTT |
|---|---:|---:|---:|
| baseline | 16.43 | 25.94 | 79.34 |
| gradient **ascent** in the inner loop, retrained from scratch | 16.19 | 25.85 | 79.61 |
| query replaced by key at the output | 16.18 | 25.95 | 79.18 |

Better inner-loop fit is worse downstream (§4.1): increasing inference-time inner steps lowers the
inner loss and degrades the task.

The reduction ladder (§6.1, Table 2, LaCT-LLM perplexity) removes the inner loop one component at a
time and ends at plain linear attention:

| variant | change | PPL |
|---|---|---:|
| 1 | update only the last layer | **15.93** |
| 2 | remove weight normalization | 16.31 |
| 3 | multi-layer MLP → single linear layer | 16.23 |
| 4 | remove per-token learning rate | 16.12 |
| 5 | remove momentum | 15.97 |
| 6 | remove gradient orthogonalization = linear attention | 16.80 |

Variant 6 costs **+0.87 PPL** against Variant 1 and **+0.37** against the published baseline; NVS
+0.24 dB; ViTTT inside its spread. In its own table the parallel form is 2.74× the recurrent form at
Variant 2 (11.02M → 30.18M tokens/s) and 1.39× at Variant 6 (89.67M → 124.6M); the headline claim is
up to 4.0× on the attention calculation (§6.2).

Two components are not reducible (Appendix I): a dynamic kernel (updating the query/key projections
in the inner loop) and weight normalization, both of which break the associativity that a parallel
scan needs. Neither is a capacity argument.

**What this settles for the fold.** A TTT-KVB read cannot be sold as representational capacity that
linear attention lacks: at equal everything it is +0.87 PPL of inner-loop machinery, and most of
that survives the reduction. The `(a4) CARRY+TTL = (a1) CARRY` result
(`fold/docs/STATE-ARCHIVE.md:477-503`, 11/12 both, 78.6 against 158.9 tok/s, full KV in the window)
is what this theorem predicts, not an anomaly. Note the direction: fold's `SessionRidge` reads
`Bᵀ(U⁻¹x)` with a rank-1 `U⁻¹` update, which is the *exact least squares* side of 2501.12352, not
the linear-attention side. Fold is not refuted by Theorem 5.1; it is refused by the capacity bound
below.

## 2501.12352: the regression frame, and what bounds recall

Associative recall is memorization (a weighted regression) followed by retrieval (applying the
regressor to a query). Three choices define a layer: the regression weights, the regressor class,
and the test-time optimizer.

- Linear attention is least squares with the covariance dropped: `KᵀK ≈ I` (§4, Vignette 1,
  equations 11–12). It is optimal only when the keys are orthonormal. Keeping
  `U⁻¹ = (KᵀK)⁻¹` is recursive least squares, computed by the Woodbury identity per token — the
  same rank-1 update `RidgeStream` already writes.
- Gated linear attention and state-space models are weighted least squares with geometric decay;
  fast-weight and online learners are one streaming least-squares step; softmax attention is
  kernel regression with the normalizer dropped, and query–key normalization makes it a
  locally-constant nonparametric regressor.
- §6: one linear-attention layer plus one length-2 convolution solves multi-query associative
  recall with no other parameters, given well-constructed key-value pairs. "Memory capacity, not
  sequence length, limits model performance" — once the state holds every association, sequence
  length is not the difficulty.

## Refusal 1: a covariance state is the worst carrier per byte

Capacity is the number of associations per byte, and the two carriers price it oppositely. For a
kv-head width `h = 128` and one association per position:

| carrier | bytes per association |
|---|---:|
| `h × h` covariance state at f32 (`U⁻¹` or `U⁻¹S`) | 512 |
| one KV row at 4 bit | 128 |
| one KV row at 1.25 bit | 40 |

The covariance state is 4× to 12.8× worse per byte. It wins on read cost — `O(h²)` against
`O(n·h)` — which is an argument about attention compute, and at 2k attention compute is 61.5 ms of
346.8 ms per token, 17.7%, so its own ceiling is 1.22× (`bench/cpu/1794919269-nq8b-split.md`,
measured; the binding term at 2k is the projection instruction stream, 79.9%). A state that trades
capacity for read cost is therefore the wrong trade at 2k and the wrong trade at 200k, where the
capacity is what fails.

The bound is measured, not only argued. `fold/docs/STATE-ARCHIVE.md:4193-4224` scores 44 candidate
readout gates over 1,499 `c4_en` boundaries: none reaches ≤ 0 on `∑ (CE_far − CE_local)`, and the
oracle ceiling — a gate that already knows the answer — is 0.0915 bits/token. The same section
attributes it to `over_capacity_crosstalk` for `d < n`, which is 2501.12352's capacity statement in
rank form. So the framework and the measurement agree: on real text a fixed-`d` readout buys
almost nothing, whatever its per-byte price.

## Refusal 2: C1's 1 GiB at 200k is below one bit per value on this geometry

The pack's fitted-weight count pins the geometry: 6,945,767,424 = 36 layers × (4096² + 2·4096·1024 +
4096² + 2·4096·12288 + 12288·4096), i.e. 36 layers, 8 KV heads, head width 128, so 36 · 8 · 256 =
73,728 KV values per position. At 200,000 positions:

| carrier | bytes at 200k | note |
|---|---:|---|
| f32 | 58.98 GB | 0.48 tok/s |
| 4 bit | 7.37 GB | disk |
| 1.25 bit | 2.30 GB | 12.34 tok/s |
| C1's 1 GiB budget | 5,368.7 B/position | **0.58 bits/value** |

C1's pinned 1.4976 bits/value implies about 28,680 KV values per position — the 28-layer,
four-KV-head geometry, `28 · 4 · 256 = 28,672` — while Qwen3-8B has 73,728, so the same 1 GiB is
0.58 bits/value here: below the sign bit, so no uniform carrier fits. The dense read is dead on
bytes alone (58.98 GB/token is 0.48 tok/s at the measured bandwidth), and the C1 carrier does not
rescue it: read uniformly, 1.25 bits/value over 200k is 2.30 GB/token, **12.34 tok/s**, below T1's
24 at 200k. A uniform carrier fails T1 at 200k whatever its width; selection is forced by the rate
target, not only by the memory budget.

## The endorsement

The only tier that escapes the capacity bound is the one that reads rows instead of a rank-`d`
readout: the block gather in [`ColdStore`](../../src/engine/step/cold_store.rs), PLAN S5/S8. Row
capacity is the block cache, not `d`, and 2501.12352 §6 says the same thing from the other side —
construct the pairs well and buy capacity, do not grow the state. That endorses `S5`/`S8` and
refuses both a larger hot ring and a learned covariance tier as the answer to 200k.

## What neither paper covers

Neither measures a 7B or 8B GGUF, a quantised pack, the 6.6 GiB byte budget, or a CPU decode at
2.24 GB/token at 2k. Their 4.0× throughput is an FLOP-side speedup on a layer that is not apeiron's
binding term. No claim here advances T1, M1, Q2 or C2, and C2 at 200k — whether a selected-block
attention keeps the model's scores — is untouched by both.
