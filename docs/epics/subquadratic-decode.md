# Subquadratic decode: O(1) and O(log t) state per token

## Intent

Per token, the engine costs Θ(active parameters) for the weights and Θ(t) for attention over the KV
cache. KV reads overtake the whole weight stream at about 1,000 positions on Bonsai and 8,250 on Qwen
(`docs/evidence/pipeline-inventory.md`). This epic formalises the published mechanisms that bound the
per-token state: O(1) for Gated DeltaNet and Mamba-3, O(log t) for log-linear attention. It then runs them
on the exact integer path, on the CPU and in one Vulkan/SPIR-V kernel.

The mechanisms:

| mechanism | per-token state | source (retrieved in `docs/evidence/subquadratic-sources.md`) |
|---|---|---|
| Gated DeltaNet | O(1): one `d_v × d_k` matrix per head, `S_t = α_t S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` | Yang, Kautz, Hatamizadeh, arXiv 2412.06464; DeltaNet chunking, arXiv 2406.06484 |
| Mamba-3 | O(1): complex (rotating) SSM state | Mamba-3 paper and `state-spaces/mamba` |
| log-linear attention | O(log t): one state per Fenwick bucket | Guo et al., arXiv 2506.04761 |
| FlashAttention-4 | O(t) exact softmax; lazy rescaling | `Dao-AILab/flash-attention` |

- Each mechanism's algorithm is stated in Lean (`proof/corpus/Nous/Seq/`). The theorem that licenses a fast kernel
  comes first: the recurrent form equals the chunked or parallel form.
- The Lean statements are read from the retrieved papers and reference code, never from memory.
- Not a goal: supporting the upstream repositories' dozens of GPU architectures. We need the CPU path and one
  SPIR-V kernel.
- Models: existing models converted to these mechanisms, or released models that already use them.
  Conversion is measured by flip rate and bits/token (`docs/evidence/decision-cell-census.md` method).

## Spec (wave 1, Lean, statements over a commutative ring or ℝ)

| module | headline |
|---|---|
| `Nous/Seq/Fenwick.lean` | the Fenwick buckets of `[0, t)` are disjoint and cover it; their count is `popcount t ≤ log₂ t + 1`; the step `t → t+1` merges only the carried levels, with total merges over T steps ≤ T; the log-linear output as a sum over bucket states equals the masked quadratic form `Σ_s λ_t^{ℓ(t,s)} (q_t·k_s) v_s` |
| `Nous/Seq/LazySoftmax.lean` | the softmax output is independent of the reference max, so rescaling may be deferred while the running max stays within τ of the reference; every accumulated term is then ≤ e^τ; merges with lazy references stay exact (extends `ListHom.os_tree`) |
| `Nous/Seq/GatedDelta.lean` | the gated delta step is an affine map `S ↦ S·M_t + N_t`; recurrent = any chunked composition (monoid, as `FaceRecurrence.recur_eq_scan`); unrolled closed form; the WY/UT form of `Π (I − β_r k_r k_rᵀ)` used by chunk kernels |


## Certified sparse softmax: exact attention without reading every key

Western labs keep softmax attention because linear and recurrent states lose long-range recall and reasoning.
Our integer softmax admits exactness and sublinear reads at once:

- `exp_at` (`src/engine/attn.rs:127`) returns exactly `0` when `score − max < −16·ONE`
  (`EngineAttnRefinement.exp_at_window`). A key more than 16 below the maximum contributes nothing to the
  denominator or the value mix, so skipping it leaves the attention output bit-identical.
- Keys are grouped in blocks, and each block carries a summary: per-channel min/max (the box bound
  `Σ_i max(q_i·lo_i, q_i·hi_i)`) or a centre and radius. The summary gives an upper bound on `q·k`
  for every key in the block. A block whose bound is below `M − 16`, where `M` is the best score found so far, is
  skipped, since `M ≤` the true maximum. This is the `PrunedArgmax` argument applied to softmax.
- Blocks at Fenwick levels (`Nous/Seq/Fenwick.lean`) give O(log t) summary checks per query before
  descending into live blocks. Cost per token ≈ summaries checked + live keys, not `t`.
- Reasoning is unaffected by construction: the model and its outputs are unchanged. The only question is
  cost, which depends on how sparse attention is and how tight the bounds are. The Quest page bound
  (Tang et al. 2024; anchor, not retrieved) is the same box bound used for approximate top-k selection. Here
  it is used for exact pruning against the underflow window.

| task | file (owner) | gate |
|---|---|---|
| SPARSE-T0 | `docs/evidence/sparse-softmax-probe.md` | measured live-key fraction and prune rate vs `t`, per layer/head |
| SPARSE-T1 | `Nous/Seq/CertifiedSoftmax.lean` | `lake build Nous.Seq.CertifiedSoftmax`: pruning keys below `M − W` leaves the windowed softmax unchanged; box and ball bounds are sound |
| SPARSE-T2 | Rust, after T0/T1 | bit-identical attention against `attn_head` on random and real KV states |

## Plan

| wave | task | file (owner) | gate |
|---|---|---|---|
| 0 | SQ-REF | `docs/evidence/subquadratic-sources.md`; clones under `/home/socol/Workspace/ref/` | path/sha checker |
| 1 | SQ-T1 | `Nous/Seq/Fenwick.lean` | `lake build Nous.Seq.Fenwick` |
| 1 | SQ-T2 | `Nous/Seq/LazySoftmax.lean` | `lake build Nous.Seq.LazySoftmax` |
| 1 | SQ-T3 | `Nous/Seq/GatedDelta.lean` | `lake build Nous.Seq.GatedDelta` |
| 2 | SQ-T4 | `Nous/Seq/Mamba3.lean` (after SQ-REF) | `lake build Nous.Seq.Mamba3` |
| 3 | CPU kernels on the exact path, then one SPIR-V kernel | after wave 2 | equality with the recurrent reference |

Worker: `gemini-flash`. The lead registers headlines in `AxiomAudit.lean`.
