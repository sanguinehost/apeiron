# Fine-grained MoE router in Q32.32

- **Owner:** lead (Claude track)
- **Source revision:** `74c037b`; working tree dirty
- **Status:** W1 complete (T1 tested: `src/engine/moe.rs`, 8 tests; T2 proved: `Nous/Model/TopKRouter.lean` — `topkSel_spec`, `topkSel_perm_invariant`, `softmax_topk_renorm(_max)`, `topk_monotone_softmax`, registered). W2 extraction done at the lead barrier: `word_route_topk`, `word_gate_weights` extracted; W3-T1 proved (`Numeric/MoeRouter.lean`: `route_topk_spec` — extracted `route_topk` = `topkSel` exactly, same order; `best_unchosen_spec`; refusals; registered; control: higher-index tie-break → red); W3-T2 proved (`Nous/Model/MoeStage.lean`: `moe_not_continuous`, `moe_lipschitz_on_stable`, `margin_stable`).
- **Epic ID:** `APEIRON-MOE`
- **Parent:** Track A of [`frontier-consumer`](frontier-consumer.md)

## Intent

### Problem

The engine's FFN is dense only. The engine's module documentation says so: "MoE/routed experts … open"
(`src/engine/mod.rs:176`). Fine-grained MoE is how a model keeps a large total parameter count while
reading few parameters per token, and per-token bytes are what bound consumer-device decode.

Two references, both retrieved and read:
- DeepSeekMoE, which introduced fine-grained and shared experts (Dai et al., arXiv `2401.06066`,
  verified);
- Qwen3-MoE's router in `transformers` 5.17.0 (`models/qwen3_moe/modeling_qwen3_moe.py:248-268`, in
  `/home/socol/oc-tmp/lm-eval/venv`): `softmax(router_logits)`, then `topk`, then optionally
  renormalize over the selected (`norm_topk_prob`).

### Improvement over the reference

With renormalization, the gate weights are **exactly** `softmax` over the k selected logits: the
common denominator cancels. Softmax is strictly monotone, so the top-k set is the same on logits and
on probabilities. The fixed-point router therefore:
- selects exactly on the integer logits, with no `exp` and no rounding in the selection;
- evaluates k exponentials instead of E.

Both facts are theorems (W3). Without renormalization the weights are the full-softmax values at the
selected indices, and the router keeps that path as a flag.

### Invariants

1. Selection is exact and deterministic: the k largest logits, ties broken by the lower expert index.
   It is independent of the order in which experts are stored.
2. The gate weights are nonnegative and, with renormalization, sum to one within a proved rounding
   bound.
3. The routed output is `Σ_{e ∈ top-k} w_e · FFN_e(x)` plus the shared expert when present, with
   every term through the proved dense FFN path.
4. Refusal, never saturation: k = 0, k > E, a shape mismatch or an arithmetic overflow returns `Err`.

### Non-goals

Load-balancing losses and training-time routing; capacity-factor token dropping, which does not occur
at inference; expert parallelism across devices.

## Spec

| item | value |
|---|---|
| operations | `route_topk(logits: &[i64], k) -> Result<Vec<u32>>`; `gate_weights(logits, sel, renorm) -> Result<Vec<i64>>`; `moe_ffn_fixed(router, experts, shared, k, renorm, x) -> Result<Vec<i64>>` in `src/engine/moe.rs` |
| selection | stable: sort by (logit desc, index asc), take k |
| weights | renorm: `softmax` over the selected via the existing max-shifted fixed `exp` (as in `attn_head`); no renorm: full softmax at the selected indices |
| combination | checked Q32.32 multiply-accumulate of the expert outputs |
| failure | `Err` on k = 0, k > E, a length mismatch or overflow |

Contracts (grade target): selection = top-k set (proved); order independence (proved);
renormalized weights = softmax over the selected (proved, real identity plus the fixed-point bound);
Qwen3-MoE parity on the cached `optimum-internal-testing/tiny-random-qwen3_moe` and `OLMoE-1B-7B`
router tensors (tested).

## Plan

| wave | task | files | gate | control |
|---|---|---|---|---|
| W1 | T1 Rust router, weights and combination, with tests against a float reference | `src/engine/moe.rs` (new), `src/engine/mod.rs` (one `mod` line) | `cargo test … moe` | tie-break by higher index → the stable-selection test red |
| W1 | T2 pure Lean: top-k specification and the renormalized-softmax identity | `proof/corpus/Nous/Model/TopKRouter.lean` (new) | `lake build Nous.Model.TopKRouter` | drop the tie-break → order-independence red |
| W2 | T1 extraction of `route_topk`/`gate_weights` (lead lane) | `proof/extract/lib.rs`, `proof/regen.sh` | `regen.sh --check` | — |
| W3 | T1 refinement: extracted `route_topk` = the Lean top-k spec | `Numeric/MoeRouter.lean` | module gate | as W1-T2 |
| W3 | T2 `moe_ffn_fixed` as a block `Stage` (expansion ≤ Σ selected weights · FFN expansion) | `Nous/Model/MoeStage.lean` | module gate | halve the constant |
| W4 | loader: `qwen3moe` / `olmoe` expert tensors; model-level parity against the HF reference | `src/engine/loader.rs`, `step.rs` | `cargo test`; lm-eval parity (Track E) | skip routing → parity red |

## Extraction note (W2)

The pinned Aeneas (`nightly-2026.09.09-505b6ca`) failed with "Could not compute a loop fixed point in
2 iterations" on two shapes of the selection scan:
- the scan nested inside the selection loop;
- the scan hoisted into its own function, but carrying the running maximum as an index and reading
  `logits[best]` inside the loop.

It accepts the shape that carries the running maximum **value** (`best_val`) beside the index. That
is the shape `attn_head_loop0` already uses. Rule for new kernels: carry values, not indices you
re-read, across loop iterations, and keep one loop per function.
