# Q2's baseline audit: the two quantised tensors are byte-identical, and the 72 differing norms are an AWQ fold

Criterion **Q2**. Grade `measured` for every hash and for the fold arithmetic; `model` for the claim
that the fold improves the Q4 baseline. Tool: `tools/gguf-tensors.py` (new; parses both GGUF tensor
tables and hashes each payload).

## The claim under audit

Q2's ratio is `PPL(nq8b-late-v4/model.nqp)` on `nq8b/skeleton-q6k-q4kembed.gguf` against
`PPL(Qwen3-8B-Q4_K_M.gguf)`. The pack replaces the 252 linears; the skeleton supplies the other 147
tensors. If those 147 were not the Q4 file's, the ratio would be a cross-quantisation or
cross-checkpoint comparison and Q2's "same quantisation" clause would not hold.

## What the bytes say

```sh
python3 tools/gguf-tensors.py \
  /home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf \
  /home/lucas/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf
```

| quantity | value |
|---|---|
| skeleton | 867,740,256 B, 147 tensors, `general.file_type` 1, name `B968826D9C46Dd6066D109Eabc6255188De91218` |
| Q4_K_M | 5,027,783,488 B, 399 tensors, `general.file_type` 15, name `Qwen3 8B Awq Compatible Instruct` |
| shared names | 147 (0 only in the skeleton, 252 only in the Q4 file — the linears) |
| payload identical | **75** |
| payload differs | **72** |

The 75 identical tensors are 73 `F32` (1,330 B) plus the two that carry the bytes:

| tensor | type | bytes | sha256 |
|---|---|---:|---|
| `token_embd.weight` | `Q4_K` | 350,060,544 | `01f3917527232d9d…` |
| `output.weight` | `Q6_K` | 510,504,960 | `05d7a76c29a32e06…` |

Those two are **99.99 %** of the skeleton's 867,740,256 B and they are byte-identical to the Q4
file's. The Q4 file's quantisation of the embedding and the head is the same conversion of the same
source bytes, so the head and the embedding are the same object in both arms.

The 73 identical `F32` tensors are `attn_q_norm` × 36, `attn_k_norm` × 36 and `output_norm` — the
norms that do not sit in front of a quantised linear.

The 72 differing tensors are exactly `blk.{0..35}.attn_norm.weight` and `blk.{0..35}.ffn_norm.weight`
— the two RMSNorm weights that sit in front of the q/k/v/gate/up projections. They are not a
different checkpoint: they are the same vector times a per-input-channel scale.

| tensor | exact-equal elements | ratio p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| `blk.0.attn_norm` | 1 of 4,096 | 1.2976 | 1.3539 | 1.3988 |
| `blk.0.ffn_norm` | 1 of 4,096 | 1.4460 | 1.4766 | 1.4980 |

First six elements of `blk.0.attn_norm`: skeleton `0.01111 0.01074 0.01257 0.01080 0.01147 0.01111`,
Q4 `0.01514 0.01477 0.01599 0.01477 0.01544 0.01538` — ratios 1.363, 1.375, 1.272, 1.368, 1.346,
1.384. The scales sit in a narrow band around a per-tensor constant, which is the signature of an
AWQ smoothing fold, not of two independent fits. `blk.0.attn_q_norm` and `output_norm` are identical
to the last bit (ratio 1.0000, min = max).

## Why this is sound, and what it implies

An AWQ-compatible conversion bakes the per-input-channel scales `s` into the model so that a plain
`llama.cpp` runner reproduces the AWQ model:

```text
γ_B = γ_A ⊙ s        W_Q4 = quantise(W ⊘ s)        (per input channel)
RMSNorm_B(x) = RMSNorm_A(x) ⊙ s
RMSNorm_B(x) @ W_Q4ᵀ = Σ_j RMSNorm_A(x)_j · s_j · W_ij / s_j = RMSNorm_A(x) @ Wᵀ
```

So the two arms implement the same function: the NQ arm with the original norms and a 1-bit fit of
`W`, the Q4 arm with the folded norms and a `Q4_K` quantisation of `W ⊘ s`. The Q4 file is a
`Q4_K_M` of **the same checkpoint** — its embedding, head and `output_norm` are byte-identical, which
pins the checkpoint independently of the fold.

Two consequences:

1. **The 0.079870 PPL cost of the coarser stock tensors still applies** (`1794896874-nq8b-depth.md`):
   the Q6_K head and Q4_K embedding are the Q4 file's own, so the same-head condition is exact.
2. **The baseline is the distributed artifact, and it may be the stronger one.** The fold exists to
   make the linears easier to quantise, so the Q4 arm may score better than a plain `Q4_K_M` of the
   same weights would; the direction is `model`, not measured here. If so, the 1.499225× and
   1.467112× are conservative against a plain `Q4_K_M`. The criterion names this file by sha256, so
   no adjustment is applied.

## What this does not cover

The linears: the 252 `Q4_K_M` weights are not compared with the original `F16`/`BF16` tensors
(neither is in the tree), so the fold's exactness is verified in arithmetic and by the Q4 file's
plausible PPL (7.587145 on window 1), not by a per-tensor residual. The Q4 file's own quantisation
error is not measured against the dense model. The tokenizer is compared by vocabulary agreement,
not by a hash of the merged `tokenizer.json` against the GGUF's `tokenizer.ggml.*` arrays.
