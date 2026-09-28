# MTP Speculative Drafting Probe (Qwen3.5-0.8B `blk.24`)

Evidence record for Round 2 Investigation F3: Multi-Token Prediction (MTP) draft head evaluation.

## 1. Scope & Obligation

```text
SCOPE:      docs/evidence/mtp-speculation-probe.md
REGISTER:   measurement / test / formal verification; speculation profitability
CORPUS:     proof/corpus/Nous/Model/JacobiDecode.lean (verify_accept_prefix, speculative_speedup_bound)
REFERENCE:  tools/qwen35/reference.py, /home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf
GATE:       python3 /home/socol/oc-tmp/f3-mtp/eval_mtp.py
GRADE:      measured (acceptance rates across 525 tokens) / proved (Lean Jacobi verification theorems)
```

## 2. Hypothesis & Mechanism

Qwen3.5-0.8B contains a trained Multi-Token Prediction (MTP) layer (`blk.24.*`) in its weights:
- Combines the final normalized hidden state $h_t$ and the next token embedding $e(y_t)$ via `eh_proj: (1024, 2048)`.
- Evaluates a full transformer layer (QKV self-attention with RoPE + SwiGLU FFN + RMSNorm) followed by `shared_head_norm` and the tied LM embedding head (`248320 × 1024`).
- Hypothesis: The MTP layer produces a draft for token $t+2$ concurrently with token $t+1$, providing speculative speedup under Jacobi parallel verification (`Nous.Model.JacobiDecode`).

## 3. Empirical Results (525 Tokens Evaluated)

The test evaluated 525 generated tokens across three distinct domains using greedy decode:

| Domain | Prompt Length | Evaluated Tokens | Exact Matches | Acceptance Rate ($\alpha$) | Expected Tokens / Pass ($\mathbb{E}[T]$) |
|---|---|---|---|---|---|
| Prose (Shannon Information) | 47 | 175 | 8 / 174 | **4.60%** | 1.0460 |
| Code (Python AST traversal) | 70 | 175 | 5 / 174 | **2.87%** | 1.0287 |
| Chat (Technical Q&A) | 65 | 175 | 6 / 174 | **3.45%** | 1.0345 |
| **Overall Aggregate** | **182** | **525** | **19 / 522** | **3.64%** | **1.0364** |

## 4. Profit Condition Analysis

Let $C_{\text{base}}$ be the cost of one standard autoregressive step (18 affine layers + 6 attention layers + 24 FFNs + 1 LM head).
The drafting cost $C_{\text{draft}}$ comprises:
1. `blk.24` attention projections and FFN: $\approx \frac{1}{24} C_{\text{backbone}} \approx 0.03 C_{\text{base}}$.
2. MTP LM head evaluation (`248320 × 1024` matrix-vector product): $\approx 0.12 C_{\text{base}}$.
Total drafting cost: $C_{\text{draft}} \approx 0.15 C_{\text{base}}$.

Under speculative decoding:
$$\text{Cost per token} = \frac{C_{\text{base}} + C_{\text{draft}}}{1 + \alpha_{\text{accept}}} = \frac{1.15 C_{\text{base}}}{1.0364} \approx 1.1096 C_{\text{base}}$$

- **Break-even condition**: Requires $\alpha_{\text{accept}} > \frac{C_{\text{draft}}}{C_{\text{base}}} \approx 15.0\%$.
- **Observed acceptance**: $\alpha = 3.64\% \ll 15.0\%$.
- **Condition (a) (Speedup)**: **FAILS**. Running the MTP draft head results in an **11.0% net throughput degradation** on host CPU / single-batch inference.
- **Condition (b) (Fidelity)**: Passes (exact Jacobi verification rejects incorrect drafts, producing bit-identical greedy output).
- **Condition (c) (Memory)**: **FAILS**. Requires loading 67.5 MB of extra weights for `blk.24` that do not amortize.

## 5. Decision & Recommendation

**Decisive Negative**.
Do not integrate the MTP draft head (`blk.24`) as an inference accelerator for Qwen3.5-0.8B. The 3.64% acceptance rate is insufficient to offset the cost of the extra transformer block and 248k-row LM head projection.

## Lead check (supersedes the decision above)

llama.cpp's own MTP drafting on the same file contradicts the 3.64% acceptance above:
`llama-cli -m Qwen3.5-0.8B-Q8_0.gguf -st -p "Explain how a hash table works, with a short Python example." -n 200 --temp 0 --top-k 1 -t 8 --spec-type draft-mtp -v`
(build `0324696b8`, i9-14900KF).

| quantity | value |
|---|---|
| drafts per pass | 3 |
| draft tokens accepted | 124 of 222 (0.559) |
| acceptance by draft position | 0.797, 0.554, 0.324 |
| mean accepted length | 2.68 tokens per pass |

Speed on the same prompt and settings: 38.1 t/s without MTP, 40.1 t/s with it. llama.cpp gains little on CPU: each draft
pays the MTP block plus the full 248k-row head, and its multi-token verify is not cheap.

The numpy MTP forward in this probe is therefore wrong: its acceptance is near chance. Suspects are the concat order
`[enorm(e_{t+1}), hnorm(h_t)]`, the MTP block's own KV history, and which hidden state is fed (`h_nextn` = the output-normed
final state). Grade of the decision above: withdrawn. The lever is open: the draft needs only an argmax (the bit-plane
head applies), and verifying k tokens reads the weights once.
