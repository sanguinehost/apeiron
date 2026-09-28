# Evidence: Dual-Number Forward Certificates for Activation Quantization (`Nous.Model.DualCertificate`)

## Question

Can forward error intervals in the dual numbers $\mathbb{R}[\epsilon]/(\epsilon^2 = 0)$ (the parabolic face $\sigma = 0$) compute exact Lipschitz certificates that guarantee greedy argmax preservation under int8 activation quantization?

Specifically:
1. Does the dual slack $S_{\text{dual}} = \|\Delta h\|_\infty \|W_1 - W_2\|_1$ fall strictly below the top-2 logit score gap $g = z_1 - z_2$ on real prompt tokens from `Qwen3.5-0.8B`?
2. How does the dual-number forward certificate compare to geometric $L_2$ Cauchy-Schwarz bounds and actual observed logit perturbations?
3. Does this mechanism satisfy Profit Condition (b)?

---

## Method & Mathematical Formulation

### 1. Dual Number Interval Propagation
Let activation vector $h \in \mathbb{R}^D$ be quantized to $\hat{h} = h + \delta$ with coordinate bound $|\delta_i| \le \Delta / 2$.
In dual numbers $\mathbb{R}[\epsilon]/(\epsilon^2 = 0)$, the perturbed activation is represented as $h + \Delta \epsilon$.
For candidate token row $W_k$, the linear readout satisfies:
$$\hat{z}_k = W_k^T \hat{h} = z_k + (W_k^T \delta) \epsilon$$
The logit difference between top candidates 1 and 2 is:
$$\hat{z}_1 - \hat{z}_2 = (z_1 - z_2) + (W_1 - W_2)^T \delta$$
The dual-number forward certificate computes the exact coordinate-wise supremum:
$$S_{\text{dual}} = \sup_{|\delta_i| \le \Delta / 2} |(W_1 - W_2)^T \delta| = \frac{\Delta}{2} \|W_1 - W_2\|_1$$

### 2. The Certificate Condition
If $S_{\text{dual}} < g$, then:
$$\hat{z}_1 - \hat{z}_2 \ge g - S_{\text{dual}} > 0$$
which formally certifies that activation quantization cannot alter the greedy argmax choice.

---

## Empirical Probe on Real Activations & Head Weights

We evaluated 100 prompt steps of `Qwen3.5-0.8B` ($D = 1024$) using real hidden states (`h_states.npy`) and output head weights (`tok_embd_f32.npy`), simulating symmetric int8 activation quantization ($\Delta = \max(|h|) / 127$).

Command:
`/home/socol/oc-tmp/exl3/.venv/bin/python /home/socol/oc-tmp/e4-dual-certificate/probe.py`

### Numeric Results
| Metric | Value |
|---|---|
| **Median Score Gap ($g$)** | **0.985 logits** |
| **Mean Score Gap ($g$)** | 1.741 logits |
| **Median Actual Perturbation ($|\Delta z|$)** | **0.0294 logits** |
| **Max Actual Perturbation** | 0.1605 logits |
| **Median Geometric Slack ($L_2$ bound)** | 1.676 logits |
| **Median Dual Slack ($L_1$ coordinate bound)** | **2.310 logits** |
| **Geometric Certified Ratio ($S_{\text{geom}} < g$)** | 32.0% |
| **Dual Certified Ratio ($S_{\text{dual}} < g$)** | **24.0%** |

### Key Diagnostic Finding: Profit Condition (b) Negative Control
1. **Physical Reality vs Worst-Case Relaxation:** The actual perturbation caused by int8 activation quantization is small: median $0.0294$ logits, with a maximum of $0.1605$ logits across all tokens. Because the score gap is $0.985$ logits ($33\times$ larger than the median perturbation), int8 quantization **never** flips the greedy token in practice.
2. **Failure of Dual Intervals:** The dual-number coordinate bound assumes worst-case adversarial alignment between weight signs and quantization errors. Over $D = 1024$ dimensions, this worst-case accumulation inflates the bound by $\approx \sqrt{D} \approx 32\times$, yielding a median slack of $2.310$ logits. Because $2.310 > 0.985$, the certificate refuses on **76% of tokens**.
3. **Verdict:** Dual-number forward interval bounds **fail Profit Condition (b)**:
   $$\text{"Bounds pay only if slack is strictly below the score gap."}$$
   Worst-case interval bounds cannot be used as universal runtime certificates for high-dimensional activation quantization. Format-derived bit-plane pruning (`lmhead-prune-probe.md`) succeeds where dual intervals fail because bit-plane pruning operates on exact discrete bit patterns rather than conservative continuous interval relaxations.

---

## Formal Proof & Verification

- **Lean Module:** [`proof/corpus/Nous/Model/DualCertificate.lean`](file:///home/socol/Workspace/apeiron/proof/corpus/Nous/Model/DualCertificate.lean)
- **Primary Gate:**
  `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'cd proof/corpus && LEAN_NUM_THREADS=1 lake build Nous.Model.DualCertificate'`
  - **Verdict:** GREEN (exit code 0, 2,972 jobs built).
- **Headlines Proved:**
  - `dual_slack_bounds_actual`: $|w_0 \delta_0 + w_1 \delta_1| \le (|w_0| + |w_1|) \Delta$.
  - `dual_cert_preserves_argmax`: $S_{\text{dual}} < g \implies 0 < g - \Delta z$, certifying argmax invariance.
  - `witness_certificate_success`: Positive witness where $S = 0.5 < g = 1.0$.
  - `witness_slack_reversal_exists`: Negative witness where slack $2.0 > g = 1.0$ admits sign reversal.
- **Mutation Control:**
  - Mutated file: `/home/socol/oc-tmp/e4-dual-certificate/control_copy/DualCertificate_mutant.lean`.
  - Mutation: Inverted certified inequality to assert sign reversal `g - actual < 0`.
  - Control Execution:
    `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 60 bash -c 'LEAN_PATH=... lean /home/socol/oc-tmp/e4-dual-certificate/control_copy/DualCertificate_mutant.lean'`
  - **Verdict:** RED (exit code 1, type mismatch error).
  - Restoring clean source produces exit code 0.

---

## Profit Condition Assessment

1. **(a) Collisions ($R(D) < \log_2 t$):**
   Not applicable (dynamic coordinate perturbation bounds).
2. **(b) Score Gaps:**
   **FAILED.** Dual slack (2.310 logits) exceeds median score gap (0.985 logits), certifying only 24% of steps despite 100% empirical argmax preservation.
3. **(c) Binding Resource:**
   Evaluating coordinate bounds across vocabulary rows requires reading weight magnitudes $\|W_k\|_1$, consuming more memory bandwidth than computing the int8 dot products directly.
