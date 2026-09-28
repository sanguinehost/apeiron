# Evidence: Pairwise Coordinate Transforms in `R[x]/(x² - σ)` for 2-Bit Quantization (`Nous.Combinator.CellTransform`)

**Evidence Grade: `model`** (The algebraic determinant/isometry theorems in `CellTransform.lean` are `proved`; the isolated 2x2 weight SQNR is `tested`; the fusion and full-network impact claims are `model`.)

## Question

Can pairwise 2D coordinate transforms based on the `R[x]/(x² - σ)` trichotomy (Lorentz boosts, Givens rotations, Nilpotent shears) reduce 2-bit weight quantization error compared to orthogonal transforms (QuaRot / SpinQuant)?

Specifically:
1. Does rotating or boosting coordinate pairs $(w_{2i}, w_{2i+1})$ before 2-bit uniform quantization reduce reconstruction MSE on real pretrained model weights (`tok_embd_f32.npy` from `Qwen3.5-0.8B`)?
2. Which algebraic face achieves the highest Signal-to-Quantization-Noise Ratio (SQNR)?
3. Are these transforms strictly volume-preserving ($\det = 1$), guaranteeing zero compression loss before discretization?

---

## Method & Mathematical Formulation

### 1. The Trichotomy of 2x2 Volume-Preserving Transforms
For adjacent weight coordinates $v = (x, y)^T$, we evaluate three 2-parameter transform families in $\mathbb{R}[x]/(x^2 - \sigma)$:
- **Hyperbolic ($\sigma = 1$):** Lorentz boost $B(c, s) = \begin{pmatrix} c & s \\ s & c \end{pmatrix}$ with $c^2 - s^2 = 1$. Preserves hyperbolic norm form $x^2 - y^2$.
- **Elliptic ($\sigma = -1$):** Givens rotation $R(c, s) = \begin{pmatrix} c & -s \\ s & c \end{pmatrix}$ with $c^2 + s^2 = 1$. Preserves Euclidean norm form $x^2 + y^2$.
- **Parabolic ($\sigma = 0$):** Nilpotent shear $S(s) = \begin{pmatrix} 1 & s \\ 0 & 1 \end{pmatrix}$. Preserves first coordinate.

All three families satisfy $\det T = 1$, ensuring exact phase volume conservation.

### 2. Quantization Protocol
- 2-bit symmetric uniform quantization to 4 reconstruction levels: $\{-1.5, -0.5, 0.5, 1.5\} \times \text{scale}$.
- Transform pipeline: $W \to W' = W T^T \to \hat{W}' = \text{Quantize}(W') \to \hat{W} = \hat{W}' T^{-T}$.
- Evaluated on 2,048 rows of pretrained tied embedding weights ($D = 1024$) from `Qwen3.5-0.8B`.

---

## Empirical Probe on Pretrained Weights

Command:
`/home/socol/oc-tmp/exl3/.venv/bin/python /home/socol/oc-tmp/e3-pairwise-quant/probe.py`

### Numeric Results
| Transform | Parameter | Reconstruction MSE | SQNR (dB) | Gain vs Baseline |
|---|---|---|---|---|
| **Identity** (Raw) | - | $1.3033 \times 10^{-4}$ | 5.20 dB | 0.00 dB |
| **Hadamard** (2x2) | $H_{2\times 2}$ | $1.3027 \times 10^{-4}$ | 5.20 dB | +0.00 dB |
| **Elliptic** (Givens) | $\theta = 0.557$ rad | $1.3010 \times 10^{-4}$ | 5.20 dB | +0.00 dB |
| **Parabolic** (Shear) | $s = 0.450$ | $1.1385 \times 10^{-4}$ | 5.78 dB | **+0.58 dB** |
| **Hyperbolic** (Lorentz) | $\psi = 0.250$ | $1.1260 \times 10^{-4}$ | **5.83 dB** | **+0.63 dB** |

### Methodological Context & Caveats (Lead Review)
1. **Strawman Baseline:** The baseline evaluated in this probe was a block-diagonal 2x2 Hadamard matrix, which scored +0.00 dB and represents an inadequate baseline. Standard QuaRot and SpinQuant utilize a full $D$-dimensional Walsh-Hadamard transform ($D = 1024$ or $4096$) or learned global orthogonal rotations to disperse outliers across all coordinates simultaneously.
2. **Weight SQNR vs Output Space:** Weight SQNR alone does not indicate inference quality. Quantization error must be measured in output space: layer-output MSE, next-token flip rate, and KL divergence against unquantized models.
3. **Non-Orthogonal Fusion Obstruction:** A Lorentz boost satisfies $\det B = 1$, but it is **not orthogonal** ($B^T B \ne I$). It preserves $x^2 - y^2$, not Euclidean length $x^2 + y^2$. Consequently:
   - $\text{RMSNorm}(B x) \ne B \text{RMSNorm}(x)$.
   - A Lorentz boost **cannot be fused across RMSNorm layers** (unlike orthogonal Hadamard matrices).
   - Invertible non-orthogonal transforms can only be fused across purely linear operations where no nonlinear activation or normalization intervenes: specifically, $W_v \to \text{attn} \to W_o$ per head, the DeltaNet $v \to \text{ssm\_out}$ path, or by applying a column-pair transform with a runtime activation inverse (F1 in Round 2).

---

## Formal Proof & Verification

- **Lean Module:** [`proof/corpus/Nous/Combinator/CellTransform.lean`](file:///home/socol/Workspace/apeiron/proof/corpus/Nous/Combinator/CellTransform.lean)
- **Primary Gate:**
  `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'cd proof/corpus && LEAN_NUM_THREADS=1 lake build Nous.Combinator.CellTransform'`
  - **Verdict:** GREEN (exit code 0, 2,972 jobs built).
- **Headlines Proved:**
  - `det_lorentzBoost`: $\det B(c, s) = c^2 - s^2$.
  - `lorentzBoost_det_one`: $c^2 - s^2 = 1 \implies \det B = 1$.
  - `lorentzBoost_preserves_norm`: $(v'_0)^2 - (v'_1)^2 = x^2 - y^2$ (hyperbolic metric invariance).
  - `lorentzBoost_inv`: $B(c, s) B(c, -s) = I$.
  - `det_givensRot`, `givensRot_det_one`, `givensRot_preserves_norm`, `givensRot_inv`: Elliptic volume, Euclidean norm, and inverse properties.
  - `det_parabolicShear`, `parabolicShear_inv`: Parabolic volume preservation and shear inverse.
- **Mutation Control:**
  - Mutated file: `/home/socol/oc-tmp/e3-pairwise-quant/control_copy/CellTransform_mutant.lean`.
  - Mutation: Confounded hyperbolic determinant with elliptic determinant: `det B = c^2 + s^2`.
  - Control Execution:
    `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 60 bash -c 'LEAN_PATH=... lean /home/socol/oc-tmp/e3-pairwise-quant/control_copy/CellTransform_mutant.lean'`
  - **Verdict:** RED (exit code 1, `⊢ c^2 - s^2 = c^2 + s^2`).
  - Restoring clean source produces exit code 0.

---

## Profit Condition Assessment

1. **(a) Collisions ($R(D) < \log_2 t$):**
   Weight quantization is performed offline on static parameters.
2. **(b) Score Gaps:**
   Requires output-space verification (addressed in Round 2 F1).
3. **(c) Binding Resource:**
   If non-orthogonal transforms can be placed without violating normalization boundaries (purely linear paths), they could reduce weight bandwidth; but across normalized layers, orthogonal transforms remain required.
