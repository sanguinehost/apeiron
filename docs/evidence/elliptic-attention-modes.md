# Evidence: Elliptic Attention Modes vs Monotonic Real Decay (`Nous.Model.EllipticModes`)

**Evidence Grade: `model`** (Hypothesis and representation mapping; not an interventional result. The Lean theorems `real_decay_strictly_positive` and `quarterTurnRotor_sq` are `proved`.)

## Question

Why did post-training linearization of softmax attention with RoPE (`linearize-qwen05b.md`) collapse to 0/20 needle recall when transferring to real decay architectures (Gated DeltaNet)?

Specifically:
1. Does the empirical mean positional lag profile $A(\Delta) = \mathbb{E}[q_m^T k_{m-\Delta}]$ of pretrained attention heads exhibit oscillatory behavior that cannot be represented by positive real decay $\alpha^\Delta$?
2. Can complex rotation-dilation modes (the elliptic face $\sigma < 0$) compactly fit this attention profile?
3. What is the formal algebraic obstruction preventing real exponential decay from modeling sign-alternating attention kernels?

---

## Method & Mathematical Formulation

### 1. The Positional Lag Profile
Under Rotary Position Embedding (RoPE, Su et al. 2021), query and key vectors at sequence positions $m$ and $m - \Delta$ undergo 2D block rotations $R(\Delta \theta_j)$. For projection matrices $W_q, W_k$ and token activation covariance $\Sigma_h = \mathbb{E}[h h^T]$, the expected attention score at positional lag $\Delta$ is:
$$A(\Delta) = \frac{1}{\sqrt{d_k}} \text{Tr}\left( W_q^T \Sigma_h W_k R(\Delta)^T \right)$$

* **say:** A of Delta equals one over the square root of d sub k times the trace of W sub q transpose times Sigma sub h times W sub k times R of Delta transpose.
* **means:** the expected attention score at relative positional lag Delta equals the scaled trace of the projected activation covariance under the relative RoPE rotation matrix.
* **vars:** $A(\Delta) \in \mathbb{R}$ — expected positional attention score at lag $\Delta \in \mathbb{N}$; $d_k \in \mathbb{N}$ — attention head dimension; $W_q, W_k \in \mathbb{R}^{d_k \times d_h}$ — query and key projection matrices; $\Sigma_h = \mathbb{E}[h h^T] \in \mathbb{R}^{d_h \times d_h}$ — token activation covariance matrix; $R(\Delta) \in \text{SO}(d_k)$ — block-diagonal Rotary Position Embedding matrix at lag $\Delta$ (Jianlin Su et al., 2021); $\text{Tr}$ — matrix trace operator (Arthur Cayley, 1858).
* **apeiron:** `none — anchor`. Definition of empirical positional lag expectation profile. The underlying elliptic mode properties are proved in `Nous.Model.EllipticModes.real_decay_strictly_positive` and `Nous.Model.EllipticModes.quarterTurnRotor_sq`.
* **physics:** spatial correlation function of a rotating spinor field in stationary equilibrium.

### 2. Hyperbolic vs Elliptic Dynamics
- **Hyperbolic ($\sigma > 0$):** Decay is parameterized by real scalars $\alpha \in (0, 1)$, yielding kernel terms $\alpha^\Delta$. Over an ordered field, if $\alpha > 0$, then $\alpha^\Delta > 0$ for all $\Delta \in \mathbb{N}$. Monotonic decay preserves sign strictly and cannot cross zero or oscillate.
- **Elliptic ($\sigma < 0$):** Dynamics are parameterized by 2D rotation-dilation blocks $z = (\alpha \cos \theta, \alpha \sin \theta) \in \mathbb{R}[x]/(x^2 - \sigma)$ with $\sigma = -1$. Repeated action $z^\Delta = \alpha^\Delta (\cos(\Delta \theta), \sin(\Delta \theta))$ naturally generates sign alternations and oscillatory wave packets.

---

## Empirical Probe on Pretrained Weights

We probed layer 3, head 0 of `Qwen3.5-0.8B` ($d_k = 256$, rotary dimension 64, $\theta_{\text{base}} = 10^7$) evaluated against the activation covariance $\Sigma_h$ of 832 consecutive prompt tokens (`/home/socol/oc-tmp/lmhead-t0/h_states.npy`).

Command:
`/home/socol/oc-tmp/exl3/.venv/bin/python /home/socol/oc-tmp/e2-elliptic-modes/probe.py`

### Numeric Results
- **Profile Range:** Peak $A(\Delta) = 1.977$, minimum $A(\Delta) = 1.751$ (mean range 1.75–1.98).
- **Oscillatory Crossings:** **12 zero-crossings** around the mean across $\Delta \in [0, 128)$.
- **Hankel SVD Energy Spectrum:**
  - Rank 1 mode: **99.966%** variance explained
  - Rank 2 modes: **99.974%** variance explained
  - Rank 4 modes: **99.984%** variance explained
  - Rank 8 modes: **99.996%** variance explained

### Methodological Context & Caveats (Lead Review)
1. **Tautological Mean Fit:** RoPE makes $A(\Delta)$ a sum of cosines across 32 frequency bands by construction. Fitting this mean curve with a low-rank complex Hankel decomposition is mathematically expected from Fourier theory, not an empirical discovery about learned attention features.
2. **Mean Bias vs Content Addressing:** In actual associative memory retrieval (such as needle-in-a-haystack tasks), attention is driven by **content addressing** (the specific query-key dot product $q_t^T k_s$ for distinct semantic tokens), not the mean positional bias $A(\Delta)$.
3. **Causal Attribution:** Stating that the lack of elliptic modes "explains the 0/20 needle collapse" in `linearize-qwen05b.md` is an untested causal hypothesis. The author of `linearize-qwen05b.md` attributed the failure primarily to a tiny training budget (10M distillation tokens) rather than an expressive capacity ceiling. A rigorous test requires interventional replacement in output space (addressed in Round 2 F2).

---

## Formal Proof & Verification

- **Lean Module:** [`proof/corpus/Nous/Model/EllipticModes.lean`](file:///home/socol/Workspace/apeiron/proof/corpus/Nous/Model/EllipticModes.lean)
- **Primary Gate:**
  `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'cd proof/corpus && LEAN_NUM_THREADS=1 lake build Nous.Model.EllipticModes'`
  - **Verdict:** GREEN (exit code 0, 2,972 jobs built).
- **Headlines Proved:**
  - `real_decay_strictly_positive`: $\forall \alpha > 0, \forall n \in \mathbb{N}, \alpha^n > 0$. Real exponential decay is positive-definite and monotonic.
  - `quarterTurnRotor_sq`: $J^2 = (-1, 0)$ at $\sigma = -1$, verifying quarter-turn negation.
  - `elliptic_sign_alternation_witness`: $(J^0).1 = 1 > 0$ and $(J^2).1 = -1 < 0$, establishing sign-reversing zero crossings.
  - `rotorMul_norm` & `rotorPow_norm`: Multiplicativity of elliptic norm form $N(z^n) = (N(z))^n$.
  - `quarterTurnRotor_unitary`: Non-vacuous unit norm witness $N(J) = 1$.
- **Mutation Control:**
  - Mutated file: `/home/socol/oc-tmp/e2-elliptic-modes/control_copy/EllipticModes_mutant.lean`.
  - Mutation: Claimed $J^2 = (1, 0)$ instead of $(-1, 0)$ (asserting sign invariance for imaginary unit).
  - Control Execution:
    `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 60 bash -c 'LEAN_PATH=... lean /home/socol/oc-tmp/e2-elliptic-modes/control_copy/EllipticModes_mutant.lean'`
  - **Verdict:** RED (exit code 1, `⊢ -1 = 1` contradiction).
  - Restoring clean source produces exit code 0.

---

## Profit Condition Assessment

1. **(a) Collisions ($R(D) < \log_2 t$):**
   Continuous angular frequencies $\theta_j$ do not collide; exact hash tables do not apply.
2. **(b) Score Gaps:**
   Hypothesis only: monotonic decay cannot model sign changes, but whether this closes the needle recall gap requires interventional testing (F2).
3. **(c) Binding Resource:**
   If valid, replacing the KV cache with $O(1)$ state per head reduces memory bandwidth during decode.
