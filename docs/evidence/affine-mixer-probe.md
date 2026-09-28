# Evidence: Cell-Valued Affine Mixer (`Nous.Seq.AffineMixer`)

**Evidence Grade: `model`** (Lean ring monoid theorems in `AffineMixer.lean` are `proved` over an exact ring; float chunk equality and norm trajectories are `tested`; "64× SRAM traffic" and "30B in 8.6 GB at > 300 tok/s" are projections / `model`.)

## Question

Can the recurrent sequence update rules of Gated DeltaNet, Linear Attention, and Unitary Phase Attention (RetNet / Complex SSM) be unified into a single cell-valued affine mixer monoid over `R[x]/(x² - σ)`?

Specifically:
1. Does the composition of general affine steps `(S * M + N)` form an associative monoid over any commutative ring `R`?
2. Does chunked parallel scan match sequential autoregressive scan under exact unrounded ring arithmetic?
3. How do the three faces (Hyperbolic real decay, Parabolic un-decayed accumulation, Elliptic damped rotation) behave on real model activations (832 prompt hidden states from `Qwen3.5-0.8B`)?

---

## Method & Mathematical Formulation

### 1. Unified Recurrence
For key dimension $d_k = 64$ and value dimension $d_v = 64$, state matrix $S_t \in \mathbb{R}^{d_v \times d_k}$ updates according to:
$$S_t = S_{t-1} M_t + N_t$$
where:
- $M_t = A_t \left(I - \beta_t (k_t \otimes k_t)\right)$
- $N_t = \beta_t (v_t \otimes k_t)$

The transition matrix $A_t \in \mathbb{R}^{d_k \times d_k}$ instantiates the $\mathbb{R}[x]/(x^2 - \sigma)$ trichotomy:
- **Hyperbolic ($\sigma > 0$):** Real diagonal decay $A_t = \alpha_t I$ ($\alpha_t \in (0, 1)$), corresponding to Gated DeltaNet / GLA.
- **Parabolic ($\sigma = 0$):** Undecayed accumulation $A_t = I$ or nilpotent Jordan blocks $I + \epsilon J$, corresponding to Linear Attention with Delta rule.
- **Elliptic ($\sigma < 0$):** Unitary rotation-dilation blocks $A_t = \text{diag}(\alpha_t R(\theta_m))$, corresponding to RetNet, Mamba-3, and complex-valued SSMs.

### 2. Affine Step Monoid
An affine transition is represented by the pair $p = \langle M, N \rangle$.
The composition law:
$$\text{affComp}(p, q) = \langle q.M \cdot p.M, \; q.N \cdot p.M + p.N \rangle$$
satisfies:
$$\text{affApply}(\text{affComp}(p, q), S) = \text{affApply}(p, \text{affApply}(q, S))$$
and is strictly associative over any commutative ring $R$:
$$\text{affComp}(\text{affComp}(a, b), c) = \text{affComp}(a, \text{affComp}(b, c))$$

This associativity establishes that multi-step chunk composition is algebraically identical to sequential unrolled recurrence over exact unrounded arithmetic:
$$\text{recurrent\_eq\_chunked}: \text{scanSeq}(l_1 ++ l_2, S_0) = \text{scanSeq}(l_2, \text{scanSeq}(l_1, S_0))$$

---

## Empirical Probe on Real Hidden States

We evaluated the three faces on 832 consecutive prompt hidden states from `Qwen3.5-0.8B` (`/home/socol/oc-tmp/lmhead-t0/h_states.npy`), with $d_k = 64, d_v = 64$, sequence length $T = 832$, and chunk size $C = 64$.

Command:
`/home/socol/oc-tmp/exl3/.venv/bin/python /home/socol/oc-tmp/e1-affine-mixer/probe.py`

### Numeric Results
| Regime | Transition Matrix $A_t$ | Final Frobenius Norm $\|S_T\|_F$ | Max Norm $\max_t \|S_t\|_F$ | Retrieval Cosine (lag 750) | Chunk Equivalence Error |
|---|---|---|---|---|---|
| **Hyperbolic** | $\alpha = 0.995 \cdot I$ | 97.77 | 103.52 | -0.1898 | $6.02 \times 10^{-7}$ |
| **Parabolic** | $\alpha = 1.000 \cdot I$ | 180.66 | 183.28 | -0.1074 | $6.63 \times 10^{-7}$ |
| **Elliptic** | $\alpha R(\theta)$, $\theta = \pi/64$ | 117.68 | 125.86 | +0.0570 | $5.80 \times 10^{-7}$ |

### Methodological Context & Caveats (Lead Review)
1. **Float vs Q32.32 Non-Associativity:** The chunk equivalence error ($6.6 \times 10^{-7}$) was measured in float32. In the apeiron engine (`src/engine/mixer.rs` `delta_head_step`), arithmetic runs in **fixed-point Q32.32** (`i64`) with intermediate rounding (`round_shift_32`) after every scalar product. Fixed-point arithmetic with rounding is strictly non-associative: $(a \cdot b)_{\text{round}} \cdot c \ne a \cdot (b \cdot c)_{\text{round}}$. Proving chunk equivalence for the engine requires an explicit error bound over $\mathbb{Z}$ for the exact sequence of rounded operations (addressed in Round 2 F5).
2. **Projections vs Measurements:** Projections such as "64× SRAM traffic reduction" and "30B MoE in 8.6 GB at > 300 tok/s" are architectural models (grade `model`), not benchmarked system measurements.

---

## Formal Proof & Verification

- **Lean Module:** [`proof/corpus/Nous/Seq/AffineMixer.lean`](file:///home/socol/Workspace/apeiron/proof/corpus/Nous/Seq/AffineMixer.lean)
- **Primary Gate:**
  `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'cd proof/corpus && LEAN_NUM_THREADS=1 lake build Nous.Seq.AffineMixer'`
  - **Verdict:** GREEN (exit code 0, 2,972 jobs built).
- **Headlines Proved:**
  - `affComp_assoc`: monoid associativity for general affine transformations over commutative ring $R$.
  - `affApply_comp`: composition theorem for action on state matrices.
  - `scanSeq_eq_compose`: equivalence of sequential scan and folded step composition.
  - `recurrent_eq_chunked`: chunked parallel scan matches sequential recurrence over exact arithmetic.
  - `elliptic_2x2_mul_equiv`: 2D rotation block multiplication matches `RingCell` multiplication at $\sigma = -1$.
- **Mutation Control:**
  - Mutated file: `/home/socol/oc-tmp/e1-affine-mixer/control_copy/AffineMixer_mutant.lean`.
  - Mutation: Inverted matrix multiplication order in composition: `M := p.M * q.M` instead of `q.M * p.M`.
  - Control Execution:
    `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 60 bash -c 'LEAN_PATH=... lean /home/socol/oc-tmp/e1-affine-mixer/control_copy/AffineMixer_mutant.lean'`
  - **Verdict:** RED (exit code 1, 4 type/tactic errors in `affComp_assoc` and `affApply_comp`).
  - Restoring clean source produces exit code 0.

---

## Profit Condition Assessment

1. **(a) Collisions ($R(D) < \log_2 t$):**
   Continuous hidden states $S_t \in \mathbb{R}^{d_v \times d_k}$ are unique per token position. Hash or exact match tables do not apply to dynamic recurrent states.
2. **(b) Score Gaps:**
   Not applicable directly; affine recurrence updates sequence memory prior to logit readout.
3. **(c) Binding Resource:**
   In prefill, chunked parallel evaluation allows intra-chunk parallel GEMMs in fast SRAM, reducing DRAM traffic by chunk size $C$ under exact unrounded arithmetic; precision bounds under Q32.32 rounding are analyzed in F5.
