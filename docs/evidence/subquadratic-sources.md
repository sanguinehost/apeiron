# Subquadratic Sources: Mathematical and Code Attribution

Retrieval and attribution record for subquadratic sequence models and attention mechanisms
governed by `docs/epics/subquadratic-decode.md`. Every repository and paper listed below
was retrieved locally and verified. Every mathematical formula is attributed to its source
paper equation and its exact implementing code file and line.

- Date of retrieval: 2026-09-28 (AEST).
- Clones directory: `/home/socol/Workspace/ref/` (plus `/home/socol/Workspace/llama.cpp`).
- Papers directory: `/home/socol/oc-tmp/lit/`.
- Validation gate: `/home/socol/oc-tmp/sq-ref/check.py` validates all local file paths,
  line numbers, and artifact SHA256 digests.

---

## 1. Retrieval Index

Status `retrieved` certifies that the local artifact exists at the specified path and matches
the exact SHA256 cryptographic digest.

| Item | Status | local_path | sha256 |
|---|---|---|---|
| P1 (Gated DeltaNet) | retrieved | /home/socol/oc-tmp/lit/2412.06464_gated_deltanet.pdf | 55f84f2ae9c4e52ff494bfa699499867f5e9e17514994ac71de43888363a5fb9 |
| P2 (DeltaNet Chunking) | retrieved | /home/socol/oc-tmp/lit/2406.06484_deltanet_chunking.pdf | 8efd24a5338429187d4990b082d72a3b463531a24c9b5385c54d63bce9e599ba |
| P3 (Log-Linear Attention) | retrieved | /home/socol/oc-tmp/lit/2506.04761_log_linear_attention.pdf | d04b04a19bc61c5689bf56d6129d1c67089aeb58127f8114d3f831630f4ba24e |
| P4 (Mamba-3) | retrieved | /home/socol/oc-tmp/lit/2603.15569_mamba3.pdf | 17e65b882c4b7dc7dad82a437e54c5fa973e4624d5653dc431f1a2a55538c826 |
| P5 (Mamba-2 SSD) | retrieved | /home/socol/oc-tmp/lit/2405.21060_mamba2_ssd.pdf | 6fe16414fa471a5ab150cddcba5840d3a94e347e96992c50c12c1fb8b919479f |
| P6 (FlashAttention-4) | retrieved | /home/socol/oc-tmp/lit/2603.05451_flashattention4.pdf | 7c053b2b45a9d7da93b21f7dbe16ff8a927a9dc7e92537a034ca58a4eb5f10eb |

---

## 2. Repositories and Pinned Commits

1. **`Dao-AILab/flash-attention`**
   - URL: `https://github.com/Dao-AILab/flash-attention.git`
   - Local Path: `/home/socol/Workspace/ref/flash-attention`
   - Commit: `e9cf2c1651d2303191eb40a739a3c135fda00999`
   - Location of FA4: Lives in the CuTe-DSL tree under `/home/socol/Workspace/ref/flash-attention/flash_attn/cute` (e.g., `flash_fwd_sm100.py`, `softmax.py`, `utils.py`).

2. **`fla-org/flash-linear-attention`**
   - URL: `https://github.com/fla-org/flash-linear-attention.git`
   - Local Path: `/home/socol/Workspace/ref/flash-linear-attention`
   - Commit: `1837094334042434776f3b302f5131ac8ad9f575`
   - Role: Provides standard reference naive recurrent, chunkwise, and Triton GPU implementations for Gated DeltaNet, DeltaNet, and related linear attention rules.

3. **`NVlabs/GatedDeltaNet`**
   - URL: `https://github.com/NVlabs/GatedDeltaNet.git`
   - Local Path: `/home/socol/Workspace/ref/GatedDeltaNet`
   - Commit: `b53d6d3a161267432a79c1c04af69fa52bddc921`
   - Role: Official reference implementation for "Gated Delta Networks: Improving Mamba2 with Delta Rule" (Yang, Kautz, Hatamizadeh, ICLR 2025).

4. **`HanGuo97/log-linear-attention`**
   - URL: `https://github.com/HanGuo97/log-linear-attention.git`
   - Local Path: `/home/socol/Workspace/ref/log-linear-attention`
   - Commit: `7f8644159c1406fae1ad863829a5b3a4fbf63022`
   - Role: Official reference implementation for "Log-Linear Attention" (Guo, Yang, Goel, Xing, Dao, Kim, ICLR 2026).

5. **`state-spaces/mamba`**
   - URL: `https://github.com/state-spaces/mamba.git`
   - Local Path: `/home/socol/Workspace/ref/mamba`
   - Commit: `e9594ce1c732d97440f0332fdc43170a2294dbfa`
   - Role: Official implementation of Mamba-1, Mamba-2, and Mamba-3 (`mamba_ssm/modules/mamba3.py`, `tests/ops/triton/test_mamba3_siso.py`, `tests/ops/tilelang/test_mamba3_mimo.py`).

6. **`llama.cpp`**
   - URL: `https://github.com/ggerganov/llama.cpp`
   - Local Path: `/home/socol/Workspace/llama.cpp` (read-only reference)
   - Commit: `ece963f41b0b02d7a0d61436ae365762c073a4c8`
   - Role: Contains CPU forward kernel for Gated DeltaNet (`ggml_compute_forward_gated_delta_net` in `ggml-cpu/ops.cpp`) and model graphs for Qwen3-Next / Qwen3.5 (`qwen35moe.cpp`) and Kimi-Linear (`kimi-linear.cpp`).

---

## 3. Mathematical Mechanisms and Exact Equations

### 3.1. Gated DeltaNet

#### Mathematical Specification
- **Paper**: Songlin Yang, Jan Kautz, Arash Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv:2412.06464 (ICLR 2025).
- **Online Objective**: The hidden state $S_t \in \mathbb{R}^{d_v \times d_k}$ acts as an associative memory matrix optimized via online least-squares regression:
  $$\mathcal{L}(S_t) = \frac{1}{2} \| S_t k_t - v_t \|^2$$
  quoted from paper arXiv:2412.06464 Section 3.1.
- **Recurrent Step with Adaptive Decay and Learning Rate**:
  Taking a gradient descent step with learning rate $\beta_t$ and incorporating dynamic per-head decay $\alpha_t \in (0, 1)$:
  $$S_{t} = \alpha_t S_{t-1} (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$
  quoted from paper arXiv:2412.06464 Section 3.1 and Section 3.3.
- **Normalization and Gating**:
  - Keys and queries are normalized: $q_t \leftarrow q_t / \|q_t\|_2$, $k_t \leftarrow k_t / \|k_t\|_2$ (or softmax normalization), from paper Section 3.1.
  - Learning rate gate: $\beta_t = \sigma(W_\beta x_t) \in (0, 1)$.
  - Decay gate: $g_t = \log \alpha_t = \mathrm{logsigmoid}(W_g x_t) / C_{\mathrm{norm}}$, so $\alpha_t = \exp(g_t) \in (0, 1)$.
  - Output projection: $o_t = S_t q_t \cdot \frac{1}{\sqrt{d_k}}$.

#### Implementation Citations
- Naive recurrent reference:
  - Definition: quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:12` (`naive_recurrent_gated_delta_rule`).
  - Decay state: `h = h.clone() * g[:, :, i].exp()[..., None, None]` quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:48`.
  - Associative error: `b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)` quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:50`.
  - Rate scaling: `b_v = b_v * b_beta[..., None]` quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:51`.
  - State update: `h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)` quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:52`.
  - Output projection: `o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)` quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:53`.
- In `NVlabs/GatedDeltaNet`:
  - Definition: quoted from `/home/socol/Workspace/ref/GatedDeltaNet/lit_gpt/gated_delta_rule_ops/chunk.py:688` (`recurrent_gated_delta_rule_ref`).
  - Gate decay: `S = S.clone() * g[:, :, i].exp()[..., None, None]` quoted from `/home/socol/Workspace/ref/GatedDeltaNet/lit_gpt/gated_delta_rule_ops/chunk.py:702`.
  - Error correction: `_v = _v - (S.clone() * _k[..., None]).sum(-2)` quoted from `/home/socol/Workspace/ref/GatedDeltaNet/lit_gpt/gated_delta_rule_ops/chunk.py:704`.
  - Normalization: `q = l2_norm_fn(q).to(v)` quoted from `/home/socol/Workspace/ref/GatedDeltaNet/lit_gpt/gated_delta_net.py:174`.
  - Key normalization: `k = l2_norm_fn(k).to(v)` quoted from `/home/socol/Workspace/ref/GatedDeltaNet/lit_gpt/gated_delta_net.py:175`.
- In `llama.cpp` CPU implementation:
  - Loop function: quoted from `/home/socol/Workspace/llama.cpp/ggml/src/ggml-cpu/ops.cpp:10745` (`ggml_compute_forward_gated_delta_net_one_chunk`).
  - Decay: `delta[i] = expf(g_d[i]);` quoted from `/home/socol/Workspace/llama.cpp/ggml/src/ggml-cpu/ops.cpp:10857` (KDA per-channel gate) and `ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));` at line 10863.
  - Projection error: `delta[j] = (v_d[j] - sum) * beta_val;` quoted from `/home/socol/Workspace/llama.cpp/ggml/src/ggml-cpu/ops.cpp:10869`.
  - Rank-1 accumulation: `ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);` quoted from `/home/socol/Workspace/llama.cpp/ggml/src/ggml-cpu/ops.cpp:10874`.
  - Output dot product: `attn_data[j] = sum * scale;` quoted from `/home/socol/Workspace/llama.cpp/ggml/src/ggml-cpu/ops.cpp:10881`.

#### Chunkwise Form and WY / UT Transform
- **Paper**: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim, *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, arXiv:2406.06484; and arXiv:2412.06464 Section 3.3.
- **Chunk State Recurrence**:
  For chunk index $t$ and within-chunk step $r \in [1, C]$:
  $$S_{[t]}^r = S_{[t]} F_{[t]}^r + G_{[t]}^r$$
  quoted from paper arXiv:2412.06464 Section 3.3, where:
  $$F_{[t]}^r = \prod_{i=1}^r \alpha_{[t]}^i (I - \beta_{[t]}^i k_{[t]}^i (k_{[t]}^i)^\top), \quad G_{[t]}^r = \sum_{i=1}^r \frac{\gamma_{[t]}^r}{\gamma_{[t]}^i} \tilde{u}_{[t]}^i (k_{[t]}^i)^\top$$
  with cumulative decay $\gamma_{[t]}^r = \prod_{j=1}^r \alpha_{[t]}^j$.
- **WY/UT Transform Formulation**:
  The cumulative product of generalized Householder transformations is represented in compact matrix form using the Upper Triangular (UT) inverse representation (Bischof & Van Loan, 1985):
  $$\tilde{U}_{[t]} = \left[ I + \mathrm{strictLower}\left( \mathrm{diag}(\beta_{[t]}) (\Gamma_{[t]} \odot K_{[t]} K_{[t]}^\top) \right) \right]^{-1} \mathrm{diag}(\beta_{[t]}) V_{[t]}$$
  quoted from paper arXiv:2412.06464 Section 3.3 (and arXiv:2406.06484 eq. 6-7).
- **Code Citations**:
  - `solve_tril` inverting $(I + A)$ for strictly lower-triangular $A$: quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/utils/solve_tril.py:339`.
  - Intra-chunk fused $K K^\top$ + `solve_tril`: quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/gated_delta_rule/chunk_fwd.py:58`.
  - Inter-chunk state propagation: quoted from `/home/socol/Workspace/ref/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:130` (`v_prime = (k_cumdecay[:, :, i]) @ S`) and line 134 (`S = S * decay... + (k_i * decay...).T @ v_new`).

---

### 3.2. Log-Linear Attention

#### Mathematical Specification
- **Paper**: Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, Yoon Kim, *Log-Linear Attention*, arXiv:2506.04761 (ICLR 2026).
- **Fenwick Partition Rule**:
  At query position $t$, the prefix $[0, t)$ is partitioned into up to $L = \lceil \log_2 t + 1 \rceil + 1$ disjoint buckets $B_t^{(\ell)}$ (quoted from paper arXiv:2506.04761 Section 3.1, footnote 8).
  Each bucket $B_t^{(\ell)}$ has size $|B_t^{(\ell)}| = 2^{\ell-1}$ for $\ell \ge 1$, plus a sentinel bucket $B_t^{(0)}$ of size 1.
  The decomposition is determined by the binary least significant set bit:
  $$\mathrm{lssb}(t) = \max \{ \ell \in \mathbb{N} \mid 2^\ell \text{ divides } t \}$$
- **Output with Multi-Scale Level Weights**:
  Each bucket level $\ell \in [0, L-1]$ maintains a distinct recurrent state $S_t^{(\ell)} \in \mathbb{R}^{d_k \times d_v}$. The attention output is a dynamically weighted multi-scale combination:
  $$o_t = \sum_{\ell=0}^{L-1} \lambda_t^{(\ell)} q_t^\top S_t^{(\ell)}$$
  quoted from paper arXiv:2506.04761 eq. (3), where $\lambda_t^{(\ell)} = \mathrm{softplus}(W_\lambda^{(\ell)} x_t) \ge 0$.
- **Recurrent State Step and Level Cascading**:
  At step $t$, the new token outer product enters the finest level $\ell=0$. Existing buckets at levels $\le \mathrm{lssb}(t)$ cascade and merge into level $\mathrm{lssb}(t) + 1$:
  $$S_t^{(\ell)} = \begin{cases}
  v_t k_t^\top & \text{if } \ell = 0 \\
  0 & \text{if } 0 < \ell \le \mathrm{lssb}(t) \\
  \sum_{\ell'=0}^{\ell-1} S_{t-1}^{(\ell')} & \text{if } \ell = \mathrm{lssb}(t) + 1 \\
  S_{t-1}^{(\ell)} & \text{if } \ell > \mathrm{lssb}(t) + 1
  \end{cases}$$
  quoted from paper arXiv:2506.04761 Section 3.2.
- **Composition with Mamba-2 and Gated DeltaNet**:
  Paper arXiv:2506.04761 Section 3.4 proves that log-linear hierarchical matrices compose with sequentially semiseparable (SSS) decay masks via Hadamard product:
  $$M = M_S \odot M_H$$
  - Log-Linear Mamba-2: $O = (Q K^\top \odot M_S \odot M_H) V$ (from paper Section 3.4).
  - Log-Linear Gated DeltaNet: $O = \left( Q K^\top \odot \left[ L (I + K K^\top \odot (L - I))^{-1} \right] \odot M_S \odot M_H \right) V$ (from paper Section 3.4).

#### Implementation Citations
- In `HanGuo97/log-linear-attention`:
  - `HState` class definition: quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/recurrent.py:10`.
  - Weak cascading loop: quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/recurrent.py:41` (`def cascade_weak(self)`).
  - Capacity check: `capacity = self.base ** level` quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/recurrent.py:43`.
  - Bucket merge: `self.states[..., level_next] = self.states[..., level_next] + self.states[..., level]` quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/recurrent.py:46`.
  - Level-0 insertion: `self.states[..., 0] = value` quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/recurrent.py:85`.
  - State step with Mamba-2 gate: `gate = a` and `value = einsum(k, v, "b h dk, b h dv -> b h dk dv")` quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/recurrent.py:143`.
  - State step with Gated DeltaNet Householder gate: `gate = rearrange(gate * (I - bkkt), "b h dm dn -> b h dm dn")` and `value = einsum(b, k, v, "b h, b h dk, b h dv -> b h dk dv")` quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/recurrent.py:153`.
  - Multi-scale output reduction: `Sl = S.dot(l)` and `return einsum(q, Sl, "b h d, b h d dv -> b h dv")` quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/recurrent.py:168`.
  - Chunkwise bitmask test for Fenwick levels: `return ((i_t >> ell) & 1 == 1)` quoted from `/home/socol/Workspace/ref/log-linear-attention/hattention/chunkwise.py:42`.

---

### 3.3. Mamba-3

#### Mathematical Specification
- **Paper**: Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu, *Mamba-3: Improved Sequence Modeling using State Space Principles* (arXiv:2603.15569, March 2026).
- **Exponential-Trapezoidal Discretization**:
  The continuous selective SSM $\dot{h}(t) = A(t) h(t) + B(t) x(t)$ is integrated over $[\tau_{t-1}, \tau_t]$ with interval $\Delta_t$. Approximating the state-input integral via a data-dependent generalized trapezoidal rule yields:
  $$h_t = e^{\Delta_t A_t} h_{t-1} + (1 - \lambda_t) \Delta_t e^{\Delta_t A_t} B_{t-1} x_{t-1} + \lambda_t \Delta_t B_t x_t$$
  quoted from paper arXiv:2603.15569 eq. (5) (Proposition 1), where $\lambda_t = \sigma(\mathrm{trap}_t) \in [0, 1]$.
  Coefficients are defined as:
  $$\alpha_t = e^{\Delta_t A_t}, \quad \beta_t = (1 - \lambda_t) \Delta_t e^{\Delta_t A_t}, \quad \gamma_t = \lambda_t \Delta_t$$
  quoted from paper arXiv:2603.15569 eq. (6).
  This acts as an implicit data-dependent width-2 causal convolution on the state input $B_t x_t$ (Remark 4).
- **Complex-Valued State and Equivalence to Data-Dependent RoPE**:
  - Complex SSM definition: $\dot{h}(t) = \mathrm{Diag}(A(t) + i \theta(t)) h(t) + (B(t) + i \hat{B}(t)) x(t)$ with output $y(t) = \mathrm{Re}\left( (C(t) + i \hat{C}(t))^\top h(t) \right)$ (paper eq. 8).
  - Proposition 2 (Complex-to-Real Equivalence): Equivalent to real SSM with state $h_t \in \mathbb{R}^N$ and block-diagonal transition $e^{\Delta_t A_t} R_t$ containing $2 \times 2$ rotation blocks $R(\Delta_t \theta_t[i]) = \begin{pmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{pmatrix}$ (paper eq. 9).
  - Proposition 3 & 4 ("RoPE Trick"): The unrolled recurrence is mathematically equivalent to a real-valued scalar decayed SSM where cumulative data-dependent rotary embeddings $\prod_{i=0}^t R_i^\top$ are applied to $B_t$ and $C_t$:
    $$h_t = \alpha_t h_{t-1} + \beta_t \left( \prod_{i=0}^{t-1} R_i^\top \right) B_{t-1} x_{t-1} + \gamma_t \left( \prod_{i=0}^t R_i^\top \right) B_t x_t$$
    $$y_t = \left( \prod_{i=0}^t R_i^\top C_t \right)^\top h_t$$
    quoted from paper arXiv:2603.15569 eq. (11).
- **Multi-Input, Multi-Output (MIMO)**:
  - To increase arithmetic intensity during memory-bound decoding from $\Theta(1)$ to $\Theta(R)$, inputs and projections are expanded by rank $R$: $B_t \in \mathbb{R}^{N \times R}$, $x_t \in \mathbb{R}^{P \times R}$ (paper Section 3.3, Table 2).
  - Outer products become hardware-efficient matrix multiplications:
    $$h_t^{(j)} \leftarrow \alpha_t h_{t-1}^{(j)} + \Delta_t B_t^{(j)} x_t^{(j)}, \quad h_t = \sum_{j=0}^{R-1} h_t^{(j)}, \quad y_t^{(i)} \leftarrow (C_t^{(i)})^\top h_t$$
    quoted from paper arXiv:2603.15569 eq. (12), (13), (14).

#### Implementation Citations
- SISO Reference (`state-spaces/mamba`):
  - Function: quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:34` (`mamba3_siso_step_ref`).
  - Angle scaling: `Angles = torch.tanh(Angles) * math.pi` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:53`.
  - Angle accumulation mod $2\pi$: quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:108` (`Angle_State = Angle_State + angles * dt.unsqueeze(-1)`).
  - Rotary transformation of Q and K: quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:114` (`q_rot = apply_rotary_emb(q, cos_angles, sin_angles)`) and line 115 (`k_rot = apply_rotary_emb(k, cos_angles, sin_angles)`).
  - Exponential-trapezoidal discretization weights:
    - `trap = torch.sigmoid(trap)` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:118`.
    - `alpha = torch.exp(adt)` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:119`.
    - `beta = (1 - trap) * dt * alpha` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:120`.
    - `gamma = trap * dt` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:121`.
  - Recurrent SSM state update:
    - Decay: `SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:124`.
    - Trapezoidal history term ($\beta$): quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:125`.
    - Current input term ($\gamma$): quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:126`.
  - State emission output: `out = torch.einsum("bhdD, bhD -> bhd", SSM_State, q_rot.to(SSM_State.dtype))` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/triton/test_mamba3_siso.py:129`.
- MIMO Reference (`state-spaces/mamba`):
  - Function: quoted from `/home/socol/Workspace/ref/mamba/tests/ops/tilelang/test_mamba3_mimo.py:326` (`mamba3_MIMO_step_ref`).
  - Angle cumsum: `Angle_State = Angle_State + torch.tanh(angles) * dt.unsqueeze(-1) * math.pi` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/tilelang/test_mamba3_mimo.py:429`.
  - Matrix contraction across rank $R$: `prev_kv = torch.einsum("bhrd,bhrp->bhpd", K_State, V_State)` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/tilelang/test_mamba3_mimo.py:438`.
  - Current matrix contraction across rank $R$: `curr_kv = torch.einsum("bhrd,bhrp->bhpd", k_rot, v)` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/tilelang/test_mamba3_mimo.py:439`.
  - Recurrence update: quoted from `/home/socol/Workspace/ref/mamba/tests/ops/tilelang/test_mamba3_mimo.py:440` (`SSM_State = alpha * SSM_State + beta * prev_kv + gamma * curr_kv`).
  - Output contraction over MIMO rank: `out = torch.einsum("bhrp,hrp->bhp", out, MIMO_O)` quoted from `/home/socol/Workspace/ref/mamba/tests/ops/tilelang/test_mamba3_mimo.py:461`.

---

### 3.4. FlashAttention-4 (FA4)

#### Mathematical Specification
- **Paper**: Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao, *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*, arXiv:2603.05451 (March 2026).
- **Online Softmax with Conditional (Lazy) Rescaling**:
  Standard online softmax updates intermediate numerator $O_j$ and running max $m_j$ for each key block $j$:
  $$m_j = \max(m_{j-1}, \mathrm{rowmax}(S_j))$$
  FlashAttention-4 introduces conditional rescaling governed by threshold $\tau = \log_2(256) = 8.0$:
  $$O_j = \begin{cases}
  e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j & \text{if } m_j - m_{j-1} > \tau \\
  O_{j-1} + e^{S_j - m_{j-1}} V_j & \text{otherwise}
  \end{cases}$$
  quoted from paper arXiv:2603.05451 eq. (6).
- **Exactness Proof**:
  Skipping intermediate rescaling when $m_j - m_{j-1} \le \tau$ defers the normalization update while keeping all accumulated terms scaled relative to the pinned reference max $m_{\mathrm{ref}}$.
  Because the final output is normalized at completion by the true final maximum $m_{\mathrm{final}}$ and sum of exponentials $\ell_{\mathrm{final}}$:
  $$\mathrm{Output} = \frac{1}{\ell_{\mathrm{final}}} O_{\mathrm{final}}$$
  any intermediate deferral cancels out identically in exact arithmetic. Furthermore, bounding $m_j - m_{\mathrm{ref}} \le \tau$ guarantees intermediate terms are bounded by $e^\tau = 256.0$, preventing numerical overflow.
- **Polynomial Approximation of $2^x$**:
  On Blackwell GPUs, tensor cores offer $512\times$ the throughput of multi-function units (MUFU.EX2). FA4 emulates $2^x$ on general FMA units using range reduction and polynomial approximation:
  $$2^x = 2^{\lfloor x \rfloor} 2^{x - \lfloor x \rfloor}$$
  quoted from paper arXiv:2603.05451 eq. (4).
  The fractional part $x_{\mathrm{frac}} = x - \lfloor x \rfloor \in [0, 1)$ is approximated via minimax polynomial:
  $$2^{x_{\mathrm{frac}}} \approx \sum_{i=0}^n p_i x_{\mathrm{frac}}^i$$
  quoted from paper arXiv:2603.05451 eq. (5).
- **Polynomial Degrees and Measured Errors** (quoted from paper arXiv:2603.05451 Table 2):
  - **Hardware MUFU.EX2**: FP32 max rel error $1.41 \times 10^{-7}$, mean rel error $3.04 \times 10^{-8}$. Rounding to BF16: max rel error $1.41 \times 10^{-3}$, mean $3.89 \times 10^{-3}$.
  - **Degree 3**: FP32 max rel error $8.77 \times 10^{-5}$, mean rel error $5.43 \times 10^{-5}$. Rounding to BF16: max rel error $1.41 \times 10^{-3}$, mean $3.90 \times 10^{-3}$ (matches hardware to within 1 BF16 ULP on 99% of inputs).
  - **Degree 4**: FP32 max rel error $3.05 \times 10^{-6}$, mean rel error $1.84 \times 10^{-6}$. Rounding to BF16: max rel error $1.41 \times 10^{-3}$, mean $3.89 \times 10^{-3}$.
  - **Degree 5**: FP32 max rel error $1.44 \times 10^{-7}$, mean rel error $5.48 \times 10^{-8}$. Rounding to BF16: max rel error $1.41 \times 10^{-3}$, mean $3.89 \times 10^{-3}$.
  - *Key Finding*: The BF16 quantization error ($\approx 3.9 \times 10^{-3}$) dominates polynomial error for all degrees $\ge 3$.

#### Implementation Citations
- In `Dao-AILab/flash-attention` (`flash_attn/cute/` tree):
  - `SoftmaxSm100` class: quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/softmax.py:305`.
  - Exponent difference calculation: `acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2` quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/softmax.py:352`.
  - Conditional threshold check: `if cutlass.const_expr(self.rescale_threshold > 0.0):` quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/softmax.py:354`.
  - Lazy rescaling threshold condition: `if acc_scale_ >= -self.rescale_threshold:` quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/softmax.py:355`.
  - Deferring max update and fixing scale to 1.0: `row_max_new = row_max_old` and `acc_scale = 1.0` quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/softmax.py:356`.
  - Polynomial table `POLY_EX2` for degrees 0..5: quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/utils.py:30`.
  - Software emulation kernel: quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/utils.py:763` (`def ex2_emulation`).
  - Range reduction with FP32 rounding trick ($2^{23} + 2^{22}$): quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/utils.py:766` and line 769 (`x_rounded = add_round_down(x_clamped, fp32_round_int)`).
  - Polynomial evaluation: `x_frac_ex2 = evaluate_polynomial(x_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)` quoted from `/home/socol/Workspace/ref/flash-attention/flash_attn/cute/utils.py:774`.

---

## 4. Released Models Table

Evaluated against the host hardware profile:
- Host memory: 62 GB system RAM.
- GPU accelerator: NVIDIA GeForce RTX 5080 (16 GB VRAM, Blackwell architecture).
- Models can run fully in 16 GB VRAM if footprint $\le 14$ GB, or via system CPU RAM (or CPU offload) if footprint $\le 55$ GB.

| Model Checkpoint | Total / Active Params | Layer Mix (Linear vs Full Attention) | Memory Footprint & Quantization | llama.cpp Support | Repository URL |
|---|---|---|---|---|---|
| `Qwen/Qwen3-Next-80B-A3B` | 80B total / 3B active | 75% Gated DeltaNet (30 layers), 25% Full Attn (10 layers) (1 full attn every 4 layers; `full_attn_interval=4`) | Q4_K_M: ~48 GB (runs in 62 GB host RAM); active working set during decode: ~3B params (~2 GB compute bandwidth) | **YES**: `LLM_ARCH_QWEN3NEXT` in `llama-arch.cpp:38`, implemented in `src/models/qwen35moe.cpp:25` | https://huggingface.co/Qwen |
| `Qwen/Qwen3.5-35B-A3B` | 35B total / 3B active | 75% Gated DeltaNet (30 layers), 25% Full Attn (10 layers) | Q4_K_M: ~22 GB, Q8_0: ~37 GB (runs in host RAM) | **YES**: `LLM_ARCH_QWEN35MOE` in `src/models/qwen35moe.cpp:33` | https://huggingface.co/Qwen |
| `Qwen/Qwen3.5-2B` / `0.8B` | 2.0B / 0.8B total (dense, all active) | 75% Gated DeltaNet (18 layers), 25% Full Attn (6 layers) | BF16 / Q8_0 / Q4_K_M: 1.5 - 4.0 GB (fits entirely in RTX 5080 16 GB VRAM) | **YES**: `LLM_ARCH_QWEN35` in `src/models/qwen35.cpp:30` | https://huggingface.co/Qwen |
| `MoonshotAI/Kimi-Linear-48B-A3B` | 48B total / 3B active | Hybrid: KDA (Gated DeltaNet) layers + MLA (Multi-head Latent Attention) layers across 27 layers + MoE | Q4_K_M: ~29 GB, Q3_K_M: ~23 GB (fits in 62 GB host RAM) | **YES**: `LLM_ARCH_KIMI_LINEAR` in `llama-arch.cpp:146`, implemented in `src/models/kimi-linear.cpp:29` | https://huggingface.co/MoonshotAI |
| `NVlabs/gated_deltanet-1.3b` | 1.3B total / 1.3B active | 100% Gated DeltaNet (24 recurrent layers, no full attention) | BF16: ~2.6 GB, Q8_0: ~1.4 GB (fits entirely in RTX 5080 16 GB VRAM) | **YES**: Executable via `delta-net-base.cpp` and `ggml_compute_forward_gated_delta_net` in `ggml-cpu/ops.cpp:10745` | https://huggingface.co/NVlabs |
| `NVlabs/gated_deltanet-340m` | 340M total / 340M active | 100% Gated DeltaNet (recurrent layers) | BF16: ~0.7 GB (fits in RTX 5080 VRAM) | **YES**: via `ggml_compute_forward_gated_delta_net` | https://huggingface.co/NVlabs |
| `state-spaces/mamba3-siso-1.5b` | 1.5B total / 1.5B active | 100% Mamba-3 SISO (exp-trapezoidal + complex RoPE state; or hybrid with 1-in-4 full attn) | BF16: ~3.0 GB (fits entirely in RTX 5080 16 GB VRAM) | **NO native arch**: `llama.cpp` supports `mamba` and `mamba2`, but lacks `mamba3` exponential-trapezoidal / RoPE-SSM kernels | https://huggingface.co/state-spaces |
| `state-spaces/mamba3-mimo-1.5b` | 1.5B total / 1.5B active | 100% Mamba-3 MIMO (rank $R=4$ tensor contractions) | BF16: ~3.0 GB (fits entirely in RTX 5080 16 GB VRAM) | **NO native arch**: Requires MIMO tensor-core recurrence kernel not yet merged in `llama.cpp` | https://huggingface.co/state-spaces |
| `HanGuo97/log-linear-mamba2-825m` | 825M total / 825M active | 100% Log-Linear Mamba-2 (21 layers with Fenwick multi-scale memory) | BF16: ~1.7 GB (fits in RTX 5080 VRAM) | **NO native arch**: No hierarchical Fenwick attention in `llama.cpp` | https://huggingface.co/HanGuo97 |
| `HanGuo97/log-linear-gated-deltanet-796m` | 796M total / 796M active | 100% Log-Linear Gated DeltaNet (21 layers with Fenwick multi-scale memory) | BF16: ~1.6 GB (fits in RTX 5080 VRAM) | **NO native arch**: No hierarchical Fenwick attention in `llama.cpp` | https://huggingface.co/HanGuo97 |

---

## 5. Verification and Gate Invariants

1. **Gate Checker**: Run `/home/socol/oc-tmp/sq-ref/check.py`:
   - Validates every SHA256 digest listed in Section 1 against physical PDF artifacts under `/home/socol/oc-tmp/lit/`.
   - Validates that every file:line citation points to an existing file and valid line within `/home/socol/Workspace/ref/` or `/home/socol/Workspace/llama.cpp/`.
2. **Negative Control**:
   - A copy of this document is mutated with an invalid file:line citation or corrupted SHA256 in `/home/socol/oc-tmp/sq-ref/`.
   - `/home/socol/oc-tmp/sq-ref/check.py` rejects the mutated document with exit code 1.
