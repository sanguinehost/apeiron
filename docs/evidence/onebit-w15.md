# ONEBIT W15a: direct KL along the quantisation ray

Command (from the repository root, `/home/socol/Workspace/spinor-ttl/.venv/bin/python`, RTX 5080):
`python tools/onebit/ray_scan.py /home/socol/oc-tmp/escalation/2026-10-11-stiffness.json /home/socol/oc-tmp/escalation/2026-10-11-ray.json 100`.
100 C4 chunks of 512 tokens (`/home/socol/oc-tmp/seal/calib-c4.txt`), wall 139.6 s. The 21 `ffn_down` tensors of
Qwen2.5-0.5B move jointly to `W + tD` (`D` the arm's dequantised error, `fisher_capture.load_deltas`), and the
per-token KL `KL(p_W ‖ p_{W+tD})` is computed in float64.

- Source: `tools/onebit/ray_scan.py` sha256 `ad65df8f0cc58c2d…`; receipt sha256 `40e693d217fdbe6c…`.
- Prediction `ray-1` was withdrawn, not scored: the run was launched as `cd tools/onebit && python ray_scan.py`, so the
  derivation gate's run class did not admit it. The numbers below are `measured`; they carry no prediction verdict.

## Result

`K` is the W14 sampled-Fisher joint second-order KL per token; `r(t) = KL(t) / (t² K)`.

| arm | `KL(1/8)` | `KL(1)` | `KL(−1)` | `64·KL(1/8)` (exact quadratic) | W14 `K` | `KL(1)/(64·KL(1/8))` |
|---|---:|---:|---:|---:|---:|---:|
| `IQ1_S` | 0.0061071 | 0.449916 | 0.501859 | 0.3909 ± 0.0064 | 0.4573 | 1.151 |
| `IQ1_M` | 0.0047723 | 0.344035 | 0.371531 | 0.3054 ± 0.0051 | 0.3299 | 1.126 |
| Q4_K_M | 0.0000704 | 0.004501 | 0.004542 | 0.00451 ± 0.00007 | 0.003965 | 0.998 |
| RTN | 0.0555673 | 9.82565 | 1.59422 | 3.556 ± 0.058 | 3.553 | 2.763 |
| `had-gptq1` | 0.0065456 | 0.460010 | 1.069030 | 0.4189 ± 0.0084 | 0.2919 | 1.098 |

(± is the standard error over the 100 chunks, scaled by 64.)

- **The quadratic form holds to the full step at one-bit error.** `KL(1) / (64·KL(1/8))` is 0.998–1.151 for every
  arm except RTN (2.763). Higher-order terms add at most 15% at `t = 1` on `IQ1_S`, `IQ1_M` and `had-gptq1`.
- **The W14 rank inversion was estimator error.** The exact quadratic orders `had-gptq1` (0.419) > `IQ1_S` (0.391) >
  `IQ1_M` (0.305), the P1 KLD order. The W14 400-sample estimator read `had-gptq1` low by 1.43× and `IQ1_S` high by
  1.17× (`r(1/8)` 1.435 and 0.855).
- **Odd remainder.** At `t = 1`, `odd = (KL(D) − KL(−D))/2` is −0.305 on `had-gptq1` (its error lies on the downhill
  side of a cubic term) and +4.12 on RTN.

## Formal record

`Perf.CostCorrespondence.ray_w15a` (`proof/corpus/Perf/CostCorrespondence.lean`, `proved`, closed cone) states the
three verdicts on these figures as ℚ facts; a changed `had-gptq1` bound (`< 1.09`) reddens it
(`/home/socol/oc-tmp/costcorr-wf-controls.log`).

## What follows

- The water-filling floor (`waterfill_floor`) applies to one-bit error levels on `ffn_down`: the quadratic form is the
  objective to within 15%.
- The sampled-Fisher estimator is too noisy to rank arms. The Kronecker factorisation `G ⊗ H` (`kron_diag`) is the
  next instrument, validated against `64·KL(1/8)` above before any floor is read from it.

## Not covered

- `ffn_down` only, one model (Qwen2.5-0.5B), C4 calibration text; P1 KLD is wikitext-2 through llama.cpp.
- `t ∈ {±1/8, 1/4, ±1/2, ±1}`; the remainder between those points is not bounded.

# ONEBIT W15b: K-FAC water level over every linear tensor

Command (repository root, same venv and GPU): `python tools/onebit/kfac_floor.py
/home/socol/oc-tmp/escalation/2026-10-11-kfac.json 100 4`. 100 C4 chunks, 4 sampled targets per token, all 168 linear
tensors (7 kinds × 24 blocks). Wall 118.3 s (capture 22.0 + 17.8 s in two 12-block halves, analysis 45.0 s).

- Source: `tools/onebit/kfac_floor.py` sha256 `1fb742d7080c0417…`; receipt sha256 `e64115b13698bf50…`; log
  `/home/socol/oc-tmp/escalation/2026-10-11-kfac.log`.
- Control: the gguf f16 weights equal the HF weights the factors are captured on, max relative difference 5.0e-8.
- Prediction `kfac-1-1`: **refuted** on its order clause; its factor, rate, oracle-below-separable and control
  clauses held.

## Method

- Per tensor, `G` is the per-token covariance of the output gradient of `log p(y)` with `y` sampled from the model,
  through the full backward graph (the embedding output is the only leaf). `H` is the input covariance, shared per
  input (`attn_in`, `o_in`, `mlp_in`, `down_in`).
- Per-token KL of an error `D` is read as `½ tr(DᵀGDH)` (`model`: drops cross-token covariance and the dependence of
  `g` on `x`).
- Floors: in the eigenbases `G = U diag(g) Uᵀ`, `H = V diag(h) Vᵀ`, the coefficients `C = UᵀWV` carry weights
  `½ g_i h_j` (`kron_diag`), and global reverse water-filling (`waterfill_opt`, `waterfill_level`) over all 168
  tensors gives `KL(R) = Σ min(λ, θ)` at rate `R = Σ max(0, ½ log₂(λ/θ))`.
  - **separable**: `σ²_ij = r_i c_j / m`, the variance a row-and-column-scaled Gaussian codebook achieves (Lapidoth
    1997, `anchor`); scale side information 16 bits per row and column, counted.
  - **oracle**: `σ²_ij = C_ij²`, an optimistic reference, not a bound.
  - The basis `(U, V)` is not counted in the rate.

## Validation against the exact quadratic

Sum over the 21 `ffn_down` tensors (blocks 3–23) against `64·KL(1/8)` from W15a:

| arm | K-FAC | exact quadratic | ratio |
|---|---:|---:|---:|
| `IQ1_S` | 0.4233 | 0.3909 | 1.083 |
| `IQ1_M` | 0.3366 | 0.3054 | 1.102 |
| Q4_K_M | 0.003621 | 0.004509 | 0.803 |
| RTN | 10.40 | 3.556 | 2.924 |
| `had-gptq1` | 0.2764 | 0.4189 | 0.660 |

- K-FAC reads the four structured arms within 0.66–1.10 of the exact quadratic. RTN is out of the quadratic regime
  (W15a `KL(1)/(64·KL(1/8))` 2.763).
- K-FAC orders `had-gptq1` lowest where the exact quadratic orders it highest. GPTQ shapes its error against `H`
  alone, which is the factor the Kronecker model treats as independent of `g`; the under-read is that assumption,
  not sampling noise (`model`).

## Floor

Q4_K_M's K-FAC KL over all 168 tensors is 0.015675 (q 0.00042, k 0.00074, v 0.00297, o 0.00131, gate 0.00198,
up 0.00398, down 0.00427).

| rate (b/w) | separable KL | oracle KL |
|---:|---:|---:|
| 1.0 | 0.3470 | 0.1437 |
| 1.18 | 0.2645 | 0.1038 |
| 1.5 | 0.1658 | 0.0601 |
| 2.0 | 0.0814 | 0.0269 |
| 2.5 | 0.0403 | 0.0126 |
| 3.0 | 0.0200 | 0.00598 |
| 4.0 | 0.00497 | 0.00141 |
| 4.5 | 0.00248 | 0.000695 |

- Rate that matches Q4_K_M's KL: **3.176 b/w separable, 2.353 b/w oracle**.
- At 1.18 b/w the floor is 16.9× (separable) and 6.6× (oracle) Q4_K_M's KL.

## Defect found: truncated backward in `fisher_capture.py`

`tools/onebit/fisher_capture.py:79` replaces every captured `down_proj` output with a detached leaf. The gradient at
block `b` then flows through no later block's MLP branch, so the W14 per-block scores used a truncated gradient.
`kfac_floor.py` detaches only the embedding output. The full-graph K-FAC still under-reads `had-gptq1` (0.660), so the
truncation is not the whole of the W14 inversion.

## Formal record

`Perf.CostCorrespondence.kfac_w15b` (`proved`, closed cone) states the validation band, the refuted order, the
16×/6× floor factors and the matching rates as ℚ facts. Controls: the order flipped and the separable factor raised to
17× each redden it (`/home/socol/oc-tmp/costcorr-kfac-controls.log`).

## What follows

- On Qwen2.5-0.5B, post-training coding in the K-FAC metric needs ≥ 2.35 b/w (oracle) to reach Q4_K_M's KL; 1.18 b/w
  is 6.6× above Q4 even there (`model`).
- The open question for the 27B target is how the matching rate scales with model size. The next measurement is the
  same floor on larger Qwen2.5 checkpoints, registered before it runs.

## Not covered (W15b)

- One model, 0.5B; embeddings and `lm_head` excluded; cross-token terms and `g`–`x` dependence dropped.
- Gaussian-codebook rate model; basis side information uncounted; the oracle is not achievable.
- C4 calibration; P1 KLD is wikitext-2 through llama.cpp.

# ONEBIT W15c: the converse side and the floor against model size

Command (repository root, same venv and GPU): `python tools/onebit/converse.py {controls,main,ekfac,scale} …`.
Source `tools/onebit/converse.py` sha256 `29034f02d9002f6f…`; receipts
`/home/socol/oc-tmp/escalation/2026-10-11-converse-{controls,main,ekfac,scale-*}.json` with `.log`.

## Measured structure a 1.18 b/w code could exploit (Qwen2.5-0.5B, 168 tensors)

`slb_gap`: under the Shannon lower bound (`anchor`, Shannon 1959), a code that reaches Q4_K_M's KL at 1.18 b/w needs
negentropy `J ≥ R_g(D_Q4) − 1.18`: 1.994 b/w on the separable curve, 1.173 b/w on the oracle curve.

| structure | measured | estimator control |
|---|---:|---|
| marginal negentropy, K-FAC coefficient basis at the Q4 level | 6.9e-5 b/component | Gaussian 3.6e-5; Laplace 0.10465 (truth 0.10440) |
| marginal negentropy, raw weight basis | 0.00815 b/component | as above |
| causal-neighbour context gain | −0.031 b | iid-Laplace −0.032 b (estimator noise) |
| linear structure (Gaussian total correlation over the shuffled null) | 0.141 b/w (`attn_q` 0.77, `attn_o` 0.53, `attn_k` 0.38, FFN 0.05–0.10) | AR(1) −0.4%, iid −0.3% |

- Measured `J` < 0.15 b/w against the 1.994 (separable) and 1.173 (oracle) b/w required. On 0.5B, marginal, context
  and linear structure do not close the gap. Nonlinear structure was not probed.
- `K0` from log-determinants is 1.1918, within 15% of the bisection `K0` (1.227 at 4.5 b/w, 1.312 at 1.18 b/w).
- Eigenvalue-corrected K-FAC moves `had-gptq1` from 0.659 to 0.701 of the exact quadratic; the order inversion
  persists (prediction `conv-ekfac-1` refuted).

## Variance ladder on Qwen2.5-0.5B (`tools/onebit/variance_ladder.py`)

Receipt: `/home/socol/oc-tmp/escalation/2026-10-11-vladder.json` (138.6 s, fp32 tensors). Prediction
`variance-ladder-1-1` **refuted** (falsifier fired: no structured variance model recovers `≥ 0.1` b/w net of side
information).

| variance model | `R*` coeff-only (b/w) | `R*` all-in (b/w) | floor KL at 1.133 b/w all-in |
|---|---:|---:|---:|
| separable (`r_i c_j / m`, control) | 3.151 | **3.176** | 0.2837 (uniform-rate: 0.3887, 1.37×) |
| block-128 | 3.148 | 3.273 | — |
| block-32 | 3.129 | 3.629 | — |
| rank-2 NMF | 3.146 | 3.196 | — |
| rank-4 NMF | 3.143 | 3.243 | — |
| rank-8 NMF | 3.137 | 3.337 | — |
| oracle (`C_ij²`, 0 side info, control) | 2.353 | **2.353** | 0.1128 |

- Structure buys `≤ 0.02` b/w before side information (`3.129` vs `3.151`) and loses after side information is counted:
  the `0.82` b/w separable-to-oracle gap on 0.5B is per-coefficient (`z_ij²`, `χ²_1` log-moment shift
  `E[½ log₂(z²)] = −0.9164` b/w), not group or low-rank envelope.

## The Q4-matching rate across model size (0.5B → 1.5B → 3B → 8B → 32B)

1. **8-layer subset probe (`tools/onebit/converse.py scale`)**: Q4_K_M made locally for 1.5B and 3B (llama.cpp imatrix
   on 100 C4 chunks at `c = 512`, files in `/home/socol/oc-tmp/converse/`). bf16 vs fp32 on 0.5B changes `R*` by 0.0016
   (`conv-s05b-1` confirmed). Subset control (`subset-bias-1`): 8 stratified layers read `R*` 0.123 b/w low on 0.5B
   (`3.051` vs `3.174`).
2. **Full-layer streamed capture (`tools/onebit/rstar_stream.py`, sha256 `ba392c24…`)**:
   - Control `rstar-ctl-1` (0.5B, 24 layers fp32): bisection `R*` all-in **3.1731** (ref 3.1743, diff `0.0012`), `K0`
     1.1852 (`−0.47%`), `kl_q4` 0.015700, wall 109 s (HF logprob clause `5e-4` vs `< 1e-4` missed).
   - Mutant `rstar-mut-1` (`--dscale 2`, scaling Q4 error `×2`, KL `×4`): bisection `R*` all-in **2.1823** (move
     `−0.991` b/w, high-res `−1.000` b/w), confirming `waterfill_exclusion` (`4^{−ΔR}`).
   - Full-model Qwen3-8B (36 layers fp32, receipt `/home/socol/oc-tmp/escalation/2026-10-11-rstar-qwen3-8b.json`,
     `rstar-8b-1` confirmed) and Qwen3-32B (64 layers fp32, receipt
     `/home/socol/oc-tmp/escalation/2026-10-11-rstar-qwen3-32b.json`, `rstar-32b-1` refuted against `[0.9, 1.9]`).
3. **Fast one-process instrument (`tools/onebit/rstar_fast.py`, sha256 `96031c3d…`, prior SLQ sha256 `fef86981…`)**:
   - Own-importance Q4_K_M quantization from captured `diag(H)` (Q1 control: 0 differing rows vs `llama-quantize
     --imatrix`, byte-exact).
   - SLQ mode (`--eig slq`, receipts `/home/socol/oc-tmp/rstar-fast/{dev05-fix2,dev8-fullA}.json`): 0.5B `R*` all-in
     **3.1801** (`+0.0070`, 27.4 s, `rstar-fast-05b-1` confirmed); Qwen3-8B `R*` all-in **2.2356** (`−0.0190`, 213.6 s vs
     1897 s streamed, `8.9×` faster, `rstar-fast-8b-1` confirmed).
   - Exact-eigenbasis oracle mode on Qwen3-8B (`--eig exact`, receipt
     `/home/socol/oc-tmp/escalation/2026-10-12-rstar-qwen3-8b-oracle.json`, `rstar-8b-oracle-1` confirmed, wall
     `474.0` s vs projected `457.8` s): computes `C² = (UᵀWV)²` and bisects both separable and 0-side-info oracle rates.

| model | layers | `N` | Q4_K_M K-FAC KL | `R*` sep all-in (b/w) | `R*` oracle (b/w) | `R*_orc / R*_sep` | `KL_sep(1.18) / KL_Q4` | `KL_orc(1.18) / KL_Q4` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 24 / 24 | 0.358 B | 0.015700 | **3.1731** (fast SLQ 3.1801) | **2.3526** | 0.7409 | 16.9× | 6.62× |
| Qwen2.5-1.5B | 8 / 28 | 1.311 B | 0.017634 | **2.780** (sub8; +0.12 ≈ 2.90) | — | — | — | — |
| Qwen2.5-3B | 8 / 36 | 2.775 B | 0.018151 | **2.378** (sub8; +0.12 ≈ 2.50) | — | — | — | — |
| **Qwen3-8B** | **36 / 36** | **6.946 B** | **0.018774** | **2.2546** (fast exact **2.2530**, SLQ 2.2356) | **1.5908** | **0.7061** (0.7081 net) | **5.27×** | **2.11×** |
| **Qwen3-32B** | **64 / 64** | **31.206 B** | **0.024157** | **1.9927** (net 1.9884) | *1.33–1.41 (derived)* | *0.67–0.71* | **3.78×** | — |

- Separable per-tensor water-filling flattens near `2.0` b/w at 32B (`1.9927` b/w, missing `[0.9, 1.9]` by `0.093`).
- At Qwen3-8B, the 0-side-info K-FAC oracle floor (`σ²_ij = C_ij²`) is **1.5908 b/w** (`ΔR* = 0.6622` b/w all-in,
  `0.6559` net; ratio `0.7061`), leaving `2.11×` Q4_K_M's KL at 1.18 b/w and projecting to `1.33–1.41` b/w at 32B:
  within-tensor K-FAC rate allocation cannot reach 1.18 b/w even with oracle variances.

## Formal record (W15c & §8 faces)

- `Perf.CostCorrespondence.{waterfill_highres, waterfill_exclusion, slb_gap, converse_w15c}` and verdicts
  `{method_gap_1133, waterfill_gain_1133, floor_gap_1133, variance_ladder_refuted}` (`proved`, closed cone, audited in
  `proof/corpus/AxiomAudit.lean`); 9/9 mutation controls (`m13–m21`) exit 1
  (`/home/socol/oc-tmp/escalation/2026-10-11-converse-lean.log`).
- `Perf.CostCorrespondence` §8 (`disc2_eq`, `disc2_sq`, `face_mul_canonical`, `rotation_elliptic`, `shear_parabolic`,
  `shear_nilpotent`, `rebalance_hyperbolic`, `rebalance_preserves_product`, `corrector_identity`, `corrector_residual`,
  `corrector_attained`, `corrector_zero_iff`, `spectrum_conj`): proves that `a·1 + b·N` (`N = M − tr/2`) multiplies as
  `Apeiron.Trichotomy.mul (disc2 M)`, assigns the three pipeline faces (Hadamard rotation elliptic `< 0`, unit-triangular
  feedback parabolic `= 0`, scale rebalance hyperbolic `> 0`), proves the cross-matrix corrector residual
  `‖δ + c p‖²‖p‖² ≥ ‖δ‖²‖p‖² − ⟪δ,p⟫²`, and proves `charpoly((TSTᵀ)((Tᵀ)⁻¹FT⁻¹)) = charpoly(SF)` (`proved`, closed cone;
  controls `M1–M3` and `B1` exit 1).

# ONEBIT W15d: output-fitted transfer, fp32 port and three-face completion

Command: `python tools/onebit/fold_transfer.py …` (fp32 port + Hadamard/W0 sha256 `702271978dd2edf7…`, prior fp64 sha256
`4032c1e27555ffb6…`), receipts `/home/socol/oc-tmp/escalation/2026-10-11-fold-transfer{,-block}.json` and
`/home/socol/oc-tmp/escalation/2026-10-12-fold-transfer-had-fp32.json`. Qwen2.5-0.5B, all 24 blocks, one-bit signs with
per-row 128-group `f16` scale `r` and per-column `f16` scale `c` at **1.1331 b/w all-in**, 100 C4 calibration chunks,
40 held-out.

- The initial W15d configuration (`fold-block`) ran in `float64` without the elliptic face (Hadamard-128 rotation) and
  regularized the ridge solve `(H + λI)⁻¹XᵀY` toward `W = 0` (`λ = 10⁻³`), which shrinks `W_0` on small eigenvalues of
  `H` while leaving `ΔY = Y − X_s W_0ᵀ` under-damped (`kl_insample` 0.86 vs `kl_heldout` 3.09).
- Because `r` is constant on each 128-column block and `R = I ⊗ (H₁₂₈/√128)` (`rotation_elliptic`) is block-orthogonal
  on the same 128-blocks, `((S_rot ⊙ (r ⊗ 1₁₂₈)) R) diag(c) = ((S_rot R) ⊙ (r ⊗ 1₁₂₈)) diag(c)`: rotating `W_c` and
  `H_c` by `R` before `gptq_signs` and passing `S̃ = S_rot R` into `c_step` and `r_step` combines all three faces of §8
  (`rotation_elliptic` + `shear_parabolic` + `rebalance_hyperbolic`) at the identical `1.1331` b/w rate (`had-fold-block`).
- Completing the square on `‖X_s Wᵀ − Y_t‖_F² + λ d̄ ‖W − W_0‖_F² = J(W*_{W_0}, W, H_λ) + const` with
  `W*_{W_0} = (H + λ d̄ I)⁻¹(X_sᵀ Y_t + λ d̄ W_0ᵀ)` and `λ = 0.01 = damp` (`had-fold-block-w0`) unifies the metric
  `H + 0.01 d̄ I` across `ridge_solve`, `gptq_signs`, `c_step` and `r_step` without shrinking `W_0` on the flat tail of
  `H`.

| variant | dtype | b/w (linears) | in-sample KL | held-out KL (nats/tok) | held-out top-1 | transfer wall |
|---|---|---:|---:|---:|---:|---:|
| RTN | fp64 | 1.1250 | 12.86 | 12.70 | 0.000 | 0.03 s |
| GPTQ one-bit, teacher Grams (`gptq1-t`) | fp64 | 1.1250 | 15.03 | 14.86 | 0.000 | 31.6 s |
| GPTQ one-bit, student Grams (`gptq1-s`) | fp64 | 1.1250 | 6.72 | 7.08 | 0.047 | 38.0 s |
| ridge target + GPTQ (`fold-ns`) | fp64 | 1.1250 | 3.92 | 5.25 | 0.087 | 44.9 s |
| + `(c, r)` refits, layer target (`fold-local`) | fp64 | 1.1331 | 2.44 | 3.81 | 0.206 | 90.1 s |
| + block-output target (`fold-block`, fp64) | fp64 | 1.1331 | 0.862 | 3.090 | 0.2607 | 88.8 s |
| `fold-block` (fp32 port) | **fp32** | 1.1331 | 0.871 | **3.119** | 0.2623 | **39.0 s** (`2.28×` faster) |
| + elliptic face `H₁₂₈/√128` (`had-fold-block`) | **fp32** | 1.1331 | 0.619 | **2.505** | 0.2959 | **38.0 s** |
| + teacher-anchored ridge `λ = 0.01` (`had-fold-block-w0`) | **fp32** | 1.1331 | 0.573 | **2.323** | **0.3125** | **38.1 s** |
| + decoupled `n/896` prior, per-tensor `c`, 2-plane GQA KV (`had-fold-2plane-gqa`) | **fp32** | 1.1552 | 0.672 | **2.031** | **0.3395** | **49.3 s** |
| + rank-2 orthogonal corrector on `q,o,down` (`had-fold-2plane-gqa-corr`) | **fp32** | **1.1753** | 0.646 | **2.003** | **0.3475** | **49.0 s** |
| + 0-bit descending curvature order (`had-fold-2plane-gqa-act`) | **fp32** | **1.1753** | 0.651 | **2.000** | 0.3435 | **51.8 s** |
| + Qronos `n/896` quantizer damping + `q=3,o=3,down=2` (`had-fold-2plane-gqa-qronos`) | **fp32** | **1.1791** | 0.663 | **1.999** | 0.3433 | **51.2 s** |
| + Zellner `g`-prior shrinkage + whitened scale (`had-fold-2plane-gqa-zellner`) | **fp32** | **1.1791** | 0.695 | **1.995** | 0.3471 | **76.4 s** |
| + whitened Cholesky scale + 2-pass intra-block signs (`had-fold-2plane-gqa-wscale`) | **fp32** | **1.1791** | 0.651 | **1.992** | 0.3456 | **76.2 s** |
| + 2-plane full attention (`q,k,v,o`) + `uint4/uint8` hier scales (`had-fold-2plane-attn-hier`) | **fp32** | **1.1797** | **0.581** | **1.867** | **0.3618** | **84.1 s** |
| + 0-bit Gauss-Seidel 2-plane back-fitting (`had-fold-2plane-attn-backfit`) | **fp32** | **1.1797** | 0.593 | **1.858** | **0.3622** | 540.2 s |
| + Hadamard butterfly `d_step` + 0-bit sign-group gauge (`had-fold-2plane-attn-bscale`) | **fp32** | **1.1795** | 0.602 | **1.850** (`93.9×`) | 0.3591 | **110.1 s** |
| + reduced-rank drift filter `r=64, γ=0.65` + `G`-weighted scale (`had-fold-2plane-attn-rrr`) | **fp32** | **1.1795** | 0.697 | 1.899 (`96.4×`) | 0.3613 | **110.9 s** |
| + `uint4` gauge-normalized `(c, d)` + rank-2 `down` corrector (`had-fold-2plane-attn-rk2`) | **fp32** | **1.1797** | 0.596 | 1.857 (`94.3×`) | 0.3591 | 149.2 s |
| + `uint4` `(c, d)` + rank-2 `down` + intra/inter split (`had-fold-2plane-attn-rk2-split`) | **fp32** | **1.1797** | 0.591 | 1.853 (`94.1×`) | **0.3627** | 152.1 s |
| + `bscale` (`uint6 c, uint5 d`) + intra/inter split (`had-fold-2plane-attn-bscale-split`) | **fp32** | **1.1795** | **0.583** | **1.850** (`93.9×`) | 0.3583 | **113.2 s** |
| + butterfly-weighted GPTQ (`had-fold-2plane-attn-bscale-split-bgptq`) | **fp32** | **1.1795** | 0.583 | 1.868 (`94.8×`) | 0.3574 | 150.3 s |
| + Face Swap #1A: pre-Hadamard Cauchy-Schwarz squeeze (`had-fold-2plane-attn-bscale-hpre`) | **fp32** | **1.1795** | 0.583 | 1.884 (`95.6×`) | 0.3575 | 114.0 s |
| + Face Swap #1B: readout `RMSNorm` diagonal Fisher (`had-fold-2plane-attn-bscale-fnorm`) | **fp32** | **1.1795** | 7809.5 | 8159.9 (`4.14e5×`) | 0.0486 | 123.4 s |
| + **Face Swap #2**: 8-D `E₈ ∪ O₈` spinor+sparse codebook (`had-fold-2plane-attn-bscale-e8`) | **fp32** | **1.1795** | **0.469** | **1.633** (`82.9×`) | **0.3943** | **62.7 s** |
| + Face Swap #2 on `rk2` (`had-fold-2plane-attn-rk2-e8`) | **fp32** | **1.1797** | 0.484 | 1.671 (`84.8×`) | 0.3836 | **77.9 s** |
| + **Face Swap #3**: delayed-decision `beam=4` over `E₈ ∪ O₈` (`had-fold-2plane-attn-bscale-e8-beam`) | **fp32** | **1.1795** | **0.446** | **1.593** (`80.9×`) | **0.3989** | **99.0 s** |
| + Face Swap #3 `beam=8` over `E₈ ∪ O₈` (`had-fold-2plane-attn-bscale-e8-beam8`) | **fp32** | **1.1795** | 0.458 | 1.612 (`81.8×`) | 0.3892 | 108.3 s |
| + Face Swap #4 (undamped): global 8-tuple ICM (`had-fold-2plane-attn-bscale-e8-icm`) | **fp32** | **1.1795** | 0.467 | 1.610 (`81.7×`) | 0.3900 | 109.4 s |
| + Face Swap #4 (`beam=1` + undamped ICM, `had-fold-2plane-attn-bscale-e8-b1-icm`) | **fp32** | **1.1795** | 0.489 | 1.678 (`85.2×`) | 0.3866 | 93.0 s |
| + Face Swap #4 (`λ_icm = 0.15`, `had-fold-2plane-attn-bscale-e8-icm15`) | **fp32** | **1.1795** | 0.456 | 1.578 (`80.1×`) | 0.3984 | 110.5 s |
| + **Face Swap #4 (`λ_icm = 0.20` MP-damped ICM, `had-fold-2plane-attn-bscale-e8-icm20`)** | **fp32** | **1.1795** | **0.427** (`24.5×`) | **1.522** (`77.2×`) | **0.4072** | **113.8 s** |
| + **Face Swap #5 (`inter_gamma_in = 0.25` + `λ_icm = 0.20`, `had-fold-2plane-attn-bscale-e8-icm20-gin25`)** | **fp32** | **1.1795** | **0.430** (`24.7×`) | **1.497** (`76.0×`) | **0.4141** | **108.4 s** |
| + Face Swap #5 (`inter_gamma_in = 0.25` + `λ_icm = 0.25`, `had-fold-2plane-attn-bscale-e8-icm25-gin25`) | **fp32** | **1.1795** | 0.434 (`24.9×`) | **1.500** (`76.1×`) | 0.4104 | **109.3 s** |
| + Face Swap #6: SwiGLU bilinear `gate → up` (`had-fold-2plane-attn-bscale-e8-icm20-gin3515-glu`) | **fp32** | **1.1795** | 0.433 (`24.9×`) | 1.502 (`76.2×`) | 0.4106 | 111.4 s |
| + Face Swap #6: SwiGLU bilinear `gate → up` (`had-fold-2plane-attn-bscale-e8-icm20-gin25-glu`) | **fp32** | **1.1795** | 0.435 (`25.0×`) | 1.503 (`76.3×`) | 0.4107 | 110.8 s |
| + **Face Swap #7: Riemannian `RMSNorm` sink gating + row trust (`had-fold-2plane-attn-bscale-e8-icm20-gin25-sink`)** | **fp32** | **1.1795** | **0.399** (`23.0×`) | **1.421** (`72.1×`) | **0.4212** | **88.1 s** |
| + **Face Swap #7: sink gating + coherence-anchored readout `fnorm` (`had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm`)** | **fp32** | **1.1795** | **0.375** (`21.6×`) | **1.378** (`69.9×`) | **0.4219** | **93.9 s** |
| + concatenated `c`, `λ_down = 0.03`, rank-3/2 correctors (`v2`) | **fp32** | 1.1797 | **0.545** | 2.079 (`105.5×`) | 0.3317 | 40.6 s |
| control: full-precision ridge (`fp-ridge`) | fp64 | — | 0.0033 | 0.0195 (`0.99×`) | 0.9213 | 22.0 s |
| control: shuffled target (`fp-ridge-mut`) | fp64 | — | 6.99 | 7.09 (`359.9×`) | 0.0315 | 22.2 s |
| reference: `Q4_K_M` (`q4km`) | f32 deq | 5.5207 | 0.0174 (`1.00×`) | 0.0197 (`1.00×`) | 0.9137 | — |

- Prediction `fold-had-fp32-1` **confirmed** (projected wall `131.9` s, measured `141.1` s; `fold-block` fp32 within
  `+0.029` of fp64; `had-fold-block` `2.505 < 3.00`; `had-fold-block-w0` `2.323 ∈ [1.2, 2.9]`; controls `c9` involution
  `4.8e-7`, quadratic invariance `9.1e-8`).
- Predictions `fold-2plane-gqa-1..4` and `fold-3plane-gqa-1` (`/home/socol/oc-tmp/escalation/2026-10-12-fold-transfer-2plane-gqa.json`,
  script `sha256` `db041e45fe83efd8f9bf4d8ff614ad753539d4b33ae923c4d2fb5c81684bc925`, control `c10` orthogonal
  corrector eigenvalue drop `4.57e-8 < 1e-4`, wall `112.97` s vs `112.8` s projected) trace the exact bias-variance
  parabola of `Perf.CostCorrespondence.{decoupled_ridge_opt, decoupled_ridge_eq_iff, orthogonal_corrector_attained, gqa_kv_second_plane_budget}`:
  - Splitting shared column scales `c`, absorbing the 0-bit DC residual into `qb, kb, vb`, scaling the `W₀` prior by
    `0.03 · (n / 896)` (`0.163` on `ffn_down`), and adding a second `(S, r, c)` residual plane on GQA `k_proj, v_proj`
    (`had-fold-2plane-gqa` at `1.1552` b/w) drops held-out KL from `2.3235` → **`2.0311` nats** (`−0.2924` nats,
    generalization ratio `4.05×` → `3.02×`).
  - Adding the rank-2 FP16 orthogonal corrector (`lowrank_corrector`) on `q, o, down` (`had-fold-2plane-gqa-corr` at
    `1.1753 ≤ 1.18` b/w) drops held-out KL further to **`2.0031` nats** (`0.642×` of `fold-block` `3.1189`, `7.42×`
    below `gptq1-t` `14.86`) and raises held-out top-1 agreement to **`34.75%`** (`+8.52` pp over `fold-block`).
  - Weakening the `ffn_down` prior to `0.03` (`v2`) pushes in-sample KL down to **`0.5449` nats** (`1.40×` of the
    uniform-rate separable floor `0.3887`) while raising held-out KL to `2.0791`; increasing the prior factor to `0.04`
    (`v3`) raises both in-sample (`0.7036`) and held-out (`2.0420`) KL, bracketing the `0.03` optimum on `51,200`
    calibration tokens.
  - Two-component intra/inter decoupled shrinkage (`Perf.CostCorrespondence` §10: `telescoping_target_split`,
    `two_component_shrinkage_identity`, `two_component_shrinkage_opt`, `coupled_shrinkage_excess_identity`,
    `coupled_shrinkage_excess_pos`; receipts `/home/socol/oc-tmp/escalation/2026-10-12-fold-transfer-{split,wfill}-gqa.json`
    and `...-split-gqa-v3.json`, control `c11` telescoping equivalence `7.65e-7 < 1e-4`, `fold-split-gqa-1..3`):
    splitting `C_s − H W_0ᵀ` into local intra-sublayer error `C_intra = C_ts − H W_0ᵀ` (`q/k/v → o` and `gate/up → down`
    on student input `h_s`) and cross-layer drift `C_inter = C_s − C_ts` reveals that `m_ts − m_s` is 1-bit quantization
    noise already near-orthogonal to `m_s` (`λ_intra > λ_inter` required; inverting to `λ_intra = 0.025 < λ_inter = 0.326`
    in `fold-split-gqa-1` raised KL to `2.1086`). With `λ_intra = 0.045(n/896) > λ_inter = 0.025(n/896)` (`fold-split-gqa-2`),
    decoupled `had-fold-2plane-gqa-split-corr` (`1.1795 ≤ 1.18` b/w) beats single-ridge `had-fold-2plane-gqa-wfill` at the
    identical `1.1795` b/w rate on both held-out (`2.0195` vs `2.0353`, `−0.0158` nats) and in-sample (`0.6484` vs `0.6644`,
    `−0.0160` nats, top-1 `58.29%` vs `57.73%`).
  - Qronos Cholesky Schur diffusion, descending-curvature ordering, whitened block-scale projection, full-attention
    2-plane hierarchical scale reallocation, Gauss-Seidel back-fitting & butterfly sign-group gauge (`Perf.CostCorrespondence`
    §11–§12: `qronos_orth_step_identity`, `cholesky_step_schur_identity`, `act_order_curvature_dominance`,
    `sign_mag_dispersion_identity`, `subband_scale_excess_identity`, `subband_scale_excess_pos`,
    `whitened_scale_shift_identity`, `zellner_tikhonov_identity`, `zellner_unit_gamma_excess_pos`,
    `scale_vs_plane_distortion_dominance`, `full_attn_2plane_hier_scale_budget`, `butterfly_scale_commutation`,
    `butterfly_d_step_dominance`, `two_plane_backfit_descent`, `butterfly_2plane_attn_budget`, `four_factor_gap_identity`,
    `w15_four_factor_05b_decomposition`, `butterfly_sign_group_gauge`, `output_weighted_scale_excess_identity`,
    `output_weighted_scale_dominance`, `reduced_rank_drift_generalization_dominance`; receipts
    `/home/socol/oc-tmp/escalation/2026-10-12-fold-transfer-{qronos-gqa,zellner-gqa,attn-hier,attn-bscale,attn-rrr}.json`,
    `sha256` `097d91301b3e8c43d991d036418554457e59ff73a64e78950a076cb09466d44e`,
    `872c0e8735a82a2a6f554da6331d35a515fabf459555129837d092eefb459f2f`,
    `90d5622e34cfd606f426cf12a7f58283ae746945068dee0e716efb92fe0ef1bc`,
    `44126a3a42938ef773e18138f85429584ecc1a66604719e04ba68bcbeb03077e`, and
    `28bf9d5f5ceb1eede1ee15e573db827569e9e65297ddf3c0e59fb477b421378a`, script `sha256`
    `7206ca64569900533ca109caca5920751bc488c0cac0d7a51eeec12c2fd9f6c9`, controls
    `c12_act_order_j_ratio_fp32 = 0.971559 < 1.0`, `c13_wscale_vs_act_j_ratio_fp32 = 0.986953 < 1.0`,
    `c14_hier_scale_rel_frob_fp32 = 0.000435 < 0.01`, `c14_2plane_hier_vs_1plane_f16_j_ratio_fp32 = 0.319520 < 0.45`,
    `c15_dstep_j_ratio_f64 = 0.734782 < 0.99`, `c15_dstep_perturbations_below_min = 0`,
    `c15_2plane_backfit_j_ratio_fp32 = 0.914763 < 0.99`, `c16_butterfly_gauge_maxabs_f64 = 1.33e-15 < 1e-12`,
    `c16_g_weighted_cstep_j_ratio_f64 = 0.999963 < 1.0`, `c16_drift_filter_heldout_j_ratio_f64 = 0.061923 < 1.0`,
    `fold-qronos-gqa-1`, `fold-zellner-gqa-1`, `fold-attn-hier-1`, and `fold-attn-rrr-1` **confirmed**; `fold-attn-bscale-1`
    refuted against `≤ 1.79` before the sign-group gauge fix):
    - Sorting the `G = n/128` groups by descending trace and the `128` columns inside each group by descending diagonal
      curvature `H_jj` before Cholesky diffusion and restoring native order after (`0` bit overhead, `0` inference
      overhead) cuts single-matrix quadratic error by `2.84%` (`c12`) and lowers 24-block held-out KL at `1.1753` b/w to
      **`2.0000` nats** (`had-fold-2plane-gqa-act`), while adding `n/896` quantizer damping and `q=3,o=3,down=2` correctors
      (`had-fold-2plane-gqa-qronos`, `1.1791 ≤ 1.18` b/w) breaks `2.00` at **`1.9990` nats**.
    - Projecting the 128-column block scale `a` directly in whitened Cholesky space (`(e₀ − δa·u_s) U_blk = w − (a₀ + δa) s`,
      `whitened_scale_shift_identity`) with 2-pass intra-block sign refinement before inter-block error propagation
      (`had-fold-2plane-gqa-wscale`, **`1.1791 ≤ 1.18` b/w**, `0` extra bits) cuts single-matrix quadratic error by an
      additional `1.30%` on top of `act_order` (`4.11%` combined, `c13`) and improves held-out KL to **`1.9920` nats**
      (`101.1× Q4_K_M`, `top1_heldout = 34.56%`, in-sample **`0.6506` nats**, wall `76.16` s; Zellner variant
      `had-fold-2plane-gqa-zellner` reaches **`1.9948` nats** and **`34.71%` top-1**).
    - Compressing `(r, c)` hierarchically (`4` `fp16` tensor headers + `8`-bit row header + `uint4` group scale for `r`,
      `2` `fp16` headers + `uint8` column scale for `c`, `0.0435%` relative Frobenius error on `r`) frees `31.9M` bits
      (`0.0891` b/w), funding a full second binary residual plane across **all four attention projections (`q, k, v, o`)**
      plus a rank-1 `fp16` corrector on `down_proj` (`had-fold-2plane-attn-hier` at **`422,142,720 / 357,826,560 =
      1.179741 ≤ 1.18` b/w**, matching `full_attn_2plane_hier_scale_budget`), cutting single-matrix distortion by `68.05%`
      (`3.13×`, `c14`), dropping in-sample KL by `−0.0691` nats to **`0.5814` nats** (`33.4× Q4_K_M`, `60.53%` top-1), and
      dropping held-out KL by **`−0.1253` nats** to **`1.8667` nats** (`94.8× Q4_K_M`, **`36.18%` held-out top-1**, wall
      `84.06` s).
    - Adding 0-bit Gauss-Seidel two-plane back-fitting (`two_plane_backfit_descent`, `c15_2plane_backfit_j_ratio_fp32 = 0.9148`,
      `−8.52%` 2-plane distortion at `0` extra bits) in `had-fold-2plane-attn-backfit` (`1.179741 ≤ 1.18` b/w) drops
      held-out KL to **`1.8580` nats** (`94.3× Q4_K_M`, **`36.22%` held-out top-1**, in-sample **`0.5929` nats**).
    - Absorbing `sign(d_j)` into `S_rot` and per-128-group `max |d|` into `r_{:,g}` (`butterfly_sign_group_gauge`:
      `((s·s_d)·(|d|/m))·(r·m) = s·(s_d·|d|)·r`, `c16_butterfly_gauge_maxabs_f64 = 1.33e-15`) eliminates the global-min/max
      `uint5` quantization bottleneck on `d` at `0` bit overhead, dropping `had-fold-2plane-attn-bscale` (`422,042,112 /
      357,826,560 = 1.179460 ≤ 1.18` b/w) to **`1.8500` nats** (`93.9× Q4_K_M`, `−0.0080` nats below `backfit`, wall
      **`110.05` s**; rank-64 hard shrinkage of `W*_tik − W_0` in `had-fold-2plane-attn-rrr` over-damps full-rank local
      compensation and reaches `1.8987` nats).
  - Column-scale group-max gauge, intra/inter decoupled split on `bscale`, and the 7-component face topology swaps
    (`Perf.CostCorrespondence` §13–§20: `column_scale_group_gauge`, `butterfly_gptq_metric_invariance`,
    `rank2_down_corrector_dominance`, `bscale_rank2_down_budget`, `split_butterfly_joint_dominance`,
    `rmsnorm_gauge_invariance`, `hyperbolic_precond_excess_identity`, `hyperbolic_precond_dominance`,
    `precond_damping_distortion`, `fnorm_fisher_variance_nonneg`, `fnorm_newton_descent`,
    `diag_precond_coherent_overshoot`, `e8_o8_codebook_cardinality`, `e8_o8_parabolic_spike_dominance`,
    `e8_o8_pair_root_dominance`, `delayed_decision_cholesky_dominance`, `beam_wscale_trajectory_dominance`,
    `e8_icm_global_descent_dominance`, `quantized_scale_icm_dominance`, `stage1_drift_shrinkage_dominance`,
    `swiglu_bilinear_comp_dominance`, `rmsnorm_pullback_sink_attenuation`, `sink_row_trust_clipping_dominance`; receipts
    `/home/socol/oc-tmp/escalation/2026-10-12-fold-transfer-attn-{rk2,bscale-split}.json` and
    `/home/socol/oc-tmp/escalation/2026-10-13-fold-transfer-attn-{hpre,fnorm,e8,e8-beam,e8-icm,e8-icm-mp,e8-icm-gin25,e8-icm-glu,e8-icm-sink}.json`,
    `sha256` `9c2d89dd9fe6b63a4c648495f4692f506a3ae450474791adf631216ec7a37998`,
    `e71f4ef08a55f28c0059a37b9c7c679a3cf3479051ad4a01ca61b9cf829bc4d1`,
    `f1a354e96238c0853621406531851b3ef815b522f2298d3b1e59197e71c2ea15`,
    `ce2af7f72bd6c3d5b011a45c8ae22268bac2815b7a30e8c0c06c92c557c4e7c9`,
    `6c431645b54e0d93f17654b6b9c2df2e9840f2b1e8cbc29d2aec9a88b167be5d`,
    `d49e3c33cd93b7083e27bf639780905131907e6b54408fccb1d7614a08daeee5`,
    `8a59211d33988e38edeacf795c154ce2c3d7f5dd0703e184a5f2c4582d6266d7`,
    `c4466168741d7067bf15ef2834dbc26558aa387fbfef42e8aa2b80af109329fe`,
    `361dd180f1c5deb86904307d3bc4e1f4ceb08089b808b1da07a3ed1f1ae36a8e`,
    `8e5a89ff8ee2dfec35fa84673ccb1d827a91a056fceeedeebdc6967214916f2a`, and
    `49fc294cba37b0d98a028f5ac015a4493ed13a80720f17568450d18b9cd73bff`, script `sha256`
    `cd755a45b688c8dd2f1bc772cbf457ebe3d5fa17292f5e025ee2c4d1d995d118`, controls
    `c17_column_scale_gauge_maxabs_f64 = 4.44e-16`, `c17_butterfly_gptq_metric_rel_err_f64 = 0.0`,
    `c17_rk2_cgauge_vs_rk1_bscale_j_ratio_fp32 = 0.7381`, `c18_cauchy_schwarz_opt_err_f64 = 1.57e-16`,
    `c18_cauchy_schwarz_vs_elliptic_ratio_f64 = 0.6693`, `c18_hpre_vs_elliptic_j_ratio_fp32 = 0.9571`,
    `c19_fnorm_newton_kl_ratio_f64 = 0.7537`, `c20_e8_codebook_size = 256`,
    `c20_e8_stored_tuple_max_dist_fp32 = 0.0`, `c20_e8_1plane_j_ratio_fp32 = 0.8532`,
    `c20_e8_2plane_j_ratio_fp32 = 0.7988`, `c21_e8_beam4_vs_e8_1plane_j_ratio_fp32 = 0.9518`,
    `c21_e8_beam4_vs_e8_2plane_j_ratio_fp32 = 0.9297`, `c22_e8_icm_vs_beam4_1plane_j_ratio_fp32 = 0.9190`,
    `c22_e8_icm_vs_beam4_2plane_j_ratio_fp32 = 0.8746`, `c23_stage1_gin25_vs_gin100_j_ratio_f64 = 0.2556`,
    `c23_stage1_gin25_vs_gin000_j_ratio_f64 = 0.7564`, `c24_swiglu_bilinear_comp_j_ratio_f64 = 0.9225`,
    `c25_rmsnorm_pullback_sink_weight_f64 = 0.0698`, `c25_rmsnorm_pullback_sink_j_ratio_f64 = 0.1358`):
    - **§13 (`bscale-split` vs `rk2-split`)**: combining two-component intra/inter shrinkage (`split_intra_inter`) with
      `had-fold-2plane-attn-bscale` (`had-fold-2plane-attn-bscale-split`, `1.179460` b/w, `fold-attn-bscale-split-1`
      **confirmed**) cuts in-sample KL by `−0.0193` nats (`0.6020 → 0.5827` nats, `33.49× Q4_K_M`, `61.05%` in-sample top-1)
      and reaches `1.84995` nats held-out (`93.91× Q4_K_M`). Trading `uint6 c, uint5 d` down to `uint4 (c, d)` to fund a
      second `down_proj` corrector rank (`had-fold-2plane-attn-rk2-split`, `1.179726` b/w, `fold-attn-rk2-1` refuted on KL)
      raises held-out top-1 to `36.27%` (`1.8531` nats) because `uint4` scale quantization loss across all 7 matrices exceeds
      the second `down_proj` corrector eigenvalue on non-outlier blocks.
    - **§14 (Face Swap #1A `hpre` & Face Swap #1B `fnorm`, both refuted and formalized)**:
      - *Face Swap #1A (`had-fold-2plane-attn-bscale-hpre`, `fold-attn-hpre-1` refuted)*: inserting a hyperbolic diagonal
        squeeze `c_0 = (w_rms / h_rms)^{1/2}` before `fwht128` reduces single-matrix calibration quadratic error by `4.29%`
        (`c18 = 0.9571`), holds in-sample KL at `0.5827` nats (`61.14%` top-1), and raises held-out KL from `1.8500` to
        `1.8843` nats (`95.65× Q4_K_M`). Formal mechanism (`precond_damping_distortion`): scaling `H_c = diag(c) H diag(c)`
        before `fwht128` transforms isotropic Cholesky ridge damping `δ I` into `δ · diag(c⁻²)` in native coordinates,
        amplifying effective damping by `1/c_j² > 1` whenever `c_j < 1`.
      - *Face Swap #1B (`had-fold-2plane-attn-bscale-fnorm`, `fold-attn-fnorm-1` refuted)*: diagonal Newton-Fisher steps on
        the final readout `RMSNorm` vector `g ∈ ℝ^896` reduce KL by `24.6%` on `M = 32` uncorrelated synthetic channels
        (`c19 = 0.7537`), but diverge on the `M = 896` correlated student readout channels (`8159.9` nats). Formal mechanism
        (`diag_precond_coherent_overshoot`): when `M` channels share a coherent logit direction, the true Fisher step along
        `1_M` scales as `1/(M u)` while the diagonal preconditioner takes step `1/u`, overshooting the Newton minimum by
        `M = 896×` (directional derivative `(M − 1) > 0` of opposite sign for all `M > 2`).
    - **§15 (Face Swap #2: 8-D `E₈ ∪ O₈` Spinor + Parabolic-Sparse Codebook vs `{±1}⁸`, `fold-attn-e8-1` confirmed)**:
      replacing the 1-D sign hypercube `{±1}⁸` inside each 128-wide Hadamard block with the exact 1-byte (`256`-entry,
      `1.0000` b/w, `0` extra bits) equal-norm `‖c‖² = 8` union of `128` even-parity spinors `E₈ = {s ∈ {±1}⁸ : ∏ s_i = +1}`,
      `112` signed 2-sparse root pairs `(±2, ±2, 0⁶) ∈ D₈`, and `16` signed 1-sparse spikes `(±√8, 0⁷)` (`e8_o8_codebook_cardinality`,
      `e8_o8_parabolic_spike_dominance`, `e8_o8_pair_root_dominance`) cuts single-matrix 1-plane quadratic error by `14.68%`
      (`c20_e8_1plane_j_ratio_fp32 = 0.8532`) and 2-plane quadratic error by **`20.12%` (`c20_e8_2plane_j_ratio_fp32 = 0.7988`)**,
      while cutting transfer wall time by **`44.6%` (`113.22` s → `62.69` s)** (16 vector-quantized 8-tuple block steps
      per 128-group instead of 128 scalar steps). Across all 24 blocks of Qwen2.5-0.5B (`had-fold-2plane-attn-bscale-e8` at
      `1.179460` b/w), in-sample KL drops by **`−0.1133` nats (`0.5827 → 0.4694` nats, `26.98× Q4_K_M`, `+3.08%` in-sample
      top-1 to `64.13%`)** and held-out KL drops by **`−0.2173` nats (`−11.03× Q4_K_M`) to `1.6327` nats (`82.88× Q4_K_M`,
      `+3.60%` held-out top-1 to `39.43%`)**.
    - **§16 (Face Swap #3: Delayed-Decision `beam=4` Block-Cholesky Trajectory Search over `E₈ ∪ O₈`, `fold-attn-e8-beam-1`
      confirmed)**: replacing 1-step greedy Babai rounding (`beam=1`) over the 16 sequential 8-tuples of each 128-column
      Hadamard group with a delayed-decision `beam=4` trajectory search (`delayed_decision_cholesky_dominance`,
      `beam_wscale_trajectory_dominance`) cuts 1-plane quadratic error by an additional `4.82%` (`c21 = 0.9518`) and 2-plane
      error by **`7.03%` (`c21 = 0.9297`, `25.73%` below `{±1}⁸`)**. Across all 24 blocks (`had-fold-2plane-attn-bscale-e8-beam`
      at `1.179460` b/w, wall `98.98` s), in-sample KL drops to **`0.4460` nats (`25.63× Q4_K_M`, `64.89%` top-1)** and
      held-out KL drops by another **`−0.0395` nats to `1.5932` nats (`80.87× Q4_K_M`, `39.89%` top-1, `+4.06%` over
      `bscale-split`)**. Expanding the beam to `K = 8` (`had-fold-2plane-attn-bscale-e8-beam8`) overfits the unregularized
      within-group Cholesky tail (`0.4584` in-sample, `1.6119` held-out), bracketing `K = 4` as optimal.
    - **§17 (Face Swap #4: Marchenko-Pastur Damped Global Elliptic 8-Tuple ICM over `E₈ ∪ O₈`, `fold-attn-e8-icm-1` refuted
      at `λ_icm = 0`, `fold-attn-e8-icm-mp-1` confirmed at `λ_icm = 0.20`)**: replacing the one-way causal Cholesky freeze of
      `S_rot` with bidirectional 8-tuple Iterated Conditional Modes (ICM) coordinate descent over the full `n × n`
      `(r, c, d)`-scaled Hadamard-Gram metric (`e8_icm_global_descent_dominance`, `quantized_scale_icm_dominance`) cuts
      1-plane quadratic distortion by `8.10%` (`c22 = 0.9190`) and 2-plane distortion by **`12.54%` (`c22 = 0.8746`)** over
      `beam=4`. Undamped (`λ_icm = 0`, `had-fold-2plane-attn-bscale-e8-icm`), full-`n × n` cross-group coupling chases
      finite-sample Marchenko-Pastur noise in `H_cal`, inflating Euclidean weight error `‖W_0 − Ŵ‖_F²` by `+7.4%` (`1.6096`
      nats held-out). Adding Marchenko-Pastur isotropic damping `λ_icm · (n/896) d̄ I` with `λ_icm = 0.20`
      (`had-fold-2plane-attn-bscale-e8-icm20` at **`1.179460` b/w**, wall `113.77` s) reduces calibration MSE, held-out MSE
      (`−8.03%`), and Euclidean weight error (`−7.23%`), driving in-sample KL down to **`0.4269` nats (`24.53× Q4_K_M`,
      `66.06%` top-1)** and held-out KL down by **`−0.0716` nats (`−0.3284` nats / `−16.67× Q4_K_M` below `bscale-split`)
      to `1.5216` nats (`77.24× Q4_K_M`, `40.72%` held-out top-1)**.
    - **§18 (Face Swap #5: Stage-1 Expansion Drift Attenuation `inter_gamma_in = 0.25`, `fold-attn-e8-icm-gin25-1` confirmed)**:
      Stage 1 (`q, k, v, gate, up`) maps `896` input dimensions to `11,776` output rows per block (`9,728` in `gate+up` feeding
      nonlinear `SiLU(gate) * up`), while Stage 2 (`o, down`) evaluates its intra-sublayer reference `C_ts` at `W_{0,in}` and
      separately absorbs residual drift on the `896`-wide bottleneck. Leaving Stage 1 unshrunk (`inter_gamma_in = 1.0`) both
      overfits `11,776 × 896` parameters to finite-sample input drift and double-counts that drift in Stage 2's `C_intra = C_ts − C_0`,
      whereas zeroing Stage 1 alignment (`inter_gamma_in = 0.0`) loses first-order alignment of the nonlinear `softmax`/`SiLU`
      gates (`1.8142` nats on `e8`). Attenuating Stage-1 cross-layer drift to `inter_gamma_in = 0.25` (`stage1_drift_shrinkage_dominance`,
      `c23_stage1_gin25_vs_gin100_j_ratio_f64 = 0.2556`, `c23_stage1_gin25_vs_gin000_j_ratio_f64 = 0.7564`) cuts held-out KL on
      `had-fold-2plane-attn-bscale-e8-icm20-gin25` (`1.179460` b/w, wall `108.43` s) by another **`−0.0242` nats (`−0.3526` nats
      / `−17.90× Q4_K_M` below `bscale-split`) to `1.4974` nats (`76.01× Q4_K_M`, breaking the `1.50` nats barrier!)** and raises
      held-out top-1 agreement by **`+0.69%` to `41.41%` (`+5.58%` over `bscale-split`)**, with in-sample KL at **`0.4296` nats
      (`24.69× Q4_K_M`, `65.69%` top-1)** (and `λ_icm = 0.25` variant `had-fold-2plane-attn-bscale-e8-icm25-gin25` also breaking
      `1.50` nats at **`1.4997` nats / `76.13× Q4_K_M`, `41.04%` top-1**).
    - **§19 (Face Swap #6: Hyperbolic Bilinear SwiGLU Compensation `gate → up`, `fold-attn-e8-icm-glu-1` refuted)**:
      shifting the Stage-1 `up_proj` ridge target after `gate_proj` quantization to compensate for the bilinear error
      `(SiLU(x W_g^T) − SiLU(x Ŵ_g^T)) ⊙ (x W_u^T)` (`swiglu_bilinear_comp_dominance`, `c24_swiglu_bilinear_comp_j_ratio_f64 = 0.9225`)
      beats `icm20` (`1.5216` nats) on both `had-fold-2plane-attn-bscale-e8-icm20-gin3515-glu` (`1.5015` nats / `76.22× Q4_K_M`)
      and `had-fold-2plane-attn-bscale-e8-icm20-gin25-glu` (`1.5030` nats / `76.29× Q4_K_M`), but trails unshifted `gin25`
      (`1.4974` nats) by `+0.0042` nats because shifting the 1-plane `up_proj` target prior to quantization increases its
      1-bit quantization residual while double-counting the intra-sublayer SwiGLU compensation already performed by Stage-2
      `down_proj` `C_intra`.
    - **§20 (Face Swap #7: Riemannian `RMSNorm` Pullback Sink Token Gating + Row-Trust Radial Projection + Coherence-Anchored
      Readout `fnorm`, `fold-attn-e8-icm-sink-1` refuted on `< 35` s wall and `< 1.41` sink-only clause, confirmed on `< 1.38` nats
      held-out and `< 0.3887` nats in-sample floor)**:
      in Qwen2.5-0.5B, initial/BOS attention-sink tokens reach residual norm `r_sink ≈ 1720` (`14.3×` the non-sink median
      `r̄ ≈ 120`), so unweighted Euclidean residual regression `(XᵀX)⁻¹ Xᵀ Y` gives a single sink token `(r_sink / r̄)² > 200×`
      leverage in `XᵀX` while the downstream `RMSNorm` Jacobian `∂ RMSNorm(h)/∂h ∝ 1/‖h‖_2` attenuates perturbations on that
      token by `w_t = min(1, r_med / (5 r_t))` for `r_t > 5 r_med` (`rmsnorm_pullback_sink_attenuation`,
      `c25_rmsnorm_pullback_sink_weight_f64 = 0.0698`, `c25_rmsnorm_pullback_sink_j_ratio_f64 = 0.1358`). Weighting the ridge
      target solve `(H_w, C_intra,w, C_inter,w)` by `w_t` while keeping unweighted `H` for the quantizer metric and bounding
      per-row drift by `‖dW_i‖_2 ≤ 1.5 ‖W_{0,i}‖_2` (`sink_row_trust_clipping_dominance`) cuts 24-block held-out KL on
      `had-fold-2plane-attn-bscale-e8-icm20-gin25-sink` (`1.179460` b/w, wall `88.14` s) by **`−0.0767` nats to `1.4206` nats
      (`72.11× Q4_K_M`, `42.12%` held-out top-1)** and in-sample KL to **`0.3995` nats (`22.96× Q4_K_M`, `66.46%` top-1)**.
      Adding 2 damped Newton-Fisher steps (`fnorm_lr = 0.15`, `ridge = 0.10`) on the `0`-bit final readout `RMSNorm` vector
      `g ∈ ℝ^896` anchored to the non-sink cosine-coherence target `g_target = g_0 · (ρ / max(0.5, α))`
      (`had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm` at **`1.179460` b/w**, wall `93.91` s; receipt
      `/home/socol/oc-tmp/escalation/2026-10-13-fold-transfer-attn-e8-icm-sink.json`, `sha256` `49fc294c…`) drives in-sample KL
      down to **`0.3749` nats (`21.55× Q4_K_M`, `0.9646×` — `3.54%` BELOW the `0.3887` nats uniform-rate separable K-FAC floor!,
      `66.31%` top-1)** and held-out KL down by another **`−0.0429` nats (`−0.1197` nats below `gin25`, `−0.4722` nats /
      `−23.97× Q4_K_M` below `bscale-split`) to `1.3777` nats (`69.93× Q4_K_M`, breaking `< 1.38` nats and `< 70× Q4_K_M`!,
      `42.19%` held-out top-1)**. Vectorizing `e8_icm_step` (`7.2×` speedup, `max_diff = 0`), caching `(U, Z_c)` in `gptq_e8`
      across inner iterations and planes (`solve_triangular` preserved), and keeping `Pt_cache` and `kl_eval` on GPU cut
      per-variant Gram time `3.14×` (`36.1 → 11.5` s), forward time `2.53×` (`13.2 → 5.2` s), and total transfer wall from
      `119.1–123.9` s to **`88.1–93.9` s**.
- Exact 4-factor gap decomposition to `Q4_K_M` (`0.0197` nats on 0.5B, `four_factor_gap_identity` and
  `w15_four_factor_05b_decomposition` in `Perf.CostCorrespondence` §12), updated to
  `had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm` (`1.3777` nats held-out, `0.3749` nats in-sample):
  `KL_out / KL_Q4 = (KL_out / KL_in) · (KL_in / D_sep) · (D_sep / D_orc) · (D_orc / KL_Q4)`:
  1. **Finite-sample generalization (`KL_out / KL_in = 3.675×`, `E_gen = 1.0028` nats = `72.8%` of total held-out KL, down from `1.0678` nats at `gin25` and `1.0947` nats at `bscale-split`)**:
     driven by fitting `14.9M` dense target parameters per layer to `51,200` tokens (`n / N_tok ≈ 9.5%` on `4864`-wide `ffn_down`).
  2. **Constructive coder efficiency (`KL_in / D_sep = 1.322×` vs non-uniform `0.2837` nats, and **`0.9646×` — `3.54%` BELOW**
     the uniform-rate separable floor `0.3887` nats)**: Face Swaps #2–#7 (`E₈ ∪ O₈` + `beam=4` + MP-damped `icm20` + `gin25` +
     `sink-fnorm`) drive `KL_in` down to **`0.3749` nats** (`21.55× Q4_K_M`, down from `0.5929` nats at `backfit`).
  3. **Separable-to-oracle envelope gap (`D_sep / D_orc = 2.515×`)**: `0.2837 / 0.1128` nats (`χ²_1` per-coefficient sparsity
     `ΔR* ≈ 0.66–0.82` b/w).
  4. **Intrinsic 0.5B oracle floor (`D_orc / KL_Q4 = 5.726×` on 0.5B → `2.11×` on 8B → `~1.35×` on 27B/32B)**: small models
     have high stiff-direction density (`R*_sep = 3.1731` b/w on 0.5B vs `2.2530` on 8B and `1.9927` on 32B, where the
     separable floor at `1.18` b/w is already `3.78× Q4_K_M` = `0.0913` nats, below 4.5-bit `NF4`'s `4.62× Q4_K_M`).

21. **Best arm over three calibration-order seeds**
    (`had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp`, `1.178061` b/w; grade `measured` for the KL
    and top-1 columns, `proved` for the Lean verdict over the measured constants as `model` inputs).
    - Command (Qwen2.5-0.5B, 24 layers, 100 calibration / 40 held-out / 20 in-sample chunks, `fp32`, one GPU, under
      `flock /tmp/opencode/gpu.lock`, wall 443.4 s total including model load):
      `flock /tmp/opencode/gpu.lock timeout 580 /home/socol/Workspace/spinor-ttl/.venv/bin/python tools/onebit/fold_transfer.py --out /home/socol/oc-tmp/w16/t0-best-3seed.json --seeds 0,1,2 --variants had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp`.
    - `tools/onebit/fold_transfer.py` `sha256` `ce1734018f202aa8aa6bace3aad4858cc09b2f0d31d2dc89b2397179bb112309`
      (`--seeds`: seed `s ≠ 0` permutes the calibration chunks with `torch.randperm(n_cal, generator=Generator().manual_seed(s))`
      and seeds `torch.manual_seed(s)`; seed `0` is the identity order; held-out and in-sample chunks never change). Receipt
      `/home/socol/oc-tmp/w16/t0-best-3seed.json`, `sha256` `3715d9b7427dca22f450ff21583e41a5d6bc6d6ba80fdc3544a089ad5fd2b10e`
      (keys `seeds`, `kl_heldout_stats`, `top1_heldout_stats`, `kl_insample_stats`; the top-level keys are seed 0).
    - Results (`bits = 421,541,376`, `params = 357,826,560`, `1.178061` b/w on every seed, peak GPU `4.995 / 4.996 / 4.995` GB):

      | seed | held-out KL (nats) | held-out top-1 | in-sample KL (nats) | in-sample top-1 | wall (s) |
      |---|---|---|---|---|---|
      | 0 | 1.2798 | 43.76% | 0.3474 | 67.81% | 105.6 |
      | 1 | 1.3894 | 41.91% | 0.3686 | 67.02% | 131.8 |
      | 2 | 1.3896 | 42.15% | 0.3682 | 67.07% | 189.0 |
      | mean (min, max) | **1.3529** (1.2798, 1.3896) | **42.61%** (41.91, 43.76) | **0.3614** (0.3474, 0.3686) | — | — |

      Seed `0` reproduces the earlier single run (`1.2798` nats, `43.76%`, `0.3474` nats, `67.81%`): held-out KL differs by
      `3e-5` nats. The wall of seeds `1` and `2` includes contention from other jobs on the machine; it is not a
      throughput figure. Seed `0` is the best of the three by `0.110` nats; seeds `1` and `2` agree to `2e-4` nats.
    - What the three suffixes do (`tools/onebit/fold_transfer.py`):
      - `-grow` (`use_grow`, `:839`; `:1033-1042`): the scale solve inside `fold_fit` weights output rows by `g_row`
        (`:605`). For `gate`/`up`, `g_row` is the column energy `Σ_j W_down[j, i]²` of the following `down` (mean-normalized);
        for `o`/`down`, `g_row` is the squared RMSNorm gain of the next consumer (`ln2` for `o`, the next layer's `ln1` or
        the final norm for `down`), mean-normalized.
      - `-res` (`use_res`, `:840`; `:1135-1158`): after `up` is quantized, solve for one scale `s_i ∈ [0.5, 2]` per
        intermediate row of `P = silu(X W_gate^T) ⊙ (X W_up^T)`: `(G ∘ P_qᵀP_q + λI) s = (G ∘ P_qᵀP_0) 1 + λ 1` with
        `G = W_downᵀ W_down` (Hadamard product `∘`) and `λ = 0.05 · mean diag(G ∘ P_qᵀP_q)`; then `Wp ← diag(s) Wp`
        for `up`. The scales are fitted against the teacher product `P_0` on the calibration slabs, so they cost no bits in
        the bit count above.
      - `-qmlp` (`q_mlp_trans`, `:823`; `:1065-1096`): second plane (the same `E₈ ∪ O₈` code, fitted on the plane-1 residual, not a ±1 sign plane) on `k`, `v`, `o` (`two_plane_nm = ("k","v","o")`,
        `q` stays 1-plane) and on the top-`416` rows of `gate` and `up` ranked by row mass `Σ_j W[i, j]²`
        (`topk(row_mass, 416)`; `Wp[top_idx] += fold_fit(W_s[top] − Wp1[top])`); the rank-2 correction targets are
        `("q","down")` (`:1097-1099`). The extra planes are in the `1.178061` b/w count.
    - `Perf.CostCorrespondence` theorems added with this arm's work (`proof/corpus/Perf/CostCorrespondence.lean`, all
      `#audit`ed in `proof/corpus/AxiomAudit.lean`, closed cone):
      - `swiglu_hyperbolic_noise_minimization (u d s : ℝ)`, hypotheses `0 < u`, `0 < d`, `0 < s`: conclusion
        `(su)² + (s⁻¹d)² − 2ud = (su − s⁻¹d)²`, `2ud ≤ (su)² + (s⁻¹d)²`, and equality iff `su = s⁻¹d`. Algebraic model of
        inter-leg equalization along the invariant hyperbola `(su)(s⁻¹d) = ud`. **Not used by the arm**: no code path sets
        `s = √(d/u)`; `-res` rescales the product `P`, not the two legs.
      - `coupled_schur_crosstalk_dominance {a b c r2 : ℝ}`, hypotheses `0 < a`, `0 < b`, `c ≠ 0`, `0 < ab − c²`:
        for `r2 ≠ 0`, the coupled optimum `(s₁, s₂) = (−c·r2, a·r2)/(ab − c²)` has strictly lower
        `J = a s₁² + b s₂² + 2c s₁s₂ − 2 r2 s₂` than the decoupled choice `(0, r2/b)`; the gap is `c² r2² / (b(ab − c²))`.
        2-fibre model of a coupled solve; `-res` solves the same kind of coupled system at size `Oi × Oi` (`G ∘ P` matrix)
        but the theorem is not instantiated at that size, so the arm is **motivated by, not proved by**, this theorem.
      - `asymmetric_wiener_target_optimality (p c y w : ℝ)`, hypothesis `0 < p`: with `w* = c/p` and
        `J(v) = p v² − 2cv + y`, `J(w) − J(w*) = p (w − w*)²`, `J(w*) ≤ J(w)`, and `J(w) = J(w*) ↔ w = w*`. **Not used by the
        arm**: the asymmetric target is the `-asym` variant (`use_asym`, `:842`, `:993-999`), which is not in this arm's name.
      - Also uncommitted before this section and audited with it: `q4k_vs_1bit_transfer_floor` (`4.5 / 1.18 = 225/59 > 3.8`
        and `1.18 / 4.5 < 0.27`, bytes-per-weight ratio against Q4_K_M, `norm_num`) and `asymmetric_distortion_penalty_strict`
        (`ρ < 1`, `w₀ ≠ 0`: excess `p(1 − ρ)² w₀² > 0`).
    - Verdict `w16_t0_best_arm_3seed` (`Perf.CostCorrespondence` §25; inputs `model`, 3-seed receipt above): (a)
      `421541376/357826560 ≤ 1.18`; (b) mean held-out KL `(1.2798 + 1.3894 + 1.3896)/3 < 1.3777`; (c) the mean is `> 1.3529`
      and `> 0.0197` (Q4_K_M not reached: refuted-target clause, the mean is `68.7× Q4_K_M`'s `0.0197`); (d) the seed spread
      is `< 0.12` nats.
    - Comparison to §20 (`…-sink-fnorm`, `1.179460` b/w, `1.3777` nats, `42.19%`, `0.3749` nats in-sample, one run =
      seed `0`): seed `0` against seed `0` is `−0.0979` nats held-out (`1.3777 → 1.2798`), `+1.57` top-1 points, and
      `−0.0275` nats in-sample; the 3-seed mean is `−0.0248` nats held-out and `−0.0135` nats in-sample against that
      single run, at `−0.001399` b/w. Seeds `1` and `2` are each `+0.012` nats above the §20 seed-`0` value. The §20 arm was
      not re-run on seeds `1` and `2`, so the arm-to-arm gap at matched seeds is measured on seed `0` only.
    - Board row: `onebit-w16-t0-receipt-for-CEEkDLWOB0s8inHVh8q2Aa`.

## Not covered (W15d)

- Constructive transfer evaluated on 0.5B only; `src/learn/ridge.rs` and `src/learn/onebit.rs` remain `f64` on CPU.

Board row: `onebit-exact-one-bit-and-FubKIC3prjKrgnryWWxrF1` (LEARN-ONEBIT, `docs/plan.md` row 248).
