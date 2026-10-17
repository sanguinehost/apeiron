# ONEBIT W15a: direct KL along the quantisation ray

Command (from the repository root, `/home/socol/Workspace/spinor-ttl/.venv/bin/python`, RTX 5080):
`python tools/onebit/ray_scan.py /home/socol/oc-tmp/escalation/2026-10-11-stiffness.json /home/socol/oc-tmp/escalation/2026-10-11-ray.json 100`.
100 C4 chunks of 512 tokens (`/home/socol/oc-tmp/seal/calib-c4.txt`), wall 139.6 s. The 21 `ffn_down` tensors of
Qwen2.5-0.5B move jointly to `W + tD` (`D` the arm's dequantised error, `fisher_capture.load_deltas`), and the
per-token KL `KL(p_W ‖ p_{W+tD})` is computed in float64.

- Source: `tools/onebit/ray_scan.py` (removed; at 2f15d6a9) sha256 `ad65df8f0cc58c2d…`; receipt sha256 `40e693d217fdbe6c…`.
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

- The water-filling floor (`waterfill_floor`, `conditional` on the exponential distortion model) applies to one-bit error levels on `ffn_down`: the quadratic form is the
  objective to within 15%.
- The sampled-Fisher estimator is too noisy to rank arms. The Kronecker factorisation `G ⊗ H` (`kron_diag`) is the
  next instrument, validated against `64·KL(1/8)` above before any `model` floor is read from it.

## Not covered

- `ffn_down` only, one model (Qwen2.5-0.5B), C4 calibration text; P1 KLD is wikitext-2 through llama.cpp.
- `t ∈ {±1/8, 1/4, ±1/2, ±1}`; the remainder between those points is not bounded.

# ONEBIT W15b: K-FAC water level over every linear tensor

Command (repository root, same venv and GPU): `python tools/onebit/kfac_floor.py
/home/socol/oc-tmp/escalation/2026-10-11-kfac.json 100 4`. 100 C4 chunks, 4 sampled targets per token, all 168 linear
tensors (7 kinds × 24 blocks). Wall 118.3 s (capture 22.0 + 17.8 s in two 12-block halves, analysis 45.0 s).

- Source: `tools/onebit/kfac_floor.py` (removed; at 2f15d6a9) sha256 `1fb742d7080c0417…`; receipt sha256 `e64115b13698bf50…`; log
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
- Floors (`model`): in the eigenbases `G = U diag(g) Uᵀ`, `H = V diag(h) Vᵀ`, the coefficients `C = UᵀWV` carry weights
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

## Floor (`model`: K-FAC quadratic KL, Gaussian rate)

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

`Perf.CostCorrespondence.kfac_w15b` (`proved` at 2f15d6a9; the module was removed by DEL-RECEIPT-LEAN, so the
statement now stands at `model`) states the validation band, the refuted order, the 16×/6× floor factors and the matching rates as ℚ facts. Controls: the order flipped and the separable factor raised to
17× each redden it (`/home/socol/oc-tmp/costcorr-kfac-controls.log`).

## What follows

- On Qwen2.5-0.5B, post-training coding in the K-FAC metric needs ≥ 2.35 b/w (oracle) to reach Q4_K_M's KL; 1.18 b/w
  is 6.6× above Q4 even there (`model`).
- The open question for the 27B target is how the matching rate scales with model size. The next measurement is the
  same `model` floor on larger Qwen2.5 checkpoints, registered before it runs.

## Not covered (W15b)

- One model, 0.5B; embeddings and `lm_head` excluded; cross-token terms and `g`–`x` dependence dropped.
- Gaussian-codebook rate model; basis side information uncounted; the oracle is not achievable.
- C4 calibration; P1 KLD is wikitext-2 through llama.cpp.

# ONEBIT W15c: the Shannon-bound side (`anchor`) and the K-FAC floor (`model`) against model size

Command (repository root, same venv and GPU): `python tools/onebit/converse.py {controls,main,ekfac,scale} …`.
Source `tools/onebit/converse.py` (removed; at 2f15d6a9) sha256 `29034f02d9002f6f…`; receipts
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

## Variance ladder on Qwen2.5-0.5B (`tools/onebit/variance_ladder.py` (removed; at 2f15d6a9))

Receipt: `/home/socol/oc-tmp/escalation/2026-10-11-vladder.json` (138.6 s, fp32 tensors). Prediction
`variance-ladder-1-1` **refuted** (falsifier fired: no structured variance model recovers `≥ 0.1` b/w net of side
information).

| variance model | `R*` coeff-only (b/w) | `R*` all-in (b/w) | floor KL at 1.133 b/w all-in |
|---|---:|---:|---:|
| separable (`r_i c_j / m`, control) | 3.151 | **3.176** | 0.2837 (uniform-rate: 0.3887, 1.37×; `refuted` as a floor by the measured in-sample KL `0.35064` (`t3-nb-3seed.json`)) |
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
2. **Full-layer streamed capture (`tools/onebit/rstar_stream.py` (removed; at 2f15d6a9), sha256 `ba392c24…`)**:
   - Control `rstar-ctl-1` (0.5B, 24 layers fp32): bisection `R*` all-in **3.1731** (ref 3.1743, diff `0.0012`), `K0`
     1.1852 (`−0.47%`), `kl_q4` 0.015700, wall 109 s (HF logprob clause `5e-4` vs `< 1e-4` missed).
   - Mutant `rstar-mut-1` (`--dscale 2`, scaling Q4 error `×2`, KL `×4`): bisection `R*` all-in **2.1823** (move
     `−0.991` b/w, high-res `−1.000` b/w), confirming `waterfill_exclusion` (`4^{−ΔR}`).
   - Full-model Qwen3-8B (36 layers fp32, receipt `/home/socol/oc-tmp/escalation/2026-10-11-rstar-qwen3-8b.json`,
     `rstar-8b-1` confirmed) and Qwen3-32B (64 layers fp32, receipt
     `/home/socol/oc-tmp/escalation/2026-10-11-rstar-qwen3-32b.json`, `rstar-32b-1` refuted against `[0.9, 1.9]`).
3. **Fast one-process instrument (`tools/onebit/rstar_fast.py` (removed; at 2f15d6a9), sha256 `96031c3d…`, prior SLQ sha256 `fef86981…`)**:
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
  within-tensor K-FAC rate allocation cannot reach 1.18 b/w even with oracle variances (`model`: K-FAC quadratic KL,
  Gaussian rate).

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
    uniform-rate separable `model` floor `0.3887`, since `refuted` as a floor by the measured in-sample KL `0.35064` (`t3-nb-3seed.json`)) while raising held-out KL to `2.0791`; increasing the prior factor to `0.04`
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
      / `−17.90× Q4_K_M` below `bscale-split`) to `1.4974` nats (`76.01× Q4_K_M`, below `1.50` nats)** and raises
      held-out top-1 agreement by **`+0.69%` to `41.41%` (`+5.58%` over `bscale-split`)**, with in-sample KL at **`0.4296` nats
      (`24.69× Q4_K_M`, `65.69%` top-1)** (and `λ_icm = 0.25` variant `had-fold-2plane-attn-bscale-e8-icm25-gin25` also below
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
      down to **`0.3749` nats (`21.55× Q4_K_M`, `0.9646×` — `3.54%` below the `0.3887` nats uniform-rate separable K-FAC `model` floor, `refuted` as a floor by the measured in-sample KL `0.35064` (`t3-nb-3seed.json`),
      `66.31%` top-1)** and held-out KL down by another **`−0.0429` nats (`−0.1197` nats below `gin25`, `−0.4722` nats /
      `−23.97× Q4_K_M` below `bscale-split`) to `1.3777` nats (`69.93× Q4_K_M`, below `1.38` nats and `70× Q4_K_M`,
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
  2. **Constructive coder efficiency (`KL_in / D_sep = 1.322×` vs non-uniform `0.2837` nats, and **`0.9646×` — `3.54%` below**
     the uniform-rate separable `model` floor `0.3887` nats, `refuted` as a floor by the measured in-sample KL `0.35064` (`t3-nb-3seed.json`))**: Face Swaps #2–#7 (`E₈ ∪ O₈` + `beam=4` + MP-damped `icm20` + `gin25` +
     `sink-fnorm`) drive `KL_in` down to **`0.3749` nats** (`21.55× Q4_K_M`, down from `0.5929` nats at `backfit`).
  3. **Separable-to-oracle envelope gap (`D_sep / D_orc = 2.515×`)**: `0.2837 / 0.1128` nats (`χ²_1` per-coefficient sparsity
     `ΔR* ≈ 0.66–0.82` b/w).
  4. **0.5B oracle K-FAC floor (`model`, `D_orc / KL_Q4 = 5.726×` on 0.5B → `2.11×` on 8B → `~1.35×` on 27B/32B)**: small models
     have high stiff-direction density (`R*_sep = 3.1731` b/w on 0.5B vs `2.2530` on 8B and `1.9927` on 32B, where the
     separable floor at `1.18` b/w is already `3.78× Q4_K_M` = `0.0913` nats, below 4.5-bit `NF4`'s `4.62× Q4_K_M`).

21. **Best arm over three calibration-order seeds**
    (`had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp`, `1.178061` b/w; grade `measured` for the KL
    and top-1 columns, `proved` for the Lean verdict over the measured constants as `model` inputs).
    - Command (Qwen2.5-0.5B, 24 layers, 100 calibration / 40 held-out / 20 in-sample chunks, `fp32`, one GPU, under
      `flock /tmp/opencode/gpu.lock`, wall 443.4 s total including model load):
      `flock /tmp/opencode/gpu.lock timeout 580 /home/socol/Workspace/spinor-ttl/.venv/bin/python tools/onebit/fold_transfer.py --out /home/socol/oc-tmp/w16/t0-best-3seed.json --seeds 0,1,2 --variants had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp`.
    - `tools/onebit/fold_transfer.py` (removed; at 2f15d6a9) `sha256` `ce1734018f202aa8aa6bace3aad4858cc09b2f0d31d2dc89b2397179bb112309`
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
    - What the three suffixes do (`tools/onebit/fold_transfer.py` (removed; at 2f15d6a9)):
      - `-grow` (`use_grow`, `:839`; `:1033-1042`): the scale solve inside `fold_fit` weights output rows by `g_row`
        (`:605`). For `gate`/`up`, `g_row` is the column energy `Σ_j W_down[j, i]²` of the following `down` (mean-normalized);
        for `o`/`down`, `g_row` is the squared RMSNorm gain of the next consumer (`ln2` for `o`, the next layer's `ln1` or
        the final norm for `down`), mean-normalized.
      - `-res` (`use_res`, `:840`; `:1135-1158`): after `up` is quantized, solve for one scale `s_i ∈ [0.5, 2]` per
        intermediate row of `P = silu(X W_gate^T) ⊙ (X W_up^T)`: `(G ∘ P_qᵀP_q + λI) s = (G ∘ P_qᵀP_0) 1 + λ 1` with
        `G = W_downᵀ W_down` (Hadamard product `∘`) and `λ = 0.05 · mean diag(G ∘ P_qᵀP_q)`; then `Wp ← diag(s) Wp`
        for `up`. The scales are fitted against the teacher product `P_0` on the calibration slabs. In `-res` they are
        continuous and applied after packing, so they are not on the `quant_r_hier` grid and the bit count above does
        not cover them (see "Row multipliers on the stored scale"); `-resq` folds them into the stored scale.
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

22. **Face-typed 2-plane code in the canonical algebra** (W16a; grade `proved`; board row
    `onebit-w16-t1-face-typed-HKnziwmCmuvo0tAP682tNM`). `proof/corpus/Perf/FaceTyped.lean` (sha256 `9d09ab5c…`, namespace
    `Perf.FaceTyped`; root: `upper`/`lower` of `Apeiron.Trichotomy.NullBasis` at `σ = 1`). Gate:
    `cd proof/corpus && lake build Perf.FaceTyped AxiomAudit` (exit 0, 9173 jobs; all eight audits
    `[propext, Classical.choice, Quot.sound]`).
    - `two_plane_split_complex`, `two_plane_null_channel`, `two_plane_value_mem`: `r₁s₁ + r₂s₂ = s₁·(upper 1 (r₁,r₂))` if
      `s₁s₂ = 1`, `s₁·(lower 1 (r₁,r₂))` otherwise, so every weight lies in `{±(r₁+r₂), ±(r₁−r₂)}`; `null_channels_eq`.
    - `null_basis_rate_eq`: `(s₁,s₂) ↦ (s₁, s₁s₂)` is an involutive bijection on `ℤˣ × ℤˣ`: the null-basis code has the
      2-plane code's bits.
    - `null_basis_diag_cross_zero`: the diagonal of `H` contributes no `a·b` term (`crossTerm H t m p p = 0`); `coefC_symm`.
    - `bilinear_junction_hyperbolic`: the product form `xy` has discriminant `1/4 > 0` and the gauge `(x,y) ↦ (sx, y/s)`
      preserves it.
    - Control: `j² = −1` in `two_plane_split_complex` is red (`Control.lean:45:36 unsolved goals ⊢ 1 = -1`).

23. **Joint null-basis fit against sequential fit, sign level** (W16a; grade `measured`; board row
    `onebit-w16-t2-junction-vs-Cg2BlFbboy15bMGMhVyNxP`). Command (CPU, float64, wall `261.5 s`):
    `PREDICTION=<id> timeout 580 /home/socol/Workspace/spinor-ttl/.venv/bin/python tools/onebit/face_probe.py --out /home/socol/oc-tmp/w16/t2-face-probe.json`
    (`face_probe.py` sha256 `87a2eb09…`; receipt sha256 `e6f19708…`). Layers `0,5,11,17,23`, Hadamard-rotated basis,
    `H` unridged, `J/J0` = curvature-weighted error over signal energy, ±1 sign planes on the original weights (the
    best arm's planes are `E₈ ∪ O₈` codes, so this bounds the mechanism, not the arm).

    | kind | SEQ | joint-cold | joint-warm |
    |---|---|---|---|
    | k | 0.0195 | 0.0162 (0.830×) | 0.0162 (0.828×) |
    | v | 0.0288 | 0.0248 (0.861×) | 0.0247 (0.858×) |
    | o | 0.0187 | 0.0159 (0.849×) | 0.0156 (0.832×) |
    | gate | 0.0115 | 0.0088 (0.768×) | 0.0088 (0.762×) |
    | up | 0.0252 | 0.0201 (0.797×) | 0.0200 (0.792×) |

    - Diagonal share of the plane cross term: k `0.435`, v `0.447`, o `0.507`, gate `0.444`, up `0.517`.
    - Junction imbalance `ρ`: up/down median `≈ 0.91–0.99` (max/min up to `7.9` at layer 23); v/o median `2.08–3.16`.
    - Self-checks (a)–(e) run first; the flipped-sign copy (`/home/socol/oc-tmp/lanes/w16/t2/face_probe.py`) is red at (e)
      (random-start descent), exit 1.
    - Decision: `docs/epics/ternary-hadamard.md` W16 opens W16b when joint `J` is below sequential+backfit `J` on the
      2-plane tensors; it is (`0.77–0.86×`). In `J0` units the gap is `0.0035`; relative to SEQ it is `14–24%`.

24. **Matched-seed baseline for the best arm** (W16a; grade `measured` for the KL columns, `proved` for the Lean verdict over
    them as `model` inputs; board row `onebit-w16-t0-receipt-for-CEEkDLWOB0s8inHVh8q2Aa`). Command (wall `311.5 s`, one
    GPU lane):
    `PREDICTION=w16-matched-seed-baseline-1 flock /tmp/opencode/gpu.lock timeout 580 /home/socol/Workspace/spinor-ttl/.venv/bin/python tools/onebit/fold_transfer.py --out /home/socol/oc-tmp/w16/t0b-sinkfnorm-3seed.json --seeds 0,1,2 --variants had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm`
    (script sha256 `ce173401…`; receipt sha256 `b26980d1…`).

    | seed | `…-sink-fnorm` (`1.179460` b/w) | best arm (`1.178061` b/w) | best − base |
    |---|---|---|---|
    | 0 | 1.3777 (42.19%) | 1.2798 (43.76%) | −0.0979 |
    | 1 | 1.3705 (42.15%) | 1.3894 (41.91%) | +0.0189 |
    | 2 | 1.4383 (40.62%) | 1.3896 (42.15%) | −0.0487 |
    | mean | 1.3955 (41.65%) | 1.3529 (42.61%) | −0.0426 |

    - Seed `0` reproduces the §20 value `1.3777`. The base arm's seed-to-seed spread is `0.068` nats (min `1.3705`, max
      `1.4383`), larger than the best arm's expected `−0.001` to `−0.015` nats from the joint `-nb` move (W16b design verdict
      in `docs/epics/ternary-hadamard.md`).
    - The earlier hypothesis that identity calibration order makes seed `0` low does not hold for the base arm: its seed `1`
      is `0.007` below seed `0`. The best arm's seed-`0` margin (`−0.0979`) is the largest paired difference; the paired mean
      is `−0.0426`, with one seed (`1`) positive.
    - Consequence: a single 3-seed KL mean cannot resolve a `−0.015` nat effect. The W16b mechanism gate is seed-0
      `J_nb/J_seq` on `k`, `v`, `o`; the KL stage is read as paired per-seed differences against the best-arm column above.
    - Lean: `Perf.CostCorrespondence.w16_t0b_matched_seed_baseline` (audited; closed cone).
    - Prediction `w16-matched-seed-baseline-1` recorded refuted: seeds `1` and `2` are `−0.007` and `+0.061` from seed `0`, not
      `+0.11`.

25. **`-nb` joint block-ICM arm: paired per-seed gain over the best W15d arm** (W16b; grade `measured` for the KL and `J` columns,
    `proved` for the Lean verdict over them as `model` inputs; board row `onebit-w16-t3-nb-null-2-F692bk4KVDww0Z4wqmWlQe`).
    Arm: `had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp-nb` at `1.178061` b/w (bits `421541376`, unchanged).
    Mechanism: after the backfit, on `k`, `v`, `o` only, a warm exact joint ICM over the `256 × 256` pair set per 8-block in the
    coupled block metric (`e8_joint_icm_2plane`), a joint `(r₁, r₂)` solve (`r_step2`), `c_step` per plane, `dh` frozen, ≤ 2
    rounds, a round kept only if `J` falls (`nb_refine`). Spec: Opus verdict `/home/socol/oc-tmp/escalation/2026-10-14-w16-e8-lift.md`.
    Script sha256 `a7351081…` (start `ce173401…`).

    Command (wall `406.9 s` for three seeds, one GPU lane; receipt sha256 `228765ed…`):
    `PREDICTION=w16-t3-nb-3seed-1 flock /tmp/opencode/gpu.lock timeout 580 /home/socol/Workspace/spinor-ttl/.venv/bin/python tools/onebit/fold_transfer.py --out /home/socol/oc-tmp/w16/t3-nb-3seed.json --seeds 0,1,2 --variants had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp-nb`

    | seed | T0 best arm | `-nb` | paired Δ | top-1 `-nb` |
    |---|---|---|---|---|
    | 0 | 1.2798 | 1.2777 | −0.0021 | 44.20% |
    | 1 | 1.3894 | 1.3086 | −0.0808 | 43.67% |
    | 2 | 1.3896 | 1.3007 | −0.0889 | 44.19% |
    | mean | 1.3529 | 1.2956 | −0.0573 | 44.02% |

    - Mechanism gate (seed 0): aggregate `k/v/o` `J_nb/J_seq = 0.9217` (k `0.9298`, v `0.9274`, o `0.8920`; `J_seq 6.183e6`,
      `J_nb 5.699e6` on `Hl_m`, 24 layers), inside the predicted `[0.90, 0.98]`. Bits `421541376`, peak GPU `4.993–4.998 GB`,
      per-seed transfer wall `130–131 s` (T0 `105.6–106 s`).
    - Controls: (a) base arm seed 0 `1.27977`, bits `421541376` (identical to T0); (b) `-nb` with `ONEBIT_NB_SWEEPS=0`
      bitwise equal to (a) in `kl_heldout` and `kl_insample`; (c) exhaustive `65536`-pair check on four (seed, block) cases
      (`/home/socol/oc-tmp/lanes/w16/t3/ctl.py`, sha256 `5ae8116a…`): live joint ICM within `1.8e-16` of brute force, with the
      cross term `X` zeroed the gap is `1.24e-2` (red); (d) the best arm without `-nb` re-run on seeds `1, 2`
      (`PREDICTION=w16-t0-rerun-seeds12-1`, receipt `/home/socol/oc-tmp/w16/t3-base-seeds12.json`, sha256 `fb0ca184…`, wall
      `226 s`) gives `1.389386` / `1.389587`, so T0's seed-1/2 values reproduce and the paired deltas are against a reproduced
      baseline.
    - Predictions: `w16-t3-jstage-1` and `w16-t3-ctl-ab-1` confirmed; `w16-t0-rerun-seeds12-1` confirmed;
      `w16-t3-nb-3seed-1` refuted on magnitude (predicted mean Δ `[−0.015, −0.001]`, measured `−0.0573`); its falsifier (mean
      `≥ 0`, any seed `> +0.005`, bits changed) did not trigger, so the outcome gate passes.
    - Reading: seed 0 moves by `−0.0021`, inside the predicted band; seeds `1` and `2` move by `−0.08`. The `-nb` seeds spread
      `0.031` nats against `0.110` for the T0 arm, and the `-nb` mean `1.2956` is `0.0999` below the matched base arm's `1.3955`
      (§24). The `J` ratio alone does not account for the seed-1/2 movement; the mechanism for it is open, and the gain is
      recorded per seed rather than as a mean-only claim.
    - Remaining gap to Q4_K_M: the `-nb` mean is `65.8×` the `0.0197` nat target.
    - Lean: `Perf.CostCorrespondence.w16_t3_nb_paired` (audited; changed-constant control `1.3086 − 1.3894 > 0` red);
      `Perf/FaceTypedE8.lean` (spinor closure, shared-channel pointwise null channel, `spinor_cross_diag`, joint block-cost
      expansion, `joint_move_le_planewise`, `spinor_card = 128`, `pair_rate_eq = 2^16`, `sparse_not_injective`; sha256
      `ce2f0d5a…`; build `Built Perf.FaceTypedE8`, all declarations audited with the closed cone).
    - Trust boundary: `J` is the `Hl_m` proxy on the calibration Hessian; KL is Qwen2.5-0.5B on the script's held-out chunks;
      the Lean statements take the measured values as `model` inputs and do not prove the arm's numerical fit.


26. **Junction side information for the `-nb` residual frame: route (b) closed at linear and product-junction order, route (a) priced** (W16c-T6;
    grade `measured` for `R²_H`, rates and comparison counts, `proved` for the Lean verdict over them as `model` inputs; board row
    `onebit-w16-t6-cross-side-o9JggmT2wCrNW9sHFePAF`). Question: the `-nb` residual `R = W − Q(W)` costs how many bits as a
    Hessian-weighted frame, and how much of it do decoded junction partners (same-head `v`/`o`, `q`/`k` through GQA, `gate`/`up`/`down`,
    sorted-key neighbour blocks) predict (decoder-side information, Wyner–Ziv 1976, conditional-rate reading)? Spec: Opus verdict
    `/home/socol/oc-tmp/escalation/2026-10-14-w16-t6-sideinfo.md`.
    Estimator: `E = fwht128(R)·Hr^{1/2}` with `Hr = had_gram(Hl_m)`; gate `‖E‖² = J` to `9.1e-15` relative. Predictors `F0` self,
    `F1` linear junction, `F2` product junction (`o`↔`v`, `down`↔`up/gate`), `F3` routed sorted-key neighbour, each with a decoded and
    an oracle-partner arm; closed-form ridge, leave-one-layer-out; `R²_H` out of fold. Script sha256 `47ccda4d…`.

    Commands (receipt sha256 `7194dce4…`, capture `cap.json` sha256 `28f0663a…`):
    `PREDICTION=w16-t6-capture-1 flock /tmp/opencode/gpu.lock timeout 580 /home/socol/Workspace/spinor-ttl/.venv/bin/python tools/onebit/side_probe.py capture --variants had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp-nb --seeds 0 --dump /home/socol/oc-tmp/lanes/w16/t6/nb_seed0.pt --cap /home/socol/oc-tmp/lanes/w16/t6/cap --out /home/socol/oc-tmp/lanes/w16/t6/cap.json`
    (held-out KL `1.27766`, bits `421541376`, wall `127.2 s`);
    `PREDICTION=w16-t6-probe-1 flock /tmp/opencode/gpu.lock timeout 580 /home/socol/Workspace/spinor-ttl/.venv/bin/python tools/onebit/side_probe.py probe --cap /home/socol/oc-tmp/lanes/w16/t6/cap --dump /home/socol/oc-tmp/lanes/w16/t6/nb_seed0.pt --out /home/socol/oc-tmp/w16/t6-sideinfo.json`
    (controls `27.4 s`, pass 1 `101.1 s`, pass 2 `43.8 s`; the prediction record states `145 s`, the stage sum is `172 s`).

    | kind | `J` share | `R²_H` self (`F0`) | best decoded increment | oracle increment | `R_a(f*)` b/w |
    |---|---|---|---|---|---|
    | `q` | 7.74% | 0.0001 | 0.00058 (`F1`) | 0.00062 | 2.037 |
    | `k` | 0.42% | −0.0001 | 0.00017 (`F1`) | 0.00027 | 0.398 |
    | `v` | 0.30% | −0.0001 | −0.00003 | −0.00003 | 0.420 |
    | `o` | 0.17% | −0.0023 | 0.00002 (`F2`) | 0.00003 | 2.145 |
    | `gate` | 46.25% | 0.0000 | 0.00000 | 0.00001 | 2.888 |
    | `up` | 42.46% | 0.0165 | 0.00223 (`F2`) | 0.00124 | 2.878 |
    | `down` | 2.66% | −0.0002 | 0.00000 | 0.00000 | 0.532 |
    | total (J-weighted) | | 0.0070 | 0.00094 (`F2`) | 0.00051 | 2.072 |

    - Route (b): the J-weighted decoded increment is `0.00094` (`F1` `0.00004`, `F3` `−0.00006`), `0.0072×` the `0.13` opening threshold
      and below `0.02` on every kind; the oracle partner adds at most `0.00124`. The whitened residual is nearly
      white: self `R²_H` is `0.0070` total. Shuffled-partner `|ΔR²| ≤ 4.4e-5`. Planted `R² ∈ {0, 0.3}` recovered within `0.007`.
    - Route (a): the rate to recover `f* = 0.98479` of `J` (the fraction of the `-nb` KL that Q4_K_M parity needs, under KL ∝ `J`) is
      `2.0723` b/w per-tensor water-filling (`1.8807` with a global threshold), against the `3.0196` b/w white bound and the `1.178061`
      b/w budget. At `f = 0.5` the rate is `0.262` b/w; at `f = 0.9` it is `1.078` b/w. Conditioning on junction context leaves it at
      `2.0682` (`F0`; `F1d`, `F2d`, `F3d` the same to four places), against `2.0723` unconditioned.
    - Routed lookup: `F3` sorted-key search used `7.0–12.3` comparisons on average and never exceeded `⌈log₂(m+1)⌉`
      (`q`/`o`/`down` `10`, `k`/`v` `8`, `gate`/`up` `13`; `2.4M–13.1M` lookups per kind).
    - Predictions: `w16-t6-capture-1` confirmed; `w16-t6-probe-1` refuted on the `R_a(f*)` magnitude (predicted `[2.6, 3.1]`,
      measured `2.0723`), its decoded-increment and oracle clauses held, and its falsifier (any kind with ≥ 5% of `J` at increment
      ≥ 0.05, or `R_a(f*) < 2.0`) did not trigger.
    - Decision (spec §5): route (b) is closed at linear and product-junction order; route (a) is closed at `R_a(f*) ≥ 2.0`
      (`1.76×` the bit budget). `gate` + `up` hold `88.7%` of the residual `J`; `v` + `o` hold `0.47%`.
    - Lean: `Perf/SideInfo.lean` (sha256 `ec389a5d…`; `char_coords_norm`, `norm_boost_blind`, `lone_channel_fiber_card`,
      `side_info_recovers`, `side_info_char_coord`, `gauss_side_info_saving`, `lowerBound_count_le`, `lowerBound_spec`,
      `w16_t6_route_b_open`, `w16_t6_route_b_closed_linear`, `w16_t6_route_a_closed`; hyperbolic face `σ = 1`, characteristic
      coordinates `u = a − b`, `v = a + b`; the elliptic `σ = −1` and a changed-constant control are red) and
      `Perf.CostCorrespondence.w16_t6_sideinfo` (§28). The old-corpus statements ported with standard vocabulary are
      `fullDecode_recovers`, `triadDecode_recovers` (`sanguine-ref/proof/Substrate/Algebra/Web/NullWebRotation.lean:195,212`);
      `retrieval_is_O1_navigation` (`NullWebRetrieval.lean:153`) stays Open and is not ported. Lineage search:
      `rg --files -g 'Web*.lean' -g 'Wire*.lean' sanguine-ref sanguine apeiron lrw.com`; `rg "Nat.clog|binary.?search" proof ../sanguine/proof`
      finds no binary search, so `lowerBound` is new.
    - Trust boundary: `J` is the calibration-Hessian proxy and generalisation is `72.8%` of held-out KL (the four-factor decomposition above, `E_gen = 1.0028` nats); `f*` assumes
      KL ∝ `J` (`model`); the Gaussian and Wyner–Ziv rates are anchors; nonlinear predictors are outside scope; seed `0`, fp32 on
      one RTX 5080; the Python is not extracted.

27. **Generalisation factors of the `-nb` gap: `G = O·T·Q` and `E_gen(N)`** (W17-C; grade `measured` for `O`, `Q`, `μ̂` and `E_gen`, `proved` for the
    sandwich in `Perf/Generalisation.lean`; board row `onebit-w17-c-model-aeILmtZsGbJ8hZYFslWj5`). Question: of `G = KL_out/KL_in = 3.653` (seed 0), how
    much is Hessian optimism `O` (finite-sample plus student-input shift), how much is the layer-output MSE ratio `T`, and how much is KL per unit `J`, `Q`?
    Script `tools/onebit/gen_probe.py` (removed; at 2f15d6a9) sha256 `43deb35c…`; held-out fixed at chunks `[100, 140)`; receipts under `docs/evidence/receipts/onebit/`
    (`w17-c-run25`, `-run50`, `-run100`, `-measure100`, `-fit`, sha256 prefixes `99e4200c…`, `4b7236a0…`, `f9e9a412…`, `adb3bced…`, `6fdfa890…`).
    The `N = 100` driver reproduces the 3-seed receipt seed 0 bit-exactly (`dKL = 0.0`, bits `421541376`).

    | `N` (calibration chunks) | `KL_out` | `KL_in` | `E_gen` | `G` |
    |---|---|---|---|---|
    | 25 | 1.5754 | 0.1403 | 1.4351 | 11.23 |
    | 50 | 1.3496 | 0.2248 | 1.1247 | 6.003 |
    | 100 | 1.2777 | 0.3498 | 0.9279 | 3.653 |

    - `E_gen ∝ N^-α`, `α = 0.3146` (local `0.352` for 25→50, `0.278` for 50→100), prefactor `3.917`; one seed, three points.
    - Factors at `N = 100`: `O = 1.4933`, `Q = 3.1229`, `T = 0.7833` (`T = G/(O·Q)` is derived and equals `J_cal/J_in`; the product is bookkeeping). `Q` is the largest.
    - `O = O_fs × O_shift = 1.1461 × 1.3029`: finite-sample (teacher calibration `[0,100)` against held-out) and student-versus-teacher input shift.
      `O_896-wide = 1.4723`, `O_down = 2.2594` (`O_fs` `1.185`, `O_shift` `1.907`; layer range `1.49–3.86`).
    - Per kind `O` (`O_fs`, `O_shift`): `q` 1.407 (1.088, 1.293), `k` 1.469, `v` 1.477, `o` 1.637 (1.160, 1.411), `gate` 1.484 (1.149, 1.292), `up` 1.470, `down` 2.259.
      `J_cal` share: `gate` 46.2%, `up` 42.5%, `q` 7.7%, `down` 2.7%.
    - Sandwich `1/μ̂_max ≤ O ≤ 1/μ̂_min`: 336 tensor pairs, 0 violations. `μ̂_min` median/min: `gate` 0.048/0.014, `down` 0.007/0.001.
    - Controls: synthetic Marchenko–Pastur (`d = 512`, `n = 5400`) `μ̂_min` 0.4811 against edge 0.4790, `O` 2.063 against `1/μ̂_min` 2.078; planted `H_out = H_cal` gives
      `O = 1`, `μ = 1` to `1e-13`; shuffled `R` gives `O` within `[0.983, 1.004]` of the trace ratio; a mutated sandwich is rejected.
    - Predictions: `w17-c-run100-1` confirmed. `w17-c-run25-1` refuted (predicted `E_gen ≈ 3.8`, measured `1.435`); `w17-c-run50-1` refuted (predicted `≈ 1.9`, measured `1.125`).
      P1 (`w17-c-measure100-1`): aggregate `O ∈ [1.3, 2.0]` holds; `O_896 = 1.472` is above `[1.1, 1.33]` and `O_down = 2.259` above `[1.6, 2.09]`, both from `O_shift`; ledger verdict refuted.
      P2 refuted: `O_down·μ̂_min` (J-weighted) `0.017` against `≥ 0.7` (falsifier `< 0.4`); `μ̂_min` of `down` is set by the 40-chunk `H_out`, so `1/μ̂_min ≈ 140` is loose.
      P3 (`w17-c-fit-1-1`) refuted: `α = 0.315` is below `[0.5, 1]` and above the `0.3` falsifier; the `E_gen(25) ≈ 3.8` clause is refuted by the measured `1.435`.
    - Decision: the iid Marchenko–Pastur finite-sample factor explains `O_fs = 1.15` only. `G` is not mainly calibration-size overfit: `E_gen` falls as `N^-0.31`.
      The `J → KL` step (`Q = 3.12`, item D) and the input shift (`O_shift = 1.30`) are the larger factors; route (a)'s `f* = 0.98479` assumed KL ∝ `J` (item 26).
    - Lean: `Perf/Generalisation.lean` (`loewner_sandwich`, `argmin_optimism`, `mp_floor_gt`, `w17_c_sandwich_verdict`; commit `2120009`) and the verdicts over
      generated `Perf.Receipts.Onebit` constants: `w17_c_bookkeeping`, `w17_c_q_dominates`, `w17_c_o_decomposition`, `w17_c_p1_aggregate_in_band`,
      `w17_c_p1_per_kind_refuted`, `w17_c_p2_refuted` (`tightness_Jw = Σ J_out·μ̂_min / Σ J_cal = 0.01718`), `w17_c_p3_alpha_band`, `w17_c_p3_egen_decreasing`,
      `w17_c_p3_egen25_refuted`, `w17_c_sandwich_measured`, `w17_c_run100_bits`; all `#audit`ed, closed cone. Control: alpha bound `3/10 → 32/100` red.
    - Trust boundary: seed `0`, single run per `N`, one RTX 5080 fp32; `H_cal` is the student-input Hessian; `k`/`v` share `q`'s `μ̂` and `up` shares `gate`'s (one stage each); `O(N)` at 25 and 50 not measured; the Python is not extracted.

28. **Split of `Q`: exponential tilt against logit-space linearisation** (W17-D; grade `measured`, with the `proved` identity `Perf.KLTilt.kl_eq_half_var`; board row
    `onebit-w17-d-j-to-kl-FODqnwdkkWselp984dWMO8`). Question: `Q = (KL/J)_out/(KL/J)_in = 3.1229` (item 27) factors as `Q_tilt · Q_lin` with
    `Q_tilt = (KL/q2)_out/(KL/q2)_in` and `Q_lin = (q2/J)_out/(q2/J)_in`, `q2 = ½·E_pos Var_p(δ)` (`p` teacher softmax, `δ` student − teacher log-probability).
    Script `tools/onebit/gen_probe.py` (removed; at 2f15d6a9) (`tilt`) sha256 `5d227370…`, receipt `w17-d-tilt.json` sha256 `94b6b8ee…`, dump `nb100.pt` sha256 `196568f9…`;
    `-nb` arm, seed 0, `N = 100`; in-sample chunks `[0, 20)` (10240 positions), held-out chunks `[100, 140)` (20480 positions); wall 87 s (prediction `w17-d-tilt-1-1`).

    | | in-sample | held-out |
    |---|---|---|
    | mean KL | 0.349802 | 1.277662 |
    | mean `q2` | 0.356854 | 1.062143 |
    | tilt = mean KL / mean `q2` | 0.9802 | 1.2029 |
    | `q2/J` | 2.2413e-5 | 5.7035e-5 |
    | per-position KL/`q2` p10 / p50 / p90 / p99 | 0.674 / 0.941 / 1.400 / 6.88 | 0.647 / 0.985 / 2.601 / 63.8 |
    | positions with `range(δ) > 8` (their share of KL) | 35.3% (56.2%) | 96.0% (99.3%) |
    | median `range(δ)` | 7.36 | 12.52 |

    - `Q_tilt = 1.2272`, `Q_lin = 2.5448`; product `3.122860` against the item-27 `Q` `3.122860` (difference `−2.8e-7`). The tilt carries `18.0%` of `log Q`.
    - `J_out/J_in = 1.170` while `q2_out/q2_in = 2.976`: the layer-local `J` rises 17% from in-sample to held-out text, the logit-space second-order KL rises 198%.
    - Instrument checks on the real positions: Hoeffding `KL ≤ range(δ)²/8` 0 violations (both sets); mean-value `2·KL ∈ [min_t Var_{p_t}(δ), max_t Var_{p_t}(δ)]` on a 41-point grid
      0 violations (an absolute clamp `1e-6` is in the receipt; without it also 0); Simpson quadrature of `2∫(1−t)Var_{p_t}(δ)dt` against `2·KL`: median relative error `7.3e-7` (in), `4.7e-7` (out), max `8.9e-3`, `6.6e-3`.
    - Controls: `δ = 0` gives `KL = q2 = 0`; `ε = 0.05` scaling gives tilt `0.9979` (in), `1.0076` (out); run100 KL reproduced to `1.2e-8` (in `0.349802402` against `0.349802399`, out `1.27766253` against `1.27766254`).
    - Prediction `w17-d-tilt-1-1` confirmed: tilt_in `0.980 ∈ [0.9, 1.3]`, tilt_out `1.203 ∈ [0.9, 1.6]`, `Q_tilt ∈ [0.9, 1.5]`, `Q_lin ≥ 2`; falsifier (`Q_tilt > 2` or `< 0.8`) not triggered.
    - Decision: `Q` is carried by `Q_lin = q2/J` (`2.54`), not by the tilt (`1.23`). `J` underpredicts held-out logit-space distortion by `2.5×` relative to in-sample, so the open
      factor is `J → q2`: per-tensor `J_out/J_in` against the propagation of the weight error to the final logits.
    - Trust boundary: seed `0` only; fp32 on one RTX 5080; `J` is the Fisher-to-Hessian proxy of item 27; per-position tilt means are dominated by a few high-KL positions (`tilt_pos_mean` held-out `4.15`), the set-level ratio is the one entering `Q`.

## Not covered (W15d)

- Constructive transfer evaluated on 0.5B only; `src/learn/ridge.rs` and `src/learn/onebit.rs` (removed; at 9f440192) remain `f64` on CPU.

Board row: `onebit-exact-one-bit-and-FubKIC3prjKrgnryWWxrF1` (LEARN-ONEBIT, `docs/plan.md` row 248).

# ONEBIT W18: the result against the literature and its own trajectory

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

Receipts (committed, rows in `docs/evidence/ledger.jsonl`):

| receipt | content | source |
|---|---|---|
| `docs/evidence/receipts/onebit/w18-literature.tsv` | 57 rows, ~38 papers: arXiv id and version, regime, model, stated and all-in b/w, metric, value, FP baseline, table, PDF sha256 | PDFs and `pdftotext` copies in `/home/socol/oc-tmp/papers/`; every number read at the named table |
| `docs/evidence/receipts/onebit/w18-trajectory.tsv` | 83 arm results, oldest first: commit, arm, b/w and its basis, eval, KL and top-1 in-sample and held-out, seeds, source | committed receipts, then `docs/evidence/onebit-*.md`, then commit messages; regenerator `/home/socol/oc-tmp/lanes/w18/traj/make.py` |
| `docs/evidence/receipts/onebit/w18-trajectory-disagreements.tsv` | 13 numbers that two sources state differently, both values with their sources | as above |

## Literature (`measured` where a row names its table)

- No paper read reports KL or top-1 agreement against the FP model at ≤ 1.6 b/w on a pretrained model of ≤ 3B.
  KL appears only as a training or calibration loss (LittleBit, NanoQuant Eq. 11, BitDistill). This scope is the papers
  in the receipt, not the field.
- Qwen2.5-0.5B appears in no binarisation PTQ paper read. The nearest same-regime results are WikiText-2 perplexity only:

| paper | model | b/w (paper's count) | PPL | FP PPL | ratio | location |
|---|---|---:|---:|---:|---:|---|
| NanoQuant 2602.06694v3 | Qwen3-0.6B | 1.00 | 27.56 | 12.66 | 2.18× | Table 2 |
| NanoQuant 2602.06694v3 | Qwen3-1.7B | 1.00 | 19.21 | 9.39 | 2.05× | Table 2 |
| 2512.21651v3 | OPT-1.3B | 1.11 | 24.30 | 14.62 | 1.66× | Table 1 |
| 2512.21651v3 | OPT-2.7B | 1.11 | 18.25 | 12.47 | 1.46× | Table 1 |
| PTQ1.61 2502.13179v2 | OPT-2.7B | 1.61 | 28.56 | 12.47 | 2.29× | Table 6 |
| PV-Tuning 2405.14852v2 | Llama-2 7B | 1.58 | 7.32 | 5.12 | 1.43× | Table 7 |

- Training-based points: BitNet b1.58 3B from scratch, PPL 9.91 against 10.04 (100B tokens, 2402.17764v1 Table 1);
  BitDistill 2510.13998v1 ternary Qwen2.5-0.5B, MNLI 79.98 against FP16-SFT 79.91 after a 10B-token warm-up (Table 3);
  CAT-Q 2606.26650v1 ternary PTQ Qwen3-1.7B, 5-task average 51.01 against 61.42 (Table 1).
- VBQ 2607.02893v1 is the one KL-against-FP figure found: a 131M model trained from scratch at 1.82 b/w, KLD 0.21 nats,
  argmax flip rate 18.7% (Sec. 5.3).
- Three claims made about the field without reading it are not supported by any paper in the receipt: a QuIP#
  sub-2-bit cliff below 7B (no paper read tests ≤ 2.1 b/w at ≤ 3B), AQLM sub-1B results (none reported), and a
  collapse of closed-form PTQ at 1 bit (the rows above).
- Placement (`model`: perplexity and KL are different metrics): the comparators are measured on perplexity, ours on KL
  and top-1. A bridging measurement, the `-nb` arm on WikiText-2 perplexity or a public 1-bit PTQ code scored by
  `tools/onebit/fold_transfer.py` (removed; at 2f15d6a9), places the two on one axis; it is not run.

## Trajectory (`measured`, `w18-trajectory.tsv`)

Qwen2.5-0.5B, all 168 linear tensors, b/w over the linear weights (embedding and `lm_head` excluded); held-out =
`fold_transfer` over C4 chunks 100–140, except the first row (paired KLD over 64×512 WikiText).

| commit | arm | b/w | held-out KL | held-out top-1 |
|---|---|---:|---:|---:|
| `d759873` | RTN / Q1_0 | 1.125 | 13.17 | 0.0% |
| `d759873` | had-gptq1 | 1.125 | 6.85 | 1.38% |
| `57d3f59` | fold-block | 1.133 | 3.09 | 26.1% |
| `a40e622` | had-fold-block-w0 | — | 2.32 | 31.3% |
| `a40e622` | 2plane-gqa-corr | 1.175 | 2.00 | 34.8% |
| `a40e622` | e8 | — | 1.633 | 39.4% |
| `a40e622` | sink-fnorm | 1.1795 | 1.378 | 42.2% |
| `2975be5` | T0, 3 seeds | 1.1781 | 1.353 | 42.6% |
| `6a958e5` | `-nb`, 3 seeds | 1.17834 | 1.2956 | 44.0% |

- From the first whole-model commit at ≤ 1.25 b/w (`d759873`) to `-nb` (`6a958e5`): 71.08 h of commit time; held-out
  KL falls 10.2× at the same width.
- `-nb` per seed: held-out KL 1.2777 / 1.3086 / 1.3007; in-sample mean 0.35064. Fields at `t3-nb-3seed.json:93–96` are
  seed 0; the 3-seed statistics start at line 134.
- IQ1_S (0.6315 KLD, 62.0% top-1) and IQ1_M (0.5191, 65.9%) quantise `ffn_down` blocks 3–23 only (`onebit-w9.md`); no
  whole-model IQ1 point exists here.
- Every arm after `57d3f59` was selected on the same 40 held-out chunks; the held-out figures carry that selection.

## Disagreements between sources (`w18-trajectory-disagreements.tsv`)

- One arm, two receipts: `had-fold-2plane-attn-bscale` is 1.8500 / 0.6020 in `attn-rrr.json` and 1.8581 / 0.5956 in
  `attn-bscale.json`.
- Q4_K_M carries five KL figures for five evaluations: 0.0243 (WikiText), 0.020054, 0.019729 (C4 held-out), 0.015675
  (K-FAC model), 0.015700.
- b/w basis: W15d "all-in" counts linear weights only; T0 and `-nb` 1.178061 omit the 98,112-bit row index (1.17834 with
  it, `Perf.BitsRefine`).
- Six arms' receipts are only under `/home/socol/oc-tmp/escalation/`; `grow24` (1.4082) and `mlp2p` (1.3890) appear in
  no committed document.

## Second-plane row selection (`measured`, `w18-rowsel-resid-3seed.json`)

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

Prediction `w18-rowsel-resid-1-1`: replacing the second-plane row mass `||W||^2` (`fold_transfer.py:1241`, which ignores
`g_row`, the Hessian `Hm` and the plane-1 residual) with `g_row * diag(R Hm R^T)`, `R = W - W_plane1`, lowers held-out
KL by 0.01 to 0.04 per seed at equal bits. Falsifier: paired mean above -0.02, or any seed worse.

| seed | held-out KL, `-nb` | held-out KL, `resid` | paired difference |
|---|---|---|---|
| 0 | 1.27766 | 1.29769 | +0.0200 |
| 1 | 1.30857 | 1.29427 | -0.0143 |
| 2 | 1.30069 | 1.33376 | +0.0331 |
| mean | 1.2956 | 1.3086 | +0.0129 |

In-sample KL mean 0.3567 against 0.3506; held-out top-1 mean 43.29% against 44.02%; accounted bits 421,541,376 in both.
Refuted (`Perf.CostCorrespondence.w18_rowsel_resid_refuted`, `measured`). A second run is identical to the first except
wall time (364.2 s and 364.6 s). The arm is a scratch copy of `fold_transfer.py` (`ROWSEL=resid`, outside the run class
of the derivation gate), so the gate ledger holds the prediction open; `tools/onebit/` is unchanged.

## Row multipliers on the stored scale (`measured`, `w18-resq-3seed.json`)

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

Cost-model defect in `-res`: `s_sol` is one continuous multiplier per up-projection row, applied after packing
(`fold_transfer.py`, `use_res` block). The stored row scale is `quant_r_hier` (6-bit mean level, 2-bit spread, 4-bit group
codes), so `r * s_sol` is not representable in the accounted `bits_r`. Stored explicitly, the `4864 * 24 = 116,736`
multipliers cost 1,867,776 bits at f16, which moves the arm from 1.17834 to 1.18356 b/w all-in and breaks the `1.18`
clause of `w16_t0_best_arm_3seed` (934,656 bits, 1.18095 b/w, at u8).

Arm `-resq` (`use_resq`): re-quantise `r * s_sol` through `quant_r_hier` for plane 1 and, for the top rows, plane 2, then
rebuild the plane with `compose(fwht128(S*dh), r_new, ch)`. Zero extra bits by construction.

Prediction `w18-resq-live-1`: paired held-out KL difference to `-nb` in `[-0.004, +0.002]`, bits unchanged.

| seed | held-out KL, `-nb` | held-out KL, `-resq` | paired difference |
|---|---|---|---|
| 0 | 1.27766 | 1.29045 | +0.0128 |
| 1 | 1.30857 | 1.28744 | -0.0211 |
| 2 | 1.30069 | 1.29815 | -0.0025 |
| mean | 1.2956 | 1.2920 | -0.0036 |

In-sample KL mean 0.3495 against 0.3506; held-out top-1 43.91 / 43.97 / 44.05% against 44.20 / 43.67 / 44.19%; accounted
bits 421,541,376 in both, `bits_r` unchanged. Confirmed (`Perf.CostCorrespondence.w18_resq_absorbed`, `measured`): the
absorbed arm matches the unrepresentable one inside the seed spread and keeps the `1.18` b/w budget. A scratch run of the
same code gave bit-identical per-variant values, and is withdrawn in the gate ledger because the run was not admitted.

## Outcome metrics against the teacher and Q4_K_M (`measured`, `w18-outcomes.json`)

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`. Every earlier arm was scored by KL and top-1 against the
teacher. This receipt scores teacher, Q4_K_M and the `-resq` arm (1.178 b/w) on outcomes, with one forward
(`fold_transfer.py --outcomes`, fp32, seed 0, 242 s wall on the RTX 5080):

- **C4 true-label perplexity**: the 40 held-out chunks of the KL evaluation, 20,440 scored tokens.
- **WikiText-2 perplexity**: test split joined with `"\n\n"`, non-overlapping 512-token windows (the first token of a
  window is unscored), 584 windows, 298,424 scored tokens. A 512 window is shorter than the usual
  2048, so the absolute value is above published figures; the comparison is across arms.
- **ARC-Easy** (test, 2376 questions) and **HellaSwag** (first 1000 validation items): continuation log-likelihood with
  the lm-evaluation-harness prompt and normalisation conventions (`acc`; `acc_norm` divides by the choice's character
  length). Standard errors: about 1.0 point (ARC) and 1.5 points (HellaSwag).
- **Greedy text**: six fixed prompts, 48 new tokens, no sampling; the texts are in the receipt.

| metric | teacher (fp32) | Q4_K_M (5.52 b/w) | `-resq` (1.178 b/w) |
|---|---|---|---|
| held-out KL / top-1 | 0 / 100% | 0.0197 / 91.37% | 1.2905 / 43.91% |
| C4 perplexity | 24.35 | 24.80 | 86.09 |
| WikiText-2 perplexity (512 windows) | 17.19 | 17.51 | 98.71 |
| ARC-Easy acc / acc_norm | 0.646 / 0.595 | 0.644 / 0.578 | 0.373 / 0.351 |
| HellaSwag acc / acc_norm (n=1000) | 0.401 / 0.496 | 0.405 / 0.490 | 0.308 / 0.371 |
| greedy text vs teacher, first divergence (tokens) | n/a | 5, 11, 46, 7, 48, 9 | 0, 0, 0, 0, 0, 1 |

Reading: Q4_K_M matches the teacher on every outcome inside the standard error (perplexity +1.9%, ARC-Easy -0.2 point,
HellaSwag +0.4 / -0.6 points). The one-bit arm does not: perplexity 3.5x (C4) and 5.7x (WikiText-2), task accuracy
between chance (0.25) and the teacher, and the generated text is degenerate (repetition: `** ** **`, `A: A: A:`, the
same clause looped). The KL metric was not hiding a good outcome; the outcome gap is the KL gap, `exp(1.29) = 3.6` against
the measured 3.5x on C4. The sparse inference pattern in the owner's note is not exercised here: this is the dense
forward with fitted weights.

Prediction `w18-outcomes-1`, confirmed with two misses: the teacher's ARC-Easy `acc` 0.646 sits above the stated
0.55-0.62 band, and the one-bit arm's HellaSwag `acc_norm` 0.371 above the stated `<= 0.35`; the falsifier
(`acc_norm` ARC > 0.45, HellaSwag > 0.40, perplexity < 2x teacher) did not fire. Verdict:
`Perf.CostCorrespondence.w18_outcomes_gap` (`measured`, with the changed-constant control).

## Perplexity and KL (`proved` identity, `measured` instances; `Perf/PerplexityKL.lean`)

For a data distribution `d`, teacher conditional `p` and student conditional `q` on a finite vocabulary,
`log(ppl(d,q) / ppl(d,p)) = KL(p || q) + cross(d,p,q)` with `cross = sum (d - p)(log p - log q)`
(`Perf.PerplexityKL.log_ppl_ratio`). On teacher-distributed data `cross = 0`, so the log perplexity ratio is exactly
`KL(p || q)` and is non-negative (`log_ppl_ratio_calibrated`, `kl_nonneg`: Gibbs). On a token sequence with one-hot labels
the same holds per position and averages (`token_ppl_identity`). The cross term is the only place the data enters:
it is the mismatch between the real text and the teacher's own distribution.

Measured instances on C4 (`Receipts.w18_outcomes`, seed 0; KL is the held-out mean):

| arm | KL | log(ppl ratio) | KL - log(ppl ratio) | verdict |
|---|---:|---:|---:|---|
| Q4_K_M | 0.01973 | 0.0183 | in (0.0010, 0.0020) | `q4km_gap` |
| `-resq` (1.178 b/w) | 1.29045 | 1.263 | in (0.020, 0.037) | `resq_gap` |

The gap is the negated cross term plus finite-sample noise: 2% of KL for the one-bit arm, 7% for Q4_K_M. So the held-out KL is the
perplexity gap to within the cross term on this corpus, and the earlier KL-only trajectory (RTN 13.17 to `-nb` 1.2956) is
an outcome trajectory. The one-bit arm's KL is between 65 and 66 times Q4_K_M's (`resq_kl_ratio`). Matching Q4_K_M's
perplexity on teacher-distributed text needs `KL = ln(1.019) = 0.019`. Out of domain the C4 KL does not transfer:
WikiText-2 gives `ln(98.71 / 17.19) = 1.75` for the same arm.

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

Not covered: the cross term is not bounded by a theorem (the receipt gives its value at two points); positions in the
KL average 512 per chunk against 511 scored targets; one seed.

## Speculative acceptance from KL, and the seed survey (`proved`; `Perf/Pinsker.lean`, `Perf/SpecDraftKL.lean`)

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

- `Perf.Pinsker.pinsker` (`proved`, cone `{propext, Classical.choice, Quot.sound}`): `(∑|p−q|)² ≤ 2 KL(p‖q)` on
  `Fin n`, `q > 0`. Route: `scalar_ineq` `3(t−1)² ≤ (2t+4)(t log t − t + 1)` for `t ≥ 0` (monotonicity of the difference,
  `F' = 4g`, `g' = log t + 1/t − 1 ≥ 0`), then Cauchy–Schwarz with `∑ q(2t+4)/3 = 2` and `∑ q h(t) = KL`
  (Csiszár 1967; Kemperman 1969). Mathlib at the pinned revision has no Pinsker.
- `Perf.SpecDraftKL.accept_ge` (`proved`): the speculative-decoding acceptance `∑ min(p,q) = 1 − TV`
  (`Nous.Model.accept_rate`; Leviathan, Kalman, Matias 2022, arXiv 2211.17192; Chen et al. 2023, arXiv 2302.01318,
  both retrieved and read) is at least `1 − √(KL(p‖q)/2)`. `mean_accept_ge`: the same bound for the mean over positions
  from the mean KL (Cauchy–Schwarz as Jensen).
- Receipt instances (`measured` KL, `proved` bound): Q4_K_M held-out KL 0.0197 gives mean acceptance `> 0.90`
  (`q4km_accept`); the one-bit arm's 1.2905 gives `> 0.19` and the bound is `< 0.2` (`resq_accept`). The bound is a
  guarantee, not the rate: the rate is `1 − TV`, measured from logits (not yet run on the arm).
- Controls: `/home/socol/oc-tmp/lanes/w18/ctl_spec.lean`, 3/3 changed constants fail (`> 0.91`, `> 0.2`, `4(t−1)²`).
- Seed survey (read, not run). `sanguine-ref/proof/Substrate/Seed.lean`: the generator interface and round trip
  (`genDecode_genEncode`) are `proved` for any predictor; two instances (`rawGen`, `deltaGen`); the file records that
  trained transformer weights are full-rank, so a regenerating generator must be nonlinear, and names that rung as not
  built. `theory/SEED_SPEC.md` (2026-06, GPT-2 / Qwen2.5-0.5B, `measured` there): a low-rank-plus-int4 split was
  larger than NF4 at equal perplexity; entropy coding the codes saved about 1% (affine fit uniformises codes); the
  gain came from the function-aware distortion (`‖X(W−Ŵ)‖²`, AWQ-class), +0.1125 vs +0.1584 bits/token at 4.25 b/w.
  Proved in this repo and usable: `Perf/SideInfo.lean` (`char_coords_norm`, Gaussian side-information saving),
  `Nous/Combinator/Orbit.lean`, `Nous/Model/RopeOrbit.lean`, `Nous/Model/SpectralTrichotomy.lean` (orbit of a unit
  under cell multiplication per face: elliptic periodic, hyperbolic and parabolic unbounded). `whispering-tides`
  `globe.slang` generates a planet from a seed with scale-free FBM (octave gain `2^−H`).
- Argon (`gemini-4-argon#high`, 101 s, `seed-argon-3`) review, checked: file and theorem citations resolve; its
  "256×" and "378× at 0.4% MSE" `SEED_SPEC.md` figures occur nowhere in that file (`rg -n "378|256×"` returns nothing)
  and are discarded. Bare `--agent argon` exits 1 ("Nothing to compact yet"); `-m "google-vertex/gemini-4-argon#high"`
  is required. Its proposed experiment (rank-`k` plus 1-plane residual at 0.185 b/w) is a `model` hypothesis.

Not covered here: the bound is from the token-level KL(teacher‖draft); the measured `1 − TV` of the finished arm is
in the next section. The seed generator that regenerates a trained tensor is not built.

## Measured speculative acceptance of the finished arm (`measured`, `proved`)

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`. Receipt
`docs/evidence/receipts/onebit/qreg-spec.json` (ledger `onebit-receipt-qreg-spec`); Lean `Perf/SpecDraftKL.lean`
§ "Block length and step-time model" and § "Receipt instances"; constants `Receipts.qreg_spec`.

Instrument: `fold_transfer.spec_stats` (called from `kl_eval(..., stats=)`, recorded by `pv_finish.py` as
`spec_heldout_after`) takes per held-out position `a_t = ∑ min(p_t, q_t) = 1 − TV` (teacher `p`, arm `q`) and the
block length `E_k = mean_t ∑_{j≤k} ∏_{i<j} a_{t+i}`, read along the held-out text (teacher-forced: the context is the
text, not the accepted drafts). Self-test in `--yaqa-controls`: identical draft gives `a = 1`, `E_k = k + 1`;
`p = (½, ½)`, `q = (¾, ¼)` gives `a = ¾` and the geometric `E_k`; both exact (error 0). Mutation (block counted from 0,
not 1): `AssertionError (0.5, 0.571)`, exit 1.

| quantity (arm `-yaqas50`, 1.178 b/w, 549 steps, 541.9 s) | value |
|---|---|
| mean `1 − TV` over 20480 held-out positions | 0.5972 |
| held-out KL / top-1 | 0.85764 / 0.5502 |
| `E_k`, k = 1..8 (teacher-forced) | 1.597, 1.968, 2.204, 2.356, 2.455, 2.521, 2.565, 2.595 |
| geometric `E_k` at `a = 0.5972`, k = 4 / 8 | 2.294 / 2.459 |
| host DRAM read, all 32 threads (i9-14900KF) | 86.82 GB/s (85.58–89.19, 5 reps) |
| pinned host-to-device (RTX 5080) | 12.80 GB/s (12.78–12.83) |
| device read | 865.20 GB/s (832.14–906.40) |

- `spec_accept_measured`: acceptance in `(0.59, 0.60)`; the Pinsker bound (`accept_ge`, `proved`) from the same KL is below `0.35`. The
  measured rate is 1.7× the guarantee.
- `spec_runs_cluster`: the teacher-forced `E_4`, `E_8` exceed the geometric `blockLen a k` at the mean acceptance.
  Rejections bunch on hard tokens, so the iid formula undercounts the block.
- `blockLen_geom` (`(1 − a)·E = 1 − a^{k+1}`), `blockLenAt_const`, `blockLen_mono`, `blockLen_bounds`: the
  block-length algebra the instrument computes.
- `spec_faster`: with a verifier pass that costs `tV` for one position or `k + 1`, speculation beats the verifier alone
  exactly when `k·tD < (E − 1)·tV`.
- The 27B tokens/s figure is in the next section (`spec27b_cpu`). The step-time model first committed here charged a
  7-token verify as one weight read and gave above 10 tokens/s; the CPU verify curve below refutes that premise.
- At the Q4_K_M acceptance (`q4km_accept`, `proved`: `a > 0.9`) the geometric `E_8` is above 6.1.

## Training the acceptance: the TV finishing objective, and the measured CPU verify cost (`measured`, `proved`, `model`)

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`. Receipt
`docs/evidence/receipts/onebit/qreg-spectv.json` (ledger `onebit-receipt-qreg-spectv`); Lean `Perf/SpecDraftKL.lean`
§ "Training the acceptance directly" and § "Receipt instances: TV-finished against KL-finished"; constants
`Receipts.qreg_spectv`.

Acceptance is `1 − TV` (Leviathan, Kalman and Matias, arXiv 2211.17192, Thm. 3.5), so `pv_finish.py --objective tv`
finishes the arm on `½ ∑ |q − p|` per position instead of KL. `TVHead` forms the logit adjoint
`½ q_y (s_y − ⟨s, q⟩)`, `s = sign(q − p)`; `softmax_adjoint` proves the softmax-Jacobian step. Control: the fused
backward against autograd of the checkpointed TV loss in every run (`grad_equiv`); mutation (adjoint without the
`−⟨s, q⟩` centring) gives requested-gradient rel L2 5.453 against a noise floor of 6.06e-3, `AssertionError`, exit 1.
DistillSpec (Zhou et al., ICLR 2024, arXiv 2310.08461) reports direct TVD "does not yield the best performance in most
of the settings explored"; on this arm it does.

| arm `-yaqas50`, 1.178 b/w, 580 s, 3 runs each | KL-finished (s50, kl2, kl3) | TV-finished (tv1–3) |
|---|---|---|
| mean `1 − TV`, 20480 held-out positions | 0.59800 (range 0.00175) | **0.61221** (range 0.00034) |
| teacher-forced `E_3` / `E_6` | 2.207 / 2.526 | 2.254 / 2.608 |
| held-out top-1 | 0.5517 | 0.5553 |
| held-out KL | 0.855 | 0.944 |
| steps | 549, 557, 554 | 520, 522, 523 |

- `spectv_accept`: every TV run above 0.612, every KL run below 0.599; the mean gain exceeds 0.014 while each arm's
  range stays under 0.002.
- `spectv_top1_kl`: every TV run's top-1 is above every KL run's while its held-out KL is above 0.94 against below
  0.86. KL does not track the outcome here.

CPU verify cost (llama.cpp `60bc8866b`, Qwen3-8B Q4_K_M 5.02 GB, `-ngl 0`, 24 threads, 5 reps; a pass of `m` tokens is
the verifier's cost for a draft of `k = m − 1`):

| m | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ms | 67.8 | 73.4 | 83.8 | 86.3 | 95.8 | 105.8 | 121.8 | 125.4 |
| `t(m)/t(1)` | 1.00 | 1.08 | 1.24 | 1.27 | 1.41 | 1.56 | 1.80 | 1.85 |

At 8 and 16 threads a 7-token pass is 2.38× and 2.06× one token; one token at 16 threads is 66.7 ms (15.0 tokens/s),
0.87 of the 57.9 ms DRAM-read bound.

- `verify_curve`: `t(7) > 7/4·t(1)` and `t` rises in `m`. The verify is not one weight read on this CPU.
- `verify_best_k`: with the TV draft, `E_k / (t(k+1)/t(1))` is largest at `k = 3` (1.77), above every other `k = 1..7`.
- `spec27b_cpu` (`model` step times, `proved` arithmetic): the 27B Q4_K_M verifier (`q4kmBpw` 5.52 b/w, 18.6 GB,
  pass time scaled per weight byte from the 8B curve) decodes below 4 tokens/s alone; with the TV-finished 1.18 b/w
  draft in VRAM at `k = 3` the pair emits above 6.5 tokens/s, over 1.65× the verifier alone, output exact to the
  verifier (`Nous.Model.specStep_eq_target`). At `k = 6` it is below 6.

The verify cost is a kernel property, not a model one: llama.cpp's CPU Q4_K path at `m = 7` runs above the DRAM-read
time. A weight-stationary multi-column kernel that reads each block once for all `m` columns moves `t(m)/t(1)` toward 1
and the best `k` toward the longest block.

Model assumptions behind `spec27b_cpu`, each an open measurement: the 27B Q4_K_M rate is the 0.5B receipt's 5.52 b/w;
the 8B curve's per-byte cost carries to 27B; the 0.5B pair's acceptance stands for the 27B pair's; `E_k` is
teacher-forced, not sampled from the draft.

## Argon review outcomes (checked against source)

- `fnorm` is a 0-bit 896-vector Newton-Fisher fit on the final RMSNorm (`fold_transfer.py:877`); the dense 896x896
  matrix at 0.0359 b/w in the review appears in no receipt.
- Held-out top-1 for `-nb` is 44.02% (`t3-nb-3seed.json`); the review's 53.20% matches no receipt.
- The gate-error compensation in `up` is `glu_comp` (`fold_transfer.py` near :1197); the per-column gauge `hpre` is
  refuted in §14.
- Open: `nb_refine` runs for `k`, `v`, `o` only (`fold_transfer.py:1229`).
- Resolved: the gate/up multipliers `s_sol` of `-res` are not counted in the bits and are not representable in them;
  `-resq` folds them into the stored scale (previous section).

## Straight-through KL distillation of a single sign plane (`measured`; `Perf/KDScaling.lean`)

Command (prediction `kdmain-1`, `kdlr-1`; script `tools/onebit/kl_distill.py` (removed; at 2f15d6a9) sha256 `3690026e…`; RTX 5080; train data
C4 shard 00001, sha256 `77078e5f…`, disjoint from the calibration file):

| run | steps | tokens | lr latent / scale | held-out KL (fp32, chunks 100–140) | top-1 | signs flipped | train wall |
|---|---|---|---|---|---|---|---|
| start (`sign(W)`, `r` = group mean abs) | 0 | 0 | | 12.694 | 0.000 | 0 | |
| `w18-kd-main1.json` | 520 | 2,129,920 | 2e-4 / 1e-3 | 2.3038 | 0.3034 | 2.4 % | 317 s |
| `w18-kd-main2.json` | 880 | 3,604,480 | 1e-3 / 2e-3 | 2.0104 | 0.3374 | 7.4 % | 467 s |

- Student: `W'[o,i] = r[o,⌊i/128⌋] · sign(L[o,i]) · c[i]`, `r`, `c` f16, latent `L` fp32 and not stored. Stored rate
  1 + 16/128 + 16/O per weight over the 168 linears: 406,487,040 bits / 357,826,560 weights = 1.13599 b/w, below the
  post-training arm's 1.178. Embeddings, norms, biases and `fnorm` are teacher values.
- Loss `KL(teacher ‖ student)` on next-token distributions; straight-through estimator on `sign` (Bengio, Léonard and
  Courville 2013, arXiv 1308.3432); fused Adam, 20-step warmup, cosine to 10 %; batch 8 × 512 tokens; student in bf16
  autocast, loss head and the held-out score in fp32. Same instrument and chunks as the post-training arm.
- Verdicts (`proved`, `Perf.KDScaling`): `bpw_below_arm`, `kd_order` (main2 < main1 < start; both < `fold-local` 3.806;
  arm mean 1.2920 < main2; Q4_K_M 0.0197 < arm), `kd_ratios` (> 1.8× over `fold-local`, > 6× over the start, between
  101× and 102× above Q4_K_M), `tokens_per_step`, `kd_accept_bound_vacuous` (the Pinsker acceptance guarantee of
  `Perf.SpecDraftKL.accept_ge` is negative at KL 2.01). Controls: `ctl_kd1`–`ctl_kd3` (changed constants) fail.
- Same-instrument comparison at ~1.13 b/w: `fold-local` 3.806 (post-training), distillation 2.010 (8 minutes). The
  2-plane E8 post-training arm (1.178 b/w) stays ahead at 1.292. The student is a single-plane fit; the arm's
  factorised structure is not used as the start.
- Cost model (`model` grade, a fit over one decade of the quick-KL trajectory, 8 held-out chunks, bf16): KL ∝ tokens^−α
  with α = 0.311 (main1, steps 52–520) and 0.325 (main2, steps 88–880). Anchored at the exact main2 KL, reaching 1.29
  takes ≈ 14M tokens (≈ 3,500 steps at batch 8, ≈ 0.53 s/step); 0.0197 extrapolates to ≈ 10¹³ tokens. The trajectory
  was still falling at the last step of both runs. 3,500 steps exceed the 600 s measurement ceiling at this step time.

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

## Calibration size and the 2-plane arm (`measured`; `Perf/CalScaling.lean`)

Prediction `calx-1`, refuted. Script `tools/onebit/fold_transfer.py` (removed; at 2f15d6a9) `--cal-extra 500` appends the first 500 chunks of C4
shard 00001 (`train-c4.qwen.npy` sha256 `74945dcc…`) to the 100 `calib-c4` calibration chunks; the run's script sha256
is `458d6efd…`. The committed file (`9183c172…`) adds `--model`/`--q4km` with unchanged defaults. Held-out chunks
`[100, 140)` and in-sample chunks `[0, 20)` are the same as the 100-chunk fit. RTX 5080, single chunk order.

| fit | calibration tokens | b/w | held-out KL | held-out top-1 | in-sample KL | transfer wall |
|---|---|---|---|---|---|---|
| `w18-resq-3seed.json` (3 orders) | 51,200 | 1.17806 | 1.2905 / 1.2874 / 1.2982 | 0.439–0.440 | 0.349 / 0.351 / 0.348 | 120 s |
| `w18-calx600.json` | 307,200 | 1.17806 | **1.2505** | 0.4456 | 0.7734 | 419 s |

- Predicted held-out 0.80–1.05; measured 1.2505, above the 1.20 falsifier. Six times the data moves held-out KL by
  −0.040 against the same chunk order (−3.2 % of the 3-order mean) while in-sample KL more than doubles: the 3.7× gap of the 100-chunk fit was
  in-sample overfitting, not recoverable held-out quality.
- Verdicts (`proved`, `Perf.CalScaling`): `calx_same_code` (bits, params, 600 × 512 tokens), `calx_heldout` (below each
  100-chunk order, by < 0.04 and < 1/30 of the mean; above 1.20), `calx_overfit` (in-sample > 2×; gap > 3.6× before,
  < 1.7× after), `calx_standing` (below distillation's 2.01; 63–64× above Q4_K_M). Control `ctl_calx` (changed
  falsifier constant) fails.
- Cost: gram + forward passes scale with calibration tokens (12.3 s + 5.0 s at 100 chunks, 154 s + 101 s at 600);
  the refit timer grows 101 s → 163 s through the `-resq` GLU solve, which streams the stored activations.
- Decision: the 2-plane arm's held-out level at 1.178 b/w is a property of the representation. Per the switch rule
  this route stops here and the work moves to a generator plus coded residual.

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

## Seeded trellis generator against E8 at 1 b/w (`measured`; `Perf/Trellis.lean`)

Prediction `trellis-2`, confirmed (`trellis-1` withdrawn: its single command slot went to a debug run). The generator
form of "generate the weights from a seed" is random coding: a seed fixes a pseudo-random codebook and the encoder
searches for the seed path whose generated values best match the weights. `tools/onebit/trellis_gen.py` (removed; at 2f15d6a9) (sha256
`a364850b…`) builds the strongest published instance, the bitshift trellis with the 1MAD hash generator (QTIP, Tseng
et al., arXiv 2406.11235): `L` state bits, one code bit per weight, Viterbi over 16-column blocks with per-survivor
GPTQ feedback, plus `L/n` start-state bits per row. It is scored on the arm's own objective `J` (Hadamard-rotated
weights, damped Gram of calibration chunks `[0, 20)`) against `gptq_e8` at beam 1 and beam 4 with the same scale
refit. 21 linears (layers 0, 11, 23 × q, k, v, o, gate, up, down), RTX 5080, 190 s. Receipt `w18-trellis2.json`.

| code | b/w | `J / J_e8_beam1` median | geomean | min over 21 | `J / J_e8_beam4` median |
|---|---|---|---|---|---|
| E8 beam 4 | 1.125 | 0.90 (layer 0 q) | | | 1 |
| 1MAD trellis, L = 8 | 1.127–1.134 | 1.616 | 1.638 | 1.539 | 1.778 |
| 1MAD trellis, L = 12 | 1.128–1.138 | **1.606** | 1.619 | 1.498 | 1.759 |
| sign trellis, L = 8 (layer 0 q) | 1.134 | 0.867 | | | ≈ 0.97 |
| sign Viterbi, L = 1 (layer 0 q) | 1.125 | 0.922 | | | |
| scalar GPTQ signs (layer 0 q) | 1.125 | 1.227 | | | |

- The hashed generator loses on every linear, at both lengths; 16× the states moves the median by 0.010.
- The search is not the loss: the same Viterbi over a sign alphabet (`v(s) = ±1` by the state's last bit) beats E8.
  At one bit per weight a Gaussian-like codebook spends rate on amplitude that the sign alphabet does not carry;
  QTIP's own operating points are 2–4 b/w.
- Instrument controls: Viterbi path cost equals `J` to 1.6e-8 relative; DP cost equals the recomputed path;
  diagonal `H` gives exact brute-force equality. For non-diagonal `H` per-survivor Viterbi is a heuristic upper bound
  (32/60 toy rows equal the 2¹⁶ brute-force minimum, max gap 79%), so the trellis ratios above are an upper bound on the
  code's best `J` under this encoder.
- Verdicts (`proved`, `Perf.Trellis`): `trellis_bits` (the trellis stores more bits than E8 at every coded width),
  `trellis2_confirmed`, `trellis_loses_everywhere` (min > 1.49 vs beam 1, > 1.6 vs beam 4), `trellis_length_flat`,
  `sign_search_wins`, `trellis_identity`. Control `ctl_trellis` (minimum bound raised to 1.5) fails.
- Decision: a seeded pseudo-random codebook is not the generator for the 1-bit arm. The 3.4 % sign-trellis gain over
  beam 4 on one matrix is a `J` refinement of the existing code, not a route to the 63× KL gap; it is not pursued
  before the size trend.

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

## Model size and the 2-plane arm (`measured`; `Perf/SizeTrend.lean`)

Switch-rule run 2: does the arm's KL gap shrink from Qwen2.5-0.5B to Qwen2.5-1.5B? The basis is the literature's
one-bit post-training trend (OPT-1.3B → 2.7B log-perplexity gap 0.51 → 0.38, `w18-literature.tsv`) and the K-FAC rate
model's `R*` falling with size (section "The Q4-matching rate across model size"). One protocol for every fit:
`fold_transfer.py --n-cal 50` (the 1.5B transfer at 100 chunks projects to 622 s, over the 600 s ceiling;
`size15-timing-1` refuted, `size15-ncal-1` confirmed), held-out `calib-c4` `[50, 90)`, in-sample `[0, 20)`, fp32,
each model against its own imatrix Q4_K_M, RTX 5080. The arm's gate/up second plane is a fixed row count, now the flag
`--qmlp-rows` (default 416, so earlier receipts are unchanged); at 1.5B, 416 of 8960 rows stores 1.142 b/w, so a third fit
widens it to 792 rows and matches 0.5B's sign bits per weight (1.1192).

| fit | script | b/w | sign bits/w | held-out KL | top-1 | in-sample KL | Q4_K_M KL (b/w) | arm / Q4_K_M | wall |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B, 416 rows (`w18-size05-n50.json`) | `9183c172` | 1.178 | 1.1192 | **1.1637** | 0.480 | 0.2224 | 0.01837 (5.52) | 63.3× | 118 s |
| 1.5B, 416 rows (`w18-size15-n50.json`) | `9183c172` | 1.142 | 1.0945 | **1.3237** | 0.468 | 0.1830 | 0.01917 (4.81) | 69.0× | 389 s |
| 1.5B, 792 rows (`w18-size15-m792.json`) | `c3af3e1e` | 1.168 | 1.1192 | **1.3426** | 0.464 | 0.1780 | — | 70.0× | 373 s |

- `size15-arm-1` refuted: `KL_1.5B / KL_0.5B` = 1.137 against the 0.95 falsifier. `size05-n50-1` refuted on its
  range (1.25–1.60 predicted for 0.5B at 50 chunks; 1.1637 measured).
- `size15-matched-1` refuted: matching the stored bits raises held-out KL to 1.3426 (ratio 1.154). The 0.036 b/w
  deficit is not the gap: the extra 376 rows per gate/up lower in-sample KL and raise held-out KL.
- In-sample KL falls with size (0.222 → 0.183 → 0.178) while held-out rises; the held-out/in-sample ratio goes from
  5.2× to 7.2–7.5×. Q4_K_M moves 4% with no fit to calibration activations.
- Verdicts (`proved`, `Perf.SizeTrend`): `size_same_arm`, `size_bits` (416 rows store > 1/50 fewer sign bits/w at
  1.5B; 792 rows match to 10⁻⁴), `size_heldout_rises` (> 1.13× and > 1.15×), `size_more_rows_overfit`,
  `size_gap_grows`, `size_q4km_ratio` (63–64× → 69–70×). Control `ctl_size` (the 1.13 bound raised to 1.15) fails;
  axioms `{propext, Classical.choice, Quot.sound}`.
- Decision: the switch rule fires on run 2. The arm's held-out loss on this pair grows with model size, and the part
  that grows is the in-sample/held-out split. The measured decomposition of that split (item 27: `G = O·T·Q`, with the
  `J → KL` step `Q = 3.12` the largest factor and calibration size `E_gen ∝ N^-0.31`) names the target: the per-layer
  quadratic objective `J`, fitted on calibration activations, is not the end-to-end KL. The next run optimises the
  arm's stored codes against end-to-end KL instead of `J`.

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

## End-to-end KL finishing of the arm's stored code (`measured`; `Perf/PVFinish.lean`)

The size-trend section names the per-layer objective `J` as the gap. This run keeps the arm's stored code and its bit
count fixed and optimises it against end-to-end KL, in the manner of PV-Tuning (Malinovskii et al., arXiv 2405.14852,
which reports Llama-2 7B at 1.58 b/w within 1.43× of fp16 perplexity).

- Method (`tools/onebit/pv_finish.py` (removed; at 2f15d6a9), sha256 `c973b38c…`; `tools/onebit/fold_transfer.py` (removed; at 2f15d6a9) sha256 `f21e276a…`, whose
  `PACK_SINK` records every linear's stored planes `(S_rot, r, c, d, rows)`): build the arm on Qwen2.5-0.5B exactly as
  `w18-size05-n50.json` (n-cal 50, held-out `calib-c4` `[50, 90)`, in-sample `[0, 20)`), then train on forward
  KL(fp32 teacher ‖ student) over C4 training chunks disjoint from calibration.
  - P-step: Adam on the scales `r`, `c`, `d` through the arm's own quantizers by a straight-through estimator, plus the
    norms and q/k/v biases (zero stored bits). Latents are log-space offsets; lr 1e-3 (scales) and 1e-4 (norms, biases),
    10-step warmup, cosine to 10% over the time box.
  - V-step every 25 steps: move E8 ∪ O8 codebook indices among each row's 16 nearest codewords by the linearised gain
    `⟨g, Δ⟩`; 8 candidates (skip-rank × fraction) scored on a disjoint training batch, the best applied only if that
    batch's loss falls.
- Prediction `pv05-1` (held-out KL in `[0.55, 0.95]`, falsifier `≥ 1.10`): confirmed. One run, RTX 5080, 541.7 s
  (transfer 107.8 s, training 418.7 s); receipt `docs/evidence/receipts/onebit/w18-pv05.json`.

| metric | before | after |
|---|---:|---:|
| held-out KL (nats/token) | 1.1637 | **0.8981** |
| held-out top-1 | 0.4797 | 0.5379 |
| in-sample KL | 0.2224 | 0.2416 |
| stored bits | 421,541,376 (1.17806 b/w) | 421,541,376 |
| ratio to Q4_K_M held-out KL | 63.3× | 48.9× |

- Training: 432 steps at batch 8 × 512, 1,769,472 tokens. 17 V-steps, one accepted (step 25, 62,376 of ~50M blocks).
- Controls, all pass: pack reconstruction `torch.equal` on 168/168 linears; pre-finishing held-out KL equals the
  size-trend receipt bit for bit; every 8-block of `S_rot` is a codebook row (288 planes); step-0 weights equal the arm;
  the 3,592 training chunks used overlap none of calibration `[0, 90)`; eval weights equal the re-quantized stored values
  (168/168); stored bits and every bit component are unchanged.
- Deviations from the brief: the brief's V-step (top 0.5% of blocks by gain) raised the loss 1.38 → 12.7 and was
  replaced by the scored candidate set above; `r` is quantized on a frozen per-row grid because `quant_r_hier` is not
  idempotent; the accepted low-rank correction (q rank 2, down rank 1) is a fixed dense plane, its bits counted.
- Mechanism (`proved` arithmetic over the receipt, `Perf.PVFinish`):
  - The gain is generalisation: in-sample KL rises 0.222 → 0.242 while held-out falls; held-out/in-sample goes from
    5.2× to 3.7× (`pv_gap_shrinks`).
  - The trained parameters are the scales, 19,252,224 bits, under 5% of the stored bits; the sign codes hold
    400,490,496 bits, over 95% (`pv_scales_small`).
  - The codes did not move because the first-order move is wrong at codeword scale: all 136 V-step candidates predicted
    a loss fall, one realised it, and that one realised −0.0022 against a predicted −0.1029, under 1/45
    (`pv_vstep_first_order_fails`). Larger candidate sets (250,080 moves) raised the loss up to 1.04 → 10.7. A codeword
    move is a finite jump; the curvature term `½ ΔᵀHΔ` outweighs `⟨g, Δ⟩`.
  - Also `pv_same_arm`, `pv05_confirmed`, `pv_heldout_drop` (> 22% drop, top-1 +5.8 points), `pv_q4km_ratio`
    (48–49×), `pv_vstep_rare`. Control `ctl_pv` (the 0.95 bound lowered to 0.89, the 1/45 ratio raised to 1/47) fails at
    both lines; axioms `{propext, Classical.choice, Quot.sound}`.
- Decision: end-to-end finishing is the first lever since W17 to move held-out KL by a fifth at fixed bits, and it did
  so through 5% of the bits. The other 95% are the next target: a V-step whose gain includes the curvature term,
  `⟨g, Δ⟩ − ½ Σᵢ Fᵢᵢ Δᵢ²` with `F` the diagonal empirical Fisher accumulated over the P-step (the Optimal Brain Surgeon
  criterion, Hassibi and Stork (1993), at diagonal order), so that predicted and realised code moves agree and the
  codes train. Single-batch training loss sits in 0.91–1.11 from step 50 on, so more tokens on the scales alone is not the lever.

### Curvature-corrected V-step (`measured`, refuted; `Perf/PVFinish.lean`)

- Method: `pv_finish.py --v-gain obs`. Over the P-steps since the last V-step, accumulate per coded plane the
  gradient `ḡ` and the diagonal empirical Fisher `F = mean g²` of `dL/dS` (gradient hooks on the `S` leaf). Each block's
  best candidate among its 16 nearest codewords minimises `⟨ḡ, Δ⟩ + (λ/2) Σ F Δ²`, the Optimal Brain Surgeon criterion
  (Hassibi and Stork (1993)) at diagonal order. Tries are `λ` × fraction {0.001, 0.01} of positive-gain blocks per
  plane, each scored on a disjoint batch and reverted unless the loss falls. `--obs-lams` sets the `λ` grid; its
  default `1,4,16,64` reproduces the first run.
- Runs (RTX 5080, same protocol and controls as `pv05-1`, all controls pass, pre-finishing KL reproduced bit for bit):

| receipt | `pv_finish.py` | `λ` | V-steps accepted | tries that lowered the loss | held-out KL after | in-sample after |
|---|---|---|---:|---:|---:|---:|
| `w18-pv05.json` | `c973b38c…` | linear gain | 1 / 17 | 1 / 136 | 0.8981 | 0.2416 |
| `w18-pvobs05.json` | `bab41be3…` | 1–64 | 0 / 18 | 0 / 144 | 0.8906 | 0.2387 |
| `w18-pvlam05.json` | `3e14c5d8…` | 1024–65536 | 0 / 17 | 0 / 136 | 0.8948 | 0.2382 |

- Predictions `pvobs05-1` (held-out in `[0.72, 0.88]`) and `pvobslam05-1` (≥ 3 accepted V-steps) are refuted
  (`pvobs_refuted`). Every try was predicted to lower the loss; none did (`pvobs_vstep_fails`). First run: realised
  +5.3 to +25.4 against predicted −1 to −13.7, and `λ` from 1 to 64 changed the prediction by at most 11%. Each
  P-step gradient averages 4096 tokens (`tokens = 4096 · steps`), so the batch-gradient `F` sits about 4096 times below the
  per-token one. Second run: at `λ = 65536 = 16 · 4096` the best try (28,603 moves) was predicted −0.399 and realised
  +0.627, and every try at that `λ` rose by over 3/2 of its predicted fall (`pvobs_damping_insufficient`).
- Measured (three runs, `pv_scales_carry_gain`): the run with one accepted V-step is not better than either run with
  none. Held-out KL after finishing has mean 0.8945 and spread 0.0075 (0.8906–0.8981), each run removing over a fifth
  of 1.1637. The 0.89 is the P-step on the scales, reproduced three times. Control `ctl_pvobs` (3/2 raised to 8/5, the
  mean bound raised to 0.8945) fails at both lines; axioms `{propext, Classical.choice, Quot.sound}`.
- Reading: no damping of the diagonal empirical Fisher makes a top-gain codeword move fall. Two readings fit, and
  neither is tested yet. (a) The empirical Fisher (squared gradients of the observed loss) is not the curvature:
  Kunstner, Balles and Hennig (arXiv 1905.12558, abstract retrieved) show that "the empirical Fisher—unlike the
  Fisher—does not generally capture second-order information". (b) The selected moves are not independent: top-gain
  blocks cluster in the same sensitive rows, and after the FWHT each 8-block spreads over 128 weights, so cross terms
  add coherently. pv05's one falling move was a low-gain set (skip-rank 0.97), which is consistent with both.
- Decision: stop scoring discrete moves. Train the codes the way `Perf/KDScaling.lean` trained a sign plane: a latent
  `S` per plane, projected in the forward pass to its nearest E8 ∪ O8 codeword with a straight-through gradient, jointly
  with the scales, at unchanged stored bits. The OBS path in `pv_finish.py` stays only until that run lands.
- Board row for these runs: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

### Straight-through code training from the codewords (`measured`, refuted; `Perf/PVFinish.lean`)

- Method: `pv_finish.py --codes ste` (sha256 `56a0eaa7…`). Each coded plane keeps an fp32 latent `L` initialised at
  its stored codewords `C[idx]`. The forward pass uses the nearest of the 16 codebook neighbours of the current index
  (`project_local`), with a straight-through gradient to `L`; the index is committed after each optimiser step.
  Latents sit in their own Adam group at `--lr-code`; scales keep their schedule; V-steps are off. All seven linear
  kinds train. Evaluation reads the stored integer codes, never `L`.
- Run `w18-pvste05.json` (RTX 5080, same protocol as `pv05-1`, all controls pass, pre-finishing KL reproduced bit for
  bit, bits unchanged at 1.17806 b/w, wall 542.2 s, 304 steps):

| metric | before | after |
|---|---:|---:|
| held-out KL `[50, 90)` | 1.1637 | 1.9991 |
| held-out top-1 | 0.480 | 0.340 |
| in-sample KL `[0, 20)` | 0.2224 | 1.9505 |
| blocks whose index changed | 0 | 13,081,296 of 50,061,312 (26.1%) |

- Prediction `pvste05-1` (held-out in `[0.65, 0.86]`) is refuted (`pvste_refuted`): held-out KL rose above 1.7× its
  start and in-sample KL more than eightfold. `lr-code` came from the batch-2 smoke: 1e-3 and 2e-2 moved no block, 0.1
  moved 4,543. At batch 8 the same rate moved 2.07M blocks (4.1%) in steps 1–25, and train loss rose 1.213 → 4.208
  over those steps (`pvste_flips_wholesale`). It then fell to 1.88 by step 300, still falling, without recovering the
  start.
- Reading: Adam divides each coordinate's step by its own gradient scale, so every latent moves about `lr` per step
  whatever its gradient. A latent that starts at a codeword centre carries no information about which side of the
  Voronoi boundary the weight wants, so the flips that cross are not ordered by the loss. `Perf/KDScaling.lean`'s
  sign plane trained because its latent started at the continuous weight `W`, so only weights near zero flipped
  first.
- Control `ctl_pvste` (17/10 raised to 18/10; 3× raised to 4×) fails at lines 178 and 191; axioms
  `{propext, Classical.choice, Quot.sound}`.
- Decision: start each latent at the plane's continuous pre-image: the teacher weight minus the other planes, mapped
  back through `compose` and `fwht128` into code coordinates. Keep the stored index until `L` is nearer another
  codeword by a margin, so the first flips are the blocks already on a boundary. The flip count per window is then set
  by the gradient, not by Adam's step size.
- Board row for this run: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

### Straight-through code training from the continuous pre-image (`measured`, refuted; `Perf/PVFinish.lean`)

- Method: `pv_finish.py --codes ste --code-init preimage --code-opt nsgd --lr-code 1e-3` (sha256 `e1e6af61…`). Plane
  `k` of a linear stores `W_k = fwht128(S·d) · r ⊗ c`, and `fwht128` is an orthonormal involution, so its pre-image in
  code coordinates is `P = fwht128((W_teacher − Σ_{j≠k} W_j) / (r ⊗ c)) / d`. Each 8-block's latent starts at
  `C[idx] + ρ (P − C[idx])`, with `ρ ≤ 1` clipped to 0.9 of the first crossing into any of the 16 neighbouring cells,
  so the step-0 projection is the stored code on every block. Latents move by plane-normalised SGD,
  `L ← L − lr · g / rms_plane(g)`, so a block moves in proportion to its gradient.
- Run `w18-pvpre05.json` (same protocol, all controls pass, before-KL exact, bits unchanged, wall 542.1 s, 292 steps):

| metric | before | after |
|---|---:|---:|
| held-out KL `[50, 90)` | 1.1637 | 0.9229 |
| held-out top-1 | 0.480 | 0.531 |
| in-sample KL `[0, 20)` | 0.2224 | 0.2590 |
| blocks whose index changed | 0 | 5,205 of 50,061,312 (1.04e-4) |
| of those, to the pre-image's nearest codeword | — | 1,941 (37.3%) |
| step-0 projection ≠ stored code | 0 | — |

- Prediction `pvpre05-1` (held-out in `[0.78, 0.88]`) is refuted (`pvpre_refuted`). The result lies above the three
  scales-only runs' mean 0.8945 by more than three times their spread. The flips were sparse and gradient-ordered,
  under one block in 5000 (`pvpre_flips_sparse`), and they still cost about 0.028 held-out KL.
- Pre-image diagnostic: the Euclidean-nearest codeword of `P` is the stored index on 39.8% of blocks, and 51.5% of
  latents were clipped. The ICM codes (`e8_joint_icm_2plane`, coupled Hessian metric) sit away from Euclidean rounding of
  the pre-image on most blocks, and the gradient sent most flips elsewhere too.
- Across five runs (`pv_code_moves_cost`), every run that changed a code index ended above the best run that changed
  none: one accepted linear V-step 0.8981, Adam straight-through 1.9991, pre-image straight-through 0.9229, against
  0.8906. At this token budget (1.2M tokens per 580 s run), the end-to-end loss has not been lowered by any measured
  code move away from the ICM codes (`measured`, five runs, one seed).
- Control `ctl_pvpre` (3× spread raised to 4×; 1/5000 tightened to 1/10000) fails at lines 213 and 224; axioms
  `{propext, Classical.choice, Quot.sound}`.
- Decision: at fixed codes, finishing has one lever left, the scales (mean 0.8945). The remaining distance to Q4_K_M
  (0.0184 on these chunks, 48×) is in the code's rate and its allocation. The K-FAC floor table above puts any
  post-training code on this model at 1.18 b/w at 16.9× Q4_K_M's KL with separable per-tensor water-filling and 6.6×
  with oracle variances (`model`: K-FAC quadratic KL, Gaussian rate; a different KL estimator from the measured
  held-out KL). The separable factor falls to 3.78× at Qwen3-32B. Next: re-allocate the arm's planes across the 168
  tensors by K-FAC water-filling at the same mean 1.178 b/w, then finish the scales on that arm. (Superseded by the
  ablation below; within-kind row reallocation was refuted, `w18_rowsel_resid_refuted`.)
- Board row for this run: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

### Which tunables carry the drop (`measured`; `Perf/PVFinish.lean`)

- Method: `pv_finish.py --train <groups> --v-every 0` (sha256 `48ababac…`). Unselected groups are frozen. The groups
  are `r` (per-(row, 128-block) scales, 3,128,832 latents through each row's frozen 16-level grid, 15,347,712 stored
  bits), `c` and `d` (per-column scales, 6 and 5 bits), and `norm` (RMSNorm gains and q/k/v biases, 71,552 parameters,
  zero stored bits). Every run fits the same 580 s box: same arm, before-KL exact, bits unchanged, all controls pass.

| receipt | groups | V-steps | steps | tokens | held-out KL | top-1 | in-sample |
|---|---|---|---:|---:|---:|---:|---:|
| before | — | — | 0 | 0 | 1.1637 | 0.480 | 0.2224 |
| `w18-pv05` / `pvobs05` / `pvlam05` | r,c,d,norm | every 25 | 432–450 | 1.77–1.84M | 0.8981 / 0.8906 / 0.8948 | — | 0.238–0.242 |
| `w18-pvaball05v0` | r,c,d,norm | off | 529 | 2.17M | **0.8824** | 0.542 | 0.2379 |
| `w18-pvabrcd05` | r,c,d | off | 540 | 2.21M | 0.8827 | 0.542 | 0.2366 |
| `w18-pvabcd05` | c,d | off | 548 | 2.24M | 0.9265 | 0.529 | 0.2342 |
| `w18-pvabr05` | r | off | 630 | 2.58M | 0.9091 | 0.536 | 0.2315 |
| `w18-pvabnorm05` | norm | off | 672 | 2.75M | 1.0758 | 0.491 | 0.2229 |

- V-steps off is the best finishing run: 0.8824, below each of the three V-step runs (`pvab_matched_best`). Without
  pauses for V-step scoring it took 529 steps against 432–450. One run; the margin under the best V-step run (0.0082)
  exceeds their spread (0.0075) and is under three times it.
- The scale groups substitute for each other. Against the joint `r, c, d` drop (0.2810), `r` alone removes 0.2546
  (over 9/10) and the column scales `c, d` alone remove 0.2372 (over 8/10) with 706,560 latents, under a quarter of
  `r`'s. The two single-group drops sum past 17/10 of the joint drop (`pvab_scales_substitute`). A per-(row, block)
  scale and a per-column scale both rescale the same stored weight, and either one recovers most of the gain.
- The norms add least: alone they remove 0.0879, under 1/3 of the all-group drop (`pvab_matched_r_share`), and once
  the stored scales train they add nothing: `r, c, d` with the norms frozen ends at 0.8827, within 1/1000 of the
  all-group run (`pvab_rcd_equals_all`).
- The match is in wall clock, not in tokens: single-group steps are cheaper, so `r` took 630 steps and the norms 672,
  against 529–548 for the multi-group runs.
- Predictions: `pvaball05v0-1` ([0.85, 0.90]), `pvabrcd05-1` ([0.86, 0.92]) and `pvabr05-1` ([0.90, 1.00]) confirmed;
  `pvabnorm05-1` ([0.95, 1.07]) refuted at 1.0758, below its 1.12 falsifier (`pvab_predictions`); `pvabcd05-1`
  ([0.98, 1.12]) refuted at 0.9265, past its 0.95 falsifier.
- Controls: tightening `pvab_predictions` 0.90 → 0.92, `pvab_matched_r_share` 9/10 → 91/100, `pvab_rcd_equals_all`
  1/1000 → 1/10000 and `pvab_scales_substitute` 8/10 → 85/100 each fails; axioms `{propext, Classical.choice,
  Quot.sound}` for all ten `pvab_*` theorems.
- Reading (`measured`, one run per group, 2.2–2.8M tokens): inside one 580 s box the scale family reaches about 0.88 by
  any route that includes `r` or `c, d`, and adding the other groups moves it by 0.03 to 0.05. The 0.88 is the box, not
  the family: held-out KL is still falling with tokens at 529 steps (next subsection). The finishing gain is a gain
  correction of the stored weights, low-dimensional enough that 706,560 column scales capture over 8/10 of it. What
  separates 0.88 from Q4_K_M's 0.0184 on these chunks lies outside what any measured scale group has moved.
- Structure: `r` is trained as `r0 · exp(pa)` and quantized onto the row's 16-level grid `lo + k/15 · sp`, which is fixed
  at the arm's values. The straight-through estimator passes gradient where the clamp saturates, so a latent can move
  past its grid with no effect on the weight. `fold_transfer.quant_r_hier` stores `4·O·G + 8·O + 64` bits for any grid
  it chooses (a 6-bit row mean index, a 2-bit spread index, four fp16 plane constants), so choosing the grid again from
  the trained latents costs zero bits.
- Decision: run the zero-bit grid re-selection (`pv_finish.py --regrid N`) against `w18-pvaball05v0` as the last
  scale-side lever, and the token-scaling points (146 and 292 steps) to read the slope at 529. The larger move is on the
  code side, where the arm's bits are: 400,490,496 of its 421,541,376 are sign bits.
- Board row for these runs: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

### Grid re-selection and token scaling (`measured`; `Perf/PVFinish.lean`)

- Method: `pv_finish.py` (sha256 `3cb9e4f7…`) with `--v-every 0 --train r,c,d,norm`, the `w18-pvaball05v0` protocol.
  `--regrid N` re-chooses each row's 16-level `r` grid from its trained latents every `N` steps (zero stored bits);
  `--steps S` completes the cosine schedule at `S` steps. Every run: same arm, before-KL exact, bits 421,541,376
  (1.17806 b/w), all controls pass.

| receipt | flag | steps | tokens | held-out KL | top-1 | in-sample | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| `w18-pvtok146` | `--steps 146` | 146 | 598,016 | 0.9653 | 0.519 | 0.2339 | 236 s |
| `w18-pvtok292` | `--steps 292` | 292 | 1,196,032 | 0.9195 | 0.532 | 0.2343 | 334 s |
| `w18-pvaball05v0` | — | 529 | 2,166,784 | 0.8824 | 0.542 | 0.2379 | 580 s box |
| `w18-pvgrid25` | `--regrid 25` | 519 | 2,125,824 | 0.8794 | 0.542 | 0.2397 | 542 s |

- The grid range does not bind: re-selection ends at 0.8794, 0.003 below the run without it and inside the 0.0075
  three-run spread; it moves 5.65% of `r` latents to a grid end, from 0 (`pvgrid_no_gain`).
- Finishing is still token-limited. Doubling tokens 146 → 292 steps drops held-out KL 0.0458; the next ×1.81 to 529
  steps drops 0.0371, over eight tenths of the previous doubling's drop in under one doubling (`pvtok_still_falling`).
  The 580 s box ends the run, not the scale family. In-sample KL stays at 0.234–0.240 across all three: the gain is
  generalisation of the gain correction, not in-sample fit.
- Fit (`model`, three points): `KL(T) = L∞ + A·T^(−α)` gives α ≈ 0.086, L∞ ≈ 0.17, poorly constrained; it projects 0.84
  at 4.3M tokens and 0.80 at 8.7M. L∞ is a fit parameter, not a bound.
- Predictions: `pvgrid25-1` ([0.86, 0.90]), `pvtok146-1` ([0.92, 0.98]) and `pvtok292-1` ([0.89, 0.93]) confirmed
  (`pvgrid_no_gain`, `pvtok_predictions`).
- Controls: tightening `pvgrid_no_gain`'s spread 75/10000 → 25/10000 and saturation bound 1/10 → 5/100,
  `pvtok_predictions` 0.92 → 0.97 and `pvtok_still_falling` 8/10 → 85/100 each fails; axioms `{propext,
  Classical.choice, Quot.sound}` for the three theorems.
- Decision: more tokens per box is a measured lever on the scale side (throughput: a step costs 0.79 s at 4096
  tokens, `419.1 s / 529`). The code side, where 400,490,496 of the 421,541,376 bits are, is the larger move: the `-guided` arm token
  (gradient-weighted Grams, GuidedQuant, Kim et al., arXiv 2505.07004 §3.2 Eq. 7) is in development.
- Board row for these runs: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

### End-loss-weighted Grams in the code solve (`measured`; `Perf/Guided.lean`)

- Method: `fold_transfer.py` token `-guided[a|b]NN` (sha256 `0a68d4ae…` for `w18-guided05`; `28b9b4c0…` for
  `w18-guidedsweep` and `w18-guidedseed`, which adds only the `a`/`b` part parsing). Each layer's Gram becomes
  `Σ_t a_t x_t x_tᵀ` with `a_t` the true-Fisher output-gradient energy at that layer (`K = 4` sampled targets,
  `gq_capture`, 9.9 s), and row weights `b` replace the arm's row proxy: GuidedQuant (Kim et al., arXiv 2505.07004 §3.2
  Eq. 7). `NN` tempers `a_t` to `a_t^(NN/100)`, mean 1; part `a` = token weights only, `b` = row weights only. The
  `--gq-controls` null control (`a_t = 1`) reproduces the base Grams and layers exactly. `--n-cal 50`, held-out
  `[50, 90)`, in-sample `[0, 20)`; bits 421,541,376 (1.17806 b/w) in every variant.

| variant | seed | held-out KL | in-sample | top-1 | Kish ESS min |
|---|---:|---:|---:|---:|---:|
| base | 0 / 1 / 2 | 1.1637 / 1.1884 / 1.1790 | 0.2224 / 0.2247 / 0.2230 | 0.480 / 0.470 / 0.476 | 1 |
| `-guided25` | 0 | 1.2977 | 0.2214 | 0.466 | 0.767 |
| `-guided50` | 0 / 1 / 2 | 1.1355 / 1.1361 / 1.2087 | 0.2520 / 0.2524 / 0.2534 | 0.479 / 0.481 / 0.473 | 0.070 |
| `-guided75` | 0 | 1.1539 | 0.2706 | 0.473 | 0.0053 |
| `-guided` (γ = 1) | 0 | 1.2032 | 0.3068 | 0.462 | 0.0023 |
| `-guideda50` | 0 | 1.2273 | 0.2513 | 0.470 | 0.070 |
| `-guidedb` | 0 | 1.3460 | 0.2210 | 0.465 | 1 |

- The weighting is a large, unordered perturbation of the code solve, not a smooth lever. The sign of the change
  alternates along `γ = 1/4, 1/2, 3/4, 1` (`guided_nonmonotone`); the Kish effective sample size `(Σa)²/(nΣa²)` does
  not order it: `γ = 3/4` keeps 0.5% at its worst stage and beats the base, `γ = 1/4` keeps 77% and loses by 0.13
  (`guided_ess_not_order`). Neither part alone beats the base (`guided_parts`).
- The base solve is stable across calibration order (seeds 0–2 within 0.025, `base_seed_spread`); `-guided50` widens
  that 2.96-fold (`guided_seed_spread`). Over three seeds its paired mean is 0.017 lower, two seeds better and one worse
  than every base seed (`guided_seed_mean`): not a robust gain at this protocol.
- In-sample KL rises with the weighting on every seed while held-out falls on two (`guided_insample_rises`): the
  weighted objective trades calibration fit for something else, as the scale finishing does. The finishing drop
  (1.16 → 0.88) is the larger and the stable lever.
- Predictions: `guided05-1` confirmed (`guided05_confirmed`); `guidedsweep-1` refuted, its 1.25 falsifier fires on
  `-guided25` and `-guidedb` (`guidedsweep_refuted`); `guidedseed-1` refuted on its first clause (base spread predicted
  `≥ 0.05`, measured 0.025).
- Controls: tightening `guided_seed_mean`'s bound −3/100 → −51/1000, `guided_nonmonotone` 13/100 → 14/100,
  `base_seed_spread` 25/1000 → 24/1000, `guided_parts` 18/100 → 19/100, `guided_ess_not_order` 3/4 → 8/10 and
  `guided_seed_spread` 29/10 → 3 each fails; axioms `{propext, Classical.choice, Quot.sound}` for all ten theorems.
- Decision: the code side keeps the unweighted Gram. Held-out falls only where the code is pulled off the in-sample
  optimum, and the stable way to do that is end-to-end scale finishing on held-apart tokens; its measured limit is
  tokens per box, so its throughput is the next lever.
- Board row for these runs: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

### Finishing throughput: one composition per step and a fused KL head (`measured`; `Perf/PVFinish.lean`)

- Method: `pv_finish.py` (sha256 `ce1095bf…`) with the `w18-pvaball05v0` protocol (`--v-every 0 --train r,c,d,norm`,
  4096 tokens a step, 580 s box) and two path flags. `--compose step`: `PlaneCompose` composes each coded plane once
  per step outside the layer checkpoints, with an analytic backward, in place of composing in the checkpoint's forward
  and again in its recompute. `--kl fused`: `KLHead` saves the student-logit adjoint `(softmax(z_s) − p_t) E` in the
  forward, so no vocabulary-head matmul is recomputed. Defaults `(layer, ckpt)` are the old path. Same arm, bits
  421,541,376 (1.17806 b/w), before-KL exact, all controls pass. Peak memory 6.68 GB.
- Cost model (`proved`, `pvfast_cost_model`): 16 KL chunks of 256 rows a step; head matmuls 80 → 48 a step (`5 → 3` per
  chunk), removing `2.23 · 10¹²` FLOPs a step; composition forwards per plane 2 → 1.
- Gradient control (`grad_equiv`, two calibration chunks): losses bit-equal on every path; trainable-gradient relative
  L2 against `(layer, ckpt)` is 5.81e-3 for `(step, fused)`, 4.62e-3 compose-only, 6.07e-3 KL-only, against a rerun
  noise of 4.34e-3 for the same path twice (`pvfast_grad_floor`). The fast path computes the same gradient to within
  1.34× the nondeterminism of the old path against itself.

| receipt | path | steps | tokens | s/step | held-out KL | top-1 | in-sample |
|---|---|---:|---:|---:|---:|---:|---:|
| `w18-pvaball05v0` | `(layer, ckpt)` | 529 | 2,166,784 | 0.792 | 0.8824 | 0.542 | 0.2379 |
| `w18-pvfast` | `(step, fused)` | 729 | 2,985,984 | 0.571 | **0.8577** | 0.548 | 0.2398 |

- 1.39× faster a step on the same box (`pvfast_faster`); 1.38× the tokens; held-out KL 0.0247 lower, over three times the
  0.0075 three-run spread, top-1 up (`pvfast_lower`). From 1.1637 before finishing, the drop is 0.306.
- Token trend, four points (`pvtok_trend_four`): 0.9653 / 0.9195 / 0.8824 / 0.8577 at 146 / 292 / 529 / 729 steps. The
  last step is under half a doubling of tokens (`729² < 2 · 529²`) and drops more than half the first doubling's 0.0458:
  drop per doubling 0.046, 0.043, ≈ 0.053. It does not flatten at 3.0M tokens; in-sample stays at 0.234–0.240, so the
  gain stays held-out generalisation of the scale correction.
- Ten-step `torch.profiler` windows (`w18-pvprof`, `w18-pvprof2`): elementwise is
  `aten::{mul, cat, sub, add, add_, copy_, fill_, index, gather}` self device time.

| receipt | path | profiled s/step | device s/step | elementwise | `aten::mm` | `cat` calls |
|---|---|---:|---:|---:|---:|---:|
| `w18-pvprof` | `(layer, ckpt)` | 2.18 | — (op + kernel double-counted) | 3.309 s of 6.219 s aten | 2.514 s | 41,760 |
| `w18-pvprof2` | `(step, fused)` | 1.33 | 0.527 | 2.234 s, 42.4% of 5.270 s | 2.126 s | 61,920 |

- Where the time went (`pvprof_refuted`, `pvprof_fast_path`): on the old path the elementwise ops of plane composition
  were over half the aten device time and above every matmul together; the vocabulary head was not the cost. The fast
  path cuts elementwise under 7/10 and matmul time. The `cat` count rises because the analytic backward runs `fwht128`
  twice per plane: the 7-stage butterfly is 61,920 `cat` + 60,480 `stack` calls per 10 steps, and `cat + sub + add` is
  ≈ 1.21 s, 23% of device time. Kernel time 0.527 s/step covers over 9/10 of the unprofiled 0.571 s: the step is
  device-bound, so the next gain must remove device work.
- Predictions: `pvprof-1` refuted (profiled step predicted `[0.4, 1.6]` s, measured 2.18; `pvprof_refuted`);
  `pvfast-1` withdrawn (its one admitted command was the 20-step smoke run; the full run was re-registered as
  `pvfast-2` with the rerun-noise gradient control); `pvfast-2` refuted on its throughput ranges (s/step predicted
  `[0.30, 0.50]`, steps `[800, 1300]`), its held-out KL inside the predicted `[0.83, 0.875]`, falsifier not fired
  (`pvfast_refuted`); `pvprof2-1` refuted (profiled step above 1.2 s, elementwise share above 35%), kernel time inside
  `[0.35, 0.55]` s and elementwise under the falsifier's 50% (`pvprof2_refuted`).
- Controls: `pvfast_cost_model` FLOPs `…904 → …905`, `pvfast_same_arm` V-steps 0 → 1, `pvfast_grad_floor` 11/10 → 1,
  `pvfast_faster` 13 → 14, `pvfast_refuted` 1/2 → 3/5, `pvfast_lower` 2/100 → 3/100, `pvtok_trend_four` 2 → 1,
  `pvprof_refuted` 2 → 3, `pvprof2_refuted` 6/5 → 7/5 and `pvprof_fast_path` 9/10 → 95/100 each fails at its own line;
  axioms `{propext, Classical.choice, Quot.sound}` for all ten (`{propext}` for the cost model). `lake build
  Perf.PVFinish` exit 0 (777 jobs).
- Decision: throughput converts directly into held-out KL at fixed bits, and the curve is still falling. The next device
  work to remove is the butterfly: `fwht128` as one `x.reshape(-1, 128) @ H₁₂₈/√128` matmul (Sylvester's `H₁₂₈` is
  symmetric and its own inverse, so it serves forward and backward), then the Hadamard folded onto activations
  (`x Wᵀ = (d ⊙ H(c ⊙ x)) (S ⊙ r)ᵀ` per 128-column group), which removes weight composition from the step and is the
  inference kernel's shape.
- Board row for these runs: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

Board row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

### The Hadamard as one matmul (`measured`; `Perf/PVFinish.lean`)

`pv_finish.py --hadamard mm` (sha256 `60b0f6bc…`) replaces each 128-group's 7-stage `fwht128` butterfly with one
`x.reshape(-1, 128) @ H₁₂₈/√128` matmul (Sylvester's `H₁₂₈ = H^⊗7`, built in float64 and cast; symmetric and its own
inverse, so it serves the forward and both adjoints). Everything else is `w18-pvfast`'s protocol. Receipt
[`receipts/onebit/w18-hadmm.json`](receipts/onebit/w18-hadmm.json) (sha256 `bae8ada3…`), RTX 5080, 580 s budget.

| run | Hadamard | steps | train s | s/step | tokens | held-out KL | top-1 | in-sample KL | peak GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `w18-pvfast` | butterfly | 729 | 416.2 | 0.571 | 2,985,984 | 0.8577 | 0.548 | 0.2398 | 6.68 |
| `w18-hadmm` | matmul | 871 | 417.4 | 0.479 | 3,567,616 | **0.8484** | 0.551 | 0.2419 | 8.05 |

Arm, bits (1.178061 b/w) and the before-finishing point (held-out 1.1637, top-1 0.480) are identical.

- Cost model (`hadmm_cost_model`): the butterfly reads and writes every fp32 word at each of 7 stages plus the `1/√128`
  scale, 64 B per word; the matmul moves 8 B per word, eight times fewer, and spends 256 FLOPs per word (32 per byte) on
  the tensor cores. The measured saving is between 0.09 and 0.10 s per step (`hadmm_faster`), against ≈ 0.121 s of
  butterfly `cat + sub + add` profiled on `pvfast`; the step is 1.1–1.2× faster. Peak memory rises 1.37 GB.
- Same function (`hadmm_same_function`): in float64 on eight coded planes the forward and the `r, c, d` adjoints agree to
  `7.13 · 10⁻¹⁶` relative. With an fp32 student the model loss moves `3.24 · 10⁻⁶` relative and the trainable gradient
  `1.344 · 10⁻³`, under ten times the fp32 rerun noise `1.68 · 10⁻⁴` (atomic reordering alone).
- bf16 amplification (`hadmm_bf16_amplification`): the matmul moves no fp32 weight by more than `8.7 · 10⁻⁷` relative
  and its bf16 relative L2 change (`2.55 · 10⁻⁵`) is below a seeded `2⁻²²` jitter's (`3.59 · 10⁻⁵`), yet it flips
  291,588 bf16 words against the jitter's 21,296 (33,800 zero → nonzero): its error is absolute per 128-group, so it
  lands on small words. The bf16 loss moves `2.93 · 10⁻³` relative, over 800 times the fp32 change and below the
  bf16-vs-fp32 cast's own `4.50 · 10⁻³`.
- Lower (`hadmm_lower`): held-out KL 0.8484, 0.0093 below `pvfast`, over the 0.0075 protocol spread; top-1 rises. The
  step 729 → 871 is over a quarter of a token doubling and its drop is under a quarter of the 146 → 292 doubling's: the
  fifth token point lies below the earlier per-doubling rate.
- Predictions: `hadmm-smoke-1` refuted (bf16 gradient tolerance `1/100` guessed; measured `3.46 · 10⁻²`);
  `hadmm-smoke-2-1` refuted (bf16 flips 13.3× the jitter's, over the 3× falsifier); `hadmm-full-1` refuted (loss 5.9×
  the jitter's; no training ran); `hadmm-full-2` refuted (fp32 gradient `1.336 · 10⁻³` over the guessed `10⁻³`); each
  was the control's assert, before any training step. `hadmm-full-3` confirmed (s/step, steps and held-out KL inside
  their predicted ranges). `hadmm_guessed_tolerances` records the two guessed thresholds the measurements exceed.
- Lesson, now in the control: the identity is checked in float64 (`< 10⁻¹²`); fp32 and bf16 tolerances derive from the
  measured fp32 rerun noise and a seeded jitter in the same run, never from a guessed threshold.
- Controls: `hadmm_cost_model` 8 → 9, `hadmm_same_arm` pvfast `"butterfly"` → `"mm"`, `hadmm_same_function` `10⁻⁵` →
  `10⁻⁶`, `hadmm_bf16_amplification` 13 → 14, `hadmm_guessed_tolerances` `1/1000` → `1/100`, `hadmm_faster` `9/100` →
  `1/10` and `hadmm_lower` `75/10000` → `100/10000` each fails at its own theorem; axioms `{propext, Classical.choice,
  Quot.sound}` for six, `{propext}` for the cost model.
- Decision: `--hadamard mm` is the finishing default. Next device work: fold the Hadamard onto activations
  (`x Wᵀ = (d ⊙ H(c ⊙ x)) (S ⊙ r)ᵀ` per 128-column group), which removes weight composition from the step; the learned
  K-angle rotation of the quantum epic generalises this fixed `H^⊗7`.
- Board row for this run: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4`.

### Input basis versus variable rate at 1.18 b/w (`model`; `src/learn/fold/stats/basis.rs`)

Question: once the coder may vary its rate per coefficient, does an input basis that concentrates the weight's energy
beat the Hadamard that flattens it? Under GPTQ feedback the metric weights are the `LDLᵀ` pivots `d_j` of `QᵀHQ`, and
`Π d_j = det H` for every orthogonal `Q` (test `b_pivot_product_is_basis_invariant`), so a basis acts only through the
coefficient density. Reverse water-filling at `λ_ij = d_j σ²_ij`, 1.18 b/w all-in, side information counted (16 bits
per row and column for the separable density, 8 per 16×16 tile, 8 per butterfly angle). Bases per 128-column block:
`H` Hadamard, `I` identity, `J` Jacobi-adapted butterfly, `V` dense eigenbasis of `WᵀW` (side information not counted:
an upper reference). Qwen2.5-0.5B, layers {2, 11, 20} × {q, k, v, o, gate, up, down}, Grams from 512 C4 tokens on the
crate forward. Command `cargo test --release --locked --test learn_fold_basis basis_rate_qwen05b -- --ignored
--nocapture`; per-tensor table `/home/socol/oc-tmp/escalation/Q-MAD1.live.log`.

| density | `D_u(H)` fixed rate | `D_wf(H)` | `D_wf(I)` | `D_wf(J)` | `D_wf(V)` | oracle `H` / `J` |
|---|---:|---:|---:|---:|---:|---:|
| separable | 346.3 | 200.8 | 196.9 | 198.8 | 183.6 | 76.5 / 74.9 |
| 16×16 tile | 348.4 | 212.1 | 211.6 | 215.2 | 195.4 | 76.5 / 74.9 |

- Basis: `D_wf(H)/D_wf(J)` = 1.010 (separable) and 0.986 (tile); the identity matches the Hadamard to 2%. The dense
  eigenbasis with free side information gains 8–9%. Concentrating the input basis does not move the distortion in this
  model; the Madelung-split rotation is closed as a rate lever.
- Rate: `D_u(H)/D_wf(H)` = 1.72 (separable) and 1.64 (tile), ≈ 0.39 and 0.36 b/w at `½ log₂` per halving of
  distortion. The gain is the variable rate itself, in any of the four bases.
- Oracle: per-coefficient variance is a further 2.6× (≈ 0.69 b/w), the per-coefficient term of the K-FAC section above.
- Controls: Jacobi angle sign flipped → `d_adapted_butterfly_concentrates_a_planted_subspace` red; side bits dropped
  → `a_water_filling_closed_form_and_side_bits` red; `d_j := M_jj` → `b_pivot_product_is_basis_invariant` red.
- Not covered: the Gaussian rate model is not a coder; output-side rotations; held-out KL.

## Limit claims

`tools/rooting/limit_check.py` refuses, in this file, `onebit-w13.md`, `docs/epics/ternary-hadamard.md` and the
docstrings of `Perf.CostCorrespondence`, `Perf.KLTilt` and `Perf.BitsRefine`, a limit word (converse, floor, lower
bound, cannot reach, unreachable, impossible, unprecedented, state of the art) without a grade label or a cited theorem
with no hypothesis binder. A registered model floor beaten by a committed receipt must be marked `refuted`: the
uniform-rate separable K-FAC floor 0.3887 is `refuted` by the `-nb` in-sample mean 0.35064. `--staged` applies the
same rule to every added line in `docs/` and `book/`. The checker establishes that a grade is stated, not that it is
correct.

## Not covered (W18)

- Outcome metrics are one seed (seed 0) on one calibration order; the 3-seed held-out KL range of the arm is 0.011, so
  the outcome spread across seeds is not measured. The sparse inference pattern is not evaluated.
- No bridging measurement against the perplexity literature at the usual 2048-token window.
- Papers read: those in the receipt. The ~40-paper scope is not a survey; Proteus was not found on arXiv.
