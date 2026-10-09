# HATTN-T1: one-bit attention on Qwen2.5-0.5B q, k

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Spec: `docs/epics/ternary-hadamard.md` Part III. Gate: `python3 tools/hattn/qk_onebit.py --check` and `--full`.

## Prediction (written before the first full run)

- **P1.** Arm A at `b = 64` (R) has a median held-out per-head KL above 0.05 nats. The sign map discards `‖k‖`, and the attention-sink heads depend on it.
- **P2.** Arm B lowers arm A's median by at least 2× at `b = 64`.
- **P3.** At `b ≥ 16`, the distinct key codes equal `t` at `t = 512` in at least 90% of kv heads, so the exact `2^b` carry does not pay at T = 512. At `b = 8` the codes collide.
- **P4.** Arm D at `r = 3` has a larger median TV than arm A at `b = 16`.

## Result (`measured`, one prompt, one model)

Command (cwd `apeiron`, host i9-14900KF x86_64, numpy 2.5.3, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, under `systemd-run --user --scope -p MemoryMax=9G -p MemorySwapMax=0`, `timeout 570`):

```sh
python3 tools/hattn/qk_onebit.py --check   # 0.1 s, exit 0
python3 tools/hattn/qk_onebit.py --full    # 79 s (78.9, 79.6, 79.8), exit 0; three runs byte-identical
```

Source sha256 `1f298ba28d00ed209b7a45549b6af4ac0839e4dc9bb128146156ec799e5fb2e1`; `--full` output sha256 `e55879ab995f70e9f58b640f…`. Data: Qwen2.5-0.5B `Q4_K_M`, prompt 0, `T = 512`, 24 layers, 14 query heads, 2 kv heads (GQA 7:1), `d = 64`. 336 heads. Scale fitted on rows `t < 256` over a 64-point log grid, reported on held-out rows `t ≥ 256`. `R` = normalized Walsh–Hadamard times a fixed ±1 diagonal (seed 0); `P_b` = first `b` rows of a Gaussian 64×64 (seed 1).

### Held-out per-head KL(float ‖ arm), nats

| arm | definition | median | p90 | TV median |
|---|---|---:|---:|---:|
| A `P` b=8 / 12 / 16 / 32 / 64 | `s · Σ sign(P_b q̃) sign(P_b k̃)` | 2.894 / 2.733 / 2.585 / 2.160 / 1.764 | 4.53 / 4.37 / 4.14 / 3.67 / 3.24 | 0.806 / 0.793 / 0.786 / 0.736 / 0.662 |
| A `R` b=64 | same, through `R` | **1.664** | 3.21 | 0.641 |
| B `R` b=64 | `c ‖k̃‖/mean‖k̃‖ · Σ` (key norm multiplied in) | 2.381 | 3.72 | 0.769 |
| C | float, query norm removed, scale unfitted | 0.0093 | 0.022 | 0.042 |
| E | float, key norm removed, scale fitted (extension) | 0.767 | 1.88 | 0.409 |
| F | float cosine only, scale fitted (extension) | 0.761 | 1.90 | 0.431 |
| G `R` b=64 | `g Σ + β log(‖k̃‖/mean‖k̃‖)`, additive key bias (extension) | 1.619 | 3.11 | 0.632 |
| D r=1 / 2 / 3 | b=64 through `R`, degree-`r` truncation | TV 0.811 / 0.767 / 0.746 | TV p90 0.96 / 0.91 / 0.95 | |

Arm D's weights are signed; the median fraction of negative weights is 0.00–0.03 and p90 0.05–0.11.

### Distinct key codes among the first `t` keys (median per kv head | fraction of kv heads with all distinct)

| proj, b | t=64 | t=128 | t=256 | t=512 |
|---|---|---|---|---|
| P 8 | 22 / 0.00 | 34 / 0.00 | 50 / 0.00 | 66 / 0.00 |
| P 12 | 38 / 0.00 | 66 / 0.00 | 116 / 0.00 | 180 / 0.00 |
| P 16 | 50 / 0.00 | 96 / 0.00 | 172 / 0.00 | 311 / 0.00 |
| P 32 | 63 / 0.42 | 126 / 0.17 | 251 / 0.10 | 502 / 0.04 |
| P 64 | 64 / 0.85 | 128 / 0.77 | 256 / 0.75 | 512 / 0.62 |
| R 64 | 64 / 0.88 | 128 / 0.83 | 256 / 0.81 | 512 / 0.75 |

### Verdicts on the registered predictions

| | verdict | numbers |
|---|---|---|
| P1 | held | A b=64 (R) median 1.664 > 0.05 |
| P2 | **refuted** | B/A = 2.381/1.664 = 1.431; B is worse than A at every `b` (the direction is opposite to the prediction) |
| P3 | **refuted** | distinct = 512 at `t = 512` in 0.00 (b=8, 12, 16), 0.04 (b=32), 0.62 (b=64, `P`), 0.75 (`R`) of kv heads. Collisions are heavy at `b ≤ 16` (311 of 512 distinct at b=16), so the exact `2^b` carry at `b ≤ 16` is a lossy hash, not an identity map; the prediction expected 90% distinct at `b ≥ 16` |
| P4 | **refuted** | D r=3 TV 0.7461 < A b=16 TV 0.7861 |

### Decision

No sign arm reaches 0.05 nats. Best is A b=64 (R) at 1.664. Attribution of the loss:

- Query norm is negligible: C 0.009.
- Key norm: removing it from the float scores costs 0.767 even with exact angles (E ≈ F 0.761).
- The sign map adds about 0.9 nats on top: A(R64) 1.664 − F 0.761.
- Carrying the key norm as a multiplicative factor (B) or as an additive log bias (G, 1.619 vs A 1.664) does not recover it. The fitted `β` is 0 in 13.7% of heads and at a grid end in 14.3%. Why B degrades is a hypothesis (noise amplification), untested.
- Layer 0 is the worst case for every sign arm (about 3.3 nats): float scores reach about 1740 and key norms about 250, so attention is near one-hot. Checked, not a bug.

The compatibility question (existing weights through a fixed sign projection) is answered no at this grade. Training or fine-tuning the projection (HATTN-T3) is not tested here.

Extensions E, F, G and the collision table at `b = 32, 64` go beyond the registered arms; the registered arms A, B, C, D are unchanged.

### Controls

| control | result |
|---|---|
| (i) identity in place of the sign map, layer 0, `s = 1/8`, `R` on both sides | max KL 5.6e-13 (< 1e-10); `‖RRᵀ − I‖` = 0.0 |
| (ii) exact `2^8` carry against direct softmax attention, `n = 1000`, `d = 8`, `s = 0.37`, `d_v = 4` | max diff 1.9e-16 (< 1e-12) |
| (iii) arm D at `r = 64` against the arm A kernel, `b = 64`, `s = 0.05` | max diff 1.7e-18 (< 1e-9) |
| mutation: negate `λ_3` in `carry_lambda` (throwaway copy `/home/socol/oc-tmp/hattn-t1-ctl/`) | `--check` exit 1, `FAIL check (ii)`, diff 1.39e-02; (i) and (iii) unaffected |
| restored source | `--check` exit 0; copy minus the mutation line equals the source |

Exact finite carry (`model`): `2^d (d_v + 1)` numbers for `q, k ∈ {±1}^d`; `d = 64` is infeasible and the `d = 8` check (ii) is the only instance run. Lean: `Nous.Model.OneBit*` (HATTN-T2, closed).

## Prior art (retrieved abstracts, `export.arxiv.org` / `arxiv.org/abs`, not full texts)

| id | quoted sentence | relation to `e^{s q·k} = Σ_S cosh^{d−|S|} s sinh^{|S|} s q_S k_S` |
|---|---|---|
| 2302.13214 Fast Attention Requires Bounded Entries | "We present two results, showing that there is a sharp transition at B = Θ(√log n)." | complexity bound on exact softmax attention at bounded entries; does not give a kernel. The ±1 case here has `B = s·d`. Different object |
| 2402.04497 Fine-Grained Complexity of Gradient Computation for Training LLMs | "In this work, we show nearly identical results for the harder-seeming problem of computing the gradient of loss function of one layer attention network..." | lower bound for the training pass; not a kernel. Different object |
| 2402.18668 BASED | "We propose BASED a simple architecture combining linear and sliding window attention." | Taylor-feature linear attention: a truncated polynomial kernel in `q·k`, degree-graded like arm D but over real scores, not the `tanh^{|S|}` basis. Related (truncation), not the same expansion |
| 2009.14794 Performers | "To approximate softmax attention-kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+)" | random-feature approximation of `e^{q·k}`; this expansion is the exact finite-rank decomposition on `{±1}^d`. Different kernel construction, same target |
| 2001.04451 Reformer | "We replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L²) to O(L log L)" | sign-of-random-projection hashing selects keys; arm A with `P_b` uses the same sign hash as a score, not a bucket selector. Related |
| 2509.05218 HoPE | "We propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations." | positional encoding with Lorentz rotations; no kernel expansion. Different object |
| binary/hashed sign attention | two `export.arxiv.org` queries returned 0 hits, then two unrelated image-hashing papers (2210.07572, 1702.00758) | NOT FOUND; no relation recorded |

No novelty is claimed. The expansion is the Walsh–Hadamard decomposition of `exp(s q·k)` on the hypercube; its prior art outside these pointers was not searched beyond the queries above.

## Does not cover

One prompt, one model, `T = 512`, `Q4_K_M` weights dequantized to f32 for the dumps, 24 layers of q, k only (no values, no whole-model output, no perplexity). The scale is fitted per head on the first half and read on the second, so a later-context shift is in the held-out number. Fixed projections only: no trained projection. Arm B's explanation, the absolute sign-attention floor on other models, and every speed claim are untested. numpy float64 arithmetic; no rustc/LLVM, kernel or GPU layer is involved.
