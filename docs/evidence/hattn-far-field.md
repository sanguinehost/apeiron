# HATTN-T3: attention as a screened N-body sum at 32k context

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Grade: `measured` (one prompt, deterministic
numpy over a fixed dump); verdicts `proved` as arithmetic over the measured constants in
`proof/corpus/Perf/HattnT3.lean`.

## Data

| item | value |
|---|---|
| model | `/home/socol/models/gguf/qwen05b-Q4_K_M.gguf`, 24 layers, 14 query heads, 2 KV heads, head dim 64 |
| prompt | `prose8.txt[0..]` contiguous, 32,768 tokens, ids sha256 `a9ff4f03a848b181…`; first 512 ids equal T1 prompt 0 |
| dump | `KV_DUMP=… KV_T=32768 examples/attn_cut_probe.rs` (`run_dump`), unrotated q/k/v per layer, f32; `/home/socol/oc-tmp/hattn-long/t32k` (3.6 GB), 327 s |
| binding | `attn_cut_probe.rs` sha256 `b4e63bbb…`, binary `dc326fee…` (`/home/socol/oc-tmp/hattn-long/binding.sha256`) |
| dump control | `KV_T=512 QK_REF=/home/socol/oc-tmp/faces-qk`: layer 0 q/k byte-identical to the T1 dumps; layers 1–23 rel-L2 ≤ 4.5e-7 (f32 AVX attention here, f64 in T1) |
| instrument | `tools/hattn/far_field.py --full`, 532 s; output `/home/socol/oc-tmp/hattn-long/far_field.out` |
| instrument controls | `--check`: Gaussian cloud second-cumulant field `|ΔlogZ|` 1.1e-4, value rel-L2 7.1e-4 (first cumulant 0.12); bimodal cloud `|ΔlogZ|` 9.1 (must fail, fails); sign-flipped second cumulant `|ΔlogZ|` 0.34 (must fail, fails) |
| predictions | registered before the run: `/home/socol/oc-tmp/hattn-long/PREDICTIONS.md`, sha256 `4cb4627c…` |

Queries: the last 8 positions before `t ∈ {512, 2048, 8192, 32768}`, all 336 (layer, head). Scale `s = 1/8`.
`ε = 1 − e^{−0.05}`: keeping the top keys holding `1 − ε` of the mass and renormalising is `KL = 0.05` exactly.

Identity under test: `s q·k = s|q|²/2 + s|k|²/2 − s|q−k|²/2`. The query term cancels in softmax, so attention
is a sum over particles of mass `e^{s|k|²/2}` screened by `e^{−s|q−k|²/2}`.

## Results

| prediction | registered | measured | verdict |
|---|---|---|---|
| P-PH1 concentration | median growth exponent of `k_ε(t)` ≤ 0.5 and ≥ 20 % of heads above 0.8 | median 0.394; 19.6 % above 0.8 | refuted (second clause misses by 0.4 points) |
| P-PH2 Gaussian far field | second-cumulant output rel-L2 ≤ 0.10 at 32k, ≥ 2× better than first cumulant | 0.345; first cumulant 0.104 | refuted, the second cumulant is 3.3× worse |
| P-PH3 mass vs screening | mass share of score variance ≤ 0.30 | 0.338 pooled (0.295 at 32k alone) | refuted, screening still carries ~2/3 |
| P-PH4 RoPE phase | coherence ≤ 0.1 for every `j < 16`, ≥ 0.5 for every `j ≥ 24` | `j < 16` max 0.027; `j ≥ 24` 0.37–0.74 | refuted; the high-frequency half holds |

Concentration (`k_ε`: median over queries, then over heads):

| t | `k_ε` median | `k_ε` p90 | `k_ε/t` | N_eff median |
|---:|---:|---:|---:|---:|
| 512 | 70 | 252 | 0.137 | 15.2 |
| 2048 | 57 | 461 | 0.028 | 12.2 |
| 8192 | 109 | 1064 | 0.013 | 25.9 |
| 32768 | 161 | 2377 | 0.0049 | 20.0 |

Far field (near = 4 sinks + last 256 keys; output rel-L2 against exact, median):

| t | drop far | first cumulant | second cumulant | second cumulant per 2048-key block | `|ΔlogZ_far|` (second) |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.046 | 0.041 | 0.059 | 0.059 | 0.65 |
| 8192 | 0.096 | 0.094 | 0.284 | 0.217 | 1.66 |
| 32768 | 0.106 | 0.104 | 0.345 | 0.230 | 1.82 |

## Findings

- **Mass concentrates sublinearly.** At 32k the median head reaches KL 0.05 with 161 keys (0.49 %), and
  `k_ε ∝ t^0.39` at the median. The tail does not: p90 needs 2,377 keys, and 66 of 336 heads grow at
  α > 0.8 (median α per layer: 0.84 at layer 3, 0.86 at layer 17, 0.73–0.74 at layers 8 and 21).
- **The far set is not a Gaussian cloud.** The moment summary is exact for Gaussian keys (control) and
  fails here by 1.8 nats of log partition; it is worse than dropping the far set. Two theorems state the
  mechanism: `first_cumulant_le` (Jensen; the first-cumulant field can only under-weight the far set, so it
  tracks truncation, 0.104 against 0.106) and `second_cumulant_overshoots` (the variance term is no bound;
  a heavy-tailed score cloud drives it arbitrarily far). Splitting the far set into 2048-key blocks recovers
  a third of the loss (0.345 → 0.230): the cloud is multimodal, not merely wide.
- **RoPE randomises the high-frequency half of the far keys.** Pairs `j < 16` average to coherence ≤ 0.027
  over 31,743 keys: their mean over distant keys is zero. The low-frequency pairs keep 0.37–0.74 coherence.
  Zero mean is not zero contribution: ranking far pages on the `j ≥ 16` pairs alone needs 3.3× the pages of the
  full bound ([HATTN-T4](hattn-page-select.md), PQ3), so the high-frequency pairs still separate the keys that
  matter.
- **Relation to ATTN-CUT.** `ddfc30e` and `3c9b5b0` certified an absolute output error ≤ `εV` with
  `ε ∈ {1e-3, 1e-4, 1e-5}` and needed 54–92 % of keys. The KL-0.05 target needs 0.49 % at the median.
  That earlier negative result was for a tolerance about two orders of magnitude tighter; at KL 0.05 it
  does not transfer.

## What this does not establish

- The `k_ε` keys come from an oracle sort over every score. Finding them costs at least one full scan
  until a sublinear selector is measured; no cost saving is claimed.
- One prompt (concatenated web documents). Per-head attention KL, not whole-model KLD.
- No Rust or Aeneas path; numpy over a dump. rustc, LLVM and numpy are outside the trust boundary.

## Next

The far field cannot be summarised by moments, so the keys have to be retrieved: measured in
[HATTN-T4](hattn-page-select.md) (page-bound selection, Quest).
