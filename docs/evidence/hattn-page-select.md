# HATTN-T4: keys read by page-bound selection at 32k context

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Grade: `measured` (one prompt, deterministic numpy
over the [HATTN-T3](hattn-far-field.md) dump); verdicts and the bound's soundness `proved` in
`proof/corpus/Perf/HattnT4.lean`.

## Method

| item | value |
|---|---|
| dump | `/home/socol/oc-tmp/hattn-long/t32k`, Qwen2.5-0.5B Q4_K_M, ids sha256 `a9ff4f03a848b181…` (HATTN-T3) |
| selector subsumed | Quest, Tang et al., arXiv 2406.10774v2 §3, retrieved to `/home/socol/oc-tmp/refs/quest-2406.10774.pdf` (sha256 `93cda144…`): per page, element-wise min `m` and max `M` key; page bound `Σ_i max(q_i m_i, q_i M_i)`; pages read in bound order |
| instrument | `tools/hattn/page_select.py --full` (sha256 `1f5d1c9c…`), 22 s; output `/home/socol/oc-tmp/hattn-long/page_select.out` (sha256 `6155f477…`) |
| instrument controls | `--check`: bound ≥ page max on 256 random pages, tree bounds monotone, a planted needle ranked first (best-first reads 82 of 4096 keys, flat 528); `HATTN_MUT=1` (min-only bound) fails soundness and monotonicity, exit 1 |
| predictions | registered before the run: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T4.md`, sha256 `3407a185…` |

Queries: the last 8 before `t ∈ {2048, 8192, 32768}`, 336 heads. Target: kept mass ≥ `1 − ε`, `ε = 1 − e^{−0.05}`,
i.e. KL 0.05. Always read: 4 sinks, the last 256 keys, the partial far page. Cost is key-vector reads per query
(decode is memory bound): a page bound reads 2 vectors (min, max), a selected page reads `P`.

| arm | ranking |
|---|---|
| A | Quest, 16-key pages, bound over all 64 dimensions |
| A64 | Quest, 64-key pages |
| B | 16-key pages, bound over the RoPE pairs `j ≥ 16` only (1 vector-equivalent per page) |
| H | best-first over a fanout-16 tree of boxes (16, 256, 4096 keys): pop the highest bound, expand it |
| OP | 16-key pages in true mass order: the floor set by how keys clump into pages |

The adaptive arms stop on the oracle kept mass, so they measure ranking quality, not a deployable stop rule.

## Results

| prediction | registered | measured at 32k | verdict |
|---|---|---|---|
| PQ1 | A selects ≤ 1024 far tokens (median) | 400 | held |
| PQ2 | A reads ≤ `0.20 t` | `0.144 t` | held |
| PQ3 | B needs ≤ A's pages | 82.5 against 25 | refuted, 3.3× worse |
| PQ4 | fixed 2048-token budget meets KL 0.05 for ≥ 90 % | 65.4 % | refuted |
| PQ5 | A's selected tokens grow as `t^α`, `α ≤ 0.5` | median α 0.750 (p90 2.41) | refuted, still sublinear |
| PQ6 | H reads ≤ `0.10 t` | `0.146 t` | refuted |

Selected far tokens (median over layer, head, query):

| t | oracle `k_ε` | A | A64 | B | OP |
|---:|---:|---:|---:|---:|---:|
| 2048 | 55 | 0 | 0 | 0 | 0 |
| 8192 | 110 | 304 | 384 | 640 | 128 |
| 32768 | 167 | 400 | 576 | 1320 | 152 |

Total reads / `t`, median (p90):

| t | A | A64 | B | H | OP |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.241 (0.763) | 0.182 (0.745) | 0.187 (0.774) | 0.134 (0.771) | 0.133 (0.539) |
| 8192 | 0.191 (0.566) | 0.116 (0.537) | 0.172 (0.593) | 0.197 (0.574) | 0.048 (0.327) |
| 32768 | 0.144 (0.449) | 0.058 (0.424) | 0.111 (0.492) | 0.146 (0.457) | 0.013 (0.203) |

Fixed far budget, arm A, share at KL ≤ 0.05:

| t | 512 | 1024 | 2048 |
|---:|---:|---:|---:|
| 2048 | 0.740 | 0.884 | 1.000 |
| 8192 | 0.561 | 0.685 | 0.818 |
| 32768 | 0.520 | 0.578 | 0.654 |

## Findings

- **Best real arm: A64 reads `0.058 t` at the median at 32k**, 17× fewer key reads than dense attention. The
  p90 head still reads `0.42 t`. This is a count of reads, not a wall-clock speedup.
- **Pages clump well.** True page order needs 152 far tokens, below the per-key `k_ε` of 167, because the near set
  already holds part of the mass. The clumping floor is `0.013 t`.
- **The cost is the bound, not the data.** A selects 2.6× the tokens OP needs (`bound_looseness`), and its
  metadata scan (2 vectors per 16-key page, `0.125 t`) is most of its `0.144 t` (`metadata_dominates`). Between
  A64 and the floor there is still a factor of 4.5 (`pages64_best_real`).
- **The high-frequency RoPE pairs carry ranking information.** Their far-key mean is zero (HATTN-T3, PH4), but a
  bound without them needs 3.3× the pages. The T3 receipt's subspace inference is corrected accordingly.
- **The tree does not help hard heads.** In the sink-dominated layers (7, 9–14) H reads `0.008 t` against A's
  `0.132 t`, because H stops after the root bounds while flat A always pays its scan. In the diffuse layers the
  256- and 4096-key boxes are too loose, and H expands nearly every page. Box bounds lose to the dimension.
- **A fixed budget is the wrong shape.** One budget meets KL 0.05 for 52–65 % of heads at 32k. The per-head
  demand spans two orders of magnitude, so a deployable selector needs a per-query stop rule.

## What this does not establish

- No deployable stop rule. The oracle stop reads the full softmax. A certified stop exists in principle: the
  unread mass is at most `Σ_unread P·e^{s·U_page}` (from `quest_bound_sound`). Its cost is unmeasured.
- One prompt; per-head attention KL, not whole-model KLD or tokens per second.
- No Rust or Aeneas path. rustc, LLVM and numpy are outside the trust boundary.
