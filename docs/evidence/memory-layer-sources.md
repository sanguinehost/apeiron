# Memory layers: retrieved sources and the split-complex reading

Retrieved and read. PDFs and text are under `/home/socol/oc-tmp/lit/`; line numbers refer to the `.txt`.

| id | paper | sha256 (pdf) |
|---|---|---|
| 1907.05242 | Lample, Sablayrolles, Ranzato, Denoyer, Jégou, *Large Memory Layers with Product Keys* (NeurIPS 2019) | `2bad22167b6d5f46424ea4fd9434d012ac4be88d58edfdefb3dfa14b344377f7` |
| 2412.09764 | Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh, *Memory Layers at Scale* (Meta FAIR 2024) | `8928f823d2d6db370c10ee7644102f408e94cbeba91d7ca8de9b50b88c520b8f` |
| 2407.04153 | He, *Mixture of A Million Experts* (PEER, Google DeepMind 2024) | `6bb96d43a38051780cbb6fa58944ef1c0653a28cf7b2668ca77ad24a050bb2c2` |
| 2411.12364 | Huang, Min, Huang et al., *Ultra-Sparse Memory Network* (UltraMem, ICLR 2025) | `e1f1689dc92c57cebf2df8a6abe62373777b8fe8feca6d4fdd3e37c8a08e5ef8` |

## What each states (quoted)

- **1907.05242:** keys are the Cartesian product `K = {(c, c′) | c ∈ C, c′ ∈ C′}`, `|K| = |C|×|C′|`. The query is
  split into sub-queries, the top-k is taken in each codebook, and the resulting k × k candidates "implicitly select
  k × k keys. The k keys maximizing the inner product with the query are guaranteed to belong to this subset"
  (lines 211–215). The abstract calls the method "fast and exact nearest neighbor search". Result: "a memory augmented
  model with only 12 layers outperforms a baseline transformer model with 24 layers, while being twice faster at
  inference time" (abstract).
- **2412.09764:** memory-augmented models "outperform dense models with more than twice the computation budget, as
  well as mixture-of-expert models", scaled to "128B memory parameters, pretrained to 1 trillion tokens", on base models
  up to 8B (lines 18–21). Factual QA accuracy improves "by over 100%" (line 93).
- **2407.04153:** PEER retrieves from over a million tiny experts with product keys, and outperforms dense FFWs and
  coarse-grained MoEs in isoFLOP comparisons (line 24).
- **2411.12364:** a 2-D logical address `(i, j)` indexes physical slot `n × i + j`, and grid scores are
  `S_grid = s_row + s_colᵀ` (eq. 2–3). An MoE at equal compute with 12× the parameters "runs 2 to 6 times slower in
  inference" because of memory access (lines 75–76). Their Tucker retrieval (eq. 5) scores
  `S_rowᵀ × C × S_col`, a rank-r core, and they state that "the top-m operation cannot be simplified with an
  equivalent two-phase top-m technique like product quantization can". So they approximate the core by rank 1,
  `C ≈ u tᵀ` (eq. 6).

## The split-complex reading (grade: `model`, with the corpus links `proved`)

The weight of slot `(i, j)` is `e^{s_row(i) + s_col(j)} = e^{s_row(i)} · e^{s_col(j)}`. Write a split-complex number
`z = a + b·j` (`j² = 1`, the hyperbolic face `σ = 1`) in its null (idempotent) coordinates `u = a + b`, `v = a − b`.
Then the norm form is `N(z) = a² − b² = u·v`.

- A product-key slot is a split-complex number whose null coordinates are the two per-codebook weights, and its
  weight is the norm form `u·v`. Level sets of the score are hyperbolas `u·v = const`.
- Exact two-phase search works because `N` is monotone in each null coordinate separately (both positive). This is the
  idempotent split of the hyperbolic cell: `Nous.Model.Decomposition.hyperbolic_decomposition` and
  `Substrate.Algebra.NullCell.SpinorFold.nmul_is_diagonal` (both audited), a product that costs two real products in
  the null basis.
- UltraMem's Tucker core is a sum of r such cells, `Σ_{r,r′} C_{rr′} a_r(i) b_{r′}(j)`. It is diagonal in one null
  basis only when `C` has rank 1, which is exactly the approximation UltraMem makes. So the exact-search boundary they
  hit is the boundary of a single split-complex cell.
- The trichotomy then reads: elliptic `σ < 0` is position and rotation (RoPE, the torus); hyperbolic `σ > 0` is
  two-channel addressing and gating (product keys, the SiLU gate `silu_gate_eq_channelWeight`); parabolic `σ = 0` is
  averaging (`attention_channel_form_parabolic`).

## Proved (audited in `proof/corpus/AxiomAudit.lean`, `lake build AxiomAudit` exit 0, 9072 jobs)

| theorem | file (sha256 prefix) | statement |
|---|---|---|
| `product_topk_subset`, `product_topk_exact` | `Nous/Seq/ProductKeys.lean` (`b0c955e3`) | top-k of `a i + b j` over the grid equals top-k over `topk a k × topk b k`, lexicographic ties |
| `multi_product_topk`, `multi_candidate_card` | same | d codebooks: top-k lies in `∏ topk (a t) k`; `k^d` candidates |
| `slot_address_bijective` | same | `(i, j) ↦ i·m + j` is `Fin n × Fin m ≃ Fin (n·m)` |
| `slot_mul`, `slot_normForm` | `Nous/Seq/HyperbolicAddress.lean` (`e3ac19b3`) | `slotCell(a,1)·slotCell(1,b) = slotCell(a,b)`, norm form `a·b`; from `ChannelsAnyRing.ringUpper_mul`, `ringLower_mul`, `ringNormForm_eq_upper_mul_lower`, `ringOfHalves_upper_lower` |
| `two_phase_exact_hyperbolic`, `log_additive` | same | positive channels: grid top-k ⊆ row top-k × column top-k; `log N = s_i + t_j` |
| `split_iff_hyperbolic`, `split_iff_nontrivial_idempotent` | same | the norm form factors into independent real linear forms iff `σ > 0`, iff a nontrivial idempotent exists (`Idempotents.exists_nontrivial_idempotent_iff_disc_pos`) |
| `parabolic_one_channel`, `elliptic_no_channels` | same | `σ = 0`: channels coincide; `σ < 0`: no channels |
| `rank1_core_is_cell`, `tucker_rank2_breaks` | same | rank-1 Tucker score is one cell; a 3×3 rank-2 core (`C = I₂`) whose marginal top-1 `(1,1)` scores 72 against the true top `(0,0)` at 100; marginals are proportional to UltraMem's rank-1 phase with `u = t = (1,1)/√2`, a top singular vector of `I₂` |
| `rank_r_threshold_exact` | same | Fagin's threshold stop for a nonnegative sum of r channel scores; generic, not yet specialised to cells |

Controls (lead, throwaway copies in `/home/socol/oc-tmp/lead-pkm-ctl/`): reversed row ranking in `prec` → exit 1
(`prec_snd_iff`, `prec_multi_update`, the `decide` witness); swapped channels in `slotCell` → exit 1 (`slot_channels`,
`slot_mul`).

| `score_as_cells`, `unseen_le_threshold`, `threshold_stop_exact`, `seen_card_le` | `Nous/Seq/ThresholdCells.lean` | a rank-r score is a sum of r canonical cell norm forms; scanning the top-D rows and columns of every cell (`Seen D = ⋃ R_l × C_l`, at most `r·D²` slot evaluations) bounds every unseen slot by `T = Σ w_l·max(α_l·Bmax_l, Amax_l·β_l)`; once k seen slots score above T, top-k over `Seen D` equals the global top-k (Fagin, Lotem, Naor 2001, rederived for products) |
| `rank1_candidate_exact`, `Witness.tucker_*` | same | r = 1 recovers product-key exactness at D = k; on the rank-2 core that breaks two-phase search, T = 120 at D = 1 (no stop) and T = 0 at D = 2, where the stop applies and returns the true top slot (0, 0), score 100 |

Controls: `α·β` in place of the max (worker) and `min` in place of `max` (lead, `/home/socol/oc-tmp/lead-pkm3-ctl/`)
both fail at `unseen_le_threshold`; the witness mutation fails at `tucker_hAmax`, `tucker_hα1`, `tucker_exact_d2`.

Open: a served memory-layer or MoE model on the exact engine; no memory-layer checkpoint is public.
