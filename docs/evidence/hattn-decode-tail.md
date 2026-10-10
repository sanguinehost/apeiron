# HATTN-T5: whole-model next-token KLD with page selection in the decode tail

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Grade: `measured` (one prompt, 16 positions, Rust
f32 forward over the [HATTN-T3](hattn-far-field.md) dump); verdicts and the certified stop's soundness `proved` in
`proof/corpus/Perf/HattnT5.lean`.

## Method

| item | value |
|---|---|
| prefill cache | `/home/socol/oc-tmp/hattn-long/t32k` (HATTN-T3), dense q/k/v for 32,768 positions |
| instrument | `KV_TAIL=<dump> examples/attn_cut_probe.rs` (`run_tail`), source sha256 `eeed33f4…`, binary `a1317587…`; output `/home/socol/oc-tmp/hattn-long/tail.out` (sha256 `ed5d3f5c…`), 8 s |
| decode tail | for `t ∈ {8192, 32768}` the last 8 positions are recomputed from token embeddings through all 24 layers, once per arm; each tail query attends to the dump's prefix keys `[0, t−8)` and the arm's own tail keys, so selection errors compound across layers and tail positions |
| output | final RMSNorm and the tied LM head (`token_embd.weight`, 896 × 151,936); KLD(dense ‖ arm) of the next-token distribution, top-1 agreement |
| control 1 | the dense arm's tail q, k, v match the dump at every layer: max rel-L2 3.6e-6 (8k), 1.5e-5 (32k); limit 1e-4 |
| control 2 | the certified arm keeps mass ≥ `1 − ε` in every (layer, head, position); `HATTN_MUT=1` (min-only bound `q·m`) breaks it in 1157 and 1146 heads and exits 1 (`tail-mut.out`, sha256 `56fa7369…`) |
| predictions | registered before the run: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T5.md`, sha256 `26c24721…` |

`ε = 1 − e^{−0.05}` (per-head KL 0.05). The page arms use [HATTN-T4](hattn-page-select.md)'s A64: 64-key pages from
position 4, near set of 4 sinks and the last ≥ 256 keys, and reads count a bound as 2 vectors and a page as 64.

| arm | selection |
|---|---|
| dense | every key |
| oracle | per head, top keys by weight to mass `1 − ε` (needs every score; not deployable) |
| near | near set only |
| a64-order | Quest pages in bound order, stopped on the true kept mass (not deployable) |
| a64-cert | Quest pages in bound order, stopped when `Σ_unread 64·e^{U − c} ≤ ε/(1 − ε) · Z_read` (deployable, sound by `cert_stop_sound`) |
| a64-fix | the 32 highest-bound pages (2048 far tokens) |

## Results

| arm | KLD mean 8k | KLD mean 32k | top-1 | reads/`t` 32k, mean (median) |
|---|---:|---:|---:|---:|
| oracle | 0.0020 | 0.0007 | 16/16 | 0.027 (0.005) |
| near | 0.393 | 0.0083 | 14/16 | 0.010 |
| a64-order | 0.0014 | 0.0006 | 16/16 | 0.155 (0.060) |
| a64-cert | 0.0000 | 0.0000 | 16/16 | 1.011 (1.031) |
| a64-fix | 0.0013 | 0.0018 | 16/16 | 0.103 |

| prediction | registered | measured | verdict |
|---|---|---|---|
| PT1 | dense tail reproduces the dump, rel-L2 ≤ 1e-4 | 1.5e-5 | held |
| PT2 | certified mass ≥ `1 − ε` everywhere; the mutation breaks it | 0 violations; 1157, 1146 | held |
| PT3 | oracle KLD ≤ 0.05 | 0.0020, 0.0007 | held |
| PT4 | a64-cert median reads/`t` at 32k > 0.5 | 1.031 | held |
| PT5 | near KLD at 32k > 0.05 | 0.0083 (0.393 at 8k) | refuted at 32k |
| PT6 | a64-fix KLD at 32k ≤ 0.05 | 0.0018 | held |
| PT7 | a64-cert top-1 16/16 | 16/16 | held |

## Findings

- **Per-head KL 0.05 is 25–70× stricter than the whole-model target.** Holding every head at KL ≈ 0.047 gives
  whole-model KLD 0.0020 at 8k and 0.0007 at 32k (`compounding_attenuates`). The errors do not compound across
  24 layers; the residual stream averages them. The T3/T4 per-head target was the wrong unit.
- **The certified stop costs more than dense.** It is sound (0 violations; the mutation gives 1146), but Quest's
  page bound is loose enough that the unread-mass sum only falls under `ε Z` after nearly every page: 1.03 `t`
  reads with the metadata. A certificate built from these bounds buys no reads.
- **A fixed 2048-token far budget reads `0.103 t` at 32k at KLD 0.0018.** T4 measured that budget failing the
  per-head target in 35 % of heads. That failure does not reach the output.
- **The 32k positions are easy.** Dropping every far key costs only 0.0083 there, while at 8k it costs 0.39
  (three positions above 0.5). Eight positions of one prompt do not separate the arms at 32k.

## What this does not establish

- 16 positions from one prompt. The 32k sample cannot tell the arms apart (`pt5_refuted`).
- The prefix cache is dense (Quest's setting: dense prefill, selective decode). Selection during prefill is
  not measured.
- Reads are a count, not wall-clock time; no tokens per second. The f32 forward is not the engine's decode path.
- No Rust/Aeneas extraction of the selector. rustc and LLVM are outside the trust boundary.

## HATTN-T6: fixed budgets and per-head tolerance, 64 positions at four contexts

Grade: `measured` (one prompt, 256 positions); verdicts `proved` as arithmetic over the measured constants in
`proof/corpus/Perf/HattnT6.lean`.

| item | value |
|---|---|
| instrument | `KV_TAIL=<dump> KV_TAIL_SWEEP=1 examples/attn_cut_probe.rs`, source sha256 `308c4e33…`, binary `f382ed6e…`; output `/home/socol/oc-tmp/hattn-long/sweep.out` (sha256 `09fee1ab…`), 46 s |
| positions | the last 64 before each `t ∈ {4096, 8192, 16384, 32768}`, recomputed through 24 layers per arm as in T5 |
| control | dense tail against the dump: max rel-L2 2.0e-6, 3.9e-6, 1.3e-5, 1.8e-5; certification violations 0 |
| predictions | `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T6.md`, sha256 `c2a6b29f…` |
| Lean controls | `fix1024Kld` at 8k set to 568 breaks `fix1024_cheapest`; `fix1024Reads32` set to 590 breaks `ps6_refuted` (`/home/socol/oc-tmp/hattn-long/mut/HattnT6{m,n}.lean`) |

Mean whole-model KLD (positions above 0.05 out of 64) and mean reads/`t`:

| arm | t = 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|
| near | 0.1886 (46) 0.072 | 0.6145 (52) 0.036 | 0.1313 (33) 0.018 | 0.0277 (8) 0.009 |
| a64-fix256 | 0.0700 (26) 0.164 | 0.2642 (43) 0.097 | 0.0561 (18) 0.064 | 0.0171 (5) 0.048 |
| a64-fix512 | 0.0377 (15) 0.227 | 0.1296 (28) 0.129 | 0.0449 (16) 0.080 | 0.0129 (1) 0.056 |
| a64-fix1024 | 0.0165 (6) 0.353 | 0.0368 (18) 0.191 | 0.0293 (14) 0.111 | 0.0089 (1) 0.071 |
| a64-fix2048 | 0.0036 (0) 0.605 | 0.0092 (1) 0.317 | 0.0158 (6) 0.174 | 0.0053 (0) 0.102 |
| oracle, head KL 0.25 | 0.0259 (7) 0.028 | 0.0876 (19) 0.017 | 0.0497 (20) 0.012 | 0.0314 (7) 0.005 |
| oracle, head KL 1.0 | 0.3443 (59) 0.004 | 0.6019 (56) 0.002 | 0.3530 (59) 0.001 | 0.4271 (57) 0.0004 |
| a64-order, head KL 1.0 | 0.0909 (37) 0.108 | 0.1327 (35) 0.073 | 0.0673 (20) 0.055 | 0.0217 (8) 0.041 |

| prediction | registered | measured | verdict |
|---|---|---|---|
| PS1 | dense tail rel-L2 ≤ 1e-4 at every `t` | ≤ 1.8e-5 | held |
| PS2 | a64-fix2048 mean KLD ≤ 0.05 at every `t` | 0.0036–0.0158 | held |
| PS3 | a64-fix512 ≤ 0.05 at every `t ≥ 8192` | 0.1296 at 8192 | refuted |
| PS4 | near > 0.05 at 8192 | 0.6145 | held |
| PS5 | oracle at head KL 1.0 ≤ 0.05 at every `t` | 0.34–0.60 | refuted |
| PS6 | cheapest passing deployable arm reads ≤ `0.06 t` at 32k | a64-fix1024, `0.071 t` | refuted |

### Findings

- **The operating point is a64-fix1024:** mean KLD ≤ 0.037 at every `t`, top-1 55–61 of 64, `0.071 t` reads at
  32k (14× fewer key reads than dense, counted per query head; per KV head in bytes it is 8.6×, see T7). The next cheaper budget fails at 8192
  (`fix1024_cheapest`). This is a count of reads, not a time.
- **The mean passes, the tail does not.** a64-fix1024 has 18 of 64 positions above 0.05 at 8192 (max 0.26). A
  per-position guarantee at 0.05 needs a64-fix2048, which still misses at 16384 (6 positions, max 0.11).
- **The whole-model tolerance sits between per-head KL 0.25 and 1.0.** Per-head 0.25 passes on the mean at three
  of four `t` (0.0876 at 8192); per-head 1.0 fails everywhere. Attenuation from head to output is 25–70× at
  head KL 0.05 (T5) and only 2.5× at 0.25 (`attenuation_weak_at_quarter`): small per-head errors are averaged
  out, large ones are not.
- **The selector is linear in `t`.** Reading every page bound costs `0.031 t`, 44 % of a64-fix1024's reads at
  32k (`bounds_dominate`). Below a 2048-token budget the scan, not the selected keys, sets the cost. The
  oracle at head KL 0.25 reads `0.005 t` at 32k: the keys themselves are not the obstacle, finding them is.
- **8192 is the hard context in this prompt.** Every arm is worst there, including the near set (0.61). Context
  length alone does not set the difficulty; the content at those positions does.

### Axiom cone (row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`, HATTN-T3 to T6)

`proof/corpus/AxiomAudit.lean` audits every verdict. The `decide` verdicts in `Perf.HattnT3`–`Perf.HattnT6`
depend on no axioms. The four real-analysis theorems `HattnT3.first_cumulant_le`,
`HattnT3.second_cumulant_overshoots`, `HattnT4.quest_bound_sound` and `HattnT5.cert_stop_sound` depend on
`{propext, Classical.choice, Quot.sound}`, the corpus's closed cone.

### What T6 does not establish

- One prompt; 256 positions. Per-prompt variance is unmeasured.
- Selection only in the decode tail; prefill is dense.
- Reads, not wall-clock or tokens per second; no Rust/Aeneas extraction of the selector; rustc and LLVM are
  outside the trust boundary.

## HATTN-T7: bytes per KV head, one page set per GQA group, reused across positions

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Grade: `measured` (one prompt, 256 positions);
verdicts `proved` as arithmetic over the measured constants in `proof/corpus/Perf/HattnT7.lean`, empty axiom
cone.

| item | value |
|---|---|
| instrument | `KV_TAIL=<dump> KV_TAIL_GROUP=1 examples/attn_cut_probe.rs`, source sha256 `e43db6ba…`, binary `71f044e0…`; output `/home/socol/oc-tmp/hattn-long/group.out` (sha256 `d9e1efd6…`), 50 s |
| cost metric | bytes/dense per (KV head, position): `(2·|∪ over the 7 query heads of keys used| + bound vectors read) / (2t)`; the bound scan is read once per KV head; dense is 1 exactly |
| control | dense tail rel-L2 ≤ 1.8e-5; dense bytes/dense = 1 at every (KV head, position); `HATTN_MUT=2` (group ranking reversed) raises grp-fix1024-r1 to 0.1730, 0.6065, 0.1136, 0.0231 (`group-mut.out`, sha256 `0d0cc148…`) |
| predictions | `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T7.md`, sha256 `54a5d2ab…` |
| Lean controls | `grp2048r8B32` set to 820 breaks `singles_agree`, `pg6_held`, `grp2048r8_thirteen_x`; `grp2048r1Kld` at 8k set to 1227 breaks `pg4_held`, `pg5_held` (`/home/socol/oc-tmp/hattn-long/mut/HattnT7{m,n}.lean`) |

| arm | selection |
|---|---|
| a64-fix{1024,2048} | T6: Quest-64 pages chosen per query head |
| grp-fix{b}-r1 | one page set per KV head: pages ranked by the max over its 7 query heads of Quest's bound, top `b/64` |
| grp-fix{b}-r8 | the same set recomputed every 8 positions and reused between; newly sealed far pages always added; the bound scan paid only at refresh |
| oracle-0.25 | union of the 7 heads' oracle keys at per-head KL 0.25 (not deployable) |

Mean whole-model KLD (positions above 0.05 out of 64) and mean bytes/dense:

| arm | t = 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|
| a64-fix1024 | 0.0165 (6) 0.568 | 0.0368 (18) 0.326 | 0.0293 (14) 0.204 | 0.0089 (1) 0.116 |
| a64-fix2048 | 0.0036 (0) 0.828 | 0.0092 (1) 0.522 | 0.0158 (6) 0.330 | 0.0053 (0) 0.189 |
| grp-fix1024-r1 | 0.0196 (4) 0.338 | 0.1551 (25) 0.176 | 0.0396 (17) 0.096 | 0.0145 (2) 0.056 |
| grp-fix2048-r1 | 0.0047 (0) 0.590 | 0.0227 (7) 0.302 | 0.0254 (12) 0.158 | 0.0104 (1) 0.087 |
| grp-fix1024-r8 | 0.0262 (8) 0.327 | 0.1764 (28) 0.164 | 0.0455 (16) 0.083 | 0.0150 (2) 0.042 |
| grp-fix2048-r8 | 0.0053 (0) 0.579 | 0.0234 (7) 0.289 | 0.0342 (13) 0.145 | 0.0110 (1) 0.074 |
| oracle-0.25 | 0.0259 (7) 0.113 | 0.0876 (19) 0.071 | 0.0497 (20) 0.046 | 0.0314 (7) 0.021 |

| prediction | registered | measured | verdict |
|---|---|---|---|
| PG1 | dense rel-L2 ≤ 1e-4, dense bytes = 1, reversed ranking > 0.2 at 8192 | 1.8e-5; true; 0.6065 | held |
| PG2 | a64-fix1024 bytes/dense at 32k ≥ 0.10 | 0.116 | held |
| PG3 | grp-fix1024-r1 ≤ 0.05 at every `t` | 0.1551 at 8192 | refuted |
| PG4 | grp-fix2048-r1 ≤ 0.05 at every `t`, bytes at 32k ≤ 0.10 | ≤ 0.0254; 0.087 | held |
| PG5 | reuse r8 moves mean KLD ≤ 0.01 at every `t` | 0.0006, 0.0007, 0.0088, 0.0006 | held |
| PG6 | cheapest passing deployable arm ≤ 0.08 bytes at 32k | grp-fix2048-r8, 0.074 | held |

### Findings

- **Correction to T6.** T6's "14× fewer key reads" for a64-fix1024 counted reads per query head. In bytes per
  KV head, with the 7 heads' keys unioned and K and V both read, it is `0.116` of dense at 32k: 8.6×
  (`gqa_correction`). The T6 figure overstated the saving because the 7 query heads of a group select
  different pages.
- **One page set per group, reused for 8 positions, is 13.6× cheaper than dense at 32k** at mean KLD ≤ 0.035
  at every `t` (`grp2048r8_thirteen_x`). It costs less than a64-fix1024 at every `t ≥ 8192` and holds a lower
  KLD at 8192 (0.023 against 0.037).
- **Reuse is free at this resolution.** Refreshing the set every 8 positions instead of every position moves
  mean KLD by ≤ 0.009 (`pg5_held`) and cuts the bound scan 8×: at r8 the scan is `1/512` of dense, and the
  selected keys (the 2048 budget, the near set, newly sealed pages) are the rest.
- **The oracle union is 3.5× below grp-fix2048-r8 at 32k** (0.021 against 0.074, `oracle_gap`) and still fails
  at 8192 on the mean: a per-head KL 0.25 oracle is not the target, a whole-model one is.
- **Per-position tail.** grp-fix2048-r8 has 13 of 64 positions above 0.05 at 16384 (max 0.25). The mean
  passes; a per-position guarantee does not.

### Derived, not measured: the cost at 1M context

At refresh interval `r` the scan costs `1/(64 r)` of dense per position (2 bound vectors per 64-key page,
against 2 vectors per key for dense) and the selected keys cost `≈ 2350/t` at grp-fix2048-r8 (its 32k figure
less the scan). At `t = 2^20` that is `0.0020 + 0.0022 ≈ 0.004`, about 240× below dense, if the whole-model
KLD held at 1M. It is unmeasured: Qwen2.5-0.5B's native context is 32,768 and a 1M dump of this model is
≈ 112 GB. At 1M the linear scan is half the cost; only a sublinear index removes it.

### What T7 does not establish

- One prompt; 256 positions; a model with a 32k native context.
- Selection in the decode tail only; prefill is dense.
- Bytes are a count, not wall-clock or tokens per second; no Rust/Aeneas extraction of the selector; rustc
  and LLVM are outside the trust boundary.

## HATTN-T8: a page index in place of the flat bound scan

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Grade: `measured` (one prompt, 64 tail positions at four contexts); verdicts, the tree's logarithmic depth and
the 7B/1M cost arithmetic `proved` in `proof/corpus/Perf/HattnT8.lean` (every headline audits with an empty
axiom set except `climb_log`: `propext`, `Quot.sound`).

| item | value |
|---|---|
| baseline | `grp-fix2048-r8` (T7): one page set per KV head, 2048-token far budget, refreshed every 8 positions, by a flat scan of every 64-key page's Quest bound |
| `tree-sS` | contiguous fanout-16 tree of Quest boxes (64, 1024, 16384 keys): read every top node, keep the `S` with the highest group bound (max over the 7 query heads), expand their children, repeat down to pages; keep the top 32 pages |
| `clu-sS` | pages k-means clustered on box centre (8 Lloyd steps) into `⌈n64/16⌉` clusters at the first refresh; later-sealed pages join the nearest centroid's cluster and widen its box; read every cluster box, keep `S` clusters, rank their pages |
| instrument | `examples/attn_cut_probe.rs` sha256 `45f38096…`, `KV_TAIL_TREE=1`, binary `attn_cut_probe.t8` sha256 `0df17577…`; 41 s; output `/home/socol/oc-tmp/hattn-long/tree.out` sha256 `64b46d6d…` |
| control | dense tail max rel-L2 `1.815e-5`; `HATTN_MUT=2` (ranking reversed at every level) output `tree-m2.out` sha256 `c5c71824…` |
| predictions | `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8.md`, sha256 `719f0fd1…`, registered before the run |
| defective run | `tree-prefix.out` (sha256 `cb90ea43…`) built clusters at the first refresh and dropped every page sealed after it (`clu` KLD 0.0245 at 4096 against 0.0053). Fixed by nearest-centroid insertion; its figures are not used. |

Bound vectors per refresh per KV head counts what the index reads; recall is the share of the flat scan's top 32
pages the index selects (the flat ranking is computed only to measure recall).

Mean whole-model KLD:

| arm | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|
| grp-fix2048-r8 (flat) | 0.0053 | 0.0234 | 0.0342 | 0.0110 |
| tree-s4 | 0.0053 | **0.0714** | 0.0335 | 0.0128 |
| tree-s8 | 0.0053 | 0.0234 | 0.0372 | 0.0118 |
| clu-s4 | 0.0053 | 0.0192 | 0.0352 | 0.0121 |
| clu-s8 | 0.0053 | 0.0234 | 0.0329 | 0.0107 |
| reversed (`HATTN_MUT=2`), every arm | 0.1158 | 0.61–0.62 | | |

Bound vectors per refresh (recall):

| arm | 8192 | 16384 | 32768 |
|---|---:|---:|---:|
| flat | 245.8 | 501.8 | 1013.8 |
| tree-s4 | 138.4 (0.749) | 158.2 (0.557) | 193.6 (0.498) |
| tree-s8 | 261.8 (1.000) | 283.7 (0.817) | 319.3 (0.727) |
| clu-s4 | 153.5 (0.822) | 174.8 (0.549) | 211.4 (0.507) |
| clu-s8 | 261.8 (1.000) | 301.2 (0.827) | 345.3 (0.712) |

| prediction | registered | measured | verdict |
|---|---|---|---|
| PT1 | dense ctl ≤ 1e-4; reversed tree-s8 KLD > 0.2 at 8192 | 1.8e-5; 0.612 | held |
| PT2 | tree-s8 passes, ≤ 1/3 of the flat scan at 32k | passes; 319.3 / 1013.8 = 0.315 | held |
| PT3 | tree-s4 fails at some `t` | 0.071 at 8192 | held |
| PT4 | clu-s4 passes | passes; best KLD at 8192 | held |
| PT5 | tree-s8 recall ≥ 0.8 at 32k | 0.727 | refuted |
| PT6 | best passing index arm ≤ 0.0025 of dense, Qwen2.5-7B-1M shape at `2^20` | tree-s8 0.00228 (derived) | held, derived |

### Findings

- **The index cuts the scan 3–5× at 32k at the flat scan's KLD.** clu-s4 reads 211 bound vectors per refresh
  against 1014 (4.8×) and tree-s8 reads 319 (3.2×); both pass at every `t`, within 0.002 of the flat arm at 32k
  (`clu4_cuts_scan`, `pt2_held`).
- **The budget has slack.** The index arms keep only 50–73% of the flat top-32 pages at 32k and the KLD does not
  move (`budget_slack`): the pages that hold the mass are found by any arm that passes, and the rest of the 2048
  tokens are interchangeable. Recall against the flat ranking is the wrong target; mass is.
- **Four 1024-key boxes are too few at 8192** (tree-s4, 22 of 64 positions above 0.05). Clustering by key
  direction instead of position fixes it at the same width: clu-s4 at 8192 is 0.019, below flat.
- **The tree scan grows as `log n`.** A tree with `L` levels above the pages has more than `16^L` pages
  (`climb_log`), so its scan is at most `2(top + 16·S·L)`. The measured tree-s8 at 32k (319.3) is under that
  model's 516 (`model_bounds_32k`). The cluster index reads `n/16` boxes: 16× below flat, still linear.

### Derived, not measured: Qwen2.5-7B-Instruct-1M at `t = 2^20`

Per (layer, KV head), per 8-position refresh period: 2048 far + 4 sink + 288 mean near tokens as K and V at
every position, plus the scan once (`pt6_held`, `bytes_7b_1m`).

| item | flat | tree-s8 | dense |
|---|---:|---:|---:|
| pages / tree levels | 16,379 | 4 roots, 3 levels | |
| scan vectors per refresh | 32,758 | 776 | |
| vectors per period | 70,198 | 38,216 | 16,777,216 |
| fraction of dense | 0.00418 (239×) | 0.00228 (439×) | 1 |
| scan share of reads | 47% | 2% | |
| KV bytes read per token, f16 (28 layers × 4 KV heads × 128) | 252 MB | 137 MB | 60.1 GB |

At 1M the tree removes the scan as a cost; what remains is the 2048-token budget and the near window. Whether the
whole-model KLD holds at 1M with a 2048 budget is the open measurement.

### What T8 does not establish

- One prompt, a 0.5B model with a 32k native context; nothing measured past 32768.
- Decode tail only, prefill dense; bytes are counts, not tokens per second.
- No Rust/Aeneas selector; rustc, LLVM and the instrument are outside the trust boundary.

## HATTN-T8 on Qwen2.5-7B-Instruct-1M

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Grade: `measured` (one prompt, 64 tail positions per `t`);
verdicts `proved` in `proof/corpus/Perf/HattnT8Q7.lean` (sha256 `465addd4…`, empty axiom cone, four changed-constant
controls red).

| item | value |
|---|---|
| model | Qwen2.5-7B-Instruct-1M Q4_K_M: 28 layers, 28 query / 4 KV heads, head dim 128 |
| dump | `/home/socol/oc-tmp/hattn-long/t32k-7b`, 32768 tokens, ids sha256 `a9ff4f03…` as T3, CPU `run_dump` (137 s/layer at 32k), 17 GB |
| instrument | `attn_cut_probe.hd128t` (`PROBE_HD=128`, source `91fae89`), sha256 `8b149f25…`; `KV_TAIL=t32k-7b KV_TAIL_TREE=1 KV_TAIL_TS=<t>`, 56–114 s per `t` |
| outputs | `tree7b-{4096,8192,16384,32768}.out` (`b6aa2fd7…`, `240e245d…`, `35112209…`, `c14179bf…`), mutation `tree7b-m2.out` (`f16fd68d…`), all TAIL PASS |
| predictions | registered before the run: `PREDICTIONS-T8-7B.md`, sha256 `360d7ea8…` |
| control | dense tail rel-L2 over q, k, v, 28 layers: 2.4e-6, 5.0e-6, 1.2e-5, 1.9e-5 at the four `t` (gate 1e-4); certification violations 0 |

Mean whole-model KLD (gate 0.05) at `t = 4096, 8192, 16384, 32768`:

| arm | 4096 | 8192 | 16384 | 32768 | bound vectors / refresh at 32768 |
|---|---:|---:|---:|---:|---:|
| `grp-fix2048-r8` (flat) | 0.0030 | 0.0070 | 0.0103 | 0.0038 | 1013.8 |
| `tree-s4` | 0.0030 | 0.0262 | 0.0148 | 0.0049 | 195.1 |
| `tree-s8` | 0.0030 | 0.0070 | 0.0115 | 0.0047 | 321.9 |
| `clu-s4` | 0.0030 | 0.0084 | 0.0163 | 0.0043 | 231.2 |
| `clu-s8` | 0.0030 | 0.0070 | 0.0117 | 0.0040 | 376.0 |

| prediction | verdict |
|---|---|
| P7-1 control: dense tail <= 1e-4; `HATTN_MUT=2` tree-s8 > 0.2 at 8192 | held: 1.9e-5 max; mutation 0.366 |
| P7-2 flat passes at every `t` | held: max 0.0103 |
| P7-3 `clu-s4` passes, <= 1/4 of the flat scan at 32768 | held: 4.4x fewer (231.2 against 1013.8) |
| P7-4 `tree-s8` passes at every `t` | held: max 0.0115 |
| P7-5 `tree-s8` scan under the model's 516 | held: 321.9 |
| P7-6 derived `t = 2^20` read <= 0.0025 of dense | held (derived): 0.00228; the model at 32768 bounds the measured bytes/dense 0.0722 (model 0.0723) |

### Findings

- **The 7B passes every arm at every `t`**, `tree-s4` included (0.0262 at 8192; 0.071 on the 0.5B).
- **The 7B is easier than the 0.5B:** the flat arm's mean KLD is lower at every `t` (0.0038 against 0.0110 at 32768).
- **The index's recall falls and quality does not:** `tree-s8` keeps 0.625 of the flat top-32 at 32768 (0.727 on the
  0.5B) at KLD 0.0047 (0.0118): the pages it misses carry less mass.
- **Reads at 32768 on the 7B:** 0.0716 of dense keys (0.0722 bytes/dense with the scan); `tree-s8` reads 3.1x fewer
  bound vectors than the flat scan, `clu-s4` 4.4x.

### What this does not establish

- One prompt; decode tail only, prefill dense; nothing past 32768. Bytes are counts, not tokens per second.
- The 1M figure is the cost model's; whether the 2048-token budget holds the KLD gate at 1M is the open measurement
  (the GPU dump, `KV_T` past 32768).
- No Rust/Aeneas selector; rustc, LLVM and the instrument are outside the trust boundary.

## HATTN-T8 on Qwen2.5-7B-Instruct-1M at 65536 and 131072

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Grade: `measured` (one prompt, 64 tail positions per `t`);
verdicts `proved` in `proof/corpus/Perf/HattnT8Q7L.lean` (sha256 `b1ac7126…`, empty axiom cone, four changed-constant
controls red).

| item | value |
|---|---|
| dump | `/home/socol/oc-tmp/hattn-long/g7-128k`, 131072 tokens, ids sha256 `9dd57952…` (prefix equals the T3 ids), `examples/gpu_dump.rs` f32 Vulkan, 379 s for 28 layers, 16 GB; `KV_QTAIL=32768,65536,131072` |
| instrument | `attn_cut_probe.hd128q` sha256 `e3261748…`; `KV_TAIL=g7-128k KV_TAIL_TREE=1 KV_QTAIL=… KV_TAIL_TS=<t>`, 103 s / 197 s / 409 s per `t` |
| outputs | `g7t-{32768,65536,131072}.out` (`4be5423a…`, `944ddbb7…`, `4974457d…`), mutation `g7t-m2.out` (`235a6c22…`) |
| predictions | registered before the run: `PREDICTIONS-T8-7B-128K.md`, sha256 `449ab1b4…` |
| control | dense tail rel-L2 over q, k, v, 28 layers: 2.0e-5, 4.7e-5, 7.0e-5 (gate 1e-4); certification violations 0; the GPU dump gives the CPU dump's mean KLD at 32768 on all five arms |

Mean whole-model KLD (gate 0.05) and cost:

| arm | 32768 | 65536 | 131072 | bound vectors / refresh at 131072 | flat-top-32 recall at 131072 | bytes/dense at 131072 |
|---|---:|---:|---:|---:|---:|---:|
| `grp-fix2048-r8` (flat) | 0.0038 | 0.0208 | 0.0242 | 4085.8 | n/a | 0.0198 |
| `tree-s4` | 0.0049 | **0.0865** | **0.0611** | 270.8 | 0.267 | 0.0180 |
| `tree-s8` | 0.0047 | **0.0646** | **0.0549** | 525.9 | 0.452 | 0.0181 |
| `clu-s4` | 0.0043 | 0.0256 | 0.0270 | 414.1 | 0.289 | 0.0181 |
| `clu-s8` | 0.0040 | 0.0236 | 0.0227 | 566.1 | 0.448 | 0.0182 |

`HATTN_MUT=2` at 65536: every arm 0.133–0.137 (flat 0.1364, `tree-s8` 0.1331).

| prediction | verdict |
|---|---|
| Q1 control: dense tail <= 1e-4; GPU = CPU dump at 32768 within 0.002; `HATTN_MUT=2` `tree-s8` > 0.2 at 65536 | held / held (identical to four digits) / refuted: 0.133, still 2x `tree-s8` and 6x flat unmutated |
| Q2 flat, `clu-s4`, `tree-s8` pass at 65536 and 131072 | flat and `clu-s4` held; `tree-s8` refuted (0.0646, 0.0549) |
| Q3 `tree-s4` passes | refuted (0.0865, 0.0611) |
| Q4 `tree-s8` scan under the model's 528; `clu-s4` <= 1/4 of flat | held: 525.9; 414.1 against 4085.8 (9.9x) |
| Q5 `tree-s8` recall below its 0.625 at 32768 | held: 0.531, 0.452 |
| Q6 `tree-s8` bytes/dense <= 0.0190, flat <= 0.0215 | held: 0.0181, 0.0198 |

### Findings

- **The cluster index holds the gate to 131072; the tree index does not.** `clu-s4` and `clu-s8` pass at 65536 and
  131072 (0.0256/0.0270 and 0.0236/0.0227), as does the flat arm; `tree-s4` and `tree-s8` cross 0.05 at 65536 and stay over at
  131072. At 131072 `tree-s8` and `clu-s8` share a flat-top-32 recall (0.452, 0.448) at KLD 0.0549 and 0.0227: the
  pages the tree misses carry more mass; top-32 recall does not predict the gate. The cause is unmeasured.
- **`clu-s4` reads 9.9x fewer bound vectors than the flat scan at 131072 (414.1 against 4085.8) at KLD 0.0270;
  `clu-s8` reads 7.2x fewer at 0.0227, under the flat arm's 0.0242.**
- **The flat arm drifts with context:** 0.0038, 0.0208, 0.0242 at a fixed 2048-token far budget (1/16 of the context
  at 32768, 1/64 at 131072); 10 of 64 positions exceed 0.05 individually at 131072. The mean still passes.
- **Bytes per generated token at 131072** (f16 KV, 28 layers x 4 KV heads x 256 B): dense 7.52 GB; `clu-s8` at
  0.0182 reads 136.8 MB (`bytes_7b_128k`).

### What this does not establish

- One prompt; decode tail only, prefill dense; context 131072 at most. Bytes are counts, not tokens per second.
- The dump is f32 SIMT, not the Q4_K_M reference path; it matches the CPU dump to 7.4e-6 rel-L2 at 32768.
- The max-over-positions KLD exceeds 0.05 for every selecting arm at 65536 and 131072; the gate here is the mean.
- Past 131072 the 1M figure is still the cost model's (`Perf.HattnT8.pt6_held`); the gate at 1M is the open
  measurement. No Rust/Aeneas selector; rustc, LLVM, the GPU kernels and the instrument are outside the trust boundary.

## HATTN-T8 on Qwen2.5-7B-Instruct-1M at 262144

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Same arms, same 0.05 mean-KLD gate, 64 tail positions before
`t = 262144`. Dump: f32 GPU dump `g7-256k` (`examples/gpu_dump.rs`, `KV_T=262144`, `KV_QTAIL=131072,262144`, 28 layers,
about 12 min wall, 32 GB). Predictions registered before the run: `PREDICTIONS-T8-7B-256K.md` (sha256 `02a5f3bc…`).
Instrument `attn_cut_probe.hd128qa`: `KV_TAIL_ARMS=<indices>` keeps the dense reference and the listed arms. Each arm is
its own whole-model forward, so a subset gives that arm's figures unchanged; one invocation of the six-arm list runs about
740 s, over the 600 s budget (exit 124 twice), so each arm ran as its own invocation (318–399 s each, `tail7b256b.sh`,
`g8a-<arm>.out`, binary and predictions sha256 in `tail7b256b.binding`). Verdicts: `proof/corpus/Perf/HattnT8Q8.lean`.

| arm | mean KLD | positions > 0.05 | bound vectors/refresh | flat-top-32 recall | bytes/dense |
|---|---|---|---|---|---|
| `grp-fix2048-r8` (flat) | 0.0386 | 12 | 8181.8 | – | 0.0109 |
| `tree-s4` | 0.0496 | 19 | 287.0 | 0.250 | 0.0090 |
| `tree-s8` | 0.0468 | 17 | 542.1 | 0.414 | 0.0091 |
| `clu-s4` | 0.0418 | 13 | 674.0 | 0.264 | 0.0091 |
| `clu-s8` | 0.0379 | 14 | 824.8 | 0.400 | 0.0091 |
| `clu-s4`, `HATTN_MUT=2` | 0.0787 | 24 | 599.3 | 0.000 | 0.0086 |

| prediction | verdict |
|---|---|
| Q1 dense control ≤ 1e-4 | refuted: 1.788e-4 (7.042e-5 at 131072); the probe prints `TAIL FAIL` on this clause |
| Q1 131072 tail from this dump reproduces g7-128k | held: all five arms equal (0.0242, 0.0611, 0.0549, 0.0270, 0.0227) |
| Q1 `HATTN_MUT=2` raises `clu-s4` ≥ 2× | refuted: 1.88× (0.0787 against 0.0418) |
| Q2 flat, `clu-s4`, `clu-s8` pass; flat in [0.028, 0.045] | held |
| Q3 `tree-s8` and `tree-s4` fail | refuted: 0.0468 and 0.0496 pass |
| Q4 `clu-s4` scan ≤ 1/12 of flat | held: 12.1× (674.0 against 8181.8); model `treeScan 4091 8 = 544`, `treeScan 4091 4 = 288` bound the tree scans |
| Q5 `clu-s4` recall below 0.289 | held: 0.264 |
| Q6 `clu-s4` bytes/dense ≤ 0.0100 | held: 0.0091 |

Grade `measured` for the table (one prompt, 64 positions, f32 SIMT dump); `proved` for the arithmetic in `Perf.HattnT8Q8`
(`lake build AxiomAudit` exit 0, cone ⊆ {propext}); mutation controls on throwaway copies (`tree8Kld` 468→520, `clu4Scan`
6740→7000) turn `q3_refuted`, `q4_held`, `all_pass_262144` and `cluster_scan_scaling` red, the restored file builds.

### Findings

- **All five arms pass the gate at 262144 on this sample.** `clu-s4` cuts the flat scan 12.1× at 0.0418 (9.8× at 131072); its
  scan grows 1.6× while the flat scan doubles (`cluster_scan_scaling`).
- **The tree's gate verdict is not monotone in `t`.** `tree-s8` was 0.0646 at 65536, 0.0549 at 131072, 0.0468 at 262144; positions
  above 0.05 fell from 26 to 17. The 64-position mean is carried by a few positions (max 0.37–0.70), and `tree-s4` clears the gate
  by 0.0004. With one prompt the sample cannot separate "the tree recovers" from noise; the claim that the tree fails from 65536
  is withdrawn for 262144 and stays measured for 65536 and 131072.
- **The control bar.** The dense-tail control is 1.788e-4 at 262144 against the probe's 1e-4 bar: f32 accumulation over a
  longer sequence. KLD is measured against the probe's own dense forward on the same dump, so the arms' figures stand; the
  dump's agreement with the Q4_K_M reference path at this length is not measured.
- **Bytes per generated token at 262144** (f16 KV): dense 15.03 GB; `clu-s4` at 0.0091 reads 136.8 MB, the same absolute
  figure as at 131072 (`bytes_7b_256k`).

### What this does not establish

- One prompt, decode tail only, prefill dense; 262144 at most; 1M is still the cost model's. Bytes are counts, not tokens per second.
- The cause of the tree's misses (recall equals `clu-s8`'s at 131072 yet KLD is 2.4×) is unmeasured.
- No Rust/Aeneas selector; rustc, LLVM, the GPU kernels and the instrument are outside the trust boundary.

## HATTN-T8 on Qwen2.5-7B-Instruct-1M at 524288 and 1048576

Board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Same arms, same 0.05 mean-KLD gate, 64 tail positions before `t`.
Dump: f32 GPU dump `g7-1m` (`examples/gpu_dump.rs`, `KV_T=1048576`, `KV_QTAIL=262144,524288,1048576`, 28 layers; 30 resumable
invocations of 480 s budget, about 4.1 h, GPU peak 9.2 GiB, host RSS peak 6.0 GiB, 128 GB on disk). Predictions registered
before the run: `PREDICTIONS-T8-7B-1M.md` (sha256 `9b221a17…`). Instrument `attn_cut_probe.hd128qf` (commit `d8e9898`), one arm
plus dense per invocation (`tails1m.sh`, `tails1m-b.sh`, outputs `g9-t<t>-a<arm>.out`, `.time`). Verdicts:
`proof/corpus/Perf/HattnT8Q9.lean`.

| arm, `t` | mean KLD | median | positions > 0.05 | bound vectors/refresh | flat-top-32 recall | bytes/dense |
|---|---|---|---|---|---|---|
| flat, 262144 (read from `g7-1m`) | 0.0386 | 0.0150 | 12 | 8181.8 | – | 0.0109 |
| flat, 524288 | 0.3370 | 0.2386 | 62 | 16373.8 | – | 0.0064 |
| flat, 1048576 | 0.5874 | 0.4569 | 62 | 32757.8 | – | 0.0042 |
| `clu-s4`, 524288 | 0.3237 | 0.2283 | 62 | 1188.5 | 0.219 | 0.0046 |
| `clu-s4`, 1048576 | 0.5826 | 0.5161 | 62 | 2209.2 | 0.242 | 0.0024 |
| `tree-s8`, 524288 | 0.3369 | 0.2349 | 63 | 579.0 | 0.379 | 0.0045 |
| `tree-s8`, 1048576 | 0.6004 | 0.5120 | 63 | 647.3 | 0.396 | 0.0023 |
| `clu-s4`, `HATTN_MUT=2`, 524288 | 0.5193 | 0.4028 | 63 | 1109.6 | 0.000 | 0.0044 |

| prediction | verdict |
|---|---|
| Q1 dense control ≤ 4e-4 at 524288, ≤ 8e-4 at 1048576 | refuted: 1.407e-3 and 2.650e-3 (1.788e-4 at 262144, equal to the 256k dump's) |
| Q1 `HATTN_MUT=2` raises `clu-s4` at 524288 ≥ 2× | refuted: 1.60× (0.5193 against 0.3237) |
| Q2 flat at 524288 in [0.040, 0.070] | refuted: 0.3370 |
| Q2 flat fails at 1048576 | held: 0.5874 |
| Q3 `clu-s4` fails at 524288; every other arm fails at 1048576 | held for the arms run (flat, `clu-s4`, `tree-s8`); `tree-s4` and `clu-s8` not run |
| Q4 `clu-s4` scan ≤ 1/14 of flat at 1048576 | held: 14.8× (2209.2 against 32757.8) |
| Q5 `clu-s4` recall below its 262144 value (0.264) | held: 0.242, with 0.219 at 524288 |
| Q6 `clu-s4` bytes/dense ≤ 0.0025 at 1048576 | held: 0.0024 |
| Q7 one invocation ≤ 570 s and ≤ 24 GB | held: 411 s and 20.1 GB peak resident (`/usr/bin/time -v`, `g9-t1048576-a4-rss.time`); 337 s and 425 s on the two earlier 1048576 invocations |

Control that the 1M dump is the same data as before: flat read at `t = 262144` from `g7-1m` gives 0.0386, 12 positions, bound vectors 8181.8
and control 1.788e-4, equal to the `g7-256k` run (`g9-t262144-a1.out`).
Grade `tested` for the table (one prompt, 64 positions, f32 SIMT dump); `proved` for the arithmetic in `Perf.HattnT8Q9`
(`lake build Perf.HattnT8Q9 AxiomAudit` exit 0, 30 s, 9148 jobs, cone ⊆ {propext, Classical.choice, Quot.sound}; `HattnT8Q9Mut`: `q6_held`
with `≤ 25` changed to `≤ 23` turns `decide` red).

### Findings

- **The cluster index does what the cost model says at 1M: 14.8× fewer bound vectors than the flat scan, 0.0024 of dense bytes (144 MB
  of 60.1 GB per generated token), at the flat arm's quality.** One arm plus dense runs in 411 s at 20.1 GB peak.
- **The fixed 2048-token far budget stops holding between 262144 and 524288.** Flat and `clu-s4` pass at 262144 (0.0386, 0.0418) and
  are at 0.337 and 0.324 at 524288, 0.587 and 0.583 at 1048576. At 262144 the failures were 12–13 of 64 positions (median 0.015–0.016);
  at 524288 and 1048576 they are 62–63 of 64 (median 0.23 to 0.52). The three index arms and the flat arm fail alike, so the
  index is not what fails.
- **The reversed ranking is only 1.6× worse than `clu-s4` at 524288** (1.88× at 262144, 6× at 65536): at 2048 tokens out of 524288
  the ranking separates little from its reverse.
- **The dense-tail control grows faster than the context** (1.8e-4, 1.4e-3, 2.65e-3), 7.9× then 1.9× against the registered doubling.
  KLD is measured against the probe's own dense forward on the same dump, so the arms' figures stand; the dump's agreement with the
  Q4_K_M reference path past 262144 is not measured.
- **The cause of the cliff is open.** `docs/epics/spinor.md` (FACES-R1 H9) measures the best-case read set (sinks, 256 near keys, top 2048
  of the rest) at 0.877 of the softmax mass at 262144 and 0.784 at 1048576; that is a smaller drop than the 15× in KLD.

### What this does not establish

- One prompt, decode tail only, prefill dense; the dump uses plain RoPE (base 1e7). That the checkpoint was trained to 262144 and extended past it with
  dual-chunk attention is recalled, not retrieved; the model card is unread. Bytes are counts, not tokens per second.
- `tree-s4` and `clu-s8` at these contexts, the mutation at 1048576, other prompts, a far budget that grows with `t`.
- No Rust/Aeneas selector; rustc, LLVM, the GPU kernels and the instrument are outside the trust boundary.

## Far-budget scaling at 524288 (Qwen2.5-7B-Instruct-1M)

Board row: `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`.

Question: does the gate hold at 524288 if the far budget `b` grows past 2048? Setup: flat arm and `clu-s4`, `b` replaces 2048 (`KV_TAIL_FARB`), 64 tail positions before 524288 of `/home/socol/oc-tmp/hattn-long/g7-1m`, instrument `attn_cut_probe.hd128qg`, one arm per invocation (`g10-t524288-b<b>-a<arm>.out`, 175-195 s each, TAIL FAIL verdict = exit 1). Predictions F1-F4 registered before any run: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-FARB.md`, sha256 `492f54aa…`. Gate: `cd proof/corpus && lake build Perf.HattnT8Farb AxiomAudit`, exit 0; mutation control (F2 negation removed) red.

| arm | b | mean KLD | over 0.05 | bytes/dense |
|---|---:|---:|---:|---:|
| flat | 2048 | 0.3370 | 62/64 | 0.0064 |
| flat | 4096 | 0.2979 | 62/64 | 0.0103 |
| flat | 8192 | 0.2612 | 63/64 | 0.0181 |
| flat | 16384 | 0.2218 | 62/64 | 0.0338 |
| clu-s4 | 16384 | 0.2903 | 62/64 | 0.0108 |

| prediction | verdict |
|---|---|
| F1 flat at b=4096 in [0.05, 0.25] | refuted: 0.2979 |
| F2 flat at b=16384 passes | refuted: 0.2218 |
| F3 clu-s4 within 0.02 of flat at b=16384 | refuted: 0.0685 above |
| F4 KLD monotone non-increasing in b | held: -0.0391, -0.0367, -0.0394 per doubling |

Grade: `measured` for the table (one prompt, one run per cell); `model` for the extrapolation. Each doubling of `b` buys at most 0.04. Four more doublings at that rate leave 0.0618 at b=262144, and the fifth is b=524288=t, the dense read (`Perf.HattnT8Farb.doublings_to_dense`). At 524288 on this prompt a bigger far budget does not restore the gate at sparse cost; the loss is spread over the far keys, which matches the 14-16x rise in effective key count in the FACES-R1 H9 section of `docs/epics/spinor.md`.

Not established: other prompts, tree arms, 1M, the Lean `b=2048` bytes entry is Q9's, a selector better than Quest bounds (flat-set recall of `clu-s4` at b=16384 is 0.101).

## Hot/cold page summary on Qwen2.5-7B-Instruct-1M (omitted pages summarised, not dropped)

Board row: `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Prior art searched: `fold/SPEC-COLDTIER.md` (Mode 12, proposed, not measured), `Arm::Cert` (T6). Predictions registered first: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-SUM.md`, sha256 `3396f4ba…`.

- Change: `KV_TAIL_SUM=m` on `examples/attn_cut_probe.rs`: the flat arm (`grp-fix2048-r8`, hot = 2048 exact tokens) adds, for each omitted far page, `mass · mean(v)` to the output and `mass` to the softmax denominator. Mode 0 mass `64·exp(mean page score)`; mode 1 exact page mass (ceiling for a scalar-mass summary); mode 2 `64·exp(μ + σ²/2)`.
- Command: `/home/socol/oc-tmp/hattn-long/sum.sh` (`g11-t<t>-sum<m>.out`), 64 tail positions, grade `measured`, one prompt.

| mean KLD | 262144 | 524288 |
|---|---|---|
| flat (no summary) | 0.0386 | 0.3370 |
| mode 0 | 0.0433 | 0.2438 |
| mode 1 (exact mass) | 0.0542 | 0.2389 |
| mode 2 | 3.4111 | 2.1127 |

- S1 refuted: exact mass does not halve the flat KLD (0.0542 > 0.0193; 0.2389 > 0.1685); at 262144 it is worse than no summary.
- S2 refuted: mode 1 ≤ mode 2 holds; mode 2 ≤ mode 0 does not (mode 2 is 8.7× mode 0). The variance term over-weights the high-frequency RoPE pairs' score spread.
- S3 held: no mode passes the 0.05 gate at 524288.
- S4 held (derived): bound scan 4 vectors per page, 70188 vectors per period against 8·2·524288, 0.0084 of dense.
- Finding: modes 0 and 1 differ by 0.005 at 524288. The page mass is not the bottleneck; a page-mean value does not hold what the omitted pages carry. A mean-value summary buys 0.09–0.10 at 524288 and costs 0.005–0.016 at 262144.
- Formalised: `proof/corpus/Perf/HattnT8Sum.lean` (6 theorems, `AxiomAudit.lean` rows; `lake build Perf.HattnT8Sum AxiomAudit` exit 0, `HATTNSUM-lake.log`). Mutation control: `mode1 := [150, 1500]` in a throwaway copy gives 3 errors, the committed file 0.
- Not covered: one prompt; rustc/LLVM; the summary uses the page mean value, so a per-page value weighting, a sub-page summary or a query-conditioned gather (`SPEC-COLDTIER`) is untested.

## Sub-page summary unit at 524288 (Qwen2.5-7B-Instruct-1M)

Board row: `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Predictions registered first: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-SUMU.md`, sha256 `2f795847…`.

- Change: `KV_TAIL_SUMU=u` on `examples/attn_cut_probe.rs`: the mode-1 summary (exact mass, mean value row) over units of `u` keys instead of 64. Command: `/home/socol/oc-tmp/hattn-long/sumu.sh` (`g12-t524288-sum1-u<u>.out`, 192–298 s each), grade `measured`, one prompt.

| unit u (keys) | 64 | 16 | 4 | 1 |
|---|---|---|---|---|
| mean KLD at 524288 | 0.2389 | 0.2140 | 0.1416 | 0.0000 |

- U1 held (control): `u = 1` is the exact softmax, KLD 0.0000, top-1 64/64.
- U2 held: KLD falls with the unit.
- U3 refuted: `u = 16` is 0.2140, not under 0.10.
- U4 held (derived): `u = 16` reads 0.0142 of dense, `u = 4` 0.0376.
- Finding: at 4 keys per unit the KLD is 0.1416 (2.8× the gate) and the reads already exceed 3.7% of dense. The omitted keys' values are not interchangeable at any unit cheaper than reading them; the far information is in individual (key, value) pairs, consistent with the rate check on contextual activations (§10).
- Formalised: `proof/corpus/Perf/HattnT8SumU.lean` (5 theorems, `AxiomAudit.lean` rows; build exit 0, `HATTNSUMU-lake.log`). Mutation control: `kld := [2389, 800, 400, 0]` in a throwaway copy gives 2 errors, the committed file 0.
- Not covered: one prompt; rustc/LLVM; mean value only, so a learned or query-conditioned page code is untested.

## Layer attribution of the 524288 failure (Qwen2.5-7B-Instruct-1M)

Board row: `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Predictions registered first: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-LAYER.md`, sha256 `3534046b…`.

- Change: `KV_TAIL_DENSE=l,..` runs the listed layers dense, `KV_TAIL_SPARSE=l,..` runs every other layer dense (`layer_arm`, `examples/attn_cut_probe.rs`). Command: `/home/socol/oc-tmp/hattn-long/layer.sh` (`g13-t524288-<run>.out`, 197–217 s each), flat arm `grp-fix2048-r8`, 64 tail positions, grade `measured`, one prompt.

| run | flat in layers | mean KLD at 524288 |
|---|---|---|
| all layers (Q9) | 0-27 | 0.3370 |
| D | all except 14, 21 | 0.3202 |
| Q1 | 0-6 only | 0.1821 |
| Q2 | 7-13 only | 0.0479 |
| Q3 | 14-20 only | 0.1149 |
| Q4 | 21-27 only | 0.0666 |

- L1 refuted: dense in layers 14 and 21 (where H9 saw the effective key count jump) removes 0.017 of 0.337.
- L2 held: D fails the gate.
- L3 refuted: the first quarter alone (0.1821) exceeds the third and fourth.
- L4 held (control): every single-quarter run is under the all-flat 0.3370.
- Finding: flat selection at 2048 tokens passes the gate in layers 7-13 alone (0.0479) and fails it in the other three quarters; layers 0-6 carry the most. The quarters sum to 0.4115, above 0.3370, so the shares are upper bounds.
- Formalised: `proof/corpus/Perf/HattnT8Layer.lean` (6 theorems, `AxiomAudit.lean` rows). `lake build Perf.HattnT8Layer` exit 0 and `#print axioms` within `{propext}` (`HattnLayerAudit.lean`); mutation control `quarters := [1000, …]` gives 3 errors, the committed file 0. The full `AxiomAudit` build was not run: a concurrent lane's dirty `Count/*.lean` forced a 3369-module rebuild that passed the 570 s budget (`HATTNLAYER-lake.log`, exit 124); that build stays open.

Second sweep, dense prefixes. Predictions registered first: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-LAYER2.md`, sha256 `025488e1…`. Command: `/home/socol/oc-tmp/hattn-long/layer2.sh` (`g14-t524288-E{1..4}.out`).

| run | dense layers | flat elsewhere, mean KLD at 524288 | reads/t (mean) |
|---|---|---|---|
| E1 | 0-1 | 0.3019 | 0.0756 |
| E2 | 0-3 | 0.3181 | 0.1467 |
| E3 | 0-6 | 0.3071 | 0.2534 |
| E4 | 0-6, 14-20 | 0.1033 | 0.5022 |

- M1 refuted: dense layers 0-1 give 0.3019, above 0.247.
- M2 refuted: dense layers 0-6 give 0.3071, outside [0.10, 0.20].
- M3 held: no dense-prefix run passes the 0.05 gate; half the stack dense (E4) gives 0.1033.
- M4 held: E1 removes at least half of what E3 removes from 0.3370.
- M5 refuted: not monotone, E2 (0.3181) is above E1 (0.3019).
- Finding: layers 0-6 flat alone cost 0.1821 (Q1), yet making them dense under flat elsewhere moves 0.3370 only to 0.3071. The error compounds through depth: a flat layer anywhere perturbs the residual stream every later layer reads. A per-layer dense budget does not give a passing sparse path on this prompt; the reads/t cost of E4 is 0.50 of dense.
- Formalised: `Perf.HattnT8Layer` `m1_refuted`..`m5_refuted`, `compounding`; `lake build Perf.HattnT8Layer` exit 0 (119 jobs), `#print axioms` within `{propext}`; mutation (E1 set to 0.2000) turns 2 theorems red. Full `AxiomAudit` build not run (lane contention).
- Not covered: one prompt; rustc/LLVM.

## Per-layer far budget at 524288 (Qwen2.5-7B-Instruct-1M)

Board row: `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Predictions registered first: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-LFARB.md`, sha256 `f435a574…` (addendum `88f08eab…`, registered after B1-B3 and before C1).

- Change: `KV_TAIL_LFARB=lo-hi:n` sets the far budget of layers `lo..=hi` (`layer_budget`, `examples/attn_cut_probe.rs`). Commands: `/home/socol/oc-tmp/hattn-long/lfarb.sh`, `lfarb2.sh` (`g15-t524288-{B1,B2,B3,C1}.out`), instrument `attn_cut_probe.hd128qb`; other layers keep 2048.

| run | far budget | mean KLD at 524288 | reads/t (mean) |
|---|---|---|---|
| B1 | layers 0-6 at 16384 | 0.2799 | 0.0113 |
| B2 | layers 0-6 at 65536 | 0.2091 | 0.0347 |
| B3 | layers 0-6 and 14-20 at 16384 | 0.2613 | 0.0181 |
| C1 | layers 0-6 at 524288 (every page) | 0.3071 | 0.2534 |

- N1-N5 held (N5 is weak: B1 reads 0.0113 against a 0.30 limit).
- C1 reproduces E3 (layers 0-6 dense): same KLD and reads/t, so the Grp and Dense paths agree and E1-E4 are comparable with B1-B3.
- Finding: 32x the budget in layers 0-6 (65536 tokens, 1/8 of the context) leaves 0.2091, four times the gate. Both B1 and B2 are below the exact-layer run C1: approximating the early layers does better than computing them exactly while layers 7-27 stay at 2048. Mechanism open; the flat error in later layers depends on the early layers' output.
- Formalised: `proof/corpus/Perf/HattnT8LFarb.lean` (`n1_held`..`n5_held`, `c1_held`, `nonmonotone`), `AxiomAudit.lean` rows; `lake build Perf.HattnT8LFarb` exit 0 (120 jobs), `#print axioms` within `{propext}`; mutation (B2 set to 0.3500) turns 2 theorems red. Full `AxiomAudit` build not run (lane contention).
- Not covered: one prompt; rustc/LLVM.

## Code prompt at 524288 (Qwen2.5-7B-Instruct-1M)

Board row: `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Predictions registered first: `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-CODE.md`, sha256 `df3c6740…`.

- Change: `KV_PROSE=path` in `examples/gpu_dump.rs` replaces the prompt text. Prompt `code8.txt` (Lean and Rust sources from `~/Workspace`, sorted, each file headed by its path; 5,021,138 bytes, sha256 `59eaa07e…`), 524288 tokens, same model and instrument (`attn_cut_probe.hd128qb`), `KV_TAIL_ARMS=0,1`. Commands: `dumpgpu-code.sh`, `codeprobe.sh`; output `g16-code-524288.out` (sha256 `1dd23382…`, 173 s).

| prompt | flat mean KLD | median | p90 | positions > 0.05 | top-1 |
|---|---|---|---|---|---|
| prose (`g9-t524288-a1.out`) | 0.3370 | 0.2386 | 0.5421 | 62 / 64 | 39 / 64 |
| code (`g16-code-524288.out`) | 0.1273 | 0.0015 | 0.3088 | 15 / 64 | 60 / 64 |

- K1-K4 held: control 6.6e-4 (< 1e-2); code mean 0.1273 < 0.3370; still over the 0.05 gate; top-1 60 > the registered 42 (the prose run's own figure is 39).
- Finding: the 524288 failure is not specific to prose. Code lowers the mean 2.6x and the median 160x, but the mean is 2.5x the gate: 49 of 64 positions pass and the 15 that fail carry it (max 2.46).
- Formalised: `proof/corpus/Perf/HattnT8Code.lean` (`k1_held`..`k4_held`, `between`, `heavy_tail`, `code_vs_prose`), `AxiomAudit.lean` rows; `lake build Perf.HattnT8Code` exit 0, `#print axioms` none; mutation (`flatKld` set to 0.0400 in a throwaway copy) turns `k3_held` and `between` red.
- Not covered: one code prompt and one model; the selection at a budget growing with `t`; rustc/LLVM.

## Tail positions on code at 524288: a deployable trigger (Qwen2.5-7B-Instruct-1M)

Board row: `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Predictions registered first (three rounds appended to one file; round 2 after round 1's results, round 3 after round 2's): `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-TAIL.md`, sha256 `c9d7b8a6…`.

- Change: `KV_TAIL_POS=1` in `examples/attn_cut_probe.rs` prints one `POS` line per arm and tail position: whole-model KLD, mean and max head KL (oracle), two page-bound proxies (`P1 = ln(1 + B/Z_read)`, `P2` = best unread bound minus best read score), and the arm's and dense output entropy and top-1 margin. Instrument `attn_cut_probe.hd128qp` (sha256 `1aa00486…`; two lint-only edits followed the build), dump `g7-code`, `KV_TAIL_ARMS=0,1`. Commands: `tailpos.sh` (`g17-tailpos.out`, sha256 `cd5f88b0…`, 194 s) and `tailwide.sh` (`KV_TAIL_FARB=8192` and `32768`: `g18-tailwide-8192.out` `be3cc349…`, `g18-tailwide-32768.out` `49f8063e…`, about 190 s each).
- Control: flat `b=2048` reproduces the code run (mean 0.1273, 15/64 over the gate, dense rel-L2 6.6e-4).

| score per position (15 failing, 49 passing) | deployable | AUC |
|---|---|---:|
| arm output entropy | yes | 0.978 |
| arm top-1 margin | yes | 0.965 |
| dense output entropy | no | 0.798 |
| `P1` page-bound mass (mean over 28x28 heads) | yes | 0.656 |
| `P2` bound gap | yes | 0.611 |
| mean head KL (oracle) | no | 0.600 |
| max head KL (oracle) | no | 0.407 |

- Finding: the failing positions are the ones where the approximate arm's next-token distribution is flat. Entropy >= 0.4 nats flags 12 of 64 positions and all 12 fail the gate (12 of the 15 failures). The page-bound proxies do not separate (the bound mass sits near `e^30` at every position) and the oracle head KL does not either.
- Finding: the code prompt takes a wider far budget where the prose prompt does not. Flat `b=8192` passes at 524288 (mean 0.0136, 3/64 over, top-1 64/64, 0.0181 of dense bytes); prose at the same `t` and `b` is 0.2612. Flat `b=32768`: mean 0.0023, 1/64 over, 0.0650 of dense.
- Composition (model; each arm is a separate whole-model run, so coupling through the tail keys is omitted; a flagged position pays the first pass and the wide pass):

| arm | mean KLD | over 0.05 | bytes/dense |
|---|---:|---:|---:|
| flat `b=2048` | 0.1273 | 15/64 | 0.0064 |
| flat `b=8192` | 0.0136 | 3/64 | 0.0181 |
| entropy >= 0.4 then `b=8192` | 0.0203 | 4/64 | 0.0098 |
| entropy >= 0.4 then `b=32768` | 0.0122 | 3/64 | 0.0186 |
| flat `b=32768` | 0.0023 | 1/64 | 0.0650 |

  The trigger with `b=8192` at the flagged positions has 1.85x fewer bytes than uniform `b=8192` at a mean under the gate: 102x fewer than dense.
- Verdicts: TP1 held; TP2-TP6 refuted; TP7-TP9 held; TP10-TP12 held (14/15 and 12/15 failures pass at `b=32768` and `b=8192`).
- Formalised: `proof/corpus/Perf/HattnT8Tail.lean` (`population`, `tp1_held`..`tp12_held`, `failures_are_uncertain`, `flat8192_passes_on_code`, `hybrid8192_cheaper`; AUCs are `decide +kernel` computations over the 64 recorded positions), `AxiomAudit.lean` rows; `lake build Perf.HattnT8Tail` and full `lake build AxiomAudit` (9155 jobs) exit 0, all 17 rows axiom-free; mutation (entropy list reflected, `20000 - v`, in a throwaway copy) turns `tp7_held`, `tp9_held`, `unflagged_mean`, `tp12_held`, `failures_are_uncertain` and `hybrid8192_cheaper` red (6 of 17; rerun). `cargo fmt --check` and `cargo clippy --all-targets -D warnings` exit 0 at both head dims.
- Not covered: one code prompt, one model; the 0.4 threshold was chosen on these 64 positions; the trigger needs a second forward pass at flagged positions (KV bytes are counted, weight bytes are not); no in-line hybrid run; rustc/LLVM.

## Entropy trigger on the prose prompt at 524288

Board row: `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`. Predictions (round 4 of `/home/socol/oc-tmp/hattn-long/PREDICTIONS-T8-TAIL.md`, sha256 `e045e2e7…`) registered before the runs. Same instrument (`attn_cut_probe.hd128qp`, `KV_TAIL_POS=1`), dump `g7-1m` (prose), `t = 524288`, one run per far budget: `tailprose.sh` -> `g19-prose-2048.out` (sha256 `23ddf993…`), `g19-prose-8192.out` (`0cb6da36…`), `g19-prose-32768.out` (`b1553c1d…`); 749 to 790 s wall each including lane-lock wait.

| arm | mean KLD | over 0.05 | top-1 | bytes/dense |
|---|---:|---:|---:|---:|
| flat `b=2048` | 0.3370 | 62/64 | 39/64 | 0.0064 |
| flat `b=8192` | 0.2612 | 63/64 | 44/64 | 0.0181 |
| flat `b=32768` | 0.1687 | 56/64 | 47/64 | 0.0650 |
| entropy >= 0.4 then `b=8192` (composition) | 0.2616 | 63/64 | | 0.0181 |
| entropy >= 0.4 then `b=32768` (composition) | 0.1715 | 56/64 | | 0.0650 |

- TQ1 held: the instrument reproduces 0.3370 (62/64 over).
- TQ2 held on a degenerate population: 62 failing, 2 passing positions; AUC 0.984 is computed over two negatives and carries little weight.
- TQ3 held, and the trigger is not selective: entropy >= 0.4 nats flags 62 of 64 positions and catches 61 of the 62 failures. The dense model's own output entropy on prose is already about 5 nats, so the arm's entropy does not separate approximation damage from intrinsic uncertainty there.
- TQ4 held: the gated arm (0.2616) is no better than flat `b=8192` (0.2612).
- TQ5 refuted: flat `b=32768` is 0.1687, 56/64 over. No far budget measured on prose (2048, 8192, 32768; 16x range) reaches the gate; mean KLD falls with the budget (0.3370, 0.2612, 0.1687).
- Reading: on code the trigger selects 12 of 64 positions and the gated budget passes. On prose it selects 62 of 64 and the budget does not pass. The code result stands as one prompt; the trigger as stated does not transfer to a prompt where the dense output is high-entropy. A trigger that compares the arm against itself at a second budget (entropy or top-1 disagreement between `b=2048` and `b=8192`) is a candidate that needs its own registered prediction.
- Formalised: `proof/corpus/Perf/HattnT8TailProse.lean` (sha256 `9a6836a3…`), 7 theorems (`tq1_held`, `population`, `tq2_held_degenerate`, `tq3_held_unselective`, `tq4_held`, `tq5_refuted`, `budget_monotone_not_enough`), `AxiomAudit.lean` rows; `lake build Perf.HattnT8TailProse` and `lake build AxiomAudit` (9156 jobs) exit 0, all axiom-free. Mutation (entropy list reflected, `200000 - v`, throwaway copy) turns `tq2_held_degenerate`, `tq3_held_unselective` and `tq4_held` red.
- Not covered: one prose prompt, one model; KV bytes only; composition omits tail-key coupling; rustc/LLVM and the instrument.

## Next

The cluster arms pass on Qwen2.5-7B-Instruct-1M to 131072 (section above); the tree arms stop at 65536. Next: a
measurement of which pages the tree misses (its recall matches `clu-s8` and its KLD is 2.4x), the same arms at
`2^20` at the same gate (the 262144 pass is above; GPU dump `KV_T` past 262144, attention is quadratic: ~6.8 s/layer at 131072, ~390 s/layer at `2^20`), more prompts for the tree's non-monotone verdict,
the selector in Rust under the Aeneas extraction, and tokens per second against llama.cpp on the same model (row
`hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`).

Instrument for the 7B (board row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`):
- `PROBE_HD=128` fixes `examples/attn_cut_probe.rs`'s head dim at build time, with `SCALE = 1/sqrt(HD)`. The
  default is 64.
- `KV_TAIL_TS` keeps only the listed tail contexts.
- Controls:
  - regression: the default build reproduces T8's `tree.out` line for line; only the dump path's spelling differs;
  - negative: the `PROBE_HD=128` binary refuses the 0.5B on its `head_dim` assertion, exit 101;
  - `clippy -D warnings` exits 0 at both head dims.
- Dump (`KV_DUMP`) runs per layer in three phases over token chunks of `KV_CHUNK` rows (default 8192), resumable
  inside a layer, so memory is bounded at long contexts (row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`).
  Control: Qwen2.5-0.5B, `KV_T=4096`, `KV_CHUNK=1024`, `KV_BUDGET=6` (5 invocations, 4 forced resumes): q, k, v of
  all 24 layers are byte-identical to the unchunked 32k dump's first 4096 rows (`cmp`). Clippy `-D warnings` exit 0.
  CPU cost is the limit: FFN 73% of a 150 s layer at 32k, attention `O(t^2)`; the 1M dump needs the GPU.
- Tail instrument speed (`examples/attn_cut_probe.rs`, `tail_group`, row `hattn-one-bit-attention-FDfofEhhnRjeMCZBetTGjX`):
  - change: Dense/Near/Grp/Tree/Clu arms run key-tiled per KV group (`TAIL_KT = 256` keys, scores and softmax maximum in pass 1, weights in pass 2), on an atomic work queue over (group, 2 positions); the prelude (RoPE, page boxes, row parse), `index_pages` (per KV head, k-means assignment) and the bytes loop are parallel. Arithmetic is unchanged: `dot32` lane order, separate mul/add (no FMA), sequential accumulation over j.
  - measured: 262144, arm 4 (`clu-s4`) plus dense, 32 logical CPUs (i9-14900KF), one invocation: 399 s before, 102 s after (1.55e3 s user, 1565% CPU, 1M dump running alongside).
  - control (positive): the 262144 `clu-s4` output is identical to the recorded pre-change `g8a-clu4.out` on every line except `elapsed`.
  - control (old path): `KV_TAIL_SLOW=1` keeps the per-head `tail_head` path as the comparison arm.
  - gates: `cargo fmt --all -- --check` 0; `clippy --workspace --all-features --all-targets --locked -- -D warnings` 0 (`TAILGROUP-clippy.log`).
  - not covered: rustc/LLVM codegen of the AVX2 intrinsics; memory at 1M is unmeasured.
