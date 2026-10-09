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

## Next

Measure on Qwen2.5-7B-Instruct-1M (`/home/socol/models/gguf/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf`): the T8 arms at 32k
first, then 128k and up, toward `2^20`, at the same whole-model KLD gate. Then the selector in Rust under the Aeneas
extraction, and tokens per second against llama.cpp on the same model.
