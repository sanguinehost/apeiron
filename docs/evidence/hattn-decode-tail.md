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
  32k (14× fewer key reads than dense; `fix1024_fourteen_x`). The next cheaper budget fails at 8192
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

## Next

Cut the `0.031 t` bound scan, which dominates reads below a 2048-token budget: a second, coarser bound level
(pages of pages) or a bound over a quantized key so each page costs less than two vector reads. Measure against
a64-fix1024 at the same whole-model KLD.
