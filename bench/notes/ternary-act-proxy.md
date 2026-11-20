# The ternary activation is 45.7× the q8 reader's dot error on real 0.6B activations

Criterion: **Q2** (screen; not closed) and PLAN workstream W2. Scope: `Qwen3-0.6B-Base-Q4_K_M`,
the first 512 tokens of wikitext-2 `wiki.test.raw`, the `attn_norm` and `ffn_norm` stage outputs of
layers 0, 13 and 27. No PPL run, no `NanoQuant` pack, no 7B. The error model below is a projection
of per-block quantizer error onto a dot error; it is not a quality result, and Q2 stays open.

| part | grade | artifact |
|---|---|---|
| the capture: 6 stage rows, 512 columns | `measured` | `cargo run --release --locked --example act_dump -- ...`; `bench/notes/ternary-act-proxy-logs/capture.log` |
| per-block ternary, q8, cell and residual errors | `measured` | `python3 tools/ternary_act_proxy.py --dir target/act-dump`; `bench/notes/ternary-act-proxy-logs/summary.json` |
| the dot-error projection and the "within reach" reading | `model` | `bench/notes/ternary-act-proxy-logs/analysis.md`, `bench/notes/ternary-act-proxy-logs/residual-depth.txt` |
| PPL, Q2 | not measured | — |

## Result

The optimal power-of-two ternary quantizer of a 64-term block (`t ∈ {−1,0,+1}`, scale and trits
chosen to minimize the block's squared error) has a pooled relative dot RMS error of **0.5416**
against the engine q8 reader's **0.0119**: **45.7× in RMS, 2086× in squared error**. Per stage the
RMS ratio is 37.7–50.8×. It is not a depth effect: the three layers and both stages sit in
0.464–0.542.

The designed cell (`bench/notes/trichotomy-cell.md` §3: per-64 scale by the q8 rule, `M = 64`,
`τ = sign(c)` when `2|c| > 64`) is worse: **0.6114** (51.6× q8) with no correction and **0.4886**
(41.1×) with the four largest `|r|` corrected exactly. It zeroes **71%** of the activation terms.

The best 4/64 residual measured — optimal ternary scale, the four largest `|x|` of each block kept
exact — reaches **0.2916** (24.5× q8); the static top-4-by-variance mask reaches **0.3845** (32.3×).
The residual is mandatory, and a 4/64 correction is ~600× short of q8 in squared error. Matching the
q8 reader needs about **52 of 64 terms** exact (0.0147, 1.24×); 56 terms beat it (0.0082). A
`2^k` correction table cannot index 52 positions, so closing the gap means a per-term residual over
~80% of the terms — the q8 code again, read by a second row term. **The ternary cell is not
plausibly within 1.25× q8 on this proxy; the residual is mandatory and nearly total. This does not
measure PPL; the 0.6B PPL arm remains the decider.**

For context, the q8 reader's own error is 1.0–1.4% per stage, and the v3 pack that uses it scores
2.52× Q4_K_M at 8,176 tokens. The PPL pair does not isolate the q8 reader's share of that 2.52×,
so it is not a bound; it does show that the q8 reader is the baseline a ternary arm has to match to
inherit that quality, and 45.7× is a different regime.

## Capture

| stage | slot | width | feeds | model file |
|---|---:|---:|---|---|
| `attn_norm` (LN1 out) | 1 | 1,024 | q, k, v projections | `Qwen3-0.6B-Base-Q4_K_M.gguf` |
| `ffn_norm` (LN2 out) | 3 | 1,024 | gate, up projections | same |

Both requested stages are exposed by the tap; no substitute stage was used. The model is 28 layers,
width 1,024, `n_ff` 3,072. The rows are `x`, the norm output, in Q32.32; the captures' per-row RMS
is the same order as the model's own norm-gain RMS (0.193, 0.512, 1.85, 1.37, 15.13 and 7.08 for
L0 attn/ffn, L13 attn/ffn, L27 attn/ffn; the captured rows are 0.214, 0.445, 1.53, 0.934, 12.60 and
3.78), which a zeroed or mis-scaled tap would not track (`stage-stats.txt`, `gain-stats.txt`).

```sh
~/Workspace/tools/agent-guard run -- cargo run --release --locked --example act_dump -- \
  --model /home/lucas/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base-Q4_K_M.gguf \
  --tokenizer /home/lucas/models/qwen3-0.6b-base-e8/tokenizer.json \
  --data /home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw \
  --tokens 512 --chunk 32 --out /home/lucas/Workspace/apeiron-nq/target/act-dump
```

512 tokens, `KernelMode::Q8Act`, tap on, 16 calls of 32 tokens over one `DecodeState`; the tap
records every stage of every layer, so the chunk is the memory bound, not a different pass
(`prefill_tap_cols` itself splits at `prefill_block()` = 256). Measured: **45.99 tok/s, 11.1 s,
peak RSS 723.8 MiB** under a 1,519 MiB guard cap (available 2,543 MiB). The guard did not refuse,
so the capture is 512 tokens, not the 128-token fallback.

`/tmp/opencode/act-dump` refused with `EDQUOT` (the `/tmp` tmpfs `usrquota` is exhausted; a 5 MB
probe also refused with 679 MB "free"). The six 4 MiB dumps live in untracked `target/act-dump/`;
the logs are in `bench/notes/ternary-act-proxy-logs/`. The file format is one ASCII line
`apeiron-act-dump v1 stage=<tag> layer=<l> width=1024 columns=512 dtype=i64le`, then 512·1024
little-endian i64, column-major over rows: column 0's 1,024 words first.

## Method

Per 64-term block, on integer Q32.32 words, exact arithmetic:

- **ternary (a):** for each power-of-two `s = 2^e`, `e = 0..bit_length(absmax)`, each term is
  independent: `t = 0` when `2|x| < s`, else `sign(x)`. The block cost is evaluated from the sorted
  magnitudes and prefix sums as `Σ_{kept} a² − 2s·Σ_{sel} a + n_sel·s²`; the minimum over `e` is
  the optimal scale. Checked against a brute-force over every scale and every assignment on 300
  random blocks of 17–64 terms (mismatch 0).
- **q8 (b):** the engine's `quant_act` (`src/engine/chart/mod.rs:238`), one int8 code per term and
  one power-of-two scale per 32 terms: `S = 2^ceil(log2(ceil(absmax/127)))`,
  `code = sign(x)·floor((2|x| + S)/(2S))`. Checked against a direct transcription on 100 blocks of
  32–96 terms (mismatch 0).
- **cell:** the design note's §3 quantizer, `M = 64`, the q8 scale over 64 terms, and the `k`
  largest `|r| = |c − 64τ|` terms corrected exactly; `k ∈ {0,1,2,4}`.
- **residual variants:** the top-4-by-`|x|` terms of each block per column kept exact, and a static
  top-4 mask per block chosen by the variance of each position over the 512 columns (ties by
  index). Both re-optimize the ternary scale over the 60 quantized terms.
- **relative dot RMS:** `sqrt(Σ e² / Σ x²)` over all 512 columns and all blocks of a stage; the
  pooled row weights the six stages by their captured energy.

The ratio is a **model** of the dot error: with sign weights independent of the activation error,
the error of `Σ ε_i x̂_i` is `sqrt(Σ e_i²)` against a signal `sqrt(Σ x_i²)`. The NQ sign planes are
fitted to the weight matrix, not the activation, so independence is plausible but untested. The
projection says nothing about PPL.

### Optimal ternary against the q8 reader

| stage | tern rel RMS | q8 rel RMS | tern/q8 SSE | tern/q8 RMS | top4-\|x\| RMS | top4-\|x\| removed | top4-var RMS | top4-var removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 attn_norm | 0.5075 | 0.0100 | 2582.2 | 50.8 | 0.3390 | 0.554 | 0.4110 | 0.344 |
| L0 ffn_norm | 0.4642 | 0.0123 | 1419.9 | 37.7 | 0.2852 | 0.623 | 0.3680 | 0.371 |
| L13 attn_norm | 0.5136 | 0.0123 | 1747.7 | 41.8 | 0.2910 | 0.679 | 0.4571 | 0.208 |
| L13 ffn_norm | 0.4959 | 0.0125 | 1566.9 | 39.6 | 0.2927 | 0.652 | 0.4225 | 0.274 |
| L27 attn_norm | 0.5423 | 0.0117 | 2156.5 | 46.4 | 0.2956 | 0.703 | 0.3886 | 0.486 |
| L27 ffn_norm | 0.5420 | 0.0136 | 1587.5 | 39.8 | 0.2430 | 0.799 | 0.3163 | 0.660 |
| **all** | **0.5416** | **0.0119** | **2086.0** | **45.7** | **0.2916** | **0.710** | **0.3845** | **0.496** |

"removed" is the fraction of the pure ternary squared error the residual deletes. The error is
concentrated in the tail: 4 of 64 terms (6.25%) carry 71% of the pooled ternary error, which is
why the residual helps at all; it still leaves 24.5× q8.

### The designed cell

| stage | k=0 rel RMS | k=1 | k=2 | k=4 | k=4 removed | cell0/q8 SSE | zeroed frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 attn_norm | 0.5873 | 0.5599 | 0.5384 | 0.5005 | 0.274 | 3457.4 | 0.653 |
| L0 ffn_norm | 0.5520 | 0.5046 | 0.4841 | 0.4497 | 0.336 | 2008.0 | 0.708 |
| L13 attn_norm | 0.5923 | 0.5526 | 0.5250 | 0.4758 | 0.355 | 2324.5 | 0.721 |
| L13 ffn_norm | 0.5654 | 0.5265 | 0.5039 | 0.4656 | 0.322 | 2036.9 | 0.695 |
| L27 attn_norm | 0.6132 | 0.5704 | 0.5422 | 0.4938 | 0.352 | 2758.0 | 0.729 |
| L27 ffn_norm | 0.5969 | 0.5349 | 0.4955 | 0.4322 | 0.476 | 1925.6 | 0.751 |
| **all** | **0.6114** | **0.5671** | **0.5381** | **0.4886** | **0.361** | **2658.4** | **0.710** |

The cell is worse than the optimal ternary at the same k because its 64-level trit drops the
7-bit code residual for all but `k` terms, and its `M = 64` level (≈ absmax/2, zero below
absmax/4) zeroes 71% of the terms; only 0.01% of terms round to the ±127 code bound. The first four `|r|` outliers remove 36% of the cell's
error; the optimal-scale top-4 removes 71%.

### What the residual would have to cover

Optimal ternary, top-k-by-`|x|` exact terms per 64, pooled over the six stages
(`residual-depth.txt`):

| k | 4 | 8 | 16 | 24 | 32 | 48 | 52 | 56 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rel RMS | 0.2916 | 0.2258 | 0.1451 | 0.0988 | 0.0670 | 0.0225 | 0.0147 | 0.0082 | 0.0031 |
| × q8 (0.0119) | 24.5 | 19.0 | 12.2 | 8.3 | 5.6 | 1.9 | 1.24 | 0.69 | 0.26 |

The cell's correction table has `2^k` entries, 16 at k=4 and 256 at k=8; 52 positions would need
`2^52`. A per-term residual at 81% coverage is 5.7 bits/term before the masks and scale — close to
the q8 code's 8 bits (10 with its per-32 scales) — so the version that matches q8 is the q8 reader
with extra steps. The activation carrier is at most 0.011% of the weight stream
(`bench/notes/trichotomy-cell.md` §3), so the byte saving is not the argument; the instruction count
is, and a per-term residual gives it up.

## Caveats

- **One prompt, one model, 512 tokens.** The first 512 tokens of `wiki.test.raw`, Qwen3-0.6B Q4_K_M;
  no second prompt, checkpoint or seed. The stages' error ratios are stable across the six stages
  (37.7–50.8×) but nothing here covers other text or the 7B.
- **The tap's stage input is `x`, not the NQ pre-scaled `t`.** A fitted linear reads
  `t = round(s_pre·x / 2^32)` and quantizes `t`; `s_pre` is a per-linear bf16 vector, and one stage
  feeds three linears (q, k, v) or two (gate, up), each with its own `s_pre`. This screen does not
  read the pack's `s_pre`, so both quantizers see `x`. The per-block absmax and the codes would
  change under the pre-scale; the ratio's sensitivity is unmeasured. Reading `s_pre` from the pack
  is the next increment, and it must be done per linear, not per stage.
- **The rank activation is not captured.** The cell also reads the `V→U` rank vector `z`; the tap
  exposes `t`-side stages, not `z`. `z` is 640–864 wide on this pack and has a different
  distribution; its cell error is not in these rows.
- **Q4_K_M weights, not the v3 pack.** The stage inputs are the Q4 model's own trunk state. The v3
  pack sits on the F16 skeleton and its activations drift from Q4's; the PPL pair uses the v3
  activations, so this screen is a proxy for that stream.
- **The dot-error projection is a model.** It assumes sign weights independent of the per-term
  error and adds no covariance across terms or layers. It is not a PPL estimate, and no PPL was run
  here; the 0.6B arms at 2,044 and 8,176 tokens are the decider. Q2 is not claimed in either
  direction.
- **The residual variants are error-only.** The 4 exact terms per block are costed as 4 words per
  64-term activation block, shared across output rows, not as the cell's 2^k table. The static
  variant's mask is fitted on this capture (a per-layer, per-block constant), which is an oracle
  for what a fixed mask can do, not a deployable per-prompt choice.
- **Chunking.** The 32-token calls bound the tap's memory; `prefill_tap_cols` splits at 256. The
  tap-on/off equality and the block/single-column equality are `tested`
  (`tests/engine_tap.rs`); a 32-token Q8Act split is not separately tested here.

## Reproduce

```sh
~/Workspace/tools/agent-guard run -- cargo run --release --locked --example act_dump -- \
  --model <gguf> --tokenizer <tokenizer.json> --data <raw text> \
  --tokens 512 --chunk 32 --out <dump dir>
python3 tools/ternary_act_proxy.py --dir <dump dir> --json <dump dir>/summary.json
```

The dump regenerates in 11 s at 46 tok/s; `bench/notes/ternary-act-proxy-logs/` holds the capture
log, the per-stage analysis, the gain check, the JSON and the residual-depth curve. `examples/act_dump.rs` is a new
example and `tools/ternary_act_proxy.py` a new tool; no `src/` file changed.

## Follow-up: the PPL decider ran

The 0.6B v3 pack with `--kernel ternary` on 2,044 tokens at ctx 512 scores **77,741,390** against
the same-binary q8act control **72.988013**: **1,065,125×**, +13.88 nats/token
(`bench/cpu/1794834226-ternary-ppl.md`). That is the direction and a much larger magnitude than this
proxy predicted. The shipped mapping is the shipped one; a bug in the ternary reader is not excluded
by the PPL alone, but either outcome refuses it as a quality path.
