# The benchmark protocol

Version **1**. A result is valid only under the version it names. Any change to an input, a window,
an item file, a metric or a command bumps the version and requires every candidate to be re-scored;
a number from an older version is labelled with that version.

The failure this file fixes: one pack's PPL ratio read five values within twenty-four hours, and each
value was a different tuple.

| reported | windows | bar | skeleton head | receipt |
|---|---:|---|---|---|
| 1.488698 | 1 | stock Q4, 7.587145 on that window | `Q8_0` | `1794896874-nq8b-depth.md` |
| 1.499225 | 1 | same | `Q6_K` + `Q4_K` embed | `1794896874-nq8b-depth.md` |
| 1.499754 | 5 | stock Q4, 9.919400 pooled | `Q6_K` + `Q4_K` embed | `1794912865-nq8b-overnight.md` |
| 1.417631 | 17 | stock Q4, 9.919400 pooled | same | `1794937300-nq8b-q2-rented-host.md` |
| 1.4641 | 17 | bartowski Q4, 9.604552 | same | `1794952981-nq8b-context-ranking.md` |

The numerator moved with the window set, the denominator with the bar file, and the ratio with the
head. All five scored 1,023 tokens per window; the span was constant, the other three fields were
not. Those are five tuples, not five answers. Under v1 there is one tuple per candidate, and the
tuple is what a receipt quotes.

## 1. The tuple

| field | definition |
|---|---|
| `rate` | pack bytes × 8 ÷ replaced weights; served bytes = pack + skeleton |
| `ppl` | pooled WikiText-2 PPL, full 2,048-token windows, no `--second-half` |
| `ppl/q4` | `ppl` ÷ the official `Qwen3-8B-Q4_K_M.gguf` pooled on the same windows |
| `ppl/q8` | `ppl` ÷ the official `Qwen3-8B-Q8_0.gguf` pooled on the same windows; the near-lossless reference |
| `ppl/q4b` | `ppl` ÷ bartowski `Qwen_Qwen3-8B-Q4_K_M.gguf` on the same windows; the standard bar |
| `shot` | six-task mean accuracy, 100 items per task, both arms |
| `shot/q4` | `shot` ÷ the official Q4 arm's mean |
| `tok/s` | decode at 2k, three runs, mean and range |
| `rss` | peak RSS at 2k |

The Q2 PPL verdict uses `ppl/q4`; `ppl/q4b` is reported beside it because the two Q4 files differ by
3.2 % on the same harness. The OG NanoQuant row uses `ppl/q8` and `shot/q4` against the published
1.78 and 0.687, and is labelled `cross-harness` until `SamsungLabs/NanoQuant`'s `evaluate_ppl` loop
is run on this checkpoint.

## 2. Inputs

| input | path | sha256 |
|---|---|---|
| text | `/home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw` | `173c87a53759e020…` |
| tokenizer | `/home/lucas/models/arelath/nq8b/tokenizer.json` | recorded in the run log |
| items | `bench/quality/items/<task>-100.jsonl` (screen), `<task>-200.jsonl` (confirm) | `bench/quality/items/SHA256SUMS` |
| bar | `Qwen3-8B-Q4_K_M.gguf` | `d98cdcbd03e17ce4…` |
| standard bar | `Qwen_Qwen3-8B-Q4_K_M.gguf` (bartowski) | `54fffa050078e984…` |
| near-lossless | `Qwen3-8B-Q8_0.gguf` | recorded in the run log |
| skeleton | `/home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf` | recorded in the run log |
| pack | `/home/lucas/models/arelath/nq8b-late-v4/model.nqp` | `271fd5a7…` |

The bar files are the two Q4_K_M images because GOAL Q2 names one Q4_K_M and the standard image is
3.2 % better in PPL than the official one. A receipt that quotes only one says which.

## 3. PPL

Screen and ladder: five full windows, `--tokens 2048 --skip S --ctx 2048 --head-batch 16`, for
`S ∈ {0, 2048, 4096, 6144, 8192}`. Confirmatory run: the seventeen windows
`{0, 2048, 4096, 6144, 8192, 10240, 22016, 33792, 45568, 57344, 69120, 80896, 92672, 104448,
116224, 128000, 139776}`. Every arm on the same list. No `--second-half`: the paper's
`evaluate_ppl` scores all `seqlen − 1` positions of each window, and llama.cpp's `perplexity` does
too, so the full window is the only span comparable to either. The `--second-half` lines in
`bench/cpu/1794937300-nq8b-q2-rented-host.md` are **legacy v0** and are not the verdict.

```sh
target/release/examples/ppl --model <gguf> [--pack <nqp>] \
  --tokenizer <tokenizer.json> --data <wiki.test.raw> \
  --tokens 2048 --skip <S> --ctx 2048 --head-batch 16 [--kernel q8act]
```

Pool with `python3 tools/ppl-pooled.py --nq <candidate logs> --q4 <bar logs>`; the pooled PPL is
`exp(Σ nll_i / N)`, not the mean of the per-window PPLs. Every table row carries the window count and
the pooled value, not a per-window best.

## 4. 0-shot

`target/release/examples/quality --items bench/quality/items/<task>-100.jsonl` — the even-indexed half
of the 200-item file and the set the v0 screen scored — both arms, all six tasks. The mean is
unweighted over tasks; the interval is the paired t over the six task differences
(`tools/quality-compare.py`). The confirmatory run reads `<task>-200.jsonl`. Do not use `--limit 100`:
it takes a different systematic sample, offset by half a step, so the screen would no longer be
paired with the published 0.7534. The v0 per-item JSONLs are the record of the same 100 items.

## 5. Decode

`examples/phase_split --steps 24` and a timed decode at `--ctx 2048`, three runs. Report the phase
split, the target phase's share and the `1/(1−s)` ceiling beside the tok/s. Quote the bytes-per-token
ceiling and the measured bandwidth in the same sentence; a target above the ceiling is refused.

## 6. Result record

`bench/results/<pack-sha12>.json`:

```json
{"protocol":1,"pack_sha256":"…","skeleton_sha256":"…","rate":1.178785,
 "served_bytes":1891196336,"windows":5,
 "ppl":{"nq":0.0,"q4":0.0,"q8":0.0,"q4b":0.0},
 "shot":{"nq":0.0,"q4":0.0},
 "decode":{"tok_s":0.0,"range":[0.0,0.0]},"host":"…","binary_sha256":"…"}
```

`tools/bench-check.py` recomputes the pooled values from the logs named in the record and exits 1 if
a hash or a pooled value disagrees. A record without its logs is not a result.

## 7. Why this is the pipeline

The pieces already existed: `tools/ppl-pooled.py` pools, `tools/quality-compare.py` tests, the item
files are hashed, and the per-campaign drivers under `bench/overnight/` ran them. What did not exist
was a frozen input list, a fixed window list, a fixed span and a result record that a checker could
recompute. v1 is those four things. The next campaign runs `bench/run-ppl.sh` and
`bench/run-screen.sh`, writes one JSON, and every answer quotes that JSON and its protocol number.
