# The Q2 0-shot item files

Criterion **Q2** (0-shot half). These files are the benchmark input the scoring harness reads. One
JSON object per line:

```json
{"task":"boolq","item":8,"gold":1,
 "candidates":[{"prompt":"<passage>\nQuestion: <q>?\nAnswer:","continuation":" no","norm_len":2},
               {"prompt":"...","continuation":" yes","norm_len":3}]}
```

`item` is the row index in the source split, `gold` the index of the correct candidate, `prompt` and
`continuation` the lm-eval strings (`target_delimiter` `" "` is inside `continuation`), and
`norm_len` the length lm-eval divides by for `acc_norm` — the choice string, which for winogrande is
the *context*, not the shared continuation. Templates and sources are quoted in
`bench/overnight/zero-shot-plan.md` §1.

## Selection

`tools/quality-items.py` takes a centred systematic sample of the split: index
`floor((2k + 1) · total / (2 · limit))` for `k` in `0..limit`, so 200 items span the whole split
instead of following its ordering (hellaswag groups by activity). The row indices are recorded, so a
rerun reproduces the same items and both arms score identical keys.

Source is the Hugging Face datasets-server `rows` API, not a local parquet read: the repo has no
Parquet reader and this keeps the tool dependency-free. The file hash is the reproducibility anchor;
`bench/quality/items/SHA256SUMS` records it.

| file | items | choices | split | prefill tok | head tok | bytes |
|---|---:|---:|---|---:|---:|---:|
| `boolq-200.jsonl` | 200 | 2 | `aps/super_glue` boolq validation (3,270) | 60,898 | 400 | 304,830 |
| `piqa-200.jsonl` | 200 | 2 | `baber/piqa` validation (1,838) | 14,426 | 9,362 | 95,615 |
| `hellaswag-200.jsonl` | 200 | 4 | `Rowan/hellaswag` validation (10,042) | 62,619 | 23,599 | 343,891 |
| `winogrande-200.jsonl` | 200 | 2 | `allenai/winogrande` `winogrande_xl` validation (1,267) | 8,797 | 2,106 | 75,847 |
| `arc_easy-200.jsonl` | 200 | 4 | `allenai/ai2_arc` `ARC-Easy` test (2,376) | 25,177 | 3,417 | 179,936 |
| `arc_challenge-200.jsonl` | 200 | 4 | `allenai/ai2_arc` `ARC-Challenge` test (1,172) | 29,523 | 4,943 | 199,357 |
| **six** | **1,200** | | **19,965 items in the splits** | **201,440** | **43,827** | **1,099,476** |

The harness smoke (`run9`, five piqa items per arm, identical scores on both) measured **94.5 s** for
the NQ arm and **70.2 s** for the Q4 arm over 360 prefill positions — **0.263** and **0.195 s per
prefill position**, so the six-task screen is **~15 h NQ + ~11 h Q4**. The token model above is
`measured`; the wall is `model` from the smoke rate.

Token counts are `measured` with the product tokenizer:
`target/release/examples/quality --items <file> --tokenizer … --check-items` reports them per file.
No candidate exceeds `--ctx 2048` (max 484 tokens) and **no candidate needs `--rebound`**: the joined
encoding starts with the prompt encoding in all 3,600 candidates, so the strict and lm-eval boundaries
agree on this item set.

Check a file with `python3 tools/quality-items.py --check bench/quality/items/*.jsonl`.

## Scoring

```sh
target/release/examples/quality --model <gguf> [--pack <nqp>] \
  --tokenizer /home/lucas/models/arelath/nq8b/tokenizer.json \
  --items bench/quality/items/boolq-200.jsonl --out boolq-200.nq.jsonl --kernel q8act
python3 tools/quality-compare.py --nq *-200.nq.jsonl --q4 *-200.q4.jsonl
```

The harness prints one JSONL row per item and a `summary` row per task. `quality-compare.py` pairs
`(task, item)` and uses each task's primary metric (`boolq` acc; `piqa`, `hellaswag`, `arc_*`
acc_norm; `winogrande` acc).

The criterion is a mean over six task accuracies, so the **six tasks are the sampling units**: the
tool forms `d(t) = acc_nq(t) − acc_q4(t)` per task and reports a 95 % t interval over the six
differences (df 5, t 2.571), then compares it with `−0.10 × mean_t(acc_q4)`:

| verdict | condition | exit |
|---|---|---:|
| `establish` | interval lower end ≥ the bar | 0 |
| `refute` | interval upper end < the bar | 1 |
| `undecided` | otherwise | 3 |

Pooling items instead of tasks would treat a hellaswag item and a boolq item as two draws from one
population; they differ in difficulty by tens of points and the criterion means over tasks.

The sample is a centred systematic sample of each split, so the interval is a statement about the
split, not only the sampled items. The `design` line prints the sample size; a receipt from a screen
says "200 items per task" and carries the interval. A screen can refute (that is its cheap use) and
can establish when the arms track closely; it cannot do so with a wide interval, and then the answer
is more items, not a looser reading.

## The 100-item screen

`<task>-100.jsonl` is the **even-indexed half** of the 200-item file — lines 0, 2, 4, …, 198. It is the
set the v0 six-task screen scored, verified item-for-item against the twelve per-item JSONLs in
`bench/cpu/1794937300-nq8b-q2-rented-host-logs/`, so a v1 screen against it is paired with the
published 0.7534. Do not rebuild it with `quality-items.py --limit 100`: that takes a different
systematic sample, offset by half a step (boolq 16, 49, 81 … against this file's 8, 40, 73 …).

## What this does not cover

The full splits (19,965 items); item-level parity with the upstream 200-item anchor (different
checkpoint and selection); few-shot variants; the measured wall of the harness on the pair (the
five-item smoke in `bench/overnight/run9.sh` measures it; the estimate below is arithmetic on the
measured PPL rate, 542 s for 2,048 prefill + 1,023 head positions). The screen is **201,440 prefill +
43,827 head positions**: **13 h** at the measured prefill cost, **29 h** if every position is charged
at the 1.9 tok/s scored-position rate, on this laptop for the NQ arm (the earlier 45.9 h model was
1.56× high on tokens); the full split is ≈4.8 M prefill + 1.4 M head, a rented-host job.
