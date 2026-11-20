# Q2 0-shot: the six-task rule, a harness, and what 200 items decide

Criterion **Q2**. Grades: `model` for template-derived token counts and hours; `measured` for rates
carried in from receipts; `anchor` for upstream numbers. No engine 0-shot measurement exists; this
file designs one and adds no row to Q2.

## 1. Scoring rule

Q2 is the mean over six tasks of each task's lm-eval primary metric: boolq `acc`, piqa `acc_norm`,
hellaswag `acc_norm`, winogrande `acc` (`winogrande_xl`), arc_easy `acc_norm`, arc_challenge
`acc_norm`. All six are multiple choice; splits and item counts are in §3. For candidate `c`:

```text
ll(c)    = Σ_{t∈s(c)} ln p(t | prompt(c), s(c)[:t])   f64; one Q32.32 softmax per position
acc      = argmax_c ll(c)                             lm-eval "acc"
acc_norm = argmax_c ll(c) / chars(choice_text(c))     lm-eval "acc_norm"
acc_tok  = argmax_c ll(c) / tokens(s(c))              the upstream anchor's "normalized"
```

`acc_norm` divides by `len(choice_string)` (`completion_len`, `lm_eval/api/task.py`; first maximum
wins a tie) — the context for winogrande, not the continuation. It is not a byte or token mean; the
anchor's per-token rule is reported as `acc_tok`, never mixed with it.

### Templates (quoted from `lm-evaluation-harness` `main`, 2026-11-17)

`target_delimiter` is `" "` (`lm_eval/config/task.py:112`); the candidate is `doc_to_text + " " +
choice`, and the delimiter is part of the scored continuation.

| task | `doc_to_text` | choices / gold | scored continuation | source |
|---|---|---|---|---|
| boolq | `"{{passage}}\nQuestion: {{question}}?\nAnswer:"` | `["no", "yes"]`; `label` (bool) | `" no"` / `" yes"` | `lm_eval/tasks/super_glue/boolq/default.yaml` |
| piqa | `"Question: {{goal}}\nAnswer:"` | `"{{[sol1, sol2]}}"`; `label` | `" " + sol_i` | `lm_eval/tasks/piqa/piqa.yaml` |
| hellaswag | `"{{query}}"`, `query = activity_label + ": " + ctx_a + " " + ctx_b.capitalize()` | `choices`; `"{{label}}"` | `" " + choice` | `lm_eval/tasks/hellaswag/hellaswag.yaml`, `utils.py` |
| winogrande | gold index (0/1); `multiple_input` | contexts `sentence[:blank] + option_i` | `" " + sentence[blank+1:].strip()` | `lm_eval/tasks/winogrande/default.yaml`, `preprocess_winogrande.py` |
| arc_easy, arc_challenge | `"Question: {{question}}\nAnswer:"` | `choices.text`; `choices.label.index(answerKey)` | `" " + choice.text` | `lm_eval/tasks/arc/arc_easy.yaml`, `arc_challenge.yaml` |

Hellaswag `preprocess`: strip; `" [title]"` → `". "`; delete `\[.*?\]`; collapse `"  "`. Winogrande
gold is `{"1":0,"2":1}[answer]`; its continuation is shared and the contexts are the candidates.
Qwen3: **no BOS** (`bos_token_id` null); EOS `<|im_end|>` 151645; pad `<|endoftext|>` 151643
(`~/models/arelath/nq8b/tokenizer.json`); `Bpe::encode` splits added tokens literally. Enumerate in
dataset order; ties are counted.

## 2. Harness: `examples/quality.rs`

No listener. One process per (arm, task) shard; items in-process; the engine opens once (2.3 GB
mapped, ~3 s). A fresh `DecodeState` per candidate: `prefill_states` appends at the state's length,
so reuse would continue the previous item. Exact engine surface:

```text
Engine::open(path: &str) -> Result<Engine, EngineError>
nq::open(path) -> Result<NqPack, _>
Engine::open_with_nq(gguf: &str, pack: NqPack) -> Result<Engine, EngineError>
Engine::state(&self, cap: usize) -> Result<DecodeState, StepError>
Engine::prefill_states(&self, st: &mut DecodeState, tokens: &[u32], mode: KernelMode)
    -> Result<Vec<Vec<i64>>, StepError>
Engine::lm_head_cols(&self, xs: &[Vec<i64>], mode: KernelMode) -> Result<Vec<Vec<i64>>, LmHeadError>
Bpe::from_tokenizer_json(path: &str) -> Result<Bpe, BpeError>
Bpe::encode(&self, text: &str) -> Result<Vec<u32>, BpeError>
```

`engine.config.{vocab_size,n_embd,context_length}` gate the item; `--kernel q8act` for both arms
(the PPL claim's kernel), `exact` on 100 items bounds the reader. The JSONL item file is built once
per task by `tools/quality-items.py` from the hub parquet, no Jinja in Rust: `{"task","item","gold",
"candidates":[{"prompt","continuation","norm_len"}]}`, `norm_len` = lm-eval choice length (default:
the continuation's).

Per candidate: `p = bpe.encode(prompt)`, `w = bpe.encode(prompt + continuation)`; require
`w[..p.len()] == p` (no merge crossed the split), non-empty span, `w.len() <= ctx`; then `st =
engine.state(w.len())`, `cols = engine.prefill_states(&mut st, &w, mode)`, slice
`cols[p.len()-1 .. w.len()-1]` in `--head-batch 32` chunks, and `lm_head_cols`. Row `c` predicts
`w[p.len()+c]`; `lp = log_softmax` in f64 with the body of `examples/ppl.rs:116` (scale 2⁻³²,
subtract max, subtract `ln Σ exp`); `ll += lp`, `n_tok = w.len()-p.len()`, `n_bytes = continuation.len()`.

Output JSONL per item: `task,item,gold,ll[],n_tok[],n_bytes[],norm_len[],correct{acc,acc_norm,
acc_tok},refused,truncated,seconds`; summary per task `n,acc,acc_norm,acc_tok,ties,refused`, across
tasks the six-task mean and, with `--q4`, per-task `nq/q4` ratios and the mean ratio. Refusals
(recorded; exit 1 above `--max-refused`): boundary (default; `--rebound` scores lm-eval's way),
empty span, `id >= vocab`, over `ctx` (left-truncate context as lm-eval, mark `truncated`), wrong
choice count, gold out of range, missing field, `LmHeadError::ShapeMismatch`. A ratio over a
different `(task,item)` set prints the mismatch and exits 2; the verdict and exit codes are in §5. Cost
`O(Σ L)`, one prefill per candidate (no cache; 2–4× lever, `model`).

## 3. Runtime

Rates `measured`: **1.9 tok/s** scoring (late-v4 PPL, ctx 2048, 542.0 s / 1,023 positions,
`1794912865-nq8b-overnight.md`); 4.66 tok/s prefill and 2.25 tok/s decode (8B 1.124 pack,
`1794879503-nq8b-1124-ctx2048.md`, also 8.6 tok/s Q4); 0.6B 16.2 tok/s NQ and 25.7 Q4
(`1794822341-nq06-ppl-vs-q4.md`). Tokens are `model` at 1.35 tokens/word, ±25%; hours are `model` =
tokens ÷ measured rate.

| task (split) | items | tok/item | 200-item ktok | h @1.9 | full MToken | h @1.9 |
|---|---:|---:|---:|---:|---:|---:|
| boolq (val) | 3,270 | 600 | 120 | 17.5 | 1.962 | 287.0 |
| piqa (val) | 1,830 | 100 | 20 | 2.9 | 0.183 | 26.8 |
| hellaswag (val) | 10,042 | 420 | 84 | 12.3 | 4.218 | 616.7 |
| winogrande (val) | 1,267 | 110 | 22 | 3.2 | 0.139 | 20.4 |
| arc_easy (test) | 2,376 | 160 | 32 | 4.7 | 0.380 | 55.6 |
| arc_challenge (test) | 1,172 | 180 | 36 | 5.3 | 0.211 | 30.8 |
| **six** | **19,957** | — | **314** | **45.9** | **7.093** | **1,037 (43.2 d)** |
| **six, measured tokens** | **19,965** | **168** | **201** | **13–29** | **4.79** | **310–700** |

The last row replaces the ±25 % `model` counts with `measured` ones: the six item files carry
**201,440 prefill and 43,827 head positions** at 200 items per task
(`bench/quality/README.md`, `quality --check-items`). The model column was 1.56× high. The hour span is
a floor at the pair's measured prefill cost (4.66 tok/s for the 201 k prefill positions plus 0.10 s per
batched head row) and a ceiling that charges every position at the PPL scored-position rate (1.9
tok/s): **13–29 h** for the screen, not 45.9 h. The full split is 310–700 h on this laptop. Items:
19,965, not 19,957 (the datasets-server splits are boolq 3,270, piqa 1,838, hellaswag 10,042,
winogrande 1,267, Arc-Easy 2,376, Arc-Challenge 1,172).

**Measured, not modelled.** The six item files carry 201,440 prefill + 43,827 head positions and
run in 13-29 h per arm on this laptop (`bench/quality/README.md`); the model column above was 1.56×
high on tokens. Both arms are queued as `bench/overnight/run11.sh` (NQ) and `run12.sh` (Q4), one job
per task with a marker, cheapest task first, so the queue resumes after a reboot. The full split is
310-700 h per arm here, a rented-host job.

Dataset files are **absent**; 8,565,369 B total (hub listing, `anchor`), pinned revisions:
`huggingface.co/datasets/aps/super_glue/tree/3de24cf8022e94f4ee4b9d55a6f539891524d646` 1,313,731 B;
`.../baber/piqa/tree/142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` 299,694 B;
`.../Rowan/hellaswag/tree/218ec52e09a7e7462a5400043bb9a69a41d06b76` 6,315,951 B;
`.../allenai/winogrande/tree/01e74176c63542e6b0bcb004dcdea22d94fb67b5` (`xl`) 85,928 B;
`.../allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453` 346,257 B + 203,808 B.

## 4. Comparison arm

Required denominator: `Qwen3-8B-Q4_K_M.gguf`, 5,027,783,488 B, sha256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`. **Present and verified** at
`~/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf`; the PPL arm has run on it in five windows, and the window-1
run reproduces the rented host's 7.587145 to the last digit. (The earlier plan recorded a 374,206,464 B
truncated local file; that was replaced by the full download, and the 32 B-off Base checkpoint is
still excluded by Q2's same-checkpoint clause.)

Both arms run the same `tokenizer.json` (`nq8b/tokenizer.json`, the Qwen3 vocabulary); `quality.rs`
refuses any id at or above the model's `vocab_size`. Both arms run identical item keys; report
`ratio(task) = acc_nq/acc_q4` and `mean_t(acc_nq)/mean_t(acc_q4)`.

The anchor is not this denominator: dense `Qwen/Qwen3-8B` vs the nanoquant GGUF under llama.cpp at
`task_limit: 200`, `selection_seed: 0` (`anchor`; a different 200 items). Its mean is 0.7225 → 0.485
= **0.671×**; its `acc_norm` is per token, ours per character.

Both arms run identical item keys; report `ratio(task) = acc_nq/acc_q4` and
`mean(acc_nq)/mean(acc_q4)`. The anchor is not this denominator: dense `Qwen/Qwen3-8B` vs the
nanoquant GGUF under llama.cpp at `task_limit: 200`, `selection_seed: 0` (`anchor`; a different 200
items). Its mean is 0.7225 → 0.485 = **0.671×**; its `acc_norm` is per token, ours per character.

## 5. Decision: screen or establish?

The criterion is a mean over six task accuracies, so the six tasks are the units of analysis. The
paired difference is `d(t) = acc_nq(t) − acc_q4(t)` on identical items, and its interval is
`mean_t d ± t_{0.975,5} · sd_t(d)/√6` = `mean_t d ± 2.571 · sd_t(d)/√6` (df 5). The bar is
`−0.10 · mean_t(acc_q4)`.

- The width is driven by `sd_t(d)`, the **between-task spread of the decrement**, not by the item
  count. If the 1-bit pack degrades all six tasks by a similar amount, `sd_t(d)` is small and 200
  items per task settles the question; if it degrades boolq by 25 points and winogrande by 2, only
  the full split will.
- Same-item pairing removes the between-item variance: the arms see identical prompts, so `d(t)` is
  not charged the task's difficulty.
- `tools/quality-compare.py` prints the interval, the task sd, the bar, the verdict and the design,
  and exits 0 (establish), 1 (refute) or 3 (undecided).
- `model` expectation for `sd_t(d)`: the upstream 200-item anchor is a mean of per-task ratios that
  spans 0.55 (boolq) to 0.80 (winogrande)-scale behaviour, so the between-task spread is not
  negligible. A refutation needs `mean_t d + half < −0.10·mean_t acc_q4`; with `mean_t acc_q4 ≈ 0.72`
  that is `mean_t d < −0.072 + half`, so a fetch well short of the anchor's −0.24 is still decisive.
- **The screen is the cheap refutation.** If the pack scores anywhere near the anchor's 0.671, 1,200
  paired items refute Q2's 0-shot half in ~14 h per arm; only a near-tie needs the full split.

## 6. The cost lever not yet built: one prompt prefill per item

`examples/quality.rs` calls `score_candidate` per candidate, and each call prefills the whole joined
string. For a multiple-choice item the prompt is identical across choices, so it is prefilled once
per choice. Sharing it — prefill the prompt once, clone the `DecodeState` per candidate, prefill only
the continuation — cuts the prefill columns from 201,440 to 75,789 (the per-task prompt is
`prefill_tok / choices`; winogrande is not shared because its two options sit in the prompt). The
43,827 head columns are unchanged.

| task | prefill tok | current cols | shared cols | choices |
|---|---:|---:|---:|---:|
| boolq | 60,898 | 61,298 | 30,849 | 2 |
| piqa | 14,426 | 23,788 | 16,575 | 2 |
| hellaswag | 62,619 | 86,218 | 39,254 | 4 |
| winogrande | 8,797 | 10,903 | 10,903 | 2 |
| arc_easy | 25,177 | 28,594 | 9,711 | 4 |
| arc_challenge | 29,523 | 34,466 | 12,324 | 4 |
| **six** | **201,440** | **245,267** | **119,616** | |

A prefill column costs 0.239 s and a head column 0.035 s on the NQ arm (the 5-item smoke's 94.5 s
solves those two rates from its 361 prefill + 234 head columns); the Q4 prefill column is 0.195 s.
The NQ screen is then `201,440×0.239 + 43,827×0.035 = 49,678 s` now against
`75,789×0.239 + 43,827×0.035 = 19,648 s` shared, and the Q4 screen 40,815 s against 16,313 s:
**~25 h both arms now, ~10 h shared (2.5×)**. That is the difference between two laptop nights and
one. Grade `model` until the path is built; it is an example-only change, it must reproduce the
5-item smoke's `ll` values bit-for-bit before it is used, and it does not touch the product kernel.
