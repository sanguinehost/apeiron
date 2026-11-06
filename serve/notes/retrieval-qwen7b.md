# Retrieval with Qwen2.5-7B: measured notes

## Result

A 1,000,473-token external log is indexed, a rare-key query selects the passage containing the
needle, and Qwen2.5-7B-Instruct Q4_K_M answers from that passage:

```text
Project Zephyr's activation code is QX-4711.
```

The model did not receive the 1M-token log. It received one selected passage, the question, and
a bounded ChatML prompt.

## Corpus and index

| quantity | value | command / instrument |
|---|---|---|
| corpus tokens | 1,000,473 | `serve/tokenize`, exact Qwen BPE |
| corpus characters | 5,145,201 | index summary |
| chunks | 4,344 | 220-word windows, 32-word overlap |
| index bytes | 6,357,175 | `ls` / index summary |
| index build | 4.00 s wall, 146,952 KiB max RSS | `/usr/bin/time -v` |
| load + retrieve + prompt | 1.56 s wall, 67,152 KiB max RSS | `/usr/bin/time -v` |
| search after load | 12.585 ms mean over 100 runs | Python timer |

The generator was run with `--tokens 1047000`; its word estimate landed at 1,000,473 exact
tokens. The 1M figure is the token log, not a dense attention window.

## Retrieval checks

Each query was run against the full index with `top_k=5`, `min_score_frac=0.25`; the table gives
the rank of the first passage containing the fact and the number of returned passages.

| query | fact | rank | returned |
|---|---|---|---|
| What is Project Zephyr's activation code? | QX-4711 | 1 | 2 |
| Which code does Zephyr keep in vault seven? | QX-4711 | 1 | 1 |
| What is the Helios calibration value? | 8842-Kappa | 1 | 2 |
| What calibration does Helios record before launch? | 8842-Kappa | 1 | 2 |
| What is the Orion archive retention period? | ninety days | 1 | 1 |
| How long does Orion keep archives? | ninety days | 1 | 1 |
| What is Project Zephyr's public launch date? | 14 March | 1 | 4 |

## 7B command and output

Model: `/home/lucas/models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4.4 GB).
Backend: `llama-cli` build `b1-149f564`, CPU, two threads.

```sh
cd /home/lucas/Workspace/apeiron-serve
bash -c 'echo 1000 > /proc/self/oom_score_adj; exec nice -n 19 python3 tools/retrieval_qwen.py ask \
  --index /tmp/opencode/needle-1m.idx.json \
  --question "What is Project Zephyr'"'"'s activation code?" \
  --top-k 1 --backend llama-cli \
  --model /home/lucas/models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --max-tokens 24 --ctx 1024 --threads 2 --timeout 1200'
```

Output:

```text
Project Zephyr's activation code is QX-4711.
```

The same path was run earlier with Qwen2.5-0.5B-Instruct Q4_K_M and returned the same answer;
that only validates the prompt and output parser. No wall-clock decode measurement was recorded
for the 7B run. The run was made alone; the concurrent quality PPL queue was stopped first.

## Commands

```sh
# build the exact token counter
cargo build --release --manifest-path serve/tokenize/Cargo.toml

# generate the controlled corpus
python3 serve/tools/make_needle_corpus.py \
  --out /tmp/opencode/needle-1m.txt --tokens 1047000 \
  --tokenizer /home/lucas/Workspace/apeiron-serve/fold7b-out/tokenizer.json \
  --tokenize-bin serve/tokenize/target/release/tokenize

# build the index
python3 serve/tools/retrieval_qwen.py index \
  --corpus /tmp/opencode/needle-1m.txt --out /tmp/opencode/needle-1m.idx.json \
  --tokenizer /home/lucas/Workspace/apeiron-serve/fold7b-out/tokenizer.json \
  --tokenize-bin serve/tokenize/target/release/tokenize

# ask
python3 serve/tools/retrieval_qwen.py ask \
  --index /tmp/opencode/needle-1m.idx.json \
  --question "What is Project Zephyr's activation code?" \
  --top-k 1 --backend llama-cli \
  --model /home/lucas/models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --ctx 1024 --max-tokens 24
```

## Tests

```sh
python3 -m unittest -v serve/tools/test_retrieval_qwen.py
```

Four tests: rare-key first, second needle, absent-term empty result, ChatML prompt shape.

## What remains open

- **Semantic retrieval.** The index is lexical (BM25 plus exact phrase/rare-term boost). It has
  not been fitted or measured for paraphrases, spectral addressing, or one-million-token semantic
  recall. The seven queries above are lexical probes.
- **Exact E8 server path.** The `--backend openai` path is implemented; the measured 7B answer
  used the Q4 GGUF through `llama-cli`. The exact-pack server was not exercised through the
  wrapper.
- **Wall time and 30 tok/s.** No 7B decode-time measurement is recorded here; this note makes no
  throughput claim.
- **Hot window and span selection.** The index returns whole 220-word windows. A production
  wrapper still needs the 1024/2048-token recent window, sentence-aligned spans, and an
  insert/eviction policy.
- **1M model context.** The model context is bounded. The 1M figure is the external token log;
  this is not a dense-attention theorem.
- **Pack quality.** This change does not alter the fold pack or its measured PPL.
