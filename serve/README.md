# Qwen 7B over a 1M-token external log

This directory is the retrieval side of the Qwen2.5-7B serving experiment. The model keeps a
bounded context. A separate exact lexical index keeps the long log; a query selects a small
number of passages, and only those passages enter the model prompt.

The change is a tool, not a listener. It does not modify the engine, the fold pack, or the
existing `apeiron-serve` chat server.

## Layout

| path | purpose |
|---|---|
| `tools/retrieval_qwen.py` | build the index, search it, build a ChatML prompt, call an OpenAI-style server or `llama-cli` |
| `tools/make_needle_corpus.py` | generate a controlled long corpus with known needles |
| `tokenize/` | a small Rust binary that counts exact Qwen BPE tokens from `tokenizer.json` |
| `notes/retrieval-qwen7b.md` | measured commands, evidence, and limits |

## Build

The tokenizer helper is a standalone package under `serve/tokenize/`:

```sh
cargo build --release --manifest-path serve/tokenize/Cargo.toml
```

The Python tools use only the standard library.

## Use

```sh
# 1. Make a controlled corpus. The target is adjusted by the generator to land near 1M tokens.
python3 serve/tools/make_needle_corpus.py \
  --out /tmp/needle-1m.txt --tokens 1047000 \
  --tokenizer /path/to/tokenizer.json \
  --tokenize-bin serve/tokenize/target/release/tokenize

# 2. Index it.
python3 serve/tools/retrieval_qwen.py index \
  --corpus /tmp/needle-1m.txt --out /tmp/needle-1m.idx.json \
  --tokenizer /path/to/tokenizer.json \
  --tokenize-bin serve/tokenize/target/release/tokenize

# 3. Ask an OpenAI-style server (for example the exact E8 pack server).
python3 serve/tools/retrieval_qwen.py ask \
  --index /tmp/needle-1m.idx.json \
  --question "What is Project Zephyr's activation code?" \
  --top-k 1 --backend openai \
  --url http://127.0.0.1:8095/v1 --model qwen2.5-7b-e8-exact

# 4. Or ask a GGUF through llama-cli.
python3 serve/tools/retrieval_qwen.py ask \
  --index /tmp/needle-1m.idx.json \
  --question "What is Project Zephyr's activation code?" \
  --top-k 1 --backend llama-cli \
  --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --ctx 1024 --max-tokens 24
```

`--dry-run --show-context` prints the selected passages and prompt without a model. The
`--tokenize-bin`/`--tokenizer` pair is optional; without it the prompt budget uses a four-
characters-per-token fallback. The exact count is preferred.

## What this is not

The index is lexical: BM25 plus an exact phrase and rare-term boost. It retrieves the exact
rare key; it is not a fitted semantic or spectral retriever. The 1M claim is a memory-hierarchy
claim about an external token log, not a claim that the model attends to 1M dense positions.
The model sees the selected passages, a bounded hot window, and the question.
