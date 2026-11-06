#!/usr/bin/env python3
"""Retrieval over a long token log for the local Qwen 7B server.

The model keeps a bounded context; this tool keeps the full corpus outside the model and
sends only the passages a query selects. It is deliberately dependency-free (Python stdlib).

    # Build an index over a corpus (word windows; exact Qwen token count when `tokenize` exists)
    python3 serve/tools/retrieval_qwen.py index --corpus long.txt --out long.idx.json

    # Retrieve and ask a local OpenAI-style server
    python3 serve/tools/retrieval_qwen.py ask --index long.idx.json \
        --question "What is Project Zephyr's activation code?" \
        --backend openai --url http://127.0.0.1:8095/v1 --model qwen2.5-7b-e8-exact

    # Or ask a GGUF through llama-cli
    python3 serve/tools/retrieval_qwen.py ask --index long.idx.json --question ... \
        --backend llama-cli --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf

The index is lexical (BM25 + exact phrase/rare-term boost). That is the exact/rarest-key lane;
a semantic/spectral lane is a separate index and is not pretended to exist here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = ROOT / "tokenizer.json"
DEFAULT_TOKENIZE_BIN = ROOT / "tokenize" / "target" / "release" / "tokenize"
DEFAULT_LLAMA = Path(os.environ.get("LLAMA_CLI", "llama-cli"))
TMP = Path(tempfile.gettempdir()) / "apeiron-retrieval"
TMP.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")
SPAN_RE = re.compile(r"\S+")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "he", "her", "his", "i", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "you", "your", "we", "our", "us",
    "do", "does", "did", "can", "could", "should", "would", "will", "not", "no",
}


def lex(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def token_count(text: str, tokenizer: Path, bin_path: Path) -> int | None:
    """Exact Qwen token count, or None when the helper is unavailable."""
    if not bin_path.is_file() or not tokenizer.is_file():
        return None
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=TMP, delete=False) as f:
        f.write(text)
        name = f.name
    try:
        p = subprocess.run(
            [str(bin_path), "--tokenizer", str(tokenizer), "--text", name],
            text=True, capture_output=True, timeout=120,
        )
    finally:
        os.unlink(name)
    if p.returncode != 0:
        raise RuntimeError(f"tokenize failed: {p.stderr.strip()}")
    line = p.stdout.strip().splitlines()[0]
    return int(json.loads(line)["tokens"])


def split_chunks(text: str, chunk_words: int, overlap_words: int) -> list[dict]:
    if chunk_words <= 0:
        raise ValueError("chunk-words must be positive")
    overlap = max(0, min(overlap_words, chunk_words - 1))
    step = chunk_words - overlap
    spans = list(SPAN_RE.finditer(text))
    if not spans:
        return []
    chunks = []
    for start in range(0, len(spans), step):
        end = min(start + chunk_words, len(spans))
        if end <= start:
            break
        a, b = spans[start].start(), spans[end - 1].end()
        chunks.append({
            "id": len(chunks),
            "start_word": start,
            "end_word": end,
            "text": text[a:b],
        })
        if end == len(spans):
            break
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tfs: list[Counter] = []
        self.lens: list[int] = []
        self.chunk_lex: list[str] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            terms = lex(c["text"])
            tf = Counter(terms)
            self.tfs.append(tf)
            self.lens.append(sum(tf.values()))
            self.chunk_lex.append(" ".join(terms))
            for term in tf:
                self.postings[term].append(i)
        self.n = len(chunks)
        self.avgdl = (sum(self.lens) / self.n) if self.n else 1.0

    def search(self, question: str, top_k: int = 4,
               min_score_frac: float = 0.25) -> list[dict]:
        qterms = lex(question)
        if not qterms:
            return []
        seen = set()
        terms = []
        for t in qterms:
            if t in seen:
                continue
            seen.add(t)
            terms.append(t)
        scored = []
        phrase = " ".join(qterms)
        rare = min(terms, key=lambda t: len(self.postings.get(t, ())))
        for i, tf in enumerate(self.tfs):
            dl = self.lens[i] or 1
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                df = len(self.postings.get(t, ()))
                idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
                s += idf * (f * (self.k1 + 1.0)) / (
                    f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                )
            if not s:
                continue
            if phrase and phrase in self.chunk_lex[i]:
                s += 8.0
            if rare in tf:
                s += 1.0
            scored.append((s, i))
        scored.sort(reverse=True)
        if not scored:
            return []
        cutoff = scored[0][0] * max(0.0, min_score_frac)
        out = []
        for s, i in scored[: max(1, top_k)]:
            if s < cutoff:
                break
            c = dict(self.chunks[i])
            c["score"] = s
            out.append(c)
        return out


def load_index(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def chatml(messages: list[dict]) -> str:
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    out.append("<|im_start|>assistant\n")
    return "".join(out)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    system = (
        "You answer from the retrieved passages only. "
        "If the answer is not in the passages, reply exactly NOT_IN_CONTEXT. "
        "Do not use outside knowledge."
    )
    if hits:
        blocks = [f"[{i}] {h['text']}" for i, h in enumerate(hits, 1)]
        context = "\n\n".join(blocks)
    else:
        context = "(no passages retrieved)"
    user = (
        f"Retrieved passages:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer using only the passages above. If absent, say NOT_IN_CONTEXT."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def trim_hits(question: str, hits: list[dict], max_tokens: int,
              tokenizer: Path, tokenize_bin: Path) -> tuple[list[dict], int | None]:
    """Drop lowest-ranked passages until the prompt fits the context budget."""
    work = list(hits)
    while True:
        msgs = build_messages(question, work)
        raw = chatml(msgs)
        n = token_count(raw, tokenizer, tokenize_bin)
        if n is None:
            # Conservative fallback: four characters per token.
            n = (len(raw) + 3) // 4
        if n <= max_tokens or not work:
            return work, n
        work.pop()


def call_openai(url: str, model: str, messages: list[dict], max_tokens: int,
                timeout: float) -> str:
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"backend HTTP {e.code}: {detail}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"backend response has no message content: {data!r}") from e


def call_llama_cli(llama: Path, model: Path, messages: list[dict], max_tokens: int,
                   ctx: int, threads: int, timeout: float) -> str:
    prompt = chatml(messages)
    cmd = [
        str(llama), "-m", str(model), "-p", prompt,
        "-n", str(max_tokens), "-c", str(ctx),
        "-t", str(threads), "--temp", "0", "--top-k", "1", "--top-p", "1",
        "--seed", "0", "--no-display-prompt", "--no-show-timings", "--no-jinja",
        "--color", "off", "--log-disable", "--no-warmup", "-st",
    ]
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"llama-cli failed ({p.returncode}): {p.stderr.strip()}")
    # This llama-cli build still echoes the prompt and banner in its single-turn mode. The
    # generated text is what follows the last assistant marker; cut the timing trailer too.
    out = p.stdout
    if "\n> " in out:
        out = out.split("\n> ", 1)[1]
    if "... (truncated)\n" in out:
        out = out.split("... (truncated)\n", 1)[1]
    else:
        marker = "<|im_start|>assistant\n"
        if marker in out:
            out = out.rsplit(marker, 1)[1]
    out = re.split(r"\n\s*> |\nExiting\.\.\.|\n\s*\[ Prompt:", out)[0]
    out = out.replace("<|im_end|>", "").strip()
    return out


def cmd_index(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    text = corpus.read_text(encoding="utf-8")
    chunks = split_chunks(text, args.chunk_words, args.overlap_words)
    ntok = token_count(text, Path(args.tokenizer), Path(args.tokenize_bin))
    idx = {
        "version": 1,
        "corpus": str(corpus),
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "tokens": ntok,
        "characters": len(text),
        "chunks": chunks,
        "built_unix": int(time.time()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    print(json.dumps({
        "index": str(out),
        "chunks": len(chunks),
        "tokens": ntok,
        "characters": len(text),
        "bytes": out.stat().st_size,
    }, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    idx = load_index(Path(args.index))
    r = Retriever(idx["chunks"])
    hits = r.search(args.question, args.top_k, args.min_score_frac)
    hits, prompt_tokens = trim_hits(
        args.question, hits, args.max_prompt_tokens,
        Path(args.tokenizer), Path(args.tokenize_bin),
    )
    messages = build_messages(args.question, hits)

    if args.show_context or args.dry_run:
        for h in hits:
            print(f"[{h['id']}] score={h['score']:.3f} words={h['start_word']}..{h['end_word']}")
            print(h["text"][:500].replace("\n", " "))
            print()
        print(f"prompt_tokens={prompt_tokens} context_chunks={len(hits)}", file=sys.stderr)

    if args.dry_run:
        print(chatml(messages))
        return 0

    if args.backend == "openai":
        answer = call_openai(args.url, args.model, messages, args.max_tokens, args.timeout)
    elif args.backend == "llama-cli":
        answer = call_llama_cli(
            Path(args.llama), Path(args.model), messages,
            args.max_tokens, args.ctx, args.threads, args.timeout,
        )
    else:
        raise ValueError(f"unknown backend {args.backend!r}")

    print(answer)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ix = sub.add_parser("index", help="build a lexical index over a long corpus")
    ix.add_argument("--corpus", required=True)
    ix.add_argument("--out", required=True)
    ix.add_argument("--chunk-words", type=int, default=220)
    ix.add_argument("--overlap-words", type=int, default=32)
    ix.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ix.add_argument("--tokenize-bin", default=str(DEFAULT_TOKENIZE_BIN))
    ix.set_defaults(func=cmd_index)

    ask = sub.add_parser("ask", help="retrieve passages and ask Qwen")
    ask.add_argument("--index", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--top-k", type=int, default=4)
    ask.add_argument("--min-score-frac", type=float, default=0.25,
                     help="drop hits below this fraction of the best BM25 score")
    ask.add_argument("--max-prompt-tokens", type=int, default=1700)
    ask.add_argument("--max-tokens", type=int, default=64)
    ask.add_argument("--backend", choices=["openai", "llama-cli"], default="openai")
    ask.add_argument("--url", default="http://127.0.0.1:8095/v1")
    ask.add_argument("--model", default="qwen2.5-7b-e8-exact")
    ask.add_argument("--llama", default=str(DEFAULT_LLAMA))
    ask.add_argument("--ctx", type=int, default=2048)
    ask.add_argument("--threads", type=int, default=4)
    ask.add_argument("--timeout", type=float, default=900.0)
    ask.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ask.add_argument("--tokenize-bin", default=str(DEFAULT_TOKENIZE_BIN))
    ask.add_argument("--show-context", action="store_true")
    ask.add_argument("--dry-run", action="store_true")
    ask.set_defaults(func=cmd_ask)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
