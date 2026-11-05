#!/usr/bin/env python3
"""Generate a long controlled corpus for the retrieval smoke test.

The filler is deliberately bland and repetitive; the needles are unique facts at known
fractions of the stream. The point is to test whether the index finds a far passage, not to
claim a realistic pretraining corpus.

    python3 serve/tools/make_needle_corpus.py --out needle-1m.txt --tokens 1000000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENIZE = ROOT / "tokenize" / "target" / "release" / "tokenize"
TOKENIZER = ROOT / "tokenizer.json"

FILLER = (
    "The archive maintenance log records routine shelf checks, humidity readings, and the "
    "ordinary transfer of boxes between rooms. Nothing in this entry changes a project decision. "
    "The clerk notes the aisle, the cart, and the time, then closes the page. "
)

NEEDLES = [
    (
        "Project Zephyr's activation code is QX-4711. The Zephyr team keeps the code in vault seven.",
        "zephyr-code",
    ),
    (
        "Project Zephyr's public launch date is 14 March. The launch team uses the codename Aurora.",
        "zephyr-launch",
    ),
    (
        "The Helios calibration value is 8842-Kappa. Helios operators record it before every launch.",
        "helios",
    ),
    (
        "The Orion archive retention period is ninety days. Orion deletion requires two signatures.",
        "orion",
    ),
]


def build(target_tokens: int) -> str:
    # Qwen's tokenizer is roughly 1.25-1.4 tokens per whitespace word on this text.
    target_words = max(1000, int(target_tokens * 0.78))
    filler_words = len(FILLER.split())
    repeats = max(1, target_words // filler_words)
    paras = [FILLER] * repeats
    positions = {
        "zephyr-code": int(repeats * 0.90),
        "zephyr-launch": int(repeats * 0.905),
        "helios": int(repeats * 0.50),
        "orion": int(repeats * 0.10),
    }
    # Insert in descending position order so earlier indices stay stable.
    for text, key in sorted(NEEDLES, key=lambda kv: positions[kv[1]], reverse=True):
        paras.insert(positions[key], text)
    return "\n\n".join(paras) + "\n"


def token_count(path: Path, tokenizer: Path, tokenize_bin: Path) -> int | None:
    if not tokenize_bin.is_file() or not tokenizer.is_file():
        return None
    p = subprocess.run(
        [str(tokenize_bin), "--tokenizer", str(tokenizer), "--text", str(path)],
        text=True, capture_output=True,
    )
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
        return None
    return int(json.loads(p.stdout.strip().splitlines()[0])["tokens"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--tokenize-bin", default=str(TOKENIZE))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build(args.tokens)
    out.write_text(text, encoding="utf-8")
    n = token_count(out, Path(args.tokenizer), Path(args.tokenize_bin))
    print(f"wrote {out} bytes={out.stat().st_size} tokens={n} target={args.tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
