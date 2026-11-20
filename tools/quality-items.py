#!/usr/bin/env python3
"""Build lm-eval-compatible 0-shot item files for the Q2 six-task screen.

The item file is the benchmark input: one JSON object per line, in the schema the scoring harness
reads. It is built once, hashed, and shared by both arms, so the NQ/Q4 ratio is paired on identical
items. Templates follow `bench/overnight/zero-shot-plan.md` §1, which quotes
`lm-evaluation-harness` `main`.

Source: the Hugging Face datasets-server rows API (parquet reading needs no local wheel). The row
index is the item key and is stable for a pinned split; the file hash is the reproducibility anchor.

Selection: a centred systematic sample of `--limit` items over the split, index
`floor((k + 0.5) · total / limit)` for `k` in `0..limit`. It spans the split instead of taking a
contiguous prefix, which would follow the split's ordering (hellaswag groups by activity).

Usage:
    python3 tools/quality-items.py --task hellaswag --limit 200 --out bench/quality/items/hellaswag-200.jsonl
    python3 tools/quality-items.py --task boolq --all --out ...
    python3 tools/quality-items.py --check bench/quality/items/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = "https://datasets-server.huggingface.co/rows"

# task -> (dataset, config, split, primary metric)
TASKS = {
    "boolq": ("aps/super_glue", "boolq", "validation", "acc"),
    "piqa": ("baber/piqa", "default", "validation", "acc_norm"),
    "hellaswag": ("Rowan/hellaswag", "default", "validation", "acc_norm"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", "acc"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", "acc_norm"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", "acc_norm"),
}


def fetch_page(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    request = urllib.request.Request(f"{ENDPOINT}?{query}", headers={"User-Agent": "apeiron-quality-items/1"})
    last = None
    for attempt in range(10):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            last = error
            # The rows API answers 429 when the page rate is too high; it names the wait.
            wait = float(error.headers.get("Retry-After") or 0) or 5.0 * (attempt + 1)
            time.sleep(min(wait, 60.0))
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
            time.sleep(2.0 * (attempt + 1))
    else:
        raise SystemExit(f"rows API failed after 10 attempts: {last}")
    if "rows" not in payload:
        raise SystemExit(f"rows API refused: {payload}")
    return payload


def fetch_all(dataset: str, config: str, split: str, workers: int = 3) -> list[dict]:
    """Every row of the split, in dataset order, in pages of the API's maximum page size.

    Pages are fetched concurrently: the server's per-request latency (2-3 s) dominates, and a
    10k-row split is 101 pages. Order is restored by offset, not by completion.
    """
    first = fetch_page(dataset, config, split, 0, 100)
    total = int(first["num_rows_total"])
    page = len(first["rows"])
    if page == 0:
        raise SystemExit(f"{dataset}/{config}/{split}: empty split")
    offsets = list(range(page, total, page))
    pages: dict[int, list[dict]] = {0: [entry["row"] for entry in first["rows"]]}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_page, dataset, config, split, offset, page): offset for offset in offsets
        }
        for future in futures:
            offset = futures[future]
            pages[offset] = [entry["row"] for entry in future.result()["rows"]]
    rows: list[dict] = []
    for offset in [0, *offsets]:
        rows.extend(pages[offset])
    if len(rows) != total:
        raise SystemExit(f"{dataset}/{config}/{split}: {len(rows)} rows, expected {total}")
    return rows


def select(total: int, limit: int) -> list[int]:
    """Centred systematic sample: `limit` indices spread over `total`."""
    if limit >= total:
        return list(range(total))
    return [(2 * k + 1) * total // (2 * limit) for k in range(limit)]


def preprocess_hellaswag(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def item_boolq(row: dict) -> tuple[int, list[tuple[str, str, int]]]:
    prompt = f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:"
    choices = ["no", "yes"]
    gold = int(bool(row["label"]))
    return gold, [(prompt, " " + c, len(c)) for c in choices]


def item_piqa(row: dict) -> tuple[int, list[tuple[str, str, int]]]:
    prompt = f"Question: {row['goal']}\nAnswer:"
    choices = [row["sol1"], row["sol2"]]
    return int(row["label"]), [(prompt, " " + c, len(c)) for c in choices]


def item_hellaswag(row: dict) -> tuple[int, list[tuple[str, str, int]]]:
    ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    query = preprocess_hellaswag(row["activity_label"] + ": " + ctx)
    choices = [preprocess_hellaswag(ending) for ending in row["endings"]]
    return int(row["label"]), [(query, " " + c, len(c)) for c in choices]


def item_winogrande(row: dict) -> tuple[int, list[tuple[str, str, int]]]:
    sentence = row["sentence"]
    blank = sentence.index("_")
    prefix, suffix = sentence[:blank], sentence[blank + 1 :]
    continuation = " " + suffix.strip()
    gold = {"1": 0, "2": 1}[str(row["answer"])]
    contexts = [prefix + row["option1"], prefix + row["option2"]]
    return gold, [(context, continuation, len(context)) for context in contexts]


def item_arc(row: dict) -> tuple[int, list[tuple[str, str, int]]]:
    prompt = f"Question: {row['question']}\nAnswer:"
    labels = row["choices"]["label"]
    gold = labels.index(row["answerKey"])
    return gold, [(prompt, " " + text, len(text)) for text in row["choices"]["text"]]


BUILDERS = {
    "boolq": item_boolq,
    "piqa": item_piqa,
    "hellaswag": item_hellaswag,
    "winogrande": item_winogrande,
    "arc_easy": item_arc,
    "arc_challenge": item_arc,
}


def build(task: str, limit: int | None, out: Path) -> dict:
    dataset, config, split, metric = TASKS[task]
    rows = fetch_all(dataset, config, split)
    indices = select(len(rows), limit) if limit else list(range(len(rows)))
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = set()
    with out.open("w", encoding="utf-8") as handle:
        for index in indices:
            gold, candidates = BUILDERS[task](rows[index])
            counts.add(len(candidates))
            if not 0 <= gold < len(candidates):
                raise SystemExit(f"{task} item {index}: gold {gold} outside {len(candidates)} choices")
            handle.write(
                json.dumps(
                    {
                        "task": task,
                        "item": index,
                        "gold": gold,
                        "candidates": [
                            {"prompt": prompt, "continuation": continuation, "norm_len": norm_len}
                            for prompt, continuation, norm_len in candidates
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if len(counts) != 1:
        raise SystemExit(f"{task}: mixed choice counts {sorted(counts)}")
    return {
        "task": task,
        "metric": metric,
        "dataset": f"{dataset}/{config}/{split}",
        "total": len(rows),
        "items": len(indices),
        "choices": counts.pop(),
        "out": str(out),
    }


def check(paths: list[Path]) -> int:
    bad = 0
    for path in paths:
        items, keys, tasks, choices = 0, set(), set(), set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            item = json.loads(line)
            items += 1
            keys.add(item["item"])
            tasks.add(item["task"])
            choices.add(len(item["candidates"]))
            if not 0 <= item["gold"] < len(item["candidates"]):
                print(f"{path}: item {item['item']}: bad gold")
                bad += 1
            for candidate in item["candidates"]:
                if not candidate["prompt"] or not candidate["continuation"].strip():
                    print(f"{path}: item {item['item']}: empty prompt or continuation")
                    bad += 1
                if candidate["norm_len"] <= 0:
                    print(f"{path}: item {item['item']}: norm_len {candidate['norm_len']}")
                    bad += 1
        if len(keys) != items:
            print(f"{path}: {items} items, {len(keys)} distinct keys")
            bad += 1
        print(f"{path}: {items} items, task {sorted(tasks)}, {sorted(choices)} choices, {path.stat().st_size} B")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--limit", type=int, default=200, help="systematic sample size (default 200)")
    parser.add_argument("--all", action="store_true", help="write the full split instead of a sample")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", nargs="+", type=Path, metavar="FILE")
    args = parser.parse_args(argv)

    if args.check:
        return check(args.check)
    if not args.task or not args.out:
        parser.error("--task and --out are required unless --check is given")
    summary = build(args.task, None if args.all else args.limit, args.out)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
