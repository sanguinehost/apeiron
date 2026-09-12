#!/usr/bin/env python3
"""Reconcile the published census ledgers with schema-2 scanner evidence.

This validates records and arithmetic, not Rust name resolution. Kind corrections
are explicit inventory joins; they do not promote an attribution to a proof.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


DATA = Path(__file__).resolve().parents[2] / "docs" / "call-census" / "data"
COHORTS = ("models", "bedrock", "other", "hosted")
FIELDS = (
    "cohort", "run", "revision", "id", "file", "line", "column", "caller",
    "callee", "target_crate", "source_kind", "inventory_kind", "kind_basis",
    "api_family", "review_status", "source_location", "is_test", "cfg",
)


def read_json(path):
    return json.loads(path.read_text())


def read_tsv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def family(record):
    symbol = record["target_symbol"]
    crate = record["target_crate"]
    root = symbol.split("::")[0]
    if crate == "hashtrinity_crypto" or root in ("crypto", "Sha256"):
        return "encoding_in_crypto" if symbol.split("::")[-1] == "to_hex" else "crypto_housed"
    if root in ("Table", "CoordMap", "Dense", "FreqMap", "Coord", "CoordSeed"):
        return "collections_and_addressing"
    if root in ("codec", "Term", "SyntaxTerm", "Atom", "AsTerm"):
        return "encoding_and_terms"
    if root in ("openapi", "validate", "Error", "Result", "error"):
        return "api_substitutes"
    if root in ("par", "runtime", "reactor", "sync", "S3"):
        return "execution_and_io"
    if root in ("hex", "base64", "base32"):
        return "encoding_helpers"
    if root in ("rustls", "webauthn", "secret", "SecretString"):
        return "security_integration"
    return "other"


def inventory_kind(record, inventory):
    source = record["operation_kind"]
    if record["record_kind"] == "method":
        return "method", "method-call syntax; owner remains a syntactic attribution"
    parts = record["target_symbol"].split("::")
    if len(parts) >= 2 and parts[-1] in inventory.get("variants", {}).get(parts[-2], []):
        return "constructor", "enum/variant suffix matched in target inventory"
    if parts[-1] in inventory.get("tuple_structs", []):
        return "constructor", "tuple-struct suffix matched in target inventory"
    if len(parts) >= 2 and parts[-2] in inventory.get("types", []):
        return "associated_fn", "owner suffix matched in target type inventory"
    return source, "scanner classification retained"


def run_paths(raw, cohort):
    base = raw / cohort
    if cohort in ("models", "bedrock"):
        return [(cohort, base)]
    return [(path.name, path) for path in sorted(base.iterdir()) if (path / "summary.json").exists()]


def published_key(cohort, row):
    run = row["repo"] if cohort == "hosted" else row["crate"] if cohort == "other" else cohort
    return run, row["id"]


def generate(raw):
    manifest = []
    combined = []
    totals = {}
    fingerprints = set()
    flagged = read_tsv(DATA / "other-manual-confirmed.tsv")
    false_positives = {
        (row["crate"], row["file"], int(row["line"]))
        for row in flagged if row["status"] == "false_positive"
    }
    matched_flags = set()

    for cohort in COHORTS:
        published = read_tsv(DATA / f"{cohort}-calls.tsv")
        index = {published_key(cohort, row): row for row in published}
        assert len(index) == len(published), f"duplicate published ids: {cohort}"
        seen = set()
        cohort_totals = Counter()
        for name, path in run_paths(raw, cohort):
            meta = read_json(path / "meta.json")
            summary = read_json(path / "summary.json")
            inventory = read_json(path / "inventory.json")
            records = read_json(path / "records.json")
            assert meta["schema_version"] == summary["schema_version"] == 2
            assert meta["tool_version"] == "0.2.0"
            assert meta["revision"]["git_status_porcelain_lines"] == 0, path
            fingerprints.add((meta["tool_source_sha256"], meta["tool_binary_sha256"]))
            assert summary["files_parse_error"] == 0, path
            assert len({row["id"] for row in records}) == len(records), path
            attributed = [row for row in records if row["confidence"] in ("high", "medium")]
            assert len(attributed) == summary["attributed_calls"], path
            assert len(records) == summary["records_total"], path
            cohort_totals.update({
                "runs": 1, "files": summary["files_parsed"],
                "attributed": len(attributed), "recorded": len(records),
                "unattributed": len(records) - len(attributed),
            })
            for row in attributed:
                key = (name, row["id"])
                assert key in index and key not in seen, (cohort, key)
                seen.add(key)
                published_row = index[key]
                assert published_row["file"] == row["file"], key
                assert int(published_row["line"]) == row["span"]["line"], key
                assert int(published_row["column"]) == row["span"]["column"], key
                assert published_row["target_symbol"] == row["target_symbol"], key
                flag = (name, row["file"], row["span"]["line"])
                is_false = cohort == "other" and flag in false_positives
                if is_false:
                    matched_flags.add(flag)
                kind, basis = inventory_kind(row, inventory)
                combined.append({
                    "cohort": cohort, "run": name,
                    "revision": meta["revision"]["git_head"], "id": row["id"],
                    "file": row["file"], "line": row["span"]["line"],
                    "column": row["span"]["column"], "caller": row["caller"]["qualified"],
                    "callee": row["target_symbol"], "target_crate": row["target_crate"],
                    "source_kind": row["operation_kind"], "inventory_kind": kind,
                    "kind_basis": basis, "api_family": family(row),
                    "review_status": "known_false_positive" if is_false else "syntactic_attribution_not_proved",
                    "source_location": row["source_location"],
                    "is_test": row["is_test_context"], "cfg": " | ".join(row["cfg"]),
                })
            manifest.append({
                "cohort": cohort, "run": name, "root": meta["root"],
                "revision": meta["revision"], "tool_version": meta["tool_version"],
                "tool_source_sha256": meta["tool_source_sha256"],
                "tool_binary_sha256": meta["tool_binary_sha256"],
                "files": summary["files_parsed"], "attributed": len(attributed),
                "unattributed_records": len(records) - len(attributed),
                "summary_sha256": hashlib.sha256((path / "summary.json").read_bytes()).hexdigest(),
            })
        assert seen == set(index), f"unmatched published rows: {cohort}"
        totals[cohort] = dict(cohort_totals)

    assert matched_flags == false_positives, "a flagged false positive disappeared or changed"
    assert len(fingerprints) == 1, f"mixed tool versions: {fingerprints}"
    combined.sort(key=lambda row: (row["cohort"], row["run"], row["file"], row["line"], row["column"]))
    clean = [row for row in combined if row["review_status"] != "known_false_positive"]
    audit = {
        "schema_version": 1,
        "scope": "13 current Fold subtrees and 21 selected hosted repository snapshots; not all portfolio repos",
        "limits": "Source syntax, not a compiler call graph. Macros and cfg are not expanded/evaluated. No runtime counts. Residual attributions remain unproved after known false positives are marked.",
        "cohorts": totals,
        "raw_attributed_sites": len(combined), "known_false_positives": len(combined) - len(clean),
        "remaining_attributed_sites": len(clean),
        "inventory_kind_after_known_exclusions": dict(sorted(Counter(row["inventory_kind"] for row in clean).items())),
        "api_family_after_known_exclusions": dict(sorted(Counter(row["api_family"] for row in clean).items())),
        "runs": manifest,
    }
    return audit, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="generate consolidated audit JSON and TSV")
    parser.add_argument("--rerun", type=Path, metavar="SCANNER", help="rerun the pinned schema-2 cohorts before reconciliation")
    args = parser.parse_args()
    if args.rerun:
        for cohort in COHORTS:
            for name, path in run_paths(args.raw_root, cohort):
                meta = read_json(path / "meta.json")
                command = [str(args.rerun.resolve()), "--root", meta["root"], "--output", str(path)]
                for target in meta["target_src_roots"]:
                    command.extend(("--target-src", target))
                for crate in meta["target_crates"]:
                    command.extend(("--target-crate", crate))
                if cohort == "hosted" and name == "Ptyktos-autofold-standalone":
                    command.extend(("--skip", "hashtrinity"))
                subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"reran {cohort}/{name}")
    audit, rows = generate(args.raw_root)
    if args.write:
        (DATA / "census-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        with (DATA / "reviewed-calls.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: value for key, value in audit.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
