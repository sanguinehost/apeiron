#!/usr/bin/env python3
"""Shared profile-level validation helpers for the joint checkers.

This module holds the record, corpus-identity, Markdown/JSON agreement and
control-harness machinery that the four joint checkers share. The
profile-specific row, blocker and control validators stay in their owning
checker; this module is parameterized by the per-profile tables those callers
supply (section names, field labels, mandated controls).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from jointlib import (
    CORPUS_ROOT,
    HEX40,
    HEX64,
    PINNED_CORPUS_REPO,
    REPO,
    SONG_ROOT,
    CheckError,
    _git_has_commit,
    _git_rev,
    _markdown_row_status,
    _mathlib_rev,
    _pinned_corpus_rev,
    _need,
    _nonempty,
    _norm,
    _repo_blob,
    _resolve_path,
    _row_section,
    _sha256,
)


def bind_source_bytes(
    path: Path,
    rel: str,
    digest: str,
    revs: dict,
    *,
    corpus_tree: bool = True,
    prefix: str = "corpus.sources",
) -> None:
    """Bind a recorded digest to the blob at the owning repository revision.

    ``revs`` maps ``apeiron``/``song``/``corpus`` to the recorded revision for
    that repository. ``corpus_tree`` selects whether sources under the canonical
    corpus tree are bound as well.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(PINNED_CORPUS_REPO):
        repo = PINNED_CORPUS_REPO
        rev = revs.get("corpus")
        sub = resolved.relative_to(PINNED_CORPUS_REPO)
    elif resolved.is_relative_to(REPO):
        repo, rev, sub = REPO, revs.get("apeiron"), resolved.relative_to(REPO)
    elif resolved.is_relative_to(SONG_ROOT):
        repo, rev, sub = SONG_ROOT, revs.get("song"), resolved.relative_to(SONG_ROOT)
    elif corpus_tree and resolved.is_relative_to(REPO.parent / "sanguine-apeiron"):
        repo = REPO.parent / "sanguine-apeiron"
        rev = revs.get("corpus")
        sub = resolved.relative_to(repo)
    else:
        return
    blob = _repo_blob(repo, str(rev), str(sub))
    _need(
        blob is not None,
        f"{prefix}[{rel}]: cannot read {sub} at {rev}",
    )
    _need(
        hashlib.sha256(blob).hexdigest() == digest,
        f"{prefix}[{rel}]: recorded digest does not match the blob at {rev}",
    )


def verify_corpus(corpus: object) -> None:
    """Require the resolved corpus checkout to match every recorded identity."""
    _need(isinstance(corpus, dict), "corpus metadata missing")
    _nonempty(corpus.get("root"), "corpus.root")
    _need(CORPUS_ROOT.is_dir(), f"pinned corpus root does not resolve: {CORPUS_ROOT}")

    git = corpus.get("git")
    _need(
        isinstance(git, str) and HEX40.fullmatch(git),
        "corpus.git must be a 40-hex revision",
    )
    pin = _pinned_corpus_rev()
    _need(
        pin is not None,
        "cannot resolve the corpus dependency revision from proof/corpus/lake-manifest.json",
    )
    _need(
        git == pin,
        f"corpus.git {git} does not match the pinned corpus dependency {pin}",
    )
    _need(
        _git_has_commit(CORPUS_ROOT, git),
        f"corpus.git {git} does not resolve in {CORPUS_ROOT}",
    )

    mathlib = corpus.get("mathlib")
    _need(
        isinstance(mathlib, str) and HEX40.fullmatch(mathlib),
        "corpus.mathlib must be a 40-hex revision",
    )
    resolved_mathlib = _mathlib_rev()
    _need(
        resolved_mathlib is not None,
        "cannot resolve the pinned Mathlib revision from a manifest",
    )
    _need(
        mathlib == resolved_mathlib,
        f"corpus.mathlib {mathlib} does not match the resolved Mathlib revision "
        f"{resolved_mathlib}",
    )

    apeiron = corpus.get("apeiron_git")
    _need(
        isinstance(apeiron, str) and HEX40.fullmatch(apeiron),
        "corpus.apeiron_git must be a 40-hex revision",
    )
    _need(
        _git_has_commit(REPO, apeiron),
        f"corpus.apeiron_git {apeiron} does not resolve in {REPO}",
    )

    song = corpus.get("song_git")
    _need(
        isinstance(song, str) and HEX40.fullmatch(song),
        "corpus.song_git must be a 40-hex revision",
    )
    _need(
        _git_has_commit(SONG_ROOT, song),
        f"corpus.song_git {song} does not resolve in {SONG_ROOT}",
    )

    sources = corpus.get("sources")
    _need(
        isinstance(sources, dict) and sources,
        "corpus.sources must be a nonempty digest map",
    )
    for rel, digest in sources.items():
        _nonempty(rel, "corpus.sources path")
        _need(
            isinstance(digest, str) and HEX64.fullmatch(digest),
            f"corpus.sources[{rel}] must be a sha256 hex digest",
        )
        path = _resolve_path(rel, f"corpus.sources[{rel}]")
        # Bind each recorded digest to the blob at the owning repository revision,
        # not to the mutable working tree: sibling streams advance their HEAD
        # independently of these recorded identities.
        bind_source_bytes(
            path,
            rel,
            digest,
            {
                "apeiron": corpus.get("apeiron_git"),
                "song": corpus.get("song_git"),
                "corpus": corpus.get("git"),
            },
        )


def validate_spec_numbered(
    text: object,
    row_ids: list[str],
    row_titles: list[str],
    row_statuses: dict[str, str],
    blocker_ids: list[str],
    hazard_ids: list[str],
    control_ids: list[str],
    field_keys: list[str],
    field_labels: dict[str, tuple[str, str]],
    required_sections: list[str],
) -> None:
    """Require the Markdown spec to carry the numbered records it claims."""
    _need(isinstance(text, str) and text.strip(), "spec document is empty")
    headings = [m.group(1).strip() for m in re.finditer(r"(?m)^#{1,6}\s+(.*)$", text)]
    for section in required_sections:
        _need(
            any(section.lower() in heading.lower() for heading in headings),
            f"spec is missing section heading: {section}",
        )
    for key in field_keys:
        _need(
            re.search(
                r"(?m)^\s*\|\s*`?" + re.escape(key) + r"`?\s*\|", text
            )
            is not None,
            f"spec is missing the required-field record {key}",
        )
    normalized = _norm(text)
    for rid, title in zip(row_ids, row_titles):
        _need(
            re.search(r"(?m)^#{1,6}\s+" + re.escape(rid) + r"\b", text) is not None,
            f"spec is missing the row heading {rid}",
        )
        _need(
            _norm(title) in normalized,
            f"spec does not carry the row record for {rid}",
        )
        section = _row_section(text, rid)
        _need(section.strip() != "", f"spec is missing a section for row {rid}")
        for key in field_keys:
            number, label = field_labels[key]
            pattern = (
                r"\*\*" + re.escape(number) + r"\s*·\s*" + re.escape(label) + r"\.\*\*"
            )
            match = re.search(pattern, section)
            _need(
                match is not None,
                f"spec row {rid} is missing its Markdown field {number} ({label})",
            )
            body_start = match.end()
            following = re.search(r"\*\*\d+\s*·", section[body_start:])
            body_end = body_start + following.start() if following else len(section)
            _need(
                section[body_start:body_end].strip() != "",
                f"spec row {rid} field {number} ({label}) has an empty body",
            )
        markdown_status = _markdown_row_status(text, rid)
        _need(
            markdown_status == row_statuses.get(rid),
            f"spec row {rid} Markdown status {markdown_status!r} disagrees with "
            f"the JSON status {row_statuses.get(rid)!r}",
        )
    for bid in blocker_ids:
        _need(
            re.search(r"(?m)^\s*\|\s*" + re.escape(bid) + r"\s*\|", text) is not None,
            f"spec is missing the blocker record {bid}",
        )
    for hid in hazard_ids:
        _need(
            re.search(r"(?m)^\s*\|\s*" + re.escape(hid) + r"\s*\|", text) is not None,
            f"spec is missing the hazard record {hid}",
        )
    for cid in control_ids:
        _need(
            re.search(r"(?m)^\s*\|\s*" + re.escape(cid) + r"\s*\|", text) is not None,
            f"spec is missing the control record {cid}",
        )


def validate_spec_labeled(
    text: object,
    row_ids: list[str],
    row_titles: list[str],
    row_statuses: dict[str, str],
    hazard_ids: list[str],
    control_ids: list[str],
    field_keys: list[str],
    row_field_labels: dict[str, str],
    required_sections: list[str],
) -> None:
    """Require the Markdown spec to carry the labelled records it claims."""
    _need(isinstance(text, str) and text.strip(), "spec document is empty")
    headings = [m.group(1).strip() for m in re.finditer(r"(?m)^#{1,6}\s+(.*)$", text)]
    for section in required_sections:
        _need(
            any(section.lower() in heading.lower() for heading in headings),
            f"spec is missing section heading: {section}",
        )
    # Required-field records, not incidental mentions.
    for key in field_keys:
        _need(
            re.search(
                r"(?m)^\s*\|\s*`?" + re.escape(key) + r"`?\s*\|", text
            )
            is not None,
            f"spec is missing the required-field record {key}",
        )
    normalized = _norm(text)
    for rid, title in zip(row_ids, row_titles):
        _need(
            re.search(r"(?m)^#{1,6}\s+" + re.escape(rid) + r"\b", text) is not None,
            f"spec is missing the row heading {rid}",
        )
        _need(
            _norm(title) in normalized,
            f"spec does not carry the row record for {rid}",
        )
        section = _row_section(text, rid)
        _need(section.strip() != "", f"spec is missing a section for row {rid}")
        for key in field_keys:
            label = row_field_labels[key]
            marker = "**" + label + ".**"
            match = re.search(re.escape(marker), section)
            _need(
                match is not None,
                f"spec row {rid} is missing its Markdown field {label}",
            )
            body_start = match.end()
            following = re.search(r"\*\*[A-Z][A-Za-z ]*\.\*\*", section[body_start:])
            body_end = body_start + following.start() if following else len(section)
            _need(
                section[body_start:body_end].strip() != "",
                f"spec row {rid} field {label} has an empty body",
            )
        markdown_status = _markdown_row_status(text, rid)
        _need(
            markdown_status == row_statuses.get(rid),
            f"spec row {rid} Markdown status {markdown_status!r} disagrees with "
            f"the JSON status {row_statuses.get(rid)!r}",
        )
    for hid in hazard_ids:
        _need(
            re.search(r"(?m)^\s*\|\s*" + re.escape(hid) + r"\s*\|", text) is not None,
            f"spec is missing the hazard record {hid}",
        )
    for cid in control_ids:
        _need(
            re.search(r"(?m)^\s*\|\s*" + re.escape(cid) + r"\s*\|", text) is not None,
            f"spec is missing the control record {cid}",
        )


def strip_record(spec_text: str, record_id: str) -> str:
    return re.sub(
        r"(?m)^\s*\|\s*" + re.escape(record_id) + r"\s*\|.*\n?",
        "",
        spec_text,
    )


def strip_field_row(spec_text: str, key: str) -> str:
    return re.sub(
        r"(?m)^\s*\|\s*`?" + re.escape(key) + r"`?\s*\|.*\n?",
        "",
        spec_text,
    )


def run_control_cases(
    validate, data: dict, spec_text: str, control_cases
) -> list[tuple[str, bool, str]]:
    """Baseline must pass; every in-memory mutation must be rejected."""
    validate(data, spec_text)
    results: list[tuple[str, bool, str]] = []
    for name, mutated, mutated_spec in control_cases(data, spec_text):
        try:
            validate(mutated, mutated_spec)
        except CheckError as exc:
            results.append((name, True, str(exc)))
        else:
            results.append((name, False, "mutation was NOT rejected"))
    return results


def load_profile(rows_path: Path, spec_path: Path, *, noun: str = "profile") -> tuple[dict, str]:
    _need(rows_path.is_file(), f"profile rows missing: {rows_path}")
    _need(spec_path.is_file(), f"{noun} spec missing: {spec_path}")
    data = json.loads(rows_path.read_text(encoding="utf-8"))
    return data, spec_path.read_text(encoding="utf-8")


def cmd_check(data: dict, spec_text: str, validate, summary) -> int:
    try:
        validate(data, spec_text)
    except CheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS (specification consistency only): {summary(data)}")
    return 0


def cmd_controls(
    data: dict,
    spec_text: str,
    validate,
    run_controls,
    *,
    self_test: bool,
    noun: str,
    mandated_controls: tuple[str, ...],
    class_markers: tuple[str, ...],
) -> int:
    try:
        validate(data, spec_text)
    except CheckError as exc:
        print(f"FAIL: baseline {noun} is inconsistent: {exc}", file=sys.stderr)
        return 1
    results = run_controls(data, spec_text)
    rejected = 0
    names: set[str] = set()
    for name, ok, detail in results:
        names.add(name)
        if ok:
            rejected += 1
            print(f"control RED (rejected): {name}")
        else:
            print(f"control GREEN (NOT rejected): {name}: {detail}", file=sys.stderr)
    if self_test:
        if mandated_controls:
            missing = [name for name in mandated_controls if name not in names]
            _need(
                not missing,
                f"self-test is missing mandated controls: {missing}",
            )
        for marker in class_markers:
            _need(
                any(name.startswith(marker) for name, _, _ in results),
                f"self-test requires a rejecting control for {marker.rstrip(':')}",
            )
        for name, ok, _ in results:
            if name in mandated_controls:
                _need(ok, f"self-test mandated control was not rejected: {name}")
    if rejected != len(results):
        print(
            f"FAIL: {len(results) - rejected} of {len(results)} controls were not rejected",
            file=sys.stderr,
        )
        return 1
    label = "self-test" if self_test else "controls"
    print(
        f"PASS ({label}): baseline valid, {rejected}/{len(results)} mutations rejected; "
        "specification consistency only"
    )
    return 0


def run_cli(
    description: str,
    load,
    validate,
    run_controls,
    summary,
    *,
    noun: str = "profile",
    mandated_controls: tuple[str, ...] = (),
    class_markers: tuple[str, ...] = (),
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate the written spec")
    group.add_argument(
        "--controls", action="store_true", help="validate and require all mutations rejected"
    )
    group.add_argument(
        "--self-test", action="store_true", help="controls plus the mandated mutation assertions"
    )
    args = parser.parse_args(argv)
    try:
        data, spec_text = load()
    except (CheckError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return cmd_check(data, spec_text, validate, summary)
    return cmd_controls(
        data,
        spec_text,
        validate,
        run_controls,
        self_test=args.self_test,
        noun=noun,
        mandated_controls=mandated_controls,
        class_markers=class_markers,
    )
