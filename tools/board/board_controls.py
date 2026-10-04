#!/usr/bin/env python3
"""board_controls: the seeded-defect suite behind ``--controls``.

Baseline green, every seeded defect red by name with the
subject it names, restored green, and the check-in path confirmed by a
read-back. Part of ``tools/board/board.py``.
"""

from __future__ import annotations

import copy
import sys
from board_core import (  # noqa: F401
    rules_of,
)
from board_check import (  # noqa: F401
    SEEDED_DEFECTS,
    _fixture_check,
    exit_code_for,
    post_offline,
)
from board_fixtures import (  # noqa: F401
    baseline_document,
)

# ---------------------------------------------------------------------------
# Control suite
# ---------------------------------------------------------------------------


def run_controls(stream=None) -> int:
    """Baseline green, every seeded defect red by name, restored green."""
    stream = stream or sys.stdout
    failures = 0
    baseline = baseline_document()
    findings = _fixture_check(baseline)
    if findings:
        failures += 1
        print(
            f"CONTROL FAIL baseline: expected green, got {sorted(rules_of(findings))}", file=stream
        )
    else:
        print("CONTROL pass baseline: 0 findings", file=stream)

    for rule, mutate, subject in SEEDED_DEFECTS:
        doc = copy.deepcopy(baseline)
        mutate(doc)
        findings = _fixture_check(doc)
        rules = rules_of(findings)
        code = exit_code_for(findings)
        if rule not in rules:
            failures += 1
            print(
                f"CONTROL FAIL {rule}: seeded defect did not redden; got "
                f"{sorted(rules) or 'green'}",
                file=stream,
            )
            continue
        if code == 0:
            failures += 1
            print(f"CONTROL FAIL {rule}: findings reported but exit status was 0", file=stream)
            continue
        offending = [f for f in findings if f["rule"] == rule and subject not in f["detail"]]
        if offending:
            failures += 1
            print(
                f"CONTROL FAIL {rule}: finding does not name {subject!r}: "
                f"{offending[0]['detail']}",
                file=stream,
            )
            continue
        print(f"CONTROL pass {rule}: exit {code} on {subject}", file=stream)

    restored = baseline_document()
    if _fixture_check(restored):
        failures += 1
        print(f"CONTROL FAIL restore: expected green, got {sorted(rules_of(_fixture_check(restored)))}", file=stream)
    else:
        print("CONTROL pass restore: 0 findings", file=stream)

    # A reconciliation record must reach the board and be confirmed by a read-back.
    landed = post_offline(baseline, "apeiron", "green")
    if not landed:
        failures += 1
        print("CONTROL FAIL post-read-back: the record did not land on read-back", file=stream)
    else:
        print("CONTROL pass post-read-back: record confirmed by read-back", file=stream)
    refused = post_offline(baseline, "apeiron", "green", transport_applies=False)
    if refused:
        failures += 1
        print("CONTROL FAIL post-refusal: a write that did not land was accepted", file=stream)
    else:
        print("CONTROL pass post-refusal: unlanded write reported as not landed", file=stream)

    total = 1 + len(SEEDED_DEFECTS) + 1 + 2
    if failures:
        print(f"CONTROL FAIL suite: {failures} of {total} controls failed", file=stream)
        return 2
    print(f"CONTROL pass suite: {total}/{total} controls", file=stream)
    return 0
