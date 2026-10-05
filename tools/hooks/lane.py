#!/usr/bin/env python3
"""The per-commit board-declaration requirement (AGENTS.md §9.3).

A dispatch names its row. Every commit carries the declaration, either in the
message or in the staged change:

    board: <row-id>
    board: none -- housekeeping
    board: none -- spike APEIRON-<EPIC>-SPIKE-<slug>

The field is parsed by ``tools/board/board.py`` itself -- ``board_declaration``,
``declared_row`` and ``slug`` -- so the hook enforces the driver's rule rather
than a second spelling of it. A named row must exist on the board and sit in a
milestone of a project: a row outside a milestone cannot be selected, ordered or
driven, which is the ``unattributed-task`` failure class the driver already
reports.

The declaration is load-bearing: it is matched against the *staged change*, so a
no-row declaration only works where there is no code to attribute. A staged path
under ``CODE_PREFIXES``/``CODE_FILES`` (below) requires ``board: <row-id>``
naming a row that resolves, or a well-formed spike id; ``none -- housekeeping``
covers a change confined to documentation and tooling, and the old phrase
``none -- internal lane`` is retired outright -- refused by name on every
commit, because it named no marker, bounded nothing, and was accepted on 12
consecutive commits of which 8 touched code (``measured``, ``git log -14`` at
``e6f2c90``).

A hook cannot call MCP and no REST move-task route exists, so the hook resolves
and *reports* the row; moving it stays the lead's action at the board.

Two tiers, because git will not give a ``pre-commit`` hook the message: at that
point ``COMMIT_EDITMSG`` still holds the *previous* commit's message and the new
one is written after the hook returns. ``pre-commit`` (``lane.py staged``)
therefore checks what the staged change carries and never demands a field it
cannot see; ``commit-msg`` (``lane.py message PATH``), which git hands the real
message file, is where presence is enforced.

The full reconciliation (``board.py check``) is a wave-boundary gate and is not
run here: it compares the whole board against ``docs/plan.md``, which is red by
design between wave boundaries. The *declaration* is per-commit; the reconcile is
per-wave.

Board source, in precedence order:

1. the live board over REST, when ``OPERATELY_API_KEY`` is in the environment.
   The live board is **enumerated**, never sampled: every route that lists rows is
   read to exhaustion, two *consecutive* enumerations must agree on the row count
   and the row id set, and the routes must agree with each other. Because lanes
   file rows while other lanes commit, the fetch re-enumerates until two
   consecutive reads agree, bounded by ``APEIRON_HOOKS_COMMITMSG_BUDGET``
   (default 90 s); if the budget expires while the board keeps moving it refuses
   with ``board-fetch-count-unstable`` and the observed counts -- never by taking
   the later read. An under-read is refused, not resolved against. A missing key
   is a named refusal.
2. ``APEIRON_HOOKS_BOARD_SOURCE`` -- a *stamped* capture (see ``lane.py
   capture``). This is the deterministic path for a board that will not quiesce
   inside the budget. Accepted only as a capture, never as a bare document: it
   must carry ``fetch.fetched_at`` and ``fetch.task_count``, it must be no older
   than the staged change, and when the key is present its row count must equal a
   fresh enumeration. A snapshot that under-read the board is therefore refused
   rather than authoritative.
3. nothing: a commit that names a row cannot be checked, so it is refused by
   ``board-unavailable`` and the fix is printed. ``housekeeping`` and a
   well-formed spike need no board: neither names a row to move, and the
   no-row decision is taken from the staged paths alone, before any board read,
   so a board outage can neither grant nor withhold it.

A ``board:`` declaration names a row by its **exact** id, or by a plan row id
that exactly one row's own description declares. Name-slug and row-id-prefix
matching are gone: a fuzzy rule is how a declaration for one row resolved to a
different, older row. The named row must not already be ``Done`` or
``Canceled`` (``board-row-closed``), and its id must appear in the committed
change (``board-row-not-in-commit``), so the association is readable from the
commit and not only from the message.

The commit's association with its row is the ``Commit`` object and the
``Commit->Task`` link of
``docs/contracts/commit-task-association.md``. The ``board:`` message field is
one projection of that link; ``lane.py link <rev>`` reads the link back out of a
committed object.

Modes: ``staged`` (the pre-commit half), ``message PATH`` (the commit-msg half),
``capture`` (write a stamped snapshot), ``rows`` (diagnostic), ``link [REV]``
(read a commit back into the object), ``self-test`` (the controls).
"""

from __future__ import annotations

import argparse
import sys

from lane_core import (  # noqa: F401 - re-exported: the entry's public surface
    BOARD_DIR,
    BOARD_IMPORT_ERROR,
    CODE_FILES,
    CODE_PREFIXES,
    COMMIT_LANES,
    COMMITMSG_BUDGET_ENV,
    COMMITMSG_BUDGET_DEFAULT,
    DECLARATION_HINT,
    DOC_FILES,
    DOC_PREFIXES,
    FetchError,
    Finding,
    HOUSEKEEPING,
    LINK_TYPE,
    REPO_ROOT,
    RULE_AMBIGUOUS,
    RULE_CONFLICT,
    RULE_DRIVER_MISSING,
    RULE_FETCH_DISAGREE,
    RULE_FETCH_KEY,
    RULE_FETCH_PAGING,
    RULE_FETCH_TRUNCATED,
    RULE_FETCH_UNAVAILABLE,
    RULE_FETCH_UNSTABLE,
    RULE_GIT,
    RULE_MISSING,
    RULE_NOT_FOUND,
    RULE_NO_MILESTONE,
    RULE_NO_ROW_ON_CODE,
    RULE_NO_ROW_RETIRED,
    RULE_ROW_CLOSED,
    RULE_ROW_NOT_IN_COMMIT,
    RULE_SNAPSHOT_STALE,
    RULE_SNAPSHOT_UNSTAMPED,
    RULE_SPIKE_ID,
    RULE_UNAVAILABLE,
    SPIKE_DECLARATION,
    SPIKE_ID,
    __init__,
    __str__,
    _finding,
    _source_namespace,
    board,
    check_declaration,
    check_no_row,
    check_row,
    classify_no_row,
    code_paths,
    commit_identity,
    commit_links,
    find_declaration,
    git,
    git_dir,
    index_digest,
    is_code_path,
    lane_of,
    load_board_documents,
    match_rows,
    milestone_of,
    now_iso,
    parse_epoch,
    project_commit,
    row_carried_in_change,
    snapshot_stamp,
    staged_added_lines,
    staged_change_epoch,
    staged_paths,
)
from lane_fetch import (  # noqa: F401 - re-exported: the entry's public surface
    CURSOR_KEYS,
    CURSOR_PARAM_FOR_KEY,
    MAX_PAGES,
    NO_PAGING,
    PAGE_SIZE_PARAMS,
    REQUESTED_PAGE,
    TOTAL_KEYS,
    _PAGE_INPUT,
    _continuation,
    _declared_total,
    _page_input,
    client_from_environment,
    enumerated_board,
    exhaustive_document,
    paginate,
)
from lane_modes import (  # noqa: F401 - re-exported: the entry's public surface
    capture_document,
    capture_freshness,
    check_commit,
    load_for_check,
    mode_capture,
    mode_link,
    mode_message,
    mode_rows,
    mode_staged,
)
from lane_controls import (  # noqa: F401 - re-exported: the entry's public surface
    CODE_STAGED,
    DOCS_STAGED,
    MANIFEST_STAGED,
    MIXED_STAGED,
    StubClient,
    _doc,
    contract_parity_case,
    control_case,
    control_document,
    fetch_case,
    link_round_trip_case,
    run_controls,
    stub_task,
    table_columns_after,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "mode", choices=("staged", "message", "capture", "rows", "link", "self-test")
    )
    parser.add_argument("path", nargs="?", default="", help="the message file, as commit-msg passes it")
    parser.add_argument("--out", help="snapshot path for capture")
    parser.add_argument("--board-source", default="", help="schema-1 board document to read")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "self-test":
        return run_controls()
    if args.mode == "capture":
        if not args.out:
            print("lane-capture: --out PATH is required", file=sys.stderr)
            return 2
        return mode_capture(args)
    if args.mode == "rows":
        return mode_rows(args)
    if args.mode == "link":
        return mode_link(args)
    if args.mode == "message":
        return mode_message(args)
    return mode_staged(args)


if __name__ == "__main__":
    sys.exit(main())
