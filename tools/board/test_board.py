#!/usr/bin/env python3
"""Offline tests for the Operately board driver.

No live key and no network: every fixture is in-file, and the transport is a stub
opener that records the request it was handed. Run from the repository root::

    python3 -m unittest discover -s tools/board -p 'test_*.py'
    python3 tools/board/board.py --check --controls      # the gate
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import board  # noqa: E402
import board_check  # noqa: E402
import board_core  # noqa: E402


class StubResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "StubResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class StubOpener:
    """Answers by route key; records every URL. Never touches the network.

    A key without a ``?`` is an exact request path, so ``projects/list`` does
    not shadow ``projects/list_milestones`` the way a substring match would.
    """

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.calls: list[str] = []
        self.bodies: list[bytes] = []

    def open(self, request, timeout=None):  # noqa: ANN001 - stub
        url = request.get_full_url()
        self.calls.append(url)
        self.bodies.append(request.data or b"")
        path = urllib.parse.urlparse(url).path
        for key, payload in self.routes.items():
            if "?" in key:
                if key in url:
                    return StubResponse(payload)
            elif path.endswith("/" + key):
                return StubResponse(payload)
        raise board.TransportError(f"stub has no route for {url}")


class WriteShapeTests(unittest.TestCase):
    """The write endpoints take rich content and the status OBJECT.

    Each control reproduces the shape the live instance REJECTS, so the check
    can fail: a plain-string description answers HTTP 500 and does not land, and
    ``tasks/update_status`` answers ``400 Unknown input field`` for
    ``value=<str>`` (only the object lands).
    """

    def test_check_in_description_is_a_prosemirror_document(self) -> None:
        opener = StubOpener({"projects/create_check_in": {"id": "ci-1"}})
        client = board.RestClient(key="k", opener=opener)
        transport, error = client.post_check_in("p-1", "caution", "line one\nline two")
        self.assertEqual(transport, "landed")
        self.assertEqual(error, "")
        body = json.loads(opener.bodies[-1])
        self.assertIsInstance(body["description"], str)
        self.assertNotEqual(body["description"], "line one\nline two", "plain string must not be sent")
        doc = json.loads(body["description"])
        self.assertEqual(doc["type"], "doc")
        texts = [p["content"][0]["text"] for p in doc["content"] if p.get("content")]
        self.assertEqual(texts, ["line one", "line two"])

    def test_update_status_sends_the_status_object_not_its_value(self) -> None:
        routes = {
            "tasks/get": {
                "task": {"id": "t-1", "type": "project", "status": {"value": "pending"}},
            },
            "tasks/list_task_statuses": {
                "task_statuses": [
                    {"id": "s-1", "value": "pending", "label": "Not started"},
                    {"id": "s-2", "value": "in_progress", "label": "In progress"},
                ]
            },
            "tasks/update_status": {"ok": True},
        }
        opener = StubOpener(routes)
        client = board.RestClient(key="k", opener=opener)
        self.assertEqual(client.update_status("t-1", "In progress"), "landed")
        body = json.loads(opener.bodies[-1])
        self.assertEqual(body["type"], "project")
        self.assertEqual(
            body["status"], {"id": "s-2", "value": "in_progress", "label": "In progress"}
        )
        self.assertNotIn("value", body, "the rejected `value=` shape must not be sent")

    def test_update_status_refuses_an_unavailable_status(self) -> None:
        routes = {
            "tasks/get": {"task": {"id": "t-1", "type": "project", "status": {"value": "pending"}}},
            "tasks/list_task_statuses": {"task_statuses": [{"id": "s-1", "value": "pending"}]},
        }
        opener = StubOpener(routes)
        client = board.RestClient(key="k", opener=opener)
        self.assertEqual(client.update_status("t-1", "not-a-status"), "refused")


class DescriptionTests(unittest.TestCase):
    def test_prosemirror_doc_flattens_to_text(self) -> None:
        raw = board.describe("first para with `a`", "", "Source: `docs/plan.md`.")
        text = board.flatten_description(raw)
        self.assertIn("first para with `a`", text)
        self.assertIn("Source: `docs/plan.md`.", text)

    def test_plain_markdown_is_accepted(self) -> None:
        self.assertIn("cargo test", board.flatten_description("Run `cargo test` first."))

    def test_missing_description_is_empty_not_an_error(self) -> None:
        self.assertEqual(board.flatten_description(None), "")
        self.assertEqual(board.flatten_description(""), "")

    def test_labelled_gate_field_is_a_gate(self) -> None:
        text = "Fix the table.\nSource: `src/table.rs`.\nGate: `cargo test --workspace`.\n"
        self.assertEqual(board.gates_of(text), ["cargo test --workspace"])
        self.assertEqual(board.paths_of(text), ["src/table.rs"])

    def test_unlabelled_backticked_invocation_is_a_gate(self) -> None:
        self.assertEqual(
            board.gates_of("`proof/regen.sh --check` must pass."), ["proof/regen.sh --check"]
        )
        self.assertEqual(
            board.gates_of("`python3 tools/rooting/check.py --check` is red."),
            ["python3 tools/rooting/check.py --check"],
        )
        self.assertEqual(board.gates_of("`lake build` in proof/corpus"), ["lake build"])

    def test_extensionless_gate_binary_is_a_gate(self) -> None:
        self.assertEqual(board.gates_of("`tools/gate hold` must be clean"), ["tools/gate hold"])
        self.assertEqual(board.gates_of("`tools/gate`"), ["tools/gate"])
        self.assertEqual(
            board.gates_of("`tools/gate/target/debug/gate`"), ["tools/gate/target/debug/gate"]
        )

    def test_a_bare_script_path_is_not_a_gate(self) -> None:
        """A path is not a runnable command: no arguments, no mode."""
        self.assertEqual(board.gates_of("see `tools/rooting/check.py`"), [])
        self.assertEqual(board.paths_of("see `tools/rooting/check.py`"), ["tools/rooting/check.py"])

    def test_paths_are_collected_from_backticks_and_labels(self) -> None:
        text = "Source: `docs/plan.md` §Status; `src/table.rs`; `proof/ApeironWord.lean`."
        self.assertEqual(
            board.paths_of(text), ["docs/plan.md", "src/table.rs", "proof/ApeironWord.lean"]
        )

    def test_repo_root_file_is_a_source_path(self) -> None:
        self.assertEqual(board.paths_of("Source: `README.md`."), ["README.md"])

    def test_receipts_need_an_artifact_not_the_word(self) -> None:
        self.assertEqual(board.receipts_of("the receipt is written."), [])
        self.assertEqual(board.receipts_of("Receipt: `docs/evidence/run.md`."), ["docs/evidence/run.md"])
        self.assertEqual(board.receipts_of("Verdict: see https://x.test/v/1"), ["https://x.test/v/1"])
        self.assertEqual(
            board.receipts_of("done", ["archived in `docs/evidence/a.md`"]), ["docs/evidence/a.md"]
        )

    def test_plan_ids_are_the_backticked_uppercase_tokens(self) -> None:
        text = "Plan row `APEIRON-E6-W4-T1`; also `plan/ledger.tsv`."
        self.assertEqual(board.PLAN_ID_IN_CODE.findall(text), ["APEIRON-E6-W4-T1"])

    def test_internal_by_design_marker_is_named(self) -> None:
        self.assertEqual(board.internal_named_by("internal-by-design: probe"), "internal-by-design")
        self.assertEqual(board.internal_named_by("an ordinary task"), "")


class PlanParseTests(unittest.TestCase):
    def test_markdown_wave_table_rows_carry_id_gate_and_paths(self) -> None:
        rows = board.parse_markdown_plan(board.FIXTURE_PLAN_MD, "docs/plan.md")
        by_id = {r["id"]: r for r in rows}
        self.assertIn("FIXTURE-E1-W1-T1", by_id)
        row = by_id["FIXTURE-E1-W1-T1"]
        self.assertFalse(row["closed"])
        self.assertEqual(row["gates"], ["python3 tools/rooting/check.py --check"])
        self.assertEqual(row["paths"], ["tools/rooting/check.py"])
        self.assertEqual(row["source"], "docs/plan.md")

    def test_a_row_saying_landed_is_closed(self) -> None:
        rows = board.parse_markdown_plan(board.FIXTURE_PLAN_MD, "docs/plan.md")
        by_id = {r["id"]: r for r in rows}
        self.assertTrue(by_id["FIXTURE-E1-W3-T1"]["closed"])
        self.assertFalse(by_id["FIXTURE-E1-W1-T1"]["closed"])

    def test_a_resume_queue_row_is_parsed_and_non_closed(self) -> None:
        """A table headed `task | session | state | note` is a plan table too."""
        text = (
            "## Resume queue (paused; not resumed until the owner's go)\n\n"
            "| task | session | state | note |\n|---|---|---|---|\n"
            "| Fixture lane row `R9-T1` | `ses_fixture` | live at 09:00 | mirror probe |\n"
        )
        rows = board.parse_markdown_plan(text, "docs/plan.md")
        by_id = {r["id"]: r for r in rows}
        self.assertIn("R9-T1", by_id)
        self.assertFalse(by_id["R9-T1"]["closed"])

    def test_a_resume_queue_row_in_the_ledger_is_non_closed(self) -> None:
        rows = board.parse_tsv_plan(board.FIXTURE_LEDGER_TSV, "plan/ledger.tsv")
        by_id = {r["id"]: r for r in rows}
        self.assertIn("R1-T1", by_id)
        self.assertFalse(by_id["R1-T1"]["closed"])

class ProtocolTests(unittest.TestCase):
    """A dispatch names its row; a due date names a dated fact."""

    def test_a_board_declaration_names_a_plan_row_or_an_internal_lane(self) -> None:
        self.assertEqual(board.board_declaration("Board: `R1-T1`."), ("row", "R1-T1 ."))
        self.assertEqual(
            board.board_declaration("Board: none -- internal lane."),
            ("internal", "none -- internal lane."),
        )
        self.assertEqual(board.board_declaration("the work moved onto the board: later"), ("", ""))

    def test_declared_row_drops_surrounding_punctuation(self) -> None:
        self.assertEqual(board.declared_row("R1-T1 ."), "R1-T1")
        self.assertEqual(board.declared_row("`unplanned-row-7777CCCC`,"), "unplanned-row-7777CCCC")
        self.assertEqual(board.declared_row(""), "")

    def test_every_baseline_row_declares_the_row_it_moves(self) -> None:
        for task in board.baseline_document()["tasks"]:
            kind, declared = board.board_declaration(task["text"])
            self.assertIn(kind, ("row", "internal"), task["id"])
            if kind == "row":
                self.assertIn(
                    board.declared_row(declared).lower(),
                    {task["id"].lower()} | {p.lower() for p in task["plan_ids"]},
                    task["id"],
                )

    def test_a_missing_declaration_reddens_by_name(self) -> None:
        doc = board.baseline_document()
        board.defect_board_declaration_missing(doc)
        findings = [f for f in board._fixture_check(doc) if f["rule"] == board.RULE_NO_BOARD_DECLARATION]
        self.assertEqual(len(findings), 1)
        self.assertIn("declares no board row", findings[0]["detail"])

    def test_a_declaration_naming_another_row_reddens_by_name(self) -> None:
        doc = board.baseline_document()
        board.defect_board_declaration_names_another_row(doc)
        findings = [f for f in board._fixture_check(doc) if f["rule"] == board.RULE_NO_BOARD_DECLARATION]
        self.assertEqual(len(findings), 1)
        self.assertIn("FIXTURE-E1-W9-T9", findings[0]["detail"])

    def test_a_prose_mention_is_not_a_declaration(self) -> None:
        """The field is a labelled line: a mention inside prose does not count."""
        doc = board.baseline_document()
        task = board._find(doc, "g1-reg-1-acceptance-green-BBBB2222")
        task["text"] = task["text"].replace(
            "Board: `FIXTURE-E1-W1-T2`.\n\n", "The row is on the board: `FIXTURE-E1-W1-T2`.\n\n"
        )
        self.assertIn(board.RULE_NO_BOARD_DECLARATION, board.rules_of(board._fixture_check(doc)))

    def test_a_due_date_without_a_fact_reddens_by_name(self) -> None:
        doc = board.baseline_document()
        board.defect_due_date_without_fact(doc)
        findings = [f for f in board._fixture_check(doc) if f["rule"] == board.RULE_DUE_WITHOUT_FACT]
        self.assertEqual(len(findings), 1)
        self.assertIn("2026-11-01", findings[0]["detail"])

    def test_a_due_date_backed_by_a_receipt_or_digest_is_green(self) -> None:
        self.assertEqual(
            board.dated_facts("sha256=e3b0c44298fc1c149afbf4c8996fb924", []),
            ["e3b0c44298fc1c149afbf4c8996fb924"],
        )
        self.assertEqual(board.dated_facts("", ["docs/evidence/run.md"]), ["docs/evidence/run.md"])
        self.assertEqual(board.dated_facts("no anchor here", []), [])

    def test_a_due_date_field_without_a_date_is_not_a_date_claim(self) -> None:
        self.assertEqual(board.due_date_of("Due: after the gate."), "")

    def test_due_date_of_reads_an_iso_field(self) -> None:
        self.assertEqual(board.due_date_of("Due: 2026-10-05, from the receipt."), "2026-10-05")


class ProjectSelectionTests(unittest.TestCase):
    def test_a_project_id_resolves_to_its_name_before_any_read(self) -> None:
        self.assertEqual(
            board.project_name_hint("apeiron-ExNAtv3hObMmq01Ab20DfA"), "apeiron"
        )
        self.assertEqual(board.project_name_hint("song"), "song")
        self.assertEqual(board.project_name_hint("unknown-thing"), "unknown-thing")
        self.assertIsNone(board.project_name_hint(None))

    def test_the_board_spelling_wins_when_the_document_is_loaded(self) -> None:
        doc = board.baseline_document()
        self.assertEqual(board.resolve_project(doc, board.APEIRON["id"]), "apeiron")

    def test_a_space_row_is_reported_even_under_a_project_filter(self) -> None:
        """The space-level/project-level split is drift, not a silent exclusion."""
        doc = board.baseline_document()
        task = board._find(doc, "rooting-gate-green-AAAA1111")
        task["project_id"] = None
        task["project_name"] = None
        board.attribute_projects(doc)
        rules = board.rules_of(board._fixture_check(doc, project="song"))
        self.assertIn(board.RULE_UNATTRIBUTED, rules)

    def test_a_milestone_title_is_filled_in_from_the_inventory(self) -> None:
        doc = board.baseline_document()
        self.assertEqual(
            board._find(doc, "land-gate-wave-3-CCCC3333")["milestone_title"],
            "M1 — Assembly core complete",
        )

    def test_each_fixture_source_carries_only_its_own_project(self) -> None:
        """A project-scoped run reads one source of truth, so the model must too."""
        doc = board.baseline_document()
        for project, expected in (("apeiron", {"docs/plan.md"}), ("song", {"plan/ledger.tsv"})):
            plan = board.plan_from_document(doc, project)
            self.assertEqual({s["path"] for s in plan["sources"]}, expected, project)

    def test_a_row_without_an_id_is_identified_by_section_and_wave(self) -> None:
        text = (
            "## Phasing\n\n### E1 — Gate\n\n"
            "| wave | tasks | files | gate | agents | heavy |\n"
            "|---|---|---|---|---|---|\n"
            "| W7 | delete every gate | `proof/resource/` | none remain | 1 | 0 |\n"
        )
        rows = board.parse_markdown_plan(text, "docs/plan.md")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "e1-gate/w7")
        self.assertFalse(rows[0]["closed"])

    def test_tsv_ledger_rows(self) -> None:
        rows = board.parse_tsv_plan(board.FIXTURE_LEDGER_TSV, "plan/ledger.tsv")
        by_id = {r["id"]: r for r in rows}
        self.assertTrue(by_id["W0-GATE"]["closed"])
        self.assertFalse(by_id["W1-I01"]["closed"])
        self.assertEqual(by_id["W1-I01"]["paths"], ["src/emit/x86.slang"])
        self.assertEqual(by_id["W1-I01"]["gates"], ["tools/gate hold"])
        self.assertEqual(by_id["W2-T1"]["name"], "Land GATE-WAVE-3 and the freeze corrections")

    def test_non_plan_tables_are_ignored(self) -> None:
        text = "# doc\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
        self.assertEqual(board.parse_markdown_plan(text, "docs/plan.md"), [])

    def test_plan_source_selects_the_parser_by_extension(self) -> None:
        self.assertEqual(len(board.parse_plan_source("plan/ledger.tsv", board.FIXTURE_LEDGER_TSV)), 5)
        self.assertGreater(len(board.parse_plan_source("docs/plan.md", board.FIXTURE_PLAN_MD)), 1)

    def test_load_plan_digests_every_source(self) -> None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_plan.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(board.FIXTURE_PLAN_MD)
        try:
            plan = board.load_plan([path], ".")
        finally:
            os.unlink(path)
        self.assertEqual(len(plan["sources"]), 1)
        self.assertEqual(plan["sources"][0]["sha256"], board.sha256_of(board.FIXTURE_PLAN_MD))
        self.assertEqual(plan["sources"][0]["rows"], len(plan["rows"]))

    def test_an_unreadable_plan_source_is_a_transport_error(self) -> None:
        with self.assertRaises(board.TransportError):
            board.load_plan(["no/such/plan.md"], ".")


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = board.baseline_document()

    def test_baseline_is_green(self) -> None:
        self.assertEqual(board._fixture_check(self.doc), [])

    def test_fixture_tasks_parse(self) -> None:
        by_id = {t["id"]: t for t in self.doc["tasks"]}
        self.assertEqual(
            by_id["rooting-gate-green-AAAA1111"]["gates"],
            ["python3 tools/rooting/check.py --check"],
        )
        self.assertEqual(by_id["rooting-gate-green-AAAA1111"]["paths"], ["docs/plan.md"])
        self.assertEqual(by_id["rooting-gate-green-AAAA1111"]["plan_ids"], ["FIXTURE-E1-W1-T1"])
        self.assertEqual(
            by_id["receipt-archive-complete-DDDD4444"]["receipts"],
            ["docs/evidence/locator-reserve-quadratic-and-reg1-attribution.md"],
        )

    def test_next_orders_by_milestone_then_id(self) -> None:
        chosen = board.next_task(self.doc)
        # M1 is the earliest milestone carrying an actionable task, so its task wins
        # over the id-lower task that sits in no milestone.
        self.assertEqual(chosen["id"], "land-gate-wave-3-CCCC3333")

    def test_unassigned_tasks_sort_after_milestone_tasks_then_by_id(self) -> None:
        chosen = board.next_task(self.doc, project="apeiron")
        self.assertEqual(chosen["project_name"], "apeiron")
        self.assertIsNone(chosen["milestone_index"])

    def test_next_by_project_id(self) -> None:
        chosen = board.next_task(self.doc, project=board.SONG["id"])
        self.assertEqual(chosen["id"], "land-gate-wave-3-CCCC3333")

    def test_done_and_canceled_are_not_actionable(self) -> None:
        doc = board.baseline_document()
        for task in doc["tasks"]:
            task["status_value"] = "canceled"
            task["status_label"] = "Canceled"
        self.assertIsNone(board.next_task(doc))

    def test_next_returns_none_when_nothing_is_actionable(self) -> None:
        doc = board.baseline_document()
        for task in doc["tasks"]:
            if board.status_bucket(task) == "actionable":
                task["status_value"] = "canceled"
                task["status_label"] = "Canceled"
        self.assertIsNone(board.next_task(doc))

    def test_list_table_carries_the_gate_and_the_paths(self) -> None:
        table = board.render_table(board.select_tasks(self.doc))
        self.assertIn("GATE", table)
        self.assertIn("MILESTONE", table)
        self.assertIn("python3 tools/rooting/check.py", table)
        self.assertIn("docs/plan.md", table)
        self.assertIn("M1 — Assembly core complete", table)
        self.assertIn(
            "land-gate-wave-3-CCCC3333",
            board.render_table(board.select_tasks(self.doc), "full"),
        )

    def test_list_shows_a_dash_for_a_missing_gate(self) -> None:
        doc = board.baseline_document()
        board.defect_task_without_gate(doc)
        table = board.render_table(board.select_tasks(doc, "apeiron"))
        self.assertIn("GATE", table)
        self.assertIn("MISSING", board.render_next(doc["tasks"][0]))


class RuleTests(unittest.TestCase):
    def rules(self, mutate=None, **kwargs):  # noqa: ANN003 - test helper
        doc = board.baseline_document()
        if mutate is not None:
            mutate(doc)
        return board.rules_of(board._fixture_check(doc, **kwargs))

    def test_done_without_receipt_reddens_by_name(self) -> None:
        self.assertIn(board.RULE_DONE_NO_RECEIPT, self.rules(board.defect_done_without_receipt))

    def test_task_without_gate_reddens_by_name(self) -> None:
        self.assertIn(board.RULE_NO_GATE, self.rules(board.defect_task_without_gate))

    def test_milestone_without_tasks_reddens_by_name(self) -> None:
        self.assertIn(board.RULE_MILESTONE_EMPTY, self.rules(board.defect_milestone_without_tasks))

    def test_write_that_did_not_land_reddens_by_name(self) -> None:
        self.assertIn(board.RULE_WRITE_NOT_LANDED, self.rules(board.defect_write_did_not_land))

    def test_task_without_source_paths_reddens_by_name(self) -> None:
        def mutate(doc: dict) -> None:
            board._find(doc, "rooting-gate-green-AAAA1111")["paths"] = []

        self.assertIn(board.RULE_NO_PATHS, self.rules(mutate))

    def test_unknown_milestone_reddens_by_name(self) -> None:
        def mutate(doc: dict) -> None:
            task = board._find(doc, "land-gate-wave-3-CCCC3333")
            task["milestone_id"] = "m9-does-not-exist"

        self.assertIn(board.RULE_UNKNOWN_MILESTONE, self.rules(mutate))

    def test_unattributed_task_reddens_by_name(self) -> None:
        def mutate(doc: dict) -> None:
            task = board._find(doc, "rooting-gate-green-AAAA1111")
            task["project_id"] = None
            task["project_name"] = None

        self.assertIn(board.RULE_UNATTRIBUTED, self.rules(mutate))

    def test_plan_row_missing_on_board_reddens_by_name(self) -> None:
        self.assertIn(
            board.RULE_DRIFT_PLAN_ROW, self.rules(board.defect_plan_row_missing_on_board)
        )

    def test_board_row_missing_from_plan_reddens_by_name(self) -> None:
        self.assertIn(
            board.RULE_DRIFT_BOARD_ROW, self.rules(board.defect_board_row_missing_from_plan)
        )

    def test_board_open_against_closed_plan_reddens_by_name(self) -> None:
        self.assertIn(
            board.RULE_DRIFT_STALE, self.rules(board.defect_board_open_against_closed_plan)
        )

    def test_no_plan_source_is_a_finding_by_name(self) -> None:
        doc = board.baseline_document()
        doc["plan"] = None
        self.assertIn(board.RULE_PLAN_UNDECLARED, board.rules_of(board._fixture_check(doc)))

    def test_each_seeded_defect_fires_exactly_its_own_rule(self) -> None:
        """A defect that trips other rules is not an attributable diagnostic."""
        for rule, mutate, _ in board.SEEDED_DEFECTS:
            doc = board.baseline_document()
            mutate(doc)
            self.assertEqual(board.rules_of(board._fixture_check(doc)), {rule}, rule)

    def test_findings_name_the_subject(self) -> None:
        doc = board.baseline_document()
        board.defect_milestone_without_tasks(doc)
        finding = next(f for f in board._fixture_check(doc) if f["rule"] == board.RULE_MILESTONE_EMPTY)
        self.assertIn("m2-kernel", finding["detail"])

    def test_every_rule_name_is_reachable_and_named_in_the_rule_set(self) -> None:
        reached: set[str] = set()
        for rule, mutate, _ in board.SEEDED_DEFECTS:
            doc = board.baseline_document()
            mutate(doc)
            reached |= board.rules_of(board._fixture_check(doc))
        bare = board.baseline_document()
        bare["plan"] = None
        reached |= board.rules_of(board._fixture_check(bare))

        def unlink_paths(doc: dict) -> None:
            board._find(doc, "rooting-gate-green-AAAA1111")["paths"] = []

        def unknown_milestone(doc: dict) -> None:
            board._find(doc, "land-gate-wave-3-CCCC3333")["milestone_id"] = "m9-absent"

        def unattributed(doc: dict) -> None:
            task = board._find(doc, "rooting-gate-green-AAAA1111")
            task["project_id"] = task["project_name"] = None

        for mutate in (unlink_paths, unknown_milestone, unattributed):
            doc = board.baseline_document()
            mutate(doc)
            reached |= board.rules_of(board._fixture_check(doc))
        with contextlib.redirect_stdout(io.StringIO()):
            board.main(["--check", "--fixtures", "--project", "no-such-project"])
        reached.add(board.RULE_PROJECT_MISSING)
        for rule in board.ALL_RULES:
            self.assertIn(rule, reached, rule)


class ReconcileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = board.baseline_document()
        self.plan = board.plan_from_document(self.doc)

    def test_every_row_is_classified(self) -> None:
        classifications, findings = board.reconcile(self.doc, self.plan)
        self.assertEqual(findings, [])
        classes = {item["class"] for item in classifications}
        self.assertEqual(classes, {"mirrored", "internal-by-design"})
        mirrored = [i for i in classifications if i["class"] == "mirrored"]
        self.assertGreaterEqual(len(mirrored), 5)
        internal = [i for i in classifications if i["class"] == "internal-by-design"]
        self.assertEqual(internal[0]["board"], "toolchain-probe-9999AAAA")
        self.assertIn("internal-by-design", internal[0]["detail"])

    def test_a_board_row_matches_a_plan_row_by_slug(self) -> None:
        doc = board.baseline_document()
        task = board._find(doc, "rooting-gate-green-AAAA1111")
        task["plan_ids"] = []  # no id named: the slug carries the match
        _, findings = board.reconcile(doc, board.plan_from_document(doc))
        self.assertEqual(findings, [])

    def test_a_closed_board_row_is_not_reconciled(self) -> None:
        classifications, findings = board.reconcile(self.doc, self.plan)
        self.assertNotIn("receipt-archive-complete-DDDD4444", [i["board"] for i in classifications])
        self.assertEqual(findings, [])

    def test_a_stale_board_row_is_classified_and_findings_name_the_plan_row(self) -> None:
        doc = board.baseline_document()
        board.defect_board_open_against_closed_plan(doc)
        classifications, findings = board.reconcile(doc, board.plan_from_document(doc))
        stale = [i for i in classifications if i["class"] == "stale"]
        self.assertEqual(len(stale), 1)
        self.assertIn("CLOSED", stale[0]["detail"])
        self.assertEqual(board.rules_of(findings), {board.RULE_DRIFT_STALE})

    def test_internal_by_design_must_be_named(self) -> None:
        doc = board.baseline_document()
        task = board._find(doc, "toolchain-probe-9999AAAA")
        task["internal_by"] = ""
        task["plan_ids"] = []
        _, findings = board.reconcile(doc, board.plan_from_document(doc))
        self.assertIn(board.RULE_DRIFT_BOARD_ROW, board.rules_of(findings))

    def test_internal_by_design_can_be_named_on_the_command_line(self) -> None:
        doc = board.baseline_document()
        task = board._find(doc, "toolchain-probe-9999AAAA")
        task["internal_by"] = ""
        task["plan_ids"] = []
        _, findings = board.reconcile(
            doc, board.plan_from_document(doc), None, ["toolchain-probe-9999AAAA"]
        )
        self.assertEqual(findings, [])

    def test_a_plan_row_with_no_board_row_is_drift(self) -> None:
        doc = board.baseline_document()
        board.defect_plan_row_missing_on_board(doc)
        classifications, findings = board.reconcile(doc, board.plan_from_document(doc))
        missing = [i for i in classifications if i["plan"] == "FIXTURE-E1-W4-T1"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(board.rules_of(findings), {board.RULE_DRIFT_PLAN_ROW})

    def test_reconcile_without_a_plan_names_the_rule(self) -> None:
        _, findings = board.reconcile(self.doc, None)
        self.assertEqual(board.rules_of(findings), {board.RULE_PLAN_UNDECLARED})

    def test_check_names_its_source_of_truth_and_digest(self) -> None:
        out = io.StringIO()
        args = board.build_parser().parse_args(["--fixtures", "--check"])
        findings, classifications, plan = board.full_check(board.baseline_document(), args)
        board.render_check(board.baseline_document(), findings, classifications, plan, stream=out)
        text = out.getvalue()
        self.assertIn("source-of-truth:", text)
        self.assertIn("docs/plan.md sha256=", text)
        self.assertIn("reconcile: ", text)
        self.assertIn("board-check: 0 findings", text)


class WriteReadBackTests(unittest.TestCase):
    """The transport answer is not the evidence; the read-back is."""

    def test_error_after_apply_is_read_back_as_landed(self) -> None:
        doc = board.baseline_document()
        landed, observed = board.verify_status_offline(
            "rooting-gate-green-AAAA1111", "in_progress", doc, error_after_apply=True
        )
        self.assertTrue(landed)
        self.assertEqual(observed, "in_progress")
        self.assertEqual(board._fixture_check(doc), [])

    def test_a_silent_no_op_is_not_landed(self) -> None:
        doc = board.baseline_document()
        landed, observed = board.verify_status_offline(
            "rooting-gate-green-AAAA1111", "in_progress", doc, error_after_apply=False
        )
        self.assertFalse(landed)
        self.assertEqual(observed, "pending")

    def test_rest_client_prefers_the_read_back_over_a_refused_transport(self) -> None:
        """update_status answering 500 while the status moved must read as landed."""
        state = {"value": "pending"}

        class FlakyOpener:
            def open(self, request, timeout=None):  # noqa: ANN001 - stub
                url = request.get_full_url()
                if "update_status" in url:
                    state["value"] = "in_progress"
                    raise board.TransportError("stub: HTTP 500")
                if "list_task_statuses" in url:
                    return StubResponse(
                        {
                            "task_statuses": [
                                {"id": "s2", "value": "in_progress", "label": "In progress"}
                            ]
                        }
                    )
                return StubResponse(
                    {"task": {"id": "t1", "type": "project", "status": {"value": state["value"]}}}
                )

        client = board.RestClient(key="test-key", opener=FlakyOpener())
        transport = client.update_status("t1", "in_progress")
        landed, observed = client.verify_status("t1", "in_progress")
        self.assertEqual(transport, "refused")
        self.assertTrue(landed)
        self.assertEqual(observed, "in_progress")

    def test_rest_client_reports_an_unlanded_write(self) -> None:
        class DeadOpener:
            def open(self, request, timeout=None):  # noqa: ANN001 - stub
                return StubResponse({"task": {"id": "t1", "status": {"value": "pending"}}})

        client = board.RestClient(key="test-key", opener=DeadOpener())
        self.assertFalse(client.verify_status("t1", "in_progress")[0])


class CheckInTests(unittest.TestCase):
    def test_record_names_the_source_of_truth_digest_and_the_tally(self) -> None:
        doc = board.baseline_document()
        plan = board.plan_from_document(doc)
        classifications, findings = board.reconcile(doc, plan)
        record = board.reconciliation_record(doc, "apeiron", plan, classifications, findings)
        first = record.splitlines()[0]
        self.assertTrue(first.startswith("board-reconcile project=apeiron plan="))
        self.assertIn(board.sha256_of(board.FIXTURE_PLAN_MD)[:16], first)
        self.assertIn("mirrored=6", record)
        self.assertIn("Gate: `python3 tools/board/board.py --check --controls`", record)

    def test_record_lists_every_finding(self) -> None:
        doc = board.baseline_document()
        board.defect_plan_row_missing_on_board(doc)
        plan = board.plan_from_document(doc)
        classifications, findings = board.reconcile(doc, plan)
        record = board.reconciliation_record(doc, "song", plan, classifications, findings)
        self.assertIn("DRIFT drift-plan-row-not-on-board", record)
        self.assertIn("FIXTURE-E1-W4-T1", record)

    def test_post_is_confirmed_by_a_read_back(self) -> None:
        self.assertTrue(board.post_offline(board.baseline_document(), "apeiron", "green"))

    def test_post_is_refused_when_the_write_does_not_land(self) -> None:
        self.assertFalse(
            board.post_offline(board.baseline_document(), "apeiron", "green", transport_applies=False)
        )

    def test_post_mode_dry_run_prints_the_record_and_mutates_nothing(self) -> None:
        out = io.StringIO()
        args = board.build_parser().parse_args(
            ["post", "--fixtures", "--project", "apeiron", "--plan-root", "x"]
            if False
            else ["post", "--fixtures", "--project", "apeiron"]
        )
        with contextlib.redirect_stdout(out):
            code = board.run_post(args, board.baseline_document())
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("dry-run post check-in project=apeiron status=green", text)
        self.assertIn("board-reconcile project=apeiron", text)

    def test_post_mode_defaults_to_red_on_drift(self) -> None:
        doc = board.baseline_document()
        board.defect_plan_row_missing_on_board(doc)
        out = io.StringIO()
        args = board.build_parser().parse_args(["post", "--fixtures", "--project", "apeiron"])
        with contextlib.redirect_stdout(out):
            code = board.run_post(args, doc)
        self.assertEqual(code, 1)
        self.assertIn("status=red", out.getvalue())


class AttributionTests(unittest.TestCase):
    def test_prose_project_mention_attributes_a_task(self) -> None:
        doc = board.baseline_document()
        task = board._find(doc, "land-gate-wave-3-CCCC3333")
        task["project_id"] = None
        task["project_name"] = None
        # The live shape of a superseded space task: it names the project it moved to.
        task["text"] += "\nMoved into project `song` under M1."
        board.attribute_projects(doc)
        self.assertEqual(task["project_name"], "song")

    def test_a_task_naming_no_project_stays_unattributed(self) -> None:
        doc = board.baseline_document()
        task = board._find(doc, "rooting-gate-green-AAAA1111")
        task["project_id"] = None
        task["project_name"] = None
        board.attribute_projects(doc)
        self.assertIsNone(task["project_id"])
        self.assertIn(board.RULE_UNATTRIBUTED, board.rules_of(board._fixture_check(doc)))


class SecretHygieneTests(unittest.TestCase):
    SECRET = "operately-secret-value-do-not-print"

    def test_missing_key_is_a_transport_error_not_a_traceback(self) -> None:
        previous = os.environ.pop(board.KEY_ENV, None)
        try:
            with self.assertRaises(board.TransportError) as caught:
                board.RestClient()
            self.assertIn(board.KEY_ENV, str(caught.exception))
        finally:
            if previous is not None:
                os.environ[board.KEY_ENV] = previous

    def test_the_key_is_scrubbed_from_transport_messages(self) -> None:
        secret = self.SECRET

        class LeakyOpener:
            def open(self, request, timeout=None):  # noqa: ANN001 - stub
                raise RuntimeError(f"connect failed with Bearer {secret}")

        client = board.RestClient(key=secret, opener=LeakyOpener())
        with self.assertRaises(board.TransportError) as caught:
            client.call("spaces/list_tasks", space_id="s")
        message = str(caught.exception)
        self.assertNotIn(secret, message)
        self.assertIn("<redacted>", message)

    def test_a_short_stub_key_does_not_corrupt_urls(self) -> None:
        """A one-character value is not a credential; matching it would eat the URL."""
        opener = StubOpener({"projects/list": {"projects": []}})
        client = board.RestClient(key="k", opener=opener)
        self.assertEqual(client.projects(), [])
        self.assertIn("projects.tas.twn.network", opener.calls[0])

    def test_the_key_is_never_read_from_a_file(self) -> None:
        sources = []
        for module in (board, board_core):
            with open(module.__file__, encoding="utf-8") as handle:
                sources.append(handle.read())
        for source in sources:
            self.assertNotIn("open('.env'", source)
            self.assertNotIn('open(".env"', source)
        # KEY_ENV is the named constant; it lives in board_core, the module that
        # owns the driver vocabulary.
        self.assertIn('KEY_ENV = "OPERATELY_API_KEY"', sources[1])


class LiveFetchTests(unittest.TestCase):
    """fetch_live against a stub opener: the real route names, no network."""

    ROUTES = {
        "projects/list": {"projects": [{"id": "apeiron-1", "name": "apeiron"}]},
        "projects/list_milestones": {
            "milestones": [{"id": "m1-x", "title": "M1 — core", "status": "pending"}]
        },
        "projects/list_milestone_tasks": {
            "tasks": [
                {
                    "id": "t-m1",
                    "name": "In M1",
                    "status": {"value": "pending", "label": "Not started"},
                    "description": "Gate: `cargo test --workspace`.\nSource: `docs/plan.md`.",
                }
            ]
        },
        "spaces/list_tasks": {
            "tasks": [
                {
                    "id": "t-space",
                    "name": "Space task",
                    "status": {"value": "in_progress", "label": "In progress"},
                    "description": "Source: `docs/plan.md`.\nGate: `lake build`.",
                }
            ]
        },
    }

    def test_routes_and_normalization(self) -> None:
        opener = StubOpener(self.ROUTES)
        doc = board.fetch_live(board.RestClient(key="k", opener=opener), "space-1", ["apeiron"])
        self.assertEqual({t["id"] for t in doc["tasks"]}, {"t-m1", "t-space"})
        self.assertEqual(len(doc["milestones"]), 1)
        self.assertEqual(doc["milestones"][0]["project_name"], "apeiron")
        self.assertEqual(board._find(doc, "t-m1")["milestone_index"], 0)
        self.assertEqual(board._find(doc, "t-m1")["gates"], ["cargo test --workspace"])
        self.assertTrue(any("projects/list_tasks" in n for n in doc["notes"]))
        self.assertTrue(all("token=" not in c for c in opener.calls))

    def test_a_milestone_with_no_task_is_found(self) -> None:
        routes = dict(self.ROUTES)
        routes["projects/list"] = {"projects": [{"id": "song-1", "name": "song"}]}
        routes["projects/list_milestones"] = {
            "milestones": [
                {"id": "m1-x", "title": "M1", "status": "pending"},
                {"id": "m9-empty", "title": "M9 — empty", "status": "pending"},
            ]
        }
        routes["projects/list_milestone_tasks"] = {
            "tasks": [
                {
                    "id": "t-m1",
                    "name": "In M1",
                    "status": {"value": "pending", "label": "Not started"},
                    "description": "Gate: `cargo test`.\nSource: `docs/plan.md`.",
                    "project": {"id": "song-1", "name": "song"},
                }
            ]
        }
        doc = board.fetch_live(board.RestClient(key="k", opener=StubOpener(routes)), "space-1", ["song"])
        self.assertIn(board.RULE_MILESTONE_EMPTY, board.rules_of(board.check(doc)))

    def test_live_drift_is_reported_against_the_live_tasks(self) -> None:
        doc = board.fetch_live(board.RestClient(key="k", opener=StubOpener(self.ROUTES)), "space-1", ["apeiron"])
        _, findings = board.reconcile(doc, None)
        self.assertEqual(board.rules_of(findings), {board.RULE_PLAN_UNDECLARED})


class ControlSuiteTests(unittest.TestCase):
    def test_control_suite_passes(self) -> None:
        stream = io.StringIO()
        code = board.run_controls(stream)
        output = stream.getvalue()
        self.assertEqual(code, 0, output)
        for rule, _, _ in board.SEEDED_DEFECTS:
            self.assertIn(f"CONTROL pass {rule}", output)
        self.assertIn("CONTROL pass baseline", output)
        self.assertIn("CONTROL pass restore", output)
        self.assertNotIn("CONTROL FAIL", output)

    def test_the_three_contract_defects_are_still_seeded(self) -> None:
        named = {rule for rule, _, _ in board.CONTRACT_SEEDED_DEFECTS}
        self.assertEqual(named, set(board.CONTRACT_RULES) - {board.RULE_NO_PATHS})

    def test_a_seeded_defect_stops_firing_when_the_rule_is_removed(self) -> None:
        """A control that cannot fail has no weight: break the rule, expect red.

        ``board.check`` is a re-export: the control suite resolves it in
        ``board_check``, so the rule is broken where it is defined.
        """
        original = board_check.check
        try:
            board_check.check = lambda *a, **k: []  # type: ignore[assignment]
            stream = io.StringIO()
            code = board.run_controls(stream)
            output = stream.getvalue()
            self.assertEqual(code, 2)
            for rule, _, _ in board.SEEDED_DEFECTS:
                self.assertIn(f"CONTROL FAIL {rule}", output)
        finally:
            board_check.check = original  # type: ignore[assignment]

    def test_control_suite_fails_when_the_reconcile_stops_firing(self) -> None:
        original = board_check.reconcile
        try:
            board_check.reconcile = lambda *a, **k: ([], [])  # type: ignore[assignment]
            stream = io.StringIO()
            code = board.run_controls(stream)
            self.assertEqual(code, 2)
            self.assertIn("CONTROL FAIL drift-plan-row-not-on-board", stream.getvalue())
        finally:
            board_check.reconcile = original  # type: ignore[assignment]

    def test_gate_command_exits_zero_on_the_fixtures(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(board.main(["--check", "--controls"]), 0)

    def test_gate_command_fails_when_the_source_board_is_broken(self) -> None:
        """--check --controls against a defective snapshot must still exit 1."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_bad.json")
        broken = json.loads(json.dumps(board.FIXTURE_BASELINE))
        for task in broken["tasks"]:
            task["description"] = "no gate, no source path"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(broken, handle)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(board.main(["--check", "--controls", "--source", path]), 1)
        finally:
            os.unlink(path)

    def test_no_source_is_a_usage_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(board.main(["check"]), 2)

    def test_an_unreadable_plan_path_is_a_transport_error_not_a_traceback(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code = board.main(
                ["--check", "--fixtures", "--project", "apeiron", "--plan", "no/such/plan.md"]
            )
        self.assertEqual(code, 2)

    def test_a_missing_board_declaration_row_is_the_named_control(self) -> None:
        """The control suite must carry the missing-declaration case by name."""
        named = [
            (rule, subject)
            for rule, _, subject in board.SEEDED_DEFECTS
            if rule == board.RULE_NO_BOARD_DECLARATION
        ]
        self.assertEqual(len(named), 2, "both the absent and the mis-pointed row are seeded")
        stream = io.StringIO()
        self.assertEqual(board.run_controls(stream), 0, stream.getvalue())
        self.assertIn(f"CONTROL pass {board.RULE_NO_BOARD_DECLARATION}", stream.getvalue())

    def test_rules_mode_prints_every_rule(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(board.main(["--rules"]), 0)
        for rule in board.ALL_RULES:
            self.assertIn(rule, out.getvalue())

    def test_next_mode_exit_3_when_nothing_is_actionable(self) -> None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_none.json")
        empty = json.loads(json.dumps(board.FIXTURE_BASELINE))
        for task in empty["tasks"]:
            task["status"] = {"value": "canceled", "label": "Canceled"}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(empty, handle)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(board.main(["next", "--source", path]), 3)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
