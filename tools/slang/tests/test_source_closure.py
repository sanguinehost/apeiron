"""Exercise the production audit source scan without starting a Lean process."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RUNNER_PATH = Path(__file__).resolve().parents[1] / "zero_axioms.py"
SPEC = importlib.util.spec_from_file_location("zero_axioms_source_closure", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)

ROOT_MODULE = "Slang.Generated.Table"
CHILD_A = ROOT_MODULE + ".Chunk01"
CHILD_B = ROOT_MODULE + ".Chunk02"
SHARED = ROOT_MODULE + ".Shared"
DECLARATION = "Slang.Fixture.checked"


class SourceClosureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="apeiron-source-closure-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.corpus = self.repo / "proof/corpus"
        self.write_module("Slang.Source.Bytes", "def byteFixture : Nat := 0\n")
        self.write_module("Slang.Generated.TableSupport", "def supportFixture : Nat := 0\n")
        self.write_module(ROOT_MODULE, "import Slang.Source.Bytes\nimport " + CHILD_A + "\n")
        self.clean_child = "import Slang.Generated.TableSupport\ndef unused : Nat := 0\n"
        self.write_module(CHILD_A, self.clean_child)
        self.write_module("Slang.Fixture", f"#audit_empty {DECLARATION} : (0 : Nat) = 0\n")
        self.data = {
            "schema": 1,
            "defaults": {"memory_max": "6G", "memory_swap_max": "0", "timeout_seconds": 900},
            "forbidden_tokens": ["sorry", "admit", "noncomputable", "classical"],
            "stages": {
                "fixture": {
                    "kind": "active", "wave": "test", "title": "source closure",
                    "audit_module": "Slang/Fixture.lean",
                    "source_files": ["proof/corpus/Slang/Generated/Table.lean", "proof/corpus/Slang/Fixture.lean"],
                    "source_import_closures": [{
                        "root": ROOT_MODULE,
                        "allowed_imports": ["Slang.Source.Bytes", "Slang.Generated.TableSupport"],
                    }],
                    "declarations": [{
                        "name": DECLARATION, "expected_type": "(0 : Nat) = 0",
                        "module": "Slang.Fixture", "path": "proof/corpus/Slang/Fixture.lean",
                    }],
                }
            },
        }

    def write_module(self, name, content):
        path = self.corpus.joinpath(*name.split(".")).with_suffix(".lean")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def registry(self, data=None):
        path = self.repo / "registry.json"
        path.write_text(json.dumps(self.data if data is None else data), encoding="utf-8")
        return RUNNER.load_registry(path)

    def baseline(self, expected_error=None):
        registry = self.registry()
        stage = registry.stages["fixture"]
        with patch.object(RUNNER, "REPO", self.repo), patch.object(RUNNER, "CORPUS", self.corpus), \
                patch.object(RUNNER, "WORK_DIR", self.repo / "work"):
            with patch.object(RUNNER, "run_lean", side_effect=self.fake_lean) as lean:
                with contextlib.redirect_stdout(io.StringIO()):
                    if expected_error is None:
                        self.assertEqual(RUNNER.run_baseline(stage, registry), ([DECLARATION], ()))
                        # once for the completeness driver, once for the audit module
                        self.assertEqual(lean.call_count, 2)
                    else:
                        with self.assertRaisesRegex(RUNNER.TokenError, expected_error):
                            RUNNER.run_baseline(stage, registry)
                        lean.assert_not_called()

    def fake_lean(self, registry, target, timeout=None, root=None):
        """The completeness copy reports an empty declared set; the module audits."""
        if root is not None:
            return 0, f"MODULE_MAIN `Slang.Fixture\nMODULE_OWN_COUNT 0\n"
        return 0, f"EMPTY_AXIOMS_OK {DECLARATION}\n"

    def test_unused_forbidden_declarations_reject_then_restore(self):
        self.baseline()
        for source in ("noncomputable def unused : Nat := 0\n", "theorem unused : True := by sorry\n"):
            with self.subTest(source=source):
                self.write_module(CHILD_A, source)
                self.baseline("forbidden token")
                self.write_module(CHILD_A, self.clean_child)
                self.baseline()

    def test_missing_child_rejects_then_restore(self):
        self.baseline()
        self.corpus.joinpath(*CHILD_A.split(".")).with_suffix(".lean").unlink()
        self.baseline("source import missing")
        self.write_module(CHILD_A, self.clean_child)
        self.baseline()

    def test_cycle_rejects_then_restore(self):
        self.baseline()
        self.write_module(CHILD_A, "import " + CHILD_B + "\n")
        self.write_module(CHILD_B, "import " + CHILD_A + "\n")
        self.baseline("source import cycle")
        self.write_module(CHILD_A, self.clean_child)
        self.baseline()

    def test_root_backedge_rejects(self):
        self.write_module(CHILD_A, "import " + ROOT_MODULE + "\n")
        self.baseline("source import cycle")

    def test_shared_ancestor_and_duplicate_import_are_not_cycles(self):
        self.write_module(ROOT_MODULE, f"import {CHILD_A} {CHILD_B} {CHILD_A}\n")
        self.write_module(CHILD_A, f"import {SHARED}\n")
        self.write_module(CHILD_B, f"import {SHARED}\n")
        self.write_module(SHARED, "def sharedValue : Nat := 0\n")
        self.baseline()
        registry = self.registry()
        paths = RUNNER.check_source_closure(
            registry.stages["fixture"].source_import_closures[0], registry, self.corpus
        )
        self.assertEqual(len(paths), 4)
        self.assertEqual(len(set(paths)), 4)
        self.write_module(SHARED, "noncomputable def sharedValue : Nat := 0\n")
        self.baseline("forbidden token")

    def test_external_import_escape_rejects_then_restore(self):
        self.baseline()
        self.write_module(CHILD_A, "import Slang.Generated.Other\n")
        self.write_module("Slang.Generated.Other", "def other : Nat := 0\n")
        self.baseline("outside registered closure")
        self.write_module(CHILD_A, self.clean_child)
        self.baseline()

    def test_malformed_path_import_rejects_then_restore(self):
        for name in ("Slang.Generated.Table../Other", '"../Other"', "Slang.Generated.Table.«../Other»"):
            with self.subTest(name=name):
                self.write_module(CHILD_A, "import " + name + "\n")
                self.baseline("invalid source import")
                self.write_module(CHILD_A, self.clean_child)
                self.baseline()

    def test_symlink_escape_rejects_then_restore(self):
        child = self.write_module(CHILD_A, self.clean_child)
        outside = self.repo / "outside.lean"
        outside.write_text("def escaped : Nat := 0\n", encoding="utf-8")
        child.unlink()
        child.symlink_to(outside)
        self.baseline("escaped source import")
        child.unlink()
        self.write_module(CHILD_A, self.clean_child)
        self.baseline()

    def test_explicit_sources_still_reject(self):
        path = self.write_module("Slang.Fixture", "noncomputable def ignored : Nat := 0\n")
        self.baseline("forbidden token")
        path.write_text(f"#audit_empty {DECLARATION} : (0 : Nat) = 0\n", encoding="utf-8")
        self.baseline()

    def test_comments_do_not_hide_or_create_imports(self):
        self.write_module(ROOT_MODULE, f"/- import Slang.Generated.Other /- nested -/ -/\nimport /- comment -/ {CHILD_A} -- tail\n")
        self.baseline()
        self.write_module(CHILD_A, "import /- nested /- c -/ -/ Slang.Generated.Other\n")
        self.baseline("outside registered closure")

    def test_invalid_registered_scope_rejects(self):
        for change in (
            {"root": "../../Other"},
            {"root": "Slang.Generated.Unlisted"},
            {"allowed_imports": [CHILD_A]},
            {"allowed_imports": ["Slang.Source.Bytes", "Slang.Source.Bytes"]},
            {"unknown": True},
        ):
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                data["stages"]["fixture"]["source_import_closures"][0].update(change)
                with self.assertRaises(RUNNER.RegistryError):
                    self.registry(data)

    def test_unsupported_import_headers_reject(self):
        for header in ("public import " + CHILD_A, "import", "import " + CHILD_A + "\n  " + CHILD_B):
            with self.subTest(header=header):
                self.write_module(ROOT_MODULE, header + "\n")
                self.baseline("(?:invalid source import|unsupported source import header)")

    def test_missing_permitted_dependency_rejects(self):
        self.corpus.joinpath("Slang/Source/Bytes.lean").unlink()
        self.baseline("missing or escaped permitted")


if __name__ == "__main__":
    unittest.main()
