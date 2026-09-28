"""Exercise real table rendering, output freshness and CLI in disposable trees."""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import export
from export_parts import table

SONG = export.REPO_ROOT.parent / "song"


class TableOutputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monolith, cls.paths = export.render(SONG)
        cls.header, cls.blocks, cls.footer = table.declaration_layout(cls.monolith)
        cls.groups = table.pack_blocks(cls.blocks)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="apeiron-table-outputs-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "Foo.lean"

    def outputs(self, prefix, output=None):
        with patch.object(table, "PREFIX_COUNT", prefix):
            return export.render_table_outputs(SONG, output or self.output)

    def write(self, outputs):
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def test_zero_preserves_monolith_and_arbitrary_legacy_output(self):
        unusual = self.root / "legacy-output.txt"
        self.assertEqual(self.outputs(0, unusual), {unusual: self.monolith})

    def test_current_layout_and_complete_declaration_order(self):
        self.assertEqual([sum(len(b.splitlines()) for b in g) for g in self.groups],
                         [1397, 1397, 1400, 1400, 1395, 1395, 1395, 1395, 1398, 1398, 429])
        self.assertEqual(self.header + "".join(self.blocks) + self.footer, self.monolith)
        for block in self.blocks:
            if "def tablePart" in block or "def cellPart" in block or "def stdPart" in block:
                self.assertIn("Length :", block)
        previous = {}
        for prefix in range(1, 12):
            outputs = self.outputs(prefix)
            for path, text in previous.items():
                if path != self.output:
                    self.assertEqual(outputs[path], text)
            shards = [outputs[self.root / "Foo" / f"Chunk{i:02d}.lean"]
                      for i in range(1, prefix + 1)]
            for i, text in enumerate(shards, 1):
                self.assertLessEqual(len(text.splitlines()), 1500)
                self.assertIn("\n".join(export.SOURCE_BYTE_HELPER), text)
                self.assertIn("".join(self.groups[i - 1]), text)
                self.assertNotIn("import Slang.Generated.Table\n", text)
                if i > 1:
                    self.assertIn(f"import Foo.Chunk{i - 1:02d}\n", text)
            root = outputs[self.output]
            self.assertIn(f"import Foo.Chunk{prefix:02d}\n", root)
            suffix = "".join(b for g in self.groups[prefix:] for b in g)
            self.assertTrue(root.endswith(suffix + self.footer))
            previous = outputs
        self.assertLessEqual(len(previous[self.output].splitlines()), 1500)

    def test_module_placement_and_invalid_paths(self):
        internal = export.REPO_ROOT / "proof/corpus/Slang/Generated/Alternate.lean"
        outputs = self.outputs(1, internal)
        self.assertIn(internal.with_suffix("") / "Chunk01.lean", outputs)
        self.assertIn("import Slang.Generated.Alternate.Chunk01", outputs[internal])
        self.assertNotIn(export.DEFAULT_OUTPUT, outputs)
        for path in (self.root / "bad-name.lean", self.root / "Foo.txt"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "module identifier|suffix"):
                self.outputs(1, path)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.outputs(12)

    def test_missing_changed_extra_and_stale_shards(self):
        outputs = self.outputs(1)
        self.write(outputs)
        table.check_output_map(outputs, self.output)
        leaf = self.root / "Foo/Chunk01.lean"
        original = leaf.read_text()
        leaf.unlink()
        with self.assertRaisesRegex(ValueError, "differs"):
            table.check_output_map(outputs, self.output)
        leaf.write_text(original.replace("sourceByte 47", "sourceByte 48", 1))
        self.assertNotEqual(leaf.read_text(), original)
        with self.assertRaisesRegex(ValueError, "differs"):
            table.check_output_map(outputs, self.output)
        leaf.write_text(original)
        extra = self.root / "Foo/Chunk99.lean"
        extra.write_text("def extra : Nat := 0\n")
        with self.assertRaisesRegex(ValueError, "unexpected table shard"):
            table.check_output_map(outputs, self.output)
        with self.assertRaisesRegex(ValueError, "unexpected table shard"):
            table.reject_stale_shards(outputs, self.output)
        self.assertTrue(extra.exists())
        extra.unlink()
        table.check_output_map(outputs, self.output)
        with self.assertRaisesRegex(ValueError, "unexpected table shard"):
            table.check_output_map(self.outputs(0), self.output)

    def test_real_declaration_order_mutation_rejects(self):
        outputs = self.outputs(1)
        self.write(outputs)
        leaf = self.root / "Foo/Chunk01.lean"
        original = leaf.read_text()
        first, second = self.groups[0][:2]
        self.assertIn(first + second, original)
        leaf.write_text(original.replace(first + second, second + first, 1))
        with self.assertRaisesRegex(ValueError, "differs"):
            table.check_output_map(outputs, self.output)
        leaf.write_text(original)
        table.check_output_map(outputs, self.output)

    def test_indivisible_count_and_full_file_overflow(self):
        with self.assertRaisesRegex(ValueError, "indivisible"):
            table.pack_blocks(["x\n" * 1401])
        with self.assertRaisesRegex(ValueError, "shard count"):
            table.pack_blocks(["x\n" * 1400] * 13)
        with patch.object(table, "PREFIX_COUNT", 1):
            with self.assertRaisesRegex(ValueError, "full-file"):
                table.render_table_outputs(self.monolith, self.output, ("x\n" * 1500,), self.root)

    def test_cli_alternate_output_and_real_prefix_mutation(self):
        # Copy only the exporter modules. Every generated write stays in this
        # temporary repository; the live source and generated tree are read-only.
        tools = self.root / "tools/slang"
        tools.mkdir(parents=True)
        shutil.copy2(TOOLS / "export.py", tools / "export.py")
        shutil.copytree(TOOLS / "export_parts", tools / "export_parts", ignore=shutil.ignore_patterns("__pycache__"))
        table_path = tools / "export_parts/table.py"
        original = table_path.read_text()
        self.assertEqual(len(re.findall(r"^PREFIX_COUNT = [0-9]+$", original, re.M)), 1)
        table_path.write_text(re.sub(r"^PREFIX_COUNT = [0-9]+$", "PREFIX_COUNT = 1", original, flags=re.M))
        command = [sys.executable, "-B", str(tools / "export.py"), "--song-root", str(SONG),
                   "--output", str(self.output)]
        def run(*args):
            return subprocess.run(command + list(args), text=True, capture_output=True)
        self.assertEqual(run().returncode, 0)
        default = self.root / "proof/corpus/Slang/Generated/Table.lean"
        self.assertFalse(default.exists())
        self.assertTrue((self.root / "Foo/Chunk01.lean").is_file())
        self.assertEqual(run("--check").returncode, 0)
        leaf = self.root / "Foo/Chunk01.lean"
        leaf.unlink()
        self.assertIn("generated source binding differs", run("--check").stderr)
        self.assertEqual(run().returncode, 0)
        restored = leaf.read_text()
        first, second = self.groups[0][:2]
        for changed in (restored.replace("sourceByte 47", "sourceByte 48", 1),
                        restored.replace(first + second, second + first, 1)):
            self.assertNotEqual(changed, restored)
            leaf.write_text(changed)
            rejected = run("--check")
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("generated source binding differs", rejected.stderr)
            leaf.write_text(restored)
            self.assertEqual(run("--check").returncode, 0)
        extra = self.root / "Foo/Chunk99.lean"
        extra.write_text("extra\n")
        for args in ((), ("--check",)):
            rejected = run(*args)
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("unexpected table shard", rejected.stderr)
            self.assertTrue(extra.exists())
        extra.unlink()
        self.assertEqual(run("--check").returncode, 0)
        before = leaf.read_bytes()
        table_path.write_text(re.sub(r"^PREFIX_COUNT = [0-9]+$", "PREFIX_COUNT = 0", original, flags=re.M))
        rejected = run("--check")
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("unexpected table shard", rejected.stderr)
        self.assertEqual(leaf.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
