#!/usr/bin/env python3
"""The checkpoint mechanism's own gate: ``tools/board/checkpoint.py self-test``.

Run it the way the gate is run::

    python3 tools/board/checkpoint.py self-test
    python3 -m unittest discover -s tools/board -p 'test_checkpoint.py'

Both are the same suite (``run_controls``); this module exists so ``unittest``
discovers it with the rest of ``tools/board`` and so the exit status is a test
runner's, not a bare script's. The controls are the mechanism's negative case
under the harness that runs it: ``checkpoint.py check`` itself, in a throwaway git
work tree, never the live tree. The per-cycle checkpoint step of
``.githooks/pre-commit`` was retired with the acceptance harness in ``c9a2d0c``, so
no hook runs that query and a control cannot assert on a commit's output. See
``tools/board/checkpoint.py`` for the mechanism and
``/home/socol/oc-tmp/escalation/CHECKPOINT-MECHANISM.md`` for the design.
"""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import checkpoint  # noqa: E402 -- sibling module; the suite is its own subject


class CheckpointControls(unittest.TestCase):
    def test_controls(self) -> None:
        """Baseline green, every seeded defect red by name, restored green.

        One test over the whole suite rather than one test per control: the
        controls share one throwaway work tree, and splitting them would let a
        later control's marker repair an earlier control's staged state.
        """

        class Capture:
            def __init__(self) -> None:
                self.lines: list[str] = []

            def write(self, text: str) -> int:
                self.lines.append(text)
                return len(text)

            def flush(self) -> None:
                return None

            @property
            def text(self) -> str:
                return "".join(self.lines)

        stream = Capture()
        code = checkpoint.run_controls(stream=stream)
        report = stream.text
        if code != 0:
            self.fail(f"control suite exit {code}\n{report}")
        self.assertNotIn("CONTROL FAIL", report, f"a control failed but exit was {code}\n{report}")
        for name in (
            "kv-quotes-survive",
            "headline-read",
            "unreadable-stream",
            "newest-readable",
            "no-marker-refuses",
            "stale-marker-refuses",
            "fresh-marker-passes",
            "staleness-returns",
        ):
            self.assertIn(f"CONTROL pass {name}", report, f"control {name} did not report\n{report}")


if __name__ == "__main__":
    unittest.main()