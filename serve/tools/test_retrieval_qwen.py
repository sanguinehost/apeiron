#!/usr/bin/env python3
"""Small unit tests for the lexical retrieval lane. No model is loaded."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_qwen import Retriever, build_messages, chatml, split_chunks


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (
            "The archive records routine shelf checks and ordinary transfers. "
            "The archive records routine shelf checks and ordinary transfers. "
            "Project Zephyr's activation code is QX-4711. The Zephyr team keeps the code in vault seven. "
            "The Helios calibration value is 8842-Kappa. Helios operators record it before launch. "
            "The archive records routine shelf checks and ordinary transfers."
        )
        self.chunks = split_chunks(self.text, 18, 4)
        self.retriever = Retriever(self.chunks)

    def test_rare_key_is_first(self) -> None:
        hits = self.retriever.search("What is Project Zephyr's activation code?", 2, 0.25)
        self.assertTrue(hits)
        self.assertIn("QX-4711", hits[0]["text"])

    def test_second_needle(self) -> None:
        hits = self.retriever.search("What is the Helios calibration value?", 2, 0.25)
        self.assertTrue(hits)
        self.assertIn("8842-Kappa", hits[0]["text"])

    def test_absent_terms_return_empty(self) -> None:
        self.assertEqual(self.retriever.search("xyzzy plugh quux", 2, 0.25), [])

    def test_prompt_is_chatml(self) -> None:
        hits = self.retriever.search("What is Project Zephyr's activation code?", 1, 0.25)
        prompt = chatml(build_messages("What is Project Zephyr's activation code?", hits))
        self.assertIn("<|im_start|>system\n", prompt)
        self.assertIn("QX-4711", prompt)
        self.assertTrue(prompt.rstrip().endswith("<|im_start|>assistant"))


if __name__ == "__main__":
    unittest.main()
