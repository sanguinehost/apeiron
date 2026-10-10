#!/usr/bin/env bash
# Build the crate, then build the book and compile every Rust block in it against the crate.
#
#   tools/book-test.sh
#
# Needs `mdbook` on PATH (cargo install mdbook). Exits non-zero if the crate, the book, or any
# example fails. Also runs tools/book-check.py (names and numbers the book states).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

command -v mdbook >/dev/null || { echo "book-test: mdbook not found (cargo install mdbook)" >&2; exit 127; }

cargo build --lib --locked
python3 tools/book-check.py
mdbook build book
mdbook test book -L target/debug/deps
echo "book-test: PASS"
