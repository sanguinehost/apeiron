#!/usr/bin/env python3
"""Compatibility entry point: admission and launch decisions execute only in Rust."""
import os
from pathlib import Path
import sys

runner = str(Path(__file__).resolve().with_name('run'))
os.execv(runner, [runner, *sys.argv[1:]])
