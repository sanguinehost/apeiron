#!/usr/bin/env python3
"""Static instruction histogram for named symbols in a release binary.

Usage: tools/asm-hist.py <binary> <symbol-substring> [<symbol-substring> ...]

Prints one block per matching symbol: byte size, total instructions, and the
counts that matter for the one-path analysis (data movement, compares,
branches, calls, vector arithmetic). The full disassembly is written to
/tmp/opencode/asm-<n>.txt. Requires binutils `nm` and `objdump`.
"""
import re
import subprocess
import sys

BIN = sys.argv[1]
WANT = sys.argv[2:]

nm = subprocess.run(["nm", "-S", "-C", BIN], capture_output=True, text=True, check=True).stdout
syms = []
for line in nm.splitlines():
    m = re.match(r"^([0-9a-f]+)\s+([0-9a-f]+)\s+([tT])\s+(.*)$", line)
    if not m:
        continue
    start, size, kind, name = int(m.group(1), 16), int(m.group(2), 16), m.group(3), m.group(4)
    if any(w in name for w in WANT):
        syms.append((start, size, name))

if not syms:
    print("no matching symbols", file=sys.stderr)
    sys.exit(1)

for idx, (start, size, name) in enumerate(syms):
    dis = subprocess.run(
        [
            "objdump", "-d", "--demangle", "--no-show-raw-insn",
            f"--start-address={start}", f"--stop-address={start + size}", BIN,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    path = f"/tmp/opencode/asm-{idx}.txt"
    with open(path, "w") as f:
        f.write(dis)
    mnemonics = []
    for line in dis.splitlines():
        m = re.match(r"^\s+[0-9a-f]+:\s+(?:\{vex\}\s+)?([a-z0-9.]+)", line)
        if m:
            mnemonics.append(m.group(1))
    total = len(mnemonics)

    def n(*names):
        return sum(mnemonics.count(x) for x in names)

    def pref(p):
        return sum(1 for x in mnemonics if x.startswith(p))

    print(f"\n=== {name}")
    print(f"    bytes {size}  instructions {total}  asm {path}")
    print(f"    mov {pref('mov')}  lea {n('lea')}  cmp/test {n('cmp', 'test')}  "
          f"jcc {pref('j')}  call {n('call')}  ret {n('ret')}")
    print(f"    add/sub/inc/dec {n('add', 'sub', 'inc', 'dec', 'adc', 'sbb')}  "
          f"imul/mul {n('imul', 'mul')}  and/or/xor {n('and', 'or', 'xor')}  "
          f"shl/shr/sar {n('shl', 'shr', 'sar', 'sal')}")
    print(f"    vector {pref('v')}  vpdpbusd {n('vpdpbusd')}  "
          f"vpmaddubsw {n('vpmaddubsw')}  vpmaddwd {n('vpmaddwd')}  "
          f"vphaddd {n('vphaddd')}  vpshufb {n('vpshufb')}  "
          f"vmov* {pref('vmov')}")
    top = {}
    for x in mnemonics:
        top[x] = top.get(x, 0) + 1
    ranked = sorted(top.items(), key=lambda kv: (-kv[1], kv[0]))[:14]
    print("    top: " + "  ".join(f"{k}={v}" for k, v in ranked))
