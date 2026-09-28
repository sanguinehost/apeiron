#!/usr/bin/env python3
"""Remove Aeneas `axiom` stubs for externals the patched Lean model defines.

The pinned Aeneas emits one `axiom` per external it does not model. The model patch
(`patches/aeneas-lean-model.patch`) defines the externals in the allowlist below in the Aeneas Lean
library, so the generated body resolves to the model instead. Removing the stub is what makes the
generated file resolve to the model.

Any `axiom` NOT in the allowlist is a real unmodeled external: it is left in place for `regen.sh`'s
fail-closed audit to reject. This script therefore never weakens the audit — it only drops the stubs
the model now supplies.

Usage: strip-modeled-axioms.py FILE
"""
import re
import sys

ALLOW = {
    "alloc.collections.TryReserveError",
    "alloc.vec.Vec.capacity",
    "alloc.vec.Vec.try_reserve",
    "alloc.vec.Vec.try_reserve_exact",
    "alloc.vec.Vec.truncate",
    "alloc.vec.Vec.clear",
    "core.option.Option.as_ref",
    "core.option.Option.Insts.CoreOpsTry_traitTry.branch",
    "core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual",
    "core.option.Option.and_then",
    "core.result.Result.is_err",
    "core.result.Result.ok",
    "core.num.I64.checked_shl",
    "core.num.Usize.div_ceil",
    "I64.Insts.CoreConvertFromU8.from",
    "I64.Insts.CoreConvertFromU16.from",
    "I32.Insts.CoreConvertFromU8.from",
}

AXIOM_LINE = re.compile(r"^axiom\s+(\S+)")
AXIOM_BARE = re.compile(r"^axiom\s*$")
CONT = re.compile(r"^\s+(\S+)")


def axiom_name(lines, i):
    m = AXIOM_LINE.match(lines[i].strip())
    if m:
        return m.group(1)
    if AXIOM_BARE.match(lines[i].strip()) and i + 1 < len(lines):
        n = CONT.match(lines[i + 1])
        return n.group(1) if n else None
    return None


def main(path):
    lines = open(path).read().split("\n")
    out = []
    removed = []
    i = 0
    while i < len(lines):
        name = axiom_name(lines, i)
        if name in ALLOW:
            removed.append(name)
            j = len(out) - 1
            while j >= 0:
                t = out[j]
                if t.strip() == "":
                    break
                if t.startswith(("--", "/--", "/-", "@[")) or t.startswith((" ", "\t")):
                    out.pop()
                    j -= 1
                else:
                    break
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            continue
        out.append(lines[i])
        i += 1
    open(path, "w").write("\n".join(out))
    removed.sort()
    print(f"strip-modeled-axioms: removed {len(removed)} model axiom(s): {removed}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: strip-modeled-axioms.py FILE", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
