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
    "core.num.Usize.trailing_zeros",
    "I64.Insts.CoreConvertFromU8.from",
    "I64.Insts.CoreConvertFromU16.from",
    "I32.Insts.CoreConvertFromU8.from",
    "I128.Insts.CoreConvertFromU64.from",
    "Usize.Insts.CoreConvertTryFromU64TryFromIntError.try_from",
    "U64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from",
    "Usize.Insts.CoreConvertTryFromU32TryFromIntError.try_from",
    "Usize.Insts.CoreConvertTryFromI64TryFromIntError.try_from",
    "I64.Insts.CoreConvertTryFromUsizeTryFromIntError.try_from",
    "U32.Insts.CoreConvertTryFromI64TryFromIntError.try_from",
    "I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from",
    "core.num.I128.saturating_mul",
    "core.num.U64.saturating_mul",
    "core.num.I128.abs",
    "core.num.I8.from_ne_bytes",
    "core.core_arch.x86.__m128i",
    "core.core_arch.x86.sse2._mm_cmpeq_epi8",
    "core.core_arch.x86.sse2._mm_set1_epi8",
    "core.core_arch.x86.sse2._mm_loadu_si128",
    "core.core_arch.x86.sse2._mm_movemask_epi8",
    "core.mem.maybe_uninit.MaybeUninit",
    "core.mem.maybe_uninit.MaybeUninit.new",
    "core.mem.maybe_uninit.MaybeUninit.uninit",
    "core.mem.maybe_uninit.MaybeUninit.assume_init_read",
    "core.mem.maybe_uninit.MaybeUninit.assume_init_drop",
    "core.mem.needs_drop",
    "core.num.U16.trailing_zeros",
    "core.num.Usize.saturating_mul",
    "core.num.Usize.next_power_of_two",
    "core.ptr.const_ptr.RawPtrConstT.cast",
    "core.ptr.const_ptr.RawPtrConstT.add",
    "core.slice.Slice.get_unchecked_mut",
    "core.slice.Slice.as_ptr",
    "alloc.vec.Vec.into_boxed_slice",
    "core.core_arch.x86.__m256i",
    "core.core_arch.x86.__m256d",
    "core.core_arch.x86.avx._mm256_loadu_si256",
    "core.core_arch.x86.avx._mm256_movemask_pd",
    "core.core_arch.x86.avx._mm256_set1_epi64x",
    "core.core_arch.x86.avx._mm256_castsi256_pd",
    "core.core_arch.x86.avx2._mm256_cmpeq_epi64",
    "std_detect.detect.arch.x86.__is_feature_detected.avx2",
    "core.mem.maybe_uninit.MaybeUninit.assume_init_ref",
    "core.mem.take",
    "core.mem.drop",
    "core.num.U32.trailing_zeros",
    "core.option.Option.as_mut",
    "BoxSlice.Insts.CoreDefaultDefault.default",
    "alloc.vec.Vec.set_len",
    "alloc.vec.Vec.pop",
    "alloc.vec.Vec.is_empty",
    "alloc.vec.Vec.Insts.CoreDefaultDefault.default",
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
