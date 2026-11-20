#!/usr/bin/env python3
"""Hash the tensor payloads of a GGUF and compare two files tensor by tensor.

Q2's same-quantisation clause rests on the NQ skeleton sharing every tensor the pack does not
replace — the 144 per-layer norms, `output_norm`, `token_embd` (Q4_K) and `output.weight` (Q6_K) —
byte for byte with the `Qwen3-8B-Q4_K_M.gguf` it is compared against. A skeleton built from a
different GGUF revision, or one whose norm tensors were re-serialised, would make the ratio a
cross-quantisation comparison. This reads both files' tensor tables and hashes each payload.

Usage:
    python3 tools/gguf-tensors.py a.gguf b.gguf          # shared / differing / only-in-one
    python3 tools/gguf-tensors.py --list a.gguf          # names, types, byte sizes, sha256
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

# ggml type -> (elements per block, bytes per block). Only the types these files use plus the common
# quantised ones; an unknown type is reported, not guessed.
TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}

# gguf metadata value types
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)
SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<B",
          U64: "<Q", I64: "<q", F64: "<d"}
SCALAR_SIZE = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4, BOOL: 1, U64: 8, I64: 8, F64: 8}


def read_exact(handle, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise SystemExit(f"truncated read: wanted {count} bytes, got {len(data)}")
    return data


def read_string(handle) -> str:
    (length,) = struct.unpack("<Q", read_exact(handle, 8))
    return read_exact(handle, length).decode("utf-8", "replace")


def skip_value(handle, value_type: int) -> None:
    if value_type in SCALAR_SIZE:
        handle.seek(SCALAR_SIZE[value_type], 1)
    elif value_type == STRING:
        (length,) = struct.unpack("<Q", read_exact(handle, 8))
        handle.seek(length, 1)
    elif value_type == ARRAY:
        elem_type, count = struct.unpack("<IQ", read_exact(handle, 12))
        if elem_type in SCALAR_SIZE:
            handle.seek(SCALAR_SIZE[elem_type] * count, 1)
        elif elem_type == STRING:
            for _ in range(count):
                (length,) = struct.unpack("<Q", read_exact(handle, 8))
                handle.seek(length, 1)
        else:
            raise SystemExit(f"array of value type {elem_type}")
    else:
        raise SystemExit(f"metadata value type {value_type}")


def parse(path: Path) -> tuple[dict, dict]:
    """(metadata, tensors) with tensors[name] = (ggml_type, dims, payload_offset, payload_bytes)."""
    meta: dict = {}
    with path.open("rb") as handle:
        magic = read_exact(handle, 4)
        if magic != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        version, n_tensors, n_meta = struct.unpack("<IQQ", read_exact(handle, 20))
        if version != 3:
            raise SystemExit(f"{path}: GGUF version {version}, expected 3")
        for _ in range(n_meta):
            key = read_string(handle)
            (value_type,) = struct.unpack("<I", read_exact(handle, 4))
            if value_type == STRING:
                meta[key] = read_string(handle)
            elif value_type == ARRAY:
                start = handle.tell()
                skip_value(handle, value_type)
                meta[key] = f"<array {handle.tell() - start} B>"
            elif value_type in SCALAR:
                raw = read_exact(handle, SCALAR_SIZE[value_type])
                meta[key] = struct.unpack(SCALAR[value_type], raw)[0]
            else:
                skip_value(handle, value_type)
                meta[key] = f"<type {value_type}>"
        infos = []
        for _ in range(n_tensors):
            name = read_string(handle)
            (n_dims,) = struct.unpack("<I", read_exact(handle, 4))
            dims = struct.unpack(f"<{n_dims}Q", read_exact(handle, 8 * n_dims))
            ggml_type, offset = struct.unpack("<IQ", read_exact(handle, 12))
            infos.append((name, ggml_type, dims, offset))
        alignment = int(meta.get("general.alignment", 32))
        data_start = handle.tell()
        data_start += (-data_start) % alignment
        tensors = {}
        for name, ggml_type, dims, offset in infos:
            if ggml_type not in TYPES:
                raise SystemExit(f"{path}: {name} has ggml type {ggml_type}; add it to TYPES")
            label, per_block, block_bytes = TYPES[ggml_type]
            elements = 1
            for d in dims:
                elements *= d
            if elements % per_block:
                raise SystemExit(f"{path}: {name} has {elements} elements, not a multiple of {per_block}")
            nbytes = elements // per_block * block_bytes
            tensors[name] = (label, dims, data_start + offset, nbytes)
    return meta, tensors


def digest(path: Path, offset: int, nbytes: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = handle.read(min(remaining, 1 << 24))
            if not chunk:
                raise SystemExit(f"{path}: truncated at {offset + nbytes - remaining}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--list", action="store_true", help="print one file's tensor table")
    ap.add_argument("--quantised", action="store_true", help="only tensors whose type is not F32/F16")
    args = ap.parse_args(argv)

    if args.list:
        for path in args.files:
            meta, tensors = parse(path)
            print(f"== {path}  {path.stat().st_size} B  {len(tensors)} tensors  "
                  f"alignment {meta.get('general.alignment', 32)}")
            for name in sorted(tensors, key=lambda n: tensors[n][2]):
                label, dims, offset, nbytes = tensors[name]
                if args.quantised and label in ("F32", "F16"):
                    continue
                print(f"{name:48s} {label:5s} {'x'.join(map(str, dims)):>16s} "
                      f"{nbytes:>12,d} B  @{offset:,d}")
        return 0

    if len(args.files) != 2:
        ap.error("two files, or --list")
    a, b = args.files
    meta_a, ta = parse(a)
    meta_b, tb = parse(b)
    shared = sorted(set(ta) & set(tb))
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))
    print(f"{a.name}: {len(ta)} tensors, {a.stat().st_size:,d} B")
    print(f"{b.name}: {len(tb)} tensors, {b.stat().st_size:,d} B")
    print(f"shared {len(shared)}, only in {a.name} {len(only_a)}, only in {b.name} {len(only_b)}")
    if only_a:
        print(f"  only-a: {only_a[:8]}{' …' if len(only_a) > 8 else ''}")
    if only_b:
        print(f"  only-b: {only_b[:8]}{' …' if len(only_b) > 8 else ''}")

    same, differ, meta_diff = [], [], []
    for name in shared:
        la, dims_a, off_a, len_a = ta[name]
        lb, dims_b, off_b, len_b = tb[name]
        if (la, dims_a, len_a) != (lb, dims_b, len_b):
            differ.append((name, f"table {la}{dims_a}/{len_a} vs {lb}{dims_b}/{len_b}"))
            continue
        ha = digest(a, off_a, len_a)
        hb = digest(b, off_b, len_b)
        if ha == hb:
            same.append((name, la, len_a, ha))
        else:
            differ.append((name, f"{ha[:16]} vs {hb[:16]}"))
    for key in ("general.architecture", "general.name", "general.file_type",
                "qwen3.block_count", "qwen3.embedding_length", "qwen3.feed_forward_length",
                "qwen3.attention.head_count", "qwen3.attention.head_count_kv",
                "qwen3.context_length", "tokenizer.ggml.model", "tokenizer.ggml.tokens"):
        va, vb = meta_a.get(key), meta_b.get(key)
        if va != vb:
            meta_diff.append((key, va if not isinstance(va, str) else va[:40],
                              vb if not isinstance(vb, str) else vb[:40]))

    print(f"payload identical: {len(same)}")
    print(f"payload differs:   {len(differ)}")
    for name, why in differ[:20]:
        print(f"  DIFFER {name}: {why}")
    if meta_diff:
        print(f"metadata differs:  {len(meta_diff)}")
        for key, va, vb in meta_diff:
            print(f"  {key}: {va!r} vs {vb!r}")
    else:
        print("metadata differs:  0 (architecture, shape, context and tokenizer keys)")

    by_type: dict[str, list[int]] = {}
    for name, label, nbytes, _ in same:
        by_type.setdefault(label, [0, 0])
        by_type[label][0] += 1
        by_type[label][1] += nbytes
    for label in sorted(by_type):
        count, nbytes = by_type[label]
        print(f"  identical {label:5s} {count:4d} tensors {nbytes:>15,d} B")
    return 0 if not differ and not only_a and not meta_diff else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
