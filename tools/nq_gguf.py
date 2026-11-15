#!/usr/bin/env python3
"""GGUF directory reading and streaming writing, shared by the NanoQuant converter.

The engine reads a GGUF skeleton plus a pack; this module is the Python side of that pair. It
parses a source image's header without touching the data section, then writes a new image whose
frozen payloads stream from the source and whose generated payloads come from a callback, so the
source and the output are never both resident.
"""

import os
import struct

GGUF_DEFAULT_ALIGN = 32
GGML_F32, GGML_F16, GGML_BF16, GGML_Q8_0 = 0, 1, 30, 8
GGML_I32 = 26
# The `K`-quant family, for reading comparison images (`llama.cpp` `ggml-common.h` block layouts).
GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_K = 10, 11, 12, 13, 14, 15

# Types with a fixed block layout, as `(elements, bytes)` per block.
GGML_BLOCKS = {
    GGML_Q8_0: (32, 34),
    GGML_I32: (1, 4),
    GGML_Q2_K: (256, 84),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    GGML_Q8_K: (256, 292),
}

GGML_NAMES = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_Q8_K: "Q8_K",
    GGML_I32: "I32",
    GGML_BF16: "BF16",
}

COPY_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# GGUF reading


def _rstr(b, o):
    (n,) = struct.unpack_from("<Q", b, o)
    o += 8
    return b[o : o + n].decode("utf8"), o + n


def _skip_value(b, o, t):
    fixed = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    if t in fixed:
        return o + fixed[t]
    if t == 8:
        return _rstr(b, o)[1]
    if t == 9:
        (at,) = struct.unpack_from("<I", b, o)
        o += 4
        (n,) = struct.unpack_from("<Q", b, o)
        o += 8
        for _ in range(n):
            o = _skip_value(b, o, at)
        return o
    raise ValueError(f"gguf value type {t}")


def _parse_head(path, head):
    if head[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", head, 4)
    if version != 3:
        raise ValueError(f"{path}: GGUF version {version}")
    o = 24
    meta_start = o
    align = GGUF_DEFAULT_ALIGN
    for _ in range(n_kv):
        key, o = _rstr(head, o)
        (t,) = struct.unpack_from("<I", head, o)
        o += 4
        if key == "general.alignment":
            if t == 4:
                (align,) = struct.unpack_from("<I", head, o)
            elif t == 10:
                (align,) = struct.unpack_from("<Q", head, o)
        o = _skip_value(head, o, t)
    meta = head[meta_start:o]
    tensors = []
    for _ in range(n_tensors):
        name, o = _rstr(head, o)
        (nd,) = struct.unpack_from("<I", head, o)
        o += 4
        dims = list(struct.unpack_from("<%dQ" % nd, head, o))
        o += 8 * nd
        (t,) = struct.unpack_from("<I", head, o)
        o += 4
        (off,) = struct.unpack_from("<Q", head, o)
        o += 8
        tensors.append({"name": name, "dims": dims, "type": t, "off": off})
    header_end = o
    data_off = (header_end + align - 1) // align * align
    for ten in tensors:
        ten["abs"] = data_off + ten["off"]
        ten["nbytes"] = ggml_nbytes(ten["type"], ten["dims"])
    if data_off > len(head):
        raise struct.error("header extends past the bytes read")
    return {
        "version": version,
        "meta": meta,
        "tensors": tensors,
        "align": align,
        "data_off": data_off,
        "path": path,
    }


def parse_gguf(path):
    """Read the header and directory, never the tensor data."""
    limit = 64 << 20
    size = os.path.getsize(path)
    while True:
        with open(path, "rb") as f:
            head = f.read(min(limit, size))
        try:
            return _parse_head(path, head)
        except (struct.error, IndexError):
            if limit >= size:
                raise
            limit *= 2


def ggml_nbytes(t, dims):
    n = 1
    for d in dims:
        n *= d
    if t == GGML_F32:
        return 4 * n
    if t in (GGML_F16, GGML_BF16):
        return 2 * n
    if t in GGML_BLOCKS:
        qk, per_block = GGML_BLOCKS[t]
        if n % qk:
            raise ValueError(f"gguf type {t}: {n} elements is not a block multiple of {qk}")
        return n // qk * per_block
    raise ValueError(f"gguf type {t} ({GGML_NAMES.get(t, '?')}): no block layout in this exporter")


# ---------------------------------------------------------------------------
# GGUF writing


def _align_up(n):
    return (n + GGUF_DEFAULT_ALIGN - 1) // GGUF_DEFAULT_ALIGN * GGUF_DEFAULT_ALIGN


def _count_kv(meta):
    o = 0
    n = 0
    while o < len(meta):
        _, o = _rstr(meta, o)
        (t,) = struct.unpack_from("<I", meta, o)
        o += 4
        o = _skip_value(meta, o, t)
        n += 1
    if o != len(meta):
        raise ValueError("metadata does not parse to its declared extent")
    return n


def _copy_payload(src, ten, out):
    with open(src, "rb") as f:
        f.seek(ten["abs"])
        left = ten["nbytes"]
        while left:
            chunk = f.read(min(left, COPY_CHUNK))
            if not chunk:
                raise ValueError(f"{ten['name']}: source ends after {ten['nbytes'] - left} B")
            out.write(chunk)
            left -= len(chunk)


def build_gguf_stream(path, meta, tensors):
    """Write a GGUF with each payload produced by `t["write"](file)`.

    `tensors` entries carry `name`, `dims`, `type`, `nbytes` and the write callback; the
    directory is laid out first and the callbacks then stream into the data section.
    """
    offsets = []
    running = 0
    for t in tensors:
        if t["nbytes"] != ggml_nbytes(t["type"], t["dims"]):
            raise ValueError(f"{t['name']}: {t['nbytes']} B for {t['dims']} type {t['type']}")
        offsets.append(running)
        running = _align_up(running + t["nbytes"])
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<IQQ", 3, len(tensors), _count_kv(meta))
    header += meta
    for t, off in zip(tensors, offsets):
        header += struct.pack("<Q", len(t["name"]))
        header += t["name"].encode()
        header += struct.pack("<I", len(t["dims"]))
        header += struct.pack("<%dQ" % len(t["dims"]), *t["dims"])
        header += struct.pack("<IQ", t["type"], off)
    header += b"\0" * (_align_up(len(header)) - len(header))
    data_off = len(header)
    with open(path, "wb") as out:
        out.write(header)
        at = data_off
        for t, off in zip(tensors, offsets):
            want = data_off + off
            if want > at:
                out.write(b"\0" * (want - at))
                at = want
            t["write"](out)
            at += t["nbytes"]
            pad = _align_up(at) - at
            out.write(b"\0" * pad)
            at += pad
    total = data_off + running
    if os.path.getsize(path) != total:
        raise ValueError(f"{path}: wrote {os.path.getsize(path)} B, want {total}")
    return total


def frozen_entry(src, ten):
    """The build entry that copies `ten` from `src` unchanged."""
    return {
        "name": ten["name"],
        "dims": ten["dims"],
        "type": ten["type"],
        "nbytes": ten["nbytes"],
        "write": lambda out, ten=ten: _copy_payload(src["path"], ten, out),
    }
