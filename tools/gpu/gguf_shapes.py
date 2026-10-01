#!/usr/bin/env python3
"""Derive the Q8_0 prefill projection shapes from a real GGUF tensor table.

Dependency-free GGUF v3 header/metadata/tensor-info parser. Reports the
attention/FFN dimensions the Apeiron Q8 kernel needs (embedding width, output
feature width, query/key-value widths, head counts and head dimensions, vocab,
block count) and the per-tensor GGML dims of every Q8_0 projection weight,
mapped to the Apeiron `(M, blocks)` projection shape with `blocks = ne[0]/32`
(ne[0] is the GGML contraction width `K`).

`--check` asserts the derived dimensions are internally consistent with the
metadata and that an EXACT expected tensor set (names, dtypes and dims) is
present for the named architecture. A guessed shape or an absent tensor is a
FAIL and exits nonzero. `--negative-control` injects a fault and requires the
checker to reject it: the run exits nonzero with the attributable diagnostic
when the fault is caught, and exits 2 if the checker wrongly accepts it.

Scope note. `qwen35` is a hybrid stack: `qwen35.full_attention_interval = 4`
marks every fourth block as full attention and the trailing
`nextn_predict_layers` blocks as the multi-token-prediction block, which is also
full attention; the remaining blocks are linear-attention (SSM) blocks whose
projections are `attn_qkv`, `attn_gate`, `ssm_out`, `ssm_alpha`, `ssm_beta`.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

# ggml/include/ggml.h:390-416
GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    26: "I32",
    27: "I64",
    28: "F64",
    30: "BF16",
}
GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 8

# ggml-common.h: QK8_0 == 32 values per Q8_0 block; one f16 scale + 32 int8 codes.
QK8_0 = 32
Q8_0_BLOCK_BYTES = 34

# gguf.h: metadata value type ids.
GGUF_META_TYPES = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

_SCALAR = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}


class GGUFError(ValueError):
    """Malformed GGUF input; carries the byte offset where parsing stopped."""

    def __init__(self, message: str, offset: int) -> None:
        super().__init__(f"{message} at byte {offset}")
        self.offset = offset


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def dtype(self) -> str:
        return GGML_TYPES.get(self.type_id, f"type_{self.type_id}")


class Reader:
    """Bounds-checked little-endian cursor over the in-memory GGUF header."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise GGUFError(f"truncated read of {n} bytes", self.pos)
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        raw = self.take(n)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("non-UTF8 metadata string", self.pos) from exc

    def value(self, type_id: int) -> object:
        if type_id == 8:
            return self.string()
        if type_id == 9:
            element = self.u32()
            count = self.u64()
            if element == 9:
                raise GGUFError("nested arrays are not valid GGUF", self.pos)
            if count > (1 << 28):
                raise GGUFError(f"implausible array count {count}", self.pos)
            return [self.value(element) for _ in range(count)]
        if type_id not in _SCALAR:
            raise GGUFError(f"unknown metadata value type {type_id}", self.pos)
        fmt, size = _SCALAR[type_id]
        return struct.unpack(fmt, self.take(size))[0]


@dataclass
class GGUF:
    path: Path
    sha256: str
    version: int
    metadata: dict[str, object]
    tensors: list[Tensor]

    def meta_int(self, key: str) -> int:
        value = self.metadata.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise GGUFError(f"metadata {key!r} missing or not an integer", 0)
        return value


def parse_gguf(path: Path) -> GGUF:
    data = path.read_bytes()
    if len(data) < 24:
        raise GGUFError("file shorter than a GGUF header", 0)
    digest = hashlib.sha256(data).hexdigest()
    rd = Reader(data)
    magic = rd.take(4)
    if magic != b"GGUF":
        raise GGUFError(f"bad magic {magic!r}", 0)
    version = rd.u32()
    if version != 3:
        raise GGUFError(f"unsupported GGUF version {version}", 4)
    tensor_count = rd.u64()
    meta_count = rd.u64()

    metadata: dict[str, object] = {}
    for _ in range(meta_count):
        key = rd.string()
        type_id = rd.u32()
        if type_id not in GGUF_META_TYPES:
            raise GGUFError(f"unknown metadata value type {type_id}", rd.pos)
        metadata[key] = rd.value(type_id)

    tensors: list[Tensor] = []
    for _ in range(tensor_count):
        name = rd.string()
        n_dims = rd.u32()
        if n_dims > 8:
            raise GGUFError(f"tensor {name!r} has {n_dims} dims", rd.pos)
        dims = tuple(rd.u64() for _ in range(n_dims))
        type_id = rd.u32()
        offset = rd.u64()
        tensors.append(Tensor(name, dims, type_id, offset))

    return GGUF(path, digest, version, metadata, tensors)


# --------------------------------------------------------------------------
# Qwen3.5 derivation and expected tensor inventory
# --------------------------------------------------------------------------

ARCH = "qwen35"


@dataclass
class Dims:
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: int
    n_head_dim: int
    n_head_dim_kv: int
    vocab: int
    n_block: int
    n_ctx: int
    attn_interval: int
    nextn_predict: int
    ssm_inner: int
    ssm_group: int
    ssm_state: int
    ssm_conv: int

    @property
    def q_width(self) -> int:
        return self.n_head * self.n_head_dim

    @property
    def kv_width(self) -> int:
        return self.n_head_kv * self.n_head_dim_kv

    @property
    def q_proj_out(self) -> int:
        # Qwen3.5 gated attention fuses the per-head output gate into `attn_q`:
        # the projection emits [query | gate], 2 * q_width wide.
        return 2 * self.q_width

    @property
    def attn_layers(self) -> list[int]:
        last = self.n_block - self.nextn_predict
        return [
            layer
            for layer in range(self.n_block)
            if (layer + 1) % self.attn_interval == 0 or layer >= last
        ]

    @property
    def ssm_layers(self) -> list[int]:
        return [layer for layer in range(self.n_block) if layer not in set(self.attn_layers)]


def _require_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise GGUFError(f"metadata {key!r} missing or not an integer", 0)
    return value


def derive(gguf: GGUF) -> Dims:
    arch = gguf.metadata.get("general.architecture")
    if arch != ARCH:
        raise GGUFError(f"architecture {arch!r} != {ARCH!r}", 0)
    p = f"{ARCH}."
    by_name = {t.name: t for t in gguf.tensors}
    token_embd = by_name.get("token_embd.weight")
    if token_embd is None:
        raise GGUFError("token_embd.weight absent; vocab is not derivable", 0)
    if len(token_embd.dims) != 2:
        raise GGUFError(f"token_embd.weight rank {len(token_embd.dims)} != 2", 0)

    n_embd = _require_int(gguf.metadata, p + "embedding_length")
    vocab = token_embd.dims[1]
    tokenizer_tokens = gguf.metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokenizer_tokens, list) or len(tokenizer_tokens) != vocab:
        raise GGUFError(
            f"tokenizer.ggml.tokens count {len(tokenizer_tokens) if isinstance(tokenizer_tokens, list) else None} "
            f"!= token_embd vocab {vocab}",
            0,
        )

    return Dims(
        n_embd=n_embd,
        n_ff=_require_int(gguf.metadata, p + "feed_forward_length"),
        n_head=_require_int(gguf.metadata, p + "attention.head_count"),
        n_head_kv=_require_int(gguf.metadata, p + "attention.head_count_kv"),
        n_head_dim=_require_int(gguf.metadata, p + "attention.key_length"),
        n_head_dim_kv=_require_int(gguf.metadata, p + "attention.value_length"),
        vocab=vocab,
        n_block=_require_int(gguf.metadata, p + "block_count"),
        n_ctx=_require_int(gguf.metadata, p + "context_length"),
        attn_interval=_require_int(gguf.metadata, p + "full_attention_interval"),
        nextn_predict=_require_int(gguf.metadata, p + "nextn_predict_layers"),
        ssm_inner=_require_int(gguf.metadata, p + "ssm.inner_size"),
        ssm_group=_require_int(gguf.metadata, p + "ssm.group_count"),
        ssm_state=_require_int(gguf.metadata, p + "ssm.state_size"),
        ssm_conv=_require_int(gguf.metadata, p + "ssm.conv_kernel"),
    )


# Expected tensor inventory: name -> (type_name, dims). Every projection weight
# is Q8_0; norms and the small SSM scalars/conv are F32.
def expected_tensors(d: Dims) -> dict[str, tuple[str, tuple[int, ...]]]:
    exp: dict[str, tuple[str, tuple[int, ...]]] = {
        "token_embd.weight": ("Q8_0", (d.n_embd, d.vocab)),
        "output_norm.weight": ("F32", (d.n_embd,)),
    }
    attn_set = set(d.attn_layers)
    for layer in range(d.n_block):
        b = f"blk.{layer}."
        exp[b + "attn_norm.weight"] = ("F32", (d.n_embd,))
        exp[b + "post_attention_norm.weight"] = ("F32", (d.n_embd,))
        exp[b + "ffn_gate.weight"] = ("Q8_0", (d.n_embd, d.n_ff))
        exp[b + "ffn_up.weight"] = ("Q8_0", (d.n_embd, d.n_ff))
        exp[b + "ffn_down.weight"] = ("Q8_0", (d.n_ff, d.n_embd))
        if layer in attn_set:
            exp[b + "attn_q.weight"] = ("Q8_0", (d.n_embd, d.q_proj_out))
            exp[b + "attn_k.weight"] = ("Q8_0", (d.n_embd, d.kv_width))
            exp[b + "attn_v.weight"] = ("Q8_0", (d.n_embd, d.kv_width))
            exp[b + "attn_output.weight"] = ("Q8_0", (d.q_width, d.n_embd))
            exp[b + "attn_q_norm.weight"] = ("F32", (d.n_head_dim,))
            exp[b + "attn_k_norm.weight"] = ("F32", (d.n_head_dim_kv,))
        else:
            exp[b + "attn_qkv.weight"] = ("Q8_0", (d.n_embd, 3 * d.ssm_inner))
            exp[b + "attn_gate.weight"] = ("Q8_0", (d.n_embd, d.ssm_inner))
            exp[b + "ssm_out.weight"] = ("Q8_0", (d.ssm_inner, d.n_embd))
            exp[b + "ssm_alpha.weight"] = ("Q8_0", (d.n_embd, d.ssm_group))
            exp[b + "ssm_beta.weight"] = ("Q8_0", (d.n_embd, d.ssm_group))
            exp[b + "ssm_a"] = ("F32", (d.ssm_group,))
            exp[b + "ssm_conv1d.weight"] = ("F32", (d.ssm_conv, 3 * d.ssm_inner))
            exp[b + "ssm_dt.bias"] = ("F32", (d.ssm_group,))
            exp[b + "ssm_norm.weight"] = ("F32", (d.ssm_state,))
        if layer >= d.n_block - d.nextn_predict:
            exp[b + "nextn.eh_proj.weight"] = ("Q8_0", (d.q_width, d.n_embd))
            exp[b + "nextn.enorm.weight"] = ("F32", (d.n_embd,))
            exp[b + "nextn.hnorm.weight"] = ("F32", (d.n_embd,))
            exp[b + "nextn.shared_head_norm.weight"] = ("F32", (d.n_embd,))
    return exp


# Prefill projection table for the Apeiron Q8 kernel: (label, K=ne0, M=ne1).
def projections(d: Dims) -> list[tuple[str, int, int]]:
    return [
        ("attn_q.weight", d.n_embd, d.q_proj_out),
        ("attn_k.weight", d.n_embd, d.kv_width),
        ("attn_v.weight", d.n_embd, d.kv_width),
        ("attn_output.weight", d.q_width, d.n_embd),
        ("ffn_gate.weight", d.n_embd, d.n_ff),
        ("ffn_up.weight", d.n_embd, d.n_ff),
        ("ffn_down.weight", d.n_ff, d.n_embd),
        ("attn_qkv.weight (SSM)", d.n_embd, 3 * d.ssm_inner),
        ("attn_gate.weight (SSM)", d.n_embd, d.ssm_inner),
        ("ssm_out.weight", d.ssm_inner, d.n_embd),
        ("ssm_alpha.weight", d.n_embd, d.ssm_group),
        ("ssm_beta.weight", d.n_embd, d.ssm_group),
    ]


def check(gguf: GGUF, dims: Dims, *, fault: str | None = None) -> list[str]:
    problems: list[str] = []

    def require(cond: bool, message: str) -> None:
        if not cond:
            problems.append(message)

    d = dims
    if fault == "dims":
        d = Dims(**{**dims.__dict__, "n_ff": dims.n_ff + QK8_0})
    require(d.n_embd % QK8_0 == 0, f"n_embd {d.n_embd} is not a multiple of {QK8_0}")
    require(d.n_ff % QK8_0 == 0, f"n_ff {d.n_ff} is not a multiple of {QK8_0}")
    require(d.n_head * d.n_head_dim == d.q_width, "n_head*n_head_dim != q_width")
    require(d.n_head_kv * d.n_head_dim_kv == d.kv_width, "n_head_kv*n_head_dim != kv_width")
    require(
        d.n_embd % d.n_head == 0,
        f"n_embd {d.n_embd} is not divisible by n_head {d.n_head}",
    )

    exp = expected_tensors(d)
    if fault == "absent":
        # Expect a tensor the file does not carry: the checker must report it absent.
        exp["blk.0.attn_qkv_absent.weight"] = ("Q8_0", (d.n_embd, 3 * d.ssm_inner))
    if fault == "head":
        # Corrupt one expected dim: the checker must report the mismatch.
        exp["blk.0.ffn_gate.weight"] = ("Q8_0", (d.n_embd, d.n_ff + QK8_0))

    by_name = {t.name: t for t in gguf.tensors}
    for name in exp:
        if name not in by_name:
            problems.append(f"absent tensor {name}")
    exp_names = set(exp)
    for t in gguf.tensors:
        if t.name not in exp_names:
            problems.append(f"unexpected tensor {t.name}")

    for name, (want_type, want_dims) in exp.items():
        t = by_name.get(name)
        if t is None:
            continue
        require(t.dtype == want_type, f"{name}: dtype {t.dtype} != {want_type}")
        require(t.dims == want_dims, f"{name}: dims {t.dims} != {want_dims}")

    # Cross-tensor identities the projection table relies on.
    def ne1(name: str) -> int | None:
        t = by_name.get(name)
        return t.dims[1] if t is not None and len(t.dims) == 2 else None

    for layer in d.attn_layers:
        b = f"blk.{layer}."
        require(
            ne1(b + "attn_q.weight") == d.q_proj_out,
            f"{b}attn_q.weight: ne[1] {ne1(b + 'attn_q.weight')} != fused q+gate {d.q_proj_out}",
        )
        require(
            ne1(b + "attn_output.weight") == d.n_embd,
            f"{b}attn_output.weight: ne[1] {ne1(b + 'attn_output.weight')} != n_embd {d.n_embd}",
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path, help="path to a .gguf file")
    ap.add_argument("--check", action="store_true", help="assert consistency; nonzero on any FAIL")
    ap.add_argument(
        "--negative-control",
        choices=("absent", "head", "dims"),
        default=None,
        help="inject a fault; exits nonzero only when the checker rejects it (exits 2 if it accepts)",
    )
    args = ap.parse_args(argv)

    if args.negative_control and not args.check:
        print("FAIL: --negative-control requires --check", file=sys.stderr)
        return 2

    try:
        gguf = parse_gguf(args.model)
        dims = derive(gguf)
    except GGUFError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    print(f"model      {gguf.path}")
    print(f"sha256     {gguf.sha256}")
    print(f"gguf       v{gguf.version}  tensors={len(gguf.tensors)}  metadata={len(gguf.metadata)}")
    print(f"arch       {gguf.metadata.get('general.architecture')}  name={gguf.metadata.get('general.name')}")
    print()
    print(f"n_embd     {dims.n_embd}")
    print(f"n_ff       {dims.n_ff}")
    print(f"n_head     {dims.n_head}  n_head_kv {dims.n_head_kv}  head_dim {dims.n_head_dim}")
    print(f"q_width    {dims.q_width}  kv_width {dims.kv_width}  attn_q out {dims.q_proj_out}")
    print(f"vocab      {dims.vocab}  n_block {dims.n_block}  n_ctx {dims.n_ctx}")
    print(f"layers     attn={dims.attn_layers} ssm={len(dims.ssm_layers)} ssm_inner={dims.ssm_inner}")
    print()
    print("projection                 K=ne0  M=ne1  blocks=K/32")
    for label, ne0, ne1 in projections(dims):
        print(f"{label:<27}{ne0:<7}{ne1:<7}{ne0 // QK8_0}")
    print()
    print("actual Q8_0 projection tensors per class (all layers share the shape):")
    table = {t.name: t for t in gguf.tensors}
    for layer in (dims.attn_layers[0], dims.ssm_layers[0]):
        for label, _, _ in projections(dims):
            suffix = label.split(" ")[0]
            t = table.get(f"blk.{layer}.{suffix}")
            if t is not None:
                print(f"  blk.{layer}.{suffix:<22} ne={t.dims}  {t.dtype}  K/32={t.dims[0] // QK8_0}")
    print()
    print(f"tensors ({len(gguf.tensors)}):")
    for t in gguf.tensors:
        dims_str = "x".join(str(x) for x in reversed(t.dims))
        print(f"  {t.name:<30} {dims_str:<14} {t.dtype}")

    if not args.check:
        return 0

    problems = check(gguf, dims, fault=args.negative_control)
    if args.negative_control:
        print()
        if not problems:
            print(
                f"FAIL: negative control {args.negative_control!r} did not fire; the checker accepted the injected fault",
                file=sys.stderr,
            )
            return 2
        for problem in problems:
            print(f"negative-control FAIL: {problem}", file=sys.stderr)
        print(
            f"negative control {args.negative_control!r} rejected the injected fault "
            f"({len(problems)} problem(s)); exit nonzero as required",
            file=sys.stderr,
        )
        return 1

    if problems:
        print()
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        print(f"check failed: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print()
    print(f"check ok: {len(gguf.tensors)} tensors, {len(expected_tensors(dims))} expected, derived dimensions consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
