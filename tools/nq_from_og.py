"""Convert a `SamsungLabs/NanoQuant` `.pt` checkpoint into an `APNQPK01` pack.

The reference writes `get_compressed_state_dict` (`src/nanoquant/utils/load_utils.py:216`): every
`NanoQuantLinear` contributes `scale_pre` `(1, in)`, `scale_post` `(1, out)`, an optional
`scale_mid` `(1, rank)` and the two packed sign matrices `U_packed` `(out, ⌈rank/32⌉)`,
`V_packed` `(rank, ⌈in/32⌉)`, each with a `_shape` word. `pack_binary_gemv`
(`src/nanoquant/kernel/utils.py:48`) packs least-significant bit first with `+1 → 0`, `-1 → 1`.

This tool writes the pack only. The dense tensors (embedding, head, norms) come from the skeleton
GGUF that `tools/nq_convert.py` writes from the same base model:

    python3 tools/nq_from_og.py --checkpoint qmodel.pt --out model.nqp --og-src /workspace/NanoQuant/src
    target/release/examples/ppl --model skeleton.gguf --pack model.nqp ...

The bit convention needs no flip, but not because the two agree. `pack_binary_gemv`
(`src/nanoquant/kernel/utils.py:48`) stores `+1 → 0`, `-1 → 1`; this crate's `U`/`V` plane stores
bit `1 → +1`, bit `0 → -1` (`src/format/nq.rs`, `PackedLinear::u`), and its kernel evaluates
`Σ(1 − 2·bit)·t`, i.e. reads every stored plane inverted (`src/engine/kern/nq.rs`, `sign_dot!`,
module doc "Bit convention"). Copying the words verbatim therefore stores `−U` and `−V`, and the
served matrix `s_post ⊙ U·diag(s_mid)·V·s_pre` is unchanged because `(−U)(−V)ᵀ = U·Vᵀ`. `s_pre`
and `s_post` are row magnitudes (`scale_pre = |B_final|.mean(dim=0)`, `admm_nq.py:245`) and carry
no sign. A residual plane is read once and its dot negated, so it would *not* be sign-neutral; this
converter writes none.

The pack is version 2 (no residual, no group), one section per `(layer, kind)` in
`q, k, v, o, gate, up, down` order, the order `src/format/nq.rs` requires.
"""

import argparse
import os
import struct

import numpy as np
import torch

MAGIC = b"APNQPK01"
HEADER_BYTES = 32
VERSION_BASE = 2
KINDS = ("q", "k", "v", "o", "gate", "up", "down")
SUFFIX = {
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k",
    "self_attn.v_proj": "v",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gate",
    "mlp.up_proj": "up",
    "mlp.down_proj": "down",
}


def bf16_words(values) -> bytes:
    """Round an `f32` array to bfloat16 bits, nearest-even, and return little-endian words."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    round_to = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + round_to) >> 16).astype("<u2").tobytes()


def layer_of(prefix: str) -> int:
    """The `N` of `model.layers.N.<module>`."""
    parts = prefix.split(".")
    return int(parts[parts.index("layers") + 1])


def find_linears(state):
    """Map `(layer, kind)` to the module prefix carrying its `U_packed`/`V_packed`."""
    out = {}
    for key in state:
        if not key.endswith(".U_packed"):
            continue
        prefix = key[: -len(".U_packed")]
        kind = SUFFIX.get(".".join(prefix.split(".")[-2:]))
        if kind is None:
            raise SystemExit(f"{prefix}: no kind for its module name")
        out[(layer_of(prefix), kind)] = prefix
    return out


def words_of(packed) -> bytes:
    """The packed words as little-endian `u32`, the pack's carrier for `U`/`V`."""
    return np.ascontiguousarray(
        packed.detach().cpu().to(torch.int32).numpy(), dtype="<i4"
    ).astype("<u4").tobytes()


def check_with_reference(unpacker, packed, shape, stored: bytes, label: str) -> None:
    """Unpack a section's words with the reference's own unpacker and compare the bits."""
    ref = unpacker(packed, tuple(int(v) for v in shape)).detach().cpu().numpy()
    rows, cols = ref.shape
    words = np.frombuffer(stored, dtype="<u4").reshape(rows, (cols + 31) // 32)
    bits = (words[:, :, None] >> np.arange(32, dtype=np.uint32)[None, None, :]) & 1
    bits = bits.reshape(rows, words.shape[1] * 32)[:, :cols]
    if not np.array_equal(bits, ((1 - ref) // 2).astype(np.uint32)):
        raise SystemExit(f"{label}: stored words differ from the reference's unpacker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--og-src", default="", help="the reference checkout's src/, enables the unpacker check"
    )
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit(f"{args.checkpoint}: not a state_dict")
    linears = find_linears(state)
    by_layer = {}
    for (layer, kind), prefix in linears.items():
        by_layer.setdefault(layer, {})[kind] = prefix
    layers = 1 + max(by_layer)
    missing = [(l, k) for l in range(layers) for k in KINDS if k not in by_layer.get(l, {})]
    if missing:
        raise SystemExit(f"missing {len(missing)} linears, first {missing[0]}")

    unpacker = None
    if args.og_src:
        import importlib.util

        path = os.path.join(args.og_src, "nanoquant", "kernel", "utils.py")
        spec = importlib.util.spec_from_file_location("og_kernel_utils", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        unpacker = mod.binary_unpacker
        print(f"check: the reference's unpacker from {path}", flush=True)

    total = HEADER_BYTES
    with open(args.out, "wb") as out:
        out.write(MAGIC)
        out.write(
            struct.pack(
                "<IIIIII",
                VERSION_BASE,
                HEADER_BYTES,
                layers,
                len(KINDS),
                layers * len(KINDS),
                0,
            )
        )
        for layer in range(layers):
            for kind in KINDS:
                p = by_layer[layer][kind]
                u_shape = tuple(int(v) for v in state[p + ".U_shape"].tolist())
                v_shape = tuple(int(v) for v in state[p + ".V_shape"].tolist())
                out_rows, rank = u_shape
                v_rank, inn = v_shape
                if v_rank != rank:
                    raise SystemExit(f"{p}: U rank {rank} vs V rank {v_rank}")
                u = words_of(state[p + ".U_packed"])
                v = words_of(state[p + ".V_packed"])
                pre = bf16_words(state[p + ".scale_pre"].flatten().float().numpy())
                post = bf16_words(state[p + ".scale_post"].flatten().float().numpy())
                mid = bf16_words(
                    state[p + ".scale_mid"].flatten().float().numpy()
                    if p + ".scale_mid" in state
                    else np.ones(rank, dtype=np.float32)
                )
                u_words = (rank + 31) // 32
                v_words = (inn + 31) // 32
                if len(u) != out_rows * u_words * 4 or len(v) != rank * v_words * 4:
                    raise SystemExit(f"{p}: payload {len(u)}/{len(v)} B for {u_shape}/{v_shape}")
                if len(pre) != inn * 2 or len(post) != out_rows * 2 or len(mid) != rank * 2:
                    raise SystemExit(f"{p}: scales do not match the factor shape")
                out.write(
                    struct.pack(
                        "<IIIIIIII",
                        layer,
                        KINDS.index(kind),
                        out_rows,
                        inn,
                        rank,
                        u_words,
                        v_words,
                        0,
                    )
                )
                for plane in (u, v, pre, mid, post):
                    out.write(plane)
                total += 32 + len(u) + len(v) + len(pre) + len(mid) + len(post)
                if unpacker is not None:
                    check_with_reference(
                        unpacker, state[p + ".U_packed"], u_shape, u, f"l{layer} {kind} U"
                    )
                    check_with_reference(
                        unpacker, state[p + ".V_packed"], v_shape, v, f"l{layer} {kind} V"
                    )
                print(
                    f"l{layer:<2} {kind:<4} {out_rows}x{inn} r{rank} "
                    f"{u_words + v_words} w/row",
                    flush=True,
                )
    size = os.path.getsize(args.out)
    if size != total:
        raise SystemExit(f"{args.out}: {size} B, wrote {total}")
    print(f"pack {args.out} {total} B, {layers * len(KINDS)} sections", flush=True)


if __name__ == "__main__":
    main()
