"""Per-window NLL of an APNQPK01 (version 2) pack, folded dense in PyTorch.

usage: scan_pack.py <pack.nqp>
"""
import sys

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

BASE = "Qwen/Qwen3-0.6B-Base"


def bf16(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def signs(words, rows, cols):
    wpr = (cols + 31) // 32
    a = np.asarray(words, dtype=np.uint32).reshape(rows, wpr)
    bits = np.unpackbits(a.view(np.uint8).reshape(rows, wpr, 4), bitorder="little",
                         axis=2).reshape(rows, wpr * 32)[:, :cols]
    return np.where(bits > 0, 1.0, -1.0).astype(np.float32)


def read_pack(path):
    raw = open(path, "rb").read()
    assert raw[:8] == b"APNQPK01"
    version, hdr, layers, kinds, count, _ = (int(x) for x in np.frombuffer(raw[8:32], dtype="<u4"))
    assert version == 2 and hdr == 32, (version, hdr)
    out = {}
    at = 32
    for _ in range(count):
        l, k, o, i, rank, uw, vw, kk = (int(x) for x in np.frombuffer(raw[at:at + 32], dtype="<u4"))
        at += 32
        u = signs(np.frombuffer(raw[at:at + 4 * o * uw], dtype="<u4"), o, rank); at += 4 * o * uw
        v = signs(np.frombuffer(raw[at:at + 4 * rank * vw], dtype="<u4"), rank, i); at += 4 * rank * vw
        pre = bf16(np.frombuffer(raw[at:at + 2 * i], dtype="<u2")); at += 2 * i
        mid = bf16(np.frombuffer(raw[at:at + 2 * rank], dtype="<u2")); at += 2 * rank
        post = bf16(np.frombuffer(raw[at:at + 2 * o], dtype="<u2")); at += 2 * o
        assert kk == 0
        w = (u @ (v * mid[:, None])) * pre[None, :] * post[:, None]
        out[(l, k)] = w
    assert at == len(raw), (at, len(raw))
    return layers, out


NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main():
    layers, ws = read_pack(sys.argv[1])
    model = Qwen3ForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    for l in range(layers):
        for k, name in enumerate(NAMES):
            mod = model.model.layers[l].self_attn if k < 4 else model.model.layers[l].mlp
            getattr(mod, name.split(".")[1]).weight.data = torch.from_numpy(ws[(l, k)])
    model.eval().cuda()
    sys.path.insert(0, "/workspace/NanoQuant/src")
    from nanoquant.utils.data_utils import get_test_loaders
    _, tl = get_test_loaders("wikitext2", seqlen=2048, model_name=BASE)
    ids = tl.input_ids[0][:299008].numpy()
    nlls = []
    with torch.inference_mode():
        for i0 in range(0, len(ids) - 2047, 2048):
            x = torch.from_numpy(ids[i0:i0 + 2048].astype(np.int64)).cuda().unsqueeze(0)
            lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
            nlls.append(float(-lp.gather(1, x[0, 1:].unsqueeze(1)).mean()))
    a = np.array(nlls)
    print(f"{sys.argv[1]}: {len(a)} windows, mean nll {a.mean():.4f}, ppl {np.exp(a.mean()):.4f}, "
          f"median {np.median(a):.4f}, max {a.max():.4f}, windows>8: {(a > 8).sum()}")
    print("windows", " ".join(f"{v:.2f}" for v in a))


if __name__ == "__main__":
    main()
