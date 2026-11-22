"""Rolling-window PPL: each 2048-token window scored with 1024 tokens of leading context.

The standard lm-eval wikitext protocol: the stream is scored once, and every window after
the first carries the previous window's tail as context, so no position is a cold start.
Reports the aggregate PPL over all scored positions, per window means, and the tail count.

usage: slide_scan.py base|ref|ours [...]   (names resolve to the pack paths below)
"""
import sys

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

BASE = "Qwen/Qwen3-0.6B-Base"
PACKS = {"ref100": "/workspace/ref100.nqp",
         "ours100": "/workspace/loop-ref/nq-100/pack.nqp",
         "ref080": "/workspace/ref080.nqp",
         "ours080": "/workspace/loop-080/nq-080/pack.nqp"}
NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
CTX = 1024


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
    count = int(np.frombuffer(raw[8:32], dtype="<u4")[4])
    out = {}
    at = 32
    for _ in range(count):
        l, k, o, i, rank, uw, vw, _ = (int(x) for x in np.frombuffer(raw[at:at + 32], dtype="<u4"))
        at += 32
        u = signs(np.frombuffer(raw[at:at + 4 * o * uw], dtype="<u4"), o, rank); at += 4 * o * uw
        v = signs(np.frombuffer(raw[at:at + 4 * rank * vw], dtype="<u4"), rank, i); at += 4 * rank * vw
        pre = bf16(np.frombuffer(raw[at:at + 2 * i], dtype="<u2")); at += 2 * i
        mid = bf16(np.frombuffer(raw[at:at + 2 * rank], dtype="<u2")); at += 2 * rank
        post = bf16(np.frombuffer(raw[at:at + 2 * o], dtype="<u2")); at += 2 * o
        out[(l, k)] = (u @ (v * mid[:, None])) * pre[None, :] * post[:, None]
    assert at == len(raw), (at, len(raw))
    return out


def build(path):
    model = Qwen3ForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    if path:
        ws = read_pack(path)
        for l in range(len(model.model.layers)):
            for k, name in enumerate(NAMES):
                mod = model.model.layers[l].self_attn if k < 4 else model.model.layers[l].mlp
                getattr(mod, name.split(".")[1]).weight.data = torch.from_numpy(ws[(l, k)])
    return model.eval().cuda()


def main():
    keys = sys.argv[1:] or list(PACKS)
    sys.path.insert(0, "/workspace/NanoQuant/src")
    from nanoquant.utils.data_utils import get_test_loaders
    _, tl = get_test_loaders("wikitext2", seqlen=2048, model_name=BASE)
    ids = tl.input_ids[0][:299008].numpy()
    starts = list(range(0, len(ids) - 2047, 2048))
    for key in keys:
        model = build(None if key == "base" else PACKS[key])
        tot, cnt, per = 0.0, 0, []
        with torch.inference_mode():
            for i0 in starts:
                lead = min(CTX, i0)
                x = torch.from_numpy(ids[i0 - lead:i0 + 2048].astype(np.int64)).cuda().unsqueeze(0)
                lp = torch.log_softmax(model(x).logits[0].float(), -1)
                nll = -lp[:-1].gather(1, x[0, 1:].unsqueeze(1)).squeeze(1)
                nll = nll[lead - 1:] if lead else nll
                tot += float(nll.sum())
                cnt += int(nll.numel())
                if lead:
                    per.append(float(nll.mean()))
                else:
                    per.append(float(nll[1:].mean()) if nll.numel() > 1 else float(nll[0]))
        a = np.array(per)
        print(f"{key}: rolling ctx {CTX}, {cnt} positions, mean nll {tot / cnt:.4f}, "
              f"ppl {np.exp(tot / cnt):.4f}, per-window nll max {a.max():.3f}, "
              f"windows>8: {(a > 8).sum()}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
