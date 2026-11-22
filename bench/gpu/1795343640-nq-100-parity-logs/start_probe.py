"""Position-0 health across all 146 window starts: base vs a pack.

For each window start token, one single-token forward pass through both models records
the layer-2 max (the massive activation on channel 35) and the layer-27 max, then prints
the correlation with the window's mean nll. Chat-template openers are probed too.

usage: start_probe.py <pack.nqp>
"""
import glob
import sys

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

BASE = "Qwen/Qwen3-0.6B-Base"
NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


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


def first_rows(model, toks):
    outs = []

    def hook(_m, _i, o):
        o = o[0] if isinstance(o, tuple) else o
        outs.append(o.detach())

    hs = [model.model.layers[i].register_forward_hook(hook) for i in range(len(model.model.layers))]
    res = []
    for tok in toks:
        outs.clear()
        with torch.inference_mode():
            model(torch.tensor([[tok]], device="cuda"))
        res.append((float(outs[2][0, 0].abs().max()), float(outs[27][0, 0].abs().max()),
                    float(outs[2][0, 0].norm())))
    for h in hs:
        h.remove()
    return np.array(res)


def main():
    sys.path.insert(0, "/workspace/NanoQuant/src")
    from nanoquant.utils.data_utils import get_test_loaders
    _, tl = get_test_loaders("wikitext2", seqlen=2048, model_name=BASE)
    ids = tl.input_ids[0][:299008].numpy()
    starts = list(range(0, len(ids) - 2047, 2048))
    toks = [int(ids[i0]) for i0 in starts]
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(glob.glob("/workspace/hf/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*/"
                                       "tokenizer.json")[0])
    base = build(None)
    b = first_rows(base, toks)
    del base
    torch.cuda.empty_cache()
    ours = build(sys.argv[1])
    o = first_rows(ours, toks)
    per = []
    with torch.inference_mode():
        for i0 in starts:
            x = torch.from_numpy(ids[i0:i0 + 2048].astype(np.int64)).cuda().unsqueeze(0)
            lp = torch.log_softmax(ours(x).logits[0, :-1].float(), -1)
            per.append(float(-lp.gather(1, x[0, 1:].unsqueeze(1)).mean()))
    per = np.array(per)
    ratio = o[:, 0] / np.maximum(b[:, 0], 1e-9)
    print(f"ours: {len(per)} windows, mean nll {per.mean():.4f}")
    print(f"l02 max ratio ours/base: min {ratio.min():.4f} median {np.median(ratio):.4f} "
          f"max {ratio.max():.4f}; below 0.5: {(ratio < 0.5).sum()}")
    for w in np.argsort(per)[-6:][::-1]:
        print(f"  w{w:3d} nll {per[w]:7.3f} start {tk.id_to_token(toks[w])!r} "
              f"base l02 {b[w,0]:8.1f} ours {o[w,0]:8.1f} ratio {ratio[w]:.4f} "
              f"base l27 {b[w,1]:7.1f} ours {o[w,1]:7.1f}")
    lo = np.argsort(ratio)[:4]
    print("lowest l02 ratios:")
    for w in lo:
        print(f"  w{w:3d} nll {per[w]:7.3f} start {tk.id_to_token(toks[w])!r} "
              f"base {b[w,0]:8.1f} ours {o[w,0]:8.1f} ratio {ratio[w]:.4f}")
    print("chat-template openers (ours):")
    for tok in (151643, 151644, 151645, 151646, 271, 873, 220):
        r = first_rows(ours, [tok])[0]
        print(f"  {tok} {tk.id_to_token(tok)!r} l02max {r[0]:8.1f} l02norm {r[2]:8.1f} "
              f"l27max {r[1]:8.1f}")


if __name__ == "__main__":
    main()
