"""Per-window NLL for base, reference-folded and our-folded Qwen3-0.6B.

Scores all 146 windows of 2048 tokens of the reference wikitext2 stream and
prints each window's mean nll plus the aggregate PPL, so the two compressed
models are compared on the same windows and the catastrophic-window count.

usage: window_scan.py [base|ref|ours ...]   (default: all three)
"""
import sys

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

BASE = "Qwen/Qwen3-0.6B-Base"
CKPT = "/workspace/ref-qwen3-0.6b-100.pt"
ROOT = "/workspace/loop-ref/nq-100"
KINDS = [("self_attn.q_proj", "attn"), ("self_attn.k_proj", "attn"),
         ("self_attn.v_proj", "attn"), ("self_attn.o_proj", "attn"),
         ("mlp.gate_proj", "mlp"), ("mlp.up_proj", "mlp"), ("mlp.down_proj", "mlp")]
FACTOR = ("U_packed", "V_packed", "U_shape", "V_shape", "scale_pre", "scale_post", "scale_mid")


def unpack(words: np.ndarray, cols: int) -> np.ndarray:
    rows, wpr = words.shape
    raw = np.ascontiguousarray(words).astype("<u4")
    bits = np.unpackbits(raw.view(np.uint8).reshape(rows, wpr, 4), bitorder="little",
                         axis=2).reshape(rows, wpr * 32)[:, :cols]
    return np.where(bits > 0, -1.0, 1.0).astype(np.float32)


def ours_w(path):
    raw = open(path, "rb").read()
    out, inn, rank, _ = (int(x) for x in np.frombuffer(raw[:32], dtype="<u8"))
    at = 32
    u = np.frombuffer(raw[at:at + 4 * out * rank], dtype="<f4").reshape(out, rank); at += 4 * out * rank
    vt = np.frombuffer(raw[at:at + 4 * inn * rank], dtype="<f4").reshape(inn, rank); at += 4 * inn * rank
    s_post = np.frombuffer(raw[at:at + 4 * out], dtype="<f4"); at += 4 * out
    s_pre = np.frombuffer(raw[at:at + 4 * inn], dtype="<f4")
    return (u @ vt.T) * s_pre[None, :] * s_post[:, None]


def build(kind):
    model = Qwen3ForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    if kind != "base":
        sd = torch.load(CKPT, map_location="cpu", weights_only=False) if kind == "ref" else None
        if sd is not None:
            model.load_state_dict({k: v.float() for k, v in sd.items() if not k.endswith(FACTOR)},
                                  strict=False)
        for l in range(len(model.model.layers)):
            for name, where in KINDS:
                if kind == "ref":
                    p = f"model.layers.{l}.{name}."
                    u = unpack(sd[p + "U_packed"].numpy(), int(sd[p + "U_shape"][1]))
                    v = unpack(sd[p + "V_packed"].numpy(), int(sd[p + "V_shape"][1]))
                    pre = sd[p + "scale_pre"].float().flatten().numpy()
                    post = sd[p + "scale_post"].float().flatten().numpy()
                    w = (u @ v) * pre[None, :] * post[:, None]
                else:
                    w = ours_w(f"{ROOT}/l{l}/{name}.param")
                mod = model.model.layers[l].self_attn if where == "attn" else model.model.layers[l].mlp
                getattr(mod, name.split(".")[1]).weight.data = torch.from_numpy(w)
    return model.eval().cuda()


def main():
    kinds = sys.argv[1:] or ["base", "ref", "ours"]
    sys.path.insert(0, "/workspace/NanoQuant/src")
    from nanoquant.utils.data_utils import get_test_loaders
    _, tl = get_test_loaders("wikitext2", seqlen=2048, model_name=BASE)
    ids = tl.input_ids[0][:299008].numpy()
    for kind in kinds:
        model = build(kind)
        nlls = []
        with torch.inference_mode():
            for i0 in range(0, len(ids) - 2047, 2048):
                x = torch.from_numpy(ids[i0:i0 + 2048].astype(np.int64)).cuda().unsqueeze(0)
                logits = model(x).logits[0, :-1].float()
                lp = torch.log_softmax(logits, -1)
                nlls.append(float(-lp.gather(1, x[0, 1:].unsqueeze(1)).mean()))
        a = np.array(nlls)
        print(f"{kind}: {len(a)} windows, mean nll {a.mean():.4f}, ppl {np.exp(a.mean()):.4f}, "
              f"median {np.median(a):.4f}, max {a.max():.4f}, "
              f"windows>8: {(a > 8).sum()}, >5: {(a > 5).sum()}")
        print(kind, "windows", " ".join(f"{v:.2f}" for v in a))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
