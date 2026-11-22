"""Score the reference NanoQuant checkpoint in plain HF PyTorch.

Folds `W = diag(s_post) U V^T diag(s_pre)` into dense weights, loads the
reference's non-factor tensors, and reports the mean NLL over the exact token
ids and positions `examples/ppl` scores.

usage: fold_ppl.py <ids.i32> [<checkpoint.pt>]
"""
import sys

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

BASE = "Qwen/Qwen3-0.6B-Base"
CKPT = "/workspace/ref-qwen3-0.6b-100.pt"
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


def main() -> None:
    ids_path = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else CKPT
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Qwen3ForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    plain = {k: v.float() for k, v in sd.items() if not k.endswith(FACTOR)}
    missing, unexpected = model.load_state_dict(plain, strict=False)
    print("missing", [k for k in missing if "rotary" not in k][:6],
          "unexpected", list(unexpected)[:6], flush=True)
    layers = model.model.layers
    for l in range(len(layers)):
        for name, where in KINDS:
            p = f"model.layers.{l}.{name}."
            u = unpack(sd[p + "U_packed"].numpy(), int(sd[p + "U_shape"][1]))
            v = unpack(sd[p + "V_packed"].numpy(), int(sd[p + "V_shape"][1]))
            pre = sd[p + "scale_pre"].float().flatten().numpy()
            post = sd[p + "scale_post"].float().flatten().numpy()
            w = (u @ v) * pre[None, :] * post[:, None]
            mod = layers[l].self_attn if where == "attn" else layers[l].mlp
            getattr(mod, name.split(".")[1]).weight.data = torch.from_numpy(w)
    ids = np.fromfile(ids_path, dtype="<i4")
    x = torch.from_numpy(ids.astype(np.int64)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(x).logits[0, :-1].float()
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, x[0, 1:].unsqueeze(1)).mean().item()
    print(f"torch folded: {len(ids)} ids, mean nll {nll:.6f}, ppl {np.exp(nll):.6f}")
    base = Qwen3ForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    base.eval()
    with torch.no_grad():
        logits = base(x).logits[0, :-1].float()
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, x[0, 1:].unsqueeze(1)).mean().item()
    print(f"torch base f32: mean nll {nll:.6f}, ppl {np.exp(nll):.6f}")


if __name__ == "__main__":
    main()
