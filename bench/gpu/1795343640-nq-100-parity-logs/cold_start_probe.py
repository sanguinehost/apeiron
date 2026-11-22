"""Cold-start probe on CPU: plain / BOS / 1024-context for the reference tail windows.

The earlier probe ran on the GPU and its script was not retained. This one is CPU-only
so it can rerun on a host without a working GPU. Plain must reproduce the window lines
in window_scan.log (ref window 6 = 18.48, 10 = 19.13, 139 = 19.53) before the BOS and
context rows are read.

usage: cold_start_probe.py <ref100.nqp> [window ...]
"""
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

BASE = "Qwen/Qwen3-0.6B-Base"
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
    assert raw[:8] == b"APNQPK01", raw[:8]
    count = int(np.frombuffer(raw[8:32], dtype="<u4")[4])
    out, at = {}, 32
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
    ws = read_pack(path)
    for l in range(len(model.model.layers)):
        for k, name in enumerate(NAMES):
            mod = model.model.layers[l].self_attn if k < 4 else model.model.layers[l].mlp
            getattr(mod, name.split(".")[1]).weight.data = torch.from_numpy(ws[(l, k)])
    return model.eval()


def window_nll(model, ids, i0, lead):
    start = i0 - lead
    x = torch.from_numpy(ids[start:i0 + 2048].astype(np.int64)).unsqueeze(0)
    with torch.inference_mode():
        logits = model(x).logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    if lead == 0:
        return float(-lp.gather(1, x[0, 1:].unsqueeze(1)).mean())
    tgt = x[0, lead:]
    return float(-lp[lead - 1:lead - 1 + 2048].gather(1, tgt.unsqueeze(1)).mean())


def main():
    pack = sys.argv[1]
    wins = [int(x) for x in sys.argv[2:]] or [6, 10, 139]
    sys.path.insert(0, "/workspace/NanoQuant/src")
    from nanoquant.utils.data_utils import get_test_loaders
    _, tl = get_test_loaders("wikitext2", seqlen=2048, model_name=BASE)
    ids = tl.input_ids[0][:299008].numpy()
    tok = AutoTokenizer.from_pretrained(BASE)
    bos = tok.bos_token_id if tok.bos_token_id is not None else 151643
    print(f"bos id {bos} token {tok.decode([bos])!r}", flush=True)
    model = build(pack)
    for i0 in wins:
        plain = window_nll(model, ids, i0, 0)
        lead = min(CTX, i0)
        ctx = window_nll(model, ids, i0, lead)
        xb = np.concatenate([[bos], ids[i0:i0 + 2048]])
        x = torch.from_numpy(xb.astype(np.int64)).unsqueeze(0)
        with torch.inference_mode():
            logits = model(x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, dim=-1)
        bos_nll = float(-lp[0:2048].gather(1, x[0, 1:2049].unsqueeze(1)).mean())
        print(f"window {i0}: plain {plain:.4f}  bos {bos_nll:.4f}  ctx{lead} {ctx:.4f}", flush=True)


if __name__ == "__main__":
    main()
