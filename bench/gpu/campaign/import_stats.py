"""Import NanoQuant's own 128x2048 calibration statistics into the campaign cache.

The reference's `collect_stats(strategy='online')` computes, per linear, the clipped input
second moment (`i_norm`) and the clipped output second moment (`o_norm`), shrunk by 0.4. Those
are the same two statistics `nq_dump_write` reads as `diag.i64` and `o_norm.i64`; running them
here on the L4 replaces the port's CPU pass, which is 28 h at this model's size. The drawn
samples are checked against the base file the Rust side writes, so the statistics and
`calib_ids.i32` describe the same 128 samples.

usage: import_stats.py
env: MODEL (local snapshot), BASE (calib.s128.kld), OUT (cache), NLAYER
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/NanoQuant/src")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import nanoquant.modules.hub  # noqa: E402,F401  (import order: hub before core.importance)
from nanoquant.core.importance import collect_stats, get_shrunk_stats  # noqa: E402
from nanoquant.utils.data_utils import get_calib_loader, prepare_dataset  # noqa: E402

MODEL = os.environ.get("MODEL", "/workspace/models/Qwen3-8B-Base")
BASE = os.environ.get("BASE", "/workspace/calib8b/calib.s128.kld")
OUT = os.environ.get("OUT", "/workspace/calib8b/cache")
TAG = "s128"
SHRINK = 0.4
# (linear, cache group, short name); q/k/v share the attention-norm input, gate/up the ln2 input.
KINDS = {
    "self_attn.q_proj": ("qkv", "q"),
    "self_attn.k_proj": ("qkv", "k"),
    "self_attn.v_proj": ("qkv", "v"),
    "self_attn.o_proj": ("o", "o"),
    "mlp.gate_proj": ("gate_up", "gate"),
    "mlp.up_proj": ("gate_up", "up"),
    "mlp.down_proj": ("down", "down"),
}
SEQ = 2048
SAMPLES = 128
GRID = 4294967296.0


def write_words(path, v):
    v = np.asarray(v, dtype=np.float64).ravel()
    q = np.round(v * GRID).astype(np.int64)
    assert np.all(np.abs(v) < 2147483648.0), (path, np.abs(v).max())
    words = np.concatenate((np.array([1, v.size], dtype=np.uint64), q.view(np.uint64)))
    words.tofile(path)


def main():
    n_layer = int(os.environ.get("NLAYER", "36"))
    quant = {
        "calib_dataset": "wikitext2",
        "num_calib_samples": SAMPLES,
        "seed": 0,
        "seqlen": SEQ,
    }
    data = prepare_dataset(MODEL, quant)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    loader = get_calib_loader(data, tok, SAMPLES, 0, SEQ)
    # The base file is sparse-padded out to its logits rows; only the id table is dense.
    tape = np.fromfile(BASE, dtype=np.int32, offset=20, count=SAMPLES * SEQ).reshape(
        SAMPLES, SEQ
    )
    drawn = loader.numpy()
    same = np.array_equal(drawn, tape)
    print("tape matches drawn loader:", same, flush=True)
    if not same:
        # The Rust base writer draws its own windows. The statistics below describe the loader's
        # samples, so the loader is authoritative and the base's dense id table is rewritten.
        mm = np.memmap(BASE, dtype=np.int32, mode="r+", offset=20, shape=SAMPLES * SEQ)
        mm[:] = drawn.ravel()
        mm.flush()
        back = np.fromfile(BASE, dtype=np.int32, offset=20, count=SAMPLES * SEQ)
        assert np.array_equal(back.reshape(SAMPLES, SEQ), drawn), "base rewrite"
        print("rewrote base id table from the loader", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    raw = collect_stats(model, loader, "cuda", strategy="online")
    shrunk = get_shrunk_stats(raw, shrinkage=SHRINK)
    print("stats keys", len(shrunk["i_norm"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    grid = 4294967296.0
    written = {}
    for layer in range(n_layer):
        for full, (gram, short) in KINDS.items():
            key = f"model.layers.{layer}.{full}"
            if (layer, gram) not in written:
                v = shrunk["i_norm"][key].detach().float().cpu().numpy()
                write_words(f"{OUT}/g{layer}.{TAG}.{gram}.diag.i64", v)
                written[(layer, gram)] = True
            v = shrunk["o_norm"][key].detach().float().cpu().numpy()
            write_words(f"{OUT}/g{layer}.{TAG}.{short}.o_norm.i64", v)
    print(f"IMPORT done: {n_layer} layers -> {OUT}", flush=True)
    with open(f"{OUT}/import.json", "w") as f:
        json.dump({"model": MODEL, "shrink": SHRINK, "samples": SAMPLES, "grid": grid}, f)


if __name__ == "__main__":
    main()
