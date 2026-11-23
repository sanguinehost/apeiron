#!/usr/bin/env python3
"""The reference R1 baseline: `factorize_admm_nanoquant` on one layer-0 linear, its Gram
diagonal and its rank, on the same L4 the port runs on.

The function is loaded from `admm_nq.py` directly: the package `__init__` pulls the block-fit
modules, which import the pinned transformers, and the factorisation reads none of that. The
call is the reference's own (`compress_block.py:125-130`): `outer_iters` from the CLI default 400,
`inner_iters` 5, `reg` 3e-2, rho `linear` (`quant_config.py:27`), seed 0, `is_transpose` from the
shape. Prints the reference's own recon line (`compress_block.py:143-152`) and the CUDA peak.
"""
import argparse
import glob
import importlib.util
import json
import random
import time

import numpy as np
import torch
from safetensors import safe_open

REF = "/workspace/NanoQuant/src/nanoquant/core/admm_nq.py"


def load_ref(path=REF):
    spec = importlib.util.spec_from_file_location("admm_nq_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/workspace/models/Qwen3-8B-Base")
    p.add_argument("--dump", default="/workspace/dump-8b")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--key", default="mlp.gate_proj")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    name = f"model.layers.{a.layer}.{a.key}.weight"
    w = None
    for path in sorted(glob.glob(f"{a.weights}/*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            if name in f.keys():
                w = f.get_tensor(name)
                break
    assert w is not None, f"{name} not in {a.weights}"
    ranks = json.load(open(f"{a.dump}/ranks.json"))
    rank = ranks[f"{a.layer}.{a.key}"]
    stats = f"{a.dump}/stats/model.layers.{a.layer}.{a.key}"
    i_norm = torch.from_numpy(np.fromfile(f"{stats}.i_norm.f32", dtype=np.float32))
    o_norm = torch.from_numpy(np.fromfile(f"{stats}.o_norm.f32", dtype=np.float32))

    ref = load_ref()
    dev = "cuda"
    W = w.to(dev)
    set_seed(a.seed)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    r = ref.factorize_admm_nanoquant(
        W,
        i_norm.to(dev),
        o_norm.to(dev),
        mid_rank=rank,
        outer_iters=a.iters,
        inner_iters=5,
        reg=3e-2,
        is_transpose=W.shape[0] < W.shape[1],
        rho_scheduler="linear",
        print_admm_steps=False,
    )
    dt = time.time() - t
    raw = (r["W_final"] - W).square().sum().item()
    orig = W.square().sum().item()
    print(
        f"REF R1 {name} rank {rank} {tuple(W.shape)}: raw={raw:.4f}, norm={raw / orig:.4f}, "
        f"per_el={raw / W.numel():.4e}, ADMM time={dt:.2f}s, "
        f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, "
        f"{torch.cuda.get_device_name(0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
