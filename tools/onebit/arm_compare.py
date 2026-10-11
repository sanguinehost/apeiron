"""ONEBIT W12: relative output error of each quantised arm on the 21 `ffn_down` tensors, against two spectral
quantities of the layer's input Gram H.

For a weight row r (the error of one output channel) the layer error is r^T H r (`Perf.OnebitW10`). A quantiser
with error-feedback (GPTQ) shapes the error spectrum; at fixed quantiser noise sigma_q^2 the shaped error satisfies
the Gerzon-Craven constraint geometric-mean(N_i) = sigma_q^2, so the error against H is
n sigma_q^2 GM(lambda) (lambda = eig(H), GM = det^(1/n)) against n sigma_q^2 AM(lambda) for white noise.
A quantiser that sees only diag(H) reaches n sigma_q^2 GM(diag H). Hadamard: GM(lambda) <= GM(diag H),
equality iff H is diagonal iff every pair discriminant H_ij^2 - H_ii H_jj is 0 (`Perf.Iq1s.pair_discriminant_nonpos`).

Per block: ln GM via Cholesky of the damped Gram (damping 0.01 mean(diag), as GPTQ), AM = mean(diag), relative error
rho(arm) = sum_rows (d H d^T) / sum_rows (w H w^T) with d = w_arm - w_f16, H undamped.

Run with ~/spinor-ttl/.venv/bin/python (torch + CUDA). Reads gguf through llama.cpp gguf-py.
usage: arm_compare.py OUT.json
"""
import json, math, sys, time
import numpy as np
import torch

sys.path.insert(0, "~/llama.cpp/gguf-py")
import gguf
from gguf.quants import dequantize

N_IN, N_OUT, BS = 4864, 896, 128
GRAM = "~/gptq-w2/gram/blk.{}.ffn_down_in.f64"
FILES = {
    "f16": "~/models/gguf/qwen05b-f16.gguf",
    "iq1_s": "~/onebit-w9/qwen05b-IQ1_S-imat.gguf",
    "iq1_m": "~/onebit-w9/qwen05b-IQ1_M-imat.gguf",
    "q4_k_m": "~/seal/qwen05b-Q4_K_M-imat.gguf",
    "rtn": "~/onebit-w9/p1-rtn-f16.gguf",
    "had_gptq1": "~/onebit-w9/p1-hadgptq1-f16.gguf",
}
dev = "cuda"
readers = {k: gguf.GGUFReader(v) for k, v in FILES.items()}


def tensor(arm, name):
    for t in readers[arm].tensors:
        if t.name == name:
            if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                a = np.asarray(t.data, dtype=np.float32)
            else:
                a = dequantize(t.data, t.tensor_type)
            return torch.from_numpy(np.ascontiguousarray(a.reshape(N_OUT, N_IN))).to(dev, torch.float64)
    raise KeyError((arm, name))


def hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


R = hadamard(BS).to(dev)


def had_rtn(w):
    wr = (w.reshape(N_OUT, N_IN // BS, BS) @ R).reshape(N_OUT, N_IN // BS, BS)
    a = wr.abs().mean(-1, keepdim=True)
    q = torch.sign(wr) * a
    return (q @ R.T).reshape(N_OUT, N_IN)


def quad(d, h):
    return float(((d @ h) * d).sum())


out = {"blocks": [], "files": FILES}
t0 = time.time()
for blk in range(3, 24):
    name = f"blk.{blk}.ffn_down.weight"
    h = torch.from_numpy(np.fromfile(GRAM.format(blk), dtype=np.float64).reshape(N_IN, N_IN)).to(dev)
    h = 0.5 * (h + h.T)
    diag = torch.diagonal(h)
    hd = h.clone()
    hd.diagonal().add_(0.01 * float(diag.mean()))
    L = torch.linalg.cholesky(hd)
    ln_gm = 2.0 * float(torch.log(torch.diagonal(L)).mean())
    ln_gm_diag = float(torch.log(torch.diagonal(hd)).mean())
    ln_am = math.log(float(torch.diagonal(hd).mean()))
    w = tensor("f16", name)
    base = quad(w, h)
    row = {
        "block": blk,
        "ln_am": ln_am,
        "ln_gm_eig": ln_gm,
        "ln_gm_diag": ln_gm_diag,
        "gain_full_db": 10 * (ln_am - ln_gm) / math.log(10),
        "gain_diag_db": 10 * (ln_am - ln_gm_diag) / math.log(10),
        "rho": {},
    }
    for arm in ("iq1_s", "iq1_m", "q4_k_m", "rtn", "had_gptq1"):
        row["rho"][arm] = quad(tensor(arm, name) - w, h) / base
    row["rho"]["had_rtn"] = quad(had_rtn(w) - w, h) / base
    out["blocks"].append(row)
    print(json.dumps(row), flush=True)
    del h, hd, L
    torch.cuda.empty_cache()
out["wall_s"] = time.time() - t0
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("wall", out["wall_s"])
