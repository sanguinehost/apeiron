"""ONEBIT W15a: direct KL along the ray W_f + t D on the 21 `ffn_down` tensors of Qwen2.5-0.5B, all perturbed jointly.

The W14 quadratic form predicts KL(t) = t^2 K (K the joint second-order KL per token, `fisher_capture.py`). The direct
value KL(p_f || p_t) on the same 100 C4 chunks gives the ratio r(t) = KL(t) / (t^2 K): r -> 1 as t -> 0 when the
estimator and its normalisation are right, and r(1) - 1 is the part of the measured KLD the quadratic form does not carry.
Signed t separates the odd (cubic) from the even (quartic and higher) remainder:
    even(t) = (KL(t) + KL(-t)) / 2,   odd(t) = (KL(t) - KL(-t)) / 2.

Run with ~/spinor-ttl/.venv/bin/python (torch + CUDA).
usage: ray_scan.py STIFFNESS.json OUT.json [N_CHUNKS]
Cost: N_CHUNKS * (1 + |ARMS| * |TS|) forward passes of 512 tokens; per pass a 512 x 151936 float64 log-softmax and KL;
the arm deltas occupy 5 * 21 * 17.4 MB = 1.8 GB (float32) on the GPU.
"""
import json, sys, time
import numpy as np
import torch
import transformers

from fisher_capture import ARMS, BLOCKS, CALIB, CHUNK, load_deltas

TS = (0.125, 0.25, 0.5, 1.0, -0.5, -1.0)


def main(stiff_path, out_path, n_chunks):
    t0 = time.time()
    k_joint = {a: v["kl_joint"] / CHUNK for a, v in json.load(open(stiff_path))["arms"].items()}
    delta = {k: v.float() for k, v in load_deltas().items()}
    torch.cuda.empty_cache()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ids = tok.encode(open(CALIB, encoding="utf-8").read())
    model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).cuda().eval()
    downs = [model.model.layers[b].mlp.down_proj.weight for b in BLOCKS]
    w0 = [w.detach().clone() for w in downs]
    kl = np.zeros((n_chunks, len(ARMS), len(TS)))
    with torch.no_grad():
        for c in range(n_chunks):
            x = torch.tensor([ids[c * CHUNK:(c + 1) * CHUNK]], device="cuda")
            lp0 = torch.log_softmax(model(x).logits[0].double(), -1)
            p0 = lp0.exp()
            for i, a in enumerate(ARMS):
                for j, t in enumerate(TS):
                    for w, w_ref, b in zip(downs, w0, BLOCKS):
                        w.copy_(w_ref + t * delta[(a, b)])
                    lp = torch.log_softmax(model(x).logits[0].double(), -1)
                    kl[c, i, j] = float((p0 * (lp0 - lp)).sum(-1).mean())
            for w, w_ref in zip(downs, w0):
                w.copy_(w_ref)
    res = {"n_chunks": n_chunks, "ts": TS, "wall_s": time.time() - t0, "arms": {}}
    for i, a in enumerate(ARMS):
        mean = kl[:, i, :].mean(0)
        sem = kl[:, i, :].std(0, ddof=1) / np.sqrt(n_chunks)
        by_t = dict(zip(TS, mean))
        row = {
            "k_joint_per_token": k_joint[a],
            "kl": {str(t): float(m) for t, m in zip(TS, mean)},
            "kl_sem": {str(t): float(s) for t, s in zip(TS, sem)},
            "r": {str(t): float(by_t[t] / (t * t * k_joint[a])) for t in TS},
            "even": {str(t): float((by_t[t] + by_t[-t]) / 2) for t in (0.5, 1.0)},
            "odd": {str(t): float((by_t[t] - by_t[-t]) / 2) for t in (0.5, 1.0)},
        }
        res["arms"][a] = row
        print(a, json.dumps({k: row[k] for k in ("kl", "r", "even", "odd")}), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print("wall", res["wall_s"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100)
