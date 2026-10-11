"""ONEBIT W13: reverse water-filling floor of the 21 `ffn_down` tensors against the spectrum of the input Gram H.

Source model (`model` grade): each weight row w ~ N(0, s^2 I) over the n inputs, layer error r^T H r with r the row error
(`Perf.OnebitW10`). In the eigenbasis of H, direction i carries weight lambda_i. Reverse water-filling for the weighted
squared error gives, at water level theta,
    D(theta) = sum_i min(theta, s^2 lambda_i),    R(theta) = (1/2n) sum_i log2^+(s^2 lambda_i / theta)   [bits / weight]
and relative error rho = D / (s^2 sum_i lambda_i). The scale s^2 cancels: set s = 1.
Per block this prints the floor rho at the arms' rates (1.125 had-gptq1, 1.5625 IQ1_S, 4.5 Q4_K) and the floor rate at
each arm's measured rho (`arm_compare.py` output), so the distance from each arm to the isotropic-Gaussian floor is read
directly. The floor is a bound for this source model only; a weight matrix whose row covariance is not isotropic can sit
below it.

Run with /home/socol/Workspace/spinor-ttl/.venv/bin/python (torch + CUDA).
usage: waterfill.py ARM_COMPARE.json OUT.json
       waterfill.py --self-test   (white spectrum gives rho = 2^(-2R); a two-level spectrum matches its closed form)
"""
import json, math, sys, time
import numpy as np

N_IN = 4864
GRAM = "/home/socol/oc-tmp/gptq-w2/gram/blk.{}.ffn_down_in.f64"


def rho_of_theta(lam, theta):
    return float(np.minimum(theta, lam).sum() / lam.sum())


def rate_of_theta(lam, theta):
    return float(0.5 * np.log2(np.maximum(lam / theta, 1.0)).mean())


def theta_for_rate(lam, rate):
    lo, hi = lam.min() * 1e-12, lam.max()
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if rate_of_theta(lam, mid) > rate:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def theta_for_rho(lam, rho):
    lo, hi = lam.min() * 1e-12, lam.max()
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if rho_of_theta(lam, mid) > rho:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def self_test():
    lam = np.ones(N_IN)
    for r in (0.5, 1.0, 2.0, 4.0):
        got = rho_of_theta(lam, theta_for_rate(lam, r))
        assert abs(got - 2.0 ** (-2 * r)) < 1e-9, (r, got)
    # two-level spectrum: at 1 b/w the water level 1/16 sits above the weak half (lambda 1e-6), which gets rate 0
    lam2 = np.concatenate([np.ones(N_IN // 2), 1e-6 * np.ones(N_IN // 2)])
    th = theta_for_rate(lam2, 1.0)
    want = (0.5 / 16 + 0.5e-6) / (0.5 + 0.5e-6)
    assert abs(th - 1 / 16) < 1e-9 and abs(rho_of_theta(lam2, th) - want) < 1e-9, (th, rho_of_theta(lam2, th), want)
    print("self-test ok")


def main(arm_path, out_path):
    import torch

    arm = {b["block"]: b for b in json.load(open(arm_path))["blocks"]}
    out = {"blocks": []}
    t0 = time.time()
    for blk in sorted(arm):
        h = torch.from_numpy(np.fromfile(GRAM.format(blk), dtype=np.float64).reshape(N_IN, N_IN)).to("cuda")
        h = 0.5 * (h + h.T)
        lam = torch.linalg.eigvalsh(h).clamp_min(0).cpu().numpy()
        lam = np.sort(lam)[::-1]
        lam_pos = lam[lam > 1e-12 * lam[0]]
        rho = arm[blk]["rho"]
        row = {
            "block": blk,
            "n_pos": int(len(lam_pos)),
            "r99": int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.99) + 1),
            "floor_rho_at_rate": {
                str(r): rho_of_theta(lam, theta_for_rate(lam, r)) for r in (1.0, 1.125, 1.5625, 2.0, 4.5)
            },
            "floor_rate_at_rho": {
                a: rate_of_theta(lam, theta_for_rho(lam, rho[a]))
                for a in ("iq1_s", "iq1_m", "q4_k_m", "had_gptq1", "had_rtn")
            },
            "measured_rho": rho,
        }
        out["blocks"].append(row)
        print(json.dumps(row), flush=True)
        del h
        torch.cuda.empty_cache()
    out["wall_s"] = time.time() - t0
    json.dump(out, open(out_path, "w"), indent=1)
    print("wall", out["wall_s"])


if __name__ == "__main__":
    if sys.argv[1] == "--self-test":
        self_test()
    else:
        main(sys.argv[1], sys.argv[2])
