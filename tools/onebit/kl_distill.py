"""ONEBIT KL-distillation of a structured one-bit student of Qwen2.5-0.5B (straight-through sign training).

Student weight of one linear tensor (the fold_transfer representation, single sign plane):
    W'[o,i] = r[o,g(i)] * sign(L[o,i]) * c[i],   g(i) = i // 128,   L fp32 latent (not stored),
r f16 per (row, 128-group), c f16 per input column.  Stored bits per weight: 1 + 16/128 + 16/O  (about 1.126 b/w
over the 168 linears, below the 1.178 b/w of the post-training arm).  Embeddings, norms and biases stay teacher values.
Trained against the teacher by  KL(teacher || student)  on next-token distributions (straight-through estimator on
sign; Adam on L, log r, log c).  Held-out score: fold_transfer.kl_eval on calib-c4 chunks [100, 140), the same
instrument and chunks as the post-training arm.  Training text is C4 shard 00001, disjoint from the calibration file.

usage: kl_distill.py --out OUT.json [--steps N] [--lr-latent X] [--lr-scale X] [--batch B] [--dump PATH]
Run with /home/socol/Workspace/spinor-ttl/.venv/bin/python (torch + CUDA).
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_transfer as ft
import transformers

DEV, KINDS, GROUP, CHUNK = ft.DEV, list(ft.KINDS), ft.GROUP, ft.CHUNK
TRAIN_TXT = "/home/socol/oc-tmp/lanes/w18/kd/c4data/train-c4.txt"
TRAIN_NPY = "/home/socol/oc-tmp/lanes/w18/kd/c4data/train-c4.qwen.npy"


def train_ids(tok):
    if os.path.exists(TRAIN_NPY):
        return np.load(TRAIN_NPY)
    out, buf, n = [], [], 0
    with open(TRAIN_TXT, encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            n += len(line)
            if n >= 1_000_000:
                out.extend(tok.encode("".join(buf))); buf, n = [], 0
    if buf:
        out.extend(tok.encode("".join(buf)))
    a = np.asarray(out, dtype=np.int32)
    np.save(TRAIN_NPY, a)
    return a


def sign_ste(lat):
    s = torch.where(lat >= 0, 1.0, -1.0)
    return lat + (s - lat).detach()


def qweight(lat, a, b, r0):
    O, n = lat.shape
    r = r0 * torch.exp(a)
    return ((sign_ste(lat).view(O, n // GROUP, GROUP) * r[:, :, None]).view(O, n)) * torch.exp(b)[None, :]


def init_params(TL):
    P = []
    for L in TL:
        d = {}
        for k in KINDS:
            W = L[k].to(DEV).float()
            O, n = W.shape
            r0 = W.abs().view(O, n // GROUP, GROUP).mean(-1)
            d[k] = (W.clone().requires_grad_(), torch.zeros(O, n // GROUP, device=DEV, requires_grad=True),
                    torch.zeros(n, device=DEV, requires_grad=True), r0)
        P.append(d)
    return P


@torch.no_grad()
def student_layers(TL, P):
    """Dense dequantised student layers on CPU (teacher norms/biases), the input of ft.kl_eval."""
    Ls = []
    for l, L in enumerate(TL):
        Lq = dict(L)
        for k in KINDS:
            Lq[k] = qweight(*P[l][k]).float().cpu()
        Ls.append(Lq)
    return Ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--dump", default=None)
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-latent", type=float, default=1e-4); ap.add_argument("--lr-scale", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cal", type=int, default=100); ap.add_argument("--n-eval", type=int, default=40)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    t00 = ft.tick()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    cal = tok.encode(open(ft.CALIB, encoding="utf-8").read())
    nch = len(cal) // CHUNK
    ids_all = torch.tensor(cal[:nch * CHUNK]).view(nch, CHUNK)
    ids_ev = ids_all[a.n_cal:a.n_cal + a.n_eval]
    tr = train_ids(tok)
    ntr = len(tr) // CHUNK
    tr = torch.from_numpy(tr[:ntr * CHUNK].astype(np.int64)).view(ntr, CHUNK)
    assert a.steps * a.batch <= ntr, (a.steps, a.batch, ntr)
    perm = torch.randperm(ntr, generator=torch.Generator().manual_seed(a.seed))
    cfg, TL, hf = ft.load_model()
    del hf
    res = dict(script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), args=vars(a),
               train_sha256=hashlib.sha256(open(TRAIN_TXT, "rb").read()).hexdigest(), train_chunks=ntr,
               eval_chunks=[a.n_cal, a.n_cal + a.n_eval], gpu=torch.cuda.get_device_name(0))
    n_lin = sum(TL[l][k].numel() for l in range(cfg.nl) for k in KINDS)
    bits = n_lin + sum(TL[l][k].numel() // GROUP * 16 + TL[l][k].shape[1] * 16 for l in range(cfg.nl) for k in KINDS)
    res.update(linear_params=n_lin, bits=bits, bpw_linears=bits / n_lin)

    P = init_params(TL)
    TLg = [{k: v.to(DEV) for k, v in L.items()} for L in TL]
    Eg, fn = cfg.E.to(DEV), cfg.fnorm.to(DEV)
    lat = [P[l][k][0] for l in range(cfg.nl) for k in KINDS]
    sc = [t for l in range(cfg.nl) for k in KINDS for t in P[l][k][1:3]]
    init_sign = [(t.detach() >= 0) for t in lat]

    # exact fp32 instrument at the initial point (same code path as the final score)
    torch.set_float32_matmul_precision("highest")
    res["kl_heldout_init"], res["top1_heldout_init"] = ft.kl_eval(cfg, TL, student_layers(TL, P), ids_ev)
    print("init", res["kl_heldout_init"], res["top1_heldout_init"], flush=True)

    def layer_fn(h, li):
        Ld = {k: qweight(*P[li][k]) for k in KINDS}
        for k in ("ln1", "ln2", "qb", "kb", "vb"):
            Ld[k] = TLg[li][k]
        return ft.parts(cfg, Ld, h)["out"]

    def student_h(ids, grad):
        h = Eg[ids]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for li in range(cfg.nl):
                h = checkpoint(layer_fn, h, li, use_reentrant=False) if grad else layer_fn(h, li)
        return h.float()

    @torch.no_grad()
    def teacher_h(ids):
        h = Eg[ids]
        for li in range(cfg.nl):
            h = ft.parts(cfg, TLg[li], h)["out"]
        return h

    def kl_chunk(hs, ht):
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad():
                lt = F.log_softmax(ft.rms(ht, fn, cfg.eps) @ Eg.T, -1)
            ls = F.log_softmax(ft.rms(hs, fn, cfg.eps) @ Eg.T, -1)
            return (lt.exp() * (lt - ls)).sum(), (lt.argmax(-1) == ls.argmax(-1)).sum()

    def kl_batch(ids, grad):
        hs, ht = student_h(ids, grad).reshape(-1, cfg.hid), teacher_h(ids).reshape(-1, cfg.hid)
        tot, agree = 0.0, 0
        for s0 in range(0, hs.shape[0], 256):
            if grad:
                k, g = checkpoint(kl_chunk, hs[s0:s0 + 256], ht[s0:s0 + 256], use_reentrant=False)
            else:
                k, g = kl_chunk(hs[s0:s0 + 256], ht[s0:s0 + 256])
            tot = tot + k
            agree += int(g)
        return tot / hs.shape[0], agree / hs.shape[0]

    @torch.no_grad()
    def quick_eval():
        ks, ag = [], []
        for s0 in range(0, 8, 4):
            k, g = kl_batch(ids_ev[s0:s0 + 4].to(DEV), False)
            ks.append(float(k)); ag.append(g)
        return sum(ks) / len(ks), sum(ag) / len(ag)

    opt = torch.optim.Adam([dict(params=lat, lr=a.lr_latent), dict(params=sc, lr=a.lr_scale)], fused=True)
    base = [a.lr_latent, a.lr_scale]
    traj, t_train = [], ft.tick()
    torch.set_float32_matmul_precision("high")
    k0, g0 = quick_eval(); traj.append(dict(step=0, tokens=0, kl_quick=k0, top1_quick=g0))
    print("step 0 quick", k0, g0, flush=True)
    for step in range(1, a.steps + 1):
        f = min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / a.steps)))
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = base[gi] * f
        ids = tr[perm[(step - 1) * a.batch: step * a.batch]].to(DEV)
        loss, agree = kl_batch(ids, True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step {step} loss {float(loss):.4f} top1 {agree:.4f} t {ft.tick() - t_train:.1f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            k, g = quick_eval()
            traj.append(dict(step=step, tokens=step * a.batch * CHUNK, kl_quick=k, top1_quick=g,
                             train_loss=float(loss), wall_s=ft.tick() - t_train))
            print(f"step {step} quick {k:.4f} {g:.4f}", flush=True)
    res["train_wall_s"] = ft.tick() - t_train
    res["trajectory"] = traj
    res["tokens_trained"] = a.steps * a.batch * CHUNK
    flips = sum(int(((t.detach() >= 0) != s).sum()) for t, s in zip(lat, init_sign))
    res["sign_flip_frac"] = flips / n_lin
    torch.set_float32_matmul_precision("highest")
    Ls = student_layers(TL, P)
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    for t in (P, TLg, lat, sc, init_sign):
        t.clear()
    del opt
    torch.cuda.empty_cache()
    res["kl_heldout"], res["top1_heldout"] = ft.kl_eval(cfg, TL, Ls, ids_ev)
    print("final", res["kl_heldout"], res["top1_heldout"], flush=True)
    if a.dump:
        torch.save({"layers": [{k: v.cpu() for k, v in L.items()} for L in Ls], "fnorm": cfg.fnorm.cpu()}, a.dump)
    res["wall_total_s"] = ft.tick() - t00
    json.dump(res, open(a.out, "w"), indent=1)
    print("wall_total_s", res["wall_total_s"])


if __name__ == "__main__":
    main()
