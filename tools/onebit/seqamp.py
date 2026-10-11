"""ONEBIT W11: sequential one-bit quantisation of Qwen2.5-0.5B on the quantised path, with an optional joint
refit of the block amplitudes, scored by paired KLD against the unquantised model.

Orientation `s = sign(W R)` (R the block-diagonal Hadamard / sqrt(128) when --rot 1) comes from `rtn` or `gptq`.
Amplitudes are one per 128-block: the block mean |w| (`rtn`, `gptq`), or the per-row joint least-squares solution
`G a = Phi^T M` (`--refit 1`) against the teacher output on the quantised path's own inputs.
Each linear is fitted on the inputs the quantised prefix delivers (`--path q`) or on the teacher's (`--path f`).

Run with ~/spinor-ttl/.venv/bin/python.
"""
import argparse, json, time
import torch
import transformers

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["rtn", "gptq"], required=True)
p.add_argument("--refit", type=int, default=0)
p.add_argument("--rot", type=int, default=1)
p.add_argument("--path", choices=["q", "f"], default="q")
p.add_argument("--calib-chunks", type=int, default=100)
p.add_argument("--eval-chunks", type=int, default=64)
p.add_argument("--blocks", type=int, default=24)
p.add_argument("--out", required=True)
a = p.parse_args()

BS, CH = 128, 512
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
teacher = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).to(dev).eval()
student = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).to(dev).eval()
for m in (teacher, student):
    m.requires_grad_(False)


def chunks(path, n):
    ids = tok.encode(open(path).read())[: n * CH]
    return torch.tensor(ids, device=dev).view(n, CH)


def hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n**0.5


R1 = hadamard(BS).to(dev)


def rotate_cols(w):  # w @ R, R block-diagonal
    m, n = w.shape
    return (w.double().view(m, n // BS, BS) @ R1).reshape(m, n)


def gptq_orient(w, h, damp=0.01):
    """GPTQ with a sign/mean-|w| 128-block quantiser. w: m x n (f64), h: n x n (f64). Returns Q."""
    m, n = w.shape
    w = w.clone()
    d = torch.arange(n, device=dev)
    h = h.clone()
    dead = h[d, d] == 0
    h[d[dead], d[dead]] = 1.0
    w[:, dead] = 0
    h[d, d] += damp * h[d, d].mean()
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    q = torch.zeros_like(w)
    for i1 in range(0, n, BS):
        i2 = i1 + BS
        w1 = w[:, i1:i2].clone()
        amp = w1.abs().mean(1)
        err1 = torch.zeros_like(w1)
        u1 = u[i1:i2, i1:i2]
        for i in range(BS):
            col = w1[:, i]
            qc = amp * torch.where(col >= 0, 1.0, -1.0)
            q[:, i1 + i] = qc
            e = (col - qc) / u1[i, i]
            w1[:, i:] -= e[:, None] * u1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ u[i1:i2, i2:]
    return q


def refit(s, hp, mp):
    """Per-row joint amplitudes. s: m x n (+-1), hp: n x n, mp: n x m (X'^T Y). Returns amp m x nb (f64)."""
    m, n = s.shape
    nb = n // BS
    out = torch.empty(m, nb, dtype=torch.float64, device=dev)
    h4 = hp.float().view(n, nb, BS)
    step = max(1, int(2e8 // (n * nb)))
    for r0 in range(0, m, step):
        r1 = min(m, r0 + step)
        sb = s[r0:r1].float().view(r1 - r0, nb, BS)
        t = torch.einsum("jcl,icl->ijc", h4, sb)                       # H Phi_i, rows x n x nb
        g = torch.einsum("ibl,iblc->ibc", sb, t.view(r1 - r0, nb, BS, nb)).double()
        rhs = torch.einsum("ibl,bli->ib", sb, mp.float().view(nb, BS, m)[:, :, r0:r1]).double()
        g = g + 1e-9 * torch.diag_embed(torch.diagonal(g, dim1=1, dim2=2).mean(1, keepdim=True).expand(-1, nb))
        out[r0:r1] = torch.linalg.solve(g, rhs.unsqueeze(-1)).squeeze(-1)
    return out


def quantise_linear(lin, xq, yt):
    """xq: N x n inputs the layer receives, yt: N x m teacher outputs. Replaces lin.weight in place."""
    w0 = lin.weight.data.double()
    m, n = w0.shape
    N = xq.shape[0]
    h = torch.zeros(n, n, dtype=torch.float64, device=dev)
    mm = torch.zeros(n, m, dtype=torch.float64, device=dev)
    for i in range(0, N, 8192):
        xc = xq[i:i + 8192].double()
        h += xc.T @ xc
        mm += xc.T @ yt[i:i + 8192].double()
    h /= N
    mm /= N
    if a.rot:
        r = torch.block_diag(*([R1] * (n // BS)))
        w = w0 @ r
        h = r.T @ h @ r
        mm = r.T @ mm
    else:
        w = w0
    if a.mode == "gptq":
        qm = gptq_orient(w, h)
        s = torch.where(qm >= 0, 1.0, -1.0)
        amp = qm.abs().view(m, n // BS, BS)[:, :, 0]
    else:
        s = torch.where(w >= 0, 1.0, -1.0)
        amp = w.abs().view(m, n // BS, BS).mean(2)
    if a.refit:
        amp = refit(s, h, mm)
    wq = (s.view(m, n // BS, BS) * amp[:, :, None]).reshape(m, n)
    wfin = (wq @ r.T if a.rot else wq).half().float()
    lin.weight.data = wfin
    num = den = 0.0
    for i in range(0, N, 8192):
        yc = yt[i:i + 8192]
        num += float(((yc - xq[i:i + 8192] @ wfin.T) ** 2).sum())
        den += float((yc ** 2).sum())
    return num / den


calib = chunks("~/seal/calib-c4.txt", a.calib_chunks)
B = 10
pos = torch.arange(CH, device=dev)[None].expand(B, -1)
with torch.no_grad():
    ht = teacher.model.embed_tokens(calib)
    hs = ht.clone()
    pe = teacher.model.rotary_emb(ht[:B], pos)
log = []
t0 = time.time()


def run_layer(layer, h):
    return layer(h, position_embeddings=pe, position_ids=pos)


for bi in range(a.blocks):
    tl, sl = teacher.model.layers[bi], student.model.layers[bi]
    groups = [
        ("attn_in", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
        ("attn_out", ["self_attn.o_proj"]),
        ("ffn_in", ["mlp.gate_proj", "mlp.up_proj"]),
        ("ffn_down", ["mlp.down_proj"]),
    ]
    for gname, names in groups:
        xs, ys = {k: [] for k in names}, {k: [] for k in names}
        for b0 in range(0, a.calib_chunks, B):
            cap = {}

            def hook(key, store):
                def f(mod, inp, out):
                    store[key] = (inp[0].detach(), out.detach())
                return f

            ht_b, hs_b = ht[b0:b0 + B], hs[b0:b0 + B]
            th, sh, tc, sc = [], [], {}, {}
            for k in names:
                th.append(dict(tl.named_modules())[k].register_forward_hook(hook(k, tc)))
                sh.append(dict(sl.named_modules())[k].register_forward_hook(hook(k, sc)))
            with torch.no_grad():
                run_layer(tl, ht_b)
                run_layer(sl, hs_b)
            for x in th + sh:
                x.remove()
            for k in names:
                xs[k].append((sc[k][0] if a.path == "q" else tc[k][0]).reshape(-1, tc[k][0].shape[-1]))
                ys[k].append(tc[k][1].reshape(-1, tc[k][1].shape[-1]))
        for k in names:
            lin = dict(sl.named_modules())[k]
            e = quantise_linear(lin, torch.cat(xs[k]), torch.cat(ys[k]))
            log.append({"block": bi, "tensor": k, "rel_err": e})
        del xs, ys
        torch.cuda.empty_cache()
    with torch.no_grad():
        for b0 in range(0, a.calib_chunks, B):
            ht[b0:b0 + B] = run_layer(tl, ht[b0:b0 + B])
            hs[b0:b0 + B] = run_layer(sl, hs[b0:b0 + B])
    print(json.dumps({"block": bi, "t": round(time.time() - t0, 1),
                      "rel_err": [round(r["rel_err"], 4) for r in log if r["block"] == bi]}), flush=True)

# paired KLD on wikitext-2 test
del ht, hs, calib, pe, pos
torch.cuda.empty_cache()
ev = chunks("~/gptq-disjoint/wikitext-2-raw/wiki.test.raw", a.eval_chunks)
kl, top = [], []
with torch.no_grad():
    for i in range(a.eval_chunks):
        x = ev[i:i + 1]
        lt = teacher(x).logits.float()
        ls = student(x).logits.float()
        lpt, lps = lt.log_softmax(-1), ls.log_softmax(-1)
        kl.append((lpt.exp() * (lpt - lps)).sum(-1).flatten())
        top.append((lt.argmax(-1) == ls.argmax(-1)).float().flatten())
kl = torch.cat(kl)
res = {"args": vars(a), "kld_mean": float(kl.mean()), "kld_sem": float(kl.std() / kl.numel() ** 0.5),
       "same_top": float(torch.cat(top).mean()), "tokens": int(kl.numel()), "per_tensor": log,
       "wall_s": round(time.time() - t0, 1)}
json.dump(res, open(a.out, "w"))
print(json.dumps({k: v for k, v in res.items() if k != "per_tensor"}), flush=True)
