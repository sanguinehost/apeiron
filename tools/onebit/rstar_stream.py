"""ONEBIT RSTAR-SCALE: layer-streamed all-in Q4_K_M-matching rate R* for models that do not fit on the GPU.

Same estimator as `converse.py` `run_scale` (K-FAC factors G, H per linear tensor from a sampled-target Fisher, eigenbases
U, V, C = U^T W V, component weight lam_ij = 1/2 g_i h_j r_i c_j / m, 16 side-information bits per row and column, Q4_K_M KL
= sum 1/2 tr(D^T G D H), global reverse water-filling by a log2(lam) histogram bisection), but the model is never resident:
  F_l    forward, block l: X_l -> X_{l+1} (fp32, on disk)
  T      final norm + lm_head, K sampled targets per chunk, gradient D_L w.r.t. X_L (fp32, on disk)
  A_l    block l recomputed from X_l with grad; G of q,k,v,o (autograd) and G of the block output (= G of down_proj),
         H of attn_in and o_in; D_l = gradient w.r.t. X_l; analysis of q,k,v,o
  GATE_l G_gate from D_{l+1} through the analytic SwiGLU backward; analysis of ffn_gate
  UP_l   G_up likewise; analysis of ffn_up
  DOWN_l H of down_in; analysis of ffn_down (G from A_l)
Each unit reduces its statistics at once (eigh, C, histogram of log2 lam, Q4 quadratic) and stores only the histogram and
scalars, so memory is bounded by one block and one 25600^2 matrix. An invocation runs units until `--budget` seconds are
spent and exits 0; rerun the same command until `DONE` (units resume from `stats/<unit>.npz`).

Precision: fp32 everywhere (no float64: not in torch, not in numpy, not in accumulators), TF32 off; stored activations are
fp32 as well. eigh of 25600^2 does not fit cusolver's 4 n^2 workspace on a 16 GB card: `eigh_managed` calls
cusolverDnXsyevd on cudaMallocManaged buffers (driver oversubscription; 17 s on the RTX 5080).

usage (repo root, ~/spinor-ttl/.venv/bin/python):
  tools/onebit/rstar_stream.py NAME HF_DIR GGUF_Q4 OUT.json CHUNKS K_SAMPLES [--gguf16 PATH] [--dscale X] [--budget S]
                               [--bs N] [--work DIR] [--check-hf]
Cost per layer at 8B/32B: Gram updates 2 x 4 x tokens x out^2 FLOP for gate/up; eigh 3 x 25600^2 at 32B (17 s each).
"""
import ctypes, json, math, os, sys, time, argparse, hashlib

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open

sys.path.insert(0, "~/llama.cpp/gguf-py")
import gguf
from gguf.quants import dequantize

CALIB = "~/seal/calib-c4.txt"
CHUNK = 512
EPS = 1e-30
LB0, LBW, LBN = 130.0, 0.02, 7500  # log2(lambda) histogram: lo edge -LB0, bin width, count
BIG = 16384  # eigh above this dimension goes through the managed-memory cusolver path
KINDS = [("attn_q", "self_attn.q_proj", "attn_in"), ("attn_k", "self_attn.k_proj", "attn_in"),
         ("attn_v", "self_attn.v_proj", "attn_in"), ("attn_output", "self_attn.o_proj", "o_in"),
         ("ffn_gate", "mlp.gate_proj", "mlp_in"), ("ffn_up", "mlp.up_proj", "mlp_in"),
         ("ffn_down", "mlp.down_proj", "down_in")]
KIND = {n: (p, k) for n, p, k in KINDS}
CUDART = "~/spinor-ttl/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
CUSOLVER = "~/spinor-ttl/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcusolver.so.12"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- eigh
_rt = _sol = None


def eigh_managed(M):
    """(w ascending, V columns) of symmetric fp32 cuda M via cusolverDnXsyevd on managed memory; consumes M's storage.
    Time: 17 s at n = 25600 (4 n^2 floats of workspace, paged by the driver)."""
    global _rt, _sol
    if _rt is None:
        _rt, _sol = ctypes.CDLL(CUDART), ctypes.CDLL(CUSOLVER)
    n = M.shape[0]
    nb = 4 * n * n
    dev = M.device

    def managed(nbytes):
        ptr = ctypes.c_void_p()
        assert _rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(1)) == 0
        return ptr.value

    h, p = ctypes.c_void_p(), ctypes.c_void_p()
    assert _sol.cusolverDnCreate(ctypes.byref(h)) == 0 and _sol.cusolverDnCreateParams(ctypes.byref(p)) == 0
    w = torch.empty(n, device=dev)
    info = torch.zeros(1, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    Am = managed(nb)
    assert _rt.cudaMemcpy(ctypes.c_void_p(Am), ctypes.c_void_p(M.data_ptr()), ctypes.c_size_t(nb), ctypes.c_int(4)) == 0
    M.untyped_storage().resize_(0)
    torch.cuda.empty_cache()
    dws, hws = ctypes.c_size_t(), ctypes.c_size_t()
    args = [h, p, ctypes.c_int(1), ctypes.c_int(0), ctypes.c_int64(n), ctypes.c_int(0), ctypes.c_void_p(Am), ctypes.c_int64(n),
            ctypes.c_int(0), ctypes.c_void_p(w.data_ptr()), ctypes.c_int(0)]
    assert _sol.cusolverDnXsyevd_bufferSize(*args, ctypes.byref(dws), ctypes.byref(hws)) == 0
    dbuf = managed(dws.value)
    hbuf = torch.empty(max(hws.value, 1), dtype=torch.uint8)
    r = _sol.cusolverDnXsyevd(*args, ctypes.c_void_p(dbuf), ctypes.c_size_t(dws.value), ctypes.c_void_p(hbuf.data_ptr()),
                              ctypes.c_size_t(hws.value), ctypes.c_void_p(info.data_ptr()))
    torch.cuda.synchronize()
    assert r == 0 and int(info) == 0, (r, int(info))
    _rt.cudaFree(ctypes.c_void_p(dbuf))
    V = torch.empty(n, n, device=dev)
    assert _rt.cudaMemcpy(ctypes.c_void_p(V.data_ptr()), ctypes.c_void_p(Am), ctypes.c_size_t(nb), ctypes.c_int(4)) == 0
    _rt.cudaFree(ctypes.c_void_p(Am))
    _sol.cusolverDnDestroyParams(p)
    _sol.cusolverDnDestroy(h)
    # column-major eigenvector k occupies Am[k n : (k + 1) n], i.e. row k of the row-major copy
    return w, V.t()


def eig(M):
    """Consumes M. Returns (eigenvalues ascending, eigenvectors as columns)."""
    if M.shape[0] > BIG:
        return eigh_managed(M)
    return torch.linalg.eigh(M)


# ---------------------------------------------------------------------------------------------------------------- model
class Streamer:
    def __init__(self, hf, n_chunks, k_samples, work, bs):
        self.hf, self.nc, self.K, self.work, self.bs = hf, n_chunks, k_samples, work, bs
        self.cfg = transformers.AutoConfig.from_pretrained(hf)
        assert self.cfg.hidden_act == "silu"
        self.L, self.H, self.I = self.cfg.num_hidden_layers, self.cfg.hidden_size, self.cfg.intermediate_size
        idx = os.path.join(hf, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wmap = json.load(open(idx))["weight_map"]
        else:
            with safe_open(os.path.join(hf, "model.safetensors"), "pt") as f:
                self.wmap = {k: "model.safetensors" for k in f.keys()}
        self._files = {}
        with torch.device("meta"):
            m = transformers.AutoModelForCausalLM.from_config(self.cfg, attn_implementation="sdpa", dtype=torch.float32)
        self.layer = m.model.layers[0].to_empty(device="cuda")
        for p in self.layer.parameters():
            p.requires_grad_(False)
        self.rot = type(m.model.rotary_emb)(config=self.cfg, device="cuda")
        self.fnorm = type(m.model.norm)(self.H, eps=self.cfg.rms_norm_eps).cuda()
        for p in self.fnorm.parameters():
            p.requires_grad_(False)
        pos = torch.arange(CHUNK, device="cuda")[None]
        self.pe = self.rot(torch.zeros(1, device="cuda"), pos)
        self.loaded = None

    def get(self, name):
        fn = self.wmap[name]
        if fn not in self._files:
            self._files[fn] = safe_open(os.path.join(self.hf, fn), "pt")
        return self._files[fn].get_tensor(name)

    def load_layer(self, l):
        if self.loaded == l:
            return
        for name, p in self.layer.state_dict().items():
            p.copy_(self.get(f"model.layers.{l}.{name}").cuda().float())
        self.loaded = l

    def weight(self, l, path):
        m = self.layer
        for q in path.split("."):
            m = getattr(m, q)
        return m.weight.detach()

    def mid(self, x):
        """h_mid = x + attn(norm(x)) (no grad) and the MLP input m = post_attention_layernorm(h_mid)."""
        L = self.layer
        a, _ = L.self_attn(hidden_states=L.input_layernorm(x), attention_mask=None, position_embeddings=self.pe)
        h = x + a
        return h, L.post_attention_layernorm(h)

    # on-disk activations
    def xp(self, l):
        return os.path.join(self.work, f"X_{l}.f32")

    def dp(self, l):
        return os.path.join(self.work, f"D_{l}.f32")

    def xmap(self, l, mode):
        return np.memmap(self.xp(l), dtype=np.float32, mode=mode, shape=(self.nc, CHUNK, self.H))

    def dmap(self, l, mode):
        return np.memmap(self.dp(l), dtype=np.float32, mode=mode, shape=(self.K, self.nc, CHUNK, self.H))


def batches(nc, bs):
    return [(c, min(c + bs, nc)) for c in range(0, nc, bs)]


# ---------------------------------------------------------------------------------------------------------------- units
def unit_forward(S, l, ids):
    if l == 0:
        emb = S.get("model.embed_tokens.weight")
        toks = torch.tensor(ids[:S.nc * CHUNK]).reshape(S.nc, CHUNK)
        x0 = S.xmap(0, "w+")
        for c in range(S.nc):
            x0[c] = emb[toks[c]].float().numpy()
        x0.flush()
        del x0, emb
    S.load_layer(l)
    xi, xo = S.xmap(l, "r"), S.xmap(l + 1, "w+")
    with torch.no_grad():
        for a, b in batches(S.nc, S.bs):
            x = torch.from_numpy(np.ascontiguousarray(xi[a:b])).cuda()
            xo[a:b] = S.layer(x, position_embeddings=S.pe).cpu().numpy()
    xo.flush()


def unit_top(S, check_hf):
    L = S.L
    S.fnorm.weight.copy_(S.get("model.norm.weight").cuda().float())
    lm = S.get("lm_head.weight") if "lm_head.weight" in S.wmap else S.get("model.embed_tokens.weight")
    lm = lm.cuda().float()
    xi = S.xmap(L, "r")
    D = S.dmap(L, "w+")
    for c in range(S.nc):
        x = torch.from_numpy(np.ascontiguousarray(xi[c:c + 1])).cuda().requires_grad_(True)
        logp = torch.log_softmax(F.linear(S.fnorm(x), lm)[0], -1)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1000 + c)
        for k in range(S.K):
            s = torch.multinomial(logp.exp().detach(), 1, generator=gen)
            g, = torch.autograd.grad(-logp.gather(1, s).sum(), x, retain_graph=k < S.K - 1)
            D[k, c] = g[0].cpu().numpy()
        if check_hf and c == 0:
            nb = sum(os.path.getsize(os.path.join(S.hf, f)) for f in set(S.wmap.values()))  # bytes of stored weights
            toks = torch.tensor([check_hf[:CHUNK]])
            if nb * 2 < 4e9:  # fp32 reference on the GPU
                hfm = transformers.AutoModelForCausalLM.from_pretrained(S.hf, dtype=torch.float32).cuda().eval()
                ref, n, tag = torch.log_softmax(hfm(toks.cuda()).logits[0].float(), -1), CHUNK, "fp32 gpu"
            elif nb < 20e9:  # bf16 reference on the CPU, first 128 tokens (causal prefix): a gross-defect check, not a fp32 match
                hfm = transformers.AutoModelForCausalLM.from_pretrained(S.hf, dtype=torch.bfloat16).eval()
                with torch.no_grad():
                    ref, n, tag = torch.log_softmax(hfm(toks[:, :128]).logits[0].float(), -1).cuda(), 128, "bf16 cpu"
            else:
                hfm = None
                log("check-hf: skipped (reference does not fit)")
            if hfm is not None:
                d = (ref - logp.detach()[:n]).abs()
                log(f"check-hf ({tag}, {n} tokens): max |logp_stream - logp_hf| =", float(d.max()), "mean |.| =", float(d.mean()))
                del hfm, ref, d
            torch.cuda.empty_cache()
    D.flush()


class Hook:
    """Capture forward inputs and outputs of the four attention projections."""

    def __init__(self, S):
        self.o, self.i, self.h = {}, {}, []
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            m = S.layer
            for q in KIND[name][0].split("."):
                m = getattr(m, q)
            self.h.append(m.register_forward_hook(self.cap(name)))

    def cap(self, name):
        def f(mod, a, out):
            self.o[name] = out
            self.i[name] = a[0]
        return f

    def clear(self):
        self.o.clear()
        self.i.clear()

    def remove(self):
        for h in self.h:
            h.remove()


def gram(acc, a):
    a = a.reshape(-1, a.shape[-1])
    acc.addmm_(a.t(), a)


def unit_A(S, l, ctx):
    S.load_layer(l)
    H, nt = S.H, S.nc * CHUNK
    names = ("attn_q", "attn_k", "attn_v", "attn_output")
    G = {n: torch.zeros(S.weight(l, KIND[n][0]).shape[0], S.weight(l, KIND[n][0]).shape[0], device="cuda") for n in names}
    Gd = torch.zeros(H, H, device="cuda")
    Hs = {"attn_in": torch.zeros(H, H, device="cuda"),
          "o_in": torch.zeros(S.weight(l, "self_attn.o_proj").shape[1], S.weight(l, "self_attn.o_proj").shape[1], device="cuda")}
    hook = Hook(S)
    xi, Din = S.xmap(l, "r"), S.dmap(l + 1, "r")
    Dout = S.dmap(l, "w+") if l > 0 else None
    for a, b in batches(S.nc, S.bs):
        x = torch.from_numpy(np.ascontiguousarray(xi[a:b])).cuda().requires_grad_(True)
        y = S.layer(x, position_embeddings=S.pe)
        gram(Hs["attn_in"], hook.i["attn_q"].detach())
        gram(Hs["o_in"], hook.i["attn_output"].detach())
        outs = [hook.o[n] for n in names] + [x]
        for k in range(S.K):
            dy = torch.from_numpy(np.ascontiguousarray(Din[k, a:b])).cuda()
            gr = torch.autograd.grad(y, outs, grad_outputs=dy, retain_graph=k < S.K - 1)
            for n, g in zip(names, gr[:4]):
                gram(G[n], g)
            gram(Gd, dy)
            if Dout is not None:
                Dout[k, a:b] = gr[4].cpu().numpy()
        hook.clear()
        del y, outs, gr
    hook.remove()
    if Dout is not None:
        Dout.flush()
    for v in G.values():
        v.div_(nt * S.K)
    Gd.div_(nt * S.K)
    for v in Hs.values():
        v.div_(nt)
    np.save(os.path.join(S.work, f"Gdown_{l}.npy"), Gd.cpu().numpy())
    del Gd
    stats = new_stats()
    ecache = {}
    for n in names:
        key = KIND[n][1]
        Hm = Hs[key]
        if key in ecache:
            h, V = ecache[key]
        else:
            h, V = eig(Hm.clone())
            ecache[key] = (h, V)
        analyze(ctx, stats, l, n, G.pop(n), Hm, h, V)
    return stats


def mlp_batches(S, l, with_grad_inputs):
    """Yield (a, b, m, gate, up) for chunk batches; m = MLP input, gate/up the pre-activation projections (no grad)."""
    xi = S.xmap(l, "r")
    L = S.layer
    for a, b in batches(S.nc, S.bs):
        x = torch.from_numpy(np.ascontiguousarray(xi[a:b])).cuda()
        _, m = S.mid(x)
        yield a, b, m, L.mlp.gate_proj(m), L.mlp.up_proj(m)


def unit_GU(S, l, ctx, which):
    S.load_layer(l)
    nt = S.nc * CHUNK
    name = "ffn_gate" if which == "gate" else "ffn_up"
    Hm = torch.zeros(S.H, S.H, device="cuda")
    G = torch.zeros(S.I, S.I, device="cuda")
    Wd = S.weight(l, "mlp.down_proj")  # (H, I)
    with torch.no_grad():
        Din = S.dmap(l + 1, "r")
        for a, b, m, gate, up in mlp_batches(S, l, False):
            gram(Hm, m)
            sg = torch.sigmoid(gate)
            silu = gate * sg
            for k in range(S.K):
                dy = torch.from_numpy(np.ascontiguousarray(Din[k, a:b])).cuda().reshape(-1, S.H)
                dact = dy @ Wd
                if which == "gate":
                    d = dact * up.reshape(-1, S.I) * (sg * (1 + gate * (1 - sg))).reshape(-1, S.I)
                else:
                    d = dact * silu.reshape(-1, S.I)
                G.addmm_(d.t(), d)
            del sg, silu, gate, up, m
    G.div_(nt * S.K)
    Hm.div_(nt)
    stats = new_stats()
    h, V = eig(Hm.clone())
    analyze(ctx, stats, l, name, G, Hm, h, V)
    return stats


def unit_down(S, l, ctx):
    S.load_layer(l)
    nt = S.nc * CHUNK
    Hd = torch.zeros(S.I, S.I, device="cuda")
    with torch.no_grad():
        for a, b, m, gate, up in mlp_batches(S, l, False):
            act = F.silu(gate) * up
            gram(Hd, act)
            del act, gate, up, m
    Hd.div_(nt)
    Gd = torch.from_numpy(np.load(os.path.join(S.work, f"Gdown_{l}.npy"))).cuda()
    stats = new_stats()
    # Hd is consumed by the eigendecomposition; the Q4 quadratic needs it first (analyze takes both)
    analyze(ctx, stats, l, "ffn_down", Gd, Hd, None, None)
    return stats


# ---------------------------------------------------------------------------------------------------------------- analysis
def new_stats():
    return {"cnt": torch.zeros(LBN, dtype=torch.int64, device="cuda"), "dsum": torch.zeros(LBN, dtype=torch.float32, device="cuda"),
            "sumlog": np.float32(0), "n_sel": 0, "side_bits": 0, "kq4": {}, "max_rel": 0.0, "max_dq": 0.0}


CEN = None


def hist_add(stats, lam):
    """Add the elements of lam (fp32 vector) to the log2 histogram: counts, and the sum of lam/centre - 1 per bin."""
    global CEN
    if CEN is None:
        CEN = torch.exp2((torch.arange(LBN, device="cuda", dtype=torch.float32) + 0.5) * LBW - LB0)
    lam = lam.clamp(min=EPS)
    idx = ((torch.log2(lam) + LB0) / LBW).long().clamp(0, LBN - 1)
    stats["cnt"] += torch.bincount(idx, minlength=LBN)
    dev = lam / CEN[idx] - 1
    s = torch.zeros(LBN, dtype=torch.float32, device="cuda")
    s.index_add_(0, idx, dev)
    stats["dsum"] += s


def analyze(ctx, stats, l, name, Gm, Hm, h, V):
    """One tensor: Q4 quadratic, eigenbases, C, sumlog, histogram. Consumes Gm (and Hm when h is None)."""
    S, rq4, r16, dscale = ctx["S"], ctx["rq4"], ctx["r16"], ctx["dscale"]
    path = KIND[name][0]
    w = S.weight(l, path)
    gname = f"blk.{l}.{name}.weight"
    dq = (read(rq4, gname).cuda() - w) * dscale
    stats["max_dq"] = max(stats["max_dq"], float(dq.norm() / w.norm()) / dscale)
    if r16 is not None:
        stats["max_rel"] = max(stats["max_rel"], float((read(r16, gname).cuda() - w).norm() / w.norm()))
    k4 = 0.5 * float(((Gm @ dq) * (dq @ Hm)).sum())
    del dq
    stats["kq4"][name] = np.float32(k4)
    rows, cols = w.shape
    g, U = eig(Gm)
    del Gm
    if h is None:
        h, V = eig(Hm)
        del Hm
    g, h = g.clamp(min=EPS), h.clamp(min=EPS)
    C2 = ((U.t() @ w) @ V) ** 2
    del U
    m = C2.mean()
    r0, c0 = C2.mean(1), C2.mean(0)
    del C2
    r, c = r0.clamp(min=EPS), c0.clamp(min=EPS)
    sl = (cols * (torch.log(g / 2) + torch.log(r)).sum() + rows * (torch.log(h) + torch.log(c)).sum()
          - rows * cols * torch.log(m))
    stats["sumlog"] = np.float32(stats["sumlog"] + np.float32(sl.item()))
    step = max(1, (1 << 25) // cols)
    for i in range(0, rows, step):
        sep = (r0[i:i + step, None] * c0[None, :] / m).clamp(min=EPS)
        hist_add(stats, (0.5 * g[i:i + step, None] * h[None, :] * sep).reshape(-1))
    stats["n_sel"] += rows * cols
    stats["side_bits"] += 16 * (rows + cols)
    torch.cuda.empty_cache()


def read(reader, name):
    for t in reader.tensors:
        if t.name == name:
            shape = [int(s) for s in reversed(t.shape)]
            if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                a = np.asarray(t.data, dtype=np.float32)
            else:
                a = dequantize(t.data, t.tensor_type)
            return torch.from_numpy(np.ascontiguousarray(a.reshape(shape))).float()
    raise KeyError(name)


def save_stats(path, st):
    kinds = sorted(st["kq4"])
    np.savez(path + ".tmp.npz", cnt=st["cnt"].cpu().numpy(), dsum=st["dsum"].cpu().numpy(), sumlog=st["sumlog"],
             n_sel=np.int64(st["n_sel"]), side_bits=np.int64(st["side_bits"]), kinds=np.array(kinds),
             kq4=np.array([st["kq4"][k] for k in kinds], dtype=np.float32), max_rel=np.float32(st["max_rel"]),
             max_dq=np.float32(st["max_dq"]))
    os.replace(path + ".tmp.npz", path)


# ---------------------------------------------------------------------------------------------------------------- aggregate
def aggregate(S, work, out, args, walls):
    files = sorted(os.listdir(os.path.join(work, "stats")))
    cnt = torch.zeros(LBN, dtype=torch.int64)
    dsum = torch.zeros(LBN, dtype=torch.float32)
    sumlog, n_sel, side, max_rel, max_dq = np.float32(0), 0, 0, 0.0, 0.0
    per_kind = {}
    kq = []
    for f in files:
        z = np.load(os.path.join(work, "stats", f))
        cnt += torch.from_numpy(z["cnt"])
        dsum += torch.from_numpy(z["dsum"])
        sumlog = np.float32(sumlog + z["sumlog"])
        n_sel += int(z["n_sel"])
        side += int(z["side_bits"])
        max_rel, max_dq = max(max_rel, float(z["max_rel"])), max(max_dq, float(z["max_dq"]))
        for k, v in zip(z["kinds"], z["kq4"]):
            per_kind[str(k)] = np.float32(per_kind.get(str(k), np.float32(0)) + v)
            kq.append(v)
    kq4 = float(torch.tensor(kq, dtype=torch.float32).sum())
    sumlog = float(sumlog)
    nl = sum(S.weight(0, p).numel() for _, p, _ in KINDS)
    N_full = nl * S.L
    layers = list(range(S.L))
    frac = 1.0
    K0 = N_full * math.exp(sumlog / n_sel)
    side_bpw = side / n_sel
    kq4_full = kq4 / frac
    res = {"model": S.hf, "dtype": "fp32", "layers": layers, "n_layers": S.L, "n_chunks": S.nc, "k_samples": S.K,
           "N_sample": n_sel, "N_full": N_full, "side_bpw": side_bpw, "max_rel_gguf_vs_hf": max_rel, "max_dq_over_w": max_dq,
           "mean_log_lam": sumlog / n_sel, "K0": K0, "K0_over_N": K0 / N_full, "kl_q4_sample": kq4, "kl_q4_full_est": kq4_full,
           "kl_q4_by_kind_sample": {k: float(v) for k, v in per_kind.items()}, "dscale": args.dscale,
           "R_star_net": math.log(K0 / kq4_full, 4), "R_star_allin": math.log(K0 / kq4_full, 4) + side_bpw}
    cen = torch.exp2((torch.arange(LBN, dtype=torch.float32) + 0.5) * LBW - LB0)
    ok = cnt > 0
    cn = cnt[ok].float()
    lb = cen[ok] * (1 + dsum[ok] / cn)
    log2l = torch.log2(lb)

    def at(th):
        kl = float((cn * torch.clamp(lb, max=th)).sum())
        rate = float((cn * torch.clamp(0.5 * (log2l - math.log2(th)), min=0)).sum())
        return kl, rate

    def theta_rate(budget):
        lo, hi = np.float32(math.log(float(lb.min())) - 50), np.float32(math.log(float(lb.max())))
        for _ in range(100):
            mid = np.float32((lo + hi) / 2)
            lo, hi = (mid, hi) if at(float(np.exp(mid)))[1] > budget else (lo, mid)
        return float(np.exp(hi))

    scale = N_full / n_sel
    res["hist_bisection"] = {}
    for rr in (1.18, 3.0):
        th = theta_rate(rr * n_sel - side)
        kl, _ = at(th)
        res["hist_bisection"][str(rr)] = {"theta": th, "kl_full_est": kl * scale}
    lo, hi = np.float32(math.log(float(lb.min())) - 50), np.float32(math.log(float(lb.max())))
    for _ in range(100):
        mid = np.float32((lo + hi) / 2)
        lo, hi = (mid, hi) if at(float(np.exp(mid)))[0] < kq4 else (lo, mid)
    th_q4 = float(np.exp(lo))
    kl, rate = at(th_q4)
    res["hist_bisection"]["q4_match"] = {"theta": th_q4, "kl_sample": kl, "rate_net_bpw": rate / n_sel,
                                         "rate_allin_bpw": rate / n_sel + side_bpw}
    res["unit_walls_s"] = walls
    res["wall_s"] = sum(walls.values())
    res["script_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    json.dump(res, open(out, "w"), indent=1)
    log(json.dumps({k: res[k] for k in ("K0_over_N", "kl_q4_sample", "R_star_allin", "wall_s")}),
        "q4_match allin:", res["hist_bisection"]["q4_match"]["rate_allin_bpw"])


# ---------------------------------------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"), ap.add_argument("hf"), ap.add_argument("ggufq4"), ap.add_argument("out")
    ap.add_argument("chunks", type=int), ap.add_argument("k", type=int)
    ap.add_argument("--gguf16"), ap.add_argument("--dscale", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=500.0), ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--work"), ap.add_argument("--check-hf", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t_start = time.time()
    work = args.work or f"~/rstar/work-{args.name}"
    os.makedirs(os.path.join(work, "stats"), exist_ok=True)
    os.makedirs(os.path.join(work, "done"), exist_ok=True)
    if not os.path.isdir(args.hf):
        from huggingface_hub import snapshot_download
        args.hf = snapshot_download(args.hf, local_files_only=True)
    S = Streamer(args.hf, args.chunks, args.k, work, args.bs)
    tok = transformers.AutoTokenizer.from_pretrained(args.hf)
    ids = tok.encode(open(CALIB, encoding="utf-8").read())
    assert len(ids) >= args.chunks * CHUNK
    ctx = {"S": S, "rq4": gguf.GGUFReader(args.ggufq4), "r16": gguf.GGUFReader(args.gguf16) if args.gguf16 else None,
           "dscale": args.dscale}
    L = S.L
    units = [f"F{l}" for l in range(L)] + ["T"]
    for l in range(L - 1, -1, -1):
        units += [f"A{l}", f"GATE{l}", f"UP{l}", f"DOWN{l}"]
    walls_path = os.path.join(work, "walls.json")
    walls = json.load(open(walls_path)) if os.path.exists(walls_path) else {}
    biggest = max(walls.values(), default=0.0)
    for u in units:
        mark = os.path.join(work, "done", u)
        if os.path.exists(mark):
            continue
        if time.time() - t_start + 1.3 * biggest > args.budget:
            log("budget reached before", u)
            return
        t0 = time.time()
        kind, l = (u.rstrip("0123456789"), int(u[len(u.rstrip("0123456789")):] or -1))
        if kind == "F":
            unit_forward(S, l, ids)
        elif kind == "T":
            unit_top(S, args.check_hf and ids)
        else:
            if kind == "A":
                st = unit_A(S, l, ctx)
            elif kind == "GATE":
                st = unit_GU(S, l, ctx, "gate")
            elif kind == "UP":
                st = unit_GU(S, l, ctx, "up")
            elif kind == "DOWN":
                st = unit_down(S, l, ctx)
                for p in (S.dp(l + 1), os.path.join(work, f"Gdown_{l}.npy"), S.xp(l + 1)):
                    if os.path.exists(p):
                        os.remove(p)
            if st is not None:
                save_stats(os.path.join(work, "stats", f"{u}.npz"), st)
        open(mark, "w").close()
        walls[u] = time.time() - t0
        biggest = max(biggest, walls[u])
        json.dump(walls, open(walls_path, "w"))
        log("unit", u, f"{walls[u]:.1f}s", f"gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak")
        torch.cuda.reset_peak_memory_stats()
    aggregate(S, work, args.out, args, walls)
    open(os.path.join(work, "DONE"), "w").close()
    log("DONE")


if __name__ == "__main__":
    main()
