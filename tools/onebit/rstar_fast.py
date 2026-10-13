"""ONEBIT RSTAR-FAST: one-process all-in Q4_K_M-matching rate R* (same estimator as `rstar_stream.py`).

Estimator (unchanged from rstar_stream.py / converse.py run_scale): K-FAC factors G (output side) and H (input side) per
linear tensor from a sampled-target Fisher (100 chunks x 512 tokens, K=4 targets), eigenbases U, V, C = U^T W V, component
weight lam_ij = 1/2 g_i h_j r_i c_j / m, 16 side-information bits per row and column, Q4_K_M KL = sum 1/2 tr(D^T G D H)
with D = dequant(Q4_K_M) - W, global reverse water-filling by a log2(lam) histogram bisection (LB0=130, LBW=0.02, LBN=7500).

What differs from rstar_stream.py (each one is a measured control, see CONTROLS below):
  1. No staging, no resume: one process, bf16 weights streamed per layer from safetensors, activations X_l in host RAM as bf16 (one tensor per layer,
     freed when its backward step has read it), K x gradients D in pinned host memory.
  2. No llama.cpp: the Q4_K_M reference is built here. imatrix = diag(H) (llama's imatrix is the per-column mean x^2),
     quantised and dequantised by ggml_quantize_chunk / type_traits.to_float through ctypes (libggml-base.so), the
     Q4_K_M per-tensor type rule (`qtype`) reimplemented, rows split over a thread pool, pipelined one layer ahead of the
     GPU. Control Q1 (`--check-quant`): bit-exact against the llama-quantize output of the same f16 weights and imatrix.
  3. One fused backward per layer (one forward with grad, K autograd backwards, hooks on all six linears) replaces the
     four units A, GATE, UP, DOWN, each of which recomputed the block.
  4. r_i, c_j from the identities r_i = |u_i^T W|^2 / cols and c_j = |W v_j|^2 / rows (V and U orthogonal), so the
     rows x cols product U^T W V is never formed; q,k,v share the eigendecomposition of H_attn_in, gate,up of H_mlp_in.
  5. The histogram is built from the two marginals (log2 a_i, log2 b_j with lam_ij = a_i b_j) on a grid LBW/4 and pair
     counts, not from rows x cols elements: same bins, same per-bin mean lam, bin assignment of a pair off by < LBW/4.
  6. Precision knobs (`--fb`, `--gram`, `--aux`): model forward/backward fp32|tf32|bf16, Gram accumulation
     fp32|tf32|fp16 with per-column scaling (`fp16s`, fp32 accumulate), K x kq4 products fp32|tf32. No float64 anywhere.

usage (repo root; python = /home/socol/Workspace/spinor-ttl/.venv/bin/python; GPU commands go through
`flock /tmp/opencode/gpu.lock systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 timeout 570 ...`):
  tools/onebit/rstar_fast.py NAME HF_DIR OUT.json [--chunks 100] [--k 4] [--fb bf16] [--gram fp16s] [--aux tf32]
                             [--bs 4] [--eig slq|exact] [--slq-min 2000] [--nv 32] [--lanczos 100]
                             [--only L,L,..] [--dump DIR] [--dump-gram]
  tools/onebit/rstar_fast.py --check-quant HF_DIR REF_Q4_GGUF IMATRIX_GGUF N_LAYERS layer:kind:hf.module.name ...
Defaults: --eig slq replaces cusolver eigh by stochastic Lanczos quadrature for every matrix with n > --slq-min
(Gaussian probes, full reorthogonalisation, fp32, CPU batched tridiagonal eigh, CDF at node midpoints, log-theta
interpolation to n quantiles, trace rescale). Without eigenvectors the alignments are r_i = c_j = mean(W^2) (SLQ branch);
`--eig exact` keeps eigenvectors and the r, c identities of item 4.

COST (per layer l; n_* are the matrix dimensions; seconds on the RTX 5080, measured on Qwen3-8B unless noted):
  grams    2 T_g sum_t n_t^2, T_g = chunks*512*(K for G, 1 for H): fp16s/bf16 mm 120 TFLOP/s (bench1.log), fp32 40, tf32 58
  fwd+bwd  (1 + K) * 2 * P_l * chunks*512 (P_l = parameters of the block; dgrad only): 2.0 s/layer at 8B (~47 TFLOP/s bf16)
  kq4      2 (rows^2 cols + cols^2 rows) per tensor, fp32 (6.0 s total over 36 layers at 8B)
  r,c      2 (rows^2 cols + cols^2 rows) per tensor, only on the exact branch
  eig      exact: cusolver syevd 0.18 s (4864), 1.03 s (8192), 2.30 s (12288), 17 s (25600); CPU eigvalsh 7-8 s (12288);
           SLQ: 2 n^2 nv steps FLOP, 1.3 s/layer at 8B (3 matrices of 12288)
  hist     marginal pair enumeration, O(occupied bins^2) per tensor
  quant    CPU, 24 threads, ~240 M elements/s, overlapped with the GPU
Host RAM: bf16 X_l, chunks*512*hidden*2 bytes per layer, all layers resident until their backward step.

PROJECTION (measured on Qwen3-8B, 36 layers, wall 212 s: bwd.compute 133 s = 36 x (chain 2.0 + Gram 1.7), eig 47 s,
kq4 6 s, forward 13 s, top 7 s). Qwen3-32B per layer, from the FLOP scaling of each term: chain ~5.0 s, Gram ~6.8 s,
eig ~2.5 s, kq4 ~1.0 s, other ~0.3 s = ~15.6 s x 64 layers = ~1000 s, plus ~80 s fixed: ~1080 s. Derived, not measured at 32B.
Chain and Gram (~11.8 s/layer) set the floor at 100 chunks; X for 100 chunks at 32B (33.5 GB bf16) plus D (8.4 GB)
exceeds the 40G cap, so 32B needs fewer chunks or an offload tier.

CONTROLS (8B receipts in /home/socol/oc-tmp/escalation/2026-10-11-rstar-fast-*.json):
  Q1  --check-quant: Q4_K_M bytes identical to llama-quantize (0.5B incl. Q5_0/Q8_0/Q6_K fallbacks: 0 differing blocks).
  --eig exact, all fp32: 0.5B R* 3.17309 against the reference 3.1731.
  Reductions that leave tolerance at 8B (reference 2.2546): --only even layers -0.104, --only l%4==1 +0.081,
  --chunks 50 -0.223 (kl_q4 +5%), 50 chunks + even layers worse.
"""
import contextlib, ctypes, hashlib, json, math, os, sys, threading, time, argparse
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F

CALIB = "/home/socol/oc-tmp/seal/calib-c4.txt"
GGML_SO = os.environ.get("GGML_BASE_SO", "/home/socol/oc-tmp/converse/lbuild/bin/libggml-base.so")
CUDART = "/home/socol/Workspace/spinor-ttl/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
CUSOLVER = "/home/socol/Workspace/spinor-ttl/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcusolver.so.12"
CHUNK = 512
EPS = 1e-30
LB0, LBW, LBN = 130.0, 0.02, 7500  # log2(lambda) histogram: lo edge -LB0, bin width, count
SUB = 4  # marginal grid = LBW / SUB
BIG = 16384  # eigh above this dimension goes through the managed-memory cusolver path
# (kind, module path, H key)
KINDS = [("attn_q", "self_attn.q_proj", "attn_in"), ("attn_k", "self_attn.k_proj", "attn_in"),
         ("attn_v", "self_attn.v_proj", "attn_in"), ("attn_output", "self_attn.o_proj", "o_in"),
         ("ffn_gate", "mlp.gate_proj", "mlp_in"), ("ffn_up", "mlp.up_proj", "mlp_in"),
         ("ffn_down", "mlp.down_proj", "down_in")]
KIND = {n: (p, k) for n, p, k in KINDS}
GK = {"attn_q": "q", "attn_k": "k", "attn_v": "v", "attn_output": "o", "ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Timer:
    def __init__(self):
        self.t = {}

    @contextlib.contextmanager
    def __call__(self, name, sync=True):
        if sync:
            torch.cuda.synchronize()
        t0 = time.time()
        yield
        if sync:
            torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + time.time() - t0


# ---------------------------------------------------------------------------------------------------------------- ggml
class Traits(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("blck", ctypes.c_int64), ("blck_il", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                ("is_q", ctypes.c_bool), ("to_float", ctypes.c_void_p), ("from_float_ref", ctypes.c_void_p)]


class Ggml:
    """Q4_K_M reference builder: quantize (with imatrix) then dequantize, rows split over a thread pool."""

    def __init__(self, nthr=24):
        self.lib = ctypes.CDLL(GGML_SO)
        L = self.lib
        L.ggml_quantize_chunk.restype = ctypes.c_size_t
        L.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_void_p]
        L.ggml_row_size.restype = ctypes.c_size_t
        L.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        L.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
        L.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.nthr = nthr
        self.pool = ThreadPoolExecutor(nthr)
        self.to_float = {}

    def tofloat(self, t):
        if t not in self.to_float:
            tr = self.lib.ggml_get_type_traits(t).contents
            self.to_float[t] = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(tr.to_float)
        return self.to_float[t]

    def qdq(self, W, t, im, want_q=False):
        """W (rows, cols) fp32 C-contiguous numpy, im (cols,) fp32. Returns (dequantised fp32 (rows, cols), quantised bytes|None)."""
        rows, cols = W.shape
        rs = self.lib.ggml_row_size(t, cols)
        q = np.empty(rows * rs, dtype=np.uint8)
        dq = np.empty((rows, cols), dtype=np.float32)
        tf = self.tofloat(t)
        step = (rows + self.nthr - 1) // self.nthr

        def job(a):
            n = min(step, rows - a)
            self.lib.ggml_quantize_chunk(t, W.ctypes.data + a * cols * 4, q.ctypes.data + a * rs, 0, n, cols, im.ctypes.data)
            tf(q.ctypes.data + a * rs, dq.ctypes.data + a * cols * 4, n * cols)

        list(self.pool.map(job, range(0, rows, step)))
        return dq, (q if want_q else None)


def qtype(kind, l, L, cols):
    """llama_tensor_get_type for LLAMA_FTYPE_MOSTLY_Q4_K_M on the seven block linears: Q4_K, with attn_v and ffn_down Q6_K
    when use_more_bits(l, L); Q4_K -> Q5_0 and Q6_K -> Q8_0 when cols % 256 != 0. ggml enum: Q5_0=6, Q8_0=8, Q4_K=12, Q6_K=14."""
    more = l < L // 8 or l >= 7 * L // 8 or (l - L // 8) % 3 == 2
    t = 14 if (kind in ("attn_v", "ffn_down") and more) else 12
    if cols % 256:
        t = {12: 6, 14: 8}[t]
    return t


# ---------------------------------------------------------------------------------------------------------------- eigh
_rt = _sol = None


def eigh_managed(M):
    """(w ascending, V columns) of symmetric fp32 cuda M via cusolverDnXsyevd on managed memory; consumes M's storage.
    Copied from rstar_stream.py (17 s at n = 25600; 4 n^2 floats of workspace, paged by the driver)."""
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
    return w, V.t()


def eig(M):
    """Consumes M. Returns (eigenvalues ascending, eigenvectors as columns)."""
    if M.shape[0] > BIG:
        return eigh_managed(M)
    return torch.linalg.eigh(M)


# ---------------------------------------------------------------------------------------------------------------- grams
@contextlib.contextmanager
def tf32(on):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class Gram:
    """acc += a^T a over token batches. fp32 | tf32 | fp16s (per-column scale fixed from the first batch, fp16 operands,
    fp32 accumulate: G_ij = acc_ij / (s_i s_j))."""

    def __init__(self, n, mode):
        self.n, self.mode = n, mode
        self.acc = torch.zeros(n, n, device="cuda")
        self.s = None

    def add(self, a):
        a = a.reshape(-1, self.n)
        if self.mode == "fp16s":
            if self.s is None:
                rms = a.float().pow(2).mean(0).sqrt()
                self.s = 1.0 / rms.clamp(min=float(rms.median()) * 1e-6 + 1e-30)
            a16 = (a.float() * self.s).half()
            self.acc.add_(torch.mm(a16.t(), a16, out_dtype=torch.float32))
        else:
            a = a.float()
            with tf32(self.mode == "tf32"):
                self.acc.addmm_(a.t(), a)

    def result(self, denom):
        r = self.acc
        if self.mode == "fp16s" and self.s is not None:
            r.div_(self.s[:, None]).div_(self.s[None, :])
        r.div_(denom)
        assert bool(torch.isfinite(r).all()), "non-finite Gram (fp16 overflow): use --gram tf32"
        self.acc = None
        return r


# ---------------------------------------------------------------------------------------------------------------- model
class Net:
    def __init__(self, hf, nc, K, bs, fb):
        import transformers
        from safetensors import safe_open
        self.safe_open = safe_open
        self.hf, self.nc, self.K, self.bs, self.fb = hf, nc, K, bs, fb
        self.xs = {}
        self.cfg = transformers.AutoConfig.from_pretrained(hf)
        assert self.cfg.hidden_act == "silu"
        self.L, self.H, self.I = self.cfg.num_hidden_layers, self.cfg.hidden_size, self.cfg.intermediate_size
        idx = os.path.join(hf, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wmap = json.load(open(idx))["weight_map"]
        else:
            with safe_open(os.path.join(hf, "model.safetensors"), "pt") as f:
                self.wmap = {k: "model.safetensors" for k in f.keys()}
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
        self.names = list(self.layer.state_dict().keys())

    def get(self, name):
        with self.safe_open(os.path.join(self.hf, self.wmap[name]), "pt") as f:
            return f.get_tensor(name)

    def read_layer(self, l):
        """bf16 (or stored dtype) CPU tensors of block l, keyed by state_dict name. Thread-safe."""
        with contextlib.ExitStack() as es:
            files = {}
            out = {}
            for n in self.names:
                full = f"model.layers.{l}.{n}"
                fn = self.wmap[full]
                if fn not in files:
                    files[fn] = es.enter_context(self.safe_open(os.path.join(self.hf, fn), "pt"))
                out[n] = files[fn].get_tensor(full)
        return out

    def put_layer(self, cpu):
        for n, p in self.layer.state_dict().items():
            p.copy_(cpu[n].cuda(non_blocking=True).float())

    def module(self, path):
        m = self.layer
        for q in path.split("."):
            m = getattr(m, q)
        return m

    def weight(self, kind):
        return self.module(KIND[kind][0]).weight.detach()

    def fbctx(self):
        if self.fb == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()


def batches(nc, bs):
    return [(c, min(c + bs, nc)) for c in range(0, nc, bs)]


# ---------------------------------------------------------------------------------------------------------------- forward
def forward_all(net, ids, tm):
    """X_0 -> X_L on the GPU, X_l stored as bf16 in host RAM, diag(H) of the four block inputs per layer (the imatrix).
    Returns (X_L on GPU, diags[l] = {attn_in, o_in, mlp_in, down_in} fp32 cpu numpy)."""
    nt = net.nc * CHUNK
    emb = net.get("model.embed_tokens.weight")
    toks = torch.tensor(ids[:nt]).reshape(net.nc, CHUNK)
    X = emb[toks].float().cuda()
    del emb
    diags = []
    loader = ThreadPoolExecutor(1)
    fut = loader.submit(net.read_layer, 0)
    sums = {}
    L = net.layer
    hooks = []
    for key, path in (("attn_in", "self_attn.q_proj"), ("o_in", "self_attn.o_proj"), ("mlp_in", "mlp.gate_proj"),
                      ("down_in", "mlp.down_proj")):
        def mk(key):
            def f(mod, a, out):
                x = a[0].reshape(-1, a[0].shape[-1]).float()
                sums[key] += x.pow(2).sum(0)
            return f
        hooks.append(net.module(path).register_forward_hook(mk(key)))
    for l in range(net.L):
        with tm("fwd.wait", sync=False):
            cpu = fut.result()
        if l + 1 < net.L:
            fut = loader.submit(net.read_layer, l + 1)
        with tm("fwd.load"):
            net.put_layer(cpu)
        with tm("fwd.xstore"):
            net.xs[l] = X.to(torch.bfloat16).cpu()
        for key, path in (("attn_in", "self_attn.q_proj"), ("o_in", "self_attn.o_proj"), ("mlp_in", "mlp.gate_proj"),
                          ("down_in", "mlp.down_proj")):
            sums[key] = torch.zeros(net.module(path).weight.shape[1], device="cuda")
        with tm("fwd.compute"), torch.no_grad(), net.fbctx():
            Y = torch.empty_like(X)
            for a, b in batches(net.nc, net.bs):
                Y[a:b] = L(X[a:b], position_embeddings=net.pe)
        X = Y
        diags.append({k: (v / nt).cpu().numpy() for k, v in sums.items()})
    for h in hooks:
        h.remove()
    loader.shutdown()
    return X, diags


def top(net, X, tm):
    """Final norm + lm_head, K sampled targets per chunk (seed 1000+c), gradient w.r.t. X_L -> pinned (K, nc, CHUNK, H)."""
    L = net.L
    net.fnorm.weight.copy_(net.get("model.norm.weight").cuda().float())
    lm = (net.get("lm_head.weight") if "lm_head.weight" in net.wmap else net.get("model.embed_tokens.weight")).cuda().float()
    D = torch.empty(net.K, net.nc, CHUNK, net.H, dtype=torch.float32).pin_memory()
    with tm("top"):
        for c in range(net.nc):
            x = X[c:c + 1].clone().requires_grad_(True)
            with net.fbctx():
                logp = torch.log_softmax(F.linear(net.fnorm(x), lm)[0].float(), -1)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(1000 + c)
            pr = logp.exp().detach()
            for k in range(net.K):
                s = torch.multinomial(pr, 1, generator=gen)
                g, = torch.autograd.grad(-logp.gather(1, s).sum(), x, retain_graph=k < net.K - 1)
                D[k, c].copy_(g[0], non_blocking=True)
    del lm
    torch.cuda.empty_cache()
    return D


# ---------------------------------------------------------------------------------------------------------------- stats
def new_stats():
    return {"cnt": torch.zeros(LBN, dtype=torch.int64, device="cuda"), "dsum": torch.zeros(LBN, dtype=torch.float32, device="cuda"),
            "orc_cnt": torch.zeros(LBN, dtype=torch.int64, device="cuda"), "orc_dsum": torch.zeros(LBN, dtype=torch.float32, device="cuda"),
            "sumlog": np.float32(0), "n_sel": 0, "side_bits": 0, "kq_list": [], "per_kind": {}}


def hist_sep(stats, La, Lb):
    """Histogram of lam_ij = 2^(La_i + Lb_j) over all pairs, on the LBW grid, from the marginals on a grid LBW/SUB: a pair
    is binned by its marginal fine bins (error < LBW/SUB in log2 lam); the per-bin sum of lam is exact for that grouping."""
    wm = LBW / SUB
    ma, mb = float(La.min()), float(Lb.min())
    ua, ub = La - ma, Lb - mb

    def marg(u):
        p = (u / wm).floor().long()
        c = torch.bincount(p).float()
        m = torch.bincount(p, weights=torch.exp2(u - (p.float() + 0.5) * wm)).float()
        nz = c.nonzero().squeeze(1)
        return nz, c[nz], m[nz]

    pa, ca, qa = marg(ua)
    pb, cb, qb = marg(ub)
    delta = ma + mb + LB0
    nf = int(pa.max()) + int(pb.max()) + 2
    cf = torch.zeros(nf, device=La.device, dtype=torch.float32)
    mf = torch.zeros(nf, device=La.device, dtype=torch.float32)
    step = max(1, (1 << 24) // max(1, pb.numel()))
    for i in range(0, pa.numel(), step):
        f = (pa[i:i + step, None] + pb[None, :]).reshape(-1)
        cf.index_add_(0, f, (ca[i:i + step, None] * cb[None, :]).reshape(-1))
        mf.index_add_(0, f, (qa[i:i + step, None] * qb[None, :]).reshape(-1))
    # fine bin f holds pairs with u_a + u_b ~ (f + 1) wm; coarse bin k = floor(((f + 1) wm + delta) / LBW)
    tc = (torch.arange(nf, device=La.device, dtype=torch.float32) + 1) * wm + delta
    k = (tc / LBW).floor().long()
    ok = (cf > 0) & (k >= 0)
    k = k.clamp(0, LBN - 1)
    rel = mf * torch.exp2(tc - (k.float() + 0.5) * LBW)  # mass of lam relative to the centre of coarse bin k
    cnt = torch.zeros(LBN, device=La.device, dtype=torch.float32)
    dsum = torch.zeros(LBN, device=La.device, dtype=torch.float32)
    cnt.index_add_(0, k[ok], cf[ok])
    dsum.index_add_(0, k[ok], (rel - cf)[ok])
    stats["cnt"] += cnt.round().long()
    stats["dsum"] += dsum


_CEN = None


def hist_exact(stats, lam):
    """The rstar_stream.py element histogram (reference for the control and the oracle floor)."""
    global _CEN
    if _CEN is None:
        _CEN = torch.exp2((torch.arange(LBN, device="cuda", dtype=torch.float32) + 0.5) * LBW - LB0)
    lam = lam.clamp(min=EPS)
    idx = ((torch.log2(lam) + LB0) / LBW).long().clamp(0, LBN - 1)
    stats["cnt"] += torch.bincount(idx, minlength=LBN)
    s = torch.zeros(LBN, dtype=torch.float32, device="cuda")
    s.index_add_(0, idx, lam / _CEN[idx] - 1)
    stats["dsum"] += s


def spectra_to_stats(stats, rows, cols, g, r0, h, c0, m, C2=None):
    """Add one tensor given its spectra: g (rows), r0 (rows), h (cols), c0 (cols), m = mean C^2, optional C2 = (U^T W V)^2."""
    g, h = g.clamp(min=EPS), h.clamp(min=EPS)
    r, c = r0.clamp(min=EPS), c0.clamp(min=EPS)
    sl = cols * (torch.log(g / 2) + torch.log(r)).sum() + rows * (torch.log(h) + torch.log(c)).sum() - rows * cols * torch.log(m)
    stats["sumlog"] = np.float32(stats["sumlog"] + np.float32(sl.item()))
    La = torch.log2((0.5 * g * r0 / m).clamp(min=EPS))
    Lb = torch.log2((h * c0).clamp(min=EPS))
    hist_sep(stats, La, Lb)
    if C2 is not None:
        ost = {"cnt": stats["orc_cnt"], "dsum": stats["orc_dsum"]}
        step = max(1, (1 << 25) // cols)
        for i in range(0, rows, step):
            lam = (0.5 * g[i:i + step, None] * h[None, :] * C2[i:i + step]).reshape(-1)
            hist_exact(ost, lam)
    stats["n_sel"] += rows * cols
    stats["side_bits"] += 16 * (rows + cols)


def slq_spectrum(M, nv, steps, seed=7):
    """Synthetic eigenvalues (ascending, n values) of symmetric PSD M from stochastic Lanczos quadrature: nv Gaussian probes,
    `steps` Lanczos steps with full reorthogonalisation, Gauss nodes/weights averaged over probes into a spectral CDF, read back at
    the n mid-quantiles (linear in log node), scaled to the exact trace.
    cost: steps matmuls (n, n) x (n, nv) = 2 n^2 nv steps FLOP (fp32), reorthogonalisation 2 n nv steps^2 FLOP, storage steps n nv floats."""
    n = M.shape[0]
    gen = torch.Generator(device=M.device)
    gen.manual_seed(seed)
    q = torch.randn(n, nv, device=M.device, generator=gen)
    q /= q.norm(dim=0)
    Q = torch.empty(steps, n, nv, device=M.device)
    al = torch.zeros(steps, nv, device=M.device)
    be = torch.zeros(steps, nv, device=M.device)
    with tf32(False):
        for j in range(steps):
            Q[j] = q
            w = M @ q
            al[j] = (w * q).sum(0)
            w -= al[j] * q
            if j:
                w -= be[j - 1] * Q[j - 1]
            c = torch.einsum("jnv,nv->jv", Q[:j + 1], w)
            w -= torch.einsum("jv,jnv->nv", c, Q[:j + 1])
            be[j] = w.norm(dim=0)
            q = w / be[j].clamp(min=1e-30)
    al, be = al.t().double().cpu(), be.t().double().cpu()  # (nv, steps): small, on the host
    T = torch.diag_embed(al) + torch.diag_embed(be[:, :-1], 1) + torch.diag_embed(be[:, :-1], -1)
    th, ev = torch.linalg.eigh(T)
    w = (ev[:, 0, :] ** 2) / nv
    th, w = th.reshape(-1), w.reshape(-1)
    o = th.argsort()
    th, w = th[o].numpy(), w[o].numpy()
    cdf = np.cumsum(w) - 0.5 * w
    u = (np.arange(n) + 0.5) / n
    g = np.exp(np.interp(u, cdf, np.log(np.maximum(th, EPS))))
    tr = float(M.diagonal().sum())
    return torch.from_numpy(g * (tr / g.sum())).float().to(M.device)


def decomp(M, mode):
    """(eigenvalues ascending, eigenvectors as columns | None). n <= slq_min: eigh, else slq_spectrum (no vectors)."""
    if M.shape[0] <= mode["slq_min"] or mode["eig"] == "exact":
        return eig(M)
    return slq_spectrum(M, mode["nv"], mode["steps"]), None


def align(U, W, axis, mode):
    """r0_i = |u_i^T W|^2 / cols (axis 0) or c0_j = |W v_j|^2 / rows (axis 1); without eigenvectors the mean
    ||W||_F^2 / (rows cols) for every index."""
    rows, cols = W.shape
    if U is None:
        return torch.full((rows if axis == 0 else cols,), float((W.float() ** 2).mean()), device="cuda")
    with tf32(mode["aux"] == "tf32"):
        return ((U.t() @ W) ** 2).sum(1) / cols if axis == 0 else ((W @ U) ** 2).sum(0) / rows


def analyze_layer(net, st, l, grams, dq, mode, dump):
    """All seven tensors of block l. grams: 'G:<q|k|v|o|gate|up|down>' and 'H:<attn_in|o_in|mlp_in|down_in>' (normalised,
    consumed). dq: kind -> host fp32 (rows, cols) dequantised Q4_K_M weights. Fills st (histogram, sumlog, kq4)."""
    tm = net.tm
    uses = {}
    for kind, _, hk in KINDS:
        uses[hk] = uses.get(hk, 0) + 1
    ecache = {}
    aux = mode["aux"] == "tf32"
    if mode["dump_gram"]:
        for key, M in grams.items():
            np.save(os.path.join(dump, f"{l}.{key.replace(':', '_')}.npy"), M.cpu().numpy())
    for kind, path, hk in KINDS:
        W = net.weight(kind)
        rows, cols = W.shape
        with tm("an.kq4"):
            D = torch.from_numpy(dq[kind]).cuda() - W
            Hm = grams["H:" + hk]
            G = grams["G:" + GK[kind]]
            with tf32(aux):
                k4 = 0.5 * float(((G @ D) * (D @ Hm)).sum())
            del D
        st["kq_list"].append(np.float32(k4))
        st["per_kind"][kind] = np.float32(st["per_kind"].get(kind, np.float32(0)) + np.float32(k4))
        with tm("an.eigh"):
            g, U = decomp(G, mode)
            grams["G:" + GK[kind]] = None
            del G
            if hk not in ecache:
                ecache[hk] = decomp(Hm.clone() if uses[hk] > 1 else Hm, mode)
            h, V = ecache[hk]
            C2 = None
            if U is not None and V is not None:
                with tf32(aux):
                    UW = U.t() @ W
                    r0 = (UW ** 2).sum(1) / cols
                    C2 = (UW @ V) ** 2
                del UW, U
            else:
                r0 = align(U, W, 0, mode)
                del U
            c0 = align(V, W, 1, mode)
            uses[hk] -= 1
            if uses[hk] == 0:
                grams["H:" + hk] = None
                del Hm, V
                ecache.pop(hk, None)
        m = (W.float() ** 2).mean()
        with tm("an.hist"):
            spectra_to_stats(st, rows, cols, g, r0, h, c0, m, C2)
        del C2
        if dump:
            np.savez(os.path.join(dump, f"{l}.{kind}.npz"), g=g.cpu().numpy(), r0=r0.cpu().numpy(), h=h.cpu().numpy(),
                     c0=c0.cpu().numpy(), m=m.item(), rows=rows, cols=cols, k4=k4)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------- backward
def prep_layer(net, ggml, l, diag, analyze):
    """Loader thread: bf16 weights, X_l, and (when analysing) the dequantised Q4_K_M weights of block l."""
    cpu = net.read_layer(l)
    x = net.xs.pop(l).pin_memory()
    dq = None
    if analyze:
        dq = {}
        for kind, path, hk in KINDS:
            w = cpu[path + ".weight"].float().half().float().numpy()  # llama converts through f16
            t = qtype(kind, l, net.L, w.shape[1])
            im = np.ascontiguousarray(diag[hk], dtype=np.float32)
            dq[kind], _ = ggml.qdq(np.ascontiguousarray(w), t, im)
    return cpu, x, dq


def backward_all(net, ggml, diags, D, mode, only, dump, tm):
    K, nc, bs = net.K, net.nc, net.bs
    nt = nc * CHUNK
    stats = new_stats()
    loader = ThreadPoolExecutor(1)
    L = net.L
    Din = D
    Dout = torch.empty_like(D).pin_memory() if L > 1 else None
    fut = loader.submit(prep_layer, net, ggml, L - 1, diags[L - 1], (L - 1) in only)
    outs_names = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up")
    cap_o, cap_i, hooks = {}, {}, []
    for kind in outs_names:
        def mk(kind):
            def f(mod, a, out):
                cap_o[kind] = out
                cap_i[kind] = a[0]
            return f
        hooks.append(net.module(KIND[kind][0]).register_forward_hook(mk(kind)))
    down_in = {}
    hooks.append(net.module("mlp.down_proj").register_forward_hook(lambda mod, a, out: down_in.__setitem__("x", a[0])))
    for l in range(L - 1, -1, -1):
        t_layer = time.time()
        with tm("bwd.wait", sync=False):
            cpu, xl, dq = fut.result()
        if l > 0:
            fut = loader.submit(prep_layer, net, ggml, l - 1, diags[l - 1], (l - 1) in only)
        analyze = l in only
        with tm("bwd.load"):
            net.put_layer(cpu)
        grams = {}
        if analyze:
            dims = {k: net.weight(k).shape for k, _, _ in KINDS}
            for kind, path, hk in KINDS:
                grams["G:" + GK[kind]] = Gram(dims[kind][0], mode["gram"])
            for hk, n in (("attn_in", dims["attn_q"][1]), ("o_in", dims["attn_output"][1]), ("mlp_in", dims["ffn_gate"][1]),
                          ("down_in", dims["ffn_down"][1])):
                grams["H:" + hk] = Gram(n, mode["gram"])
        with tm("bwd.compute"):
            for a, b in batches(nc, bs):
                x = xl[a:b].cuda(non_blocking=True).float().requires_grad_(True)
                with net.fbctx():
                    y = net.layer(x, position_embeddings=net.pe)
                if analyze:
                    grams["H:attn_in"].add(cap_i["attn_q"].detach())
                    grams["H:o_in"].add(cap_i["attn_output"].detach())
                    grams["H:mlp_in"].add(cap_i["ffn_gate"].detach())
                    grams["H:down_in"].add(down_in["x"].detach())
                    outs = [cap_o[k] for k in outs_names] + [x]
                else:
                    outs = [x]
                per = [[] for _ in outs]
                dys = []
                for k in range(K):
                    dy = Din[k, a:b].cuda(non_blocking=True)
                    gr = torch.autograd.grad(y, outs, grad_outputs=dy, retain_graph=k < K - 1)
                    for lst, g in zip(per, gr):
                        lst.append(g.detach())
                    dys.append(dy)
                    if Dout is not None:
                        Dout[k, a:b].copy_(gr[-1], non_blocking=True)
                if analyze:
                    for kind, lst in zip(outs_names, per):
                        grams["G:" + GK[kind]].add(torch.cat(lst))
                    grams["G:down"].add(torch.cat(dys))
                cap_o.clear(), cap_i.clear(), down_in.clear()
                del y, outs, per, dys
        if analyze:
            with tm("bwd.norm"):
                res = {}
                for key, gm in grams.items():
                    res[key] = gm.result(nt * K if key.startswith("G:") else nt)
            analyze_layer(net, stats, l, res, dq, mode, dump)
        if Dout is not None:
            torch.cuda.synchronize()
            Din, Dout = Dout, Din
        log(f"layer {l} {time.time() - t_layer:.1f}s", "analysed" if analyze else "chain only")
    for h in hooks:
        h.remove()
    loader.shutdown()
    return stats


# ---------------------------------------------------------------------------------------------------------------- aggregate
def aggregate(net, st, args, out_extra):
    cnt, dsum = st["cnt"].cpu(), st["dsum"].cpu()
    sumlog, n_sel, side = float(st["sumlog"]), st["n_sel"], st["side_bits"]
    kq4 = float(torch.tensor(st["kq_list"], dtype=torch.float32).sum())
    nl = sum(net.weight(k).numel() for k, _, _ in KINDS)
    N_full = nl * net.L
    K0 = N_full * math.exp(sumlog / n_sel)
    side_bpw = side / n_sel
    scale = N_full / n_sel
    kq4_full = kq4 * scale
    res = {"model": net.hf, "layers_analysed": n_sel // nl, "n_layers": net.L, "n_chunks": net.nc, "k_samples": net.K,
           "N_sample": n_sel, "N_full": N_full, "side_bpw": side_bpw, "mean_log_lam": sumlog / n_sel, "K0": K0,
           "K0_over_N": K0 / N_full, "kl_q4_sample": kq4, "kl_q4_full_est": kq4_full,
           "kl_q4_by_kind_sample": {k: float(v) for k, v in st["per_kind"].items()},
           "R_star_net": math.log(K0 / kq4_full, 4), "R_star_allin": math.log(K0 / kq4_full, 4) + side_bpw}
    cen = torch.exp2((torch.arange(LBN, dtype=torch.float32) + 0.5) * LBW - LB0)

    def bisect_hist(cnt_t, dsum_t, side_b):
        ok = cnt_t > 0
        cn = cnt_t[ok].float()
        lb = cen[ok] * (1 + dsum_t[ok] / cn)
        log2l = torch.log2(lb)

        def at(th):
            kl = float((cn * torch.clamp(lb, max=th)).sum())
            rate = float((cn * torch.clamp(0.5 * (log2l - math.log2(th)), min=0)).sum())
            return kl, rate

        lo, hi = np.float32(math.log(float(lb.min())) - 50), np.float32(math.log(float(lb.max())))
        for _ in range(100):
            mid = np.float32((lo + hi) / 2)
            lo, hi = (mid, hi) if at(float(np.exp(mid)))[0] < kq4 else (lo, mid)
        th_q4 = float(np.exp(lo))
        kl, rate = at(th_q4)
        lo_r, hi_r = np.float32(math.log(float(lb.min())) - 50), np.float32(math.log(float(lb.max())))
        bud_118 = 1.18 * n_sel - side_b
        for _ in range(100):
            mid = np.float32((lo_r + hi_r) / 2)
            lo_r, hi_r = (mid, hi_r) if at(float(np.exp(mid)))[1] > bud_118 else (lo_r, mid)
        th_118 = float(np.exp(hi_r))
        kl_118, _ = at(th_118)
        return {"theta": th_q4, "kl_sample": kl, "rate_net_bpw": rate / n_sel,
                "rate_allin_bpw": (rate + side_b) / n_sel,
                "kl_at_1.18_sample": kl_118, "kl_at_1.18_full_est": kl_118 * scale,
                "kl_at_1.18_over_q4": kl_118 / kq4}

    res["hist_bisection"] = {"q4_match": bisect_hist(cnt, dsum, side)}
    ocnt, odsum = st["orc_cnt"].cpu(), st["orc_dsum"].cpu()
    if int(ocnt.sum()) == n_sel:
        om = bisect_hist(ocnt, odsum, 0)
        sm = res["hist_bisection"]["q4_match"]
        om["ratio_to_sep_allin"] = om["rate_allin_bpw"] / sm["rate_allin_bpw"]
        om["ratio_to_sep_net"] = om["rate_net_bpw"] / sm["rate_net_bpw"]
        om["diff_from_sep_allin_bpw"] = sm["rate_allin_bpw"] - om["rate_allin_bpw"]
        om["diff_from_sep_net_bpw"] = sm["rate_net_bpw"] - om["rate_net_bpw"]
        res["hist_bisection"]["oracle_q4_match"] = om
    res.update(out_extra)
    res["script_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    return res


# ---------------------------------------------------------------------------------------------------------------- control Q1
def check_quant(argv):
    """Q1: quantise f16-rounded HF weights with ggml_quantize_chunk and the llama imatrix; compare bytes to the llama
    Q4_K_M gguf. argv = HF_DIR REF_GGUF IMATRIX_GGUF N_LAYERS layer:kind:hf.module ..."""
    from safetensors import safe_open
    sys.path.insert(0, "/home/socol/Workspace/llama.cpp/gguf-py")
    import gguf
    hf, gq, imf, nl = argv[0], argv[1], argv[2], int(argv[3])
    ip = os.path.join(hf, "model.safetensors.index.json")
    idx = json.load(open(ip))["weight_map"] if os.path.exists(ip) else {k: "model.safetensors" for k in safe_open(os.path.join(hf, "model.safetensors"), "pt").keys()}
    ref = {t.name: t for t in gguf.GGUFReader(gq).tensors}
    imr = {t.name: t for t in gguf.GGUFReader(imf).tensors}
    gg = Ggml()
    bad = 0
    for spec in argv[4:]:
        l, kind, hfn = spec.split(":")
        l = int(l)
        name = f"model.layers.{l}.{hfn}.weight"
        W = safe_open(os.path.join(hf, idx[name]), "pt").get_tensor(name).float().numpy().astype(np.float16).astype(np.float32)
        g = f"blk.{l}.{kind}.weight"
        im = np.asarray(imr[g + ".in_sum2"].data, dtype=np.float32).reshape(-1).copy()
        im = (im / float(np.asarray(imr[g + ".counts"].data).reshape(-1)[0])).astype(np.float32)
        t = qtype(kind, l, nl, W.shape[1])
        assert int(ref[g].tensor_type) == t, (g, int(ref[g].tensor_type), t)
        t0 = time.time()
        dq, q = gg.qdq(np.ascontiguousarray(W), t, im, want_q=True)
        dt = time.time() - t0
        r = np.asarray(ref[g].data).reshape(-1)
        assert r.size == q.size, (r.size, q.size)
        rs = gg.lib.ggml_row_size(t, W.shape[1])
        diff = int((q.reshape(-1, rs) != r.reshape(-1, rs)).any(1).sum())
        bad += diff
        from gguf.quants import dequantize
        refdq = dequantize(ref[g].data, ref[g].tensor_type).reshape(W.shape)
        print(g, "type", t, "rows", W.shape[0], "differing rows", diff, "max|dq - gguf dq|", float(np.abs(dq - refdq).max()),
              f"{dt:.2f}s", flush=True)
    print("Q1", "PASS" if bad == 0 else f"FAIL ({bad} rows differ)")
    return bad


# ---------------------------------------------------------------------------------------------------------------- driver
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check-quant":
        sys.exit(1 if check_quant(sys.argv[2:]) else 0)
    ap = argparse.ArgumentParser()
    ap.add_argument("name"), ap.add_argument("hf"), ap.add_argument("out")
    ap.add_argument("--chunks", type=int, default=100), ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--fb", default="bf16", choices=["fp32", "tf32", "bf16"])
    ap.add_argument("--gram", default="fp16s", choices=["fp32", "tf32", "fp16s"])
    ap.add_argument("--aux", default="tf32", choices=["fp32", "tf32"])
    ap.add_argument("--eig", default="slq", choices=["exact", "slq"])
    ap.add_argument("--slq-min", type=int, default=2000), ap.add_argument("--nv", type=int, default=32)
    ap.add_argument("--lanczos", type=int, default=100), ap.add_argument("--dump-gram", action="store_true")
    ap.add_argument("--only"), ap.add_argument("--dump")
    args = ap.parse_args()
    import transformers
    torch.backends.cuda.matmul.allow_tf32 = args.fb == "tf32"
    torch.backends.cudnn.allow_tf32 = args.fb == "tf32"
    t_start = time.time()
    net = Net(args.hf, args.chunks, args.k, args.bs, args.fb)
    net.tm = tm = Timer()
    only = set(range(net.L)) if not args.only else {int(x) for x in args.only.split(",")}
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
    tok = transformers.AutoTokenizer.from_pretrained(args.hf)
    ids = tok.encode(open(CALIB, encoding="utf-8").read())
    assert len(ids) >= args.chunks * CHUNK
    ggml = Ggml()
    mode = {"gram": args.gram, "aux": args.aux, "eig": args.eig, "slq_min": args.slq_min, "nv": args.nv,
            "steps": args.lanczos, "dump_gram": args.dump_gram}
    t0 = time.time()
    X, diags = forward_all(net, ids, tm)
    log(f"forward {time.time() - t0:.1f}s")
    t0 = time.time()
    D = top(net, X, tm)
    del X
    log(f"top {time.time() - t0:.1f}s")
    t0 = time.time()
    st = backward_all(net, ggml, diags, D, mode, only, args.dump, tm)
    log(f"backward {time.time() - t0:.1f}s")
    wall = time.time() - t_start
    res = aggregate(net, st, args, {"wall_s": wall, "phase_s": tm.t, "args": vars(args),
                                    "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30})
    json.dump(res, open(args.out, "w"), indent=1)
    log(json.dumps({k: res[k] for k in ("K0_over_N", "kl_q4_sample", "R_star_allin", "wall_s")}),
        "q4_match allin:", res["hist_bisection"]["q4_match"]["rate_allin_bpw"],
        *(["oracle_q4_match:", res["hist_bisection"]["oracle_q4_match"]["rate_allin_bpw"],
           "ratio:", res["hist_bisection"]["oracle_q4_match"]["ratio_to_sep_allin"]]
          if "oracle_q4_match" in res["hist_bisection"] else []))
    log("phases", {k: round(v, 1) for k, v in tm.t.items()})


if __name__ == "__main__":
    main()
