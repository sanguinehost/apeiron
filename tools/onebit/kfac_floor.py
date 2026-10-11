"""ONEBIT W15b: Kronecker (K-FAC) Fisher factors on all 168 linear tensors of Qwen2.5-0.5B, validated against the
exact quadratic of W15a, and the global reverse water-filling KL at a total rate.

Capture. For a 512-token chunk and a target sequence s sampled from the model's own softmax (true Fisher), g_t is the
gradient of the sequence NLL with respect to the output y_t of a linear W (out x in) and x_t its input. The full
backward graph is built (the embedding output is the leaf; no linear output is detached). Per token
    G = mean_t E[g_t g_t^T]  (out x out),   H = mean_t x_t x_t^T  (in x in),
and the per-token second-order KL of an error D is  1/2 tr(D^T G D H)  (`model`: drops cross-token score covariance
and the dependence between g and x). In the eigenbases G = U diag(g) U^T, H = V diag(h) V^T, with E = U^T D V,
1/2 tr(D^T G D H) = sum_ij 1/2 g_i h_j E_ij^2  (`Perf.CostCorrespondence.kron_diag`).

Validation. On the 21 `ffn_down` tensors (blocks 3-23) the K-FAC sum for each arm is compared with the exact
quadratic 64 * KL(1/8) of `docs/evidence/onebit-w15.md`.

Floor. Coefficients C = U^T W V of the f16 weights, component weight w_ij = 1/2 g_i h_j. Two variance models:
  separable: sigma2_ij = r_i c_j / m (row and column means of C^2, grand mean m; side information 16 bits per row and
             column), coded by Gaussian codebooks: achievable at the Gaussian water-filling distortion for any source
             with those second moments (Lapidoth 1997, `anchor`, not retrieved);
  oracle:    sigma2_ij = C_ij^2 (no source model; an optimistic reference, not a bound).
lambda_ij = w_ij sigma2_ij; at level theta the rate is sum max(0, 1/2 log2(lambda/theta)) bits and the KL is
sum min(lambda, theta) (`waterfill_level`, `waterfill_floor`). Q4_K_M's K-FAC KL on the same 168 tensors is the
reference.

Run with /home/socol/Workspace/spinor-ttl/.venv/bin/python (torch + CUDA), from the repository root.
usage: python tools/onebit/kfac_floor.py OUT.json [N_CHUNKS] [K_SAMPLES]
Cost: N_CHUNKS forwards and N_CHUNKS*K backwards; per backward 168 Gram updates (sum over tensors of 512*out^2 MACs,
~1.2 TMAC), run twice over 12-layer halves so the factors (H shared per input, 4 per layer) take 3.65 GB float32
on the GPU at a time. Analysis: 2 eigendecompositions per
tensor (at most 4864^2, float32), 168 + 21*5 dequantisations, and ~60 bisection steps over 357.8M components on the
GPU (two float32 arrays, 2.9 GB).
"""
import json, math, sys, time
import numpy as np
import torch
import transformers

from fisher_capture import ARMS, BLOCKS, CALIB, CHUNK, FILES, gguf, dequantize

LAYERS = range(24)
# (gguf name, HF module path, input key shared by modules reading the same activation)
KINDS = [("attn_q", "self_attn.q_proj", "attn_in"), ("attn_k", "self_attn.k_proj", "attn_in"),
         ("attn_v", "self_attn.v_proj", "attn_in"), ("attn_output", "self_attn.o_proj", "o_in"),
         ("ffn_gate", "mlp.gate_proj", "mlp_in"), ("ffn_up", "mlp.up_proj", "mlp_in"),
         ("ffn_down", "mlp.down_proj", "down_in")]
EXACT_QUAD = {"iq1_s": 0.39085, "iq1_m": 0.30543, "q4_k_m": 0.0045087, "rtn": 3.5563, "had_gptq1": 0.41892}
RATES = (1.0, 1.18, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5)


def module(model, layer, path):
    m = model.model.layers[layer]
    for p in path.split("."):
        m = getattr(m, p)
    return m


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


def capture(model, ids, n_chunks, k_samples, layers):
    """Accumulate G per tensor and H per shared input of `layers` over n_chunks * 512 tokens and k_samples targets."""
    G, H, outs, ins, handles = {}, {}, {}, {}, []
    for l in layers:
        for name, path, key in KINDS:
            m = module(model, l, path)
            G[(l, name)] = torch.zeros(m.out_features, m.out_features, device="cuda")
            if (l, key) not in H:
                H[(l, key)] = torch.zeros(m.in_features, m.in_features, device="cuda")

            def pre(mod, args, l=l, key=key):
                if (l, key) not in ins:
                    ins[(l, key)] = args[0].detach().reshape(-1, args[0].shape[-1])

            def post(mod, args, out, l=l, name=name):
                outs[(l, name)] = out

            handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    keys = list(G)
    for c in range(n_chunks):
        x = torch.tensor([ids[c * CHUNK:(c + 1) * CHUNK]], device="cuda")
        logp = torch.log_softmax(model(x).logits[0].float(), -1)
        for k, a in ins.items():
            H[k].addmm_(a.t(), a)
        for s_i in range(k_samples):
            s = torch.multinomial(logp.exp().detach(), 1)
            grads = torch.autograd.grad(-logp.gather(1, s).sum(), [outs[k] for k in keys],
                                        retain_graph=s_i < k_samples - 1)
            for k, g in zip(keys, grads):
                g = g.reshape(-1, g.shape[-1])
                G[k].addmm_(g.t(), g)
            del grads
        outs.clear()
        ins.clear()
    for hd in handles:
        hd.remove()
    n_tok = n_chunks * CHUNK
    for v in G.values():
        v.div_(n_tok * k_samples)
    for v in H.values():
        v.div_(n_tok)
    return G, H


def level_for_rate(lam, budget_bits):
    """Bisection on log theta: rate(theta) = sum max(0, 1/2 log2(lam/theta)) is decreasing in theta. 80 steps."""
    lo, hi = math.log(float(lam[lam > 0].min())) - 50, math.log(float(lam.max()))
    for _ in range(80):
        mid = (lo + hi) / 2
        rate = float(torch.clamp(0.5 * (torch.log2(lam) - mid / math.log(2)), min=0).sum(dtype=torch.float64))
        lo, hi = (mid, hi) if rate > budget_bits else (lo, mid)
    theta = math.exp(hi)
    return theta, float(torch.clamp(lam, max=theta).sum(dtype=torch.float64))


def level_for_kl(lam, kl_target):
    """Bisection on log theta: KL(theta) = sum min(lam, theta) is increasing in theta. Returns the rate in bits."""
    lo, hi = math.log(float(lam[lam > 0].min())) - 50, math.log(float(lam.max()))
    for _ in range(80):
        mid = (lo + hi) / 2
        kl = float(torch.clamp(lam, max=math.exp(mid)).sum(dtype=torch.float64))
        lo, hi = (mid, hi) if kl < kl_target else (lo, mid)
    return float(torch.clamp(0.5 * (torch.log2(lam) - lo / math.log(2)), min=0).sum(dtype=torch.float64))


def main(out_path, n_chunks, k_samples):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ids = tok.encode(open(CALIB, encoding="utf-8").read())
    assert len(ids) >= n_chunks * CHUNK
    model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hf_w = {(l, name): module(model, l, path).weight.detach().cpu() for l in LAYERS for name, path, _ in KINDS}
    model.model.embed_tokens.register_forward_hook(lambda mod, a, out: out.detach().requires_grad_(True))

    readers = {k: gguf.GGUFReader(FILES[k]) for k in ("f16", "q4_k_m")}
    arm_readers = {a: gguf.GGUFReader(FILES[a]) for a in ARMS}
    kfac_down = {a: 0.0 for a in ARMS}
    kfac_q4_all, n_weights, side_bits, max_rel = 0.0, 0, 0, 0.0
    lam_sep, lam_orc, per_kind = [], [], {}
    t_cap = 0.0
    for half in (range(0, 12), range(12, 24)):
        t1 = time.time()
        G, H = capture(model, ids, n_chunks, k_samples, half)
        t_cap += time.time() - t1
        print("capture", list(half)[0], time.time() - t1, flush=True)
        for l in half:
            for name, _, key in KINDS:
                gname = f"blk.{l}.{name}.weight"
                w = read(readers["f16"], gname)
                max_rel = max(max_rel, float((w - hf_w[(l, name)]).norm() / hf_w[(l, name)].norm()))
                Gm, Hm = G.pop((l, name)), H[(l, key)]
                w = w.cuda()
                q4 = read(readers["q4_k_m"], gname).cuda() - w
                k4 = 0.5 * float(((Gm @ q4) * (q4 @ Hm)).sum(dtype=torch.float64))
                kfac_q4_all += k4
                per_kind[name] = per_kind.get(name, 0.0) + k4
                if name == "ffn_down" and l in BLOCKS:
                    for a in ARMS:
                        d = read(arm_readers[a], gname).cuda() - w
                        kfac_down[a] += 0.5 * float(((Gm @ d) * (d @ Hm)).sum(dtype=torch.float64))
                g, U = torch.linalg.eigh(Gm)
                h, V = torch.linalg.eigh(Hm)
                g, h = g.clamp(min=0), h.clamp(min=0)
                wt = 0.5 * g[:, None] * h[None, :]
                C2 = (U.t() @ w @ V) ** 2
                m = C2.mean()
                lam_orc.append((wt * C2).flatten().cpu())
                lam_sep.append((wt * (C2.mean(1, keepdim=True) * C2.mean(0, keepdim=True) / m)).flatten().cpu())
                n_weights += w.numel()
                side_bits += 16 * (w.shape[0] + w.shape[1])
                del Gm, U, V, C2, wt, q4
            for name, _, key in KINDS:
                H.pop((l, key), None)
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    t_an = time.time() - t0 - t_cap
    print("analysis", t_an, "max_rel_gguf_vs_hf", max_rel, flush=True)
    res = {"n_chunks": n_chunks, "k_samples": k_samples, "seed": 0, "n_weights": n_weights,
           "side_bits_separable": side_bits, "max_rel_gguf_vs_hf": max_rel, "validation": {}, "floor": {}}
    for a in ARMS:
        res["validation"][a] = {"kfac_down": kfac_down[a], "exact_quadratic": EXACT_QUAD[a],
                                "ratio": kfac_down[a] / EXACT_QUAD[a]}
        print(a, res["validation"][a], flush=True)
    res["q4_k_m_kfac_all"] = kfac_q4_all
    res["q4_k_m_kfac_by_kind"] = per_kind
    print("q4_k_m kfac all 168", kfac_q4_all, per_kind, flush=True)
    for model_name, parts in (("separable", lam_sep), ("oracle", lam_orc)):
        lam = torch.cat(parts).cuda()
        parts.clear()
        extra = side_bits if model_name == "separable" else 0
        row = {"total_kl_zero_rate": float(lam.sum(dtype=torch.float64)), "by_rate": {}}
        for r in RATES:
            theta, kl = level_for_rate(lam, r * n_weights - extra)
            row["by_rate"][str(r)] = {"theta": theta, "kl": kl}
        row["rate_bpw_at_q4_kl"] = (level_for_kl(lam, kfac_q4_all) + extra) / n_weights
        res["floor"][model_name] = row
        print(model_name, json.dumps(row), flush=True)
        del lam
        torch.cuda.empty_cache()
    res["wall_s"] = time.time() - t0
    res["capture_s"], res["analysis_s"] = t_cap, t_an
    json.dump(res, open(out_path, "w"), indent=1)
    print("wall", res["wall_s"])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
