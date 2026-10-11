"""ONEBIT W14: Fisher stiffness of each quantised arm on the 21 `ffn_down` tensors of Qwen2.5-0.5B, with no Kronecker
factorisation.

For a sequence c of 512 tokens and a target s sampled from the model's own softmax at every position (true Fisher,
one Monte-Carlo target sequence per sample), the sequence score with respect to the weight W of one `ffn_down` is
J = d(-log p(s))/dW = sum_t g_t x_t^T, with x_t the input and g_t the gradient of the sequence NLL with respect to the
output y_t. A weight perturbation D has second-order KL  1/2 E[<J, D>^2]  (Gauss-Newton; `model` grade beyond the
small-perturbation regime). Per sample, per arm a and block l this script records s = <J_l, D_{a,l}>, D = W_a - W_f16.
With all 21 blocks perturbed at once the joint second-order KL is  1/2 E[(sum_l s_l)^2]  and the additivity ratio
K / sum_l k_l = E[(sum_l s_l)^2] / sum_l E[s_l^2]: equal to 1 when the per-layer scores are uncorrelated.

Run with ~/spinor-ttl/.venv/bin/python (torch + CUDA).
usage: fisher_capture.py OUT.json [N_CHUNKS] [K_SAMPLES]
Cost: per chunk one forward and K backward passes (retain_graph), O(N_CHUNKS * K * 512 * params) FLOPs, plus
21 blocks * 5 arms inner products of 896 x 4864 per sample; the arm deltas occupy 5 * 21 * 17.4 MB = 1.8 GB on the GPU.
"""
import json, os, sys, time
import numpy as np
import torch
import transformers

sys.path.insert(0, "~/llama.cpp/gguf-py")
import gguf
from gguf.quants import dequantize

N_IN, N_OUT, CHUNK = 4864, 896, 512
BLOCKS = list(range(3, 24))
CALIB = "~/seal/calib-c4.txt"
FILES = {
    "f16": "~/models/gguf/qwen05b-f16.gguf",
    "iq1_s": "~/onebit-w9/qwen05b-IQ1_S-imat.gguf",
    "iq1_m": "~/onebit-w9/qwen05b-IQ1_M-imat.gguf",
    "q4_k_m": "~/seal/qwen05b-Q4_K_M-imat.gguf",
    "rtn": "~/onebit-w9/p1-rtn-f16.gguf",
    "had_gptq1": "~/onebit-w9/p1-hadgptq1-f16.gguf",
}
ARMS = [a for a in FILES if a != "f16"]


def read_tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                a = np.asarray(t.data, dtype=np.float32)
            else:
                a = dequantize(t.data, t.tensor_type)
            return torch.from_numpy(np.ascontiguousarray(a.reshape(N_OUT, N_IN))).float()
    raise KeyError(name)


def load_deltas():
    readers = {k: gguf.GGUFReader(v) for k, v in FILES.items()}
    out = {}
    for b in BLOCKS:
        name = f"blk.{b}.ffn_down.weight"
        w = read_tensor(readers["f16"], name)
        for a in ARMS:
            out[(a, b)] = (read_tensor(readers[a], name) - w).cuda().to(torch.float64)
    return out


def main(out_path, n_chunks, k_samples):
    torch.manual_seed(0)
    t0 = time.time()
    delta = load_deltas()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ids = tok.encode(open(CALIB, encoding="utf-8").read())
    assert len(ids) >= n_chunks * CHUNK, (len(ids), n_chunks * CHUNK)
    model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    leaves, xs = {}, {}

    def make_hooks(b):
        def pre(module, args):
            xs[b] = args[0].detach().reshape(-1, N_IN)

        def post(module, args, out):
            y = out.detach().requires_grad_(True)
            leaves[b] = y
            return y

        return pre, post

    for b in BLOCKS:
        pre, post = make_hooks(b)
        mlp = model.model.layers[b].mlp.down_proj
        mlp.register_forward_pre_hook(pre)
        mlp.register_forward_hook(post)
    s_all = np.zeros((n_chunks * k_samples, len(ARMS), len(BLOCKS)))
    row = 0
    for c in range(n_chunks):
        x = torch.tensor([ids[c * CHUNK:(c + 1) * CHUNK]], device="cuda")
        logp = torch.log_softmax(model(x).logits[0].float(), -1)
        for k in range(k_samples):
            s = torch.multinomial(logp.exp().detach(), 1)
            loss = -logp.gather(1, s).sum()
            for b in BLOCKS:
                leaves[b].grad = None
            loss.backward(retain_graph=k < k_samples - 1)
            for j, b in enumerate(BLOCKS):
                g = leaves[b].grad.reshape(-1, N_OUT).double()
                jac = g.t() @ xs[b].double()
                for i, a in enumerate(ARMS):
                    s_all[row, i, j] = float((jac * delta[(a, b)]).sum())
            row += 1
        leaves.clear()
        xs.clear()
    n = s_all.shape[0]
    res = {"n_samples": n, "n_chunks": n_chunks, "k_samples": k_samples, "seed": 0, "wall_s": time.time() - t0,
           "arms": {}}
    for i, a in enumerate(ARMS):
        per_block = (s_all[:, i, :] ** 2).mean(0) / 2
        joint = (s_all[:, i, :].sum(1) ** 2).mean() / 2
        sq = (s_all[:, i, :].sum(1) ** 2) / 2
        res["arms"][a] = {
            "kl_per_block": per_block.tolist(),
            "kl_sum": float(per_block.sum()),
            "kl_joint": float(joint),
            "kl_joint_sem": float(sq.std(ddof=1) / np.sqrt(n)),
            "additivity_ratio": float(joint / per_block.sum()),
        }
        print(a, "sum", per_block.sum(), "joint", joint, "ratio", joint / per_block.sum(), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    np.save(out_path + ".scores.npy", s_all)
    print("wall", res["wall_s"])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
