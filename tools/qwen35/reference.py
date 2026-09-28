#!/usr/bin/env python3
"""Float64 NumPy reference forward pass for Qwen3.5-0.8B.

Serves as the bit-level debugging oracle for exact integer hybrid engine development.
Matches llama.cpp greedy decoding on CPU token by token.
"""

import argparse
import os
import sys

import numpy as np

# Ensure tokenizers and gguf-py are discoverable in environments where not installed globally
for p in [
    "/home/socol/Workspace/llama.cpp/gguf-py",
    "/home/socol/oc-tmp/exl3/.venv/lib/python3.12/site-packages",
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

import gguf
from gguf import quants
import tokenizers

DEFAULT_MODEL = "/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf"
DEFAULT_TOKENIZER = "/home/socol/models/qwen3.5-0.8b/tokenizer.json"

PROMPT_1_TEXT = "The capital of France is"

# Ground-truth 32 greedy decode IDs from llama.cpp build-cuda / build-cpu (-ngl 0, temp 0, top-k 1)
LLAMA_PROMPT_1_IDS = [11751, 13, 198, 760, 6511, 314, 9338, 369] * 4

LLAMA_PROMPT_2_IDS = [
    12182, 9251, 2349, 1074, 5039, 877, 14, 1825, 42427, 14, 1825, 42427,
    33673, 14905, 14, 52838, 20668, 553, 271, 760, 1510, 49, 8065, 9477,
    7, 87, 61, 17, 471, 19638, 44646, 17189,
]


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class Qwen35Model:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        reader = gguf.GGUFReader(model_path)
        t_map = {t.name: t for t in reader.tensors}

        def load_w(name: str) -> np.ndarray:
            t = t_map[name]
            return quants.dequantize(t.data, t.tensor_type).astype(np.float64)

        self.tok_embd = load_w("token_embd.weight")  # (248320, 1024)
        self.output_norm = load_w("output_norm.weight")  # (1024,)

        self.layers = []
        for il in range(24):
            pfx = f"blk.{il}."
            is_attn = (il % 4 == 3)
            layer = {
                "is_attn": is_attn,
                "attn_norm": load_w(pfx + "attn_norm.weight"),
                "post_attn_norm": load_w(pfx + "post_attention_norm.weight"),
                "ffn_gate": load_w(pfx + "ffn_gate.weight"),
                "ffn_up": load_w(pfx + "ffn_up.weight"),
                "ffn_down": load_w(pfx + "ffn_down.weight"),
            }
            if is_attn:
                layer["attn_q"] = load_w(pfx + "attn_q.weight")
                layer["attn_k"] = load_w(pfx + "attn_k.weight")
                layer["attn_v"] = load_w(pfx + "attn_v.weight")
                layer["attn_output"] = load_w(pfx + "attn_output.weight")
                layer["attn_q_norm"] = load_w(pfx + "attn_q_norm.weight")
                layer["attn_k_norm"] = load_w(pfx + "attn_k_norm.weight")
            else:
                layer["attn_qkv"] = load_w(pfx + "attn_qkv.weight")
                layer["attn_gate"] = load_w(pfx + "attn_gate.weight")
                layer["ssm_conv1d"] = load_w(pfx + "ssm_conv1d.weight")
                layer["ssm_dt"] = load_w(pfx + "ssm_dt.bias")
                layer["ssm_a"] = load_w(pfx + "ssm_a")
                layer["ssm_beta"] = load_w(pfx + "ssm_beta.weight")
                layer["ssm_alpha"] = load_w(pfx + "ssm_alpha.weight")
                layer["ssm_norm"] = load_w(pfx + "ssm_norm.weight")
                layer["ssm_out"] = load_w(pfx + "ssm_out.weight")
            self.layers.append(layer)


class ModelState:
    def __init__(self, rope_mode: str = "neox"):
        self.rope_mode = rope_mode
        self.conv_states = [np.zeros((6144, 3), dtype=np.float64) for _ in range(24)]
        self.ssm_states = [np.zeros((16, 128, 128), dtype=np.float64) for _ in range(24)]
        self.kv_k = [[] for _ in range(24)]
        self.kv_v = [[] for _ in range(24)]


def forward_step(
    model: Qwen35Model,
    token_id: int,
    pos: int,
    state: ModelState,
    record_dumps: bool = False,
) -> tuple[np.ndarray, dict | None]:
    dumps = {} if record_dumps else None
    x = model.tok_embd[token_id].copy()

    for il, layer in enumerate(model.layers):
        if record_dumps and il in (0, 3):
            dumps[f"layer{il}_input"] = x.copy()

        h = rms_norm(x, layer["attn_norm"])

        if layer["is_attn"]:
            q_full = layer["attn_q"] @ h
            q_heads = q_full.reshape(8, 2, 256)[:, 0, :].copy()
            gate_heads = q_full.reshape(8, 2, 256)[:, 1, :].copy()

            k_heads = (layer["attn_k"] @ h).reshape(2, 256).copy()
            v_heads = (layer["attn_v"] @ h).reshape(2, 256).copy()

            for h_idx in range(8):
                q_heads[h_idx] = rms_norm(q_heads[h_idx], layer["attn_q_norm"])
            for kv_idx in range(2):
                k_heads[kv_idx] = rms_norm(k_heads[kv_idx], layer["attn_k_norm"])

            # RoPE on first 64 dimensions
            j_indices = np.arange(32)
            thetas = pos * (1e7 ** (-2.0 * j_indices / 64.0))
            cos_th = np.cos(thetas)
            sin_th = np.sin(thetas)

            if state.rope_mode == "neox":
                # NeoX halves: (j, j + 32)
                q0 = q_heads[:, :32].copy()
                q1 = q_heads[:, 32:64].copy()
                q_heads[:, :32] = q0 * cos_th - q1 * sin_th
                q_heads[:, 32:64] = q0 * sin_th + q1 * cos_th

                k0 = k_heads[:, :32].copy()
                k1 = k_heads[:, 32:64].copy()
                k_heads[:, :32] = k0 * cos_th - k1 * sin_th
                k_heads[:, 32:64] = k0 * sin_th + k1 * cos_th
            else:
                # Adjacent pairing control: (2*j, 2*j + 1)
                q0 = q_heads[:, 0:64:2].copy()
                q1 = q_heads[:, 1:64:2].copy()
                q_heads[:, 0:64:2] = q0 * cos_th - q1 * sin_th
                q_heads[:, 1:64:2] = q0 * sin_th + q1 * cos_th

                k0 = k_heads[:, 0:64:2].copy()
                k1 = k_heads[:, 1:64:2].copy()
                k_heads[:, 0:64:2] = k0 * cos_th - k1 * sin_th
                k_heads[:, 1:64:2] = k0 * sin_th + k1 * cos_th

            state.kv_k[il].append(k_heads)
            state.kv_v[il].append(v_heads)

            K = np.stack(state.kv_k[il], axis=0)  # (T, 2, 256)
            V = np.stack(state.kv_v[il], axis=0)  # (T, 2, 256)

            scale = 1.0 / 16.0
            attn_out_heads = np.zeros((8, 256), dtype=np.float64)
            for head_idx in range(8):
                kv_idx = head_idx // 4
                scores = (K[:, kv_idx, :] @ q_heads[head_idx]) * scale
                max_sc = np.max(scores)
                exp_sc = np.exp(scores - max_sc)
                attn_w = exp_sc / np.sum(exp_sc)
                head_out = attn_w @ V[:, kv_idx, :]
                gate_sig = 1.0 / (1.0 + np.exp(-gate_heads[head_idx]))
                attn_out_heads[head_idx] = head_out * gate_sig

            mixer_out = layer["attn_output"] @ attn_out_heads.reshape(2048)
        else:
            qkv = layer["attn_qkv"] @ h
            z = layer["attn_gate"] @ h
            beta = 1.0 / (1.0 + np.exp(-(layer["ssm_beta"] @ h)))
            alpha = layer["ssm_alpha"] @ h + layer["ssm_dt"]
            g = layer["ssm_a"] * softplus(alpha)

            conv_in = np.column_stack([state.conv_states[il], qkv])
            state.conv_states[il] = conv_in[:, 1:]
            conv_raw = np.sum(conv_in * layer["ssm_conv1d"], axis=1)
            conv_out = silu(conv_raw)

            q_conv = conv_out[0:2048].reshape(16, 128).copy()
            k_conv = conv_out[2048:4096].reshape(16, 128).copy()
            v_conv = conv_out[4096:6144].reshape(16, 128).copy()

            # L2 normalise per head: x / max(||x||, 1e-6)
            q_l2 = np.maximum(np.linalg.norm(q_conv, axis=-1, keepdims=True), 1e-6)
            k_l2 = np.maximum(np.linalg.norm(k_conv, axis=-1, keepdims=True), 1e-6)
            q_norm = q_conv / q_l2
            k_norm = k_conv / k_l2
            q_scaled = q_norm / np.sqrt(128.0)

            decay = np.exp(g)
            o_heads = np.zeros((16, 128), dtype=np.float64)
            for h_idx in range(16):
                S = decay[h_idx] * state.ssm_states[il][h_idx]
                sk = k_norm[h_idx] @ S
                d = beta[h_idx] * (v_conv[h_idx] - sk)
                S = S + np.outer(k_norm[h_idx], d)
                state.ssm_states[il][h_idx] = S
                o_heads[h_idx] = q_scaled[h_idx] @ S

            z_heads = z.reshape(16, 128)
            norm_o = np.zeros((16, 128), dtype=np.float64)
            for h_idx in range(16):
                norm_o[h_idx] = rms_norm(o_heads[h_idx], layer["ssm_norm"]) * silu(z_heads[h_idx])

            mixer_out = layer["ssm_out"] @ norm_o.reshape(2048)

            if record_dumps and il == 0:
                dumps["layer0_conv_out"] = conv_out.copy()
                dumps["layer0_q_norm"] = q_norm.copy()
                dumps["layer0_k_norm"] = k_norm.copy()
                dumps["layer0_v_norm"] = v_conv.copy()
                dumps["layer0_beta"] = beta.copy()
                dumps["layer0_g"] = g.copy()
                dumps["layer0_state_frob_norm"] = np.linalg.norm(state.ssm_states[0])
                dumps["layer0_o_pregate"] = norm_o.copy()

        if record_dumps and il in (0, 3):
            dumps[f"layer{il}_mixer_out"] = mixer_out.copy()

        x = x + mixer_out

        h_ffn = rms_norm(x, layer["post_attn_norm"])
        gate_val = silu(layer["ffn_gate"] @ h_ffn)
        up_val = layer["ffn_up"] @ h_ffn
        ffn_out = layer["ffn_down"] @ (gate_val * up_val)
        x = x + ffn_out

        if record_dumps and il in (0, 3):
            dumps[f"layer{il}_block_out"] = x.copy()

    final_x = rms_norm(x, model.output_norm)
    logits = model.tok_embd @ final_x
    return logits, dumps


def get_prompt2_ids(tok: tokenizers.Tokenizer, readme_path: str = "README.md") -> list[int]:
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()
    body = text.split("# apeiron\n\n", 1)[1] if "# apeiron\n\n" in text else text
    enc = tok.encode(body)
    return enc.ids[:200]


def run_decode(
    model: Qwen35Model,
    prompt_ids: list[int],
    n: int = 32,
    dump_dir: str | None = None,
    rope_mode: str = "neox",
) -> tuple[list[int], list[tuple[int, float, float]]]:
    state = ModelState(rope_mode=rope_mode)
    pos = 0

    # Ingest prompt
    logits = None
    for tid in prompt_ids:
        logits, _ = forward_step(model, tid, pos, state, record_dumps=False)
        pos += 1

    gen_ids = []
    top2_gaps = []

    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)

    for gen_idx in range(n):
        record_dumps = (dump_dir is not None and gen_idx < 2)

        # Logit top-2 gap
        top2 = np.partition(logits, -2)[-2:]
        gap = float(top2[1] - top2[0])
        next_id = int(np.argmax(logits))

        gen_ids.append(next_id)
        top2_gaps.append((next_id, logits[next_id], gap))

        logits, dumps = forward_step(model, next_id, pos, state, record_dumps=record_dumps)
        pos += 1

        if record_dumps and dumps is not None:
            for k, arr in dumps.items():
                out_path = os.path.join(dump_dir, f"pos{gen_idx}_{k}.npy")
                np.save(out_path, arr)

    if dump_dir is not None:
        ids_path = os.path.join(dump_dir, "generated_ids.txt")
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write(" ".join(str(i) for i in gen_ids) + "\n")

    return gen_ids, top2_gaps


def check_prompts(model: Qwen35Model, tok: tokenizers.Tokenizer) -> int:
    print("=== Running --check verification against llama.cpp greedy decoding ===")
    p1_ids = tok.encode(PROMPT_1_TEXT).ids
    print(f"Prompt 1: {repr(PROMPT_1_TEXT)} ({len(p1_ids)} tokens)")
    gen1, gaps1 = run_decode(model, p1_ids, n=32)

    p1_match = (gen1 == LLAMA_PROMPT_1_IDS)
    print(f"Prompt 1 greedy 32 IDs match: {p1_match}")
    if not p1_match:
        for idx, (got, exp) in enumerate(zip(gen1, LLAMA_PROMPT_1_IDS)):
            if got != exp:
                gap = gaps1[idx][2]
                print(f"  P1 mismatch at step {idx}: got {got} expected {exp}, top2 gap={gap:.6f}")
                if gap >= 1e-3:
                    return 1

    p2_ids = get_prompt2_ids(tok)
    print(f"Prompt 2: README.md opening ({len(p2_ids)} tokens)")
    gen2, gaps2 = run_decode(model, p2_ids, n=32)

    p2_match = (gen2 == LLAMA_PROMPT_2_IDS)
    print(f"Prompt 2 greedy 32 IDs match: {p2_match}")
    if not p2_match:
        for idx, (got, exp) in enumerate(zip(gen2, LLAMA_PROMPT_2_IDS)):
            if got != exp:
                gap = gaps2[idx][2]
                print(f"  P2 mismatch at step {idx}: got {got} expected {exp}, top2 gap={gap:.6f}")
                if gap >= 1e-3:
                    return 1

    print("ALL 32 TOKENS MATCH LLAMA.CPP GREEDY DECODE FOR BOTH PROMPTS.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B float64 NumPy reference forward")
    parser.add_argument("--prompt-ids", type=str, help="Path to file containing prompt token IDs")
    parser.add_argument("--prompt", type=str, help="Raw prompt string")
    parser.add_argument("--n", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--dump-dir", type=str, help="Directory to save activation dumps and generated IDs")
    parser.add_argument("--check", action="store_true", help="Run parity check on standard prompts against llama.cpp")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to Q8_0 GGUF model")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="Path to tokenizer.json")
    parser.add_argument("--rope-mode", type=str, default="neox", choices=["neox", "adjacent"])

    args = parser.parse_args()

    tok = tokenizers.Tokenizer.from_file(args.tokenizer)
    model = Qwen35Model(args.model)

    if args.check:
        sys.exit(check_prompts(model, tok))

    if args.prompt_ids:
        with open(args.prompt_ids, "r", encoding="utf-8") as f:
            content = f.read().strip()
        tokens = [int(x) for x in content.replace(",", " ").split()]
    elif args.prompt:
        tokens = tok.encode(args.prompt).ids
    else:
        tokens = tok.encode(PROMPT_1_TEXT).ids

    gen_ids, _ = run_decode(model, tokens, n=args.n, dump_dir=args.dump_dir, rope_mode=args.rope_mode)
    print("Generated token IDs:")
    print(" ".join(str(i) for i in gen_ids))
    print("\nDecoded text:")
    print(tok.decode(gen_ids))


if __name__ == "__main__":
    main()
