# The 200k KV budget: what 1 GiB buys, and why the file-backed plane does not close C1

Status: `model` (geometry and byte arithmetic from the named source constants; no 200k state has
been allocated). Criterion: **C1**. The checkpoint is Qwen2.5-7B-Instruct: 28 layers, 4 KV heads,
head_dim 128 (`GOAL.md`, `bench/cpu/1794781140-nq7b-118-full-pack.md`).

## The state

Per token, per layer: `2 · 4 · 128 = 1024` values (one key and one value for each of 4 KV heads).
Over 28 layers: **28,672 values per token**; at **200,000 tokens**: **5,734,400,000 values**.

| carrier | bytes/value | 200k state | source |
|---|---:|---:|---|
| classic f16 KV | 2 | 11,468,800,000 B (10.68 GiB) | `src/engine/attn.rs`, f16 path |
| `Carrier::F32` | 4 | 22,937,600,000 B (21.36 GiB) | `KvState::with_carrier` |
| `Carrier::Q32` | 8 | 45,875,200,000 B (42.7 GiB) | `KvState::with_carrier` |

C1 allows **1 GiB = 1,073,741,824 B** for hot tier, cold store and index together. Divided by
5,734,400,000 values that is **1.4976 bits per value**, all-in. The cap is a rate, not a memory
tier: moving the same words to a file (`KvState::with_file`, `src/engine/attn/kvplane.rs`) leaves
the state at 22.9 GB on disk and does not advance C1. What it changes is the resident anonymous
set, which is M1's number, not C1's.

## What fits

| store | bits/value | 200k bytes | fits 1 GiB |
|---|---:|---:|---|
| 8-bit + per-32 scale | 8.5 | 6.09 GB | no |
| 4-bit + per-32 scale | 4.5 | 3.23 GB | no |
| 2-bit + per-32 scale | 2.5 | 1.79 GB | no |
| 1-bit sign + per-64 f16 scale | 1.25 | 896 MB | yes, with ~178 MB left |
| 1-bit sign, one f16 scale per vector | 1.03 | 739 MB | yes |

The index is not free. The exact-on-certificate summary (`src/engine/attn/cert.rs`) stores
`head_dim` `i64` means and one `i128` radius per block per KV head per layer: 1,040 B per block
per head, 4 heads × 28 layers = 116,480 B per block. At block 256, 782 blocks = **91.1 MB**; at
block 512, 391 blocks = **45.5 MB**. A sign-only cold store for the first 198,000 tokens
(710 MB), a 2,000-token f16 hot tier (115 MB) and the block-512 index are **869 MB**. A 1.5-bit
store is 1.075 GB and misses the cap by 1.5 MB before any index is counted.

## Consequence

C1 requires a **sign or 1.25-bit cold tier**, and the hot tier is at most a few thousand tokens.
The open question is C2, not bytes: a sign-only key changes every score and a sign-only value
changes every output. The certificate above proves which blocks contribute exactly zero; it
cannot make a sign-only block exact.

The next step is a carrier of 1 bit per value with a per-64 `f16` scale, read by
`attn_head` with the same fixed-point contract, compared with the F32 carrier under a stated
error bound on a small geometry (the agreement test) and then scored at 200k (C2). The
file-backed plane stays as the M1 mechanism. Neither exists yet; this note fixes the rate the
design must hit.
