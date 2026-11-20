# One bit against four: 2.4× fewer instructions per token, 4.2× fewer bytes, and neither reaches 20

Criterion: **T1** (decode) on the 1.174 b/w production path, with **Q1** as the byte context. Grade:
`measured` for the per-token byte counts (GGUF directories, `tools/nq_gguf.py`), the Q4_K decode
stream (543 static instructions per 256 weights), the byte-plane and `i64`-lane sign dots (paired
passes in `bench/cpu/1794812057-nq-sign-dot-simd.md`) and the 0.6B end-to-end run; `model` for every
tok/s figure, which combines those counts with a core and IPC assumption. Host: i5-10210U, 4C/8T,
15 W, 6.6 GiB, Ubuntu 24.04, `performance`, 27.97 GB/s measured streaming, 42.656 GB/s spec. This
note closes no criterion and no 1.174 b/w pack exists.

**Correction.** An earlier revision of this note divided the `sign_dot_avx2` loop count by four
times its real term count: the loop advances `add $0x4,%r10`, one `u32` word, **32 terms** per
iteration, so its 61 instructions are 1.91 per term, not 0.477. Every figure that inherited the
0.477 (8.7 tok/s, "0.208 instructions/term", "3.4–3.8 instructions per packed byte", "the 1-bit
advantage is entirely the bytes") is replaced here.

## 1. The premise: 7B Q4_K_M is not a 20 tok/s configuration on this host

| configuration | per-token read | b/w | 27.97 GB/s | 42.656 GB/s |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct Q4_K_M (the live demo) | 391,859,712 B | 4.82 | 71.38 | 108.86 |
| Qwen2.5-7B-Instruct Q4_K_M | 4,370,558,976 B | 4.945 | **6.40** | 9.76 |
| 1.174 b/w NQ trunk + Q6_K head | 1,404,683,392 B | — | **19.91** | 30.37 |
| 1.174 b/w NQ trunk + rank-4096 NQ head | 1,037,626,532 B | 1.17404 | **26.96** | 41.11 |

The 35–41 tok/s in `demo/README.md` is the **0.5B**: 0.392 GB/token, a 71.4 tok/s bytes ceiling, and
the demo holds 49–57% of it. The 7B Q4_K_M file reads 4.371 GB/token and cannot exceed **6.40 tok/s**
at 27.97 GB/s regardless of the kernel; at the spec 42.656 GB/s it cannot exceed 9.76. Q4_K_M's
`token_embd` (306,561,024 B) is not read at decode; its `output.weight` is Q6_K (447,068,160 B) and
is. 7,070,619,136 per-token weights.

Against the **T1 target** (30 tok/s at 2k, 24 at 200k) rather than the 20 floor: the NQ-head pack's
26.96 bytes ceiling is **1.11× short of 30**, and the instruction bound in §2 is **4.0–5.8× short of
24** and worse against 30. Nothing here is near the target.

## 2. Instructions per token: the 1-bit path is 2.4× cheaper, not equal

| carrier | static stream | per term | per weight | per token | tok/s at 1 IPC | at 2 IPC |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K (`Q4_KDecoder::decode_step_avx2`) | 543 instr / 256 weights | — | 2.121 | 14.997G | — | 2.13 (3.20 at 3 IPC) |
| NQ, `sign_dot_avx2` (`i64` lanes) | 58 instr / 32 terms | 1.8125 | 2.115 | 13.799G | 1.16 | 2.32 |
| NQ, `sign_dot_planes_avx2` (byte planes) | 24 instr / 32 terms | **0.75** | 0.875 | 5.710G | 2.80 | **5.60** |
| NQ byte planes, trunk + NQ head | — | — | — | 6.190G | 2.59 | **5.17** |

`5.710G = 7,613,455,360 terms × 0.75`; the head adds `637,534,208` terms (`U` 152,064×4,096 plus
`V` 4,096×3,584) at `0.478G` instructions, 8.4% on top of the trunk, and it replaces a Q6_K head of
544,997,376 weights at ~2.1 instructions each = **1.145G** (`model`: the Q6_K carrier's rate is not
measured here). The fitted head is both 5.6× smaller and ~2.4× cheaper in instructions.

IPC is 4 physical cores × 4.0 GHz: 1 IPC = 16G instructions/s, 2 IPC = 32G. The byte-plane kernel
*measures* 2.1 IPC on one core (`1.49 tok/s × 5.710G = 8.5G instructions/s`), so 5.60 at 2 IPC is
the extrapolation and 5.96 is what four cores at the measured single-core IPC would give. The pool
measured **3.64 tok/s** (`274.91 ms/token`) on a shell that sees 2 CPUs with width 8 — 2.44× the
single-core rate, not 4×. The honest range is **3.6 measured to 5.6 modelled**.

The carriers do **not** issue the same instructions per packed byte:

| carrier | instructions/weight | bytes/weight | instructions/packed byte |
|---|---:|---:|---:|
| Q4_K | 2.121 | 0.618 | 3.43 |
| NQ, `i64` lanes | 2.115 | 0.147 | 14.4 |
| NQ, byte planes | 0.875 | 0.147 | 5.96 |

The 1-bit pack issues **1.7× more instructions per byte** than Q4_K and reads **4.2× fewer bytes**,
so it is **2.4× cheaper per token** on the binding bound. Its advantage is not the bytes alone: the
sign dot converts a 4-bit decode step into 0.875 instructions per weight, and that is what makes the
bytes advantage spendable.

## 3. The budget, and why 20 tok/s is unreachable here

At 1.0376 GB/token and the NQ head, `8.251G terms` and `6.19G instructions/token`:

| target | bytes/s | share of measured 27.97 | instructions/token at 2 IPC | instructions/term | per 32-term word |
|---|---:|---:|---:|---:|---:|
| 20 tok/s | 20.75 GB/s | 74.2% | 1.600G | 0.194 | **6.2** |
| 24 tok/s | 24.90 GB/s | 89.0% | 1.333G | 0.162 | 5.2 |
| 30 tok/s | 31.13 GB/s | 111.3% | 1.067G | 0.129 | 4.1 |

The byte-plane form is 24 instructions per word, 3.9× over the 20 budget. Of those 24, five build
the 32 `i8` selectors (`vpbroadcastd`, `vpsrlvd`, two `vpshufb`, `vpand`) and twelve are the four
digit planes (four each of `vpmaddubsw`, `vpmaddwd`, `vpaddd`), plus loop control. **Any exact form
needs one `vpmaddubsw`/`vpmaddwd`/`vpaddd` triple per byte plane of `t`, and `t` needs four planes**
(`max|t| = 13,414,400 = 2^23.68`, so three planes are inadmissible; a three-plane form would be
6.41 tok/s at 2 IPC and 3.12× short anyway). Four planes alone are twelve instructions, twice the
whole 20 tok/s budget, before a selector is built.

The single-plane escape is an `i8` activation quantisation: five to build the selectors, one
`vpmaddubsw`+`vpmaddwd`+`vpaddd` triple, plus control, about nine per word = 0.28/term →
**14.5 tok/s at 2 IPC**, still short of 20, and lossy (`Q2`). `vpdpbusd` on AVX-VNNI removes one
instruction from that chain and this host does not have it. At 3 IPC the four-plane form reaches
7.7, the single-plane form 21.8 — and 3 IPC on a 15 W 4-core part is not a measured regime here.

**Therefore: at 1.174 b/w, 20 tok/s is not reachable on this laptop.** To fit the budget with the
current 0.75/term kernel, the pack would need ≤ 2.13G terms/token, 3.6× fewer, i.e. about 0.33 b/w —
not a 7B at 1.15–1.18. At 0.25/term it would need ≤ 6.4G terms, about 0.92 b/w. The 1-bit pack is
the right idea and it is 2.4× the 4-bit path; the 20 floor is a property of the issue rate, not of
the quantiser.

## 4. The 0.6B end-to-end run

The upgraded v1 pack (`tools/nq_upgrade_v1.py`, 196 sections, 53,815,472 B, bit-identical scales)
ran through `examples/generate` on the 0.6B skeleton: **1.23 / 2.19 / 3.44 tok/s at 1 / 2 / 4
threads**, 418,381,824 terms/token. That is 1.44G terms/s, 19× below the 27.69G terms/s the isolated
7B shapes reach, so the 0.6B token is **not** spent in the sign dot: it is the F16 head
(155.7M weights read per token) and the per-call overhead of linears that are ~20× smaller than the
7B's. The 0.6B number is a fixture for `serve/`, not evidence about the 7B kernel.

## 5. What this does not establish

- No 1.174 b/w pack exists; every 7B figure is a static count times an IPC assumption, except the
  3.64 tok/s pool run and the 1.49 tok/s single-core pair.
- 2 and 3 IPC are assumptions; the measured single-core rate is 2.1 IPC for the byte-plane kernel.
- The pool measured 2.44× on 2 CPUs; four-core behaviour on the bare laptop is not measured.
- The Q6_K head's ~2.1 instructions/weight is extrapolated from Q4_K, not counted.
- The activation and `s_pre` in the paired pass are synthetic; the real `t`/`z` distribution is
  unmeasured.
- rustc and LLVM code generation, the kernel ABI, Vulkan and the GPU remain outside every claim.
