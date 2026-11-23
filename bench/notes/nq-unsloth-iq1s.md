# Unsloth's 1-bit Qwen3-8B: IQ1_S allocation, instruct, 1.81 b/w on the linears

The question: can `unsloth/Qwen3-8B-GGUF` replace the Base campaign or supply a 1.00 b/w pack?
No. The smallest file is the **instruct** checkpoint at **1.8094 b/w over the linear weights**
(2.2147 b/w whole payload). The method is llama.cpp's IQ1_S/IQ1_M codebooks plus an importance
matrix and a per-tensor map; the transferable part is the map, not the carrier or the result.

## The file

`Qwen3-8B-UD-IQ1_S.gguf`, 2,275,379,008 B, 8,190,735,360 parameters, 399 tensors, GGUF v3.
`base_model` is `Qwen/Qwen3-8B` (instruct), not `Qwen3-8B-Base`. Header 5,957,439 B; computed
tensor bytes 2,267,455,488; total 2,273,412,927, within 0.09 % of the file size (trailing padding).

The first parse read 4.2540 b/w because it used a type enum without the two removed IDs at 4-5;
the corrected enum (`ggml.h`, `GGML_TYPE_Q4_2`/`Q4_3` removed) gives the mix below. The 4.2540 is
withdrawn; the byte total is the check.

| type | tensors | elements | b/w |
|---|---:|---:|---:|
| IQ1_S | 106 | 4,936,695,808 | 1.5625 |
| IQ1_M | 52 | 545,259,520 | 1.75 |
| IQ2_XXS | 46 | 708,837,376 | 2.0625 |
| IQ2_S | 5 | 251,658,240 | 2.5 |
| IQ3_S | 5 | 251,658,240 | 3.4375 |
| Q2_K | 2 | 100,663,296 | 2.625 |
| Q3_K | 1 | 622,329,856 | 3.4375 |
| Q4_K | 36 | 150,994,944 | 4.5 |
| Q5_K | 1 | 622,329,856 | 5.5 |
| F32 | 145 | 308,224 | 32 |

Whole payload 2.2147 b/w; linear-only 1.8094 b/w over 6,945,767,424 elements.

## The allocation

| kind | type | layers |
|---|---|---|
| `attn_v` | Q4_K | all 36 |
| `attn_output` | IQ2_XXS | all 36 |
| `ffn_gate`, `ffn_up` | IQ1_S | all 36 |
| `attn_q`, `attn_k` | IQ1_M | 2-5, 9-10, 12-15, 19-34 |
| | IQ2_XXS | 0, 1, 6, 16, 35 |
| | IQ1_S | 7, 8, 11, 17, 18 |
| `ffn_down` | IQ1_S | default |
| | Q2_K | 2, 3 |
| | IQ3_S | 0, 1, 6, 16, 35 |
| | IQ2_S | 7, 8, 11, 17, 18 |
| `token_embd` | Q3_K | - |
| `output` | Q5_K | - |
| norms | F32 | - |

The map protects `v` (4.5) and `o` (2.06), keeps gate/up at the 1.56 floor, and raises `down` at
the first/last layers and the same periodic set. It is the same lever the Q2 receipt names:
allocation, not the average rate.

## What the method is

- IQ1_S is 50 B per 256 weights (1.5625 b/w); IQ1_M is 56 B (1.75). Both are llama.cpp codebooks
  (`ggml-quants.c`), selected per block by a scale search weighted by the importance matrix.
- Unsloth's repository is training kernels; the GGUF quantizer is llama.cpp's `llama-quantize`
  with `--imatrix` and a per-tensor type map. The "Dynamic" part is the map and the matrix.
- The 1.58-bit headline is MoE: DeepSeek-R1's 131 GB at IQ1_S is 88 % sparsely routed expert
  weights. A dense 8B has no equivalent redundancy, and the smallest dense Qwen3-8B file here is
  1.81 linear b/w.

## What is borrowable

1. **The allocation prior.** Protect `v`/`o`, floor gate/up, raise `down` at the ends and the
   periodic set. The absolute types do not transfer to the sign-factor rank dial; the relative
   protection does.
2. **The imatrix as a cross-check.** The reference path already uses the activation Gram diagonal
   (`nq_fit --diag-dir`). `imatrix_unsloth.dat` (5,316,778 B) is another estimate of the same
   object; it can confirm the per-kind ranking, not replace the preconditioner mid-comparison.
3. **Nothing else.** The IQ1_S carrier is a new decode and packer. At a uniform 1.00 b/w it sits
   below its design point, and no 8B Base result exists for it.

## Disposition

- **Not Q3.** The checkpoint is instruct; Q3 names Base.
- **Not a 1.00 b/w replacement.** 1.81 linear b/w is 81 % above the target; at 1.00 uniform the
  protected tensors lose the bits the map spends on them.
- **Useful as an external baseline and an allocation prior.** The file runs under llama.cpp, not
  this engine; porting IQ1_S is a separate carrier and fit, not a config change.
- The Base campaign stays. The next pack's per-kind plan uses the prior above; the six-task 0-shot
  screen judges it.
