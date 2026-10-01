# Spike: Can Qwen on Ternary Planes Beat Qwen on its GGUF Formats, on this CPU?

Task `APEIRON-SPIKE-TP` (id `spike-tp`).
Investigation and empirical measurement of radix-3 balanced ternary-plane representations ($L \in \{2, 3\}$) for `Qwen2.5-0.5B` shapes on x86_64 AVX2 hardware (`Intel Core i9-14900KF`), compared directly against the current native Q8Act chart kernels (`Q5_0`, `Q6_K`, `Q8_0`).

---

## 1. Questions & Scope

1. **Compute**: How does the wall-clock execution time of $L \in \{2, 3\}$ ternary-plane matvecs—which share a single activation table build across all $L$ planes—compare to current Q8Act kernels (`Q5_0`, `Q6_K`, `Q8_0`) across standard `Qwen2.5-0.5B` shapes (`gate/up` $4864 \times 896$, `down` $896 \times 4864$, `lm_head` $151936 \times 896$) at single-thread and 8-thread (`Pool::global`) execution?
2. **Bytes**: How do the bytes/token and effective bits/weight compare across:
   - (a) `PQ2_0` layout per plane as-is (34 bytes / 128 weights / plane);
   - (b) A packed layout sharing one scale set across planes (1 scale per 128 weights, 2-bit codes per plane);
   - (c) The entropy-coded ideal from `TERNARY_FINDINGS.md` ($4.51\text{ b/w}$ for $L=3$, $2.86\text{ b/w}$ for $L=2$), against the real Qwen GGUF model inventory ($405,573,560\text{ bytes/token}$), with LM head included and excluded?
3. **Projection**: What is the memory-bandwidth floor (at $82\text{ GB/s}$) versus the compute-bound execution time across 8 threads, compared against today's measured Qwen decode throughput ($\approx 128\text{–}133\text{ tok/s}$)?
4. **Exactness**: Is exactness preserved under the integer linear combination $\sum_{k=0}^{L-1} 3^k \cdot \text{LUT}_k$, and how does it relate to the formal representation in `Numeric/LutPQ2.lean`?
5. **Verdict**: Is a trit-plane Qwen worth building, and if so, with which layout?

---

## 2. Integer Argument & Exactness Preservation

### 2.1 The Decomposition

Let a weight matrix $W \in \mathbb{R}^{o \times n}$ be quantized into $L$ balanced ternary planes ($T_k \in \{-1, 0, 1\}^{o \times n}$ for $k \in \{0, \dots, L-1\}$) with a shared block scale $s_b \in \mathbb{Q}_{32.32}$ per 128-weight block $b$:

$$W_{b, l} = s_b \sum_{k=0}^{L-1} 3^k T_{k, b, l}, \quad l \in \{0, \dots, 127\}.$$

* **say:** W sub b l equals s sub b times the sum over k from zero to L minus one of three to the k times T sub k b l, for l from zero to 127.
* **means:** a weight is the block scale times a balanced-ternary polyadic number, one digit per plane.
* **vars:** $W_{b,l}$ — weight of a block, $\mathbb{R}$; $s_b$ — block scale, $\mathbb{Q}_{32.32}$; $T_{k,b,l}$ — ternary digit in $\{-1,0,1\}$; $L$ — planes, $3$ here.
* **apeiron:** `none — anchor`. No corpus declaration states the $3^k$ plane decomposition. `Numeric.LutPQ2.ternaryByte` fixes the ternary digit domain $\{(\text{field } b\,k) - 1\} \subseteq \{-1,0,1\}$ only for the PQ2_0 byte split.
* **physics:** a redundant signed-digit encoding of a weight.

The activation vector $x \in \mathbb{R}^n$ is quantized by `quant_act` into 32-lane blocks with integer scale $S_{sb}$ and signed codes $xq_j \in [-127, 127]$.

In `Numeric/LutPQ2.lean`, theorem `block_dot_lut` proves that the exact integer dot product of a 128-weight block with the quantized activation decomposes linearly into the block scale multiplied by the sum of 4-column group table lookups:

$$\sum_{l=0}^{127} W_{b, l} \cdot xq_{b, l} = s_b \sum_{j=0}^{31} \text{lutEntry}(xq, \text{codeByte}_{b, j}).$$

* **say:** the sum over l of W sub b l times xq sub b l equals s sub b times the sum over j of the table entry of x q and the code byte.
* **means:** the block dot product is the block scale times a sum of 32 four-column table lookups, so no per-weight multiply runs.
* **vars:** $W_{b,l}$ — weight, $\mathbb{R}$; $xq_{b,l}$ — signed activation code in $[-127,127]$, $\mathbb{Z}$; $s_b$ — block scale; $\text{codeByte}_{b,j}$ — one byte of four codes; $\text{lutEntry}$ — the 4-entry table row, $\mathbb{Z}$.
* **apeiron:** `Numeric.LutPQ2.block_dot_lut`, proved.
* **physics:** table lookup replacing a multiply-accumulate.

### 2.2 Linearity of the LUT Combination

For any plane $k$, each 4-weight group has integer lookups $\text{inner}_{k, sb} = 2 \cdot \text{sum\_hi}_k + \text{sum\_lo}_k - \text{sum\_x}$ bounded by $| \text{inner}_{k, sb} | \le 32 \times 508 = 16,256$ (`INNER_ABS_MAX`).
Because integer multiplication and summation distribute over $\mathbb{Z}$:

$$\sum_{k=0}^{L-1} 3^k \cdot \left( s_b \sum_{sb=0}^3 S_{sb} \cdot \text{inner}_{k, sb} \right) = s_b \sum_{sb=0}^3 S_{sb} \left( \sum_{k=0}^{L-1} 3^k \cdot \text{inner}_{k, sb} \right).$$

* **say:** the sum over k of three to the k times s sub b times the sum over sb of S sub sb times inner, equals s sub b times the sum over sb of S sub sb times the sum over k of three to the k times inner.
* **means:** the plane weight and the sub-block scale both factor out of the sums, so the planes combine after the sub-block sum.
* **vars:** $3^k$ — plane weight, $\mathbb{N}$; $s_b$ — block scale; $S_{sb}$ — sub-block scale; $\text{inner}_{k,sb}$ — table lookup, $\mathbb{Z}$.
* **apeiron:** `none — anchor`. Integer distributivity; no corpus declaration.
* **physics:** associativity of summation, no physical content.

With $L=3$, the maximal combined inner across planes is:

$$\left| \sum_{k=0}^2 3^k \cdot \text{inner}_{k, sb} \right| \le (1 + 3 + 9) \times 16,256 = 13 \times 16,256 = 211,328.$$

* **say:** the absolute value of the sum over k from zero to two of three to the k times inner is at most thirteen times 16,256, which is 211,328.
* **means:** with three planes the combined lookup fits in 18 bits, so the i64 accumulator cannot overflow.
* **vars:** $\text{inner}_{k,sb}$ — table lookup with $|\text{inner}| \le 16{,}256$; $3^k$ — plane weight.
* **apeiron:** `Numeric.LutPQ2.lutEntry_bound` proves the per-entry bound $|\text{lutEntry}| \le 508$ for ternary bytes. The 13-fold combination bound over planes is `none — anchor`.
* **physics:** a range argument, no physical content.

With $S_{\max} \le 2^{40}$, the 4-sub-block accumulator $4 \times 211,328 \times 2^{40} \approx 2^{60} < 2^{63} - 1$, which fits entirely within a standard signed 64-bit integer (`i64`) without overflow.

Therefore, accumulating the exact $i128$ row sum across planes:

$$\text{Acc}_{\text{row}} = \sum_{k=0}^{L-1} 3^k \cdot \text{Acc}_{\text{row}, k}$$

* **say:** Acc row equals the sum over k of three to the k times Acc row k.
* **means:** the row accumulator is the plane-weighted sum of per-plane accumulators, in i128, so nothing rounds until the end.
* **vars:** $\text{Acc}_{\text{row}}$ — exact row accumulator, $\mathbb{Z}$; $\text{Acc}_{\text{row},k}$ — per-plane accumulator, $\mathbb{Z}$; $3^k$ — plane weight.
* **apeiron:** `none — anchor`. The corpus states no plane-weighted accumulator identity.
* **physics:** exact accumulation before a single rounding.

and performing a single final `round_shift_32` ($\lfloor (\text{Acc}_{\text{row}} + 2^{31})/2^{32} \rfloor$) produces bit-for-bit the identical result as the dequantized integer matvec of $W = s_b \sum_k 3^k T_k$. Exactness is strictly preserved.

---

## 3. Empirical Benchmark Measurements

Measurements conducted on `Intel Core i9-14900KF` (32 cores, 8 P-cores pinned, AVX2 enabled) using random payloads across three representative shapes from `Qwen2.5-0.5B`:
- `gate/up`: $4864 \times 896$ ($4,358,144$ weights)
- `down`: $896 \times 4864$ ($4,358,144$ weights)
- `lm_head`: $151936 \times 896$ ($136,134,656$ weights)

For ternary planes, the activation table is built once (via `build_tables` for 1 thread, `build_tables_pooled` for 8 threads), and the row lookups run $L$ times, combined with exact weights $3^k$.

### 3.1 Measured Performance Table

| Shape | Kernel | Threads | Median (us) | ns/weight | Effective GB/s |
|---|---|---|---|---|---|
| gate/up | ternary_L2 | 1 | 214.36 | 0.0492 | 10.80 |
| gate/up | ternary_L2 | 8 | 97.77 | 0.0224 | 23.68 |
| gate/up | ternary_L3 | 1 | 314.35 | 0.0721 | 11.05 |
| gate/up | ternary_L3 | 8 | 144.51 | 0.0332 | 24.03 |
| gate/up | Q5_0 | 1 | 210.10 | 0.0482 | 14.26 |
| gate/up | Q5_0 | 8 | 25.14 | 0.0058 | 119.16 |
| down | ternary_L2 | 1 | 183.80 | 0.0422 | 12.60 |
| down | ternary_L2 | 8 | 102.11 | 0.0234 | 22.68 |
| down | ternary_L3 | 1 | 281.96 | 0.0647 | 12.32 |
| down | ternary_L3 | 8 | 146.82 | 0.0337 | 23.65 |
| down | Q6_K | 1 | 201.69 | 0.0463 | 17.73 |
| down | Q6_K | 8 | 31.72 | 0.0073 | 112.70 |
| lm_head | ternary_L2 | 1 | 6794.46 | 0.0499 | 10.64 |
| lm_head | ternary_L2 | 8 | 2980.66 | 0.0219 | 24.26 |
| lm_head | ternary_L3 | 1 | 10075.63 | 0.0740 | 10.77 |
| lm_head | ternary_L3 | 8 | 4634.60 | 0.0340 | 23.41 |
| lm_head | Q8_0 | 1 | 9793.54 | 0.0719 | 14.77 |
| lm_head | Q8_0 | 8 | 2354.03 | 0.0173 | 61.44 |

### 3.2 Compute Observations
1. **Single-thread parity**: At 1 thread, `ternary_L2` ($0.042\text{–}0.049\text{ ns/w}$) is neck-and-neck with native `Q5_0` ($0.048\text{ ns/w}$) and `Q6_K` ($0.046\text{ ns/w}$), and faster than `Q8_0` ($0.072\text{ ns/w}$). `ternary_L3` is $\approx 1.5\times$ slower ($0.065\text{–}0.074\text{ ns/w}$).
2. **Multi-thread scaling barrier**: At 8 threads, native Q8Act chart kernels scale aggressively (e.g. `Q5_0` drops to $25.14\ \mu\text{s}$, achieving $119.16\text{ GB/s}$ effective decode bandwidth, and `Q6_K` drops to $31.72\ \mu\text{s}$, achieving $112.70\text{ GB/s}$). In contrast, `ternary_L2` scales only to $97.77\ \mu\text{s}$ ($23.68\text{ GB/s}$) and `ternary_L3` to $144.51\ \mu\text{s}$ ($24.03\text{ GB/s}$).
3. **Execution Bottleneck**: The T-MAC / PQ2_0 AVX2 kernel requires 4 `vpshufb` instructions, 4 unpacks, 8 additions, shifts, and 4 `fold8` lane accumulators per 4-column group per 32 rows. In contrast, the chart AVX2 kernels (`q8act_q5_0_row_fast` and `q8act_q8_0_row_fast`) use fused multiply-accumulate (`_mm256_maddubs_epi16` and `_mm256_madd_epi16`), processing 32 activations in parallel per single vector instruction. On x86_64, `vpshufb` port-5 pressure limits throughput to $\approx 24\text{ GB/s}$, making multi-threaded execution compute-bound rather than memory-bandwidth bound.

---

## 4. Byte Footprint & Layout Comparison

Model parameters from `docs/evidence/pipeline-inventory.md`:
- 24 Transformer Layers: $357,826,560\text{ weights}$
- Tied LM Head / Token Embedding: $151,936 \times 896 = 136,134,656\text{ weights}$
- Total Model Weights: $493,961,216\text{ weights}$

GGUF Actual Footprint:
- Layer Weights: $260,929,536\text{ B}$
- LM Head Payload: $144,643,072\text{ B}$
- Embedding Row: $952\text{ B}$
- Total GGUF Bytes/Token: $405,573,560\text{ B}$ ($386.8\text{ MiB}$)

### 4.1 Candidate Trit-Plane Layouts

1. **(a) `PQ2_0` per plane as-is**:
   - Each plane is stored as independent `PQ2_0` blocks: 2 bytes `f16` scale + 32 bytes codes per 128 weights = 34 bytes / 128 weights / plane ($2.125\text{ b/w}$ per plane).
   - $L=2$: $68\text{ bytes / 128 w} = 4.25\text{ b/w}$.
   - $L=3$: $102\text{ bytes / 128 w} = 6.375\text{ b/w}$.
2. **(b) Packed layout (shared scale across planes)**:
   - One `f16` scale (2 bytes) shared across all $L$ planes per 128 weights, plus $L \times 32\text{ bytes}$ of 2-bit codes.
   - $L=2$: $2 + 64 = 66\text{ bytes / 128 w} = 4.125\text{ b/w}$.
   - $L=3$: $2 + 96 = 98\text{ bytes / 128 w} = 6.125\text{ b/w}$.
3. **(c) Entropy-coded ideal (`TERNARY_FINDINGS.md`)**:
   - GPTQ-ternary $L=3$ (27 levels): $4.51\text{ b/w}$ ($0.5638\text{ bytes/weight}$).
   - GPTQ-ternary $L=2$ (9 levels): $2.86\text{ b/w}$ ($0.3575\text{ bytes/weight}$).

### 4.2 Bytes per Token Table

| Layout / Variant | Effective b/w | Layer Bytes (excl. LM head) | Total Bytes/Token (incl. LM head) | vs GGUF (405.6 MB) |
|---|---|---|---|---|
| **Qwen GGUF baseline** | 6.57 (avg) | 260,929,536 | 405,573,560 | 1.000× |
| **(a) PQ2_0 as-is $L=2$** | 4.250 | 190,095,360 | 262,417,372 | 0.647× (-35.3%) |
| **(a) PQ2_0 as-is $L=3$** | 6.375 | 285,143,040 | 393,626,058 | 0.971× (-2.9%) |
| **(b) Packed shared $L=2$**| 4.125 | 184,504,320 | 254,699,214 | 0.628× (-37.2%) |
| **(b) Packed shared $L=3$**| 6.125 | 273,960,960 | 378,189,742 | 0.932× (-6.8%) |
| **(c) Entropy ideal $L=2$** | 2.860 | 127,922,995 | 176,591,135 | 0.435× (-56.5%) |
| **(c) Entropy ideal $L=3$** | 4.510 | 201,724,723 | 278,470,635 | 0.687× (-31.3%) |

*Note*: Embedding row contribution ($476\text{ B}$ for $L=2$, $714\text{ B}$ for $L=3$) is included in total bytes/token.

---

## 5. Performance Projections (Grade: Model)

On host `Intel Core i9-14900KF`, measured memory bandwidth is $82.0\text{ GB/s}$.
The Roofline decode latency model per token is:

$$T_{\text{token}} = \max\left( \frac{\text{Bytes}_{\text{model}}}{\text{Bandwidth}}, \ T_{\text{compute}} \right) + T_{\text{overhead}}.$$

* **say:** T token equals the larger of Bytes model over Bandwidth, and T compute, plus T overhead.
* **means:** decode time per token is the slower of the memory and compute terms, plus fixed overhead.
* **vars:** $T_{\text{token}}$ — decode time per token, ms; $\text{Bytes}_{\text{model}}$ — bytes read per token, MB; $\text{Bandwidth}$ — measured bandwidth, GB/s; $T_{\text{compute}}$ — compute time, ms; $T_{\text{overhead}}$ — fixed per-token cost, ms.
* **apeiron:** `none — anchor`. A roofline model, not a theorem. The host bandwidth it uses is a measurement.
* **physics:** the roofline bound: a kernel cannot exceed the slower of its traffic and its arithmetic.

Today's measured baseline on this host:
- Bandwidth floor: $405.6\text{ MB} / 82\text{ GB/s} = 4.95\text{ ms} \implies 202.2\text{ tok/s}$.
- Measured decode: $7.57\text{–}7.80\text{ ms} \implies 128.2\text{–}133.0\text{ tok/s}$.
- Compute + overhead gap: $\approx 2.8\text{ ms}$.

### 5.1 Bandwidth Floor vs Compute-Bound Latency (8 Threads)

Compute latency across all 24 layers + LM head ($493,961,216\text{ weights}$) at measured 8-thread throughput:
- $L=2$: $493.96\text{M weights} \times 0.0228\text{ ns/w} = 11.26\text{ ms} \implies 88.8\text{ tok/s}$.
- $L=3$: $493.96\text{M weights} \times 0.0337\text{ ns/w} = 16.65\text{ ms} \implies 59.7\text{ tok/s}$.

| Layout | BW Floor (LM head incl.) | Tok/s Floor @ 82 GB/s | 8-Thread Compute Time | Compute-Bound Tok/s | Projected Actual Tok/s | vs Measured Qwen (130 tok/s) |
|---|---|---|---|---|---|---|
| **Qwen GGUF (Current)**| 4.95 ms | 202.2 | ~4.50 ms | ~222.0 | **130.0** | **1.00×** |
| **(a) PQ2_0 $L=2$** | 3.20 ms | 312.5 | 11.26 ms | 88.8 | **~75.0** | **0.58×** (Slower) |
| **(a) PQ2_0 $L=3$** | 4.80 ms | 208.3 | 16.65 ms | 59.7 | **~53.0** | **0.41×** (Slower) |
| **(b) Packed $L=2$** | 3.11 ms | 321.9 | 11.26 ms | 88.8 | **~75.0** | **0.58×** (Slower) |
| **(b) Packed $L=3$** | 4.61 ms | 216.8 | 16.65 ms | 59.7 | **~53.0** | **0.41×** (Slower) |
| **Hybrid: $L=3$ Layers + Q8_0 LM Head** | 4.98 ms | 200.8 | 14.43 ms | 69.3 | **~60.0** | **0.46×** (Slower) |

*If LM head is excluded from ternary conversion* (e.g. kept as Q8_0):
- 24 Layers $L=2$ compute = $8.16\text{ ms}$; plus Q8_0 LM head ($2.35\text{ ms}$) = $10.51\text{ ms} \implies 95.1\text{ tok/s}$.
- 24 Layers $L=3$ compute = $12.06\text{ ms}$; plus Q8_0 LM head ($2.35\text{ ms}$) = $14.41\text{ ms} \implies 69.4\text{ tok/s}$.

---

## 6. Synthesis & Quality Context

From `docs/evidence/decision-cell-census.md` and `/home/socol/Workspace/spinor-ttl/TERNARY_FINDINGS.md`:
1. **$L=3$ (27 levels)**:
   - Flip rate: $12.50\%$ (beats 4-bit seal at $15.55\%$).
   - Output error: $E = 1.6 \times 10^{-3}$ ($0.34\times$ of 4-bit seal).
   - Matches and exceeds int4 quality on real trained weights.
   - **Cost**: $6.125\text{ b/w}$ (packed) or $6.375\text{ b/w}$ (as-is), which is *larger* than int4 ($4.0\text{–}4.5\text{ b/w}$). Only drops to $4.51\text{ b/w}$ with entropy coding.
2. **$L=2$ (9 levels)**:
   - Flip rate: $39.44\%$ (severe degradation; mean greedy prefix collapses from $8.66$ to $2.74$).
   - Output error: $3.3\times\text{–}3.6\times$ worse than 4-bit seal.
   - Fails the quality acceptance threshold.

---

## 7. Verdict: Is a Trit-Plane Qwen Worth Building?

### Verdict: **NO.**

### Key Reasons:

1. **Compute Inversion on CPU**:
   While ternary planes reduce weight memory transfer ($378.2\text{ MB}$ for Packed $L=3$ vs $405.6\text{ MB}$ for GGUF), the CPU kernel is **compute-bound, not memory-bound**. The AVX2 lookup path (`vpshufb` nibble shuffle + horizontal byte unpacking) achieves only $\approx 24\text{ GB/s}$ of effective execution bandwidth across 8 threads, compared to $>110\text{ GB/s}$ for the hardware DP4A-style vector multiply-accumulate (`_mm256_maddubs_epi16`) used in `Q5_0` and `Q6_K`.
   Running $L=3$ passes per layer pushes decode compute time to $>16\text{ ms/token}$, cutting throughput from **$130\text{ tok/s}$ down to $\approx 53\text{ tok/s}$** ($2.4\times$ slower).

2. **No Rate-Distortion Gain**:
   As proven in `TERNARY_FINDINGS.md` §9–10 (`pretrained_ternary_is_lossy` and `storage_pigeonhole`), post-hoc ternary conversion cannot escape the weight entropy $H \approx 3.9\text{ b/w}$. Matching 4-bit output fidelity requires $L=3$ ($27\text{ levels}$), which requires $6.125\text{ b/w}$ packed (or $4.51\text{ b/w}$ entropy-coded). At equal or higher bit cost than GGUF int4/int5, there is zero compression benefit.

3. **$L=2$ Destroys Generation Quality**:
   $L=2$ reduces footprint to $254.7\text{ MB}$ ($4.125\text{ b/w}$), but the decision-cell flip rate reaches $39.44\%$, collapsing model coherence.

4. **Which Layout if Forced?**:
   If an architecture with hardware ternary dot-product support (e.g. specialized FPGA/ASIC ternary matrix units where 1-trit compute is essentially free) were targeted, layout **(b) Packed shared scale** ($6.125\text{ b/w}$ for $L=3$, $4.125\text{ b/w}$ for $L=2$) is strictly superior to `PQ2_0` as-is, saving $\approx 15.4\text{ MB}$ of redundant scale metadata. But on x86_64 CPU, GGUF mixed-precision (`Q4_K_M` / `Q5_0` / `Q6_K`) remains decisively superior in both throughput and quality.

---

## 8. Verification Gate & Controls

### Gate
Re-derivation and consistency check of all reported benchmark results and analytical model constants against `/home/socol/oc-tmp/spike-tp/spike_bench_raw.json`:
```sh
python3 /home/socol/oc-tmp/spike-tp/check_spike_tp.py docs/evidence/trit-plane-spike.md /home/socol/oc-tmp/spike-tp/spike_bench_raw.json
```
Observed output:
```text
OK: All 18 benchmark rows and analytical constants verified against /home/socol/oc-tmp/spike-tp/spike_bench_raw.json
Exit: 0
```

### Negative Control
Mutating any benchmark number in a scratch copy of `docs/evidence/trit-plane-spike.md` causes the gate checker to fail with exit status 1:
```sh
cp docs/evidence/trit-plane-spike.md /home/socol/oc-tmp/spike-tp/mutant.md
sed -i 's/214.36/999.99/' /home/socol/oc-tmp/spike-tp/mutant.md
python3 /home/socol/oc-tmp/spike-tp/check_spike_tp.py /home/socol/oc-tmp/spike-tp/mutant.md /home/socol/oc-tmp/spike-tp/spike_bench_raw.json
```
Observed output:
```text
FAIL: 1 errors found in /home/socol/oc-tmp/spike-tp/mutant.md:
  - Missing or mismatched row for gate/up ternary_L2 thr=1: expected med=214.36 us, ns/w=0.0492, GB/s=10.80
Exit: 1
```
