# The goal

Run a 7B-class model on the 6.6 GiB laptop, locally, at about 30 tok/s on the CPU, with a
200,000-token context, while four or five coding-agent sessions keep running. Serve it through this
library into `../intent`, replace DeepSeek usage to a measured degree, and ship the quantiser, the
context method and the Vulkan/SPIR-V kernels as open source, with the kernels formally verified
through Aeneas and Lean.

This file is the canonical statement of work and acceptance criteria. `AGENTS.md` carries the same
goal for contributors and agents. A receipt, a card, a benchmark row or a single speedup is an
artifact of this goal; it does not replace it and it does not close it.

## Acceptance criteria

A criterion is closed only by the named evidence. A narrower result does not close it.

| id | criterion | threshold | evidence |
|---|---|---|---|
| **Q1** | quantisation rate | a full 7B-class pack stores its replaced linear weights at **1.15–1.18 b/w** (bits per weight), not a 12-row sample and not a single projection kind | pack file, weight count, byte count and the command that produced them in a receipt |
| **Q2** | quality | wikitext-2 perplexity at ctx ≥ 2048 is **≤ 1.5×** the same checkpoint's Q4_K_M (goal) and **≤ 1.15×** (stretch, the north star: 1.15× PPL at 1.15 b/w); mean 0-shot accuracy over boolq, piqa, hellaswag, winogrande, arc_easy and arc_challenge is **≥ 0.90×** Q4_K_M | PPL and 0-shot receipt against a named Q4_K_M file with its sha256 |
| **Q3** | NanoQuant parity | on the Q2 protocol at **≤ 1.18 b/w**, the pack beats `SamsungLabs/NanoQuant`'s published Qwen3-8B-Base row (**1.00 b/w**, **1.78×** BF16 PPL, **0.687×** BF16 six-task) on both PPL and 0-shot, with the rate and the embedding/head precision named; the paired control is their 1.00 b/w row re-scored on this protocol when their code or weights permit, otherwise the published row is labelled cross-harness | one v1 result record (`bench/PROTOCOL.md`) with a near-lossless arm and the OG row |
| **M1** | memory | on the 6.6 GiB laptop, with the server loaded and a 200,000-token context allocated, `MemAvailable` stays **≥ 2 GiB** and the server's anonymous resident set is **≤ 2 GiB** | `/proc/meminfo`, `smem`/`status` samples and a five-session soak in a receipt |
| **T1** | decode throughput | **≥ 30 tok/s** at 2k context and **≥ 24 tok/s** at 200k context (at least 0.8× the 2k rate), mean of at least three runs with the observed spread. The 30 is **98.8 % of the laptop's 42.656 GB/s spec peak** on the 1.174 b/w 7B path (`bench/cpu/1794793277-kernel-asm.md` §6); the measured-bandwidth ceiling is **19.9 tok/s**, and the 24 at 200k needs 127.9 % of peak (`bench/cpu/1794799742-decode-ceiling.md` §5) | timed decode on a named CPU, command and interval in a receipt |
| **T2** | turn latency | a 1,000-token incremental turn at 200k context prefills in **≤ 60 s**; a full 200k prefill is measured and reported separately | prefill and decode timings at 2k, 32k, 128k and 200k |
| **C1** | context state | the serving path replaces the classic f16 KV cache; context state at 200k is **≤ 1 GiB** total (hot tier, cold store and index), with an O(1) or O(log n) hot path; an O(n) cold store is allowed only inside that 1 GiB cap | allocation trace, `/proc` samples and the method's growth argument in a receipt |
| **C2** | context quality | the replacement scores within the Q2 bar at 200k, and its output agrees with full causal attention on a named agreement test | PPL/0-shot at 200k and the agreement test |
| **I1** | Intent integration | `../intent` drives a real coding turn through `serve/` with its full assembled prompt; no `--drop-system` shortcut in the acceptance run | Intent transcript, served request and `finish_reason` in a receipt |
| **I2** | DeepSeek replacement | over a 20-task coding suite, **≥ 50%** of Intent turns are served locally, task completion is **≥ 0.90×** the DeepSeek baseline, and p50/p95 latency and cost are reported | suite definition, per-task outcomes, local/DeepSeek split and cost table |
| **V1** | Vulkan path | the 7B pack runs end-to-end through the Vulkan compute path on a named GPU; logits agree with the CPU path under a stated test | GPU/CPU agreement run, driver and device named |
| **V2** | SPIR-V provenance | the Vulkan kernels are built from in-tree source to SPIR-V in the build; no opaque shader blob is the serving path | build command, source path and SPIR-V hash |
| **V3** | formal verification | the kernel source that produces the SPIR-V is extracted by Aeneas/Charon and a Lean refinement theorem links the extracted body to the canonical arithmetic; the axiom audit stays within `propext`, `Classical.choice` and `Quot.sound` | `proof/regen.sh --check`, `lake build`, the theorem name and `AxiomAudit.lean` |
| **V4** | proof control | a mutation of the real kernel source turns the proof red in a throwaway copy; restoring it turns the proof green | mutation log in the receipt |
| **O1** | open source | a clean clone builds the serving path and reproduces the pack recipe under Apache-2.0; no proprietary component on the token path | clean-clone build log, licence and dependency audit |
| **G1** | contributor gate | `cargo test --workspace --all-features --locked`, clippy, fmt, `tools/check-claims.py`, `tools/book-check.py` and the book build pass on the target commit; environment-gated GPU tests are listed, not hidden | gate output in the receipt |
| **G2** | maintainer gate | `proof/regen.sh --check` and `lake build` pass for every extracted file the change touches | gate output in the receipt |
| **G3** | receipt discipline | every receipt names the criterion it advances and the grade and scope of its result; a narrower claim says which part of the criterion it leaves open | the receipt header and `tools/check-claims.py` |
| **S1** | load safety | a preflight check refuses a model load that would exceed the derived memory cap; a 30-minute soak with five sessions produces no OOM kill | preflight refusal test and soak log |

## Statement of work

The order is the dependency order. A step may run in parallel only where it does not consume the
same measured resource.

1. **Freeze the checkpoint and the harness.** Name the 7B-class checkpoint (primary:
   Qwen2.5-7B-Instruct, with Qwen3-8B v3 as the measured comparison), its tokenizer, its Q4_K_M
   reference file and sha256, its KV geometry and the evaluation harness. Rebuild the calibration
   Grams and fix the pack recipe before measuring anything else.
2. **Build and measure the full pack.** Produce the full 1.15–1.18 b/w pack on a rented box, not the
   laptop. Measure Q1 and Q2, then peak RSS and tok/s at 2k, 32k, 128k and 200k. Publish the pack
   hash, the weight count and the exact commands.
3. **Replace the context path.** Evaluate the candidates against C1 and C2: a ColdKV-class cold tier,
   a quantised KV cache, a linear/state-space attention, and a hybrid. ColdKV's full-KV cold store is
   not sufficient at 200k: the 8B f16 KV is about 28.8 GB and Qwen2.5-7B f16 KV is about 11.5 GB, so
   a cold tier must be quantised or bounded to fit the 1 GiB cap. Port the chosen method into the
   engine, remove the classic f16 KV from the serving path, and measure the 200k memory and speed
   curve.
4. **Serve and integrate.** Expose the pack through `serve/` (the only listener; `src/` and
   `examples/` stay socket-free), drive it from `../intent`'s `local` provider with the full assembled
   prompt, and run the 20-task suite. Measure the local/DeepSeek split, completion rate, latency and
   the five-session memory headroom.
5. **Vulkan, SPIR-V and Lean.** Write the kernel source once in a form that both compiles to SPIR-V
   and extracts through Aeneas/Charon; prove the refinement theorem; run the GPU/CPU agreement,
   mutation and axiom-audit gates. The compiler, the driver and the device remain outside the proof.
6. **Close the gates and release.** Run the contributor and maintainer gates, write the receipts,
   verify a clean clone, and publish the source and the recipe under Apache-2.0.

## Current status

The goal is open. The table records what is established, by what artifact, and what the artifact
does not cover.

| criterion | status | established | not covered |
|---|---|---|---|
| Q1 | **partial** | a 12-row sample packs to 4,978,784 B at 1.1510 b/w on 34,603,008 weights; the shipped factor stores 1.166 b/w; receipt `bench/cpu/1794740406-nq-118-distortion.md`. Qwen2.5-7B's calibration covers all 28 layers (112 diagonals, 17,920 tokens, full-Gram control ≤2.9e-6) and the recipe computes 1.174036 b/w, 957,615,232 B on 6,525,288,448 weights at `APEIRON_NQ_BITS_PPM=1180000`; the 7B `down` fits at 1.099715 b/w on an AVX-VNNI box (T=2 213.18 s), the full 196-linear T=200 is **504 h** (the earlier 18 h was per layer; `tools/fit-budget.py` refuses it), a T=2 campaign stopped at **12 of 196 sections** for cost and host instability (no full pack; `bench/cpu/1794835100-nq7b-t2-abort.md`), and the 3.0 b/w layer-0 arm needs a residual (dense cap 1.19 b/w); receipts `bench/cpu/1794781140-nq7b-118-full-pack.md`, `1794828501-nq7b-d6-pack.md`. A full 8B pack from the upstream GPU pipeline stores 6,945,767,424 weights at **1.124186 b/w** in 976,049,680 B, 2,305,675,376 B with its skeleton (`bench/cpu/1794879503-nq8b-1124-ctx2048.md`); 1.124 is below the 1.15–1.18 band and the pack is not from `nq_fit`. The converter stores the fused qkv V, `s_pre`, `s_mid` and the salient indices three times: the duplicate is **91,450,016 B** (9.37%), and a shared-section format would be **1.018865 b/w** at identical factors (`bench/cpu/1794885786-nq8b-uv-signs.md`) | the Qwen2.5-7B pack itself; Q2 |
| Q3 | **open** | the published row is **1.78× BF16 PPL** and **0.687× BF16 0-shot** at 1.00 b/w; this pack is **1.525440× Q8_0** (near-lossless) and **0.7534×** Q4_K_M 0-shot at 1.1788 b/w, on the v1 five-window screen (record `bench/results/271fd5a7862b.json`, 10,235 positions per arm, full 2,047-token spans). Against OG's published 1.78 the PPL half is **0.857×**, i.e. 14.3 % lower excess at 18 % more bits; at 1.18 b/w their three points (1.00→1.78, 0.80→2.12, 0.55→2.86) interpolate to **1.35–1.58**, so the screen does not separate the two artifacts at equal rate. The 0-shot half has no v1 arm, so Q3's second half is unmeasured; the earlier "~30 % denominator gap" compared a ctx-128 Q8_0 with a 2048 BF16 and is withdrawn (`bench/cpu/1794966727-nq8b-v1-screen.md`). `bench/PROTOCOL.md` v1 fixes the tuple: the 17-window confirm, the bartowski arm and the six-task arm remain. `SamsungLabs/NanoQuant` ships code, not weights, so the paired control is their `evaluate_ppl` on this checkpoint or a labelled cross-harness comparison | a v1 record with a near-lossless arm; OG's row re-scored, or the cross-harness label kept |
| Q2 | **open** | **v1 five-window screen: PPL 1.462553× the official Q4_K_M (2.50 % under the 1.5 bar) and 1.525440× Q8_0; the bartowski ratio is 1.5105, 0.70 % over.** **The 0-shot half refutes this pack: 0.7534× against the 0.90 bar.** Six tasks × 100 items, both arms, paired (`tools/quality-compare.py`, exit 1): NQ **0.5600** against Q4_K_M **0.7433**, paired difference **−0.1833 ± 0.0812** (95 % t, df 5), allowance −0.0743, so the interval [−0.2645, −0.1021] is entirely below it. Only piqa clears (0.9221); boolq 0.7159, hellaswag 0.7703, arc_easy 0.7342, arc_challenge 0.7308, winogrande 0.6447 (0.49, chance on two choices). **PPL does not predict this**: the same pack is 1.42–1.46× on wikitext-2 (established) and 0.7534× on 0-shot, so the 1.178785 b/w allocation loses a quarter of task accuracy while PPL reads as a mild loss — the next pack is judged on the 0-shot screen, not on PPL alone. Receipt `bench/cpu/1794937300-nq8b-q2-rented-host.md`, logs in `bench/cpu/1794937300-nq8b-q2-rented-host-logs/` (twelve per-item JSONLs, `compare.txt`), with the shared-prompt scorer validated bit-for-bit against the joint path on piqa and on one item of each task. The gap decomposes with `tools/quality-gap.py`: the NQ candidate margin is **0.36×** Q4's (slope 0.16–0.78) with a residual the size of the signal, boolq's yes/no contrast shrinks **6.51 → 1.92 nats** and P(yes) **0.58 → 0.87** against a 0.56 label prior (accuracy 0.63, 7 points above always-yes), winogrande sits at chance (0.49), while piqa's long tercile survives at **0.88 against 0.85**; 134 Q4→NQ losses against 24 gains, and 19 % of 1–3 sd items still flip. Receipt `bench/cpu/1794948388-nq8b-q2-zeroshot-gap.md`. The fix path is the format's own duplicate: 91,450,016 B of qkv `V`/`s_pre`/`s_mid`/`salient_idx` stored three times is 8.94 % of this pack, so a shared-section rewrite is **1.073466 b/w** at identical factors and frees 0.1053 b/w for a 3.39 MB per-channel bias and a rank-16 f16 residual (1.1740 b/w), or a KL refit at the same rate; the 0-shot screen, gated on boolq ≥ 0.792 and winogrande ≥ 0.684, is the judge. The **PPL half** stands against the named official bar: text-weighted over skips 0–139,776 (58 % of the file), **1.417631** [1.355945, 1.482123], gap 0.348987 ± 0.044489 nats (df 16); five windows alone give 1.499754 [1.391674, 1.616228] and are undecided. **The bar is load-bearing**: on the same five second-half windows and binary, bartowski's standard Q4_K_M (`54fffa05…`) pools to 9.604552 against the official bar's 9.919400, factor 0.96826, so the ratio against a standard Q4_K_M is **1.4641** and its upper end is ≈**1.531** — the pass holds against the named file and its upper end crosses 1.5 against a standard one; every receipt quotes the bar with the number (`bench/cpu/1794952981-nq8b-context-ranking.md`). The eight tail windows were stopped as superseded by the next pack edit. The same-quantisation condition is exact (skeleton `token_embd` Q4_K and `output.weight` Q6_K byte-identical to the bar, 72 differing norms carrying its AWQ fold undone in 252 linears, `bench/cpu/1794921128-nq8b-tensor-audit.md`). At 1.178785 b/w the rate lever is exhausted: 1,044,615 B of headroom against ≥2,097,152 B for one plane on a 4096² linear. The 1.15× stretch needs 0.209 of the 0.349-nat gap removed; the 0-shot result says the lever is the **allocation**, not the rate: protect the knowledge-bearing linears, and score every candidate on the six-task screen | a 0-shot-clean pack at ≤1.18 b/w; a fit that moves the PPL gap by 0.209 nats without repeating the 0.75× 0-shot loss; the 7B checkpoint |
| M1 | **open** | the 8B v3 pair peaks at 2399.2 MiB RSS at ctx 128 and 2567.1 MiB at ctx 512; receipt `bench/cpu/1794690665-nq8b-laptop-serve.md`. The 1.124186 b/w pair is 2,305,675,376 B on disk and runs under a 2.0 GiB cgroup with both files mapped, 2,470.0 MiB RSS in the PPL pass (`bench/cpu/1794879503-nq8b-1124-ctx2048.md`) | 200k context; five sessions; a 1.15 b/w pack |
| T1 | **open** | **The 8B Q2 pair is measured: 285.26 ms/token (3.5 tok/s) at ctx 64 and 2.884 tok/s (346.8 ms/token) at 2k over three runs, ffn dominant at 56.4% (ceiling 2.29×), head 15.5%, the three NQ phases 79.9%; the shipped Q6_K head reads 17.32 G weight/s against the exact reader's 1.17. NQ runs at 33.4 G terms/s aggregate (8.36 G/core, 3.14 G instr/s/core); 20 tok/s needs 190.6 G terms/s (5.7×) and, at 2k with the f32 KV, 42.76 GB/s, which is 1.002× the 42.656 GB/s spec peak and 1.91× the measured 22.33; a perfect 6-instruction selector lands at 5.84 tok/s. Receipt `bench/cpu/1794919269-nq8b-split.md`.** Earlier: 0.09 tok/s on the 8B v3 pair on the laptop; the 1.124186 b/w pair decodes at **2.25 tok/s** under a 2.0 GiB cgroup, its F16 head reading 1.245 GB/token at 2.80 GB/s and binding the token (`bench/cpu/1794879503-nq8b-1124-ctx2048.md`); the 1.15 b/w projection model is 66× at T=400 and 137× at T=200 against the pre-fix projection, with a same-host A/B of 53–63×; `bench/notes/fit-cost.md`. The scalar `NanoQuant` sign dot is vectorized 4.95× on the pre-popcount loop (`bench/cpu/1794806002-nq-sign-dot.md`) and the `i64` lane loop is 1.39× faster again (`bench/cpu/1794888094-nq-sign-dot-dedup.md`); `Q4_K` folded into the shared loop at 0.89–1.00× of the deleted sibling and `gemm_q4_k` is deleted (`bench/cpu/1794807761-q4k-direct-write.md`). On the 1.174 b/w production path the Q6_K head makes the bytes ceiling 19.9 tok/s at the measured 27.97 GB/s; a rank-4096 NQ head is 80.0 MB and lowers the pack to 1.0376 GB/token, a 26.95 tok/s measured-bandwidth ceiling (`bench/cpu/1794810332-nq-head-in-pack.md`). The instruction wall is 8.25G sign terms/token with the head. The byte-plane `sign_dot_planes_avx2` issues 24 instructions per 32 terms (0.75/term) against the `i64` lanes' 44 (1.38/term, after the bit sum collapsed to the word's popcount, `bench/cpu/1794888094-nq-sign-dot-dedup.md`), measures 2.1 IPC on one core, and puts the 7B projections at 5.17 tok/s at 2 IPC on four cores (6.19G instructions/token), 3.64 tok/s as the pool actually ran on a 2-CPU shell; receipt `bench/cpu/1794812057-nq-sign-dot-simd.md`. 20 tok/s allows 0.194 instructions/term, six per 32-term word, while four exact byte planes cost twelve before a selector is built, so 20 tok/s is not reachable on that four-plane path at 1.174 b/w — the pack would need ≤2.13G terms/token, about 0.33 b/w. The 1-bit path is 2.4× the 4-bit path: Qwen2.5-7B Q4_K_M reads 4.3706 GB/token (6.40 tok/s bytes ceiling) at 2.121 instructions/weight (2.1–3.2 instruction ceiling). Neither is a 20 tok/s configuration on this host (`bench/cpu/1794811020-one-bit-vs-four-ceiling.md`). An agent-opened exploratory 64-term ternary-activation cell was measured standalone at 1.95–1.98 cycles per 64 terms, 120.4–123.0 G terms/s per core, 63.0 tok/s on four cores at 3.75 GHz and 43.7 at 2.6 GHz, with the pack's bytes binding at 28.03 tok/s (`bench/cpu/1794828200-ternary-cell-micro.md`). It is implemented in the engine (`Reader::Ternary`, safe twin `ternary_block`); the 0.6B PPL arm measured **77,741,390** against the same-binary q8act control **72.988013**, **1,065,125×** (+13.88 nats/token), consistent with the **45.7×** activation proxy (`bench/cpu/1794834226-ternary-ppl.md`). The cell is a rate ceiling, not a quality path; it was not requested and is not a product route. A full 7B T=2 pack now exists at **1.098871 b/w** (896,312,736 B, 6.854 h), but it is not scorable: Q4_K_M scores 9.298465 on 2,044 tokens, while the packed arm refuses at the first norm because the T=2 v/o factors are unconverged (`admm_loss` 9,094/10,546 against the 0.6B T=400 band of 0.22–0.35) and gain layer-0 o/down outputs by 10^4–10^6; the exact reader agrees (`bench/cpu/1794862546-nq7b-t2-ppl.md`). A T=2 PPL would be an upper bound on the converged ratio only if it ran; this one does not. The end-to-end decode number is still the open half | 30 tok/s at 2k and 24 tok/s at 200k on the whole model |
| T2 | **open** | no 1k turn at 200k has run | the latency bar |
| C1 | **open** | the classic f16 KV is still the serving path; ColdKV E0/E1c supports a fixed A=1024 budget across 31× context, but it is a torch probe and its full-KV cold store does not fit 200k. The 7B state is 5,734,400,000 values at 200k, so the 1 GiB cap is 1.4976 bits/value including hot tier, cold store and index; the file-backed plane relocates bytes to disk and does not meet it; `bench/notes/context-budget.md` | a carrier at ≤1.5 bits/value, a 1 GiB allocation at 200k, and the growth argument |
| C2 | **open** | no replacement has been scored at 200k | the Q2 bar at 200k and the agreement test |
| I1 | **partial** | one 25-token turn through Intent's `local` provider reached `finish_reason: stop` and Intent `outcome: complete` on the 8B v3 pack; receipt `bench/cpu/1794696786-nq8b-intent-loop.md`. The 0.6B v3 fixture now decodes at 21.42 tok/s (width 4, three runs 20.40–22.43; 9.01 at width 1) on a 482.7 MB skeleton with a `Q8_0` head plus the 53.8 MB pack, after the head left the `F16` reference reader; `bench/cpu/1794813322-q8-head-fixture.md` | the full 2,672-token assembled prompt, a real coding turn and a PPL ratio on the new head |
| I2 | **open** | no 20-task suite and no local/DeepSeek split | the replacement bar |
| V1 | **open** | the Vulkan path uploads GGUF payloads and does not read a NanoQuant pack. The device initializes on a rented RTX 3060 (proprietary driver 580.173.02, `vulkaninfo` lists the NVIDIA device, `Gpu::new()` selects it) once the container has `NVIDIA_DRIVER_CAPABILITIES=all` and the X/GL userspace. The 7B-shaped Q4 GEMV now submits, matches its CPU reference on every checked row and times at 336.30 GB/s (`bench/gpu/1794864414-vulkan-gemv-debug.md`); the earlier zero was the benchmark reading its output before submitting, and the remaining error was its activation pack, so no `src/gpu` change was needed. No 7B pack runs end to end and no logits-vs-CPU run exists. A CUDA probe of the same 1-bit carrier reads 539.5 GB/s, a 540.5 tok/s ceiling for the 0.998 GB token, but it is not the repository's kernel; receipt `bench/gpu/1794831271-gpu-gemv.md` | a 7B pack on the Vulkan path and a GPU/CPU agreement run |
| V2 | **open** | no in-tree SPIR-V kernel source for the serving path | the build and hash |
| V3 | **partial** | the CPU `Cell<i64>` word bodies and the ring operations are refined in Lean; the bridge is in `proof/corpus/` | any SPIR-V kernel; the GPU path |
| V4 | **open** | no kernel-source mutation test | the red/green control |
| O1 | **partial** | the repository is Apache-2.0 and builds from source | a clean-clone serving build and a reproducible pack recipe |
| G1 | **partial** | 1,032 tests pass, 32 fail in 20 GPU/`glslc` targets and 81 are ignored on this laptop; clippy with `-D warnings` and `fmt --check` exit 0; `check-claims.py`, `book-check.py`, `mdbook build` and `mdbook test` exit 0; receipt `bench/cpu/1794801248-g1-contributor-gate.md` | the unfiltered gate on a host with `glslc` and `vkCmdWriteTimestamp2` |
| G2 | **partial** | the CPU extraction and bridge build where the pinned toolchain is available | `proof/regen.sh --check` and `lake build` for the uncommitted `src/engine/kern/matmat.rs` |
| G3 | **partial** | the current receipt names its rows and its non-coverage | every receipt naming a criterion id |
| S1 | **open** | the 8B v3 pair served under a 1 GiB cgroup at 0.09 tok/s; the 1.124186 b/w pair served under a 2.0 GiB cgroup at 2.25 tok/s (`bench/cpu/1794879503-nq8b-1124-ctx2048.md`) | a preflight refusal and a five-session 200k soak |

### What the 1.15 b/w sample does and does not show

The sample replaces five projection kinds and measures 1.1510 b/w on the replaced linears. Against
Q4_K_M it stores 4.14× fewer bytes over those five kinds and 4.15× on the four kinds measured row by
row. The whole-payload ratio is 1.04× and the per-token ratio is 1.03× because the tied F16 LM head
is 311.16 MB. The sample does not establish a 7B model's quality, its 200k context, its throughput,
or its memory. Those are Q2, C1, T1 and M1.

The tied F16 head is a named blocker for the served per-token DRAM win: `Engine::lm_head` in
`src/engine/step.rs` reads the F16 head on the token path. A full-model rate claim includes it or
states it separately.

### The Q4 bar

The revolution is measured against the 4-bit path on the same host, not against the floor. Q4_K_M
reads 4.3706 GB/token, so at the measured 27.97 GB/s its whole-token ceiling is **6.40 tok/s**
(`bench/cpu/1794811020-one-bit-vs-four-ceiling.md`). **1.5× of Q4_K_M is 9.6 tok/s.** The 1.174 b/w
production path with the `Q6_K` head reads 1.4055 GB/token, 3.11× fewer bytes, and its own ceiling
is 19.9 tok/s: the byte advantage already exceeds the 1.5× milestone by 2.1×. The projections alone
are 5.17 tok/s modelled, **0.81× of Q4_K_M's ceiling and 1.86× short of 9.6** (3.64 tok/s as the pool
ran on a 2-CPU shell). The gap is terms/s, not bytes/token.

The working target is **20 tok/s on a 7B**, above the 9.6 milestone, and it forces the head onto the
same carrier. With the `Q6_K` head the whole-token read is 1.4055 GB, so 20 tok/s needs 28.1 GB/s,
above the measured 27.97: impossible at any kernel rate. A head fitted at the projections' rank
(2080) stores 0.040 GB at 0.59 b/w, so the token is 0.998 GB — 28.0 tok/s of bytes, 20 tok/s at 71%
of measured bandwidth. Projections plus head are 7.94 G sign terms/token. The kernel issues 11 uops
per 32 terms, three of them on the one shuffle port, so one core sustains at most 41.6 G terms/s at
3.9 GHz and four cores at most 166 G — **20.9 tok/s at 100% of the absolute issue bound**. The
measured loop reaches 66% of its per-core bound and the pool 2.44–2.87×, which delivers 8–10 tok/s.
The laptop has no GPU. On the q8 reader its own ceiling is **11.0 tok/s at 2.1 IPC** (12
instructions per 32 terms), so 20 tok/s on this host is a **rented-GPU claim (V1)** or a new carrier.
An agent-opened exploratory 64-term ternary-activation cell — weights at one bit, activations at
`{−1,0,+1}`, the dot `2·popcount(w&n) − 2·popcount(w&p) + popcount(p) − popcount(n)` — was measured
standalone: **1.95–1.98 cycles per 64-term group**, one percent from the two-`popcnt` port floor,
**120.4–123.0 G terms/s on one core** at 3.72–3.76 GHz. Four cores project **63.0 tok/s at 3.75 GHz
and 43.7 at a sustained 2.6 GHz all-core**; the loop is port-bound, not instruction-bound
(`bench/cpu/1794828200-ternary-cell-micro.md`). The pack's 0.998 GB/token then binds first at
**28.03 tok/s**, and 20 tok/s needs 71% of the measured 27.97 GB/s. The cell is built in the engine
(`Reader::Ternary`, safe twin `ternary_block`) but was not requested and is **refuted on quality**:
on the 0.6B v3 pack the ternary reader scores **77,741,390 PPL** against the same-binary q8act
control's 72.988013, **1,065,125×**, +13.88 nats/token, consistent with the 45.7× proxy
(`bench/cpu/1794834226-ternary-ppl.md`). It is a rate ceiling, not a quality path and not a product
route; the CPU route is the int4 selector at 14–18 tok/s, still below 20. `PLAN.md` tracks it.

## Non-goals

- The card in `../sanguine.sh/src/client/hero.ts` is a presentation artifact, not the goal.
- A single benchmark row, a 12-row sample, a 0.6B pack or one projection's speedup does not close
  the goal.
- Beating Q4_K_M on every metric is not required; Q2 is the bar.
- Training a new base model is out of scope. The work quantises an existing 7B-class checkpoint.
- rustc and LLVM code generation, the kernel ABI, Vulkan drivers and the GPU, and the `unsafe`
  container are outside the proof. The Lean theorem covers the kernel source, not the compiler or
  the device.
- A full 200k-token prefill at interactive speed is not claimed unless measured. The interactive
  bar is decode at 200k and a 1k incremental turn.
- A network listener in `src/` or `examples/` is out of scope by policy. `serve/` is the listener.

## How to resume

1. Read this file.
2. Read `AGENTS.md` for the gates, grades and boundaries.
3. Read the latest receipt named in the status table and the source it measures.
4. Pick the lowest-numbered open criterion that is not blocked. Work it to evidence.
5. If blocked, clear the blocker or move to the next criterion; report the blocker with the
   command and the one-line fix.
6. Write the receipt with the criterion id, the grade, the scope and the non-coverage. Run
   `python3 tools/check-claims.py` and `python3 tools/book-check.py`.

## Reminders

The owner's OpenCode configuration loads `~/.config/opencode/plugins/goal-reminder`, which sends a
`[goal-reminder]` prompt to the named session at a random 30–45 minute cadence. It pauses when a real
user message arrives, when `/tmp/opencode/goal-reminder.stop` exists, or when the plugin is unloaded.
The plugin is local to the owner's machine and is not part of this repository.

## Definitions

- **7B-class**: 6.5–8.5 billion parameters. Primary checkpoint: Qwen2.5-7B-Instruct. Comparison:
  Qwen3-8B v3, which is measured at 3.13 b/w and does not meet Q1.
- **b/w**: bits per weight, computed over the replaced linear weights; the whole-payload rate is
  reported separately and never substituted.
- **resident**: `RssAnon` plus the mapped file pages that are resident; file-backed pages that can be
  reclaimed are reported separately.
- **headroom**: `MemAvailable` in `/proc/meminfo` on the laptop with the server and the agent
  sessions running.
- **200k**: a 200,000-token context allocated and decoded through, not a configured maximum.
- **trivial**: the criterion's numbers (M1, T1, T2), not an adjective.
