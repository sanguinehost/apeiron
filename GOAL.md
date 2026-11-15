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
| **Q2** | quality | wikitext-2 perplexity at ctx ≥ 2048 is **≤ 1.15×** the same checkpoint's Q4_K_M; mean 0-shot accuracy over boolq, piqa, hellaswag, winogrande, arc_easy and arc_challenge is **≥ 0.90×** Q4_K_M | PPL and 0-shot receipt against a named Q4_K_M file with its sha256 |
| **M1** | memory | on the 6.6 GiB laptop, with the server loaded and a 200,000-token context allocated, `MemAvailable` stays **≥ 2 GiB** and the server's anonymous resident set is **≤ 2 GiB** | `/proc/meminfo`, `smem`/`status` samples and a five-session soak in a receipt |
| **T1** | decode throughput | **≥ 30 tok/s** at 2k context and **≥ 24 tok/s** at 200k context (at least 0.8× the 2k rate), mean of at least three runs with the observed spread | timed decode on a named CPU, command and interval in a receipt |
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
| Q1 | **partial** | a 12-row sample packs to 4,978,784 B at 1.1510 b/w on 34,603,008 weights; the shipped factor stores 1.166 b/w; receipt `bench/cpu/1794740406-nq-118-distortion.md`. Qwen2.5-7B's calibration covers all 28 layers (112 diagonals, 17,920 tokens, full-Gram control ≤2.9e-6) and the recipe computes 1.174036 b/w, 957,615,232 B on 6,525,288,448 weights at `APEIRON_NQ_BITS_PPM=1180000`; one layer-0 `attn_output` fits at 1.1339 b/w, `ffn_down` still refuses, and the campaign is out of credit; receipt `bench/cpu/1794781140-nq7b-118-full-pack.md` | the pack itself; the 18,944-wide `down` fit; Q2 |
| Q2 | **open** | the 8B v3 pair measures 16.295301 PPL at ctx 128, 1.011× Q4_K_M, at 3.13 b/w; receipt `bench/cpu/1794653685-nq8b-residual-ppl.md` | the 1.15–1.18 b/w pack; ctx ≥ 2048; the 0-shot half |
| M1 | **open** | the 8B v3 pair peaks at 2399.2 MiB RSS at ctx 128 and 2567.1 MiB at ctx 512; receipt `bench/cpu/1794690665-nq8b-laptop-serve.md` | 200k context; five sessions; a 1.15 b/w pack |
| T1 | **open** | 0.09 tok/s on the 8B v3 pair on the laptop; the 1.15 b/w projection model is 66× at T=400 and 137× at T=200 against the pre-fix projection, with a same-host A/B of 53–63×; `bench/notes/fit-cost.md` | 30 tok/s at 2k and 24 tok/s at 200k on the whole model |
| T2 | **open** | no 1k turn at 200k has run | the latency bar |
| C1 | **open** | the classic f16 KV is still the serving path; ColdKV E0/E1c supports a fixed A=1024 budget across 31× context, but it is a torch probe and its full-KV cold store does not fit 200k. The 7B state is 5,734,400,000 values at 200k, so the 1 GiB cap is 1.4976 bits/value including hot tier, cold store and index; the file-backed plane relocates bytes to disk and does not meet it; `bench/notes/context-budget.md` | a carrier at ≤1.5 bits/value, a 1 GiB allocation at 200k, and the growth argument |
| C2 | **open** | no replacement has been scored at 200k | the Q2 bar at 200k and the agreement test |
| I1 | **partial** | one 25-token turn through Intent's `local` provider reached `finish_reason: stop` and Intent `outcome: complete` on the 8B v3 pack; receipt `bench/cpu/1794696786-nq8b-intent-loop.md` | the full 2,672-token assembled prompt and a real coding turn |
| I2 | **open** | no 20-task suite and no local/DeepSeek split | the replacement bar |
| V1 | **open** | the Vulkan path uploads GGUF payloads and does not read a NanoQuant pack | a 7B pack on the Vulkan path and a GPU/CPU agreement run |
| V2 | **open** | no in-tree SPIR-V kernel source for the serving path | the build and hash |
| V3 | **partial** | the CPU `Cell<i64>` word bodies and the ring operations are refined in Lean; the bridge is in `proof/corpus/` | any SPIR-V kernel; the GPU path |
| V4 | **open** | no kernel-source mutation test | the red/green control |
| O1 | **partial** | the repository is Apache-2.0 and builds from source | a clean-clone serving build and a reproducible pack recipe |
| G1 | **partial** | 656 tests pass; 18 GPU/Vulkan tests fail on the laptop environment; clippy and fmt pass except one uncommitted example lint | the target commit's full gate |
| G2 | **partial** | the CPU extraction and bridge build where the pinned toolchain is available | `proof/regen.sh --check` and `lake build` for the uncommitted `src/engine/kern/matmat.rs` |
| G3 | **partial** | the current receipt names its rows and its non-coverage | every receipt naming a criterion id |
| S1 | **open** | the 8B v3 pair served under a 1 GiB cgroup at 0.09 tok/s | a preflight refusal and a five-session 200k soak |

### What the 1.15 b/w sample does and does not show

The sample replaces five projection kinds and measures 1.1510 b/w on the replaced linears. Against
Q4_K_M it stores 4.14× fewer bytes over those five kinds and 4.15× on the four kinds measured row by
row. The whole-payload ratio is 1.04× and the per-token ratio is 1.03× because the tied F16 LM head
is 311.16 MB. The sample does not establish a 7B model's quality, its 200k context, its throughput,
or its memory. Those are Q2, C1, T1 and M1.

The tied F16 head is a named blocker for the served per-token DRAM win: `Engine::lm_head` in
`src/engine/step.rs` reads the F16 head on the token path. A full-model rate claim includes it or
states it separately.

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
