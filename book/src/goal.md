# The goal

apeiron's deployment target is a 7B-class model on the 6.6 GiB laptop, running locally:

- about **30 tok/s** on the CPU at 2k context, and at least **24 tok/s** at 200k;
- a **200,000-token** context whose state is at most **1 GiB**, with the classic f16 KV cache
  replaced;
- at least **2 GiB** of `MemAvailable` left for four or five coding-agent sessions;
- served through `serve/` into `../intent`, replacing DeepSeek usage to a reported share;
- the quantiser, the context method and the Vulkan/SPIR-V kernels open source, with the kernels
  formally verified through Aeneas and Lean.

`GOAL.md` at the repository root is the canonical statement of work, the acceptance criteria, the
current status and the non-goals. This chapter is the reader's summary.

## Acceptance criteria

A criterion is closed only by the named evidence. A narrower result does not close it.

| id | criterion | threshold |
|---|---|---|
| **Q1** | quantisation rate | a full 7B-class pack stores its replaced linear weights at 1.15–1.18 b/w, not a 12-row sample |
| **Q2** | quality | wikitext-2 perplexity at ctx ≥ 2048 ≤ 1.5× Q4_K_M (goal) and ≤ 1.15× (stretch); mean 0-shot accuracy ≥ 0.90× Q4_K_M over boolq, piqa, hellaswag, winogrande, arc_easy and arc_challenge |
| **Q3** | NanoQuant parity | on the Q2 protocol at ≤ 1.18 b/w, beats NanoQuant's published Qwen3-8B-Base row (1.00 b/w, 1.78× BF16 PPL, 0.687× BF16 six-task) on both axes; the paired control is their row re-scored on this protocol, otherwise the comparison is labelled cross-harness |
| **M1** | memory | `MemAvailable` ≥ 2 GiB with a 200k context and five sessions; server anonymous RSS ≤ 2 GiB |
| **T1** | decode | ≥ 30 tok/s at 2k and ≥ 24 tok/s at 200k, mean of three runs with the spread |
| **T2** | turn latency | a 1,000-token incremental turn at 200k prefills in ≤ 60 s; a full 200k prefill is reported separately |
| **C1** | context state | classic f16 KV removed; context state at 200k ≤ 1 GiB total, with an O(1) or O(log n) hot path; an O(n) cold store only inside the cap |
| **C2** | context quality | the replacement meets the Q2 bar at 200k and passes a named agreement test against full causal attention |
| **I1** | Intent | `../intent` drives a real coding turn with its full assembled prompt, no `--drop-system` |
| **I2** | DeepSeek replacement | over a 20-task suite, ≥ 50% of turns local, completion ≥ 0.90× the DeepSeek baseline, p50/p95 latency and cost reported |
| **V1** | Vulkan | the 7B pack runs end-to-end on a named GPU; logits agree with the CPU path under a stated test |
| **V2** | SPIR-V | the kernels build from in-tree source to SPIR-V; no opaque shader blob is the serving path |
| **V3** | Lean | the kernel source is extracted by Aeneas/Charon; a refinement theorem links the extracted body to the canonical arithmetic; the axiom audit is within `propext`, `Classical.choice` and `Quot.sound` |
| **V4** | proof control | a mutation of the real kernel source turns the proof red; restoring it turns it green |
| **O1** | open source | a clean clone builds the serving path and reproduces the pack recipe under Apache-2.0 |
| **G1** | contributor gate | tests, clippy, fmt, `tools/check-claims.py`, `tools/book-check.py` and the book build pass on the target commit |
| **G2** | maintainer gate | `proof/regen.sh --check` and `lake build` pass for every extracted file the change touches |
| **G3** | receipt discipline | every receipt names the criterion it advances and the part it leaves open |
| **S1** | load safety | a preflight refuses a load over the derived memory cap; a 30-minute five-session soak produces no OOM kill |

## Workstreams

1. Freeze the checkpoint, tokenizer, Q4_K_M reference, KV geometry and evaluation harness.
2. Build the full 1.15–1.18 b/w pack and measure Q1, Q2, M1 and T1 at 2k, 32k, 128k and 200k.
3. Replace the context path. ColdKV-class, quantised KV, linear/state-space attention and hybrids are
   candidates. ColdKV's full-KV cold store alone does not fit 200k; the 8B f16 KV is about 28.8 GB
   and Qwen2.5-7B f16 KV is about 11.5 GB, so the cold tier must be quantised or bounded.
4. Serve through `serve/`, wire `../intent`, and run the 20-task suite.
5. Write the kernel source once for both SPIR-V and Aeneas/Charon, prove refinement, and run the
   GPU/CPU agreement and mutation tests.
6. Close the gates, write the receipts, verify a clean clone and publish under Apache-2.0.

## What a receipt must say

A receipt names the criterion id it advances, the grade (`proved`, `tested`, `measured`, `model` or
`anchor`), the command and machine, and the scope it does not cover. A pack, a row, a 0.6B result or
one projection's speedup is an artifact. It does not close the goal.

## What is not claimed

The proof covers the kernel source, not rustc, LLVM, the kernel ABI, the Vulkan driver or the GPU.
Training a new base model is out of scope. A full 200k-token prefill at interactive speed is not
claimed unless it is measured; the interactive bar is decode at 200k and a 1k incremental turn.
`src/` and `examples/` open no socket; `serve/` is the only listener.
