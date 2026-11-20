# A language model on an ordinary laptop

### What you will see

A 2019 laptop with 6.7 GiB of RAM and no discrete graphics card. A 491 MB language model runs on
its CPU. The Wi-Fi is off. You type a question; on the `performance` power profile the first token
arrives in about 0.4 seconds and the answer is complete in under a second to three seconds. On
`power-saver` it is about two seconds and four to six seconds — the same binary, with the package
clocks 4–5× lower.
The live answer is computed on the machine in front of you, and nothing is sent anywhere. (The
larger benchmark numbers in this page come from the receipts in `bench/cpu/`, not from the live
model.)

The model is deliberately small for the live demo: half a billion parameters. It is there to show
the whole path working — model, tokenizer, kernels, server, terminal — at a speed a conversation
can follow. The project's target is a **7-billion-parameter** model on this same laptop.

### Why the industry cares

Most AI today is a visit to someone else's data centre. Your words leave the room, are billed by
the token, and are governed by a contract you did not write. For a clinic, a law firm, a defence
team or a person keeping a private journal, that is not a detail. It is the reason they cannot use
the tool at all.

On-device AI removes that constraint. It is also where the industry is moving: phones and laptops
now ship neural hardware (Apple Intelligence, Copilot+ PCs, Snapdragon X, RTX AI PCs), and the
competitive question has changed from "who has the biggest model" to "who can make a useful model
fit and run fast on the device the person already owns". The bottleneck has moved from arithmetic
to memory: a 7-billion-parameter model at 4 bits is about 4 GB, which needs a new machine; at 1.17
bits it is under 1 GB, which fits the one already in the clinic.

The hard part is memory. A 7-billion-parameter model is about 14 GB at 16-bit precision. Most
laptops have 8–16 GB in total. To fit one you need roughly **one bit per weight** without
destroying quality, a **200,000-token context in about one gigabyte**, and kernels that turn those
compressed formats into tokens at **20 tokens per second** (30 is the stretch goal). Those three
numbers are the project.

There is a second, rarer layer: the low-level kernels are proved correct in Lean, a formal
mathematics system. That is not common in machine learning, and it is valuable wherever a silent
arithmetic bug is expensive — regulated industries, medical devices, finance, safety systems.

### What the project has actually shown

The honest status, with the artifacts behind it:

- **0.978 bits per weight** is the measured 0.6B v3 pack: 53,815,472 B, 196 sections, **2.52×
  Q4_K_M** perplexity at 8,176 tokens (4.10× at 2,044). The layer controls locate the loss: layer 0
  alone is **3.15×**, layer 13 **1.018×**, layer 27 **1.136×**.
- **1.174 bits per weight** is the 7B geometry: 957.6 MB over 6.525 billion weights, all 28 layers
  calibrated. The full pack is not built. The measured fit is 94 s per `down` iteration and 2.52 h
  per model iteration, so T=200 is 504 h on a CPU box; a T=2 screen stopped at 12 of 196 sections
  when the four-hour cap met a 5.2–6.3 h fit.
- An **8B pack at 1.124 bits per weight** is **1.755×** Q4_K_M at 128 context; at 3.13 bits it is
  **1.011×** (16.295 PPL) and runs on this laptop at 0.09 tokens per second, which is why it is not
  the live demo.
- On rate, the 7B **q8 reader is 11.0 tokens per second** at 2.1 IPC, the int4 selector 14–18.
  Both are under the 20 target; the open route is the GPU/Vulkan kernel.
- A **200,000-token context budget** of 1.4976 bits per value (about 1 GiB for the key/value state)
  is derived and written down; the compressed carrier is not implemented.
- In Lean, the CPU word bodies and the ring operations are refined; the Vulkan/SPIR-V path is not.

A benchmark row, a pack or a speedup is an artifact of the goal, not the goal. The goal is the
whole thing: 7B, 200k context, 20–30 tokens per second, five agent sessions, verified kernels.

### Why this is a career

The work sits where three scarce skills meet:

- **Machine-learning systems**: making a model fit and run fast, not training another one.
- **Performance engineering**: memory bandwidth, quantisation, CPU and GPU kernels, benchmarking.
- **Formal verification**: proving the low-level code does what it claims.

Most candidates have one of those. The combination is the differentiator, and the repository is
the portfolio: code, measurements, receipts and proofs that a hiring manager can run.

Roles that hire for it: inference or ML-systems engineer, performance engineer, kernel or compiler
engineer, research engineer for efficient ML, quant developer, and edge or privacy-preserving AI.
The employers are chip companies, labs, cloud providers, inference startups, finance and health
technology. Indicative senior compensation is roughly A$180k–300k in Australia and US$200k–400k+
total compensation in the United States; it varies with company, level and location, and it is not
a promise.

The path is concrete: finish and publish the 1.15–1.18 b/w pack; write the "1.17 bits per weight"
report; give a talk at a Rust or ML meetup; contribute a kernel or a benchmark to an open-source
inference project; apply to teams that measure tokens per second and bytes per weight. The live
demo is the communication step: it makes the constraint visible to someone who does not work in
the field.

The caveat matters as much as the case. The 7B-at-30-tokens-per-second goal is **not met** today.
The demo shows a 0.5B model at about 0.4 seconds to the first token and 35–41 tokens per second
in `performance` (ten tokens per second in `power-saver`), and the path to the target. The 7B is
20× the weights per token: at 1.174 b/w the pack is 0.998 GB per token, and 20 tokens per second
needs 20.0 GB/s, 71% of the measured 27.97 GB/s. The career case rests on the method — measure,
compress, verify, publish — not on a single number.

### What this is not

It is not a therapist, a diagnosis, or a medical device. A clinician stays in the loop: the model
drafts, the human decides. The value is in privacy, cost, access and control, not in replacing
judgment.
