# The card — what to say in sixty seconds

**Open.** "This laptop is from 2019: 6.7 GiB of RAM, integrated graphics, no discrete GPU. The Wi-Fi
is off. It is running a language model, and everything typed stays on this machine."

**Why it matters.** "Most AI today sends your words to someone else's data centre. For a clinic,
that is patient text leaving the room. A local model is private, free at the margin, works offline,
and can be inspected. The industry is moving this way because phones and laptops now have the
hardware; the bottleneck has moved to memory."

**The work.** "A 7-billion-parameter model is about 14 GB at full precision. The 7B target is
roughly 1.17 bits per weight — under a gigabyte — and a 200,000-token memory to about a gigabyte,
with kernels fast enough to be useful. What is measured today is a 0.6B pack at 0.978 bits per
weight: 53.8 MB, 2.52× Q4_K_M perplexity at 8,176 tokens, with layer 0 carrying 3.15× and the other
layers 1.02–1.14×. The full 7B pack is the remaining campaign. I also prove the low-level arithmetic
correct in Lean. Most people have one of ML, systems or proofs; this is the intersection."

**The career.** "The skills are ML systems, performance engineering and formal verification. The
roles are inference, performance, kernel or research engineer, in chip companies, labs, cloud,
inference startups, finance and health technology. This repository is the portfolio: code,
measurements, receipts and proofs that someone can run. The next steps are concrete: publish the
1.15–1.18 bits-per-weight pack, write the report, give a talk, contribute a kernel or benchmark,
apply to teams that measure tokens per second and bytes per weight."

**The caveat.** "The 7B-at-20-tokens-per-second goal is not done. Today's live model is half a
billion parameters: about 0.4 seconds to the first token, then 35–41 tokens per second on a quiet
box. The measured 7B projection is 11.0 tokens per second on the q8 reader (int4 14–18), and the
full 7B pack is not built: the T=200 CPU fit is 504 h, and a T=2 screen stopped at 12 of 196
sections for cost. The method is the point: measure, compress,
verify, publish."

## The three numbers

| | |
|---|---|
| 0.978 bits/weight | the measured 0.6B v3 pack: 53.8 MB, 196 sections, 2.52× Q4_K_M at 8,176 tokens |
| 2.52× Q4_K_M | the 0.6B pack's perplexity ratio at 8,176 tokens (4.10× at 2,044); layer 0 is 3.15× alone |
| 11.0 tokens/second | the 7B q8 projection on this laptop; the target is 20, the int4 selector 14–18, and the GPU/Vulkan path is open |

## If they ask "is it faster than llama.cpp?"

Not established on this laptop. The one same-model comparison ran in `power-saver`: llama.cpp
b12428 did 27.7 prompt tokens/s (~1.5 s to first token) and 10.5 generated tokens/s; this listener
did 22.1–23.5 and 10.0–10.3. llama.cpp was 1.2–1.3× faster to the first token and about 1.02–1.05×
faster generating. The listener has since measured 35–41 generated tokens/s in `performance`, so
re-run llama.cpp under the same profile before making a cross-engine claim. The project's measured
wins are on a 32-thread i9 (1.4–1.8×) and an RTX 5080 (1.1–1.8×). The demo's claim is privacy and
locality, not speed.

## If the live demo fails

Open `demo/recorded/live-answer.txt`, say "here is the same run recorded before the meeting", and
continue with the explanation. Do not debug in front of the visitor.
