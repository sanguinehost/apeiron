# Demo runbook — ten minutes, one laptop, no cloud

For a visitor who does not work in software. The goal is not to impress with a chatbot; it is to
make one constraint visible: **the model runs here, and the data does not leave**.

## What runs, and what does not

| | |
|---|---|
| live model | Qwen2.5-0.5B-Instruct, Q4_K_M, 491,400,032 B, CPU only |
| path | `serve/` listener on `127.0.0.1:8080`, greedy decode, no network |
| power profile | `performance`, quiet box: 0.29–0.42 s to first token, 35–41 tok/s; earlier session under load: 0.46–0.60 s, 27.5–32.2 tok/s |
| power profile | `power-saver`: 1.8–1.9 s to first token, 10.0–10.3 tok/s |
| end-to-end | performance 0.8–0.9 s for 18 tokens, 1.5 s for 50; loaded session 1.0–1.3 s; power-saver 3.6 s |
| memory | about 600 MB peak resident |
| not live | the 7B pack (the T=200 CPU fit is 504 h; a T=2 screen stopped at 12 of 196 sections for cost), the 200k context carrier (not implemented), the GPU path (benchmarked separately) |

The 0.5B model is a demonstrator of the stack, not a quality claim. Say that out loud once, early.

## Pre-flight

Do this **the night before** and again **ten minutes before**:

**Power profile.** Run `powerprofilesctl set performance` for the fast arm (0.29–0.42 s to first
token, 35–41 tok/s on a quiet box; 0.46–0.60 s and 27.5–32.2 tok/s in a loaded session).
`power-saver` gives the 1.8–1.9 s / 10.0–10.3 tok/s arm. Check
`powerprofilesctl get` before quoting a number; do not mix the two arms. Restore `power-saver`
after the demo if the battery matters.

1. Close browsers, Zotero, other model jobs and any `cargo`/`rustc` build. This box has 6.7 GiB and
   a full 4 GiB swap; a 4 GB process can take the desktop down. A 4 GB `python3` outside
   `agent-guard` has done exactly that twice; do not start one before the demo. Check with
   `free -m`: you want `MemAvailable` above 2,500 MiB.
2. `./demo/check.sh` — it reports memory, load, the model hash, the listener and any build job.
3. `./demo/start.sh` — starts the listener under `agent-guard`, waits for `/health`, prints the
   model size and the warmup time. About 15–30 s.
4. `./demo/ask.sh "Summarise this client note in one plain sentence, without adding anything: 'I have been sleeping badly and worrying about work. I would like to talk to someone.'"`
   — one warm run, then **stop talking to the model** until the visitor arrives. The answers are
   greedy and deterministic: the warm run is the same text you will get live.
5. Turn the terminal font up. Put the window on the projector. If you want the offline proof, turn
   Wi-Fi off **after** the server is up; nothing in the demo needs the network.
6. Keep `demo/recorded/live-answer.txt` open in another tab. It is the fallback.

## Speed honesty

The same binary, model and prompt give different arms: the platform profile changes the package
frequency by 4–5×, and a loaded session costs another 1.2–1.4×:

| arm | prompt | generation |
|---|---:|---:|
| this listener, `performance`, quiet box | 0.29–0.42 s (100–145 tok/s for 28–42 tokens) | 39.6–41.1 tok/s over 18 tokens; 35.0–38.2 over 96; 40.6 over 50 |
| this listener, `performance`, loaded session | 0.46–0.60 s (70–91 tok/s for 42 tokens) | 27.5–32.2 tok/s; 29.24 over 96 tokens, 26.04 over 50 |
| this listener, `power-saver` | 1.79–1.90 s (22.1–23.5 tok/s) | 10.0–10.3 tok/s |
| llama.cpp b12428, `power-saver` | 27.7 tok/s (~1.5 s) | 10.5 tok/s |

The llama.cpp row was measured in `power-saver`; it is not comparable to the listener's
`performance` row. Re-run llama.cpp under `performance` before making a cross-engine claim on this
laptop. The project's measured wins against llama.cpp are on a 32-thread i9-14900KF (1.4–1.8×) and
an RTX 5080 (1.1–1.8×), not on this 4-core/8-thread 15 W laptop. The demo's claim is locality,
privacy and the compression path — not speed. The live model is a stock Q4_K_M file, not the
project's NanoQuant pack.

The 0.5B is not the 7B target. At 1.174 b/w the 7B pack is 957.6 MB and reads 0.998 GB/token with
the fitted head; the laptop's measured 27.97 GB/s caps that at **28.03 tok/s**, and 20 tok/s needs
20.0 GB/s, 71% of measured. The kernel is the other bound: the q8 reader is 11.0 tok/s at 2.1 IPC,
the int4 selector 14–18. The honest 7B status is therefore **11.0 tok/s on the measured q8 path,
2.0× short of 20**, with the GPU/Vulkan path as the open route. A CUDA probe of the same 1-bit carrier
reads **539.5 GB/s** on an RTX 2080 Ti (87.6% of spec), a **540.5 tok/s** ceiling for the 0.998
GB/token pack — **27× the 20 target**. That probe is not the repository's Vulkan kernel. Vulkan
now initializes on a rented RTX 3060 (proprietary driver 580.173.02 plus the X/GL userspace), and
the repo's benchmark compiles and submits dispatches, but the decode GEMV returns **zero** on every
checked row, so V1 is open with the failure localized to the first dispatch. The arithmetic is in
`bench/cpu/1794799742-decode-ceiling.md`, `1794822551-nq-loop-ports.md`,
`bench/gpu/1794831271-gpu-gemv.md` and `bench/gpu/1794836795-vulkan-bringup.md`.


**0:00–1:30 — Frame.** Show `free -m`, `lscpu | grep -E "Model name|^CPU\(s\)"` (4 cores /
8 threads, no discrete GPU), and `lspci | grep -i vga`. Point at the model file:
`ls -lh /home/lucas/models/qwen2.5-0.5b/*.gguf` — 491 MB. Say: "This is the whole model on disk.
It runs on the CPU. The Wi-Fi is off."

**1:30–3:30 — Ask it something.** Start with the note summary, because it is the clinical case:

```sh
./demo/ask.sh "Summarise this client note in one plain sentence, without adding anything: 'I have been sleeping badly and worrying about work. I would like to talk to someone.'"
```

It returns in about one second in `performance` (about four in `power-saver`): *"The client is
experiencing sleep issues and worries about work. They want to talk to someone."* Say: "That is the
kind of text that must not leave the room." While it prefills, do not stand in silence: point at
the 600 MB and the 491 MB file and say the first token takes under a second in `performance` and
writes at about 35–41 tokens a second; in `power-saver` it is about two seconds and ten tokens a
second. If there is time, the welcome message: about 1.5 s end-to-end in `performance`, 50 tokens
at about 40 tok/s after a 0.3 s first token. Show `ps`/RSS: about 600 MB. "That is the entire
runtime, not a client talking to a server farm."

**3:30–5:30 — Why it matters.** One sentence each:
- **Privacy**: a clinic's patient text stays in the room; there is no vendor, no upload, no
  third-party terms.
- **Cost**: after the laptop, each question is free at the margin; a clinic with no cloud budget
  can still use it.
- **Access**: it works offline, in a rural clinic, on a plane, in a room with no reliable network.
- **Control**: the code is open and the kernels are being proved correct; you can inspect what
  runs.

**5:30–8:00 — The real target.** A 7B model is ~14 GB at 16-bit; this laptop has 6.7 GiB. The
project's numbers: **1.174 bits per weight** is the 7B geometry (957.6 MB over 6.525 billion
weights, all 28 layers calibrated), a **200k-token context at 1.4976 bits per value** (~1 GiB), and
**20 tokens per second** on the 7B. Show `demo/explain.md` or the one-page handout. Be explicit
about what is measured: the **0.6B v3 pack at 0.978 b/w** scores **2.52× Q4_K_M at 8,176 tokens**
(4.10× at 2,044); its layer controls put the loss in layer 0 (**3.15×** alone, against **1.018×**
for layer 13 and **1.136×** for layer 27); the 8B pack at 1.124 b/w is **1.755×** at 128 context,
and at 3.13 b/w it is **1.011×**. The full 7B pack is **not built**: the measured fit is 94 s per
`down` iteration, 2.52 h per model iteration, so T=200 is 504 h on a CPU box, and a T=2 screen
stopped at **12 of 196 sections** when the four-hour cap met a 5.2–6.3 h fit. On rate, the q8 reader
is **11.0 tok/s** on the 7B projection, the int4 selector 14–18. The distance is the work, not a
hidden failure.

**8:00–10:00 — Why it is a career.** The three skills (ML systems, performance, verification), the
roles, and the portfolio: this repository is the evidence. Say what is next: publish the pack,
write the report, give a talk, contribute a kernel, apply to inference teams. Invite the obvious
question: "What would you want me to explain more plainly?"

## If it goes wrong

- **First token takes more than about 20 s**: another build or browser is competing. Stop, run
  `./demo/check.sh`, and use the recorded answer.
- **The listener dies**: `./demo/stop.sh`, then `./demo/start.sh`. If it dies twice, switch to the
  recording and spend the time on the explanation; do not debug in front of the visitor.
- **The answer is wrong or odd**: say so. "This is a half-billion-parameter demonstrator; quality
  is the 7B target, not this model. The point today is where it runs." Do not defend a bad answer.
- **A crash or OOM**: close the terminal, check `free -m`, and restart. The demo is not worth
  risking the desktop; the recorded path is enough.

## Questions a clinician may ask

- *Is it safe for patients?* It is not a medical device. Data stays local; a clinician reviews
  every output; the model can be wrong; evaluation is required before any clinical use.
- *Will it replace therapists?* No. It drafts, summarises and organises; the relationship and the
  judgment stay human.
- *Why not just use ChatGPT?* Quality today is better in the cloud. Privacy, cost, offline use and
  control are the reasons to accept a smaller local model — and the gap is closing.
- *Is this a real career?* Yes: inference and performance roles are growing as models move onto
  devices. The portfolio is the proof; the project is not finished, and finishing it is the work.
- *What is the hard part?* Memory bandwidth and quality at one bit per weight, a 200k context in a
  gigabyte, kernels that are fast and proved correct.

## Cleanup

`./demo/stop.sh` releases the listener and the `agent-guard` lock.

## Files

| file | use |
|---|---|
| `demo/start.sh` | start the listener under the resource guard |
| `demo/ask.sh "..."` | one question, streamed, with timings and RSS |
| `demo/stop.sh` | stop it |
| `demo/check.sh` | pre-flight: memory, load, model, listener |
| `demo/explain.md` | the one-page explanation to leave behind |
| `demo/handout.html` | the same page, printable |
| `demo/recorded/live-answer.txt` | the fallback transcript |
