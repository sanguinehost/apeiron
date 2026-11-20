# Decode ceiling on the 6.7 GiB laptop: memory, the LM head, and instructions

Criterion: **T1** (decode), with **T2** as the latency context. Grade: `measured` for the
streaming-read and demo arms, `model` for the per-token and ceiling arithmetic. Scope: Qwen2.5-0.5B
Q4_K_M on this host and Qwen2.5-7B-Instruct at the 1.174 b/w projection. The 7B pack is not built,
no 200k state is allocated, and the 200k rows are arithmetic. This receipt does not close T1.

## 1. The host and the power profile

ThinkPad L13 Yoga 20R6, i5-10210U (4C/8T, 1.6 GHz base, 4.2 GHz max), 8 GB soldered DDR4-2666,
dual channel (`lscpu`; the Lenovo PSREF lists 8 GB and 16 GB models as dual-channel), 6.9 GiB
visible, Ubuntu 24.04, on battery.

The first measurements ran with `powerprofilesctl get` = `power-saver` and
`/sys/firmware/acpi/platform_profile` = `low-power`; the package sat at **0.6–0.9 GHz** under a
four-thread read. `powerprofilesctl set performance` set both to `performance`; the same read held
**3.9–4.2 GHz**. The two arms differ by 3.0× in decode and 3.6× in streaming bandwidth. The profile
is part of the measurement, not a detail. The profile is currently `performance`; restore it with
`powerprofilesctl set power-saver`.

## 2. Streaming read

`tools/membw.c`; four threads pinned to CPUs 0–3 (the four physical cores; siblings are 0/4, 1/5,
2/6, 3/7), 128 MiB per thread, 60 timed passes after two warmups. The chunk is far larger than the
6 MiB L3. Commands:

```sh
agent-guard run -- cc -O3 -march=native -pthread -o /tmp/membw tools/membw.c
agent-guard run -- /tmp/membw 4 128 60    # three times
agent-guard run -- /tmp/membw 1 256 60
```

| profile | aggregate GB/s | mean | spread |
|---|---|---:|---:|
| power-saver | 7.78 (one 120-pass run) | 7.78 | — |
| performance | 28.83, 27.70, 27.38 | **27.97** | 1.45 (5.2%) |
| performance, one thread | 18.67 | 18.67 | — |

The spec peak is `2 × 2666 MT/s × 8 B = 42.656 GB/s`. The four-thread mean is **65.6%** of peak. The
best single pass was 34.58 GB/s; 28.0 is the sustained number.

## 3. What one token reads

The token embedding is a lookup; the output head is a full projection. The file size is not the
per-token bytes.

**Qwen2.5-0.5B-Instruct Q4_K_M** (491,400,032 B, sha256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`): `token_embd.weight` is Q5_0,
136,134,656 weights = 93,592,576 B, and is not read in full; `output.weight` is Q8_0, 136,134,656
weights = 144,643,072 B, and is read every token. The per-token read is
`491,400,032 − 93,592,576 + 616 = **397,808,072 B**` (the 616 B is one Q5_0 embedding row).

**Qwen2.5-7B-Instruct at 1.174 b/w**: the pack is 957,615,232 B over the seven per-layer linears
(6,525,288,448 weights; `bench/cpu/1794781140-nq7b-118-full-pack.md`). The GGUF `output.weight` is
Q6_K, 544,997,376 weights = **447,068,160 B**, read every token by `Engine::lm_head`
(`src/engine/step.rs:1231` reads `output.weight` when present). The per-token read is
`957,615,232 + 447,068,160 = **1,404,683,392 B**`. The 957.6 MB projection excludes this head; the
GOAL already names the LM head as a blocker for a full-model rate claim (`GOAL.md`, "What the 1.15
b/w sample does and does not show"). The token embedding is a lookup and is not counted.

## 4. Ceiling table

`B/token` includes weights, head and the named KV read. `peak` uses 42.656 GB/s; `meas` uses the
27.97 GB/s four-thread mean. Rows are `model` unless stated.

| model | context | KV read | B/token | peak tok/s | meas tok/s |
|---|---:|---|---:|---:|---:|
| 0.5B, 42-token demo | 42 | Q32, negligible | 397.8 MB | 107.2 | 70.4 |
| 0.5B | 2k | Q32 `KvState::new` 98.3 MB | 496.1 MB | 86.0 | 56.4 |
| 0.5B | 2k | 1.5-bit 2.3 MB | 400.1 MB | 106.6 | 69.9 |
| 0.5B, hypothetical | 200k | C1-rate 196.2 MB | 594.0 MB | 71.8 | 47.1 |
| 7B, no KV | — | — | 1,404.7 MB | 30.4 | 19.9 |
| 7B | 2k | Q32 458.8 MB | 1,863.4 MB | 22.9 | 15.0 |
| 7B | 2k | F16 114.7 MB | 1,519.4 MB | 28.1 | 18.4 |
| 7B | 2k | 1.5-bit 10.8 MB | 1,415.4 MB | 30.1 | 19.8 |
| 7B | 200k | C1 plan 869 MB | 2,273.7 MB | 18.8 | 12.3 |
| 7B | 200k | C1 cap 1 GiB | 2,478.4 MB | 17.2 | 11.3 |

The 0.5B checkpoint declares a 32,768-token context; the 200k row is a byte model for a compressed
context, not a runnable demo. The 7B checkpoint also declares 32,768; 200k is the C1 design target,
not a native context.

The measured 0.5B demo has two arms on the same binary, model and profile. Session A, earlier on
the same profile with other work on the box: 18-token runs 32.22, 28.34, 27.52 tok/s; a 96-token
run 29.24; a 50-token run 26.04. Session B, quiet box (loadavg 0.9, package 70 °C, 3.9–4.1 GHz):
18-token runs **41.05, 40.31, 39.55 (mean 40.30)**; 96-token runs **38.16, 35.80, 35.04 (mean
36.33)**; a 50-token run 40.64. At 397.8 MB/token, 40.30 tok/s is **16.03 GB/s**, **57.3%** of the
measured streaming ceiling and 37.6% of spec peak; the 96-token mean is 14.45 GB/s (51.7%). The
power-saver arm measured 10.24 tok/s = 4.07 GB/s. The observed spread across sessions is 26.0–42.1
tok/s; the two sessions differ in background load and package temperature, and the cause was not
isolated. The 0.5B is not bandwidth-saturated even at 40 tok/s; the instruction stream is the other
limit.

## 5. The T1 bars

**30 tok/s at 2k on the 7B at 1.174 b/w.** Weights plus the Q6_K head need
`1,404,683,392 × 30 = 42.14 GB/s`, **98.8% of spec peak** before any KV or activation. With a
1.5-bit KV at 2k it is 42.46 GB/s, **99.5% of peak**. The measured 28.0 GB/s gives **19.8 tok/s**.
The bar is not reachable at 1.174 b/w with the untied Q6_K head on this host, even with a perfect
kernel and zero overhead.

Compressing the head into the 1.174 b/w pack would save 367 MB/token: 1,037.6 MB, and 30 tok/s
would need 31.13 GB/s (73.0% of peak, 111% of the measured 28.0). With a 1.5-bit KV, 31.45 GB/s. That
is the difference between "at the peak" and "needs a lower-b/w head plus AC/cooler bandwidth." A
lower-b/w linear pack is the other lever.

**24 tok/s at 200k.** With the 869 MB C1 plan read in full, 2,273.7 MB/token, 24 tok/s needs
**54.57 GB/s**, 127.9% of peak. The peak ceiling is 18.8 tok/s; the measured ceiling is 12.3. At the
measured 28.0 GB/s, 24 tok/s allows 1.167 GB/token, which is less than the 1,404.7 MB of weights and
head alone: **24 tok/s at 200k is impossible at the measured bandwidth even with a free context**.
At peak, 24 tok/s allows 1.777 GB, so the context read must be at most **372 MB/token** (43% of the
869 MB plan). A lower-b/w head and pack are also required.

The 30 tok/s 2k bar and the 24 tok/s 200k bar therefore need the LM head in the compression budget,
and the 200k bar additionally needs a context method that reads a minority of the state per token.

## 6. The instruction ceiling

The current path is about **1 instruction per weight**: the Q5_0 decode is 54 instructions per four
32-lane blocks (0.42/weight), the four-block dot is 55 (0.43/weight), and the four `i128` scale folds
are about 0.19/weight (`bench/cpu/1794793277-kernel-asm.md`). At four cores and 4 GHz, 1 IPC is
16G instructions/s, so 6.525G weights/token cap at **2.4 tok/s** at 1 IPC and **4.9 tok/s** at 2 IPC.
The dot core itself is 2 instructions per 32 weights (0.0625); the decode and scale dominate.

The measured 0.5B at 40.3 tok/s processes about 494M weights/token (357.9M linears plus the 136.1M
Q8_0 head; the token embedding is a lookup and the norms are negligible) = **19.9G weights/s**. At
one instruction per weight on 15.6G cycles/s (4 × 3.9 GHz) that is about **1.28 IPC**. The 7B at
30 tok/s processes 7.07G weights/token = **212G weights/s**, 10.7× the 0.5B's weight rate. At 2 IPC
the CPU can issue 32G instructions/s: the 7B needs **≤0.151 instructions/weight**, and at 1 IPC
≤0.075. The current ~1.0 is 6–13× too heavy. An ideal 4-bit `vpshufb` carrier (one shuffle, one
`vpmaddubsw`, one `vpmaddwd` per 32 weights, plus the scale) is ~0.12–0.19 and could approach the
memory ceiling at ~1.6–2 IPC; the current Q5_0 decode cannot.

The 0.5B at 57% of the measured bandwidth is not evidence that the 7B will reach 30: the 7B is 20×
the weights per token, and the memory ceiling at 1.174 b/w is already at the spec peak once the head
is counted.

## 7. What remains

- Build the full pack with the LM head included, or compress the head separately; recompute T1.
- Measure on AC and after a thermal soak. The battery `performance` arm is 28.0 GB/s; the spec peak
  is 42.7 and the AC/cooler number is not measured.
- Build and run the 7B pack. The 8B v3 pair at 0.09 tok/s is a different pack and path
  (`bench/cpu/1794690665-nq8b-laptop-serve.md`).
- Choose a context method that reads ≤372 MB/token at 200k for 24 tok/s at peak, and much less at
  28.0 GB/s. The C1 allocation cap alone does not establish the T1 rate.
- Reduce the instruction count with the 4-bit LUT carrier and the assembly work; the 0.5B result
  shows software headroom, but the 7B target needs both the bytes and the instructions.

## 8. Reproduce

```sh
powerprofilesctl get
powerprofilesctl set performance
agent-guard run -- cc -O3 -march=native -pthread -o /tmp/membw tools/membw.c
agent-guard run -- /tmp/membw 4 128 60    # three times
agent-guard run -- /tmp/membw 1 256 60
./demo/start.sh
./demo/ask.sh "Summarise this client note in one plain sentence, without adding anything: \
  'I have been sleeping badly and worrying about work. I would like to talk to someone.'"
./demo/ask.sh "Write a warm, two-sentence welcome message for a community mental health clinic. \
  Use plain language."
./demo/stop.sh
```

The demo uses `serve/target/release/nq-serve` built with `cargo build --release --locked` and
`APEIRON_THREADS=4`; the power-saver and performance arms are the same binary and model file. For a
state claim, quote the power profile, the load average and the package temperature with the rate:
the two performance sessions above differ by 1.4× with the profile unchanged.
