# Overnight queue: Q2 sits on the 1.5× bar over five windows, and the T1 ceiling

Criterion: **Q2** (PPL half) and **T1**. Grade: `measured` for every row below; the ceiling
arithmetic is `model` over the measured byte and instruction counts. This file is the durable state
for a compacted session: it names the result, the artifacts, the queue, the decision rules and the
resume protocol. The queue is executed by `bench/overnight/run.sh`; results append to
`bench/cpu/1794912865-nq8b-overnight-logs/` and to `bench/overnight/status.txt`.

## The result that matters

**Correction, 00:28.** The five-window measurement completed after this file was first written. The
pack does **not** hold the 1.5× bar on the dataset; it sits on it. Pooled over 5,115 scored positions
the ratio is **1.499754** (NQ 14.876661 against 9.919400), 0.0164% under the bar, with a window-level
95% interval of **[1.391674, 1.616228]**. Three windows pass (1.499226, 1.467112, 1.394272) and two
fail (**1.646477** at skip 4096, **1.502682** at skip 6144). The two-window claim below is withdrawn:
its 0.005879 PPL margin is 0.0005 nats, 1/116 of the window-to-window sd (0.060247 nats), and 0.0005
nats needs ~55,800 windows to resolve at 95% (the earlier 516 figure was the 0.0052-nat count). Three
repeats (`run13`) reproduce every printed value to the last digit (delta 0.000000), so the interval is
text variance, not run noise. Receipt
`bench/cpu/1794922136-nq8b-q2-five-windows.md`; the tool is `tools/ppl-pooled.py`. **Q2 is open on
both halves**, and `run14` extends the span to 25 windows for a ±0.025-nat bound.

The table below is retained as the window-1/2 detail it is:

| window | arm | PPL | Q4_K_M PPL | ratio | margin to 1.5× |
|---|---|---:|---:|---:|---:|
| 1 (skip 0) | **conservative, product kernel** | **11.374839** | 7.587145 | **1.499225** | **+0.005879 (+0.0517%)** |
| 2 (skip 2048) | **conservative, product kernel** | **17.361910** | 11.834074 | **1.467112** | **+0.389201 (+2.1925%)** |
| 3 (skip 4096) | conservative, product kernel | 17.226503 | 10.462647 | **1.646477** | **−1.532533 (−9.7922%)** |
| 4 (skip 6144) | conservative, product kernel | 14.057420 | 9.354884 | **1.502682** | **−0.025094 (−0.1786%)** |
| 5 (skip 8192) | conservative, product kernel | 15.236382 | 10.927844 | **1.394272** | **+1.155384 (+7.5755%)** |
| 1 | conservative, exact oracle | 11.381891 | 7.587145 | 1.500155 | −0.001174 (−0.0103%) |

Both arms ran on **this host**; the Q4 window-1 run reproduces the rented host's **7.587145** to the
last digit (`host-reproduction delta +0.000000`), so the window-1 ratio is not cross-host. Three less
conservative window-1 arms:

| arm | `output.weight` | `token_embd.weight` | PPL | ratio to 7.587145 |
|---|---|---|---:|---:|
| Q6_K head, Q8_0 embed | `Q6_K` | `Q8_0` | 11.346170 | 1.495447 |
| stock-file | `Q8_0` | `Q8_0` | 11.294969 | 1.488698 |

Q4_K_M bar on the same token stream, per window: **7.587145** (skip 0) and **11.834074** (skip
2048), from the full file sha256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`, 5,027,783,488 B. The 1.5× bars
are **11.3807175** and **17.751111**. The conservative skeleton matches the Q4_K_M file on all
**147** common non-linear tensor types and dimensions, its two quantised tensors are byte-identical to
the bar file's, and the 72 differing norms carry that file's AWQ fold
(`bench/cpu/1794921128-nq8b-tensor-audit.md`). Only the 252 fitted linears differ in function. PPL is
1,023 scored positions per window of ctx 2048, `--second-half`, `--kernel q8act`.

**Window 1 is a tie; window 2 is not.** The shipped q8act reader lands 0.0517% under 1.5× on
window 1, while the exact reader, which reads the Q32.32 activation words instead of the int8 codes,
lands **0.0103% over it** (`J4`, 11.381891, mean nll 2.432024, 2,213.5 s, 0.5 tok/s, peak
2,336.2 MiB). The 0.007052 PPL between them (0.062%) is the activation-quantisation error, and on
window 1 it straddles the bar. **Window 2's margin is 2.1925%, 35× that 0.062% term**, so window 2
passes under either reader (the exact reader would score ≈17.3727, ratio 1.468021). Window 1 and
window 2 are two of the three windows that pass; the pooled five-window estimate is 1.499754 with an
interval that contains the bar. The reader term (0.062%) is 1/38 of the interval's half-width
(±2.3%), so the reader choice does not decide Q2 on this span. The 1.15× stretch is open. The upstream `quality.json` pointer is piqa 0.800→0.535, arc_easy
0.805→0.400, arc_challenge 0.535→0.340, hellaswag 0.635→0.405, winogrande 0.705→0.570, boolq
0.855→0.660 on its harness; it says the 0-shot half is the larger risk even where PPL is 1.6×.

## Artifacts

| file | bytes | sha256 |
|---|---:|---|
| `/home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf` | 867,740,256 | `4a1978ceca0424ca37d244b1909f4eb6dcfc1748dc1447db923a49ea95eec59b` |
| `/home/lucas/models/arelath/nq8b-late-v4/model.nqp` | 1,023,456,080 | `271fd5a7862b1559c9cdc3338bc5dc5d0e8f50a5ec52fcd2a1bf18342b0dfea3` |
| `/home/lucas/models/arelath/nq8b/skeleton-q6k.gguf` | 1,178,905,184 | `28b49f7382bb13323f6027503f6115d367b623e2683d8a8effd2eafa471cd194` |
| `/home/lucas/models/arelath/nq8b/skeleton.gguf` | 1,329,625,696 | `3927454ca530ce26afdcd5dafe78f2d29d5ac006ea09151c8387d973a2ceaa28` |
| `/home/lucas/models/arelath/qwen3-8b-nanoquant.gguf` (converter source) | 2,214,297,824 | `75d3f4f150195e0c6a7ccd2ce3372f919534fb7ec609f6b9ba52eb71af194253` |
| Q6_K head blob | 510,504,960 | `05d7a76c29a32e067f41ad91df565341b21f4bedeeffbbf9e10957aaa9f78cc1` |
| Q4_K token blob | 350,060,544 | `01f3917527232d9df450f31e629c053104026f86173fcd9ee7dd5c14d33c8f15` |

The four earlier PPL runs are `ppl-late-v4-{stock,q6k,q6k-q4kembed}-ctx2048.log` and the stock v4
run in `1794896874-nq8b-depth-logs/`. The conservative log is
`ppl-late-v4-q6k-q4kembed-ctx2048.log` (mean nll 2.431404, 542.0 s, 1.9 tok/s, peak 2337.1 MiB).

The compact skeleton was built from `skeleton.gguf` by replacing `output.weight` with the stock Q6_K
blob and `token_embd.weight` with the stock Q4_K blob, rewriting every later offset. The first build
zeroed `output_norm.weight` (the 16,384 B gap after `output.weight`) and produced uniform logits
(PPL 151936); the fix copies those bytes. The build recipe belongs in `tools/` before the next agent
needs it.

### The NQ arm on windows 2–5

| window | skip | NQ PPL | mean nll | break-even Q4 (= NQ/1.5) |
|---|---:|---:|---:|---:|
| 2 | 2048 | **17.361910** | 2.854279 | **11.574607** |
| 3 | 4096 | 17.226503 | 2.846449 | 11.484335 |
| 4 | 6144 | 14.057420 | 2.643150 | 9.371613 |
| 5 | 8192 | 15.236382 | 2.723686 | 10.157588 |

Window 1 is 11.374839. The NQ arm is **1.24–1.53× harder on windows 2–5 than on window 1**, so the
pack's own ratio there is unknown until the Q4 arm runs the same windows. Each row's break-even is
the Q4 PPL that would put that window exactly at 1.5×: Q4 must score **above 11.574607** on window 2
for the pack to pass there. `run4` measures Q4 windows 1–2; `run7` measures 3–5 if 1–2 succeed.
Grade `measured`; the break-even is arithmetic on it. A window is *not* comparable to window 1's
7.587145 — pooling the two would be a different-window ratio.

### What would move Q2 off the bar

The pack is **1,023,456,080 B**; the 1.18 b/w cap over 6,945,767,424 weights is **1,024,500,695 B**,
so the headroom is **1,044,615 B**. One extra plane on a 4096×4096 linear is 2,097,152 B and on a
4096×12288 linear 6,291,456 B, so the cap cannot fund a third plane on any fitted shape. The qkv
dedup that took v4 from 1.284126 to 1.178785 b/w was spent on staying under the cap, not on a
reserve. **At ≤1.18 b/w the only route to a margin is a better fit at the same bit count** — more
iterations, a different layer/rank allocation, or a better preconditioner — because the rate lever
is exhausted. A T increase is a GPU-campaign item (`tools/fit-budget.py`), not a night item.

## The speed ceiling, before any kernel work

The laptop is an Intel i5-10210U (Comet Lake, 4 cores / 8 threads, AVX2, BMI2, POPCNT; **no
AVX-512 and no AVX-VNNI**). `nproc` reports 2 because the session exports `OMP_NUM_THREADS=2`; the
affinity mask is 0-7 and the receipts' `APEIRON_THREADS=4` runs used four threads.

| input | value | source |
|---|---:|---|
| pack bytes/token | 1,023,456,080 | v4 pack |
| Q6_K head bytes/token | 510,504,960 | stock head |
| f32 KV read at 2k | 603,979,776 | 36 layers × 8 kv heads × 128 × 2 × 4 B × 2,048; `Carrier::F32` is the default (`mixer.rs:321`) |
| **total bytes/token at 2k** | **2,137,940,816** | sum |
| measured read bandwidth | 21.11 GB/s mean, 29.47 best-pass at 4 threads | `J7b-membw.log` |
| **bytes ceiling at 2k** | **9.9–13.8 tok/s** | 21.11–29.47e9 / 2.1379e9 |
| sign terms/token | 7,623,016,448 | 238,219,264 pack words × 32 |
| current dot cost | 0.75 instructions/term | `sign_dot_planes_avx2`, 24/32 terms |
| instruction stream, exact | 5.717e9 / token | terms × 0.75 |
| instruction stream, q8act | 2.86–3.34e9 / token | terms × (12–14)/32 |
| **instruction ceiling, exact** | **4.62 tok/s** | 26.4e9 instructions/s at 2 IPC × 4 × 3.3 GHz ÷ 5.717e9 |
| **instruction ceiling, q8act** | **7.9–9.2 tok/s** | 26.4e9 ÷ (2.86–3.34e9) |

**20 tok/s is 1.45–2.03× the measured-bandwidth ceiling** at the shipped pack + Q6_K head + f32 KV,
and **2.2–4.3× above the instruction ceiling** of whichever reader runs. It also needs
**42.76 GB/s**, above the 42.656 GB/s spec peak, so it is unreachable at 2k by any kernel on this
pair: both the byte payload and the instruction stream must shrink. The pack would need ≤1.40 GB/token
(a head inside the pack at ≤80 MB) **and** ≤0.173 instructions/term, which is 5.5 instructions per
32-term group against the exact reader's 24 and the shipped q8act reader's 12–14. The prior receipt's
19.9 tok/s figure is the 7B, excludes the KV, and assumed f16; the engine's KV is f32 by default and
`APEIRON_F32_CARRIER=0` makes it Q32, which is worse. `bench/overnight/speed-plan.md` carries the
per-core counts and the lever ranking; the night's T1 work targets the dominant phase and reports the
achieved number against both ceilings. It does not promise 20.

## Night 1 findings (updated by the queue)

- **The split is measured and ffn owns the token.** `phase_split` on the conservative pair: **285.26
  ms/token (3.5 tok/s)** at ctx 64, **ffn 160.76 ms / 56.4% (ceiling 2.29×)**, attn.proj_in 47.65 /
  16.7%, **head 44.29 / 15.5%**, attn.proj_out 19.27 / 6.8%, attn.core 8.53 / 3.0%, untraced 0.31 /
  0.1%, `pool.wait` 20.44 (overlay). The three NQ phases are **227.68 ms, 79.9%**. `--target ffn`
  exits 0 on the unmodified tree (K2). Receipt `bench/cpu/1794919269-nq8b-split.md`.
- **The shipped head is 14.8× the exact reader.** `head_time` K3: head `Exact` 0.534 s (1.17 G
  weight/s) against head `Q8Act` **0.036 s (17.32 G weight/s)**; step `Exact` 1.058 s (0.94 tok/s)
  against step `Q8Act` **0.280 s (3.58 tok/s)**. The head is 12.9–15.5% of the token, not 92%: the
  0.408–0.534 s figure belongs to the reference reader. The head's own floor is 23 ms at 22.33 GB/s,
  so a perfect head kernel is ≤1.08× of the token — not the target.
- **20 tok/s is 5.7× short on the measured instruction rate.** NQ runs at **33.4 G terms/s aggregate
  (8.36 G/core)**; 20 tok/s needs 190.6 G terms/s. A hypothetical 6-instruction selector (half the
  shipped 12–14) lands at **5.84 tok/s**, 3.4× short. At 2k the f32 KV makes the payload 2.14 GB and
  needs **42.76 GB/s**, 1.002× the spec peak and 1.91× the measured 22.33 GB/s. Bytes are not binding
  today (5.4 GB/s used at ctx 64); instructions are the first wall, payload the second.
- **The q8act decode uses the code reader.** K3's step lines already decide what `run6` was queued
  to test: step `Exact` 1.058 s against step `Q8Act` 0.280 s, with the heads at 0.534 / 0.036 s, is a
  trunk of 0.524 s exact against 0.244 s q8act — **2.15×**. The code reader runs in decode; `run6`
  was cancelled rather than spend a 36-minute exact prefill on a question K3 answered.
- **K4 is cut to its three 4-thread runs.** The 1- and 2-thread 2k runs pay a 30-minute prefill to
  measure a decode width that `phase_split` gives in 25 s; the 1-thread 2k *prefill* rate is a T2
  number and T2 is already refused. `stop-k4.sh` kills the job when the third `decode` line lands,
  and `run8` measures widths 1, 2 and 8 with `phase_split`.
- **The exact reader misses the bar.** `J4` scores **11.381891 PPL, 1.500155×** on the conservative
  pair with `--kernel exact`: **0.001174 PPL (0.0103%) over** the 11.3807175 bar, at 0.5 tok/s
  (2,213.5 s, peak 2,336.2 MiB). The q8act reader on the same pair is 11.374839 (1.499225×). The
  0.007052 PPL between them is the activation-quantisation error, so the product's side of 1.5× is
  decided by a 0.062% term. The PPL half of Q2 is a tie, not a pass; a robust margin needs a fit or
  rate change worth ≥0.1% of PPL, or a second window whose Q4 bar moves the other way.
- **The Q6_K head is the first target.** `head_time` on the conservative pair measured the **Exact**
  head at **0.408 s, 1.53 G weight/s** per token before it reached the pack (`J6`). That is a
  2.45 tok/s head-only cap. The shipped `Q8Act` head is a different reader; `J6b` measures it after
  the release rebuild. If the head holds share `s`, its ceiling is `1/(1−s)`; at 0.25 s of a 0.444 s
  token it is 56% and the ceiling is 2.27×.
- **The release examples predate the v4 reader.** `phase_split`, `head_time` and `generate` exit 1
  with `Pack(BadVersion { version: 4 })`; `ppl` was rebuilt at 18:46 and reads v4. `run2.sh` rebuilds
  every release example before J5b/J6b/J8b. The failure is a stale binary, not a pack defect.
- **Bandwidth is not settled.** `membw` means are 16.69 / 16.96 / 21.11 / 17.62 GB/s at 1/2/4/8
  threads, with per-thread "best" sums of 18.21 / 23.96 / 29.47 / 50.51. The 8-thread 50.51 is above
  the 42.7 GB/s spec peak, so the tool's best-pass sum is not a simultaneous aggregate; the mean at
  4 threads (21.11) is the conservative read ceiling and the 4-thread best (29.47) is the optimistic
  one. At 2,137,940,816 B/token that is 9.9–13.8 tok/s, so 20 tok/s needs ≥42.76 GB/s, above the
  42.656 GB/s spec peak, and is not supported by any measurement of this host. `J7c` reruns a clean
  4-thread probe after the test sweep.
- **The CPU test suite is green as of J3.** 20 binaries fail and every one is a GPU or SPIR-V
  target: 19 `tests/gpu_*.rs` and `tests/spirv_storage_race.rs`, for two host causes — no Vulkan
  device (`vkCmdWriteTimestamp2 not found via vkGetDeviceProcAddr`, `src/gpu/vulkan.rs:1570`) and no
  `glslc` on `PATH`. The `--skip gpu` filter misses 31 tests whose names do not contain `gpu`; the
  contributor gate's CPU half passes. J2 clippy is green in 16 s.
- **The 0-shot inputs now exist; the verdict does not.** `tools/quality-items.py` builds the six
  lm-eval-compatible item files (200 items each, centred systematic sample, hash-anchored at
  `bench/quality/items/SHA256SUMS`): **201,440 prefill + 43,827 head positions**, 1,099,476 B, no
  candidate over ctx 2048 and **no boundary refusal** in 3,600 candidates with the product tokenizer.
  `examples/quality.rs` scores them (strict-boundary default, lm-eval `--rebound` opt-in, per-item
  JSONL, `acc`/`acc_norm`/`acc_tok`) and `tools/quality-compare.py` pairs the two arms and prints
  `screen` while any task is below its split. The model estimate was 1.56× high: the screen is
  **13–29 h** on this laptop, not 45.9 h. `run9` smoke-tests five piqa items per arm on the Q2 pair.
- **0-shot cannot be established tonight.** A 200-item-per-task screen is 314 ktok, **45.9 h** at the
  measured 1.9 tok/s scoring rate (18.7 h at the 4.66 tok/s prefill rate); the full six-task split is
  7.09 Mtok, **1,037 h per arm**. The upstream `quality.json` anchor is 200 items against the
  *dense base*, not Q4_K_M, so it is a warning. See `bench/overnight/zero-shot-plan.md`.
- **The Q4 file is complete and verified.** `5,027,783,488 B`, sha256
  `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`, on the local disk at
  `/home/lucas/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf`. `target/debug` (9.0 GB of build products) was
  removed to leave 9.9 GB free; the J3 evidence is in its log and a later `cargo test` rebuilds it.
  `run4.sh` re-checks size and sha itself before it reads the file.
- **The second window is unblocked.** `q4-go.md` finds no other full Q4 copy (the 5,027,783,520 B
  `qwen3-8b-base/qwen3-8b-base-q4_k_m.gguf` is a different checkpoint, byte 17 of the tensor count)
  and says a local run is GO under `MemoryMax=3600M`: the GGUF pages are clean and evictable, the
  anonymous set is ~0.72 GiB, and the NVMe reads 1.9 GB/s at steady state, so a 5.03 GB re-read per
  token is 1.5 h at worst. Disk was freed by deleting two verified byte-identical duplicates:
  `nq8b-v2-check/skeleton.gguf` (sha `3927454c…`, kept at `nq8b/skeleton.gguf`) and
  `nq8b-v2-check/model.nqp` (sha `1f4cc4b4…`, kept at `nq8b/model.nqp`). The Q6_K/Q8_0 skeleton was
  also deleted; it is rebuilt by `build_q6k_skeleton.py`. `run4.sh` runs Q4 window 1, Q4 window 2 and
  NQ window 2 on this host, then writes the same-host ratio and the host-reproduction delta against
  the rented host's 7.587145.
- **The KV row was wrong by 2×.** `Carrier::F32` stores native `f32` (`mixer.rs:321`, default), so the
  2k KV read is **603,979,776 B**, not 302,776,320; `APEIRON_F32_CARRIER=0` selects Q32 and doubles it.
  Bytes/token at 2k is **2,137,940,816**, the measured-bandwidth ceiling **9.9–13.8 tok/s**, and 20
  tok/s needs **42.76 GB/s** — above the 42.656 GB/s spec peak. The T1 ceiling table and the paragraph
  under it are corrected.

## Queue

Each job is idempotent: `run.sh` skips a job whose `done/<id>` marker exists. Heavy jobs run one at a
time; builds go through `agent-guard`, measurements through `systemd-run --user --scope -p
MemoryMax=3200M -p MemorySwapMax=0`. A job's log is
`bench/cpu/1794912865-nq8b-overnight-logs/<id>.log`; its exit code and wall time append to
`bench/overnight/status.txt`.

| id | job | command (abbreviated) | expected | decision |
|---|---|---|---|---|
| J1 | fast gate | `fmt --check`, `check-claims.py`, `book-check.py` | 2 min | **green, 4 s** |
| J2 | clippy | `agent-guard run -- cargo clippy … -j 1 -- -D warnings` | 15 min | **green, 16 s** |
| J3 | tests | filtered workspace sweep | 45 min | **CPU green, 690 s**; 20 GPU/SPIR-V binaries fail on host causes |
| J4 | exact PPL | conservative skeleton + v4, `--kernel exact`, ctx 2048 | 40–70 min | **11.381891, 1.500155×, 0.0103% over the bar** |
| J5 | phase split | `phase_split … --steps 24 --json` | 10 min | **refused: stale binary**; J5b reruns after the rebuild |
| J6 | head time | `head_time conservative v4` | 10 min | **refused: stale binary**; J6b reruns |
| J7 | bandwidth | `membw` at 1, 2, 4, 8 threads | 5 min | **21.11 GB/s at 4 threads**, 29.47 best-pass |
| J8 | 2k decode | `generate`, ~8 KB prompt, `--n 32`, three runs | 15 min | **refused: stale binary**; J8b reruns |
| J5b–J8c | second driver | rebuild `--release --examples`, then split, target gate, head time, decode 4t/8t | 40 min | `bench/overnight/run2.sh` |
| W* | NQ windows 2–4 | `--skip 2048/4096/6144/8192` | 40 min | Q2 risk screen; break-even Q4 line per window |
| R2–Q4W2 | Q4 windows 1–2 | size+sha, then two PPL runs at 3.6 GiB | 40–120 min | same-host ratio; `bench/overnight/run4.sh` |

There are four drivers: `run.sh` (J1–J8), `run2.sh` (rebuild and the v4 split), `run3.sh` (NQ
windows 2–4) and `run4.sh` (Q4 windows 1–2). They serialize on `bench/overnight/.lock`, so the order
after `run.sh` releases is not fixed; every job is idempotent and the markers record progress.

J9 became the Q4 download instead of a probe: `q4-go.md` finds a local run admissible, so the night
spends the free disk on the file and the two windows rather than on a rented host. Launching a rented
instance remains the owner's decision; `vastai show instances` is 0 and the credit is 2.60.

## Parallel research

Three read-only subagents write their findings while the queue runs:

| file | question |
|---|---|
| `bench/overnight/speed-plan.md` | per-weight instruction counts of the shipped dot cores and the Q6_K head reader; the top three speed levers ranked by the `1/(1−s)` ceiling at the measured split |
| `bench/overnight/q4-go.md` | where the full Q4 file or a rented host is; whether a cgroup run can hold a 5.03 GB mmap at 2k; the exact go/no-go command |
| `bench/overnight/zero-shot-plan.md` | the 0-shot harness (piqa, arc_easy), the length-normalised scoring rule, dataset URLs, and the token cost at 2.25 tok/s |

They do not edit source, delete files, download models or run builds.

## Resume protocol

1. Read this file, then `bench/overnight/status.txt` and the last lines of the newest log.
2. `git status --short`; the tree carries the campaign's uncommitted kernel work. Do not reset it.
3. If the queue stopped on a red gate, fix the named failure first; the queue's `done/` markers are
   the progress record.
4. If the queue finished, read K1–K6 (split, target gate, head time, decode scaling, exact decode),
   then update `GOAL.md` T1 and the depth receipt with the measured rows. Choose the speed target from
   `speed-plan.md` using the split's dominant term, and only then write a kernel.
5. The Q2 claim stays **partial** until a second window and 0-shot exist, whatever the PPL margin.
   The window-2 ratio is in `status.txt` as `Q2 samehost window2 …`; the window-1 host-reproduction
   delta is `Q2 host-reproduction …`. If the local Q4 window 1 does not reproduce 7.587145 within
   ~0.5%, window 1 is restated on this host's own Q4 run and window 2 is the only same-host ratio.
6. If the split names `untraced` as dominant, instrument before choosing a kernel (`--target
   untraced` is the gate). If it names `Ffn` or `AttnProj*`, the lever is the sign-dot stream; if it
   names `Head`, the lever is the Q6_K 2-row tile. `pool.wait` is an overlay: a large value means
   the work is cut too fine, not that the kernel is slow.
7. Do not reopen the ternary activation. It is refuted at 77,741,390 PPL against 72.988013.

## What this file does not cover

The 0-shot half of Q2; a second PPL window; the 1.15× stretch; the 7B checkpoint; the `serve/` path;
C1/C2 at 200k; and every number above the measured bandwidth. J7 replaces the bandwidth row if it
disagrees; the ceiling is recomputed, not restated.

## Rented host for the two Q2 halves

The 0-shot screen is ~10 h for both arms on this laptop even after the shared-prompt lever, and the
25-window PPL line is another ~7 h. Both are measurement campaigns, not kernel work, so they run on
a rented host per AGENTS.md ("a verification campaign too slow for this laptop runs on a rented
host").

| field | value |
|---|---|
| instance | `61104165`, label `q2-screen`, created 04:33 local |
| offer | `60749449`, AMD EPYC 7551, 64 effective threads (cgroup quota 61.4 CPUs), 62 GB, 64 GB disk, France, $0.137/h |
| account | credit $2.60, no other instance |
| bar file | downloaded on the host from `Qwen/Qwen3-8B-GGUF`, 5,027,783,488 B, sha `d98cdcbd…5785` — matches the laptop copy |
| inputs | 1.8 GB tar over ssh: `model.nqp` (sha `271fd5a7…0fea3`), `skeleton-q6k-q4kembed.gguf` (`4a1978ce…c59b`), tokenizer, wikitext-2 test, six 200-item files, source at the current uncommitted state (including the new shared-prompt scorer and `--no-share`) |

Program, in order, all logs under `/workspace/results`: (1) the shared-prompt scorer must reproduce
the joint path bit for bit on five piqa items and on one item of each other task; a mismatch sets
`--no-share` for the screen and is recorded; (2) the six-task screen on both arms, 200 items each,
`tools/quality-compare.py` verdict in `compare.txt`; (3) all 25 PPL windows on this one host, pooled
and text-weighted, in `ppl-all.txt` and `ppl-weighted.txt`. The laptop's Q4 window at skip 139776
(mean nll 2.047649, ratio 1.202) is the cross-host check for that pair. Projected wall 2–3 h,
$0.30–0.45 of the credit; the instance is destroyed after the results are copied back.
