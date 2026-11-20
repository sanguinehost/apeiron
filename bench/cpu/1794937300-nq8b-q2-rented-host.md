# Q2 on a rented host: the 25-window PPL line and the 0-shot screen

Criterion **Q2**. Grade `measured` for the host, the file hashes, the scorer cross-check and the
cross-host reproduction; the two verdicts are filled from `/workspace/results` when the campaign
finishes.

## Why a rented host

The 0-shot screen is ~10 h for both arms on the laptop after the shared-prompt lever, and the
25-window PPL line is another ~7 h. Both are measurement campaigns over the same binary, not kernel
work, so they run on a rented host per AGENTS.md. The laptop keeps the 17 pairs it completed; this
campaign is the clean same-host 25-window line and the first 0-shot measurement.

## Host and inputs

| field | value |
|---|---|
| instance | `61104165`, label `q2-screen`, offer `60749449` |
| CPU | AMD EPYC 7551, 64 effective threads (`cpu.max` 6143999/100000 = 61.4 CPUs), 62 GB, 64 GB disk |
| price | $0.137/h, account credit $2.60, no other instance |
| kernel path | AVX2, no AVX-VNNI — the same dot core the laptop selects |

| file | bytes | sha256 |
|---|---:|---|
| `Qwen3-8B-Q4_K_M.gguf` (downloaded on the host from `Qwen/Qwen3-8B-GGUF`) | 5,027,783,488 | `d98cdcbd…c5785` |
| `model.nqp` (uploaded) | 1,023,456,080 | `271fd5a7…0dfea3` |
| `skeleton-q6k-q4kembed.gguf` (uploaded) | 867,740,256 | `4a1978ce…eec59b` |
| inputs tar | 1,907,722,240 | size checked on the host before extraction |

The tar carries the source at the campaign's uncommitted state, including `examples/quality.rs`
with the shared-prompt scorer and the `--no-share` switch.

## The scorer cross-check

`examples/quality.rs` prefills a choice's prompt once per item and rewinds the keys and values to the
choice boundary; `--no-share` runs the original one-prefill-per-choice path. The two must agree bit
for bit, and the joint path must reproduce the laptop's five-item smoke.

```text
shared_vs_joint True
joint_vs_laptop True
shared [[-38.807668498, -44.04348531], [-38.185021003, -41.53102617],
        [-217.289790616, -123.467113794], [-38.081035505, -34.603766456],
        [-83.503242798, -74.466753838]]
joint  (identical)
laptop (identical)
task boolq shared_vs_joint True [[-4.08425508, -3.03544004]]
```

The first item of each remaining task is checked the same way; the screen runs with the shared path
only if every task agrees. The joint path reproducing the laptop's values on an EPYC 7551 is also
the cross-host control: the integer projections and the stable pool order make `ll` a property of
the model and the text, not the host.

## Protocol

Identical to the laptop runs: `--tokens 2048 --ctx 2048 --head-batch 16`, `--second-half` for PPL,
`--kernel q8act`; `APEIRON_THREADS=60`. PPL windows: skips 0, 2048, 4096, 6144, 8192 and 20
systematic windows spaced 11,776 tokens from 10,240 to 233,984. The stratified estimate weights the
five base windows at 2,048 tokens and each systematic window at 11,776.

## PPL

The laptop's 17 pairs (skips 0…139,776) pool with `--spans 2048×5 + 11776×12` to **1.417631**,
95 % **[1.355945, 1.482123]**, gap 0.348987 ± 0.044489 nats (df 16, effective n 13.6). The upper
end is below 1.5, so the PPL half is **established on the measured span**: skips 0–139,776 are
141,824 of the 245,568 test tokens, 58 % of the file.

| window set | pooled ratio | 95 % interval | verdict |
|---|---:|---|---|
| five base windows (2,048 each) | 1.499754 | [1.391674, 1.616228] | undecided |
| 17 windows, weighted | **1.417631** | **[1.355945, 1.482123]** | under the bar |

The eight tail windows (skips 151,552…233,984) were **stopped** after two NQ windows had run. The
pack is the artifact under edit, so a 25-window line on this pack would be superseded by the next
fit before it could be cited; the measured statement is the 17-window span, 58 % of the file. The
protocol and the bar arm's per-window values are the durable part: a refit re-measures the NQ arm on
the same 17 windows, not a new design.

The two hosts are interchangeable at this protocol: host A reproduced the laptop's NQ window 1
(`11.374839`), NQ window 2 (`11.834074`) and Q4 window 1 (`7.587145`) to the last digit.

## 0-shot screen

**Verdict: refute.** Six tasks, 100 items each (the systematic half of the 200-item lm-eval-faithful
files), both arms, paired items, lm-eval primary metric per task:

| task | metric | NQ | Q4_K_M | ratio |
|---|---|---:|---:|---:|
| boolq | acc | 0.6300 | 0.8800 | 0.7159 |
| piqa | acc_norm | 0.7100 | 0.7700 | 0.9221 |
| hellaswag | acc_norm | 0.5700 | 0.7400 | 0.7703 |
| winogrande | acc | 0.4900 | 0.7600 | 0.6447 |
| arc_easy | acc_norm | 0.5800 | 0.7900 | 0.7342 |
| arc_challenge | acc_norm | 0.3800 | 0.5200 | 0.7308 |
| **mean** | | **0.5600** | **0.7433** | **0.7534** |

Paired difference over the six tasks: **−0.1833 ± 0.0812** (95 % t, df 5, t 2.571). The allowance is
−0.0743, so the interval [−0.2645, −0.1021] is entirely below it; `tools/quality-compare.py` exits 1.
Only piqa clears 0.90 (0.9221). Winogrande is at chance (0.49 on two choices) and the four knowledge
tasks sit at 0.64–0.77.

**PPL does not predict this.** The same pack is 1.4176× on wikitext-2 (established) and 0.7534× on the
six-task mean. A PPL ratio of 1.42 would have been read as a mild loss; the 0-shot half shows a
one-quarter loss of task accuracy, concentrated in the tasks that need stored knowledge rather than
local fluency. The 1.178785 b/w allocation is therefore not a Q2 candidate whatever the fit does to
PPL: the next pack has to be judged on the 0-shot screen, and the rate has to move (or the allocation
has to protect the knowledge-bearing linears) for the 0.90 bar.

The Q4 denominators above are pack-independent and are the reference for the next NQ arm: re-running
the candidate costs one arm, ~45 min on a 40-CPU box, with the same 100-item files.

## Artifacts

`bench/cpu/1794937300-nq8b-q2-rented-host-logs/`: the twelve per-item JSONL files (600 items per arm),
`compare.txt`, the Q4 sha and the smoke diffs. The scorer validation is bit-for-bit:
`shared_vs_joint True`, `joint_vs_laptop True` on five piqa items and on one item of each other task.

## What this receipt does not cover

The 0-shot screen is 100 items per task, 600 of the 19,965 split items; the interval is a t interval
over six task means (df 5) and a different 100-item subsample would move each task by a few points.
The PPL line is 17 windows, 58 % of the test file. The 7B checkpoint, decode speed and every
position of wikitext-2 are outside the claim. rustc and LLVM codegen, the kernel ABI and the
`unsafe` container are outside it too.
