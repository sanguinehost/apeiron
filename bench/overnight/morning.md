# Morning brief: the 8B pack sits on the 1.5× bar, 20 tok/s is refused, and the 0-shot harness is ready

Night of 17–18 Nov. Everything below is `measured` on the i5-10210U (4C/8T, 15 W, AVX2/BMI2/POPCNT)
unless marked `model`. Durable state, queue and resume protocol:
`bench/cpu/1794912865-nq8b-overnight.md`; job log of record: `bench/overnight/status.txt`.

## 1. Q2: the PPL half is on the bar, not under it; 0-shot is unmeasured

Pair: `nq8b-late-v4/model.nqp` (1,023,456,080 B, **1.178785 b/w**, 252 linears, sha
`271fd5a7…0dfea3`) on `nq8b/skeleton-q6k-q4kembed.gguf` (867,740,256 B, sha `4a1978ce…eec59b`,
stock **Q6_K** `output.weight` + **Q4_K** `token_embd.weight`). The bar is the full
`Qwen3-8B-Q4_K_M.gguf` (5,027,783,488 B, sha `d98cdcbd…c5785`), now local and verified.

| window | skip | NQ PPL | Q4_K_M PPL | ratio | margin to 1.5× |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 11.374839 | 7.587145 | 1.499226 | +0.005879 (+0.0517%) |
| 2 | 2048 | 17.361910 | 11.834074 | 1.467112 | +0.389201 (+2.1925%) |
| 3 | 4096 | 17.226503 | 10.462647 | **1.646477** | **−1.532533 (−9.7922%)** |
| 4 | 6144 | 14.057420 | 9.354884 | **1.502682** | **−0.025094 (−0.1786%)** |
| 5 | 8192 | 15.236382 | 10.927844 | 1.394272 | +1.155384 (+7.5755%) |
| **pooled** | 0–8192 | **14.876661** | **9.919400** | **1.499754** | **+0.000246 (+0.0164%)** |

**The pooled estimate over 5,115 scored positions is 1.499754 — 0.016% under the bar — with a
window-level 95% interval of [1.391674, 1.616228].** The bar is inside the interval, so the PPL half is
**undecided, sitting on the bar**, not established. Three windows pass and two fail; the per-window
ratio spans 1.394–1.646.

**The earlier two-window pass is withdrawn.** It claimed 1.499225 and 1.467112 as "the PPL half
holds"; those are two of the three windows that pass, and the first two happened to be among them.
The margin claimed on window 1 was 0.005879 PPL = **0.0005 nats**, 1/116 of the window-to-window sd
(0.060247 nats) — two windows cannot resolve it, and neither can 500. A margin that small needs ~55,800
windows (57 M scored positions, ~14,000 h). The route is a fit that moves the gap, not more sampling: at 25 windows the half-width is
±0.025 nats, so the pool must reach **≤0.380 nats** (6.3% less added nll) before a 25-window run can
put the bar outside its interval. `run14` runs windows 6–25 (~5.5 h) for that bound. Receipt
`bench/cpu/1794922136-nq8b-q2-five-windows.md`.

- **Both arms ran on this laptop on one protocol.** The local Q4 window-1 run reproduces the rented
  host's 7.587145 to the last digit (delta 0.000000).
- **The same-quantisation condition is exact.** The skeleton's `token_embd` (Q4_K, 350,060,544 B) and
  `output.weight` (Q6_K, 510,504,960 B) are **byte-identical** to the bar file's — 99.99% of its bytes.
  The 72 `attn_norm`/`ffn_norm` tensors differ because the Q4 file is an *AWQ compatible* conversion
  whose per-channel scales are folded into those norms and undone in its 252 linears, an exact
  reparameterisation. `tools/gguf-tensors.py`, receipt `bench/cpu/1794921128-nq8b-tensor-audit.md`.
  If the AWQ fold makes the Q4 baseline stronger than a plain `Q4_K_M` would be, the ratio is
  conservative; that direction is `model`, not measured.
- The exact reader scores 11.381891 on window 1 (1.500155×) where q8act scores 11.374839; the 0.007052
  PPL reader gap is activation-quantisation error, 1/38 of the pooled interval's half-width.
- The pack's added nll across the five windows is 0.405, 0.383, 0.499, 0.407, 0.332 — it does not
  order with the Q4 arm's nll, so no difficulty model is fitted.
- **The interval is text variance, not run noise.** Three repeats (`run13`) reproduce NQ window 1
  11.374839, Q4 window 1 7.587145 and Q4 window 3 10.462647 to the last digit, delta 0.000000. The
  integer projections and stable pool order make a window's ratio a property of the text.

**Not a Q2 pass yet.** The criterion needs 0-shot ≥0.90× as well, and no 0-shot *measurement* exists.
The inputs and harness do: six 200-item lm-eval-compatible files (`bench/quality/items/`, hash-anchored,
**201,440 prefill + 43,827 head positions**, no candidate over ctx, no boundary refusal) plus
`examples/quality.rs` and `tools/quality-compare.py`. The measured token count is 1.56× below the
earlier model: the smoke measures **0.263 s per prefill position** on the NQ arm and **0.195** on the
Q4 arm, so the six-task screen is **~15 h + ~11 h** on this laptop, not 45.9 h, and roughly 6–7 h on
a 3–4× rented host (~$0.30–0.40 of the 2.60 credit). Upstream's 200-item anchor is 0.671 mean against the
**dense base**, not Q4_K_M, so it is a warning, not the bar. Design, templates and statistics:
`bench/overnight/zero-shot-plan.md`.

**What would buy PPL margin**: not rate. The 1.18 cap over 6,945,767,424 weights is 1,024,500,695 B
against the pack's 1,023,456,080 B — **1,044,615 B headroom**, and one extra plane is 2,097,152 B on
a 4096² linear and 6,291,456 B on 4096×12288. Only a better fit at the same bit count (more T,
different layer/rank allocation, better preconditioner) can move it, and that is a rented-GPU
campaign (`tools/fit-budget.py`), not a laptop night.

## 2. T1: the split is measured, ffn owns the token, 20 tok/s is refused

`phase_split` on the Q2 pair, 24 steps, 4 participants: **285.26 ms/token (3.5 tok/s)** at ctx 64.
With a 2,048-token prompt, three `generate --n 32` runs measure **2.884 tok/s (346.8 ms/token)**,
range 2.857–2.908; the extra 62 ms is attention over the longer KV. Prefill is 0.239 s/position, so a
1,000-token turn is ≈239 s: T2's 60 s is 4.0× refused before any decode.

| phase | ms/token | share | ceiling 1/(1−s) |
|---|---:|---:|---:|
| **ffn** | **160.760** | **56.4%** | **2.29×** |
| attn.proj_in | 47.645 | 16.7% | 1.20× |
| head | 44.292 | 15.5% | 1.18× |
| attn.proj_out | 19.272 | 6.8% | 1.07× |
| attn.core | 8.527 | 3.0% | 1.03× |
| norm / residual / embed | 4.450 | 1.6% | — |
| untraced | 0.311 | 0.1% | — |
| pool.wait (overlay) | 20.440 | 7.2% | — |

The three NQ phases are **227.68 ms, 79.9%**; `--target ffn` exits 0 on the unmodified tree.

- **The shipped head is fast.** `head_time`: exact reader 0.534 s (1.17 G weight/s), shipped `Q8Act`
  **0.036 s (17.32 G weight/s)**. Step exact 1.058 s (0.94 tok/s) vs step q8act **0.280 s (3.58
  tok/s)**. The head is 15.5% of the token, and its own floor is 23 ms at the measured bandwidth, so
  a perfect head kernel is ≤1.08× — not the target. The 92% figure in the earlier plan came from the
  reference reader, not this one.
- **The instruction wall**: 7,623,016,448 sign terms/token at 33.4 G terms/s aggregate (8.36 G/core,
  3.14 G instr/s/core at 12–14 instructions per 32 terms). 20 tok/s needs 190.6 G terms/s —
  **5.70×** at ctx 64, **6.94×** at 2k. A hypothetical 6-instruction selector halves it and lands at
  **5.84 tok/s**, still **3.42× short**.
- **The byte wall at 2k**: pack + Q6_K head + f32 KV is 2,137,940,816 B/token (KV is 603,979,776 B;
  `Carrier::F32` is the default, `mixer.rs:321`). 20 tok/s needs **42.76 GB/s**, above the
  **42.656 GB/s spec peak** and **1.91×** the measured 22.33 GB/s (4 threads, 256 MiB, 100 passes).
  Bytes are not binding today: ctx 64 streams 1.534 GB/token at 5.38 GB/s.

Both walls must move: a head inside the pack (~80 MB) and KV below f32 for bytes; ≈5.5 instructions
per 32-term group, 2.3× below even a perfect 6-instruction selector, for instructions. The ternary
activation reaches that arithmetic and is refuted on quality (77,741,390 PPL vs 72.988013). No kernel
closes 20 tok/s on this pair; the report says so with both margins.

## 3. Gates and correctness

- J1 fast gate green (4 s), J2 clippy green (16 s), J3 CPU tests green (690 s, 20 failing binaries
  all GPU/SPIR-V host causes: no `vkCmdWriteTimestamp2`, no `glslc`).
- J4 exact PPL on the product pair: 11.381891, 2,213.5 s, 0.5 tok/s, peak 2,336.2 MiB.
- Q6_K equality test extended from 2 to all 16 groups of the block (`src/engine/chart/tests.rs`),
  release run queued as `K5`.
- `examples/head_time.rs` now also times the shipped head through `lm_head_cols(…, Q8Act)`.

## 4. Disk and artifacts

- 9.9 GB free. Deleted only sha-verified byte-identical duplicates (`nq8b-v2-check/{skeleton.gguf,
  model.nqp}`) and `target/debug` (9.0 GB of build products; J3's evidence is its log).
- New receipts: `bench/cpu/1794922136-nq8b-q2-five-windows.md` (the five-window pooled Q2 and the
  withdrawn two-window pass), `bench/cpu/1794921128-nq8b-tensor-audit.md` (the baseline audit),
  `bench/cpu/1794919269-nq8b-split.md` (split, head rates, refusals), `bench/cpu/1794912865-nq8b-
  overnight.md` (state of record, corrected at 00:28). Updated: `bench/cpu/1794896874-nq8b-depth.md`
  (superseded scope), `GOAL.md` (Q2 open on five windows, T1 split). Tools:
  `tools/gguf-tensors.py`, `tools/ppl-pooled.py`.
- Plans: `bench/overnight/{speed-plan,q4-go,zero-shot-plan}.md`.
- 0-shot inputs: `bench/quality/items/` (six 200-item files, `SHA256SUMS`), `bench/quality/README.md`,
  `examples/quality.rs`, `tools/quality-items.py`, `tools/quality-compare.py`.
- All work is uncommitted; no commits or pushes.

## 5. What remains

1. **0-shot** — the harness and the six item files exist and compile; the five-item smoke on the Q2
   pair is queued (`run9`). The screen is 201 k prefill + 44 k head positions (13–29 h on this laptop,
   ≈5–7 h on a 3–4× host). This is Q2's last part.
2. **The 25-window bound** (`run14`, running): ±0.025 nats, which quotes the pooled ratio to ±2.5%.
   The criterion needs a fit that moves the gap below 0.380 nats; the 1.15× stretch is far open.
3. **T1** — the dominant phase is ffn; a faster selector is capped at 2.29× and lands at 5.84 tok/s.
   20 tok/s is refused on this pack/head/KV. The honest next speed step is the ffn dot stream (with a
   safe twin and an equality test) or the Q6_K 2-row tile for the remaining 15.5%.
4. **C1/C2 at 200k, T2's 1k turn** (prefill is ~0.27 s/position, ~4× refused), M1's 2 GiB headroom,
   I1/I2.
