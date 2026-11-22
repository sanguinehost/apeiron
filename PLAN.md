# PLAN — 20 tok/s on the L13, at context up to 1M

This file tracks state. It is not a receipt. Every number names the artifact behind it; criteria are
`GOAL.md`'s. The working pair is **Qwen3-0.6B** (proof) then **Qwen3-8B** (product). Serving stays
CPU-only on the ThinkPad L13 Yoga 20R6 (i5-10210U, 4 cores / 8 threads, 6.6 GiB); no GPU path is
planned, built or measured here. Mobile and training from scratch are later horizons.

The rate target is **1.00 b/w**, with **0.80 b/w** as the sub-1-bit point — the two rows NanoQuant
publishes. The target is **20 tok/s at 2k and at 1M**, decode, with context state near-flat. `GOAL.md` T1's
30 tok/s at 2k and 24 at 200k stay the formal thresholds; this plan reaches 20 first and states
which of T1's rows the result closes.

## 1. The target is 50 ms

| term | 0.6B | 8B | source |
|---|---:|---:|---|
| pack, all linears | 53.8 MB | 1,023.5 MB | `nq::open`; pack parse |
| head, shipped | 165.3 MB Q8_0 | 510.5 MB Q6_K | GGUF tensor table |
| head, NQ at 1.00 b/w | 19.3 MB (rank 992) | 77.4 MB (rank 4,032) | arithmetic |
| fixed: norm, residual, embed, untraced | 1.1 ms | 3.7 ms | `1795170568`, `1794919269` |
| KV per position, F32 | 229,376 B | 294,912 B | `KvState` geometry |
| KV per position, int8 + scale | 59,136 B | 76,032 B | `L·8·(2d+8)` |
| KV per position, 4 bit + scale | 30,464 B | 39,168 B | same |
| native context | 32,768 | 40,960 | GGUF metadata |

20 tok/s = **50.0 ms/token**. The 8B weight term with the head in the pack is 1,115.2 MB =
**38.7 ms at the 28.8 GB/s four-thread mean**, and 29.4 ms at the 37.93 GB/s best pass. After the
3.7 ms fixed term the 8B has **7.6 ms of context at the mean** (11.3 at the best pass). The 0.6B
weight term after S2 is 76.8 MB = 2.7 ms at the mean; its measured pack reader runs at 3.2 GB/s, so
its non-KV work is **26 ms at the empty prefix and 38 ms at 1,827** (the KV scan evicts the pack and
head).

The context budget is a **position count**:

| configuration | KV carrier | positions at 20 tok/s, mean | at 24 tok/s, 37.9 GB/s |
|---|---|---:|---:|
| 8B, head in pack + int8 dot | int8 | ~2,900 | ~4,300 |
| 8B, head in pack + int8 dot | 4 bit | ~5,600 | ~8,300 |
| 0.6B, shipped head | int8 | ~11,700 | — |
| 0.6B, shipped head | 4 bit | ~22,700 | — |

Context lengths below are **1M = 1,048,576 positions** and **200k = 200,000 positions**.

At 2k the 8B reads its whole 155.7 MB int8 KV (5.4 ms); the 0.6B reads 108 MB (4.9 ms). At 1M the
support is read from disk, so the same milliseconds buy less: at 3 GB/s and 4 bit, the 0.6B's ~32 ms
is ~96 MB, about **3,100 positions**, and the 8B's ~7.6 ms is ~23 MB, about **580 positions**. A
512 MB block cache at a 50% hit rate roughly doubles those (fold's measured GPU carryover is 63.5%;
our own hit rate is a stage gate). The 0.6B's 11,700 assumes the empty-prefix non-KV time; at 1,827
the same terms are 38 ms and the budget is ~6,000 until S1 measures the post-fix split. This is the
whole design: **context beyond the hot ring is retrieval, and the block cache is what makes 1M
reachable.**

## 2. Where the time goes now

### 0.6B, measured (`1795170568`)

| prefix | decode | ms/token | RSS | dominant term |
|---:|---:|---:|---:|---|
| 5 | 38.5 tok/s | 25.9 | — | head 36.0% |
| 1,827 | **13.5 tok/s** | **74.8** (32-token mean) | 860 MB | **attn.core 49.8%** |

At 1,827: pack + Q8_0 head + F32 KV = 638.2 MB unique, 1,057.3 MB issued (the two query heads per
KV head read each row twice). `attn.core` is 38.11 ms for 419.07 MB unique / 838.14 MB issued =
11.0 / 22.0 GB/s. The kernel is at 76% of the streaming mean on issued bytes; the NQ pack reader is
the slow term at ~3.2 GB/s.

### 8B, measured (`1794919269`)

| point | decode | ms/token | bytes/token |
|---|---:|---:|---:|
| ctx 64, traced split | 3.5 tok/s | 285.3 | 1,534 MB |
| ctx 2,048, three runs | **2.884 tok/s** (2.857–2.908) | **346.8** | 2,138 MB (1,534 weights + 604 F32 KV) |

At ctx 64: `ffn` 56.4%, `attn.proj_in` 16.7%, head 15.5%, `attn.proj_out` 6.8%, `attn.core` 3.0%.
At 2k the position-dependent term is ~62 ms (346.8 − 285.3), so `ffn` is still **46%** and
`attn.core` ~18%. The 2k split is a `model` decomposition until `phase_split --prefix 2048` runs on
the 8B.

The NQ projections read 1,023.5 MB in 227.7 ms at ctx 64 = **4.5 GB/s**, against the head's 19.2 and
the 28.8 mean. That is the instruction-bound sign dot, not the DRAM wall.

## 3. The architecture

```text
token ── embed ── 28/36 layers ── final norm ── head (NQ section) ── logits
                     │
                     ├─ projections: NQ pack, one loop, k columns
                     └─ attention:
                          hot ring      2k positions, int8/4-bit, kv-head-major
                          sealed blocks 256 positions, record = K|V|codes|scales
                          cold file     one record per (layer, kv-head, block)
                          block cache   LRU over gathered records, bounded in RAM
                          selector      two-level mean-key tree per (layer, kv-head)
                          read          exact softmax over hot ∪ selected, int8 arithmetic
```

Five decisions carry the design:

1. **KV carrier is a dial, not a second path.** One carrier loop reads F32, int8 or 4-bit codes
   through a per-`(kv-head, position)` scale. F32 stays the oracle. int8 is the 2k carrier;
   4 bit is the 1M carrier because disk and gather, not RAM, bind there.
   `APEIRON_F32_CARRIER=0` keeps the Q32 exact path for the proof and the certificate.
2. **GQA is read once.** A task owns one `(layer, kv-head)` and computes all `G = n_head/n_head_kv`
   query heads from one K row and one V row. Today the 0.6B issues 2× and the 8B 4× the unique
   bytes; this removes the multiplier and makes the scan contiguous within a kv head.
3. **Context beyond the hot ring is retrieval.** Blocks of 256 positions are sealed once they leave
   the hot ring. A two-level tree over per-block key means picks `keep` blocks per
   `(layer, kv-head)`; block 0 (sink) and the newest block ride unconditionally. Selection is
   approximate and measured; the softmax over the gathered support is exact for that support.
4. **Weights are read at most once per token.** The head joins the pack as a reserved section
   (rank 4,702 for the 8B, 1,200 for the 0.6B). The NQ dot gets a single int8 activation plane.
   When more margin is needed, the k-column verify folds k activation columns per weight read; the
   emitted tokens stay exactly greedy, so quality is unchanged by construction.
5. **Prefill is an indexing job.** A 1M prefill is ~10 h at 0.6B (29.4 tok/s measured) and ~66 h at
   8B (4.18 tok/s at 2k) before the O(n²) attention term; it is run once (overnight or on the rented
   box), persisted as the cold file plus index, and reused. Incremental turns use a block-tiled
   kernel that loads each selected K/V block once and updates all query rows, which is how T2's
   1,000-token turn stays inside 60 s.

C1, as arithmetic: at 200k, 8B int8, the hot ring is 155.7 MB, the index at block 256 is ~29 MB and
a 512 MB block cache fits; **resident state is ~0.7 GiB**, under 1 GiB. The cold store itself is
7.8 GB on disk at 4 bit. C1's wording counts the cold store inside the 1 GiB; §6 proposes the
amendment, because no 200k 8B KV store of any carrier ≥0.5 bit/value fits 1 GiB.

## 4. Stages, 0.6B first

Each stage names its criterion, the files it touches, the command that decides it, and the gate.
A stage that misses its gate stops the chain; the fix is the stage, not the next one.

### S0 — instrument (this change)

`phase_split --prefix N` appends N untraced tokens before the traced decode. Receipt
`1795170568`. Files: `examples/phase_split.rs`. Gate: prefix 0 and 1,827 runs print their prefix and
the dominant term. **Done.**

### S1 — int8 KV, kv-head-major, GQA group read (0.6B)

Files: `src/engine/attn.rs`, `src/engine/attn/i8.rs` (new), `src/engine/step/softmax.rs`,
`examples/ppl.rs`, tests. A `KvCarrier` dial `Q32 | F32 | I8`; the I8 plane stores
`(layer, kv_head, position, dim)` signed codes plus one binary32 scale per
`(layer, kv_head, position, K|V)`. The attention task becomes `(layer, kv_head)` and writes G
query-head outputs; under I8 one pass reads each K and V code row once for the whole group. The I8
scalar body and its AVX2 body are bit-identical; the Q32 and F32 arms keep the per-head call.
Selector: `APEIRON_KV_CARRIER=q32|f32|i8`, falling back to `APEIRON_F32_CARRIER` (`0` → Q32). The
block certificate stays Q32-only.

Measured at 1,827 on 4 pinned cores under a concurrent `node` process, three runs per arm: F32 9.61
tok/s → I8 14.75 tok/s (1.54×); `attn.core` 58.71 → 21.76 ms (2.70×); token 102.34 → 66.08 ms. PPL
at ctx 2,048 (2,047 positions): F32 571.89, I8 515.80 (0.902×); teacher-forced top-1 agreement
1,620/2,047 (79.1%); peak RSS 783.5 → 450.9 MiB. Receipt `1795177681`. The absolute is the 0.6B
pack's, not the carrier's: it is 54.807 at ctx 512/8,176 and 74.297 at ctx 512/2,044
(`1794822341`), rising to 111.067 at ctx 1,024 and 571.891 at ctx 2,048 (`1795202555`), and
`fd6ad4b0` returns the same two rows. The pack is the 0.977 b/w end of the measured dial; its PPL
is the quality at that rate, not a separate approach. The 8B family is 1.124–3.131 b/w,
1.011–1.607× Q4, the 1.18 points 1.499–1.555× (`1794896874`, `1794893449`). The dial is monotone in
rate; the 0.6B's long-window degradation is the mechanism to fix at that setting.

Commands:

```sh
APEIRON_KV_CARRIER=f32 generate --model <0.6B skeleton> --pack <nq06-v3> --prompt "$(cat <2k prompt>)" --n 32
APEIRON_KV_CARRIER=i8  generate --model <0.6B skeleton> --pack <nq06-v3> --prompt "$(cat <2k prompt>)" --n 32
APEIRON_KV_CARRIER=i8  ppl --model <0.6B skeleton> --pack <nq06-v3> --tokenizer <tok> --data <text> --tokens 2048 --ctx 2048
```

Gate: **≥20 tok/s mean of three runs at 2k** (open: 14.75 tok/s under the concurrent process; the
same split scaled to the clean F32 baseline ratio is ~19, short of 20), PPL at ctx 2,048 within
**1.01×** the F32 carrier (pass, 0.902×; a carrier gate on a fixed pack, not a quality result,
`1795202555`), and top-1 agreement over 64 tokens at 2k recorded
(79.1% over 2,047 teacher-forced positions). Receipt advances T1; C1 stays open. The 20 tok/s gate
moves to S2: at the traced 66.08 ms token the 18.7 ms head leaves 47.38 ms = 21.1 tok/s before the
head's pack read and ~49.1 ms = 20.4 after it at the 67.8 ms rate mean, so S2 is marginal against 20
on this host.

### S2 — head into the pack (0.6B, local rehearsal)

Files: `src/format/nq.rs` (reserved head section), `src/engine/loader/nq.rs`, `src/engine/step.rs`
(head slot), `examples/nq_fit.rs`. Rank 1,200 at 1.18 b/w, 23.0 MB, fitted against the final hidden
state with the Gram diagonal.

Gate: head bytes ≤ 25 MB, PPL within 1.01× the Q8_0 head on the same pack, head phase ≤ 2 ms at
2k. The PPL gate is a head comparison on the same pack; the 0.6B onepath trunk point is
unmeasured, so S2 does not establish product quality. Expected token
~49.1 ms = 20.4 tok/s from the 67.8 ms S1 rate mean under the same contention, and 50.6–51.1 ms =
19.6–19.8 tok/s once the head's own pack read is counted: the gate is marginal, 0.4 ms of predicted
margin against a 1.9 ms spread, and the `ffn` (14.29 ms, 21.6%) is the next term if it falls short.
Receipt advances Q1, T1. This is the rehearsal for the 8B S6.

### S3 — selection at 32k (0.6B)

Files: `src/engine/attn/select.rs` (new; the certificate in `cert.rs` stays exact and Q32-only),
`src/engine/attn.rs`, `src/engine/step/`. Block seal at 256; per-block mean key in int8 with a
per-block scale; exact softmax on the support; a dense-int8 reference behind a flag. No cold file
yet: 32k int8 KV is 1.94 GB and fits.

Gate: **≥20 tok/s at 32k**, anonymous RSS ≤ 2 GiB, NIAH at 32k 100% over 10 depths, PPL within
1.02× the dense int8 read at 32k. Expected: dense is 1.94 GB = 67 ms at 28.8 (14 tok/s before the
fixed term); a 4k-position support with block 256 is 242 MB = 8.4 ms. Receipt advances C1, C2.

### S4 — int8 activation dot (shared kernel)

Files: `src/engine/kern/nq.rs`, the projection call sites. One byte plane on the activation, 8
instructions per 32 signs versus the four-plane 24. The scalar twin and the equality test over the
boundary lengths stay. This is the 8B's dominant-term change; it runs here because the 0.6B's 1M
budget needs the pack term down.

Gate: `phase_split --target ffn` exits 0 on the unmodified 8B tree first; after, `ffn` at ≥20 GB/s
and the 8B projection total ≤ half of 227.7 ms. Receipt advances T1.

### S5 — cold file and 1M (0.6B)

Files: `src/engine/step/cold_store.rs` (wire it), `src/engine/attn/cold.rs` (new),
`src/engine/attn.rs`, `src/engine/step/softmax.rs`. Sparse file per model run, one `write_all_at`
per sealed record, one `pread` per gathered block, `MADV_DONTNEED` on the released hot region,
tmpfs refused. Two-level mean-key tree, 512 MB block cache.

Gate: **≥20 tok/s at 200k and 1M**, resident state ≤ 1 GiB (hot + index + cache), block-cache hit
≥50%, NIAH at 1M ≥90% over 10 depths, throughput at 1M within 0.8× of 2k (T1's ratio). Expected at
1M with 4-bit KV: pack 11 ms (S4), head 1.6 (S2), fixed 1.1, hot 2.2, selector 2, cold read ~21 ms
at 3 GB/s with keep 16 and a 50% cache = 39 ms (26 tok/s). Receipt advances C1, C2, T1's 1M row.

The 1M configuration needs 31.9 GB of disk at 4 bit; `/` has 67 GB free and `/tmp` is tmpfs, so the
tier goes under the model directory, never `/tmp`.

### S6 — head into the pack (8B, rented fit)

Same format and loader as S2. 151,936×4,096, rank 4,702, Gram-diagonal preconditioner
(`nq_fit --diag-dir`).

Gate: head bytes ≤ 95 MB, PPL within 1.01× the Q6_K head, head phase ≤ 4 ms. Receipt advances Q1,
T1.

### S7 — 8B at 2k

Gate: **≥20 tok/s mean of three runs at 2k**, with S1's KV, S4's dot and S6's head. The byte budget is
1,115.2 MB + 155.7 MB int8 KV + 3.7 ms fixed = 47.8 ms at 28.8; at 25 GB/s it is 54.5 ms (18.3). A
4-bit KV lowers the reader requirement to 25.8 GB/s (19.4 at 25); the margin is S9's k-column
verify, not a larger cap. Receipt advances T1's 2k row.

### S8 — 8B selection and cold tier (200k, 1M)

Reuse S3/S5 with the 8B geometry. At 200k, 4-bit KV is 7.8 GB on disk; at 1M, 41.1 GB. The 1M read
of 1,024 kept positions is 40 MB, 13 ms at 3 GB/s; the block cache carries the repeated blocks.

Gate: ≥20 tok/s at 200k, ≥20 at 1M with S9, resident ≤1 GiB, NIAH ≥90% at 200k and 1M. T1's 24 at
200k needs the 37.93 GB/s pass and ~3–6k positions; report it, do not assume it.

### S9 — k-column verify (margin)

A restricted-head 0.6B draft (shortlist ~4k head rows, ~4 ms/token) drafts k = 2; the 8B verifies
both columns in one pass over the pack and the selected KV. Verify is the same loop with two
activation columns. Output is greedy-exact by the existing speculative rule
(`src/engine/step/spec.rs`).

Gate: acceptance ≥0.7, ≥24 tok/s at 2k and ≥20 at 1M, top-1 agreement 100% with greedy. Receipt
advances T1.

### S10 — serve and Intent

`serve/` gets the cold tier and the pack; `../intent` drives a real coding turn; I1/I2 receipts.
This is the only stage that touches the listener.

## 5. What the ceilings refuse

- **30 tok/s at 2k on the 8B** needs 35.9–38.1 GB/s (84–89% of the 42.656 GB/s spec peak) before the
  fixed term, against a 28.8 GB/s measured mean. It is not a target this plan can promise; 20 is.
- **24 tok/s at 200k at the mean** is arithmetically impossible: weights alone are 38.7 ms + 3.7 ms
  fixed = 42.4 ms > 41.7 ms. It needs the best-pass 37.9 GB/s, or S9 to halve the weight term.
- **Dense 1M** is 31.9/41.1 GB at 4 bit and 62.0/79.7 GB at int8 (Mi positions); it is a disk tier,
  never a RAM state, and the read is bounded by selection.
- **1M prefill** is ~10 h (0.6B, 29.4 tok/s measured) and ~66 h (8B, 4.18 tok/s at 2k) before the
  O(n²) attention term; it is an offline index build, not a turn. T2's 1,000-token turn is the
  separate gate.
- **1M quality** is retrieval quality at positions far beyond the model's native 32,768 / 40,960.
  C2 at 1M is NIAH plus agreement against a dense reference run on the rented box, not PPL parity
  at the native context. The RoPE guard refuses a cap above `context_length` today; raising it is
  part of S5 and is labeled extrapolation.
- **Disk**: 67 GB free; the 0.6B 1M 4-bit store is 31.9 GB, the 8B 41.1 GB. One model at a time on
  this laptop, or an external SSD.

## 6. Decisions

- **C1 counting (proposed).** Count resident state (hot ring, index, block cache) against 1 GiB and
  give the cold store its own disk budget. The literal wording includes the cold store; at 200k
  that is 7.8 GB (8B, 4 bit), so the literal form cannot be met by any carrier at ≥0.5 bit/value.
  Default until the owner amends: report both rows, resident and disk.
- **Working model.** Qwen3-8B is the measured product pair; `GOAL.md`'s Qwen2.5-7B primary is
  unchanged. Q1/Q2 are model-agnostic; the pack rate is 1.178785 b/w over the 6.946 G replaced
  weights (1.000 b/w over all 8.19 G).
- **Carrier order.** int8 first for the 2k proof; 4 bit for the 1M cold store; 2 bit only if a
  measured quality gate passes.
- **Rented box.** The 8B head fit, the dense 32k/200k reference runs and the 1M index builds run
  there; serving stays on the laptop.
- **No GPU.** The Vulkan criteria V1–V4 stay open but are not on this path.

## 7. Next actions

1. Land S1: the int8 KV carrier and the GQA group read on the 0.6B; the gate is 2k ≥20 tok/s and
   the 1.01× PPL bound.
2. In parallel, run `phase_split --prefix 2048` on the 8B to replace the ~62 ms attention delta
   with a measured split.
3. S2's head fit on the 0.6B, then S3's selection at 32k, then S5's cold file at 200k and 1M.
4. S4's shared int8 dot before the 1M budget is final; S6/S7 on the 8B after the 0.6B method is
   proven.
5. Run the quality campaign as one command, `tools/nq_campaign.sh` (statistics → dump → block loop
   → model KD → pack → PPL). The inputs are the reference's: 128 samples of 2048 tokens drawn
   from wikitext2 **train** (the builder tokenizes the joined split once, slices, shuffles and
   draws with replacement; `wiki.train.raw` is under `~/models/wikitext2/wikitext-2-raw/`), the
   `s128` diagonal and output-gradient statistics with the `online` clip, and 2048-token blocks.
   The wall gate is `tools/fit-budget.py`; the measured 0.6B layer-0
   seven linears are 1,134.5 s at T=400 on the AVX-VNNI box (`1794834689`), 8.8 h over 28 layers.
   The block loop and the KD epochs need a Vulkan host with cooperative matrices; the Vulkan path
   refuses on this laptop and the rented host is unpaid (balance −$0.47), so the campaign is
   blocked on credit. The stages are in the tree; the block loop and the KD epochs have not run
   end to end on any host, so the campaign's first execution is also their first validation.

The one-line version: **the 0.6B at 1,827 is 49.8% attention; the 8B at 2k is 46% ffn and 18%
attention; 20 tok/s is a 2.9k-position RAM budget at 8B and 11.7k at 0.6B, cut to ~0.6–3.1k by the
disk at 1M; the cold tier, the block cache and selection are how 1M fits, and the 8B needs the head
and the int8 dot before any of it shows.**
