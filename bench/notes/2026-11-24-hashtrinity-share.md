# The table's share of the wall

Criterion: **T2** (turn latency); the fit-cost section below measures a share and advances no
criterion. T2 is a 1,000-token incremental turn at 200k in ≤ 60 s; this note measures the mechanism
at 1,827 positions on the 0.6B and does not close T2. The rule is wall seconds: a component that is
faster is cheaper in exact proportion to the share of the phase it owns. The share is the whole
question; this note measures it in the three places the table could sit.

Grades: `measured` for the 0.6B cold/warm/shared walls, the phase splits and the prefill/decode
rates (logs named below), `tested` for the batched tail and the snapshot charge, and `model` for
the 8B turn upper bound (its hit rate is a usage assumption, not a run). No part of Q2 or Q3 is
advanced.

## The three paths

| path | the table's share | why |
|---|---:|---|
| 8B fit, 4,587.4 s/layer | **0%** | `Table` is constructed only by the prefix cache (`src/engine/prefix/mod.rs`) and the cold store (`src/engine/step/cold_store.rs`). The campaign trainer (`src/gpu/nn/train/`) and `src/learn/nanoquant.rs` contain no `Table`, no `FreqMap`; `examples/nq_fit.rs` links neither. |
| 8B decode, 346.8 ms/token | **0%** | `matmat_q8act` reads the mmap'd pack bytes; the weight path touches no table. |
| 8B prefill, 239 ms/position | **the alternative is on the path** | `serve/src/main.rs` prefilled the whole prompt per request and `PrefixCache` had zero call sites. `prefill_with_cache_mode` now wires it: probe the table, solve the missed tail in block passes, step only the last token. |

The fit's 0% is not a container-versus-arithmetic split. `tune` is `o·n²` multiply-adds through
`gemm_q32`; a table performs a key-to-value probe. A probe is not a MAC, so it is not the same
operation done faster — it is a different operation. Put the table where
the fit performs a gather of packed rows and its rate becomes the phase's rate by the same rule.

## Where the table is faster than the alternative, and what that is worth

`PrefixCache` replaces `O(prompt)` prefill with `O(n/B)` probes plus one block pass over the
`n − m − 1` missed positions, where `m` is the shared prefix. It is exact by causality: positions
`< m` depend only on tokens `< m`, and the full-key identity rule confirms the token sequence
behind every hash hit.

### Measured, 0.6B at 1,827 positions

Qwen3-0.6B-Base (`skeleton-q8head.gguf`), `nq06-v3.nqp` (196 linears, 0.977441 b/w), tokenizer
`qwen3-0.6b-base-e8/tokenizer.json`, `APEIRON_THREADS=4`, three runs per row, `--differ 205`.
Raw lines, host and the discarded repeats: `bench/cpu/1795480700-nq06-prefix-solve.md`. The cold
row reproduces the published baseline (29.44 tok/s, spread 28.01–31.95):

| turn | prefill s (3 runs) | mean | prefill tok/s | vs cold | decode tok/s |
|---|---|---:|---:|---:|---:|
| cold, no hit | 63.166 / 65.138 / 69.577 | 65.96 | 27.7 | 1.00× | 14.53 / 14.30 / 8.93 |
| warm, hit 1,824 | 0.268 / 0.287 / 0.416 | 0.324 | 5,640 | **203.8×** | 14.46 / 14.31 / 10.87 |
| shared, hit 1,616, 211-token tail | 11.765 / 12.454 / 13.227 | 12.48 | 146.4 | **5.29×** | 10.81 / 11.03 / 13.53 |

A 16-token turn is `65.96 + 16/12.59 = 67.2 s` cold and `12.48 + 16/12.59 = 13.7 s` shared, **4.9×**.
Decode is unchanged: the cache holds KV, not weights, and `matmat_q8act` reads the pack every token,
so the table is 0% of decode and this work cannot move decode tok/s.

The hit is 1,616/1,827 = 88.5% of the positions and removes 81.1% of the wall. The tail is why it is
not 88.5%: its 211 positions attend to all 1,616 cached keys, so they carry
`1 − (1,616/1,827)² = 21.8%` of the key reads and cost 59.2 ms/position against 36.1 cold.
`attn.core` is 49.8% of a token at 1,827 (`bench/cpu/1795170568-nq06-context-split.md`). At a
304-token prefix the same 208-position tail costs 25.8 ms/position against 25.4 cold (512-token
run: cold 13.00 s, warm hit 496 in 0.498 s = 26.1×, shared hit 304 in 5.36 s = 2.4×; the third
shared run of 11.90 s is an outlier, excluded in the receipt). The remaining cost is the tail's
attention, not its projections.

### Measured 8B, for comparison

2,048-token prompt, four threads (`bench/cpu/1794919269-nq8b-split.md`, K4 log):

| term | measured |
|---|---:|
| prefill, 2,048 positions | 489.754 s (0.239 s/position, 4.18 tok/s) |
| decode, 32 tokens | 11.1 s (2.884 tok/s) |
| turn | 500.9 s, **97.8% prefill** |

At `m = 1,843` (a 90% shared prefix), a *linear* projection from that rate gives 49.0 s and 8.3×
for the turn. That is an upper bound, not the prediction: it prices the 205 tail positions at the
cold average, while the 0.6B measurement shows the tail is attention-bound — 211 positions
attending to 1,616 keys hold the ratio at 5.29×. The 8B tail attends to 1,843 keys, so its ceiling
is the same 1/(1 − 0.90²) ≈ 5.3×, not 8.3×. Grade: `model`.

| term | model (upper bound) |
|---|---:|
| prefill the remaining 205 positions | 49.0 s |
| decode 32 | 11.1 s |
| turn | 60.1 s |
| ratio | **8.3×** linear; **≤5.3×** with the tail's attention |

The snapshot that makes it work is the cost: 294,912 B/position at F32 (73,728 values × 4 B), so a
1,843-position prefix is **543 MB** — more than a quarter of the 2 GiB left for agent sessions on
this laptop. A 2k capacity (2,064 positions) is 580 MiB, so the served 8B needs
`--prefix-mb 700`; the 0.6B at 1,827 charges **424,574,976 B (405 MiB)** at capacity 1,851
(229,376 B/position), measured as `cache ... 424574976 bytes stored`, so the 512 MiB default holds
it. `KvCarrier::I8` is 1 B/value and would make the 8B snapshot 136 MB; it is a carrier change with
its own quality gate, unmeasured here.

The charge was wrong until this turn: `estimate_snapshot_bytes` hardcoded 8 B/value and counted the
occupied length, so the 1,827 entry was charged 838,139,904 B — twice its F32 footprint and without
the spare capacity the clone carries. It now charges `cap × bytes_per_position`, the carrier width
(F32 4 B, Q32 8 B, I8 1 B plus its scales), and
`the_snapshot_charge_follows_the_carrier_and_the_capacity` checks the three widths.

## The fit's lever, which is not the table

The layer is ADMM 39.7%, `tune_nonfact` 28.3%, `tune_fact` 29.8%
(`bench/notes/2026-11-23-admm-split.md`). The cost model's joint seven-slot tune removes the
frozen-slot rematerialisation and prices the tune term at 64.6 s against 2,662.4 s
(`bench/notes/2026-11-24-quant-cost-argon.md`, `model`). The resulting rows are 6.24 h, 5.38 h with
a 14× `inv` (3-limb Cholesky), and 2.04 h with the BF16 fused ADMM. That is the fit's route under
the 2 h gate; a table is not in any of those terms.

## What this does not cover

- The cache has ten unit tests on a synthetic GGUF: the six pre-existing ones plus
  `the_batched_tail_equals_the_per_token_path` (Exact and Q8Act, cold and warm, against a
  per-token oracle) and `the_snapshot_charge_follows_the_carrier_and_the_capacity`. All ten pass:
  `cargo test --lib prefix::tests`. The batched tail is `tested`; the measured runs above are on
  the 0.6B only.
- T2 itself: its target is a 1,000-token incremental turn at 200k in ≤ 60 s. The runs here are a
  0.6B mechanism measurement at 1,827 positions.
- No 8B run of the wired server: the 580 MiB 2k snapshot plus the live state does not fit beside
  the model under the laptop's 2 GiB agent reserve. The 8B 8.3× is a linear upper bound from the
  measured prefill rate; the tail's attention caps it at the same ≈5.3× the 0.6B measured.
- The 90% hit rate is a coding-agent usage assumption, `model`. A session with no shared prefix
  gets 0% of the win. The measured hits are exact-prefix repeats, not a workload.
- Decode is untouched: 8.9–14.5 tok/s across the 1,827 runs (nine values, mean 12.5) against the
  13.51 baseline with its 11.58–14.64 spread. The dent is prefill and turn latency.
- The tabulated runs predate the example's warmup pass and charge fix; the timing path is the same
  code, and the corrected charge is 405 MiB (measured `424574976 bytes stored` in the later run).
  The later repeats (11:35–11:47) fell to 4.1 tok/s decode under the intent workspace's clippy and
  are discarded in the receipt.
- One shared run at 512 positions (11.90 s against 5.23/5.50) is an outlier excluded from the
  mean; the machine carried load average 6–7 during the 1,827 runs.
- The I8 snapshot figure is arithmetic on the carrier width, not a measurement.
- The 2k prefill is 4.18 tok/s on the `NqLinear` path that re-reads the pack per column
  (`src/engine/step/cols.rs:503-507`); that defect is separate, and fixing it lowers the prefill
  term the cache hides.
