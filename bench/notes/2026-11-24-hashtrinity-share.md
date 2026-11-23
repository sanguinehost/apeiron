# The table's share of the wall

Criterion: Q3 (fit cost) and T2 (turn latency). The rule is wall seconds: a component that is
faster is cheaper in exact proportion to the share of the phase it owns. The share is the whole
question; this note measures it in the three places the table could sit.

Grades: `measured` for the phase splits and the prefill/decode rates (logs named below), `model` for
every turn projection (the hit rate is a usage assumption, not a run). No part of Q3's PPL bar is
advanced.

## The three paths

| path | the table's share | why |
|---|---:|---|
| 8B fit, 4,587.4 s/layer | **0%** | `Table` is constructed in exactly two files, `src/engine/prefix/mod.rs:227,258` and `src/engine/step/cold_store.rs:77,98`. The campaign trainer (`src/gpu/nn/train/`) and `src/learn/nanoquant.rs` contain no `Table`, no `FreqMap`; `examples/nq_fit.rs` links neither. |
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

Measured 8B, 2,048-token prompt, four threads (`bench/cpu/1794919269-nq8b-split.md`, K4 log):

| term | measured |
|---|---:|
| prefill, 2,048 positions | 489.754 s (0.239 s/position, 4.18 tok/s) |
| decode, 32 tokens | 11.1 s (2.884 tok/s) |
| turn | 500.9 s, **97.8% prefill** |

At `m = 1,843` (a 90% shared prefix):

| term | model |
|---|---:|
| prefill the remaining 205 positions | 49.0 s |
| decode 32 | 11.1 s |
| turn | 60.1 s |
| ratio | **8.3×**, 88% of the turn removed |

The snapshot that makes it work is the cost: 294,912 B/position at f32 KV (73,728 values × 4 B), so
a 1,843-position prefix is **543 MB** — more than a quarter of the 2 GiB left for agent sessions on
this laptop. The 8B cannot hold a live 2k session (604 MB) and that snapshot at f32 together under
the serving scope. `KvCarrier::I8` is 1 B/value and would make the snapshot 136 MB; it is a carrier
change with its own quality gate, unmeasured here.

## The fit's lever, which is not the table

The layer is ADMM 39.7%, `tune_nonfact` 28.3%, `tune_fact` 29.8%
(`bench/notes/2026-11-23-admm-split.md`). The cost model's joint seven-slot tune removes the
frozen-slot rematerialisation and prices the tune term at 64.6 s against 2,662.4 s
(`bench/notes/2026-11-24-quant-cost-argon.md`, `model`). The resulting rows are 6.24 h, 5.38 h with
a 14× `inv` (3-limb Cholesky), and 2.04 h with the BF16 fused ADMM. That is the fit's route under
the 2 h gate; a table is not in any of those terms.

## What this does not cover

- The cache has six unit tests on a synthetic GGUF, including warm-vs-cold bit equality at the
  logits checksum. The new batched-tail equality test (`the_batched_tail_equals_the_per_token_path`)
  cannot be built on this laptop: the lib test target exceeds the guard's derived cap. It is
  committed for CI; until it runs, the batched tail is `model`, not `tested`.
- No 8B run of the wired server. A 2k 8B snapshot is 604 MB, above the 256 MiB default budget, so
  the served 8B win needs `--prefix-mb 700` (or the I8 carrier); the 0.6B fits at 256 MiB.
- The 90% hit rate is a coding-agent usage assumption, `model`. A session with no shared prefix
  gets 0% of the win.
- The 543 MB snapshot is f32. The I8 figure is arithmetic on the carrier width, not a measurement.
- The 2k prefill is 4.18 tok/s on the `NqLinear` path that re-reads the pack per column
  (`src/engine/step/cols.rs:503-507`); that defect is separate, and fixing it lowers the prefill
  term the cache hides.
