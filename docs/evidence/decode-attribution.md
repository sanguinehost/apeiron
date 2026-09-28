# Decode throughput: which change moved it, and what the proofs contributed

Machine: i9-14900KF, 8 P-cores, host read 82.0 GB/s at 8 threads. Command:
`APEIRON_KERNEL=<mode> target/release/examples/engine_decode <gguf> /home/socol/models/qwen2.5-0.5b/tokenizer.json <n>`.
Baselines: `llama-bench -p 0 -n 64 -r 5 -t 8`, run back to back. Grade `measured` unless stated.
Every row keeps `generated_ids` equal to the exact-path reference (`/tmp/opencode/t9/q8act-1.txt` for Qwen,
`/tmp/opencode/idle/bonsai-1.txt` for Bonsai).

## Current standing (load 1.0–2.4)

| model | ours, 5 reps | baseline tg64 | median ratio |
|---|---|---|---|
| `qwen05b-Q4_K_M.gguf` | 128.2 / 137.7 / 133.0 / 127.2 / 128.1 | llama.cpp `0324696b8`: 119.64 ± 9.86 | 1.07 |
| `Ternary-Bonsai-1.7B-PQ2_0.gguf` | 82.0 / 81.8 / 91.3 / 81.9 / 84.3 | PrismML fork `adfffbe`: 94.82 ± 6.20 | 0.87 |

## Same-session ablation, Qwen, 32 tokens, 3 reps

| configuration | tok/s |
|---|---|
| `q8act`, 8 threads, pinned | 130.7 / 130.5 / 131.3 |
| `q8act`, 8 threads, `APEIRON_PIN=0` | 120.4 / 119.3 / 115.6 |
| `q8act`, 4 threads | 82.2 / 89.5 / 81.9 |
| `q8act`, 1 thread | 26.5 / 26.5 / 26.5 |
| `exact`, 8 threads | 8.9 / 9.0 / 9.0 |

## Attribution, largest first

| change | Qwen tok/s | factor | structure behind it | did the proof shape the code? |
|---|---|---|---|---|
| AVX2 row kernels ported from `fold` (`src/engine/chart/{mod,x86}.rs`, W2-T6B) | 8.4 → 35 | ×4.2 | `fold` `rowdot.rs`; the Q8Act contract `matvec_q8act = matvec_fixed ∘ dequant ∘ quant_act` | Indirectly: the refined exact path (`matvec_fixed`, `Numeric/DequantRefinement.lean`) is the equality oracle every kernel is tested against. The kernels are not extracted; they are `tested`, not `proved`. |
| Persistent pinned pool (`src/engine/chart/pool.rs`, W0-T2) | 35 → 78 | ×2.2 | `fold/hashtrinity/src/par.rs`; spawn-cost measurement (86.5 µs × 169 dispatches) | No. `Numeric/MatvecChunk.lean` (chunked = unchunked) was proved after the pool existed. Pinning alone is worth ×1.09 in the ablation. |
| One dispatch per projection group, pooled SwiGLU (`chart/ffn.rs` `projections_q8act`, W2-T9) | 100 → 132 | ×1.32 | exact integer accumulation: any partition of the rows or of the sum is bit-identical | The property that licensed it is exactness of integer sums, a design choice of the proven basis; no theorem was consulted when writing it. |
| Parallel attention heads (W2-T6C) | 78 → 90–101 | ×1.2 | — | No. |
| Pooled `quant_act`, pooled QK-norm + `RoPE` (W2-T9B) | ≈129 → ≈138 (frozen A/B, load 3–4) | ×1.07 | exact integer RMS sums | Same as T9. |
| Row cores: 1.93 → 1.38 → 1.24 → 1.21 instructions/weight (W2-T6D, T6E, T6F) | ≈101 → ≈106 | ×1.05 | roofline model (compute ceiling ≈171 tok/s at 2.0 instructions/weight) | Model-directed. A 37% instruction cut gave ≈5% at 8 threads: the compute ceiling overstated this lever. |

Bonsai: the lookup-table kernel `src/engine/kern/lut.rs` is the whole path (0.0279 vs 1.75 ns/weight single thread
against `matvec_fixed`). `Numeric/LutPQ2.lean` (bit-plane decomposition, mirror `T[0xAA−b] = −T[b]`, `|P| ≤ 508`)
was proved concurrently with the kernel (worker logs finished 18:46 and 18:47); both derive `508 = 4·127` from the same
decomposition in the task statement. The decomposition shaped the code; the Lean module certifies it. W2-T11 raised
single-thread LUT rows 5–24% per class with no 8-thread change: `perf` puts ≈48% of Bonsai decode cycles in the pool
worker loop, so dispatch cost, not the kernel, bounds Bonsai.

## Levers the measurements refuted or deferred

| lever | proof | measurement | result |
|---|---|---|---|
| L3 zero-block skip | — | `examples/decode_probe.rs`: all-zero 32-weight blocks 0.000000 (Qwen), ≈1e-6 (Bonsai) | refuted on the weight side; activation side unmeasured |
| L2 lookup/speculative decoding | `Nous/Model/JacobiDecode.lean` (`jacobi_exact`, `verify_accept_prefix`) | prompt-lookup n=1 on `README.md`: coverage 0.6458, mean accepted run 0.337 at k=2 → 1.22 tokens/pass; `matmat_q8act` k=2 costs 1.42–1.53× a single column (`kern/matmat.rs` probe) | net loss at present column cost; needs `time(k)` near flat |
| L1 k-column passes | — | `time(8) = 3.37·time(1)` | falsifier fired |
| MoE router | `Numeric/MoeRouter.lean` | neither model is MoE | no effect |
| `ChartDot.lean` bounds | `qcode_range`, `row_err_bound` | proved after the kernels | certificate only; no width chosen from it |

## After W2-T12 (pool), T13 (pooled LUT table build), T6G (Q5_0 plane decode)

Idle host (load 0.9 at start), two interleaved passes, 64 tokens, ids equal on every run:

| model | ours, pass 1 | ours, pass 2 | baseline pass 1 / pass 2 | median ratio |
|---|---|---|---|---|
| Qwen | 126.1 125.3 135.3 125.0 125.5 | 123.1 130.8 125.8 125.2 125.2 | llama.cpp 111.77 ± 3.01 / 107.68 ± 2.47 | 1.14 |
| Bonsai | 88.2 84.2 82.3 81.9 82.2 | 82.5 86.2 82.5 86.8 82.2 | fork 84.29 ± 1.91 / 83.54 ± 1.90 | 0.98 |

| change | measured effect | structure behind it | did a proof shape the code? |
|---|---|---|---|
| T12: per-worker padded slots, static round-robin, caller runs its share first | empty dispatch p50 at width 8: 806 → 412 ns (n=8), 3100 → 421 ns (n=64); task spread `[41,13,10,0,…]` → `[8,…,8]`; engine +2–7% paired, inside noise | concurrency measurement; T11's perf (48% cycles in the worker loop) | No (worker attribution). |
| T13: table build pooled and unrolled | build 13.0 → 4.0 µs at `n_in` 6144; Bonsai +3.2% paired median over 30 pairs | `LutPQ2.lean` `planeSum` recurrence read, not built | No: the split is disjoint group ranges of the same recurrence. |
| T6G: Q5_0 decoded in plane form with one `16·Σxq` correction per block | 1.2114 → 1.1628 instructions/weight; tok/s flat | linearity of the exact integer dot | No Lean module determined it; the exact-integer contract licensed it. |

Lead controls, each exit 101 on the named test: T12 mutant (a worker drops its last owned task) in a real crate copy
`/home/socol/oc-tmp/tput-t12/lead-ctl` → 5 of 6 pool tests; T13 mirror-entry corruption → `matvec_lut_matches_fixed_on_random_payloads`;
T6G fifth-bit force → `chart_matches_fixed_on_random_payloads_for_every_family`.

## Reading

The measured gains come from porting measured `fold` kernels and the pool, then removing dispatches. The proven
basis contributes one decisive property: exact integer arithmetic makes every schedule, partition and fusion
bit-identical, so each rewrite is checked by equality against one refined oracle instead of by tolerance or
benchmark accuracy. No Lean theorem so far preceded and determined a speed change. The one theorem positioned to do so,
`jacobi_exact`, pays only when verifying k tokens costs about one pass, which the current per-column cost does not
give.

Not covered: rustc/LLVM codegen, the `unsafe` pool and AVX2 intrinsics (tested, not proved), host noise
(llama.cpp ± 6–10 tok/s in these windows).

## After T14B, T15, L7, L8 (worker receipts; idle rerun owed where marked)

| change | effect | structure behind it | did a proof shape the code? |
|---|---|---|---|
| T14B: `i64` two-limb row lift in `kern/lut.rs`, guarded per call | LUT kernel 0.023–0.032 → 0.0205–0.0217 ns/weight single thread; the guard admits 600/600 real Bonsai calls | `LutPQ2.planeSum_bound` | Yes: the first speed change whose data width was chosen from a Lean bound. |
| T15: `LayerIndex` (one tensor lookup pass per token), `FfnScratch` reuse | caller-serial 0.752 → 0.730 ms (Qwen), 1.47 → 1.32 ms (Bonsai); allocations −5–7% | — | No. |
| L7: bound-pruned argmax probe (`examples/decode_probe.rs`) | Qwen k=6 summary: median candidate set 283 rows, projected LM head 1.35 vs 1.76 ms; Bonsai: bound ≈63 logits vs median top-2 gap 0.72, nothing pruned | `Numeric/PrunedArgmax.lean` `pruned_argmax_exact` | Theorem first. Sound and weak: ≈5% of Qwen, nothing for Bonsai. |
| L8-T1/T2: FFN split by hidden slice, i128 partials, one reduction (`kern/tp.rs` `ffn_tp`, not wired) | contended host (load 5–8): `ffn_tp` 1.29–1.57× slower than the row split on Qwen layer-0 shapes, 1.09–1.25× on a balanced 8-slice control; idle rerun owed | `Numeric/ColumnCover.lean` `split_ffn_exact`, `quant_locality` (registered in `AxiomAudit`, `lake build Numeric.ColumnCover AxiomAudit` exit 0, 9058 jobs) | Theorem first, and it licensed the design. Falsifier (faster at 8 P-cores) did not fire; loss attributed to 256-column slice granularity (19 slices over 8 cores). |
| L8-T4E: E-cores 16–31 with a speed-weighted split (`APEIRON_ECORES=k`) | contended host: Qwen k=0 137.3, k=16 132.7; Bonsai k=0 81.6, k=16 80.2 tok/s; no k>0 wins | host read bandwidth | No. |

Lead gates on the live tree after these tasks: `cargo test` exit 0 (484 passed), clippy exit 0, fmt exit 0.

Per-phase share of the bandwidth floor (phase table `tput-t15.out`, bytes from `docs/evidence/pipeline-inventory.md`): Qwen Q/K/V 93%, LM head 81%, FFN 68%, O 56%; Bonsai FFN 60%, Q/K/V 51%, O 51%, LM head 74%. The FFN is the largest absolute excess on both models (Qwen 1.31 ms, Bonsai 2.30 ms per token).

## Prefix Table (`src/engine/prefix/`)

| change | effect | structure behind it | did a proof shape the code? |
|---|---|---|---|
| Prefix Table on the crate `Table`: an exact token prefix at 16-token boundaries → the saved `KvState` | 512-token prompt, 496 shared: cold prefill 62.2 s → warm 2.19 s (28.4×), argmax bit-identical; lookup 2.30 µs; one state per prompt (O(n) memory, ≈24 MB); forced hash collision misses (control red) | causality: rows `0..m` of an append-only KV cache do not depend on later tokens | No theorem shaped it; the rate test explains why it pays: an exact prefix is a low-rate address. Loaded host, grade `tested` + `measured`. |
| `KvState::truncate` (`src/engine/attn.rs`) replaces a raw-pointer write into `KvState`'s assumed field layout in `truncate_kv_state` | removes the `unsafe` block from `src/engine/prefix/` (remaining `unsafe`: `chart/`, `kern/`, `fold.rs`, `table/partition.rs`); refuses when a layer holds fewer than `m` positions | same causality | No. Test `truncate_kv_state_resets_occupied_and_resumes_append` now also checks the rewound step equals a fresh state's step (token and logits checksum) and the refusal; control (refusal removed, crate copy `/home/socol/oc-tmp/lead-trunc-ctl`) fails at `!truncate_kv_state(&mut kv, 6)`. Placed after `attn_head` so no extracted source line moves: `proof/regen.sh --check` exit 0. `cargo test` 496 passed, clippy 0, fmt 0. |

| LRU eviction (`PrefixCache::insert`, `lru_victim`, `evict_entry`; PREFIX-T3) | inserts past the budget evict least-recently-used prompts (insert and confirmed hit count as use); a snapshot larger than the whole budget refuses with the cache unchanged; `O(entries)` per eviction | stale boundary keys resolve only through full token confirmation, so a reused slot never answers for another prompt's tokens | No. `cargo test engine::prefix` 8 passed (`mod.rs` `cea4e3c8…`, `tests.rs` `ed3c0549…`); worker control (evict most-recent) red at `tests.rs:544`; lead control (token confirmation removed) red in 3 tests. |

Residuals: the crate `Table` has no removal, so boundary keys of evicted prompts stay (safe misses, unaccounted bytes); a shared-prefix key that points at an evicted prompt misses even when a live prompt holds that prefix.
