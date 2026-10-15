# Onebit W12: the IQ1_S port, byte-identical, and why its search is exact

Subsume record (AGENTS §3a): empirically best known 1.5625 b/w codebook quantizer for this role is
llama.cpp `IQ1_S`. Repository `/home/socol/Workspace/llama.cpp`, rev `ece963f41`, `ggml/src/ggml-quants.c`
sha256 `07143d7068936ae4…`. Status: `subsumed`. Search command:
`rg -n "quantize_row_iq1_s_impl|iq1_find_best_neighbour2|iq2xs_init_impl|kgrid_1bit_2048" ggml/src/ggml-quants.c`.

## Port

| item | value |
|---|---|
| source | `src/learn/iq1s.rs` (sha256 `703aaad894c74ed3…`), `src/learn/iq1s_grid.rs` (`db62582133440219…`) |
| control | `examples/iq1s_check.rs` (removed; at 2f15d6a9): quantize `blk.3..23.ffn_down.weight` of `qwen05b-f16.gguf` under the C4 imatrix, compare to the `llama-quantize` `IQ1_S` file |
| result | 21 tensors × 17,024 blocks = 357,504 blocks, **0 differ**, 1.9 s on 16 threads (`/home/socol/oc-tmp/escalation/2026-10-11-iq1s-check.log`) |
| mutation | `d·1.125 → d·1.0` in a throwaway copy: 34,048 of 34,048 blocks differ on `blk.3`, `blk.4`, exit 1 (`/home/socol/oc-tmp/escalation/2026-10-11-iq1s-mutation.log`); copy removed, working tree untouched |
| tests | `cargo test --release --locked --test learn_iq1s`: 3 passed (grid equals the C table, tables well-formed, decode is scaled grid plus shift) |
| clippy | `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`: exit 0 |

Grade: `tested` (byte identity on the population above). The grid has 2048 distinct codewords, entries in
`{-1, 0, +1}`; its nonzero-count histogram is 0:1, 1:16, 2:112, 3:410, 4:150, 5:992, 6:114, 7:3, 8:250. It is an
empirical selection; no algebraic characterisation was found.

## Why the search is exact (`proof/corpus/Perf/Iq1s.lean`, `proved`, closed cone)

| statement | theorem |
|---|---|
| rate is `25/16` b/w | `rate` |
| weighted error in the scale is a parabola; `Σwqx / Σwq²` minimises it | `err_expand`, `err_eq_best_add_sq`, `refit_scalar_le` |
| for sorted levels the nearest-level labelling is monotone in `x` and no labelling beats it, so a search over splits of the sorted weights is exhaustive over all `3ⁿ` | `idx_mono`, `idx_nearest`, `exists_monotone_le` |
| the pair form of `H = XᵀX` has discriminant `H_ij² − H_ii H_jj ≤ 0`: elliptic or parabolic, never hyperbolic | `pair_discriminant_nonpos` |
| on the direction `(H_ij, −H_ii)` the form equals `−H_ii` times the discriminant | `pair_form_on_direction` |

`IQ1_S` weights the squared error by a diagonal proxy of `H`; the layer error is `rᵀHr` (`Perf.OnebitW10`), whose
off-diagonal part `Σ_{i≠j} H_ij r_i r_j` is what the proxy omits. The pair discriminant is the per-pair measure of
that part. The correspondence between monotone labellings and the `(i1, i2)` enumeration of the Rust loop is
`tested` by the byte identity, not a theorem.

## Like-for-like (P1: 21 `ffn_down` tensors, 91,521,024 weights; all else from the `IQ1_S` file)

Source: `docs/evidence/onebit-w9.md`. Verdict `p1_verdict` (`proved` over the recorded figures).

| arm | b/w | KLD (nats) |
|---|---:|---:|
| `IQ1_S` imatrix | 1.5625 | 0.6315 |
| `had-gptq1` | 1.125 | 0.8345 |

The all-168-tensor figures (P2: `had-gptq1` 6.8496, W11 5.642) are a different population and are not compared with
the P1 rows.

## Not covered

rustc/LLVM codegen (the identity is for this build and the f32 operation order the C reference uses), the
population beyond the 21 tensors of one model, and the quality of `IQ1_S` itself, which is a `measured` row in W9.
