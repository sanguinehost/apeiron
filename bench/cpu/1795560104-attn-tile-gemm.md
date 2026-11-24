# Tiled attention: llama.cpp's 6×2 GEMM microkernel, one rescale per key tile

Criterion: **T2** (turn latency), partial: the `attn.core` leaf of the cold prefill, not the turn.
Grades: `measured` for the leaf and wall A/B and the phase split; `tested` for the
twin/body equality and the per-tile-rescale bound; `model` for the instruction count, the K/V
stream arithmetic and the laptop projection. Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets ×
1 core × 1 thread, 31.3 GiB, 256 MiB L3, no cpufreq governor), 8 participants, no GPU. Times UTC.
The laptop, the only device a timed claim may name, is **not measured**: `BAT0` 0%, CPU pinned at
400 MHz. Every run under `~/bin/box-run` (agent-guard, 16 GiB cap, 8 jobs); the example was rebuilt
with `touch src/lib.rs` before the timed runs.

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack
`~/models/nq-scratch/nq06-v3.nqp` (196 linears, 28 layers, 0.977441 b/w), the harness default,
`examples/prefix_solve.rs:54`, `--len 1827 --differ 205 --decode 0 --mb 440`.

## What changed

`attn_core_f32_tile` (`src/engine/attn/f32_core.rs`) now has llama.cpp's
`ggml_compute_forward_flash_attn_ext_tiled` loop shape (`ggml/src/ggml-cpu/ops.cpp:8641`,
`049326a`) and its `simd_gemm` 6×2 AVX2 microkernel (`ggml/src/ggml-cpu/simd-gemm.h:24`):

- **One key tile of 64 positions** (`KV_TILE`, llama's `GGML_FA_TILE_KV`). K is packed transposed
  (`kt[d·64 + t]`) and V row-major; the two GEMMs read the packed tiles with contiguous 8-lane
  loads.
- **Scores are one GEMM**: `P[rows][64] += Q[rows][head_dim] · kt[head_dim][64]`, six query rows
  per register tile, two 8-lane columns per microkernel. The old pass called a per-(row, key) dot
  with an 8-lane horizontal reduction for every score.
- **One row maximum per tile** drives one rescale of `den` and the value accumulator
  (`exp(m_prev − m_next)`), then one vector `exp` over the tile's scores. The old pass rescaled the
  accumulator once per position, which is 64 reads and writes of `rows · head_dim` words per tile.
- **Values are one GEMM**: `acc[rows][head_dim] += P[rows][64] · vt[64][head_dim]`. The accumulator
  is read from memory once per tile and written once, against once per position before.
- The microkernel is llama's AVX2 register tile: `GEMM_RM = 6`, `GEMM_RN = 2` (12 accumulator
  YMMs, two B loads, six A broadcasts, 12 FMAs per `k`). The row tail is llama's one-row kernel.

The scalar twin implements the same tiled algorithm and the AVX2 body is bit-equal to it, as
before (`test_attn_core_tiled_equals_scalar_on_tile_edges`,
`test_attn_core_tile_exp_group_edges_agrees_with_the_per_query_core`).

**The guarantee narrows.** The tile no longer reproduces the per-position
`attn_core_f32_strided` recurrence word for word: one rescale per 64 positions rounds differently
from one per position. The per-column core stays the oracle, and the tests now assert a bound
(`2 · (prefix + 64) · f32::EPSILON` relative) instead of equality:
`test_attn_core_query_tile_agrees_with_the_per_query_core`,
`the_tile_group_read_agrees_with_the_per_column_read`,
`the_attention_tile_task_agrees_with_the_per_column_task`,
`the_nq_prefill_agrees_with_the_per_column_prefill`. The per-tile form is the more accurate of the
two: it applies one explicit rescale per tile instead of one multiply of the accumulator per
position. `APEIRON_ATTN_EXP` and `softmax_terms` are removed with the per-position row loop they
switched.

## Result

Cold 1,827-position prefill, 8 participants, `--decode 0 --phases`, the harness default model and
pack. The two arms are one tree with one file swapped: `src/engine/attn/f32_core.rs` from `HEAD`
(the per-position tile) against the new file (the tiled GEMM), each arm rebuilt with `touch
src/lib.rs` and `cargo build --release --example prefix_solve` before its run; six runs in order
new/old × 3, log `/tmp/opencode/box-attn-ab2.log`.

| round | `attn.core` new | old | old/new | cold wall new | old | old/new |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2,224.4 ms | 3,214.9 ms | 1.445 | 13.525 s | 14.463 s | 1.069 |
| 2 | 2,154.0 ms | 3,105.9 ms | 1.442 | 13.092 s | 14.233 s | 1.087 |
| 3 | 2,154.6 ms | 3,130.9 ms | 1.453 | 13.100 s | 14.329 s | 1.094 |
| paired mean | | | **1.447** | | | **1.083** |

The leaf is **1.453×** by medians (3,130.9 → 2,154.6 ms) and the cold wall **1.094×** (14.329 →
13.100 s). The wall move is the leaf move: per round the wall falls 0.94–1.23 s while the leaf falls
0.95–0.99 s, and the projection sum (`ffn` + `attn.proj_in` + `attn.proj_out`) moves between −0.7%
and +1.4% across the arms — window drift on a phase this change does not touch, not a second gain.

The post-change cold split (round 3, 13.100 s, `pass` 97.0%):

| phase | ms | share |
|---|---:|---:|
| `ffn` | 5,300.0 | 40.5% |
| `attn.proj_in` | 2,476.3 | 18.9% |
| `attn.core` | 2,154.6 | 16.4% |
| `residual` | 1,166.0 | 8.9% |
| `attn.proj_out` | 945.5 | 7.2% |
| `norm` | 614.0 | 4.7% |
| `pool.wait` | 476.3 | 3.6% |

`attn.core` moves 21.9% → 16.4% of the cold wall (medians); `ffn` is again the dominant term.

The target's share fixes the ceiling before the run: with `attn.core` at 21.9% of the unmodified
cold prefill, the phase can move the cold wall by at most `1/(1 − 0.219) = 1.28×` however fast its
kernel becomes. The measured wall ratio is 1.094×, which is 34% of the excess the ceiling admits
(`0.094 / 0.279`), and the leaf ratio is 1.447×. `ffn` is the dominant term at 36.9–37.4% on the
unmodified tree, so this change is not the dominant-term fix for the 1,827-token cold turn: the
projection phases are. It is the dominant-term fix for prefill at longer contexts, where attention's
quadratic term overtakes the projection's linear one; the 1,827-token turn is the screen, not the
long-context claim. Decode is untouched, because the tile needs more than one query column; the
decode-time `attn.core` share at 1,827 positions (49.8%, `1795170568-nq06-context-split.md`) is the
per-column path and belongs to the context work, not to this port.

The 211-token shared turn (1,616-position cache hit) gains more on the same leaf: 379.0 / 361.1 /
359.2 ms against 588.2 / 581.8 / 581.2 ms (paired 1.552 / 1.611 / 1.618, mean **1.594**), and its
prefill 1.677 / 1.619 / 1.619 s against 1.858 / 1.847 / 1.866 s (mean **1.134**). The cold turn has
58 tiles per KV head where the shared turn has 7, and the kernel allocates and zero-packs its
`kt`/`vt`/`p`/`acc` scratch once per tile task; the per-task fixed cost is amortised over 8× fewer
tasks in the shared turn, so the cold ratio is the lower of the two.

The instruction model that motivated the port predicted about 3× on the leaf from 10.9K → 3.5K
instructions per key. The measured leaf gain is **1.447×**, so the model is wrong as a rate
prediction. The K/V stream is unchanged by the port: each key and value row is still read once per
32-column query tile, `Σ_t 32(t+1)` rows per (KV head, layer) × 1 KiB × 28 layers × 8 KV heads =
12.5 GB per cold turn. The leaf moves that stream at 4.0 GB/s per core on the old arm and 5.8 GB/s
on the new one, so the bytes are not the whole token and not removed either. No `perf` counter was
read: which of issue rate, L2/L3 traffic and the per-tile pack binds the remaining time is open.
Doubling the query tile halves the stream; that is a candidate for the next change, not a result.

## Why the words change only within the bound

Both paths evaluate one real-valued recurrence over the causal prefix:

```text
m_j    = max(m_{j-1}, max_t s_t)
den_j  = den_{j-1}·exp(m_{j-1} − m_j) + Σ_t exp(s_t − m_j)
acc_j  = acc_{j-1}·exp(m_{j-1} − m_j) + Σ_t exp(s_t − m_j)·v_t
```

with the same score `s_t = (q · k_t)·scale`, the same `exp` kernel, the same score FMA order
(ascending `d`) and the same value accumulation order (ascending `t`). The only difference is
where the accumulator rescale lands: the oracle applies `exp(m_{j-1} − m_j)` to `den` and to every
`acc` lane once per position, the tile applies it once per 64 positions. When the maximum does not
move, the factor is `exp(0) = 1` exactly and the multiply is exact on both paths; when it does, the
oracle rounds `den` and each accumulator lane 64 times per tile where the tile rounds once. With
`prefix` positions in the row's causal past and at most one move per position, the accumulated
relative deviation is bounded by `(prefix + KV_TILE)` roundings on the oracle against one on the
tile; the factor two in `2·(prefix + KV_TILE)·f32::EPSILON` covers both paths' own roundoff.

The old tile was bit-equal to the per-position oracle — it ran the oracle's per-position
recurrence itself, so its words matched. The new tile is llama.cpp's `flash_attn_ext_tiled` shape
and keeps only the bound. Nothing else narrows: the query conversion, the scale, the `exp` kernel,
the readout conversion and the refusal order on the tile are unchanged, and the twin/body equality
stays word-for-word, because both run the tile structure with the same accumulation order and
`exp_f32_avx2` equals eight `exp_f32_scalar` calls lane for lane.

The step's own tests assert the bound in Q32.32 words: `64 + (m·512·(prefix+64) >> 32)`, from
`512/2^32 = 2^-23`, the same construction as the attention core's f32 bound.

One tap assertion narrows with it. `tests/engine_tap.rs:tapped_q_k_v_agree_with_attn_out` rebuilds
`attn_out` from the tapped post-`RoPE` q/k/v through the per-column read; the forward it reproduces
is now the tiled one, so the test asserts the same bound (max 33 words on the fixture's four
columns) instead of word equality, and the two are word-equal with `APEIRON_ATTN_TILE=0`. The
backward path's recomputed probabilities inherit that bound, as
`src/learn/fold/stats/hocapture.rs` now states.

## Gates

The contributor tier ran on the box copy after pushing the ten modified files and this receipt
(`/tmp/box-gate.sh`, log `/tmp/opencode/box-gate6.log`, host `indra-dev-1`, rustc
1.99.0-nightly):

| step | command | exit | result |
|---|---|---:|---|
| 1 | `cargo test --workspace --locked --no-fail-fast` | 101 | 1,100 passed, 3 failed, 85 ignored; the 3 are the `glslc` byte checks (`attn_coopmat_spirv_is_the_glsl` in the lib and in `spirv_storage_race`, `splitk_reduce_spirv_is_the_glsl`) |
| 2 | `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 | clean |
| 3 | `cargo fmt --all -- --check` | 0 | clean |
| 4 | `python3 tools/check-claims.py` | 0 | every graded claim resolves |
| 5 | `python3 tools/book-check.py` | 0 | 20 glossary entries, 0 findings |
| 6 | `mdbook build book` + `mdbook test book` | 0 | build and test clean |
| 7 | `cargo test --workspace --all-features --locked --no-fail-fast` | 101 | 1,127 passed, 82 failed, 110 ignored; every failing test is GPU/Vulkan (`vkCmdWriteTimestamp2 not found`, `glslc` byte comparisons), matching the gate-4 baseline of 1,127/82/109 |

The three `glslc` reds and the 82 all-features reds are environmental: the box has no device and
its shaderc 2023.8 does not reproduce the committed SPIR-V. The maintainer gate does not run on
either host; see Omissions.

## Omissions

- The laptop timing is not measured; the box is the screen (Zen 2, double-pumped 256-bit AVX2;
  the laptop is Comet Lake with two full 256-bit FMA ports), so the gain is a `model` for the
  laptop until the laptop run.
- The maintainer gate (`proof/regen.sh --check`, `lake build`) does not run on either host; the
  new `gemm_tile`/`gemm_ukernel` opaque entries and the removed process-flag entry are unverified
  against the pinned Aeneas and Charon.
- The GPU/SPIR-V tier is not run (no device, no `glslc` pin).
- The GEMM is measured at the 0.6B geometry. The 8B's `group` and `n_head_kv` change the row count
  per tile; the microkernel handles any row count, the tail is llama's one-row kernel.
