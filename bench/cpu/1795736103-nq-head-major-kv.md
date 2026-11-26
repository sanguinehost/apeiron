# nq: KV-head-major plane — decode 23.53 → 17.52 ms/token, `attn.core` 2.50×

Criterion: **T1** (decode), with **T2** (turn) moved by the same wall. `attn.core` is the dominant
decode term on the unmodified tree (43.0%, ceiling 1.75×), so `phase_split --target attn.core`
exits 0 there and this change is a dominant-term campaign step: paired F32-KV decode
**23.53 → 17.52 ms/token**, **42.5 → 57.1 tok/s** (1.343×, 3/3 rounds). `attn.core` itself is
**10.11 → 4.05 ms** (2.50×). Prefill is neutral (3.837 → 3.822 s, inside the run spread).

Grade: `measured` for the paired walls, rates, splits, STREAM and A/B; `model` for the ratios, the
`1/(1 − s)` ceilings and the bytes-per-token arithmetic; `tested` for the carrier equality tests,
the layout A/B and the PPL/first-token identity. Nothing is `proved`: the extraction and the
maintainer gate did not run.

## Machine, binaries and command

Machine: OVH box `ubuntu@139.99.223.161` (8 vCPU EPYC Milan, 32 GiB), `APEIRON_THREADS=8`,
`~/bin/adg`. CPU-tier evidence; the timed target is the laptop (i5-10210U), which did not run this
comparison.

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 0.977441 b/w, 53,808,384 B);
tokenizer `tokenizer.json`.

| arm | tree | binary |
|---|---|---|
| O, committed `14fc601d` | `~/Workspace/apeiron-nq-base` | `target/release/examples/phase_split`, `prefix_solve` |
| N, this change | `~/Workspace/apeiron-nq` | same paths |

Commands, three rounds interleaved O/N:

```sh
APEIRON_THREADS=8 APEIRON_KV_CARRIER=f32 phase_split \
  --model skeleton-q8head.gguf --pack nq06-v3.nqp --steps 32 --prefix 1795 --target attn.core
APEIRON_THREADS=8 APEIRON_KV_CARRIER=f32 prefix_solve \
  --model skeleton-q8head.gguf --pack nq06-v3.nqp --tokenizer tokenizer.json \
  --len 1827 --differ 0 --decode 1 --mb 512
```

Raw evidence: `1795736103-nq-head-major-kv.log` beside this file.

## Paired walls

| phase | O, mean | N, mean | N / O | rounds |
|---|---:|---:|---:|---|
| decode, ms/token (32 steps after 1,795-token prefix) | 23.53 | **17.52** | **0.745** | 3/3 |
| decode, tok/s | 42.5 | **57.1** | **1.343** | 3/3 |
| prefill, s (1,827 tokens) | 3.837 | 3.822 | 0.996 | inside spread |
| prefill, tok/s | 476.0 | 478.1 | 1.004 | inside spread |

Decode O: 23.26 / 24.27 / 23.05. Decode N: 17.45 / 17.48 / 17.63. Prefill O: 3.861 / 3.809 /
3.841. Prefill N: 3.836 / 3.815 / 3.814. The first decoded token is 315 on both arms in every
round.

## Split

Decode leaf shares, mean of three rounds, F32 KV:

| leaf | O, ms | O, share | N, ms | N, share |
|---|---:|---:|---:|---:|
| `attn.proj_in` | 2.68 | 11.4% | 2.65 | 15.2% |
| `attn.core` | **10.11** | **43.0%** | **4.05** | **23.1%** |
| `attn.proj_out` | 1.58 | 6.7% | 1.64 | 9.3% |
| `ffn` | 5.14 | 21.9% | 5.21 | **29.7%** |
| `head` | 3.12 | 13.3% | 3.09 | 17.6% |
| untraced | 0.10 | 0.4% | 0.09 | 0.5% |

The dominant term moves from `attn.core` (ceiling 1.75×) to `ffn` (ceiling 1.42×). The next
campaign step targets `ffn`; `phase_split --target ffn` exits 0 on this tree.

`attn.core`'s 2.50× wall ratio against a 43.0% share is 1.75× of the token ceiling, so the change
captured most of what the former dominant term could give: 5.46 ms of the 6.01 ms saved is
`attn.core`, the rest is its own arithmetic moving into the unshared `untraced` and the other
leaves' shares rising as the denominator falls.

## The change

The planes were position-major: position `p` carried all `n_head_kv` rows contiguously, and the
row for `(p, kvh)` was at `(p · n_head_kv + kvh) · head_dim`. One KV head's causal history was
therefore strided by `n_head_kv · head_dim` — for the 0.6B, 1,024 elements apart per position. The
decode tile core reads a tile's K and V rows once per 8 positions and folds the whole query group
against them; on the position-major plane every row of a tile opened a different 4 KiB region, so
the tile read was a scatter and the row-outer loop was neutral or negative (the earlier
`f32_core` loop was chunk-outer for that reason).

Every carrier now stores KV-head-major: the row for `(p, kvh)` is at
`(kvh · cap + p) · head_dim`. One head's history is one contiguous run, the per-key stride is
`head_dim`, and `attn_tile8_avx2` is row-outer — K row `j` loaded once into registers and reused
for every 8-lane chunk, then V row `j` once. The arithmetic is unchanged: the same tile, the same
8 independent dot chains, the same vector `exp`, the same sequential max/denominator recurrence,
the same accumulator order. Only the address of each row changes, so every output word is
identical (equality tests below, and PPL/first token unchanged).

Touched for the layout: `KvState::append` (one write per head, source row `h · head_dim`),
`plane_at`/`row_bounds`/`score_at`/`value_term` (index `(kvh · cap + s) · head_dim`), the hoisted
`first + s · stride` bases in `attn_head`, the certified read, `attn_group_f32`/`attn_group_f16`,
the F16/F32 tile paths, and `cert::summarise`/`KvBlockIndex`, which now carry `cap`. The i8 plane
already used this layout; the F32/F16/Q32 planes and the block index are what move.

## Layout A/B, core only

`kv_layout_ab_bench` (ignored test, `APEIRON_AB_POS`), 28 layers × 8 KV heads × group 2,
`head_dim` 128, the same group core over a plane built both ways:

| positions | threads | position-major, ms/token | head-major, ms/token | ratio |
|---:|---:|---:|---:|---:|
| 1,826 | 8 | 9.224 | 3.419 | 2.70× |
| 8,190 | 8 | 41.401 | 17.060 | 2.43× |
| 8,190 | 1 | 293.759 | 73.394 | 4.00× |

The core-level gain is larger than the token-level 2.50× on `attn.core` because the token also
carries the score/softmax/readout arithmetic outside the tile loop.

## STREAM, re-measured

`tools/membw.c`, 128 MiB per thread × 25 passes × 8 threads: **193.28 GB/s mean, 237.64 GB/s best
pass**. The 100.74 / 112.72 GB/s row in `1795727609-nq-05-baseline.md` was measured on a loaded
host and is withdrawn as a ceiling input.

At 1,827 positions the decode read is 638,184,704 B/token (weights 218.9 MB, q8 head 165.3 MB,
KV 419.1 MB), so the bandwidth ceiling is **302.9 tok/s mean / 372.4 tok/s best pass**, not
157.8. The 0.5× decode target of 183.1 tok/s is inside the byte ceiling; the earlier statement
that 0.5× decode is unreachable at the F32 KV carrier is withdrawn. The gap is kernel
efficiency: `attn.core` moves 419 MB in 4.05 ms ≈ 103 GB/s, about half the measured mean, so
attention still has roughly 2× of bandwidth headroom.

## Identity and tests

| check | O | N |
|---|---|---|
| PPL, F32 KV (`--tokens 2048 --ctx 512`, 2,044 scored) | 67.257790 | **67.257790** (NLL 4.208533) |
| PPL, F16 KV | 71.612941 | **71.612941** (NLL 4.271276) |
| first decoded token, F32 / F16 | 315 / 3845 | **315 / 3845** |

`cargo test --release --locked --lib engine::attn`: all equality, certificate and hoisted-index
tests pass on the new layout, including `hoisted_row_base_matches_plane_index_oracles` against the
rewritten `plane_at`/`row_bounds` oracles. Full release lib: 706 passed, 1 failed — the known
`gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` (box `glslc` 2023.8, unrelated GPU
tier). `cargo fmt --all -- --check` green; `cargo clippy --workspace --all-features --all-targets
--locked -- -D warnings` green; `tools/check-claims.py` exit 0; `tools/book-check.py` 0 findings.

The f16 hardware-rounding oracle is fixed in the same change: `_mm256_cvtps_ph` and the assertion
now live in one `#[target_feature(enable = "avx2,f16c")]` body. Release inlining had folded the
intrinsic into a caller compiled without those features, which returned `0x0000` for `-0` and
`0x0000` for `2^-24`; the carrier was correct, the harness was not.

## What this does not cover

- **Extraction and Lean.** The layout changes the extracted index arithmetic in `attn.rs` and
  `cert.rs`. `proof/regen.sh --check`, `lake build` and the axiom audit did not run; the committed
  extraction is stale and no `proved` claim is made here.
- **File cache compatibility.** `KvState::with_file` planes are head-major now. The file carries no
  header, so a cache file written by a position-major build is silently misread by this one. It is
  a cache, not a format; a stale file must be deleted, and there is no version tag to refuse it.
- **GPU tier.** Not run; the box has no device and its `glslc` does not reproduce the committed
  SPIR-V.
- **Laptop.** No timed claim names the laptop; the box numbers above are CPU-tier evidence.
- **Prefill.** The change is decode-facing; prefill is neutral within the spread and its
  `attn.core` share (33.4%) is untouched by this receipt.
- **Remaining gap.** 17.52 ms/token is 3.21× from the 5.46 ms (183.1 tok/s) decode target; the turn
  is 3.822 + 128 × 0.01752 = 6.07 s against 2.681 s, 2.26× short. `ffn` is now dominant at 29.7%
  (ceiling 1.42×) and is the next target.
