# The sign dot's bit sum is the word's popcount: 58 → 44 instructions, 1.39×; and the only exact duplicate in the pack is qkv

Criterion **T1** (decode), with Q1 context. Grades: `measured` for every instruction count
(`objdump`), every ratio (three interleaved runs, minimum of fifteen samples each), every byte and
word count, the Hamming distances and the GF(2) ranks; `tested` for the vector source's equality
with the scalar macro; `model` for the 58/44 prediction and the port explanation. Scope: the exact
readers' `i64`-lane fallback and the sign planes of the 8B pack. **The q8act product path is not
touched by the kernel change and T1's number does not move**; the sections say where the time
actually goes.

## The two source subtractions are one conditional negate, and the binary has neither

`sign_dot_avx2` writes `(t ^ mask) − mask` twice in the source, where `mask = 0 − bit`. The
compiled loop contains **no `vpsubq`**: LLVM turns `0 − (word >> b & 1)` into `vpsllvq` (bit to
position 63) plus `vpcmpgtq` against zero (the `0/−1` mask), and the conditional negate into
`vpxor` plus one `vpaddq`. The two source subtractions are not two machine subtractions; the
negate itself has nothing to remove. What the loop did duplicate is the next section's subject.

The plane loop's two `vpshufb` are also not a repetition: the first (`repeat`) copies byte `4k` of
each dword to the four bytes of that dword; the second (`select_one`) is a 32-byte table indexed
by the masked nibble. Different controls, different tables, different jobs.

## The repetition is in the bit sum, and it is 16 of 58 instructions

`(t ^ mask) − mask` is `(t ^ mask) + b`, so the dot is `Σ (t ^ mask) + Σ b`. `Σ b` over one packed
word is **`popcount(word)`**: a property of the word, not of the four lanes. The lane-wise form
rebuilt it with eight `vpsrlq` and eight `vpaddq` per word.

| loop, per 32 terms | r1 | r2 |
|---|---:|---:|
| `vpsllvq` | 8 | 8 |
| `vpsrlq` | 8 | 0 |
| `vpcmpgtq` | 8 | 8 |
| `vpxor` | 8 | 8 |
| `vpaddq` | 16 | 8 |
| `popcnt` | 0 | 1 |
| `vpsubq` | 0 | 0 |
| total | **58** | **44** |

`sign_dot_avx2` now enables `avx2,popcnt` and `sign_dot_i64` detects both; an AVX2 host without
POPCNT keeps the plane reader and the scalar macro. The equality test
`the_vector_source_equals_the_scalar_source` covers lengths 0, 1, 31, 32, 33, 64, 127, 128, 1000
with random words and with the tail's padding bits set, and it fails if `bits_total` is dropped.

## The `vpshufb` table is `vpminub(x, 1)`, and it is slower

After `vpand` with `0x0804_0201` the bytes are `{0,1,2,4,8}`, and `min_epu8(x, 1)` maps them to
`{0,1,1,1,1}` — exactly `select_one` on its live domain. Bit equality holds; the swap is one
instruction for one instruction and moves a uop off port 5. Measured, it is **1.11–1.17× slower**
(three runs: 1.173, 1.112, 1.131): port 0/1 is the binding port in that loop, so taking work from
port 5 adds to the queue that is already full. Not applied.

## Which kernel the product runs

`Reader::Q8` builds `CodeTerms`, and `Stage::dot` returns `codes.sum − 2·selected` before it looks
at `Terms`. When `CodeTerms::new` admits the activation — the measured q8act configuration — the
token runs `code_select_avx2`, not either kernel above:

```text
broadcast, vpsrlvd, vpshufb, vpand, vpshufb, vpmaddubsw, vpmaddwd, vpsllvd, vpaddd
```

Nine vector instructions and three loop instructions per 32 terms. It already carries the bit
frequency: `CodeTerms::sum` is `Σ_b S_b·Σ_j codes[j]`, computed **once per activation** and shared
by every output row. `sign_dot_avx2` is the exact reader's fallback, reached when the terms do not
split into four signed bytes or the lane sums leave `i32`.

So the change is a 1.39× on the fallback and the oracle, not on the token. The measured A/B is
`pop/current = 0.711, 0.728, 0.721` → **1.37–1.41×**.

## The sign planes have no second copy

Density is 0.4978 (U) and 0.4980 (V), so one bit per sign is the entropy. Beyond that:

| test, layer 0 | result |
|---|---|
| `V` GF(2) rank | 2528 of 2528 rows |
| `U` GF(2) rank, q | 2528 of 4096 rows (= the column count) |
| nearest-neighbour Hamming, `V` (4096 bits) | min 1921, median 1950, chance 2048 |
| nearest-neighbour Hamming, `U` q (2528 bits) | min 13, median 1117, chance 1264 |

The rows are 6% (V) to 12–20% (U) closer than chance and full rank over GF(2). A row coded as a
base plus its XOR delta costs `n·H(d/n)` bits, `n` at `d = n/2` and half only below `d ≈ 0.11n`.
At the measured medians (`d/n` 0.40–0.44) that is 0.97–0.99 of the raw row: no saving. The tail
that does qualify is layer-local and small: 1.98% of `U` rows at layer 0, 0.15% at layer 17, **0%
at layer 35** within 256 bits of any other row. It is not a basis and not a lever.

The one exact duplicate is the qkv split. The source fit shares `V`, `s_pre`, `s_mid` and the
indices across q, k and v (`factorization.shared_input`, group `self_attn.attn_qkv`); the converter
writes them three times.

```text
duplicate V                 90,505,216 B   (2 of 3 copies, 45,252,608 B each)
with s_pre, s_mid, indices  91,450,016 B   9.369% of 976,049,680 B
rate                       1.124186 → 1.018855 b/w (the 8,096 B header excluded)
V words per token           33,939,456 of 114,749,440 = 29.6%
fused V words per token     11,313,152, saving 22,626,304
share of all sign words     22,626,304 / 238,219,264 = 9.50%
```

A fused block is not a packing detail: the three sections share one `z = V·diag(s_pre)·t`. Stacked,
the source fit stores one common right factor,

```text
[W_q; W_k; W_v] = [U_q; U_k; U_v] · diag(s_mid) · V · diag(s_pre),
```

and the converter writes `V·diag(s_pre)` three times. Fusing it runs the V stage once and leaves
the three U stages, which is 9.50% of the sign terms per token, unmeasured end to end. The engine's
`NqLinear` holds one `v` per section and has no group; the pack has no shared-input block. Both are
required; neither is implemented.

This is a Q1, T1 and — through the rate it frees — Q2 lever, and the measurement puts its Q2 value
at **48%** of the bar's bytes, not 100%. The freed 91,450,016 B plus the 48,459,111 B headroom is
**0.161145 b/w** (cap 1,024,500,695 B, body 884,591,568 B after fusion). The two-plane residual is
worth 0.2041 nats/bw at ctx 2048 — the full 3.130640 b/w pack scores 8.094085, 1.067×, against the
base's 12.189530 — so the 1.5× bar needs **0.3364 b/w** and the cap's 0.161 is 47.8% of it. A uniform
8% spend is modelled at 1.554981; the 19-section salient-mass selection measures 1.598041 at
1.284229 b/w, 0.164× of uniform per byte (`1794893449`). Rank does not buy the difference: 341,496 B
per rank-step is +1.63 rank/section, 0.06%. The dedup is 0.105 b/w and 9.5% of the sign terms, and it
makes the 1.598 pack shippable at 1.178909 b/w; it is half of the Q2 need, not all of it.

## What the sibling corpus actually proves

The words in `../sanguine` and `../sanguine-ref` do not contain a frequency-dedup theorem, and
`Icarus` occurs in neither (it is a third repository, `~/Workspace/icarus`, ported at
`proof/corpus/Icarus/`). The declarations that do exist and that name this operation:

| declaration | statement |
|---|---|
| `sanguine-ref/Substrate/SeedCore.lean:131` `rexpand_rcollapse` | `rexpand F (rcollapse F t) = t`: a keyed table expands losslessly |
| `sanguine-ref/Substrate/SeedCore.lean:80` `dedup'` | keeps one entry per key; membership preserved |
| `sanguine/proof/Apeiron/Combinator/Pair.lean:93` `iota_eq_pack` | `iota = pack S K`, by `rfl` |
| `sanguine/proof/Apeiron/Combinator/Basic.lean:208` `iota_app` | `iota M → M S K` |
| `sanguine-ref/Substrate/Algebra/Bulk/GoldenRouting.lean:85` `route_injective` | Zeckendorf routing is a perfect hash |
| `…/GoldenRouting.lean:159` `tag_is_lossy` | a coarse tag provably collides |

`rcollapse`/`rexpand` is the shared-V block's law: one key per distinct payload, expansion exact.
`iota = pack S K` is the reading of one stored object that yields the pair at each use — a mapping,
not a tensor theorem. `route_injective` versus `tag_is_lossy` is the exact-versus-lossy choice, and
the Hamming table above says only the exact key is admissible for these rows. The kernel's
`popcount` collapse is not from that corpus; it is the identity `Σ b = popcount(word)`.

## What this does not cover

The q8act token's time, the pack size after fusion, and the fused kernel's throughput: no shared-V
block exists, and no pack was run. Rust code generation, the kernel ABI, and the `unsafe`
container are outside. The 1.39× is the fallback reader's loop on this laptop, not a model result.

## Gate state

`cargo fmt --all -- --check`, `cargo clippy --workspace --all-features --all-targets --locked
-j 1 -- -D warnings`, `tools/check-claims.py` and `tools/book-check.py` pass. The workspace test
run reports 645 passed, 41 failed, 61 ignored, and every failure is the external-fixture test or
the device path. 40 tests take the device path (39 `gpu::*` plus
`learn::nanoquant::tests::spd_inverse_matches_serial`) and need a device or `glslc` on this host.
`engine::tests::lm_head_and_argmax_through_the_real_tensor_table` asserts the Qwen3-style tied head
against `qwen05b-Q4_K_M.gguf`, which is a Qwen2.5 file with a separate `output.weight` (the
symlink's target, not this repository). The fixture test fails identically on the binary built
before this change; the `gpu::*` tests are absent there (that binary has no `gpu` feature) and fail
here on the missing device and `glslc`.
`the_ternary_projection_equals_the_exact_reader_on_trit_exact_words` failed for a fixture reason,
not a reader reason: stage 2's activation was `Σx − 2x_j = 3..7` words, whose ternary code carries
a nonzero residual that the sign-only reader drops; the fixture now feeds one nonzero word, so both
stages are trit-exact and the test passes.
