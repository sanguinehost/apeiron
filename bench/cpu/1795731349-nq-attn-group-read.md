# nq: one K/V row read per attention tile for the decode group — 1.189× decode

Criterion: **T1** (decode). `attn.core` is the dominant decode term (44.6% on the committed tree,
41.9% after this change), so `phase_split --target attn.core` exits 0 on the unmodified tree and
this change is a dominant-term campaign step: default-carrier decode **3.6507 → 3.0693 s** per
128 tokens, 35.06 → **41.70 tok/s** (1.189×, 3/3 paired rounds). The i8 carrier is unchanged
(0.997×), because it already folds the group against one row read.

Grade: `measured` for the walls, rates, split and binary hashes; `model` for the ratios, the
`1/(1 − s)` ceilings and the bytes-per-token arithmetic; `tested` for the bit-equality of the
grouped core against the per-head calls and the workspace suite. Nothing is `proved`; the
maintainer gate is blocked (below).

## Machine, binaries and command

Machine: OVH box `ubuntu@139.99.223.161` (8 vCPU EPYC Milan, 32 GiB), `APEIRON_THREADS=8`,
`~/bin/adg`, load gate < 1.5. CPU-tier evidence; the timed target is the laptop (i5-10210U),
which did not run this comparison.

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 0.977441 b/w, 53,808,384 B);
tokenizer `tokenizer.json`.

| arm | binary | sha256 |
|---|---|---|
| O, committed `e058890d` | `prefix_solve.rows` | `6605f74c…2fa4c4a6` |
| N, this change | `prefix_solve.group` | `da44f38b…c62529c4` |

Command: `prefix_solve --len 1827 --differ 0 --decode 128 --mb 512`, untraced, three rounds
interleaved O/N on each carrier. Traced split: `phase_split --steps 32 --prefix 1795`. Raw
evidence in the `.log` beside this file.

## Paired decode

| carrier | O, s | N, s | O, tok/s | N, tok/s | N / O time | rounds N wins |
|---|---:|---:|---:|---:|---:|---:|
| F32 KV | 3.6507 | **3.0693** | 35.06 | **41.70** | **0.8407** | 3/3 |
| i8 KV | 2.3357 | 2.3427 | 54.80 | 54.64 | 1.0030 | 1/3 |

The first decoded token is identical on both carriers (315 F32, 3845 i8), and prefill is inside
the run spread: the change touches the decode column path only.

## The change

`KvState::attn_head_group`'s F32 decode branch previously called `attn_head_f32` once per query
head of the KV head. Each call walked the same K/V rows (the head's `head_dim`-wide slice, strided
by the full row) over the whole causal prefix, so a group of `g` query heads read the same
`2 · (pos+1) · head_dim` bytes `g` times — four times for the 0.6B's GQA ratio.

The branch now makes one call to `f32_core::attn_core_f32_group`, whose AVX2 body keeps the tile
loop outside the head loop: tile `s`'s K and V rows are loaded by head 0 and are in L1 for heads
`1..group`. Each head still runs `attn_tile8_avx2` — the same 8-position tile, 8 independent dot
chains, the same vector `exp`, the same sequential max/denominator recurrence, the same
chunk-outer/position-inner accumulator order — so every head's words are unchanged. Heads share no
state, so interleaving them across tiles changes no head's sequence: the outputs are **bit for
bit** the per-head calls' (test below, and a standalone probe measured max|a−b| = 0 over
1,828 positions × 8 KV heads × 4 query heads).

The change is decode-only by construction (`tile == 0`); prefill keeps the per-column tile core
and the exact scan keeps `attn_head_fused`. The AVX2 body is added to `proof/regen.sh`'s opaque
list with the per-head scalar loop as its twin.

## Traced split and the dominant term

F32, 32 traced steps after a 1,795-token prefix:

| term | before, ms | share | after, ms | share |
|---|---:|---:|---:|---:|
| `attn.core` | 12.198 | 44.6% | 9.679 | 41.9% |
| `ffn` | 6.286 | 23.0% | 5.207 | 22.5% |
| `head` | 3.256 | 11.9% | 3.076 | 13.3% |
| `attn.proj_in` | 2.920 | 10.7% | 2.729 | 11.8% |
| `attn.proj_out` | 1.842 | 6.7% | 1.595 | 6.9% |
| wall | 27.37 | | 23.11 | |

`attn.core` falls 20.7% and stays dominant at 41.9% (1.72× ceiling), so the gate passes:

```text
TARGET attn.core: share 43.2%, ceiling 1.76x
exit 0
```

The standalone probe isolates the locality effect at 1.46× (34.237 → 23.432 ns per head-position,
head-outer against tile-outer, same arithmetic); the wall ratio is 1.19 because the pool dispatch
and trace sit outside the core's saving.

## Ceilings and the campaign target

Decode bytes/token at F32 KV remain 638,184,704 B, so the measured 100.74 GB/s STREAM mean still
caps decode at **157.8 tok/s**; the 0.5× target is **183.1 tok/s** (5.46 ms/token), which needs an
F16 KV carrier (428.6 MB/token, ceiling 235.0 tok/s). At 41.70 tok/s this change is 26.4% of the
F32 ceiling and **4.4× below the target**. Attention itself now costs 9.679 ms/token against
5.46 ms for the whole target token: the next lever is still `attn.core`, and after it the rank
stage (`ffn` + projections, 39% combined).

## Refusals

1. **"The i8 carrier improves."** Refused: 0.997×, inside the per-round spread; the i8 path
   (`attn_head_group_i8`) already reads each K/V row once for the group.
2. **"This is worth 1.46× like the probe."** Refused: the probe measures the core alone; the
   engine wall gains 1.189× because the traced attention delta (−2.519 ms) is 20.7% of a 12.198 ms
   leaf that holds 44.6% of the token.
3. **"It reaches 0.5× on decode."** Refused: 41.70 against 183.1 tok/s is 4.4× short, and the
   F32-KV bandwidth ceiling (157.8) is below the target before any kernel is timed.
4. **"The maintainer gate ran."** Refused: the pinned Aeneas/Charon/Lean toolchain is not
   bootstrapped on either host; `proof/regen.sh --check` was already failing on the committed
   extraction. The new body is opaque and its twin is the per-head scalar loop, but the extraction
   and the Lean axiom audit are not regenerated here and remain the owner's.

## What this does not cover

The laptop's wall (not run there), the GPU path, code generation for rustc/LLVM, the `unsafe`
container (no `unsafe` block is added or changed; the new body is AVX2 intrinsics inside the
existing pattern with a scalar twin), quality beyond the first token (the arithmetic is
bit-identical, so no PPL run is made), and the maintainer gate above.
