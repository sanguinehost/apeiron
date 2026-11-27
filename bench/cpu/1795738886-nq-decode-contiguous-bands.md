# nq: one contiguous row band per participant — decode ffn −7.9%, projections −6%

Criterion: **T1** (decode) and **T2** (turn latency). The decode projection's `y` and `z` stages
split their output into `width·4` round-robin chunks; each participant therefore owned four
disjoint row bands and kept 32 weight-row streams live. Splitting into `width` chunks gives each
participant one contiguous band and eight streams. The paired run below measures the change against
`2f66b922` on the same host, model and pack.

Grades: `measured` for every wall, phase mean and ratio of walls; `model` for the `1/(1 − s)`
ceilings, the turn arithmetic and the rates. Nothing is `proved` or `tested` beyond the 29 `nq`
unit tests named below.

## Machine, binaries and command

Machine: OVH box `ubuntu@139.99.223.161` (8 single-core sockets, EPYC Milan, 8 vCPU, 32 GiB, 256 MiB
L3), `APEIRON_THREADS=8`, `APEIRON_KV_CARRIER=f32`, `~/bin/adg`. CPU-tier evidence; the timed
target is the laptop (i5-10210U), which did not run this measurement.

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 0.977441 b/w, 53,808,384 B).

| arm | tree | `phase_split` sha256 |
|---|---|---|
| base | `2f66b922` | `5cc6e017e8641e0c3ab41baa943ed6488fa236a67932e519f4c498bd037a64cf` |
| new | this commit | `04565924af5a08fd6783dd614c4de91be9e31389862800587e0b4ea3cf5f7853` |

Command, three interleaved rounds, base before new in each:

```sh
phase_split --model skeleton-q8head.gguf --pack nq06-v3.nqp \
  --steps 32 --prefix 1795 --target ffn
phase_split --model skeleton-q8head.gguf --pack nq06-v3.nqp \
  --steps 64 --prefix 0 --target ffn
```

Raw output: `1795738886-nq-decode-contiguous-bands.log` beside this file.

## The change

`NqLinear::project_with`'s `z` and `y` dispatches used `chunk = len.div_ceil(width·4)`, so a
3,072-row `y` stage was 32 tasks of 96 rows and the pool's round-robin partition handed each
participant tasks `k, k+8, k+16, k+24`: four row bands, each with eight row streams. `chunk =
len.div_ceil(width)` is eight tasks of 384 rows; each participant owns one band, and the row-block
inner loop keeps eight streams. The per-row arithmetic is unchanged: a row's sign dot reads its own
`words[r·stride..]` and folds the same 32-term blocks in the same order, so the split changes task
ownership only, not the result.

The same commit replaces the second `vpshufb(select_one, ·)` in each of the six sign-dot sites with
`vpminub(and(nibble, digit_mask), 1)` and drops the per-row bounds checks in the eight-row block.
That half was measured separately at width 8 before the split: ffn 5.217 → 5.177 ms, inside the
run's spread, and is **not** the cause of the change below; it is carried for the smaller
instruction stream (144 → 102 instructions per 8-row, 32-term block).

## Paired result

Means of three rounds, ms/token; the spread is min–max.

| prefix | term | base | new | Δ |
|---|---|---:|---:|---:|
| 1,795 | **decode** | **17.103** (16.88–17.39) | **16.810** (16.72–16.96) | **−1.7%** |
| 1,795 | `ffn` | 5.183 (5.086–5.370) | 4.771 (4.730–4.828) | −7.9% |
| 1,795 | `attn.proj_in` | 2.647 (2.617–2.700) | 2.474 (2.457–2.495) | −6.5% |
| 1,795 | `attn.proj_out` | 1.599 (1.579–1.631) | 1.502 (1.474–1.517) | −6.1% |
| 1,795 | `attn.core` | 3.717 (3.669–3.764) | 4.057 (4.040–4.087) | **+9.1%** |
| 1,795 | `head` | 3.096 (3.071–3.117) | 3.110 (3.078–3.132) | +0.5% |
| 0 | **decode** | **14.720** (14.67–14.76) | **14.170** (14.03–14.25) | **−3.7%** |
| 0 | `ffn` | 5.020 (5.008–5.041) | 4.701 (4.648–4.744) | −6.3% |
| 0 | `attn.proj_in` | 2.510 (2.508–2.512) | 2.388 (2.355–2.416) | −4.9% |
| 0 | `attn.proj_out` | 1.674 (1.669–1.684) | 1.577 (1.562–1.591) | −5.8% |
| 0 | `attn.core` | 1.512 (1.475–1.547) | 1.470 (1.451–1.489) | −2.8% |
| 0 | `head` | 3.102 (3.093–3.107) | 3.092 (3.085–3.102) | −0.3% |

At prefix 0 every leaf is neutral or faster and the decode falls 3.7%. At 1,795 the projection
leaves fall 6–8%, `attn.core` rises 9.1%, and the net is −1.7%: the slower attention is a cache
effect of the new projection partition (the bands change which core holds which rows when the KV
plane is read), not an arithmetic change, and it is reported rather than netted away.

## The turn

Prefill is unchanged by this commit: the pooled prefill path partitions columns by `band` in
`project_pooled_body`, not by these `chunk` sizes, and was not re-measured. Quoting the committed
split (`1795732570`): 3.903 s. With the new decode mean:

| half | ours | llama `049326a00` | ours / llama time |
|---|---:|---:|---:|
| prefill 1,827 | 3.903 s (468.1 tok/s) | 3.965 s (460.8 tok/s) | 0.984× |
| decode 128 | 2.152 s (59.5 tok/s) | 1.398 s (91.6 tok/s) | 1.540× |
| turn | **6.055 s** | **5.363 s** | **1.129×** |

The 0.5× turn target is 2.681 s, so the tree is **2.26× short**; the decode target is 5.46 ms/token,
so decode is **3.08× short**. `ffn` remains the decode dominant term at 28.3–33.3% (ceiling
1.39–1.50×); this change is inside that ceiling and does not close T1.

## Refusals

1. **"This closes T1."** Refused: 16.81 ms/token against 5.46 is 3.08×, and the dominant term's own
   ceiling is 1.50×.
2. **"The selector change is the win."** Refused: that substitution measured ffn 5.217 → 5.177 ms
   (inside the spread) before the split; the split is the cause of the 6–8% leaf movement.
3. **"Prefill moved."** Refused: prefill does not take `project_with`'s chunk sizes; it was not
   re-measured and is quoted from `1795732570`.
4. **"Attention improved too."** Refused: `attn.core` rose 9.1% at prefix 1,795. The net decode is
   still −1.7%, and the regression is named.
5. **"This names the laptop."** Refused: box only; the i5-10210U has not run this fixture.
6. **"The maintainer gate ran."** Refused: the pinned Aeneas/Charon/Lean toolchain is not
   bootstrapped and the committed extraction is stale. The change is in `src/engine/kern/nq.rs`,
   which the extraction includes; `cargo test --release --locked --lib engine::kern::nq` is 29
   passed, 0 failed, 4 ignored, and no Lean claim is made.

## What this does not cover

The laptop wall and its llama ratio; the 7B and 200k context; the GPU path; rustc/LLVM code
generation for arms the box did not take; and quality, which is unchanged by construction because
each row's sign dot is independent of which participant owns it.
