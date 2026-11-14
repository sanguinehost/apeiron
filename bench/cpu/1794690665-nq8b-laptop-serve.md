# Qwen3-8B v3 NanoQuant served end-to-end from the laptop

The 16.295301 PPL in `1794653685-nq8b-residual-ppl.md` is the full 36-layer model through
`examples/ppl`: the skeleton plus all 252 packed linears, not the residual layers alone. This
receipt is the other half of "end-to-end": the same full pair loaded by `examples/generate`,
tokenized prompt in, greedy tokens and detokenized text out, on the laptop.

## Machine and artifacts

Intel Core i5-10210U (4 cores / 8 threads, 1.6 GHz base, 3.9 GHz observed), 6.7 GiB RAM, 4 GiB
swap, NVMe, Ubuntu, no GPU. Governor `powersave`; the rates below are on a shared desktop, not a
quiet gate.

| file | bytes | sha256 |
|---|---:|---|
| `skeleton.gguf` | 1,329,625,696 | `3927454ca530ce26afdcd5dafe78f2d29d5ac006ea09151c8387d973a2ceaa28` |
| r2 pack `model.nqp` | 2,718,095,872 | `9c7cd2a88b396f2f7a0e2ca52ec9c67350ecad9b7c5bf590062c2a7037f81823` |
| `tokenizer.json` | 4,790,085 | `9ac67dd678dec2bb53ba7ffcc4524d24fb50a70f3996e9db37d13e7552c16485` |
| `skeleton.gguf` + pack | 4,047,721,568 | 80.5% of Q4_K_M |

The served tokenizer is the Qwen3 base vocabulary (`qwen3-0.6b-base-e8/tokenizer.json`, whose
151,643 base entries match the skeleton's `tokenizer.ggml.tokens` byte-for-byte) plus the 271
special tokens the skeleton carries at ids 151,665–151,935. Without them the first Qwen3 thinking
token, id 151,667 `<think>`, had no decode string and generation stopped with
`no token for id 151667`.

## Command

`examples/generate` gained `--pack`, which installs the pack through `Engine::open_with_nq`. It is
the native CLI path; `src/` and `examples/` still contain no network listener.

```
cargo build --release --locked --example generate

APEIRON_THREADS=4 systemd-run --user --scope -p MemoryMax=4600M -p MemorySwapMax=0 -- \
  target/release/examples/generate \
    --model  /home/lucas/models/arelath/nq8b/skeleton.gguf \
    --pack   /home/lucas/models/arelath/nq8b-v3s-r2/model.nqp \
    --tokenizer /home/lucas/models/arelath/nq8b/tokenizer.json \
    --chat --prompt "What is 17 times 23?" --n 8 --kernel exact
```

The cgroup is not optional on this box: the first r2 load ran outside one and the global OOM
killer selected the terminal (journal `07:58:11`, `Alacritty` 5.2G peak, 1.9G swap). With
`MemorySwapMax=0` an over-limit run kills only the scoped job.

## Measured

These four runs copy the pack and the skeleton into the process (the code before the mapping
change below), so their RSS is the resident pair plus scratch:

| run | prompt tokens | prefill | decode | peak RSS | wall |
|---|---:|---|---|---:|---:|
| v2, `q8act`, chat, `n=8` | 29 | 85.925 s, 0.34 tok/s | 23.160 s, 0.35 tok/s | 3,030,772 KiB | 111.5 s |
| r2, `exact`, raw `Hi`, `n=1` | 1 | 7.085 s, 0.14 tok/s | — | 4,472,232 KiB | 11.4 s |
| r2, `exact`, chat, `n=8` | 29 | 292.426 s, 0.10 tok/s | 93.684 s, 0.09 tok/s | 4,325,176 KiB | 391.5 s |
| r2, `exact`, raw `Hi`, `n=1`, 8 threads | 1 | 7.956 s, 0.13 tok/s | — | 4,228,660 KiB | 11.9 s |

The chat output was `<think>\nOkay, the user is asking` after 8 greedy tokens; the raw `Hi`
completion was ` is`. `--chat` wraps one ChatML user turn and stops on id 151,645
(`<|im_end|>`); it does not strip the thinking block, and there is no temperature.

Four threads beat eight: 7.085 s against 7.956 s for the one-token prefill, because the box has
four physical cores. The sustained 0.09–0.10 tok/s is the NQ trunk cost: 6.95 G fitted weights at
3.13 b/w, with two dense sign planes per linear read for every output row. The dense head and the
sampler are not the bottleneck.

## Memory, copied baseline

The served pair is 4,047,721,568 B. `exact` peaks at 4.33–4.47 GiB including the engine's scratch,
inside a 4.6 GiB cgroup with no swap. `q8act` does not fit: a one-token run under
`MemoryHigh=4G, MemoryMax=5G, MemorySwapMax=1.5G` printed `prompt_ids` and was OOM-killed at
3.9G RSS plus 340M swap. The extra resident set is the `q8act` tile/cache, about 0.7 GiB on this
model. The exact path is therefore the only laptop serving mode at the r2 footprint.

## Memory, file-backed weights

Both files are now mapped read-only and the sign planes stay in the mapping:

* `nq::open` parses the pack in place. `PackedLinear`'s planes are a `Plane<T>`, either an owned
  `Vec<T>` (a pack parsed from a byte buffer, the wasm path) or a slice of the pack's `MappedFile`.
  The 2.72 GB of sign words are never copied; the small scale and index planes are mapped and
  decoded once (~150 MB of Q32.32 scales and indices). `NqLinear` holds the `u`, `v` and `res`
  planes as `Plane<u32>`.
* `Engine::open` maps the skeleton on Linux through the same `MappedFile` and serves
  `Engine::payload` slices of it. `Engine::from_bytes` keeps the owned `Vec<u8>` path.

The anonymous set is then the decoded scales, the KV plane, the tokenizer and scratch; the 4.05 GB
pair is file-backed page cache the kernel can evict and re-fault from NVMe.

```
APEIRON_THREADS=4 nice -n 5 systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- \
  target/release/examples/generate \
    --model  /home/lucas/models/arelath/nq8b/skeleton.gguf \
    --pack   /home/lucas/models/arelath/nq8b-v3s-r2/model.nqp \
    --tokenizer /home/lucas/models/arelath/nq8b/tokenizer.json \
    --chat --prompt "What is 17 times 23?" --n 8 --kernel exact
```

`memory.stat` was sampled every 0.2–0.5 s inside the cgroup; `anon` and `file` are the maxima of
those samples, not `/usr/bin/time` RSS (which counts resident mapped pages too).

| run | cgroup cap | cache | max anon | max file | prefill | decode |
|---|---:|---|---:|---:|---|---|
| `exact`, raw `Hi`, `n=1` | 1 GiB | cold | 157,110,272 B (149.8 MiB) | 910,778,368 B (868.6 MiB) | 27.899 s | — |
| `exact`, raw `Hi`, `n=1` | 1 GiB | cold | 157,196,288 B (149.9 MiB) | 912,830,464 B (870.5 MiB) | 28.034 s | — |
| `exact`, raw `Hi`, `n=1` | 1 GiB | warm | 156,909,568 B (149.6 MiB) | 274,575,360 B (261.9 MiB) | 6.652 s | — |
| `exact`, raw `Hi`, `n=1` | 2 GiB | warm | 157,868,032 B (150.6 MiB) | 1,322,749,952 B (1.23 GiB) | 8.718 s | — |
| `exact`, chat, `n=8` | 1 GiB | warm | 179,646,464 B (171.3 MiB) | 882,585,600 B (841.7 MiB) | 340.916 s, 0.09 tok/s | 90.837 s, 0.09 tok/s |
| `q8act`, raw `Hi`, `n=1` | 1 GiB | — | 817,799,168 B (780.0 MiB) | 912,719,872 B (870.5 MiB) | 14.841 s | — |

"Cold" means the pair was dropped from the page cache with
`posix_fadvise(POSIX_FADV_DONTNEED)` immediately before the run; "warm" means the page cache held
it from an earlier run. The two cold runs are consecutive and identical apart from timing noise,
and the warm one is the earlier measurement from the same binary.

The 29-token chat printed `<think>\nOkay, the user is asking` under the 1 GiB cap, the same output
text as the copied run; `format::nq::tests::a_mapped_pack_equals_the_owned_parse` checks the mapped
parse against the owned parse of the same bytes. Prefill rises from 292.4 s in the copied run to
340.9 s because the mapped weights are re-faulted; decode is unchanged within the run-to-run spread
(90.8 s against 93.7 s for 8 tokens).

The cold/warm gap is the page cache, not the cap. A cold one-token run reads about 3.22 GiB from
NVMe (6.75M–6.76M 512-byte blocks, 17.1k major faults) and takes 27.9–28.0 s; the same run with the
pair already cached takes 6.65 s. The earlier 6.7 s against 8.7 s for the 1 GiB and 2 GiB caps was
two warm runs and is not a cap effect. At a 1 GiB cap the resident file pages are capped near
0.9 GiB, so a fresh process cannot keep the whole 4.05 GB pair cached; the chat run's 340.9 s is a
warm-cache rate, and a cold chat adds the same ~3.2 GiB stream to the first prefill.

A 1 GiB cap leaves about 0.8 GiB of anonymous headroom for KV (the file pages are reclaimable); the
cgroup never swapped.

`q8act` is not the footprint choice here: its tile/cache is 780 MiB of anonymous memory against
150–171 MiB for `exact`, and it does not remove the file-backed stream. Its row's cache state was
not controlled, so only the anonymous peak is used; `exact` remains the serving mode.

## The KV is now the binding constraint

The engine's `KvState` is resident Q32.32: `2 · n_head_kv · head_dim · 8` bytes per token per
layer. This model has 36 layers, 8 KV heads and head_dim 128, so one token costs
`36 · 2 · 8 · 128 · 8 = 589,824 B` (576 KiB, 0.5625 MiB) across the trunk:

| context | resident KV |
|---:|---:|
| 128 | 72.0 MiB |
| 2,048 | 1.125 GiB |
| 4,096 | 2.25 GiB |
| 40,960 (the model's `context_length`) | 22.5 GiB |

The weight side now leaves about 1.8 GiB of a 2 GiB process cap free, an upper bound of ~3.3k
tokens of dense KV; the page cache competes for the same cap, so the practical figure is lower. At
the 1 GiB cap used above, the bound is ~1.4k tokens. Long context does not fit by adding weights
headroom; it needs the KV tier.

`src/engine/step/cold_store.rs` (`ColdStore`, grade `tested` by `tests/engine_cold_store.rs`) is a
step towards it: one sealed record per `(layer, kv-head, block)` in the canonical flat/spill
`Table`, with an exact `gather` and a `Missing` refusal instead of a dense-plane fallback. It is
**not** on disk: the table's spill region is still memory, and its payload is
`2 · head_dim · 8` bytes per stored position, the same Q32.32 words as the dense plane. It moves
KV out of the dense plane; it does not shrink it. Nothing in `attn.rs` calls it yet.

The coldkv port (`Ptyktos/fold`, `bedrock/src/serve/model/cold.rs`, `kvpool.rs`) is a disk tier plus
a hot pool: seal a block, evict it to a file-backed record, and at each step gather the blocks the
attention reads, with the delta-gather pool holding the hot working set. The
blocker is not the store but the selector: exact dense attention scores every key, so a disk tier
without a sound pruning rule reads the whole KV every token. The bedrock prototype measures 89.9%
one-step and 94.9% cumulative carryover on int8 KV (`1 B/word`) with a per-block Cauchy–Schwarz
bound; the certificate is per step, not for the whole run, and Apeiron's KV is Q32.32 (`8 B/word`),
so the same fraction moves eight times the bytes. Porting it changes `src/engine/attn.rs`, which is
extraction source (`proof/extract/lib.rs` `#path`), so it needs the maintainer gate.

## Scope

This is `measured` for: the full 36-layer r2 pair opens and generates on the laptop with both files
mapped, the 4.05 GB pair runs a 29-token chat under a 1 GiB cgroup with 171.3 MiB of anonymous
memory (warm page cache; a cold one-token start reads 3.22 GiB from NVMe in 27.9–28.0 s), and the
native CLI rate is 0.09–0.15 tok/s. The same output text as the copied run is the
check that the mapping did not change the computation. It is not a quiet-machine throughput claim,
not `q8act`, not the browser wasm artifact (which holds the pack resident and is at the 32-bit 4 GiB
limit), not an HTTP server, and not a long-context or cold-KV claim. The NQ serving path is not in
the Charon/Aeneas extraction (`format::nq` has no mirror in `proof/extract/lib.rs`), so no
refinement claim is made for it.
