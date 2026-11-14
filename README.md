# apeiron

A Rust library for the algebra `R[x]/(x² − σ)`, a growable hash table built on it, and an inference
engine for GGUF models that runs on the CPU and, through Vulkan, on the GPU. The algebra and the
word-level arithmetic are specified and checked in Lean 4.

- **`Cell<R>`**: a pair `(a, b)` over a ring `R`, with multiplication `(a₁a₂ + σb₁b₂, a₁b₂ + b₁a₂)`
  and norm form `a² − σb²`. `σ = −1, 0, 1` gives the elliptic, parabolic and hyperbolic faces.
- **`freq::Table`, `CoordMap`, `FreqMap`**: flat tables with a spill region. Lookup is a fixed number
  of probes; allocation failure and size overflow are refused and change nothing.
- **`engine`**: GGUF loading, Q32.32 fixed-point kernels, AVX2 with a portable fallback, and a
  `gpu` feature that loads Vulkan at run time.
- **`proof/`**: Lean 4 extraction of the Rust word bodies with Aeneas and Charon, and the
  refinement theorems that link them to the canonical definitions in `proof/canonical/`.

The default build has no dependencies; Linux adds `libc` for thread affinity. Rust 1.89 or later.

## Use

```sh
git clone https://github.com/sanguinehost/apeiron
cd apeiron
cargo build --lib
```

```rust
use apeiron::{Cell, Sigma};

let x = Cell::<i64>::of(3, 2);
let y = Cell::<i64>::of(1, 5);
let p = x.mul(Sigma::Hyperbolic, y);
assert_eq!((p.a, p.b), (3 + 2 * 5, 3 * 5 + 2 * 1));
```

```rust
use apeiron::freq::Table;

let mut counts = Table::<u64, i64>::new();
for word in [7u64, 9, 7, 7, 9, 4] {
    counts.insert_with(word, 1, |old, new| old + new);
}
assert_eq!(counts.get(&7), Some(&3));
assert_eq!(counts.len(), 3);
```

The [book](book/src/introduction.md) covers installation, the algebra, the tables, running a model
on the CPU and the GPU, and the verification. Build it with `mdbook build book`; its examples run
under `mdbook test book -L target/debug/deps`.

## Inference engine against llama.cpp

Current figures. `pp512` is 512 prompt tokens per second, `tg128` is 128 generated tokens per
second. Ratios are apeiron over llama.cpp. The commit is the apeiron revision the figure was
recorded at; the figures are copied from those recorded measurements, not re-run for this
release.

| metric | model / device | apeiron | llama.cpp | ratio | commit | receipt |
|---|---|---:|---:|---:|---|---|
| pp512 | Qwen2.5-0.5B Q4_K_M / RTX 5080 | 113,099 tok/s | 63,018.5 tok/s (CUDA) | **1.795×** | `5e476afb` | quotient of recorded medians |
| tg128 | Qwen2.5-0.5B Q4_K_M / RTX 5080 | 973.8 tok/s | 880.9 tok/s (CUDA) | **1.105×** | `ecd1fcf1` | quotient of recorded medians |
| pp512 | Qwen3.5-0.8B Q8_0 / i9-14900KF | 917.82 tok/s | 416.41 tok/s (CPU) | **2.24×** (2.14–2.25) | `64cd46dd` | [`1792648053-qwen08b-q8-cpu`](bench/cpu/1792648053-qwen08b-q8-cpu.md) |
| tg128 | Qwen3.5-0.8B Q8_0 / i9-14900KF | 92.13 tok/s | 51.42 tok/s (CPU) | **1.80×** (1.70–1.81) | `64cd46dd` | [`1792648053-qwen08b-q8-cpu`](bench/cpu/1792648053-qwen08b-q8-cpu.md) |
| pp512 | Qwen3.5-0.8B Q4_K_M / i9-14900KF | 658.10 tok/s | 632.67 tok/s (CPU) | 1.02× (1.00–1.10) | `5064c500` | [`1792653496-qwen08b-q4km-kqc-live`](bench/cpu/1792653496-qwen08b-q4km-kqc-live.md) |
| tg128 | Qwen3.5-0.8B Q4_K_M / i9-14900KF | 84.19 tok/s | 60.39 tok/s (CPU) | **1.39×** (1.33–1.40) | `5064c500` | [`1792653496-qwen08b-q4km-kqc-live`](bench/cpu/1792653496-qwen08b-q4km-kqc-live.md) |

- CPU rows: llama.cpp `60bc8866b` at its best of 16 and 32 threads, median of 5 paired rounds, range
  is the minimum to maximum round, every round passed the quiet-host gate (busy cores < 1).
- GPU rows: the CUDA comparator is `llama-bench -ngl 99`. The prefill comparator was measured on a
  contended host, and the ratios are quotients of recorded medians, not paired-round medians.
- The Q4_K_M prefill row is parity (1.02×); decode is ahead. First-token agreement has not been run
  on the Q4_K_M files.
- Q4_K_M receipts were recorded from a working tree with uncommitted paths; the receipt names the
  source sha256.

### Earlier CPU receipts

More models and prompt lengths, Q8_0, measured earlier at revision `644190d7771f` against llama.cpp
`0324696b8` at its best of 8, 16, 24 and 32 threads. `tg` here is 64 generated tokens. Every round
passed the quiet-host gate (busy cores < 0.05).

| weights | pp tokens | rounds | pp ratio | tg ratio | receipt |
|---|---:|---:|---|---|---|
| Qwen3.5-0.8B | 200 | 5 | 2.17 (2.11–2.28) | 1.76 (1.52–1.77) | [`1790795647-w6e-vs-llama-r3`](bench/cpu/1790795647-w6e-vs-llama-r3.md) |
| Qwen3.5-0.8B | 512 | 5 | 2.45 (2.30–2.79) | 1.68 (1.55–1.79) | [`1790798160-w7-q35r2-pp512`](bench/cpu/1790798160-w7-q35r2-pp512.md) |
| qwen05b | 200 | 5 | 1.57 (1.56–1.64) | 1.47 (1.35–1.54) | [`1790798706-w7-qwen05b-r2-pp200`](bench/cpu/1790798706-w7-qwen05b-r2-pp200.md) |
| qwen05b | 512 | 5 | 1.56 (1.37–1.58) | 1.55 (1.45–1.59) | [`1790797618-w7-qwen05b-pp512`](bench/cpu/1790797618-w7-qwen05b-pp512.md) |
| qwen15b | 200 | 5 | 1.77 (1.74–2.00) | 1.10 (1.08–1.14) | [`1790798386-w7-qwen15b-r5-pp200`](bench/cpu/1790798386-w7-qwen15b-r5-pp200.md) |
| qwen15b | 512 | 3 | 1.91 (1.89–1.96) | 1.19 (1.06–1.21) | [`1790797926-w7-qwen15b-pp512`](bench/cpu/1790797926-w7-qwen15b-pp512.md) |

All six were measured at revision `644190d7771f`, source sha256 `0218e7d911a3…`, with the default `f32`
carrier. Each receipt holds the model and prompt digests, the machine, the compiler and the
per-phase trace. Reproduce one with:

```sh
cargo run --release --locked --example engine_profile -- \
  --model <Qwen3.5-0.8B-Q8_0.gguf> --ids tests/fixtures/qwen35-prompt2.ids \
  --pp 200 --tg 64 --reps 5 \
  --llama <path/to/llama-bench> --llama-threads 8,16,24,32 --max-busy 0.05 --label repro --out <dir>
```

All rows are one machine (the i9-14900KF above); `measured` here means that host and no other. The
GGUF files are not distributed; each receipt names the model by sha256. Only the first row's prompt
(`tests/fixtures/qwen35-prompt2.ids`, sha256 `894a166a…`) is in the repository. The other five
name their prompt ids by sha256 in the receipt and are not reproducible from a clone.

Throughput is accepted only alongside agreement with the comparator on the same model (first
tokens and perplexity); the book's [Results](book/src/engine/results.md) chapter gives the protocol
and the GPU commands (`gpu_decode_bench`, `gpu_ppl`). The hash table comparison against
`hashbrown` is not in these tables: its ratios are not claimed here.

## Quantisation: NanoQuant

NanoQuant (`SamsungLabs/NanoQuant` @ `a9e0a43`) stores a linear as binary low-rank factors
`W ≈ diag(s_post)·U·diag(s_mid)·V·diag(s_pre)`, `U` and `V` signs, plus a sparse residual on a few
input columns. The engine has one served route:

| piece | file |
|---|---|
| converter: arelath `nq_*` GGUF → pack + skeleton | `tools/nq_convert.py`, `tools/nq_gguf.py` |
| pack `APNQPK01` v2/v3: reader, writer, validation | `src/format/nq.rs` |
| product `y = s_post ⊙ U(s_mid ⊙ (V(s_pre ⊙ x))) + Σ_j salient[·,j]·x[idx[j]] + Σ_l s_res_l ⊙ (sign_l x)` | `src/engine/kern/nq.rs` |
| install and the one dispatch | `Engine::open_with_nq`, `Engine::step_proj_sched` |
| in-tree ADMM fit, not the source of these artifacts | `src/learn/nanoquant.rs`, `tests/learn_nanoquant.rs` |

The pack stores the sign planes as `I32` words (4 B per 32 signs), the three scale vectors as
bfloat16, the residual as binary16 and the indices as `I32`, all in the exporter's layout; the
kernel decodes the scales to Q32.32 once at load and keeps the signs bit-packed. Version 3 adds a
per-linear dense sign residual: `out × ⌈inn/32⌉` sign words and one bfloat16 row scale per layer.
A pack whose sections carry no residual is written as version 2; a reader that predates the
residual accepts it. Version 1 packs are refused: they carry no `s_mid`, residual or residual
indices. The fused `attn_qkv` factor is split
by output row into q, k and v sections that share `V`, `s_pre`, `s_mid` and the indices; the split
is exact for the fused factor and duplicates those sections, which is 91,450,016 B of the 8B pack
and 9.5% of its projection additions. On Linux the pack is mapped read-only and the sign planes
stay in the mapping; its pages are reclaimable and are not a second resident copy.

Measured with `cargo run --release --locked --example ppl`: the 8B rows on a rented vast.ai CPU
instance (Intel Xeon E5-2690 v4, 14 vCPU, 62 GB RAM, `APEIRON_THREADS=12`), the 0.6B row on the
laptop, over the raw wikitext-2 test file, consecutive non-overlapping windows, `Q8Act` trunk and
the packed linears exact. The receipt is `bench/cpu/1794553293-nq8b-q4km-ppl.md`:

| checkpoint | pack | linear-only rate | skeleton | PPL | run |
|---|---:|---:|---:|---:|---|
| `arelath/Qwen3-0.6B-nanoquant-GGUF` (sha256 `b0ffbcc3…`) | 63,370,960 B | 1.151033 b/w on 440,401,920 | 336,814,976 B | 81.100342 | ctx 128, second half (laptop); F16 image 16.056083 |
| `arelath/Qwen3-8B-nanoquant-GGUF` (sha256 `75d3f4f1…`) | 976,049,680 B | 1.124186 b/w on 6,945,767,424 | 1,329,625,696 B | 33.041986 / 28.307562 / 14.903005 | 2048 tokens at ctx 64 / 128 / 512; 992 / 1008 / 1020 scored (vast.ai) |

Both artifacts are the exporter's; neither is an in-tree fit. `--second-half` scores positions
`ctx/2 … ctx-2` of each window, llama.cpp's perplexity span. The 8B source GGUF is 2,214,297,824 B
and the served pair 2,305,675,376 B. Its skeleton holds 1,323,683,840 B of payload (a Q8_0
embedding and a different Q8_0 head, so both are kept, plus the norms) and 5,941,856 B of metadata,
mostly the tokenizer. In those runs, which read both files into memory, the resident weight budget is
2,299,725,424 B (skeleton payload plus the
pack's stored factors); the peak resident set is 2399.2 MiB at ctx 128 and 2567.1 MiB at ctx 512.

The bar is 7B-class at 1.18 b/w on the replaced linears within 2 GB resident, wikitext-2 perplexity
at most 1.15× Q4_K_M and mean 0-shot accuracy at least 0.90× of it. The v2 pack misses it on both
bits and quality: 1.12 b/w on the replaced linears but 1.756× Q4_K_M. The earlier receipt compared
the instruct-derived pack to `Qwen3-8B-Base`, which is a different checkpoint. Official
`Qwen3-8B-Q4_K_M.gguf` (sha256 `d98cdcbd…`) scores 16.124694 at ctx 128 over 1008 tokens on the
same engine. The ctx 64 and 512 rows of that receipt
(14.520989 / 6.557804, against the base checkpoint) are not re-measured against the official
model. The exporter's 1.7568× is over 64 random 128-token samples and includes each sample's
low-context first positions; the same ratio appears here only on the first 31 positions (1.7367×),
not once 64 or more positions of context are scored.

A version-3 pack adds per-linear dense sign-residual layers. Two refit layers measure
4,047,721,568 B (80.5% of Q4_K_M) and 16.295301 PPL at ctx 128 over 1008 tokens (1.011× Q4_K_M);
the v2 pack is 28.307562 (1.756×). One refit layer is 51.153269 (3.172×), and one layer without
the sparse refit is catastrophic (1.38M on the first window), so the refit and the second bit are
both required. The receipt is `bench/cpu/1794653685-nq8b-residual-ppl.md`. V3 therefore meets the
perplexity half of the bar at 80.5% of Q4_K_M's bytes, not the 1.18 b/w rate (it is 3.13 b/w on the
replaced linears), and the 0-shot half of the bar was not re-measured.

The same pair serves end-to-end from the laptop through `examples/generate --pack`, which installs
the factors with `Engine::open_with_nq`: with four threads and the exact kernel the 4,047,721,568 B
pair runs a 29-token chat under a 1 GiB cgroup at 0.09 tok/s (prefill 0.09 tok/s) on the 6.7 GiB
i5-10210U; a cold one-token start reads 3.22 GiB from NVMe and takes 28 s. The pack and the
skeleton are mapped read-only and the sign planes stay in the mapping, so the anonymous set is
171.3 MiB and the 4.05 GB pair is reclaimable file-backed page cache. Before the mapping change the
copied pair peaked at 4.33 GiB. `q8act` raises the anonymous set to 780 MiB, so `exact` is the
serving mode. The receipt is `bench/cpu/1794690665-nq8b-laptop-serve.md`.

`serve/nq-serve` is a sibling workspace (outside `src/` and `examples/`) that exposes the same pair
as an OpenAI-compatible `/v1/chat/completions` endpoint. Intent's `local` provider drove the 8B v3
pack through a one-turn `Hello World` with the assembled system prompt replaced by a 25-token
request: 457.390 s prefill and 215.983 s for four tokens at load average 35–49, ending
`finish_reason: stop` and Intent `outcome: complete`; the 0.6B pack took 60.095 s and 27.614 s for
nine tokens on the same path. The full 2,672-token prompt Intent assembles was not run. The receipt
is `bench/cpu/1794696786-nq8b-intent-loop.md`.

The 0.6B pack is 5.05× its F16 image (81.100342 against 16.056083) and 292.263889 against 55.166982
(5.2978×) on the exporter's 64-sample protocol, with 0-shot ratios of 0.61–0.95 per task; the 8B
0-shot ratio is 0.670× its reference across boolq, piqa, hellaswag, winogrande, arc_easy and
arc_challenge. The 36-layer reconstruction error is spread across the trunk: up, gate and down
carry 78.1% of the squared error and the largest layer (L33) is 3.19% of the total, so replacing
one layer's section with a dense tensor cannot close the gap on its own; the dense residual layers
above are what move it.

The 2 GB *stored* target is not met: the v3 pair is 4.05 GB, and even the v2 skeleton alone is
1.32 GB with 0.98 GB of factors, because the Q8_0 embedding and head dominate at this size. What
the mapping changes is *residency*: the process's anonymous set is 171 MiB and the pair is
file-backed page cache. What is established, by
`cargo test --release --test engine_nq -- --ignored` and by the 8B runs above, is narrower: a
7B-class pack of this form opens, serves and scores on a CPU instance, with the sign product as the
only projection path, prefill bit-identical to one-token steps, and logits stable across loads.

A pack is a CPU-executor artifact: the Vulkan path uploads GGUF payloads and does not read one.
Rust code generation, the kernel ABI, Vulkan drivers and the GPU remain outside the evidence, as
stated below.

## What is proved, tested and open

| statement | grade | gate |
|---|---|---|
| `Cell<i64>` multiply and norm form refine the canonical ring operations for every `σ` and input | proved | `word_mul_denote`, `word_norm_denote` in `proof/corpus/WordRefinement.lean` |
| the Lean axiom cone of every headline is within `propext`, `Classical.choice`, `Quot.sound` | proved | `proof/corpus/AxiomAudit.lean`, run by `lake build` |
| `Cell`, the tables, the engine and the SPIR-V fragment checker behave as documented on their boundary cases | tested | `cargo test --workspace --all-features --locked` |
| the SPIR-V fragment header check | proved | `Spirv.checkFragment_header` |
| the HashDoS bound on the tables | tested | `cargo test --test hashdos --locked` |
| speed against llama.cpp | measured | `cargo run --release --locked --example engine_profile`, receipts in `bench/cpu/` |
| the executed `freq::Table` body refines the canonical `Apeiron.Table` | open | |
| the whole SPIR-V fragment checker | open | |

Two extraction facts bound the proofs: the Lean files are generated from the Rust source by Aeneas
and Charon at pinned revisions (`proof/README.md`), and `proof/regen.sh --check` fails if the
committed extraction differs from a fresh one; `cd proof/corpus && lake build` checks the theorems.

## What this does not claim

- **Cryptography is unaudited.** `aes`, `chacha`, `sha256` and `crypto` are tested against
  known-answer vectors only and are not side-channel resistant (`aes` indexes its S-box by secret
  bytes). Do not use them to protect real data.
- **`unsafe` is mostly unproved.** Count it with
  `rg -n '\bunsafe\b' src --glob '*.rs' | rg -v ':\s*(//|\*)' | wc -l`; `SAFETY:` comments are
  counted with `rg SAFETY src --glob '*.rs' | wc -l`. Only the `src/table` window model is under a proof.
- **The GPU path is single-tenant.** Vulkan buffers run without `robustBufferAccess` and workgroup
  memory is not cleared between dispatches.
- **No claim below the extraction.** rustc and LLVM code generation, the kernel ABI, Vulkan drivers
  and GPU hardware are outside the evidence.
- **`i64` is not the real trichotomy.** `Z/2⁶⁴` carries the structural half of the algebra
  (commutativity, associativity, multiplicative norm) and not the classification; the book's
  [scalars](book/src/theory/scalars.md) chapter states where it stops.
- **Callbacks.** The tables assume `Eq`, `Clone` and merge callbacks that are total and stable; a
  callback that allocates, panics or diverges is outside the statements.

Security reports: [SECURITY.md](SECURITY.md). Licence: Apache-2.0 ([LICENSE](LICENSE),
[NOTICE](NOTICE)).
