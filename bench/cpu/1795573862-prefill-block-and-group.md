# nq: the wide tile assembled once per column group, and a pool-scaled prefill block

Criterion: **T2** (turn latency), partial: the cold 1,827-position prefill of the 0.6B pack, not a
1,000-token turn at 200k, not decode. Grades: `measured` for the isolated rate, the cold walls, the
phase split, the RSS and the instruction profile; `tested` for the wide-vs-pair and pooled bit
equalities; `model` for the laptop. Machine: OVH box
`ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets × 1 core × 1 thread, 31.3 GiB,
256 MiB L3, no cpufreq governor, no AVX-512, no VNNI), no GPU. Times UTC. The laptop, the only
device a timed claim may name, is **not measured** (Omissions).

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack
`~/models/nq-scratch/nq06-v3.nqp` (196 linears, 28 layers, 418,381,824 sign terms/token, 0.977441
b/w); harness `examples/prefix_solve.rs`, `--len 1827 --differ 205 --decode 0 --mb 440`.
Every run under `~/bin/box-run` (agent-guard, 16 GiB cap, 8 jobs); `touch src/lib.rs` before the
timed builds.

## What changed

Two code changes and a test probe, in three commits.

**1. The wide tile's readers are assembled once per column group, not once per output row.**
`Stage::cols8` builds `StageCols` (the eight `ColRef`s and the dequantized block sums) for one group;
`StageCols::dot_row` is the per-row dot over those views. The `V` and `U` closures of
`project_pooled_shape` now run group-outer, row-inner, so the assembly is paid once per group and the
group's `rank` (`V`) or `out` (`U`) rows read it. `wide_from` advances one group at a time, so a
group that refuses leaves every earlier group complete and the block loop starts at the refusal.
The arithmetic is unchanged: each column keeps its codes, shift vectors, `e_min`, block sum and
accumulation order. `the_wide_source_equals_eight_per_column_sources` and
`the_pooled_block_is_bit_equal_at_every_block` assert the equality.

The per-row admission was **10.34%** of the width-8 instruction stream in the previous profile
(`Stage::dot_row_cols8`, `+1.63%` in the caller); the group's `dot_row` is **1.61%** in the worker on
the same workload, and its caller row is below the 0.5% cut (`/tmp/opencode/box-insn.log`,
`/tmp/opencode/box-gate-final.log`). The 10.34% and 1.61% are rows of the `apeiron-q8act` worker
process under `APEIRON_THREADS=8`; the 1.63% and the post-change <0.5% are the `prefix_solve`
caller's own rows, and the `code_select` shares later in this receipt sum the two. That is the
mechanism the cold wall follows below, and it is the part of the tile that the previous receipt's
1.009× wall did not reach.

**2. The prefill block scales with the pool.** `prefill_block()` returns **1024** columns when
`width_global() >= 4` and **256** below it; `APEIRON_PREFILL_COLS` still overrides both. The
previous default was 256 at every width. At four and eight participants the pass pays enough
per-block fixed cost that the wider block wins; at one participant it is instruction-bound and the
coarser block loses 3.3% (below).

**3. A test-only probe for `quant_act`.** `probe_quant_act_rate` (in `src/engine/chart/tests.rs`,
`#[ignore]`) times `quant_act` on one token's pre-scaled `t` array (286,720 elements, the pack's 196
linears' in-features) and reports ns/element; under `perf stat` it also reports the instruction
count. It is commit 3, changes no production code, and is the measurement the remaining-distance
paragraph reads.

## Result

### Isolated projection rate

Same test binary rebuilt from the old (`5079a597`) and new trees, one `project_pooled_shape` call of
the `ffn` gate/up shape (3072×1024, rank 736), 228 columns = 0.687 G terms, block 8, two repetitions;
the tile-2 row is the same test's control and exercises neither `cols8` nor `dot_row`
(`/tmp/opencode/box-hoist.log` §2a):

| participants | arm | before G terms/s | after G terms/s | after/before |
|:---:|---|---:|---:|---:|
| 1 | tile 2 (control) | 25.77 | 25.80 | 1.001 |
| 1 | tile 8 | 21.97 | **28.30** | **1.288** |
| 8 | tile 2 (control) | 106.56 | 105.59 | 0.991 |
| 8 | tile 8 | 130.84 | **142.41** | **1.088** |

Each cell is the mean of the two repetitions. At one participant the tile was 0.85× the pair before
the change and is 1.10× after it: the per-row assembly, not the expansion, was what made the wide
tile lose on one core.

### Cold prefill, block 256: the hoist

Old binary = the committed tree (`5079a597`) built at `/tmp/prefix_solve-before`; new = the final
tree, both at `APEIRON_PREFILL_COLS=256`, alternating, no tracing (`/tmp/opencode/box-final2.log`
§2):

| participants | rep | before s | after s |
|:---:|---:|---:|---:|
| 8 | 1 | 13.096 | 12.061 |
| 8 | 2 | 13.156 | 12.040 |
| 8 | 3 | 13.019 | 12.101 |
| 4 | 1 | 17.836 | 16.438 |
| 4 | 2 | 17.759 | 16.440 |
| 4 | 3 | 17.909 | 16.589 |

Means: width 8 **13.090 → 12.067 s (1.085×)**, width 4 **17.835 → 16.489 s (1.082×)**. The new
arm's spread is 0.5% (width 8) and 0.9% (width 4), and every pair moves the same direction, so this
is a wall gain, not a sign test. Width 1 is not paired old-vs-final: the final width-1 block-256
wall is 41.422 s (below), against 44.529 s for the old binary in `box-phase.log`.

An intermediate tree also ran the pooled `SwiGLU` through the serial eight-lane body. That change
was reverted before this land: `swiglu_piece` and `swiglu_one_carrier` are extracted bodies in
`proof/ApeironWord.lean`, so their behaviour cannot change without regenerating the extraction,
which no host here can do. The tables in this receipt are the final tree, which keeps the committed
scalar body; the reverted arm measured 12.078 s at width 8, within the final arm's spread, so no
part of the gain below is attributed to it.

### Cold prefill, block width: the new default

Final tree, the same harness, block 256 vs 1024, three alternating repetitions at widths 8 and 4 and
two at width 1 (`/tmp/opencode/box-final2.log` §3, §4):

| participants | block 256 s | block 1024 s | 256/1024 |
|:---:|---:|---:|---:|
| 8 | 12.061 / 12.040 / 12.101 | 10.595 / 10.110 / 10.213 | **1.171** |
| 4 | 16.438 / 16.440 / 16.589 | 14.212 / 14.255 / 14.468 | **1.152** |
| 1 | 41.672 / 41.172 | 42.657 / 42.770 | 0.970 |

Means: width 8 12.067 → 10.306 s (177.28 tok/s), width 4 16.489 → 14.312 s (127.66 tok/s), width 1
41.422 → 42.714 s. The unset default (`APEIRON_PREFILL_COLS` absent) measures 10.209 / 10.384 s at
width 8 (mean 10.297), 14.304 / 14.675 s at width 4 (14.490) and 41.484 / 41.986 s at width 1
(41.735): the `OnceLock` selects 1024 at 8 and 4 and 256 at 1. The width-4 default mean is 1.2%
slower than the explicit 1024 arm (14.490 against 14.312 s), inside the two arms' spans of 2.6% and
1.8%.

On the old tree the width-4 block effect was absent (17.904 at 256, 17.785 at 1024, 1.007) and only
width 8 moved (13.196 at 256, 11.168 at 1024, 1.182; earlier hour, unpaired). On the final tree the
width-4 effect is 1.152, so it appears **only in combination with the hoist**, and this receipt does
not separate the two mechanisms.

### Phase split, width 8

`--phases` on the final tree, cold turn (`/tmp/opencode/box-final2.log` §5):

| phase | block 256 ms | share | block 1024 ms | share | 256/1024 |
|---|---:|---:|---:|---:|---:|
| `ffn` | 4480.480 | 37.2% | 4478.067 | 42.6% | 1.001 |
| `attn.proj_in` | 2240.135 | 18.6% | 2202.143 | 20.9% | 1.017 |
| `attn.core` | 2152.256 | 17.9% | 1785.194 | 17.0% | 1.206 |
| `residual` | 1197.708 | 9.9% | 408.609 | 3.9% | **2.931** |
| `attn.proj_out` | 895.082 | 7.4% | 797.317 | 7.6% | 1.123 |
| `norm` | 617.782 | 5.1% | 450.045 | 4.3% | 1.373 |
| `pool.wait` | 507.853 | 4.2% | 398.655 | 3.8% | 1.274 |
| `pass` | 11634.381 | 96.6% | 10155.496 | 96.5% | 1.146 |
| cold wall | 12.039 s | | 10.519 s | | 1.145 |

The 256-column turn runs nine passes (eight near-equal blocks of 229 or 228 columns over the 1,826
prefix tokens, plus a one-column step for the final token) and the 1024 turn three (two 913-column
blocks plus that step); `prefill_state_with_width` divides the prompt with `div_ceil` into near-equal
blocks. The `residual` call counts are 504 and 168 and the `pool.wait` call counts 11,045 and 3,143.
The phase that carries the block effect is `residual` — one serial
`add_cols` of the block's `mixed`/`ffn_out` into `hidden` — whose cost is per block (2.93× for 3× the
calls at the same total column count), with `norm` (1.373), `pool.wait` (1.274) and `attn.core`'s
per-pass tile setup (1.206) next. The dominant leaf `ffn` is flat (1.001): the weight stream does not
change, and the projection phases move 1.02–1.12× only through their own per-pass work. The block
width is a scheduling geometry, not a kernel.

### Memory

Peak RSS of the same cold run at width 8 (`/tmp/opencode/box-final2.log` §7): block 256
**1,280,488 KiB**, 512 **1,282,028 KiB**, 1024 **1,294,188 KiB**. The 1024 block costs **+13,700 KiB
(+13.4 MiB, +1.1%)** over 256: the extra column buffers of the wider pass. Peak resident is
1,294,188 KiB = **1.234 GiB**, the M1-relevant figure at this length, not a C1 figure.

### Quality

`examples/ppl --tokens 2048 --ctx 2048 --head-batch 16 --kernel q8act`, `APEIRON_THREADS=8`
(`/tmp/opencode/box-final2.log`):

| arm | PPL | mean NLL |
|---|---:|---:|
| block 256 | 572.461723 | 6.349946 |
| block 1024 | 572.461723 | 6.349946 |
| default (1024) | 572.461723 | 6.349946 |

The pair and tile arms of `bench/cpu/1795569181-nq-eight-column.md` measured 572.461723 on this
pack, head and token stream, so all three arms here are bit-identical to the unmodified tree's
reading. The two commits change no arithmetic: the equality tests above assert the tile word for
word, and the block width changes how the same columns are grouped. Every cold `prefix_solve` run in
the wall tables above printed the same first argmax, 315; the PPL command prints PPL and mean NLL.

### Ceiling

The dominant leaf at block 256 is `ffn` at 37.2% of the cold wall, so an infinitely fast `ffn` moves
that wall by at most `1/(1 − 0.372) = 1.59×`. That share is the width-8 phase split; the width-4
split was not run, so applying the same ceiling at width 4 is a `model`. Under it, the hoist arm
moves the width-4 wall 17.835 → 16.489 s = **1.082×**, 13.9% of the 0.59 excess; the block-width arm
moves it a further 16.489 → 14.312 s = **1.152×** by removing per-block work outside the projection
dot. The two together are 17.835 → 14.312 s = **1.246×** at width 4, 41.6% of that ceiling. `ffn`
itself did not move (1.001 at width 8), so the remaining distance is inside `ffn`'s own non-dot
share: the final-tree instruction profile puts `quant_act` at 9.76% + 1.44%, `project_row` at 5.77%
+ 0.97%, `TBuild` at 5.61% + 0.89% and `gemm_tile` at 5.79% + 0.81%, against
`code_select_row8_avx2`'s 22.14% + 3.29% (`box-gate-final.log`). The probe prices one of those rows:
`quant_act` over one column's `t` array (286,720 elements) is 1.126 ms on one core — 3.93 ns and
79.8 whole-process instructions per element, of which about 2.0 are the harness (an empty filtered
run is 2.8 M instructions of the probe's 114.4 M) (`/tmp/opencode/box-probe2.log`,
`/tmp/opencode/box-final2.log` §6) — and its body is in `proof/ApeironWord.lean`, so it is not a
lever this tree can move without the maintainer gate.

### Position against llama.cpp

The same box, `llama.cpp 049326a` `llama-bench -m Qwen3-0.6B-Base.f16.gguf -p 1827 -n 0 -t <w> -r 2
-o csv` (`/tmp/opencode/box-headtohead.log`): **262.90 / 265.31 tok/s** at width 4 (mean 264.11) and
**459.71 / 458.02** at width 8 (mean 458.87). This tree at the new default: 1,827 / 14.312 s =
**127.66 tok/s** at width 4 and 1,827 / 10.306 s = **177.28 tok/s** at width 8. The ratio is
**2.07×** and **2.59×**; with the old binary at block 256 the same comparison was 2.58× and 3.29×.
The llama runs are from the earlier hour; the engine A/B is paired, the cross-engine ratio is not.

## Commands

```sh
ARGS=(--model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
      --pack ~/models/nq-scratch/nq06-v3.nqp \
      --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
      --len 1827 --differ 205 --decode 0 --mb 440)
E=target/release/examples/prefix_solve

# cold prefill, old binary vs new, block 256, alternating (box-final2.log §2)
for w in 8 4; do for rep in 1 2 3; do for b in /tmp/prefix_solve-before "$E"; do
  APEIRON_THREADS=$w APEIRON_PREFILL_COLS=256 "$b" "${ARGS[@]}" | grep '^cold'
done; done; done

# old tree at block 1024 (box-hoist.log §2b; the 17.785 / 11.168 means)
for w in 8 4; do for rep in 1 2 3; do
  APEIRON_THREADS=$w APEIRON_PREFILL_COLS=1024 /tmp/prefix_solve-before \
    "${ARGS[@]}" | grep '^cold'
done; done

# old tree at width 1, block 256 (box-phase.log; unpaired context only)
for rep in 1 2; do
  APEIRON_THREADS=1 APEIRON_PREFILL_COLS=256 /tmp/prefix_solve-before \
    "${ARGS[@]}" | grep '^cold'
done

# the same wall at block 1024, final tree (box-final2.log §3)
for w in 8 4; do for rep in 1 2 3; do
  APEIRON_THREADS=$w APEIRON_PREFILL_COLS=1024 "$E" "${ARGS[@]}" | grep '^cold'
done; done
for rep in 1 2; do
  APEIRON_THREADS=1 APEIRON_PREFILL_COLS=1024 "$E" "${ARGS[@]}" | grep '^cold'
done

# unset default, final tree
for w in 8 4 1; do for rep in 1 2; do
  APEIRON_THREADS=$w "$E" "${ARGS[@]}" | grep '^cold'
done; done

# phase split, both block widths
for cols in 256 1024; do
  APEIRON_THREADS=8 APEIRON_PREFILL_COLS=$cols "$E" "${ARGS[@]}" --phases 2>&1 | grep -E '^ +\w|^cold'
done

# isolated rate, old (/tmp/apeiron-before) and new test binaries, two reps at w1 and w8; the
# test prints every block and tile, the logged rows are block 8
BIN=$(ls -t target/release/deps/apeiron-* | grep -v '\.d$' | head -1)
for w in 1 8; do APEIRON_THREADS=$w /tmp/apeiron-before \
  engine::kern::nq::tests::the_pooled_block_rate_by_block --ignored --exact --nocapture \
  2>/dev/null | grep -E 'tile 8 block 8|tile 2 block 8'; done
for w in 1 8; do APEIRON_THREADS=$w "$BIN" \
  engine::kern::nq::tests::the_pooled_block_rate_by_block --ignored --exact --nocapture \
  2>/dev/null | grep -E 'tile 8 block 8|tile 2 block 8'; done

# instruction share (final tree: box-gate-final.log; pre-change: box-insn.log)
APEIRON_THREADS=8 perf record -q -o /tmp/ins.data -e instructions -c 500000 \
  "$E" "${ARGS[@]}" >/dev/null 2>&1
perf report -i /tmp/ins.data --stdio --percent-limit 0.5

# quant_act element cost (5 reps of 286,720 elements behind the counter, three runs)
for rep in 1 2 3; do
  APEIRON_THREADS=1 perf stat -e instructions,cycles -x, "$BIN" \
    engine::chart::tests::probe_quant_act_rate --ignored --exact --nocapture
done

# pack geometry the probe's 286,720 / 124,544 come from: layers, linears, Σ inn, Σ rank,
# residual sections; the section walk asserts it ends at the file's last byte
python3 - <<'PY'
import os
import struct

path = os.path.expanduser("~/models/nq-scratch/nq06-v3.nqp")
b = open(path, "rb").read()
magic, ver, hlen, layers, kinds, n, _ = struct.unpack_from("<8s6I", b, 0)
assert magic == b"APNQPK01" and hlen == 32 and kinds == 7, (magic, hlen, kinds)
at = 32
inn = rank = res = 0
for _ in range(n):
    layer, kind, out, i, r, uw, vw, k = struct.unpack_from("<8I", b, at)
    at += 32
    if ver == 4:
        rl, group = struct.unpack_from("<2I", b, at)
        at += 8
    elif ver == 3:
        rl, = struct.unpack_from("<I", b, at)
        at += 4
        group = 0
    else:
        rl = group = 0
    assert uw == (r + 31) // 32, (layer, kind, uw, r)
    inn += i
    rank += r
    res += 1 if rl else 0
    u = out * uw * 4
    if group == 1:
        payload = u + out * 2 + out * k * 2 + rl * out * ((i + 31) // 32) * 4 + rl * out * 2
    else:
        payload = (
            u
            + r * vw * 4
            + i * 2
            + r * 2
            + out * 2
            + k * 4
            + out * k * 2
            + rl * out * ((i + 31) // 32) * 4
            + rl * out * 2
        )
    at += payload
assert at == len(b), (at, len(b))
print(f"version {ver} layers {layers} linears {n} sum_inn {inn} sum_rank {rank} res_sections {res}")
PY
# version 3 layers 28 linears 196 sum_inn 286720 sum_rank 124544 res_sections 0

# contributor gate (box copy; the GPU/SPIR-V tier needs the glslc pin and a device)
cargo fmt --all -- --check
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo test --workspace --locked --no-fail-fast
cargo test --lib --locked engine::
python3 tools/check-claims.py
python3 tools/book-check.py
CARGO_TARGET_DIR=$HOME/apeiron-mdbook cargo build --lib --all-features --locked
mdbook build book
mdbook test book -L "$HOME/apeiron-mdbook/debug/deps"

# peak RSS
for cols in 256 512 1024; do
  /usr/bin/time -f '%M' env APEIRON_THREADS=8 APEIRON_PREFILL_COLS=$cols \
    "$E" "${ARGS[@]}" >/dev/null
done

# llama.cpp baseline at the matched participant counts
LB=~/llama.cpp/build/bin/llama-bench
LM=~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf
for t in 4 8; do "$LB" -m "$LM" -p 1827 -n 0 -t "$t" -r 2 -o csv; done

# PPL
P=target/release/examples/ppl
PARGS=(--model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
       --pack ~/models/nq-scratch/nq06-v3.nqp \
       --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
       --data ~/models/wiki.test.raw --tokens 2048 --ctx 2048 --head-batch 16 --kernel q8act)
APEIRON_THREADS=8 APEIRON_PREFILL_COLS=256 "$P" "${PARGS[@]}"
APEIRON_THREADS=8 APEIRON_PREFILL_COLS=1024 "$P" "${PARGS[@]}"
APEIRON_THREADS=8 "$P" "${PARGS[@]}"
```

Logs `/tmp/opencode/box-final2.log` (the final-tree cold walls, phase split, RSS and PPL),
`/tmp/opencode/box-hoist.log` (the isolated tile rate and the old-tree block control),
`/tmp/opencode/box-insn.log` (the pre-change instruction profile),
`/tmp/opencode/box-gate-final.log` (the final-tree profile and the contributor gate),
`/tmp/opencode/box-headtohead.log` (llama.cpp). Scripts `/tmp/opencode/box-hoist.sh`,
`/tmp/opencode/box-insn.sh`, `/tmp/opencode/box-headtohead.sh`, `/tmp/opencode/box-final2.sh`,
`/tmp/opencode/box-gate.sh`.

## Gates

| step | command | result |
|---|---|---|
| 1 | `cargo test --workspace --locked --no-fail-fast` | two targets red, both the box's shaderc 2023.8 SPIR-V byte checks: the lib's `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` (674 passed, 1 failed, 54 ignored; `/tmp/opencode/box-lib-final.log`) and `tests/spirv_storage_race`'s `attn_coopmat_spirv_is_the_glsl` and `splitk_reduce_spirv_is_the_glsl` (2 passed, 2 failed); no other target red. `cargo test --lib engine::` is 347 passed, 48 ignored (`/tmp/opencode/box-gate2-final.log`). |
| 2 | `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | first run exit 101 on two exact `usize → f64` casts in the new probe; `#[allow(clippy::cast_precision_loss, reason = …)]` added, re-run exit 0, and exit 0 again on the final tree (`/tmp/opencode/box-lint-final.log`). |
| 3 | `cargo fmt --all -- --check` | exit 0 (`/tmp/opencode/box-lint-final.log`) |
| 4 | `python3 tools/check-claims.py` | exit 0 |
| 5 | `python3 tools/book-check.py` | 20 glossary entries, 0 findings |
| 6 | `mdbook build book` + `mdbook test book -L $HOME/apeiron-mdbook/debug/deps` | both exit 0 after `cargo build --lib --all-features` into that clean target dir (`/tmp/opencode/box-mdbook.log`) |
| 7 | `probe_quant_act_rate` under `perf stat` | 1.126 ms for 286,720 elements = 3.93 ns/elem, 79.8 whole-process instructions/elem (about 2.0 of them the empty harness), mean of three runs 1.125–1.127 ms (`/tmp/opencode/box-probe2.log`) |

The whole table is the box copy of this tree (`/tmp/opencode/box-gate-final.log`,
`/tmp/opencode/box-gate2-final.log`, `/tmp/opencode/box-lib-final.log`, `/tmp/opencode/box-mdbook.log`,
`/tmp/opencode/box-probe.log` and `/tmp/opencode/box-probe2.log`), host `indra-dev-1`. The
contributor gate runs where the toolchain and the model files are; the laptop cannot compile the
test targets in the guard's cap (the one local attempt timed out after an hour with nothing run).

## Review

Reviewer `gemini-4-hydrogen-high` via `agy --dangerously-skip-permissions -p "<a–e>" --model
gemini-4-hydrogen-high` (`/tmp/opencode/review-hoist.log`). It confirmed the group-outer/row-inner
rewrite keeps each column's arithmetic, accumulation order and the `wide_from`/partial-group refusal
contract, and that every index in `dot_row` and the caller is bounds-checked; it refused the items
below, each fixed in this receipt or the source.

| refusal | fix |
|---|---|
| `Stage::cols8` doc said bounds are checked once per group; `selected_row_cols` re-checks per row | doc now says the admission is re-checked on every `dot_row` call |
| llama.cpp baseline 259.67 / 455.01 with no command | 262.90 / 265.31 and 459.71 / 458.02 from `box-headtohead.log`, command added, ratios 2.07× / 2.59× |
| width-8 block-1024 rep 3 transcribed as `10.240` | re-measured on the final tree: 10.595 / 10.110 / 10.213, mean 10.306 s, 1.171× |
| width-1 isolated rate was rep 1, width 8 the 2-rep mean | both are 2-rep means: 25.77/25.80, 21.97/28.30 |
| `1.27 GiB` RSS | 1,294,188 KiB = 1.234 GiB; +13,700 KiB = +13.4 MiB |
| `## Commands` lacked block-1024 walls, width 1, both phase splits, `perf`, RSS and `llama-bench` | all added |
| width-8 `ffn` ceiling applied to the width-4 wall | labelled `model`; the width-4 split was not run (Omissions) |
| PPL and gate placeholders empty | filled from `box-final2.log`, `box-gate-final.log` and `box-probe2.log` |

The first review prompt also described a pooled-`SwiGLU` change that had been reverted before the
review (`chart/ffn.rs` is at HEAD); the receipt no longer mentions it. The first reviewer's refusals
are against the draft it read; the table above records the fix in the final tree.

The second review (`gemini-4-hydrogen-high`, `/tmp/opencode/review-final.log`) ran on the final tree.
It confirmed that the rewrite preserves each column's arithmetic, accumulation order, overflow and
the partial-group contract in both closures, and that the block change alters no arithmetic, and
refused:

| refusal | fix |
|---|---|
| the phase paragraph's "seven full blocks, a 35-column block" contradicts `prefill_state_with_width`'s near-equal `div_ceil` blocks | eight blocks of 229 or 228 columns and two of 913, plus the one-column final step; the 504 `residual` calls = 28 × 2 × 9 agree |
| `prefill_block`'s "pays it once instead of four times" | "once per 1,024 columns instead of once per 256" |
| the lib's SPIR-V red named `attn_coopmat_spirv_is_the_glsl` | the lib's is `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl`; the `attn_` and `splitk_` checks are the `spirv_storage_race` target's |
| gate counts came from `cargo test --lib` in `box-probe.log`, before the probe's `n` edit, and the workspace log truncated its lib line | gate re-run on the final tree (`/tmp/opencode/box-gate2-final.log`, `/tmp/opencode/box-lib-final.log`) |
| "The reference is 572.461723" had no artifact | cites the pair/tile rows of `1795569181-nq-eight-column.md` |
| the probe's three-run mean was shown as one command, and `perf stat` counts the whole process | Commands loop three runs; the text says whole-process and subtracts the 2.8 M empty-harness run |
| 124,544 had no pack artifact in the change | a pack-geometry command is in Commands and the probe doc cites it |
| old-tree block-1024, old width-1 and the gate commands were missing from Commands | all added |
| the book said "10.3% of the width-8 instruction stream" without the worker/caller split or block width, and gave the width-4 gain without its hoist caveat | the book names the worker and caller, block 256, the hoist-only width-4 gain and width 8 for RSS |
| `prefill_block` applied a one-model, one-host result as a general default | the doc says the rows are one model on one host and the threshold is the lowest measured winner |
| the isolated-rate shape was called "down" | `ffn` gate/up (3072×1024) |

The reviewer found no arithmetic or contract defect; its refusals are measurement naming, scope and
command provenance.

## Omissions

- The laptop is not measured. The box is eight single-core sockets: `residual`'s per-block cost and
  the pool's dispatch are cross-socket effects a single-socket laptop does not pay the same way, so
  the 1.152× block gain at width 4 is a `model` for the laptop until a quiet-window pair. The
  instruction-bound width-1 result (0.970×) is the one row that should transfer as a loss.
- Widths 2–3, 5–7 and block 512 were not timed; block 512 appears only in the RSS row.
- The width-1 hoist row is not paired old-vs-final: the final width-1 block-256 wall is 41.422 s
  (`box-final2.log` §3) against 44.529 s for the old binary in `box-phase.log`, an earlier hour.
- The width-4 block effect exists only with the hoist and is not attributed. The phase split was
  taken at width 8; the width-4 split would separate the same phases and was not run.
- The maintainer gate (`proof/regen.sh --check`, `lake build`) does not run on either host, and the
  committed extraction is **already stale with respect to HEAD**: `proof/ApeironWord.lean` annotates
  `step.prefill_block` at `step.rs` lines 385–394 and `prefill_cols` at 1800–1807, while HEAD has
  them at 410 and 1946 (this change moves the definitions to 419 and 1961); the file was last
  regenerated on 2026-10-20 (`7292279f`), before several `step.rs` commits. This change adds no
  extracted body: the Lean has no `nq` symbol (so the new `StageCols` is outside the extraction),
  `prefill_block` is an opaque `axiom` whose signature is unchanged, and `chart/tests.rs` is
  `#[cfg(test)]` and excluded. Regenerating would move the `prefill_block` annotation with the
  file; it would not add a body.
- The GPU/SPIR-V tier is not run (no device, no `glslc` pin).
- Decode is untouched; `prefill_block()` is read only by the prefill path. The head is untouched.
- The 8B is not measured. Its `out`, `inn` and rank change the rows per group; the body handles any
  multiple of 32 plus a scalar tail.
- The PPL window is 2,048 tokens at `--ctx 2048`, not a long-context or 7B result.
