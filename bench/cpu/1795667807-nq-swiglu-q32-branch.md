# nq: the SwiGLU Q32 conversion, one branch per eight lanes

Criterion: **T2** (turn latency), partial: the cold 1,827-position CPU prefill of the 0.6B pack at
block 1024, width 8, the scope of `1795660694-nq-select-sign-dot.md` and
`1795664197-nq-prefill-gap.md`. The final arm returns PPL `53.991898` on the same stream as the
control, so this is not a quality claim. It is not a 7B result, not a 200k turn and not a decode
result.

**Refused under the sub-step rule.** `ffn` is the largest prefill leaf (34.5% of the traced cold
wall, `1/(1 − 0.345) = 1.527×`), but the `SwiGLU` conversion is 3.88% of cycles against the
select's 22.82%, and the `ffn` sub-steps were not timed in ms before it was chosen. The paired wall
cut is reported as a measured branch-miss reduction, not a dominant-term-targeted speed change;
it does not close T2. Refusal 1 in `## Review` stays open and no owner waiver is recorded.

Grades: `measured` for every wall, paired ratio, PPL, `perf` count and `perf` share below; `model`
for the absolute per-symbol miss counts (share × the final mean) and the cycle/ceiling projections,
whose formula is stated; `tested` for the equality tests and the mutation. No `proved` claim:
`src/engine/chart/ffn.rs` is reachable from the extraction input `src/engine/chart/mod.rs`, but the
changed function is the explicit `--opaque` entry `crate::chart::ffn::swiglu_f32_avx2_8_into`
(`proof/regen.sh:284`), its safe twin `swiglu_one_f32_scalar` is unchanged, and the maintainer gate
did not run here (Omissions).

Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets × 1 core × 1 thread,
31.3 GiB, 256 MiB L3, 2.4 GHz base, no AVX-512, no VNNI), no GPU. The laptop — the only device a
timed claim may name — is **not measured**. Times UTC; every job under `~/bin/adg run --` (16 GiB
cap, 8 jobs, swap zero). The box is shared; the paired ratio is the graded quantity and the absolute
wall is quoted beside it. The raw lines behind this receipt's measurements are in
`1795667807-nq-swiglu-q32-branch.log`; the 0.082 instructions/MAC and the 18-cycle miss cost are
quoted from `1795664197-nq-prefill-gap.md` and keep that receipt's log.

**Base.** The box's shared tree carried another lane's uncommitted narrow path in
`src/engine/num/value.rs` (`51de84e3…`, against HEAD `1fe984e0…`) and a cosmetic
`src/learn/nanoquant.rs` edit (`9afecaac…`, against `d3af6a12…`) during an earlier capture. Every
number here is from a private clean copy of committed HEAD `0b294ba7` with those two files at their
committed hashes and the two uncompiled files `src/engine/ffn.rs` and `src/engine/nq.rs` removed.
The control is `git show HEAD:src/engine/chart/ffn.rs` (`7f448386…`); the treatment is this tree
(`95679bce…`). An earlier shared-tree window, taken before this base was isolated, is **not
quoted**; it measured this change on top of the other lane's `mul_q32` path.

## Commands

```sh
# clean base: private copy of HEAD with this diff, built twice
git show HEAD:src/engine/chart/ffn.rs > ffn.head.rs
install -m 644 ffn.head.rs  src/engine/chart/ffn.rs && touch src/lib.rs
cargo build --release --locked --example prefix_solve --example ppl   # -> cleanctl
install -m 644 ffn.final.rs src/engine/chart/ffn.rs && touch src/lib.rs
cargo build --release --locked --example prefix_solve --example ppl   # -> cleanfin
# cold prefill, under perf stat -x, -e
# branches,branch-misses,instructions,cycles -- <the line below>
~/bin/prefix_solve.<arm> --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 0 --decode 0 --mb 440
# the same line plus --phases for the split
# PPL, same stream and window for both arms
~/bin/ppl.<arm> --model <same gguf> --pack <same nqp> --tokenizer <same json> \
  --data ~/models/wiki.test.raw --tokens 8192 --ctx 512 --head-batch 16 --kernel q8act
# llama.cpp at 049326a00, before and after the pairs
~/llama.cpp/build/bin/llama-bench \
  -m ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf -p 1827 -n 0 -t 8 -r 3
# branch-miss and cycle shares
perf record -q -F 4000 -e branch-misses:u -o <data> -- <the prefill line>, then
  perf report -i <data> --no-children --stdio --sort symbol --percent-limit 1.0
# same with -e cycles:u
# symbol shape and instruction count
objdump -d --no-show-raw-insn --disassemble=<symbol> ~/bin/prefix_solve.<arm>
# refused-arm symbol byte size and opcode count
nm -S --size-sort ~/bin/prefix_solve.<arm> | grep code_select_row8_avx2
objdump -d --no-show-raw-insn --disassemble=<symbol> ~/bin/prefix_solve.<arm> | grep -c vpmaddubsw
# pack header: 32 B file header, then 36 B linear headers interleaved with payload;
# the "### pack header" block in the log carries the script and its output
# equality sweeps (release, clean copy)
cargo test --release --locked --lib q32_from_f32_bits
cargo test --release --locked --lib -- --ignored q32_from_f32_bits_is_total
```

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 28 layers, 0.977441 b/w; the pack
header in the log reads `k = 0`, `res_layers = 0` for every linear). `--differ 0` makes the warm
and shared turns full cache hits, so the run is one cold prefill. `APEIRON_THREADS` and
`APEIRON_PREFILL_COLS` were unset: `pool::global` defaults to 8 (seven workers plus the caller,
capped at the pinnable cores, `src/engine/chart/pool.rs:463-488`) and `prefill_block` defaults to
1024 at four or more participants (`src/engine/step.rs:472-487`). The width-8, block-1024 labels in
this receipt are those defaults, and the log's command is the line that ran. The timed builds were
rebuilt before their runs (`touch src/lib.rs`) and every job ran under `~/bin/adg run --`.

The arms, by `prefix_solve` sha256: `cleanctl` `2998343b…` (HEAD `ffn.rs` `7f448386…`) and
`cleanfin` `57476450…` (this tree, `ffn.rs` `95679bce…`). Their PPL binaries are `0c7c9913…` and
`5612d855…`. `swiglu_f32_avx2_8_into` is 201 instructions with 14 conditional branches on the
control and 358 with 15 on the treatment; `swiglu_slice` 491 with 46, then 520 with 51;
`code_select_row8_avx2` 610 with 35 and 3,054 B in both. The treatment binary and the test gate
were built from the same `ffn.rs` (`95679bce…`), the exact working tree; no separate test build
exists between the timed source and the source this receipt carries.

## What changed

`src/engine/chart/ffn.rs` only. `swiglu_f32_avx2_8_into` forms eight binary32 products and
converted each to Q32 through `f32_to_q32_checked`, whose `F32::to_q32`
(`src/engine/num/fp.rs:186`) returns `Option<i64>`: a data-dependent sign branch and a range branch
per lane. The control's eight-lane body carries 14 conditional branches in its disassembly. On the
unmodified tree the symbol owns **22.69%** of all branch misses and its scalar wrapper
`swiglu_slice` another **3.16%**, 25.85% together.

The conversion now calls `q32_from_f32_bits`, which returns `(i64, bool)` and forms both parts as
masks:

- the sign is `(mag ^ neg_mask).wrapping_sub(neg_mask)` with `neg_mask = 0 − neg`;
- validity is `(e != 0xff) & (k <= 40) & mag_ok`, with `mag_ok` the `i64::MIN` exception for a
  negative magnitude of exactly `2^63`;
- the eight validity bits are folded into one `u8` mask and tested once, `invalid == 0`.

If any lane is invalid, the eight products go through `f32_to_q32_checked` in lane order, so the
first error and its index are those of the scalar oracle. The mechanism is the live-lane scan of
`src/map/hashtrinity/face.rs:1140-1155` — glance eight lanes, form a bitmask, take one branch per
eight instead of one per lane — applied here to eight Q32 decodes instead of eight table slots.
`F32::to_q32` remains the oracle and the production refusal route; this is not a second `SwiGLU`
path, and the AVX2 body is still the one `SwiGLU` kernel with `swiglu_one_f32_scalar` as its safe
twin.

### The disassembly

`swiglu_f32_avx2_8_into` on `cleanctl` is **201 instructions with 14 conditional branches** at
`+521 … +833`; on `cleanfin` it is **358 instructions with 15**, the first at `+1266` (the
`invalid == 0` test) and the other 14 at `+1291 … +1685`, in the cold scalar fallback. The hot path
inside the symbol executes exactly that one conditional branch; the caller's loop back edge is
separate. The per-lane sign and `Option` tests are gone. The eight-lane decode lowered to 4-lane
integer SIMD (`vpcmpgtq`, `vblendvpd`, `vpsllvq`), not to a scalar lane loop; that is the compiler's
choice, and the source states the mask rather than relying on it. The caller `swiglu_slice` on
`cleanfin` is 520 instructions with 51 conditional branches (491 with 46 on `cleanctl`): its
per-group `while`, the `match` on the batch result and the scalar tail still branch, and the tail is
data-independent per group, so the measured cut is the per-lane part, not every branch in the
wrapper.

## Equality and the mutation

| test | domain | result |
|---|---|---|
| `q32_from_f32_bits_equals_to_q32` | all 256 exponents × 7 mantissas × both signs, plus 100,000 LCG words | pass |
| `q32_from_f32_bits_is_total` | every `u32` bit pattern, `2^32` (`#[ignore]`, release) | pass, 19.23 s |
| `swiglu_f32_avx2_matches_scalar` | 1,000 random 8-lane groups, bit for bit | pass |
| `swiglu_f32_avx2_refuses_like_scalar` | failing lanes 1 and 7: error lane and written prefix equal to the scalar twin | pass |
| `swiglu_slice_refuses_at_the_same_lane` | 16 lanes, failing lane 10 through the production caller | pass |

One mutation on the exact final source (`95679bce…` → `243951a1…`), applied with a backup and
restored:

- `let value = mag as i64` (drop the sign mask) turns `q32_from_f32_bits_equals_to_q32` red at bits
  `0xaf000000` (`left: Some(-1)`, `right: Some(1)`) and `swiglu_f32_avx2_matches_scalar` red with
  sign-flipped lanes. Restoring returns `ffn.rs` to `95679bce…` and both tests green.

The tests cannot pass on the broken source, so they carry weight.

## The counts

`perf stat`, all 13 interleaved pairs:

| metric | `cleanctl` | `cleanfin` | Δ | sd (ctrl / final) |
|---|---:|---:|---:|---:|
| branches | 35.878 G | 34.640 G | −1.237 G (−3.45%) | 0.118% / 0.067% |
| branch misses | 344.65 M | 265.47 M | **−79.18 M (−22.97%)** | 0.174% / 0.132% |
| instructions | 330.54 G | 324.93 G | −5.61 G (−1.70%) | 0.052% / 0.029% |
| cycles | 139.90 G | 138.21 G | −1.69 G (−1.21%) | 0.923% / 0.559% |

The two miss distributions do not overlap: control 343.7–346.1 M, final 264.6–265.8 M, a gap of
77.9 M against spreads of 2.4 M and 1.2 M. The count is the settled result; the wall is the
noisier one.

The 18 cycles per miss is the gap receipt's `model`. With the measured cycles, 79.18 M misses are
1.425 G cycles, **1.019%** of the control's 139.90 G. The final arm's remaining 265.47 M are 4.778 G
cycles, **3.457%** of its 138.21 G: removing every remaining miss at that model is a **1.036×**
ceiling. The select's count, `54.17% × 265.47 M = 143.8 M` (`model`), is 2.589 G, 1.873%, a
**1.019×** ceiling. The branch-miss avenue is therefore nearly exhausted; the gap to llama is not
made of the remaining misses.

## The wall

Thirteen interleaved pairs, control and treatment alternating in one window:

| pairs | mean ratio | sd | 95% interval | control mean | final mean |
|---|---:|---:|---:|---:|---:|
| 13, 10 below 1 | **0.9882** | 0.0171 | [0.9778, 0.9985] | 4.828 s | 4.770 s |

The mean pair ratio is a **1.18%** wall reduction; the ratio of the window means is 0.98792,
**1.21%**, 0.0583 s. The interval is the t-interval over the 13 pair ratios; its upper bound 0.9985
leaves a 0.15% lower bound on the reduction. A one-sided sign test on 10 of 13 gives `p = 0.046`.
The median ratio is 0.9884. The smallest ratio, the largest single reduction, is pair 2 at 0.9494:
the control run at 5.038 s is the window's only wall above 5 s and the treatment before it read
4.783 s. Keeping it is the registered rule, and it is the pair that carries most of the mean's
distance from 1.

This is the number this receipt quotes, on the clean base. The paired labels are in the log; no
pair was dropped.

## Where the misses went

Branch-miss shares, same prefill line, `perf record -e branch-misses:u`:

| symbol | `cleanctl` | `cleanfin` |
|---|---:|---:|
| `code_select_row8_avx2` | 40.55% | 54.17% |
| `swiglu_f32_avx2_8_into` | 22.69% | — |
| `rms_norm_fixed` | 10.27% | 13.80% |
| `swiglu_slice` | 3.16% | — |
| `ColPipeline::run` | 3.28% | 4.43% |
| `q_mul` | 3.24% | 4.50% |
| `attn_head_group_tile_f32` | 3.00% | 4.43% |
| `fixed_sin_cos` | 2.30% | 3.12% |
| `div_fac` | 2.24% | 3.29% |
| `attn_core_tile_avx2_inner` | 1.16% | 1.61% |
| `rms_with_gain_slice` | — | 1.00% |

The two `SwiGLU` symbols together were 25.85% of control misses and neither appears above the 1%
reporting threshold in the final list. The measured total cut is 22.97%; the 2.9-point difference
between that share and the total is not located by these profiles (sampled at 4 kHz, symbols below
1% not shown), so the receipt claims the measured count, not the share. The select
owns 54.17% of what remains and `rms_norm_fixed` 13.80%; below them `q_mul` 4.50%,
`ColPipeline::run` 4.43% and `attn_head_group_tile_f32` 4.43% sit within a point of each other, and
the fixed-point trig is largely the same per-column work seen from different frames. The cycle
profile puts the select at 22.82% control and the SwiGLU symbol at **3.88%**.

## The phase split and the target

`prefix_solve --phases`, one traced cold run per arm:

| phase | `cleanctl` ms | share | `cleanfin` ms | share |
|---|---:|---:|---:|---:|
| `ffn` | 1643.267 | 34.5% | 1587.014 | 34.0% |
| `attn.core` | 1514.089 | 31.8% | 1506.892 | 32.3% |
| `attn.proj_in` | 619.864 | 13.0% | 609.367 | 13.0% |
| `attn.proj_out` | 243.860 | 5.1% | 240.446 | 5.1% |
| `norm` | 156.076 | 3.3% | 154.132 | 3.3% |
| `residual` | 142.237 | 3.0% | 134.858 | 2.9% |
| `embed` | 10.187 | 0.2% | 10.249 | 0.2% |
| `head` | 5.452 | 0.1% | 5.500 | 0.1% |
| `pass` | 4374.711 | 91.9% | 4286.462 | 91.7% |
| `pool.wait` | 257.381 | 5.4% | 276.159 | 5.9% |
| cold wall | 4762 ms | | 4672 ms | |

The phase is `ffn`, the largest leaf on the unmodified tree at **34.5%** (a **1.527×** ceiling);
`attn.core` is second at 31.8%. `phase_split` measures a decode token, so the split for a prefill
leaf is `prefix_solve --phases` on the unmodified tree, as in the select receipt.

Within `ffn`, the chosen target is the branch-miss owner, and that choice is **refused** by the
sub-step rule: the `SwiGLU` Q32 conversion and its wrapper are **25.85%** of all branch misses and
the symbol is **3.88%** of cycles (a 1.040× ceiling), while the projection loop is larger (the select
alone is 22.82% of cycles) and the `ffn` sub-steps were not timed in ms. The conversion's own wall
share is not measured separately. The traced pair is one run per arm: the `ffn` leaf falls 56.25 ms
(3.42% of its own time, 1.18% of the traced wall), and the graded 13-pair mean is 1.18% (1.21% by
the ratio of window means). The 1.18% is just above the 1.019% miss model and below the whole-run
1.70% instruction cut. It is a branch-miss result, not closure of the leaf or of T2.

## Refused arms

Both are reverted; `kern/nq.rs` is back at HEAD. Both were built on the earlier shared-tree base (it
included the other lane's `mul_q32` path), so their absolute counts are comparable only within that
base, each against `q32mask` built there; their symbol shapes are static.

- **`brmask`** (`6e114a52…`), the select's word-span split plus a pointer loop. The select symbol
  falls 610 → 586 instructions and 35 → 33 conditional branches; one perf run gives 242.91 M
  misses, +0.54 M (+0.22%) against `q32mask`'s 242.36 M, and 322.07 G instructions, the same as the
  Q32 arm's 322.00 G. The disassembly still holds a block-loop bound branch (`jae 1a4e51`) and a
  pointer-loop back edge (`jne 1a4dbf`). `NQ_ACT_GROUP = NQ_GROUP_MAX = 32`; the pack's 196 linears
  have `full = inn/32` of 32 (kinds 0, 1, 2, 4, 5), 64 (kind 3) and 96 (kind 6), so the plain
  32-block loop is one to three iterations on the measured path, not absent. The unchanged miss
  count is the refutation; the comment's "second exit branch in the hot loop" is not the
  mechanism. The arm's wall is a single run (4.578 s in the log), not a graded pair. The source is
  not in the tree; the box binary hashes, the symbol sizes and opcode counts, and the run lines in
  the log are what remains. Reverted.
- **`unroll`** (`569a66b2…`), the select's 32-block run written straight-line (32 × 8 columns).
  The symbol grows 3,054 → 11,280 B, 610 → 2,021 instructions and 16 → 272 `vpmaddubsw` (256
  straight-line). One perf run each in the same window: misses 242.36 → 243.34 M (+0.98 M, +0.40%,
  unchanged within the arm's spread), instructions 322.00 → 340.62 G (+18.61 G, +5.78%), wall
  4.662 → 4.858 s (+4.2%, a single run, not a graded pair). The select's ~144 M misses are not its
  block-chain back edge. Reverted.

## Against llama.cpp

llama.cpp at `049326a00`, `-p 1827 -n 0 -t 8 -r 3`, before and after the 13 pairs:

| | llama.cpp | ours (`cleanfin`, mean of 13 walls) | ratio |
|---|---:|---:|---:|
| cold prefill 1,827 | 3.984 s (458.6 tok/s) | 4.770 s (383.1 tok/s) | **1.20×** |

The two anchors bracket the window: 454.10 ± 11.65 tok/s before it and 463.15 ± 1.15 after. Against
their mean, 458.6 tok/s, the unrounded ratio is 1.197, quoted as **1.20×** because the anchors alone
span 1.185–1.209× and the 2% anchor spread is wider than the 1.18% effect. The change buys 1.18% of
our wall, not 20% of the gap to llama.

The target is 1×–0.5×. Parity needs the wall to fall a further **0.786 s (16.5% of ours)**; half
of llama is **1.992 s**, which this algorithm does not reach (2.39× away). The remaining gap is not
branch misses: removing every one of the 265.47 M at 18 cycles is 3.46% of the run. It is the
execution rate — the select's 22.82% of cycles and 0.082 instructions/MAC, `ColPipeline::run`'s
per-column `mul_q32` pass and the fixed-point trig — the subjects of `1795664197-nq-prefill-gap.md`.

## Review

`gemini-4-hydrogen-high` ran the first pass on the implementation, the receipt and the diff. It
refused the llama binary filename, a window overwritten by a later capture, eight pair means
reported as thirteen, the percent rounding, unlogged cycle counts, the hot-branch and
caller-branch wording, the missing AVX2 refusal test, the `#[allow]` reason, the unroll counts, a
digest overclaim, "every arm PPL", and the "decode" filename. All are fixed: the anchor names
`Qwen3-0.6B-Base.f16.gguf` (no `.0`) and reads 454–463 tok/s here; the re-captured window is this
one; the pair count is 13; the two reductions are 1.18% and 1.21%; cycles are logged and quoted;
only the `invalid == 0` branch is in the hot path and the wrapper's branches are not claimed;
`swiglu_f32_avx2_refuses_like_scalar` runs the AVX2 refusal; the lint reason is stated; sizes and
opcodes are in the log; `chart/ffn.rs` is not in `SOURCE_FILES` and the omission says so; the
refused arms have no PPL; and the criterion is prefill.

`gemini-4-hydrogen-xhigh` ran the second pass and returned **FAIL** with 13 refusals. The numbers
those refusals name belong to the superseded shared-tree window and are not results; they are not
quoted here.

| # | refusal | disposition |
|---:|---|---|
| 1 | target is not the dominant sub-step; no `ffn` ms breakdown | **open, and the change is refused under the sub-step rule.** The target is the miss owner (25.85% of misses, 3.88% of cycles); the projection loop is larger (select 22.82% of cycles) and the `ffn` sub-steps are not timed in ms. The measured 1.18% is reported as a branch-miss reduction, not a T2 closure; no owner waiver is recorded. |
| 2 | Lean `Source:` spans shift, so "regenerated Lean unchanged" is false | fixed: Omissions says this change adds no extracted body, only `Source:` spans would shift, the check did not run, and the committed Lean already differs at `swiglu_piece`. |
| 3 | the final miss range and gap were wrong for that window | fixed for that window; the window itself is superseded by the clean base. |
| 4 | the control miss mean was rounded a hundredth too high | fixed; superseded. |
| 5 | the miss cut was rounded a hundredth too low | fixed; superseded. |
| 6 | two pair ratios were rounded at the wrong digit | fixed; superseded. |
| 7 | the pair mean was called the ratio of means | fixed: pair mean and ratio of means are separate. |
| 8 | ungraded `model` absolutes and a mixed-window denominator | fixed: `model` grade in the header, formula in the counts, A′ ceiling on A′'s own count, and the clean window has cycles on every pair. |
| 9 | log lacks symbol counts and disassembly | fixed: the log carries the symbol counts, both arms' full `SwiGLU` disassembly (offsets `+521`, `+833`, `+1266`, `+1685` all resolve there), and the sizes and `vpmaddubsw` counts of the refused arms. |
| 10 | refused-arm artifacts missing; `group = 32 = full` is false | fixed: `nm -S` and opcode lines are logged, `full` is 32/64/96, the `brmask` wall is quoted as a single run, and the shared base is named. |
| 11 | pack `k`/`res_layers` unbacked | fixed: the pack-header block in the log reads them from the file. |
| 12 | clippy results unlogged, `--lib` skips the tests | fixed: `cargo clippy --locked --lib --tests` is logged, exit 0, zero `chart/ffn.rs` diagnostics. |
| 13 | helper is x86-gated but the two new tests are not | fixed: both tests carry `#[cfg(target_arch = "x86_64")]`; the helper stays gated because its only caller is. |

Refusal 1 remains open and is not counted as closed by the cycle share.

`gemini-4-hydrogen-xhigh` ran the third pass on the clean-window receipt, the log and the diff and
returned **FAIL** with eight refusals: the pair-mean rounding (1.18%, not 1.19%); the disassembly
offsets and the single-hot-branch claim absent from the log; the excluded shared-tree window's
numbers quoted; the `brmask` artifact overstated; the open sub-step refusal plus the 1.70% cut
labeled as the conversion's; the llama bracket (1.185, not 1.186) and four-figure precision across
a 2% anchor spread; the Lean extraction sentence stale against the committed `swiglu_piece` body;
and two source ranges. All eight are fixed: the pair mean is 1.18%; the log carries both full
`SwiGLU` disassemblies; the superseded window is named without its numbers; the refused-arm record
is narrowed to the logged hashes, sizes, opcodes and run lines; the header and phase section state
the change is refused under the sub-step rule with no owner waiver; the ratio is 1.20× with the
1.185–1.209× bracket; the Lean omission says no extracted body is added, the check did not run, and
the committed `swiglu_piece` body differs from HEAD; and the citations are `pool.rs:463-488` and
`step.rs:472-487`.

`gemini-4-hydrogen-xhigh` re-reviewed those eight fixes and returned **PASS**, with no remaining
refusals and no new false statement. It verified the two disassembly counts and offsets (201 with
14 conditional branches, 358 with 15), the 13-pair arithmetic, the phase table, the llama anchors,
the Lean omission and both source citations.

## Omissions

- The laptop is not measured; the box is shared and absolute walls drift across windows. The
  paired ratio, not an absolute wall, is the graded wall. The two llama anchors differ by 2% within
  this window and the ratio is quoted against their mean with the bracket.
- No 7B, no 200k context, no decode, no quality claim. PPL is the same-stream check only.
- The conversion's own wall share was not timed separately. The shares quoted are its 3.88% cycle
  share and its 25.85% branch-miss share; the 1.70% instruction cut is whole-run, not the
  conversion's. The wall result is the paired 13-pair mean, reported as a branch-miss reduction. The
  sub-step rule is refused: refusal 1 above stays open and no owner waiver is recorded.
- The changed function is `--opaque` in the extraction (`proof/regen.sh:284`) and the new helper is
  called only from that opaque body, so this change adds no extracted body; only the `Source:` line
  spans in `proof/ApeironWord.lean` would shift. That is unverified because `proof/regen.sh
  --check` did not run, and the committed Lean already differs from HEAD at `swiglu_piece` (the
  committed loop body at `proof/ApeironWord.lean:9457-9515` calls `swiglu_one_carrier`, while
  `src/engine/chart/ffn.rs:534-548` calls `swiglu_slice`), so the state of the extraction is not
  established here. No Lean build ran and the function is `tested`, not `proved`. `chart/ffn.rs`
  is not in `SOURCE_FILES` (only `chart/mod.rs` is), so the digest line does not record this file,
  and that gap is the owner's to close.
- `--all-features` and the GPU suite need the `glslc` pin and a device; neither host has them.
  `cargo test --release --locked --lib` on the clean tree gives **691 passed, 1 failed, 55 ignored**;
  the failure is the pre-existing `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` byte
  check, which runs with `gpu` off and needs the pinned compiler.
- `mdbook build` / `mdbook test` did not run: the box has no `mdbook`, and the laptop's resource
  guard refused the cargo invocation. The change touches no `book/` source.
- `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` did not run to green
  on either host. `cargo clippy --locked --lib --tests` on the clean tree is exit 0 with zero
  `chart/ffn.rs` diagnostics. The `needless_bitwise_bool` lints in `q32_from_f32_bits` are allowed
  with the reason that a short-circuit is the branch the change removes.
- `code_select_row8_avx2`'s 143.8 M misses are unattributed. `brmask` and `unroll` refuted the two
  block-chain hypotheses; the next attempt needs a new mechanism, not another restructure.
