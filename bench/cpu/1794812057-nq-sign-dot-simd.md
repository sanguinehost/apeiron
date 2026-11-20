# NQ sign dot: byte planes, 2.73× paired, 5.60 tok/s instruction ceiling

Criterion: **T1** (decode), the 1.174 b/w trunk. Grades: `measured` for the paired arms, the
per-token totals and `project`; `tested` for the equality and the mutation; `model` for the
instruction ceilings, the four-digit guard arithmetic and every four-core extrapolation. Scope: the
seven per-layer linears, 7,613,448,192 sign terms and 6,525,288,448 replaced weights per token
(`bench/cpu/1794781140-nq7b-118-full-pack.md`). The LM head, the KV read, attention, prefill and an
end-to-end decode run are outside this receipt. Host: i5-10210U, 4C/8T, 6.6 GiB, `performance`,
27.97 GB/s measured, 42.656 GB/s spec, rustc 1.99.0-nightly. The session's shell and cgroup see 2
CPUs; the pool is configured at width 8; the measurement below runs at pool width 1.

The sign dot is 2.65–2.79× as fast as the `i64`-lane arm on the same planes, and moves its 2 IPC
instruction ceiling from 2.32 to **5.60 tok/s: 3.6× short of 20, 4.3× short of T1's 24 at 200k, 5.4×
short of 30 at 2k.** At the 1,404.7 MB/token bytes bound 20 tok/s is 100.4% of the measured 27.97
GB/s, zero margin; the kernel is the lower ceiling by 3.6×, so 20 tok/s is compute-bound after this
change, not bytes-bound.

## 1. What changed

`src/engine/kern/nq.rs` only; the `i64`-lane source and the scalar macro are unchanged. The `i64`
arm is the baseline every ratio here pairs against; it measured 4.95× the scalar macro on a
4096-wide microbenchmark (`1794806002-nq-sign-dot.md`), so the ratios below are arm-to-arm, not
against the macro.

- `BytePlanes`: each stage term split into four signed byte digits, little-endian,
  `Σ_b 2^{8b}·plane[b][i] = t[i]`. The split is exact over `−2,155,905,152 … 2,139,062,143`; a
  fifth digit returns `None` (`the_four_digit_split_refuses_a_fifth_digit`).
- `sign_dot_planes_avx2`: 32 terms per iteration. One `u32` word → 32 `i8` selectors in five
  instructions (`vpbroadcastd`, `vpsrlvd`, `vpshufb` repeating each nibble, `vpand` with
  `[1,2,4,8]`, `vpshufb` mapping `2^j → 1`). Each of the four planes costs one `vpmaddubsw`
  (unsigned selector × signed digit, load folded), one `vpmaddwd` with ones, one `vpaddd`. Result:
  `Σ t − 2·Σ_{bit=1} t`, the macro's `Σ (1 − 2·bit)·t`.
- `sign_dot_i64(t, planes, sum, words, len)` is the one dispatch point: byte arm when the stage's
  planes are present, `i64`-lane arm when they are not, scalar macro without AVX2. `Terms::new`
  builds the sum and the planes once per stage (length `inn` or `rank`), under the same
  `n·max|t| ≤ i64::MAX` bound the narrow arm already required; a term outside the four-digit range
  leaves `planes = None` and the whole stage takes the `i64` arm, exact.
- Exactness of the `i32` lanes: `512·⌈len/32⌉ ≤ i32::MAX` (`lane_sums_fit`), i.e. `len` up to
  134,217,695; the largest production length is 18,944. At most 31 trailing terms run the scalar
  macro with the tail word's padding bits masked exactly as before.

Equality: new safe twin `sign_dot_planes_twin` and
`the_plane_source_equals_the_scalar_source` over lengths `{0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65,
96, 127, 128, 129, 1000, 1023}` × three magnitudes up to `2,139,062,143`, with the tail padding
bits both clear and set; twin, intrinsic and macro agree. The existing
`the_vector_source_equals_the_scalar_source`, `the_kernel_equals_the_sequential_staged_product`
and `the_wide_accumulator_agrees_when_the_bound_leaves_i64` stay green.

Mutation: `digit_mask` `0x0804_0201 → 0x0804_0200` in a copy, `len 32`: the byte source returns
1255 against the scalar 873, test red; restoring the constant turns it green. A check that cannot
fail carries no weight.

## 2. Disassembly

`nm -S -C target/release/deps/apeiron-947ff21c07d159c8 | grep sign_dot`, then
`objdump -d --demangle --no-show-raw-insn --start-address=<sym> --stop-address=<sym+size>`. The
binary is built from this receipt's source after the last edit.

| symbol | static size | instructions | loop body | per 32 terms | per term | per weight |
|---|---:|---:|---:|---:|---:|---:|
| `sign_dot_avx2` | 642 B | 143 | `0x5f3700–0x5f3834` | 58 | 1.8125 | 2.115 |
| `sign_dot_planes_avx2` | 1,014 B | 250 | `0x5f39b0–0x5f3a1d` | 24 | 0.75 | 0.875 |

Per replaced weight uses the measured 1.1668 terms/weight (7.613G / 6.525G). The static size grows
because the byte arm adds a four-plane epilogue and an inlined scalar tail; the loop is the count
that matters.

`sign_dot_avx2` loop, exactly: 16 `vpaddq`, 8 `vpsllvq`, 8 `vpsrlq`, 8 `vpcmpgtq`, 8 `vpxor`
(with the term loads folded), 1 `vmovq`, 1 `vpbroadcastq`, 2 `cmp`, 2 `mov`, 1 `lea`, 1 `add`,
1 `ja`, 1 `jne` = 58 per 32 terms.

`sign_dot_planes_avx2` loop, exactly: `lea`, `cmp`, `ja`; `vpbroadcastd`, `vpsrlvd`,
`vpshufb`, `vpand`, `vpshufb`; 4 × (`vpmaddubsw` with its memory load, `vpmaddwd`, `vpaddd`);
`add`, `mov`, `cmp`, `jne` = 24 per 32 terms. The loop reads 32 B per digit plane and keeps all four
`i32` accumulators in registers; the planes are stage-level and shared by every row.

## 3. The measurement

```sh
APEIRON_THREADS=1 APEIRON_ECORES=0 agent-guard run -- cargo test -j 1 --release \
  --lib --all-features --locked nq_production_shape_throughput -- --ignored --nocapture
```

Three runs, each with three timed passes per arm, the arms alternating order inside every run and
the per-shape totals asserted equal. Pool width 1; the pass times the two kernel bodies directly,
`project` times the whole `NqLinear::project`. Load average at each run: 1.40, 1.32, 1.71.

| run | `i64` lanes ms/token | G terms/s | byte planes ms/token | G terms/s | ratio | `project` ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,903.84 | 4.00 | 683.38 | 11.14 | 2.79 | 751.83 |
| 2 | 1,762.94 | 4.32 | 665.33 | 11.44 | 2.65 | 703.11 |
| 3 | 1,823.01 | 4.18 | 660.80 | 11.52 | 2.76 | 710.72 |
| **mean** | **1,829.93** | **4.17** | **669.84** | **11.37** | **2.73** | **721.89** |

Spread `(max−min)/mean`: `i64` **7.7%**, byte **3.3%**, ratio **5.1%**, `project` **6.7%**. One
core: 0.55 tok/s (`i64`), **1.49** (byte), 1.39 (`project`); four-core linear extrapolation 2.19,
**5.97**, 5.54. The byte arm reads 9.74 G replaced weights/s on one core.

An earlier triple measured before a lint-only edit (same kernel, quieter box, load 1.0–1.2) gave
byte 12.28–12.83 G terms/s and ratios 2.63–2.73; both arms moved together by ~8%, so the ratio is
the stable quantity and the absolute rate tracks background load. `project` is 7.8% above the bare
two-stage arm: the remaining stage arithmetic (plane building is `O(inn + rank)` per layer, the
scale folds and the pool dispatch) is small, and the sign dot is ~93% of `project`.

Per-shape magnitudes (identical in all runs; deterministic seeds). `max|t|` is the same on all four
shapes because `t = mul_q32(x, s_pre)` depends only on `inn = 3584` and the test's `s_pre`:

| shape | out × inn | rank | max abs t | max abs z | four-digit z headroom | planes admitted |
|---|---|---:|---:|---:|---:|---|
| q,o | 3584 × 3584 | 2080 | 13,414,400 (2^23.7) | 710,706,688 (2^29.4) | 3.01× | yes |
| k,v | 512 × 3584 | 512 | 13,414,400 | 559,438,644 (2^29.1) | 3.82× | yes |
| gate,up | 18944 × 3584 | 3520 | 13,414,400 | 676,746,028 (2^29.3) | 3.16× | yes |
| down | 3584 × 18944 | 3520 | 13,414,400 | 1,746,295,852 (2^30.7) | 1.22× | yes |

The three-plane guard: signed 24-bit terms cover `−8,388,608 … 8,388,607`. The measured
`max|t| = 13,414,400` is **1.60× that bound**, so a 3-plane form is not admissible on this
activation, and it would only remove 3 of 24 instructions anyway (§4). The largest `z` leaves 1.22×
four-digit headroom on the `down` shape; a real activation above `2,139,062,143` at the U stage
routes that stage to the `i64` arm, still exact (2.65–2.79× that stage's byte-arm time).

## 4. Ceilings

Instructions per token at the true geometry; 1 IPC = 16G instructions/s over 4 cores × 4 GHz, 2 IPC =
32G. No KV read, no head, no attention or sampling.

| arm | instr/term | instr/weight | instr/token | tok/s at 1 IPC | tok/s at 2 IPC |
|---|---:|---:|---:|---:|---:|
| `i64` lanes | 1.8125 | 2.115 | 13.799G | 1.16 | 2.32 |
| byte planes | 0.75 | 0.875 | 5.710G | 2.80 | **5.60** |

Against the bars: the 2 IPC ceiling is 3.57× short of 20, 4.28× short of T1's 24 at 200k, 5.35×
short of 30 at 2k. The measured one-core byte rate gives 5.97 at four cores, 3.35× / 4.02× / 5.03×
short. The budget at 2 IPC is 1.6G instructions/token = **0.210 instructions/term = 0.245 per
replaced weight**; the arm is at 0.75/0.875, 3.6× over.

Bytes, the other bound:

| per-token read | B | measured 27.97 GB/s | spec 42.656 GB/s |
|---|---:|---:|---:|
| trunk + Q6_K head (`1794799742-decode-ceiling.md`) | 1,404,683,392 | 19.91 tok/s | 30.37 tok/s |
| trunk + NQ head (`1794810332-nq-head-in-pack.md`) | 1,037,626,532 | 26.96 tok/s | 41.11 tok/s |

20 tok/s needs 28.09 GB/s with the Q6_K head = **100.4% of the measured streaming mean**, zero
margin, and 65.9% of spec; with the NQ head it needs 20.75 GB/s = 74.2% measured, 48.6% spec.
At 2k with the NQ head the byte ceiling is 26.96 (2.96 above the 24 bar, 3.04 below 30); at 200k
with the C1 plan the byte ceiling is 12.3 measured (`1794799742-decode-ceiling.md`, §4). The byte
arm's 5.60 at 2 IPC (5.97 extrapolated from one core) is below both, so after this change the sign
dot, not the per-token bytes, is the binding constraint at both T1 rows: at 2k the fitted-head byte
ceiling is 4.8× the 2 IPC instruction ceiling, and at 200k the measured byte ceiling is 12.3
against 5.6–6.0.

The head is excluded from every count above. If it is fitted at rank 4096 it adds 637,534,208
terms/token (+8.4%), 6.188G instructions/token and 5.17 tok/s at 2 IPC; the rate per weight stays
0.875 because the head sits at the same 1.174 b/w.

A 3-plane form at 21 instructions per 32 terms would be 6.41 tok/s at 2 IPC, still 3.12× short of
20; it does not change the verdict, and §3 shows the guard fails on the measured `|t|`.

## 5. What remains

- The arm does not reach 20 tok/s, and neither lane width nor digit count closes the gap. The
  24-instruction loop spends 5 on selectors and 12 on four planes; a form that consumes 32 terms in
  fewer instructions, or a target with fewer than 0.210 instructions/term, is a different design.
- The activation and `s_pre` are the test's synthetic pattern (`|x| ≤ 50·2^24`, `s_pre`
  ≈ 0.01–0.016). The real `t` distribution, and therefore the four-digit headroom (1.22× on the U
  stage of `down`), is unmeasured. A real `|z| > 2,139,062,143` falls back per stage, exact.
- No end-to-end decode, no prefill (`k > 1`), no KV read, no attention. `project` at pool width 1
  is 1.39 tok/s on these shapes; the four-core numbers are extrapolations.
- The pool at its configured width 8 over-subscribes a shell that sees 2 CPUs: the same test at
  width 8 measured `project` 175–546 ms/token (105% spread), so no pool-width-8 number is quoted
  as evidence.
- The safe twin lives in the `nq.rs` test module, not in `proof/extract/lib.rs`, and
  `sign_dot_planes_avx2` is outside the extraction (intrinsics). Per `AGENTS.md` that twin's
  placement in the extraction is part of the `unsafe`'s debt and is not done here: this change's
  scope is `nq.rs` plus this receipt.
- rustc/LLVM code generation, the kernel ABI, Vulkan and the GPU are outside every claim.

## 6. Reproduce

```sh
powerprofilesctl get
agent-guard run -- cargo test -j 1 --lib --all-features --locked kern::nq
agent-guard run -- cargo clippy -j 1 --lib --all-features --locked -- -D warnings
cargo fmt --all -- --check
agent-guard run -- cargo test -j 1 --release --lib --all-features --locked kern::nq --no-run
APEIRON_THREADS=1 APEIRON_ECORES=0 agent-guard run -- cargo test -j 1 --release \
  --lib --all-features --locked nq_production_shape_throughput -- --ignored --nocapture   # ×3
nm -S -C target/release/deps/apeiron-947ff21c07d159c8 | grep sign_dot
```

Final gates on this tree: `cargo test ... kern::nq` 8 passed, 2 ignored, exit 0; clippy exit 0;
`cargo fmt --all -- --check` exit 0.
