# T1 speed plan: the shipped dot cores, the Q6_K head, and the gap to 20

Product pair: `nq8b-late-v4/model.nqp` (1,023,456,080 B, 1.178785 b/w on 6,945,767,424 fitted
weights) + `skeleton-q6k-q4kembed.gguf` (867,740,256 B, stock Q6_K `output.weight`, Q4_K embed).
Host: i5-10210U, 4C/8T, AVX2/BMI2/POPCNT, no AVX-512, no AVX-VNNI; 27.97 GB/s measured, 42.656
spec. Grades are per row: `measured` = a named command ran and its output was read; `model` =
static count × an IPC/clock assumption. **The phase split is unmeasured on this pair**: J5 and J6
both refused with `Pack(BadVersion { version: 4 })` (`bench/cpu/1794912865-nq8b-overnight-logs/J5-phase-split.log`,
`J6-head-time.log`) because the release examples predate v4 support (binaries built at epochs
1794813594/1794812830; `src/format/nq.rs` v4 is uncommitted). Every share below is `model`.

## 1. Dot cores

| core | file:line | instr / 32 terms | rate | grade | pointer |
|---|---|---:|---|---|---|
| `sign_dot!` scalar macro | `src/engine/kern/nq.rs:114` | unmeasured (≈8/term in source) | 4.95× slower than the 58-instr AVX2 arm | `measured` ratio, `model` count | `1794806002` |
| `sign_dot_avx2` (`i64` lanes + POPCNT) | `nq.rs:216` | **44** (58 before the popcount fold) | 4.17 G terms/s at the 58-instr build; post-fold absolute **unmeasured**, A/B 1.37–1.41× | `measured` instr (objdump), `measured` old rate | `1794888094`, `1794812057` |
| `sign_dot_planes_avx2` (four byte planes) | `nq.rs:342` | **24** | **11.37 G terms/s**, 2.1 IPC one core (8.5 G instr/s) | `measured` | `1794812057` |
| `code_select_avx2` (q8act product reader) | `nq.rs:761` | **12–14** (9–11 vector + 3 loop; the two receipts count the 2 `vmovdqu` loads differently) | **unmeasured**; the 2.25 tok/s 8B run bounds the whole machine at ≥6.4 G instr/s | `model` | `1794885786` §Execution, `1794879503` |
| `ternary_block` (64-term cell) | `nq.rs:179`; `TernaryTerms::dot` `nq.rs:711` | ≈9/64 = 0.14 | **123 G terms/s** one core, 1.02 popcnt/cycle | `measured` loop; refuted on Q2 (77,741,390 vs 72.988013 PPL) | `1794828200`, `AGENTS.md` |
| `q6_k_half_dots` (head, 128 terms/call) | `src/engine/chart/x86.rs:775` | ≈52/128 = 0.41 | **unmeasured** (§2) | `model` | — |
| `high_dot_avx2` / `low_dot_avx2` (Q8_0 bitplane head) | `src/engine/kern/bitplane.rs:714`, `:823` | ≈8/32 each | unmeasured | — | not the shipped head |

`sign_dot_i64` (`nq.rs:139`) is the dispatch, not a core. `sign_dot_planes_avx2` has its safe twin
`sign_dot_planes_twin` + `the_plane_source_equals_the_scalar_source`; `sign_dot_avx2` has
`sign_dot_i64` + `the_vector_source_equals_the_scalar_source`; `code_select_avx2` has
`code_select_twin` (`nq.rs:456`, test use at `nq.rs:1778`). Any new `unsafe` needs its twin and
equality test in the same change (`AGENTS.md` Unsafe).

## 2. The Q6_K head reader, per 256-weight block

`Q6_KDecoder::decode_step_avx2` (`src/engine/chart/mod.rs:938`) runs two 128-weight halves through
`q6_k_half_dots` (`chart/x86.rs:775`), then `RowAcc::emit` (`chart/mod.rs:593`). Per block, this
host's AVX2 path (no VNNI), source count:

| step | ops | where |
|---|---:|---|
| unpack 6-bit codes (2 nibble planes + 2-bit high plane, −32 bias) | 50 vector (25/half) | `x86.rs:784–795` |
| 32-lane dots, 8 per block (`vpabsb`+`vpsignb`+`vpmaddubsw`+`vpmaddwd`) | 32 vector | `x86.rs:189–194`, `:808` |
| `vphaddd` reduce | 6 vector | `x86.rs:233` |
| weight + activation loads | 6 + 8 | `x86.rs:778–799` |
| group scale fold `sclo·d_lo + schi·d_hi` | 32 `imul` + 16 `add` | `chart/mod.rs:964–966` |
| `emit`: `w_scale·dot`, `act_scale·inner` | 32 wide `imul` + 32 wide `add` + 16 loads | `chart/mod.rs:593–597` |
| `f16` `d` decode + loop control | ≈10 | `chart/mod.rs:949` |
| **total** | **≈228 per 256 weights = 0.89/weight** (`model`; objdump unmeasured) | |

Per token the head is 622,329,856 weights (151,936 × 4,096) in 510,504,960 B → ≈554M instructions,
9.2% of the 5.72G-instruction trunk stream. **Measured rate: 1.53 G weight/s** — but that is
`lm_head_fixed` = `matvec_fixed` on the dequantized rows (`src/engine/step.rs:1266` → `:1385`), the
reference reader, not this one (`head_time` 0.408 s, `J6-head-time.log`). The fused reader's rate is
**unmeasured**: J6's `step Q8Act` line and J5 both died on the v4 pack.

Before the head is targeted, `phase_split --target head` must exit 0 on a rebuilt binary. Static
counts do not make it dominant: 0.55G of 5.72G reader instructions. It is dominant only if its
measured per-weight cost is ≥5× the sign dot's, i.e. if `q8act_plan` takes `Reference` for a Q6_K
head (`chart/mod.rs:1760`; the bound is `q8act_fast_bound_ok(4096, 4096, …)`, `:365`) or the fast
row is slower than the reference — `head_time`'s `step Q8Act` line is the direct test.

## 3. T1 ceilings at the shipped pair

| term | bytes/token | share | at 27.97 GB/s | at 42.656 GB/s |
|---|---:|---:|---:|---:|
| pack | 1,023,456,080 | 55.7% | | |
| Q6_K head | 510,504,960 | 27.8% | | |
| f16 KV @2k (36 layers × 8 kv heads × 128 × 2 × 2 B × 2,048; receipt value 302,776,320) | 302,776,320 | 16.5% | | |
| **total** | **1,836,737,360** | | **15.2 tok/s** | **23.2 tok/s** |

Sign terms/token: **7,623,016,448**. Measured from the shipped v4 pack's 252 section headers
(238,219,264 U+V words; the parse reproduces the 1,023,456,080 B file exactly; the v3 `stats.log`
row agrees). The overnight receipt's 6,945,767,424 row is the *fitted weight count*, not terms
(`1794912865` §The speed ceiling; `1794885786` §Execution) — its 5.21G stream is 8.8% low.
Storage is 215,592,960 words because 72 shared members omit the qkv V block, but the engine runs
the V stage per section (`kern/nq.rs:1082`), so the work is the full 238,219,264 words.

| bound | instr/token | tok/s at 26.4G instr/s (2 IPC × 4 × 3.3 GHz) | gap to 20 |
|---|---:|---:|---:|
| 0.75/term (`sign_dot_planes_avx2`, exact) | 5.717G | **4.62** | 4.33× |
| 0.375–0.4375/term (`code_select_avx2`, the shipped q8act reader) | 2.859–3.335G | **7.92–9.24** | 2.16–2.53× |
| 0.19/term (hypothetical) | 1.448G | **18.2** | 1.10× |
| 20 tok/s budget | ≤1.320G | 20.0 | — (0.173/term, 5.5 instr/32 terms) |
| bytes, measured bandwidth | 1.837 GB | 15.2 | 1.32× |
| bytes, spec peak | 1.837 GB | 23.2 | 1.16× |

The binding bound is the instruction stream of whichever reader runs: 4.62 (exact) or 7.9–9.2
(q8act), both under the 15.2 byte ceiling. 20 tok/s misses the byte ceiling by 1.32× and the q8act
instruction ceiling by 2.2–2.5×; T1's 30 at 2k misses the measured byte ceiling by 1.97× and the
spec peak by 1.29×. **T1 at 24 for 200k is not computable here**: the KV read dominates and the C1
plan is absent (unmeasured). 20 tok/s needs ≤0.173 instructions/term *and* ≤1.40 GB/token, and the
shipped pair is over both.

## 4. Top three levers

Plausible split, `model` from two anchors — J6's 0.408 s reference head is 92% of the 444 ms token
(upper bound) and the static head share is 9% (lower bound): head 0.30–0.55, trunk NQ 0.45–0.60,
attention/norm/untraced 0.05–0.15. The ordering flips with the head's real share, which is why
step 1 below is the split, not a kernel.

| # | lever | file / function | bound on the gain | unsafe twin required? |
|---|---|---|---:|---|
| 1 | trunk dot stream: selector/nibble-LUT form of the 12–14 instr/32-terms core (or of the 24, if the exact reader is the one that runs) | `src/engine/kern/nq.rs:761` `code_select_avx2`; `:342` `sign_dot_planes_avx2` | phase ≤ `1/(1−0.45…0.60)` = **1.8–2.5×**; a 6-instr selector is 1.43G instr/token = 18.4 tok/s reader ceiling, so ≤2.3× inside the phase | **yes**: `code_select_twin` / `sign_dot_planes_twin` + the named equality test |
| 1b | compute the shared qkv V once (already stored once: 22,626,304 of 238,219,264 words = 9.50%) | `src/engine/kern/nq.rs:1082` `project_with` — needs a group, no new kernel | **1.105×** on the trunk phase; 5.10 vs 4.62 tok/s at 0.75/term | no |
| 2 | Q6_K × int8 head kernel: 2-row tile (share the 8 activation loads), hoist the 3 broadcast constants, fold the 16 scale multiplies into one `vpmaddwd` per half | `src/engine/chart/x86.rs:775` `q6_k_half_dots`; `chart/mod.rs:938` `decode_step_avx2` | phase ≤ `1/(1−0.30…0.55)` = **1.43–2.22×**; 0.89 → ~0.45 instr/weight saves 0.28G, i.e. ≤1.05× of the token today | **yes**: twin is `Q6_KDecoder::decode_step_portable` (`chart/mod.rs:980`); the test at `chart/tests.rs:694` covers only groups 0–1 and neither half's upper groups — extend to all 8 × 2 |
| 3 | bytes: head inside the pack (rank 4096 ≈ 80,208,640 B), then KV quantisation | `src/format/nq.rs` (v5 head section), `loader/nq.rs`, `step.rs:1339` | byte ceiling 15.2 → **19.9** with f16 KV (1,406,441,076 B/token), **22.3** with a half-size KV; **1.00× today** because bytes are not binding | no |

Not a lever: AVX-512 (absent), AVX-VNNI (absent; `dpbusd_vex` is dead on this host). The bitplane
head (`kern/bitplane.rs`) is a Q8_0 argmax route, not the shipped Q6_K head. `head_cols_kernel`
(`step/cols.rs:598`) returns `None` for Q6_K, so `ppl --head-batch 16` reads the head 16× per
block — a 16× head saving for prefill and the PPL harness, 1× for decode; cheap, and it does not
move decode.

## 5. 24-hour order

Every command below is the gate for its step. Builds through `agent-guard`, model runs in the
3.2 GiB scope, `APEIRON_THREADS=4`.

| T+ | step | command | gate |
|---:|---|---|---|
| 0:00 | rebuild the two examples (fixes J5/J6's `BadVersion`) | `agent-guard run -- cargo build --release --locked --example phase_split --example head_time` | exit 0; then `phase_split … --steps 1 --json` exits 0 |
| 0:30 | the split | `systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- target/release/examples/phase_split --model /home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf --pack /home/lucas/models/arelath/nq8b-late-v4/model.nqp --steps 24 --json` | log's `dominant_share`; if `untraced` wins, `--target untraced` and instrument first |
| 1:00 | the gate on the hypothesis | same, `--target head`; then `--target ffn` | exactly one exits 0. `head` 0 → lever 2 first; `ffn` 0 → lever 1 first |
| 1:30 | head vs trunk on the shipped reader | `systemd-run … target/release/examples/head_time <model> <pack>` | `head best … G weight/s` and `step Q8Act … tok/s` both present; head ms / step ms within a few points of the split's head share |
| 2:00 | the chosen phase's equality test, extended first | `agent-guard run -- cargo test -j 1 --release --lib --all-features --locked chart::` (head) or `kern::nq` (dot) | green before the kernel changes; a mutation of the constant/table turns it red |
| 3:00–14:00 | the kernel change, one branch only | edit the named function; `touch src/lib.rs`; rebuild `--release --example phase_split --example generate` | `cargo test … chart::` / `kern::nq` green; `cargo clippy -j 1 … -D warnings`; `cargo fmt --check` |
| 14:00 | re-split and re-gate | the T+1:00 pair again | `--target <changed phase>` exits 0 **after**; the post-change share is printed |
| 15:00 | timed protocol, mean of ≥3 | `for i in 1 2 3; do systemd-run … target/release/examples/generate --model <model> --pack <pack> --tokenizer /home/lucas/models/arelath/nq8b/tokenizer.json --prompt "$(cat /tmp/opencode/nq8b-2k-prompt.txt)" --n 32 --kernel q8act; done` | mean and spread; a change inside the spread is not a change |
| 17:00 | the receipt | `bench/cpu/<epoch>-nq8b-<lever>.md` | criterion id, grade, command, machine, before/after split, both bounds, what it does not cover |
| 20:00 | remaining hours | lever 3 preflight only: size the head section and the Q2 fit cost; do not start a format change | a written go/no-go; the v5 format change is a separate 1–2 day item |

Expected outcome, stated against the target: if the head holds 0.40 and a halved head kernel lands,
444 ms → 302 ms (**3.3 tok/s**); if the trunk dot also halves, → 191 ms (**5.2 tok/s**). Against 20
that is 6.1× and 3.8× short. The configuration's own ceiling stays **15.2 tok/s** at the measured
bandwidth until the head leaves the dense payload, and **19.9** after that — 20 needs KV
quantisation on top. No step in this plan promises 20.

## 6. What this does not cover

The split, the head's shipped rate, the `code_select_avx2` rate and the `q6_k_half_dots` rate are
all unmeasured until T+1:30; every share and every `1/(1−s)` above is `model`. The 200k row, the
200k KV plan (C1), 0-shot quality (Q2), rustc/LLVM codegen, the kernel ABI, Vulkan and the GPU are
outside. A pack with the head inside it does not exist; its 80,208,640 B is `rank_for_bits`
arithmetic at 1.03 b/w, and the head fit's quality is unknown (`1794810332` §4).
