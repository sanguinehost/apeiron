# NQ loop ports: the q8 reader is 12 instructions, the ternary cell 11 per 64

Criterion: **T1** (decode), PLAN workstream W3. Grades: `measured` for the binary,
symbols, addresses, instruction counts and folded-load forms; `anchor` for the port map
(uops.info Skylake/Ice Lake tables, not host counters); `model` for every throughput
projection and the ternary shape, which is not in the binary.

Scope: `code_select_avx2` per 32 terms, `sign_dot_planes_avx2` as the control, and the
proposed 64-term ternary popcount cell. No end-to-end decode, no pool, no head, no KV.
Host: Intel i5-10210U (Comet Lake-U, Skylake core), 4C/8T, 4.2 GHz max, 3.9 GHz at audit
(93% scaling reported by `lscpu`).

## 1. Binary

| field | value |
|---|---|
| path | `target/release/deps/apeiron-8aa17308f824d420` |
| build id | `4020e9a7848152213c6a5b1fbefef0c8022656ad` |
| sha256 | `3a96ae579d69c2c37eb80271165cd206b2cc006e121da5b147407a848a4e0713` |
| format | ELF x86-64 PIE, not stripped, rustc 1.99.0-nightly `0e29c21d9`, LLD 22.1.8 |
| built | 19:55:49; `src/engine/kern/nq.rs` mtime 20:42:37 |

This is the newest release executable carrying all three symbols. `serve/target/release/nq-serve`
(13:47) does not carry them. `ppl` (19:50) and `gpu_decode_step` (19:54) carry the same function
sizes. The binary predates the working tree's last `nq.rs` edit; the loop below matches the source
shape in `nq.rs` (the three zips, the folded plane and shift loads), but the tree is dirty and the
build is not tied to a commit, so a byte-for-byte source match is not established.

| symbol | start | end | size | hot loop | loop size |
|---|---:|---:|---:|---|---:|
| `nq::sign_dot_avx2` | `0x3fc7c0` | `0x3fca42` | `0x282` | not audited | — |
| `nq::code_select_avx2` | `0x3fca50` | `0x3fce42` | `0x3f2` | `0x3fcaf0–0x3fcb24` | 54 B, 12 insn |
| `nq::sign_dot_planes_avx2` | `0x3fce50` | `0x3fd246` | `0x3f6` | `0x3fced0–0x3fcf3d` | 111 B, 24 insn |

Addresses are PIE virtual addresses at file scope; ASLR does not change the differences.

## 2. `code_select_avx2`: 12 instructions per 32 terms

Disassembly, `objdump -d --demangle --no-show-raw-insn --start-address=0x3fcaf0
--stop-address=0x3fcb26`:

| # | address | instruction | uops | port | role |
|---:|---|---|---:|---|---|
| 1 | `0x3fcaf0` | `vpbroadcastd (%r9,%r11,1),%ymm6` | 1 | p23 | load the block's `u32` word |
| 2 | `0x3fcaf6` | `vpsrlvd %ymm1,%ymm6,%ymm6` | 1 | p01 | nibble split by `[0,4,8,…,28]` |
| 3 | `0x3fcafb` | `vpshufb %ymm2,%ymm6,%ymm6` | 1 | p5 | repeat each nibble four times |
| 4 | `0x3fcb00` | `vpand %ymm3,%ymm6,%ymm6` | 1 | p015 | mask `0x08040201` |
| 5 | `0x3fcb04` | `vpshufb %ymm6,%ymm4,%ymm6` | 1 | p5 | map `2^j` to `1` |
| 6 | `0x3fcb09` | `vpmaddubsw (%r10,%r11,8),%ymm6,%ymm6` | 2 | p01 + p23 | **activation plane load folded** |
| 7 | `0x3fcb0f` | `vpmaddwd %ymm5,%ymm6,%ymm6` | 1 | p01 | fold byte pairs to `i32` |
| 8 | `0x3fcb13` | `vpsllvd (%r8,%r11,8),%ymm6,%ymm6` | 2 | p01 + p23 | **shift-vector load folded** |
| 9 | `0x3fcb19` | `vpaddd %ymm0,%ymm6,%ymm0` | 1 | p015 | block accumulator |
| 10 | `0x3fcb1d` | `add $0x4,%r11` | 1 | p0156 | 4 B word / 32 B plane / 32 B shift |
| 11 | `0x3fcb21` | `cmp %r11,%rdx` | 1 | p6, fused | with #12 |
| 12 | `0x3fcb24` | `jne 0x3fcaf0` | — | p6, fused | loop |

Totals: **12 static instructions**, **11 front-end uops** (two branches fuse; the three loads are
micro-fused), **13 execution uops** (8 vector-ALU + 3 load + 1 scalar `add` + 1 fused `cmp`/`jne`;
14 if the branch pair does not fuse). There is no standalone load in the loop: all three loads ride
the broadcast and the two memory-form vector instructions.

Port pressure per 32 terms, best schedule:

| resource | uops | cycles |
|---|---:|---:|
| p0/p1 | 4 fixed (`vpsrlvd`, `vpmaddubsw`, `vpmaddwd`, `vpsllvd`) + 2 of `vpand`/`vpaddd`/`add` | 3.0 |
| p5 | 2 `vpshufb` + 1 of `vpand`/`vpaddd`/`add` | 3.0 |
| p2/p3 | 3 loads | 1.5 |
| p6 | 1 fused branch | 1.0 |
| front end | 11 uops / 4-wide | 2.75 |
| scheduler | 13 execution uops / 4-wide | 3.25 |

p0/p1 and p5 tie at 3.0 cycles/block, the highest port pressure; issue sits at 2.75 cycles in the
fused domain and 3.25 in the execution domain, so the block costs 3.0–3.25 cycles in the model and
no single port is clear of the issue limit. Multiplies and variable shifts are p01 on both Skylake
and Ice Lake (`vpmaddubsw`, `vpmaddwd`, `vpsrlvd`, `vpsllvd`, throughput 0.5); `vpshufb` is p5-only;
`vpbroadcastd ymm,m32` is one p23 load uop, not a p5 broadcast.

### PLAN check

- **"eleven vector ops"** — refuted as written. There are **9 vector instructions** (8 ALU plus the
  broadcast, which is a load-form instruction), **8 vector-ALU uops** and **3 vector load uops**.
  Eleven is right only as 8 ALU + 3 loads, and the text's "two loads" is then wrong.
- **"~14 instructions"** — right as an upper bound on execution uops (13 with the branch fused, 14
  without), wrong as a static count (12) and as a front-end count (11). The rate row is consistent
  with the execution-uop count, not the instruction count: at 2.1 IPC, 13 uops give 10.2 tok/s and
  14 give 9.4 at 7.94G terms/token; the row's 9.6 sits between them. A 12-instruction count would
  give 11.0.
- **"two `vpshufb` are the selector expansion"** — confirmed, with the whole selector being 5
  instructions: broadcast (load), `vpsrlvd`, `vpshufb`, `vpand`, `vpshufb`. The `.rodata` constants
  are `bit_at = 0,4,…,28` at `0x61de0`, `repeat = 0,0,0,0,4,4,4,4,…` at `0x61640`,
  `digit_mask = 01 02 04 08` at `0x50c10`, `select_one` with ones at byte indices 1,2,4,8 at
  `0x61a80`.
- **"load folded into `vpmaddubsw`"** — confirmed: `(%r10,%r11,8)` is the activation plane. The
  shift vector is likewise folded into `vpsllvd`; the block word is the broadcast load.

## 3. `sign_dot_planes_avx2`: 24 instructions per 32 terms (control)

| group | instructions | count | ports |
|---|---|---:|---|
| head control | `lea`, `cmp`, `ja` | 3 | p15, p0156, p6 (cmp/ja fused) |
| selector | broadcast, `vpsrlvd`, `vpshufb`, `vpand`, `vpshufb` | 5 | p23, p01, p5, p015, p5 |
| four planes | 4 × (`vpmaddubsw` mem, `vpmaddwd`, `vpaddd`) | 12 | 4 × (p01+p23, p01, p015) |
| tail control | `add`, `mov`, `cmp`, `jne` | 4 | p0156, eliminated, fused p6 |
| **total** | | **24** | |

The count is exactly 24, matching `bench/cpu/1794812057-nq-sign-dot-simd.md`. Port pressure: p0/p1
carries 9 fixed execution uops (1 `vpsrlvd` + 4 `vpmaddubsw` + 4 `vpmaddwd`) and takes 2 of the 5
`vpand`/`vpaddd` uops = 5.5 cycles; p5 carries 5.0. The static 24 are 22 front-end uops (two fused
branches) and 25 execution uops (four folded plane loads, the move eliminated); 4-wide issue is 5.5
cycles in the fused domain and 6.25 in the execution domain. The receipt's 5.60 tok/s at 2 IPC is
5.24 at the same 2 IPC, 3.9 GHz and 7.94G terms/token (5.50 at 2.1 IPC); 8.0 becomes 7.86. The
24-instruction loop is not the target; it is the arithmetic and control-count check.

## 4. Proposed 64-term ternary popcount cell

Not present in any binary. The exact shape, per 64 terms per output row, masks resident:

```asm
        mov    (%rdi), %rax      # w: one u64 weight word, p23
        mov    %rax, %rcx        # copy (rename-eliminated, 0 execution ports)
        and    %r8, %rax         # w & p
        and    %r9, %rcx         # w & n
        popcnt %rax, %rax        # pc(w & p), p1
        popcnt %rcx, %rcx        # pc(w & n), p1
        add    %rax, %r10        # acc_p
        add    %rcx, %r11        # acc_n
        add    $8, %rdi
        cmp    %rdi, %r12
        jne    .Lloop
```

`p` in `r8`, `n` in `r9`, `acc_p` in `r10`, `acc_n` in `r11`. The copy cannot be dropped: both
`and`s consume the same `w`. A second load of `(%rdi)` instead of the copy keeps 11 static and 10
front-end uops but costs 10 execution uops rather than 9. One accumulator is the same static count
(`sub` + `add` instead of `add` + `add`) but serialises the two popcounts through one chain; two
accumulators are independent. The per-row epilogue is `sub acc_n,acc_p; lea (acc_p,acc_p),acc_p; add C,acc_p`,
three instructions once per row, with `C = Σ_j (pc(n_j) − pc(p_j))` computed once per activation
block set and shared by every row. No per-block constant enters the row loop.

Counts and projections. 4 cores × 3.9 GHz = 15.6G cycles/s; at IPC 2.1, 32.76G instructions/s;
at IPC 3.0, 46.8G; 7.94G terms/token as specified (the PLAN ledger's trunk figure is 7.613G, 4.3%
lower).

| variant | static / 64 | allocated uops / 64 | exec uops / 64 | tok/s at 2.1 | tok/s at 3.0 |
|---|---:|---:|---:|---:|---:|
| arithmetic core only (PLAN's ~6/64) | 6 | — | 6 | 44.0 | 62.9 |
| exact, masks in registers, 1× | **11** | 10 | 9 | **24.0** | **34.3** |
| 4× unrolled, masks in registers | 8.75 | 8.5 | 7.5 | **30.2** | **43.1** |
| exact, masks reloaded per row, 1× | 13 | 12 | 11 | 20.3 | 29.0 |
| 4× unrolled, masks reloaded | 10.75 | 10.5 | 9.5 | 24.6 | 35.1 |
| port-1 popcount ceiling (IPC-free) | — | — | 2 popcnt | 62.9 | 62.9 |

Arithmetic, exact case: 11 instructions/64 = 0.171875 instructions/term; 32.76 / 0.171875 =
190.6G terms/s; 190.6 / 7.94 = **24.0 tok/s**, 1.20× the 20 target. At 3.0 IPC: 46.8 / 0.171875 =
272.3G; 272.3 / 7.94 = **34.3 tok/s**, 1.71×. The 6/64 row reproduces PLAN's 44 and 63 exactly,
which shows that row counts only 2 `and` + 2 `popcnt` + 2 `add`; it omits the load, the copy and
the three loop-control instructions. Ten instructions (copy omitted) would project 26.4, but the
copy is required. IPC is applied to the static count, as PLAN applies its 14 to the q8 row; on the
same IPC and 10 allocated uops the exact cell is 26.4/37.7, and on 9 execution uops 29.3/41.9. The
11-instruction count is the conservative one.

4× unrolling: 4 × (load, copy, 2 `and`, 2 `popcnt`, 2 `add`) + `add` + `cmp` + `jne` = 35/256 =
8.75 per 64, 30.2 tok/s at 2.1 and 43.1 at 3.0. The four groups use offsets 0, 8, 16, 24 from one
base and one pointer add; four weight registers plus the two masks and two accumulators fit the
16 GPRs.

Masks in registers: `p` and `n` are per activation block and are shared by every output row, so
the row loop must be block-outer/row-inner (or the weight pack tiled) to amortise them. In a
row-outer loop each 64-term block reloads both masks: 13 static/64 and 20.3 tok/s at 2.1 — a 1.5%
margin over 20, not room. The 4× unroll recovers to 24.6.

### Binding port and margin

Port 1 executes both `popcnt`: 2 per 64, one per cycle, 2.0 cycles per 64 terms per core. Four
cores give 7.8G groups/s = 499.2G terms/s = **62.9 tok/s**, a **3.14× margin** over 20. The other
ports are below it: p0/p1/p5 carry 5 `and`/`add`/pointer uops = 1.67 cycles, the load ports 0.5,
p6 1. But the loop cannot issue at port 1's rate: 10 allocated uops / 4-wide = 2.5 cycles per 64
at unit unroll, so the front end binds first and the realised margin at 2.1 IPC is **1.20× (24.0
tok/s)**. The 4× unroll brings issue to 2.125 cycles (8.5 allocated uops) and port 1 to 2.0: on
execution uops (7.5/64 = 1.875) port 1 is the binding port, on allocated uops the front end still
edges it by 6%. The margin the plan should quote is therefore 1.20× at 2.1 IPC, 1.71× at 3.0, and
1.51–2.16× with the 4× unroll — not 3.1×.

Bytes, for orientation only: 8 B per 64 terms = 0.125 B/term; 7.94G terms/token = 0.9925 GB/token.
20 tok/s reads 19.85 GB/s = 71.0% of the measured 27.97 GB/s; the weight-only byte ceiling is
**28.2 tok/s**. Rows above 28.2 (e.g. 30.2 at 4×/2.1 IPC, 34.3 at 3.0 IPC, 43.1 at 4×/3.0) are
instruction ceilings, not reachable rates on this host. Masks (16 B per 64-term block per row
unless block-outer reuse), the head, KV and the pack are not counted.

## 5. What this does not cover

- **rustc/LLVM code generation of the ternary cell**: the cell is hand-written here and is not in
  the binary. LLVM may spill, fail to keep both masks in registers, or fail to eliminate the copy;
  no compiled ternary loop was disassembled.
- **The pool**: no threading, no width, no per-core scaling measurement. Four-core numbers are
  linear extrapolations, as in the earlier receipt.
- **Bytes**: only the weight-bit arithmetic is stated. Mask traffic, head, KV, the 200k context and
  the measured bandwidth under four concurrent readers are outside this audit.
- **Timing**: no performance counter was read. Ports come from the Skylake/Ice Lake uops.info
  tables; the host is Comet Lake (Skylake core), and Ice Lake agrees for every instruction used.
  The 2.1 and 3.0 IPC values are assumptions supplied by the task, not measurements of this loop.
- **The q8 row's IPC origin**: 2.1 IPC is the earlier receipt's figure for the q8 loop; it is not
  re-measured here. The front-end and port conclusions hold only inside the Skylake model.
- **Binary/source identity**: the executable predates the working tree's last `nq.rs` edit and the
  tree is dirty; no commit hash binds the two.
- **Tails and shapes**: 3,584 and 18,944 are multiples of 64, so the cell needs no tail there; a
  non-multiple length is not costed. No per-layer term distribution is used beyond the 7.94G total.
- **Correctness**: the popcount identity and its safe twin belong to W2/W4; this file audits cost,
  not the formula.

## 6. Reproduce

```sh
nm -S -C target/release/deps/apeiron-8aa17308f824d420 | grep -E 'nq::(code_select|sign_dot)'
objdump -d --demangle --no-show-raw-insn --start-address=0x3fcaf0 --stop-address=0x3fcb26 \
  target/release/deps/apeiron-8aa17308f824d420
objdump -d --demangle --no-show-raw-insn --start-address=0x3fced0 --stop-address=0x3fcf3f \
  target/release/deps/apeiron-8aa17308f824d420
objdump -s -j .rodata --start-address=0x61640 --stop-address=0x61660 \
  target/release/deps/apeiron-8aa17308f824d420
readelf -n target/release/deps/apeiron-8aa17308f824d420 | grep 'Build ID'
sha256sum target/release/deps/apeiron-8aa17308f824d420
```
