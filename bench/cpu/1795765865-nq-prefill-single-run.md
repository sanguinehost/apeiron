# Prefill: one-run whole-row NQ projection

Criterion: `GOAL.md` Q1 prefill rate. Grade: `measured` on the OVH box (8 vCPU EPYC
Milan, `APEIRON_THREADS=8`). The laptop pair is open (`MemAvailable` below the 2 GiB
guard). The timed claim names the box; it is CPU-tier evidence, not laptop evidence.

## Result

`prefill_width --tokens 1827 --reps 3`, 0.6B v3 pack:

| quantity | before | after |
|---|---|---|
| wall, 3 reps | 3586 / 3624 ms best | 3440.449 ms best, 3478.503 ms mean |
| wall, repeat | — | 3545.623 ms |
| instructions (`perf stat`) | 312e9 | 302.9e9 |
| cycles | — | 102.6e9 |
| IPC | 2.93 | 2.95 |
| CPUs utilized | 7.15 | 7.235 |

The instruction count falls 2.9% (`312e9 → 302.9e9`); the wall best falls from 3586
to 3440 ms, but `3545.623` in the repeat is inside the old spread, so the wall claim
is neutral and the receipt claims the instruction count only. llama.cpp prefill is
`3.965 s`; this is `0.867×` at the best wall. The 0.5× target is `1.982 s`, so the
result is `1.74–1.79×` short.

## Mechanism

`code_select_row8_avx2` gains a one-run whole-row body: when the row's equal-scale
run covers the row (`run == full ≥ 2`), eight `i16` pair accumulators carry the whole
row and one `vpsllvd` per column lands the sum in `acc`. The general loop keeps a
`pair` set and an `acc` set live together; at the widths past 1,024 that costs a
store and a load per column per block. The new body is the same arithmetic with the
second set removed, not a second path: every load, shuffle and selector is the
existing one, and the fall-through general loop owns every other run shape.

## Instruction budget

`nq_stats`: 196 linears, 0.977441 b/w, rank sum 124,544, 0.4184 GMAC/token; a
1827-token prefill is `764.4 GMAC`. `302.9e9` instructions is `2.52 MAC/instruction`.

`1.982 s` at the measured aggregate `3.651 GHz × 7.235 CPUs × 2.95 IPC = 77.9e9
instructions/s` needs `≤ 154e9` instructions, i.e. about `5 MAC/instruction` at this
MAC count. The split: attention `192 GMAC` at about `2.3 MAC/instruction` (`82e9`),
projections `572 GMAC` at about `5.3` (`108e9`), and about `76e9` outside the kernels
(25%). The gap is instruction count, not clock: the host has AVX2 but no AVX-VNNI and
no AVX-512, and the pack's `0.977 b/w` fixes the MAC count. A halving needs a
lower-MAC pack or a wider-ISA host; the receipt claims neither.

## Scope

- Prefill only. The lookup decode re-ran with this tree: 5.87–5.93 ms/token, exact,
  102/102 drafts; no regression.
- The box is not the laptop; no `measured` laptop row exists in this receipt.
- Rust/LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.

## Files

- `src/engine/kern/nq.rs`: the one-run body in `code_select_row8_avx2`.
