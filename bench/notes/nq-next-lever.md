# NQ projection: decision on the row block and the next lever

Receipt: `bench/cpu/1795513383-nq-blocking.md`. This note is the handoff, not evidence.

## Decision

`NQ_BLOCK = 4` stays the default (`src/engine/kern/nq.rs`). Eleven laptop pairs, 8 favour block 4,
projection median **1.081**; the box screen is 1.063. The projection reduction is `measured`. The
wall reduction holds only in the first series and is not claimed. `APEIRON_NQ_BLOCK=0` is the
same-binary control and stays.

Do not add a block size, a second row loop or a `gemm_*` sibling. The row block is a dial on
`code_select_block_avx2`; `R` is monomorphised 2..8 by `code_select_rows_avx2`.

## Retraction: the pack should not be repacked

An earlier version of the receipt named llama.cpp's `repack.h` 8-block-interleaved layout as the
next lever. That is wrong for this plane and is withdrawn:

- `U` is already 1 bit/term dense. One `u32` holds 32 sign terms and the per-row stride is
  `in/32` words, so a row's words are 128–352 B contiguous (the pack's `in` is 1024, 2048 or
  2816) and every cache line it touches is fully consumed.
- llama's interleave exists because a `Q8_0` block is 34 B (2-byte scale + 32-byte quants), so a
  block straddles lines and a row's blocks cannot be loaded as a unit. This plane has no scale in
  the row stream and no sub-line block.
- The nest's gain is instruction count: two plane loads (`codes`, `shift_vecs`) and the eight loop
  constants leave the row loop. A column's plane is at most ~6 KB and is L1-sized on the laptop and
  the box alike (`model`: no cache-miss counter was read), so the box cannot understate the effect
  through L3 capacity.

## Open questions, in order

Run each on the box before touching the laptop. A box ratio is `model` for the laptop; a claim
needs the laptop pairs.

1. **Confirm block 8 on the laptop.** The box says it leads: two alternating pairs, projection
   13,505.9 → 12,938.4 ms (1.044) and 13,581.3 → 12,809.5 ms (1.060), wall 1.029 and 1.042, with
   block 8 winning even when it ran last at load 5.44. The laptop's only block-8 run is unpaired,
   45.009 s inside block 4's 43.752–47.547 s range. Run two alternating laptop pairs at 1,827 and
   set `NQ_BLOCK = 8` only if the ratio clears ~1.02 in the same direction. The register budget is
   13 of 16 YMM lanes; if it clears, `NQ_BLOCK_MAX = 8` is already the clamp.
2. **If block 8 lands, which per-row-block cost binds next?** The inner loop per 32 terms per row is
   `vpsrlvd`, `vpshufb`, `vpshufb`, `vpand`, `vpmaddubsw`, `vpmaddwd`, `vpsllvd`, `vpaddd`, plus
   one scalar `words[r*stride + b]` load and its address arithmetic. Probe three candidates, one at
   a time, same binary behind a dial, 2 alternating pairs, projection phase only:
   - **address arithmetic**: four running row pointers instead of `r * stride + b` per row per
     block — removes two scalar ops per row-block;
   - **the nibble spread**: `vpsrlvd` + `vpshufb` turns one `u32` into 32 selector bytes. If a
     `vpshufb`-only form exists for the 8-nibble dispatch, it removes one p5-op per row-block;
   - **the dependency chain**: two blocks per iteration, so two independent chains are in flight
     and the `maddubs`→`madd`→`sllv`→`add` chain is not the loop's critical path.
   A candidate that does not move the projection by 1.02 on the box is dropped; do not carry three
   at once. `perf stat` on the box is acceptable if it is permitted there — one `instructions` and
   `cycles` count per arm settles which of the three is real before any A/B.

Do not start with the dot's arithmetic: the blocked path is already at 24.76 G/s at 1,827 against
the 33.60 G/s the 229 window reaches, and the gap is the KV stream and issue headroom, not the
select.

## The box loop

```sh
~/bin/box-push apeiron-nq                     # only git-status files, ~0.5 MB
~/bin/box-run apeiron-nq cargo build --release --locked --example prefix_solve
~/bin/box-run apeiron-nq bash /tmp/nq-screen.sh
```

The screen script lives at `/tmp/opencode/box-nq-screen.sh` locally and `/tmp/nq-screen.sh` on the
box; it alternates the arms and prints `cold` plus the projection leaves. Copy it with `scp` after
editing. The box runs one guarded job at a time (16 GiB cap, 8 jobs, 800% CPU); a lane does not
rsync into a tree another lane is building.

## Gates for a change here

- `cargo test --lib --locked engine::kern::nq` — the pooled/per-column equality and the twin
  equality must pass, on the modified reader.
- `cargo test --lib --locked engine::step::prefill_tests` — the prefill equals the per-column path.
- A mutation of the real source turns the equality red and restoring it turns the green.
- `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`, `cargo fmt --all
  -- --check`, `python3 tools/check-claims.py` on the box.
- The receipt names the phase, its share, and `1/(1 − s)`; a projection-only ratio with no wall
  change is stated as such.
