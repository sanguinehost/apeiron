# NQ projection: decision on the row block and the next lever

Receipts: `bench/cpu/1795518973-nq-blocking.md` (the row block), `bench/cpu/1795557505-nq-two-column.md`
(the two-column reader and the new default). This note is the handoff, not evidence.

## Decision

`NQ_BLOCK = 8` is the default (`src/engine/kern/nq.rs`). The earlier receipt kept 4 because the
laptop's single-column block-8 runs were **inside** the block-4 arm's own spread (`1795518973`,
§ Block 8: 2.3% below the bracketing mean, but the two block-4 runs differ by 2.8% and the sequence
is not paired). This change moves the default on a paired, same-binary box screen at 8 participants
with the two-column reader: block 8 13.922 / 14.044 / 14.016 s against block 4 15.542 / 15.704 /
15.638, pairs 1.157 / 1.118 / 1.098, mean **1.124**. The 0.6B shapes divide by 8 (rank 736, outputs
1024/2048/3072), so no stage pays a short remainder block. The laptop confirmation of block 8 is
**open** (AC absent), and the prior laptop evidence does not contradict the direction: it was
inconclusive, not negative.

The two-column reader (`code_select_block2_avx2`) is the production Q8 path: one weight-word
expansion against two activation columns. Do not add a second row loop, a `gemm_*` sibling or a
third column. The pair is 13 vector instructions per `(row, block, 2 columns)`; a third column adds
four more against the same five-instruction expansion, but `R = 8` already needs sixteen
accumulators and the five constants, so a triple cannot fit the sixteen YMM registers.

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

1. **Confirm block 8 on the laptop.** The box with the two-column reader says it leads by 1.124
   (three alternating pairs above). The earlier single-column laptop sequence is block 4 40,682 ms,
   block 8 40,208, block 8 40,394, block 4 41,820 (walls 56.593, 56.056, 56.270, 58.205 s): block 8
   is 2.3% below the bracketing mean, but those two block-4 runs differ by 2.8% and the sequence is
   not alternating. Run two alternating laptop pairs at 1,827 on this binary; keep 8 if the wall
   ratio clears 1.02 in the same direction, otherwise revert the constant and keep the pair as the
   dial. The pair at `R = 8` holds sixteen accumulators plus five constants, so a spill is expected;
   the box microbench shows block 8 (101.2 G terms/s at width 8) below block 7 (118.7), and the cold
   turn still wins on the remainder that block 7 pays.
2. **If block 8 lands, which per-row-block cost binds next?** The inner loop per 32 terms per row is
   `vpsrlvd`, `vpshufb`, `vpshufb`, `vpand`, `vpmaddubsw`, `vpmaddwd`, `vpsllvd`, `vpaddd`, plus
   one scalar `words[r*stride + b]` load and its address arithmetic; the pair shares the first five
   across two columns. Probe three candidates, one at a time, same binary behind a dial, 2
   alternating pairs, projection phase only:
   - **address arithmetic**: four running row pointers instead of `r * stride + b` per row per
     block — removes two scalar ops per row-block;
   - **the nibble spread**: `vpsrlvd` + `vpshufb` turns one `u32` into 32 selector bytes. If a
     `vpshufb`-only form exists for the 8-nibble dispatch, it removes one p5-op per row-block;
   - **the dependency chain**: two blocks per iteration, so two independent chains are in flight
     and the `maddubs`→`madd`→`sllv`→`add` chain is not the loop's critical path.
   A candidate that does not move the projection by 1.02 on the box is dropped; do not carry three
   at once. `perf stat` on the box is acceptable if it is permitted there — one `instructions` and
   `cycles` count per arm settles which of the three is real before any A/B.

Do not start with the dot's arithmetic. The reader measures 24.9 G terms/s on one participant and
101.2 G/s at eight with the pair at block 8 (`the_pooled_block_rate_by_block`, pack, stages and
gather included); the single-column count was 10 instructions per 32 terms per row, and the pair's
13 instructions per two columns put it near 7.5 per column. The bound those counts imply was
computed against the whole-turn instruction rate in `1795518973`; no `perf` counter was read for
the pair, so which port binds is open.

## The box loop

```sh
~/bin/box-push apeiron-nq                     # only git-status files, ~0.5 MB
~/bin/box-run apeiron-nq cargo build --release --locked --example prefix_solve
~/bin/box-run apeiron-nq bash /tmp/nq-screen.sh
```

The screen script is on the box at `/tmp/nq-screen8.sh` (two alternating pairs, 1,827 + 16); the
local copy is transient, and the receipt's Evidence block is the command to recreate it. The box
runs one guarded job at a time (16 GiB cap, 8 jobs, 800% CPU); a lane does not rsync into a tree
another lane is building.

## Gates for a change here

- `cargo test --lib --locked engine::kern::nq` — the pooled/per-column equality and the twin
  equality must pass, on the modified reader.
- `cargo test --lib --locked engine::step::prefill_tests` — the prefill equals the per-column path.
- A mutation of the real source turns the equality red and restoring it turns the green.
- `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`, `cargo fmt --all
  -- --check`, `python3 tools/check-claims.py` on the box.
- The receipt names the phase, its share, and `1/(1 − s)`; a projection-only ratio with no wall
  change is stated as such.
