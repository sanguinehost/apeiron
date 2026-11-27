# The eight-column tile at every width

Criterion: **T1** decode rate and **T2** turn latency, through the projection tile's participant
gate. Grade: `measured` on the OVH box (CPU tier, 8 vCPU EPYC Milan). The laptop timed pair is
still blocked (`MemAvailable` 678 MiB against the 1 GiB guard), so this is box evidence; the
ratios are paired within one session.

## Result

`parse_cols` defaulted to the two-column pair kernel below four participants. The partial-group
change (`594a26fe`) now serves 2..=7 columns on the tile, so the gate only cost the low-width runs.
Removing it (the default is the tile at every width):

| threads | pair, ms/token | tile, ms/token | ratio |
|---:|---:|---:|---:|
| 1 | 26.73–27.17 | 20.60–20.63 | **1.30–1.32×** |
| 2 | 16.82–17.48 | 13.82–14.63 | **1.20–1.22×** |
| 4 | 10.41–10.52 | 9.68–9.74 | **1.08×** |
| 8 | 7.02–7.11 | 7.01–7.02 | flat |

Decode is `lookup_accept --lookup-only --draft 8`, 130 tokens, 28 rounds; pair is
`APEIRON_NQ_COLS=2`, tile is the default. Prefill, `prefill_width --tokens 1827`, paired:

| threads | pair, ms | tile, ms | ratio |
|---:|---:|---:|---:|
| 2 | 19,767–19,934 | 12,829–12,885 | **1.54×** |
| 4 | 10,121–10,341 | 6,305–6,319 | **1.61×** |

Exactness: `lookup_accept` reported `exact`, 102/102, at every width; `engine::kern` 88 passed;
PPL is **238.498293** on 2,047 tokens, mean nll 5.474362, in both arms — the tile is bit-equal to
the pair per column, so the rate is the only change. The PPL run itself reads 177.5 → 214.3 tok/s.

## Mechanism

The origin commit (`5079a597`) measured the tile at **0.867×** one participant and gated it to four
or more. That measurement predates partial groups: a band of nine columns folded eight on the tile
and sent the ninth to the pair kernel, and the wide loop's setup bought less than the leftover cost.
With the partial group the whole band of 2..=7 columns is on the tile, whose measured rate is
24.5 MACs/cycle at two columns and 35.8 at five, against the pair kernel's 15.9–16.7
(`the_tile_beats_the_pair_kernel_at_a_verify_width`). The gate now only selects the pair for an
explicit `APEIRON_NQ_COLS` in `0..8`.

## Scope

- Box only; no laptop row. The laptop is `nproc = 2`, so the width-2 row is its shape, but the
  timed claim waits for the guard.
- Box drift: width-8 decode reads 7.0 ms/token in this session against 5.8 in
  `bench/cpu/1795773686-nq-partial-column-group.md`; every ratio above is interleaved in one
  session, and the width-8 row is flat in both.
- Width 8 is unchanged because `band = ceil(9/8) = 2` already took the tile through the old
  default (`width >= 4`); the win is at 1, 2 and 4 participants.
- Prefill is the full layer pass without the head; decode is the verify pass.
- Rust/LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.
