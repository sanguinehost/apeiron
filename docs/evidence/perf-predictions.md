# Performance predictions

Row `perf-theory-formal-cost-HiI4vx6ezYPSHkfGWbq7YX` (W3). A prediction is committed before the measurement it concerns is read; the commit order is the evidence. The model under test is the roofline of `proof/corpus/Perf/Roofline.lean` (`timeLB f = max_r d_r(f)/c_r`, chain term inside the max per `Perf/Concurrency.lean`). Grade of every row until its measurement is read: `anchor`.

| id | case | cost equation | prediction | refuted if | measured | verdict |
|---|---|---|---|---|---|---|
| P1 | ATTN-HD128: online softmax with one K/V walk at hd=128, node ratio old/new | binding field is shared-slab bytes; walks 3 to 1; the FMA term is unchanged and bounds the ratio below 3 (`walk_mix`); hd=128 forces a smaller tile, which lowers reuse per staged slab | point 1.8, band [1.5, 1.95] | outside [1.4, 2.05] over at least 5 interleaved rounds | none: the old arm does not exist at hd=128 (exact tile K slab 65536 B > `WG_SHARED_MAX` 49152 B) | untestable as stated |

## P1 provenance

- Basis: hd=64 measured node 2214 us to 1160 us, ratio 1.91 (the epic's evidence table); same binding field.
- A first version of this prediction used a recalled walk count of 3 to 2 (band 1.0 to 1.5). It contradicted the hd=64 measurement and was corrected before any hd=128 number was read. The miss was a count recalled instead of read from the kernel.
- The measurement is `ATTN-HD128` (board row `inf-resident-prefill-GEB633UfORCK7XSmzJifHd`), paired, at least 5 rounds, forced rebuild, through `cargo-gate.sh`.

## P1 outcome

Board row `inf-resident-prefill-GEB633UfORCK7XSmzJifHd` (ATTN-HD128); receipt `/home/socol/oc-tmp/escalation/ATTN-HD128-RECEIPT.md`.

- Command (live tree, `touch src/lib.rs`, shared target via `cargo-gate.sh`, 5 interleaved rounds, qwen2.5-1.5b q8, `PREFILL_PROFILE=1 cargo run --release --features gpu --example gpu_decode_bench -- <gguf> --prefill 512 --reps 5 --carrier q8`; logs `/home/socol/oc-tmp/escalation/HD128-bench-{untiled,tiled}-{1..5}.log`):

| arm | attn node us, mean [min, max] | pp512 tok/s, mean [min, max] |
|---|---|---|
| `DEC_ATTN_TILE=0` (untiled) | 10491.2 [10270.3, 10815.4] | 16114.5 [15461.7, 16417.9] |
| `attn_tiled_16_4_4_kp4_false_m3` | 1805.9 [1799.8, 1814.8] | 22286.0 [21300.8, 23877.4] |
| ratio | 5.81 | 1.383 |

- The predicted pair (exact 3-walk tile against online one-walk tile) cannot be built at hd=128, so P1 is neither confirmed nor refuted. The measured pair is untiled against online dim-split tile (`kp=4`, `rows=16`); the model made no prediction for it, so 5.81 is a post-hoc figure and carries no predictive grade.
- Rule from P1: a prediction names a baseline that can be constructed on the measured configuration.
- The tile that exists at hd=128 (`rows=16`, `kp=4`) also changes reuse per staged byte (FMAs per staged byte `rows/kp` against `rows/2` for the fp16 slab); the model has no row for that term yet.

## Held-out cases still to predict

- the `b`-parameterised prefill GEMM (UNIFIED-CARRIER W3)
- a map layout change (compact tag array)
