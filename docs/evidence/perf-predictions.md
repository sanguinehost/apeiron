# Performance predictions

Row `perf-theory-formal-cost-HiI4vx6ezYPSHkfGWbq7YX` (W3). A prediction is committed before the measurement it concerns is read; the commit order is the evidence. The model under test is the roofline of `proof/corpus/Perf/Roofline.lean` (`timeLB f = max_r d_r(f)/c_r`, chain term inside the max per `Perf/Concurrency.lean`). Grade of every row until its measurement is read: `anchor`.

| id | case | cost equation | prediction | refuted if | measured | verdict |
|---|---|---|---|---|---|---|
| P1 | ATTN-HD128: online softmax with one K/V walk at hd=128, node ratio old/new | binding field is shared-slab bytes; walks 3 to 1; the FMA term is unchanged and bounds the ratio below 3 (`walk_mix`); hd=128 forces a smaller tile, which lowers reuse per staged slab | point 1.8, band [1.5, 1.95] | outside [1.4, 2.05] over at least 5 interleaved rounds | open | open |

## P1 provenance

- Basis: hd=64 measured node 2214 us to 1160 us, ratio 1.91 (the epic's evidence table); same binding field.
- A first version of this prediction used a recalled walk count of 3 to 2 (band 1.0 to 1.5). It contradicted the hd=64 measurement and was corrected before any hd=128 number was read. The miss was a count recalled instead of read from the kernel.
- The measurement is `ATTN-HD128` (board row `inf-resident-prefill-GEB633UfORCK7XSmzJifHd`), paired, at least 5 rounds, forced rebuild, through `cargo-gate.sh`.

## Held-out cases still to predict

- the `b`-parameterised prefill GEMM (UNIFIED-CARRIER W3)
- a map layout change (compact tag array)
