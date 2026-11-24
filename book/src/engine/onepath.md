# One path

The CPU engine has one projection path. `matmat_q8act` is its entry. A call takes a weight payload,
its geometry and `k` quantised activation columns. It walks the weight rows once; for each row it
decodes each 32-lane group once; each decoded group is folded against all `k` columns. Decode is
`k = 1`, prefill is `k > 1`, and the loop is the same.

## Dials

| dial | values | what it selects |
|---|---|---|
| carrier | `F16`, `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`, ternary, packed | weights per block, bytes per block, scale and code decode |
| columns | `1..4096` | activation columns folded per weight read |
| dot | AVX2, AVX-VNNI, portable | the 32-lane integer dot |
| accumulator | power-of-two `i64` lanes, checked `i128` | range proof and refusal route |

`ternary` is an exploratory activation dial, opened outside a product request and refuted on quality
(77,741,390 PPL against the same-binary q8act control's 72.988013); it is not a product carrier.

`format::quant::block_geom` is the one `(weights, bytes)` table. A carrier supplies the decode from
those bytes to `(w_scale, w_min, codes)`; `chart::FastBlockDecoder` is that dial. The row loop, the
thread split, the output band and the refusal assembly are shared.

A format is a row in the carrier dial. It is not a projection function, a second row loop or a
dispatch arm beside an existing kernel. Adding a bit depth adds a carrier, and a carrier that needs
another instruction adds a dot dial; neither adds a path.

## The loop

For row `r`, group `g` and column `c`, with `S[c,g]` the activation scale, `xq` the activation
codes, and `d[r,g]`, `m[r,g]` the weight scale and minimum:

```text
dot    = Σ_j q[r,g,j] · xq[c,g,j]
xsum   = Σ_j xq[c,g,j]
acc[c] += S[c,g] · (d[r,g] · dot − m[r,g] · xsum)
out[r,c] = round_shift_32(acc[c])
```

The group sums are `i32`; the group and block products are checked `i128`; one rounding returns
each output to Q32.32. A bound that cannot rule out an overflow sends the call to the per-column
oracle, so no refusal differs between the fast loop and the contract.

The weight bytes are read once per call, not once per column. A 42-token prompt on a `Q4_K_M` file
reads its 266.7 MB of `Q5_0` weights once. The per-column route reads them 42 times, 11.2 GB, which
is the first-token wait this loop removes.

## The oracle

`matvec_fixed` on the dequantised weights is exact and slow. It is the correctness oracle, not a
production route. An acceptance run keeps it to at most about ten tokens. A verification campaign
that needs thousands of tokens runs on a rented host, not on this laptop and not as a second path.

## Current state

| carrier | pooled loop | note |
|---|---|---|
| `Q8_0` | yes, AVX2 row kernel | AVX-VNNI pooled entry still beside the loop |
| `Q5_0` | yes, AVX2 row kernel | added with the prefill fix |
| `Q4_K` | yes, packed AVX2 lane | its sibling `gemm_q4_k` deleted; parity or better at `k = 32/128/512` |
| `Q6_K` | yes, packed AVX2 lane | its sibling `gemm_q6_k` deleted; parity at `k = 32/128/512` |
| `F16` | limb tile | sibling route; folds or becomes a carrier core |
| `Q2_K`, `Q3_K` | per-column oracle | no pooled carrier yet |
| ternary, packed | column group | `NqLinear::project_*_pooled`; factor rows read once per group |

Each row band writes its values straight into the `k` output columns, so the `n_out × k` result is
never staged interleaved and transposed. The band is why the 7B FFN up-projection, `18,944 × 3,584`
at `k = 256`, takes the shared loop: the old `n_out·k ≤ 2^22` staging cap refused it.

The pooled `Q8_0` VNNI entry is the remaining sibling debt: it is the shape this chapter forbids,
kept until the fold. New work does not copy it. `Q4_K` folded in two steps: the packed `i64` lane
replaced the exact `i128` carrier as the production core (5.8–9.6× slower before,
`bench/cpu/1794806002-q4k-carrier.md`), then the direct-write band replaced the interleaved staging
and transpose (`bench/cpu/1794807761-q4k-direct-write.md`). In one interleaved release run on this
host the shared loop is 0.89×, 0.94× and 1.00× of the deleted entry's wall at `k = 32/128/512`, so
the entry deletes rather than stays in front. `Q6_K` folded the same way: its sibling `gemm_q6_k`
is deleted, the shared carrier's wide bound replaces its narrow bound, and in two idle release
runs on this host the shared loop is 0.89–1.11× of the deleted entry's wall at `k = 32/128/512`
(`bench/cpu/1794810316-q6k-carrier.md`). Calls above `k = 4096` fall to the per-column contract. The
exact `i128` fold remains inside each lane as the past-`limit` fallback, not as a second row loop.

The `Q8_0` VNNI entry is not a duplicate of the shared row kernel. It transposes the activation panel
once per call, folds eight columns per lane, tiles two rows against shared activation loads, and
keeps a power-of-two `i64` accumulator with one rounding. It admits only AVX-VNNI, power-of-two
scales and `n_in` within 512 blocks, and its output equals the per-column contract
(`gemm_equals_the_per_column_contract`). The fold therefore adds those as dial values — packed
activation panel, eight-column lanes, two-row tile, power-of-two accumulator — and keeps the AVX2
twin. Deleting the entry before the shared loop carries them moves every `Q8_0` `k > 1` call to a
slower kernel, and no value test would catch the loss.
