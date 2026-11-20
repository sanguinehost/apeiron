# The 1.174 b/w head: the 447 MB/token that keeps T1 under 20

Criterion: **Q1** (the 1.15–1.18 b/w 7B pack) and **T1** (decode). Grade: `model` for the byte and
rank arithmetic (the formula and `rank_for_bits` are source; neither the head fit nor the pack
exists), `anchor` for the format and fit-path design. Host: i5-10210U, 4C/8T, 6.6 GiB,
`performance`, 27.97 GB/s measured, 42.656 GB/s spec. This note changes no code and closes no
criterion.

## 1. The gap

The 1.174 b/w campaign (`bench/cpu/1794781140-nq7b-118-full-pack.md`) replaces the seven per-layer
linears: 6,525,288,448 weights, 957,615,232 B. It does **not** include `output.weight`. The source
GGUF stores that head as `Q6_K`: 152,064 × 3,584 = 544,997,376 weights, 447,068,160 B, read in
full on every token by `Engine::lm_head` (`src/engine/step.rs:1231`).

Per token: `957,615,232 + 447,068,160 = 1,404,683,392 B`. At the measured 27.97 GB/s that is
**19.9 tok/s**, which is 20 tok/s to a reader's precision: the head consumes the whole floor and
leaves **zero margin**, so 20 tok/s would need 100.2% of the measured streaming mean and 30 tok/s
needs 42.14 GB/s = **98.8% of spec peak** (`1794799742-decode-ceiling.md`, §5). A zero-margin
configuration is not reachable in practice — the 0.5B demo holds 49–57% of its own bytes ceiling —
but the reason is the absent margin, not a 0.4% shortfall. Adding a 2k KV makes it worse.

The head is 31.8% of the per-token bytes. No kernel change recovers it; it must leave the dense
payload.

## 2. The head at the same rate

`rank_for_bits(152064, 3584, 1_180_000, 32)` = **4096** (a+b = 155,648). The section's stored bits
are `32·(o·⌈r/32⌉ + r·⌈n/32⌉) + 16·(n + r + o)`:

| term | bits | bytes |
|---|---:|---:|
| `U`, `V` signs, rank 4096 | 4096 × 155,648 = 637,534,208 | 79,691,776 |
| three scale vectors | 16 × 159,744 = 2,555,904 | 319,488 |
| section header | — | 36 |
| **head** | **640,090,112** | **80,011,264** |

The head's own rate is 640,090,112 / 544,997,376 = **1.17448 b/w**, the trunk's rate. The pack
becomes 957,615,232 + 80,011,264 + 36 = **1,037,626,532 B** over 7,070,285,824 weights =
**1.17404 b/w**, inside Q1's 1.15–1.18 interval. The 28 native layers keep their 7 kinds; the head is one extra
section, not a 29th layer.

| per-token read | B | tok/s at 27.97 GB/s | tok/s at 42.656 GB/s |
|---|---:|---:|---:|
| trunk + Q6_K head (today) | 1,404,683,392 | 19.9 | 30.4 |
| trunk + NQ head | 1,037,626,532 | **26.95** | **41.1** |
| trunk + NQ head, 2k F16 KV (114.7 MB) | 1,152,326,532 | 24.3 | 37.0 |

At 20 tok/s the NQ-head pack needs 20.75 GB/s, **74.2%** of the measured streaming mean; at 30 tok/s
it needs 31.13 GB/s, **111%** of measured and **73.0%** of spec. The accepted floor becomes a
bandwidth-utilization question with 26% headroom instead of a bytes refusal. The formal 30 tok/s
target still needs AC/cooler bandwidth or a lower-b/w head, as §5 of the ceiling receipt says.

## 3. What has to change

1. **Format** (`src/format/nq.rs`). Add the head as a section outside the `layers × kinds` grid.
   The clean shape is version 4: after the trunk sections, the header's reserved word flags one
   optional head section, written with the existing `PackedLinear` layout. A pseudo-layer
   (`layers = 29`, key `(28, kind)`) is smaller but lies about the model's layer count and breaks
   `build_nq_table`'s `pack.layers == config.n_layer` check (`src/engine/loader/nq.rs:100`); a
   reserved bit on the existing header is the smaller honest change. `read` keeps accepting v2/v3.
2. **Loader** (`src/engine/loader/nq.rs`). Store the head as `Option<NqLinear>` beside the
   `layer × NQ_KINDS` table; `nq_storage` counts its factors. No `TensorKey` is needed: the head is
   not a block linear.
3. **Engine** (`src/engine/step.rs`). `lm_head_fixed_sched` checks the head slot before reading the
   GGUF payload and calls `NqLinear::project` on the final hidden state. The output is already
   `Vec<i64>`; the sign-rank epilogue is the same one the trunk uses. The `lm_head` payload is then
   only the skeleton fallback when a pack omits the head.
4. **Calibration** (`src/learn/fold/stats/heldout.rs`, the `nq_gram_diag_write` path). The fit
   needs only `i_norm` (an `inn`-vector) and `o_norm` (an `out`-vector), not a full Gram
   (`factorize`'s signature, `src/learn/nanoquant.rs:534`). `o_norm` is the head's row norms, read
   from the GGUF. `i_norm` is the final-norm output's per-coordinate RMS over the same 17,920
   calibration tokens: one extra vector capture at the head input, summed in f64, not a
   3584 × 3584 Gram.
5. **Fit and pack** (`src/learn/nanoquant`, `tools/nq_pack_factors.py`, `pack.sh`). One fit at
   T = the campaign's T, rank 4096, then write the head section. The factor's cost is
   `O(r²(o + n) + 2·r·o·n)` per iteration; at the measured 2.8G MAC/s the 18944 × 3584 rank-3520
   `gate`/`up` fit takes 270 s, so the head (152,064 × 3,584, rank 4096) is about
   **2,500 s per iteration** — +42 min per outer iteration on top of the 6.6 h T = 1 campaign,
   about +10%.

## 4. The risk

The head is the most quality-sensitive matrix in the model: a sign-rank approximation is fitted
against the dense logits, not against the trunk's hidden state, and Q2 is measured on the whole
model. Two failure modes are named and not measured:

- A rank-4096 factor at 1.174 b/w may lose more KL than the trunk does at the same rate. The
  fallback is residual layers, which raise the head's rate (v3 `res_layers` already exist in the
  format) and eat the bytes margin.
- The final-norm input distribution is heavy-tailed; the `scales` floor (`FLOOR_REL`) that keeps a
  dead coordinate out of the limb kernel's exponent range may bind at rank 4096.

Both are Q2 questions. The head fit must run once and be scored before the 1.174 b/w pack is called
a T1 candidate; a pack without the head has zero margin at 20 tok/s and cannot reach 24 at the
measured bandwidth, while the NQ head brings the ceiling to 26.96 and leaves 26% margin at 20, so the
campaign should carry the head from the start rather than discover it at the end.

## 5. What this does not establish

- No head factor exists; `rank_for_bits` is a formula, not a measurement.
- The quality effect on PPL/0-shot is unknown; the fit cost is extrapolated from the `gate`/`up`
  wall, not measured on 152,064 rows.
- The 4-bit comparison is unaffected: a Q4_K head is 306 MB and still leaves the pack above the
  measured 20 tok/s bytes budget (1.264 GB → 22.1 tok/s at 27.97 GB/s), so it is a fallback, not the
  target.
- rustc/LLVM codegen, the kernel ABI and the GPU remain outside every claim here.
