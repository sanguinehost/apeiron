# A 1.174 b/w 0.6B pack: the rank table, the fit bill, and why the full pack is not in this tree

Status: the v3 ranks and bytes are `measured` from `/tmp/opencode/nq06-v3.nqp` (196 sections parsed,
53,815,472 B); the 1.174 table is `model` over the format's byte layout (`src/format/nq.rs`,
`NqFactor::stored_bits`) and the published fit walls of `bench/cpu/1794740406-nq-118-distortion.md`
§5, except the probe rows §5 marks `measured` on this laptop. The v3/Q4_K_M pair in §6 is `measured`
in the other lane's logs and cited, not re-run. Criterion: **Q1** (rate) directly and **Q2**
(quality) as the PPL it cannot yet reach. Scope: `Qwen3-0.6B-Base`, 28 layers, the 196 linears of
`q k v o gate up down`. No full 1.174 pack exists; §5 is the probe and the fit bill, §6 the PPL
refusal.

## 1. The v3 pack is 0.978 b/w, below the Q1 target

`/tmp/opencode/nq06-v3.nqp` is 53,815,472 B over 440,401,920 replaced weights
(`28 · (2048·1024 + 1024·1024 + 1024·1024 + 1024·2048 + 3·3072·1024)`):
**0.978 b/w**, not 1.03. The 1.03 figure counts the pack's bits against the 418.4M sign terms a
token touches (`r·(out + in)` summed), a different denominator. The pack stores three bf16 scale
vectors in addition to U and V, so its rate over the weights it replaces is the lower number.

The Q1 target is 1.15–1.18 b/w. v3 is **below** it by 15–17%. The 1.174 pack in this note is
therefore a *higher* rate than the pack the other lane is scoring. v3's measured screen is **2.52×**
Q4_K_M at 8176 tokens (§6), so the second branch of the comparison applies: the 1.174 pack must
supply the 1.68× PPL reduction that closes the gap to 1.5×. Whether its 0.196 b/w headroom does
that is not measured here.

## 2. The arithmetic

Per linear, with `out` rows, `in` columns and rank `r`, every sign plane bit-packed at 32 bits per
u32 word and every scale bf16:

```text
stored_bits = 32·(out·⌈r/32⌉ + r·⌈in/32⌉) + 16·(in + r + out)     # U, V, s_pre/s_mid/s_post
residual   = 32·out·⌈in/32⌉ + 16·out                                # one dense sign layer
salient    = 32 + 16·out                                            # one sparse column
```

`python3 tools/nq_rate.py` prints the tables. The 0.6B kinds and their per-layer rates at v3's
ranks:

| kind | shape | v3 rank | v3 b/w | balanced rank | balanced b/w | 1.174 rank | 1.174 b/w |
|---|---|---:|---:|---:|---:|---:|---:|
| q | 2048×1024 | 640 | 0.9658 | 768 | 1.1543 | 800 | 1.2014 |
| k | 1024×1024 | 480 | 0.9761 | 576 | 1.1650 | 608 | 1.2280 |
| v | 1024×1024 | 480 | 0.9761 | 576 | 1.1650 | 608 | 1.2280 |
| o | 1024×2048 | 640 | 0.9658 | 768 | 1.1543 | 800 | 1.2014 |
| gate | 3072×1024 | 736 | 0.9829 | 864 | 1.1502 | 864 | 1.1502 |
| up | 3072×1024 | 736 | 0.9829 | 864 | 1.1502 | 864 | 1.1502 |
| down | 1024×3072 | 736 | 0.9829 | 864 | 1.1502 | 864 | 1.1502 |
| **whole** | 440,401,920 w | | **0.9774** | | **1.1533** | | **1.1743** |

The "balanced" column is exactly `rank_for_bits(a, b, 1_174_000, 32)` (`src/learn/nanoquant.rs`):
q/o 768, k/v 576, gate/up/down 864. It is **1.1533 b/w on the pack's own formula**, not 1.174:
`rank_for_bits` subtracts 16 from `a·b·ppm/(a+b)` for the two scale vectors of the reference's
export, while this format stores three and charges `16·r` for `s_mid`. The 0.021 gap is that
term. The same 0.02 appears in the 1.18 campaign, which calls rank 864 "1.18 b/w" while the pack
it wrote is 1.1510 b/w (`1794740406-nq-118-distortion.md` §1).

The 1.174 ±0.002 column adds one 32-block to each of q, k, v and o, and leaves the FFN at its
per-kind budget: `rank_for_bits` plus 32 = **800/608/608/800/864/864/864**. The payload is
64,642,816 B. A pack with no residual section is written as version 2 (`write`, `src/format/nq.rs`),
whose section header is 32 B, so the 196 headers and the file header add 6,304 B: the pack is
**64,649,120 B = 1.17425 b/w payload, 1.17437 b/w file**. That is inside the ±0.002 band
(payload err 0.00025, file err 0.00037). The next 32-step on any kind moves the rate by
0.006–0.008 b/w, so there is no table between 1.166 and 1.174 that stays with the three rank
values; the four attention ranks are where the rounding lands. Other distributions inside the band exist (e.g. q 768, o 832, down 896),
but no per-kind quality measurement distinguishes them; the symmetric attention choice is the one
this note carries.

## 3. v3 against the balanced table

v3's ranks are a scaled-down copy of the balanced table, but its per-kind rates are not flat:
0.966 (q, o), 0.976 (k, v) and 0.983 (gate, up, down), a 0.966–0.983 band under a 0.978 mean.
The balanced table's per-kind rates spread the other way, 1.150 (FFN) to 1.165 (k, v), because the
`16·(in + r + out)` scale term is a larger fraction of a small matrix. v3 is **0.833–0.852× the
balanced rank of every kind** (q/o 640/768 = 0.833, k/v 480/576 = 0.833, FFN 736/864 = 0.852), so
the claim that v3 over-spends on k/v relative to q/o is not what the bit counts say: the two
allocations are the same policy at 0.83–0.85× the rate. What is true is the headline: v3's
0.978 is below the Q1 band. The band already holds one 0.6B pack — the shipped SamsungLabs export
at 1.166 b/w on the twelve campaign rows, with a rank per row
(`1794740406-nq-118-distortion.md` §1) — so the 1.174 table is a second allocation inside it, flat
across the seven kinds rather than per row.

## 4. The alternative: spend the bits on residual layers

A dense sign-residual layer is `32·out·⌈in/32⌉ + 16·out` per linear: **1.0125 b/w** over all 196
linears (445,906,944 bits, 55,738,368 B), because the padding to 32 columns adds 0.0125. Above
v3's 0.977, the 1.174 budget leaves `(1.174 − 0.977)·W = 86.6M` bits, i.e. **0.194 of one
residual layer**: a layer on 5.4 of the 28 layers, or on the q linears alone (0.68 b/w), or on the
down linears alone (1.005 b/w, already over the delta). There is no allocation that gives every
linear one residual bit at 1.174.

The repo's own residual measurement says one layer is worse than none: on the 8B pack, r0 (v2,
1.124 b/w) 28.3076 PPL, r1 (one refit residual layer, 2.127 b/w) 51.1533, r2 (two layers,
3.1306 b/w) 16.2953, against Q4_K_M 16.1247 over the same 1008 tokens
(`bench/notes/nq-residual-layer.md` §8). The second bit is what recovers the first, and that costs
2.025 b/w. At 1.174 the pack cannot buy even the first, and the measured one-layer result is a
3.17× regression, not a gain. **The rank table is the only allocation the 1.174 budget supports.**

## 5. The probe and the fit bill

Receipt `1794740406-nq-118-distortion.md` §5, Raptor Lake i5-13400F, 16 vCPU, ten pool
participants, T=400: 161.8–174.7 s for a rank-864 `down`/`gate`/`up`, 93.8–113.1 s for a rank-768
`attn_output`. Two measured scales:

| linear | rank | T=400 wall (Raptor Lake) |
|---|---:|---:|
| down, gate, up | 864 | 161.8–174.7 s |
| attn_output | 768 | 93.8–113.1 s |

Interpolating the data pass (time ≈ linear in rank at these ranks, the `r²` Gram and inverse terms
being the residue) gives the 1.174 table's per-layer bill: q 800 ≈ 108 s, k/v 608 ≈ 82 s each,
o 800 ≈ 108 s, gate/up/down 864 ≈ 168 s each → **≈ 884 s per layer, 6.9 h for the 196 linears on
that host at T=400**; T=200 halves it to 3.4 h.

The probe is layer 0's seven kinds at the 1.174 table, `examples/nq_fit.rs` (new, an example, no
library change), T=10, column-RMS preconditioner (the activation `diag(H)` cache is not on this
laptop; `nq_fit`'s path needs `~/kld08`, which does not exist here):

```sh
/home/lucas/Workspace/tools/agent-guard run -- cargo run --release --locked --example nq_fit -- \
  --model ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base-F16.gguf --layers 0 \
  --kinds q,k,v,o,gate,up,down --iters 10 \
  --ranks q:800,k:608,v:608,o:800,gate:864,up:864,down:864 \
  --out /tmp/opencode/nq06-r174-l0-t10.nqp --rel
```

Measured on this laptop (i5-10210U, 2 visible CPUs, default pool width 8 capped at 2 participants,
guard cap 1162–1169 MiB), three runs; logs `/tmp/opencode/nq06-r174-t10.log` and
`/tmp/opencode/nq06-r174-t10-r23.log`. The three packs are byte-identical
(`sha256 f7adb2ef…82818c`), so the fit is deterministic and only the wall varies; the wall is the
mean with the observed range:

| kind | rank | wall T=10, mean (range) | ×40 → T=400 | `admm_loss` | `relF` |
|---|---:|---:|---:|---:|---:|
| q | 800 | 65.8 s (53.2–77.8) | 43.9 min | 5.99026 | 1.68976 |
| k | 608 | 22.9 s (21.1–23.9) | 15.2 min | 4.50561 | 2.14998 |
| v | 608 | 23.9 s (22.5–25.3) | 15.9 min | 7.26298 | 2.87178 |
| o | 800 | 59.8 s (59.5–60.4) | 39.9 min | 10.65034 | 2.59190 |
| gate | 864 | 98.2 s (91.6–104.0) | 65.5 min | 4.03916 | 1.42614 |
| up | 864 | 94.0 s (92.1–96.3) | 62.7 min | 14.56236 | 2.26382 |
| down | 864 | 93.7 s (92.4–96.1) | 62.5 min | 13.81052 | 1.91898 |
| **layer** | | **458.3 s (433.1–475.0)** | **5.09 h** | | |

The fit-only layer total is 458.3 s; the driver's printed 468.8 s mean adds the model open and pack
write, which happen once per pack, not once per layer. The run-to-run spread is 433.1–475.0 s
(±4.6%). The q row carries the widest spread (53.2–77.8 s) although `o` is the same 2048×1024
rank-800 shape and is stable at 59.5–60.4 s; the per-kind means are scheduling noise, not a shape
effect. The ×40 column is an upper bound on the T=400 wall: the setup term (limb prepare, the
first step and the one-shot `--rel`) is inside the T=10 wall and is multiplied by 40 as well. On the
rented host the setup is 2.5 s of the 168 s `down` fit at T=400 (the T=50/200/400 line in
`1794740406-nq-118-distortion.md` §5); the same setup is a larger share of a T=10 wall on this
laptop, so the true bill is below the ×40 column. At 28 layers that is **143 h (5.9 days) at
T=400**, with the run spread **135–148 h**, and **71 h at T=200** (67–74 h).

That is **≈20.7× the rented host's 6.9 h**, not the 4–8× the brief assumed. The two anchored rows —
`attn_output` at rank 768 (93.8–113.1 s measured) and the FFN at rank 864 (161.8–174.7 s) — are
22–23× slower when the probe's mean T=10 wall is multiplied by 40; the difference is the
2-participant pool against the rented ten plus the missing AVX-VNNI and clock. A T=10 layer fits
the brief's 20-minute cap (458.3 s, 7.6 min), but a full pack at T=10 is 28× that (3.56 h) and a
T=10 factor is not a quality artifact: `admm_loss` is 4.04–14.56 and `relF` 1.43–2.87 under the
column-RMS weighting, and the campaign's reference budget is T=400 (T=200 is within 1.3%). At T=200
the layer alone is 2.55 h and the pack ≈71 h. The full pack on this laptop is **3.6–7.2× the
brief's 20–40 h** at the mean, so it is not started and no PPL follows.

The probe pack is `/tmp/opencode/nq06-r174-l0-t10.nqp`: **2,308,928 B, 7 sections, 1.174251 b/w**.
`write` emits version 2 with 32-byte section headers, the same arithmetic that gives the
196-section pack 64,649,120 B in §2. The seven sections are not a skeleton image (§6).

## 6. PPL: refused, with the v3 screen as the reference

No 196-linear 1.174 pack was built in this turn. A partial pack cannot be scored on
`skeleton-q8head.gguf`: that image omits all 196 linears (`115` tensors: 112 block norms,
`token_embd`, `output_norm` and the Q8_0 `output.weight`), and `build_nq_table`
(`src/engine/loader/nq.rs`) accepts a partial pack only when the skeleton still carries the
linears it omits. The PPL of the 1.174 pack therefore remains **unmeasured**. The v3 screen the
other lane ran is the reference the 1.174 pack would have to beat; its receipt
`bench/cpu/1794822341-nq06-ppl-vs-q4.md` (logs under `/tmp/opencode/nq06-ppl-logs/`) is cited here,
not re-run:

| model | wikitext-2 PPL | source |
|---|---:|---|
| Q4_K_M | 21.715296 | `/tmp/opencode/nq06-ppl-logs/ppl-q4-8192.log` |
| v3, 0.978 b/w | 54.806671 | `/tmp/opencode/nq06-ppl-logs/ppl-nq-q8act-8192.log` |
| ratio | **2.524×** | `1794822341-nq06-ppl-vs-q4.md` |

The earlier 2044-token screen gave 18.118871 and 74.296766, 4.101×; the longer run moves both and is
the one used here. v3 is not under 1.5× Q4_K_M, so the 1.174 pack has room to improve: on this pair
the 1.5× screen is ≤ 32.572944, and v3's 54.806671 must fall by **1.68×**. The added 0.196 b/w is a
+20% rate step (+25% rank on q/o, +27% on k/v, +17% on the FFN); no measured rank-to-PPL curve on
the 0.6B says whether it buys 1.68×. The only same-family PPL-versus-bits measurement is the 8B residual
sweep (+2.0 b/w for 1.74×, §4), which buys residual planes rather than rank; as an anchor it says
the 0.196 b/w step is unlikely to close the gap on its own, and it is not a measurement of this
pack. Q2 stays open: the screen is at ctx 512, below Q2's 2048, and the 1.174 pack is unbuilt.

The minimum next step that produces the 1.174 row: fit all 196 linears at the table in §2 on a host
where the bill is 6.9 h (T=400) or 3.4 h (T=200) — not this laptop, where the probe says 143 h
(135–148) and 71 h — write the pack, and run
`examples/ppl --model skeleton-q8head.gguf --pack <pack> --tokenizer ... --tokens 8192 --ctx 512
--head-batch 16 --kernel q8act`. At T=200 the 0.6B row's norm penalty is 1.3% above T=400
(measured on layer-0 `down` in `bench/notes/fit-cost.md`); at T=50 it is 7.4%. T=200 is the
smallest budget that keeps the artifact within 1.3% of the reference iteration count.

## 7. What this does not cover

- The PPL of any 1.174 pack, and its ratio to Q4_K_M. No 196-linear pack exists, and the v3 screen
  is at ctx 512, below Q2's 2048.
- The PPL pair is the other lane's run; this note cites the logs and does not re-run them. The token
  count moves both sides (v3 is 74.30 at 2044 tokens and 54.81 at 8176), so the ratio is a screen at
  this length, not a Q2 result.
- The probe's preconditioner is the tensor's column RMS, not the activation Gram diagonal the
  reference pipeline uses; `admm_loss` and `relF` are under that weighting, not the `relH` metric of
  the 1.18 campaign. The T=400 column is ×40 of a T=10 wall, an upper bound that repeats the setup.
- The rank distribution inside the ±0.002 band. Four tables fit the rate; none is measured for
  quality.
- The residual comparison reuses the 8B pack's PPL at 2.13 and 3.13 b/w, not a 0.6B residual run.
- One dataset, one tokenizer, one context length; nothing here is a proof.
