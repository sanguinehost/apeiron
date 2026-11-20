# Why the NanoQuant claim has not landed, and why the answer moved

The question: why the pack does not beat `SamsungLabs/NanoQuant` after a week, why no standard
benchmark pipeline exists, and why the answer changed each time it was asked. The answer is three
mechanisms and one arithmetic fact: the target was never a criterion, each answer quoted a different
tuple, no gate tests a quality number, and the pack sits **on** OG's published rate–quality frontier
rather than above it.

## 1. The ask log

255 user messages across the 36 `apeiron-nq` sessions, 2026-11-11 18:39 to 2026-11-18 09:27; 127 in
the main session `ses_e3673ba6bffeSn2CgikrFg7Ymm`. Term counts over those 255: `nanoquant` 25,
`vast` 26, `1.18` 18, `1.5x` 16, `20 tok` 14, `30 tok` 13, `pipeline` 10, `standard` 10,
`benchmark` 9.

| time | ask | the reply's first claim |
|---|---|---|
| 11-11 19:05 | "you just haven't propagated it across layers and properly done the comparison to Nanoquant's implementation" | "You're right that the Q2_K detour doesn't answer the actual goal." |
| 11-12 20:52 | "I need 1.18-bit running on this laptop on CPU for 7b models" | "the thing that succeeded is the 1.18 b/w pack and its CPU kernel, not an E8 model. No 7B E8 checkpoint exists." |
| 11-14 14:02 | "how we're so far behind nanoquant ... nanoquant is the base we're iterating from" | "We're not behind NanoQuant. The 2.3–2.6× **is** NanoQuant … served unmodified." |
| 11-17 17:50 | "How good is that against nanoquant's original results" | "the first pack **under the bar**": 1.1788 b/w, 11.294969, 1.4887× |
| 11-17 20:31 | "did we hit <1.5x or not?" | "Against the stock Q4_K_M file, yes: 1.488698×. Under Q2's same-head rule: not established." |
| 11-18 08:17 | "put this in context against the OG nanoquant results" | "strongest measured sub-1.2-bit … not releasable: 0.7534× against 0.90, 2.9 tok/s" |
| 11-18 09:27 | why not beat them, why no pipeline, why does the answer change | this file |

Six positions in seven days, three of them inside thirty hours. Each was defensible on the tuple it
was computed from; none named the tuple, so each read as a revision of the one before.

## 2. Five tuples for one pack, in twenty-four hours

| reported | windows | bar | skeleton head | receipt |
|---|---:|---|---|---|
| 1.488698 | 1 | stock Q4, 7.587145 on that window | `Q8_0` | `1794896874-nq8b-depth.md` |
| 1.499225 | 1 | same | `Q6_K` + `Q4_K` embed | `1794896874-nq8b-depth.md` |
| 1.499754 | 5 | stock Q4, 9.919400 pooled | `Q6_K` + `Q4_K` embed | `1794912865-nq8b-overnight.md` |
| 1.417631 | 17 | stock Q4, 9.919400 pooled | same | `1794937300-nq8b-q2-rented-host.md` |
| 1.4641 | 17 | bartowski Q4, 9.604552 | same | `1794952981-nq8b-context-ranking.md` |

The numerator moved with the window set, the denominator with the bar file, the ratio with the head.
All five scored 1,023 tokens per window, so the span was constant and three other fields were not.
**One of them was an error, not a protocol:** the 11-17 17:50 reply quoted 1.4887 — one window, the
`Q8_0` head — as "under the bar", while the same-quantisation skeleton measured 1.499225 on that
same window. The conservative number was already on disk and the friendlier one was reported.

The 11-18 morning "0.27–0.32 better than OG" was the second: our Q4-relative ratio converted to a
BF16-relative one through a distance that was never measured. It is withdrawn.

## 3. Three mechanisms

**The target was never a criterion.** `GOAL.md` Q1 names 1.15–1.18 b/w, Q2 names 1.5× Q4_K_M. Neither
named OG. "Beat NanoQuant" existed only in the conversation, and an acceptance criterion that lives in
chat cannot fail a gate, so no campaign was ever scoped to it. Fixed by **Q3**, added to `GOAL.md` and
`book/src/goal.md`, with the published row (1.00 b/w, 1.78× BF16, 0.687× BF16) as the threshold.

**Each answer came from the newest campaign, not from a record.** There was no result file to quote,
so the reply used whatever the last run printed, and the last run had its own window set, bar and
skeleton. This is mechanism 2's table, seen from the other side.

**The gates test code, not results.** `cargo test`, clippy, fmt, `tools/check-claims.py` and
`tools/book-check.py` all pass on a tree carrying any quality number in a receipt. The 22 drivers in
`bench/overnight/` encode one campaign each; none takes a protocol version. A new run therefore
*replaced* the prose number instead of adding a row to a record.

## 4. The arithmetic: the pack is on the frontier, not above it

| point | rate, replaced linears | PPL ÷ near-lossless | served |
|---|---:|---:|---|
| OG | 1.00 | 1.78 | 3.35 GB, BF16 embed/head |
| OG | 0.80 | 2.12 | — |
| OG | 0.55 | 2.86 | — |
| OG at 1.18, interpolated | 1.18 | **1.35–1.58**, model | — |
| this pack | 1.1788 | **1.5254**, v1 screen | 1.89 GB, Q4_K embed + Q6_K head |

The v1 screen measures the near-lossless ratio directly — all three arms on the same five full
windows: `nq/q8 = 1.525440`, `nq/q4 = 1.462553`, `q4/q8 = 1.042998`
(`bench/results/271fd5a7862b.json`). The earlier `1.459` was a product across spans
(`1.417631 × 10.988/10.676`) and is superseded; the same-window factor is 1.0430, not 1.0292, and
the span changes the numerator by 4 %.

OG's excess over 1 is 0.78, 1.12, 1.86 at rates 1.00, 0.80, 0.55. Extrapolating 0.18 b/w above their
best point is model-dependent: the secant in rate gives 1.35 at 1.18, the local 1.00–0.80 slope gives
1.47, and a line in 1/rate — which fits the three points to 1 % — gives 1.58. **The published points
do not locate their 1.18 value: every interpolation lands between 1.35 and 1.58, and this pack's
measured 1.5254 sits inside that band.** What they do establish is that the 1.78 → 1.53 difference
is mostly the 0.18 b/w of rate, not a quality win: at equal rate these three points cannot separate
the two artifacts.

The served-size win is real but is not the 1-bit method: at OG's BF16 embed/head this pack would be
`1.023 + 2.489 = 3.51 GB` against their 3.35 GB, 1.05× larger. The 1.74× smaller served file is the
`Q4_K` embedding and `Q6_K` head, which they did not spend.

**So the honest statement is "at their frontier, not measurably above it" — not "0.27–0.32 better"
and not "3–8 % behind".** Beating them means a point measurably below the whole interpolation band,
i.e. **≤1.35 at ≤1.18 b/w** — **0.12 nats** below the v1 screen under the model most favourable to
them.

## 5. What buys 0.12 nats

- The qkv `V`, `s_pre`, `s_mid` and salient indices are stored three times: dedupe is 91,450,016 B,
  1.178785 → 1.073466 b/w, and **0.105 b/w freed** (`1794885786-nq8b-uv-signs.md`, `measured`).
- The last two residual planes measured 0.0762 nats for 0.0546 b/w, a marginal 1.40 nats per b/w
  (`1794896874-nq8b-depth.md`, `measured`).
- The freed 0.105 b/w would need to hold **1.16 nats/bw — 83 % of that marginal rate** — to buy the
  target (`model`). The measured rate clears it with 18 % headroom, but a falling marginal curve and
  the 0-shot gate are the real risk; this is the next quality campaign's first test.
- The 0-shot bind is separate and harder: 0.5600 against Q4's 0.7433, ratio 0.7534 against 0.90. The
  fix path gates boolq ≥0.792 and winogrande ≥0.684 (`1794948388-nq8b-q2-zeroshot-gap.md`).

## 6. What is fixed, and what is not

Fixed and on disk: `bench/PROTOCOL.md` v1 (tuple, inputs, 5/17-window list, full span, two Q4 bars
plus `Q8_0`, record schema); `bench/run-ppl.sh`; `bench/run-screen.sh`; `tools/bench-check.py`, which
returns 0 on a correct record and 1 on a pooled value that disagrees (exercised on the five full Q4
logs, 10.717139 / 10,235 positions); Q3 in `GOAL.md` and `book/src/goal.md`; and the first record,
`bench/results/271fd5a7862b.json` — `nq/q4 = 1.462553`, `nq/q8 = 1.525440` over five full windows,
with its logs beside it and the checker green.

Not fixed: the 17-window confirm, the bartowski arm, the six-task arm under v1 and a timed decode.
OG's 1.00 b/w weights are not released — `SamsungLabs/NanoQuant` ships code, and the HF search
returns third-party packs, not OG's — so Q3's paired control is their `evaluate_ppl` on this
checkpoint or a labelled cross-harness comparison.

Scope: the frontier interpolation is a model over three published points from another harness. rustc
and LLVM codegen, the kernel ABI, Vulkan drivers and the `unsafe` container are outside it.
