# nq: exact prompt-lookup speculative decode — 16.07 → 6.45 ms/token

Criterion: **T1** (decode) and **T2** (turn latency). The decode step is 2.49× faster on the same
stream, and the ids are the greedy ids. The mechanism is a drafter, not a cheaper kernel: a
prompt-lookup n-gram proposes up to eight tokens, one trunk pass verifies all of them at `k = 9`,
and a full-reject round keeps only column 0. The dominant term moves from the projection pass to the
same pass widened; the ceiling is the draft's column cost, not a phase's share.

Grades: `measured` for the walls, the paired ratios and the phase split; `model` for the `1/(1 − s)`
ceilings, the turn arithmetic and every rate quoted against llama.cpp. `tested` covers the shared
`compose_cols` path the round calls; the drafter itself is checked by the example's exactness
assertion on the measured stream, not by a unit test.

## Machine, binaries and command

Machine: OVH box `ubuntu@139.99.223.161` (8 single-core sockets, EPYC Milan, 8 vCPU, 32 GiB),
`APEIRON_THREADS=8`, `~/bin/adg`. CPU-tier evidence; the timed target is the laptop (i5-10210U),
which did not run this measurement.

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 0.977441 b/w, 53,808,384 B).
Binary `lookup_accept` sha256 `887cbce6c1509596b9fbeda6ff9fc32e29ef2be41bc99c60dcb05ca5b136574c`.

```sh
APEIRON_THREADS=8 target/release/examples/lookup_accept \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --first 42 --n 128 --ngram 2 --draft 8 [--trace]
```

The two arms are the same process and the same stream: the example first generates 128 greedy ids
(`Engine::next_token_state`, the `k = 1` path) and times them, then runs
`Engine::generate_lookup` from id 42 and compares every emitted id with that stream.

| arm | run 1 | run 2 | run 3 | mean | ratio |
|---|---|---|---|---|---|
| greedy `k = 1` | 16.045 | 16.014 | 16.160 | **16.073 ms/token** | 1.00× |
| prompt-lookup | 6.460 | 6.431 | 6.444 | **6.445 ms/token** | **2.49×** |

llama.cpp on the same host and model class is 10.92 ms/token (`1795727609-nq-05-baseline.md`), so
the lookup decoder is **0.59× llama**. The 0.5× target is 5.46 ms/token: **1.18× short**.

Turn, arithmetic: 3.903 s prefill (`1795732570-nq-prefill-split.md`) + 128 × 6.445 ms = 0.825 s is
**4.728 s** against llama's 5.363 s (**0.88×**); the 0.5× target is 2.681 s, so the turn is
**1.76× short** and prefill owns the gap: decode is already 0.59×.

## Exactness

The emitted ids equal the greedy stream id for id on all three runs (`exact` in the log). The
argument is structural: a draft is kept only where the verifier's own `Q8Act` head argmax at that
column equals it; a mismatch truncates the KV to `pos + 1` and restores every affine layer from the
snapshot taken before the pass, and only column 0 (the verifier's own token) is emitted. The
verifier cannot emit a token the greedy step would not, and it cannot drop one it would.

## Acceptance

The acceptance is measured on the benchmark `tg` stream and is a property of that stream, not of
text in general: from id 42 the model emits a six-token loop
`[220, 16, 15, 339, 315, 279]`, so every non-overlapping earlier occurrence continues identically.

| ngram | draft cap | rounds | ids/round | drafts | accepted |
|---|---|---|---|---|---|
| 2 | 8 | 29 | 4.448 | 100 | 100 (100.0%) |
| 3 | 8 | 30 | 4.300 | 99 | 99 (100.0%) |
| 4 | 8 | 31 | 4.161 | 98 | 98 (100.0%) |
| 2 | 16 | 29 | 4.448 | 100 | 100 (100.0%) |

The cap is set by causality: the drafter may only read the prefix ending at the current position, so
a match contributes at most `p + 1 − at` ids. A draft longer than 8 cannot be proposed from a
six-token loop before the loop has been read twice. The real decoder on the same run: 28 rounds,
102 drafts, 102 accepted.

## Phase split of the lookup decode

`--trace`, 128 tokens, one run (6.509 ms/token in that run against the 6.445 mean):

| phase | ms/token | share | calls |
|---|---|---|---|
| ffn | 2.201 | 33.8% | 812 |
| head | 1.806 | 27.7% | 29 |
| attn.proj_in | 0.953 | 14.6% | 812 |
| attn.core | 0.648 | 10.0% | 812 |
| attn.proj_out | 0.450 | 6.9% | 812 |
| norm | 0.175 | 2.7% | 1653 |
| residual | 0.089 | 1.4% | 1624 |
| embed | 0.004 | 0.1% | 29 |
| untraced | 0.050 | 0.8% | |
| pass (encloses the above) | 4.570 | 70.2% | 29 |
| pool.wait (overlay) | 0.382 | 5.9% | 11948 |

Dominant term `ffn` at 33.8%: ceiling `1/(1 − 0.338) = 1.51×`. `head` at 27.7% has ceiling 1.38×.
Either alone cannot close the 1.18× gap; both together can (a 1.51× `ffn` is 0.73 ms/token and a
1.38× `head` is 0.50 ms/token, together below 5.46 only if the rest is unchanged at 2.44). The
kernel work is therefore on the `ffn` group and the head, not on a decode-side scheduling change.

## What this does not cover

- The acceptance is the benchmark loop. On ordinary text a lookup drafter accepts less and the
  speedup is smaller; MTP drafting is unavailable on the 0.6B fixture (no `nextn` tensors), so no
  model-drafted number exists here.
- Partial acceptance: a run that matches only `j < m` drafts rejects all of them and emits one token.
  Keeping the matched prefix needs a per-column affine snapshot, which `compose_cols` does not take.
- The kernel's own `k = 9` cost was not split from the verifier's head; the pass column count is
  `drafts + 1` and the example's trace is the only split.
- The laptop did not run this; the timings are CPU-tier evidence.
