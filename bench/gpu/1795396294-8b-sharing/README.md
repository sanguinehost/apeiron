# The 8B block loop on one L4 with one parameter set

Measured on `nq-l4` (GCE spot `g2-standard-8`, 1×L4, heap 23,034 MiB, 31 GB host) on
2026-11-23, from `/workspace/dump-8b` (128 calibration samples, 2048 tokens each, Qwen3-8B-Base
statistics) and the tree at `6bb49bd2` + `272397a5`.

Each run is the block loop's first phase, `APEIRON_NQ_PHASES=1`, so it stops after the first
linear's non-fact training:

```sh
export APEIRON_NQ_MODEL=/workspace/q3-8b-base-bf16.gguf
export APEIRON_NQ_DUMP=/workspace/dump-8b
export APEIRON_NQ_LOOP=/workspace/loop-8b-s128
export APEIRON_NQ_TAG=probe
export APEIRON_NQ_BLOCKS=0..1
export APEIRON_NQ_SAMPLES=<samples>
export APEIRON_NQ_NCTX=2048
export APEIRON_NQ_BATCH=<batch>
export APEIRON_NQ_PHASES=1
export APEIRON_NQ_HOST_STREAMS=1
/usr/bin/time -v cargo test --release --locked --features gpu --test learn_block_tune -- \
  --ignored --exact block_loop --nocapture
# beside it: nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits every 0.5 s
```

| log | samples | batch | tape `T` | device peak | step time | max RSS |
|---|---:|---:|---:|---:|---:|---:|
| `fit-share128.log` + `gpu-used128.log` | 128 | 4 | 8192 | 17,153 MiB | 256 steps, 203.8 s | 22,796,092 KiB |
| `fit-share4.log` + `gpu-used4.log` | 4 | 4 | 8192 | 17,122 MiB | 8 steps, 10.4 s | — |
| `fit-share.log` + `gpu-used.log` | 4 | 1 | 2048 | 12,558 MiB | 32 steps, 10.9 s | — |

The 128-sample device peak is 17,153 MiB on a 23,034 MiB heap; the four calibration streams are
host-visible at 16.0 GiB (17.18 GB) under `APEIRON_NQ_HOST_STREAMS=1`, and `round total 217.8 s`
covers the first linear's non-fact phase only. The peak does not grow with the sample count
because the streams are not on the device; the tape does not grow because the fact-phase tuner
shares the non-fact tuner's factors, gradients, work buffers, Adam moments and dense moments
(`BlockTune::new_sharing`, `FrontTune::new_sharing`, `Factor::sharing`).

Not measured here: the fact phase, ADMM, the pack write, `eval_pass`, `preflight` and `kd_fit` at
128 samples. A full layer is those phases over seven linears; this probe does not license a
per-layer or per-model wall-time projection.

`clippy-sharing.log` and `clippy-lib-sharing.log` are the pinned-nightly clippy runs over the
sharing change (`--workspace --lib --tests --all-features --locked -j 4 -- -D warnings`).

## The gate ADMM after the tape and frame release

The preflight's `mlp.gate_proj` phase 1 was refused with `vkAllocateMemory failed: VkResult -2` at
22,540 MiB of the 23,034 MiB heap (`logs/trace-gate.log:238`, `logs/gpu-trace.log:1520,1524`), with
the full non-fact tape live and every earlier linear's frame cached.
`bench/notes/2026-11-23-memory-release.md` describes the release. The resumed probe
(`probe-gate.sh`, `APEIRON_NQ_PHASES=2`) ran gate ADMM in 505.34 s at recon norm 0.2209 and gate
fact in 176.4 s, `round total 702.5 s`, with an `nvidia-smi` peak of **18,452 MiB**
(`logs/probe-gate.log`, `logs/probe-gpu.log`). The tape release, the frame release and the narrow
reallocation at `T = 2048` are all on that path.

The four `--exact` equality tests show the released-and-reallocated and the reallocated-only
tapes' evaluation and forward outputs are bitwise equal to a fresh tuner's at the narrow width
(`f32::to_bits`, `0.00e0` relative; `logs/release-tape-tests.log`); training, stage changes and 8B
shapes are not covered. The preflight then finished the layer: `up_proj` ADMM
510.23 s and fact 188.7 s, `down_proj` ADMM 509.32 s and fact 214.0 s; the layer-0 `pack-l0.nqp` is
0.995619 b/w over its 192,937,984 replaced weights (the header names 36 layers; the other 35 stay
dense), 7-token PPL 17,039.4, rc 0, `nvidia-smi` peak 19,130 MiB in the last polled interval (the
advance re-records the 8192-column tape; `logs/preflight-8b.log`,
`logs/preflight-gpu.log:11639-11651`).

The reference R1 on the same L4, same `gate_proj` (`bench/gpu/campaign/ref-r1.py`): 47.37 s,
per-element error 1.4739e-4, 1,915 MiB peak (`logs/ref-r1-gate.log`). The port's 505.34 s at
1.4947e-4 is 10.7× slower at parity-quality. The baseline is same-device, same-shape, same-rank
and same-iteration, but not matched arithmetic: raw `bf16` against non-fact-tuned `f32`→Q32.32,
different RNG, and the port's figure is the whole `factorize_floor` against the reference's
factorisation call alone.

## The traced layer-0 round and the split

`trace-round.sh` on a fresh `loop-8b-trace` root, one layer-0 round over all seven linears, rc 0:
`round total 4587.4 s`, `nvidia-smi` peak **19,905 MiB** (`logs/trace-round.log`,
`logs/trace-gpu.log`). The run is one block (`APEIRON_NQ_BLOCKS=0..1`), `n = 1`, at
`DEV_INV_CUT=96`, without the pool trace; the peak is layer 0 only, not a 36-layer bound.
`inv` is the largest sub-phase in every linear, 20.4% of the layer wall per side; halving it
bounds the layer at 1.113× (q loop 1.392×). The split and the inverse-cut sweep
(cut 48 is 110.96 s on the q ADMM against 123.58 s at 96, the same recon) are in
`bench/notes/2026-11-23-admm-split.md`. The campaign sets `APEIRON_NQ_INV_CUT=48` as an engineering
choice on the q sweep; the `r = 3040` gate/up/down sweep at n ≥ 3 is open and no campaign gain is
claimed from the cut. The host-import probe reads 67 MiB with 16 GiB of imports touched and
4,163 MiB with the 4 GiB copy target (`logs/hostmem-probe*.log`).
