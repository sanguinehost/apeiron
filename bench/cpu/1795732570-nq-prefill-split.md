# nq: prefill phase split — `attn.core` 34.6%, the same dominant term as decode

Criterion: **T1** (decode) and **T2** (turn latency), measurement only. After the decode group
read (`cff8429d`), the turn is 3.90 s prefill + 3.07 s decode, so prefill is the larger half and is
at llama parity (468.1 against 460.8 tok/s). This receipt measures its phase split on the committed
binary and names the next target: **`attn.core` at 34.6%** (1.53× ceiling), the same dominant leaf
as decode (39.7%). A change to that leaf advances both halves of the turn.

Grade: `measured` for the split, the wall and the rates; `model` for the shares, the `1/(1 − s)`
ceilings and the turn arithmetic. Nothing is `proved` or `tested` here.

## Machine, binary and command

Machine: OVH box `ubuntu@139.99.223.161` (8 vCPU EPYC Milan, 32 GiB), `APEIRON_THREADS=8`,
`APEIRON_KV_CARRIER=f32`, `~/bin/adg`, load gate < 1.5. CPU-tier evidence; the timed target is the
laptop (i5-10210U), which did not run this measurement.

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 0.977441 b/w, 53,808,384 B);
tokenizer `tokenizer.json`.

| arm | binary | sha256 |
|---|---|---|
| committed `cff8429d` | `prefix_solve.group` | `da44f38b9ba51d43aa85f80d2ded3696627f3d1f03f37a1550745f72c62529c4` |

Command: `prefix_solve --model skeleton-q8head.gguf --pack nq06-v3.nqp --tokenizer tokenizer.json
--len 1827 --differ 0 --decode 1 --mb 512 --phases`. The trace covers the cold prefill and its one
decode step; the denominator is that run's measured wall, prefill + decode = **3.929 s**, so the
shares partition the traced turn rather than the prefill pass alone. Raw evidence in the `.log`
beside this file.

## The split

| term | ms | share | `1/(1 − s)` |
|---|---:|---:|---:|
| `attn.core` | 1360.103 | **34.6%** | **1.53×** |
| `ffn` | 1242.828 | 31.6% | 1.46× |
| `attn.proj_in` | 558.974 | 14.2% | 1.17× |
| `attn.proj_out` | 247.766 | 6.3% | 1.07× |
| `norm` | 150.202 | 3.8% | 1.04× |
| `residual` | 90.772 | 2.3% | 1.02× |
| `embed` | 12.437 | 0.3% | 1.00× |
| `head` | 5.611 | 0.1% | 1.00× |
| `pass` (encloses) | 3698.086 | 94.1% | — |
| `pool.wait` (overlay) | 190.931 | 4.9% | overlay |
| `outside.alloc` | 183.085 | 4.7% | — |
| `outside.insert` | 32.881 | 0.8% | — |
| untraced | ≈15.5 | 0.4% | — |

Cold run: **3.903 s** prefill over 1,827 tokens (468.10 tok/s), first token 315. `pool.wait` is an
overlay and is not summed into the leaves. `outside.alloc` is the 421,134,336 B KV plane's
allocation and first touch in the caller; it is wall the turn pays but the prefill pass does not.

## The turn and the campaign target

| half | ours | llama `049326a00` | ours / llama time | ceiling at F32 KV |
|---|---:|---:|---:|---:|
| prefill 1,827 | 3.903 s (468.1 tok/s) | 3.965 s (460.8 tok/s) | 0.984× | — |
| decode 128 | 3.069 s (41.7 tok/s) | 1.398 s (91.6 tok/s) | 2.196× | 157.8 tok/s |
| turn | **6.972 s** | **5.363 s** | **1.300×** | — |

The 0.5× turn target is 2.681 s, so the tree is **2.60× short**. Decomposed at the decode target of
5.46 ms/token (183.1 tok/s, 0.699 s for 128 tokens), the prefill budget is **1.982 s**
(922 tok/s): prefill must fall **1.97×** and decode **4.39×**.

A free `attn.core` takes prefill to 2.543 s (718 tok/s), still **1.28× above** the prefill budget;
the remaining leaves (2.54 s, `ffn` + projections + norm + residual) must halve as well. The
`attn.core` ceiling is 1.53×, not the 1.97× prefill needs — attention is the first target because
it is dominant in both phases, not because it closes either.

## Refusals

1. **"Attention is the whole prefill gap."** Refused: the 1.53× ceiling leaves 2.543 s against a
   1.982 s prefill budget; `ffn` (31.6%), `proj_in` (14.2%) and `proj_out` (6.3%) carry 52.1%
   combined and must also fall.
2. **"Prefill is behind llama."** Refused: 468.10 against 460.81 tok/s is 0.984× the time, inside
   the baseline's three-round spread.
3. **"The shares are shares of prefill."** Clarified: the harness denominator is prefill + the one
   decode step (3.929 s), per the trace comment at `examples/prefix_solve.rs:176`; the prefill pass
   is 94.1% of that wall.
4. **"This names the laptop."** Refused: box only; the i5-10210U has not run this fixture.
5. **"The maintainer gate ran."** Refused: the pinned Aeneas/Charon/Lean toolchain is not
   bootstrapped on either host and the committed extraction is stale. This receipt changes no Rust.

## What this does not cover

The laptop's wall, the 7B, 200k context, the GPU path, code generation for rustc/LLVM, and quality
(it is a split of the committed arithmetic, not a change). The turn arithmetic uses the decode mean
from `1795731349-nq-attn-group-read` on the same host and pack; the two runs were not interleaved.
