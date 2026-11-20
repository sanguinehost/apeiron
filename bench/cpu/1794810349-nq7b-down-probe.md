# Qwen2.5-7B `down`: the limb refusal is `Range`

Criterion: **Q1** (a full 7B-class pack at 1.174 b/w). Grade: `measured` for the two refusals and the
wall times; `model` for the call-site attribution in §4. The pack does not exist, so Q1 is not closed.
Machine: vast.ai instance `60889280` (offer `25469153`, machine `31507`, host `81456`, RTX 2060S
unused), Xeon E5-2680 v4, 14 cores / 28 threads at 2.4 GHz, 31 GB RAM, 64 GB disk, Ubuntu 24.04,
rustc 1.100.0, `$0.08444/h` all-in. Destroyed at the end of the run; `vastai show instances` is
empty.

## 1. Source

The run used HEAD `735a7b06` with the working tree's 55 dirty paths, the state that carries
`NqError::Kernel(LimbRefusal)` (`6728b0a2`) and the two tolerance changes. The probe did not read
`.git`; the remote copy is identified by the hashes of the files the path uses:

| file | sha256 (local = remote) |
|---|---|
| `src/learn/nanoquant.rs` | `563c25c88e2094aa231a12fe6e1f83748984b76a4e9a9624c9b99526488501e0` |
| `src/engine/kern/limb.rs` | `7544a0d723eed635d2ac50e7a586a2fcb7f92f300c618bc8eadfd5858d290f35` |
| `tests/learn_nanoquant.rs` | `bd27155fb6ffbcc1c68e8c2d30475f9cbd7964054b7c214bf7578e6c0eef8874` |
| `Cargo.toml` | `d90252afe437143b4decd0eb6113ffe7f4bd5eef4a7fec46acbeb3dcd0432b89` |
| `.cargo/config.toml` | `7af9ec6837490d1ca073d1db582eab96aff336a76a7fff3236d3dbad3bdd3756` |

`git diff | sha256sum` is `9c3f1f42710c699ddff9c30ab96a14a908fa0f3146a84005c4d15de5cf685fce`.
Transfer, excluding build products and history:

```sh
tar -C /home/lucas/Workspace/apeiron-nq --exclude=target --exclude='*/target' \
  --exclude=.git --exclude='*/.lake' --exclude=./book/book -cf - . \
  | ssh -p 34160 root@ssh8.vast.ai \
      'mkdir -p /workspace/apeiron && tar -C /workspace/apeiron -xf -'
```

## 2. Materials

Provisioning, for the record:

```sh
vastai create instance 25469153 --image ubuntu:24.04 --disk 64 --ssh \
  --label nq7b-down-probe --cancel-unavail
# instance 60889280, ssh8.vast.ai:34160
...
vastai destroy instance 60889280 -y
```

`setup.sh` ran once as `bash /workspace/setup.sh` (nohup, log at `/workspace/logs/setup.log`). The two
GGUF files match the pinned sizes and hashes:

| file | bytes | sha256 |
|---|---:|---|
| `Qwen2.5-7B-Instruct-f16.gguf` | 15,237,853,600 | `863c978275bca3fdacdde06cbfc1a65f2bd65210bb7ecfd9ded3b24219d81b54` |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4,683,074,240 | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` |

`setup.sh` exits 1 at its final `sha256sum tokenizer.json …` because it runs in `/workspace/models`
while the tokenizer was written to `/workspace/data`; both GGUF hashes verify before that line and
the probe does not read the tokenizer. The script now names `/workspace/data/tokenizer.json` (the
one-line path fix, applied after the run).

The 112 calibration diagonals were uploaded to `/workspace/models/q7b-cache/` (6.8 MB,
`diag-c6x29/`). `probe.sh`'s `APEIRON_NQ_KIND` line was changed on the box from the hardcoded list to
`export APEIRON_NQ_KIND=${APEIRON_NQ_KIND:-down,gate,up,o,q,k,v}` so the per-run kind is honored;
`down` is first in the list either way. The probe keeps `APEIRON_NQ_BITS_PPM=1150000`,
`APEIRON_NQ_LAYERS=0`, `APEIRON_NQ_TAG=-probe`, `APEIRON_NQ_GRAM_TAG=c6x29`, `APEIRON_NQ_RELH=0`,
`APEIRON_NQ_DENSE=0`.

## 3. Probe

```sh
APEIRON_NQ_KIND=down ITERS=0 CARGO_BUILD_JOBS=8 APEIRON_THREADS=20 \
  timeout 2400 /workspace/probe.sh
APEIRON_NQ_KIND=down ITERS=1 CARGO_BUILD_JOBS=8 APEIRON_THREADS=20 \
  timeout 2400 /workspace/probe.sh
```

The release test built once in 21.77 s; `CARGO_BUILD_JOBS=8` overrides the tree's
`.cargo/config.toml` `jobs = 2`. Both runs use one layer (`APEIRON_NQ_LAYERS=0`) and the `down` kind.
`down` is 3584×18944. `probe.sh` pins `APEIRON_NQ_BITS_PPM=1150000`, so
`rank_for_bits(3584, 18944, 1_150_000, 32) = 3,424`; the 1.174 b/w campaign at 1,180,000 uses 3,520.
`factorize` takes the `out < inn` transpose, so the fitting frame is 18,944×3,584.

| T | wall | exact result |
|---:|---:|---|
| 0 | 111.86 s | `factorize: Kernel(Range)` |
| 1 | 481.64 s | `factorize: Kernel(Range)` |

The exact lines:

```text
thread 'nq_fit' (2505) panicked at tests/learn_nanoquant.rs:790:81:
factorize: Kernel(Range)
...
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 10 filtered out; finished in 111.86s
```

```text
thread 'nq_fit' (2590) panicked at tests/learn_nanoquant.rs:790:81:
factorize: Kernel(Range)
...
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 10 filtered out; finished in 481.64s
```

`790:81` is the test's `.expect("factorize")` on `factorize(...)`; `.expect` prints the `Debug` of
`NqError`, so the variant is `LimbRefusal::Range`. It is not `Shape` and not `NoKernel`.

The T=0 refusal has moved: before the `MAX_INNER` raise it was `Plane::build`'s shape refusal at
`n = 18,944 > 16,384`; at `MAX_INNER = 32,768` the plane builds and the refusal is now a range
refusal. The prior box's T=1 270.0 s against this box's 481.64 s is the machine ratio (1.78×, 2.4 GHz
E5 against the prior i5-14600KF) on the same iteration path. The T=0 63.8 s and 111.86 s are not a
machine comparison: the prior run stopped at the plane build, this one runs past it to the range
refusal.

## 4. What `Range` means, and where it can come from

`LimbRefusal::Range` is either a block pair whose product exponent is past `MAX_PAIR_EXP = 91`, or a
partial sum or output word that leaves `i64` (`src/engine/kern/limb.rs`). `NoKernel` is never
returned, and `Shape` would be a dimension or a width past `MAX_INNER`. The variant names the class,
not the call.

The T=0 path runs no ADMM iteration. Its limb calls are `prepare` (shape only), the `rowdot_q32` and
`gemm_q32` in `admm_loss` (both now turn `Range` into `f64::INFINITY`), and the four `fro2` norms:
`admm_loss`'s `‖e‖²` and `‖W_n‖²`, and `export`'s `‖A‖²` and `‖B‖²`. Only `fro2` still propagates
`Range` to `factorize`.

The measured `down` diagonal makes `export`'s `‖A‖²` the likely T=0 site:

| quantity | value |
|---|---:|
| `g0.c6x29.down.diag.i64` header | `(1, 18944)` |
| smallest `diag` | 2.828892320394516e-7 |
| smallest `sqrt(diag)` (`no`) | 5.318733233011894e-4 |
| mean `sqrt(diag)` | 3.774297227384278e-2 |
| `scales` floor, `1e-6 · mean` | 3.774297227384278e-8 |

No coordinate reaches the floor, so the smallest row scale is `5.3187e-4`. `export` forms
`a = a_z / no` and then `fro2(&a, 18944, 3424)`. At T=0, `a_z` is the seeded standard-normal draw
(`a_ls`), so one row's expected `Σ a²` is `3424 · (1/no)² = 1.210e10`, whose Q32.32 word is
`1.210e10 · 2^32 = 5.20e19`, 5.6× `i64::MAX = 9.22e18` (5.8× at the campaign's rank 3,520).
`rowdot_q32` therefore refuses that row with `Range`. The other three norms stay near the
standard-normal scale: `‖W_n‖²` is shrunk by the same `no`, while `‖e‖²` and `‖B‖²` sum about
3,500 unit squares per row, `1.5e13` words, six orders below `i64::MAX`. This is `model`: the
diagonal, the floor and the code path are read; the call site itself is not logged, so the
attribution is arithmetic, not a measured backtrace.

The T=1 path is not located by this probe. Its one outer iteration runs side B's `x_update` on the
transposed `down` frame, where `syrk_q32` and `gemm` contract over `n = 18,944` — the width the
`MAX_INNER` raise admitted — as well as the same `export` norms. The `Range` can be either. No log
line distinguishes them.

## 5. Cost and credit

| item | quantity | rate | amount |
|---|---:|---:|---:|
| credit before | | | `$0.4515307643500428` |
| credit after destroy | | | `$0.3803822946500368` |
| delta | | | `$0.071148469700006` |
| instance `60889280`, GPU | 0.526 h | `$0.0667/h` | `$0.035` |
| instance `60889280`, storage | 0.537 h | `$0.0178/h` | `$0.010` |
| instance `60889280`, download | 20.300 GB | `$0.001/GB` | `$0.026` |

The invoice's three lines sum to `$0.071`; the credit delta is the unrounded `$0.07115`. The GPU
quantity is 0.526 h = 31.6 min, from instance creation through teardown. The 20.3 GB download is the
teacher plus the `Q4_K_M` bar; it is the largest single line because this run is half an hour long.

## 6. What this establishes

- The `down` refusal on the current tree is `LimbRefusal::Range`, at both `ITERS=0` and `ITERS=1`.
- The pre-raise T=0 `Shape` refusal is gone; `MAX_INNER = 32,768` admits the 18,944-wide plane, and
  the T=0 refusal now comes later.
- The T=0 `Range` has a concrete arithmetic account in `export`'s `‖A‖²`: the measured down diagonal's
  smallest row scale against rank 3,424 makes the Q32.32 word 5.6× `i64::MAX`.
- The exact call site is not established, and the T=1 refusal is not located.

## 7. What this does not establish

- No `.nqf` was written and no `model.nqp` exists; Q1 stays partial.
- No wikitext-2 PPL, no 0-shot suite, no quality-bearing `T`; Q2 is untouched.
- No memory sample, no 200k context, no throughput; M1, T1 and T2 are untouched.
- No backtrace, no call-site tag, so `Range` is a class, not a call. The next diagnostic is a
  call-site label on the four `fro2` sites (and the iteration's wide products), or a
  `RUST_BACKTRACE=1` rerun of T=0 if the release symbols survive inlining.
- The two tolerance changes execute on this path, but the test prints `admm_loss` only after
  `factorize` returns, so these logs do not show whether the loss became `INFINITY`; they are not
  evidence for or against the tolerances.
- rustc, LLVM, the kernel ABI and the device remain outside every claim here.
