# A dev box: OVH b3-32 in Sydney for the credit month

Criterion: none directly; this unblocks the **maintainer gate** (Lean/Aeneas) and takes the
contributor gate off the 6.6 GiB laptop. Grades: `measured` for the credit, the instance, the
catalogue price and the gate outputs quoted below, from commands named in each row; the vendor
prices are the catalogue's AUD figures read through the API, not measurements of this repository.

## What the box is for

| job | needs | on the laptop |
|---|---|---|
| `cargo test`/`clippy`/`fmt` for one worktree | 4 cores, 2–3 GiB | admitted one job at a time; the lock is held by other lanes for minutes at a time |
| three agents' worktrees at once | 4 cores, 6 GiB | refused: the guard's cap is `MemAvailable − 1 GiB` |
| `proof/bootstrap.sh` (Aeneas + Charon) | 4+ cores, ~10 GiB disk | never run; the toolchain is absent |
| Aeneas `-backend lean` on the 250 MB `.llbc` | **13.7 GiB peak RSS**, 510 s | impossible (6.7 GiB total) |

`proof/README.md:548` is the binding number: the extraction generator peaks at 13.7 GiB. No 8 GiB
VPS can hold it, so the laptop cannot run the maintainer gate and neither can the A$13.09 VPS.

## Provisioned

| field | value | source |
|---|---|---|
| project | `ae64542b964b4a0d9dd26cab89ffc061`, `Project 2026-11-24`, plan `project.2018` | `GET /cloud/project` |
| instance | `0a047be9-131f-46fe-9dd6-2ec78c22584f`, `indra-dev-1` | `POST /cloud/project/{id}/instance` |
| flavour | `b3-32`, 8 vCPU, 32 GiB, 200 GB, `SYD1` | `GET /cloud/project/{id}/flavor?region=SYD1` |
| image | `Ubuntu 24.04`, `f6b3ae83-1b29-4b95-b0c1-c820326cd36c` | `GET /cloud/project/{id}/image?region=SYD1` |
| address | `139.99.223.161` (IPv4), `2402:1f00:8102:100::82f` (IPv6) | `GET /cloud/project/{id}/instance/{id}` |
| billing | hourly, `monthlyBilling: false` | create body |
| rate | **A$0.3622/h ex GST** = A$0.31058 compute + A$0.05162 local disk | catalogue `b3-32.consumption`, `price: 36220000` in 1e-8 AUD |
| 720 h | A$260.78 ex GST, **A$286.86 incl 10% GST** | arithmetic on the rate |
| credit | **A$300 Free Trial**, available A$300, used A$0, expires **2026-12-24** | `GET /cloud/project/{id}/credit/284478` |

The credit is valid for 30 days and the instance costs A$286.86 of it over 720 h, so the box costs
no cash in its lifetime. The choice is not the cheapest box: `b3-8` (2 vCPU/8 GiB) is A$0.0907/h,
A$66.21 for the same 30 days, and would leave A$234 of a credit that expires on 2026-12-24. The
credit selects the largest box that fits its lifetime, and 32 GiB is the size that holds the
extraction.

A `b3-8` is 2 vCPU, not 4: the survey's core count for that row is wrong; the v1 `flavor` payload
reports `vcpus: 2` and `ram: 8` (GiB, not MB — the field is not scaled).

## The box is not the target device

| | dev box | laptop (target) |
|---|---|---|
| CPU | AMD EPYC Milan, 8 vCPU, 1 thread/core | i5-10210U, 2 vCPU, 15 W |
| L3 | 256 MiB (8 instances) | 6 MiB |
| clock control | no `cpufreq` | governor `powersave`; `performance` needs sudo |
| memory | 31.3 GiB total | 6.7 GiB total |

A kernel ratio on the box is `measured` for the box and `model` for the laptop: the cache per core
differs by more than an order of magnitude, so a change that wins by fitting a working set in L3
can lose on the laptop. T1/T2 claims stay on the laptop; the box runs gates, builds, extraction,
Lean, and wide A/B screens whose ratios are then confirmed on the laptop.

## Guard policy on the box

The laptop's policy caps a job at 2048 MiB and 2 jobs. Applied to a 32 GiB box that cap OOM-kills
Aeneas at 13.7 GiB and idles six of eight cores. `~/Workspace/tools/agent-guard` now reads four
overrides whose defaults are the laptop's values, so `agent-guard cap 3072` still prints
`cap=2048MiB` there:

| variable | laptop default | box |
|---|---|---|
| `AGENT_GUARD_MAX_CAP_MIB` | 2048 | 16384 |
| `AGENT_GUARD_MIN_AVAILABLE_MIB` | 1024 | 4096 |
| `AGENT_GUARD_MAX_JOBS` | 2 | 8 |
| `AGENT_GUARD_CPU_QUOTA` | 400% | 800% |

The invariants do not move: one heavy job at a time, a cap derived from `MemAvailable`, swap zero,
and a refusal when the machine is loaded. The job count is a ceiling, not a default — the guard
clamps a count down and never raises the 2-job default, so the box's wrapper also exports
`CARGO_BUILD_JOBS=8` and the matching `NUM_JOBS`/`RAYON_NUM_THREADS`/`OMP_NUM_THREADS`.

## Bootstrap, as run

1. `apt`: `build-essential pkg-config libssl-dev git curl rsync python3 htop tmux unzip jq zstd`.
2. `rustup` with `nightly-2026-07-21` (the laptop's channel; the box resolves `1.99.0-nightly
   (87e5904f5 2026-07-20)`, the laptop `1.99.0-nightly (0e29c21d9 2026-07-21)`), plus `clippy` and
   `rustfmt`.
3. `rsync -az --exclude target/ --exclude .git`: 197,302,689 bytes of tree; the fixture GGUF
   (482,672,896 B), `nq06-v3.nqp` (53,815,472 B) and `tokenizer.json` (7,031,645 B).
4. `~/bin/adg`: the box wrapper above. `~/bin/box-gate.sh`: the contributor gate.
5. `cargo install mdbook --locked` for `mdbook build`/`mdbook test`.
6. `apt`: `glslc mesa-vulkan-drivers vulkan-tools` (shaderc 2023.8; `llvmpipe`, LLVM 20.1.2).
7. `bun` 1.4.5 at `~/.bun/bin`. It runs `bun install --frozen-lockfile` and `bun test` for
   `lrw.com` (36 pass, 0 fail); `sanguine.sh` uses the same toolchain.
8. `~/Workspace/tools/agent-guard` is a policy shim here; the real script is
   `agent-guard.real`. The shim supplies the box's four variables and prepends
   `~/.cargo/bin:~/.bun/bin` to `PATH`, because a non-login `ssh` shell has neither. Every
   caller therefore gets the box policy, including the `package.json` scripts that name the
   guard by absolute path.

## Gate on the box, as run

| step | result |
|---|---|
| `cargo test --workspace --all-features --locked` | 680 passed, 45 failed, 62 ignored — all GPU |
| `cargo test --workspace --locked` (`gpu` off) | 668 passed, 1 failed, 51 ignored |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 |
| `cargo fmt --all -- --check` | 0 |
| `tools/check-claims.py`, `tools/book-check.py`, `mdbook build book` | 0 |
| `mdbook test book -L target/debug/deps` | `E0464`; passes against a clean `CARGO_TARGET_DIR` |

The 45 GPU failures are 14 SPIR-V byte comparisons and 31 device tests. The box has no GPU, and
shaderc 2023.8 does not reproduce the committed SPIR-V, so those 45 are hardware. The one failure
in the `gpu`-off suite is the `coopmat` SPIR-V check, which runs without the feature and needs the
same compiler. The laptop has no `glslc`, so this tier is not green on either host; it needs a host
with the pinned compiler, and the device tests need a Vulkan device. Everything else in the
contributor gate is green on the box.

`target/debug/deps` after a test run holds several `libapeiron` rlibs, so `mdbook test` needs a
directory that was built once:

```sh
CARGO_TARGET_DIR=$HOME/apeiron-mdbook cargo build --lib --all-features --locked
mdbook test book -L $HOME/apeiron-mdbook/debug/deps
```

## Iterating from the laptop

Two commands, both safe to run from here; the full-tree rsync above is only the first sync:

```sh
~/bin/box-push apeiron-nq                 # only `git status` files, excludes target/ and owner state
~/bin/box-run apeiron-nq <command...>     # one command under the box guard
```

A three-file edit is ~20 KB; the current uncommitted tree was 524 KB against 300 MB for the
sources. `box-run` passes its arguments through a non-login `ssh`, so the box shim supplies `PATH`
and the guard policy. A release build of `prefix_solve` there is 13 s, and a four-run screen at
1,827 + 16 is about 90 s: the two-pair block screen ran 10:13:57–10:15:30 for four runs, where
each laptop run of the same shape is 45–65 s before the guard admits it.

## Latency, measured from this laptop (Melbourne)

| endpoint | RTT | what it is |
|---|---:|---|
| `www.abc.net.au` | 18.8 ms avg (16.8–20.3) | Akamai edge, Sydney |
| `www.ovhcloud.com/en-au/…` | 25.8 ms TCP connect | OVH AU edge |
| `mirror.aarnet.edu.au` | 43.6 ms avg (37.5–52.0) | Canberra |
| `www.netcup.com` | 187 ms TCP connect | Nuremberg |
| `nbg1-speed.hetzner.com` | 472 ms TCP connect | Nuremberg |

SSH keystroke echo is what this buys: 19 ms is invisible, 190–470 ms is not. It also keeps the
model files a Sydney hop away instead of an intercontinental one; the 543 MB upload from Melbourne
ran at 0.6–2.7 MB/s. The link is not stable: a 5 MB `scp` on 2026-11-24 ran at 80 KB/s with a
185 ms average RTT to the box and 291 ms to `1.1.1.1`, so the first sync of a tree with fixtures
takes an hour and a per-edit sync is only the changed files. Copy an artifact, not a tree.

## After 2026-12-24

The credit expires; the instance is destroyed by then or starts billing cash at A$0.3622/h. The
persistent option remains the VPS-2 at A$13.09 incl GST, which cannot run Aeneas, so the Lean
campaign then needs an hourly b3-16/b3-32 burst from cash at A$0.18–0.36/h, ten hours a month
being A$1.81–3.62. `../indra`'s OVH adapter can create, poll and destroy that burst; ordering the
project itself is an `/order/cart` call the current key does not hold.

## The shortlist, A$20/month ceiling (persistent box)

AUD per EUR = 1.6532 (ECB reference, 2026-11-24).

| offer | specs | price | AUD/mo | verdict |
|---|---|---|---|---:|
| **OVH VPS-2, Sydney** | 4 vCore, 8 GB, 75 GB NVMe, 1 IPv4, daily backup, 1 Gbps | A$11.90 ex GST | **13.09** | **chosen after the credit** |
| OVH VPS-3, Sydney | 6 vCore, 12 GB, 100 GB | A$17.08 ex GST | 18.79 | alternative if 12 GB is wanted; still short of 13.7 |
| OVH VPS-1, Sydney | 2 vCore, 4 GB, 40 GB | A$6.29 ex GST | 6.92 | too little RAM to be worth the setup |
| Hetzner CX33, Nuremberg/Singapore | 4 vCPU, 8 GB, 80 GB | €8.49 + €0.50 IPv4 = €8.99 | 14.86 | same size, 190–470 ms away unless Singapore |
| Netcup VPS 1000 G12.5 | 4 vCore, 8 GB, 128 GB | €14.50 incl. 19% VAT (12-month term) | 23.97 | over budget |
| BinaryLane, Sydney | 2 vCPU, 4 GB, 60 GB | A$19.60 | 19.60 | at budget, half the RAM |
| BinaryLane, Sydney | 4 vCPU, 8 GB, 100 GB | A$39.20 | 39.20 | 2× over |
| GitHub Codespaces free | 2 core, 8 GB | 120 core-h/mo = 60 h | 0 | 60 h/month, and `github/main` is 61 commits behind and 3 ahead of this tree |
| Oracle Always Free A1 | 4 OCPU, 24 GB, ARM | A$0 | 0 | fits Aeneas; ARM (no AVX2 gate), capacity lottery |

## What this does not cover

- No T1/T2 claim may name the box: it is not the target device.
- OVH shared vCPU throughput is not the laptop's. The build/test wall times are `measured` for the
  box only.
- The free-trial credit's product list is empty (`products: null`), so coverage of every line is
  inferred from the usage endpoint, not from a settled invoice; if OVH excludes a line, the first
  invoice will say so.
- The instance is hourly and uncommitted; the A$286.86 figure assumes 720 h, the whole life of the
  credit.
- Prices are catalogue figures read 2026-11-24 and move.
