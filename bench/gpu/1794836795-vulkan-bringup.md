# Vulkan bring-up: the RTX 3060 device initializes, the first kernel returns zeros

| field | value |
|---|---|
| criterion | **V1** (Vulkan/SPIR-V/Lean) — **not advanced**; the device and the dispatch path ran, the kernel result is wrong |
| grade | **measured** (device init, build, three dispatch attempts); no timing and no validated output |
| instance | vast.ai `60930285`, host driver 580.173.02, South Korea; destroyed after the run |
| GPU | NVIDIA GeForce RTX 3060, 12288 MiB, Ampere, 360 GB/s spec |
| driver | 580.173.02, **proprietary** kernel module; userspace 580.173.02 |
| container | `ubuntu:24.04`, glibc 2.39; host `NVRM ... 580.173.02`, GCC 11.4 (Ubuntu 22.04) |
| host | 20 vCPUs, 15.9 GiB RAM (sister 3060 host `60926662`: kernel 5.15.0-186) |
| price | $0.04417/h |

## Result

Vulkan initializes on a vast.ai NVIDIA host. `vulkaninfo --summary` exits 0 and lists the
discrete GPU as `GPU0`, with `driverID = DRIVER_ID_NVIDIA_PROPRIETARY` and
`driverInfo = 580.173.02`:

```
VULKAN_EXIT=0
GPU0:
	apiVersion         = 1.4.312
	deviceName         = NVIDIA GeForce RTX 3060
	driverID           = DRIVER_ID_NVIDIA_PROPRIETARY
	driverInfo         = 580.173.02
GPU1:
	apiVersion         = 1.4.318
	deviceName         = llvmpipe (LLVM 20.1.2, 256 bits)
	driverID           = DRIVER_ID_MESA_LLVMPIPE
	driverInfo         = Mesa 25.2.8-0ubuntu0.24.04.4 (LLVM 20.1.2)
VULKAN_NVIDIA=present
```

The repository then builds and enters its own runtime: `Gpu::new()` selected the NVIDIA device
(`device: NVIDIA GeForce RTX 3060`), and `examples/gpu_gemv_bench` recorded and submitted its
dispatches. The result is wrong: the decode GEMV writes **zero** to every checked row.

```
build: BUILD_EXIT=0
device: NVIDIA GeForce RTX 3060
pack: weights=67895296 words_bytes=33947648 dm_bytes=8486912 total=42434560 bits_per_weight=5.0000
check decode segs=8 row=0    got=0e0 want=-4.2433590087890625e3 diff=4.2433590087890625e3
check decode segs=8 row=1792 got=0e0 want=-9.385692291259766e3 diff=9.385692291259766e3
check decode segs=8 row=3583 got=0e0 want=-7.822440856933594e3 diff=7.822440856933594e3
thread 'main' panicked at examples/gpu_gemv_bench.rs:294:9:
decode segs=8: rel 1e0 > 1e-4
```

All three runs exit 101 at the same check (`RUN1_EXIT=101 RUN2_EXIT=101 RUN3_EXIT=101`), so the
failure is deterministic. The f32 face runs first and fails; the exact integer face, the prefill
tile and the float control never execute. There is no timed row and no GB/s from this run.

### Ceiling

No Vulkan bandwidth was measured, so no measured ceiling exists for this path. The arithmetic:

| bandwidth | source | bytes/token | ceiling | vs 20 tok/s |
|---|---|---|---|---|
| 360 GB/s | RTX 3060 **spec peak** (model) | 0.998244352 GB | 360.7 tok/s | 18.0× |
| 539.5 GB/s | measured CUDA probe on an **RTX 2080 Ti** (`1794831271-gpu-gemv.md`) | 0.998244352 GB | 540.5 tok/s | 27.0× |

The first row is a spec number, not a measurement of this card. The second is a measurement of a
different card and a different kernel (CUDA, not `src/gpu`). Neither is a Vulkan result.

## What made initialization work

Two container conditions, both necessary on the hosts seen:

1. **The GL/Vulkan userspace is injected.** The working host had
   `/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.173.02` and
   `/etc/vulkan/icd.d/nvidia_icd.json` (`library_path libGLX_nvidia.so.0`, `api_version 1.4.312`)
   at container start.
2. **The X/GL userland that `libGLX_nvidia` dlopens is installed.** After
   `libvulkan1 vulkan-tools`, the bring-up installs `libx11-6 libxext6 libglvnd0 libgl1 libegl1
   libxcb1 libxau6 libxdmcp6`. Without these the ICD negotiates and fails; `ldd` does not show
   them because they are loaded with `dlopen`.

A control host makes condition 1 visible: an RTX 3080 on driver **595.71.05 (open kernel
module)**, created without `NVIDIA_DRIVER_CAPABILITIES=all`, got no graphics userspace at all —
`/etc/vulkan/icd.d/` was empty and `libGLX_nvidia.so.595.71.05` did not exist. `vulkaninfo
--summary` still exited 0, listing only llvmpipe. Exit 0 is therefore not success; the NVIDIA
device must be in the list. (The working 3060 was created with `--env '-e
NVIDIA_DRIVER_CAPABILITIES=all'`; the variable reads empty inside the container, so the flag's
effect is at container-create time, not in the shell.)

That control does **not** isolate the kernel-module flavour: it had neither the proprietary module
nor the graphics libs. The four earlier hosts that returned `-3` from
`vk_icdNegotiateLoaderICDInterfaceVersion` (570.133.07, open module) did have `libGLX_nvidia`, but
lacked the X/GL userland. The common factor in every failure is the userspace, not the module.

## Commands

Created with the contents of an onstart script (SSH key injection was broken on this account for
every instance after the first two, so the run used vast's onstart plus container logs):

```sh
vastai create instance <offer> --image ubuntu:24.04 --disk 40 --ssh --direct --cancel-unavail \
  --env '-e NVIDIA_DRIVER_CAPABILITIES=all' --onstart-cmd "$(cat onstart.sh)"
```

The script's load-bearing part:

```sh
apt-get update -qq
apt-get install -y -qq libvulkan1 vulkan-tools libx11-6 libxext6 libglvnd0 libgl1 libegl1 \
  libxcb1 libxau6 libxdmcp6 curl git build-essential ca-certificates
vulkaninfo --summary > /root/vulkaninfo.txt 2>&1; echo "VULKAN_EXIT=$?"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
  --default-toolchain 1.89.0
git clone https://github.com/sanguinehost/apeiron /root/apeiron && cd /root/apeiron
git checkout 735a7b06
# examples/gpu_gemv_bench.rs is untracked at HEAD; it is restored from a gzip+base64 payload.
cargo build --release --locked --features gpu --example gpu_gemv_bench -j 20
for r in 1 2 3; do ./target/release/examples/gpu_gemv_bench --quick; done
```

`cargo build` compiled the example unchanged: there were no compile errors to fix. The example is
the untracked file at repository HEAD `735a7b06a78a4a5a0e41f1591d83557f88303313`; a local
`cargo check --release --locked --features gpu --example gpu_gemv_bench` also passed before the
run.

Raw shape: `m=3584 k=18944 carrier=Q4 local=256 batch_tokens=512 quick=true`. `--quick` is one
split-K segment count (`segs=8`) and one prefill tile (`bt=32`). Because the first check panics,
neither `--no-prefill` nor `--no-float` changes what ran: nothing past the decode f32 face ran.

## What this does not cover

- **A correct Vulkan kernel.** The dispatch writes zeros. V1 is not advanced; initialization and
  submission are not a result.
- **Timing.** No timed median, no spread, no GB/s. The only bandwidth rows are a spec peak and a
  CUDA measurement on different silicon.
- **The cooperative-matrix test.** `tests/gpu_coopmat_gemm.rs` was not run: the benchmark panicked
  first, the box had no Qwen2.5-0.5B model fixture, and the RTX 3060's
  `VK_KHR_cooperative_matrix` support was not recorded.
- **The 7B pack.** One synthetic `ffn_down` projection at 5.0 b/w (42.4 MB), not 0.998 GB/token,
  no attention, no KV, no head, no tokenizer.
- **Quality.** No PPL, no Q4_K_M comparison, no real weights.
- **Prefill.** `k > 1` batching did not execute.
- **The open-module question.** The RTX 3080 control lacked graphics userspace entirely, so it
  cannot say whether a graphics-injected open-module host initializes.
- **rustc, LLVM, the driver's shader compiler, the kernel ABI, and the `unsafe` container.**
- **SSH.** The vast.ai container stopped accepting the account key after roughly ten minutes on
  two hosts and never accepted it on three later hosts, though `attach ssh` reports the key
  associated. The onstart/log channel is why the run completed at all.

## Next lever

The zero is in the first dispatch, not the device. `got=0e0` on all three rows means `partial` (or
`out` when `segs=1`) was never written. Before any timing, run the **integer face first** on
`segs=1` and on `segs=8`:

- `segs=1`, unit `(d, lo)`, codes in `[-7, 7]` isolates the dyn GEMV's six bindings and its
  `groups=(ceil(M/256), 1)` dispatch. If that is exact, the stage-2 reduction's
  `(partial, ones, out)` binding order or its `segs*M` layout is wrong.
- If `segs=1` is still zero, compare `gemv_dp4a_dyn_bytes(M, K, 256, 1, Carrier::Q4)`'s
  `OpDecorate ... Binding` list with the example's `[W, XQ, OUT, DM, XS, SX]` order, and its
  `local_size_x` with `LOCAL=256`. A mismatch asserts in `bind`, so the remaining candidates are
  the entry-point name and the `gl_GlobalInvocationID` row guard.

The device itself is no longer the question: this host initializes it, and `Gpu::new()` accepts it.
The next run should use the same `ubuntu:24.04` + X/GL userland recipe and print the NVIDIA line
from `vulkaninfo`, not merely its exit status.

## Instances and credit

Instances created and destroyed by this task: `60925916` (host DNS failure during image setup, no
work), `60926662` (RTX 3060, first successful `vulkaninfo`, SSH lost), `60928105` (RTX 3080, no
graphics userspace, only llvmpipe), `60928665` (RTX 3060, SSH lost, reboot did not run onstart),
`60930285` (RTX 3060, the run above).

`vastai show invoices` for those five ids totals **$0.046** (GPU $0.033, storage $0.008, download
$0.005), under the $0.35 cap. Credit was `4.45956891655` before the first create and
`4.23432088655` after the last destroy; the larger credit drop includes other sessions' instances
(`60919432`, `60924353`, then `60929592`), which this task did not touch.

`vastai show instances` after the campaign lists only `60929592` (not this task's). Every instance
this task created is destroyed.
