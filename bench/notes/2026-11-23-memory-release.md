# The layer-0 tape and frame release

Status: `tested` for the release/reallocation equality on the evaluation and forward paths;
`measured` for the L4 memory probes, the gate probe and the traced layer-0 round. Machine: `nq-l4`
(GCE L4, heap 23,034 MiB, 31 GB host), 2026-11-23. Tree: `272397a5` plus the tape/frame release.
Criterion: Q3 campaign execution (L4 VRAM and preflight unblock). This note does not advance
`GOAL.md` M1, which is laptop serving memory at 200k context; it does not close Q3 either (see
"What this does not cover").

## The failure

The 128-sample layer-0 preflight failed at `mlp.gate_proj` phase 1 (ADMM): the failing allocation
was refused with `vkAllocateMemory failed: VkResult -2` (`bench/gpu/1795396294-8b-sharing/logs/trace-gate.log:238`), and the 0.5 s
`nvidia-smi` poll reads 22,540 MiB against the 23,034 MiB heap (`bench/gpu/1795396294-8b-sharing/logs/gpu-trace.log:1520,1524`).
Two live sets grew with the linear:

- the non-fact tape at `T = batch·n_ctx = 8192`. `TapeBufs::alloc` at d 4096, a 4096, f 12288
  allocates `11·T·d + 2·T·a + 6·T·f = 3.875 GiB` for the block tape (`block.rs`, 21 buffers) and
  `15·T·d + 6·T·kw + 3·T·nh = 2.065 GiB` for the front tape (`front.rs`, 25 buffers), 5.94 GiB
  together (the formulas omit the two `T`-word loss buffers and `amax`, 96 KiB in total), never
  released before the fit;
- the ADMM frame cache (`src/gpu/limb/admm.rs`, `FRAMES`): one frame per `[k, n, r, …]` for the
  process. The `gate_proj` pair adds about 10.44 GiB across its build
  (`bench/gpu/1795396294-8b-sharing/logs/probe-gate.log:209-286`), and `admm_loss`'s limb product
  allocates a further 1.028 GiB in `LimbGemm::bind` (`:288-292`), 11.5 GiB across `209-292`.

The gate frame needs both sides resident during the iteration, and the 5.94 GiB wide tape was
still live. The release separates them in time: the tapes go before ADMM, the frames go before the
fact tape.

## The change

- `Vk` records every buffer allocation (`src/gpu/vulkan.rs`): handle, memory, whether `Vk` owns
  the memory, and the byte size. `free_handles` destroys a named set after a queue drain,
  unmapping first; `handles_since(mark)` names a suffix for a composite object; `free_since`
  releases a suffix through the same path. Removal preserves allocation order so an
  outstanding mark keeps naming the same suffix.
- `BlockTune::TapeBufs` / `FrontTune::TapeBufs` hold exactly the `t`-sized buffers. `release_tape`
  frees them and keeps every parameter, latent, Adam moment and the host staging pair; it also
  clears the step's `pre`/`post` copy lists and the recorded tape lists, which name the freed
  handles. `set_t` allocates the tape at the new width when none is live or the width changed,
  then re-records. `run`, `forward`, `train`, `batch`, `record`, `retape`, `set_loss_scale`,
  `output`, `dy`, `stage` and `FrontTune::backward` assert the tape is live. The raw handle
  getters (`BlockTune::{cols,out,da,dh,dy_col}`, `FrontTune::{ao,cols,dao,dx,dres}`) do not; they
  return a freed handle if called between `release_tape` and `set_t`, and the block driver never
  calls them in that window (debt, not a path).
- `FrontTune::retape` frees the previous `attn_prep` metadata buffer after the new tapes replace
  it, so that buffer no longer accumulates. The three `u32` metadata buffers of `attn_bwd::tape`
  and the pipeline/descriptor objects each `set_t` creates are not freed (below).
- `Frame` captures the handles it and its lazy `Rank1` allocate, and `release_frames` drains
  `FRAMES` and frees them. A `Frame::build` that refuses partway frees its partial suffix. A
  release bumps an epoch that empties every `Rank1::commits` cache, because a commit command
  buffer is keyed by the peer's `Z_x` handle and the driver reissues handle integers.
  `learn::nanoquant::release_device_frames` exposes the release.
- `tests/learn_block_tune.rs` phase 1 releases both tapes before the fit, then releases the
  frames and calls `set_t(seq)` before the stage change. The fact phase runs on the 2048-column
  tape instead of the 8192-column one.

## Equality

Four `--exact` tests on `nq-l4` (not the library suite; `bench/gpu/1795396294-8b-sharing/logs/release-tape-tests.log`). Two
compare a released-and-reallocated tuner with a fresh tuner, and two compare a tuner whose `set_t`
reallocates the tape without an explicit release with a fresh one, on the evaluation forward
(`BlockTune::output`) and the front forward (`FrontTune::output`) at `t: 16 → 8, d: 32` and
`n_ctx: 16, chunks: 2 → 1`:

| test | result |
|---|---|
| `release_tape_then_set_t_matches_a_fresh_tuner` | loss wide `2.226209e2`, narrow `1.054162e2`, fresh `1.054162e2`; narrow loss and output bitwise equal, output rel `0.00e0` |
| `release_tape_then_set_t_matches_a_fresh_front` | output bitwise equal, rel `0.00e0` |
| `set_t_matches_a_fresh_tuner` | narrow loss and output bitwise equal to fresh; the wide output's first 8 columns bitwise equal to the narrow |
| `set_t_matches_a_fresh_front` | output bitwise equal to fresh and the wide prefix bitwise equal to the narrow |

The tests assert `f32::to_bits` equality on the outputs and on the narrow/fresh loss. They cover
the evaluation forward (`BlockTune::run(false)`, `FrontTune::forward`) only. Training/backward/Adam
(`run(true)`, `backward`, `backward_cols`, `train`, the moments and `set_loss_scale`),
`set_stage`, `set_cols`, widening `8 → 16`, repeated release/re-tape cycles, the guard panics,
`release_frames`/`PartialFrame`/`Rank1::commits`, and the 8B 4-chunk/host-stream shapes are not
covered. The four `--exact` runs filter the other 772 library tests; the full workspace suite is
the separate gate below.

## The gate probe

`bench/gpu/campaign/probe-gate.sh`, resumed from the layer-0 checkpoint at slot 4 phase 1
(`APEIRON_NQ_PHASES=2`): gate ADMM then gate fact.

| quantity | before | after |
|---|---:|---:|
| gate ADMM | refused (`VkResult -2`, 22,540 MiB heap) | 505.34 s, recon norm 0.2209, per_el `1.4947e-4` |
| gate fact | — | 1024 steps, 176.4 s |
| `nvidia-smi` peak | 22,540 MiB | **18,452 MiB** |
| round total | — | 702.5 s |

Logs: `bench/gpu/1795396294-8b-sharing/logs/probe-gate.log`, `bench/gpu/1795396294-8b-sharing/logs/probe-gpu.log` (0.5 s poll). `APEIRON_MEM_TRACE=1` prints each
allocation of at least 16 MiB with the live `device-local`, `host-visible` and `imported` bytes and
the device-local peak (`src/gpu/vulkan.rs`, `trace_alloc`); the split is by requested property, not
by selected heap, and this driver counts unimported host-visible buffers in the heap (the four
4 GiB calibration streams read as device use before the host import). `nvidia-smi` is the heap
artifact; the counter names the pools. The probe's log predates the split and prints one `live`
sum over every registered buffer, so it reads high.

The layer then finished at the same tree: `mlp.up_proj` ADMM 510.23 s and fact 188.7 s,
`mlp.down_proj` nonfact 153.0 s, ADMM 509.32 s (recon norm 0.3311) and fact 214.0 s. The layer-0
pack is 0.995619 b/w over its 192,937,984 replaced weights, 24,011,584 factor bytes; the header
names 36 layers and the other 35 stay dense in the skeleton. The 7-token PPL is 17,039.4
(mean nll 9.7433), rc 0; the run's `nvidia-smi` peak is 19,130 MiB, in the last polled interval
before the fit returns (the advance is the last 12 s; the poll carries no pass label)
(`bench/gpu/1795396294-8b-sharing/logs/preflight-8b.log`, `bench/gpu/1795396294-8b-sharing/logs/preflight-gpu.log:11639-11651`). The preflight is a resumed run: slots 0–3 ran before this
change, slot 4 phase 1–2 under `probe-gate.sh`, slots 4 phase 3 and 5 under the release.

The single-run measurement on the final tree is `bench/gpu/campaign/trace-round.sh`: one layer-0
round over all seven linears, rc 0, `round total 4587.4 s` (76.5 min), `nvidia-smi` peak
**19,905 MiB** (`bench/gpu/1795396294-8b-sharing/logs/trace-round.log`,
`bench/gpu/1795396294-8b-sharing/logs/trace-gpu.log`), the poll's maximum. The run is layer 0 only
(`APEIRON_NQ_BLOCKS=0..1`, 1 of 36 blocks), `n = 1`, at `DEV_INV_CUT=96` (the campaign's 48 was not
this run) and without `APEIRON_MEM_TRACE=1`, so it has no pool split; the 0.5 s poll carries no
pass label, so the pass that owns the peak is not asserted. The peak is a layer-0 figure, not a
36-layer or cut-48 bound. The tape release, the frame release and the narrow fact tape are on that
path throughout. The ADMM sub-phase split is [2026-11-23-admm-split.md](2026-11-23-admm-split.md).

The host-import probe (`bench/gpu/campaign/hostmem-probe.sh`, `examples/nq_hostmem_probe.rs`)
imports four 4 GiB anonymous host mappings, touches all 16 GiB and copies each 4 GiB both ways
through one 4 GiB device buffer: rc 0, `nvidia-smi` peak **4,163 MiB** over 548 polls
(`bench/gpu/1795396294-8b-sharing/logs/hostmem-probe.log`, `bench/gpu/1795396294-8b-sharing/logs/hostmem-probe-gpu.log`). The same poll reads 67 MiB while the
16 GiB of imports are touched and no copy target exists, then 4,163 MiB once the 4 GiB copy target
is allocated; the imports are host pages, not heap.

The final-tree export (`bench/gpu/1795396294-8b-sharing/logs/export-l0.log`) skips every fit because all seven layer-0 checkpoints
are done, re-advances and writes `loop-8b-trace/nq-100/pack-l0.nqp`: 0.995619 b/w, 24,011,840 bytes,
sha256 `adf8354424f8990e9d013453628fe9b83ce6fa16493a503bb8fe0150e01924a0`, mirrored to
`gs://sanguine-checkpoints-scribe-461316/nq/packs/q3-8b-l0-final.nqp`. The hash is identical to the
earlier export from the same checkpoints; the b/w is the preflight's.

The compile, lint and format gate at the CI pin (`nightly-2026-04-14`, `bench/gpu/1795396294-8b-sharing/logs/final-check.log`) is green on the
final tree: `cargo check --tests`, the four `--exact` release tests, `cargo clippy --workspace
--all-features --all-targets --locked -- -D warnings` and `cargo fmt --all -- --check`, rc 0. It is
not the full contributor gate; that suite is red on this host (below). The
L4's default stable 1.100 flags pre-existing `src/map`/`src/engine` lints and the 1.89 MSRV cannot
resolve the source's `clippy::manual_is_multiple_of` expectation, so neither is the gate toolchain;
the CI file pins the nightly.

The full workspace suite is red on the same host, and the failures are outside the diff. Under the
CI test job's `VK_ICD_FILENAMES=/nonexistent`, the library target reads 694 passed, 18 failed, 62
ignored (`bench/gpu/1795396294-8b-sharing/logs/full-tests.log`, rc 101; the run stops at the first
failing target). Fourteen failures are committed-SPIR-V byte checks (thirteen `*_spirv_is_the_glsl`
and `kdb_spv_is_fresh`): the host's `glslc` is shaderc 2023.8, which lacks
`GL_EXT_integer_dot_product`, so the committed blobs are not what that compiler makes of the GLSL.
Four are device tests that assert a device instead of skipping under the empty ICD list
(`attn_bwd_device_matches_ref`, `_ungated`, `_ungated_small_dy`, `delta_bwd_device_matches_ref`;
each panics at `Vk::load` with `vkCreateInstance failed (no ICD?)`). The `--no-fail-fast
--skip gpu::` run over every target reads 1,101 passed, 4 failed, 97 ignored
(`bench/gpu/1795396294-8b-sharing/logs/cpu-tests.log`): three integration-test SPIR-V checks
(`coopmat_spirv_is_the_glsl`, `attn_coopmat_spirv_is_the_glsl`, `splitk_reduce_spirv_is_the_glsl`)
and `gpu_train_baseline` (`models/gguf/qwen05b-Q4_K_M.gguf` not found; `APEIRON_MODELS` names the
fixture root). None of the four equality tests failed. This note records the suite red rather than
claiming it green.

The reference R1 on the same L4 and the same linear (`bench/gpu/campaign/ref-r1.py`, 400 outer
iterations, linear rho, seed 0) takes 47.37 s at 1.4739e-4 per-element error and 1,915 MiB peak
(`bench/gpu/1795396294-8b-sharing/logs/ref-r1-gate.log`); the port's gate ADMM is 505.34 s at 1.4947e-4 (+1.4%). The baseline is
not an equality: the reference factors the raw `bf16` `.safetensors` weight, the port the
non-fact-tuned `f32` weight rounded to Q32.32; the reference draws its initialization from
NumPy/PyTorch `randn`, the port from SplitMix64; and the port's 505.34 s is the whole
`factorize_floor` (plane build, both frames, pipelines, 400 iterations, `admm_loss`, export),
while 47.37 s is the reference's `factorize_admm_nanoquant` call alone with the inputs already on
the device. It is a same-device, same-shape, same-rank, same-iteration comparison of the two
calls at the two libraries' boundaries; it is not a matched-arithmetic ratio. The traced
sub-phase split is [2026-11-23-admm-split.md](2026-11-23-admm-split.md).

## What this does not cover

- The freeing returns buffers to the driver; `nvidia-smi` may not show the drop immediately, and
  no compaction is attempted. The measured peak is the live set at allocation time.
- Pipelines, descriptor set layouts, pools and sets, shader modules and reusable command buffers
  are not freed. `retape` allocates a layout, pool, set, shader module and pipeline on every width
  change (`Vk::ssbo_set_at`, `Vk::pipeline`), and `attn_bwd::tape` allocates three `u32` metadata
  buffers per `FrontTune::retape` that no list tracks; these are process leaks bounded by the
  number of width changes, not by the tape size. The four equality tests filter the other 772
  library tests, so training/backward, `set_stage`, `set_cols`, widening and 8B shapes are not
  covered.
- The counter does not see pipelines, descriptor pools, shader modules, command pools or driver
  padding, and imported memory is outside the heap only while the caller has faulted its pages in.
- The reference baseline uses a different RNG, a different input dtype and a narrower timing
  boundary (above); the `+1.4%` recon error is not a matched-draw comparison.
- The 0.6B fixtures do not exercise the 8B frame sizes; the probe and the layer-0 run are the
  measured evidence.

## Review

`gemini-4-hydrogen-xhigh` reviewed the change and both 2026-11-23 notes; transcript
`bench/gpu/1795396294-8b-sharing/hydrogen-review-2.txt`. It found no use-after-free or double free
on the campaign's sequential path. It refused, and the text above corrects: the `M1` criterion
(this work is Q3 campaign execution); repo-relative `logs/` paths; the 11.5 GiB `FRAMES`
attribution; "`set_t` re-records, not reallocates" (each width change allocates a layout, pool, set,
shader module and pipeline, and `attn_bwd::tape` three metadata buffers, which is the leak listed
above); the unqualified 19,905 MiB peak; the missing full-suite run; and the global cut-48 from a
q-only sweep (the split note states the `r = 3040` sweep is open and claims no gain). It named the
unguarded raw handle getters, listed in "The change" as debt; the block driver does not call them
between `release_tape` and `set_t`. It corrected the split note's `commit` division, its rounded
ceiling and its `inv` lock wait; those are fixed there. The review is a critique, not evidence:
every number above resolves to a log or a source file named beside it.
