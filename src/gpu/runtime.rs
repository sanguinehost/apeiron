//! runtime — the model-agnostic GPU dispatch layer over [`super::vulkan::Vk`].
//!
//! The device-side parallel of the CPU `Table`: a device + buffers + pipelines + a dispatch
//! abstraction, so the `simt`/`simt_nn` `_bytes` builders (the ops) run standalone, not only
//! through the decode harness. Sovereign — every call bottoms out in the dlopen'd `Vk` entry
//! points, zero crates. The generic glue is lifted from `examples/simt_decode.rs`
//! (`GpuCtx::dev`/`up`, the staging upload, the timestamp read); the model-specific graph
//! (`Node`/`Step`/`Gemv`, the layer loop) stays in that example.

#[cfg(feature = "gpu")]
use super::vulkan::*;

#[cfg(feature = "gpu")]
const STAGE_CAP: usize = 256 << 20; // 256 MiB staging window (matches simt_decode)

// ── the pure handoff arithmetic ────────────────────────────────────────────────────────────────
//
// These are the parts of this file with no Vulkan dependency; `proof/extract/ports/gpu_runtime_port.rs`
// roots them as `word_gpu_runtime_*`, and `proof/corpus/Gpu/Handoff.lean` proves their statements.
// The bodies below call them, so the proofs bind the executing path (`Gpu::scatter`,
// `Gpu::staged_dispatch`, `Gpu::write`, `Gpu::bind`, `Gpu::step_barrier`).
//
// `allow(dead_code)` applies only to a build without the `gpu` feature, where the Vulkan half of
// this file is not compiled and the pure half has no caller.

/// 256-byte alignment of the next staged block: `(n + 255) & !255`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn stage_align(n: usize) -> usize {
    (n + 255) & !255
}

/// The staging-window fit test: does a block of `len` bytes at `off` fit `cap`?
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn stage_fit(off: usize, len: usize, cap: usize) -> bool {
    off + len <= cap
}

/// The `BufferCopy` `staged_dispatch` records for a block staged at `off`: `(src, dst, size)`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn stage_region(off: usize, len: usize) -> (u64, u64, u64) {
    (off as u64, 0, len as u64)
}

/// The staging source offset of a `scatter` region.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn scatter_src(off: usize, s: u64) -> u64 {
    off as u64 + s
}

/// The `bind` count check: `buffers.len() == pipe.bindings`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn binds_exactly(n: usize, bindings: u32) -> bool {
    n == bindings as usize
}

/// The next chunk of `write`: `min(bytes - done, cap)`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn write_chunk(done: usize, bytes: usize, cap: usize) -> usize {
    let n = bytes - done;
    if n < cap {
        n
    } else {
        cap
    }
}

/// Does `step_barrier` emit a barrier for this `PREFILL_BAR` mode? Mode 9 does not.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn bar_emits(mode: u32) -> bool {
    mode != 9
}

/// `PREFILL_BAR` — the inter-dispatch sync formulation for [`Gpu::record`]. See
/// [`Gpu::step_barrier`] for the modes.
#[cfg(feature = "gpu")]
pub fn bar_mode() -> u32 {
    static BAR_MODE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BAR_MODE.get_or_init(|| {
        std::env::var("PREFILL_BAR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// An owned device (or host-visible) buffer: the Vulkan buffer + its backing allocation. The
/// handles are leaked on drop for now (the decode path leaks too; a later pass adds `Drop`).
#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
pub struct Buffer {
    /// The Vulkan buffer handle.
    pub buffer: Handle,
    /// Its backing device allocation.
    pub memory: Handle,
}

/// A compiled compute pipeline + the layouts its descriptor sets bind against. The SPIR-V is
/// compiled ONCE here; `bind` allocates a fresh descriptor set against `dsl` per buffer set, so
/// one pipeline serves many inputs (the reusable-primitive shape).
#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
pub struct Pipeline {
    /// The compute pipeline handle.
    pub pipe: Handle,
    /// The pipeline layout (one descriptor set of storage buffers).
    pub layout: Handle,
    /// The descriptor set layout — `bindings` storage buffers.
    pub dsl: Handle,
    /// Storage buffer binding count.
    pub bindings: u32,
}

/// A descriptor set: its `bindings` storage buffers bound at (set 0, binding i).
#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
pub struct DescriptorSet {
    /// The descriptor set handle.
    pub set: Handle,
}

/// One dispatch: run `pipe` over `dset` with `groups` workgroups (x, y; z = 1).
#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
pub struct Dispatch {
    /// The pipeline to run.
    pub pipe: Pipeline,
    /// The buffers it reads/writes.
    pub dset: DescriptorSet,
    /// Workgroup counts (x, y).
    pub groups: (u32, u32),
}

/// The runtime handle — the loaded Vulkan device plus a persistent mapped staging buffer for
/// host→device uploads.
#[cfg(feature = "gpu")]
pub struct Gpu {
    /// The loaded Vulkan device — the low-level surface, for calls the runtime doesn't wrap.
    pub vk: Vk,
    stage_buf: Handle,
    stage_ptr: *mut u8,
    stage_cap: usize,
    /// One persistent command buffer, reset + re-recorded each submit (no per-dispatch pool alloc).
    cmd_buf: CommandBuffer,
}

#[cfg(feature = "gpu")]
impl Gpu {
    /// Load the Vulkan device and allocate the staging buffer. `Err(msg)` = honest skip (no
    /// loader / no compute device / no host-visible memory), never a hard failure — the same
    /// contract as [`Vk::load`].
    pub fn new() -> Result<Gpu, String> {
        Self::with_stage_capacity(STAGE_CAP)
    }

    /// Loads the Vulkan device with a caller-sized persistent upload window.
    pub fn with_stage_capacity(stage_cap: usize) -> Result<Gpu, String> {
        let stage_cap = stage_cap.max(4);
        let vk = unsafe { Vk::load()? };
        let (stage_buf, stage_mem) = unsafe {
            vk.buffer(
                stage_cap as u64,
                BUFFER_USAGE_TRANSFER_SRC,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
        }
        .ok_or_else(|| "no host-visible memory for the staging buffer".to_string())?;
        let stage_ptr = unsafe { vk.map(stage_mem) }.cast::<u8>();
        let cmd_buf = unsafe { vk.reusable_command_buffer() };
        Ok(Gpu {
            vk,
            stage_buf,
            stage_ptr,
            stage_cap,
            cmd_buf,
        })
    }

    /// The persistent command buffer, reset for a fresh recording. Every `submit_wait` fully drains
    /// the queue, so re-recording the one buffer is race-free (no in-flight overlap).
    unsafe fn cmd(&self) -> CommandBuffer {
        self.vk.reset_cb(self.cmd_buf);
        self.cmd_buf
    }

    /// The device name (e.g. "NVIDIA GeForce RTX 5080").
    pub fn device_name(&self) -> &str {
        &self.vk.device_name
    }

    /// A device-local storage buffer of `bytes` (min 4). Uninitialized — fill via a dispatch or
    /// `upload`. `TRANSFER_DST` so a staging copy can target it.
    pub fn device_buffer(&self, bytes: usize) -> Buffer {
        let (buffer, memory) = unsafe {
            self.vk.buffer(
                bytes.max(4) as u64,
                BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST,
                MEMORY_DEVICE_LOCAL,
            )
        }
        .expect("device-local storage buffer");
        Buffer { buffer, memory }
    }

    /// A host-visible, coherent storage buffer of `bytes`, mapped for CPU read/write. Returns the
    /// buffer and its mapped pointer — the readback path (copy a result out, or a small param
    /// buffer the CPU writes each step).
    pub fn host_buffer(&self, bytes: usize) -> (Buffer, *mut u8) {
        let (buffer, memory) = unsafe {
            self.vk.buffer(
                bytes.max(4) as u64,
                BUFFER_USAGE_STORAGE,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
        }
        .expect("host-visible storage buffer");
        let ptr = unsafe { self.vk.map(memory) }.cast::<u8>();
        (Buffer { buffer, memory }, ptr)
    }

    /// A host-visible storage buffer of `bytes` optimized for CPU **reads** (readback): prefers a
    /// HOST_CACHED mapping so reading the result is cached-DRAM speed, not the ~0.3 GB/s write-combined
    /// path a plain [`Gpu::host_buffer`] gives (MILESTONE 3: the C-tensor readback dominated the
    /// coopmat step — 150 ms to read 48 MB of write-combined memory). Falls back to plain host-visible
    /// if no cached type exists.
    pub fn host_read_buffer(&self, bytes: usize) -> (Buffer, *mut u8) {
        let cached = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT | MEMORY_HOST_CACHED;
        let plain = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
        for props in [cached, plain] {
            if let Some((buffer, memory)) = unsafe {
                self.vk.buffer(
                    bytes.max(4) as u64,
                    BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST,
                    props,
                )
            } {
                let ptr = unsafe { self.vk.map(memory) }.cast::<u8>();
                return (Buffer { buffer, memory }, ptr);
            }
        }
        panic!("no host-visible memory for the readback buffer");
    }

    /// Upload `data` into a fresh device-local buffer via the staging copy (chunked at the
    /// staging window). The `simt_decode` `GpuCtx::up` path, generic over element type.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        let bytes = std::mem::size_of_val(data);
        let buf = self.device_buffer(bytes);
        unsafe { self.write(buf, data.as_ptr().cast::<u8>(), bytes) };
        buf
    }

    /// Overwrite the start of an EXISTING device buffer with `data` (staging copy) — the
    /// persistent-buffer reuse path (no fresh allocation), so a training loop that redraws its
    /// operands each step keeps one device buffer instead of leaking one per call.
    pub fn write_into<T: Copy>(&self, buf: &Buffer, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        unsafe { self.write(*buf, data.as_ptr().cast::<u8>(), bytes) };
    }

    /// **Scatter compacted host data into existing device buffers as region copies — one staging
    /// pass, one command buffer, one submit.** Each group is `(dst buffer, compacted bytes, regions)`
    /// with regions `(src_off, dst_off, size)` in bytes, `src_off` relative to that group's staged
    /// block. This is the partial-upload path [`Gpu::write_into`] cannot express: a strided device
    /// buffer whose occupied PREFIX per row is a fraction of the allocation re-lays only that
    /// fraction, instead of the whole allocation (the prefill stack's KV seed was uploading full
    /// `tmax` buffers — ~1.2 GB at 8B — for a `pos0` of a few hundred). All groups must fit the
    /// staging window together; the caller chunks (the same contract `staged_dispatch` states).
    pub fn scatter(&self, groups: &[(&Buffer, &[u8], &[(u64, u64, u64)])]) {
        unsafe {
            let cb = self.cmd();
            self.vk.begin(cb);
            let mut off = 0usize;
            for (buf, bytes, regions) in groups {
                assert!(
                    stage_fit(off, bytes.len(), self.stage_cap),
                    "scatter: uploads exceed the staging window"
                );
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.stage_ptr.add(off), bytes.len());
                let regs: Vec<BufferCopy> = regions
                    .iter()
                    .map(|&(s, d, n)| BufferCopy {
                        src_offset: scatter_src(off, s),
                        dst_offset: d,
                        size: n,
                    })
                    .collect();
                if !regs.is_empty() {
                    (self.vk.copy_buffer)(
                        cb,
                        self.stage_buf,
                        buf.buffer,
                        regs.len() as u32,
                        regs.as_ptr(),
                    );
                }
                off = stage_align(off + bytes.len()); // 256-byte align the next group
            }
            self.vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_COMPUTE,
                ACCESS_TRANSFER_WRITE,
                ACCESS_SHADER_READ,
            );
            check((self.vk.end_cb)(cb), "vkEndCommandBuffer");
            self.vk.submit_wait(cb);
        }
    }

    /// Overwrite `buf`'s first `bytes` from host memory at `src` (staging copy, chunked).
    unsafe fn write(&self, buf: Buffer, src: *const u8, bytes: usize) {
        let mut done = 0usize;
        while done < bytes {
            let n = write_chunk(done, bytes, self.stage_cap);
            std::ptr::copy_nonoverlapping(src.add(done), self.stage_ptr, n);
            let cb = self.cmd();
            self.vk.begin(cb);
            let (src_offset, _, size) = stage_region(0, n);
            let region = BufferCopy {
                src_offset,
                dst_offset: done as u64,
                size,
            };
            (self.vk.copy_buffer)(cb, self.stage_buf, buf.buffer, 1, &region);
            self.vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_COMPUTE,
                ACCESS_TRANSFER_WRITE,
                ACCESS_SHADER_READ,
            );
            check((self.vk.end_cb)(cb), "vkEndCommandBuffer");
            self.vk.submit_wait(cb);
            done += n;
        }
    }

    /// Compile a compute pipeline from SPIR-V `spv` with `n_bindings` storage buffers at set 0.
    /// The pipeline and its layouts are created once; `bind` reuses them per buffer set.
    pub fn pipeline(&self, spv: &[u8], n_bindings: u32) -> Pipeline {
        unsafe {
            let dsl = self.dsl(n_bindings);
            let layout = self.pipeline_layout(dsl);
            let pipe = self.vk.pipeline(spv, layout);
            Pipeline {
                pipe,
                layout,
                dsl,
                bindings: n_bindings,
            }
        }
    }

    /// Is `VK_KHR_buffer_device_address` live? When it is, a kernel may be built in its
    /// `PhysicalStorageBuffer` form and bound with [`Gpu::bind_addrs`] instead of [`Gpu::bind`].
    pub fn has_bda(&self) -> bool {
        self.vk.has_bda
    }

    /// The GPU address of `b`, or `None` without the extension.
    pub fn device_address(&self, b: &Buffer) -> Option<u64> {
        unsafe { self.vk.device_address(b.buffer) }
    }

    /// **Bind an ADDRESS TABLE instead of descriptors.** `pipe` must be the `PhysicalStorageBuffer`
    /// form of a kernel: ONE storage-buffer binding holding the `u64` GPU addresses of `buffers`, in
    /// order. The kernel `OpConvertUToPtr`s each and dereferences it directly, so the driver has no
    /// descriptor range to compare an index against — the mandala mask inside the kernel becomes the
    /// only bound, which is what `MandalaMemory.masked_load_never_oob` describes.
    ///
    /// The table is a 48-byte device buffer built once at plan time and read once per workgroup, so
    /// it costs one uniform load against the ~200 range compares per hot-loop iteration it removes.
    pub fn bind_addrs(&self, pipe: &Pipeline, buffers: &[&Buffer]) -> DescriptorSet {
        assert_eq!(
            pipe.bindings, 1,
            "the PhysicalStorageBuffer form takes exactly one binding: the address table"
        );
        let addrs: Vec<u64> = buffers
            .iter()
            .map(|b| {
                self.device_address(b)
                    .expect("bind_addrs without VK_KHR_buffer_device_address")
            })
            .collect();
        let table = self.upload(&addrs);
        self.bind(pipe, &[&table])
    }

    /// Allocate a descriptor set binding `buffers` (one per binding, in order) against `pipe`.
    pub fn bind(&self, pipe: &Pipeline, buffers: &[&Buffer]) -> DescriptorSet {
        assert!(
            binds_exactly(buffers.len(), pipe.bindings),
            "bind count must match pipeline bindings"
        );
        let handles: Vec<Handle> = buffers.iter().map(|b| b.buffer).collect();
        DescriptorSet {
            set: unsafe { self.descriptor_set(pipe.dsl, &handles) },
        }
    }

    /// Record `dispatches` into one command buffer (a memory barrier between each, so a write is
    /// visible to the next dispatch's read), submit, and wait. The whole graph, one submit.
    pub fn run(&self, dispatches: &[Dispatch]) {
        unsafe {
            let cb = self.cmd();
            self.vk.begin(cb);
            self.record(cb, dispatches);
            check((self.vk.end_cb)(cb), "vkEndCommandBuffer");
            self.vk.submit_wait(cb);
        }
    }

    /// Like `run`, but returns the GPU-active time in microseconds (device timestamps around the
    /// whole recorded graph). `f64::NAN` if the device reports no timestamp support.
    pub fn time(&self, dispatches: &[Dispatch]) -> f64 {
        if self.vk.timestamp_valid_bits == 0 {
            return f64::NAN;
        }
        unsafe {
            let pool = self.vk.timestamp_pool(2);
            self.vk.reset_timestamps_host(&pool);
            let cb = self.vk.command_buffer();
            self.vk.begin(cb);
            self.vk
                .cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, 0);
            self.record(cb, dispatches);
            self.vk
                .cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, 1);
            check((self.vk.end_cb)(cb), "vkEndCommandBuffer");
            self.vk.submit_wait(cb);
            let ts = self.vk.read_timestamps(&pool);
            ts[1].wrapping_sub(ts[0]) as f64 * self.vk.timestamp_period as f64 / 1000.0
        }
    }

    /// Fused upload+dispatch: stage every `(dst, bytes)` into the shared staging buffer at distinct
    /// offsets, record the copies + one transfer→compute barrier + the `dispatches`, and submit ONCE
    /// (one `vkQueueWaitIdle` instead of one-per-upload-plus-one-per-run). The whole-step killer at
    /// MILESTONE 3 was ~3 queue drains per GEMM × ~75 GEMMs; this collapses each GEMM to one drain.
    /// All uploads must fit the staging window together (they do — f16 operands are ≤ tens of MiB).
    pub fn staged_dispatch(&self, uploads: &[(&Buffer, &[u8])], dispatches: &[Dispatch]) {
        unsafe {
            let cb = self.cmd();
            self.vk.begin(cb);
            let mut off = 0usize;
            for (buf, bytes) in uploads {
                assert!(
                    stage_fit(off, bytes.len(), self.stage_cap),
                    "staged_dispatch: uploads exceed the staging window"
                );
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.stage_ptr.add(off), bytes.len());
                let (src_offset, dst_offset, size) = stage_region(off, bytes.len());
                let region = BufferCopy {
                    src_offset,
                    dst_offset,
                    size,
                };
                (self.vk.copy_buffer)(cb, self.stage_buf, buf.buffer, 1, &region);
                off = stage_align(off + bytes.len()); // 256-byte align the next operand
            }
            self.vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_COMPUTE,
                ACCESS_TRANSFER_WRITE,
                ACCESS_SHADER_READ,
            );
            self.record(cb, dispatches);
            check((self.vk.end_cb)(cb), "vkEndCommandBuffer");
            self.vk.submit_wait(cb);
        }
    }

    unsafe fn record(&self, cb: CommandBuffer, dispatches: &[Dispatch]) {
        for (i, d) in dispatches.iter().enumerate() {
            (self.vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, d.pipe.pipe);
            (self.vk.bind_sets)(
                cb,
                PIPELINE_BIND_POINT_COMPUTE,
                d.pipe.layout,
                0,
                1,
                &d.dset.set,
                0,
                std::ptr::null(),
            );
            (self.vk.dispatch)(cb, d.groups.0, d.groups.1, 1);
            if i + 1 < dispatches.len() {
                self.step_barrier(cb);
            }
        }
    }

    /// The inter-dispatch sync formulation for the recorded graph, dialled by `PREFILL_BAR` — the
    /// prefill twin of `servefold::gpu::step_barrier`'s `DEC_BAR`. The prefill stack records ONE
    /// command buffer and submits ONCE, so it pays no per-submit cost; what it still pays is a
    /// pipeline drain between every pair of dependent dispatches, and this is the instrument that
    /// prices it.
    ///
    ///   0 = legacy `vkCmdPipelineBarrier`, global `VkMemoryBarrier` (the shipped default)
    ///   1 = sync2 `vkCmdPipelineBarrier2`, global `VkMemoryBarrier2` — same semantics, other path
    ///   2 = sync2, RAW-only (dst access = SHADER_READ, no WAR)
    ///   3 = sync2, execution-only (access 0 → 0) — MAY BE WRONG, isolates the drain from memory
    ///   9 = NO barrier: WRONG OUTPUT by construction. The compute floor of the same recorded
    ///       dispatch sequence, i.e. the number a graph/megakernel formulation is chasing.
    ///
    /// **MEASURED, and it says a Vulkan CUDA-graphs equivalent is NOT the prefill wall.** qwen2.5
    /// -0.5B, b = 640, one chunk of 267 dispatches / 266 barriers in ONE recorded command buffer
    /// with ONE submit, `SERVEFOLD_PREFILL_STACK`, POOL_WIDTH 6, idle box, five interleaved pairs:
    ///
    /// ```text
    ///   PREFILL_BAR=0 (correct)   23.4 25.1 22.7 25.3 23.8 ms   mean 24.06   26 632–28 186 tok/s
    ///   PREFILL_BAR=9 (no bar)    18.7 20.8 18.8 19.0 18.2 ms   mean 19.10   30 758–35 152
    /// ```
    ///
    /// So the barrier + pipeline-drain term is **4.96 ms of a 24.06 ms chunk = 20.6 %** (18.6 µs
    /// per barrier at this batch — the drain is the dispatch's own tail, not the ~4 µs constant the
    /// b = 1 decode path saw). Twice the 2.44 ms a subtraction had predicted, and still not enough:
    /// deleting ALL of it leaves 19.10 ms = 33 500 tok/s against a 12.04 ms / 53 163 tok/s bar. The
    /// GEMM stages alone are 7.4 ms of the chunk, so the lever went there instead
    /// (`simt::sb_lds`). The decode path's arithmetic is the opposite one and that is why the
    /// megakernel argument held there: 607 µs of a 1065 µs token, 57 %.
    pub(crate) unsafe fn step_barrier(&self, cb: CommandBuffer) {
        let mode = bar_mode();
        if !bar_emits(mode) {
            return;
        }
        match mode {
            1 => self.vk.barrier2(
                cb,
                STAGE2_COMPUTE,
                STAGE2_COMPUTE,
                ACCESS2_SHADER_WRITE,
                ACCESS2_SHADER_READ | ACCESS2_SHADER_WRITE,
            ),
            2 => self.vk.barrier2(
                cb,
                STAGE2_COMPUTE,
                STAGE2_COMPUTE,
                ACCESS2_SHADER_WRITE,
                ACCESS2_SHADER_READ,
            ),
            3 => self.vk.barrier2(cb, STAGE2_COMPUTE, STAGE2_COMPUTE, 0, 0),
            9 => {}
            _ => self.vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_COMPUTE,
                ACCESS_SHADER_WRITE,
                ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
            ),
        }
    }

    // ── descriptor plumbing: the buffer-independent parts of `Vk::ssbo_set`, split so one
    //    pipeline (built from the binding count) serves many descriptor sets ──

    /// A descriptor set layout of `n` compute storage buffers at (set 0, binding 0..n).
    unsafe fn dsl(&self, n: u32) -> Handle {
        let bindings: Vec<DescriptorSetLayoutBinding> = (0..n)
            .map(|i| DescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: SHADER_STAGE_COMPUTE,
                p_immutable_samplers: std::ptr::null(),
            })
            .collect();
        let dslci = DescriptorSetLayoutCreateInfo {
            s_type: ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: n,
            p_bindings: bindings.as_ptr(),
        };
        let mut dsl: Handle = 0;
        check(
            (self.vk.create_dsl)(self.vk.device, &dslci, std::ptr::null(), &mut dsl),
            "vkCreateDescriptorSetLayout",
        );
        dsl
    }

    /// A pipeline layout over a single descriptor set layout, no push constants.
    unsafe fn pipeline_layout(&self, dsl: Handle) -> Handle {
        let plci = PipelineLayoutCreateInfo {
            s_type: ST_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &dsl,
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
        };
        let mut layout: Handle = 0;
        check(
            (self.vk.create_pl)(self.vk.device, &plci, std::ptr::null(), &mut layout),
            "vkCreatePipelineLayout",
        );
        layout
    }

    /// A descriptor set (fresh pool) with `buffers` written at bindings 0..n.
    unsafe fn descriptor_set(&self, dsl: Handle, buffers: &[Handle]) -> Handle {
        let n = buffers.len() as u32;
        let dps = DescriptorPoolSize {
            ty: DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: n,
        };
        let dpci = DescriptorPoolCreateInfo {
            s_type: ST_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &dps,
        };
        let mut dpool: Handle = 0;
        check(
            (self.vk.create_dpool)(self.vk.device, &dpci, std::ptr::null(), &mut dpool),
            "vkCreateDescriptorPool",
        );
        let dsai = DescriptorSetAllocateInfo {
            s_type: ST_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: dpool,
            descriptor_set_count: 1,
            p_set_layouts: &dsl,
        };
        let mut dset: Handle = 0;
        check(
            (self.vk.alloc_dsets)(self.vk.device, &dsai, &mut dset),
            "vkAllocateDescriptorSets",
        );
        let infos: Vec<DescriptorBufferInfo> = buffers
            .iter()
            .map(|&b| DescriptorBufferInfo {
                buffer: b,
                offset: 0,
                range: WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<WriteDescriptorSet> = (0..n)
            .map(|i| WriteDescriptorSet {
                s_type: ST_WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: dset,
                dst_binding: i,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &infos[i as usize],
                p_texel_buffer_view: std::ptr::null(),
            })
            .collect();
        (self.vk.update_dsets)(self.vk.device, n, writes.as_ptr(), 0, std::ptr::null());
        dset
    }
}
