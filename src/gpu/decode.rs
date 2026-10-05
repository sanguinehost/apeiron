// Row: `inf-decode-subsume-C9XByuwivyxA4RMxX5KVlG` (INF-DECODE-SUBSUME-W2..W4): fold `bedrock/src/serve/gpu.rs` at `d497161f`.
//! GPU decode serving primitives ported from fold `bedrock/src/serve/gpu.rs`.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's bedrock/src/serve/gpu.rs; lint profile is fold's, not apeiron's"
)]

use crate::gpu::vulkan::*;
use crate::simt::{gemv_bytes, Carrier};
use crate::simt_nn::*;
use std::sync::OnceLock;

pub mod layer;
pub mod model;
pub mod pack;
pub mod state;
pub mod upload;
pub use model::GpuDecode;
pub use pack::*;

static LOCAL_CELL: OnceLock<u32> = OnceLock::new();
pub fn local_wg() -> u32 {
    *LOCAL_CELL.get_or_init(|| envu("DEC_LOCAL", 128))
}

/// **A dial's value, or its default when UNSET — never a default because the value did not parse.**
pub fn envu(name: &str, dflt: u32) -> u32 {
    match std::env::var(name) {
        Err(_) => dflt,
        Ok(v) => v.trim().parse().unwrap_or_else(|_| {
            panic!(
                "{name}={v:?} is not a u32. A dial that is SET but unreadable is a typo, not a \
request for the default ({dflt}) — refusing rather than serving something else."
            )
        }),
    }
}

/// Partial count for the two-stage GPU argmax.
pub const AMAX_P: u32 = 256;
/// Staging-buffer chunk for host→device upload.
pub const STAGE_CAP: usize = 256 << 20;

pub struct Gemv {
    pub pipe: Handle,
    pub layout: Handle,
    pub dset: Handle,
    pub groups: (u32, u32),
    pub stage2: Option<(Handle, Handle, Handle, (u32, u32))>,
    /// when fold=true and >1, `y` holds this many stage-1 partials (no stage2)
    pub segs: u32,
    pub bytes: u64,
    pub m: u32,
    pub k: u32,
}

/// Chart bytes for one `m×k` projection: `k/pack` code words + `k/32` scale words, per row.
pub fn chart_bytes(m: u32, k: u32, carrier: Carrier) -> u64 {
    chart_bytes_tl(m, k, carrier, false)
}

/// [`chart_bytes`] with the scale face declared.
pub fn chart_bytes_tl(m: u32, k: u32, carrier: Carrier, two_level: bool) -> u64 {
    GemvSpec::decode(carrier, false)
        .with_two_level(two_level)
        .chart_bytes(m, k)
}

/// Bytes ONE row-sweep of the SUPERBLOCK chart moves.
pub fn sb_chart_bytes(m: u32, k: u32) -> u64 {
    GemvSpec::decode_superblock(false).chart_bytes(m, k)
}

pub struct Step {
    pub pipe: Handle,
    pub layout: Handle,
    pub dset: Handle,
    pub groups: (u32, u32),
    pub name: &'static str,
    pub bytes: u64,
}

impl Step {
    pub fn with_bytes(mut self, b: u64) -> Self {
        self.bytes = b;
        self
    }
}

/// A recorded layer is a flat list of task nodes — a Step or a Gemv.
pub enum Node {
    Step(Step),
    Gemv(Gemv),
}

pub struct GpuCtx<'a> {
    pub vk: &'a Vk,
    pub stage_buf: Handle,
    pub stage_ptr: *mut u8,
    pub alloc: std::cell::Cell<u64>,
    pub charts: std::cell::Cell<u64>,
    pub tl_taken: std::cell::Cell<u64>,
    pub disk_bytes: std::cell::Cell<u64>,
}

impl<'a> GpuCtx<'a> {
    pub fn new(vk: &'a Vk) -> Result<(Self, Handle), String> {
        let (stage_buf, stage_mem) = unsafe {
            vk.buffer(
                STAGE_CAP as u64,
                BUFFER_USAGE_TRANSFER_SRC,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
        }
        .ok_or_else(|| "stage buffer alloc failed".to_string())?;
        let stage_ptr = unsafe { vk.map(stage_mem) }.cast::<u8>();
        Ok((
            Self {
                vk,
                stage_buf,
                stage_ptr,
                alloc: std::cell::Cell::new(0),
                charts: std::cell::Cell::new(0),
                tl_taken: std::cell::Cell::new(0),
                disk_bytes: std::cell::Cell::new(0),
            },
            stage_mem,
        ))
    }

    /// # Safety
    /// Vulkan handle creation; returned buffer outlives use by construction.
    pub unsafe fn dev_zeroed(&self, bytes_len: usize) -> Handle {
        self.up_u32(&vec![0u32; bytes_len.max(4).div_ceil(4)])
    }

    /// # Safety
    /// Vulkan handle creation.
    pub unsafe fn dev_maybe_zeroed(&self, bytes_len: usize) -> Handle {
        if envu("DEC_ZEROKV", 0) == 1 {
            self.dev_zeroed(bytes_len)
        } else {
            self.dev(bytes_len)
        }
    }

    pub unsafe fn dev(&self, bytes_len: usize) -> Handle {
        self.alloc.set(self.alloc.get() + bytes_len.max(4) as u64);
        self.vk
            .buffer(
                bytes_len.max(4) as u64,
                BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST,
                MEMORY_DEVICE_LOCAL,
            )
            .unwrap()
            .0
    }

    /// A HOST-VISIBLE storage buffer the device reads over PCIe.
    pub unsafe fn host(&self, bytes_len: usize) -> Handle {
        if std::env::var_os("DEC_KNOT_ANONKV").is_some() {
            match self.vk.buffer_from_anon_ptr(
                bytes_len.max(4) as u64,
                BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST,
            ) {
                Some((buf, _mem, _ptr)) => {
                    self.disk_bytes
                        .set(self.disk_bytes.get() + bytes_len as u64);
                    return buf;
                }
                None => {
                    eprintln!(
                        "gpu: DEC_KNOT_ANONKV — anonymous import refused; falling back to RAM"
                    )
                }
            }
        }
        if let Some(dir) = std::env::var_os("DEC_KNOT_DISKKV") {
            static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path =
                std::path::PathBuf::from(&dir).join(format!("kv_{}_{n}.bin", std::process::id()));
            match self.vk.buffer_from_host_ptr(
                &path,
                bytes_len.max(4) as u64,
                BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST,
            ) {
                Some((buf, _mem, _ptr)) => {
                    self.disk_bytes
                        .set(self.disk_bytes.get() + bytes_len as u64);
                    return buf;
                }
                None => eprintln!(
                    "gpu: DEC_KNOT_DISKKV — {} was refused (this driver imports only PINNABLE host \
memory; a file-backed mapping is evictable and cannot be DMA'd). Falling back to RAM. See \
modelfold/examples/host_ptr_probe.rs.",
                    path.display()
                ),
            }
        }
        self.vk
            .buffer(
                bytes_len.max(4) as u64,
                BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .unwrap()
            .0
    }

    /// # Safety
    /// `ptr` must be valid for `bytes_len` bytes.
    pub unsafe fn up(&self, ptr: *const u8, bytes_len: usize) -> Handle {
        let buf = self.dev(bytes_len);
        let mut done = 0usize;
        while done < bytes_len {
            let n = (bytes_len - done).min(STAGE_CAP);
            std::ptr::copy_nonoverlapping(ptr.add(done), self.stage_ptr, n);
            let cb = self.vk.command_buffer();
            self.vk.begin(cb);
            let region = BufferCopy {
                src_offset: 0,
                dst_offset: done as u64,
                size: n as u64,
            };
            (self.vk.copy_buffer)(cb, self.stage_buf, buf, 1, &region);
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
        buf
    }

    /// # Safety
    /// See [`GpuCtx::up`].
    pub unsafe fn up_f32(&self, v: &[f32]) -> Handle {
        self.up(v.as_ptr().cast(), v.len() * 4)
    }

    /// # Safety
    /// See [`GpuCtx::up`].
    pub unsafe fn up_u32(&self, v: &[u32]) -> Handle {
        self.up(v.as_ptr().cast(), v.len() * 4)
    }
}

/// The split-K knee: segments = next_pow2(32768 / m), clamped to the span.
pub fn segs(m: u32, span: u32) -> u32 {
    let tt = envu("DEC_SPLITK", 16_384);
    if m >= tt {
        1
    } else {
        ((tt / m).next_power_of_two()).min(span)
    }
}

/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_gemv(
    g: &GpuCtx,
    carrier: Carrier,
    w: Handle,
    dm: Handle,
    x: Handle,
    xs_sx: (Handle, Handle),
    y: Handle,
    partials: Handle,
    ones: Handle,
    m: u32,
    k: u32,
    fold: bool,
    scale: ScaleFace,
) -> Gemv {
    let spec = resolve_decode(g, GemvSpec::decode(carrier, fold), m, k, scale);
    let s = spec.reduce.segs();
    let out1 = if spec.reduce.needs_stage2() {
        partials
    } else {
        y
    };
    let (l1, d1) = if spec.carrier.is_chart() {
        g.vk.ssbo_set(&[w, x, out1, dm, xs_sx.0, xs_sx.1])
    } else {
        g.vk.ssbo_set(&[w, x, out1])
    };
    let pipe = g.vk.pipeline(
        &spec
            .emit(m, k, local_wg())
            .unwrap_or_else(|e| panic!("illegal gemv spec at {m}x{k}: {e}")),
        l1,
    );
    let stage2 = spec.reduce.needs_stage2().then(|| {
        let (l2, d2) = g.vk.ssbo_set(&[partials, ones, y]);
        let p2 =
            g.vk.pipeline(&gemv_bytes(m, s, local_wg(), 1, Carrier::F32), l2);
        (p2, l2, d2, (m.div_ceil(local_wg()), 1u32))
    });
    Gemv {
        pipe,
        layout: l1,
        dset: d1,
        groups: spec.groups(m, local_wg()),
        stage2,
        segs: if fold { s } else { 1 },
        bytes: spec.chart_bytes(m, k),
        m,
        k,
    }
}

/// **The decode mapping/split-K policy.**
pub fn resolve_decode(g: &GpuCtx, spec: GemvSpec, m: u32, k: u32, face: ScaleFace) -> GemvSpec {
    let mut spec = match face {
        ScaleFace::Superblock => {
            GemvSpec::decode_superblock(matches!(spec.reduce, Reduce::Fold(_))).fused(spec.fusion)
        }
        ScaleFace::TwoLevel => spec.with_two_level(true),
        ScaleFace::Dm32 => spec.with_two_level(false),
    };
    let span = if spec.carrier.is_chart() { k / 32 } else { k };
    let _over_limit = spec.over_shared_limit(k, g.vk.max_shared);
    let coop = spec.carrier.is_chart()
        && spec.scale != ScaleFace::Superblock
        && matches!(spec.fusion, Fusion::None)
        && envu("DEC_COOP", 0) == 1
        && m.div_ceil(local_wg()) <= envu("DEC_COOP_MAXWG", 256);
    if coop {
        spec.mapping = Mapping::Coop {
            rows_per_wg: envu("DEC_COOP_RPW", 4),
            local: envu("DEC_COOP_LOCAL", 64),
            subgroup: envu("DEC_COOP_SG", 32),
        };
        return spec.with_segs(1);
    }
    spec.with_segs(segs(m, span))
}

pub unsafe fn make_step(g: &GpuCtx, spv: &[u8], bufs: &[Handle], gx: u32, gy: u32) -> Step {
    make_step_named(g, spv, bufs, gx, gy, "step")
}

pub unsafe fn make_step_named(
    g: &GpuCtx,
    spv: &[u8],
    bufs: &[Handle],
    gx: u32,
    gy: u32,
    name: &'static str,
) -> Step {
    let (layout, dset) = g.vk.ssbo_set(bufs);
    Step {
        pipe: g.vk.pipeline(spv, layout),
        layout,
        dset,
        groups: (gx, gy),
        name,
        bytes: 0,
    }
}

static BAR_MODE: OnceLock<u32> = OnceLock::new();
pub fn bar_mode() -> u32 {
    *BAR_MODE.get_or_init(|| envu("DEC_BAR", 0))
}

pub unsafe fn step_barrier(vk: &Vk, cb: CommandBuffer) {
    match bar_mode() {
        1 => vk.barrier2(
            cb,
            STAGE2_COMPUTE,
            STAGE2_COMPUTE,
            ACCESS2_SHADER_WRITE,
            ACCESS2_SHADER_READ | ACCESS2_SHADER_WRITE,
        ),
        2 => vk.barrier2(
            cb,
            STAGE2_COMPUTE,
            STAGE2_COMPUTE,
            ACCESS2_SHADER_WRITE,
            ACCESS2_SHADER_READ,
        ),
        3 => vk.barrier2(cb, STAGE2_COMPUTE, STAGE2_COMPUTE, 0, 0),
        9 => {
            let _ = (vk, cb);
        }
        _ => vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_COMPUTE,
            ACCESS_SHADER_WRITE,
            ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
        ),
    }
}

pub unsafe fn rec_step(vk: &Vk, cb: CommandBuffer, s: &Step) -> u32 {
    (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, s.pipe);
    (vk.bind_sets)(
        cb,
        PIPELINE_BIND_POINT_COMPUTE,
        s.layout,
        0,
        1,
        &s.dset,
        0,
        std::ptr::null(),
    );
    (vk.dispatch)(cb, s.groups.0, s.groups.1, 1);
    step_barrier(vk, cb);
    1
}

pub unsafe fn rec_gemv(vk: &Vk, cb: CommandBuffer, gm: &Gemv) -> u32 {
    (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, gm.pipe);
    (vk.bind_sets)(
        cb,
        PIPELINE_BIND_POINT_COMPUTE,
        gm.layout,
        0,
        1,
        &gm.dset,
        0,
        std::ptr::null(),
    );
    (vk.dispatch)(cb, gm.groups.0, gm.groups.1, 1);
    step_barrier(vk, cb);
    if let Some((p2, l2, d2, g2)) = &gm.stage2 {
        (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, *p2);
        (vk.bind_sets)(
            cb,
            PIPELINE_BIND_POINT_COMPUTE,
            *l2,
            0,
            1,
            d2,
            0,
            std::ptr::null(),
        );
        (vk.dispatch)(cb, g2.0, g2.1, 1);
        step_barrier(vk, cb);
        return 2;
    }
    1
}
