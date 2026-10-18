// A faithful port of fold's `modelfold::gpu` (the dlopen Vulkan runtime + dispatch). See the same
// reason at `crate::gpu::simt`.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's modelfold::gpu Vulkan runtime; lint profile is fold's, not \
              apeiron's"
)]

//! gpu — the GPU computational primitive: the runtime (device / buffer / pipeline / dispatch) plus
//! the `simt` kernel builders, the device-side parallel of the CPU `Table`.
//!
//! Two halves, matching the CPU side (`Coord` + `KeyStore` + `Merge` → one `Table`): the runtime
//! ([`runtime`]) is the device and its dispatch loop; the ops are the `_bytes` builders in
//! [`crate::gpu::simt`] (GEMM, reduction, elementwise, the sparse-attention knot), re-exported
//! here so `gpu::gemv_dp4a_bytes` reads as one namespace. Sovereign — the whole stack bottoms out
//! in [`vulkan::Vk`], a hand-rolled dlopen of `libvulkan.so.1`, zero crates. Behind the `gpu`
//! feature; the kernel builders themselves stay always-on in `simt`.
//!
//! A quantized GEMM, standalone (`y = W·x`, q4, one call) — see `examples/gemm.rs`:
//!
//! ```ignore
//! let gpu = Gpu::new()?;
//! let (w, x) = (gpu.upload(&weights_q4), gpu.upload(&input));
//! let y = gpu.device_buffer(m * 4);
//! let pipe = gpu.pipeline(&gemv_dp4a_bytes(m, k, 256, segs, Carrier::Q4), 5);
//! let ds = gpu.bind(&pipe, &[&w, &x, &y, &dm, &xs]);
//! gpu.run(&[Dispatch { pipe, dset: ds, groups: (m.div_ceil(256), segs) }]);
//! ```

pub mod nn;
pub mod simt;
pub mod spirv;

#[cfg(feature = "gpu")]
pub mod decode;
#[cfg(feature = "gpu")]
pub mod runtime;
#[cfg(feature = "gpu")]
pub mod vulkan;

#[cfg(feature = "gpu")]
pub use runtime::{Buffer, DescriptorSet, Dispatch, Gpu, Pipeline};
#[cfg(feature = "gpu")]
pub use vulkan::*;
