//! The fixed-point value path: the Q32.32 epilogue helpers, IEEE 754 binary32 as a bit pattern,
//! the `F32` load-boundary conversion and the GGML block dequantizers.

pub(super) mod dequant;
pub(super) mod f32_convert;
pub(crate) mod fp;
pub(super) mod value;
