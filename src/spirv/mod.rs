//! The SPIR-V DP4A fragment checker: the executing seam between the fold emitter's integer
//! `gemv_dp4a_module_dyn` binary and the proved `Spirv.Fragment`/`Spirv.Decode` fragment.
//!
//! [`check::check_fragment`] mirrors the `Spirv.Decode` D1-D4 obligations and the served opcode set
//! of `Spirv.Fragment.IsDp4aFragment`. It is the extraction root for `Spirv.Check` in wave 3.

pub mod check;

pub use check::{check_fragment, CheckError};
