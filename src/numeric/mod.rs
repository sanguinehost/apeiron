//! Fixed-point numeric kernels for the integer executing path.
//!
//! The executing path carries no float. Each kernel is a total Rust function over an `i64`
//! fixed-point representation that returns [`Result`], refusing outside its documented window
//! rather than wrapping or saturating. A kernel is also the extraction root for the
//! Charon/Aeneas bridge: a wrapper may fix a type, never restate the arithmetic.

pub mod fixed_exp;
pub mod fixed_ln;
pub mod fixed_rsqrt;
pub mod fixed_trig;
pub mod register;
