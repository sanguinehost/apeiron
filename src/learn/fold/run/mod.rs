//! Conversions between the engine-loaded `Q32.32` [`crate::learn::grad::model::Weights`] and the
//! student `GGUF` bytes ([`convert::student_gguf_bytes`]).

/// Conversions between the engine-loaded `Q32.32` weights and binary16/binary32 `GGUF` payloads.
pub mod convert;
