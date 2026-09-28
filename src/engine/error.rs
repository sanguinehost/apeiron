//! The engine refusal type, isolated from the loader so a module that needs it does not reach the
//! loader's `std::io` and filesystem path.
//!
//! [`EngineError`] is the shared refusal of the loader, the whole-row dequantizers (`dequant`) and
//! the projection kernel (`matvec`). Its only inputs are the GGUF, quant and `F32` error enums, so
//! the Charon/Aeneas extraction can root `matvec_fixed`/`row_dot_fixed` by including this leaf
//! instead of `loader.rs`. The type, its variants, derives and trait impls are unchanged; the
//! loader re-exports it at its former path.

use crate::engine::f32_convert::F32Error;
use crate::format::gguf::GgufError;
use crate::format::quant::QuantError;

/// A refused engine operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// An OS read or open failed.
    Io(String),
    /// The GGUF layer refused.
    Gguf(GgufError),
    /// The quant layer refused.
    Quant(QuantError),
    /// A required metadata key is absent.
    MissingMetadata(String),
    /// A metadata value exists but cannot form a config field.
    BadMetadata(String),
    /// A cross-check between two metadata sources failed.
    ConfigMismatch(String),
    /// A named tensor is absent.
    MissingTensor(String),
    /// A tensor's directory extent leaves the file.
    TensorOutOfBounds {
        /// Tensor name.
        name: String,
        /// Absolute start offset.
        at: u64,
        /// Byte count claimed.
        n: u64,
        /// File length.
        file_len: u64,
    },
    /// A tensor's type has no geometry in `crate::format::quant`, so it cannot be read.
    UnknownQuantKind {
        /// Tensor name.
        name: String,
        /// ggml type id.
        ggml_type: u32,
    },
    /// A length did not fit the target integer width.
    LengthOverflow,
    /// The requested token id is outside the vocabulary.
    TokenOutOfRange {
        /// Requested id.
        id: u32,
        /// Vocabulary size.
        vocab: usize,
    },
    /// An embedding row would leave the tensor's checked extent.
    RowOutOfBounds {
        /// Absolute start offset.
        at: u64,
        /// Byte count claimed.
        n: u64,
        /// Tensor extent end.
        end: u64,
    },
    /// An `f16` scale field was `Inf` or `NaN`.
    NonFiniteScale {
        /// The raw bit pattern.
        bits: u16,
    },
    /// A fixed-point product did not fit `i64`; unreachable for a finite `f16` scale.
    FixedOverflow,
    /// The embedding tensor's quant family has no fixed-point reader.
    UnsupportedEmbeddingQuant {
        /// ggml type id.
        ggml_type: u32,
    },
    /// A decode `F32` gain or bias vector refused conversion to Q32.32.
    F32(F32Error),
    /// A decode gain/bias tensor is present but not the required `[len]` `F32` shape.
    BadVector {
        /// Tensor name.
        name: String,
        /// Static description of the refusal.
        reason: &'static str,
    },
}

impl core::fmt::Display for EngineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EngineError::Io(e) => write!(f, "io: {e}"),
            EngineError::Gguf(e) => write!(f, "gguf: {e}"),
            EngineError::Quant(e) => write!(f, "quant: {e}"),
            EngineError::MissingMetadata(k) => write!(f, "missing metadata {k}"),
            EngineError::BadMetadata(k) => write!(f, "bad metadata value for {k}"),
            EngineError::ConfigMismatch(d) => write!(f, "config mismatch: {d}"),
            EngineError::MissingTensor(n) => write!(f, "no tensor {n}"),
            EngineError::TensorOutOfBounds {
                name,
                at,
                n,
                file_len,
            } => write!(f, "tensor {name} [{at}, {at}+{n}) past EOF {file_len}"),
            EngineError::UnknownQuantKind { name, ggml_type } => {
                write!(f, "tensor {name} has unknown ggml type {ggml_type}")
            }
            EngineError::LengthOverflow => write!(f, "length overflow"),
            EngineError::TokenOutOfRange { id, vocab } => {
                write!(f, "token id {id} >= vocab {vocab}")
            }
            EngineError::RowOutOfBounds { at, n, end } => {
                write!(f, "embedding row [{at}, {at}+{n}) past extent end {end}")
            }
            EngineError::NonFiniteScale { bits } => {
                write!(f, "non-finite f16 scale 0x{bits:04x}")
            }
            EngineError::FixedOverflow => write!(f, "fixed-point product overflow"),
            EngineError::UnsupportedEmbeddingQuant { ggml_type } => {
                write!(
                    f,
                    "no fixed-point embedding reader for ggml type {ggml_type}"
                )
            }
            EngineError::F32(e) => write!(f, "gain/bias: {e}"),
            EngineError::BadVector { name, reason } => {
                write!(
                    f,
                    "gain/bias tensor {name} is not a valid F32 vector ({reason})"
                )
            }
        }
    }
}

impl From<F32Error> for EngineError {
    fn from(e: F32Error) -> Self {
        EngineError::F32(e)
    }
}

impl std::error::Error for EngineError {}

impl From<GgufError> for EngineError {
    fn from(e: GgufError) -> Self {
        EngineError::Gguf(e)
    }
}

impl From<QuantError> for EngineError {
    fn from(e: QuantError) -> Self {
        EngineError::Quant(e)
    }
}
