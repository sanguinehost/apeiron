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

/// A tensor of the GGUF directory named by layer and part, the key the decode path looks up with.
///
/// `part` indexes [`TENSOR_PART_BYTES`]: parts below [`N_BLK_PARTS`] render as
/// `blk.{layer}.{part text}`; the rest are top-level tensors and ignore `layer`. A key is `Copy`
/// and holds no text, so a refusal that names a tensor is built without allocating on the
/// extracted path; [`core::fmt::Display`] renders the directory name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorKey {
    /// Block (layer) index; ignored for a top-level part.
    pub layer: u32,
    /// Index into [`TENSOR_PART_BYTES`].
    pub part: u8,
}

/// `blk.{layer}.attn_q.weight` .. `blk.{layer}.nextn.eh_proj.weight` occupy parts `0..N_LAYER_PARTS`
/// (the order of `LayerIndex`).
pub const N_LAYER_PARTS: usize = 13;
/// Parts below this render as `blk.{layer}.{text}`; parts from it are top-level names.
pub const N_BLK_PARTS: usize = 22;
/// Number of named tensor parts.
pub const N_TENSOR_PARTS: usize = 25;
/// Part: `blk.{layer}.ffn_norm.weight`.
pub const PART_FFN_NORM: u8 = 13;
/// Part: `blk.{layer}.attn_norm.weight`.
pub const PART_ATTN_NORM: u8 = 14;
/// Part: `blk.{layer}.post_attention_norm.weight`.
pub const PART_POST_ATTENTION_NORM: u8 = 15;
/// Part: `blk.{layer}.nextn.enorm.weight`.
pub const PART_NEXTN_ENORM: u8 = 16;
/// Part: `blk.{layer}.nextn.hnorm.weight`.
pub const PART_NEXTN_HNORM: u8 = 17;
/// Part: `blk.{layer}.ssm_conv1d.weight`.
pub const PART_SSM_CONV1D: u8 = 18;
/// Part: `blk.{layer}.ssm_dt.bias`.
pub const PART_SSM_DT_BIAS: u8 = 19;
/// Part: `blk.{layer}.ssm_a`.
pub const PART_SSM_A: u8 = 20;
/// Part: `blk.{layer}.ssm_norm.weight`.
pub const PART_SSM_NORM: u8 = 21;
/// Part: `token_embd.weight`.
pub const PART_TOKEN_EMBD: u8 = 22;
/// Part: `output_norm.weight`.
pub const PART_OUTPUT_NORM: u8 = 23;
/// Part: `output.weight`.
pub const PART_OUTPUT: u8 = 24;

/// The text of each part, NUL-padded to 32 bytes; the first [`TENSOR_PART_LEN`] bytes are the text.
/// A by-value byte-array table, so the extraction needs no `&str`.
pub const TENSOR_PART_BYTES: [[u8; 32]; N_TENSOR_PARTS] = [
    *b"attn_q.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // attn_q.weight
    *b"attn_k.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // attn_k.weight
    *b"attn_v.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // attn_v.weight
    *b"attn_output.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0",      // attn_output.weight
    *b"ffn_gate.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",   // ffn_gate.weight
    *b"ffn_up.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // ffn_up.weight
    *b"ffn_down.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",   // ffn_down.weight
    *b"attn_qkv.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",   // attn_qkv.weight
    *b"attn_gate.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",    // attn_gate.weight
    *b"ssm_beta.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",   // ssm_beta.weight
    *b"ssm_alpha.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",    // ssm_alpha.weight
    *b"ssm_out.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",  // ssm_out.weight
    *b"nextn.eh_proj.weight\0\0\0\0\0\0\0\0\0\0\0\0",        // nextn.eh_proj.weight
    *b"ffn_norm.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",   // ffn_norm.weight
    *b"attn_norm.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",    // attn_norm.weight
    *b"post_attention_norm.weight\0\0\0\0\0\0",              // post_attention_norm.weight
    *b"nextn.enorm.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0",      // nextn.enorm.weight
    *b"nextn.hnorm.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0",      // nextn.hnorm.weight
    *b"ssm_conv1d.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",     // ssm_conv1d.weight
    *b"ssm_dt.bias\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // ssm_dt.bias
    *b"ssm_a\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // ssm_a
    *b"ssm_norm.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",   // ssm_norm.weight
    *b"token_embd.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",     // token_embd.weight
    *b"output_norm.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0",      // output_norm.weight
    *b"output.weight\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // output.weight
];

/// The byte length of each [`TENSOR_PART_BYTES`] text.
pub const TENSOR_PART_LEN: [usize; N_TENSOR_PARTS] = [
    13, 13, 13, 18, 15, 13, 15, 15, 16, 15, 16, 14, 20, 15, 16, 26, 18, 18, 17, 11, 5, 15, 17, 18,
    13,
];

impl TensorKey {
    /// The key of `blk.{layer}` part `part` (an index below [`N_BLK_PARTS`]).
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "guarded: a layer past u32::MAX saturates to a key no directory name matches"
    )]
    pub fn blk(layer: usize, part: u8) -> Self {
        let l = if layer > u32::MAX as usize {
            u32::MAX
        } else {
            layer as u32
        };
        TensorKey { layer: l, part }
    }

    /// The key of top-level part `part`.
    #[must_use]
    pub fn top(part: u8) -> Self {
        TensorKey { layer: 0, part }
    }
}

impl core::fmt::Display for TensorKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let p = usize::from(self.part);
        let (Some(text), Some(&len)) = (TENSOR_PART_BYTES.get(p), TENSOR_PART_LEN.get(p)) else {
            return write!(f, "?");
        };
        let text = core::str::from_utf8(&text[..len]).unwrap_or("?");
        if p < N_BLK_PARTS {
            write!(f, "blk.{}.{text}", self.layer)
        } else {
            write!(f, "{text}")
        }
    }
}

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
    MissingTensor(TensorKey),
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
        reason: String,
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

// Defined here, not in the opaque `trace` module: extracted code constructs it, and Aeneas has no field list for a type of an opaque module.
/// A traced phase. The leaf phases partition a pass; [`Phase::Pass`] and [`Phase::Mtp`] enclose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Token embedding lookup and dequantization.
    Embed,
    /// RMS norms (pre-mixer, pre-FFN, output).
    Norm,
    /// Residual additions.
    Residual,
    /// Softmax-layer Q/K/V (and gate) projections.
    AttnProjIn,
    /// Bias, gate split, q/k norm, `RoPE`, cache append and the attention heads.
    AttnCore,
    /// Softmax-layer output projection.
    AttnProjOut,
    /// Affine-layer input projections (qkv, gate, beta, alpha).
    AffineProjIn,
    /// Affine-layer causal convolution.
    AffineConv,
    /// Affine-layer per-head state updates and reads.
    AffineHeads,
    /// Affine-layer output projection.
    AffineProjOut,
    /// The `SwiGLU` FFN: quantization, gate, up, hidden, down.
    Ffn,
    /// The output head (dense or bit-plane) and its argmax.
    Head,
    /// The MTP draft block (encloses its attention and FFN leaf phases).
    Mtp,
    /// Speculative rollback: cache truncation and affine restore.
    Restore,
    /// One `compose_cols` pass (encloses the leaf phases).
    Pass,
    /// Overlay: the dispatching thread's wait for the pool's other participants after finishing
    /// its own share, one span per dispatch (`calls` counts dispatches, `cols` their tasks). It
    /// overlaps the leaf phases and is excluded from the partition.
    PoolWait,
}
