//! The LM-head projection refusals and the pure projection leaf (port-map row 7).
//!
//! [`lm_head_fixed`] is the projection half of `crate::engine::Engine::lm_head_fixed` with the
//! file read and the tensor-directory checks removed, so the Charon/Aeneas extraction can root the
//! real projection body without pulling `Engine`, `loader` or `std::io`.
//! `Engine::lm_head_fixed` validates the stored head, sizes and reads its payload, then delegates
//! here; [`head_payload_len`] is the read adapter's sizing half and is not on the extracted path.

use crate::format::quant;

use super::matvec::matvec_fixed;
use super::{EngineError, MatvecError};

/// A refused LM-head projection (port-map row 7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LmHeadError {
    /// The activation width is not the residual width `n_embd`.
    ShapeMismatch {
        /// Activation length supplied.
        x: usize,
        /// Configured residual width.
        n_embd: usize,
    },
    /// The head tensor's stored shape is not `[n_embd, vocab]`.
    BadHeadShape {
        /// The stored dimensions, fastest-varying first.
        dims: Vec<u64>,
        /// Configured residual width.
        n_embd: usize,
        /// Configured vocabulary size.
        vocab: usize,
    },
    /// The head tensor's quant family has no fixed-point reader.
    UnsupportedQuant {
        /// ggml type id.
        ggml_type: u32,
    },
    /// The tensor extent holds fewer than `vocab` whole rows.
    ShortPayload {
        /// Bytes needed for `vocab` rows.
        need: u64,
        /// Bytes available in the tensor extent.
        have: u64,
    },
    /// The row stride or the whole-payload byte count left `usize`/`u64`.
    LengthOverflow,
    /// The underlying file read failed.
    Read(EngineError),
    /// The projection refused through [`MatvecError`].
    Matvec(MatvecError),
}

impl From<MatvecError> for LmHeadError {
    fn from(e: MatvecError) -> Self {
        LmHeadError::Matvec(e)
    }
}

/// The exact head payload byte count for stored dims `[n_embd, vocab]` and quant family
/// `ggml_type`: `vocab · ceil(n_embd / qk) · block_bytes`.
///
/// This is the read adapter's sizing half of port-map row 7: it checks the stored
/// `[n_embd, vocab]` shape, resolves the quant geometry and returns the byte count
/// `Engine::lm_head_fixed` reads. It is not on the extracted path; [`lm_head_fixed`] takes an
/// already-read payload.
///
/// # Errors
///
/// Returns [`LmHeadError::BadHeadShape`] when `dims` is not `[n_embd, vocab]`,
/// [`LmHeadError::UnsupportedQuant`] for a type with no fixed-point reader, and
/// [`LmHeadError::LengthOverflow`] when a row stride, byte count or `usize`/`u64` bound leaves its
/// range.
pub fn head_payload_len(
    ggml_type: u32,
    dims: &[u64],
    n_embd: usize,
    vocab: usize,
) -> Result<u64, LmHeadError> {
    let n_embd_u64 = u64::try_from(n_embd).map_err(|_| LmHeadError::LengthOverflow)?;
    let vocab_u64 = u64::try_from(vocab).map_err(|_| LmHeadError::LengthOverflow)?;
    if dims.len() != 2 || dims[0] != n_embd_u64 || dims[1] != vocab_u64 {
        return Err(LmHeadError::BadHeadShape {
            dims: dims.to_vec(),
            n_embd,
            vocab,
        });
    }
    let (qk, bb) =
        quant::block_geom(ggml_type).ok_or(LmHeadError::UnsupportedQuant { ggml_type })?;
    let blocks = n_embd.div_ceil(qk);
    let row_bytes = blocks.checked_mul(bb).ok_or(LmHeadError::LengthOverflow)?;
    let need = vocab
        .checked_mul(row_bytes)
        .ok_or(LmHeadError::LengthOverflow)?;
    u64::try_from(need).map_err(|_| LmHeadError::LengthOverflow)
}

/// The pure LM-head application (port-map row 7): project the residual activation `x` through one
/// `[n_embd, vocab]` head payload to one Q32.32 logit per vocabulary entry.
///
/// `raw` is the head's row-major quantized payload (`output.weight` when the directory carries it,
/// else the tied `token_embd.weight`) and `ggml_type` its quant family. This composes the real
/// [`matvec_fixed`] over `vocab` rows of `n_embd`, so the extracted body is the production
/// projection and its single `>> 32` epilogue, not a restatement. The caller is the unextracted
/// read adapter, which has already checked the stored dims and read the exact
/// [`head_payload_len`] payload.
///
/// # Errors
///
/// Returns [`LmHeadError::ShapeMismatch`] when `x.len() != n_embd`,
/// [`LmHeadError::UnsupportedQuant`] for a type with no fixed-point reader, and
/// [`LmHeadError::Matvec`] for a payload, quant, accumulator or range refusal from the projection.
pub fn lm_head_fixed(
    ggml_type: u32,
    raw: &[u8],
    vocab: usize,
    n_embd: usize,
    x: &[i64],
) -> Result<Vec<i64>, LmHeadError> {
    if x.len() != n_embd {
        return Err(LmHeadError::ShapeMismatch { x: x.len(), n_embd });
    }
    if quant::block_geom(ggml_type).is_none() {
        return Err(LmHeadError::UnsupportedQuant { ggml_type });
    }
    matvec_fixed(ggml_type, raw, vocab, n_embd, x).map_err(LmHeadError::Matvec)
}
