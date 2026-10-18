//! The projection kernel and its row-dot helper (port-map row 4).

use crate::format::quant;

use crate::engine::num::dequant::dequant_row_fixed;
use crate::engine::num::value::round_shift_32;
use crate::engine::EngineError;

/// A refused fixed-point projection (port-map row 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatvecError {
    /// The activation length is not the weight row width `n_in`.
    ShapeMismatch {
        /// Activation length supplied.
        x: usize,
        /// Declared row width.
        n_in: usize,
    },
    /// A dequantized weight row length does not match the activation length.
    RowLength {
        /// Output-row index, `0` for the standalone [`row_dot_fixed`] helper.
        row: usize,
        /// Dequantized weight row length.
        weights: usize,
        /// Activation length.
        n_in: usize,
    },
    /// The weight payload is shorter than `n_out` whole rows.
    ShortPayload {
        /// Bytes needed for `n_out` rows.
        need: usize,
        /// Bytes supplied.
        have: usize,
    },
    /// The row stride or payload byte count left `usize`.
    LengthOverflow,
    /// The weight quant family has no fixed-point row reader.
    UnsupportedQuant {
        /// ggml type id.
        ggml_type: u32,
    },
    /// A weight row's dequantizer refused a malformed or overflowing block.
    Dequant(EngineError),
    /// The `i128` dot accumulator left its range.
    AccumulatorOverflow,
    /// A rounded output does not fit `i64`.
    OutputOverflow,
}

impl From<EngineError> for MatvecError {
    fn from(e: EngineError) -> Self {
        MatvecError::Dequant(e)
    }
}

/// The row-dot helper of port-map row 4: `round(Σ_i w_i·x_i / 2^32)` for one Q32.32 weight row and
/// a Q32.32 activation vector.
///
/// Every product `w_i·x_i` is an `i64` square-scale quantity, so the sum is accumulated exactly in
/// `i128`; the single reduction is the value-path epilogue [`round_shift_32`], one round-to-nearest
/// `>> 32` (ties upward). There is no intermediate shift and no float on the executing path.
///
/// # Errors
///
/// Returns [`MatvecError::RowLength`] when `w_row` and `x` differ in length,
/// [`MatvecError::AccumulatorOverflow`] when the dot leaves `i128`, and
/// [`MatvecError::OutputOverflow`] when the rounded value does not fit `i64`. No value is clamped.
///
/// # Complexity
///
/// `O(n)` time in checked `i128` multiply-adds, `O(1)` extra space; `n` is the row width.
///
/// The accumulation loop carries a `bool` failure flag and stops on the first refusal, so no
/// `return` and no `?` appears inside the loop; the refusals, the `i128` arithmetic and the single
/// round-to-nearest epilogue are the same operations in the same order. The pinned Charon/Aeneas
/// fragment lowers an early return inside a loop to an unsupported control-flow pattern, so the
/// explicit flag form is the extractable one; the one `as i64` narrowing is range-checked against
/// `i64::MAX`/`MIN` immediately before it (`TryFrom` emits an unmodeled external).
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    reason = "the plain `i64`/`i128` comparisons and the `as i64` cast after an explicit range \
              check extract through Charon/Aeneas, while `i64::try_from` emits an unmodeled \
              external the fail-closed audit rejects"
)]
pub fn row_dot_fixed(w_row: &[i64], x: &[i64]) -> Result<i64, MatvecError> {
    if w_row.len() != x.len() {
        return Err(MatvecError::RowLength {
            row: 0,
            weights: w_row.len(),
            n_in: x.len(),
        });
    }
    let mut acc: i128 = 0;
    let mut i = 0usize;
    let mut acc_ok = true;
    while i < w_row.len() && acc_ok {
        let a = i128::from(w_row[i]);
        let b = i128::from(x[i]);
        match a.checked_mul(b) {
            Some(prod) => match acc.checked_add(prod) {
                Some(total) => acc = total,
                None => acc_ok = false,
            },
            None => acc_ok = false,
        }
        i += 1;
    }
    if !acc_ok {
        return Err(MatvecError::AccumulatorOverflow);
    }
    let rounded = round_shift_32(acc);
    if rounded > i64::MAX as i128 || rounded < i64::MIN as i128 {
        return Err(MatvecError::OutputOverflow);
    }
    Ok(rounded as i64)
}

/// The fixed-point projection kernel of port-map row 4: one Q32.32 weight row per output.
///
/// `raw` is the row-major payload of an `n_out × n_in` weight matrix of ggml type `ggml_type`, with
/// the fastest-varying dimension `n_in` (so output row `o` occupies the `row_bytes`-sized slice at
/// `o·row_bytes`). Each row is dequantized by [`dequant_row_fixed`] to Q32.32 and reduced against
/// `x` by [`row_dot_fixed`], giving `y[o] = round(Σ_i W[o,i]·x_i / 2^32)`. One kernel serves every
/// served projection shape — Q/K/V with the GQA head counts (`n_head·head_dim` and
/// `n_head_kv·head_dim` rows over `n_embd`), the FFN (`n_ff` rows) and the LM head (`vocab` rows) —
/// because the caller passes `n_out` and `n_in`; there is no float on the executing path.
///
/// # Errors
///
/// Returns [`MatvecError::UnsupportedQuant`] for a type with no fixed-point reader,
/// [`MatvecError::ShapeMismatch`] when `x.len() != n_in`, [`MatvecError::LengthOverflow`] when the
/// row stride or payload byte count leaves `usize`, [`MatvecError::ShortPayload`] when `raw` holds
/// fewer than `n_out` whole rows, [`MatvecError::Dequant`] when a block is malformed or a
/// dequantized product leaves `i64`, and [`MatvecError::AccumulatorOverflow`] or
/// [`MatvecError::OutputOverflow`] from the dot. No input is clamped.
///
/// # Complexity
///
/// `O(n_out·n_in)` time in checked integer multiply-adds plus one dequantization per weight
/// element; `O(n_in)` scratch space for the current row plus `O(n_out)` output. Both loops are
/// linear in their bounds and each element is visited once.
///
/// The row loop carries a `bool` failure flag and stops on the first refusal, so no `?` and no
/// early `return` appears inside it; the refusal is recorded in an `Option` and returned after the
/// loop. The row stride, the row slice, the dequantizer dispatch and the dot are the same
/// operations in the same order. The pinned Charon/Aeneas fragment rejects an early return inside
/// a loop, so the explicit flag form is the extractable one.
pub fn matvec_fixed(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    x: &[i64],
) -> Result<Vec<i64>, MatvecError> {
    let (qk, bb) =
        quant::block_geom(ggml_type).ok_or(MatvecError::UnsupportedQuant { ggml_type })?;
    if x.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: x.len(), n_in });
    }
    let blocks = n_in.div_ceil(qk);
    let row_bytes = blocks.checked_mul(bb).ok_or(MatvecError::LengthOverflow)?;
    let need = n_out
        .checked_mul(row_bytes)
        .ok_or(MatvecError::LengthOverflow)?;
    if raw.len() < need {
        return Err(MatvecError::ShortPayload {
            need,
            have: raw.len(),
        });
    }
    let mut out: Vec<i64> = Vec::with_capacity(n_out);
    let mut row_buf: Vec<i64> = Vec::with_capacity(n_in);
    let mut o = 0usize;
    let mut ok = true;
    let mut err: Option<MatvecError> = None;
    while o < n_out && ok {
        if let Some(start) = o.checked_mul(row_bytes) {
            if let Some(row) = raw.get(start..start + row_bytes) {
                row_buf.clear();
                match dequant_row_fixed(ggml_type, row, n_in, &mut row_buf) {
                    Ok(()) => {
                        if row_buf.len() == n_in {
                            match row_dot_fixed(&row_buf, x) {
                                Ok(v) => out.push(v),
                                Err(e) => {
                                    err = Some(e);
                                    ok = false;
                                }
                            }
                        } else {
                            err = Some(MatvecError::RowLength {
                                row: o,
                                weights: row_buf.len(),
                                n_in,
                            });
                            ok = false;
                        }
                    }
                    Err(e) => {
                        err = Some(MatvecError::Dequant(e));
                        ok = false;
                    }
                }
            } else {
                err = Some(MatvecError::ShortPayload {
                    need,
                    have: raw.len(),
                });
                ok = false;
            }
        } else {
            err = Some(MatvecError::LengthOverflow);
            ok = false;
        }
        o += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out)
}
