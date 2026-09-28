//! Fixed-point row dequantizers for the six low-bit GGUF faces (port-map row 1).
//!
//! `Q1_0`, `Q2_0`, `PQ2_0`, `TQ2_0`, `TQ1_0` and `PTQ1_0` each hold one `f16` scale and a small
//! integer code per weight. Every value is an integer multiple of the Q32.32 scale, so the product
//! is checked `i64` arithmetic with no rounding; the `f16` scale decodes exactly through
//! [`super::f16_to_fixed`]. The base-3 trit faces (`TQ1_0`, `PTQ1_0`) recover a trit from a packed
//! byte with the fork's rule in `u16` arithmetic and an explicit `& 0xFF`, so no wrapping multiply
//! appears and Aeneas extracts the body.
//!
//! Layouts and reference arithmetic: `prism-llama.cpp` `ggml/src/ggml-common.h:185-316` and
//! `ggml/src/ggml-quants.c` `dequantize_row_q1_0:454`, `dequantize_row_q2_0:474`,
//! `dequantize_row_pq2_0:494`, `dequantize_row_ptq1_0:2255`, `dequantize_row_tq1_0:2599` and
//! `dequantize_row_tq2_0` immediately after it.

use crate::engine::EngineError;
use crate::format::quant::{
    PQ2_0_BLOCK_BYTES, PQ2_0_QK, PTQ1_0_BLOCK_BYTES, PTQ1_0_QK, Q1_0_BLOCK_BYTES, Q1_0_QK, Q2_0_QK,
    TQ1_0_BLOCK_BYTES, TQ1_0_QK, TQ2_0_BLOCK_BYTES, TQ2_0_QK,
};

use super::{block_slice, mul, read_scale, sub};

/// Powers of three for trit plane `n` of a packed byte (`3^0 .. 3^5`).
const POW3: [u16; 6] = [1, 3, 9, 27, 81, 243];

/// Stage widths over `qs` for `TQ1_0` (`ggml-quants.c:2601-2622`): 32 bytes, then 16.
const TQ1_0_STAGES: [usize; 2] = [32, 16];

/// Stage widths over `qs` for `PTQ1_0` (`ggml-quants.c:2217-2233`): 32, 16, then 8. The 32-byte
/// stage does not fit the 24-byte `qs` and is skipped, leaving the effective 16-then-8 order.
const PTQ1_0_STAGES: [usize; 3] = [32, 16, 8];

/// Recover trit `n` (0-based, most significant first) of a packed base-3 byte.
///
/// The fork multiplies the byte by `3^n` in `u8` arithmetic and then takes `(q * 3) >> 8`. The
/// `u16` product with the explicit `& 0xFF` mask reproduces the `u8` wrap without a `wrapping_*`
/// call.
fn trit(b: u8, n: usize) -> i64 {
    i64::from((((u16::from(b) * POW3[n]) & 0xFF) * 3) >> 8)
}

/// `Q1_0`/`Q2_0` block `block`'s `f16` scale (Q32.32) and 16 code bytes: `d`@0, `qs`@2.
///
/// Both types share the 18-byte block; only the code width and the block length differ.
fn q1_q2_block(raw: &[u8], block: usize) -> Result<(i64, [u8; 16]), EngineError> {
    let blk = block_slice(raw, block, Q1_0_BLOCK_BYTES)?;
    let scale = read_scale(blk, 0)?;
    let mut qs = [0u8; 16];
    let mut k = 0usize;
    while k < 16 {
        qs[k] = blk[2 + k];
        k += 1;
    }
    Ok((scale, qs))
}

/// `PQ2_0` block `block`'s `f16` scale (Q32.32) and 32 code bytes: `d`@0, `qs`@2.
fn pq2_0_block(raw: &[u8], block: usize) -> Result<(i64, [u8; 32]), EngineError> {
    let blk = block_slice(raw, block, PQ2_0_BLOCK_BYTES)?;
    let scale = read_scale(blk, 0)?;
    let mut qs = [0u8; 32];
    let mut k = 0usize;
    while k < 32 {
        qs[k] = blk[2 + k];
        k += 1;
    }
    Ok((scale, qs))
}

/// `TQ2_0` block `block`'s `f16` scale (Q32.32) and 64 code bytes: `qs`@0, `d`@64.
fn tq2_0_block(raw: &[u8], block: usize) -> Result<(i64, [u8; 64]), EngineError> {
    let blk = block_slice(raw, block, TQ2_0_BLOCK_BYTES)?;
    let scale = read_scale(blk, 64)?;
    let mut qs = [0u8; 64];
    let mut k = 0usize;
    while k < 64 {
        qs[k] = blk[k];
        k += 1;
    }
    Ok((scale, qs))
}

/// `TQ1_0` block `block`'s `f16` scale (Q32.32), 48 `qs` bytes and 4 `qh` bytes: `qs`@0, `qh`@48,
/// `d`@52.
fn tq1_0_block(raw: &[u8], block: usize) -> Result<(i64, [u8; 48], [u8; 4]), EngineError> {
    let blk = block_slice(raw, block, TQ1_0_BLOCK_BYTES)?;
    let scale = read_scale(blk, 52)?;
    let mut qs = [0u8; 48];
    let mut k = 0usize;
    while k < 48 {
        qs[k] = blk[k];
        k += 1;
    }
    let mut qh = [0u8; 4];
    let mut h = 0usize;
    while h < 4 {
        qh[h] = blk[48 + h];
        h += 1;
    }
    Ok((scale, qs, qh))
}

/// `PTQ1_0` block `block`'s `f16` scale (Q32.32), 24 `qs` bytes and 2 `qh` bytes: `qs`@0, `qh`@24,
/// `d`@26.
fn ptq1_0_block(raw: &[u8], block: usize) -> Result<(i64, [u8; 24], [u8; 2]), EngineError> {
    let blk = block_slice(raw, block, PTQ1_0_BLOCK_BYTES)?;
    let scale = read_scale(blk, 26)?;
    let mut qs = [0u8; 24];
    let mut k = 0usize;
    while k < 24 {
        qs[k] = blk[k];
        k += 1;
    }
    let mut qh = [0u8; 2];
    let mut h = 0usize;
    while h < 2 {
        qh[h] = blk[24 + h];
        h += 1;
    }
    Ok((scale, qs, qh))
}

/// `Q1_0` row: `w = bit ? d : -d`, 128 signed bits per `f16` scale.
///
/// Temporal: `ceil(n / 128)` blocks, each a 16-byte scan and a lane pass, so `O(n)` lane steps
/// counted in checked operations. Spatial: `O(1)` beyond `out` (a 16-byte code image). The block
/// and lane loops carry a `bool` failure flag and stop on the first refusal, so no `?` and no early
/// `return` appears inside either loop.
pub(super) fn dequant_q1_0_row(
    raw: &[u8],
    n_elems: usize,
    out: &mut Vec<i64>,
) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(Q1_0_QK);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match q1_q2_block(raw, block) {
            Ok((scale, qs)) => {
                let neg = match sub(0, scale) {
                    Ok(v) => v,
                    Err(e) => {
                        err = Some(e);
                        ok = false;
                        0
                    }
                };
                let take = Q1_0_QK.min(n_elems - block * Q1_0_QK);
                let mut lane = 0usize;
                while lane < take && ok {
                    let bit = (qs[lane / 8] >> (lane % 8)) & 1;
                    out.push(if bit == 1 { scale } else { neg });
                    lane += 1;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `Q2_0` row: `w = ((q >> 2i) & 3 - 1) * d`, 64 four-level codes per `f16` scale.
///
/// Temporal: `ceil(n / 64)` blocks, each a 16-byte scan and a lane pass, so `O(n)` lane steps.
/// Spatial: `O(1)` beyond `out` (a 16-byte code image). The block and lane loops carry a `bool`
/// failure flag and stop on the first refusal, so no `?` and no early `return` appears inside either
/// loop.
pub(super) fn dequant_q2_0_row(
    raw: &[u8],
    n_elems: usize,
    out: &mut Vec<i64>,
) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(Q2_0_QK);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match q1_q2_block(raw, block) {
            Ok((scale, qs)) => {
                let take = Q2_0_QK.min(n_elems - block * Q2_0_QK);
                let mut lane = 0usize;
                while lane < take && ok {
                    let code = (qs[lane / 4] >> (2 * (lane % 4))) & 3;
                    match mul(scale, i64::from(code) - 1) {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                        }
                    }
                    lane += 1;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `PQ2_0` row: `Q2_0` codes over 128 weights per `f16` scale.
///
/// Temporal: `ceil(n / 128)` blocks, each a 32-byte scan and a lane pass, so `O(n)` lane steps.
/// Spatial: `O(1)` beyond `out` (a 32-byte code image). The block and lane loops carry a `bool`
/// failure flag and stop on the first refusal, so no `?` and no early `return` appears inside either
/// loop.
pub(super) fn dequant_pq2_0_row(
    raw: &[u8],
    n_elems: usize,
    out: &mut Vec<i64>,
) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(PQ2_0_QK);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match pq2_0_block(raw, block) {
            Ok((scale, qs)) => {
                let take = PQ2_0_QK.min(n_elems - block * PQ2_0_QK);
                let mut lane = 0usize;
                while lane < take && ok {
                    let code = (qs[lane / 4] >> (2 * (lane % 4))) & 3;
                    match mul(scale, i64::from(code) - 1) {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                        }
                    }
                    lane += 1;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `TQ2_0` row: `w = (code - 1) * d`, 256 four-level codes per `f16` scale.
///
/// Output order is the fork nesting: for each 32-byte `qs` group, plane `l < 4`, lane `m < 32`, so
/// the first 128 weights come from `qs[0..32]` and the next 128 from `qs[32..64]`.
///
/// Temporal: `ceil(n / 256)` blocks, each four planes over 64 bytes, so `O(n)` lane steps. Spatial:
/// `O(1)` beyond `out` (a 64-byte code image). The loops carry a `bool` failure flag and stop on the
/// first refusal, so no `?` and no early `return` appears inside any loop.
pub(super) fn dequant_tq2_0_row(
    raw: &[u8],
    n_elems: usize,
    out: &mut Vec<i64>,
) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(TQ2_0_QK);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match tq2_0_block(raw, block) {
            Ok((scale, qs)) => {
                let take = TQ2_0_QK.min(n_elems - block * TQ2_0_QK);
                let mut pos = 0usize;
                let mut group = 0usize;
                while group < 64 && ok {
                    let mut plane = 0usize;
                    while plane < 4 && ok {
                        let mut lane = 0usize;
                        while lane < 32 && ok {
                            let code = (qs[group + lane] >> (2 * plane)) & 3;
                            match mul(scale, i64::from(code) - 1) {
                                Ok(v) => {
                                    if pos < take {
                                        out.push(v);
                                    }
                                    pos += 1;
                                }
                                Err(e) => {
                                    err = Some(e);
                                    ok = false;
                                }
                            }
                            lane += 1;
                        }
                        plane += 1;
                    }
                    group += 32;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `TQ1_0` row: 240 base-3 trits in `qs` (stages 32 then 16 bytes) and 16 in `qh`, times `d`.
///
/// Emission per stage is plane-major: for stage width `c`, trit plane `n < 5`, byte `m < c`, emit
/// `trit(qs[j + m], n)`; the `qh` tail is plane-major over 4 bytes. Output order is the fork's
/// (`ggml-quants.c:2599-2631`).
///
/// Temporal: `ceil(n / 256)` blocks, each 48 `qs` bytes over 5 planes plus 4 `qh` bytes over 4
/// planes, so `O(n)` lane steps. Spatial: `O(1)` beyond `out` (48 + 4 code bytes). The stage, plane
/// and byte loops carry a `bool` failure flag and stop on the first refusal, so no `?` and no early
/// `return` appears inside any loop.
pub(super) fn dequant_tq1_0_row(
    raw: &[u8],
    n_elems: usize,
    out: &mut Vec<i64>,
) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(TQ1_0_QK);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match tq1_0_block(raw, block) {
            Ok((scale, qs, qh)) => {
                let take = TQ1_0_QK.min(n_elems - block * TQ1_0_QK);
                let mut pos = 0usize;
                let mut base = 0usize;
                let mut stage = 0usize;
                while stage < 2 && ok {
                    let width = TQ1_0_STAGES[stage];
                    while base + width <= 48 && ok {
                        let mut plane = 0usize;
                        while plane < 5 && ok {
                            let mut lane = 0usize;
                            while lane < width && ok {
                                match mul(scale, trit(qs[base + lane], plane) - 1) {
                                    Ok(v) => {
                                        if pos < take {
                                            out.push(v);
                                        }
                                        pos += 1;
                                    }
                                    Err(e) => {
                                        err = Some(e);
                                        ok = false;
                                    }
                                }
                                lane += 1;
                            }
                            plane += 1;
                        }
                        base += width;
                    }
                    stage += 1;
                }
                let mut plane = 0usize;
                while plane < 4 && ok {
                    let mut tail = 0usize;
                    while tail < 4 && ok {
                        match mul(scale, trit(qh[tail], plane) - 1) {
                            Ok(v) => {
                                if pos < take {
                                    out.push(v);
                                }
                                pos += 1;
                            }
                            Err(e) => {
                                err = Some(e);
                                ok = false;
                            }
                        }
                        tail += 1;
                    }
                    plane += 1;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `PTQ1_0` row: the `TQ1_0` trit packing at 128 weights per `f16` scale.
///
/// Emission is the same stage rule over the 24-byte `qs` (effective stages 16 then 8) and the
/// 2-byte `qh`, with output order from `dequantize_row_ptq1_0` (`ggml-quants.c:2255-2283`).
///
/// Temporal: `ceil(n / 128)` blocks, each 24 `qs` bytes over 5 planes plus 2 `qh` bytes over 4
/// planes, so `O(n)` lane steps. Spatial: `O(1)` beyond `out` (24 + 2 code bytes). The stage, plane
/// and byte loops carry a `bool` failure flag and stop on the first refusal, so no `?` and no early
/// `return` appears inside any loop.
pub(super) fn dequant_ptq1_0_row(
    raw: &[u8],
    n_elems: usize,
    out: &mut Vec<i64>,
) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(PTQ1_0_QK);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match ptq1_0_block(raw, block) {
            Ok((scale, qs, qh)) => {
                let take = PTQ1_0_QK.min(n_elems - block * PTQ1_0_QK);
                let mut pos = 0usize;
                let mut base = 0usize;
                let mut stage = 0usize;
                while stage < 3 && ok {
                    let width = PTQ1_0_STAGES[stage];
                    while base + width <= 24 && ok {
                        let mut plane = 0usize;
                        while plane < 5 && ok {
                            let mut lane = 0usize;
                            while lane < width && ok {
                                match mul(scale, trit(qs[base + lane], plane) - 1) {
                                    Ok(v) => {
                                        if pos < take {
                                            out.push(v);
                                        }
                                        pos += 1;
                                    }
                                    Err(e) => {
                                        err = Some(e);
                                        ok = false;
                                    }
                                }
                                lane += 1;
                            }
                            plane += 1;
                        }
                        base += width;
                    }
                    stage += 1;
                }
                let mut plane = 0usize;
                while plane < 4 && ok {
                    let mut tail = 0usize;
                    while tail < 2 && ok {
                        match mul(scale, trit(qh[tail], plane) - 1) {
                            Ok(v) => {
                                if pos < take {
                                    out.push(v);
                                }
                                pos += 1;
                            }
                            Err(e) => {
                                err = Some(e);
                                ok = false;
                            }
                        }
                        tail += 1;
                    }
                    plane += 1;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::many_single_char_names,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::manual_div_ceil,
        reason = "the reference reimplementations mirror the fork's short C loop indices and its \
                  explicit ((q * 256) + 242) / 243 ceiling division; the f64 round-trip is the \
                  comparison target the spec names, and its values are small exact multiples of d"
    )]

    use super::*;
    use crate::format::quant::{
        QuantError, PQ2_0_BLOCK_BYTES, PTQ1_0_BLOCK_BYTES, Q1_0_BLOCK_BYTES, Q2_0_BLOCK_BYTES,
        TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES,
    };

    /// Q32.32 from a real value; exact for the small multiples of `d` used here.
    fn q32(v: f64) -> i64 {
        (v * 4_294_967_296.0).round() as i64
    }

    /// Independent reimplementation of the fork's `u8` trit extraction.
    fn fork_trit(b: u8, n: usize) -> i64 {
        let pow3 = [1u8, 3, 9, 27, 81, 243];
        let q = b.wrapping_mul(pow3[n]);
        i64::from((u16::from(q) * 3) >> 8)
    }

    /// `dequantize_row_q1_0` (`ggml-quants.c:454-471`) over `f64`.
    fn ref_q1_0(qs: &[u8], d: f64) -> Vec<i64> {
        let mut out = Vec::new();
        let mut j = 0usize;
        while j < 128 {
            let bit = (qs[j / 8] >> (j % 8)) & 1;
            out.push(q32(if bit == 1 { d } else { -d }));
            j += 1;
        }
        out
    }

    /// `dequantize_row_q2_0` (`ggml-quants.c:474-491`) over `f64`.
    fn ref_q2_0(qs: &[u8], d: f64) -> Vec<i64> {
        let mut out = Vec::new();
        let mut j = 0usize;
        while j < 64 {
            let q = (qs[j / 4] >> (2 * (j % 4))) & 3;
            out.push(q32((f64::from(q) - 1.0) * d));
            j += 1;
        }
        out
    }

    /// `dequantize_row_pq2_0` (`ggml-quants.c:494-511`) over `f64`.
    fn ref_pq2_0(qs: &[u8], d: f64) -> Vec<i64> {
        let mut out = Vec::new();
        let mut j = 0usize;
        while j < 128 {
            let q = (qs[j / 4] >> (2 * (j % 4))) & 3;
            out.push(q32((f64::from(q) - 1.0) * d));
            j += 1;
        }
        out
    }

    /// `dequantize_row_tq2_0` over `f64`.
    fn ref_tq2_0(qs: &[u8], d: f64) -> Vec<i64> {
        let mut out = Vec::new();
        let mut group = 0usize;
        while group < 64 {
            let mut plane = 0usize;
            while plane < 4 {
                let mut lane = 0usize;
                while lane < 32 {
                    let q = (qs[group + lane] >> (2 * plane)) & 3;
                    out.push(q32((f64::from(q) - 1.0) * d));
                    lane += 1;
                }
                plane += 1;
            }
            group += 32;
        }
        out
    }

    /// `dequantize_row_tq1_0` (`ggml-quants.c:2599-2631`) over `f64`: stage 32, stage 16, `qh`.
    fn ref_tq1_0(qs: &[u8], qh: &[u8], d: f64) -> Vec<i64> {
        let mut out = Vec::new();
        let mut base = 0usize;
        while base < 32 {
            let mut plane = 0usize;
            while plane < 5 {
                let mut lane = 0usize;
                while lane < 32 {
                    out.push(q32((fork_trit(qs[base + lane], plane) - 1) as f64 * d));
                    lane += 1;
                }
                plane += 1;
            }
            base += 32;
        }
        let mut base = 32usize;
        while base < 48 {
            let mut plane = 0usize;
            while plane < 5 {
                let mut lane = 0usize;
                while lane < 16 {
                    out.push(q32((fork_trit(qs[base + lane], plane) - 1) as f64 * d));
                    lane += 1;
                }
                plane += 1;
            }
            base += 16;
        }
        let mut plane = 0usize;
        while plane < 4 {
            let mut tail = 0usize;
            while tail < qh.len() {
                out.push(q32((fork_trit(qh[tail], plane) - 1) as f64 * d));
                tail += 1;
            }
            plane += 1;
        }
        out
    }

    /// `dequantize_row_ptq1_0` (`ggml-quants.c:2255-2283`) over `f64`: stage 16, stage 8, `qh`.
    fn ref_ptq1_0(qs: &[u8], qh: &[u8], d: f64) -> Vec<i64> {
        let mut out = Vec::new();
        let mut base = 0usize;
        while base < 16 {
            let mut plane = 0usize;
            while plane < 5 {
                let mut lane = 0usize;
                while lane < 16 {
                    out.push(q32((fork_trit(qs[base + lane], plane) - 1) as f64 * d));
                    lane += 1;
                }
                plane += 1;
            }
            base += 16;
        }
        let mut base = 16usize;
        while base < 24 {
            let mut plane = 0usize;
            while plane < 5 {
                let mut lane = 0usize;
                while lane < 8 {
                    out.push(q32((fork_trit(qs[base + lane], plane) - 1) as f64 * d));
                    lane += 1;
                }
                plane += 1;
            }
            base += 8;
        }
        let mut plane = 0usize;
        while plane < 4 {
            let mut tail = 0usize;
            while tail < qh.len() {
                out.push(q32((fork_trit(qh[tail], plane) - 1) as f64 * d));
                tail += 1;
            }
            plane += 1;
        }
        out
    }

    fn check(name: &str, got: &[i64], want: &[i64]) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(a, b, "{name}: index {i}");
        }
    }

    #[test]
    fn q1_0_row_matches_fork_reference() {
        let mut raw = vec![0u8; Q1_0_BLOCK_BYTES];
        raw[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        let mut j = 0usize;
        while j < 16 {
            raw[2 + j] = (j as u8).wrapping_mul(7).wrapping_add(1);
            j += 1;
        }
        let expected = ref_q1_0(&raw[2..18], 0.5);
        // both bit values appear
        assert!(expected.contains(&q32(0.5)) && expected.contains(&q32(-0.5)));
        let mut out = Vec::new();
        dequant_q1_0_row(&raw, 128, &mut out).expect("q1_0 row");
        check("q1_0", &out, &expected);
        let mut part = Vec::new();
        dequant_q1_0_row(&raw, 100, &mut part).expect("q1_0 partial");
        check("q1_0 partial", &part, &expected[..100]);
    }

    #[test]
    fn q2_0_row_codes_cover_every_value() {
        let mut raw = vec![0u8; Q2_0_BLOCK_BYTES];
        raw[..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
                                                            // byte 0b11100100 packs codes 0, 1, 2, 3 for four consecutive weights.
        let mut j = 0usize;
        while j < 16 {
            raw[2 + j] = 0b1110_0100;
            j += 1;
        }
        let expected = ref_q2_0(&raw[2..18], 1.0);
        assert_eq!(&expected[..4], &[q32(-1.0), q32(0.0), q32(1.0), q32(2.0)]);
        let mut out = Vec::new();
        dequant_q2_0_row(&raw, 64, &mut out).expect("q2_0 row");
        check("q2_0", &out, &expected);
        let mut part = Vec::new();
        dequant_q2_0_row(&raw, 40, &mut part).expect("q2_0 partial");
        check("q2_0 partial", &part, &expected[..40]);
    }

    #[test]
    fn pq2_0_row_codes_cover_every_value() {
        let mut raw = vec![0u8; PQ2_0_BLOCK_BYTES];
        raw[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        let mut j = 0usize;
        while j < 32 {
            raw[2 + j] = 0b1110_0100;
            j += 1;
        }
        let expected = ref_pq2_0(&raw[2..34], 0.5);
        assert_eq!(&expected[..4], &[q32(-0.5), q32(0.0), q32(0.5), q32(1.0)]);
        let mut out = Vec::new();
        dequant_pq2_0_row(&raw, 128, &mut out).expect("pq2_0 row");
        check("pq2_0", &out, &expected);
        let mut part = Vec::new();
        dequant_pq2_0_row(&raw, 100, &mut part).expect("pq2_0 partial");
        check("pq2_0 partial", &part, &expected[..100]);
    }

    #[test]
    fn tq2_0_row_matches_fork_reference() {
        let mut raw = vec![0u8; TQ2_0_BLOCK_BYTES];
        let mut j = 0usize;
        while j < 64 {
            raw[j] = (j as u8).wrapping_mul(11).wrapping_add(5);
            j += 1;
        }
        raw[64..66].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        let expected = ref_tq2_0(&raw[0..64], 1.0);
        let mut out = Vec::new();
        dequant_tq2_0_row(&raw, 256, &mut out).expect("tq2_0 row");
        check("tq2_0", &out, &expected);
        let mut part = Vec::new();
        dequant_tq2_0_row(&raw, 200, &mut part).expect("tq2_0 partial");
        check("tq2_0 partial", &part, &expected[..200]);
    }

    #[test]
    fn tq1_0_row_trit_order_matches_stages() {
        let mut raw = vec![0u8; TQ1_0_BLOCK_BYTES];
        let mut j = 0usize;
        while j < 48 {
            raw[j] = (j as u8).wrapping_mul(7).wrapping_add(1);
            j += 1;
        }
        raw[48..52].copy_from_slice(&[1u8, 2, 3, 4]);
        raw[52..54].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
                                                               // 5 planes over the 32-byte stage, then 5 over the 16-byte stage, then the 4 `qh` planes;
                                                               // the 32/16 boundary lands at output index 160.
        let expected = ref_tq1_0(&raw[0..48], &raw[48..52], 1.0);
        let mut out = Vec::new();
        dequant_tq1_0_row(&raw, 256, &mut out).expect("tq1_0 row");
        check("tq1_0", &out, &expected);
        let mut part = Vec::new();
        dequant_tq1_0_row(&raw, 200, &mut part).expect("tq1_0 partial");
        check("tq1_0 partial", &part, &expected[..200]);
    }

    #[test]
    fn ptq1_0_row_trit_order_matches_stages() {
        let mut raw = vec![0u8; PTQ1_0_BLOCK_BYTES];
        let mut j = 0usize;
        while j < 24 {
            raw[j] = (j as u8).wrapping_mul(7).wrapping_add(1);
            j += 1;
        }
        raw[24..26].copy_from_slice(&[1u8, 2]);
        raw[26..28].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
                                                               // effective stages 16 then 8; the 16/8 boundary lands at output index 80.
        let expected = ref_ptq1_0(&raw[0..24], &raw[24..26], 0.5);
        let mut out = Vec::new();
        dequant_ptq1_0_row(&raw, 128, &mut out).expect("ptq1_0 row");
        check("ptq1_0", &out, &expected);
        let mut part = Vec::new();
        dequant_ptq1_0_row(&raw, 100, &mut part).expect("ptq1_0 partial");
        check("ptq1_0 partial", &part, &expected[..100]);
    }

    #[test]
    fn trit_decodes_the_fork_quantizer_digits() {
        // quantize_row_tq1_0_ref, `ggml-quants.c:2508-2515`: q = sum t_n * 3^(4-n) and
        // byte = ((q * 256) + 242) / 243 (ceiling division, 243 = 3^5).
        let pow3 = [1u32, 3, 9, 27, 81];
        let mut q = 0u32;
        while q < 243 {
            let byte = u8::try_from((q * 256 + 242) / 243).expect("packed byte fits u8");
            let mut n = 0usize;
            while n < 5 {
                let digit = (q / pow3[4 - n]) % 3;
                assert_eq!(trit(byte, n), i64::from(digit), "q={q} n={n}");
                n += 1;
            }
            q += 1;
        }
        // The `qh` encoding shifts the leading trit up one place before the same division, so its
        // trits are the base-3 digits of (q * 3).
        let mut q = 0u32;
        while q < 81 {
            let byte = u8::try_from((q * 3 * 256 + 242) / 243).expect("packed byte fits u8");
            let mut n = 0usize;
            while n < 4 {
                let digit = (q / pow3[3 - n]) % 3;
                assert_eq!(trit(byte, n), i64::from(digit), "qh q={q} n={n}");
                n += 1;
            }
            q += 1;
        }
    }

    #[test]
    fn short_payload_is_refused() {
        let mut out = Vec::new();
        assert_eq!(
            dequant_q1_0_row(&[0u8; Q1_0_BLOCK_BYTES - 1], 1, &mut out),
            Err(EngineError::Quant(QuantError::Truncated {
                need: Q1_0_BLOCK_BYTES,
                have: Q1_0_BLOCK_BYTES - 1,
            }))
        );
        assert_eq!(
            dequant_q2_0_row(&[0u8; Q2_0_BLOCK_BYTES - 1], 1, &mut out),
            Err(EngineError::Quant(QuantError::Truncated {
                need: Q2_0_BLOCK_BYTES,
                have: Q2_0_BLOCK_BYTES - 1,
            }))
        );
        assert_eq!(
            dequant_pq2_0_row(&[0u8; PQ2_0_BLOCK_BYTES - 1], 1, &mut out),
            Err(EngineError::Quant(QuantError::Truncated {
                need: PQ2_0_BLOCK_BYTES,
                have: PQ2_0_BLOCK_BYTES - 1,
            }))
        );
        assert_eq!(
            dequant_tq2_0_row(&[0u8; TQ2_0_BLOCK_BYTES - 1], 1, &mut out),
            Err(EngineError::Quant(QuantError::Truncated {
                need: TQ2_0_BLOCK_BYTES,
                have: TQ2_0_BLOCK_BYTES - 1,
            }))
        );
        assert_eq!(
            dequant_tq1_0_row(&[0u8; TQ1_0_BLOCK_BYTES - 1], 1, &mut out),
            Err(EngineError::Quant(QuantError::Truncated {
                need: TQ1_0_BLOCK_BYTES,
                have: TQ1_0_BLOCK_BYTES - 1,
            }))
        );
        assert_eq!(
            dequant_ptq1_0_row(&[0u8; PTQ1_0_BLOCK_BYTES - 1], 1, &mut out),
            Err(EngineError::Quant(QuantError::Truncated {
                need: PTQ1_0_BLOCK_BYTES,
                have: PTQ1_0_BLOCK_BYTES - 1,
            }))
        );
        assert!(out.is_empty(), "no partial append on refusal");
    }

    #[test]
    fn non_finite_scale_is_refused() {
        // 0x7C00 is +Inf; `f16_to_fixed` refuses it before any code is read.
        let inf = 0x7C00u16;
        let mut out = Vec::new();

        let mut r = vec![0u8; Q1_0_BLOCK_BYTES];
        r[..2].copy_from_slice(&inf.to_le_bytes());
        assert_eq!(
            dequant_q1_0_row(&r, 1, &mut out),
            Err(EngineError::NonFiniteScale { bits: inf })
        );

        let mut r = vec![0u8; Q2_0_BLOCK_BYTES];
        r[..2].copy_from_slice(&inf.to_le_bytes());
        assert_eq!(
            dequant_q2_0_row(&r, 1, &mut out),
            Err(EngineError::NonFiniteScale { bits: inf })
        );

        let mut r = vec![0u8; PQ2_0_BLOCK_BYTES];
        r[..2].copy_from_slice(&inf.to_le_bytes());
        assert_eq!(
            dequant_pq2_0_row(&r, 1, &mut out),
            Err(EngineError::NonFiniteScale { bits: inf })
        );

        let mut r = vec![0u8; TQ2_0_BLOCK_BYTES];
        r[64..66].copy_from_slice(&inf.to_le_bytes());
        assert_eq!(
            dequant_tq2_0_row(&r, 1, &mut out),
            Err(EngineError::NonFiniteScale { bits: inf })
        );

        let mut r = vec![0u8; TQ1_0_BLOCK_BYTES];
        r[52..54].copy_from_slice(&inf.to_le_bytes());
        assert_eq!(
            dequant_tq1_0_row(&r, 1, &mut out),
            Err(EngineError::NonFiniteScale { bits: inf })
        );

        let mut r = vec![0u8; PTQ1_0_BLOCK_BYTES];
        r[26..28].copy_from_slice(&inf.to_le_bytes());
        assert_eq!(
            dequant_ptq1_0_row(&r, 1, &mut out),
            Err(EngineError::NonFiniteScale { bits: inf })
        );
    }
}
