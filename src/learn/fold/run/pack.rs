//! The fold pack: the student's `E8 ∪ O8` code planes as one file.
//!
//! The `1.18` b/w accounting (`learn::onebit_bits`, `TensorBits`) prices a plane as one codebook
//! index per 8 weights, one 4-bit level per `(row, group)` on a per-row grid, an 8-bit row index,
//! and 6- and 5-bit column fields with an `f16` pair each. This module writes exactly those fields
//! and reads them back, so the rate is counted from the bytes on disk rather than from a model:
//!
//! | field | bytes |
//! |---|---|
//! | codebook index | `o·n/8` (1 b/w) |
//! | row grid | one `(lo, sp)` word pair per distinct grid, one 1- or 2-byte index per row |
//! | row level | `o·n/128` 4-bit codes (0.031 b/w) |
//! | `c`, `d` | `n` 6- and 5-bit codes plus two `f16` endpoints |
//!
//! The row grid is a dictionary rather than one stored pair per row: `SinkPlane::rh` keeps only the
//! quantizer's decoded words, and `quant_r_hier`'s own `(r_lo, r_sp)` header is not in the
//! checkpoint, so the writer recovers a grid from the words themselves ([`fold_grid`]) and stores
//! each distinct pair once. `fold_grid` searches the span implied by the smallest gap between a
//! row's distinct words — a difference of two rounded grid points differs from the exact one by at
//! most one word, so the fitted span lies inside the searched window — and verifies that every word
//! of the row reproduces on the pair it returns. A row that no pair carries is refused rather than
//! stored lossily.
//!
//! Every field is verified to decode to the plane's exact word before it is reported, so the file
//! is lossless: the loaded plane recomposes the weight the fitted plane recomposes.
//!
//! # Open obligation
//!
//! The dictionary costs `16·distinct + 1..2·o` bytes; on the 4-layer `Qwen2.5-0.5B` transfer that
//! is 76,928 B a layer, and it puts the model rate at 1.204 b/w against the 1.18 bar.
//! `quant_r_hier` already stores the grid as an `8·o + 64` bit header (14,752 B a layer) and
//! `RHier` carries it (`r_lo`, `r_sp`, `u6`, `sp_idx`), but `fold_fit::PackOut` and
//! `nb_refine::Plane` drop it before the plane reaches this writer, so it cannot be read here yet.
//!
//! # Grade
//!
//! `tested`: `tests/learn_fold_pack.rs` on the crate's build path (the `896 × 896` round trip, the
//! rate, the code-bit control and the ignored 4-layer checkpoint run). Nothing here is `proved`.
//!
//! # Trust boundary
//!
//! Not covered: rustc/LLVM codegen, and the fit that produced the row-scale words. The record
//! binds its own fields to those words; it does not establish that the fit's grid is the best one.

#![allow(
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "`o`/`n`/`g` are the plane's output rows, input columns and group width, `p` the sink \
              plane being packed and `i`/`j` the row and group the fit indexes; the field names \
              `lo`/`sp`/`c`/`d` are the one-bit accounting's own"
)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::format::packing::{self, FoldPlane};
use crate::learn::fold::finish::plane::{CodedPlane, ONE};
use crate::learn::fold::fit::icm::{codebook, BLOCK, CODE_LEN};
use crate::learn::fold::fit::quant::r_level;
use crate::learn::fold::fit::steps::GROUP;
use crate::learn::fold::layer::tensor::{SinkPlane, TensorFit};
use crate::learn::fold::run::convert::f16_bits;
use crate::learn::fold::run::transfer::{io_err, write_atomic, TransferError};
use crate::learn::solve::q32::{div_rne_pos, round_f16, rshift_rne, Fx};

/// What a fold-pack write produced: the file length, the weights its planes carry, the plane count
/// and the dense corrector planes it does not encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldPackStats {
    /// The file length in bytes.
    pub bytes: u64,
    /// The weights the written planes carry (`Σ o·n`).
    pub weights: u64,
    /// The number of planes written.
    pub planes: usize,
    /// Corrector planes skipped: their dense product is not an `E8 ∪ O8` field, and the
    /// checkpoint drops the `u8` factors a compact record would need.
    pub skipped_dense: usize,
}

impl FoldPackStats {
    /// The file's rate in thousandths of a bit per weight: `8000·bytes / weights`, rounded to
    /// nearest, or `None` on an empty pack. `O(1)`.
    #[must_use]
    pub fn bits_per_weight_milli(&self) -> Option<u64> {
        if self.weights == 0 {
            return None;
        }
        Some((self.bytes * 8000 + self.weights / 2) / self.weights)
    }
}

/// The binary16 word of a `Q32.32` value: `round_f16` then the grid shift back, the map
/// `finish::plane`'s column quantizer rounds its endpoints through. `O(1)`.
///
/// # Errors
/// [`TransferError::F16`] when the value has no finite binary16 encoding.
fn f16_word(v: i64) -> Result<i64, TransferError> {
    let f =
        round_f16(Fx { m: v, e: -32 }).map_err(|e| TransferError::F16(format!("{v}: {e:?}")))?;
    if f.m == 0 {
        return Ok(0);
    }
    let shift = f.e + 32;
    let wide = if shift >= 0 {
        let s = u32::try_from(shift).map_err(|_| TransferError::F16(format!("{v}")))?;
        i128::from(f.m) << s
    } else {
        let s = u32::try_from(-shift).map_err(|_| TransferError::F16(format!("{v}")))?;
        rshift_rne(i128::from(f.m), s).ok_or_else(|| TransferError::F16(format!("{v}")))?
    };
    i64::try_from(wide).map_err(|_| TransferError::F16(format!("{v}")))
}

/// The binary16 bits of a `Q32.32` value. `O(1)`.
///
/// # Errors
/// [`TransferError::F16`] when the value has no finite binary16 encoding.
fn f16_word_bits(v: i64) -> Result<u16, TransferError> {
    let f =
        round_f16(Fx { m: v, e: -32 }).map_err(|e| TransferError::F16(format!("{v}: {e:?}")))?;
    f16_bits(&f).map_err(|e| TransferError::F16(format!("{v}: {e:?}")))
}

/// The 4-bit level of a row-scale word on the `(lo, sp)` grid, `0..=15`: the map [`r_level`]
/// decodes and `finish::plane`'s `snap_r` rounds, with the same round-to-nearest-even division.
/// `O(1)`.
fn row_level(v: i64, lo: i64, sp: i64) -> u8 {
    let k = div_rne_pos((i128::from(v) - i128::from(lo)) * 15, i128::from(sp)).clamp(0, 15);
    u8::try_from(k).unwrap_or_default()
}

/// A lower endpoint `lo` on which `sp` carries every word of `w`, whose minimum is `mn` and maximum
/// `mx`, or `None`.
///
/// The grid is `lo + round(k·sp/15)` for `k = 0..=15`; `mn` is one of its points, so `lo` is `mn`
/// less the level word of some `j`, the row must lie inside `[lo, lo + sp]` (which leaves `j = 0`
/// for a row that spans its grid), and every word must round to a level that decodes back to it.
/// `O(g)` divisions per surviving `j`.
fn grid_endpoint(w: &[i64], mn: i64, mx: i64, sp: i64) -> Option<i64> {
    if sp <= 0 {
        return None;
    }
    let mut level = [0_i64; 31];
    for (t, slot) in level.iter_mut().enumerate().skip(1) {
        let v = div_rne_pos(i128::try_from(t).ok()? * i128::from(sp), 15);
        *slot = i64::try_from(v).ok()?;
    }
    for j in 0..16_usize {
        let Some(lo) = mn.checked_sub(level[j]) else {
            continue;
        };
        if lo.checked_add(sp).is_none_or(|hi| hi < mx) {
            continue;
        }
        let fits = w.iter().all(|&v| {
            let k = row_level(v, lo, sp);
            r_level(lo, sp, k) == Some(v)
        });
        if fits {
            return Some(lo);
        }
    }
    None
}

/// The `(f16 endpoint bits, codes)` form of one column-scale field: `finish::plane`'s
/// `quant_c_q32` at `bits` (6 for `c`, 5 for `d`), every code verified to decode to the stored
/// word. `O(n)`.
///
/// # Errors
/// [`TransferError::Shape`] on an empty or off-grid field, or a code wider than its field.
fn col_codes(v: &[i64], bits: u32) -> Result<([u16; 2], Vec<u8>), TransferError> {
    let empty = || TransferError::Shape("fold pack: empty column scale".to_string());
    let mn = *v.iter().min().ok_or_else(empty)?;
    let mx = *v.iter().max().ok_or_else(empty)?;
    let c_min = f16_word(mn)?;
    let c_max = f16_word(mx)?;
    let span = (i128::from(c_max) - i128::from(c_min)).max(1);
    let levels = (1_i128 << bits) - 1;
    let mut q = Vec::with_capacity(v.len());
    for &x in v {
        let t = div_rne_pos((i128::from(x) - i128::from(c_min)) << 32, span);
        let code = rshift_rne(t.clamp(0, i128::from(ONE)) * levels, 32)
            .ok_or_else(|| TransferError::Shape("fold pack: column code".to_string()))?;
        let back = i128::from(c_min) + div_rne_pos(code * span, levels);
        if back != i128::from(x) || code > levels {
            return Err(TransferError::Shape(
                "fold pack: column scale off its grid".to_string(),
            ));
        }
        q.push(
            u8::try_from(code)
                .map_err(|_| TransferError::Shape("fold pack: column code width".to_string()))?,
        );
    }
    Ok(([f16_word_bits(c_min)?, f16_word_bits(c_max)?], q))
}

/// The inverse of [`col_codes`]: the stored words from the endpoint bits and the codes. `O(n)`.
///
/// # Errors
/// [`TransferError::Shape`] on a code wider than its field, or an endpoint with no finite word.
fn col_words(head: [u16; 2], q: &[u8], bits: u32) -> Result<Vec<i64>, TransferError> {
    let end = |b: u16| {
        packing::f16_bits_to_fixed(b)
            .map_err(|e| TransferError::Shape(format!("fold pack: column endpoint: {e:?}")))
    };
    let c_min = end(head[0])?;
    let c_max = end(head[1])?;
    let span = (i128::from(c_max) - i128::from(c_min)).max(1);
    let levels = (1_i128 << bits) - 1;
    let mut out = Vec::with_capacity(q.len());
    for &code in q {
        if i128::from(code) > levels {
            return Err(TransferError::Shape(
                "fold pack: column code width".to_string(),
            ));
        }
        let v = i128::from(c_min) + div_rne_pos(i128::from(code) * span, levels);
        out.push(
            i64::try_from(v)
                .map_err(|_| TransferError::Shape("fold pack: column word".to_string()))?,
        );
    }
    Ok(out)
}

/// A per-row `(lo, sp)` grid on which every stored row-scale word of `row` is a fixed point of the
/// 4-bit level map, or `None` when no candidate carries the row.
///
/// Candidates are `round(15·δ/m) + off` for `m = 1..=15` and `off` in `−16..=16`, where `δ` is the
/// smallest gap between two distinct words of the row: two grid points `m` levels apart differ
/// from `m·sp/15` by at most one word, so `sp` lies within `15/m + 1 ≤ 16` of the candidate.
/// `O(15·33·16·g)` word ops for a row of `g` groups.
fn fold_grid(row: &[i64]) -> Option<(i64, i64)> {
    let mn = *row.iter().min()?;
    let mx = *row.iter().max()?;
    if mn == mx {
        return Some((mn, 1));
    }
    let mut w = row.to_vec();
    w.sort_unstable();
    w.dedup();
    if w.len() < 2 {
        return Some((mn, 1));
    }
    let mut delta = i64::MAX;
    for i in 1..w.len() {
        delta = delta.min(w[i].checked_sub(w[i - 1])?);
    }
    for m in 1..=15_i128 {
        let base = div_rne_pos(15 * i128::from(delta), m);
        for off in -16_i128..=16 {
            let Ok(sp) = i64::try_from(base + off) else {
                continue;
            };
            if let Some(lo) = grid_endpoint(&w, mn, mx, sp) {
                return Some((lo, sp));
            }
        }
    }
    None
}

/// One fitted plane as its stored record, or `None` for a dense corrector plane, whose product is
/// not an `E8 ∪ O8` field. Every block, row level and column code is verified to decode to the
/// fit's own word: that check, not the caller, is what makes the record lossless.
/// `O(o·n + 495·16·o·g)`.
///
/// # Errors
/// [`TransferError::Shape`] on a shape disagreement, a block that is not a codebook row, a row no
/// 16-level grid carries, or a column scale that is not a fixed point of its own quantizer.
fn fold_plane_from_fit(
    p: &SinkPlane,
    cb: &HashMap<[i64; BLOCK], u8>,
) -> Result<Option<FoldPlane>, TransferError> {
    if p.dense.is_some() {
        return Ok(None);
    }
    let n = p.ch.len();
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(TransferError::Shape(format!("fold pack: plane n {n}")));
    }
    let o = match &p.rows {
        Some(rw) => rw.len(),
        None => p.s_rot.len() / n,
    };
    let g = n / GROUP;
    if o == 0 || p.s_rot.len() != o * n || p.rh.len() != o * g || p.dh.len() != n {
        return Err(TransferError::Shape(format!(
            "fold pack: plane {} words != {o}x{n}",
            p.s_rot.len()
        )));
    }
    let mut idx = Vec::with_capacity(o * (n / BLOCK));
    for blk in p.s_rot.chunks_exact(BLOCK) {
        let mut w = [0_i64; BLOCK];
        w.copy_from_slice(blk);
        let j = *cb.get(&w).ok_or_else(|| {
            TransferError::Shape("fold pack: plane block is not a codebook row".to_string())
        })?;
        idx.push(j);
    }
    let mut lo = Vec::with_capacity(o);
    let mut sp = Vec::with_capacity(o);
    let mut r_k = Vec::with_capacity(o * g);
    for i in 0..o {
        let row = &p.rh[i * g..(i + 1) * g];
        let (l, s) = fold_grid(row).ok_or_else(|| {
            TransferError::Shape(format!("fold pack: row {i} has no 16-level row grid"))
        })?;
        for &v in row {
            let k = row_level(v, l, s);
            if r_level(l, s, k) != Some(v) {
                return Err(TransferError::Shape(format!(
                    "fold pack: row {i} level does not decode"
                )));
            }
            r_k.push(k);
        }
        lo.push(l);
        sp.push(s);
    }
    let (c_head, c_q) = col_codes(&p.ch, 6)?;
    let (d_head, d_q) = col_codes(&p.dh, 5)?;
    Ok(Some(FoldPlane {
        o,
        n,
        rows: p.rows.as_ref().map(|r| {
            r.iter()
                .map(|&v| u32::try_from(v).unwrap_or(u32::MAX))
                .collect()
        }),
        idx,
        lo,
        sp,
        r_k,
        c_head,
        c_q,
        d_head,
        d_q,
    }))
}

/// The fold pack of `planes`: one [`FoldPlane`] record per `E8 ∪ O8` plane, and the bytes/weight
/// its planes carry. A corrector plane is skipped and counted in
/// [`FoldPackStats::skipped_dense`]. `O(Σ o·n + 495·16·Σ o·g)`.
///
/// # Errors
/// [`TransferError::Shape`] from [`fold_plane_from_fit`] or the byte layout.
pub fn fold_pack_of_planes(
    planes: &[&SinkPlane],
) -> Result<(Vec<FoldPlane>, FoldPackStats), TransferError> {
    let mut cb: HashMap<[i64; BLOCK], u8> = HashMap::with_capacity(CODE_LEN);
    for (j, w) in codebook().into_iter().enumerate() {
        cb.entry(w)
            .or_insert_with(|| u8::try_from(j).unwrap_or(u8::MAX));
    }
    let mut records: Vec<FoldPlane> = Vec::with_capacity(planes.len());
    let mut weights = 0u64;
    let mut skipped = 0usize;
    for p in planes {
        match fold_plane_from_fit(p, &cb)? {
            Some(r) => {
                weights += u64::try_from(r.o * r.n).unwrap_or(u64::MAX);
                records.push(r);
            }
            None => skipped += 1,
        }
    }
    let bytes = packing::fold_planes_write(&records)
        .map_err(|e| TransferError::Shape(format!("fold pack: {e:?}")))?;
    let count = records.len();
    Ok((
        records,
        FoldPackStats {
            bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            weights,
            planes: count,
            skipped_dense: skipped,
        },
    ))
}

/// Write the fold pack of `fits` (every layer's planes, in fit order) to `path` atomically.
/// `O(file bytes)`.
///
/// # Errors
/// [`TransferError::Shape`] from [`fold_pack_of_planes`]; [`TransferError::Io`] on a write failure.
pub fn fold_pack_write(
    fits: &[Vec<TensorFit>],
    path: &str,
) -> Result<FoldPackStats, TransferError> {
    let planes: Vec<&SinkPlane> = fits
        .iter()
        .flatten()
        .flat_map(|f| f.planes.iter())
        .collect();
    let (records, stats) = fold_pack_of_planes(&planes)?;
    let bytes = packing::fold_planes_write(&records)
        .map_err(|e| TransferError::Shape(format!("fold pack: {e:?}")))?;
    let stats = FoldPackStats {
        bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        ..stats
    };
    write_atomic(Path::new(path), &bytes)?;
    Ok(stats)
}

/// The stored records of a fold-pack file. `O(file bytes)`.
///
/// # Errors
/// [`TransferError::Io`] on a read failure; [`TransferError::Shape`] on a refused layout.
pub fn fold_pack_read(path: &str) -> Result<Vec<FoldPlane>, TransferError> {
    let bytes = fs::read(path).map_err(|e| io_err(Path::new(path), &e))?;
    packing::fold_planes_read(&bytes).map_err(|e| TransferError::Shape(format!("fold pack: {e:?}")))
}

/// The executable [`CodedPlane`] a stored record names: the codebook indices, the decoded row
/// levels and the two column fields, with zero latent offsets.
/// `O(o·n/8 + o·n/128 + n)`.
///
/// # Errors
/// [`TransferError::Shape`] on a field whose length disagrees with the shape, or a row level or a
/// column code outside its grid.
pub fn fold_plane_coded(f: &FoldPlane) -> Result<CodedPlane, TransferError> {
    let g = f.n / GROUP;
    if f.o == 0 || f.n == 0 || !f.n.is_multiple_of(GROUP) || f.idx.len() != f.o * (f.n / BLOCK) {
        return Err(TransferError::Shape("fold pack: plane shape".to_string()));
    }
    if f.lo.len() != f.o || f.sp.len() != f.o || f.r_k.len() != f.o * g {
        return Err(TransferError::Shape("fold pack: plane lengths".to_string()));
    }
    let mut r0 = Vec::with_capacity(f.o * g);
    for i in 0..f.o {
        for j in 0..g {
            let v = r_level(f.lo[i], f.sp[i], f.r_k[i * g + j])
                .ok_or_else(|| TransferError::Shape("fold pack: row level".to_string()))?;
            r0.push(v);
        }
    }
    let c0 = col_words(f.c_head, &f.c_q, 6)?;
    let d0 = col_words(f.d_head, &f.d_q, 5)?;
    Ok(CodedPlane {
        o: f.o,
        n: f.n,
        idx: f.idx.clone(),
        s_fix: None,
        rows: f.rows.as_ref().map(|r| {
            r.iter()
                .map(|&v| usize::try_from(v).unwrap_or(usize::MAX))
                .collect()
        }),
        lo: f.lo.clone(),
        sp: f.sp.clone(),
        r0,
        c0,
        d0,
        pa: vec![0_i64; f.o * g],
        pb: vec![0_i64; f.n],
        pd: vec![0_i64; f.n],
    })
}
