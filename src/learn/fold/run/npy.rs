//! NumPy `.npy` token reads and the calibration / in-sample / held-out chunk split of the
//! fold-transfer runs.
//!
//! Ports the data-selection half of the Python specs (read-only, outside this crate):
//! `fold_transfer.py` lines 2763–2778, `pv_finish.py` lines 1089–1095 and 1401–1404,
//! `kl_distill.py` lines 95–103.
//!
//! # What the Python selects
//!
//! A flat token-id array is cut into non-overlapping rows of `CHUNK = 512` tokens: `nch = n // 512`
//! with the trailing `n % 512` tokens dropped, and row `c` is tokens `[c·512, (c+1)·512)`.
//!
//! The three roles are plain slices of that row array, in this order:
//!
//! | role | chunks | code |
//! |---|---|---|
//! | calibration | `0 .. n_cal` | `ids_all[:n_cal]` |
//! | held-out (eval) | `n_cal .. n_cal + n_eval` | `ids_all[n_cal:n_cal + n_eval]` |
//! | in-sample | `0 .. n_in` | `ids_all[:n_in]` |
//!
//! So the held-out block starts at chunk offset `n_cal` with stride 1 and length `n_eval`; the
//! in-sample block is a **prefix of the calibration set**, not a disjoint third block
//! (`fold_transfer.py` default `n_cal = 100`, `n_in = 20`). No stride is applied and no shuffle
//! touches the held-out set. The Python asserts only `nch >= n_cal + n_eval`
//! (`fold_transfer.py:2765`).
//!
//! `--cal-extra k` (`fold_transfer.py:2774`) appends the **first** `k` chunks of the second token
//! file to the calibration rows: `np.load(...)[:k·512]`, reshaped to `(k, 512)`. That path is the
//! reason [`read_npy_i4`] exists; the append itself is the caller's concatenation.
//!
//! # What is not ported, and why
//!
//! The Python has no numpy `Generator` in this path. `fold_transfer.py:2797` permutes the
//! calibration **rows** for `seed != 0` with
//! `ids_cal[torch.randperm(ids_cal.shape[0], generator=torch.Generator().manual_seed(sd))]`, and
//! `pv_finish.py:1404` / `kl_distill.py:103` draw the training order with
//! `torch.randperm(ntr, generator=torch.Generator().manual_seed(seed))`. Both are torch's CPU
//! generator (Mersenne Twister 19937), not `np.random.default_rng` and not `np.random.RandomState`.
//! Neither generator is ported here: this module consumes the chunk index list as an input slice,
//! and a caller that needs seed `s != 0` supplies the permuted list. Seed `0` is the identity
//! permutation in all three scripts, which is what [`split`] computes.
//!
//! # Rules
//!
//! - No panic on caller input: every slice access is checked and every length is computed with
//!   checked arithmetic; refusals carry [`NpyError`].
//! - The header is the `numpy.lib.format` v1.0 (2-byte little-endian header length) or v2.0
//!   (4-byte) form: magic `\x93NUMPY`, version bytes, header length, then an ASCII Python dict
//!   literal terminated by `\n` and space-padded. `numpy.lib.format.MAX_HEADER_SIZE = 10000` is the
//!   cap on the declared header length.
//! - The one accepted dtype is `<i4`. Every other descriptor (endianness, width, float, bool),
//!   `fortran_order: True` and any shape that is not a 1-tuple are [`NpyError::Unsupported`]; a
//!   payload whose length is not exactly `4·N` is [`NpyError::ShapeMismatch`].

#![allow(
    clippy::doc_markdown,
    reason = "the doc names the upstream artifacts verbatim (`NumPy`, `numpy.lib.format`, \
              `torch.randperm` and the Python spec file names); backticking every occurrence \
              obscures which of them is a path, a dict key or a name."
)]

/// The NumPy magic prefix: `\x93NUMPY` (`numpy.lib.format.MAGIC_PREFIX`).
const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Largest accepted declared header length, `numpy.lib.format.MAX_HEADER_SIZE`.
const MAX_HEADER_LEN: usize = 10_000;

/// Bytes of one `i4` element.
const I4_BYTES: usize = 4;

/// A refused `.npy` read, chunking or split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyError {
    /// The first six bytes are not `\x93NUMPY`.
    NotNpy,
    /// The buffer ends before the declared header or payload does.
    Truncated,
    /// The header is not the ASCII dict literal the format defines: no terminating newline, a
    /// missing key, a malformed value, a non-decimal shape entry, or a declared length above
    /// `MAX_HEADER_LEN`.
    MalformedHeader,
    /// A well-formed header this reader does not accept: major version other than 1 or 2, a
    /// descriptor other than `<i4`, `fortran_order: True`, or a shape that is not a 1-tuple.
    Unsupported,
    /// The payload length is not exactly `4·N` for the declared `N`, or a chunk slice left the
    /// buffer.
    ShapeMismatch,
    /// [`chunks`] was called with `chunk == 0`.
    ZeroChunk,
    /// A token id in [`chunks`] is negative, so it has no `u32` token encoding.
    NegativeToken,
    /// [`split`] was asked for a chunk index outside the stream: `n_cal + n_eval > n_chunks`, or
    /// `n_in > n_cal` (the in-sample set is a prefix of the calibration set).
    SplitOutOfRange,
}

/// The three chunk-index sets one transfer run reads.
///
/// `cal` and `heldout` are disjoint; `in_sample` is a prefix of `cal`, matching
/// `ids_cal = ids_all[:n_cal]`, `ids_ev = ids_all[n_cal:n_cal+n_eval]`,
/// `ids_in = ids_all[:n_in]` in `fold_transfer.py:2772`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    /// Calibration chunk indices, `0 .. n_cal`.
    pub cal: Vec<usize>,
    /// In-sample chunk indices, `0 .. n_in` (a prefix of [`Split::cal`]).
    pub in_sample: Vec<usize>,
    /// Held-out chunk indices, `n_cal .. n_cal + n_eval`.
    pub heldout: Vec<usize>,
}

/// Read a 1-D little-endian `int32` `.npy` buffer into its token ids.
///
/// # Complexity
///
/// Time `O(len + h)` byte reads, `h` the header length (`h <= MAX_HEADER_LEN`): one scan per dict
/// key over the `h` header bytes plus one 4-byte decode per element. Space `O(len)` for the output,
/// `len = N` elements. Derivation: the element count is fixed by the header before the payload is
/// touched, and each element is read once.
///
/// # Errors
///
/// [`NpyError::NotNpy`] on a wrong magic; [`NpyError::Truncated`] when the buffer ends inside the
/// declared header or payload; [`NpyError::MalformedHeader`] on a header that is not the expected
/// ASCII dict; [`NpyError::Unsupported`] on a major version other than 1 or 2, a descriptor other
/// than `<i4`, `fortran_order: True`, or a shape that is not a 1-tuple;
/// [`NpyError::ShapeMismatch`] when the payload length is not `4·N`.
pub fn read_npy_i4(bytes: &[u8]) -> Result<Vec<i32>, NpyError> {
    let (n, payload) = header_i4(bytes)?;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let at = k * I4_BYTES;
        let word = payload
            .get(at..at + I4_BYTES)
            .ok_or(NpyError::ShapeMismatch)?;
        let mut b = [0u8; I4_BYTES];
        b.copy_from_slice(word);
        out.push(i32::from_le_bytes(b));
    }
    Ok(out)
}

/// Parse the header of a 1-D `<i4` `.npy` buffer: return its element count and payload slice.
///
/// # Complexity
///
/// Time `O(h·k)` byte comparisons, `h` the header length and `k <= 12` the longest dict key; space
/// `O(1)`. Derivation: three fixed-length key scans over the header, no allocation.
fn header_i4(bytes: &[u8]) -> Result<(usize, &[u8]), NpyError> {
    if bytes.get(..MAGIC.len()) != Some(MAGIC.as_slice()) {
        return Err(NpyError::NotNpy);
    }
    let major = *bytes.get(MAGIC.len()).ok_or(NpyError::Truncated)?;
    let minor = *bytes.get(MAGIC.len() + 1).ok_or(NpyError::Truncated)?;
    if minor != 0 {
        return Err(NpyError::Unsupported);
    }
    let (header_len, prefix): (usize, usize) = match major {
        1 => {
            let word = bytes.get(8..10).ok_or(NpyError::Truncated)?;
            let mut b = [0u8; 2];
            b.copy_from_slice(word);
            (usize::from(u16::from_le_bytes(b)), 10)
        }
        2 => {
            let word = bytes.get(8..12).ok_or(NpyError::Truncated)?;
            let mut b = [0u8; 4];
            b.copy_from_slice(word);
            let len =
                usize::try_from(u32::from_le_bytes(b)).map_err(|_| NpyError::MalformedHeader)?;
            (len, 12)
        }
        _ => return Err(NpyError::Unsupported),
    };
    if header_len > MAX_HEADER_LEN {
        return Err(NpyError::MalformedHeader);
    }
    let end = prefix
        .checked_add(header_len)
        .ok_or(NpyError::MalformedHeader)?;
    let header = bytes.get(prefix..end).ok_or(NpyError::Truncated)?;
    if header.last() != Some(&b'\n') || !header.is_ascii() {
        return Err(NpyError::MalformedHeader);
    }

    let descr = quoted(dict_value(header, b"'descr'")?)?;
    if descr != b"<i4" {
        return Err(NpyError::Unsupported);
    }
    match dict_value(header, b"'fortran_order'")? {
        b"False" => {}
        b"True" => return Err(NpyError::Unsupported),
        _ => return Err(NpyError::MalformedHeader),
    }
    let n = shape_len(dict_value(header, b"'shape'")?)?;
    let need = n.checked_mul(I4_BYTES).ok_or(NpyError::ShapeMismatch)?;
    let payload = bytes.get(end..).ok_or(NpyError::Truncated)?;
    if payload.len() != need {
        return Err(NpyError::ShapeMismatch);
    }
    Ok((n, payload))
}

/// The value of `key` in a `.npy` header dict: the bytes after `key:` up to the next top-level `,`
/// or the closing `}`, trimmed.
///
/// # Complexity
///
/// Time `O(h)` byte comparisons in the worst case for one key over the `h` header bytes; space
/// `O(1)`. Derivation: one substring search plus one left-to-right scan.
fn dict_value<'a>(header: &'a [u8], key: &[u8]) -> Result<&'a [u8], NpyError> {
    let at = find(header, key).ok_or(NpyError::MalformedHeader)?;
    let rest = header
        .get(at + key.len()..)
        .ok_or(NpyError::MalformedHeader)?;
    let rest = rest.trim_ascii_start();
    let rest = rest.strip_prefix(b":").ok_or(NpyError::MalformedHeader)?;
    let rest = rest.trim_ascii_start();
    let mut depth = 0usize;
    let mut end = rest.len();
    for (i, &c) in rest.iter().enumerate() {
        match c {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth = depth.saturating_sub(1),
            b'}' | b',' if depth == 0 => {
                end = i;
                break;
            }
            _ => {}
        }
    }
    rest.get(..end)
        .map(<[u8]>::trim_ascii)
        .ok_or(NpyError::MalformedHeader)
}

/// Split a quoted dict string value off its quotes.
///
/// # Complexity
///
/// Time and space `O(1)`: one length check and one subslice.
fn quoted(value: &[u8]) -> Result<&[u8], NpyError> {
    let inner = value
        .strip_prefix(b"'")
        .and_then(|v| v.strip_suffix(b"'"))
        .ok_or(NpyError::MalformedHeader)?;
    Ok(inner)
}

/// The single entry of a 1-tuple shape such as `(5219098,)`, or [`NpyError::Unsupported`] for any
/// other dimensionality (`()` is 0-D, `(2, 3)` is 2-D).
///
/// # Complexity
///
/// Time `O(len)` byte reads over the shape text (at most 64 bytes in a header), space `O(1)`.
/// Derivation: one comma split and one decimal accumulation.
fn shape_len(shape: &[u8]) -> Result<usize, NpyError> {
    let inner = shape
        .trim_ascii()
        .strip_prefix(b"(")
        .and_then(|s| s.strip_suffix(b")"))
        .ok_or(NpyError::MalformedHeader)?;
    let mut entries = inner
        .split(|&c| c == b',')
        .filter(|e| !e.trim_ascii().is_empty());
    let entry = entries.next().ok_or(NpyError::Unsupported)?;
    if entries.next().is_some() {
        return Err(NpyError::Unsupported);
    }
    let mut n: u64 = 0;
    for &d in entry.trim_ascii() {
        if !d.is_ascii_digit() {
            return Err(NpyError::MalformedHeader);
        }
        n = n
            .checked_mul(10)
            .and_then(|x| x.checked_add(u64::from(d - b'0')))
            .ok_or(NpyError::MalformedHeader)?;
    }
    usize::try_from(n).map_err(|_| NpyError::MalformedHeader)
}

/// First index at which `needle` occurs in `hay`, or `None`.
///
/// # Complexity
///
/// Time `O(len(hay)·len(needle))` byte comparisons in the worst case; space `O(1)`. Derivation:
/// naive scan over the `len(hay) - len(needle) + 1` start positions.
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| hay.get(i..i + needle.len()) == Some(needle))
}

/// Cut `ids` into non-overlapping chunks of exactly `chunk` tokens, dropping the trailing
/// `ids.len() % chunk` tokens (`nch = len(ids) // CHUNK`, `fold_transfer.py:2764`).
///
/// # Complexity
///
/// Time `O(len)` element reads (`len = ids.len()`): one `u32` conversion per element. Space
/// `O(len)` across the returned rows — each of the `len / chunk` rows holds exactly `chunk`
/// elements, and the dropped remainder is never copied. Derivation: one pass over the input, each
/// element placed in one row.
///
/// # Errors
///
/// [`NpyError::ZeroChunk`] if `chunk == 0`; [`NpyError::NegativeToken`] on the first negative id.
pub fn chunks(ids: &[i32], chunk: usize) -> Result<Vec<Vec<u32>>, NpyError> {
    if chunk == 0 {
        return Err(NpyError::ZeroChunk);
    }
    let whole = ids.len() / chunk;
    let mut out = Vec::with_capacity(whole);
    for c in 0..whole {
        let start = c * chunk;
        let row = ids
            .get(start..start + chunk)
            .ok_or(NpyError::ShapeMismatch)?;
        let mut row_out = Vec::with_capacity(chunk);
        for &id in row {
            row_out.push(u32::try_from(id).map_err(|_| NpyError::NegativeToken)?);
        }
        out.push(row_out);
    }
    Ok(out)
}

/// The calibration / in-sample / held-out chunk-index split of `fold_transfer.py:2772` for the
/// identity seed.
///
/// `cal = 0..n_cal`, `in_sample = 0..n_in`, `heldout = n_cal..n_cal + n_eval`. `n_in <= n_cal` is
/// required, because the Python's in-sample set is a prefix of its calibration set; a value above
/// `n_cal` would name chunks the transfer was not fitted on.
///
/// # Complexity
///
/// Time and space `O(n_cal + n_in + n_eval)`: one index push per named chunk. Derivation: three
/// contiguous ranges, no search.
///
/// # Errors
///
/// [`NpyError::SplitOutOfRange`] when `n_cal + n_eval > n_chunks` (the Python's own assertion
/// `nch >= n_cal + n_eval`) or when `n_in > n_cal`; [`NpyError::SplitOutOfRange`] also on
/// `n_cal + n_eval` overflow.
pub fn split(n_chunks: usize, n_cal: usize, n_in: usize, n_eval: usize) -> Result<Split, NpyError> {
    let end = n_cal.checked_add(n_eval).ok_or(NpyError::SplitOutOfRange)?;
    if end > n_chunks || n_in > n_cal {
        return Err(NpyError::SplitOutOfRange);
    }
    Ok(Split {
        cal: (0..n_cal).collect(),
        in_sample: (0..n_in).collect(),
        heldout: (n_cal..end).collect(),
    })
}
