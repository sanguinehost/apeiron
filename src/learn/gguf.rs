//! GGUF v3 container writer: header, typed metadata table, tensor directory, aligned data section.
//!
//! This is the inverse of the reader in [`crate::format::gguf`]: it takes the reader's
//! [`GgufHeader`], [`GVal`] and [`TensorDesc`] as input, so a file parsed by
//! [`parse_header`](crate::format::gguf::parse_header) can be written back byte-for-byte. The
//! on-disk layout follows `llama.cpp/gguf-py/gguf/gguf_writer.py` (llama.cpp rev `ece963f41`,
//! sha256 `9216da38c4bc5d7d9f2693125708fbaf64868e5447d26f186dbd07e0d059e594`): the four magic bytes,
//! a `u32` version, a `u64` tensor count, a `u64` metadata count, the metadata key/value pairs, the
//! tensor directory, zero padding to `general.alignment` (default 32), then the tensor payloads.
//!
//! This is a new primitive: a lineage search
//! (`rg 'GgufWriter|fn write_gguf|gguf_write|GGUFWriter'` over `fold`, `spinor-ttl`, `sanguine`,
//! `sanguine` and `lyrium`) returned no writer; the reader had no counterpart in this tree.
//!
//! Checked sizing is the contract: every string length, list count, dimension product, offset and
//! padding length is computed with checked arithmetic and refused with [`GgufError`] on overflow.
//! No length is truncated and no arithmetic panics on caller input.

use crate::format::gguf::{GVal, GgufError, GgufHeader, DEFAULT_ALIGNMENT, GGUF_MAGIC};

/// Metadata type ids, as written on disk and as the reader tags [`GVal`].
const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;

/// Append a `u64` little-endian string length and the raw key/value bytes.
///
/// The length is converted with `try_from`, so a key or value longer than `u64::MAX` (impossible in
/// memory on a 64-bit host) refuses rather than truncating. Temporal and spatial cost are O(bytes of
/// `s`): one length write and one copy of `s`.
fn write_string(out: &mut Vec<u8>, s: &[u8]) -> Result<(), GgufError> {
    let n = u64::try_from(s.len()).map_err(|_| GgufError::LengthOverflow)?;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(s);
    Ok(())
}

/// Append one integer payload in the width and signedness its metadata type id declares.
///
/// The value is range-checked with `try_from` for its declared type; a value outside that type
/// refuses with [`GgufError::LengthOverflow`] rather than wrapping. A type id that is not one of the
/// eight integer ids refuses with [`GgufError::UnknownType`].
fn write_int(out: &mut Vec<u8>, t: u32, x: i128) -> Result<(), GgufError> {
    match t {
        TYPE_UINT8 => out.push(u8::try_from(x).map_err(|_| GgufError::LengthOverflow)?),
        TYPE_INT8 => out.extend_from_slice(
            &i8::try_from(x)
                .map_err(|_| GgufError::LengthOverflow)?
                .to_le_bytes(),
        ),
        TYPE_UINT16 => out.extend_from_slice(
            &u16::try_from(x)
                .map_err(|_| GgufError::LengthOverflow)?
                .to_le_bytes(),
        ),
        TYPE_INT16 => out.extend_from_slice(
            &i16::try_from(x)
                .map_err(|_| GgufError::LengthOverflow)?
                .to_le_bytes(),
        ),
        TYPE_UINT32 => out.extend_from_slice(
            &u32::try_from(x)
                .map_err(|_| GgufError::LengthOverflow)?
                .to_le_bytes(),
        ),
        TYPE_INT32 => out.extend_from_slice(
            &i32::try_from(x)
                .map_err(|_| GgufError::LengthOverflow)?
                .to_le_bytes(),
        ),
        TYPE_UINT64 => out.extend_from_slice(
            &u64::try_from(x)
                .map_err(|_| GgufError::LengthOverflow)?
                .to_le_bytes(),
        ),
        TYPE_INT64 => out.extend_from_slice(
            &i64::try_from(x)
                .map_err(|_| GgufError::LengthOverflow)?
                .to_le_bytes(),
        ),
        _ => return Err(GgufError::UnknownType(t)),
    }
    Ok(())
}

/// Append one value's payload, without its leading type id.
///
/// A scalar uses its own type tag; an array writes its declared element type and element count, then
/// each element's payload. Array elements carry no per-element type tag on disk, so this recurses on
/// the element variants (the reader accepts nested arrays up to its depth bound). Cost is O(bytes).
fn write_payload(out: &mut Vec<u8>, v: &GVal) -> Result<(), GgufError> {
    match v {
        GVal::Int(x, t) => write_int(out, *t, *x)?,
        GVal::F32Bits(bits, _) => out.extend_from_slice(&bits.to_le_bytes()),
        GVal::F64Bits(bits, _) => out.extend_from_slice(&bits.to_le_bytes()),
        GVal::Bool(b) => out.push(u8::from(*b)),
        GVal::Str(s) => write_string(out, s)?,
        GVal::Arr(et, elems) => {
            out.extend_from_slice(&et.to_le_bytes());
            let n = u64::try_from(elems.len()).map_err(|_| GgufError::LengthOverflow)?;
            out.extend_from_slice(&n.to_le_bytes());
            for e in elems {
                write_payload(out, e)?;
            }
        }
    }
    Ok(())
}

/// Append one value: its `u32` type id, then its payload.
///
/// `ARRAY` (type 9) is written by [`write_payload`], which emits the element type id. Cost O(bytes).
fn write_value(out: &mut Vec<u8>, v: &GVal) -> Result<(), GgufError> {
    let t = match v {
        GVal::Int(_, t) | GVal::F32Bits(_, t) | GVal::F64Bits(_, t) => *t,
        GVal::Bool(_) => TYPE_BOOL,
        GVal::Str(_) => TYPE_STRING,
        GVal::Arr(_, _) => TYPE_ARRAY,
    };
    out.extend_from_slice(&t.to_le_bytes());
    write_payload(out, v)
}

/// Serialize the header, from the magic through the alignment padding before the data section.
///
/// The admissible domain is any [`GgufHeader`]; the function never panics on that domain and returns
/// `Result<Vec<u8>, GgufError>`.
///
/// The returned buffer is `header.data0` bytes long: the header fields and the zero padding to the
/// alignment boundary. [`GgufHeader::alignment`], [`GgufHeader::header_end`] and
/// [`GgufHeader::data0`] are not trusted here: the alignment is taken from the `general.alignment`
/// metadata by the reader's rule (the last such key with a positive integer value, else
/// [`DEFAULT_ALIGNMENT`]), so the padding written is the padding the reader expects, and the offsets
/// are recomputed with checked arithmetic. Cost is O(bytes of metadata and directory).
///
/// # Errors
///
/// * a list count, string length or dimension product that overflows `u64` returns
///   [`GgufError::LengthOverflow`]; a tensor whose dimension product overflows is refused via
///   [`crate::format::gguf::TensorDesc::n_elements`];
/// * an integer metadata value outside its declared type's range returns
///   [`GgufError::LengthOverflow`]; an integer type id outside the eight declared ids returns
///   [`GgufError::UnknownType`].
pub fn write_header(header: &GgufHeader) -> Result<Vec<u8>, GgufError> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    out.extend_from_slice(&header.version.to_le_bytes());
    let n_tensors = u64::try_from(header.tensors.len()).map_err(|_| GgufError::LengthOverflow)?;
    let n_kv = u64::try_from(header.kv.len()).map_err(|_| GgufError::LengthOverflow)?;
    out.extend_from_slice(&n_tensors.to_le_bytes());
    out.extend_from_slice(&n_kv.to_le_bytes());

    for (key, val) in &header.kv {
        write_string(&mut out, key)?;
        write_value(&mut out, val)?;
    }

    for t in &header.tensors {
        // Checked dimension product: an overflowing `u64` product refuses here.
        let _ = t.n_elements()?;
        write_string(&mut out, &t.name)?;
        let nd = u32::try_from(t.dims.len()).map_err(|_| GgufError::LengthOverflow)?;
        out.extend_from_slice(&nd.to_le_bytes());
        for &d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.ggml_type.to_le_bytes());
        out.extend_from_slice(&t.offset.to_le_bytes());
    }

    // Alignment padding, the same quotient/remainder ceiling the reader uses (no `div_ceil`, which
    // is an unmodeled external in the extraction). Every step is checked.
    let mut align = DEFAULT_ALIGNMENT;
    for (key, val) in &header.kv {
        if key.as_slice() == b"general.alignment" {
            if let Some(a) = val.as_i64().filter(|&a| a > 0) {
                align = u64::try_from(a).map_err(|_| GgufError::LengthOverflow)?;
            }
        }
    }
    let header_end = u64::try_from(out.len()).map_err(|_| GgufError::LengthOverflow)?;
    let whole = header_end / align;
    let rem = header_end % align;
    let blocks = if rem == 0 {
        whole
    } else {
        whole.checked_add(1).ok_or(GgufError::LengthOverflow)?
    };
    let data0 = blocks.checked_mul(align).ok_or(GgufError::LengthOverflow)?;
    let pad = data0
        .checked_sub(header_end)
        .ok_or(GgufError::LengthOverflow)?;
    let pad = usize::try_from(pad).map_err(|_| GgufError::LengthOverflow)?;
    let new_len = out
        .len()
        .checked_add(pad)
        .ok_or(GgufError::LengthOverflow)?;
    out.resize(new_len, 0u8);
    Ok(out)
}

/// Serialize a whole GGUF file: the header, then `data` as the tensor data section.
///
/// `data` is the byte range starting at the data-section offset (`data0`); it must contain every
/// tensor's payload at its declared offset. The admissible domain is any [`GgufHeader`] and any
/// `&[u8]`; the function never panics on that domain and returns `Result<Vec<u8>, GgufError>`.
///
/// Cost is O(header bytes + `data.len()`): the header is copied once and `data` once.
///
/// # Errors
///
/// * every refusal of [`write_header`] applies;
/// * a tensor whose declared offset plus its storage size (`offset + n_bytes`, both checked) runs
///   past `data` returns [`GgufError::OutOfBounds`]; types with no geometry in
///   [`crate::format::quant`] are not size-checked;
/// * a total length that overflows `usize` returns [`GgufError::LengthOverflow`].
pub fn write_file(header: &GgufHeader, data: &[u8]) -> Result<Vec<u8>, GgufError> {
    let mut out = write_header(header)?;
    let data_len = u64::try_from(data.len()).map_err(|_| GgufError::LengthOverflow)?;

    let mut i = 0usize;
    while i < header.tensors.len() {
        let t = &header.tensors[i];
        if let Some(n) = t.n_bytes()? {
            let end = t.offset.checked_add(n).ok_or(GgufError::LengthOverflow)?;
            if end > data_len {
                return Err(GgufError::OutOfBounds {
                    at: t.offset,
                    n,
                    file_len: data_len,
                });
            }
        }
        i += 1;
    }

    let _total = out
        .len()
        .checked_add(data.len())
        .ok_or(GgufError::LengthOverflow)?;
    out.reserve(data.len());
    out.extend_from_slice(data);
    Ok(out)
}
