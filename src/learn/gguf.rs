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

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::collapsible_match,
    clippy::manual_range_contains,
    clippy::single_match,
    reason = "the writer's integer payloads are range-checked against their declared type's bounds \
              and then cast, the form the pinned extraction fragment accepts (`try_from` is \
              modeled only for the wide-to-word casts); the range tests are written out because \
              `RangeInclusive::contains` is an unmodeled external; the metadata sweeps destructure \
              one `GVal` arm at a time"
)]

use crate::format::gguf::{
    GVal, GgufError, GgufHeader, DEFAULT_ALIGNMENT, GGUF_MAGIC, MAX_ARRAY_DEPTH,
};

/// Bytes of the `general.alignment` key and its length; matched in a dedicated pass so the
/// metadata sweeps above carry no second loop (the extractor's "could not match the contexts"
/// shape is a second loop under one index).
const ALIGN_BYTES: [u8; 17] = *b"general.alignment";
const ALIGN_LEN: usize = 17;

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
/// The value is range-checked against its declared type's bounds and then cast; a value outside
/// that range refuses with [`GgufError::LengthOverflow`] rather than wrapping. The signed widths
/// are cast through their unsigned width (two's-complement, byte-identical) so the pinned model's
/// signed `to_le_bytes` (`Array I8`) is not reached. A type id that is not one of the eight
/// integer ids refuses with [`GgufError::UnknownType`].
fn write_int(out: &mut Vec<u8>, t: u32, x: i128) -> Result<(), GgufError> {
    // The bound tests are written out and the payload follows the `as` cast: the pinned Aeneas
    // models `try_from` only for the wide-to-word casts and no `signum`, so the explicit comparison
    // is the form the fragment accepts. The refused values and the emitted bytes are unchanged.
    match t {
        TYPE_UINT8 => {
            if x < 0 || x > 255 {
                return Err(GgufError::LengthOverflow);
            }
            out.push(x as u8);
        }
        TYPE_INT8 => {
            if x < -128 || x > 127 {
                return Err(GgufError::LengthOverflow);
            }
            out.extend_from_slice(&(x as i8 as u8).to_le_bytes());
        }
        TYPE_UINT16 => {
            if x < 0 || x > 65_535 {
                return Err(GgufError::LengthOverflow);
            }
            out.extend_from_slice(&(x as u16).to_le_bytes());
        }
        TYPE_INT16 => {
            if x < -32_768 || x > 32_767 {
                return Err(GgufError::LengthOverflow);
            }
            out.extend_from_slice(&(x as i16 as u16).to_le_bytes());
        }
        TYPE_UINT32 => {
            if x < 0 || x > 4_294_967_295 {
                return Err(GgufError::LengthOverflow);
            }
            out.extend_from_slice(&(x as u32).to_le_bytes());
        }
        TYPE_INT32 => {
            if x < -2_147_483_648 || x > 2_147_483_647 {
                return Err(GgufError::LengthOverflow);
            }
            out.extend_from_slice(&(x as i32 as u32).to_le_bytes());
        }
        TYPE_UINT64 => {
            if x < 0 || x > 18_446_744_073_709_551_615 {
                return Err(GgufError::LengthOverflow);
            }
            out.extend_from_slice(&(x as u64).to_le_bytes());
        }
        TYPE_INT64 => match i64::try_from(x) {
            Ok(v) => out.extend_from_slice(&(v as u64).to_le_bytes()),
            Err(_) => return Err(GgufError::LengthOverflow),
        },
        _ => return Err(GgufError::UnknownType(t)),
    }
    Ok(())
}

/// Append a non-array payload. An array reaching this function is at the ninth nesting level: the
/// reader refuses that depth, so the writer refuses it with [`GgufError::TooDeep`].
///
/// # Complexity
///
/// O(bytes of the value), one write and one copy per byte.
fn write_payload_scalar(out: &mut Vec<u8>, v: &GVal) -> Result<(), GgufError> {
    match v {
        GVal::Int(x, t) => write_int(out, *t, *x),
        GVal::F32Bits(bits, _) => {
            out.extend_from_slice(&bits.to_le_bytes());
            Ok(())
        }
        GVal::F64Bits(bits, _) => {
            out.extend_from_slice(&bits.to_le_bytes());
            Ok(())
        }
        GVal::Bool(b) => {
            out.push(u8::from(*b));
            Ok(())
        }
        GVal::Str(s) => write_string(out, s),
        GVal::Arr(_, _) => Err(GgufError::TooDeep {
            depth: MAX_ARRAY_DEPTH,
        }),
    }
}

/// One nesting level of the payload writer; `$next` writes an array element at the next level.
///
/// The chain is `write_payload`, `write_payload_2` … `write_payload_8`, then
/// [`write_payload_scalar`]. It is unrolled because the pinned Charon/Aeneas emitter cannot
/// elaborate a recursion whose call sits inside a loop, and refuses a container of references
/// (`ty_regions shouldn't be called on erased or body regions`); the bound is the reader's
/// [`MAX_ARRAY_DEPTH`].
///
/// # Complexity
///
/// One level writes O(bytes of its subtree); the fixed chain is O(bytes).
macro_rules! payload_level {
    ($name:ident, $next:ident) => {
        fn $name(out: &mut Vec<u8>, v: &GVal) -> Result<(), GgufError> {
            match v {
                GVal::Arr(et, elems) => {
                    out.extend_from_slice(&et.to_le_bytes());
                    let n = u64::try_from(elems.len()).map_err(|_| GgufError::LengthOverflow)?;
                    out.extend_from_slice(&n.to_le_bytes());
                    // Flag loop: a `?` inside a loop has no fixed point in the pinned
                    // Charon/Aeneas fragment, so the first refusal is carried in `err` and a
                    // single `return` follows the loop. The write order is the element order.
                    let mut err: Option<GgufError> = None;
                    let mut i = 0usize;
                    while i < elems.len() && err.is_none() {
                        match $next(out, &elems[i]) {
                            Ok(()) => {}
                            Err(e) => err = Some(e),
                        }
                        i += 1;
                    }
                    if let Some(e) = err {
                        return Err(e);
                    }
                    Ok(())
                }
                _ => write_payload_scalar(out, v),
            }
        }
    };
}

payload_level!(write_payload, write_payload_2);
payload_level!(write_payload_2, write_payload_3);
payload_level!(write_payload_3, write_payload_4);
payload_level!(write_payload_4, write_payload_5);
payload_level!(write_payload_5, write_payload_6);
payload_level!(write_payload_6, write_payload_7);
payload_level!(write_payload_7, write_payload_8);
payload_level!(write_payload_8, write_payload_scalar);

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

    // Flag loops: a `?` inside a loop has no fixed point in the pinned Charon/Aeneas fragment, so
    // each sweep carries its first refusal in `err`, every loop condition is guarded by
    // `err.is_none()`, and one `return` follows the sweeps. The byte order and the first-refusal
    // order are the sequential order, unchanged.
    let mut err: Option<GgufError> = None;
    let mut k = 0usize;
    while k < header.kv.len() && err.is_none() {
        let (key, val) = &header.kv[k];
        match write_string(&mut out, key) {
            Ok(()) => match write_value(&mut out, val) {
                Ok(()) => {}
                Err(e) => err = Some(e),
            },
            Err(e) => err = Some(e),
        }
        k += 1;
    }

    let mut t = 0usize;
    while t < header.tensors.len() && err.is_none() {
        let td = &header.tensors[t];
        // Checked dimension product: an overflowing `u64` product refuses here.
        match td.n_elements() {
            Err(e) => err = Some(e),
            Ok(_) => match write_string(&mut out, &td.name) {
                Err(e) => err = Some(e),
                Ok(()) => match u32::try_from(td.dims.len()) {
                    Err(_) => err = Some(GgufError::LengthOverflow),
                    Ok(nd) => {
                        out.extend_from_slice(&nd.to_le_bytes());
                        let mut d = 0usize;
                        while d < td.dims.len() {
                            out.extend_from_slice(&td.dims[d].to_le_bytes());
                            d += 1;
                        }
                        out.extend_from_slice(&td.ggml_type.to_le_bytes());
                        out.extend_from_slice(&td.offset.to_le_bytes());
                    }
                },
            },
        }
        t += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    // Alignment padding, the same quotient/remainder ceiling the reader uses (no `div_ceil`, which
    // is an unmodeled external in the extraction). Every step is checked.
    let mut align = DEFAULT_ALIGNMENT;
    let mut ai = 0usize;
    while ai < header.kv.len() {
        if header.kv[ai].0.len() == ALIGN_LEN {
            let mut j = 0usize;
            let mut same = true;
            while j < ALIGN_LEN {
                if header.kv[ai].0[j] != ALIGN_BYTES[j] {
                    same = false;
                }
                j += 1;
            }
            if same {
                match header.kv[ai].1.as_i64() {
                    Some(a) => {
                        if a > 0 {
                            align = a as u64;
                        }
                    }
                    None => {}
                }
            }
        }
        ai += 1;
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

    // Flag loop: the first refusal is carried in `err` so no `?` or `return` sits inside the
    // sweep; the refusal order (n_bytes, offset overflow, out of bounds) is unchanged.
    let mut err: Option<GgufError> = None;
    let mut i = 0usize;
    while i < header.tensors.len() && err.is_none() {
        let t = &header.tensors[i];
        match t.n_bytes() {
            Err(e) => err = Some(e),
            Ok(None) => {}
            Ok(Some(n)) => match t.offset.checked_add(n) {
                None => err = Some(GgufError::LengthOverflow),
                Some(end) => {
                    if end > data_len {
                        err = Some(GgufError::OutOfBounds {
                            at: t.offset,
                            n,
                            file_len: data_len,
                        });
                    }
                }
            },
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    let _total = out
        .len()
        .checked_add(data.len())
        .ok_or(GgufError::LengthOverflow)?;
    out.reserve(data.len());
    out.extend_from_slice(data);
    Ok(out)
}
