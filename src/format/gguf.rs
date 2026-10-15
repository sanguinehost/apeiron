//! GGUF reader: fixed header, typed metadata table, tensor directory, aligned data offset.
//!
//! Ported from the served inference path in the sibling `fold` tree
//! (`fold/bedrock/src/serve/gguf.rs`, which reads the directory layout defined by
//! `llama.cpp/ggml/src/gguf.cpp`). This module keeps the metadata/tensor-directory parse and the
//! payload reads; the quant unpack lives in [`super::quant`].
//!
//! Every read is bounds-checked and every failure is a [`GgufError`]. The parser never indexes
//! before a check, never pre-allocates from an unchecked count, and never panics on file data.

use std::io::{Read, Seek, SeekFrom};

use super::quant::{self, QuantError};

/// The four magic bytes at the start of every GGUF file.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// Default `general.alignment` when the metadata omits it.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// Maximum tensor rank this reader accepts.
pub const MAX_DIMS: usize = 4;
/// Maximum nesting depth of a metadata array.
const MAX_ARRAY_DEPTH: u32 = 8;

/// The header or table field a read ran past (the payload of [`GgufError::Truncated`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufField {
    /// Metadata key string.
    KvKey,
    /// Metadata value type id.
    KvType,
    /// Metadata value.
    KvValue,
    /// Tensor name string.
    TensorName,
    /// Tensor rank.
    TensorNDims,
    /// Tensor dimension list.
    TensorDims,
    /// Tensor ggml type id.
    TensorType,
    /// Tensor data offset.
    TensorOffset,
    /// Leading magic bytes.
    Magic,
    /// Format version.
    Version,
    /// Tensor count.
    TensorCount,
    /// Metadata count.
    KvCount,
    /// Metadata and tensor counts together exceed the remaining bytes.
    KvTensorCount,
}

impl GgufField {
    /// The field's display name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GgufField::KvKey => "kv key",
            GgufField::KvType => "kv type",
            GgufField::KvValue => "kv value",
            GgufField::TensorName => "tensor name",
            GgufField::TensorNDims => "tensor n_dims",
            GgufField::TensorDims => "tensor dims",
            GgufField::TensorType => "tensor type",
            GgufField::TensorOffset => "tensor offset",
            GgufField::Magic => "magic",
            GgufField::Version => "version",
            GgufField::TensorCount => "tensor count",
            GgufField::KvCount => "kv count",
            GgufField::KvTensorCount => "kv/tensor count",
        }
    }
}

/// A refused GGUF operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// An OS read failed.
    Io(String),
    /// The leading four bytes were not `GGUF`.
    BadMagic([u8; 4]),
    /// The version is outside the supported `2..=3`.
    UnsupportedVersion(u32),
    /// A field ran past the available bytes; the payload names the field.
    Truncated(GgufField),
    /// A tensor declared more than [`MAX_DIMS`] dimensions.
    TooManyDims {
        /// Rank declared by the file.
        dims: u32,
    },
    /// A length or byte count did not fit the target integer width.
    LengthOverflow,
    /// The tensor's ggml type has no geometry here.
    UnknownType(u32),
    /// A tensor payload extends past the end of the file.
    OutOfBounds {
        /// Absolute start offset.
        at: u64,
        /// Requested byte count.
        n: u64,
        /// File length.
        file_len: u64,
    },
    /// No tensor carries the requested name.
    MissingTensor(String),
    /// The quant layer refused.
    Quant(QuantError),
}

impl core::fmt::Display for GgufError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io: {e}"),
            GgufError::BadMagic(m) => write!(f, "bad magic {m:?} — not a GGUF file"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            GgufError::Truncated(what) => write!(f, "truncated: {}", what.name()),
            GgufError::TooManyDims { dims } => write!(f, "tensor rank {dims} > {MAX_DIMS}"),
            GgufError::LengthOverflow => write!(f, "length overflow"),
            GgufError::UnknownType(t) => write!(f, "unknown ggml type {t}"),
            GgufError::OutOfBounds { at, n, file_len } => {
                write!(f, "tensor [{at}, {at}+{n}) past EOF {file_len}")
            }
            GgufError::MissingTensor(name) => write!(f, "no tensor {name}"),
            GgufError::Quant(e) => write!(f, "quant: {e}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<QuantError> for GgufError {
    fn from(e: QuantError) -> Self {
        GgufError::Quant(e)
    }
}

// ── metadata values ─────────────────────────────────────────────────────────────────────────────

/// One GGUF metadata value. Arrays keep their declared element type.
///
/// `FLOAT32`/`FLOAT64` metadata is kept as its little-endian bit pattern rather than a constructed
/// `f32`/`f64`: the extracting parse path is integer-only (Aeneas models no float), and the engine
/// boundary interprets the bits where it consumes them (`crate::engine::loader`). The second field
/// is the declared GGUF type id, exactly as for [`GVal::Int`].
#[derive(Clone, Debug, PartialEq)]
pub enum GVal {
    /// The ten integer types (`UINT8`…`INT64`), widened to `i128` so a `u64`'s top bit survives.
    Int(i128, u32),
    /// `FLOAT32`: the raw little-endian 32-bit pattern.
    F32Bits(u32, u32),
    /// `FLOAT64`: the raw little-endian 64-bit pattern.
    F64Bits(u64, u32),
    /// `BOOL`.
    Bool(bool),
    /// `STRING`: raw bytes (the extractor fragment carries no `String`).
    Str(Vec<u8>),
    /// `ARRAY`: element type id, then the elements.
    Arr(u32, Vec<GVal>),
}

impl GVal {
    /// The value as an integer, if it is one (or a bool).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            GVal::Int(v, _) => {
                if *v < (i64::MIN as i128) || *v > (i64::MAX as i128) {
                    None
                } else {
                    Some(*v as i64)
                }
            }
            GVal::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }
    /// The value as a byte slice, if it is a string.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            GVal::Str(s) => Some(s.as_slice()),
            _ => None,
        }
    }
}

// ── tensor directory ────────────────────────────────────────────────────────────────────────────

/// One tensor-directory entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDesc {
    /// GGUF name as raw bytes, e.g. `blk.0.attn_q.weight`.
    pub name: Vec<u8>,
    /// Dimensions as stored, fastest-varying first.
    pub dims: Vec<u64>,
    /// ggml type id.
    pub ggml_type: u32,
    /// Offset from the start of the data section.
    pub offset: u64,
}

impl TensorDesc {
    /// Total element count (a 0-dim tensor holds one scalar), checked.
    pub fn n_elements(&self) -> Result<u64, GgufError> {
        let mut acc: u64 = 1;
        for &d in &self.dims {
            acc = acc.checked_mul(d).ok_or(GgufError::LengthOverflow)?;
        }
        Ok(acc)
    }

    /// Storage bytes implied by the type and element count, checked; `None` for an unimplemented
    /// type.
    pub fn n_bytes(&self) -> Result<Option<u64>, GgufError> {
        if quant::block_geom(self.ggml_type).is_none() {
            return Ok(None);
        }
        Ok(Some(quant::required_bytes(
            self.ggml_type,
            self.n_elements()?,
        )?))
    }
}

// ── the byte cursor ─────────────────────────────────────────────────────────────────────────────

struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

impl Cur<'_> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.i)?;
        self.i += 1;
        Some(v)
    }
    fn u16(&mut self) -> Option<u16> {
        let a = self.u8()?;
        let b = self.u8()?;
        Some(u16::from_le_bytes([a, b]))
    }
    fn u32(&mut self) -> Option<u32> {
        let a = self.u8()?;
        let b = self.u8()?;
        let c = self.u8()?;
        let d = self.u8()?;
        Some(u32::from_le_bytes([a, b, c, d]))
    }
    fn u64(&mut self) -> Option<u64> {
        let a = self.u32()?;
        let b = self.u32()?;
        Some(u64::from(a) | (u64::from(b) << 32))
    }
    /// The four magic bytes, owned.
    fn four(&mut self) -> Option<[u8; 4]> {
        let a = self.u8()?;
        let b = self.u8()?;
        let c = self.u8()?;
        let d = self.u8()?;
        Some([a, b, c, d])
    }
    /// A length-prefixed byte string, copied into an owned `Vec<u8>`.
    ///
    /// The copy is deliberate: a slice returned through `&mut self` is the "nested borrows" shape
    /// the extractor rejects, and the metadata keys are compared by bytes.
    fn string(&mut self) -> Option<Vec<u8>> {
        let n = self.u64()?;
        if n > usize::MAX as u64 {
            return None;
        }
        let n = n as usize;
        let start = self.i;
        let end = start.checked_add(n)?;
        if end > self.b.len() {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        let mut k = start;
        while k < end {
            out.push(self.b[k]);
            k += 1;
        }
        self.i = end;
        Some(out)
    }
    /// One non-array typed value, or `None` for an unsupported type id. Type id 9 (`ARRAY`) is
    /// handled by [`Cur::value`], which never calls this with `t == 9`.
    fn scalar(&mut self, t: u32) -> Option<GVal> {
        Some(match t {
            0 => GVal::Int(self.u8()? as i128, t),
            1 => GVal::Int(self.u8()? as i8 as i128, t),
            2 => GVal::Int(self.u16()? as i128, t),
            3 => GVal::Int(self.u16()? as i16 as i128, t),
            4 => GVal::Int(self.u32()? as i128, t),
            5 => GVal::Int(self.u32()? as i32 as i128, t),
            6 => GVal::F32Bits(self.u32()?, t),
            7 => GVal::Bool(self.u8()? != 0),
            8 => GVal::Str(self.string()?),
            10 => GVal::Int(self.u64()? as i128, t),
            11 => GVal::Int(self.u64()? as i64 as i128, t),
            12 => {
                let lo = self.u32()?;
                let hi = self.u32()?;
                let bits = u64::from(lo) | (u64::from(hi) << 32);
                GVal::F64Bits(bits, t)
            }
            _ => return None,
        })
    }

    /// Read and validate an `ARRAY` header: element type id, then element count. `None` on a
    /// truncated read, a count that does not fit `usize`, or a count larger than the bytes the
    /// buffer still holds (a bogus count must not drive an allocation).
    fn array_header(&mut self) -> Option<(u32, usize)> {
        let et = self.u32()?;
        let n = self.u64()?;
        if n > usize::MAX as u64 {
            return None;
        }
        let n = n as usize;
        // The smallest element is one byte, so `n` elements need at least `n` bytes still
        // available.
        if n > self.b.len().saturating_sub(self.i) {
            return None;
        }
        Some((et, n))
    }

    /// One typed value, `depth` bounding array nesting.
    ///
    /// Array values are parsed by an explicit flat output stack plus a stack of open-array
    /// descriptors (`open_et`/`open_rem`/`open_start`), not by recursion: Aeneas cannot prove a
    /// recursive `value` monotone over the generated element loop, and a loop-only body compiles.
    /// `open_et.len()` is the number of open arrays, so the depth still available to the pending
    /// value is `depth - open_et.len()`; a pending array is refused when that would be zero.
    ///
    /// Header and scalar reads are hoisted out of the loop ([`Cur::array_header`], [`Cur::scalar`])
    /// so the loop body carries no `?`/early return, and failures are collected in `bad`.
    ///
    /// A completed array's elements are the tail of `flat` from `open_start[len-1]`; that tail is
    /// moved out with `mem::replace` (reversed in place and reversed back) and truncated, never
    /// copied: `GVal`'s derived `Clone` is self-referential in the generated Lean. Each element is
    /// moved once per enclosing level, so temporal and spatial cost are O(bytes · depth), i.e.
    /// O(bytes) at the fixed `MAX_ARRAY_DEPTH` of 8.
    fn value(&mut self, t: u32, depth: u32) -> Option<GVal> {
        if t != 9 {
            return self.scalar(t);
        }
        // The outer array is opened before the loop, so the loop only ever closes frames. `depth`
        // is the nesting budget; an array needs `depth != 0` and the empty array reads nothing.
        if depth == 0 {
            return None;
        }
        let (root_et, root_n) = self.array_header()?;
        if root_n == 0 {
            return Some(GVal::Arr(root_et, Vec::new()));
        }
        // Completed values, in depth-first order. A completed nested array is one entry.
        let mut flat: Vec<GVal> = Vec::new();
        // Open-array descriptors: element type, elements still owed, and `flat` index where the
        // array's elements begin. All three have the same length; index 0 is the outer array.
        let mut open_et: Vec<u32> = Vec::new();
        let mut open_rem: Vec<usize> = Vec::new();
        let mut open_start: Vec<usize> = Vec::new();
        open_et.push(root_et);
        open_rem.push(root_n);
        open_start.push(0usize);
        // The type of the value to parse next, when parsing.
        let mut pending = root_et;
        // Loop mode: 1 parses the pending value, 0 closes filled arrays. A `u32` rather than a
        // `bool`: Aeneas computes a loop fixed point over an integer mode but not over a
        // loop-carried boolean branch selector (`Could not compute a loop fixed point`).
        let mut mode = 1u32;
        let mut bad = false;
        let mut finished = false;
        // The outer array's value, assigned when its last element is attached.
        let mut result = GVal::Bool(false);
        while !bad && !finished {
            if mode == 0 {
                let li = open_rem.len() - 1;
                open_rem[li] -= 1;
                if open_rem[li] == 0 {
                    // The innermost open array filled: move its elements out of `flat` and close
                    // it. The elements are moved (not copied) with `mem::replace`, the temporary
                    // reversal undone in place, because `GVal`'s derived `Clone` is
                    // self-referential in the generated Lean.
                    let et = open_et[li];
                    let s = open_start[li];
                    let mut elems: Vec<GVal> = Vec::with_capacity(flat.len() - s);
                    let mut k = flat.len();
                    while k > s {
                        k -= 1;
                        elems.push(core::mem::replace(&mut flat[k], GVal::Bool(false)));
                    }
                    elems.reverse();
                    if li == 0 {
                        result = GVal::Arr(et, elems);
                        finished = true;
                    } else {
                        flat.truncate(s);
                        flat.push(GVal::Arr(et, elems));
                        open_et.truncate(li);
                        open_rem.truncate(li);
                        open_start.truncate(li);
                    }
                } else {
                    pending = open_et[li];
                    mode = 1;
                }
            } else if pending == 9 {
                // The open-array count is the depth already spent; a pending array needs one more.
                if open_et.len() as u32 >= depth {
                    bad = true;
                } else {
                    match self.array_header() {
                        Some((et, n)) => {
                            if n == 0 {
                                flat.push(GVal::Arr(et, Vec::new()));
                                mode = 0;
                            } else {
                                open_et.push(et);
                                open_rem.push(n);
                                open_start.push(flat.len());
                                pending = et;
                            }
                        }
                        None => bad = true,
                    }
                }
            } else {
                match self.scalar(pending) {
                    Some(v) => {
                        flat.push(v);
                        mode = 0;
                    }
                    None => bad = true,
                }
            }
        }
        if bad {
            return None;
        }
        Some(result)
    }
}

// ── the parsed header ───────────────────────────────────────────────────────────────────────────

/// A parsed GGUF header: version, metadata, tensor directory and the aligned data offset.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufHeader {
    /// GGUF version (2 or 3).
    pub version: u32,
    /// Metadata in file order; keys are raw bytes.
    pub kv: Vec<(Vec<u8>, GVal)>,
    /// Tensor directory in file order.
    pub tensors: Vec<TensorDesc>,
    /// `general.alignment` (default 32).
    pub alignment: u64,
    /// Bytes of metadata consumed, before alignment padding.
    pub header_end: u64,
    /// File offset where the tensor data section begins.
    pub data0: u64,
}

impl GgufHeader {
    /// The first metadata value with `key`, compared by bytes.
    pub fn meta(&self, key: &str) -> Option<&GVal> {
        let kb = key.as_bytes();
        let mut i = 0usize;
        while i < self.kv.len() {
            if self.kv[i].0.as_slice() == kb {
                return Some(&self.kv[i].1);
            }
            i += 1;
        }
        None
    }
    /// Metadata as an integer.
    pub fn meta_i64(&self, key: &str) -> Option<i64> {
        match self.meta(key) {
            Some(v) => v.as_i64(),
            None => None,
        }
    }
    /// Metadata as a byte string.
    pub fn meta_str(&self, key: &str) -> Option<&[u8]> {
        match self.meta(key) {
            Some(v) => v.as_bytes(),
            None => None,
        }
    }
    /// The architecture prefix `general.architecture`.
    pub fn arch(&self) -> String {
        match self.meta_str("general.architecture") {
            Some(a) => String::from_utf8_lossy(a).into_owned(),
            None => String::new(),
        }
    }
    /// `<arch>.<suffix>` as an integer.
    pub fn arch_i64(&self, suffix: &str) -> Option<i64> {
        let a = self.arch();
        self.meta_i64(&format!("{a}.{suffix}"))
    }
    /// The tensor with `name`.
    pub fn tensor(&self, name: &str) -> Option<&TensorDesc> {
        let mut found: Option<&TensorDesc> = None;
        let mut i = 0usize;
        while i < self.tensors.len() && found.is_none() {
            if self.tensors[i].name.as_slice() == name.as_bytes() {
                found = Some(&self.tensors[i]);
            }
            i += 1;
        }
        found
    }
}

/// Bytes of the `general.alignment` key and its length, matched without a slice-parameter helper
/// (a slice indexed through a reference is the extractor's "nested borrows" shape).
const ALIGN_BYTES: [u8; 17] = *b"general.alignment";
const ALIGN_LEN: usize = 17;

/// Parse one `(key, value)` metadata entry.
///
/// Hoisted out of [`parse_header`] so the function-level early returns (`?`) sit in a single-level
/// function rather than lexically inside the header's loops: the extractor rejects returns inside
/// nested loops.
fn parse_kv(c: &mut Cur<'_>) -> Result<(Vec<u8>, GVal), GgufError> {
    let key = c.string().ok_or(GgufError::Truncated(GgufField::KvKey))?;
    let t = c.u32().ok_or(GgufError::Truncated(GgufField::KvType))?;
    let v = c
        .value(t, MAX_ARRAY_DEPTH)
        .ok_or(GgufError::Truncated(GgufField::KvValue))?;
    Ok((key, v))
}

/// Parse one tensor-directory entry.
///
/// Holds the rank loop: keeping it here means the return in the rank loop is not lexically nested
/// inside the directory loop in [`parse_header`].
fn parse_tensor(c: &mut Cur<'_>) -> Result<TensorDesc, GgufError> {
    let name = c
        .string()
        .ok_or(GgufError::Truncated(GgufField::TensorName))?;
    let nd = c
        .u32()
        .ok_or(GgufError::Truncated(GgufField::TensorNDims))?;
    if nd as usize > MAX_DIMS {
        return Err(GgufError::TooManyDims { dims: nd });
    }
    let mut dims: Vec<u64> = Vec::with_capacity(nd as usize);
    let mut dim_err: Option<GgufError> = None;
    let mut k = 0u32;
    while k < nd && dim_err.is_none() {
        match c.u64() {
            Some(d) => {
                dims.push(d);
                k += 1;
            }
            None => dim_err = Some(GgufError::Truncated(GgufField::TensorDims)),
        }
    }
    if let Some(e) = dim_err {
        return Err(e);
    }
    let ggml_type = c.u32().ok_or(GgufError::Truncated(GgufField::TensorType))?;
    let offset = c
        .u64()
        .ok_or(GgufError::Truncated(GgufField::TensorOffset))?;
    Ok(TensorDesc {
        name,
        dims,
        ggml_type,
        offset,
    })
}

/// Parse a GGUF header from `buf`. `file_len` is the real file size, used only to record the
/// bounds; the parse itself never reads past `buf`.
pub fn parse_header(buf: &[u8], file_len: u64) -> Result<GgufHeader, GgufError> {
    let mut c = Cur { b: buf, i: 0 };
    let m = c.four().ok_or(GgufError::Truncated(GgufField::Magic))?;
    let magic = m;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic(magic));
    }
    let version = c.u32().ok_or(GgufError::Truncated(GgufField::Version))?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let n_tensors = c
        .u64()
        .ok_or(GgufError::Truncated(GgufField::TensorCount))?;
    let n_kv = c.u64().ok_or(GgufError::Truncated(GgufField::KvCount))?;

    // A count larger than the remaining bytes must not drive an allocation. A metadata entry is at
    // least one byte, so `n_kv > remaining` is impossible in a well-formed file. The guard is
    // expressed as `Option::ok_or` + `?` rather than an explicit `return Err(Truncated)`: Aeneas
    // reports "There should be no bottoms in the value" for a direct return of the
    // `&'static str`-carrying variant. The refusal value is unchanged.
    let remaining = buf.len().saturating_sub(c.i) as u64;
    let counts_ok = n_kv <= remaining && n_tensors <= remaining;
    let counts_guard: Option<()> = if counts_ok { Some(()) } else { None };
    counts_guard.ok_or(GgufError::Truncated(GgufField::KvTensorCount))?;

    let mut kv: Vec<(Vec<u8>, GVal)> = Vec::with_capacity(n_kv as usize);
    let mut kv_err: Option<GgufError> = None;
    let mut ki = 0u64;
    while ki < n_kv && kv_err.is_none() {
        match parse_kv(&mut c) {
            Ok(entry) => {
                kv.push(entry);
                ki += 1;
            }
            Err(e) => kv_err = Some(e),
        }
    }
    if let Some(e) = kv_err {
        return Err(e);
    }

    // `general.alignment`, scanned in its own pass so the metadata loop carries no moved key and no
    // second loop (the extractor's "could not match the contexts" shape).
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut ai = 0usize;
    while ai < kv.len() {
        if kv[ai].0.len() == ALIGN_LEN {
            let mut j = 0usize;
            let mut same = true;
            while j < ALIGN_LEN {
                if kv[ai].0[j] != ALIGN_BYTES[j] {
                    same = false;
                }
                j += 1;
            }
            if same {
                match kv[ai].1.as_i64() {
                    Some(a) if a > 0 => alignment = a as u64,
                    _ => {}
                }
            }
        }
        ai += 1;
    }

    let mut tensors: Vec<TensorDesc> = Vec::with_capacity(n_tensors as usize);
    let mut ts_err: Option<GgufError> = None;
    let mut ti = 0u64;
    while ti < n_tensors && ts_err.is_none() {
        match parse_tensor(&mut c) {
            Ok(desc) => {
                tensors.push(desc);
                ti += 1;
            }
            Err(e) => ts_err = Some(e),
        }
    }
    if let Some(e) = ts_err {
        return Err(e);
    }

    let header_end = c.i as u64;
    // `div_ceil` is an unmodeled external in the extraction; the quotient/remainder form is the
    // same ceiling for the nonzero `alignment` the scan guarantees, and `checked_mul` keeps the
    // refusal.
    let whole = header_end / alignment;
    let rem = header_end % alignment;
    let blocks = if rem == 0 { whole } else { whole + 1 };
    let data0 = blocks
        .checked_mul(alignment)
        .ok_or(GgufError::LengthOverflow)?;
    let _ = file_len;

    Ok(GgufHeader {
        version,
        kv,
        tensors,
        alignment,
        header_end,
        data0,
    })
}

// ── the file ────────────────────────────────────────────────────────────────────────────────────

/// An opened GGUF file. Tensor payloads are not resident; they are read on demand.
pub struct Gguf {
    /// Path the file was opened from.
    pub path: String,
    /// The parsed header.
    pub header: GgufHeader,
    /// Total file size.
    pub file_len: u64,
}

// Host file I/O: the opaque boundary of the extraction (`regen.sh` `--opaque`). Each returns the
// OS error text on failure; the callers wrap it in their own refusal type.

/// The first `min(n, file length)` bytes of `path` and the file length.
pub fn host_read_prefix(path: &str, n: usize) -> Result<(Vec<u8>, u64), String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    let mut buf = vec![0u8; n.min(len as usize)];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok((buf, len))
}

/// `n` bytes of `path` starting at byte offset `at`.
pub fn host_read_at(path: &str, at: u64, n: usize) -> Result<Vec<u8>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// The whole file at `path`.
pub fn host_read_all(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| e.to_string())
}

impl Gguf {
    /// Open a GGUF file and parse its header. The prefix read grows until the header parses, so a
    /// model with a large embedded vocabulary still opens without reading the whole payload.
    pub fn open(path: &str) -> Result<Gguf, GgufError> {
        let mut want = 8usize << 20;
        loop {
            let (buf, len) =
                host_read_prefix(path, want).map_err(|e| GgufError::Io(format!("{path}: {e}")))?;
            match parse_header(&buf, len) {
                Ok(header) => {
                    return Ok(Gguf {
                        path: path.to_string(),
                        header,
                        file_len: len,
                    })
                }
                Err(e) => {
                    if want as u64 >= len {
                        return Err(e);
                    }
                    want = (want * 4).min(len as usize);
                }
            }
        }
    }

    /// Read `n` bytes of `t`'s payload starting at the data section, checked against EOF.
    pub fn tensor_bytes_prefix(&self, t: &TensorDesc, n: u64) -> Result<Vec<u8>, GgufError> {
        let at = self
            .header
            .data0
            .checked_add(t.offset)
            .ok_or(GgufError::LengthOverflow)?;
        let end = at.checked_add(n).ok_or(GgufError::LengthOverflow)?;
        if end > self.file_len {
            return Err(GgufError::OutOfBounds {
                at,
                n,
                file_len: self.file_len,
            });
        }
        host_read_at(&self.path, at, n as usize).map_err(GgufError::Io)
    }

    /// Read a tensor's full payload.
    pub fn tensor_bytes(&self, t: &TensorDesc) -> Result<Vec<u8>, GgufError> {
        let n = t.n_bytes()?.ok_or(GgufError::UnknownType(t.ggml_type))?;
        self.tensor_bytes_prefix(t, n)
    }

    /// Dequantize a tensor by name to `f32`.
    pub fn dequant(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        self.dequant_prefix(name, usize::MAX)
    }

    /// Dequantize the first `max` elements (rounded down to whole blocks) of a tensor by name.
    pub fn dequant_prefix(&self, name: &str, max: usize) -> Result<Vec<f32>, GgufError> {
        let t = self
            .header
            .tensor(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))?;
        let (qk, bb) = quant::block_geom(t.ggml_type).ok_or(GgufError::UnknownType(t.ggml_type))?;
        let n_elem = t.n_elements()?;
        let n = if max == usize::MAX {
            usize::try_from(n_elem).map_err(|_| GgufError::LengthOverflow)?
        } else {
            let blocks = (max / qk) as u64;
            let capped = blocks.saturating_mul(qk as u64).min(n_elem);
            usize::try_from(capped).map_err(|_| GgufError::LengthOverflow)?
        };
        let nb = if qk == 0 {
            0
        } else {
            (n as u64).div_ceil(qk as u64)
        };
        let need = nb.checked_mul(bb as u64).ok_or(GgufError::LengthOverflow)?;
        let raw = self.tensor_bytes_prefix(t, need)?;
        match t.ggml_type {
            quant::GGML_TYPE_F32 => {
                let mut out = Vec::with_capacity(n);
                for c in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
                Ok(out)
            }
            quant::GGML_TYPE_F16 => {
                let mut out = Vec::with_capacity(n);
                for c in raw.chunks_exact(2) {
                    out.push(quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
                }
                Ok(out)
            }
            quant::GGML_TYPE_Q8_0 => {
                let mut out = Vec::with_capacity(n);
                for b in 0..nb as usize {
                    let d = quant::q8_0_scale(&raw, b)?;
                    let codes = quant::q8_0_codes(&raw, b)?;
                    for &q in codes.iter().take(n - out.len()) {
                        out.push(d * f32::from(q));
                    }
                }
                Ok(out)
            }
            other => {
                let mut out = vec![0f32; n];
                quant::dequant_super_block(other, &raw, n, &mut out)?;
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal little-endian writer for synthetic fixtures.
    struct W(Vec<u8>);
    impl W {
        fn new() -> Self {
            W(Vec::new())
        }
        fn u32(&mut self, v: u32) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn u64(&mut self, v: u64) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn bytes(&mut self, b: &[u8]) {
            self.0.extend_from_slice(b);
        }
        fn string(&mut self, s: &str) {
            self.u64(s.len() as u64);
            self.bytes(s.as_bytes());
        }
    }

    fn header(n_tensors: u64, n_kv: u64) -> W {
        let mut w = W::new();
        w.bytes(b"GGUF");
        w.u32(3);
        w.u64(n_tensors);
        w.u64(n_kv);
        w
    }

    #[test]
    fn wrong_magic_is_refused() {
        let mut w = W::new();
        w.bytes(b"XXXX");
        w.u32(3);
        w.u64(0);
        w.u64(0);
        assert_eq!(
            parse_header(&w.0, w.0.len() as u64),
            Err(GgufError::BadMagic(*b"XXXX"))
        );
    }

    #[test]
    fn unsupported_version_is_refused() {
        let mut w = W::new();
        w.bytes(b"GGUF");
        w.u32(99);
        w.u64(0);
        w.u64(0);
        assert_eq!(
            parse_header(&w.0, w.0.len() as u64),
            Err(GgufError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn corrupt_length_field_is_refused() {
        // One string KV whose declared length runs past the buffer.
        let mut w = header(0, 1);
        w.string("general.alignment");
        w.u32(8); // STRING type id
        w.u64(1 << 40); // absurd length
        w.bytes(b"short");
        assert_eq!(
            parse_header(&w.0, w.0.len() as u64),
            Err(GgufError::Truncated(GgufField::KvValue))
        );
    }

    #[test]
    fn truncated_tensor_directory_is_refused() {
        let mut w = header(1, 0);
        w.string("blk.0.weight");
        w.u32(2); // rank 2
        w.u64(4);
        // missing second dim / type / offset
        assert_eq!(
            parse_header(&w.0, w.0.len() as u64),
            Err(GgufError::Truncated(GgufField::TensorDims))
        );
    }

    #[test]
    fn too_many_dims_is_refused() {
        let mut w = header(1, 0);
        w.string("t");
        w.u32(5);
        assert_eq!(
            parse_header(&w.0, w.0.len() as u64),
            Err(GgufError::TooManyDims { dims: 5 })
        );
    }

    #[test]
    fn valid_header_parses_alignment_and_directory() {
        let mut w = header(1, 3);
        w.string("general.architecture");
        w.u32(8);
        w.string("llama");
        w.string("general.alignment");
        w.u32(10); // UINT64
        w.u64(64);
        w.string("llama.block_count");
        w.u32(4); // UINT32
        w.u32(32);
        w.string("blk.0.attn_q.weight");
        w.u32(2);
        w.u64(4096);
        w.u64(4096);
        w.u32(quant::GGML_TYPE_Q4_K);
        w.u64(0);
        let h = parse_header(&w.0, w.0.len() as u64).unwrap();
        assert_eq!(h.version, 3);
        assert_eq!(h.alignment, 64);
        assert_eq!(h.arch(), "llama");
        assert_eq!(h.arch_i64("block_count"), Some(32));
        let t = h.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(t.n_elements().unwrap(), 4096 * 4096);
        assert_eq!(t.n_bytes().unwrap(), Some(4096 * 4096 / 256 * 144));
        assert_eq!(h.data0 % 64, 0);
        assert!(h.data0 >= h.header_end);
    }

    #[test]
    fn float_metadata_keeps_raw_little_endian_bits() {
        // 1.5 has asymmetric bit patterns in both widths, so a byte-order or shift error changes
        // the observed limb.
        let mut w = header(0, 2);
        w.string("test.f32");
        w.u32(6); // FLOAT32
        w.u32(1.5f32.to_bits());
        w.string("test.f64");
        w.u32(12); // FLOAT64
        w.u64(1.5f64.to_bits());
        let h = parse_header(&w.0, w.0.len() as u64).unwrap();
        assert_eq!(h.meta("test.f32"), Some(&GVal::F32Bits(0x3fc0_0000, 6)));
        assert_eq!(
            h.meta("test.f64"),
            Some(&GVal::F64Bits(0x3ff8_0000_0000_0000, 12))
        );
    }

    #[test]
    fn element_overflow_is_refused() {
        let t = TensorDesc {
            name: b"big".to_vec(),
            dims: vec![u64::MAX, 2],
            ggml_type: quant::GGML_TYPE_Q4_K,
            offset: 0,
        };
        assert_eq!(t.n_elements(), Err(GgufError::LengthOverflow));
        assert_eq!(t.n_bytes(), Err(GgufError::LengthOverflow));
    }

    #[test]
    fn unknown_tensor_type_has_no_byte_size() {
        let t = TensorDesc {
            name: b"mystery".to_vec(),
            dims: vec![256],
            ggml_type: 77,
            offset: 0,
        };
        assert_eq!(t.n_bytes().unwrap(), None);
    }

    #[test]
    fn oversized_counts_do_not_allocate() {
        let mut w = W::new();
        w.bytes(b"GGUF");
        w.u32(3);
        w.u64(0);
        w.u64(u64::MAX);
        assert_eq!(
            parse_header(&w.0, w.0.len() as u64),
            Err(GgufError::Truncated(GgufField::KvTensorCount))
        );
    }

    #[test]
    fn array_count_past_buffer_is_refused() {
        let mut w = header(0, 1);
        w.string("tok_embd");
        w.u32(9); // ARRAY
        w.u32(8); // of STRING
        w.u64(1 << 30); // absurd element count
        assert_eq!(
            parse_header(&w.0, w.0.len() as u64),
            Err(GgufError::Truncated(GgufField::KvValue))
        );
    }
}
