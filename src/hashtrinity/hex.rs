//! `hex` masquerade — encode/decode hex strings, the same public API as the real crate (`encode`,
//! `decode`, `encode_upper`, `encode_to_slice[_upper]`, `decode_to_slice`, `decode_in_slice`,
//! `ToHex`/`FromHex`, `FromHexError`). Zero dependencies — this REPLACES `hex`, it doesn't wrap it.
//!
//! ```ignore
//! use apeiron::hashtrinity::hex;
//!
//! assert_eq!(hex::encode("Hello world!"), "48656c6c6f20776f726c6421");
//! assert_eq!(hex::decode("48656c6c6f20776f726c6421").unwrap(), b"Hello world!");
//! assert_eq!(hex::encode_upper([1u8, 2, 15, 16]), "01020F10");
//! ```
//!
//! ## What's ported and what's dropped
//!
//! The real crate is `#![no_std]` with an `alloc`/`std` feature split (its allocation-owning
//! functions — `encode`/`decode`/`ToHex`/`Vec<u8>: FromHex` — only exist under `feature =
//! "alloc"`). `hashtrinity` is never `no_std`, so that split doesn't apply here: everything is
//! always available. The real crate's `serde` feature is `#[serde(with = "hex")]` — a per-FIELD
//! annotation hooking real `serde`'s derive at the field level. Feature `syntax` here gives the
//! same round-trip as a pair of free functions ([`to_term`]/[`from_term`]) instead, since
//! hashtrinity's `AsTerm` (its serde-derive analog) is per-TYPE, not per-field — a named, narrow
//! difference in shape, not silently dropped surface (see `MASQUERADE.md`).

/// The error type for decoding a hex string — mirrors the real crate's `FromHexError` exactly
/// (same variants, same `Display` text).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FromHexError {
    /// An invalid character was found. Valid ones are `0...9`, `a...f`, `A...F`.
    InvalidHexCharacter { c: char, index: usize },
    /// A hex string's length must be even (two digits per byte).
    OddLength,
    /// Decoding into a fixed-size container (e.g. `[u8; N]`) requires `len() * 2 == N`.
    InvalidStringLength,
}

impl std::fmt::Display for FromHexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            FromHexError::InvalidHexCharacter { c, index } => {
                write!(f, "Invalid character {c:?} at position {index}")
            }
            FromHexError::OddLength => write!(f, "Odd number of digits"),
            FromHexError::InvalidStringLength => write!(f, "Invalid string length"),
        }
    }
}
impl std::error::Error for FromHexError {}

const LOWER: &[u8; 16] = b"0123456789abcdef";
const UPPER: &[u8; 16] = b"0123456789ABCDEF";
#[allow(non_upper_case_globals)]
const __: u8 = u8::MAX;

#[rustfmt::skip]
static DECODE_TABLE: [u8; 256] = [
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, __, __, __, __, __, __,
    __, 10, 11, 12, 13, 14, 15, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, 10, 11, 12, 13, 14, 15, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,
];

#[inline]
fn val(bytes: &[u8], idx: usize) -> Result<u8, FromHexError> {
    let upper = DECODE_TABLE[bytes[0] as usize];
    let lower = DECODE_TABLE[bytes[1] as usize];
    if upper == u8::MAX {
        return Err(FromHexError::InvalidHexCharacter {
            c: bytes[0] as char,
            index: idx,
        });
    }
    if lower == u8::MAX {
        return Err(FromHexError::InvalidHexCharacter {
            c: bytes[1] as char,
            index: idx + 1,
        });
    }
    Ok((upper << 4) | lower)
}

#[inline]
fn byte2hex(byte: u8, table: &[u8; 16]) -> (u8, u8) {
    (
        table[((byte & 0xf0) >> 4) as usize],
        table[(byte & 0x0f) as usize],
    )
}

fn encode_to_slice_inner(
    input: &[u8],
    output: &mut [u8],
    table: &[u8; 16],
) -> Result<(), FromHexError> {
    if input.len() * 2 != output.len() {
        return Err(FromHexError::InvalidStringLength);
    }
    for (byte, out) in input.iter().zip(output.chunks_exact_mut(2)) {
        let (hi, lo) = byte2hex(*byte, table);
        out[0] = hi;
        out[1] = lo;
    }
    Ok(())
}

/// Encode bytes into a mutable ASCII slice using lowercase characters. `output` must be exactly
/// `input.len() * 2` bytes.
pub fn encode_to_slice<T: AsRef<[u8]>>(input: T, output: &mut [u8]) -> Result<(), FromHexError> {
    encode_to_slice_inner(input.as_ref(), output, LOWER)
}
/// Encode bytes into a mutable ASCII slice using uppercase characters.
pub fn encode_to_slice_upper<T: AsRef<[u8]>>(
    input: T,
    output: &mut [u8],
) -> Result<(), FromHexError> {
    encode_to_slice_inner(input.as_ref(), output, UPPER)
}

/// Encode `data` as a lowercase hex string (e.g. `f9b4ca`) — the string is always
/// `2 * data.len()` bytes long.
#[must_use]
pub fn encode<T: AsRef<[u8]>>(data: T) -> String {
    let data = data.as_ref();
    let mut out = vec![0u8; data.len() * 2];
    encode_to_slice(data, &mut out).expect("output sized exactly");
    String::from_utf8(out).expect("hex digits are always valid ASCII/UTF-8")
}
/// Encode `data` as an uppercase hex string (e.g. `F9B4CA`).
#[must_use]
pub fn encode_upper<T: AsRef<[u8]>>(data: T) -> String {
    let data = data.as_ref();
    let mut out = vec![0u8; data.len() * 2];
    encode_to_slice_upper(data, &mut out).expect("output sized exactly");
    String::from_utf8(out).expect("hex digits are always valid ASCII/UTF-8")
}

/// Decode a hex string into bytes. Upper/lower case are both valid and may be mixed.
pub fn decode<T: AsRef<[u8]>>(data: T) -> Result<Vec<u8>, FromHexError> {
    FromHex::from_hex(data)
}

/// Decode a hex string directly into a mutable byte slice (`out.len() == data.len() / 2`).
pub fn decode_to_slice<T: AsRef<[u8]>>(data: T, out: &mut [u8]) -> Result<(), FromHexError> {
    let data = data.as_ref();
    if data.len() % 2 != 0 {
        return Err(FromHexError::OddLength);
    }
    if data.len() / 2 != out.len() {
        return Err(FromHexError::InvalidStringLength);
    }
    for (i, (chunk, byte)) in data.chunks_exact(2).zip(out).enumerate() {
        *byte = val(chunk, 2 * i)?;
    }
    Ok(())
}

/// Decode a hex string in place: the first half of `in_out` becomes the decoded bytes.
pub fn decode_in_slice(in_out: &mut [u8]) -> Result<(), FromHexError> {
    if in_out.len() % 2 != 0 {
        return Err(FromHexError::OddLength);
    }
    for i in 0..(in_out.len() / 2) {
        let byte = val(&in_out[2 * i..2 * i + 2], 2 * i)?;
        in_out[i] = byte;
    }
    Ok(())
}

/// Encode a byte sequence as a hex string — implemented for anything `AsRef<[u8]>`
/// (`String`/`str`/`Vec<u8>`/`[u8]`), matching the real crate's blanket impl.
pub trait ToHex {
    fn encode_hex<T: FromIterator<char>>(&self) -> T;
    fn encode_hex_upper<T: FromIterator<char>>(&self) -> T;
}

struct BytesToHexChars<'a> {
    inner: std::slice::Iter<'a, u8>,
    table: &'static [u8; 16],
    next: Option<char>,
}
impl Iterator for BytesToHexChars<'_> {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        match self.next.take() {
            Some(c) => Some(c),
            None => self.inner.next().map(|byte| {
                let cur = self.table[(byte >> 4) as usize] as char;
                self.next = Some(self.table[(byte & 0x0f) as usize] as char);
                cur
            }),
        }
    }
}

impl<T: AsRef<[u8]> + ?Sized> ToHex for T {
    fn encode_hex<U: FromIterator<char>>(&self) -> U {
        BytesToHexChars {
            inner: self.as_ref().iter(),
            table: LOWER,
            next: None,
        }
        .collect()
    }
    fn encode_hex_upper<U: FromIterator<char>>(&self) -> U {
        BytesToHexChars {
            inner: self.as_ref().iter(),
            table: UPPER,
            next: None,
        }
        .collect()
    }
}

/// Types decodable from a hex string — implemented for `Vec<u8>` and `[u8; N]`, matching the
/// real crate.
pub trait FromHex: Sized {
    type Error;
    fn from_hex<T: AsRef<[u8]>>(hex: T) -> Result<Self, Self::Error>;
}

impl FromHex for Vec<u8> {
    type Error = FromHexError;
    fn from_hex<T: AsRef<[u8]>>(hex: T) -> Result<Self, Self::Error> {
        let hex = hex.as_ref();
        if hex.len() % 2 != 0 {
            return Err(FromHexError::OddLength);
        }
        let mut out = vec![0u8; hex.len() / 2];
        decode_to_slice(hex, &mut out)?;
        Ok(out)
    }
}
impl<const N: usize> FromHex for [u8; N] {
    type Error = FromHexError;
    fn from_hex<T: AsRef<[u8]>>(hex: T) -> Result<Self, Self::Error> {
        let mut out = [0u8; N];
        decode_to_slice(hex, &mut out as &mut [u8])?;
        Ok(out)
    }
}
