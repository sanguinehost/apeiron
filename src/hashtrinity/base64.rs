//! `base64` masquerade — the same `Engine` API as the real crate (v0.21+): an `Engine` trait,
//! `GeneralPurpose` engines, the four standard presets (`STANDARD`, `STANDARD_NO_PAD`,
//! `URL_SAFE`, `URL_SAFE_NO_PAD`), and a `DecodeError` with the same variants. Zero
//! dependencies — this REPLACES `base64`, it doesn't wrap it.
//!
//! Built from a real, measured need, not a hypothetical one: `base64::Engine` (`STANDARD` and
//! `URL_SAFE_NO_PAD`) is a direct dependency in at least two sibling projects
//! (`galvanize-k8s/kubeconfig.rs`, `sang-codspeed/main.rs`), and RFC 4648 base64 has been   CITATION-EXEMPT: `galvanize-k8s`, a tree outside `fold`
//! hand-rolled at least three more times independently (`saga/helm.rs`,   CITATION-EXEMPT: `saga`, a tree outside `fold`
//! `saga/cozystack-provider/account.rs` — near-identical decode-only copies — and   CITATION-EXEMPT: `saga`, a tree outside `fold`
//! `galvanize-template/filters.rs`'s standalone encode-only `b64encode`) — the exact   CITATION-EXEMPT: `galvanize-template`, a tree outside `fold`
//! multiple-independent-reimplementation pattern a masquerade exists to close.
//!
//! ```ignore
//! use apeiron::hashtrinity::base64::{engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD}, Engine};
//!
//! assert_eq!(STANDARD.encode(b"foobar"), "Zm9vYmFy");
//! assert_eq!(STANDARD.decode("Zm9vYmFy").unwrap(), b"foobar");
//! assert_eq!(URL_SAFE_NO_PAD.encode(b"foobar"), "Zm9vYmFy"); // no `+`/`/`/`=` in this input
//! ```
//!
//! ## What's ported and what's dropped
//!
//! The four `general_purpose` presets cover every alphabet/padding combination seen in the wild
//! (standard RFC 4648 with/without `=` padding, URL-safe `-`/`_` with/without padding). The real
//! crate's fully custom `Alphabet`/`GeneralPurposeConfig` builder (for a caller-defined alphabet)
//! isn't built — nothing found needs a non-standard alphabet, and the four presets are a
//! `pub const` table lookup away from adding a fifth if one ever does. `encode_slice`/
//! `decode_slice` (write into a caller-owned buffer, no allocation) aren't built either — narrow,
//! honestly-labeled gaps, not silently dropped surface (see `MASQUERADE.md`).
//!
//! **Verified against the real crate directly** (not assumed): every VALID encoding decodes to
//! the identical bytes, and every genuinely malformed input is rejected (never silently accepted)
//! — the two guarantees any caller here has actually exercised. The SPECIFIC [`DecodeError`]
//! variant for a few malformed-padding shapes can differ from the real crate's (e.g. a padded
//! engine handed `"Zm9v="` — a complete 4-symbol group plus one stray trailing `=`, which isn't a
//! valid padding position — reports `InvalidLength` here vs. the real crate's `InvalidByte`; real
//! `base64` validates padding placement incrementally per 4-byte group, this validates by
//! stripping trailing `=` first, and the two only disagree on inputs shaped like this one). No
//! caller found (`galvanize-k8s`, `sang-codspeed`) matches on the specific variant — both just
//! `format!("{e}")` it — so this hasn't been chased to byte-for-byte error-classification parity
//! on malformed input; a named, narrow gap, not a silent one.

/// RFC 4648 standard alphabet (`+`/`/`, used by `STANDARD`/`STANDARD_NO_PAD`).
const STANDARD_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
/// RFC 4648 URL-and-filename-safe alphabet (`-`/`_`, used by `URL_SAFE`/`URL_SAFE_NO_PAD`).
const URL_SAFE_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Decode error — same variants and `Display` text as the real crate's `DecodeError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// A byte that isn't part of this engine's alphabet (and isn't padding) at this position.
    InvalidByte(usize, u8),
    /// The input's length doesn't correspond to any valid encoding (for a padded engine: not a
    /// multiple of 4; for a no-pad engine: `len % 4 == 1`, which is impossible for any byte count).
    InvalidLength(usize),
    /// The last symbol encodes bits that must be zero — a real-world sign of truncated/corrupt
    /// input, not just "an unexpected character."
    InvalidLastSymbol(usize, u8),
    /// A padded engine's input either has no `=` where required or has one where padding isn't
    /// expected.
    InvalidPadding,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            DecodeError::InvalidByte(index, byte) => {
                write!(f, "Invalid symbol {byte}, offset {index}")
            }
            DecodeError::InvalidLength(len) => write!(f, "Invalid input length: {len}"),
            DecodeError::InvalidLastSymbol(index, byte) => {
                write!(f, "Invalid last symbol {byte}, offset {index}")
            }
            DecodeError::InvalidPadding => write!(f, "Invalid padding"),
        }
    }
}
impl std::error::Error for DecodeError {}

/// A configured base64 codec — an alphabet plus a padding policy. Implements [`Engine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneralPurpose {
    alphabet: &'static [u8; 64],
    pad: bool,
}

impl GeneralPurpose {
    /// Build a decode table mapping this engine's alphabet bytes back to their 6-bit value —
    /// `None` (via a sentinel) everywhere else. Computed on demand rather than stored `const`:
    /// the four presets are the only instances that exist in practice, and a 256-byte scratch
    /// table per `decode` call is cheap next to the allocation decoding itself already does.
    fn decode_table(&self) -> [i8; 256] {
        let mut table = [-1i8; 256];
        for (i, &b) in self.alphabet.iter().enumerate() {
            table[b as usize] = i as i8;
        }
        table
    }
}

/// The real crate's `Engine` trait — `encode`/`decode` dispatch on whichever [`GeneralPurpose`]
/// instance (alphabet + padding policy) you call them on.
pub trait Engine {
    fn encode<T: AsRef<[u8]>>(&self, input: T) -> String;
    fn decode<T: AsRef<[u8]>>(&self, input: T) -> Result<Vec<u8>, DecodeError>;

    /// Buffer-writing variant of [`Engine::encode`] — appends to `output` instead of allocating.
    fn encode_string<T: AsRef<[u8]>>(&self, input: T, output: &mut String) {
        output.push_str(&self.encode(input));
    }

    /// Buffer-writing variant of [`Engine::decode`] — appends to `output` instead of allocating.
    fn decode_vec<T: AsRef<[u8]>>(
        &self,
        input: T,
        output: &mut Vec<u8>,
    ) -> Result<(), DecodeError> {
        output.extend_from_slice(&self.decode(input)?);
        Ok(())
    }
}

impl Engine for GeneralPurpose {
    fn encode<T: AsRef<[u8]>>(&self, input: T) -> String {
        let input = input.as_ref();
        let mut out = String::with_capacity((input.len().div_ceil(3)) * 4);
        for chunk in input.chunks(3) {
            let b0 = u32::from(chunk[0]);
            let b1 = u32::from(chunk.get(1).copied().unwrap_or(0));
            let b2 = u32::from(chunk.get(2).copied().unwrap_or(0));
            let triple = (b0 << 16) | (b1 << 8) | b2;
            out.push(self.alphabet[((triple >> 18) & 0x3f) as usize] as char);
            out.push(self.alphabet[((triple >> 12) & 0x3f) as usize] as char);
            if chunk.len() > 1 {
                out.push(self.alphabet[((triple >> 6) & 0x3f) as usize] as char);
            } else if self.pad {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(self.alphabet[(triple & 0x3f) as usize] as char);
            } else if self.pad {
                out.push('=');
            }
        }
        out
    }

    fn decode<T: AsRef<[u8]>>(&self, input: T) -> Result<Vec<u8>, DecodeError> {
        let input = input.as_ref();
        let table = self.decode_table();

        // Padded engines: real input, minus any trailing `=`, still needs len % 4 != 1.
        // No-pad engines: the raw length itself needs len % 4 != 1 (0/2/3 are all valid tails).
        let unpadded_len = if self.pad {
            let stripped = input.iter().rposition(|&b| b != b'=').map_or(0, |i| i + 1);
            if (input.len() - stripped) > 2 {
                return Err(DecodeError::InvalidPadding);
            }
            if self.pad && input.len() % 4 != 0 {
                return Err(DecodeError::InvalidLength(input.len()));
            }
            stripped
        } else {
            // `=` is padding syntax, not an alphabet character — a no-pad engine seeing one at
            // all is a padding-shape problem, not "an unrecognized byte" (verified against the
            // real crate: `STANDARD_NO_PAD.decode("Zg==")` is `Err(InvalidPadding)`, not
            // `InvalidByte` at the first `=`, even though `=` isn't in this engine's alphabet).
            if input.contains(&b'=') {
                return Err(DecodeError::InvalidPadding);
            }
            if input.len() % 4 == 1 {
                return Err(DecodeError::InvalidLength(input.len()));
            }
            input.len()
        };

        let mut out = Vec::with_capacity(unpadded_len * 3 / 4 + 3);
        let mut chunks = input[..unpadded_len].chunks_exact(4);
        for chunk in &mut chunks {
            let vals = decode_quad(chunk, &table, 4)?;
            out.push(((vals[0] << 2) | (vals[1] >> 4)) as u8);
            out.push(((vals[1] << 4) | (vals[2] >> 2)) as u8);
            out.push(((vals[2] << 6) | vals[3]) as u8);
        }
        let rem = chunks.remainder();
        match rem.len() {
            0 => {}
            2 => {
                let vals = decode_quad(rem, &table, 2)?;
                if vals[1] & 0b1111 != 0 {
                    return Err(DecodeError::InvalidLastSymbol(unpadded_len - 1, rem[1]));
                }
                out.push(((vals[0] << 2) | (vals[1] >> 4)) as u8);
            }
            3 => {
                let vals = decode_quad(rem, &table, 3)?;
                if vals[2] & 0b11 != 0 {
                    return Err(DecodeError::InvalidLastSymbol(unpadded_len - 1, rem[2]));
                }
                out.push(((vals[0] << 2) | (vals[1] >> 4)) as u8);
                out.push(((vals[1] << 4) | (vals[2] >> 2)) as u8);
            }
            // len % 4 == 1 is unreachable for any byte-derived encoding — already rejected above.
            _ => return Err(DecodeError::InvalidLength(input.len())),
        }
        Ok(out)
    }
}

fn decode_quad(bytes: &[u8], table: &[i8; 256], n: usize) -> Result<[i32; 4], DecodeError> {
    let mut vals = [0i32; 4];
    for (i, &b) in bytes.iter().take(n).enumerate() {
        let v = table[b as usize];
        if v < 0 {
            return Err(DecodeError::InvalidByte(i, b));
        }
        vals[i] = v as i32;
    }
    Ok(vals)
}

/// The four standard engine presets — matches `base64::engine::general_purpose`'s public API
/// exactly, down to the module path.
pub mod engine {
    pub mod general_purpose {
        use super::super::{GeneralPurpose, STANDARD_ALPHABET, URL_SAFE_ALPHABET};

        /// RFC 4648 standard alphabet, `=`-padded (e.g. `Zm9vYmFy`, `Zg==`).
        pub const STANDARD: GeneralPurpose = GeneralPurpose {
            alphabet: STANDARD_ALPHABET,
            pad: true,
        };
        /// RFC 4648 standard alphabet, no padding.
        pub const STANDARD_NO_PAD: GeneralPurpose = GeneralPurpose {
            alphabet: STANDARD_ALPHABET,
            pad: false,
        };
        /// RFC 4648 URL-and-filename-safe alphabet, `=`-padded.
        pub const URL_SAFE: GeneralPurpose = GeneralPurpose {
            alphabet: URL_SAFE_ALPHABET,
            pad: true,
        };
        /// RFC 4648 URL-and-filename-safe alphabet, no padding — the common JWT/URL-token shape.
        pub const URL_SAFE_NO_PAD: GeneralPurpose = GeneralPurpose {
            alphabet: URL_SAFE_ALPHABET,
            pad: false,
        };
    }
}
