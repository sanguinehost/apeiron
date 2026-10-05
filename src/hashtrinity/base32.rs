//! `base32` — RFC 4648 base32 and Zooko's **z-base32**, zero dependencies.
//!
//! Built from a measured need, like [`crate::base64`] beside it: a 32-byte Ed25519 public key
//! has to become a DNS label to be published or resolved, and z-base32 is the encoding the
//! signed-record world agreed on (`bedrock::transport::pkarr`). The `base32` crate arrived here
//! only as a transitive dependency of that lift; the alphabet is 32 characters and the codec is
//! a 5-bit window, so depending on a crate for it is the reimplementation pattern a masquerade
//! exists to close.
//!
//! Two alphabets, and they are NOT interchangeable:
//!
//! * [`RFC4648`] — `A-Z2-7`, the standard. Case-insensitive on decode.
//! * [`ZBASE32`] — `ybndrfg8ejkmcpqxot1uwisza345h769`, Zooko's ordering, chosen so the glyphs
//!   people confuse least often carry the most information. Same 5-bit window, different table.
//!
//! Neither pads: a 32-byte key is 52 characters and the trailing bits are zero-filled, which is
//! what every consumer of these keys expects.
//!
//! ```
//! use hashtrinity::base32::{RFC4648, ZBASE32};
//!
//! assert_eq!(RFC4648.encode(b"foobar"), "MZXW6YTBOI");
//! assert_eq!(RFC4648.decode("MZXW6YTBOI").unwrap(), b"foobar");
//! assert_eq!(ZBASE32.decode(&ZBASE32.encode(b"foobar")).unwrap(), b"foobar");
//! ```

/// What went wrong decoding — one variant, because there is only one way to fail: a character
/// that is not in the alphabet. A truncated trailing group is not an error (the encoding is
/// unpadded, so the last group is legitimately short).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeError {
    /// Byte offset of the offending character in the input.
    pub position: usize,
    /// The character itself.
    pub character: char,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "invalid base32 character {:?} at position {}",
            self.character, self.position
        )
    }
}

impl std::error::Error for DecodeError {}

/// A base32 alphabet — the 32 output glyphs, in value order.
///
/// The whole codec is the 5-bit window; an alphabet is the only thing that varies between
/// base32 dialects, so it is data rather than a second implementation.
#[derive(Debug, Clone, Copy)]
pub struct Alphabet {
    table: &'static [u8; 32],
    /// Whether an ASCII-case-folded decode is accepted (true for RFC 4648, whose alphabet is
    /// upper-case; z-base32's is lower-case and folds the other way).
    fold_case: bool,
}

/// RFC 4648 base32: `A-Z2-7`, unpadded. Decode accepts either case.
pub const RFC4648: Alphabet = Alphabet {
    table: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567",
    fold_case: true,
};

/// Zooko's z-base32: `ybndrfg8ejkmcpqxot1uwisza345h769`, unpadded. Decode accepts either case.
pub const ZBASE32: Alphabet = Alphabet {
    table: b"ybndrfg8ejkmcpqxot1uwisza345h769",
    fold_case: true,
};

impl Alphabet {
    /// Encode `data` — 5 bits per output character, zero-filled at the tail, never padded.
    #[must_use]
    pub fn encode(&self, data: impl AsRef<[u8]>) -> String {
        let data = data.as_ref();
        if data.is_empty() {
            return String::new();
        }
        // ceil(bits / 5), where bits = 8 * len
        let mut out = String::with_capacity((data.len() * 8).div_ceil(5));
        let mut acc: u16 = 0;
        let mut bits: u8 = 0;
        for &b in data {
            acc = (acc << 8) | u16::from(b);
            bits += 8;
            while bits >= 5 {
                bits -= 5;
                let idx = ((acc >> bits) & 0x1f) as usize;
                out.push(self.table[idx] as char);
            }
        }
        if bits > 0 {
            // the tail: the remaining bits become the HIGH bits of one more symbol
            let idx = ((acc << (5 - bits)) & 0x1f) as usize;
            out.push(self.table[idx] as char);
        }
        out
    }

    /// Decode `text`. Unpadded input; a short trailing group is fine and its leftover bits — which
    /// the encoder zero-filled — are discarded.
    ///
    /// # Errors
    /// [`DecodeError`] if any character is outside this alphabet.
    pub fn decode(&self, text: impl AsRef<str>) -> Result<Vec<u8>, DecodeError> {
        let text = text.as_ref();
        let mut out = Vec::with_capacity(text.len() * 5 / 8);
        let mut acc: u16 = 0;
        let mut bits: u8 = 0;
        for (position, ch) in text.chars().enumerate() {
            let v = self.value_of(ch).ok_or(DecodeError { position, character: ch })?;
            acc = (acc << 5) | u16::from(v);
            bits += 5;
            if bits >= 8 {
                bits -= 8;
                out.push(((acc >> bits) & 0xff) as u8);
            }
        }
        Ok(out)
    }

    fn value_of(&self, ch: char) -> Option<u8> {
        let c = u8::try_from(u32::from(ch)).ok()?;
        // The table is small enough that a scan beats a 256-byte reverse map's cache miss, and
        // it keeps the alphabet the ONLY thing that differs between dialects.
        if let Some(i) = self.table.iter().position(|&t| t == c) {
            return u8::try_from(i).ok();
        }
        if self.fold_case {
            let flipped = if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else {
                return None;
            };
            return self
                .table
                .iter()
                .position(|&t| t == flipped)
                .and_then(|i| u8::try_from(i).ok());
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 4648 §10's own test vectors — the encoding is pinned to the spec, not to us.
    #[test]
    fn rfc4648_matches_the_specs_vectors() {
        for (input, expect) in [
            ("", ""),
            ("f", "MY"),
            ("fo", "MZXQ"),
            ("foo", "MZXW6"),
            ("foob", "MZXW6YQ"),
            ("fooba", "MZXW6YTB"),
            ("foobar", "MZXW6YTBOI"),
        ] {
            assert_eq!(RFC4648.encode(input.as_bytes()), expect, "encode {input:?}");
            assert_eq!(
                RFC4648.decode(expect).unwrap(),
                input.as_bytes(),
                "decode {expect:?}"
            );
        }
    }

    #[test]
    fn both_alphabets_round_trip_every_length_up_to_a_full_key() {
        for len in 0..=64usize {
            let data: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(37) ^ 0x5a) as u8).collect();
            for (name, a) in [("rfc4648", RFC4648), ("zbase32", ZBASE32)] {
                let enc = a.encode(&data);
                assert_eq!(a.decode(&enc).unwrap(), data, "{name} round trip at len {len}");
            }
        }
    }

    /// The reason this module exists: a 32-byte Ed25519 key is a 52-character label.
    #[test]
    fn a_thirty_two_byte_key_is_a_fifty_two_character_label() {
        let key = [0xABu8; 32];
        let enc = ZBASE32.encode(key);
        assert_eq!(enc.len(), 52);
        assert_eq!(ZBASE32.decode(&enc).unwrap(), key);
    }

    /// The alphabets are genuinely different — a value encoded under one must not read back the
    /// same under the other, or the two dialects would be interchangeable and this would be one
    /// alphabet with two names.
    #[test]
    fn the_two_alphabets_are_not_interchangeable() {
        let data = b"the quick brown fox";
        assert_ne!(RFC4648.encode(data), ZBASE32.encode(data));
    }

    #[test]
    fn decode_reports_the_offending_character_and_its_position() {
        let err = RFC4648.decode("MZXW6!TB").unwrap_err();
        assert_eq!(err.character, '!');
        assert_eq!(err.position, 5);
    }

    /// Case folding is a decode convenience, never an encode choice: encoding is always the
    /// alphabet's own case.
    #[test]
    fn decode_folds_case_but_encode_does_not() {
        assert_eq!(RFC4648.decode("mzxw6ytboi").unwrap(), b"foobar");
        assert_eq!(RFC4648.encode(b"foobar"), "MZXW6YTBOI");
        let z = ZBASE32.encode(b"foobar");
        assert_eq!(ZBASE32.decode(z.to_ascii_uppercase()).unwrap(), b"foobar");
    }
}
