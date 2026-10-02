#![allow(
    clippy::doc_markdown,
    reason = "`ChaCha20` and `Poly1305` are the specification's names for the algorithms, so prose \
              is not backticked around them"
)]
//! RFC 8439 known-answer tests for the ChaCha20 / Poly1305 / AEAD port in `apeiron::chacha`.
//!
//! The vectors are the RFC's own: the section 2.3.2 keystream block, the section 2.4.2 ChaCha20
//! ciphertext, the section 2.5.2 Poly1305 tag, and the section 2.8.2 AEAD ciphertext and tag. Each
//! expected value was produced by an independent implementation (Python `cryptography` for the
//! Poly1305 tag and the AEAD, and a separate Python ChaCha20 that reproduces the RFC 2.3.2 block
//! answer before use); none is a value recomputed by the code under test.

use std::fmt::Write as _;

use apeiron::chacha::{
    chacha20_block, chacha20_xor, chacha20poly1305_decrypt, chacha20poly1305_encrypt, poly1305_mac,
    Error, KEY_LEN, NONCE_LEN,
};

/// The RFC 8439 section 2.4.2 plaintext, also used by the section 2.8.2 example.
const LADIES: &str = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip \
                      for the future, sunscreen would be it.";

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    let mut i = 0usize;
    while i < bytes.len() {
        let _ = write!(s, "{:02x}", bytes[i]);
        i += 1;
    }
    s
}

/// A 32-byte key from a 64-character hex string.
fn key32(hexstr: &str) -> [u8; KEY_LEN] {
    let raw = hexstr.as_bytes();
    let mut k = [0u8; KEY_LEN];
    let mut i = 0usize;
    while i < KEY_LEN {
        k[i] = (hex_digit(raw[2 * i]) << 4) | hex_digit(raw[2 * i + 1]);
        i += 1;
    }
    k
}

/// A 12-byte nonce from a 24-character hex string.
fn nonce12(hexstr: &str) -> [u8; NONCE_LEN] {
    let raw = hexstr.as_bytes();
    let mut n = [0u8; NONCE_LEN];
    let mut i = 0usize;
    while i < NONCE_LEN {
        n[i] = (hex_digit(raw[2 * i]) << 4) | hex_digit(raw[2 * i + 1]);
        i += 1;
    }
    n
}

fn hex_digit(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("hex digit expected"),
    }
}

#[test]
fn rfc8439_2_3_2_keystream_block() {
    // Key 00:01:..:1f, nonce 00:00:00:09:00:00:00:4a:00:00:00:00, block counter 1.
    let block = chacha20_block(
        &key32("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"),
        1,
        &nonce12("000000090000004a00000000"),
    );
    assert_eq!(
        hex(&block),
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c\
         4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
    );
}

#[test]
fn rfc8439_2_4_2_chacha20_ciphertext() {
    // Same key, nonce 00:00:00:00:00:00:00:4a:00:00:00:00, block counter starting at 1, so the
    // payload spans keystream blocks 1 to 4 and is the last block short.
    let key = key32("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
    let nonce = nonce12("000000000000004a00000000");
    let mut buf = LADIES.as_bytes().to_vec();
    chacha20_xor(&key, 1, &nonce, &mut buf);
    assert_eq!(
        hex(&buf),
        "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0b\
         f91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d80\
         7ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab77937365a\
         f90bbf74a35be6b40b8eedf2785e42874d"
    );

    // ChaCha20 is its own inverse, so the same call restores the plaintext.
    chacha20_xor(&key, 1, &nonce, &mut buf);
    assert_eq!(buf, LADIES.as_bytes());
}

#[test]
fn rfc8439_2_5_2_poly1305_tag() {
    let otk = key32("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b");
    let tag = poly1305_mac(&otk, b"Cryptographic Forum Research Group");
    assert_eq!(hex(&tag), "a8061dc1305136c6c22b8baf0c0127a9");
}

#[test]
fn rfc8439_2_8_2_aead_seal() {
    let key = key32("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
    let nonce = nonce12("070000004041424344454647");
    let aad = [
        0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    ];
    let sealed = chacha20poly1305_encrypt(&key, &nonce, &aad, LADIES.as_bytes());
    assert_eq!(
        hex(&sealed[..sealed.len() - 16]),
        "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6\
         3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36\
         92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc\
         3ff4def08e4b7a9de576d26586cec64b6116"
    );
    assert_eq!(
        hex(&sealed[sealed.len() - 16..]),
        "1ae10b594f09e26a7e902ecbd0600691"
    );
}

#[test]
fn rfc8439_2_8_2_aead_open() {
    let key = key32("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
    let nonce = nonce12("070000004041424344454647");
    let aad = [
        0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    ];
    let mut sealed = Vec::new();
    sealed.extend_from_slice(
        &(hexstr_to_bytes(
            "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6\
             3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36\
             92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc\
             3ff4def08e4b7a9de576d26586cec64b6116\
             1ae10b594f09e26a7e902ecbd0600691",
        )),
    );
    let opened = chacha20poly1305_decrypt(&key, &nonce, &aad, &sealed).expect("tag verifies");
    assert_eq!(opened, LADIES.as_bytes());
}

/// Decodes a whitespace-free hex string of even length.
fn hexstr_to_bytes(s: &str) -> Vec<u8> {
    let raw = s.as_bytes();
    let mut out = Vec::with_capacity(raw.len() / 2);
    let mut i = 0usize;
    while i < raw.len() {
        out.push((hex_digit(raw[i]) << 4) | hex_digit(raw[i + 1]));
        i += 2;
    }
    out
}

#[test]
fn tampered_ciphertext_is_rejected() {
    let key = key32("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
    let nonce = nonce12("070000004041424344454647");
    let aad = [
        0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    ];
    let sealed = chacha20poly1305_encrypt(&key, &nonce, &aad, LADIES.as_bytes());

    // One flipped ciphertext bit.
    let mut tampered = sealed.clone();
    tampered[0] ^= 1;
    assert_eq!(
        chacha20poly1305_decrypt(&key, &nonce, &aad, &tampered),
        Err(Error::AuthenticationFailed)
    );

    // One flipped tag bit.
    let mut tampered = sealed.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert_eq!(
        chacha20poly1305_decrypt(&key, &nonce, &aad, &tampered),
        Err(Error::AuthenticationFailed)
    );

    // Changed associated data.
    let mut wrong_aad = aad;
    wrong_aad[0] ^= 1;
    assert_eq!(
        chacha20poly1305_decrypt(&key, &nonce, &wrong_aad, &sealed),
        Err(Error::AuthenticationFailed)
    );

    // A different nonce, i.e. a different one-time key.
    let mut wrong_nonce = nonce;
    wrong_nonce[0] ^= 1;
    assert_eq!(
        chacha20poly1305_decrypt(&key, &wrong_nonce, &aad, &sealed),
        Err(Error::AuthenticationFailed)
    );

    // Truncated below one tag.
    let short = &sealed[..8];
    assert_eq!(
        chacha20poly1305_decrypt(&key, &nonce, &aad, short),
        Err(Error::CiphertextTooShort)
    );
}
