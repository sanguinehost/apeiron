//! Extraction roots for the hashtrinity crypto face (`src/map/hashtrinity/crypto.rs`).
//!
//! The module is already in the translation unit as `crate::hashtrinity` (declared in `lib.rs` over
//! the real `src/map/hashtrinity/mod.rs`), so each `word_crypto_*` wrapper fixes the monomorphic
//! root and calls the real `crate::hashtrinity::crypto::<module>::<fn>` body; it restates no
//! formula. Roots are grouped as the public surface is: hashes, MAC/KDF, AEAD/stream cipher,
//! signatures and the protocol helpers.
//!
//! A root whose body reaches an item the pinned Aeneas does not model (a `Vec` mutation without a
//! model, `try_into` at a slice width, an OS read) surfaces as an unlisted axiom and is dropped
//! here; the refusal is recorded against its name, not waived.

use crate::hashtrinity::crypto;

// ── Hashes ───────────────────────────────────────────────────────────────────────────────────────

/// Hex is not rooted: `to_hex` reaches the `ToHex` iterator implementation (`hex.rs`).
pub fn word_crypto_ct_eq(a: &[u8], b: &[u8]) -> bool {
    crypto::ct_eq(a, b)
}

/// FIPS 180-4 SHA-256 at the concrete `&[u8]`/`[u8; 32]` types.
pub fn word_crypto_sha256(msg: &[u8]) -> [u8; 32] {
    crypto::sha256(msg)
}

/// FIPS 180-4 SHA-512 at the concrete `&[u8]`/`[u8; 64]` types.
pub fn word_crypto_sha512(msg: &[u8]) -> [u8; 64] {
    crypto::sha512(msg)
}

/// FIPS 180-4 SHA-384 (SHA-512 fold, 384 IV, truncated to 48 bytes).
pub fn word_crypto_sha384(msg: &[u8]) -> [u8; 48] {
    crypto::sha384(msg)
}

/// RFC 1321 MD5 at the concrete `&[u8]`/`[u8; 16]` types.
pub fn word_crypto_md5(msg: &[u8]) -> [u8; 16] {
    crypto::md5(msg)
}

/// RFC 7693 BLAKE2b at a caller-fixed output length.
pub fn word_crypto_blake2b(msg: &[u8], out_len: usize) -> Vec<u8> {
    crypto::blake2::blake2b(msg, out_len)
}

/// RFC 7693 BLAKE2b-512.
pub fn word_crypto_blake2b512(msg: &[u8]) -> [u8; 64] {
    crypto::blake2::blake2b512(msg)
}

/// RIPEMD-160 (the Bitcoin address hash's outer fold).
pub fn word_crypto_ripemd160(msg: &[u8]) -> [u8; 20] {
    crypto::ripemd160::ripemd160(msg)
}

/// Bitcoin `Hash160` = RIPEMD160(SHA256(x)).
pub fn word_crypto_hash160(data: &[u8]) -> [u8; 20] {
    crypto::ripemd160::hash160(data)
}

// ── MAC / KDF ────────────────────────────────────────────────────────────────────────────────────

/// RFC 2104 HMAC-SHA256.
pub fn word_crypto_hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    crypto::hmac_sha256(key, msg)
}

/// RFC 2104 HMAC-SHA512.
pub fn word_crypto_hmac_sha512(key: &[u8], msg: &[u8]) -> [u8; 64] {
    crypto::hmac_sha512(key, msg)
}

/// RFC 2104 HMAC-SHA384.
pub fn word_crypto_hmac_sha384(key: &[u8], msg: &[u8]) -> [u8; 48] {
    crypto::hmac_sha384(key, msg)
}

/// RFC 8018 PBKDF2-HMAC-SHA256.
pub fn word_crypto_pbkdf2_sha256(pass: &[u8], salt: &[u8], iters: u32, dklen: usize) -> Vec<u8> {
    crypto::pbkdf2_sha256(pass, salt, iters, dklen)
}

/// RFC 5869 HKDF-Extract (SHA-256).
pub fn word_crypto_hkdf_extract_sha256(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    crypto::hkdf_extract_sha256(salt, ikm)
}

/// RFC 5869 HKDF-Expand (SHA-256).
pub fn word_crypto_hkdf_expand_sha256(prk: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    crypto::hkdf_expand_sha256(prk, info, len)
}

/// RFC 5869 HKDF (SHA-256).
pub fn word_crypto_hkdf_sha256(salt: &[u8], ikm: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    crypto::hkdf_sha256(salt, ikm, info, len)
}

/// RFC 5869 HKDF (SHA-384).
pub fn word_crypto_hkdf_sha384(salt: &[u8], ikm: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    crypto::hkdf_sha384(salt, ikm, info, len)
}

/// RFC 5869 HKDF (SHA-512).
pub fn word_crypto_hkdf_sha512(salt: &[u8], ikm: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    crypto::hkdf_sha512(salt, ikm, info, len)
}

// ── AEAD / stream cipher ────────────────────────────────────────────────────────────────────────

/// RFC 8439 section 2.3 ChaCha20 keystream block.
pub fn word_crypto_chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    crypto::chacha_poly::chacha20_block(key, counter, nonce)
}

// ── Signatures ──────────────────────────────────────────────────────────────────────────────────

/// FIPS 186-4 ECDSA P-256 verification at the raw `r‖s` signature.
pub fn word_crypto_ecdsa_p256_verify(pubkey: &[u8; 64], hash: &[u8; 32], sig: &[u8; 64]) -> bool {
    crypto::ecdsa_p256_verify(pubkey, hash, sig)
}

/// P-256 public key from a 32-byte secret.
pub fn word_crypto_p256_pubkey(secret: &[u8; 32]) -> [u8; 64] {
    crypto::p256_pubkey(secret)
}

/// secp256k1 public key (uncompressed `x‖y`).
pub fn word_crypto_secp256k1_pubkey(secret: &[u8; 32]) -> [u8; 64] {
    crypto::secp256k1::pubkey(secret)
}

/// FIPS 186-4 ECDSA P-384 verification.
pub fn word_crypto_ecdsa_p384_verify(
    pubkey: &[u8; 96],
    hash: &[u8; 48],
    sig: &[u8; 96],
) -> bool {
    crypto::p384::ecdsa_p384_verify(pubkey, hash, sig)
}

/// RSASSA-PKCS1-v1_5 SHA-256 verification over a big-endian modulus.
pub fn word_crypto_rsa_pkcs1_sha256_verify(
    n_be: &[u8],
    e: u64,
    sig_be: &[u8],
    msg_hash: &[u8; 32],
) -> bool {
    crypto::rsa::rsa_pkcs1_sha256_verify(n_be, e, sig_be, msg_hash)
}

// ── Protocol helpers ───────────────────────────────────────────────────────────────────────────

/// BIP-173/BIP-350 8-bit to 5-bit regrouping.
pub fn word_crypto_bech32_convert_bits(
    data: &[u8],
    from: u32,
    to: u32,
    pad: bool,
) -> Option<Vec<u8>> {
    crypto::bech32::convert_bits(data, from, to, pad)
}