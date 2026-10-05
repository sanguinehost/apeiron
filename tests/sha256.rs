//! FIPS 180-4 known-answer tests for the SHA-256 port in `apeiron::sha256`.
//!
//! The four vectors are the FIPS 180-4 Appendix B examples: SHA-256 of `""` and `"abc"`, of the
//! 56-byte `"abcdbcde...nopq"` (which forces the two-block padding path), and the NIST million-`a`
//! vector. The expected strings are the published digests, cross-checked with Python
//! `hashlib.sha256`; they are not values recomputed by this implementation.

use std::fmt::Write as _;

use apeiron::sha256::sha256;

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[test]
fn sha256_empty() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn sha256_abc() {
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn sha256_two_block_case() {
    let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(msg.len(), 56);
    assert_eq!(
        hex(&sha256(msg)),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn sha256_million_a() {
    assert_eq!(
        hex(&sha256(&vec![b'a'; 1_000_000])),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

/// The padding boundary: 55 bytes pad into one block, 56 bytes require a second.
#[test]
fn padding_boundary() {
    assert_eq!(
        hex(&sha256(&[b'a'; 55])),
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
    );
    assert_eq!(
        hex(&sha256(&[b'a'; 56])),
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
    );
}

/// Block-boundary lengths must agree with the incremental padding they are equivalent to.
#[test]
fn exact_block_multiple() {
    // 64 bytes: one full block plus a padded empty remainder.
    assert_eq!(
        hex(&sha256(&[b'a'; 64])),
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
    );
}
