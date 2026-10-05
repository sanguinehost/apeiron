//! A minimal, pure-Rust TLS 1.3 client (zero deps, zero `unsafe`) — enough to
//! speak HTTPS to AWS S3 with the mandatory cipher suite TLS_AES_128_GCM_SHA256
//! and X25519 key exchange.
//!
//! Sub-modules are the crypto primitives (each KAT-verified) and the handshake
//! state machine over `std::net::TcpStream`.

pub mod aes_gcm {
    //! AES-128 (FIPS-197) + GCM AEAD (NIST SP 800-38D) — pure safe Rust.
    //! The record-protection cipher for TLS_AES_128_GCM_SHA256.
    //! Verified against the FIPS-197 block vector and NIST GCM test vectors.
    //!
    //! The GHASH multiply by the fixed key H is a **linear operator**; H is constant
    //! for the connection, so we certify it ONCE into a byte-indexed table
    //! (`GcmKey`, the streaming-law rest state) and each block multiply becomes 16
    //! indexed XORs instead of 128 bit-iterations. The table is built from the
    //! verified bit-by-bit `gf_mul`, so it is correct by linearity.
    //!
    //! NOTE: table-free AES round (the S-box is a data-dependent index — not hardened
    //! against timing side channels). Correct first; constant-time is a follow-on.

    #![allow(dead_code)]

    const SBOX: [u8; 256] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ];

    const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

    fn key_schedule(key: &[u8; 16]) -> [[u8; 16]; 11] {
        let mut w = [[0u8; 4]; 44];
        for i in 0..4 {
            w[i] = [key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]];
        }
        for i in 4..44 {
            let mut t = w[i - 1];
            if i % 4 == 0 {
                t = [t[1], t[2], t[3], t[0]];
                for b in t.iter_mut() {
                    *b = SBOX[*b as usize];
                }
                t[0] ^= RCON[i / 4 - 1];
            }
            for j in 0..4 {
                w[i][j] = w[i - 4][j] ^ t[j];
            }
        }
        let mut rk = [[0u8; 16]; 11];
        for r in 0..11 {
            for c in 0..4 {
                rk[r][4 * c..4 * c + 4].copy_from_slice(&w[4 * r + c]);
            }
        }
        rk
    }

    #[inline]
    fn xtime(a: u8) -> u8 {
        (a << 1) ^ (0x1b & (0u8.wrapping_sub(a >> 7)))
    }

    fn encrypt_block(rk: &[[u8; 16]; 11], input: &[u8; 16]) -> [u8; 16] {
        let mut s = *input;
        for j in 0..16 {
            s[j] ^= rk[0][j];
        }
        for r in 1..10 {
            for b in s.iter_mut() {
                *b = SBOX[*b as usize];
            }
            shift_rows(&mut s);
            mix_columns(&mut s);
            for j in 0..16 {
                s[j] ^= rk[r][j];
            }
        }
        for b in s.iter_mut() {
            *b = SBOX[*b as usize];
        }
        shift_rows(&mut s);
        for j in 0..16 {
            s[j] ^= rk[10][j];
        }
        s
    }

    fn shift_rows(s: &mut [u8; 16]) {
        let t = *s;
        for r in 1..4 {
            for c in 0..4 {
                s[r + 4 * c] = t[r + 4 * ((c + r) % 4)];
            }
        }
    }

    fn mix_columns(s: &mut [u8; 16]) {
        for c in 0..4 {
            let i = 4 * c;
            let (a0, a1, a2, a3) = (s[i], s[i + 1], s[i + 2], s[i + 3]);
            let t = a0 ^ a1 ^ a2 ^ a3;
            s[i] ^= t ^ xtime(a0 ^ a1);
            s[i + 1] ^= t ^ xtime(a1 ^ a2);
            s[i + 2] ^= t ^ xtime(a2 ^ a3);
            s[i + 3] ^= t ^ xtime(a3 ^ a0);
        }
    }

    // ── GCM ──────────────────────────────────────────────────────────────────────

    /// GF(2^128) multiply, big-endian bit convention (NIST SP 800-38D). The slow but
    /// obviously-correct reference; used only to BUILD the per-key table.
    fn gf_mul(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
        let mut z = [0u8; 16];
        let mut v = *y;
        for i in 0..128 {
            let xbit = (x[i / 8] >> (7 - (i % 8))) & 1;
            if xbit == 1 {
                for j in 0..16 {
                    z[j] ^= v[j];
                }
            }
            let lsb = v[15] & 1;
            for j in (1..16).rev() {
                v[j] = (v[j] >> 1) | (v[j - 1] << 7);
            }
            v[0] >>= 1;
            if lsb == 1 {
                v[0] ^= 0xe1;
            }
        }
        z
    }

    /// The certified operator: `table[p][b]` = (byte `b` at position `p`) · H, so
    /// `x · H = ⊕_p table[p][x[p]]` — 16 indexed XORs, by linearity of mul-by-H.
    type GTab = [[[u8; 16]; 256]; 16];

    fn build_table(h: &[u8; 16]) -> Box<GTab> {
        // basis[i] = (unit vector, bit i set) · H, via the verified gf_mul.
        let mut basis = [[0u8; 16]; 128];
        for i in 0..128 {
            let mut e = [0u8; 16];
            e[i / 8] = 1 << (7 - (i % 8));
            basis[i] = gf_mul(&e, h);
        }
        let mut t: Box<GTab> = Box::new([[[0u8; 16]; 256]; 16]);
        for p in 0..16 {
            for v in 0..256usize {
                let mut acc = [0u8; 16];
                for j in 0..8 {
                    if (v >> (7 - j)) & 1 == 1 {
                        let b = &basis[p * 8 + j];
                        for k in 0..16 {
                            acc[k] ^= b[k];
                        }
                    }
                }
                t[p][v] = acc;
            }
        }
        t
    }

    #[inline]
    fn gmul_tab(t: &GTab, x: &[u8; 16]) -> [u8; 16] {
        let mut z = [0u8; 16];
        for p in 0..16 {
            let row = &t[p][x[p] as usize];
            for k in 0..16 {
                z[k] ^= row[k];
            }
        }
        z
    }

    fn ghash_tab(t: &GTab, aad: &[u8], ct: &[u8]) -> [u8; 16] {
        let mut y = [0u8; 16];
        let absorb = |data: &[u8], y: &mut [u8; 16]| {
            for chunk in data.chunks(16) {
                for (j, b) in chunk.iter().enumerate() {
                    y[j] ^= b;
                }
                *y = gmul_tab(t, y);
            }
        };
        absorb(aad, &mut y);
        absorb(ct, &mut y);
        let mut lenblk = [0u8; 16];
        lenblk[..8].copy_from_slice(&((aad.len() as u64) * 8).to_be_bytes());
        lenblk[8..].copy_from_slice(&((ct.len() as u64) * 8).to_be_bytes());
        for j in 0..16 {
            y[j] ^= lenblk[j];
        }
        gmul_tab(t, &y)
    }

    fn inc32(ctr: &mut [u8; 16]) {
        let mut n = u32::from_be_bytes([ctr[12], ctr[13], ctr[14], ctr[15]]);
        n = n.wrapping_add(1);
        ctr[12..].copy_from_slice(&n.to_be_bytes());
    }

    /// A GCM key with round keys and the H-multiplication table precomputed once —
    /// build per connection/direction, then seal/open every record cheaply.
    pub struct GcmKey {
        rk: [[u8; 16]; 11],
        table: Box<GTab>,
    }

    impl GcmKey {
        pub fn new(key: &[u8; 16]) -> Self {
            let rk = key_schedule(key);
            let h = encrypt_block(&rk, &[0u8; 16]);
            Self {
                rk,
                table: build_table(&h),
            }
        }

        pub fn seal(&self, nonce: &[u8; 12], aad: &[u8], pt: &[u8]) -> (Vec<u8>, [u8; 16]) {
            let mut j0 = [0u8; 16];
            j0[..12].copy_from_slice(nonce);
            j0[15] = 1;
            let mut ct = Vec::with_capacity(pt.len());
            let mut counter = j0;
            for chunk in pt.chunks(16) {
                inc32(&mut counter);
                let ks = encrypt_block(&self.rk, &counter);
                for (i, &b) in chunk.iter().enumerate() {
                    ct.push(b ^ ks[i]);
                }
            }
            let s = ghash_tab(&self.table, aad, &ct);
            let ej0 = encrypt_block(&self.rk, &j0);
            let mut tag = [0u8; 16];
            for i in 0..16 {
                tag[i] = s[i] ^ ej0[i];
            }
            (ct, tag)
        }

        pub fn open(&self, nonce: &[u8; 12], aad: &[u8], ct: &[u8], tag: &[u8; 16]) -> Option<Vec<u8>> {
            let s = ghash_tab(&self.table, aad, ct);
            let mut j0 = [0u8; 16];
            j0[..12].copy_from_slice(nonce);
            j0[15] = 1;
            let ej0 = encrypt_block(&self.rk, &j0);
            let mut want = [0u8; 16];
            for i in 0..16 {
                want[i] = s[i] ^ ej0[i];
            }
            let mut diff = 0u8;
            for i in 0..16 {
                diff |= want[i] ^ tag[i];
            }
            if diff != 0 {
                return None;
            }
            let mut pt = Vec::with_capacity(ct.len());
            let mut counter = j0;
            for chunk in ct.chunks(16) {
                inc32(&mut counter);
                let ks = encrypt_block(&self.rk, &counter);
                for (i, &b) in chunk.iter().enumerate() {
                    pt.push(b ^ ks[i]);
                }
            }
            Some(pt)
        }
    }

    /// One-shot AES-128-GCM (builds the table each call) — for tests / handshake.
    pub fn seal(key: &[u8; 16], nonce: &[u8; 12], aad: &[u8], pt: &[u8]) -> (Vec<u8>, [u8; 16]) {
        GcmKey::new(key).seal(nonce, aad, pt)
    }

    pub fn open(
        key: &[u8; 16],
        nonce: &[u8; 12],
        aad: &[u8],
        ct: &[u8],
        tag: &[u8; 16],
    ) -> Option<Vec<u8>> {
        GcmKey::new(key).open(nonce, aad, ct, tag)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::sync::crypto::hex;
        fn bytes(s: &str) -> Vec<u8> {
            (0..s.len() / 2)
                .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
                .collect()
        }
        fn b16(s: &str) -> [u8; 16] {
            bytes(s).try_into().unwrap()
        }
        fn b12(s: &str) -> [u8; 12] {
            bytes(s).try_into().unwrap()
        }

        #[test]
        fn aes128_fips197_block() {
            let key = b16("000102030405060708090a0b0c0d0e0f");
            let pt = b16("00112233445566778899aabbccddeeff");
            let rk = key_schedule(&key);
            assert_eq!(
                hex(&encrypt_block(&rk, &pt)),
                "69c4e0d86a7b0430d8cdb78070b4c55a"
            );
        }

        #[test]
        fn gcm_nist_case1_empty() {
            let (ct, tag) = seal(&[0u8; 16], &[0u8; 12], &[], &[]);
            assert!(ct.is_empty());
            assert_eq!(hex(&tag), "58e2fccefa7e3061367f1d57a4e7455a");
        }

        #[test]
        fn gcm_nist_case2_oneblock() {
            let (ct, tag) = seal(&[0u8; 16], &[0u8; 12], &[], &[0u8; 16]);
            assert_eq!(hex(&ct), "0388dace60b6a392f328c2b971b2fe78");
            assert_eq!(hex(&tag), "ab6e47d42cec13bdf53a67b21257bddf");
            assert_eq!(
                open(&[0u8; 16], &[0u8; 12], &[], &ct, &tag).unwrap(),
                vec![0u8; 16]
            );
        }

        // NIST GCM test case 4 — multi-block CTR + GHASH with AAD (exercises the
        // table over many blocks). The known answer pins the table-based GHASH.
        #[test]
        fn gcm_nist_case4_multiblock_aad() {
            let key = b16("feffe9928665731c6d6a8f9467308308");
            let iv = b12("cafebabefacedbaddecaf888");
            let pt = bytes(
                "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
            );
            let aad = bytes("feedfacedeadbeeffeedfacedeadbeefabaddad2");
            let (ct, tag) = seal(&key, &iv, &aad, &pt);
            assert_eq!(
                hex(&ct),
                "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091"
            );
            assert_eq!(hex(&tag), "5bc94fbc3221a5db94fae95ae7121a47");
            assert_eq!(open(&key, &iv, &aad, &ct, &tag).unwrap(), pt);
        }
    }
}
pub mod client {
    //! A minimal TLS 1.3 client over `std::net::TcpStream` — pure safe Rust.
    //!
    //! Cipher suite TLS_AES_128_GCM_SHA256, key exchange X25519. Enough to speak
    //! HTTPS to AWS S3 / minio. The key schedule and record layer follow RFC 8446.
    //!
    //! The AEAD key for each direction is a `GcmKey` (round keys + the H-multiply
    //! operator) built ONCE at the handshake; the bulk send/recv then seal/open every
    //! record as a parallel Web sweep (`Web.pmapAt`, schedule-free).
    //!
    //! SECURITY (labeled, Axiom XIII): completes the handshake and verifies the
    //! server Finished MAC (the key schedule is sound), but does NOT yet validate the
    //! certificate chain. No active-MITM protection — trusted routes only until cert
    //! validation lands.

    #![allow(dead_code)]

    use std::fs::File;
    use std::io::{self, Error, ErrorKind, Read, Write};
    use std::net::TcpStream;
    use std::os::unix::net::UnixStream;
    use std::thread;

    use std::time::{SystemTime, UNIX_EPOCH};

    use super::aes_gcm::GcmKey;
    use super::ecdsa;
    use super::x25519;
    use super::x509;
    use crate::sync::crypto::{hkdf_expand_label, hkdf_extract, hmac_sha256, sha256, sha512, Sha256};

    /// A client's own identity for TLS 1.3 mutual authentication (`CertificateRequest`)
    /// — what go-plugin's AutoMTLS handshake needs the host to present. P-521/
    /// ECDSA-SHA512 specifically (`ecdsa_secp521r1_sha512` = 0x0603), matching the
    /// real captured `PLUGIN_CLIENT_CERT`'s own algorithm.
    pub struct ClientIdentity {
        pub cert_der: Vec<u8>,
        pub private_key: Vec<u8>, // P-521 raw scalar, 66 bytes big-endian
    }

    fn rand_bytes(n: usize) -> io::Result<Vec<u8>> {
        let mut f = File::open("/dev/urandom")?;
        let mut b = vec![0u8; n];
        f.read_exact(&mut b)?;
        Ok(b)
    }

    fn u8len(body: &[u8]) -> Vec<u8> {
        let mut v = vec![body.len() as u8];
        v.extend_from_slice(body);
        v
    }
    fn u16len(body: &[u8]) -> Vec<u8> {
        let mut v = (body.len() as u16).to_be_bytes().to_vec();
        v.extend_from_slice(body);
        v
    }
    fn u24len(body: &[u8]) -> Vec<u8> {
        let l = body.len();
        let mut v = vec![(l >> 16) as u8, (l >> 8) as u8, l as u8];
        v.extend_from_slice(body);
        v
    }
    fn ext(t: u16, data: &[u8]) -> Vec<u8> {
        let mut v = t.to_be_bytes().to_vec();
        v.extend_from_slice(&u16len(data));
        v
    }

    fn nonce(iv: &[u8; 12], seq: u64) -> [u8; 12] {
        let mut n = *iv;
        let s = seq.to_be_bytes();
        for i in 0..8 {
            n[4 + i] ^= s[i];
        }
        n
    }

    fn a16(v: &[u8]) -> [u8; 16] {
        let mut a = [0u8; 16];
        a.copy_from_slice(&v[..16]);
        a
    }
    fn a12(v: &[u8]) -> [u8; 12] {
        let mut a = [0u8; 12];
        a.copy_from_slice(&v[..12]);
        a
    }
    fn a32(v: &[u8]) -> [u8; 32] {
        let mut a = [0u8; 32];
        a.copy_from_slice(&v[..32]);
        a
    }

    /// Transport-agnostic byte stream: go-plugin's own handshake line names
    /// EITHER `tcp` or `unix` as the network for a given provider connection (a
    /// per-provider, per-OS choice — go-plugin defaults to a Unix domain socket on
    /// Linux/macOS, TCP mainly as the Windows fallback), and AutoMTLS runs over
    /// either one identically (TLS is a byte-stream protocol, indifferent to what
    /// carries the bytes). One handshake implementation, boxed over whichever
    /// concrete stream `connect_with_identity`/`connect_unix_with_identity` opened.
    trait Sock: Read + Write + Send {}
    impl<T: Read + Write + Send> Sock for T {}

    pub struct TlsClient {
        sock: Box<dyn Sock>,
        host_hdr: String,
        c_gcm: GcmKey,
        c_iv: [u8; 12],
        c_seq: u64,
        s_gcm: GcmKey,
        s_iv: [u8; 12],
        s_seq: u64,
        inbuf: Vec<u8>,
        inpos: usize,
    }

    fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    impl TlsClient {
        fn read_record(sock: &mut dyn Sock) -> io::Result<(u8, Vec<u8>)> {
            let mut hdr = [0u8; 5];
            sock.read_exact(&mut hdr)?;
            let len = ((hdr[3] as usize) << 8) | hdr[4] as usize;
            let mut body = vec![0u8; len];
            sock.read_exact(&mut body)?;
            Ok((hdr[0], body))
        }

        fn send_record(sock: &mut dyn Sock, ct: u8, payload: &[u8]) -> io::Result<()> {
            let mut rec = vec![ct, 0x03, 0x03];
            rec.extend_from_slice(&(payload.len() as u16).to_be_bytes());
            rec.extend_from_slice(payload);
            sock.write_all(&rec)
        }

        fn send_encrypted(
            sock: &mut dyn Sock,
            gcm: &GcmKey,
            iv: &[u8; 12],
            seq: u64,
            inner: &[u8],
        ) -> io::Result<()> {
            Self::send_record(sock, 0x17, &seal_record(gcm, iv, seq, inner)[5..])
        }

        fn recv_decrypted(
            sock: &mut dyn Sock,
            gcm: &GcmKey,
            iv: &[u8; 12],
            seq: u64,
        ) -> io::Result<(u8, Vec<u8>)> {
            loop {
                let (ct, body) = Self::read_record(sock)?;
                if ct == 0x14 {
                    continue; // ChangeCipherSpec
                }
                if ct != 0x17 {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("unexpected record type {ct}"),
                    ));
                }
                return open_record(gcm, iv, seq, &body);
            }
        }

        /// Bulk send — the Web sweep: seal every record across all cores (each record
        /// an independent node, `pmapAt`/`preduce_schedule_free`), flush in order.
        pub fn send_bulk(&mut self, data: &[u8]) -> io::Result<()> {
            if data.is_empty() {
                return Ok(());
            }
            let recs: Vec<&[u8]> = data.chunks(16384).collect();
            let n = recs.len();
            let iv = self.c_iv;
            let base = self.c_seq;
            let p = thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(8)
                .min(n);
            let per = n.div_ceil(p);
            let recs_ref = &recs;
            let groups: Vec<Vec<Vec<u8>>> = {
                let gcm = &self.c_gcm;
                thread::scope(|s| {
                    let handles: Vec<_> = (0..p)
                        .map(|t| {
                            let lo = t * per;
                            let hi = ((t + 1) * per).min(n);
                            s.spawn(move || {
                                (lo..hi)
                                    .map(|k| {
                                        seal_record(
                                            gcm,
                                            &iv,
                                            base + k as u64,
                                            &with_type(recs_ref[k], 0x17),
                                        )
                                    })
                                    .collect::<Vec<Vec<u8>>>()
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap_or_default())
                        .collect()
                })
            };
            for g in groups {
                for rec in g {
                    self.sock.write_all(&rec)?;
                }
            }
            self.c_seq += n as u64;
            Ok(())
        }

        /// Bulk receive to EOF — the Web sweep: read every record (I/O), then decrypt
        /// across all cores, concatenate application data in order.
        pub fn recv_all(&mut self) -> io::Result<Vec<u8>> {
            let mut records: Vec<Vec<u8>> = Vec::new();
            loop {
                match Self::read_record(&mut self.sock) {
                    Ok((0x17, body)) => records.push(body),
                    Ok((_, _)) => {}
                    Err(ref e) if e.kind() == ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e),
                }
            }
            let n = records.len();
            let iv = self.s_iv;
            let base = self.s_seq;
            let p = thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(8)
                .min(n.max(1));
            let per = n.div_ceil(p.max(1));
            let recs_ref = &records;
            let groups: Vec<Vec<io::Result<(u8, Vec<u8>)>>> = {
                let gcm = &self.s_gcm;
                thread::scope(|s| {
                    let handles: Vec<_> = (0..p)
                        .map(|t| {
                            let lo = t * per;
                            let hi = ((t + 1) * per).min(n);
                            s.spawn(move || {
                                (lo..hi)
                                    .map(|k| open_record(gcm, &iv, base + k as u64, &recs_ref[k]))
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap_or_default())
                        .collect()
                })
            };
            self.s_seq += n as u64;
            let mut out = Vec::new();
            for g in groups {
                for r in g {
                    let (inner_ct, data) = r?;
                    match inner_ct {
                        0x17 => out.extend_from_slice(&data),
                        0x16 => {}
                        0x15 => return Ok(out),
                        other => {
                            return Err(Error::new(
                                ErrorKind::InvalidData,
                                format!("unexpected inner type {other}"),
                            ));
                        }
                    }
                }
            }
            Ok(out)
        }

        /// Same handshake, no client certificate presented — the S3/HTTPS path,
        /// unchanged from before mTLS support landed.
        pub fn connect(host: &str, port: u16, verify: bool) -> io::Result<TlsClient> {
            Self::connect_with_identity(host, port, verify, None)
        }

        /// Same handshake, over an already-connected Unix domain socket — go-plugin's
        /// default transport on Linux/macOS. `sni_host` is what goes in the
        /// ClientHello's `server_name` (the provider's cert SAN, e.g. `"localhost"`),
        /// independent of the socket path. Server-cert CA-chain validation is always
        /// skipped here (`verify=false`): go-plugin's own trust model pins the
        /// server's cert against the value in its handshake line, a channel already
        /// trusted (our own spawned child's stdout), not a public CA chain.
        pub fn connect_unix_with_identity(
            path: &str,
            sni_host: &str,
            identity: Option<&ClientIdentity>,
        ) -> io::Result<TlsClient> {
            let sock = UnixStream::connect(path)?;
            sock.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
            Self::handshake(sock, sni_host, 0, false, identity)
        }

        // The transcript-hash placeholders below are seeded then overwritten as the
        // handshake flight arrives; the seed write is intentional, not dead.
        #[allow(unused_assignments)]
        pub fn connect_with_identity(
            host: &str,
            port: u16,
            verify: bool,
            identity: Option<&ClientIdentity>,
        ) -> io::Result<TlsClient> {
            let sock = TcpStream::connect((host, port))?;
            sock.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
            Self::handshake(sock, host, port, verify, identity)
        }

        // The transcript-hash placeholders below are seeded then overwritten as the
        // handshake flight arrives; the seed write is intentional, not dead.
        #[allow(unused_assignments)]
        fn handshake<S: Sock + 'static>(
            mut sock: S,
            host: &str,
            port: u16,
            verify: bool,
            identity: Option<&ClientIdentity>,
        ) -> io::Result<TlsClient> {
            let priv_key = a32(&rand_bytes(32)?);
            let pubkey = x25519::base_mul(&priv_key);
            let client_random = rand_bytes(32)?;
            let session_id = rand_bytes(32)?;

            let mut exts = Vec::new();
            {
                let name = host.as_bytes();
                let mut entry = vec![0u8];
                entry.extend_from_slice(&u16len(name));
                exts.extend_from_slice(&ext(0x0000, &u16len(&entry)));
            }
            exts.extend_from_slice(&ext(0x000a, &u16len(&[0x00, 0x1d])));
            // signature_algorithms: schemes we can verify — ECDSA P256/P384/P521, RSA-PSS 256/384/512.
            // P521/SHA512 (0x0603) matters beyond "can we verify a P521 cert": RFC 8446 requires the
            // PEER to pick its OWN CertificateVerify scheme from THIS list — go-plugin's AutoMTLS server
            // cert is P-521/ecdsa-with-SHA512 (confirmed live), so omitting 0x0603 here left the server
            // with no valid scheme for its own signature, which is why it sent a bare `handshake_failure`
            // alert immediately after ClientHello rather than any more specific complaint.
            exts.extend_from_slice(&ext(
                0x000d,
                &u16len(&[
                    0x04, 0x03, 0x05, 0x03, 0x06, 0x03, 0x08, 0x04, 0x08, 0x05, 0x08, 0x06,
                ]),
            ));
            exts.extend_from_slice(&ext(0x002b, &u8len(&[0x03, 0x04])));
            // ALPN (RFC 7301): only when presenting a client identity — currently the
            // go-plugin/gRPC path exclusively, which REQUIRES ALPN negotiating "h2"
            // (RFC 7540 §3.1). Measured live: Go's HTTP/2 server closes the connection
            // (a bare TCP RST on our next write, surfacing as "broken pipe") the moment
            // it realizes no ALPN was negotiated — omitting this left every subsequent
            // write past a successful TLS handshake failing. The plain `connect()` path
            // (S3/HTTPS, HTTP/1.1) sends none, unchanged, since ALPN-negotiating "h2"
            // there would make the SERVER expect an HTTP/2 preface this client never sends.
            if identity.is_some() {
                let mut proto_list = Vec::new();
                proto_list.extend_from_slice(&u8len(b"h2"));
                exts.extend_from_slice(&ext(0x0010, &u16len(&proto_list)));
            }
            {
                let mut share = vec![0x00, 0x1d];
                share.extend_from_slice(&u16len(&pubkey));
                exts.extend_from_slice(&ext(0x0033, &u16len(&share)));
            }
            let mut body = vec![0x03, 0x03];
            body.extend_from_slice(&client_random);
            body.extend_from_slice(&u8len(&session_id));
            body.extend_from_slice(&u16len(&[0x13, 0x01]));
            body.extend_from_slice(&u8len(&[0x00]));
            body.extend_from_slice(&u16len(&exts));
            let mut ch_msg = vec![0x01u8];
            ch_msg.extend_from_slice(&u24len(&body));

            let mut transcript = Sha256::new();
            transcript.update(&ch_msg);
            Self::send_record(&mut sock, 0x16, &ch_msg)?;

            let sh_msg = loop {
                let (ct, body) = Self::read_record(&mut sock)?;
                if ct == 0x14 {
                    continue;
                }
                if ct != 0x16 {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("expected ServerHello, got {ct}, body={body:02x?}"),
                    ));
                }
                break body;
            };
            transcript.update(&sh_msg);
            let server_pub = parse_server_hello(&sh_msg)?;
            let shared = x25519::x25519(&priv_key, &server_pub);

            let empty_hash = sha256(b"");
            let zero = [0u8; 32];
            let early = hkdf_extract(&zero, &zero);
            let derived = hkdf_expand_label(&early, "derived", &empty_hash, 32);
            let hs_secret = hkdf_extract(&derived, &shared);
            let th_sh = transcript.clone().finalize();
            let c_hs = hkdf_expand_label(&hs_secret, "c hs traffic", &th_sh, 32);
            let s_hs = hkdf_expand_label(&hs_secret, "s hs traffic", &th_sh, 32);
            let c_hs_gcm = GcmKey::new(&a16(&hkdf_expand_label(&c_hs, "key", b"", 16)));
            let c_hs_iv = a12(&hkdf_expand_label(&c_hs, "iv", b"", 12));
            let s_hs_gcm = GcmKey::new(&a16(&hkdf_expand_label(&s_hs, "key", b"", 16)));
            let s_hs_iv = a12(&hkdf_expand_label(&s_hs, "iv", b"", 12));

            // encrypted handshake flight: EE, [CertificateRequest], Cert, CertVerify, Finished
            let mut hs_buf: Vec<u8> = Vec::new();
            let mut s_seq = 0u64;
            let mut th_before_fin = [0u8; 32];
            let mut th_before_cv = [0u8; 32];
            let mut server_finished: Vec<u8> = Vec::new();
            let mut cert_msg: Vec<u8> = Vec::new();
            let mut cv_msg: Vec<u8> = Vec::new();
            let mut cert_request_context: Option<Vec<u8>> = None;
            'flight: loop {
                while !has_full_msg(&hs_buf) {
                    let (inner_ct, data) = Self::recv_decrypted(&mut sock, &s_hs_gcm, &s_hs_iv, s_seq)?;
                    s_seq += 1;
                    if inner_ct != 0x16 {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("non-handshake in flight: {inner_ct}"),
                        ));
                    }
                    hs_buf.extend_from_slice(&data);
                }
                while let Some((msg, rest)) = take_msg(&hs_buf) {
                    hs_buf = rest;
                    match msg[0] {
                        0x0d => {
                            // CertificateRequest: mTLS is being requested. Body is
                            // certificate_request_context (u8-len-prefixed) then extensions —
                            // echoed back verbatim in the client's own Certificate message.
                            let body = &msg[4..];
                            let ctx_len = *body.first().unwrap_or(&0) as usize;
                            cert_request_context =
                                Some(body.get(1..1 + ctx_len).unwrap_or(&[]).to_vec());
                            transcript.update(&msg);
                        }
                        0x0b => {
                            cert_msg = msg[4..].to_vec(); // Certificate body
                            transcript.update(&msg);
                        }
                        0x0f => {
                            th_before_cv = transcript.clone().finalize(); // signs CH..Certificate
                            cv_msg = msg[4..].to_vec(); // CertificateVerify body
                            transcript.update(&msg);
                        }
                        0x14 => {
                            th_before_fin = transcript.clone().finalize();
                            transcript.update(&msg);
                            server_finished = msg[4..].to_vec();
                            break 'flight;
                        }
                        _ => transcript.update(&msg),
                    }
                }
            }

            let s_fin_key = hkdf_expand_label(&s_hs, "finished", b"", 32);
            let expect = hmac_sha256(&s_fin_key, &th_before_fin);
            if server_finished.len() != 32 || server_finished != expect {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "server Finished verify_data mismatch",
                ));
            }

            // Certificate-chain validation (the MITM defense). Verify the chain to a
            // system trust anchor, the leaf hostname + dates, and that the leaf signed
            // this handshake (CertificateVerify). Skipped only when `verify` is false.
            if verify {
                let chain = parse_certificate_message(&cert_msg)?;
                let (scheme, sig) = parse_cert_verify(&cv_msg)?;
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                let trust = x509::TrustStore::system();
                let leaf = x509::validate(&chain, host, now, &trust).map_err(|e| {
                    Error::new(
                        ErrorKind::InvalidData,
                        format!("certificate validation failed: {e}"),
                    )
                })?;
                if !x509::verify_cert_verify(&leaf, scheme, &sig, &th_before_cv) {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "CertificateVerify signature invalid (key not proven)",
                    ));
                }
            }

            // Application traffic secrets always derive from the transcript through the
            // SERVER's Finished (RFC 8446 §7.1) — unaffected by whether the client goes
            // on to authenticate below.
            let th_sfin = transcript.clone().finalize();
            let derived2 = hkdf_expand_label(&hs_secret, "derived", &empty_hash, 32);
            let master = hkdf_extract(&derived2, &zero);
            let c_ap = hkdf_expand_label(&master, "c ap traffic", &th_sfin, 32);
            let s_ap = hkdf_expand_label(&master, "s ap traffic", &th_sfin, 32);
            let c_gcm = GcmKey::new(&a16(&hkdf_expand_label(&c_ap, "key", b"", 16)));
            let c_iv = a12(&hkdf_expand_label(&c_ap, "iv", b"", 12));
            let s_gcm = GcmKey::new(&a16(&hkdf_expand_label(&s_ap, "key", b"", 16)));
            let s_iv = a12(&hkdf_expand_label(&s_ap, "iv", b"", 12));

            Self::send_record(&mut sock, 0x14, &[0x01])?;

            let mut c_hs_seq = 0u64;
            // Mutual auth: send Certificate + CertificateVerify ONLY if the server
            // asked (CertificateRequest) AND we were given an identity to present —
            // go-plugin's AutoMTLS is exactly this case. The client's own Finished
            // then MACs over the transcript INCLUDING these two messages, not `th_sfin`
            // (RFC 8446 §4.4.4 — "Handshake Context" for the client's Finished includes
            // its own Certificate*/CertificateVerify* when sent).
            let th_for_client_finished =
                if let (Some(ctx), Some(id)) = (&cert_request_context, identity) {
                    let mut cert_body = u8len(ctx);
                    let entry = {
                        let mut e = u24len(&id.cert_der);
                        e.extend_from_slice(&[0x00, 0x00]); // per-cert extensions: none
                        e
                    };
                    cert_body.extend_from_slice(&u24len(&entry));
                    let mut cert_msg_out = vec![0x0bu8];
                    cert_msg_out.extend_from_slice(&u24len(&cert_body));
                    transcript.update(&cert_msg_out);
                    Self::send_encrypted(
                        &mut sock,
                        &c_hs_gcm,
                        &c_hs_iv,
                        c_hs_seq,
                        &with_type(&cert_msg_out, 0x16),
                    )?;
                    c_hs_seq += 1;

                    let th_before_client_cv = transcript.clone().finalize();
                    let mut signed = vec![0x20u8; 64];
                    signed.extend_from_slice(b"TLS 1.3, client CertificateVerify");
                    signed.push(0x00);
                    signed.extend_from_slice(&th_before_client_cv);
                    let (r, s) = ecdsa::p521().sign(&id.private_key, &sha512(&signed));
                    let mut ec_sig = vec![0x30u8]; // DER SEQUENCE { r INTEGER, s INTEGER }
                    let mut sig_body = Vec::new();
                    for part in [&r, &s] {
                        let mut v: Vec<u8> = part.iter().copied().skip_while(|&b| b == 0).collect();
                        if v.is_empty() {
                            v.push(0);
                        }
                        if v[0] & 0x80 != 0 {
                            v.insert(0, 0);
                        }
                        sig_body.push(0x02);
                        sig_body.push(v.len() as u8);
                        sig_body.extend_from_slice(&v);
                    }
                    ec_sig.push(sig_body.len() as u8);
                    ec_sig.extend_from_slice(&sig_body);

                    let mut cv_body = vec![0x06, 0x03]; // ecdsa_secp521r1_sha512
                    cv_body.extend_from_slice(&u16len(&ec_sig));
                    let mut cv_msg_out = vec![0x0fu8];
                    cv_msg_out.extend_from_slice(&u24len(&cv_body));
                    transcript.update(&cv_msg_out);
                    Self::send_encrypted(
                        &mut sock,
                        &c_hs_gcm,
                        &c_hs_iv,
                        c_hs_seq,
                        &with_type(&cv_msg_out, 0x16),
                    )?;
                    c_hs_seq += 1;

                    transcript.clone().finalize()
                } else {
                    th_sfin
                };

            let c_fin_key = hkdf_expand_label(&c_hs, "finished", b"", 32);
            let verify = hmac_sha256(&c_fin_key, &th_for_client_finished);
            let mut fin_msg = vec![0x14u8];
            fin_msg.extend_from_slice(&u24len(&verify));
            Self::send_encrypted(
                &mut sock,
                &c_hs_gcm,
                &c_hs_iv,
                c_hs_seq,
                &with_type(&fin_msg, 0x16),
            )?;

            let host_hdr = if port == 443 {
                host.to_string()
            } else {
                format!("{host}:{port}")
            };
            Ok(TlsClient {
                sock: Box::new(sock),
                host_hdr,
                c_gcm,
                c_iv,
                c_seq: 0,
                s_gcm,
                s_iv,
                s_seq: 0,
                inbuf: Vec::new(),
                inpos: 0,
            })
        }

        /// One HTTP/1.1 request over a PERSISTENT (keep-alive) connection: send, read
        /// exactly one Content-Length-bounded response, leave the connection open for
        /// the next request. The basis for the connection pool (raw per-object sync)
        /// and for retry/resume.
        pub fn request_one(
            &mut self,
            method: &str,
            path: &str,
            extra_headers: &[(String, String)],
            body: &[u8],
        ) -> io::Result<crate::sync::http::Response> {
            let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\n", self.host_hdr);
            for (k, v) in extra_headers {
                req.push_str(&format!("{k}: {v}\r\n"));
            }
            req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
            self.write_all(req.as_bytes())?;
            if !body.is_empty() {
                self.send_bulk(body)?;
            }
            self.flush()?;
            self.read_response()
        }

        fn read_response(&mut self) -> io::Result<crate::sync::http::Response> {
            let mut buf: Vec<u8> = Vec::new();
            let hdr_end = loop {
                if let Some(i) = find(&buf, b"\r\n\r\n") {
                    break i;
                }
                let mut tmp = [0u8; 8192];
                let n = Read::read(self, &mut tmp)?;
                if n == 0 {
                    return Err(Error::new(ErrorKind::UnexpectedEof, "no response headers"));
                }
                buf.extend_from_slice(&tmp[..n]);
            };
            let head = std::str::from_utf8(&buf[..hdr_end])
                .map_err(|_| Error::new(ErrorKind::InvalidData, "non-utf8 headers"))?;
            let mut lines = head.split("\r\n");
            let status_line = lines.next().unwrap_or("");
            let status = status_line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u16>().ok())
                .ok_or_else(|| {
                    Error::new(ErrorKind::InvalidData, format!("bad status: {status_line}"))
                })?;
            let mut headers = Vec::new();
            let mut clen: Option<usize> = None;
            let mut chunked = false;
            for l in lines {
                if let Some((k, v)) = l.split_once(':') {
                    let k = k.trim().to_string();
                    let v = v.trim().to_string();
                    if k.eq_ignore_ascii_case("content-length") {
                        clen = v.parse().ok();
                    }
                    if k.eq_ignore_ascii_case("transfer-encoding") && v.eq_ignore_ascii_case("chunked")
                    {
                        chunked = true;
                    }
                    headers.push((k, v));
                }
            }
            let mut body: Vec<u8> = buf[hdr_end + 4..].to_vec();
            if chunked {
                while find(&body, b"\r\n0\r\n\r\n").is_none() && find(&body, b"0\r\n\r\n") != Some(0) {
                    let mut tmp = [0u8; 8192];
                    let n = Read::read(self, &mut tmp)?;
                    if n == 0 {
                        break;
                    }
                    body.extend_from_slice(&tmp[..n]);
                }
                body = dechunk(&body);
            } else {
                let want = clen.unwrap_or(0);
                while body.len() < want {
                    let need = want - body.len();
                    let mut tmp = vec![0u8; need.min(65536)];
                    let n = Read::read(self, &mut tmp)?;
                    if n == 0 {
                        break;
                    }
                    body.extend_from_slice(&tmp[..n]);
                }
                body.truncate(want);
            }
            Ok(crate::sync::http::Response {
                status,
                headers,
                body,
            })
        }
    }

    fn dechunk(b: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut p = 0;
        while p < b.len() {
            let Some(nl) = find(&b[p..], b"\r\n") else {
                break;
            };
            let size = usize::from_str_radix(
                std::str::from_utf8(&b[p..p + nl])
                    .unwrap_or("0")
                    .trim()
                    .split(';')
                    .next()
                    .unwrap_or("0"),
                16,
            )
            .unwrap_or(0);
            p += nl + 2;
            if size == 0 || p + size > b.len() {
                break;
            }
            out.extend_from_slice(&b[p..p + size]);
            p += size + 2;
        }
        out
    }

    impl Write for TlsClient {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let chunk = buf.len().min(16384);
            let inner = with_type(&buf[..chunk], 0x17);
            TlsClient::send_encrypted(&mut self.sock, &self.c_gcm, &self.c_iv, self.c_seq, &inner)?;
            self.c_seq += 1;
            Ok(chunk)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.sock.flush()
        }
    }

    impl Read for TlsClient {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            while self.inpos >= self.inbuf.len() {
                let (inner_ct, data) = match TlsClient::recv_decrypted(
                    &mut self.sock,
                    &self.s_gcm,
                    &self.s_iv,
                    self.s_seq,
                ) {
                    Ok(v) => v,
                    Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(0),
                    Err(e) => return Err(e),
                };
                self.s_seq += 1;
                match inner_ct {
                    0x17 => {
                        self.inbuf = data;
                        self.inpos = 0;
                    }
                    0x16 => continue,
                    // A TLS alert from the peer (fatal or warning) surfaces as a clean EOF to the
                    // caller — same as `UnexpectedEof` above — rather than a raw decode error, since
                    // "the peer told us it's closing" and "the socket just closed" are both properly
                    // read()-returns-0 conditions, not I/O failures.
                    0x15 => return Ok(0),
                    other => {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("unexpected inner type {other}"),
                        ));
                    }
                }
            }
            let n = (self.inbuf.len() - self.inpos).min(buf.len());
            buf[..n].copy_from_slice(&self.inbuf[self.inpos..self.inpos + n]);
            self.inpos += n;
            Ok(n)
        }
    }

    /// inner = payload || content_type (TLS 1.3 records carry the type as a trailer).
    fn with_type(payload: &[u8], ct: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(payload.len() + 1);
        v.extend_from_slice(payload);
        v.push(ct);
        v
    }

    /// Seal `inner` (already ending in its content-type byte) into a full record.
    fn seal_record(gcm: &GcmKey, iv: &[u8; 12], seq: u64, inner: &[u8]) -> Vec<u8> {
        let total = inner.len() + 16;
        let aad = [0x17u8, 0x03, 0x03, (total >> 8) as u8, total as u8];
        let (ct, tag) = gcm.seal(&nonce(iv, seq), &aad, inner);
        let mut rec = aad.to_vec();
        rec.extend_from_slice(&ct);
        rec.extend_from_slice(&tag);
        rec
    }

    /// Open one record body → (inner content type, plaintext).
    fn open_record(gcm: &GcmKey, iv: &[u8; 12], seq: u64, body: &[u8]) -> io::Result<(u8, Vec<u8>)> {
        if body.len() < 16 {
            return Err(Error::new(ErrorKind::InvalidData, "short record"));
        }
        let aad = [
            0x17u8,
            0x03,
            0x03,
            (body.len() >> 8) as u8,
            body.len() as u8,
        ];
        let (cipher, tag) = body.split_at(body.len() - 16);
        let pt = gcm
            .open(&nonce(iv, seq), &aad, cipher, &a16(tag))
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "AEAD tag mismatch"))?;
        let mut end = pt.len();
        while end > 0 && pt[end - 1] == 0 {
            end -= 1;
        }
        if end == 0 {
            return Err(Error::new(ErrorKind::InvalidData, "empty inner record"));
        }
        Ok((pt[end - 1], pt[..end - 1].to_vec()))
    }

    fn u24(b: &[u8]) -> usize {
        ((b[0] as usize) << 16) | ((b[1] as usize) << 8) | b[2] as usize
    }

    /// Parse the TLS 1.3 Certificate message body → the chain of DER certs (leaf first).
    fn parse_certificate_message(body: &[u8]) -> io::Result<Vec<Vec<u8>>> {
        let err = || Error::new(ErrorKind::InvalidData, "malformed Certificate message");
        let ctx_len = *body.first().ok_or_else(err)? as usize;
        let mut p = 1 + ctx_len;
        if p + 3 > body.len() {
            return Err(err());
        }
        p += 3; // certificate_list length
        let mut certs = Vec::new();
        while p + 3 <= body.len() {
            let clen = u24(&body[p..]);
            p += 3;
            if p + clen > body.len() {
                break;
            }
            certs.push(body[p..p + clen].to_vec());
            p += clen;
            if p + 2 > body.len() {
                break;
            }
            let elen = ((body[p] as usize) << 8) | body[p + 1] as usize;
            p += 2 + elen; // per-cert extensions
        }
        if certs.is_empty() {
            return Err(err());
        }
        Ok(certs)
    }

    /// Parse the CertificateVerify body → (SignatureScheme, signature).
    fn parse_cert_verify(body: &[u8]) -> io::Result<(u16, Vec<u8>)> {
        if body.len() < 4 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "short CertificateVerify",
            ));
        }
        let scheme = ((body[0] as u16) << 8) | body[1] as u16;
        let slen = ((body[2] as usize) << 8) | body[3] as usize;
        let sig = body
            .get(4..4 + slen)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "bad CertVerify sig"))?
            .to_vec();
        Ok((scheme, sig))
    }

    fn has_full_msg(buf: &[u8]) -> bool {
        if buf.len() < 4 {
            return false;
        }
        let len = ((buf[1] as usize) << 16) | ((buf[2] as usize) << 8) | buf[3] as usize;
        buf.len() >= 4 + len
    }

    fn take_msg(buf: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        if !has_full_msg(buf) {
            return None;
        }
        let len = ((buf[1] as usize) << 16) | ((buf[2] as usize) << 8) | buf[3] as usize;
        let end = 4 + len;
        Some((buf[..end].to_vec(), buf[end..].to_vec()))
    }

    fn parse_server_hello(msg: &[u8]) -> io::Result<[u8; 32]> {
        let err = || Error::new(ErrorKind::InvalidData, "malformed ServerHello");
        let b = msg.get(4..).ok_or_else(err)?;
        let mut p = 2 + 32;
        let sid_len = *b.get(p).ok_or_else(err)? as usize;
        p += 1 + sid_len;
        let cipher = ((*b.get(p).ok_or_else(err)? as u16) << 8) | *b.get(p + 1).ok_or_else(err)? as u16;
        if cipher != 0x1301 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("server cipher 0x{cipher:04x}, want 0x1301"),
            ));
        }
        p += 2 + 1;
        let ext_len =
            ((*b.get(p).ok_or_else(err)? as usize) << 8) | *b.get(p + 1).ok_or_else(err)? as usize;
        p += 2;
        let end = p + ext_len;
        while p + 4 <= end {
            let et = ((b[p] as u16) << 8) | b[p + 1] as u16;
            let el = ((b[p + 2] as usize) << 8) | b[p + 3] as usize;
            p += 4;
            let data = b.get(p..p + el).ok_or_else(err)?;
            if et == 0x0033 {
                if data.len() < 4 {
                    return Err(err());
                }
                let klen = ((data[2] as usize) << 8) | data[3] as usize;
                let key = data.get(4..4 + klen).ok_or_else(err)?;
                if klen != 32 {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "server key_share not x25519/32",
                    ));
                }
                return Ok(a32(key));
            }
            p += el;
        }
        Err(Error::new(
            ErrorKind::InvalidData,
            "no key_share in ServerHello",
        ))
    }

    #[cfg(test)]
    mod live_check {
        use super::*;
        use std::io::BufRead;
        use std::process::{Command, Stdio};

        /// The full AutoMTLS chain end to end against a REAL provider binary:
        /// generate our own P-521 identity, spawn the provider with the real
        /// go-plugin handshake env vars (magic cookie captured live from `tofu`
        /// itself), read its handshake line, connect over WHICHEVER transport it
        /// names (measured live: go-plugin defaults to a Unix domain socket on
        /// Linux even with AutoMTLS active — `tcp` is not assumed), and complete
        /// the mutual-TLS 1.3 handshake. No gRPC/HTTP2 yet — this proves the
        /// crypto + cert + mTLS-presentation layer alone talks to the real thing,
        /// which no synthetic unit test can substitute for. Point
        /// `SYNCFOLD_TEST_PROVIDER_BIN` at a provider that actually implements
        /// AutoMTLS (measured live: `hashicorp/random` does; `hashicorp/null`
        /// does not — it always answers as plain v5/unix regardless of the env).
        #[test]
        #[ignore = "spawns a real provider binary and hits its live TLS port — run explicitly"]
        fn automtls_handshake_completes_against_a_real_provider() {
            let provider = std::env::var("SYNCFOLD_TEST_PROVIDER_BIN").unwrap_or_else(|_| {
                unreachable!("set SYNCFOLD_TEST_PROVIDER_BIN to an AutoMTLS-capable provider path")
            });

            let curve = ecdsa::p521();
            let (d, qx, qy) = curve
                .generate_keypair()
                .unwrap_or_else(|e| unreachable!("keygen: {e}"));
            let der = x509::build_self_signed_p521(
                &d,
                &qx,
                &qy,
                "localhost",
                &[0x2a],
                1_700_000_000,
                2_100_000_000,
            );
            let pem = x509::to_pem(&der);
            let identity = ClientIdentity {
                cert_der: der,
                private_key: d,
            };

            let mut child = Command::new(&provider)
                .env(
                    "TF_PLUGIN_MAGIC_COOKIE",
                    "d602bf8f470bc67ca7faa0386276bbdd4330efaf76d1a219cb4d6991ca9872b2",
                )
                .env("PLUGIN_PROTOCOL_VERSIONS", "6")
                .env("PLUGIN_MIN_PORT", "10000")
                .env("PLUGIN_MAX_PORT", "25000")
                .env("PLUGIN_CLIENT_CERT", &pem)
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .unwrap_or_else(|e| unreachable!("spawn provider: {e}"));

            let stdout = child
                .stdout
                .take()
                .unwrap_or_else(|| unreachable!("no stdout"));
            let mut lines = std::io::BufReader::new(stdout).lines();
            // Bounded, not an infinite loop: a handful of debug JSON lines can precede
            // the real handshake line, but if it never shows up that's a real failure
            // to surface, not something to hang on.
            let mut handshake = None;
            for _ in 0..20 {
                let Some(Ok(line)) = lines.next() else { break };
                if line.starts_with("1|") && line.matches('|').count() == 5 {
                    handshake = Some(line);
                    break;
                }
            }
            let handshake =
                handshake.unwrap_or_else(|| unreachable!("no handshake line seen from provider"));

            let fields: Vec<&str> = handshake.split('|').collect();
            let network = fields[2];
            let addr = fields[3];
            let server_cert_b64 = fields[5];
            assert!(
                !server_cert_b64.is_empty(),
                "AutoMTLS must return a server cert in the handshake line, got: {handshake}"
            );

            let result = match network {
                "unix" => TlsClient::connect_unix_with_identity(addr, "localhost", Some(&identity)),
                "tcp" => {
                    let (host, port_str) = addr
                        .rsplit_once(':')
                        .unwrap_or_else(|| unreachable!("bad addr: {addr}"));
                    let port: u16 = port_str
                        .parse()
                        .unwrap_or_else(|e| unreachable!("bad port {port_str}: {e}"));
                    let connect_host = if host == "127.0.0.1" {
                        "localhost"
                    } else {
                        host
                    };
                    TlsClient::connect_with_identity(connect_host, port, false, Some(&identity))
                }
                other => unreachable!("unexpected network type in handshake line: {other}"),
            };

            let _ = child.kill();
            let _ = child.wait();

            match result {
                Ok(_) => {}
                Err(e) => unreachable!("AutoMTLS handshake against the real provider failed: {e}"),
            }
        }
    }
}
pub mod ecdsa {
    //! ECDSA over the NIST prime curves P-256 and P-384 — pure safe Rust, what R2's
    //! certificate chain needs (the leaf/intermediate are ECDSA P-256, the root
    //! P-384) to verify, and what go-plugin's AutoMTLS handshake (tfplugin6) needs
    //! to sign: a self-signed ephemeral cert, then a `CertificateVerify` proving
    //! possession of that same key during the TLS handshake. Same generic
    //! fixed-width bignum + Jacobian point arithmetic serves both; verify is KAT'd
    //! against the RFC 6979 deterministic-ECDSA test vectors for both curves, and
    //! `sign` reuses the identical `mul`/`modinv` primitives `verify` already proved
    //! correct — round-trip-verified (sign then verify) rather than against an
    //! external vector, since the private key for RFC 6979's published vectors
    //! isn't itself published.
    //!
    //! Not constant-time (verification is over public values, so timing is moot;
    //! signing uses RFC 6979 deterministic k-generation, which is naturally
    //! branch-free in the HMAC steps, but the scalar multiplication below is the
    //! same non-constant-time ladder as `verify`'s — fine for an ephemeral
    //! handshake cert with a one-connection lifetime, not fine for a long-lived
    //! signing key subject to timing side channels).

    #![allow(dead_code)]

    const L: usize = 9; // 576-bit limbs — covers P-521 (521, go-plugin AutoMTLS's curve) with margin.
                        // Was 8 (512-bit, "covers P-384 with margin") until the real go-plugin AutoMTLS handshake cert
                        // (captured live from `tofu`, see PLUGIN_CLIENT_CERT) turned out to be P-521/ecdsa-with-SHA512, not
                        // P-256 — confirmed by decoding its DER (curve OID 1.3.132.0.35). Every arithmetic op below (`add`/
                        // `sub`/`mulmod`/`mul`/…) is limb-count-generic, so this widening changes nothing for P-256/P-384 —
                        // verified by the existing RFC 6979 KATs for both still passing unchanged after this edit.
    type Big = [u64; L];

    const ZERO: Big = [0; L];
    fn one() -> Big {
        let mut b = ZERO;
        b[0] = 1;
        b
    }

    fn from_hex(s: &str) -> Big {
        let mut b = ZERO;
        let bytes: Vec<u8> = (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect();
        from_be(&bytes, &mut b);
        b
    }

    fn from_be(bytes: &[u8], out: &mut Big) {
        *out = ZERO;
        // big-endian bytes → little-endian limbs
        for (i, &byte) in bytes.iter().rev().enumerate() {
            out[i / 8] |= (byte as u64) << ((i % 8) * 8);
        }
    }

    pub fn big_from_be(bytes: &[u8]) -> Big {
        let mut b = ZERO;
        from_be(bytes, &mut b);
        b
    }

    fn to_be32(a: &Big, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        for i in 0..n {
            out[n - 1 - i] = (a[i / 8] >> ((i % 8) * 8)) as u8;
        }
        out
    }

    fn cmp(a: &Big, b: &Big) -> core::cmp::Ordering {
        for i in (0..L).rev() {
            if a[i] != b[i] {
                return a[i].cmp(&b[i]);
            }
        }
        core::cmp::Ordering::Equal
    }
    fn is_zero(a: &Big) -> bool {
        a.iter().all(|&x| x == 0)
    }
    fn bit(a: &Big, i: usize) -> bool {
        i < L * 64 && (a[i / 64] >> (i % 64)) & 1 == 1
    }
    fn bit_len(a: &Big) -> usize {
        for i in (0..L * 64).rev() {
            if bit(a, i) {
                return i + 1;
            }
        }
        0
    }
    fn add(a: &Big, b: &Big) -> Big {
        let mut r = ZERO;
        let mut carry = 0u128;
        for i in 0..L {
            let s = a[i] as u128 + b[i] as u128 + carry;
            r[i] = s as u64;
            carry = s >> 64;
        }
        r
    }
    fn sub(a: &Big, b: &Big) -> Big {
        // a >= b
        let mut r = ZERO;
        let mut borrow = 0i128;
        for i in 0..L {
            let d = a[i] as i128 - b[i] as i128 - borrow;
            if d < 0 {
                r[i] = (d + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                r[i] = d as u64;
                borrow = 0;
            }
        }
        r
    }
    fn addmod(a: &Big, b: &Big, m: &Big) -> Big {
        let s = add(a, b);
        if cmp(&s, m) != core::cmp::Ordering::Less {
            sub(&s, m)
        } else {
            s
        }
    }
    fn submod(a: &Big, b: &Big, m: &Big) -> Big {
        if cmp(a, b) != core::cmp::Ordering::Less {
            sub(a, b)
        } else {
            add(&sub(m, b), a)
        }
    }
    fn mulmod(a: &Big, b: &Big, m: &Big) -> Big {
        let mut r = ZERO;
        let mut aa = *a;
        let bits = bit_len(b);
        for i in 0..bits {
            if bit(b, i) {
                r = addmod(&r, &aa, m);
            }
            aa = addmod(&aa, &aa, m);
        }
        r
    }
    fn modexp(a: &Big, e: &Big, m: &Big) -> Big {
        let mut r = one();
        let mut base = *a;
        let bits = bit_len(e);
        for i in 0..bits {
            if bit(e, i) {
                r = mulmod(&r, &base, m);
            }
            base = mulmod(&base, &base, m);
        }
        r
    }
    fn modinv(a: &Big, m: &Big) -> Big {
        // Fermat: a^(m-2) mod m, valid for prime m
        let mut two = ZERO;
        two[0] = 2;
        modexp(a, &sub(m, &two), m)
    }

    /// A short-Weierstrass curve y² = x³ + a·x + b mod p, with order n and base G.
    pub struct Curve {
        p: Big,
        a: Big,
        b: Big,
        n: Big,
        gx: Big,
        gy: Big,
        pub bytes: usize, // field size in bytes (32 for P-256, 48 for P-384)
    }

    #[derive(Clone, Copy)]
    struct Pt {
        x: Big,
        y: Big,
        z: Big, // Jacobian; z==0 is the point at infinity
    }
    fn infinity() -> Pt {
        Pt {
            x: one(),
            y: one(),
            z: ZERO,
        }
    }

    impl Curve {
        fn dbl(&self, q: &Pt) -> Pt {
            if is_zero(&q.z) || is_zero(&q.y) {
                return infinity();
            }
            let m = &self.p;
            let a = mulmod(&q.y, &q.y, m); // A = Y^2
            let b = mulmod(&addmod(&addmod(&a, &a, m), &addmod(&a, &a, m), m), &q.x, m); // 4*X*A
            let a2 = mulmod(&a, &a, m);
            let c4 = addmod(&addmod(&a2, &a2, m), &addmod(&a2, &a2, m), m); // 4*A^2
            let c = addmod(&c4, &c4, m); // 8*A^2
            let x2 = mulmod(&q.x, &q.x, m);
            let three_x2 = addmod(&addmod(&x2, &x2, m), &x2, m);
            let z2 = mulmod(&q.z, &q.z, m);
            let z4 = mulmod(&z2, &z2, m);
            let d = addmod(&three_x2, &mulmod(&self.a, &z4, m), m); // D = 3X^2 + a*Z^4
            let x3 = submod(&mulmod(&d, &d, m), &addmod(&b, &b, m), m); // D^2 - 2B
            let y3 = submod(&mulmod(&d, &submod(&b, &x3, m), m), &c, m); // D*(B-X3) - C
            let z3 = mulmod(&addmod(&q.y, &q.y, m), &q.z, m); // 2*Y*Z
            Pt {
                x: x3,
                y: y3,
                z: z3,
            }
        }

        fn add(&self, p1: &Pt, p2: &Pt) -> Pt {
            if is_zero(&p1.z) {
                return *p2;
            }
            if is_zero(&p2.z) {
                return *p1;
            }
            let m = &self.p;
            let z12 = mulmod(&p1.z, &p1.z, m);
            let z2_2 = mulmod(&p2.z, &p2.z, m);
            let u1 = mulmod(&p1.x, &z2_2, m);
            let u2 = mulmod(&p2.x, &z12, m);
            let s1 = mulmod(&p1.y, &mulmod(&z2_2, &p2.z, m), m);
            let s2 = mulmod(&p2.y, &mulmod(&z12, &p1.z, m), m);
            if cmp(&u1, &u2) == core::cmp::Ordering::Equal {
                if cmp(&s1, &s2) == core::cmp::Ordering::Equal {
                    return self.dbl(p1);
                }
                return infinity();
            }
            let h = submod(&u2, &u1, m);
            let r = submod(&s2, &s1, m);
            let h2 = mulmod(&h, &h, m);
            let h3 = mulmod(&h2, &h, m);
            let u1h2 = mulmod(&u1, &h2, m);
            let x3 = submod(
                &submod(&mulmod(&r, &r, m), &h3, m),
                &addmod(&u1h2, &u1h2, m),
                m,
            );
            let y3 = submod(
                &mulmod(&r, &submod(&u1h2, &x3, m), m),
                &mulmod(&s1, &h3, m),
                m,
            );
            let z3 = mulmod(&mulmod(&h, &p1.z, m), &p2.z, m);
            Pt {
                x: x3,
                y: y3,
                z: z3,
            }
        }

        fn mul(&self, k: &Big, q: &Pt) -> Pt {
            let mut r = infinity();
            for i in (0..bit_len(k)).rev() {
                r = self.dbl(&r);
                if bit(k, i) {
                    r = self.add(&r, q);
                }
            }
            r
        }

        /// Affine x of a Jacobian point (x = X / Z²), or None at infinity.
        fn affine_x(&self, q: &Pt) -> Option<Big> {
            if is_zero(&q.z) {
                return None;
            }
            let zi = modinv(&q.z, &self.p);
            let zi2 = mulmod(&zi, &zi, &self.p);
            Some(mulmod(&q.x, &zi2, &self.p))
        }

        /// Whether affine `(x, y)` (field-width big-endian bytes) satisfies
        /// `y² = x³ + a·x + b (mod p)` — an independent sanity check on a curve's
        /// own parameters (used to cross-validate a real, externally-sourced public
        /// key against transcribed constants), distinct from `verify`'s signature
        /// equation.
        pub fn is_on_curve(&self, x: &[u8], y: &[u8]) -> bool {
            let x = big_from_be(x);
            let y = big_from_be(y);
            let lhs = mulmod(&y, &y, &self.p);
            let x3 = mulmod(&mulmod(&x, &x, &self.p), &x, &self.p);
            let rhs = addmod(
                &addmod(&x3, &mulmod(&self.a, &x, &self.p), &self.p),
                &self.b,
                &self.p,
            );
            cmp(&lhs, &rhs) == core::cmp::Ordering::Equal
        }

        /// Verify `(r, s)` over `hash` for public key `(qx, qy)` (raw field-width byte
        /// arrays). Returns true iff valid.
        pub fn verify(&self, qx: &[u8], qy: &[u8], hash: &[u8], r: &[u8], s: &[u8]) -> bool {
            let r = big_from_be(r);
            let s = big_from_be(s);
            if is_zero(&r) || cmp(&r, &self.n) != core::cmp::Ordering::Less {
                return false;
            }
            if is_zero(&s) || cmp(&s, &self.n) != core::cmp::Ordering::Less {
                return false;
            }
            // e = leftmost min(hashbits, fieldbits) of the hash, as an integer
            let take = self.bytes.min(hash.len());
            let mut e = big_from_be(&hash[..take]);
            if cmp(&e, &self.n) != core::cmp::Ordering::Less {
                e = sub(&e, &self.n); // n ≤ e < 2n at most for our sizes
            }
            let w = modinv(&s, &self.n);
            let u1 = mulmod(&e, &w, &self.n);
            let u2 = mulmod(&r, &w, &self.n);
            let g = Pt {
                x: self.gx,
                y: self.gy,
                z: one(),
            };
            let q = Pt {
                x: big_from_be(qx),
                y: big_from_be(qy),
                z: one(),
            };
            let p1 = self.mul(&u1, &g);
            let p2 = self.mul(&u2, &q);
            let rpt = self.add(&p1, &p2);
            match self.affine_x(&rpt) {
                None => false,
                Some(mut x) => {
                    if cmp(&x, &self.n) != core::cmp::Ordering::Less {
                        x = sub(&x, &self.n);
                    }
                    cmp(&x, &r) == core::cmp::Ordering::Equal
                }
            }
        }

        /// A fresh, uniformly-random key pair `(private, Qx, Qy)` — rejection-sampled
        /// from `/dev/urandom` (the same direct-syscall source `client.rs` already
        /// uses for TLS's own ephemeral X25519 keys) so `1 <= private < n` with no
        /// modular bias. Field-width big-endian bytes throughout.
        pub fn generate_keypair(&self) -> std::io::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
            let g = Pt {
                x: self.gx,
                y: self.gy,
                z: one(),
            };
            loop {
                let raw = rand_bytes(self.bytes)?;
                let d = big_from_be(&raw);
                if is_zero(&d) || cmp(&d, &self.n) != core::cmp::Ordering::Less {
                    continue; // reject 0 and out-of-range draws, try again
                }
                let q = self.mul(&d, &g);
                let Some(qx) = self.affine_x(&q) else {
                    continue;
                };
                // affine y: recompute from the same Jacobian point via z-inverse cubed
                let zi = modinv(&q.z, &self.p);
                let zi3 = mulmod(&mulmod(&zi, &zi, &self.p), &zi, &self.p);
                let qy = mulmod(&q.y, &zi3, &self.p);
                return Ok((
                    to_be32(&d, self.bytes),
                    to_be32(&qx, self.bytes),
                    to_be32(&qy, self.bytes),
                ));
            }
        }

        /// Sign `hash` with private key `d` (field-width big-endian bytes), returning
        /// field-width `(r, s)`. Nonce `k` is RFC 6979 deterministic (HMAC-DRBG,
        /// §3.2, HMAC-SHA256 or HMAC-SHA512 dispatched by `hash.len()`) — no fresh
        /// randomness needed per signature, and a repeated `(d, hash)` pair always
        /// yields the identical `(r, s)`, the property this module's own test suite
        /// checks it against.
        pub fn sign(&self, d: &[u8], hash: &[u8]) -> (Vec<u8>, Vec<u8>) {
            let dd = big_from_be(d);
            let take = self.bytes.min(hash.len());
            let mut h = big_from_be(&hash[..take]);
            if cmp(&h, &self.n) != core::cmp::Ordering::Less {
                h = sub(&h, &self.n);
            }
            let hmac_bytes = if hash.len() > 32 { 64 } else { 32 };
            let g = Pt {
                x: self.gx,
                y: self.gy,
                z: one(),
            };
            loop {
                let k = rfc6979_k(&dd, &h, &self.n, self.bytes, hmac_bytes);
                let r_pt = self.mul(&k, &g);
                let Some(mut rx) = self.affine_x(&r_pt) else {
                    continue;
                };
                if cmp(&rx, &self.n) != core::cmp::Ordering::Less {
                    rx = sub(&rx, &self.n);
                }
                if is_zero(&rx) {
                    continue; // vanishingly rare; RFC 6979 §3.2's own retry loop covers it too
                }
                let kinv = modinv(&k, &self.n);
                let s = mulmod(
                    &kinv,
                    &addmod(&h, &mulmod(&rx, &dd, &self.n), &self.n),
                    &self.n,
                );
                if is_zero(&s) {
                    continue;
                }
                return (to_be32(&rx, self.bytes), to_be32(&s, self.bytes));
            }
        }
    }

    fn rand_bytes(n: usize) -> std::io::Result<Vec<u8>> {
        use std::io::Read;
        let mut f = std::fs::File::open("/dev/urandom")?;
        let mut b = vec![0u8; n];
        f.read_exact(&mut b)?;
        Ok(b)
    }

    /// RFC 6979 §3.2's deterministic nonce (HMAC-DRBG), dispatched to HMAC-SHA256
    /// or HMAC-SHA512 by the hash's own length — the two combinations this codebase
    /// needs (P-256/SHA-256 for the existing verify KATs' shape, P-521/SHA-512 for
    /// go-plugin's AutoMTLS cert, `ecdsa-with-SHA512` per its real captured cert).
    /// `V`/`K` are sized to the HASH's own output (`hmac_bytes` — 32 or 64), per
    /// spec; `int2octets(d)`/`bits2octets(h1)` are sized to the FIELD/order width
    /// (`bytes` — 32 for P-256, 66 for P-521) — the two are the same length only by
    /// coincidence for P-256.
    ///
    /// One honest, labelled simplification from strict RFC 6979: `bits2int` on the
    /// generated `T` should keep only the leftmost `qlen` bits (521 for P-521, 7
    /// bits short of the 528-bit/66-byte `rlen` this builds), not the full byte
    /// string. Skipping that last bit-shift makes `k` deterministic and uniformly
    /// distributed over a very slightly different (still overwhelmingly close to
    /// uniform on `[1, n-1]` after the existing reject-and-retry) generator than a
    /// byte-exact RFC 6979 reference implementation would produce — it does NOT
    /// weaken the signature: any `k` this loop returns is checked `< n` before use
    /// (same as the caller's own reduction), so every signature it produces is
    /// fully valid and verifies correctly (checked directly against `verify`,
    /// `sign_then_verify_round_trips_for_generated_keys`) and still deterministic
    /// per `(d, hash)` (checked directly, `rfc6979_nonce_is_deterministic`) — it
    /// just won't reproduce an external RFC 6979 P-521 test vector byte-for-byte.
    fn rfc6979_k(d: &Big, h1: &Big, n: &Big, bytes: usize, hmac_bytes: usize) -> Big {
        fn hmac(hmac_bytes: usize, key: &[u8], msg: &[u8]) -> Vec<u8> {
            if hmac_bytes == 64 {
                crate::sync::crypto::hmac_sha512(key, msg).to_vec()
            } else {
                crate::sync::crypto::hmac_sha256(key, msg).to_vec()
            }
        }
        let x_bytes = to_be32(d, bytes);
        let h_bytes = to_be32(h1, bytes); // h1 already reduced mod n by the caller (`bits2octets`)
        let mut v = vec![0x01u8; hmac_bytes];
        let mut k = vec![0x00u8; hmac_bytes];
        let mut msg = v.clone();
        msg.push(0x00);
        msg.extend_from_slice(&x_bytes);
        msg.extend_from_slice(&h_bytes);
        k = hmac(hmac_bytes, &k, &msg);
        v = hmac(hmac_bytes, &k, &v);
        let mut msg2 = v.clone();
        msg2.push(0x01);
        msg2.extend_from_slice(&x_bytes);
        msg2.extend_from_slice(&h_bytes);
        k = hmac(hmac_bytes, &k, &msg2);
        v = hmac(hmac_bytes, &k, &v);
        loop {
            let mut t = Vec::new();
            while t.len() < bytes {
                v = hmac(hmac_bytes, &k, &v);
                t.extend_from_slice(&v);
            }
            let candidate = big_from_be(&t[..bytes]);
            if !is_zero(&candidate) && cmp(&candidate, n) == core::cmp::Ordering::Less {
                return candidate;
            }
            let mut msg3 = v.clone();
            msg3.push(0x00);
            k = hmac(hmac_bytes, &k, &msg3);
            v = hmac(hmac_bytes, &k, &v);
        }
    }

    pub fn p256() -> Curve {
        Curve {
            p: from_hex("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff"),
            a: from_hex("ffffffff00000001000000000000000000000000fffffffffffffffffffffffc"),
            b: from_hex("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b"),
            n: from_hex("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"),
            gx: from_hex("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"),
            gy: from_hex("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"),
            bytes: 32,
        }
    }

    pub fn p384() -> Curve {
        Curve {
            p: from_hex(
                "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff",
            ),
            a: from_hex(
                "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000fffffffc",
            ),
            b: from_hex(
                "b3312fa7e23ee7e4988e056be3f82d19181d9c6efe8141120314088f5013875ac656398d8a2ed19d2a85c8edd3ec2aef",
            ),
            n: from_hex(
                "ffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52973",
            ),
            gx: from_hex(
                "aa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741e082542a385502f25dbf55296c3a545e3872760ab7",
            ),
            gy: from_hex(
                "3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f",
            ),
            bytes: 48,
        }
    }

    /// P-521 (secp521r1) — go-plugin's AutoMTLS curve (confirmed live: the real
    /// `PLUGIN_CLIENT_CERT` `tofu` hands its providers is `ecdsa-with-SHA512` over
    /// this curve, decoded from its actual DER — curve OID `1.3.132.0.35`).
    /// Parameters transcribed programmatically (never retyped by hand) from
    /// RustCrypto's own `p521` crate (`NIST SP 800-186 §3.2.1.5`), and cross-checked
    /// two ways before use: the generator satisfies `y² = x³ - 3x + b (mod p)`, and
    /// the real captured certificate's own public key point ALSO lies on this exact
    /// curve — both checked programmatically, not eyeballed.
    pub fn p521() -> Curve {
        Curve {
            p: from_hex("01ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
            a: from_hex("01fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc"),
            b: from_hex("51953eb9618e1c9a1f929a21a0b68540eea2da725b99b315f3b8b489918ef109e156193951ec7e937b1652c0bd3bb1bf073573df883d2c34f1ef451fd46b503f00"),
            n: from_hex("01fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa51868783bf2f966b7fcc0148f709a5d03bb5c9b8899c47aebb6fb71e91386409"),
            gx: from_hex("c6858e06b70404e9cd9e3ecb662395b4429c648139053fb521f828af606b4d3dbaa14b5e77efe75928fe1dc127a2ffa8de3348b3c1856a429bf97e7e31c2e5bd66"),
            gy: from_hex("011839296a789a3bc0045c8a5fb42c7d1bd998f54449579b446817afbd17273e662c97ee72995ef42640c550b9013fad0761353c7086a272c24088be94769fd16650"),
            bytes: 66,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::sync::crypto::{sha256, sha384};
        fn hx(s: &str) -> Vec<u8> {
            (0..s.len() / 2)
                .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
                .collect()
        }

        // RFC 6979 A.2.5 — P-256, SHA-256, message "sample".
        #[test]
        fn ecdsa_p256_rfc6979() {
            let c = p256();
            let qx = hx("60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6");
            let qy = hx("7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299");
            let r = hx("EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716");
            let s = hx("F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8");
            let e = sha256(b"sample");
            assert!(c.verify(&qx, &qy, &e, &r, &s), "valid sig must verify");
            // tamper r → reject
            let mut bad = r.clone();
            bad[0] ^= 1;
            assert!(!c.verify(&qx, &qy, &e, &bad, &s), "tampered sig must fail");
            // wrong message → reject
            assert!(
                !c.verify(&qx, &qy, &sha256(b"other"), &r, &s),
                "wrong msg must fail"
            );
        }

        /// No published RFC 6979 vector includes the private key for the "sample"
        /// test above (only qx/qy/r/s), so `sign` is checked the other honest way:
        /// round-trip through `verify`, which the test above already proved correct
        /// against the real published `(r, s)`. Also checks the RFC 6979 determinism
        /// property directly — signing the same `(d, hash)` twice must be bit-identical.
        #[test]
        fn sign_then_verify_round_trips_for_generated_keys() {
            let c = p256();
            for msg in [b"sample".as_slice(), b"test", b"another message entirely"] {
                let (d, qx, qy) = c
                    .generate_keypair()
                    .unwrap_or_else(|e| unreachable!("keygen: {e}"));
                let hash = sha256(msg);
                let (r1, s1) = c.sign(&d, &hash);
                assert!(
                    c.verify(&qx, &qy, &hash, &r1, &s1),
                    "freshly generated key must sign+verify for {msg:?}"
                );
                let (r2, s2) = c.sign(&d, &hash);
                assert_eq!(
                    (r1, s1),
                    (r2, s2),
                    "RFC 6979 nonce must be deterministic — same (d, hash) twice"
                );

                // Signing the wrong hash with a valid (r, s) for another message must not verify.
                let other_hash = sha256(b"a totally different message");
                let (r3, s3) = c.sign(&d, &other_hash);
                assert!(
                    !c.verify(&qx, &qy, &hash, &r3, &s3),
                    "sig over a different message must not verify here"
                );
            }
        }

        #[test]
        fn generate_keypair_is_actually_random() {
            let c = p256();
            let (d1, ..) = c
                .generate_keypair()
                .unwrap_or_else(|e| unreachable!("keygen: {e}"));
            let (d2, ..) = c
                .generate_keypair()
                .unwrap_or_else(|e| unreachable!("keygen: {e}"));
            assert_ne!(d1, d2, "two draws from /dev/urandom must not collide");
        }

        /// P-521/SHA-512 — go-plugin's AutoMTLS combination. Exercises the
        /// hmac_bytes=64 path through `rfc6979_k` (the 32-vs-64 dispatch, and V/K
        /// sized to the hash's 64 bytes while r/s/int2octets stay at the field's 66).
        #[test]
        fn p521_sign_then_verify_round_trips_with_sha512() {
            use crate::sync::crypto::sha512;
            let c = p521();
            for msg in [b"sample".as_slice(), b"go-plugin AutoMTLS handshake cert"] {
                let (d, qx, qy) = c
                    .generate_keypair()
                    .unwrap_or_else(|e| unreachable!("p521 keygen: {e}"));
                assert_eq!(d.len(), 66);
                assert_eq!(qx.len(), 66);
                let hash = sha512(msg);
                let (r1, s1) = c.sign(&d, &hash);
                assert!(
                    c.verify(&qx, &qy, &hash, &r1, &s1),
                    "p521/sha512 sign+verify must round-trip for {msg:?}"
                );
                let (r2, s2) = c.sign(&d, &hash);
                assert_eq!(
                    (r1, s1),
                    (r2, s2),
                    "p521 RFC 6979 nonce must be deterministic too"
                );
            }
        }

        /// The real public key from `tofu`'s own live `PLUGIN_CLIENT_CERT` (captured
        /// 2026-07-06 against a real `hashicorp/null` provider run, decoded via its
        /// DER — curve OID 1.3.132.0.35, uncompressed point, x/y extracted
        /// programmatically, not retyped) must lie on this exact curve. Confirms
        /// these transcribed parameters match the real world go-plugin actually
        /// runs, not just RustCrypto's source.
        #[test]
        fn real_go_plugin_automtls_pubkey_is_on_this_curve() {
            let c = p521();
            let qx = hx("00f60f2e8f52bda7de573c3a17a63f7fc623529089c9db99c866fa55658d7ba03a5ef7f4de0444b73c232e89e2bffba063a88de6af4bf9fc25a51844504120b96214");
            let qy = hx("016495b1d442726145e3b262bcb99a7dc04d13723c107466cdfff70713d57107439c73a89c969c3b15c2be7b3d9200aa7d73c9a98349bd855a570e1e9a633d83d780");
            assert!(
                c.is_on_curve(&qx, &qy),
                "the real captured AutoMTLS cert's own public key must lie on P-521"
            );
        }

        // RFC 6979 A.2.6 — P-384, SHA-384, message "sample".
        #[test]
        fn ecdsa_p384_rfc6979() {
            let c = p384();
            let qx = hx(
                "EC3A4E415B4E19A4568618029F427FA5DA9A8BC4AE92E02E06AAE5286B300C64DEF8F0EA9055866064A254515480BC13",
            );
            let qy = hx(
                "8015D9B72D7D57244EA8EF9AC0C621896708A59367F9DFB9F54CA84B3F1C9DB1288B231C3AE0D4FE7344FD2533264720",
            );
            let r = hx(
                "94EDBB92A5ECB8AAD4736E56C691916B3F88140666CE9FA73D64C4EA95AD133C81A648152E44ACF96E36DD1E80FABE46",
            );
            let s = hx(
                "99EF4AEB15F178CEA1FE40DB2603138F130E740A19624526203B6351D0A3A94FA329C145786E679E7B82C71A38628AC8",
            );
            let e = sha384(b"sample");
            assert!(
                c.verify(&qx, &qy, &e, &r, &s),
                "valid P-384 sig must verify"
            );
            let mut bad = s.clone();
            bad[0] ^= 1;
            assert!(
                !c.verify(&qx, &qy, &e, &r, &bad),
                "tampered P-384 sig must fail"
            );
        }
    }
}
pub mod rsa {
    //! RSA signature *verification* — pure safe Rust. Two schemes:
    //!   * RSASSA-PKCS1-v1.5 — the dominant X.509 *certificate* signature.
    //!   * RSASSA-PSS — the TLS 1.3 *CertificateVerify* scheme for RSA leaf keys.
    //!
    //! A fixed-width bignum (up to RSA-4096) with `s^e mod n` (e is tiny, so modexp
    //! is cheap). KAT'd against openssl-generated vectors in the test module.

    #![allow(dead_code)]

    use crate::sync::crypto::{sha256, sha384, sha512};

    const W: usize = 72; // u64 limbs → 4608-bit capacity (covers RSA-4096 + margin)
    type Big = [u64; W];
    const ZERO: Big = [0; W];

    #[derive(Clone, Copy, PartialEq)]
    pub enum H {
        Sha256,
        Sha384,
        Sha512,
    }

    fn hash(k: H, data: &[u8]) -> Vec<u8> {
        match k {
            H::Sha256 => sha256(data).to_vec(),
            H::Sha384 => sha384(data).to_vec(),
            H::Sha512 => sha512(data).to_vec(),
        }
    }
    fn hlen(k: H) -> usize {
        match k {
            H::Sha256 => 32,
            H::Sha384 => 48,
            H::Sha512 => 64,
        }
    }

    fn from_be(bytes: &[u8]) -> Big {
        let mut b = ZERO;
        for (i, &byte) in bytes.iter().rev().enumerate() {
            if i / 8 < W {
                b[i / 8] |= (byte as u64) << ((i % 8) * 8);
            }
        }
        b
    }
    fn to_be(a: &Big, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        for i in 0..n {
            out[n - 1 - i] = (a[i / 8] >> ((i % 8) * 8)) as u8;
        }
        out
    }
    fn cmp(a: &Big, b: &Big) -> core::cmp::Ordering {
        for i in (0..W).rev() {
            if a[i] != b[i] {
                return a[i].cmp(&b[i]);
            }
        }
        core::cmp::Ordering::Equal
    }
    fn bit(a: &Big, i: usize) -> bool {
        i < W * 64 && (a[i / 64] >> (i % 64)) & 1 == 1
    }
    fn bit_len(a: &Big) -> usize {
        for i in (0..W * 64).rev() {
            if bit(a, i) {
                return i + 1;
            }
        }
        0
    }
    fn add(a: &Big, b: &Big) -> Big {
        let mut r = ZERO;
        let mut c = 0u128;
        for i in 0..W {
            let s = a[i] as u128 + b[i] as u128 + c;
            r[i] = s as u64;
            c = s >> 64;
        }
        r
    }
    fn sub(a: &Big, b: &Big) -> Big {
        let mut r = ZERO;
        let mut br = 0i128;
        for i in 0..W {
            let d = a[i] as i128 - b[i] as i128 - br;
            if d < 0 {
                r[i] = (d + (1i128 << 64)) as u64;
                br = 1;
            } else {
                r[i] = d as u64;
                br = 0;
            }
        }
        r
    }
    fn addmod(a: &Big, b: &Big, m: &Big) -> Big {
        let s = add(a, b);
        if cmp(&s, m) != core::cmp::Ordering::Less {
            sub(&s, m)
        } else {
            s
        }
    }
    fn mulmod(a: &Big, b: &Big, m: &Big) -> Big {
        let mut r = ZERO;
        let mut aa = *a;
        for i in 0..bit_len(b) {
            if bit(b, i) {
                r = addmod(&r, &aa, m);
            }
            aa = addmod(&aa, &aa, m);
        }
        r
    }
    fn modexp(base: &Big, e: &Big, m: &Big) -> Big {
        let mut r = ZERO;
        r[0] = 1;
        let mut b = *base;
        for i in 0..bit_len(e) {
            if bit(e, i) {
                r = mulmod(&r, &b, m);
            }
            b = mulmod(&b, &b, m);
        }
        r
    }

    /// RSA primitive: `sig^e mod n`, returned as `k` big-endian bytes (k = |n|).
    fn rsavp1(n: &[u8], e: &[u8], sig: &[u8]) -> Vec<u8> {
        let m = modexp(&from_be(sig), &from_be(e), &from_be(n));
        to_be(&m, n.len())
    }

    const fn digestinfo_prefix(k: H) -> &'static [u8] {
        match k {
            H::Sha256 => &[
                0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x01, 0x05, 0x00, 0x04, 0x20,
            ],
            H::Sha384 => &[
                0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x02, 0x05, 0x00, 0x04, 0x30,
            ],
            H::Sha512 => &[
                0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x03, 0x05, 0x00, 0x04, 0x40,
            ],
        }
    }

    /// RSASSA-PKCS1-v1.5 verify over `msg` (hashed with `k`).
    pub fn pkcs1_verify(n: &[u8], e: &[u8], k: H, msg: &[u8], sig: &[u8]) -> bool {
        if sig.len() != n.len() {
            return false;
        }
        let em = rsavp1(n, e, sig);
        // EM = 0x00 0x01 (0xFF * ps) 0x00 || DigestInfo
        let t: Vec<u8> = [digestinfo_prefix(k), &hash(k, msg)].concat();
        let klen = n.len();
        if klen < t.len() + 11 {
            return false;
        }
        let ps = klen - t.len() - 3;
        let mut expect = vec![0x00u8, 0x01];
        expect.extend(std::iter::repeat_n(0xffu8, ps));
        expect.push(0x00);
        expect.extend_from_slice(&t);
        em == expect
    }

    fn mgf1(seed: &[u8], len: usize, k: H) -> Vec<u8> {
        let mut out = Vec::new();
        let mut counter: u32 = 0;
        while out.len() < len {
            let mut d = seed.to_vec();
            d.extend_from_slice(&counter.to_be_bytes());
            out.extend_from_slice(&hash(k, &d));
            counter += 1;
        }
        out.truncate(len);
        out
    }

    /// RSASSA-PSS verify (salt length = hash length, MGF1 with the same hash — the
    /// TLS 1.3 `rsa_pss_rsae_*` convention) over `msg`.
    pub fn pss_verify(n: &[u8], e: &[u8], k: H, msg: &[u8], sig: &[u8]) -> bool {
        if sig.len() != n.len() {
            return false;
        }
        let h_len = hlen(k);
        let mod_bits = bit_len(&from_be(n));
        let em_bits = mod_bits - 1;
        let em_len = em_bits.div_ceil(8);
        let raw = rsavp1(n, e, sig);
        // EM is the low em_len bytes (big-endian) of the RSA result
        if raw.len() < em_len {
            return false;
        }
        let em = &raw[raw.len() - em_len..];
        let m_hash = hash(k, msg);
        if em_len < h_len + 2 {
            return false;
        }
        if *em.last().unwrap() != 0xbc {
            return false;
        }
        let db_len = em_len - h_len - 1;
        let masked_db = &em[..db_len];
        let h = &em[db_len..db_len + h_len];
        // leftmost (8*em_len - em_bits) bits of maskedDB must be zero
        let zbits = 8 * em_len - em_bits;
        if zbits > 0 && (masked_db[0] >> (8 - zbits)) != 0 {
            return false;
        }
        let db_mask = mgf1(h, db_len, k);
        let mut db: Vec<u8> = masked_db
            .iter()
            .zip(db_mask.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        if zbits > 0 {
            db[0] &= 0xff >> zbits;
        }
        // DB = PS(0x00..) || 0x01 || salt ; salt length = h_len
        let salt_start = db_len - h_len;
        if salt_start == 0 || db[salt_start - 1] != 0x01 {
            return false;
        }
        if db[..salt_start - 1].iter().any(|&b| b != 0) {
            return false;
        }
        let salt = &db[salt_start..];
        // M' = (0x00 * 8) || mHash || salt ; H' = Hash(M') ; check == H
        let mut mp = vec![0u8; 8];
        mp.extend_from_slice(&m_hash);
        mp.extend_from_slice(salt);
        hash(k, &mp) == h
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn hx(s: &str) -> Vec<u8> {
            let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
            (0..s.len() / 2)
                .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
                .collect()
        }
        // `tls/rsa_kat.rs`, not `rsa_kat.rs`: this file moved from `sync/tls/mod.rs` up to
        // `sync/tls.rs`, and `include!` resolves RELATIVE TO THE INCLUDING FILE — so the bare
        // name silently started pointing at `sync/rsa_kat.rs`.
        include!("tls/rsa_kat.rs");
    }
}
pub mod x25519 {
    //! X25519 ECDHE (RFC 7748) — pure safe Rust. The TLS 1.3 key-exchange.
    //!
    //! GF(2^255-19) in 5×51-bit limbs with u128 products; Montgomery ladder.
    //! Verified against the RFC 7748 §5.2 test vector.
    //!
    //! NOTE: this is correct, not yet constant-time (`cswap` and the ladder branch on
    //! secret bits). A timing-side-channel hardening pass is a labeled follow-on; for
    //! a backup client to S3 it is not the threat that matters first.

    #![allow(dead_code)]

    type Fe = [u64; 5];
    const MASK: u64 = (1 << 51) - 1;

    fn carry_reduce(mut c: [u128; 5]) -> Fe {
        let mut out = [0u64; 5];
        c[1] += c[0] >> 51;
        out[0] = (c[0] as u64) & MASK;
        c[2] += c[1] >> 51;
        out[1] = (c[1] as u64) & MASK;
        c[3] += c[2] >> 51;
        out[2] = (c[2] as u64) & MASK;
        c[4] += c[3] >> 51;
        out[3] = (c[3] as u64) & MASK;
        let carry = (c[4] >> 51) as u64;
        out[4] = (c[4] as u64) & MASK;
        out[0] += carry * 19;
        out[1] += out[0] >> 51;
        out[0] &= MASK;
        out
    }

    fn fe_add(a: &Fe, b: &Fe) -> Fe {
        [
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
            a[4] + b[4],
        ]
    }

    fn fe_sub(a: &Fe, b: &Fe) -> Fe {
        // add 2p so no underflow, then reduce
        const T0: u64 = 0xfffffffffffda; // 2*(2^51-19)
        const TN: u64 = 0xffffffffffffe; // 2*(2^51-1)
        carry_reduce([
            (a[0] + T0 - b[0]) as u128,
            (a[1] + TN - b[1]) as u128,
            (a[2] + TN - b[2]) as u128,
            (a[3] + TN - b[3]) as u128,
            (a[4] + TN - b[4]) as u128,
        ])
    }

    fn fe_mul(a: &Fe, b: &Fe) -> Fe {
        let (a0, a1, a2, a3, a4) = (
            a[0] as u128,
            a[1] as u128,
            a[2] as u128,
            a[3] as u128,
            a[4] as u128,
        );
        let (b0, b1, b2, b3, b4) = (
            b[0] as u128,
            b[1] as u128,
            b[2] as u128,
            b[3] as u128,
            b[4] as u128,
        );
        let (b1e, b2e, b3e, b4e) = (b1 * 19, b2 * 19, b3 * 19, b4 * 19);
        carry_reduce([
            a0 * b0 + a1 * b4e + a2 * b3e + a3 * b2e + a4 * b1e,
            a0 * b1 + a1 * b0 + a2 * b4e + a3 * b3e + a4 * b2e,
            a0 * b2 + a1 * b1 + a2 * b0 + a3 * b4e + a4 * b3e,
            a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + a4 * b4e,
            a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0,
        ])
    }

    fn fe_sq(a: &Fe) -> Fe {
        fe_mul(a, a)
    }

    fn fe_mul121666(a: &Fe) -> Fe {
        carry_reduce([
            a[0] as u128 * 121666,
            a[1] as u128 * 121666,
            a[2] as u128 * 121666,
            a[3] as u128 * 121666,
            a[4] as u128 * 121666,
        ])
    }

    /// a^(p-2) mod p — the modular inverse, via the standard addition chain.
    fn fe_invert(z: &Fe) -> Fe {
        let z2 = fe_sq(z);
        let z9 = fe_mul(&fe_sq(&fe_sq(&z2)), z); // z^9
        let z11 = fe_mul(&z9, &z2);
        let z2_5_0 = fe_mul(&fe_sq(&z11), &z9); // z^31 = 2^5-1
        let mut t = fe_sq(&z2_5_0);
        for _ in 0..4 {
            t = fe_sq(&t);
        }
        let z2_10_0 = fe_mul(&t, &z2_5_0);
        t = fe_sq(&z2_10_0);
        for _ in 0..9 {
            t = fe_sq(&t);
        }
        let z2_20_0 = fe_mul(&t, &z2_10_0);
        t = fe_sq(&z2_20_0);
        for _ in 0..19 {
            t = fe_sq(&t);
        }
        let z2_40_0 = fe_mul(&t, &z2_20_0);
        t = fe_sq(&z2_40_0);
        for _ in 0..9 {
            t = fe_sq(&t);
        }
        let z2_50_0 = fe_mul(&t, &z2_10_0);
        t = fe_sq(&z2_50_0);
        for _ in 0..49 {
            t = fe_sq(&t);
        }
        let z2_100_0 = fe_mul(&t, &z2_50_0);
        t = fe_sq(&z2_100_0);
        for _ in 0..99 {
            t = fe_sq(&t);
        }
        let z2_200_0 = fe_mul(&t, &z2_100_0);
        t = fe_sq(&z2_200_0);
        for _ in 0..49 {
            t = fe_sq(&t);
        }
        let z2_250_0 = fe_mul(&t, &z2_50_0);
        t = fe_sq(&z2_250_0);
        for _ in 0..4 {
            t = fe_sq(&t);
        }
        fe_mul(&t, &z11) // z^(2^255-21)
    }

    fn unpack(b: &[u8; 32]) -> Fe {
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        }
        [
            limbs[0] & MASK,
            ((limbs[0] >> 51) | (limbs[1] << 13)) & MASK,
            ((limbs[1] >> 38) | (limbs[2] << 26)) & MASK,
            ((limbs[2] >> 25) | (limbs[3] << 39)) & MASK,
            (limbs[3] >> 12) & MASK, // top bit (255) is masked off per RFC 7748
        ]
    }

    fn pack(fe: &Fe) -> [u8; 32] {
        // final reduce mod p
        let mut t = *fe;
        for _ in 0..3 {
            t = carry_reduce([
                t[0] as u128,
                t[1] as u128,
                t[2] as u128,
                t[3] as u128,
                t[4] as u128,
            ]);
        }
        // conditional subtract p
        let mut q = (t[0] + 19) >> 51;
        q = (t[1] + q) >> 51;
        q = (t[2] + q) >> 51;
        q = (t[3] + q) >> 51;
        q = (t[4] + q) >> 51;
        t[0] += 19 * q;
        t[1] += t[0] >> 51;
        t[0] &= MASK;
        t[2] += t[1] >> 51;
        t[1] &= MASK;
        t[3] += t[2] >> 51;
        t[2] &= MASK;
        t[4] += t[3] >> 51;
        t[3] &= MASK;
        t[4] &= MASK;
        let words = [
            t[0] | (t[1] << 51),
            (t[1] >> 13) | (t[2] << 38),
            (t[2] >> 26) | (t[3] << 25),
            (t[3] >> 39) | (t[4] << 12),
        ];
        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..i * 8 + 8].copy_from_slice(&words[i].to_le_bytes());
        }
        out
    }

    fn cswap(swap: u64, a: &mut Fe, b: &mut Fe) {
        let mask = 0u64.wrapping_sub(swap);
        for i in 0..5 {
            let t = mask & (a[i] ^ b[i]);
            a[i] ^= t;
            b[i] ^= t;
        }
    }

    /// X25519(scalar, u) — the raw curve function.
    pub fn x25519(scalar: &[u8; 32], point: &[u8; 32]) -> [u8; 32] {
        let mut e = *scalar;
        e[0] &= 248;
        e[31] &= 127;
        e[31] |= 64;

        let x1 = unpack(point);
        let mut x2: Fe = [1, 0, 0, 0, 0];
        let mut z2: Fe = [0, 0, 0, 0, 0];
        let mut x3 = x1;
        let mut z3: Fe = [1, 0, 0, 0, 0];
        let mut swap = 0u64;

        for pos in (0..255).rev() {
            let bit = ((e[pos >> 3] >> (pos & 7)) & 1) as u64;
            swap ^= bit;
            cswap(swap, &mut x2, &mut x3);
            cswap(swap, &mut z2, &mut z3);
            swap = bit;

            let a = fe_add(&x2, &z2);
            let b = fe_sub(&x2, &z2);
            let aa = fe_sq(&a);
            let bb = fe_sq(&b);
            let e_ = fe_sub(&aa, &bb);
            let c = fe_add(&x3, &z3);
            let d = fe_sub(&x3, &z3);
            let da = fe_mul(&d, &a);
            let cb = fe_mul(&c, &b);
            x3 = fe_sq(&fe_add(&da, &cb));
            z3 = fe_mul(&x1, &fe_sq(&fe_sub(&da, &cb)));
            x2 = fe_mul(&aa, &bb);
            z2 = fe_mul(&e_, &fe_add(&bb, &fe_mul121666(&e_)));
        }
        cswap(swap, &mut x2, &mut x3);
        cswap(swap, &mut z2, &mut z3);

        pack(&fe_mul(&x2, &fe_invert(&z2)))
    }

    /// Public key from a secret scalar = X25519(scalar, basepoint 9).
    pub fn base_mul(scalar: &[u8; 32]) -> [u8; 32] {
        let mut base = [0u8; 32];
        base[0] = 9;
        x25519(scalar, &base)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn hx(s: &str) -> [u8; 32] {
            let mut o = [0u8; 32];
            for i in 0..32 {
                o[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
            }
            o
        }

        #[test]
        fn rfc7748_vector() {
            let k = hx("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
            let u = hx("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
            let out = x25519(&k, &u);
            assert_eq!(
                crate::sync::crypto::hex(&out),
                "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552"
            );
        }
    }
}
pub mod x509 {
    //! Minimal X.509 (DER) parsing + chain validation — ECDSA (P-256/P-384) **and**
    //! RSA (PKCS#1 v1.5 chain signatures, RSA-PSS for the TLS CertificateVerify).
    //! Pure safe Rust over `ecdsa` + `rsa` + `crypto`.
    //!
    //! Validates: leaf hostname (SAN, incl wildcard), validity windows, each chain
    //! link's signature, and termination at a system trust anchor (subject + public
    //! key match, so cross-signed roots work).

    #![allow(dead_code)]

    use crate::sync::crypto::{sha256, sha384};
    use crate::sync::tls::ecdsa::{self, Curve};
    use crate::sync::tls::rsa;

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum EcKind {
        P256,
        P384,
        P521, // go-plugin's AutoMTLS curve
    }
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum SigAlg {
        EcdsaSha256,
        EcdsaSha384,
        EcdsaSha512, // go-plugin's AutoMTLS signature algorithm (paired with P521 above)
        RsaSha256,
        RsaSha384,
        RsaSha512,
    }
    #[derive(Clone, PartialEq)]
    enum Pub {
        Ec(EcKind, Vec<u8>, Vec<u8>), // (curve, x, y)
        Rsa(Vec<u8>, Vec<u8>),        // (n, e)
    }

    pub struct Cert {
        tbs: Vec<u8>,
        sig_alg: Option<SigAlg>,
        sig_raw: Vec<u8>, // BIT STRING content minus the unused-bits byte
        subject: Vec<u8>,
        issuer: Vec<u8>,
        not_before: i64,
        not_after: i64,
        key: Pub,
        sans: Vec<String>,
    }

    // ── DER reader ─────────────────────────────────────────────────────────────

    struct Der<'a> {
        b: &'a [u8],
        p: usize,
    }
    impl<'a> Der<'a> {
        fn new(b: &'a [u8]) -> Self {
            Der { b, p: 0 }
        }
        fn tlv(&mut self) -> Option<(u8, &'a [u8], &'a [u8])> {
            let start = self.p;
            let tag = *self.b.get(self.p)?;
            self.p += 1;
            let l0 = *self.b.get(self.p)?;
            self.p += 1;
            let len = if l0 < 0x80 {
                l0 as usize
            } else {
                let k = (l0 & 0x7f) as usize;
                let mut v = 0usize;
                for _ in 0..k {
                    v = (v << 8) | *self.b.get(self.p)? as usize;
                    self.p += 1;
                }
                v
            };
            let content = self.b.get(self.p..self.p + len)?;
            self.p += len;
            Some((tag, content, &self.b[start..self.p]))
        }
    }

    const OID_EC_PUBKEY: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
    const OID_P256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
    const OID_P384: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x22];
    const OID_RSA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
    const OID_ECDSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
    const OID_ECDSA_SHA384: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x03];
    const OID_RSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
    const OID_RSA_SHA384: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c];
    const OID_RSA_SHA512: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0d];
    const OID_SAN: &[u8] = &[0x55, 0x1d, 0x11];
    const OID_P521: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x23]; // 1.3.132.0.35 — go-plugin's AutoMTLS curve
    const OID_ECDSA_SHA512: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x04];
    const OID_BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1d, 0x13];
    const OID_COMMON_NAME: &[u8] = &[0x55, 0x04, 0x03];
    const OID_ORG_NAME: &[u8] = &[0x55, 0x04, 0x0a];

    fn parse_sig_alg(content: &[u8]) -> Option<SigAlg> {
        let mut d = Der::new(content);
        let (_, o, _) = d.tlv()?;
        Some(match o {
            x if x == OID_ECDSA_SHA256 => SigAlg::EcdsaSha256,
            x if x == OID_ECDSA_SHA384 => SigAlg::EcdsaSha384,
            x if x == OID_ECDSA_SHA512 => SigAlg::EcdsaSha512,
            x if x == OID_RSA_SHA256 => SigAlg::RsaSha256,
            x if x == OID_RSA_SHA384 => SigAlg::RsaSha384,
            x if x == OID_RSA_SHA512 => SigAlg::RsaSha512,
            _ => return None,
        })
    }

    fn strip_int(b: &[u8]) -> Vec<u8> {
        let mut i = 0;
        while i + 1 < b.len() && b[i] == 0 {
            i += 1;
        }
        b[i..].to_vec()
    }

    fn parse_ecdsa_sig(sig_raw: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        let mut d = Der::new(sig_raw); // DER SEQUENCE { r, s }
        let (_, seq, _) = d.tlv()?;
        let mut s = Der::new(seq);
        let (_, r, _) = s.tlv()?;
        let (_, sv, _) = s.tlv()?;
        Some((strip_int(r), strip_int(sv)))
    }

    fn parse_pubkey(spki: &[u8]) -> Option<Pub> {
        let mut d = Der::new(spki);
        let (_, algid, _) = d.tlv()?;
        let (_, bitstr, _) = d.tlv()?;
        let mut a = Der::new(algid);
        let (_, oid, _) = a.tlv()?;
        if oid == OID_EC_PUBKEY {
            let (_, curve_oid, _) = a.tlv()?;
            let k = if curve_oid == OID_P256 {
                EcKind::P256
            } else if curve_oid == OID_P384 {
                EcKind::P384
            } else if curve_oid == OID_P521 {
                EcKind::P521
            } else {
                return None;
            };
            let point = bitstr.get(1..)?;
            if point.first() != Some(&0x04) {
                return None;
            }
            let body = &point[1..];
            let n = body.len() / 2;
            Some(Pub::Ec(k, body[..n].to_vec(), body[n..].to_vec()))
        } else if oid == OID_RSA {
            // BIT STRING content: 0x00 || RSAPublicKey SEQUENCE { modulus INTEGER, exponent INTEGER }
            let inner = bitstr.get(1..)?;
            let mut r = Der::new(inner);
            let (_, seq, _) = r.tlv()?;
            let mut s = Der::new(seq);
            let (_, n, _) = s.tlv()?;
            let (_, e, _) = s.tlv()?;
            Some(Pub::Rsa(strip_int(n), strip_int(e)))
        } else {
            None
        }
    }

    fn parse_time(tag: u8, content: &[u8]) -> Option<i64> {
        let s = std::str::from_utf8(content).ok()?;
        let (y, rest) = if tag == 0x17 {
            let yy: i64 = s.get(0..2)?.parse().ok()?;
            (if yy >= 50 { 1900 + yy } else { 2000 + yy }, &s[2..])
        } else {
            (s.get(0..4)?.parse().ok()?, &s[4..])
        };
        let mo: i64 = rest.get(0..2)?.parse().ok()?;
        let da: i64 = rest.get(2..4)?.parse().ok()?;
        let h: i64 = rest.get(4..6)?.parse().ok()?;
        let mi: i64 = rest.get(6..8)?.parse().ok()?;
        let se: i64 = rest.get(8..10).and_then(|x| x.parse().ok()).unwrap_or(0);
        Some(days_from_civil(y, mo, da) * 86400 + h * 3600 + mi * 60 + se)
    }

    fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
        let y = if m <= 2 { y - 1 } else { y };
        let era = (if y >= 0 { y } else { y - 399 }) / 400;
        let yoe = y - era * 400;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146097 + doe - 719468
    }

    /// Inverse of `days_from_civil` (Howard Hinnant's `civil_from_days`) — needed to
    /// render a self-signed cert's own `notBefore`/`notAfter` back from Unix time,
    /// the direction the parser above never needed.
    fn civil_from_days(z: i64) -> (i64, u32, u32) {
        let z = z + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = (z - era * 146097) as u64;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        (if m <= 2 { y + 1 } else { y }, m, d)
    }

    fn generalized_time_string(unix_secs: i64) -> String {
        let days = unix_secs.div_euclid(86400);
        let secs_of_day = unix_secs.rem_euclid(86400);
        let (y, mo, d) = civil_from_days(days);
        let h = secs_of_day / 3600;
        let mi = (secs_of_day % 3600) / 60;
        let s = secs_of_day % 60;
        format!("{y:04}{mo:02}{d:02}{h:02}{mi:02}{s:02}Z")
    }

    // ── DER writer ─────────────────────────────────────────────────────────────
    // The inverse of the `Der` reader above: minimal, purpose-built TLV encoding —
    // not a general ASN.1 library, just what a self-signed go-plugin AutoMTLS
    // handshake cert needs (SEQUENCE/INTEGER/OID/BIT STRING/strings/time/context
    // tags), mirroring the exact shapes `parse_cert` already reads.

    fn w_len(n: usize, out: &mut Vec<u8>) {
        if n < 0x80 {
            out.push(n as u8);
            return;
        }
        let bytes = (n as u64).to_be_bytes();
        let sig: Vec<u8> = bytes.into_iter().skip_while(|&b| b == 0).collect();
        out.push(0x80 | sig.len() as u8);
        out.extend_from_slice(&sig);
    }

    fn w_tlv(tag: u8, content: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        w_len(content.len(), &mut out);
        out.extend_from_slice(content);
        out
    }

    fn w_seq(children: &[u8]) -> Vec<u8> {
        w_tlv(0x30, children)
    }
    fn w_set(children: &[u8]) -> Vec<u8> {
        w_tlv(0x31, children)
    }
    fn w_oid(oid: &[u8]) -> Vec<u8> {
        w_tlv(0x06, oid)
    }
    fn w_null() -> Vec<u8> {
        vec![0x05, 0x00]
    }
    fn w_bool(b: bool) -> Vec<u8> {
        w_tlv(0x01, &[if b { 0xff } else { 0x00 }])
    }
    fn w_utf8(s: &str) -> Vec<u8> {
        w_tlv(0x0c, s.as_bytes())
    }
    fn w_octet_string(bytes: &[u8]) -> Vec<u8> {
        w_tlv(0x04, bytes)
    }
    fn w_generalized_time(unix_secs: i64) -> Vec<u8> {
        w_tlv(0x18, generalized_time_string(unix_secs).as_bytes())
    }
    fn w_context(tag_num: u8, content: &[u8]) -> Vec<u8> {
        w_tlv(0xa0 | tag_num, content)
    }

    /// Unsigned big-endian `bytes` as a DER INTEGER: strip leading zero bytes, then
    /// re-prepend exactly one if the high bit would otherwise read as a negative
    /// two's-complement value (DER integers are always signed).
    fn w_uint(bytes: &[u8]) -> Vec<u8> {
        let mut b: Vec<u8> = bytes.iter().copied().skip_while(|&x| x == 0).collect();
        if b.is_empty() {
            b.push(0);
        }
        if b[0] & 0x80 != 0 {
            b.insert(0, 0);
        }
        w_tlv(0x02, &b)
    }

    /// `BIT STRING` wrapping an already byte-aligned value (0 unused bits) — every
    /// use in a cert (SPKI point, signature) is byte-aligned.
    fn w_bitstring(bytes: &[u8]) -> Vec<u8> {
        let mut content = vec![0x00];
        content.extend_from_slice(bytes);
        w_tlv(0x03, &content)
    }

    /// `Name ::= RDNSequence` with exactly the two RDNs go-plugin's own AutoMTLS
    /// cert uses (`O=HashiCorp, CN=<common_name>`) — SEQUENCE of SET of
    /// AttributeTypeAndValue, matching what `parse_cert`/`parse_sans` above expect
    /// to read back.
    fn w_name(common_name: &str) -> Vec<u8> {
        let org_rdn = w_set(&w_seq(&[w_oid(OID_ORG_NAME), w_utf8("HashiCorp")].concat()));
        let cn_rdn = w_set(&w_seq(
            &[w_oid(OID_COMMON_NAME), w_utf8(common_name)].concat(),
        ));
        w_seq(&[org_rdn, cn_rdn].concat())
    }

    /// A minimal, self-signed X.509v3 certificate DER, signed with its own P-521
    /// key — go-plugin's AutoMTLS handshake shape (subject == issuer, SAN =
    /// `common_name`, `ecdsa-with-SHA512`), built from the real captured `tofu`
    /// cert's own field shapes (subject/issuer `O=HashiCorp, CN=<name>`, a
    /// `basicConstraints` CA:false extension, SAN) rather than guessed. `d` is the
    /// P-521 private key (66-byte big-endian) that signs its own TBS; `qx`/`qy` are
    /// the matching public point.
    pub fn build_self_signed_p521(
        d: &[u8],
        qx: &[u8],
        qy: &[u8],
        common_name: &str,
        serial: &[u8],
        not_before: i64,
        not_after: i64,
    ) -> Vec<u8> {
        let sig_alg_id = w_seq(&w_oid(OID_ECDSA_SHA512));
        let name = w_name(common_name);
        let validity = w_seq(
            &[
                w_generalized_time(not_before),
                w_generalized_time(not_after),
            ]
            .concat(),
        );

        let ec_params = w_seq(&[w_oid(OID_EC_PUBKEY), w_oid(OID_P521)].concat());
        let mut point = vec![0x04u8];
        point.extend_from_slice(qx);
        point.extend_from_slice(qy);
        let spki = w_seq(&[ec_params, w_bitstring(&point)].concat());

        let basic_constraints_ext = w_seq(
            &[
                w_oid(OID_BASIC_CONSTRAINTS),
                w_bool(true),                           // critical
                w_octet_string(&w_seq(&w_bool(false))), // CA:false
            ]
            .concat(),
        );
        let san_ext = w_seq(
            &[
                w_oid(OID_SAN),
                w_octet_string(&w_seq(&w_tlv(0x82, common_name.as_bytes()))),
            ]
            .concat(),
        );
        let extensions = w_context(3, &w_seq(&[basic_constraints_ext, san_ext].concat()));

        let tbs = w_seq(
            &[
                w_context(0, &w_uint(&[2])), // version v3
                w_uint(serial),
                sig_alg_id.clone(),
                name.clone(), // issuer
                validity,
                name, // subject (self-signed: issuer == subject)
                spki,
                extensions,
            ]
            .concat(),
        );

        let hash = crate::sync::crypto::sha512(&tbs);
        let (r, s) = ecdsa::p521().sign(d, &hash);
        let ec_sig = w_seq(&[w_uint(&r), w_uint(&s)].concat());

        w_seq(&[tbs, sig_alg_id, w_bitstring(&ec_sig)].concat())
    }

    /// DER → standard PEM (`-----BEGIN CERTIFICATE-----`, 64-column base64,
    /// `-----END CERTIFICATE-----`) — what `PLUGIN_CLIENT_CERT` and any other
    /// PEM-consuming boundary expects; the inverse of this module's own `b64`
    /// decoder.
    pub fn to_pem(der: &[u8]) -> String {
        const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut b64 = String::new();
        for chunk in der.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            b64.push(TBL[(b[0] >> 2) as usize] as char);
            b64.push(TBL[(((b[0] & 0x03) << 4) | (b[1] >> 4)) as usize] as char);
            b64.push(if chunk.len() > 1 {
                TBL[(((b[1] & 0x0f) << 2) | (b[2] >> 6)) as usize] as char
            } else {
                '='
            });
            b64.push(if chunk.len() > 2 {
                TBL[(b[2] & 0x3f) as usize] as char
            } else {
                '='
            });
        }
        let mut out = String::from("-----BEGIN CERTIFICATE-----\n");
        for line in b64.as_bytes().chunks(64) {
            out.push_str(std::str::from_utf8(line).unwrap_or(""));
            out.push('\n');
        }
        out.push_str("-----END CERTIFICATE-----\n");
        out
    }

    fn parse_sans(ext_content: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut d = Der::new(ext_content);
        if let Some((_, seq, _)) = d.tlv() {
            let mut s = Der::new(seq);
            while let Some((_, ext, _)) = s.tlv() {
                let mut e = Der::new(ext);
                if let Some((_, o, _)) = e.tlv() {
                    if o == OID_SAN {
                        let mut octet = None;
                        while let Some((t, c, _)) = e.tlv() {
                            if t == 0x04 {
                                octet = Some(c);
                            }
                        }
                        if let Some(os) = octet {
                            let mut g = Der::new(os);
                            if let Some((_, names, _)) = g.tlv() {
                                let mut gn = Der::new(names);
                                while let Some((t, c, _)) = gn.tlv() {
                                    if t == 0x82 {
                                        if let Ok(s) = std::str::from_utf8(c) {
                                            out.push(s.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    pub fn parse_cert(der: &[u8]) -> Option<Cert> {
        let mut top = Der::new(der);
        let (_, cert_body, _) = top.tlv()?;
        let mut c = Der::new(cert_body);
        let (_, tbs_content, tbs_full) = c.tlv()?;
        let (_, sigalg, _) = c.tlv()?;
        let (_, sigval, _) = c.tlv()?;
        let sig_alg = parse_sig_alg(sigalg);
        let sig_raw = sigval.get(1..).unwrap_or(&[]).to_vec(); // skip BIT STRING unused-bits byte

        let mut t = Der::new(tbs_content);
        if let Some((tag, _, _)) = peek(&t) {
            if tag == 0xa0 {
                t.tlv()?; // version [0]
            }
        }
        t.tlv()?; // serialNumber
        t.tlv()?; // signature algid
        let (_, _issuer, issuer_full) = t.tlv()?;
        let (_, validity, _) = t.tlv()?;
        let (_, _subject, subject_full) = t.tlv()?;
        let (_, spki, _) = t.tlv()?;
        let mut sans = Vec::new();
        while let Some((tag, content, _)) = t.tlv() {
            if tag == 0xa3 {
                sans = parse_sans(content);
            }
        }

        let mut v = Der::new(validity);
        let (tb, nb, _) = v.tlv()?;
        let (ta, na, _) = v.tlv()?;
        let not_before = parse_time(tb, nb)?;
        let not_after = parse_time(ta, na)?;
        let key = parse_pubkey(spki)?;

        Some(Cert {
            tbs: tbs_full.to_vec(),
            sig_alg,
            sig_raw,
            subject: subject_full.to_vec(),
            issuer: issuer_full.to_vec(),
            not_before,
            not_after,
            key,
            sans,
        })
    }

    fn peek(d: &Der) -> Option<(u8, usize, usize)> {
        let mut c = Der { b: d.b, p: d.p };
        let (t, content, _) = c.tlv()?;
        Some((t, content.len(), 0))
    }

    fn curve(k: EcKind) -> Curve {
        match k {
            EcKind::P256 => ecdsa::p256(),
            EcKind::P384 => ecdsa::p384(),
            EcKind::P521 => ecdsa::p521(),
        }
    }

    /// Verify `child`'s signature with `issuer`'s public key, dispatched by alg.
    fn verify_link(child: &Cert, issuer: &Cert) -> bool {
        match (child.sig_alg, &issuer.key) {
            (Some(SigAlg::EcdsaSha256), Pub::Ec(k, x, y)) => {
                let Some((r, s)) = parse_ecdsa_sig(&child.sig_raw) else {
                    return false;
                };
                curve(*k).verify(x, y, &sha256(&child.tbs), &r, &s)
            }
            (Some(SigAlg::EcdsaSha384), Pub::Ec(k, x, y)) => {
                let Some((r, s)) = parse_ecdsa_sig(&child.sig_raw) else {
                    return false;
                };
                curve(*k).verify(x, y, &sha384(&child.tbs), &r, &s)
            }
            (Some(SigAlg::EcdsaSha512), Pub::Ec(k, x, y)) => {
                let Some((r, s)) = parse_ecdsa_sig(&child.sig_raw) else {
                    return false;
                };
                curve(*k).verify(x, y, &crate::sync::crypto::sha512(&child.tbs), &r, &s)
            }
            (Some(SigAlg::RsaSha256), Pub::Rsa(n, e)) => {
                rsa::pkcs1_verify(n, e, rsa::H::Sha256, &child.tbs, &child.sig_raw)
            }
            (Some(SigAlg::RsaSha384), Pub::Rsa(n, e)) => {
                rsa::pkcs1_verify(n, e, rsa::H::Sha384, &child.tbs, &child.sig_raw)
            }
            (Some(SigAlg::RsaSha512), Pub::Rsa(n, e)) => {
                rsa::pkcs1_verify(n, e, rsa::H::Sha512, &child.tbs, &child.sig_raw)
            }
            _ => false,
        }
    }

    fn host_matches(sans: &[String], host: &str) -> bool {
        let host = host.to_ascii_lowercase();
        sans.iter().any(|s| {
            let s = s.to_ascii_lowercase();
            if let Some(suffix) = s.strip_prefix("*.") {
                host.strip_suffix(suffix)
                    .map(|p| p.ends_with('.') && !p[..p.len() - 1].contains('.'))
                    .unwrap_or(false)
                    || host == suffix
            } else {
                host == s
            }
        })
    }

    pub struct TrustStore {
        anchors: Vec<Cert>,
    }
    impl TrustStore {
        pub fn system() -> TrustStore {
            let mut anchors = Vec::new();
            for path in [
                "/etc/ssl/certs/ca-certificates.crt",
                "/etc/pki/tls/certs/ca-bundle.crt",
                "/etc/ssl/cert.pem",
            ] {
                if let Ok(pem) = std::fs::read_to_string(path) {
                    for der in pem_certs(&pem) {
                        if let Some(c) = parse_cert(&der) {
                            anchors.push(c);
                        }
                    }
                    break;
                }
            }
            TrustStore { anchors }
        }
        fn anchor_for_subject(&self, subject: &[u8]) -> Option<&Cert> {
            self.anchors.iter().find(|a| a.subject == subject)
        }
        fn is_anchor(&self, c: &Cert) -> bool {
            self.anchors
                .iter()
                .any(|a| a.subject == c.subject && a.key == c.key)
        }
    }

    fn pem_certs(pem: &str) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut inside = false;
        for line in pem.lines() {
            if line.contains("BEGIN CERTIFICATE") {
                inside = true;
                cur.clear();
            } else if line.contains("END CERTIFICATE") {
                inside = false;
                if let Some(der) = b64(&cur) {
                    out.push(der);
                }
            } else if inside {
                cur.push_str(line.trim());
            }
        }
        out
    }

    fn b64(s: &str) -> Option<Vec<u8>> {
        fn val(c: u8) -> Option<u8> {
            match c {
                b'A'..=b'Z' => Some(c - b'A'),
                b'a'..=b'z' => Some(c - b'a' + 26),
                b'0'..=b'9' => Some(c - b'0' + 52),
                b'+' => Some(62),
                b'/' => Some(63),
                _ => None,
            }
        }
        let mut out = Vec::new();
        let mut acc = 0u32;
        let mut bits = 0;
        for &c in s.as_bytes() {
            if c == b'=' {
                break;
            }
            let Some(v) = val(c) else { continue };
            acc = (acc << 6) | v as u32;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
            }
        }
        Some(out)
    }

    /// Validate a server certificate chain (leaf first). Returns the leaf on success.
    pub fn validate(
        chain_der: &[Vec<u8>],
        host: &str,
        now: i64,
        trust: &TrustStore,
    ) -> Result<Cert, String> {
        if chain_der.is_empty() {
            return Err("empty certificate chain".into());
        }
        let mut chain: Vec<Cert> = Vec::new();
        for (i, d) in chain_der.iter().enumerate() {
            match parse_cert(d) {
                Some(c) => chain.push(c),
                None => return Err(format!("cert {i}: unsupported key/sig or bad DER")),
            }
        }
        let leaf = &chain[0];
        if !host_matches(&leaf.sans, host) {
            return Err(format!("hostname {host} not in leaf SAN {:?}", leaf.sans));
        }
        for (i, c) in chain.iter().enumerate() {
            if now < c.not_before || now > c.not_after {
                return Err(format!("cert {i} outside validity window"));
            }
        }
        for i in 0..chain.len() - 1 {
            if !verify_link(&chain[i], &chain[i + 1]) {
                return Err(format!("chain link {i}→{} signature invalid", i + 1));
            }
        }
        let top = chain.last().unwrap();
        if trust.is_anchor(top) {
            return Ok(chain.remove(0));
        }
        if let Some(anchor) = trust.anchor_for_subject(&top.issuer) {
            if verify_link(top, anchor) {
                return Ok(chain.remove(0));
            }
            return Err("top cert not signed by its trust-store issuer".into());
        }
        Err("chain does not terminate at a system trust anchor".into())
    }

    /// Verify the TLS 1.3 CertificateVerify (leaf signs the transcript) — ECDSA
    /// (P-256/P-384) or RSA-PSS, per the SignatureScheme.
    pub fn verify_cert_verify(leaf: &Cert, scheme: u16, sig: &[u8], transcript_hash: &[u8]) -> bool {
        let mut signed = vec![0x20u8; 64];
        signed.extend_from_slice(b"TLS 1.3, server CertificateVerify");
        signed.push(0x00);
        signed.extend_from_slice(transcript_hash);
        match (scheme, &leaf.key) {
            (0x0403, Pub::Ec(EcKind::P256, x, y)) => {
                let Some((r, s)) = parse_ecdsa_sig(sig) else {
                    return false;
                };
                ecdsa::p256().verify(x, y, &sha256(&signed), &r, &s)
            }
            (0x0503, Pub::Ec(EcKind::P384, x, y)) => {
                let Some((r, s)) = parse_ecdsa_sig(sig) else {
                    return false;
                };
                ecdsa::p384().verify(x, y, &sha384(&signed), &r, &s)
            }
            (0x0804, Pub::Rsa(n, e)) => rsa::pss_verify(n, e, rsa::H::Sha256, &signed, sig),
            (0x0805, Pub::Rsa(n, e)) => rsa::pss_verify(n, e, rsa::H::Sha384, &signed, sig),
            (0x0806, Pub::Rsa(n, e)) => rsa::pss_verify(n, e, rsa::H::Sha512, &signed, sig),
            _ => false,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn to_pem_round_trips_through_the_existing_pem_decoder() {
            let curve = ecdsa::p521();
            let (d, qx, qy) = curve
                .generate_keypair()
                .unwrap_or_else(|e| unreachable!("keygen: {e}"));
            let der = build_self_signed_p521(
                &d,
                &qx,
                &qy,
                "localhost",
                &[0x07],
                1_800_000_000,
                1_900_000_000,
            );
            let pem = to_pem(&der);
            assert!(pem.starts_with("-----BEGIN CERTIFICATE-----\n"));
            let decoded = pem_certs(&pem);
            assert_eq!(decoded.len(), 1);
            assert_eq!(
                decoded[0], der,
                "to_pem -> pem_certs must reconstruct the identical DER"
            );
        }

        /// Build a self-signed AutoMTLS-shaped cert, parse it back through the
        /// SAME `parse_cert` a real peer's cert goes through, and check every
        /// field the handshake actually needs survives the round trip — subject/
        /// SAN/validity/public key — AND that the self-signature verifies (the
        /// cert signs its own TBS with its own key, so `verify_link(&c, &c)` must
        /// succeed exactly like a real go-plugin AutoMTLS cert self-verifying).
        #[test]
        fn self_signed_p521_cert_round_trips_and_self_verifies() {
            let curve = ecdsa::p521();
            let (d, qx, qy) = curve
                .generate_keypair()
                .unwrap_or_else(|e| unreachable!("keygen: {e}"));
            let not_before = 1_800_000_000i64;
            let not_after = not_before + 30 * 365 * 86400;
            let der = build_self_signed_p521(
                &d,
                &qx,
                &qy,
                "localhost",
                &[0x01, 0x02, 0x03],
                not_before,
                not_after,
            );

            let cert = parse_cert(&der).unwrap_or_else(|| unreachable!("must parse back its own DER"));
            assert_eq!(cert.sans, vec!["localhost".to_string()]);
            assert_eq!(cert.not_before, not_before);
            assert_eq!(cert.not_after, not_after);
            assert_eq!(cert.sig_alg, Some(SigAlg::EcdsaSha512));
            let Pub::Ec(EcKind::P521, parsed_x, parsed_y) = &cert.key else {
                unreachable!("expected an EC P521 key, got a different key shape");
            };
            assert_eq!(
                *parsed_x, qx,
                "SPKI point bytes are raw, not ASN.1 INTEGER-stripped"
            );
            assert_eq!(*parsed_y, qy);
            assert_eq!(
                cert.subject, cert.issuer,
                "self-signed: subject and issuer must be identical"
            );

            assert!(
                verify_link(&cert, &cert),
                "a self-signed cert must verify against its own public key"
            );

            // Tamper one byte of the DER and the signature must stop verifying —
            // otherwise this would be verifying nothing.
            let mut tampered = der.clone();
            let last = tampered.len() - 1;
            tampered[last] ^= 0x01;
            if let Some(bad_cert) = parse_cert(&tampered) {
                assert!(
                    !verify_link(&bad_cert, &bad_cert),
                    "tampering the DER must break self-verification"
                );
            }
        }
    }
}
