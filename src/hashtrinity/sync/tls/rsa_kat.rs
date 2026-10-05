// openssl-generated RSA-2048 KAT vectors (message = "syncfold rsa kat").
const N: &str = "e451a3209eaaff8667852cb6b0c69ec512906d5df33fc017bc6e9846f6f09e7dde988bc4cfded09d844d67034ef01f0716d2befb8bf33bc3a5dd0039cfd822b54dffc3e28105404f830b4bf2618964b057bb413c6433a9498293b8dcdf6e6452e69776e44c5ceb6237cf610ae924fcc7c6aa912e05a624cb648cdfa8b6a1eb49eef32d601afabc016d7a0bc2f36965b9e2518e3daa95fdddb1d68f834b71cd3607486cc297dba02070c2616507b3712e4085d91355d8300b48a45860c651c13a094f1177bfd84dbfcb1c74bc7e823dacb9b27e22e1c18b0a123f201fb8a07debac89a83a3f5da6c6f32ed1369c3f0a89bfb6eb34de6aae721235cd1a0bf14913";
const E: &str = "010001";
const SIG_PKCS1: &str = "d752902ed5b8d26727c9c09660075a18813084d1abb386ebe2891c1aa70204d6d5074395591477186e8b6bce5241c5d3c83f7b1ed4018817cb43e3c164992a251e191d7c247847195d6db7db4e91d322dcca2ca772c346f699e175751a7653274ed6b8b371a3c98ec3f8a02d649f9b3e5fe2635524751f7edb79db6182e7a82192629f856f074827f1b46d16d79fe6427367f16cd962d81e7574b5ef1757e63df33a9618667c5d6dce0f25d3694ca71641438f13a94b1f25cfa742f1ca8dfa751ea7aaaca2524a236241917c56774ac33f83a16ad6925e318006fe6ad2c499e2aeb5919c718b2643102e041035ae19f093d1ad3ae1adfb8127bc7cb9ea3ecef8";
const SIG_PSS: &str = "5b109572e14e5fb2df6c59325b818535d0e65da02367e1c25ef76c6688425f6ce508a335a8d95d4ed5a29206182e96ab737c4abf05fe11b751d6c232354b67c2787d680004f01d9a087d03d0781b28e8fd1406481d01b78176363c8883372b3a35feec820f1a765e6e59c7b7a3d60c3b0d893eb8f67161813f43547f7fb26f362305dbe8ac93095cd8664f39400207c613f66fcc8e4320c16d1c45c8713be58160439b0a5262433b0fc940b2ee83f9c701e01c6bf33a9a3d12047f84836d9279ccfb951344798b65ea4d63b3df2f1f6680c0701b03f6373e131082b5abc3e7289a131a435f82c9c023450304e7fa7a48821129e6779b9c2beacc5ed4bef9633d";

#[test]
fn rsa_pkcs1_v15_sha256() {
    let (n, e, sig) = (hx(N), hx(E), hx(SIG_PKCS1));
    assert!(pkcs1_verify(&n, &e, H::Sha256, b"syncfold rsa kat", &sig), "valid PKCS1 sig must verify");
    assert!(!pkcs1_verify(&n, &e, H::Sha256, b"tampered", &sig), "wrong msg must fail");
    let mut bad = sig.clone();
    bad[0] ^= 1;
    assert!(!pkcs1_verify(&n, &e, H::Sha256, b"syncfold rsa kat", &bad), "tampered sig must fail");
}

#[test]
fn rsa_pss_sha256() {
    let (n, e, sig) = (hx(N), hx(E), hx(SIG_PSS));
    assert!(pss_verify(&n, &e, H::Sha256, b"syncfold rsa kat", &sig), "valid PSS sig must verify");
    assert!(!pss_verify(&n, &e, H::Sha256, b"tampered", &sig), "wrong msg must fail");
}
