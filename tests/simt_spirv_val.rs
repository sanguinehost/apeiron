//! Gate 1 for the ported SIMT substrate: every emitted module is accepted by the Khronos validator
//! (`spirv-val --target-env vulkan1.1`), and a corrupted module is rejected — the validator actually
//! binds. Honest skip when `spirv-val` is not installed (the `execVia` pattern: a missing external
//! runner is a skip, never a silent pass). Gate 2 — running on the metal — is `examples/simt_check.rs`
//! and `examples/simt_gemv.rs`.
//!
//! Faithful port of fold `modelfold/tests/simt.rs`, adapted to `apeiron::simt`.

use apeiron::simt::{gemv_bytes, gemv_dp4a_bytes, rmsnorm_bytes, spv_bytes, Carrier, Op};
use std::io::Write;
use std::process::Command;

/// `spirv-val` on raw bytes via a temp file; `None` = validator not installed (skip).
fn spirv_val(bytes: &[u8], label: &str) -> Option<bool> {
    if Command::new("spirv-val").arg("--version").output().is_err() {
        return None;
    }
    let path = std::env::temp_dir().join(format!("apeiron_simt_{label}.spv"));
    let mut f = std::fs::File::create(&path).expect("temp .spv");
    f.write_all(bytes).expect("write .spv");
    drop(f);
    let out = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.1"])
        .arg(&path)
        .output()
        .expect("run spirv-val");
    if !out.status.success() {
        eprintln!(
            "spirv-val [{label}]: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Some(out.status.success())
}

#[test]
fn every_op_module_validates() {
    let mut checked = 0;
    for op in Op::ALL {
        match spirv_val(&spv_bytes(op), op.name()) {
            None => {
                eprintln!("SKIP: spirv-val not installed");
                return;
            }
            Some(ok) => assert!(ok, "{op:?}: module rejected by spirv-val"),
        }
        checked += 1;
    }
    assert_eq!(checked, 12);
}

/// Every carrier's GEMV module validates, plain and split-K.
#[test]
fn gemv_module_validates() {
    for carrier in [Carrier::F32, Carrier::F16, Carrier::Q8, Carrier::Q4] {
        for (m, k, l, sg) in [
            (64u32, 64u32, 64u32, 1u32),
            (8192, 8192, 256, 1),
            (4096, 14336, 256, 8),
        ] {
            if carrier.is_chart() && k % 32 != 0 {
                continue;
            }
            let label = format!("gemv_{}_{m}x{k}_s{sg}", carrier.name());
            match spirv_val(&gemv_bytes(m, k, l, sg, carrier), &label) {
                None => {
                    eprintln!("SKIP: spirv-val not installed");
                    return;
                }
                Some(ok) => assert!(ok, "{label}: rejected by spirv-val"),
            }
        }
    }
}

/// The integer-dot (DP4A) GEMV module validates for every chart carrier, plain and split-K.
#[test]
fn dp4a_module_validates() {
    for carrier in [Carrier::Q8, Carrier::Q4, Carrier::Q2] {
        for (m, k, l, sg) in [
            (64u32, 64u32, 64u32, 1u32),
            (8192, 8192, 256, 1),
            (4096, 14336, 256, 8),
        ] {
            let label = format!("gemv_dp4a_{}_{m}x{k}_s{sg}", carrier.name());
            match spirv_val(&gemv_dp4a_bytes(m, k, l, sg, carrier), &label) {
                None => {
                    eprintln!("SKIP: spirv-val not installed");
                    return;
                }
                Some(ok) => assert!(ok, "{label}: rejected by spirv-val"),
            }
        }
    }
}

/// The `RMSNorm` glue kernel validates (hand-emitted SPIR-V, the forward's per-token norm).
#[test]
fn rmsnorm_module_validates() {
    for (n, eps) in [(576u32, 1e-5f32), (896, 1e-6), (4096, 1e-5)] {
        match spirv_val(&rmsnorm_bytes(n, eps), &format!("rmsnorm_{n}")) {
            None => {
                eprintln!("SKIP: spirv-val not installed");
                return;
            }
            Some(ok) => assert!(ok, "rmsnorm({n}): rejected by spirv-val"),
        }
    }
}

/// Negative control: flip one opcode word — the validator must reject, or it validates nothing.
#[test]
fn corrupted_module_is_rejected() {
    let mut bytes = spv_bytes(Op::AAdd);
    // Word 5 is the first instruction (OpCapability); zeroing it breaks the stream.
    bytes[20] = 0xFF;
    bytes[21] = 0xFF;
    match spirv_val(&bytes, "corrupt") {
        None => eprintln!("SKIP: spirv-val not installed"),
        Some(ok) => assert!(!ok, "validator accepted a corrupted module"),
    }
}
