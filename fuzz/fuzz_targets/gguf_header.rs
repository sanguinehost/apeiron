//! GGUF header, tensor-directory arithmetic and bounded dequantization over attacker bytes.
//!
//! The input is a whole GGUF file. Every refusal is a pass; a panic, an out-of-bounds index, an
//! allocation past `-rss_limit_mb` or a sanitizer report is a failure. Seeds in `fuzz/seeds/` cover
//! the llama.cpp header CVE classes: CVE-2024-21825 (tensor count drives an allocation),
//! CVE-2024-23605 (array element count drives an allocation), CVE-2024-21802 (`n_dims` overflow).
#![no_main]

use apeiron::format::{gguf, quant};
use libfuzzer_sys::fuzz_target;

/// Per-tensor dequantization cap in weights, so one input stays under the RSS limit.
const MAX_WEIGHTS: usize = 1 << 16;

fuzz_target!(|data: &[u8]| {
    let file_len = data.len() as u64;
    let Ok(h) = gguf::parse_header(data, file_len) else {
        return;
    };
    let _ = h.meta_i64("general.alignment");
    let _ = h.arch();
    for t in &h.tensors {
        let _ = t.n_elements();
        let Ok(Some(bytes)) = t.n_bytes() else {
            continue;
        };
        // The extent check `Engine::open` applies (`build_table`): a payload that leaves the file
        // is refused before any kernel reads it.
        let Some(at) = h.data0.checked_add(t.offset) else {
            continue;
        };
        let Some(end) = at.checked_add(bytes) else {
            continue;
        };
        if end > file_len {
            continue;
        }
        let raw = &data[at as usize..end as usize];
        let Some((qk, _)) = quant::block_geom(t.ggml_type) else {
            continue;
        };
        let n = t
            .n_elements()
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
            .min(MAX_WEIGHTS);
        if qk == quant::QK_K {
            let mut out = vec![0f32; n];
            let _ = quant::dequant_super_block(t.ggml_type, raw, n, &mut out);
        } else if t.ggml_type == quant::GGML_TYPE_Q8_0 {
            let mut b = 0usize;
            while b * qk < n {
                if quant::q8_0_scale(raw, b).is_err() || quant::q8_0_codes(raw, b).is_err() {
                    break;
                }
                b += 1;
            }
        }
    }
});
