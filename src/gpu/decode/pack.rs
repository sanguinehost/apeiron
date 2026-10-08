//! Weight packing for DP4A GPU decode serving ported from fold `bedrock/src/serve/model/pack.rs`.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's bedrock/src/serve/model/pack.rs; lint profile is fold's, not apeiron's"
)]

use crate::format::gguf::{Gguf, GgufError, TensorDesc};
use crate::format::quant::f16_to_f32;
use crate::simt::Carrier;

/// Convert `f32` to IEEE 754 half-precision `f16` bit pattern with round-to-nearest-even.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | u16::from(m != 0);
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00;
    }
    let (mant, shift) = if e16 <= 0 {
        if e16 < -10 {
            return s;
        }
        (m | 0x80_0000, (13 + 1 - e16) as u32)
    } else {
        (m, 13u32)
    };
    let base = if e16 <= 0 { 0u32 } else { (e16 as u32) << 10 };
    let kept = mant >> shift;
    let rem = mant & ((1 << shift) - 1);
    let half = 1u32 << (shift - 1);
    let round_up = rem > half || (rem == half && kept & 1 == 1);
    s | ((base | (kept & 0x3ff)) + u32::from(round_up)) as u16
}

/// Decode f16-packed (d, lo) chart words to interleaved f32 pairs (`[d0, lo0, d1, lo1, …]`).
pub fn dmf_of(dm: &[u32]) -> Vec<f32> {
    let mut out = vec![0f32; dm.len() * 2];
    for (i, &w) in dm.iter().enumerate() {
        out[2 * i] = f16_to_f32((w & 0xffff) as u16);
        out[2 * i + 1] = f16_to_f32((w >> 16) as u16);
    }
    out
}

/// One projection prepared for GPU DP4A chart GEMV.
#[derive(Clone, Debug)]
pub struct Proj {
    pub words: Vec<u32>,
    pub dm: Vec<u32>,
    pub dmf: Vec<f32>,
    pub cpu: Vec<f32>,
    pub qsub: Vec<u8>,
    pub qsup: Vec<u16>,
    pub m: usize,
    pub k: usize,
}

impl Proj {
    pub fn empty() -> Self {
        Self {
            words: Vec::new(),
            dm: Vec::new(),
            dmf: Vec::new(),
            cpu: Vec::new(),
            qsub: Vec::new(),
            qsup: Vec::new(),
            m: 0,
            k: 0,
        }
    }

    pub fn served_bytes(&self) -> usize {
        self.words.len() * 4 + self.dm.len() * 4 + self.qsub.len() + self.qsup.len() * 2
    }
}

/// Whether this projection carries the two-level scale face.
pub fn tl_face(p: &Proj) -> bool {
    !p.qsub.is_empty()
        && !p.qsup.is_empty()
        && !p.words.is_empty()
        && p.k % 32 == 0
        && p.qsub.len() == p.m * (p.k / 32) * 2
        && p.qsup.len() == p.m * 2
}

/// Pack an f32 row-major [m, k] tensor into the DP4A k-packed chart (Carrier::Q4).
pub fn pack_proj_f32(carrier: Carrier, rows: &[&[f32]], k: usize, mirror: bool) -> Proj {
    assert!(k % 32 == 0, "chart carriers need k % 32 == 0");
    let m: usize = rows.iter().map(|r| r.len() / k).sum();
    let pack = carrier.pack() as usize;
    let width = (32 / pack) as u32;
    let levels = carrier.qmask() as f32;
    let blocks = k / 32;

    let mut words = vec![0u32; (k / pack) * m];
    let mut dm = vec![0u32; m * blocks];
    let mut cpu = vec![0f32; if mirror { m * k } else { 0 }];

    let mut rr = 0usize;
    for part in rows {
        for r in 0..part.len() / k {
            let row = &part[r * k..(r + 1) * k];
            for b in 0..blocks {
                let vals = &row[b * 32..b * 32 + 32];
                let lo = vals.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let d = if hi > lo { (hi - lo) / levels } else { 1.0 };
                let d16 = f16_to_f32(f32_to_f16(d));
                let d16 = if d16 == 0.0 { 1.0 } else { d16 };
                let lo16 = f16_to_f32(f32_to_f16(lo));
                let dm_idx = b * m + rr;
                dm[dm_idx] = (f32_to_f16(d16) as u32) | ((f32_to_f16(lo16) as u32) << 16);
                for (j, &v) in vals.iter().enumerate() {
                    let q = (((v - lo16) / d16).round() as i64).clamp(0, levels as i64) as u32;
                    let kk = b * 32 + j;
                    words[(kk / pack) * m + rr] |= q << ((kk % pack) as u32 * width);
                    if mirror {
                        cpu[rr * k + kk] = d16 * (q as f32) + lo16;
                    }
                }
            }
            rr += 1;
        }
    }

    Proj {
        dmf: dmf_of(&dm),
        qsub: Vec::new(),
        qsup: Vec::new(),
        words,
        dm,
        cpu,
        m,
        k,
    }
}

/// Pack a row-concatenation of ALL-Q5_0 tensors from raw GGUF bytes into the [`Carrier::Q5`]
/// two-plane chart, 5.0 bits/weight — the Q4 nibble plane (`q & 0xF`, 8 codes/word, the identical
/// k-packed layout [`pack_proj_f32`] emits for `Carrier::Q4`) followed by one permuted 1-bit
/// high-plane word per (32-block, row), and `dm = f16(d) ‖ f16(−16·d)` so the chart reconstructs
/// `w = d·q + lo = d·(q−16)` exactly (the Q4/Q8 epilogue already reads `d·Σ(q·x) + lo·Σx`; the
/// reference [`crate::format::quant::q5_0_codes`] is that same code offset by −16).
///
/// The high plane is permuted so a DP4A phase's four codes are adjacent in the word: code `j` of a
/// block gets its bit at `pos(j) = (j>>3)·8 + (j&1)·4 + ((j>>1)&3)`. The kernel expands the 4-bit
/// field with `nib · 0x0204_0810 & 0x1010_1010` (llama.cpp CUDA `vecdotq.cuh:182-196`), putting
/// each code's fifth bit into bit 4 of its byte lane.
///
/// Temporal cost: O(m·k) — one plane-A nibble write and one plane-B bit set per element.
/// Spatial footprint: `Carrier::Q5.code_words_per_row(k)·m` u32 words (plane A plus the plane-B
/// plane, no constant restated here) plus `(k/32)·m` `dm` words.
pub fn pack_q5_0_raw(
    raw: &[u8],
    m: usize,
    k: usize,
) -> Result<Proj, crate::format::quant::QuantError> {
    use crate::format::quant::{q5_0_codes, q5_0_d, Q5_0_BLOCK_BYTES, Q5_0_QK};
    assert!(k % 32 == 0, "chart carriers need k % 32 == 0");
    let pack = Carrier::Q5.pack() as usize;
    let width = 4u32;
    let blocks = k / 32;
    let blocks_per_row = k.div_ceil(Q5_0_QK);
    let row_bytes = blocks_per_row * Q5_0_BLOCK_BYTES;
    let hi_off = Carrier::Q5.plane_b_off(m as u32, k as u32) as usize; // the high plane follows plane A
    let mut words = vec![0u32; Carrier::Q5.code_words_per_row(k as u32) as usize * m];
    let mut dm = vec![0u32; m * blocks];

    for r in 0..m {
        let row = &raw[r * row_bytes..(r + 1) * row_bytes];
        for b in 0..blocks {
            let blk = &row[b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES];
            let d = f16_to_f32(q5_0_d(blk)?);
            let lo = -16.0f32 * d;
            // d is already f16; -16·d is exact in f16 for every finite block scale here.
            let dm_idx = b * m + r;
            dm[dm_idx] = (f32_to_f16(d) as u32) | ((f32_to_f16(lo) as u32) << 16);
            let codes = q5_0_codes(blk)?;
            for j in 0..32 {
                let q = (i32::from(codes[j]) + 16) as u32; // 0..31
                let kk = b * 32 + j;
                words[(kk / pack) * m + r] |= (q & 0xF) << ((kk % pack) as u32 * width);
                let pos = ((j >> 3) * 8 + (j & 1) * 4 + ((j >> 1) & 3)) as u32;
                words[hi_off + b * m + r] |= (q >> 4) << pos;
            }
        }
    }

    Ok(Proj {
        dmf: dmf_of(&dm),
        qsub: Vec::new(),
        qsup: Vec::new(),
        words,
        dm,
        cpu: Vec::new(),
        m,
        k,
    })
}

/// Pack a Q4_K tensor directly from raw GGUF bytes into the DP4A k-packed chart (Carrier::Q4).
pub fn pack_q4_k_raw(
    raw: &[u8],
    m: usize,
    k: usize,
    mirror: bool,
) -> Result<Proj, crate::format::quant::QuantError> {
    assert!(k % 32 == 0, "chart carriers need k % 32 == 0");
    let pack = 8usize; // Carrier::Q4.pack()
    let width = 4u32;
    let blocks = k / 32;
    let (qk, bb) = crate::format::quant::block_geom(crate::format::quant::GGML_TYPE_Q4_K)
        .expect("Q4_K block geom");
    let blocks_per_row = k.div_ceil(qk);
    let row_bytes = blocks_per_row * bb;

    let mut words = vec![0u32; (k / pack) * m];
    let mut dm = vec![0u32; m * blocks];
    let mut cpu = vec![0f32; if mirror { m * k } else { 0 }];

    for r in 0..m {
        let row_slice = &raw[r * row_bytes..(r + 1) * row_bytes];
        for super_b in 0..blocks_per_row {
            let blk = &row_slice[super_b * bb..(super_b + 1) * bb];
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let pairs = crate::format::quant::q4_k_scale_mins(blk)?;
            let codes = crate::format::quant::q4_k_codes(blk)?;

            for sub_b in 0..8 {
                let b_32 = super_b * 8 + sub_b;
                if b_32 >= blocks {
                    break;
                }
                let (sl, ml) = pairs[sub_b];
                let is_high = (sub_b % 2) == 1;
                let pair_b = sub_b / 2;
                let d_sub = d * f32::from(sl);
                let m_sub = dmin * f32::from(ml);
                let lo_sub = -m_sub;

                let d16 = f16_to_f32(f32_to_f16(d_sub));
                let lo16 = f16_to_f32(f32_to_f16(lo_sub));
                let dm_idx = b_32 * m + r;
                dm[dm_idx] = (f32_to_f16(d16) as u32) | ((f32_to_f16(lo16) as u32) << 16);

                let code_offset = 64 * pair_b + if is_high { 32 } else { 0 };
                for j in 0..32 {
                    let q = codes[code_offset + j] as u32;
                    let kk = b_32 * 32 + j;
                    words[(kk / pack) * m + r] |= q << ((kk % pack) as u32 * width);
                    if mirror {
                        cpu[r * k + kk] = d16 * (q as f32) + lo16;
                    }
                }
            }
        }
    }

    Ok(Proj {
        dmf: dmf_of(&dm),
        qsub: Vec::new(),
        qsup: Vec::new(),
        words,
        dm,
        cpu,
        m,
        k,
    })
}

/// Pack ONE projection from a GGUF file via apeiron's own loader (`src/format/gguf.rs`).
pub fn pack_gguf_tensor(gguf: &Gguf, tensor_name: &str) -> Result<Proj, GgufError> {
    let t = gguf
        .header
        .tensor(tensor_name)
        .ok_or_else(|| GgufError::MissingTensor(tensor_name.to_string()))?;
    pack_tensor_desc(gguf, t)
}

/// Pack a projection described by [`TensorDesc`] via apeiron's own loader.
pub fn pack_tensor_desc(gguf: &Gguf, desc: &TensorDesc) -> Result<Proj, GgufError> {
    let k = desc.dims.first().copied().unwrap_or(0) as usize;
    let m = desc.dims.get(1).copied().unwrap_or(1) as usize;
    let raw = gguf.tensor_bytes(desc)?;
    if desc.ggml_type == crate::format::quant::GGML_TYPE_Q4_K {
        pack_q4_k_raw(&raw, m, k, false).map_err(GgufError::Quant)
    } else {
        let name_str = String::from_utf8_lossy(&desc.name);
        let weights = gguf.dequant(&name_str)?;
        Ok(pack_proj_f32(Carrier::Q4, &[&weights], k, false))
    }
}

#[cfg(test)]
mod tests {
    use crate::format::gguf::Gguf;
    use crate::format::quant::{
        f16_to_f32, q5_0_codes, q5_0_d, GGML_TYPE_Q5_0, Q5_0_BLOCK_BYTES, Q5_0_QK,
    };
    use crate::simt::Carrier;

    const QWEN05B_PATH: &str = "models/gguf/qwen05b-Q4_K_M.gguf";

    /// [`Carrier::Q5`]'s two-plane chart must reproduce the crate's own Q5_0 dequantizer
    /// bit-for-bit, over every element of a real Q5_0 tensor.
    #[test]
    fn q5_0_chart_is_exact() {
        let Some(path) = crate::engine::testdata::model_path(QWEN05B_PATH) else {
            eprintln!("skip q5_0_chart_is_exact: {QWEN05B_PATH} absent");
            return;
        };
        let gguf = Gguf::open(&path).expect("open qwen05b");
        let t = gguf
            .header
            .tensor("blk.0.ffn_gate.weight")
            .expect("blk.0.ffn_gate.weight")
            .clone();
        assert_eq!(t.ggml_type, GGML_TYPE_Q5_0, "gate tensor is Q5_0");
        let k = usize::try_from(t.dims[0]).expect("k fits usize");
        let m = usize::try_from(t.dims[1]).expect("m fits usize");
        let raw = gguf.tensor_bytes(&t).expect("gate tensor bytes");
        let p = super::pack_q5_0_raw(&raw, m, k).expect("pack_q5_0_raw");

        assert_eq!(
            p.words.len(),
            Carrier::Q5.code_words_per_row(k as u32) as usize * m,
            "word count is code_words_per_row(k)·m"
        );
        let pack = Carrier::Q5.pack() as usize;
        let hi_off = Carrier::Q5.plane_b_off(m as u32, k as u32) as usize;
        let row_bytes = k.div_ceil(Q5_0_QK) * Q5_0_BLOCK_BYTES;

        let mut checked = 0usize;
        for r in 0..m {
            let row = &raw[r * row_bytes..(r + 1) * row_bytes];
            for b in 0..k / 32 {
                let blk = &row[b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES];
                let want_d = f16_to_f32(q5_0_d(blk).expect("q5_0_d"));
                let codes = q5_0_codes(blk).expect("q5_0_codes");
                let dmw = p.dm[b * m + r];
                let d = f16_to_f32((dmw & 0xffff) as u16);
                let lo = f16_to_f32((dmw >> 16) as u16);
                for j in 0..32 {
                    let kk = b * 32 + j;
                    let nib = (p.words[(kk / pack) * m + r] >> ((kk % pack) as u32 * 4)) & 0xF;
                    // The permuted plane-B bit for code j, stated independently of the packer.
                    let bit = (j >> 3) * 8 + (j & 1) * 4 + ((j >> 1) & 3);
                    let hi = (p.words[hi_off + b * m + r] >> bit) & 1;
                    let q = nib | (hi << 4);
                    let got = d * (q as f32) + lo;
                    let want = want_d * f32::from(codes[j]);
                    // The fold is exact: `d·q + lo` and `d·(q−16)` differ only in the SIGN OF ZERO
                    // (for `d < 0`, `q = 16` the fold gives `+0.0` and the product `−0.0`; IEEE `==`
                    // equates them and no other value differs). Compare bits, canonicalizing that
                    // one non-value difference.
                    assert!(
                        got.to_bits() == want.to_bits() || (got == 0.0 && want == 0.0),
                        "chart weight mismatch r={r} b={b} j={j}: got {got} ({:#010x}) want \
{want} ({:#010x})",
                        got.to_bits(),
                        want.to_bits()
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, m * k, "every element decoded");
    }
}
