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
