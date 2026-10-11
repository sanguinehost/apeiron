//! `IQ1_S` (1.5625 b/w): the port of llama.cpp's weighted codebook quantizer and decoder.
//!
//! Source: `ggml/src/ggml-quants.c` at llama.cpp `ece963f41`: `quantize_row_iq1_s_impl` (4508),
//! `iq1_find_best_neighbour2` (4443), `iq2xs_init_impl` (grid, map, neighbours; 2849-3255),
//! `dequantize_row_iq1_s` (2653), `IQ1S_DELTA = 0.125` (`ggml-common.h:1132`). The 2048-entry grid is
//! [`crate::learn::iq1s_grid::KGRID_1BIT_2048`].
//!
//! **Format.** A 256-weight block is `d: f16`, `qs: [u8; 32]`, `qh: [u16; 8]` (50 bytes). Weights split
//! into eight sub-blocks of 32; a sub-block is four 8-vectors, each a codeword of the grid
//! (index = `qs | 3 bits of qh << 8`, 11 bits). The sub-block carries a 3-bit scale `s` (`qh >> 12`) and
//! a sign bit for the shift `±1/8` (`qh >> 15`). The decoded weight is
//! `d (2s+1) (g_j + δ)` with `g_j ∈ {-1, 0, +1}` and `δ = ±1/8`.
//!
//! **Quantizer.** Per sub-block, with importance `w_i = qw_i sqrt(σ² + x_i²)` (σ² = 2 mean x² over the
//! 256-block): sort `x`, enumerate every split `i1 ≤ i2` of the sorted order into three groups
//! labelled `-1, 0, +1`, score each by `(Σ w q x)² / Σ w q²` for both shifts, keep the best; snap each
//! 8-vector to the grid (exact hit through `kmap`, else the weighted-nearest of the precomputed
//! neighbours); refit the scale when any vector moved; then quantize the sub-block scales to 3 bits
//! against `d`.
//!
//! **Evidence.** `tested`: byte-identity with `llama-quantize` on the tensors of
//! `/home/socol/oc-tmp/onebit-w9/qwen05b-IQ1_S-imat.gguf` (`examples/iq1s_check.rs`). Not `proved`.
//!
//! **Complexity** (AGENTS §8). [`Grid::new`]: time `O(M · G)` distance evaluations for `M = 43692`
//! packed 8-vectors and `G = 2048` codewords, plus `M` sorts of `G` pairs (`O(M G log G)` comparisons);
//! space `O(M + neighbours)`. [`quantize_row`]: per sub-block `O(32²)` split evaluations plus `4 · |N|`
//! neighbour scores; `O(n)` for a row of `n` weights.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::explicit_iter_loop,
    clippy::must_use_candidate,
    clippy::missing_panics_doc,
    reason = "byte-exact port of `quantize_row_iq1_s_impl` and `iq2xs_init_impl`: the C index, cast and \
              loop shapes are kept so a line-by-line diff against ggml-quants.c stays reviewable, and \
              every cast is bounded by the 2048-entry grid and the 43692-entry map; the pedantic lints \
              stay on for the rest of the crate."
)]

use super::iq1s_grid::KGRID_1BIT_2048;
use crate::simt::f16_from_f32;

/// Weights per super-block.
pub const QK_K: usize = 256;
/// Weights per sub-block.
pub const SUB: usize = 32;
/// Bytes per super-block: `d` (2), `qs` (32), `qh` (16).
pub const BLOCK_BYTES: usize = 50;
/// `IQ1S_DELTA`.
pub const DELTA: f32 = 0.125;
/// Number of codewords.
pub const NGRID: usize = 2048;
/// Packed 8-vectors over `{0,1,2,3}^8` (`3` unused): `0b1010101010101010 + 2`.
const KMAP_SIZE: usize = 43692;
/// `GROUP_MAX_EPS_IQ1_S`.
const GROUP_MAX_EPS: f32 = 1e-12;
/// Distinct distance classes kept per off-grid vector (`nwant` for `IQ1_S`).
const NWANT: usize = 3;

/// Codeword table, packed-vector map and neighbour lists (`iq2xs_init_impl` for `IQ1_S`).
pub struct Grid {
    /// `grid[k][i] = 2 l + 1` for code `l` of codeword `k`, entry `i`.
    pub grid: Vec<[i8; 8]>,
    /// Packed vector (2 bits per entry) to codeword index, or `-(offset+1)` into `neighbours`.
    pub map: Vec<i32>,
    /// `[n, c_1 .. c_n]` runs, one per off-grid packed vector.
    pub neighbours: Vec<u16>,
}

impl Grid {
    /// Build the tables from [`KGRID_1BIT_2048`].
    pub fn new() -> Grid {
        let mut grid = vec![[0i8; 8]; NGRID];
        for (k, g) in grid.iter_mut().enumerate() {
            for (i, v) in g.iter_mut().enumerate() {
                *v = 2 * ((KGRID_1BIT_2048[k] >> (2 * i)) & 3) as i8 + 1;
            }
        }
        let mut map = vec![-1i32; KMAP_SIZE];
        for (i, g) in grid.iter().enumerate() {
            let mut index = 0usize;
            for (k, &v) in g.iter().enumerate() {
                index |= (((v - 1) / 2) as usize) << (2 * k);
            }
            map[index] = i as i32;
        }
        let mut neighbours: Vec<u16> = Vec::new();
        let mut dist2 = vec![(0i32, 0i32); NGRID];
        for i in 0..KMAP_SIZE {
            if map[i] >= 0 {
                continue;
            }
            let mut pos = [0i32; 8];
            for (k, p) in pos.iter_mut().enumerate() {
                *p = 2 * ((i >> (2 * k)) & 3) as i32 + 1;
            }
            for (j, d) in dist2.iter_mut().enumerate() {
                let mut d2 = 0;
                for k in 0..8 {
                    let t = i32::from(grid[j][k]) - pos[k];
                    d2 += t * t;
                }
                *d = (d2, j as i32);
            }
            dist2.sort_unstable();
            let offset = neighbours.len();
            map[i] = -((offset as i32) + 1);
            neighbours.push(0);
            let mut d2 = dist2[0].0;
            let mut nhave = 1;
            let mut n = 0u16;
            for &(dj, j) in dist2.iter() {
                if dj > d2 {
                    if nhave == NWANT {
                        break;
                    }
                    d2 = dj;
                    nhave += 1;
                }
                neighbours.push(j as u16);
                n += 1;
            }
            neighbours[offset] = n;
        }
        Grid {
            grid,
            map,
            neighbours,
        }
    }
}

impl Default for Grid {
    fn default() -> Self {
        Grid::new()
    }
}

/// `nearest_int` of `ggml-quants.c:621`: round to nearest even through the `1.5·2²³` offset.
fn nearest_int(x: f32) -> i32 {
    let v = x + 12_582_912.0;
    (v.to_bits() as i32 & 0x007f_ffff) - 0x0040_0000
}

/// `iq1_find_best_neighbour2`: the weighted-nearest neighbour of an off-grid 8-vector; writes the
/// chosen labels into `l`.
fn best_neighbour(
    g: &Grid,
    run: usize,
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    xg: &[f32; 3],
    l: &mut [i8],
) -> usize {
    let n = g.neighbours[run] as usize;
    assert!(n > 0, "off-grid vector with no neighbours");
    let mut best = f32::MAX;
    let mut idx = usize::MAX;
    for j in 1..=n {
        let c = g.neighbours[run + j] as usize;
        let pg = &g.grid[c];
        let mut d2 = 0f32;
        for i in 0..8 {
            let q = xg[((pg[i] - 1) / 2) as usize];
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best {
            best = d2;
            idx = c;
        }
    }
    assert!(idx != usize::MAX, "no finite neighbour score");
    for i in 0..8 {
        l[i] = (g.grid[idx][i] - 1) / 2;
    }
    idx
}

/// Quantize one row (`x.len() % 256 == 0`) with importance `qw` (one per column) into `out`
/// (`50` bytes per super-block).
pub fn quantize_row(g: &Grid, x: &[f32], qw: &[f32], out: &mut [u8]) {
    assert!(x.len() % QK_K == 0 && qw.len() == x.len());
    assert_eq!(out.len(), x.len() / QK_K * BLOCK_BYTES);
    let x_p = [-1.0 + DELTA, DELTA, 1.0 + DELTA];
    let x_m = [-1.0 - DELTA, -DELTA, 1.0 - DELTA];
    for (ibl, ob) in out.chunks_exact_mut(BLOCK_BYTES).enumerate() {
        ob.fill(0);
        let xbl = &x[QK_K * ibl..QK_K * (ibl + 1)];
        let mut sumx2 = 0f32;
        for &v in xbl {
            sumx2 += v * v;
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;
        let mut qs = [0u8; 32];
        let mut qh = [0u16; 8];
        let mut scales = [0f32; 8];
        let mut shifts = [0i8; 8];
        let mut max_scale = 0f32;
        for ib in 0..QK_K / SUB {
            let xb = &xbl[SUB * ib..SUB * (ib + 1)];
            let qwb = &qw[QK_K * ibl + SUB * ib..QK_K * ibl + SUB * (ib + 1)];
            let mut weight = [0f32; SUB];
            for i in 0..SUB {
                weight[i] = qwb[i] * (sigma2 + xb[i] * xb[i]).sqrt();
            }
            let mut max = xb[0].abs();
            for &v in &xb[1..] {
                max = max.max(v.abs());
            }
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                shifts[ib] = 1;
                continue;
            }
            let mut order: Vec<usize> = (0..SUB).collect();
            order.sort_by(|&a, &b| xb[a].partial_cmp(&xb[b]).expect("finite weights"));
            let mut sumx = [0f32; SUB + 1];
            let mut sumw = [0f32; SUB + 1];
            for j in 0..SUB {
                let i = order[j];
                sumx[j + 1] = sumx[j] + weight[i] * xb[i];
                sumw[j + 1] = sumw[j] + weight[i];
            }
            let mut best_score = -f32::MAX;
            let mut scale = max;
            let (mut besti1, mut besti2, mut best_shift) = (-1i32, -1i32, 0i32);
            for i1 in 0..=SUB {
                for i2 in i1..=SUB {
                    let (a, b, c) = (
                        sumx[i1] - sumx[0],
                        sumx[i2] - sumx[i1],
                        sumx[SUB] - sumx[i2],
                    );
                    let (wa, wb, wc) = (
                        sumw[i1] - sumw[0],
                        sumw[i2] - sumw[i1],
                        sumw[SUB] - sumw[i2],
                    );
                    for (xs, sh) in [(&x_p, 1), (&x_m, -1)] {
                        let sumqx = a * xs[0] + b * xs[1] + c * xs[2];
                        let sumq2 = wa * xs[0] * xs[0] + wb * xs[1] * xs[1] + wc * xs[2] * xs[2];
                        if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                            scale = sumqx / sumq2;
                            best_score = scale * sumqx;
                            besti1 = i1 as i32;
                            besti2 = i2 as i32;
                            best_shift = sh;
                        }
                    }
                }
            }
            if besti1 < 0 || besti2 < 0 || best_shift == 0 {
                scales[ib] = 0.0;
                shifts[ib] = 1;
                continue;
            }
            let mut l = [0i8; SUB];
            for j in 0..SUB {
                l[order[j]] = if (j as i32) < besti1 {
                    0
                } else if (j as i32) < besti2 {
                    1
                } else {
                    2
                };
            }
            if scale < 0.0 {
                for v in &mut l {
                    *v = 2 - *v;
                }
                scale = -scale;
                best_shift = -best_shift;
            }
            let xx = if best_shift == 1 { &x_p } else { &x_m };
            let mut all_on_grid = true;
            let mut index = [0usize; 4];
            for k in 0..4 {
                let mut u = 0usize;
                for j in 0..8 {
                    u |= (l[8 * k + j] as usize) << (2 * j);
                }
                let m = g.map[u];
                if m >= 0 {
                    index[k] = m as usize;
                } else {
                    all_on_grid = false;
                    let run = (-m - 1) as usize;
                    index[k] = best_neighbour(
                        g,
                        run,
                        &xb[8 * k..8 * k + 8],
                        &weight[8 * k..8 * k + 8],
                        scale,
                        xx,
                        &mut l[8 * k..8 * k + 8],
                    );
                }
            }
            if !all_on_grid {
                let (mut sumqx, mut sumq2) = (0f32, 0f32);
                for k in 0..4 {
                    let pg = &g.grid[index[k]];
                    for j in 0..8 {
                        let w = weight[8 * k + j];
                        let q = xx[((pg[j] - 1) / 2) as usize];
                        sumqx += w * q * xb[8 * k + j];
                        sumq2 += w * q * q;
                    }
                }
                if sumqx > 0.0 && sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            let mut h = 0u16;
            for k in 0..4 {
                qs[4 * ib + k] = (index[k] & 255) as u8;
                h |= ((index[k] >> 8) as u16) << (3 * k);
            }
            qh[ib] = h;
            assert!(scale >= 0.0);
            scales[ib] = scale;
            shifts[ib] = best_shift as i8;
            max_scale = max_scale.max(scale);
        }
        let mut d16 = 0u16;
        if max_scale != 0.0 {
            let d = max_scale / 15.0;
            d16 = f16_from_f32(d * 1.125);
            let id = 1.0 / d;
            for ib in 0..8 {
                let mut l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 7) as u16;
                if shifts[ib] == -1 {
                    l |= 8;
                }
                qh[ib] |= l << 12;
            }
        }
        ob[0..2].copy_from_slice(&d16.to_le_bytes());
        ob[2..34].copy_from_slice(&qs);
        for (i, h) in qh.iter().enumerate() {
            ob[34 + 2 * i..36 + 2 * i].copy_from_slice(&h.to_le_bytes());
        }
    }
}

/// `dequantize_row_iq1_s`: decode `blocks` (50 bytes each) into `y`.
pub fn dequantize_row(blocks: &[u8], y: &mut [f32]) {
    assert_eq!(blocks.len() / BLOCK_BYTES * QK_K, y.len());
    let mut at = 0;
    for b in blocks.chunks_exact(BLOCK_BYTES) {
        let d = crate::format::quant::f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for ib in 0..8 {
            let h = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]);
            let dl = d * (2 * ((h >> 12) & 7) + 1) as f32;
            let delta = if h & 0x8000 != 0 { -DELTA } else { DELTA };
            for l in 0..4 {
                let idx = usize::from(b[2 + 4 * ib + l]) | ((usize::from(h) >> (3 * l)) & 7) << 8;
                let code = KGRID_1BIT_2048[idx];
                for j in 0..8 {
                    let gj = ((code >> (2 * j)) & 3) as i32 - 1;
                    y[at] = dl * (gj as f32 + delta);
                    at += 1;
                }
            }
        }
    }
}
