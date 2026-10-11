//! `IQ1_S` port: grid identity with the llama.cpp source, grid-table well-formedness, and the decoder's
//! agreement with the quantizer on a deterministic row.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "test harness: numeric casts"
)]

use apeiron::learn::iq1s::{dequantize_row, quantize_row, Grid, BLOCK_BYTES, NGRID, QK_K};
use apeiron::learn::iq1s_grid::KGRID_1BIT_2048;

const C_SRC: &str = "~/llama.cpp/ggml/src/ggml-quants.c";

#[test]
fn grid_matches_llama_cpp_source() {
    let Ok(src) = std::fs::read_to_string(C_SRC) else {
        return;
    };
    let at = src.find("kgrid_1bit_2048[NGRID_IQ1S] = {").expect("table");
    let body = &src[at..];
    let body = &body[body.find('{').unwrap() + 1..body.find("};").unwrap()];
    let v: Vec<u16> = body
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(v.as_slice(), KGRID_1BIT_2048.as_slice());
}

#[test]
fn grid_tables_are_well_formed() {
    let g = Grid::new();
    let mut seen = std::collections::HashSet::new();
    for (i, c) in KGRID_1BIT_2048.iter().enumerate() {
        assert!(seen.insert(*c), "duplicate codeword {i}");
        for j in 0..8 {
            assert!((c >> (2 * j)) & 3 <= 2, "code 3 at {i}");
        }
        assert_eq!(
            g.map[usize::from(*c)],
            i as i32,
            "codeword {i} not at its own packed index"
        );
    }
    assert_eq!(seen.len(), NGRID);
    for (u, &m) in g.map.iter().enumerate() {
        if m < 0 {
            let run = (-m - 1) as usize;
            assert!(g.neighbours[run] > 0, "empty neighbour list at {u}");
        }
    }
}

#[test]
fn decode_is_scaled_grid_plus_shift() {
    let g = Grid::new();
    let n = 2 * QK_K;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.003)
        .collect();
    let qw: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();
    let mut q = vec![0u8; n / QK_K * BLOCK_BYTES];
    quantize_row(&g, &x, &qw, &mut q);
    let mut y = vec![0f32; n];
    dequantize_row(&q, &mut y);
    // Each decoded sub-block of 8 is dl * (g + delta): value set per sub-block has at most 3 levels.
    for sb in y.chunks(32) {
        let mut lv: Vec<u32> = sb.iter().map(|v| v.to_bits()).collect();
        lv.sort_unstable();
        lv.dedup();
        assert!(lv.len() <= 3, "{} levels in a sub-block", lv.len());
    }
    let err: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum();
    let tot: f32 = x.iter().map(|a| a * a).sum();
    assert!(err < tot, "decode no better than zero: {err} vs {tot}");
}
