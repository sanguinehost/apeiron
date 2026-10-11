//! Byte-identity control for the `IQ1_S` port: quantize `blk.<lo..=hi>.ffn_down.weight` of an f16 GGUF with
//! [`apeiron::learn::iq1s`] under a llama.cpp imatrix and compare to the same tensors of a `llama-quantize`
//! `IQ1_S` file.
//!
//! `iq1s_check <f16.gguf> <imatrix.gguf> <iq1_s.gguf> <lo> <hi>`
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "control harness: flat row-major index arithmetic"
)]

use apeiron::format::gguf::Gguf;
use apeiron::format::quant::f16_to_f32;
use apeiron::learn::imatrix::read_imatrix;
use apeiron::learn::iq1s::{quantize_row, Grid, BLOCK_BYTES, QK_K};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() == 6, "usage: iq1s_check f16 imatrix iq1_s lo hi");
    let (lo, hi): (usize, usize) = (a[4].parse().unwrap(), a[5].parse().unwrap());
    let f16 = Gguf::open(&a[1]).unwrap();
    let refg = Gguf::open(&a[3]).unwrap();
    let im = read_imatrix(&a[2]).unwrap();
    let t0 = Instant::now();
    let grid = Grid::new();
    eprintln!("grid built in {:.2}s", t0.elapsed().as_secs_f64());
    let (mut tot_blocks, mut bad_blocks) = (0usize, 0usize);
    for blk in lo..=hi {
        let name = format!("blk.{blk}.ffn_down.weight");
        let td = f16.header.tensor(&name).unwrap();
        let (n, rows) = (td.dims[0] as usize, td.dims[1] as usize);
        let raw = f16.tensor_bytes(td).unwrap();
        let w: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let qw = im.vector(&name).unwrap();
        assert_eq!(qw.len(), n);
        let rb = n / QK_K * BLOCK_BYTES;
        let mut out = vec![0u8; rows * rb];
        let nt = 16usize;
        let per = rows.div_ceil(nt);
        std::thread::scope(|s| {
            for (ti, chunk) in out.chunks_mut(per * rb).enumerate() {
                let (w, grid) = (&w, &grid);
                s.spawn(move || {
                    for (ri, orow) in chunk.chunks_mut(rb).enumerate() {
                        let r = ti * per + ri;
                        quantize_row(grid, &w[r * n..(r + 1) * n], qw, orow);
                    }
                });
            }
        });
        let rt = refg.header.tensor(&name).unwrap();
        assert_eq!(rt.ggml_type, 19, "{name} is not IQ1_S in the reference");
        let want = refg
            .tensor_bytes_prefix(
                rt,
                (rt.n_elements().unwrap() as usize / QK_K * BLOCK_BYTES) as u64,
            )
            .unwrap();
        assert_eq!(want.len(), out.len());
        let nb = out.len() / BLOCK_BYTES;
        let bad = (0..nb)
            .filter(|&i| {
                out[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES]
                    != want[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES]
            })
            .count();
        println!("{name}: {nb} blocks, {bad} differ");
        tot_blocks += nb;
        bad_blocks += bad;
    }
    println!(
        "TOTAL {tot_blocks} blocks, {bad_blocks} differ, wall {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    std::process::exit(i32::from(bad_blocks != 0));
}
