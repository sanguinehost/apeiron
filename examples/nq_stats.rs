//! Pack geometry and the per-token projection arithmetic.
//!
//! `nq_stats --pack <nqp>` prints every linear's `out × inn` and rank, the per-kind totals, the
//! factor bytes, the dense rank-decomposed multiply-adds per token (`inn·rank + rank·out`) and the
//! dense `inn·out` it replaces, so a wall can be quoted against the arithmetic it actually ran.
use std::process::ExitCode;

use apeiron::format::nq;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nq_stats: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut path = String::new();
    let mut per_linear = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--pack" => path = val()?,
            "--per-linear" => per_linear = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if path.is_empty() {
        return Err("--pack is required".into());
    }
    let pack = nq::open(&path).map_err(|e| format!("{path}: {e:?}"))?;
    let linears: Vec<&nq::PackedLinear> = pack.linears().collect();
    let mut bits = 0u64;
    let mut macs = 0u64;
    let mut dense = 0u64;
    let mut rank_sum = 0usize;
    let mut res_macs = 0u64;
    let mut salient = 0u64;
    let mut res_layers = 0usize;
    for &l in &linears {
        bits += l.stored_bits();
        let m = l.inn as u64 * l.rank as u64 + l.rank as u64 * l.out as u64;
        macs += m;
        res_macs += l.res_layers as u64 * l.inn as u64 * l.out as u64;
        salient += l.k() as u64 * l.out as u64;
        res_layers += l.res_layers;
        if l.res_layers > 0 {
            eprintln!(
                "residual: layer {} kind {} out {} inn {} layers {}",
                l.layer, l.kind, l.out, l.inn, l.res_layers
            );
        }
        dense += l.inn as u64 * l.out as u64;
        rank_sum += l.rank;
        if per_linear {
            println!(
                "layer {:2} kind {:>2} out {:5} inn {:5} rank {:4} salient {:3} macs {:9}",
                l.layer,
                l.kind,
                l.out,
                l.inn,
                l.rank,
                l.k(),
                m
            );
        }
    }
    let n = linears.len();
    #[allow(
        clippy::cast_precision_loss,
        reason = "display only: byte and MAC totals"
    )]
    let gmacs = macs as f64 / 1e9;
    #[allow(clippy::cast_precision_loss, reason = "display only")]
    let gflop = 2.0 * gmacs;
    #[allow(clippy::cast_precision_loss, reason = "display only")]
    let gdense = 2.0 * dense as f64 / 1e9;
    println!(
        "linears {n}; factor bytes {}; bits/weight {:.6}; rank sum {rank_sum}",
        bits / 8,
        bits as f64 / dense as f64,
    );
    println!("rank-decomposed: {gmacs:.4} GMAC/token = {gflop:.3} GFLOP/token");
    println!("dense equivalent: {gdense:.3} GFLOP/token");
    #[allow(clippy::cast_precision_loss, reason = "display only: MAC totals")]
    {
        println!(
            "residual: {res_layers} layer-planes, {:.4} GMAC/token; salient {:.4} GMAC/token",
            res_macs as f64 / 1e9,
            salient as f64 / 1e9
        );
    }
    #[allow(clippy::cast_precision_loss, reason = "display only: a ratio")]
    {
        println!("density {:.3}", macs as f64 / dense as f64);
    }
    Ok(())
}
