//! Board APEIRON-QUALITY-118-W1-T01. `apeiron::learn::nanoquant`: the `NanoQuant` ADMM factor
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `core/admm_nq.py`).
//!
//! Positive tests on small Gaussian matrices (rank formula against the bit budget, the export's
//! dense form, the wide-matrix transpose path, loss falling with the iteration count), and one
//! ignored stage that fits the teacher's linears of `Qwen3.5-0.8B` under the cached calibration
//! Grams and writes `l{l}.{kind}-nq{tag}.i64` for `learn_gate_code`'s `gate_code_kl`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    reason = "test geometry: single letters are the reference's symbols, loops are index maps"
)]

use std::fmt::Write as _;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::learn::grad::load::layer_from_engine;
use apeiron::learn::grad::model::Config;
use apeiron::learn::nanoquant::{
    factorize, rank_for_bits, stored_bits, NqConfig, NqFactor, Schedule,
};
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

const MODEL: &str = "/home/socol/oc-tmp/kld08/qwen08b-f16.gguf";
const CACHE: &str = "/home/socol/oc-tmp/kld08/gatecode-tok";

fn gauss(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / 9_007_199_254_740_992.0
    };
    (0..n)
        .map(|_| {
            let (a, b) = (next().max(1e-300), next());
            scale * (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        })
        .collect()
}

fn words(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("grid"))
        .collect()
}

/// The dense form by definition: `s_post[i]·Σ_k U[i][k]·V[k][j]·s_pre[j]`.
fn dense(f: &NqFactor) -> Vec<f64> {
    let (o, n, r) = (f.out, f.inn, f.rank);
    let mut w = vec![0.0; o * n];
    for i in 0..o {
        for j in 0..n {
            let mut acc = 0i64;
            for k in 0..r {
                acc += i64::from(f.u[i * r + k]) * i64::from(f.v[k * n + j]);
            }
            w[i * n + j] = f.s_post[i] * acc as f64 * f.s_pre[j];
        }
    }
    w
}

#[test]
fn rank_fits_budget_and_is_maximal() {
    // Qwen3-0.6B projections and the 0.8B proxy's, at 1.00 and 1.18 bits per weight.
    for &(a, b) in &[
        (2048, 1024),
        (1024, 2048),
        (3072, 1024),
        (1024, 3072),
        (1024, 3584),
        (3584, 1024),
    ] {
        for &ppm in &[1_000_000u64, 1_180_000] {
            let r = rank_for_bits(a, b, ppm, 32);
            assert_eq!(r % 32, 0);
            let budget = (a * b) as u128 * u128::from(ppm);
            assert!(
                u128::from(stored_bits(a, b, r)) * 1_000_000 <= budget,
                "{a}x{b} r {r}"
            );
            assert!(
                u128::from(stored_bits(a, b, r + 32)) * 1_000_000 > budget,
                "{a}x{b} r {r}+32"
            );
        }
    }
    // A budget below the scale vectors falls back to the minimum rank.
    assert_eq!(rank_for_bits(64, 64, 100_000, 32), 32);
}

#[test]
fn export_matches_its_dense_form_and_loss_falls() {
    let (o, n, r) = (96, 64, 32);
    let w = gauss(o * n, 1, 0.05);
    let i_norm: Vec<f64> = gauss(n, 2, 1.0).iter().map(|x| 0.5 + x * x).collect();
    let o_norm = vec![1.0; o];
    let ww = words(&w);
    let mut cfg = NqConfig::reference(r);
    cfg.outer_iters = 2;
    let short = factorize(&ww, o, n, &i_norm, &o_norm, &cfg).expect("2 iterations");
    cfg.outer_iters = 60;
    let long = factorize(&ww, o, n, &i_norm, &o_norm, &cfg).expect("60 iterations");
    assert!(
        long.admm_loss < short.admm_loss,
        "{} !< {}",
        long.admm_loss,
        short.admm_loss
    );
    assert!(long.admm_loss < 0.9, "loss {}", long.admm_loss);
    assert!(long.u.iter().chain(&long.v).all(|&s| s == 1 || s == -1));
    let kernel = long.weight().expect("weight");
    let defn = dense(&long);
    let max_w = w.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (k, d) in kernel.iter().zip(&defn) {
        assert!(
            (from_grid(*k) - d).abs() <= 1e-6 * max_w,
            "{} vs {d}",
            from_grid(*k)
        );
    }
}

#[test]
fn wide_matrix_runs_on_the_transpose() {
    let (o, n, r) = (48, 80, 32);
    let w = words(&gauss(o * n, 3, 0.05));
    let cfg = NqConfig {
        outer_iters: 20,
        ..NqConfig::reference(r)
    };
    let f = factorize(&w, o, n, &vec![1.0; n], &vec![1.0; o], &cfg).expect("wide");
    assert_eq!((f.out, f.inn, f.u.len(), f.v.len()), (o, n, o * r, r * n));
    assert_eq!((f.s_post.len(), f.s_pre.len()), (o, n));
    assert_eq!(f.weight().expect("weight").len(), o * n);
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn read_words(path: &str) -> (usize, usize, Vec<i64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let o = u64::from_le_bytes(b[0..8].try_into().expect("o")) as usize;
    let n = u64::from_le_bytes(b[8..16].try_into().expect("n")) as usize;
    let w = b[16..]
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("word")))
        .collect();
    (o, n, w)
}

fn write_words(path: &str, o: usize, n: usize, w: &[i64]) {
    let mut b = Vec::with_capacity(16 + 8 * w.len());
    b.extend_from_slice(&(o as u64).to_le_bytes());
    b.extend_from_slice(&(n as u64).to_le_bytes());
    for x in w {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}

/// `tr(E H Eᵀ)` over rows, threaded: `O(o·n²)` multiply-adds.
fn energy(h: &[f64], e: &[f64], o: usize, n: usize) -> f64 {
    let th = 32usize;
    let per = o.div_ceil(th);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..th)
            .map(|t| {
                s.spawn(move || {
                    let mut acc = 0.0;
                    for u in (t * per)..((t + 1) * per).min(o) {
                        let row = &e[u * n..(u + 1) * n];
                        for i in 0..n {
                            let hi = &h[i * n..(i + 1) * n];
                            let mut d = 0.0;
                            for j in 0..n {
                                d += hi[j] * row[j];
                            }
                            acc += row[i] * d;
                        }
                    }
                    acc
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("thread")).sum()
    })
}

/// Fit stage. `APEIRON_NQ_KIND` (comma list of `gate`, `up`, `down`, `o`), `APEIRON_NQ_LAYERS`
/// (default `0,3,7,11,15,19,23`, the cached Gram layers), `APEIRON_NQ_ITERS` (400),
/// `APEIRON_NQ_BITS_PPM` (1000000), `APEIRON_NQ_SCHEDULE` (`linear`, or `cubic`), `APEIRON_NQ_TAG`. `i_norm = diag(H)` of the cached
/// `g{l}.c6x29.*` Gram (the reference's unclipped `dbf` statistic up to a constant, to which the
/// ADMM is equivariant); `o_norm = 1`. Prints the ADMM loss and the relative `H`-energy
/// `tr(EHEᵀ)/tr(WHWᵀ)` of the fit and of the cached same-scope `Q4_K_M` part.
#[test]
#[ignore = "opens the teacher and runs the ADMM per tensor; run explicitly"]
fn nq_fit() {
    let engine = Engine::open(MODEL).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let kinds = std::env::var("APEIRON_NQ_KIND").unwrap_or_else(|_| String::from("down"));
    let layers: Vec<usize> = std::env::var("APEIRON_NQ_LAYERS")
        .unwrap_or_else(|_| String::from("0,3,7,11,15,19,23"))
        .split(',')
        .map(|s| s.parse().expect("layer"))
        .collect();
    let iters = env_usize("APEIRON_NQ_ITERS", 400);
    let ppm = env_usize("APEIRON_NQ_BITS_PPM", 1_000_000) as u64;
    let tag = std::env::var("APEIRON_NQ_TAG").unwrap_or_default();
    for &l in &layers {
        let ll = layer_from_engine(&engine, &cfg, l).expect("layer linears");
        for kind in kinds.split(',') {
            let (names, gram): (&[&str], &str) = match kind {
                "gate" => (&["ffn_gate"], "gate_up"),
                "up" => (&["ffn_up"], "gate_up"),
                "down" => (&["ffn_down"], "down"),
                "o" => (&["attn_output", "ssm_out"], "o"),
                k => panic!("kind {k}"),
            };
            let lin = ll
                .linears
                .iter()
                .find(|x| names.contains(&x.name))
                .unwrap_or_else(|| panic!("blk.{l} {kind}"));
            let (o, n) = (lin.rows, lin.cols);
            let (ho, hn, hw) = read_words(&format!("{CACHE}/g{l}.c6x29.{gram}.i64"));
            assert_eq!((ho, hn), (n, n), "layer {l} {kind}: Gram");
            let h: Vec<f64> = hw.iter().map(|&x| from_grid(x)).collect();
            let i_norm: Vec<f64> = (0..n).map(|i| h[i * n + i]).collect();
            let r = rank_for_bits(o, n, ppm, 32);
            let c = NqConfig {
                outer_iters: iters,
                ..NqConfig::reference(r)
            };
            let t = Instant::now();
            let f = factorize(&lin.w, o, n, &i_norm, &vec![1.0; o], &c).expect("factorize");
            let fit_s = t.elapsed().as_secs_f64();
            let wq = f.weight().expect("weight");
            let path = format!("{CACHE}/l{l}.{kind}-nq{tag}.i64");
            write_words(&path, o, n, &wq);
            let w: Vec<f64> = lin.w.iter().map(|&x| from_grid(x)).collect();
            let base = energy(&h, &w, o, n);
            let e: Vec<f64> = wq.iter().zip(&w).map(|(&q, x)| from_grid(q) - x).collect();
            let rel = energy(&h, &e, o, n) / base;
            let mut line = format!(
                "NQ layer {l:02} {kind} {o}x{n} r {r} b/w {:.4} iters {iters} {fit_s:.1} s admm_loss {:.5} relH {rel:.5}",
                stored_bits(o, n, r) as f64 / (o * n) as f64,
                f.admm_loss
            );
            for cmp in std::env::var("APEIRON_NQ_CMP")
                .unwrap_or_else(|_| String::from("q4km"))
                .split(',')
            {
                let p = format!("{CACHE}/l{l}.{kind}-{cmp}.i64");
                if !std::path::Path::new(&p).exists() {
                    continue;
                }
                let (_, _, qw) = read_words(&p);
                let e: Vec<f64> = qw.iter().zip(&w).map(|(&q, x)| from_grid(q) - x).collect();
                let rc = energy(&h, &e, o, n) / base;
                let _ = write!(line, " | {cmp} relH {rc:.6} nq/{cmp} x{:.2}", rel / rc);
            }
            println!("{line}");
        }
    }
}
