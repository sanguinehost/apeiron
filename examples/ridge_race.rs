//! `examples/ridge_race.rs` — the sparse fold + direct solve raced against the spinor-ttl
//! reference (GPU fold vs full-batch Adam). The reference's `make_data` output is dumped to raw
//! files (scratch python, `spinor-ttl`'s own `.venv`) so both arms see identical bytes; this
//! example reads those files, runs [`FoldTrainer::fold_sparse_par`] + [`FoldTrainer::solve_par`],
//! times the fold and the solve *separately* over 5 runs (median and spread), and compares the f64
//! objective of the apeiron head to the reference fold objective.
//!
//! `threads = 0` runs the pre-change serial attribution (`fold_sparse` + scalar `solve_spd`) so the
//! before/after split is measured on one binary; `threads >= 1` runs the parallel arm.
//!
//! Run: `ridge_race [data_dir] [threads]` (defaults `~/ridge-race/data` and the
//! process's available parallelism).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "a measurement harness: raw-byte decoding and f64 statistics legitimately cross the \
              integer/float and width boundaries, the flat row-major kernels read as `r/i/k`, and \
              `main` is one linear per-config pass that keeps the attribution and race tables \
              readable from a single loop."
)]

use std::path::Path;
use std::time::Instant;

use apeiron::learn::ridge::{solve_spd, FoldTrainer};

fn read_u32(p: &Path) -> Vec<u32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32(p: &Path) -> Vec<f32> {
    let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// `Σ_r (Σ_k vals[r,k]·w[idx_k] − y_r)² + λ·Σ_i w_i²` in f64, from the sparse support.
fn objective_f64(
    idxs: &[u32],
    vals: &[f32],
    y: &[f32],
    w: &[f64],
    lam: f64,
    rows: usize,
    nnz: usize,
) -> f64 {
    let mut obj = 0.0f64;
    for i in 0..w.len() {
        obj += lam * w[i] * w[i];
    }
    for r in 0..rows {
        let mut pred = 0.0f64;
        for k in 0..nnz {
            pred += f64::from(vals[r * nnz + k]) * w[idxs[r * nnz + k] as usize];
        }
        let res = pred - f64::from(y[r]);
        obj += res * res;
    }
    obj
}

/// Median and (max − min) spread over `v`, sorted in place.
fn median_spread(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[v.len() - 1] - v[0])
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .unwrap_or_else(|| "~/ridge-race/data".to_string());
    let dir = Path::new(&dir);
    let avail = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let threads: usize = args.next().map_or(avail, |s| s.parse().unwrap());
    let serial = threads == 0;

    println!("cpu: {}", cpu_model());
    println!(
        "available_parallelism: {avail}   arm: {}   threads: {}   runs: 5",
        if serial {
            "serial fold_sparse + scalar solve_spd (pre-change)"
        } else {
            "parallel fold_sparse_par + blocked solve_par"
        },
        threads
    );

    for ci in 0..3usize {
        let meta = std::fs::read_to_string(dir.join(format!("c{ci}_meta.txt"))).unwrap();
        let (l1, l2) = {
            let mut lines = meta.lines();
            (
                lines.next().unwrap().split_whitespace().collect::<Vec<_>>(),
                lines.next().unwrap().split_whitespace().collect::<Vec<_>>(),
            )
        };
        let rows: usize = l1[0].parse().unwrap();
        let d: usize = l1[1].parse().unwrap();
        let nnz: usize = l1[2].parse().unwrap();
        let lam: f64 = l1[3].parse().unwrap();
        let obj_fold: f64 = l2[0].parse().unwrap();
        let obj_grad: f64 = l2[1].parse().unwrap();
        let iters: usize = l2[2].parse().unwrap();

        let idxs = read_u32(&dir.join(format!("c{ci}_idxs.u32")));
        let vals = read_f32(&dir.join(format!("c{ci}_vals.f32")));
        let y = read_f32(&dir.join(format!("c{ci}_y.f32")));
        let w_ref = read_f32(&dir.join(format!("c{ci}_w.f32")));
        assert_eq!(idxs.len(), rows * nnz, "idxs length");
        assert_eq!(vals.len(), rows * nnz, "vals length");
        assert_eq!(y.len(), rows, "y length");
        assert_eq!(w_ref.len(), d, "w_ref length");

        // The owner-partitioned fold allocates no per-worker partial, so the old partial-count cap
        // (keep `T·d²` accumulator bytes within a few L3 sizes) no longer applies. The fold's
        // redundant index re-scan is latency-bound and scales well past the solve's thread count, so
        // it gets `7/4` the requested threads (capped at the machine's parallelism) — on the
        // benchmark box `16 → 28`, which measured the lowest fold at every config; the solve keeps
        // the requested count, where contention on the trailing update stops paying. `FOLD_THREADS`
        // overrides for tuning; `serial` runs the single-threaded reference.
        let fold_threads = if serial {
            0
        } else {
            std::env::var("FOLD_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or((threads * 7 / 4).min(avail))
        };

        // One untimed warmup (page-faults the fresh `d²` accumulator), then 5 timed runs. Fold and
        // solve are timed separately so the attribution is per stage.
        {
            let mut t = FoldTrainer::new(d, 1, lam, false);
            if serial {
                t.fold_sparse(&idxs, &vals, &y, rows, nnz).unwrap();
            } else {
                t.fold_sparse_par(&idxs, &vals, &y, rows, nnz, fold_threads)
                    .unwrap();
            }
            let _ = t.solve(None).unwrap();
        }
        let mut fold_times = Vec::with_capacity(5);
        let mut solve_times = Vec::with_capacity(5);
        let mut total_times = Vec::with_capacity(5);
        let mut head_last: Option<Vec<f64>> = None;
        for _ in 0..5 {
            let mut t = FoldTrainer::new(d, 1, lam, false);
            let t0 = Instant::now();
            if serial {
                t.fold_sparse(&idxs, &vals, &y, rows, nnz).unwrap();
            } else {
                t.fold_sparse_par(&idxs, &vals, &y, rows, nnz, fold_threads)
                    .unwrap();
            }
            let t1 = Instant::now();
            let w: Vec<f64> = if serial {
                let mut a = t.drag().to_vec();
                for i in 0..d {
                    a[i * d + i] += lam;
                }
                solve_spd(&a, t.cross_term(), d, 1).unwrap()
            } else {
                let head = t.solve_par(None, threads).unwrap();
                head.weight
            };
            let t2 = Instant::now();
            fold_times.push((t1 - t0).as_secs_f64() * 1e3);
            solve_times.push((t2 - t1).as_secs_f64() * 1e3);
            total_times.push((t2 - t0).as_secs_f64() * 1e3);
            head_last = Some(w);
        }
        let (fold_med, fold_spread) = median_spread(&mut fold_times);
        let (solve_med, solve_spread) = median_spread(&mut solve_times);
        let (total_med, total_spread) = median_spread(&mut total_times);
        let weight = head_last.unwrap();

        let obj = objective_f64(&idxs, &vals, &y, &weight, lam, rows, nnz);
        let rel = (obj - obj_fold).abs() / obj_fold.abs().max(1.0);
        let mut max_dw = 0.0f64;
        for i in 0..d {
            max_dw = max_dw.max((weight[i] - f64::from(w_ref[i])).abs());
        }

        println!(
            "config {ci}: rows={rows} D={d} nnz={nnz} lam={lam} fold_threads={fold_threads} solve_threads={threads}"
        );
        println!(
            "  apeiron fold ms:      [{} {} {} {} {}]  median={fold_med:.3} spread={fold_spread:.3}",
            fold_times[0], fold_times[1], fold_times[2], fold_times[3], fold_times[4]
        );
        println!(
            "  apeiron solve ms:     [{} {} {} {} {}]  median={solve_med:.3} spread={solve_spread:.3}",
            solve_times[0], solve_times[1], solve_times[2], solve_times[3], solve_times[4]
        );
        println!("  apeiron fold+solve ms: median={total_med:.3} spread={total_spread:.3}");
        println!(
            "  obj apeiron(f64)={obj:.6}  obj ref fold={obj_fold:.6}  obj ref grad={obj_grad:.6}  iters={iters}  rel_diff={rel:.3e}"
        );
        println!("  max|dW| apeiron vs ref-fold = {max_dw:.3e}");
    }
}
