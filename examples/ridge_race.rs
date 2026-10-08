//! `examples/ridge_race.rs` — the sparse fold + direct solve raced against the spinor-ttl
//! reference (GPU fold vs full-batch Adam). The reference's `make_data` output is dumped to raw
//! files (scratch python, `spinor-ttl`'s own `.venv`) so both arms see identical bytes; this
//! example reads those files, runs [`FoldTrainer::fold_sparse`] + [`FoldTrainer::solve`], times the
//! pair over 5 runs, and compares the f64 objective of the apeiron head to the reference fold
//! objective.
//!
//! Run: `ridge_race [data_dir]` (default `~/ridge-race/data`).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "a measurement harness: raw-byte decoding and f64 statistics legitimately cross the \
              integer/float and width boundaries, and the flat row-major kernels read as `r/i/k`."
)]

use std::path::Path;
use std::time::Instant;

use apeiron::learn::ridge::FoldTrainer;

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

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "~/ridge-race/data".to_string());
    let dir = Path::new(&dir);

    println!("cpu: {}", cpu_model());
    println!(
        "available_parallelism: {}   fold_sparse+solve: single-threaded (no rayon / no threads)",
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
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

        // One untimed warmup (page-faults the fresh `d²` accumulator), then 5 timed runs.
        {
            let mut t = FoldTrainer::new(d, 1, lam, false);
            t.fold_sparse(&idxs, &vals, &y, rows, nnz).unwrap();
            let _ = t.solve(None).unwrap();
        }
        let mut times = Vec::with_capacity(5);
        let mut head_last = None;
        for _ in 0..5 {
            let mut t = FoldTrainer::new(d, 1, lam, false);
            let t0 = Instant::now();
            t.fold_sparse(&idxs, &vals, &y, rows, nnz).unwrap();
            let head = t.solve(None).unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1e3);
            head_last = Some(head);
        }
        times.sort_by(f64::total_cmp);
        let median = times[2];
        let spread = times[4] - times[0];
        let head = head_last.unwrap();

        let obj = objective_f64(&idxs, &vals, &y, &head.weight, lam, rows, nnz);
        let rel = (obj - obj_fold).abs() / obj_fold.abs().max(1.0);
        let mut max_dw = 0.0f64;
        for i in 0..d {
            max_dw = max_dw.max((head.weight[i] - f64::from(w_ref[i])).abs());
        }

        println!("config {ci}: rows={rows} D={d} nnz={nnz} lam={lam}");
        println!(
            "  apeiron fold_sparse+solve ms: [{} {} {} {} {}]  median={median:.3} spread={spread:.3}",
            times[0], times[1], times[2], times[3], times[4]
        );
        println!(
            "  obj apeiron(f64)={obj:.6}  obj ref fold={obj_fold:.6}  obj ref grad={obj_grad:.6}  iters={iters}  rel_diff={rel:.3e}"
        );
        println!("  max|dW| apeiron vs ref-fold = {max_dw:.3e}");
    }
}
