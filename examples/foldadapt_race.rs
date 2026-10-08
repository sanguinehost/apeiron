//! apeiron `FoldAdapt` race arm: the `blend` contender of
//! `spinor-ttl/bench/finetune_race.py` @ `a05bb1b`, evaluated on the out-of-tree
//! dump written by `dump.py`.
//!
//! Reads `train_ids.u32` (folded once into a [`FoldAdapter`] of order 3),
//! `eval_ids.u32`, and `base_logp.f64` (the base-softmax probability of the true
//! next eval token, already reduced to the single index the refinement reads, in
//! float32 as `blend_bits` computes it). Two arms: `pr_max = f64::INFINITY` (the
//! race has no doorway gate) and `pr_max = 2.0` (the [`DOORWAY`] participation-ratio
//! cut). Metric: bits per held-out byte over `EVAL_N = 8192` bytes, identical to
//! every contender.
//!
//! # Control
//!
//! `FOLDADAPT_ZERO_DISCOUNTS=1` forces the absolute discounts to `(0, 0, 0)`; the
//! `pr_max = inf` b/B must then move away from the Python `blend` value 0.7244.
//!
//! # Usage
//!
//! `cargo run --release --example foldadapt_race -- <dumpdir>`
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "measurement harness: integer counts and stream lengths are cast to f64 for \
              probability normalisation; the numeric casts are the point of the example"
)]

use std::fs;
use std::path::Path;
use std::time::Instant;

use apeiron::learn::foldadapt::{FoldAdapter, DOORWAY};

/// Held-out tail length in bytes; the denominator every contender divides by.
const EVAL_N: usize = 8192;

/// Read a little-endian `u32` vector (token ids).
///
/// # Complexity
///
/// Temporal: `O(n)`. Spatial: `O(n)` words.
fn read_u32_le(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read a little-endian `f64` vector (base probabilities of the true token).
///
/// # Complexity
///
/// Temporal: `O(n)`. Spatial: `O(n)` words.
fn read_f64_le(path: &Path) -> Vec<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

/// Score one arm: `sum_t -log2(max(p_new[y_t], 1e-30))`, returned as `(b/B, per-position bits)`.
///
/// `base[y_t] = base_logp[t]` with zeros elsewhere. Only index `y_t` is read from the
/// refined distribution, the KN lane's `gamma` rescaling is elementwise, and every
/// continuation add lands on an index `< base.len()`; the remaining entries therefore
/// cannot change `p_new[y_t]`, so a one-hot base vector is exact for the metric.
///
/// # Complexity
///
/// Temporal: `Theta(P * (V + k * D))` where `P = base_logp.len()` eval positions,
/// `V = base.len()`, and `k <= 3` lanes over at most `D <= V` continuations.
/// Derivation: one [`FoldAdapter::refine_sparse`] per position; each clones and
/// rescales the full length-`V` vector once per applying order.
/// Spatial: `O(P)` per-position bits plus one length-`V` scratch vector (caller-owned).
fn score_arm(
    adapt: &FoldAdapter,
    eval: &[u32],
    base_logp: &[f64],
    base: &mut [f64],
    d: (f64, f64, f64),
    pr_max: f64,
) -> (f64, Vec<f64>) {
    let mut pos = Vec::with_capacity(base_logp.len());
    let mut prev_y: Option<usize> = None;
    for (t, &bp) in base_logp.iter().enumerate() {
        let y = eval[t + 1] as usize;
        if let Some(py) = prev_y {
            base[py] = 0.0;
        }
        base[y] = bp;
        prev_y = Some(y);
        // ctx = eval tokens before position t+1; refine_sparse reads the last `o` of them.
        let out = adapt
            .refine_sparse(base, &eval[..=t], d, pr_max)
            .expect("non-empty base");
        pos.push(-out[y].max(1e-30).log2());
    }
    let bits: f64 = pos.iter().sum();
    (bits / EVAL_N as f64, pos)
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: foldadapt_race <dumpdir>");
    let dir = Path::new(&dir);

    let train = read_u32_le(&dir.join("train_ids.u32"));
    let eval = read_u32_le(&dir.join("eval_ids.u32"));
    let base_logp = read_f64_le(&dir.join("base_logp.f64"));
    assert_eq!(
        base_logp.len() + 1,
        eval.len(),
        "base_logp must have eval_tokens - 1 entries"
    );

    let mut adapt = FoldAdapter::new(3).expect("order 3 is within 1..=MAX_ORDER");
    let t0 = Instant::now();
    adapt.fold(&train);
    let fold_s = t0.elapsed().as_secs_f64();

    let mut d = adapt.discounts();
    let zeroed = std::env::var("FOLDADAPT_ZERO_DISCOUNTS").is_ok();
    if zeroed {
        d = (0.0, 0.0, 0.0);
    }

    // Base-vector length: one past the largest token id seen. Only index `y` is read,
    // and the KN rescale is elementwise, so this equals the full-vocab result for `y`.
    let v = eval
        .iter()
        .chain(&train)
        .map(|&x| x as usize)
        .max()
        .map_or(0, |m| m + 1);
    let mut base = vec![0.0f64; v];

    let t0 = Instant::now();
    let (bpb_inf, pos_inf) = score_arm(&adapt, &eval, &base_logp, &mut base, d, f64::INFINITY);
    let score_inf_s = t0.elapsed().as_secs_f64();

    base.fill(0.0);
    let t0 = Instant::now();
    let (bpb_door, _pos_door) = score_arm(&adapt, &eval, &base_logp, &mut base, d, DOORWAY);
    let score_door_s = t0.elapsed().as_secs_f64();

    println!(
        "apeiron FoldAdapt (order 3) pr_max=inf : {bpb_inf:.6} b/B  fold {fold_s:.2}s  \
         score {score_inf_s:.2}s  positions {}  V={v}  d=({:.6},{:.6},{:.6})  \
         keys {}",
        pos_inf.len(),
        d.0,
        d.1,
        d.2,
        adapt.len(),
    );
    if zeroed {
        println!("  [CONTROL FOLDADAPT_ZERO_DISCOUNTS=1: discounts forced to (0,0,0)]");
    }
    println!(
        "apeiron FoldAdapt (order 3) pr_max=2.0 : {bpb_door:.6} b/B  fold {fold_s:.2}s  \
         score {score_door_s:.2}s  positions {}  V={v}",
        pos_inf.len(),
    );

    if let Ok(path) = std::env::var("FOLDADAPT_POS_OUT") {
        let mut buf = Vec::with_capacity(pos_inf.len() * 8);
        for b in &pos_inf {
            buf.extend_from_slice(&b.to_le_bytes());
        }
        fs::write(&path, buf).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("wrote {} per-position bits to {path}", pos_inf.len());
    }
}
