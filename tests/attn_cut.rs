//! `Apeiron.Numeric.AttentionCut` refinement, grade `tested`: `ratio_error`, `certified_read`,
//! `dropped_mass_bound`, `partition_error` over `f64` arithmetic on `f32` data.
//!
//! Rounding allowance: the Lean bound is over exact reals; `f64` rounding of the reader and of the
//! reference adds at most a few ulps of `V`, so the dominance checks allow `ROUND * (1 + V)` with
//! `ROUND = 1e-13`. That allowance is test-only; the library bound carries none.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test data generation: f64 draws are rounded to f32 on purpose; counts are below 2^53"
)]
#![allow(
    clippy::float_cmp,
    reason = "an exact zero bound and exact identity merges are the claims"
)]

use apeiron::attn_cut::{certified_read, CutError, Part};

const ROUND: f64 = 1e-13;

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n).unwrap()).unwrap()
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        f64::from(u32::try_from(self.next() >> 40).unwrap()) / f64::from(1u32 << 24)
    }
    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }
}

struct Instance {
    dim: usize,
    scores: Vec<f32>,
    values: Vec<f32>,
    /// max |v| over all keys.
    vmax: f64,
    gmax: f64,
    delta: f64,
    /// key -> group: Some(g) kept in exact group g, None dropped.
    kept_group: Vec<Option<usize>>,
    groups: usize,
}

/// Scores are multiples of 1/64, so adding an integer shift is exact in `f32`.
fn instance(rng: &mut Rng, shift: f32) -> Instance {
    let len = 1 + rng.below(60);
    let dim = 1 + rng.below(8);
    let spread = 0.5 + 24.0 * rng.unit();
    let delta = if rng.below(8) == 0 {
        0.0
    } else {
        20.0 * rng.unit()
    };
    let vscale = 0.1 + 10.0 * rng.unit();
    let mut scores = Vec::with_capacity(len);
    for _ in 0..len {
        let s = ((2.0 * rng.unit() - 1.0) * spread * 64.0).round() / 64.0;
        scores.push(s as f32 + shift);
    }
    let values: Vec<f32> = (0..len * dim)
        .map(|_| ((2.0 * rng.unit() - 1.0) * vscale) as f32)
        .collect();
    let vmax = values
        .iter()
        .map(|v| f64::from(v.abs()))
        .fold(0.0, f64::max);
    let gmax = scores
        .iter()
        .map(|&s| f64::from(s))
        .fold(f64::NEG_INFINITY, f64::max);
    let groups = 1 + rng.below(4);
    // A key may be dropped only when it sits `delta` below the global max, which the kept set contains.
    let kept_group = scores
        .iter()
        .map(|&s| {
            let droppable = f64::from(s) <= gmax - delta && f64::from(s) < gmax;
            if droppable && rng.below(3) != 0 {
                None
            } else {
                Some(rng.below(groups))
            }
        })
        .collect();
    Instance {
        dim,
        scores,
        values,
        vmax,
        gmax,
        delta,
        kept_group,
        groups,
    }
}

fn parts_of(inst: &Instance, rng: &mut Rng) -> Vec<Part> {
    let mut parts = Vec::new();
    for g in 0..inst.groups {
        let idx: Vec<usize> = (0..inst.scores.len())
            .filter(|&i| inst.kept_group[i] == Some(g))
            .collect();
        let sc: Vec<f32> = idx.iter().map(|&i| inst.scores[i]).collect();
        let vs: Vec<f32> = idx
            .iter()
            .flat_map(|&i| {
                inst.values[i * inst.dim..(i + 1) * inst.dim]
                    .iter()
                    .copied()
            })
            .collect();
        // each exact group in its own frame: its own max, or a random higher frame
        let own = sc
            .iter()
            .map(|&s| f64::from(s))
            .fold(f64::NEG_INFINITY, f64::max);
        let frame = if own.is_finite() && rng.below(2) == 0 {
            own + 3.0 * rng.unit()
        } else {
            own
        };
        parts.push(Part::exact(
            &sc,
            &vs,
            inst.dim,
            if own.is_finite() { frame } else { 0.0 },
        ));
    }
    let dropped = inst.kept_group.iter().filter(|g| g.is_none()).count();
    // split the dropped set into two summaries to exercise merge of dropped parts
    let first = dropped / 2;
    parts.push(Part::dropped(
        inst.dim, first, inst.delta, inst.vmax, inst.gmax,
    ));
    parts.push(Part::dropped(
        inst.dim,
        dropped - first,
        inst.delta,
        inst.vmax,
        inst.gmax,
    ));
    parts
}

fn reference(inst: &Instance) -> Vec<f64> {
    let mut z = 0.0;
    let mut n = vec![0.0; inst.dim];
    for (i, &s) in inst.scores.iter().enumerate() {
        let w = (f64::from(s) - inst.gmax).exp();
        z += w;
        for (j, acc) in n.iter_mut().enumerate() {
            *acc += w * f64::from(inst.values[i * inst.dim + j]);
        }
    }
    n.iter().map(|x| x / z).collect()
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

const INSTANCES: usize = 4000;

#[test]
fn bound_dominates_realised_error() {
    let mut rng = Rng(0xA77E_C0DE);
    let mut ratios = Vec::new();
    let mut with_drops = 0usize;
    for _ in 0..INSTANCES {
        let inst = instance(&mut rng, 0.0);
        let parts = parts_of(&inst, &mut rng);
        let (out, bound) = certified_read(&parts, inst.vmax).unwrap();
        let realised = max_err(&out, &reference(&inst));
        assert!(
            realised <= bound + ROUND * (1.0 + inst.vmax),
            "realised {realised:e} exceeds bound {bound:e}"
        );
        if inst.kept_group.iter().any(Option::is_none) {
            with_drops += 1;
            if realised > 1e-9 {
                ratios.push(bound / realised);
            }
        }
    }
    assert!(
        with_drops > INSTANCES / 4,
        "too few instances exercise a dropped part: {with_drops}"
    );
    ratios.sort_by(f64::total_cmp);
    let med = ratios[ratios.len() / 2];
    println!(
        "tightness: {} instances with drops, {} with realised > 1e-9; median bound/realised = {med:.3}, \
         min = {:.3}, max = {:.3e}",
        with_drops,
        ratios.len(),
        ratios[0],
        ratios[ratios.len() - 1]
    );
}

#[test]
fn dropped_bound_is_at_most_two_v_count_exp_minus_delta() {
    // dropped_mass_bound: kept frame is the kept max, so Z' >= 1 and the bound is <= 2 V |D| e^{-delta}.
    let mut rng = Rng(0xD0D0);
    let mut checked = 0;
    for _ in 0..INSTANCES {
        let inst = instance(&mut rng, 0.0);
        let dropped = inst.kept_group.iter().filter(|g| g.is_none()).count();
        if dropped == 0 {
            continue;
        }
        let parts = parts_of(&inst, &mut rng);
        let (_, bound) = certified_read(&parts, inst.vmax).unwrap();
        let lean = 2.0 * inst.vmax * dropped as f64 * (-inst.delta).exp();
        assert!(
            bound <= lean * (1.0 + 1e-12) + 1e-300,
            "bound {bound:e} > Lean {lean:e}"
        );
        checked += 1;
    }
    assert!(checked > 500);
}

#[test]
fn merge_is_order_independent() {
    let mut rng = Rng(0x0BDE);
    for _ in 0..INSTANCES {
        let inst = instance(&mut rng, 0.0);
        let mut parts = parts_of(&inst, &mut rng);
        let (o1, b1) = certified_read(&parts, inst.vmax).unwrap();
        rng.shuffle(&mut parts);
        let (o2, b2) = certified_read(&parts, inst.vmax).unwrap();
        let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1e-300);
        assert!(
            rel(b1, b2) <= 1e-12 || (b1 - b2).abs() <= 1e-300,
            "bound {b1:e} vs {b2:e}"
        );
        for (x, y) in o1.iter().zip(&o2) {
            assert!(
                (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0),
                "out {x:e} vs {y:e}"
            );
        }
    }
}

#[test]
fn max_shift_invariance() {
    for (seed, c) in [(0x5417u64, 8.0f32), (0x5418, -16.0), (0x5419, 100.0)] {
        let (mut r0, mut r1) = (Rng(seed), Rng(seed));
        for _ in 0..INSTANCES / 2 {
            let base = instance(&mut r0, 0.0);
            let shifted = instance(&mut r1, c);
            assert_eq!(base.values, shifted.values);
            // the random streams are identical, so the structure is identical and only frames move
            let (mut p0, mut p1) = (Rng(seed ^ 1), Rng(seed ^ 1));
            let a = parts_of(&base, &mut p0);
            let b = parts_of(&shifted, &mut p1);
            let (oa, ba) = certified_read(&a, base.vmax).unwrap();
            let (ob, bb) = certified_read(&b, shifted.vmax).unwrap();
            let rel = |x: f64, y: f64| (x - y).abs() / x.abs().max(y.abs()).max(1e-300);
            assert!(
                rel(ba, bb) <= 1e-12 || (ba - bb).abs() <= 1e-300,
                "bound {ba:e} vs {bb:e}"
            );
            for (x, y) in oa.iter().zip(&ob) {
                assert!(
                    (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0),
                    "out {x:e} vs {y:e}"
                );
            }
        }
    }
}

#[test]
fn exact_parts_have_bound_zero() {
    let mut rng = Rng(0xE0AC7);
    for _ in 0..INSTANCES {
        let mut inst = instance(&mut rng, 0.0);
        inst.kept_group.iter_mut().for_each(|g| *g = Some(0));
        inst.groups = 1 + rng.below(4);
        let len = inst.scores.len();
        for i in 0..len {
            inst.kept_group[i] = Some(rng.below(inst.groups));
        }
        let parts = parts_of(&inst, &mut rng);
        let (out, bound) = certified_read(&parts, inst.vmax).unwrap();
        assert_eq!(bound, 0.0);
        assert!(max_err(&out, &reference(&inst)) <= ROUND * (1.0 + inst.vmax));
    }
}

#[test]
fn empty_part_is_identity() {
    let sc = [0.5f32, -1.0];
    let vs = [1.0f32, 2.0, 3.0, 4.0];
    let p = Part::exact(&sc, &vs, 2, 0.5);
    let e = Part::empty(2);
    assert_eq!(p.merge(&e), p);
    assert_eq!(e.merge(&p), p);
    assert_eq!(Part::exact(&[], &[], 2, 0.0), e);
    assert_eq!(Part::dropped(2, 0, 3.0, 1.0, 0.0), e);
    assert_eq!(e.merge(&e), e);
    assert_eq!(
        certified_read(std::slice::from_ref(&e), 1.0),
        Err(CutError::NonPositiveDenominator)
    );
    assert_eq!(certified_read(&[], 1.0), Err(CutError::NoParts));
}

#[test]
fn single_key_reads_its_value_with_zero_bound() {
    let p = Part::exact(&[3.25], &[1.5, -2.0], 2, 3.25);
    let (out, bound) = certified_read(&[p], 2.0).unwrap();
    assert_eq!(out, vec![1.5, -2.0]);
    assert_eq!(bound, 0.0);
}

#[test]
fn nonpositive_denominator_takes_the_failure_branch() {
    // only a dropped summary: Z' = 0
    let d = Part::dropped(1, 5, 2.0, 1.0, 0.0);
    assert_eq!(
        certified_read(&[d], 1.0),
        Err(CutError::NonPositiveDenominator)
    );
    // kept part underflowed to z = 0 after rebasing to a far higher frame
    let kept = Part::exact(&[0.0], &[1.0], 1, 0.0);
    let far = Part::dropped(1, 1, 0.0, 1.0, 1.0e4);
    assert_eq!(
        certified_read(&[kept, far], 1.0),
        Err(CutError::NonPositiveDenominator)
    );
    // non-finite and negative V
    let p = Part::exact(&[0.0], &[1.0], 1, 0.0);
    assert_eq!(
        certified_read(std::slice::from_ref(&p), -1.0),
        Err(CutError::BadValueBound)
    );
    assert_eq!(certified_read(&[p], f64::NAN), Err(CutError::BadValueBound));
}

#[test]
fn delta_zero_ties_are_certified() {
    // two keys tied at the max; one dropped with delta = 0: still within the bound
    let sc = [1.0f32, 1.0];
    let vs = [1.0f32, -1.0];
    let kept = Part::exact(&sc[..1], &vs[..1], 1, 1.0);
    let dropped = Part::dropped(1, 1, 0.0, 1.0, 1.0);
    let (out, bound) = certified_read(&[kept, dropped], 1.0).unwrap();
    let truth = 0.0; // (1 - 1) / 2
    assert!(
        (out[0] - truth).abs() <= bound,
        "out {} bound {bound}",
        out[0]
    );
    assert!((bound - 2.0).abs() < 1e-15, "bound {bound}"); // (1*1 + 1*1) / 1
}
