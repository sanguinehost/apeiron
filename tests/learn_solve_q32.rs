//! The shared `f64` ↔ Q32.32 boundary: rounding, refusals, the power-of-two fill and panel shift.

use apeiron::learn::solve::q32::{
    fill_scale, from_grid, panel_shift, to_grid, GRID_BOUND, GRID_SCALE, INPUT_BOUND,
};

#[test]
fn round_trip_is_within_half_ulp_of_the_grid() {
    for &x in &[0.0, 1.0, -1.0, 0.1, -3.25e-7, 1234.5678, 15.999] {
        let q = to_grid(x, GRID_BOUND).expect("in range");
        assert!((from_grid(q) - x).abs() <= 0.5 / GRID_SCALE, "x={x}");
    }
}

#[test]
fn refuses_non_finite_and_out_of_bound() {
    for &x in &[
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        16.0,
        -16.0,
        1e30,
    ] {
        assert_eq!(to_grid(x, INPUT_BOUND), None, "x={x}");
    }
    assert!(to_grid(16.0, GRID_BOUND).is_some());
    assert_eq!(to_grid(GRID_BOUND, GRID_BOUND), None);
    assert_eq!(to_grid(-GRID_BOUND, GRID_BOUND), None);
}

#[test]
fn fill_scale_is_an_exact_power_of_two_filling_the_range() {
    assert!((fill_scale(0.0) - 1.0).abs() < f64::EPSILON);
    for &m in &[1e-12, 1e-3, 0.5, 1.0, 7.0, 1e3, 1e9, 3e12] {
        let s = fill_scale(m);
        assert_eq!(
            s.to_bits() & ((1u64 << 52) - 1),
            0,
            "m={m}: not a power of two"
        );
        let scaled = m * s;
        assert!(
            scaled > 1_073_741_823.0 && scaled <= 2_147_483_647.0,
            "m={m}: scaled={scaled}"
        );
        assert!(to_grid(scaled, GRID_BOUND).is_some());
    }
}

#[test]
fn panel_shift_brings_the_maximum_under_eight() {
    assert_eq!(panel_shift(0.0), 0);
    assert_eq!(panel_shift(8.0), 0);
    for &m in &[8.5, 16.0, 100.0, 1e6] {
        let k = panel_shift(m);
        assert!(k >= 1 && m * 0.5f64.powi(k) <= 8.0, "m={m} k={k}");
        assert!(m * 0.5f64.powi(k - 1) > 8.0, "m={m} k={k} not least");
    }
}
