//! Boundary, bit-exact, independent-oracle and falsifier tests for the
//! canonical elliptic Kepler kernel.
//!
//! The golden table is an independent reference, not produced by the kernel:
//! 110-digit solutions of `E = M + e sin E` from the spike golden grid
//! (`/tmp/apeiron/kepler-spike/data/golden.tsv`, instrument
//! `tools/golden.py`, self-checks V1-V5), read here as a double-double
//! `(hi, lo)` pair plus a 25-significant-digit decimal. The falsifier arms
//! below (naive placeholder schedule, naive phase accumulation) are measured
//! contrasts only.
//!
//! No Aeneas f64 refinement is claimed for this kernel or these tests.

use apeiron::cell::Sigma;
use apeiron::kepler::{solve_kepler, KeplerError};

/// `eps = 2^-52`.
const EPS: f64 = f64::EPSILON;

/// The binary64 service endpoint (`0.95_f64`).
const SERVICE_ECC: f64 = 0.95_f64;

/// `pi`'s binary64 approximation (admitted: its denotation is below `pi`).
const PI: f64 = std::f64::consts::PI;

/// Oracle reading slack: the double-double distance and the service ceilings
/// are compared with this allowance, far below every mismatch it is meant to
/// expose (the smallest such margin measured is ~1e-18).
const ORACLE_SLACK: f64 = 8.0e-30;

/// Exact error-free sum: `a + b = s + err` (Knuth two-sum). Local copy: the
/// integration test may only use the public API.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// Exact error-free product: `a * b = p + err` via one correctly rounded
/// `mul_add`. Local copy: the integration test may only use the public API.
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let err = a.mul_add(b, -p);
    (p, err)
}

/// Exact sign test: `true` iff `sum(vals) >= 0` over the reals. The two-sum
/// cascade keeps `lead + residuals = sum(vals)` exactly; a dominating lead
/// decides the sign, otherwise the (much smaller) exact residuals are cascaded
/// again. The cap panics rather than guessing.
fn exact_sum_nonnegative(vals: &[f64]) -> bool {
    let mut terms: Vec<f64> = vals.iter().copied().filter(|v| *v != 0.0).collect();
    for _ in 0..64 {
        if terms.is_empty() {
            return true;
        }
        let mut lead = terms[0];
        let mut rest: Vec<f64> = Vec::new();
        for &v in &terms[1..] {
            let (s, err) = two_sum(lead, v);
            lead = s;
            if err != 0.0 {
                rest.push(err);
            }
        }
        if lead == 0.0 {
            terms = rest;
            continue;
        }
        let mut magnitude = 0.0_f64;
        for &err in &rest {
            magnitude = (magnitude + err.abs()).next_up();
        }
        if lead.abs() > magnitude {
            return lead > 0.0;
        }
        rest.push(lead);
        terms = rest;
    }
    panic!("exact sum sign undecided after 64 cascade rounds");
}

/// The certificate invariant `error_bound * (1 - e) >= defect_bound` decided
/// over the EXACT reals (`1 - e = d0 + d1` and `r (d0 + d1)` decomposed by
/// `two_sum`/`two_prod`; no rounded division anywhere).
fn error_bound_covers_exact(defect_bound: f64, eccentricity: f64, error_bound: f64) -> bool {
    let (d0, d1) = two_sum(1.0, -eccentricity);
    let (p0, t0) = two_prod(error_bound, d0);
    let (p1, t1) = two_prod(error_bound, d1);
    exact_sum_nonnegative(&[p0, t0, p1, t1, -defect_bound])
}

/// One independent-oracle row: `(m, e, golden decimal, hi, lo)`.
///
/// `hi` is the binary64 nearest to the golden root `E*` and `lo` the binary64
/// nearest to `E* - hi`, so `(hi, lo)` reads `E*` to about 1e-32; the decimal
/// is `E*` to 25 significant digits (provenance: `golden.tsv` `E_hi`). At
/// `e == 0` the root is exactly the denoted `m`: `hi = m`, `lo = +0.0`.
const GOLDEN: &[(f64, f64, &str, u64, u64)] = &[
    (0.0, 0.0, "0E0", 0x0000000000000000, 0x0000000000000000),
    (
        0.001,
        0.0,
        "1.000000000000000020816682E-3",
        0x3F50624DD2F1A9FC,
        0x0000000000000000,
    ),
    (
        -0.001,
        0.0,
        "-1.000000000000000020816682E-3",
        0xBF50624DD2F1A9FC,
        0x0000000000000000,
    ),
    (
        1.5707963267948966,
        0.0,
        "1.570796326794896557998982E+0",
        0x3FF921FB54442D18,
        0x0000000000000000,
    ),
    (
        -1.5707963267948966,
        0.0,
        "-1.570796326794896557998982E+0",
        0xBFF921FB54442D18,
        0x0000000000000000,
    ),
    (
        3.141592653589793,
        0.0,
        "3.141592653589793115997963E+0",
        0x400921FB54442D18,
        0x0000000000000000,
    ),
    (
        -3.141592653589793,
        0.0,
        "-3.141592653589793115997963E+0",
        0xC00921FB54442D18,
        0x0000000000000000,
    ),
    (0.0, 0.0167, "0E0", 0x0000000000000000, 0x0000000000000000),
    (
        0.001,
        0.0167,
        "1.016983623586323038216657E-3",
        0x3F50A989D9D9050B,
        0x3BFE1F532C1EAC1B,
    ),
    (
        -0.001,
        0.0167,
        "-1.016983623586323038216657E-3",
        0xBF50A989D9D9050B,
        0xBBFE1F532C1EAC1B,
    ),
    (
        1.5707963267948966,
        0.0167,
        "1.587493998766706070412846E+0",
        0x3FF966601B74C9CC,
        0x3C786562540BA935,
    ),
    (
        -1.5707963267948966,
        0.0167,
        "-1.587493998766706070412846E+0",
        0xBFF966601B74C9CC,
        0xBC786562540BA935,
    ),
    (
        3.141592653589793,
        0.0167,
        "3.141592653589793118009530E+0",
        0x400921FB54442D18,
        0x3C428DADA2320E9B,
    ),
    (
        -3.141592653589793,
        0.0167,
        "-3.141592653589793118009530E+0",
        0xC00921FB54442D18,
        0xBC428DADA2320E9B,
    ),
    (0.0, 0.15, "0E0", 0x0000000000000000, 0x0000000000000000),
    (
        0.001,
        0.15,
        "1.176470540343156412400075E-3",
        0x3F5346799FB5F739,
        0xBBF7D88E50689917,
    ),
    (
        -0.001,
        0.15,
        "-1.176470540343156412400075E-3",
        0xBF5346799FB5F739,
        0x3BF7D88E50689917,
    ),
    (
        1.5707963267948966,
        0.15,
        "1.719148719463454884604401E+0",
        0x3FFB81A21670E4A1,
        0xBC8BDA3FF8469EF3,
    ),
    (
        -1.5707963267948966,
        0.15,
        "-1.719148719463454884604401E+0",
        0xBFFB81A21670E4A1,
        0x3C8BDA3FF8469EF3,
    ),
    (
        3.141592653589793,
        0.15,
        "3.141592653589793131971617E+0",
        0x400921FB54442D18,
        0x3C726A972A2B816B,
    ),
    (
        -3.141592653589793,
        0.15,
        "-3.141592653589793131971617E+0",
        0xC00921FB54442D18,
        0xBC726A972A2B816B,
    ),
    (0.0, 0.45, "0E0", 0x0000000000000000, 0x0000000000000000),
    (
        0.001,
        0.45,
        "1.818180998566915711064031E-3",
        0x3F5DCA00FB54D850,
        0xBBEB1D4A12BA5F77,
    ),
    (
        -0.001,
        0.45,
        "-1.818180998566915711064031E-3",
        0xBF5DCA00FB54D850,
        0x3BEB1D4A12BA5F77,
    ),
    (
        1.5707963267948966,
        0.45,
        "1.983088463151614627662594E+0",
        0x3FFFBABAF7E4FA35,
        0x3C8CC7526A0FA219,
    ),
    (
        -1.5707963267948966,
        0.45,
        "-1.983088463151614627662594E+0",
        0xBFFFBABAF7E4FA35,
        0xBC8CC7526A0FA219,
    ),
    (
        3.141592653589793,
        0.45,
        "3.141592653589793154004243E+0",
        0x400921FB54442D18,
        0x3C85E8BCA9571123,
    ),
    (
        -3.141592653589793,
        0.45,
        "-3.141592653589793154004243E+0",
        0xC00921FB54442D18,
        0xBC85E8BCA9571123,
    ),
    (0.0, 0.9, "0E0", 0x0000000000000000, 0x0000000000000000),
    (
        0.001,
        0.9,
        "9.998500682086272127244950E-3",
        0x3F847A180B823DB1,
        0x3C23E8B0979413E6,
    ),
    (
        -0.001,
        0.9,
        "-9.998500682086272127244950E-3",
        0xBF847A180B823DB1,
        0xBC23E8B0979413E6,
    ),
    (
        1.5707963267948966,
        0.9,
        "2.263415106356942807861636E+0",
        0x40021B7961189A7C,
        0xBCA98C9169240D73,
    ),
    (
        -1.5707963267948966,
        0.9,
        "-2.263415106356942807861636E+0",
        0xC0021B7961189A7C,
        0x3CA98C9169240D73,
    ),
    (
        3.141592653589793,
        0.9,
        "3.141592653589793174007549E+0",
        0x400921FB54442D18,
        0x3C90B85A15719A8D,
    ),
    (
        -3.141592653589793,
        0.9,
        "-3.141592653589793174007549E+0",
        0xC00921FB54442D18,
        0xBC90B85A15719A8D,
    ),
    (0.0, 0.95, "0E0", 0x0000000000000000, 0x0000000000000000),
    (
        0.001,
        0.95,
        "1.997476294996826614348796E-2",
        0x3F947443A6763CC5,
        0xBC35A62F595EED2C,
    ),
    (
        -0.001,
        0.95,
        "-1.997476294996826614348796E-2",
        0xBF947443A6763CC5,
        0x3C35A62F595EED2C,
    ),
    (
        1.5707963267948966,
        0.95,
        "2.287236912493829973222568E+0",
        0x40024C42DDCAEB8C,
        0x3C9BCC3128C0B95A,
    ),
    (
        -1.5707963267948966,
        0.95,
        "-2.287236912493829973222568E+0",
        0xC0024C42DDCAEB8C,
        0xBC9BCC3128C0B95A,
    ),
    (
        3.141592653589793,
        0.95,
        "3.141592653589793175660243E+0",
        0x400921FB54442D18,
        0x3C91324C9ACBA1DF,
    ),
    (
        -3.141592653589793,
        0.95,
        "-3.141592653589793175660243E+0",
        0xC00921FB54442D18,
        0xBC91324C9ACBA1DF,
    ),
];

/// Accepted/refused boundary matrix at the exact-real service boundary, all
/// six typed refusals and their deterministic precedence.
#[test]
fn accepted_refused_boundary_matrix() {
    // Refused: NonFiniteInput on either numerical input, checked first.
    assert_eq!(
        solve_kepler(Sigma::Elliptic, f64::NAN, 0.1),
        Err(KeplerError::NonFiniteInput)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.1, f64::NAN),
        Err(KeplerError::NonFiniteInput)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, f64::INFINITY, 0.1),
        Err(KeplerError::NonFiniteInput)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, f64::NEG_INFINITY, 0.1),
        Err(KeplerError::NonFiniteInput)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.1, f64::INFINITY),
        Err(KeplerError::NonFiniteInput)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.1, f64::NEG_INFINITY),
        Err(KeplerError::NonFiniteInput)
    );

    // Refused: UnsupportedFace on the non-elliptic faces, explicitly, even at
    // e = 0 (Sigma::default() is Parabolic and never a solver face).
    assert_eq!(
        solve_kepler(Sigma::Parabolic, 0.0, 0.0),
        Err(KeplerError::UnsupportedFace)
    );
    assert_eq!(
        solve_kepler(Sigma::Hyperbolic, 0.0, 0.0),
        Err(KeplerError::UnsupportedFace)
    );
    assert_eq!(
        solve_kepler(Sigma::Parabolic, 1.0, 0.5),
        Err(KeplerError::UnsupportedFace)
    );
    assert_eq!(
        solve_kepler(Sigma::Hyperbolic, 1.0, 0.5),
        Err(KeplerError::UnsupportedFace)
    );

    // Refused: InvalidEccentricity for e < 0 or e >= 1 over the exact denoted
    // reals; no clamping, no substitution.
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.0, -f64::MIN_POSITIVE),
        Err(KeplerError::InvalidEccentricity)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.0, -0.5),
        Err(KeplerError::InvalidEccentricity)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.0, -1.0),
        Err(KeplerError::InvalidEccentricity)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.0, 1.0),
        Err(KeplerError::InvalidEccentricity)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.0, 4.0),
        Err(KeplerError::InvalidEccentricity)
    );

    // Refused: UnsupportedDomain for m outside [-pi, pi] (a large finite m is
    // never silently reduced: this API carries no turn count) and for physical
    // e above the binary64 service endpoint.
    assert_eq!(
        solve_kepler(Sigma::Elliptic, PI.next_up(), 0.1),
        Err(KeplerError::UnsupportedDomain)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, -PI.next_up(), 0.1),
        Err(KeplerError::UnsupportedDomain)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 1e9, 0.1),
        Err(KeplerError::UnsupportedDomain)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, -8.0, 0.1),
        Err(KeplerError::UnsupportedDomain)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.0, SERVICE_ECC.next_up()),
        Err(KeplerError::UnsupportedDomain)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.0, 0.99),
        Err(KeplerError::UnsupportedDomain)
    );

    // Deterministic precedence: NonFiniteInput, then UnsupportedFace, then
    // InvalidEccentricity, then UnsupportedDomain.
    assert_eq!(
        solve_kepler(Sigma::Parabolic, f64::NAN, 2.0),
        Err(KeplerError::NonFiniteInput)
    );
    assert_eq!(
        solve_kepler(Sigma::Parabolic, 1e9, 2.0),
        Err(KeplerError::UnsupportedFace)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 1e9, 2.0),
        Err(KeplerError::InvalidEccentricity)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 1e9, SERVICE_ECC.next_up()),
        Err(KeplerError::UnsupportedDomain)
    );

    // Accepted boundary endpoints. PI's denotation
    // 3.141592653589793115997963468544185161590576171875 is below pi
    // 3.141592653589793238462643383279502884197169399375, while the next
    // binary64 (3.141592653589793559893113...) is above it, so the f64 test
    // |m| <= PI is exactly the exact-real comparison on the admitted set.
    assert!(solve_kepler(Sigma::Elliptic, PI, SERVICE_ECC).is_ok());
    assert!(solve_kepler(Sigma::Elliptic, -PI, SERVICE_ECC).is_ok());
    assert!(solve_kepler(Sigma::Elliptic, 0.0, SERVICE_ECC).is_ok());
    assert!(solve_kepler(Sigma::Elliptic, PI / 2.0, SERVICE_ECC).is_ok());
    assert!(solve_kepler(Sigma::Elliptic, 0.0, 0.0).is_ok());
    assert!(solve_kepler(Sigma::Elliptic, 0.0, -0.0).is_ok());
    assert!(solve_kepler(Sigma::Elliptic, -0.0, 0.0).is_ok());

    // The two execution refusals are typed and carry no value: no partial
    // result exists behind any Err variant (the return type has none).
    let _execution_refusals = (
        KeplerError::IterationExhaustion,
        KeplerError::CertificateFailure,
    );
}

/// The circular monad is bit-exact: `e == 0` returns the input anomaly bits
/// (including `-0.0`), zero rounds and an exact zero certificate.
#[test]
fn circular_monad_is_bit_exact() {
    let cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.001,
        -0.001,
        1e-300,
        -1e-300,
        f64::MIN_POSITIVE,
        PI,
        -PI,
        PI.next_down(),
        (-PI).next_up(),
    ];
    for m in cases {
        for e in [0.0, -0.0] {
            let sol = solve_kepler(Sigma::Elliptic, m, e).expect("circular monad admitted");
            assert_eq!(
                sol.anomaly.to_bits(),
                m.to_bits(),
                "bit-exact anomaly at m={m} e={e}"
            );
            assert_eq!(sol.rounds, 0, "zero rounds at m={m} e={e}");
            assert_eq!(sol.defect_bound, 0.0, "exact zero defect at m={m} e={e}");
            assert_eq!(
                sol.error_bound, 0.0,
                "exact zero error bound at m={m} e={e}"
            );
        }
    }
    let neg_zero = solve_kepler(Sigma::Elliptic, -0.0, 0.0).expect("circular monad admitted");
    assert!(neg_zero.anomaly.is_sign_negative(), "-0.0 sign preserved");
}

/// Independent oracle: every golden row must land within a few kappa-scaled
/// ulp of the reference root, and the reported error bound must cover the
/// golden distance (the certificate reads sound from outside the kernel).
#[test]
fn golden_oracle_rows_certify() {
    let mut max_scaled = 0.0f64;
    for &(m, e, decimal, hi, lo) in GOLDEN {
        let golden = f64::from_bits(hi);
        let golden_tail = f64::from_bits(lo);
        assert_eq!(
            decimal.parse::<f64>().expect("golden decimal parses"),
            golden,
            "golden decimal {decimal} must round-trip to its hi word"
        );
        let sol = solve_kepler(Sigma::Elliptic, m, e).expect("golden row admitted");
        // |x - golden| is exact (Sterbenz: the two are within a factor of 2),
        // so the double-double distance below reads |x - E*| to ~1e-32.
        let dist = ((sol.anomaly - golden) - golden_tail).abs();
        let kappa = 1.0 / (1.0 - e);
        let ulp_tol = 4.0 * kappa * EPS * sol.anomaly.abs();
        assert!(
            dist <= ulp_tol + ORACLE_SLACK,
            "anomaly mismatch at m={m} e={e}: golden distance {dist:e} exceeds 4 kappa eps |x| {ulp_tol:e}"
        );
        assert!(
            dist <= sol.error_bound + ORACLE_SLACK,
            "certificate mismatch at m={m} e={e}: golden distance {dist:e} exceeds reported error bound {:e} (d={:e}, rounds={})",
            sol.error_bound,
            sol.defect_bound,
            sol.rounds
        );
        assert!(
            sol.error_bound <= 4.0 * kappa * EPS + ORACLE_SLACK,
            "error bound above the service ceiling at m={m} e={e}: {:e}",
            sol.error_bound
        );
        max_scaled = max_scaled.max(dist / (kappa * EPS));
    }
    println!(
        "golden oracle: {} rows, max dist/(kappa eps) = {max_scaled:.3}",
        GOLDEN.len()
    );
}

/// The measured population grid: 1024 cell midpoints plus named specials.
fn grid_mean_anomalies() -> Vec<f64> {
    let mut out = Vec::with_capacity(1039);
    for i in 0..1024 {
        out.push(-PI + 2.0 * PI * ((i as f64) + 0.5) / 1024.0);
    }
    out.extend_from_slice(&[
        0.0,
        1e-300,
        -1e-300,
        1e-12,
        -1e-12,
        1e-6,
        -1e-6,
        1e-3,
        -1e-3,
        PI,
        -PI,
        PI / 2.0,
        -PI / 2.0,
        PI.next_down(),
        (-PI).next_up(),
    ]);
    out
}

/// Eccentricity bands of the sampled population run: the measured bands plus
/// the adversarial `1.0 - e` rounding bands named by review.
const GRID_E: [f64; 12] = [
    0.0167,
    0.1,
    0.15,
    0.2,
    0.3,
    0.34019998659089296,
    0.45,
    0.5,
    0.65,
    0.8,
    0.9,
    0.95,
];

/// Certificate invariant for every `Ok` on the sampled grid, per-band
/// accepted/refused population, and the count caps.
#[test]
fn certificate_invariant_on_the_sampled_grid() {
    let m_values = grid_mean_anomalies();
    let mut total_refused = 0usize;
    for &e in &GRID_E {
        let cap: u8 = if e <= 0.45 { 5 } else { 8 };
        let kappa = 1.0 / (1.0 - e);
        let mut accepted = 0usize;
        let mut refused = 0usize;
        let mut max_d_over_eps = 0.0f64;
        let mut max_r_over_ceiling = 0.0f64;
        let mut max_rounds = 0u8;
        for &m in &m_values {
            match solve_kepler(Sigma::Elliptic, m, e) {
                Ok(sol) => {
                    accepted += 1;
                    assert_eq!(sol.face, Sigma::Elliptic, "face echo at m={m} e={e}");
                    assert!(
                        sol.defect_bound.is_finite() && sol.defect_bound >= 0.0,
                        "defect at m={m} e={e}"
                    );
                    assert!(
                        sol.error_bound.is_finite() && sol.error_bound >= 0.0,
                        "error at m={m} e={e}"
                    );
                    assert!(
                        sol.error_bound >= sol.defect_bound / (1.0 - e),
                        "inward error bound at m={m} e={e}: r={:e} below d/(1-e)={:e}",
                        sol.error_bound,
                        sol.defect_bound / (1.0 - e)
                    );
                    // The real invariant, decided over the exact reals (the
                    // rounded comparison above cannot see denominator
                    // rounding: `1.0 - e` is inexact for `e < 0.5`).
                    assert!(
                        error_bound_covers_exact(sol.defect_bound, e, sol.error_bound),
                        "inward error bound at m={m} e={e}: r * (1 - e) < d over the reals \
                         (d={:e} 0x{:016x}, r={:e} 0x{:016x})",
                        sol.defect_bound,
                        sol.defect_bound.to_bits(),
                        sol.error_bound,
                        sol.error_bound.to_bits()
                    );
                    assert!(
                        sol.defect_bound <= 4.0 * EPS,
                        "defect above 4 eps at m={m} e={e}: {:e}",
                        sol.defect_bound
                    );
                    // 1e-31 covers the one-ulp asymmetry between the kernel's
                    // rounded-down ceiling and this test's rounded product.
                    assert!(
                        sol.error_bound <= 4.0 * kappa * EPS + 1e-31,
                        "error above 4 kappa eps at m={m} e={e}: {:e}",
                        sol.error_bound
                    );
                    assert!(
                        sol.rounds >= 1,
                        "a non-circular solve counts its constructions at m={m} e={e}"
                    );
                    assert!(
                        sol.rounds <= cap,
                        "rounds above the cap at m={m} e={e}: {}",
                        sol.rounds
                    );
                    max_d_over_eps = max_d_over_eps.max(sol.defect_bound / EPS);
                    max_r_over_ceiling =
                        max_r_over_ceiling.max(sol.error_bound / (4.0 * kappa * EPS));
                    max_rounds = max_rounds.max(sol.rounds);
                }
                Err(KeplerError::CertificateFailure) => {
                    refused += 1;
                }
                Err(other) => {
                    panic!("unexpected refusal {other:?} on admitted input m={m} e={e}");
                }
            }
        }
        total_refused += refused;
        println!(
            "band e={e}: rows {}, accepted {accepted}, refused {refused}, max d/eps {max_d_over_eps:.3}, max r/(4 kappa eps) {max_r_over_ceiling:.3}, max rounds {max_rounds}",
            m_values.len()
        );
    }
    // CertificateFailure is a minority escape at the count cap (measured: 2 of
    // 9351 rows, e = 0.95), never a routine escape for the promised population.
    assert!(
        total_refused <= 32,
        "refusal population {total_refused}/9351 is not a minority escape"
    );
}

/// The shipped placeholder schedule (Penglai `kepler_evaluate` shape):
/// `E <- M + e sin E` from `E0 = M`, stopping at update <= 1e-12, capped at
/// 256. Measured as a contrast arm only; it is not the operating kernel.
fn placeholder_fixed_point(m: f64, e: f64) -> u32 {
    let mut x = m;
    let mut used = 0u32;
    for _ in 0..256 {
        let next = m + e * x.sin();
        let update = (next - x).abs();
        x = next;
        used += 1;
        if update <= 1e-12 {
            break;
        }
    }
    used
}

/// Naive-schedule falsifier: the placeholder fixed-point schedule is not the
/// operating kernel (it needs far more than 40 rounds at e = 0.9), and the
/// chosen schedule stays inside the 5/8 count caps on the sampled grid.
#[test]
fn naive_placeholder_fixed_point_is_not_the_operating_kernel() {
    let mut max_placeholder = 0u32;
    let mut max_kernel_high = 0u8;
    let mut max_kernel_low = 0u8;
    for i in 0..64 {
        let m = -PI + 2.0 * PI * ((i as f64) + 0.5) / 64.0;
        max_placeholder = max_placeholder.max(placeholder_fixed_point(m, 0.9));
        let high = solve_kepler(Sigma::Elliptic, m, 0.9).expect("admitted grid row");
        max_kernel_high = max_kernel_high.max(high.rounds);
        let low = solve_kepler(Sigma::Elliptic, m, 0.45).expect("admitted grid row");
        max_kernel_low = max_kernel_low.max(low.rounds);
    }
    assert!(
        max_placeholder > 40,
        "placeholder schedule needed only {max_placeholder} rounds at e=0.9"
    );
    assert!(
        max_kernel_high <= 8,
        "operating kernel used {max_kernel_high} rounds at e=0.9"
    );
    assert!(
        max_kernel_low <= 5,
        "operating kernel used {max_kernel_low} rounds at e=0.45"
    );
}

/// Naive accumulation falsifier: naive `M += dt` phase accumulation drifts
/// against the compensated (Kahan) form; a horizon-stable upstream reducer
/// must not use it (`SLOT-SPIKE-PHASE` context).
#[test]
fn naive_phase_accumulation_drifts_against_compensated() {
    let mut naive = 0.0f64;
    for _ in 0..10_000 {
        naive += 0.1;
    }
    let mut kahan = 0.0f64;
    let mut carry = 0.0f64;
    for _ in 0..10_000 {
        let y = 0.1 - carry;
        let t = kahan + y;
        carry = (t - kahan) - y;
        kahan = t;
    }
    assert!(kahan == 1000.0, "compensated accumulation moved: {kahan:?}");
    assert!(
        (naive - 1000.0).abs() > (kahan - 1000.0).abs(),
        "naive accumulation {naive:?} did not drift above compensated {kahan:?}"
    );
}
