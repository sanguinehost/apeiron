//! Canonical elliptic Kepler solve: the executing kernel of .
//!
//! `solve_kepler(face, mean_anomaly, eccentricity)` returns the normalized
//! eccentric anomaly `x` of the exact-real equation
//! `F_{m,e}(x) = x - e * sin(x) - m = 0` over `R`, together with an outward
//! certified absolute defect upper bound `d` on `|F_{m,e}(x)|`, an outward
//! absolute-error enclosure `r` with `r >= d / (1 - e)` and the complete
//! candidate construction count. This module is the authority for the
//! solution, the defect bound, the error bound and the round count; host
//! readouts and caches are not authorities.
//!
//! The additive per-orbit entry point of is
//! [`solve_with_plan`] over a [`KeplerPlan`]: one plan per eccentricity folds
//! the schedule band, the e-powers, the exact denominator split of `1 - e` and
//! the e-dependent constant allowances once, and every solve of the orbit
//! shares them. [`solve_kepler`] constructs one plan per call and delegates;
//! the two entries return bit-identical solutions (tested on the sampled
//! population).
//!
//! # Admitted domain (contract: the kepler-canonical-spec note)
//!
//! Only `face == Sigma::Elliptic` executes. `Sigma::default()` is Parabolic and
//! is never a solver face. The mean anomaly is finite with `-pi <= m <= pi` over
//! the exact reals (endpoints admitted). This API carries no turn count and
//! never reduces a large finite `m`: the caller's phase normalization is a
//! separate, upstream obligation. The eccentricity is finite with
//! `0 <= e <= 0.95` over the exact reals, where the service endpoint is the
//! denotation of the exactly represented `0.95_f64`
//! (0.94999999999999995559...), not the unrepresentable decimal 0.95. A `-0.0`
//! mean anomaly is admitted with its sign preserved (circular bit identity),
//! and `e == 0.0` (either zero sign) takes the circular monad branch below.
//! Everything else is a typed refusal (`KeplerError`), never a clamp and never
//! a default substitute. Refusals have the deterministic precedence
//! `NonFiniteInput`, `UnsupportedFace`, `InvalidEccentricity`,
//! `UnsupportedDomain`, then execution/certificate; [`solve_kepler`] enforces
//! that precedence across its three arguments, while [`KeplerPlan::new`] and
//! [`solve_with_plan`] check the arguments each receives.
//!
//! At `e == 0.0` the returned anomaly is the input mean anomaly with its exact
//! bit pattern, `rounds: 0`, `defect_bound: 0.0`, `error_bound: 0.0`, and no
//! trigonometric evaluation happens after the finiteness/face/domain checks.
//!
//! # Certificate: constructive side of `proof/corpus/Kepler/Certificate.lean`
//!
//! The bridge certificate interprets `Sound` as a *supplied* sound outward
//! bound on the exact-real residual `|x - e * sin(x) - m|` of the returned
//! pattern `x`; the rounded evaluation `|fl(x - e * fl(sin x) - m)|` is not
//! that premise. This kernel constructs the bound. With `s = fl_sin(x)`, the
//! exact-error-free decompositions
//!
//! ```text
//! two_sum(x, -m)        = (sm,  tail_m)    x - m        = sm + tail_m
//! two_prod(e, s)        = (prod, tail_sin)  e * s        = prod + tail_sin
//! two_sum(sm, -prod)    = (lead, tail_lead) sm - prod    = lead + tail_lead
//! ```
//!
//! give the exact residual of the rounded sine
//! `chi = x - e * s - m = lead + (tail_m - tail_sin + tail_lead)`. With
//! `combine = fl(tail_m - tail_sin)`, `tail = fl(combine + tail_lead)` and
//! `residual_word = fl(lead + tail)` the three round-to-nearest steps obey
//!
//! ```text
//! |chi - residual_word| <= 0.5 ulp(combine) + 0.5 ulp(tail)
//!                          + 0.5 ulp(residual_word),
//! ```
//!
//! covered by `residual_allowance`. The transcendental gap is covered by
//! `sine_allowance = e * (2^-52 * |s| + 2^-1074)`. Every allowance is built
//! from `outward_product`/`outward_sum`, each of which adds one `next_up` past
//! its single rounding, and the reported bounds escalate once more:
//!
//! ```text
//! margin       = sine_allowance + residual_allowance        (outward)
//! defect_bound = next_up(|residual_word| + margin)          (outward)
//! error_bound  = next_up(defect_bound / (1 - e))            (outward)
//! ```
//!
//! so `|F_{m,e}(x)| <= defect_bound` and `|x - E*| <= defect_bound / (1 - e)
//! <= error_bound` over the exact reals, which is exactly the denoted `Sound`
//! premise the conditional bridge theorem consumes. The quotient is NOT taken
//! over the rounded `1.0 - e` denominator (that subtraction is inexact for
//! `e < 0.5`): the exact `two_sum(1.0, -e) = (denom, denom_tail)` split of
//! `1 - e` is folded **once** per [`KeplerPlan`] (once per solve, and once per
//! orbit under [`solve_with_plan`]) and shared by
//! [`KeplerPlan::outward_error_bound`], which escalates past the split's
//! `|denom_tail| <= 2^-53 denom` slack and every rounding on the way, and by
//! [`KeplerPlan::service_error_ceiling`], which mirrors the argument downward
//! so acceptance implies `r <= 4 kappa eps` exactly. Both carry their bound
//! argument, no allowance is smaller than before the folding (the folded
//! constants are the same values computed in the same order), and the
//! `#[cfg(test)]` regression checks the conclusions by exact
//! cross-multiplication (no rounded division) on adversarial pairs.
//!
//! # Platform assumptions (external trust, spec line 101)
//!
//! 1. IEEE-754 binary64, round-to-nearest, and `f64::mul_add` as one correctly
//!    rounded `a * b + c` (so `two_prod` is error-free).
//! 2. `ASSUMPTION-SIN`: for `|x| <= SIN_ASSUMPTION_DOMAIN` the platform
//!    `f64::sin` (Rust std -> glibc `sin` here) is within one ulp of `sin(x)`:
//!    `|fl_sin(x) - sin(x)| <= 2^-52 * |fl_sin(x)| + 2^-1074`. The certificate
//!    refuses candidates outside that domain rather than extrapolating the
//!    assumption. Compiler, ABI, hardware and libm are external trust; this
//!    source provides no f64 refinement of them.
//! 3. The sine reuse additionally assumes `f64::sin` is a pure function
//!    of its argument pattern within one platform (same pattern, same word);
//!    the reuse guard demands an identical candidate bit pattern before
//!    consuming the cached word, and `tests/kepler.rs` re-derives every
//!    reported bound with an afresh sine and demands bit equality on the
//!    sampled grid.
//!
//! # Refusal policy above the service ceiling
//!
//! Successful service delivers `d <= 4 eps` and `r <= 4 kappa eps` with
//! `eps = 2^-52` and `kappa = 1 / (1 - e)` from the exact denoted `e`. The
//! kernel first runs the scheduled series arm, then one bounded
//! escalation correction on a stalled series iterate, then the safeguarded
//! cubic fallback arm within the remaining global count cap (5 candidate
//! constructions for `e <= 0.45_f64`, 8 above). Every stage's candidate is
//! accepted only through the same certificate and both service ceilings. If
//! no stage meets both ceilings at the count cap, the input is refused
//! `CertificateFailure`: a sound-but-oversized bound is never returned
//! silently (it would exceed the contract's success criterion), never
//! clamped, and the candidate is never returned uncertified. The SLA row for
//! the refused population stays OPEN.
//!
//! The escalation is one complete candidate construction inside the
//! unchanged cap and schedules (`rounds + 1 <= cap` on every admitted band);
//! it replaces no schedule row and weakens no bound. Its measured effect on
//! the sampled population: the two strict-policy refusals at e = 0.95 (a
//! mid-cascade stall ~4e-15 from the root, tens of ulp) now certify at 5
//! constructions, and the four rows previously saved by the cubic arm switch
//! to the continued iterate at 5 instead of 8 constructions.
//!
//! No Aeneas f64 refinement is claimed for this kernel: none exists for this
//! path, and the bridge theorem's supplied-premise hypothesis is not a Rust
//! correspondence result.

use crate::cell::Sigma;
use std::f64::consts::PI;

/// The service endpoint for the eccentricity: the exactly represented
/// `0.95_f64` denotation, not the decimal real 0.95.
pub const SERVICE_ECCENTRICITY: f64 = 0.95_f64;

/// Split of the correction-count caps: `<= 5` constructions at
/// `e <= 0.45_f64`, `<= 8` above.
const ROUND_SPLIT_ECC: f64 = 0.45_f64;

/// Count cap (complete candidate constructions, starters included) for
/// `e <= ROUND_SPLIT_ECC`.
const ROUND_CAP_LOW_ECC: u8 = 5;

/// Count cap for `ROUND_SPLIT_ECC < e <= SERVICE_ECCENTRICITY`.
const ROUND_CAP_HIGH_ECC: u8 = 8;

/// Fixed correction schedule of the series arm: `(e_max, Halley corrections)`
/// rows applying while `e <= e_max`. Combined with the starter construction
/// this gives 2/3/4/5 constructions per band.
const SCHEDULE_SERIES: [(f64, u8); 4] = [(0.15, 1), (0.65, 2), (0.95, 3), (f64::INFINITY, 4)];

/// Fixed correction schedule of the safeguarded cubic fallback arm (same
/// row semantics as [`SCHEDULE_SERIES`]).
const SCHEDULE_CUBIC: [(f64, u8); 5] = [
    (0.25, 1),
    (0.75, 2),
    (0.90, 3),
    (0.95, 4),
    (f64::INFINITY, 6),
];

/// Service ceiling `4 eps` with `eps = 2^-52` (exact: `2^-50`).
const SERVICE_DEFECT_LIMIT: f64 = 4.0 * f64::EPSILON;

/// One ulp relative allowance scale, `2^-52` (the `ASSUMPTION-SIN` factor).
const SINE_ULP_FACTOR: f64 = f64::EPSILON;

/// Half-ulp relative allowance scale, `2^-53`.
const HALF_ULP_FACTOR: f64 = 0.5 * f64::EPSILON;

/// Absolute allowance floor for subnormal gaps, `2^-1074` (one ulp of the
/// smallest subnormal).
const SUBNORMAL_FLOOR: f64 = f64::from_bits(1);

/// Absolute guard of [`KeplerPlan::outward_error_bound`], `8 * 2^-1074`: it
/// dominates the subnormal-regime quotient deficit (bounded by
/// `6.5 * 2^-1074` there; the method comment carries the accounting).
const QUOTIENT_SUBNORMAL_GUARD: f64 = f64::from_bits(8);

/// Domain of `ASSUMPTION-SIN`; a candidate beyond it is refused instead of
/// certified. Every candidate the kernel forms on admitted input stays inside
/// `|x| <= pi + 4 * 0.95 < SIN_ASSUMPTION_DOMAIN`.
const SIN_ASSUMPTION_DOMAIN: f64 = 8.0;

/// The cached sine witness: `(iterate, f64::sin(iterate))` as evaluated
/// by a refinement correction, consumed only at an identical candidate bit
/// pattern (see [`KeplerPlan::certificate`]).
type SinCache = Option<(f64, f64)>;

/// Typed refusals, in the contract's deterministic precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeplerError {
    /// NaN or infinity in either numerical input; checked before any
    /// comparison or iteration.
    NonFiniteInput,
    /// Finite `e < 0` or `e >= 1` over the exact denoted reals; no clamping or
    /// substitution.
    InvalidEccentricity,
    /// `Sigma::Parabolic` or `Sigma::Hyperbolic`; checked explicitly, even at
    /// `e = 0`.
    UnsupportedFace,
    /// Finite `m` outside `[-pi, pi]`, or physical `0 <= e < 1` above the
    /// service endpoint `SERVICE_ECCENTRICITY`; no implicit time reduction.
    UnsupportedDomain,
    /// The solver could not produce a finite candidate in the allocated count;
    /// unreachable on the admitted domain (bounded arithmetic), never a
    /// routine escape.
    IterationExhaustion,
    /// The outward real-residual and error bounds could not be formed or miss
    /// the service ceilings at the count cap; no uncertified candidate is
    /// returned.
    CertificateFailure,
}

/// The certified solution of the normalized elliptic Kepler equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerSolution {
    /// The face the host passed (always `Sigma::Elliptic` on `Ok`).
    pub face: Sigma,
    /// The normalized eccentric anomaly `x` (an `f64` pattern whose exact
    /// denotation is what the bounds talk about).
    pub anomaly: f64,
    /// Outward certified absolute defect upper bound `d` with
    /// `|x - e * sin(x) - m| <= d` over the exact reals.
    pub defect_bound: f64,
    /// Outward absolute-error enclosure `r` with `|x - E*| <= r` and
    /// `r >= d / (1 - e)`.
    pub error_bound: f64,
    /// Complete candidate constructions (each starter and each correction),
    /// bounded by 5 for `e <= 0.45_f64` and 8 above.
    pub rounds: u8,
}

/// The per-orbit solve plan of : everything the schedule
/// and the certificate need that depends only on the eccentricity, folded once
/// by [`KeplerPlan::new`] and shared by every solve over the orbit.
///
/// Folded contents (all private; the constructor is the only authority):
///
/// - the schedule band: series and cubic correction counts plus the global
///   construction cap (from [`SCHEDULE_SERIES`], [`SCHEDULE_CUBIC`],
///   [`ROUND_SPLIT_ECC`]);
/// - the e-powers `e^2, e^3, e^4` of the series starter;
/// - the exact denominator split `1 - e = denom + denom_tail` (Knuth two-sum,
///   read out by [`KeplerPlan::denominator_split`]);
/// - the e-dependent constant allowance: the conservative service error
///   ceiling used by [`KeplerPlan::service_error_ceiling`].
///
/// The fields cannot be forged (private, validated construction only), so a
/// plan cannot manufacture a defect or error bound: every reported bound is
/// still constructed by [`KeplerPlan::certificate`] from the exact-real
/// residual decomposition and the allowances above.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerPlan {
    /// The admitted eccentricity (finite, `0 <= e <= SERVICE_ECCENTRICITY`);
    /// the exact denoted value the bounds talk about.
    eccentricity: f64,
    /// `e^2` for the series starter (folded once).
    e2: f64,
    /// `e^3` for the series starter (folded once).
    e3: f64,
    /// `e^4` for the series starter (folded once).
    e4: f64,
    /// Lead of the exact split `1 - e = denom + denom_tail`.
    denom: f64,
    /// Tail of the exact split; `|denom_tail| <= 2^-53 denom` is the slack the
    /// [`KeplerPlan::outward_error_bound`] domination argument escalates past.
    denom_tail: f64,
    /// Conservative service error ceiling `<= 4 eps / (1 - e)` over the exact
    /// reals, folded once (see [`KeplerPlan::service_error_ceiling`]).
    error_ceiling: f64,
    /// Series-arm Halley corrections of the schedule band (starter included,
    /// the complete construction count is `series_corrections + 1`).
    series_corrections: u8,
    /// Cubic-arm corrections of the schedule band.
    cubic_corrections: u8,
    /// Global construction cap of one solve for this band.
    round_cap: u8,
}

impl KeplerPlan {
    /// Fold one plan from an eccentricity; the validated constructor.
    ///
    /// The eccentricity checks are `NonFiniteInput` (NaN or infinity),
    /// `InvalidEccentricity` (finite `e < 0` or `e >= 1` over the exact denoted
    /// reals) and `UnsupportedDomain` (above `SERVICE_ECCENTRICITY`). Together
    /// with [`solve_with_plan`]'s checks on the face and mean anomaly this is
    /// the frozen precedence of [`solve_kepler`]; a plan outlives many solves,
    /// so face and mean anomaly are re-checked per solve, not per plan.
    ///
    /// # Errors
    ///
    /// Returns `KeplerError::NonFiniteInput` for a non-finite eccentricity,
    /// `KeplerError::InvalidEccentricity` for `e < 0` or `e >= 1`, and
    /// `KeplerError::UnsupportedDomain` above the service endpoint. No
    /// substitution or clamping happens on any refusal.
    pub fn new(eccentricity: f64) -> Result<KeplerPlan, KeplerError> {
        if !eccentricity.is_finite() {
            return Err(KeplerError::NonFiniteInput);
        }
        if !(0.0..1.0).contains(&eccentricity) {
            return Err(KeplerError::InvalidEccentricity);
        }
        if eccentricity > SERVICE_ECCENTRICITY {
            return Err(KeplerError::UnsupportedDomain);
        }
        Ok(KeplerPlan::fold(eccentricity))
    }

    /// Fold every eccentricity-dependent constant once. The caller's contract
    /// is the admitted eccentricity (finite, `0 <= e <= SERVICE_ECCENTRICITY`);
    /// [`KeplerPlan::new`] validates it.
    fn fold(eccentricity: f64) -> KeplerPlan {
        let e2 = eccentricity * eccentricity;
        let e3 = e2 * eccentricity;
        let e4 = e3 * eccentricity;
        // Exact split: `denom + denom_tail = 1 - e` (Knuth two-sum), computed
        // once here and shared by both folded bounds below.
        let (denom, denom_tail) = two_sum(1.0, -eccentricity);
        // Conservative ceiling, folded once per plan (its bound argument is on
        // `service_error_ceiling`).
        let error_ceiling = {
            let scaled = (4.0 * f64::EPSILON) / denom;
            (scaled * (1.0 - 4.0 * f64::EPSILON)).next_down()
        };
        KeplerPlan {
            eccentricity,
            e2,
            e3,
            e4,
            denom,
            denom_tail,
            error_ceiling,
            series_corrections: scheduled_corrections(eccentricity, &SCHEDULE_SERIES),
            cubic_corrections: scheduled_corrections(eccentricity, &SCHEDULE_CUBIC),
            round_cap: round_cap(eccentricity),
        }
    }

    /// The exact denoted eccentricity this plan was folded from.
    #[must_use]
    pub fn eccentricity(&self) -> f64 {
        self.eccentricity
    }

    /// The exact denominator split `(denom, denom_tail)` with `denom +
    /// denom_tail = 1 - e` over the exact reals and `|denom_tail| <= 2^-53
    /// denom`. A pure readout: it manufactures no bound.
    #[must_use]
    pub fn denominator_split(&self) -> (f64, f64) {
        (self.denom, self.denom_tail)
    }

    /// The circular monad flag: `e == 0.0` (either zero sign).
    fn is_circular(&self) -> bool {
        self.eccentricity == 0.0
    }

    /// Fitzpatrick (1970) fixed-point series starter through `e^4`
    /// (`M + e S + e^2 C S + e^3 S (3 C^2 - 1) / 2 + e^4 C S (8 C^2 - 5) / 3`,
    /// `S = sin M`, `C = cos M`) from the plan's folded e-powers: one complete
    /// first candidate.
    fn series_starter(&self, mean_anomaly: f64) -> f64 {
        let s = mean_anomaly.sin();
        let c = mean_anomaly.cos();
        let c2 = c * c;
        mean_anomaly
            + self.eccentricity * s
            + self.e2 * c * s
            + self.e3 * s * (3.0 * c2 - 1.0) * 0.5
            + self.e4 * c * s * (8.0 * c2 - 5.0) / 3.0
    }

    /// Outward absolute-error enclosure: `error_bound >= defect_bound / (1 - e)`
    /// over the exact reals, for `d = defect_bound >= 0` and admitted
    /// `0 < e < 1`.
    ///
    /// Bound argument (all comparisons over `R`): `1.0 - e` is NOT an exact
    /// subtraction for `e < 0.5`, so [`KeplerPlan::fold`] splits the
    /// denominator exactly with `two_sum(1.0, -e) = (denom, denom_tail)`,
    /// giving `denom + denom_tail = 1 - e` with `|denom_tail| <= 2^-53 denom`
    /// (half an ulp of the rounded lead; `denom >= 2^-6 > 0` on the admitted
    /// domain). Put `Q = d / denom`, so `d / (1 - e) = Q / (1 + x)` with
    /// `x = denom_tail / denom`, `|x| <= 2^-53`:
    ///
    /// 1. `denom_tail <= 0` needs no denominator inflation (`d / (1 - e) <= Q`);
    ///    `denom_tail > 0` needs at most `1 / (1 - 2^-53) <= 1 + 2^-52`.
    /// 2. Normal regime `Q >= 2^-1021`: `quotient`, `scaled = quotient * (1 +
    ///    4eps)` and `w = scaled + guard` each round at relative `<= 2^-53`, so
    ///    `w >= Q (1 - 2^-53)^3 (1 + 2^-50) >= Q (1 + 5 * 2^-53)` while
    ///    `d / (1 - e) <= Q (1 + 2^-52)`; the exact factor `1 + 4eps = 1 + 2^-50`
    ///    covers the `2^-52 + 3 * 2^-53 = 5 * 2^-53` deficit with 1.6x headroom.
    /// 3. Subnormal regime `Q < 2^-1021` (every intermediate `< 2^-1020`): each
    ///    step rounds at absolute `<= 2^-53 * 2^-1020 + 2^-1075 < 1.5 * 2^-1074`,
    ///    the denominator case adds `< 2 * 2^-1074`, so the total deficit is
    ///    `< 6.5 * 2^-1074`; `QUOTIENT_SUBNORMAL_GUARD = 8 * 2^-1074` plus the two
    ///    `next_up` steps (`>= 2^-1074` each here) dominate it.
    /// 4. The two `next_up` steps are pure headroom in the normal regime, the
    ///    guard is negligible there.
    ///
    /// The `#[cfg(test)]` regression re-checks the conclusion `r (1 - e) >= d` by
    /// exact cross-multiplication (no rounded division) on adversarial pairs,
    /// including the known counterexample against the single-`next_up` quotient.
    fn outward_error_bound(&self, defect_bound: f64) -> f64 {
        // The exact split `denom + denom_tail = 1 - e` is folded once in
        // [`KeplerPlan::fold`]; only the quotient chain runs per candidate.
        let quotient = defect_bound / self.denom;
        let scaled = quotient * (1.0 + 4.0 * f64::EPSILON);
        (scaled + QUOTIENT_SUBNORMAL_GUARD).next_up().next_up()
    }

    /// Conservative service ceiling `C <= 4 eps / (1 - e)` over the exact reals,
    /// so `error_bound <= C` implies `r <= 4 kappa eps` exactly.
    ///
    /// Bound argument (same exact split `1 - e = denom + denom_tail`,
    /// `|denom_tail| <= 2^-53 denom`): `4eps / (1 - e) = (4eps / denom) / (1 + x)`
    /// with `|x| <= 2^-53`, hence `4eps / (1 - e) >= (4eps / denom) * (1 - 2^-53)`.
    /// The computed `fl(4eps / denom)` is at most `(4eps / denom) * (1 + 2^-53)`,
    /// so `4eps / (1 - e) >= q_c (1 - 2^-53)^2 >= q_c (1 - 2^-52)`. Down-scaling
    /// by the exact `1 - 4eps` and taking `next_down` gives at worst
    /// `q_c (1 - 2^-50)(1 + 2^-53) <= q_c (1 - 7 * 2^-53)`, strictly below that
    /// bound. All values here are normal (`2^-50 <= 4eps / (1 - e) <= 2^-45` for
    /// `e <= SERVICE_ECCENTRICITY`), so no subnormal regime arises. The whole
    /// chain is folded once per plan in [`KeplerPlan::fold`].
    fn service_error_ceiling(&self) -> f64 {
        self.error_ceiling
    }

    /// Construct the outward certificate `(defect_bound, error_bound)` for the
    /// returned candidate, or refuse `CertificateFailure`.
    ///
    /// The construction is the module-level formula: exact-error-free residual
    /// decomposition, `residual_allowance` over the three combination roundings,
    /// `sine_allowance = e * ulp_allowance(s)` under `ASSUMPTION-SIN`, one
    /// outward escalation into `defect_bound`, and
    /// [`KeplerPlan::outward_error_bound`] for the absolute-error enclosure.
    ///
    /// `cached` is the refinement sin reuse: the last refinement
    /// correction evaluated `f64::sin` at its input iterate and returned that
    /// word beside the candidate. The cached word is consumed only when the
    /// input iterate IS the candidate (identical bit pattern, identical
    /// platform function, therefore identical value); when the correction
    /// moved the iterate the cached sine belongs to a different point and the
    /// sine is evaluated afresh here, never reused. Either way `s` is exactly
    /// `candidate.sin()`, so the reported bounds are bit-identical to the
    /// fresh-evaluation construction (checked on the sampled grid by
    /// `tests/kepler.rs`).
    fn certificate(
        &self,
        mean_anomaly: f64,
        candidate: f64,
        cached: SinCache,
    ) -> Result<(f64, f64), KeplerError> {
        if candidate.abs() > SIN_ASSUMPTION_DOMAIN {
            return Err(KeplerError::CertificateFailure);
        }
        let s = match cached {
            Some((cached_at, cached_sin)) if cached_at.to_bits() == candidate.to_bits() => {
                cached_sin
            }
            _ => candidate.sin(),
        };
        if !s.is_finite() {
            return Err(KeplerError::CertificateFailure);
        }
        let (sm, tail_m) = two_sum(candidate, -mean_anomaly);
        let (prod, tail_sin) = two_prod(self.eccentricity, s);
        let (lead, tail_lead) = two_sum(sm, -prod);
        // Exact: candidate - eccentricity * s - mean_anomaly
        //      = lead + (tail_m - tail_sin + tail_lead).
        let combine = tail_m - tail_sin;
        let tail = combine + tail_lead;
        let residual_word = lead + tail;
        let residual_allowance = outward_sum(
            outward_sum(half_ulp_allowance(combine), half_ulp_allowance(tail)),
            half_ulp_allowance(residual_word),
        );
        let sine_allowance = outward_product(self.eccentricity, ulp_allowance(s));
        let margin = outward_sum(sine_allowance, residual_allowance);
        let defect_bound = (residual_word.abs() + margin).next_up();
        let error_bound = self.outward_error_bound(defect_bound);
        if !defect_bound.is_finite() || !error_bound.is_finite() {
            return Err(KeplerError::CertificateFailure);
        }
        Ok((defect_bound, error_bound))
    }

    /// Certificate plus the service ceilings `d <= 4 eps` and `r <= 4 kappa eps`.
    ///
    /// `kappa = 1 / (1 - e)` comes from the exact denoted eccentricity. The
    /// error ceiling is [`KeplerPlan::service_error_ceiling`], conservative
    /// over the exact reals and folded once per plan, so acceptance implies
    /// both exact-real inequalities. A ceiling miss yields `None` (escalate or
    /// refuse), never a silently oversized success.
    fn certified_service(
        &self,
        mean_anomaly: f64,
        candidate: f64,
        cached: SinCache,
    ) -> Option<(f64, f64)> {
        let (defect_bound, error_bound) = self.certificate(mean_anomaly, candidate, cached).ok()?;
        if defect_bound <= SERVICE_DEFECT_LIMIT && error_bound <= self.service_error_ceiling() {
            Some((defect_bound, error_bound))
        } else {
            None
        }
    }
}

/// Solve the normalized Kepler equation on the elliptic face.
///
/// See the module documentation for the admitted domain, the certificate
/// construction, the platform assumptions and the refusal policy. This entry
/// folds one [`KeplerPlan`] per call and delegates to [`solve_with_plan`];
/// per-orbit callers construct the plan once and call [`solve_with_plan`]
/// directly (bit-identical results, tested).
///
/// # Errors
///
/// Returns the typed refusals in the frozen precedence: `NonFiniteInput` for
/// NaN or infinity in either numerical input, `UnsupportedFace` for a face
/// other than `Sigma::Elliptic`, `InvalidEccentricity` for finite `e < 0` or
/// `e >= 1`, `UnsupportedDomain` for `m` outside `[-pi, pi]` or `e` above the
/// service endpoint, then the execution refusals `IterationExhaustion` and
/// `CertificateFailure`. No partial result is ever returned on `Err`.
pub fn solve_kepler(
    face: Sigma,
    mean_anomaly: f64,
    eccentricity: f64,
) -> Result<KeplerSolution, KeplerError> {
    if !mean_anomaly.is_finite() || !eccentricity.is_finite() {
        return Err(KeplerError::NonFiniteInput);
    }
    if face != Sigma::Elliptic {
        return Err(KeplerError::UnsupportedFace);
    }
    let plan = KeplerPlan::new(eccentricity)?;
    solve_with_plan(face, mean_anomaly, &plan)
}

/// Solve the normalized Kepler equation against a shared [`KeplerPlan`]: the
/// per-orbit entry point of .
///
/// The plan already carries the validated eccentricity and every folded
/// constant; this function re-checks what it receives (`NonFiniteInput` for a
/// non-finite mean anomaly, `UnsupportedFace`, `UnsupportedDomain` for `m`
/// outside `[-pi, pi]`) and then runs the same monad, arms, certificate and
/// refusal semantics as [`solve_kepler`]. For every `(face, m, e)` the result
/// is bit-identical to `solve_kepler(face, m, e)` (tested on the sampled
/// population).
///
/// # Errors
///
/// Returns `KeplerError::NonFiniteInput`, `KeplerError::UnsupportedFace` or
/// `KeplerError::UnsupportedDomain` for the checks above and, during
/// execution, `KeplerError::IterationExhaustion` or
/// `KeplerError::CertificateFailure` exactly where [`solve_kepler`] does.
pub fn solve_with_plan(
    face: Sigma,
    mean_anomaly: f64,
    plan: &KeplerPlan,
) -> Result<KeplerSolution, KeplerError> {
    if !mean_anomaly.is_finite() {
        return Err(KeplerError::NonFiniteInput);
    }
    if face != Sigma::Elliptic {
        return Err(KeplerError::UnsupportedFace);
    }
    if mean_anomaly.abs() > PI {
        return Err(KeplerError::UnsupportedDomain);
    }
    if plan.is_circular() {
        // Circular monad: F(x) = x - m has the denoted input as its exact
        // root, so the bit pattern (including -0.0) is returned untouched and
        // the certificate is the exact zero.
        return Ok(KeplerSolution {
            face,
            anomaly: mean_anomaly,
            defect_bound: 0.0,
            error_bound: 0.0,
            rounds: 0,
        });
    }

    let (candidate, mut rounds, cache) = refine(
        plan.series_starter(mean_anomaly),
        mean_anomaly,
        plan.eccentricity,
        plan.series_corrections,
    )?;
    if let Some((defect_bound, error_bound)) =
        plan.certified_service(mean_anomaly, candidate, cache)
    {
        return Ok(KeplerSolution {
            face,
            anomaly: candidate,
            defect_bound,
            error_bound,
            rounds,
        });
    }

    // bounded escalation: the scheduled series arm can stall
    // mid-cascade (measured at e = 0.95: the last scheduled correction still
    // moved ~3e-5 and left |x - E*| ~ 4e-15, so the certificate missed the
    // service ceilings while one more cubic-convergent correction drives the
    // error below 2^-53). Spend ONE further complete candidate construction
    // on that iterate; `rounds + 1 <= plan.round_cap` on every admitted band,
    // so the count caps and both schedules are unchanged. The candidate is
    // accepted only through the same certificate and both service ceilings;
    // on failure the safeguarded cubic fallback runs with the remaining
    // budget, and if that also misses, the strict refusal stands.
    if plan.round_cap - rounds >= 1 {
        let (escalated, sin_at_candidate) =
            halley_correction(candidate, mean_anomaly, plan.eccentricity);
        rounds += 1;
        if escalated.is_finite() {
            if let Some((defect_bound, error_bound)) =
                plan.certified_service(mean_anomaly, escalated, Some((candidate, sin_at_candidate)))
            {
                return Ok(KeplerSolution {
                    face,
                    anomaly: escalated,
                    defect_bound,
                    error_bound,
                    rounds,
                });
            }
        }
    }

    // Safeguarded fallback: a second starter on the same map, spending the
    // remaining global construction budget only.
    let remaining = plan.round_cap - rounds;
    if remaining >= 1 {
        let fallback_corrections = plan.cubic_corrections.min(remaining - 1);
        if let Ok((fallback, used, fallback_cache)) = refine(
            cubic_starter(mean_anomaly, plan.eccentricity),
            mean_anomaly,
            plan.eccentricity,
            fallback_corrections,
        ) {
            if let Some((defect_bound, error_bound)) =
                plan.certified_service(mean_anomaly, fallback, fallback_cache)
            {
                return Ok(KeplerSolution {
                    face,
                    anomaly: fallback,
                    defect_bound,
                    error_bound,
                    rounds: rounds + used,
                });
            }
        }
    }
    Err(KeplerError::CertificateFailure)
}

/// The global construction cap of one solve: 5 at `e <= 0.45_f64`, 8 above.
fn round_cap(eccentricity: f64) -> u8 {
    if eccentricity <= ROUND_SPLIT_ECC {
        ROUND_CAP_LOW_ECC
    } else {
        ROUND_CAP_HIGH_ECC
    }
}

/// Scheduled correction count: the first `(e_max, corrections)` row with
/// `e <= e_max` (the schedule is total through its infinite last row).
fn scheduled_corrections(eccentricity: f64, schedule: &[(f64, u8)]) -> u8 {
    for &(limit, corrections) in schedule {
        if eccentricity <= limit {
            return corrections;
        }
    }
    schedule[schedule.len() - 1].1
}

/// Safeguarded fallback starter: the truncated Kepler cubic in `x = E - M`
/// (`(e C / 6) x^3 + (e S / 2) x^2 + (1 - e C) x - e S = 0`) with one
/// polynomial Newton step (no trigonometry in the step).
fn cubic_starter(mean_anomaly: f64, eccentricity: f64) -> f64 {
    let sin_m = mean_anomaly.sin();
    let cos_m = mean_anomaly.cos();
    let cubic = eccentricity * cos_m / 6.0;
    let quadratic = eccentricity * sin_m * 0.5;
    let linear = 1.0 - eccentricity * cos_m;
    let constant = eccentricity * sin_m;
    let x1 = constant / linear;
    let poly = ((cubic * x1 + quadratic) * x1 + linear) * x1 - constant;
    let dpoly = (3.0 * cubic * x1 + 2.0 * quadratic) * x1 + linear;
    mean_anomaly + x1 - poly / dpoly
}

/// One Halley correction on `F(x) = x - m - e sin(x)`:
/// `x - 2 f f' / (2 f'^2 - f f'')`. Returns the corrected iterate beside the
/// exact `f64::sin` word evaluated at the INPUT iterate (the cache).
fn halley_correction(anomaly: f64, mean_anomaly: f64, eccentricity: f64) -> (f64, f64) {
    let s = anomaly.sin();
    let c = anomaly.cos();
    let f = anomaly - mean_anomaly - eccentricity * s;
    let fp = 1.0 - eccentricity * c;
    let fpp = eccentricity * s;
    (anomaly - 2.0 * f * fp / (2.0 * fp * fp - f * fpp), s)
}

/// Starter plus `corrections` complete Halley corrections on the same map.
/// The returned count covers every complete candidate construction, the
/// starter included; certificate evaluation is separate cost and not counted.
/// The cached pair `(x, f64::sin(x))` is the last correction's input iterate
/// and its sine word (; consumed only when the iterate did not move).
fn refine(
    starter: f64,
    mean_anomaly: f64,
    eccentricity: f64,
    corrections: u8,
) -> Result<(f64, u8, SinCache), KeplerError> {
    let mut anomaly = starter;
    let mut rounds = 1u8;
    let mut cached: SinCache = None;
    if !anomaly.is_finite() {
        return Err(KeplerError::IterationExhaustion);
    }
    for _ in 0..corrections {
        let (next, sin_at_input) = halley_correction(anomaly, mean_anomaly, eccentricity);
        cached = Some((anomaly, sin_at_input));
        anomaly = next;
        rounds += 1;
        if !anomaly.is_finite() {
            return Err(KeplerError::IterationExhaustion);
        }
    }
    Ok((anomaly, rounds, cached))
}

/// Knuth two-sum: `(s, err)` with `a + b = s + err` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// Two-product via `mul_add`: `(p, err)` with `a * b = p + err` exactly
/// (platform assumption 1).
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let err = a.mul_add(b, -p);
    (p, err)
}

/// Outward product of nonnegative bounds: one `next_up` past the single
/// multiplication's at-most-half-ulp rounding.
fn outward_product(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}

/// Outward sum of nonnegative bounds: one `next_up` past the single
/// addition's at-most-half-ulp rounding.
fn outward_sum(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}

/// Outward allowance for one ulp of `v`: `2^-52 |v| + 2^-1074`, which covers
/// `ulp(v)` for every finite `v` (subnormals through the floor).
fn ulp_allowance(v: f64) -> f64 {
    outward_sum(outward_product(SINE_ULP_FACTOR, v.abs()), SUBNORMAL_FLOOR)
}

/// Outward allowance for half an ulp of `v`: `2^-53 |v| + 2^-1074`.
fn half_ulp_allowance(v: f64) -> f64 {
    outward_sum(outward_product(HALF_ULP_FACTOR, v.abs()), SUBNORMAL_FLOOR)
}

#[cfg(test)]
mod tests {
    // Exact-real regression for the error-bound and ceiling construction
    //. The checks below use no rounded division:
    // `error_bound * (1 - e) >= defect_bound` is decided by exact
    // cross-multiplication (`two_sum`/`two_prod` are error-free) and an exact
    // sign test for the resulting sums of binary64 values.

    use super::*;

    // Raw-input shims over the folded `KeplerPlan`: the exact-real
    // sweeps below pass `(d, e)` pairs, while the kernel shares one plan per
    // solve. Each shim folds one plan per call and delegates to the same
    // methods the solve path runs, so the construction under test is the
    // shipped one; the plan folding itself is exercised end to end by
    // `tests/kepler.rs` (`solve_with_plan` over the sampled population).
    fn outward_error_bound(defect_bound: f64, eccentricity: f64) -> f64 {
        KeplerPlan::fold(eccentricity).outward_error_bound(defect_bound)
    }

    fn service_error_ceiling(eccentricity: f64) -> f64 {
        KeplerPlan::fold(eccentricity).service_error_ceiling()
    }

    fn certificate(
        mean_anomaly: f64,
        eccentricity: f64,
        candidate: f64,
    ) -> Result<(f64, f64), KeplerError> {
        KeplerPlan::fold(eccentricity).certificate(mean_anomaly, candidate, None)
    }

    /// Exact sign test: `true` iff `sum(vals) >= 0` over the reals.
    ///
    /// A two-sum cascade keeps `lead + residuals = sum(vals)` exactly at every
    /// round. When the leading term strictly dominates the rounded-up residual
    /// magnitude the sign is that of the lead; otherwise the exact residuals
    /// (each round shrinks the largest magnitude by at least ~2^-48) are
    /// cascaded again. The cap panics rather than guessing; it cannot be
    /// reached on this domain.
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

    /// `error_bound * (1 - e) >= defect_bound` over the exact reals:
    /// `1 - e = d0 + d1` exactly and `r (d0 + d1) = (p0 + t0) + (p1 + t1)`
    /// exactly (two-product), so the sign test below is exact.
    fn covers_exact(defect_bound: f64, eccentricity: f64, error_bound: f64) -> bool {
        let (d0, d1) = two_sum(1.0, -eccentricity);
        let (p0, t0) = two_prod(error_bound, d0);
        let (p1, t1) = two_prod(error_bound, d1);
        exact_sum_nonnegative(&[p0, t0, p1, t1, -defect_bound])
    }

    /// `service_error_ceiling(e) * (1 - e) <= 4 eps` over the exact reals (the
    /// same exact cross-multiplication, inequality flipped).
    fn ceiling_within_sla(eccentricity: f64) -> bool {
        let (d0, d1) = two_sum(1.0, -eccentricity);
        let ceiling = service_error_ceiling(eccentricity);
        let (p0, t0) = two_prod(ceiling, d0);
        let (p1, t1) = two_prod(ceiling, d1);
        exact_sum_nonnegative(&[SERVICE_DEFECT_LIMIT, -p0, -t0, -p1, -t1])
    }

    /// The lead-review counterexample against the single-`next_up` quotient
    /// (`e = 0.34019998659089296`, `d = 0x1.51d14eaefc25dp-65`), plus
    /// adversarial `(e, d)` pairs: subnormal `d` (the `m = 0`, `m = +/-1e-300`
    /// rows land there), the `4 eps` defect ceiling, one-ulp neighbours of the
    /// counterexample and zero.
    #[test]
    fn error_bound_covers_the_exact_real_quotient() {
        let counterexample_d = f64::from_bits(0x3BE5_1D14_EAEF_C25D);
        let eccentricities = [
            0.340_199_986_590_892_96,
            0.1,
            0.2,
            0.45,
            0.0167,
            0.15,
            0.5,
            0.9,
            SERVICE_ECCENTRICITY,
            0.001,
        ];
        let defects = [
            counterexample_d,
            f64::from_bits(counterexample_d.to_bits() + 1),
            f64::from_bits(counterexample_d.to_bits() - 1),
            f64::from_bits(1),
            f64::from_bits(2),
            f64::from_bits(3),
            f64::from_bits(7),
            f64::from_bits(8),
            f64::from_bits(14),
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE * 2.0,
            7.888_609_052_210_118e-31,
            3.6e-20,
            SERVICE_DEFECT_LIMIT,
            SERVICE_DEFECT_LIMIT * 0.75,
            0.0,
        ];
        for &e in &eccentricities {
            for &d in &defects {
                let r = outward_error_bound(d, e);
                assert!(
                    covers_exact(d, e, r),
                    "inward error bound: r * (1 - e) < d over the reals at \
                     e={e:?} d={d:?} (0x{:016x}) r={r:?} (0x{:016x})",
                    d.to_bits(),
                    r.to_bits()
                );
            }
        }
        // End-to-end: the certificate's own construction must satisfy the same
        // exact conclusion (subnormal defect at the circular fixed point).
        for (m, e, x) in [
            (0.0, 0.45, 0.0),
            (0.0, 0.340_199_986_590_892_96, 0.0),
            (1e-300, 0.9, 1e-299),
            (1.0, 0.340_199_986_590_892_96, 1.2),
            (-2.5, 0.15, -2.4),
        ] {
            let (d, r) = certificate(m, e, x).expect("bounded candidate certifies");
            assert!(
                covers_exact(d, e, r),
                "inward certificate error bound at m={m:?} e={e:?} x={x:?}: \
                 d={d:?} (0x{:016x}) r={r:?} (0x{:016x})",
                d.to_bits(),
                r.to_bits()
            );
        }
    }

    /// Acceptance must imply `r <= 4 kappa eps` over the exact reals: the
    /// ceiling is conservative everywhere on the admitted eccentricity range.
    #[test]
    fn service_ceiling_is_exact_real_conservative() {
        for i in 1..950 {
            let e = f64::from(i) * 0.001;
            assert!(
                ceiling_within_sla(e),
                "ceiling exceeds 4 kappa eps at e={e:?}"
            );
        }
        let base = 0.340_199_986_590_892_96_f64.to_bits();
        for bits in base..base + 0x1_0000 {
            // mantissa sweep around the lead-review counterexample's e
            let e = f64::from_bits(bits);
            assert!(
                ceiling_within_sla(e),
                "ceiling exceeds 4 kappa eps at e={e:?} (0x{:016x})",
                e.to_bits()
            );
        }
        for e in [
            0.1,
            0.2,
            0.340_199_986_590_892_96,
            0.45,
            0.9,
            SERVICE_ECCENTRICITY,
        ] {
            assert!(
                ceiling_within_sla(e),
                "ceiling exceeds 4 kappa eps at e={e:?}"
            );
        }
    }
}
