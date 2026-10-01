//! Isolated machine-time baseline for the executing `matvec::row_dot_fixed` primitive.
//!
//! What this measures: the wall-clock time of one call to the unchanged production primitive
//! `apeiron::engine::row_dot_fixed` (`src/engine/matvec.rs`) on retained Q32.32 `i64` rows, at a
//! small set of row widths. Nothing else is inside the timed region.
//!
//! What this is not: it is a hardware baseline, not evidence of resource improvement. It cannot
//! support a claim that any change to the primitive is faster; the reduction the loop brief admits
//! is in the extracted visit-cost model, and a latency number neither establishes nor refutes it.
//! The GPU and CUDA throughput figures elsewhere in the internal notes time a Vulkan `q8_tile`
//! dispatch and MMQ, not this primitive, and are not attributable to it.
//!
//! Method
//!
//! * Every fixture (both rows, the domain predicate, the expected value) is built before the timed
//!   region starts and is reused unchanged across every warmup and sampled run.
//! * `row_dot_fixed` is `pub`, non-generic and not `#[inline]`, so a call from this example binary
//!   is a real cross-crate call that LLVM cannot hoist or common-subexpress; the slices and the
//!   returned value additionally pass through [`std::hint::black_box`], the per-width values are
//!   accumulated, and the accumulator is printed, so nothing measured can be dropped as dead.
//! * One sample is `iters` consecutive calls. `iters` is chosen per width to hold a sample near
//!   1 ms, so the two `Instant::now()` calls stay a small fraction of the reported quantity.
//! * The reported metric is the **median** of [`SAMPLES`] samples taken in this one process after
//!   [`WARMUP`] untimed warmup samples. `min`/`max` over the same samples are printed beside it,
//!   so the interval is shown rather than asserted. There is no cross-process repetition: the
//!   printed min..max over the sampled runs is the stated interval.
//! * The value oracle re-derives the epilogue by integer division rather than by shift, so a wrong
//!   shift or a wrong tie direction appears as a counted value mismatch. The three refusal cells
//!   check the exact `MatvecError` variant and payload.
//!
//! Admissible domain
//!
//! The timed rows are equal-length Q32.32 `i64` pairs with `|element| < 2^33`, so every integer
//! product is below `2^66` and the whole `i128` prefix sum is below `2^78` even at the widest row
//! here: the accumulator stays in range and no timed call can refuse. The predicate is *checked*,
//! not assumed — [`admissible`] recomputes every product and every running sum with `checked_mul`
//! and `checked_add`, and a fixture outside the domain is a counted `domain_violation`, never a
//! silently timed row.
//!
//! Control seam
//!
//! `APEIRON_MICRO_ROW_DOT_WIDTHS=65536` replaces the frozen width set and labels the run
//! `widths_source=env baseline=false`. That is the instrument-sensitivity control: the same binary
//! and the same primitive at a known multiplicative workload change must move `ns_per_call`. Only
//! a run printing `widths_source=default` is the citable baseline.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use apeiron::engine::{row_dot_fixed, MatvecError};

/// The frozen width set of the cited baseline: three powers of two, one odd width beside the 1024
/// row, and one power of two above them.
const WIDTHS: [usize; 6] = [8, 64, 256, 1023, 1024, 4096];
/// Untimed warmup samples per width, discarded.
const WARMUP: u32 = 3;
/// Timed samples per width; the reported metric is the median of these.
const SAMPLES: u32 = 25;
/// The calls per sample target. `iters` is this spread over the width, so one sample is a fixed
/// target length in calls at every width and the per-call figure is the sample cost over `iters`.
const SAMPLE_CALL_TARGET: usize = 2_000_000;
/// The lower bound on `iters`, so a width above the target still accumulates a measurable sample.
const MIN_ITERS: usize = 64;
/// The half-unit the epilogue adds before shifting, `2^31`.
const ROUND_HALF: i128 = 1 << 31;
/// The epilogue divisor, `2^32`.
const ROUND_SHIFT: i128 = 1 << 32;
/// The fixture magnitude bound, `2^33`, one integer Q32.32 word below the element range.
const ELEMENT_BOUND: i64 = 1 << 33;

/// The calls per sample at width `n`: the sample call target over the width, floored at
/// [`MIN_ITERS`]. One sample costs `iters·n` checked `i128` multiply-adds.
fn iters_for(n: usize) -> usize {
    (SAMPLE_CALL_TARGET / n).max(MIN_ITERS)
}

/// splitmix64, the fixture stream. A fixed seed per width makes the rows a pure function of `n`, so
/// a rerun on the same source measures the same data.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One Q32.32 element with `|v| < 2^33` (below `4.0` in Q32.32 units), so the product of any two of
/// them is below `2^66`.
fn q32_element(state: &mut u64) -> i64 {
    let bits = i64::try_from(splitmix64(state) >> 30 & 0x3_FFFF_FFFF).unwrap_or(0);
    bits - ELEMENT_BOUND
}

/// The two retained rows of one width, built once per width and never rebuilt.
fn fixtures(n: usize) -> (Vec<i64>, Vec<i64>) {
    let mut ws = 0x5EED_0000_0000_0000u64 ^ n as u64;
    let mut xs = 0xC0FF_EE00_0000_0000u64 ^ n as u64;
    let w = (0..n).map(|_| q32_element(&mut ws)).collect();
    let x = (0..n).map(|_| q32_element(&mut xs)).collect();
    (w, x)
}

/// The admissible-domain predicate of the baseline, checked rather than assumed: equal length, and
/// every `i128` product and every running sum in range. `Some(sum)` carries the exact `i128`
/// accumulator the primitive should reach, which [`oracle`] then rounds independently; `None` is a
/// fixture outside the domain, which must not be timed.
fn admissible(w_row: &[i64], x: &[i64]) -> Option<i128> {
    if w_row.len() != x.len() {
        return None;
    }
    let mut acc: i128 = 0;
    for (&weight, &activation) in w_row.iter().zip(x.iter()) {
        let prod = i128::from(weight).checked_mul(i128::from(activation))?;
        acc = acc.checked_add(prod)?;
    }
    Some(acc)
}

/// The value oracle: `floor((sum + 2^31) / 2^32)` narrowed to `i64`, re-derived by integer
/// division and remainder instead of by the arithmetic shift the primitive uses. `None` is the
/// oracle's `OutputOverflow`.
///
/// A wrong shift, a wrong rounding direction or a wrong tie direction changes this value, so a
/// defect in the epilogue is counted as a value mismatch rather than timed through.
fn oracle(sum: i128) -> Option<i64> {
    let biased = sum.checked_add(ROUND_HALF)?;
    let quotient = biased / ROUND_SHIFT;
    let remainder = biased % ROUND_SHIFT;
    let floored = if remainder < 0 {
        quotient.checked_sub(1)?
    } else {
        quotient
    };
    if floored > i128::from(i64::MAX) || floored < i128::from(i64::MIN) {
        return None;
    }
    // The two range tests above are exactly `i64::try_from`'s domain, so this narrowing is
    // infallible; the check is kept explicit so the oracle states the refusal it is checking.
    i64::try_from(floored).ok()
}

/// One timed width: the median/min/max ns per call, and the calls per sample behind each sample.
#[derive(Debug)]
struct Row {
    /// Row width `n`.
    n: usize,
    /// Calls per sample.
    iters: usize,
    /// Median ns per call over the samples.
    median: f64,
    /// Smallest sampled ns per call.
    min: f64,
    /// Largest sampled ns per call.
    max: f64,
    /// The accumulated result of the last sample, so every returned value is consumed.
    checksum: i64,
}

/// The checked outcomes of one width, before and after timing.
struct Outcome {
    /// The timing row, absent when the width is outside the admissible domain.
    row: Option<Row>,
    /// The fixture left the `i128` domain and was not timed.
    domain_violation: bool,
    /// The returned value disagreed with the oracle.
    value_mismatch: bool,
    /// The primitive refused this admissible fixture, which the domain argument says it cannot.
    refused: bool,
}

/// Time `iters` calls of the real primitive on the retained rows, `SAMPLES` times after `WARMUP`
/// untimed runs, and return the per-call ns distribution in the unit ns/call.
///
/// The timed region is exactly the `iters`-call loop: both argument slices and every returned value
/// pass through `black_box`, the values are accumulated, and the accumulator is returned, so the
/// whole chain is live at the end of the sample.
#[allow(
    clippy::cast_precision_loss,
    reason = "the ns/call quotient is a diagnostic display of an integer nanosecond count over a \
              call count; f64 carries 52 bits of mantissa and these are sub-microsecond values"
)]
fn measure(width: usize, w_row: &[i64], x: &[i64], iters: usize) -> Row {
    let run = |iters: usize| -> (u64, i64) {
        let start = Instant::now();
        let mut acc: i64 = 0;
        for _ in 0..iters {
            let value = row_dot_fixed(black_box(w_row), black_box(x)).unwrap_or(i64::MIN);
            acc = acc.wrapping_add(black_box(value));
        }
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        black_box(acc);
        (elapsed, acc)
    };
    for _ in 0..WARMUP {
        black_box(run(iters).1);
    }
    let mut samples: Vec<f64> = Vec::with_capacity(SAMPLES as usize);
    let mut checksum = 0i64;
    for _ in 0..SAMPLES {
        let (elapsed, acc) = run(iters);
        checksum = acc;
        samples.push(elapsed as f64 / iters as f64);
    }
    samples.sort_by(f64::total_cmp);
    Row {
        n: width,
        iters,
        median: samples[samples.len() / 2],
        min: samples[0],
        max: samples[samples.len() - 1],
        checksum,
    }
}

/// Build the fixtures for `n`, check the domain and the returned value, then time the width. Every
/// construction and every check happens before the timed region of [`measure`].
#[allow(
    clippy::cast_precision_loss,
    reason = "the ns/element quotient and the spread percentage are a diagnostic display of an \
              integer nanosecond median over an element count and over itself"
)]
fn measure_width(n: usize) -> Outcome {
    let (w_row, x) = fixtures(n);
    let Some(sum) = admissible(&w_row, &x) else {
        println!("row width={n} domain=outside-i128 not-timed=true");
        return Outcome {
            row: None,
            domain_violation: true,
            value_mismatch: false,
            refused: false,
        };
    };
    let expected = oracle(sum);
    let got = row_dot_fixed(black_box(&w_row), black_box(&x));
    let value_match = match (expected, &got) {
        (Some(oracle_value), Ok(observed)) => oracle_value == *observed,
        (None, Err(MatvecError::OutputOverflow)) => true,
        _ => false,
    };
    println!(
        "row width={n} domain=admissible i128_sum={sum} oracle={expected:?} got={got:?} \
         value_match={value_match}"
    );
    let mut outcome = Outcome {
        row: None,
        domain_violation: false,
        value_mismatch: !value_match,
        refused: got.is_err(),
    };
    let iters = iters_for(n);
    let row = measure(n, &w_row, &x, iters);
    println!(
        "time width={n} iters={iters} multiply_adds_per_sample={} samples={SAMPLES} \
         median_ns_per_call={:.4} min_ns_per_call={:.4} max_ns_per_call={:.4} \
         ns_per_element={:.5} spread_pct={:.2} checksum={}",
        iters * n,
        row.median,
        row.min,
        row.max,
        row.median / n as f64,
        100.0 * (row.max - row.min) / row.median,
        row.checksum,
    );
    outcome.row = Some(row);
    outcome
}

/// The CPU model line from `/proc/cpuinfo`, or `unknown`.
fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// `uname` with `flag`, trimmed, or `unknown`.
fn uname(flag: &str) -> String {
    Command::new("uname")
        .arg(flag)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

/// The value of `key` in `rustc -vV`, trimmed, or `unknown`.
///
/// This is the compiler of record for the numbers: cargo sets `RUSTC` for build scripts, not for
/// the toolchain that produced this binary's object code, so the tool is asked directly.
fn rustc_vv(key: &str) -> String {
    Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .find(|line| line.starts_with(key))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// The sha256 of a repository-relative file through `sha256sum`, or `unavailable(reason)`.
fn sha256(relative: &str) -> String {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    match Command::new("sha256sum").arg(&path).output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .split_whitespace()
            .next()
            .unwrap_or("unavailable(no-digest-field)")
            .to_string(),
        Ok(o) => format!("unavailable(sha256sum-exit-{})", o.status),
        Err(e) => format!("unavailable(sha256sum-{e})"),
    }
}

/// The width set, and whether it is the frozen baseline. `APEIRON_MICRO_ROW_DOT_WIDTHS` is the
/// control seam; only the default set is citable.
fn widths() -> (Vec<usize>, bool) {
    let Ok(spec) = std::env::var("APEIRON_MICRO_ROW_DOT_WIDTHS") else {
        return (WIDTHS.to_vec(), true);
    };
    let parsed: Vec<usize> = spec
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .collect();
    if parsed.is_empty() {
        (WIDTHS.to_vec(), true)
    } else {
        (parsed, false)
    }
}

/// Print the identity block every number below is attributed to: the machine, the toolchain, and
/// the sha256 of the two sources the evidence binds to (the primitive and this example).
fn print_identity() {
    println!("identity host={}", uname("-n"));
    println!("identity kernel={}", uname("-sr"));
    println!("identity cpu={}", cpu_model());
    println!("identity rustc_release={}", rustc_vv("release"));
    println!("identity rustc_commit_hash={}", rustc_vv("commit-hash"));
    println!("identity rustc_host={}", rustc_vv("host"));
    println!(
        "identity primitive=apeiron::engine::row_dot_fixed path=src/engine/matvec.rs sha256={}",
        sha256("src/engine/matvec.rs")
    );
    println!(
        "identity harness=examples/micro_row_dot.rs sha256={}",
        sha256("examples/micro_row_dot.rs")
    );
}

/// Check the three refusal cells — the length refusal, the `i128` accumulator refusal and the
/// `i64` output refusal — against the exact variant and payload. Anything else is a counted refusal
/// mismatch.
///
/// `AccumulatorOverflow` needs three `i64::MAX` products: two of them sum to
/// `2·(2^63−1)² = 2^127 − 2^65 + 2 < i128::MAX`, so the second element still fits and the third is
/// the first to leave `i128`. One `i64::MAX` by `i64::MAX` product is about `2^126`, so after the
/// single epilogue the rounded value is about `2^94` and leaves `i64`: that is the output refusal.
fn check_refusals() -> usize {
    let max = i64::MAX;
    let cases: [(&str, Vec<i64>, Vec<i64>, MatvecError); 3] = [
        (
            "row-length",
            vec![1 << 20, 2, 3, 4, 5, 6, 7, 8],
            vec![1 << 20, 2, 3, 4, 5, 6, 7],
            MatvecError::RowLength {
                row: 0,
                weights: 8,
                n_in: 7,
            },
        ),
        (
            "accumulator-overflow",
            vec![max, max, max],
            vec![max, max, max],
            MatvecError::AccumulatorOverflow,
        ),
        (
            "output-overflow",
            vec![max],
            vec![max],
            MatvecError::OutputOverflow,
        ),
    ];
    let mut mismatches = 0usize;
    for (name, w, x, expected) in cases {
        let got = row_dot_fixed(black_box(&w), black_box(&x));
        let match_ok = got == Err(expected.clone());
        println!("refusal case={name} expected={expected:?} got={got:?} match={match_ok}");
        if !match_ok {
            mismatches += 1;
        }
    }
    mismatches
}

/// Print the citable baseline block: the per-width figure with its sample count and interval, then
/// the identity of the primitive and the metric the numbers are.
fn print_baseline(rows: &[Row], is_baseline: bool) {
    for row in rows {
        println!(
            "baseline width={} iters={} samples={SAMPLES} median_ns_per_call={:.4} \
             min_ns_per_call={:.4} max_ns_per_call={:.4}",
            row.n, row.iters, row.median, row.min, row.max
        );
    }
    println!(
        "baseline primitive=apeiron::engine::row_dot_fixed source=src/engine/matvec.rs \
         sha256={} metric=ns_per_call statistic=median interval=min..max samples={SAMPLES} \
         process=single grade=measured claim=hardware_baseline_not_resource_improvement \
         baseline={is_baseline}",
        sha256("src/engine/matvec.rs")
    );
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the ns/call, ns/element and spread figures are a diagnostic display of integer \
              nanosecond samples; f64 carries 52 bits of mantissa and these are microsecond values"
)]
fn main() {
    let (ws, is_baseline) = widths();
    print_identity();
    println!(
        "widths={:?} widths_source={} baseline={} warmup_samples={WARMUP} samples={SAMPLES} \
         process=single timer=Instant::now metric=ns_per_call",
        ws,
        if is_baseline { "default" } else { "env" },
        is_baseline
    );

    let mut domain_violations = 0usize;
    let mut value_mismatches = 0usize;
    let mut refused_widths = 0usize;
    let mut rows: Vec<Row> = Vec::with_capacity(ws.len());
    for n in ws {
        let outcome = measure_width(n);
        domain_violations += usize::from(outcome.domain_violation);
        value_mismatches += usize::from(outcome.value_mismatch);
        refused_widths += usize::from(outcome.refused);
        rows.extend(outcome.row);
    }

    let refusal_mismatches = check_refusals();
    println!(
        "check timed_widths={} domain_violations={domain_violations} \
         value_mismatches={value_mismatches} refused_widths={refused_widths} refusal_cases=3 \
         refusal_mismatches={refusal_mismatches}",
        rows.len()
    );
    print_baseline(&rows, is_baseline);

    let failed = domain_violations + value_mismatches + refusal_mismatches;
    if failed == 0 {
        println!("RESULT row_dot_fixed isolated baseline OK");
    } else {
        println!("RESULT row_dot_fixed FAILED mismatches={failed}");
        std::process::exit(1);
    }
}
