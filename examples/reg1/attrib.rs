//! The absent-probe attribution instrument: what one current-strata `absent-miss` cell's ratio is
//! made of, on one arm at a time.
//!
//! The `--check performance` run reports a ratio. This mode reports the *mechanism* behind a ratio,
//! and it is a diagnostic, never an acceptance path: it never calls
//! [`crate::measure_guard::acquire`], it accepts no `--accept`, and it changes no campaign number.
//!
//! **What it exports, per absent probe, for the selected arm.**
//
//! * *Probe length* — slots the bounded walk read (`apeiron::table::read_stats::sweep_steps`). For
//!   the incumbent this is `not-available`; see [`ht_probe_length_reason`].
//! * *Lines* — 64-byte cache lines the walk opened (`read_stats::sweep_lines`). For the incumbent
//!   this is `not-available` from the source, and the hardware's
//!   `mem_load_retired.l1_miss` is the substitute, reported by `perf stat` on this binary.
//! * *Shape* — capacity, live length and `hashtrinity::Table::longest_probe` for the incumbent;
//!   the region bound for Apeiron.
//! * *Elapsed* — the same lookup loop `strata::measure_apeiron` / `strata::measure_ht` time, so the
//!   per-probe nanoseconds here and the campaign's `ns/op` are the same measurement.
//!
//! **The hardware window.** `--window CTL:ACK` is the `perf stat --control=fifo:` pair. The mode
//! builds and touches its table, then sends `enable instructions` / `disable instructions` around
//! the probe loop only, so `perf stat` reports the counters for the probes and not for the build.
//! Without `--window` the whole process is counted and the output line says so.
//!
//! ```text
//! reg1 --check absent-attrib --cell occ16-clustered-skewed-absent-miss-replacement \
//!       --arm apeiron --reps 5
//! reg1 --check absent-attrib --cell occ16-clustered-skewed-absent-miss-replacement --arm ht
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::matrix::MergeMode;
use crate::strata::{
    build_apeiron, build_ht, build_trace, fold, make_plan, merge_fn, probe_capacity, CellSpec,
    Distribution, KeyPlan, Operation, StrataKey, Trace, FNV_OFFSET, SHARDS,
};

/// The occupancy strata the current cells are enumerated over.
const OCCUPANCIES: [usize; 5] = [1, 2, 4, 8, 16];

/// The default `perf stat` event group this mode enables and disables.
///
/// `--control` addresses a group by its **full** first-event spec, including the PMU prefix: on
/// this hybrid host the group `{cpu_core/instructions/u,...}` is named `cpu_core/instructions/u`,
/// and `enable instructions` is refused with `can't find 'instructions' event` while leaving the
/// whole process counted. `--window-group` overrides it. `cpu_core/` is the P-core PMU; the
/// E-core PMU (`cpu_atom`) has no `mem_load_retired`, so an unpinned run reports
/// `<not counted>` for both miss events.
const WINDOW_GROUP: &str = "cpu_core/instructions/u";

/// Why the incumbent's probe length and line count cannot be read from the source.
const HT_PROBE_LENGTH_REASON: &str =
    "hashtrinity::Table::get returns Option<&V>, not a slot index; the walk's home is the private \
     fn home and the tail's private fn descend, both private on 4cc14471 \
     (hashtrinity/src/lib.rs:409, :662, :741), and the coords slice is reached through the private \
     Table::get_unchecked. No public entry point hands out the lanes a probe visited, so a wrapper \
     around get cannot count them; the hardware l1_miss counter on this binary is the substitute";

/// Run the absent-probe attribution for one cell and one arm.
///
/// Returns `64` for a usage error and `1` when the cell cannot be built.
pub(crate) fn run(args: &[String]) -> i32 {
    let Some(cell_id) = flag(args, "--cell") else {
        eprintln!(
            "usage: reg1 --check absent-attrib --cell <id> --arm apeiron|ht [--reps N] \
                   [--window CTL:ACK]"
        );
        return 64;
    };
    let Some(arm) = flag(args, "--arm") else {
        eprintln!(
            "usage: reg1 --check absent-attrib --cell <id> --arm apeiron|ht [--reps N] \
                   [--window CTL:ACK]"
        );
        return 64;
    };
    let reps = flag(args, "--reps").map_or(5_usize, |raw| raw.parse::<usize>().unwrap_or(0));
    if reps == 0 {
        eprintln!("--reps must be a positive integer this platform can hold");
        return 64;
    }
    if !matches!(arm.as_str(), "apeiron" | "ht") {
        eprintln!("--arm must be apeiron or ht, not {arm}");
        return 64;
    }
    let Some(spec) = absent_miss_spec(&cell_id) else {
        eprintln!(
            "{cell_id} is not an absent-miss cell of the current strata; \
             usage: reg1 --check absent-attrib --cell <id> --arm apeiron|ht"
        );
        return 64;
    };

    let total = SHARDS * spec.occupancy;
    let (capacity, bits) = probe_capacity(total);
    let plan = match make_plan(spec, capacity, bits) {
        Ok(plan) => plan,
        Err(reason) => {
            eprintln!("cell {cell_id} has no plan: {reason}");
            return 1;
        }
    };
    let trace = build_trace(spec, &plan);
    let merge = merge_fn(spec.merge);

    let present = plan.present.len();
    let absent = plan.absent.len();
    let ops = trace.lookups.len();
    let window = if flag(args, "--window").is_some() {
        "perf-control"
    } else {
        "whole-process"
    };
    println!(
        "absent_attrib cell={cell_id} arm={arm} reps={reps} population={present} \
         ht_capacity={capacity} present={present} absent={absent} ops={ops} \
         walk_window={window}"
    );

    match arm.as_str() {
        "apeiron" => measure_apeiron_arm(&plan, &trace, merge, reps, args),
        _ => measure_ht_arm(&plan, &trace, merge, reps, args),
    }
}

/// Run the Apeiron arm: the timed absent loop, plus the `read-stats` geometry of the same loop.
///
/// The counters are read around *each* rep and printed per rep, so the spread is the spread of the
/// same quantity the elapsed column reports.
#[cfg(feature = "read-stats")]
fn measure_apeiron_arm(
    plan: &KeyPlan,
    trace: &Trace,
    merge: crate::strata::MergeFn,
    reps: usize,
    args: &[String],
) -> i32 {
    let mut window = Window::open(args);
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        // A fresh table per rep, exactly as the campaign builds one per sample, so the walk sees
        // the same populated geometry every time.
        let (table, admitted) = build_apeiron(&plan.present, merge);
        if admitted != plan.present.len() {
            eprintln!("apeiron admitted {admitted} != {}", plan.present.len());
            return 1;
        }
        let keys: Vec<StrataKey> = trace.lookups.clone();
        // Warm the table so the timed window reads a resident region, as `touch_apeiron` does.
        let mut warm = 0i64;
        for key in &plan.present {
            warm = warm.wrapping_add(*table.get(key));
        }
        std::hint::black_box(warm);

        apeiron::table::read_stats::clear();
        window.enable();
        let mut hash = FNV_OFFSET;
        let start = std::time::Instant::now();
        for key in &keys {
            fold(&mut hash, *table.get(std::hint::black_box(key)));
        }
        let nanos = crate::strata::nanos_since(start);
        window.disable();
        let stats = apeiron::table::read_stats::snapshot();
        std::hint::black_box(hash);
        samples.push(AttributedSample {
            nanos,
            probes: u64::try_from(keys.len()).unwrap_or(u64::MAX),
            sweep_steps: stats.sweep_steps,
            sweep_lines: stats.sweep_lines,
            tag_matches: stats.tag_matches,
            full_compares: stats.full_compares,
            absence_by_vacancy: stats.summary_hits,
            absence_by_full_cycle: stats.summary_misses,
        });
    }
    report("apeiron", &samples, Some(HT_PROBE_LENGTH_REASON));
    0
}

/// The Apeiron arm without the `read-stats` feature: the same loop, no geometry.
#[cfg(not(feature = "read-stats"))]
fn measure_apeiron_arm(
    plan: &KeyPlan,
    trace: &Trace,
    merge: crate::strata::MergeFn,
    reps: usize,
    _args: &[String],
) -> i32 {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (table, admitted) = build_apeiron(&plan.present, merge);
        if admitted != plan.present.len() {
            eprintln!("apeiron admitted {admitted} != {}", plan.present.len());
            return 1;
        }
        let keys: Vec<StrataKey> = trace.lookups.clone();
        let mut hash = FNV_OFFSET;
        let start = std::time::Instant::now();
        for key in &keys {
            fold(&mut hash, *table.get(std::hint::black_box(key)));
        }
        let nanos = crate::strata::nanos_since(start);
        std::hint::black_box(hash);
        samples.push(AttributedSample {
            nanos,
            probes: u64::try_from(keys.len()).unwrap_or(u64::MAX),
            sweep_steps: 0,
            sweep_lines: 0,
            tag_matches: 0,
            full_compares: 0,
            absence_by_vacancy: 0,
            absence_by_full_cycle: 0,
        });
    }
    report(
        "apeiron",
        &samples,
        Some("built without --features read-stats: probe length and lines are not compiled in"),
    );
    0
}

/// Run the incumbent arm: the same timed absent loop over the pinned `hashtrinity::Table`.
///
/// The incumbent exposes no probe geometry, so this arm reports shape and elapsed only, and names
/// the missing half with its reason. The hardware counters `perf stat` prints for this process are
/// the incumbent's own line count.
fn measure_ht_arm(
    plan: &KeyPlan,
    trace: &Trace,
    merge: crate::strata::MergeFn,
    reps: usize,
    args: &[String],
) -> i32 {
    let mut window = Window::open(args);
    let mut samples = Vec::with_capacity(reps);
    let mut shape = (0_usize, 0_usize);
    for _ in 0..reps {
        let (table, len) = build_ht(&plan.present, merge);
        if len != plan.present.len() {
            eprintln!("hashtrinity len {len} != {}", plan.present.len());
            return 1;
        }
        shape = (table.capacity(), table.longest_probe());
        let keys: Vec<StrataKey> = trace.lookups.clone();
        let mut warm = 0i64;
        for key in &plan.present {
            warm = warm.wrapping_add(table.get(key).copied().unwrap_or(0));
        }
        std::hint::black_box(warm);

        window.enable();
        let mut hash = FNV_OFFSET;
        let start = std::time::Instant::now();
        for key in &keys {
            fold(
                &mut hash,
                table.get(std::hint::black_box(key)).copied().unwrap_or(0),
            );
        }
        let nanos = crate::strata::nanos_since(start);
        window.disable();
        std::hint::black_box(hash);
        samples.push(AttributedSample {
            nanos,
            probes: u64::try_from(keys.len()).unwrap_or(u64::MAX),
            sweep_steps: 0,
            sweep_lines: 0,
            tag_matches: 0,
            full_compares: 0,
            absence_by_vacancy: 0,
            absence_by_full_cycle: 0,
        });
    }
    let (ht_capacity, longest_probe) = shape;
    let load = extent(plan.present.len()) / extent(ht_capacity.max(1));
    println!(
        "absent_attrib_shape arm=ht capacity={ht_capacity} len={} load={load:.4} \
         longest_probe={longest_probe} linear_prefix=8 descent=radix \
         coords_bytes={} ht_rev={}",
        plan.present.len(),
        ht_capacity * core::mem::size_of::<u64>(),
        crate::PINNED_HASHTRINITY_REV,
    );
    report("ht", &samples, Some(HT_PROBE_LENGTH_REASON));
    0
}

/// One rep's measurement of one arm.
struct AttributedSample {
    /// Elapsed nanoseconds of the absent-probe loop.
    nanos: u64,
    /// Absent probes issued in the rep.
    probes: u64,
    /// Slots the bounded walk read.
    sweep_steps: u64,
    /// 64-byte lines the walk opened.
    sweep_lines: u64,
    /// Lanes equal to the query fingerprint.
    tag_matches: u64,
    /// Full `(Sigma, K)` compares reached.
    full_compares: u64,
    /// Reads that stopped on a vacancy.
    absence_by_vacancy: u64,
    /// Reads that completed a full cycle with no vacancy.
    absence_by_full_cycle: u64,
}

/// One report operand as the `f64` its ratio is computed in.
///
/// Exact over every population this instrument builds. Every operand is a per-rep counter over one
/// bounded table region, and a `StrataKey` is a single `u64` tag, so the population reaches
/// `f64`'s 52-bit mantissa at 36 PB of resident keys; the one duration operand, `nanos`, reaches it
/// after 142 years of a single rep. Both are outside the report's domain, and every ratio is
/// printed to four decimals, so a rounding at the mantissa would sit under the printed precision
/// rather than change the number reported.
#[allow(
    clippy::cast_precision_loss,
    reason = "ratio operands are counts and one duration, bounded far below f64's 52-bit mantissa as derived above"
)]
fn operand(value: u64) -> f64 {
    value as f64
}

/// [`operand`] for a length or capacity, which `u64` holds on the whole of every Rust target.
fn extent(value: usize) -> f64 {
    operand(value as u64)
}

/// Print the per-rep lines and the pooled mean and spread.
fn report(arm: &str, samples: &[AttributedSample], missing: Option<&str>) {
    for (index, sample) in samples.iter().enumerate() {
        println!(
            "absent_attrib_rep arm={arm} rep={index} probes={} ns={} ns_per_probe={:.4} \
             probe_len={:.4} lines_per_probe={:.4} tag_matches={} full_compares={} \
             absence_by_vacancy={} absence_by_full_cycle={}",
            sample.probes,
            sample.nanos,
            operand(sample.nanos) / operand(sample.probes),
            operand(sample.sweep_steps) / operand(sample.probes),
            operand(sample.sweep_lines) / operand(sample.probes),
            sample.tag_matches,
            sample.full_compares,
            sample.absence_by_vacancy,
            sample.absence_by_full_cycle,
        );
    }
    let ns: Vec<f64> = samples
        .iter()
        .map(|s| operand(s.nanos) / operand(s.probes))
        .collect();
    let (mean, lo, hi) = mean_and_spread(&ns);
    let probed = samples.iter().any(|s| s.sweep_steps != 0);
    let lens: Vec<f64> = samples
        .iter()
        .map(|s| operand(s.sweep_steps) / operand(s.probes))
        .collect();
    let lines: Vec<f64> = samples
        .iter()
        .map(|s| operand(s.sweep_lines) / operand(s.probes))
        .collect();
    let (len_mean, len_lo, len_hi) = mean_and_spread(&lens);
    let (line_mean, line_lo, line_hi) = mean_and_spread(&lines);
    println!(
        "absent_attrib_mean arm={arm} reps={} ns_per_probe_mean={mean:.4} \
         ns_per_probe_min={lo:.4} ns_per_probe_max={hi:.4} probe_len_mean={len_mean:.4} \
         probe_len_min={len_lo:.4} probe_len_max={len_hi:.4} lines_per_probe_mean={line_mean:.4} \
         lines_per_probe_min={line_lo:.4} lines_per_probe_max={line_hi:.4} probe_len_source={src}",
        samples.len(),
        src = if probed {
            "read-stats"
        } else {
            "not-available"
        },
    );
    if !probed {
        if let Some(reason) = missing {
            println!(
                "absent_attrib_missing arm={arm} quantity=probe_len,distinct_lines \
                      reason=\"{reason}\""
            );
        }
    }
}

/// The mean and the observed min/max spread of one per-rep quantity.
fn mean_and_spread(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let sum: f64 = values.iter().sum();
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (sum / extent(values.len()), lo, hi)
}

/// The `perf stat --control` window, opened on the fifo pair and driven around the probe loop.
struct Window {
    ctl: Option<File>,
    ack: Option<File>,
    group: String,
}

impl Window {
    /// Open the `CTL:ACK` pair if `--window` named one, otherwise count the whole process.
    fn open(args: &[String]) -> Self {
        let group = flag(args, "--window-group").unwrap_or_else(|| WINDOW_GROUP.to_string());
        let Some(pair) = flag(args, "--window") else {
            return Self {
                ctl: None,
                ack: None,
                group,
            };
        };
        let mut parts = pair.splitn(2, ':');
        let (Some(ctl), Some(ack)) = (parts.next(), parts.next()) else {
            eprintln!("--window wants CTL:ACK");
            return Self {
                ctl: None,
                ack: None,
                group,
            };
        };
        // `perf stat` holds both ends open before it execs this process, so neither open blocks.
        match (
            File::options().write(true).open(Path::new(ctl)),
            File::open(Path::new(ack)),
        ) {
            (Ok(ctl), Ok(ack)) => Self {
                ctl: Some(ctl),
                ack: Some(ack),
                group,
            },
            (ctl, ack) => {
                eprintln!("--window {pair} did not open: ctl={ctl:?} ack={ack:?}");
                Self {
                    ctl: None,
                    ack: None,
                    group,
                }
            }
        }
    }

    /// Turn the event group on and wait for `perf`'s acknowledgement.
    fn enable(&mut self) {
        self.send("enable");
    }

    /// Turn the event group off and wait for `perf`'s acknowledgement.
    fn disable(&mut self) {
        self.send("disable");
    }

    fn send(&mut self, verb: &str) {
        let (Some(ctl), Some(ack)) = (self.ctl.as_mut(), self.ack.as_mut()) else {
            return;
        };
        let mut ack_buf = [0_u8; 1];
        if let Err(error) = writeln!(ctl, "{verb} {}", self.group) {
            eprintln!("window {verb} write failed: {error}");
            return;
        }
        let _ = ack.read(&mut ack_buf);
    }
}

/// The value of `flag`, or `None`.
fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find_map(|w| (w[0] == name).then(|| w[1].clone()))
}

/// The `absent-miss` cell with this identifier, or `None`.
///
/// The current strata enumerate `occupancy x distribution x operation x merge`; this selects the
/// one `absent-miss` cell whose generated identifier is `cell_id`, so the mode measures the same
/// cell the campaign measures and never a reconstruction of it.
fn absent_miss_spec(cell_id: &str) -> Option<CellSpec> {
    let distributions = [
        Distribution::DirectIndexResident,
        Distribution::OffsetBeyondDirect,
        Distribution::SeededRandom,
        Distribution::ClusteredSkewed,
    ];
    let operations = [
        Operation::AbsentMiss,
        Operation::PresentHit,
        Operation::ExistingUpdate,
    ];
    let merges = [MergeMode::Replacement, MergeMode::WrappingAdd];
    for occupancy in OCCUPANCIES {
        for distribution in distributions {
            for operation in operations {
                for merge in merges {
                    let spec = CellSpec {
                        occupancy,
                        distribution,
                        operation,
                        merge,
                    };
                    if spec.id() == cell_id {
                        return Some(spec);
                    }
                }
            }
        }
    }
    None
}
