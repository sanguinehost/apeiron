#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the per-pair acceptance decision and its report types"
)]

use std::collections::{HashMap, HashSet};
use std::process::Command;

use crate::matrix::{self, median, ratio, RegisteredCell};
use crate::performance::{self, Profile};
use crate::strata::{self, StrataWitness};

#[path = "adapters/serial.rs"]
pub(crate) mod serial;

pub(crate) const RATIO_LIMIT: f64 = 1.0;

pub(crate) const SELECTED_IMPL: &str = "apeiron_table";

pub(crate) const SELECTED_WORKERS: usize = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AddressFace {
    Matched,
    Unmatched,
    NotApplicable,
}

impl AddressFace {
    pub(crate) fn name(self) -> &'static str {
        match self {
            AddressFace::Matched => "matched",
            AddressFace::Unmatched => "unmatched",
            AddressFace::NotApplicable => "n/a",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Concurrency {
    Matched,
    Mismatch,
    NotApplicable,
}

impl Concurrency {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Concurrency::Matched => "matched",
            Concurrency::Mismatch => "MISMATCH",
            Concurrency::NotApplicable => "n/a",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PairStatus {
    Pass,
    Fail,
    Missing,
    NotApplicable,
    Informational,
    Unauthorized,
}

impl PairStatus {
    pub(crate) fn name(self) -> &'static str {
        match self {
            PairStatus::Pass => "pass",
            PairStatus::Fail => "fail",
            PairStatus::Missing => "missing",
            PairStatus::NotApplicable => "n/a",
            PairStatus::Informational => "info",
            PairStatus::Unauthorized => "unauthorized",
        }
    }
}

fn spatial_address_face(id: &str) -> AddressFace {
    match id {
        "foldhash_par16" | "fxhash_par16" => AddressFace::Matched,
        _ => AddressFace::Unmatched,
    }
}

/// The timed-region worker width of a spatial incumbent, corrected for the combined arm.
///
/// `hashtrinity_get_each` is measured by `run_count_par`, the combined `count_par` build plus
/// grouped probe: its build phase runs on the hashtrinity global pool, while the reported policy
/// names only the single-core probe (`workers = 1`, note `P=16`). Comparing that reported width
/// against the single-core selected run silently matches a parallel-build arm to a serial-build arm.
/// Return the build pool width so [`pair_concurrency`] reports the mismatch instead.
fn spatial_incumbent_workers(id: &str, reported: usize) -> usize {
    match id {
        "hashtrinity_get_each" => hashtrinity::par::Pool::global().width().max(1),
        _ => reported,
    }
}

fn serial_address_face(_: &str) -> AddressFace {
    AddressFace::Matched
}

fn serial_selected_address(cell: &str) -> &'static str {
    if cell.contains("clustered-skewed") {
        "raw-word;direct_word=None"
    } else {
        "raw-word;direct_word=Some"
    }
}

fn pair_concurrency(sel_workers: usize, inc_workers: Option<usize>) -> Concurrency {
    match inc_workers {
        None => Concurrency::NotApplicable,
        Some(inc) if sel_workers == inc => Concurrency::Matched,
        Some(_) => Concurrency::Mismatch,
    }
}

pub(crate) const HASHTRINITY_TABLE: &str = "hashtrinity_table";

pub(crate) const SPATIAL_INCUMBENTS: [&str; 5] = [
    "hashtrinity_count_par",
    "hashtrinity_get_each",
    "hashbrown_rayon",
    "foldhash_par16",
    "fxhash_par16",
];

pub(crate) fn is_spatial_incumbent(id: &str) -> bool {
    SPATIAL_INCUMBENTS.contains(&id)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AdapterState {
    /// The adapter is built; the pair is measured.
    Built,
    /// The adapter is not built; the pair is missing and rejects full acceptance. Retained in the
    /// vocabulary so a future incumbent cannot silently omit its adapter; no registered incumbent
    /// is `Missing` in this harness.
    #[allow(
        dead_code,
        reason = "the missing vocabulary is the contrast to the built adapter; every registered \
                  incumbent is built in this harness, so the variant is not constructed here"
    )]
    Missing(&'static str),
}

impl AdapterState {
    /// The adapter state's report name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            AdapterState::Built => "built",
            AdapterState::Missing(_) => "missing",
        }
    }

    /// The adapter state's registered reason.
    pub(crate) fn reason(self) -> &'static str {
        match self {
            AdapterState::Built => "adapter built in this harness",
            AdapterState::Missing(reason) => reason,
        }
    }
}

/// The pair's weight in the acceptance decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PairClass {
    /// Load-bearing: must be measured and pass.
    LoadBearing,
    /// Informational: reported, excluded from the load-bearing pass.
    Informational,
}

impl PairClass {
    /// The class's report name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            PairClass::LoadBearing => "load-bearing",
            PairClass::Informational => "informational",
        }
    }
}

/// The pair's applicability, registered before timing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Applicability {
    /// Applicable; the string is the registered semantic/API reason.
    Applicable(&'static str),
    /// Not applicable with an authorizing reason; reported without the bound.
    AuthorizedNotApplicable(&'static str),
    /// Not applicable without authorization; rejects.
    UnauthorizedNotApplicable(&'static str),
}

impl Applicability {
    /// The applicability's report name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Applicability::Applicable(_) => "applicable",
            Applicability::AuthorizedNotApplicable(_) => "authorized-not-applicable",
            Applicability::UnauthorizedNotApplicable(_) => "unauthorized-not-applicable",
        }
    }

    /// The applicability's registered reason.
    pub(crate) fn reason(self) -> &'static str {
        match self {
            Applicability::Applicable(reason)
            | Applicability::AuthorizedNotApplicable(reason)
            | Applicability::UnauthorizedNotApplicable(reason) => reason,
        }
    }
}

pub(crate) struct Incumbent {
    pub(crate) id: &'static str,
    pub(crate) symbol: &'static str,
    pub(crate) class: PairClass,
    pub(crate) adapter: AdapterState,
    pub(crate) applicability: Applicability,
}

pub(crate) const INCUMBENTS: &[Incumbent] = &[
    Incumbent {
        id: "hashtrinity_table",
        symbol: "hashtrinity::Table<StrataKey,i64>",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "current-strata serial map incumbent on the same stored-key face",
        ),
    },
    Incumbent {
        id: "hashtrinity_count_par",
        symbol: "hashtrinity::FreqMap::count_par + get_each (packed u64)",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::AuthorizedNotApplicable(
            "per-cell: the packed spatial count workload only; the fixed-shard strata cells are the \
             serial-map face",
        ),
    },
    Incumbent {
        id: "hashtrinity_get_each",
        symbol: "hashtrinity::FreqMap::get_each (packed u64 probe)",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::AuthorizedNotApplicable(
            "per-cell: the packed spatial count workload only; the fixed-shard strata cells are the \
             serial-map face",
        ),
    },
    Incumbent {
        id: "hashbrown_foldhash",
        symbol: "hashbrown::HashMap<StrataKey,i64,foldhash>",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "opaque-key exact map face; the same hit/miss/update/insert surface",
        ),
    },
    Incumbent {
        id: "hashbrown_fxhash",
        symbol: "hashbrown::HashMap<StrataKey,i64,fxhash>",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "opaque-key exact map face; the same hit/miss/update/insert surface",
        ),
    },
    Incumbent {
        id: "hashbrown_rayon",
        symbol: "hashbrown::HashMap + rayon par_iter",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::AuthorizedNotApplicable(
            "per-cell: the packed spatial count workload only; the fixed-shard strata cells are the \
             serial-map face",
        ),
    },
    Incumbent {
        id: "std_siphash",
        symbol: "std::collections::HashMap<StrataKey,i64> (SipHash-1-3)",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "correctness/determinism reference on the same total-map face",
        ),
    },
    Incumbent {
        id: "foldhash_par16",
        symbol: "foldhash par16",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::AuthorizedNotApplicable(
            "per-cell: the packed spatial count workload only; the fixed-shard strata cells are the \
             serial-map face",
        ),
    },
    Incumbent {
        id: "fxhash_par16",
        symbol: "fxhash par16",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::AuthorizedNotApplicable(
            "per-cell: the packed spatial count workload only; the fixed-shard strata cells are the \
             serial-map face",
        ),
    },
    Incumbent {
        id: "xxhash_xxh3",
        symbol: "twox-hash XXH3",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "opaque-key exact map face; the same hit/miss/update/insert surface",
        ),
    },
    Incumbent {
        id: "xxhash_xxh64",
        symbol: "twox-hash XXH64",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "opaque-key exact map face; the same hit/miss/update/insert surface",
        ),
    },
    Incumbent {
        id: "wyhash",
        symbol: "wyhash",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "opaque-key exact map face; the same hit/miss/update/insert surface",
        ),
    },
    Incumbent {
        id: "ahash",
        symbol: "ahash",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "opaque-key exact map face; the same hit/miss/update/insert surface",
        ),
    },
    Incumbent {
        id: "fnv",
        symbol: "fnv-1a",
        class: PairClass::LoadBearing,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "opaque-key exact map face; the same hit/miss/update/insert surface",
        ),
    },
    Incumbent {
        id: "blake3",
        symbol: "blake3",
        class: PairClass::Informational,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "informational cryptographic comparator; no named authentication/integrity contract in \
             this harness",
        ),
    },
    Incumbent {
        id: "sha256",
        symbol: "sha2 SHA-256",
        class: PairClass::Informational,
        adapter: AdapterState::Built,
        applicability: Applicability::Applicable(
            "informational cryptographic comparator; no named authentication/integrity contract in \
             this harness",
        ),
    },
];

/// The load-bearing incumbents the acceptance registry requires. Every id here must be registered
/// exactly once, classified [`PairClass::LoadBearing`], built, and applicable (never
/// [`Applicability::UnauthorizedNotApplicable`]).
///
/// The five hasher comparators (`xxhash_xxh3`, `xxhash_xxh64`, `wyhash`, `ahash`, `fnv`) are
/// load-bearing on the opaque-key exact map face: each is the same `hashbrown::HashMap` surface as
/// `hashbrown_foldhash` with a different `BuildHasher`, measured over the same admitted key set,
/// lookup trace, update trace and canonical readback. Classifying them informational excluded them
/// from the `selected/incumbent <= 1.0` bound and let the acceptance report a pass without their
/// comparison.
#[allow(
    dead_code,
    reason = "the registry contract is read by the cfg(test) acceptance tests; the non-test \
              example build carries no reader"
)]
pub(crate) const REQUIRED_LOAD_BEARING: [&str; 14] = [
    "hashtrinity_table",
    "hashtrinity_count_par",
    "hashtrinity_get_each",
    "hashbrown_foldhash",
    "hashbrown_fxhash",
    "hashbrown_rayon",
    "std_siphash",
    "foldhash_par16",
    "fxhash_par16",
    "xxhash_xxh3",
    "xxhash_xxh64",
    "wyhash",
    "ahash",
    "fnv",
];

/// The comparators that stay informational, each with the reason it carries no bound. A
/// cryptographic comparator is load-bearing only where a named authentication or integrity contract
/// exists; this harness names none, so the digest is reported and excluded from the ratio pass.
#[allow(
    dead_code,
    reason = "the registry contract is read by the cfg(test) acceptance tests; the non-test \
              example build carries no reader"
)]
pub(crate) const INFORMATIONAL_COMPARATORS: [(&str, &str); 2] = [
    (
        "blake3",
        "informational cryptographic comparator; no named authentication/integrity contract in this \
         harness",
    ),
    (
        "sha256",
        "informational cryptographic comparator; no named authentication/integrity contract in this \
         harness",
    ),
];

/// One measured `(cell, incumbent)` pair.
pub(crate) struct PairOutcome {
    pub(crate) cell_id: String,
    pub(crate) incumbent_id: &'static str,
    pub(crate) class: PairClass,
    pub(crate) adapter: AdapterState,
    pub(crate) applicability: Applicability,
    pub(crate) sel_impl: &'static str,
    pub(crate) sel_workers: usize,
    pub(crate) sample_basis: &'static str,
    pub(crate) inc_impl: &'static str,
    pub(crate) inc_workers: Option<usize>,
    pub(crate) address_face: AddressFace,
    pub(crate) selected_address: &'static str,
    pub(crate) oracle_digest: u64,
    pub(crate) expected_oracle_digest: u64,
    pub(crate) selected_samples: Vec<u64>,
    pub(crate) incumbent_samples: Vec<u64>,
    /// The per-campaign `selected/incumbent` ratios.
    pub(crate) campaign_ratios: Vec<f64>,
}

pub(crate) struct Campaign {
    pub(crate) cells: Vec<String>,
    pub(crate) pairs: Vec<PairOutcome>,
}

pub(crate) struct IncumbentTally {
    pub(crate) id: &'static str,
    pub(crate) class: PairClass,
    pub(crate) adapter: AdapterState,
    pub(crate) applicable: usize,
    pub(crate) passed: usize,
    pub(crate) failed: usize,
    pub(crate) missing: usize,
    pub(crate) not_applicable: usize,
    pub(crate) informational: usize,
}

pub(crate) struct AcceptanceOutcome {
    pub(crate) registered: usize,
    pub(crate) load_bearing: usize,
    pub(crate) applicable: usize,
    pub(crate) passed: usize,
    pub(crate) informational: usize,
    pub(crate) not_applicable: usize,
    /// Measured applicable pairs whose two sides ran at different worker widths.
    pub(crate) concurrency_unmatched: usize,
    /// Measured applicable pairs whose incumbent routes by a raw-coordinate face.
    pub(crate) address_face_unmatched: usize,
    /// The per-pair status, in `campaign.pairs` order.
    pub(crate) pair_status: Vec<PairStatus>,
    pub(crate) failing: Vec<String>,
    pub(crate) missing: Vec<String>,
    pub(crate) unauthorized: Vec<String>,
    pub(crate) worst: Option<(String, f64)>,
    pub(crate) full_acceptance: bool,
    pub(crate) tallies: Vec<IncumbentTally>,
}

impl AcceptanceOutcome {
    /// Whether every applicable pair passed and no required pair is missing or unauthorized.
    pub(crate) fn passed(&self) -> bool {
        self.failing.is_empty() && self.missing.is_empty() && self.unauthorized.is_empty()
    }
}

fn not_applicable_pair(
    cell_id: &str,
    inc: &Incumbent,
    reason: &'static str,
    sel_impl: &'static str,
    sel_workers: usize,
) -> PairOutcome {
    PairOutcome {
        cell_id: cell_id.to_string(),
        incumbent_id: inc.id,
        class: inc.class,
        adapter: inc.adapter,
        applicability: Applicability::AuthorizedNotApplicable(reason),
        sel_impl,
        sel_workers,
        sample_basis: "serial",
        inc_impl: inc.id,
        inc_workers: None,
        address_face: AddressFace::NotApplicable,
        selected_address: "n/a",
        oracle_digest: 0,
        expected_oracle_digest: 0,
        selected_samples: Vec::new(),
        incumbent_samples: Vec::new(),
        campaign_ratios: Vec::new(),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one function states the whole per-cell/per-incumbent pair construction"
)]
pub(crate) fn build_campaign(
    strata: &StrataWitness,
    registered: &[RegisteredCell],
    spatial: &performance::spatial::SpatialWitness,
) -> Campaign {
    let serial = serial::measure(strata);
    let mut cells = Vec::with_capacity(strata.cells.len() + spatial.cells.len());
    let mut pairs = Vec::new();
    for cell in &strata.cells {
        cells.push(cell.cell_id.clone());
        let expected_oracle_digest = registered
            .iter()
            .find(|r| r.id == cell.cell_id)
            .and_then(|r| u64::from_str_radix(&r.oracle_digest, 16).ok())
            .unwrap_or(cell.oracle_digest);
        let cell_serial = serial.iter().find(|s| s.cell_id == cell.cell_id);
        for inc in INCUMBENTS {
            if is_spatial_incumbent(inc.id) {
                pairs.push(not_applicable_pair(
                    &cell.cell_id,
                    inc,
                    "per-cell: the packed spatial count face applies to the registered spatial cells",
                    SELECTED_IMPL,
                    SELECTED_WORKERS,
                ));
                continue;
            }
            let (selected_samples, incumbent_samples, campaign_ratios, oracle_digest) =
                match inc.adapter {
                    AdapterState::Built if inc.id == HASHTRINITY_TABLE => (
                        cell.apeiron_samples.clone(),
                        cell.ht_samples.clone(),
                        cell.campaign_ratios.clone(),
                        cell.oracle_digest,
                    ),
                    AdapterState::Built => {
                        match cell_serial.and_then(|s| s.outcomes.iter().find(|o| o.id == inc.id)) {
                            Some(outcome) => (
                                cell.apeiron_samples.clone(),
                                outcome.samples.clone(),
                                outcome.campaign_ratios.clone(),
                                if outcome.digest_ok {
                                    cell.oracle_digest
                                } else {
                                    cell.oracle_digest.wrapping_add(1)
                                },
                            ),
                            None => (Vec::new(), Vec::new(), Vec::new(), 0),
                        }
                    }
                    AdapterState::Missing(_) => (Vec::new(), Vec::new(), Vec::new(), 0),
                };
            // Every applicable serial incumbent (hashtrinity::Table and the opaque-key maps) runs
            // one thread, so its width is 1; a missing adapter has no measured width.
            let inc_workers = match inc.adapter {
                AdapterState::Built => Some(1usize),
                AdapterState::Missing(_) => None,
            };
            pairs.push(PairOutcome {
                cell_id: cell.cell_id.clone(),
                incumbent_id: inc.id,
                class: inc.class,
                adapter: inc.adapter,
                applicability: inc.applicability,
                sel_impl: SELECTED_IMPL,
                sel_workers: SELECTED_WORKERS,
                sample_basis: "serial",
                inc_impl: inc.id,
                inc_workers,
                address_face: serial_address_face(inc.id),
                selected_address: serial_selected_address(&cell.cell_id),
                oracle_digest,
                expected_oracle_digest,
                selected_samples,
                incumbent_samples,
                campaign_ratios,
            });
        }
    }
    for cell in &spatial.cells {
        cells.push(cell.cell_id.clone());
        for inc in INCUMBENTS {
            if !is_spatial_incumbent(inc.id) {
                pairs.push(not_applicable_pair(
                    &cell.cell_id,
                    inc,
                    "per-cell: the serial-map face applies to the fixed-shard strata cells",
                    SELECTED_IMPL,
                    SELECTED_WORKERS,
                ));
                continue;
            }
            let outcome = cell.outcomes.iter().find(|o| o.id == inc.id);
            let (
                selected_samples,
                incumbent_samples,
                campaign_ratios,
                oracle_digest,
                inc_workers,
                sel_workers,
                sample_basis,
            ) = match outcome {
                Some(outcome) => (
                    outcome.selected_samples.clone(),
                    outcome.samples.clone(),
                    outcome.campaign_ratios.clone(),
                    if outcome.digest_ok {
                        cell.oracle_digest
                    } else {
                        cell.oracle_digest.wrapping_add(1)
                    },
                    // The width the arm actually ran under, read from its reported policy and
                    // corrected for the combined get_each arm, whose build phase runs on the
                    // hashtrinity pool while only the probe is single-core.
                    Some(spatial_incumbent_workers(inc.id, outcome.policy.workers)),
                    outcome.selected_policy.workers,
                    if outcome.selected_policy.workers == 1 {
                        "serial"
                    } else {
                        "parallel"
                    },
                ),
                None => (Vec::new(), Vec::new(), Vec::new(), 0, None, 0, "unmeasured"),
            };
            pairs.push(PairOutcome {
                cell_id: cell.cell_id.clone(),
                incumbent_id: inc.id,
                class: inc.class,
                adapter: inc.adapter,
                applicability: Applicability::Applicable(
                    "packed spatial count workload on the registered 71942db scene",
                ),
                sel_impl: SELECTED_IMPL,
                sel_workers,
                sample_basis,
                inc_impl: inc.id,
                inc_workers,
                address_face: spatial_address_face(inc.id),
                selected_address: "spread-word;direct_word=mixed",
                oracle_digest,
                expected_oracle_digest: cell.oracle_digest,
                selected_samples,
                incumbent_samples,
                campaign_ratios,
            });
        }
    }
    Campaign { cells, pairs }
}

fn update_worst(worst: &mut Option<(String, f64)>, cell: &str, incumbent: &str, r: f64) {
    if worst.as_ref().is_none_or(|(_, w)| r > *w) {
        *worst = Some((format!("{cell}/{incumbent}"), r));
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one function states the whole per-pair acceptance contract in report order"
)]
pub(crate) fn evaluate(campaign: &Campaign, full_profile: bool) -> AcceptanceOutcome {
    let mut tallies: Vec<IncumbentTally> = INCUMBENTS
        .iter()
        .map(|inc| IncumbentTally {
            id: inc.id,
            class: inc.class,
            adapter: inc.adapter,
            applicable: 0,
            passed: 0,
            failed: 0,
            missing: 0,
            not_applicable: 0,
            informational: 0,
        })
        .collect();
    let tally_index: HashMap<&str, usize> =
        tallies.iter().enumerate().map(|(i, t)| (t.id, i)).collect();
    let cell_set: HashSet<&str> = campaign.cells.iter().map(String::as_str).collect();
    let mut seen: HashSet<(&str, &str)> = HashSet::with_capacity(campaign.pairs.len());
    let mut failing = Vec::new();
    let mut missing = Vec::new();
    let mut unauthorized = Vec::new();
    let mut worst: Option<(String, f64)> = None;
    let mut passed = 0usize;
    let mut load_bearing = 0usize;
    let mut applicable = 0usize;
    let mut informational = 0usize;
    let mut not_applicable = 0usize;
    let mut concurrency_unmatched = 0usize;
    let mut address_face_unmatched = 0usize;
    let mut pair_status: Vec<PairStatus> = Vec::with_capacity(campaign.pairs.len());

    for pair in &campaign.pairs {
        let Some(&index) = tally_index.get(pair.incumbent_id) else {
            failing.push(format!("unknown incumbent {}", pair.incumbent_id));
            pair_status.push(PairStatus::Fail);
            continue;
        };
        if !cell_set.contains(pair.cell_id.as_str()) {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) is not a registered cell",
                pair.cell_id, pair.incumbent_id
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        if !seen.insert((pair.cell_id.as_str(), pair.incumbent_id)) {
            tallies[index].failed += 1;
            failing.push(format!(
                "duplicate pair ({}, {})",
                pair.cell_id, pair.incumbent_id
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        if pair.class == PairClass::Informational {
            // A built informational adapter is reported and excluded from the load-bearing ratio
            // pass, but a wrong result is a harness failure and still rejects.
            if let AdapterState::Built = pair.adapter {
                if pair.oracle_digest != pair.expected_oracle_digest {
                    tallies[index].failed += 1;
                    failing.push(format!(
                        "({}, {}) wrong oracle {:016x} != {:016x}",
                        pair.cell_id,
                        pair.incumbent_id,
                        pair.oracle_digest,
                        pair.expected_oracle_digest
                    ));
                    pair_status.push(PairStatus::Fail);
                    continue;
                }
            }
            tallies[index].informational += 1;
            informational += 1;
            pair_status.push(PairStatus::Informational);
            continue;
        }
        tallies[index].applicable += 1;
        load_bearing += 1;
        match pair.applicability {
            Applicability::UnauthorizedNotApplicable(reason) => {
                tallies[index].failed += 1;
                unauthorized.push(format!(
                    "({}, {}): {reason}",
                    pair.cell_id, pair.incumbent_id
                ));
                pair_status.push(PairStatus::Unauthorized);
                continue;
            }
            Applicability::AuthorizedNotApplicable(_) => {
                tallies[index].not_applicable += 1;
                not_applicable += 1;
                pair_status.push(PairStatus::NotApplicable);
                continue;
            }
            Applicability::Applicable(_) => {}
        }
        applicable += 1;
        if let AdapterState::Missing(reason) = pair.adapter {
            tallies[index].missing += 1;
            missing.push(format!(
                "({}, {}): {reason}",
                pair.cell_id, pair.incumbent_id
            ));
            pair_status.push(PairStatus::Missing);
            continue;
        }
        if pair.oracle_digest != pair.expected_oracle_digest {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) wrong oracle {:016x} != {:016x}",
                pair.cell_id, pair.incumbent_id, pair.oracle_digest, pair.expected_oracle_digest
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        if pair.selected_samples.is_empty()
            || pair.incumbent_samples.is_empty()
            || pair.campaign_ratios.is_empty()
        {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) invalid/empty samples",
                pair.cell_id, pair.incumbent_id
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        if pair.selected_samples.contains(&0) || pair.incumbent_samples.contains(&0) {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) zero-nanosecond sample",
                pair.cell_id, pair.incumbent_id
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        if pair.campaign_ratios.iter().any(|r| !r.is_finite()) {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) non-finite ratio interval",
                pair.cell_id, pair.incumbent_id
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        // The pair is measured and valid. A phase/concurrency mismatch is checked first: two sides
        // that did not run the same timed work at the same width are not comparable, regardless of
        // address face. The comparable address mismatch is a distinct acceptance failure; `n/a` is
        // not a claim of comparability and remains non-gating.
        let concurrency = pair_concurrency(pair.sel_workers, pair.inc_workers);
        if concurrency == Concurrency::Mismatch {
            concurrency_unmatched += 1;
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) concurrency=MISMATCH selected_workers={} incumbent_workers={}",
                pair.cell_id,
                pair.incumbent_id,
                pair.sel_workers,
                pair.inc_workers.map_or(0, |w| w)
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        if pair.address_face == AddressFace::Unmatched {
            address_face_unmatched += 1;
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) address_face=unmatched selected_address={}",
                pair.cell_id, pair.incumbent_id, pair.selected_address
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        let r_max = pair
            .campaign_ratios
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if r_max > RATIO_LIMIT {
            tallies[index].failed += 1;
            update_worst(&mut worst, &pair.cell_id, pair.incumbent_id, r_max);
            failing.push(format!(
                "({}, {}) rmax={r_max:.4} > {RATIO_LIMIT:.2}",
                pair.cell_id, pair.incumbent_id
            ));
            pair_status.push(PairStatus::Fail);
            continue;
        }
        tallies[index].passed += 1;
        passed += 1;
        pair_status.push(PairStatus::Pass);
    }

    for cell in &campaign.cells {
        for inc in INCUMBENTS
            .iter()
            .filter(|i| i.class == PairClass::LoadBearing)
        {
            if !seen.contains(&(cell.as_str(), inc.id)) {
                if let Some(&index) = tally_index.get(inc.id) {
                    tallies[index].missing += 1;
                }
                missing.push(format!("({cell}, {}) omitted required pair", inc.id));
            }
        }
    }

    let full_acceptance =
        full_profile && failing.is_empty() && missing.is_empty() && unauthorized.is_empty();
    AcceptanceOutcome {
        registered: campaign.pairs.len(),
        load_bearing,
        applicable,
        passed,
        informational,
        not_applicable,
        concurrency_unmatched,
        address_face_unmatched,
        pair_status,
        failing,
        missing,
        unauthorized,
        worst,
        full_acceptance,
        tallies,
    }
}

fn print_registration() {
    for inc in INCUMBENTS {
        println!(
            "acceptance_registration incumbent={} class={} adapter={} applicability={} \
             symbol=\"{}\" adapter_reason=\"{}\" applicability_reason=\"{}\"",
            inc.id,
            inc.class.name(),
            inc.adapter.name(),
            inc.applicability.name(),
            inc.symbol,
            inc.adapter.reason(),
            inc.applicability.reason()
        );
    }
}

fn print_acceptance(outcome: &AcceptanceOutcome) {
    println!(
        "acceptance_rule selected/incumbent <= {RATIO_LIMIT:.2} on every applicable (cell, \
         incumbent) pair; the campaign interval max must be <= {RATIO_LIMIT:.2}"
    );
    for tally in &outcome.tallies {
        println!(
            "acceptance_incumbent id={} class={} adapter={} applicable={} pass={} fail={} \
             missing={} not_applicable={} informational={}",
            tally.id,
            tally.class.name(),
            tally.adapter.name(),
            tally.applicable,
            tally.passed,
            tally.failed,
            tally.missing,
            tally.not_applicable,
            tally.informational
        );
    }
    let worst_pair = outcome
        .worst
        .as_ref()
        .map_or_else(|| "NONE".to_string(), |(id, _)| id.clone());
    let worst_ratio = outcome.worst.as_ref().map_or(0.0, |(_, r)| *r);
    println!(
        "acceptance_summary registered={} load_bearing={} applicable={} pass={} fail={} \
         missing={} unauthorized={} not_applicable={} informational={} concurrency_unmatched={} \
         address_face_unmatched={} worst_pair={worst_pair} worst_ratio={worst_ratio:.4} \
         full_acceptance={}",
        outcome.registered,
        outcome.load_bearing,
        outcome.applicable,
        outcome.passed,
        outcome.failing.len(),
        outcome.missing.len(),
        outcome.unauthorized.len(),
        outcome.not_applicable,
        outcome.informational,
        outcome.concurrency_unmatched,
        outcome.address_face_unmatched,
        outcome.full_acceptance
    );
    if let Some(first) = outcome.failing.first() {
        println!("acceptance_first_failure={first}");
    }
    if let Some(first) = outcome.missing.first() {
        println!("acceptance_first_missing={first}");
    }
    if let Some(first) = outcome.unauthorized.first() {
        println!("acceptance_first_unauthorized={first}");
    }
}

/// The pair-summary table: one row per registered `(cell, incumbent)` pair. `concurrency` is
#[allow(
    clippy::too_many_lines,
    reason = "one function states the whole fixed-width pair-summary table"
)]
pub(crate) fn print_pair_table(campaign: &Campaign, outcome: &AcceptanceOutcome) {
    let headers = [
        "family",
        "cell",
        "sel_impl",
        "sel_workers",
        "inc_impl",
        "inc_workers",
        "sel_median_ns",
        "inc_median_ns",
        "ratio",
        "ratio_interval",
        "concurrency",
        "address_face",
        "selected_address",
        "sample_basis",
        "status",
    ];
    let mut rows: Vec<[String; 15]> = Vec::with_capacity(campaign.pairs.len());
    for (i, pair) in campaign.pairs.iter().enumerate() {
        let status = outcome
            .pair_status
            .get(i)
            .copied()
            .unwrap_or(PairStatus::Fail);
        let concurrency = pair_concurrency(pair.sel_workers, pair.inc_workers);
        let sel_median = if pair.selected_samples.is_empty() {
            "-".to_string()
        } else {
            median(&pair.selected_samples).to_string()
        };
        let inc_median = if pair.incumbent_samples.is_empty() {
            "-".to_string()
        } else {
            median(&pair.incumbent_samples).to_string()
        };
        let ratio_text = if pair.selected_samples.is_empty() || pair.incumbent_samples.is_empty() {
            "-".to_string()
        } else {
            format!(
                "{:.4}",
                ratio(
                    median(&pair.selected_samples),
                    median(&pair.incumbent_samples)
                )
            )
        };
        let family = if pair.cell_id.starts_with("spatial/") {
            "spatial"
        } else {
            "current-strata"
        };
        rows.push([
            family.to_string(),
            pair.cell_id.clone(),
            pair.sel_impl.to_string(),
            pair.sel_workers.to_string(),
            pair.inc_impl.to_string(),
            pair.inc_workers
                .map_or_else(|| "-".to_string(), |w| w.to_string()),
            sel_median,
            inc_median,
            ratio_text,
            interval_text(&pair.campaign_ratios),
            concurrency.name().to_string(),
            pair.address_face.name().to_string(),
            pair.selected_address.to_string(),
            pair.sample_basis.to_string(),
            status.name().to_string(),
        ]);
    }
    let mut widths = headers.map(str::len);
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row.iter()) {
            *width = (*width).max(cell.len());
        }
    }
    println!(
        "pair_table rows={} columns=family|cell|sel_impl|sel_workers|inc_impl|inc_workers|\
         sel_median_ns|inc_median_ns|ratio|ratio_interval|concurrency|address_face|selected_address|sample_basis|status \
         concurrency=matched|MISMATCH|n/a address_face=matched|unmatched|n/a",
        rows.len()
    );
    print_row(&headers.map(str::to_string), &widths);
    let rule: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    print_row(&rule, &widths);
    for row in &rows {
        print_row(row, &widths);
    }
}

fn print_row(row: &[String], widths: &[usize]) {
    let mut line = String::new();
    for (i, (cell, width)) in row.iter().zip(widths.iter()).enumerate() {
        if i > 0 {
            line.push_str(" | ");
        }
        line.push_str(cell);
        line.push_str(&" ".repeat(width.saturating_sub(cell.len())));
    }
    println!("{line}");
}

fn interval_text(ratios: &[f64]) -> String {
    if ratios.is_empty() {
        return "-".to_string();
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &r in ratios {
        if r < lo {
            lo = r;
        }
        if r > hi {
            hi = r;
        }
    }
    format!("[{lo:.4},{hi:.4}]")
}

pub(crate) fn print_headline(outcome: &AcceptanceOutcome) {
    println!(
        "HEADLINE registered={} load_bearing={} applicable={} pass={} fail={} missing={} \
         concurrency_unmatched={} address_face_unmatched={} full_acceptance={}",
        outcome.registered,
        outcome.load_bearing,
        outcome.applicable,
        outcome.passed,
        outcome.failing.len(),
        outcome.missing.len(),
        outcome.concurrency_unmatched,
        outcome.address_face_unmatched,
        outcome.full_acceptance
    );
}

/// Print, per registered incumbent, the observed `selected/incumbent` ratio interval across every
/// cell whose pair was measured and whose oracle digest agreed. A missing adapter has no pairs.
fn print_ratio_summary(campaign: &Campaign) {
    for inc in INCUMBENTS {
        let mut pairs = 0usize;
        let mut digest_fail = 0usize;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for pair in campaign.pairs.iter().filter(|p| p.incumbent_id == inc.id) {
            if let AdapterState::Built = pair.adapter {
                if pair.oracle_digest != pair.expected_oracle_digest {
                    digest_fail += 1;
                    continue;
                }
            }
            if pair.campaign_ratios.is_empty() {
                continue;
            }
            pairs += 1;
            for &r in &pair.campaign_ratios {
                if r < lo {
                    lo = r;
                }
                if r > hi {
                    hi = r;
                }
            }
        }
        if pairs == 0 {
            println!(
                "acceptance_ratio incumbent={} class={} pairs=0 digest_fail={digest_fail} \
                 r_min=NAN r_max=NAN",
                inc.id,
                inc.class.name()
            );
        } else {
            println!(
                "acceptance_ratio incumbent={} class={} pairs={pairs} digest_fail={digest_fail} \
                 r_min={lo:.4} r_max={hi:.4}",
                inc.id,
                inc.class.name()
            );
        }
    }
}

pub(crate) const CONTROLS: [&str; 8] = [
    "acceptance-failing-non-hashtrinity-pair",
    "acceptance-omitted-incumbent-pair",
    "acceptance-unauthorized-not-applicable",
    "acceptance-crossing-ratio-interval",
    "acceptance-wrong-oracle-digest",
    "acceptance-serial-adapter-wrong-result",
    "acceptance-concurrency-mismatch",
    "acceptance-address-face-mismatch",
];

/// Whether `name` is an acceptance control.
pub(crate) fn is_control(name: &str) -> bool {
    CONTROLS.contains(&name)
}

fn synthetic_campaign() -> Campaign {
    let cells = vec![
        "synthetic-cell-a".to_string(),
        "synthetic-cell-b".to_string(),
    ];
    let mut pairs = Vec::with_capacity(cells.len() * INCUMBENTS.len());
    for cell in &cells {
        for inc in INCUMBENTS {
            pairs.push(PairOutcome {
                cell_id: cell.clone(),
                incumbent_id: inc.id,
                class: inc.class,
                adapter: AdapterState::Built,
                applicability: Applicability::Applicable("synthetic control pair"),
                sel_impl: SELECTED_IMPL,
                sel_workers: SELECTED_WORKERS,
                sample_basis: "serial",
                inc_impl: inc.id,
                inc_workers: Some(1),
                address_face: serial_address_face(inc.id),
                selected_address: "raw-word;direct_word=Some",
                oracle_digest: 0xABCD,
                expected_oracle_digest: 0xABCD,
                selected_samples: vec![100, 101, 102],
                incumbent_samples: vec![100, 101, 102],
                campaign_ratios: vec![0.5, 0.5, 0.5],
            });
        }
    }
    Campaign { cells, pairs }
}

fn run_crossing_control() -> i32 {
    let name = "acceptance-crossing-ratio-interval";
    let mut crossing = synthetic_campaign();
    let Some(pair) = crossing
        .pairs
        .iter_mut()
        .find(|p| p.class == PairClass::LoadBearing)
    else {
        return 1;
    };
    pair.campaign_ratios = vec![0.8, 1.3, 1.1];
    let mut invalid = synthetic_campaign();
    let Some(pair) = invalid
        .pairs
        .iter_mut()
        .find(|p| p.class == PairClass::LoadBearing)
    else {
        return 1;
    };
    pair.campaign_ratios = vec![f64::NAN, f64::NAN];
    let crossing_rejected = !evaluate(&crossing, false).failing.is_empty();
    let invalid_rejected = !evaluate(&invalid, false).failing.is_empty();
    if crossing_rejected && invalid_rejected {
        eprintln!(
            "performance-control {name} rejected: crossing and invalid ratio intervals rejected"
        );
        return 2;
    }
    eprintln!(
        "performance-control {name} NOT rejected: crossing_rejected={crossing_rejected} \
         invalid_rejected={invalid_rejected}"
    );
    1
}

fn run_concurrency_control() -> i32 {
    let name = "acceptance-concurrency-mismatch";
    let mut campaign = synthetic_campaign();
    let Some(pair) = campaign
        .pairs
        .iter_mut()
        .find(|p| p.class == PairClass::LoadBearing)
    else {
        return 1;
    };
    pair.sel_workers = 16;
    pair.inc_workers = Some(24);
    let computed = pair_concurrency(pair.sel_workers, pair.inc_workers);
    let outcome = evaluate(&campaign, false);
    if computed != Concurrency::Mismatch {
        eprintln!(
            "performance-control {name} NOT rejected: computed concurrency {} != MISMATCH",
            computed.name()
        );
        return 1;
    }
    if outcome.concurrency_unmatched == 0
        || !outcome
            .failing
            .iter()
            .any(|f| f.contains("concurrency=MISMATCH"))
    {
        eprintln!(
            "performance-control {name} NOT rejected: strict acceptance accepted the \
              W=16/W=24 pair (concurrency_unmatched={} failing={})",
            outcome.concurrency_unmatched,
            outcome.failing.len()
        );
        return 1;
    }
    eprintln!(
        "performance-control {name} rejected: W=16/W=24 pair is MISMATCH and rejected \
         (concurrency_unmatched={} failing={})",
        outcome.concurrency_unmatched,
        outcome.failing.len()
    );
    2
}

fn run_address_face_control() -> i32 {
    let name = "acceptance-address-face-mismatch";
    let mut not_comparable = synthetic_campaign();
    let Some(na_pair) = not_comparable
        .pairs
        .iter_mut()
        .find(|p| p.class == PairClass::LoadBearing)
    else {
        return 1;
    };
    na_pair.address_face = AddressFace::NotApplicable;
    na_pair.selected_address = "n/a";
    if !evaluate(&not_comparable, false).passed() {
        eprintln!("performance-control {name} NOT rejected: n/a address face was gated");
        return 1;
    }
    let mut campaign = synthetic_campaign();
    let Some(pair) = campaign
        .pairs
        .iter_mut()
        .find(|p| p.class == PairClass::LoadBearing)
    else {
        return 1;
    };
    pair.address_face = AddressFace::Unmatched;
    pair.selected_address = "control-wrong-coordinate";
    let outcome = evaluate(&campaign, false);
    if outcome.address_face_unmatched == 0
        || !outcome
            .failing
            .iter()
            .any(|f| f.contains("address_face=unmatched"))
    {
        eprintln!(
            "performance-control {name} NOT rejected: strict acceptance accepted the comparable \
             address mismatch (address_face_unmatched={} failing={})",
            outcome.address_face_unmatched,
            outcome.failing.len()
        );
        return 1;
    }
    eprintln!(
        "performance-control {name} rejected: n/a non-gating and comparable address mismatch rejected \
         (address_face_unmatched={} failing={})",
        outcome.address_face_unmatched,
        outcome.failing.len()
    );
    2
}

#[allow(
    clippy::too_many_lines,
    reason = "one function dispatches the controls"
)]
pub(crate) fn run_control_child(name: &str) -> i32 {
    let baseline = synthetic_campaign();
    if !evaluate(&baseline, false).passed() {
        eprintln!(
            "performance-control {name} NOT rejected: the baseline synthetic campaign did not pass"
        );
        return 1;
    }
    if name == "acceptance-crossing-ratio-interval" {
        return run_crossing_control();
    }
    if name == "acceptance-concurrency-mismatch" {
        return run_concurrency_control();
    }
    if name == "acceptance-address-face-mismatch" {
        return run_address_face_control();
    }
    let mut campaign = synthetic_campaign();
    let applied =
        match name {
            "acceptance-failing-non-hashtrinity-pair" => {
                let Some(pair) = campaign.pairs.iter_mut().find(|p| {
                    p.class == PairClass::LoadBearing && p.incumbent_id != HASHTRINITY_TABLE
                }) else {
                    return 1;
                };
                pair.campaign_ratios = vec![1.2, 1.2, 1.2];
                true
            }
            "acceptance-omitted-incumbent-pair" => {
                let before = campaign.pairs.len();
                campaign.pairs.retain(|p| {
                    !(p.cell_id == "synthetic-cell-a" && p.incumbent_id == "hashbrown_foldhash")
                });
                campaign.pairs.len() != before
            }
            "acceptance-unauthorized-not-applicable" => {
                let Some(pair) = campaign
                    .pairs
                    .iter_mut()
                    .find(|p| p.class == PairClass::LoadBearing)
                else {
                    return 1;
                };
                pair.applicability =
                    Applicability::UnauthorizedNotApplicable("control: no authorizing reason");
                true
            }
            "acceptance-wrong-oracle-digest" => {
                let Some(pair) = campaign
                    .pairs
                    .iter_mut()
                    .find(|p| p.class == PairClass::LoadBearing)
                else {
                    return 1;
                };
                pair.oracle_digest = pair.oracle_digest.wrapping_add(1);
                true
            }
            "acceptance-serial-adapter-wrong-result" => {
                // A built serial adapter whose computed digest disagrees with the cell oracle is a
                // corrupted adapter result and must reject through the same wrong-oracle channel.
                let Some(pair) = campaign
                    .pairs
                    .iter_mut()
                    .find(|p| p.incumbent_id == "hashbrown_foldhash")
                else {
                    return 1;
                };
                pair.oracle_digest = pair.oracle_digest.wrapping_add(1);
                true
            }
            _ => {
                eprintln!("unknown acceptance control: {name}");
                return 2;
            }
        };
    if !applied {
        eprintln!("performance-control {name} could not be applied to the synthetic campaign");
        return 1;
    }
    let outcome = evaluate(&campaign, false);
    if outcome.passed() {
        eprintln!(
            "performance-control {name} NOT rejected: validator accepted the corrupted campaign"
        );
        return 1;
    }
    let intended = match name {
        "acceptance-failing-non-hashtrinity-pair"
        | "acceptance-wrong-oracle-digest"
        | "acceptance-serial-adapter-wrong-result" => !outcome.failing.is_empty(),
        "acceptance-omitted-incumbent-pair" => !outcome.missing.is_empty(),
        "acceptance-unauthorized-not-applicable" => !outcome.unauthorized.is_empty(),
        _ => false,
    };
    if !intended {
        eprintln!(
            "performance-control {name} rejected for the wrong channel: failing={} missing={} \
             unauthorized={}",
            outcome.failing.len(),
            outcome.missing.len(),
            outcome.unauthorized.len()
        );
        return 1;
    }
    eprintln!(
        "performance-control {name} rejected: failing={} missing={} unauthorized={}",
        outcome.failing.len(),
        outcome.missing.len(),
        outcome.unauthorized.len()
    );
    2
}

fn run_control(name: &str, profile: Profile) -> Result<i32, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let output = Command::new(exe)
        .args(["--performance-control", name, "--profile", profile.name()])
        .output()
        .map_err(|e| format!("spawn control {name}: {e}"))?;
    if output.status.success() {
        return Err(format!(
            "control {name} exited 0; corrupted campaign was accepted"
        ));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let marker = format!("{name} rejected:");
    if !stderr.contains(&marker) {
        return Err(format!(
            "control {name} failed for an unexpected reason: {}",
            stderr.trim()
        ));
    }
    Ok(output.status.code().unwrap_or(-1))
}

pub(crate) fn run_accept(profile: Profile, controls: bool, json: bool) -> i32 {
    // Phase 1: the existing measurement, registration enforcement and diagnostics, unchanged except
    // that the acceptance path prints its own table, so the phase-1 dump is silent.
    let diagnostic =
        performance::run_performance(profile, false, false, performance::Output::Silent);
    if diagnostic != 0 {
        return diagnostic;
    }
    // Phase 2: the registered per-pair acceptance.
    let strata = strata::run_strata(profile);
    let registered = match matrix::load_registered_cells() {
        Ok(cells) => cells,
        Err(e) => {
            eprintln!("acceptance registration load failed: {e}");
            return 1;
        }
    };
    if let Err(e) = matrix::validate_registration(&strata, &registered) {
        eprintln!("acceptance registration validation failed: {e}");
        return 1;
    }
    if let Err(e) = strata::validate_strata(&strata) {
        eprintln!("acceptance strata validation failed: {e}");
        return 1;
    }
    let spatial = performance::spatial::measure(&strata);
    if json {
        performance::spatial::print(&spatial);
    }
    if let Err(e) = performance::spatial::validate(&spatial) {
        eprintln!("acceptance spatial validation failed: {e}");
        return 1;
    }
    let campaign = build_campaign(&strata, &registered, &spatial);
    println!(
        "acceptance_phase=per-pair profile={} registered_cells={} spatial_cells={} json={json}",
        profile.name(),
        strata.cells.len(),
        spatial.cells.len()
    );
    if json {
        print_registration();
        print_ratio_summary(&campaign);
    }
    let outcome = evaluate(&campaign, profile == Profile::Full);
    if json {
        print_acceptance(&outcome);
    } else {
        print_pair_table(&campaign, &outcome);
    }
    print_headline(&outcome);
    if controls {
        for name in performance::CONTROLS {
            match run_control(name, profile) {
                Ok(code) => println!("control={name} observed=red child_exit={code}"),
                Err(e) => {
                    eprintln!("control {name} failed: {e}");
                    return 1;
                }
            }
        }
        for name in CONTROLS {
            match run_control(name, profile) {
                Ok(code) => println!("control={name} observed=red child_exit={code}"),
                Err(e) => {
                    eprintln!("control {name} failed: {e}");
                    return 1;
                }
            }
        }
    }
    if outcome.passed() {
        println!(
            "strict_acceptance=pass applicable={} failing=0 missing=0 concurrency_unmatched={} \
             full_acceptance={}",
            outcome.applicable, outcome.concurrency_unmatched, outcome.full_acceptance
        );
        0
    } else {
        eprintln!(
            "acceptance failed: applicable={} failing={} missing={} unauthorized={} \
             concurrency_unmatched={} full_acceptance=false",
            outcome.applicable,
            outcome.failing.len(),
            outcome.missing.len(),
            outcome.unauthorized.len(),
            outcome.concurrency_unmatched
        );
        1
    }
}

/// Registry-integrity tests. These read the compiled registry and the pure classification helpers;
/// they run no benchmark and no adapter. A required incumbent that is omitted, demoted to
/// informational, given an applicability that rejects, or mis-classified fails here, so the
/// acceptance report cannot silently drop a required comparison.
#[cfg(test)]
mod registry_tests {
    use super::{
        evaluate, is_spatial_incumbent, pair_concurrency, serial_address_face,
        spatial_address_face, spatial_incumbent_workers, synthetic_campaign, AdapterState,
        AddressFace, Applicability, Concurrency, PairClass, INCUMBENTS, INFORMATIONAL_COMPARATORS,
        RATIO_LIMIT, REQUIRED_LOAD_BEARING, SELECTED_WORKERS,
    };

    /// Every required incumbent is registered exactly once, load-bearing, built, and carries an
    /// applicability that is not `UnauthorizedNotApplicable`. An omission or demotion fails here.
    #[test]
    fn required_incumbents_are_present_load_bearing_and_built() {
        for id in REQUIRED_LOAD_BEARING {
            let mut matches = INCUMBENTS.iter().filter(|inc| inc.id == id);
            let Some(inc) = matches.next() else {
                panic!("required incumbent {id} is absent from the registry");
            };
            assert!(
                matches.next().is_none(),
                "required incumbent {id} is registered more than once"
            );
            assert_eq!(
                inc.class,
                PairClass::LoadBearing,
                "required incumbent {id} is not load-bearing"
            );
            assert_eq!(
                inc.adapter,
                AdapterState::Built,
                "required incumbent {id} has no built adapter"
            );
            match inc.applicability {
                Applicability::Applicable(reason)
                | Applicability::AuthorizedNotApplicable(reason) => {
                    assert!(
                        !reason.is_empty(),
                        "required incumbent {id} has an empty applicability reason"
                    );
                }
                Applicability::UnauthorizedNotApplicable(_) => {
                    panic!("required incumbent {id} is unauthorized-not-applicable");
                }
            }
        }
    }

    /// Every registered incumbent is either a required load-bearing comparator or a named
    /// informational comparator carrying its stated reason. An unclassified incumbent fails here.
    #[test]
    fn every_registered_incumbent_is_classified() {
        for inc in INCUMBENTS {
            let required = REQUIRED_LOAD_BEARING.contains(&inc.id);
            let informational = INFORMATIONAL_COMPARATORS
                .iter()
                .find(|(id, _)| *id == inc.id);
            match (required, informational) {
                (true, None) => assert_eq!(
                    inc.class,
                    PairClass::LoadBearing,
                    "required incumbent {} is not load-bearing",
                    inc.id
                ),
                (false, Some((id, reason))) => {
                    assert_eq!(*id, inc.id);
                    assert_eq!(
                        inc.class,
                        PairClass::Informational,
                        "informational comparator {} is not informational",
                        inc.id
                    );
                    assert_eq!(
                        inc.applicability.reason(),
                        *reason,
                        "informational comparator {} reason drifted",
                        inc.id
                    );
                }
                (true, Some(_)) => {
                    panic!("incumbent {} is both required and informational", inc.id);
                }
                (false, None) => {
                    panic!(
                        "incumbent {} is neither required load-bearing nor informational",
                        inc.id
                    );
                }
            }
        }
        for (id, _) in INFORMATIONAL_COMPARATORS {
            assert!(
                INCUMBENTS.iter().any(|inc| inc.id == id),
                "informational comparator {id} is absent from the registry"
            );
        }
    }

    /// The owner threshold stays `selected/incumbent <= 1.0`. A threshold above one fails here.
    #[test]
    fn acceptance_threshold_is_one() {
        assert!(
            (RATIO_LIMIT - 1.0).abs() < f64::EPSILON,
            "the acceptance threshold must stay 1.0, found {RATIO_LIMIT}"
        );
    }

    /// The required serial incumbents run at width one on the raw-coordinate face; the required
    /// spatial incumbents carry the address face their routing mechanism uses. A mis-classified
    /// required incumbent fails here.
    #[test]
    fn required_incumbent_classification_is_correct() {
        for id in REQUIRED_LOAD_BEARING
            .iter()
            .filter(|&&id| !is_spatial_incumbent(id))
        {
            assert_eq!(
                serial_address_face(id),
                AddressFace::Matched,
                "serial incumbent {id} must be address-face matched"
            );
            assert_eq!(
                pair_concurrency(SELECTED_WORKERS, Some(1)),
                Concurrency::Matched,
                "serial incumbent {id} must be concurrency matched at width one"
            );
        }
        assert_eq!(spatial_address_face("foldhash_par16"), AddressFace::Matched);
        assert_eq!(spatial_address_face("fxhash_par16"), AddressFace::Matched);
        assert_eq!(
            spatial_address_face("hashtrinity_count_par"),
            AddressFace::Unmatched
        );
        assert_eq!(
            spatial_address_face("hashtrinity_get_each"),
            AddressFace::Unmatched
        );
        assert_eq!(
            spatial_address_face("hashbrown_rayon"),
            AddressFace::Unmatched
        );
        // The registered applicability rule follows the face: a spatial incumbent is authorized
        // not applicable on the serial cells, a serial incumbent is applicable there.
        for inc in INCUMBENTS {
            if is_spatial_incumbent(inc.id) {
                assert!(
                    matches!(inc.applicability, Applicability::AuthorizedNotApplicable(_)),
                    "spatial incumbent {} must register an authorized-not-applicable rule",
                    inc.id
                );
            } else {
                assert!(
                    matches!(inc.applicability, Applicability::Applicable(_)),
                    "serial incumbent {} must register an applicable rule",
                    inc.id
                );
            }
        }
    }

    /// `hashtrinity_get_each` is timed through the combined `count_par` build plus grouped probe,
    /// so its build width is the hashtrinity pool, not the reported single-core probe width. A
    /// regression that trusts the reported `workers = 1` fails here.
    #[test]
    fn get_each_combined_arm_is_concurrency_unmatched() {
        let pool = hashtrinity::par::Pool::global().width().max(1);
        for reported in [1usize, 2, 7, pool, pool + 3] {
            assert_eq!(
                spatial_incumbent_workers("hashtrinity_get_each", reported),
                pool,
                "get_each must carry the build pool width, not the reported probe width {reported}"
            );
        }
        assert_eq!(
            spatial_incumbent_workers("hashtrinity_get_each", 1),
            spatial_incumbent_workers("hashtrinity_count_par", pool),
            "the combined get_each and count_par arms run the same build width"
        );
        if pool > 1 {
            assert_eq!(
                pair_concurrency(SELECTED_WORKERS, Some(pool)),
                Concurrency::Mismatch,
                "a combined-arm incumbent against the single-core selected run must be a mismatch"
            );
        }
    }

    /// An omitted required pair is reported missing and rejects the campaign. A registry change
    /// that drops a required `(cell, incumbent)` pair fails here.
    #[test]
    fn omitted_required_pair_is_reported_missing() {
        let mut campaign = synthetic_campaign();
        let before = campaign.pairs.len();
        campaign.pairs.retain(|pair| {
            !(pair.cell_id == "synthetic-cell-a" && pair.incumbent_id == "xxhash_xxh3")
        });
        assert!(campaign.pairs.len() < before);
        let outcome = evaluate(&campaign, false);
        assert!(
            outcome
                .missing
                .iter()
                .any(|entry| entry.contains("xxhash_xxh3")),
            "omitted required pair must be reported missing: {:?}",
            outcome.missing
        );
        assert!(!outcome.passed());
    }
}
