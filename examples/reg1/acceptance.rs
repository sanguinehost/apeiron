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

mod report;
use report::{print_acceptance, print_ratio_summary, print_registration};
pub(crate) use report::{print_headline, print_pair_table};

pub(crate) const RATIO_LIMIT: f64 = 1.0;

/// The environment variable that selects the acceptance ratio target for one invocation.
///
/// Unset is the registered gate: `RATIO_LIMIT` on every applicable pair, unchanged. `half` is the
/// owner's 0.5 target: `0.50` on every applicable hashtrinity pair and `RATIO_LIMIT` on every other
/// pair. A finite value in `(0, RATIO_LIMIT]` is that bound on the hashtrinity pairs. Anything else
/// refuses the invocation with [`USAGE_EXIT`], so a misspelled variable cannot silently run the
/// registered gate in the reader's place: a refusal is a usage error, distinct from the acceptance
/// failure `1`.
pub(crate) const ENV_TARGET_RATIO: &str = "REG1_TARGET_RATIO";

/// The `half` value of [`ENV_TARGET_RATIO`]: the owner's `selected/incumbent <= 0.5` target on every
/// applicable hashtrinity pair.
pub(crate) const TARGET_HALF: f64 = 0.5;

/// The incumbents the 0.5 target is scoped to: every registered arm of the pinned hashtrinity
/// baseline `4cc14471`.
///
/// Closed over the registry rather than derived from a name prefix at the decision point, so a newly
/// registered hashtrinity arm is a [`target_scope_is_closed_over_the_registry`] failure rather than a
/// silently widened exemption. The list is the scope; `TARGET_SCOPED_INCUMBENT_PREFIX` is the
/// property that pins it to the registry.
pub(crate) const TARGET_SCOPED_INCUMBENTS: [&str; 3] = [
    "hashtrinity_table",
    "hashtrinity_count_par",
    "hashtrinity_get_each",
];

/// The id prefix that makes an incumbent an arm of the hashtrinity baseline. Compiled only for the
/// registry-closure test that pins [`TARGET_SCOPED_INCUMBENTS`] to it; never read at the decision
/// point, which reads the list itself.
#[cfg(test)]
pub(crate) const TARGET_SCOPED_INCUMBENT_PREFIX: &str = "hashtrinity_";

/// The exit status of a refused ratio-target selection: a usage error, as in [`main::run`]'s `64`,
/// and distinct from the acceptance failure `1` and the measurement refusal `3`.
pub(crate) const USAGE_EXIT: i32 = 64;

/// The per-family `selected/incumbent` bound one acceptance invocation applies.
///
/// Two bounds, not one: the 0.5 target is scoped to the hashtrinity arms, and every other
/// incumbent keeps the registered [`RATIO_LIMIT`]. Both are decided on the same pooled campaign
/// interval max as the registered gate, so neither bound relaxes or tightens the uncertainty rule,
/// the oracle-identity check, the applicability channels or the concurrency and address-face checks;
/// those all run before the ratio and are unchanged.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct RatioTarget {
    /// The bound on every applicable arm of [`TARGET_SCOPED_INCUMBENTS`].
    scoped: f64,
    /// The bound on every other applicable pair.
    other: f64,
}

impl RatioTarget {
    /// The registered gate: [`RATIO_LIMIT`] on every applicable pair. This is the default of every
    /// path that does not name a target, so the `1.0` gate and its recorded history are unaffected by
    /// the existence of [`RatioTarget::half`].
    pub(crate) const fn registered() -> Self {
        Self {
            scoped: RATIO_LIMIT,
            other: RATIO_LIMIT,
        }
    }

    /// The owner's 0.5 target: [`TARGET_HALF`] on every applicable hashtrinity pair,
    /// [`RATIO_LIMIT`] on every other pair.
    pub(crate) const fn half() -> Self {
        Self {
            scoped: TARGET_HALF,
            other: RATIO_LIMIT,
        }
    }

    /// Whether this target is the registered gate, whose report lines are printed unchanged.
    pub(crate) fn is_registered(&self) -> bool {
        *self == Self::registered()
    }

    /// The target's report name.
    pub(crate) fn name(&self) -> &'static str {
        if *self == Self::registered() {
            "registered-1.00"
        } else if *self == Self::half() {
            "hashtrinity-half"
        } else {
            "hashtrinity-custom"
        }
    }

    /// The bound this target applies to `incumbent_id`.
    ///
    /// Total: an incumbent outside [`TARGET_SCOPED_INCUMBENTS`] takes `other`, and an id the
    /// registry does not carry is already rejected by [`evaluate`] as an unknown incumbent before
    /// this is reached.
    pub(crate) fn limit_for(&self, incumbent_id: &str) -> f64 {
        if TARGET_SCOPED_INCUMBENTS.contains(&incumbent_id) {
            self.scoped
        } else {
            self.other
        }
    }

    /// Whether this target tightens any pair at all, i.e. whether it is worth reporting separately.
    pub(crate) fn is_stricter(&self) -> bool {
        self.scoped < RATIO_LIMIT
    }

    /// Resolve the invocation's target from [`ENV_TARGET_RATIO`].
    ///
    /// # Errors
    /// Returns a description when the variable is set to a value that is not `half` and not a finite
    /// ratio in `(0, RATIO_LIMIT]`. Unset is the registered target, not an error: the `1.0` gate is
    /// the default and stays the gate.
    pub(crate) fn from_env() -> Result<Self, String> {
        match std::env::var(ENV_TARGET_RATIO) {
            Ok(raw) => Self::parse(Some(&raw)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{ENV_TARGET_RATIO} is not valid unicode"))
            }
        }
    }

    /// The selection rule behind [`RatioTarget::from_env`], as a pure function of the raw value, so
    /// the parse is testable without mutating the process environment.
    ///
    /// `None` is the unset variable, which is the registered gate. `Some("half")` is the owner's
    /// target. Any other value is parsed as the scoped bound and must be finite and in
    /// `(0, RATIO_LIMIT]`; the `other` bound stays [`RATIO_LIMIT`], so no selection can loosen the
    /// registered gate for a pair the target does not scope.
    ///
    /// # Errors
    /// Returns a description of why `raw` does not name a target.
    pub(crate) fn parse(raw: Option<&str>) -> Result<Self, String> {
        let Some(raw) = raw else {
            return Ok(Self::registered());
        };
        let trimmed = raw.trim();
        if trimmed == "half" {
            return Ok(Self::half());
        }
        let Ok(scoped) = trimmed.parse::<f64>() else {
            return Err(format!(
                "{ENV_TARGET_RATIO}={raw:?} is neither \"half\" nor a ratio in (0, {RATIO_LIMIT}]"
            ));
        };
        if !scoped.is_finite() || scoped <= 0.0 || scoped > RATIO_LIMIT {
            return Err(format!(
                "{ENV_TARGET_RATIO}={raw:?} is outside (0, {RATIO_LIMIT}]"
            ));
        }
        Ok(Self {
            scoped,
            other: RATIO_LIMIT,
        })
    }

    /// The `acceptance_target` line: the bound this invocation applied, and the registered bound it
    /// replaced.
    pub(crate) fn print_target(&self) {
        println!(
            "acceptance_target name={} scoped_limit={:.2} other_limit={:.2} \
             scoped_incumbents={} registered_limit={RATIO_LIMIT:.2} source={ENV_TARGET_RATIO} \
             decision=pooled_campaign_interval_max",
            self.name(),
            self.scoped,
            self.other,
            TARGET_SCOPED_INCUMBENTS.join(","),
        );
    }
}

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

/// Whether an incumbent's `FreqMap` can distinguish two keys, from its own surface.
///
/// hashtrinity's `FreqLine<V>` holds `ctrl` plus `slots: Box<[(u64, MaybeUninit<V>)]>` and no key
/// column (`4cc14471` `src/freq.rs:271-286`), and `FreqMap::add` is
/// `insert_with(freq(k), v, |new, old| new + old)` (`src/freq.rs:920-923`). Identity is therefore
/// the address `freq(k) = k.coord().min(EMPTY - 2)` (`src/freq.rs:737-739`) and nothing else, so the
/// arm is exact over a key population exactly when that fold is injective on it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AddressIdentity {
    /// The map stores the whole key and compares it on lookup: identity is the key.
    WholeKeyStored,
    /// The map stores no key: identity is the address fold, and two keys sharing an address are one
    /// entry whose `add` sums them.
    AddressFoldOnly,
}

fn spatial_identity_mechanism(id: &str) -> Option<AddressIdentity> {
    match id {
        // `FreqMap` holds no key column; identity is `coord`.
        "hashtrinity_count_par" | "hashtrinity_get_each" => Some(AddressIdentity::AddressFoldOnly),
        // `hashbrown::HashMap<TupleKey, i64>` and `sharded_16::<TupleKey, S>` both store and compare
        // the whole key.
        "hashbrown_rayon" | "foldhash_par16" | "fxhash_par16" => {
            Some(AddressIdentity::WholeKeyStored)
        }
        _ => None,
    }
}

/// The address-face verdict for a spatial pair, with the reason the mechanism produced.
///
/// Decided from the arm's registered identity surface, the cell's key face and the cell's own
/// observed [`CellIdentity`] — never from a per-arm list of labels.
///
/// * A stored whole key is `Matched` on both faces: identity is the key itself, so the incumbent
///   preserves whole-key identity over the whole population.
/// * hashtrinity's address fold on the **packed** face is `Matched`: `Coord for u64` is the identity
///   (`4cc14471` `src/algebra.rs:18-23`) and the `min(EMPTY - 2)` clamp's only distortion is the
///   three-point top fiber, while `(i21,i21,i21)::packed()` places three 21-bit fields at bits 0, 21
///   and 42 (`src/coord.rs:461-463, 552`) and so leaves bit 63 clear. `CellIdentity::max_packed`
///   records the observed maximum, and the verdict requires `merged_keys == 0` on this cell.
/// * hashtrinity's address fold on the **tuple** face is `Unmatched` with the `no key store` reason:
///   `FreqLine` stores no key, so identity is the 93→64 fold `((a·PHI ^ b)·PHI ^ c)`
///   (`src/algebra.rs:358-363`) and distinct tuples that share a coordinate are one entry. Where this
///   cell's own population merges, the count of merged keys is named in the reason.
/// * A spatial incumbent with no registered identity surface is `NotApplicable`; the harness then
///   has no mechanism for the pair, which is not the same as a matched one.
fn spatial_address_face(
    id: &str,
    key_face: crate::performance::spatial::KeyFace,
    identity: &crate::performance::spatial::CellIdentity,
) -> (AddressFace, String) {
    use crate::performance::spatial::KeyFace;
    let Some(mechanism) = spatial_identity_mechanism(id) else {
        return (
            AddressFace::NotApplicable,
            format!("arm {id} has no registered address-identity mechanism"),
        );
    };
    if mechanism == AddressIdentity::WholeKeyStored {
        return (
            AddressFace::Matched,
            format!(
                "whole key stored and compared on lookup; cell keys={} addresses={} merged_keys={}",
                identity.distinct_keys, identity.distinct_addresses, identity.merged_keys
            ),
        );
    }
    match key_face {
        KeyFace::Packed => {
            if identity.merged_keys == 0 && identity.clamp_injective() {
                (
                    AddressFace::Matched,
                    format!(
                        "address fold is exact on the packed face: Coord(u64) is the identity and \
                         min(EMPTY-2) leaves max_packed=0x{:016x} below the three-point top fiber; \
                         merged_keys=0 of {} keys",
                        identity.max_packed, identity.distinct_keys
                    ),
                )
            } else {
                (
                    AddressFace::Unmatched,
                    format!(
                        "packed address fold merged {} of {} keys (max_packed=0x{:016x})",
                        identity.merged_keys, identity.distinct_keys, identity.max_packed
                    ),
                )
            }
        }
        KeyFace::Tuple => (
            AddressFace::Unmatched,
            format!(
                "no key store: FreqLine holds (u64, V) with no key column, so identity is the \
                 93->64 Coord fold and this cell merges {} of {} tuples into {} addresses",
                identity.merged_keys, identity.distinct_keys, identity.distinct_addresses
            ),
        ),
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
    /// The recorded reason the address-face mechanism produced this verdict.
    pub(crate) address_reason: String,
    pub(crate) selected_address: &'static str,
    pub(crate) oracle_digest: u64,
    pub(crate) expected_oracle_digest: u64,
    pub(crate) selected_samples: Vec<u64>,
    pub(crate) incumbent_samples: Vec<u64>,
    /// The per-campaign `selected/incumbent` ratios.
    pub(crate) campaign_ratios: Vec<f64>,
    /// The timed region this pair compares, in and out. Recorded on the pair so a ratio cannot be
    /// read without the phases it covers.
    pub(crate) timed_region: &'static str,
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
    /// Per pair and in `campaign.pairs` order: whether the pair reached the ratio decision, i.e.
    /// whether it was measured, structurally valid and comparable, so that
    /// `campaign_ratios` is a real interval for it. The repetition report reads this to decide
    /// which pairs a per-campaign pass tally may count; a pair that failed for a structural reason
    /// has no timing to spread and is excluded from every sweep identically.
    pub(crate) ratio_decided: Vec<bool>,
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
        address_reason: reason.to_string(),
        selected_address: "n/a",
        oracle_digest: 0,
        expected_oracle_digest: 0,
        selected_samples: Vec::new(),
        incumbent_samples: Vec::new(),
        campaign_ratios: Vec::new(),
        timed_region: "n/a: the pair carries no samples",
    }
}

/// The selected serial pair's timed region. The serial selected face admits one reverse-link slot
/// per fresh key and assembles no directory (`src/table.rs:1279-1311`), so nothing is excluded and
/// the charged region is the whole elapsed one: `insert_with` per entity, `get` per occupied cell,
/// then the order-sensitive fold.
pub(crate) const SERIAL_TIMED_REGION: &str = concat!(
    "IN=[the whole serial region: Table::insert_with per entity, Table::get per occupied cell, ",
    "query-order fold; no directory is assembled and nothing is excluded] OUT=[]"
);

#[allow(
    clippy::too_many_lines,
    reason = "one function states the whole per-cell/per-incumbent pair construction"
)]
pub(crate) fn build_campaign(
    strata: &StrataWitness,
    registered: &[RegisteredCell],
    spatial: &performance::spatial::SpatialWitness,
) -> Campaign {
    let t_serial = std::time::Instant::now();
    let serial = serial::measure(strata);
    eprintln!(
        "phase=accept.build_campaign.serial elapsed_s={:.3}",
        t_serial.elapsed().as_secs_f64()
    );
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
                address_reason: "the raw-coordinate serial face: the selected table and every \
                     serial incumbent key the same u64 coordinate word"
                    .to_string(),
                selected_address: serial_selected_address(&cell.cell_id),
                oracle_digest,
                expected_oracle_digest,
                selected_samples,
                incumbent_samples,
                campaign_ratios,
                timed_region: SERIAL_TIMED_REGION,
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
                    // The width the arm actually ran under, as its own reported policy states. Both
                    // hashtrinity arms report the width their timed region used, and the selected
                    // side reports the width its parallel APIs dispatched.
                    Some(outcome.policy.workers),
                    outcome.selected_policy.workers,
                    if outcome.selected_policy.workers == 1 {
                        "serial"
                    } else {
                        "parallel"
                    },
                ),
                None => (Vec::new(), Vec::new(), Vec::new(), 0, None, 0, "unmeasured"),
            };
            let (address_face, address_reason) =
                spatial_address_face(inc.id, cell.key_face, &cell.identity);
            pairs.push(PairOutcome {
                cell_id: cell.cell_id.clone(),
                incumbent_id: inc.id,
                class: inc.class,
                adapter: inc.adapter,
                applicability: Applicability::Applicable(
                    "spatial count workload on the registered 71942db scene and key face",
                ),
                sel_impl: SELECTED_IMPL,
                sel_workers,
                sample_basis,
                inc_impl: inc.id,
                inc_workers,
                address_face,
                address_reason,
                selected_address: "spread-word;direct_word=mixed",
                oracle_digest,
                expected_oracle_digest: cell.oracle_digest,
                selected_samples,
                incumbent_samples,
                campaign_ratios,
                timed_region: performance::spatial::SELECTED_TIMED_REGION,
            });
        }
    }
    Campaign { cells, pairs }
}

/// Print the spatial pairs' timed region with the numbers it covers, so the exclusion is read with
/// its magnitude rather than in the abstract. `|`, `"` and newlines are sanitised out of the text so
/// the report format is unaffected.
fn print_spatial_timed_regions(
    campaign: &Campaign,
    spatial: &performance::spatial::SpatialWitness,
) {
    for cell in &spatial.cells {
        for outcome in &cell.outcomes {
            let Some(pair) = campaign
                .pairs
                .iter()
                .find(|p| p.cell_id == cell.cell_id && p.incumbent_id == outcome.id)
            else {
                continue;
            };
            println!(
                "acceptance_timed_region cell={} incumbent={} sel_charged_ns={} sel_elapsed_ns={} \
                 excluded_ns={} rmax={:.4} region=\"{}\"",
                pair.cell_id,
                pair.incumbent_id,
                median(&outcome.selected_samples),
                median(&outcome.selected_elapsed_samples),
                median(&outcome.selected_excluded_samples),
                pair.campaign_ratios
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
                pair.timed_region.replace(['|', '\n', '"'], " ")
            );
        }
    }
}

/// Print the address-face mechanism's recorded reason for every spatial pair.
///
/// The pair table's `address_face` column carries only the verdict. The reason a mechanism produced
/// that verdict is what makes it checkable, so it is printed per pair rather than collapsed into the
/// `address_face_unmatched` counter. One line, `|` and newlines sanitised out of the reason so the
/// table format is unaffected.
fn print_address_faces(campaign: &Campaign) {
    for pair in &campaign.pairs {
        if !pair.cell_id.starts_with("spatial/") {
            continue;
        }
        println!(
            "acceptance_address_face cell={} incumbent={} face={} reason=\"{}\"",
            pair.cell_id,
            pair.incumbent_id,
            pair.address_face.name(),
            pair.address_reason.replace(['|', '\n'], " ")
        );
    }
}

fn update_worst(worst: &mut Option<(String, f64)>, cell: &str, incumbent: &str, r: f64) {
    if worst.as_ref().is_none_or(|(_, w)| r > *w) {
        *worst = Some((format!("{cell}/{incumbent}"), r));
    }
}

/// The registered gate's evaluation: [`RATIO_LIMIT`] on every applicable pair. Every caller that
/// does not name a target reaches the ratio decision through here, so the `1.0` gate is the
/// default of the whole harness and not a variant a caller has to choose.
pub(crate) fn evaluate(campaign: &Campaign, full_profile: bool) -> AcceptanceOutcome {
    evaluate_with(campaign, full_profile, RatioTarget::registered())
}

/// [`evaluate`] under an explicit [`RatioTarget`]: the same channels in the same order, with
/// `target.limit_for(pair.incumbent_id)` in place of [`RATIO_LIMIT`] at the ratio decision.
///
/// Nothing above the ratio decision is parameterised, which is what makes the target a bound change
/// and not a weaker harness: the oracle-identity check, the applicability channels, the missing
/// adapter, the empty/zero/non-finite sample checks, the concurrency match and the address face are
/// each evaluated exactly as before, and each still records its pair before this bound is consulted.
#[allow(
    clippy::too_many_lines,
    reason = "one function states the whole per-pair acceptance contract in report order"
)]
pub(crate) fn evaluate_with(
    campaign: &Campaign,
    full_profile: bool,
    target: RatioTarget,
) -> AcceptanceOutcome {
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
    let mut ratio_decided: Vec<bool> = Vec::with_capacity(campaign.pairs.len());
    // One recorder for both per-pair vectors, so a new rejection path cannot fill `pair_status`
    // and leave the repetition report's decided flags one entry short.
    let mut record = |status: PairStatus, decided: bool| {
        pair_status.push(status);
        ratio_decided.push(decided);
    };

    for pair in &campaign.pairs {
        let Some(&index) = tally_index.get(pair.incumbent_id) else {
            failing.push(format!("unknown incumbent {}", pair.incumbent_id));
            record(PairStatus::Fail, false);
            continue;
        };
        if !cell_set.contains(pair.cell_id.as_str()) {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) is not a registered cell",
                pair.cell_id, pair.incumbent_id
            ));
            record(PairStatus::Fail, false);
            continue;
        }
        if !seen.insert((pair.cell_id.as_str(), pair.incumbent_id)) {
            tallies[index].failed += 1;
            failing.push(format!(
                "duplicate pair ({}, {})",
                pair.cell_id, pair.incumbent_id
            ));
            record(PairStatus::Fail, false);
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
                    record(PairStatus::Fail, false);
                    continue;
                }
            }
            tallies[index].informational += 1;
            informational += 1;
            record(PairStatus::Informational, false);
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
                record(PairStatus::Unauthorized, false);
                continue;
            }
            Applicability::AuthorizedNotApplicable(_) => {
                tallies[index].not_applicable += 1;
                not_applicable += 1;
                record(PairStatus::NotApplicable, false);
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
            record(PairStatus::Missing, false);
            continue;
        }
        if pair.oracle_digest != pair.expected_oracle_digest {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) wrong oracle {:016x} != {:016x}",
                pair.cell_id, pair.incumbent_id, pair.oracle_digest, pair.expected_oracle_digest
            ));
            record(PairStatus::Fail, false);
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
            record(PairStatus::Fail, false);
            continue;
        }
        if pair.selected_samples.contains(&0) || pair.incumbent_samples.contains(&0) {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) zero-nanosecond sample",
                pair.cell_id, pair.incumbent_id
            ));
            record(PairStatus::Fail, false);
            continue;
        }
        if pair.campaign_ratios.iter().any(|r| !r.is_finite()) {
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) non-finite ratio interval",
                pair.cell_id, pair.incumbent_id
            ));
            record(PairStatus::Fail, false);
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
                pair.inc_workers.unwrap_or(0)
            ));
            record(PairStatus::Fail, false);
            continue;
        }
        if pair.address_face == AddressFace::Unmatched {
            address_face_unmatched += 1;
            tallies[index].failed += 1;
            failing.push(format!(
                "({}, {}) address_face=unmatched selected_address={}",
                pair.cell_id, pair.incumbent_id, pair.selected_address
            ));
            record(PairStatus::Fail, false);
            continue;
        }
        let r_max = pair
            .campaign_ratios
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let limit = target.limit_for(pair.incumbent_id);
        if r_max > limit {
            tallies[index].failed += 1;
            update_worst(&mut worst, &pair.cell_id, pair.incumbent_id, r_max);
            failing.push(format!(
                "({}, {}) rmax={r_max:.4} > {limit:.2}",
                pair.cell_id, pair.incumbent_id
            ));
            record(PairStatus::Fail, true);
            continue;
        }
        tallies[index].passed += 1;
        passed += 1;
        record(PairStatus::Pass, true);
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
        ratio_decided,
        failing,
        missing,
        unauthorized,
        worst,
        full_acceptance,
        tallies,
    }
}

pub(crate) const CONTROLS: [&str; 13] = [
    "acceptance-failing-non-hashtrinity-pair",
    "acceptance-omitted-incumbent-pair",
    "acceptance-unauthorized-not-applicable",
    "acceptance-crossing-ratio-interval",
    "acceptance-wrong-oracle-digest",
    "acceptance-serial-adapter-wrong-result",
    "acceptance-concurrency-mismatch",
    "acceptance-address-face-mismatch",
    "acceptance-address-face-mechanism",
    "acceptance-registered-baseline-regression",
    "acceptance-hashtrinity-half-target",
    "spatial-wrong-tuple-bucket-routing",
    "spatial-excluded-registry-overcharge",
];

/// The spatial adapter's own rejecting controls.
///
/// The adapter registers its controls in `spatial::CONTROLS` and `main.rs` dispatches a
/// `--performance-control` name to [`is_control`]; listing the tuple-path routing control and the
/// timed-region accounting control here is what puts them in the run. `performance::CONTROLS` carries
/// the three packed-face spatial controls and this list carries the acceptance, tuple-path and
/// accounting ones, and `run_accept` chains both, so all of them run under `--accept --controls`.
pub(crate) fn is_spatial_adapter_control(name: &str) -> bool {
    performance::spatial::is_control(name)
}

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
                address_reason: "synthetic control pair".to_string(),
                selected_address: "raw-word;direct_word=Some",
                oracle_digest: 0xABCD,
                expected_oracle_digest: 0xABCD,
                selected_samples: vec![100, 101, 102],
                incumbent_samples: vec![100, 101, 102],
                campaign_ratios: vec![0.5, 0.5, 0.5],
                timed_region: SERIAL_TIMED_REGION,
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

/// The address-face mechanism must reject a mechanism change, not just accept a correct one.
///
/// Two directions, both required:
/// * `address_face_unmatched == 0` for a packed cell where every identity-exact arm is matched — a
///   mechanism that returned `Unmatched` for a `Matched` arm would leave the counter at 0 and pass a
///   counter-only check.
/// * a tuple cell still counts its hashtrinity pairs, so the exemption is not silently widened.
fn run_address_mechanism_control() -> i32 {
    use crate::performance::spatial::{CellIdentity, KeyFace};
    let name = "acceptance-address-face-mechanism";
    let packed_exact = CellIdentity {
        distinct_keys: 1000,
        distinct_addresses: 1000,
        merged_keys: 0,
        max_packed: 0x7FFF_FFFF_FFFF_FFFF,
    };
    let tuple_colliding = CellIdentity {
        distinct_keys: 99_943,
        distinct_addresses: 99_897,
        merged_keys: 46,
        max_packed: 0x7FFF_FFFF_FFFF_FFFF,
    };
    let mut checked = 0usize;
    for id in [
        "hashtrinity_count_par",
        "hashbrown_rayon",
        "foldhash_par16",
        "fxhash_par16",
    ] {
        let (face, _) = spatial_address_face(id, KeyFace::Packed, &packed_exact);
        if face != AddressFace::Matched {
            eprintln!(
                "performance-control {name} NOT rejected: packed identity-exact arm {id} \
                 reported {}",
                face.name()
            );
            return 1;
        }
        checked += 1;
    }
    let (ht, reason) =
        spatial_address_face("hashtrinity_count_par", KeyFace::Tuple, &tuple_colliding);
    if ht != AddressFace::Unmatched || !reason.contains("no key store") {
        eprintln!(
            "performance-control {name} NOT rejected: tuple hashtrinity arm reported {} with \
             reason {reason}",
            ht.name()
        );
        return 1;
    }
    // The merged-key count in the reason must be this cell's observation, so the verdict is evidence.
    if !reason.contains("46 of 99943") {
        eprintln!(
            "performance-control {name} NOT rejected: the tuple reason does not carry this cell's \
             merged-key observation: {reason}"
        );
        return 1;
    }
    let (whole_key, _) = spatial_address_face("hashbrown_rayon", KeyFace::Tuple, &tuple_colliding);
    if whole_key != AddressFace::Matched {
        eprintln!(
            "performance-control {name} NOT rejected: hashbrown_rayon stores the whole tuple key \
             and must be matched on the tuple face, reported {}",
            whole_key.name()
        );
        return 1;
    }
    eprintln!(
        "performance-control {name} rejected: {checked} packed identity-exact arms matched, \
         tuple hashtrinity unmatched with the no-key-store reason carrying the observed merge"
    );
    2
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

/// The 0.5 target's own control: the bound separates the hashtrinity pairs at 0.75 and it is
/// scoped to them.
///
/// Six directions, all required, so the check cannot pass by rejecting everything or by tightening
/// one pair the wrong way:
/// * the unseeded synthetic campaign passes the half target, at exactly `0.50` — the bound is
///   inclusive, so a check that rejected `>=` would be rejected here;
/// * a single seeded hashtrinity pair at `0.75` is **rejected** by the half target through the
///   ratio channel, by name, with `failing.len() == 1` so the rejection is attributable to that
///   pair and not to collateral damage;
/// * the same seeded pair at `0.75` **passes** the registered gate, which is the separation the
///   target claims and which is what makes the `1.0` gate's unchanged behaviour part of the proof
///   rather than an assumption;
/// * the same `0.75` on a non-hashtrinity pair is **accepted** by the half target, so a limit
///   applied globally instead of to [`TARGET_SCOPED_INCUMBENTS`] is caught;
/// * a hashtrinity pair at exactly `0.50` passes and one at `0.5001` fails, pinning the boundary
///   from both sides;
/// * the repetition block's per-campaign flags follow the same per-pair bound, so a target that
///   gated the pooled decision but reported `1.00` flags is caught.
#[allow(
    clippy::too_many_lines,
    reason = "one function states the whole target control and its six rejecting directions"
)]
fn run_target_control() -> i32 {
    let name = "acceptance-hashtrinity-half-target";
    let half = RatioTarget::half();
    let registered = RatioTarget::registered();
    let seeded_ratio = 0.75;

    let campaign_with = |incumbent: &str, ratio: f64| -> Campaign {
        let mut campaign = synthetic_campaign();
        let Some(pair) = campaign
            .pairs
            .iter_mut()
            .find(|p| p.incumbent_id == incumbent && p.class == PairClass::LoadBearing)
        else {
            return campaign;
        };
        pair.campaign_ratios = vec![ratio, ratio, ratio];
        campaign
    };

    // Non-vacuity, and the inclusive boundary: every pair of the synthetic campaign sits at exactly
    // `TARGET_HALF`, so the half target decides all of them and passes all of them.
    let unseeded = evaluate_with(&synthetic_campaign(), false, half);
    if !unseeded.passed() || unseeded.applicable == 0 {
        eprintln!(
            "performance-control {name} NOT rejected: the unseeded synthetic campaign at \
             TARGET_HALF={TARGET_HALF:.2} reported failing={:?} applicable={}",
            unseeded.failing, unseeded.applicable
        );
        return 1;
    }

    // The seeded hashtrinity pair is rejected by the half target, through the ratio channel alone.
    let seeded_ht = evaluate_with(&campaign_with(HASHTRINITY_TABLE, seeded_ratio), false, half);
    let seeded_line = format!("rmax={seeded_ratio:.4} > {TARGET_HALF:.2}");
    if seeded_ht.passed()
        || seeded_ht.failing.len() != 1
        || !seeded_ht.failing[0].contains(&seeded_line)
        || !seeded_ht.failing[0].contains(HASHTRINITY_TABLE)
        || !seeded_ht.failing[0].contains("synthetic-cell-a")
    {
        eprintln!(
            "performance-control {name} NOT rejected: a {seeded_ratio:.2} hashtrinity pair reported \
             failing={:?} missing={:?} unauthorized={:?}",
            seeded_ht.failing, seeded_ht.missing, seeded_ht.unauthorized
        );
        return 1;
    }

    // The separation claim: the registered gate passes the same pair at the same ratio.
    let seeded_registered = evaluate_with(
        &campaign_with(HASHTRINITY_TABLE, seeded_ratio),
        false,
        registered,
    );
    if !seeded_registered.passed() {
        eprintln!(
            "performance-control {name} NOT rejected: the registered gate rejected a {seeded_ratio:.2} \
             hashtrinity pair (failing={:?}), so the target does not separate the two gates",
            seeded_registered.failing
        );
        return 1;
    }

    // The scope claim: the same ratio on a non-hashtrinity incumbent is still accepted, so the
    // target is not a global tightening.
    let seeded_other = evaluate_with(
        &campaign_with("hashbrown_foldhash", seeded_ratio),
        false,
        half,
    );
    if !seeded_other.passed() {
        eprintln!(
            "performance-control {name} NOT rejected: the half target tightened a non-hashtrinity \
             pair at {seeded_ratio:.2} (failing={:?}), so the bound is not scoped to \
             TARGET_SCOPED_INCUMBENTS",
            seeded_other.failing
        );
        return 1;
    }

    // The boundary from both sides: `TARGET_HALF` exactly passes and the next representable ratio
    // above it fails, so the comparison is `r_max > limit` and not `>=`.
    let at_bound = evaluate_with(&campaign_with(HASHTRINITY_TABLE, TARGET_HALF), false, half);
    let above_bound = evaluate_with(
        &campaign_with(HASHTRINITY_TABLE, f64::from_bits(TARGET_HALF.to_bits() + 1)),
        false,
        half,
    );
    if !at_bound.passed() || above_bound.passed() {
        eprintln!(
            "performance-control {name} NOT rejected: at TARGET_HALF={TARGET_HALF:.2} passed={}, the \
             next ratio above it passed={}",
            at_bound.passed(),
            above_bound.passed()
        );
        return 1;
    }

    // The reported per-campaign flags follow the same per-pair bound, so the block cannot advertise
    // `1.00` flags over a population the gate decided at 0.50.
    let campaign = campaign_with(HASHTRINITY_TABLE, seeded_ratio);
    let outcome = evaluate_with(&campaign, false, half);
    let repetitions = match collect_repetitions_with(&campaign, &outcome, half) {
        Ok(repetitions) => repetitions,
        Err(e) => {
            eprintln!("performance-control {name} NOT rejected: the seeded repetition block: {e}");
            return 1;
        }
    };
    let Some(flags) = repetitions
        .pairs
        .iter()
        .find(|p| p.incumbent_id == HASHTRINITY_TABLE && p.cell_id == "synthetic-cell-a")
        .map(|p| p.pass_per_campaign.clone())
    else {
        eprintln!(
            "performance-control {name} NOT rejected: the seeded pair is absent from the block"
        );
        return 1;
    };
    if flags.iter().any(|passed| *passed) || flags.len() != 3 {
        eprintln!(
            "performance-control {name} NOT rejected: the {seeded_ratio:.2} hashtrinity pair reported \
             per-campaign pass flags {flags:?} under the half target"
        );
        return 1;
    }
    eprintln!(
        "performance-control {name} rejected: a {seeded_ratio:.2} hashtrinity pair is rejected by \
         the half target through the ratio channel alone (failing=1 \"{seeded_line}\") and accepted \
         by the registered {RATIO_LIMIT:.2} target; the same ratio on a non-hashtrinity incumbent is \
         accepted by both, so the bound is scoped; TARGET_HALF={TARGET_HALF:.2} itself passes and the \
         next ratio above it fails; the per-campaign flags for the seeded pair are {flags:?}"
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
    if name == "acceptance-address-face-mechanism" {
        return run_address_mechanism_control();
    }
    // The tuple-path routing control: the adapter's `Corruption::Routing` branch inside
    // `run_count_par_tuple`, which no packed-face control could reach. Dispatched here so it runs.
    if is_spatial_adapter_control(name) {
        return performance::spatial::run_control_child(name, Profile::Smoke);
    }
    if name == "acceptance-concurrency-mismatch" {
        return run_concurrency_control();
    }
    if name == "acceptance-address-face-mismatch" {
        return run_address_face_control();
    }
    if name == "acceptance-registered-baseline-regression" {
        return run_regression_control();
    }
    if name == "acceptance-hashtrinity-half-target" {
        return run_target_control();
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

/// One registered spill-read comparator cell: a registered current-strata cell whose selected read
/// enters the spill region, with the observed present-key split that names it.
///
/// The registry is the coverage record the spill read needs. Every current-strata cell is measured
/// against every applicable incumbent already, but the selected read's route — resident flat array
/// or spill region — was not recorded, so a reader could not tell which registered cells exercise
/// the ported `spill_read` and which exercise the resident read. The split is observed on the real
/// table (see [`crate::strata::read_path_split_by_cell`]), not inferred from a cell name.
struct SpillReadCell {
    id: String,
    spill_read_keys: usize,
    resident_read_keys: usize,
}

/// The registered comparison matrix, embedded so the gate reads the compiled contract.
const MATRIX_JSON: &str = include_str!("matrix.json");

/// Load the registered spill-read comparator cells from the embedded matrix.
///
/// # Errors
/// Returns a description of a malformed registration document or a missing required field.
fn load_spill_read_cells() -> Result<Vec<SpillReadCell>, String> {
    let root = crate::Reader::new(MATRIX_JSON).parse()?;
    let obj = root.as_obj()?;
    let crate::Json::Arr(entries) = crate::field(obj, "spill_read_cells")? else {
        return Err("field spill_read_cells: expected array".to_string());
    };
    let mut cells = Vec::with_capacity(entries.len());
    for entry in entries {
        let fields = entry.as_obj()?;
        let count = |key: &str| -> Result<usize, String> {
            crate::str_field(fields, key)?
                .parse::<usize>()
                .map_err(|e| format!("registered spill-read cell field {key}: {e}"))
        };
        cells.push(SpillReadCell {
            id: crate::str_field(fields, "id")?,
            spill_read_keys: count("spill_read_keys")?,
            resident_read_keys: count("resident_read_keys")?,
        });
    }
    Ok(cells)
}

/// Load the registered spill-read comparator cells and enforce them against the observed read-path
/// split: every registered cell must be a current cell whose split agrees, and every current cell
/// whose reads enter the spill region must be registered. The registry is closed in both
/// directions, so no spill-read cell is measured and unrecorded, and no recorded cell is
/// resident-only. A registered cell with a zero spill count, a registered split that does not sum
/// to the cell's present population, or an absent cell rejects.
///
/// # Errors
/// Returns a description of the first duplicate, absent, unregistered or disagreeing cell. A caller
/// that observes `Ok(())` has a closed spill-read registry, not a spill-read acceptance.
fn validate_spill_read_registry() -> Result<usize, String> {
    let registered = load_spill_read_cells()?;
    let observed = strata::read_path_split_by_cell();
    let mut seen = HashSet::with_capacity(registered.len());
    for cell in &registered {
        if !seen.insert(cell.id.as_str()) {
            return Err(format!("duplicate registered spill-read cell {}", cell.id));
        }
        if cell.spill_read_keys == 0 {
            return Err(format!(
                "registered spill-read cell {} records zero spill-read keys",
                cell.id
            ));
        }
        if cell.spill_read_keys + cell.resident_read_keys == 0 {
            return Err(format!(
                "registered spill-read cell {} records an empty present population",
                cell.id
            ));
        }
        let Some(&(spill, resident)) = observed.get(cell.id.as_str()) else {
            return Err(format!(
                "registered spill-read cell {} is absent from the current enumeration",
                cell.id
            ));
        };
        if (spill, resident) != (cell.spill_read_keys, cell.resident_read_keys) {
            return Err(format!(
                "registered spill-read cell {} split {spill}+{resident} != registered {}+{}",
                cell.id, cell.spill_read_keys, cell.resident_read_keys
            ));
        }
    }
    let mut spill_read_cells = 0usize;
    for (id, (spill, resident)) in &observed {
        if *spill > 0 {
            spill_read_cells += 1;
            if !seen.contains(id.as_str()) {
                return Err(format!(
                    "required spill-read cell {id} ({spill} spill / {resident} resident) is absent \
                     from the registration"
                ));
            }
        }
    }
    for cell in &registered {
        println!(
            "spill_read_cell={} spill_read_keys={} resident_read_keys={} incumbent={} \
             baseline_revision={}",
            cell.id,
            cell.spill_read_keys,
            cell.resident_read_keys,
            HASHTRINITY_TABLE,
            matrix::REGISTERED_BASELINE_REVISION
        );
    }
    println!(
        "spill_read_registry cells={} spill_read_cells={spill_read_cells} \
         resident_only_cells={} incumbent={} baseline_revision={} observed=search_decision_at",
        observed.len(),
        observed.len() - spill_read_cells,
        HASHTRINITY_TABLE,
        matrix::REGISTERED_BASELINE_REVISION
    );
    Ok(registered.len())
}

// ── Registered-baseline regression check ─────────────────────────────────────────────────────
//
// The per-pair acceptance above answers one question only: is the candidate ahead of the incumbent
// on this `(cell, incumbent)` pair. It never compares the candidate with the numbers the same cell
// recorded in `matrix.json`, so a candidate that loses three-fold against its own registered median
// while still trailing the incumbent is reported `fail` before and after the loss, and `fail` has
// been the verdict on these cells for a long time. `matrix.rs`'s registration check deliberately
// ignores the registered medians (`matrix.rs:968`); this check is the separate, named comparison
// that reads them.
//
// Two deltas, never one. The candidate median is compared with the registered candidate median and
// the incumbent median with the registered incumbent median, each against a tolerance built from
// its own measured spread. An incumbent that got slower never widens the candidate's tolerance and
// an incumbent that got faster never hides a candidate regression.
//
// The tolerance is derived from the spread this harness measures on every run: the per-campaign
// medians of one cell's pooled samples. `strata::run_cell` appends `STRATA_SAMPLES` samples per
// campaign in campaign order, so the pooled vector's blocks are exactly the campaigns whose medians
// it already used for `campaign_ratios`. `max/min` over those block medians is the campaign-to-
// campaign spread of the median, and its excess over 1 is this cell's measured headroom: a cell
// whose own campaigns disagree gets more room than a cell whose campaigns agree.
//
// That in-run spread is not the whole bound, and the difference decides the check's sensitivity.
// Measured on this source over independent full-profile runs of the same binary with nothing
// changed between them, one cell's pooled median moves between runs by up to 1.895x on the
// candidate side (p50 1.050, p90 1.121) and 2.154x on the incumbent side (p50 1.050, p99 2.093),
// and the per-cell in-run spread does not predict it: 8 of 160 cells whose in-run spread is
// 1.02-1.24 moved 1.22-1.90x across two runs of identical code. That movement is machine load
// between processes and no statistic of one run observes it. So the check reports two named tiers
// and gates on the one the evidence supports:
//
// * `regressed` — the delta is beyond `1 + max(this cell's own in-run spread excess, the measured
//   cross-run floor)`. Reachable only by a loss larger than anything run-to-run noise produced.
// * `regressed-noise-band` — the delta is beyond the quiet floor but inside the cross-run floor.
//   Reported per cell and counted, never gating: on this box a movement of that size is not
//   separable from load, and a gate that fires on 5% of cells under load teaches its readers to
//   ignore it.
//
// The two floors differ because the two sides were measured separately, and neither side's noise
// widens the other's bound.

/// The registered incumbent every `current_strata` receipt is paired with.
const REGRESSION_INCUMBENT: &str = HASHTRINITY_TABLE;

/// The percentile of the measured per-cell median spread distribution the summary reports. The
/// per-cell verdict does not use it: a cell is judged against its own measured spread, and a
/// global quantile would either over-tolerate the quiet cells or hand a noisy cell extra slack.
const REGRESSION_SPREAD_PERCENTILE: usize = 99;

/// The multiplier on a cell's own measured spread excess. One: the cell is allowed exactly the
/// spread it measured, and no more.
const REGRESSION_SPREAD_FACTOR: f64 = 1.0;

/// The candidate floor for the non-gating noise band, in units of the registered candidate median:
/// the cross-run movement of the *quiet* cells, which is what a candidate has to beat before its
/// movement is worth naming at all.
const REGRESSION_CANDIDATE_QUIET_FLOOR: f64 = 0.20;

/// The incumbent floor for the non-gating noise band, the same quantity on the incumbent side.
const REGRESSION_INCUMBENT_QUIET_FLOOR: f64 = 0.32;

/// The widest cross-run movement of the pooled candidate median observed on this machine over
/// independent full-profile runs of one source (max 1.895). The gating floor: a candidate may be
/// this much slower than its registered median before the check calls it a regression.
const REGRESSION_CANDIDATE_CROSS_RUN_FLOOR: f64 = 0.90;

/// The widest cross-run movement of the pooled incumbent median observed over the same runs
/// (max 2.154), on the incumbent side only.
const REGRESSION_INCUMBENT_CROSS_RUN_FLOOR: f64 = 1.16;

/// The largest tolerance either side may be granted, in units of its registered median. A cell that
/// would need more is reported as regressed, not granted.
const REGRESSION_SPREAD_CEILING: f64 = 1.00;

/// One registered `current_strata` performance receipt: the numbers the registering run measured.
///
/// The registered medians are receipts of another run on another revision. They are read here and
/// nowhere else, and only for this comparison.
#[derive(Clone)]
struct RegisteredReceipt {
    id: String,
    apeiron_median_ns: u64,
    ht_median_ns: u64,
    ratio: f64,
    revision: String,
    baseline_revision: String,
}

/// One cell's regression verdict against its registered receipt.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RegressionVerdict {
    /// The candidate is slower than its registered median beyond its tolerance.
    Regressed,
    /// The candidate is beyond the quiet floor but inside the measured cross-run floor. Named and
    /// counted, never gating: the movement is not separable from machine load on this box.
    RegressedNoiseBand,
    /// The candidate is faster than its registered median beyond its tolerance.
    Improved,
    /// Both sides moved within their own tolerances.
    Within,
    /// The candidate is within tolerance and the incumbent is slower than its registered median
    /// beyond its own tolerance. Reported, never charged to the candidate.
    IncumbentSlower,
    /// The registered receipt's revisions disagree with the compiled-in pins. Reported, never
    /// compared.
    RevisionMismatch,
    /// The measured cell has no registered receipt.
    Unregistered,
    /// The receipt exists but the run did not measure this `(cell, incumbent)` pair.
    NotMeasured,
}

impl RegressionVerdict {
    fn name(self) -> &'static str {
        match self {
            RegressionVerdict::Regressed => "regressed",
            RegressionVerdict::RegressedNoiseBand => "regressed-noise-band",
            RegressionVerdict::Improved => "improved",
            RegressionVerdict::Within => "within",
            RegressionVerdict::IncumbentSlower => "incumbent-slower",
            RegressionVerdict::RevisionMismatch => "revision-mismatch",
            RegressionVerdict::Unregistered => "unregistered",
            RegressionVerdict::NotMeasured => "not-measured",
        }
    }
}

/// One cell's regression row.
struct RegressionCell {
    id: String,
    registered_ratio: f64,
    current_ratio: f64,
    candidate_delta: f64,
    incumbent_delta: f64,
    candidate_spread: f64,
    incumbent_spread: f64,
    candidate_tolerance: f64,
    incumbent_tolerance: f64,
    verdict: RegressionVerdict,
    reason: String,
}

/// The whole regression comparison and its named tally.
struct RegressionOutcome {
    cells: Vec<RegressionCell>,
    registered: usize,
    compared: usize,
    regressed: usize,
    regressed_noise_band: usize,
    improved: usize,
    within: usize,
    incumbent_slower: usize,
    revision_mismatch: usize,
    unregistered: usize,
    not_measured: usize,
    candidate_spread_p50: f64,
    candidate_spread_p90: f64,
    candidate_spread_p99: f64,
    candidate_spread_max: f64,
    incumbent_spread_p99: f64,
    incumbent_spread_max: f64,
    failing: Vec<String>,
    full_acceptance: bool,
}

impl RegressionOutcome {
    /// No cell regressed and every registered cell was actually compared against a revision-matched
    /// receipt. A revision mismatch or an unmeasured pair leaves the comparison incomplete, so it
    /// does not pass silently.
    fn passed(&self) -> bool {
        self.regressed == 0 && self.compared == self.registered && self.revision_mismatch == 0
    }

    /// The named non-gating band: cells that moved beyond the quiet floor but not beyond the
    /// measured cross-run floor. Reported in the summary so the sharp reading is never lost.
    fn noise_band(&self) -> usize {
        self.regressed_noise_band
    }
}

/// Load the registered performance receipts from the embedded matrix.
///
/// # Errors
/// Returns a description of a malformed registration document or a missing/unparsable required
/// field. A missing median is an error, never a skipped cell.
fn load_registered_receipts() -> Result<Vec<RegisteredReceipt>, String> {
    let root = crate::Reader::new(MATRIX_JSON).parse()?;
    let obj = root.as_obj()?;
    let crate::Json::Arr(entries) = crate::field(obj, "current_strata")? else {
        return Err("field current_strata: expected array".to_string());
    };
    let number = |fields: &[(String, crate::Json)], key: &str| -> Result<String, String> {
        crate::str_field(fields, key)
    };
    let mut cells = Vec::with_capacity(entries.len());
    for entry in entries {
        let fields = entry.as_obj()?;
        let apeiron_median_ns = number(fields, "apeiron_median_ns")?
            .parse::<u64>()
            .map_err(|e| format!("registered cell apeiron_median_ns: {e}"))?;
        let ht_median_ns = number(fields, "ht_median_ns")?
            .parse::<u64>()
            .map_err(|e| format!("registered cell ht_median_ns: {e}"))?;
        let ratio = number(fields, "ratio")?
            .parse::<f64>()
            .map_err(|e| format!("registered cell ratio: {e}"))?;
        cells.push(RegisteredReceipt {
            id: crate::str_field(fields, "id")?,
            apeiron_median_ns,
            ht_median_ns,
            ratio,
            revision: crate::str_field(fields, "revision")?,
            baseline_revision: crate::str_field(fields, "baseline_revision")?,
        });
    }
    Ok(cells)
}

/// The per-campaign medians of one pooled sample vector, as `max/min` over those medians.
// The nanosecond medians of a bounded run are far below 2^53, the same bound `matrix::ratio`
// states for the same conversion.
#[allow(
    clippy::cast_precision_loss,
    reason = "a bounded-run nanosecond median is far below 2^53"
)]
///
/// The vector's blocks are the campaigns (`strata::run_cell` appends `STRATA_SAMPLES` samples per
/// campaign in order), and the count is checked against the pair's own `campaign_ratios` rather than
/// assumed, so a pooled vector of a different shape yields `None` instead of a fabricated spread.
fn campaign_median_spread(samples: &[u64], campaigns: usize) -> Option<f64> {
    let block = strata::STRATA_SAMPLES as usize;
    if campaigns < 2 || samples.len() != campaigns * block {
        return None;
    }
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    for chunk in samples.chunks_exact(block) {
        let m = median(chunk);
        if m == 0 {
            return None;
        }
        lo = lo.min(m);
        hi = hi.max(m);
    }
    (lo > 0).then(|| hi as f64 / lo as f64)
}

/// The `percentile`-th quantile of a sample by nearest rank, on an integer percentile so no
/// float-to-index conversion is involved.
fn quantile(values: &[f64], percentile: usize) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = sorted.len() * percentile.min(100);
    let rank = rank.div_ceil(100).max(1);
    sorted[rank.min(sorted.len()) - 1]
}

/// One side's tolerance: the cell's own measured spread excess, floored by the measured cross-run
/// bound for that side and clamped so no cell can grant itself unbounded room. A cell with no
/// measurable in-run spread gets the floor, which is the cross-run bound and nothing more.
fn tolerance(spread: f64, floor: f64) -> f64 {
    let excess = if spread.is_finite() {
        spread - 1.0
    } else {
        f64::NEG_INFINITY
    };
    (REGRESSION_SPREAD_FACTOR * excess.max(floor)).clamp(0.0, REGRESSION_SPREAD_CEILING)
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "one function states the whole regression contract; the nanosecond medians of a \
              bounded run are far below 2^53, the bound matrix::ratio states for the same cast"
)]
fn evaluate_regression(
    campaign: &Campaign,
    receipts: &[RegisteredReceipt],
    full_profile: bool,
) -> RegressionOutcome {
    // The measured `(cell, incumbent)` pairs this comparison reads: one per cell, against the
    // incumbent every registered receipt names.
    let mut measured: HashMap<&str, &PairOutcome> = HashMap::new();
    for pair in &campaign.pairs {
        if pair.incumbent_id != REGRESSION_INCUMBENT {
            continue;
        }
        if pair.selected_samples.is_empty() || pair.incumbent_samples.is_empty() {
            continue;
        }
        measured.insert(pair.cell_id.as_str(), pair);
    }

    // Pass one: the measured spreads, so the tolerance is built from this run's data before any
    // verdict is assigned.
    let mut candidate_spreads = Vec::new();
    let mut incumbent_spreads = Vec::new();
    for receipt in receipts {
        let Some(pair) = measured.get(receipt.id.as_str()) else {
            continue;
        };
        let campaigns = pair.campaign_ratios.len();
        if let Some(spread) = campaign_median_spread(&pair.selected_samples, campaigns) {
            candidate_spreads.push(spread);
        }
        if let Some(spread) = campaign_median_spread(&pair.incumbent_samples, campaigns) {
            incumbent_spreads.push(spread);
        }
    }
    let candidate_spread_p50 = quantile(&candidate_spreads, 50);
    let candidate_spread_p90 = quantile(&candidate_spreads, 90);
    let candidate_spread_p99 = quantile(&candidate_spreads, REGRESSION_SPREAD_PERCENTILE);
    let candidate_spread_max = candidate_spreads.iter().copied().fold(0.0, f64::max);
    let incumbent_spread_p99 = quantile(&incumbent_spreads, REGRESSION_SPREAD_PERCENTILE);
    let incumbent_spread_max = incumbent_spreads.iter().copied().fold(0.0, f64::max);

    // Pass two: the verdicts, each against its own side's tolerance.
    let mut cells = Vec::with_capacity(receipts.len());
    let mut failing = Vec::new();
    let mut regressed = 0usize;
    let mut improved = 0usize;
    let mut within = 0usize;
    let mut incumbent_slower = 0usize;
    let mut revision_mismatch = 0usize;
    let mut compared = 0usize;
    let mut regressed_noise_band = 0usize;
    for receipt in receipts {
        let mut row = RegressionCell {
            id: receipt.id.clone(),
            registered_ratio: receipt.ratio,
            current_ratio: f64::NAN,
            candidate_delta: f64::NAN,
            incumbent_delta: f64::NAN,
            candidate_spread: f64::NAN,
            incumbent_spread: f64::NAN,
            candidate_tolerance: REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
            incumbent_tolerance: REGRESSION_INCUMBENT_CROSS_RUN_FLOOR,
            verdict: RegressionVerdict::NotMeasured,
            reason: String::new(),
        };
        let Some(pair) = measured.get(receipt.id.as_str()) else {
            row.verdict = RegressionVerdict::NotMeasured;
            row.reason = format!("no measured {REGRESSION_INCUMBENT} pair in this run");
            cells.push(row);
            continue;
        };
        let (candidate_median, incumbent_median) = (
            median(&pair.selected_samples),
            median(&pair.incumbent_samples),
        );
        row.current_ratio = ratio(candidate_median, incumbent_median);
        row.candidate_delta = candidate_median as f64 / receipt.apeiron_median_ns as f64;
        row.incumbent_delta = incumbent_median as f64 / receipt.ht_median_ns as f64;
        let campaigns = pair.campaign_ratios.len();
        row.candidate_spread =
            campaign_median_spread(&pair.selected_samples, campaigns).unwrap_or(f64::NAN);
        row.incumbent_spread =
            campaign_median_spread(&pair.incumbent_samples, campaigns).unwrap_or(f64::NAN);
        row.candidate_tolerance =
            tolerance(row.candidate_spread, REGRESSION_CANDIDATE_CROSS_RUN_FLOOR);
        row.incumbent_tolerance =
            tolerance(row.incumbent_spread, REGRESSION_INCUMBENT_CROSS_RUN_FLOOR);
        // A receipt recorded against other revisions is reported, never compared: the medians on
        // both sides then describe different source than the ones the comparison is about. A
        // receipt whose revision does not *contain* the artifact it registers is the same defect
        // reached a second way, and the string test alone cannot see it: the pin and the row are
        // written together, so they agree by construction even when neither names a revision that
        // has the harness. Containment is settled against the object store.
        let mut revision_faults: Vec<String> = Vec::new();
        if receipt.revision != matrix::REGISTERED_CURRENT_REVISION {
            revision_faults.push(format!(
                "candidate revision {} != pin {}",
                receipt.revision,
                matrix::REGISTERED_CURRENT_REVISION
            ));
        }
        if receipt.baseline_revision != matrix::REGISTERED_BASELINE_REVISION {
            revision_faults.push(format!(
                "baseline revision {} != pin {}",
                receipt.baseline_revision,
                matrix::REGISTERED_BASELINE_REVISION
            ));
        }
        if let Err(e) = matrix::revision_contains_registered_artifact(&receipt.revision) {
            revision_faults.push(e);
        }
        if !revision_faults.is_empty() {
            row.verdict = RegressionVerdict::RevisionMismatch;
            row.reason = format!(
                "registered medians do not describe this source: {}",
                revision_faults.join("; ")
            );
            revision_mismatch += 1;
            failing.push(format!("({}) {}", receipt.id, row.reason));
            cells.push(row);
            continue;
        }
        compared += 1;
        if !row.candidate_delta.is_finite() || !row.incumbent_delta.is_finite() {
            row.verdict = RegressionVerdict::NotMeasured;
            row.reason =
                "a registered median is zero or the current median is not finite".to_string();
            cells.push(row);
            continue;
        }
        let candidate_over = row.candidate_delta > 1.0 + row.candidate_tolerance;
        let candidate_under = row.candidate_delta < 1.0 - row.candidate_tolerance;
        let incumbent_over = row.incumbent_delta > 1.0 + row.incumbent_tolerance;
        row.verdict = match (candidate_over, candidate_under) {
            (true, _) => RegressionVerdict::Regressed,
            (false, true) => RegressionVerdict::Improved,
            (false, false) if row.candidate_delta > 1.0 + REGRESSION_CANDIDATE_QUIET_FLOOR => {
                RegressionVerdict::RegressedNoiseBand
            }
            (false, false) if incumbent_over => RegressionVerdict::IncumbentSlower,
            (false, false) => RegressionVerdict::Within,
        };
        row.reason = match row.verdict {
            RegressionVerdict::Regressed => format!(
                "candidate median {} ns is {:.4}x the registered {} ns, beyond its tolerance 1+{:.4}",
                candidate_median,
                row.candidate_delta,
                receipt.apeiron_median_ns,
                row.candidate_tolerance
            ),
            RegressionVerdict::Improved => format!(
                "candidate median {} ns is {:.4}x the registered {} ns",
                candidate_median, row.candidate_delta, receipt.apeiron_median_ns
            ),
            RegressionVerdict::RegressedNoiseBand => format!(
                "candidate median {} ns is {:.4}x the registered {} ns, beyond the quiet floor \
                 1+{:.4} but inside the measured cross-run floor 1+{:.4}, so this movement is not \
                 separable from machine load and is reported without gating",
                candidate_median,
                row.candidate_delta,
                receipt.apeiron_median_ns,
                REGRESSION_CANDIDATE_QUIET_FLOOR,
                row.candidate_tolerance
            ),
            RegressionVerdict::IncumbentSlower => format!(
                "candidate within its tolerance 1+{:.4}; incumbent median {} ns is {:.4}x \
                 the registered {} ns, beyond its own tolerance 1+{:.4}",
                row.candidate_tolerance,
                incumbent_median,
                row.incumbent_delta,
                receipt.ht_median_ns,
                row.incumbent_tolerance
            ),
            _ => format!(
                "candidate {:.4}x (tolerance 1+{:.4}) and incumbent {:.4}x (tolerance 1+{:.4}) of \
                 registered",
                row.candidate_delta,
                row.candidate_tolerance,
                row.incumbent_delta,
                row.incumbent_tolerance
            ),
        };
        match row.verdict {
            RegressionVerdict::Regressed => {
                regressed += 1;
                failing.push(format!("({}) {}", receipt.id, row.reason));
            }
            RegressionVerdict::RegressedNoiseBand => regressed_noise_band += 1,
            _ => {}
        }
        if row.verdict == RegressionVerdict::Improved {
            improved += 1;
        }
        if row.verdict == RegressionVerdict::IncumbentSlower {
            incumbent_slower += 1;
        }
        if row.verdict == RegressionVerdict::Within {
            within += 1;
        }
        cells.push(row);
    }

    // A measured cell the registration does not carry gets its own row and is never compared against
    // a neighbouring receipt.
    for (id, pair) in &measured {
        if receipts.iter().any(|r| r.id == *id) {
            continue;
        }
        let candidate_median = median(&pair.selected_samples);
        let incumbent_median = median(&pair.incumbent_samples);
        cells.push(RegressionCell {
            id: (*id).to_string(),
            registered_ratio: f64::NAN,
            current_ratio: ratio(candidate_median, incumbent_median),
            candidate_delta: f64::NAN,
            incumbent_delta: f64::NAN,
            candidate_spread: campaign_median_spread(
                &pair.selected_samples,
                pair.campaign_ratios.len(),
            )
            .unwrap_or(f64::NAN),
            incumbent_spread: campaign_median_spread(
                &pair.incumbent_samples,
                pair.campaign_ratios.len(),
            )
            .unwrap_or(f64::NAN),
            candidate_tolerance: REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
            incumbent_tolerance: REGRESSION_INCUMBENT_CROSS_RUN_FLOOR,
            verdict: RegressionVerdict::Unregistered,
            reason: format!(
                "measured against {REGRESSION_INCUMBENT} but absent from the current_strata \
                 registration, so it has no registered median to compare against"
            ),
        });
    }
    let unregistered = cells
        .iter()
        .filter(|c| c.verdict == RegressionVerdict::Unregistered)
        .count();
    let not_measured = cells
        .iter()
        .filter(|c| c.verdict == RegressionVerdict::NotMeasured)
        .count();
    let full_acceptance =
        full_profile && regressed == 0 && compared == receipts.len() && revision_mismatch == 0;
    RegressionOutcome {
        cells,
        registered: receipts.len(),
        compared,
        regressed,
        regressed_noise_band,
        improved,
        within,
        incumbent_slower,
        revision_mismatch,
        unregistered,
        not_measured,
        candidate_spread_p50,
        candidate_spread_p90,
        candidate_spread_p99,
        candidate_spread_max,
        incumbent_spread_p99,
        incumbent_spread_max,
        failing,
        full_acceptance,
    }
}

/// The per-cell regression table and its named tally.
fn print_regression(outcome: &RegressionOutcome, gate_full_acceptance: bool) {
    println!(
        "regression_rule candidate and incumbent medians are each compared with their own \
         registered matrix.json median; candidate_delta = current/registered candidate median, \
         incumbent_delta = current/registered incumbent median; tolerance = clamp(\
         {REGRESSION_SPREAD_FACTOR:.1} * max(spread - 1, floor), 0, {REGRESSION_SPREAD_CEILING:.2}) \
         of that cell's own measured per-campaign median spread, floored by the measured \
         cross-run bound of that side's pooled median; a delta above 1+tolerance is regressed \
         (gating), above 1+quiet_floor and inside the cross-run floor is regressed-noise-band \
         (named, counted, not gating); candidate_cross_run_floor=\
         {REGRESSION_CANDIDATE_CROSS_RUN_FLOOR:.2} incumbent_cross_run_floor=\
         {REGRESSION_INCUMBENT_CROSS_RUN_FLOOR:.2} candidate_quiet_floor=\
         {REGRESSION_CANDIDATE_QUIET_FLOOR:.2} incumbent_quiet_floor=\
         {REGRESSION_INCUMBENT_QUIET_FLOOR:.2}; incumbent movement is reported by name and never \
         charged to the candidate"
    );
    let (run_head, run_dirty) = crate::git_head();
    println!(
        "regression_revisions registered_candidate_revision={} registered_baseline_revision={} \
         pinned_candidate_revision={} pinned_baseline_revision={} hashtrinity_rev={} \
         apeiron_head={} dirty={}",
        matrix::REGISTERED_CURRENT_REVISION,
        matrix::REGISTERED_BASELINE_REVISION,
        matrix::REGISTERED_CURRENT_REVISION,
        matrix::REGISTERED_BASELINE_REVISION,
        crate::PINNED_HASHTRINITY_REV,
        run_head,
        run_dirty,
    );
    print_regression_table(outcome);
    println!(
        "regression_summary registered={} compared={} regressed={} improved={} within={} \
         regressed_noise_band={} incumbent_slower={} revision_mismatch={} not_measured={} \
         unregistered={} candidate_spread_p50={:.4} candidate_spread_p90={:.4} candidate_spread_p99={:.4} \
         candidate_spread_max={:.4} incumbent_spread_p99={:.4} incumbent_spread_max={:.4} \
         candidate_cross_run_floor={:.4} incumbent_cross_run_floor={:.4} spread_factor={:.2} \
         spread_ceiling={:.2} candidate_quiet_floor={:.2} incumbent_quiet_floor={:.2} \
         regression_full_acceptance={} gate_full_acceptance={}",
        outcome.registered,
        outcome.compared,
        outcome.regressed,
        outcome.improved,
        outcome.within,
        outcome.noise_band(),
        outcome.incumbent_slower,
        outcome.revision_mismatch,
        outcome.not_measured,
        outcome.unregistered,
        outcome.candidate_spread_p50,
        outcome.candidate_spread_p90,
        outcome.candidate_spread_p99,
        outcome.candidate_spread_max,
        outcome.incumbent_spread_p99,
        outcome.incumbent_spread_max,
        REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
        REGRESSION_INCUMBENT_CROSS_RUN_FLOOR,
        REGRESSION_SPREAD_FACTOR,
        REGRESSION_SPREAD_CEILING,
        REGRESSION_CANDIDATE_QUIET_FLOOR,
        REGRESSION_INCUMBENT_QUIET_FLOOR,
        outcome.full_acceptance,
        gate_full_acceptance,
    );
    for line in outcome.failing.iter().take(REGRESSION_FAILURE_LINES) {
        println!("regression_failure={line}");
    }
    if outcome.failing.len() > REGRESSION_FAILURE_LINES {
        println!(
            "regression_failure_truncated shown={REGRESSION_FAILURE_LINES} total={}",
            outcome.failing.len()
        );
    }
}

/// The per-cell regression table: one row per registered cell, its two deltas, both tolerances and
/// its verdict. The table is printed in full; only the named failure lines are truncated.
fn print_regression_table(outcome: &RegressionOutcome) {
    let headers = [
        "cell",
        "registered_ratio",
        "current_ratio",
        "candidate_delta",
        "candidate_tolerance",
        "incumbent_delta",
        "incumbent_tolerance",
        "candidate_spread",
        "incumbent_spread",
        "verdict",
    ];
    let mut rows: Vec<[String; 10]> = Vec::with_capacity(outcome.cells.len());
    for cell in &outcome.cells {
        rows.push([
            cell.id.clone(),
            format!("{:.4}", cell.registered_ratio),
            format!("{:.4}", cell.current_ratio),
            format!("{:.4}", cell.candidate_delta),
            format!("{:.4}", cell.candidate_tolerance),
            format!("{:.4}", cell.incumbent_delta),
            format!("{:.4}", cell.incumbent_tolerance),
            format!("{:.4}", cell.candidate_spread),
            format!("{:.4}", cell.incumbent_spread),
            cell.verdict.name().to_string(),
        ]);
    }
    let mut widths = headers.map(str::len);
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row.iter()) {
            *width = (*width).max(cell.len());
        }
    }
    let header_row: Vec<String> = headers.map(str::to_string).to_vec();
    let rule: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    print_regression_row(&header_row, &widths);
    print_regression_row(&rule, &widths);
    for row in &rows {
        print_regression_row(row, &widths);
    }
}

fn print_regression_row(row: &[String], widths: &[usize]) {
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

/// How many named regression failures the report prints before truncating. The tally is the
/// authority; the named lines are the pointer.
const REGRESSION_FAILURE_LINES: usize = 20;

/// The registered regression control: a seeded candidate regression is rejected by name, a seeded
/// incumbent regression is not, and the unseeded cell is within tolerance.
///
/// Three directions, all required:
/// * the control cell reproduces its registered receipt, so the check is shown to say `within`
///   rather than rejecting everything (non-vacuity);
/// * inflating only the candidate's median turns the verdict to `regressed` and the failure list to
///   a named `regression` entry;
/// * a candidate movement inside the measured cross-run floor is named `regressed-noise-band`,
///   counted and not gated, so the check reports what it cannot separate from machine load;
/// * inflating only the incumbent's median turns the verdict to `incumbent-slower` and leaves the
///   candidate `within` tolerance, which is the separation the check claims;
/// * a receipt rewritten to another revision is reported `revision-mismatch` and not compared.
#[allow(
    clippy::too_many_lines,
    reason = "one function states the whole regression control and its five rejecting directions"
)]
fn run_regression_control() -> i32 {
    let name = "acceptance-registered-baseline-regression";
    let Ok(receipts) = load_registered_receipts() else {
        eprintln!("performance-control {name} NOT rejected: the registered receipts did not load");
        return 1;
    };
    let Some(receipt) = receipts
        .iter()
        .find(|r| r.id == matrix::CONTROL_REGISTERED_CELL)
        .cloned()
    else {
        eprintln!(
            "performance-control {name} NOT rejected: the registered control cell {} is absent",
            matrix::CONTROL_REGISTERED_CELL
        );
        return 1;
    };
    let campaign = |candidate_median: u64, incumbent_median: u64| -> Campaign {
        // Three campaigns of `STRATA_SAMPLES` samples, the shape the real run produces, so the
        // control exercises the same block-median spread path rather than a special case.
        let block = strata::STRATA_SAMPLES as usize;
        let candidate: Vec<u64> = vec![candidate_median; block * 3];
        let incumbent: Vec<u64> = vec![incumbent_median; block * 3];
        Campaign {
            cells: vec![receipt.id.clone()],
            pairs: vec![PairOutcome {
                cell_id: receipt.id.clone(),
                incumbent_id: REGRESSION_INCUMBENT,
                class: PairClass::LoadBearing,
                adapter: AdapterState::Built,
                applicability: Applicability::Applicable("registered regression control"),
                sel_impl: SELECTED_IMPL,
                sel_workers: SELECTED_WORKERS,
                sample_basis: "serial",
                inc_impl: REGRESSION_INCUMBENT,
                inc_workers: Some(1),
                address_face: AddressFace::Matched,
                address_reason: "control: the raw-coordinate serial face".to_string(),
                selected_address: "raw-word;direct_word=Some",
                oracle_digest: 0x1234,
                expected_oracle_digest: 0x1234,
                campaign_ratios: vec![1.0; 3],
                selected_samples: candidate,
                incumbent_samples: incumbent,
                timed_region: SERIAL_TIMED_REGION,
            }],
        }
    };
    let verdict_of = |outcome: &RegressionOutcome| outcome.cells[0].verdict;

    // Non-vacuity: the check reads a cell that reproduces its registered receipt as within tolerance,
    // so the rejections below are not a check that rejects everything.
    let unseeded = evaluate_regression(
        &campaign(receipt.apeiron_median_ns, receipt.ht_median_ns),
        std::slice::from_ref(&receipt),
        true,
    );
    if verdict_of(&unseeded) != RegressionVerdict::Within || !unseeded.passed() {
        eprintln!(
            "performance-control {name} NOT rejected: the unseeded control cell reported {} \
             (regressed={} compared={} registered={})",
            verdict_of(&unseeded).name(),
            unseeded.regressed,
            unseeded.compared,
            unseeded.registered
        );
        return 1;
    }
    // The seeded candidate regression is rejected by name.
    let seeded_candidate = evaluate_regression(
        &campaign(receipt.apeiron_median_ns * 3, receipt.ht_median_ns),
        std::slice::from_ref(&receipt),
        true,
    );
    if verdict_of(&seeded_candidate) != RegressionVerdict::Regressed
        || !seeded_candidate
            .failing
            .iter()
            .any(|line| line.contains(") ") && line.contains("beyond its tolerance"))
        || seeded_candidate.passed()
        || seeded_candidate.full_acceptance
    {
        eprintln!(
            "performance-control {name} NOT rejected: a 3x seeded candidate regression reported {} \
             with failing={:?} (regressed={} full_acceptance={})",
            verdict_of(&seeded_candidate).name(),
            seeded_candidate.failing,
            seeded_candidate.regressed,
            seeded_candidate.full_acceptance
        );
        return 1;
    }
    // The incumbent's movement is reported separately and is not charged to the candidate.
    let seeded_incumbent = evaluate_regression(
        &campaign(receipt.apeiron_median_ns, receipt.ht_median_ns * 3),
        std::slice::from_ref(&receipt),
        true,
    );
    if verdict_of(&seeded_incumbent) != RegressionVerdict::IncumbentSlower
        || seeded_incumbent.regressed != 0
        || seeded_incumbent.incumbent_slower != 1
    {
        eprintln!(
            "performance-control {name} NOT rejected: a 3x seeded incumbent regression reported {} \
             with regressed={} incumbent_slower={}, so the incumbent's movement was charged to the \
             candidate or went unreported",
            verdict_of(&seeded_incumbent).name(),
            seeded_incumbent.regressed,
            seeded_incumbent.incumbent_slower
        );
        return 1;
    }
    // A candidate movement inside the cross-run floor is named, counted and not gated, so the
    // check reports what it cannot separate from machine load instead of rejecting it.
    let seeded_band = evaluate_regression(
        &campaign(receipt.apeiron_median_ns * 3 / 2, receipt.ht_median_ns),
        std::slice::from_ref(&receipt),
        true,
    );
    if verdict_of(&seeded_band) != RegressionVerdict::RegressedNoiseBand
        || seeded_band.noise_band() != 1
        || seeded_band.regressed != 0
        || !seeded_band.passed()
    {
        eprintln!(
            "performance-control {name} NOT rejected: a 1.5x seeded candidate movement reported {} \
             with noise_band={} regressed={} passed={}",
            verdict_of(&seeded_band).name(),
            seeded_band.noise_band(),
            seeded_band.regressed,
            seeded_band.passed()
        );
        return 1;
    }
    // A receipt recorded against another revision is reported and not compared.
    let mut foreign = receipt.clone();
    foreign.revision = "0000000000000000000000000000000000000000".to_string();
    let mismatched = evaluate_regression(
        &campaign(receipt.apeiron_median_ns, receipt.ht_median_ns),
        std::slice::from_ref(&foreign),
        true,
    );
    if verdict_of(&mismatched) != RegressionVerdict::RevisionMismatch
        || mismatched.compared != 0
        || mismatched.revision_mismatch != 1
        || mismatched.passed()
    {
        eprintln!(
            "performance-control {name} NOT rejected: a foreign-revision receipt reported {} with \
             compared={} revision_mismatch={}",
            verdict_of(&mismatched).name(),
            mismatched.compared,
            mismatched.revision_mismatch
        );
        return 1;
    }
    eprintln!(
        "performance-control {name} rejected: cell {} unseeded=within, seeded candidate x3={} \
         (named regression), seeded candidate x1.5={} (named band, not gated), seeded incumbent \
         x3={} (not charged to the candidate), foreign revision={} (reported, not compared)",
        receipt.id,
        verdict_of(&seeded_candidate).name(),
        verdict_of(&seeded_band).name(),
        verdict_of(&seeded_incumbent).name(),
        verdict_of(&mismatched).name(),
    );
    2
}

/// The registered cross-run spread of the acceptance pass tally, in pairs: the widest movement of
/// `acceptance_summary ... pass=` between independent full-profile invocations of one unchanged
/// source on this machine (673, 635, 617 — a spread of 56).
///
/// This is the floor a reader must use when comparing two *separate* invocations. The in-run
/// campaign spread [`Repetitions::spread`] is measured over the registered campaigns of a single
/// invocation and is narrower, because those campaigns are adjacent; reporting only the narrower
/// figure would license a reader to call a 40-pair movement between two runs a change, which is
/// the movement this constant already measured. The reported bound is the wider of the two.
const ACCEPT_PASS_COUNT_CROSS_RUN_SPREAD: usize = 56;

/// One decided pair's per-campaign pass flags and its ratio statistics over the campaign axis.
///
/// The registered full profile already runs `strata::FULL_CAMPAIGNS` campaigns of each cell's timed
/// region, and each campaign is one complete interleaved pass over the selected arm and every
/// incumbent arm (`strata::run_cell`: warmup both arms, then `STRATA_SAMPLES` samples alternating
/// arm order; `spatial::run_cell` the same per campaign). Each pass is therefore already a full
/// sweep of the timed region for that pair, so the repetition axis the harness reports is the one
/// it already pays for. No campaign is added and none is removed.
struct PairRepetitions {
    cell_id: String,
    incumbent_id: &'static str,
    /// One flag per campaign: whether that campaign's own `selected/incumbent` ratio met the
    /// registered limit.
    pass_per_campaign: Vec<bool>,
    ratio_mean: f64,
    ratio_min: f64,
    ratio_max: f64,
    /// `ratio_max / ratio_min`, the same multiplicative form the regression check measures, so one
    /// reader reads both with one definition.
    ratio_spread: f64,
    ratio_spread_abs: f64,
    /// Whether the pooled campaign-interval-max rule passed this pair. Reported beside the flags
    /// because the pooled verdict is the gate and can differ from every individual flag.
    pooled_pass: bool,
}

/// The per-incumbent aggregate over the campaign axis.
struct IncumbentRepetitions {
    id: &'static str,
    decided_pairs: usize,
    ratio_mean: f64,
    ratio_min: f64,
    ratio_max: f64,
    ratio_spread: f64,
    pass_per_campaign: Vec<usize>,
}

/// The whole repetition report: one invocation's mean and spread for the pass tally and for every
/// decided pair's ratio, so no reader has to invoke the harness three times to learn whether a
/// movement is inside the spread already measured on unchanged code.
struct Repetitions {
    /// The number of campaigns the profile ran, i.e. the repetition count. One for the smoke
    /// profile, which registers a single campaign; reported, never assumed to be three.
    campaigns: usize,
    /// Pairs that reached the ratio decision, the population every mean below is taken over.
    decided_pairs: usize,
    /// The gated pooled verdict, unchanged and unsmoothed.
    pooled_pass: usize,
    /// The pass tally of each campaign: the number of decided pairs whose own ratio met the limit.
    pass_per_campaign: Vec<usize>,
    pairs: Vec<PairRepetitions>,
    incumbents: Vec<IncumbentRepetitions>,
}

impl Repetitions {
    /// The pass tally's arithmetic mean over the campaigns.
    #[allow(
        clippy::cast_precision_loss,
        reason = "a pair count and a campaign count are both far below 2^53"
    )]
    fn pass_mean(&self) -> f64 {
        if self.pass_per_campaign.is_empty() {
            return f64::NAN;
        }
        let sum: usize = self.pass_per_campaign.iter().sum();
        sum as f64 / self.pass_per_campaign.len() as f64
    }

    fn pass_min(&self) -> usize {
        self.pass_per_campaign.iter().copied().min().unwrap_or(0)
    }

    fn pass_max(&self) -> usize {
        self.pass_per_campaign.iter().copied().max().unwrap_or(0)
    }

    /// The in-run pass-tally spread, `max - min` over the campaigns.
    fn spread(&self) -> usize {
        self.pass_max().saturating_sub(self.pass_min())
    }

    /// The population standard deviation of the pass tally over the campaigns. Named
    /// `sd` because it divides by the number of campaigns, not by one fewer.
    #[allow(
        clippy::cast_precision_loss,
        reason = "a pass tally is a pair count far below 2^53"
    )]
    fn pass_sd(&self) -> f64 {
        if self.pass_per_campaign.is_empty() {
            return f64::NAN;
        }
        let mean = self.pass_mean();
        let variance = self
            .pass_per_campaign
            .iter()
            .map(|&p| {
                let d = p as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / self.pass_per_campaign.len() as f64;
        variance.sqrt()
    }

    /// The bound a reader must respect: the wider of the spread this invocation measured over its
    /// own campaigns and the registered cross-invocation spread.
    fn spread_bound(&self) -> usize {
        self.spread().max(ACCEPT_PASS_COUNT_CROSS_RUN_SPREAD)
    }
}

/// Collect the per-campaign statistics over every pair that reached the ratio decision.
///
/// # Errors
/// Returns a description when `outcome.ratio_decided` is not one entry per pair, when a decided
/// pair's ratio interval is empty or non-finite, or when the pairs disagree about the campaign
/// count. Every one of those means the repetition axis is not what the profile says it is, and a
/// mean over it would be a fabricated number, so the run fails closed instead.
/// The repetition aggregation for one invocation's target.
///
/// The reported statistics — mean, minimum, maximum, spread — are the same numbers over the same
/// campaigns under every target; only the `pass_at_campaign` flags read `target.limit_for`, so the
/// block reports and never decides, as its own rule line states.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "one function states the whole repetition aggregation; a pair count, a campaign count \
              and a nanosecond-derived ratio are all far below 2^53, the bound matrix::ratio states \
              for the same conversion"
)]
fn collect_repetitions_with(
    campaign: &Campaign,
    outcome: &AcceptanceOutcome,
    target: RatioTarget,
) -> Result<Repetitions, String> {
    if outcome.ratio_decided.len() != campaign.pairs.len() {
        return Err(format!(
            "{} decided flags for {} pairs",
            outcome.ratio_decided.len(),
            campaign.pairs.len()
        ));
    }
    let campaigns = campaign
        .pairs
        .iter()
        .zip(&outcome.ratio_decided)
        .filter(|(_, &decided)| decided)
        .map(|(pair, _)| pair.campaign_ratios.len())
        .max()
        .unwrap_or(0);
    let mut pass_per_campaign = vec![0usize; campaigns];
    let mut pairs = Vec::new();
    let mut by_incumbent: HashMap<&'static str, IncumbentRepetitions> = HashMap::new();
    let mut decided_pairs = 0usize;
    let mut pooled_pass = 0usize;

    for (index, pair) in campaign.pairs.iter().enumerate() {
        if !outcome.ratio_decided[index] {
            continue;
        }
        if pair.campaign_ratios.is_empty() || pair.campaign_ratios.len() != campaigns {
            return Err(format!(
                "({}, {}) carries {} campaign ratios, expected {campaigns}",
                pair.cell_id,
                pair.incumbent_id,
                pair.campaign_ratios.len()
            ));
        }
        if pair.campaign_ratios.iter().any(|r| !r.is_finite()) {
            return Err(format!(
                "({}, {}) non-finite campaign ratio",
                pair.cell_id, pair.incumbent_id
            ));
        }
        let ratio_min = pair
            .campaign_ratios
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let ratio_max = pair
            .campaign_ratios
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = pair.campaign_ratios.iter().sum();
        let pass_flags: Vec<bool> = pair
            .campaign_ratios
            .iter()
            .map(|&r| r <= target.limit_for(pair.incumbent_id))
            .collect();
        for (k, passed) in pass_flags.iter().enumerate() {
            if *passed {
                pass_per_campaign[k] += 1;
            }
        }
        decided_pairs += 1;
        let pooled = outcome.pair_status[index] == PairStatus::Pass;
        if pooled {
            pooled_pass += 1;
        }
        let entry = by_incumbent
            .entry(pair.incumbent_id)
            .or_insert_with(|| IncumbentRepetitions {
                id: pair.incumbent_id,
                decided_pairs: 0,
                ratio_mean: 0.0,
                ratio_min: f64::INFINITY,
                ratio_max: f64::NEG_INFINITY,
                ratio_spread: 0.0,
                pass_per_campaign: vec![0; campaigns],
            });
        entry.decided_pairs += 1;
        entry.ratio_mean += sum / campaigns as f64;
        entry.ratio_min = entry.ratio_min.min(ratio_min);
        entry.ratio_max = entry.ratio_max.max(ratio_max);
        for (k, passed) in pass_flags.iter().enumerate() {
            if *passed {
                entry.pass_per_campaign[k] += 1;
            }
        }
        pairs.push(PairRepetitions {
            cell_id: pair.cell_id.clone(),
            incumbent_id: pair.incumbent_id,
            pass_per_campaign: pass_flags,
            ratio_mean: sum / campaigns as f64,
            ratio_min,
            ratio_max,
            ratio_spread: if ratio_min > 0.0 {
                ratio_max / ratio_min
            } else {
                f64::NAN
            },
            ratio_spread_abs: ratio_max - ratio_min,
            pooled_pass: pooled,
        });
    }
    for entry in by_incumbent.values_mut() {
        entry.ratio_mean /= entry.decided_pairs as f64;
        entry.ratio_spread = if entry.ratio_min > 0.0 {
            entry.ratio_max / entry.ratio_min
        } else {
            f64::NAN
        };
    }
    let mut incumbents: Vec<IncumbentRepetitions> = by_incumbent.into_values().collect();
    // The registration order, so the block reads against `acceptance_ratio incumbent=` above it.
    incumbents.sort_by_key(|e| {
        INCUMBENTS
            .iter()
            .position(|i| i.id == e.id)
            .unwrap_or(usize::MAX)
    });
    Ok(Repetitions {
        campaigns,
        decided_pairs,
        pooled_pass,
        pass_per_campaign,
        pairs,
        incumbents,
    })
}

/// The registered block's rule line, byte-identical to the one printed before a ratio target
/// existed. Kept as a literal rather than formatted, so the `1.0` report is unchanged.
const REPEATS_RULE_REGISTERED: &str = "pass_at_campaign_k is campaign k's own selected/incumbent \
                                      <= 1.00; the gate remains the pooled campaign interval max \
                                      over all campaigns, so the mean reports and never decides";

/// The rule line when a stricter target is in force: the per-campaign flag compares with the bound
/// that pair's own incumbent carries, and the gate is still the pooled campaign interval max.
const REPEATS_RULE_SCOPED: &str = "pass_at_campaign_k is campaign k's own selected/incumbent <= \
                                   0.50 on a hashtrinity pair and <= 1.00 on every other pair; \
                                   the gate remains the pooled campaign interval max over all \
                                   campaigns, so the mean reports and never decides";

/// Print the averaged block: the pass tally's mean and spread, then each incumbent's and each
/// decided pair's ratio mean and spread, over the campaigns this invocation already ran.
fn print_repetitions(profile: Profile, r: &Repetitions, target: RatioTarget) {
    let rule = if target.is_registered() {
        REPEATS_RULE_REGISTERED
    } else {
        REPEATS_RULE_SCOPED
    };
    println!(
        "acceptance_repeats profile={} axis=campaign campaigns={} decided_pairs={} pooled_pass={} \
         pass_mean={:.2} pass_min={} pass_max={} pass_spread={} pass_sd={:.2} \
         cross_run_floor={} spread_bound={} spread_bound_source={} rule=\"{rule}\"",
        profile.name(),
        r.campaigns,
        r.decided_pairs,
        r.pooled_pass,
        r.pass_mean(),
        r.pass_min(),
        r.pass_max(),
        r.spread(),
        r.pass_sd(),
        ACCEPT_PASS_COUNT_CROSS_RUN_SPREAD,
        r.spread_bound(),
        if r.spread() > ACCEPT_PASS_COUNT_CROSS_RUN_SPREAD {
            "max(measured_in_run,registered_cross_run)"
        } else {
            "registered_cross_run"
        },
    );
    for (k, passed) in r.pass_per_campaign.iter().enumerate() {
        println!(
            "acceptance_repeats_campaign profile={} campaign={} decided_pairs={} pass={passed}",
            profile.name(),
            k + 1,
            r.decided_pairs
        );
    }
    for inc in &r.incumbents {
        println!(
            "acceptance_repeats_incumbent incumbent={} decided_pairs={} ratio_mean={:.4} \
             ratio_min={:.4} ratio_max={:.4} ratio_spread={:.4} pass_at_campaign={:?}",
            inc.id,
            inc.decided_pairs,
            inc.ratio_mean,
            inc.ratio_min,
            inc.ratio_max,
            inc.ratio_spread,
            inc.pass_per_campaign
        );
    }
    for pair in &r.pairs {
        let flags: String = pair
            .pass_per_campaign
            .iter()
            .map(|passed| if *passed { 'P' } else { 'F' })
            .collect();
        println!(
            "acceptance_cell_ratio cell={} incumbent={} campaigns={} ratio_mean={:.4} \
             ratio_min={:.4} ratio_max={:.4} ratio_spread={:.4} ratio_spread_abs={:.4} \
             pass_at_campaign={flags} pooled={}",
            pair.cell_id,
            pair.incumbent_id,
            r.campaigns,
            pair.ratio_mean,
            pair.ratio_min,
            pair.ratio_max,
            pair.ratio_spread,
            pair.ratio_spread_abs,
            if pair.pooled_pass { "pass" } else { "fail" },
        );
    }
}

/// One scoped pair's verdict under the invocation's target: the bound that pair carries, the pooled
/// campaign interval max that was compared with it, the `PairStatus` the shared decision produced,
/// and the failure line when it failed.
///
/// A scoped pair that never reached the ratio decision — not applicable, missing, unauthorized, or
/// rejected on identity, concurrency or address face — is printed with `rmax=-` and its own status,
/// so an applicability or identity exemption is visible as such rather than as a 0.5 pass. The
/// counts returned are the scoped population's applicable/decided/failed tallies, and a target that
/// decided no scoped pair at all is vacuous and rejects: see [`scoped_verdicts`]'s caller.
fn print_scoped_verdicts(campaign: &Campaign, outcome: &AcceptanceOutcome, target: RatioTarget) {
    for (index, pair) in campaign.pairs.iter().enumerate() {
        if !TARGET_SCOPED_INCUMBENTS.contains(&pair.incumbent_id) {
            continue;
        }
        let limit = target.limit_for(pair.incumbent_id);
        let status = outcome
            .pair_status
            .get(index)
            .copied()
            .unwrap_or(PairStatus::Fail);
        let r_max_text = if outcome.ratio_decided.get(index).copied().unwrap_or(false) {
            format!(
                "{:.4}",
                pair.campaign_ratios
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            )
        } else {
            "-".to_string()
        };
        let failure = outcome
            .failing
            .iter()
            .find(|line| line.starts_with(&format!("({}, {}) ", pair.cell_id, pair.incumbent_id)))
            .map_or_else(|| "-".to_string(), Clone::clone);
        println!(
            "acceptance_target_pair cell={} incumbent={} limit={limit:.2} rmax={r_max_text} \
             campaigns={} status={} failing=\"{}\"",
            pair.cell_id,
            pair.incumbent_id,
            pair.campaign_ratios.len(),
            status.name(),
            failure
        );
    }
}

/// The scoped population's tallies under `target`: applicable, ratio-decided and failed scoped
/// pairs, in `campaign.pairs` order against `outcome`'s per-pair records.
///
/// `decided == 0` means the target gated nothing, so a stricter target must not report it as a
/// pass over an empty population.
fn scoped_tallies(campaign: &Campaign, outcome: &AcceptanceOutcome) -> (usize, usize, usize) {
    let mut applicable = 0usize;
    let mut decided = 0usize;
    let mut failed = 0usize;
    for (index, pair) in campaign.pairs.iter().enumerate() {
        if !TARGET_SCOPED_INCUMBENTS.contains(&pair.incumbent_id) {
            continue;
        }
        let status = outcome
            .pair_status
            .get(index)
            .copied()
            .unwrap_or(PairStatus::Fail);
        if matches!(
            status,
            PairStatus::Pass | PairStatus::Fail | PairStatus::Informational
        ) {
            applicable += 1;
        }
        if outcome.ratio_decided.get(index).copied().unwrap_or(false) {
            decided += 1;
        }
        if status == PairStatus::Fail {
            failed += 1;
        }
    }
    (applicable, decided, failed)
}

#[allow(
    clippy::too_many_lines,
    reason = "one function states the acceptance campaign's full phase sequence"
)]
pub(crate) fn run_accept(profile: Profile, controls: bool, json: bool) -> i32 {
    // The ratio target this invocation applies. Resolved before any measurement so a misspelled
    // selection costs no timed region, and refused rather than defaulted: a reader who typed a bound
    // must not receive the registered gate in its place.
    let target = match RatioTarget::from_env() {
        Ok(target) => target,
        Err(e) => {
            eprintln!("acceptance target refused: {e}");
            return USAGE_EXIT;
        }
    };
    // `report.rs`'s `acceptance_rule` line names `RATIO_LIMIT` as a constant, so under a stricter
    // target the `--json` dump would print a rule the gate does not apply. Refuse that combination
    // instead of printing a false rule; the pair table carries the per-pair verdict unchanged.
    if target.is_stricter() && json {
        eprintln!(
            "acceptance target refused: --json prints acceptance_rule with the registered \
             {RATIO_LIMIT:.2} bound, which is not the rule this invocation applies; run without \
             --json"
        );
        return USAGE_EXIT;
    }
    // Phase 1: the existing measurement, registration enforcement and diagnostics, unchanged except
    // that the acceptance path prints its own table, so the phase-1 dump is silent.
    let t_total = std::time::Instant::now();
    let mut t_last = t_total;
    let diagnostic =
        performance::run_performance(profile, false, false, performance::Output::Silent);
    if diagnostic != 0 {
        return diagnostic;
    }
    performance::phase_mark("accept.phase1", t_total, &mut t_last);
    // Phase 2: the registered per-pair acceptance.
    let strata = strata::run_strata(profile);
    performance::phase_mark("accept.strata", t_total, &mut t_last);
    let registered = match matrix::load_registered_cells() {
        Ok(cells) => cells,
        Err(e) => {
            eprintln!("acceptance registration load failed: {e}");
            return 1;
        }
    };
    performance::phase_mark("accept.registered", t_total, &mut t_last);
    if let Err(e) = matrix::validate_registration(&strata, &registered) {
        eprintln!("acceptance registration validation failed: {e}");
        return 1;
    }
    let spill_read_cells = match validate_spill_read_registry() {
        Ok(count) => count,
        Err(e) => {
            eprintln!("spill-read registration validation failed: {e}");
            return 1;
        }
    };
    performance::phase_mark("accept.spill_read_validation", t_total, &mut t_last);
    performance::phase_mark("accept.registration_validation", t_total, &mut t_last);
    if let Err(e) = strata::validate_strata(&strata) {
        eprintln!("acceptance strata validation failed: {e}");
        return 1;
    }
    performance::phase_mark("accept.strata_validation", t_total, &mut t_last);
    let spatial = performance::spatial::measure(&strata);
    performance::phase_mark("accept.spatial", t_total, &mut t_last);
    if json {
        performance::spatial::print(&spatial);
    }
    if let Err(e) = performance::spatial::validate(&spatial) {
        eprintln!("acceptance spatial validation failed: {e}");
        return 1;
    }
    let spatial_registry_cells = match performance::spatial::validate_registry(&spatial) {
        Ok(count) => count,
        Err(e) => {
            eprintln!("acceptance spatial registration validation failed: {e}");
            return 1;
        }
    };
    let campaign = build_campaign(&strata, &registered, &spatial);
    performance::phase_mark("accept.campaign", t_total, &mut t_last);
    // The registered-baseline regression comparison. Separate from the per-pair acceptance above,
    // which is untouched: this one reads the registered medians the registration check ignores, and
    // it gates the exit status as a conjunct of the pair acceptance, never as a replacement for it.
    let receipts = match load_registered_receipts() {
        Ok(receipts) => receipts,
        Err(e) => {
            eprintln!("regression registration load failed: {e}");
            return 1;
        }
    };
    let regression = evaluate_regression(&campaign, &receipts, profile == Profile::Full);
    performance::phase_mark("accept.regression", t_total, &mut t_last);
    println!(
        "acceptance_phase=per-pair profile={} registered_cells={} spatial_cells={} \
         spatial_registry_cells={spatial_registry_cells} spill_read_cells={spill_read_cells} json={json}",
        profile.name(),
        strata.cells.len(),
        spatial.cells.len()
    );
    if json {
        print_registration();
        print_ratio_summary(&campaign);
    }
    let outcome = evaluate_with(&campaign, profile == Profile::Full, target);
    performance::phase_mark("accept.evaluate", t_total, &mut t_last);
    print_address_faces(&campaign);
    print_spatial_timed_regions(&campaign, &spatial);
    if json {
        print_acceptance(&outcome);
    } else {
        print_pair_table(&campaign, &outcome);
    }
    print_headline(&outcome);
    // The scoped per-pair verdict under this invocation's target. Printed beside the shared pair
    // table and the HEADLINE, never instead of them: the table's `status` column and the HEADLINE
    // tallies are the same decision, read through the target.
    let (scoped_applicable, scoped_decided, scoped_failed) = scoped_tallies(&campaign, &outcome);
    if target.is_stricter() {
        target.print_target();
        print_scoped_verdicts(&campaign, &outcome, target);
        println!(
            "acceptance_target_summary applicable={scoped_applicable} ratio_decided={scoped_decided} \
             failed={scoped_failed} target={}",
            target.name()
        );
        // A target that decided no scoped pair gated nothing. The registered gate reports an empty
        // population as it always has; a stricter target must reject it rather than pass vacuously.
        if scoped_decided == 0 {
            eprintln!(
                "acceptance target rejected: {} decided 0 of {} scoped pairs, so it gated nothing",
                target.name(),
                scoped_applicable
            );
            return 1;
        }
    }
    // The mean and spread block. The gated verdict above is untouched: the pooled campaign interval
    // max decides, and this reports what the registered campaigns of this one invocation measured.
    let repetitions = match collect_repetitions_with(&campaign, &outcome, target) {
        Ok(repetitions) => repetitions,
        Err(e) => {
            eprintln!("acceptance repetition aggregation failed: {e}");
            return 1;
        }
    };
    performance::phase_mark("accept.repetitions", t_total, &mut t_last);
    print_repetitions(profile, &repetitions, target);
    // `gate_full_acceptance` is the pair acceptance AND the registered-baseline regression. The
    // HEADLINE line above keeps reporting the pair acceptance alone, so the registered pair tallies
    // it states are unchanged by the regression check; the regression is a separate conjunct here.
    let gate_full_acceptance = outcome.full_acceptance && regression.full_acceptance;
    print_regression(&regression, gate_full_acceptance);
    if controls {
        for name in performance::CONTROLS.into_iter().chain(CONTROLS) {
            match run_control(name, profile) {
                Ok(code) => println!("control={name} observed=red child_exit={code}"),
                Err(e) => {
                    eprintln!("control {name} failed: {e}");
                    return 1;
                }
            }
        }
    }
    performance::phase_mark("accept.controls", t_total, &mut t_last);
    // The registered gate's verdict lines carry no target field, so a `1.0` run's output is
    // unchanged; a stricter run names the bound its verdicts were decided under.
    let target_suffix = if target.is_registered() {
        String::new()
    } else {
        format!(
            " target={} scoped_limit={:.2}",
            target.name(),
            target.scoped
        )
    };
    if outcome.passed() && regression.passed() {
        println!(
            "strict_acceptance=pass applicable={} failing=0 missing=0 concurrency_unmatched={} \
             full_acceptance={} regression_registered={} regression_regressed={}{target_suffix}",
            outcome.applicable,
            outcome.concurrency_unmatched,
            outcome.full_acceptance,
            regression.registered,
            regression.regressed
        );
        0
    } else {
        eprintln!(
            "acceptance failed: applicable={} failing={} missing={} unauthorized={} \
             concurrency_unmatched={} full_acceptance=false regressed={} revision_mismatch={} \
             registered={} compared={} gate_full_acceptance=false{target_suffix}",
            outcome.applicable,
            outcome.failing.len(),
            outcome.missing.len(),
            outcome.unauthorized.len(),
            outcome.concurrency_unmatched,
            regression.regressed,
            regression.revision_mismatch,
            regression.registered,
            regression.compared,
        );
        1
    }
}

/// Registry-integrity tests. These read the compiled registry and the pure classification helpers;
/// they run no benchmark and no adapter. A required incumbent that is omitted, demoted to
/// informational, given an applicability that rejects, or mis-classified fails here, so the
/// acceptance report cannot silently drop a required comparison.
#[cfg(test)]
mod registry_tests;

/// Ratio-target tests: the selection rule, the registry closure of its scope, the separation it
/// claims against the registered gate, and the per-pair bound the repetition block reports.
#[cfg(test)]
mod target_tests {
    use super::{
        collect_repetitions_with, evaluate, evaluate_with, synthetic_campaign, AdapterState,
        Applicability, PairClass, RatioTarget, INCUMBENTS, RATIO_LIMIT, TARGET_HALF,
        TARGET_SCOPED_INCUMBENTS, TARGET_SCOPED_INCUMBENT_PREFIX,
    };

    /// A campaign whose one named incumbent's every pair sits at `ratio`, every other pair left at
    /// the synthetic default of `TARGET_HALF`.
    fn campaign_with(incumbent: &str, ratio: f64) -> super::Campaign {
        let mut campaign = synthetic_campaign();
        let Some(pair) = campaign
            .pairs
            .iter_mut()
            .find(|p| p.incumbent_id == incumbent && p.class == PairClass::LoadBearing)
        else {
            panic!("synthetic campaign has a load-bearing ({incumbent}) pair");
        };
        pair.campaign_ratios = vec![ratio, ratio, ratio];
        campaign
    }

    /// The selection rule: unset is the registered gate, `half` is the owner's target, a ratio in
    /// `(0, RATIO_LIMIT]` tightens only the scoped arms, and every other value is refused rather than
    /// defaulted — a misspelled selection must not leave the reader the registered gate.
    #[test]
    fn target_selection_is_total_and_fail_closed() {
        assert_eq!(RatioTarget::parse(None), Ok(RatioTarget::registered()));
        assert_eq!(
            RatioTarget::parse(Some("1.0")),
            Ok(RatioTarget::registered())
        );
        assert_eq!(RatioTarget::parse(Some("1")), Ok(RatioTarget::registered()));
        assert_eq!(RatioTarget::parse(Some("half")), Ok(RatioTarget::half()));
        assert_eq!(RatioTarget::parse(Some(" half ")), Ok(RatioTarget::half()));
        let custom = RatioTarget::parse(Some("0.75")).expect("0.75 is a ratio in (0, 1.0]");
        assert_eq!(custom.limit_for("hashtrinity_table"), 0.75);
        // No selection loosens the bound for a pair it does not scope.
        assert_eq!(custom.limit_for("hashbrown_foldhash"), RATIO_LIMIT);
        assert_eq!(
            RatioTarget::half().limit_for("hashbrown_foldhash"),
            RATIO_LIMIT
        );
        assert!(custom.is_stricter() && RatioTarget::half().is_stricter());
        assert!(!RatioTarget::registered().is_stricter());
        assert_eq!(RatioTarget::registered().name(), "registered-1.00");
        assert_eq!(RatioTarget::half().name(), "hashtrinity-half");
        assert_eq!(custom.name(), "hashtrinity-custom");
        // A set-but-unparsable or out-of-range selection is refused, including an empty value: a variable
        // that was typed and carries no bound is a typo, not an unset variable.
        for refused in [
            "", "0", "0.0", "-0.5", "1.5", "nan", "inf", "0.5x", "faster", "HALF",
        ] {
            let parsed = RatioTarget::parse(Some(refused));
            assert!(
                parsed.is_err(),
                "{refused:?} must be refused, got {parsed:?}"
            );
        }
    }

    /// The target's scope is closed over the registry: every registered id carrying the hashtrinity
    /// prefix is in [`TARGET_SCOPED_INCUMBENTS`], and vice versa. A newly registered hashtrinity arm
    /// therefore fails here instead of silently inheriting the looser bound.
    #[test]
    fn target_scope_is_closed_over_the_registry() {
        let registered_ht: Vec<&str> = INCUMBENTS
            .iter()
            .map(|inc| inc.id)
            .filter(|id| id.starts_with(TARGET_SCOPED_INCUMBENT_PREFIX))
            .collect();
        assert_eq!(
            registered_ht,
            TARGET_SCOPED_INCUMBENTS.to_vec(),
            "the scoped list must name exactly the registered hashtrinity arms"
        );
        // Each scoped incumbent is registered exactly once, load-bearing and built, so no scoped
        // bound is carried by an arm that never reaches the ratio decision.
        for id in TARGET_SCOPED_INCUMBENTS {
            let matching: Vec<&super::Incumbent> =
                INCUMBENTS.iter().filter(|inc| inc.id == id).collect();
            assert_eq!(matching.len(), 1, "{id} must be registered exactly once");
            assert_eq!(matching[0].class, PairClass::LoadBearing, "{id}");
            assert_eq!(matching[0].adapter, AdapterState::Built, "{id}");
            assert!(
                !matches!(
                    matching[0].applicability,
                    Applicability::UnauthorizedNotApplicable(_)
                ),
                "{id} must not be an unauthorized not-applicable arm; the scoped bound reaches it \
                 where a pair is applicable"
            );
        }
    }

    /// The separation the target claims: a single otherwise-valid hashtrinity pair at 0.75 is
    /// rejected by the half target through the ratio channel and accepted by the registered gate,
    /// while the same ratio on a pair the target does not scope is accepted by both.
    #[test]
    fn the_half_target_rejects_a_075_hashtrinity_pair_and_the_registered_gate_passes_it() {
        let half = RatioTarget::half();
        let registered = RatioTarget::registered();
        for id in TARGET_SCOPED_INCUMBENTS {
            let seeded = evaluate_with(&campaign_with(id, 0.75), false, half);
            assert!(!seeded.passed(), "{id} at 0.75 must fail the half target");
            assert_eq!(
                seeded.failing.len(),
                1,
                "{id}: the rejection must be attributable to the seeded pair alone: {:?}",
                seeded.failing
            );
            assert!(
                seeded.failing[0].contains(&format!("rmax=0.7500 > {TARGET_HALF:.2}")),
                "{id}: {} must name the bound it crossed",
                seeded.failing[0]
            );
            assert!(seeded.missing.is_empty() && seeded.unauthorized.is_empty());
            let unseeded = evaluate_with(&campaign_with(id, TARGET_HALF), false, half);
            assert!(unseeded.passed(), "{id}: the bound is inclusive");
            assert!(
                evaluate_with(&campaign_with(id, 0.75), false, registered).passed(),
                "{id}: the registered gate must still pass 0.75"
            );
        }
        assert!(evaluate_with(&campaign_with("hashbrown_foldhash", 0.75), false, half).passed());
    }

    /// The registered gate is the default of every target-free path, and it is bit-for-bit the
    /// behaviour of the ratio decision before the target existed: a pair at exactly `RATIO_LIMIT`
    /// passes and one above it fails, under `evaluate` itself.
    #[test]
    fn the_registered_gate_is_unchanged_and_is_the_default() {
        for id in TARGET_SCOPED_INCUMBENTS {
            assert!(evaluate(&campaign_with(id, RATIO_LIMIT), false).passed());
            let above = evaluate(&campaign_with(id, 1.0000001), false);
            assert!(
                !above.passed(),
                "{id}: 1.0000001 must exceed the registered bound"
            );
            assert_eq!(RatioTarget::parse(None), Ok(RatioTarget::registered()));
            assert!(RatioTarget::registered().limit_for(id) == RATIO_LIMIT);
        }
    }

    /// The repetition block's per-campaign flags read the same per-pair bound the pooled decision
    /// used, so a target that gated at 0.50 cannot report `1.00` flags over its own population.
    #[test]
    fn the_repetition_flags_follow_the_per_pair_bound() {
        let campaign = campaign_with("hashtrinity_table", 0.75);
        let half = RatioTarget::half();
        let outcome = evaluate_with(&campaign, false, half);
        let r = collect_repetitions_with(&campaign, &outcome, half).expect("well formed");
        let flags: Vec<bool> = r
            .pairs
            .iter()
            .find(|p| p.incumbent_id == "hashtrinity_table" && p.cell_id == "synthetic-cell-a")
            .expect("the seeded pair is in the block")
            .pass_per_campaign
            .clone();
        assert_eq!(
            flags,
            vec![false, false, false],
            "0.75 exceeds the half bound"
        );
        // And the registered target's flags over the same pair are all true.
        let registered_outcome = evaluate_with(&campaign, false, RatioTarget::registered());
        let registered_r =
            collect_repetitions_with(&campaign, &registered_outcome, RatioTarget::registered())
                .expect("well formed");
        let registered_flags: Vec<bool> = registered_r
            .pairs
            .iter()
            .find(|p| p.incumbent_id == "hashtrinity_table" && p.cell_id == "synthetic-cell-a")
            .expect("the seeded pair is in the block")
            .pass_per_campaign
            .clone();
        assert_eq!(
            registered_flags,
            vec![true, true, true],
            "0.75 meets the registered bound"
        );
    }
}

#[cfg(test)]
mod repetition_tests {
    use super::{
        collect_repetitions_with, synthetic_campaign, RatioTarget, Repetitions,
        ACCEPT_PASS_COUNT_CROSS_RUN_SPREAD, RATIO_LIMIT,
    };
    use crate::matrix::median;

    /// A synthetic campaign whose pairs carry `ratios` per campaign, with every other pair left at
    /// the synthetic default. Returns the campaign, the repetition report, and the synthetic
    /// campaign's own decided-pair count.
    fn report_for(cell: &str, incumbent: &str, ratios: &[f64]) -> Repetitions {
        let mut campaign = synthetic_campaign();
        let Some(pair) = campaign
            .pairs
            .iter_mut()
            .find(|p| p.cell_id == cell && p.incumbent_id == incumbent)
        else {
            panic!("synthetic campaign has no ({cell}, {incumbent}) pair");
        };
        pair.campaign_ratios = ratios.to_vec();
        let outcome = super::evaluate(&campaign, false);
        collect_repetitions_with(&campaign, &outcome, RatioTarget::registered())
            .expect("the synthetic campaign is well formed")
    }

    /// The mean is over the campaigns the profile ran, and the spread is the range over the same
    /// campaigns: three campaigns whose pass flags are all, none, and one true give `pass_per_campaign`
    /// `[n+1, n, n+1]` about this pair and a mean strictly between the extremes.
    #[test]
    fn pass_mean_and_spread_cover_every_campaign() {
        let r = report_for("synthetic-cell-a", "hashtrinity_table", &[0.5, 0.5, 0.5]);
        assert_eq!(r.campaigns, 3);
        assert_eq!(r.pass_per_campaign.len(), 3);
        // Every decided pair of the synthetic campaign sits at 0.5, so all three campaigns tally
        // the same and the spread is exactly zero.
        assert_eq!(r.spread(), 0);
        assert_eq!(r.pass_mean(), r.pass_min() as f64);
        assert_eq!(r.pass_mean(), r.pass_max() as f64);
        assert!((r.pass_sd()).abs() < f64::EPSILON);
    }

    /// A single-pass outlier is the case the mean exists for. Seeding one campaign of one pair
    /// past the limit moves that campaign's tally by exactly one and leaves the other two
    /// unchanged, and the reported spread names the movement.
    #[test]
    fn a_seeded_outlier_moves_its_campaign_only_and_widens_the_spread() {
        let baseline = report_for("synthetic-cell-a", "hashtrinity_table", &[0.5, 0.5, 0.5]);
        // The third campaign of this one pair is seeded above the limit.
        let seeded = report_for(
            "synthetic-cell-a",
            "hashtrinity_table",
            &[0.5, 0.5, RATIO_LIMIT + 0.5],
        );
        assert_eq!(seeded.campaigns, 3);
        assert_eq!(
            seeded.pass_per_campaign.len(),
            baseline.pass_per_campaign.len()
        );
        // Exactly one campaign lost exactly one pair, and it is the seeded one.
        for (k, (before, after)) in baseline
            .pass_per_campaign
            .iter()
            .zip(&seeded.pass_per_campaign)
            .enumerate()
        {
            let delta = *before as i64 - *after as i64;
            assert_eq!(delta, i64::from(k == 2), "campaign {k} moved by {delta}");
        }
        // The mean is bounded: one pair out of the population cannot drag it below the untouched
        // campaigns' tally, and the spread is exactly the one pair.
        assert!(seeded.spread() >= baseline.spread());
        assert_eq!(seeded.spread(), 1);
        assert!(seeded.pass_mean() < seeded.pass_max() as f64);
        assert!(seeded.pass_mean() >= seeded.pass_min() as f64 - f64::EPSILON);
        // The seeded pair names itself: its own spread is the outlier and its mean is bounded by
        // the limit it crossed.
        let pair = seeded
            .pairs
            .iter()
            .find(|p| p.cell_id == "synthetic-cell-a" && p.incumbent_id == "hashtrinity_table")
            .expect("the seeded pair is in the report");
        assert_eq!(pair.pass_per_campaign, vec![true, true, false]);
        assert_eq!(pair.ratio_min, 0.5);
        assert_eq!(pair.ratio_max, RATIO_LIMIT + 0.5);
        assert_eq!(pair.ratio_spread, (RATIO_LIMIT + 0.5) / 0.5);
        assert!(pair.ratio_mean > 0.5 && pair.ratio_mean < RATIO_LIMIT + 0.5);
        assert!(
            !pair.pooled_pass,
            "the pooled gate still fails the seeded pair"
        );
    }

    /// The per-cell ratio mean is the mean of that pair's own campaign ratios and the spread its own
    /// `max/min`, so a reader needs no second definition to compare two cells.
    #[test]
    fn per_cell_ratio_mean_and_spread_come_from_that_pairs_campaigns() {
        let ratios = [0.4, 0.5, 0.6];
        let r = report_for("synthetic-cell-a", "hashtrinity_table", &ratios);
        let pair = r
            .pairs
            .iter()
            .find(|p| p.cell_id == "synthetic-cell-a" && p.incumbent_id == "hashtrinity_table")
            .expect("the pair is in the report");
        let want_mean = ratios.iter().sum::<f64>() / 3.0;
        assert!((pair.ratio_mean - want_mean).abs() < 1e-12);
        assert_eq!(pair.ratio_min, 0.4);
        assert_eq!(pair.ratio_max, 0.6);
        assert!((pair.ratio_spread - 1.5).abs() < 1e-12);
        assert!((pair.ratio_spread_abs - 0.2).abs() < 1e-12);
        // The untouched pairs of the same incumbent are still reported, at their own 0.5.
        let others: Vec<f64> = r
            .pairs
            .iter()
            .filter(|p| p.cell_id == "synthetic-cell-a" && p.incumbent_id != "hashtrinity_table")
            .map(|p| p.ratio_mean)
            .collect();
        assert!(others.iter().all(|m| (*m - 0.5).abs() < 1e-12));
    }

    /// The incumbent aggregate is the mean over its own decided pairs' means, so one noisy pair
    /// cannot dominate it.
    #[test]
    fn incumbent_mean_is_the_mean_over_its_pairs() {
        let r = report_for("synthetic-cell-a", "hashtrinity_table", &[1.5, 1.5, 1.5]);
        let inc = r
            .incumbents
            .iter()
            .find(|i| i.id == "hashtrinity_table")
            .expect("the incumbent is in the report");
        // One pair at 1.5 among the incumbent's other pairs at 0.5.
        let want = (1.5 + 0.5 * (inc.decided_pairs as f64 - 1.0)) / inc.decided_pairs as f64;
        assert!(inc.decided_pairs >= 2);
        assert!(
            (inc.ratio_mean - want).abs() < 1e-12,
            "{} vs {want}",
            inc.ratio_mean
        );
        assert_eq!(inc.ratio_min, 0.5);
        assert_eq!(inc.ratio_max, 1.5);
        assert!((inc.ratio_spread - 3.0).abs() < 1e-12);
        // And the failing pair contributes no pass to any campaign.
        assert_eq!(inc.pass_per_campaign, vec![inc.decided_pairs - 1; 3]);
    }

    /// The reported bound is never narrower than the registered cross-invocation spread, which is
    /// what keeps a reader from calling a cross-run movement a change.
    #[test]
    fn spread_bound_is_never_narrower_than_the_registered_cross_run_spread() {
        let r = report_for("synthetic-cell-a", "hashtrinity_table", &[0.5, 0.5, 0.5]);
        assert_eq!(r.spread(), 0);
        assert_eq!(r.spread_bound(), ACCEPT_PASS_COUNT_CROSS_RUN_SPREAD);
        assert!(r.spread_bound() >= r.spread());
        assert!(r.spread_bound() >= ACCEPT_PASS_COUNT_CROSS_RUN_SPREAD);
    }

    /// The aggregation fails closed on a decided-flag vector that is not one entry per pair, rather
    /// than silently aggregating a prefix.
    #[test]
    fn a_short_decided_vector_is_rejected() {
        let campaign = synthetic_campaign();
        let mut outcome = super::evaluate(&campaign, false);
        outcome.ratio_decided.pop();
        let e = collect_repetitions_with(&campaign, &outcome, RatioTarget::registered())
            .err()
            .expect("a short decided vector must not aggregate");
        assert!(e.contains("decided flags"), "{e}");
    }

    /// A decided pair whose campaign count disagrees with the rest is rejected: a mean over a
    /// ragged axis would compare campaigns that no pair ran together.
    #[test]
    fn a_ragged_campaign_count_is_rejected() {
        let mut campaign = synthetic_campaign();
        campaign.pairs[0].campaign_ratios = vec![0.5, 0.5];
        let outcome = super::evaluate(&campaign, false);
        let e = collect_repetitions_with(&campaign, &outcome, RatioTarget::registered())
            .err()
            .expect("a ragged campaign count must not aggregate");
        assert!(e.contains("campaign ratios"), "{e}");
    }

    /// The per-cell statistics agree with the medians the pair table already prints: the reported
    /// ratio interval is the same interval the pair table's `ratio_interval` column shows.
    #[test]
    fn per_cell_interval_matches_the_pair_table_column() {
        let ratios = [0.30, 0.42, 0.55];
        let r = report_for("synthetic-cell-a", "hashtrinity_table", &ratios);
        let pair = r
            .pairs
            .iter()
            .find(|p| p.cell_id == "synthetic-cell-a" && p.incumbent_id == "hashtrinity_table")
            .expect("the pair is in the report");
        let mut sorted = ratios.to_vec();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(pair.ratio_min, sorted[0]);
        assert_eq!(pair.ratio_max, sorted[sorted.len() - 1]);
        // And the median of a constant-ratio synthetic pair is 100 ns on both sides, so the
        // pair-table ratio is 1.0 while the campaign ratios above are the seeded values: the two
        // are different statistics and the block names the campaign one.
        let synthetic = median(&[100, 101, 102]);
        assert_eq!(synthetic, 101);
    }
}
