use std::collections::HashMap;

use super::{
    campaign_median_spread, evaluate, evaluate_regression, is_spatial_incumbent,
    load_registered_receipts, pair_concurrency, serial_address_face, spatial_address_face,
    synthetic_campaign, tolerance, AdapterState, AddressFace, Applicability, Concurrency,
    PairClass, PairOutcome, RegisteredReceipt, INCUMBENTS, INFORMATIONAL_COMPARATORS, RATIO_LIMIT,
    REGRESSION_CANDIDATE_CROSS_RUN_FLOOR, REGRESSION_CANDIDATE_QUIET_FLOOR, REGRESSION_INCUMBENT,
    REGRESSION_INCUMBENT_CROSS_RUN_FLOOR, REGRESSION_INCUMBENT_QUIET_FLOOR,
    REGRESSION_SPREAD_CEILING, REGRESSION_SPREAD_FACTOR, REQUIRED_LOAD_BEARING, SELECTED_WORKERS,
    SERIAL_TIMED_REGION, SPATIAL_INCUMBENTS,
};
use crate::performance::spatial::{self, CellIdentity, KeyFace, INCUMBENT_REGISTRY_SYMBOLS};

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
            Applicability::Applicable(reason) | Applicability::AuthorizedNotApplicable(reason) => {
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
    // The address face is decided by the identity mechanism, not by an arm list. A cell whose packed
    // population is below hashtrinity's `min(EMPTY-2)` top fiber merges no keys, so every
    // identity-exact arm is matched on it.
    let packed_exact = CellIdentity {
        distinct_keys: 1000,
        distinct_addresses: 1000,
        merged_keys: 0,
        max_packed: 0x7FFF_FFFF_FFFF_FFFF,
    };
    for id in [
        "hashtrinity_count_par",
        "hashbrown_rayon",
        "foldhash_par16",
        "fxhash_par16",
    ] {
        assert_eq!(
            spatial_address_face(id, KeyFace::Packed, &packed_exact).0,
            AddressFace::Matched,
            "{id} stores or recovers the packed key exactly and must be matched on the packed face"
        );
    }
    // On the packed face hashtrinity's `Coord` is the identity, so `hashtrinity_get_each` — which is
    // the same combined arm as `count_par` — is matched too.
    assert_eq!(
        spatial_address_face("hashtrinity_get_each", KeyFace::Packed, &packed_exact).0,
        AddressFace::Matched,
        "hashtrinity_get_each runs the same identity-exact packed arm as hashtrinity_count_par"
    );
    // The tuple face is where `FreqLine`'s absent key column bites: distinct tuples sharing a
    // coordinate are one entry, and no amount of measurement on this population makes the surface
    // exact.
    let tuple_colliding = CellIdentity {
        distinct_keys: 99_943,
        distinct_addresses: 99_897,
        merged_keys: 46,
        max_packed: 0x7FFF_FFFF_FFFF_FFFF,
    };
    for id in ["hashtrinity_count_par", "hashtrinity_get_each"] {
        let (face, reason) = spatial_address_face(id, KeyFace::Tuple, &tuple_colliding);
        assert_eq!(
            face,
            AddressFace::Unmatched,
            "{id} has no key store and must be unmatched on the tuple face"
        );
        assert!(
            reason.contains("no key store"),
            "{id} tuple-face reason must name the absent key store: {reason}"
        );
    }
    for id in ["hashbrown_rayon", "foldhash_par16", "fxhash_par16"] {
        assert_eq!(
            spatial_address_face(id, KeyFace::Tuple, &tuple_colliding).0,
            AddressFace::Matched,
            "{id} stores the whole tuple key and must be matched on the tuple face"
        );
    }
    // An arm with no registered identity surface is not applicable, and is not silently matched.
    assert_eq!(
        spatial_address_face("not_a_registered_arm", KeyFace::Packed, &packed_exact).0,
        AddressFace::NotApplicable,
        "an arm with no registered identity mechanism must not be reported matched"
    );
    // A packed population that does merge keys is unmatched with the count named: the mechanism is
    // driven by this cell's observation, not only by the face.
    let packed_colliding = CellIdentity {
        distinct_keys: 100,
        distinct_addresses: 99,
        merged_keys: 1,
        max_packed: u64::MAX,
    };
    let (face, reason) =
        spatial_address_face("hashtrinity_count_par", KeyFace::Packed, &packed_colliding);
    assert_eq!(
        face,
        AddressFace::Unmatched,
        "a packed population above the top fiber merges keys and must be unmatched"
    );
    assert!(
        reason.contains("merged 1 of 100 keys"),
        "the packed reason must name the observed merge: {reason}"
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

/// `hashtrinity_get_each` is timed through the combined `count_par` build plus grouped probe, so it
/// reports the width that work used rather than a single-core probe width it never ran alone.
///
/// The defect this replaces was `arm_get_each` being `run_count_par` with `workers: 1` written over
/// its policy, then asserted here as a permanently *failing* pair. A test must not pin a mislabelled
/// arm as green: twelve registered pairs were locked to `Concurrency::Mismatch` by the assertion
/// below. What is asserted now is the arm's accuracy, and that a width-correct pair at the pool
/// width is *matched*, so the permanent mismatch cannot return.
#[test]
fn get_each_reports_the_width_it_actually_ran_at() {
    let pool = hashtrinity::par::Pool::global().width().max(1);
    // The recorded width is the arm's own policy, taken verbatim from what it ran: no correction,
    // no rewrite. `spatial::validate_policy` is the gate that checks each arm's reported policy
    // against its mechanism; a `workers: 1` label over a pool-width build is now a failing gate.
    assert!(
        pool >= 1,
        "the hashtrinity global pool must report a positive width"
    );
    // At the pool width on both sides the pair is comparable, so the concurrency verdict is
    // `Matched` rather than a mismatch the suite asserted as correct.
    assert_eq!(
        pair_concurrency(pool, Some(pool)),
        Concurrency::Matched,
        "two sides that both ran at the hashtrinity pool width are comparable"
    );
    // And a genuine width disagreement is still a mismatch: the check was not weakened to make the
    // combined arm pass.
    assert_eq!(
        pair_concurrency(SELECTED_WORKERS, Some(pool)),
        if pool == SELECTED_WORKERS {
            Concurrency::Matched
        } else {
            Concurrency::Mismatch
        },
        "a width disagreement must remain a mismatch"
    );
}

/// The fair-protocol premise: **no spatial incumbent's timed region builds, links or consults a
/// locator registry**. This is what makes excluding the selected side's `install_and_finalize`
/// bucket defensible, and it is checked against the adapter's own source rather than asserted.
///
/// The check is the transitive call closure of every registered arm body reached from `run_arm`, so
/// delegating the registry build to a helper does not hide it. An arm that gains any symbol in
/// [`INCUMBENT_REGISTRY_SYMBOLS`] fails here. The mutation control for it is an incumbent arm that
/// builds the excluded structure: see the receipt.
///
/// The second half is the non-vacuity check on this test itself: the selected side's closure *does*
/// contain registry symbols, so a scan that silently matched nothing cannot pass.
#[test]
fn no_incumbent_timed_region_builds_a_locator_registry() {
    const SOURCE: &str = include_str!("../adapters/spatial.rs");

    /// The body of every top-level `fn` in the source, keyed by name.
    fn bodies() -> HashMap<&'static str, &'static str> {
        let mut map = HashMap::new();
        let bytes = SOURCE.as_bytes();
        let mut at = 0usize;
        while let Some(found) = SOURCE[at..].find("fn ") {
            let name_start = at + found + 3;
            let name_end = name_start
                + SOURCE[name_start..]
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(0);
            let name = &SOURCE[name_start..name_end];
            let Some(open) = SOURCE[name_end..].find('{') else {
                at = name_end;
                continue;
            };
            let open = name_end + open;
            // Brace-matched body. Braces inside format strings in this file are balanced
            // (`{ident}`), so counting pairs is exact here.
            let mut depth = 0usize;
            let mut cursor = open;
            let mut close = open;
            while cursor < SOURCE.len() {
                match bytes[cursor] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            close = cursor + 1;
                            break;
                        }
                    }
                    _ => {}
                }
                cursor += 1;
            }
            map.entry(name).or_insert(&SOURCE[open..close]);
            at = close.max(name_end);
        }
        map
    }

    let bodies = bodies();

    /// The names of every `fn` called from `body`, restricted to locally defined ones.
    fn calls(
        body: &'static str,
        bodies: &HashMap<&'static str, &'static str>,
    ) -> Vec<&'static str> {
        let mut found = Vec::new();
        let bytes = body.as_bytes();
        let mut at = 0usize;
        while let Some(open) = body[at..].find('(') {
            let mut start = at + open;
            while start > 0
                && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_')
            {
                start -= 1;
            }
            let name = &body[start..at + open];
            if let Some(known) = bodies.get(name) {
                if *known == body {
                    at = at + open + 1;
                    continue;
                }
                found.push(name);
            }
            at = at + open + 1;
        }
        found.sort_unstable();
        found.dedup();
        found
    }

    /// The transitive local call closure of `root`, as one body.
    fn closure<'a>(root: &'a str, bodies: &'a HashMap<&'static str, &'static str>) -> String {
        let mut seen: Vec<&str> = Vec::new();
        let mut queue = vec![root];
        while let Some(name) = queue.pop() {
            if seen.contains(&name) {
                continue;
            }
            seen.push(name);
            let Some(body) = bodies.get(name) else {
                continue;
            };
            queue.extend(calls(body, bodies));
        }
        seen.sort_unstable();
        seen.iter()
            .filter_map(|name| bodies.get(name).copied())
            .collect::<Vec<&str>>()
            .join("\n")
    }

    // Each registered arm reaches its timed region through `run_arm`'s match arm.
    for id in SPATIAL_INCUMBENTS {
        let anchor = format!("(\"{id}\", KeyFace::");
        let arm_at = SOURCE
            .find(&anchor)
            .unwrap_or_else(|| panic!("run_arm has no arm for registered incumbent {id}"));
        let rest = &SOURCE[arm_at..];
        // The arm body is the call after this match arm's `=>`.
        let arrow = rest
            .find("=>")
            .unwrap_or_else(|| panic!("run_arm's arm for {id} has no `=>`"));
        let after = &rest[arrow + 2..];
        let open = after
            .find('(')
            .unwrap_or_else(|| panic!("run_arm's arm for {id} calls nothing"));
        let start = after[..open]
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(0, |at| at + 1);
        let root = &after[start..open];
        let region = closure(root, &bodies);
        assert!(
            !region.is_empty(),
            "the scan for {id} matched no body, so it cannot reject anything"
        );
        for symbol in INCUMBENT_REGISTRY_SYMBOLS {
            assert!(
                !region.contains(symbol),
                "incumbent {id} reaches {symbol} through {root}: its timed region builds or consults \
                 a locator registry, so the selected side's registry exclusion is not like-for-like \
                 for this arm"
            );
        }
    }

    // Non-vacuity: the same scan over the selected side does find the registry, so the loop above is
    // discriminating and not passing on an empty match.
    let selected = closure("run_selected_parallel", &bodies);
    assert!(
        INCUMBENT_REGISTRY_SYMBOLS
            .iter()
            .any(|symbol| selected.contains(symbol)),
        "the selected side's closure must reference a registry symbol, or this scan rejects nothing"
    );
}

/// The spatial pair's charged timed region is the elapsed region minus exactly the phase bucket the
/// build's own instrumentation measured, and the serial selected path excludes nothing because it
/// assembles no directory. Both halves are stated on the pair, so a ratio cannot be read without the
/// phases it covers.
#[test]
fn spatial_timed_region_is_recorded_and_names_both_halves() {
    let region = spatial::SELECTED_TIMED_REGION;
    assert!(
        region.starts_with("IN=[") && region.contains("OUT=["),
        "the selected spatial timed region must name both what it charges and what it excludes: \
         {region}"
    );
    for phase in [
        "qualify",
        "partition",
        "validate",
        "schedule",
        "reserve",
        "kernel",
    ] {
        assert!(
            region.contains(phase),
            "the charged region must name the {phase} phase: {region}"
        );
    }
    for out in [
        "install_and_finalize",
        "finalize_recorded_locations",
        "assemble_directory",
    ] {
        assert!(
            region.contains(out),
            "the excluded bucket must name {out}: {region}"
        );
    }
    assert!(
        SERIAL_TIMED_REGION.contains("no directory is assembled"),
        "the serial selected pair must record that it excludes nothing: {SERIAL_TIMED_REGION}"
    );
    // The registry bucket the parallel build excludes is a phase of the crate's own split, so the
    // exclusion is a measurement and not an adapter estimate.
    let phases = apeiron::table::parallel::PhaseNanos {
        qualify: 1,
        partition: 2,
        validate: 3,
        schedule: 4,
        reserve: 5,
        kernel: 6,
        assemble: 7,
    };
    assert_eq!(
        phases.total(),
        28,
        "PhaseNanos::total must sum every named phase, or the recorded region names a phase the \
         instrumentation does not measure"
    );
    assert_eq!(
        phases.assemble, 7,
        "the excluded bucket must be a field the build's own phase split reports"
    );
}

/// An omitted required pair is reported missing and rejects the campaign. A registry change
/// that drops a required `(cell, incumbent)` pair fails here.
#[test]
fn omitted_required_pair_is_reported_missing() {
    let mut campaign = synthetic_campaign();
    let before = campaign.pairs.len();
    campaign
        .pairs
        .retain(|pair| !(pair.cell_id == "synthetic-cell-a" && pair.incumbent_id == "xxhash_xxh3"));
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

/// Every registered `current_strata` row carries the four performance fields and the two revisions
/// the regression check reads, and every row names the incumbent this check compares against.
///
/// Without this the regression check would silently compare against fewer cells than the
/// registration carries, and a registration that dropped its medians would leave the comparison
/// empty rather than red.
#[test]
fn every_registered_receipt_carries_its_performance_fields() {
    let receipts = load_registered_receipts().expect("registered receipts must parse");
    assert!(
        !receipts.is_empty(),
        "the registered receipts are empty, so the regression check compares nothing"
    );
    let root = crate::Reader::new(super::MATRIX_JSON)
        .parse()
        .expect("matrix must parse");
    let crate::Json::Arr(entries) = crate::field(
        root.as_obj().expect("matrix is an object"),
        "current_strata",
    )
    .expect("current_strata field") else {
        panic!("current_strata is not an array");
    };
    assert_eq!(
        receipts.len(),
        entries.len(),
        "the regression check read a different number of receipts than the registration holds"
    );
    let mut ids = HashMap::new();
    for receipt in &receipts {
        assert!(
            ids.insert(receipt.id.as_str(), ()).is_none(),
            "registered cell {} appears twice in the performance receipts",
            receipt.id
        );
        assert!(
            receipt.apeiron_median_ns > 0 && receipt.ht_median_ns > 0,
            "registered cell {} carries a zero median ({} / {})",
            receipt.id,
            receipt.apeiron_median_ns,
            receipt.ht_median_ns
        );
        assert!(
            receipt.ratio.is_finite() && receipt.ratio > 0.0,
            "registered cell {} carries a non-positive ratio {}",
            receipt.id,
            receipt.ratio
        );
        assert_eq!(
            receipt.revision,
            super::matrix::REGISTERED_CURRENT_REVISION,
            "registered cell {} names candidate revision {}",
            receipt.id,
            receipt.revision
        );
        assert_eq!(
            receipt.baseline_revision,
            super::matrix::REGISTERED_BASELINE_REVISION,
            "registered cell {} names baseline revision {}",
            receipt.id,
            receipt.baseline_revision
        );
    }
    let incumbent = crate::str_field(
        &entries[0].as_obj().expect("receipt object"),
        "baseline_impl_id",
    )
    .expect("baseline_impl_id");
    assert_eq!(
        incumbent, REGRESSION_INCUMBENT,
        "the registration pairs its receipts with {incumbent}, not with {REGRESSION_INCUMBENT}, so \
         the regression check would compare the wrong incumbent"
    );
}

/// The tolerance cannot exceed what the check is able to detect: a three-fold seeded regression is
/// outside the widest tolerance either side can be granted, and a cell with no measurable spread
/// gets exactly its side's measured cross-run floor.
#[test]
fn regression_tolerance_cannot_reach_a_seeded_regression() {
    assert_eq!(
        tolerance(f64::NAN, REGRESSION_CANDIDATE_CROSS_RUN_FLOOR),
        REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
        "a cell with no measurable in-run spread must get its side's cross-run floor, not more"
    );
    assert_eq!(
        tolerance(1.0, REGRESSION_CANDIDATE_CROSS_RUN_FLOOR),
        REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
        "a cell whose campaigns agree must still get the cross-run floor"
    );
    assert_eq!(
        tolerance(1.0 + 0.05, REGRESSION_CANDIDATE_CROSS_RUN_FLOOR),
        REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
        "a spread below the floor must not widen the tolerance past the floor"
    );
    assert_eq!(
        tolerance(1.0 + 0.40, REGRESSION_CANDIDATE_CROSS_RUN_FLOOR),
        REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
        "a cell whose own spread is inside the cross-run floor must not narrow it"
    );
    assert!(
        (tolerance(1.0 + 0.95, REGRESSION_CANDIDATE_CROSS_RUN_FLOOR)
            - REGRESSION_SPREAD_FACTOR * 0.95)
            .abs()
            < 1e-12,
        "a cell that measured 95% of spread, above the floor, must be granted exactly that much, \
         found {}",
        tolerance(1.0 + 0.95, REGRESSION_CANDIDATE_CROSS_RUN_FLOOR)
    );
    assert_eq!(
        tolerance(1.0 + 5.0, REGRESSION_CANDIDATE_CROSS_RUN_FLOOR),
        REGRESSION_SPREAD_CEILING,
        "an unbounded spread must clamp to the ceiling, not grant the whole spread"
    );
    for floor in [
        REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
        REGRESSION_INCUMBENT_CROSS_RUN_FLOOR,
    ] {
        for spread in [1.0, 1.02, 1.10, 1.50, 2.0, 10.0] {
            let granted = tolerance(spread, floor);
            assert!(
                granted <= REGRESSION_SPREAD_CEILING,
                "tolerance {granted} for spread {spread} exceeds the ceiling \
                 {REGRESSION_SPREAD_CEILING}"
            );
            assert!(
                granted + 1.0 < 3.0,
                "tolerance {granted} for spread {spread} would tolerate a 3x seeded regression"
            );
        }
    }
    assert!(
        REGRESSION_INCUMBENT_CROSS_RUN_FLOOR > REGRESSION_CANDIDATE_CROSS_RUN_FLOOR,
        "the incumbent side measured a wider cross-run bound and must carry the wider floor"
    );
    assert!(
        REGRESSION_CANDIDATE_QUIET_FLOOR < REGRESSION_CANDIDATE_CROSS_RUN_FLOOR
            && REGRESSION_INCUMBENT_QUIET_FLOOR < REGRESSION_INCUMBENT_CROSS_RUN_FLOOR,
        "the non-gating band must sit strictly inside the gating tolerance on both sides"
    );
}

/// A pooled sample vector is only read as campaigns when its shape is the one `strata::run_cell`
/// produces, so the measured spread is never fabricated from a differently shaped vector.
#[test]
fn campaign_spread_requires_the_campaign_sample_shape() {
    let block = crate::strata::STRATA_SAMPLES as usize;
    let pooled: Vec<u64> = (0..block * 3).map(|i| 100 + (i % block) as u64).collect();
    let spread = campaign_median_spread(&pooled, 3).expect("three campaigns of 15 samples");
    // Three identical medians (each block is a uniform ramp, so each block median is the same).
    assert!(
        (spread - 1.0).abs() < f64::EPSILON,
        "three identical block medians must measure a spread of 1.0, found {spread}"
    );
    assert!(
        campaign_median_spread(&pooled[..block * 2], 3).is_none(),
        "a vector shorter than three campaigns must not yield a spread"
    );
    assert!(
        campaign_median_spread(&pooled, 2).is_none(),
        "a campaign count that disagrees with the vector length must not yield a spread"
    );
    assert!(
        campaign_median_spread(&vec![0u64; block * 3], 3).is_none(),
        "a zero-nanosecond block must not yield a spread"
    );
}

/// The verdict separation the check claims, as a plain test: the candidate is judged against the
/// candidate tolerance, the incumbent against the incumbent tolerance, and neither may decide the
/// other's verdict.
#[test]
fn regression_verdicts_separate_the_two_deltas() {
    let receipt = RegisteredReceipt {
        id: "registry-control-cell".to_string(),
        apeiron_median_ns: 1_000,
        ht_median_ns: 1_000,
        ratio: 1.0,
        revision: super::matrix::REGISTERED_CURRENT_REVISION.to_string(),
        baseline_revision: super::matrix::REGISTERED_BASELINE_REVISION.to_string(),
    };
    let block = crate::strata::STRATA_SAMPLES as usize;
    let campaign = |candidate: u64, incumbent: u64| -> super::Campaign {
        super::Campaign {
            cells: vec![receipt.id.clone()],
            pairs: vec![PairOutcome {
                cell_id: receipt.id.clone(),
                incumbent_id: REGRESSION_INCUMBENT,
                class: PairClass::LoadBearing,
                adapter: AdapterState::Built,
                applicability: Applicability::Applicable("registry control pair"),
                sel_impl: super::SELECTED_IMPL,
                sel_workers: SELECTED_WORKERS,
                sample_basis: "serial",
                inc_impl: REGRESSION_INCUMBENT,
                inc_workers: Some(1),
                address_face: AddressFace::Matched,
                address_reason: "registry control pair".to_string(),
                selected_address: "raw-word;direct_word=Some",
                oracle_digest: 0,
                expected_oracle_digest: 0,
                selected_samples: vec![candidate; block * 3],
                incumbent_samples: vec![incumbent; block * 3],
                campaign_ratios: vec![1.0; 3],
                timed_region: SERIAL_TIMED_REGION,
            }],
        }
    };
    let verdict = |campaign: super::Campaign| {
        let outcome = evaluate_regression(&campaign, std::slice::from_ref(&receipt), true);
        (outcome.cells[0].verdict, outcome)
    };

    let (unseeded, unseeded_outcome) = verdict(campaign(1_000, 1_000));
    assert_eq!(
        unseeded,
        super::RegressionVerdict::Within,
        "a cell reproducing its registered receipt must read within tolerance"
    );
    assert!(unseeded_outcome.passed() && unseeded_outcome.full_acceptance);

    let (candidate_3x, candidate_outcome) = verdict(campaign(3_000, 1_000));
    assert_eq!(
        candidate_3x,
        super::RegressionVerdict::Regressed,
        "a 3x candidate median is beyond any tolerance the check can grant"
    );
    assert!(
        !candidate_outcome.passed() && !candidate_outcome.full_acceptance,
        "a regressed cell must block the verdict"
    );
    assert!(
        candidate_outcome
            .failing
            .iter()
            .any(|line| line.contains("beyond its tolerance")),
        "a regressed cell must be reported by name with its numbers: {:?}",
        candidate_outcome.failing
    );

    let (incumbent_3x, incumbent_outcome) = verdict(campaign(1_000, 3_000));
    assert_eq!(
        incumbent_3x,
        super::RegressionVerdict::IncumbentSlower,
        "a 3x incumbent median must be reported as the incumbent's movement, not the candidate's"
    );
    assert_eq!(
        incumbent_outcome.regressed, 0,
        "the incumbent's movement must not be charged to the candidate"
    );
    assert_eq!(incumbent_outcome.incumbent_slower, 1);
    assert!(
        incumbent_outcome.passed(),
        "an incumbent-only movement is reported and does not block the verdict"
    );

    let (candidate_1_5x, band_outcome) = verdict(campaign(1_500, 1_000));
    assert_eq!(
        candidate_1_5x,
        super::RegressionVerdict::RegressedNoiseBand,
        "a 1.5x candidate median is inside the measured cross-run floor and must be named, not gated"
    );
    assert_eq!(band_outcome.regressed, 0);
    assert_eq!(band_outcome.noise_band(), 1);
    assert!(
        band_outcome.passed(),
        "a movement inside the cross-run floor must not block the verdict"
    );

    let mut foreign = receipt.clone();
    foreign.revision = "1111111111111111111111111111111111111111".to_string();
    let mismatched = evaluate_regression(
        &campaign(1_000, 1_000),
        std::slice::from_ref(&foreign),
        true,
    );
    assert_eq!(
        mismatched.cells[0].verdict,
        super::RegressionVerdict::RevisionMismatch,
        "a receipt recorded against another revision must be reported, not compared"
    );
    assert_eq!(
        mismatched.compared, 0,
        "a revision-mismatched receipt must not be compared at all"
    );
    assert!(
        !mismatched.passed(),
        "an incomplete comparison must not read as a pass"
    );
}
