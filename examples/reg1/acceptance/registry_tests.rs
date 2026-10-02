use super::{
    evaluate, is_spatial_incumbent, pair_concurrency, serial_address_face, spatial_address_face,
    spatial_incumbent_workers, synthetic_campaign, AdapterState, AddressFace, Applicability,
    Campaign, Concurrency, PairClass, PairOutcome, INCUMBENTS, INFORMATIONAL_COMPARATORS,
    RATIO_LIMIT, REQUIRED_LOAD_BEARING, SELECTED_IMPL, SELECTED_WORKERS, SPATIAL_INCUMBENTS,
};
use crate::performance::spatial::{cell_id, SpatialCell, SpatialWitness, SPATIAL_CELLS};

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

/// Build a spatial witness and campaign for exactly the required cells: one `SpatialCell` per
/// `SPATIAL_CELLS` entry (empty outcome lists — the registration contract reads the cells and the
/// campaign pairs, not the adapter outcomes), and the five spatial incumbents applicable on each.
fn synthetic_spatial_witness_and_campaign() -> (SpatialWitness, Campaign) {
    let registered = super::load_registered_spatial_cells().expect("spatial registration parses");
    let mut witness_cells = Vec::new();
    let mut cells = Vec::new();
    let mut pairs = Vec::new();
    for &(size, face) in SPATIAL_CELLS.iter() {
        let id = cell_id(size, face);
        let reg = registered.iter().find(|r| r.id == id).unwrap_or_else(|| {
            panic!("measured spatial cell {id} is absent from the registration")
        });
        assert_eq!(reg.key_face, face.name(), "spatial cell {id} key face");
        assert_eq!(
            reg.population,
            size.to_string(),
            "spatial cell {id} population"
        );
        witness_cells.push(SpatialCell {
            cell_id: id.clone(),
            workload: face.workload(),
            key_face: face.name(),
            population: size,
            ops: size,
            oracle_digest: 0,
            outcomes: Vec::new(),
        });
        cells.push(id.clone());
        for &incumbent in SPATIAL_INCUMBENTS.iter() {
            pairs.push(PairOutcome {
                cell_id: id.clone(),
                incumbent_id: incumbent,
                class: PairClass::LoadBearing,
                adapter: AdapterState::Built,
                applicability: Applicability::Applicable("synthetic spatial registration pair"),
                sel_impl: SELECTED_IMPL,
                sel_workers: SELECTED_WORKERS,
                sample_basis: "parallel",
                inc_impl: incumbent,
                inc_workers: Some(1),
                address_face: AddressFace::Matched,
                selected_address: "spread-word;direct_word=mixed",
                oracle_digest: 0,
                expected_oracle_digest: 0,
                selected_samples: Vec::new(),
                incumbent_samples: Vec::new(),
                campaign_ratios: Vec::new(),
            });
        }
    }
    (
        SpatialWitness {
            cells: witness_cells,
            campaigns: 1,
            warmup: 1,
            samples: 1,
        },
        Campaign { cells, pairs },
    )
}

/// The registered `matrix.json` `spatial_cells` must equal the harness's required cells, and the
/// five spatial incumbents must be applicable on every one. A divergence or a demoted incumbent
/// fails here.
#[test]
fn required_spatial_cells_are_registered_and_applicable() {
    let registered = super::load_registered_spatial_cells().expect("spatial registration parses");
    assert_eq!(
        registered.len(),
        SPATIAL_CELLS.len(),
        "registered spatial cell count"
    );
    let (witness, campaign) = synthetic_spatial_witness_and_campaign();
    super::validate_spatial_registration(&witness, &campaign)
        .expect("every required spatial cell is registered with its five incumbents applicable");
    for &(size, face) in SPATIAL_CELLS.iter() {
        let id = cell_id(size, face);
        for &incumbent in SPATIAL_INCUMBENTS.iter() {
            let pair = campaign
                .pairs
                .iter()
                .find(|p| p.cell_id == id && p.incumbent_id == incumbent)
                .unwrap_or_else(|| panic!("spatial cell {id} is missing incumbent {incumbent}"));
            assert!(
                matches!(pair.applicability, Applicability::Applicable(_)),
                "spatial cell {id} incumbent {incumbent} must be applicable"
            );
        }
    }
}

/// A required spatial cell absent from the run is rejected by name. The `matrix.json` registration
/// counterpart — a measured cell absent from the registration — is the byte-identical delete
/// control; this test covers the symmetric direction the unit harness can construct.
#[test]
fn registered_spatial_cell_absent_from_run_is_rejected_by_name() {
    let (witness, campaign) = synthetic_spatial_witness_and_campaign();
    super::validate_spatial_registration(&witness, &campaign)
        .expect("the full synthetic spatial registration passes");
    let removed = witness.cells[0].cell_id.clone();
    let mut shortened = witness;
    shortened.cells.retain(|cell| cell.cell_id != removed);
    let error = super::validate_spatial_registration(&shortened, &campaign)
        .expect_err("a registered cell absent from the run must reject");
    assert!(
        error.contains(&removed),
        "rejection must name the absent cell {removed}: {error}"
    );
}
