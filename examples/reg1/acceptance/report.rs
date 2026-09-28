use super::{
    pair_concurrency, AcceptanceOutcome, AdapterState, Campaign, PairStatus, INCUMBENTS,
    RATIO_LIMIT,
};
use crate::matrix::{median, ratio};

pub(super) fn print_registration() {
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

pub(super) fn print_acceptance(outcome: &AcceptanceOutcome) {
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
pub(super) fn print_ratio_summary(campaign: &Campaign) {
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
