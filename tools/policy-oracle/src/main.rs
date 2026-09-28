//! APEIRON-POLICY-W2-T2 — Cedar differential oracle over the real
//! `apeiron::policy::decide` (`docs/epics/policy-engine.md`, decision D6).
//!
//! **What is compared.** `apeiron::policy::decide` is run beside `cedar-policy` 4.13.0 on a
//! fixed, reproducible fragment, and the two decisions plus their named ids are compared.
//! Fragment: 1–8 rules, each `Permit` or `Forbid` (no `Ask`); action 7 (occasionally 8, to
//! exercise action filtering); `rights = 1`; `want = 1`; 0–3 atoms per rule with
//! `Op in {Eq, Le, Ge}`, slot in 0..4, value in 0..6. Facts: 4 slots, values 0..6,
//! `known = 0b1111` (all known). Excluded: `Ask`, attenuation, unknown facts, `All`/`Any`
//! atoms and Cedar evaluation errors. Each rule is a Cedar policy
//! `permit|forbid(principal, action == Action::"a<action>", resource) [when { … }];` with
//! policy id = rule id, request principal `User::"u"`, resource `Res::"r"` and an empty
//! entity store.
//!
//! **Seed.** `SEED = 0x9E37_79B9_7F4A_7C15`, printed as `seed=0x9E3779B97F4A7C15`. Generation
//! is a xorshift64, so a run is reproducible from the seed and the case count.
//!
//! **Documented divergence class (one).** The candidate returns `Deny(NoPermit)` while Cedar
//! returns `Deny` with a nonempty `reason`: when some forbid is satisfied but no permit is,
//! Cedar's `must_be_determining` reports the satisfied forbids even though the decision is the
//! default deny, and `Response::from` copies that under-approximation into `reason`
//! (`cedar-policy-core-4.13.0/src/authorizer/partial_response.rs:206-218` and `:473-487`; the
//! decision at `:122-135`; spike S2 read the same file at the then-resolved 4.11.2). The
//! candidate names `NoPermit`, the actual cause; Cedar's `reason` is an under-approximation of
//! the determining set, not a claim that the forbid caused the deny. This class is counted as
//! `documented_divergence`, never as a mismatch. Any other difference in decision or in the
//! minimum named id is a mismatch, and `main` exits non-zero.
//!
//! **Complexity (AGENTS §8).** Per case, with R ≤ 8 rules and A ≤ 3 atoms per rule:
//! generation and `decide` are O(R·(1+A)) atom evaluations; building the Cedar policies and
//! authorizing is O(R) policies of `cedar-policy`'s own cost, not measured here; the
//! comparison is O(R) over the reported reasons. Space is O(R·A) words for one case, O(1)
//! across cases. Over N cases: O(N·R·A) oracle work.
//!
//! Usage: `cargo run --release --offline -- [cases]` (default 20000). Exit 0 only when
//! `mismatches = 0`, `cedar_errors = 0`, `allow > 0` and `deny > 0`.

use std::str::FromStr;

use apeiron::policy::{decide, Atom, Cause, Effect, Op, Rule, Verdict};
use cedar_policy::{
    Authorizer, Context, Decision, Entities, EntityUid, Policy, PolicyId, PolicySet, Request,
};

/// Fixed seed for the xorshift64 generator; printed so a run is reproducible.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const NSLOTS: usize = 4;
/// Fact values and atom values live in `0..NVALS` (0..=6).
const NVALS: u64 = 7;

/// xorshift64 — deterministic, no external `rand` dependency.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x % n
    }
}

/// One generated case: the rule image, its shared atom pool and the four known facts.
struct Case {
    rules: Vec<Rule>,
    atoms: Vec<Atom>,
    facts: [u64; NSLOTS],
}

/// The fixed request entities.
struct Endpoints {
    principal: EntityUid,
    action: EntityUid,
    resource: EntityUid,
}

impl Endpoints {
    fn new() -> Self {
        Self {
            principal: EntityUid::from_str(r#"User::"u""#).expect("principal"),
            action: EntityUid::from_str(r#"Action::"a7""#).expect("action"),
            resource: EntityUid::from_str(r#"Res::"r""#).expect("resource"),
        }
    }
}

/// Class of one differential case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Agreement {
    /// Decision and named id agree.
    Match,
    /// Decision agrees; the named cause differs by the documented `NoPermit` class.
    DocumentedDivergence,
    /// Decision or minimum named id disagree.
    Mismatch,
}

/// Both decisions and the comparison outcome for one case.
struct Outcome {
    ours: Verdict,
    cedar: Decision,
    reasons: Vec<u32>,
    error_count: usize,
    agreement: Agreement,
}

/// Generate a case from the shrunken fragment. O(R·A) time and space.
fn gen_case(rng: &mut Rng) -> Case {
    // Distinct policy ids 1..=8, shuffled per case.
    let mut ids: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    for i in (1..8).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        ids.swap(i, j);
    }
    let nrules = 1 + rng.below(8) as usize; // 1..=8
    let mut rules: Vec<Rule> = Vec::with_capacity(nrules);
    let mut atoms: Vec<Atom> = Vec::new();
    for &id in &ids[..nrules] {
        let effect = if rng.below(2) == 0 {
            Effect::Permit
        } else {
            Effect::Forbid
        };
        let action = if rng.below(8) == 0 { 8 } else { 7 };
        let natoms = rng.below(4) as usize; // 0..=3
        let atom_start = atoms.len() as u32;
        for _ in 0..natoms {
            let op = match rng.below(3) {
                0 => Op::Eq,
                1 => Op::Le,
                _ => Op::Ge,
            };
            let slot = rng.below(4) as u32; // 0..=3, all known
            let value = rng.below(NVALS); // 0..=6
            atoms.push(Atom { op, slot, value });
        }
        rules.push(Rule {
            id,
            effect,
            action,
            rights: 1,
            atom_start,
            atom_len: natoms as u32,
        });
    }
    let mut facts = [0u64; NSLOTS];
    for f in &mut facts {
        *f = rng.below(NVALS);
    }
    Case {
        rules,
        atoms,
        facts,
    }
}

const KEYWORDS: [&str; 3] = ["permit", "forbid", "ask"];

fn effect_keyword(effect: Effect) -> &'static str {
    match effect {
        Effect::Permit => KEYWORDS[0],
        Effect::Forbid => KEYWORDS[1],
        Effect::Ask => KEYWORDS[2],
    }
}

/// Cedar translation of one rule. An empty guard means no `when` clause. O(A) time, O(A) space.
fn policy_text(rule: &Rule, atoms: &[Atom]) -> String {
    let action = format!("Action::\"a{}\"", rule.action);
    let start = rule.atom_start as usize;
    let len = rule.atom_len as usize;
    let guard_atoms = &atoms[start..start + len];
    if guard_atoms.is_empty() {
        return format!(
            "{}(principal, action == {action}, resource);",
            effect_keyword(rule.effect)
        );
    }
    let mut guard = String::new();
    for (i, a) in guard_atoms.iter().enumerate() {
        if i > 0 {
            guard.push_str(" && ");
        }
        let cmp = match a.op {
            Op::Eq => "==",
            Op::Le => "<=",
            Op::Ge => ">=",
            // Op::All / Op::Any are excluded from this fragment.
            Op::All | Op::Any => "==",
        };
        guard.push_str(&format!("context.f{} {cmp} {}", a.slot, a.value));
    }
    format!(
        "{}(principal, action == {action}, resource) when {{ {guard} }};",
        effect_keyword(rule.effect)
    )
}

/// The four facts as a Cedar context object. O(NSLOTS) time and space.
fn context_json(facts: &[u64; NSLOTS]) -> String {
    format!(
        r#"{{"f0":{},"f1":{},"f2":{},"f3":{}}}"#,
        facts[0], facts[1], facts[2], facts[3]
    )
}

/// Compare the two decisions and the minimum named id. O(1) space, O(1) time.
fn classify(ours: Verdict, cedar: Decision, cedar_min: Option<u32>) -> Agreement {
    match (ours, cedar) {
        (Verdict::Allow { by, .. }, Decision::Allow) => {
            if Some(by) == cedar_min {
                Agreement::Match
            } else {
                Agreement::Mismatch
            }
        }
        (Verdict::Deny(Cause::Forbid(id)), Decision::Deny) => {
            if cedar_min == Some(id) {
                Agreement::Match
            } else {
                Agreement::Mismatch
            }
        }
        (Verdict::Deny(Cause::NoPermit), Decision::Deny) => {
            if cedar_min.is_some() {
                // Documented divergence: Cedar's default deny names satisfied forbids; ours
                // names NoPermit. Not a mismatch.
                Agreement::DocumentedDivergence
            } else {
                Agreement::Match
            }
        }
        _ => Agreement::Mismatch,
    }
}

/// Run one case through `apeiron::policy::decide` and cedar-policy 4.13.0 and classify the
/// result. Time O(R·(1+A)) atom evaluations plus cedar-policy's parse/authorize cost; space
/// O(R·A) words for the policy set and this case.
fn evaluate_case(
    authorizer: &Authorizer,
    entities: &Entities,
    endpoints: &Endpoints,
    case: &Case,
) -> Outcome {
    let mut pset = PolicySet::new();
    for rule in &case.rules {
        let text = policy_text(rule, &case.atoms);
        let policy = Policy::parse(Some(PolicyId::new(rule.id.to_string())), &text)
            .unwrap_or_else(|e| panic!("policy parse error on rule {}: {e}\n{text}", rule.id));
        pset.add(policy).expect("distinct policy ids");
    }
    let ctx = context_json(&case.facts);
    let context = Context::from_json_str(&ctx, None).expect("context");
    let request = Request::new(
        endpoints.principal.clone(),
        endpoints.action.clone(),
        endpoints.resource.clone(),
        context,
        None,
    )
    .expect("request");
    let response = authorizer.is_authorized(&request, &pset, entities);

    let ours = decide(&case.rules, &case.atoms, &case.facts, 0b1111, 7, 1);
    let cedar = response.decision();
    let reasons: Vec<u32> = response
        .diagnostics()
        .reason()
        .map(|p| p.to_string().parse::<u32>().expect("numeric policy id"))
        .collect();
    let error_count = response.diagnostics().errors().count();
    let cedar_min = reasons.iter().copied().min();
    let agreement = classify(ours, cedar, cedar_min);
    Outcome {
        ours,
        cedar,
        reasons,
        error_count,
        agreement,
    }
}

fn main() {
    let cases: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);

    println!("seed=0x{SEED:016X}");
    println!("cases={cases}");

    let mut rng = Rng(SEED);
    let authorizer = Authorizer::new();
    let entities = Entities::empty();
    let endpoints = Endpoints::new();

    let mut allow = 0usize;
    let mut deny = 0usize;
    let mut mismatches = 0usize;
    let mut documented_divergence = 0usize;
    let mut cedar_errors = 0usize;

    for idx in 0..cases {
        let case = gen_case(&mut rng);
        let outcome = evaluate_case(&authorizer, &entities, &endpoints, &case);
        match outcome.ours {
            Verdict::Allow { .. } => allow += 1,
            Verdict::Deny(_) => deny += 1,
            _ => {}
        }
        cedar_errors += outcome.error_count;
        match outcome.agreement {
            Agreement::Match => {}
            Agreement::DocumentedDivergence => documented_divergence += 1,
            Agreement::Mismatch => {
                mismatches += 1;
                println!(
                    "MISMATCH case {idx}: ours={:?} cedar={:?} cedar_reasons={:?} facts={:?}",
                    outcome.ours, outcome.cedar, outcome.reasons, case.facts
                );
                for t in case.rules.iter().map(|r| policy_text(r, &case.atoms)) {
                    println!("  {t}");
                }
            }
        }
    }

    println!("allow={allow}");
    println!("deny={deny}");
    println!("mismatches={mismatches}");
    println!("documented_divergence={documented_divergence}");
    println!("cedar_errors={cedar_errors}");

    if mismatches > 0 || cedar_errors > 0 || allow == 0 || deny == 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(rules: Vec<Rule>, atoms: Vec<Atom>, facts: [u64; NSLOTS]) -> Outcome {
        let case = Case {
            rules,
            atoms,
            facts,
        };
        evaluate_case(
            &Authorizer::new(),
            &Entities::empty(),
            &Endpoints::new(),
            &case,
        )
    }

    /// The live differential run: ≥ 2,000 cases, no mismatch, no Cedar error, both verdicts.
    #[test]
    fn differential_2000_cases() {
        let authorizer = Authorizer::new();
        let entities = Entities::empty();
        let endpoints = Endpoints::new();
        let mut rng = Rng(SEED);
        let mut allow = 0usize;
        let mut deny = 0usize;
        let mut mismatches = 0usize;
        let mut cedar_errors = 0usize;
        for _ in 0..2_000 {
            let case = gen_case(&mut rng);
            let outcome = evaluate_case(&authorizer, &entities, &endpoints, &case);
            match outcome.ours {
                Verdict::Allow { .. } => allow += 1,
                Verdict::Deny(_) => deny += 1,
                _ => {}
            }
            cedar_errors += outcome.error_count;
            if outcome.agreement == Agreement::Mismatch {
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "mismatches over 2000 generated cases");
        assert_eq!(cedar_errors, 0, "cedar errors over 2000 generated cases");
        assert!(allow > 0, "no Allow verdict observed");
        assert!(deny > 0, "no Deny verdict observed");
    }

    /// Fixed case for the documented divergence class: a satisfied forbid, no satisfied
    /// permit. Ours `Deny(NoPermit)`; Cedar `Deny` naming the satisfied forbid 2.
    #[test]
    fn fixed_case_nopermit_naming_divergence() {
        let atoms = vec![
            Atom {
                op: Op::Eq,
                slot: 0,
                value: 1,
            },
            Atom {
                op: Op::Eq,
                slot: 0,
                value: 9,
            },
        ];
        let rules = vec![
            // Permit 1 guard is false (fact 0 is 1, not 9).
            Rule {
                id: 1,
                effect: Effect::Permit,
                action: 7,
                rights: 1,
                atom_start: 1,
                atom_len: 1,
            },
            // Forbid 2 guard is true.
            Rule {
                id: 2,
                effect: Effect::Forbid,
                action: 7,
                rights: 1,
                atom_start: 0,
                atom_len: 1,
            },
        ];
        let outcome = evaluate(rules, atoms, [1, 0, 0, 0]);
        assert_eq!(outcome.ours, Verdict::Deny(Cause::NoPermit));
        assert_eq!(outcome.cedar, Decision::Deny);
        assert_eq!(outcome.reasons, vec![2]);
        assert_eq!(outcome.agreement, Agreement::DocumentedDivergence);
    }

    /// Fixed cases on the agreement side: allow names the min satisfied permit; a satisfied
    /// forbid that removes the permitted bit names the min satisfied forbid.
    #[test]
    fn fixed_case_allow_and_forbid_agree() {
        let atoms = vec![Atom {
            op: Op::Eq,
            slot: 0,
            value: 1,
        }];
        let permit = Rule {
            id: 5,
            effect: Effect::Permit,
            action: 7,
            rights: 1,
            atom_start: 0,
            atom_len: 1,
        };
        let allowed = evaluate(vec![permit], atoms.clone(), [1, 0, 0, 0]);
        assert_eq!(allowed.ours, Verdict::Allow { granted: 1, by: 5 });
        assert_eq!(allowed.cedar, Decision::Allow);
        assert_eq!(allowed.agreement, Agreement::Match);

        let forbid = Rule {
            id: 8,
            effect: Effect::Forbid,
            action: 7,
            rights: 1,
            atom_start: 0,
            atom_len: 0,
        };
        let denied = evaluate(vec![permit, forbid], atoms, [1, 0, 0, 0]);
        assert_eq!(denied.ours, Verdict::Deny(Cause::Forbid(8)));
        assert_eq!(denied.cedar, Decision::Deny);
        assert_eq!(denied.reasons, vec![8]);
        assert_eq!(denied.agreement, Agreement::Match);
    }
}
