//! Bounded authorization decision core.
//!
//! Canonical target: `Apeiron.Policy.decide` (`proof/corpus/Policy/Core.lean`). This body is
//! extracted by Charon/Aeneas (`proof/extract/lib.rs`, `proof/regen.sh`) and its extracted form is
//! refined in `proof/corpus/Policy/Refinement.lean`.
//!
//! Loops are `while i < len` with index access after the bound check; no iterators, closures, traits
//! or allocation, so the body stays inside the Aeneas fragment witnessed in `proof/ApeironWord.lean`.
//!
//! Complexity ([`decide`]): time O(R·(1+A)) atom evaluations and O(1) words of space, where
//! R = `rules.len()` and A is the longest guard. Derivation: one pass over `rules`, each relevant
//! rule costing one `guard_eval` pass over its atom range, plus a second pass over `rules` only when
//! the grant is zero and some permit was present; each atom evaluation is O(1) (one bound check, one
//! shift, one comparison). The measure is atom evaluations, not comparisons.

/// Atom comparison.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// `fact == value`
    Eq,
    /// `fact <= value`
    Le,
    /// `fact >= value`
    Ge,
    /// `value ⊆ fact` (every bit of `value` set in `fact`)
    All,
    /// `value ∩ fact ≠ ∅`
    Any,
}

/// Rule effect.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Effect {
    /// Contributes its rights to the permitted set when the guard is true.
    Permit,
    /// Removes its rights from the grant when the guard is true or unknown.
    Forbid,
    /// Marks its rights pending when the guard is true or unknown.
    Ask,
}

/// One guard literal over fact slot `slot`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Atom {
    /// Comparison applied to the fact slot.
    pub op: Op,
    /// Fact slot index; `slot >= 64` is unknown.
    pub slot: u32,
    /// Literal compared against the fact.
    pub value: u64,
}

/// One rule; its guard is the conjunction of `atoms[atom_start .. atom_start + atom_len]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rule {
    /// Rule identity; the verdict names the minimum id of the contributing set.
    pub id: u32,
    /// Effect applied when the guard is not false.
    pub effect: Effect,
    /// Request action this rule applies to.
    pub action: u32,
    /// Rights mask this rule permits, forbids or defers.
    pub rights: u64,
    /// Index of the first atom in the guard.
    pub atom_start: u32,
    /// Number of atoms in the guard.
    pub atom_len: u32,
}

/// Why a request is denied.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cause {
    /// The minimum id of an applicable forbid that removed permitted rights.
    Forbid(u32),
    /// No relevant permit granted any requested right.
    NoPermit,
}

/// Decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// Every requested right is granted.
    Allow {
        /// The granted rights.
        granted: u64,
        /// The minimum id of the contributing permit.
        by: u32,
    },
    /// A strict nonempty subset of the requested rights is granted.
    Attenuate {
        /// The granted rights.
        granted: u64,
        /// The minimum id of the contributing permit.
        by: u32,
    },
    /// The request awaits approval on `pending` bits of the granted rights.
    Ask {
        /// The granted rights.
        granted: u64,
        /// The granted bits that are pending.
        pending: u64,
        /// The minimum id of the contributing ask rule.
        by: u32,
    },
    /// Nothing is granted.
    Deny(Cause),
}

/// Kleene truth value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tri {
    T,
    F,
    U,
}

fn atom_eval(atom: &Atom, facts: &[u64], known: u64) -> Tri {
    let slot = atom.slot as usize;
    if atom.slot >= 64 || slot >= facts.len() {
        return Tri::U;
    }
    if (known >> atom.slot) & 1 == 0 {
        return Tri::U;
    }
    let fact = facts[slot];
    // Each arm decides with its own `if`: a `bool`-valued `match` over comparisons extracts to a
    // `Prop`-typed bind that has no `Decidable` instance under the pinned Aeneas model (spike S1).
    match atom.op {
        Op::Eq => {
            if fact == atom.value {
                Tri::T
            } else {
                Tri::F
            }
        }
        Op::Le => {
            if fact <= atom.value {
                Tri::T
            } else {
                Tri::F
            }
        }
        Op::Ge => {
            if fact >= atom.value {
                Tri::T
            } else {
                Tri::F
            }
        }
        Op::All => {
            if fact & atom.value == atom.value {
                Tri::T
            } else {
                Tri::F
            }
        }
        Op::Any => {
            if fact & atom.value != 0 {
                Tri::T
            } else {
                Tri::F
            }
        }
    }
}

/// Kleene conjunction of the rule's atom range; an out-of-range range is unknown.
///
/// Time `O(atom_len)` atom evaluations, space O(1): one pass over the range.
fn guard_eval(rule: &Rule, atoms: &[Atom], facts: &[u64], known: u64) -> Tri {
    let start = rule.atom_start as usize;
    let len = rule.atom_len as usize;
    if start > atoms.len() || len > atoms.len() - start {
        return Tri::U;
    }
    let mut acc = Tri::T;
    let mut i = 0usize;
    while i < len {
        match atom_eval(&atoms[start + i], facts, known) {
            Tri::F => return Tri::F,
            Tri::U => acc = Tri::U,
            Tri::T => {}
        }
        i += 1;
    }
    acc
}

// Kept in the candidate's `Option`-returning shape: it mirrors the Lean `Option Nat` min-fold, and
// the extraction is revalidated at W2.
#[allow(
    clippy::unnecessary_wraps,
    reason = "Option shape mirrors the Lean min-fold; W2 revalidates"
)]
fn min_id(best: Option<u32>, id: u32) -> Option<u32> {
    match best {
        None => Some(id),
        Some(b) => {
            if id < b {
                Some(id)
            } else {
                Some(b)
            }
        }
    }
}

/// Decide `want` rights for `action` under `rules` and the known facts.
///
/// Time O(R·(1+A)) atom evaluations, space O(1) words, where R = `rules.len()` and A is the
/// longest guard: two passes over `rules` (the second only names the forbid cause), each relevant
/// rule costing one `guard_eval`. Total over every input; no panic path.
// The `Option` matches below are kept in the candidate's extraction-safe form (spike S1); the
// lint-only rewrite to `unwrap_or_default` is revalidated with the extracted body at W2.
#[allow(
    clippy::manual_unwrap_or_default,
    clippy::manual_unwrap_or,
    reason = "extraction-safe Option match kept for the W2 fragment"
)]
#[must_use]
pub fn decide(
    rules: &[Rule],
    atoms: &[Atom],
    facts: &[u64],
    known: u64,
    action: u32,
    want: u64,
) -> Verdict {
    let mut permit: u64 = 0;
    let mut forbid: u64 = 0;
    let mut ask: u64 = 0;
    let mut permit_by: Option<u32> = None;
    let mut ask_by: Option<u32> = None;
    let mut i = 0usize;
    while i < rules.len() {
        let r = rules[i];
        if r.action == action {
            let g = guard_eval(&r, atoms, facts, known);
            match r.effect {
                Effect::Permit => {
                    if g == Tri::T && r.rights & want != 0 {
                        permit |= r.rights;
                        permit_by = min_id(permit_by, r.id);
                    }
                }
                Effect::Forbid => {
                    if g != Tri::F {
                        forbid |= r.rights;
                    }
                }
                Effect::Ask => {
                    if g != Tri::F {
                        ask |= r.rights;
                        ask_by = min_id(ask_by, r.id);
                    }
                }
            }
        }
        i += 1;
    }
    let granted = want & permit & !forbid;
    if granted == 0 {
        if want & permit == 0 {
            return Verdict::Deny(Cause::NoPermit);
        }
        let mut forbid_by: Option<u32> = None;
        let mut j = 0usize;
        while j < rules.len() {
            let r = rules[j];
            if r.action == action
                && r.effect == Effect::Forbid
                && r.rights & want & permit != 0
                && guard_eval(&r, atoms, facts, known) != Tri::F
            {
                forbid_by = min_id(forbid_by, r.id);
            }
            j += 1;
        }
        return match forbid_by {
            Some(id) => Verdict::Deny(Cause::Forbid(id)),
            None => Verdict::Deny(Cause::NoPermit),
        };
    }
    let by = match permit_by {
        Some(id) => id,
        None => 0,
    };
    if granted & ask != 0 {
        let a = match ask_by {
            Some(id) => id,
            None => 0,
        };
        return Verdict::Ask {
            granted,
            pending: granted & ask,
            by: a,
        };
    }
    if granted == want {
        Verdict::Allow { granted, by }
    } else {
        Verdict::Attenuate { granted, by }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const READ: u64 = 1;
    const WRITE: u64 = 2;

    fn rule(id: u32, effect: Effect, rights: u64, start: u32, len: u32) -> Rule {
        Rule {
            id,
            effect,
            action: 7,
            rights,
            atom_start: start,
            atom_len: len,
        }
    }

    #[test]
    fn default_deny_names_no_permit() {
        assert_eq!(
            decide(&[], &[], &[], 0, 7, READ),
            Verdict::Deny(Cause::NoPermit)
        );
    }

    #[test]
    fn allow_attenuate_ask_deny_are_reachable() {
        let atoms = [Atom {
            op: Op::Eq,
            slot: 0,
            value: 1,
        }];
        let p = rule(3, Effect::Permit, READ | WRITE, 0, 1);
        assert_eq!(
            decide(&[p], &atoms, &[1], 1, 7, READ),
            Verdict::Allow {
                granted: READ,
                by: 3
            }
        );
        let f = rule(9, Effect::Forbid, WRITE, 0, 0);
        assert_eq!(
            decide(&[p, f], &atoms, &[1], 1, 7, READ | WRITE),
            Verdict::Attenuate {
                granted: READ,
                by: 3
            }
        );
        let a = rule(5, Effect::Ask, READ, 0, 0);
        assert_eq!(
            decide(&[p, a], &atoms, &[1], 1, 7, READ),
            Verdict::Ask {
                granted: READ,
                pending: READ,
                by: 5
            }
        );
        assert_eq!(
            decide(&[p, f], &atoms, &[1], 1, 7, WRITE),
            Verdict::Deny(Cause::Forbid(9))
        );
    }

    #[test]
    fn unknown_forbid_applies_and_unknown_permit_is_inert() {
        let atoms = [Atom {
            op: Op::Eq,
            slot: 0,
            value: 1,
        }];
        let p_unknown = rule(1, Effect::Permit, READ, 0, 1);
        assert_eq!(
            decide(&[p_unknown], &atoms, &[1], 0, 7, READ),
            Verdict::Deny(Cause::NoPermit)
        );
        let p = rule(1, Effect::Permit, READ, 0, 0);
        let f_unknown = rule(2, Effect::Forbid, READ, 0, 1);
        assert_eq!(
            decide(&[p, f_unknown], &atoms, &[], 0, 7, READ),
            Verdict::Deny(Cause::Forbid(2))
        );
    }

    #[test]
    fn out_of_range_atoms_are_unknown_not_panic() {
        let p = rule(1, Effect::Permit, READ, u32::MAX, u32::MAX);
        assert_eq!(
            decide(&[p], &[], &[], u64::MAX, 7, READ),
            Verdict::Deny(Cause::NoPermit)
        );
        let atoms = [Atom {
            op: Op::Eq,
            slot: 64,
            value: 0,
        }];
        let p64 = rule(1, Effect::Permit, READ, 0, 1);
        assert_eq!(
            decide(&[p64], &atoms, &[0; 65], u64::MAX, 7, READ),
            Verdict::Deny(Cause::NoPermit)
        );
    }

    #[test]
    fn order_independent_with_min_cause() {
        let p = rule(4, Effect::Permit, READ, 0, 0);
        let f1 = rule(8, Effect::Forbid, READ, 0, 0);
        let f2 = rule(6, Effect::Forbid, READ, 0, 0);
        let a = decide(&[p, f1, f2], &[], &[], 0, 7, READ);
        let b = decide(&[f2, p, f1], &[], &[], 0, 7, READ);
        assert_eq!(a, b);
        assert_eq!(a, Verdict::Deny(Cause::Forbid(6)));
    }

    #[test]
    fn zero_want_is_denied() {
        let p = rule(1, Effect::Permit, READ | WRITE, 0, 0);
        assert_eq!(
            decide(&[p], &[], &[], 0, 7, 0),
            Verdict::Deny(Cause::NoPermit)
        );
        assert_eq!(
            decide(&[], &[], &[], u64::MAX, 7, 0),
            Verdict::Deny(Cause::NoPermit)
        );
    }

    #[test]
    fn duplicate_rule_ids_are_allowed() {
        let p1 = rule(3, Effect::Permit, READ, 0, 0);
        let p2 = rule(3, Effect::Permit, WRITE, 0, 0);
        assert_eq!(
            decide(&[p1, p2], &[], &[], 0, 7, READ | WRITE),
            Verdict::Allow {
                granted: READ | WRITE,
                by: 3
            }
        );
        let f1 = rule(5, Effect::Forbid, READ, 0, 0);
        let f2 = rule(5, Effect::Forbid, READ, 0, 0);
        assert_eq!(
            decide(&[p1, f1, f2], &[], &[], 0, 7, READ),
            Verdict::Deny(Cause::Forbid(5))
        );
    }

    #[test]
    fn other_actions_are_ignored() {
        let other = Rule {
            id: 1,
            effect: Effect::Permit,
            action: 8,
            rights: READ,
            atom_start: 0,
            atom_len: 0,
        };
        assert_eq!(
            decide(&[other], &[], &[], 0, 7, READ),
            Verdict::Deny(Cause::NoPermit)
        );
        let f = rule(2, Effect::Forbid, READ, 0, 0);
        let p = rule(3, Effect::Permit, READ, 0, 0);
        assert_eq!(
            decide(&[p, f], &[], &[], 0, 8, READ),
            Verdict::Deny(Cause::NoPermit)
        );
    }

    #[test]
    fn max_rights_and_want() {
        let p = rule(1, Effect::Permit, u64::MAX, 0, 0);
        assert_eq!(
            decide(&[p], &[], &[], 0, 7, u64::MAX),
            Verdict::Allow {
                granted: u64::MAX,
                by: 1
            }
        );
        let f = rule(2, Effect::Forbid, u64::MAX, 0, 0);
        assert_eq!(
            decide(&[p, f], &[], &[], 0, 7, u64::MAX),
            Verdict::Deny(Cause::Forbid(2))
        );
    }

    #[test]
    fn slot_63_known_and_slot_64_unknown() {
        let atom = [Atom {
            op: Op::Eq,
            slot: 63,
            value: 1,
        }];
        let p = rule(1, Effect::Permit, READ, 0, 1);
        let mut facts = [0u64; 64];
        facts[63] = 1;
        assert_eq!(
            decide(&[p], &atom, &facts, 1u64 << 63, 7, READ),
            Verdict::Allow {
                granted: READ,
                by: 1
            }
        );
        assert_eq!(
            decide(&[p], &atom, &facts, 0, 7, READ),
            Verdict::Deny(Cause::NoPermit)
        );
        let atom64 = [Atom {
            op: Op::Eq,
            slot: 64,
            value: 1,
        }];
        assert_eq!(
            decide(&[p], &atom64, &facts, u64::MAX, 7, READ),
            Verdict::Deny(Cause::NoPermit)
        );
    }

    #[test]
    fn ask_precedes_partial_grant() {
        let p = rule(3, Effect::Permit, READ, 0, 0);
        let a = rule(4, Effect::Ask, READ, 0, 0);
        assert_eq!(
            decide(&[p, a], &[], &[], 0, 7, READ | WRITE),
            Verdict::Ask {
                granted: READ,
                pending: READ,
                by: 4
            }
        );
        assert_eq!(
            decide(&[p], &[], &[], 0, 7, READ | WRITE),
            Verdict::Attenuate {
                granted: READ,
                by: 3
            }
        );
    }

    #[test]
    fn forbid_on_disjoint_rights_attenuates() {
        let p = rule(3, Effect::Permit, READ, 0, 0);
        let f = rule(9, Effect::Forbid, WRITE, 0, 0);
        assert_eq!(
            decide(&[p, f], &[], &[], 0, 7, READ | WRITE),
            Verdict::Attenuate {
                granted: READ,
                by: 3
            }
        );
    }
}
