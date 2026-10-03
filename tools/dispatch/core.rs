//! Zero-dependency admission state machine for verified tool dispatch.
//!
//! # Unforgeable Permit
//! External callers cannot construct a [`Permit`] directly:
//! ```compile_fail
//! use dispatch_core::{Digest, Permit};
//! let _ = Permit { identity: Digest { d0: 0, d1: 0, d2: 0, d3: 0 } };
//! ```

/// 256-bit hash represented as four 64-bit limbs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Digest {
    pub d0: u64,
    pub d1: u64,
    pub d2: u64,
    pub d3: u64,
}

impl Digest {
    /// O(1) comparison of all four limbs.
    #[inline]
    pub const fn same(self, other: Self) -> bool {
        self.d0 == other.d0 && self.d1 == other.d1 && self.d2 == other.d2 && self.d3 == other.d3
    }
}

/// Ordered stages required for execution admission.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Extraction,
    Dependencies,
    Certificate,
    Revalidation,
    Done,
}

/// Single stage observation recording exit code and evaluated identity.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub stage: Stage,
    pub exit: i32,
    pub identity: Digest,
}

/// Linear session tracking current admission state; non-Copy and non-Clone.
#[derive(Debug)]
pub struct Session {
    identity: Digest,
    next: Stage,
}

impl Session {
    #[inline]
    pub const fn identity(&self) -> Digest {
        self.identity
    }

    #[inline]
    pub const fn next(&self) -> Stage {
        self.next
    }
}

/// Proof of verified admission; non-Copy and non-Clone with no public constructor.
#[derive(Debug)]
pub struct Permit {
    identity: Digest,
}

impl Permit {
    #[inline]
    pub const fn identity(&self) -> Digest {
        self.identity
    }
}

/// O(1) session initialization; succeeds only when registration matches observed digest.
pub fn start(registered: Option<Digest>, observed: Digest) -> Option<Session> {
    match registered {
        Some(reg) => {
            if reg.same(observed) {
                Some(Session {
                    identity: observed,
                    next: Stage::Extraction,
                })
            } else {
                None
            }
        }
        None => None,
    }
}

/// O(1) state transition consuming the active session; succeeds only on expected stage, exit 0, and matching identity.
pub fn advance(session: Session, obs: Observation) -> Option<Session> {
    if obs.exit != 0 || !session.identity.same(obs.identity) {
        return None;
    }
    let next_stage = match session.next {
        Stage::Extraction => match obs.stage {
            Stage::Extraction => Stage::Dependencies,
            _ => return None,
        },
        Stage::Dependencies => match obs.stage {
            Stage::Dependencies => Stage::Certificate,
            _ => return None,
        },
        Stage::Certificate => match obs.stage {
            Stage::Certificate => Stage::Revalidation,
            _ => return None,
        },
        Stage::Revalidation => match obs.stage {
            Stage::Revalidation => Stage::Done,
            _ => return None,
        },
        Stage::Done => return None,
    };
    Some(Session {
        identity: session.identity,
        next: next_stage,
    })
}

/// O(1) completion consuming the active session; returns Permit only when Done and identity revalidated.
pub fn finish(session: Session, observed: Digest) -> Option<Permit> {
    match session.next {
        Stage::Done => {
            if session.identity.same(observed) {
                Some(Permit {
                    identity: session.identity,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// O(1) composite admission sequence chaining actual start, four advances, and finish.
#[allow(clippy::question_mark)]
pub fn complete(
    registered: Option<Digest>,
    initial_digest: Digest,
    obs0: Observation,
    obs1: Observation,
    obs2: Observation,
    obs3: Observation,
    final_digest: Digest,
) -> Option<Permit> {
    let s0 = match start(registered, initial_digest) {
        Some(s) => s,
        None => return None,
    };
    let s1 = match advance(s0, obs0) {
        Some(s) => s,
        None => return None,
    };
    let s2 = match advance(s1, obs1) {
        Some(s) => s,
        None => return None,
    };
    let s3 = match advance(s2, obs2) {
        Some(s) => s,
        None => return None,
    };
    let s4 = match advance(s3, obs3) {
        Some(s) => s,
        None => return None,
    };
    finish(s4, final_digest)
}

/// Representative reachable digest witness for specification and tests.
pub const WITNESS_DIGEST: Digest = Digest {
    d0: 0x0123_4567_89ab_cdef,
    d1: 0x1122_3344_5566_7788,
    d2: 0x99aa_bbcc_ddee_ff00,
    d3: 0xa1b2_c3d4_e5f6_0718,
};

/// Constant helper constructor for nominal witness observations.
#[inline]
pub const fn witness_obs(stage: Stage) -> Observation {
    Observation {
        stage,
        exit: 0,
        identity: WITNESS_DIGEST,
    }
}

/// Joint reachable witness term demonstrating nominal permit acquisition.
pub fn witness_success() -> Option<Permit> {
    complete(
        Some(WITNESS_DIGEST),
        WITNESS_DIGEST,
        witness_obs(Stage::Extraction),
        witness_obs(Stage::Dependencies),
        witness_obs(Stage::Certificate),
        witness_obs(Stage::Revalidation),
        WITNESS_DIGEST,
    )
}

/// Reachable witness term demonstrating fail-closed refusal on missing registration.
pub fn witness_refusal_unregistered() -> Option<Permit> {
    complete(
        None,
        WITNESS_DIGEST,
        witness_obs(Stage::Extraction),
        witness_obs(Stage::Dependencies),
        witness_obs(Stage::Certificate),
        witness_obs(Stage::Revalidation),
        WITNESS_DIGEST,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: Digest = WITNESS_DIGEST;

    fn o(stage: Stage, exit: i32, identity: Digest) -> Observation {
        Observation {
            stage,
            exit,
            identity,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        reg: Option<Digest>,
        init: Digest,
        e0: i32,
        id0: Digest,
        e1: i32,
        id1: Digest,
        e2: i32,
        id2: Digest,
        e3: i32,
        id3: Digest,
        fin: Digest,
    ) -> Option<Permit> {
        complete(
            reg,
            init,
            o(Stage::Extraction, e0, id0),
            o(Stage::Dependencies, e1, id1),
            o(Stage::Certificate, e2, id2),
            o(Stage::Revalidation, e3, id3),
            fin,
        )
    }

    #[test]
    fn test_nominal_and_witnesses() {
        let p = run(Some(D), D, 0, D, 0, D, 0, D, 0, D, D);
        assert_eq!(p.unwrap().identity(), D);
        assert_eq!(witness_success().unwrap().identity(), D);
        assert!(witness_refusal_unregistered().is_none());
    }

    #[test]
    fn test_empty_registry_and_mismatch() {
        assert!(start(None, D).is_none());
        assert!(run(None, D, 0, D, 0, D, 0, D, 0, D, D).is_none());
        let bad = Digest { d0: D.d0 ^ 1, ..D };
        assert!(start(Some(bad), D).is_none());
        assert!(run(Some(bad), D, 0, D, 0, D, 0, D, 0, D, D).is_none());
    }

    #[test]
    fn test_wrong_digest_limbs_all_four() {
        let b0 = Digest { d0: D.d0 ^ 1, ..D };
        let b1 = Digest { d1: D.d1 ^ 1, ..D };
        let b2 = Digest { d2: D.d2 ^ 1, ..D };
        let b3 = Digest { d3: D.d3 ^ 1, ..D };
        for bad in [b0, b1, b2, b3] {
            assert!(!D.same(bad));
            assert!(!bad.same(D));
            assert!(start(Some(D), bad).is_none());
            assert!(run(Some(D), bad, 0, D, 0, D, 0, D, 0, D, D).is_none());
            assert!(run(Some(D), D, 0, bad, 0, D, 0, D, 0, D, D).is_none());
            assert!(run(Some(D), D, 0, D, 0, bad, 0, D, 0, D, D).is_none());
            assert!(run(Some(D), D, 0, D, 0, D, 0, bad, 0, D, D).is_none());
            assert!(run(Some(D), D, 0, D, 0, D, 0, D, 0, bad, D).is_none());
            assert!(run(Some(D), D, 0, D, 0, D, 0, D, 0, D, bad).is_none());
        }
    }

    #[test]
    fn test_nonzero_exits_all_stages() {
        assert!(run(Some(D), D, 1, D, 0, D, 0, D, 0, D, D).is_none());
        assert!(run(Some(D), D, 0, D, 2, D, 0, D, 0, D, D).is_none());
        assert!(run(Some(D), D, 0, D, 0, D, 137, D, 0, D, D).is_none());
        assert!(run(Some(D), D, 0, D, 0, D, 0, D, -1, D, D).is_none());
    }

    #[test]
    fn test_missing_and_reorder_stages() {
        let s0 = start(Some(D), D).unwrap();
        assert!(advance(s0, o(Stage::Dependencies, 0, D)).is_none());

        let s0 = start(Some(D), D).unwrap();
        let s1 = advance(s0, o(Stage::Extraction, 0, D)).unwrap();
        assert!(advance(s1, o(Stage::Certificate, 0, D)).is_none());

        let s0 = start(Some(D), D).unwrap();
        let s1 = advance(s0, o(Stage::Extraction, 0, D)).unwrap();
        let s2 = advance(s1, o(Stage::Dependencies, 0, D)).unwrap();
        assert!(advance(s2, o(Stage::Revalidation, 0, D)).is_none());

        let s0 = start(Some(D), D).unwrap();
        let s1 = advance(s0, o(Stage::Extraction, 0, D)).unwrap();
        let s2 = advance(s1, o(Stage::Dependencies, 0, D)).unwrap();
        let s3 = advance(s2, o(Stage::Certificate, 0, D)).unwrap();
        assert!(advance(s3, o(Stage::Extraction, 0, D)).is_none());
    }

    #[test]
    fn test_replay_and_unexpected_done() {
        let s0 = start(Some(D), D).unwrap();
        let s1 = advance(s0, o(Stage::Extraction, 0, D)).unwrap();
        assert!(advance(s1, o(Stage::Extraction, 0, D)).is_none());

        let s0 = start(Some(D), D).unwrap();
        let s1 = advance(s0, o(Stage::Extraction, 0, D)).unwrap();
        let s2 = advance(s1, o(Stage::Dependencies, 0, D)).unwrap();
        let s3 = advance(s2, o(Stage::Certificate, 0, D)).unwrap();
        let s4 = advance(s3, o(Stage::Revalidation, 0, D)).unwrap();
        assert_eq!(s4.next(), Stage::Done);
        assert_eq!(s4.identity(), D);
        assert!(advance(s4, o(Stage::Done, 0, D)).is_none());
    }

    #[test]
    fn test_stale_final_and_premature_finish() {
        let s0 = start(Some(D), D).unwrap();
        assert!(finish(s0, D).is_none());

        let s0 = start(Some(D), D).unwrap();
        let s1 = advance(s0, o(Stage::Extraction, 0, D)).unwrap();
        let s2 = advance(s1, o(Stage::Dependencies, 0, D)).unwrap();
        let s3 = advance(s2, o(Stage::Certificate, 0, D)).unwrap();
        let s4 = advance(s3, o(Stage::Revalidation, 0, D)).unwrap();
        let bad = Digest { d3: D.d3 ^ 7, ..D };
        assert!(finish(s4, bad).is_none());
    }

    #[test]
    fn test_failure_session_consumed() {
        let s0 = start(Some(D), D).unwrap();
        let res = advance(s0, o(Stage::Extraction, 1, D));
        assert!(res.is_none());
    }
}
