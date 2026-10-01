//! Pure artifact-admission core for `tools/gate`.
//!
//! The dispatch manifest (`docs/artifacts.toml`) names the artifacts allowed to exist; a worker
//! submission enters the indexed receipt store only when it is registered, its payload digest
//! matches the registered identity, and every reference it carries resolves to a registered
//! artifact. This module decides that admission.
//!
//! The body is pure: no IO, no hashing, no clock, no allocation beyond owned identifiers. The
//! adapter supplies a parsed [`Manifest`] and the submitted [`Submitted`] record with digests
//! already computed; the adapter's filesystem/hash/subprocess boundary is stated there. Purity is
//! deliberate so the real body can be extracted with Charon and refined in Lean, following the
//! `tools/dispatch/{core.rs,main.rs}` split (`tools/dispatch/admit.py` is the thin entry).
//!
//! A finding of a theorem's name establishes nothing; this core admits a *record*, and the record's
//! named acceptance gate establishes its result.

/// A payload identity as the adapter prints it, for example `sha256:<hex>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Digest(pub String);

/// One artifact the manifest registers: its id, the digest of its payload, and the artifacts it
/// references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registered {
    pub id: String,
    pub digest: Digest,
    pub refs: Vec<String>,
}

/// The set of artifacts a submission may enter as.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Manifest {
    pub artifacts: Vec<Registered>,
}

/// A worker submission: the artifact it claims to be, the digest of what it wrote, and the
/// artifacts it references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submitted {
    pub id: String,
    pub digest: Digest,
    pub refs: Vec<String>,
}

/// Why a submission is refused. One variant per required condition, so a refusal names its cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The submitted id is not registered in the manifest.
    Unregistered(String),
    /// The submitted payload digest differs from the registered identity.
    IdentityMismatch {
        id: String,
        expected: String,
        actual: String,
    },
    /// A reference does not resolve to a registered artifact.
    DanglingReference { from: String, to: String },
}

impl Manifest {
    /// The registered artifact with this id, if any. Linear in the artifact count; the manifest is
    /// the size of the dispatch, not of the data.
    fn lookup(&self, id: &str) -> Option<&Registered> {
        self.artifacts.iter().find(|a| a.id == id)
    }
}

/// Decide admission. Returns every refusal; an empty vector admits.
///
/// Refuses, and reports all causes it can: an id absent from the manifest; a payload digest that
/// differs from the registered identity; and every reference that does not resolve to a registered
/// artifact. A submission is admitted only when it is registered, its digest equals the registered
/// digest, and every reference resolves. An unregistered id returns immediately, because no
/// registered identity exists to compare against.
pub fn admit(manifest: &Manifest, submitted: &Submitted) -> Vec<Refusal> {
    let mut refusals = Vec::new();

    match manifest.lookup(&submitted.id) {
        None => {
            refusals.push(Refusal::Unregistered(submitted.id.clone()));
            return refusals;
        }
        Some(registered) => {
            if registered.digest.0 != submitted.digest.0 {
                refusals.push(Refusal::IdentityMismatch {
                    id: submitted.id.clone(),
                    expected: registered.digest.0.clone(),
                    actual: submitted.digest.0.clone(),
                });
            }
        }
    }

    for reference in &submitted.refs {
        if manifest.lookup(reference).is_none() {
            refusals.push(Refusal::DanglingReference {
                from: submitted.id.clone(),
                to: reference.clone(),
            });
        }
    }

    refusals
}

#[cfg(test)]
mod controls {
    use super::*;

    fn digest(hex: &str) -> Digest {
        Digest(format!("sha256:{hex}"))
    }

    fn manifest() -> Manifest {
        Manifest {
            artifacts: vec![
                Registered {
                    id: "docs/plan.md".into(),
                    digest: digest("aa11"),
                    refs: vec![],
                },
                Registered {
                    id: "artifacts/reads/0001".into(),
                    digest: digest("bb22"),
                    refs: vec!["docs/plan.md".into()],
                },
            ],
        }
    }

    /// A registered artifact with the registered digest and resolvable references is admitted.
    #[test]
    fn registered_artifact_passes() {
        let submitted = Submitted {
            id: "artifacts/reads/0001".into(),
            digest: digest("bb22"),
            refs: vec!["docs/plan.md".into()],
        };
        assert_eq!(admit(&manifest(), &submitted), Vec::new());
    }

    /// An artifact the manifest does not register is refused, naming the id.
    #[test]
    fn unregistered_artifact_fails() {
        let submitted = Submitted {
            id: "artifacts/reads/0002".into(),
            digest: digest("cc33"),
            refs: vec![],
        };
        assert_eq!(
            admit(&manifest(), &submitted),
            vec![Refusal::Unregistered("artifacts/reads/0002".into())]
        );
    }

    /// A registered id with a payload digest other than the registered one is refused, naming both.
    #[test]
    fn identity_mismatch_fails() {
        let submitted = Submitted {
            id: "artifacts/reads/0001".into(),
            digest: digest("dead"),
            refs: vec!["docs/plan.md".into()],
        };
        assert_eq!(
            admit(&manifest(), &submitted),
            vec![Refusal::IdentityMismatch {
                id: "artifacts/reads/0001".into(),
                expected: "sha256:bb22".into(),
                actual: "sha256:dead".into(),
            }]
        );
    }

    /// A reference to an id the manifest does not register is refused, naming source and target.
    #[test]
    fn dangling_reference_fails() {
        let submitted = Submitted {
            id: "artifacts/reads/0001".into(),
            digest: digest("bb22"),
            refs: vec!["docs/gone.md".into()],
        };
        assert_eq!(
            admit(&manifest(), &submitted),
            vec![Refusal::DanglingReference {
                from: "artifacts/reads/0001".into(),
                to: "docs/gone.md".into(),
            }]
        );
    }

    /// Every cause is reported, not only the first: a wrong digest and a dangling reference both
    /// appear for one submission.
    #[test]
    fn all_causes_are_reported() {
        let submitted = Submitted {
            id: "artifacts/reads/0001".into(),
            digest: digest("dead"),
            refs: vec!["docs/gone.md".into()],
        };
        let refusals = admit(&manifest(), &submitted);
        assert_eq!(refusals.len(), 2);
        assert!(refusals.contains(&Refusal::IdentityMismatch {
            id: "artifacts/reads/0001".into(),
            expected: "sha256:bb22".into(),
            actual: "sha256:dead".into(),
        }));
        assert!(refusals.contains(&Refusal::DanglingReference {
            from: "artifacts/reads/0001".into(),
            to: "docs/gone.md".into(),
        }));
    }
}
