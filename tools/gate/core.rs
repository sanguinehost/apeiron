//! Pure artifact-admission core for `tools/gate`.
//!
//! The dispatch manifest (`docs/artifacts.toml`) names the artifacts allowed to exist; a worker
//! submission enters the indexed receipt store only when the manifest itself is well formed, the
//! submitted artifact is registered, its payload digest matches the registered identity, and every
//! reference it carries resolves to a registered artifact. This module decides that admission.
//!
//! Lifecycle (settled; see `docs/plan.md`): **reserve task -> submit provisional output ->
//! independently verify -> seal accepted artifact.** Before dispatch the manifest freezes
//! ownership, inputs, required dependencies, gates and budgets. After execution the trusted adapter
//! computes the output identity. A verifier other than the worker checks the named gate. Only the
//! lead seals. A worker cannot change its own admission criteria or mark a submission accepted.
//! `admit` decides the provisional submission against a frozen manifest; sealing is the lead's act
//! and is not a property of this function.
//!
//! The body is pure: no IO, no hashing, no clock, no allocation beyond owned identifiers. The
//! adapter supplies a parsed [`Manifest`] and the submitted [`Submitted`] record with digests
//! already computed; the adapter's filesystem/hash/subprocess boundary is stated there. Purity is
//! deliberate so the real body can be extracted with Charon and refined in Lean, following the
//! `tools/dispatch/{core.rs,main.rs}` split.
//!
//! A finding of a theorem's name establishes nothing; this core admits a *record*, and the record's
//! named acceptance gate establishes its result.

/// A payload identity as the adapter prints it, for example `sha256:<hex>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Digest(pub String);

/// One artifact the manifest registers: its id, the digest of its payload, and the artifacts it
/// requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registered {
    pub id: String,
    pub digest: Digest,
    pub refs: Vec<String>,
}

/// The set of artifacts a submission may enter as. Validated before any submission is decided: ids
/// are unique and every registered requirement resolves.
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

/// Why a submission or its manifest is refused. One variant per required condition, so a refusal
/// names its cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Two registered artifacts share an id, so no identity is authoritative for it.
    DuplicateRegistration(String),
    /// A registered artifact requires an artifact the manifest does not register.
    MissingDependency { from: String, to: String },
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

/// Validate the manifest before admission. Reports every duplicate id once, and every registered
/// requirement that does not resolve. An empty vector means the manifest is well formed.
///
/// A missing registered requirement is a closure failure of the manifest itself: the manifest is
/// the dispatch contract, so a dependency it declares but does not register cannot be satisfied by
/// any later submission and must be refused before any output is admitted.
pub fn validate(manifest: &Manifest) -> Vec<Refusal> {
    let mut refusals = Vec::new();

    for (index, artifact) in manifest.artifacts.iter().enumerate() {
        let duplicate = manifest.artifacts[..index]
            .iter()
            .any(|earlier| earlier.id == artifact.id);
        if duplicate && !refusals.contains(&Refusal::DuplicateRegistration(artifact.id.clone())) {
            refusals.push(Refusal::DuplicateRegistration(artifact.id.clone()));
        }
    }

    for artifact in &manifest.artifacts {
        for required in &artifact.refs {
            if manifest.lookup(required).is_none() {
                let refusal = Refusal::MissingDependency {
                    from: artifact.id.clone(),
                    to: required.clone(),
                };
                if !refusals.contains(&refusal) {
                    refusals.push(refusal);
                }
            }
        }
    }

    refusals
}

/// Decide admission of a provisional submission against a frozen manifest. Returns every refusal;
/// an empty vector admits.
///
/// Refuses, and reports all causes it can: a malformed manifest (duplicate id, or a registered
/// requirement that does not resolve) refuses before anything else, because no identity in it is
/// authoritative; then an id absent from the manifest; then a payload digest that differs from the
/// registered identity; then every reference that does not resolve. An unregistered id returns
/// after the manifest check, because no registered identity exists to compare against. Admission is
/// provisional; sealing is a separate act by the lead.
pub fn admit(manifest: &Manifest, submitted: &Submitted) -> Vec<Refusal> {
    let mut refusals = validate(manifest);
    if !refusals.is_empty() {
        return refusals;
    }

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

    fn registered(id: &str, hex: &str, refs: &[&str]) -> Registered {
        Registered {
            id: id.into(),
            digest: digest(hex),
            refs: refs.iter().map(|r| (*r).into()).collect(),
        }
    }

    /// A well-formed manifest: unique ids, every registered requirement resolves.
    fn manifest() -> Manifest {
        Manifest {
            artifacts: vec![
                registered("docs/plan.md", "aa11", &[]),
                registered("artifacts/reads/0001", "bb22", &["docs/plan.md"]),
            ],
        }
    }

    #[test]
    fn registered_artifact_passes() {
        let submitted = Submitted {
            id: "artifacts/reads/0001".into(),
            digest: digest("bb22"),
            refs: vec!["docs/plan.md".into()],
        };
        assert_eq!(admit(&manifest(), &submitted), Vec::new());
    }

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

    /// A manifest that registers the same id twice is refused before any submission is decided:
    /// no identity is authoritative, and reversing the duplicates must not change the verdict.
    #[test]
    fn duplicate_registration_fails() {
        let mut m = Manifest {
            artifacts: vec![
                registered("artifacts/reads/0001", "bb22", &["docs/plan.md"]),
                registered("docs/plan.md", "aa11", &[]),
                registered("artifacts/reads/0001", "dead", &["docs/plan.md"]),
            ],
        };
        assert_eq!(
            validate(&m),
            vec![Refusal::DuplicateRegistration("artifacts/reads/0001".into())]
        );

        let submitted = Submitted {
            id: "artifacts/reads/0001".into(),
            digest: digest("dead"),
            refs: vec!["docs/plan.md".into()],
        };
        // Order-independent: reversing the two conflicting registrations must not change the verdict.
        assert_eq!(admit(&m, &submitted), validate(&m));
        m.artifacts.swap(0, 2);
        assert_eq!(admit(&m, &submitted), validate(&m));
    }

    /// A registered artifact that requires an artifact the manifest does not register is refused,
    /// even when the submission itself carries no dangling reference.
    #[test]
    fn missing_registered_dependency_fails() {
        let m = Manifest {
            artifacts: vec![
                registered("docs/plan.md", "aa11", &[]),
                registered("artifacts/reads/0001", "bb22", &["docs/plan.md", "docs/policy.md"]),
            ],
        };
        assert_eq!(
            validate(&m),
            vec![Refusal::MissingDependency {
                from: "artifacts/reads/0001".into(),
                to: "docs/policy.md".into(),
            }]
        );

        let submitted = Submitted {
            id: "artifacts/reads/0001".into(),
            digest: digest("bb22"),
            refs: vec!["docs/plan.md".into()],
        };
        assert_eq!(admit(&m, &submitted), validate(&m));
    }
}
