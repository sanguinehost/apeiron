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

/// The repository-layout rules `gate structure` enforces, parsed from `docs/contracts/tree.toml`
/// by `structure.rs`. `dirs` is sorted by `path` and unique; the parser enforces both.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rules {
    pub max_lines: u64,
    pub max_dir_files: u64,
    pub source: Vec<String>,
    pub generated: Vec<String>,
    pub dirs: Vec<DirRule>,
    pub debts: Vec<DebtRule>,
    pub frozen: Vec<String>,
}

/// One `[[dir]]` entry: the directory, whether it governs its descendants, and the basename globs
/// a tracked file admitted by it must match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirRule {
    pub path: String,
    pub recursive: bool,
    pub files: Vec<String>,
}

/// One `[[debt]]` entry: a violation of `rule` present at `path` when the file was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebtRule {
    pub rule: String,
    pub path: String,
}

/// One refusal of the structure gate, printed as `refuse <rule> <path> <detail>`. The derived
/// order is `(rule, path, detail)`, which is the printed order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StructureRefusal {
    pub rule: &'static str,
    pub path: String,
    pub detail: String,
}

/// The `[[dir]]` admitting `parent`, as an index into the sorted `dirs`: the exact entry for the
/// directory when one exists, else the nearest strict ancestor whose entry is recursive. `None`
/// when no entry admits it.
///
/// Cost: time O(d log D) comparisons, where d is the number of path components above `parent` and
/// D is the number of `[[dir]]` entries (one binary search per level); space O(1).
fn governing(dirs: &[&DirRule], parent: &str) -> Option<usize> {
    if let Ok(i) = dirs.binary_search_by(|d| d.path.as_str().cmp(parent)) {
        return Some(i);
    }
    let mut ancestor = parent;
    while ancestor != "." {
        ancestor = parent_of(ancestor);
        if let Ok(i) = dirs.binary_search_by(|d| d.path.as_str().cmp(ancestor)) {
            if dirs[i].recursive {
                return Some(i);
            }
        }
    }
    None
}

/// The directory a tracked path sits in; `.` for a root file.
fn parent_of(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => ".",
    }
}

fn basename(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[i + 1..],
        None => path,
    }
}

/// The extension of a path's basename; a leading dot is not an extension (`.gitignore` has none).
fn extension(path: &str) -> Option<&str> {
    let base = basename(path);
    match base.rfind('.') {
        Some(i) if i > 0 => Some(&base[i + 1..]),
        _ => None,
    }
}

/// Whether `path` is a source file under `rules.source`.
pub fn is_source(rules: &Rules, path: &str) -> bool {
    match extension(path) {
        Some(ext) => rules.source.iter().any(|s| s.as_str() == ext),
        None => false,
    }
}

/// Glob match of a basename: `*` matches any run of characters other than `/`, every other byte
/// matches itself. The greedy scan with one backtrack point (the last `*`) is complete for this
/// glob language because `*` cannot match across `/`.
///
/// Cost: time O(|pattern| * |name|) comparisons in the worst case (each backtrack advances the
/// name by one and rescans the pattern tail); space O(1).
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let (p, n) = (pattern.as_bytes(), name.as_bytes());
    let (mut pi, mut ni) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;
    while ni < n.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = Some((pi, ni));
            pi += 1;
        } else if pi < p.len() && p[pi] == n[ni] {
            pi += 1;
            ni += 1;
        } else if let Some((sp, sn)) = star {
            if n[sn] == b'/' {
                return false;
            }
            pi = sp + 1;
            ni = sn + 1;
            star = Some((sp, sn + 1));
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

/// Decides the repository layout: every refusal of the seven rules in the `tree.toml` header, sorted.
/// `files` is the tracked list with each source file's newline count (non-source entries carry 0 and
/// are not read for lines). The body is pure: no IO and no clock; its owned allocations are sorted
/// copies of the inputs' keys.
///
/// Cost: time O((F + D + Q + P) log(F + D + Q + P)) comparisons, with F files, D `[[dir]]` entries,
/// Q `[[debt]]` entries and P frozen Python paths: the four sorts dominate, and each per-file lookup
/// is O(d log D) (see [`governing`]) or O(log Q) or O(log F). Space O(F + D + Q + P) for the sorted
/// copies and the refusals. The per-directory source count is a run-length pass over the sorted list
/// of source-file parents, so it adds O(F) after its sort.
pub fn structure_verdict(rules: &Rules, files: &[(String, usize)]) -> Vec<StructureRefusal> {
    let refuse = |rule: &'static str, path: &str, detail: String| StructureRefusal {
        rule,
        path: path.to_string(),
        detail,
    };
    let mut out = Vec::new();

    let mut tracked: Vec<(&str, usize)> = files.iter().map(|(p, n)| (p.as_str(), *n)).collect();
    tracked.sort_unstable();
    let mut dirs: Vec<&DirRule> = rules.dirs.iter().collect();
    dirs.sort_unstable_by(|a, b| a.path.cmp(&b.path));
    let mut debts: Vec<(&str, &str)> = rules
        .debts
        .iter()
        .map(|d| (d.rule.as_str(), d.path.as_str()))
        .collect();
    debts.sort_unstable();
    let mut frozen: Vec<&str> = rules.frozen.iter().map(String::as_str).collect();
    frozen.sort_unstable();

    let lines_of = |path: &str| -> Option<usize> {
        tracked
            .binary_search_by(|(p, _)| p.cmp(&path))
            .ok()
            .map(|i| tracked[i].1)
    };
    let has_debt = |rule: &str, path: &str| {
        debts
            .binary_search_by(|(r, p)| (*r, *p).cmp(&(rule, path)))
            .is_ok()
    };

    // Rules 1 and 2: admission of each tracked path, and each entry's use.
    let mut used = vec![false; dirs.len()];
    for &(path, _) in &tracked {
        let parent = parent_of(path);
        match governing(&dirs, parent) {
            None => out.push(refuse(
                "unadmitted",
                path,
                "no [[dir]] admits its directory".into(),
            )),
            Some(i) => {
                let base = basename(path);
                if dirs[i].files.iter().any(|g| glob_match(g, base)) {
                    used[i] = true;
                } else {
                    out.push(refuse(
                        "unadmitted",
                        path,
                        format!("no files glob of [[dir]] {} matches `{base}`", dirs[i].path),
                    ));
                }
            }
        }
    }
    for (i, dir) in dirs.iter().enumerate() {
        if !used[i] {
            out.push(refuse(
                "empty-dir-entry",
                dir.path.as_str(),
                "[[dir]] admits no tracked path".into(),
            ));
        }
    }

    // Rule 3: physical lines of a source file, unless generated or carrying a max_lines debt.
    for &(path, lines) in &tracked {
        if is_source(rules, path)
            && !rules.generated.iter().any(|g| g.as_str() == path)
            && lines as u64 > rules.max_lines
            && !has_debt("max_lines", path)
        {
            out.push(refuse(
                "max_lines",
                path,
                format!("{lines} lines > max_lines {}", rules.max_lines),
            ));
        }
    }

    // Rule 4: source files directly in one directory, as run lengths over the sorted parents.
    let mut source_parents: Vec<&str> = tracked
        .iter()
        .filter(|(p, _)| is_source(rules, p))
        .map(|(p, _)| parent_of(p))
        .collect();
    source_parents.sort_unstable();
    let mut dir_counts: Vec<(&str, u64)> = Vec::new();
    for dir in source_parents {
        match dir_counts.last_mut() {
            Some((d, n)) if *d == dir => *n += 1,
            _ => dir_counts.push((dir, 1)),
        }
    }
    for &(dir, n) in &dir_counts {
        if n > rules.max_dir_files && !has_debt("max_dir_files", dir) {
            out.push(refuse(
                "max_dir_files",
                dir,
                format!("{n} source files > max_dir_files {}", rules.max_dir_files),
            ));
        }
    }

    // Rule 5: a debt entry whose violation no longer holds.
    for d in &rules.debts {
        let stale = match d.rule.as_str() {
            "max_lines" => match lines_of(d.path.as_str()) {
                None => Some("path not tracked".to_string()),
                Some(_) if !is_source(rules, &d.path) => Some("not a source file".to_string()),
                Some(_) if rules.generated.contains(&d.path) => {
                    Some("generated, exempt from max_lines".to_string())
                }
                Some(lines) if lines as u64 <= rules.max_lines => {
                    Some(format!("{lines} lines <= max_lines {}", rules.max_lines))
                }
                Some(_) => None,
            },
            "max_dir_files" => {
                match dir_counts.binary_search_by(|(dir, _)| dir.cmp(&d.path.as_str())) {
                    Err(_) => Some("no tracked source file in it".to_string()),
                    Ok(i) if dir_counts[i].1 <= rules.max_dir_files => {
                        Some(format!("{} source files <= max_dir_files", dir_counts[i].1))
                    }
                    Ok(_) => None,
                }
            }
            other => Some(format!("unknown rule `{other}`")),
        };
        if let Some(why) = stale {
            out.push(refuse(
                "stale-debt",
                d.path.as_str(),
                format!("{} debt: {why}", d.rule),
            ));
        }
    }

    // Rules 6 and 7: Python is admitted only by the frozen list, which only shrinks.
    for &(path, _) in &tracked {
        if extension(path) == Some("py") && frozen.binary_search_by(|f| (*f).cmp(path)).is_err() {
            out.push(refuse(
                "python-unfrozen",
                path,
                "not listed in [python].frozen".into(),
            ));
        }
    }
    for &entry in &frozen {
        if lines_of(entry).is_none() {
            out.push(refuse(
                "python-stale",
                entry,
                "frozen entry is not tracked".into(),
            ));
        }
    }

    out.sort();
    out
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
            vec![Refusal::DuplicateRegistration(
                "artifacts/reads/0001".into()
            )]
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
                registered(
                    "artifacts/reads/0001",
                    "bb22",
                    &["docs/plan.md", "docs/policy.md"],
                ),
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
