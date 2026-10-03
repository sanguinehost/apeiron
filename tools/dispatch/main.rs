//! IO adapter for the extracted observation-level admission core.
//! IO provenance, codegen, hash security and post-check races remain trusted boundaries.
use dispatch_core::{Digest, Observation, Permit, Session, Stage};
use serde_json::{json, Value};
use sha2::{Digest as ShaDigest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
mod self_test;
type Checked<T> = Result<T, String>;

fn require(ok: bool, diagnostic: &str) -> Checked<()> {
    if ok {
        Ok(())
    } else {
        Err(diagnostic.to_owned())
    }
}
fn text<'a>(v: &'a Value, key: &str) -> Checked<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("ADMISSION_FIELD_REJECT {key}"))
}
/// O(B) input bytes, O(64KiB) streaming scratch; each byte contributes once to SHA256.
fn sha(path: &Path) -> Checked<String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut block = [0; 65536];
    loop {
        let n = file.read(&mut block).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&block[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
/// O(P) components and storage; reject traversal and every existing symlink, allow new leaf paths.
fn inside(root: &Path, relative: &str) -> Checked<PathBuf> {
    require(
        !relative.is_empty() && !Path::new(relative).is_absolute(),
        "ADMISSION_PATH_REJECT",
    )?;
    let mut p = root.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            Component::Normal(x) => p.push(x),
            _ => return Err("ADMISSION_PATH_REJECT".into()),
        }
        match fs::symlink_metadata(&p) {
            Ok(m) => require(!m.file_type().is_symlink(), "ADMISSION_PATH_REJECT")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(p)
}
/// O(B+P) source bytes and path components; bounded streaming hash scratch per file.
fn identities(root: &Path, value: &Value) -> Checked<()> {
    let files = value.as_object().ok_or("ADMISSION_IDENTITY_MISSING")?;
    require(!files.is_empty(), "ADMISSION_IDENTITY_MISSING")?;
    for (rel, expected) in files {
        let expected = expected.as_str().ok_or("ADMISSION_DIGEST_REJECT")?;
        require(
            expected.len() == 64
                && expected
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "ADMISSION_DIGEST_REJECT",
        )?;
        let p = inside(root, rel)?;
        require(
            p.is_file() && sha(&p)? == expected,
            &format!("ADMISSION_STALE_SOURCE {rel}"),
        )?;
    }
    Ok(())
}
/// O(N) identifier bytes, no regex engine or allocation beyond caller input.
fn lean_name(s: &str) -> bool {
    !s.is_empty()
        && s.split('.').all(|part| {
            let mut chars = part.bytes();
            matches!(chars.next(),Some(b) if b.is_ascii_alphabetic() || b==b'_')
                && chars.all(|b| b.is_ascii_alphanumeric() || b == b'_')
        })
}

struct Context {
    root: PathBuf,
    package: PathBuf,
    start: Instant,
    parent_deadline: Option<u64>,
    logs: PathBuf,
    stage: PathBuf,
    corpus_lean_path: String,
    records: Vec<Value>,
}
impl Context {
    /// O(P) path setup and directory creation; each invocation has an isolated proof stage and logs.
    fn new() -> Checked<Self> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("ADMISSION_ROOT_REJECT")?
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let unique = format!(
            "admission-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_nanos()
        );
        let stage = Path::new("/home/socol/oc-tmp/rust-admission-fixture").join(&unique);
        let logs = Path::new("/tmp/opencode/rust-admission").join(unique);
        fs::create_dir_all(&stage).map_err(|e| e.to_string())?;
        fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
        let package = root.join("proof/corpus");
        require(
            fs::read_to_string(package.join("lean-toolchain"))
                .map_err(|e| e.to_string())?
                .trim()
                == "leanprover/lean4:v4.31.0",
            "ADMISSION_TOOLCHAIN_PIN_REJECT",
        )?;
        // Read-only capture of the corpus Lean environment. `lake env lean` overwrites
        // LEAN_PATH, so a stage-local certificate must carry the corpus search path itself;
        // `lake env printenv` re-derives it without building or mutating `proof/corpus/.lake`.
        let corpus_lean_path = {
            let out = Command::new("lake")
                .args(["env", "printenv", "LEAN_PATH"])
                .current_dir(&package)
                .env("LEAN_NUM_THREADS", "1")
                .output()
                .map_err(|e| e.to_string())?;
            require(out.status.success(), "ADMISSION_CORPUS_ENV_REJECT")?;
            String::from_utf8(out.stdout)
                .map_err(|e| e.to_string())?
                .trim()
                .to_owned()
        };
        require(!corpus_lean_path.is_empty(), "ADMISSION_CORPUS_ENV_REJECT")?;
        Ok(Self {
            root,
            package,
            start: Instant::now(),
            parent_deadline: match std::env::var("APEIRON_DISPATCH_DEADLINE_EPOCH") {
                Ok(value) => Some(value.parse().map_err(|_| "ADMISSION_BUDGET_PARSE_REJECT")?),
                Err(std::env::VarError::NotPresent) => None,
                Err(_) => return Err("ADMISSION_BUDGET_PARSE_REJECT".into()),
            },
            logs,
            stage,
            corpus_lean_path,
            records: vec![],
        })
    }
    fn remaining(&self) -> Checked<u64> {
        let mut left = 530_u64
            .checked_sub(self.start.elapsed().as_secs())
            .ok_or("ADMISSION_BUDGET_EXHAUSTED")?;
        if let Some(deadline) = self.parent_deadline {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| "ADMISSION_BUDGET_CLOCK_REJECT")?
                .as_secs();
            left = left.min(
                deadline
                    .checked_sub(now)
                    .ok_or("ADMISSION_BUDGET_EXHAUSTED")?,
            );
        }
        left.checked_sub(3)
            .filter(|n| *n > 0)
            .ok_or("ADMISSION_BUDGET_EXHAUSTED".into())
    }

    /// External work shares one deadline; O(L) captured log bytes and O(1) polling state.
    fn run(
        &mut self,
        argv: &[String],
        cwd: &Path,
        label: &str,
        lean_path: Option<&str>,
    ) -> Checked<(i32, String)> {
        let seconds = self.remaining()?;
        let log = self.logs.join(format!("{label}.log"));
        let f = File::create(&log).map_err(|e| e.to_string())?;
        let mut command = Command::new("timeout");
        command
            .args(["--kill-after=2", &seconds.to_string()])
            .args(argv)
            .current_dir(cwd)
            .stdout(Stdio::from(f.try_clone().map_err(|e| e.to_string())?))
            .stderr(Stdio::from(f))
            .process_group(0);
        command.env("LEAN_NUM_THREADS", "1").env(
            "LEAN_PATH",
            lean_path.map_or_else(String::new, |p| p.to_owned()),
        );
        let begin = Instant::now();
        let mut child = command.spawn().map_err(|e| e.to_string())?;
        let status = loop {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                break status;
            }
            if self.remaining().is_err() {
                let _ = Command::new("kill")
                    .args(["-KILL", "--", &format!("-{}", child.id())])
                    .status();
                let _ = child.wait();
                return Err("ADMISSION_BUDGET_EXHAUSTED".into());
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let code = status
            .code()
            .unwrap_or_else(|| 128 + status.signal().unwrap_or(0));
        let output = fs::read_to_string(&log).map_err(|e| e.to_string())?;
        self.records.push(json!({"label":label,"argv":argv,"cwd":cwd,"exit":code,"seconds":begin.elapsed().as_secs_f64()}));
        Ok((code, output))
    }
    /// One bounded Lean call; locking, memory cap and owning package/toolchain remain explicit.
    /// The pinned toolchain equals the corpus pin, and the corpus search path is passed
    /// explicitly because `lake env` cannot carry the stage root (`-R`) alongside it.
    fn lean(
        &mut self,
        file: &Path,
        label: &str,
        output: Option<&Path>,
        production: bool,
    ) -> Checked<(i32, String)> {
        let seconds = self.remaining()?.to_string();
        let mut argv = vec![
            "flock".into(),
            "-w".into(),
            seconds.clone(),
            "/tmp/opencode/apeiron-lane.lock".into(),
            "systemd-run".into(),
            "--user".into(),
            "--scope".into(),
            "-q".into(),
            "-p".into(),
            "MemoryMax=12G".into(),
            "-p".into(),
            "MemorySwapMax=0".into(),
            "timeout".into(),
            "--kill-after=2".into(),
            seconds,
            "elan".into(),
            "run".into(),
            "leanprover/lean4:v4.31.0".into(),
            "lean".into(),
        ];
        argv.extend(["-R".into(), self.stage.to_string_lossy().into_owned()]);
        if let Some(out) = output {
            argv.extend(["-o".into(), out.to_string_lossy().into_owned()]);
        }
        argv.push(file.to_string_lossy().into_owned());
        let lean_path = if production {
            format!("{}:{}", self.stage.to_string_lossy(), self.corpus_lean_path)
        } else {
            self.stage.to_string_lossy().into_owned()
        };
        self.run(&argv, &self.package.clone(), label, Some(&lean_path))
    }
    /// O(B) schema copy plus bounded Lean compilation, using current source and a fresh olean.
    fn prepare(&mut self) -> Checked<()> {
        let file = self.stage.join("DispatchAdmission.lean");
        fs::copy(self.root.join("proof/DispatchAdmission.lean"), &file)
            .map_err(|e| e.to_string())?;
        let (code, out) = self.lean(
            &file,
            "schema",
            Some(&self.stage.join("DispatchAdmission.olean")),
            false,
        )?;
        require(code == 0, &format!("ADMISSION_SCHEMA_COMPILE_REJECT {out}"))
    }
    /// O(B+L) proof bytes and one bounded kernel module compile; no cached contract is accepted.
    fn certificate_module(
        &mut self,
        contract: &Value,
        root: &Path,
        label: &str,
        fixture: bool,
    ) -> Checked<()> {
        let file = self.stage.join("Contract.lean");
        fs::copy(inside(root, text(contract, "proof_file")?)?, &file).map_err(|e| e.to_string())?;
        let (code, out) = self.lean(
            &file,
            &format!("{label}-module"),
            Some(&self.stage.join("Contract.olean")),
            !fixture,
        )?;
        require(code == 0, &format!("ADMISSION_LEAN_REJECT {label}: {out}"))
    }
    /// O(B+L) generated query bytes and one bounded kernel check; the closed axiom cone is required.
    fn certificate_audit(&mut self, contract: &Value, label: &str, fixture: bool) -> Checked<()> {
        for key in [
            "namespace",
            "before",
            "after",
            "reachable",
            "admissible",
            "before_cost",
            "after_cost",
            "certificate",
        ] {
            require(
                lean_name(text(contract, key)?),
                &format!("ADMISSION_NAME_REJECT {key}"),
            )?;
        }
        let linter_options = if fixture {
            ""
        } else {
            "set_option linter.apeiron.notation false\nset_option linter.apeiron.citation false\n"
        };
        let mut query = format!(
            "import DispatchAdmission\nimport Contract\n{linter_options}set_option warningAsError true\ndef checkedAdmission : ProofDispatch.Admission "
        );
        for key in [
            "before",
            "after",
            "reachable",
            "admissible",
            "before_cost",
            "after_cost",
        ] {
            query.push_str(text(contract, key)?);
            query.push(' ');
        }
        query.push_str(" := ");
        query.push_str(text(contract, "certificate")?);
        query.push_str("\n#audit_admission checkedAdmission\n");
        let file = self.stage.join("Check.lean");
        fs::write(&file, query).map_err(|e| e.to_string())?;
        let (code, out) = self.lean(&file, &format!("{label}-check"), None, !fixture)?;
        require(
            code == 0 && out.contains("ADMISSION_KERNEL_OK"),
            &format!("ADMISSION_CERTIFICATE_REJECT {label}: {out}"),
        )
    }
    /// O(B+L) proof/log bytes and two bounded kernel checks; no cached contract is accepted.
    fn kernel(&mut self, contract: &Value, root: &Path, label: &str, fixture: bool) -> Checked<()> {
        self.certificate_module(contract, root, label, fixture)?;
        self.certificate_audit(contract, label, fixture)
    }
    /// O(B+P) identity sweep and O(J) serialized binding bytes; hashing security is an assumption.
    fn binding(&self, p: &Value, registry: &Path, root: &Path) -> Checked<Digest> {
        identities(root, &p["identity"])?;
        let current = sha(registry)?;
        require(
            current == text(p, "registry_sha256")?,
            "ADMISSION_STALE_REGISTRY",
        )?;
        let bytes = serde_json::to_vec(&json!([p, current])).map_err(|e| e.to_string())?;
        let hash = Sha256::digest(bytes);
        let limb = |i: usize| {
            let mut a = [0; 8];
            a.copy_from_slice(&hash[i..i + 8]);
            u64::from_be_bytes(a)
        };
        Ok(Digest {
            d0: limb(0),
            d1: limb(8),
            d2: limb(16),
            d3: limb(24),
        })
    }
    /// O(B+P) fresh identities; the extracted transition consumes the previous session.
    fn observe(
        &self,
        session: Session,
        p: &Value,
        registry: &Path,
        root: &Path,
        stage: Stage,
        exit: i32,
    ) -> Checked<Session> {
        dispatch_core::advance(
            session,
            Observation {
                stage,
                exit,
                identity: self.binding(p, registry, root)?,
            },
        )
        .ok_or("ADMISSION_STATE_REJECT".into())
    }
    /// O(B+P+J) validation, then bounded extraction/build/kernel work with freshly observed transitions.
    fn check(
        &mut self,
        p: &Value,
        registry: &Path,
        root: &Path,
        label: &str,
        fixture: bool,
    ) -> Checked<Admitted> {
        let object = p.as_object().ok_or("ADMISSION_SCHEMA_REJECT")?;
        require(
            object.len() == 6
                && [
                    "task",
                    "contract",
                    "registry_sha256",
                    "identity",
                    "owned_files",
                    "worker",
                ]
                .iter()
                .all(|k| object.contains_key(*k)),
            "ADMISSION_SCHEMA_REJECT",
        )?;
        require(
            text(p, "task")?.starts_with("APEIRON-"),
            "ADMISSION_TASK_REJECT",
        )?;
        require(
            text(p, "registry_sha256")? == sha(registry)?,
            "ADMISSION_STALE_REGISTRY",
        )?;
        let data: Value = serde_json::from_slice(&fs::read(registry).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        require(
            (data["schema"] == 1 || data["schema"] == 2) && data["contracts"].is_object(),
            "ADMISSION_REGISTRY_REJECT",
        )?;
        let contract = data["contracts"]
            .get(text(p, "contract")?)
            .filter(|v| v.is_object())
            .ok_or("ADMISSION_UNREGISTERED_CONTRACT: source/refinement/cost-model task required")?;
        require(
            text(contract, "grade")? == "source-connected-cost-model",
            "ADMISSION_UNLINKED_MODEL",
        )?;
        require(
            p["identity"] == contract["identity"],
            "ADMISSION_IDENTITY_SUBSTITUTION",
        )?;
        let paths = p["owned_files"]
            .as_array()
            .ok_or("ADMISSION_OWNERSHIP_REJECT")?;
        let registered_paths = contract["owned_files"]
            .as_array()
            .ok_or("ADMISSION_OWNERSHIP_REJECT")?;
        require(
            !paths.is_empty() && paths.len() <= 3,
            "ADMISSION_OWNERSHIP_REJECT",
        )?;
        for (i, path) in paths.iter().enumerate() {
            require(!paths[..i].contains(path), "ADMISSION_OWNERSHIP_REJECT")?;
            require(registered_paths.contains(path), "ADMISSION_UNOWNED_PATH")?;
            inside(root, path.as_str().ok_or("ADMISSION_PATH_REJECT")?)?;
        }
        let values = p["worker"].as_array().ok_or("ADMISSION_ARGV_REJECT")?;
        require(!values.is_empty(), "ADMISSION_ARGV_REJECT")?;
        let mut worker = vec![];
        for value in values {
            let arg = value.as_str().ok_or("ADMISSION_ARGV_REJECT")?;
            require(!arg.contains('\0'), "ADMISSION_ARGV_REJECT")?;
            worker.push(arg.to_owned());
        }
        require(
            p["worker"] == contract["worker"],
            "ADMISSION_WORKER_SUBSTITUTION",
        )?;
        let source_kind = text(contract, "source_kind")?;
        require(
            matches!(source_kind, "rust-aeneas" | "shader-spirv"),
            "ADMISSION_SOURCE_KIND_REJECT",
        )?;
        // The extraction-equivalent fresh source check for a non-Rust executing source.
        let mut source_gate_argv: Option<Vec<String>> = None;
        if !fixture {
            let identity = p["identity"]
                .as_object()
                .ok_or("ADMISSION_IDENTITY_MISSING")?;
            require(
                identity.contains_key(text(contract, "proof_file")?),
                "ADMISSION_PROOF_IDENTITY_MISSING",
            )?;
            match source_kind {
                "rust-aeneas" => {
                    let source_root = text(contract, "source_root")?;
                    require(
                        lean_name(source_root) && !source_root.contains('.'),
                        "ADMISSION_NOT_EXTRACTED",
                    )?;
                    let script = fs::read_to_string(root.join("proof/regen.sh"))
                        .map_err(|e| e.to_string())?;
                    let words: Vec<_> = script.split_whitespace().collect();
                    let selected = format!("crate::{source_root}");
                    require(
                        words
                            .windows(2)
                            .any(|w| w[0] == "--start-from" && w[1] == selected),
                        "ADMISSION_NOT_EXTRACTED",
                    )?;
                    require(
                        lean_name(text(contract, "build_target")?),
                        "ADMISSION_BUILD_TARGET_REJECT",
                    )?;
                    for rel in [
                        "proof/extract/lib.rs",
                        "proof/regen.sh",
                        "proof/ApeironWord.lean",
                        "proof/DispatchAdmission.lean",
                        "tools/dispatch/admit.py",
                        "tools/dispatch/core.rs",
                        "tools/dispatch/main.rs",
                        "tools/dispatch/self_test.rs",
                        "tools/dispatch/Cargo.toml",
                        "tools/dispatch/Cargo.lock",
                        "tools/dispatch/run",
                        "proof/corpus/lean-toolchain",
                        "proof/dispatch/extract/lib.rs",
                        "proof/dispatch/extract/Cargo.toml",
                        "proof/dispatch/extract/Cargo.lock",
                        "proof/dispatch/Dispatch/Extracted.lean",
                        "proof/dispatch/Dispatch/Spec.lean",
                        "proof/dispatch/Dispatch/Refinement.lean",
                        "proof/dispatch/Dispatch/Audit.lean",
                        "proof/dispatch/lakefile.lean",
                        "proof/dispatch/lake-manifest.json",
                        "proof/dispatch/lean-toolchain",
                        "proof/dispatch/regen.sh",
                        "proof/dispatch/gate.py",
                    ] {
                        require(
                            identity.contains_key(rel),
                            &format!("ADMISSION_IDENTITY_CLOSURE_MISSING {rel}"),
                        )?;
                    }
                }
                "shader-spirv" => {
                    // The registered source is the shader, its dispatch path and its toolchain,
                    // not a Rust model: name every leg and bind it into the identity map.
                    let source = contract["shader_source"]
                        .as_object()
                        .ok_or("ADMISSION_SHADER_SOURCE_REJECT")?;
                    for key in [
                        "shader",
                        "entry_point",
                        "dispatch_driver",
                        "shader_abi",
                        "toolchain",
                        "target_env",
                        "source_gate",
                        "identity_file",
                    ] {
                        require(
                            source
                                .get(key)
                                .and_then(Value::as_str)
                                .is_some_and(|s| !s.is_empty()),
                            &format!("ADMISSION_SHADER_SOURCE_REJECT {key}"),
                        )?;
                    }
                    // The observation and cost interfaces must be explicit, not inferred.
                    for kind in ["observation", "cost"] {
                        let iface = contract[kind]
                            .as_object()
                            .ok_or("ADMISSION_OBSERVATION_REJECT")?;
                        for key in ["protocol", "metric", "interval"] {
                            require(
                                iface
                                    .get(key)
                                    .and_then(Value::as_str)
                                    .is_some_and(|s| !s.is_empty()),
                                &format!("ADMISSION_OBSERVATION_REJECT {kind}.{key}"),
                            )?;
                        }
                    }
                    require(
                        lean_name(text(contract, "build_target")?),
                        "ADMISSION_BUILD_TARGET_REJECT",
                    )?;
                    let gate = source["source_gate"]
                        .as_str()
                        .ok_or("ADMISSION_SHADER_SOURCE_REJECT")?;
                    let idfile = source["identity_file"]
                        .as_str()
                        .ok_or("ADMISSION_SHADER_SOURCE_REJECT")?;
                    for key in ["shader", "dispatch_driver", "source_gate", "identity_file"] {
                        let rel = source[key]
                            .as_str()
                            .ok_or("ADMISSION_SHADER_SOURCE_REJECT")?;
                        require(
                            identity.contains_key(rel),
                            "ADMISSION_SHADER_IDENTITY_MISSING",
                        )?;
                        inside(root, rel)?;
                    }
                    for rel in [
                        "proof/DispatchAdmission.lean",
                        "tools/dispatch/admit.py",
                        "tools/dispatch/core.rs",
                        "tools/dispatch/main.rs",
                        "tools/dispatch/self_test.rs",
                        "tools/dispatch/Cargo.toml",
                        "tools/dispatch/Cargo.lock",
                        "tools/dispatch/run",
                        "proof/corpus/lean-toolchain",
                        "proof/dispatch/lakefile.lean",
                        "proof/dispatch/lean-toolchain",
                        "proof/dispatch/regen.sh",
                        "proof/dispatch/gate.py",
                    ] {
                        require(
                            identity.contains_key(rel),
                            &format!("ADMISSION_IDENTITY_CLOSURE_MISSING {rel}"),
                        )?;
                    }
                    source_gate_argv = Some(vec![
                        "python3".into(),
                        gate.to_owned(),
                        "--check".into(),
                        "--identity".into(),
                        idfile.to_owned(),
                    ]);
                }
                _ => return Err("ADMISSION_SOURCE_KIND_REJECT".into()),
            }
        }
        let binding = self.binding(p, registry, root)?;
        let session =
            dispatch_core::start(Some(binding), binding).ok_or("ADMISSION_STATE_REJECT")?;
        let shader = source_gate_argv.is_some();
        let code = if fixture {
            0
        } else {
            let args = match &source_gate_argv {
                Some(cmd) => self.bounded_command(cmd.clone())?,
                None => self.bounded_command(vec!["proof/regen.sh".into(), "--check".into()])?,
            };
            let stage_label = if shader { "source" } else { "regen" };
            let (code, out) = self.run(&args, root, &format!("{label}-{stage_label}"), None)?;
            if code != 0 {
                let _ = self.observe(session, p, registry, root, Stage::Extraction, code);
                let diagnostic = if shader {
                    "ADMISSION_SOURCE_IDENTITY_REJECT"
                } else {
                    "ADMISSION_EXTRACTION_REJECT"
                };
                return Err(format!("{diagnostic} {out}"));
            }
            if shader {
                require(
                    out.contains("ADMISSION_SOURCE_IDENTITY_OK"),
                    &format!("ADMISSION_SOURCE_IDENTITY_REJECT {out}"),
                )?;
            }
            code
        };
        let session = self.observe(session, p, registry, root, Stage::Extraction, code)?;
        let code = if fixture {
            0
        } else if shader {
            // The certificate module compile is the dependency leg: it must resolve the
            // canonical corpus imports under the pinned toolchain before the audit.
            if let Err(e) = self.certificate_module(contract, root, label, fixture) {
                let _ = self.observe(session, p, registry, root, Stage::Dependencies, 1);
                return Err(e);
            }
            0
        } else {
            let args = self.bounded_command(vec![
                "lake".into(),
                "build".into(),
                text(contract, "build_target")?.into(),
            ])?;
            let (code, out) = self.run(
                &args,
                &root.join("proof/corpus"),
                &format!("{label}-build"),
                None,
            )?;
            if code != 0 {
                let _ = self.observe(session, p, registry, root, Stage::Dependencies, code);
                return Err(format!("ADMISSION_DEPENDENCY_REJECT {out}"));
            }
            code
        };
        let session = self.observe(session, p, registry, root, Stage::Dependencies, code)?;
        let certificate = if shader {
            self.certificate_audit(contract, label, fixture)
        } else {
            self.kernel(contract, root, label, fixture)
        };
        if let Err(e) = certificate {
            let _ = self.observe(session, p, registry, root, Stage::Certificate, 1);
            return Err(e);
        }
        let session = self.observe(session, p, registry, root, Stage::Certificate, 0)?;
        let session = self.observe(session, p, registry, root, Stage::Revalidation, 0)?;
        let permit = dispatch_core::finish(session, self.binding(p, registry, root)?)
            .ok_or("ADMISSION_STATE_REJECT")?;
        Ok(Admitted { permit, worker })
    }
    /// O(A) argv bytes; external work retains lane, memory cap and the remaining whole deadline.
    fn bounded_command(&self, tail: Vec<String>) -> Checked<Vec<String>> {
        let seconds = self.remaining()?.to_string();
        let mut args = vec![
            "flock".into(),
            "-w".into(),
            seconds.clone(),
            "/tmp/opencode/apeiron-lane.lock".into(),
            "systemd-run".into(),
            "--user".into(),
            "--scope".into(),
            "-q".into(),
            "-p".into(),
            "MemoryMax=12G".into(),
            "-p".into(),
            "MemorySwapMax=0".into(),
            "timeout".into(),
            "--kill-after=2".into(),
            seconds,
        ];
        args.extend(tail);
        Ok(args)
    }
    /// O(B+P) final identity sweep; consumes Permit before running the immutable captured worker argv.
    fn launch(
        &mut self,
        p: &Value,
        registry: &Path,
        root: &Path,
        label: &str,
        fixture: bool,
    ) -> Checked<()> {
        let Admitted { permit, worker } = self.check(p, registry, root, label, fixture)?;
        require(
            permit.identity().same(self.binding(p, registry, root)?),
            "ADMISSION_STATE_REJECT",
        )?;
        let (code, out) = self.run(&worker, root, &format!("{label}-worker"), None)?;
        require(code == 0, &format!("ADMISSION_WORKER_FAILED {out}"))
    }
}
struct Admitted {
    permit: Permit,
    worker: Vec<String>,
}

/// O(J+B+L) proposal/registry/source/log bytes plus bounded external work; no reusable tokens.
fn execute(ctx: &mut Context, args: &[String]) -> Checked<()> {
    ctx.prepare()?;
    if args.len() == 1 && args[0] == "--self-test" {
        return self_test::run(ctx);
    }
    require(
        args.len() == 2 && matches!(args[0].as_str(), "--check" | "--launch"),
        "ADMISSION_USAGE_REJECT",
    )?;
    let proposal: Value = serde_json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let root = ctx.root.clone();
    let registry = root.join("tools/dispatch/contracts.json");
    if args[0] == "--launch" {
        ctx.launch(&proposal, &registry, &root, "production", false)?;
    } else {
        let _ = ctx.check(&proposal, &registry, &root, "production", false)?;
    }
    println!("ADMISSION_CHECKED_NOW: no reusable launch token");
    Ok(())
}
/// O(J+L) receipt bytes and cleanup of this invocation's private stage.
fn main() {
    let result=Context::new().and_then(|mut ctx| {
        let args:Vec<String>=std::env::args().skip(1).collect();let result=execute(&mut ctx,&args);
        let receipt=json!({"status":if result.is_ok(){"passed"}else{"failed"},"error":result.as_ref().err(),"elapsed_seconds":ctx.start.elapsed().as_secs_f64(),"records":ctx.records,"grade":"conditional source-refined core; trusted/tested IO adapter"});
        let receipt_result=fs::write(ctx.logs.join("receipt.json"),serde_json::to_vec_pretty(&receipt).map_err(|e|e.to_string())?).map_err(|e|e.to_string());
        let _=fs::remove_dir_all(&ctx.stage);result.and(receipt_result)
    });
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    /// O(P) private directory creation; the unique leaf prevents touching an existing test population.
    fn directory(label: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let parent = Path::new("/home/socol/oc-tmp/rust-admission-boundary-tests");
        fs::create_dir_all(parent).expect("test parent");
        let root = parent.join(format!("{}-{stamp}-{label}", std::process::id()));
        fs::create_dir(&root).expect("unique test directory");
        root
    }
    #[test]
    fn parent_deadline_expiry() {
        let ctx = Context {
            root: PathBuf::new(),
            package: PathBuf::new(),
            start: Instant::now(),
            parent_deadline: Some(0),
            logs: PathBuf::new(),
            stage: PathBuf::new(),
            corpus_lean_path: String::new(),
            records: vec![],
        };
        assert!(ctx.parent_deadline.is_some());
        assert_eq!(ctx.remaining().unwrap_err(), "ADMISSION_BUDGET_EXHAUSTED");
    }
    #[test]
    fn rejects_parent_and_absolute_paths() {
        let root = directory("traversal");
        assert_eq!(
            inside(&root, "../outside").unwrap_err(),
            "ADMISSION_PATH_REJECT"
        );
        assert_eq!(
            inside(&root, "/tmp/outside").unwrap_err(),
            "ADMISSION_PATH_REJECT"
        );
        fs::remove_dir(root).expect("cleanup");
    }
    #[test]
    fn rejects_symlink_parent_before_missing_leaf() {
        let root = directory("symlink");
        std::os::unix::fs::symlink("/tmp", root.join("escape")).expect("test symlink");
        assert_eq!(
            inside(&root, "escape/new-leaf").unwrap_err(),
            "ADMISSION_PATH_REJECT"
        );
        fs::remove_file(root.join("escape")).expect("remove symlink");
        fs::remove_dir(root).expect("cleanup");
    }
    #[test]
    fn accepts_new_owned_leaf_without_filesystem_escape() {
        let root = directory("new-leaf");
        assert_eq!(
            inside(&root, "new/result.rs").expect("valid new path"),
            root.join("new/result.rs")
        );
        fs::remove_dir(root).expect("cleanup");
    }
    #[test]
    fn hash_instrument_matches_empty_sha256_vector() {
        let root = directory("hash");
        let file = root.join("empty");
        fs::write(&file, []).expect("empty fixture");
        assert_eq!(
            sha(&file).expect("hash"),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        fs::remove_file(file).expect("remove fixture");
        fs::remove_dir(root).expect("cleanup");
    }
}
