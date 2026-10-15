//! Fixed synthetic controls; no caller-provided fixture or production registration.
use super::*;
const FIXTURE: &str = r#"import DispatchAdmission
set_option warningAsError true
namespace Fixture
def before (x : Nat) := x + 0
def after (x : Nat) := x
def reachable (_ : Nat) : Prop := True
def admissible (_ : Nat) : Prop := True
def beforeCost (_ : Nat) := 10
def afterCost (_ : Nat) := 8
def admission : ProofDispatch.Admission before after reachable admissible beforeCost afterCost where
  witness := 1
  joint := ⟨True.intro, True.intro⟩
  preserves := fun _ _ _ => rfl
  improves := fun _ _ _ => by dsimp [beforeCost, afterCost]; decide
end Fixture
"#;
/// O(J) JSON bytes and registry write; exact immutable worker remains fixture-local.
fn proposal(contract: &Value, registry: &Path) -> Checked<Value> {
    fs::write(
        registry,
        serde_json::to_vec(&json!({"schema":2,"contracts":{"fixture":contract}}))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(
        json!({"task":"APEIRON-RUSTADMIT-CONTROL","contract":"fixture","registry_sha256":sha(registry)?,"identity":contract["identity"],"owned_files":["src/result.rs"],"worker":contract["worker"]}),
    )
}
/// O(B+L) current proof/log bytes plus the bounded admission pipeline; marker must stay absent.
fn reject(
    ctx: &mut Context,
    root: &Path,
    registry: &Path,
    p: &Value,
    name: &str,
    expected: &str,
) -> Checked<()> {
    let error = ctx
        .launch(p, registry, root, name, true)
        .err()
        .ok_or_else(|| format!("CONTROL_SURVIVED {name}"))?;
    require(
        error.contains(expected),
        &format!("CONTROL_WRONG_DIAGNOSTIC {name}: {error}"),
    )?;
    require(
        !root.join("worker-ran").exists(),
        &format!("CONTROL_WORKER_STARTED {name}"),
    )?;
    println!("REJECT {name}");
    Ok(())
}
/// Fixed fifteen controls share the runtime deadline; O(B+J+L) source/JSON/log bytes.
pub(super) fn run(ctx: &mut Context) -> Checked<()> {
    let root = ctx.stage.join("fixture");
    fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    let source = root.join("src/source.rs");
    fs::write(&source, "// fixture only\n").map_err(|e| e.to_string())?;
    let proof = root.join("Contract.lean");
    fs::write(&proof, FIXTURE).map_err(|e| e.to_string())?;
    let marker = root.join("worker-ran");
    let registry = root.join("registry.json");
    let contract = json!({"grade":"source-connected-cost-model","source_kind":"rust-aeneas","namespace":"Fixture","before":"Fixture.before","after":"Fixture.after","reachable":"Fixture.reachable","admissible":"Fixture.admissible","before_cost":"Fixture.beforeCost","after_cost":"Fixture.afterCost","certificate":"Fixture.admission","proof_file":"Contract.lean","identity":{"src/source.rs":sha(&source)?},"owned_files":["src/result.rs"],"worker":["python3","-c","from pathlib import Path; Path('worker-ran').write_text('checked')"]});
    let p = proposal(&contract, &registry)?;
    ctx.launch(&p, &registry, &root, "positive", true)?;
    require(
        fs::read_to_string(&marker).map_err(|e| e.to_string())? == "checked",
        "POSITIVE_NO_WORKER",
    )?;
    fs::remove_file(&marker).map_err(|e| e.to_string())?;
    let weak = format!(
        "{}theorem admission : True := True.intro\nend Fixture\n",
        FIXTURE
            .split("def admission")
            .next()
            .ok_or("CONTROL_FIXTURE_MISSING")?
    );
    let extra=FIXTURE.replace("  preserves := fun _ _ _ => rfl","  preserves := untrusted").replace("def admission :","axiom untrusted : ∀ x, reachable x → admissible x → after x = before x\ndef admission :");
    let changes = [
        (
            "vacuous-domain",
            "ADMISSION_LEAN_REJECT",
            FIXTURE.replace(
                "admissible (_ : Nat) : Prop := True",
                "admissible (_ : Nat) : Prop := False",
            ),
        ),
        (
            "unreachable-domain",
            "ADMISSION_LEAN_REJECT",
            FIXTURE.replace(
                "reachable (_ : Nat) : Prop := True",
                "reachable (_ : Nat) : Prop := False",
            ),
        ),
        (
            "rounded-reassociation",
            "ADMISSION_LEAN_REJECT",
            FIXTURE.replace("x + 0", "(x / 2) * 2"),
        ),
        (
            "no-cost-reduction",
            "ADMISSION_LEAN_REJECT",
            FIXTURE.replace("afterCost (_ : Nat) := 8", "afterCost (_ : Nat) := 10"),
        ),
        ("weaker-theorem", "ADMISSION_CERTIFICATE_REJECT", weak),
        (
            "sorry",
            "ADMISSION_LEAN_REJECT",
            FIXTURE.replace("preserves := fun _ _ _ => rfl", "preserves := by sorry"),
        ),
        ("extra-axiom", "ADMISSION_CERTIFICATE_REJECT", extra),
    ];
    for (name, diagnostic, changed) in changes {
        fs::write(&proof, changed).map_err(|e| e.to_string())?;
        let p = proposal(&contract, &registry)?;
        reject(ctx, &root, &registry, &p, name, diagnostic)?;
    }
    fs::write(&proof, FIXTURE).map_err(|e| e.to_string())?;
    for (name, diagnostic) in [
        ("unknown-contract", "ADMISSION_UNREGISTERED_CONTRACT"),
        ("stale-source", "ADMISSION_STALE_SOURCE"),
        ("stale-registry", "ADMISSION_STALE_REGISTRY"),
        ("identity-substitution", "ADMISSION_IDENTITY_SUBSTITUTION"),
        ("worker-substitution", "ADMISSION_WORKER_SUBSTITUTION"),
        ("unowned-path", "ADMISSION_UNOWNED_PATH"),
        ("too-many-files", "ADMISSION_OWNERSHIP_REJECT"),
        ("boolean-approval", "ADMISSION_SCHEMA_REJECT"),
    ] {
        let mut p = proposal(&contract, &registry)?;
        match name {
            "unknown-contract" => p["contract"] = json!("float-block"),
            "stale-source" => fs::write(&source, "// changed\n").map_err(|e| e.to_string())?,
            "stale-registry" => p["registry_sha256"] = json!("0".repeat(64)),
            "identity-substitution" => p["identity"] = json!({"src/source.rs":"0".repeat(64)}),
            "worker-substitution" => {
                p["worker"] = json!(["python3", "-c", "raise SystemExit('bypass')"])
            }
            "unowned-path" => p["owned_files"] = json!(["src/other.rs"]),
            "too-many-files" => p["owned_files"] = json!(["a", "b", "c", "d"]),
            "boolean-approval" => p["approved"] = json!(true),
            _ => return Err("CONTROL_NAME_REJECT".into()),
        }
        reject(ctx, &root, &registry, &p, name, diagnostic)?;
        fs::write(&source, "// fixture only\n").map_err(|e| e.to_string())?;
    }
    let p = proposal(&contract, &registry)?;
    ctx.launch(&p, &registry, &root, "restored", true)?;
    require(
        fs::read_to_string(marker).map_err(|e| e.to_string())? == "checked",
        "RESTORE_NO_WORKER",
    )?;
    println!("SELFTEST_SYNTHETIC_ONLY positive=1 controls=15 restored=1");
    Ok(())
}
