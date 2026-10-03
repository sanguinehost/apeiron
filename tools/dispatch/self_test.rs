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
    run_shader(ctx)?;
    Ok(())
}

/// Machinery identities the launcher requires for a `shader-spirv` contract.
const SHADER_CLOSURE: [&str; 13] = [
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
];

/// O(P) file hashes for the registered shader execution unit and its machinery closure.
fn shader_identity(
    root: &Path,
    proof_rel: &str,
    proof_hash: &str,
    id_rel: &str,
    id_hash: &str,
) -> Checked<Value> {
    let mut map = serde_json::Map::new();
    for rel in [
        "src/spirv/q8_tile.comp",
        "tools/gpu/q8_driver.c",
        "tools/gpu/q8_admission_gate.py",
    ] {
        map.insert(rel.to_owned(), json!(sha(&root.join(rel))?));
    }
    map.insert(id_rel.to_owned(), json!(id_hash));
    map.insert(proof_rel.to_owned(), json!(proof_hash));
    for rel in SHADER_CLOSURE {
        map.insert(rel.to_owned(), json!(sha(&root.join(rel))?));
    }
    Ok(Value::Object(map))
}

/// The registered non-Rust contract shape; every leg is explicit and bound into `identity`.
fn shader_contract(
    identity: Value,
    proof_rel: &str,
    id_rel: &str,
    certificate: &str,
    worker: &Value,
) -> Value {
    json!({
        "grade": "source-connected-cost-model",
        "source_kind": "shader-spirv",
        "namespace": "Q8TileSpirv",
        "before": "Q8TileSpirv.before",
        "after": "Q8TileSpirv.after",
        "reachable": "Q8TileSpirv.reachable",
        "admissible": "Q8TileSpirv.admissible",
        "before_cost": "Q8TileSpirv.beforeCost",
        "after_cost": "Q8TileSpirv.afterCost",
        "certificate": certificate,
        "proof_file": proof_rel,
        "build_target": "Q8TileSpirv",
        "shader_source": {
            "shader": "src/spirv/q8_tile.comp",
            "entry_point": "main",
            "dispatch_driver": "tools/gpu/q8_driver.c",
            "shader_abi": "set0 storage buffers + push-constant Params{M,tokens,blocks}; local_size 8x8x1",
            "toolchain": "glslc",
            "target_env": "vulkan1.3",
            "source_gate": "tools/gpu/q8_admission_gate.py",
            "identity_file": id_rel
        },
        "observation": {
            "protocol": "python3 tools/gpu/q8_gate.py --profile synthetic --fixture <path>",
            "metric": "device output value and refusal flag parity against the CPU oracle",
            "interval": "one whole gate invocation"
        },
        "cost": {
            "protocol": "python3 tools/gpu/q8_shapes.py --reps 5",
            "metric": "device_ns median (GMAC/s derived) across one dispatch",
            "interval": "five reps per shape; median"
        },
        "identity": identity,
        "owned_files": ["src/spirv/q8_tile.comp"],
        "worker": worker
    })
}

/// O(B+P+J+L) real source/identity/certificate bytes plus the bounded admission pipeline.
/// Production-path positive and controls for the non-Rust source kind; no GPU execution.
pub(super) fn run_shader(ctx: &mut Context) -> Checked<()> {
    let root = ctx.root.clone();
    let dir = root.join("tools/dispatch/contracts/.selftest");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let registry = dir.join("registry.json");
    let marker = ctx.stage.join("shader-worker-ran");
    let _ = fs::remove_file(&marker);
    let proof_file = "proof/dispatch/contracts/Q8TileSpirv.lean";
    let proof_src = fs::read_to_string(root.join(proof_file)).map_err(|e| e.to_string())?;
    let worker = json!([
        "python3",
        "-c",
        format!(
            "import pathlib; pathlib.Path({:?}).write_text('checked')",
            marker
        )
    ]);
    let owned = json!(["src/spirv/q8_tile.comp"]);

    // A registered proposal against the current registry bytes.
    let proposal = |contract: &Value, registry: &Path| -> Checked<Value> {
        Ok(json!({
            "task": "APEIRON-GPU-ADMISSION-CONTROL",
            "contract": "q8-tile-spirv",
            "registry_sha256": sha(registry)?,
            "identity": contract["identity"],
            "owned_files": owned,
            "worker": contract["worker"],
        }))
    };
    let write_registry = |contract: &Value, registry: &Path| -> Checked<()> {
        fs::write(
            registry,
            serde_json::to_vec(&json!({"schema": 2, "contracts": {"q8-tile-spirv": contract}}))
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    };

    // Positive: real shader, real source-identity gate, real certificate.
    let proof_hash = sha(&root.join(proof_file))?;
    let id_hash = sha(&root.join("tools/gpu/q8_spirv_identity.json"))?;
    let identity = shader_identity(
        &root,
        proof_file,
        &proof_hash,
        "tools/gpu/q8_spirv_identity.json",
        &id_hash,
    )?;
    let contract = shader_contract(
        identity,
        proof_file,
        "tools/gpu/q8_spirv_identity.json",
        "Q8TileSpirv.admission",
        &worker,
    );
    write_registry(&contract, &registry)?;
    let p = proposal(&contract, &registry)?;
    ctx.launch(&p, &registry, &root, "shader-positive", false)?;
    require(
        fs::read_to_string(&marker).map_err(|e| e.to_string())? == "checked",
        "SHADER_POSITIVE_NO_WORKER",
    )?;
    fs::remove_file(&marker).map_err(|e| e.to_string())?;

    // Refusal controls: mismatch, absent improvement, unauthorized domain, wrong certificate.
    let reject = |ctx: &mut Context, p: &Value, name: &str, expected: &str| -> Checked<()> {
        let error = ctx
            .launch(p, &registry, &root, name, false)
            .err()
            .ok_or_else(|| format!("SHADER_CONTROL_SURVIVED {name}"))?;
        require(
            error.contains(expected),
            &format!("SHADER_CONTROL_WRONG_DIAGNOSTIC {name}: {error}"),
        )?;
        require(
            !marker.exists(),
            &format!("SHADER_CONTROL_WORKER_STARTED {name}"),
        )?;
        println!("SHADER_REJECT {name}");
        Ok(())
    };

    // (1) Source-identity mismatch: a stale SPIR-V digest in the declared identity file.
    let stale_id = dir.join("q8_spirv_identity_stale.json");
    let mut stale: Value = serde_json::from_slice(
        &fs::read(root.join("tools/gpu/q8_spirv_identity.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    stale["spirv_sha256"] = json!("0".repeat(64));
    fs::write(
        &stale_id,
        serde_json::to_vec(&stale).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let stale_rel = "tools/dispatch/contracts/.selftest/q8_spirv_identity_stale.json";
    let stale_hash = sha(&stale_id)?;
    let id2 = shader_identity(&root, proof_file, &proof_hash, stale_rel, &stale_hash)?;
    let c2 = shader_contract(id2, proof_file, stale_rel, "Q8TileSpirv.admission", &worker);
    write_registry(&c2, &registry)?;
    reject(
        ctx,
        &proposal(&c2, &registry)?,
        "shader-source-mismatch",
        "ADMISSION_SOURCE_IDENTITY_REJECT",
    )?;

    // (2) Absent improvement: afterCost no longer strictly below beforeCost.
    let nocost_rel = "tools/dispatch/contracts/.selftest/Q8TileSpirvNoCost.lean";
    let nocost = proof_src.replace(
        "def afterCost (_ : Request) : Nat := 1",
        "def afterCost (_ : Request) : Nat := 8",
    );
    require(nocost != proof_src, "SHADER_CONTROL_ANCHOR no-cost")?;
    fs::write(root.join(nocost_rel), &nocost).map_err(|e| e.to_string())?;
    let id3 = shader_identity(
        &root,
        nocost_rel,
        &sha(&root.join(nocost_rel))?,
        "tools/gpu/q8_spirv_identity.json",
        &id_hash,
    )?;
    let c3 = shader_contract(
        id3,
        nocost_rel,
        "tools/gpu/q8_spirv_identity.json",
        "Q8TileSpirv.admission",
        &worker,
    );
    write_registry(&c3, &registry)?;
    reject(
        ctx,
        &proposal(&c3, &registry)?,
        "shader-absent-improvement",
        "ADMISSION_LEAN_REJECT",
    )?;

    // (3) Unauthorized domain: the declared admissible domain becomes empty.
    let empty_rel = "tools/dispatch/contracts/.selftest/Q8TileSpirvEmpty.lean";
    let empty = proof_src.replace(
        "def admissible (_ : Request) : Prop := True",
        "def admissible (_ : Request) : Prop := False",
    );
    require(empty != proof_src, "SHADER_CONTROL_ANCHOR empty-domain")?;
    fs::write(root.join(empty_rel), &empty).map_err(|e| e.to_string())?;
    let id4 = shader_identity(
        &root,
        empty_rel,
        &sha(&root.join(empty_rel))?,
        "tools/gpu/q8_spirv_identity.json",
        &id_hash,
    )?;
    let c4 = shader_contract(
        id4,
        empty_rel,
        "tools/gpu/q8_spirv_identity.json",
        "Q8TileSpirv.admission",
        &worker,
    );
    write_registry(&c4, &registry)?;
    reject(
        ctx,
        &proposal(&c4, &registry)?,
        "shader-unauthorized-domain",
        "ADMISSION_LEAN_REJECT",
    )?;

    // (4) Wrong certificate: the named certificate term does not exist.
    let id5 = shader_identity(
        &root,
        proof_file,
        &proof_hash,
        "tools/gpu/q8_spirv_identity.json",
        &id_hash,
    )?;
    let c5 = shader_contract(
        id5,
        proof_file,
        "tools/gpu/q8_spirv_identity.json",
        "Q8TileSpirv.absent",
        &worker,
    );
    write_registry(&c5, &registry)?;
    reject(
        ctx,
        &proposal(&c5, &registry)?,
        "shader-wrong-certificate",
        "ADMISSION_CERTIFICATE_REJECT",
    )?;

    // (5) Unregistered contract.
    let id6 = shader_identity(
        &root,
        proof_file,
        &proof_hash,
        "tools/gpu/q8_spirv_identity.json",
        &id_hash,
    )?;
    let c6 = shader_contract(
        id6,
        proof_file,
        "tools/gpu/q8_spirv_identity.json",
        "Q8TileSpirv.admission",
        &worker,
    );
    write_registry(&c6, &registry)?;
    let mut p6 = proposal(&c6, &registry)?;
    p6["contract"] = json!("float-block");
    reject(
        ctx,
        &p6,
        "shader-unregistered",
        "ADMISSION_UNREGISTERED_CONTRACT",
    )?;

    // Restoration: the unmodified contract is accepted again on the same identities.
    write_registry(&contract, &registry)?;
    let p = proposal(&contract, &registry)?;
    ctx.launch(&p, &registry, &root, "shader-restored", false)?;
    require(
        fs::read_to_string(&marker).map_err(|e| e.to_string())? == "checked",
        "SHADER_RESTORE_NO_WORKER",
    )?;
    fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    println!("SELFTEST_SHADER positive=1 controls=5 restored=1");
    Ok(())
}
