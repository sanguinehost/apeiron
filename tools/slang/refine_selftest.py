#!/usr/bin/env python3
"""Self-test for the Slang source-refinement gate harness.

Extracted from ``refine.py`` to keep each verified-checking file within the
1500-line bound. It imports the harness by name.
"""

from __future__ import annotations

from refine import *  # noqa: F401,F403

def _synthetic_repo(root: Path) -> Path:
    (root / "Slang").mkdir(parents=True, exist_ok=True)
    (root / "song").mkdir(parents=True, exist_ok=True)
    (root / "Slang" / "Synthetic.lean").write_text(
        "namespace Slang.Synthetic\n\ndef ok : Nat := 1\n\nend Slang.Synthetic\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Comment.lean").write_text(
        "namespace Slang.Comment\n\n/-\ndef gone : Nat := 1\n-/\n\nend Slang.Comment\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Broken.lean").write_text(
        "namespace Slang.Broken\n\ndef bad : Nat := \"not a nat\"\n\nend Slang.Broken\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Axiom.lean").write_text(
        "namespace Slang.Axiom\n\naxiom bad : False\n\nend Slang.Axiom\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Sorry.lean").write_text(
        "namespace Slang.Sorry\n\ntheorem bad : False := by sorry\n\nend Slang.Sorry\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Trivial.lean").write_text(
        "namespace Slang.Trivial\n\ntheorem source_binding : True := trivial\n\n"
        "end Slang.Trivial\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Legacy.lean").write_text(
        "namespace Slang.Legacy\n\n"
        "set_option warningAsError false\n"
        "theorem cone : (True = True) ∧ True ∧ "
        "Quot.mk (fun _ _ : Nat => True) 0 = Quot.mk (fun _ _ : Nat => True) 1 :=\n"
        "  ⟨propext Iff.rfl, Classical.choice ⟨True.intro⟩, Quot.sound trivial⟩\n\n"
        "end Slang.Legacy\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Canonical.lean").write_text(
        "namespace Apeiron.Table\n\n"
        "abbrev HMap (K V : Type) := K → V\n"
        "def get {K V : Type} (h : HMap K V) (k : K) : V := h k\n"
        "theorem get_insertWith_self : True := True.intro\n\n"
        "end Apeiron.Table\n\n"
        "namespace Slang.Canonical\n\n"
        "theorem allowed : Apeiron.Table.get (fun _ : Nat => 0) 0 = 0 := rfl\n"
        "theorem citation : True := Apeiron.Table.get_insertWith_self\n\n"
        "end Slang.Canonical\n",
        encoding="utf-8",
    )
    (root / "lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\n\npackage \"synthetic\"\n\n"
        "lean_lib «Synthetic» where\n  globs := #[.submodules `Slang]\n",
        encoding="utf-8",
    )
    (root / "lean-toolchain").write_text(SELFTEST_TOOLCHAIN + "\n", encoding="utf-8")
    (root / "song" / "synthetic.slang").write_text(
        "body { return 1; }\n", encoding="utf-8"
    )
    (root / "check.py").write_text(
        "import pathlib, sys\n"
        "text = pathlib.Path('song/synthetic.slang').read_text()\n"
        "if 'return 2;' in text:\n"
        "    print('synthetic body change rejected')\n"
        "    sys.exit(2)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    (root / "crash.py").write_text(
        "import pathlib, sys\n"
        "text = pathlib.Path('song/synthetic.slang').read_text()\n"
        "if 'boom' in text:\n"
        "    print('unrelated crash')\n"
        "    sys.exit(3)\n"
        "sys.exit(2 if 'return 2;' in text else 0)\n",
        encoding="utf-8",
    )
    return root

SYNTHETIC_COPY = [
    "song/synthetic.slang",
    "Slang/Synthetic.lean",
    "Slang/Comment.lean",
    "Slang/Broken.lean",
    "Slang/Axiom.lean",
    "Slang/Sorry.lean",
    "Slang/Trivial.lean",
    "Slang/Legacy.lean",
    "Slang/Canonical.lean",
    "lakefile.lean",
    "lean-toolchain",
    "check.py",
    "crash.py",
]

def _synthetic_stage_dict(
    root: Path,
    *,
    module: str = "Slang/Synthetic.lean",
    declaration: str | None = "Slang.Synthetic.ok",
    declaration_type: str | None = "Nat",
    prefix: str | None = None,
    binding: str = "exact",
    digest: str | None = None,
    steps: list[list[str]] | None = None,
    timeout: int = 60,
    old: str = "return 1;",
    new: str = "return 2;",
    expected_step: int = 0,
    expected_output: str = "synthetic body change rejected",
    control_status: str = ACTIVE,
    anchor: str | None = None,
    copy: list[str] | None = None,
    link: list[str] | None = None,
) -> dict:
    if digest is None:
        digest = sha256_file(root / "song" / "synthetic.slang")
    proof: dict = {"kind": "proof", "root": "bridge", "module": module}
    if declaration is not None:
        proof["declaration"] = declaration
        if declaration_type is not None:
            proof["declaration_type"] = declaration_type
    elif prefix is not None:
        proof["declaration_prefix"] = prefix
        proof["binding"] = binding
    else:
        proof["declaration_prefix"] = "Slang.Synthetic"
    raw: dict = {
        "id": "synthetic",
        "order": 1,
        "title": "synthetic stage",
        "summary": "self-test synthetic stage",
        "source_entry": "SE0",
        "rejecting_control": "SC0",
        "artifacts": [
            proof,
            {"kind": "source", "path": "song/synthetic.slang", "sha256": digest},
        ],
        "command": {
            "cwd": ".",
            "timeout": timeout,
            "steps": steps or [["python3", "check.py"]],
        },
        "copy": list(copy if copy is not None else SYNTHETIC_COPY),
        "link": list(link if link is not None else []),
        "control_status": control_status,
    }
    if control_status == ACTIVE:
        mutation: dict = {
            "path": "song/synthetic.slang",
            "old": old,
            "new": new,
            "expect": "synthetic body change rejected",
            "expected_step": expected_step,
            "expected_output": expected_output,
        }
        if anchor is not None:
            mutation["anchor"] = anchor
        raw["mutation"] = mutation
    else:
        raw["awaiting_reason"] = "synthetic control awaiting its real definition"
    return raw

def self_test() -> int:
    """Prove the rejection classes fire and the positive accepts."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    def expect(
        name: str, result: StageResult, status: str, *needles: str, extra: bool = True
    ) -> None:
        ok = result.status == status and all(n in result.reason for n in needles) and extra
        record(name, ok, f"status={result.status} reason={result.reason}")

    workdir = Path(tempfile.mkdtemp(prefix="slang-refine-self-", dir=tempfile.gettempdir()))
    try:
        repo = _synthetic_repo(workdir / "repo")
        roots = {"bridge": "."}
        default_manifest = (
            Artifact(
                kind=SOURCE_ARTIFACT,
                path="song/synthetic.slang",
                sha256=sha256_file(repo / "song" / "synthetic.slang"),
            ),
        )
        song_root = repo / "song"

        def stage_from(raw: dict) -> Stage:
            return parse_stage(
                raw,
                "synthetic",
                roots=roots,
                default_timeout=60,
                default_copy=(),
                default_link=(),
            )

        def run(
            raw: dict,
            *,
            controls: bool = True,
            manifest: tuple[Artifact, ...] | None = None,
        ) -> StageResult:
            return check_stage(
                stage_from(raw),
                roots,
                manifest if manifest is not None else default_manifest,
                repo,
                song_root,
                controls=controls,
                throwaway_root=workdir / "throwaway",
            )

        # 0. the positive control: fresh artifact, green baseline, mutation red at the
        #    declared step with the declared result, restored green.
        positive = run(_synthetic_stage_dict(repo), controls=True)
        record(
            "positive_present_and_fresh",
            positive.status == "pass"
            and positive.red_step == 0
            and positive.red_exit not in (None, 0)
            and positive.green_exit == 0,
            f"status={positive.status} reason={positive.reason}",
        )

        canonical_positive = run(
            _synthetic_stage_dict(
                repo,
                module="Slang/Canonical.lean",
                declaration="Slang.Canonical.allowed",
                declaration_type="Apeiron.Table.get (fun _ : Nat => 0) 0 = 0",
            ),
            controls=False,
        )
        record(
            "positive_admissible_canonical_binding",
            canonical_positive.status == "pass"
            and canonical_positive.canonical_bindings == ["Apeiron.Table.get"],
            f"status={canonical_positive.status} bindings={canonical_positive.canonical_bindings} "
            f"reason={canonical_positive.reason}",
        )

        # 1. every simple rejection class, including hole 1 (a block-commented
        #    declaration is text, not Lean code) and dependency-building.
        cases = [
            ("reject_missing_artifact", {"module": "Slang/Absent.lean"}, "blocked", False, ("missing proof target",)),
            ("reject_missing_declaration", {"declaration": "Slang.Synthetic.absent"}, "blocked", False, ("not loaded from real code",)),
            ("reject_commented_declaration", {"module": "Slang/Comment.lean", "declaration": "Slang.Comment.gone"}, "blocked", False, ("not loaded from real code",)),
            ("reject_stale_digest", {"digest": "0" * 64}, "blocked", False, ("stale source input",)),
            ("reject_timeout", {"steps": [["python3", "-c", "import time; time.sleep(30)"]], "timeout": 1}, "blocked", False, ("timeout",)),
            ("reject_nonzero_subprocess", {"steps": [["python3", "-c", "import sys; sys.exit(7)"]]}, "blocked", False, ("nonzero subprocess", "exit 7")),
            ("reject_unstartable_subprocess", {"steps": [["__apeiron_absent_tool__"]]}, "blocked", False, ("failed to start",)),
            ("reject_dependency_build_failure", {"module": "Slang/Broken.lean", "declaration": "Slang.Broken.bad", "declaration_type": "Nat"}, "blocked", False, ("proof dependency build failed",)),
            ("reject_axiom_target", {"module": "Slang/Axiom.lean", "declaration": "Slang.Axiom.bad", "declaration_type": "False"}, "blocked", False, ("is an axiom",)),
            ("reject_sorry_target", {"module": "Slang/Sorry.lean", "declaration": "Slang.Sorry.bad", "declaration_type": "False"}, "blocked", False, ("nonempty Stream 0 axiom set", "sorryAx")),
            ("reject_same_named_true_typed_theorem", {"module": "Slang/Trivial.lean", "declaration": "Slang.Trivial.source_binding", "declaration_type": "False"}, "blocked", False, ("wrong type",)),
            ("reject_wrong_exact_type", {"declaration_type": "Bool"}, "blocked", False, ("wrong type",)),
            ("reject_legacy_axiom_cone", {"module": "Slang/Legacy.lean", "declaration": "Slang.Legacy.cone", "declaration_type": "(True = True) ∧ True ∧ Quot.mk (fun _ _ : Nat => True) 0 = Quot.mk (fun _ _ : Nat => True) 1"}, "blocked", False, ("nonempty Stream 0 axiom set", "propext", "Classical.choice", "Quot.sound")),
            ("reject_canonical_citation", {"module": "Slang/Canonical.lean", "declaration": "Slang.Canonical.citation", "declaration_type": "True"}, "blocked", False, ("non-admissible Stream 0 canonical", "Apeiron.Table.get_insertWith_self")),
            ("reject_unbound_stage", {"declaration": None, "declaration_type": None, "prefix": "Slang.Synthetic", "binding": UNBOUND}, "blocked", False, ("unbound proof headline",)),
            ("reject_surviving_mutation", {"old": "body {", "new": "body  {"}, "fail", True, ("mutation survived",)),
            ("reject_awaiting_control", {"control_status": AWAITING}, "blocked", True, (AWAITING,)),
        ]
        for name, kwargs, status, controls, needles in cases:
            expect(
                name,
                run(_synthetic_stage_dict(repo, **kwargs), controls=controls),
                status,
                *needles,
            )
        # A schema violation is a parse rejection, not a stage result.
        parse_cases = [
            ("reject_exact_without_type", {"declaration_type": None}, None),
            ("reject_prefix_without_unbound", {"declaration": None, "declaration_type": None, "prefix": "Slang.Other", "binding": "exact"}, None),
            ("reject_unbound_with_type", {"declaration": None, "declaration_type": None, "prefix": "Slang.Other", "binding": UNBOUND}, ("declaration_type", "Nat")),
        ]
        for name, kwargs, inject in parse_cases:
            raw = _synthetic_stage_dict(repo, **kwargs)
            if inject is not None:
                raw["artifacts"][0][inject[0]] = inject[1]
            try:
                stage_from(raw)
            except ControlsError as exc:
                record(name, True, f"raised: {exc}")
            else:
                record(name, False, "schema violation was not rejected")

        # 2. hole 2: an unrelated crash, and a red run at the wrong step, are not
        #    controls even though the command exits nonzero.
        expect(
            "reject_unrelated_crash",
            run(
                _synthetic_stage_dict(
                    repo,
                    steps=[["python3", "crash.py"]],
                    old="return 1;",
                    new="boom",
                    expected_output="synthetic body change rejected",
                )
            ),
            "fail",
            "expected rejecting result",
        )
        expect(
            "reject_wrong_red_step",
            run(
                _synthetic_stage_dict(
                    repo,
                    steps=[["python3", "crash.py"], ["python3", "check.py"]],
                    old="return 1;",
                    new="boom",
                    expected_step=1,
                    expected_output="synthetic body change rejected",
                )
            ),
            "fail",
            "control target not reached",
        )

        # 3. the mutation engine fails loudly on an absent or ambiguous target.
        try:
            Mutation(path="x", old="absent", new="y").apply("alpha\nbeta\n")
            record("mutation_absent_raises", False, "missing target did not raise")
        except MutationError as exc:
            record("mutation_absent_raises", True, f"raised: {exc}")
        try:
            Mutation(path="x", old="a", new="b").apply("a\na\n")
            record("mutation_ambiguous_raises", False, "ambiguous target did not raise")
        except MutationError as exc:
            record("mutation_ambiguous_raises", True, f"raised: {exc}")

        # 4. hole 4: a symlinked ancestor of the mutation target is refused and the
        #    real source is left untouched.
        before = (repo / "song" / "synthetic.slang").read_bytes()
        escape = run(_synthetic_stage_dict(repo, link=["song"]), controls=True)
        after = (repo / "song" / "synthetic.slang").read_bytes()
        record(
            "reject_symlink_ancestor_escape",
            escape.status == "blocked"
            and "outside the resolved throwaway root" in escape.reason
            and before == after,
            f"status={escape.status} reason={escape.reason}",
        )

        # 5. hole 3: single-root resolution. The explicit Song root wins over an
        #    ordinary sibling checkout, and there is no sibling fallback.
        shadow = workdir / "shadow"
        explicit = shadow / "repo" / "song"
        sibling = shadow / "song"
        explicit.mkdir(parents=True, exist_ok=True)
        sibling.mkdir(parents=True, exist_ok=True)
        (explicit / "synthetic.slang").write_text("explicit\n", encoding="utf-8")
        (sibling / "synthetic.slang").write_text("sibling\n", encoding="utf-8")
        won = resolve_one("song/synthetic.slang", shadow / "repo", explicit)
        record(
            "explicit_song_root_wins",
            won is not None
            and won.resolve() == (explicit / "synthetic.slang").resolve(),
            f"resolved={won}",
        )
        (explicit / "synthetic.slang").unlink()
        fallback = resolve_one("song/synthetic.slang", shadow / "repo", explicit)
        record(
            "explicit_song_root_no_sibling_fallback",
            fallback is None,
            f"resolved={fallback}",
        )

        # 6. the manifest rejects a stale or missing selected source.
        for name, path, needle in (
            ("reject_manifest_stale", "song/synthetic.slang", "stale source input"),
            ("reject_manifest_missing", "song/absent.slang", "missing source input"),
        ):
            manifest = (Artifact(kind=SOURCE_ARTIFACT, path=path, sha256="0" * 64),)
            expect(
                name,
                run(_synthetic_stage_dict(repo), controls=False, manifest=manifest),
                "blocked",
                needle,
            )

        # 7. controls.json parses, declares the eight stages and the expected
        #    control status of each.
        try:
            loaded = load_controls()
            ids = {stage.id for stage in loaded.stages}
            expected = {
                "decode",
                "machine",
                "source",
                "init",
                "lookup",
                "write",
                "resources",
                "replacement-link",
            }
            record(
                "controls_schema",
                ids == expected and len(loaded.stages) == 8,
                f"ids={sorted(ids)}",
            )
            by_id = {stage.id: stage for stage in loaded.stages}
            for name, root in loaded.roots.items():
                record(
                    f"controls_root:{name}",
                    (REPO_ROOT / root).exists() or (REPO_ROOT.parent / root).exists(),
                    f"root={root}",
                )
            for name in ("lookup",):
                stage = by_id[name]
                record(
                    f"controls_awaiting:{name}",
                    stage.control_status == AWAITING
                    and stage.mutation is None
                    and bool(stage.awaiting_reason),
                    f"status={stage.control_status}",
                )
            write = by_id["write"].mutation
            record(
                "controls_anchor:write",
                write is not None
                and write.anchor is not None
                and "oldVal" in write.new
                and "else value" in write.old,
                f"anchor={write.anchor if write else None}",
            )
            for name, step, needle in (
                ("decode", 1, "statement count differs"),
                ("machine", 0, "EvalExpr.conditionalTrue"),
            ):
                control = by_id[name].mutation
                record(
                    f"controls_expect:{name}",
                    control is not None
                    and control.expected_step == step
                    and needle in control.expected_output,
                    f"step={control.expected_step if control else None}",
                )
            record(
                "controls_manifest",
                len(loaded.manifest) >= 8
                and all(entry.kind == SOURCE_ARTIFACT for entry in loaded.manifest),
                f"entries={len(loaded.manifest)}",
            )
        except ControlsError as exc:
            record("controls_schema", False, str(exc))

        # 8. every declared active mutation resolves uniquely against its real source.
        try:
            loaded = load_controls()
        except ControlsError:
            loaded = Controls(roots={}, manifest=(), stages=())
        for stage in loaded.stages:
            if stage.mutation is None:
                continue
            resolved = resolve_one(stage.mutation.path, REPO_ROOT, DEFAULT_SONG_ROOT)
            if resolved is None:
                record(
                    f"mutation_resolves:{stage.id}",
                    False,
                    f"missing source {stage.mutation.path}",
                )
                continue
            try:
                pristine = resolved.read_text(encoding="utf-8")
                mutated = stage.mutation.apply(pristine)
                record(
                    f"mutation_resolves:{stage.id}",
                    mutated != pristine,
                    "target resolves uniquely and changes the source",
                )
            except (MutationError, OSError) as exc:
                record(f"mutation_resolves:{stage.id}", False, str(exc))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    for check in checks:
        print(json.dumps(check, sort_keys=True))
    failed = [check for check in checks if not check["ok"]]
    print(
        json.dumps(
            {"self_test": "pass" if not failed else "fail", "checks": len(checks)},
            sort_keys=True,
        )
    )
    return EXIT_OK if not failed else EXIT_BLOCKED
