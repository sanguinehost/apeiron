#!/usr/bin/env python3
"""Extract and check the real relocation traversal in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost. The
temporary manifest wires the real `src/table.rs` by `#[path]`, which pulls the real
`src/table/{state,locator,prepared,keyrep,probe,partition,scan,parallel}.rs`; the only new code is
the typed forwarding wrapper that calls the real `Table::relocate_all` body. No substitute
algorithm is extracted, and the shared `proof/extract/lib.rs` is not used: it does not currently
compile, so this gate builds its own crate root and reaches the same real body through the same
inherent-method call.

  proof/resource/RegenRelocateAll_gate.py --prepare   regenerate proof/resource/RegenRelocateAllSource.lean
  proof/resource/RegenRelocateAll_gate.py             check extraction freshness, then build the proof
  proof/resource/RegenRelocateAll_gate.py --controls  also run the real-source mutation/restore control

The audited resource `proof/resource/RegenRelocateAllResource.lean` is absent: no committed revision
exists to recover, so every mode but `--prepare` fails closed on the first stage with
`REGEN_RELOCATE_ALL_PROOF_ARTIFACT_MISSING` before extraction is paid for. The gate is kept.

The lane is /tmp/opencode/apeiron-lane-2.lock. Budget: 540 s wall clock for the whole invocation,
serialized on that lane, LEAN_NUM_THREADS=1, MemoryMax=12G, MemorySwapMax=0.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-2.lock'
if __name__ == '__main__' and os.environ.get('REGEN_RELOCATE_ALL_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','REGEN_RELOCATE_ALL_IN_LANE=1',
        'REGEN_RELOCATE_ALL_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts
# and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/regen-relocate-all-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/regen-relocate-all-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

# The real source closure the wrapper compiles. `src/table.rs` declares its children textually and
# imports `crate::cell::Sigma` and `crate::coord::{address,locator}`, so the wrapper also wires the
# real `src/{scalar,cell,coord}.rs` and the real `src/coord/` and `src/table/` directories.
# `src/table/tests.rs` is `#[cfg(test)]` and is compiled by the `cargo test` step only.
CLOSURE = ['scalar.rs', 'cell.rs', 'coord.rs', 'table.rs']
COPY_DIRS = ['coord', 'table']
# The mutation target: the traversal's single index advance. `entry_at += 1` is the whole work
# measure of `Table::relocate_all`; making it advance by two makes each step skip one planned
# entry, which the per-step increment, the strictly decreasing rank and the closed-form step count
# (`body_step`, `rank_strictly_decreases`, `loop_steps_closed_form`) all forbid. Same byte length, so
# source line numbers do not move.
MUTATION = ('entry_at += 1;', 'entry_at += 2;')

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('REGEN_RELOCATE_ALL_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('REGEN_RELOCATE_ALL_BUDGET_EXHAUSTED')
    log = OUT / (SCRATCH.name+'-'+label+'.log')
    begin = time.monotonic()
    with log.open('wb') as f:
        p = subprocess.Popen(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ,'LEAN_NUM_THREADS':'1'}, start_new_session=True)
        try:
            p.wait(timeout=left)
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); raise
    text = log.read_text(errors='replace')
    RECORDS.append(dict(label=label,argv=argv,exit=p.returncode,seconds=time.monotonic()-begin,log=str(log)))
    if require and p.returncode:
        raise RuntimeError(f'{label}_REJECT {p.returncode}\n{text[-3500:]}')
    return p.returncode,text

LIB_TEMPLATE = '''#[path = {scalar}]
pub mod scalar;
#[path = {cell}]
pub mod cell;
#[path = {coord}]
pub mod coord;
#[path = {table}]
pub mod table;

/// Crate-root alias of the face token the real `src/lib.rs` re-exports as `pub use cell::{{Sigma}}`;
/// the real `#[cfg(test)]` modules in `src/table/{{partition,probe}}.rs` import `crate::Sigma`.
pub use cell::Sigma;

/// The elliptic word table instance the real `proof/extract/lib.rs:138-139` names, so the wrapper
/// reaches the same real body at the same carrier.
pub type WordRegenTable = table::Table<i64, i64, table::WordHome, 16,
    table::keyrep::Regen<table::keyrep::I64WordCodec>>;

/// Typed forwarding wrapper for the real relocation traversal, fixing the three argument types of
/// `proof/extract/lib.rs:581-591` and calling the real `Table::relocate_all` body
/// (`src/table.rs:1017-1027`) through the same inherent method the production caller uses. It
/// restates no guard, no loop and no per-entry write, and the shared `proof/extract/lib.rs` is not
/// pulled in.
pub fn word_regen_relocate_all(
    mut state: WordRegenTable,
    plan: &table::RebuildPlan,
    mut reserved: table::RebuildReserved<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64>,
) -> (
    WordRegenTable,
    table::RebuildReserved<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64>,
) {{
    state.relocate_all(plan, &mut reserved);
    (state, reserved)
}}
'''

def lib_source(base):
    """Absolute `#[path]` into the real source root; only the root varies with the control copy."""
    return LIB_TEMPLATE.format(**{k: json.dumps(str(base/(k+'.rs'))) for k in
                                  ['scalar', 'cell', 'coord', 'table']})

def copy_closure(dest):
    """Bounded copy of the real extraction closure; the working tree is never edited."""
    dest.mkdir(parents=True)
    for name in CLOSURE:
        shutil.copyfile(ROOT/'src'/name, dest/name)
    for name in COPY_DIRS:
        shutil.copytree(ROOT/'src'/name, dest/name)
    return dest

def extract(source_root, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="regen_relocate_all"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[features]\ndefault = []\ninjection = []\nread-stats = []\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source_root))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'regen_relocate_all.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_regen_relocate_all','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'RegenRelocateAll.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below.
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source_root/'table.rs'), 'src/table.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('REGEN_RELOCATE_ALL_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage regenRelocateAllProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib RegenRelocateAll where\n  roots := #[`RegenRelocateAllSource,`RegenRelocateAllResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='regenRelocateAllProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

PROOF_ARTIFACT = ROOT/'proof/resource/RegenRelocateAllResource.lean'

def proof_artifact_guard():
    """Fail closed on the unproven hand-written resource before any extraction is paid for.

    `RegenRelocateAllResource.lean` has no generator: `--prepare` writes only
    `RegenRelocateAllSource.lean`. This script is untracked at 028c8aa and the resource is absent from
    every ref, from `git log --all`, from `--diff-filter=D`, from every worktree and from the
    registered scratch roots, so the audit marker `REGEN_RELOCATE_ALL_KERNEL_OK` this gate requires
    has no artifact to come from. The gate is kept: the obligation is open until the resource is
    written and this gate re-runs green.
    """
    if not PROOF_ARTIFACT.is_file():
        print('REGEN_RELOCATE_ALL_PROOF_ARTIFACT_MISSING '+str(PROOF_ARTIFACT))
        print('REGEN_RELOCATE_ALL_OPEN_OBLIGATION: this resource has no committed revision to '
              'recover; it must be written and audited (kernel marker '
              'REGEN_RELOCATE_ALL_KERNEL_OK) before this gate can pass. Unproven: the '
              'Table::relocate_all refinement. No silent pass.')
        raise SystemExit(2)

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    # `--prepare` writes the generated source and returns; the resource is the audited proof.
    if not a.prepare:
        proof_artifact_guard()
    source=ROOT/'src'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/RegenRelocateAllSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('REGEN_RELOCATE_ALL_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('REGEN_RELOCATE_ALL_EXTRACTION_DRIFT')
    proof=PROOF_ARTIFACT
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('REGEN_RELOCATE_ALL_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'RegenRelocateAllSource.lean');shutil.copyfile(proof,p/'RegenRelocateAllResource.lean')
    code,text=run(['lake','build','RegenRelocateAllResource'],p,'fresh-proof')
    if 'REGEN_RELOCATE_ALL_KERNEL_OK' not in text:raise RuntimeError('REGEN_RELOCATE_ALL_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: `entry_at += 1` is the traversal's whole index advance. Advancing by
        # two makes each step skip one planned entry, which the per-step increment, the strictly
        # decreasing rank and the closed-form step count below forbid. The copy is in scratch; the
        # working tree is never edited.
        original=(source/'table.rs').read_text()
        if original.count(MUTATION[0])!=1:raise RuntimeError('REGEN_RELOCATE_ALL_CONTROL_TARGET_MISSING')
        mutant=copy_closure(SCRATCH/'mutant-src')
        (mutant/'table.rs').write_text(original.replace(*MUTATION))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'RegenRelocateAllSource.lean')
        code,text=run(['lake','build','RegenRelocateAllResource'],p,'mutant-proof',require=False)
        if code!=1 or 'RegenRelocateAllResource.lean:' not in text or not any(x.lower() in text.lower() for x in ['unsolved goals','type mismatch','Tactic','Not a definitional equality']):
            raise RuntimeError('REGEN_RELOCATE_ALL_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('REGEN_RELOCATE_ALL_REAL_SOURCE_CONTROL_RED')
        (mutant/'table.rs').write_text(original)
        restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('REGEN_RELOCATE_ALL_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'RegenRelocateAllSource.lean');run(['lake','build','RegenRelocateAllResource'],p,'restored-proof')
        print('REGEN_RELOCATE_ALL_REAL_SOURCE_CONTROL_RESTORED')
    print('REGEN_RELOCATE_ALL_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    except SystemExit as e:
        status=e.code if isinstance(e.code,int) else 1
        raise
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/table.rs',ROOT/'src/table/state.rs',ROOT/'src/table/prepared.rs',ROOT/'src/table/locator.rs',ROOT/'src/table/keyrep.rs',ROOT/'src/table/partition.rs',ROOT/'src/table/probe.rs',ROOT/'src/table/parallel.rs',ROOT/'src/table/tests.rs',ROOT/'src/cell.rs',ROOT/'src/scalar.rs',ROOT/'src/coord.rs',ROOT/'src/coord/address.rs',ROOT/'src/coord/locator.rs',ROOT/'src/coord/locator_core.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/RegenRelocateAll_gate.py',ROOT/'proof/resource/RegenRelocateAllSource.lean',ROOT/'proof/resource/RegenRelocateAllResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('REGEN_RELOCATE_ALL_DEADLINE'),records=RECORDS,proof_artifact=str(PROOF_ARTIFACT),proof_artifact_present=PROOF_ARTIFACT.is_file(),open_obligation=None if PROOF_ARTIFACT.is_file() else 'RegenRelocateAllResource.lean absent from the tree and from every git ref; the Table::relocate_all refinement (kernel marker REGEN_RELOCATE_ALL_KERNEL_OK) is unproven',sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts planned entries visited and relocation steps of the extracted traversal itself under the Aeneas model, not cycles or GPU time; the transitively reached bodies (Table::relocate_entry, TableLocator.retarget_held, spill_locate_held, Region.publish_live, alloc.vec.Vec index), the plan builder, the reservation adapter, the pending merge and the canonical-corpus correspondence are outside this root'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
