#!/usr/bin/env python3
"""Extract and check the real identity-word projection in a bounded private build.

The extraction root is a narrow `#[path]` wrapper over the production `src/coord.rs`, its real
`scalar`/`cell`/`table` closure and two forwarding functions; the shared `proof/extract/lib.rs`
unit is never used. O(B) source/tool/log bytes; external compilation is timed, not assigned
constant cost. Temporary manifests/wrappers call the production `#[path]` body; no substitute
algorithm.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
LANE = '/tmp/opencode/apeiron-lane-2.lock'
if __name__ == '__main__' and os.environ.get('COORD_ID_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','525',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','520s','env','LEAN_NUM_THREADS=1','COORD_ID_IN_LANE=1',
        'COORD_ID_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Disposable copies live off the tmpfs (see AGENTS.md failure register); only logs and the
# receipt are named under /tmp/opencode.
OUT = Path('/tmp/opencode/coordid-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'coordid-resource'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='coordid-', dir=SCRATCH_ROOT))
# The real `src/coord.rs` closure: the identity word lives in `Addr`, declared in `src/coord.rs`,
# and its production constructor `spread` is in the same file. `scalar`, `cell` and `table` are the
# modules `src/coord.rs` resolves `crate::` paths through.
CLOSURE = ['src/scalar.rs', 'src/cell.rs', 'src/table.rs', 'src/coord.rs']
RELATIVE = ['src/scalar.rs', 'src/cell.rs', 'src/table.rs', 'src/coord.rs']
WRAPPER = """
// Forwarding roots for the extraction. Each module is the production `#[path]` source; the two
// functions below name the real bodies and add no arithmetic of their own.
{mods}
pub use cell::Sigma;

/// [`apeiron_word::word_coord_id`]: the identity word of a `coord::Addr`, the real field
/// projection that selects the `id` half [`coord::spread`] produced.
pub fn word_coord_id(address: coord::Addr) -> u64 {{ address.id }}

/// [`coord::spread`], the one production constructor of `coord::Addr`. Called so the resource
/// witness is reached from a real constructor call and not from a model literal.
pub fn word_coord_spread(x: u64) -> coord::Addr {{ coord::spread(x) }}
"""

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(520 - (time.monotonic()-START), float(os.environ.get('COORD_ID_DEADLINE', time.time()+520))-time.time()-3)
    if left <= 0:
        raise RuntimeError('COORD_ID_BUDGET_EXHAUSTED')
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

def extract(root, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="coord_id_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[features]\ndefault = []\ninjection = []\nread-stats = []\n[target.\'cfg(target_os = "linux")\'.dependencies]\nlibc = "=0.2.189"\n[workspace]\n')
    mods = ''.join(f'#[path = {json.dumps(str(root/r))}]\npub mod {Path(r).stem};\n' for r in RELATIVE)
    (project/'lib.rs').write_text(WRAPPER.format(mods=mods))
    run(['cargo','generate-lockfile','--offline'], project, label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'coord_id_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_coord_id','--start-from','crate::word_coord_spread','--',
         '--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'CoordIdSource.lean'
    # Disposable absolute source roots become the repository-relative path they have in the crate.
    generated.write_text(generated.read_text().replace(str(root)+os.sep, ''))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('COORD_ID_UNMODELED_SOURCE')
    return generated

def mutant_source():
    """A throwaway copy of the real `src/` tree; the control edits the copy, never the work tree."""
    target = SCRATCH/'mutant-src'
    target.mkdir()
    shutil.copytree(ROOT/'src', target/'src', symlinks=True)
    return target

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage coordIdProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib CoordId where\n  roots := #[`CoordIdSource,`CoordIdResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='coordIdProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/coord.rs'
    for r in RELATIVE:
        if not (ROOT/r).is_file(): raise RuntimeError('COORD_ID_CLOSURE_MISSING '+r)
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(ROOT,'fresh')
    dest=ROOT/'proof/resource/CoordIdSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('COORD_ID_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('COORD_ID_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/CoordIdResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('COORD_ID_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'CoordIdSource.lean');shutil.copyfile(proof,p/'CoordIdResource.lean')
    code,text=run(['lake','build','CoordIdResource'],p,'fresh-proof')
    if 'COORD_ID_KERNEL_OK' not in text:raise RuntimeError('COORD_ID_AUDIT_MISSING')
    if a.controls:
        mutant=mutant_source();original=(mutant/'src/coord.rs').read_text()
        needle='id: lo,'
        if original.count(needle)!=1:raise RuntimeError('COORD_ID_CONTROL_TARGET_MISSING')
        # The real defect this control injects: wire the identity word to the high half of the
        # widening product, so `word_coord_id` returns the placement-side read instead of `x * PHI`.
        (mutant/'src/coord.rs').write_text(original.replace(needle,'id: hi,'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'CoordIdSource.lean')
        code,text=run(['lake','build','CoordIdResource'],p,'mutant-proof',require=False)
        if code!=1 or 'CoordIdResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','forbidden axioms','Tactic']):
            raise RuntimeError('COORD_ID_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('COORD_ID_REAL_SOURCE_CONTROL_RED')
        (mutant/'src/coord.rs').write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('COORD_ID_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'CoordIdSource.lean');run(['lake','build','CoordIdResource'],p,'restored-proof')
        print('COORD_ID_REAL_SOURCE_CONTROL_RESTORED')
    print('COORD_ID_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/coord.rs',ROOT/'proof/resource/CoordId_gate.py',ROOT/'proof/resource/CoordIdSource.lean',ROOT/'proof/resource/CoordIdResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']+[ROOT/r for r in RELATIVE]
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,deadline=os.environ.get('COORD_ID_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, rustc/cargo, the lean build of the aeneas model, and IO/process/dependencies; the scalar resource claim covers no hardware timing, no GPU/SPIR-V and no canonical ring correspondence'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
