#!/usr/bin/env python3
"""Extract and check the real LPT schedule completeness scan in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost. The
temporary manifest wires the real `src/{scalar,cell,coord,table}.rs` by `#[path]`, which pulls the
real `src/table/partition.rs`; the only new code is the typed forwarding wrapper that calls the
real `Schedule::lpt` then `Schedule::is_complete` bodies, exactly as
`proof/extract/lib.rs::word_schedule_is_complete` does. The shared `proof/extract/lib.rs` unit is
not compiled: this gate never depends on it. No substitute algorithm is extracted.

Work is counted in scan steps (counter increments of the extracted loops) and in `Bool`/`Usize`
comparisons under named premises; it is never a cycle or GPU-time count.

  proof/resource/Schedule_gate.py --prepare   regenerate proof/resource/ScheduleSource.lean
  proof/resource/Schedule_gate.py             check extraction freshness, then build the proof
  proof/resource/Schedule_gate.py --controls  also run the real-source mutation/restore control

The lane is /tmp/opencode/apeiron-lane-2.lock. Budget: 540 s wall clock for the whole invocation,
serialized on that lane, LEAN_NUM_THREADS=1, MemoryMax=12G, MemorySwapMax=0.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-2.lock'
if __name__ == '__main__' and os.environ.get('SCHEDULE_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','SCHEDULE_IN_LANE=1',
        'SCHEDULE_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts
# and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/schedule-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/schedule-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

# The real source closure the wrapper compiles. `src/table.rs` declares `#[path = "table/..."]`
# children and `include!`s `table/{state,locator,prepared}.rs`, and it imports `crate::cell::Sigma`
# and `crate::coord::{address,locator}`, so the wrapper also wires the real `src/{scalar,cell,coord}.rs`
# and the real `src/coord/` children. `src/table/tests.rs` is `#[cfg(test)]` and is compiled by the
# `cargo test` step only. The mutant control copies this closure into a scratch directory and mutates
# the copied `table/partition.rs`, never the working tree.
COPY_FILES = ['scalar.rs', 'cell.rs', 'coord.rs', 'table.rs']
COPY_DIRS = ['coord', 'table']
# The mutation target: the scan step of the real `is_complete` inner loop. Advancing the read
# position by two instead of one halves the covered positions of every bin, so the unit rank
# decrease and the reachability of the success branch both stop holding.
MUTATION_OLD = '                }\n                at += 1;\n            }\n            worker += 1;'
MUTATION_NEW = '                }\n                at += 2;\n            }\n            worker += 1;'

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('SCHEDULE_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('SCHEDULE_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles, wired at the crate root exactly as
# `proof/extract/lib.rs` is wired. The wrapper body below is the only new code.
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

/// Real deterministic LPT schedule completeness check. Typed forwarding wrapper over the real
/// `src/table/partition.rs` bodies of `Schedule::lpt` and `Schedule::is_complete`; it fixes the
/// argument types and restates no formula.
pub fn word_schedule_is_complete(
    loads: &[usize; table::partition::SHARDS],
    workers: usize,
) -> bool {{
    table::partition::Schedule::lpt(loads, workers).is_complete()
}}
'''

def lib_source(source_root):
    """Absolute `#[path]` into the real source root; only the root varies with the control copy."""
    return LIB_TEMPLATE.format(**{k: json.dumps(str(source_root/(k+'.rs'))) for k in
                                  ['scalar', 'cell', 'coord', 'table']})

def copy_closure(dest):
    """Bounded copy of the real extraction closure; the working tree is never edited."""
    dest.mkdir(parents=True)
    for name in COPY_FILES:
        shutil.copyfile(ROOT/'src'/name, dest/name)
    for name in COPY_DIRS:
        shutil.copytree(ROOT/'src'/name, dest/name)
    return dest

def extract(source_root, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="schedule_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[features]\ndefault = []\ninjection = []\nread-stats = []\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source_root))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'schedule_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_schedule_is_complete','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'ScheduleSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below.
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source_root)+'/', 'src/'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('SCHEDULE_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage scheduleProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Schedule where\n  roots := #[`ScheduleSource,`ScheduleResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='scheduleProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/ScheduleSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('SCHEDULE_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('SCHEDULE_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/ScheduleResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('SCHEDULE_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'ScheduleSource.lean');shutil.copyfile(proof,p/'ScheduleResource.lean')
    code,text=run(['lake','build','ScheduleResource'],p,'fresh-proof')
    if 'SCHEDULE_KERNEL_OK' not in text:raise RuntimeError('SCHEDULE_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: the real `is_complete` inner scan advances the read position by one
        # per step, which is what the unit rank decrease and the success-branch reachability read.
        # Stepping by two leaves every odd position unscanned, so the coverage scan cannot succeed.
        # The copy is in scratch; the working tree is never edited.
        original=(source/'table/partition.rs').read_text()
        if original.count(MUTATION_OLD)!=1:raise RuntimeError('SCHEDULE_CONTROL_TARGET_MISSING')
        mutant=copy_closure(SCRATCH/'mutant-src')
        (mutant/'table/partition.rs').write_text(original.replace(MUTATION_OLD,MUTATION_NEW))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'ScheduleSource.lean')
        code,text=run(['lake','build','ScheduleResource'],p,'mutant-proof',require=False)
        if code!=1 or 'ScheduleResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic','Not a definitional equality']):
            raise RuntimeError('SCHEDULE_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('SCHEDULE_REAL_SOURCE_CONTROL_RED')
        (mutant/'table/partition.rs').write_text(original)
        restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('SCHEDULE_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'ScheduleSource.lean');run(['lake','build','ScheduleResource'],p,'restored-proof')
        print('SCHEDULE_REAL_SOURCE_CONTROL_RESTORED')
    print('SCHEDULE_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/table.rs',ROOT/'src/table/partition.rs',ROOT/'src/table/resources.rs',ROOT/'src/table/state.rs',ROOT/'src/table/prepared.rs',ROOT/'src/table/locator.rs',ROOT/'src/table/keyrep.rs',ROOT/'src/table/parallel.rs',ROOT/'src/table/probe.rs',ROOT/'src/table/tests.rs',ROOT/'src/cell.rs',ROOT/'src/scalar.rs',ROOT/'src/coord.rs',ROOT/'src/coord/address.rs',ROOT/'src/coord/locator.rs',ROOT/'src/coord/locator_core.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/Schedule_gate.py',ROOT/'proof/resource/ScheduleSource.lean',ROOT/'proof/resource/ScheduleResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('SCHEDULE_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts scan steps and scalar comparisons of the extracted is_complete and lpt loops under the Aeneas model, not cycles or GPU time; the LPT selection sort, the worker bin fill, the Vec push/index/update adapter and the heap allocation it performs are inside this root'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
