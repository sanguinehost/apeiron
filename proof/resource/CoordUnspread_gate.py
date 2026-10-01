#!/usr/bin/env python3
"""Extract and check the real coordinate `unspread` body in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost. The
temporary manifest wires the real `src/coord.rs` by `#[path]`, which pulls the real
`src/coord/{address,locator}.rs`; those pull `crate::cell`, `crate::coord` and `crate::table`, so
the wrapper also wires the real `src/{scalar,cell,table}.rs` and the real `src/table/` children.
The only new code is the typed forwarding wrapper that calls the real `coord::unspread` body of
`src/coord.rs:385-387`. No substitute algorithm is extracted. The shared `proof/extract/lib.rs`
is not used: it is not required to compile here and this root is proved from its own narrow
closure.

  proof/resource/CoordUnspread_gate.py --prepare   regenerate proof/resource/CoordUnspreadSource.lean
  proof/resource/CoordUnspread_gate.py             check extraction freshness, then build the proof
  proof/resource/CoordUnspread_gate.py --controls  also run the real-source mutation/restore control

The lane is /tmp/opencode/apeiron-lane-2.lock. Budget: 540 s wall clock for the whole invocation,
serialized on that lane, LEAN_NUM_THREADS=1, MemoryMax=12G, MemorySwapMax=0.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-2.lock'
if __name__ == '__main__' and os.environ.get('COORD_UNSPREAD_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','COORD_UNSPREAD_IN_LANE=1',
        'COORD_UNSPREAD_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts
# and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/coord-unspread-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/coord-unspread-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

# The real source closure the wrapper compiles. `src/coord.rs` declares `#[path = "coord/..."]`
# children; `src/coord/address.rs` imports `crate::cell::Sigma` and `crate::coord::{spread,Packed}`,
# `src/coord/locator.rs` imports `crate::cell::Sigma`, `crate::coord::address` and
# `crate::table::{keyrep,TableAddress,WordHome}`, and `src/table.rs` imports `crate::cell` and
# `crate::coord`. The mutant control copies this closure into a scratch directory and mutates the
# copied `coord.rs`, never the working tree.
COPY_FILES = ['scalar.rs', 'cell.rs', 'coord.rs', 'table.rs']
COPY_DIRS = ['coord', 'table']
# The mutation target: the single wrapping multiply of the real body, `src/coord.rs:386`. The
# mutant performs the same wrap multiply twice, so the one-multiply work count
# (`body_is_one_wrapping_mul`, the kernel certificate `work_...` is read from) and the
# `PHI_INV`-constant witness must both go red.
MUTATION = ('id.wrapping_mul(PHI_INV)', 'id.wrapping_mul(PHI_INV).wrapping_mul(2)')

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('COORD_UNSPREAD_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('COORD_UNSPREAD_BUDGET_EXHAUSTED')
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

/// Typed forwarding wrapper: it fixes the `u64` argument type and calls the real
/// `src/coord.rs:385-387` `coord::unspread` body. It restates no multiply; the same body is
/// reached by the shared `proof/extract/lib.rs:504-506` wrapper named `word_coord_unspread`.
pub fn word_coord_unspread(id: u64) -> u64 {{
    coord::unspread(id)
}}
'''

def lib_source(base):
    """Absolute `#[path]` into the real source root; only the root varies with the control copy."""
    return LIB_TEMPLATE.format(**{k: json.dumps(str(base/(k+'.rs'))) for k in
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
    (project/'Cargo.toml').write_text('[package]\nname="coord_unspread_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[features]\ndefault = []\ninjection = []\nread-stats = []\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source_root))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'coord_unspread_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_coord_unspread','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar',str(llbc)],project,label+'-aeneas')
    generated=project/'CoordUnspreadSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below.
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source_root/'coord.rs'), 'src/coord.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('COORD_UNSPREAD_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage coordUnspreadProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib CoordUnspread where\n  roots := #[`CoordUnspreadSource,`CoordUnspreadResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='coordUnspreadProof'
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
    dest=ROOT/'proof/resource/CoordUnspreadSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('COORD_UNSPREAD_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('COORD_UNSPREAD_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/CoordUnspreadResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('COORD_UNSPREAD_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'CoordUnspreadSource.lean');shutil.copyfile(proof,p/'CoordUnspreadResource.lean')
    code,text=run(['lake','build','CoordUnspreadResource'],p,'fresh-proof')
    if 'COORD_UNSPREAD_KERNEL_OK' not in text:raise RuntimeError('COORD_UNSPREAD_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: the real body is one `wrapping_mul` by `PHI_INV`. The mutant wraps
        # once more, so the one-multiply certificate, the counted work and the constant witness
        # must all fail. The copy is in scratch; the working tree is never edited.
        original=(source/'coord.rs').read_text()
        if original.count(MUTATION[0])!=1:raise RuntimeError('COORD_UNSPREAD_CONTROL_TARGET_MISSING')
        mutant=copy_closure(SCRATCH/'mutant-src')
        (mutant/'coord.rs').write_text(original.replace(*MUTATION))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'CoordUnspreadSource.lean')
        code,text=run(['lake','build','CoordUnspreadResource'],p,'mutant-proof',require=False)
        if code!=1 or 'CoordUnspreadResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic','Not a definitional equality']):
            raise RuntimeError('COORD_UNSPREAD_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        lines=sorted({int(m) for m in re.findall(r'CoordUnspreadResource\.lean:(\d+)', text)})
        print('COORD_UNSPREAD_REAL_SOURCE_CONTROL_RED at CoordUnspreadResource.lean:'+','.join(map(str,lines[:6])))
        (mutant/'coord.rs').write_text(original)
        restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('COORD_UNSPREAD_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'CoordUnspreadSource.lean');run(['lake','build','CoordUnspreadResource'],p,'restored-proof')
        print('COORD_UNSPREAD_REAL_SOURCE_CONTROL_RESTORED')
    print('COORD_UNSPREAD_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/coord.rs',ROOT/'src/coord/address.rs',ROOT/'src/coord/locator.rs',ROOT/'src/coord/locator_core.rs',ROOT/'src/cell.rs',ROOT/'src/scalar.rs',ROOT/'src/table.rs',ROOT/'src/table/state.rs',ROOT/'src/table/locator.rs',ROOT/'src/table/prepared.rs',ROOT/'src/table/keyrep.rs',ROOT/'src/table/partition.rs',ROOT/'src/table/probe.rs',ROOT/'src/table/resources.rs',ROOT/'src/table/scan.rs',ROOT/'src/table/parallel.rs',ROOT/'src/table/tests.rs',ROOT/'proof/extract/lib.rs',ROOT/'proof/regen.sh',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/CoordUnspread_gate.py',ROOT/'proof/resource/CoordUnspreadSource.lean',ROOT/'proof/resource/CoordUnspreadResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('COORD_UNSPREAD_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts U64 wrapping multiplies, word comparisons and heap allocations in the extracted body under the Aeneas model, not cycles or GPU time; the algebra correspondence unspread(spread x).id = x, the spread body, the table key codec and the shared proof/extract/lib.rs unit are outside this root'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)