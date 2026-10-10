#!/usr/bin/env python3
"""Extract and check the real `Table::relocate_all` traversal in a bounded private build.

The shared `proof/extract/lib.rs` unit does not compile on its own, so this gate writes a narrow
typed forwarding wrapper that pulls the real `src/{scalar,cell,coord,table}.rs` by `#[path]` (and
with them the real `src/coord/` and `src/table/` children) and calls the real
`Table::relocate_all` body through `crate::word_stored_relocate_all`, the same root
`proof/extract/lib.rs:566-576` declares and `proof/regen.sh:171` names. No substitute algorithm is
extracted; the proof file is checked against a fresh extraction of that wrapper.

  proof/resource/StoredRelocateAll_gate.py --prepare    regenerate proof/resource/StoredRelocateAllSource.lean
  proof/resource/StoredRelocateAll_gate.py              check extraction freshness, then build the proof
  proof/resource/StoredRelocateAll_gate.py --controls   also run the real-source mutation/restore control

The lane is /tmp/opencode/apeiron-lane-1.lock. Budget: 540 s wall clock for the whole invocation,
serialized on that lane, LEAN_NUM_THREADS=1, MemoryMax=12G, MemorySwapMax=0.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
LANE = '/tmp/opencode/apeiron-lane-1.lock'
if __name__ == '__main__' and os.environ.get('STORED_RELOCATE_ALL_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','STORED_RELOCATE_ALL_IN_LANE=1',
        'STORED_RELOCATE_ALL_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts
# and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/storedrelocateall-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'storedrelocateall-resource'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

# The real source closure the wrapper compiles. `src/table.rs` declares `#[path = "table/..."]`
# children and `include!`s `table/{state,locator,prepared}.rs`, and it imports `crate::cell::Sigma`
# and `crate::coord::{address,locator}`, so the wrapper also wires the real `src/{scalar,cell,coord}.rs`
# and the real `src/coord/` children. `src/table/tests.rs` is `#[cfg(test)]` and is compiled by the
# `cargo test` step only. The mutant control copies this closure into a scratch directory and mutates
# the copied `table.rs`, never the working tree.
CLOSURE = ['scalar.rs', 'cell.rs', 'coord.rs', 'table.rs', 'coord', 'table']
COPY_FILES = ['scalar.rs', 'cell.rs', 'coord.rs', 'table.rs']
COPY_DIRS = ['coord', 'table']
# The mutation target: the cursor advance of the relocation traversal, `entry_at += 1;`.
MUTATION = ('            entry_at += 1;', '            entry_at += 2;')

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START),
               float(os.environ.get('STORED_RELOCATE_ALL_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('STORED_RELOCATE_ALL_BUDGET_EXHAUSTED')
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

/// Crate-root alias of the face token the real `src/lib.rs` re-exports as `pub use cell::Sigma`;
/// the real `#[cfg(test)]` modules in `src/table/{{partition,probe}}.rs` import `crate::Sigma`.
pub use cell::Sigma;

/// The carried-key word instance the real `src/lib.rs` declares.
pub type WordStoredTable = table::Table<i64, i64, table::WordHome, 16, table::keyrep::Stored<i64>>;

/// Typed forwarding wrapper: it fixes the table, plan and reserved-storage types and calls the real
/// `Table::relocate_all` body, exactly as `proof/extract/lib.rs` `word_stored_relocate_all` does. It
/// restates no traversal and no per-entry write.
pub fn word_stored_relocate_all(
    mut state: WordStoredTable,
    plan: &table::RebuildPlan,
    mut reserved: table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
) -> (
    WordStoredTable,
    table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
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
    for name in COPY_FILES:
        shutil.copyfile(ROOT/'src'/name, dest/name)
    for name in COPY_DIRS:
        shutil.copytree(ROOT/'src'/name, dest/name)
    return dest

def extract(source_root, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="stored_relocate_all_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[features]\ndefault = []\ninjection = []\nread-stats = []\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source_root))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],
        project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'StoredRelocateAllSource.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_stored_relocate_all','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'StoredRelocateAllSource.lean'
    if not generated.is_file():
        raise RuntimeError('STORED_RELOCATE_ALL_GENERATION_MISSING')
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below.
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source_root), 'src'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('STORED_RELOCATE_ALL_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage storedRelocateAllProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib StoredRelocateAll where\n  roots := #[`LoopTrace,`StoredRelocateAllSource,`StoredRelocateAllResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='storedRelocateAllProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    # The finite visit-count certificate is the repository-owned model, copied unchanged.
    shutil.copyfile(ROOT/'proof/native/LoopTrace.lean', p/'LoopTrace.lean')
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/StoredRelocateAllSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('STORED_RELOCATE_ALL_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('STORED_RELOCATE_ALL_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/StoredRelocateAllResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('STORED_RELOCATE_ALL_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'StoredRelocateAllSource.lean');shutil.copyfile(proof,p/'StoredRelocateAllResource.lean')
    code,text=run(['lake','build','StoredRelocateAllResource'],p,'fresh-proof')
    if 'STORED_RELOCATE_ALL_KERNEL_OK' not in text:raise RuntimeError('STORED_RELOCATE_ALL_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: the traversal's cursor advance, `entry_at += 1;` at
        # src/table.rs:1025, becomes `entry_at += 2;`. Every relocation step then advances by two,
        # which the closed-form work count (`relocate_all_total_work`, `root_total_work`) and the
        # reachable success witness read off the extracted body must see. A count that does not see
        # it has no weight. The copy is in scratch; the working tree is never edited.
        original=(source/'table.rs').read_text()
        if original.count(MUTATION[0])!=1:raise RuntimeError('STORED_RELOCATE_ALL_CONTROL_TARGET_MISSING')
        mutant=copy_closure(SCRATCH/'mutant-src')
        (mutant/'table.rs').write_text(original.replace(*MUTATION))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'StoredRelocateAllSource.lean')
        code,text=run(['lake','build','StoredRelocateAllResource'],p,'mutant-proof',require=False)
        if code!=1 or 'StoredRelocateAllResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic','Not a definitional equality']):
            raise RuntimeError('STORED_RELOCATE_ALL_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('STORED_RELOCATE_ALL_REAL_SOURCE_CONTROL_RED')
        (mutant/'table.rs').write_text(original)
        restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('STORED_RELOCATE_ALL_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'StoredRelocateAllSource.lean')
        code,text=run(['lake','build','StoredRelocateAllResource'],p,'restored-proof')
        if 'STORED_RELOCATE_ALL_KERNEL_OK' not in text:raise RuntimeError('STORED_RELOCATE_ALL_RESTORE_AUDIT_MISSING')
        print('STORED_RELOCATE_ALL_REAL_SOURCE_CONTROL_RESTORED')
    print('STORED_RELOCATE_ALL_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/table.rs',ROOT/'src/cell.rs',ROOT/'src/scalar.rs',ROOT/'src/coord.rs',ROOT/'proof/native/LoopTrace.lean',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/StoredRelocateAll_gate.py',ROOT/'proof/resource/StoredRelocateAllSource.lean',ROOT/'proof/resource/StoredRelocateAllResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('STORED_RELOCATE_ALL_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts planned entries moved and machine-word payload copies in the extracted body under the pinned Aeneas model, not cycles, allocation bytes or GPU time; the reservation adapter, the checked-plan premise and the commit step are outside this root'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
