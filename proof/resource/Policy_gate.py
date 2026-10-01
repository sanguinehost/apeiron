#!/usr/bin/env python3
"""Extract and check the real bounded policy decision core in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/policy/mod.rs`, which pulls the real
`src/policy/decide.rs` by `pub mod decide;`; the only new code is the typed forwarding wrapper
that calls the real `decide` body. No substitute algorithm is extracted.

  proof/resource/Policy_gate.py --prepare   regenerate proof/resource/PolicySource.lean
  proof/resource/Policy_gate.py             check extraction freshness, then build the proof
  proof/resource/Policy_gate.py --controls  also run the real-source mutation/restore control

The lane is /tmp/opencode/apeiron-lane-3.lock, distinct from the default lane lock, so this gate
does not serialize against the other resource gates. Budget: 540 s wall clock for the whole
invocation, serialized on that lane, LEAN_NUM_THREADS=1, MemoryMax=12G, MemorySwapMax=0.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-3.lock'
if __name__ == '__main__' and os.environ.get('POLICY_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','POLICY_IN_LANE=1',
        'POLICY_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts
# and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/policy-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/policy-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('POLICY_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('POLICY_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles. `src/policy/mod.rs` declares `pub mod decide;`,
# so a `#[path]` on the module root pulls the real `src/policy/decide.rs`; that body imports only
# the standard library, so no crate mirror is needed. The mutant control copies both real files
# into a scratch directory and mutates the copied `decide.rs`, never the working tree.
LIB_TEMPLATE = '''#[path = {mod_rs}]
pub mod policy;

/// Typed forwarding wrapper: it fixes the slice and scalar argument types and calls the real
/// `src/policy/decide.rs` `decide` body. It restates no formula.
pub fn word_policy_decide(
    rules: &[policy::Rule],
    atoms: &[policy::Atom],
    facts: &[u64],
    known: u64,
    action: u32,
    want: u64,
) -> policy::Verdict {{
    policy::decide(rules, atoms, facts, known, action, want)
}}
'''

def lib_source(mod_rs):
    """Absolute `#[path]` to the real `mod.rs`; only the module root varies with the control copy."""
    return LIB_TEMPLATE.format(mod_rs=json.dumps(str(mod_rs)))

def extract(source_dir, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    decide_rs = source_dir/'decide.rs'
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="policy_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source_dir/'mod.rs'))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'policy_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_policy_decide','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'PolicySource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below.
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(decide_rs), 'src/policy/decide.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('POLICY_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage policyProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Policy where\n  roots := #[`PolicySource,`PolicyResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='policyProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source_dir=ROOT/'src/policy'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source_dir,'fresh')
    dest=ROOT/'proof/resource/PolicySource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('POLICY_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('POLICY_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/PolicyResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('POLICY_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'PolicySource.lean');shutil.copyfile(proof,p/'PolicyResource.lean')
    code,text=run(['lake','build','PolicyResource'],p,'fresh-proof')
    if 'POLICY_KERNEL_OK' not in text:raise RuntimeError('POLICY_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: the guard loop's atom index advances by one per continuation.
        # Changing it to two makes the loop skip atoms, which is exactly what the work-measure
        # theorem (`guard_loop_work`, exact count on the T path) forbids. The copy is in scratch;
        # the working tree is never edited.
        original=(source_dir/'decide.rs').read_text()
        needle='''            Tri::T => {}
        }
        i += 1;'''
        if original.count(needle)!=1:raise RuntimeError('POLICY_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'mutant-policy';mutant.mkdir()
        shutil.copyfile(source_dir/'mod.rs',mutant/'mod.rs')
        (mutant/'decide.rs').write_text(original.replace(needle,needle.replace('i += 1;','i += 2;')))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'PolicySource.lean')
        code,text=run(['lake','build','PolicyResource'],p,'mutant-proof',require=False)
        if code!=1 or 'PolicyResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('POLICY_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('POLICY_REAL_SOURCE_CONTROL_RED')
        (mutant/'decide.rs').write_text(original)
        restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('POLICY_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'PolicySource.lean');run(['lake','build','PolicyResource'],p,'restored-proof')
        print('POLICY_REAL_SOURCE_CONTROL_RESTORED')
    print('POLICY_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/policy/mod.rs',ROOT/'src/policy/decide.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/Policy_gate.py',ROOT/'proof/resource/PolicySource.lean',ROOT/'proof/resource/PolicyResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('POLICY_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the scalar work measure is counted in rule visits and atom evaluations under named slice premises, not in cycles or GPU time'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)