#!/usr/bin/env python3
"""Extract and check the real half-word split source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/engine/kern/pow2.rs` by `#[path]`; the only new
code is the typed forwarding wrapper that calls the real `f16_split` body. No substitute
algorithm is extracted. Work is counted in branch tests, scalar additions and sign negations
under the named premise "the extracted body is straight-line: no loop, no recursive call";
it is never a cycle or GPU-time count.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
if __name__ == '__main__' and os.environ.get('F16_SPLIT_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane-2.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','F16_SPLIT_IN_LANE=1',
        'F16_SPLIT_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named
# receipts and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/f16split-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'f16split-resource'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('F16_SPLIT_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('F16_SPLIT_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles: `pow2.rs` is self-contained (its tests use
# only `super::*`). The wrapper fixes the argument and result types and calls the real
# `src/engine/kern/pow2.rs` `f16_split` body; it restates no formula.
LIB_TEMPLATE = '''#[path = {pow2}]
pub mod pow2;

/// Typed forwarding wrapper: it fixes the argument and result types and calls the real
/// `src/engine/kern/pow2.rs` `f16_split` body. It restates no formula.
pub fn split_forward(h: u16) -> Option<(i32, u32)> {{
    pow2::f16_split(h)
}}
'''

def extract(pow2, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="f16_split_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(LIB_TEMPLATE.format(pow2=json.dumps(str(pow2))))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'f16_split_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::split_forward','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
          '-no-progress-bar',str(llbc)],project,label+'-aeneas')
    generated=project/'F16SplitSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(pow2), 'src/engine/kern/pow2.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('F16_SPLIT_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage f16SplitProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib F16Split where\n  roots := #[`F16SplitSource,`F16SplitResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='f16SplitProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/engine/kern/pow2.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/F16SplitSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('F16_SPLIT_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('F16_SPLIT_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/F16SplitResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('F16_SPLIT_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'F16SplitSource.lean');shutil.copyfile(proof,p/'F16SplitResource.lean')
    code,text=run(['lake','build','F16SplitResource'],p,'fresh-proof')
    if 'F16_SPLIT_KERNEL_OK' not in text:raise RuntimeError('F16_SPLIT_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: move the non-finite guard `e == 31` of the refusal branch to
        # `e == 30`. A non-finite pattern then takes the finite path, so the refusal
        # transition, the refusal witness and the branch-count theorem must all go red.
        original=source.read_text();needle='if e == 31 {'
        if original.count(needle)!=1:raise RuntimeError('F16_SPLIT_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'pow2.rs';mutant.write_text(original.replace(needle,'if e == 30 {'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'F16SplitSource.lean')
        code,text=run(['lake','build','F16SplitResource'],p,'mutant-proof',require=False)
        if code!=1 or 'F16SplitResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('F16_SPLIT_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('F16_SPLIT_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('F16_SPLIT_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'F16SplitSource.lean');run(['lake','build','F16SplitResource'],p,'restored-proof')
        print('F16_SPLIT_REAL_SOURCE_CONTROL_RESTORED')
    print('F16_SPLIT_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/kern/pow2.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/F16Split_gate.py',ROOT/'proof/resource/F16SplitSource.lean',ROOT/'proof/resource/F16SplitResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane='/tmp/opencode/apeiron-lane-2.lock',memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('F16_SPLIT_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; work is counted in branch tests, additions and negations of the extracted body, never in cycles or GPU time'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)