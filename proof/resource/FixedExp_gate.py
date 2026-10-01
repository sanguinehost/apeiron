#!/usr/bin/env python3
"""Extract and check the real fixed-point `exp` source in a bounded private build.

Adapted from proof/resource/gate.py (and its row-dot/route-topk siblings) for the extraction root
`crate::word_fixed_exp`, the same root proof/regen.sh names, whose forwarding wrapper is
`proof/extract/lib.rs:686` calling the real `src/numeric/fixed_exp.rs` `fixed_exp` body. O(B)
source/tool/log bytes; external compilation is timed, not assigned constant cost. The temporary
manifest wires the real `src/numeric/fixed_exp.rs` by `#[path]`; the only new code is the typed
forwarding wrapper that calls the real body, so no substitute algorithm is extracted. The resource
measure is a scalar work count (rounds and guard selects) under named premises, never a cycle
count, wall-clock or GPU time.

Lane 3: /tmp/opencode/apeiron-lane-3.lock. Budget 540 s for the whole invocation, enforced on every
descendant through one shared deadline.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-3.lock'
if __name__ == '__main__' and os.environ.get('FIXED_EXP_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','FIXED_EXP_IN_LANE=1',
        'FIXED_EXP_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (AGENTS.md failure register: /tmp tmpfs headroom); only named
# receipts and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/fixedexp-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/fixedexp-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('FIXED_EXP_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('FIXED_EXP_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles: `src/numeric/fixed_exp.rs` has no crate-internal
# imports, so the only module wired by `#[path]` is that file. The wrapper fixes the argument and
# result types and calls the real body; it restates no formula.
LIB_TEMPLATE = '''#[path = {fixed_exp}]
pub mod fixed_exp;

/// Typed forwarding wrapper, the same shape as `proof/extract/lib.rs:686`: it fixes the `i64`
/// argument and the `Result<i64, ExpError>` result and calls the real `fixed_exp` body.
pub fn fixed_exp_forward(x: i64) -> Result<i64, fixed_exp::ExpError> {{
    fixed_exp::fixed_exp(x)
}}
'''

def lib_source(fixed_exp):
    """Absolute `#[path]` to the real module; only that path varies with the control copy."""
    return LIB_TEMPLATE.format(fixed_exp=json.dumps(str(fixed_exp)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="fixed_exp_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'fixed_exp_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::fixed_exp_forward','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'FixedExpSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source), 'src/numeric/fixed_exp.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('FIXED_EXP_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage fixedExpProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib FixedExp where\n  roots := #[`FixedExpSource,`FixedExpResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='fixedExpProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/numeric/fixed_exp.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/FixedExpSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('FIXED_EXP_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('FIXED_EXP_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/FixedExpResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('FIXED_EXP_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'FixedExpSource.lean');shutil.copyfile(proof,p/'FixedExpResource.lean')
    code,text=run(['lake','build','FixedExpResource'],p,'fresh-proof')
    if 'FIXED_EXP_KERNEL_OK' not in text:raise RuntimeError('FIXED_EXP_AUDIT_MISSING')
    if a.controls:
        # Real-source control: delete one Horner round from the degree-7 chain in a throwaway copy.
        # The resource claim is a fixed round count (7 rounds, 21 scalar operations) of the actual
        # extracted body, so removing a round from src/numeric/fixed_exp.rs must falsify it.
        original=source.read_text();needle='    acc = round_div_2p32(acc * r) + 715_827_883; // 1 / 3!\n'
        if original.count(needle)!=1:raise RuntimeError('FIXED_EXP_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'fixed_exp.rs';mutant.write_text(original.replace(needle,''))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'FixedExpSource.lean')
        code,text=run(['lake','build','FixedExpResource'],p,'mutant-proof',require=False)
        if code!=1 or 'FixedExpResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('FIXED_EXP_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('FIXED_EXP_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('FIXED_EXP_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'FixedExpSource.lean');run(['lake','build','FixedExpResource'],p,'restored-proof')
        print('FIXED_EXP_REAL_SOURCE_CONTROL_RESTORED')
    print('FIXED_EXP_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/numeric/fixed_exp.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/FixedExp_gate.py',ROOT/'proof/resource/FixedExpSource.lean',ROOT/'proof/resource/FixedExpResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('FIXED_EXP_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar work count under named premises excludes cycles, wall-clock and GPU time'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
