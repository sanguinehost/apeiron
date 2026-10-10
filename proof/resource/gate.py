#!/usr/bin/env python3
"""Extract and check the real activation-scale source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
Temporary manifests/wrappers call the production #[path] body; no substitute algorithm.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
if __name__ == '__main__' and os.environ.get('ACT_SCALE_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','ACT_SCALE_IN_LANE=1',
        'ACT_SCALE_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
OUT = Path('/tmp/opencode/actscale-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=OUT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('ACT_SCALE_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('ACT_SCALE_BUDGET_EXHAUSTED')
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

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="act_scale_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text('#[path = '+json.dumps(str(source))+']\npub mod pow2;\npub fn scale(a:u64)->Option<u64>{pow2::act_scale(a)}\n')
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'act_scale_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::scale','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'ActScaleSource.lean'
    generated.write_text(generated.read_text().replace(str(source), 'src/engine/kern/pow2.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('ACT_SCALE_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage actScaleProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib ActScale where\n  roots := #[`ActScaleSource,`ActScaleResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='actScaleProof'
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
    dest=ROOT/'proof/resource/ActScaleSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('ACT_SCALE_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('ACT_SCALE_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/ActScaleResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('ACT_SCALE_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'ActScaleSource.lean');shutil.copyfile(proof,p/'ActScaleResource.lean')
    code,text=run(['lake','build','ActScaleResource'],p,'fresh-proof')
    if 'ACT_SCALE_KERNEL_OK' not in text:raise RuntimeError('ACT_SCALE_AUDIT_MISSING')
    if a.controls:
        original=source.read_text();needle='s *= 2;'
        if original.count(needle)!=1:raise RuntimeError('ACT_SCALE_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'pow2.rs';mutant.write_text(original.replace(needle,'s *= 3;'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'ActScaleSource.lean')
        code,text=run(['lake','build','ActScaleResource'],p,'mutant-proof',require=False)
        if code!=1 or 'ActScaleResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('ACT_SCALE_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('ACT_SCALE_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('ACT_SCALE_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'ActScaleSource.lean');run(['lake','build','ActScaleResource'],p,'restored-proof')
        print('ACT_SCALE_REAL_SOURCE_CONTROL_RESTORED')
    print('ACT_SCALE_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/kern/pow2.rs',ROOT/'proof/resource/gate.py',ROOT/'proof/resource/ActScaleSource.lean',ROOT/'proof/resource/ActScaleResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane='/tmp/opencode/apeiron-lane.lock',memory_max='12G',swap_max=0,threads=1,deadline=os.environ.get('ACT_SCALE_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar source/resource claim excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH)
