#!/usr/bin/env python3
"""Extract and check the real fixed-point RMSNorm source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
Temporary manifests/wrappers call the production #[path] bodies; no substitute algorithm.
Scratch and control copies live outside the tmpfs; receipts and logs under /tmp/opencode.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
if __name__ == '__main__' and os.environ.get('RMS_NORM_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','RMS_NORM_IN_LANE=1',
        'RMS_NORM_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
OUT = Path('/tmp/opencode/rmsnorm-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_BASE = Path(tempfile.gettempdir())
SCRATCH_BASE.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='rmsnorm-source-', dir=SCRATCH_BASE))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('RMS_NORM_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('RMS_NORM_BUDGET_EXHAUSTED')
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
    """Create only typed forwarding mirrors and extract the real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="rms_norm_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(
        '#[path = '+json.dumps(str(TOOLS/'src/engine/num/value.rs'))+']\npub mod value;\n'
        '#[path = '+json.dumps(str(TOOLS/'src/numeric/fixed_rsqrt.rs'))+']\npub mod fixed_rsqrt;\n'
        'pub mod numeric { pub use super::fixed_rsqrt; }\n'
        'pub const FIXED_FRAC_BITS: u32 = 32;\n'
        '#[path = '+json.dumps(str(source))+']\npub mod norm;\n'
        'pub fn word_rms_norm_fixed(row: &[i64], eps_fixed: i64)'
        ' -> Result<Vec<i64>, norm::NormError> { norm::rms_norm_fixed(row, eps_fixed) }\n')
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'rms_norm_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_rms_norm_fixed','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'RmsNormSource.lean'
    generated.write_text(generated.read_text().replace(str(source), 'src/engine/ops/norm.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('RMS_NORM_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage rmsNormProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib RmsNorm where\n  roots := #[`RmsNormSource,`RmsNormResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='rmsNormProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/engine/ops/norm.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/RmsNormSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('RMS_NORM_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('RMS_NORM_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/RmsNormResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('RMS_NORM_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'RmsNormSource.lean');shutil.copyfile(proof,p/'RmsNormResource.lean')
    code,text=run(['lake','build','RmsNormResource'],p,'fresh-proof')
    if 'RMS_NORM_KERNEL_OK' not in text:raise RuntimeError('RMS_NORM_AUDIT_MISSING')
    if a.controls:
        original=source.read_text();needle='i += 1;'
        if original.count(needle)!=2:raise RuntimeError('RMS_NORM_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'norm.rs';mutant.write_text(original.replace(needle,'i += 2;'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'RmsNormSource.lean')
        code,text=run(['lake','build','RmsNormResource'],p,'mutant-proof',require=False)
        if code!=1 or 'RmsNormResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('RMS_NORM_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('RMS_NORM_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths, and Charon's
        # declaration order depends on the crate path, so compare the stripped line multiset.
        strip=lambda s: sorted(re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S).split('\n'))
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('RMS_NORM_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'RmsNormSource.lean');run(['lake','build','RmsNormResource'],p,'restored-proof')
        print('RMS_NORM_REAL_SOURCE_CONTROL_RESTORED')
    print('RMS_NORM_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/ops/norm.rs',ROOT/'src/engine/num/value.rs',ROOT/'src/numeric/fixed_rsqrt.rs',ROOT/'proof/resource/RmsNorm_gate.py',ROOT/'proof/resource/RmsNormSource.lean',ROOT/'proof/resource/RmsNormResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane='/tmp/opencode/apeiron-lane.lock',memory_max='12G',swap_max=0,threads=1,deadline=os.environ.get('RMS_NORM_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar source/resource claim excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH)