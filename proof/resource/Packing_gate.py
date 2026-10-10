#!/usr/bin/env python3
"""Extract and check the real canonical packing block decoder in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
Temporary manifests/wrappers call the production #[path] body; no substitute algorithm.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
if __name__ == '__main__' and os.environ.get('PACKING_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','PACKING_IN_LANE=1',
        'PACKING_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
OUT = Path(tempfile.gettempdir()) / 'packing-resource'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=OUT))

# The real Rust body: `src/format/packing.rs`. The scratch crate declares the three `src/format`
# modules by `#[path]` plus the `crate::format` mirror the module's own test uses, and forwards to
# the real `Packing::decode_block` through the same wrapper shape as `proof/extract/lib.rs`.
WRAPPER = '''#[path = {quant}]
pub mod quant;
#[path = {gguf}]
pub mod gguf;
#[path = {packing}]
pub mod packing;
pub mod format {{
  pub use super::{{gguf, packing, quant}};
}}
pub fn word_packing_decode_block(p: packing::Packing, raw: &[u8]) -> Result<Vec<i64>, packing::PackingError> {{ p.decode_block(raw) }}
'''

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('PACKING_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('PACKING_BUDGET_EXHAUSTED')
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
    (project/'Cargo.toml').write_text('[package]\nname="packing_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(WRAPPER.format(
        quant=json.dumps(str(TOOLS/'src/format/quant.rs')),
        gguf=json.dumps(str(TOOLS/'src/format/gguf.rs')),
        packing=json.dumps(str(source))))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'packing_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_packing_decode_block','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'PackingSource.lean'
    generated.write_text(generated.read_text().replace(str(source), 'src/format/packing.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('PACKING_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage packingProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Packing where\n  roots := #[`PackingSource,`PackingResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='packingProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/format/packing.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/PackingSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('PACKING_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('PACKING_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/PackingResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('PACKING_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'PackingSource.lean');shutil.copyfile(proof,p/'PackingResource.lean')
    code,text=run(['lake','build','PackingResource'],p,'fresh-proof')
    if 'PACKING_KERNEL_OK' not in text:raise RuntimeError('PACKING_AUDIT_MISSING')
    if a.controls:
        # `unpack_code`'s plane-loop advance: `k += 1` -> `k += 2` makes the extracted body's rank
        # transition non-unit, which the per-bit and per-lane work theorems must reject.
        original=source.read_text();needle='v |= bit << shift;\n            k += 1;'
        if original.count(needle)!=1:raise RuntimeError('PACKING_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'packing.rs';mutant.write_text(
            original.replace(needle,'v |= bit << shift;\n            k += 2;'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'PackingSource.lean')
        code,text=run(['lake','build','PackingResource'],p,'mutant-proof',require=False)
        if code!=1 or 'PackingResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('PACKING_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('PACKING_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Aeneas orders declarations by crate path and prints a disposable relative path for a
        # scratch copy, so the restore control compares the comment-free body as a multiset of
        # lines: same declarations, same executable text, order and source locations excluded.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        signature=lambda s: sorted(l for l in strip(s).split('\n') if l.strip())
        if signature(restored.read_text())!=signature(dest.read_text()):
            raise RuntimeError('PACKING_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'PackingSource.lean');run(['lake','build','PackingResource'],p,'restored-proof')
        print('PACKING_REAL_SOURCE_CONTROL_RESTORED')
    print('PACKING_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/format/packing.rs',ROOT/'src/format/quant.rs',ROOT/'src/format/gguf.rs',ROOT/'proof/resource/Packing_gate.py',ROOT/'proof/resource/PackingSource.lean',ROOT/'proof/resource/PackingResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        receipt=OUT/(SCRATCH.name+'-receipt.json')
        receipt.write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane='/tmp/opencode/apeiron-lane.lock',memory_max='12G',swap_max=0,threads=1,deadline=os.environ.get('PACKING_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar source/resource claim excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.copyfile(receipt, '/tmp/opencode/packing-resource-gate-receipt.json')
        shutil.rmtree(SCRATCH)
