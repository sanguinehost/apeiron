#!/usr/bin/env python3
"""Extract and check the real GGUF header-parse source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/format/{quant,gguf}.rs` modules by `#[path]`; the only
new code is the typed forwarding wrapper that calls the real `parse_header` body. No substitute
algorithm is extracted.

Lane 2 (`/tmp/opencode/apeiron-lane-2.lock`), not the default lane, so a concurrent run on the
default lock cannot starve or interleave with this gate.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-2.lock'
if __name__ == '__main__' and os.environ.get('GGUF_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','GGUF_IN_LANE=1',
        'GGUF_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts and
# logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/gguf-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/gguf-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('GGUF_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('GGUF_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles: `gguf.rs` imports `super::quant`, which resolves
# against the crate root, so `quant` and `gguf` are declared at the crate root exactly as
# `proof/extract/lib.rs` declares them for the same module pair.
LIB_TEMPLATE = '''#[path = {quant}]
pub mod quant;
#[path = {gguf}]
pub mod gguf;

/// Typed forwarding wrapper: it fixes the slice type and the file-length argument to the slice
/// length and calls the real `src/format/gguf.rs` `parse_header` body. It restates no formula.
pub fn word_gguf_parse_header(buf: &[u8]) -> Result<gguf::GgufHeader, gguf::GgufError> {{
    gguf::parse_header(buf, buf.len() as u64)
}}
'''

def lib_source(gguf):
    """Absolute `#[path]`s to the real modules; only `gguf` varies with the control copy."""
    src = ROOT/'src'
    def p(rel):
        return json.dumps(str(src/rel))
    return LIB_TEMPLATE.format(quant=p('format/quant.rs'), gguf=json.dumps(str(gguf)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="gguf_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(not label.startswith('mutant')))
    llbc = project/'gguf_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_gguf_parse_header','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'GgufSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source), 'src/format/gguf.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('GGUF_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage ggufProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Gguf where\n  roots := #[`GgufSource,`GgufResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='ggufProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

# Two real-source controls, each a single occurrence in `src/format/gguf.rs`:
#   R1 the length-prefixed-string scan step (the byte count and its decreasing rank), and
#   R2 the per-byte cursor advance of `Cur::u8` (the cursor/length-check arithmetic).
CONTROLS = [
    ('rank', 'out.push(self.b[k]);\n            k += 1;', 'out.push(self.b[k]);\n            k += 2;'),
    ('cursor', '        self.i += 1;\n        Some(v)', '        self.i += 2;\n        Some(v)'),
]

def control_red(text, label):
    """The mutated extraction must turn the proof red with an attributable diagnostic."""
    if 'GgufResource.lean:' not in text or not any(x in text for x in
            ['unsolved goals','type mismatch','Tactic','unsolved metavariables']):
        raise RuntimeError('GGUF_CONTROL_NOT_ATTRIBUTABLE_'+label+' '+text[-1800:])

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/format/gguf.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/GgufSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('GGUF_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('GGUF_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/GgufResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('GGUF_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'GgufSource.lean');shutil.copyfile(proof,p/'GgufResource.lean')
    code,text=run(['lake','build','GgufResource'],p,'fresh-proof')
    if 'GGUF_KERNEL_OK' not in text:raise RuntimeError('GGUF_AUDIT_MISSING')
    if a.controls:
        original=source.read_text()
        mutant=SCRATCH/'gguf.rs'
        for label,needle,replacement in CONTROLS:
            if original.count(needle)!=1:raise RuntimeError('GGUF_CONTROL_TARGET_MISSING_'+label)
            mutant.write_text(original.replace(needle,replacement))
            freshmut=extract(mutant,'mutant-'+label);shutil.copyfile(freshmut,p/'GgufSource.lean')
            code,text=run(['lake','build','GgufResource'],p,'mutant-'+label+'-proof',require=False)
            if code!=1:raise RuntimeError('GGUF_CONTROL_NOT_RED_'+label+' exit '+str(code)+'\n'+text[-1800:])
            control_red(text,label)
            print('GGUF_REAL_SOURCE_CONTROL_RED_'+label)
        mutant.write_text(original)
        # Restoration: the disposable copy is back to the original text, and a fresh extraction of
        # the real source must reproduce the committed artifact byte for byte. (Extracting from the
        # disposable path is not comparable: Aeneas orders the emitted items by the source path.)
        restored=extract(source,'restored')
        if restored.read_bytes()!=dest.read_bytes():raise RuntimeError('GGUF_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'GgufSource.lean');run(['lake','build','GgufResource'],p,'restored-proof')
        print('GGUF_REAL_SOURCE_CONTROL_RESTORED')
    print('GGUF_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/format/gguf.rs',ROOT/'src/format/quant.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/Gguf_gate.py',ROOT/'proof/resource/GgufSource.lean',ROOT/'proof/resource/GgufResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('GGUF_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar byte/iteration work measure excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)