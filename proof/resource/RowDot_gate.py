#!/usr/bin/env python3
"""Extract and check the real fixed-point row-dot source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/engine/{quant,gguf,value,f32_convert,error,dequant,
matvec}.rs` modules by `#[path]`; the only new code is the typed forwarding wrapper that calls
the real `row_dot_fixed` body. No substitute algorithm is extracted.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
if __name__ == '__main__' and os.environ.get('ROW_DOT_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','ROW_DOT_IN_LANE=1',
        'ROW_DOT_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (docs/AGENTS failure register: /tmp tmpfs headroom); only
# named receipts and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/rowdot-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'rowdot-resource'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('ROW_DOT_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('ROW_DOT_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles: `matvec.rs` imports `crate::format::quant`,
# `super::{dequant,value,EngineError}`, and `error.rs` imports `crate::engine::f32_convert` and
# `crate::format::{gguf,quant}`. Modules are declared at the crate root (a `#[path]` inside an
# inline module resolves against a virtual directory) with `format`/`engine` mirrors, exactly the
# wiring `proof/extract/lib.rs` uses; the engine mirror supplies `super::EngineError`.
LIB_TEMPLATE = '''pub mod format {{
    pub use super::{{gguf, quant}};
}}
pub mod engine {{
    pub use super::{{dequant, error, f32_convert, matvec, value}};
    pub use super::error::EngineError;
    pub use super::matvec::MatvecError;
}}
#[path = {quant}]
pub mod quant;
#[path = {gguf}]
pub mod gguf;
#[path = {value}]
pub mod value;
#[path = {f32_convert}]
pub mod f32_convert;
#[path = {error}]
pub mod error;
#[path = {dequant}]
pub mod dequant;
#[path = {matvec}]
pub mod matvec;
pub use error::EngineError;

/// Typed forwarding wrapper: it fixes the slice and result types and calls the real
/// `src/engine/matvec.rs` `row_dot_fixed` body. It restates no formula.
pub fn row_dot_forward(w_row: &[i64], x: &[i64]) -> Result<i64, engine::matvec::MatvecError> {{
    engine::matvec::row_dot_fixed(w_row, x)
}}
'''

def lib_source(matvec):
    """Absolute `#[path]`s to the real modules; only `matvec` varies with the control copy."""
    src = ROOT/'src'
    def p(rel):
        return json.dumps(str(src/rel))
    return LIB_TEMPLATE.format(quant=p('format/quant.rs'), gguf=p('format/gguf.rs'),
        value=p('engine/value.rs'), f32_convert=p('engine/f32_convert.rs'),
        error=p('engine/error.rs'), dequant=p('engine/dequant.rs'), matvec=json.dumps(str(matvec)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="row_dot_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'row_dot_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::row_dot_forward','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'RowDotSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(ROOT)+'/', '').replace(str(source), 'src/engine/matvec.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('ROW_DOT_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage rowDotProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib RowDot where\n  roots := #[`RowDotSource,`RowDotResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='rowDotProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/engine/matvec.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/RowDotSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('ROW_DOT_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('ROW_DOT_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/RowDotResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('ROW_DOT_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'RowDotSource.lean');shutil.copyfile(proof,p/'RowDotResource.lean')
    code,text=run(['lake','build','RowDotResource'],p,'fresh-proof')
    if 'ROW_DOT_KERNEL_OK' not in text:raise RuntimeError('ROW_DOT_AUDIT_MISSING')
    if a.controls:
        original=source.read_text();needle='i += 1;'
        if original.count(needle)!=1:raise RuntimeError('ROW_DOT_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'matvec.rs';mutant.write_text(original.replace(needle,'i += 2;'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'RowDotSource.lean')
        code,text=run(['lake','build','RowDotResource'],p,'mutant-proof',require=False)
        if code!=1 or 'RowDotResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('ROW_DOT_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('ROW_DOT_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # The `Source:` comments and the order in which Aeneas emits the declarations both follow
        # the disposable absolute source path; compare the declaration bodies as a multiset.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        chunks=lambda s: sorted(c.strip() for c in re.split(r'\n\s*\n', strip(s)) if c.strip())
        if chunks(restored.read_text())!=chunks(dest.read_text()):raise RuntimeError('ROW_DOT_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'RowDotSource.lean');run(['lake','build','RowDotResource'],p,'restored-proof')
        print('ROW_DOT_REAL_SOURCE_CONTROL_RESTORED')
    print('ROW_DOT_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/matvec.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/RowDot_gate.py',ROOT/'proof/resource/RowDotSource.lean',ROOT/'proof/resource/RowDotResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane='/tmp/opencode/apeiron-lane.lock',memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('ROW_DOT_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar visit-count/resource claim excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)