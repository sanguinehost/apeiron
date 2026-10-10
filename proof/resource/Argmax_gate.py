#!/usr/bin/env python3
"""Extract and check the real fixed-point argmax source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/numeric/fixed_exp.rs`, `src/engine/value.rs` and
`src/engine/sampler.rs` modules by `#[path]`, plus the two `pub const` lines quoted verbatim out of
`src/engine/mod.rs`; the only new code is the typed forwarding wrapper that calls the real
`argmax_fixed` body. No substitute algorithm is extracted.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
LANE = '/tmp/opencode/apeiron-lane-3.lock'
if __name__ == '__main__' and os.environ.get('ARGMAX_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','ARGMAX_IN_LANE=1',
        'ARGMAX_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (docs/AGENTS failure register: /tmp tmpfs headroom); only
# named receipts and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/argmax-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'argmax-resource'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('ARGMAX_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('ARGMAX_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles. `sampler.rs` reads `super::value::div_round_nearest`,
# `super::{FIXED_FRAC_BITS, ONE}` and `crate::numeric::fixed_exp::fixed_exp`, so `sampler`, `value`
# and `fixed_exp` sit at the temporary crate root (`sampler`'s `super` is that root) and the two
# constants are quoted out of the real `src/engine/mod.rs` rather than restated here.
LIB_TEMPLATE = '''pub mod numeric {{
    pub use super::fixed_exp;
}}
#[path = {fixed_exp}]
pub mod fixed_exp;
{engine_consts}
#[path = {value}]
pub mod value;
#[path = {sampler}]
pub mod sampler;

/// Typed forwarding wrapper: it fixes the logit and index types and calls the real
/// `src/engine/sampler.rs` `argmax_fixed` body. It restates no formula.
pub fn word_argmax_fixed(logits: &[i64]) -> Result<u32, sampler::SamplerError> {{
    sampler::argmax_fixed(logits)
}}
'''

# `src/engine/mod.rs` is not `#[path]`-able as a parent module here (it declares the whole engine),
# so the two constants `sampler.rs` imports are read out of it verbatim and placed in the temporary
# crate root that plays the role of `super`. A change to either real declaration changes this text.
CONST_RE = re.compile(r'^pub const (FIXED_FRAC_BITS|ONE)\s*:[^\n]*$', re.M)

def lib_source(sampler):
    """Absolute `#[path]`s to the real modules; only `sampler` varies with the control copy."""
    src = ROOT/'src'
    mod = (src/'engine/mod.rs').read_text()
    consts = CONST_RE.findall(mod)
    if sorted(consts) != ['FIXED_FRAC_BITS', 'ONE']:
        raise RuntimeError('ARGMAX_CONST_SOURCE_MISSING')
    consts = '\n'.join(line for line in mod.split('\n') if CONST_RE.match(line))
    def p(rel):
        return json.dumps(str(src/rel))
    return LIB_TEMPLATE.format(fixed_exp=p('numeric/fixed_exp.rs'), value=p('engine/value.rs'),
        engine_consts=consts, sampler=json.dumps(str(sampler)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="argmax_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'argmax_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_argmax_fixed','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'ArgmaxSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source), 'src/engine/sampler.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('ARGMAX_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage argmaxProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Argmax where\n  roots := #[`ArgmaxSource,`ArgmaxResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='argmaxProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/engine/sampler.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/ArgmaxSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('ARGMAX_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('ARGMAX_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/ArgmaxResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('ARGMAX_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'ArgmaxSource.lean');shutil.copyfile(proof,p/'ArgmaxResource.lean')
    code,text=run(['lake','build','ArgmaxResource'],p,'fresh-proof')
    if 'ARGMAX_KERNEL_OK' not in text:raise RuntimeError('ARGMAX_AUDIT_MISSING')
    if a.controls:
        original=source.read_text()
        needle='        if logits[i] > logits[best] {\n            best = i;\n        }\n        i += 1;'
        if original.count(needle)!=1:raise RuntimeError('ARGMAX_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'sampler.rs'
        mutant.write_text(original.replace(needle,needle[:-len('i += 1;')]+'i += 2;'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'ArgmaxSource.lean')
        code,text=run(['lake','build','ArgmaxResource'],p,'mutant-proof',require=False)
        if code!=1 or 'ArgmaxResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('ARGMAX_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('ARGMAX_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('ARGMAX_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'ArgmaxSource.lean');run(['lake','build','ArgmaxResource'],p,'restored-proof')
        print('ARGMAX_REAL_SOURCE_CONTROL_RESTORED')
    print('ARGMAX_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/mod.rs',ROOT/'src/engine/sampler.rs',ROOT/'src/engine/value.rs',ROOT/'src/numeric/fixed_exp.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/Argmax_gate.py',ROOT/'proof/resource/ArgmaxSource.lean',ROOT/'proof/resource/ArgmaxResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('ARGMAX_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar comparison-count/resource claim excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
