#!/usr/bin/env python3
"""Extract and check the real fixed-point SiLU source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/numeric/fixed_exp.rs`, `src/engine/value.rs` and
`src/engine/silu.rs` modules by `#[path]`; the only new code is the typed forwarding wrapper
`word_silu_fixed` that calls the real `silu_fixed` body. No substitute algorithm is extracted.

Usage:
  Silu_gate.py --prepare     regenerate and write proof/resource/SiluSource.lean, then stop
  Silu_gate.py --controls    fresh build, real-source mutation (red), restoration (green)
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-1.lock'
if __name__ == '__main__' and os.environ.get('SILU_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','SILU_IN_LANE=1',
        'SILU_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts
# and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/silu-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/silu-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('SILU_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('SILU_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles: `src/engine/silu.rs` imports
# `crate::engine::value::div_round_nearest`, `crate::engine::ONE` and
# `crate::numeric::fixed_exp::fixed_exp`. Modules are declared at the crate root (a `#[path]`
# inside an inline module resolves against a virtual directory) with `numeric`/`engine` mirrors,
# exactly the wiring `proof/extract/lib.rs` uses.
LIB_TEMPLATE = '''pub mod numeric {{
    pub use super::{{fixed_exp}};
}}
pub mod engine {{
    pub use super::{{silu, value}};

    /// Fractional bits of the executing fixed-point representation (Q32.32).
    pub const FIXED_FRAC_BITS: u32 = 32;
    /// The Q32.32 unit: `ONE` denotes `1.0`.
    pub const ONE: i64 = 1 << FIXED_FRAC_BITS;
}}
#[path = {fixed_exp}]
pub mod fixed_exp;
#[path = {value}]
pub mod value;
#[path = {silu}]
pub mod silu;

/// Typed forwarding wrapper, the extraction root of `proof/regen.sh` (`--start-from
/// crate::word_silu_fixed`): it fixes the argument and result types and calls the real
/// `src/engine/silu.rs` `silu_fixed` body. It restates no formula.
pub fn word_silu_fixed(v: i64) -> Result<i64, engine::silu::SiluError> {{
    engine::silu::silu_fixed(v)
}}
'''

def lib_source(silu):
    """Absolute `#[path]`s to the real modules; only `silu` varies with the control copy."""
    src = ROOT/'src'
    def p(rel):
        return json.dumps(str(src/rel))
    return LIB_TEMPLATE.format(fixed_exp=p('numeric/fixed_exp.rs'), value=p('engine/value.rs'),
                               silu=json.dumps(str(silu)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="silu_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'silu_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_silu_fixed','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'SiluSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    text = generated.read_text().replace(str(source), 'src/engine/silu.rs')
    # The remaining real modules are named by their repository-relative path, so the committed
    # extraction does not carry this checkout's absolute prefix.
    for rel in ('src/numeric/fixed_exp.rs','src/engine/value.rs'):
        text = text.replace(str(ROOT/rel), rel)
    generated.write_text(text)
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('SILU_UNMODELED_SOURCE')
    # The work measure of `SiluResource.lean` is the closed-form operation count of the guard chain,
    # which holds only while the extracted closure has no loop. Check the artifact for that, so the
    # zero-iteration claim is a rejection and not a comment.
    loops = [m for m in re.finditer(r'#\[rust_loop', generated.read_text())]
    if loops:
        raise RuntimeError('SILU_LOOP_PRESENT %d' % len(loops))
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage siluProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Silu where\n  roots := #[`SiluSource,`SiluResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='siluProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/engine/silu.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/SiluSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('SILU_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('SILU_EXTRACTION_DRIFT')
    print('SILU_LOOP_FREE 0 iterations')
    proof=ROOT/'proof/resource/SiluResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('SILU_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'SiluSource.lean');shutil.copyfile(proof,p/'SiluResource.lean')
    code,text=run(['lake','build','SiluResource'],p,'fresh-proof')
    if 'SILU_KERNEL_OK' not in text:raise RuntimeError('SILU_AUDIT_MISSING')
    if a.controls:
        original=source.read_text();needle='if den <= 0 {'
        if original.count(needle)!=1:raise RuntimeError('SILU_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'silu.rs';mutant.write_text(original.replace(needle,'if den < 0 {'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'SiluSource.lean')
        code,text=run(['lake','build','SiluResource'],p,'mutant-proof',require=False)
        # A diagnostic in this file, not the axiom audit, must fail: the audit line names the same
        # file but is not the proof failure the control is about.
        markers=('unsolved goals','type mismatch','tactic')
        red=[l for l in text.splitlines() if 'SiluResource.lean:' in l
             and any(m in l.lower() for m in markers) and 'forbidden axioms' not in l]
        if code!=1 or not red:
            raise RuntimeError('SILU_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('SILU_REAL_SOURCE_CONTROL_RED ' + red[0].strip())
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Charon/Aeneas emits the module graph in an order that depends on the module's absolute
        # path, so the control copy's extraction is the committed one permuted; the invariant is the
        # multiset of executable lines, not their order. Comment text carries the path and the line
        # numbers and is dropped first.
        strip=lambda s: sorted(re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S).split('\n'))
        if strip(restored.read_text())!=strip(dest.read_text()):
            raise RuntimeError('SILU_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'SiluSource.lean');run(['lake','build','SiluResource'],p,'restored-proof')
        print('SILU_REAL_SOURCE_CONTROL_RESTORED')
    print('SILU_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/silu.rs',ROOT/'src/engine/value.rs',ROOT/'src/numeric/fixed_exp.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/Silu_gate.py',ROOT/'proof/resource/SiluSource.lean',ROOT/'proof/resource/SiluResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('SILU_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the scalar guard-chain operation count excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
