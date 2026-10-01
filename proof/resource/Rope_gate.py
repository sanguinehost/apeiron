#!/usr/bin/env python3
"""Extract and check the real RoPE rotation source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/engine/rope.rs`, `src/engine/value.rs` and
`src/numeric/fixed_trig.rs` modules by `#[path]`; the only new code is the typed forwarding
wrapper that calls the real `rope_rotate_fixed` body. No substitute algorithm is extracted.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
if __name__ == '__main__' and os.environ.get('ROPE_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane-2.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','ROPE_IN_LANE=1',
        'ROPE_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts
# and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/rope-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/rope-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def strip(s):
    """Executable text: drop the doc and block comments that carry source paths and line numbers."""
    return re.sub(r'/\*.*?\*/|/--.*?-/', '', s, flags=re.S)

def canon(s):
    """Order-insensitive comparison key. Aeneas emits declarations in an order that is not stable
    across runs, so the freshness and restoration controls compare the multiset of executable
    lines: any body change is still a difference, a reordering is not."""
    return sorted(strip(s).split('\n'))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('ROPE_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('ROPE_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles: `rope.rs` imports `super::value::round_shift_32`
# and `crate::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians}`. Modules are declared at the
# crate root (a `#[path]` inside an inline module resolves against a virtual directory) with a
# `numeric` mirror, exactly the wiring `proof/extract/lib.rs` uses; nothing is restated.
LIB_TEMPLATE = '''pub mod numeric {{
    pub use super::fixed_trig;
}}
#[path = {value}]
pub mod value;
#[path = {fixed_trig}]
pub mod fixed_trig;
#[path = {rope}]
pub mod rope;

/// Typed forwarding wrapper: it fixes the vector, position and table types and calls the real
/// `src/engine/rope.rs` `rope_rotate_fixed` body. It restates no formula.
pub fn rope_rotate_forward(
    q: &mut [i64],
    k: &mut [i64],
    pos: u32,
    table: &rope::RopeTable,
) -> Result<(), rope::RopeError> {{
    rope::rope_rotate_fixed(q, k, pos, table)
}}
'''

def lib_source(rope):
    """Absolute `#[path]`s to the real modules; only `rope` varies with the control copy."""
    src = ROOT/'src'
    def p(rel):
        return json.dumps(str(src/rel))
    return LIB_TEMPLATE.format(value=p('engine/value.rs'),
        fixed_trig=p('numeric/fixed_trig.rs'), rope=json.dumps(str(rope)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="rope_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'rope_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::rope_rotate_forward','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'RopeSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    # Charon records `Source:` against the working directory; pin the three real files to their
    # repository-relative paths so the committed Lean file is tool- and machine-independent.
    text=generated.read_text()
    for rel in ('src/engine/rope.rs','src/engine/value.rs','src/numeric/fixed_trig.rs'):
        text=text.replace(str(ROOT/rel), rel)
    generated.write_text(text.replace(str(source), 'src/engine/rope.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('ROPE_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage ropeProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Rope where\n  roots := #[`RopeSource,`RopeResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='ropeProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/engine/rope.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/RopeSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('ROPE_EXTRACTION_READY');return
    if canon(generated.read_text())!=canon(dest.read_text()):raise RuntimeError('ROPE_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/RopeResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('ROPE_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'RopeSource.lean');shutil.copyfile(proof,p/'RopeResource.lean')
    code,text=run(['lake','build','RopeResource'],p,'fresh-proof')
    if 'ROPE_KERNEL_OK' not in text:raise RuntimeError('ROPE_AUDIT_MISSING')
    if a.controls:
        original=source.read_text()
        # The pair-loop index increment of `RopeTable::rotate_head` (line 254), disambiguated from
        # the load-time table loop's `j += 1;` (line 84) by the following lines.
        needle='j += 1;\n        }\n        if !ok {'
        if original.count(needle)!=1:raise RuntimeError('ROPE_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'rope.rs';mutant.write_text(original.replace(needle,'j += 2;\n        }\n        if !ok {'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'RopeSource.lean')
        code,text=run(['lake','build','RopeResource'],p,'mutant-proof',require=False)
        if code!=1 or 'RopeResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('ROPE_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('ROPE_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        if canon(restored.read_text())!=canon(dest.read_text()):
            keep=OUT/(SCRATCH.name+'-restore-drift')
            keep.mkdir(parents=True, exist_ok=True)
            (keep/'restored.lean').write_text(strip(restored.read_text()))
            (keep/'committed.lean').write_text(strip(dest.read_text()))
            raise RuntimeError('ROPE_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'RopeSource.lean');run(['lake','build','RopeResource'],p,'restored-proof')
        print('ROPE_REAL_SOURCE_CONTROL_RESTORED')
    print('ROPE_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/rope.rs',ROOT/'src/engine/value.rs',ROOT/'src/numeric/fixed_trig.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/Rope_gate.py',ROOT/'proof/resource/RopeSource.lean',ROOT/'proof/resource/RopeResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane='/tmp/opencode/apeiron-lane-2.lock',memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('ROPE_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts loop iterations and fixed_sin_cos evaluations under named premises, not cycles, wall-clock or GPU time'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
