#!/usr/bin/env python3
"""Extract and check the real SPIR-V DP4A fragment checker in a bounded private build.

The extraction unit is its own narrow forwarding wrapper over `src/gpu/spirv/mod.rs` alone: the shared
`proof/extract/lib.rs` does not currently compile, so this gate never reads it. `src/gpu/spirv/check.rs`
imports only the standard library, so the real closure is `src/gpu/spirv/mod.rs` -> `src/gpu/spirv/check.rs`
with no crate mirror. The only new code is the typed forwarding wrapper, byte-identical in effect to
`proof/extract/lib.rs`'s `word_check_fragment`; it restates no formula.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
Lives on lane 3 (`/tmp/opencode/apeiron-lane-3.lock`), 540 s whole-invocation budget. Scratch copies
live off the tmpfs (AGENTS failure register: `/tmp` tmpfs headroom); only named receipts and logs
are kept under `/tmp/opencode`.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
LANE = '/tmp/opencode/apeiron-lane-3.lock'
BUDGET = 540
INNER = 525
if __name__ == '__main__' and os.environ.get('CHECK_FRAGMENT_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','CHECK_FRAGMENT_IN_LANE=1',
        'CHECK_FRAGMENT_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
OUT = Path('/tmp/opencode/checkfragment-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'checkfragment-resource'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(INNER - (time.monotonic()-START), float(os.environ.get('CHECK_FRAGMENT_DEADLINE', time.time()+INNER))-time.time()-3)
    if left <= 0:
        raise RuntimeError('CHECK_FRAGMENT_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles. `src/gpu/spirv/mod.rs` declares `pub mod check;`, so the
# `#[path]`-declared `spirv` module resolves `check` against the real `src/gpu/spirv/` directory and pulls
# the real `src/gpu/spirv/check.rs`; the checker body imports only the standard library, so no other
# mirror is needed. Only `mod` varies with the control copy; the wrapper is byte-identical to
# `proof/extract/lib.rs` lines 956-967.
LIB_TEMPLATE = '''#[path = {mod}]
pub mod spirv;

/// Typed forwarding wrapper: it fixes the slice and result element types and calls the real
/// `src/gpu/spirv/check.rs` `check_fragment` body, the same call `proof/extract/lib.rs`
/// `word_check_fragment` makes. It restates no formula.
pub fn word_check_fragment(words: &[u32]) -> Result<(), spirv::check::CheckError> {{
    spirv::check_fragment(words)
}}
'''

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="check_fragment_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(LIB_TEMPLATE.format(mod=json.dumps(str(source))))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'check_fragment_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_check_fragment','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'CheckFragmentSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text()
        .replace(str(source), 'src/gpu/spirv/mod.rs')
        .replace(str(ROOT/'src/gpu/spirv/check.rs'), 'src/gpu/spirv/check.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('CHECK_FRAGMENT_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage checkFragmentProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib CheckFragment where\n  roots := #[`CheckFragmentSource,`CheckFragmentResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='checkFragmentProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/gpu/spirv/mod.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/CheckFragmentSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('CHECK_FRAGMENT_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('CHECK_FRAGMENT_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/CheckFragmentResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('CHECK_FRAGMENT_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'CheckFragmentSource.lean');shutil.copyfile(proof,p/'CheckFragmentResource.lean')
    code,text=run(['lake','build','CheckFragmentResource'],p,'fresh-proof')
    if 'CHECK_FRAGMENT_KERNEL_OK' not in text:raise RuntimeError('CHECK_FRAGMENT_AUDIT_MISSING')
    if a.controls:
        # REAL-SOURCE mutation: the D1 zero-`WordCount` guard of the actual scan loop. The mutated
        # tree is a scratch mirror of `src/gpu/spirv/` (real `mod.rs`, mutated `check.rs`), so the
        # `#[path]` wiring and the module layout are the production ones.
        original=(ROOT/'src/gpu/spirv/check.rs').read_text();needle='if word_count == 0 {'
        if original.count(needle)!=1:raise RuntimeError('CHECK_FRAGMENT_CONTROL_TARGET_MISSING')
        mirror=SCRATCH/'spirv-mutant';mirror.mkdir()
        shutil.copyfile(ROOT/'src/gpu/spirv/mod.rs',mirror/'mod.rs')
        mutant=mirror/'check.rs';mutant.write_text(original.replace(needle,'if word_count == 1 {'))
        freshmut=extract(mirror/'mod.rs','mutant');shutil.copyfile(freshmut,p/'CheckFragmentSource.lean')
        code,text=run(['lake','build','CheckFragmentResource'],p,'mutant-proof',require=False)
        if code!=1 or 'CheckFragmentResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic','motive is not type correct']):
            raise RuntimeError('CHECK_FRAGMENT_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('CHECK_FRAGMENT_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mirror/'mod.rs','restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('CHECK_FRAGMENT_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'CheckFragmentSource.lean');run(['lake','build','CheckFragmentResource'],p,'restored-proof')
        print('CHECK_FRAGMENT_REAL_SOURCE_CONTROL_RESTORED')
    print('CHECK_FRAGMENT_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/gpu/spirv/mod.rs',ROOT/'src/gpu/spirv/check.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/CheckFragment_gate.py',ROOT/'proof/resource/CheckFragmentSource.lean',ROOT/'proof/resource/CheckFragmentResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=BUDGET,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('CHECK_FRAGMENT_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts scan iterations and instruction words under named premises and excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)