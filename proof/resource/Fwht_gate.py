#!/usr/bin/env python3
"""Extract and check the real integer Walsh-Hadamard rotation source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/engine/{value,hadamard}.rs` by `#[path]`, exactly the
mirror wiring `proof/extract/lib.rs` uses for `crate::word_fwht_fixed`; the only new code is the
typed forwarding wrapper that calls the real `fwht_fixed` body. No substitute algorithm is
extracted, and the `engine::FIXED_FRAC_BITS` mirror is checked against its real definition.

Lives on lane 1 (`/tmp/opencode/apeiron-lane-1.lock`), 540 s whole-invocation budget.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-1.lock'
BUDGET = 540
INNER = 525
if __name__ == '__main__' and os.environ.get('FWHT_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','FWHT_IN_LANE=1',
        'FWHT_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (AGENTS failure register: /tmp tmpfs headroom); only named
# receipts and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/fwht-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/fwht-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(INNER - (time.monotonic()-START), float(os.environ.get('FWHT_DEADLINE', time.time()+INNER))-time.time()-3)
    if left <= 0:
        raise RuntimeError('FWHT_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles: `hadamard.rs` imports `super::value::div_round_nearest`
# and `super::FIXED_FRAC_BITS`. Modules are declared at the crate root (a `#[path]` inside an inline
# module resolves against a virtual directory) with an `engine` mirror, exactly the wiring
# `proof/extract/lib.rs` uses at lines 48-51 and 91-112. The `FIXED_FRAC_BITS` value is read back
# from the real `src/engine/mod.rs` by `mirror_frac_bits` and checked, so the mirror is bound to the
# real constant rather than restated.
LIB_TEMPLATE = '''/// Mirror of `crate::engine` for this extraction unit: `hadamard.rs`'s `super::value` and
/// `super::FIXED_FRAC_BITS` resolve to these two items, as they do in `proof/extract/lib.rs`.
pub mod engine {{
    pub use super::{{hadamard, value}};
    /// Fractional bits of the Q32.32 executing representation; value read from `src/engine/mod.rs`.
    pub const FIXED_FRAC_BITS: u32 = {frac};
}}
#[path = {value}]
pub mod value;
#[path = {hadamard}]
pub mod hadamard;
pub use engine::FIXED_FRAC_BITS;

/// Typed forwarding wrapper: it fixes the slice and result types and calls the real
/// `src/engine/hadamard.rs` `fwht_fixed` body, the same call `proof/extract/lib.rs`
/// `word_fwht_fixed` makes. It restates no formula.
pub fn word_fwht_fixed(x: &mut [i64], signs: Option<&[i8]>, block: usize)
    -> Result<(), engine::hadamard::HadamardError> {{
    engine::hadamard::fwht_fixed(x, signs, block)
}}
'''

def mirror_frac_bits():
    """The real `FIXED_FRAC_BITS` literal, so the engine mirror cannot drift from `src/engine/mod.rs`."""
    text = (ROOT/'src/engine/mod.rs').read_text()
    m = re.search(r'pub const FIXED_FRAC_BITS: u32 = (\d+);', text)
    if not m:
        raise RuntimeError('FWHT_FRAC_BITS_DEFINITION_MISSING')
    return int(m.group(1))

def lib_source(hadamard):
    """Absolute `#[path]`s to the real modules; only `hadamard` varies with the control copy."""
    src = ROOT/'src/engine'
    return LIB_TEMPLATE.format(frac=mirror_frac_bits(),
        value=json.dumps(str(src/'value.rs')), hadamard=json.dumps(str(hadamard)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="fwht_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'fwht_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_fwht_fixed','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'FwhtSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text()
        .replace(str(source), 'src/engine/hadamard.rs')
        .replace(str(ROOT/'src/engine/value.rs'), 'src/engine/value.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('FWHT_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage fwhtProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Fwht where\n  roots := #[`FwhtSource,`FwhtResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='fwhtProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/engine/hadamard.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/FwhtSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('FWHT_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('FWHT_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/FwhtResource.lean'
    # A `sorry`, an `admit`, a declared `axiom` or a defeated heartbeat guard is a rejection. The
    # scan strips comments first, so the audit command's own text and the documentation that names
    # the declared `native_decide` width boundary do not trip it, while a real `axiom` line in the
    # proof does.
    def strip_comments(s):
        return re.sub(r'/-[^-]*(?:-(?!-)[^-]*)*-/', ' ', s, flags=re.S)
    body=strip_comments(proof.read_text())
    if re.search(r'\b(sorry|admit)\b|^\s*axiom\b|set_option maxHeartbeats 0', body, flags=re.M):
        raise RuntimeError('FWHT_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'FwhtSource.lean');shutil.copyfile(proof,p/'FwhtResource.lean')
    code,text=run(['lake','build','FwhtResource'],p,'fresh-proof')
    if 'FWHT_KERNEL_OK' not in text:raise RuntimeError('FWHT_AUDIT_MISSING')
    if a.controls:
        # The lane increment is the transition `lane_body_step` and `lane_visits_exact` state:
        # `lane'.val = lane.val + 1` and the rank `span - lane` dropping by one. Mutating it must
        # turn the proof red; the `span *= 2` line is not a witness of that, because the audited
        # theorems take the span as a parameter rather than deriving it from the source loop.
        original=source.read_text();needle='lane += 1;'
        if original.count(needle)!=1:raise RuntimeError('FWHT_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'hadamard.rs';mutant.write_text(original.replace(needle,'lane += 2;'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'FwhtSource.lean')
        code,text=run(['lake','build','FwhtResource'],p,'mutant-proof',require=False)
        if code!=1 or 'FwhtResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic','motive is not type correct']):
            raise RuntimeError('FWHT_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('FWHT_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Aeneas records the source path and line range of every item, and the disposable copy's
        # absolute path differs from the committed file's, so compare the executable text: drop
        # block comments, doc comments and the plain `--` line comments that carry those paths.
        # Aeneas also emits mutually independent declarations in an order that is not stable across
        # runs, so compare the multiset of declaration blocks rather than the file order; a change
        # to any body still appears as a difference in that multiset.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/|--[^\n]*', '',s,flags=re.S)
        blocks=lambda s: sorted(b.strip() for b in re.split(
            r'\n(?=@\[|def |abbrev |theorem |inductive |structure |/)', strip(s)) if b.strip())
        if blocks(restored.read_text())!=blocks(dest.read_text()):
            raise RuntimeError('FWHT_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'FwhtSource.lean');run(['lake','build','FwhtResource'],p,'restored-proof')
        print('FWHT_REAL_SOURCE_CONTROL_RESTORED')
    print('FWHT_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/hadamard.rs',ROOT/'src/engine/value.rs',ROOT/'src/engine/mod.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/Fwht_gate.py',ROOT/'proof/resource/FwhtSource.lean',ROOT/'proof/resource/FwhtResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=BUDGET,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('FWHT_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts loop iterations and span products under named premises and excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
